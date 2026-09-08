"""Capture control-plane state in a gzip tar under `<data_dir>/backups/`.

`nerdit-backup-<UTC stamp>-<hex>.tar.gz` contains a SQLite `VACUUM INTO`
snapshot without WAL sidecars, encrypted secrets and their master key, and
Caddy's `pki`, `certificates`, and `acme` trees. Preserving issued certificates
and the ACME account avoids reissuance and CA rate limits after restore.
Named-volume data, model weights, audit archives, uploads, and logs are excluded
from this control-plane archive.

The archive contains the secrets master key and unencrypted TLS private keys;
any holder can decrypt stored secrets. Copy it off-box, then delete it locally.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import json
import logging
import ntpath
import os
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from secrets import token_hex
from typing import TYPE_CHECKING, Any, Literal
from urllib.request import pathname2url

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .secrets import SecretRotationInProgress
from .volumes import dump_staging_root

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..db.database import Database
    from .secrets import SecretManager

logger = logging.getLogger(__name__)


class BackupError(RuntimeError):
    """Staging/packaging failure.

    The message is built from the exception class name + errno text only —
    never `str(exc)`, which embeds absolute paths for `OSError` (security
    M3). The full exception goes to the daemon log, never the HTTP body.
    """


@dataclass
class BackupResult:
    """Outcome of `create_backup`."""

    path: str
    basename: str
    size_bytes: int
    kid: str
    manifest: dict[str, Any]


def _sanitize(exc: BaseException) -> str:
    """Path-free message: exc class name + errno text only (security M3)."""
    errno = getattr(exc, "errno", None)
    detail = os.strerror(errno) if errno else "see daemon logs"
    return f"backup staging failed: {type(exc).__name__}: {detail}"


def _fsync_dir(path: Path) -> None:
    """fsync a directory fd so a rename is durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


async def snapshot_db(db: Database, dest: Path) -> None:
    """Write a single consistent DB snapshot to *dest* (no `-wal`/`-shm`).

    Held under `db.write_lock` on the event-loop side (never nested with the
    secrets rotation lock — D3). `VACUUM INTO` runs in autocommit (legacy
    isolation + no open writer transaction while the lock is held, D8) and
    checkpoints WAL into a single file. NEVER call any `@_serialized` method
    inside this block — `write_lock` is non-reentrant (H5).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)  # VACUUM INTO refuses an existing file
    async with db.write_lock:
        try:
            await db.conn.execute("VACUUM INTO ?", (str(dest),))
        except sqlite3.OperationalError:
            # A failed VACUUM can leave a partial file behind; never back up
            # onto it (L7). Fall back to the online-backup API under the same
            # lock.
            dest.unlink(missing_ok=True)
            async with aiosqlite.connect(str(dest)) as target:
                await db.conn.backup(target)


def _schema_info(snapshot_path: Path) -> dict[str, Any]:
    """Read `user_version` + a stable schema digest off the snapshot file.

    Runs stdlib `sqlite3` read-only against the offline snapshot (no lock,
    never the shared aiosqlite connection). `tables_sha256` = SHA-256 over
    `"\\n".join(f"{type}|{name}|{sql}")` of every non-internal table/index
    ordered by `(type, name)`.
    """
    # URL-escape the path: a data_dir containing ?, #, or % would otherwise
    # break the file: URI parse (finding L5).
    uri = f"file:{pathname2url(str(snapshot_path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('table','index') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
    finally:
        conn.close()
    blob = "\n".join(f"{row[0]}|{row[1]}|{row[2]}" for row in rows)
    return {
        "user_version": int(user_version),
        "tables_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
    }


def _copy_caddy_subtree(
    data_dir: Path,
    dest: Path,
    name: str,
    skipped: list[str],
    excluded: frozenset[Path] = frozenset(),
) -> bool:
    """Copy the `caddy/<name>/` subtree into *dest* (regular-only).

    *name* is one of `pki` (the internal CA), `certificates` (every leaf the
    proxy has issued — the INTERNAL ones under `local/` on any proxy-on node,
    public ACME ones under a key derived from the directory URL) or `acme`
    (the ACME account key + object). One helper for all three: they hold the
    same class of material — unencrypted PEM private keys — and so must share
    one capture discipline rather than drift apart.

    Manual walk with `followlinks=False` — never `copytree`, which follows
    symlinks and would ship link targets off-box (security M2). Only entries
    whose `lstat` is a real dir/regular file are copied; anything else (a
    symlink, device, FIFO, …) is reported by basename in *skipped* and never
    copied. Files whose resolved path is in *excluded* (the node-link private
    key, wherever `[link].key_file` points — P27 WP-C3 item 3) are likewise
    reported by basename and never copied: a backup tar that can impersonate a
    node changes the custody story. Returns `True` when the subtree existed,
    `False` when absent (proxy off, or ACME never enabled — tolerated; the
    manifest records which trees were captured).
    """
    src = data_dir / "caddy" / name
    # Reject a symlinked subtree root *before* walking: is_dir() follows the
    # link and os.walk's followlinks=False only guards symlinks found during
    # the walk, not the starting path — so a planted `caddy/acme -> /elsewhere`
    # would otherwise ship off-box files into the archive unreported. Report
    # it by basename and treat as absent (security M2 root case).
    if src.is_symlink():
        skipped.append(src.name)
        return False
    if not src.is_dir():
        return False
    for root, dirs, files in os.walk(src, followlinks=False):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        target_dir = dest / rel
        # Filter symlinked dirs out of the descent set and report them.
        # (`entry` here, not `name` — that is the subtree parameter.)
        real_dirs: list[str] = []
        for entry in list(dirs):
            dpath = root_path / entry
            if not dpath.is_symlink() and stat.S_ISDIR(dpath.lstat().st_mode):
                real_dirs.append(entry)
            else:
                skipped.append(entry)
        dirs[:] = real_dirs
        target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        for entry in files:
            fpath = root_path / entry
            if fpath.resolve() in excluded:
                skipped.append(entry)
                continue
            if not fpath.is_symlink() and stat.S_ISREG(fpath.lstat().st_mode):
                shutil.copy2(fpath, target_dir / entry)
                os.chmod(target_dir / entry, 0o600)
            else:
                skipped.append(entry)
    return True


def _pack_tar(staging: Path, tmp_tar: Path) -> None:
    """Pack *staging* into *tmp_tar* — fd-based so the fsync is real.

    Members carry relative, sorted arcnames; only regular files (`0o600`) and
    dirs (`0o700`) are ever added (no links, no specials — enforced by
    `lstat`). `O_CREAT|O_EXCL` opens at `0o600`, then flush + fsync before
    the caller's `os.replace`.
    """
    entries: list[Path] = []
    for root, dirs, files in os.walk(staging):
        root_path = Path(root)
        for name in dirs:
            entries.append(root_path / name)
        for name in files:
            entries.append(root_path / name)
    entries.sort(key=lambda p: p.relative_to(staging).as_posix())

    fd = os.open(tmp_tar, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as fobj:
        with tarfile.open(fileobj=fobj, mode="w:gz") as tar:
            for path in entries:
                arcname = path.relative_to(staging).as_posix()
                st = path.lstat()
                if stat.S_ISDIR(st.st_mode):
                    info = tarfile.TarInfo(name=arcname)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o700
                    info.mtime = int(st.st_mtime)
                    tar.addfile(info)
                elif stat.S_ISREG(st.st_mode):
                    info = tarfile.TarInfo(name=arcname)
                    info.type = tarfile.REGTYPE
                    info.mode = 0o600
                    info.size = st.st_size
                    info.mtime = int(st.st_mtime)
                    with open(path, "rb") as fh:
                        tar.addfile(info, fh)
                # Anything else is impossible (we built staging); skip silently.
        fobj.flush()
        os.fsync(fobj.fileno())


async def create_backup(
    *,
    db: Database,
    secret_manager: SecretManager,
    data_dir: Path,
    exclude_paths: tuple[Path, ...] = (),
) -> BackupResult:
    """Stage + pack a control-plane backup tar; return its metadata.

    `exclude_paths` are resolved file paths that must never enter the tar
    even when they sit inside a captured tree — the node-link private key
    (P27 WP-C3 item 3), threaded in by the route from `[link].key_file`.

    Sequenced never nested (D3): the secrets snapshot (rotation RLock, in a
    worker thread) then the DB snapshot (`write_lock` on the loop). Staging
    is always `rmtree`'d and any leftover `.tmp` tar unlinked. Every failure
    except `SecretRotationInProgress` (which propagates typed) is wrapped
    in `BackupError` with a path-free message (M3).
    """
    backups = data_dir / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    os.chmod(backups, 0o700)

    staging = backups / f".staging-{os.getpid()}-{token_hex(4)}"
    staging.mkdir(mode=0o700)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    basename = f"nerdit-backup-{stamp}-{token_hex(3)}.tar.gz"
    tmp_tar = backups / f".tmp-{basename}"

    try:
        # 1. secrets snapshot — RLock is taken inside the sync body (D3).
        kid, skipped = await asyncio.to_thread(secret_manager.snapshot_to, staging / "secrets")
        # 2. DB snapshot — write_lock on the event loop.
        await snapshot_db(db, staging / "db" / "nerdit.db")
        # 3. Caddy identity + public certificate material (each optional).
        #    pki = the internal CA; certificates = every issued leaf (internal
        #    ones under `local/`, public ones under the directory key, so this
        #    tree exists on ANY proxy-on node, ACME or not);
        #    acme = the ACME account key. Captured separately so the manifest
        #    can say which trees a given tar actually carries — restore swaps
        #    only what is present, so a pre-WP2 tar never wipes live ACME
        #    material.
        excluded = frozenset(p.resolve() for p in exclude_paths)
        caddy_present = await asyncio.to_thread(
            _copy_caddy_subtree, data_dir, staging / "caddy" / "pki", "pki", skipped, excluded
        )
        certs_present = await asyncio.to_thread(
            _copy_caddy_subtree,
            data_dir,
            staging / "caddy" / "certificates",
            "certificates",
            skipped,
            excluded,
        )
        acme_present = await asyncio.to_thread(
            _copy_caddy_subtree, data_dir, staging / "caddy" / "acme", "acme", skipped, excluded
        )
        # 4. schema digest off the offline snapshot.
        schema = await asyncio.to_thread(_schema_info, staging / "db" / "nerdit.db")

        store_dir = staging / "secrets" / "store"
        secrets_count = len(list(store_dir.glob("*.enc"))) if store_dir.is_dir() else 0

        manifest: dict[str, Any] = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kid": kid,
            "schema": schema,
            # P27 WP-C3 item 3 — the node-link private key is deliberately NOT
            # captured. The default location (<data_dir>/link/node.key) falls
            # outside every captured path, and an operator-chosen
            # `[link].key_file` planted INSIDE one (e.g. under caddy/pki/) is
            # excluded by `exclude_paths` — the route threads the resolved key
            # path in — because a backup tar that can impersonate a node changes
            # the custody story; document "re-link after restore" instead. The
            # final call lands at the WP-C2 review; if it overturns the default,
            # wire the capture and a contents.node_key manifest flag HERE.
            #
            # P17d D-LIC7 — the offline product license (<data_dir>/license.jws,
            # or an operator-chosen `[license].file`, likewise threaded in via
            # `exclude_paths`) is deliberately NOT captured either, for a
            # different reason worth stating precisely: node.key is excluded
            # because a tar that can impersonate a node changes the custody
            # story; license.jws is excluded because it is **re-issuable, not
            # unique** — the restore runbook says "reinstall the license" the way
            # it says "re-link" (docs/guide/backup-restore.md,
            # docs/guide/license.md) — and because keeping it out also keeps
            # `customer_id` out of a tar that already demands careful custody
            # for the master key. Zero code, zero manifest schema bump. Revisit
            # only if restore-support burden materialises.
            # `caddy_certificates`/`caddy_acme` are ADDITIVE:
            # no schema bump, because restore still requires only `db` +
            # `secrets_key` and reads the caddy flags as advisory. A v1 tar
            # written before WP2 simply lacks both keys.
            "contents": {
                "db": True,
                "secrets": secrets_count,
                "secrets_key": True,
                "caddy_pki": caddy_present,
                "caddy_certificates": certs_present,
                "caddy_acme": acme_present,
            },
            "skipped": skipped,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.chmod(manifest_path, 0o600)

        # 5. pack + durably promote.
        await asyncio.to_thread(_pack_tar, staging, tmp_tar)
        final = backups / basename
        os.replace(tmp_tar, final)
        _fsync_dir(backups)

        size_bytes = final.stat().st_size
        return BackupResult(
            path=str(final),
            basename=basename,
            size_bytes=size_bytes,
            kid=kid,
            manifest=manifest,
        )
    except SecretRotationInProgress:
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped path-free, full exc logged
        logger.exception("Backup staging failed")
        raise BackupError(_sanitize(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            tmp_tar.unlink(missing_ok=True)
        except OSError:
            pass


# --- Backup v2 — per-database volume tars ---------------------------
#
# A volume backup is a single gzip tar under `<data_dir>/backups/` named
# `nerdit-volumes-<service>-<UTC stamp>-<hex>.tar.gz`. The name sits OUTSIDE
# the v1 `nerdit-backup-*` glob so the P14b control-plane retention/disk/gc
# walkers never touch it (its own key + walker live in the daemon layer). It
# carries a best-effort, file-level copy of `<data_dir>/services/<service>/**`
# (all named-volume data) plus a trailing `volume-manifest.json` — deliberately
# NOT the secrets master key or any `.enc` ciphertext (those stay exclusive to
# the v1 control-plane tar). Because the files are read at slightly different
# instants (not an atomic filesystem snapshot), a live-tar of an actively-writing
# database is NOT guaranteed application- or crash-consistent; ``nerdit services
# stop`` before the capture gives a guaranteed-clean one.


@dataclass
class VolumeBackupResult:
    """Outcome of `create_volume_backup`."""

    path: str
    basename: str
    size_bytes: int
    service: str
    backend: str | None
    manifest: dict[str, Any]


def _volume_names(service_root: Path) -> list[str]:
    """Sorted immediate child directory names of *service_root* (the volume leaves).

    Reported in the manifest `contents.volumes` — regular directories only,
    symlinks/specials never counted (they are skipped by the capture too).
    """
    try:
        return sorted(e.name for e in os.scandir(service_root) if e.is_dir(follow_symlinks=False))
    except OSError:
        return []


def _collect_volume_entries(service_root: Path, skipped: list[str]) -> list[Path]:
    """lstat-allowlisted walk of *service_root*: real dirs + regular files only.

    Same discipline as `_copy_caddy_subtree` (security M2): a symlinked root is
    reported and treated as absent; during the walk symlinks/devices/FIFOs are
    filtered out of the descent set and reported by basename in *skipped*, never
    followed off the volume tree. Returns the entry paths (dirs + files) to pack.
    """
    entries: list[Path] = []
    # A symlinked service root must never be followed off-box (mirror of the pki
    # root guard): report it and capture nothing.
    if service_root.is_symlink():
        skipped.append(service_root.name)
        return entries
    if not service_root.is_dir():
        return entries

    def _walk_error(err: OSError) -> None:
        # os.walk() swallows scandir errors by default. Two cases diverge:
        #   * FileNotFoundError — a live database removed a dir/file (WAL/tmp
        #     recycling) between the parent scandir and our visit. Benign and
        #     expected on a best-effort live snapshot: skip + record it,
        #     never fail the capture (finding #8).
        #   * PermissionError — the daemon user cannot read this tree. Under the
        #     adopted daemon-uid posture the data is daemon-owned so
        #     this should not occur, but a data dir left by a pre-amendment
        #     feat/p15 build (owned by the in-image uid) still would. Left
        #     silent the walk yields an empty tar that falsely claims to hold
        #     "all database data" — so re-raise and fail LOUDLY rather than ship
        #     a lying backup (the ddeea7a contract, kept as defense in depth).
        if isinstance(err, FileNotFoundError):
            name = os.path.basename(err.filename) if err.filename else "?"
            skipped.append(name)
            return
        raise err

    for root, dirs, files in os.walk(service_root, followlinks=False, onerror=_walk_error):
        root_path = Path(root)
        real_dirs: list[str] = []
        for name in list(dirs):
            dpath = root_path / name
            try:
                is_link = dpath.is_symlink()
                mode = dpath.lstat().st_mode
            except FileNotFoundError:
                # Vanished between scandir and lstat — a live DB rewriting itself.
                skipped.append(name)
                continue
            if not is_link and stat.S_ISDIR(mode):
                real_dirs.append(name)
                entries.append(dpath)
            else:
                skipped.append(name)
        dirs[:] = real_dirs
        for name in files:
            fpath = root_path / name
            try:
                is_link = fpath.is_symlink()
                mode = fpath.lstat().st_mode
            except FileNotFoundError:
                skipped.append(name)
                continue
            if not is_link and stat.S_ISREG(mode):
                entries.append(fpath)
            else:
                skipped.append(name)
    return entries


#: Snapshot a regular file into memory up to this size, then spill to a temp file
#: (bounds peak RSS to one file's snapshot for a multi-GB Postgres data dir).
_SNAPSHOT_SPOOL_MAX = 16 * 1024 * 1024


def _snapshot_regular(path: Path, *, spool_dir: Path) -> tuple[Any, int] | None:
    """Copy a regular file into a bounded spooled tempfile; return `(spool, size)`.

    The snapshot is taken FIRST so the `TarInfo` size we then record is EXACTLY
    the number of bytes we hold. A live database truncating the file mid-capture
    (`VACUUM`, WAL recycling) between our `lstat` and the read would otherwise
    make the header claim more bytes than exist — `tarfile` raises "unexpected
    end of data" AFTER emitting the header + partial payload, misaligning the
    whole stream into a silently corrupt archive (finding #2). `None` when the
    file vanished before we could open it (a live DB removing WAL/tmp files) — the
    caller skips + manifest-reports it, never a partial member. A snapshot larger
    than the in-RAM bound spills into *spool_dir* (the `0o700` backups dir, the
    same filesystem the tar lands on) — never the system tempdir, which is often
    a size-limited tmpfs a multi-GB database segment would exhaust.
    """
    spool: Any = tempfile.SpooledTemporaryFile(max_size=_SNAPSHOT_SPOOL_MAX, dir=str(spool_dir))
    try:
        with open(path, "rb") as fh:
            shutil.copyfileobj(fh, spool)
    except FileNotFoundError:
        spool.close()
        return None
    except BaseException:
        spool.close()
        raise
    size = spool.tell()
    spool.seek(0)
    return spool, size


def _pack_volume_tar(
    service_root: Path,
    service: str,
    entries: list[Path],
    manifest: dict[str, Any],
    tmp_tar: Path,
) -> None:
    """Pack the service tree + manifest into *tmp_tar* — fd-based so the fsync is real.

    Members: the service tree under `services/<service>/**` (dirs `0o700`,
    files `0o600`, arcnames sorted) then `volume-manifest.json` (`0o600`)
    LAST — so a file that vanishes between capture and pack (a live database
    rewriting its files) is recorded in the manifest `skipped` list before it is
    serialized, never dropped silently (finding #2 / #8). Each regular file is
    snapshotted via `_snapshot_regular` so its recorded size matches its
    payload exactly (no misaligned partial member). Numeric `uid`/`gid` are
    recorded for forensics; they are NOT re-applied on restore — under the
    daemon-uid posture plain extraction as the daemon user yields the
    correct ownership (finding #3). `O_CREAT|O_EXCL` opens at `0o600`, then
    flush + fsync before the caller's `os.replace`.
    """
    prefix = PurePosixPath("services") / service
    ordered = sorted(entries, key=lambda p: p.relative_to(service_root).as_posix())
    now = int(datetime.now(timezone.utc).timestamp())

    fd = os.open(tmp_tar, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as fobj:
        with tarfile.open(fileobj=fobj, mode="w:gz") as tar:
            # The top-level service dir, so the tree always round-trips.
            top = tarfile.TarInfo(name=prefix.as_posix())
            top.type = tarfile.DIRTYPE
            top.mode = 0o700
            top.mtime = now
            try:
                root_st = service_root.lstat()
                top.uid = root_st.st_uid
                top.gid = root_st.st_gid
            except OSError:  # pragma: no cover — service_root validated by the caller
                pass
            tar.addfile(top)
            for path in ordered:
                arcname = (prefix / path.relative_to(service_root).as_posix()).as_posix()
                try:
                    st = path.lstat()
                except FileNotFoundError:
                    # Removed by a live database between capture and pack —
                    # best-effort live copy by design: skip + manifest-report it.
                    manifest["skipped"].append(path.name)
                    continue
                if stat.S_ISDIR(st.st_mode):
                    info = tarfile.TarInfo(name=arcname)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o700
                    info.uid = st.st_uid
                    info.gid = st.st_gid
                    info.mtime = int(st.st_mtime)
                    tar.addfile(info)
                elif stat.S_ISREG(st.st_mode):
                    snap = _snapshot_regular(path, spool_dir=tmp_tar.parent)
                    if snap is None:  # vanished between the lstat and the open
                        manifest["skipped"].append(path.name)
                        continue
                    spool, size = snap
                    try:
                        info = tarfile.TarInfo(name=arcname)
                        info.type = tarfile.REGTYPE
                        info.mode = 0o600
                        info.uid = st.st_uid
                        info.gid = st.st_gid
                        info.size = size
                        info.mtime = int(st.st_mtime)
                        tar.addfile(info, spool)
                    finally:
                        spool.close()
                # Anything else was filtered by the capture; skip silently.
            # Manifest LAST: its `skipped` list now reflects pack-time vanishes.
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            minfo = tarfile.TarInfo(name="volume-manifest.json")
            minfo.type = tarfile.REGTYPE
            minfo.mode = 0o600
            minfo.size = len(manifest_bytes)
            minfo.mtime = now
            tar.addfile(minfo, io.BytesIO(manifest_bytes))
        fobj.flush()
        os.fsync(fobj.fileno())


def create_volume_backup(
    *,
    data_dir: Path,
    service: str,
    backend: str | None,
) -> VolumeBackupResult:
    """Stage + pack a per-database volume tar; return its metadata.

    Captures `<data_dir>/services/<service>/**` under the P14c lstat allowlist
    (regular files + dirs only; symlinks/specials skipped + manifest-reported)
    and promotes it via tmp-then-`os.replace` (dir `0o700`, tar `0o600`).
    The tar holds all database data **and the SCRAM password verifiers** but
    NEVER the secrets master key or `.enc` ciphertexts. Every failure is
    wrapped in `BackupError` with a path-free message (M3); a leftover
    `.tmp` tar is always unlinked.
    """
    backups = data_dir / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    os.chmod(backups, 0o700)

    service_root = data_dir / "services" / service

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    basename = f"nerdit-volumes-{service}-{stamp}-{token_hex(3)}.tar.gz"
    tmp_tar = backups / f".tmp-{basename}"

    try:
        # #4 — never stage a success tar over a non-existent, symlinked, or empty
        # data tree. A row created but never launched has no data dir (or an empty
        # one); restoring such a "backup" later would rmtree + replace real data
        # with an empty tree. Fail loudly with an actionable, path-free message.
        if service_root.is_symlink():
            raise BackupError(
                "volume backup failed: the database data path is a symlink — refusing to capture it"
            )
        if not service_root.is_dir():
            raise BackupError(
                "volume backup failed: this database has no on-disk data yet — "
                "has it launched? (create and start it, then back it up)"
            )
        skipped: list[str] = []
        entries = _collect_volume_entries(service_root, skipped)
        if not any(p.is_file() for p in entries):
            raise BackupError(
                "volume backup failed: this database has no on-disk data yet — "
                "has it launched? (create and start it, then back it up)"
            )
        manifest: dict[str, Any] = {
            "version": 1,
            "service": service,
            "backend": backend,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contents": {"volumes": _volume_names(service_root)},
            "skipped": skipped,
        }
        _pack_volume_tar(service_root, service, entries, manifest, tmp_tar)
        final = backups / basename
        os.replace(tmp_tar, final)
        _fsync_dir(backups)

        size_bytes = final.stat().st_size
        return VolumeBackupResult(
            path=str(final),
            basename=basename,
            size_bytes=size_bytes,
            service=service,
            backend=backend,
            manifest=manifest,
        )
    except BackupError:
        # An actionable, already-path-free reject (#4) — never re-wrap it in the
        # generic _sanitize message below.
        raise
    except PermissionError as exc:
        # Non-root database data the daemon user cannot read. Fail
        # loudly with an actionable, path-free hint rather than a silent-empty
        # tar. A working capture of container-owned data needs the P14.5 run
        # primitive (container-side tar) — deferred by decision.
        logger.exception("Volume backup staging failed")
        raise BackupError(
            "backup staging failed: the daemon user cannot read this "
            "database's on-disk data (it is owned by the container's non-root "
            "user). Run nerditd as a user that can read <data_dir>/services/, "
            "or restore/back up the volume out-of-band."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - wrapped path-free, full exc logged
        logger.exception("Volume backup staging failed")
        raise BackupError(_sanitize(exc)) from exc
    finally:
        try:
            tmp_tar.unlink(missing_ok=True)
        except OSError:
            pass


# --- Hardened tar extraction (P37 D-P37-13) -----------------------------------
#
# Moved here from ``cli/commands/backup.py`` (pure motion) so the daemon's live
# restore and the CLI's offline restores share ONE extractor; ``core`` may never
# import from ``nerdit.cli``, so the CLI re-imports these names from here.


class RestoreError(RuntimeError):
    """A restore precondition/extraction/move-in failure (rendered, exit 1)."""


def _reject_unsafe(members: list[tarfile.TarInfo]) -> None:
    """Reject the WHOLE archive unless every member is a safe regular file/dir.

    Allowlist (strictly stronger than a denylist, security L6/D7): only
    ``isreg``/``isdir`` members survive; symlink/hardlink/device/FIFO/unknown
    typeflags are rejected. Names must not be absolute, drive-lettered, or
    contain ``..`` — our own tars only ever hold relative regular files + dirs,
    so anything else is an attack, not a compat case.
    """
    for m in members:
        if not (m.isreg() or m.isdir()):
            raise RestoreError(
                f"unsafe archive member (not a regular file or directory): {m.name!r}"
            )
        name = m.name
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts or ntpath.splitdrive(name)[0]:
            raise RestoreError(f"unsafe archive member path: {name!r}")


def _within(child: str, parent: str) -> bool:
    """True iff realpath(*child*) is *parent* or lives under it."""
    parent = os.path.realpath(parent)
    child = os.path.realpath(child)
    return child == parent or child.startswith(parent + os.sep)


def _extract_hardened(tar: tarfile.TarFile, staging: Path, *, force_manual: bool = False) -> None:
    """Extract *tar* into *staging* after the universal type pre-pass (D7).

    ``data`` filter when present (defense in depth on top of the pre-pass);
    otherwise a manual fallback (3.11.0-3.11.3) that checks realpath containment
    BEFORE opening and opens ``O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW`` — tar-recorded
    modes/owners are never applied.
    """
    try:
        members = tar.getmembers()
    except tarfile.TarError as exc:
        raise RestoreError(f"corrupt or unreadable archive: {type(exc).__name__}") from exc
    _reject_unsafe(members)

    if not force_manual and hasattr(tarfile, "data_filter"):
        tar.extractall(staging, filter="data")
        return

    for m in members:
        dest = staging / m.name
        if m.isdir():
            os.makedirs(dest, mode=0o700, exist_ok=True)
            continue
        os.makedirs(dest.parent, mode=0o700, exist_ok=True)
        # realpath re-check: catch a parent a previously extracted member could
        # have aliased before we open (L6 ordering).
        if not _within(str(dest.parent), str(staging)):
            raise RestoreError(f"archive member escapes staging: {m.name!r}")
        src = tar.extractfile(m)
        if src is None:  # pragma: no cover — isreg guaranteed by the pre-pass
            raise RestoreError(f"archive member is not extractable: {m.name!r}")
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as out, src:
            shutil.copyfileobj(src, out)


def _perm_pass(staging: Path) -> None:
    """Force every staged dir ``0o700`` and every file ``0o600`` before move-in."""
    for root, dirs, files in os.walk(staging):
        for d in dirs:
            os.chmod(Path(root) / d, 0o700)
        for f in files:
            os.chmod(Path(root) / f, 0o600)
    os.chmod(staging, 0o700)


# --- Backup v3 — per-database logical dumps (P37) -----------------------------
#
# ``<data_dir>/backups/nerdit-dump-<service>-<UTC stamp>-<hex>.tar.gz``: exactly
# two members, the engine tool's dump (``dump.pgdump`` / ``dump.rdb``) and a
# trailing ``dump-manifest.json``. The prefix keeps the glob disjoint from v1
# (``nerdit-backup-*``) and v2 (``nerdit-volumes-*``), so this flavour owns its
# own retention key, prune phase and ``/system/disk`` bucket. The payload was
# written by a CONTAINER (D-P37-7): ``lstat``-checked, opened ``O_NOFOLLOW``,
# size = bytes read, and any extra staging entry fails the capture loudly.

#: ``nerdit-dump-<service>-<UTC %Y%m%dT%H%M%SZ>-<6 hex>.tar.gz``; the greedy
#: service group is pinned by the anchored stamp (the ``_VOLUME_TAR_RE`` shape).
#: Lives here because this module mints the name; ``daemon/sweeps.py`` imports it.
DUMP_TAR_RE = re.compile(r"^nerdit-dump-(?P<service>.+)-\d{8}T\d{6}Z-[0-9a-f]{6}\.tar\.gz$")

#: The one glob every dump walker (retention prune, ``/system/disk``) uses.
DUMP_TAR_GLOB = "nerdit-dump-*.tar.gz"

#: The manifest member, always packed LAST (the ``_pack_volume_tar`` rule).
DUMP_MANIFEST_NAME = "dump-manifest.json"

#: Hard cap on the manifest member, checked against the tar HEADER before any
#: ``extractfile`` (D-P37-10): a hostile tar must never be able to make the
#: daemon read an unbounded "manifest" into memory.
DUMP_MANIFEST_MAX_BYTES = 64 * 1024

#: DNS-label grammar for the manifest ``service`` field — the same rule the
#: offline volume restore applies (D9). Provenance only for a dump (the restore
#: target is the route's path parameter, so a cross-name restore is allowed and
#: is how a clone is made), but a forged value must still never be a path.
_DUMP_SERVICE_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

#: The dump member's own name. A plain, dot-free-prefixed basename: it selects
#: BOTH the tar member and the file the restore writes into the staging dir, so
#: it must be structurally incapable of naming a path (no ``/``, no ``..``).
_DUMP_MEMBER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

#: ``engine`` / ``format`` are machine tokens (``postgres``/``redis``,
#: ``pg_custom``/``rdb`` today). Kept as a grammar rather than an enum so a
#: future backend does not make every older tar unreadable; the value that
#: MATTERS is checked against the row's own backend key by the restore route.
_DUMP_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: 64 lowercase hex — the sha256 of the dump member, verified before a restore
#: hands the file to ``pg_restore`` or installs it as a Redis AOF base.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Stream chunk for hashing/copying a dump (bounded RSS on a multi-GB dump).
_DUMP_CHUNK = 1024 * 1024


class DumpOutputError(RuntimeError):
    """The sibling's output failed its post-run checks (D-P37-7/11). ``reason`` ∈
    ``output_not_regular`` (missing, symlink, device, dir), ``empty_output``
    (``pg_dump`` creates its file before connecting) or ``unexpected_output``
    (extra entries in staging). Message is path-free (M3).
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or f"dump output rejected: {reason}")
        self.reason = reason


class DumpManifestError(RuntimeError):
    """A dump tar failed extraction or validation (D-P37-10). ``reason`` ∈
    ``members``, ``size`` (over :data:`DUMP_MANIFEST_MAX_BYTES`), ``schema``,
    ``sha256_mismatch``, ``unsafe_member``. The route projects it as ``422
    restore.manifest_invalid`` with ``detail.reason``; message path-free.
    """

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or f"dump manifest rejected: {reason}")
        self.reason = reason


class DumpTool(BaseModel):
    """Which binary produced the dump — ``argv0`` only, never the argv.

    The full command line is recorded in ``config['last_dump']`` and hashed into
    the audit row (D-P37-4/5); the manifest carries just the tool name, so a tar
    handed to a support engineer says what wrote it and nothing more.
    """

    model_config = ConfigDict(extra="forbid")

    argv0: str = Field(min_length=1, max_length=64)


class DumpManifest(BaseModel):
    """The ``dump-manifest.json`` member — manifest v1 (D-P37-7). ``extra="forbid"``
    and a grammar per free-form field: the file arrives from outside the daemon,
    and ``file`` selects both the tar member extracted and the staging name written.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    kind: Literal["dump"]
    service: str
    engine: str
    format: str
    image: str = Field(min_length=1, max_length=256)
    created_at: str = Field(min_length=1, max_length=64)
    file: str
    bytes: int = Field(ge=0)
    sha256: str
    tool: DumpTool

    @field_validator("service")
    @classmethod
    def _check_service(cls, value: str) -> str:
        if not _DUMP_SERVICE_RE.fullmatch(value):
            raise ValueError("service is not a valid DNS label")
        return value

    @field_validator("engine", "format")
    @classmethod
    def _check_token(cls, value: str) -> str:
        if not _DUMP_TOKEN_RE.fullmatch(value):
            raise ValueError("value is not a machine token")
        return value

    @field_validator("file")
    @classmethod
    def _check_file(cls, value: str) -> str:
        # Structurally incapable of naming a path — this value is joined onto
        # the staging dir at restore time.
        if not _DUMP_MEMBER_RE.fullmatch(value) or value == DUMP_MANIFEST_NAME:
            raise ValueError("file is not a plain dump member name")
        return value

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 is not 64 lowercase hex chars")
        return value


@dataclass
class DumpBackupResult:
    """Outcome of :func:`create_dump_backup`."""

    path: str
    basename: str
    #: Size of the packed tar (what lands on disk), NOT of the dump member.
    size_bytes: int
    #: sha256 of the dump member — the value the restore verifies.
    sha256: str
    service: str
    engine: str
    manifest: DumpManifest


def _sha256_nofollow(path: Path) -> tuple[str, int]:
    """Hash *path* through an ``O_NOFOLLOW`` fd; return ``(hexdigest, size)``.

    ``O_NOFOLLOW`` (not a plain ``open``) because the file was written by a
    container process that could have replaced it with a symlink between the
    ``lstat`` and this read (verified live, §0). The size returned is the number
    of bytes actually read, so the manifest can never claim more than we hashed.
    """
    digest = hashlib.sha256()
    size = 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as fh:
        while True:
            chunk = fh.read(_DUMP_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _rmtree_staging(staging: Path) -> None:
    """``rmtree`` *staging* only when it is a REAL directory (never a symlink).

    The staging dir is daemon-computed (``core.volumes.dump_staging_dir``) and
    created ``exist_ok=False``, so this guard should never fire; it exists
    because the alternative failure mode — following a planted symlink out of
    the staging root and deleting the target tree — is unrecoverable.
    """
    try:
        st = os.lstat(staging)
    except OSError:
        return
    if not stat.S_ISDIR(st.st_mode):
        logger.error("Refusing to remove a dump staging path that is not a directory")
        return
    shutil.rmtree(staging, ignore_errors=True)


def _verify_dump_output(staging: Path, dump_name: str) -> Path:
    """Validate the sibling's output and return its path (D-P37-7): plain member
    name; ``lstat`` is a REGULAR file (``output_not_regular``); non-empty
    (``empty_output``); nothing else in staging (``unexpected_output`` — staging is
    packed wholesale).
    """
    if not _DUMP_MEMBER_RE.fullmatch(dump_name) or dump_name == DUMP_MANIFEST_NAME:
        raise DumpOutputError(
            "unexpected_output", "the backend's dump filename is not a plain name"
        )
    output = staging / dump_name
    try:
        st = os.lstat(output)
    except OSError as exc:
        raise DumpOutputError("output_not_regular", "the dump tool wrote no output file") from exc
    if not stat.S_ISREG(st.st_mode):
        raise DumpOutputError(
            "output_not_regular", "the dump output is not a regular file (refusing to follow it)"
        )
    if st.st_size == 0:
        raise DumpOutputError("empty_output", "the dump tool wrote an empty output file")
    extras = sorted(entry.name for entry in os.scandir(staging) if entry.name != dump_name)
    if extras:
        raise DumpOutputError(
            "unexpected_output", f"the dump run left {len(extras)} unexpected file(s) behind"
        )
    return output


def _pack_dump_tar(staging: Path, manifest: DumpManifest, tmp_tar: Path) -> None:
    """Pack the dump member + manifest into *tmp_tar*, fd-based so the fsync is real.
    Manifest LAST (the ``_pack_volume_tar`` rule). The payload is opened
    ``O_NOFOLLOW``, its ``fstat`` size feeds the ``TarInfo`` and must equal the
    manifest's (a change between hash and pack would misalign the archive).
    ``O_CREAT|O_EXCL`` at ``0o600``, flush + fsync before the caller's ``os.replace``.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    dump_path = staging / manifest.file

    fd = os.open(tmp_tar, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as fobj:
        with tarfile.open(fileobj=fobj, mode="w:gz") as tar:
            src_fd = os.open(dump_path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(src_fd, "rb") as payload:
                st = os.fstat(payload.fileno())
                if not stat.S_ISREG(st.st_mode):  # pragma: no cover - lstat'd already
                    raise DumpOutputError("output_not_regular", "the dump output is not a file")
                if st.st_size != manifest.bytes:
                    raise DumpOutputError(
                        "unexpected_output", "the dump output changed size after it was hashed"
                    )
                info = tarfile.TarInfo(name=manifest.file)
                info.type = tarfile.REGTYPE
                info.mode = 0o600
                info.size = st.st_size
                info.mtime = int(st.st_mtime)
                tar.addfile(info, payload)
            # Manifest LAST (the v2 rule): a reader that got this far holds the
            # whole payload, so the manifest it then reads describes bytes it
            # already has.
            manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
            minfo = tarfile.TarInfo(name=DUMP_MANIFEST_NAME)
            minfo.type = tarfile.REGTYPE
            minfo.mode = 0o600
            minfo.size = len(manifest_bytes)
            minfo.mtime = now
            tar.addfile(minfo, io.BytesIO(manifest_bytes))
        fobj.flush()
        os.fsync(fobj.fileno())


def create_dump_backup(  # noqa: PLR0913 - the manifest's own fields, all keyword-only
    data_dir: Path,
    *,
    service: str,
    engine: str,
    image: str,
    staging: Path,
    backend_dump_filename: str,
    backend_dump_format: str,
    tool_argv0: str,
) -> DumpBackupResult:
    """Verify a sibling's dump output, pack it with its manifest, return metadata.

    Sync (run under ``asyncio.to_thread``). *staging* is ALWAYS removed before
    return, on every path (D-P37-2). The backend's three facts arrive as scalars so
    this module imports nothing from the data plane. Raises
    :class:`DumpOutputError` (machine ``reason``) or :class:`BackupError`
    (path-free, M3).
    """
    backups = data_dir / "backups"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    basename = f"nerdit-dump-{service}-{stamp}-{token_hex(3)}.tar.gz"
    tmp_tar = backups / f".tmp-{basename}"
    # Set the instant the rename lands and cleared at the commit point, so a
    # failure between the two (the directory fsync, the size stat) cannot leave
    # a complete, listable, restorable tar behind a 500 that told the caller
    # nothing was produced (D-P37-11: failure is loud and leaves nothing).
    promoted: Path | None = None

    try:
        # Inside the boundary: an unusable backups dir is daemon I/O, so it must
        # reach ``_pack_or_fail`` as a ``BackupError``, not a raw OSError
        # (D-P37-11).
        backups.mkdir(parents=True, exist_ok=True)
        os.chmod(backups, 0o700)
        # Defence in depth (D-P37-7): a symlinked staging dir would pack the link's
        # target and then, correctly, refuse to remove it.
        try:
            st_staging = os.lstat(staging)
        except OSError as exc:
            raise DumpOutputError(
                "output_not_regular", "the dump staging path is not a directory"
            ) from exc
        if not stat.S_ISDIR(st_staging.st_mode):
            raise DumpOutputError("output_not_regular", "the dump staging path is not a directory")
        output = _verify_dump_output(staging, backend_dump_filename)
        sha256, size = _sha256_nofollow(output)
        manifest = DumpManifest(
            version=1,
            kind="dump",
            service=service,
            engine=engine,
            format=backend_dump_format,
            image=image,
            created_at=datetime.now(timezone.utc).isoformat(),
            file=backend_dump_filename,
            bytes=size,
            sha256=sha256,
            tool=DumpTool(argv0=tool_argv0),
        )
        _pack_dump_tar(staging, manifest, tmp_tar)
        final = backups / basename
        os.replace(tmp_tar, final)
        promoted = final
        _fsync_dir(backups)
        size_bytes = final.stat().st_size

        result = DumpBackupResult(
            path=str(final),
            basename=basename,
            size_bytes=size_bytes,
            sha256=sha256,
            service=service,
            engine=engine,
            manifest=manifest,
        )
        # The commit point: past here the caller is told the tar exists, so the
        # ``finally`` must stop treating it as a leftover.
        promoted = None
        return result
    except (DumpOutputError, BackupError):
        # Already a typed, path-free refusal — never re-wrap it in the generic
        # _sanitize message below (the #4 rule from the volume packer).
        raise
    except Exception as exc:  # noqa: BLE001 - wrapped path-free, full exc logged
        logger.exception("Dump packing failed")
        raise BackupError(_sanitize(exc)) from exc
    finally:
        _rmtree_staging(staging)
        try:
            tmp_tar.unlink(missing_ok=True)
        except OSError:
            pass
        if promoted is not None:
            # The rename landed but the call is failing anyway (a directory
            # fsync or a stat that faulted): the request answers 500
            # ``dump.failed`` with ``no_outcome``, so the tar must not survive
            # to be listed by ``GET /databases/{name}/dumps`` and restored.
            with contextlib.suppress(OSError):
                promoted.unlink(missing_ok=True)


def extract_dump_tar(tar_path: Path, staging: Path) -> DumpManifest:
    """Extract a dump tar into *staging* and return its validated manifest (D-P37-10).

    Sync; the archive is untrusted and every step refuses rather than repairs:
    (1) :func:`_reject_unsafe` allowlist; (2) the member set is EXACTLY
    ``{<manifest.file>, dump-manifest.json}``; (3) the manifest HEADER size is
    checked against :data:`DUMP_MANIFEST_MAX_BYTES` before ``extractfile``; (4)
    :class:`DumpManifest` (``extra="forbid"``); (5) the payload is written
    ``O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW`` at ``0o600`` while hashed, and the
    digest must match. Any failure removes the partial payload and raises
    :class:`DumpManifestError` with a machine ``reason`` — a missing *tar_path*
    or *staging* included, so no raw path ever escapes.
    """
    try:
        with tarfile.open(tar_path, mode="r:gz") as tar:
            members = tar.getmembers()
            try:
                _reject_unsafe(members)
            except RestoreError as exc:
                raise DumpManifestError("unsafe_member", str(exc)) from exc

            names = [m.name for m in members]
            # ``count() == 1`` makes the two ``next()`` calls below total: a duplicated
            # manifest name would otherwise leak a bare ``StopIteration`` (D-P37-10).
            if len(members) != 2 or names.count(DUMP_MANIFEST_NAME) != 1:
                raise DumpManifestError(
                    "members", "a dump tar holds exactly the dump file and its manifest"
                )
            manifest_member = next(m for m in members if m.name == DUMP_MANIFEST_NAME)
            dump_member = next(m for m in members if m.name != DUMP_MANIFEST_NAME)
            if not manifest_member.isreg() or not dump_member.isreg():
                raise DumpManifestError("members", "a dump tar member is not a regular file")
            if manifest_member.size > DUMP_MANIFEST_MAX_BYTES:
                raise DumpManifestError("size", "the manifest member is too large")

            manifest = _parse_dump_manifest(tar, manifest_member)
            if dump_member.name != manifest.file:
                raise DumpManifestError("members", "the manifest does not name the packed member")
            _extract_dump_member(tar, dump_member, staging / manifest.file, manifest)
            return manifest
    except DumpManifestError:
        raise
    except Exception as exc:  # noqa: BLE001 - untrusted archive, mapped path-free
        # Deliberately total: a damaged archive raises ``EOFError`` / ``zlib.error``
        # / ``OSError`` as readily as ``TarError``, and every one must reach the
        # route as the contracted 422. The full exception goes to the log (M3).
        logger.exception("Dump tar extraction failed")
        raise DumpManifestError("members", "corrupt or unreadable archive") from exc


def _parse_dump_manifest(tar: tarfile.TarFile, member: tarfile.TarInfo) -> DumpManifest:
    """Read + validate the manifest member (header size already bounded)."""
    src = tar.extractfile(member)
    if src is None:  # pragma: no cover - isreg checked by the caller
        raise DumpManifestError("members", "the manifest member is not extractable")
    with src:
        # One byte over the cap: a tar header can lie about a member's size, so
        # the read is bounded independently of the header check above.
        raw = src.read(DUMP_MANIFEST_MAX_BYTES + 1)
    if len(raw) > DUMP_MANIFEST_MAX_BYTES:
        raise DumpManifestError("size", "the manifest member is too large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DumpManifestError("schema", "the manifest is not valid JSON") from exc
    try:
        return DumpManifest.model_validate(payload)
    except ValidationError as exc:
        # Never echo the pydantic error: it embeds the submitted values.
        raise DumpManifestError("schema", "the manifest does not match the v1 schema") from exc


def _extract_dump_member(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    dest: Path,
    manifest: DumpManifest,
) -> None:
    """Write the payload to *dest* (``O_EXCL|O_NOFOLLOW``) and verify its sha256. The
    tar HEADER size is compared to the manifest's ``bytes`` before the first write
    (the restore route has no disk preflight, §1.4), with the same ``reason`` as
    the post-write count.
    """
    if member.size != manifest.bytes:
        raise DumpManifestError(
            "sha256_mismatch", "the packed dump does not match the manifest digest"
        )
    src = tar.extractfile(member)
    if src is None:  # pragma: no cover - isreg checked by the caller
        raise DumpManifestError("members", "the dump member is not extractable")
    digest = hashlib.sha256()
    written = 0
    fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as out, src:
            while True:
                chunk = src.read(_DUMP_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                written += len(chunk)
                out.write(chunk)
    except BaseException:
        with contextlib.suppress(OSError):
            dest.unlink()
        raise
    if written != manifest.bytes or digest.hexdigest() != manifest.sha256:
        with contextlib.suppress(OSError):
            dest.unlink()
        raise DumpManifestError(
            "sha256_mismatch", "the packed dump does not match the manifest digest"
        )


def sweep_staging_orphans(data_dir: Path) -> int:
    """Remove every leftover staging entry; return how many were removed (D-P37-2).

    Two roots: ``<data_dir>/dump-staging/`` slots and ``<data_dir>/backups/``
    ``.staging-*`` dirs plus ``.tmp-nerdit-*`` partial tars (the full prefix all
    three packers mint). At boot no operation is in flight, so everything there
    is an orphan. A symlinked dump-staging root is refused, not swept (``scandir``
    would follow it; ``dump_staging_dir`` refuses it too). Removals are
    ``lstat``-guarded and never follow a symlink. Called from the daemon lifespan.
    """
    removed = 0
    # (root, name prefixes or None for every entry, refuse a symlinked root).
    # P14c has always followed a symlinked ``backups/``, so that root is swept
    # by name prefix instead of refused.
    roots: list[tuple[Path, tuple[str, ...] | None, bool]] = [
        (dump_staging_root(data_dir), None, True),
        (data_dir / "backups", (".staging-", ".tmp-nerdit-"), False),
    ]
    for root, prefixes, refuse_symlink_root in roots:
        if refuse_symlink_root and root.is_symlink():
            logger.warning("Refusing to sweep a symlinked dump-staging root")
            continue
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if prefixes is not None and not entry.name.startswith(prefixes):
                continue
            path = Path(entry.path)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            try:
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(path)
                else:
                    # A symlink or stray file: unlink it (never follow it).
                    os.unlink(path)
            except OSError:
                logger.warning("Could not remove a leftover staging entry at boot")
                continue
            removed += 1
    if removed:
        logger.info("Removed %d leftover staging entr(ies) at boot", removed)
    return removed
