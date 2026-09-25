"""Control-plane and volume backups, with offline restore.

Restore holds an exclusive `.restore.lock`, validates archive types and
manifests, and uses hardened extraction. Repeating the same tar recovers from
an interrupted restore. Caddy trees absent from an older archive are preserved.
Escape all server-derived Rich text.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import tarfile
from pathlib import Path
from uuid import uuid4

import httpx
import typer
from rich.markup import escape

from nerdit.cli.display import _plain, console, fmt_bytes, render_client_error
from nerdit.config.settings import load_settings

# The hardened tar extractor lives in ``core.backup`` (P37 D-P37-13): the
# daemon's live dump-restore needs the same allowlist pre-pass + O_NOFOLLOW
# fallback this offline command has always used, and ``core``/``daemon`` may
# never import from ``nerdit.cli``. Pure motion — same bodies, same names, and
# they stay importable from here because the restore suites reach for them by
# this module path.
from nerdit.core.backup import (  # noqa: F401 - re-exported for the restore suites
    RestoreError,
    _extract_hardened,
    _perm_pass,
    _reject_unsafe,
    _within,
)
from nerdit.daemon.lifecycle import DaemonLifecycle
from nerdit.utils.names import DNS_LABEL_RE

#: Test seam — force the manual (non-``data_filter``) extraction fallback so it
#: is exercised on interpreters that ship ``tarfile.data_filter`` (D7).
_FORCE_MANUAL = False


# --------------------------------------------------------------------------- #
# nerdit backup
# --------------------------------------------------------------------------- #


def backup(
    volume: str | None = typer.Option(
        None,
        "--volume",
        help="Capture a managed database's data volumes (backup v2) instead of control-plane.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the custody confirmation."),
) -> None:
    """Stage a backup tar.

    Without `--volume`: a control-plane tar (DB + secrets + the Caddy CA and
    ACME certificate material) — it contains the secrets master key and
    unencrypted PEM private keys, so copy it off-box and delete the local file.
    With `--volume <db>`: a per-database volume tar (all data + the SCRAM
    password verifiers, but NEVER the master key). Exit code 1 when the daemon is
    unreachable, else 0.
    """
    import asyncio

    if volume:
        code = asyncio.run(_volume_backup_async(service=volume, yes=yes))
    else:
        code = asyncio.run(_backup_async(yes=yes))
    if code:
        raise typer.Exit(code)


async def _backup_async(*, yes: bool) -> int:
    if not yes and not typer.confirm(
        "This backup will contain the secrets master key. Continue?", default=False
    ):
        console.print("[dim]Aborted.[/dim]")
        return 0

    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.create_backup(idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    console.print(f"[green]Backup written:[/green] {_plain(result.get('path'))}")
    console.print(f"  size: {fmt_bytes(result.get('size_bytes'))}")
    console.print(f"  kid:  {_plain(result.get('kid'))}")
    console.print(
        "\n[yellow]This archive contains the secrets master key (secrets.key). "
        "Copy it off-box and delete the local file; anyone holding it can read "
        "every stored secret.[/yellow]"
    )
    # M4 honesty line — printed unconditionally: the sweep is a no-op at the
    # default keep==0, so key-bearing tars accumulate forever until configured.
    console.print(
        "[dim]"
        + escape(
            "By default no backups are ever deleted — set "
            "[retention].backup_keep_last to bound key-bearing archives."
        )
        + "[/dim]"
    )
    return 0


async def _volume_backup_async(*, service: str, yes: bool) -> int:
    if not yes and not typer.confirm(
        f"Back up the data volumes of '{service}'? The archive contains the "
        "database data and password verifiers.",
        default=False,
    ):
        console.print("[dim]Aborted.[/dim]")
        return 0

    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.create_volume_backup(service, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    console.print(f"[green]Volume backup written:[/green] {_plain(result.get('path'))}")
    console.print(f"  size: {fmt_bytes(result.get('size_bytes'))}")
    console.print(
        "\n[yellow]This archive contains all database data and the SCRAM password "
        "verifiers, but NOT the secrets master key. Store it securely.[/yellow]"
    )
    # M4 honesty line — the volume sweep is a no-op at the default keep==0.
    console.print(
        "[dim]"
        + escape(
            "By default no volume backups are ever deleted — set "
            "[retention].volume_backup_keep_last to bound them."
        )
        + "[/dim]"
    )
    # (P37, D-P37-7) The cross-reference to the third tar flavour. This one is
    # PHYSICAL — a crash-consistent copy of the data directory, restorable only
    # onto a stopped daemon of the same engine layout. An operator who wanted a
    # portable, application-consistent capture wanted `nerdit db dump`, and the
    # only place they will find that out in time is here.
    console.print(
        f"[dim]For a logical, application-consistent capture of '{_plain(service)}' "
        f"(pg_dump / redis-cli --rdb) use `nerdit db dump {_plain(service)}` instead.[/dim]"
    )
    return 0


# --------------------------------------------------------------------------- #
# nerdit restore (offline)
# --------------------------------------------------------------------------- #


def restore(
    tar_path: Path = typer.Argument(..., help="Path to a nerdit-*.tar.gz archive."),
    volume: bool = typer.Option(
        False,
        "--volume",
        help="Restore a per-database volume tar (nerdit-volumes-*) instead of control-plane state.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the overwrite confirmation."),
) -> None:
    """Restore state from a backup tar — **daemon must be stopped**.

    Without `--volume`: overwrites the local DB, secrets key, `.enc`
    ciphertexts and whichever Caddy trees (`pki`/`certificates`/`acme`)
    the archive carries. With `--volume`: overwrites the manifest-named
    database's data dir (`<data_dir>/services/<service>`) — the target is bound
    to the tar's validated manifest `service` field, never its filename (D9).
    Both are rerunnable from the same tar after any crash. Exit code 1 on any
    refusal/failure, 0 on success (or a declined confirmation).
    """
    if volume:
        _run_volume_restore(tar_path, yes=yes, force_manual=_FORCE_MANUAL)
    else:
        _run_restore(tar_path, yes=yes, force_manual=_FORCE_MANUAL)


def _run_restore(tar_path: Path, *, yes: bool, force_manual: bool = False) -> None:
    tar_path = Path(tar_path).expanduser()
    if not tar_path.is_file():
        console.print(f"[red]No such backup archive:[/red] {_plain(tar_path)}")
        raise typer.Exit(1)

    settings = load_settings()
    data_dir = Path(settings.data_dir).expanduser()

    # Step 2 — stopped-daemon check, BOTH legs (contract F1). These are the
    # friendly-message layer; the authoritative exclusion is the flock (step 3).
    lifecycle = DaemonLifecycle(
        host=settings.daemon.host,
        port=settings.daemon.port,
        pid_file=settings.daemon.pid_file,
    )
    if lifecycle.is_running():
        console.print("[red]The daemon is running. Stop it first (`nerdit exit`).[/red]")
        raise typer.Exit(1)
    if _health_probe(settings.daemon.port):
        console.print(
            "[red]A daemon is answering on the configured port. "
            "Stop it first (`nerdit exit`).[/red]"
        )
        raise typer.Exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 3 — hold the exclusive .restore.lock for the whole restore. A live
    # daemon holds LOCK_SH, so this fails even when the pid/health heuristics
    # miss (D6/H1). Crash-safe: flock dies with the process.
    lock_fd = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        console.print(
            "[red]A daemon is running or another restore is in progress on this data dir.[/red]"
        )
        raise typer.Exit(1) from exc

    staging = data_dir / f".restore-staging-{os.getpid()}"
    try:
        # Under the held exclusive flock, sweep every prior crashed restore's
        # staging dir — not just this pid's. A crashed run under a different pid
        # leaves an extracted master-key copy behind forever otherwise (L6).
        for stale in data_dir.glob(".restore-staging-*"):
            shutil.rmtree(stale, ignore_errors=True)
        staging.mkdir(mode=0o700)
        try:
            # Step 4 — hardened extraction.
            try:
                with tarfile.open(tar_path, mode="r:*") as tar:
                    _extract_hardened(tar, staging, force_manual=force_manual)
            except tarfile.TarError as exc:
                raise RestoreError(f"corrupt or unreadable archive: {type(exc).__name__}") from exc

            # Step 5 — manifest validation + confirmation.
            manifest = _load_manifest(staging)
            schema = manifest.get("schema") or {}
            console.print("[bold]Restore plan:[/bold]")
            console.print(f"  kid:          {_plain(manifest.get('kid'))}")
            console.print(f"  created_at:   {_plain(manifest.get('created_at'))}")
            console.print(f"  user_version: {_plain(schema.get('user_version'))}")
            # Which Caddy trees this archive carries — the operator needs to
            # see that an older tar has no certificates/acme, because those
            # live trees will be left exactly as they are (see _move_into_place).
            console.print(f"  caddy:        {_plain(_caddy_trees(manifest))}")
            console.print(
                "[yellow]This will overwrite the DB, secrets key, secret "
                "ciphertexts and the Caddy trees listed above under "
                f"{_plain(data_dir)}.[/yellow]"
            )
            if not yes and not typer.confirm("Proceed with the restore?", default=False):
                console.print("[dim]Aborted.[/dim]")
                return

            # Step 6 — permission pass on staging.
            _perm_pass(staging)

            # Step 7 — move into place (rerunnable from the same tar, L8).
            _move_into_place(staging, data_dir, settings)
        except RestoreError as exc:
            console.print(f"[red]Restore failed:[/red] {_plain(exc)}")
            raise typer.Exit(1) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        os.close(lock_fd)

    # Step 8 — summary.
    console.print("[green]Restore complete.[/green]")
    console.print(f"  kid: {_plain(manifest.get('kid'))}")
    console.print("[dim]Start the daemon (`nerdit init` or your service manager) to resume.[/dim]")


def _run_volume_restore(tar_path: Path, *, yes: bool, force_manual: bool = False) -> None:
    """Restore the manifest-named database volume while holding the offline lock.

    The validated DNS-label service field selects the target, never the filename.
    Repeating the same tar is safe; the next launch restores sticky volume modes.
    """
    tar_path = Path(tar_path).expanduser()
    if not tar_path.is_file():
        console.print(f"[red]No such backup archive:[/red] {_plain(tar_path)}")
        raise typer.Exit(1)

    settings = load_settings()
    data_dir = Path(settings.data_dir).expanduser()

    # Stopped-daemon check, BOTH legs (the flock in step 3 is authoritative).
    lifecycle = DaemonLifecycle(
        host=settings.daemon.host,
        port=settings.daemon.port,
        pid_file=settings.daemon.pid_file,
    )
    if lifecycle.is_running():
        console.print("[red]The daemon is running. Stop it first (`nerdit exit`).[/red]")
        raise typer.Exit(1)
    if _health_probe(settings.daemon.port):
        console.print(
            "[red]A daemon is answering on the configured port. "
            "Stop it first (`nerdit exit`).[/red]"
        )
        raise typer.Exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        console.print(
            "[red]A daemon is running or another restore is in progress on this data dir.[/red]"
        )
        raise typer.Exit(1) from exc

    staging = data_dir / f".restore-staging-{os.getpid()}"
    try:
        for stale in data_dir.glob(".restore-staging-*"):
            shutil.rmtree(stale, ignore_errors=True)
        staging.mkdir(mode=0o700)
        try:
            try:
                with tarfile.open(tar_path, mode="r:*") as tar:
                    _extract_hardened(tar, staging, force_manual=force_manual)
            except tarfile.TarError as exc:
                raise RestoreError(f"corrupt or unreadable archive: {type(exc).__name__}") from exc

            manifest = _load_volume_manifest(staging)
            # D9: the target is bound to the validated manifest service, never the
            # tar filename. Our own tars store the tree under services/<service>/.
            service = manifest["service"]
            src_tree = staging / "services" / service
            if not src_tree.is_dir():
                raise RestoreError(f"archive is missing services/{service}")
            target = data_dir / "services" / service

            volumes = (manifest.get("contents") or {}).get("volumes") or []
            console.print("[bold]Volume restore plan:[/bold]")
            console.print(f"  service:    {_plain(service)}")
            console.print(f"  backend:    {_plain(manifest.get('backend'))}")
            console.print(f"  created_at: {_plain(manifest.get('created_at'))}")
            console.print(f"  volumes:    {_plain(', '.join(volumes) or '(none)')}")
            console.print(f"[yellow]This will overwrite {_plain(target)}.[/yellow]")
            if not yes and not typer.confirm("Proceed with the volume restore?", default=False):
                console.print("[dim]Aborted.[/dim]")
                return

            _perm_pass(staging)

            # Move into place (rerunnable from the same tar): the per-service root
            # and its parent are daemon-owned (0o700). The database's next launch
            # re-applies the sticky 0o1777 leaf perms (P14a _ensure_volume_dirs).
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _replace_tree(src_tree, target)
        except RestoreError as exc:
            console.print(f"[red]Restore failed:[/red] {_plain(exc)}")
            raise typer.Exit(1) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    finally:
        os.close(lock_fd)

    console.print("[green]Volume restore complete.[/green]")
    console.print(f"  service: {_plain(service)}")
    console.print(
        "[dim]Start the daemon to resume; the database re-applies volume perms at launch.[/dim]"
    )


def _health_probe(port: int) -> bool:
    """Return True for any HTTP health response; only transport failure means offline."""
    try:
        httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
    except httpx.HTTPError:
        return False
    return True


def _load_manifest(staging: Path) -> dict:
    """Read + validate `manifest.json` (version 1, db + secrets_key present)."""
    mpath = staging / "manifest.json"
    if not mpath.is_file():
        raise RestoreError("archive has no manifest.json")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestoreError("manifest.json is unreadable") from exc
    # Compute the version defensively: a non-dict top level (e.g. a JSON array)
    # must render the clean RestoreError, not ``AttributeError`` on ``.get``.
    version = manifest.get("version") if isinstance(manifest, dict) else None
    if version != 1:
        raise RestoreError(f"unsupported backup version: {version!r}")
    contents = manifest.get("contents") or {}
    if not contents.get("db") or not contents.get("secrets_key"):
        raise RestoreError("manifest does not declare db + secrets_key contents")
    return manifest


#: manifest ``contents`` flag → the ``caddy/<name>/`` subtree it stands for.
#: ``caddy_certificates``/``caddy_acme`` are additive: a tar written
#: before ACME certificate support simply has neither key, which reads as
#: "not captured".
_CADDY_TREE_FLAGS: tuple[tuple[str, str], ...] = (
    ("caddy_pki", "pki"),
    ("caddy_certificates", "certificates"),
    ("caddy_acme", "acme"),
)


def _caddy_trees(manifest: dict) -> str:
    """Render the Caddy subtrees a manifest declares, or `none`."""
    contents = manifest.get("contents") or {}
    present = [name for flag, name in _CADDY_TREE_FLAGS if contents.get(flag)]
    return ", ".join(present) if present else "none"


def _load_volume_manifest(staging: Path) -> dict:
    """Validate a version-1 volume manifest and its DNS-label service name.

    The service selects the restore target; the archive filename has no authority.
    """
    mpath = staging / "volume-manifest.json"
    if not mpath.is_file():
        raise RestoreError("archive has no volume-manifest.json")
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RestoreError("volume-manifest.json is unreadable") from exc
    # Compute the version defensively: a non-dict top level (e.g. a JSON array)
    # must render the clean RestoreError, not ``AttributeError`` on ``.get``.
    version = manifest.get("version") if isinstance(manifest, dict) else None
    if version != 1:
        raise RestoreError(f"unsupported volume backup version: {version!r}")
    # The restore target is bound to this validated field, never the tar filename (D9).
    service = manifest.get("service")
    if not isinstance(service, str) or not DNS_LABEL_RE.fullmatch(service):
        raise RestoreError(f"manifest service is not a valid DNS label: {service!r}")
    return manifest


def _replace_tree(src: Path, target: Path) -> None:
    """Replace a target tree, reporting removal or rename failures without paths.

    Do not ignore removal errors: leftover files can prevent replacement.
    The operation is rerunnable.

    Raises:
        RestoreError: The old tree cannot be removed or the new tree moved in.
    """
    if target.is_symlink() or target.is_file():
        try:
            target.unlink()
        except OSError as exc:
            raise RestoreError(
                "could not remove the existing database data path before "
                "restoring. Remove it manually as its owner (or root), then retry."
            ) from exc
    elif target.exists():
        try:
            shutil.rmtree(target)
        except OSError as exc:
            raise RestoreError(
                "could not clear the existing database data dir — it holds files "
                "owned by another user (a database container from an older build). "
                "Remove <data_dir>/services/<service> manually as its owner (or "
                "root), then retry."
            ) from exc
    try:
        os.replace(src, target)
    except OSError as exc:
        raise RestoreError(
            "could not move the restored data into place — ensure the target dir "
            "is writable and not held by a running container, then retry."
        ) from exc


def _move_into_place(staging: Path, data_dir: Path, settings) -> None:
    """Swap the staged DB / key / ciphertexts / Caddy trees into the live data dir.

    Order matters and the whole sequence is rerunnable from the same tar.
    A pre-existing legacy `.json` in the live `secrets/` is never touched —
    only the `.enc` ciphertext set is replaced. Each Caddy
    subtree is likewise replaced only when the archive carries it.
    """
    # Stale WAL/SHM sidecars against a fresh snapshot corrupt reads — drop them.
    for sidecar in ("nerdit.db-wal", "nerdit.db-shm"):
        (data_dir / sidecar).unlink(missing_ok=True)

    db_src = staging / "db" / "nerdit.db"
    if not db_src.is_file():
        raise RestoreError("archive is missing db/nerdit.db")
    os.replace(db_src, data_dir / "nerdit.db")

    # Key destination honors the [security].secrets_key_file override (F2):
    # restoring to the default path on an overridden install would fail to
    # decrypt every secret.
    key_override = settings.security.secrets_key_file
    key_dest = Path(key_override).expanduser() if key_override else data_dir / "secrets.key"
    key_src = staging / "secrets" / "secrets.key"
    if not key_src.is_file():
        raise RestoreError("archive is missing secrets/secrets.key")
    key_dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(key_src, key_dest)
    # A restored key invalidates any staged rotation — drop a stale `.new`.
    (key_dest.with_name(key_dest.name + ".new")).unlink(missing_ok=True)

    # Secrets ciphertexts: replace the .enc set only, preserving any pre-existing
    # legacy .json (a supported straggler state). A crash
    # between the key replace above and here leaves the daemon booting into a
    # SecretDecryptError (never a silent {}); recovery is a rerun.
    secrets_dir = data_dir / "secrets"
    secrets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    for existing in secrets_dir.glob("*.enc"):
        existing.unlink(missing_ok=True)
    store_src = staging / "secrets" / "store"
    if store_src.is_dir():
        for enc in store_src.glob("*.enc"):
            os.replace(enc, secrets_dir / enc.name)
    os.chmod(secrets_dir, 0o700)

    # Replace only archived Caddy trees. Missing certificates/acme in older backups
    # means preserve live state, avoiding needless reissuance and CA rate limits.
    for subtree in ("pki", "certificates", "acme"):
        src = staging / "caddy" / subtree
        if not src.is_dir():
            continue
        dest = data_dir / "caddy" / subtree
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(src, dest)
