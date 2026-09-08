"""Tests for the offline ``nerdit restore`` (P14c WP6).

Direct-construction only: real ``create_backup`` tars (Database + SecretManager)
plus hand-crafted attack tars built with the ``tarfile`` module. ``_run_restore``
is driven with ``load_settings`` / ``_health_probe`` monkeypatched — no daemon, no
TestClient. The tar-slip battery hits ``_extract_hardened`` on BOTH the
``data_filter`` and the forced-manual branches (D7).
"""

from __future__ import annotations

import fcntl
import http.server
import io
import os
import stat
import tarfile
import threading

import pytest
import typer

from nerdit.cli.commands import backup as backup_mod
from nerdit.cli.commands.backup import (
    RestoreError,
    _extract_hardened,
    _run_restore,
)
from nerdit.config.settings import NerditSettings
from nerdit.core.backup import create_backup
from nerdit.core.secrets import SecretManager
from nerdit.db.database import Database
from nerdit.db.queries import Queries

# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


async def _make_db(path) -> Database:
    db = Database(str(path))
    await db.connect()
    await db.init_schema()
    return db


async def _build_backup(data_dir, *, key_file=None):
    """Populate a data dir + produce a real backup tar; return (tar_path, manifest)."""
    db = await _make_db(data_dir / "nerdit.db")
    q = Queries(db)
    for i in range(6):
        await q.insert_audit_log(action=f"seed.{i}", result="ok")
    mgr = SecretManager(data_dir / "secrets", key_path=key_file)
    mgr.set("app", {"API_KEY": "abc"})
    mgr.set("other", {"X": "y"})
    result = await create_backup(db=db, secret_manager=mgr, data_dir=data_dir)
    await db.close()
    return data_dir / "backups" / result.basename, result.manifest


def _settings(data_dir, *, port=59999, pid_file=None, key_file=None) -> NerditSettings:
    pid_file = pid_file if pid_file is not None else str(data_dir / "nonexistent.pid")
    security = {"secrets_key_file": key_file} if key_file else {}
    return NerditSettings(
        data_dir=str(data_dir),
        daemon={"pid_file": pid_file, "port": port},
        security=security,
    )


def _patch(monkeypatch, settings, *, health=False):
    monkeypatch.setattr(backup_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(backup_mod, "_health_probe", lambda port: health)


def _write_tar(path, members):
    """members = [(TarInfo, data|None)]."""
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)


def _reg(name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = len(data)
    return info, data


# --------------------------------------------------------------------------- #
# tar-slip battery — both branches reject the WHOLE archive
# --------------------------------------------------------------------------- #


def _attack_members():
    absolute = tarfile.TarInfo("/etc/passwd")
    absolute.type = tarfile.REGTYPE

    dotdot = tarfile.TarInfo("../escape")
    dotdot.type = tarfile.REGTYPE

    symlink = tarfile.TarInfo("link")
    symlink.type = tarfile.SYMTYPE
    symlink.linkname = "/etc/passwd"

    hardlink = tarfile.TarInfo("hard")
    hardlink.type = tarfile.LNKTYPE
    hardlink.linkname = "db/nerdit.db"

    fifo = tarfile.TarInfo("fifo")
    fifo.type = tarfile.FIFOTYPE

    chardev = tarfile.TarInfo("chr")
    chardev.type = tarfile.CHRTYPE

    blockdev = tarfile.TarInfo("blk")
    blockdev.type = tarfile.BLKTYPE

    unknown = tarfile.TarInfo("weird")
    unknown.type = b"X"  # unknown typeflag → allowlist rejects

    return {
        "absolute": absolute,
        "dotdot": dotdot,
        "symlink": symlink,
        "hardlink": hardlink,
        "fifo": fifo,
        "chardev": chardev,
        "blockdev": blockdev,
        "unknown": unknown,
    }


@pytest.mark.parametrize("attack", list(_attack_members().keys()))
@pytest.mark.parametrize("force_manual", [False, True])
def test_tar_slip_rejected_whole_archive(tmp_path, attack, force_manual):
    """Every non-regular/unsafe member rejects the whole archive; staging clean."""
    info = _attack_members()[attack]
    tar_path = tmp_path / "attack.tar.gz"
    # Pair the attack member with an otherwise-legit regular file.
    _write_tar(tar_path, [_reg("db/nerdit.db"), (info, None)])

    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    with tarfile.open(tar_path) as tar:
        with pytest.raises(RestoreError):
            _extract_hardened(tar, staging, force_manual=force_manual)
    # Nothing was extracted (reject happens before any write).
    assert list(staging.iterdir()) == []


def test_manual_realpath_alias_rejected(tmp_path):
    """A staged parent aliased to an out-of-tree dir fails the containment check."""
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    # Pre-plant `staging/db` as a symlink to an out-of-tree directory.
    (staging / "db").symlink_to(outside, target_is_directory=True)

    tar_path = tmp_path / "ok.tar.gz"
    _write_tar(tar_path, [_reg("db/nerdit.db", b"payload")])
    with tarfile.open(tar_path) as tar:
        with pytest.raises(RestoreError):
            _extract_hardened(tar, staging, force_manual=True)
    # The payload never landed on the aliased target.
    assert not (outside / "nerdit.db").exists()


def test_manual_o_nofollow_refuses_preexisting_symlink(tmp_path):
    """O_NOFOLLOW/O_EXCL refuses to write through a symlink already at dest."""
    staging = tmp_path / "staging"
    (staging / "db").mkdir(parents=True, mode=0o700)
    target = tmp_path / "target.txt"
    target.write_text("untouched")
    (staging / "db" / "nerdit.db").symlink_to(target)

    tar_path = tmp_path / "ok.tar.gz"
    _write_tar(tar_path, [_reg("db/nerdit.db", b"payload")])
    with tarfile.open(tar_path) as tar:
        with pytest.raises((OSError, RestoreError)):
            _extract_hardened(tar, staging, force_manual=True)
    assert target.read_text() == "untouched"


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_restore_round_trip(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, manifest = await _build_backup(data_dir)

    # Wipe DB + secrets, but leave a pre-existing legacy .json in the live dir.
    for name in ("nerdit.db", "nerdit.db-wal", "nerdit.db-shm", "secrets.key"):
        (data_dir / name).unlink(missing_ok=True)
    import shutil

    shutil.rmtree(data_dir / "secrets", ignore_errors=True)
    (data_dir / "secrets").mkdir(mode=0o700)
    (data_dir / "secrets" / "legacy.json").write_text('{"OLD": "kept"}')

    _patch(monkeypatch, _settings(data_dir))
    _run_restore(tar_path, yes=True)

    # DB rows readable.
    import sqlite3

    conn = sqlite3.connect(str(data_dir / "nerdit.db"))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 6
    finally:
        conn.close()

    # No stale sidecars.
    assert not (data_dir / "nerdit.db-wal").exists()
    assert not (data_dir / "nerdit.db-shm").exists()

    # Perms (before a migrating manager runs).
    assert stat.S_IMODE((data_dir / "secrets.key").stat().st_mode) == 0o600
    assert stat.S_IMODE((data_dir / "secrets").stat().st_mode) == 0o700
    assert stat.S_IMODE((data_dir / "secrets" / "app.enc").stat().st_mode) == 0o600

    # F4: the restore left the pre-existing legacy .json untouched (asserted
    # before we construct a SecretManager, which would migrate/consume it).
    assert (data_dir / "secrets" / "legacy.json").read_text() == '{"OLD": "kept"}'

    # Secrets load under the restored key; kid matches the manifest.
    mgr = SecretManager(data_dir / "secrets")
    assert mgr.load("app") == {"API_KEY": "abc"}
    assert mgr.load("other") == {"X": "y"}
    assert SecretManager._kid(mgr._read_key_file(mgr.key_path)) == manifest["kid"]


@pytest.mark.asyncio
async def test_restore_forced_manual_round_trip(tmp_path, monkeypatch):
    """The manual (non-data_filter) extraction path also restores cleanly (D7)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, manifest = await _build_backup(data_dir)

    import shutil

    for name in ("nerdit.db", "secrets.key"):
        (data_dir / name).unlink(missing_ok=True)
    shutil.rmtree(data_dir / "secrets", ignore_errors=True)

    _patch(monkeypatch, _settings(data_dir))
    _run_restore(tar_path, yes=True, force_manual=True)

    mgr = SecretManager(data_dir / "secrets")
    assert mgr.load("app") == {"API_KEY": "abc"}


@pytest.mark.asyncio
async def test_restore_honors_secrets_key_file_override(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    key_override = tmp_path / "custom" / "secrets.key"
    tar_path, manifest = await _build_backup(data_dir, key_file=str(key_override))

    import shutil

    (data_dir / "nerdit.db").unlink(missing_ok=True)
    key_override.unlink(missing_ok=True)
    shutil.rmtree(data_dir / "secrets", ignore_errors=True)
    # A stale staged-rotation sibling of the override path must be removed.
    key_override.parent.mkdir(parents=True, exist_ok=True)
    (key_override.parent / "secrets.key.new").write_bytes(b"stale")

    _patch(monkeypatch, _settings(data_dir, key_file=str(key_override)))
    _run_restore(tar_path, yes=True)

    assert key_override.is_file()
    assert not (key_override.parent / "secrets.key.new").exists()
    assert not (data_dir / "secrets.key").exists()

    mgr = SecretManager(data_dir / "secrets", key_path=str(key_override))
    assert mgr.load("app") == {"API_KEY": "abc"}
    assert SecretManager._kid(mgr._read_key_file(mgr.key_path)) == manifest["kid"]


@pytest.mark.asyncio
async def test_restore_rerunnable_after_partial(tmp_path, monkeypatch):
    """L8: a store-swap crash window is recovered by rerunning the same tar."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, _ = await _build_backup(data_dir)

    import shutil

    (data_dir / "nerdit.db").unlink(missing_ok=True)
    shutil.rmtree(data_dir / "secrets", ignore_errors=True)
    (data_dir / "secrets.key").unlink(missing_ok=True)

    _patch(monkeypatch, _settings(data_dir))
    _run_restore(tar_path, yes=True)

    # Simulate a crash after the key was replaced but before the store swap:
    # key present, ciphertexts gone → daemon would SecretDecryptError.
    shutil.rmtree(data_dir / "secrets", ignore_errors=True)
    assert (data_dir / "secrets.key").is_file()

    # Rerun from the same tar recovers a consistent state.
    _run_restore(tar_path, yes=True)
    mgr = SecretManager(data_dir / "secrets")
    assert mgr.load("app") == {"API_KEY": "abc"}


# --------------------------------------------------------------------------- #
# live-daemon refusal — both legs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_restore_refused_by_pid_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, _ = await _build_backup(data_dir)
    (data_dir / "nerdit.db").unlink(missing_ok=True)

    pid_file = data_dir / "live.pid"
    pid_file.write_text(str(os.getpid()))  # our own pid → is_running() True

    _patch(monkeypatch, _settings(data_dir, pid_file=str(pid_file)))
    with pytest.raises(typer.Exit) as exc:
        _run_restore(tar_path, yes=True)
    assert exc.value.exit_code == 1
    # Refused before touching anything.
    assert not (data_dir / "nerdit.db").exists()


@pytest.mark.asyncio
async def test_restore_refused_by_health_probe(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, _ = await _build_backup(data_dir)
    (data_dir / "nerdit.db").unlink(missing_ok=True)

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        # Real _health_probe (not patched); no pid file so is_running() is False.
        monkeypatch.setattr(backup_mod, "load_settings", lambda: _settings(data_dir, port=port))
        with pytest.raises(typer.Exit) as exc:
            _run_restore(tar_path, yes=True)
        assert exc.value.exit_code == 1
        assert not (data_dir / "nerdit.db").exists()
    finally:
        srv.shutdown()
        thread.join()


# --------------------------------------------------------------------------- #
# flock mutual exclusion
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_restore_refused_by_shared_lock_holder(tmp_path, monkeypatch):
    """A LOCK_SH holder (simulated live daemon) blocks the restore's LOCK_EX."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, _ = await _build_backup(data_dir)
    (data_dir / "nerdit.db").unlink(missing_ok=True)

    holder = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(holder, fcntl.LOCK_SH)
    try:
        _patch(monkeypatch, _settings(data_dir))
        with pytest.raises(typer.Exit) as exc:
            _run_restore(tar_path, yes=True)
        assert exc.value.exit_code == 1
        assert not (data_dir / "nerdit.db").exists()
    finally:
        os.close(holder)


@pytest.mark.asyncio
async def test_second_restore_refused_by_exclusive_lock(tmp_path, monkeypatch):
    """A concurrent restore (LOCK_EX holder) refuses a second restore."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tar_path, _ = await _build_backup(data_dir)
    (data_dir / "nerdit.db").unlink(missing_ok=True)

    holder = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        _patch(monkeypatch, _settings(data_dir))
        with pytest.raises(typer.Exit) as exc:
            _run_restore(tar_path, yes=True)
        assert exc.value.exit_code == 1
    finally:
        os.close(holder)
