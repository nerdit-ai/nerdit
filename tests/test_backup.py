"""Tests for core/backup.py — DB snapshot, capture orchestration, manifest.

Direct-construction only (SecretManager(tmp_path) / Database(tmp file)) — no
TestClient. The concurrent-writer + VACUUM-autocommit rows here double as the
§9.4 aiosqlite-floor decision artifact (pyproject.toml stays at >=0.20).
"""

from __future__ import annotations

import asyncio
import sqlite3
import stat
import tarfile

import pytest

from nerdit.core import backup as backup_mod
from nerdit.core.backup import (
    BackupError,
    _schema_info,
    create_backup,
    snapshot_db,
)
from nerdit.core.secrets import SecretManager
from nerdit.db.database import Database
from nerdit.db.queries import Queries


async def _make_db(tmp_path) -> Database:
    db = Database(str(tmp_path / "nerdit.db"))
    await db.connect()
    await db.init_schema()
    return db


async def _seed_rows(db: Database, n: int = 5) -> None:
    q = Queries(db)
    for i in range(n):
        await q.insert_audit_log(action=f"test.row{i}", result="ok")


def _make_secrets(tmp_path) -> SecretManager:
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("app", {"API_KEY": "abc"})
    mgr.set("other", {"X": "y"})
    return mgr


# --- snapshot_db --------------------------------------------------------------


@pytest.mark.asyncio
async def test_vacuum_into_under_write_lock_autocommit(tmp_path):
    """§9.4 decision artifact: VACUUM INTO ? under write_lock runs autocommit."""
    db = await _make_db(tmp_path)
    await _seed_rows(db, 7)
    dest = tmp_path / "snap" / "nerdit.db"
    await snapshot_db(db, dest)

    # No WAL/SHM sidecars beside the snapshot.
    assert not (dest.parent / "nerdit.db-wal").exists()
    assert not (dest.parent / "nerdit.db-shm").exists()

    conn = sqlite3.connect(str(dest))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        rows = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        assert rows == 7
    finally:
        conn.close()
    await db.close()


@pytest.mark.asyncio
async def test_snapshot_under_concurrent_writer_integrity(tmp_path):
    """A writer looping @_serialized inserts while snapshot_db runs → snapshot ok."""
    db = await _make_db(tmp_path)
    q = Queries(db)
    stop = False

    async def writer():
        i = 0
        while not stop:
            await q.insert_audit_log(action=f"w{i}", result="ok")
            i += 1
            await asyncio.sleep(0)
        return i

    task = asyncio.create_task(writer())
    await asyncio.sleep(0)  # let the writer get going
    dest = tmp_path / "snap" / "nerdit.db"
    await snapshot_db(db, dest)
    stop = True
    written = await task

    conn = sqlite3.connect(str(dest))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    assert written > 0
    await db.close()


@pytest.mark.asyncio
async def test_backup_fallback_unlinks_partial_dest(tmp_path, monkeypatch):
    """L7: a forced OperationalError falls back to .backup after unlinking a partial."""
    db = await _make_db(tmp_path)
    await _seed_rows(db, 3)

    dest = tmp_path / "snap" / "nerdit.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"garbage-partial-vacuum-output")  # pre-planted partial

    real_execute = db.conn.execute

    async def fake_execute(sql, *args, **kwargs):
        if sql.lstrip().upper().startswith("VACUUM"):
            raise sqlite3.OperationalError("forced VACUUM failure")
        return await real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(db.conn, "execute", fake_execute)

    await snapshot_db(db, dest)

    conn = sqlite3.connect(str(dest))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 3
    finally:
        conn.close()
    await db.close()


# --- create_backup: tar member set -------------------------------------------


@pytest.mark.asyncio
async def test_tar_member_set(tmp_path):
    db = await _make_db(tmp_path)
    await _seed_rows(db, 4)
    mgr = _make_secrets(tmp_path)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    tar_path = tmp_path / "backups" / result.basename
    assert tar_path.exists()
    assert tar_path.match("nerdit-backup-*.tar.gz")
    assert stat.S_IMODE(tar_path.stat().st_mode) == 0o600

    # No partial / staging inodes leak into the backups dir.
    leftovers = [p.name for p in (tmp_path / "backups").iterdir() if p != tar_path]
    assert leftovers == []

    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        names = {m.name for m in members}
        regular = {m.name for m in members if m.isreg()}
        # Only regular files + dirs, ever.
        for m in members:
            assert m.isreg() or m.isdir(), m.name
            assert not m.name.startswith("/")  # relative arcnames
            assert ".." not in m.name.split("/")

    assert "manifest.json" in regular
    assert "db/nerdit.db" in regular
    assert "secrets/secrets.key" in regular
    assert "secrets/store/app.enc" in regular
    assert "secrets/store/other.enc" in regular
    # No temp/leftover artifacts inside the tar.
    assert not any(n.endswith(".tmp") for n in names)
    assert not any(n.endswith(".json") and n != "manifest.json" for n in names)
    # tar file member mode is 0o600.
    for m in members:
        if m.isreg():
            assert m.mode == 0o600, m.name
        if m.isdir():
            assert m.mode == 0o700, m.name


@pytest.mark.asyncio
async def test_capture_side_symlink_rejection(tmp_path):
    """M2: a symlinked .enc and a symlink inside caddy/pki are skipped, never packed."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)

    secret_target = tmp_path / "outside-secret.txt"
    secret_target.write_text("TOP-SECRET-EXFIL")
    (tmp_path / "secrets" / "evil.enc").symlink_to(secret_target)

    pki = tmp_path / "caddy" / "pki" / "authorities" / "local"
    pki.mkdir(parents=True)
    (pki / "root.crt").write_text("real-ca-cert")
    caddy_target = tmp_path / "outside-ca.txt"
    caddy_target.write_text("OUTSIDE-CA-EXFIL")
    (tmp_path / "caddy" / "pki" / "evil.crt").symlink_to(caddy_target)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    assert "evil.enc" in result.manifest["skipped"]
    assert "evil.crt" in result.manifest["skipped"]

    tar_path = tmp_path / "backups" / result.basename
    with tarfile.open(tar_path) as tar:
        names = {m.name for m in tar.getmembers()}
        blobs = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isreg())
    assert "secrets/store/evil.enc" not in names
    assert b"TOP-SECRET-EXFIL" not in blobs
    assert b"OUTSIDE-CA-EXFIL" not in blobs
    # the real cert IS present
    assert any(n.endswith("root.crt") for n in names)


@pytest.mark.asyncio
async def test_capture_side_symlinked_pki_root_rejected(tmp_path):
    """M2 root case: a symlinked caddy/pki root is refused, not walked."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)

    # A directory of real files OUTSIDE the data dir the pki root points at.
    outside = tmp_path / "outside-pki"
    outside.mkdir()
    (outside / "stolen.crt").write_text("OUTSIDE-CA-EXFIL")
    (tmp_path / "caddy").mkdir()
    (tmp_path / "caddy" / "pki").symlink_to(outside, target_is_directory=True)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    assert "pki" in result.manifest["skipped"]
    assert result.manifest["contents"]["caddy_pki"] is False

    tar_path = tmp_path / "backups" / result.basename
    with tarfile.open(tar_path) as tar:
        names = {m.name for m in tar.getmembers()}
        blobs = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isreg())
    assert not any(n.startswith("caddy/") for n in names)
    assert b"OUTSIDE-CA-EXFIL" not in blobs


@pytest.mark.asyncio
async def test_staging_rmtree_on_failure(tmp_path, monkeypatch):
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)

    async def boom(db_, dest):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(backup_mod, "snapshot_db", boom)

    with pytest.raises(BackupError) as exc_info:
        await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    # Path-free message (M3): no absolute paths leak.
    assert "/" not in str(exc_info.value)
    assert "No space left on device" in str(exc_info.value)

    backups = tmp_path / "backups"
    leftovers = [p.name for p in backups.iterdir()]
    assert not any(n.startswith(".staging-") or n.startswith(".tmp-") for n in leftovers)


@pytest.mark.asyncio
async def test_manifest_schema_fields(tmp_path):
    db = await _make_db(tmp_path)
    await _seed_rows(db, 2)
    mgr = _make_secrets(tmp_path)

    r1 = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    m1 = r1.manifest
    assert m1["version"] == 1
    assert "created_at" in m1
    assert m1["schema"]["user_version"] == 1
    assert len(m1["schema"]["tables_sha256"]) == 64
    assert "skipped" in m1
    assert m1["kid"] == r1.kid
    assert m1["contents"]["db"] is True
    assert m1["contents"]["secrets_key"] is True
    assert m1["contents"]["secrets"] == 2

    # Stable across two snapshots of the same schema.
    r2 = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    assert r2.manifest["schema"]["tables_sha256"] == m1["schema"]["tables_sha256"]

    # Changes when a table is added.
    await db.conn.execute("CREATE TABLE extra_table (id INTEGER PRIMARY KEY)")
    await db.conn.commit()
    r3 = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()
    assert r3.manifest["schema"]["tables_sha256"] != m1["schema"]["tables_sha256"]


@pytest.mark.asyncio
async def test_schema_info_reads_snapshot_readonly(tmp_path):
    """_schema_info uses a read-only URI and returns user_version + digest."""
    db = await _make_db(tmp_path)
    dest = tmp_path / "snap" / "nerdit.db"
    await snapshot_db(db, dest)
    await db.close()

    info = _schema_info(dest)
    assert info["user_version"] == 1
    assert len(info["tables_sha256"]) == 64


@pytest.mark.asyncio
async def test_exclude_paths_keeps_node_key_out_of_the_tar(tmp_path):
    """PR #113 review (P27 WP-C3 item 3): an operator-chosen [link].key_file
    planted INSIDE a captured tree (caddy/pki/) must never enter the tar —
    possession of a backup would otherwise permit node impersonation."""
    db = await _make_db(tmp_path)
    await _seed_rows(db, 2)
    mgr = _make_secrets(tmp_path)

    pki = tmp_path / "caddy" / "pki" / "authorities" / "local"
    pki.mkdir(parents=True)
    (pki / "root.crt").write_bytes(b"fake-root-cert-not-a-secret")
    node_key = pki / "node.key"
    node_key.write_text("fixture-node-key-not-a-real-secret\n")

    result = await create_backup(
        db=db, secret_manager=mgr, data_dir=tmp_path, exclude_paths=(node_key,)
    )
    await db.close()

    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        names = {m.name for m in tar.getmembers()}
    assert "caddy/pki/authorities/local/root.crt" in names
    assert not any(n.endswith("node.key") for n in names)
    # The exclusion is reported, never silent.
    assert "node.key" in result.manifest["skipped"]


# --- P26 WP2: the ACME subtrees ride along ------------------------------------


def _plant_acme_trees(tmp_path) -> None:
    """Write the measured Caddy ACME layout under ``<data_dir>/caddy/``.

    Mirrors a real post-issuance tree (explorer §5): the sanitized directory
    URL is the first level under BOTH ``certificates/`` and ``acme/``.
    """
    leaf = tmp_path / "caddy" / "certificates" / "127.0.0.1-14000-dir" / "app.example.test"
    leaf.mkdir(parents=True)
    (leaf / "app.example.test.crt").write_text("leaf-chain-pem")
    (leaf / "app.example.test.key").write_text("leaf-key-pem-not-a-real-key")
    (leaf / "app.example.test.json").write_text('{"sans": ["app.example.test"]}')

    account = tmp_path / "caddy" / "acme" / "127.0.0.1-14000-dir" / "users" / "ops@example.test"
    account.mkdir(parents=True)
    (account / "admin.key").write_text("account-key-pem-not-a-real-key")
    (account / "admin.json").write_text('{"status": "valid"}')


@pytest.mark.asyncio
async def test_capture_includes_certificates_and_acme_trees(tmp_path):
    """§0.2-13 / D-P26-7: a restore must not have to re-issue, so both ACME
    subtrees join the tar — at the same 0o600 as every other captured file."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)
    _plant_acme_trees(tmp_path)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    assert result.manifest["contents"]["caddy_certificates"] is True
    assert result.manifest["contents"]["caddy_acme"] is True
    # pki was never created here — the three flags are independent.
    assert result.manifest["contents"]["caddy_pki"] is False

    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        members = tar.getmembers()
    names = {m.name for m in members}
    assert "caddy/certificates/127.0.0.1-14000-dir/app.example.test/app.example.test.crt" in names
    assert "caddy/certificates/127.0.0.1-14000-dir/app.example.test/app.example.test.key" in names
    assert "caddy/acme/127.0.0.1-14000-dir/users/ops@example.test/admin.key" in names
    for m in members:
        if m.name.startswith("caddy/") and m.isreg():
            assert m.mode == 0o600, m.name


@pytest.mark.asyncio
async def test_capture_absent_acme_trees_report_false(tmp_path):
    """ACME never enabled (or proxy off): absent trees are tolerated and the
    manifest says so — the flags are facts about the tar, not about intent."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    contents = result.manifest["contents"]
    assert contents["caddy_pki"] is False
    assert contents["caddy_certificates"] is False
    assert contents["caddy_acme"] is False
    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        names = {m.name for m in tar.getmembers()}
    assert not any(n.startswith("caddy/") for n in names)


@pytest.mark.asyncio
async def test_capture_side_symlink_inside_acme_is_skipped(tmp_path):
    """M2, extended to the new trees: a symlink planted under caddy/acme/ is
    reported by basename and its target never enters the tar."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)
    _plant_acme_trees(tmp_path)

    outside = tmp_path / "outside-account.txt"
    outside.write_text("OUTSIDE-ACME-EXFIL")
    (tmp_path / "caddy" / "acme" / "evil.key").symlink_to(outside)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    assert "evil.key" in result.manifest["skipped"]
    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        names = {m.name for m in tar.getmembers()}
        blobs = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isreg())
    assert not any(n.endswith("evil.key") for n in names)
    assert b"OUTSIDE-ACME-EXFIL" not in blobs
    # The real account key IS present — the skip is surgical, not a bail-out.
    assert any(n.endswith("admin.key") for n in names)


@pytest.mark.asyncio
async def test_capture_side_symlinked_acme_root_rejected(tmp_path):
    """M2 root case for the new trees: a symlinked caddy/acme root is refused
    before the walk, reported by basename, and reads as absent."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)

    outside = tmp_path / "outside-acme"
    outside.mkdir()
    (outside / "stolen.key").write_text("OUTSIDE-ACME-EXFIL")
    (tmp_path / "caddy").mkdir()
    (tmp_path / "caddy" / "acme").symlink_to(outside, target_is_directory=True)

    result = await create_backup(db=db, secret_manager=mgr, data_dir=tmp_path)
    await db.close()

    assert "acme" in result.manifest["skipped"]
    assert result.manifest["contents"]["caddy_acme"] is False
    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        blobs = b"".join(tar.extractfile(m).read() for m in tar.getmembers() if m.isreg())
    assert b"OUTSIDE-ACME-EXFIL" not in blobs


@pytest.mark.asyncio
async def test_exclude_paths_apply_to_the_acme_tree_too(tmp_path):
    """The node-link key exclusion follows the material, not the subtree: a
    ``[link].key_file`` planted under caddy/acme/ must be excluded exactly as
    one planted under caddy/pki/ (P27 WP-C3 item 3)."""
    db = await _make_db(tmp_path)
    mgr = _make_secrets(tmp_path)
    _plant_acme_trees(tmp_path)

    node_key = tmp_path / "caddy" / "acme" / "node.key"
    node_key.write_text("fixture-node-key-not-a-real-secret\n")

    result = await create_backup(
        db=db, secret_manager=mgr, data_dir=tmp_path, exclude_paths=(node_key,)
    )
    await db.close()

    with tarfile.open(tmp_path / "backups" / result.basename) as tar:
        names = {m.name for m in tar.getmembers()}
    assert not any(n.endswith("node.key") for n in names)
    assert "caddy/acme/127.0.0.1-14000-dir/users/ops@example.test/admin.key" in names
    assert "node.key" in result.manifest["skipped"]
