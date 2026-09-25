"""Tests for backup v2 — per-database volume tars (P15 WP7).

Direct-construction only (scratch ``data_dir`` trees, no TestClient/aiosqlite —
the standing sandbox caveat): the capture allowlist + naming + manifest of
``create_volume_backup``, the per-service keep-last sweep, the v1 control-plane
walkers staying blind to the new glob, the ``/system/disk`` ``volume_backups``
bucket via ``_build_disk_report``, and the offline ``_run_volume_restore``
round-trip (incl. the D9 renamed-tar-does-not-retarget guarantee).
"""

from __future__ import annotations

import io
import os
import re
import stat
import tarfile

import pytest
import typer

import nerdit.core.backup as core_backup
from nerdit.cli.commands import backup as backup_mod
from nerdit.cli.commands.backup import _run_volume_restore
from nerdit.config.settings import NerditSettings
from nerdit.core.backup import BackupError, _collect_volume_entries, create_volume_backup
from nerdit.daemon.routes.system import (
    _build_disk_report,
    _walk_tars,
)
from nerdit.daemon.server import _prune_backups, _prune_volume_backups

_NAME_RE = re.compile(r"^nerdit-volumes-(?P<svc>.+)-\d{8}T\d{6}Z-[0-9a-f]{6}\.tar\.gz$")


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #


def _seed_db_tree(data_dir, service="pg", *, files=(("data/pgdata/PG_VERSION", b"16"),)):
    """Create ``<data_dir>/services/<service>/`` with some files; return its root."""
    root = data_dir / "services" / service
    for rel, content in files:
        fpath = root / rel
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(content)
    return root


def _tar_names(tar_path):
    with tarfile.open(tar_path, "r:*") as tar:
        return sorted(tar.getnames())


# --------------------------------------------------------------------------- #
# capture / naming / manifest
# --------------------------------------------------------------------------- #


def test_naming_and_manifest(tmp_path):
    _seed_db_tree(tmp_path, "pg", files=(("data/pgdata/PG_VERSION", b"16"), ("data/base/1", b"x")))
    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")

    # Naming grammar (outside the v1 nerdit-backup-* glob).
    assert _NAME_RE.match(result.basename), result.basename
    assert _NAME_RE.match(result.basename).group("svc") == "pg"
    assert result.path.endswith(result.basename)

    # Dir 0o700, tar 0o600 (P14c write discipline).
    backups = tmp_path / "backups"
    assert stat.S_IMODE(os.stat(backups).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(result.path).st_mode) == 0o600

    # Manifest shape.
    m = result.manifest
    assert m["version"] == 1
    assert m["service"] == "pg"
    assert m["backend"] == "postgres"
    assert m["contents"]["volumes"] == ["data"]
    assert m["skipped"] == []
    assert "created_at" in m

    # Members: the manifest + the service tree under services/<name>/.
    names = _tar_names(result.path)
    assert "volume-manifest.json" in names
    assert "services/pg/data/pgdata/PG_VERSION" in names
    assert "services/pg/data/base/1" in names
    # No leftover tmp tar.
    assert not list(backups.glob(".tmp-*"))


def test_unreadable_volume_dir_fails_loudly(tmp_path):
    """An unreadable volume subtree must raise, never ship a silent-empty tar.

    A database container runs as its own non-root user (D-P15-6), so PGDATA
    lands 0o700 owned by the in-image uid; a daemon on a different uid cannot
    descend into it. os.walk() would swallow the EACCES and produce a tar that
    falsely claims to hold the data — the capture must fail loudly instead.
    """
    root = _seed_db_tree(tmp_path, "pg", files=(("data/pgdata/PG_VERSION", b"16"),))
    locked = root / "data" / "pgdata"
    os.chmod(locked, 0o000)  # owner without r+x cannot traverse
    try:
        with pytest.raises(BackupError):
            create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
        # No lying tar left behind.
        assert not list((tmp_path / "backups").glob("nerdit-volumes-*.tar.gz"))
        assert not list((tmp_path / "backups").glob(".tmp-*"))
    finally:
        os.chmod(locked, 0o700)


def test_hyphenated_service_name_parses(tmp_path):
    _seed_db_tree(tmp_path, "my-db")
    result = create_volume_backup(data_dir=tmp_path, service="my-db", backend="postgres")
    m = _NAME_RE.match(result.basename)
    assert m is not None and m.group("svc") == "my-db"


def test_capture_skips_symlink_and_reports(tmp_path):
    root = _seed_db_tree(tmp_path, "pg")
    # A symlink inside the tree must be skipped + reported, never followed.
    (tmp_path / "outside.txt").write_bytes(b"secret")
    os.symlink(tmp_path / "outside.txt", root / "leak.link")

    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    assert "leak.link" in result.manifest["skipped"]
    assert "services/pg/leak.link" not in _tar_names(result.path)
    # The real file is still captured.
    assert "services/pg/data/pgdata/PG_VERSION" in _tar_names(result.path)


def test_symlinked_service_root_captures_nothing(tmp_path):
    # A symlinked service root is reported and never followed off-box.
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "loot").write_bytes(b"x")
    (tmp_path / "services").mkdir(parents=True, exist_ok=True)
    os.symlink(tmp_path / "elsewhere", tmp_path / "services" / "pg")

    skipped: list[str] = []
    entries = _collect_volume_entries(tmp_path / "services" / "pg", skipped)
    assert entries == []
    assert "pg" in skipped


# --------------------------------------------------------------------------- #
# integrity: never a corrupt/partial member, never a lying empty tar (#2 #4 #8)
# --------------------------------------------------------------------------- #


def test_shrink_during_pack_stays_valid(tmp_path, monkeypatch):
    """#2: a file truncated between the pack lstat and the read must not corrupt.

    The old code stamped ``info.size`` from the lstat then copied the file — a
    live DB shrinking it mid-copy left a header claiming more bytes than exist,
    misaligning the whole gzip stream into a silently corrupt archive reported as
    success. The snapshot-first path records the ACTUAL byte count, so the tar
    always opens cleanly and the member is intact.
    """
    _seed_db_tree(tmp_path, "pg", files=(("data/big", b"X" * 100),))
    real_open = open

    def fake_open(file, *args, **kwargs):
        if os.fspath(file).endswith("/data/big"):
            return io.BytesIO(b"X" * 10)  # lstat said 100, only 10 bytes here now
        return real_open(file, *args, **kwargs)

    # Shadow the builtin `open` used inside core.backup's snapshot read only.
    monkeypatch.setattr(core_backup, "open", fake_open, raising=False)
    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")

    with tarfile.open(result.path, "r:*") as tar:  # clean open == not corrupt
        member = tar.getmember("services/pg/data/big")
        data = tar.extractfile(member).read()
    assert member.size == len(data) == 10


def test_vanished_file_during_pack_is_skipped_and_reported(tmp_path, monkeypatch):
    """#2/#8: a file removed between capture and pack → skipped + manifest-reported."""
    _seed_db_tree(tmp_path, "pg", files=(("data/keep", b"k"), ("data/gone", b"g")))
    real_open = open

    def fake_open(file, *args, **kwargs):
        if os.fspath(file).endswith("/data/gone"):
            raise FileNotFoundError(2, "No such file or directory", os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(core_backup, "open", fake_open, raising=False)
    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")

    assert "gone" in result.manifest["skipped"]
    names = _tar_names(result.path)
    assert "services/pg/data/keep" in names
    assert "services/pg/data/gone" not in names


def test_pack_records_uid_gid(tmp_path):
    """#3: numeric uid/gid are recorded in the TarInfo (forensics; not re-applied)."""
    _seed_db_tree(tmp_path, "pg")
    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    with tarfile.open(result.path, "r:*") as tar:
        member = tar.getmember("services/pg/data/pgdata/PG_VERSION")
    assert member.uid == os.getuid()
    assert member.gid == os.getgid()


def test_never_launched_row_rejected(tmp_path):
    """#4: an empty (never-launched) data dir must fail loudly, never a success tar."""
    (tmp_path / "services" / "pg").mkdir(parents=True)  # created, never launched
    with pytest.raises(BackupError):
        create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    assert not list((tmp_path / "backups").glob("nerdit-volumes-*.tar.gz"))
    assert not list((tmp_path / "backups").glob(".tmp-*"))


def test_missing_service_root_rejected(tmp_path):
    """#4: a service whose data dir does not exist yet is rejected loudly."""
    with pytest.raises(BackupError):
        create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    assert not list((tmp_path / "backups").glob("nerdit-volumes-*.tar.gz"))


def test_symlink_service_root_rejected(tmp_path):
    """#4: a symlinked data root is refused (never followed off-box into a tar)."""
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "loot").write_bytes(b"x")
    (tmp_path / "services").mkdir(parents=True)
    os.symlink(tmp_path / "elsewhere", tmp_path / "services" / "pg")
    with pytest.raises(BackupError):
        create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    assert not list((tmp_path / "backups").glob("nerdit-volumes-*.tar.gz"))


# --------------------------------------------------------------------------- #
# sweep + walkers
# --------------------------------------------------------------------------- #


def _touch_tar(backups, name, mtime):
    backups.mkdir(parents=True, exist_ok=True)
    p = backups / name
    with tarfile.open(p, "w:gz"):
        pass
    os.utime(p, (mtime, mtime))
    return p


def test_prune_volume_backups_keep_last_per_service(tmp_path):
    backups = tmp_path / "backups"
    # Two services, three tars each, distinct mtimes (older -> newer).
    for svc in ("pg", "redis"):
        for i in range(3):
            _touch_tar(backups, f"nerdit-volumes-{svc}-2026071{i}T000000Z-aaaaaa.tar.gz", 1000 + i)

    removed = _prune_volume_backups(tmp_path, keep_last=1)
    assert removed == 4  # 2 removed per service

    survivors = sorted(p.name for p in backups.glob("nerdit-volumes-*.tar.gz"))
    # The newest of each service survives (mtime 1002).
    assert survivors == [
        "nerdit-volumes-pg-20260712T000000Z-aaaaaa.tar.gz",
        "nerdit-volumes-redis-20260712T000000Z-aaaaaa.tar.gz",
    ]


def test_prune_volume_backups_keep_zero_is_noop(tmp_path):
    backups = tmp_path / "backups"
    _touch_tar(backups, "nerdit-volumes-pg-20260712T000000Z-aaaaaa.tar.gz", 1000)
    # keep<=0 short-circuits the retention loop (never called), but the walker
    # itself is defensive: files[:-0] == [] so it removes nothing regardless.
    assert _prune_volume_backups(tmp_path, keep_last=0) == 0
    assert len(list(backups.glob("nerdit-volumes-*.tar.gz"))) == 1


def test_v1_walkers_blind_to_volume_glob(tmp_path):
    backups = tmp_path / "backups"
    # Two control-plane tars (so keep-last=1 has a real candidate) + two volume
    # tars for the same service (so its keep-last=1 has a candidate too).
    _touch_tar(backups, "nerdit-backup-20260711T000000Z-aaaaaa.tar.gz", 1000)
    cp_new = _touch_tar(backups, "nerdit-backup-20260713T000000Z-cccccc.tar.gz", 2000)
    vol_old = _touch_tar(backups, "nerdit-volumes-pg-20260711T000000Z-bbbbbb.tar.gz", 1000)
    vol_new = _touch_tar(backups, "nerdit-volumes-pg-20260713T000000Z-dddddd.tar.gz", 2000)

    # The v1 control-plane prune removes exactly the older CONTROL-PLANE tar and
    # never a volume tar — even though a volume tar is older than the survivor.
    assert _prune_backups(tmp_path, keep_last=1) == 1
    assert cp_new.exists()
    assert vol_old.exists() and vol_new.exists()  # both untouched by the v1 walker

    # Conversely the volume prune removes only the older VOLUME tar, never the
    # control-plane one.
    assert _prune_volume_backups(tmp_path, keep_last=1) == 1
    assert not vol_old.exists()
    assert vol_new.exists()
    assert cp_new.exists()  # control-plane tar untouched by the volume walker

    # The disk walkers count disjoint sets: only the surviving tar of each flavour.
    assert _walk_tars(backups, "nerdit-backup-*.tar.gz")["count"] == 1
    assert _walk_tars(backups, "nerdit-volumes-*.tar.gz")["count"] == 1


# --------------------------------------------------------------------------- #
# /system/disk bucket
# --------------------------------------------------------------------------- #


class _FakeRuntime:
    async def disk_usage(self):  # noqa: ANN201
        return {"images_bytes": 0, "containers_bytes": 0, "volumes_bytes": 0}

    async def list_images_detailed(self):  # noqa: ANN201
        return []


class _FakeQueries:
    async def list_workload_configs(self):  # noqa: ANN201
        return []


@pytest.mark.asyncio
async def test_disk_bucket_counts_volume_backups(tmp_path):
    # A staged volume tar must be visible in the disk report's own bucket.
    _seed_db_tree(tmp_path, "pg")
    result = create_volume_backup(data_dir=tmp_path, service="pg", backend="postgres")
    size = os.stat(result.path).st_size

    report = await _build_disk_report(_FakeRuntime(), _FakeQueries(), tmp_path, None)
    bucket = report["data_dir"]["volume_backups"]
    assert bucket["count"] == 1
    assert bucket["bytes"] == size
    # The control-plane bucket does not double-count it.
    assert report["data_dir"]["backups"]["count"] == 0


# --------------------------------------------------------------------------- #
# offline restore round-trip (D9)
# --------------------------------------------------------------------------- #


def _settings(data_dir, *, port=59998, pid_file=None) -> NerditSettings:
    pid_file = pid_file if pid_file is not None else str(data_dir / "nonexistent.pid")
    return NerditSettings(data_dir=str(data_dir), daemon={"pid_file": pid_file, "port": port})


def _patch(monkeypatch, settings):
    monkeypatch.setattr(backup_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(backup_mod, "_health_probe", lambda port: False)


def test_volume_restore_round_trip(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _seed_db_tree(src, "pg", files=(("data/pgdata/PG_VERSION", b"16"), ("data/base/x", b"hello")))
    result = create_volume_backup(data_dir=src, service="pg", backend="postgres")

    dst = tmp_path / "dst"
    _patch(monkeypatch, _settings(dst))
    _run_volume_restore(result.path, yes=True)

    restored = dst / "services" / "pg" / "data"
    assert (restored / "pgdata" / "PG_VERSION").read_bytes() == b"16"
    assert (restored / "base" / "x").read_bytes() == b"hello"


def test_volume_restore_is_rerunnable(tmp_path, monkeypatch):
    src = tmp_path / "src"
    _seed_db_tree(src, "pg")
    result = create_volume_backup(data_dir=src, service="pg", backend="postgres")

    dst = tmp_path / "dst"
    _patch(monkeypatch, _settings(dst))
    _run_volume_restore(result.path, yes=True)
    # A second run from the same tar succeeds (idempotent move-in).
    _run_volume_restore(result.path, yes=True)
    assert (dst / "services" / "pg" / "data" / "pgdata" / "PG_VERSION").is_file()


def test_renamed_tar_does_not_retarget(tmp_path, monkeypatch):
    # D9: the manifest ``service`` field selects the target, never the filename.
    src = tmp_path / "src"
    _seed_db_tree(src, "pg", files=(("data/marker", b"pg-data"),))
    result = create_volume_backup(data_dir=src, service="pg", backend="postgres")

    # Rename the tar to impersonate a different service in its filename.
    renamed = (tmp_path / "src" / "backups") / "nerdit-volumes-evil-20260712T000000Z-abcdef.tar.gz"
    os.replace(result.path, renamed)

    dst = tmp_path / "dst"
    _patch(monkeypatch, _settings(dst))
    _run_volume_restore(renamed, yes=True)

    # Restored under the manifest service (pg), NOT the filename service (evil).
    assert (dst / "services" / "pg" / "data" / "marker").read_bytes() == b"pg-data"
    assert not (dst / "services" / "evil").exists()


def test_volume_restore_rejects_bad_manifest_service(tmp_path, monkeypatch):
    # A forged manifest with a non-DNS-label service is rejected before move-in.
    backups = tmp_path / "src" / "backups"
    backups.mkdir(parents=True)
    bad = backups / "nerdit-volumes-x-20260712T000000Z-abcdef.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        import io

        payload = b'{"version": 1, "service": "../etc", "backend": "postgres"}'
        info = tarfile.TarInfo("volume-manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    dst = tmp_path / "dst"
    _patch(monkeypatch, _settings(dst))
    with pytest.raises(typer.Exit) as exc:
        _run_volume_restore(bad, yes=True)
    assert exc.value.exit_code == 1


def test_restored_files_owned_by_restoring_user(tmp_path, monkeypatch):
    """#3: under the daemon-uid posture, restored data is owned by the restorer.

    Plain extraction as the daemon user IS the correct ownership — the recorded
    tar uid/gid are never re-applied. Files land 0o600, dirs 0o700.
    """
    src = tmp_path / "src"
    _seed_db_tree(src, "pg", files=(("data/pgdata/PG_VERSION", b"16"),))
    result = create_volume_backup(data_dir=src, service="pg", backend="postgres")

    dst = tmp_path / "dst"
    _patch(monkeypatch, _settings(dst))
    _run_volume_restore(result.path, yes=True)

    f = dst / "services" / "pg" / "data" / "pgdata" / "PG_VERSION"
    assert f.stat().st_uid == os.getuid()
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    d = dst / "services" / "pg" / "data" / "pgdata"
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_move_in_failure_surfaces_restore_error(tmp_path, monkeypatch):
    """#5: an unclearable target (ENOTEMPTY) → RestoreError, never a raw traceback.

    Simulate a target dir whose contents ``rmtree`` cannot clear (a pre-amendment
    build's container-uid files) by making ``rmtree`` a no-op — ``os.replace``
    onto the still-non-empty dir raises ``ENOTEMPTY``, which must be wrapped as a
    RestoreError (rendered "[red]Restore failed" + exit 1), not escape as a bare
    OSError traceback.
    """
    src = tmp_path / "src"
    _seed_db_tree(src, "pg")
    result = create_volume_backup(data_dir=src, service="pg", backend="postgres")

    dst = tmp_path / "dst"
    target = dst / "services" / "pg"
    target.mkdir(parents=True)
    (target / "stale").write_bytes(b"x")  # os.replace onto a non-empty dir → ENOTEMPTY

    _patch(monkeypatch, _settings(dst))
    monkeypatch.setattr(backup_mod.shutil, "rmtree", lambda *a, **k: None)
    with pytest.raises(typer.Exit) as exc:
        _run_volume_restore(result.path, yes=True)
    assert exc.value.exit_code == 1
