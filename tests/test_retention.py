"""P14b WP-B2b — retention sweeps + archive-first audit prune.

Every sweep query is exercised against the real in-memory ``Database`` (never a
mock) so an SQL-syntax regression — the id-subquery chunk form, the per-table
cutoff format — is caught. Cutoffs follow the writer's format per table (§1.8):
``job_logs.timestamp`` and ``audit_log.ts`` are both ``datetime('now')`` space
strings.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI

import nerdit.core.retention as retention_module
import nerdit.daemon.server as server_module
from nerdit.core import workspaces as core_workspaces
from nerdit.core.discovery import GpuDiscoveryResult
from nerdit.core.retention import archive_and_prune_audit
from nerdit.core.runtime.stub import StubRuntime
from nerdit.db.database import Database
from nerdit.db.queries import Queries

# ``--asyncio-mode=auto`` marks the async tests; the sync helper tests below run
# as plain functions (no module-level asyncio mark, which would warn on them).


# --- direct-DB helpers --------------------------------------------------------


async def _insert_job(
    db: Database,
    job_id: str,
    *,
    kind: str = "batch",
    status: str = "completed",
    created_at: str | None = None,
    finished_at: str | None = None,
    service_name: str | None = None,
) -> None:
    await db.conn.execute(
        "INSERT INTO jobs (id, kind, status, created_at, finished_at, service_name) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, kind, status, created_at, finished_at, service_name),
    )
    await db.conn.commit()


async def _insert_log(db: Database, job_id: str, *, aged: bool) -> None:
    """Insert one job_log; ``aged`` rewrites the timestamp to 30 days ago."""
    await db.conn.execute("INSERT INTO job_logs (job_id, message) VALUES (?, ?)", (job_id, "line"))
    if aged:
        await db.conn.execute(
            "UPDATE job_logs SET timestamp = datetime('now', '-30 days') "
            "WHERE id = (SELECT MAX(id) FROM job_logs)"
        )
    await db.conn.commit()


async def _count_logs(db: Database, job_id: str) -> int:
    cur = await db.conn.execute("SELECT COUNT(*) FROM job_logs WHERE job_id = ?", (job_id,))
    return (await cur.fetchone())[0]


def _space_cutoff(days_ago: int = 0) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


# --- job_logs sweep -----------------------------------------------------------


async def test_sweep_job_logs_sweeps_every_kind(db, queries):
    """D-S-11: the ``kind='batch'`` exemption is gone — service logs are swept too.

    The old scoping spared service/model build+deploy logs; once batch was
    removed that made ``[retention].job_log_days`` a default-on knob pruning
    exactly nothing while ``job_logs`` kept growing.
    """
    await _insert_job(db, "batchjob0001", kind="batch")
    await _insert_job(db, "svcjob000001", kind="service", service_name="svc")
    await _insert_job(db, "modeljob0001", kind="model", service_name="mdl")
    await _insert_log(db, "batchjob0001", aged=True)
    await _insert_log(db, "svcjob000001", aged=True)
    await _insert_log(db, "modeljob0001", aged=True)

    removed = await queries.sweep_job_logs(_space_cutoff())

    assert removed == 3
    assert await _count_logs(db, "batchjob0001") == 0
    assert await _count_logs(db, "svcjob000001") == 0
    assert await _count_logs(db, "modeljob0001") == 0


async def test_sweep_job_logs_service_age_cutoff(db, queries):
    """D-S-11 mandate: a *service* row's logs are pruned by age, not by kind.

    Seeds two ``kind='service'`` log lines either side of the 14-day cutoff —
    without this the re-scope is indistinguishable from the inert version it
    replaces (the old query would return 0 for both).
    """
    await _insert_job(db, "svcjob000001", kind="service", service_name="svc")
    await _insert_log(db, "svcjob000001", aged=False)  # fresh
    await _insert_log(db, "svcjob000001", aged=True)  # 30 days old

    removed = await queries.sweep_job_logs(_space_cutoff(days_ago=14))

    assert removed == 1
    assert await _count_logs(db, "svcjob000001") == 1  # the fresh line remains


async def test_sweep_job_logs_cutoff_format_pin(db, queries):
    """A space-format cutoff matches the ``datetime('now')`` timestamp column.

    A fresh log (timestamp = now) survives; an aged one (‑30 d) is pruned by the
    ‑14 d cutoff — proving the cutoff format lines up with the writer's format.
    """
    await _insert_job(db, "batchjob0001", kind="batch")
    await _insert_log(db, "batchjob0001", aged=False)  # fresh
    await _insert_log(db, "batchjob0001", aged=True)  # 30 days old

    removed = await queries.sweep_job_logs(_space_cutoff(days_ago=14))

    assert removed == 1
    assert await _count_logs(db, "batchjob0001") == 1  # the fresh line remains


# --- idx_job_logs_ts ----------------------------------------------------------


async def _index_names(db: Database) -> set[str]:
    cur = await db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    return {r[0] for r in await cur.fetchall()}


async def test_index_present_after_init_schema():
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    assert "idx_job_logs_ts" in await _index_names(db)
    await db.close()


async def test_index_created_on_preexisting_db():
    """A pre-existing job_logs table (no index) → init_schema adds the index.

    Mirrors ``tests/test_migrations.py`` — job_logs is never rebuilt, so the
    ``CREATE INDEX IF NOT EXISTS`` in ``_SCHEMA`` runs safely on an upgrade.
    """
    db = Database(":memory:")
    await db.connect()
    # Stand up a legacy job_logs table by hand, without the P14b index.
    await db.conn.execute(
        """CREATE TABLE job_logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               job_id TEXT,
               stream TEXT DEFAULT 'stdout',
               message TEXT,
               timestamp TEXT DEFAULT (datetime('now'))
           )"""
    )
    await db.conn.commit()
    assert "idx_job_logs_ts" not in await _index_names(db)

    await db.init_schema()

    assert "idx_job_logs_ts" in await _index_names(db)
    await db.close()


# --- archive-first audit prune ------------------------------------------------


async def _insert_audit(queries: Queries, action: str) -> None:
    await queries.insert_audit_log(action=action, result="ok", principal_id="system")


async def _age_all_audit(db: Database) -> None:
    await db.conn.execute("UPDATE audit_log SET ts = datetime('now', '-30 days')")
    await db.conn.commit()


async def _audit_count(db: Database) -> int:
    return (await (await db.conn.execute("SELECT COUNT(*) FROM audit_log")).fetchone())[0]


async def test_archive_and_prune_happy_path(db, queries, tmp_path):
    for i in range(5):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)
    archive_dir = tmp_path / "archive"
    cutoff = _space_cutoff()

    result = await archive_and_prune_audit(queries, archive_dir, cutoff)

    assert result.archived == 5
    assert result.deleted == 5
    assert result.basename is not None and result.basename.endswith(".jsonl.gz")
    assert await _audit_count(db) == 0  # all exported rows pruned

    path = archive_dir / result.basename
    assert path.exists()
    # Immutable-file + dir perms.
    assert oct(os.stat(path).st_mode & 0o777) == oct(0o600)
    assert oct(os.stat(archive_dir).st_mode & 0o777) == oct(0o700)
    # sha256 matches the closed file.
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.sha256 == h
    # Content: 5 JSON lines, id-ascending, each a real audit row.
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    assert len(lines) == 5
    assert [ln["action"] for ln in lines] == [f"act.{i}" for i in range(5)]


async def test_archive_no_eligible_rows_creates_no_file(db, queries, tmp_path):
    """Only fresh (non-aged) rows → nothing archived, no file written."""
    await _insert_audit(queries, "fresh.row")  # ts = now, not below cutoff
    archive_dir = tmp_path / "archive"

    result = await archive_and_prune_audit(queries, archive_dir, _space_cutoff())

    assert result == retention_module.AuditArchiveResult(0, 0, None, None)
    assert not archive_dir.exists() or not list(archive_dir.glob("*.jsonl.gz"))
    assert await _audit_count(db) == 1


async def test_archive_export_failure_deletes_nothing(db, queries, tmp_path, monkeypatch):
    """D-H: any export failure ⇒ ZERO deletes and the partial file is abandoned."""
    for i in range(4):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)
    archive_dir = tmp_path / "archive"

    def _boom(_fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(retention_module.os, "fsync", _boom)

    with pytest.raises(OSError):
        await archive_and_prune_audit(queries, archive_dir, _space_cutoff())

    assert await _audit_count(db) == 4  # every row survives
    assert not list(archive_dir.glob("*.jsonl.gz"))  # partial file unlinked


async def test_archive_watermark_excludes_rows_inserted_during_archiving(
    db, queries, tmp_path, monkeypatch
):
    """A row inserted mid-archiving (id > snapshot max_id) survives the prune.

    Even with an old ``ts`` (below the cutoff), an id above the pre-stream
    snapshot is never fetched, archived, or deleted — the watermark, not the ts
    cutoff alone, bounds the delete.
    """
    for i in range(3):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)

    original = queries.fetch_audit_for_archive
    inserted = {"done": False}

    async def _wrapper(*args, **kwargs):
        if not inserted["done"]:
            inserted["done"] = True
            # A late arrival with an OLD ts but a fresh (higher) id.
            await db.conn.execute(
                "INSERT INTO audit_log (action, result, principal_id, ts) "
                "VALUES ('late.row', 'ok', 'system', datetime('now', '-30 days'))"
            )
            await db.conn.commit()
        return await original(*args, **kwargs)

    monkeypatch.setattr(queries, "fetch_audit_for_archive", _wrapper)

    result = await archive_and_prune_audit(queries, tmp_path / "archive", _space_cutoff())

    assert result.archived == 3  # only the snapshot rows
    assert result.deleted == 3
    rows = await (await db.conn.execute("SELECT action FROM audit_log")).fetchall()
    assert [r["action"] for r in rows] == ["late.row"]  # the late arrival survives


async def test_archive_dir_reused_uses_distinct_files(db, queries, tmp_path):
    """A second sweep into the same dir never appends to the first file."""
    archive_dir = tmp_path / "archive"
    await _insert_audit(queries, "first.batch")
    await _age_all_audit(db)
    first = await archive_and_prune_audit(queries, archive_dir, _space_cutoff())

    await _insert_audit(queries, "second.batch")
    await _age_all_audit(db)
    second = await archive_and_prune_audit(queries, archive_dir, _space_cutoff())

    assert first.basename != second.basename
    assert {p.name for p in archive_dir.glob("*.jsonl.gz")} == {first.basename, second.basename}


def _is_gzip_write(target: object) -> bool:
    """True for a bound ``gzip.GzipFile.write``, whatever the file object."""
    return (
        isinstance(getattr(target, "__self__", None), gzip.GzipFile)
        and getattr(target, "__name__", None) == "write"
    )


async def test_archive_offloads_blocking_io_to_threads(db, queries, tmp_path, monkeypatch):
    """F6: every blocking step routes through ``asyncio.to_thread``.

    A timing assertion on loop responsiveness is flaky by nature, so pin the
    mechanics instead: record the offload targets and assert the gzip writes
    (one per CHUNK, not per row), the flush+fsync, the dir fsync and the file
    sha256 all leave the event loop — while the pass still produces a correct
    archive (proving the recorder delegates for real).
    """
    for i in range(5):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)
    archive_dir = tmp_path / "archive"

    real_to_thread = asyncio.to_thread
    targets: list[object] = []

    async def _recording_to_thread(func, /, *args, **kwargs):
        targets.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(retention_module.asyncio, "to_thread", _recording_to_thread)

    result = await archive_and_prune_audit(queries, archive_dir, _space_cutoff(), chunk_size=2)

    # 5 rows / chunk_size=2 ⇒ 3 chunks ⇒ 3 write hops (NOT 5, one per row).
    assert sum(1 for t in targets if _is_gzip_write(t)) == 3
    assert targets.count(retention_module._flush_and_fsync) == 1
    assert targets.count(retention_module._fsync_dir) == 1
    assert targets.count(retention_module._sha256_file) == 1

    # The offloaded work really happened: a correct, complete, hashed archive.
    assert result.archived == 5
    assert result.deleted == 5
    assert await _audit_count(db) == 0
    path = archive_dir / result.basename
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh if ln.strip()]
    assert [ln["action"] for ln in lines] == [f"act.{i}" for i in range(5)]


async def test_archive_offloaded_write_failure_deletes_nothing(db, queries, tmp_path, monkeypatch):
    """D-H holds ACROSS the thread boundary: a failure inside a to_thread write
    re-raises at the ``await`` (inside ``except BaseException``) ⇒ zero deletes,
    partial file unlinked."""
    for i in range(4):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)
    archive_dir = tmp_path / "archive"

    real_to_thread = asyncio.to_thread

    async def _boom_on_write(func, /, *args, **kwargs):
        if _is_gzip_write(func):
            raise OSError("disk full")
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(retention_module.asyncio, "to_thread", _boom_on_write)

    with pytest.raises(OSError):
        await archive_and_prune_audit(queries, archive_dir, _space_cutoff())

    assert await _audit_count(db) == 4  # every row survives
    assert not list(archive_dir.glob("*.jsonl.gz"))  # partial file unlinked


# --- sweep loop wiring (server module) ----------------------------------------


def _settings_for_test(tmp_path, monkeypatch):
    from nerdit.config.settings import NerditSettings

    settings = NerditSettings(
        data_dir=str(tmp_path / "data"),
        daemon={
            "host": "127.0.0.1",
            "port": 9321,
            "auth_token": "test-token",
            "upload_dir": str(tmp_path / "uploads"),
        },
    )
    monkeypatch.setattr(server_module, "load_settings", lambda: settings)
    return settings


async def test_lifespan_starts_and_cancels_retention_task(tmp_path, monkeypatch):
    """The retention sweeper is started, stashed, and cancelled before db.close."""
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: GpuDiscoveryResult(backend=_StubBackend(), gpus=[]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self, *a, **k):
            raise RuntimeError("no docker")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.runtime, StubRuntime)
        task = app.state.retention_task
        assert task is not None
        assert not task.done()

    assert task.done()  # cancelled + reaped before db.close


class _StubBackend:
    name = "stub"

    def collect_metrics(self):
        return {}


# --- _resolve_archive_dir guard -----------------------------------------------


def test_resolve_archive_dir_default(tmp_path):
    from nerdit.config.settings import RetentionSettings

    data_dir = tmp_path / "data"
    resolved = server_module._resolve_archive_dir(RetentionSettings(), data_dir)
    assert resolved == data_dir / "archive"


def test_resolve_archive_dir_rejects_under_services(tmp_path):
    from nerdit.config.settings import RetentionSettings

    data_dir = tmp_path / "data"
    bad = str(data_dir / "services" / "evil")
    resolved = server_module._resolve_archive_dir(
        RetentionSettings(audit_archive_dir=bad), data_dir
    )
    assert resolved is None  # disabled rather than let a delete destroy evidence


def test_resolve_archive_dir_rejects_under_secrets(tmp_path):
    from nerdit.config.settings import RetentionSettings

    data_dir = tmp_path / "data"
    bad = str(data_dir / "secrets")
    resolved = server_module._resolve_archive_dir(
        RetentionSettings(audit_archive_dir=bad), data_dir
    )
    assert resolved is None


def test_prune_backups_keeps_last_n(tmp_path):
    backups = tmp_path / "backups"
    backups.mkdir()
    import time

    for i in range(5):
        p = backups / f"nerdit-backup-{i}.tar.gz"
        p.write_bytes(b"x")
        os.utime(p, (time.time() + i, time.time() + i))  # ascending mtime
    # An unrelated file must never be touched.
    (backups / "keepme.txt").write_bytes(b"y")

    removed = server_module._prune_backups(tmp_path, keep_last=2)

    assert removed == 3
    survivors = {p.name for p in backups.glob("nerdit-backup-*.tar.gz")}
    assert survivors == {"nerdit-backup-3.tar.gz", "nerdit-backup-4.tar.gz"}
    assert (backups / "keepme.txt").exists()


def test_prune_backups_absent_dir_is_noop(tmp_path):
    assert server_module._prune_backups(tmp_path, keep_last=2) == 0


async def test_real_backup_tar_pruned_by_glob(db, tmp_path):
    """D1 naming contract: real ``create_backup`` tars are matched by the
    ``_prune_backups`` glob — the integration pin the plan declared load-bearing."""
    import time

    from nerdit.core.backup import create_backup
    from nerdit.core.secrets import SecretManager

    sm = SecretManager(tmp_path / "secrets")
    sm.set("app", {"K": "v"})
    first = await create_backup(db=db, secret_manager=sm, data_dir=tmp_path)
    second = await create_backup(db=db, secret_manager=sm, data_dir=tmp_path)
    assert first.basename != second.basename
    # Deterministic mtime ordering (same-second creation otherwise ties).
    os.utime(tmp_path / "backups" / first.basename, (time.time(), time.time()))
    os.utime(tmp_path / "backups" / second.basename, (time.time() + 1, time.time() + 1))

    removed = server_module._prune_backups(tmp_path, keep_last=1)
    assert removed == 1
    survivors = {p.name for p in (tmp_path / "backups").glob("nerdit-backup-*.tar.gz")}
    assert survivors == {second.basename}


async def test_anchor_callback_runs_before_any_delete(db, queries, tmp_path):
    """D-H anchor ordering (Codex P1): ``on_archived`` fires after fsync+sha but
    BEFORE the first delete, and a callback failure deletes zero rows — so a
    crash between archive and prune can never leave archived rows gone with no
    sha256 anchor surviving in the DB."""
    for i in range(3):
        await _insert_audit(queries, f"anchor.{i}")
    await _age_all_audit(db)
    archive_dir = tmp_path / "archive"
    cutoff = _space_cutoff()

    seen: dict = {}

    async def anchor(meta):
        # At callback time: file fsynced + hashed, NOTHING deleted yet.
        seen["meta"] = meta
        seen["rows_at_anchor"] = await _audit_count(db)

    result = await archive_and_prune_audit(queries, archive_dir, cutoff, on_archived=anchor)
    assert seen["rows_at_anchor"] == 3  # anchor ran pre-delete
    assert seen["meta"].archived == 3 and seen["meta"].deleted == 0
    assert seen["meta"].sha256 == result.sha256 and seen["meta"].basename == result.basename
    assert result.deleted == 3 and await _audit_count(db) == 0

    # Callback failure ⇒ zero deletes, fsynced file kept (re-archives next sweep).
    for i in range(2):
        await _insert_audit(queries, f"anchor2.{i}")
    await _age_all_audit(db)

    async def failing_anchor(meta):
        raise RuntimeError("anchor insert failed")

    with pytest.raises(RuntimeError):
        await archive_and_prune_audit(
            queries, archive_dir, _space_cutoff(), on_archived=failing_anchor
        )
    assert await _audit_count(db) == 2  # nothing deleted


async def _sweep_rows(db: Database) -> list[dict]:
    """The ``retention.sweep`` audit rows this pass wrote, id-ascending.

    Anchor rows precede their summary row (the anchor is inserted before the
    deletes), so ordering by id keeps [anchor, summary] pairing intact.
    """
    cur = await db.conn.execute(
        "SELECT params_redacted FROM audit_log WHERE action = 'retention.sweep' ORDER BY id"
    )
    return [json.loads(r[0]) for r in await cur.fetchall()]


async def test_retention_sweep_summary_written_when_archive_phase_raises(
    db, queries, tmp_path, monkeypatch
):
    """Finding 4: an archive-phase failure must not skip auditing the phase-1
    deletions that already happened (Invariant #3)."""
    from nerdit.config.settings import RetentionSettings

    await _insert_job(db, "batchjob0001", kind="batch")
    await _insert_log(db, "batchjob0001", aged=True)

    async def _boom(*_a, **_k):
        raise RuntimeError("archive exploded")

    monkeypatch.setattr(retention_module, "archive_and_prune_audit", _boom)

    retention = RetentionSettings(job_log_days=14, audit_days=14, backup_keep_last=0)
    await server_module._run_retention_sweep(
        queries, retention, tmp_path / "data", tmp_path / "archive"
    )

    assert await _count_logs(db, "batchjob0001") == 0  # phase 1 ran before the failure
    rows = await _sweep_rows(db)
    assert len(rows) == 1  # the plain summary row survived the archive failure
    assert rows[0]["job_logs"] == 1
    assert "audit_deleted" not in rows[0]  # nothing was pruned from audit_log


async def test_retention_sweep_archiving_pass_writes_two_rows(db, queries, tmp_path):
    """Finding 5: a successful archiving pass writes an anchor row (pre-delete:
    sha256 + audit_archived, NO audit_deleted) AND a summary row (post-delete:
    audit_deleted == archived, with the archive basename)."""
    from nerdit.config.settings import RetentionSettings

    for i in range(4):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)

    retention = RetentionSettings(job_log_days=0, audit_days=14, backup_keep_last=0)
    await server_module._run_retention_sweep(
        queries, retention, tmp_path / "data", tmp_path / "archive"
    )

    rows = await _sweep_rows(db)
    assert len(rows) == 2
    anchor, summary = rows
    # Anchor row = pre-delete tamper anchor.
    assert anchor["audit_archived"] == 4
    assert anchor["sha256"]
    assert anchor["archive"].endswith(".jsonl.gz")
    assert "audit_deleted" not in anchor
    # Summary row = post-delete completion record.
    assert summary["audit_archived"] == 4
    assert summary["audit_deleted"] == 4
    assert summary["archive"] == anchor["archive"]


async def test_retention_sweep_delete_failure_leaves_anchor_without_summary(
    db, queries, tmp_path, monkeypatch
):
    """Finding 5(c): a delete failure mid-prune leaves the anchor row with NO
    matching summary row — the durable aborted-prune signature — and the audit
    rows survive."""
    from nerdit.config.settings import RetentionSettings

    for i in range(3):
        await _insert_audit(queries, f"act.{i}")
    await _age_all_audit(db)

    async def _boom(*_a, **_k):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(queries, "delete_audit_range", _boom)

    retention = RetentionSettings(job_log_days=0, audit_days=14, backup_keep_last=0)
    await server_module._run_retention_sweep(
        queries, retention, tmp_path / "data", tmp_path / "archive"
    )

    rows = await _sweep_rows(db)
    assert len(rows) == 1  # anchor only — no summary row ⇒ aborted prune
    assert rows[0]["audit_archived"] == 3
    assert rows[0]["sha256"]
    assert "audit_deleted" not in rows[0]
    # The three aged rows survive — the delete never committed.
    survivors = await (
        await db.conn.execute("SELECT action FROM audit_log WHERE action LIKE 'act.%'")
    ).fetchall()
    assert len(survivors) == 3


# --- P24a durable event feed (D-P24-2: plain keep-last-N, NEVER archive-first) ---


async def _seed_events(db: Database, count: int) -> None:
    """Insert ``count`` feed rows straight through SQL (fast, id-ascending)."""
    await db.conn.executemany(
        "INSERT INTO events (type, service_name) VALUES ('service.stopped', ?)",
        [(f"svc-{i}",) for i in range(count)],
    )
    await db.conn.commit()


async def _event_ids(db: Database) -> list[int]:
    cur = await db.conn.execute("SELECT id FROM events ORDER BY id")
    return [r[0] for r in await cur.fetchall()]


async def test_sweep_events_keeps_the_newest_n(db, queries):
    """Keep-last-N keeps the *newest* rows — the id watermark, not the oldest."""
    await _seed_events(db, 10)
    all_ids = await _event_ids(db)

    removed = await queries.sweep_events(4)

    assert removed == 6
    assert await _event_ids(db) == all_ids[-4:]


async def test_sweep_events_is_chunked(db, queries):
    """One call deletes at most ``_RETENTION_SWEEP_CHUNK`` rows, so the write
    lock is released between chunks; the caller loops via ``_chunked_delete``."""
    from nerdit.daemon.sweeps import _chunked_delete
    from nerdit.db.queries._base import _RETENTION_SWEEP_CHUNK

    await _seed_events(db, _RETENTION_SWEEP_CHUNK + 210)

    first = await queries.sweep_events(10)
    assert first == _RETENTION_SWEEP_CHUNK  # capped, not the full 700

    total = first + await _chunked_delete(lambda: queries.sweep_events(10))
    assert total == _RETENTION_SWEEP_CHUNK + 200
    assert len(await _event_ids(db)) == 10


async def test_sweep_events_zero_prunes_nothing(db, queries):
    """0 = never prune (the ``[retention]`` convention), not "delete everything"."""
    await _seed_events(db, 5)
    assert await queries.sweep_events(0) == 0
    assert len(await _event_ids(db)) == 5


async def test_sweep_events_under_the_bound_is_a_noop(db, queries):
    await _seed_events(db, 3)
    assert await queries.sweep_events(10) == 0
    assert len(await _event_ids(db)) == 3


async def test_retention_sweep_reports_events_deleted_and_writes_no_archive(db, queries, tmp_path):
    """The count lands on the EXISTING ``retention.sweep`` row as
    ``events_deleted`` — the slot the old ``P16`` comment reserved — and the
    phase writes **no archive file and no anchor row** (D-P24-2). Guard against
    someone "fixing" this into the archive-first D-H sweep audit_log gets."""
    from nerdit.config.settings import RetentionSettings

    await _seed_events(db, 12)
    archive_dir = tmp_path / "archive"

    retention = RetentionSettings(
        job_log_days=0, audit_days=0, backup_keep_last=0, events_keep_last=5
    )
    await server_module._run_retention_sweep(queries, retention, tmp_path / "data", archive_dir)

    rows = await _sweep_rows(db)
    assert len(rows) == 1  # one summary row, NO anchor row
    assert rows[0]["events_deleted"] == 7
    assert "archive" not in rows[0] and "sha256" not in rows[0]
    assert len(await _event_ids(db)) == 5
    assert not archive_dir.exists() or list(archive_dir.iterdir()) == []


async def test_retention_sweep_events_phase_default_is_a_real_bound():
    """D-P24-2: a REAL default bound, unlike ``backup_keep_last``'s dark 0."""
    from nerdit.config.settings import RetentionSettings

    assert RetentionSettings().events_keep_last == 10000


async def test_retention_sweep_events_phase_failure_does_not_block_the_row(
    db, queries, tmp_path, monkeypatch
):
    """Isolated like the backup-keep phases: a failure is logged, the sweep runs on."""
    from nerdit.config.settings import RetentionSettings

    await _insert_job(db, "batchjob0001", kind="batch")
    await _insert_log(db, "batchjob0001", aged=True)
    # Seeded so the watermark actually resolves and the exploding chunk delete
    # is genuinely reached (the phase resolves the bound before its first chunk).
    await _seed_events(db, 3)

    async def _boom(*_a, **_k):
        raise RuntimeError("events sweep exploded")

    monkeypatch.setattr(queries, "sweep_events", _boom)

    retention = RetentionSettings(
        job_log_days=14, audit_days=0, backup_keep_last=0, events_keep_last=1
    )
    await server_module._run_retention_sweep(queries, retention, tmp_path / "data", None)

    rows = await _sweep_rows(db)
    assert len(rows) == 1
    assert rows[0]["job_logs"] == 1
    assert "events_deleted" not in rows[0]


async def test_delete_service_checked_atomic_guard(db, queries):
    """Codex P2: the dependent re-check runs inside the serialized delete —
    a non-empty checker result rolls back (row survives); empty deletes
    child-first; the checker sees the list_workload_configs row shape."""
    from nerdit.db.models import Job, JobKind, JobStatus, LogStream

    model = Job(
        id="mdl-x",
        name="llama",
        kind=JobKind.model,
        image="ollama/ollama",
        command="serve",
        status=JobStatus.running,
        service_name="llama",
        config='{"model": "llama3.1:8b"}',
    )
    await queries.create_job(model)
    await queries.append_log("mdl-x", "started", LogStream.system)

    seen_rows: list = []

    def blocking_checker(rows):
        seen_rows.extend(rows)
        return [{"service": "app", "id": "svc-app", "binding": "default"}]

    deps = await queries.delete_service_checked("mdl-x", blocking_checker)
    assert deps == [{"service": "app", "id": "svc-app", "binding": "default"}]
    assert await queries.get_job("mdl-x") is not None  # rolled back, row survives
    assert any(r["id"] == "mdl-x" and r["config"].get("model") == "llama3.1:8b" for r in seen_rows)

    deps = await queries.delete_service_checked("mdl-x", lambda rows: [])
    assert deps == []
    assert await queries.get_job("mdl-x") is None  # deleted (children first — no FK error)

    # Absent row: still a no-op, but reported as ``None`` — the commit-point
    # discriminator (review round-2, Codex 3803596881). ``[]`` would be
    # indistinguishable from "this request deleted the row", which let a stale
    # delete run its name-keyed purges against a same-name replacement.
    assert await queries.delete_service_checked("mdl-x", None) is None
    assert await queries.delete_service_checked("absent-id", None) is None


async def test_delete_service_checked_releases_endpoint_in_txn(db, queries):
    """FK regression (review-fix verifier): service_endpoints.job_id references
    jobs(id), so the endpoint row must be deleted INSIDE the atomic delete's
    transaction — and must SURVIVE a blocked re-check (stable-port contract)."""
    from nerdit.db.models import Job, JobKind, JobStatus

    model = Job(
        id="mdl-ep",
        name="llama",
        kind=JobKind.model,
        image="ollama/ollama",
        command="serve",
        status=JobStatus.running,
        service_name="llama-ep",
        config='{"model": "llama3.1:8b"}',
    )
    await queries.create_job(model)
    ep = await queries.acquire_service_port("llama-ep", "mdl-ep", 11434, (9400, 9499))
    assert ep is not None

    # Blocked re-check → rollback: row AND endpoint both survive (port kept).
    deps = await queries.delete_service_checked(
        "mdl-ep", lambda rows: [{"service": "app", "id": "x", "binding": "default"}]
    )
    assert deps and await queries.get_job("mdl-ep") is not None
    assert await queries.get_service_endpoint("llama-ep") is not None

    # Clear re-check → the delete succeeds WITH the endpoint FK in place
    # (pre-fix: IntegrityError('FOREIGN KEY constraint failed') → 500).
    deps = await queries.delete_service_checked("mdl-ep", lambda rows: [])
    assert deps == []
    assert await queries.get_job("mdl-ep") is None
    assert await queries.get_service_endpoint("llama-ep") is None


# --- P29 / D-P29-8: phase 3d — orphan agent workspaces -----------------------


def _seed_ws(data_dir, name: str, *, age_days: float | None, payload: bytes = b"x" * 32) -> None:  # noqa: ANN001
    """Write a workspace on disk; ``age_days=None`` omits ``meta.json`` entirely.

    When a meta is written, its ``last_written_at`` is backdated by ``age_days``
    — the sweep's primary age source. Omitting it exercises the dir-mtime
    fallback (R18), so the caller backdates the dir itself.
    """
    root = data_dir / "workspaces" / name
    (root / "tree").mkdir(parents=True)
    (root / "tree" / "main.py").write_bytes(payload)
    if age_days is not None:
        written = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
        (root / "meta.json").write_text(
            json.dumps({"owner_token_id": "tok-a", "last_written_at": written}),
            encoding="utf-8",
        )


async def _orphan_rows(db: Database) -> list[dict]:
    cur = await db.conn.execute(
        "SELECT params_redacted, principal_id, principal_role FROM audit_log "
        "WHERE action = 'workspace.purge_orphan' ORDER BY id"
    )
    return [
        {"params": json.loads(r[0]), "principal_id": r[1], "principal_role": r[2]}
        for r in await cur.fetchall()
    ]


def _retention(**over):  # noqa: ANN001, ANN201
    from nerdit.config.settings import RetentionSettings

    fields = {"job_log_days": 0, "audit_days": 0, "backup_keep_last": 0, "events_keep_last": 0}
    fields.update(over)
    return RetentionSettings(**fields)


async def test_workspace_orphan_sweep_removes_old_orphan(db, queries, tmp_path):
    """An aged workspace with no service row is removed and audited names+bytes only."""
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ghost", age_days=40, payload=b"z" * 64)

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert not (data_dir / "workspaces" / "ghost").exists()
    rows = await _orphan_rows(db)
    assert len(rows) == 1
    assert rows[0]["principal_id"] == "system"
    assert rows[0]["principal_role"] == "system"
    # Names + bytes, nothing else: no paths, no file names, no content.
    assert set(rows[0]["params"]) == {"names", "total_bytes"}
    assert rows[0]["params"]["names"] == ["ghost"]
    assert rows[0]["params"]["total_bytes"] >= 64
    # And the count rides the sweep summary row.
    assert (await _sweep_rows(db))[-1]["workspace_orphans_removed"] == 1


async def test_live_service_workspace_never_swept_regardless_of_age(db, queries, tmp_path):
    """A workspace whose name matches ANY row is live — a *stopped* one included."""
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "running-app", age_days=400)
    _seed_ws(data_dir, "stopped-app", age_days=400)
    await _insert_job(db, "svc-run", kind="service", service_name="running-app")
    await _insert_job(db, "svc-stop", kind="service", service_name="stopped-app")
    await db.conn.execute("UPDATE jobs SET desired_state = 'stopped' WHERE id = 'svc-stop'")
    await db.conn.commit()

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert (data_dir / "workspaces" / "running-app" / "tree" / "main.py").is_file()
    assert (data_dir / "workspaces" / "stopped-app" / "tree" / "main.py").is_file()
    assert await _orphan_rows(db) == []


async def test_young_orphan_kept(db, queries, tmp_path):
    """Falsifier for the age gate: an orphan younger than N days survives."""
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "fresh", age_days=29)
    _seed_ws(data_dir, "stale", age_days=31)

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert (data_dir / "workspaces" / "fresh" / "tree" / "main.py").is_file()
    assert not (data_dir / "workspaces" / "stale").exists()
    assert (await _orphan_rows(db))[0]["params"]["names"] == ["stale"]


async def test_zero_days_disables_sweep(db, queries, tmp_path):
    """``workspace_orphan_days = 0`` = never: even an ancient orphan is untouched."""
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ancient", age_days=9999)

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=0), data_dir, None
    )

    assert (data_dir / "workspaces" / "ancient" / "tree" / "main.py").is_file()
    assert await _orphan_rows(db) == []


async def test_missing_meta_falls_back_to_dir_mtime(db, queries, tmp_path):
    """R18: no/unparseable ``meta.json`` ⇒ the dir's mtime decides the age.

    Both ways: a backdated dir is swept, a fresh one is kept — so a corrupted
    sidecar can neither exempt a workspace forever nor age a fresh one out.

    Also the reconciliation pin for the review round-1 fail-closed change:
    ``read_meta`` now RAISES on a corrupt sidecar, and the sweep's recorded
    mtime fallback survives it because tolerance lives at this one call site
    (``_workspace_written_at``'s ``except WorkspaceError``) instead of inside
    ``read_meta`` where it was an authz hole.
    """
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "no-meta-old", age_days=None)
    _seed_ws(data_dir, "no-meta-new", age_days=None)
    # A sidecar that parses as nothing usable reads exactly like an absent one.
    _seed_ws(data_dir, "bad-meta-old", age_days=None)
    (data_dir / "workspaces" / "bad-meta-old" / "meta.json").write_text("{not json", "utf-8")
    _seed_ws(data_dir, "bad-meta-new", age_days=None)
    (data_dir / "workspaces" / "bad-meta-new" / "meta.json").write_text("{not json", "utf-8")
    aged = (datetime.now(UTC) - timedelta(days=45)).timestamp()
    for name in ("no-meta-old", "bad-meta-old"):
        os.utime(data_dir / "workspaces" / name, (aged, aged))

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert not (data_dir / "workspaces" / "no-meta-old").exists()
    assert not (data_dir / "workspaces" / "bad-meta-old").exists()
    assert (data_dir / "workspaces" / "no-meta-new" / "tree" / "main.py").is_file()
    # A corrupt sidecar on a FRESH dir is kept: the raising read must not age it out.
    assert (data_dir / "workspaces" / "bad-meta-new" / "tree" / "main.py").is_file()
    assert (await _orphan_rows(db))[0]["params"]["names"] == ["bad-meta-old", "no-meta-old"]


async def test_absent_workspaces_root_is_a_noop(db, queries, tmp_path):
    """A daemon that never took a workspace write has no ``workspaces/`` dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert await _orphan_rows(db) == []
    assert await _sweep_rows(db) == []  # nothing pruned ⇒ no summary row at all


async def test_orphan_sweep_removal_respects_lock(db, queries, tmp_path):
    """(review B4) The sweep's ``rmtree`` must serialize against a live writer.

    Same hazard as the DELETE-time purge, one loop away: today the removal runs
    in a worker thread with no lock at all, so an aged orphan can be deleted out
    from under a ``write_files`` apply pass that is mid-batch.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ghost", age_days=40)
    root = data_dir / "workspaces" / "ghost"

    lock = core_workspaces.workspace_lock("ghost")
    async with lock:
        task = asyncio.create_task(
            server_module._run_retention_sweep(
                queries, _retention(workspace_orphan_days=30), data_dir, None
            )
        )
        await asyncio.sleep(0.1)
        assert root.exists(), "the sweep removed the workspace while a writer held the lock"
        assert not task.done()

    await asyncio.wait_for(task, 5)
    assert not root.exists()
    assert (await _orphan_rows(db))[0]["params"]["names"] == ["ghost"]
    core_workspaces._WORKSPACE_LOCKS.clear()


async def test_orphan_sweep_recheck_skips_new_row(db, queries, tmp_path, monkeypatch):
    """(review B5) A service created DURING the scan keeps its workspace.

    The live-name set is a prefilter read once; the authoritative check is a
    fresh row read per removal (the P14b GC precedent).
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ghost", age_days=40)
    calls = {"n": 0}

    async def _rows_appearing_mid_sweep():
        calls["n"] += 1
        # The prefilter sees nothing; by the time the removal loop re-reads, the
        # row exists.
        return [] if calls["n"] == 1 else [{"service_name": "ghost"}]

    monkeypatch.setattr(queries, "list_workload_configs", _rows_appearing_mid_sweep)
    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert calls["n"] >= 2, "the removal loop must re-read the rows"
    assert (data_dir / "workspaces" / "ghost" / "tree" / "main.py").is_file()
    assert await _orphan_rows(db) == []


async def test_orphan_sweep_recheck_skips_rewritten(db, queries, tmp_path, monkeypatch):
    """A workspace rewritten DURING the scan is no longer aged out."""
    core_workspaces._WORKSPACE_LOCKS.clear()
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ghost", age_days=40)
    calls = {"n": 0}

    async def _rewrite_mid_sweep():
        calls["n"] += 1
        if calls["n"] > 1:
            (data_dir / "workspaces" / "ghost" / "meta.json").write_text(
                json.dumps(
                    {"owner_token_id": "tok-a", "last_written_at": datetime.now(UTC).isoformat()}
                ),
                encoding="utf-8",
            )
        return []

    monkeypatch.setattr(queries, "list_workload_configs", _rewrite_mid_sweep)
    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert (data_dir / "workspaces" / "ghost" / "tree" / "main.py").is_file()
    assert await _orphan_rows(db) == []


async def test_illegal_name_dirs_are_still_removed_in_thread(db, queries, tmp_path):
    """A dir no writer could ever have created holds no lock — it is reclaimed
    in the scan thread, beside the loop-side removals, in ONE audit row."""
    core_workspaces._WORKSPACE_LOCKS.clear()
    data_dir = tmp_path / "data"
    _seed_ws(data_dir, "ghost", age_days=40)
    junk = data_dir / "workspaces" / "Not_A_Label"
    junk.mkdir(parents=True)
    (junk / "blob").write_bytes(b"x" * 16)
    aged = (datetime.now(UTC) - timedelta(days=45)).timestamp()
    os.utime(junk, (aged, aged))

    await server_module._run_retention_sweep(
        queries, _retention(workspace_orphan_days=30), data_dir, None
    )

    assert not junk.exists()
    assert not (data_dir / "workspaces" / "ghost").exists()
    rows = await _orphan_rows(db)
    assert len(rows) == 1
    assert rows[0]["params"]["names"] == ["Not_A_Label", "ghost"]
