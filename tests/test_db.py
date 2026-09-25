"""Tests for the database layer — CRUD operations with in-memory SQLite."""

import aiosqlite
import pytest

from nerdit.db.database import Database
from nerdit.db.models import Gpu, GpuStatus, GpuVendor, Job, JobStatus, LogStream
from nerdit.db.queries import Queries


async def _idle_gpus(queries) -> list[Gpu]:
    """Schedulable idle devices — the local stand-in for the removed ``get_idle_gpus``."""
    return [g for g in await queries.list_gpus() if g.status is GpuStatus.idle and g.schedulable]


@pytest.mark.asyncio
async def test_reconcile_and_list_gpus(queries):
    gpu = Gpu(id="GPU-TEST-001", name="Test GPU", memory_mb=8192, compute_cap="8.6")
    await queries.reconcile_gpus([gpu])
    gpus = await queries.list_gpus()
    assert len(gpus) == 1
    assert gpus[0].id == "GPU-TEST-001"
    assert gpus[0].name == "Test GPU"


@pytest.mark.asyncio
async def test_reconcile_gpus_updates_existing(queries):
    gpu = Gpu(id="GPU-TEST-001", name="Old Name", memory_mb=8192)
    await queries.reconcile_gpus([gpu])
    gpu.name = "New Name"
    await queries.reconcile_gpus([gpu])
    gpus = await queries.list_gpus()
    assert len(gpus) == 1
    assert gpus[0].name == "New Name"


@pytest.mark.asyncio
async def test_idle_gpus_exclude_allocated_devices(queries, sample_gpus):
    assert len(await _idle_gpus(queries)) == 2

    await queries.create_job(Job(id="alloc-1", script_path="/tmp/t.py"))
    await queries.allocate_gpus("alloc-1", ["GPU-0000-0001"])

    idle = await _idle_gpus(queries)
    assert [g.id for g in idle] == ["GPU-0000-0002"]


@pytest.mark.asyncio
async def test_idle_gpus_exclude_unschedulable_devices(queries):
    await queries.reconcile_gpus(
        [
            Gpu(
                id="gpu:amd:0",
                name="AMD Instinct MI300X",
                memory_mb=196608,
                vendor=GpuVendor.amd,
                device_index=0,
                runtime_id="0",
                schedulable=False,
            )
        ]
    )

    assert await _idle_gpus(queries) == []


@pytest.mark.asyncio
async def test_reconcile_gpus_marks_missing_devices_offline(queries):
    await queries.reconcile_gpus([Gpu(id="gpu:nvidia:0", name="H100", memory_mb=81920)])
    amd = Gpu(
        id="gpu:amd:0",
        name="MI300X",
        memory_mb=196608,
        vendor=GpuVendor.amd,
        schedulable=False,
    )

    await queries.reconcile_gpus([amd])
    gpus = {gpu.id: gpu for gpu in await queries.list_gpus()}

    assert gpus["gpu:nvidia:0"].status == GpuStatus.offline
    assert gpus["gpu:amd:0"].status == GpuStatus.idle


@pytest.mark.asyncio
async def test_reconcile_gpus_adopts_legacy_uuid_identity(queries):
    await queries.reconcile_gpus(
        [Gpu(id="GPU-legacy-uuid", name="Old Name", memory_mb=8192, device_index=None)]
    )
    discovered = Gpu(
        id="gpu:nvidia:0",
        name="Current Name",
        memory_mb=8192,
        device_index=0,
        runtime_id="GPU-legacy-uuid",
    )

    await queries.reconcile_gpus([discovered])
    gpus = await queries.list_gpus()

    assert [gpu.id for gpu in gpus] == ["gpu:nvidia:0"]
    assert gpus[0].runtime_id == "GPU-legacy-uuid"


@pytest.mark.asyncio
async def test_reconcile_gpus_adopts_multiple_legacy_devices_by_index(queries):
    await queries.reconcile_gpus(
        [
            Gpu(id="old-0", name="Old 0", memory_mb=8192),
            Gpu(id="old-1", name="Old 1", memory_mb=8192),
        ]
    )
    discovered = [
        Gpu(
            id=f"gpu:nvidia:{index}",
            name=f"Current {index}",
            memory_mb=8192,
            device_index=index,
            runtime_id=str(index),
        )
        for index in range(2)
    ]

    await queries.reconcile_gpus(discovered)

    assert [gpu.id for gpu in await queries.list_gpus()] == [
        "gpu:nvidia:0",
        "gpu:nvidia:1",
    ]


@pytest.mark.asyncio
async def test_create_and_get_job(queries):
    job = Job(id="test-job-001", script_path="/tmp/train.py", gpu_count=2)
    created = await queries.create_job(job)
    assert created.id == "test-job-001"

    fetched = await queries.get_job("test-job-001")
    assert fetched is not None
    assert fetched.script_path == "/tmp/train.py"
    assert fetched.gpu_count == 2
    assert fetched.status == JobStatus.pending


@pytest.mark.asyncio
async def test_get_job_not_found(queries):
    result = await queries.get_job("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_jobs_with_filter(queries):
    job1 = Job(id="job-1", script_path="/tmp/a.py", status=JobStatus.pending)
    job2 = Job(id="job-2", script_path="/tmp/b.py", status=JobStatus.running)
    await queries.create_job(job1)
    await queries.create_job(job2)

    all_jobs = await queries.list_jobs()
    assert len(all_jobs) == 2

    pending = await queries.list_jobs(status=JobStatus.pending)
    assert len(pending) == 1
    assert pending[0].id == "job-1"

    running = await queries.list_jobs(status=JobStatus.running)
    assert len(running) == 1
    assert running[0].id == "job-2"


@pytest.mark.asyncio
async def test_update_job_status(queries):
    job = Job(id="job-u", script_path="/tmp/test.py")
    await queries.create_job(job)

    await queries.update_job_status("job-u", JobStatus.running, container_id="ctr-123")
    updated = await queries.get_job("job-u")
    assert updated.status == JobStatus.running
    assert updated.container_id == "ctr-123"


@pytest.mark.asyncio
async def test_allocate_and_release_gpus(queries, sample_gpus):
    job = Job(id="alloc-job", script_path="/tmp/test.py", gpu_count=2)
    await queries.create_job(job)

    await queries.allocate_gpus("alloc-job", ["GPU-0000-0001", "GPU-0000-0002"])

    # GPUs should be busy
    assert await _idle_gpus(queries) == []

    # Check allocation
    gpu_ids = await queries.get_job_gpus("alloc-job")
    assert set(gpu_ids) == {"GPU-0000-0001", "GPU-0000-0002"}

    # Release
    await queries.release_gpus("alloc-job")
    assert len(await _idle_gpus(queries)) == 2
    gpu_ids = await queries.get_job_gpus("alloc-job")
    assert len(gpu_ids) == 0


@pytest.mark.asyncio
async def test_append_and_get_logs(queries):
    job = Job(id="log-job", script_path="/tmp/test.py")
    await queries.create_job(job)

    await queries.append_log("log-job", "hello world", LogStream.stdout)
    await queries.append_log("log-job", "error!", LogStream.stderr)
    await queries.append_log("log-job", "system msg", LogStream.system)

    logs = await queries.get_logs("log-job")
    assert len(logs) == 3
    assert logs[0].message == "hello world"
    assert logs[0].stream == LogStream.stdout
    assert logs[1].stream == LogStream.stderr
    assert logs[2].stream == LogStream.system


@pytest.mark.asyncio
async def test_get_logs_since_id(queries):
    job = Job(id="log-job-2", script_path="/tmp/test.py")
    await queries.create_job(job)

    await queries.append_log("log-job-2", "line 1")
    await queries.append_log("log-job-2", "line 2")
    await queries.append_log("log-job-2", "line 3")

    all_logs = await queries.get_logs("log-job-2")
    assert len(all_logs) == 3

    # Get only logs after the first one
    since = all_logs[0].id
    newer = await queries.get_logs("log-job-2", since_id=since)
    assert len(newer) == 2
    assert newer[0].message == "line 2"


@pytest.mark.asyncio
async def test_get_logs_tail_bounds_and_orders(queries):
    """L7: tail returns only the most recent N entries, oldest→newest."""
    await queries.create_job(Job(id="tail-job", script_path="/tmp/t.py"))
    for i in range(10):
        await queries.append_log("tail-job", f"line {i}")

    tail = await queries.get_logs("tail-job", tail=3)
    assert [e.message for e in tail] == ["line 7", "line 8", "line 9"]
    # tail larger than the set returns everything, still ascending.
    assert len(await queries.get_logs("tail-job", tail=100)) == 10
    # tail=0 / None falls back to the full ascending fetch.
    assert len(await queries.get_logs("tail-job")) == 10


@pytest.mark.asyncio
async def test_migrate_legacy_jobs_script_path_not_null(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                name TEXT,
                script_path TEXT NOT NULL,
                gpu_count INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                container_id TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                started_at TEXT,
                finished_at TEXT,
                exit_code INTEGER,
                retries INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                config TEXT
            );
            """
        )
        await conn.commit()

    db = Database(str(db_path))
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    created = await queries.create_job(Job(id="cmd-job", script_path=None, command="echo hi"))
    await db.close()
    assert created.script_path is None


@pytest.mark.asyncio
async def test_migrate_legacy_gpu_schema(tmp_path):
    db_path = tmp_path / "legacy-gpus.sqlite3"
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            """
            CREATE TABLE gpus (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                memory_mb INTEGER NOT NULL,
                compute_cap TEXT,
                status TEXT DEFAULT 'idle'
            );
            INSERT INTO gpus VALUES ('GPU-legacy', 'Legacy NVIDIA GPU', 8192, '8.6', 'idle');
            """
        )
        await conn.commit()

    db = Database(str(db_path))
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    gpu = (await queries.list_gpus())[0]

    assert gpu.vendor == GpuVendor.nvidia
    assert gpu.runtime_id == "GPU-legacy"
    assert gpu.schedulable is True
    await db.close()


@pytest.mark.asyncio
async def test_project_display_name_migration_preserves_namespace_and_identity(tmp_path):
    path = tmp_path / "old-projects.sqlite3"
    async with aiosqlite.connect(path) as conn:
        await conn.executescript("""
            CREATE TABLE projects (
                id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                submitted_by_token TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO projects (id, name, submitted_by_token)
            VALUES ('prj_old', 'original', 'owner');
        """)
        await conn.commit()
    db = Database(str(path))
    await db.connect()
    try:
        await db.init_schema()
        queries = Queries(db)
        original = await queries.get_project("prj_old")
        assert original.label == original.name == "original"
        await queries.rename_project(original.id, "new-label")
        await db.init_schema()
        renamed = await queries.get_project(original.id)
        assert renamed.name == "original"
        assert renamed.label == "new-label"
        assert renamed.submitted_by_token == "owner"
    finally:
        await db.close()
