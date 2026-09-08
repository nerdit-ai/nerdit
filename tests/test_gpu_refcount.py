"""GPU refcount model: derived idle/shared/busy status and the kind gate.

Covers P0 Step 2 — allocations carry an ``exclusive`` flag and GPU status is
*derived* from the live allocation set rather than written as a literal.
"""

from __future__ import annotations

import pytest

from nerdit.db.models import GpuStatus, Job, JobKind, JobStatus


async def _status(queries, gpu_id: str) -> GpuStatus:
    """Return the current status of a single GPU."""
    gpus = {gpu.id: gpu for gpu in await queries.list_gpus()}
    return gpus[gpu_id].status


async def _idle_ids(queries) -> set[str]:
    """Schedulable idle devices — the local stand-in for the removed ``get_idle_gpus``."""
    return {
        gpu.id
        for gpu in await queries.list_gpus()
        if gpu.status is GpuStatus.idle and gpu.schedulable
    }


@pytest.mark.asyncio
async def test_two_nonexclusive_allocations_share_one_gpu(queries, sample_gpus):
    """Two non-exclusive allocations on one GPU make it ``shared``, not idle."""
    await queries.create_job(Job(id="svc-a", script_path=None, command="a"))
    await queries.create_job(Job(id="svc-b", script_path=None, command="b"))

    await queries.allocate_gpus("svc-a", ["GPU-0000-0001"], exclusive=False)
    await queries.allocate_gpus("svc-b", ["GPU-0000-0001"], exclusive=False)

    assert await _status(queries, "GPU-0000-0001") == GpuStatus.shared
    # A shared GPU is not a first-fit (exclusive) candidate.
    assert "GPU-0000-0001" not in await _idle_ids(queries)


@pytest.mark.asyncio
async def test_releasing_one_of_two_keeps_gpu_shared(queries, sample_gpus):
    """Releasing one non-exclusive holder leaves the GPU ``shared``."""
    await queries.create_job(Job(id="svc-a", script_path=None, command="a"))
    await queries.create_job(Job(id="svc-b", script_path=None, command="b"))
    await queries.allocate_gpus("svc-a", ["GPU-0000-0001"], exclusive=False)
    await queries.allocate_gpus("svc-b", ["GPU-0000-0001"], exclusive=False)

    await queries.release_gpus("svc-a")

    assert await _status(queries, "GPU-0000-0001") == GpuStatus.shared
    assert await queries.get_job_gpus("svc-b") == ["GPU-0000-0001"]


@pytest.mark.asyncio
async def test_releasing_last_allocation_returns_gpu_to_idle(queries, sample_gpus):
    """A GPU returns to ``idle`` only when its last allocation is released."""
    await queries.create_job(Job(id="svc-a", script_path=None, command="a"))
    await queries.create_job(Job(id="svc-b", script_path=None, command="b"))
    await queries.allocate_gpus("svc-a", ["GPU-0000-0001"], exclusive=False)
    await queries.allocate_gpus("svc-b", ["GPU-0000-0001"], exclusive=False)

    await queries.release_gpus("svc-a")
    await queries.release_gpus("svc-b")

    assert await _status(queries, "GPU-0000-0001") == GpuStatus.idle
    assert "GPU-0000-0001" in await _idle_ids(queries)


@pytest.mark.asyncio
async def test_exclusive_allocation_is_busy_and_blocks_sharing(queries, sample_gpus):
    """Exclusive (batch) allocation marks the GPU ``busy`` and blocks re-allocation."""
    await queries.create_job(Job(id="batch-a", script_path="/tmp/a.py"))
    await queries.create_job(Job(id="svc-b", script_path=None, command="b"))

    await queries.allocate_gpus("batch-a", ["GPU-0000-0001"])  # exclusive=True default
    assert await _status(queries, "GPU-0000-0001") == GpuStatus.busy

    # Neither an exclusive nor a non-exclusive allocation can join a busy GPU.
    with pytest.raises(RuntimeError):
        await queries.allocate_gpus("svc-b", ["GPU-0000-0001"], exclusive=False)
    with pytest.raises(RuntimeError):
        await queries.allocate_gpus("svc-b", ["GPU-0000-0001"])


@pytest.mark.asyncio
async def test_release_preserves_offline_status(db, queries, sample_gpus):
    """Releasing an unrelated job must not flip an out-of-pool GPU to idle."""
    await queries.create_job(Job(id="batch-a", script_path="/tmp/a.py"))
    await queries.allocate_gpus("batch-a", ["GPU-0000-0002"])

    # GPU-0000-0001 is taken out of the pool by discovery. Written directly:
    # `update_gpu_status` was a write-a-literal path with no production caller
    # and WP5 removed it; status is derived everywhere else.
    await db.conn.execute("UPDATE gpus SET status = 'offline' WHERE id = ?", ("GPU-0000-0001",))
    await db.conn.commit()

    await queries.release_gpus("batch-a")

    assert await _status(queries, "GPU-0000-0001") == GpuStatus.offline
    assert await _status(queries, "GPU-0000-0002") == GpuStatus.idle


@pytest.mark.asyncio
async def test_job_kind_round_trips_through_db(queries):
    """A job's kind survives a create → fetch round trip."""
    await queries.create_job(
        Job(id="model-job", script_path=None, command="serve", kind=JobKind.model)
    )
    fetched = await queries.get_job("model-job")
    assert fetched is not None
    assert fetched.kind == JobKind.model
    assert fetched.status == JobStatus.pending
