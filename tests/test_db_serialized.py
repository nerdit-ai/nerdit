"""Test transaction cleanup and deletion races against a real in-memory database.

Serialized writers roll back failed transactions so the next BEGIN IMMEDIATE
succeeds. Roll back only active transactions: unconditional rollback fails under
legacy isolation. Reconcile teardown tolerates a row deleted before append_log.
Real foreign-key enforcement exposes failures hidden by mocked route queries.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from nerdit.core.services import ServiceController
from nerdit.db.models import Job, JobKind, JobStatus, LogStream
from nerdit.db.queries import _serialized
from nerdit.db.queries._base import REQUEST_WRITE_MARKER, RequestWriteMarker
from tests.test_services_reconcile import FakeRuntime, _controller


def _service_job(name: str = "svc") -> Job:
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        container_id=None,
        config='{"image": "demo:latest", "port": 8000}',
    )


# --- (a) poisoning regression -------------------------------------------------


async def test_append_log_on_deleted_job_does_not_poison_connection(queries):
    """append_log for a vanished row raises, but the connection stays usable.

    Pre-fix, the failed INSERT left an implicit transaction open; the next
    ``BEGIN IMMEDIATE`` writer then raised "cannot start a transaction within a
    transaction". WP-R rolls back before releasing the lock, so the subsequent
    idempotency claim succeeds.
    """
    job = _service_job("svc-a")
    await queries.create_job(job)
    await queries.delete_service_checked(job.id)

    # The FK to jobs(id) is now dangling → the log INSERT trips foreign_keys=ON.
    with pytest.raises(sqlite3.IntegrityError):
        await queries.append_log(job.id, "orphan line", LogStream.system)

    # The connection is NOT poisoned: the next BEGIN IMMEDIATE writer succeeds.
    claimed = await queries.insert_idempotency_inprogress(
        principal_id="tok-1",
        idem_key="key-1",
        method="POST",
        path="/services",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    assert claimed is True
    # And the connection is left clean.
    assert queries._db.conn.in_transaction is False


# --- (b) teardown tolerates a vanished row ------------------------------------


async def test_teardown_tolerates_concurrent_delete(queries):
    """_teardown_to_stopped completes without raising when the row is gone."""
    runtime = FakeRuntime()
    controller: ServiceController = _controller(queries, runtime)
    job = _service_job("svc-b")
    await queries.create_job(job)

    # The DELETE route wins the race: the row (and its FK-linked job_logs) is
    # already gone by the time teardown reaches its terminal append_log.
    await queries.delete_service_checked(job.id)

    # Must NOT raise — the tolerant helper swallows the IntegrityError.
    await controller._teardown_to_stopped(job)

    # Connection stays clean and writable afterwards.
    assert queries._db.conn.in_transaction is False
    claimed = await queries.insert_idempotency_inprogress(
        principal_id="tok-2",
        idem_key="key-2",
        method="DELETE",
        path="/services/svc-b",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    assert claimed is True


async def test_teardown_vs_delete_interleave_no_escaping_exception(queries):
    """Gathered teardown + delete on the same row: neither escapes a 500-class."""
    import asyncio

    runtime = FakeRuntime()
    controller: ServiceController = _controller(queries, runtime)
    job = _service_job("svc-c")
    await queries.create_job(job)

    # Both writers serialize on the write lock; whichever order the terminal
    # append_log lands relative to delete_service_checked, no exception escapes.
    results = await asyncio.gather(
        controller._teardown_to_stopped(job),
        queries.delete_service_checked(job.id),
        return_exceptions=True,
    )
    assert all(not isinstance(r, Exception) for r in results), results


# --- (c) rollback only when in a transaction ----------------------------------


async def test_serialized_no_rollback_on_clean_connection(db):
    """An exception on a clean connection re-raises verbatim, no secondary error.

    If the wrapper issued an unconditional ROLLBACK, legacy isolation mode would
    raise ``sqlite3.OperationalError('cannot rollback - no transaction is
    active')`` and mask the real error. The ``in_transaction`` guard prevents it.
    """

    class BoomError(Exception):
        pass

    @_serialized
    async def raiser(self: object) -> None:
        # Raise BEFORE touching the connection → no transaction is open.
        raise BoomError("application-level failure")

    class _Holder:
        def __init__(self, database: object) -> None:
            self._db = database

    holder = _Holder(db)
    assert db.conn.in_transaction is False

    with pytest.raises(BoomError):
        await raiser(holder)

    # No rollback attempted, connection still clean.
    assert db.conn.in_transaction is False


async def test_serialized_rollback_on_cancellation(db, queries):
    """A CancelledError mid-transaction must roll back before the lock releases.

    The sweep loops are cancelled at every daemon shutdown; a cancel delivered
    between a bare-commit writer's statement and its ``commit()`` used to leave
    the shared connection mid-transaction (``except Exception`` missed
    BaseException), poisoning the next ``BEGIN IMMEDIATE`` writer.
    """
    job = _service_job("svc-cancel")
    await queries.create_job(job)

    started = asyncio.Event()
    release = asyncio.Event()

    @_serialized
    async def slow_writer(self: object) -> None:
        await db.conn.execute(
            "INSERT INTO job_logs (job_id, message, stream) VALUES (?, ?, ?)",
            (job.id, "x", "system"),
        )
        started.set()
        await release.wait()  # cancellation lands here, mid-transaction
        await db.conn.commit()

    class _Holder:
        def __init__(self, database: object) -> None:
            self._db = database

    task = asyncio.create_task(slow_writer(_Holder(db)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert db.conn.in_transaction is False
    # The next BEGIN IMMEDIATE writer must not see a poisoned connection.
    claimed = await queries.insert_idempotency_inprogress(
        principal_id="tok-1",
        idem_key="key-cancel",
        method="POST",
        path="/services",
        expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    )
    assert claimed is True


async def test_cancelled_writer_flips_the_marker_conservatively(db, queries):
    """A cancelled writer must PIN the claim, even though the wrapper rolled back.

    aiosqlite executes every operation the coroutine already submitted on its
    worker thread regardless of whether the awaiting future was cancelled, so a
    writer cancelled while awaiting its ``commit()`` can still COMMIT after this
    wrapper took the exception path. The outcome is unknowable from the loop
    side, and the rule is fail-toward-keep: a claim wrongly kept costs
    availability for 24 h, a claim wrongly deleted re-executes a write.
    """
    job = _service_job("svc-cancel-marker")
    await queries.create_job(job)

    started = asyncio.Event()
    release = asyncio.Event()

    @_serialized
    async def slow_writer(self: object) -> None:
        await db.conn.execute(
            "INSERT INTO job_logs (job_id, message, stream) VALUES (?, ?, ?)",
            (job.id, "x", "system"),
        )
        started.set()
        await release.wait()  # cancellation lands here, mid-transaction
        await db.conn.commit()

    class _Holder:
        def __init__(self, database: object) -> None:
            self._db = database

    marker = RequestWriteMarker()
    token = REQUEST_WRITE_MARKER.set(marker)
    try:
        task = asyncio.create_task(slow_writer(_Holder(db)))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert marker.committed is True, (
            "a cancelled writer's outcome is unknowable — the claim must be pinned"
        )
    finally:
        REQUEST_WRITE_MARKER.reset(token)


# --- (d) the P22 write marker flips on success only ---------------------------


async def test_marker_stays_unset_when_the_writer_rolls_back(queries):
    """A writer that raised committed nothing, so the claim must stay deletable.

    D-P22-4's fail-safety rests on the flip sitting AFTER a successful return: a
    flip moved into a ``finally`` (or above the ``raise``) would mark a
    rolled-back request as committed, and a client that then disconnects gets
    its key pinned ``interrupted`` for 24 h — every retry answered with a 409
    although nothing durable ever landed.
    """
    job = _service_job("svc-marker")
    await queries.create_job(job)

    marker = RequestWriteMarker()
    token = REQUEST_WRITE_MARKER.set(marker)
    try:
        # An FK violation: the row is inserted, then the commit is refused, so
        # the wrapper's ``in_transaction`` rollback branch is the one exercised.
        with pytest.raises(sqlite3.IntegrityError):
            await queries.append_log("no-such-job", "orphan line", LogStream.system)
        assert marker.committed is False, "a rolled-back write must not pin the claim"

        # The same marker flips as soon as a writer really succeeds.
        await queries.append_log(job.id, "real line", LogStream.system)
        assert marker.committed is True
    finally:
        REQUEST_WRITE_MARKER.reset(token)
