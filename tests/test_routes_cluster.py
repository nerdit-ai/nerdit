"""Route-level tests for ``GET /cluster/stats`` — ``services_up`` (P6 / A1).

Mirrors the ``test_routes_services`` harness: the real cluster router with
``AsyncMock`` state (no real aiosqlite connection crossing event loops), plus
a direct-DB asyncio smoke proving :meth:`Queries.count_services_up` counts
``running``/``degraded`` service+model rows and ignores batch and terminal
states — deliberately *not* built on ``list_jobs``, which defaults to
``kind='batch'``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.daemon.routes.cluster import router as cluster_router
from nerdit.db.models import Job, JobKind, JobStatus


def _make_app(queries: AsyncMock) -> FastAPI:
    app = FastAPI()
    app.include_router(cluster_router)
    app.state.queries = queries
    monitor = MagicMock()
    monitor.get_metrics = MagicMock(return_value={})
    app.state.monitor = monitor
    app.state.settings = MagicMock()
    return app


def _queries(services_up: int = 0) -> AsyncMock:
    q = AsyncMock()
    q.list_gpus = AsyncMock(return_value=[])
    q.list_jobs = AsyncMock(return_value=[])
    q.count_services_up = AsyncMock(return_value=services_up)
    return q


# --- route: services_up surfaces in the stats payload -------------------------


def test_cluster_stats_surfaces_services_up():
    queries = _queries(services_up=3)
    client = TestClient(_make_app(queries), raise_server_exceptions=False)
    resp = client.get("/cluster/stats")
    assert resp.status_code == 200
    assert resp.json()["services_up"] == 3
    queries.count_services_up.assert_awaited_once()


def test_cluster_stats_services_up_zero_when_none():
    client = TestClient(_make_app(_queries(services_up=0)), raise_server_exceptions=False)
    resp = client.get("/cluster/stats")
    assert resp.status_code == 200
    assert resp.json()["services_up"] == 0


# --- count_services_up (direct-DB; avoids TestClient over aiosqlite) ----------


def _row(job_id: str, kind: JobKind, status: JobStatus, service_name: str | None = None) -> Job:
    return Job(
        id=job_id,
        name=job_id,
        kind=kind,
        service_name=service_name,
        script_path="/x.py" if kind is JobKind.batch else None,
        gpu_count=0,
        status=status,
    )


def test_count_services_up_counts_running_and_degraded_only():
    async def _run() -> None:
        from nerdit.db.database import Database
        from nerdit.db.queries import Queries

        db = Database(":memory:")
        await db.connect()
        await db.init_schema()
        q = Queries(db)
        try:
            # Counted: running/degraded services and models.
            await q.create_job(_row("svc-run", JobKind.service, JobStatus.running, "svc-run"))
            await q.create_job(_row("svc-deg", JobKind.service, JobStatus.degraded, "svc-deg"))
            await q.create_job(_row("mdl-run", JobKind.model, JobStatus.running, "mdl-run"))
            # Ignored: batch (even running) and non-up service states.
            await q.create_job(_row("batch-run", JobKind.batch, JobStatus.running))
            await q.create_job(_row("svc-stop", JobKind.service, JobStatus.stopped, "svc-stop"))
            await q.create_job(_row("svc-fail", JobKind.service, JobStatus.failed, "svc-fail"))
            await q.create_job(_row("svc-build", JobKind.service, JobStatus.building, "svc-build"))
            await q.create_job(
                _row("mdl-restart", JobKind.model, JobStatus.restarting, "mdl-restart")
            )
            assert await q.count_services_up() == 3
        finally:
            await db.close()

    asyncio.run(_run())


def test_count_services_up_empty_db():
    async def _run() -> None:
        from nerdit.db.database import Database
        from nerdit.db.queries import Queries

        db = Database(":memory:")
        await db.connect()
        await db.init_schema()
        q = Queries(db)
        try:
            assert await q.count_services_up() == 0
        finally:
            await db.close()

    asyncio.run(_run())
