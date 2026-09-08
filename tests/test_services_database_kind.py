"""WP0 — ``JobKind.database`` + ``MANAGED_KINDS`` centralization + ``ContainerConfig.user``.

Pure-async tests over the in-memory DB (no TestClient), so they run under
pytest-asyncio without the Starlette/aiosqlite cross-loop hazard. The
database kind must ride every managed-kind query the way ``service``/``model``
already do; the source-grep test pins that the widening lives in one place
(``MANAGED_KINDS`` / ``MANAGED_KINDS_SQL``), never a fresh ``('service',
'model')`` literal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nerdit
from nerdit.daemon.routes.services import _resolve_service
from nerdit.db.models import (
    MANAGED_KINDS,
    MANAGED_KINDS_SQL,
    ContainerConfig,
    Job,
    JobKind,
    JobStatus,
)


def _db_job(
    name: str = "pg",
    *,
    status: JobStatus = JobStatus.running,
    desired_state: str = "running",
    **kw: object,
) -> Job:
    return Job(
        name=name,
        kind=JobKind.database,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state=desired_state,
        restart_policy="on-failure",
        **kw,
    )


# --- the centralization constants --------------------------------------------


def test_managed_kinds_shape():
    assert MANAGED_KINDS == (JobKind.service, JobKind.model, JobKind.database)
    assert JobKind.batch not in MANAGED_KINDS
    # SQL fragment is derived from the constant, single-quote style.
    assert MANAGED_KINDS_SQL == "kind IN ('service', 'model', 'database')"


# --- WP0-T1: reconcile picks up a kind=database row --------------------------


async def test_reconcile_includes_database_row(queries):
    live_db = _db_job("pg", status=JobStatus.running, desired_state="running")
    settled_db = _db_job("pg-done", status=JobStatus.stopped, desired_state="stopped")
    wants_restart = _db_job("pg-again", status=JobStatus.stopped, desired_state="running")
    await queries.create_job(live_db)
    await queries.create_job(settled_db)
    await queries.create_job(wants_restart)

    names = {j.service_name for j in await queries.get_reconcilable_services()}
    assert "pg" in names  # not terminal
    assert "pg-again" in names  # desired != status
    assert "pg-done" not in names  # settled (stopped == stopped)


async def test_container_ids_include_database_row(queries):
    await queries.create_job(_db_job("pg", container_id="cid-db-1"))
    ids = await queries.get_service_container_ids()
    assert "cid-db-1" in ids


async def test_workload_configs_include_database_row(queries):
    await queries.create_job(_db_job("pg", config='{"backend": "postgres"}'))
    rows = await queries.list_workload_configs()
    by_name = {r["service_name"]: r for r in rows}
    assert "pg" in by_name
    assert by_name["pg"]["kind"] == "database"
    assert by_name["pg"]["config"] == {"backend": "postgres"}


async def test_count_services_up_includes_database_row(queries):
    await queries.create_job(_db_job("pg", status=JobStatus.running))
    await queries.create_job(_db_job("pg-degraded", status=JobStatus.degraded))
    await queries.create_job(_db_job("pg-stopped", status=JobStatus.stopped))
    # running + degraded count as up; stopped does not.
    assert await queries.count_services_up() == 2


# --- WP0-T2: GET /services lists it; the quartet resolves it -----------------


async def test_list_services_lists_database_row(queries):
    await queries.create_job(_db_job("pg"))
    services, _ = await queries.list_services()
    assert "pg" in {j.service_name for j in services}
    # Narrowing kinds still works.
    only_db, _ = await queries.list_services(kinds=(JobKind.database,))
    assert {j.kind for j in only_db} == {JobKind.database}


async def test_resolve_service_accepts_database_by_id_and_name(queries):
    row = _db_job("pg")
    await queries.create_job(row)
    # By id (the batch-id-can-never-be-acted-on guard now admits database).
    resolved = await _resolve_service(queries, row.id)
    assert resolved is not None and resolved.id == row.id
    # By service_name.
    resolved = await _resolve_service(queries, "pg")
    assert resolved is not None and resolved.kind is JobKind.database


async def test_resolve_service_still_rejects_batch_id(queries):
    batch = Job(script_path="run.py")
    await queries.create_job(batch)
    assert await _resolve_service(queries, batch.id) is None


async def test_delete_service_removes_database_row(queries):
    row = _db_job("pg")
    await queries.create_job(row)
    await queries.delete_service_checked(row.id)
    assert await queries.get_service_by_name("pg") is None


# --- WP0-T3: no stray ('service', 'model') literal survives in src/ -----------


def test_no_service_model_literal_in_src():
    """The kind widening lives in ``MANAGED_KINDS`` / ``MANAGED_KINDS_SQL`` only.

    A raw ``('service', 'model')`` (or double-quoted) literal anywhere under
    ``src/`` means a widening site was missed — a fourth kind would silently
    fail to be reconciled/listed/deleted through it.
    """
    src_root = Path(nerdit.__file__).resolve().parent
    needles = ("'service', 'model'", '"service", "model"')
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            offenders.append(str(path))
    assert not offenders, f"stale ('service', 'model') literal(s): {offenders}"


# --- WP0-T4: DockerRuntime forwards ContainerConfig.user ----------------------


def _config(**kw) -> ContainerConfig:
    kw.setdefault("gpu_ids", [])
    return ContainerConfig(image="postgres:16", command=None, **kw)


@pytest.mark.asyncio
async def test_docker_forwards_container_user(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    # Absent by default → no user kwarg (every existing caller unchanged).
    await runtime.run(_config())
    assert "user" not in mock_docker.containers.run.call_args.kwargs

    # Set → forwarded to docker-py verbatim.
    await runtime.run(_config(user="postgres"))
    assert mock_docker.containers.run.call_args.kwargs["user"] == "postgres"
