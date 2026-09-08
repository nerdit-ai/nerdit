"""Tests for the P15 ``DataController`` — off-tick image pull + ensure_ready.

Pure-async unit tests over the in-memory DB, a fake runtime and a real
``PostgresBackend`` probing a ``FakePostgres`` asyncio TCP server (no
TestClient). The lifecycle integration (pull → launch → ensure_ready via the
``ServiceController``) is WP3; this file exercises the controller's own guards
and persistence directly, mirroring ``test_model_controller.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nerdit.core.data import DataController, PostgresBackend
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

pytestmark = pytest.mark.asyncio

POSTGRES_IMAGE = "postgres:16"


class FakeRuntime:
    """Tracks image presence and pull calls; can fail pulls on demand."""

    def __init__(self) -> None:
        self.present: set[str] = set()
        self.pulled: list[str] = []
        self.pull_error: str | None = None

    async def image_exists(self, image_name: str) -> bool:
        return image_name in self.present

    async def pull_image(self, image: str) -> None:
        if self.pull_error:
            raise ContainerRuntimeError(self.pull_error)
        self.pulled.append(image)
        self.present.add(image)

    async def inspect_state(self, container_id: str):
        return None


class FakePostgres:
    """An asyncio TCP server speaking the Postgres SSLRequest handshake.

    ``ready`` is toggled between reconcile ticks to model a slow start: while
    ``False`` the server accepts then resets (the transient case); once ``True``
    it reads the SSLRequest and answers a single ``N`` byte (alive)."""

    def __init__(self) -> None:
        self.ready = True
        self._server: asyncio.AbstractServer | None = None
        self.port = 0

    async def __aenter__(self) -> FakePostgres:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        await self._server.start_serving()
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            if not self.ready:
                writer.close()
                return
            await reader.read(8)
            writer.write(b"N")
            await writer.drain()
        except OSError:
            pass
        finally:
            writer.close()


def _database(name: str = "pg", *, ready: bool = False, **kw: object) -> Job:
    cfg: dict = {
        "backend": "postgres",
        "image": POSTGRES_IMAGE,
        "port": 5432,
        "volumes": ["data:/var/lib/postgresql/data"],
    }
    if ready:
        cfg["db_ready"] = True
    return Job(
        name=name,
        kind=JobKind.database,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
        **kw,
    )


def _controller(queries, runtime, tmp_path) -> DataController:
    return DataController(
        PostgresBackend(ready_timeout_s=2.0), runtime, queries, data_dir=str(tmp_path)
    )


async def _settle(controller: DataController) -> None:
    """Await every in-flight pull/ensure task (they pop themselves on exit)."""
    for task in list(controller._pull_tasks.values()) + list(controller._ensure_tasks.values()):
        await task


async def _log_lines(queries, job_id: str) -> list[str]:
    return [entry.message for entry in await queries.get_logs(job_id)]


# --- off-tick image pull --------------------------------------------------------


async def test_ensure_image_present_is_true_without_pull(queries, tmp_path):
    runtime = FakeRuntime()
    runtime.present.add(POSTGRES_IMAGE)
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is True
    assert runtime.pulled == []
    assert controller._pull_tasks == {}


async def test_missing_image_spawns_exactly_one_pull_task(queries, tmp_path):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)
    cfg = json.loads(job.config)

    assert await controller.ensure_image(job, cfg) is False
    first_task = controller._pull_tasks[job.id]
    assert await controller.ensure_image(job, cfg) is False
    assert controller._pull_tasks[job.id] is first_task

    await _settle(controller)
    assert runtime.pulled == [POSTGRES_IMAGE]
    assert await controller.ensure_image(job, cfg) is True
    lines = await _log_lines(queries, job.id)
    assert any(f"Pulling image {POSTGRES_IMAGE}" in line for line in lines)
    assert any("pulled" in line for line in lines)


async def test_pull_failure_settles_row_failed(queries, tmp_path):
    runtime = FakeRuntime()
    runtime.pull_error = "registry unreachable"
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)

    row = await queries.get_job(job.id)
    assert row.status is JobStatus.failed
    assert row.desired_state == JobStatus.failed.value
    assert row.error_class is ErrorClass.image_pull_fail
    assert row.error_message == "Image pull failed: registry unreachable"
    assert any("Image pull failed" in line for line in await _log_lines(queries, job.id))
    audits, _ = await queries.list_audit_log(limit=10)
    entry = next(a for a in audits if a.action == "database.image_pull_failed")
    assert entry.principal_id == "system"
    assert entry.target_type == "database"


# --- ensure_ready (wire probe) --------------------------------------------------


async def test_on_running_success_persists_db_ready(queries, tmp_path):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    async with FakePostgres() as pg:
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)

    row = await queries.get_job(job.id)
    assert json.loads(row.config)["db_ready"] is True
    # Status/desired_state untouched — readiness is the config flag only.
    assert row.status is JobStatus.building
    assert any("ready" in line.lower() for line in await _log_lines(queries, job.id))


async def test_on_running_skips_when_already_ready(queries, tmp_path):
    # on_running is the RE-ADOPTION hook (same still-running container): a true
    # db_ready is still true of that container, so the attempt is legitimately
    # skipped. (The fresh-launch path uses on_launched, which does NOT trust it.)
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(ready=True)
    await queries.create_job(job)

    async with FakePostgres() as pg:
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)
    # No probe attempt recorded (already ready).
    assert controller._ensure_attempted == set()


async def test_on_running_readoption_preserves_db_ready(queries, tmp_path):
    """Re-adoption of the same container keeps db_ready and does not re-probe (C5)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(ready=True)
    await queries.create_job(job)

    async with FakePostgres() as pg:
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)

    row = await queries.get_job(job.id)
    assert json.loads(row.config)["db_ready"] is True  # NOT cleared
    assert controller._ensure_attempted == set()  # not re-probed


# --- on_launched: a fresh container's inherited db_ready is stale (C5) ----------


async def test_on_launched_clears_stale_flag_and_reprobes_new_container(queries, tmp_path):
    """A relaunch clears the previous container's db_ready before re-probing (C5).

    ``db_ready`` is a per-container wire-probe result, not a durable artifact:
    a new container has never answered, so its inherited flag must be cleared
    (synchronously, before the caller flips the row ``running``) and the probe
    must re-run for the new container id.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(ready=True)  # container A had stamped db_ready
    await queries.create_job(job)

    async with FakePostgres() as pg:
        # New container B launches. on_launched awaits the persisted clear before
        # spawning the probe task, so right after it returns (task not yet run)
        # the flag is already gone and the probe is armed for B.
        await controller.on_launched(job, "B", pg.port)
        row = await queries.get_job(job.id)
        assert "db_ready" not in json.loads(row.config)  # cleared, pre-running
        assert "B" in controller._ensure_attempted  # probe armed for the NEW id

        await _settle(controller)

    row = await queries.get_job(job.id)
    assert json.loads(row.config).get("db_ready") is True  # re-stamped on B's success


async def test_on_launched_leaves_cleared_when_new_container_never_ready(queries, tmp_path):
    """A replacement that never answers stays db_ready=false — no stale true survives."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(ready=True)
    await queries.create_job(job)

    async with FakePostgres() as pg:
        pg.ready = False  # stuck in recovery / never answers the probe
        await controller.on_launched(job, "B", pg.port)
        await _settle(controller)

    row = await queries.get_job(job.id)
    assert "db_ready" not in json.loads(row.config)  # NOT a false positive


async def test_on_launched_cancels_in_flight_probe_for_old_container(queries, tmp_path):
    """A relaunch cancels the previous container's in-flight probe before arming B.

    An in-flight ``_ensure_ready`` for container A could otherwise stamp db_ready
    AFTER on_launched's clear (its success write racing the relaunch), and its
    presence in ``_ensure_tasks`` would also block arming B's probe (the
    ``job.id in self._ensure_tasks`` guard). on_launched pops+cancels+reaps it
    first, so the clear sticks and B's probe arms cleanly.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(ready=True)  # container A had stamped db_ready
    await queries.create_job(job)

    # A hung in-flight probe for container A, registered exactly as ``_run`` would.
    hung = asyncio.Event()

    async def _never() -> None:
        await hung.wait()

    old_task = asyncio.create_task(_never())
    controller._ensure_tasks[job.id] = old_task
    controller._ensure_attempted.add("A")

    async with FakePostgres() as pg:
        await controller.on_launched(job, "B", pg.port)
        assert old_task.cancelled()  # the stale probe was reaped
        row = await queries.get_job(job.id)
        assert "db_ready" not in json.loads(row.config)  # the clear stuck
        assert "B" in controller._ensure_attempted  # B's probe armed
        await _settle(controller)

    row = await queries.get_job(job.id)
    assert json.loads(row.config).get("db_ready") is True  # re-stamped on B's success


async def test_probe_does_not_stamp_after_container_moved(queries, tmp_path):
    """A probe for container A must not stamp db_ready once the row moved to B.

    Defense in depth: an in-flight probe whose success write lands after the row's
    container_id already advanced to a replacement would re-assert the stale-ready
    lie ``on_launched`` just cleared. ``_ensure_ready`` re-checks the current
    container_id before stamping and skips on a mismatch.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database(container_id="B")  # the row already moved to container B
    await queries.create_job(job)

    async with FakePostgres() as pg:
        # Drive the probe as if it had been launched for the OLD container A.
        await controller._ensure_ready(job, json.loads(job.config), pg.port, "A")

    row = await queries.get_job(job.id)
    assert "db_ready" not in json.loads(row.config)  # A's success did NOT stamp


async def test_not_ready_rearms_and_retries_same_container(queries, tmp_path):
    """A transient 'not answering yet' must NOT spend the one attempt.

    On a slow start the first ensure_ready hits the port before initdb finishes
    (accept-then-reset). That is transient: the attempt is re-armed so the SAME
    container retries on the next tick and eventually persists db_ready — never
    stranded at db_ready=false. Mirrors the model slow-start regression.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    async with FakePostgres() as pg:
        pg.ready = False  # still starting

        # Tick 1: not ready → re-armed, nothing persisted.
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)
        row = await queries.get_job(job.id)
        assert "db_ready" not in json.loads(row.config)
        assert controller.needs_ensure(Job(**{**row.model_dump(), "container_id": "c1"})) is True

        # Tick 2: server now answers → db_ready persisted, attempt spent.
        pg.ready = True
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)

    row = await queries.get_job(job.id)
    assert json.loads(row.config).get("db_ready") is True
    assert controller.needs_ensure(Job(**{**row.model_dump(), "container_id": "c1"})) is False


async def test_new_container_rearms_attempt(queries, tmp_path):
    """A new container id (crash restart) re-arms the bounded attempt."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    async with FakePostgres() as pg:
        pg.ready = False
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)
        # c1's attempt is spent (transient re-arms it, so discard leaves it open);
        # simulate a genuinely spent attempt by a fresh container after readiness.
        pg.ready = True
        controller.on_running(job, "c2", pg.port)
        await _settle(controller)

    assert "c2" in controller._ensure_attempted
    row = await queries.get_job(job.id)
    assert json.loads(row.config).get("db_ready") is True


async def test_fresh_row_reread_before_stamping(queries, tmp_path):
    """The stamp re-reads the row so a concurrent config write is not clobbered."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, tmp_path)
    job = _database()
    await queries.create_job(job)

    async with FakePostgres() as pg:
        controller.on_running(job, "c1", pg.port)
        await _settle(controller)

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    # The stamp preserves the pre-existing config keys (backend/image/port/volumes)
    # rather than overwriting the whole blob from the stale snapshot.
    assert cfg["backend"] == "postgres"
    assert cfg["image"] == POSTGRES_IMAGE
    assert cfg["volumes"] == ["data:/var/lib/postgresql/data"]
    assert cfg["db_ready"] is True


# --- bridge posture (D-P15-4: databases reuse [models].bridge_host) ------------


async def test_bridge_host_follows_models_settings(queries, tmp_path):
    """The DB bridge_host tracks a non-default ``[models].bridge_host`` (D-P15-4).

    One bridge posture: a database port is advertised to app containers on the
    exact host a model endpoint is — so the lifespan MUST thread the real
    ``[models]`` settings (not a defaulted ``ModelsSettings()``), else a
    user-configured bridge host would silently diverge for databases.
    """
    from nerdit.config.settings import DatabasesSettings, ModelsSettings

    models = ModelsSettings(bridge_host="172.30.0.1")
    controller = DataController(
        PostgresBackend(),
        FakeRuntime(),
        queries,
        databases_settings=DatabasesSettings(),
        models_settings=models,
        data_dir=str(tmp_path),
    )
    assert controller.bridge_host == models.bridge_advertise_host == "172.30.0.1"
