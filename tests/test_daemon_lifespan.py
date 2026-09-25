"""Tests for ``nerdit.daemon.server.lifespan`` — startup/shutdown wiring."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI

import nerdit.daemon.server as server_module
from nerdit.config.defaults import DEFAULT_DB_NAME
from nerdit.core.discovery import GpuDiscoveryResult
from nerdit.core.monitor import ResourceMonitor
from nerdit.core.proxy.manager import ProxyManager
from nerdit.core.runtime.stub import StubRuntime
from nerdit.core.services import ServiceController
from nerdit.core.workload import WorkloadManager
from nerdit.db.database import Database
from nerdit.db.models import Gpu, JobStatus
from nerdit.db.queries import Queries


class _TestBackend:
    name = "test"

    def collect_metrics(self):
        return {}


def _discovery_result(gpus):
    return GpuDiscoveryResult(backend=_TestBackend(), gpus=gpus)


def _settings_for_test(tmp_path, monkeypatch):
    """Return a NerditSettings pointing at tmp dirs so startup touches nothing real."""
    from nerdit.config.settings import NerditSettings

    data_dir = tmp_path / "data"
    upload_dir = tmp_path / "uploads"
    settings = NerditSettings(
        data_dir=str(data_dir),
        daemon={
            "host": "127.0.0.1",
            "port": 9321,
            "auth_token": "test-token",
            "upload_dir": str(upload_dir),
        },
    )
    # Make lifespan pick up our settings instead of reading the real config
    monkeypatch.setattr(server_module, "load_settings", lambda: settings)
    return settings


@pytest.mark.asyncio
async def test_lifespan_happy_path_populates_app_state(tmp_path, monkeypatch):
    settings = _settings_for_test(tmp_path, monkeypatch)

    # Two discovered GPUs
    gpus = [
        Gpu(id="GPU-0", name="H100", memory_mb=81920),
        Gpu(id="GPU-1", name="H100", memory_mb=81920),
    ]
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result(gpus),
        raising=False,
    )

    # Keep Docker out of the test — force StubRuntime selection
    class _BadDocker:
        def __init__(self):
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    app = FastAPI(lifespan=server_module.lifespan)

    async with app.router.lifespan_context(app):
        state = app.state
        assert isinstance(state.db, Database)
        assert state.queries is not None
        assert isinstance(state.workload_manager, WorkloadManager)
        assert isinstance(state.runtime, StubRuntime)
        assert isinstance(state.monitor, ResourceMonitor)
        assert state.settings is settings
        assert isinstance(state.service_controller, ServiceController)
        assert state.zombie_task is not None
        assert not state.zombie_task.done()

        # Discovered GPUs landed in the DB
        rows = await state.queries.list_gpus()
        assert len(rows) == 2

    # After exit
    assert state.zombie_task.done()
    assert state.workload_manager._running is False


@pytest.mark.asyncio
async def test_lifespan_adds_configured_upload_dir_to_allowed_mount_roots(tmp_path, monkeypatch):
    """PR review #42: a non-default upload_dir must become an allowed mount root.

    Uploaded job workspaces are bind-mounted as /workspace; without this a
    scoped-token job uploaded to a custom upload_dir is rejected by the Tier-B
    allowlist (and Tier-A if under ~/.nerdit)."""
    settings = _settings_for_test(tmp_path, monkeypatch)  # upload_dir = tmp/uploads (non-default)
    assert settings.daemon.upload_dir not in settings.containers.allowed_mount_roots  # precondition

    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        assert settings.daemon.upload_dir in app.state.settings.containers.allowed_mount_roots


@pytest.mark.asyncio
async def test_lifespan_wires_run_registry_into_the_zombie_sweeper(tmp_path, monkeypatch):
    """The sweeper's ``extra_protected`` hook IS the controller's bound
    ``protected_container_ids`` — runs/releases (P20) ∪ cutover greens (P24b).

    One-off run and ``[deploy].release`` containers are nerdit-labelled but have
    no ``jobs`` row, so ~30 s in they look exactly like orphans to the sweep; a
    cutover green is a SECOND live container for a row that already has one, so
    it is invisible to the row-backed set for its whole verify window. The kwarg
    being required means the wiring cannot be *omitted*; this pins that it is
    not wired to something else (in particular not to the narrower
    ``active_run_container_ids``, which would leave every green unprotected),
    and that the hook is live.
    """
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    real_sweeper_cls = server_module.ZombieSweeper
    built: list = []

    def _capturing_sweeper(**kwargs):
        sweeper = real_sweeper_cls(**kwargs)
        built.append(sweeper)
        return sweeper

    monkeypatch.setattr(server_module, "ZombieSweeper", _capturing_sweeper)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        assert len(built) == 1
        controller = app.state.service_controller
        hook = built[0]._extra_protected
        assert hook.__self__ is controller
        assert hook.__func__ is ServiceController.protected_container_ids

        # ...and it reads the live registry, not a snapshot taken at startup.
        controller._register_run("job-1", "run-1", is_release=False)
        controller._bind_run_container("job-1", "run-1", "run-container-id")
        assert hook() == {"run-container-id"}
        controller._discard_run("job-1", "run-1")
        assert hook() == set()
        # ...and the cutover half of the union is live through the same hook.
        controller._cutover_containers["job-2"] = "green-container-id"
        assert hook() == {"green-container-id"}
        controller._cutover_containers.clear()
        assert hook() == set()


@pytest.mark.asyncio
async def test_lifespan_handles_gpu_discovery_failure(tmp_path, monkeypatch, caplog):
    import logging

    _settings_for_test(tmp_path, monkeypatch)

    def _boom(_path, **_kwargs):
        raise RuntimeError("nvml-init failed")

    monkeypatch.setattr("nerdit.core.discovery.discover_gpu_system", _boom, raising=False)
    monkeypatch.setattr(
        server_module, "DockerRuntime", lambda: (_ for _ in ()).throw(RuntimeError("no"))
    )
    app = FastAPI(lifespan=server_module.lifespan)

    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            rows = await app.state.queries.list_gpus()
            assert rows == []

    assert any("GPU discovery failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_falls_back_to_stub_runtime_without_docker(tmp_path, monkeypatch, caplog):
    import logging

    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("no docker daemon")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)
    app = FastAPI(lifespan=server_module.lifespan)

    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            assert isinstance(app.state.runtime, StubRuntime)

    assert any("Docker not available" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_shutdown_cancels_zombie_sweep(tmp_path, monkeypatch):
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("no")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)
    # Speed up the zombie-sweep interval so we're sure the coroutine is scheduled
    monkeypatch.setattr(server_module, "ZOMBIE_SWEEP_INTERVAL_SECONDS", 0.05)

    app = FastAPI(lifespan=server_module.lifespan)

    async with app.router.lifespan_context(app):
        task = app.state.zombie_task
        assert not task.done()

    # After the context exits, the task must have been cancelled/reaped
    assert task.done()


@pytest.mark.asyncio
async def test_lifespan_wires_service_controller_into_workload_manager(tmp_path, monkeypatch):
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("no")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)
    app = FastAPI(lifespan=server_module.lifespan)

    async with app.router.lifespan_context(app):
        manager = app.state.workload_manager
        # The reconcile loop drives the very controller exposed on app.state —
        # the routes and the loop must never see two different controllers.
        assert manager._controllers[0] is app.state.service_controller
        # (P24c WP10) The GitWatch poller joins the same loop via register()
        # when [git] is enabled; nothing else is ever appended.
        assert manager._controllers[1:] == [app.state.gitwatch]


@pytest.mark.asyncio
async def test_lifespan_starts_proxy_before_workload_loop(tmp_path, monkeypatch):
    """The URL layer must be up before the first reconcile tick.

    ``WorkloadManager.start()`` runs one tick before its first sleep. If the
    proxy were still unstarted at that point, ``proxy.available`` would read
    False on a healthy boot: a pending cutover-eligible redeploy would take the
    destroy-first branch and GitWatch would report ``proxy_off``. Pinning that
    ``ProxyManager.start()`` has RETURNED — not merely been called — before the
    loop spawns is what closes that window."""
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("no")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    order: list[str] = []
    real_proxy_start = ProxyManager.start
    real_workload_start = WorkloadManager.start

    async def _proxy_start(self):
        await real_proxy_start(self)
        order.append("proxy")  # recorded once availability is settled

    async def _workload_start(self):
        order.append("workload")  # recorded before the reconcile task spawns
        await real_workload_start(self)

    monkeypatch.setattr(ProxyManager, "start", _proxy_start)
    monkeypatch.setattr(WorkloadManager, "start", _workload_start)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        pass

    assert order == ["proxy", "workload"]


@pytest.mark.asyncio
async def test_lifespan_readiness_check_failure_keeps_docker_runtime(tmp_path, monkeypatch, caplog):
    """A transient failure in a startup readiness check (e.g. dockerd restarting)
    must never downgrade the daemon to StubRuntime — only client construction may."""
    import logging

    from nerdit.db.models import GpuVendor

    _settings_for_test(tmp_path, monkeypatch)

    gpus = [
        Gpu(id="GPU-0", name="H100", memory_mb=81920),
        Gpu(
            id="gpu:amd:0",
            name="MI300X",
            memory_mb=196608,
            vendor=GpuVendor.amd,
            runtime_id="0",
            schedulable=True,
        ),
    ]
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result(gpus),
        raising=False,
    )

    class _FlakyDocker:
        """Constructor succeeds; every readiness check raises like a socket hiccup."""

        def __init__(self, *args, **kwargs):
            # Accept the sandbox kwargs (denied/allowed mount roots) the server
            # now passes to DockerRuntime.
            pass

        async def check_nvidia_runtime(self):
            raise ConnectionError("dockerd restarting")

        async def check_rocm_runtime(self):
            raise ConnectionError("dockerd restarting")

        async def image_exists(self, image_name):
            raise ConnectionError("dockerd restarting")

        async def list_managed_containers(self):
            return []

    monkeypatch.setattr(server_module, "DockerRuntime", _FlakyDocker)
    app = FastAPI(lifespan=server_module.lifespan)

    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            assert isinstance(app.state.runtime, _FlakyDocker)
            assert not isinstance(app.state.runtime, StubRuntime)

    assert any("check failed" in r.message for r in caplog.records)
    assert not any("Docker not available" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_amd_inventory_only_hint_logged_once(tmp_path, monkeypatch, caplog):
    import logging

    from nerdit.db.models import GpuVendor

    _settings_for_test(tmp_path, monkeypatch)

    gpus = [
        Gpu(
            id="gpu:amd:0",
            name="MI300X",
            memory_mb=196608,
            vendor=GpuVendor.amd,
            runtime_id="0",
            schedulable=False,
        ),
    ]
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result(gpus),
        raising=False,
    )

    class _BadDocker:
        def __init__(self):
            raise RuntimeError("no")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)
    app = FastAPI(lifespan=server_module.lifespan)

    with caplog.at_level(logging.INFO):
        async with app.router.lifespan_context(app):
            pass

    hints = [r for r in caplog.records if "inventory-only" in r.message]
    assert len(hints) == 1
    assert "enable_amd" in hints[0].message


# --- P13c §3: the MCP session manager is exited whatever startup does ---------


class _RecordingSessionCm:
    """Stand-in for ``StreamableHTTPSessionManager.run()``."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *_exc) -> bool:
        self.exited = True
        return False


@pytest.mark.asyncio
async def test_lifespan_exits_the_mcp_session_manager_when_startup_fails(tmp_path, monkeypatch):
    """A startup step raising after ``run().__aenter__()`` must not leak it.

    The session manager owns an anyio task group that can only be exited from
    the task that entered it — this lifespan. Without the ``finally`` the
    context stays entered for the rest of the process: uvicorn logs
    "Application startup failed. Exiting." and the daemon dies with a live task
    group and its cancel scope never unwound (under a ``TestClient``, an
    unexited scope in the middle of the stack). Nothing is inherited across a
    boot — ``/daemon/restart`` self-execs a new process and
    ``StreamableHTTPSessionManager.run()`` refuses a second call per instance
    anyway; the leak is confined to the dying process, which is reason enough.
    """
    _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self, **_kwargs):
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    session_cm = _RecordingSessionCm()

    class _FakeSessionManager:
        def run(self):
            return session_cm

    class _FakeMcpServer:
        session_manager = _FakeSessionManager()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("startup step failed after the session manager was entered")

    monkeypatch.setattr(server_module, "_warn_if_unauthenticated_exposure", _boom)

    app = FastAPI(lifespan=server_module.lifespan)
    app.state.mcp_server = _FakeMcpServer()

    with pytest.raises(RuntimeError, match="startup step failed"):
        async with app.router.lifespan_context(app):
            pass  # pragma: no cover — startup never completes

    assert session_cm.entered
    assert session_cm.exited

    # The failed boot leaves the rest of the lifespan's resources open (the
    # pre-existing posture — only the session manager is in scope here); tidy
    # them so the loop shuts down without pending-task warnings.
    for attr in ("zombie_task", "idempotency_task", "proxy_task", "retention_task"):
        task = getattr(app.state, attr, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await app.state.workload_manager.stop()
    await app.state.monitor.stop()
    await app.state.db.close()


# --- S9: daemon-restart recovery (re-adopt vs relaunch) -----------------------


class _RecoveryRuntime:
    """Fake runtime modelling the container set surviving a daemon restart.

    ``live`` is seeded with the containers still running after the reboot. A
    relaunch (``run``) registers the new container as live and records its
    config so the test can assert the published-port mapping. ``logs`` blocks so
    a re-adopted/launched log task stays tracked in ``_log_tasks`` (it is
    cancelled by the controller's shutdown on lifespan exit)."""

    def __init__(self, live: dict[str, datetime]) -> None:
        self.live: dict[str, datetime] = dict(live)
        self.run_count = 0
        self.run_configs: list = []
        self._counter = 0

    def __call__(self, *args, **kwargs) -> _RecoveryRuntime:
        # ``lifespan`` constructs ``DockerRuntime(denied_mount_paths=...,
        # allowed_mount_roots=...)``; return this same instance so the test holds
        # a handle on it.
        return self

    async def run(self, config) -> str:
        self.run_count += 1
        self._counter += 1
        cid = f"relaunch-{self._counter}"
        self.live[cid] = datetime.now(UTC)
        self.run_configs.append(config)
        return cid

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        self.live.pop(container_id, None)

    async def kill(self, container_id: str) -> None:
        self.live.pop(container_id, None)

    async def remove(self, container_id: str, force: bool = False) -> None:
        self.live.pop(container_id, None)

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        return 1  # non-zero exit — an 'always' policy still relaunches

    async def status(self, container_id: str) -> str | None:
        return "running" if container_id in self.live else None

    async def inspect_state(self, container_id: str):
        return None

    async def image_exists(self, image_name: str) -> bool:
        return True

    async def list_images(self) -> list[str]:
        return []

    async def list_managed_containers(self) -> list[tuple[str, datetime]]:
        return [(cid, ts) for cid, ts in self.live.items()]

    async def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
        since: int | None = None,
    ):
        # Only a *follow* stream is unbounded (that is what keeps the live log
        # task tracked). A one-shot tail read — the P21 crash-tail capture — must
        # return promptly, exactly like ``DockerRuntime.logs(follow=False)``;
        # blocking here would wedge the reconcile tick instead of the log task.
        if not follow:
            return
        await asyncio.Event().wait()  # block so the log task stays tracked
        yield  # pragma: no cover — makes this an async generator


async def _quiesce_reconcile_loop(manager: WorkloadManager) -> None:
    """Stop the autonomous reconcile loop so the test drives ticks by hand.

    The loop's idempotent recovery work (re-adopt a live service, crash-account a
    dead one) may have already run once on startup; that is harmless — the manual
    ticks below converge to the same known state."""
    manager._running = False
    if manager._task is not None:
        manager._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await manager._task
        manager._task = None


@pytest.mark.asyncio
async def test_lifespan_readopts_surviving_service_and_relaunches_dead_one(
    tmp_path, monkeypatch, sample_service, fake_clock
):
    """Daemon-restart recovery (S9): a survivor is re-adopted (no ``run``, logs
    re-spawned, host port unchanged) while a service whose container died while
    the daemon was down is relaunched **once** on the **same** published port."""
    settings = _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    # 1. Pre-seed the daemon's on-disk DB as if a prior run had left two services
    #    in desired_state='running': one container survives the reboot, one died.
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(data_dir / DEFAULT_DB_NAME)
    seed_db = Database(db_path)
    await seed_db.connect()
    await seed_db.init_schema()
    seed_q = Queries(seed_db)
    alive = sample_service("alive-svc", status=JobStatus.running, container_id="cid-alive")
    dead = sample_service("dead-svc", status=JobStatus.running, container_id="cid-dead")
    await seed_q.create_job(alive)
    await seed_q.create_job(dead)
    await seed_q.acquire_service_port("alive-svc", alive.id, 8000, (9400, 9499))
    await seed_q.acquire_service_port("dead-svc", dead.id, 8000, (9400, 9499))
    alive_port = (await seed_q.get_service_endpoint("alive-svc")).host_port
    dead_port = (await seed_q.get_service_endpoint("dead-svc")).host_port
    assert alive_port != dead_port
    await seed_db.close()

    # 2. Only the survivor's container is still live in the runtime.
    runtime = _RecoveryRuntime(live={"cid-alive": datetime.now(UTC)})
    monkeypatch.setattr(server_module, "DockerRuntime", runtime)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        manager = app.state.workload_manager
        assert app.state.runtime is runtime  # our fake, not the stub
        await _quiesce_reconcile_loop(manager)
        controller = app.state.service_controller
        queries = app.state.queries

        # --- Tick 1: re-adopt the survivor, crash-account the dead one ---
        await controller.reconcile()

        alive_job = await queries.get_service_by_name("alive-svc")
        assert alive_job.status is JobStatus.running
        assert alive_job.container_id == "cid-alive"  # untouched
        # Re-adopted: its log stream was re-spawned (tracked by container id)…
        assert "cid-alive" in controller._log_tasks
        # …and it was NEVER relaunched.
        assert runtime.run_count == 0
        assert (await queries.get_service_endpoint("alive-svc")).host_port == alive_port

        dead_job = await queries.get_service_by_name("dead-svc")
        assert dead_job.status is JobStatus.restarting
        assert dead_job.restart_count == 1
        assert runtime.run_count == 0  # still in backoff — not yet relaunched

        # --- Advance past the restart backoff, then Tick 2: relaunch once ---
        fake_clock.advance(120)
        await controller.reconcile()

        dead_job = await queries.get_service_by_name("dead-svc")
        assert dead_job.status is JobStatus.running
        assert runtime.run_count == 1  # relaunched exactly once
        relaunch_cfg = runtime.run_configs[-1]
        # Same published host port as before the reboot, on a forced bridge net.
        assert relaunch_cfg.ports == {8000: dead_port}
        assert relaunch_cfg.network_mode == "bridge"
        assert (await queries.get_service_endpoint("dead-svc")).host_port == dead_port

        # The survivor stayed put through the whole recovery.
        assert (await queries.get_service_by_name("alive-svc")).container_id == "cid-alive"


# --- P37 D-P37-2: the dump-staging carve-out and the boot sweep ---------------


@pytest.mark.asyncio
async def test_lifespan_carves_the_dump_staging_root_out_of_tier_a(tmp_path, monkeypatch):
    """(P37 D-P37-2) The staging root joins ``system_mount_roots``.

    A dump sibling bind-mounts exactly one host path, and it lives under the
    Tier-A-denied ``~/.nerdit``. Without this carve-out every dump on a default
    install is refused by the mount denylist — which is a *silent* class of
    failure in the sense that nothing but the sibling's launch would say so.
    The list is read off the real construction call rather than re-spelled here,
    so a future edit to the lifespan cannot leave this passing.
    """
    from nerdit.core.volumes import dump_staging_root

    settings = _settings_for_test(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )
    captured: dict = {}

    class _CapturingDocker:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            # Fall back to the stub: this test is about the arguments, and a
            # real client is neither available nor wanted in the lifespan.
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _CapturingDocker)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        pass

    data_dir = Path(settings.data_dir).expanduser()
    assert str(dump_staging_root(data_dir)) in captured["system_mount_roots"]
    # Tier-A carve-out ONLY (D-P37-2): a scoped token must still not be able to
    # bind-mount another daemon's staging dir through the user allowlist.
    assert str(dump_staging_root(data_dir)) not in settings.containers.allowed_mount_roots


def test_the_staging_carve_out_never_opens_the_backups_dir(mock_docker):
    """(P37 D-P37-2, §4.3) The carve-out is a DEDICATED root, not ``backups/``.

    ``_under_allowed`` is prefix-based, so carving out ``<data_dir>/backups``
    would have made the P14c control-plane tars — which carry the secrets
    MASTER KEY — mountable by any container the daemon launches. This pins the
    three answers on a ``~/.nerdit``-shaped data dir, the shape a default
    install actually has: the staging slot is accepted, the backups dir is not,
    and the data dir itself is not.
    """
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.protocol import SandboxViolationError
    from nerdit.core.volumes import dump_staging_dir, dump_staging_root

    data_dir = Path("~/.nerdit").expanduser()
    runtime = DockerRuntime(
        client=mock_docker,
        denied_mount_paths=["~/.nerdit"],
        allowed_mount_roots=[],
        system_mount_roots=[
            str(data_dir / "models"),
            str(data_dir / "services"),
            str(dump_staging_root(data_dir)),
        ],
    )

    # Accepted: the one path a dump sibling mounts.
    runtime._validate_mounts({str(dump_staging_dir(data_dir, "abc123def456")): "/nerdit-dump"})

    for blocked in (data_dir / "backups" / "x", data_dir):
        with pytest.raises(SandboxViolationError) as exc:
            runtime._validate_mounts({str(blocked): "/x"})
        assert exc.value.reason == "denied_mount"


@pytest.mark.asyncio
async def test_lifespan_sweeps_orphan_staging_dirs_at_boot(tmp_path, monkeypatch):
    """(P37 D-P37-2) Every staging dir present at boot is an orphan.

    The dump-slot registry is in-memory, so a daemon that crashed mid-dump left
    its staging dir behind with nothing tracking it — invisible disk, and a
    carved-out mountable directory nobody is watching. The v1 backup
    ``.staging-*`` dirs had the same gap and are swept by the same pass.
    """
    from nerdit.core.volumes import dump_staging_root

    settings = _settings_for_test(tmp_path, monkeypatch)
    data_dir = Path(settings.data_dir).expanduser()
    orphan = dump_staging_root(data_dir) / "abc123def456"
    orphan.mkdir(parents=True)
    (orphan / "dump.pgdump").write_bytes(b"leftover")
    backup_orphan = data_dir / "backups" / ".staging-deadbeef"
    backup_orphan.mkdir(parents=True)

    monkeypatch.setattr(
        "nerdit.core.discovery.discover_gpu_system",
        lambda _path, **_kwargs: _discovery_result([]),
        raising=False,
    )

    class _BadDocker:
        def __init__(self, **_kwargs):
            raise RuntimeError("docker unavailable in tests")

    monkeypatch.setattr(server_module, "DockerRuntime", _BadDocker)

    app = FastAPI(lifespan=server_module.lifespan)
    async with app.router.lifespan_context(app):
        assert not orphan.exists()
        assert not backup_orphan.exists()
        # The roots themselves survive — only their contents are orphans.
        assert dump_staging_root(data_dir).is_dir()
