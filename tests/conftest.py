"""Shared test fixtures for Nerdit."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nerdit.db.database import Database
from nerdit.db.models import Gpu, Job, JobKind, JobStatus
from nerdit.db.queries import Queries


@pytest.fixture(autouse=True)
def _isolate_mcp_transport_mode():
    """Keep ``nerdit.mcp.transport``'s process globals from leaking between tests.

    ``build_http_app()`` flips them and never resets — correct for a daemon
    (one process, one transport), wrong inside a shared pytest process: any
    test that calls ``create_app()`` while the ambient ``~/.nerdit/config.toml``
    carries ``[mcp].http_enabled`` turns every later test's tool call into an
    HTTP-mode one, and the HTTP-only refusals (``_local_path_refusal`` /
    ``_readonly_write_refusal``) then fire in files that never asked for them.
    """
    from nerdit.mcp import transport

    saved = (transport._HTTP_MODE, transport._HTTP_HOST, transport._HTTP_PORT)
    yield
    transport._HTTP_MODE, transport._HTTP_HOST, transport._HTTP_PORT = saved


@pytest.fixture
async def db():
    """In-memory database fixture."""
    database = Database(":memory:")
    await database.connect()
    await database.init_schema()
    yield database
    await database.close()


@pytest.fixture
async def queries(db):
    """Queries fixture with in-memory DB."""
    return Queries(db)


@pytest.fixture
async def sample_gpus(queries):
    """Insert 2 sample GPUs and return them."""
    gpus = [
        Gpu(id="GPU-0000-0001", name="NVIDIA H100 80GB", memory_mb=81920, compute_cap="9.0"),
        Gpu(id="GPU-0000-0002", name="NVIDIA H100 80GB", memory_mb=81920, compute_cap="9.0"),
    ]
    await queries.reconcile_gpus(gpus)
    return gpus


@pytest.fixture
def sample_service():
    """Factory for a P2 service ``Job`` row (mirrors the S5 controller helper).

    Shared by the S5 reconcile tests and the S9 daemon-restart harness: builds a
    ``kind='service'`` row with the service-mode columns populated and a minimal
    ``{image, port}`` config blob. ``gpu_count=0`` keeps placement off the GPU
    path so the factory works without seeded GPUs.
    """

    def _make(
        name: str = "svc",
        *,
        status: JobStatus = JobStatus.building,
        desired_state: str = "running",
        restart_policy: str = "always",
        container_id: str | None = None,
        config: str = '{"image": "demo:latest", "port": 8000}',
        **kw: object,
    ) -> Job:
        return Job(
            name=name,
            kind=JobKind.service,
            service_name=name,
            gpu_count=0,
            status=status,
            desired_state=desired_state,
            restart_policy=restart_policy,
            container_id=container_id,
            config=config,
            **kw,
        )

    return _make


@pytest.fixture
def fake_clock(monkeypatch):
    """Controllable clock patched into :mod:`nerdit.core.services`.

    The :class:`~nerdit.core.services.ServiceController` derives its restart
    backoff and rate-window purely from ``datetime.now(UTC)``; pinning that lets
    S5/S9 advance time deterministically (e.g. past a restart backoff) with no
    real sleeps. Only the ``datetime`` name imported into ``core.services`` is
    patched, so the rest of the process keeps real wall-clock time. Attributes
    other than ``now`` delegate to the real ``datetime`` class, so parsing and
    arithmetic still work.
    """
    from datetime import UTC, timedelta
    from datetime import datetime as _real_datetime

    import nerdit.core.services as services_module

    class _Clock:
        def __init__(self, start: _real_datetime) -> None:
            self._now = start

        def now(self, tz=None) -> _real_datetime:
            return self._now

        def advance(self, seconds: float) -> _real_datetime:
            self._now = self._now + timedelta(seconds=seconds)
            return self._now

        def set(self, when: _real_datetime) -> None:
            self._now = when

        def __getattr__(self, name):
            return getattr(_real_datetime, name)

    clock = _Clock(_real_datetime.now(UTC))
    monkeypatch.setattr(services_module, "datetime", clock)
    return clock


@pytest.fixture
def mock_runtime():
    """Mock container runtime."""
    runtime = AsyncMock()
    runtime.run = AsyncMock(return_value="container-abc123")
    runtime.stop = AsyncMock()
    runtime.kill = AsyncMock()
    runtime.wait = AsyncMock(return_value=0)
    runtime.remove = AsyncMock()
    runtime.status = AsyncMock(return_value="running")
    runtime.inspect_state = AsyncMock(return_value=None)
    runtime.image_exists = AsyncMock(return_value=True)
    runtime.list_managed_containers = AsyncMock(return_value=[])
    runtime.list_own_managed_containers = AsyncMock(return_value=[])
    # (P20 WP3) Explicit rather than auto-satisfied: an unset AsyncMock
    # attribute returns a child mock, so the crash-orphan reap would iterate a
    # MagicMock, raise TypeError and be eaten by its own blanket handler — the
    # reap would silently be a no-op for every test on this fixture.
    runtime.list_own_labeled_containers = AsyncMock(return_value=[])
    # (P20 WP6) Explicit for the same reason: the boot-side run-orphan kill
    # iterates this listing, so an unset attribute would hand it a child mock,
    # raise TypeError and be eaten by the loop's blanket handler — the boot kill
    # would silently be a no-op for every test on this fixture.
    runtime.list_own_run_containers = AsyncMock(return_value=[])
    # (P24 WP4) Explicit for the same reason as the three above: an unset
    # AsyncMock attribute returns a coroutine-of-MagicMock, so the /stats route
    # would receive a truthy non-``ContainerStats`` object and every field
    # projection would be a MagicMock. ``None`` is the honest default — this
    # fixture has no container to sample (the P13 ``inspect_state`` lesson).
    runtime.stats = AsyncMock(return_value=None)

    # (P20) ``tail``/``max_bytes`` (and ``since``) widen the Protocol's bounded-read contract;
    # the fake ignores them (its canned output is already tiny).
    async def mock_logs(container_id, follow=False, tail=None, max_bytes=None, since=None):
        for line in ["output line 1", "output line 2"]:
            yield line

    runtime.logs = mock_logs
    return runtime


@pytest.fixture
def mock_pynvml(monkeypatch):
    """Mock pynvml to simulate 2 GPUs."""
    mock = MagicMock()
    mock.nvmlInit = MagicMock()
    mock.nvmlShutdown = MagicMock()
    mock.nvmlDeviceGetCount = MagicMock(return_value=2)

    handles = [MagicMock(), MagicMock()]
    mock.nvmlDeviceGetHandleByIndex = MagicMock(side_effect=lambda i: handles[i])

    mock.nvmlDeviceGetUUID = MagicMock(side_effect=lambda h: f"GPU-{handles.index(h):04d}-0001")
    mock.nvmlDeviceGetName = MagicMock(return_value="NVIDIA H100 80GB")

    mem_info = MagicMock()
    mem_info.total = 81920 * 1024 * 1024
    mem_info.used = 1024 * 1024 * 1024
    mem_info.free = (81920 - 1024) * 1024 * 1024
    mock.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem_info)

    mock.nvmlDeviceGetCudaComputeCapability = MagicMock(return_value=(9, 0))

    util = MagicMock()
    util.gpu = 25
    util.memory = 10
    mock.nvmlDeviceGetUtilizationRates = MagicMock(return_value=util)
    mock.nvmlDeviceGetTemperature = MagicMock(return_value=55)
    mock.NVML_TEMPERATURE_GPU = 0

    monkeypatch.setitem(__import__("sys").modules, "pynvml", mock)
    return mock


@pytest.fixture
def mock_docker(monkeypatch):
    """Mock docker-py client."""
    mock_client = MagicMock()
    container = MagicMock()
    container.id = "container-abc123"
    container.status = "running"
    container.logs = MagicMock(return_value=b"hello world\n")
    container.wait = MagicMock(return_value={"StatusCode": 0})
    container.reload = MagicMock()
    # (P24 WP4) Explicit rather than auto-satisfied: ``DockerRuntime.stats``
    # calls ``container.stats(stream=False)`` and then indexes the payload. An
    # unset MagicMock attribute returns a child mock, which is not a ``dict``,
    # so the route would silently take the "unavailable" branch for every test
    # on this fixture instead of exercising the derivation. This canned payload
    # is a well-formed two-sample cgroup-v2 reply: a 2-core host, 25% of one
    # core busy (cpu_delta/system_delta = 1/8 * 2 cores * 100 = 25.0).
    container.stats = MagicMock(
        return_value={
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2_000_000},
                "system_cpu_usage": 16_000_000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1_000_000},
                "system_cpu_usage": 8_000_000,
            },
            "memory_stats": {
                "usage": 200 * 1024 * 1024,
                "limit": 1024 * 1024 * 1024,
                "stats": {"inactive_file": 100 * 1024 * 1024},
            },
            "networks": {"eth0": {"rx_bytes": 1024, "tx_bytes": 2048}},
            "pids_stats": {"current": 7},
        }
    )
    container.attrs = {
        "State": {
            "ExitCode": 0,
            "OOMKilled": False,
            "Error": "",
            "StartedAt": "",
            "FinishedAt": "",
        }
    }
    mock_client.containers.run = MagicMock(return_value=container)
    mock_client.containers.get = MagicMock(return_value=container)
    mock_client.containers.list = MagicMock(return_value=[])

    mock_module = MagicMock()
    mock_module.from_env = MagicMock(return_value=mock_client)
    mock_module.DockerClient = MagicMock(return_value=mock_client)
    mock_module.types.DeviceRequest = MagicMock()
    mock_module.errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
    mock_module.errors.NotFound = type("NotFound", (Exception,), {})
    mock_module.errors.APIError = type("APIError", (Exception,), {})
    mock_module.errors.BuildError = type("BuildError", (Exception,), {})
    # `nerdit.core.runtime.docker` imports `docker.client.DockerClient` (used
    # for `DockerClient.from_env()`) and `docker.models.containers` (a
    # `from __future__ import annotations`-deferred return-type reference
    # only, never evaluated at runtime) — both must be pre-registered so the
    # dotted imports resolve against this mock instead of trying to search
    # the real (mocked) `docker` package for a submodule.
    mock_module.client.DockerClient = MagicMock()
    mock_module.client.DockerClient.from_env = MagicMock(return_value=mock_client)

    monkeypatch.setitem(__import__("sys").modules, "docker", mock_module)
    monkeypatch.setitem(__import__("sys").modules, "docker.errors", mock_module.errors)
    monkeypatch.setitem(__import__("sys").modules, "docker.types", mock_module.types)
    monkeypatch.setitem(__import__("sys").modules, "docker.client", mock_module.client)
    monkeypatch.setitem(__import__("sys").modules, "docker.models", mock_module.models)
    monkeypatch.setitem(
        __import__("sys").modules, "docker.models.containers", mock_module.models.containers
    )
    return mock_client
