"""Tests for the P2 ServiceController reconcile loop (S5).

Pure-async tests over the in-memory DB and a fake container runtime (no
TestClient), so they run under pytest-asyncio without the Starlette/aiosqlite
cross-loop hazard. They exercise the autonomous state machine directly:
restart-on-crash, the rate-cap → ``failed`` after 3 restarts, health → degraded
(which stays running), the ``stopped`` terminal/no-op, host-port stability across
a restart, and CRIT-2 zombie-sweep protection of a live service.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nerdit.config.settings import ContainerSettings, ModelsSettings, ServicesSettings
from nerdit.core.launch import _RUN_LINE_MAX_BYTES, _RUN_TAIL_MAX_BYTES
from nerdit.core.models.backend import ModelPullError, OllamaBackend, sanitize_model_name
from nerdit.core.models.controller import ModelController
from nerdit.core.runtime.protocol import ContainerNotFoundError
from nerdit.core.services import RunPreconditionError, ServiceController
from nerdit.core.sweeper import ZombieSweeper
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus, LogStream, TokenRole

pytestmark = pytest.mark.asyncio


class FakeRuntime:
    """Minimal in-memory ContainerRuntime: tracks live containers by id."""

    def __init__(self) -> None:
        self.live: dict[str, datetime] = {}
        self.counter = 0
        self.run_configs: list = []
        self.exit_code = 1  # non-zero so 'on-failure'/'always' restart
        self.oom_killed = False  # (P13) inspect_state forensics seam
        # (P13) records the runtime call sequence so tests can pin ordering
        # (e.g. inspect_state MUST precede remove — once removed forensics are gone).
        self.calls: list[str] = []
        # (P13) when set, inspect_state returns this object verbatim (garbage-input
        # test) instead of a well-formed ContainerStateInfo.
        self.inspect_return: object | None = None
        # (P5) image-pull seam: images listed here are absent until pulled.
        self.missing_images: set[str] = set()
        self.pulled: list[str] = []
        # (P20) one-off-run seams. ``run_error`` makes ``run()`` raise so a
        # registry leak is observable; ``wait_calls`` records the bounded-wait
        # budget and ``wait_error`` (typically ``TimeoutError``) drives the
        # kill-on-timeout branch; ``log_lines`` is the canned tail and
        # ``log_calls`` records ``(cid, follow, tail, max_bytes)`` so the byte
        # budget can be pinned. Defaults keep every pre-P20 test byte-identical
        # (no error, empty tail).
        self.run_error: BaseException | None = None
        self.wait_calls: list[tuple[str, float | None]] = []
        self.wait_error: BaseException | None = None
        self.log_calls: list[tuple[str, bool, int | None, int | None]] = []
        self.log_lines: list[str] = []
        self.log_error: BaseException | None = None
        # (P20 WP3) label seams: seeded orphan containers per (label, value)
        # pair, and the ids kill() was called with (self.calls stays the
        # coarse "kill" marker existing ordering pins rely on).
        self.labeled_containers: dict[tuple[str, str], list[str]] = {}
        self.killed: list[str] = []

    async def run(self, config) -> str:
        if self.run_error is not None:
            raise self.run_error
        self.counter += 1
        cid = f"c{self.counter}"
        self.live[cid] = datetime.now(UTC)
        self.run_configs.append(config)
        return cid

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        self.live.pop(container_id, None)

    async def kill(self, container_id: str) -> None:
        self.calls.append("kill")
        self.killed.append(container_id)
        self.live.pop(container_id, None)

    async def remove(self, container_id: str, force: bool = False) -> None:
        self.calls.append("remove")
        self.live.pop(container_id, None)

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        self.wait_calls.append((container_id, timeout_s))
        if self.wait_error is not None:
            raise self.wait_error
        return self.exit_code

    async def status(self, container_id: str) -> str | None:
        return "running" if container_id in self.live else None

    async def inspect_state(self, container_id: str):
        from nerdit.core.runtime.protocol import ContainerStateInfo

        self.calls.append("inspect_state")
        if self.inspect_return is not None:
            return self.inspect_return
        return ContainerStateInfo(
            exit_code=self.exit_code,
            oom_killed=self.oom_killed,
            error=None,
            started_at=None,
            finished_at=None,
        )

    async def image_exists(self, image_name: str) -> bool:
        return image_name not in self.missing_images

    async def pull_image(self, image: str) -> None:
        self.pulled.append(image)
        self.missing_images.discard(image)

    async def list_images(self) -> list[str]:
        return []

    async def list_managed_containers(self) -> list[tuple[str, datetime]]:
        return [(cid, ts) for cid, ts in self.live.items()]

    async def list_own_managed_containers(self) -> list[tuple[str, datetime]]:
        # The instance-scoped variant the ZombieSweeper actually calls. Same
        # answer as the unscoped listing for a single-instance fake.
        return [(cid, ts) for cid, ts in self.live.items()]

    async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
        self.calls.append(f"list_labeled:{label}={value}")
        return list(self.labeled_containers.get((label, value), []))

    async def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
    ):
        self.log_calls.append((container_id, follow, tail, max_bytes))
        # (P21) Only the one-shot reads join the ordering marker list: the
        # background follow task races the reconcile tick, so recording it would
        # make every `calls` ordering assertion flaky.
        if not follow:
            self.calls.append("logs")
        if self.log_error is not None:
            raise self.log_error
        for line in self.log_lines:
            yield line


class FakeProxy:
    """Records ProxyManager register/deregister calls (P3 wiring tests)."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self.registered: dict[str, int] = {}
        self.deregistered: list[str] = []
        # (P25) The raw ``edge_auth`` blob the launch fast-path forwarded, per
        # service — the real manager parses/resolves it; here it is recorded so
        # the wiring is assertable without a secret store.
        self.edge_auth: dict[str, object] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def register(self, service_name: str, host_port: int, edge_auth: object = None) -> None:
        self.registered[service_name] = host_port
        self.edge_auth[service_name] = edge_auth

    async def deregister(self, service_name: str) -> None:
        self.deregistered.append(service_name)


def _controller(
    queries,
    runtime,
    *,
    proxy=None,
    model_controller=None,
    secrets=None,
    proxy_mode="path",
    **kw,
) -> ServiceController:
    settings = ServicesSettings(service_port_range="9400-9499", **kw)
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=settings,
        proxy=proxy,
        proxy_mode=proxy_mode,
        model_controller=model_controller,
        secrets=secrets,
    )


def _svc(
    name: str = "svc",
    *,
    status: JobStatus = JobStatus.building,
    desired_state: str = "running",
    restart_policy: str = "always",
    container_id: str | None = None,
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
        config='{"image": "demo:latest", "port": 8000}',
        **kw,
    )


# --- restart on crash ---------------------------------------------------------


async def test_restart_on_kill(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    # Initial launch.
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    first_cid = job.container_id
    assert first_cid in runtime.live

    # Kill the container; next tick records the crash and schedules a restart.
    runtime.live.pop(first_cid)
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.restarting
    assert job.restart_count == 1
    # (P13) Forensics persist on the NON-terminal restart path too.
    cfg = json.loads(job.config)
    assert cfg["last_exit_code"] == 1
    assert cfg["oom_killed"] is False
    assert "last_crash_at" in cfg

    # Backoff cleared → relaunch on the same row.
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    assert job.container_id != first_cid
    assert job.container_id in runtime.live
    await controller.shutdown()


# --- rate cap → failed after 3 ------------------------------------------------


async def test_backoff_cap_fails_after_three_restarts(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, service_max_restarts=3)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await queries.get_service_by_name("svc")
    for _ in range(40):
        job = await queries.get_service_by_name("svc")
        if job.status is JobStatus.failed:
            break
        # Kill the container whenever the service comes up to force a crash loop.
        if job.status is JobStatus.running and job.container_id in runtime.live:
            runtime.live.pop(job.container_id)
        await controller.reconcile()

    assert job.status is JobStatus.failed
    # 3 restarts allowed, the 4th crash exhausts the budget.
    assert job.restart_count == 4
    assert runtime.counter == 4  # initial launch + 3 relaunches
    # (P13) Budget-exhausted terminal carries the copied exit-code taxonomy.
    assert job.error_class is ErrorClass.user_error  # exit_code 1 → user_error
    assert "Restart budget exhausted" in (job.error_message or "")
    # PORTS-1: a terminally-failed service releases its host port back to the pool.
    assert await queries.get_service_endpoint("svc") is None
    await controller.shutdown()


async def test_oom_crash_classified_and_forensics_persisted(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 137
    runtime.oom_killed = True
    controller = _controller(queries, runtime, service_max_restarts=1)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await queries.get_service_by_name("svc")
    for _ in range(40):
        job = await queries.get_service_by_name("svc")
        if job.status is JobStatus.failed:
            break
        if job.status is JobStatus.running and job.container_id in runtime.live:
            runtime.live.pop(job.container_id)
        await controller.reconcile()

    assert job.status is JobStatus.failed
    assert job.error_class is ErrorClass.oom
    cfg = json.loads(job.config)
    assert cfg["oom_killed"] is True
    assert cfg["last_exit_code"] == 137
    await controller.shutdown()


async def test_garbage_inspect_result_degrades_to_no_forensics(queries):
    # (P13) A runtime whose inspect_state returns a garbage object (an un-specced
    # mock) — and whose wait() is also lost — must NOT crash the reconcile tick:
    # the defensive coercion degrades to no-forensics, never a serialization crash.
    from unittest.mock import MagicMock

    from nerdit.core.runtime.protocol import ContainerRuntimeError

    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # launch → running
    job = await queries.get_service_by_name("svc")
    first_cid = job.container_id

    # Kill the container; make BOTH forensic sources garbage/lost for the crash tick.
    runtime.live.pop(first_cid)
    runtime.inspect_return = MagicMock()  # garbage — no int exit_code / bool oom

    async def _lost_wait(container_id: str) -> int:
        raise ContainerRuntimeError("wait lost")

    runtime.wait = _lost_wait  # type: ignore[assignment]

    await controller.reconcile()  # crash path must complete cleanly

    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.restarting  # crash handled, not raised
    cfg = json.loads(job.config)
    # No int exit_code could be coerced → last_exit_code was NOT written; oom
    # degraded to False; the tick still stamped last_crash_at without raising.
    assert "last_exit_code" not in cfg
    assert cfg["oom_killed"] is False
    assert "last_crash_at" in cfg
    await controller.shutdown()


async def test_inspect_state_ordered_before_remove(queries):
    # (P13 §1.4) The inspect→remove order is load-bearing: once the container is
    # removed, ExitCode/OOMKilled are gone. Pin it via the recorded call order.
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # launch → running
    job = await queries.get_service_by_name("svc")
    runtime.live.pop(job.container_id)  # crash
    runtime.calls.clear()

    await controller.reconcile()  # crash path: inspect_state then remove

    assert "inspect_state" in runtime.calls
    assert "remove" in runtime.calls
    assert runtime.calls.index("inspect_state") < runtime.calls.index("remove")
    await controller.shutdown()


# --- PORTS-1: terminal exit releases the host port ----------------------------


async def test_clean_exit_releases_endpoint(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 0  # clean exit
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="on-failure"))

    await controller.reconcile()  # launch (reserves a port)
    job = await queries.get_service_by_name("svc")
    assert await queries.get_service_endpoint("svc") is not None
    runtime.live.pop(job.container_id)  # container exits 0

    await controller.reconcile()  # on-failure + clean exit → completed (terminal)
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.completed
    assert await queries.get_service_endpoint("svc") is None
    await controller.shutdown()


# --- restart_policy is respected at terminal exit (Codex Comment 2) -----------


async def test_restart_policy_no_does_not_relaunch(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 1  # non-clean exit
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    await controller.reconcile()  # launch
    job = await queries.get_service_by_name("svc")
    runtime.live.pop(job.container_id)  # crash

    await controller.reconcile()  # policy 'no' → terminal failed
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.failed
    launched = runtime.counter

    # Subsequent ticks must NOT relaunch it (restart_policy='no' honored) and the
    # settled row must drop out of the reconcilable set.
    await controller.reconcile()
    await controller.reconcile()
    assert runtime.counter == launched
    assert "svc" not in [j.service_name for j in await queries.get_reconcilable_services()]
    await controller.shutdown()


async def test_clean_on_failure_exit_completes_without_relaunch(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 0  # clean exit
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="on-failure"))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    runtime.live.pop(job.container_id)

    await controller.reconcile()  # clean + on-failure → completed (terminal)
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.completed
    launched = runtime.counter
    await controller.reconcile()
    assert runtime.counter == launched  # not relaunched
    await controller.shutdown()


# --- restart replaces a running container (Codex Comment 1) -------------------


async def test_restart_replaces_running_container(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # launch
    job = await queries.get_service_by_name("svc")
    first_cid = job.container_id
    assert first_cid in runtime.live

    # Simulate the /restart route on a *running* service.
    await queries.set_desired_state(job.id, "running")
    await queries.update_job_status(job.id, JobStatus.restarting)
    await queries.bump_restart_count(job.id, 0, None)

    await controller.reconcile()  # restarting + live → destroy old, relaunch fresh
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    assert job.container_id != first_cid  # a brand-new container
    assert first_cid not in runtime.live  # old one torn down
    assert runtime.counter == 2
    await controller.shutdown()


# --- prebuilt-image service defers to the image CMD (Codex review) ------------


async def test_prebuilt_image_service_command_is_none(queries):
    # No command and no script_path → the container command must be None so the
    # image's own CMD runs; an empty list would override CMD with nothing.
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc"))  # _svc sets neither command nor script_path

    await controller.reconcile()
    assert runtime.run_configs[-1].command is None
    await controller.shutdown()


# --- SANDBOX-1: non-admin mount allowlist on the service launch path ----------


async def test_non_admin_script_path_mount_denied(queries):
    from nerdit.daemon.auth import generate_token, hash_token
    from nerdit.db.models import ApiToken

    token = await queries.create_api_token(
        ApiToken(name="bot", role=TokenRole.submitter, token_hash=hash_token(generate_token()))
    )
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    # A submitter-owned service whose script_path would bind-mount an arbitrary
    # host dir outside the Tier-B allowlist.
    svc = _svc("svc", script_path="/home/victim/.ssh/id_rsa", submitted_by_token=token.id)
    svc.config = '{"image": "demo:latest", "port": 8000}'
    await queries.create_job(svc)

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.failed  # denied, not launched or retry-looped
    assert runtime.counter == 0  # runtime.run never called
    assert await queries.get_service_endpoint("svc") is None
    await controller.shutdown()


# --- health → degraded (stays running) ----------------------------------------


async def test_health_failure_degrades_but_keeps_running(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    started = datetime.now(UTC) - timedelta(seconds=60)
    svc = _svc(
        "svc",
        status=JobStatus.running,
        container_id="c1",
        started_at=started,
        health_check={"path": "/healthz", "unhealthy_threshold": 2},
    )
    await queries.create_job(svc)
    # Stable endpoint so the controller has a host port to probe.
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    async def unhealthy(host_port, path, timeout):
        return 500

    controller._check_health = unhealthy  # type: ignore[assignment]

    await controller.reconcile()  # failure 1 (< threshold)
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running

    await controller.reconcile()  # failure 2 → degraded
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.degraded
    # Decision #2: degraded never auto-kills — the container is still live.
    assert "c1" in runtime.live

    # A later 2xx flips degraded → running.
    async def healthy(host_port, path, timeout):
        return 200

    controller._check_health = healthy  # type: ignore[assignment]
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    assert "c1" in runtime.live
    await controller.shutdown()


# --- TCP health probes (P14 WP-C1) --------------------------------------------


async def test_check_tcp_against_real_listener():
    """The module-level ``check_tcp`` connects to a live listener, refuses a dead one."""
    from nerdit.core.services import check_tcp

    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        assert await check_tcp(port, 2.0) is True
    finally:
        server.close()  # NB: no ``async with``/wait_closed — 3.12.0 can hang there
    # Listener gone → nothing accepts on the port → connection refused.
    assert await check_tcp(port, 2.0) is False


async def test_tcp_health_degrades_but_keeps_running(queries):
    """A ``type=tcp`` blob routes to the TCP probe; failures degrade (never kill)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    started = datetime.now(UTC) - timedelta(seconds=60)
    svc = _svc(
        "svc",
        status=JobStatus.running,
        container_id="c1",
        started_at=started,
        health_check={"type": "tcp", "unhealthy_threshold": 2},
    )
    await queries.create_job(svc)
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    http_called = False

    async def http_probe(host_port, path, timeout):
        nonlocal http_called
        http_called = True
        return 500

    controller._check_health = http_probe  # type: ignore[assignment]

    async def tcp_down(host_port, timeout):
        return False

    controller._check_tcp = tcp_down  # type: ignore[assignment]

    await controller.reconcile()  # failure 1 (< threshold)
    assert (await queries.get_service_by_name("svc")).status is JobStatus.running

    await controller.reconcile()  # failure 2 → degraded
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.degraded
    assert "c1" in runtime.live  # Decision #2: degraded stays running
    assert http_called is False  # tcp blob never touched the http probe

    # A later successful connect (synthesized 200) flips degraded → running.
    async def tcp_up(host_port, timeout):
        return True

    controller._check_tcp = tcp_up  # type: ignore[assignment]
    await controller.reconcile()
    assert (await queries.get_service_by_name("svc")).status is JobStatus.running
    await controller.shutdown()


async def test_junk_health_type_falls_through_to_http(queries):
    """An unknown probe type falls closed to the HTTP path (junk-type tolerance pin)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    started = datetime.now(UTC) - timedelta(seconds=60)
    svc = _svc(
        "svc",
        status=JobStatus.running,
        container_id="c1",
        started_at=started,
        health_check={"type": "grpc", "path": "/healthz", "unhealthy_threshold": 1},
    )
    await queries.create_job(svc)
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    seen_path: list[str] = []

    async def http_probe(host_port, path, timeout):
        seen_path.append(path)
        return 200

    controller._check_health = http_probe  # type: ignore[assignment]

    tcp_called = False

    async def tcp_probe(host_port, timeout):
        nonlocal tcp_called
        tcp_called = True
        return False

    controller._check_tcp = tcp_probe  # type: ignore[assignment]

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running  # http 200 → healthy
    assert seen_path == ["/healthz"]  # junk type used the http path
    assert tcp_called is False
    await controller.shutdown()


# --- stop terminal / no-op ----------------------------------------------------


async def test_stop_is_terminal_and_noop(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    svc = _svc("svc", status=JobStatus.running, desired_state="stopped", container_id="c1")
    await queries.create_job(svc)
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.stopped
    assert "c1" not in runtime.live
    # PORTS-1: the host port is released on stop (a retained port + freed quota
    # slot let a create→stop loop drain the pool). A later restart re-acquires.
    assert await queries.get_service_endpoint("svc") is None

    # A stopped/desired-stopped row is settled — never re-reconciled.
    reconcilable = await queries.get_reconcilable_services()
    assert "svc" not in [j.service_name for j in reconcilable]

    # Reconciling again is an explicit no-op.
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.stopped
    await controller.shutdown()


# --- container log caps (P14b WP-B2c) -----------------------------------------


async def test_launch_sets_container_log_config(queries):
    """A service launch carries the retention-configured json-file log caps."""
    from nerdit.config.settings import RetentionSettings

    runtime = FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        retention_settings=RetentionSettings(
            container_log_max_size="20m", container_log_max_file=5
        ),
    )
    await queries.create_job(_svc("svc"))
    await controller.reconcile()
    assert runtime.run_configs[-1].log_config == {"max-size": "20m", "max-file": "5"}
    await controller.shutdown()


async def test_launch_no_log_config_when_size_empty(queries):
    """(F7) An empty size is the opt-out: NO log_config at all on the container.

    Setting one would force LogConfig(type="json-file") in DockerRuntime.run and
    silently override an operator's journald/local/fluentd default driver.
    """
    from nerdit.config.settings import RetentionSettings

    runtime = FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        retention_settings=RetentionSettings(container_log_max_size="", container_log_max_file=3),
    )
    await queries.create_job(_svc("svc"))
    await controller.reconcile()
    assert runtime.run_configs[-1].log_config is None
    await controller.shutdown()


async def test_launch_no_log_config_without_retention(queries):
    """No retention settings ⇒ log_config stays None (byte-identical launch)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc"))
    await controller.reconcile()
    assert runtime.run_configs[-1].log_config is None
    await controller.shutdown()


async def test_launch_aborts_and_removes_container_when_row_deleted(queries, monkeypatch):
    """(F11) A REAL delete race ⇒ tear the fresh container down, wire nothing.

    The FK failure on the post-launch append IS the deleted-row signal. Continuing
    would spawn a log task and register an HTTPS route for a container with no DB
    owner — a deleted service stays reachable until the proxy reconcile prunes the
    route and the zombie sweep reaps the container. Here the delete really lands
    (the row is gone at the FK), so _launch must abort and destroy the container.
    """
    import sqlite3

    runtime = FakeRuntime()
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy)
    svc = _svc("svc")
    await queries.create_job(svc)

    real_append = queries.append_log
    fired = {"done": False}

    async def _delete_then_raise(job_id, message, stream):  # noqa: ANN001, ANN202
        # The post-launch line: the container is already live. Simulate the DELETE
        # route landing right here — the row (and its job_logs) are gone, so the
        # INSERT trips the FK.
        if "started" in message and not fired["done"]:
            fired["done"] = True
            # Mirror the DELETE route's order: release the endpoint (its FK points
            # at jobs(id)) and only then hard-delete the row + its job_logs. The
            # INSERT this append would have done now trips the job_logs FK.
            await queries.release_service_endpoint("svc")
            await queries.delete_service_checked(svc.id)
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
        return await real_append(job_id, message, stream)

    monkeypatch.setattr(queries, "append_log", _delete_then_raise)

    await controller.reconcile()  # must not raise

    assert fired["done"], "the post-launch append_log was expected to fire"
    assert runtime.run_configs, "the container WAS launched (run() ran) before the delete"
    # THE pins: the orphan container is torn down, and none of the post-launch
    # wiring ran for a row that no longer exists.
    assert runtime.live == {}, "the orphan container must be destroyed"
    assert "remove" in runtime.calls
    assert controller._log_tasks == {}, "no log task for a deleted service"
    assert proxy.registered == {}, "no HTTPS route for a deleted service"
    await controller.shutdown()


async def test_launch_tolerates_row_delete_race(queries, monkeypatch):
    """(F11 defensive leg) A SPURIOUS FK — the row still exists — stays tolerated.

    WP-R: reconcile-path append_log writes are FK-tolerant. The post-launch site is
    the one exception (see the sibling test above), but only when the row actually
    vanished. Here the append raises without the row being deleted (should not
    happen in practice), so the re-check finds the row and _launch keeps its
    tolerate-and-continue: the container is live and correctly wired, and the tick
    completes without an escaping exception.
    """
    import sqlite3

    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc"))

    real_append = queries.append_log
    raised = {"done": False}

    async def _raise_on_started(job_id, message, stream):  # noqa: ANN001, ANN202
        # Target the post-launch line specifically (run() has already succeeded).
        if "started" in message and not raised["done"]:
            raised["done"] = True
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
        return await real_append(job_id, message, stream)

    monkeypatch.setattr(queries, "append_log", _raise_on_started)

    # Must NOT raise even though the post-launch append_log FK-fails.
    await controller.reconcile()

    assert raised["done"], "the post-launch append_log was expected to fire"
    assert runtime.run_configs, "the container was still launched (run() ran)"
    # THE pin (fails on pre-fix code): the wiring AFTER the FK-raising append —
    # the log-collection task spawn — still ran. Pre-fix, the IntegrityError
    # aborted _launch there (reconcile's per-job try/except swallowed it, so the
    # old no-raise assert alone passed pre-fix too — deliberately insufficient).
    live_container = next(iter(runtime.live))
    assert live_container in controller._log_tasks, (
        "post-launch wiring (log task spawn) must complete despite the FK race"
    )
    await controller.shutdown()


# --- host-port stability across a restart -------------------------------------


async def test_host_port_stable_across_restart(queries):
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # initial launch reserves a host port
    endpoint = await queries.get_service_endpoint("svc")
    assert endpoint is not None
    host_port = endpoint.host_port
    job = await queries.get_service_by_name("svc")
    first_cid = job.container_id

    runtime.live.pop(first_cid)
    await controller.reconcile()  # crash → restarting
    await controller.reconcile()  # relaunch on the SAME port

    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    assert job.restart_count == 1
    endpoint = await queries.get_service_endpoint("svc")
    assert endpoint.host_port == host_port
    # The published port handed to docker is the stable host port.
    last_config = runtime.run_configs[-1]
    assert last_config.ports == {8000: host_port}
    assert last_config.network_mode == "bridge"
    await controller.shutdown()


# --- host-port bindability probe must mirror Docker (SO_REUSEADDR) ------------


async def test_is_port_bindable_sets_so_reuseaddr(monkeypatch):
    """The probe must set SO_REUSEADDR so a port lingering in TIME_WAIT right
    after a hard ``docker kill`` is reported bindable (Docker can bind it).

    Without this, the CRIT-4 fallback fires on a false negative and the stable
    host port drifts to a new one on every hard restart — caught only by the
    live kill/recover smoke, not the stubbed reconcile tests.
    """
    import socket as socket_mod

    opts: list[tuple[int, int]] = []
    real_socket = socket_mod.socket

    class _SpySocket:
        def __init__(self, *a, **k):
            self._s = real_socket(*a, **k)

        def setsockopt(self, level, optname, value):
            opts.append((level, optname))
            return self._s.setsockopt(level, optname, value)

        def __getattr__(self, name):
            return getattr(self._s, name)

    monkeypatch.setattr(socket_mod, "socket", _SpySocket)

    # A free ephemeral port is reported bindable...
    with real_socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert ServiceController._is_port_bindable(free_port) is True
    assert (socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR) in opts

    # ...and a port held by a live listener is reported NOT bindable.
    with real_socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM) as listener:
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        busy_port = listener.getsockname()[1]
        assert ServiceController._is_port_bindable(busy_port) is False


# --- CRIT-2: zombie sweep must not kill a live service ------------------------


async def test_zombie_sweep_protects_running_service(queries):
    runtime = FakeRuntime()
    # An old (> threshold) managed container belonging to a running service.
    runtime.live["czombie"] = datetime.now(UTC) - timedelta(seconds=120)
    await queries.create_job(_svc("svc", status=JobStatus.running, container_id="czombie"))

    sweeper = ZombieSweeper(queries=queries, runtime=runtime, extra_protected=lambda: set())
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    assert "czombie" in runtime.live  # protected — not swept


# --- P3: proxy route register/deregister with the service lifecycle -----------


async def test_proxy_registers_on_running_and_persists_route(queries):
    runtime = FakeRuntime()
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # building → running

    assert proxy.registered.get("svc") is not None
    ep = await queries.get_service_endpoint("svc")
    assert ep.route == "/svc"  # persisted projection (path mode)


async def test_proxy_registers_on_running_and_persists_route_subdomain_mode(queries):
    # P3.5 subdomain twin: the inline register hook persists the empty-string
    # route projection instead of the path prefix.
    runtime = FakeRuntime()
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy, proxy_mode="subdomain")
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # building → running

    assert proxy.registered.get("svc") is not None
    ep = await queries.get_service_endpoint("svc")
    assert ep.route == ""  # persisted projection (subdomain mode)


async def test_proxy_route_survives_restart_on_same_port(queries):
    runtime = FakeRuntime()
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # running
    first_port = proxy.registered["svc"]
    cid = next(iter(runtime.live))
    await runtime.kill(cid)  # crash

    # crash → restarting → running takes a couple of ticks (backoff 0).
    for _ in range(5):
        job = await queries.get_service_by_name("svc")
        if job.status is JobStatus.running and job.container_id in runtime.live:
            break
        await controller.reconcile()

    # Re-registered on the SAME stable host port (URL stability).
    assert proxy.registered["svc"] == first_port


async def test_proxy_deregisters_on_user_stop(queries):
    runtime = FakeRuntime()
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))
    await controller.reconcile()  # running

    await queries.set_desired_state((await queries.get_service_by_name("svc")).id, "stopped")
    await controller.reconcile()  # teardown → stopped

    assert "svc" in proxy.deregistered


async def test_proxy_deregisters_on_restart_cap_exhausted(queries):
    runtime = FakeRuntime()
    proxy = FakeProxy()
    # 0 restarts allowed → first crash exhausts the cap → terminal failed.
    controller = _controller(queries, runtime, proxy=proxy, service_max_restarts=0)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc", restart_policy="on-failure"))
    await controller.reconcile()  # running

    cid = next(iter(runtime.live))
    await runtime.kill(cid)
    for _ in range(5):
        job = await queries.get_service_by_name("svc")
        if job.status is JobStatus.failed:
            break
        if job.status is JobStatus.running and job.container_id in runtime.live:
            await runtime.kill(job.container_id)
        await controller.reconcile()  # crash → cap exhausted → failed

    job = await queries.get_service_by_name("svc")
    assert job.status == JobStatus.failed
    assert "svc" in proxy.deregistered


# --- P5: kind=model lifecycle + [ai.*] binding injection ------------------------


MODEL_NAME = sanitize_model_name("llama3.1:8b")  # 'ollama-llama3-1-8b'


class RecordingBackend(OllamaBackend):
    """Real OllamaBackend launch shape; ensure_model recorded (no HTTP)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.ensured: list[tuple[str, str]] = []
        self.fail = False

    async def ensure_model(self, base_url: str, model: str) -> None:
        self.ensured.append((base_url, model))
        if self.fail:
            raise ModelPullError(f"pull of {model!r} failed: manifest not found")


class FakeSecrets:
    """SecretManager stand-in returning a fixed env for every service."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def load(self, service: str) -> dict[str, str]:
        return dict(self.values)


# --- edge-case #5b: FORWARDED_ALLOW_IPS injection in path mode ---------------


async def test_launch_env_injects_forwarded_allow_ips_in_path_mode(queries):
    controller = _controller(queries, FakeRuntime(), proxy=FakeProxy(), proxy_mode="path")
    resolved = await controller._resolve_launch_env(_svc("svc"), {"port": 8000}, 8000, None, None)
    assert resolved.env["FORWARDED_ALLOW_IPS"] == "*"


async def test_launch_env_preserves_explicit_forwarded_allow_ips(queries):
    controller = _controller(queries, FakeRuntime(), proxy=FakeProxy(), proxy_mode="path")
    cfg = {"port": 8000, "env": {"FORWARDED_ALLOW_IPS": "172.17.0.1"}}
    resolved = await controller._resolve_launch_env(_svc("svc"), cfg, 8000, None, None)
    assert resolved.env["FORWARDED_ALLOW_IPS"] == "172.17.0.1"


async def test_launch_env_no_forwarded_allow_ips_for_model(queries):
    controller = _controller(queries, FakeRuntime(), proxy=FakeProxy(), proxy_mode="path")
    resolved = await controller._resolve_launch_env(
        _model_job(), {"port": 11434}, 11434, None, None
    )
    assert "FORWARDED_ALLOW_IPS" not in resolved.env


async def test_launch_env_no_forwarded_allow_ips_without_proxy(queries):
    controller = _controller(queries, FakeRuntime(), proxy=None, proxy_mode="path")
    resolved = await controller._resolve_launch_env(_svc("svc"), {"port": 8000}, 8000, None, None)
    assert "FORWARDED_ALLOW_IPS" not in resolved.env


async def test_launch_env_no_forwarded_allow_ips_when_proxy_disabled(queries):
    controller = _controller(
        queries, FakeRuntime(), proxy=FakeProxy(enabled=False), proxy_mode="path"
    )
    resolved = await controller._resolve_launch_env(_svc("svc"), {"port": 8000}, 8000, None, None)
    assert "FORWARDED_ALLOW_IPS" not in resolved.env


async def test_launch_env_no_forwarded_allow_ips_in_subdomain_mode(queries):
    controller = _controller(queries, FakeRuntime(), proxy=FakeProxy(), proxy_mode="subdomain")
    resolved = await controller._resolve_launch_env(_svc("svc"), {"port": 8000}, 8000, None, None)
    assert "FORWARDED_ALLOW_IPS" not in resolved.env
    # PORT still wired regardless of mode.
    assert resolved.env["PORT"] == "8000"


def _model_job(*, pulled: bool = False, **kw: object) -> Job:
    cfg: dict = {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
    }
    if pulled:
        cfg["model_pulled"] = True
    return Job(
        name=MODEL_NAME,
        kind=JobKind.model,
        service_name=MODEL_NAME,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
        **kw,
    )


def _ai_app(name: str = "app", *, gpu_count: int = 0, extra_env: dict | None = None) -> Job:
    cfg: dict = {
        "image": "demo:latest",
        "port": 8000,
        "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
    }
    if extra_env:
        cfg["env"] = extra_env
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=gpu_count,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(cfg),
    )


def _mc(queries, runtime, tmp_path, *, backend=None) -> tuple[ModelController, RecordingBackend]:
    backend = backend or RecordingBackend()
    return (
        ModelController(
            backend,
            runtime,
            queries,
            # Pinned so the bind/URL assertions hold on every dev platform
            # ("auto" resolves differently on macOS).
            models_settings=ModelsSettings(bridge_host="172.17.0.1"),
            data_dir=str(tmp_path),
        ),
        backend,
    )


async def _settle_models(mc: ModelController) -> None:
    """Await the ModelController's in-flight pull/ensure tasks."""
    for task in list(mc._pull_tasks.values()) + list(mc._ensure_tasks.values()):
        await task


async def test_model_full_path_pull_launch_ensure(queries, tmp_path):
    runtime = FakeRuntime()
    runtime.missing_images.add("ollama/ollama")
    mc, backend = _mc(queries, runtime, tmp_path)
    controller = _controller(queries, runtime, model_controller=mc)
    await queries.create_job(_model_job())

    # Tick 1: image absent → off-tick pull spawned, NO launch this tick.
    await controller.reconcile()
    assert runtime.counter == 0
    await _settle_models(mc)
    assert runtime.pulled == ["ollama/ollama"]

    # Tick 2: image present → launch with the backend-composed shape.
    await controller.reconcile()
    job = await queries.get_service_by_name(MODEL_NAME)
    assert job.status is JobStatus.running
    ep = await queries.get_service_endpoint(MODEL_NAME)
    cfg = runtime.run_configs[-1]
    assert cfg.image == "ollama/ollama"
    assert cfg.ports == {11434: ep.host_port}  # Ollama's fixed container port
    # System-owned weights volume (bypasses the Tier-B allowlist narrowly).
    assert cfg.volumes == {str(Path(tmp_path) / "models" / "ollama"): "/root/.ollama"}
    assert cfg.env["OLLAMA_HOST"] == "0.0.0.0:11434"
    # Loopback + bridge-gateway dual bind — reachable from app containers only.
    assert cfg.extra_port_bind_ips == ["172.17.0.1"]
    assert cfg.network_mode == "bridge"

    # Post-launch hook: ensure_model fired on loopback; success persists the flag.
    await _settle_models(mc)
    assert backend.ensured == [(f"http://127.0.0.1:{ep.host_port}", "llama3.1:8b")]
    job = await queries.get_service_by_name(MODEL_NAME)
    assert json.loads(job.config)["model_pulled"] is True
    await controller.shutdown()


async def test_model_crash_restart_relaunches_and_rearms_ensure(queries, tmp_path):
    runtime = FakeRuntime()
    mc, backend = _mc(queries, runtime, tmp_path)
    backend.fail = True  # first ensure attempt fails (weights unavailable)
    controller = _controller(queries, runtime, model_controller=mc)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_model_job())

    await controller.reconcile()  # launch
    await _settle_models(mc)
    assert len(backend.ensured) == 1
    job = await queries.get_service_by_name(MODEL_NAME)
    assert "model_pulled" not in json.loads(job.config)
    first_cid = job.container_id

    # Same container on later ticks → NO per-tick ensure retry (bounded).
    await controller.reconcile()
    await _settle_models(mc)
    assert len(backend.ensured) == 1

    # Crash → relaunch re-arms the ensure attempt on the NEW container.
    backend.fail = False
    runtime.live.pop(first_cid)
    for _ in range(5):
        job = await queries.get_service_by_name(MODEL_NAME)
        if job.status is JobStatus.running and job.container_id in runtime.live:
            break
        await controller.reconcile()
    assert job.container_id != first_cid
    await _settle_models(mc)
    assert len(backend.ensured) == 2
    job = await queries.get_service_by_name(MODEL_NAME)
    assert json.loads(job.config)["model_pulled"] is True
    await controller.shutdown()


async def test_model_is_never_proxied_and_route_stays_null(queries, tmp_path):
    runtime = FakeRuntime()
    mc, _backend = _mc(queries, runtime, tmp_path)
    proxy = FakeProxy()
    controller = _controller(queries, runtime, proxy=proxy, model_controller=mc)
    await queries.create_job(_model_job(pulled=True))

    await controller.reconcile()  # → running
    job = await queries.get_service_by_name(MODEL_NAME)
    assert job.status is JobStatus.running

    # Loopback-only (Decision #3): no Caddy route, route projection stays NULL.
    assert proxy.registered == {}
    ep = await queries.get_service_endpoint(MODEL_NAME)
    assert ep.route is None
    # And the proxy desired-set never contains the model row.
    routes = await queries.list_active_service_routes()
    assert MODEL_NAME not in [r.service_name for r in routes]
    await controller.shutdown()


async def test_model_is_never_proxied_and_route_stays_null_subdomain_mode(queries, tmp_path):
    # P3.5 subdomain twin: the model exclusion (P5 decision #3) holds regardless
    # of proxy mode — kind=model rows stay loopback-only, route stays NULL (not
    # the subdomain "" projection).
    runtime = FakeRuntime()
    mc, _backend = _mc(queries, runtime, tmp_path)
    proxy = FakeProxy()
    controller = _controller(
        queries, runtime, proxy=proxy, model_controller=mc, proxy_mode="subdomain"
    )
    await queries.create_job(_model_job(pulled=True))

    await controller.reconcile()  # → running
    job = await queries.get_service_by_name(MODEL_NAME)
    assert job.status is JobStatus.running

    assert proxy.registered == {}
    ep = await queries.get_service_endpoint(MODEL_NAME)
    assert ep.route is None
    routes = await queries.list_active_service_routes()
    assert MODEL_NAME not in [r.service_name for r in routes]
    await controller.shutdown()


async def test_app_with_ai_waits_without_leaks_then_launches_with_env(
    queries, tmp_path, sample_gpus
):
    runtime = FakeRuntime()
    mc, backend = _mc(queries, runtime, tmp_path)
    controller = _controller(queries, runtime, model_controller=mc)
    await queries.create_job(_model_job())
    app = _ai_app("app", gpu_count=1)
    await queries.create_job(app)

    # Tick 1: the model launches; the app's binding is NOT ready (weights not
    # pulled) → the app must not launch and must hold NO resources.
    await controller.reconcile()
    app_row = await queries.get_service_by_name("app")
    assert app_row.status is JobStatus.building  # untouched, retried next tick
    assert await queries.get_service_endpoint("app") is None  # no port acquired
    assert await queries.get_job_gpus(app.id) == []  # no GPU allocated
    assert len(runtime.live) == 1  # only the model container runs

    # Tick 2 with the same not-ready state: the wait line is logged ONCE per
    # distinct message, not once per tick.
    await controller.reconcile()
    logs = [entry.message for entry in await queries.get_logs(app.id)]
    waits = [line for line in logs if "Waiting on AI binding" in line]
    assert len(waits) == 1
    assert "still pulling" in waits[0]

    # Weights land → the app launches on the following tick with the contract env.
    await _settle_models(mc)
    assert json.loads((await queries.get_service_by_name(MODEL_NAME)).config)["model_pulled"]
    await controller.reconcile()
    app_row = await queries.get_service_by_name("app")
    assert app_row.status is JobStatus.running
    assert await queries.get_job_gpus(app.id) != []
    model_port = (await queries.get_service_endpoint(MODEL_NAME)).host_port
    env = runtime.run_configs[-1].env
    assert env["OPENAI_BASE_URL"] == f"http://172.17.0.1:{model_port}/v1"
    assert env["OPENAI_API_KEY"] == "nerdit-local"
    assert env["OPENAI_MODEL"] == "llama3.1:8b"
    assert env["NERDIT_AI_DEFAULT_URL"] == f"http://172.17.0.1:{model_port}/v1"
    assert env["NERDIT_AI_DEFAULT_KEY"] == "nerdit-local"
    assert env["NERDIT_AI_DEFAULT_MODEL"] == "llama3.1:8b"
    assert len(backend.ensured) == 1
    await controller.shutdown()


async def test_resolved_binding_overrides_same_named_user_secret(queries, tmp_path):
    runtime = FakeRuntime()
    mc, _backend = _mc(queries, runtime, tmp_path)
    secrets = FakeSecrets({"OPENAI_API_KEY": "user-key", "FOO": "bar"})
    controller = _controller(queries, runtime, model_controller=mc, secrets=secrets)
    model = _model_job(pulled=True)
    await queries.create_job(model)
    await queries.acquire_service_port(MODEL_NAME, model.id, 11434, (9400, 9499))
    await queries.update_job_status(model.id, JobStatus.running, container_id="cm")
    runtime.live["cm"] = datetime.now(UTC)
    await queries.create_job(_ai_app("app"))

    await controller.reconcile()
    app_row = await queries.get_service_by_name("app")
    assert app_row.status is JobStatus.running
    env = runtime.run_configs[-1].env
    # Decision #4: the platform-computed binding wins over the user secret...
    assert env["OPENAI_API_KEY"] == "nerdit-local"
    # ...while unrelated user secrets still flow through.
    assert env["FOO"] == "bar"
    await controller.shutdown()


async def test_reboot_readopts_live_model_and_refires_ensure(queries, tmp_path):
    # A daemon reboot mid-weights-pull: the model container survived, but
    # model_pulled never persisted. A FRESH controller pair over the same DB
    # must re-adopt the container (no relaunch) and re-fire ensure_model.
    runtime = FakeRuntime()
    runtime.live["cm"] = datetime.now(UTC)
    model = _model_job()
    model.status = JobStatus.running
    model.container_id = "cm"
    await queries.create_job(model)
    await queries.acquire_service_port(MODEL_NAME, model.id, 11434, (9400, 9499))

    mc, backend = _mc(queries, runtime, tmp_path)  # fresh (post-reboot) instances
    controller = _controller(queries, runtime, model_controller=mc)
    await controller.reconcile()
    await _settle_models(mc)

    assert runtime.counter == 0  # re-adopted, never relaunched
    ep = await queries.get_service_endpoint(MODEL_NAME)
    assert backend.ensured == [(f"http://127.0.0.1:{ep.host_port}", "llama3.1:8b")]
    row = await queries.get_service_by_name(MODEL_NAME)
    assert json.loads(row.config)["model_pulled"] is True
    await controller.shutdown()


# --- P11: vLLM shared-secret (HF_TOKEN) launch injection -------------------------


async def test_inject_model_launch_secrets_adds_hf_token_for_vllm(queries, tmp_path):
    from nerdit.core.models.backend import VllmBackend
    from nerdit.db.models import ContainerConfig, GpuVendor

    runtime = FakeRuntime()
    mc = ModelController(
        OllamaBackend(),
        runtime,
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
        models_settings=ModelsSettings(bridge_host="172.17.0.1"),
        data_dir=str(tmp_path),
    )
    controller = _controller(
        queries, runtime, model_controller=mc, secrets=FakeSecrets({"HF_TOKEN": "hf-abc"})
    )
    job = Job(
        name="v", kind=JobKind.model, service_name="v", config=json.dumps({"backend": "vllm"})
    )
    await queries.create_job(job)
    config = ContainerConfig(
        image="vllm/vllm-openai", gpu_ids=["0"], vendor=GpuVendor.nvidia, env={}
    )

    await controller._inject_model_launch_secrets(job, {"backend": "vllm"}, config)

    # Only the backend-declared key is injected, from the shared scope.
    assert config.env == {"HF_TOKEN": "hf-abc"}


async def test_inject_model_launch_secrets_noop_for_ollama(queries, tmp_path):
    from nerdit.db.models import ContainerConfig, GpuVendor

    runtime = FakeRuntime()
    mc, _ = _mc(queries, runtime, tmp_path)  # ollama-only registry
    controller = _controller(
        queries, runtime, model_controller=mc, secrets=FakeSecrets({"HF_TOKEN": "hf-abc"})
    )
    job = _model_job()  # backend=ollama
    await queries.create_job(job)
    config = ContainerConfig(image="ollama/ollama", gpu_ids=[], vendor=GpuVendor.nvidia, env={})

    await controller._inject_model_launch_secrets(job, {"backend": "ollama"}, config)

    # Ollama declares no launch secret keys → nothing injected.
    assert config.env == {}


async def test_model_relaunch_clears_stale_pull_error(queries, tmp_path):
    """A model relaunched after a failed weights pull clears the stale
    'Model pull failed' message BEFORE the row goes ``running``, so /wait and
    /diagnose do not report ``failed`` on a fresh container that is actively
    re-pulling (Codex review of the #14 fix). The off-tick ensure_model re-sets
    it only if THIS attempt also fails.
    """
    runtime = FakeRuntime()
    mc, _ = _mc(queries, runtime, tmp_path)
    controller = _controller(queries, runtime, model_controller=mc)
    job = _model_job(error_message="Model pull failed: pull of 'x:y' failed")
    await queries.create_job(job)

    await controller.reconcile()  # launches a fresh container

    row = await queries.get_job(job.id)
    assert row is not None
    assert row.status == JobStatus.running
    # The clear is synchronous in _launch (before update_job_status(running)),
    # so it holds regardless of the async ensure task.
    assert row.error_message is None
    await _settle_models(mc)
    await controller.shutdown()


async def test_service_relaunch_does_not_touch_error_message(queries):
    """The stale-error clear is model-only: a normal service launch never runs
    the clear path (guarded on ``is_model``), so this is a byte-identical launch.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    job = _svc("svc", status=JobStatus.building)
    await queries.create_job(job)
    await controller.reconcile()
    row = await queries.get_job(job.id)
    assert row is not None and row.status == JobStatus.running
    await controller.shutdown()


# --- P13 WP8: global build semaphore ------------------------------------------


class _GatingBuildRuntime(FakeRuntime):
    """FakeRuntime whose build_image blocks until released, tracking concurrency.

    Lets a test assert the global build semaphore caps how many builds run at
    once: each build increments ``active`` on entry (after the semaphore is
    acquired) and blocks on ``release`` until the test lets it finish.
    """

    def __init__(self, release) -> None:
        super().__init__()
        self._release = release
        self.active = 0
        self.max_active = 0
        self.started = 0

    async def build_image(self, context_dir, image, dockerfile=None):
        self.active += 1
        self.started += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await self._release.wait()
            return
            yield  # pragma: no cover — async-generator shape
        finally:
            self.active -= 1


async def _anoop(*args, **kwargs):
    return None


class _NoopQueries:
    """Async no-op stand-in for Queries.

    The semaphore test drives ``_build_app_image`` concurrently, which trips the
    aiosqlite single-connection cross-loop hazard if it touches the real DB. The
    build's incidental DB writes are not what's under test (the concurrency cap
    is), so every query method is a no-op here.
    """

    def __getattr__(self, _name):
        return _anoop


# --- P13 WP6: restart drain gate ----------------------------------------------


async def test_busy_builds_counts_only_unfinished_tasks(queries):
    import asyncio

    controller = _controller(queries, FakeRuntime())
    assert controller.busy_builds() == 0
    done = asyncio.create_task(_anoop())
    await done
    running = asyncio.create_task(asyncio.Event().wait())
    controller._build_tasks["a"] = done
    controller._build_tasks["b"] = running
    # Only the still-running task counts (the finished one is ignored).
    assert controller.busy_builds() == 1
    running.cancel()
    try:
        await running
    except asyncio.CancelledError:
        pass


async def test_spawn_build_task_parks_when_draining(queries, tmp_path):
    """While draining, a build is neither spawned nor registered, and its build
    context is left untouched so the re-exec'd daemon can rebuild it."""
    controller = _controller(queries, _CompletingBuildRuntime())
    controller.draining = True
    job = _svc("svc", status=JobStatus.building)
    ctx = tmp_path / "ctx"
    ctx.mkdir()

    controller._spawn_build_task(job, str(ctx), "img:1", None, str(ctx))

    assert job.id not in controller._build_tasks  # never registered
    assert controller.busy_builds() == 0  # no in-flight build
    assert ctx.is_dir()  # build context NOT rmtree'd by the finally cleanup


async def test_spawn_build_task_runs_when_not_draining(queries, tmp_path):
    """The drain gate is off by default: a build is registered and counted."""
    controller = _controller(queries, _CompletingBuildRuntime())
    assert controller.draining is False
    job = _svc("svc", status=JobStatus.building)
    await queries.create_job(job)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    controller._prune_old_images = _anoop  # type: ignore[assignment]

    controller._spawn_build_task(job, str(ctx), "img:1", None, str(ctx))
    assert job.id in controller._build_tasks
    await controller.shutdown()


async def test_build_semaphore_caps_concurrent_builds(tmp_path):
    import asyncio

    release = asyncio.Event()
    runtime = _GatingBuildRuntime(release)
    controller = _controller(_NoopQueries(), runtime, max_concurrent_builds=2)
    # The success tail (_prune_old_images) is not under test; keep it off the DB.
    controller._prune_old_images = _anoop  # type: ignore[assignment]

    # Three services, each with its own build context dir.
    jobs = [_svc(f"svc{i}", status=JobStatus.building) for i in range(3)]

    tasks = []
    for i, job in enumerate(jobs):
        ctx = tmp_path / f"ctx{i}"
        ctx.mkdir()
        tasks.append(
            asyncio.create_task(
                controller._build_app_image(
                    job, str(ctx), f"nerdit-app/svc{i}:1", "Dockerfile.nerdit", str(ctx)
                )
            )
        )

    # Let the two admitted builds enter build_image; the third waits on the
    # semaphore and must NOT have started.
    for _ in range(20):
        await asyncio.sleep(0)
        if runtime.started >= 2:
            break
    await asyncio.sleep(0)
    assert runtime.active == 2
    assert runtime.started == 2  # cap holds: the third build has not begun
    assert runtime.max_active == 2

    # Release the gate; all three complete and the peak never exceeded the cap.
    release.set()
    await asyncio.gather(*tasks)
    assert runtime.started == 3
    assert runtime.max_active == 2
    await controller.shutdown()


# --- WP17.1: build-surface extraction interception pins (R-B1 tripwire) ------
#
# ``core/app_build.py`` moves the build surface off ``ServiceController`` into
# ``AppImageBuilder``, but every §1.3 patch point must still be interceptable
# by assigning a plain attribute on the *controller* instance (tests, and any
# future caller, dispatch through the controller — never a private
# builder-internal method). These two pins exercise today's self-dispatch and
# MUST stay green, byte-unmodified, after the extraction (and after WP17.2).


async def test_prune_interception_survives_extraction(queries, tmp_path):
    """An instance-assigned ``_prune_old_images`` intercepts the success-tail
    call from inside ``_build_app_image``; the real prune never runs."""
    runtime = _CompletingBuildRuntime()
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(_deploy_svc("app", ctx=ctx))
    row = await queries.get_service_by_name("app")

    calls: list[str] = []

    async def _recorder(job: Job) -> None:
        calls.append(job.id)

    controller._prune_old_images = _recorder  # type: ignore[assignment]

    await controller._build_app_image(row, str(ctx), "nerdit-app/app:1", None, str(ctx))

    # The recorder ran exactly once, for this row; the real _prune_old_images
    # body (which would call runtime.list_images/remove_image) never executed.
    assert calls == [row.id]
    await controller.shutdown()


async def test_build_command_interception_survives_extraction(queries):
    """An instance-assigned ``_build_command`` intercepts the launch-time
    command lookup inside ``_launch`` (unchanged in WP17.1 — ``_build_command``
    is a launch-side helper, W17.2's concern — but the interception point must
    keep working through this and every later extraction)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._build_command = lambda job, cfg: (["echo", "MARKER"], {}, None)
    await queries.create_job(_svc("svc"))  # prebuilt image → no build phase, launches directly

    await controller.reconcile()

    assert runtime.run_configs[-1].command == ["echo", "MARKER"]
    await controller.shutdown()


async def test_needs_build_interception_survives_extraction(queries, tmp_path):
    """An instance-assigned ``_needs_build`` intercepts the build predicate
    inside ``ensure_built`` — the reconcile entry to the build surface. The
    builder must dispatch through the controller delegate, never its own
    ``needs_build`` (the PR #93 review found this seam bypassed post-WP17.1)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(_deploy_svc("app", ctx=ctx))
    row = await queries.get_service_by_name("app")

    calls: list[str] = []

    async def _recorder(job: Job) -> bool:
        calls.append(job.id)
        return False

    controller._needs_build = _recorder  # type: ignore[assignment]

    # With the override answering False, ensure_built must decline to build —
    # and must have asked the OVERRIDE, not the builder's real predicate.
    assert await controller._builder.ensure_built(row, None) is False
    assert calls == [row.id]
    await controller.shutdown()


# --- P13 WP2: last_deploy phase machine ---------------------------------------

from nerdit.core.runtime.protocol import ContainerRuntimeError  # noqa: E402

_LD_TS = "2026-07-10T12:00:00+00:00"


def _last_deploy(*, version: int, action: str, phase: str, image: str) -> dict:
    return {
        "version": version,
        "action": action,
        "phase": phase,
        "image": image,
        "started_at": _LD_TS,
        "updated_at": _LD_TS,
        "reason": None,
        "error_class": None,
        "error_message": None,
    }


def _deploy_svc(
    name: str = "app",
    *,
    version: int = 1,
    action: str = "create",
    phase: str = "queued",
    ctx: object = None,
    status: JobStatus = JobStatus.building,
    container_id: str | None = None,
    health_check: dict | None = None,
    **cfg_extra: object,
) -> Job:
    image = f"nerdit-app/{name}:{version}"
    cfg: dict = {
        "image": image,
        "build_version": version,
        "port": 8000,
        "last_deploy": _last_deploy(version=version, action=action, phase=phase, image=image),
    }
    if ctx is not None:
        cfg["build_context_dir"] = str(ctx)
    cfg.update(cfg_extra)
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state="running",
        restart_policy="always",
        container_id=container_id,
        health_check=health_check,
        config=json.dumps(cfg),
    )


class _CompletingBuildRuntime(FakeRuntime):
    """build_image marks the image present (a successful build) then returns."""

    async def build_image(self, context_dir, image, dockerfile=None):
        self.calls.append("build_image")
        self.missing_images.discard(image)
        yield f"Built {image}"


class _FailingBuildRuntime(FakeRuntime):
    """build_image always raises — a deterministic build failure."""

    async def build_image(self, context_dir, image, dockerfile=None):
        self.calls.append("build_image")
        raise ContainerRuntimeError("npm install failed")
        yield  # pragma: no cover — marks this an async generator


class _PlatformFailingBuildRuntime(FakeRuntime):
    """(BUG-1) build_image raises the HOST-fault subclass — no buildx on the node."""

    async def build_image(self, context_dir, image, dockerfile=None):
        from nerdit.core.runtime.protocol import BuildPlatformError

        self.calls.append("build_image")
        raise BuildPlatformError("docker build failed: the BuildKit builder ... is not available")
        yield  # pragma: no cover — marks this an async generator


class _ConcurrentWriteBuildRuntime(FakeRuntime):
    """A failing build that first lands a concurrent config write (regression seam).

    Simulates a config-blob write landing on the row *while the build task holds
    a stale tick-start snapshot*: the revert path must re-read the row, so the
    marker survives. With the old stale-``dict(cfg)`` rewrite it was clobbered.
    """

    def __init__(self, queries, job_id: str) -> None:
        super().__init__()
        self._q = queries
        self._job_id = job_id

    async def build_image(self, context_dir, image, dockerfile=None):
        row = await self._q.get_job(self._job_id)
        cfg = json.loads(row.config)
        cfg["concurrent_marker"] = "kept"
        await self._q.update_job_config(self._job_id, json.dumps(cfg))
        raise ContainerRuntimeError("build failed")
        yield  # pragma: no cover


async def _drain_builds(controller) -> None:
    for task in list(controller._build_tasks.values()):
        await task


def _phase(job) -> str:
    return json.loads(job.config)["last_deploy"]["phase"]


async def test_deploy_phase_machine_full_happy_path(queries, tmp_path):
    runtime = _CompletingBuildRuntime()
    image = "nerdit-app/app:1"
    runtime.missing_images.add(image)
    controller = _controller(queries, runtime)
    controller._prune_old_images = _anoop  # type: ignore[assignment]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(_deploy_svc("app", ctx=ctx))

    # Tick 1: build in flight → queued advances to building (semaphore acquired).
    await controller.reconcile()
    await _drain_builds(controller)
    job = await queries.get_service_by_name("app")
    assert _phase(job) == "building"

    # Tick 2: image present → launch; phase advances to launching (version-guarded).
    await controller.reconcile()
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert _phase(job) == "launching"

    # Tick 3: liveness-only live container → healthy (stamped once, from launching).
    await controller.reconcile()
    job = await queries.get_service_by_name("app")
    assert _phase(job) == "healthy"
    await controller.shutdown()


async def test_launch_stamps_last_launch_env_keys(queries, tmp_path):
    runtime = FakeRuntime()  # image already present → no build, straight to launch
    controller = _controller(queries, runtime)
    await queries.create_job(_deploy_svc("app", env={"FOO": "bar", "BAZ": "qux"}))

    await controller.reconcile()  # launch
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    keys = json.loads(job.config)["last_launch_env_keys"]
    # Names only (never values) + the platform PORT convention.
    assert set(keys) >= {"FOO", "BAZ", "PORT"}
    assert "bar" not in keys and "qux" not in keys
    await controller.shutdown()


async def test_rollback_skips_building_phase(queries, tmp_path):
    runtime = FakeRuntime()  # rollback image already built → _needs_build False
    controller = _controller(queries, runtime)
    # A rollback generation: image present, phase queued, action rollback.
    await queries.create_job(_deploy_svc("app", version=2, action="rollback"))

    await controller.reconcile()  # straight to launch (no build)
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert "build_image" not in runtime.calls  # never built
    assert _phase(job) == "launching"

    await controller.reconcile()  # liveness → healthy
    job = await queries.get_service_by_name("app")
    assert _phase(job) == "healthy"
    assert json.loads(job.config)["last_deploy"]["action"] == "rollback"
    await controller.shutdown()


async def test_crash_loop_marks_phase_failed(queries, tmp_path):
    runtime = FakeRuntime()  # exit_code 1 → non-clean, always-restart
    controller = _controller(queries, runtime, service_max_restarts=1)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_deploy_svc("app"))

    for _ in range(40):
        job = await queries.get_service_by_name("app")
        if job.status is JobStatus.failed:
            break
        if job.status is JobStatus.running and job.container_id in runtime.live:
            runtime.live.pop(job.container_id)
        await controller.reconcile()

    assert job.status is JobStatus.failed
    ld = json.loads(job.config)["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "crash_loop"
    assert ld["error_class"] == ErrorClass.user_error.value
    assert "Restart budget exhausted" in ld["error_message"]
    await controller.shutdown()


async def test_fresh_build_failure_marks_phase_failed(queries, tmp_path):
    runtime = _FailingBuildRuntime()
    image = "nerdit-app/app:1"
    runtime.missing_images.add(image)
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(_deploy_svc("app", ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    ld = json.loads(job.config)["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "build_failed"
    assert ld["error_class"] == ErrorClass.user_error.value
    await controller.shutdown()


async def test_platform_build_failure_settles_with_platform_error_class(queries, tmp_path):
    """(BUG-1) leg A: a HOST-fault build settles PLATFORM_ERROR, reason unchanged.

    ``reason`` stays ``build_failed`` — that vocabulary feeds audit and the
    message derivation. Only the *class* carries the new information,
    because redeploying the same source cannot help: the fix is on the machine.
    """
    runtime = _PlatformFailingBuildRuntime()
    image = "nerdit-app/app:1"
    runtime.missing_images.add(image)
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(_deploy_svc("app", ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    ld = json.loads(job.config)["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "build_failed"
    assert ld["error_class"] == ErrorClass.platform_error.value
    await controller.shutdown()


async def test_redeploy_build_failure_marks_failed_while_row_reverts_running(queries, tmp_path):
    runtime = _FailingBuildRuntime()
    new_image = "nerdit-app/app:2"
    runtime.missing_images.add(new_image)
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    # A redeploy over a still-live OLD container (version 1), building version 2.
    job = _deploy_svc(
        "app",
        version=2,
        action="redeploy",
        ctx=ctx,
        status=JobStatus.restarting,
        container_id="c-old",
        previous_image="nerdit-app/app:1",
    )
    await queries.create_job(job)
    row = await queries.get_service_by_name("app")
    runtime.live["c-old"] = datetime.now(UTC)  # old container keeps serving

    await controller.reconcile()
    await _drain_builds(controller)
    row = await queries.get_service_by_name("app")
    # Row reverts to running on the old image; phase records the build failure.
    assert row.status is JobStatus.running
    cfg = json.loads(row.config)
    assert cfg["image"] == "nerdit-app/app:1"
    ld = cfg["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "build_failed"
    await controller.shutdown()


async def test_build_revert_preserves_concurrent_config_write(queries, tmp_path):
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    job = _deploy_svc(
        "app",
        version=2,
        action="redeploy",
        ctx=ctx,
        status=JobStatus.restarting,
        container_id="c-old",
        previous_image="nerdit-app/app:1",
    )
    await queries.create_job(job)
    row = await queries.get_service_by_name("app")
    runtime = _ConcurrentWriteBuildRuntime(queries, row.id)
    runtime.missing_images.add("nerdit-app/app:2")  # force a build attempt
    runtime.live["c-old"] = datetime.now(UTC)
    controller = _controller(queries, runtime)

    await controller.reconcile()
    await _drain_builds(controller)
    row = await queries.get_service_by_name("app")
    cfg = json.loads(row.config)
    # The write that landed mid-build survives the revert (fresh read-modify-write).
    assert cfg["concurrent_marker"] == "kept"
    assert cfg["last_deploy"]["reason"] == "build_failed"
    await controller.shutdown()


class _GatingFailBuildRuntime(FakeRuntime):
    """A failing build that blocks on an event before raising.

    Lets a test land a concurrent redeploy write on the row while the (doomed)
    build is in flight, then release it to fail — exercising the F1 version-
    guarded revert CAS.
    """

    def __init__(self, release) -> None:
        super().__init__()
        self._release = release

    async def build_image(self, context_dir, image, dockerfile=None):
        self.calls.append("build_image")
        await self._release.wait()
        raise ContainerRuntimeError("build failed")
        yield  # pragma: no cover — async-generator shape


async def test_f1_clobber_build_revert_skips_newer_generation(queries, tmp_path):
    """A superseded gen-N build-failure revert must NOT clobber gen-N+1 (F1)."""
    import asyncio

    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    # Gen 1: a redeploy over a still-live OLD container (version 0).
    job = _deploy_svc(
        "app",
        version=1,
        action="redeploy",
        ctx=ctx1,
        status=JobStatus.restarting,
        container_id="c-old",
        previous_image="nerdit-app/app:0",
    )
    await queries.create_job(job)
    row = await queries.get_service_by_name("app")

    release = asyncio.Event()
    runtime = _GatingFailBuildRuntime(release)
    runtime.live["c-old"] = datetime.now(UTC)
    controller = _controller(queries, runtime)

    # Kick off the gen-1 build off a tick and let it enter build_image.
    build = asyncio.create_task(
        controller._build_app_image(row, str(ctx1), "nerdit-app/app:1", None, str(ctx1))
    )
    for _ in range(100):
        await asyncio.sleep(0.005)
        if "build_image" in runtime.calls:
            break
    assert "build_image" in runtime.calls

    # While gen-1 is blocked, a redeploy (gen 2) lands its blob on the row:
    # build_version 2, a fresh context dir, last_deploy queued v2.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen2_cfg = json.loads(
        _deploy_svc(
            "app", version=2, action="redeploy", ctx=ctx2, previous_image="nerdit-app/app:1"
        ).config
    )
    await queries.update_service_config(
        row.id,
        json.dumps(gen2_cfg),
        status=JobStatus.restarting,
        desired_state="running",
    )

    # Release → the gen-1 build fails and reverts. The version-guarded CAS must
    # see gen-2 owns the row and write NOTHING.
    release.set()
    await build

    row = await queries.get_service_by_name("app")
    cfg = json.loads(row.config)
    assert cfg["build_version"] == 2  # gen-2 blob intact (NOT reverted to 1)
    assert cfg["build_context_dir"] == str(ctx2)  # context dir NOT popped
    ld = cfg["last_deploy"]
    assert ld["phase"] == "queued" and ld["version"] == 2  # gen-2 not failed-stamped

    # Self-heal: gen-2 still builds + launches (the bug wedged this forever).
    heal_runtime = _CompletingBuildRuntime()
    heal_runtime.missing_images.add("nerdit-app/app:2")
    heal_runtime.live["c-old"] = datetime.now(UTC)
    heal = _controller(queries, heal_runtime)
    heal._prune_old_images = _anoop  # type: ignore[assignment]
    await heal.reconcile()  # spawns the gen-2 build
    await _drain_builds(heal)
    await heal.reconcile()  # image present → swap old container, launch
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.running
    assert json.loads(row.config)["image"] == "nerdit-app/app:2"
    await controller.shutdown()
    await heal.shutdown()


async def test_f1_clobber_revert_cas_rejects_version_mismatch(queries, tmp_path):
    """The revert CAS applies only to its own generation; a mismatch is a no-op (F1)."""
    await queries.create_job(
        _deploy_svc("app", version=2, action="redeploy", status=JobStatus.restarting)
    )
    row = await queries.get_service_by_name("app")

    # A superseded gen-1 revert (expect_build_version=1) must NOT apply — the row
    # owns build_version 2.
    applied = await queries.revert_service_config_guarded(
        row.id,
        json.dumps({"build_version": 1, "image": "nerdit-app/app:0"}),
        expect_build_version=1,
        status=JobStatus.running,
        desired_state="running",
    )
    assert applied is False
    after = await queries.get_service_by_name("app")
    assert json.loads(after.config)["build_version"] == 2  # blob untouched

    # The owning gen-2 revert (expect_build_version=2) applies.
    applied = await queries.revert_service_config_guarded(
        row.id,
        json.dumps({"build_version": 2, "image": "nerdit-app/app:1"}),
        expect_build_version=2,
        status=JobStatus.running,
        desired_state="running",
    )
    assert applied is True
    after = await queries.get_service_by_name("app")
    assert json.loads(after.config)["image"] == "nerdit-app/app:1"
    assert after.status is JobStatus.running


async def test_f5_crash_loop_stamp_version_guarded(queries, tmp_path):
    """A gen-N crash-loop failed stamp must NOT settle a fresh gen-N+1 record (F5)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, service_max_restarts=1)

    # The DB row is a FRESH gen-2 queued generation (a redeploy that landed this
    # tick, seeding a new record).
    await queries.create_job(
        _deploy_svc(
            "app",
            version=2,
            action="redeploy",
            phase="queued",
            status=JobStatus.restarting,
            container_id="c-old",
        )
    )
    row = await queries.get_service_by_name("app")

    # The controller observed the OLD gen-1 container exhausting its budget: its
    # tick-start snapshot carries last_deploy v1 and restart_count at the cap.
    snap_cfg = {
        "image": "nerdit-app/app:1",
        "build_version": 1,
        "port": 8000,
        "last_deploy": _last_deploy(
            version=1, action="create", phase="healthy", image="nerdit-app/app:1"
        ),
    }
    snapshot = Job(
        id=row.id,
        name="app",
        kind=JobKind.service,
        service_name="app",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        container_id="c-old",
        restart_count=1,  # at cap (service_max_restarts=1) → next crash is terminal
        restart_window_start=datetime.now(UTC),
        config=json.dumps(snap_cfg),
    )

    await controller._count_restart(snapshot, datetime.now(UTC), exit_code=1)

    row = await queries.get_service_by_name("app")
    ld = json.loads(row.config)["last_deploy"]
    # The gen-2 record is NOT failed-stamped by the gen-1 crash loop.
    assert ld["phase"] == "queued"
    assert ld["version"] == 2
    assert ld["reason"] is None
    # The row-level "any → failed" transition still applies (plan row semantics).
    assert row.status is JobStatus.failed


async def test_f5_crash_loop_skips_stamp_without_snapshot_last_deploy(queries, tmp_path):
    """A crashing row with no last_deploy at snapshot leaves the fresh record alone (F5)."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, service_max_restarts=1)

    # Fresh DB row DOES carry a gen-1 queued last_deploy (seeded concurrently).
    await queries.create_job(
        _deploy_svc(
            "app",
            version=1,
            action="create",
            phase="queued",
            status=JobStatus.restarting,
            container_id="c-old",
        )
    )
    row = await queries.get_service_by_name("app")

    # The crashing snapshot predates the phase machine — no last_deploy at all
    # (a POST /services row).
    snapshot = Job(
        id=row.id,
        name="app",
        kind=JobKind.service,
        service_name="app",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        container_id="c-old",
        restart_count=1,
        restart_window_start=datetime.now(UTC),
        config=json.dumps({"image": "demo:latest", "port": 8000}),
    )

    await controller._count_restart(snapshot, datetime.now(UTC), exit_code=1)

    row = await queries.get_service_by_name("app")
    ld = json.loads(row.config)["last_deploy"]
    assert ld["phase"] == "queued"  # untouched — no stamp landed
    assert ld["reason"] is None
    assert row.status is JobStatus.failed


async def test_model_phase_healthy_requires_model_pulled(queries, tmp_path):
    runtime = FakeRuntime()
    mc, _backend = _mc(queries, runtime, tmp_path)
    controller = _controller(queries, runtime, model_controller=mc)
    cfg = {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
        "build_version": 1,
        "last_deploy": _last_deploy(
            version=1, action="create", phase="launching", image="ollama/ollama"
        ),
    }
    job = Job(
        name=MODEL_NAME,
        kind=JobKind.model,
        service_name=MODEL_NAME,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        container_id="c1",
        config=json.dumps(cfg),
    )
    await queries.create_job(job)
    runtime.live["c1"] = datetime.now(UTC)
    mc._ensure_attempted.add("c1")  # suppress the ensure_model side-path

    # Weights not yet pulled → healthy is gated, phase stays launching.
    await controller.reconcile()
    await _settle_models(mc)
    row = await queries.get_service_by_name(MODEL_NAME)
    assert _phase(row) == "launching"

    # Mark weights present → next tick advances launching → healthy.
    c = json.loads(row.config)
    c["model_pulled"] = True
    await queries.update_job_config(row.id, json.dumps(c))
    await controller.reconcile()
    await _settle_models(mc)
    row = await queries.get_service_by_name(MODEL_NAME)
    assert _phase(row) == "healthy"
    await controller.shutdown()


# --- P14 WP-A1: named volumes at launch --------------------------------------


def _vol_controller(queries, runtime, tmp_path, **kw):
    """A ServiceController rooted at ``tmp_path`` so named volumes resolve."""
    from nerdit.config.settings import ContainerSettings

    settings = ServicesSettings(service_port_range="9400-9499", **kw)
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=settings,
        container_settings=ContainerSettings(),
        data_dir=str(tmp_path),
    )


def _vol_svc(name: str, volumes: list[str], *, seed_deploy: bool = False, **kw) -> Job:
    cfg = {"image": "demo:latest", "port": 8000, "volumes": volumes}
    if seed_deploy:
        # Deploy-shaped row so the launch-failure settle can stamp last_deploy
        # (the phase machine is a no-op on bare POST /services rows).
        cfg["build_version"] = 1
        cfg["last_deploy"] = {"phase": "launching", "version": 1, "action": "create"}
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(cfg),
        **kw,
    )


async def test_named_volume_mounts_with_perms(queries, tmp_path):
    """A valid named volume is mounted (host->container) with D-P14-3 perms."""
    import stat

    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    await queries.create_job(_vol_svc("svc", ["data:/data", "cache:/var/cache"]))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running

    config = runtime.run_configs[-1]
    services_root = tmp_path / "services"
    svc_root = services_root / "svc"
    data_host = svc_root / "data"
    cache_host = svc_root / "cache"
    assert config.volumes[str(data_host.resolve())] == "/data"
    assert config.volumes[str(cache_host.resolve())] == "/var/cache"

    # D-P14-3: leaf dirs are sticky world-writable (0o1777); the parents are 0o700.
    assert stat.S_IMODE(services_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(svc_root.stat().st_mode) == 0o700
    assert stat.S_IMODE(data_host.stat().st_mode) == 0o1777
    assert stat.S_IMODE(cache_host.stat().st_mode) == 0o1777

    # Provenance: names only, sorted.
    cfg = json.loads(job.config)
    assert cfg["last_launch_volumes"] == ["cache", "data"]
    await controller.shutdown()


async def test_corrupted_volume_blob_fails_closed(queries, tmp_path):
    """A forged/corrupted volumes blob (bypassing parse) settles volume_invalid."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    # A space in the volname could never pass DeployConfig, but a hand-written or
    # forged config blob must be re-validated at launch and fail closed.
    await queries.create_job(_vol_svc("svc", ["bad name:/x"], seed_deploy=True))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.failed
    assert job.error_class is ErrorClass.user_error
    cfg = json.loads(job.config)
    assert cfg["last_deploy"]["reason"] == "volume_invalid"
    # Fail-closed: nothing launched, endpoint released.
    assert runtime.counter == 0
    assert await queries.get_service_endpoint("svc") is None
    await controller.shutdown()


async def test_traversal_volume_blob_fails_closed(queries, tmp_path):
    """A traversal host-path attempt in the blob never mounts — volume_invalid."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    await queries.create_job(_vol_svc("svc", ["../../etc:/x"], seed_deploy=True))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.failed
    cfg = json.loads(job.config)
    assert cfg["last_deploy"]["reason"] == "volume_invalid"
    assert runtime.counter == 0
    await controller.shutdown()


async def test_volume_nesting_conflict_fails_closed(queries, tmp_path):
    """A named volume nested under a Tier-B mount settles volume_conflict (L2)."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    # Inject a Tier-B mount at /data so data:/data nests under nothing but a
    # user mount at /data/sub would; here the named volume /data collides with a
    # Tier-B mount at /data/sub via the normalized prefix check.
    controller._build_command = lambda job, cfg: (None, {str(tmp_path / "h"): "/data/sub"}, None)
    await queries.create_job(_vol_svc("svc", ["data:/data"], seed_deploy=True))

    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.failed
    cfg = json.loads(job.config)
    assert cfg["last_deploy"]["reason"] == "volume_conflict"
    await controller.shutdown()


async def test_volume_path_conflict_helper(queries, tmp_path):
    """The prefix-based collision check catches nesting + trailing-slash variants."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    conflict = controller._volume_path_conflict
    assert conflict(["/data"], ["/data/sub"]) is not None  # named parent of mount
    assert conflict(["/data/sub"], ["/data"]) is not None  # named child of mount
    assert conflict(["/data"], ["/data"]) is not None  # exact
    assert conflict(["/data"], ["/logs"]) is None  # disjoint
    assert conflict(["/data"], ["/database"]) is None  # sibling prefix, not nested
    # Doubled leading slashes collapse to a single slash (kernel semantics), so a
    # Tier-B mount at ``//data`` still collides with a named volume at ``/data``.
    assert conflict(["/data"], ["//data"]) is not None
    assert conflict(["//data"], ["/data"]) is not None


async def test_last_launch_volumes_empty_when_data_dir_absent(queries):
    """Provenance reflects what was mounted: no data_dir ⇒ no volumes claimed (#4)."""
    from nerdit.config.settings import ContainerSettings

    runtime = FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        container_settings=ContainerSettings(),
        data_dir=None,  # no data_dir ⇒ named volumes are never mounted
    )
    await queries.create_job(_vol_svc("svc", ["data:/data"]))
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    stored = json.loads(job.config)
    # cfg still carries the raw volumes list, but none were mounted this launch.
    assert stored["last_launch_volumes"] == []
    await controller.shutdown()


async def test_named_volume_in_last_launch_env_keys(queries, tmp_path):
    """NERDIT_DATA_DIR set in the config env surfaces in last_launch_env_keys."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path)
    cfg = {
        "image": "demo:latest",
        "port": 8000,
        "volumes": ["data:/data"],
        "env": {"NERDIT_DATA_DIR": "/data"},
    }
    job = Job(
        name="svc",
        kind=JobKind.service,
        service_name="svc",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(cfg),
    )
    await queries.create_job(job)
    await controller.reconcile()
    job = await queries.get_service_by_name("svc")
    stored = json.loads(job.config)
    assert "NERDIT_DATA_DIR" in stored["last_launch_env_keys"]
    await controller.shutdown()


# --- C5: a relaunched database must not report ready before the new probe -------


async def test_relaunch_clears_stale_db_ready_before_running(queries, tmp_path):
    """C5: a fresh database container's inherited db_ready is cleared at launch.

    A database row carrying a STALE ``db_ready=true`` (stamped by a previous
    container) is launched afresh; the new container has never answered the wire
    probe. The launch must clear the flag BEFORE the row goes ``running`` so
    ``/wait`` — which converges on ``running ∧ db_ready`` — keeps polling instead
    of reporting ready instantly. Fails pre-fix: ``on_running`` early-returned on
    the stale flag and never cleared it, so ``_evaluate_wait`` returned
    ``converged`` on a server that had never answered a byte.
    """
    from nerdit.core.data import DataController, PostgresBackend
    from nerdit.core.secrets import SecretManager
    from nerdit.daemon.routes.service_wait import _evaluate_wait

    runtime = FakeRuntime()
    secrets = SecretManager(tmp_path / "secrets")
    # The minted credential the launch branch requires for a database row.
    secrets.set("pg", {"POSTGRES_PASSWORD": "minted-pw"})
    # A short ready_timeout keeps the (always-refused) probe fast; nothing binds
    # the allocated host port under FakeRuntime, so the probe never succeeds.
    data = DataController(
        PostgresBackend(ready_timeout_s=0.5), runtime, queries, data_dir=str(tmp_path)
    )
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        secrets=secrets,
        data_controller=data,
        data_dir=str(tmp_path),
    )
    row = Job(
        name="pg",
        kind=JobKind.database,
        service_name="pg",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(
            {"backend": "postgres", "image": "postgres:16", "port": 5432, "db_ready": True}
        ),
    )
    await queries.create_job(row)

    await controller.reconcile()  # fresh container launch

    launched = await queries.get_service_by_name("pg")
    assert launched.status is JobStatus.running
    assert launched.container_id  # a brand-new container id
    cfg = json.loads(launched.config)
    assert "db_ready" not in cfg  # stale flag cleared before running (C5)
    # /wait must NOT converge — the new container has not been probed yet.
    outcome, _ = _evaluate_wait(launched, None, False)
    assert outcome is None  # keep polling (pre-fix: "converged")

    await data.shutdown()
    await controller.shutdown()


# --- WP17 PR2 tests-first (D-B8 _launch decomposition pins, §3) ----------------


async def test_relaunch_never_clears_model_pulled(queries, tmp_path):
    """D-T-6 asymmetry companion to the C5 pin above: ``model_pulled`` is a
    durable artifact — a relaunch (crash restart) must NEVER clear it, unlike
    the ephemeral per-container ``db_ready`` wire-probe flag (which the sibling
    test above shows IS cleared on replacement). Written before the ``_launch``
    decomposition (WP17 PR2) so the phase-function split cannot silently move
    the model branch's settle tail onto the database branch's clear.
    """
    runtime = FakeRuntime()
    mc, backend = _mc(queries, runtime, tmp_path)
    controller = _controller(queries, runtime, model_controller=mc)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_model_job(pulled=True))

    await controller.reconcile()  # launch
    job = await queries.get_service_by_name(MODEL_NAME)
    assert job.status is JobStatus.running
    assert json.loads(job.config)["model_pulled"] is True
    first_cid = job.container_id
    assert backend.ensured == []  # already pulled — on_running is a no-op

    # Crash the container → _launch replaces it with a fresh one.
    runtime.live.pop(first_cid)
    for _ in range(5):
        job = await queries.get_service_by_name(MODEL_NAME)
        if job.status is JobStatus.running and job.container_id in runtime.live:
            break
        await controller.reconcile()
    assert job.container_id != first_cid
    # THE pin: model_pulled survives the relaunch untouched.
    assert json.loads(job.config)["model_pulled"] is True
    await controller.shutdown()


async def test_launch_env_not_ready_acquires_nothing(queries, sample_gpus, monkeypatch):
    """``LaunchEnvNotReady`` acquires NOTHING — no GPU, no host port, no status
    write, no terminal stamp — the retry-next-tick contract (P14 WP-0 C2.1),
    re-pinned before the ``_launch`` decomposition (WP17 PR2) so a phase split
    cannot accidentally acquire resources ahead of the env-resolve gate.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)  # no model_controller wired
    app = _ai_app("app", gpu_count=1)
    await queries.create_job(app)

    gpu_calls: list = []
    port_calls: list = []
    real_allocate = queries.allocate_gpus
    real_acquire = queries.acquire_service_port

    async def _rec_allocate(*a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        gpu_calls.append((a, kw))
        return await real_allocate(*a, **kw)

    async def _rec_acquire(*a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        port_calls.append((a, kw))
        return await real_acquire(*a, **kw)

    monkeypatch.setattr(queries, "allocate_gpus", _rec_allocate)
    monkeypatch.setattr(queries, "acquire_service_port", _rec_acquire)

    # Binding NOT ready: the model it needs is not yet served anywhere.
    await controller.reconcile()
    assert gpu_calls == []
    assert port_calls == []
    app_row = await queries.get_service_by_name("app")
    assert app_row.status is JobStatus.building  # untouched — no write at all
    assert await queries.get_job_gpus(app.id) == []
    assert await queries.get_service_endpoint("app") is None

    # Make the binding ready: the model it needs is now served.
    model = _model_job(pulled=True)
    await queries.create_job(model)
    await queries.acquire_service_port(MODEL_NAME, model.id, 11434, (9400, 9499))
    await queries.update_job_status(model.id, JobStatus.running, container_id="cm")
    runtime.live["cm"] = datetime.now(UTC)

    await controller.reconcile()
    assert gpu_calls  # the retry proceeds once the binding resolves
    app_row = await queries.get_service_by_name("app")
    assert app_row.status is JobStatus.running
    await controller.shutdown()


async def test_launch_on_launched_ordered_before_running(queries, tmp_path, monkeypatch):
    """R-B3: ``DataController.on_launched`` (the C5 clear) must be awaited
    BEFORE the row's status flips to ``running`` — the ordering the settle-tail
    phase function (WP17 PR2's ``settle_started``) must preserve intact.
    """
    from nerdit.core.data import DataController, PostgresBackend
    from nerdit.core.secrets import SecretManager

    runtime = FakeRuntime()
    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("pg", {"POSTGRES_PASSWORD": "minted-pw"})
    data = DataController(
        PostgresBackend(ready_timeout_s=0.5), runtime, queries, data_dir=str(tmp_path)
    )
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        secrets=secrets,
        data_controller=data,
        data_dir=str(tmp_path),
    )
    row = Job(
        name="pg",
        kind=JobKind.database,
        service_name="pg",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps({"backend": "postgres", "image": "postgres:16", "port": 5432}),
    )
    await queries.create_job(row)

    order: list[str] = []
    real_on_launched = data.on_launched
    real_update_status = queries.update_job_status

    async def _rec_on_launched(job, container_id, host_port):  # noqa: ANN001, ANN202
        order.append("on_launched")
        return await real_on_launched(job, container_id, host_port)

    async def _rec_update_status(job_id, status, **kw):  # noqa: ANN001, ANN003, ANN202
        if status is JobStatus.running:
            order.append("running")
        return await real_update_status(job_id, status, **kw)

    monkeypatch.setattr(data, "on_launched", _rec_on_launched)
    monkeypatch.setattr(queries, "update_job_status", _rec_update_status)

    await controller.reconcile()  # fresh container launch

    launched = await queries.get_service_by_name("pg")
    assert launched.status is JobStatus.running
    assert order == ["on_launched", "running"], (
        "on_launched (the C5 clear) must be awaited before the running status write"
    )

    await data.shutdown()
    await controller.shutdown()


# =============================================================================
# (P20) rowless-run registry + bounded one-off container execution
# =============================================================================
#
# Runs and ``[deploy].release`` executions are containers with NO ``jobs`` row
# (D-C). The in-memory registry on the controller is therefore the only thing
# that (a) blocks a concurrent DELETE, (b) enforces the D-P20-2 caps and (c)
# keeps the zombie sweep off the container once it ages past 30 s. Everything
# below is the controller tier: real ``queries``, hand-written FakeRuntime, real
# ServiceController, no HTTP.


def _run_config(image: str = "nerdit-app/demo:1") -> object:
    """A minimal portless run ContainerConfig, as WP3's run_once will build it."""
    from nerdit.core.launch import build_run_container_config

    return build_run_container_config(
        {"image": image},
        ContainerSettings(),
        image=image,
        command=["python", "migrate.py"],
        env={"PORT": "8000"},
        workdir=None,
    )


# --- registry: caps, single-flight, hygiene ----------------------------------


async def test_register_run_is_single_flight_per_service(queries):
    """D-P20-2: a service already executing anything refuses a second run."""
    controller = _controller(queries, FakeRuntime())
    controller._register_run("svc-1", "run-a", is_release=False)

    with pytest.raises(RunPreconditionError) as exc:
        controller._register_run("svc-1", "run-b", is_release=False)
    assert exc.value.reason == "run_in_progress"

    # A run on a DIFFERENT service is unaffected.
    controller._register_run("svc-2", "run-c", is_release=False)
    assert controller.busy_runs() == 2


async def test_register_run_refuses_a_run_while_a_release_holds_the_service(queries):
    """A release counts as an in-flight execution for the single-flight check —
    a run against a service mid-migration is exactly as unsafe."""
    controller = _controller(queries, FakeRuntime())
    controller._register_run("svc-1", "rel-a", is_release=True)

    with pytest.raises(RunPreconditionError) as exc:
        controller._register_run("svc-1", "run-b", is_release=False)
    assert exc.value.reason == "run_in_progress"


async def test_register_run_enforces_the_daemon_wide_cap(queries):
    """``[services].max_concurrent_runs`` caps route-initiated runs across ALL
    services; the cap is snapshotted at construction (restart-required key)."""
    controller = _controller(queries, FakeRuntime(), max_concurrent_runs=2)
    controller._register_run("svc-1", "r1", is_release=False)
    controller._register_run("svc-2", "r2", is_release=False)

    with pytest.raises(RunPreconditionError) as exc:
        controller._register_run("svc-3", "r3", is_release=False)
    assert exc.value.reason == "too_many_runs"

    # Freeing one slot re-opens the cap.
    controller._discard_run("svc-1", "r1")
    controller._register_run("svc-3", "r3", is_release=False)
    assert controller.busy_runs() == 2


async def test_releases_are_exempt_from_both_caps(queries):
    """A deploy must never fail because unrelated one-off runs are in flight —
    releases are bounded by ``max_concurrent_builds`` instead."""
    controller = _controller(queries, FakeRuntime(), max_concurrent_runs=1)
    controller._register_run("svc-1", "r1", is_release=False)  # cap now full

    for i in range(5):
        controller._register_run(f"rel-svc-{i}", f"rel-{i}", is_release=True)
    # A release on a service that ALREADY has a run also goes through.
    controller._register_run("svc-1", "rel-same", is_release=True)

    assert controller.busy_runs() == 7  # busy_runs counts releases


async def test_register_run_raise_leaks_nothing(queries):
    """Security S1: a refused registration must not leave a phantom job key
    behind — an empty dict would make ``has_active_run`` False but keep the
    single-flight check tripping on the next real registration."""
    controller = _controller(queries, FakeRuntime(), max_concurrent_runs=1)
    controller._register_run("svc-1", "r1", is_release=False)

    with pytest.raises(RunPreconditionError):
        controller._register_run("svc-2", "r2", is_release=False)  # global cap
    with pytest.raises(RunPreconditionError):
        controller._register_run("svc-1", "r-dup", is_release=False)  # single flight

    assert controller.has_active_run("svc-2") is False
    assert "svc-2" not in controller._active_runs
    assert set(controller._active_runs["svc-1"]) == {"r1"}


async def test_has_active_run_and_container_ids_track_the_slot_lifecycle(queries):
    """``has_active_run`` is true from registration; ``active_run_container_ids``
    only reports BOUND ids (an unbound slot has no container yet)."""
    controller = _controller(queries, FakeRuntime())

    assert controller.has_active_run("svc-1") is False
    controller._register_run("svc-1", "r1", is_release=False)
    assert controller.has_active_run("svc-1") is True
    assert controller.active_run_container_ids() == set()  # registered, not bound

    controller._bind_run_container("svc-1", "r1", "cid-1")
    assert controller.active_run_container_ids() == {"cid-1"}

    controller._discard_run("svc-1", "r1")
    assert controller.has_active_run("svc-1") is False
    assert controller.active_run_container_ids() == set()
    assert "svc-1" not in controller._active_runs  # the job key is dropped


async def test_discard_run_is_idempotent(queries):
    """It is called from a ``finally``, so a double discard (or one for a slot
    that never existed) must be a no-op, never a KeyError."""
    controller = _controller(queries, FakeRuntime())
    controller._register_run("svc-1", "r1", is_release=False)
    controller._discard_run("svc-1", "r1")
    controller._discard_run("svc-1", "r1")
    controller._discard_run("nope", "never")
    assert controller._active_runs == {}


async def test_bind_run_container_never_resurrects_a_discarded_slot(queries):
    """The caller's ``finally`` can win the race against a slow ``runtime.run``;
    binding afterwards must not re-create the slot (it would wedge DELETE)."""
    controller = _controller(queries, FakeRuntime())
    controller._register_run("svc-1", "r1", is_release=False)
    controller._discard_run("svc-1", "r1")

    controller._bind_run_container("svc-1", "r1", "cid-1")

    assert controller._active_runs == {}
    assert controller.has_active_run("svc-1") is False
    assert controller.active_run_container_ids() == set()


async def test_kill_active_run_containers_kills_bound_slots_only(queries):
    """(P20 WP6) The restart drain's deadline teeth: kill every BOUND run
    container, never remove it (``run_once`` still needs ``logs()`` off the dead
    container for its scrubbed tail), and leave unbound slots alone — there is
    nothing to kill yet, uvicorn's graceful-shutdown timeout backstops those."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._register_run("svc-1", "r1", is_release=False)
    controller._bind_run_container("svc-1", "r1", "c1")
    controller._register_run("svc-2", "r2", is_release=False)  # left UNBOUND

    assert await controller.kill_active_run_containers() == 1
    assert runtime.killed == ["c1"]
    assert "remove" not in runtime.calls  # kill-only


async def test_kill_active_run_containers_survives_a_missing_container(queries):
    """Best-effort per container: an already-gone one (``ContainerNotFoundError``)
    is skipped without aborting the drain, and nothing escapes."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    controller._register_run("svc-1", "r1", is_release=False)
    controller._bind_run_container("svc-1", "r1", "gone")
    controller._register_run("svc-2", "r2", is_release=False)
    controller._bind_run_container("svc-2", "r2", "c2")

    real_kill = runtime.kill

    async def kill(container_id: str) -> None:
        if container_id == "gone":
            raise ContainerNotFoundError("no such container")
        await real_kill(container_id)

    runtime.kill = kill

    assert await controller.kill_active_run_containers() == 1
    assert runtime.killed == ["c2"]


async def test_claim_and_release_run_slot_counter(queries):
    """The per-token pending counter backing the route's quota claim: the claim
    returns the count INCLUDING itself, release floors at 0 and drops the key."""
    controller = _controller(queries, FakeRuntime())

    assert controller.claim_run_slot("tok-a") == 1
    assert controller.claim_run_slot("tok-a") == 2
    assert controller.claim_run_slot("tok-b") == 1

    controller.release_run_slot("tok-a")
    assert controller._run_counts["tok-a"] == 1
    controller.release_run_slot("tok-a")
    assert "tok-a" not in controller._run_counts
    controller.release_run_slot("tok-a")  # unknown token → no-op, no KeyError
    controller.release_run_slot("tok-b")
    assert controller._run_counts == {}


# --- _execute_container_once --------------------------------------------------


async def test_execute_container_once_happy_path(queries):
    """Start → bounded wait → post-mortem → bounded tail → force-remove."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["migrating", "done"]
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job,
            run_id="run-1",
            config=_run_config(),
            timeout_s=42,
            log_tail=50,
            scrub_values=[],
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert result.run_id == "run-1"
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.oom_killed is False
    assert result.log_tail == ["migrating", "done"]
    assert result.duration_s >= 0.0
    # The budget really reached the runtime: 42 s wait, 50-line + 16 KiB tail.
    assert runtime.wait_calls == [("c1", 42)]
    assert runtime.log_calls == [("c1", False, 50, _RUN_TAIL_MAX_BYTES)]
    # Forensics BEFORE teardown, and the container is always force-removed.
    assert runtime.calls.index("inspect_state") < runtime.calls.index("remove")
    assert "c1" not in runtime.live


async def test_execute_container_once_binds_the_container_for_the_sweep(queries):
    """The container id must be visible to ``active_run_container_ids`` while the
    run executes — that set is the zombie sweeper's only protection for it."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    seen: list[set[str]] = []

    async def _spy_wait(container_id, timeout_s=None):
        seen.append(controller.active_run_container_ids())
        return 0

    controller = _controller(queries, runtime)
    runtime.wait = _spy_wait  # type: ignore[assignment]
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=5, log_tail=10, scrub_values=[]
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert seen == [{"c1"}]
    assert controller.active_run_container_ids() == set()


async def test_execute_container_once_timeout_kills_and_falls_back_to_inspect(queries):
    """A run past its server-enforced cap is killed, flagged, and still reports
    the exit code the kill produced (``wait`` never returned one)."""
    runtime = FakeRuntime()
    runtime.wait_error = TimeoutError("bounded wait expired")
    runtime.exit_code = 137
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=1, log_tail=10, scrub_values=[]
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert result.timed_out is True
    assert result.exit_code == 137  # from the post-mortem inspect, not from wait
    assert runtime.calls.index("kill") < runtime.calls.index("inspect_state")
    assert runtime.calls.index("inspect_state") < runtime.calls.index("remove")
    assert controller.has_active_run(job.id) is False


async def test_execute_container_once_reports_oom(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 137
    runtime.oom_killed = True
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=5, log_tail=10, scrub_values=[]
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert result.oom_killed is True
    assert result.exit_code == 137


async def test_execute_container_once_tail_is_scrubbed_and_line_truncated(queries):
    """D-P20-1: there is exactly ONE scrub choke point, and it is here — the
    response, ``config['last_run']`` and a release's ``job_logs`` all consume
    this list. Oversized lines are clamped on top of the runtime byte budget."""
    secret = "super-secret-password"
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = [
        f"connecting with {secret}",
        f"dsn=postgres://u:{secret}@db/app",
        "x" * (_RUN_LINE_MAX_BYTES + 500),
    ]
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job,
            run_id="run-1",
            config=_run_config(),
            timeout_s=5,
            log_tail=10,
            scrub_values=[secret],
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert secret not in "\n".join(result.log_tail)
    assert result.log_tail[0] == "connecting with ***"
    assert result.log_tail[1] == "dsn=postgres://u:***@db/app"
    assert result.log_tail[2].endswith("…[truncated]")
    assert len(result.log_tail[2].encode()) <= _RUN_LINE_MAX_BYTES + len("…[truncated]".encode())


async def test_execute_container_once_tolerates_a_failed_log_read(queries):
    """A completed run must not be reported as a runtime failure just because
    its tail could not be read (the route would surface 503)."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_error = ContainerRuntimeError("log stream gone")
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=5, log_tail=10, scrub_values=[]
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert result.exit_code == 0
    assert result.log_tail == []
    assert "remove" in runtime.calls


async def test_execute_container_once_removes_the_container_when_the_run_path_raises(queries):
    """The ``finally`` teardown holds even when the wait blows up mid-run —
    otherwise a rowless container is leaked the moment the slot is discarded."""
    runtime = FakeRuntime()
    runtime.wait_error = ContainerRuntimeError("docker went away")
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        with pytest.raises(ContainerRuntimeError):
            await controller._execute_container_once(
                job,
                run_id="run-1",
                config=_run_config(),
                timeout_s=5,
                log_tail=10,
                scrub_values=[],
            )
    finally:
        controller._discard_run(job.id, "run-1")

    assert "remove" in runtime.calls
    assert "c1" not in runtime.live
    assert controller.has_active_run(job.id) is False


async def test_execute_container_once_leaves_the_slot_unbound_when_run_fails(queries):
    """Security S1 contract: ``_execute_container_once`` BINDS but never
    DISCARDS — a raise out of ``runtime.run()`` leaves the (unbound) slot for
    the caller's ``finally`` to release. Nothing is bound, so the sweep hook
    reports nothing to protect."""
    runtime = FakeRuntime()
    runtime.run_error = ContainerRuntimeError("no such image")
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    with pytest.raises(ContainerRuntimeError):
        await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=5, log_tail=10, scrub_values=[]
        )

    assert controller.has_active_run(job.id) is True  # still the caller's to release
    assert controller.active_run_container_ids() == set()
    controller._discard_run(job.id, "run-1")
    assert controller.has_active_run(job.id) is False


async def test_execute_container_once_writes_no_job_logs(queries):
    """D-P14-6: run stdout never reaches ``job_logs`` — it is returned to the
    owner and mirrored into ``config['last_run']``, nowhere else."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["chatty", "output"]
    controller = _controller(queries, runtime)
    job = _svc("svc")
    await queries.create_job(job)

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        await controller._execute_container_once(
            job, run_id="run-1", config=_run_config(), timeout_s=5, log_tail=10, scrub_values=[]
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert await queries.get_logs(job.id) == []


# --- the zombie sweep sees the registry --------------------------------------


async def test_zombie_sweep_spares_a_registered_run_container(queries):
    """End-to-end of the P20 hook at the controller tier: a rowless run
    container older than the age gate survives only because the registry
    reports it."""
    runtime = FakeRuntime()
    runtime.live["run-cid"] = datetime.now(UTC) - timedelta(seconds=120)
    runtime.live["orphan-cid"] = datetime.now(UTC) - timedelta(seconds=120)
    controller = _controller(queries, runtime)
    controller._register_run("svc-1", "r1", is_release=False)
    controller._bind_run_container("svc-1", "r1", "run-cid")

    sweeper = ZombieSweeper(
        queries=queries,
        runtime=runtime,
        extra_protected=controller.active_run_container_ids,
    )
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    assert "run-cid" in runtime.live
    assert "orphan-cid" not in runtime.live

    # ...and once the slot is discarded the same container IS reaped.
    controller._discard_run("svc-1", "r1")
    assert await sweeper.cleanup_zombies() == 1
    assert "run-cid" not in runtime.live


# --- (P21 WP1) crash-tail capture + gpu_oom classification --------------------

_CUDA_TAIL = [
    "INFO 07-12 10:00:00 llm_engine.py:73] Initializing an LLM engine",
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
]


async def _launch_then_crash(controller, queries, runtime, name: str = "svc"):
    """Drive one launch → crash cycle and return the post-crash row."""
    await controller.reconcile()  # launch → running
    job = await queries.get_service_by_name(name)
    assert job.status is JobStatus.running
    runtime.live.pop(job.container_id, None)
    await controller.reconcile()  # crash tick
    return await queries.get_service_by_name(name)


def _stream(entries, stream: LogStream) -> list[str]:
    return [e.message for e in entries if e.stream is stream]


async def test_crash_tail_captured_before_remove(queries):
    runtime = FakeRuntime()
    runtime.log_lines = list(_CUDA_TAIL)
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    await controller.reconcile()  # launch → running
    job = await queries.get_service_by_name("svc")
    cid = job.container_id
    runtime.calls.clear()
    runtime.log_calls.clear()
    runtime.live.pop(cid)
    await controller.reconcile()  # crash tick

    entries = await queries.get_logs(job.id)
    assert _stream(entries, LogStream.crash) == _CUDA_TAIL
    # The capture window is load-bearing: after inspect_state, before remove.
    assert runtime.calls == ["inspect_state", "logs", "remove"]
    # Bounded twice at READ time (D1).
    one_shot = [c for c in runtime.log_calls if not c[1]]
    assert one_shot == [(cid, False, 100, 64 * 1024)]
    await controller.shutdown()


async def test_crash_tail_replaced_not_appended(queries):
    runtime = FakeRuntime()
    runtime.log_lines = ["first crash line"]
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert _stream(await queries.get_logs(job.id), LogStream.crash) == ["first crash line"]
    system_before = _stream(await queries.get_logs(job.id), LogStream.system)
    assert system_before  # the restart bookkeeping line

    runtime.log_lines = ["second crash line A", "second crash line B"]
    await controller.reconcile()  # backoff cleared → relaunch
    job = await queries.get_service_by_name("svc")
    assert job.status is JobStatus.running
    runtime.live.pop(job.container_id)
    await controller.reconcile()  # second crash

    entries = await queries.get_logs(job.id)
    # Delete-and-replace: only the latest crash's capture survives...
    assert _stream(entries, LogStream.crash) == ["second crash line A", "second crash line B"]
    # ...and the other streams are untouched (they only grow).
    assert _stream(entries, LogStream.system)[: len(system_before)] == system_before
    await controller.shutdown()


async def test_crash_tail_empty_read_keeps_previous(queries):
    runtime = FakeRuntime()
    runtime.log_lines = ["first crash line"]
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert _stream(await queries.get_logs(job.id), LogStream.crash) == ["first crash line"]

    # A failed read on the SECOND crash must keep the first capture rather than
    # blanking it (the _persist_forensics lost-wait semantics).
    from nerdit.core.runtime.protocol import ContainerRuntimeError

    runtime.log_error = ContainerRuntimeError("log stream gone")
    await controller.reconcile()  # relaunch
    job = await queries.get_service_by_name("svc")
    runtime.live.pop(job.container_id)
    await controller.reconcile()  # second crash — capture fails

    assert _stream(await queries.get_logs(job.id), LogStream.crash) == ["first crash line"]
    await controller.shutdown()


async def test_crash_tail_empty_success_clears_previous(queries):
    runtime = FakeRuntime()
    runtime.log_lines = ["first crash line"]
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert _stream(await queries.get_logs(job.id), LogStream.crash) == ["first crash line"]

    # Latest-crash contract (PR #102 review): crash #2 printing NOTHING is a
    # successful read, and must clear crash #1's capture — /diagnose would
    # otherwise pair crash #1's tail with crash #2's forensics flags.
    runtime.log_lines = []
    await controller.reconcile()  # relaunch
    job = await queries.get_service_by_name("svc")
    runtime.live.pop(job.container_id)
    await controller.reconcile()  # second crash — read succeeds, zero lines

    assert _stream(await queries.get_logs(job.id), LogStream.crash) == []
    await controller.shutdown()


async def test_crash_tail_scrubbed_before_persistence(queries):
    # (PR #102 review) The container ran with the full resolved env, and
    # job_logs is readable by any authenticated principal — a credential
    # printed just before the crash goes through the D-P20-1 pipeline.
    runtime = FakeRuntime()
    runtime.log_lines = ["boom: token is sekritvalue123 and more"]
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    row = _svc("svc")
    row.config = '{"image": "demo:latest", "port": 8000, "env": {"API_KEY": "sekritvalue123"}}'
    await queries.create_job(row)

    job = await _launch_then_crash(controller, queries, runtime)
    captured = _stream(await queries.get_logs(job.id), LogStream.crash)
    assert captured == ["boom: token is *** and more"]
    await controller.shutdown()


async def test_crash_tail_line_clamped(queries):
    runtime = FakeRuntime()
    runtime.log_lines = ["x" * (_RUN_LINE_MAX_BYTES + 500)]
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    await queries.create_job(_svc("svc"))

    job = await _launch_then_crash(controller, queries, runtime)
    captured = _stream(await queries.get_logs(job.id), LogStream.crash)
    assert len(captured) == 1
    assert captured[0].endswith("…[truncated]")
    assert len(captured[0].encode()) < _RUN_LINE_MAX_BYTES + 100
    await controller.shutdown()


async def test_gpu_oom_terminal_classification(queries):
    # restart_policy='no' → the terminal branch of _handle_crash.
    runtime = FakeRuntime()
    runtime.exit_code = 1
    runtime.oom_killed = False
    runtime.log_lines = list(_CUDA_TAIL)
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert job.status is JobStatus.failed
    assert job.error_class is ErrorClass.gpu_oom
    cfg = json.loads(job.config)
    assert cfg["gpu_oom"] is True
    assert cfg["oom_killed"] is False
    await controller.shutdown()


async def test_gpu_oom_budget_exhausted(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 1
    runtime.log_lines = list(_CUDA_TAIL)
    controller = _controller(queries, runtime, service_max_restarts=1)
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]
    # A last_deploy generation so the crash_loop phase stamp fires too.
    job = _svc("svc")
    job.config = json.dumps(
        {
            "image": "demo:latest",
            "port": 8000,
            "last_deploy": {"version": 2, "action": "create", "phase": "launching"},
        }
    )
    await queries.create_job(job)

    job = await queries.get_service_by_name("svc")
    for _ in range(40):
        job = await queries.get_service_by_name("svc")
        if job.status is JobStatus.failed:
            break
        if job.status is JobStatus.running and job.container_id in runtime.live:
            runtime.live.pop(job.container_id)
        await controller.reconcile()

    assert job.status is JobStatus.failed
    assert job.error_class is ErrorClass.gpu_oom
    cfg = json.loads(job.config)
    assert cfg["last_deploy"]["error_class"] == "GPU_OOM"
    await controller.shutdown()


async def test_cgroup_oom_still_oom(queries):
    # A real cgroup kill with NO CUDA marker keeps mapping to ErrorClass.oom.
    runtime = FakeRuntime()
    runtime.exit_code = 137
    runtime.oom_killed = True
    runtime.log_lines = ["worker exited", "out of memory"]
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert job.error_class is ErrorClass.oom
    assert json.loads(job.config)["gpu_oom"] is False
    await controller.shutdown()


async def test_plain_exit1_still_user_error(queries):
    runtime = FakeRuntime()
    runtime.exit_code = 1
    runtime.log_lines = ["ModuleNotFoundError: no module named 'app'"]
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert job.error_class is ErrorClass.user_error
    assert json.loads(job.config)["gpu_oom"] is False
    await controller.shutdown()


async def test_priv_denied_is_persisted_from_the_crash_tail(queries):
    """(P33) The nginx field failure: the flag is stamped, the class is unchanged.

    ``priv_denied`` is a REMEDIATION signal, not an error class — the container
    really did exit 1 by its own hand, so ``user_error`` stays right and only
    /diagnose's advice changes.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 1
    runtime.log_lines = [
        "/docker-entrypoint.sh: Configuration complete; ready for start up",
        'chown("/var/cache/nginx/client_temp", 101) failed (1: Operation not permitted)',
    ]
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    job = await _launch_then_crash(controller, queries, runtime)
    cfg = json.loads(job.config)
    assert cfg["priv_denied"] is True
    assert cfg["gpu_oom"] is False
    assert job.error_class is ErrorClass.user_error
    await controller.shutdown()


async def test_priv_denied_false_on_an_ordinary_crash(queries):
    """Every crash overwrites the flag, so a later unrelated crash clears it."""
    runtime = FakeRuntime()
    runtime.exit_code = 1
    runtime.log_lines = ["ModuleNotFoundError: no module named 'app'"]
    controller = _controller(queries, runtime)
    await queries.create_job(_svc("svc", restart_policy="no"))

    job = await _launch_then_crash(controller, queries, runtime)
    assert json.loads(job.config)["priv_denied"] is False
    await controller.shutdown()


@pytest.mark.parametrize(
    "line, expected",
    [
        ("torch.OutOfMemoryError: CUDA out of memory", True),
        ("TORCH.CUDA.OUTOFMEMORYERROR: boom", True),
        ("RuntimeError: HIP out of memory", True),
        ("hipErrorOutOfMemory", True),
        ("cuda out of memory. Tried to allocate 2 GiB", True),
        (
            # vLLM v0.25 preflight refusal, verbatim from a live 2026-08-06 crash
            "ValueError: Free memory on device cuda:0 (0.03/7.62 GiB) on startup "
            "is less than desired GPU memory utilization (0.95, 7.24 GiB). "
            "Decrease GPU memory utilization or reduce GPU memory used by other processes.",
            True,
        ),
        ("Killed: out of memory", False),
        ("everything is fine", False),
        ("", False),
    ],
)
async def test_unit_tail_indicates_gpu_oom(line, expected):
    from nerdit.core.services import _tail_indicates_gpu_oom

    assert _tail_indicates_gpu_oom([line]) is expected


async def test_unit_tail_indicates_gpu_oom_empty_tail():
    from nerdit.core.services import _tail_indicates_gpu_oom

    assert _tail_indicates_gpu_oom([]) is False


# --- (P33) dropped-privilege refusal on the same captured tail ----------------


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            # The field failure, verbatim from the 2026-08-23 crash: a stock
            # ``FROM nginx`` static site under cap_drop=ALL.
            'chown("/var/cache/nginx/client_temp", 101) failed (1: Operation not permitted)',
            True,
        ),
        ("chmod() failed (1: Operation not permitted)", True),
        ("setgid(101) failed: Operation not permitted", True),
        ("su-exec: setgroups: Operation not permitted", True),
        ("capset failed: Operation not permitted", True),
        ("nginx: [emerg] can't open /var/run/nginx.pid: Permission denied", True),
        ("s6-svscan: warning: unable to open /run/service: Permission denied", True),
        # Two markers are required on BOTH shapes, so neither half alone fires.
        ("mount: Operation not permitted", False),
        ("chown: changing ownership of '/app/data': Read-only file system", False),
        # An app failing on its OWN tree is ordinary user error → rule 5.
        ("EACCES: permission denied, open '/app/data/db.sqlite'", False),
        ("PermissionError: [Errno 13] Permission denied: 'output.txt'", False),
        # (Codex, PR #138) The root markers are FILESYSTEM ROOTS: a path that
        # merely *contains* one deeper in its own tree is the app's bug, not the
        # sandbox's. A substring test matched all three of these.
        ("PermissionError: [Errno 13] Permission denied: '/app/run/state.pid'", False),
        ("EACCES: permission denied, open '/home/app/var/cache/x'", False),
        ('open("/srv/etc/config") failed: Permission denied', False),
        # ...while a real root-owned path still fires from every token position.
        ('open("/var/run/nginx.pid") failed (13: Permission denied)', True),
        ("PermissionError: [Errno 13] Permission denied: '/var/lib/nginx/tmp'", True),
        ("/usr/local/share/x: Permission denied", True),
        ("Killed: out of memory", False),
        ("everything is fine", False),
        ("", False),
    ],
)
async def test_unit_tail_indicates_priv_denied(line, expected):
    from nerdit.core.services import _tail_indicates_priv_denied

    assert _tail_indicates_priv_denied([line]) is expected


async def test_unit_tail_indicates_priv_denied_empty_tail():
    from nerdit.core.services import _tail_indicates_priv_denied

    assert _tail_indicates_priv_denied([]) is False


# --- (P21 WP2 threading) GpuPlacement.min_memory_mb ---------------------------


async def test_unit_plan_gpu_placement_min_memory_mb():
    from nerdit.core.launch import plan_gpu_placement
    from nerdit.db.models import Gpu, GpuVendor

    def _gpu(gid: str, mb: int) -> Gpu:
        return Gpu(id=gid, name="rtx", memory_mb=mb, vendor=GpuVendor.nvidia, index=0)

    placement = plan_gpu_placement([_gpu("g1", 24576), _gpu("g2", 8192)], 2, None)
    assert placement is not None
    assert placement.min_memory_mb == 8192

    # An unknown/zero size is excluded; all-unknown degrades to None (no injection).
    unknown = plan_gpu_placement([_gpu("g1", 0)], 1, None)
    assert unknown is not None
    assert unknown.min_memory_mb is None

    # The field is defaulted, so a 3-field construction still type-checks.
    from nerdit.core.launch import GpuPlacement

    assert GpuPlacement(["a"], ["0"], GpuVendor.nvidia).min_memory_mb is None


def _spy_build_config(mc, captured: dict):
    """Replace ``build_container_config`` with a capturing stub.

    The launch branch is what is under test here, not the backend's argv
    assembly (that is pinned in ``test_model_backend.py``), so the stub returns a
    minimal config instead of delegating.
    """
    from nerdit.db.models import ContainerConfig

    def _stub(model, gpu_ids, vendor, host_port, **kw):
        captured.update(kw)
        captured["model"] = model
        captured["gpu_ids"] = list(gpu_ids)
        return ContainerConfig(image="ollama/ollama", name="ollama-demo")

    mc.build_container_config = _stub  # type: ignore[method-assign]


async def test_model_launch_threads_gpu_memory_and_overrides(queries, tmp_path, monkeypatch):
    """The model launch branch passes the placement VRAM + the row's per-serve
    overrides through to ``ModelController.build_container_config`` (P21 D3/D4)."""
    from nerdit.core.launch import GpuPlacement
    from nerdit.db.models import GpuVendor

    runtime = FakeRuntime()
    mc, _ = _mc(queries, runtime, tmp_path)
    captured: dict = {}
    _spy_build_config(mc, captured)
    controller = _controller(queries, runtime, model_controller=mc)

    monkeypatch.setattr(
        "nerdit.core.services.plan_gpu_placement",
        lambda candidates, count, requested: GpuPlacement(
            ["g1"], ["0"], GpuVendor.nvidia, min_memory_mb=8192
        ),
    )

    async def _noop_allocate(job_id, gpu_ids, exclusive=False):
        return None

    monkeypatch.setattr(queries, "allocate_gpus", _noop_allocate)

    await queries.create_job(
        Job(
            name="m1",
            kind=JobKind.model,
            service_name="m1",
            gpu_count=1,
            status=JobStatus.building,
            desired_state="running",
            config=json.dumps(
                {
                    "model": "demo",
                    "backend": "ollama",
                    "image": "ollama/ollama",
                    "max_model_len": 2048,
                    "gpu_memory_utilization": 0.5,
                    "port": 11434,
                }
            ),
        )
    )
    await controller.reconcile()
    await _settle_models(mc)

    assert captured["gpu_memory_mb"] == 8192
    assert captured["max_model_len"] == 2048
    assert captured["gpu_memory_utilization"] == 0.5
    await controller.shutdown()


async def test_model_launch_without_gpu_passes_no_vram(queries, tmp_path):
    """gpu_count == 0 short-circuits placement, so no VRAM is claimed (D3), and a
    row with no per-serve overrides threads ``None`` for both (not "None")."""
    runtime = FakeRuntime()
    mc, _ = _mc(queries, runtime, tmp_path)
    captured: dict = {}
    _spy_build_config(mc, captured)
    controller = _controller(queries, runtime, model_controller=mc)

    await queries.create_job(_model_job())
    await controller.reconcile()
    await _settle_models(mc)

    assert captured["gpu_memory_mb"] is None
    assert captured["max_model_len"] is None
    assert captured["gpu_memory_utilization"] is None
    await controller.shutdown()
