"""Test health-gated cutover with real controllers, database and ProxyManager.

FakeRuntime and FakeCaddy isolate infrastructure, but keep ProxyManager real to
verify dial convergence rather than merely recording register calls.

Pin active-port update before proxy registration, dial read-back before promotion,
durable records before the final pending-map pop, and blue destruction afterward.
Failed or unreadable registration preserves blue. Uncommitted crash recovery
reverts and restores blue's recorded port; committed recovery replays the tail at least
once without destroying containers. Ineligible apps keep the same-port swap.
Patch sleep and monotonic time for deterministic budgets.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from nerdit.config.settings import ContainerSettings, ServicesSettings
from nerdit.core import cutover as cutover_mod
from nerdit.core import eventlog
from nerdit.core import launch as launch_mod
from nerdit.core.cutover import _eligible, _reserve_ephemeral_port
from nerdit.core.launch import AppContainerSpec, build_app_container_config
from nerdit.core.proxy import LiveRoute
from nerdit.core.services import LaunchEnvNotReady, ServiceController
from nerdit.core.sweeper import ZombieSweeper
from nerdit.db.models import ErrorClass, GpuVendor, JobKind, JobStatus
from tests.test_proxy import FakeCaddy, _proxy
from tests.test_services_reconcile import FakeRuntime, _deploy_svc

pytestmark = pytest.mark.asyncio

BLUE = "c-blue"
NAME = "app"


# --- fakes / harness ----------------------------------------------------------


class CutoverRuntime(FakeRuntime):
    """``FakeRuntime`` + a gate on the green launch and an image-removal seam."""

    def __init__(self, *, run_gate: asyncio.Event | None = None) -> None:
        super().__init__()
        self.run_gate = run_gate
        self.removed_images: list[str] = []
        self.ever_ran: set[str] = set()

    async def run(self, config) -> str:
        if self.run_gate is not None:
            await self.run_gate.wait()
        cid = await super().run(config)
        self.ever_ran.add(cid)
        return cid

    async def status(self, container_id: str) -> str | None:
        """Model the REAL docker contract, unlike the base fake.

        A stopped-but-present container reports the truthy string
        ``"exited"``, never ``None`` — the P24b live run caught
        ``_is_container_live``'s truthiness check reading exactly that corpse
        as live, so the fake must be able to produce it.
        """
        if container_id in self.live:
            return "running"
        if container_id in self.ever_ran:
            return "exited"
        return None

    async def remove_image(self, tag: str, force: bool = False) -> None:
        self.removed_images.append(tag)


class RecordingEvents:
    """A stand-in for :class:`~nerdit.core.eventlog.EventRecorder`.

    ``record_job_event`` only ever calls ``.record(type, **kw)``, so a list is
    enough — and it keeps the durable-feed assertions (at-least-once across a
    crash) free of DB plumbing.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    async def record(self, type, **kw) -> None:  # noqa: A002 — the locked field name
        self.rows.append((type, kw))

    def types(self) -> list[str]:
        return [t for t, _ in self.rows]

    def count(self, type_: str) -> int:
        return sum(1 for t, _ in self.rows if t == type_)


@pytest.fixture(autouse=True)
def virtual_clock(monkeypatch):
    """Instant sleeps that advance a virtual monotonic clock.

    Budgets stay real (they come from ``[services]``); only the waiting is
    free, so a 90 s verify budget costs 90 loop iterations, not 90 seconds.
    """
    state = {"t": 0.0}

    async def _sleep(seconds: float) -> None:
        state["t"] += seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(cutover_mod, "_SLEEP", _sleep)
    monkeypatch.setattr(cutover_mod, "_MONOTONIC", lambda: state["t"])
    return state


def _controller(queries, runtime, *, proxy=None, tmp_path=None, events=None, **kw):
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499", **kw),
        container_settings=ContainerSettings(),
        proxy=proxy,
        data_dir=str(tmp_path) if tmp_path is not None else None,
        events=events,
    )


def _cutover_svc(
    name: str = NAME,
    *,
    version: int = 2,
    phase: str = "building",
    container_id: str | None = BLUE,
    health: bool = True,
    status: JobStatus = JobStatus.restarting,
    **cfg_extra,
):
    """A redeploy row over a still-live blue, eligible for a cutover."""
    return _deploy_svc(
        name,
        version=version,
        action="redeploy",
        phase=phase,
        status=status,
        container_id=container_id,
        health_check={"path": "/health"} if health else None,
        previous_image=f"nerdit-app/{name}:{version - 1}",
        **cfg_extra,
    )


async def _seed(queries, job, *, blue_live: bool = True, runtime=None):
    """Create the row + its durable endpoint, and make blue live."""
    await queries.create_job(job)
    endpoint = await queries.acquire_service_port(job.service_name, job.id, 8000, (9400, 9499))
    if blue_live and runtime is not None and job.container_id:
        runtime.live[job.container_id] = datetime.now(UTC)
    return endpoint


async def _drain_cutovers(controller) -> None:
    for task in list(controller._cutover_tasks.values()):
        await task


async def _await_green(controller, timeout: float = 5.0) -> str:
    """Wait (real time) for the verify task to register its green container.

    A bare ``sleep(0)``-yield loop is NOT enough: the verify task's DB awaits
    round-trip through the aiosqlite worker thread, which event-loop yields do
    not wait for — 20 yields pass before the thread returns on a slow CI
    runner, the registry is still empty, and ``next(iter(set()))`` raises
    ``StopIteration`` inside the coroutine (the 3.11/3.12 CI red).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ids = controller.cutover_container_ids()
        if ids:
            return next(iter(ids))
        await asyncio.sleep(0.005)
    pytest.fail("the verify task never registered its green container")


async def _await_promotion(queries, job_id, *, popped: bool = False, timeout: float = 5.0) -> str:
    """Wait (real time) for the verify task to reach its commit point.

    The ``_await_green`` idiom applied to the DB side: the promotion lands
    through the aiosqlite worker thread, which event-loop yields do not wait
    for. With ``popped`` the wait also covers the commit tail's marker pop, so
    the caller lands in the f6-f8 window rather than inside the tail.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = await queries.get_job(job_id)
        promoted = row is not None and bool(row.container_id) and row.container_id != BLUE
        if promoted and (not popped or "cutover_pending" not in json.loads(row.config)):
            return row.container_id
        await asyncio.sleep(0.005)
    pytest.fail("the verify never reached its commit point")


def _park_commit_tail(controller) -> asyncio.Event:
    """Hold the verify INSIDE the f4-f6 commit tail — the f3-f6 window.

    At the park the row is promoted to the green, the marker is still armed and
    the green is still in the registry: the exact state layer 4b resumes into.
    Only the FIRST call parks; later ones delegate to the real tail, so a cancel
    that replays it runs to completion instead of deadlocking on the same event.
    """
    hold = asyncio.Event()
    manager = controller._cutover
    orig = manager._complete_commit_tail
    parked = {"once": False}

    async def tail(job, pending):
        if not parked["once"]:
            parked["once"] = True
            await hold.wait()
        await orig(job, pending)

    manager._complete_commit_tail = tail  # type: ignore[assignment]
    return hold


def _park_probe(controller) -> asyncio.Event:
    """Hold the verify INSIDE its probe loop — green registered, task parked.

    Deterministic where a gate on ``runtime.run`` is not: that gate parks the
    task BEFORE the green registers, so a fast verify can arm, commit and
    unregister entirely between two polls of the registry (the 3.11/3.12 CI
    red raced exactly that window). A probe that awaits an Event cannot pass,
    cannot time out (the budget clock only advances through the patched
    ``_SLEEP``), and sits with the green already in the registry. Releasing
    the event makes every later probe report 500, so a released task settles
    through the ordinary probe-failure path.
    """
    hold = asyncio.Event()

    async def parked_check(host_port, path, timeout):
        await hold.wait()
        return 500

    controller._check_health = parked_check  # type: ignore[assignment]
    return hold


def _healthy(controller, *, code: int = 200, ports: list[int] | None = None):
    async def _check(host_port, path, timeout):
        if ports is not None:
            ports.append(host_port)
        return code

    controller._check_health = _check  # type: ignore[assignment]


async def _cfg(queries, job_id) -> dict:
    row = await queries.get_job(job_id)
    return json.loads(row.config)


async def _audit_actions(queries) -> list[str]:
    items, _ = await queries.list_audit_log(limit=200)
    return [item.action for item in items]


def _dials(fake: FakeCaddy) -> list[str]:
    out = []
    for route in fake.routes:
        handles = route.get("handle") or []
        for handle in handles:
            for sub in handle.get("routes", [handle]):
                for inner in sub.get("handle", [sub]):
                    for upstream in inner.get("upstreams") or []:
                        out.append(upstream["dial"])
    return out


async def _full_setup(
    queries, tmp_path, *, job=None, events=None, runtime=None, blue_live=True, **settings
):
    """Row + endpoint + real proxy + controller, all wired and ready to tick."""
    runtime = runtime if runtime is not None else CutoverRuntime()
    fake = FakeCaddy()
    proxy = _proxy(queries, fake)
    job = job if job is not None else _cutover_svc()
    endpoint = await _seed(queries, job, blue_live=blue_live, runtime=runtime)
    controller = _controller(
        queries, runtime, proxy=proxy, tmp_path=tmp_path, events=events, **settings
    )
    return controller, runtime, proxy, fake, job, endpoint


# --- 1. the happy path + the full step-(f) call order -------------------------


def _record_step_order(queries, proxy, controller, events) -> tuple[list[str], dict]:
    """Wrap every step-(f) side effect so the ordering lock is one flat sequence."""
    order: list[str] = []
    green_port: dict[str, int] = {}

    orig_active = queries.set_endpoint_active_port
    orig_register = proxy.register
    orig_live = proxy.live_routes
    orig_status = queries.update_job_status
    orig_audit = queries.insert_audit_log
    orig_guarded = queries.update_job_config_guarded
    orig_update_config = queries.update_job_config
    orig_destroy = controller._destroy_container
    orig_record = events.record

    async def active(name, port):
        order.append("active_port" if port is not None else "active_port_clear")
        if port is not None:
            green_port["p"] = port
        await orig_active(name, port)

    async def register(name, port, edge_auth=None, **kw):
        order.append("register")
        await orig_register(name, port, edge_auth, **kw)

    async def live_routes():
        order.append("read_back")
        return await orig_live()

    async def status(job_id, new_status, **kw):
        if kw.get("container_id"):
            order.append("promote")
        await orig_status(job_id, new_status, **kw)

    async def audit(**kw):
        order.append(f"audit:{kw['action']}")
        await orig_audit(**kw)

    async def guarded(job_id, config, *, expect_build_version):
        if "cutover_pending" not in json.loads(config):
            order.append("pop")
        return await orig_guarded(job_id, config, expect_build_version=expect_build_version)

    async def update_config(job_id, config):
        # ``stamp_last_deploy`` lands through here; f5 is the write that flips
        # the phase to ``healthy``. Recorded explicitly so the "pop is LAST"
        # assertion below names every tail step instead of comparing the pop
        # against a max() that includes itself.
        if (json.loads(config).get("last_deploy") or {}).get("phase") == "healthy":
            order.append("stamp_healthy")
        await orig_update_config(job_id, config)

    async def destroy(cid):
        order.append(f"destroy:{cid}")
        await orig_destroy(cid)

    async def record(type, **kw):  # noqa: A002 — the locked field name
        order.append(f"event:{type}")
        await orig_record(type, **kw)

    for owner, attr, wrapper in (
        (queries, "set_endpoint_active_port", active),
        (queries, "update_job_status", status),
        (queries, "insert_audit_log", audit),
        (queries, "update_job_config_guarded", guarded),
        (queries, "update_job_config", update_config),
        (proxy, "register", register),
        (proxy, "live_routes", live_routes),
        (controller, "_destroy_container", destroy),
        (events, "record", record),
    ):
        setattr(owner, attr, wrapper)
    return order, green_port


async def test_step_f_call_order_and_phase_machine(queries, tmp_path):
    """The whole ordering lock, asserted as one recorded sequence."""
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events
    )
    _healthy(controller)
    order, green_port = _record_step_order(queries, proxy, controller, events)

    await controller.reconcile()
    await _drain_cutovers(controller)

    # The lock, edge by edge.
    assert order.index("active_port") < order.index("register")
    assert order.index("register") < order.index("read_back")
    assert order.index("read_back") < order.index("promote")
    assert order.index("promote") < order.index("audit:service.cutover")
    assert order.index("audit:service.cutover") < order.index("pop")
    assert order.index("event:service.cutover_succeeded") < order.index("pop")
    assert order.index("stamp_healthy") < order.index("pop")
    assert order.index("pop") < order.index(f"destroy:{BLUE}")
    # The pop is the LAST step of the commit tail — asserted against every
    # OTHER tail step by name (a max() that includes the pop's own index is a
    # tautology and would pass with the pop moved to the front).
    tail = order[order.index("promote") :]
    assert tail.index("pop") == len(tail) - 1 - tail[::-1].index("pop")  # exactly one pop
    for step in ("audit:service.cutover", "event:service.cutover_succeeded", "stamp_healthy"):
        assert tail.index(step) < tail.index("pop"), f"{step} must precede the pop"

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert row.status is JobStatus.running
    assert row.container_id != BLUE  # promoted to the green
    assert cfg["last_deploy"]["phase"] == "healthy"
    assert "cutover_pending" not in cfg
    # The endpoint keeps its stable identity and points at the green.
    ep = await queries.get_service_endpoint(NAME)
    assert ep.host_port == endpoint.host_port
    assert ep.active_host_port == green_port["p"]
    assert ep.live_port == green_port["p"]
    # Blue is gone, the green is live, and the proxy dials the green.
    assert BLUE not in runtime.live
    assert row.container_id in runtime.live
    assert _dials(fake) == [f"127.0.0.1:{green_port['p']}"]
    assert events.types() == ["service.cutover_started", "service.cutover_succeeded"]


async def test_the_green_probes_the_transient_port_not_the_stable_one(queries, tmp_path):
    controller, runtime, _proxy_mgr, _fake, job, endpoint = await _full_setup(queries, tmp_path)
    probed: list[int] = []
    _healthy(controller, ports=probed)

    await controller.reconcile()
    await _drain_cutovers(controller)

    assert probed  # the probe ran
    assert endpoint.host_port not in probed
    launched = runtime.run_configs[-1]
    assert launched.ports == {8000: probed[0]}
    assert launched.extra_labels == {"nerdit-cutover": job.id}


async def test_reserve_ephemeral_port_is_outside_the_service_range():
    port = _reserve_ephemeral_port()
    assert 1024 < port <= 65535
    assert not (9400 <= port <= 9499)  # never the configured service range


# --- 2. (g') the unverifiable repoint -----------------------------------------


@pytest.mark.parametrize("mode", ["unreadable", "swallowed_raise"])
async def test_caddy_down_at_f2_diverts_to_g_prime_and_never_destroys_blue(queries, tmp_path, mode):
    """The finding-1 pin: ``register`` returning cleanly proves nothing.

    ``unreadable`` makes ``live_routes()`` read ``None`` from the moment
    ``register`` is called; ``swallowed_raise`` makes the upsert raise (which
    ``register`` swallows by construction). Both must behave IDENTICALLY,
    because ``register`` cannot report either failure.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_repoint_timeout_s=3
    )
    _healthy(controller)

    orig_register = proxy.register

    async def register(name, port, edge_auth=None, **kw):
        if mode == "unreadable":
            fake.routes_status = 503  # the admin API goes unreadable right here
        else:
            fake.fail_upserts = 999  # every upsert 503s; register swallows it
        await orig_register(name, port, edge_auth, **kw)

    proxy.register = register  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    # Blue was NEVER destroyed and still owns the row on the stable port.
    assert BLUE in runtime.live
    assert row.container_id == BLUE
    assert row.status is JobStatus.running
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None
    assert ep.live_port == endpoint.host_port
    # The green is destroyed and the image reverted.
    assert cfg["image"] == f"nerdit-app/{NAME}:1"
    assert "cutover_pending" not in cfg
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    # The settle names the stage machine-readably.
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items and items[0].params_redacted["stage"] == "repoint"
    assert items[0].params_redacted["data_rollback"] is False
    failed = [kw for t, kw in events.rows if t == "service.cutover_failed"]
    assert failed and failed[0]["data"] == {"stage": "repoint"}


async def test_g_prime_unwinds_the_pointer_before_destroying_the_green(queries, tmp_path):
    """(g') step order: DB back to P FIRST, only then the green dies."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, cutover_repoint_timeout_s=2
    )
    _healthy(controller)
    order: list[str] = []

    orig_active = queries.set_endpoint_active_port

    async def active(name, port):
        order.append("clear" if port is None else "set")
        await orig_active(name, port)

    queries.set_endpoint_active_port = active  # type: ignore[assignment]

    orig_register = proxy.register

    async def register(name, port, edge_auth=None, **kw):
        fake.routes_status = 503
        await orig_register(name, port, edge_auth, **kw)

    proxy.register = register  # type: ignore[assignment]

    orig_destroy = controller._destroy_container

    async def destroy(cid):
        order.append("destroy_green")
        await orig_destroy(cid)

    controller._destroy_container = destroy  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    assert order == ["set", "clear", "destroy_green"]
    assert BLUE in runtime.live


# --- 3. the read-back asserts the END STATE, not the writer -------------------


async def test_a_dial_converged_by_a_reconcile_tick_still_commits(queries, tmp_path):
    """The inline ``register`` writes nothing; a proxy reconcile converges it."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _healthy(controller)

    async def register(name, port, edge_auth=None, **kw):  # a no-op — the three silent paths
        return None

    proxy.register = register  # type: ignore[assignment]

    orig_live = proxy.live_routes
    reads = {"n": 0}

    async def live_routes():
        reads["n"] += 1
        if reads["n"] == 1:
            # A ProxyManager.reconcile() tick lands between the read-backs and
            # converges the dial off the DB (the COALESCE) all by itself.
            await proxy.reconcile()
        return await orig_live()

    proxy.live_routes = live_routes  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE, "a converged dial must commit, not settle failed"
    assert json.loads(row.config)["last_deploy"]["phase"] == "healthy"


# --- 4. the f1 -> f2 revert hazard (the ordering lock's load-bearing case) ----


async def test_a_reconcile_between_f1_and_f2_converges_to_the_green(queries, tmp_path):
    """With the DB written first, an interleaved reconcile dials the GREEN."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _healthy(controller)
    seen: dict[str, list[str]] = {}

    orig_register = proxy.register

    async def register(name, port, edge_auth=None, **kw):
        # A proxy reconcile tick fires in the gap between (1) and (2).
        await proxy.reconcile()
        seen["dials"] = _dials(fake)
        await orig_register(name, port, edge_auth, **kw)

    proxy.register = register  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    ep = await queries.get_service_endpoint(NAME)
    assert seen["dials"] == [f"127.0.0.1:{ep.active_host_port}"]
    assert seen["dials"] != [f"127.0.0.1:{endpoint.host_port}"]


async def test_the_inverted_order_reverts_the_dial_to_the_stable_port(queries, tmp_path):
    """The NEGATIVE pin: register-first + an interleaved reconcile = blue again.

    Written against a deliberately-wrong ordering (``register`` before the
    ``active_host_port`` write) so the regression the lock prevents can never
    reappear unnoticed. If this ever stops reverting, the ``COALESCE`` in
    ``list_active_service_routes`` has stopped being what the reconcile reads.
    """
    fake = FakeCaddy()
    proxy = _proxy(queries, fake)
    job = _cutover_svc(status=JobStatus.running)
    endpoint = await _seed(queries, job, blue_live=False)
    green_port = endpoint.host_port + 1000

    await proxy.register(NAME, green_port)  # (2) first — the wrong order
    assert _dials(fake) == [f"127.0.0.1:{green_port}"]
    await proxy.reconcile()  # ... and a tick lands before (1)
    assert _dials(fake) == [f"127.0.0.1:{endpoint.host_port}"], (
        "the reconcile recomputes the dial from the DB: with active_host_port "
        "still NULL it reverts to blue, and step (7) would then 502 the route"
    )


# --- 5. (g) the green never verifies ------------------------------------------


async def test_green_never_healthy_destroys_it_and_leaves_blue_untouched(queries, tmp_path):
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_verify_timeout_s=5
    )
    _healthy(controller, code=500)
    await proxy.reconcile()  # blue is routed before we start

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert BLUE in runtime.live
    assert row.container_id == BLUE
    assert cfg["image"] == f"nerdit-app/{NAME}:1"  # reverted
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert "cutover_pending" not in cfg
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None
    assert _dials(fake) == [f"127.0.0.1:{endpoint.host_port}"]
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items and items[0].params_redacted["stage"] == "probe"
    assert events.count("service.cutover_failed") == 1


async def test_a_failed_cutover_also_records_the_deploy_settle(queries, tmp_path):
    """The unwind is a deploy outcome too (P24c live-run find).

    It settles through ``_settle_failed_generation``, which stamps
    ``phase=failed`` outside ``stamp_last_deploy`` — so without the settle's own
    durable emit, a generation that announced ``service.deploy_started`` would
    never announce an end. The resulting PAIR is deliberate and symmetric with
    the success path, which already records ``cutover_succeeded`` AND
    ``deploy_succeeded``. ``deploy_state`` emits through the process singleton
    rather than the controller's recorder, so the same fake is installed there.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_verify_timeout_s=5
    )
    _healthy(controller, code=500)
    eventlog.set_recorder(events)
    try:
        await controller.reconcile()
        await _drain_cutovers(controller)
    finally:
        eventlog.set_recorder(None)

    assert events.count("service.cutover_failed") == 1
    assert events.count("service.deploy_failed") == 1
    settled = next(kw for t, kw in events.rows if t == "service.deploy_failed")
    assert settled["service_name"] == NAME
    assert settled["reason"] == "cutover_failed"
    assert settled["build_version"] == 2  # the generation that failed, not the revert target
    assert settled["data"] == {
        "repo": None,
        "ref": None,
        "sha": None,
        "remediation_code": None,
        "error_class": ErrorClass.user_error.value,
    }


async def test_a_green_that_exits_fails_immediately(queries, tmp_path):
    """A dead candidate must not burn the whole verify budget."""
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries, tmp_path, cutover_verify_timeout_s=900
    )
    probes = {"n": 0}

    async def check(host_port, path, timeout):
        probes["n"] += 1
        # The container dies right after the first probe attempt.
        for cid in list(runtime.live):
            if cid != BLUE:
                runtime.live.pop(cid)
        return 500

    controller._check_health = check  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    assert probes["n"] == 1, "the exit must short-circuit, not wait out the budget"
    # The corpse is still *present* (status "exited", a truthy string — see
    # CutoverRuntime.status): the live-run regression where truthiness read it
    # as alive and burned the whole 900 s budget must stay dead.
    row = await queries.get_job(job.id)
    assert row.container_id == BLUE
    assert json.loads(row.config)["last_deploy"]["reason"] == "cutover_failed"


async def test_no_health_spec_uses_the_grace_window(queries, tmp_path):
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries, tmp_path, job=_cutover_svc(health=False), cutover_grace_s=4
    )

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE
    assert json.loads(row.config)["last_deploy"]["phase"] == "healthy"


# --- 6. layer 4a — daemon restart mid-verify, NOT committed -------------------


async def test_crash_mid_verify_settles_failed_and_clears_the_active_port(queries, tmp_path):
    """4a: revert, reap the labelled orphan, and clear the pointer explicitly.

    The clear is the permanent-502 regression pin: without it the COALESCE
    keeps dialling the dead green, the surviving blue is probed on a dead port
    and flips ``degraded`` — which stays running and never relaunches, so
    ``settle_started``'s clear is never reached.
    """
    runtime = CutoverRuntime()
    job = _cutover_svc(phase="verifying", cutover_pending={"version": 2, "blue": BLUE})
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, job=job, runtime=runtime
    )
    green_port = endpoint.host_port + 500
    await queries.set_endpoint_active_port(NAME, green_port)
    runtime.labeled_containers[("nerdit-cutover", job.id)] = ["c-green"]
    runtime.live["c-green"] = datetime.now(UTC)

    await controller.reconcile()

    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None, "the clear is mandatory — see the docstring"
    routes = await queries.list_active_service_routes()
    assert routes[0].host_port == endpoint.host_port, "must dial blue, never the dead green"
    assert "c-green" in runtime.killed
    assert BLUE in runtime.live
    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert row.container_id == BLUE
    assert cfg["image"] == f"nerdit-app/{NAME}:1"
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert "cutover_pending" not in cfg
    assert "service.cutover_failed" in await _audit_actions(queries)


async def test_an_abandoned_g_prime_unwind_resumes_into_4a(queries, tmp_path):
    """Killed between g'1 and g'3: the clear is a no-op and the settle is identical."""
    job = _cutover_svc(phase="verifying", cutover_pending={"version": 2, "blue": BLUE})
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path, job=job)
    # g'1 already ran: the pointer is back at P, but the green is still alive.
    runtime.labeled_containers[("nerdit-cutover", job.id)] = ["c-green"]
    runtime.live["c-green"] = datetime.now(UTC)

    await controller.reconcile()

    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None
    assert "c-green" in runtime.killed
    assert BLUE in runtime.live
    cfg = await _cfg(queries, job.id)
    assert cfg["last_deploy"]["reason"] == "cutover_failed"


async def test_the_unwinds_restore_a_promoted_blues_own_port(queries, tmp_path):
    """Back-to-back cutovers (the live-run composition find): blue may itself
    be a promoted green on an ephemeral port. Every unwind must restore THAT
    port — clearing to NULL would dial the stable port, where nothing listens,
    and a degraded row never relaunches: a permanent 502."""
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_repoint_timeout_s=2
    )
    _healthy(controller)
    blues_port = endpoint.host_port + 77  # blue serves on an EPHEMERAL port
    await queries.set_endpoint_active_port(NAME, blues_port)

    orig_register = proxy.register

    async def register(name, port, edge_auth=None, **kw):
        fake.routes_status = 503  # the repoint can never confirm -> (g')
        await orig_register(name, port, edge_auth, **kw)

    proxy.register = register  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    # The arm recorded blue's port; the g' unwind restored it, not NULL.
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == blues_port
    routes = await queries.list_active_service_routes()
    assert routes[0].host_port == blues_port, "the dial must go back to where blue LISTENS"
    assert BLUE in runtime.live
    cfg = await _cfg(queries, job.id)
    assert cfg["last_deploy"]["reason"] == "cutover_failed"


async def test_the_crash_settle_restores_a_promoted_blues_own_port(queries, tmp_path):
    """4a with a ``blue_port``-carrying marker restores it (and a legacy marker
    without the field degrades to NULL — the pre-composition behaviour)."""
    blues_port = 9477
    job = _cutover_svc(
        phase="verifying",
        cutover_pending={"version": 2, "blue": BLUE, "blue_port": blues_port},
    )
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path, job=job)
    green_port = endpoint.host_port + 500
    await queries.set_endpoint_active_port(NAME, green_port)
    runtime.labeled_containers[("nerdit-cutover", job.id)] = ["c-green"]
    runtime.live["c-green"] = datetime.now(UTC)

    await controller.reconcile()

    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == blues_port
    assert "c-green" in runtime.killed
    assert BLUE in runtime.live


async def test_cancel_restores_a_promoted_blues_own_port(queries, tmp_path):
    """The shared cancel() unwind restores the marker's ``blue_port`` too."""
    gate = asyncio.Event()
    runtime = CutoverRuntime(run_gate=gate)
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, runtime=runtime
    )
    _healthy(controller)
    blues_port = endpoint.host_port + 77
    await queries.set_endpoint_active_port(NAME, blues_port)

    await controller.reconcile()  # arm; the verify parks on the run gate
    assert controller._cutover_tasks
    marker = (await _cfg(queries, job.id))["cutover_pending"]
    assert marker["blue_port"] == blues_port, "the arm must record blue's serving port"

    await controller.cancel_cutover(job.id)
    gate.set()

    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == blues_port
    assert BLUE in runtime.live


async def test_a_settle_that_dies_keeps_the_marker_armed_for_the_next_tick(queries, tmp_path):
    """(PR #108 / amendment 8) The pop lands AFTER the settle, never before.

    With the old pop-first order, a settle that raised (or a daemon death
    between the pop and the settle) left the row ``restarting`` on the failed
    candidate with no marker — so the next tick re-armed and re-verified the
    very generation that just failed, the automatic re-run D-P20-4's twin
    forbids. Now the marker survives the failed settle and layer 4a finishes
    the job idempotently on the next tick.
    """
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    _healthy(controller, code=500)  # the green can never verify

    orig_settle = controller._builder._settle_failed_generation
    blown = {"n": 0}

    async def settle_once_raises(*a, **kw):
        if blown["n"] == 0:
            blown["n"] += 1
            raise RuntimeError("transient settle failure")
        return await orig_settle(*a, **kw)

    controller._builder._settle_failed_generation = settle_once_raises  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    # The settle blew up mid-way: the marker MUST still be armed.
    cfg = await _cfg(queries, job.id)
    assert "cutover_pending" in cfg, "the pop must never precede the settle"

    await controller.reconcile()  # layer 4a picks the armed marker up

    cfg = await _cfg(queries, job.id)
    assert "cutover_pending" not in cfg
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert cfg["image"] == f"nerdit-app/{NAME}:1"
    assert BLUE in runtime.live


async def test_a_stop_racing_a_committed_cutover_tears_down_the_green(queries, tmp_path):
    """(PR #108) The stop teardown acts on the FRESH row, not the tick snapshot.

    A verify that commits between the tick's row fetch and the stop branch has
    already promoted the row to the green and finished its task —
    ``has_active_cutover`` is honestly False, and the snapshot still names the
    destroyed blue. Tearing the snapshot down would mark the row stopped while
    the green (row-backed, sweep-protected regardless of status) serves
    forever.
    """
    runtime = CutoverRuntime()
    job = _cutover_svc(status=JobStatus.running)
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries, tmp_path, job=job, runtime=runtime
    )
    # The cutover committed after the snapshot was taken: the DB row names the
    # green; the ``job`` object this tick carries still names blue.
    green = "c-green-committed"
    runtime.live[green] = datetime.now(UTC)
    runtime.live.pop(BLUE, None)  # f7 destroyed blue
    await queries.update_job_status(job.id, JobStatus.running, container_id=green)
    await queries.set_desired_state(job.id, "stopped")
    job.desired_state = "stopped"  # the stale snapshot

    await controller._reconcile_one(job, {green}, [1])

    assert green not in runtime.live, "the freshly promoted green must be torn down"
    row = await queries.get_job(job.id)
    assert row.status is JobStatus.stopped


async def test_a_stop_during_the_verify_survives_the_failure_settle(queries, tmp_path):
    """(PR #108) The autonomous settle must not erase an operator's stop.

    The revert previously hard-wrote ``desired_state='running'``: a stop that
    landed after the arm but before the probe failure settled was silently
    overwritten, and blue kept serving a service the operator had stopped. The
    revert now preserves a concurrently terminal desired state atomically.
    """
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    hold = _park_probe(controller)
    await controller.reconcile()
    await _await_green(controller)

    # The operator stops the service while the verify is still probing.
    await queries.set_desired_state(job.id, "stopped")
    hold.set()  # probes 500 from here: the verify settles cutover_failed
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.desired_state == "stopped", "the settle must never revive a stopped service"
    cfg = json.loads(row.config)
    assert cfg["last_deploy"]["reason"] == "cutover_failed"


async def test_cancel_on_a_completed_cutover_leaves_the_pointer_alone(queries, tmp_path):
    """(PR #108) A cancel with no owned marker must not touch the pointer.

    Reachable via the drain's sequential cancels and the stop branch's
    check-to-cancel gap: the verify commits (promotes the green on its
    ephemeral port, pops the marker) just before ``cancel()`` runs. Writing
    ``active_host_port = None`` then redirects the dial to the unused stable
    port — a 502 on a healthy, freshly promoted service.
    """
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _healthy(controller)
    await controller.reconcile()
    await _drain_cutovers(controller)  # the cutover commits

    ep = await queries.get_service_endpoint(NAME)
    promoted = ep.active_host_port
    assert promoted is not None and promoted != endpoint.host_port

    await controller.cancel_cutover(job.id)

    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == promoted, "no owned marker => the pointer is not ours"
    row = await queries.get_job(job.id)
    assert row.container_id != BLUE  # the promoted green still owns the row


async def test_a_grace_window_larger_than_the_verify_budget_still_commits(queries, tmp_path):
    """(PR #108) The no-health-spec budget widens to cover the grace window.

    ``cutover_grace_s > cutover_verify_timeout_s`` is schema-legal; without the
    widening every healthy no-spec redeploy deterministically settled
    ``cutover_failed`` — it cannot pass before the grace elapses and cannot
    outlive the budget.
    """
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries,
        tmp_path,
        job=_cutover_svc(health=False),
        cutover_grace_s=30,
        cutover_verify_timeout_s=5,
    )

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE
    assert json.loads(row.config)["last_deploy"]["phase"] == "healthy"


async def test_the_reap_never_kills_the_rows_own_container(queries, tmp_path):
    """The extra belt: a labelled container equal to ``job.container_id`` survives."""
    job = _cutover_svc(phase="verifying", cutover_pending={"version": 2, "blue": BLUE})
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path, job=job)
    runtime.labeled_containers[("nerdit-cutover", job.id)] = [BLUE]

    await controller.reconcile()

    assert BLUE not in runtime.killed
    assert BLUE in runtime.live


# --- 7. layer 4b — daemon restart AFTER the promotion committed ---------------


async def test_crash_after_the_promotion_replays_the_commit_tail(queries, tmp_path):
    """4b: never kill the green, replay the tail, pop the marker LAST."""
    events = RecordingEvents()
    green = "c-green"
    job = _cutover_svc(
        phase="verifying",
        container_id=green,
        cutover_pending={"version": 2, "blue": BLUE},
    )
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, job=job, events=events
    )
    green_port = endpoint.host_port + 500
    await queries.set_endpoint_active_port(NAME, green_port)
    runtime.live[BLUE] = datetime.now(UTC)  # the orphaned blue is still around
    runtime.labeled_containers[("nerdit-cutover", job.id)] = [green]

    await controller.reconcile()

    assert green in runtime.live and green not in runtime.killed, "4b kills nothing"
    cfg = await _cfg(queries, job.id)
    assert cfg["last_deploy"]["phase"] == "healthy"
    assert "cutover_pending" not in cfg
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == green_port, "the pointer names the live green — leave it"
    assert "service.cutover" in await _audit_actions(queries)
    assert events.count("service.cutover_succeeded") == 1
    # The blue is left to the zombie sweep, not torn down here.
    assert BLUE in runtime.live


@pytest.mark.parametrize("died_at", ["f3", "f4", "f5", "f6"])
async def test_the_commit_tail_records_survive_a_crash_at_least_once(queries, tmp_path, died_at):
    """The finding-3 pin: never ZERO durable records for a committed transition.

    A duplicate is a PASS, not a failure — moving the pop back to the front of
    the tail is what this forbids.
    """
    events = RecordingEvents()
    green = "c-green"
    phase = "healthy" if died_at in ("f5", "f6") else "verifying"
    job = _cutover_svc(
        phase=phase, container_id=green, cutover_pending={"version": 2, "blue": BLUE}
    )
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries, tmp_path, job=job, events=events
    )
    if died_at in ("f4", "f5", "f6"):
        # f4 already landed before the crash: the records exist once already.
        await queries.insert_audit_log(
            action="service.cutover",
            result="ok",
            principal_id="system",
            principal_role="system",
            target_type="service",
            target_id=job.id,
            params_redacted=json.dumps({"service": NAME, "version": 2}),
        )
        await events.record("service.cutover_succeeded")

    await controller.reconcile()

    actions = await _audit_actions(queries)
    assert actions.count("service.cutover") >= 1
    assert events.count("service.cutover_succeeded") >= 1
    assert "cutover_pending" not in await _cfg(queries, job.id)


# --- 8. the ineligible fallbacks (pre-P24 same-port swap, byte-identical) -----


@pytest.mark.parametrize("why", ["proxy_off", "gpus", "model", "opt_out"])
async def test_ineligible_rows_run_the_pre_p24_same_port_swap(queries, tmp_path, why):
    runtime = CutoverRuntime()
    fake = FakeCaddy()
    proxy = None if why == "proxy_off" else _proxy(queries, fake)
    job = _cutover_svc(cutover=False) if why == "opt_out" else _cutover_svc()
    if why == "gpus":
        job.gpu_count = 1
    if why == "model":
        job.kind = JobKind.model
    endpoint = await _seed(queries, job, runtime=runtime)
    controller = _controller(queries, runtime, proxy=proxy, tmp_path=tmp_path)
    _healthy(controller)

    await controller.reconcile()
    await _drain_cutovers(controller)

    # The destroy-first branch ran: blue was torn down and relaunched on the
    # STABLE port, with no cutover label and no active_host_port pointer.
    assert BLUE not in runtime.live
    assert not controller._cutover_tasks
    if why != "gpus":
        # (The GPU row parks waiting for a card — no GPUs are seeded here — so
        # it has no relaunch to shape-check; the destroy-first assertion above
        # is what that case pins.)
        launched = runtime.run_configs[-1]
        assert launched.ports == {8000: endpoint.host_port}
        assert launched.extra_labels is None
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def test_eligibility_predicate_is_exported_and_narrow():
    """WP10's poller imports ``_eligible``; pin its shape here."""
    settings = ServicesSettings()

    class _Proxy:
        enabled = True
        available = True

    job = _cutover_svc()
    cfg = json.loads(job.config)
    assert _eligible(job, cfg, settings, _Proxy()) is True
    assert _eligible(job, {**cfg, "cutover": False}, settings, _Proxy()) is False
    assert _eligible(job, {**cfg, "build_version": None}, settings, _Proxy()) is False
    assert _eligible(job, cfg, settings, None) is False

    class _Down(_Proxy):
        available = False

    assert _eligible(job, cfg, settings, _Down()) is False
    gpu_job = _cutover_svc()
    gpu_job.gpu_count = 2
    assert _eligible(gpu_job, cfg, settings, _Proxy()) is False
    model_job = _cutover_svc()
    model_job.kind = JobKind.model
    assert _eligible(model_job, cfg, settings, _Proxy()) is False


# --- 9. env-not-ready, stamping, CAS ------------------------------------------


async def test_launch_env_not_ready_retries_next_tick_and_regresses_the_phase(queries, tmp_path):
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)

    async def not_ready(*a, **kw):
        raise LaunchEnvNotReady("binding", "waiting for model 'llama'")

    controller._resolve_launch_env = not_ready  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    cfg = await _cfg(queries, job.id)
    assert "cutover_pending" not in cfg, "the marker is popped — nothing to settle"
    assert cfg["last_deploy"]["phase"] == "building", "the third D-P24-4 edge"
    assert cfg["last_deploy"]["reason"] is None, "retry-next-tick is NOT a settle"
    row = await queries.get_job(job.id)
    assert row.container_id == BLUE
    assert row.status is JobStatus.restarting
    assert len(runtime.run_configs) == 0, "nothing was launched"
    assert "service.cutover_failed" not in await _audit_actions(queries)


async def test_verifying_is_stamped_exactly_once(queries, tmp_path, monkeypatch):
    """The arm stamps it; the green's ``stamp_launching`` must not re-stamp."""
    phases: list[object] = []
    real = cutover_mod.stamp_last_deploy

    async def spy(q, job_id, **kw):
        phases.append(kw.get("phase"))
        return await real(q, job_id, **kw)

    monkeypatch.setattr(cutover_mod, "stamp_last_deploy", spy)
    monkeypatch.setattr(launch_mod, "stamp_last_deploy", spy)

    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    _healthy(controller)

    await controller.reconcile()
    await _drain_cutovers(controller)

    assert phases.count("verifying") == 1
    assert "launching" not in phases, "the green launch is provenance-only"
    cfg = await _cfg(queries, job.id)
    # ... but the provenance side-stamp DID land.
    assert "last_launch_env_keys" in cfg
    assert cfg["last_launch_volumes"] == []


async def test_a_redeploy_landing_mid_arm_is_never_clobbered(queries, tmp_path):
    """A CAS miss on the marker write aborts the arm — no task, no clobber."""
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)

    async def miss(job_id, config, *, expect_build_version):
        return False

    queries.update_job_config_guarded = miss  # type: ignore[assignment]

    armed = await controller._cutover.maybe_cutover(job, True)

    assert armed is False
    assert not controller._cutover_tasks
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def test_the_pop_never_disarms_a_newer_generations_marker(queries, tmp_path):
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    cfg = await _cfg(queries, job.id)
    cfg["build_version"] = 3
    cfg["cutover_pending"] = {"version": 3, "blue": "c-newer"}
    await queries.update_job_config(job.id, json.dumps(cfg))

    await controller._cutover._pop_marker(job.id, 2)  # the OLD generation's pop

    fresh = await _cfg(queries, job.id)
    assert fresh["cutover_pending"] == {"version": 3, "blue": "c-newer"}


# --- 10. reaper / DELETE / drain parity ---------------------------------------


async def test_the_green_is_protected_from_the_zombie_sweep(queries, tmp_path):
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    hold = _park_probe(controller)

    await controller.reconcile()
    green = await _await_green(controller)
    assert green in controller.protected_container_ids()
    assert controller.has_active_cutover(job.id) is True
    assert controller.busy_cutovers() == 1

    # Age the green well past the sweep threshold and sweep: nothing dies.
    runtime.live[green] = datetime.now(UTC) - timedelta(seconds=3600)
    sweeper = ZombieSweeper(
        queries=queries,
        runtime=runtime,
        extra_protected=controller.protected_container_ids,
    )
    killed = await sweeper.cleanup_zombies()
    assert killed == 0
    assert green in runtime.live

    hold.set()  # release the probe (500s from here) so the verify settles
    await _drain_cutovers(controller)


async def test_cancel_cutover_kills_the_green_and_unwinds_the_pointer(queries, tmp_path):
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _park_probe(controller)

    await controller.reconcile()
    green = await _await_green(controller)
    await queries.set_endpoint_active_port(NAME, endpoint.host_port + 500)

    await controller.cancel_cutover(job.id)

    assert green not in runtime.live
    assert not controller.cutover_container_ids()
    assert controller.has_active_cutover(job.id) is False
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def _arm_and_hold(queries, tmp_path, **kw):
    """Tick once and return with the verify task in flight, green registered."""
    setup = await _full_setup(queries, tmp_path, **kw)
    controller = setup[0]
    _park_probe(controller)
    await controller.reconcile()
    green = await _await_green(controller)
    assert controller.has_active_cutover(setup[4].id) is True
    return (*setup, green)


async def test_the_drain_cancels_each_verify_before_killing_its_green(queries, tmp_path):
    """(U1) A raw kill alone races a verify already past its probe.

    Killing the green without cancelling the task leaves that task free to run
    on into the f1-f3 window, promote a container the drain has just destroyed
    and then destroy blue at f7 — a row naming a dead container with nothing
    serving. The cancel must land FIRST.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, _ep, green = await _arm_and_hold(
        queries, tmp_path, events=events
    )
    task = controller._cutover_tasks[job.id]
    orig_destroy = controller._destroy_container
    done_at_destroy: dict[str, bool] = {}

    async def destroy(cid):
        done_at_destroy[cid] = task.done()
        await orig_destroy(cid)

    controller._destroy_container = destroy  # type: ignore[assignment]

    killed = await controller.kill_transient_containers()

    assert killed == 1
    assert done_at_destroy == {green: True}, (
        "the verify task must be cancelled before its green is destroyed"
    )
    assert green not in runtime.live
    assert not controller.cutover_container_ids()
    # ... and nothing was promoted behind the drain.
    row = await queries.get_job(job.id)
    assert row.container_id == BLUE
    assert events.count("service.cutover_succeeded") == 0
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def test_cancel_after_the_commit_point_never_destroys_the_promoted_green(queries, tmp_path):
    """A cancel landing in f3-f6 finishes the cutover; it does not undo it.

    Past the commit point the green IS the row's own live container: destroying
    it (and rewinding the pointer to blue's port) would tear down the service
    the commit just made live. The marker is still armed, so the cancel owes the
    generation its unfinished f4-f6 tail — layer 4b's contract, replayed here.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events
    )
    _healthy(controller)
    _park_commit_tail(controller)

    await controller.reconcile()
    green = await _await_promotion(queries, job.id)
    ep = await queries.get_service_endpoint(NAME)
    green_port = ep.active_host_port
    assert green_port is not None and green_port != endpoint.host_port

    killed = await controller.cancel_cutover(job.id)

    assert killed == set(), "a promoted green is the row's own container, never a kill target"
    assert green in runtime.live
    assert BLUE in runtime.live, "4b parity: kill NOTHING — the orphaned blue is the sweep's"
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == green_port, "the pointer names the green's live port"
    cfg = await _cfg(queries, job.id)
    assert "cutover_pending" not in cfg
    assert cfg["last_deploy"]["phase"] == "healthy"
    assert events.count("service.cutover_succeeded") == 1
    assert "service.cutover" in await _audit_actions(queries)
    assert not controller.cutover_container_ids()


async def test_cancel_in_the_commit_tails_wake_leaves_state_untouched(queries, tmp_path):
    """f6-f8: the tail is complete but the registry still names the green.

    The marker is gone, the records are written and blue is already destroyed —
    there is nothing left to unwind and nothing left to replay. The cancel must
    be a pure no-op, in particular never duplicating the success records.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, _endpoint = await _full_setup(
        queries, tmp_path, events=events
    )
    _healthy(controller)
    hold = asyncio.Event()

    async def parked_prune(row):
        await hold.wait()

    controller._prune_old_images = parked_prune  # type: ignore[assignment]

    await controller.reconcile()
    green = await _await_promotion(queries, job.id, popped=True)
    ep = await queries.get_service_endpoint(NAME)
    promoted_port = ep.active_host_port
    assert events.count("service.cutover_succeeded") == 1
    assert green in controller.cutover_container_ids(), "the registry still holds the green"

    killed = await controller.cancel_cutover(job.id)

    assert killed == set()
    assert green in runtime.live
    row = await queries.get_job(job.id)
    assert row.container_id == green
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == promoted_port
    assert events.count("service.cutover_succeeded") == 1, "no duplicate replay"
    assert "service.cutover_failed" not in await _audit_actions(queries)
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def test_the_drain_never_destroys_a_promoted_green(queries, tmp_path):
    """The restart drain leg of the same window: a promoted green is not transient.

    ``kill_transient_containers`` cancels every verify before its raw label-kill
    fallback, so a verify parked in its commit tail is cancelled here — and the
    cancel must leave the row's own container serving, not report it killed.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, _endpoint = await _full_setup(
        queries, tmp_path, events=events
    )
    _healthy(controller)
    _park_commit_tail(controller)

    await controller.reconcile()
    green = await _await_promotion(queries, job.id)
    ep = await queries.get_service_endpoint(NAME)
    promoted_port = ep.active_host_port

    killed = await controller.kill_transient_containers()

    assert killed == 0, "a row-backed container is never counted as drained"
    assert green in runtime.live
    row = await queries.get_job(job.id)
    assert row.container_id == green
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port == promoted_port
    assert "cutover_pending" not in await _cfg(queries, job.id), "the tail was completed"
    assert events.count("service.cutover_succeeded") == 1


# --- 11. adjudicated review findings (U1-U14) --------------------------------


async def test_a_stop_landing_mid_verify_cancels_the_cutover(queries, tmp_path):
    """(U2/U9) ``desired_state='stopped'`` must not leave the green running.

    ``_teardown_to_stopped`` only knows ``job.container_id``: without the
    cancel it tears blue down and the orphaned green keeps probing, still able
    to reach its commit and promote onto a row the operator just stopped.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, _ep, green = await _arm_and_hold(
        queries, tmp_path, events=events
    )

    await queries.set_desired_state(job.id, "stopped")
    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.status is JobStatus.stopped
    assert green not in runtime.live
    assert not controller.cutover_container_ids()
    assert controller.has_active_cutover(job.id) is False
    assert _dials(fake) == [], "no route was ever registered for the green"
    actions = await _audit_actions(queries)
    assert "service.cutover" not in actions
    assert "service.cutover_failed" not in actions, "a stop is not a verification verdict"
    assert events.count("service.cutover_succeeded") == 0
    assert "cutover_pending" not in await _cfg(queries, job.id)


async def test_a_stale_row_never_settles_an_already_committed_cutover(queries, tmp_path):
    """(U7) Layer 4 must re-read before acting on the tick's row snapshot.

    A verify that commits between the tick's fetch and this row's turn leaves
    the snapshot carrying the marker AND the old ``container_id``. Dispatching
    4a on it would clear the pointer and — because the never-kill belt compares
    the STALE blue — SIGKILL the freshly promoted green.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path, events=events)
    _healthy(controller)
    await controller.reconcile()
    await _drain_cutovers(controller)

    committed = await queries.get_job(job.id)
    ep_before = await queries.get_service_endpoint(NAME)
    assert committed.container_id != BLUE  # the cutover really did commit
    # The tick's snapshot, as it looked before the commit: marker armed, blue.
    stale = await queries.get_job(job.id)
    stale.container_id = BLUE
    stale.config = json.dumps(
        {**json.loads(committed.config), "cutover_pending": {"version": 2, "blue": BLUE}}
    )
    runtime.labeled_containers[("nerdit-cutover", job.id)] = [committed.container_id]

    handled = await controller._cutover.maybe_cutover(stale, True)

    assert handled is True, "the marker is still claimed — the caller must not swap"
    assert committed.container_id not in runtime.killed, "the promoted green must survive"
    assert committed.container_id in runtime.live
    after = await queries.get_job(job.id)
    assert after.container_id == committed.container_id
    assert after.status is JobStatus.running
    assert json.loads(after.config)["last_deploy"]["phase"] == "healthy"
    ep_after = await queries.get_service_endpoint(NAME)
    assert ep_after.active_host_port == ep_before.active_host_port
    assert "service.cutover_failed" not in await _audit_actions(queries)


async def test_the_arm_refuses_a_terminally_stopped_row(queries, tmp_path):
    """(U3) A stop landing between the tick's fetch and the arm wins.

    Driven through ``maybe_cutover`` directly: the reconcile's own
    ``desired == "stopped"`` branch returns before the gate is reached, so only
    the arm's own fresh-row refusal covers this race.
    """
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    _healthy(controller)
    await queries.set_desired_state(job.id, "stopped")  # ... after the tick's fetch

    armed = await controller._cutover.maybe_cutover(job, True)

    assert armed is False
    assert not controller._cutover_tasks
    assert "cutover_pending" not in await _cfg(queries, job.id)
    assert runtime.run_configs == [], "no green was launched"


async def test_a_crash_that_took_blue_down_too_never_launches_the_candidate(queries, tmp_path):
    """(U12) The layer-order pin: layer 4 runs even when ``live`` is False.

    A crash can take blue down with the daemon. If layer 2's applicability gate
    ran first, ``maybe_cutover`` would return False and ``_reconcile_dead``
    would relaunch — on the stable port, from the UNVERIFIED candidate image.
    That is exactly the silent swap D-P24-4 forbids.
    """
    job = _cutover_svc(phase="verifying", cutover_pending={"version": 2, "blue": BLUE})
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, job=job, blue_live=False
    )
    runtime.labeled_containers[("nerdit-cutover", job.id)] = ["c-green"]
    runtime.live["c-green"] = datetime.now(UTC)

    await controller.reconcile()

    assert runtime.run_configs == [], "the unverified candidate must never be launched"
    assert "c-green" in runtime.killed
    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    # With blue gone too there is no live container to keep serving, so
    # ``_settle_failed_generation`` takes its TERMINAL branch: the row settles
    # ``failed`` (which is what keeps the candidate off the reconcile loop)
    # rather than reverting the image behind a survivor.
    assert row.status is JobStatus.failed
    assert row.desired_state == "failed"
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert cfg["last_deploy"]["phase"] == "failed"
    assert "cutover_pending" not in cfg
    ep = await queries.get_service_endpoint(NAME)
    assert ep is None or ep.active_host_port is None


async def test_a_green_dying_between_the_dial_verify_and_the_promotion_unwinds(queries, tmp_path):
    """(U1) The residual window: the green is not probed during ``_await_dial``."""
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events
    )
    _healthy(controller)
    orig_live_routes = proxy.live_routes

    async def live_routes():
        live = await orig_live_routes()
        # The dial now names the green — and the green dies right here, in the
        # window between the successful read-back and the f3 promotion.
        for cid in list(runtime.live):
            if cid != BLUE:
                runtime.live.pop(cid)
        return live

    proxy.live_routes = live_routes  # type: ignore[assignment]

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert row.container_id == BLUE, "a dead green must never be promoted"
    assert BLUE in runtime.live, "blue is untouched"
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert "cutover_pending" not in cfg
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None, "the pointer is unwound back to the stable port"
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items and items[0].params_redacted["stage"] == "repoint"
    assert events.count("service.cutover_succeeded") == 0


async def test_a_declared_start_period_widens_the_verify_budget(queries, tmp_path):
    """(U6) An app that declares a warm-up must not be failed inside it."""
    job = _cutover_svc()
    job.health_check = {"path": "/health", "start_period_s": 30}
    controller, runtime, proxy, fake, job, _ep = await _full_setup(
        queries,
        tmp_path,
        job=job,
        cutover_verify_timeout_s=5,  # nominal budget is far too short
        cutover_grace_s=10,
    )
    clock = {"t": 0.0}

    async def check(host_port, path, timeout):
        return 200 if cutover_mod._MONOTONIC() >= 8 else 500

    controller._check_health = check  # type: ignore[assignment]
    del clock

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE, "the green passed inside start_period + grace"
    assert json.loads(row.config)["last_deploy"]["phase"] == "healthy"


async def test_a_named_volume_colliding_with_a_mount_settles_before_the_launch(queries, tmp_path):
    """(U10) ``_launch``'s conflict check, mirrored for the green.

    The green becomes the live container on success, so ``_launch`` never
    re-checks it — without this, a cutover would be the one path that lands a
    shadowed mount.
    """
    script = tmp_path / "src" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('hi')\n")
    # ``/workspace`` itself is forbidden as a named-volume target, but a path
    # NESTED under it is not — and nesting is exactly what the check catches
    # (normalized prefix containment, not equality).
    job = _cutover_svc(volumes=["ws:/workspace/data"])
    job.script_path = str(script)  # Tier-B mount lands at /workspace
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path, job=job)
    _healthy(controller)

    await controller.reconcile()
    await _drain_cutovers(controller)

    assert runtime.run_configs == [], "the green is never launched"
    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert row.container_id == BLUE and BLUE in runtime.live
    assert cfg["last_deploy"]["reason"] == "cutover_failed"
    assert "cutover_pending" not in cfg
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items and items[0].params_redacted["stage"] == "probe"


async def test_stamp_launching_with_phase_none_leaves_the_deploy_blob_untouched(queries, tmp_path):
    """(U14) ``phase=None`` skips ONLY the phase stamp, provenance still lands.

    ``updated_at`` included: the blob must be byte-identical, or a "no-op"
    stamp would still look like a phase transition to anything reading it.
    """
    controller, runtime, proxy, fake, job, _ep = await _full_setup(queries, tmp_path)
    before = (await _cfg(queries, job.id))["last_deploy"]
    config = build_app_container_config(
        {"image": "nerdit-app/app:2", "port": 8000},
        ContainerSettings(),
        AppContainerSpec(
            command=None,
            volumes={},
            env={"PORT": "8000", "FOO": "bar"},
            workdir=None,
            gpu_ids=[],
            vendor=GpuVendor.nvidia,
            container_port=8000,
            host_port=9999,
        ),
    )

    await launch_mod.stamp_launching(controller, job, config, {}, phase=None)

    cfg = await _cfg(queries, job.id)
    assert cfg["last_deploy"] == before, "the phase blob is byte-identical"
    assert cfg["last_launch_env_keys"] == ["FOO", "PORT"]
    assert cfg["last_launch_volumes"] == []


# --- 12. (P26 D-P26-2) the repoint is the WHOLE id set, all-or-nothing --------


async def _bind(queries, job, *domains: str) -> None:
    for domain in domains:
        outcome, _row = await queries.add_service_domain(
            job.service_name, domain, acme=False, job_id=job.id
        )
        assert outcome == "inserted"


def _ids(*domains: str) -> list[str]:
    return ["nerdit-route-app", *(f"nerdit-route-app@{d}" for d in domains)]


async def test_cutover_repoints_every_domain_route(queries, tmp_path):
    """All four ids dial the green before the row is promoted and blue dies."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _healthy(controller)
    await _bind(queries, job, "a.example.com", "b.example.com", "c.example.com")
    await proxy.reconcile()  # blue is serving on all four ids

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE and row.status is JobStatus.running
    ep = await queries.get_service_endpoint(NAME)
    green_dial = f"127.0.0.1:{ep.active_host_port}"
    live = await proxy.live_routes()
    ids = _ids("a.example.com", "b.example.com", "c.example.com")
    assert {rid: live[rid].dial for rid in ids} == dict.fromkeys(ids, green_dial)
    assert BLUE not in runtime.live, "blue is destroyed only once the SET converged"
    # And the Host routes are still ahead of the path route after the repoint.
    assert [obj["@id"] for obj in fake.routes][-1] == "nerdit-route-app"


async def test_three_domain_partial_repoint_reverts(queries, tmp_path):
    """One id that cannot converge diverts the WHOLE cutover to (g').

    Without all-or-nothing this is the bad outcome the decision exists to
    prevent: two domains happily dialling a container step (7) is about to
    destroy, and a permanent 502 on the operator's own name.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_repoint_timeout_s=2
    )
    _healthy(controller)
    await _bind(queries, job, "a.example.com", "b.example.com", "c.example.com")
    await proxy.reconcile()
    blue_dial = f"127.0.0.1:{endpoint.host_port}"
    # The third domain's route write fails permanently; the other three land.
    fake.fail_ids = {"nerdit-route-app@c.example.com"}

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    cfg = json.loads(row.config)
    assert row.container_id == BLUE and BLUE in runtime.live
    assert row.status is JobStatus.running
    ep = await queries.get_service_endpoint(NAME)
    assert ep.active_host_port is None, "the pointer went back to blue"
    assert cfg["image"] == f"nerdit-app/{NAME}:1"
    assert "cutover_pending" not in cfg
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items and items[0].params_redacted["stage"] == "repoint"
    failed = [kw for t, kw in events.rows if t == "service.cutover_failed"]
    assert failed and failed[0]["data"] == {"stage": "repoint"}
    # The green is gone and every id — the two that DID move included — is back
    # on blue after one ordinary proxy tick.
    fake.fail_ids = set()
    await proxy.reconcile()
    live = await proxy.live_routes()
    ids = _ids("a.example.com", "b.example.com", "c.example.com")
    assert {rid: live[rid].dial for rid in ids} == dict.fromkeys(ids, blue_dial)


async def test_await_dials_needs_every_id_on_one_read(queries, tmp_path):
    """A set satisfied one id per poll is NOT a converged set."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, cutover_repoint_timeout_s=2
    )
    manager = controller._cutover
    reads = {"n": 0}
    port = 51999
    good = LiveRoute(dial=f"127.0.0.1:{port}", shape="host")
    stale = LiveRoute(dial="127.0.0.1:1", shape="host")

    async def live_routes():
        reads["n"] += 1
        # Each read has exactly one of the two ids converged — alternating.
        first = reads["n"] % 2 == 1
        return {"one": good if first else stale, "two": stale if first else good}

    proxy.live_routes = live_routes  # type: ignore[assignment]
    assert await manager._await_dials(("one", "two"), port) is False
    assert reads["n"] > 1

    async def both():
        return {"one": good, "two": good}

    proxy.live_routes = both  # type: ignore[assignment]
    assert await manager._await_dials(("one", "two"), port) is True


async def test_a_domain_removed_between_the_snapshot_and_register_does_not_unwind(
    queries, tmp_path
):
    """(Codex round 1, P2 #3831777105) The commit derived ``caddy_ids`` from one
    read of ``service_domains`` and ``register(with_domains=True)`` took a
    SECOND, independent one. A ``DELETE …/domains/{domain}`` landing between
    those two awaits left the removed id in the awaited set while nothing would
    ever write it — the route kept dialling blue until the next reconcile tick
    dropped it — so ``_await_dials`` burned its whole budget and unwound a
    green that had already verified healthy.

    Both sides now read the same snapshot, so the removal is simply cleaned up
    by the next tick, as the DELETE route's docstring promises.
    """
    events = RecordingEvents()
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(
        queries, tmp_path, events=events, cutover_repoint_timeout_s=2
    )
    _healthy(controller)
    await _bind(queries, job, "gone.example.com")
    await proxy.reconcile()  # blue is serving both ids

    orig_get = queries.get_service_domains
    calls: list[int] = []

    async def get_service_domains(name):
        rows = await orig_get(name)
        calls.append(len(rows))
        if len(calls) == 1:
            # The operator releases the name in the gap the commit used to have.
            await queries.remove_service_domain(NAME, "gone.example.com")
        return rows

    queries.get_service_domains = get_service_domains

    await controller.reconcile()
    await _drain_cutovers(controller)

    row = await queries.get_job(job.id)
    assert row.container_id != BLUE, "the healthy green was promoted, not unwound"
    assert row.status is JobStatus.running
    assert events.count("service.cutover_failed") == 0
    items, _ = await queries.list_audit_log(action="service.cutover_failed", limit=10)
    assert items == []
    # ONE read of the table, by the commit — register no longer takes a second.
    assert len(calls) == 1


async def test_a_domain_bound_during_the_verify_window_is_repointed(queries, tmp_path):
    """The id set is read at COMMIT time, not when the cutover was armed."""
    controller, runtime, proxy, fake, job, endpoint = await _full_setup(queries, tmp_path)
    _healthy(controller)
    await proxy.reconcile()
    orig_register = proxy.register

    async def register(name, port, edge_auth=None, **kw):
        await orig_register(name, port, edge_auth, **kw)

    proxy.register = register  # type: ignore[assignment]
    # Bind the domain after the row exists but before the commit reads it.
    await _bind(queries, job, "late.example.com")

    await controller.reconcile()
    await _drain_cutovers(controller)

    ep = await queries.get_service_endpoint(NAME)
    live = await proxy.live_routes()
    assert live["nerdit-route-app@late.example.com"].dial == f"127.0.0.1:{ep.active_host_port}"
