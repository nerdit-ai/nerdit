"""Durable event feed — the writer, its vocabulary, and the payload discipline.

P24a WP1 (D-P24-2 / D-P24-3). Everything here is about the three properties the
feed is only useful if it has: it never breaks a reconcile tick, it does not
spam, and it never carries free text (the rows are readable by any
authenticated principal and are POSTed to a third-party host in P24c).
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from nerdit.core import eventlog
from nerdit.core.eventlog import EVENT_TYPES, EventRecorder, record_job_event
from nerdit.core.events import EventBus
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

#: (P33 D-GH-10) The provenance quartet every settle event carries — null for a
#: ZIP/workspace generation, and ``remediation_code`` null unless a crash-loop
#: rule fired at settle.
_NO_PROVENANCE = {"repo": None, "ref": None, "sha": None, "remediation_code": None}


async def _all_events(db):
    cur = await db.conn.execute("SELECT * FROM events ORDER BY id")
    return list(await cur.fetchall())


# --- vocabulary ---------------------------------------------------------------


def test_event_types_pinned():
    """The D-P24-3 table, verbatim. A new type is a plan edit, not a diff."""
    expected = frozenset(
        {
            "service.deploy_started",
            "service.deploy_succeeded",
            "service.deploy_failed",
            "service.healthy",
            "service.degraded",
            "service.restarting",
            "service.failed",
            "service.stopped",
            "service.cutover_started",
            "service.cutover_succeeded",
            "service.cutover_failed",
            "model.ready",
            "model.failed",
            "database.ready",
            "database.failed",
            # (P37 D-P37-11) managed-database dump/restore outcomes, emitted by
            # the daemon/routes/databases.py trio. A success carries the tar
            # BASENAME (and, for a dump, its size); a failure carries a fixed
            # ``reason`` token and NOTHING else — never the tool's log tail,
            # which this feed would POST verbatim to third-party webhook hosts
            # and which ``pg_restore`` fills with row values.
            "database.dump_succeeded",
            "database.dump_failed",
            "database.restore_succeeded",
            "database.restore_failed",
            "gitwatch.redeploy_triggered",
            "gitwatch.skipped_no_cutover",
            "gitwatch.poll_failed",
            "gitwatch.redeploy_failed",
            "gitwatch.nudge_ignored",
            # (P27 WP-C1) node-link session bookends, emitted by
            # core/link/manager.py on offline<->online transitions only.
            "link.connected",
            "link.disconnected",
            # (P32) the account entitlement mirror moved, emitted by the
            # cloud-written PUT /api/link/entitlement route on a CHANGE only.
            # ``data`` is one key — the bool — and nothing else.
            "link.entitlement",
            # (P26 WP-H) hosted-share edges, emitted by the share router and
            # the delete/purge route. ``share.ready`` carries the private URL
            # only; a public one is omitted (this feed reaches webhook hosts).
            "share.ready",
            "share.removed",
            # (P26 WP1) custom-domain edges, emitted by the domains router and
            # the delete/purge cascade. The domain NAME travels in ``data``:
            # public DNS the operator chose, never a capability.
            "domain.added",
            "domain.removed",
            # (P17d WP-D1) offline product-license edges, emitted by the boot
            # loader (core/license.py + daemon/bootstrap.py) and the install
            # route. Machine tokens only: never the blob, never customer_id.
            "license.installed",
            "license.removed",
            "license.rejected",
            "daemon.started",
            "daemon.stopping",
        }
    )
    assert expected == EVENT_TYPES


def test_database_failed_is_declared_without_a_v1_emitter():
    """D-P24-3 states it explicitly, so pin it: declared, not fired in v1.

    Its only emitter is the ``DataProvisionError`` branch, which the v1
    readiness probes never raise. Without this pin someone hunts for a missing
    emitter, finds none, and deletes the vocabulary entry.
    """
    assert "database.failed" in EVENT_TYPES
    source = Path("src/nerdit/core/data/backend.py").read_text(encoding="utf-8")
    assert "DataProvisionError" in source


async def test_unknown_type_is_refused_and_writes_nothing(db, queries):
    recorder = EventRecorder(queries)
    await recorder.record("service.exploded")
    assert await _all_events(db) == []


# --- writer tolerance (D-P24-2) ----------------------------------------------


async def test_insert_failure_never_propagates(queries):
    """A feed write must never abort a reconcile tick."""
    recorder = EventRecorder(queries)

    async def _boom(**_kwargs):
        raise RuntimeError("db on fire")

    recorder._queries = type("Q", (), {"insert_event": staticmethod(_boom)})()
    await recorder.record("service.failed", service_name="svc")  # must not raise


async def test_a_failed_insert_does_not_reserve_the_coalescing_window(db, queries):
    """The window opens on the ROW, never on the attempt.

    Stamping the key before the insert made a transient write failure suppress
    every retry of that same event for the next 60 s — the flap guard eating
    exactly the transition it exists to report once. The retry must land.
    """
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    real = queries.insert_event

    async def _boom(**_kwargs):
        raise RuntimeError("db on fire")

    queries.insert_event = _boom  # type: ignore[method-assign]
    try:
        await recorder.record("service.failed", service_name="svc", reason="oom")
    finally:
        queries.insert_event = real  # type: ignore[method-assign]

    assert recorder._recent == {}  # nothing reserved by the failed attempt
    assert await _all_events(db) == []

    # The identical record, well inside the window, IS written.
    await recorder.record("service.failed", service_name="svc", reason="oom")
    rows = await _all_events(db)
    assert [r["type"] for r in rows] == ["service.failed"]

    # ...and now the window is open, so the next identical one still coalesces.
    await recorder.record("service.failed", service_name="svc", reason="oom")
    assert len(await _all_events(db)) == 1


async def test_bus_failure_never_propagates(queries):
    recorder = EventRecorder(queries, bus=None)

    class _AngryBus:
        def publish(self, _event):
            raise RuntimeError("bus on fire")

    recorder._bus = _AngryBus()
    await recorder.record("service.failed", service_name="svc")  # must not raise


async def test_no_recorder_is_a_noop(queries):
    """A bare-constructed controller passes ``None`` — every emit is inert."""
    job = Job(id="jobnorec0001", kind=JobKind.service, service_name="svc")
    await record_job_event(None, "service.failed", job)


# --- the row + the bus tee ----------------------------------------------------


async def test_row_and_bus_payload_carry_the_id(db, queries):
    """WP2's SSE resume dedups by id, so the bus frame MUST carry the row id."""
    bus = EventBus()
    sub = bus.subscribe()
    recorder = EventRecorder(queries, bus)

    await recorder.record(
        "service.failed",
        kind="service",
        service_name="svc",
        reason="crash_loop",
        build_version=4,
        data={"restart_count": 3, "exit_code": 137},
    )

    rows = await _all_events(db)
    assert len(rows) == 1
    assert rows[0]["type"] == "service.failed"
    assert rows[0]["kind"] == "service"
    assert rows[0]["service_name"] == "svc"
    assert rows[0]["reason"] == "crash_loop"
    assert rows[0]["build_version"] == 4
    assert rows[0]["ts"]

    frame = sub._queue.get_nowait()
    assert frame["id"] == rows[0]["id"]
    assert frame["type"] == "service.failed"
    assert frame["service_name"] == "svc"
    assert frame["data"] == {"restart_count": 3, "exit_code": 137}


async def test_the_live_frame_and_the_stored_row_share_one_instant(db, queries):
    """ONE clock read per event, for both surfaces.

    A consumer merging the live tail with a replay of the same rows sees each
    event twice — once from the bus, once from ``GET /events``. Minting the
    row's ``ts`` from the column default and the frame's from a second
    ``now()`` made those two copies disagree, so the writer now owns the
    instant. The shapes differ (stored: the ``datetime('now')`` space form;
    frame: the aware ISO-8601 form) — the instant does not.
    """
    from datetime import UTC, datetime

    bus = EventBus()
    sub = bus.subscribe()
    recorder = EventRecorder(queries, bus)

    await recorder.record("service.healthy", service_name="svc")

    rows = await _all_events(db)
    frame = sub._queue.get_nowait()

    stored = datetime.fromisoformat(rows[0]["ts"]).replace(tzinfo=UTC)
    live = datetime.fromisoformat(frame["ts"])
    assert live.tzinfo is not None  # the aware form the D-P24-3 payload pins
    assert live == stored
    # Whole seconds on both sides: the stored column has no sub-second slot, so
    # a microsecond-bearing frame could never equal its own replay.
    assert live.microsecond == 0


async def test_record_job_event_projects_the_row(db, queries):
    job = Job(
        id="jobproject01",
        kind=JobKind.model,
        service_name="ollama-llama",
        status=JobStatus.running,
        config='{"build_version": 7}',
    )
    await record_job_event(EventRecorder(queries), "model.ready", job)

    rows = await _all_events(db)
    assert (rows[0]["kind"], rows[0]["service_name"], rows[0]["build_version"]) == (
        "model",
        "ollama-llama",
        7,
    )


async def test_record_job_event_tolerates_a_malformed_config(db, queries):
    job = Job(id="jobbadcfg001", kind=JobKind.service, service_name="svc", config="{not json")
    await record_job_event(EventRecorder(queries), "service.stopped", job)

    rows = await _all_events(db)
    assert rows[0]["build_version"] is None


# --- coalescing (D-P24-3 rule 2) ---------------------------------------------


async def test_identical_triple_inside_the_window_is_dropped(db, queries):
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    await recorder.record("service.degraded", service_name="svc", reason="health_check_failed")
    await recorder.record("service.degraded", service_name="svc", reason="health_check_failed")
    assert len(await _all_events(db)) == 1


async def test_a_changed_reason_passes_immediately(db, queries):
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    await recorder.record("service.restarting", service_name="svc", reason="oom")
    await recorder.record("service.restarting", service_name="svc", reason="crash")
    assert len(await _all_events(db)) == 2


async def test_a_different_service_is_a_different_key(db, queries):
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    await recorder.record("service.degraded", service_name="a", reason="health_check_failed")
    await recorder.record("service.degraded", service_name="b", reason="health_check_failed")
    assert len(await _all_events(db)) == 2


async def test_window_expiry_lets_the_same_triple_through(db, queries):
    """A triple older than the window re-opens.

    The stored sample is aged in place rather than by patching ``time.monotonic``
    globally — aiosqlite's worker thread reads the same clock, and patching it
    out from under the connection is how this test wedges the DB.
    """
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    key = ("service.degraded", "svc", "health_check_failed", None)

    await recorder.record("service.degraded", service_name="svc", reason="health_check_failed")
    await recorder.record("service.degraded", service_name="svc", reason="health_check_failed")
    assert len(await _all_events(db)) == 1

    recorder._recent[key] -= 61.0
    await recorder.record("service.degraded", service_name="svc", reason="health_check_failed")
    assert len(await _all_events(db)) == 2


async def test_a_new_build_version_is_a_different_key(db, queries):
    """The flap guard must never swallow a distinct deploy generation.

    Two deploys of one service settling inside the window carry the same
    ``(type, service_name, reason)`` triple — the coalescing key includes
    ``build_version`` precisely so the second one is still reported.
    """
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    for version in (1, 2):
        await recorder.record(
            "service.deploy_succeeded",
            service_name="svc",
            reason="healthy",
            build_version=version,
        )

    rows = await _all_events(db)
    assert [r["build_version"] for r in rows] == [1, 2]


async def test_same_generation_still_coalesces(db, queries):
    """The guard is only widened, never disarmed: one generation, one row."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    for _ in range(3):
        await recorder.record(
            "service.degraded",
            service_name="svc",
            reason="health_check_failed",
            build_version=7,
        )

    assert len(await _all_events(db)) == 1


async def test_two_deploy_generations_of_one_service_both_report(db, queries):
    """End-to-end through ``record_job_event``: nothing is dropped."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)
    for version in (1, 2):
        job = Job(
            id="jobgen000001",
            kind=JobKind.service,
            service_name="svc",
            status=JobStatus.running,
            config=json.dumps({"build_version": version}),
        )
        await record_job_event(recorder, "service.healthy", job)

    rows = await _all_events(db)
    assert [r["build_version"] for r in rows] == [1, 2]


def test_coalescing_uses_the_monotonic_clock():
    """A wall-clock step backwards would wedge the window open indefinitely."""
    source = inspect.getsource(EventRecorder._coalesce_suppressed) + inspect.getsource(
        EventRecorder._coalesce_commit
    )
    assert "time.monotonic()" in source
    assert "datetime" not in source


async def test_recent_map_is_bounded_and_evicts_oldest(queries):
    recorder = EventRecorder(queries)
    for i in range(eventlog._MAX_RECENT + 20):
        await recorder.record("service.stopped", service_name=f"svc-{i}")

    assert len(recorder._recent) == eventlog._MAX_RECENT
    # The 20 oldest keys were evicted; the newest survive.
    assert ("service.stopped", "svc-0", None, None) not in recorder._recent
    assert ("service.stopped", f"svc-{eventlog._MAX_RECENT + 19}", None, None) in recorder._recent


async def test_eviction_lets_an_evicted_triple_re_emit(db, queries):
    """Eviction is a *correctness* trade: an evicted key re-emits, never drops."""
    recorder = EventRecorder(queries)
    await recorder.record("service.stopped", service_name="first")
    for i in range(eventlog._MAX_RECENT + 1):
        await recorder.record("service.stopped", service_name=f"filler-{i}")

    before = len(await _all_events(db))
    await recorder.record("service.stopped", service_name="first")
    assert len(await _all_events(db)) == before + 1


# --- the process singleton ----------------------------------------------------


def test_recorder_singleton_defaults_to_none_and_round_trips(queries):
    assert eventlog.get_recorder() is None
    recorder = EventRecorder(queries)
    eventlog.set_recorder(recorder)
    try:
        assert eventlog.get_recorder() is recorder
    finally:
        eventlog.set_recorder(None)
    assert eventlog.get_recorder() is None


# --- payload discipline: no free text ever reaches reason/data ----------------


def _emission_calls(path: str) -> list[ast.Call]:
    """Every ``record``/``record_job_event`` call in one emission-site module."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in ("record", "record_job_event"):
            calls.append(node)
    return calls


#: Every module that emits a durable event in P24a WP1.
EMISSION_SITES = [
    "src/nerdit/core/services.py",
    "src/nerdit/core/cutover.py",
    "src/nerdit/core/deploy_state.py",
    "src/nerdit/core/models/controller.py",
    "src/nerdit/core/data/controller.py",
    "src/nerdit/daemon/deploy_pipeline.py",
]

#: Names a call site is allowed to pass as a ``reason``/``data`` value. All of
#: them are already-machine-shaped: ``ErrorClass``/``last_deploy`` tokens, the
#: loop's own counters, or a projection off the settled phase blob.
#: (P24b) ``stage`` joins them: it is a pinned two-value literal
#: (``"probe"`` | ``"repoint"``) chosen by the cutover's own settle branch,
#: never derived from container output.
_ALLOWED_VALUE_NAMES = {
    "reason",
    "exit_code",
    "new_count",
    "failures",
    "error_class",
    "version",
    "stage",
    # (P33 D-GH-10) The settled generation's provenance and its remediation
    # rule id: a canonical ``host/owner/repo``, a validated ref, a 40-hex sha
    # and a ``RemediationCode`` member — projections off the phase blob, never
    # container output.
    "repo",
    "ref",
    "sha",
    "remediation_code",
}

#: Names that must NEVER be passed: they interpolate container output.
_FORBIDDEN_VALUE_NAMES = {"message", "error_message", "exc", "detail", "stderr", "log_tail"}


def _value_is_machine_shaped(node: ast.AST) -> bool:
    """Whether an emission-site argument is provably not free text."""
    if isinstance(node, ast.Constant):
        # A literal is a fixed token/int/bool/None chosen at author time, never
        # runtime text. An f-string is an ast.JoinedStr and fails here.
        return isinstance(node.value, str | int | float | bool | type(None))
    if isinstance(node, ast.Name):
        return node.id in _ALLOWED_VALUE_NAMES
    if isinstance(node, ast.Attribute):
        # ``ErrorClass.user_error.value`` / ``some_enum.value`` — an enum member.
        return node.attr == "value" or _value_is_machine_shaped(node.value)
    if isinstance(node, ast.Call):
        # ``_classify_service_exit(...).value`` — an ErrorClass member.
        return isinstance(node.func, ast.Attribute) and node.func.attr == "value"
    if isinstance(node, ast.IfExp):
        return _value_is_machine_shaped(node.body) and _value_is_machine_shaped(node.orelse)
    # ``new_ld["reason"]`` — a projection off the machine-shaped phase blob.
    return isinstance(node, ast.Subscript)


@pytest.mark.parametrize("path", EMISSION_SITES)
def test_no_free_text_reaches_reason_or_data(path):
    """The D-P24-3 rule-1 guard, enforced structurally rather than by review.

    An ``error_message`` routinely quotes container output; the feed is
    any-authenticated AND is POSTed to a third-party host in P24c. So every
    ``reason`` value and every ``data`` value at every emission site must be a
    literal token, an int/bool, an enum member, or a named counter — never an
    f-string, never a caught exception, never a message variable.
    """
    calls = _emission_calls(path)
    assert calls, f"{path} has no emission call — the site list is stale"
    for call in calls:
        for kw in call.keywords:
            if kw.arg == "reason":
                assert _value_is_machine_shaped(kw.value), (
                    f"{path}:{kw.value.lineno} passes a non-machine-shaped reason"
                )
            elif kw.arg == "data":
                # Walk rather than isinstance-check: a site may pass the dict
                # behind a conditional (``{...} if present else None``).
                dicts = [n for n in ast.walk(kw.value) if isinstance(n, ast.Dict)]
                assert dicts or _value_is_machine_shaped(kw.value), (
                    f"{path}:{kw.value.lineno} passes an opaque data value"
                )
                for literal in dicts:
                    for value in literal.values:
                        assert _value_is_machine_shaped(value), (
                            f"{path}:{value.lineno} passes a non-machine-shaped data value"
                        )


@pytest.mark.parametrize("path", EMISSION_SITES)
def test_no_forbidden_name_is_ever_emitted(path):
    """Belt-and-braces over the allowlist: the message-shaped names, by name."""
    for call in _emission_calls(path):
        for kw in call.keywords:
            if kw.arg not in ("reason", "data"):
                continue
            for node in ast.walk(kw.value):
                if isinstance(node, ast.Name):
                    assert node.id not in _FORBIDDEN_VALUE_NAMES, (
                        f"{path}:{node.lineno} emits {node.id!r} into {kw.arg}"
                    )
                assert not isinstance(node, ast.JoinedStr), (
                    f"{path}:{kw.value.lineno} emits an f-string into {kw.arg}"
                )


# --- wiring: the controller really writes rows -------------------------------


def _service_controller(queries, recorder):
    from nerdit.config.settings import ServicesSettings
    from nerdit.core.runtime.stub import StubRuntime
    from nerdit.core.services import ServiceController

    return ServiceController(
        queries=queries,
        runtime=StubRuntime(),
        services_settings=ServicesSettings(service_port_range="9400-9499", service_max_restarts=1),
        events=recorder,
    )


async def _zero_exit(container_id, timeout_s=None):
    """A container that exited cleanly (``StubRuntime.wait`` always raises)."""
    return 0


async def _noop_remove(container_id, force=False):
    return None


async def _service_row(queries, name="wired"):
    job = Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config='{"build_version": 2}',
    )
    await queries.create_job(job)
    return job


async def test_teardown_to_stopped_writes_a_feed_row(db, queries):
    controller = _service_controller(queries, EventRecorder(queries))
    job = await _service_row(queries, "wired-stop")

    await controller._teardown_to_stopped(job)

    rows = await _all_events(db)
    assert [(r["type"], r["service_name"], r["build_version"]) for r in rows] == [
        ("service.stopped", "wired-stop", 2)
    ]


async def test_restart_budget_exhaustion_writes_the_killer_event(db, queries):
    """The 03:00 crash-loop settle — the event the whole phase exists for."""
    from datetime import UTC, datetime

    controller = _service_controller(queries, EventRecorder(queries))
    job = await _service_row(queries, "wired-fail")

    # max_restarts is 1, so the second counted restart exhausts the budget.
    await controller._count_restart(job, datetime.now(UTC), exit_code=137, oom_killed=True)
    refreshed = await queries.get_job(job.id)
    await controller._count_restart(refreshed, datetime.now(UTC), exit_code=137, oom_killed=True)

    rows = await _all_events(db)
    types = [r["type"] for r in rows]
    assert types == ["service.restarting", "service.failed"]
    assert rows[-1]["reason"] == "crash_loop"
    payload = json.loads(rows[-1]["data"])
    assert payload["exit_code"] == 137
    assert payload["error_class"] == ErrorClass.oom.value
    assert "error_message" not in payload


async def test_restart_policy_no_terminal_failure_writes_a_feed_row(db, queries):
    """The OTHER terminal-failure path: ``restart_policy='no'`` never reaches
    ``_count_restart``, so it needs its own emission or the settle is silent."""
    from datetime import UTC, datetime

    controller = _service_controller(queries, EventRecorder(queries))
    job = Job(
        name="wired-nopolicy",
        kind=JobKind.service,
        service_name="wired-nopolicy",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="no",
        config='{"build_version": 3}',
    )
    await queries.create_job(job)

    await controller._handle_crash(job, datetime.now(UTC))

    settled = await queries.get_job(job.id)
    assert settled.status is JobStatus.failed

    rows = await _all_events(db)
    assert [r["type"] for r in rows] == ["service.failed"]
    assert rows[0]["service_name"] == "wired-nopolicy"
    assert rows[0]["build_version"] == 3
    assert rows[0]["reason"] == "exited"
    payload = json.loads(rows[0]["data"])
    # No container to inspect ⇒ no exit code ⇒ the UNKNOWN class. What matters
    # is that both keys are machine-shaped and the error_message is absent.
    assert payload["exit_code"] is None
    assert payload["error_class"] == ErrorClass.unknown.value
    assert "error_message" not in payload


async def test_clean_exit_settles_completed_and_writes_nothing(db, queries):
    """``completed`` is not a failure — the terminal branch must stay silent."""
    from datetime import UTC, datetime

    controller = _service_controller(queries, EventRecorder(queries))
    controller._runtime.wait = _zero_exit  # type: ignore[method-assign]
    controller._runtime.remove = _noop_remove  # type: ignore[method-assign]
    job = Job(
        name="wired-clean",
        kind=JobKind.service,
        service_name="wired-clean",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        container_id="c-clean",
        config='{"build_version": 1}',
    )
    await queries.create_job(job)

    await controller._handle_crash(job, datetime.now(UTC))

    assert (await queries.get_job(job.id)).status is JobStatus.completed
    assert await _all_events(db) == []


async def test_controller_without_a_recorder_writes_nothing(db, queries):
    controller = _service_controller(queries, None)
    job = await _service_row(queries, "wired-none")

    await controller._teardown_to_stopped(job)

    assert await _all_events(db) == []


# --- the bypass settle: a terminal phase written outside the writer ----------


async def test_redeploy_over_live_build_failure_records_deploy_failed(db, queries, tmp_path):
    """The bypass settle must not be silent (the P24c live-run find).

    A redeploy whose build fails over a still-live old container reverts the
    row and stamps ``phase=failed`` straight into the blob — never through
    ``stamp_last_deploy``, whose tail owns the durable emit. Before the fix a
    consumer saw ``service.deploy_started`` and then permanent silence for that
    generation, which D-P24-3 forbids.
    """
    from datetime import UTC, datetime

    from tests.test_services_reconcile import (
        _controller,
        _deploy_svc,
        _drain_builds,
        _FailingBuildRuntime,
    )

    runtime = _FailingBuildRuntime()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = datetime.now(UTC)  # the old version keeps serving
    controller = _controller(queries, runtime)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    await queries.create_job(
        _deploy_svc(
            "app",
            version=2,
            action="redeploy",
            ctx=ctx,
            status=JobStatus.restarting,
            container_id="c-old",
            previous_image="nerdit-app/app:1",
        )
    )

    eventlog.set_recorder(EventRecorder(queries))
    try:
        await controller.reconcile()
        await _drain_builds(controller)
    finally:
        eventlog.set_recorder(None)
        await controller.shutdown()

    rows = [r for r in await _all_events(db) if r["type"] == "service.deploy_failed"]
    assert len(rows) == 1
    assert rows[0]["service_name"] == "app"
    assert rows[0]["reason"] == "build_failed"
    assert rows[0]["build_version"] == 2
    assert json.loads(rows[0]["data"]) == {
        **_NO_PROVENANCE,
        "error_class": ErrorClass.user_error.value,
    }
    # D-P24-3 rule 1: the builder's ``error_message`` quotes container output.
    assert "npm install failed" not in json.dumps(dict(rows[0]), default=str)


async def test_a_bypass_settled_generation_never_re_emits_through_the_writer(db, queries):
    """Double-emit safety: the writer's tail keys on the PRIOR phase.

    Once the bypass has stamped ``failed``, a later ``stamp_last_deploy`` over
    the same row still WRITES (the phase machine is unchanged) but records
    nothing — so the settle event stays exactly-once per generation.
    """
    from nerdit.core.deploy_state import record_deploy_settled, stamp_last_deploy

    settled = {"version": 4, "phase": "failed", "reason": "build_failed"}
    job = Job(
        name="resettled",
        kind=JobKind.service,
        service_name="resettled",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config=json.dumps({"build_version": 4, "last_deploy": settled}),
    )
    await queries.create_job(job)

    eventlog.set_recorder(EventRecorder(queries))
    try:
        await record_deploy_settled("resettled", settled)  # the bypass emit
        assert await stamp_last_deploy(queries, job.id, phase="healthy") is True
    finally:
        eventlog.set_recorder(None)

    assert [r["type"] for r in await _all_events(db)] == ["service.deploy_failed"]


def test_emit_status_change_is_untouched():
    """D-P24-3 rule 3: ``job.status_changed`` stays a bus-only legacy event."""
    from nerdit.core.services import ServiceController

    source = inspect.getsource(ServiceController._emit_status_change)
    assert "record" not in source
    assert "job.status_changed" in source


async def test_link_session_edges_are_never_coalesced(db, queries):
    """PR #114 review: all ``link.connected`` rows share one coalescing key
    (no service_name/reason/build_version), so the flap guard would swallow
    the reconnect in a connect → error → reconnect cycle inside one window —
    and a consumer could never reconstruct the session state. Session EDGES
    are exempt: every occurrence is the signal."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("link.connected", data={"node_id": "n1"})
    await recorder.record("link.disconnected", reason="error", data={"node_id": "n1"})
    await recorder.record("link.connected", data={"node_id": "n1"})
    await recorder.record("link.disconnected", reason="error", data={"node_id": "n1"})

    rows = await _all_events(db)
    assert [r["type"] for r in rows] == [
        "link.connected",
        "link.disconnected",
        "link.connected",
        "link.disconnected",
    ]


async def test_license_edges_are_never_coalesced(db, queries):
    """(P17d D-LIC6) Same structural reason as the link edges: the three
    ``license.*`` types share a coalescing key (no service_name, no
    build_version, and ``license.rejected`` repeats one ``reason``), so an
    install → remove → install sequence inside one window would collapse to a
    single row and leave a consumer unable to reconstruct what the operator
    did. Every occurrence of an admin-action / boot edge is the signal."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("license.installed", data={"lid": "abc", "plan": "pro", "state": "valid"})
    await recorder.record("license.removed", data={"lid": "abc"})
    await recorder.record("license.installed", data={"lid": "abc", "plan": "pro", "state": "valid"})
    await recorder.record("license.rejected", reason="bad_signature")
    await recorder.record("license.rejected", reason="bad_signature")

    rows = await _all_events(db)
    assert [r["type"] for r in rows] == [
        "license.installed",
        "license.removed",
        "license.installed",
        "license.rejected",
        "license.rejected",
    ]


async def test_neither_share_event_is_ever_coalesced(db, queries):
    """(P26 WP-H) Both share types are owner-action EDGES whose every
    occurrence is the signal: all rows of one type for one service share a
    coalescing key, so a share → unshare → share → unshare cycle inside one
    window would collapse and leave a consumer believing the app is still
    exposed — or still private."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    for _ in range(2):
        await recorder.record("share.ready", service_name="demo", data={"access": "private"})
        await recorder.record("share.removed", service_name="demo", reason="unshared")

    rows = await _all_events(db)
    assert [r["type"] for r in rows] == [
        "share.ready",
        "share.removed",
        "share.ready",
        "share.removed",
    ]


async def test_an_access_flip_inside_the_window_is_never_dropped(db, queries):
    """The reason ``share.ready`` needs the exemption (PR review, P26 WP-H).

    The fact it carries lives in ``data``, and ``data`` is NOT part of the
    coalescing key — so a private→public flip inside one window is a different
    fact under an identical key. Dropping it would leave the durable feed (and
    every webhook consumer reading it) saying "private" about an app that is
    now world-reachable."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("share.ready", service_name="demo", data={"access": "private"})
    await recorder.record("share.ready", service_name="demo", data={"access": "public"})

    rows = await _all_events(db)
    assert [json.loads(r["data"])["access"] for r in rows] == ["private", "public"]


async def test_share_removed_reasons_are_distinguishable(db, queries):
    """``unshared`` (the owner) and ``service_deleted`` (the delete route) are
    different facts and must both reach the feed."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("share.removed", service_name="demo", reason="unshared")
    await recorder.record("share.removed", service_name="demo", reason="service_deleted")

    assert [r["reason"] for r in await _all_events(db)] == ["unshared", "service_deleted"]


async def test_neither_domain_event_is_ever_coalesced(db, queries):
    """(P26 WP1) Both domain types are owner-action EDGES, exactly like the
    share pair: all rows of one type for one service share a coalescing key, so
    an add → remove → add cycle inside one window would collapse and leave a
    consumer believing the node no longer answers for the name."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    for _ in range(2):
        await recorder.record(
            "domain.added", service_name="demo", data={"domain": "a.example.com", "acme": False}
        )
        await recorder.record(
            "domain.removed",
            service_name="demo",
            reason="removed",
            data={"domain": "a.example.com"},
        )

    rows = await _all_events(db)
    assert [r["type"] for r in rows] == [
        "domain.added",
        "domain.removed",
        "domain.added",
        "domain.removed",
    ]


async def test_two_different_domains_added_in_one_window_both_land(db, queries):
    """The reason the exemption matters most here: the fact that distinguishes
    two ``domain.added`` rows lives in ``data``, and ``data`` is NOT part of the
    coalescing key. Dropping the second would leave the feed naming one of the
    two names this node now answers for."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("domain.added", service_name="demo", data={"domain": "a.example.com"})
    await recorder.record("domain.added", service_name="demo", data={"domain": "b.example.com"})

    rows = await _all_events(db)
    assert [json.loads(r["data"])["domain"] for r in rows] == ["a.example.com", "b.example.com"]


async def test_domain_removed_reasons_are_distinguishable(db, queries):
    """``removed`` (the owner) and ``service_deleted`` (the purge cascade) are
    different facts and must both reach the feed."""
    recorder = EventRecorder(queries, coalesce_window_s=60.0)

    await recorder.record("domain.removed", service_name="demo", reason="removed")
    await recorder.record("domain.removed", service_name="demo", reason="service_deleted")

    assert [r["reason"] for r in await _all_events(db)] == ["removed", "service_deleted"]
