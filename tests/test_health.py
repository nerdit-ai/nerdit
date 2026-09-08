"""Tests for ``core/health.py`` (WP13 — F6-PROBE-DUP unification).

Two layers:

* the §1.5-item-1 divergence pins — written RED against today's split, where a
  string ``health_check.timeout_s`` resolves at 5.0s in the reconcile loop
  (``core/services.py``) but silently falls back to 2.0s in ``/diagnose``
  (``daemon/routes/services.py``, an ``isinstance`` guard), and the sibling
  ``start_period_s`` coercion asymmetry in ``_within_start_period``; and
* unit tests over the new ``core/health.py`` primitives themselves
  (``as_float``/``as_int`` junk tolerance, ``within_start_period``,
  ``run_probe`` dispatch) once they exist.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nerdit.db.models import Job, JobKind, JobStatus, ServiceEndpoint


def _job(**over: object) -> Job:
    fields: dict = dict(
        id="svc-1",
        service_name="my-app",
        name="my-app",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        restart_count=0,
    )
    fields.update(over)
    return Job(**fields)


# --- §1.5 item 1: divergence pin (timeout_s) ---------------------------------


@pytest.mark.asyncio
async def test_reconcile_side_coerces_string_timeout(queries):
    """Reconcile-loop side of the pin: a string ``timeout_s`` probes at 5.0s
    (GREEN today via ``_as_float`` — must stay GREEN after the unification)."""
    from tests.test_services_reconcile import FakeRuntime, _controller, _svc

    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    started = datetime.now(UTC) - timedelta(seconds=60)
    svc = _svc(
        "svc",
        status=JobStatus.running,
        container_id="c1",
        started_at=started,
        health_check={"path": "/healthz", "timeout_s": "5"},
    )
    await queries.create_job(svc)
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    captured: dict = {}

    async def capture(host_port, path, timeout):
        captured["timeout"] = timeout
        return 200

    controller._check_health = capture  # type: ignore[assignment]
    await controller.reconcile()
    assert captured["timeout"] == 5.0


@pytest.mark.asyncio
async def test_diagnose_side_coerces_string_timeout(monkeypatch):
    """§1.5 item 1: ``/diagnose``'s fresh probe must resolve a string
    ``timeout_s`` identically (5.0s) — RED today (falls back to 2.0s)."""
    from nerdit.daemon.routes import service_diagnose as svc_routes

    captured: dict = {}

    async def capture(host_port, path, timeout):
        captured["timeout"] = timeout
        return 200

    monkeypatch.setattr(svc_routes, "check_health", capture)

    job = _job(container_id="c1", health_check={"path": "/healthz", "timeout_s": "5"})
    endpoint = ServiceEndpoint(service_name="my-app", container_port=8000, host_port=9400)
    await svc_routes._fresh_health_probe(job, endpoint)
    assert captured["timeout"] == 5.0


# --- §1.5 item 1 family: start-period coercion asymmetry ---------------------


@pytest.mark.asyncio
async def test_diagnose_within_start_period_coerces_string_start_period():
    """``_within_start_period`` must coerce a string ``start_period_s`` via the
    same tolerant path as ``timeout_s`` — RED today (isinstance guard drops
    the string to the 0.0 default, so a still-warming service is reported
    as past its start period)."""
    from nerdit.daemon.routes import service_wait as wait_routes

    job = _job(
        started_at=datetime.now(UTC),
        health_check={"path": "/", "start_period_s": "3600"},
    )
    assert wait_routes._within_start_period(job) is True


@pytest.mark.asyncio
async def test_reconcile_side_coerces_string_start_period(queries):
    """Reconcile-loop side of the same family: a string ``start_period_s``
    already defers health judgement via ``_as_float`` (GREEN today — pin it
    survives the unification onto ``health.within_start_period``)."""
    from tests.test_services_reconcile import FakeRuntime, _controller, _svc

    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    runtime.live["c1"] = datetime.now(UTC)
    started = datetime.now(UTC)  # just started — well within any real window
    svc = _svc(
        "svc",
        status=JobStatus.running,
        container_id="c1",
        started_at=started,
        health_check={"path": "/healthz", "start_period_s": "3600"},
    )
    await queries.create_job(svc)
    await queries.acquire_service_port("svc", svc.id, 8000, (9400, 9499))

    called = False

    async def capture(host_port, path, timeout):
        nonlocal called
        called = True
        return 200

    controller._check_health = capture  # type: ignore[assignment]
    await controller.reconcile()
    # Still within the (coerced) 3600s start period → health judgement deferred,
    # so the probe must never have been called.
    assert called is False


# --- unit tests over core/health.py primitives -------------------------------


def test_as_float_tolerates_junk():
    from nerdit.core.health import as_float

    assert as_float("5", 2.0) == 5.0
    assert as_float(5, 2.0) == 5.0
    assert as_float(None, 2.0) == 2.0
    assert as_float("not-a-number", 2.0) == 2.0
    assert as_float(object(), 2.0) == 2.0


def test_as_int_tolerates_junk():
    from nerdit.core.health import as_int

    assert as_int("3", 1) == 3
    assert as_int(3, 1) == 3
    assert as_int(None, 1) == 1
    assert as_int("not-a-number", 1) == 1
    assert as_int(object(), 1) == 1


def test_within_start_period_no_health_check():
    from nerdit.core.health import within_start_period

    assert within_start_period(None, datetime.now(UTC)) is False
    assert within_start_period({}, datetime.now(UTC)) is False


def test_within_start_period_no_started_at():
    from nerdit.core.health import within_start_period

    assert within_start_period({"start_period_s": 3600}, None) is False


def test_within_start_period_naive_started_at_normalized_to_utc():
    from nerdit.core.health import within_start_period

    naive_now = datetime.now(UTC).replace(tzinfo=None)  # naive, mirrors a pre-P8 style row
    assert within_start_period({"start_period_s": 3600}, naive_now) is True


def test_within_start_period_zero_or_negative_period():
    from nerdit.core.health import within_start_period

    now = datetime.now(UTC)
    assert within_start_period({"start_period_s": 0}, now) is False
    assert within_start_period({"start_period_s": -5}, now) is False


def test_within_start_period_elapsed():
    from nerdit.core.health import within_start_period

    started = datetime.now(UTC) - timedelta(seconds=100)
    assert within_start_period({"start_period_s": 10}, started) is False
    assert within_start_period({"start_period_s": 1000}, started) is True


@pytest.mark.asyncio
async def test_run_probe_tcp_dispatch_synthesizes_200():
    from nerdit.core.health import run_probe

    async def http_check(host_port, path, timeout):
        raise AssertionError("must not call the http probe for a tcp blob")

    async def tcp_check(host_port, timeout):
        return True

    code, kind = await run_probe(
        {"type": "tcp"}, 9400, 2.0, http_check=http_check, tcp_check=tcp_check
    )
    assert (code, kind) == (200, "tcp")


@pytest.mark.asyncio
async def test_run_probe_tcp_dispatch_down_is_none():
    from nerdit.core.health import run_probe

    async def http_check(host_port, path, timeout):
        raise AssertionError("must not call the http probe for a tcp blob")

    async def tcp_check(host_port, timeout):
        return False

    code, kind = await run_probe(
        {"type": "tcp"}, 9400, 2.0, http_check=http_check, tcp_check=tcp_check
    )
    assert (code, kind) == (None, "tcp")


@pytest.mark.asyncio
async def test_run_probe_junk_type_falls_through_to_http():
    from nerdit.core.health import run_probe

    seen: dict = {}

    async def http_check(host_port, path, timeout):
        seen["path"] = path
        return 200

    async def tcp_check(host_port, timeout):
        raise AssertionError("must not call the tcp probe for an http/junk blob")

    code, kind = await run_probe(
        {"type": "grpc", "path": "/healthz"},
        9400,
        2.0,
        http_check=http_check,
        tcp_check=tcp_check,
    )
    assert (code, kind) == (200, "http")
    assert seen["path"] == "/healthz"


@pytest.mark.asyncio
async def test_run_probe_default_path():
    from nerdit.core.health import run_probe

    seen: dict = {}

    async def http_check(host_port, path, timeout):
        seen["path"] = path
        return 200

    async def tcp_check(host_port, timeout):
        raise AssertionError("must not call the tcp probe for an http blob")

    code, kind = await run_probe({}, 9400, 2.0, http_check=http_check, tcp_check=tcp_check)
    assert (code, kind) == (200, "http")
    assert seen["path"] == "/"
