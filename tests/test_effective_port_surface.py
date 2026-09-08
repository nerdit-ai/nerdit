"""Expose both reserved and effective ports throughout a promoted generation.

The green container remains on ephemeral port G while host_port reserves P.
Service URLs, route dial comparisons and diagnose probes must use G. With no
active_host_port, both fields equal P and behavior remains unchanged.

Round-trip cutover/auto_deploy through app config without requires_restart, and
never overwrite an armed cutover_pending marker. Use real routers/auth with
mocked state and no lifespan.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.proxy import LiveRoute
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.app_config import router as app_config_router
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.models import (
    ApiToken,
    Job,
    JobKind,
    JobStatus,
    RouteEndpoint,
    ServiceEndpoint,
    TokenRole,
)

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
_AUTH = {"Authorization": f"Bearer {LEGACY}"}

# The stable allocation (P) and the transient green (G) of D-P24-4b.
STABLE_PORT = 8101
GREEN_PORT = 41777

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
}


# --- GET /services/{ident} -----------------------------------------------------


def _service_job(**over) -> Job:
    fields = dict(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config=json.dumps({"image": "nerdit-app/demo:2", "port": 8000}),
        submitted_by_token="tok-sub",
    )
    fields.update(over)
    return Job(**fields)


def _endpoint(active: int | None) -> ServiceEndpoint:
    return ServiceEndpoint(
        service_name="demo",
        job_id="svc-1",
        container_port=8000,
        host_port=STABLE_PORT,
        active_host_port=active,
        route="/demo",
    )


def _services_client(endpoint: ServiceEndpoint, tmp_path) -> TestClient:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.touch_api_token = AsyncMock()
    q.insert_audit_log = AsyncMock()
    q.get_job = AsyncMock(return_value=_service_job())
    q.get_service_by_name = AsyncMock(return_value=_service_job())
    q.get_service_endpoint = AsyncMock(return_value=endpoint)
    q.get_job_gpus = AsyncMock(return_value=[])
    # (P26 D-P26-14) No shares on this node — the port ruling is what is
    # under test, and ``public_urls`` must not perturb it.
    q.list_service_shares = AsyncMock(return_value={})

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)
    app.state.queries = q
    # ``data_dir`` must be a real path: the detail route walks the named-volume
    # tree off-loop and only a missing dir short-circuits to None.
    app.state.settings = SimpleNamespace(data_dir=str(tmp_path))
    # No proxy manager ⇒ ``available`` is False ⇒ ``public_url`` stays None, so
    # this test isolates the loopback URL (the field the ruling is about).
    app.state.proxy_manager = None
    app.state.hostname = "box"
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def test_service_detail_surfaces_both_ports_and_dials_the_green(tmp_path):
    """The promoted-cutover window: P stays advertised, the URL names G."""
    client = _services_client(_endpoint(GREEN_PORT), tmp_path)
    body = client.get("/services/demo", headers=_AUTH).json()
    ep = body["endpoint"]

    assert ep["host_port"] == STABLE_PORT  # the stable allocation, unchanged
    assert ep["effective_host_port"] == GREEN_PORT
    # The advertised loopback URL must reach the live container, not the port
    # nothing is listening on for the whole healthy generation.
    assert ep["url"] == f"http://127.0.0.1:{GREEN_PORT}"


def test_service_detail_is_byte_compatible_with_no_cutover(tmp_path):
    """NULL ``active_host_port``: both fields equal ``host_port``, url unchanged."""
    client = _services_client(_endpoint(None), tmp_path)
    ep = client.get("/services/demo", headers=_AUTH).json()["endpoint"]

    assert ep["host_port"] == STABLE_PORT
    assert ep["effective_host_port"] == STABLE_PORT
    assert ep["url"] == f"http://127.0.0.1:{STABLE_PORT}"


# --- GET /routes ---------------------------------------------------------------


def _route_endpoint(active: int | None) -> RouteEndpoint:
    return RouteEndpoint(
        service_name="demo",
        kind=JobKind.service,
        status=JobStatus.running,
        container_port=8000,
        host_port=STABLE_PORT,
        active_host_port=active,
        route="/demo",
    )


def _routes_client(endpoint: RouteEndpoint, *, dial_port: int) -> TestClient:
    q = SimpleNamespace(
        list_service_endpoints=AsyncMock(return_value=([endpoint], None)),
        # (P26 D-P26-14) The one batched share read ``GET /routes`` makes.
        list_service_shares=AsyncMock(return_value={}),
    )
    mgr = SimpleNamespace(
        enabled=True,
        available=True,
        live_routes=AsyncMock(
            return_value={
                "nerdit-route-demo": LiveRoute(dial=f"127.0.0.1:{dial_port}", shape="path")
            }
        ),
        build_route=lambda name, port: SimpleNamespace(caddy_id=f"nerdit-route-{name}"),
    )
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(proxy_router)
    app.include_router(api)
    app.state.proxy_manager = mgr
    app.state.queries = q
    app.state.settings = SimpleNamespace(
        proxy=SimpleNamespace(
            mode="path", base_domain=None, scheme="https", https_port=443, public_port=None
        )
    )
    app.state.hostname = "box"
    app.state.mdns_advertiser = None
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def test_routes_surfaces_both_ports_and_matches_the_green_dial():
    """A correctly-repointed cutover route reads CONVERGED, not drifted."""
    client = _routes_client(_route_endpoint(GREEN_PORT), dial_port=GREEN_PORT)
    item = client.get("/api/routes", headers=_AUTH).json()["items"][0]

    assert item["host_port"] == STABLE_PORT
    assert item["effective_host_port"] == GREEN_PORT
    assert item["live"] == {"registered": True, "dial_matches": True}


def test_routes_dial_matches_is_false_when_the_dial_still_names_the_stable_port():
    """The negative pin: mid-repoint the proxy still dials P — that IS drift.

    This is the assertion that would silently invert if ``dial_matches`` were
    ever moved back onto ``host_port``: it would then read TRUE here (the wrong
    answer, the route has not been repointed yet) and FALSE in the test above.
    """
    client = _routes_client(_route_endpoint(GREEN_PORT), dial_port=STABLE_PORT)
    item = client.get("/api/routes", headers=_AUTH).json()["items"][0]

    assert item["effective_host_port"] == GREEN_PORT
    assert item["live"] == {"registered": True, "dial_matches": False}


def test_routes_is_byte_compatible_with_no_cutover():
    client = _routes_client(_route_endpoint(None), dial_port=STABLE_PORT)
    item = client.get("/api/routes", headers=_AUTH).json()["items"][0]

    assert item["host_port"] == STABLE_PORT
    assert item["effective_host_port"] == STABLE_PORT
    assert item["live"] == {"registered": True, "dial_matches": True}


# --- /diagnose fresh probe -----------------------------------------------------


@pytest.mark.asyncio
async def test_diagnose_probe_dials_the_effective_port(monkeypatch):
    """The fresh probe must reach the green, or a healthy service reads dead."""
    import nerdit.daemon.routes.service_diagnose as diag

    dialled: list[int] = []

    async def _fake_check_health(port, path, timeout):  # noqa: ANN001
        dialled.append(port)
        return 200

    monkeypatch.setattr(diag, "check_health", _fake_check_health)

    job = _service_job(container_id="c-green", health_check={"path": "/health"})
    probe = await diag._fresh_health_probe(job, _endpoint(GREEN_PORT))
    assert probe is not None
    assert probe["status_code"] == 200
    assert dialled == [GREEN_PORT]

    dialled.clear()
    await diag._fresh_health_probe(job, _endpoint(None))
    assert dialled == [STABLE_PORT]


# --- [deploy].cutover / [deploy].auto_deploy -----------------------------------


def _app_job(**config_extra) -> Job:
    config = {
        "image": "nerdit-app/demo:2",
        "build_version": 2,
        "port": 8000,
        "command": "npm start",
        "config_source": "deploy",
        "config_revision": 3,
    }
    config.update(config_extra)
    return Job(
        id="job-demo-0001",
        kind=JobKind.service,
        service_name="demo",
        name="demo",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        health_check={"path": "/health"},
        config=json.dumps(config),
        submitted_by_token="tok-sub",
    )


def _config_client(job: Job) -> tuple[TestClient, AsyncMock]:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # The write path re-reads the row to graft the build-tier keys off a FRESH
    # blob; answering with the same row models a steady state.
    q.get_job = AsyncMock(return_value=job)
    q.update_app_config = AsyncMock()

    def _current(_name: str) -> Job:
        # The handler re-reads the row to build the response view; reflect what
        # was actually persisted so the view assertions test the route, not the
        # mock's memory of the pre-write blob.
        call = q.update_app_config.call_args
        if call is None:
            return job
        return job.model_copy(update={"config": call.args[1]})

    q.get_service_by_name = AsyncMock(side_effect=_current)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(app_config_router)
    app.state.queries = q
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False), q


def _written_config(q: AsyncMock) -> dict:
    return json.loads(q.update_app_config.call_args.args[1])


def _put_deploy(client: TestClient, body: dict):
    return client.put(
        "/config/apps/demo/deploy",
        json=body,
        headers={"Authorization": f"Bearer {SUB_RAW}", "Idempotency-Key": "ik-1"},
    )


def test_cutover_and_auto_deploy_round_trip_without_requiring_a_restart():
    client, q = _config_client(_app_job())
    resp = _put_deploy(client, {"cutover": False, "auto_deploy": True})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True
    # The P20 asymmetry: next-deploy keys a restart would not apply.
    assert body["requires_restart"] is False
    assert body["restarted"] is False
    assert body["view"]["deploy"]["cutover"] is False
    assert body["view"]["deploy"]["auto_deploy"] is True

    written = _written_config(q)
    # ``False`` must PERSIST — it is the opt-out, not an absent value.
    assert written["cutover"] is False
    assert written["auto_deploy"] is True


def test_cutover_null_deletes_back_to_the_daemon_default():
    """null is the documented disarm: the key is dropped, not written False."""
    client, q = _config_client(_app_job(cutover=False))
    resp = _put_deploy(client, {"cutover": None})

    assert resp.status_code == 200, resp.text
    assert resp.json()["view"]["deploy"]["cutover"] is None
    assert "cutover" not in _written_config(q)


def test_cutover_write_cannot_clobber_an_armed_cutover_pending_marker():
    """``cutover_pending`` is build-tier state; the ETag does not cover it.

    A PUT rewrites the WHOLE config blob with no CAS, so losing the marker here
    would strand a green container and leave ``active_host_port`` naming a dead
    port on a ``degraded`` row that is never relaunched (D-P24-4b).
    """
    marker = {"version": 7, "blue": "c-blue"}
    client, q = _config_client(_app_job(cutover_pending=marker))
    resp = _put_deploy(client, {"cutover": False})

    assert resp.status_code == 200, resp.text
    assert _written_config(q)["cutover_pending"] == marker


def test_a_bad_cutover_value_is_a_structured_422():
    client, _ = _config_client(_app_job())
    resp = _put_deploy(client, {"cutover": "sometimes"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid"


def test_cutover_is_writable_but_never_restart_shaped():
    from nerdit.daemon.routes.app_config import (
        _BUILD_TIER_KEYS,
        _DEPLOY_MUTABLE_KEYS,
        _DEPLOY_TRACKED_KEYS,
        _RESTART_DEPLOY_KEYS,
    )

    for key in ("cutover", "auto_deploy"):
        assert key in _DEPLOY_MUTABLE_KEYS
        assert key in _DEPLOY_TRACKED_KEYS
        assert key not in _RESTART_DEPLOY_KEYS
    # The marker rides with ``release_pending`` in the build tier.
    assert "cutover_pending" in _BUILD_TIER_KEYS


def test_deploy_config_declares_the_two_keys_as_tri_state():
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="demo")
    assert cfg.cutover is None and cfg.auto_deploy is None
    assert DeployConfig(name="demo", cutover=False).cutover is False
    with pytest.raises(ValidationError):
        DeployConfig(name="demo", cutover="sometimes")


def test_deploy_keys_carry_forward_across_a_silent_redeploy():
    """A source that stays silent must not disarm a prior opt-out (the
    ``release`` carry-forward shape ``_eligible`` depends on)."""
    from pathlib import Path

    from nerdit.core.builder import BuildPlan
    from nerdit.daemon.deploy_pipeline import assemble_build_fields, resolve_effective_fields

    effective = resolve_effective_fields(
        name="demo",
        port=None,
        gpus=None,
        start=None,
        health=None,
        zip_deploy={},  # the source declares neither key
        existing=_app_job(),
        prev_cfg={"cutover": False, "auto_deploy": True},
    )
    assert effective.deploy_cfg.cutover is False
    assert effective.deploy_cfg.auto_deploy is True

    plan = BuildPlan(
        language="node",
        dockerfile_name="Dockerfile",
        dockerfile_text=None,
        port=8000,
        start_command=None,
    )
    fields, _ = assemble_build_fields(
        name="demo",
        plan=plan,
        context_dir=Path("/tmp/ctx"),
        context_root=None,
        deploy_cfg=effective.deploy_cfg,
    )
    assert fields["cutover"] is False
    assert fields["auto_deploy"] is True


def test_a_declared_source_value_overrides_the_carried_one():
    """The source's own ``[deploy].cutover`` wins over the prior row's value."""
    from nerdit.daemon.deploy_pipeline import resolve_effective_fields

    effective = resolve_effective_fields(
        name="demo",
        port=None,
        gpus=None,
        start=None,
        health=None,
        zip_deploy={"cutover": True},
        existing=_app_job(),
        prev_cfg={"cutover": False},
    )
    assert effective.deploy_cfg.cutover is True
