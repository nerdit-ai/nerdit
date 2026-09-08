"""Test audit action mapping, middleware and database persistence.

Pin bounded action names, /api normalization and secret redaction. Routed
mutations record successes, errors and replays; publish audit events and retain
route-provided params across BaseHTTPMiddleware. Exercise cursor pagination and
secrets-at-rest masking against the database.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.requests import Request

from nerdit.core.events import EventBus
from nerdit.daemon.audit import AuditMiddleware, audit_params, derive_action, redact
from nerdit.daemon.auth import LEGACY_ADMIN, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.audit import router as audit_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.models import (
    ApiToken,
    AuditLogEntry,
    AuditLogPage,
    Job,
    JobKind,
    JobStatus,
    TokenRole,
)

# --- pure unit: derive_action -------------------------------------------------


def test_action_service_create():
    assert derive_action("POST", "/services") == ("service.create", "service", None)


def test_action_service_create_api_prefix_collapses():
    # Bare-root and /api mounts must yield the same action (no scope['route']).
    assert derive_action("POST", "/api/services")[0] == "service.create"


def test_action_service_stop_templated_with_target_id():
    action, target_type, target_id = derive_action("POST", "/services/abc123/stop")
    assert action == "service.stop"
    assert target_type == "service"
    assert target_id == "abc123"
    # The concrete id is NOT in the action string (bounded cardinality).
    assert "abc123" not in action


def test_action_app_config_put():
    # A two-group rule: the id comes from the first capture, not the last.
    assert derive_action("PUT", "/api/config/apps/xyz/ai") == (
        "config.app_update",
        "service",
        "xyz",
    )


def test_action_system_and_daemon():
    assert derive_action("POST", "/system/gc")[0] == "system.gc"
    assert derive_action("POST", "/api/daemon/restart")[0] == "daemon.restart"


def test_action_tokens():
    assert derive_action("POST", "/tokens")[0] == "token.create"
    assert derive_action("DELETE", "/tokens/tok-9") == ("token.revoke", "token", "tok-9")


def test_action_fallback_is_templated():
    # An unmapped mutation still logs with bounded cardinality.
    action, target_type, target_id = derive_action("POST", "/services/abc123/unknown")
    assert action == "POST /services/{id}/unknown"
    assert target_type is None
    assert target_id is None


def test_removed_batch_routes_have_no_action_rule():
    """The batch HTTP surface is gone; its audit rules went with it (WP2)."""
    for method, path in (
        ("POST", "/jobs"),
        ("POST", "/jobs/abc/stop"),
        ("POST", "/queue/reorder"),
        ("POST", "/cleanup"),
        ("POST", "/files/upload"),
    ):
        action, target_type, target_id = derive_action(method, path)
        assert action.startswith(f"{method} ")  # fallback, not a mapped action
        assert target_type is None and target_id is None


# --- pure unit: redaction -----------------------------------------------------


def test_redact_masks_secret_keys():
    out = redact(
        {
            "authorization": "Bearer x",
            "token": "nrd_secret",
            "api_key": "sk-123",
            "password": "hunter2",
            "secret": "s",
            "secrets": {"db": "p"},
            "name": "keep",
            "nested": {"auth_token": "x", "ok": 1},
        }
    )
    assert out["authorization"] == "***"
    assert out["token"] == "***"
    assert out["api_key"] == "***"
    assert out["password"] == "***"
    assert out["secret"] == "***"
    assert out["secrets"] == "***"
    assert out["name"] == "keep"
    assert out["nested"]["auth_token"] == "***"
    assert out["nested"]["ok"] == 1


def test_audit_params_from_pydantic_model():
    job = Job(id="j1", script_path="/tmp/x.py", gpu_count=1)
    params = audit_params(job)
    assert params["script_path"] == "/tmp/x.py"
    assert params["gpu_count"] == 1


def test_audit_params_masks_secret_in_dict():
    params = audit_params({"name": "n", "api_key": "sk-123"})
    assert params["api_key"] == "***"
    assert params["name"] == "n"


def test_redact_masks_all_env_values_by_value():
    # AUTHZ-1: env is a free-form map of user-chosen names → secret values, so
    # name-based matching can't catch keys like OPENAI_API_KEY. The whole map is
    # opaque: keys kept (which vars were set), every value masked.
    out = redact(
        {
            "name": "svc",
            "env": {
                "OPENAI_API_KEY": "sk-live-abc",
                "DB_PASSWORD": "hunter2",
                "LOG_LEVEL": "debug",
            },
        }
    )
    assert out["name"] == "svc"
    assert set(out["env"].keys()) == {"OPENAI_API_KEY", "DB_PASSWORD", "LOG_LEVEL"}
    assert all(v == "***" for v in out["env"].values())


# --- middleware: row recording via TestClient ---------------------------------

LEGACY = "legacy-global"
ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


def _service() -> Job:
    return Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "nerdit-runtime:0.1", "port": 8000}),
        submitted_by_token="tok-sub",
    )


def _queries() -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_job = AsyncMock(return_value=_service())
    q.get_service_by_name = AsyncMock(return_value=None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    q.set_desired_state = AsyncMock()
    q.list_audit_log = AsyncMock(return_value=([], None))
    return q


def _make_app(queries: AsyncMock, event_bus: EventBus | None = None) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)
    app.include_router(audit_router, prefix="/api")

    app.state.queries = queries
    if event_bus is not None:
        app.state.event_bus = event_bus
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime
    app.state.settings = MagicMock()

    # inner → outer: Audit added first (innermost), Auth next, RequestId last.
    app.add_middleware(
        AuditMiddleware,
        get_queries=lambda: queries,
        get_event_bus=lambda: event_bus,
    )
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _client(queries: AsyncMock, event_bus: EventBus | None = None) -> TestClient:
    return TestClient(_make_app(queries, event_bus), raise_server_exceptions=False)


def _create_body(**over) -> dict:
    base = {"name": "demo", "image": "nerdit-runtime:0.1", "port": 8000, "gpus": 0}
    base.update(over)
    return base


def test_service_create_records_audit_row():
    q = _queries()
    client = _client(q)
    resp = client.post("/services", json=_create_body(), headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "service.create"
    assert kwargs["result"] == "ok"
    assert kwargs["status_code"] == 201
    assert kwargs["principal_id"] == "tok-sub"
    assert kwargs["principal_role"] == "submitter"
    # The route's audit_params propagated across the middleware boundary.
    assert kwargs["params_redacted"] is not None
    assert "demo" in kwargs["params_redacted"]


def test_service_stop_records_templated_action():
    q = _queries()
    client = _client(q)
    resp = client.post("/services/svc-1/stop", headers=_auth(LEGACY))
    assert resp.status_code == 200
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "service.stop"
    assert kwargs["target_id"] == "svc-1"


def test_failed_validation_is_audited_as_error():
    q = _queries()
    client = _client(q)
    # Missing required image → 422 from the router, inside AuditMiddleware.
    resp = client.post("/services", json={"name": "demo", "port": 8000}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "service.create"
    assert kwargs["result"] == "error"
    assert kwargs["status_code"] == 422


def test_safe_get_is_not_audited():
    q = _queries()
    client = _client(q)
    client.get("/services/svc-1", headers=_auth(SUB_RAW))
    q.insert_audit_log.assert_not_awaited()


def test_idempotency_key_header_recorded():
    q = _queries()
    client = _client(q)
    client.post(
        "/services",
        json=_create_body(),
        headers={**_auth(SUB_RAW), "Idempotency-Key": "abc-123"},
    )
    assert q.insert_audit_log.await_args.kwargs["idempotency_key"] == "abc-123"


# --- audit route: admin-only + pagination -------------------------------------


def test_get_audit_admin_only():
    q = _queries()
    q.list_audit_log = AsyncMock(
        return_value=([AuditLogEntry(id=1, action="job.create", result="ok")], None)
    )
    client = _client(q)
    assert client.get("/api/audit", headers=_auth(RO_RAW)).status_code == 403
    assert client.get("/api/audit", headers=_auth(SUB_RAW)).status_code == 403
    resp = client.get("/api/audit", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["action"] == "job.create"


def test_get_audit_passes_pagination_params():
    q = _queries()
    client = _client(q)
    client.get("/api/audit?limit=10&cursor=42&result=denied", headers=_auth(ADMIN_RAW))
    kwargs = q.list_audit_log.await_args.kwargs
    assert kwargs["limit"] == 10
    assert kwargs["cursor"] == "42"
    assert kwargs["result"] == "denied"


def test_get_audit_passes_target_filter_params():
    q = _queries()
    client = _client(q)
    client.get("/api/audit?target=app-a&target_type=service", headers=_auth(ADMIN_RAW))
    kwargs = q.list_audit_log.await_args.kwargs
    assert kwargs["target"] == "app-a"
    assert kwargs["target_type"] == "service"


def test_get_audit_passes_p24_filter_params():
    """principal_id / action_prefix / since / until reach the query layer (WP3.1).

    The FastMCP lesson applies at the HTTP layer too: an unwired param makes a
    call look filtered when it is not, which is worse than an error.
    """
    q = _queries()
    client = _client(q)
    client.get(
        "/api/audit?principal_id=tok-7&action_prefix=deploy.&since=2026-08-07T10:00:00Z"
        "&until=2026-08-07T12:30:00Z",
        headers=_auth(ADMIN_RAW),
    )
    kwargs = q.list_audit_log.await_args.kwargs
    assert kwargs["principal_id"] == "tok-7"
    assert kwargs["action_prefix"] == "deploy."
    # Normalized out of ISO-8601 into the stored ``datetime('now')`` space
    # format — a raw 'T' string would compare lexically against the column and
    # put every row on the wrong side of the boundary.
    assert kwargs["since"] == "2026-08-07 10:00:00"
    assert kwargs["until"] == "2026-08-07 12:30:00"


def test_get_audit_normalizes_offset_timestamps_to_utc():
    q = _queries()
    client = _client(q)
    client.get("/api/audit?since=2026-08-07T12:00:00%2B02:00", headers=_auth(ADMIN_RAW))
    assert q.list_audit_log.await_args.kwargs["since"] == "2026-08-07 10:00:00"


def test_get_audit_rejects_out_of_grammar_action_prefix():
    """A prefix outside ``[a-z0-9._]{1,40}`` is 400 bad_request, not a wildcard."""
    q = _queries()
    client = _client(q)
    for bad in ("deploy*", "Deploy.", "a" * 41, "dep loy", "svc[a]"):
        resp = client.get(f"/api/audit?action_prefix={bad}", headers=_auth(ADMIN_RAW))
        assert resp.status_code == 400, bad
        assert resp.json()["code"] == "bad_request", bad
    q.list_audit_log.assert_not_awaited()


def test_get_audit_rejects_unparseable_timestamps():
    q = _queries()
    client = _client(q)
    resp = client.get("/api/audit?since=yesterday", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
    q.list_audit_log.assert_not_awaited()


def test_get_audit_p24_filters_still_admin_only():
    """The admin gate runs BEFORE the new validators.

    Otherwise a readonly token could probe the filter grammar (a 400 vs a 403
    is an oracle) on a surface it may not read at all.
    """
    q = _queries()
    client = _client(q)
    for raw in (RO_RAW, SUB_RAW):
        resp = client.get("/api/audit?action_prefix=NOT*VALID", headers=_auth(raw))
        assert resp.status_code == 403


def test_get_audit_target_filter_still_admin_only():
    # The admin gate stays first even with the new filter params present.
    q = _queries()
    client = _client(q)
    assert client.get("/api/audit?target=app-a", headers=_auth(RO_RAW)).status_code == 403
    assert client.get("/api/audit?target=app-a", headers=_auth(SUB_RAW)).status_code == 403


def test_audit_stream_admin_only():
    q = _queries()
    client = _client(q)
    # readonly: GET is a safe method so the coarse gate passes, but the route's
    # require_role(admin) rejects before the SSE response is built.
    assert client.get("/api/audit/stream", headers=_auth(RO_RAW)).status_code == 403


# --- middleware: replay + publish (direct _record, no SSE streaming) ----------


def _make_request(method: str, path: str, headers: dict | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "headers": raw_headers,
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("test", 1234),
    }
    req = Request(scope)
    req.state.request_id = "rid-1"
    req.state.principal = LEGACY_ADMIN
    return req


async def test_record_publishes_audit_event_on_bus():
    bus = EventBus()
    sub = bus.subscribe()
    q = AsyncMock()
    q.insert_audit_log = AsyncMock()
    mw = AuditMiddleware(app=MagicMock(), get_queries=lambda: q, get_event_bus=lambda: bus)
    req = _make_request("POST", "/services")
    req.state.audit_params = {"name": "demo"}
    await mw._record(req, status_code=201, result="ok")
    event = await sub.__anext__()
    assert event["type"] == "audit.service.create"
    assert event["result"] == "ok"
    assert event["params"] == {"name": "demo"}
    q.insert_audit_log.assert_awaited_once()


# --- middleware: audit_target stamp seam (PR1) --------------------------------


def _stamp_recorder() -> tuple[AuditMiddleware, dict, "EventBus"]:
    """A middleware wired to capture the recorded row + the published event."""
    bus = EventBus()
    captured: dict = {}
    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))
    mw = AuditMiddleware(app=MagicMock(), get_queries=lambda: q, get_event_bus=lambda: bus)
    return mw, captured, bus


async def test_stamp_fills_null_target_for_allowlisted_action():
    """An allowlisted action (deploy.create) whose path leaves target_id null gets
    the route's stamp on BOTH the DB row and the published bus event."""
    mw, captured, bus = _stamp_recorder()
    sub = bus.subscribe()
    req = _make_request("POST", "/deploy")
    req.state.audit_target = "my-app"
    await mw._record(req, status_code=201, result="ok")
    assert captured["action"] == "deploy.create"
    assert captured["target_type"] == "service"
    assert captured["target_id"] == "my-app"
    event = await sub.__anext__()
    assert event["target_type"] == "service"
    assert event["target_id"] == "my-app"


async def test_stamp_ignored_for_non_allowlisted_action():
    """A stamp on an action outside the allowlist (service.create) is dropped: the
    path-derived attribution is authoritative."""
    mw, captured, _ = _stamp_recorder()
    req = _make_request("POST", "/services")
    req.state.audit_target = "my-app"
    await mw._record(req, status_code=201, result="ok")
    assert captured["action"] == "service.create"
    assert captured["target_type"] == "service"
    assert captured["target_id"] is None


async def test_stamp_cannot_overwrite_path_target_for_fill_only_action(monkeypatch):
    """A fill-only allowlisted action never rewrites a non-null path-derived
    target — only template.deploy is sanctioned to re-point (guard below)."""
    import nerdit.daemon.audit as audit_module

    monkeypatch.setattr(
        audit_module, "derive_action", lambda method, path: ("deploy.create", "service", "path-id")
    )
    mw, captured, _ = _stamp_recorder()
    req = _make_request("POST", "/deploy")
    req.state.audit_target = "body-name"
    await mw._record(req, status_code=201, result="ok")
    assert captured["target_id"] == "path-id"  # stamp did NOT overwrite


async def test_template_deploy_stamp_repoints_target_to_service():
    """template.deploy is the one sanctioned rewrite: the path-derived (template,
    <tid>) becomes (service, <app>) while the template id stays in the params."""
    mw, captured, _ = _stamp_recorder()
    req = _make_request("POST", "/app-templates/my-tmpl/deploy")
    req.state.audit_target = "my-app"
    req.state.audit_params = {"template_id": "my-tmpl", "name": "my-app"}
    await mw._record(req, status_code=201, result="ok")
    assert captured["action"] == "template.deploy"
    assert captured["target_type"] == "service"
    assert captured["target_id"] == "my-app"
    # The template id survives in the event params (per-template usage stays queryable).
    assert json.loads(captured["params_redacted"])["template_id"] == "my-tmpl"


async def test_no_stamp_leaves_target_null_for_deploy_create():
    """No stamp set (route 4xx'd early, or a replay where the route never ran) →
    the null path is untouched, matching audit_params/audit_action semantics."""
    mw, captured, _ = _stamp_recorder()
    req = _make_request("POST", "/deploy")  # no request.state.audit_target
    await mw._record(req, status_code=422, result="error")
    assert captured["action"] == "deploy.create"
    assert captured["target_id"] is None


async def test_replay_records_result_replay_with_null_params():
    # A short-circuited idempotency replay (header set) is audited result='replay'
    # and, because the route never ran, params are null (audit_params unset).
    captured = {}

    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))

    async def replay_route(request: Request) -> Response:
        return Response(status_code=200, headers={"Idempotent-Replay": "true"})

    app = FastAPI()
    app.add_api_route("/replay-thing", replay_route, methods=["POST"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/replay-thing")
    assert resp.status_code == 200
    assert captured["result"] == "replay"
    assert captured["params_redacted"] is None


# --- middleware: route-level 401/403 relabelled 'denied' ----------------------


async def test_route_level_403_recorded_as_denied():
    """A route-level role/owner denial (past auth, denied in the route) is
    recorded ``result='denied'`` — joining the coarse-gate label so a
    ``denied``-filtered query does not miss it."""
    captured = {}
    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))

    async def forbidden_route(request: Request) -> Response:
        return Response(status_code=403)

    app = FastAPI()
    app.add_api_route("/admin-thing", forbidden_route, methods=["POST"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/admin-thing")
    assert resp.status_code == 403
    assert captured["result"] == "denied"
    assert captured["status_code"] == 403


async def test_route_level_401_recorded_as_denied():
    captured = {}
    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))

    async def unauth_route(request: Request) -> Response:
        return Response(status_code=401)

    app = FastAPI()
    app.add_api_route("/gated-thing", unauth_route, methods=["POST"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/gated-thing")
    assert resp.status_code == 401
    assert captured["result"] == "denied"


async def test_route_level_422_still_recorded_as_error():
    """A non-auth 4xx (e.g. validation) stays ``result='error'`` — only 401/403
    move to ``denied``."""
    captured = {}
    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))

    async def bad_request_route(request: Request) -> Response:
        return Response(status_code=422)

    app = FastAPI()
    app.add_api_route("/unprocessable-thing", bad_request_route, methods=["POST"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/unprocessable-thing")
    assert resp.status_code == 422
    assert captured["result"] == "error"


# --- middleware: the change-only opt-out (P32) --------------------------------


async def test_a_route_may_skip_recording_a_successful_no_op():
    """``request.state.audit_skip`` exists for repeated machine assertions —
    the cloud's entitlement push re-asserts on a timer, and a row per re-assert
    is noise carrying no information."""
    q = AsyncMock()
    q.insert_audit_log = AsyncMock()
    bus = MagicMock()

    async def unchanged_route(request: Request) -> Response:
        request.state.audit_skip = True
        return Response(status_code=200)

    app = FastAPI()
    app.add_api_route("/quiet-thing", unchanged_route, methods=["PUT"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: bus)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.put("/quiet-thing").status_code == 200
    q.insert_audit_log.assert_not_awaited()
    bus.publish.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 403, 422, 500])
async def test_the_skip_is_ignored_for_every_failure(status_code: int):
    """One-sided by design: a denial or an error is never a route's to
    suppress, so a future misuse of the flag cannot hide a 403."""
    captured = {}
    q = AsyncMock()
    q.insert_audit_log = AsyncMock(side_effect=lambda **kw: captured.update(kw))

    async def failing_route(request: Request) -> Response:
        request.state.audit_skip = True
        return Response(status_code=status_code)

    app = FastAPI()
    app.add_api_route("/loud-thing", failing_route, methods=["PUT"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.put("/loud-thing").status_code == status_code
    assert captured["status_code"] == status_code
    assert captured["result"] in {"denied", "error"}


async def test_a_route_that_sets_nothing_is_recorded_as_before():
    """The flag is opt-in: absence must not be read as a skip."""
    q = AsyncMock()
    q.insert_audit_log = AsyncMock()

    async def ordinary_route(request: Request) -> Response:
        return Response(status_code=200)

    app = FastAPI()
    app.add_api_route("/ordinary-thing", ordinary_route, methods=["PUT"])
    app.add_middleware(AuditMiddleware, get_queries=lambda: q, get_event_bus=lambda: None)
    client = TestClient(app, raise_server_exceptions=False)

    assert client.put("/ordinary-thing").status_code == 200
    q.insert_audit_log.assert_awaited_once()


# --- pure unit: the P32 route rule --------------------------------------------


def test_action_link_entitlement():
    assert derive_action("PUT", "/link/entitlement") == ("link.entitlement", "link", None)
    assert derive_action("PUT", "/api/link/entitlement")[0] == "link.entitlement"


def test_action_link_github_token():
    assert derive_action("PUT", "/link/github-token") == ("link.github_token", "link", None)
    assert derive_action("PUT", "/api/link/github-token")[0] == "link.github_token"


def test_action_git_nudge():
    assert derive_action("POST", "/link/git-nudge") == ("gitwatch.nudge", "link", None)
    assert derive_action("POST", "/api/link/git-nudge")[0] == "gitwatch.nudge"


# --- DB: insert + list round-trip --------------------------------------------


async def test_insert_and_list_audit_round_trip(queries):
    await queries.insert_audit_log(
        action="job.create",
        result="ok",
        principal_id="tok-sub",
        principal_role="submitter",
        target_type="job",
        target_id="j1",
        params_redacted='{"name": "n", "api_key": "***"}',
        status_code=201,
        request_id="rid-1",
        idempotency_key="idem-1",
    )
    items, next_cursor = await queries.list_audit_log(limit=10)
    assert next_cursor is None
    assert len(items) == 1
    entry = items[0]
    assert isinstance(entry, AuditLogEntry)
    assert entry.action == "job.create"
    assert entry.result == "ok"
    assert entry.target_id == "j1"
    # params parsed back to a dict; secret already masked at the write site.
    assert entry.params_redacted == {"name": "n", "api_key": "***"}
    assert entry.idempotency_key == "idem-1"


async def test_list_audit_cursor_pagination(queries):
    for i in range(5):
        await queries.insert_audit_log(action=f"job.{i}", result="ok")
    page1, cursor1 = await queries.list_audit_log(limit=2)
    assert len(page1) == 2
    assert cursor1 is not None
    # Newest-first ordering by id DESC.
    assert page1[0].id > page1[1].id
    page2, cursor2 = await queries.list_audit_log(limit=2, cursor=cursor1)
    assert len(page2) == 2
    assert page2[0].id < page1[1].id
    page3, cursor3 = await queries.list_audit_log(limit=2, cursor=cursor2)
    assert len(page3) == 1
    assert cursor3 is None


async def test_list_audit_filters_by_result(queries):
    await queries.insert_audit_log(action="job.create", result="ok")
    await queries.insert_audit_log(action="job.create", result="denied")
    denied, _ = await queries.list_audit_log(result="denied")
    assert len(denied) == 1
    assert denied[0].result == "denied"


# --- DB: target / target_type filter (PR1) -----------------------------------


async def _seed_targets(queries) -> None:
    """Three rows across two services + one model (mirrors the deploy attribution)."""
    await queries.insert_audit_log(
        action="deploy.create", result="ok", target_type="service", target_id="app-a"
    )
    await queries.insert_audit_log(
        action="deploy.git_create", result="ok", target_type="service", target_id="app-b"
    )
    await queries.insert_audit_log(
        action="model.serve", result="ok", target_type="model", target_id="ollama-x"
    )


async def test_list_audit_filters_by_target(queries):
    await _seed_targets(queries)
    items, _ = await queries.list_audit_log(target="app-a")
    assert [i.target_id for i in items] == ["app-a"]


async def test_list_audit_filters_by_target_type(queries):
    await _seed_targets(queries)
    items, _ = await queries.list_audit_log(target_type="service")
    assert {i.target_id for i in items} == {"app-a", "app-b"}
    assert all(i.target_type == "service" for i in items)


async def test_list_audit_filters_by_target_and_type_composed(queries):
    await _seed_targets(queries)
    # A model whose id collides with a service name must NOT leak across types.
    await queries.insert_audit_log(
        action="service.create", result="ok", target_type="service", target_id="ollama-x"
    )
    items, _ = await queries.list_audit_log(target="ollama-x", target_type="model")
    assert len(items) == 1
    assert items[0].action == "model.serve"


async def test_list_audit_target_composes_with_action(queries):
    await queries.insert_audit_log(
        action="deploy.create", result="ok", target_type="service", target_id="app-a"
    )
    await queries.insert_audit_log(
        action="service.restart", result="ok", target_type="service", target_id="app-a"
    )
    items, _ = await queries.list_audit_log(target="app-a", action="deploy.create")
    assert len(items) == 1
    assert items[0].action == "deploy.create"


async def test_list_audit_target_composes_with_cursor(queries):
    for _ in range(3):
        await queries.insert_audit_log(
            action="deploy.create", result="ok", target_type="service", target_id="app-a"
        )
    page1, cursor1 = await queries.list_audit_log(target="app-a", limit=2)
    assert len(page1) == 2
    assert cursor1 is not None
    page2, cursor2 = await queries.list_audit_log(target="app-a", limit=2, cursor=cursor1)
    assert len(page2) == 1
    assert cursor2 is None
    assert all(i.target_id == "app-a" for i in page1 + page2)


async def test_list_audit_target_no_match_returns_empty(queries):
    await _seed_targets(queries)
    items, cursor = await queries.list_audit_log(target="ghost")
    assert items == []
    assert cursor is None


# --- DB: principal / action_prefix / since / until (P24a WP3) -----------------


async def test_list_audit_filters_by_principal_id(queries):
    await queries.insert_audit_log(action="deploy.create", result="ok", principal_id="tok-a")
    await queries.insert_audit_log(action="service.stop", result="ok", principal_id="tok-b")
    items, _ = await queries.list_audit_log(principal_id="tok-a")
    assert [i.action for i in items] == ["deploy.create"]


async def test_list_audit_filters_by_action_prefix(queries):
    for action in ("deploy.create", "deploy.git_create", "service.restart"):
        await queries.insert_audit_log(action=action, result="ok")
    items, _ = await queries.list_audit_log(action_prefix="deploy.")
    assert {i.action for i in items} == {"deploy.create", "deploy.git_create"}


async def test_action_prefix_percent_and_underscore_are_literal(queries):
    """The GLOB choice, pinned.

    Under ``LIKE`` a caller-supplied ``_`` is a single-character wildcard, so a
    prefix of ``deploy_`` would silently match ``deployX`` and a bare ``%``
    would match the entire trail. ``GLOB ? || '*'`` gives both characters their
    literal meaning — the filter answers the question that was asked.
    """
    await queries.insert_audit_log(action="deploy_create", result="ok")
    await queries.insert_audit_log(action="deployxcreate", result="ok")
    await queries.insert_audit_log(action="100%done", result="ok")

    items, _ = await queries.list_audit_log(action_prefix="deploy_")
    assert [i.action for i in items] == ["deploy_create"]

    items, _ = await queries.list_audit_log(action_prefix="100%")
    assert [i.action for i in items] == ["100%done"]

    # A lone '%' is a literal too: it matches nothing, rather than everything.
    items, _ = await queries.list_audit_log(action_prefix="%")
    assert items == []


async def test_list_audit_filters_by_since_and_until(queries):
    await queries.insert_audit_log(action="job.old", result="ok")
    await queries.insert_audit_log(action="job.new", result="ok")
    # Rewrite ts directly: ``datetime('now')`` gives every row the same second.
    await queries._db.conn.execute(
        "UPDATE audit_log SET ts = '2026-01-01 00:00:00' WHERE action = 'job.old'"
    )
    await queries._db.conn.execute(
        "UPDATE audit_log SET ts = '2026-06-01 00:00:00' WHERE action = 'job.new'"
    )
    await queries._db.conn.commit()

    items, _ = await queries.list_audit_log(since="2026-03-01 00:00:00")
    assert [i.action for i in items] == ["job.new"]
    items, _ = await queries.list_audit_log(until="2026-03-01 00:00:00")
    assert [i.action for i in items] == ["job.old"]
    # Both bounds are inclusive.
    items, _ = await queries.list_audit_log(
        since="2026-01-01 00:00:00", until="2026-01-01 00:00:00"
    )
    assert [i.action for i in items] == ["job.old"]


async def test_p24_filters_compose_with_each_other(queries):
    await queries.insert_audit_log(action="deploy.create", result="ok", principal_id="tok-a")
    await queries.insert_audit_log(action="deploy.plan", result="ok", principal_id="tok-b")
    await queries.insert_audit_log(action="service.stop", result="ok", principal_id="tok-a")
    items, _ = await queries.list_audit_log(principal_id="tok-a", action_prefix="deploy")
    assert [i.action for i in items] == ["deploy.create"]


async def test_list_audit_invalid_cursor_raises(queries):
    with pytest.raises(ValueError):
        await queries.list_audit_log(cursor="not-an-int")


def test_audit_log_page_model_is_paginated():
    page = AuditLogPage(items=[], next_cursor="5")
    assert page.next_cursor == "5"
    assert page.items == []
