"""Tests for the streamable-HTTP MCP transport (P13c WP10b, plan §3).

Two levels, per the brief:

* **Unit-level** (no app): ``_request_client`` fail-closed behavior, the
  ``_TransportGuard`` contextvar capture, and the guard's own Host/Origin/body
  checks driven with a crafted ASGI scope.
* **Full-stack** (Starlette ``TestClient``): a hand-built FastAPI app that
  mirrors ``create_app`` — the real auth/audit/idempotency middleware trio and
  the real ``_McpMount`` + wrapped transport — with an ``AsyncMock`` queries
  layer so no aiosqlite connection crosses event loops. The session manager is
  driven by a light lifespan (a ``Mount`` never runs the sub-app lifespan).

If ``TestClient`` hangs in the sandbox, the unit-level set stands alone; narrow
with ``-k`` and report which ran.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("mcp")

import nerdit.mcp.server as mcp_server  # noqa: E402
import nerdit.mcp.transport as mcp_transport  # noqa: E402
from nerdit.config.settings import NerditSettings  # noqa: E402
from nerdit.daemon.audit import AuditMiddleware  # noqa: E402
from nerdit.daemon.auth import hash_token  # noqa: E402
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers  # noqa: E402
from nerdit.daemon.idempotency import IdempotencyMiddleware  # noqa: E402
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware  # noqa: E402
from nerdit.daemon.server import _McpMount  # noqa: E402
from nerdit.db.models import TokenRole  # noqa: E402

# httpx's TestClient sends this Host by default; drive with a loopback base_url
# instead so the guard's Host allowlist (localhost/127.0.0.1) passes.
_BASE_URL = "http://localhost:9321"
_MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


@pytest.fixture(autouse=True)
def _restore_mcp_globals():
    """``build_http_app`` mutates module globals — save/restore around each test."""
    saved = (mcp_transport._HTTP_MODE, mcp_transport._HTTP_PORT, mcp_transport._HTTP_HOST)
    yield
    mcp_transport._HTTP_MODE, mcp_transport._HTTP_PORT, mcp_transport._HTTP_HOST = saved


# --- helpers ------------------------------------------------------------------


def _settings(
    *, http_enabled: bool = True, auth_token: str | None = "test-token"
) -> NerditSettings:
    s = NerditSettings()
    s.daemon.auth_token = auth_token
    s.daemon.port = 9321
    s.mcp.http_enabled = http_enabled
    return s


def _token_row(role: TokenRole) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"tok-{role.value}",
        name="ci",
        role=role,
        max_gpus=2,
        max_concurrent_jobs=2,
        # P25 WP1: the middleware reads both on every scoped-token request.
        expires_at=None,
        scope_services=None,
    )


def _queries(mapping: dict[str, TokenRole] | None = None) -> SimpleNamespace:
    """A queries stand-in resolving scoped tokens by hash to a role row."""
    rows = {hash_token(raw): _token_row(role) for raw, role in (mapping or {}).items()}
    return SimpleNamespace(
        get_api_token_by_hash=AsyncMock(side_effect=lambda h: rows.get(h)),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(),
        insert_idempotency_inprogress=AsyncMock(return_value=True),
        get_idempotency_record=AsyncMock(return_value=None),
    )


def _build_app(settings: NerditSettings, queries: SimpleNamespace):
    """A FastAPI app mirroring ``create_app``'s middleware trio + MCP mount."""
    from fastapi import FastAPI

    @asynccontextmanager
    async def _lifespan(app):  # noqa: ANN001, ANN202
        cm = None
        server = getattr(app.state, "mcp_server", None)
        if server is not None:
            cm = server.session_manager.run()
            await cm.__aenter__()
        yield
        if cm is not None:
            await cm.__aexit__(None, None, None)

    app = FastAPI(lifespan=_lifespan)
    register_error_handlers(app)
    app.add_middleware(
        IdempotencyMiddleware,
        get_queries=lambda: getattr(app.state, "queries", None),
        require_idempotency_key=False,
    )
    app.add_middleware(
        AuditMiddleware,
        get_queries=lambda: getattr(app.state, "queries", None),
        get_event_bus=lambda: None,
    )
    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=settings.daemon.auth_token,
        get_queries=lambda: getattr(app.state, "queries", None),
    )
    app.add_middleware(RequestIdMiddleware)
    if settings.mcp.http_enabled:
        mcp_asgi, mcp_srv = mcp_server.build_http_app(settings)
        app.router.routes.append(_McpMount("/api/mcp", app=mcp_asgi))
        app.state.mcp_server = mcp_srv
    app.state.queries = queries
    return app


def _client(app, *, follow_redirects: bool = True):
    from fastapi.testclient import TestClient

    return TestClient(
        app,
        base_url=_BASE_URL,
        raise_server_exceptions=False,
        follow_redirects=follow_redirects,
    )


def _rpc(method: str, id_: int = 1, **params) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def _sse_result(response) -> dict:  # noqa: ANN001
    """Parse the single ``data:`` frame of a streamable-HTTP SSE response."""
    body = response.text
    _, _, payload = body.partition("data: ")
    return json.loads(payload.splitlines()[0])


async def _drive_guard(guard, scope, body: bytes = b""):
    """Drive an ASGI app once with a crafted scope; return (status, body)."""
    state = {"more": False}

    async def receive():
        if state["more"]:
            return {"type": "http.disconnect"}
        state["more"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    sent: dict = {}
    chunks: list[bytes] = []

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await guard(scope, receive, send)
    return sent.get("status"), b"".join(chunks)


def _http_scope(
    *, path: str = "/", method: str = "POST", headers: dict[str, str], content_length: int | None
) -> dict:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    if content_length is not None:
        raw.append((b"content-length", str(content_length).encode()))
    return {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": b"/api/mcp" + path.encode(),
        "query_string": b"",
        "headers": raw,
        "scheme": "http",
        "server": ("localhost", 9321),
        "client": ("127.0.0.1", 1234),
        "root_path": "",
    }


# --- unit: _request_client fail-closed pin ------------------------------------


def test_request_client_pins_loopback_with_context_token():
    mcp_transport._HTTP_MODE = True
    mcp_transport._HTTP_PORT = 9999
    tok = mcp_transport._REQUEST_TOKEN.set("alice-token")
    try:
        client = mcp_transport._request_client()
    finally:
        mcp_transport._REQUEST_TOKEN.reset(tok)
    assert client._base_url == "http://127.0.0.1:9999"
    assert client._headers.get("Authorization") == "Bearer alice-token"


@pytest.mark.parametrize(
    "daemon_host, expected",
    [
        ("127.0.0.1", "127.0.0.1"),  # loopback default
        ("0.0.0.0", "127.0.0.1"),  # wildcard covers loopback
        ("::", "127.0.0.1"),  # v6 wildcard
        ("", "127.0.0.1"),  # empty == wildcard
        ("192.168.1.50", "192.168.1.50"),  # specific interface: sole listener
        ("localhost", "localhost"),  # loopback name kept
    ],
)
def test_inner_hop_host_resolution(daemon_host, expected):
    assert mcp_transport._inner_hop_host(daemon_host) == expected


def test_request_client_dials_specific_interface_host():
    # A non-loopback `[daemon].host` is the only address uvicorn listens on, so
    # the inner hop must dial it — dialing 127.0.0.1 would connection_error.
    mcp_transport._HTTP_MODE = True
    mcp_transport._HTTP_PORT = 9333
    mcp_transport._HTTP_HOST = "192.168.1.50"
    tok = mcp_transport._REQUEST_TOKEN.set("alice-token")
    try:
        client = mcp_transport._request_client()
    finally:
        mcp_transport._REQUEST_TOKEN.reset(tok)
    assert client._base_url == "http://192.168.1.50:9333"


def test_http_mode_without_token_fails_closed(monkeypatch):
    monkeypatch.setattr(mcp_transport, "get_configured_client", lambda: pytest.fail("fell back"))
    mcp_transport._HTTP_MODE = True
    assert mcp_transport._REQUEST_TOKEN.get() is None
    with pytest.raises(mcp_transport.McpHttpAuthError) as exc:
        mcp_transport._request_client()
    envelope = json.loads(str(exc.value))
    assert envelope["error"]["status"] == 401
    assert envelope["error"]["code"] == "unauthenticated"


def test_stdio_mode_falls_back_to_config_client(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mcp_transport, "get_configured_client", lambda: sentinel)
    mcp_transport._HTTP_MODE = False
    assert mcp_transport._REQUEST_TOKEN.get() is None
    assert mcp_transport._request_client() is sentinel


# --- unit: _TransportGuard ----------------------------------------------------


@pytest.mark.asyncio
async def test_transport_guard_sets_and_resets_contextvar():
    seen: dict = {}

    async def _inner(scope, receive, send):
        seen["token"] = mcp_transport._REQUEST_TOKEN.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(
        headers={"host": "localhost:9321", "authorization": "Bearer bob-token"},
        content_length=0,
    )
    status, _ = await _drive_guard(guard, scope)
    assert status == 200
    assert seen["token"] == "bob-token"
    assert mcp_transport._REQUEST_TOKEN.get() is None


@pytest.mark.asyncio
async def test_post_without_content_length_411():
    async def _inner(scope, receive, send):  # pragma: no cover - must not run
        pytest.fail("inner reached despite missing Content-Length")

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(headers={"host": "localhost:9321"}, content_length=None)
    status, body = await _drive_guard(guard, scope)
    assert status == 411
    assert json.loads(body)["code"] == "length_required"


@pytest.mark.asyncio
async def test_body_over_1mb_rejected_413():
    async def _inner(scope, receive, send):  # pragma: no cover - must not run
        pytest.fail("inner reached despite oversized body")

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(
        headers={"host": "localhost:9321"}, content_length=mcp_transport.MAX_MCP_BODY_BYTES + 1
    )
    status, body = await _drive_guard(guard, scope)
    assert status == 413
    assert json.loads(body)["code"] == "payload_too_large"


@pytest.mark.asyncio
async def test_transport_guard_honors_configured_body_cap():
    """(P29) ``[mcp].max_body_bytes`` is bound into the guard, not read off the
    module constant — falsifies the "the knob is inert" failure mode.

    Both ways over the CONFIGURED cap: one byte over is a 413, exactly at it
    reaches the inner app. The default (:func:`test_body_over_1mb_rejected_413`)
    is unchanged.
    """
    reached = {"v": False}

    async def _inner(scope, receive, send):
        reached["v"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = mcp_transport._TransportGuard(
        _inner, allowed_hostnames=frozenset({"localhost"}), max_body_bytes=70_000
    )

    status, body = await _drive_guard(
        guard, _http_scope(headers={"host": "localhost:9321"}, content_length=70_001)
    )
    assert status == 413
    assert json.loads(body)["code"] == "payload_too_large"
    assert reached["v"] is False

    status, _ = await _drive_guard(
        guard, _http_scope(headers={"host": "localhost:9321"}, content_length=70_000)
    )
    assert status == 200
    assert reached["v"] is True

    # And a body the module default would still allow is refused by the tighter
    # configured cap — proof the constant is not the operative value.
    status, _ = await _drive_guard(
        guard,
        _http_scope(
            headers={"host": "localhost:9321"}, content_length=mcp_transport.MAX_MCP_BODY_BYTES
        ),
    )
    assert status == 413


@pytest.mark.asyncio
async def test_build_http_app_binds_the_configured_body_cap(monkeypatch):
    """The one construction site actually passes the setting through (P29 R9)."""
    captured = {}

    class _Recorder(mcp_transport._TransportGuard):
        def __init__(self, inner, **kw):  # noqa: ANN001, ANN003
            captured.update(kw)
            super().__init__(inner, **kw)

    monkeypatch.setattr(mcp_transport, "_TransportGuard", _Recorder)
    settings = NerditSettings()
    settings.mcp.max_body_bytes = 262_144

    mcp_server.build_http_app(settings)
    assert captured["max_body_bytes"] == 262_144


@pytest.mark.asyncio
async def test_foreign_host_rejected_421():
    guard = mcp_transport._TransportGuard(
        lambda *a: pytest.fail("inner reached"), allowed_hostnames=frozenset({"localhost"})
    )
    scope = _http_scope(headers={"host": "rebind.attacker.example"}, content_length=0)
    status, body = await _drive_guard(guard, scope)
    assert status == 421
    assert json.loads(body)["code"] == "invalid_host"


@pytest.mark.asyncio
async def test_ip_literal_host_allowed():
    reached = {"v": False}

    async def _inner(scope, receive, send):
        reached["v"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(headers={"host": "192.168.1.50:9321"}, content_length=0)
    status, _ = await _drive_guard(guard, scope)
    assert status == 200
    assert reached["v"] is True


@pytest.mark.asyncio
async def test_foreign_origin_rejected():
    guard = mcp_transport._TransportGuard(
        lambda *a: pytest.fail("inner reached"), allowed_hostnames=frozenset({"localhost"})
    )
    scope = _http_scope(
        headers={"host": "localhost:9321", "origin": "https://evil.example"}, content_length=0
    )
    status, body = await _drive_guard(guard, scope)
    assert status == 403
    assert json.loads(body)["code"] == "origin_forbidden"


@pytest.mark.asyncio
async def test_absent_origin_allowed():
    reached = {"v": False}

    async def _inner(scope, receive, send):
        reached["v"] = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(headers={"host": "localhost:9321"}, content_length=0)
    status, _ = await _drive_guard(guard, scope)
    assert status == 200
    assert reached["v"] is True


# --- full-stack: mount gating + startup hard-errors ---------------------------


def test_mount_absent_when_disabled():
    app = _build_app(_settings(http_enabled=False), _queries())
    with _client(app) as c:
        # Authenticated (legacy admin token) so the 404 is routing, not auth.
        r = c.post(
            "/api/mcp",
            json=_rpc("tools/list"),
            headers={**_MCP_HEADERS, "Authorization": "Bearer test-token"},
        )
    assert r.status_code == 404


def test_startup_hard_error_when_token_none(monkeypatch):
    import nerdit.daemon.server as srv

    monkeypatch.setattr(srv, "load_settings", lambda: _settings(auth_token=None))
    with pytest.raises(RuntimeError, match=r"\[mcp\]\.http_enabled"):
        srv.create_app()


def test_startup_hard_error_when_extra_missing(monkeypatch):
    import importlib.util

    import nerdit.daemon.server as srv

    monkeypatch.setattr(srv, "load_settings", lambda: _settings())
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "mcp" else real(name, *a, **k),
    )
    with pytest.raises(RuntimeError, match="mcp"):
        srv.create_app()


# --- full-stack: auth + escalation regressions --------------------------------


def test_unauthenticated_mcp_post_401():
    app = _build_app(_settings(), _queries())
    with _client(app) as c:
        r_none = c.post("/api/mcp", json=_rpc("tools/list"), headers=_MCP_HEADERS)
        r_bad = c.post(
            "/api/mcp",
            json=_rpc("tools/list"),
            headers={**_MCP_HEADERS, "Authorization": "Bearer nope"},
        )
    assert r_none.status_code == 401
    assert r_bad.status_code == 403


def test_readonly_tools_list_succeeds():
    app = _build_app(_settings(), _queries({"ro-token": TokenRole.readonly}))
    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc("tools/list"),
            headers={**_MCP_HEADERS, "Authorization": "Bearer ro-token"},
        )
    assert r.status_code == 200
    tools = _sse_result(r)["result"]["tools"]
    # (P26 WP-H) 43 → 45 with share_service/unshare_service; (P26 WP1) 45 → 47
    # with add_domain/remove_domain; (P37) 47 → 49 with dump_database/
    # list_database_dumps; the authoritative ledger pin lives in
    # tests/test_mcp.py.
    assert len(tools) == 49


def test_readonly_write_tool_gets_inner_403(monkeypatch):
    app = _build_app(_settings(), _queries({"ro-token": TokenRole.readonly}))
    real_client_cls = mcp_transport.NerditClient

    class _InProcClient(real_client_cls):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.ASGITransport(app=app))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mcp_transport, "NerditClient", _InProcClient)
    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc("tools/call", name="stop_service", arguments={"name": "demo"}),
            headers={**_MCP_HEADERS, "Authorization": "Bearer ro-token"},
        )
    assert r.status_code == 200
    # The write refusal on the inner loopback hop surfaces as the tool result.
    assert "forbidden" in r.text


# --- full-stack: sibling gate, path spellings ---------------------------------


def test_sibling_path_not_exempt():
    app = _build_app(_settings(), _queries({"ro-token": TokenRole.readonly}))
    with _client(app) as c:
        hdr = {**_MCP_HEADERS, "Authorization": "Bearer ro-token"}
        r_sibling = c.post("/api/mcpX", json=_rpc("tools/list"), headers=hdr)
        r_other = c.post("/api/jobs", json={}, headers=hdr)
    assert r_sibling.status_code == 403
    assert r_other.status_code == 403


def test_bare_and_slash_paths_both_served():
    app = _build_app(_settings(), _queries({"ro-token": TokenRole.readonly}))
    hdr = {**_MCP_HEADERS, "Authorization": "Bearer ro-token"}
    with _client(app, follow_redirects=False) as c:
        r_bare = c.post("/api/mcp", json=_rpc("tools/list"), headers=hdr)
        r_slash = c.post("/api/mcp/", json=_rpc("tools/list"), headers=hdr)
    # _McpMount serves both directly — no 307 to the slash form, no 404.
    for r in (r_bare, r_slash):
        assert r.status_code == 200
        assert len(_sse_result(r)["result"]["tools"]) == 49


# --- full-stack: audit + idempotency exemptions -------------------------------


def test_audit_excluded_but_denials_recorded():
    # A successful framing POST leaves no audit row for the mount path.
    ok_q = _queries({"ro-token": TokenRole.readonly})
    app_ok = _build_app(_settings(), ok_q)
    with _client(app_ok) as c:
        c.post(
            "/api/mcp",
            json=_rpc("tools/list"),
            headers={**_MCP_HEADERS, "Authorization": "Bearer ro-token"},
        )
    ok_q.insert_audit_log.assert_not_called()

    # An unauthenticated POST still leaves a denied row (action ``POST /mcp``).
    deny_q = _queries()
    app_deny = _build_app(_settings(), deny_q)
    with _client(app_deny) as c:
        c.post("/api/mcp", json=_rpc("tools/list"), headers=_MCP_HEADERS)
    deny_q.insert_audit_log.assert_called_once()
    kwargs = deny_q.insert_audit_log.call_args.kwargs
    assert kwargs["result"] == "denied"
    assert kwargs["action"] == "POST /mcp"


def test_idempotency_key_ignored_on_mcp_path():
    q = _queries({"ro-token": TokenRole.readonly})
    app = _build_app(_settings(), q)
    with _client(app) as c:
        c.post(
            "/api/mcp",
            json=_rpc("tools/list"),
            headers={
                **_MCP_HEADERS,
                "Authorization": "Bearer ro-token",
                "Idempotency-Key": "K",
            },
        )
    q.insert_idempotency_inprogress.assert_not_called()
