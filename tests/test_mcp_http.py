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
        ("[::]", "127.0.0.1"),  # bracketed v6 wildcard: same value
        ("", "127.0.0.1"),  # empty == wildcard
        ("192.168.1.50", "192.168.1.50"),  # specific interface: sole listener
        ("localhost", "localhost"),  # loopback name kept
        ("::1", "::1"),  # v6 loopback literal is a specific bind
        ("[::1]", "::1"),  # ...and brackets are not part of the host
    ],
)
def test_inner_hop_host_resolution(daemon_host, expected):
    assert mcp_transport._inner_hop_host(daemon_host) == expected


def test_request_client_brackets_an_ipv6_inner_hop():
    # httpx.InvalidURL is not an HTTPError, so an unbracketed `::1` would escape
    # the MCP error mapper and surface raw on every tool call.
    mcp_transport._HTTP_MODE = True
    mcp_transport._HTTP_PORT = 9321
    # Unbracketed on purpose: `_inner_hop_host` strips brackets, so this is the
    # value it really hands the client, and it is the one the old
    # `f"http://{host}:{port}"` turned into the invalid `http://::1:9321`.
    mcp_transport._HTTP_HOST = mcp_transport._inner_hop_host("::1")
    tok = mcp_transport._REQUEST_TOKEN.set("alice-token")
    try:
        client = mcp_transport._request_client()
    finally:
        mcp_transport._REQUEST_TOKEN.reset(tok)
        mcp_transport._HTTP_HOST = "127.0.0.1"
    assert client._base_url == "http://[::1]:9321"
    assert httpx.URL(client._base_url).port == 9321


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


async def _drive_guard_headers(guard, scope) -> tuple[int, bytes, dict[str, str]]:
    """Drive the guard once and also return the response headers."""
    sent: dict = {}
    chunks: list[bytes] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
            sent["headers"] = {
                k.decode("latin-1").lower(): v.decode("latin-1") for k, v in message["headers"]
            }
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await guard(scope, receive, send)
    return sent.get("status"), b"".join(chunks), sent.get("headers", {})


@pytest.mark.asyncio
async def test_transport_rejection_carries_a_request_id():
    """A transport refusal is the one daemon error a route never sees — it must
    still be correlatable: echo the caller's/middleware's id, else mint one, and
    mirror it into ``X-Request-Id`` the way ``RequestIdMiddleware`` does."""
    guard = mcp_transport._TransportGuard(
        lambda *a: pytest.fail("inner reached"), allowed_hostnames=frozenset({"localhost"})
    )

    # Echoed from the request header.
    scope = _http_scope(
        headers={"host": "localhost:9321", "x-request-id": "cafe1234"}, content_length=None
    )
    status, body, resp_headers = await _drive_guard_headers(guard, scope)
    assert status == 411
    assert json.loads(body)["request_id"] == "cafe1234"
    assert resp_headers["x-request-id"] == "cafe1234"

    # Taken from the scope state RequestIdMiddleware populates, when it ran.
    scope = _http_scope(headers={"host": "rebind.attacker.example"}, content_length=0)
    scope["state"] = {"request_id": "from-middleware"}
    status, body, resp_headers = await _drive_guard_headers(guard, scope)
    assert status == 421
    assert json.loads(body)["request_id"] == "from-middleware"
    assert resp_headers["x-request-id"] == "from-middleware"

    # Neither present: minted, never null, and consistent with the header.
    scope = _http_scope(
        headers={"host": "localhost:9321", "origin": "https://evil.example"}, content_length=0
    )
    status, body, resp_headers = await _drive_guard_headers(guard, scope)
    assert status == 403
    minted = json.loads(body)["request_id"]
    assert minted and minted == resp_headers["x-request-id"]


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


# --- full-stack: the deploy tool body never touches the daemon host -----------


def _in_proc_client(monkeypatch, app):
    """Pin the inner loopback hop onto the test app — never a real daemon."""
    real_client_cls = mcp_transport.NerditClient

    class _InProcClient(real_client_cls):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.ASGITransport(app=app))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mcp_transport, "NerditClient", _InProcClient)


def _no_zip(monkeypatch):
    """Make any host walk/zip a loud failure; return the (empty) call log."""
    import nerdit.cli.upload as upload

    calls: list = []

    def _boom(directory, *args, **kwargs):  # noqa: ANN001, ANN202
        calls.append(directory)
        raise AssertionError(f"create_dir_zip ran on the daemon host: {directory}")

    monkeypatch.setattr(upload, "create_dir_zip", _boom)
    return calls


def _tool_error(response) -> dict:  # noqa: ANN001
    """The ``{"error": {...}}`` dict a tool returned, out of its SSE frame."""
    content = _sse_result(response)["result"]["content"]
    return json.loads(content[0]["text"])["error"]


def test_http_deploy_with_path_refused_before_any_host_read(monkeypatch, tmp_path):
    """H3: the HTTP transport refuses a caller-supplied daemon-host path."""
    app = _build_app(_settings(), _queries({"sub-token": TokenRole.submitter}))
    _in_proc_client(monkeypatch, app)
    calls = _no_zip(monkeypatch)
    victim = tmp_path / "secrets"
    victim.mkdir()
    (victim / "id_ed25519").write_text("SENTINEL-PRIVATE-KEY")

    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc(
                "tools/call",
                name="deploy",
                arguments={"path": str(victim), "name": "pwn"},
            ),
            headers={**_MCP_HEADERS, "Authorization": "Bearer sub-token"},
        )

    assert r.status_code == 200
    error = _tool_error(r)
    assert error["code"] == "mcp.local_path_unavailable"
    for alternative in ("deploy_app", "deploy_git", "deploy_template"):
        assert alternative in error["hint"]
    assert calls == []
    # The refusal echoes neither the path it was handed nor anything behind it.
    assert str(victim) not in r.text
    assert "SENTINEL-PRIVATE-KEY" not in r.text


def test_http_deploy_rollback_still_works_without_a_path(monkeypatch):
    """The path refusal is scoped to the path branch: rollback is untouched."""
    app = _build_app(_settings(), _queries({"sub-token": TokenRole.submitter}))
    _in_proc_client(monkeypatch, app)
    _no_zip(monkeypatch)

    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc(
                "tools/call",
                name="deploy",
                arguments={"name": "demo", "rollback": True},
            ),
            headers={**_MCP_HEADERS, "Authorization": "Bearer sub-token"},
        )

    assert r.status_code == 200
    # The harness app mounts no /api/deploy route, so the hop 404s — which is
    # itself the proof that the body ran past the refusal and reached it.
    assert _tool_error(r)["code"] != "mcp.local_path_unavailable"


def test_readonly_deploy_refused_before_the_tool_body_runs(monkeypatch, tmp_path):
    """M2: a readonly caller never reaches the host walk/zip, only the 403."""
    app = _build_app(_settings(), _queries({"ro-token": TokenRole.readonly}))
    _in_proc_client(monkeypatch, app)
    calls = _no_zip(monkeypatch)
    source = tmp_path / "app"
    source.mkdir()
    (source / "package.json").write_text("{}")

    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc(
                "tools/call",
                name="deploy",
                arguments={"path": str(source), "name": "demo"},
            ),
            headers={**_MCP_HEADERS, "Authorization": "Bearer ro-token"},
        )

    assert r.status_code == 200
    error = _tool_error(r)
    assert error["code"] == "forbidden"
    assert error["status"] == 403
    assert calls == []


def _denial_rows(queries) -> list[dict]:
    """Every ``result='denied'`` audit row the harness recorded."""
    return [
        call.kwargs
        for call in queries.insert_audit_log.call_args_list
        if call.kwargs.get("result") == "denied"
    ]


def test_readonly_refusal_still_records_the_denial(monkeypatch, tmp_path):
    """The pre-body short-circuit skips the inner loopback hop, and that hop is
    what wrote the ``deploy.create result=denied`` row. Recording it here keeps
    the standing "auth denials are themselves audit-logged" invariant true for
    the one path the fix touches — an admin watching ``/audit`` must still see a
    readonly token hammering a write tool."""
    queries = _queries({"ro-token": TokenRole.readonly})
    app = _build_app(_settings(), queries)
    _in_proc_client(monkeypatch, app)
    _no_zip(monkeypatch)
    source = tmp_path / "app"
    source.mkdir()

    with _client(app) as c:
        c.post(
            "/api/mcp",
            json=_rpc(
                "tools/call",
                name="deploy",
                arguments={"path": str(source), "name": "demo"},
            ),
            headers={**_MCP_HEADERS, "Authorization": "Bearer ro-token"},
        )

    rows = _denial_rows(queries)
    assert len(rows) == 1
    assert rows[0]["action"] == "deploy.create"
    assert rows[0]["status_code"] == 403
    assert rows[0]["principal_role"] == "readonly"
    assert rows[0]["principal_id"] == "tok-readonly"


def test_host_path_refusal_records_the_denial(monkeypatch, tmp_path):
    """A submitter (the tunnel principal's role) probing daemon-host paths is
    exactly the attempt an admin needs on the record; pre-fix it was silent."""
    queries = _queries({"sub-token": TokenRole.submitter})
    app = _build_app(_settings(), queries)
    _in_proc_client(monkeypatch, app)
    _no_zip(monkeypatch)
    victim = tmp_path / "secrets"
    victim.mkdir()

    with _client(app) as c:
        r = c.post(
            "/api/mcp",
            json=_rpc(
                "tools/call",
                name="deploy",
                arguments={"path": str(victim), "name": "pwn"},
            ),
            headers={**_MCP_HEADERS, "Authorization": "Bearer sub-token"},
        )

    rows = _denial_rows(queries)
    assert len(rows) == 1
    assert rows[0]["action"] == "deploy.create"
    assert rows[0]["principal_role"] == "submitter"
    # The row names no path, the same discipline as the envelope.
    assert str(victim) not in json.dumps(rows[0])
    assert str(victim) not in r.text


@pytest.mark.asyncio
async def test_transport_guard_captures_and_resets_the_role():
    seen: dict = {}

    async def _inner(scope, receive, send):
        seen["role"] = mcp_transport._REQUEST_ROLE.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    guard = mcp_transport._TransportGuard(_inner, allowed_hostnames=frozenset({"localhost"}))
    scope = _http_scope(
        headers={"host": "localhost:9321", "authorization": "Bearer ro-token"},
        content_length=0,
    )
    scope["state"] = {"principal": SimpleNamespace(role=TokenRole.readonly)}
    status, _ = await _drive_guard(guard, scope)
    assert status == 200
    assert seen["role"] == "readonly"
    assert mcp_transport._REQUEST_ROLE.get() is None


async def test_stdio_mode_keeps_path_deploy_and_needs_no_role():
    """Stdio (``nerdit mcp``) is unchanged: both refusals are HTTP-only."""
    mcp_transport._HTTP_MODE = False
    assert await mcp_transport._local_path_refusal("/api/deploy") is None
    ctx = mcp_transport._REQUEST_ROLE.set("readonly")
    try:
        assert await mcp_transport._readonly_write_refusal("/api/deploy") is None
    finally:
        mcp_transport._REQUEST_ROLE.reset(ctx)
