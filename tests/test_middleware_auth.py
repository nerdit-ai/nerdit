"""Tests for ScopedTokenAuthMiddleware (P1 / S3).

Uses ``AsyncMock`` queries under the Starlette TestClient (the working pattern
from test_api.py) so there is no real aiosqlite connection crossing event
loops. The denial-audit and last_used_at behaviours are asserted via the mock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from nerdit.daemon.auth import current_principal, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import (
    BearerAuthMiddleware,
    ScopedTokenAuthMiddleware,
)
from nerdit.db.models import ApiToken, TokenRole

RAW = "rawtoken-value"


def _queries(token_obj: ApiToken | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(return_value=token_obj)
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    return q


def _scoped_token(role: TokenRole = TokenRole.submitter) -> ApiToken:
    return ApiToken(
        id="tok-1",
        name="ci-bot",
        role=role,
        token_hash=hash_token(RAW),
        max_gpus=4,
        max_concurrent_jobs=2,
    )


def _make_app(*, token: str | None = "secret", queries: AsyncMock | None = None) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/whoami")
    def whoami(request: Request) -> dict:
        p = current_principal(request)
        return {
            "name": p.name,
            "role": p.role.value,
            "token_id": p.token_id,
            "is_legacy_admin": p.is_legacy_admin,
            "max_gpus": p.max_gpus,
        }

    @app.post("/mutate")
    def mutate(request: Request) -> dict:
        p = current_principal(request)
        return {"name": p.name, "role": p.role.value, "token_id": p.token_id}

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=token,
        get_queries=(lambda: queries) if queries is not None else None,
    )
    app.add_middleware(RequestIdMiddleware)
    return app


# --- bypass principals --------------------------------------------------------


def test_bearer_alias_points_at_scoped_middleware():
    assert BearerAuthMiddleware is ScopedTokenAuthMiddleware


def test_no_token_configured_yields_local_principal():
    client = TestClient(_make_app(token=None), raise_server_exceptions=False)
    assert client.get("/whoami").json()["name"] == "local"
    # Local is admin → mutation passes too.
    assert client.post("/mutate").json()["role"] == "admin"


def test_legacy_global_token_yields_legacy_admin():
    client = TestClient(_make_app(queries=_queries()), raise_server_exceptions=False)
    body = client.get("/whoami", headers={"Authorization": "Bearer secret"}).json()
    assert body["name"] == "legacy-admin"
    assert body["role"] == "admin"
    assert body["is_legacy_admin"] is True


def test_public_path_skips_auth():
    client = TestClient(_make_app(queries=_queries()), raise_server_exceptions=False)
    assert client.get("/health").status_code == 200


# --- scoped token resolution + state propagation ------------------------------


def test_valid_scoped_token_attaches_role_principal_visible_to_route():
    q = _queries(_scoped_token(TokenRole.submitter))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    body = client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"}).json()
    # Principal set by the (outer) middleware is visible to the route.
    assert body["token_id"] == "tok-1"
    assert body["role"] == "submitter"
    assert body["max_gpus"] == 4
    # Looked up by hash, never plaintext.
    q.get_api_token_by_hash.assert_awaited_once_with(hash_token(RAW))


def test_unknown_or_revoked_token_returns_403_invalid_token():
    q = _queries(None)  # get_api_token_by_hash filters revoked → None
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": "Bearer bogus"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "invalid_token"


# --- header validation --------------------------------------------------------


def test_missing_header_returns_401():
    client = TestClient(_make_app(queries=_queries()), raise_server_exceptions=False)
    resp = client.get("/whoami")
    assert resp.status_code == 401
    assert resp.json()["code"] == "unauthenticated"


def test_bad_auth_format_returns_401():
    client = TestClient(_make_app(queries=_queries()), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": "Token x"})
    assert resp.status_code == 401
    assert resp.json()["code"] == "invalid_auth_format"


# --- coarse readonly gate -----------------------------------------------------


def test_readonly_token_blocked_on_mutation():
    q = _queries(_scoped_token(TokenRole.readonly))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "forbidden"
    # English message + empty-object ``detail`` (pre-route denial shape).
    assert body["message"] == "This token is read-only and cannot perform this operation."
    assert body["detail"] == {}


def test_readonly_token_allowed_on_safe_read():
    q = _queries(_scoped_token(TokenRole.readonly))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 200
    assert resp.json()["role"] == "readonly"


# --- denial audit (Auth is outermost; logs its own denials) -------------------


def _denied_calls(q: AsyncMock) -> list[dict]:
    return [
        c.kwargs for c in q.insert_audit_log.await_args_list if c.kwargs.get("result") == "denied"
    ]


def test_invalid_token_writes_denied_audit_row():
    q = _queries(None)
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": "Bearer bogus"})
    denials = _denied_calls(q)
    assert len(denials) == 1
    assert denials[0]["status_code"] == 403
    # Action templated to bounded cardinality.
    assert denials[0]["action"] == "POST /mutate"


def test_missing_header_writes_denied_audit_row():
    q = _queries()
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate")
    denials = _denied_calls(q)
    assert len(denials) == 1
    assert denials[0]["status_code"] == 401
    assert denials[0]["principal_id"] is None


def test_readonly_gate_writes_denied_audit_row_with_principal():
    q = _queries(_scoped_token(TokenRole.readonly))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    denials = _denied_calls(q)
    assert len(denials) == 1
    assert denials[0]["status_code"] == 403
    assert denials[0]["principal_id"] == "tok-1"
    assert denials[0]["principal_role"] == "readonly"


def test_templated_action_strips_concrete_service_id():
    # Denials share the S6 route→action map (derive_action), so a known route
    # records its mapped action — never the concrete row id (bounded cardinality).
    q = _queries(None)
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/services/abc123def456/stop", headers={"Authorization": "Bearer bogus"})
    denials = _denied_calls(q)
    assert denials[0]["action"] == "service.stop"
    assert "abc123def456" not in denials[0]["action"]


# --- throttled last_used_at ---------------------------------------------------


def test_mutation_touches_last_used_at():
    q = _queries(_scoped_token(TokenRole.submitter))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    q.touch_api_token.assert_awaited_once_with("tok-1")


def test_safe_get_does_not_touch_last_used_at():
    q = _queries(_scoped_token(TokenRole.submitter))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"})
    q.touch_api_token.assert_not_awaited()


def test_repeated_mutations_throttle_touch():
    q = _queries(_scoped_token(TokenRole.submitter))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    for _ in range(3):
        client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    # Throttled to one write within the 60s window.
    q.touch_api_token.assert_awaited_once()


def test_first_touch_fires_when_monotonic_below_interval(monkeypatch):
    """Regression: on a freshly-booted host monotonic() can be < the throttle
    interval; the first touch must still fire (it was wrongly skipped when the
    'last' default was 0.0)."""
    import nerdit.daemon.middleware as mw

    # Simulate a host that booted ~5s ago: monotonic() well under the 60s window.
    monkeypatch.setattr(mw.time, "monotonic", lambda: 5.0)
    q = _queries(_scoped_token(TokenRole.submitter))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    q.touch_api_token.assert_awaited_once_with("tok-1")


# --- I1: fail-closed default when no principal was attached -------------------


def test_current_principal_defaults_to_anonymous_readonly():
    """A request that never went through the auth middleware is non-privileged."""
    from starlette.datastructures import State

    from nerdit.daemon.auth import ANONYMOUS, current_principal, require_role
    from nerdit.daemon.errors import NerditError
    from nerdit.db.models import TokenRole

    class _Bare:
        # No ``principal`` set on state — simulates a missing/reordered middleware.
        state = State()

    req = _Bare()
    assert current_principal(req) is ANONYMOUS
    assert current_principal(req).role == TokenRole.readonly
    # readonly is denied where submitter/admin is required (fail-closed).
    try:
        require_role(req, TokenRole.submitter, TokenRole.admin)
        raise AssertionError("expected NerditError")
    except NerditError as exc:
        assert exc.status_code == 403


# --- P25 D-P25-1/2: token expiry at the resolution choke point ----------------


def _expiring_token(delta_s: float, role: TokenRole = TokenRole.submitter) -> ApiToken:
    """A scoped token whose ``expires_at`` is ``delta_s`` from now (UTC-aware)."""
    from datetime import UTC, datetime, timedelta

    return ApiToken(
        id="tok-1",
        name="ci-bot",
        role=role,
        token_hash=hash_token(RAW),
        expires_at=datetime.now(UTC) + timedelta(seconds=delta_s),
    )


def test_null_expiry_authenticates_forever():
    """The anti-bricking default: NULL ``expires_at`` never expires."""
    q = _queries(_scoped_token())
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 200


def test_future_expiry_authenticates():
    q = _queries(_expiring_token(60))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 200


def test_past_expiry_is_denied_with_token_expired():
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "token_expired"
    # Pre-route denial envelope shape: ``detail`` is an empty OBJECT.
    assert body["detail"] == {}
    assert "message" in body


def test_expiry_applies_to_safe_reads_too():
    """Expiry is not method-scoped: a GET is denied exactly like a POST."""
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.get("/whoami", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"


def test_expired_readonly_token_gets_token_expired_not_forbidden():
    """Ordering is load-bearing: the expiry check precedes the readonly gate."""
    q = _queries(_expiring_token(-1, TokenRole.readonly))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"


def test_expired_token_does_not_stamp_last_used_at():
    """Ordering: the expiry check precedes ``_touch``."""
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    q.touch_api_token.assert_not_awaited()


def test_expired_denial_row_carries_real_attribution():
    """(D-P25-2) The row stays resolvable precisely so the audit keeps the identity."""
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    denials = _denied_calls(q)
    assert len(denials) == 1
    assert denials[0]["status_code"] == 403
    assert denials[0]["principal_id"] == "tok-1"
    assert denials[0]["principal_role"] == "submitter"


def test_expired_denial_message_names_the_admin_remint_path():
    """The remediation must never point at an endpoint the caller can no longer reach."""
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    message = resp.json()["message"]
    assert "cannot rotate itself" in message
    assert "nerdit token create" in message
    # The public expiry instant is named; the token id and hash never are.
    assert "tok-1" not in message
    assert hash_token(RAW) not in message


def test_expired_token_is_denied_on_the_mcp_mount_too():
    """(P13c) The readonly gate's boundary-exact MCP exemption does NOT cover expiry.

    An MCP POST is transport framing rather than a mutation, which is why the
    coarse readonly gate skips it — but an expired credential must not be able
    to open a session at all.
    """
    q = _queries(_expiring_token(-1, TokenRole.readonly))
    app = _make_app(queries=q)

    @app.post("/api/mcp")
    def _mcp(request: Request) -> dict:  # pragma: no cover - must never run
        return {"reached": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/mcp", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"


def test_readonly_mcp_framing_still_passes_when_not_expired():
    """Positive control for the test above: the P13c exemption is intact."""
    q = _queries(_scoped_token(TokenRole.readonly))
    app = _make_app(queries=q)

    @app.post("/api/mcp")
    def _mcp(request: Request) -> dict:
        return {"reached": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/mcp", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 200
    assert resp.json() == {"reached": True}


def test_naive_stored_expiry_does_not_500_the_middleware():
    """A hand-edited tz-naive ``expires_at`` is coerced to UTC, never compared raw."""
    from datetime import datetime

    token = _scoped_token()
    # Unambiguously past in every timezone, so the assertion pins the coercion
    # (no TypeError) rather than a local-vs-UTC offset.
    naive = datetime(2020, 1, 1, 0, 0)  # noqa: DTZ001 — the hand-edited shape under test
    token = token.model_copy(update={"expires_at": naive})
    q = _queries(token)
    client = TestClient(_make_app(queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": f"Bearer {RAW}"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"


# --- P25 D-P25-3: scope + expiry never touch the bypass sentinels -------------


def test_legacy_admin_sentinel_carries_no_scope_or_expiry():
    from nerdit.daemon.auth import LEGACY_ADMIN, LOCAL

    for sentinel in (LEGACY_ADMIN, LOCAL):
        assert sentinel.scope_services is None
        assert sentinel.expires_at is None
        assert sentinel.in_scope("anything") is True


def test_legacy_global_token_passes_even_when_a_row_would_be_expired():
    """The legacy branch short-circuits before the scoped-token lookup entirely."""
    q = _queries(_expiring_token(-1))
    client = TestClient(_make_app(token="secret", queries=q), raise_server_exceptions=False)
    resp = client.post("/mutate", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "legacy-admin"


def test_local_bypass_passes_with_no_queries_at_all():
    client = TestClient(_make_app(token=None), raise_server_exceptions=False)
    resp = client.post("/mutate")
    assert resp.status_code == 200
    assert resp.json()["name"] == "local"


def test_principal_in_scope_semantics():
    """``None`` grants everything; an empty frozenset grants nothing."""
    from nerdit.daemon.auth import Principal

    unscoped = Principal(token_id="t", name="n", role=TokenRole.submitter)
    assert unscoped.in_scope("api") is True

    scoped = Principal(
        token_id="t", name="n", role=TokenRole.submitter, scope_services=frozenset({"api"})
    )
    assert scoped.in_scope("api") is True
    assert scoped.in_scope("worker") is False

    empty = Principal(token_id="t", name="n", role=TokenRole.submitter, scope_services=frozenset())
    assert empty.in_scope("api") is False


def test_scope_column_reaches_the_principal_as_a_frozenset():
    q = _queries(
        ApiToken(
            id="tok-1",
            name="ci-bot",
            role=TokenRole.submitter,
            token_hash=hash_token(RAW),
            scope_services=["api", "worker"],
        )
    )
    seen: dict = {}
    app = _make_app(queries=q)

    @app.get("/scope")
    def _scope(request: Request) -> dict:
        seen["principal"] = current_principal(request)
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/scope", headers={"Authorization": f"Bearer {RAW}"}).status_code == 200
    principal = seen["principal"]
    assert principal.scope_services == frozenset({"api", "worker"})
    assert principal.in_scope("api") and not principal.in_scope("db")
