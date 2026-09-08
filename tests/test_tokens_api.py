"""Token CRUD route tests (P1 / S9).

Covers ``POST``/``GET``/``DELETE /api/tokens`` end-to-end through the full custom
middleware trio (Auth -> Audit -> Idempotency): admin-only access, the plaintext
shown exactly once and never stored (DB hash only, idempotency body suppressed,
audit redacted), and revocation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.config.settings import NerditSettings, SecuritySettings
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.tokens import router as tokens_router
from nerdit.db.models import ApiToken, TokenRole

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


def _queries() -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.get_api_token_by_id = AsyncMock(
        side_effect=lambda tid: next((t for t in _TOKENS.values() if t.id == tid), None)
    )
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # create_api_token echoes back the persisted token.
    q.create_api_token = AsyncMock(side_effect=lambda token: token)
    q.list_api_tokens = AsyncMock(return_value=list(_TOKENS.values()))
    q.revoke_api_token = AsyncMock(return_value=True)
    # Idempotency store: claim succeeds, completion captured.
    q.insert_idempotency_inprogress = AsyncMock(return_value=True)
    q.complete_idempotency_record = AsyncMock()
    q.get_idempotency_record = AsyncMock(return_value=None)
    q.delete_idempotency_record = AsyncMock()
    return q


def _make_app(queries: AsyncMock, *, token_default_ttl_s: int | None = None) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(tokens_router, prefix="/api")
    app.state.queries = queries
    # (P25) The create route reads ``[security].token_default_ttl_s`` off the
    # boot settings snapshot; the daemon always attaches one.
    app.state.settings = NerditSettings(
        security=SecuritySettings(token_default_ttl_s=token_default_ttl_s)
    )
    # inner -> outer: Idempotency (innermost), Audit, Auth, RequestId.
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, *, token_default_ttl_s: int | None = None) -> TestClient:
    return TestClient(
        _make_app(queries, token_default_ttl_s=token_default_ttl_s),
        raise_server_exceptions=False,
    )


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


# --- create -------------------------------------------------------------------


def test_admin_creates_token_plaintext_shown_once():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "role": "submitter"}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 201
    body = resp.json()
    raw = body["token"]
    assert raw.startswith("nrd_")
    assert body["name"] == "ci"
    assert body["role"] == "submitter"
    # Only the hash is persisted — the plaintext never reaches the DB.
    q.create_api_token.assert_awaited()
    stored = q.create_api_token.await_args.args[0]
    assert stored.token_hash == hash_token(raw)
    assert stored.token_hash != raw


def test_create_legacy_admin_allowed():
    q = _queries()
    client = _client(q)
    resp = client.post("/api/tokens", json={"name": "x"}, headers=_auth(LEGACY))
    assert resp.status_code == 201


def test_create_non_admin_forbidden():
    for raw in (SUB_RAW, RO_RAW):
        q = _queries()
        client = _client(q)
        resp = client.post("/api/tokens", json={"name": "x"}, headers=_auth(raw))
        assert resp.status_code == 403
        assert resp.json()["code"] == "forbidden"
        q.create_api_token.assert_not_awaited()


def test_create_audit_row_has_no_plaintext():
    q = _queries()
    client = _client(q)
    resp = client.post("/api/tokens", json={"name": "ci"}, headers=_auth(ADMIN_RAW))
    raw = resp.json()["token"]
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "token.create"
    assert kwargs["result"] == "ok"
    assert raw not in (kwargs["params_redacted"] or "")


def test_create_plaintext_never_stored_in_idempotency():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci"},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k-token-1"},
    )
    raw = resp.json()["token"]
    # The idempotency record for token.create stores no response body.
    q.complete_idempotency_record.assert_awaited()
    ck = q.complete_idempotency_record.await_args.kwargs
    assert ck["response_body"] is None
    assert raw not in (ck["response_body"] or "")


def test_create_replay_withholds_plaintext():
    q = _queries()
    # First request claims the key and completes; second sees it as completed.
    completed = {
        "state": "completed",
        "method": "POST",
        "path": "/api/tokens",
        "response_status": 201,
        "response_body": None,
        "content_type": None,
        "resource_id": "tok-new",
        "body_hash": None,
    }

    class _Rec:
        def __init__(self, d):
            self.__dict__.update(d)

    q.insert_idempotency_inprogress = AsyncMock(side_effect=[True, False])
    q.get_idempotency_record = AsyncMock(return_value=_Rec(completed))
    client = _client(q)
    headers = {**_auth(ADMIN_RAW), "Idempotency-Key": "k-token-2"}
    first = client.post("/api/tokens", json={"name": "ci"}, headers=headers)
    raw = first.json()["token"]
    second = client.post("/api/tokens", json={"name": "ci"}, headers=headers)
    assert second.headers.get("Idempotent-Replay") == "true"
    assert "token" not in second.json()
    assert raw not in second.text
    assert second.json()["code"] == "idempotent_replay"


# --- list ---------------------------------------------------------------------


def test_list_admin_only_and_no_hashes():
    q = _queries()
    client = _client(q)
    resp = client.get("/api/tokens", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    items = resp.json()
    assert {i["id"] for i in items} == {"tok-admin", "tok-sub", "tok-ro"}
    for item in items:
        assert "token_hash" not in item
        assert "token" not in item


def test_list_non_admin_forbidden():
    q = _queries()
    client = _client(q)
    resp = client.get("/api/tokens", headers=_auth(RO_RAW))
    assert resp.status_code == 403


# --- revoke -------------------------------------------------------------------


def test_revoke_admin_only():
    q = _queries()
    client = _client(q)
    resp = client.request("DELETE", "/api/tokens/tok-sub", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert resp.json()["revoked"] is True
    q.revoke_api_token.assert_awaited_with("tok-sub")


def test_revoke_non_admin_forbidden():
    q = _queries()
    client = _client(q)
    resp = client.request("DELETE", "/api/tokens/tok-sub", headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    q.revoke_api_token.assert_not_awaited()


def test_revoke_unknown_404():
    q = _queries()
    q.revoke_api_token = AsyncMock(return_value=False)
    client = _client(q)
    resp = client.request("DELETE", "/api/tokens/nope", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


# --- P25 create semantics: expiry (D-P25-1) + scope (D-P25-3) ------------------


def _created(q: AsyncMock) -> ApiToken:
    """The ``ApiToken`` the route handed to ``create_api_token``."""
    return q.create_api_token.await_args.args[0]


def test_omitted_expires_in_s_applies_the_configured_default():
    q = _queries()
    client = _client(q, token_default_ttl_s=3600)
    resp = client.post("/api/tokens", json={"name": "ci"}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201
    stored = _created(q)
    assert stored.expires_at is not None
    delta = (stored.expires_at - datetime.now(UTC)).total_seconds()
    assert 3500 < delta <= 3600
    assert resp.json()["expires_at"] is not None


def test_explicit_null_expires_in_s_beats_the_configured_default():
    """(D-P25-1) An explicit ``null`` is "never expires", NOT "use the default"."""
    q = _queries()
    client = _client(q, token_default_ttl_s=3600)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "expires_in_s": None}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 201
    assert _created(q).expires_at is None
    assert resp.json()["expires_at"] is None


def test_explicit_value_beats_the_configured_default():
    q = _queries()
    client = _client(q, token_default_ttl_s=31_536_000)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "expires_in_s": 120}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 201
    stored = _created(q)
    assert stored.expires_at is not None
    assert (stored.expires_at - datetime.now(UTC)).total_seconds() <= 120


def test_omitted_expires_in_s_with_no_default_never_expires():
    """The shipped posture: expiry is opt-in, so a bare create is unchanged."""
    q = _queries()
    client = _client(q)
    resp = client.post("/api/tokens", json={"name": "ci"}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201
    assert _created(q).expires_at is None


def test_expires_in_s_below_the_floor_is_422():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "expires_in_s": 59}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422
    q.create_api_token.assert_not_awaited()


def test_expires_in_s_above_the_ceiling_is_422():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "expires_in_s": 31_536_001}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422
    q.create_api_token.assert_not_awaited()


def test_scope_services_are_persisted_and_returned():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci", "role": "submitter", "scope_services": ["api", "worker", "api"]},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 201
    # Deduped, order preserved.
    assert _created(q).scope_services == ["api", "worker"]
    assert resp.json()["scope_services"] == ["api", "worker"]


def test_admin_role_with_scope_is_refused():
    """(D-P25-3) A scoped admin is a contradiction — refused at creation."""
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci", "role": "admin", "scope_services": ["api"]},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "token.scope_not_allowed"
    q.create_api_token.assert_not_awaited()


def test_admin_role_without_scope_is_still_allowed():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "role": "admin"}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 201
    assert _created(q).scope_services is None


def test_empty_scope_services_is_422_pointing_at_null():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens", json={"name": "ci", "scope_services": []}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422
    assert "null" in resp.text
    q.create_api_token.assert_not_awaited()


def test_non_dns_label_scope_entry_is_422():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci", "scope_services": ["Not A Label"]},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 422
    q.create_api_token.assert_not_awaited()


def test_over_long_scope_list_is_422():
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci", "scope_services": [f"svc{i}" for i in range(33)]},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 422
    q.create_api_token.assert_not_awaited()


def test_audit_row_records_the_lifecycle_parameters():
    """(§3.1.4) Minting a scoped, short-lived token is exactly what an auditor wants."""
    q = _queries()
    client = _client(q)
    resp = client.post(
        "/api/tokens",
        json={"name": "ci", "expires_in_s": 3600, "scope_services": ["api"]},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 201
    rows = [c.kwargs for c in q.insert_audit_log.await_args_list if c.kwargs.get("action")]
    creates = [r for r in rows if r["action"] == "token.create"]
    assert len(creates) == 1
    params = json.loads(creates[0]["params_redacted"])
    assert params["expires_in_s"] == 3600
    assert params["scope_services"] == ["api"]
    # The plaintext is still never in the audit row.
    assert resp.json()["token"] not in creates[0]["params_redacted"]


def test_token_list_projects_expiry_and_scope():
    q = _queries()
    q.list_api_tokens = AsyncMock(
        return_value=[
            ApiToken(
                id="tok-x",
                name="scoped",
                role=TokenRole.submitter,
                token_hash="h",
                expires_at=datetime(2030, 1, 1, tzinfo=UTC),
                scope_services=["api"],
            )
        ]
    )
    client = _client(q)
    resp = client.get("/api/tokens", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["expires_at"].startswith("2030-01-01")
    assert row["scope_services"] == ["api"]
    assert "token_hash" not in row


# --- P25 WP3: self-service inspect (GET /tokens/self, D-P25-4) -----------------


def _expiring_queries(*, expires_at: datetime | None, role: TokenRole = TokenRole.submitter):
    """A queries stand-in whose ONE token carries ``expires_at``.

    Kept out of the module-level ``_TOKENS`` map so the existing list/create
    assertions (which pin the exact id set) stay untouched.
    """
    row = ApiToken(
        id="tok-exp",
        name="expiring",
        role=role,
        token_hash=hash_token("exp-raw"),
        max_gpus=2,
        max_concurrent_jobs=4,
        expires_at=expires_at,
        scope_services=["api"],
    )
    q = _queries()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: row if h == row.token_hash else None)
    q.get_api_token_by_id = AsyncMock(side_effect=lambda tid: row if tid == row.id else None)
    return q, row


def test_self_view_projects_the_callers_own_row():
    q = _queries()
    client = _client(q)
    resp = client.get("/api/tokens/self", headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "tok-sub"
    assert body["role"] == "submitter"
    assert body["rotatable"] is True
    # Never a hash, never a plaintext.
    assert "token_hash" not in body
    assert "token" not in body


def test_self_view_computes_expires_in_s():
    q, _row = _expiring_queries(expires_at=datetime.now(UTC) + timedelta(hours=1))
    client = _client(q)
    body = client.get("/api/tokens/self", headers=_auth("exp-raw")).json()
    assert 3500 < body["expires_in_s"] <= 3600
    assert body["scope_services"] == ["api"]


def test_self_view_of_a_never_expiring_token_has_null_expires_in_s():
    q = _queries()
    client = _client(q)
    body = client.get("/api/tokens/self", headers=_auth(SUB_RAW)).json()
    assert body["expires_at"] is None
    assert body["expires_in_s"] is None


def test_self_view_readonly_is_allowed_but_rotate_is_not():
    """(D-P25-4) Inspect is universal; rotate is denied by the coarse write gate."""
    q = _queries()
    client = _client(q)
    assert client.get("/api/tokens/self", headers=_auth(RO_RAW)).status_code == 200

    resp = client.post("/api/tokens/self/rotate", headers=_auth(RO_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    # The denial happens in the middleware — the route never ran.
    q.rotate_api_token_hash.assert_not_awaited()


def test_self_view_of_a_sentinel_principal_is_synthetic():
    """(D-P25-4) The legacy global token gets a true answer, not a 404."""
    q = _queries()
    client = _client(q)
    resp = client.get("/api/tokens/self", headers=_auth(LEGACY))
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] is None
    assert body["role"] == "admin"
    assert body["rotatable"] is False
    assert body["expires_at"] is None
    assert body["expires_in_s"] is None
    assert body["scope_services"] is None
    q.get_api_token_by_id.assert_not_awaited()


# --- P25 WP3: expiry denies BOTH self routes (D-P25-2 sub-ruling) --------------


def test_expired_token_cannot_read_its_own_row():
    q, _row = _expiring_queries(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    client = _client(q)
    resp = client.get("/api/tokens/self", headers=_auth("exp-raw"))
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"
    q.get_api_token_by_id.assert_not_awaited()


def test_expired_token_cannot_rotate_itself():
    """(D-P25-2 sub-ruling) Rotation is a BEFORE-expiry action, by design.

    The regression this pins: exempting ``/tokens/self/rotate`` from the expiry
    check would turn ``expires_at`` into "expired tokens keep one live privilege
    forever" — a posture change, not an ergonomics fix.
    """
    q, _row = _expiring_queries(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    client = _client(q)
    resp = client.post("/api/tokens/self/rotate", headers=_auth("exp-raw"))
    assert resp.status_code == 403
    assert resp.json()["code"] == "token_expired"
    q.rotate_api_token_hash.assert_not_awaited()


def test_token_expired_message_names_the_admin_re_mint_path():
    """The remediation string must never drift back to an unreachable endpoint."""
    q, _row = _expiring_queries(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    client = _client(q)
    for resp in (
        client.get("/api/tokens/self", headers=_auth("exp-raw")),
        client.post("/api/tokens/self/rotate", headers=_auth("exp-raw")),
    ):
        message = resp.json()["message"]
        assert "nerdit token create" in message
        assert "cannot rotate itself" in message


# --- P25 WP3: rotate (POST /tokens/self/rotate, D-P25-4) ----------------------


def _rotatable(role: TokenRole = TokenRole.submitter, *, expires_at: datetime | None = None):
    """A queries stand-in backed by a tiny in-memory ``token_hash -> row`` store.

    Real enough to prove the swap: the old plaintext stops resolving and the new
    one starts, through the same middleware lookup a live daemon uses.
    """
    row = ApiToken(
        id="tok-rot",
        name="ci-runner",
        role=role,
        token_hash=hash_token("rot-raw"),
        max_gpus=3,
        max_concurrent_jobs=5,
        last_used_at=datetime(2026, 1, 1, tzinfo=UTC),
        expires_at=expires_at,
        scope_services=["api"],
    )
    store: dict[str, ApiToken] = {row.token_hash: row}

    async def _rotate(token_id, new_hash, *, expires_at, set_expiry):
        current = next((t for t in store.values() if t.id == token_id), None)
        if current is None or current.revoked:
            return False
        del store[current.token_hash]
        update: dict = {"token_hash": new_hash, "last_used_at": None}
        if set_expiry:
            update["expires_at"] = expires_at
        store[new_hash] = current.model_copy(update=update)
        return True

    q = _queries()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: store.get(h))
    q.get_api_token_by_id = AsyncMock(
        side_effect=lambda tid: next((t for t in store.values() if t.id == tid), None)
    )
    q.rotate_api_token_hash = AsyncMock(side_effect=_rotate)
    return q, store


def test_rotate_swaps_the_hash_old_plaintext_dies_new_one_works():
    q, store = _rotatable()
    client = _client(q)
    resp = client.post("/api/tokens/self/rotate", headers=_auth("rot-raw"))
    assert resp.status_code == 200
    new_raw = resp.json()["token"]
    assert new_raw.startswith("nrd_")
    assert new_raw != "rot-raw"

    # Only the hash was persisted, and only one row exists.
    assert list(store) == [hash_token(new_raw)]
    assert store[hash_token(new_raw)].token_hash != new_raw

    # The old secret no longer authenticates; the new one does.
    assert client.get("/api/tokens/self", headers=_auth("rot-raw")).status_code == 403
    assert client.get("/api/tokens/self", headers=_auth(new_raw)).status_code == 200


def test_rotate_preserves_identity_quotas_and_scope_and_resets_last_used_at():
    q, store = _rotatable()
    client = _client(q)
    body = client.post("/api/tokens/self/rotate", headers=_auth("rot-raw")).json()

    assert body["id"] == "tok-rot"
    assert body["name"] == "ci-runner"
    assert body["role"] == "submitter"
    assert body["max_gpus"] == 3
    assert body["max_concurrent_jobs"] == 5
    assert body["scope_services"] == ["api"]
    # ``last_used_at`` describes THIS secret's usage, so the swap clears it.
    assert body["last_used_at"] is None
    assert next(iter(store.values())).last_used_at is None


def test_rotate_without_extend_leaves_the_expiry_exactly_where_it_was():
    """(D-P25-4) A rotation that quietly reset the clock would defeat D-P25-1."""
    original = datetime.now(UTC) + timedelta(hours=2)
    q, store = _rotatable(expires_at=original)
    client = _client(q, token_default_ttl_s=31_536_000)
    body = client.post("/api/tokens/self/rotate", headers=_auth("rot-raw")).json()

    assert q.rotate_api_token_hash.await_args.kwargs["set_expiry"] is False
    assert next(iter(store.values())).expires_at == original
    assert body["expires_at"].startswith(original.isoformat()[:16])


def test_rotate_accepts_an_empty_body():
    """A bare POST is the common case — the body is optional, not merely empty."""
    q, _store = _rotatable()
    client = _client(q)
    assert client.post("/api/tokens/self/rotate", headers=_auth("rot-raw")).status_code == 200
    assert q.rotate_api_token_hash.await_args.kwargs["set_expiry"] is False


def test_rotate_with_extend_and_an_explicit_value_pushes_the_expiry():
    q, store = _rotatable(expires_at=datetime.now(UTC) + timedelta(seconds=90))
    client = _client(q)
    resp = client.post(
        "/api/tokens/self/rotate",
        json={"extend": True, "expires_in_s": 7200},
        headers=_auth("rot-raw"),
    )
    assert resp.status_code == 200
    assert q.rotate_api_token_hash.await_args.kwargs["set_expiry"] is True
    stored = next(iter(store.values()))
    assert stored.expires_at is not None
    delta = (stored.expires_at - datetime.now(UTC)).total_seconds()
    assert 7100 < delta <= 7200


def test_rotate_with_extend_falls_back_to_the_site_default_ttl():
    q, store = _rotatable()
    client = _client(q, token_default_ttl_s=3600)
    resp = client.post("/api/tokens/self/rotate", json={"extend": True}, headers=_auth("rot-raw"))
    assert resp.status_code == 200
    stored = next(iter(store.values()))
    assert stored.expires_at is not None
    assert 3500 < (stored.expires_at - datetime.now(UTC)).total_seconds() <= 3600


def test_rotate_with_extend_and_no_policy_leaves_the_expiry_untouched():
    """No TTL anywhere ⇒ nothing to push to; ``extend`` must not CLEAR an expiry."""
    original = datetime.now(UTC) + timedelta(hours=2)
    q, store = _rotatable(expires_at=original)
    client = _client(q)  # no token_default_ttl_s
    resp = client.post("/api/tokens/self/rotate", json={"extend": True}, headers=_auth("rot-raw"))
    assert resp.status_code == 200
    assert q.rotate_api_token_hash.await_args.kwargs["set_expiry"] is False
    assert next(iter(store.values())).expires_at == original


def test_rotate_rejects_an_unknown_body_key():
    q, _store = _rotatable()
    client = _client(q)
    resp = client.post(
        "/api/tokens/self/rotate", json={"extend": True, "nope": 1}, headers=_auth("rot-raw")
    )
    assert resp.status_code == 422
    q.rotate_api_token_hash.assert_not_awaited()


def test_rotate_is_refused_for_a_sentinel_principal():
    q = _queries()
    client = _client(q)
    resp = client.post("/api/tokens/self/rotate", headers=_auth(LEGACY))
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "token.not_rotatable"
    assert "[daemon].auth_token" in body["hint"]
    q.rotate_api_token_hash.assert_not_awaited()


def test_rotate_audit_row_stamps_the_token_and_carries_no_plaintext():
    q, _store = _rotatable()
    client = _client(q)
    resp = client.post("/api/tokens/self/rotate", json={"extend": False}, headers=_auth("rot-raw"))
    raw = resp.json()["token"]

    rows = [c.kwargs for c in q.insert_audit_log.await_args_list if c.kwargs.get("action")]
    rotates = [r for r in rows if r["action"] == "token.rotate"]
    assert len(rotates) == 1
    row = rotates[0]
    assert row["result"] == "ok"
    assert row["target_type"] == "token"
    assert row["target_id"] == "tok-rot"
    assert row["principal_id"] == "tok-rot"
    assert raw not in json.dumps(row, default=str)


def test_rotate_response_body_is_never_cached_and_a_replay_withholds_it():
    """(D-P25-4 trade 2) A replay is an envelope, never a second plaintext."""
    q, store = _rotatable()

    class _Rec:
        def __init__(self, d):
            self.__dict__.update(d)

    q.insert_idempotency_inprogress = AsyncMock(side_effect=[True, False])
    q.get_idempotency_record = AsyncMock(
        return_value=_Rec(
            {
                "state": "completed",
                "method": "POST",
                "path": "/api/tokens/self/rotate",
                "response_status": 200,
                "response_body": None,
                "content_type": None,
                "resource_id": "tok-rot",
                "body_hash": None,
            }
        )
    )
    client = _client(q)
    headers = {**_auth("rot-raw"), "Idempotency-Key": "k-rot-1"}
    first = client.post("/api/tokens/self/rotate", headers=headers)
    raw = first.json()["token"]

    # ``token.rotate`` is in NO_BODY_CACHE_ACTIONS: nothing was stored.
    ck = q.complete_idempotency_record.await_args.kwargs
    assert ck["response_body"] is None

    second = client.post("/api/tokens/self/rotate", headers={**headers, **_auth(raw)})
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json()["code"] == "idempotent_replay"
    assert "token" not in second.json()
    assert raw not in second.text
    # Exactly one swap happened — the replay never reached the route.
    assert q.rotate_api_token_hash.await_count == 1
    assert list(store) == [hash_token(raw)]


def test_self_view_when_the_row_vanished_answers_404_not_500():
    """Unreachable in a live daemon — but a race must not become a 500."""
    q = _queries()
    q.get_api_token_by_id = AsyncMock(return_value=None)
    client = _client(q)
    resp = client.get("/api/tokens/self", headers=_auth(SUB_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert "nerdit token create" in resp.json()["hint"]


def test_rotate_when_the_swap_finds_no_active_row_answers_404():
    """A row revoked between the read and the UPDATE: the guard is in the SQL."""
    q, _store = _rotatable()
    q.rotate_api_token_hash = AsyncMock(return_value=False)
    client = _client(q)
    resp = client.post("/api/tokens/self/rotate", headers=_auth("rot-raw"))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    # No plaintext leaks out of a failed rotation.
    assert "nrd_" not in resp.text
