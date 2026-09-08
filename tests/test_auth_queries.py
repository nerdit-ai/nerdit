"""Tests for the scoped-token query layer and the auth primitives (P1 / S3).

Pure-async tests over the in-memory DB (no TestClient), so they run under
pytest-asyncio without the Starlette/aiosqlite cross-loop hazard.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nerdit.daemon.auth import (
    LEGACY_ADMIN,
    LOCAL,
    Principal,
    QuotaExceeded,
    generate_token,
    hash_token,
    require_owner_or_admin,
    require_role,
)
from nerdit.daemon.errors import NerditError
from nerdit.db.models import ApiToken, Job, TokenRole


def _token(raw: str, *, name: str = "t", role: TokenRole = TokenRole.submitter, **kw) -> ApiToken:
    return ApiToken(name=name, role=role, token_hash=hash_token(raw), **kw)


# --- hash_token / generate_token ---------------------------------------------


def test_generate_token_has_prefix_and_entropy():
    tok = generate_token()
    assert tok.startswith("nrd_")
    # Two calls never collide.
    assert generate_token() != generate_token()
    assert len(tok) > 20


def test_hash_token_is_deterministic_and_hides_plaintext():
    h1 = hash_token("super-secret")
    h2 = hash_token("super-secret")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest
    assert "super-secret" not in h1
    assert hash_token("other") != h1


# --- create / get_by_hash -----------------------------------------------------


async def test_create_and_get_by_hash_roundtrip(queries):
    raw = generate_token()
    created = await queries.create_api_token(
        _token(raw, name="ci-bot", role=TokenRole.submitter, max_gpus=4, max_concurrent_jobs=2)
    )
    fetched = await queries.get_api_token_by_hash(hash_token(raw))
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "ci-bot"
    assert fetched.role == TokenRole.submitter
    assert fetched.max_gpus == 4
    assert fetched.max_concurrent_jobs == 2
    assert fetched.revoked is False


async def test_get_by_hash_unknown_returns_none(queries):
    assert await queries.get_api_token_by_hash(hash_token("nope")) is None


async def test_stored_row_holds_hash_not_plaintext(queries):
    raw = generate_token()
    await queries.create_api_token(_token(raw, name="x"))
    cursor = await queries._db.conn.execute("SELECT token_hash FROM api_tokens")
    row = await cursor.fetchone()
    assert row["token_hash"] == hash_token(raw)
    assert raw not in row["token_hash"]


# --- revoke / list ------------------------------------------------------------


async def test_revoke_hides_token_from_lookup_and_list(queries):
    raw = generate_token()
    tok = await queries.create_api_token(_token(raw))
    assert await queries.revoke_api_token(tok.id) is True
    # Revoked token no longer resolves.
    assert await queries.get_api_token_by_hash(hash_token(raw)) is None
    # Hidden from the default listing, visible with include_revoked.
    assert [t.id for t in await queries.list_api_tokens()] == []
    assert tok.id in [t.id for t in await queries.list_api_tokens(include_revoked=True)]


async def test_revoke_unknown_or_already_revoked_returns_false(queries):
    assert await queries.revoke_api_token("missing-id") is False
    raw = generate_token()
    tok = await queries.create_api_token(_token(raw))
    assert await queries.revoke_api_token(tok.id) is True
    assert await queries.revoke_api_token(tok.id) is False


async def test_list_orders_newest_first(queries):
    a = await queries.create_api_token(_token(generate_token(), name="a"))
    b = await queries.create_api_token(_token(generate_token(), name="b"))
    listed = await queries.list_api_tokens()
    assert {t.id for t in listed} == {a.id, b.id}


# --- touch --------------------------------------------------------------------


async def test_touch_sets_last_used_at(queries):
    raw = generate_token()
    tok = await queries.create_api_token(_token(raw))
    assert tok.last_used_at is None
    await queries.touch_api_token(tok.id)
    refetched = await queries.get_api_token_by_hash(hash_token(raw))
    assert refetched is not None
    assert refetched.last_used_at is not None


# --- rotate_api_token_hash (P25 D-P25-4) --------------------------------------


async def test_rotate_swaps_the_hash_and_keeps_everything_else(queries):
    old_raw, new_raw = generate_token(), generate_token()
    tok = await queries.create_api_token(
        _token(
            old_raw,
            name="ci-bot",
            role=TokenRole.submitter,
            max_gpus=4,
            max_concurrent_jobs=2,
            scope_services=["api"],
        )
    )
    await queries.touch_api_token(tok.id)

    assert (
        await queries.rotate_api_token_hash(
            tok.id, hash_token(new_raw), expires_at=None, set_expiry=False
        )
        is True
    )

    assert await queries.get_api_token_by_hash(hash_token(old_raw)) is None
    rotated = await queries.get_api_token_by_hash(hash_token(new_raw))
    assert rotated is not None
    assert rotated.id == tok.id
    assert rotated.name == "ci-bot"
    assert rotated.role == TokenRole.submitter
    assert rotated.max_gpus == 4
    assert rotated.max_concurrent_jobs == 2
    assert rotated.scope_services == ["api"]
    # ``last_used_at`` describes THIS secret's usage — the swap clears it.
    assert rotated.last_used_at is None


async def test_rotate_without_set_expiry_leaves_the_column_alone(queries):
    original = datetime.now(UTC) + timedelta(hours=3)
    old_raw, new_raw = generate_token(), generate_token()
    tok = await queries.create_api_token(_token(old_raw, expires_at=original))

    await queries.rotate_api_token_hash(
        tok.id, hash_token(new_raw), expires_at=None, set_expiry=False
    )
    rotated = await queries.get_api_token_by_hash(hash_token(new_raw))
    assert rotated is not None
    assert rotated.expires_at == original


async def test_rotate_with_set_expiry_pushes_the_column(queries):
    pushed = datetime.now(UTC) + timedelta(days=30)
    old_raw, new_raw = generate_token(), generate_token()
    tok = await queries.create_api_token(
        _token(old_raw, expires_at=datetime.now(UTC) + timedelta(minutes=5))
    )

    await queries.rotate_api_token_hash(
        tok.id, hash_token(new_raw), expires_at=pushed, set_expiry=True
    )
    rotated = await queries.get_api_token_by_hash(hash_token(new_raw))
    assert rotated is not None
    assert rotated.expires_at == pushed


async def test_rotate_refuses_a_revoked_or_missing_row(queries):
    raw, new_raw = generate_token(), generate_token()
    tok = await queries.create_api_token(_token(raw))
    await queries.revoke_api_token(tok.id)

    # The by-id read deliberately still returns revoked rows, so the guard has
    # to live in the UPDATE itself.
    assert await queries.get_api_token_by_id(tok.id) is not None
    assert (
        await queries.rotate_api_token_hash(
            tok.id, hash_token(new_raw), expires_at=None, set_expiry=False
        )
        is False
    )
    assert (
        await queries.rotate_api_token_hash(
            "nope", hash_token(new_raw), expires_at=None, set_expiry=False
        )
        is False
    )


# --- insert_audit_log ---------------------------------------------------------


async def test_insert_audit_log_roundtrip(queries):
    await queries.insert_audit_log(
        action="POST /jobs",
        result="denied",
        principal_id="tok123",
        principal_role="readonly",
        status_code=403,
        request_id="rid-1",
    )
    cursor = await queries._db.conn.execute(
        "SELECT action, result, principal_id, principal_role, status_code, request_id "
        "FROM audit_log"
    )
    row = await cursor.fetchone()
    assert row["action"] == "POST /jobs"
    assert row["result"] == "denied"
    assert row["principal_id"] == "tok123"
    assert row["principal_role"] == "readonly"
    assert row["status_code"] == 403
    assert row["request_id"] == "rid-1"


# --- Principal sentinels ------------------------------------------------------


def test_sentinels_are_admin():
    assert LEGACY_ADMIN.role == TokenRole.admin
    assert LEGACY_ADMIN.is_legacy_admin is True
    assert LOCAL.role == TokenRole.admin
    assert LOCAL.is_legacy_admin is False
    assert LEGACY_ADMIN.is_admin and LOCAL.is_admin


# --- require_role -------------------------------------------------------------


class _FakeState:
    pass


class _FakeRequest:
    def __init__(self, principal=None):
        self.state = _FakeState()
        if principal is not None:
            self.state.principal = principal


def test_require_role_allows_and_rejects():
    admin_req = _FakeRequest(Principal(token_id="a", name="a", role=TokenRole.admin))
    assert require_role(admin_req, TokenRole.admin).role == TokenRole.admin

    ro_req = _FakeRequest(Principal(token_id="r", name="r", role=TokenRole.readonly))
    with pytest.raises(NerditError) as exc:
        require_role(ro_req, TokenRole.admin, TokenRole.submitter)
    assert exc.value.status_code == 403
    assert exc.value.code == "forbidden"


def test_require_role_fails_closed_when_principal_unset():
    # I1: no principal attached → non-privileged ANONYMOUS, denied (fail-closed).
    import pytest

    req = _FakeRequest()
    with pytest.raises(NerditError) as exc:
        require_role(req, TokenRole.admin, TokenRole.submitter)
    assert exc.value.status_code == 403


# --- require_owner_or_admin ---------------------------------------------------


def _job(owner: str | None) -> Job:
    return Job(script_path="train.py", submitted_by_token=owner)


def test_owner_passes_admin_passes_other_rejected():
    sub = Principal(token_id="tok-1", name="s", role=TokenRole.submitter)
    # Owner can act.
    assert require_owner_or_admin(_FakeRequest(sub), _job("tok-1")).token_id == "tok-1"
    # Admin can act on anyone's job.
    admin = Principal(token_id="adm", name="a", role=TokenRole.admin)
    assert require_owner_or_admin(_FakeRequest(admin), _job("tok-1")).role == TokenRole.admin
    # Non-owner submitter rejected.
    with pytest.raises(NerditError) as exc:
        require_owner_or_admin(_FakeRequest(sub), _job("tok-2"))
    assert exc.value.status_code == 403


def test_null_owner_job_is_admin_only():
    sub = Principal(token_id="tok-1", name="s", role=TokenRole.submitter)
    with pytest.raises(NerditError):
        require_owner_or_admin(_FakeRequest(sub), _job(None))
    admin = Principal(token_id="adm", name="a", role=TokenRole.admin)
    assert require_owner_or_admin(_FakeRequest(admin), _job(None)).role == TokenRole.admin


# --- QuotaExceeded ------------------------------------------------------------


def test_quota_exceeded_to_error():
    err = QuotaExceeded("max_concurrent_jobs", limit=2, current=2).to_error()
    assert err.status_code == 403
    assert err.code == "quota_exceeded"
    assert err.extra["limit"] == 2
    assert err.extra["current"] == 2
