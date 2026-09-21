"""Route-level tests for the write-only ``/secrets`` surface (P4 / S6; P8
rotate-key + shared scope)."""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nerdit.core.secrets import SecretManager
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, Project, SecretClaim, TokenRole
from nerdit.db.queries import Queries

LEGACY = "legacy-global"
ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
OTHER_RAW = "other-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(OTHER_RAW): ApiToken(
        id="tok-other", name="o", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


def _service(owner: str | None = "tok-sub", name: str = "demo") -> Job:
    """A service-kind row owned by ``owner`` (a token id)."""
    return Job(
        id="svc-1",
        service_name=name,
        name=name,
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "nerdit-runtime:0.1", "port": 8000}),
        submitted_by_token=owner,
    )


def _queries(service: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # By default the target service exists and is owned by the submitter (tok-sub)
    # so ownership-agnostic tests exercise the happy path.
    q.get_service_by_name = AsyncMock(return_value=service if service is not None else _service())
    # Dict-backed `secret_claims` (P39): mint is INSERT OR IGNORE, so a second
    # mint on a taken name loses.
    claims: dict[str, SecretClaim] = {}
    q.claims = claims

    async def _mint(name: str, token_id: str | None, *, admin: bool = False) -> bool:
        if name in claims:
            return False
        claims[name] = SecretClaim(service_name=name, token_id=token_id)
        return True

    q.get_secret_claim = AsyncMock(side_effect=lambda name: claims.get(name))
    # (P40b) No `projects` row for any name unless a test plants one: a bare
    # AsyncMock would return a truthy MagicMock and read as a foreign project.
    q.get_project_by_name = AsyncMock(return_value=None)
    q.mint_secret_claim = AsyncMock(side_effect=_mint)
    q.delete_secret_claim = AsyncMock(side_effect=lambda name: claims.pop(name, None) is not None)
    return q


def _app(tmp_path, queries, *, with_audit: bool = False) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(secrets_router)
    app.state.queries = queries
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(tmp_path, queries=None, **kw) -> TestClient:
    return TestClient(_app(tmp_path, queries or _queries(), **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def test_set_and_list_names_only(tmp_path):
    c = _client(tmp_path)
    resp = c.post("/secrets/demo", json={"values": {"API_KEY": "s3cr3t"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "demo"
    assert body["keys"] == ["API_KEY"]
    # The value must never appear in the response.
    assert "s3cr3t" not in resp.text

    # GET returns names only.
    got = c.get("/secrets/demo", headers=_auth(SUB_RAW))
    assert got.json()["keys"] == ["API_KEY"]
    assert "s3cr3t" not in got.text


def test_set_merges(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    resp = c.post("/secrets/demo", json={"values": {"B": "2"}}, headers=_auth(SUB_RAW))
    assert resp.json()["keys"] == ["A", "B"]


def test_unknown_field_422_does_not_echo_the_secret_values(tmp_path):
    """A mistyped body field must not reflect the caller's secrets back.

    ``StrictRequestModel`` (P22 WP-C) turns ``{"valus": {...}}`` into an
    ``extra_forbidden`` error whose ``input`` is the whole submitted map — and
    the sibling ``missing`` error on ``values`` echoes the entire body. Both are
    masked (``errors._SECRET_INPUT_FIELDS`` + the unconditional
    ``extra_forbidden`` rule); the ``loc`` still names the offending key, which
    is what the caller has to fix.
    """
    c = _client(tmp_path)
    resp = c.post(
        "/secrets/demo",
        json={"valus": {"STRIPE_SECRET_KEY": "sk_live_hunter2"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    assert "sk_live_hunter2" not in resp.text
    diagnostics = resp.json()["diagnostics"]
    assert diagnostics, "masking must not empty the pydantic error list"
    assert any("valus" in (e.get("loc") or []) for e in diagnostics), (
        "the rejected field name is not a secret and must stay visible"
    )
    assert all(e.get("input", "***") == "***" for e in diagnostics)


def test_wrong_value_type_422_does_not_echo_the_secret_values(tmp_path):
    """A type error INSIDE ``values`` must not echo the neighbouring secret."""
    c = _client(tmp_path)
    resp = c.post(
        "/secrets/demo",
        json={"values": {"GOOD": "ok", "BAD": 7}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422
    assert "ok" not in resp.json()["diagnostics"][0].get("input", "")


def test_delete_key(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/demo", json={"values": {"A": "1", "B": "2"}}, headers=_auth(SUB_RAW))
    resp = c.delete("/secrets/demo/A", headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    assert c.get("/secrets/demo", headers=_auth(SUB_RAW)).json()["keys"] == ["B"]
    # Deleting a missing key is a 404.
    assert c.delete("/secrets/demo/ghost", headers=_auth(SUB_RAW)).status_code == 404


def test_delete_all(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    resp = c.delete("/secrets/demo", headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert c.get("/secrets/demo", headers=_auth(SUB_RAW)).json()["keys"] == []


def test_readonly_blocked_on_write(tmp_path):
    c = _client(tmp_path)
    assert (
        c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(RO_RAW)).status_code
        == 403
    )
    assert c.delete("/secrets/demo/A", headers=_auth(RO_RAW)).status_code == 403


def test_invalid_service_name_rejected(tmp_path):
    resp = _client(tmp_path).post(
        "/secrets/Bad_Name", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "secret.invalid_service"


def test_invalid_key_name_rejected_and_nothing_stored(tmp_path):
    """POST with an empty / malformed key name -> 422 secret.invalid_key, and
    nothing is stored (fix #3)."""
    c = _client(tmp_path)
    resp = c.post(
        "/secrets/shared", json={"values": {"": "x", "A=B": "y"}}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "secret.invalid_key"
    # Nothing stored: the shared scope still lists no keys.
    got = c.get("/secrets/shared", headers=_auth(ADMIN_RAW))
    assert got.json()["keys"] == []


def test_nul_value_rejected_and_no_value_fragment_leaks(tmp_path):
    """POST with a NUL-bearing value -> 422 secret.invalid_value; the response
    carries no fragment of the value and nothing is stored (fix #15)."""
    c = _client(tmp_path)
    # Build the NUL in code so no literal control byte lands in the test file;
    # json.dumps escapes it, which pydantic parses back to a real NUL.
    # The halves MUST avoid the hex alphabet: the envelope carries a
    # ``request_id`` of ``uuid.uuid4().hex`` (daemon/errors.py), so a marker
    # like "abc" is a substring a random id genuinely produces — measured at
    # ~0.6% of runs, i.e. an intermittent red with no bug behind it.
    bad_value = "zqp" + chr(0) + "wvu"
    body = json.dumps({"values": {"SECRETKEY": bad_value}})
    resp = c.post(
        "/secrets/demo",
        content=body,
        headers={**_auth(SUB_RAW), "Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "secret.invalid_value"
    # No fragment of the value appears in the response (only the key name may).
    assert "zqp" not in resp.text and "wvu" not in resp.text
    assert c.get("/secrets/demo", headers=_auth(SUB_RAW)).json()["keys"] == []


def test_multiline_value_accepted(tmp_path):
    """A multi-line (PEM-shaped) value is accepted end-to-end (fix #15)."""
    c = _client(tmp_path)
    pem = "-----BEGIN KEY-----\nAB\nCD\n-----END KEY-----\n"
    resp = c.post("/secrets/demo", json={"values": {"PEM": pem}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    assert resp.json()["keys"] == ["PEM"]


def test_audit_masks_secret_values(tmp_path):
    q = _queries()
    c = _client(tmp_path, q, with_audit=True)
    resp = c.post(
        "/secrets/demo", json={"values": {"API_KEY": "supersecret"}}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert "supersecret" not in logged
    # The key name is fine to log (auditability).
    assert "API_KEY" in logged


# --- ownership authorization (Codex #2) --------------------------------------


def test_non_owner_submitter_forbidden_on_set(tmp_path):
    # tok-other is a submitter but does not own 'demo' (owned by tok-sub).
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(OTHER_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_non_owner_submitter_forbidden_on_list(tmp_path):
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.get("/secrets/demo", headers=_auth(OTHER_RAW))
    assert resp.status_code == 403


def test_non_owner_submitter_forbidden_on_delete_key(tmp_path):
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.delete("/secrets/demo/A", headers=_auth(OTHER_RAW))
    assert resp.status_code == 403


def test_non_owner_submitter_forbidden_on_delete_all(tmp_path):
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.delete("/secrets/demo", headers=_auth(OTHER_RAW))
    assert resp.status_code == 403


def test_owner_can_set_secrets(tmp_path):
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    assert resp.json()["keys"] == ["A"]


def test_admin_can_set_secrets_for_any_service(tmp_path):
    # 'demo' is owned by tok-sub; an admin may still manage it.
    c = _client(tmp_path, _queries(_service("tok-sub")))
    resp = c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200


def test_null_owner_service_is_admin_only(tmp_path):
    # NULL-owner rows are admin-only: a submitter cannot squat their secrets.
    c = _client(tmp_path, _queries(_service(None)))
    assert (
        c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW)).status_code
        == 403
    )
    assert (
        _client(tmp_path, _queries(_service(None)))
        .post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(ADMIN_RAW))
        .status_code
        == 200
    )


def _queries_no_service(claim: str | None = None, *, claimed: bool = False) -> AsyncMock:
    """No row for any name; optionally a pre-existing claim by token `claim`."""
    q = _queries()
    q.get_service_by_name = AsyncMock(return_value=None)
    if claimed:
        q.claims["ghost"] = SecretClaim(service_name="ghost", token_id=claim)
    return q


# --- rowless names: the D-P39-2 table ------------------------------------------


def test_set_on_rowless_name_mints_a_claim_and_returns_names(tmp_path):
    q = _queries_no_service()
    c = _client(tmp_path, q, with_audit=True)
    resp = c.post("/secrets/ghost", json={"values": {"A": "s3cr3t"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"service": "ghost", "keys": ["A"]}
    assert "s3cr3t" not in resp.text
    assert q.claims["ghost"].token_id == "tok-sub"
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert "s3cr3t" not in logged
    assert json.loads(q.insert_audit_log.call_args.kwargs["params_redacted"])["claimed"] is True
    # The claimant may read and delete; a later write merges without re-minting.
    assert c.get("/secrets/ghost", headers=_auth(SUB_RAW)).json()["keys"] == ["A"]
    resp = c.post("/secrets/ghost", json={"values": {"B": "2"}}, headers=_auth(SUB_RAW))
    assert resp.json()["keys"] == ["A", "B"]
    assert "claimed" not in json.loads(q.insert_audit_log.call_args.kwargs["params_redacted"])


def test_claim_by_another_token_is_the_same_403_as_a_foreign_row(tmp_path):
    q = _queries_no_service("tok-sub", claimed=True)
    c = _client(tmp_path, q)
    # Same envelope as require_owner_or_admin: no "claimed, not deployed" oracle.
    expected = (
        _client(tmp_path, _queries(_service("tok-sub")))
        .post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(OTHER_RAW))
        .json()
    )
    for resp in (
        c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(OTHER_RAW)),
        c.get("/secrets/ghost", headers=_auth(OTHER_RAW)),
        c.delete("/secrets/ghost", headers=_auth(OTHER_RAW)),
        c.delete("/secrets/ghost/A", headers=_auth(OTHER_RAW)),
    ):
        assert resp.status_code == 403
        assert resp.json()["code"] == "forbidden"
        assert resp.json()["message"] == expected["message"]
        assert resp.json()["hint"] == expected["hint"]
    assert not SecretManager(tmp_path / "secrets").exists("ghost")


def test_null_token_claim_is_admin_only(tmp_path):
    q = _queries_no_service(None, claimed=True)
    c = _client(tmp_path, q)
    assert c.get("/secrets/ghost", headers=_auth(SUB_RAW)).status_code == 403
    assert c.get("/secrets/ghost", headers=_auth(ADMIN_RAW)).status_code == 200


def test_admin_bypasses_a_foreign_claim(tmp_path):
    q = _queries_no_service("tok-sub", claimed=True)
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert q.claims["ghost"].token_id == "tok-sub"  # the claim is not re-pointed
    assert c.get("/secrets/ghost", headers=_auth(ADMIN_RAW)).json()["keys"] == ["A"]


def test_read_or_delete_on_rowless_unclaimed_name_404_for_non_admin(tmp_path):
    q = _queries_no_service()
    c = _client(tmp_path, q)
    resp = c.get("/secrets/ghost", headers=_auth(SUB_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert "nerdit secrets set ghost" in resp.json()["hint"]
    assert c.delete("/secrets/ghost", headers=_auth(SUB_RAW)).status_code == 404
    assert c.delete("/secrets/ghost/A", headers=_auth(SUB_RAW)).status_code == 404
    assert q.mint_secret_claim.await_count == 0


def test_admin_read_and_delete_on_rowless_unclaimed_name(tmp_path):
    """An admin removes an orphan file nobody claims (the D-P39-2 'admin only' cell)."""
    SecretManager(tmp_path / "secrets").set("ghost", {"A": "1"})
    q = _queries_no_service()
    c = _client(tmp_path, q)
    assert c.get("/secrets/ghost", headers=_auth(ADMIN_RAW)).json()["keys"] == ["A"]
    resp = c.delete("/secrets/ghost", headers=_auth(ADMIN_RAW))
    assert resp.json() == {"service": "ghost", "deleted": True}
    assert not SecretManager(tmp_path / "secrets").exists("ghost")


def test_orphan_file_409_for_non_admin_admin_claims_it(tmp_path):
    """A pre-P39 leftover file is never adopted by a stranger's write."""
    SecretManager(tmp_path / "secrets").set("ghost", {"OLD": "prev-owner-value"})
    q = _queries_no_service()
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "secret.orphaned_scope"
    assert "nerdit secrets rm ghost" in resp.json()["hint"]
    assert "prev-owner-value" not in resp.text
    assert q.claims == {}
    # A read still 404s: the file's presence is no oracle for a non-admin.
    assert c.get("/secrets/ghost", headers=_auth(SUB_RAW)).status_code == 404
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert resp.json()["keys"] == ["A", "OLD"]
    assert q.claims["ghost"].token_id == "tok-admin"


def test_legacy_json_orphan_counts_as_a_file(tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "ghost.json").write_text('{"OLD": "v"}', encoding="utf-8")
    c = _client(tmp_path, _queries_no_service())
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "secret.orphaned_scope"


def test_delete_all_drops_the_claim_single_key_delete_keeps_it(tmp_path):
    q = _queries_no_service()
    c = _client(tmp_path, q)
    c.post("/secrets/ghost", json={"values": {"A": "1", "B": "2"}}, headers=_auth(SUB_RAW))
    assert c.delete("/secrets/ghost/A", headers=_auth(SUB_RAW)).status_code == 200
    assert "ghost" in q.claims
    resp = c.delete("/secrets/ghost", headers=_auth(SUB_RAW))
    assert resp.json() == {"service": "ghost", "deleted": True}
    assert "ghost" not in q.claims
    # Nothing left: the name is free again for anyone.
    assert c.get("/secrets/ghost", headers=_auth(OTHER_RAW)).status_code == 404


def _losing_mint(q: AsyncMock, winner: str | None) -> None:
    """The mint loses to a claim `winner` minted after the route's claim read."""

    async def _mint(name: str, token_id: str | None, *, admin: bool = False) -> bool:
        q.claims[name] = SecretClaim(service_name=name, token_id=winner)
        return False

    q.mint_secret_claim = AsyncMock(side_effect=_mint)


def test_lost_mint_race_to_a_foreign_claim_is_403(tmp_path):
    q = _queries_no_service()
    _losing_mint(q, "tok-other")
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert not SecretManager(tmp_path / "secrets").exists("ghost")
    assert q.claims["ghost"].token_id == "tok-other"


def test_lost_mint_race_to_own_claim_still_writes(tmp_path):
    """Two same-token sets in flight: the loser merges instead of being refused."""
    q = _queries_no_service()
    _losing_mint(q, "tok-sub")
    c = _client(tmp_path, q, with_audit=True)
    resp = c.post("/secrets/ghost", json={"values": {"B": "2"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"service": "ghost", "keys": ["B"]}
    assert "claimed" not in json.loads(q.insert_audit_log.call_args.kwargs["params_redacted"])


def test_lost_mint_race_to_a_vanished_claim_is_403(tmp_path):
    """Nothing landed on the re-read (claim minted and deleted again): fail closed."""
    q = _queries_no_service()
    q.mint_secret_claim = AsyncMock(return_value=False)
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert not SecretManager(tmp_path / "secrets").exists("ghost")


def _project(owner: str | None) -> Project:
    return Project(id="prj_aaaaaaaaaaaaaaaa", name="ghost", submitted_by_token=owner)


@pytest.mark.parametrize("owner", ["tok-other", None])
def test_lost_mint_to_a_foreign_project_is_the_claim_403_with_no_oracle(tmp_path, owner):
    """(P40b / D-P40-5 rule 2) The mint's `projects` predicate refused the claim;
    `_landed_is_callers` re-judges the project and answers the claim envelope --
    a stranger cannot tell "project reserved" from "claimed". A NULL-owner project
    is foreign to every non-admin. Nothing minted, nothing written."""
    q = _queries_no_service()
    q.mint_secret_claim = AsyncMock(return_value=False)
    q.get_project_by_name = AsyncMock(return_value=_project(owner))
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert "project" not in resp.text.lower()
    assert not SecretManager(tmp_path / "secrets").exists("ghost")
    assert q.claims == {}
    q.mint_secret_claim.assert_awaited_once_with("ghost", "tok-sub", admin=False)


def test_admin_mint_passes_admin_and_bypasses_a_foreign_project(tmp_path):
    q = _queries_no_service()
    q.get_project_by_name = AsyncMock(return_value=_project("tok-other"))
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200, resp.text
    q.mint_secret_claim.assert_awaited_once_with("ghost", "tok-admin", admin=True)
    assert q.claims["ghost"].token_id == "tok-admin"


def test_row_landing_between_the_row_read_and_the_mint_is_403_for_a_stranger(tmp_path):
    """A fresh deploy committing after the unlocked row read: the mint refuses the
    row and the loser is judged against it, never written into its scope."""
    q = _queries_no_service()
    q.get_service_by_name = AsyncMock(side_effect=[None, _service("tok-other")])
    q.mint_secret_claim = AsyncMock(return_value=False)  # the row guard in the query
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert not SecretManager(tmp_path / "secrets").exists("ghost")
    assert q.claims == {}


def test_row_landing_between_the_row_read_and_the_mint_writes_for_its_owner(tmp_path):
    q = _queries_no_service()
    row = _service("tok-sub")
    # Third read: (P40c) the flag demotion maps the label to its project.
    q.get_service_by_name = AsyncMock(side_effect=[None, row, row])
    q.mint_secret_claim = AsyncMock(return_value=False)
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    assert q.claims == {}


def test_stale_claim_read_over_a_same_token_file_is_not_an_orphan(tmp_path):
    """The claim read predates a same-token winner that already wrote the file."""
    q = _queries_no_service()
    q.claims["ghost"] = SecretClaim(service_name="ghost", token_id="tok-sub")
    q.get_secret_claim = AsyncMock(side_effect=[None, q.claims["ghost"]])
    SecretManager(tmp_path / "secrets").set("ghost", {"A": "1"})
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"B": "2"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["keys"] == ["A", "B"]
    assert q.mint_secret_claim.await_count == 0


def test_bad_item_never_mints_a_claim(tmp_path):
    q = _queries_no_service()
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"1bad": "v"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    assert q.claims == {}


def test_out_of_scope_rowless_name_403_with_no_oracle(tmp_path):
    scoped = ApiToken(
        id="tok-scoped",
        name="sc",
        role=TokenRole.submitter,
        token_hash=hash_token("scoped-raw"),
        scope_services=["other-app"],
    )
    q = _queries_no_service()
    tokens = {**_TOKENS, scoped.token_hash: scoped}
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: tokens.get(h))
    c = _client(tmp_path, q)
    for resp in (
        c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth("scoped-raw")),
        c.get("/secrets/ghost", headers=_auth("scoped-raw")),
        c.delete("/secrets/ghost", headers=_auth("scoped-raw")),
    ):
        assert resp.status_code == 403
        assert resp.json()["code"] == "forbidden"
    q.get_service_by_name.assert_not_awaited()
    q.get_secret_claim.assert_not_awaited()
    assert q.claims == {}


def test_readonly_still_403_on_rowless_name(tmp_path):
    q = _queries_no_service()
    c = _client(tmp_path, q)
    resp = c.post("/secrets/ghost", json={"values": {"A": "1"}}, headers=_auth(RO_RAW))
    assert resp.status_code == 403
    assert q.claims == {}


# --- POST /secrets/rotate-key (P8) --------------------------------------------


def test_rotate_key_admin_only(tmp_path):
    c = _client(tmp_path)
    # Submitter and readonly are both refused (readonly at the coarse gate).
    resp = c.post("/secrets/rotate-key", headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert c.post("/secrets/rotate-key", headers=_auth(RO_RAW)).status_code == 403
    # Admin rotates (nothing stored yet => zero files rewritten).
    ok = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert ok.status_code == 200
    assert ok.json() == {"services_rewritten": 0}


def test_rotate_key_literal_route_wins_over_service_param(tmp_path):
    # The literal route must win over POST /secrets/{service}: the response is
    # the rotation shape, never a set_secrets(service="rotate-key") outcome.
    q = _queries()
    c = _client(tmp_path, q, with_audit=True)
    resp = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert "services_rewritten" in resp.json()
    assert "keys" not in resp.json()
    # And the audit rule maps it to secret.rotate_key, not secret.set.
    assert q.insert_audit_log.call_args.kwargs["action"] == "secret.rotate_key"


def test_rotate_key_reencrypts_stored_files_and_counts(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/demo", json={"values": {"API_KEY": "s3cr3t"}}, headers=_auth(SUB_RAW))
    resp = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert resp.json() == {"services_rewritten": 1}
    # The secrets still resolve under the new key.
    assert c.get("/secrets/demo", headers=_auth(SUB_RAW)).json()["keys"] == ["API_KEY"]


def test_rotate_key_conflict_409_while_staged_key_exists(tmp_path):
    c = _client(tmp_path)
    # A leftover staged key (crashed rotation) refuses a second rotation until
    # the daemon restart resumes it.
    (tmp_path / "secrets.key.new").write_text("deadbeef\n", encoding="utf-8")
    resp = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "secret.rotation_in_progress"


@pytest.mark.asyncio
async def test_rotate_key_runs_off_loop_and_keeps_daemon_responsive(tmp_path):
    """M5: the rotate route dispatches ``rotate_key`` via ``asyncio.to_thread``.

    A slow rotation (holding the secrets RLock in a worker thread) must not
    freeze the event loop — an unrelated request completes while it is blocked.
    A regression to a synchronous on-loop call would freeze the loop, so the
    ``wait_for`` guards fail (never hang).
    """
    entered = threading.Event()
    release = threading.Event()

    def _slow_rotate() -> int:
        entered.set()
        release.wait(timeout=5)
        return 3

    app = _app(tmp_path, _queries())
    app.state.secret_manager.rotate_key = _slow_rotate  # type: ignore[method-assign]

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        rotate = asyncio.create_task(client.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW)))
        try:
            # A frozen loop would never let this poll tick.
            async def _await_entered() -> None:
                while not entered.is_set():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_await_entered(), timeout=3.0)

            # The loop is live: an unrelated request returns while rotate blocks.
            listed = await asyncio.wait_for(
                client.get("/secrets/demo", headers=_auth(ADMIN_RAW)), timeout=3.0
            )
            assert listed.status_code == 200
        finally:
            release.set()

        resp = await asyncio.wait_for(rotate, timeout=3.0)
        assert resp.status_code == 200
        assert resp.json() == {"services_rewritten": 3}


def test_legacy_rotate_key_service_blocks_rotation(tmp_path):
    # Startup guard: a pre-P8 service literally named 'rotate-key' disables the
    # global rotation so the literal route never silently rotates when the admin
    # may have meant to write that service's secrets.
    app = _app(tmp_path, _queries())
    app.state.rotate_key_blocked = True
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "secret.rotate_key_unavailable"


def test_rotate_key_audited_without_key_material(tmp_path):
    q = _queries()
    c = _client(tmp_path, q, with_audit=True)
    c.post("/secrets/demo", json={"values": {"API_KEY": "s3cr3t"}}, headers=_auth(SUB_RAW))
    resp = c.post("/secrets/rotate-key", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert json.loads(q.insert_audit_log.call_args.kwargs["params_redacted"]) == {
        "services_rewritten": 1
    }
    # Neither the key hex nor a stored value may reach the audit row.
    key_hex = (tmp_path / "secrets.key").read_text(encoding="utf-8").strip()
    assert key_hex not in logged
    assert "s3cr3t" not in logged


def test_decrypt_failure_maps_to_structured_500_on_crud(tmp_path):
    # A corrupt/wrong-key .enc surfaces through the whole CRUD quartet (set and
    # delete_key load() internally to merge) — the client must get the
    # structured envelope with the kid-bearing hint, never a bare 500.
    c = _client(tmp_path)
    c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW))
    (tmp_path / "secrets" / "demo.enc").write_text("not json at all", encoding="utf-8")
    for resp in (
        c.get("/secrets/demo", headers=_auth(SUB_RAW)),
        c.post("/secrets/demo", json={"values": {"B": "2"}}, headers=_auth(SUB_RAW)),
        c.delete("/secrets/demo/A", headers=_auth(SUB_RAW)),
    ):
        assert resp.status_code == 500
        assert resp.json()["code"] == "secret.decrypt_failed"


# --- shared scope (P8) ---------------------------------------------------------


def test_shared_write_is_admin_only(tmp_path):
    c = _client(tmp_path)
    resp = c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    ok = c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(ADMIN_RAW))
    assert ok.status_code == 200
    assert ok.json() == {"service": "shared", "keys": ["K"]}


def test_shared_delete_is_admin_only(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(ADMIN_RAW))
    assert c.delete("/secrets/shared/K", headers=_auth(SUB_RAW)).status_code == 403
    assert c.delete("/secrets/shared", headers=_auth(SUB_RAW)).status_code == 403
    assert c.delete("/secrets/shared/K", headers=_auth(ADMIN_RAW)).status_code == 200
    assert c.delete("/secrets/shared", headers=_auth(ADMIN_RAW)).json()["deleted"] is False


def test_shared_get_open_to_readonly_names_only(tmp_path):
    c = _client(tmp_path)
    c.post("/secrets/shared", json={"values": {"OPENAI_KEY": "sk-live"}}, headers=_auth(ADMIN_RAW))
    got = c.get("/secrets/shared", headers=_auth(RO_RAW))
    assert got.status_code == 200
    assert got.json() == {"service": "shared", "keys": ["OPENAI_KEY"]}
    assert "sk-live" not in got.text


def test_shared_skips_service_row_check(tmp_path):
    # 'shared' is a scope, not a service: no 404 even with zero service rows.
    c = _client(tmp_path, _queries_no_service())
    assert (
        c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(ADMIN_RAW)).status_code
        == 200
    )
    assert c.get("/secrets/shared", headers=_auth(ADMIN_RAW)).status_code == 200


def test_shared_stored_under_internal_name(tmp_path):
    # The user-facing 'shared' scope maps to _shared.enc — a name no service
    # file can ever take (DNS labels cannot start with '_').
    c = _client(tmp_path)
    c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(ADMIN_RAW))
    assert (tmp_path / "secrets" / "_shared.enc").is_file()
    assert not (tmp_path / "secrets" / "shared.enc").exists()


def test_legacy_shared_service_blocks_scope_ops(tmp_path):
    # Startup guard: a pre-P8 service literally named 'shared' disables the
    # scope entirely (its secrets are never reinterpreted as shared).
    app = _app(tmp_path, _queries())
    app.state.shared_scope_blocked = True
    c = TestClient(app, raise_server_exceptions=False)
    for resp in (
        c.post("/secrets/shared", json={"values": {"K": "v"}}, headers=_auth(ADMIN_RAW)),
        c.get("/secrets/shared", headers=_auth(RO_RAW)),
        c.delete("/secrets/shared", headers=_auth(ADMIN_RAW)),
        c.delete("/secrets/shared/K", headers=_auth(ADMIN_RAW)),
    ):
        assert resp.status_code == 409
        assert resp.json()["code"] == "secret.shared_unavailable"


def test_shared_audit_masks_values_and_targets_shared(tmp_path):
    q = _queries()
    c = _client(tmp_path, q, with_audit=True)
    resp = c.post(
        "/secrets/shared", json={"values": {"OPENAI_KEY": "sk-live"}}, headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    kwargs = q.insert_audit_log.call_args.kwargs
    assert kwargs["action"] == "secret.set"
    assert kwargs["target_id"] == "shared"  # target disambiguates the scope
    logged = json.dumps(kwargs)
    assert "sk-live" not in logged
    assert "OPENAI_KEY" in logged


def test_no_values_field_in_any_secrets_response(tmp_path):
    # Guard for the idempotency body cache: no secrets response ever carries a
    # 'values' field (names only), for services and the shared scope alike.
    c = _client(tmp_path)
    for resp in (
        c.post("/secrets/demo", json={"values": {"A": "1"}}, headers=_auth(SUB_RAW)),
        c.get("/secrets/demo", headers=_auth(SUB_RAW)),
        c.post("/secrets/shared", json={"values": {"B": "2"}}, headers=_auth(ADMIN_RAW)),
        c.get("/secrets/shared", headers=_auth(ADMIN_RAW)),
    ):
        assert resp.status_code == 200
        assert "values" not in resp.json()


# --- reserved service names (P8) ------------------------------------------------


def _services_app(tmp_path) -> FastAPI:
    """Minimal harness over the services + deploy routers (reserved-name guard
    fires before any runtime/upload state is touched)."""
    from nerdit.daemon.routes.deploy import router as deploy_router
    from nerdit.daemon.routes.services import router as services_router

    queries = _queries()
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)
    app.include_router(deploy_router)
    app.state.queries = queries
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


@pytest.mark.parametrize("name", ["shared", "rotate-key"])
def test_reserved_names_rejected_on_service_create(tmp_path, name):
    c = TestClient(_services_app(tmp_path), raise_server_exceptions=False)
    resp = c.post(
        "/services", json={"name": name, "image": "nerdit-runtime:0.1"}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"


@pytest.mark.parametrize("name", ["shared", "rotate-key"])
def test_reserved_names_rejected_on_deploy(tmp_path, name):
    c = TestClient(_services_app(tmp_path), raise_server_exceptions=False)
    resp = c.post(
        "/deploy",
        data={"name": name},
        files={"archive": ("app.zip", b"not-a-zip")},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"


# --- Idempotency-Key replay + no values in the pinned body (real DB) -----------


async def _make_db_env(tmp_path) -> tuple[FastAPI, Database, Queries]:
    """Full middleware stack over a real in-memory DB (test_idempotency pattern)."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(secrets_router)
    app.state.queries = queries
    app.state.secret_manager = SecretManager(tmp_path / "secrets")

    # inner → outer: Idempotency (innermost), Audit, Auth, RequestId.
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, db, queries


def _async_client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_rotate_key_idempotent_replay(tmp_path):
    app, db, _queries_db = await _make_db_env(tmp_path)
    try:
        async with _async_client(app) as client:
            # Seed one shared secret so the rotation rewrites a real file.
            await client.post(
                "/secrets/shared",
                json={"values": {"K": "v"}},
                headers={**_auth(LEGACY), "Idempotency-Key": "K-seed"},
            )
            headers = {**_auth(LEGACY), "Idempotency-Key": "K-rotate"}
            first = await client.post("/secrets/rotate-key", headers=headers)
            assert first.status_code == 200
            assert first.headers.get("Idempotent-Replay") is None
            assert first.json() == {"services_rewritten": 1}

            second = await client.post("/secrets/rotate-key", headers=headers)
            assert second.status_code == 200
            assert second.headers.get("Idempotent-Replay") == "true"
            # The replay returns the pinned body — no second rotation ran.
            assert second.json() == first.json()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_secret_values_never_reach_audit_or_idempotency_stores(tmp_path):
    app, db, _queries_db = await _make_db_env(tmp_path)
    try:
        async with _async_client(app) as client:
            resp = await client.post(
                "/secrets/shared",
                json={"values": {"OPENAI_KEY": "sk-live-secret"}},
                headers={**_auth(LEGACY), "Idempotency-Key": "K-shared"},
            )
            assert resp.status_code == 200
        for table, column in (
            ("audit_log", "params_redacted"),
            ("idempotency_keys", "response_body"),
        ):
            cursor = await db.conn.execute(f"SELECT {column} FROM {table}")  # noqa: S608
            rows = [row[0] or "" for row in await cursor.fetchall()]
            assert rows, f"expected at least one {table} row"
            assert all("sk-live-secret" not in row for row in rows)
    finally:
        await db.close()
