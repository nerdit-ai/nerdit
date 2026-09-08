"""Test cloud GitHub-token pushes through the real middleware stack.

Require the tunnel principal plus github-token control header; all other
combinations return the same 403. Tokens reach only the manager, never responses,
audit rows, logs or validation echoes; repository names also stay out of rows and
responses. Audit effective changes with installation ID, expiry and repo count.
Keep the idempotency exemption route-scoped. Test expiry/order in test_link_manager.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.link.manager import ENTITLEMENT_MAX_FUTURE_SKEW_S, GithubTokenUpdate
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import CLOUD_CONTROL_HEADER, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import router as link_router
from nerdit.db.models import ApiToken, TokenRole

NODE_ID = "00000000-0000-4000-8000-00000000000b"
LINK_PRINCIPAL_ID = f"link:{NODE_ID}"
CAPABILITY = "live-capability-token-not-a-secret"  # noqa: S105 - a test literal
ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SUBMITTER_TOKEN = "scoped-raw"  # noqa: S105 - a test literal
GITHUB_TOKEN = "ghs_pushed_installation_token_value"  # noqa: S105 - a test literal

PATH = "/api/link/github-token"
CONTROL = {CLOUD_CONTROL_HEADER: "github-token"}
_TUNNEL = {"Authorization": f"Bearer {CAPABILITY}", **CONTROL}
_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", **CONTROL}
_SUBMITTER = {"Authorization": f"Bearer {SUBMITTER_TOKEN}", **CONTROL}

_EXPIRES = datetime(2026, 1, 1, 13, 0, tzinfo=UTC)


class FakeLinkManager:
    """The things this route touches, and nothing else."""

    def __init__(self, *, token: str | None = CAPABILITY) -> None:
        self._token = token
        self.clock: datetime | None = None
        self.calls: list[dict[str, Any]] = []
        self.update = GithubTokenUpdate(
            applied=True, changed=True, installation_id=11, expires_at=_EXPIRES, repos_count=2
        )

    def now(self) -> datetime:
        return self.clock if self.clock is not None else datetime.now(UTC)

    def validate_capability(self, token: str, role: str) -> bool:
        return self._token is not None and token == self._token and role == "submitter"

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(node_id=NODE_ID)

    def set_github_token(self, **kwargs: Any) -> GithubTokenUpdate:
        self.calls.append(kwargs)
        return self.update


_NO_MANAGER = object()

_SCOPED_ROW = ApiToken(
    id="tok-1",
    name="ci-bot",
    role=TokenRole.submitter,
    token_hash=hash_token(SUBMITTER_TOKEN),
    max_gpus=1,
    max_concurrent_jobs=1,
)


def _queries() -> AsyncMock:
    queries = AsyncMock()

    async def by_hash(token_hash: str) -> ApiToken | None:
        return _SCOPED_ROW if token_hash == hash_token(SUBMITTER_TOKEN) else None

    queries.get_api_token_by_hash = AsyncMock(side_effect=by_hash)
    queries.insert_audit_log = AsyncMock()
    queries.touch_api_token = AsyncMock()
    return queries


class Harness(SimpleNamespace):
    client: TestClient
    queries: AsyncMock
    manager: FakeLinkManager | None
    app: FastAPI

    def audit_rows(self) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.queries.insert_audit_log.await_args_list]

    def token_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.audit_rows() if row["action"] == "link.github_token"]


def _build(
    *,
    link_manager: object = _NO_MANAGER,
    require_idempotency_key: bool = False,
) -> Harness:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(link_router)
    app.include_router(api)

    manager = FakeLinkManager() if link_manager is _NO_MANAGER else link_manager
    app.state.link_manager = manager
    app.state.settings = SimpleNamespace(daemon=SimpleNamespace(host="127.0.0.1", port=9321))

    queries = _queries()
    app.add_middleware(
        IdempotencyMiddleware,
        get_queries=lambda: queries,
        require_idempotency_key=require_idempotency_key,
    )
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=ADMIN_TOKEN, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    return Harness(
        client=TestClient(app, raise_server_exceptions=False),
        queries=queries,
        manager=manager if isinstance(manager, FakeLinkManager) else None,
        app=app,
    )


def _body(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC)
    body: dict[str, Any] = {
        "installation_id": 11,
        "token": GITHUB_TOKEN,
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "issued_at": now.isoformat(),
        "repos": ["acme/web", "acme/api"],
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# the principal gate
# ---------------------------------------------------------------------------


def test_the_tunnel_principal_with_the_control_header_may_write_the_mirror() -> None:
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 200
    assert response.json() == {
        "applied": True,
        "installation_id": 11,
        "expires_at": "2026-01-01T13:00:00Z",
        "repos_count": 2,
    }
    assert h.manager is not None
    assert len(h.manager.calls) == 1
    call = h.manager.calls[0]
    assert call["installation_id"] == 11
    assert call["token"] == GITHUB_TOKEN
    assert call["repos"] == ["acme/web", "acme/api"]
    assert call["issued_at"].tzinfo is not None
    assert call["expires_at"].tzinfo is not None


def test_a_local_admin_token_is_refused_even_with_the_control_header() -> None:
    """No node holds a GitHub credential the cloud did not just mint (D-GH-1)."""
    h = _build()
    response = h.client.put(PATH, headers=_ADMIN, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.manager is not None
    assert h.manager.calls == []


def test_a_scoped_submitter_token_is_refused_too() -> None:
    h = _build()
    response = h.client.put(PATH, headers=_SUBMITTER, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


def test_the_tunnel_principal_without_the_control_header_is_refused() -> None:
    h = _build()
    response = h.client.put(PATH, headers={"Authorization": f"Bearer {CAPABILITY}"}, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.manager is not None
    assert h.manager.calls == []


@pytest.mark.parametrize("value", ["", "other", "entitlement", "GitHub-Token", "github-token "])
def test_the_control_header_must_carry_the_exact_value(value: str) -> None:
    """Exact match — and the ENTITLEMENT value does not open this route, so a
    pusher framed for one cannot be replayed at the other."""
    h = _build()
    response = h.client.put(
        PATH,
        headers={"Authorization": f"Bearer {CAPABILITY}", CLOUD_CONTROL_HEADER: value},
        json=_body(),
    )
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


def test_both_refusals_carry_the_same_code_so_neither_is_an_oracle() -> None:
    h = _build()
    wrong_principal = h.client.put(PATH, headers=_ADMIN, json=_body()).json()
    wrong_header = h.client.put(
        PATH, headers={"Authorization": f"Bearer {CAPABILITY}"}, json=_body()
    ).json()
    assert wrong_principal["code"] == wrong_header["code"]
    assert wrong_principal["message"] == wrong_header["message"]


def test_a_stale_capability_never_reaches_the_route() -> None:
    h = _build(link_manager=FakeLinkManager(token=None))
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] == "invalid_token"


def test_a_daemon_with_no_link_manager_never_grants_the_cloud_principal() -> None:
    h = _build(link_manager=None)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] in {"invalid_token", "link.cloud_principal_required"}


# ---------------------------------------------------------------------------
# body validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
def test_a_naive_stamp_is_refused_before_the_route_runs(field: str) -> None:
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(**{field: "2026-01-01T12:00:00"}))
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert h.manager is not None
    assert h.manager.calls == []


def test_a_mixed_case_repo_is_normalised() -> None:
    """Lower-case ``owner/name`` only (D-GH-6): a mixed-case entry is
    normalised rather than refused."""
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(repos=["Acme/Web"]))
    assert response.status_code == 200
    assert h.manager is not None
    assert h.manager.calls[0]["repos"] == ["acme/web"]


@pytest.mark.parametrize(
    "bad",
    [["acme"], ["acme/web/extra"], ["https://github.com/acme/web"], [""]],
)
def test_a_malformed_repo_is_dropped_not_a_422(bad: list[str]) -> None:
    """(D4) One malformed entry must NOT 422 the whole push — that would leave
    nothing mirrored, GitWatch quiet and the doctor check skipped. It is
    dropped; valid siblings survive. Fail-closed (fewer resolvable repos) is
    the safe direction."""
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(repos=[*bad, "keep/me"]))
    assert response.status_code == 200
    assert h.manager is not None
    assert h.manager.calls[0]["repos"] == ["keep/me"]


def test_an_all_malformed_push_degrades_to_an_empty_repo_set() -> None:
    """(D4) Even a push whose every entry is malformed applies with an empty
    set rather than 422-ing — the token is still mirrored, it simply resolves
    no repo."""
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(repos=["not a repo"]))
    assert response.status_code == 200
    assert h.manager is not None
    assert h.manager.calls[0]["repos"] == []


def test_an_over_cap_repo_list_truncates_and_still_applies() -> None:
    """(D4) An org-wide App with more repos than the cap must not 422; it
    mirrors the first ``_REPOS_CAP`` and the push still applies."""
    from nerdit.daemon.schemas.link import _REPOS_CAP  # noqa: PLC0415

    h = _build()
    repos = [f"org/r{i}" for i in range(_REPOS_CAP + 25)]
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(repos=repos))
    assert response.status_code == 200
    assert h.manager is not None
    assert h.manager.calls[0]["repos"] == repos[:_REPOS_CAP]


def test_an_unknown_key_is_refused_rather_than_dropped() -> None:
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json={**_body(), "installation": 1})
    assert response.status_code == 422


_PINNED_NOW = datetime(2031, 6, 1, 12, 0, tzinfo=UTC)


def test_a_stamp_beyond_the_allowed_skew_is_refused() -> None:
    h = _build()
    assert h.manager is not None
    h.manager.clock = _PINNED_NOW
    future = _PINNED_NOW + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S + 1)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(issued_at=future.isoformat()))
    assert response.status_code == 422
    assert h.manager.calls == []


def test_a_stamp_inside_the_allowed_skew_is_accepted_on_the_managers_clock() -> None:
    h = _build()
    assert h.manager is not None
    h.manager.clock = _PINNED_NOW
    near = _PINNED_NOW + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S - 30)
    response = h.client.put(
        PATH,
        headers=_TUNNEL,
        json=_body(issued_at=near.isoformat(), expires_at=(near + timedelta(hours=1)).isoformat()),
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# audit: change-only; id + expiry + COUNT, never the token or the names
# ---------------------------------------------------------------------------


def test_a_change_writes_one_row_carrying_id_expiry_and_count() -> None:
    h = _build()
    h.client.put(PATH, headers=_TUNNEL, json=_body())

    rows = h.token_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["result"] == "ok"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    assert row["principal_id"] == LINK_PRINCIPAL_ID
    assert row["principal_role"] == "submitter"
    assert row["params_redacted"] == (
        '{"installation_id": 11, "expires_at": "2026-01-01T13:00:00+00:00", "repos_count": 2}'
    )


def test_an_unchanged_push_writes_no_row() -> None:
    h = _build()
    assert h.manager is not None
    h.manager.update = GithubTokenUpdate(
        applied=True, changed=False, installation_id=11, expires_at=_EXPIRES, repos_count=2
    )
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 200
    assert response.json()["applied"] is True
    assert h.token_rows() == []


def test_an_out_of_order_push_is_a_no_op_reporting_applied_false() -> None:
    h = _build()
    assert h.manager is not None
    h.manager.update = GithubTokenUpdate(
        applied=False, changed=False, installation_id=11, expires_at=_EXPIRES, repos_count=1
    )
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 200
    assert response.json()["applied"] is False
    assert response.json()["repos_count"] == 1
    assert h.token_rows() == []


def test_a_denial_is_always_recorded_even_though_the_route_may_skip() -> None:
    h = _build()
    h.client.put(PATH, headers=_ADMIN, json=_body())
    rows = h.token_rows()
    assert len(rows) == 1
    assert rows[0]["result"] == "denied"
    assert rows[0]["status_code"] == 403
    assert rows[0]["params_redacted"] is None


# ---------------------------------------------------------------------------
# nothing secret travels
# ---------------------------------------------------------------------------


def test_the_token_and_the_repo_names_reach_no_row_body_or_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One grep over every durable surface, for the accepted push, the two
    refusals and a 422 on a SIBLING field (whose echoed ``input`` is the raw
    request dict, token included, unless the handler scrubs it)."""
    h = _build()
    with caplog.at_level(logging.DEBUG):
        bodies = [
            h.client.put(PATH, headers=_TUNNEL, json=_body()).text,
            h.client.put(PATH, headers=_ADMIN, json=_body()).text,
            h.client.put(
                PATH, headers={"Authorization": f"Bearer {CAPABILITY}"}, json=_body()
            ).text,
            h.client.put(PATH, headers=_TUNNEL, json=_body(repos=["not a repo"])).text,
            h.client.put(PATH, headers=_TUNNEL, json=_body(installation_id=0)).text,
        ]

    haystack = "\n".join(
        [
            *bodies,
            *(record.getMessage() for record in caplog.records),
            *(str(row) for row in h.audit_rows()),
        ]
    )
    assert GITHUB_TOKEN not in haystack
    assert CAPABILITY not in haystack
    assert ADMIN_TOKEN not in haystack
    assert "acme/web" not in haystack
    assert "x-nerdit-cloud-control: github-token" not in haystack


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_a_key_less_push_survives_require_idempotency_key() -> None:
    h = _build(require_idempotency_key=True)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 200
    assert response.json()["applied"] is True


def test_the_exemption_does_not_leak_to_the_other_link_routes() -> None:
    h = _build(require_idempotency_key=True)
    response = h.client.post(
        "/api/link/refresh",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"api_url": "https://app.example.test"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
