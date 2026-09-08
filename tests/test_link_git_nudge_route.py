"""Test cloud git nudges with real middleware and a recording controller double.

Require the tunnel principal plus git-nudge control header; other combinations
return the same 403. Require an in-route Idempotency-Key so redeliveries replay.
Normalize owner/name, short ref and full lowercase SHA before dispatch; invalid
bodies return 422 without calling the controller. Record public facts in one
gitwatch.nudge audit row. Controller matching/deduplication lives in test_gitwatch.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.gitwatch import NudgeResult
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import CLOUD_CONTROL_HEADER, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import router as link_router
from nerdit.db.models import ApiToken, TokenRole

NODE_ID = "00000000-0000-4000-8000-00000000000c"
CAPABILITY = "live-capability-token-not-a-secret"  # noqa: S105 - a test literal
ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SUBMITTER_TOKEN = "scoped-raw"  # noqa: S105 - a test literal
SHA = "0123456789abcdef0123456789abcdef01234567"

PATH = "/api/link/git-nudge"
CONTROL = {CLOUD_CONTROL_HEADER: "git-nudge"}
_KEY = {"Idempotency-Key": "gh-delivery-1"}
_TUNNEL = {"Authorization": f"Bearer {CAPABILITY}", **CONTROL, **_KEY}
_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", **CONTROL, **_KEY}
_SUBMITTER = {"Authorization": f"Bearer {SUBMITTER_TOKEN}", **CONTROL, **_KEY}


class FakeLinkManager:
    def __init__(self, *, token: str | None = CAPABILITY) -> None:
        self._token = token

    def validate_capability(self, token: str, role: str) -> bool:
        return self._token is not None and token == self._token and role == "submitter"

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(node_id=NODE_ID)


class FakeGitWatch:
    """Records what the route hands over; answers a canned result."""

    def __init__(self, result: NudgeResult | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.result = result or NudgeResult(matched=["web"], ignored=["docs"], deduped=[])

    async def nudge(self, repo: str, ref: str, sha: str) -> NudgeResult:
        self.calls.append((repo, ref, sha))
        return self.result


_NO_MANAGER = object()
_NO_GITWATCH = object()

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
    gitwatch: FakeGitWatch | None

    def audit_rows(self) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.queries.insert_audit_log.await_args_list]

    def nudge_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.audit_rows() if row["action"] == "gitwatch.nudge"]


def _build(
    *,
    link_manager: object = _NO_MANAGER,
    gitwatch: object = _NO_GITWATCH,
    require_idempotency_key: bool = False,
) -> Harness:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(link_router)
    app.include_router(api)

    app.state.link_manager = FakeLinkManager() if link_manager is _NO_MANAGER else link_manager
    watch = FakeGitWatch() if gitwatch is _NO_GITWATCH else gitwatch
    if watch is not None:
        app.state.gitwatch = watch
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
        gitwatch=watch if isinstance(watch, FakeGitWatch) else None,
    )


def _body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"repo": "acme/web", "repo_id": 42, "ref": "main", "sha": SHA}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# the principal gate
# ---------------------------------------------------------------------------


def test_the_tunnel_principal_with_the_control_header_may_nudge() -> None:
    h = _build()
    response = h.client.post(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 202
    assert response.json() == {"matched": ["web"], "ignored": ["docs"], "deduped": []}
    assert h.gitwatch is not None
    assert h.gitwatch.calls == [("acme/web", "main", SHA)]


def test_a_local_admin_token_is_refused_even_with_the_control_header() -> None:
    h = _build()
    response = h.client.post(PATH, headers=_ADMIN, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.gitwatch is not None
    assert h.gitwatch.calls == []


def test_a_scoped_submitter_token_is_refused_too() -> None:
    h = _build()
    response = h.client.post(PATH, headers=_SUBMITTER, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


def test_the_tunnel_principal_without_the_control_header_is_refused() -> None:
    h = _build()
    response = h.client.post(
        PATH, headers={"Authorization": f"Bearer {CAPABILITY}", **_KEY}, json=_body()
    )
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.gitwatch is not None
    assert h.gitwatch.calls == []


@pytest.mark.parametrize(
    "value", ["", "other", "entitlement", "github-token", "Git-Nudge", "git-nudge "]
)
def test_the_control_header_must_carry_the_exact_value(value: str) -> None:
    """Exact match — neither mirror's value opens this route, so a pusher
    framed for one cannot be replayed at another."""
    h = _build()
    response = h.client.post(
        PATH,
        headers={"Authorization": f"Bearer {CAPABILITY}", CLOUD_CONTROL_HEADER: value, **_KEY},
        json=_body(),
    )
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


def test_both_refusals_carry_the_same_code_so_neither_is_an_oracle() -> None:
    h = _build()
    wrong_principal = h.client.post(PATH, headers=_ADMIN, json=_body()).json()
    wrong_header = h.client.post(
        PATH, headers={"Authorization": f"Bearer {CAPABILITY}", **_KEY}, json=_body()
    ).json()
    assert wrong_principal["code"] == wrong_header["code"]
    assert wrong_principal["message"] == wrong_header["message"]


def test_a_stale_capability_never_reaches_the_route() -> None:
    h = _build(link_manager=FakeLinkManager(token=None))
    response = h.client.post(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] == "invalid_token"


def test_a_daemon_with_no_link_manager_never_grants_the_cloud_principal() -> None:
    h = _build(link_manager=None)
    response = h.client.post(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 403
    assert response.json()["code"] in {"invalid_token", "link.cloud_principal_required"}


def test_a_denial_is_always_recorded() -> None:
    h = _build()
    h.client.post(PATH, headers=_ADMIN, json=_body())
    (row,) = h.nudge_rows()
    assert row["result"] == "denied"


# ---------------------------------------------------------------------------
# idempotency — the ORDINARY gate, not the exemption
# ---------------------------------------------------------------------------


def test_a_key_less_nudge_is_refused_in_route() -> None:
    h = _build()
    response = h.client.post(
        PATH, headers={"Authorization": f"Bearer {CAPABILITY}", **CONTROL}, json=_body()
    )
    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    assert h.gitwatch is not None
    assert h.gitwatch.calls == []


def test_a_key_less_nudge_is_refused_by_the_middleware_when_the_operator_opted_in() -> None:
    """No exemption: ``[security].require_idempotency_key`` 400s it ahead of
    routing, exactly like every other mutating route."""
    h = _build(require_idempotency_key=True)
    response = h.client.post(
        PATH, headers={"Authorization": f"Bearer {CAPABILITY}", **CONTROL}, json=_body()
    )
    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"


def test_the_principal_gate_runs_before_the_key_gate() -> None:
    """An outsider without a key learns nothing about the key requirement."""
    h = _build()
    response = h.client.post(
        PATH, headers={"Authorization": f"Bearer {ADMIN_TOKEN}", **CONTROL}, json=_body()
    )
    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


# ---------------------------------------------------------------------------
# body normalisation / validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expect"),
    [
        ({"repo": "Acme/Web.git"}, ("acme/web", "main", SHA)),
        ({"ref": "refs/heads/release/1.x"}, ("acme/web", "release/1.x", SHA)),
        ({"sha": SHA.upper()}, ("acme/web", "main", SHA)),
        ({"repo_id": None}, ("acme/web", "main", SHA)),
    ],
)
def test_the_body_is_normalised_before_the_controller_sees_it(
    given: dict[str, Any], expect: tuple[str, str, str]
) -> None:
    h = _build()
    response = h.client.post(PATH, headers=_TUNNEL, json=_body(**given))
    assert response.status_code == 202
    assert h.gitwatch is not None
    assert h.gitwatch.calls == [expect]


@pytest.mark.parametrize(
    "bad",
    [
        {"repo": "acme"},
        {"repo": "https://github.com/acme/web"},
        {"repo": "acme/web/extra"},
        {"ref": ""},
        {"ref": "refs/heads/"},
        {"ref": "-x"},
        {"ref": "main branch"},
        {"sha": SHA[:7]},
        {"sha": "g" * 40},
        {"repo_id": 0},
        {"extra": 1},
    ],
)
def test_a_malformed_nudge_is_a_422_that_reaches_no_controller(bad: dict[str, Any]) -> None:
    h = _build()
    body = _body(**bad)
    response = h.client.post(PATH, headers=_TUNNEL, json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert h.gitwatch is not None
    assert h.gitwatch.calls == []


# ---------------------------------------------------------------------------
# outcome + audit
# ---------------------------------------------------------------------------


def test_a_daemon_without_gitwatch_answers_202_with_nothing_matched() -> None:
    """``[git]`` off or ``watch_interval_s = 0``: best-effort, never an error
    — the cloud reads the outcome off the service view (D-GH-5)."""
    h = _build(gitwatch=None)
    response = h.client.post(PATH, headers=_TUNNEL, json=_body())
    assert response.status_code == 202
    assert response.json() == {"matched": [], "ignored": [], "deduped": []}


def test_every_accepted_nudge_writes_one_row_with_the_public_facts() -> None:
    h = _build()
    h.client.post(PATH, headers=_TUNNEL, json=_body(repo="Acme/Web", ref="refs/heads/main"))
    (row,) = h.nudge_rows()
    assert row["result"] == "ok"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    params = json.loads(row["params_redacted"])
    assert params == {"repo": "acme/web", "ref": "main", "sha": SHA, "matched": ["web"]}


def test_a_replayed_delivery_id_is_a_cached_replay_not_a_second_nudge() -> None:
    """The whole point of keying by ``X-GitHub-Delivery``: a redelivery replays
    the 202 and the controller is not asked twice."""
    h = _build()
    first = h.client.post(PATH, headers=_TUNNEL, json=_body())
    assert first.status_code == 202

    # Second arrival with the same key: the middleware finds the record.
    completed = SimpleNamespace(
        method="POST",
        path=PATH,
        body_hash=None,
        state="completed",
        response_status=202,
        response_body=first.text,
        content_type="application/json",
    )
    h.queries.insert_idempotency_inprogress = AsyncMock(return_value=False)
    h.queries.get_idempotency_record = AsyncMock(return_value=completed)
    second = h.client.post(PATH, headers=_TUNNEL, json=_body())
    assert second.status_code == 202
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    assert h.gitwatch is not None
    assert len(h.gitwatch.calls) == 1
