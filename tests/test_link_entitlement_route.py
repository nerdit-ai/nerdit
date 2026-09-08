"""Test cloud entitlement pushes through real auth/audit/idempotency middleware.

Require both the tunnel principal and entitlement control header; every denied
combination returns the same code. Audit only the boolean, never capability or
header values. Effective changes alone produce success audit rows and durable
events; always record denials. Keep this route exempt from Idempotency-Key.
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

from nerdit.core.link.manager import (
    ENTITLEMENT_MAX_FUTURE_SKEW_S,
    ENTITLEMENT_TTL_S,
    EntitlementUpdate,
)
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import CLOUD_CONTROL_HEADER, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import router as link_router
from nerdit.db.models import ApiToken, TokenRole

NODE_ID = "00000000-0000-4000-8000-00000000000a"
LINK_PRINCIPAL_ID = f"link:{NODE_ID}"
CAPABILITY = "live-capability-token-not-a-secret"  # noqa: S105 - a test literal
ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SUBMITTER_TOKEN = "scoped-raw"  # noqa: S105 - a test literal

PATH = "/api/link/entitlement"
CONTROL = {CLOUD_CONTROL_HEADER: "entitlement"}
_TUNNEL = {"Authorization": f"Bearer {CAPABILITY}", **CONTROL}
_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", **CONTROL}
_SUBMITTER = {"Authorization": f"Bearer {SUBMITTER_TOKEN}", **CONTROL}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class FakeLinkManager:
    """The three things this route touches, and nothing else.

    Deliberately NOT the real manager: the TTL/ordering matrix is pinned on the
    real one in ``tests/test_link_manager.py``, and mixing the two would make a
    route regression look like a seam regression.
    """

    def __init__(self, *, token: str | None = CAPABILITY) -> None:
        self._token = token
        #: (review round 1) The seam's clock, which the route's future-skew
        #: check must read too. ``None`` = follow the wall clock, so every test
        #: that does not care about skew keeps building stamps from
        #: ``datetime.now(UTC)``; a fixed value proves the route and the seam
        #: share one clock rather than two that happen to agree.
        self.clock: datetime | None = None
        self.calls: list[tuple[bool, datetime]] = []
        self.events: list[bool] = []
        self.update = EntitlementUpdate(
            applied=True,
            changed=True,
            effective=True,
            received_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
            expires_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
            + timedelta(seconds=ENTITLEMENT_TTL_S),
        )

    def now(self) -> datetime:
        return self.clock if self.clock is not None else datetime.now(UTC)

    def validate_capability(self, token: str, role: str) -> bool:
        return self._token is not None and token == self._token and role == "submitter"

    def status(self) -> SimpleNamespace:
        return SimpleNamespace(node_id=NODE_ID)

    def set_hosted_public_entitled(self, value: bool, issued_at: datetime) -> EntitlementUpdate:
        self.calls.append((value, issued_at))
        return self.update

    async def record_entitlement_change(self, value: bool) -> None:
        self.events.append(value)


#: Sentinel for "build the app with no link manager at all" — distinct from
#: ``None``, which the builder would otherwise read as "use the default".
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
    """Hash-faithful on purpose: an unknown bearer must resolve to no row.

    A blanket ``return_value`` would hand every stale capability a valid scoped
    principal, and the "a bearer that is not the live capability is refused"
    cases would pass for the wrong reason.
    """
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

    def entitlement_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.audit_rows() if row["action"] == "link.entitlement"]


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
    # Inner→outer, matching ``appfactory``: Auth → Audit → Idempotency.
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


def _body(*, entitled: bool = True, issued_at: datetime | None = None) -> dict[str, Any]:
    stamp = issued_at if issued_at is not None else datetime.now(UTC)
    return {"hosted_public_entitled": entitled, "issued_at": stamp.isoformat()}


# ---------------------------------------------------------------------------
# the principal gate (D-P32-2)
# ---------------------------------------------------------------------------


def test_the_tunnel_principal_with_the_control_header_may_write_the_mirror() -> None:
    h = _build()
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 200
    body = response.json()
    assert body["hosted_public_entitled"] is True
    assert body["applied"] is True
    assert body["changed"] is True
    assert body["received_at"] == "2026-01-01T12:00:00Z"
    assert body["expires_at"] == "2026-01-02T12:00:00Z"
    assert h.manager is not None
    assert len(h.manager.calls) == 1


def test_a_local_admin_token_is_refused_even_with_the_control_header() -> None:
    """The operator does not set entitlement — that is the whole point (D-P32-6).

    An admin token is the strongest credential the daemon has, so if the route
    were merely role-gated an operator could grant themselves a plan the cloud
    never sold them.
    """
    h = _build()
    response = h.client.put(PATH, headers=_ADMIN, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.manager is not None
    assert h.manager.calls == []


def test_a_scoped_submitter_token_is_refused_too() -> None:
    """Same role as the tunnel principal, different provenance — still no."""
    h = _build()
    response = h.client.put(PATH, headers=_SUBMITTER, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"


def test_the_tunnel_principal_without_the_control_header_is_refused() -> None:
    """A proxied console/MCP/hosted request rides the same bearer.

    The header is what separates "the cloud's own code framed this stream" from
    "a user's browser did"; without it the request is indistinguishable from
    one an owner's console session could make, so it is refused.
    """
    h = _build()
    response = h.client.put(PATH, headers={"Authorization": f"Bearer {CAPABILITY}"}, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "link.cloud_principal_required"
    assert h.manager is not None
    assert h.manager.calls == []


@pytest.mark.parametrize("value", ["", "other", "Entitlement ", "entitlement,entitlement"])
def test_the_control_header_must_carry_the_exact_value(value: str) -> None:
    """Exact match, not a prefix or a case-fold — a header list is opaque
    payload on the wire, so anything looser is a parser to get wrong."""
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
        PATH,
        headers={"Authorization": f"Bearer {CAPABILITY}"},
        json=_body(),
    ).json()

    assert wrong_principal["code"] == wrong_header["code"]
    assert wrong_principal["message"] == wrong_header["message"]


def test_a_stale_capability_never_reaches_the_route() -> None:
    """Between sockets the manager answers ``False``; the AUTH middleware
    refuses first, so the route never runs and never sees the body."""
    h = _build(link_manager=FakeLinkManager(token=None))
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] == "invalid_token"


def test_a_daemon_with_no_link_manager_never_grants_the_cloud_principal() -> None:
    """Unreachable in the real app — the manager IS what validates the bearer —
    so the guard must not become an oracle for daemon internals. Whichever
    layer answers, the answer is a 403 and the body is not read."""
    h = _build(link_manager=None)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 403
    assert response.json()["code"] in {"invalid_token", "link.cloud_principal_required"}


# ---------------------------------------------------------------------------
# body validation
# ---------------------------------------------------------------------------


def test_a_naive_issued_at_is_refused_before_the_route_runs() -> None:
    """It is the ORDERING key; a stamp without a zone cannot order anything."""
    h = _build()
    response = h.client.put(
        PATH,
        headers=_TUNNEL,
        json={"hosted_public_entitled": True, "issued_at": "2026-01-01T12:00:00"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert h.manager is not None
    assert h.manager.calls == []


def test_an_unknown_key_is_refused_rather_than_dropped() -> None:
    h = _build()
    response = h.client.put(
        PATH,
        headers=_TUNNEL,
        json={**_body(), "hosted_disabled": True},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


#: An arbitrary fixed instant, far from any wall clock, used to prove the skew
#: check reads the MANAGER's clock and not ``datetime.now(UTC)`` (review round
#: 1). A stamp built from it would be ~decades in the future for the wall clock,
#: so a route that consulted the wall clock would refuse the "inside the skew"
#: case below.
_PINNED_NOW = datetime(2031, 6, 1, 12, 0, tzinfo=UTC)


def test_a_stamp_beyond_the_allowed_skew_is_refused() -> None:
    """Accepting it would pin the mirror against every later, correctly
    ordered push — the failure mode is silent and lasts 24 h."""
    h = _build()
    assert h.manager is not None
    h.manager.clock = _PINNED_NOW
    future = _PINNED_NOW + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S + 1)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(issued_at=future))

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert h.manager.calls == []


def test_a_stamp_inside_the_allowed_skew_is_accepted() -> None:
    """Two machines' clocks drift; ordinary NTP skew is not a malformed push.

    Also the load-bearing half of the one-clock pin: ``_PINNED_NOW`` is years
    ahead of the wall clock, so this stamp is "inside the skew" only for a route
    that reads :meth:`LinkManager.now`.
    """
    h = _build()
    assert h.manager is not None
    h.manager.clock = _PINNED_NOW
    near_future = _PINNED_NOW + timedelta(seconds=ENTITLEMENT_MAX_FUTURE_SKEW_S - 30)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body(issued_at=near_future))

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# audit: change-only, bool-only (D-P32-5)
# ---------------------------------------------------------------------------


def test_a_change_writes_one_row_carrying_only_the_bool() -> None:
    h = _build()
    h.client.put(PATH, headers=_TUNNEL, json=_body())

    rows = h.entitlement_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["result"] == "ok"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    assert row["principal_id"] == LINK_PRINCIPAL_ID
    assert row["principal_role"] == "submitter"
    assert row["params_redacted"] == '{"hosted_public_entitled": true}'


def test_a_change_publishes_one_durable_event() -> None:
    h = _build()
    h.client.put(PATH, headers=_TUNNEL, json=_body())
    assert h.manager is not None
    assert h.manager.events == [True]


def test_an_unchanged_push_writes_no_row_and_no_event() -> None:
    """The cloud re-asserts every few minutes; a row per re-assert would bury
    the log in noise that carries no information."""
    h = _build()
    assert h.manager is not None
    h.manager.update = EntitlementUpdate(
        applied=True,
        changed=False,
        effective=True,
        received_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        expires_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )

    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 200
    assert response.json()["changed"] is False
    assert h.entitlement_rows() == []
    assert h.manager.events == []


def test_an_out_of_order_push_writes_no_row_and_reports_applied_false() -> None:
    h = _build()
    assert h.manager is not None
    h.manager.update = EntitlementUpdate(
        applied=False,
        changed=False,
        effective=True,
        received_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        expires_at=datetime(2026, 1, 2, 12, 0, tzinfo=UTC),
    )

    body = h.client.put(PATH, headers=_TUNNEL, json=_body(entitled=False)).json()

    assert body["applied"] is False
    assert body["hosted_public_entitled"] is True
    assert h.entitlement_rows() == []


def test_a_denial_is_always_recorded_even_though_the_route_may_skip() -> None:
    """The skip is honoured for 2xx ONLY. A 403 here means something tried to
    write the mirror from outside the tunnel — the row that matters most."""
    h = _build()
    h.client.put(PATH, headers=_ADMIN, json=_body())

    rows = h.entitlement_rows()
    assert len(rows) == 1
    assert rows[0]["result"] == "denied"
    assert rows[0]["status_code"] == 403
    assert rows[0]["params_redacted"] is None


def test_a_validation_failure_is_recorded_as_an_error() -> None:
    h = _build()
    h.client.put(
        PATH,
        headers=_TUNNEL,
        json={"hosted_public_entitled": True, "issued_at": "2026-01-01T12:00:00"},
    )

    rows = h.entitlement_rows()
    assert len(rows) == 1
    assert rows[0]["result"] == "error"


# ---------------------------------------------------------------------------
# nothing secret travels
# ---------------------------------------------------------------------------


def test_no_credential_or_carrier_value_reaches_a_row_a_body_or_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One sweep over every durable surface, for the accepted push and for the
    two refusals — the failure this guards is a *silent* leak."""
    h = _build()
    with caplog.at_level(logging.DEBUG):
        bodies = [
            h.client.put(PATH, headers=_TUNNEL, json=_body()).text,
            h.client.put(PATH, headers=_ADMIN, json=_body()).text,
            h.client.put(
                PATH, headers={"Authorization": f"Bearer {CAPABILITY}"}, json=_body()
            ).text,
        ]

    haystack = "\n".join(
        [
            *bodies,
            *(record.getMessage() for record in caplog.records),
            *(str(row) for row in h.audit_rows()),
        ]
    )
    assert CAPABILITY not in haystack
    assert ADMIN_TOKEN not in haystack
    # The carrier's NAME may be discussed in a docstring; its value must not
    # ride along into a row or a log line as if it were data.
    assert "x-nerdit-cloud-control: entitlement" not in haystack


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_the_push_needs_no_idempotency_key() -> None:
    h = _build()
    assert h.client.put(PATH, headers=_TUNNEL, json=_body()).status_code == 200


def test_a_key_less_push_survives_require_idempotency_key() -> None:
    """``[security].require_idempotency_key = true`` 400s a key-less mutation.

    The push is a machine write that is idempotent BY VALUE, so it joins the
    MCP mount in the exemption set rather than forcing the cloud to mint keys
    for a bool it re-asserts on a timer.
    """
    h = _build(require_idempotency_key=True)
    response = h.client.put(PATH, headers=_TUNNEL, json=_body())

    assert response.status_code == 200
    assert response.json()["applied"] is True


def test_the_exemption_does_not_leak_to_the_other_link_routes() -> None:
    """A route-scoped bypass that widened silently would remove replay
    protection from the claim, which commits config and burns a link code."""
    h = _build(require_idempotency_key=True)
    response = h.client.post(
        "/api/link/refresh",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        json={"api_url": "https://app.example.test"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
