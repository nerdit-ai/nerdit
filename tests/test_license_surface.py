"""Test daemon-owned license installation and removal with real middleware.

Never expose blobs or customer IDs in durable feeds or responses, except the
admin-only install envelope's customer ID. Invalid input returns a reasoned 422
without replacing the file, runtime state or events. Content changes refresh the
holder immediately; changing the file path requires restart. Repeated removal
returns removed=false without another event.
Use mocked queries and injected frozen test keys, which shipped daemons do not trust.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.config.settings import LicenseSettings
from nerdit.core import eventlog
from nerdit.core.eventlog import EVENT_TYPES
from nerdit.core.license import (
    LICENSE_GRACE_S,
    LicenseState,
    verify_license,
)
from nerdit.daemon.audit import AuditMiddleware, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import NO_BODY_CACHE_ACTIONS, IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.license import router as license_router
from nerdit.db.models import ApiToken, TokenRole
from tests.license_vectors import (
    GOLDEN_BLOB,
    GOLDEN_CLAIMS,
    TEST_TRUSTED_KEYS,
    sign_license,
)

ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SCOPED_TOKEN = "scoped-raw"  # noqa: S105 - a test literal

_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "idem-1"}
_SCOPED = {"Authorization": f"Bearer {SCOPED_TOKEN}", "Idempotency-Key": "idem-1"}

#: The instant every temporal case is expressed against — the golden claims
#: expire 2027-01-01, so "now" sits comfortably inside their validity.
NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
CUSTOMER = GOLDEN_CLAIMS["customer_id"]


def _clock():
    return NOW


def _blob(*, expires_at: datetime, plan: str = "pro", features: list[str] | None = None) -> str:
    claims = dict(GOLDEN_CLAIMS)
    claims["plan"] = plan
    claims["features"] = ["remote_link"] if features is None else features
    claims["expires_at"] = expires_at.isoformat()
    return sign_license(claims)


VALID_BLOB = _blob(expires_at=NOW + timedelta(days=30))
GRACE_BLOB = _blob(expires_at=NOW - timedelta(days=1))
EXPIRED_BLOB = _blob(expires_at=NOW - timedelta(seconds=LICENSE_GRACE_S + 60))


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class FakeRecorder:
    """Stands in for the process-wide :class:`EventRecorder`.

    Only :meth:`record` is used by the routes. Every call is asserted against
    :data:`EVENT_TYPES` here rather than trusted, so a typo'd type — which the
    real recorder silently refuses — fails the test instead of vanishing.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def record(self, type: str, **kwargs: Any) -> None:  # noqa: A002 - the locked field name
        assert type in EVENT_TYPES, f"{type} is not in the event vocabulary"
        self.events.append((type, kwargs))


def _queries() -> AsyncMock:
    queries = AsyncMock()
    queries.get_api_token_by_hash = AsyncMock(
        return_value=ApiToken(
            id="tok-1",
            name="ci-bot",
            role=TokenRole.submitter,
            token_hash=hash_token(SCOPED_TOKEN),
            max_gpus=1,
            max_concurrent_jobs=1,
        )
    )
    queries.insert_audit_log = AsyncMock()
    queries.touch_api_token = AsyncMock()
    return queries


class Harness(SimpleNamespace):
    """The built app plus the handles a test needs to assert against."""

    client: TestClient
    app: FastAPI
    queries: AsyncMock
    path: Path
    recorder: FakeRecorder

    def audit_rows(self) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.queries.insert_audit_log.await_args_list]

    @property
    def holder(self) -> LicenseState:
        return self.app.state.license

    def installed_text(self) -> str | None:
        return self.path.read_text() if self.path.exists() else None


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(license_router)
    app.include_router(api)

    app.state.settings = SimpleNamespace(data_dir=str(data_dir), license=LicenseSettings())
    app.state.license = LicenseState(now=_clock)
    # Inject test trust explicitly; shipped keys do not include this fixture key.
    app.state.license_trusted_keys = TEST_TRUSTED_KEYS
    app.state.link_manager = None

    queries = _queries()
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=ADMIN_TOKEN, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    recorder = FakeRecorder()
    monkeypatch.setattr(eventlog, "_recorder", recorder)

    return Harness(
        client=TestClient(app, raise_server_exceptions=False),
        app=app,
        queries=queries,
        path=data_dir / "license.jws",
        recorder=recorder,
    )


def _install(h: Harness, blob: str, headers: dict[str, str] | None = None):
    return h.client.post("/api/license", json={"blob": blob}, headers=headers or _ADMIN)


# ---------------------------------------------------------------------------
# install — the happy path
# ---------------------------------------------------------------------------


def test_install_persists_verifies_and_refreshes_the_running_daemon(harness: Harness) -> None:
    response = _install(harness, VALID_BLOB)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "valid"
    assert body["plan"] == "pro"
    assert body["features"] == ["remote_link"]
    assert body["lid"] == GOLDEN_CLAIMS["lid"]
    # Admin-gated route ⇒ the customer id rides the envelope (unlike
    # /capabilities, where it is omitted for non-admins).
    assert body["customer_id"] == CUSTOMER
    assert body["expires_in_s"] > 0

    # Custody: owner-only, from the open flags — never a chmod window.
    assert oct(harness.path.stat().st_mode & 0o777) == "0o600"
    assert harness.installed_text() == f"{VALID_BLOB}\n"
    # The blob is stored verbatim: a re-read must verify byte-for-byte, since
    # the signature covers the exact received segments.
    stored = harness.installed_text() or ""
    assert verify_license(stored, trusted_keys=TEST_TRUSTED_KEYS, now=_clock).state == "valid"

    # (D5) The running daemon is refreshed IN PLACE — no restart, and the very
    # same holder object every surface already captured.
    assert harness.holder.installed is True
    assert harness.holder.state == "valid"
    assert harness.holder.claims is not None
    assert harness.holder.claims.lid == GOLDEN_CLAIMS["lid"]


def test_install_audits_lid_plan_state_and_never_the_blob(harness: Harness) -> None:
    _install(harness, VALID_BLOB)

    row = harness.audit_rows()[-1]
    assert row["action"] == "license.install"
    assert row["target_type"] == "license"
    assert row["target_id"] == GOLDEN_CLAIMS["lid"]
    params = json.loads(row["params_redacted"])
    assert params == {"lid": GOLDEN_CLAIMS["lid"], "plan": "pro", "state": "valid"}


def test_install_emits_one_machine_shaped_event(harness: Harness) -> None:
    _install(harness, VALID_BLOB)

    assert len(harness.recorder.events) == 1
    name, kwargs = harness.recorder.events[0]
    assert name == "license.installed"
    # The feed reaches third-party webhook hosts: machine tokens only.
    assert kwargs["data"] == {"lid": GOLDEN_CLAIMS["lid"], "plan": "pro", "state": "valid"}
    assert CUSTOMER not in json.dumps(kwargs)


def test_install_overwrites_an_existing_license(harness: Harness) -> None:
    """Renewal semantics: install replaces, atomically (D-LIC5)."""
    harness.path.write_text("stale-not-a-license\n")

    assert _install(harness, VALID_BLOB).status_code == 200
    assert harness.installed_text() == f"{VALID_BLOB}\n"
    assert oct(harness.path.stat().st_mode & 0o777) == "0o600"
    # No temp file survives the atomic replace.
    assert [p.name for p in harness.path.parent.iterdir()] == ["license.jws"]


def test_install_accepts_the_frozen_golden_blob(harness: Harness) -> None:
    """The vector every other suite shares installs unchanged (D-LIC9)."""
    response = _install(harness, GOLDEN_BLOB)

    assert response.status_code == 200, response.text
    assert response.json()["lid"] == GOLDEN_CLAIMS["lid"]


# ---------------------------------------------------------------------------
# install — temporal states are ACCEPTED (D4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blob,state",
    [(GRACE_BLOB, "expired_grace"), (EXPIRED_BLOB, "expired")],
)
def test_install_accepts_temporal_states_and_reports_them(
    harness: Harness, blob: str, state: str
) -> None:
    """A signature-valid license is authentic; the response tells the truth.

    Refusing an in-grace renewal would be wrong, and re-deciding temporal policy
    here would duplicate the boot-time policy that already owns it.
    """
    response = _install(harness, blob)

    assert response.status_code == 200, response.text
    assert response.json()["state"] == state
    assert harness.holder.state == state
    assert harness.installed_text() == f"{blob}\n"
    assert harness.recorder.events[0][1]["data"]["state"] == state


def test_install_accepts_a_license_without_the_remote_link_feature(harness: Harness) -> None:
    """Feature coverage is a doctor/entitlement question, not an install gate."""
    blob = _blob(expires_at=NOW + timedelta(days=30), plan="basic", features=["sso"])

    response = _install(harness, blob)

    assert response.status_code == 200, response.text
    assert response.json()["features"] == ["sso"]
    assert harness.holder.require_entitlement("remote_link").allowed is False


# ---------------------------------------------------------------------------
# install — refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "blob,reason",
    [
        ("not-a-jws", "malformed"),
        (GOLDEN_BLOB.replace(".", "x", 1), "malformed"),
        # A valid signature under an unenrolled key ID must still be refused.
        (sign_license(GOLDEN_CLAIMS, kid="not-enrolled"), "unknown_kid"),
    ],
)
def test_install_refuses_an_invalid_blob_without_persisting(
    harness: Harness, blob: str, reason: str
) -> None:
    harness.path.write_text(f"{VALID_BLOB}\n")
    harness.holder.replace(verify_license(VALID_BLOB, trusted_keys=TEST_TRUSTED_KEYS, now=_clock))

    response = _install(harness, blob)

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "license.invalid"
    assert body["reason"] == reason
    # The machine token reaches the operator through the message/detail too.
    assert reason in body["detail"]
    assert blob not in response.text

    # (D3) Nothing was displaced: the working license is byte-identical and the
    # running daemon still describes it.
    assert harness.installed_text() == f"{VALID_BLOB}\n"
    assert harness.holder.state == "valid"
    assert harness.recorder.events == []


def test_install_refuses_a_tampered_payload(harness: Harness) -> None:
    """The signature is checked over the EXACT received bytes."""
    header_b64, payload_b64, signature_b64 = VALID_BLOB.split(".")
    other = _blob(expires_at=NOW + timedelta(days=3650)).split(".")[1]
    tampered = f"{header_b64}.{other}.{signature_b64}"

    response = _install(harness, tampered)

    assert response.status_code == 422
    assert response.json()["reason"] == "bad_signature"
    assert harness.installed_text() is None


def test_install_rejects_an_oversized_blob_before_the_route_runs(harness: Harness) -> None:
    """The schema bound mirrors the file reader's cap — and echoes nothing."""
    response = _install(harness, "A" * 9000)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    # The P22 lesson: a constraint failure echoes ``input`` — ``blob`` is in
    # ``errors._SECRET_INPUT_FIELDS`` precisely so it comes back masked.
    assert "A" * 100 not in response.text
    assert harness.installed_text() is None


def test_install_rejects_an_unknown_body_field(harness: Harness) -> None:
    """``StrictRequestModel``: no silently ignored knobs on a custody surface."""
    response = harness.client.post(
        "/api/license", json={"blob": VALID_BLOB, "force": True}, headers=_ADMIN
    )

    assert response.status_code == 422
    assert VALID_BLOB not in response.text


# ---------------------------------------------------------------------------
# authz + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", [TokenRole.submitter, TokenRole.readonly])
def test_install_and_remove_are_admin_only(harness: Harness, role: TokenRole) -> None:
    harness.queries.get_api_token_by_hash.return_value = ApiToken(
        id="tok-1",
        name="ci-bot",
        role=role,
        token_hash=hash_token(SCOPED_TOKEN),
        max_gpus=1,
        max_concurrent_jobs=1,
    )

    assert _install(harness, VALID_BLOB, headers=_SCOPED).status_code == 403
    assert harness.client.delete("/api/license", headers=_SCOPED).status_code == 403
    assert harness.installed_text() is None


@pytest.mark.parametrize("method", ["post", "delete"])
def test_both_verbs_require_an_idempotency_key(harness: Harness, method: str) -> None:
    headers = {"Authorization": f"Bearer {ADMIN_TOKEN}"}
    if method == "post":
        response = harness.client.post("/api/license", json={"blob": VALID_BLOB}, headers=headers)
    else:
        response = harness.client.delete("/api/license", headers=headers)

    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    assert harness.installed_text() is None


def test_the_install_response_body_is_never_cached(harness: Harness) -> None:
    """In ``NO_BODY_CACHE_ACTIONS``: the response carries ``customer_id``.

    The response mints no *secret* — everything in it is derived from claims the
    caller already holds — but it does carry ``customer_id``, which D-LIC5 and
    D-LIC7 bar from every durable store. Caching it would park that PII in
    ``idempotency_keys.response_body`` for the retention window and copy it into
    every ``POST /system/backup`` DB snapshot, which is exactly what excluding
    ``license.jws`` from the tar was meant to prevent.

    ``license.remove`` is the ruled non-member: its body is ``{"removed":
    bool}`` and the ``lid`` rides audit params by design.
    """
    assert derive_action("POST", "/api/license") == ("license.install", "license", None)
    assert "license.install" in NO_BODY_CACHE_ACTIONS
    assert derive_action("DELETE", "/api/license")[0] == "license.remove"
    assert "license.remove" not in NO_BODY_CACHE_ACTIONS


# ---------------------------------------------------------------------------
# the ASSEMBLED middleware stack (Auth -> Audit -> Idempotency -> routes)
#
# The harness above wires the routes without IdempotencyMiddleware (the routes
# carry their own in-route key gate), so nothing else in this file exercises
# ``_hashable_body`` or ``_settle`` against the license verbs — which is where
# both at-rest copies of the two never-persist values would be written.
# ---------------------------------------------------------------------------


async def _stack_app(tmp_path: Path) -> tuple[FastAPI, Any]:
    """A real DB behind the real middleware order (test_backup_route idiom)."""
    from nerdit.db.database import Database
    from nerdit.db.queries import Queries

    database = Database(":memory:")
    await database.connect()
    await database.init_schema()
    queries = Queries(database)

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(license_router)
    app.include_router(api)
    app.state.settings = SimpleNamespace(data_dir=str(data_dir), license=LicenseSettings())
    app.state.license = LicenseState(now=_clock)
    app.state.license_trusted_keys = TEST_TRUSTED_KEYS
    app.state.queries = queries

    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=ADMIN_TOKEN, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, database


@pytest.mark.asyncio
async def test_the_idempotency_row_stores_neither_the_blob_digest_nor_the_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing derived from the blob, and no ``customer_id``, reaches the store.

    ``license.install`` is in **both** exclusion sets, and this is the only test
    that proves it through the assembled stack rather than by set membership:

    * ``NO_BODY_HASH_ACTIONS`` ⇒ ``body_hash IS NULL``. The request body is the
      bare blob; the hard rule is that it never lives outside the license file
      in any derived form, and this row rides every backup tar's DB snapshot.
    * ``NO_BODY_CACHE_ACTIONS`` ⇒ ``response_body IS NULL``. The live response
      carries ``customer_id`` on purpose (admin-gated render, D-LIC5) — the
      replay must not, and neither must the row.
    """
    from httpx import ASGITransport, AsyncClient

    recorder = FakeRecorder()
    monkeypatch.setattr(eventlog, "_recorder", recorder)
    app, database = await _stack_app(tmp_path)
    try:
        headers = {"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "K-license"}
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/api/license", json={"blob": VALID_BLOB}, headers=headers)
            assert first.status_code == 200, first.text
            assert first.headers.get("Idempotent-Replay") is None
            # The live render is deliberately admin-visible (D-LIC5).
            assert first.json()["customer_id"] == CUSTOMER

            cur = await database.conn.execute(
                "SELECT body_hash, response_body FROM idempotency_keys"
            )
            rows = [tuple(row) for row in await cur.fetchall()]
            assert rows == [(None, None)], rows

            second = await client.post("/api/license", json={"blob": VALID_BLOB}, headers=headers)
            assert second.status_code == 200, second.text
            assert second.headers.get("Idempotent-Replay") == "true"
            replay = second.text
            assert CUSTOMER not in replay
            assert VALID_BLOB not in replay
    finally:
        await database.close()


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_deletes_the_file_and_clears_the_running_state(harness: Harness) -> None:
    assert _install(harness, VALID_BLOB).status_code == 200

    response = harness.client.delete("/api/license", headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json() == {"removed": True}
    assert harness.installed_text() is None
    assert harness.holder.installed is False
    assert harness.holder.state is None

    row = harness.audit_rows()[-1]
    assert row["action"] == "license.remove"
    assert row["target_id"] == GOLDEN_CLAIMS["lid"]
    params = json.loads(row["params_redacted"])
    assert params["removed"] is True
    assert params["lid"] == GOLDEN_CLAIMS["lid"]
    assert CUSTOMER not in json.dumps(params)

    assert harness.recorder.events[-1] == (
        "license.removed",
        {"data": {"lid": GOLDEN_CLAIMS["lid"]}},
    )


def test_remove_on_an_unlicensed_daemon_is_an_idempotent_no_op(harness: Harness) -> None:
    """A 200 reporting what happened, never a 404 — retries must be safe."""
    response = harness.client.delete("/api/license", headers=_ADMIN)

    assert response.status_code == 200
    assert response.json() == {"removed": False}
    # No file, no state, no event: nothing happened, and nothing is claimed.
    assert harness.recorder.events == []
    row = harness.audit_rows()[-1]
    assert row["action"] == "license.remove"
    assert json.loads(row["params_redacted"])["removed"] is False


def test_double_remove_reports_the_second_one_honestly(harness: Harness) -> None:
    _install(harness, VALID_BLOB)
    first = harness.client.delete("/api/license", headers=_ADMIN)
    second = harness.client.delete(
        "/api/license", headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "k2"}
    )

    assert first.json() == {"removed": True}
    assert second.json() == {"removed": False}
    # Exactly one install event and one removal event — the second remove found
    # nothing and said nothing.
    assert [name for name, _ in harness.recorder.events] == [
        "license.installed",
        "license.removed",
    ]


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------


def test_never_leak_sweep(harness: Harness, caplog: pytest.LogCaptureFixture) -> None:
    """Neither the blob nor ``customer_id`` reaches a log, an audit row or a
    non-envelope response — across the happy path, every refusal and the remove.

    A leak is most likely on an error path (an exception message quoting the
    request, a debug log of the body), which is why the refusals are swept too.
    """
    caplog.set_level(logging.DEBUG)
    blobs = [
        VALID_BLOB,
        GRACE_BLOB,
        EXPIRED_BLOB,
        "not-a-jws",
        sign_license(GOLDEN_CLAIMS, kid="not-enrolled"),
        "A" * 9000,
    ]
    texts: list[str] = []
    for index, blob in enumerate(blobs):
        response = harness.client.post(
            "/api/license",
            json={"blob": blob},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": f"k{index}"},
        )
        texts.append(response.text)
    texts.append(
        harness.client.delete(
            "/api/license",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "kdel"},
        ).text
    )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    audited = json.dumps(harness.audit_rows(), default=str)
    events = json.dumps(harness.recorder.events, default=str)
    for blob in blobs:
        assert blob not in logged
        assert blob not in audited
        assert blob not in events
        for text in texts:
            assert blob not in text
    assert CUSTOMER not in logged
    assert CUSTOMER not in audited
    assert CUSTOMER not in events
    # ``customer_id`` appears in exactly one place by design: the admin-gated
    # install envelope. It must be nowhere else, including the refusals.
    assert [CUSTOMER in text for text in texts] == [True, True, True, False, False, False, False]
