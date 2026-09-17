"""Declarative config apply route tests (P7).

Covers ``POST /api/config/daemon/apply`` end-to-end through the Auth + Audit
middleware stack: multi-section commit-once, all-or-nothing on failure with
aggregated namespaced diagnostics, mandatory If-Match (with ``current_etag``
on 409) + Idempotency-Key on real applies (both exempt on dry-run), same-
document idempotence (``changed: false``, same ETag), the auth_token block,
admin-only access, and the redacted per-section audit row.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.config.store import ConfigStore
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.config import router as config_router
from nerdit.db.models import ApiToken, TokenRole

LEGACY = "legacy-global"
ADMIN_RAW = "admin-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}

APPLY = "/api/config/daemon/apply"


def _queries() -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    return q


def _make_app(path: Path, queries: AsyncMock) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(config_router, prefix="/api")
    app.state.queries = queries
    app.state.config_store = ConfigStore(path)
    # inner → outer: Audit (innermost), Auth, RequestId.
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(path: Path, queries: AsyncMock) -> TestClient:
    return TestClient(_make_app(path, queries), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _seed(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def _etag(client: TestClient) -> str:
    return client.get("/api/config/daemon", headers=_auth(ADMIN_RAW)).headers["ETag"]


def _apply_headers(client: TestClient, key: str = "k1") -> dict:
    return {**_auth(ADMIN_RAW), "Idempotency-Key": key, "If-Match": _etag(client)}


# --- happy path -----------------------------------------------------------------


def test_multi_section_apply_commits_once(tmp_path):
    path = _seed(tmp_path, "[services]\nservice_max_restarts = 5\n")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={
            "sections": {
                "services": {"service_max_restarts": 9},
                "monitor": {"interval_seconds": 10},
            }
        },
        headers=_apply_headers(client),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is True
    assert body["changed"] is True
    assert body["etag"] == resp.headers["ETag"]
    keys = {entry["key"] for entry in body["diff"]}
    assert keys == {"services.service_max_restarts", "monitor.interval_seconds"}
    store = ConfigStore(path)
    assert store.effective_section("services")["service_max_restarts"] == 9
    assert store.effective_section("monitor")["interval_seconds"] == 10
    assert store.current_etag() == body["etag"]


def test_diff_entries_carry_section_and_op(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers=_apply_headers(client),
    )
    entry = resp.json()["diff"][0]
    assert entry["section"] == "monitor"
    assert entry["op"] == "change"
    assert entry["secret"] is False


def test_restart_keys_aggregated(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"daemon": {"port": 9999}, "proxy": {"enabled": True}}},
        headers=_apply_headers(client),
    )
    body = resp.json()
    assert body["requires_restart"] is True
    assert body["restart_keys"] == ["daemon.port", "proxy.enabled"]


# --- validation / all-or-nothing --------------------------------------------------


def test_failure_leaves_file_untouched_with_aggregated_diagnostics(tmp_path):
    path = _seed(tmp_path, "[services]\nservice_max_restarts = 5\n")
    before = path.read_bytes()
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={
            "sections": {
                "services": {"service_max_restarts": 9},
                "daemon": {"port": "not-int"},
                "monitor": {"nope": 1},
            }
        },
        headers=_apply_headers(client),
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "config.invalid"
    locs = {tuple(d["loc"]) for d in body["diagnostics"]}
    assert ("daemon", "port") in locs
    assert ("monitor", "nope") in locs
    # All-or-nothing: nothing written, valid sibling section included.
    assert path.read_bytes() == before


def test_unknown_section_422(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"nope": {"x": 1}}},
        headers=_apply_headers(client),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "config.unknown_section"


def test_auth_token_write_blocked(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"daemon": {"auth_token": "x"}}},
        headers=_apply_headers(client),
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "config.forbidden"


# --- concurrency / idempotency ceremony -------------------------------------------


def test_if_match_required_on_real_apply(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "config.if_match_required"


def test_stale_etag_conflicts_with_current_etag(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    stale = _etag(client)
    # Move the config forward.
    client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 7}}},
        headers=_apply_headers(client),
    )
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 8}}},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k2", "If-Match": stale},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "config.stale"
    assert body["current_etag"] == ConfigStore(path).current_etag()


def test_if_match_wildcard_applies_without_stale(tmp_path):
    """RFC 7232 ``If-Match: *`` satisfies the mandatory-presence precondition and
    applies without a config.stale 409; the header stays mandatory when absent."""
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1", "If-Match": "*"},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True
    assert ConfigStore(path).effective_section("monitor")["interval_seconds"] == 9


def test_if_match_still_required_when_absent(tmp_path):
    """The wildcard escape does not remove the mandatory-presence check."""
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "config.if_match_required"


def test_idempotency_key_required_on_real_apply(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers={**_auth(ADMIN_RAW), "If-Match": _etag(client)},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "idempotency_key_required"


def test_dry_run_exempt_from_if_match_and_idempotency_key(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    before = path.read_bytes()
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        params={"dry_run": "true"},
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["changed"] is True
    assert body["diff"][0]["key"] == "monitor.interval_seconds"
    assert body["etag"] == resp.headers["ETag"]
    # Nothing written.
    assert path.read_bytes() == before


def test_same_document_reapply_is_noop(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    doc = {"sections": {"monitor": {"interval_seconds": 9}}}
    first = client.post(APPLY, json=doc, headers=_apply_headers(client, "k1"))
    assert first.json()["changed"] is True
    second = client.post(APPLY, json=doc, headers=_apply_headers(client, "k2"))
    assert second.status_code == 200
    body = second.json()
    assert body["applied"] is True
    assert body["changed"] is False
    assert body["diff"] == []
    assert body["etag"] == first.json()["etag"]


# --- authz / audit ----------------------------------------------------------------


def test_apply_is_admin_only(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}}},
        headers={**_auth(RO_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_apply_audited_with_redacted_params(tmp_path):
    path = _seed(tmp_path, "")
    q = _queries()
    client = _client(path, q)
    resp = client.post(
        APPLY,
        json={"sections": {"monitor": {"interval_seconds": 9}, "daemon": {"port": 9999}}},
        headers=_apply_headers(client),
    )
    assert resp.status_code == 200
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "config.apply"
    assert kwargs["result"] == "ok"
    assert kwargs["principal_role"] == "admin"
    params = json.loads(kwargs["params_redacted"])
    assert params["dry_run"] is False
    assert params["requires_restart"] is True
    assert params["changed_keys"] == {
        "daemon": ["daemon.port"],
        "monitor": ["monitor.interval_seconds"],
    }


# --- no submitted value ever reaches the audit trail --------------------------
#
# Same defect as the section PUT: the pre-validation stamp carried the submitted
# document, masked only by three leaf names, so a refused literal landed in
# `audit_log.params_redacted` and on the admin `audit.*` stream.

_CANARY = "s3kr3t-HMAC-CANARY"


class _RecordingBus:
    """Event bus stand-in that keeps every published frame."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def publish(self, frame: dict) -> None:
        self.frames.append(frame)


def _audited_client(path: Path, queries: AsyncMock, bus: _RecordingBus) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(config_router, prefix="/api")
    app.state.queries = queries
    app.state.config_store = ConfigStore(path)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: bus)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _recorded(queries: AsyncMock, bus: _RecordingBus) -> tuple[str, str]:
    """Return the audit row's params JSON and the published frame, serialized."""
    queries.insert_audit_log.assert_awaited()
    row = queries.insert_audit_log.await_args.kwargs["params_redacted"] or ""
    assert bus.frames
    return row, json.dumps(bus.frames[-1])


def test_apply_rejection_never_records_the_submitted_value(tmp_path):
    path = _seed(tmp_path, "")
    q = _queries()
    bus = _RecordingBus()
    client = _audited_client(path, q, bus)
    resp = client.post(
        APPLY,
        json={
            "sections": {
                "notifications": {
                    "enabled": True,
                    "targets": [{"url": "https://hooks.example.com/x", "secret_ref": _CANARY}],
                }
            }
        },
        headers=_apply_headers(client),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "config.invalid"
    row, frame = _recorded(q, bus)
    assert _CANARY not in row and _CANARY not in frame
    assert json.loads(row) == {
        "dry_run": False,
        "submitted_keys": {"notifications": ["enabled", "targets"]},
    }


def test_apply_dry_run_never_records_the_submitted_value(tmp_path):
    path = _seed(tmp_path, "")
    q = _queries()
    bus = _RecordingBus()
    resp = _audited_client(path, q, bus).post(
        APPLY,
        json={"sections": {"proxy": {"acme": {"email": f"ops+{_CANARY}@example.com"}}}},
        params={"dry_run": "true"},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is False
    row, frame = _recorded(q, bus)
    assert _CANARY not in row and _CANARY not in frame
    assert json.loads(row) == {"dry_run": True, "submitted_keys": {"proxy": ["acme"]}}
