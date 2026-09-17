"""Config-as-API route tests (P1 / S8).

Covers ``GET``/``PUT /api/config/daemon`` end-to-end through the Auth + Audit
middleware stack: redacted reads with an ETag, admin-only writes, dry-run vs
real write, the Idempotency-Key requirement, ETag staleness (409), the
auth_token block, validation diagnostics, and the audit row recorded on commit.
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


# --- reads --------------------------------------------------------------------


def test_get_section_redacts_secret_and_sets_etag(tmp_path):
    path = _seed(tmp_path, '[daemon]\nauth_token = "sekret"\n')
    client = _client(path, _queries())
    resp = client.get("/api/config/daemon/daemon", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    assert resp.json()["values"]["auth_token"] == "***"
    assert resp.headers.get("ETag")


def test_get_all_sections(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.get("/api/config/daemon", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    sections = {v["section"] for v in resp.json()}
    assert {"daemon", "monitor", "containers"} <= sections


def test_get_unknown_section_404(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.get("/api/config/daemon/nope", headers=_auth(RO_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "config.unknown_section"


# --- writes -------------------------------------------------------------------


def test_dry_run_returns_diff_without_writing(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    store_client = _client(path, _queries())
    resp = store_client.put(
        "/api/config/daemon/monitor",
        params={"dry_run": "true"},
        json={"interval_seconds": 9},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is False
    assert body["diff"][0]["key"] == "monitor.interval_seconds"
    # Nothing written.
    assert ConfigStore(path).effective_section("monitor")["interval_seconds"] == 5


def test_real_write_requires_idempotency_key(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 9},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "idempotency_key_required"


def test_real_write_applies_and_audits(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    q = _queries()
    client = _client(path, q)
    resp = client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 9},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True
    assert ConfigStore(path).effective_section("monitor")["interval_seconds"] == 9
    # Audited as config.update.
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "config.update"
    assert kwargs["result"] == "ok"
    assert kwargs["principal_role"] == "admin"


def test_sandbox_hardening_write_reports_requires_restart(tmp_path):
    # [containers] is captured once at boot (DockerRuntime + ServiceController),
    # so answering requires_restart=false would tell the operator the hardening
    # is live when the running daemon is still on the boot value.
    path = _seed(tmp_path, "[containers]\nread_only_rootfs = false\n")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/containers",
        json={"read_only_rootfs": True, "drop_all_caps": False},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k-containers"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["applied"] is True
    assert body["requires_restart"] is True
    assert sorted(entry["key"] for entry in body["diff"]) == [
        "containers.drop_all_caps",
        "containers.read_only_rootfs",
    ]
    assert all(entry["requires_restart"] is True for entry in body["diff"])


def test_write_is_admin_only(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 9},
        headers={**_auth(RO_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_stale_etag_conflicts(tmp_path):
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    get = client.get("/api/config/daemon/monitor", headers=_auth(ADMIN_RAW))
    etag = get.headers["ETag"]
    # First write moves the ETag forward.
    client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 7},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    # Second write with the now-stale ETag is rejected.
    resp = client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 8},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k2", "If-Match": etag},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "config.stale"


def test_if_match_wildcard_writes_if_exists(tmp_path):
    """RFC 7232 ``If-Match: *`` (write-if-exists) applies without a config.stale
    409 — the representation exists, so the precondition is satisfied."""
    path = _seed(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/monitor",
        json={"interval_seconds": 9},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1", "If-Match": "*"},
    )
    assert resp.status_code == 200
    assert resp.json()["applied"] is True
    assert ConfigStore(path).effective_section("monitor")["interval_seconds"] == 9


def test_auth_token_write_blocked(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/daemon",
        json={"auth_token": "x"},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "config.forbidden"


def test_invalid_value_returns_diagnostics(tmp_path):
    path = _seed(tmp_path, "")
    client = _client(path, _queries())
    resp = client.put(
        "/api/config/daemon/daemon",
        json={"port": "not-int"},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "config.invalid"
    assert body["diagnostics"]


# --- no submitted value ever reaches the audit trail --------------------------
#
# The stamp the AuditMiddleware reads is set BEFORE validation, so it is what a
# 422 / 409 / dry run records. Stamping the body (masked only by three leaf
# names, shallowly) put an operator's literal — pasted where a
# ${secrets.shared.KEY} ref belonged, and therefore refused — verbatim into
# `audit_log.params_redacted` and onto the admin `audit.*` stream.

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


def test_section_put_rejection_never_records_the_submitted_value(tmp_path):
    path = _seed(tmp_path, "")
    q = _queries()
    bus = _RecordingBus()
    resp = _audited_client(path, q, bus).put(
        "/api/config/daemon/notifications",
        json={
            "enabled": True,
            "targets": [{"url": "https://hooks.example.com/x", "secret_ref": _CANARY}],
        },
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "config.invalid"
    row, frame = _recorded(q, bus)
    assert _CANARY not in row and _CANARY not in frame
    assert json.loads(row) == {
        "section": "notifications",
        "dry_run": False,
        "submitted_keys": ["enabled", "targets"],
    }


def test_section_put_dry_run_never_records_the_submitted_value(tmp_path):
    # `?dry_run=true` answers 200/ok and writes nothing — the audit row it still
    # produces must not be where the submitted value lands instead.
    path = _seed(tmp_path, "")
    q = _queries()
    bus = _RecordingBus()
    resp = _audited_client(path, q, bus).put(
        "/api/config/daemon/proxy",
        json={"acme": {"email": f"ops+{_CANARY}@example.com"}},
        params={"dry_run": "true"},
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is False
    row, frame = _recorded(q, bus)
    assert _CANARY not in row and _CANARY not in frame
    assert json.loads(row) == {"section": "proxy", "dry_run": True, "submitted_keys": ["acme"]}


def test_section_put_commit_records_changed_key_names_only(tmp_path):
    # Success-path content is unchanged: the committed keys, by name.
    path = _seed(tmp_path, "")
    q = _queries()
    bus = _RecordingBus()
    resp = _audited_client(path, q, bus).put(
        "/api/config/daemon/proxy",
        json={"acme": {"email": f"ops+{_CANARY}@example.com"}},
        headers={**_auth(ADMIN_RAW), "Idempotency-Key": "k1"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True
    row, frame = _recorded(q, bus)
    assert _CANARY not in row and _CANARY not in frame
    assert json.loads(row) == {
        "section": "proxy",
        "dry_run": False,
        "changed_keys": ["proxy.acme"],
    }
