"""Route-level tests for ``POST /system/backup`` (P14c WP-B3, WP4).

Harness mirrors ``test_system_disk_gc.py``: a hand-built FastAPI app with mocked
state on ``app.state`` and the real auth + audit + error middleware. The heavy
``create_backup`` orchestration (covered by ``test_backup.py``) is monkeypatched
so these tests exercise only the route contract — admin gating, the two 409s,
the 500, the audit row shape, and idempotency replay.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import LicenseSettings, LinkSettings
from nerdit.core.backup import BackupError, BackupResult, VolumeBackupResult
from nerdit.core.secrets import SecretManager, SecretRotationInProgress
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import system as system_routes
from nerdit.daemon.routes.system import router as system_router
from nerdit.db.database import Database
from nerdit.db.models import JobKind, TokenRole
from nerdit.db.queries import Queries
from nerdit.db.queries._base import REQUEST_WRITE_MARKER

_ADMIN = {"Authorization": "Bearer admin-raw-token"}
_READONLY = {"Authorization": "Bearer readonly-raw-token"}
_SUBMITTER = {"Authorization": "Bearer submitter-raw-token"}


def _result(basename: str = "nerdit-backup-20260713T000000Z-abcdef.tar.gz") -> BackupResult:
    return BackupResult(
        path=f"/data/backups/{basename}",
        basename=basename,
        size_bytes=4096,
        kid="deadbeef",
        manifest={"created_at": "2026-07-13T00:00:00+00:00", "kid": "deadbeef"},
    )


# --- lightweight harness (AsyncMock queries; create_backup monkeypatched) ------


def _token_row(role: TokenRole) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"tok-{role.value}",
        name="ci",
        role=role,
        max_gpus=2,
        max_concurrent_jobs=4,
        # P25 WP1: the middleware reads both on every scoped-token request.
        expires_at=None,
        scope_services=None,
    )


def _queries(role) -> SimpleNamespace:  # noqa: ANN001
    row = _token_row(role) if role is not None else None
    return SimpleNamespace(
        get_api_token_by_hash=AsyncMock(return_value=row),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(),
    )


def _app(*, role: TokenRole, data_dir) -> TestClient:  # noqa: ANN001
    queries = _queries(role)
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = SimpleNamespace(
        data_dir=str(data_dir), link=LinkSettings(), license=LicenseSettings()
    )
    app.state.db = SimpleNamespace()
    app.state.secret_manager = SimpleNamespace()
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    client = TestClient(app)
    client._queries = queries  # type: ignore[attr-defined]
    return client


# --- tests --------------------------------------------------------------------


def test_backup_requires_admin(tmp_path, monkeypatch):
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(return_value=_result()))
    client = _app(role=TokenRole.readonly, data_dir=tmp_path)
    # readonly is coarse-gated on mutating methods before the route.
    assert client.post("/api/system/backup", headers=_READONLY).status_code == 403


def test_backup_submitter_forbidden(tmp_path, monkeypatch):
    # A submitter passes the coarse mutating-method gate; the actual guard is the
    # in-route require_role(admin), which must still 403 (finding L4).
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(return_value=_result()))
    client = _app(role=TokenRole.submitter, data_dir=tmp_path)
    assert client.post("/api/system/backup", headers=_SUBMITTER).status_code == 403


def test_backup_concurrent_409(tmp_path, monkeypatch):
    monkeypatch.setattr(system_routes._backup_lock, "locked", lambda: True)
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(return_value=_result()))
    client = _app(role=TokenRole.admin, data_dir=tmp_path)
    resp = client.post("/api/system/backup", headers=_ADMIN)
    assert resp.status_code == 409
    assert resp.json()["code"] == "backup.in_progress"


def test_backup_staged_rotation_409_no_path_in_body(tmp_path, monkeypatch):
    # The real exception embeds an absolute staged-key path; the route must
    # substitute a fixed, path-free message (D2/M3).
    exc = SecretRotationInProgress("rotation staged at /home/x/.nerdit/secrets.key.new")
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(side_effect=exc))
    client = _app(role=TokenRole.admin, data_dir=tmp_path)
    resp = client.post("/api/system/backup", headers=_ADMIN)
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "secret.rotation_in_progress"
    assert "/" not in json.dumps(body)


def test_backup_failed_500_no_path_in_message(tmp_path, monkeypatch):
    exc = BackupError("backup staging failed: OSError: No such file or directory")
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(side_effect=exc))
    client = _app(role=TokenRole.admin, data_dir=tmp_path)
    resp = client.post("/api/system/backup", headers=_ADMIN)
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "backup.failed"
    assert "/" not in body["message"]


def test_backup_success_body_and_audit(tmp_path, monkeypatch):
    result = _result()
    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(return_value=result))
    client = _app(role=TokenRole.admin, data_dir=tmp_path)
    resp = client.post("/api/system/backup", headers=_ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert body["backup"] == result.basename
    assert body["path"] == result.path
    assert body["size_bytes"] == 4096
    assert body["kid"] == "deadbeef"
    assert body["created_at"] == "2026-07-13T00:00:00+00:00"
    assert body["contains_master_key"] is True
    assert "master key" in body["hint"]

    q = client._queries  # type: ignore[attr-defined]
    call = q.insert_audit_log.await_args.kwargs
    assert call["action"] == "system.backup"
    params = json.loads(call["params_redacted"])
    # Exactly basename + contains_master_key — no path, no kid, no size (D10).
    assert params == {"backup": result.basename, "contains_master_key": True}
    assert "/" not in json.dumps(params)


# --- idempotency replay (real DB + IdempotencyMiddleware) ---------------------


async def _make_db_app(tmp_path, monkeypatch) -> tuple[FastAPI, Database]:
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    monkeypatch.setattr(system_routes, "create_backup", AsyncMock(return_value=_result()))

    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = SimpleNamespace(
        data_dir=str(tmp_path), link=LinkSettings(), license=LicenseSettings()
    )
    app.state.db = SimpleNamespace()
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.queries = queries

    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="legacy-global", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    return app, db


@pytest.mark.asyncio
async def test_backup_idempotent_replay(tmp_path, monkeypatch):
    app, db = await _make_db_app(tmp_path, monkeypatch)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": "Bearer legacy-global", "Idempotency-Key": "K-backup"}
            first = await client.post("/api/system/backup", headers=headers)
            assert first.status_code == 200
            assert first.headers.get("Idempotent-Replay") is None

            second = await client.post("/api/system/backup", headers=headers)
            assert second.status_code == 200
            assert second.headers.get("Idempotent-Replay") == "true"
            # The replay returns the pinned body — create_backup ran only once.
            assert second.json() == first.json()
            assert system_routes.create_backup.await_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_backup_pre_marks_the_write_marker(tmp_path, monkeypatch):
    """Staging runs in worker threads that a cancellation cannot stop.

    ``@_serialized`` never fires on this route, so without an explicit pre-mark a
    cancelled capture would look "nothing committed" to the middleware, which
    deletes the claim — and a same-key retry publishes a SECOND key-bearing tar.
    The marker must therefore already be flipped when staging begins.
    """
    app, db = await _make_db_app(tmp_path, monkeypatch)
    seen: dict = {}

    async def _capture(**kwargs):  # noqa: ANN003, ANN202
        marker = REQUEST_WRITE_MARKER.get()
        seen["marker"] = marker
        seen["committed"] = marker is not None and marker.committed
        return _result()

    monkeypatch.setattr(system_routes, "create_backup", _capture)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/backup",
                headers={"Authorization": "Bearer legacy-global", "Idempotency-Key": "K-mark"},
            )
            assert resp.status_code == 200
        assert seen["marker"] is not None, "the middleware must install a marker"
        # No @_serialized writer runs in this route before staging, so a True
        # here can only come from the explicit pre-mark.
        assert seen["committed"] is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_volume_backup_pre_marks_the_write_marker(tmp_path, monkeypatch):
    """Same contract on the volume tar (staged via ``asyncio.to_thread``, which
    copies the context, so the marker is visible from the worker thread)."""
    app, db = await _make_db_app(tmp_path, monkeypatch)
    queries = app.state.queries
    queries.get_service_by_name = AsyncMock(return_value=_db_job())
    seen: dict = {}

    def _capture(**kwargs):  # noqa: ANN003, ANN202
        marker = REQUEST_WRITE_MARKER.get()
        seen["marker"] = marker
        seen["committed"] = marker is not None and marker.committed
        return _vol_result()

    monkeypatch.setattr(system_routes, "create_volume_backup", _capture)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/system/backup/volumes",
                headers={"Authorization": "Bearer legacy-global", "Idempotency-Key": "K-vmark"},
                json={"service": "pg"},
            )
            assert resp.status_code == 200
        assert seen["marker"] is not None
        assert seen["committed"] is True
    finally:
        await db.close()


# --- POST /system/backup/volumes (P15 WP7 — backup v2) ------------------------


def _vol_result(
    basename: str = "nerdit-volumes-pg-20260714T000000Z-abcdef.tar.gz",
) -> VolumeBackupResult:
    return VolumeBackupResult(
        path=f"/data/backups/{basename}",
        basename=basename,
        size_bytes=8192,
        service="pg",
        backend="postgres",
        manifest={"created_at": "2026-07-14T00:00:00+00:00", "version": 1},
    )


def _db_job(kind: JobKind = JobKind.database, backend: str | None = "postgres") -> SimpleNamespace:
    cfg = {"backend": backend} if backend is not None else {}
    return SimpleNamespace(kind=kind, config=json.dumps(cfg))


def _vol_app(*, role: TokenRole, data_dir, job) -> TestClient:  # noqa: ANN001
    queries = _queries(role)
    queries.get_service_by_name = AsyncMock(return_value=job)
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = SimpleNamespace(
        data_dir=str(data_dir), link=LinkSettings(), license=LicenseSettings()
    )
    app.state.queries = queries
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    client = TestClient(app)
    client._queries = queries  # type: ignore[attr-defined]
    return client


def test_volume_backup_requires_admin(tmp_path, monkeypatch):
    client = _vol_app(role=TokenRole.readonly, data_dir=tmp_path, job=_db_job())
    resp = client.post("/api/system/backup/volumes", headers=_READONLY, json={"service": "pg"})
    assert resp.status_code == 403


def test_volume_backup_404_unknown(tmp_path, monkeypatch):
    client = _vol_app(role=TokenRole.admin, data_dir=tmp_path, job=None)
    resp = client.post("/api/system/backup/volumes", headers=_ADMIN, json={"service": "nope"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_volume_backup_422_not_a_database(tmp_path, monkeypatch):
    client = _vol_app(role=TokenRole.admin, data_dir=tmp_path, job=_db_job(kind=JobKind.service))
    resp = client.post("/api/system/backup/volumes", headers=_ADMIN, json={"service": "web"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "backup.not_a_database"


def test_volume_backup_success_body_and_audit(tmp_path, monkeypatch):
    result = _vol_result()
    monkeypatch.setattr(system_routes, "create_volume_backup", lambda **kw: result)
    client = _vol_app(role=TokenRole.admin, data_dir=tmp_path, job=_db_job())
    resp = client.post("/api/system/backup/volumes", headers=_ADMIN, json={"service": "pg"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "pg"
    assert body["backend"] == "postgres"
    assert body["backup"] == result.basename
    assert body["path"] == result.path
    assert body["size_bytes"] == 8192
    # Never claims the master key (unlike the control-plane tar).
    assert body["contains_master_key"] is False
    assert "master key" in body["hint"]

    q = client._queries  # type: ignore[attr-defined]
    call = q.insert_audit_log.await_args.kwargs
    assert call["action"] == "system.backup_volume"
    params = json.loads(call["params_redacted"])
    # Exactly service + basename — no path, no size (§1.4/§1.5).
    assert params == {"service": "pg", "backup": result.basename}


# --- exclusions threaded by the route (node key + product license) ------------


def _exclusions(tmp_path, monkeypatch, license_settings) -> set[str]:  # noqa: ANN001
    """POST a backup and return the ``exclude_paths`` the route threaded in."""
    spy = AsyncMock(return_value=_result())
    monkeypatch.setattr(system_routes, "create_backup", spy)
    client = _app(role=TokenRole.admin, data_dir=tmp_path)
    client.app.state.settings.license = license_settings
    assert client.post("/api/system/backup", headers=_ADMIN).status_code == 200
    return {str(p) for p in spy.await_args.kwargs["exclude_paths"]}


def test_backup_excludes_the_product_license_by_default(tmp_path, monkeypatch):
    """P17d D-LIC7: the license is re-issuable, so it never rides a backup tar.

    The default path already sits outside every captured tree — this pins that
    the route says so explicitly, which is what protects an operator-chosen
    ``[license].file`` planted INSIDE one (the [link].key_file precedent).
    """
    excluded = _exclusions(tmp_path, monkeypatch, LicenseSettings())

    assert str(tmp_path / "license.jws") in excluded
    # The node key exclusion (P27 WP-C3) is untouched by the addition.
    assert str(tmp_path / "link" / "node.key") in excluded


def test_backup_exclusion_follows_a_license_file_override(tmp_path, monkeypatch):
    planted = tmp_path / "caddy" / "pki" / "license.jws"
    excluded = _exclusions(tmp_path, monkeypatch, LicenseSettings(file=str(planted)))

    assert str(planted) in excluded
    assert str(tmp_path / "license.jws") not in excluded
