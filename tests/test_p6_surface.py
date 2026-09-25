"""Verify service status, audit access, and MCP tools against real route handlers.

Pure-async at the route+DB level (mirrors ``tests/test_p5_northstar.py`` —
never the Starlette ``TestClient``): the real cluster/services/audit routers
run under ``httpx.ASGITransport`` over the same in-memory DB the tests seed.
Asserts the P6 surface contract:

1. ``GET /cluster/stats`` populates ``services_up`` (running/degraded
   service+model rows; batch never counts — trap 4);
2. ``ServiceResponse`` carries the additive P6 (D3) fields
   ``rollback_available`` / ``build_version``, projected from the deploy
   config blob;
3. ``GET /audit`` returns the ``AuditLogPage`` shape (cursor pagination,
   newest first) and stays admin-only (readonly token → 403 ``forbidden``);
4. the MCP server projects the P6 agent tools (``service_logs`` /
   ``get_audit`` / ``get_config`` / ``set_config``) — guarded by
   ``importorskip`` like ``test_mcp.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import NerditSettings
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.audit import router as audit_router
from nerdit.daemon.routes.cluster import router as cluster_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries

pytestmark = pytest.mark.asyncio

LEGACY = "legacy-global"
RO_RAW = "readonly-raw"


def _app(queries: Queries, *, token: str | None = None) -> FastAPI:
    """The real P6 read surface under the production auth middleware."""
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(cluster_router)
    app.include_router(services_router)
    app.include_router(audit_router)
    app.state.queries = queries
    app.state.settings = NerditSettings()
    monitor = MagicMock()
    monitor.get_metrics = MagicMock(return_value={})
    app.state.monitor = monitor
    app.add_middleware(ScopedTokenAuthMiddleware, token=token, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: Queries, *, token: str | None = None) -> AsyncClient:
    transport = ASGITransport(app=_app(queries, token=token))
    return AsyncClient(transport=transport, base_url="http://127.0.0.1")


def _row(
    job_id: str,
    kind: JobKind,
    status: JobStatus,
    service_name: str | None = None,
    config: dict | None = None,
) -> Job:
    return Job(
        id=job_id,
        name=job_id,
        kind=kind,
        service_name=service_name,
        script_path="/x.py" if kind is JobKind.batch else None,
        gpu_count=0,
        status=status,
        desired_state=None if kind is JobKind.batch else "running",
        config=json.dumps(config) if config is not None else None,
    )


# --- 1. services_up is populated (demo step 4) --------------------------------


async def test_cluster_stats_populates_services_up(queries: Queries):
    # Counted: running/degraded services + models. Ignored: batch (even
    # running) and non-up service states.
    await queries.create_job(_row("svc-run", JobKind.service, JobStatus.running, "svc-run"))
    await queries.create_job(_row("svc-deg", JobKind.service, JobStatus.degraded, "svc-deg"))
    await queries.create_job(_row("mdl-run", JobKind.model, JobStatus.running, "mdl-run"))
    await queries.create_job(_row("batch-run", JobKind.batch, JobStatus.running))
    await queries.create_job(_row("svc-stop", JobKind.service, JobStatus.stopped, "svc-stop"))

    async with _client(queries) as client:
        resp = await client.get("/cluster/stats")
    assert resp.status_code == 200
    assert resp.json()["services_up"] == 3


async def test_cluster_stats_services_up_zero_on_empty_db(queries: Queries):
    async with _client(queries) as client:
        resp = await client.get("/cluster/stats")
    assert resp.status_code == 200
    assert resp.json()["services_up"] == 0


# --- 2. ServiceResponse carries rollback_available / build_version (D3) --------


async def test_service_response_carries_rollback_fields(queries: Queries):
    # A redeployed app: the deploy pipeline recorded previous_image + version.
    await queries.create_job(
        _row(
            "svc-redeploy",
            JobKind.service,
            JobStatus.running,
            "web-app",
            config={
                "image": "nerdit-app/web-app:3",
                "previous_image": "nerdit-app/web-app:2",
                "build_version": 3,
                "port": 3000,
            },
        )
    )
    async with _client(queries) as client:
        resp = await client.get("/services/web-app")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rollback_available"] is True
    assert body["build_version"] == 3


async def test_service_response_rollback_fields_default_off(queries: Queries):
    # A first deploy / register-only service: no rollback target, no version.
    await queries.create_job(
        _row(
            "svc-fresh",
            JobKind.service,
            JobStatus.running,
            "fresh-app",
            config={"image": "nerdit-runtime:0.1", "port": 8000},
        )
    )
    async with _client(queries) as client:
        detail = await client.get("/services/fresh-app")
        listing = await client.get("/services")
    assert detail.status_code == 200
    body = detail.json()
    assert body["rollback_available"] is False
    assert body["build_version"] is None
    # The list projection carries the same schema (agents read either).
    (item,) = listing.json()["items"]
    assert item["rollback_available"] is False
    assert item["build_version"] is None


# --- 3. GET /audit: page shape, ordering, and admin-only ------------------------


async def test_audit_page_shape_and_cursor_pagination(queries: Queries):
    for n in (1, 2, 3):
        await queries.insert_audit_log(
            action="deploy.create",
            result="ok",
            principal_id="tok-admin",
            principal_role="admin",
            target_type="service",
            target_id=f"app-{n}",
            status_code=201,
            request_id=f"req-{n}",
        )

    async with _client(queries) as client:
        resp = await client.get("/audit", params={"limit": 2})
        assert resp.status_code == 200
        page = resp.json()
        assert set(page) == {"items", "next_cursor"}
        assert len(page["items"]) == 2
        assert page["next_cursor"] is not None
        # Newest first, and every entry exposes the full audit record shape.
        assert [e["target_id"] for e in page["items"]] == ["app-3", "app-2"]
        entry = page["items"][0]
        assert {
            "id",
            "ts",
            "principal_id",
            "principal_role",
            "action",
            "target_type",
            "target_id",
            "params_redacted",
            "result",
            "status_code",
            "request_id",
            "idempotency_key",
        } <= set(entry)
        assert entry["action"] == "deploy.create"
        assert entry["result"] == "ok"
        assert entry["principal_role"] == "admin"

        # The cursor walks to the last entry, then the log is exhausted.
        resp2 = await client.get("/audit", params={"limit": 2, "cursor": page["next_cursor"]})
    assert resp2.status_code == 200
    page2 = resp2.json()
    assert [e["target_id"] for e in page2["items"]] == ["app-1"]
    assert page2["next_cursor"] is None


async def test_audit_is_admin_only(queries: Queries):
    await queries.create_api_token(
        ApiToken(
            id="tok-ro",
            name="readonly",
            role=TokenRole.readonly,
            token_hash=hash_token(RO_RAW),
        )
    )
    async with _client(queries, token=LEGACY) as client:
        denied = await client.get("/audit", headers={"Authorization": f"Bearer {RO_RAW}"})
        allowed = await client.get("/audit", headers={"Authorization": f"Bearer {LEGACY}"})
    assert denied.status_code == 403
    assert denied.json()["code"] == "forbidden"
    assert allowed.status_code == 200


# --- 4. the MCP projection includes the P6 agent tools ---------------------------


async def test_mcp_build_server_includes_p6_tools():
    pytest.importorskip("mcp")
    from nerdit.mcp import server

    mcp_server = server.build_server()
    tools = await mcp_server.list_tools()
    names = {t.name for t in tools}
    assert {"service_logs", "get_audit", "get_config", "set_config"} <= names
