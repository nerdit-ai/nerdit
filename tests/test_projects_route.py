"""Route-level tests for the P40b project noun (``/api/projects``).

A real in-memory database under the real routers (projects, secrets, services,
models, databases, deploy) and the real auth + audit middleware — the D-P40-5
judgments live in the reserve/mint/create transactions, so mocking the queries
would pin the mapping and skip the judgment. Token rows are real
``api_tokens`` rows: A and B are unscoped submitters, ``scoped`` is narrowed to
``asso``, ``ro`` is readonly, and the legacy global token is the admin.
"""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import DatabasesSettings, ModelsSettings, ServicesSettings
from nerdit.core.data.backend import PostgresBackend
from nerdit.core.data.controller import DataController
from nerdit.core.models.backend import OllamaBackend
from nerdit.core.models.controller import ModelController
from nerdit.core.secrets import SecretManager
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.databases import router as databases_router
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.routes.models import router as models_router
from nerdit.daemon.routes.projects import router as projects_router
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries

LEGACY = "legacy-global"  # → LEGACY_ADMIN, token_id None
A_RAW, B_RAW, SCOPED_RAW, RO_RAW = "a-raw", "b-raw", "scoped-raw", "ro-raw"
TOKEN_IDS = ("tok-a", "tok-b", "tok-scoped", "tok-ro")


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "x", "scripts": {"start": "node i.js"}}))
        zf.writestr("i.js", "")
    return buf.getvalue()


class _Env:
    """One app over one real database; `client` is an `httpx` ASGI client."""

    def __init__(self, app: FastAPI, queries: Queries, db: Database) -> None:
        self.app, self.queries, self.db = app, queries, db
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def audit(self, action: str) -> list[tuple[str | None, dict]]:
        cursor = await self.db.conn.execute(
            "SELECT target_id, params_redacted FROM audit_log WHERE action = ? ORDER BY id",
            (action,),
        )
        return [(r[0], json.loads(r[1])) for r in await cursor.fetchall()]


@pytest.fixture
async def env(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for raw, tid, role, scope in (
        (A_RAW, "tok-a", TokenRole.submitter, None),
        (B_RAW, "tok-b", TokenRole.submitter, None),
        (SCOPED_RAW, "tok-scoped", TokenRole.submitter, ["asso"]),
        (RO_RAW, "tok-ro", TokenRole.readonly, None),
    ):
        await queries.create_api_token(
            ApiToken(id=tid, name=tid, role=role, token_hash=hash_token(raw), scope_services=scope)
        )

    app = FastAPI()
    register_error_handlers(app)
    for router in (
        projects_router,
        secrets_router,
        services_router,
        models_router,
        databases_router,
        deploy_router,
    ):
        app.include_router(router)
    app.state.queries = queries
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.hostname = "node-1"
    app.state.settings = SimpleNamespace(
        daemon=SimpleNamespace(max_upload_bytes=1 << 20, upload_dir=str(tmp_path / "uploads")),
        models=ModelsSettings(),
        databases=DatabasesSettings(),
        services=ServicesSettings(),
        proxy=None,
        link=None,
    )
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime
    app.state.model_controller = ModelController(
        OllamaBackend(), MagicMock(), queries, default_backend="ollama", data_dir=str(tmp_path)
    )
    app.state.data_controller = DataController(
        PostgresBackend(),
        runtime=MagicMock(),
        queries=queries,
        default_backend="postgres",
        databases_settings=DatabasesSettings(),
        models_settings=ModelsSettings(),
        data_dir=str(tmp_path),
    )
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    e = _Env(app, queries, db)
    async with e.client:
        yield e
    await db.close()


async def _create(env: _Env, name: str, raw: str = A_RAW):
    return await env.client.post("/projects", json={"name": name}, headers=_auth(raw))


async def _svc(env: _Env, name: str, raw: str):
    return await env.client.post(
        "/services", json={"name": name, "image": "demo:1"}, headers=_auth(raw)
    )


def _no_foreign_ids(resp) -> None:
    """No token id but the caller's own may ever ride a body — here, none at all."""
    for tid in TOKEN_IDS:
        assert tid not in resp.text, (tid, resp.text)


# --- create ---------------------------------------------------------------------


async def test_create_returns_id_and_empty_addresses(env):
    resp = await _create(env, "asso")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "asso" and body["id"].startswith("prj_") and len(body["id"]) == 20
    assert body["addresses"] == [] and body["services"] == []
    _no_foreign_ids(resp)
    assert await env.audit("project.create") == [("asso", {"project": "asso"})]


async def test_duplicate_is_409_project_exists(env):
    await _create(env, "asso")
    for raw in (A_RAW, B_RAW):
        resp = await _create(env, "asso", raw)
        assert resp.status_code == 409 and resp.json()["code"] == "project.exists"
        _no_foreign_ids(resp)


@pytest.mark.parametrize("name", ["shared", "rotate-key"])
async def test_reserved_name_is_422(env, name):
    resp = await _create(env, name)
    assert resp.status_code == 422 and resp.json()["code"] == "service.reserved_name"


@pytest.mark.parametrize("name", ["Bad", "a--b", "-a", "a" * 41, "a_b", ""])
async def test_grammar_is_422_project_invalid_name(env, name):
    resp = await _create(env, name)
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "project.invalid_name"
    # 41 chars pass the legacy 63-char label rule and must still be refused here.


async def test_readonly_is_403_on_every_write(env):
    await _create(env, "asso")
    assert (await _create(env, "blog", RO_RAW)).status_code == 403
    assert (await env.client.delete("/projects/asso", headers=_auth(RO_RAW))).status_code == 403
    assert (await env.client.get("/projects/asso", headers=_auth(RO_RAW))).status_code == 200


# --- rule 1: a foreign project refuses every fresh row --------------------------


async def _assert_project_owned(resp) -> None:
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "project.owned"
    assert "nerdit projects delete asso" in body["hint"]
    _no_foreign_ids(resp)


async def test_b_post_services_inside_a_project_is_409_project_owned(env):
    await _create(env, "asso")
    await _assert_project_owned(await _svc(env, "asso", B_RAW))
    assert await env.queries.get_service_by_name("asso") is None


async def test_b_deploy_inside_a_project_is_409_before_extraction(env, monkeypatch):
    await _create(env, "asso")
    extracted = AsyncMock()
    monkeypatch.setattr("nerdit.daemon.routes.deploy.extract_upload", extracted)
    resp = await env.client.post(
        "/deploy",
        data={"name": "asso"},
        files={"archive": ("app.zip", _zip(), "application/zip")},
        headers=_auth(B_RAW),
    )
    await _assert_project_owned(resp)
    extracted.assert_not_awaited()


async def test_b_post_databases_inside_a_project_is_409_project_owned(env):
    await _create(env, "asso")
    resp = await env.client.post("/databases", json={"name": "asso"}, headers=_auth(B_RAW))
    await _assert_project_owned(resp)


async def test_b_post_models_inside_a_project_is_409_project_owned(env):
    # `POST /models` derives its label from the model ref; the project carries it.
    await _create(env, "ollama-llama3-1-8b")
    resp = await env.client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(B_RAW))
    assert resp.status_code == 409 and resp.json()["code"] == "project.owned"
    _no_foreign_ids(resp)


async def test_b_cannot_squat_a_project_on_a_foreign_live_database_or_model(env):
    """Rule 3's row predicate: model/database rows own no `projects` row, so
    without it a stranger's `POST /projects` would lock the owner out (rule 1)."""
    assert (
        await env.client.post("/databases", json={"name": "pg1"}, headers=_auth(A_RAW))
    ).status_code == 201
    assert (
        await env.client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(A_RAW))
    ).status_code == 201
    for name in ("pg1", "ollama-llama3-1-8b"):
        resp = await _create(env, name, B_RAW)
        assert resp.status_code == 409 and resp.json()["code"] == "service.name_taken", resp.text
        _no_foreign_ids(resp)
        assert (await env.client.get(f"/projects/{name}", headers=_auth(A_RAW))).status_code == 404
    # The row's owner may still name its own project.
    assert (await _create(env, "pg1", A_RAW)).status_code == 201


async def test_owner_and_admin_may_add_rows_and_service_response_carries_the_triple(env):
    create = (await _create(env, "asso")).json()
    resp = await _svc(env, "asso", A_RAW)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert (body["project"], body["project_id"], body["service"]) == ("asso", create["id"], "web")
    assert body["name"] == "asso"
    # Admin bypasses a foreign project (its own fresh label inside B's project).
    await _create(env, "blog", B_RAW)
    assert (await _svc(env, "blog", LEGACY)).status_code == 201


async def test_null_owner_project_is_admin_only(env):
    resp = await _create(env, "asso", LEGACY)
    assert resp.status_code == 201
    await _assert_project_owned(await _svc(env, "asso", B_RAW))
    assert (await _svc(env, "asso", LEGACY)).status_code == 201


async def test_project_outlives_its_last_service_and_keeps_refusing_b(env):
    await _create(env, "asso")
    assert (await _svc(env, "asso", A_RAW)).status_code == 201
    resp = await env.client.delete("/services/asso", headers=_auth(A_RAW))
    assert resp.status_code == 200, resp.text
    got = await env.client.get("/projects/asso", headers=_auth(A_RAW))
    assert got.status_code == 200 and got.json()["services"] == []
    await _assert_project_owned(await _svc(env, "asso", B_RAW))


# --- rules 2 and 3: the claim and the project refuse each other, no oracle ------


async def test_rule2_secret_set_inside_a_foreign_project_is_the_row_403(env):
    await _create(env, "asso", B_RAW)
    # The reference envelope: A on a row B owns.
    assert (await _svc(env, "demo", B_RAW)).status_code == 201
    row_denial = (
        await env.client.post("/secrets/demo", json={"values": {"K": "v"}}, headers=_auth(A_RAW))
    ).json()
    assert row_denial["code"] == "forbidden"
    resp = await env.client.post("/secrets/asso", json={"values": {"K": "v"}}, headers=_auth(A_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert resp.json()["message"] == row_denial["message"]
    assert resp.json()["hint"] == row_denial["hint"]
    assert await env.queries.get_secret_claim("asso") is None
    assert not env.app.state.secret_manager.exists("asso")
    _no_foreign_ids(resp)


async def test_rule3_create_project_on_a_foreign_claim_is_409_name_claimed(env):
    resp = await env.client.post("/secrets/blog", json={"values": {"K": "v"}}, headers=_auth(B_RAW))
    assert resp.status_code == 200
    resp = await _create(env, "blog", A_RAW)
    assert resp.status_code == 409 and resp.json()["code"] == "service.name_claimed"
    _no_foreign_ids(resp)
    assert await env.queries.get_project_by_name("blog") is None
    # The claimant itself may create the project (the claim survives for the deploy).
    assert (await _create(env, "blog", B_RAW)).status_code == 201
    assert (await env.queries.get_secret_claim("blog")).token_id == "tok-b"


# --- read authorization (D-P40-15) ----------------------------------------------


async def test_out_of_scope_get_is_the_scope_403_before_any_lookup(env):
    await _create(env, "blog", B_RAW)
    expected = (
        await env.client.post(
            "/secrets/blog", json={"values": {"K": "v"}}, headers=_auth(SCOPED_RAW)
        )
    ).json()
    for name in ("blog", "ghost"):  # existing and absent answer identically
        resp = await env.client.get(f"/projects/{name}", headers=_auth(SCOPED_RAW))
        assert resp.status_code == 403
        assert resp.json()["message"].replace(name, "blog") == expected["message"]
        assert resp.json()["hint"] == expected["hint"]
        _no_foreign_ids(resp)


async def test_scoped_token_sees_and_creates_only_its_scope(env):
    await _create(env, "blog", B_RAW)
    assert (await _create(env, "asso", SCOPED_RAW)).status_code == 201
    assert (await _create(env, "other", SCOPED_RAW)).status_code == 403
    resp = await env.client.get("/projects", headers=_auth(SCOPED_RAW))
    assert [p["name"] for p in resp.json()["items"]] == ["asso"]
    assert resp.json()["next_cursor"] is None
    everything = await env.client.get("/projects", headers=_auth(A_RAW))
    assert [p["name"] for p in everything.json()["items"]] == ["asso", "blog"]
    _no_foreign_ids(everything)


async def test_list_is_bounded_and_paginates_by_name(env):
    for name in ("c", "a", "b"):
        await _create(env, name)
    page1 = (await env.client.get("/projects?limit=2", headers=_auth(RO_RAW))).json()
    assert [p["name"] for p in page1["items"]] == ["a", "b"] and page1["next_cursor"]
    page2 = (
        await env.client.get(
            f"/projects?limit=2&cursor={page1['next_cursor']}", headers=_auth(RO_RAW)
        )
    ).json()
    assert [p["name"] for p in page2["items"]] == ["c"] and page2["next_cursor"] is None
    # `_w==` is base64 for a lone 0xff byte: decodes, then fails as UTF-8.
    resp = await env.client.get("/projects?cursor=_w%3D%3D", headers=_auth(RO_RAW))
    assert resp.status_code == 400


async def test_non_owner_in_scope_reads_the_project_without_variables_or_owner(env):
    await _create(env, "asso", A_RAW)
    assert (await _svc(env, "asso", A_RAW)).status_code == 201
    resp = await env.client.get("/projects/asso", headers=_auth(B_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "variables" not in body and "environment" not in resp.text
    assert body["home"] == {"hostname": "node-1", "node_id": None}
    assert [s["service"] for s in body["services"]] == ["web"]
    assert body["services"][0]["manageable"] is False
    assert body["resources"] == [] and body["addresses"] == []
    _no_foreign_ids(resp)
    assert (await env.client.get("/projects/ghost", headers=_auth(B_RAW))).status_code == 404


async def test_get_project_lists_referenced_resources_with_inventory_readiness(env):
    await _create(env, "asso")
    assert (
        await env.client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(A_RAW))
    ).status_code == 201
    # A service row whose config declares one ollama binding and one external db.
    cfg = {
        "image": "demo:1",
        "port": 8000,
        "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
        "db": {
            "ext": {"provider": "external", "url": "postgres://h/db"},
            "main": {"provider": "managed", "database": "pg1"},
        },
    }
    pg1 = await env.queries.reserve_service_for_token(
        Job(
            name="pg1",
            kind=JobKind.database,
            service_name="pg1",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="running",
            restart_policy="always",
            submitted_by_token="tok-a",
        )
    )
    await env.queries.reserve_service_for_token(
        Job(
            name="asso",
            kind=JobKind.service,
            service_name="asso",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="running",
            restart_policy="always",
            config=json.dumps(cfg),
            submitted_by_token="tok-a",
        )
    )
    body = (await env.client.get("/projects/asso", headers=_auth(A_RAW))).json()
    assert body["resources"] == [
        {
            "service": "asso",
            "type": "ai",
            "binding": "default",
            "provider": "ollama",
            "target": "llama3.1:8b",
            "ready": False,
        },  # the model row is still building
        {
            "service": "asso",
            "type": "db",
            "binding": "ext",
            "provider": "external",
            "target": None,
            "ready": None,
        },
        {
            "service": "asso",
            "type": "db",
            "binding": "main",
            "provider": "managed",
            "target": "pg1",
            "ready": True,
        },
    ]
    # The positive branches: a running model reads ready, a stopped db does not.
    model = await env.queries.get_model_by_ref("llama3.1:8b")
    await env.queries.update_job_status(model.id, JobStatus.running)
    await env.queries.update_job_status(pg1.id, JobStatus.stopped)
    body = (await env.client.get("/projects/asso", headers=_auth(A_RAW))).json()
    assert [r["ready"] for r in body["resources"]] == [True, None, False]
    # A scoped reader learns nothing about targets outside its scope: readiness
    # is null there, exactly as for a secret-backed target (security review).
    body = (await env.client.get("/projects/asso", headers=_auth(SCOPED_RAW))).json()
    assert [r["target"] for r in body["resources"]] == ["llama3.1:8b", None, "pg1"]
    assert [r["ready"] for r in body["resources"]] == [None, None, None]


# --- delete ---------------------------------------------------------------------


async def test_delete_is_owner_or_admin_on_the_project_token(env):
    await _create(env, "asso")
    resp = await env.client.delete("/projects/asso", headers=_auth(B_RAW))
    assert resp.status_code == 403 and resp.json()["code"] == "forbidden"
    assert await env.queries.get_project_by_name("asso") is not None
    # NULL-owner project: admin-only, like a NULL-owner row.
    await _create(env, "blog", LEGACY)
    assert (await env.client.delete("/projects/blog", headers=_auth(A_RAW))).status_code == 403
    assert (await env.client.delete("/projects/blog", headers=_auth(LEGACY))).status_code == 200
    assert (await env.client.delete("/projects/asso", headers=_auth(A_RAW))).json() == {
        "name": "asso",
        "deleted": [],
    }
    assert (await env.client.delete("/projects/asso", headers=_auth(A_RAW))).status_code == 404
    assert (await env.client.get("/projects/asso", headers=_auth(A_RAW))).status_code == 404
    assert await env.audit("project.delete") == [
        ("asso", {"project": "asso", "purge": ["secrets"]}),  # the 403 (denied before the sweep)
        ("blog", {"project": "blog", "purge": ["secrets"]}),
        ("blog", {"project": "blog", "purge": ["secrets"], "deleted": [], "failed": []}),
        ("asso", {"project": "asso", "purge": ["secrets"], "deleted": [], "failed": []}),
        ("asso", {"project": "asso", "purge": ["secrets"]}),
    ]


async def test_delete_with_a_failing_purge_keeps_the_row_and_rerun_succeeds(env):
    await _create(env, "asso")
    assert (await _svc(env, "asso", A_RAW)).status_code == 201
    row = await env.queries.get_service_by_name("asso")
    busy = {row.id}
    env.app.state.service_controller = SimpleNamespace(
        has_active_run=lambda job_id: job_id in busy,
        has_active_cutover=lambda job_id: False,
    )
    resp = await env.client.delete("/projects/asso", headers=_auth(A_RAW))
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "project.delete_incomplete"
    assert body["deleted"] == []
    assert [f["name"] for f in body["failed"]] == ["asso"]
    assert body["failed"][0]["code"] == "service.run_in_progress"
    _no_foreign_ids(resp)
    assert await env.queries.get_project_by_name("asso") is not None
    assert await env.queries.get_service_by_name("asso") is not None
    busy.clear()
    resp = await env.client.delete("/projects/asso", headers=_auth(A_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"name": "asso", "deleted": ["asso"]}
    assert await env.queries.get_project_by_name("asso") is None
    assert await env.queries.get_service_by_name("asso") is None
    assert (await env.audit("project.delete"))[-1][1] == {
        "project": "asso",
        "purge": ["secrets"],
        "deleted": ["asso"],
        "failed": [],
    }


async def test_delete_audit_row_names_what_went_when_the_request_ends_early(env):
    """The row reflects the sweep however the request ends, never a nested delete's params."""
    await _create(env, "asso")
    assert (await _svc(env, "asso", A_RAW)).status_code == 201
    # The project vanishes between the sweep and the final step (a concurrent owner delete).
    env.queries.delete_project_checked = AsyncMock(return_value=None)
    resp = await env.client.delete("/projects/asso", headers=_auth(A_RAW))
    assert resp.status_code == 404, resp.text
    assert await env.queries.get_service_by_name("asso") is None
    assert (await env.audit("project.delete"))[-1] == (
        "asso",
        {"project": "asso", "purge": ["secrets"], "deleted": ["asso"], "failed": []},
    )


async def test_delete_validates_purge_before_touching_anything(env):
    await _create(env, "asso")
    resp = await env.client.delete("/projects/asso?purge=bogus", headers=_auth(A_RAW))
    assert resp.status_code == 422 and resp.json()["code"] == "service.invalid_purge"
    assert await env.queries.get_project_by_name("asso") is not None
