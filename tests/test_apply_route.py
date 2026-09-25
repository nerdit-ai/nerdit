"""Route-level tests for P40d ``POST /api/projects/{project}/apply``.

The ``test_projects_route.py`` rig: a real in-memory database under the real
routers and the real auth + audit middleware, because the judgments apply
leans on (rule 1 on the JOINED project, the label collision, the scope
widening) live in transactions and row reads a mocked ``queries`` would skip.
``extract_upload`` / ``clone_source`` are wrapped, not replaced, so "refused
before the first build context" is an ``assert_not_awaited`` and an applied
service still runs the real deploy tail.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import nerdit.daemon.deploy_pipeline as pipeline
import nerdit.daemon.routes.deploy as deploy_mod
import nerdit.daemon.routes.projects as projects_mod
from nerdit.config.settings import ServicesSettings
from nerdit.core.gitsource import GitSourceInfo
from nerdit.core.secrets import SecretManager
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.routes.projects import router as projects_router
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.routes.workspaces import router as workspaces_router
from nerdit.daemon.secret_scope import demote_flags
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, TokenRole
from nerdit.db.queries import Queries

LEGACY = "legacy-global"  # -> LEGACY_ADMIN, token_id None
A_RAW, B_RAW, SCOPED_RAW, RO_RAW = "a-raw", "b-raw", "scoped-raw", "ro-raw"
REPO = "https://github.com/acme/asso"
SECRET_VALUE = "s3cr3t-value-never-echoed"

DECLARATION = """
[project]
name = "asso"

[services.web]
port = 8000

[services.api]
port = 9000
build_settings = { subdir = "apps/api" }
"""
_PACKAGE = json.dumps({"name": "x", "scripts": {"start": "node i.js"}})


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _tree(toml: str | None) -> dict[str, str]:
    files = {"package.json": _PACKAGE, "i.js": "", "apps/api/package.json": _PACKAGE}
    if toml is not None:
        files["nerdit.toml"] = toml
    return files


def _zip(toml: str | None = DECLARATION, extra: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, text in (_tree(toml) | (extra or {})).items():
            zf.writestr(path, text)
    return buf.getvalue()


def _fake_clone(toml: str | None = DECLARATION) -> AsyncMock:
    async def _run(repo_url, *, dest_dir, ref=None, **_kw) -> GitSourceInfo:
        for path, text in _tree(toml).items():
            target = Path(dest_dir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        return GitSourceInfo(
            commit_sha="a" * 40, resolved_ref=ref or "main", context_dir=Path(dest_dir)
        )

    return AsyncMock(side_effect=_run)


class _Env:
    def __init__(self, app: FastAPI, queries: Queries, db: Database, extract, clone) -> None:
        self.app, self.queries, self.db = app, queries, db
        self.extract, self.clone = extract, clone
        self.client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    async def apply(self, raw: str = A_RAW, *, project: str = "asso", toml=DECLARATION, **kw):
        return await self.client.post(
            f"/projects/{project}/apply",
            files={"archive": ("app.zip", _zip(toml), "application/zip")},
            headers=_auth(raw),
            **kw,
        )

    async def apply_git(self, raw: str = A_RAW, *, dry_run: bool = False, **data):
        return await self.client.post(
            "/projects/asso/apply",
            data={"repo_url": REPO, **data},
            params={"dry_run": "true"} if dry_run else None,
            headers=_auth(raw),
        )

    async def count(self, table: str) -> int:
        cursor = await self.db.conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        return (await cursor.fetchone())[0]

    async def audit(self, action: str) -> list[tuple[str | None, dict]]:
        cursor = await self.db.conn.execute(
            "SELECT target_id, params_redacted FROM audit_log WHERE action = ? ORDER BY id",
            (action,),
        )
        return [(r[0], json.loads(r[1])) for r in await cursor.fetchall()]

    def untouched(self) -> None:
        """No build context was created: neither an extraction nor a clone ran."""
        self.extract.assert_not_awaited()
        self.clone.assert_not_awaited()


@pytest.fixture
async def env(tmp_path, monkeypatch):
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
        deploy_router,
        workspaces_router,
    ):
        app.include_router(router)
    app.state.queries = queries
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.hostname = "node-1"
    app.state.settings = SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        daemon=SimpleNamespace(max_upload_bytes=1 << 20, upload_dir=str(tmp_path / "uploads")),
        git=SimpleNamespace(
            enabled=True,
            allowed_hosts=["github.com"],
            clone_timeout_s=30,
            max_clone_bytes=1 << 20,
        ),
        services=ServicesSettings(),
        proxy=None,
        link=None,
    )
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    extract = AsyncMock(side_effect=projects_mod.extract_upload)
    monkeypatch.setattr(projects_mod, "extract_upload", extract)
    legacy_extract = AsyncMock(side_effect=deploy_mod.extract_upload)
    monkeypatch.setattr(deploy_mod, "extract_upload", legacy_extract)
    clone = _fake_clone()
    monkeypatch.setattr(pipeline, "clone_source", clone)

    e = _Env(app, queries, db, extract, clone)
    async with e.client:
        yield e
    await db.close()


async def _rows(env: _Env) -> dict[str, tuple]:
    cursor = await env.db.conn.execute(
        "SELECT j.service_name, p.name, j.environment, j.service, j.submitted_by_token "
        "FROM jobs j LEFT JOIN projects p ON p.id = j.project_id ORDER BY j.service_name"
    )
    return {r[0]: tuple(r[1:]) for r in await cursor.fetchall()}


# --- the happy path -------------------------------------------------------------


async def test_two_services_get_the_labels_and_the_project_triple(env):
    resp = await env.apply()
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "applied" and body["dry_run"] is False
    assert [(s["label"], s["service"], s["action"]) for s in body["services"]] == [
        ("asso", "web", "fresh"),
        ("api--asso", "api", "fresh"),
    ]
    assert all(s["status"] == "building" for s in body["services"])
    assert [s["build_version"] for s in body["services"]] == [1, 1]
    assert body["public_urls"] == []  # no endpoint before the first launch
    # Both rows are stamped into THIS project; no implicit `api--asso` project.
    assert await _rows(env) == {
        "asso": ("asso", "production", "web", "tok-a"),
        "api--asso": ("asso", "production", "api", "tok-a"),
    }
    assert await env.count("projects") == 1
    assert env.extract.await_count == 2  # one context per service
    assert await env.audit("project.apply") == [
        (
            "asso",
            {
                "project": "asso",
                "services": ["asso", "api--asso"],
                "dry_run": False,
                "source": "archive",
            },
        )
    ]
    # The per-service build context is the declared subdir.
    api = await env.queries.get_service_by_name("api--asso")
    assert json.loads(api.config)["build_plan"]["subdir"] == "apps/api"

    shown = await env.client.get("/projects/asso", headers=_auth(A_RAW))
    assert [s["name"] for s in shown.json()["services"]] == ["api--asso", "asso"]


async def test_reapply_redeploys_both(env):
    assert (await env.apply()).status_code == 200
    resp = await env.apply()
    assert resp.status_code == 200, resp.text
    assert [s["action"] for s in resp.json()["services"]] == ["redeploy", "redeploy"]
    assert [s["build_version"] for s in resp.json()["services"]] == [2, 2]
    assert await env.count("jobs") == 2 and await env.count("projects") == 1
    for label in ("asso", "api--asso"):
        row = await env.queries.get_service_by_name(label)
        assert json.loads(row.config)["build_version"] == 2


async def test_apply_never_carries_auto_deploy_forward(env):
    # A legacy git row converted by apply: its recorded-source redeploy answers
    # `deploy.use_apply`, so an inherited flag would leave GitWatch polling it.
    assert (await env.apply()).status_code == 200
    row = await env.queries.get_service_by_name("asso")
    config = json.dumps(json.loads(row.config) | {"auto_deploy": True})
    await env.db.conn.execute("UPDATE jobs SET config = ? WHERE id = ?", (config, row.id))
    await env.db.conn.commit()
    assert (await env.apply()).status_code == 200
    row = await env.queries.get_service_by_name("asso")
    assert json.loads(row.config)["auto_deploy"] is False


async def test_git_source_clones_once_for_every_service(env, monkeypatch):
    """One clone per apply: a moving ref cannot split the services across commits."""
    clone = _fake_clone()
    shas = iter("abc")

    async def _moving_head(*args, **kwargs) -> GitSourceInfo:
        info = await clone.side_effect(*args, **kwargs)
        return GitSourceInfo(
            commit_sha=next(shas) * 40,
            resolved_ref=info.resolved_ref,
            context_dir=info.context_dir,
        )

    moving = AsyncMock(side_effect=_moving_head)
    monkeypatch.setattr(pipeline, "clone_source", moving)
    resp = await env.apply_git(ref="main")
    assert resp.status_code == 200, resp.text
    assert moving.await_count == 1
    sources = [
        json.loads((await env.queries.get_service_by_name(label)).config)["source"]
        for label in ("asso", "api--asso")
    ]
    assert {s["commit_sha"] for s in sources} == {"a" * 40}
    source = sources[1]
    assert source["type"] == "git" and source["repo_url"] == REPO and source["ref"] == "main"
    assert (await env.audit("project.apply"))[0][1]["source"] == "git"


# --- dry run --------------------------------------------------------------------


async def test_dry_run_plans_both_and_writes_nothing(env):
    resp = await env.apply(params={"dry_run": "true"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "planned" and body["dry_run"] is True
    assert [s["label"] for s in body["services"]] == ["asso", "api--asso"]
    assert all(s["plan"]["action"] == "create" for s in body["services"])
    assert all(s.get("build_version") is None for s in body["services"])
    for table in ("projects", "jobs", "secret_claims", "variables"):
        assert await env.count(table) == 0, table
    assert (await env.audit("project.apply"))[0][1] == {
        "project": "asso",
        "services": [],
        "dry_run": True,
        "source": "archive",
    }


async def test_git_dry_run_writes_nothing_and_leaves_no_clone_behind(env):
    resp = await env.apply_git(dry_run=True)
    assert resp.status_code == 200 and resp.json()["status"] == "planned", resp.text
    assert await env.count("jobs") == 0 and await env.count("projects") == 0
    assert not list(Path(env.app.state.settings.daemon.upload_dir).glob("*"))


# --- the key-name oracle --------------------------------------------------------

_NEEDS_KEY = DECLARATION + '\n[vars]\nrequired = ["API_KEY", "OTHER_KEY"]\n'


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("owner", [A_RAW, LEGACY])  # a NULL-owner project is admin-only too
async def test_non_owner_apply_is_409_project_owned_with_no_key_names(env, dry_run, owner):
    created = await env.client.post("/projects", json={"name": "asso"}, headers=_auth(owner))
    assert created.status_code == 201
    resp = await env.apply(B_RAW, toml=_NEEDS_KEY, params={"dry_run": str(dry_run).lower()})
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "project.owned"
    for leak in ("API_KEY", "OTHER_KEY", "missing", "waiting_for_variables", "tok-a"):
        assert leak not in resp.text, leak
    env.untouched()
    assert await env.count("jobs") == 0 and await env.count("secret_claims") == 0


async def test_non_owner_git_apply_is_refused_before_the_clone(env):
    await env.client.post("/projects", json={"name": "asso"}, headers=_auth(A_RAW))
    resp = await env.apply_git(B_RAW, token_ref="${secrets.GH}")
    assert resp.status_code == 409 and resp.json()["code"] == "project.owned"
    env.untouched()


@pytest.mark.parametrize("dry_run", [True, False])
async def test_owner_is_told_which_variables_are_missing(env, dry_run):
    await env.client.put(
        "/projects/asso/variables",
        json={"values": {"OTHER_KEY": SECRET_VALUE}},
        headers=_auth(A_RAW),
    )
    resp = await env.apply(toml=_NEEDS_KEY, params={"dry_run": str(dry_run).lower()})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "waiting_for_variables" and body["missing"] == ["API_KEY"]
    assert "nerdit vars set asso --secret --prompt" in body["hint"]
    assert body["services"] == [] and SECRET_VALUE not in resp.text
    env.untouched()
    assert await env.count("jobs") == 0


async def test_dry_run_on_an_absent_project_reports_missing_without_creating_it(env):
    resp = await env.apply(toml=_NEEDS_KEY, params={"dry_run": "true"})
    assert resp.json()["missing"] == ["API_KEY", "OTHER_KEY"]
    assert await env.count("projects") == 0


async def test_a_service_scope_value_satisfies_only_that_service(env):
    assert (await env.apply()).status_code == 200
    # `nerdit vars set asso/api` now reaches the composed row (the P40c ceiling
    # lifted by a row that maps the label back to this project).
    put = await env.client.put(
        "/projects/asso/variables?service=api",
        json={"values": {"API_KEY": SECRET_VALUE, "OTHER_KEY": "x"}},
        headers=_auth(A_RAW),
    )
    assert put.status_code == 200, put.text
    assert put.json()["scope"] == "production/api"
    resp = await env.apply(toml=_NEEDS_KEY)
    # `web` still launches without them, so both are still missing.
    assert resp.json()["missing"] == ["API_KEY", "OTHER_KEY"]
    await env.client.put(
        "/projects/asso/variables",
        json={"values": {"API_KEY": SECRET_VALUE, "OTHER_KEY": "x"}},
        headers=_auth(A_RAW),
    )
    assert (await env.apply(toml=_NEEDS_KEY)).json()["status"] == "applied"


async def test_a_rowless_composed_label_still_takes_no_variables(env):
    await env.client.post("/projects", json={"name": "asso"}, headers=_auth(A_RAW))
    put = await env.client.put(
        "/projects/asso/variables?service=api",
        json={"values": {"K": "v"}},
        headers=_auth(A_RAW),
    )
    assert put.status_code == 404  # the documented ceiling: deploy (apply) first


async def test_demote_flags_maps_a_composed_row_to_its_project_and_service(env):
    assert (await env.apply()).status_code == 200
    await env.client.put(
        "/projects/asso/variables?service=api",
        json={"values": {"MODE": "fast"}, "secret": False},
        headers=_auth(A_RAW),
    )
    project = await env.queries.get_project_by_name("asso")

    async def flags() -> list[tuple]:
        return [
            (f.service, f.key, f.plain) for f in await env.queries.list_variable_flags(project.id)
        ]

    assert await flags() == [("api", "MODE", True)]
    request = SimpleNamespace(app=env.app)
    await demote_flags(request, "api--asso", ["MODE"])
    assert await flags() == [("api", "MODE", False)]
    # And through the legacy route, which is the caller that matters.
    await env.client.put(
        "/projects/asso/variables?service=api",
        json={"values": {"MODE": "fast"}, "secret": False},
        headers=_auth(A_RAW),
    )
    resp = await env.client.post(
        "/secrets/api--asso", json={"values": {"MODE": SECRET_VALUE}}, headers=_auth(A_RAW)
    )
    assert resp.status_code == 200, resp.text
    assert await flags() == [("api", "MODE", False)]


# --- every refusal lands before the first build context -------------------------


def _with(section: str) -> str:
    return DECLARATION + "\n" + section


@pytest.mark.parametrize(
    ("toml", "status", "code"),
    [
        (None, 422, "project.invalid_declaration"),
        ("not = [toml", 422, "project.invalid_declaration"),
        ('[deploy]\nname = "asso"\n', 422, "project.invalid_declaration"),
        (DECLARATION.replace('"asso"', '"blog"'), 422, "project.name_mismatch"),
        (DECLARATION.replace('"asso"', '"as--so"'), 422, "project.invalid_name"),
        (_with("[services.Bad_Name]\nport = 1\n"), 422, "project.invalid_service"),
        (_with("[services.shared]\nport = 1\n"), 422, "service.reserved_name"),
        (_with("[services.bad]\nport = 0\n"), 422, "project.invalid_declaration"),
        (_with('[vars]\nrequired = ["lower"]\n'), 422, "project.invalid_declaration"),
    ],
)
async def test_declaration_refusals_create_no_context_and_no_row(env, toml, status, code):
    resp = await env.apply(toml=toml)
    assert resp.status_code == status, resp.text
    assert resp.json()["code"] == code
    env.untouched()
    assert await env.count("jobs") == 0


async def test_db_engine_gets_the_create_database_hint(env):
    resp = await env.apply(toml=_with('[db.default]\nengine = "postgres"\n'))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "project.invalid_declaration"
    assert "nerdit db create" in resp.json()["message"]
    env.untouched()


async def test_the_63_octet_label_is_refused(env):
    # A grammar-exempt legacy project of 60 chars: `api--<60>` cannot be a label.
    long_name = "a" * 60
    legacy = await env.client.post(
        "/deploy",
        files={"archive": ("a.zip", _zip(None), "application/zip")},
        data={"name": long_name},
        headers=_auth(A_RAW),
    )
    assert legacy.status_code == 201, legacy.text
    resp = await env.apply(project=long_name, toml=DECLARATION.replace("asso", long_name))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "project.label_too_long"
    env.untouched()
    assert await env.count("jobs") == 1


async def test_a_legacy_row_on_the_composed_label_is_409_name_taken(env):
    legacy = await env.client.post(
        "/deploy",
        files={"archive": ("a.zip", _zip(None), "application/zip")},
        data={"name": "api--asso"},
        headers=_auth(A_RAW),  # the caller's OWN legacy row is still not this project's
    )
    assert legacy.status_code == 201, legacy.text
    for dry_run in ("true", "false"):
        resp = await env.apply(params={"dry_run": dry_run})
        assert resp.status_code == 409, resp.text
        assert resp.json()["code"] == "service.name_taken"
    env.untouched()
    # Nothing of the apply landed: `web` was pre-flighted too, not deployed first.
    assert list(await _rows(env)) == ["api--asso"]
    assert (await _rows(env))["api--asso"][0] == "api--asso"  # its own implicit project


async def test_a_foreign_claim_on_a_composed_label_is_refused_before_any_context(env):
    claim = await env.client.post(
        "/secrets/api--asso", json={"values": {"K": "v"}}, headers=_auth(B_RAW)
    )
    assert claim.status_code == 200, claim.text
    resp = await env.apply()
    assert resp.status_code == 409 and resp.json()["code"] == "service.name_claimed"
    env.untouched()
    assert await env.count("jobs") == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": {"repo_url": REPO}, "files": {"archive": ("a.zip", b"PK", "application/zip")}},
        {"data": {"ref": "main"}},
    ],
)
async def test_exactly_one_source(env, kwargs):
    resp = await env.client.post("/projects/asso/apply", headers=_auth(A_RAW), **kwargs)
    assert resp.status_code == 422 and resp.json()["code"] == "project.apply_source"
    env.untouched()


async def test_git_declaration_refusal_happens_after_the_read_clone_and_writes_nothing(
    env, monkeypatch
):
    clone = _fake_clone(DECLARATION.replace('"asso"', '"blog"'))
    monkeypatch.setattr(pipeline, "clone_source", clone)
    resp = await env.apply_git()
    assert resp.status_code == 422 and resp.json()["code"] == "project.name_mismatch"
    assert clone.await_count == 1  # the one clone per apply
    assert await env.count("jobs") == 0 and await env.count("projects") == 0
    assert not list((Path(env.app.state.settings.daemon.upload_dir)).glob("*"))


# --- authorization --------------------------------------------------------------


async def test_readonly_is_403(env):
    resp = await env.apply(RO_RAW)
    assert resp.status_code == 403
    env.untouched()


async def test_out_of_scope_is_403_before_any_lookup(env):
    resp = await env.apply(SCOPED_RAW, project="blog", toml=DECLARATION.replace('"asso"', '"blog"'))
    assert resp.status_code == 403, resp.text
    env.untouched()
    assert await env.count("projects") == 0


async def test_a_token_scoped_to_the_project_applies_and_reads_both(env):
    resp = await env.apply(SCOPED_RAW)
    assert resp.status_code == 200, resp.text
    assert [s["label"] for s in resp.json()["services"]] == ["asso", "api--asso"]
    shown = await env.client.get("/projects/asso", headers=_auth(SCOPED_RAW))
    assert shown.status_code == 200
    assert sorted(s["name"] for s in shown.json()["services"]) == ["api--asso", "asso"]
    # The widening is by the ROW's project: the composed label is writable too.
    put = await env.client.post("/services/api--asso/stop", headers=_auth(SCOPED_RAW))
    assert put.status_code != 403, put.text
    assert (await env.apply(SCOPED_RAW)).json()["status"] == "applied"


async def test_rule_1_judges_the_joined_project_not_the_label(env):
    """A composed row cannot be slipped into a foreign project past the route."""
    from nerdit.db.models import Job, JobKind, JobStatus
    from nerdit.db.queries import ProjectOwned

    await env.client.post("/projects", json={"name": "asso"}, headers=_auth(A_RAW))
    project = await env.queries.get_project_by_name("asso")

    def job(owner: str | None, project_id: str) -> Job:
        return Job(
            kind=JobKind.service,
            service_name="api--asso",
            name="api--asso",
            status=JobStatus.building,
            config="{}",
            submitted_by_token=owner,
            project_id=project_id,
            environment="production",
            service="api",
        )

    with pytest.raises(ProjectOwned):
        await env.queries.reserve_service_for_token(job("tok-b", project.id))
    with pytest.raises(ProjectOwned):  # a vanished project refuses an admin too
        await env.queries.reserve_service_for_token(job(None, "prj_gone"), admin=True)
    landed = await env.queries.reserve_service_for_token(job("tok-a", project.id))
    assert landed.project_id == project.id and landed.service == "api"
    assert await env.count("projects") == 1  # `_stamp_project` minted no `api--asso` project


async def test_delete_project_cascades_over_composed_rows(env):
    assert (await env.apply()).status_code == 200
    resp = await env.client.delete("/projects/asso", headers=_auth(A_RAW))
    assert resp.status_code == 200, resp.text
    assert sorted(resp.json()["deleted"]) == ["api--asso", "asso"]
    assert await env.count("jobs") == 0 and await env.count("projects") == 0


# --- a failure on service N -----------------------------------------------------


async def test_a_failure_on_the_second_service_reports_the_first_as_done(env):
    toml = DECLARATION.replace("apps/api", "apps/missing")
    resp = await env.apply(toml=toml)
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "project.apply_incomplete"
    assert body["failed"] == {
        "label": "api--asso",
        "service": "api",
        "code": "deploy.invalid_build_settings",
    }
    assert [s["label"] for s in body["services"]] == ["asso"]
    assert list(await _rows(env)) == ["asso"]  # no rollback
    assert (await env.audit("project.apply"))[0][1]["services"] == ["asso"]


async def test_a_nested_nerdit_toml_under_a_declared_subdir_is_ignored(env):
    """The declaration is the spec (D-P40-12): a leftover subdir file changes nothing."""
    nested = '[deploy]\nport = 3000\nrelease = "echo nested"\n\n[ai.default]\nprovider = "nope"\n'
    resp = await env.client.post(
        "/projects/asso/apply",
        files={
            "archive": (
                "app.zip",
                _zip(extra={"apps/api/nerdit.toml": nested}),
                "application/zip",
            )
        },
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 200, resp.text
    config = json.loads((await env.queries.get_service_by_name("api--asso")).config)
    assert config["port"] == 9000 and config["build_plan"]["subdir"] == "apps/api"
    assert "release" not in config and "ai" not in config


@pytest.mark.parametrize(
    "alias", ["nerdit.toml", "./nerdit.toml", "././nerdit.toml", "NERDIT.TOML"]
)
async def test_duplicate_declaration_paths_are_rejected_before_mutation(env, alias):
    buf = io.BytesIO(_zip())
    with zipfile.ZipFile(buf, "a") as zf:
        zf.writestr(alias, DECLARATION.replace("port = 8000", "port = 8080"))
    resp = await env.client.post(
        "/projects/asso/apply",
        files={"archive": ("app.zip", buf.getvalue(), "application/zip")},
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "deploy.invalid_zip"
    assert await env.count("projects") == 0 and await env.count("jobs") == 0
    env.untouched()


# --- the legacy ingresses refuse a declaration ----------------------------------


@pytest.mark.parametrize(
    ("subdir", "alias"),
    [
        (".", "NERDIT.TOML"),
        ("apps/api", "apps/api/NERDIT.TOML"),
        ("apps/api", "apps/API/NERDIT.TOML"),
    ],
)
async def test_legacy_deploy_rejects_case_aliased_declaration(env, subdir, alias):
    resp = await env.client.post(
        "/deploy",
        files={
            "archive": (
                "app.zip",
                _zip(
                    None,
                    extra={
                        str(Path(subdir) / "nerdit.toml"): DECLARATION,
                        alias: "[deploy]\nport = 8000\n",
                    },
                ),
                "application/zip",
            )
        },
        data={"name": "asso", "build_settings": json.dumps({"subdir": subdir})},
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "deploy.invalid_zip"
    assert await env.count("projects") == 0 and await env.count("jobs") == 0


async def test_legacy_zip_deploy_of_a_declaration_is_422_use_apply(env):
    resp = await env.client.post(
        "/deploy",
        files={"archive": ("a.zip", _zip(), "application/zip")},
        data={"name": "asso"},
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "deploy.use_apply"
    assert "apply" in resp.json()["hint"]
    assert await env.count("jobs") == 0 and await env.count("projects") == 0


@pytest.mark.parametrize("toml", ['[project]\nname = "asso"\n', "[services]\n"])
async def test_legacy_git_deploy_of_a_declaration_is_422_use_apply(env, monkeypatch, toml):
    monkeypatch.setattr(pipeline, "clone_source", _fake_clone(toml))
    resp = await env.client.post(
        "/deploy/git", json={"repo_url": REPO, "name": "asso"}, headers=_auth(A_RAW)
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "deploy.use_apply"
    assert await env.count("jobs") == 0 and await env.count("projects") == 0


@pytest.mark.parametrize(
    ("root", "form"),
    [
        (None, {"build_settings": json.dumps({"subdir": "apps/api"})}),
        ('[deploy]\nbuild_settings = { subdir = "apps/api" }\n', {}),
    ],
)
async def test_legacy_deploy_of_a_declaration_in_the_selected_subdir_is_422(env, root, form):
    """`deploy.use_apply` fires wherever the effective `nerdit.toml` is a declaration."""
    resp = await env.client.post(
        "/deploy",
        files={
            "archive": (
                "a.zip",
                _zip(root, extra={"apps/api/nerdit.toml": DECLARATION}),
                "application/zip",
            )
        },
        data={"name": "legacyapp", **form},
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "deploy.use_apply"
    assert await env.count("jobs") == 0 and await env.count("projects") == 0


# --- the workspace source (P40f) ------------------------------------------------


async def _write_workspace(env: _Env, raw: str = A_RAW, toml: str = DECLARATION):
    return await env.client.put(
        "/workspaces/asso/files", json={"files": _tree(toml)}, headers=_auth(raw)
    )


async def _apply_workspace(env: _Env, raw: str = A_RAW, **kw):
    return await env.client.post(
        "/projects/asso/apply", data={"workspace": "true"}, headers=_auth(raw), **kw
    )


async def test_workspace_source_applies_the_callers_own_workspace(env):
    assert (await _write_workspace(env)).status_code == 200
    resp = await _apply_workspace(env)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "applied"
    rows = await _rows(env)
    assert set(rows) == {"asso", "api--asso"}
    assert {row[-1] for row in rows.values()} == {"tok-a"}
    assert (await env.audit("project.apply"))[-1][1]["source"] == "workspace"


async def test_workspace_source_without_a_workspace_is_404_and_builds_nothing(env):
    resp = await _apply_workspace(env)
    assert resp.status_code == 404, resp.text
    env.untouched()


async def test_workspace_source_refuses_an_admin_on_somebody_elses_workspace(env):
    """Rows and project would be the ADMIN's: refused rather than mis-owned."""
    assert (await _write_workspace(env)).status_code == 200
    resp = await _apply_workspace(env, LEGACY)
    assert resp.status_code == 403, resp.text
    assert await env.count("jobs") == 0 and await env.count("projects") == 0
    env.untouched()


async def test_workspace_source_is_refused_while_the_workspace_is_locked(env):
    from nerdit.core import workspaces as core_workspaces

    assert (await _write_workspace(env)).status_code == 200
    async with core_workspaces.workspace_lock("asso"):
        resp = await _apply_workspace(env)
    assert resp.status_code == 409 and resp.json()["code"] == "workspace.deploy_in_progress"
    env.untouched()


async def test_workspace_is_a_third_exclusive_source(env):
    resp = await env.client.post(
        "/projects/asso/apply",
        data={"workspace": "true", "repo_url": REPO},
        headers=_auth(A_RAW),
    )
    assert resp.status_code == 422 and resp.json()["code"] == "project.apply_source"


async def test_apply_immutable_id_uses_current_name_and_preserves_owner(env):
    row = await env.queries.create_project("asso", "tok-a")
    result = await env.apply(project=row.id)
    assert result.status_code == 200, result.text
    assert result.json()["project"] == "asso"
    jobs = await env.queries.list_project_services(row.id)
    assert len(jobs) == 2 and {job.project_id for job in jobs} == {row.id}
    denied = await env.apply(B_RAW, project=row.id)
    assert denied.status_code == 409 and denied.json()["code"] == "project.owned"


async def test_apply_retired_id_never_uses_same_name_replacement(env):
    original = await env.queries.create_project("asso", "tok-a")
    assert await env.queries.delete_project_checked(original.id) == []
    replacement = await env.queries.create_project("asso", "tok-a")
    for dry_run in (False, True):
        result = await env.apply(project=original.id, params={"dry_run": str(dry_run).lower()})
        assert result.status_code == 404
    assert await env.queries.list_project_services(replacement.id) == []
    env.untouched()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("toml", [DECLARATION, _NEEDS_KEY], ids=["complete", "missing"])
async def test_apply_project_id_stays_pinned_during_source_read(env, monkeypatch, dry_run, toml):
    original = await env.queries.create_project("asso", "tok-a")
    read = projects_mod._read_declaration

    async def replace_after_read(request, project, row, source):
        data = await read(request, project, row, source)
        assert await env.queries.delete_project_checked(original.id) == []
        await env.queries.create_project("asso", "tok-a")
        return data

    monkeypatch.setattr(projects_mod, "_read_declaration", replace_after_read)
    result = await env.apply(
        project=original.id, toml=toml, params={"dry_run": str(dry_run).lower()}
    )
    assert result.status_code == 404, result.text
    assert await env.count("jobs") == 0
    assert (await env.queries.get_project_by_name("asso")).id != original.id
    env.untouched()


@pytest.mark.parametrize(
    ("awaited", "toml", "dry_run"),
    [
        ("_missing_variables", _NEEDS_KEY, False),
        ("_missing_variables", _NEEDS_KEY, True),
        ("_preflight_service", DECLARATION, True),
        ("_service_context", DECLARATION, True),
        ("_finalize_deploy", DECLARATION, True),
    ],
)
async def test_apply_project_id_is_rechecked_before_early_success(
    env, monkeypatch, awaited, toml, dry_run
):
    original = await env.queries.create_project("asso", "tok-a")
    original_call = getattr(projects_mod, awaited)
    replaced = False

    async def replace_after_await(*args, **kwargs):
        nonlocal replaced
        result = await original_call(*args, **kwargs)
        if not replaced:
            assert await env.queries.delete_project_checked(original.id) == []
            await env.queries.create_project("asso", "tok-a")
            replaced = True
        return result

    monkeypatch.setattr(projects_mod, awaited, replace_after_await)
    result = await env.apply(
        project=original.id, toml=toml, params={"dry_run": str(dry_run).lower()}
    )
    assert result.status_code == 404, result.text
    assert result.json()["code"] == "not_found"
    assert "missing" not in result.json() and "services" not in result.json()
    assert await env.count("jobs") == 0
    assert (await env.queries.get_project_by_name("asso")).id != original.id
