"""Tests for the embedded app template store (P11.5)."""

from __future__ import annotations

import pytest

from nerdit.config.app_templates import (
    AppTemplateLoadError,
    app_templates_by_id,
    load_app_templates,
)
from nerdit.config.app_templates import loader as app_templates_loader
from nerdit.db.models import AppTemplateEnvVar

_EXPECTED_IDS = {"fastapi-ai-chat", "node-starter", "fastapi-api-starter", "static-site"}


def test_load_app_templates_returns_three_entries():
    templates = load_app_templates()
    assert len(templates) == 4
    assert {t.id for t in templates} == _EXPECTED_IDS


def test_o2_no_drift_invariant():
    # Every seed must point at the pinned templates repo over https, on the
    # v1.1.0 tag, with a subdir equal to its id (contracts §13 / O2).
    for template in load_app_templates():
        assert template.repo_url.startswith("https://")
        assert "github.com" in template.repo_url
        assert template.ref == "v1.1.0"
        assert template.subdir == template.id


def test_ai_hint_only_where_an_agent_needs_one():
    """The AI chat needs a model hint; static-site carries the sandbox rule
    (non-root nginx, port 8080) because the stock image is the obvious wrong
    guess. The plain starters say nothing."""
    by_id = app_templates_by_id()
    assert by_id["fastapi-ai-chat"].ai_hint is not None
    assert by_id["static-site"].ai_hint is not None
    assert "nginx" in by_id["static-site"].ai_hint
    assert by_id["node-starter"].ai_hint is None
    assert by_id["fastapi-api-starter"].ai_hint is None


def test_app_templates_by_id_cached_mapping():
    first = app_templates_by_id()
    second = app_templates_by_id()
    assert first is second  # lru_cache(maxsize=1)
    assert set(first.keys()) == _EXPECTED_IDS
    for key, template in first.items():
        assert template.id == key


def test_bad_json_raises_load_error(monkeypatch):
    class _BadResource:
        def read_text(self, encoding="utf-8"):
            return "{ not valid json"

    class _BadFiles:
        def __truediv__(self, other):
            return _BadResource()

    monkeypatch.setattr(app_templates_loader, "files", lambda _pkg: _BadFiles())
    with pytest.raises(AppTemplateLoadError):
        load_app_templates()


def test_schema_invalid_item_raises_load_error(monkeypatch):
    class _InvalidResource:
        def read_text(self, encoding="utf-8"):
            # Missing required fields (id/name/description/...) -> ValidationError.
            return '[{"id": "broken"}]'

    class _InvalidFiles:
        def __truediv__(self, other):
            return _InvalidResource()

    monkeypatch.setattr(app_templates_loader, "files", lambda _pkg: _InvalidFiles())
    with pytest.raises(AppTemplateLoadError):
        load_app_templates()


def test_env_var_defaults():
    var = AppTemplateEnvVar(name="API_KEY")
    assert var.required is False
    assert var.secret is False
    assert var.description == ""


# --- T2-T4: app-template routes (deploy router harness, clone mocked) ---------

import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from nerdit.core.gitsource import GitSourceInfo  # noqa: E402
from nerdit.core.secrets import InvalidServiceName  # noqa: E402
from nerdit.daemon.audit import AuditMiddleware, derive_action  # noqa: E402
from nerdit.daemon.auth import hash_token  # noqa: E402
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers  # noqa: E402
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware  # noqa: E402
from nerdit.daemon.routes import app_templates as app_templates_route  # noqa: E402
from nerdit.daemon.routes.app_templates import router as app_templates_router  # noqa: E402
from nerdit.db.models import (  # noqa: E402
    ApiToken,
    AppTemplate,
    AppTemplateDeployDefaults,
    Job,
    JobKind,
    JobStatus,
    TokenRole,
)

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


def _synthetic_template(**overrides) -> AppTemplate:
    """A template exercising deploy_defaults + a required env + a required secret."""
    fields = {
        "id": "synth",
        "name": "Synthetic",
        "description": "test template",
        "icon": "box",
        "category": "test",
        "repo_url": "https://github.com/nerdit-ai/nerdit-templates",
        "ref": "v1.0.0",
        "subdir": "synth",
        "deploy_defaults": AppTemplateDeployDefaults(port=5000, start="npm run start"),
        "env_schema": [
            AppTemplateEnvVar(name="API_URL", required=True, secret=False),
            AppTemplateEnvVar(name="SERVICE_KEY", required=True, secret=True),
        ],
    }
    fields.update(overrides)
    return AppTemplate(**fields)


def _queries(existing: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_by_name = AsyncMock(return_value=existing)
    q.get_model_by_ref = AsyncMock(return_value=None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    q.update_service_config = AsyncMock()
    return q


def _make_app(
    queries: AsyncMock, tmp_path, *, with_audit: bool = False, git_enabled=True
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(app_templates_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    settings.git.enabled = git_enabled
    settings.git.allowed_hosts = ["github.com"]
    settings.git.clone_timeout_s = 30
    settings.git.max_clone_bytes = 10 * 1024 * 1024
    app.state.settings = settings
    mgr = MagicMock()
    mgr.list_keys.return_value = []
    mgr.set.return_value = []
    app.state.secret_manager = mgr
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path, **kw) -> TestClient:
    return TestClient(_make_app(queries, tmp_path, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _node_context(tmp_path, *, toml: str | None = None, dockerfile: bool = False) -> Path:
    """Materialize a buildable context dir a mocked clone_source hands back."""
    ctx = tmp_path / "clone" / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    if dockerfile:
        (ctx / "Dockerfile").write_text('FROM scratch\nCMD ["true"]\n')
    else:
        (ctx / "package.json").write_text(
            json.dumps({"name": "demo", "scripts": {"start": "node index.js"}})
        )
        (ctx / "index.js").write_text("console.log('hi')")
    if toml is not None:
        (ctx / "nerdit.toml").write_text(toml)
    return ctx


def _patch_clone(monkeypatch, context_dir: Path, *, create_dest=False):
    """Patch clone_source on the route to return a fixed context dir.

    ``create_dest=True`` also materializes the generated ``dest_dir`` so a
    failure-cleanup test can assert it was rmtree'd.
    """
    captured: dict = {}

    async def _fake_clone(
        repo_url, *, ref, subdir, dest_dir, token, timeout_s, max_bytes, allowed_hosts
    ):
        captured["dest_dir"] = dest_dir
        captured["token"] = token
        if create_dest:
            Path(dest_dir).mkdir(parents=True, exist_ok=True)
        return GitSourceInfo(
            commit_sha="a" * 40, resolved_ref=ref or "v1.0.0", context_dir=context_dir
        )

    monkeypatch.setattr(app_templates_route, "clone_source", AsyncMock(side_effect=_fake_clone))
    return app_templates_route.clone_source, captured


class _FakeSecrets:
    """A stateful stand-in for SecretManager so secret writes are observable."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}

    def set(self, service: str, values: dict[str, str]) -> list[str]:
        self.store.setdefault(service, {}).update(values)
        return list(self.store[service])

    def load(self, service: str) -> dict[str, str]:
        return dict(self.store.get(service, {}))

    def list_keys(self, service: str) -> list[str]:
        return list(self.store.get(service, {}))

    def delete_key(self, service: str, key: str) -> bool:
        return self.store.get(service, {}).pop(key, None) is not None


def _patch_catalog(monkeypatch, template: AppTemplate) -> None:
    monkeypatch.setattr(app_templates_route, "app_templates_by_id", lambda: {template.id: template})
    monkeypatch.setattr(app_templates_route, "load_app_templates", lambda: [template])


# --- T2: GET list / detail / 404 ---------------------------------------------


def test_t2_list_returns_three_seeds_readonly_ok(tmp_path):
    resp = _client(_queries(), tmp_path).get("/app-templates", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    ids = {t["id"] for t in resp.json()}
    assert ids == {"fastapi-ai-chat", "node-starter", "fastapi-api-starter", "static-site"}


def test_t2_detail_returns_template(tmp_path):
    resp = _client(_queries(), tmp_path).get("/app-templates/node-starter", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == "node-starter"


def test_t2_detail_unknown_id_404(tmp_path):
    resp = _client(_queries(), tmp_path).get("/app-templates/ghost", headers=_auth(RO_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


# --- T3: precedence (request > template defaults > repo nerdit.toml) ----------


def test_t3_template_default_port_wins_over_repo_toml(tmp_path, monkeypatch):
    """body.port=None → the template default (5000) beats the repo nerdit.toml port."""
    _patch_catalog(monkeypatch, _synthetic_template())
    ctx = _node_context(tmp_path, toml='[deploy]\nname = "demo"\nport = 8080\n')
    _patch_clone(monkeypatch, ctx)
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["port"] == 5000


def test_t3_request_port_wins(tmp_path, monkeypatch):
    """An explicit body.port beats the template default."""
    _patch_catalog(monkeypatch, _synthetic_template())
    ctx = _node_context(tmp_path, toml='[deploy]\nname = "demo"\nport = 8080\n')
    _patch_clone(monkeypatch, ctx)
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={
            "name": "demo",
            "port": 7000,
            "env": {"API_URL": "u"},
            "secrets": {"SERVICE_KEY": "k"},
        },
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["port"] == 7000


def test_t3_repo_toml_start_used_when_template_omits(tmp_path, monkeypatch):
    """When the template default omits start, the repo nerdit.toml [deploy].start wins."""
    # Template with no start default and no required env (empty schema).
    template = _synthetic_template(
        deploy_defaults=AppTemplateDeployDefaults(port=5000), env_schema=[]
    )
    _patch_catalog(monkeypatch, template)
    ctx = _node_context(
        tmp_path, toml='[deploy]\nname = "demo"\nstart = "./from-toml.sh"\n', dockerfile=True
    )
    _patch_clone(monkeypatch, ctx)
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy", json={"name": "demo"}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["command"] == "./from-toml.sh"


# --- T4: env_schema enforcement, secrets write, failure cleanup ---------------


def test_t4_required_env_missing_422_names_both(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path).post(
        "/app-templates/synth/deploy", json={"name": "demo"}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "template.missing_env"
    assert "API_URL" in body["detail"]
    assert "SERVICE_KEY" in body["detail"]
    clone_mock.assert_not_awaited()  # missing-env check precedes the clone


def test_t4_required_env_sent_as_null_is_missing_422(tmp_path, monkeypatch):
    """B2: a required non-secret input sent as JSON null must fail the gate — the
    fresh-deploy null-filter would otherwise drop it and launch without it."""
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": None}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "template.missing_env"
    assert "API_URL" in body["detail"]
    clone_mock.assert_not_awaited()  # gate precedes the clone


def test_t4_optional_env_sent_as_null_is_allowed(tmp_path, monkeypatch):
    """B2: an OPTIONAL input sent as null passes the gate (dropped harmlessly on a
    fresh deploy) — only REQUIRED inputs treat null as missing."""
    template = _synthetic_template(
        env_schema=[AppTemplateEnvVar(name="OPT_URL", required=False, secret=False)]
    )
    _patch_catalog(monkeypatch, template)
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"OPT_URL": None}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text


def test_t4_satisfied_request_writes_secrets_and_masks_audit(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    app = _make_app(q, tmp_path, with_audit=True)
    mgr = app.state.secret_manager
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/app-templates/synth/deploy",
        json={
            "name": "demo",
            "env": {"API_URL": "https://api"},
            "secrets": {"SERVICE_KEY": "s3cr3t"},
        },
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    # The secret was written through the SecretManager with the right map.
    mgr.set.assert_called_once_with("demo", {"SERVICE_KEY": "s3cr3t"})
    # The response never carries the secrets.
    assert "secrets" not in resp.json()
    assert "s3cr3t" not in resp.text
    # The audit params mask the secrets map value.
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert "s3cr3t" not in logged
    params = json.loads(q.insert_audit_log.call_args.kwargs["params_redacted"])
    assert params["secrets"] == "***"


def test_t4_finalize_failure_writes_no_secrets_and_removes_dest(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    _, captured = _patch_clone(monkeypatch, _node_context(tmp_path), create_dest=True)
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=RuntimeError("boom"))
    app = _make_app(q, tmp_path)
    mgr = _FakeSecrets()
    app.state.secret_manager = mgr
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/app-templates/synth/deploy",
        json={
            "name": "demo",
            "env": {"API_URL": "u"},
            "secrets": {"SERVICE_KEY": "k"},
        },
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code >= 500
    # Secrets are written AFTER finalize's row write; a failed FRESH deploy never
    # reached the write, so the scope has no keys (nothing to clean up).
    assert mgr.load("demo") == {}
    # _finalize_deploy owns rmtree of the clone root on any failure.
    assert not Path(captured["dest_dir"]).exists()


def test_f1_create_race_cannot_write_secrets_into_foreign_scope(tmp_path, monkeypatch):
    """A create race must not let a losing principal write into another owner's scope.

    The pre-clone existing-row read sees no row, so nothing gates the clone; a
    rival principal creates ``demo`` during the clone window. _finalize_deploy
    re-reads that row and require_owner_or_admin rejects the non-owning submitter
    BEFORE any secret write, so the victim's value is never touched.
    """
    _patch_catalog(monkeypatch, _synthetic_template())
    ctx = _node_context(tmp_path)
    q = _queries()  # get_service_by_name -> None pre-clone

    raced = Job(
        id="svc-raced",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="someone-else",
        config=json.dumps({"image": "nerdit-app/demo:1"}),
    )
    captured: dict = {}

    async def _racing_clone(
        repo_url, *, ref, subdir, dest_dir, token, timeout_s, max_bytes, allowed_hosts
    ):
        captured["dest_dir"] = dest_dir
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        # Another principal wins the name during the clone window.
        q.get_service_by_name = AsyncMock(return_value=raced)
        return GitSourceInfo(commit_sha="a" * 40, resolved_ref=ref or "v1.0.0", context_dir=ctx)

    monkeypatch.setattr(app_templates_route, "clone_source", AsyncMock(side_effect=_racing_clone))

    app = _make_app(q, tmp_path)
    mgr = _FakeSecrets()
    mgr.set("demo", {"SERVICE_KEY": "victim-value"})  # pre-seed the victim scope
    app.state.secret_manager = mgr
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "evil"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "forbidden"
    # The victim's secret was never overwritten — the write never happened.
    assert mgr.load("demo")["SERVICE_KEY"] == "victim-value"
    # The clone root was removed by _finalize_deploy on the 403.
    assert not Path(captured["dest_dir"]).exists()


def test_f1_post_finalize_secret_failure_surfaces_actionable_hint(tmp_path, monkeypatch):
    """A secret-write failure after the row exists surfaces the partial state."""
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    app = _make_app(q, tmp_path)
    mgr = app.state.secret_manager
    mgr.set.side_effect = InvalidServiceName("bad service name")
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "secret.invalid_service"
    assert "nerdit secrets set demo" in body["hint"]
    # The row was created before the secret write failed (partial state surfaced).
    q.reserve_service_for_token.assert_awaited_once()


def test_t4_git_disabled_returns_git_disabled(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path, git_enabled=False).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "deploy.git_disabled"
    clone_mock.assert_not_awaited()


def test_t4_reserved_name_rejected(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "shared", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"
    clone_mock.assert_not_awaited()


def test_t4_bad_name_422_before_clone(tmp_path, monkeypatch):
    """Edge-case #10: a bad DNS-label name is a pydantic validation_error and
    never burns a clone."""
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "Bad_Name", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "validation_error"
    clone_mock.assert_not_awaited()


def test_t4_rejects_model_name_kind_mismatch(tmp_path, monkeypatch):
    """Edge-case #1: a name owned by a kind=model row is not a redeploy target."""
    model_row = Job(
        id="mdl-1",
        service_name="demo",
        name="demo",
        kind=JobKind.model,
        gpu_count=0,
        status=JobStatus.running,
        config="{}",
    )
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(model_row), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "deploy.kind_mismatch"
    clone_mock.assert_not_awaited()


def test_t4_readonly_blocked(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(RO_RAW),
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_t4_non_owner_redeploy_forbidden_pre_clone(tmp_path, monkeypatch):
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="someone-else",
        config=json.dumps({"image": "nerdit-app/demo:1"}),
    )
    _patch_catalog(monkeypatch, _synthetic_template())
    clone_mock, _ = _patch_clone(monkeypatch, _node_context(tmp_path))
    resp = _client(_queries(existing), tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 403
    clone_mock.assert_not_awaited()  # authz precedes the clone


def test_t4_derive_action_maps_template_deploy():
    action, target_type, target_id = derive_action("POST", "/app-templates/x/deploy")
    assert action == "template.deploy"
    assert target_type == "template"
    assert target_id == "x"


def test_t4_template_deploy_stamps_queued_last_deploy(tmp_path, monkeypatch):
    """P13 WP2: a template deploy seeds ``config['last_deploy']`` = queued through
    the shared ``_finalize_deploy`` (create, version 1) and the 201 body projects
    it — the queued-stamp regression pin for the template ingress."""
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    ld = cfg["last_deploy"]
    assert ld["phase"] == "queued"
    assert ld["action"] == "create"
    assert ld["version"] == 1
    assert ld["reason"] is None
    # No stale crash forensics on a fresh generation.
    for key in ("last_exit_code", "oom_killed", "last_crash_at"):
        assert key not in cfg
    # The 201 body carries the stamp (ServiceResponse.last_deploy projection).
    assert resp.json()["last_deploy"] == ld


def test_t4_source_meta_carries_template_id(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["source"]["type"] == "git"
    assert cfg["source"]["template_id"] == "synth"
    assert cfg["source"]["commit_sha"] == "a" * 40


def test_template_deploy_repoints_audit_target_to_service(tmp_path, monkeypatch):
    """PR1: derive_action targets the *template* id; the route re-points the audit
    row to (service, <app name>) while the template id stays in the params, so
    store-created projects are visible to GET /audit?target=<app>."""
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    client = TestClient(_make_app(q, tmp_path, with_audit=True), raise_server_exceptions=False)
    resp = client.post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    rows = [
        c.kwargs
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "template.deploy"
    ]
    assert rows, "expected a template.deploy audit row"
    row = rows[0]
    assert row["target_type"] == "service"
    assert row["target_id"] == "demo"
    # The template id survives in the params (per-template usage stays queryable).
    assert json.loads(row["params_redacted"])["template_id"] == "synth"


# --- G1: network-gated real-clone smoke ---------------------------------------


def _github_reachable() -> bool:
    try:
        with socket.create_connection(("github.com", 443), timeout=3):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    os.environ.get("NERDIT_NET_TESTS") != "1" or not _github_reachable(),
    reason="network-gated: set NERDIT_NET_TESTS=1 with github.com reachable",
)
async def test_g1_real_clone_smoke(tmp_path):
    """Real clone_source against the pinned public seed repo (no mocks)."""
    from nerdit.core.gitsource import GitSourceError, clone_source

    dest = tmp_path / "real-clone"
    try:
        info = await clone_source(
            "https://github.com/nerdit-ai/nerdit-templates",
            ref="v1.0.0",
            subdir="node-starter",
            dest_dir=dest,
            token=None,
            timeout_s=60,
            max_bytes=50 * 1024 * 1024,
            allowed_hosts=["github.com"],
        )
    except GitSourceError as exc:
        pytest.skip(f"seed repo/tag not published yet (Part B merge gate, O2): {exc.code}")
    assert len(info.commit_sha) == 40
    assert info.resolved_ref == "v1.0.0"
    assert not (dest / ".git").exists()  # .git stripped
    assert info.context_dir == dest / "node-starter"


def test_template_deploy_response_carries_summary_and_hints(tmp_path, monkeypatch):
    """(Agent-DX) Same shared ``_finalize_deploy`` tail as ZIP/git — one edit,
    five ingresses; the route declares no ``response_model`` either."""
    _patch_catalog(monkeypatch, _synthetic_template())
    _patch_clone(monkeypatch, _node_context(tmp_path))
    q = _queries()
    resp = _client(q, tmp_path).post(
        "/app-templates/synth/deploy",
        json={"name": "demo", "env": {"API_URL": "u"}, "secrets": {"SERVICE_KEY": "k"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body["summary"]) == {"app", "status", "version", "public_url"}
    assert body["summary"]["app"] == body["name"] == "demo"
    assert isinstance(body["hints"], list)
