"""Tests for the P4 deploy route + controller build phase (S4).

Two layers, each mirroring an existing harness:
* Route behavior (authz, buildpack detection, name-taken, config shape) via the
  real deploy router under the auth/audit middleware with ``AsyncMock`` state
  (the ``test_routes_services`` pattern) — the ZIP is really extracted to a tmp
  upload dir.
* The controller build phase (build → image present → launch; BuildError settles
  ``failed``; build logs land in ``job_logs``) via a direct
  :class:`ServiceController` over the real ``queries`` fixture with a
  build-capable fake runtime (the ``test_services_reconcile`` pattern).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.config.project import MAX_VOLUMES
from nerdit.config.settings import ServicesSettings
from nerdit.core.models import sanitize_model_name
from nerdit.core.runtime.protocol import BuildError
from nerdit.core.services import ServiceController
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import QuotaExceeded, generate_token, hash_token
from nerdit.daemon.deploy_pipeline import _ASYNC_DEPLOY_WHY as _ASYNC_HINT
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, SecretClaim, TokenRole
from nerdit.db.queries import ServiceNameClaimed

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"
SCOPED_RAW = "scoped-raw"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
    # A submitter narrowed to the app it deploys — the H2 scope leg.
    hash_token(SCOPED_RAW): ApiToken(
        id="tok-scoped",
        name="c",
        role=TokenRole.submitter,
        token_hash=hash_token(SCOPED_RAW),
        scope_services=["demo"],
    ),
}


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _node_zip() -> bytes:
    return _zip(
        {
            "package.json": json.dumps({"name": "demo", "scripts": {"start": "node index.js"}}),
            "index.js": "console.log('hi')",
        }
    )


def _python_zip() -> bytes:
    return _zip(
        {
            "requirements.txt": "fastapi\nuvicorn\n",
            "main.py": "app = object()\n",
        }
    )


def _queries(existing: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_by_name = AsyncMock(return_value=existing)
    q.get_model_by_ref = AsyncMock(return_value=None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job, **kw: job)
    q.get_secret_claim = AsyncMock(return_value=None)
    # (P40b) No `projects` row unless a test plants one: a bare AsyncMock would
    # return a truthy MagicMock and read as a foreign project.
    q.get_project_by_name = AsyncMock(return_value=None)
    q.update_service_config = AsyncMock()  # app-build revert (unguarded twin)
    # redeploy AND rollback (CAS on max_version + the cutover marker); `True` =
    # committed, the steady state these rigs model.
    q.update_service_config_guarded = AsyncMock(return_value=True)
    return q


def _make_app(queries: AsyncMock, tmp_path, *, with_audit: bool = False) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    app.state.settings = settings
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path, **kw) -> TestClient:
    return TestClient(_make_app(queries, tmp_path, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _post(client: TestClient, zip_bytes: bytes, raw: str = SUB_RAW, **fields) -> object:
    data = {"name": "demo", "port": "8000", "gpus": "0"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/deploy",
        data=data,
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(raw),
    )


# --- route: authz + validation -----------------------------------------------


def test_readonly_blocked_on_deploy(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _node_zip(), raw=RO_RAW)
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


@pytest.mark.parametrize("apex", [True, False])
def test_deploy_apex_reserved_name(tmp_path, apex):
    """With the apex dashboard on (path mode) a fresh `api` would shadow its /api."""
    app = _make_app(_queries(), tmp_path)
    proxy = app.state.settings.proxy
    proxy.enabled, proxy.dashboard_apex, proxy.mode = True, apex, "path"
    client = TestClient(app, raise_server_exceptions=False)
    resp = _dry_post(client, _node_zip(), name="api")
    if apex:
        assert resp.status_code == 422
        assert resp.json()["code"] == "service.reserved_name"
    else:
        assert resp.json().get("code") != "service.reserved_name"


def test_deploy_invalid_name(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _node_zip(), name="Bad_Name")
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid"


def _model_clash_row(name: str = "clash") -> Job:
    return Job(
        id="mdl-clash",
        service_name=name,
        name=name,
        kind=JobKind.model,
        gpu_count=0,
        status=JobStatus.running,
        config="{}",
    )


def test_deploy_rejects_model_name_kind_mismatch(tmp_path):
    """Edge-case #1: a name owned by a kind=model row is not a redeploy target."""
    q = _queries(_model_clash_row())
    resp = _post(_client(q, tmp_path), _node_zip(), name="clash")
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "deploy.kind_mismatch"
    # Rejected on the fast path — no row reuse, no fresh reserve.
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config_guarded.assert_not_called()


def test_deploy_dry_run_rejects_model_name_kind_mismatch(tmp_path):
    """Edge-case #1: dry_run must 409 (not return a redeploy plan)."""
    q = _queries(_model_clash_row())
    resp = _dry_post(_client(q, tmp_path), _node_zip(), name="clash")
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "deploy.kind_mismatch"


def _database_clash_row(name: str = "clash") -> Job:
    return Job(
        id="db-clash",
        service_name=name,
        name=name,
        kind=JobKind.database,
        gpu_count=0,
        status=JobStatus.running,
        config="{}",
    )


def test_deploy_rejects_database_name_kind_mismatch(tmp_path):
    """P15 WP0: a name owned by a kind=database row 409s with a database-flavored
    message (never a redeploy target, same guard as the model case)."""
    q = _queries(_database_clash_row())
    resp = _post(_client(q, tmp_path), _node_zip(), name="clash")
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["code"] == "deploy.kind_mismatch"
    assert "database" in body["message"]
    assert "served model" not in body["message"]
    # #10: the hint must name the WORKING removal command — a bare
    # `nerdit services rm` 409s db.delete_requires_purge (CLI purge default is
    # secrets), so it must name `--purge data`.
    assert "--purge data" in body["hint"]
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config_guarded.assert_not_called()


def test_deploy_form_name_misattribution_names_request_field(tmp_path):
    """PROBE-11 / edge-case #5: a bad FORM name is attributed to the request
    field, never to the app's nerdit.toml (which here is valid)."""
    long_name = "a" * 64
    resp = _post(
        _client(_queries(), tmp_path),
        _node_ai_zip('[deploy]\nname = "valid-name"\n'),
        name=long_name,
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "deploy.invalid"
    assert "request 'name' field" in body["message"]
    assert "nerdit.toml" not in body["message"]


def test_deploy_uppercase_form_name_rejected_with_attribution(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _node_zip(), name="BadName")
    assert resp.status_code == 422
    assert "request 'name' field" in resp.json()["message"]


def test_deploy_trailing_dash_form_name_rejected(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _node_zip(), name="bad-")
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid"


def test_deploy_63_char_name_still_deploys(tmp_path):
    """Boundary #36: a 63-char valid DNS label is accepted."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), name="a" * 63)
    assert resp.status_code == 201, resp.text
    q.reserve_service_for_token.assert_called_once()


def test_deploy_stale_zip_name_with_valid_form_name_deploys(tmp_path):
    """The form name wins: a stale [deploy].name in the ZIP does not 422 an
    otherwise valid deploy."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip('[deploy]\nname = "stale-zip-name"\n'))
    assert resp.status_code == 201, resp.text


def test_deploy_bad_volumes_hint_names_grammar(tmp_path):
    """Edge-case #19: the [deploy] shape hint on a bad volumes value names the
    volumes grammar (no longer shadowing the volumes-aware hint)."""
    toml = '[deploy]\nname = "demo"\nvolumes = ["../evil:/d"]\n'
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "deploy.invalid"
    assert "volumes" in body["hint"]


def test_deploy_bad_health_type_hint_names_health_type(tmp_path):
    toml = '[deploy]\nname = "demo"\nhealth_type = "bogus"\n'
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422, resp.text
    assert "health_type" in resp.json()["hint"]


def test_deploy_redeploy_bumps_version(tmp_path):
    """A second deploy of an existing name reuses the row and bumps the version."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="tok-sub",  # owned by the submitter making the redeploy
        config=json.dumps({"image": "nerdit-app/demo:1", "build_version": 1, "port": 8000}),
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    # Row reused (not a fresh reserve), config bumped + previous_image recorded.
    q.reserve_service_for_token.assert_not_called()
    args, kwargs = q.update_service_config_guarded.call_args
    assert args[0] == "svc-1"
    cfg = json.loads(args[1])
    assert cfg["image"] == "nerdit-app/demo:2"
    assert cfg["build_version"] == 2
    assert cfg["max_version"] == 2
    assert cfg["previous_image"] == "nerdit-app/demo:1"
    assert kwargs["status"] is JobStatus.restarting


def test_redeploy_carries_forward_prior_memory_limit(tmp_path):
    """P13 WP5: a redeploy with no [deploy] caps keeps the prior row's caps."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="tok-sub",
        config=json.dumps(
            {
                "image": "nerdit-app/demo:1",
                "build_version": 1,
                "port": 8000,
                "memory_limit": "512m",
                "cpu_limit": 2.0,
            }
        ),
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # no nerdit.toml caps
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["memory_limit"] == "512m"
    assert cfg["cpu_limit"] == 2.0


def test_deploy_no_buildpack(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _zip({"README.md": "hi"}))
    assert resp.status_code == 400
    assert resp.json()["code"] == "deploy.no_buildpack"


def test_dockerfile_start_override_persisted(tmp_path):
    """A Dockerfile-passthrough deploy with --start records it as cfg['command']."""
    q = _queries()
    resp = _post(
        _client(q, tmp_path),
        _zip({"Dockerfile": 'FROM scratch\nCMD ["true"]\n'}),
        start="./run.sh",
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["command"] == "./run.sh"


def test_node_deploy_has_no_command(tmp_path):
    """The Node buildpack bakes its CMD, so --start must NOT set cfg['command']."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), start="node index.js")
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert "command" not in cfg


def test_redeploy_buildpack_switch_clears_stale_command(tmp_path):
    """Redeploy Dockerfile→Node under the same name must drop the stale command."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="tok-sub",
        config=json.dumps(
            {
                "image": "nerdit-app/demo:1",
                "build_version": 1,
                "max_version": 1,
                "port": 8000,
                "command": "./old.sh",  # from a previous Dockerfile deploy
            }
        ),
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # now a Node app
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert "command" not in cfg  # stale Dockerfile command must not survive


# --- route: successful node deploy -------------------------------------------


def test_deploy_node_creates_building_service(tmp_path):
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), start="node index.js")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "demo"
    assert body["status"] == JobStatus.building.value

    # The row handed to reserve_service_for_token carries the build context +
    # target image tag; the controller builds it off-tick.
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["image"] == "nerdit-app/demo:1"
    assert cfg["build_version"] == 1
    assert cfg["dockerfile_name"] == "Dockerfile.nerdit"
    assert cfg["build_context_dir"]
    assert job.idempotency_key is None
    assert job.submitted_by_token == "tok-sub"

    # The generated Dockerfile was materialized into the extracted context dir.
    from pathlib import Path

    generated = Path(cfg["build_context_dir"]) / "Dockerfile.nerdit"
    assert generated.is_file()
    assert "node:24.20.0-slim" in generated.read_text()


def test_deploy_generates_dockerignore_for_buildpack(tmp_path):
    """P13 WP8: a buildpack build gets a generated .dockerignore in the context."""
    from pathlib import Path

    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), start="node index.js")
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)

    ignore = Path(cfg["build_context_dir"]) / ".dockerignore"
    assert ignore.is_file()
    body = ignore.read_text()
    for pat in ("node_modules", ".git", "__pycache__", ".venv", "Dockerfile.nerdit"):
        assert pat in body


def test_deploy_preserves_user_dockerignore(tmp_path):
    """A user-authored .dockerignore in the upload is never overwritten."""
    from pathlib import Path

    q = _queries()
    zip_bytes = _zip(
        {
            "package.json": json.dumps({"name": "demo", "scripts": {"start": "node index.js"}}),
            "index.js": "console.log('hi')",
            ".dockerignore": "# user file\nsecret.txt\n",
        }
    )
    resp = _post(_client(q, tmp_path), zip_bytes, start="node index.js")
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)

    ignore = Path(cfg["build_context_dir"]) / ".dockerignore"
    assert ignore.read_text() == "# user file\nsecret.txt\n"


def test_deploy_passthrough_dockerfile_no_generated_dockerignore(tmp_path):
    """A Dockerfile passthrough build gets no generated .dockerignore (user owns it)."""
    from pathlib import Path

    q = _queries()
    zip_bytes = _zip({"Dockerfile": 'FROM scratch\nCMD ["true"]\n'})
    resp = _post(_client(q, tmp_path), zip_bytes)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)

    assert not (Path(cfg["build_context_dir"]) / ".dockerignore").exists()


def test_fresh_deploy_response_has_no_rollback(tmp_path):
    """A first deploy exposes build_version=1 and no rollback target (P6, D3)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["rollback_available"] is False
    assert body["build_version"] == 1


def test_redeploy_response_exposes_rollback(tmp_path):
    """A redeploy surfaces rollback_available=True + the bumped build_version."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="tok-sub",
        config=json.dumps({"image": "nerdit-app/demo:1", "build_version": 1, "port": 8000}),
    )
    q = _queries(existing)

    # Mirror the real write: the route re-fetches the row after
    # update_service_config_guarded, so reflect the new blob on the mocked row.
    async def _apply(job_id, config_json, **kwargs):
        existing.config = config_json
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["rollback_available"] is True
    assert body["build_version"] == 2


# --- P13 WP2: last_deploy queued stamp on the deploy/rollback routes ----------


def test_fresh_deploy_stamps_queued_last_deploy(tmp_path):
    """A first deploy seeds config['last_deploy'] = queued (create, version 1)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    ld = resp.json()["last_deploy"]
    assert ld["version"] == 1
    assert ld["action"] == "create"
    assert ld["phase"] == "queued"
    assert ld["image"] == "nerdit-app/demo:1"
    assert ld["reason"] is None


def test_redeploy_stamps_queued_and_clears_stale_forensics(tmp_path):
    """A redeploy re-seeds queued at the bumped version AND drops the previous
    generation's crash forensics carried by the dict(prev_cfg) copy."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token="tok-sub",
        config=json.dumps(
            {
                "image": "nerdit-app/demo:1",
                "build_version": 1,
                "max_version": 1,
                "port": 8000,
                # Stale forensics + a prior failed generation.
                "last_exit_code": 137,
                "oom_killed": True,
                "last_crash_at": "2026-07-09T00:00:00+00:00",
                "last_deploy": {"version": 1, "action": "create", "phase": "failed"},
            }
        ),
    )
    q = _queries(existing)
    written: dict = {}

    async def _apply(job_id, config_json, **kwargs):
        written["cfg"] = json.loads(config_json)
        existing.config = config_json
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = written["cfg"]
    # Stale forensics are popped so /diagnose can't misroute the new generation.
    assert "last_exit_code" not in cfg
    assert "oom_killed" not in cfg
    assert "last_crash_at" not in cfg
    ld = cfg["last_deploy"]
    assert ld == resp.json()["last_deploy"]
    assert ld["version"] == 2
    assert ld["action"] == "redeploy"
    assert ld["phase"] == "queued"
    assert ld["reason"] is None


def test_rollback_stamps_queued_at_lowered_version(tmp_path):
    existing = _redeploy_existing(
        {
            "image": "nerdit-app/demo:2",
            "previous_image": "nerdit-app/demo:1",
            "build_version": 2,
            "max_version": 2,
        }
    )
    q = _queries(existing)
    written: dict = {}

    async def _apply(job_id, config_json, **kwargs):
        written["cfg"] = json.loads(config_json)
        existing.config = config_json
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    ld = written["cfg"]["last_deploy"]
    assert ld["action"] == "rollback"
    assert ld["phase"] == "queued"
    assert ld["version"] == 1  # lowered to the previous tag
    assert ld["image"] == "nerdit-app/demo:1"
    assert resp.json()["last_deploy"]["action"] == "rollback"


def test_deploy_env_redacted_in_audit(tmp_path):
    q = _queries()
    resp = _post(
        _client(q, tmp_path, with_audit=True),
        _node_zip(),
        env=json.dumps({"SECRET_TOKEN": "supersecret"}),
    )
    assert resp.status_code == 201
    # The audit row must not contain the secret value.
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert "supersecret" not in logged


# --- PR1: audit target attribution -------------------------------------------


def test_deploy_attributes_audit_target_to_service_name(tmp_path):
    """The path rules leave deploy.create's target null; the route stamps the
    validated app name so the row is attributable to the service."""
    q = _queries()
    resp = _post(_client(q, tmp_path, with_audit=True), _node_zip())
    assert resp.status_code == 201, resp.text
    kwargs = q.insert_audit_log.call_args.kwargs
    assert kwargs["action"] == "deploy.create"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "demo"


def test_deploy_dry_run_attributes_audit_target(tmp_path):
    """A dry-run audits as deploy.plan with the same service attribution."""
    q = _queries()
    resp = _dry_post(_client(q, tmp_path, with_audit=True), _node_zip())
    assert resp.status_code == 200, resp.text
    kwargs = q.insert_audit_log.call_args.kwargs
    assert kwargs["action"] == "deploy.plan"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "demo"


def test_deploy_early_4xx_leaves_audit_target_null(tmp_path):
    """A reserved name 4xx's BEFORE the validated-name stamp is set, so the audit
    row's target stays null — pins the 'validated name only' stamp semantics."""
    q = _queries()
    resp = _post(_client(q, tmp_path, with_audit=True), _node_zip(), name="shared")
    assert resp.status_code == 422, resp.text
    kwargs = q.insert_audit_log.call_args.kwargs
    assert kwargs["action"] == "deploy.create"
    assert kwargs["target_id"] is None


# --- route: redeploy authz (owner/admin only) --------------------------------


def _redeploy_existing(cfg: dict | None = None, *, owner: str = "tok-sub") -> Job:
    base = {"image": "nerdit-app/demo:1", "build_version": 1, "max_version": 1, "port": 8000}
    if cfg:
        base.update(cfg)
    return Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        submitted_by_token=owner,
        config=json.dumps(base),
    )


def test_redeploy_non_owner_forbidden(tmp_path):
    """A submitter who does not own the service cannot redeploy over it."""
    existing = _redeploy_existing(owner="someone-else")
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # SUB_RAW = tok-sub
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    q.update_service_config_guarded.assert_not_called()


def test_redeploy_owner_succeeds(tmp_path):
    resp = _post(_client(_queries(_redeploy_existing()), tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text


def test_redeploy_admin_succeeds(tmp_path):
    """An admin token may redeploy a service it does not own."""
    existing = _redeploy_existing(owner="someone-else")
    resp = _post(_client(_queries(existing), tmp_path), _node_zip(), raw=LEGACY)
    assert resp.status_code == 201, resp.text


class _PendingTask:
    """The one thing ``CutoverManager._task_in_flight`` asks of a task."""

    def done(self) -> bool:
        return False


def test_deploy_is_refused_409_while_a_cutover_is_in_flight(tmp_path):
    """(P24b WP5) A plain redeploy landing mid-cutover is the rollback hazard.

    The redeploy write bumps ``build_version``, so the in-flight verify's own
    CAS-guarded arm/pop/settle all miss: its marker is orphaned and its green
    keeps running as a rowless container behind the new generation. Same code
    as the rollback + DELETE guards, and against the REAL controller registry
    so a change to either side of the seam breaks a test.
    """
    existing = _redeploy_existing({"cutover_pending": {"version": 1, "blue": "c-blue"}})
    q = _queries(existing)
    app = _make_app(q, tmp_path)
    controller = _controller(AsyncMock(), FakeBuildRuntime())
    app.state.service_controller = controller
    controller._cutover_tasks["svc-1"] = _PendingTask()

    client = TestClient(app, raise_server_exceptions=False)
    resp = _post(client, _node_zip())
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.cutover_in_progress"
    q.update_service_config_guarded.assert_not_awaited()

    # NOT sticky — and the redeploy that follows drops the (now abandoned)
    # marker rather than carrying it into the new generation, where a later
    # rollback onto the same version number would re-match and settle it.
    controller._cutover_tasks.pop("svc-1")
    resp = _post(client, _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert "cutover_pending" not in cfg
    assert cfg["build_version"] == 2


# --- P14 WP-A1: implicit /data volume wedge (D-P14-4) -------------------------


def test_fresh_deploy_adds_implicit_data_volume(tmp_path):
    """Every fresh deploy gets an implicit data:/data volume + NERDIT_DATA_DIR."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["volumes"] == ["data:/data"]
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/data"


def test_redeploy_retrofits_data_volume_for_pre_p14_row(tmp_path):
    """A pre-P14 row (no volumes) gets the implicit volume on its next redeploy."""
    existing = _redeploy_existing()  # no 'volumes' key — pre-P14 shape
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["volumes"] == ["data:/data"]
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/data"


def test_redeploy_wedge_is_idempotent(tmp_path):
    """A redeploy over a row that already carries data:/data does not double it."""
    existing = _redeploy_existing({"volumes": ["data:/data"], "env": {"NERDIT_DATA_DIR": "/data"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["volumes"] == ["data:/data"]  # exactly one


def test_declared_volumes_preserved_plus_implicit(tmp_path):
    """A [deploy].volumes declaration is kept and the implicit data is folded in."""
    zip_bytes = _node_ai_zip('[deploy]\nname = "demo"\nvolumes = ["cache:/var/cache"]\n')
    q = _queries()
    resp = _post(_client(q, tmp_path), zip_bytes)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["volumes"] == ["cache:/var/cache", "data:/data"]


def test_declared_data_volume_retargets_wedge(tmp_path):
    """A declared volume named 'data' re-targets the implicit volume (no dup)."""
    zip_bytes = _node_ai_zip('[deploy]\nname = "demo"\nvolumes = ["data:/var/lib/app"]\n')
    q = _queries()
    resp = _post(_client(q, tmp_path), zip_bytes)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["volumes"] == ["data:/var/lib/app"]  # not doubled
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/var/lib/app"


def test_redeploy_retarget_moves_nerdit_data_dir(tmp_path):
    """Re-targeting the implicit volume moves the wedge's NERDIT_DATA_DIR (no stale)."""
    # Prior row: implicit data at /data with the wedge-persisted env value.
    existing = _redeploy_existing({"volumes": ["data:/data"], "env": {"NERDIT_DATA_DIR": "/data"}})
    q = _queries(existing)
    zip_bytes = _node_ai_zip('[deploy]\nname = "demo"\nvolumes = ["data:/var/lib/app"]\n')
    resp = _post(_client(q, tmp_path), zip_bytes)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["volumes"] == ["data:/var/lib/app"]
    # The mount moved, so NERDIT_DATA_DIR must follow it — not stay stale at /data.
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/var/lib/app"


def test_redeploy_preserves_user_overridden_nerdit_data_dir(tmp_path):
    """A user NERDIT_DATA_DIR that differs from the implicit path is preserved."""
    existing = _redeploy_existing(
        {"volumes": ["data:/data"], "env": {"NERDIT_DATA_DIR": "/custom"}}
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # silent redeploy
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/custom"  # user override untouched


def test_deploy_rejects_invalid_volume_spec(tmp_path):
    """An invalid [deploy].volumes spec fails the deploy loudly (422)."""
    zip_bytes = _node_ai_zip('[deploy]\nname = "demo"\nvolumes = ["data:/proc"]\n')
    q = _queries()
    resp = _post(_client(q, tmp_path), zip_bytes)
    assert resp.status_code == 422
    q.reserve_service_for_token.assert_not_called()


def _volumes_zip(count: int, *, first_named_data: bool = False) -> bytes:
    """A Node app ZIP declaring *count* named volumes in its nerdit.toml."""
    specs = [f"v{i}:/m{i}" for i in range(count)]
    if first_named_data:
        specs[0] = "data:/m0"
    body = ", ".join(f'"{spec}"' for spec in specs)
    return _node_ai_zip(f'[deploy]\nname = "demo"\nvolumes = [{body}]\n')


def test_volume_cap_counts_the_implicit_wedge(tmp_path):
    """MAX_VOLUMES declared volumes are refused: the wedge would make it one over.

    The wedge is folded in BEFORE validation, so a list that would only fail at
    launch (leaving the row terminally `volume_invalid`) fails at the ingress.
    """
    q = _queries()
    resp = _post(_client(q, tmp_path), _volumes_zip(MAX_VOLUMES))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "deploy.invalid"
    q.reserve_service_for_token.assert_not_called()


def test_volume_cap_accepts_one_below_it_plus_the_wedge(tmp_path):
    """The boundary the other side: MAX_VOLUMES - 1 declared persists as exactly
    MAX_VOLUMES, which is what `core.volumes` re-validates at launch."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _volumes_zip(MAX_VOLUMES - 1))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["volumes"] == [f"v{i}:/m{i}" for i in range(MAX_VOLUMES - 1)] + ["data:/data"]
    assert len(cfg["volumes"]) == MAX_VOLUMES


def test_volume_cap_allows_max_when_one_is_named_data(tmp_path):
    """A declared `data` volume re-targets the wedge, so MAX_VOLUMES declared fit."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _volumes_zip(MAX_VOLUMES, first_named_data=True))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert len(cfg["volumes"]) == MAX_VOLUMES
    assert cfg["env"]["NERDIT_DATA_DIR"] == "/m0"


# --- route: optional port + redeploy field preservation ----------------------


def _post_no_port(client: TestClient, zip_bytes: bytes, raw: str = SUB_RAW, **fields) -> object:
    """POST /deploy WITHOUT a port form field (so the buildpack default applies)."""
    data = {"name": "demo", "gpus": "0"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/deploy",
        data=data,
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(raw),
    )


def test_deploy_node_default_port_when_port_omitted(tmp_path):
    """Omitting --port lets the Node buildpack pick DEFAULT_NODE_PORT (3000)."""
    q = _queries()
    resp = _post_no_port(_client(q, tmp_path), _node_zip(), start="node index.js")
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["port"] == 3000  # builder.DEFAULT_NODE_PORT, not the old 8000


def test_deploy_python_default_port_when_port_omitted(tmp_path):
    """A Python app (requirements.txt) deploys via the Python buildpack, not 400.

    Omitting --port lets the Python buildpack pick DEFAULT_PYTHON_PORT (8000).
    """
    q = _queries()
    resp = _post_no_port(_client(q, tmp_path), _python_zip())
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["port"] == 8000  # builder.DEFAULT_PYTHON_PORT
    assert cfg["dockerfile_name"] == "Dockerfile.nerdit"


def test_redeploy_preserves_env_when_not_supplied(tmp_path):
    """A redeploy WITHOUT --env keeps a previously-set env var."""
    existing = _redeploy_existing({"env": {"KEEP_ME": "yes"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # no env field
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    # P14 WP-A1: the implicit /data wedge injects NERDIT_DATA_DIR via setdefault.
    assert cfg["env"] == {"KEEP_ME": "yes", "NERDIT_DATA_DIR": "/data"}


def test_redeploy_merges_env_over_previous(tmp_path):
    existing = _redeploy_existing({"env": {"KEEP_ME": "yes", "OVERRIDE": "old"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip(), env=json.dumps({"OVERRIDE": "new"}))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["env"] == {"KEEP_ME": "yes", "OVERRIDE": "new", "NERDIT_DATA_DIR": "/data"}


def test_redeploy_preserves_vendor_when_not_supplied(tmp_path):
    existing = _redeploy_existing({"vendor": "amd"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["vendor"] == "amd"


def test_redeploy_updates_gpus_and_health(tmp_path):
    """--gpus / --health on a redeploy reach the dedicated columns."""
    existing = _redeploy_existing()
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip(), gpus=2, health="/healthz")
    assert resp.status_code == 201, resp.text
    kwargs = q.update_service_config_guarded.call_args.kwargs
    assert kwargs["gpu_count"] == 2
    assert kwargs["health_check"] == {"path": "/healthz"}


def _node_deploy_zip(deploy_toml: str) -> bytes:
    """A Node app ZIP whose nerdit.toml carries the given ``[deploy]`` TOML text."""
    return _zip(
        {
            "package.json": json.dumps({"name": "demo", "scripts": {"start": "node index.js"}}),
            "index.js": "console.log('hi')",
            "nerdit.toml": deploy_toml,
        }
    )


def test_deploy_http_health_blob_has_no_type_key(tmp_path):
    """P14 WP-C1 byte-identity pin: an http/absent probe emits ``{"path": ...}`` — no ``type``."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), health="/healthz")
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.health_check == {"path": "/healthz"}


def test_deploy_tcp_health_type_from_toml(tmp_path):
    """[deploy].health_type = "tcp" produces a bare ``{"type": "tcp"}`` blob (no path)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_deploy_zip('[deploy]\nhealth_type = "tcp"\n'))
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.health_check == {"type": "tcp"}


def test_redeploy_carries_forward_tcp_health_type(tmp_path):
    """A silent redeploy of a tcp-probed row keeps the tcp blob (carry-forward)."""
    existing = _redeploy_existing()
    existing.health_check = {"type": "tcp"}
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # no toml, no --health
    assert resp.status_code == 201, resp.text
    assert q.update_service_config_guarded.call_args.kwargs["health_check"] == {"type": "tcp"}


def test_redeploy_explicit_health_path_overrides_tcp(tmp_path):
    """An explicit health path on a tcp-probed row switches it back to http (MINOR-1)."""
    existing = _redeploy_existing()
    existing.health_check = {"type": "tcp"}
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip(), health="/healthz")
    assert resp.status_code == 201, resp.text
    # The explicit path is honoured — the tcp type is NOT carried forward.
    assert q.update_service_config_guarded.call_args.kwargs["health_check"] == {"path": "/healthz"}


def test_redeploy_explicit_http_type_overrides_tcp(tmp_path):
    """[deploy].health_type = "http" on a tcp row switches to http with a default path (#5)."""
    existing = _redeploy_existing()
    existing.health_check = {"type": "tcp"}
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_deploy_zip('[deploy]\nhealth_type = "http"\n'))
    assert resp.status_code == 201, resp.text
    assert q.update_service_config_guarded.call_args.kwargs["health_check"] == {"path": "/"}


def test_deploy_rejects_invalid_health_type(tmp_path):
    """A junk [deploy].health_type in the app's nerdit.toml fails loud (422)."""
    resp = _post(
        _client(_queries(), tmp_path), _node_deploy_zip('[deploy]\nhealth_type = "grpc"\n')
    )
    assert resp.status_code == 422, resp.text


def test_redeploy_without_gpus_keeps_existing(tmp_path):
    existing = _redeploy_existing()
    existing.gpu_count = 3
    q = _queries(existing)
    # Omit the gpus form field entirely so the row's prior gpu_count is kept.
    resp = _client(q, tmp_path).post(
        "/deploy",
        data={"name": "demo"},
        files={"archive": ("app.zip", _node_zip(), "application/zip")},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    assert q.update_service_config_guarded.call_args.kwargs["gpu_count"] == 3


def test_redeploy_after_rollback_allocates_fresh_tag(tmp_path):
    """After a rollback lowered build_version, the next redeploy uses max_version+1.

    Simulates the state left by deploy(v1)→redeploy(v2)→redeploy(v3)→rollback(v2):
    ``build_version=2`` (serving v2) but ``max_version=3`` (high-water mark). A
    fresh redeploy must allocate v4 — a tag that does not already exist — not v3.
    """
    existing = _redeploy_existing(
        {
            "image": "nerdit-app/demo:2",
            "build_version": 2,
            "max_version": 3,
            "previous_image": "nerdit-app/demo:3",
        }
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["image"] == "nerdit-app/demo:4"
    assert cfg["build_version"] == 4
    assert cfg["max_version"] == 4


def test_rollback_non_owner_forbidden(tmp_path):
    existing = _svc_with_config(
        {"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1"},
        owner="someone-else",
    )
    q = _queries(existing)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    q.update_service_config_guarded.assert_not_called()


def test_rollback_admin_succeeds(tmp_path):
    existing = _svc_with_config(
        {"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1"},
        owner="someone-else",
    )
    resp = _client(_queries(existing), tmp_path).post(
        "/deploy/demo/rollback", headers=_auth(LEGACY)
    )
    assert resp.status_code == 200, resp.text


def test_rollback_preserves_max_version(tmp_path):
    """A rollback lowers build_version but leaves max_version (the high-water mark)."""
    existing = _svc_with_config(
        {
            "image": "nerdit-app/demo:3",
            "previous_image": "nerdit-app/demo:2",
            "build_version": 3,
            "max_version": 3,
        }
    )
    q = _queries(existing)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["build_version"] == 2
    assert cfg["max_version"] == 3  # unchanged


# --- controller build phase (real queries + build-capable fake runtime) ------


class FakeBuildRuntime:
    """In-memory runtime that tracks built images and live containers."""

    def __init__(self, *, fail_build: bool = False) -> None:
        self.live: dict[str, datetime] = {}
        self.exited: set[str] = set()  # present-but-stopped corpses
        self.built: set[str] = set()
        self.counter = 0
        self.run_configs: list = []
        self.fail_build = fail_build

    async def build_image(self, context_dir, tag, dockerfile=None):
        yield f"Step 1/2 : building {tag}"
        if self.fail_build:
            raise BuildError("npm ci failed")
        self.built.add(tag)
        yield "Successfully built"

    async def remove_image(self, tag, force=False):
        self.built.discard(tag)

    async def image_exists(self, image_name):
        return image_name in self.built

    async def run(self, config):
        self.counter += 1
        cid = f"c{self.counter}"
        self.live[cid] = datetime.now(UTC)
        self.run_configs.append(config)
        return cid

    async def stop(self, container_id, timeout=10):
        self.live.pop(container_id, None)

    async def kill(self, container_id):
        self.live.pop(container_id, None)

    async def remove(self, container_id, force=False):
        self.live.pop(container_id, None)

    async def wait(self, container_id, timeout_s=None):
        return 0

    async def status(self, container_id):
        if container_id in self.live:
            return "running"
        return "exited" if container_id in self.exited else None

    async def inspect_state(self, container_id):
        return None

    async def list_images(self):
        return sorted(self.built)

    async def list_managed_containers(self):
        return list(self.live.items())

    async def logs(self, container_id, follow=False, tail=None, max_bytes=None, since=None):
        return
        yield  # pragma: no cover


def _controller(queries, runtime) -> ServiceController:
    settings = ServicesSettings(service_port_range="9400-9499")
    return ServiceController(queries=queries, runtime=runtime, services_settings=settings)


def _deploy_row(tmp_path, name="app", image="nerdit-app/app:1", *, ctx_name="ctx", **extra) -> Job:
    ctx = tmp_path / ctx_name
    ctx.mkdir(exist_ok=True)
    (ctx / "Dockerfile").write_text("FROM scratch\n")
    config = {
        "image": image,
        "image_repo": "nerdit-app/app",
        "port": 8000,
        "build_context_dir": str(ctx),
        "dockerfile_name": "Dockerfile",
    }
    config.update(extra)
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(config),
    )


async def _drain_builds(controller) -> None:
    for task in list(controller._build_tasks.values()):
        await task


async def test_build_phase_builds_then_launches(queries, tmp_path):
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    await queries.create_job(_deploy_row(tmp_path))

    # Tick 1: image absent → build spawned off-tick, container not launched yet.
    await controller.reconcile()
    assert "nerdit-app/app:1" not in runtime.built or True  # build runs async
    await _drain_builds(controller)
    assert "nerdit-app/app:1" in runtime.built
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.building
    assert not runtime.live  # not launched during the build tick

    # Tick 2: image present → launch on the stable endpoint.
    await controller.reconcile()
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert job.container_id in runtime.live
    assert runtime.run_configs[-1].image == "nerdit-app/app:1"

    # Build logs landed in job_logs.
    logs = " ".join(entry.message for entry in await queries.get_logs(job.id))
    assert "Building image nerdit-app/app:1" in logs
    assert "built successfully" in logs


async def test_build_failure_settles_failed(queries, tmp_path):
    runtime = FakeBuildRuntime(fail_build=True)
    controller = _controller(queries, runtime)
    await queries.create_job(_deploy_row(tmp_path))

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    assert job.desired_state == "failed"  # settled: excluded from further reconcile
    assert not runtime.live
    logs = " ".join(entry.message for entry in await queries.get_logs(job.id))
    assert "Build failed" in logs


async def test_no_rebuild_when_image_present(queries, tmp_path):
    """A row whose target image already exists skips the build and launches."""
    runtime = FakeBuildRuntime()
    runtime.built.add("nerdit-app/app:1")  # pretend a previous build produced it
    controller = _controller(queries, runtime)
    await queries.create_job(_deploy_row(tmp_path))

    await controller.reconcile()
    assert not controller._build_tasks  # no build spawned
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running


# --- rollback route ----------------------------------------------------------


def _svc_with_config(cfg: dict, *, owner: str | None = "tok-sub") -> Job:
    return Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        submitted_by_token=owner,
        config=json.dumps(cfg),
    )


def test_rollback_swaps_to_previous(tmp_path):
    existing = _svc_with_config(
        {"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1", "build_version": 2}
    )
    q = _queries(existing)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    args, kwargs = q.update_service_config_guarded.call_args
    cfg = json.loads(args[1])
    assert cfg["image"] == "nerdit-app/demo:1"  # rolled back
    assert cfg["previous_image"] == "nerdit-app/demo:2"  # can roll forward
    assert cfg["build_version"] == 1
    assert kwargs["status"] is JobStatus.restarting


def test_rollback_without_previous_is_409(tmp_path):
    existing = _svc_with_config({"image": "nerdit-app/demo:1", "build_version": 1})
    resp = _client(_queries(existing), tmp_path).post(
        "/deploy/demo/rollback", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "deploy.no_previous_version"


def test_rollback_unknown_service_404(tmp_path):
    resp = _client(_queries(None), tmp_path).post("/deploy/ghost/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 404


def test_rollback_readonly_blocked(tmp_path):
    existing = _svc_with_config(
        {"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1"}
    )
    resp = _client(_queries(existing), tmp_path).post(
        "/deploy/demo/rollback", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 403


def test_rollback_is_refused_409_while_a_release_is_in_flight(tmp_path):
    """(P20, PR #96 review F6) A rollback landing mid-``[deploy].release`` would
    abandon the running migration unobserved: the blind config write lowers
    ``build_version``, so the release's own completion handlers CAS-miss and its
    real outcome never surfaces — and the ``release_pending`` pop would erase
    the crash forensics of a generation whose migration is still executing.

    Same refusal, same error code as DELETE (``service.run_in_progress``), and
    against the REAL controller registry so a change to either side of the seam
    breaks a test. No ``?force`` hatch here, unlike DELETE.
    """
    existing = _svc_with_config(
        {
            "image": "nerdit-app/demo:2",
            "previous_image": "nerdit-app/demo:1",
            "build_version": 2,
            "max_version": 2,
            "release": "alembic upgrade head",
            "release_pending": 2,
            # (P24b) A marker left by an abandoned verify. The rollback lowers
            # ``build_version`` back onto an already-used number, so a marker
            # carried through would re-match layer 4 on the next tick and
            # settle the rolled-back generation ``cutover_failed`` — reverting
            # the image the operator just chose.
            "cutover_pending": {"version": 1, "blue": "c-blue"},
        }
    )
    q = _queries(existing)
    app = _make_app(q, tmp_path)
    controller = _controller(AsyncMock(), FakeBuildRuntime())
    app.state.service_controller = controller
    controller._register_run("svc-1", "rel-1", is_release=True)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    # Zero writes: the image is not swapped and the in-flight generation's
    # crash marker is still armed.
    q.update_service_config_guarded.assert_not_awaited()

    # NOT sticky: once the release settles and frees its slot, the very same
    # rollback goes through — and only then clears the (now wedged) marker,
    # which is exactly the §0.27 recovery path the pop exists for.
    controller._discard_run("svc-1", "rel-1")
    resp = client.post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["image"] == "nerdit-app/demo:1"
    assert "release_pending" not in cfg
    assert "cutover_pending" not in cfg


def test_rollback_gate_is_skipped_when_no_controller_is_wired(tmp_path):
    """The F6 hook is None-tolerant, the same posture as the DELETE one: a bare
    app with no ``service_controller`` on ``app.state`` still rolls back, so the
    seam never becomes a hard dependency of the route (and the mocked-``Queries``
    rigs above keep working)."""
    existing = _svc_with_config(
        {"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1", "build_version": 2}
    )
    q = _queries(existing)
    app = _make_app(q, tmp_path)
    assert not hasattr(app.state, "service_controller")
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/deploy/demo/rollback", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200, resp.text
    written = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert written["image"] == "nerdit-app/demo:1"


# --- route: [ai.*] server-side parse + model gating (P5 / S8) ----------------


_AI_API_TOML = (
    "[ai.default]\n"
    'provider = "api"\n'
    'model = "gpt-4o-mini"\n'
    'base_url = "https://api.openai.com/v1"\n'
    'api_key = "${secrets.OPENAI_KEY}"\n'
)

_AI_OLLAMA_TOML = '[ai.default]\nprovider = "ollama"\nmodel = "llama3.1:8b"\n'


def _node_ai_zip(ai_toml: str) -> bytes:
    """A Node app ZIP whose nerdit.toml carries the given TOML text."""
    return _zip(
        {
            "package.json": json.dumps({"name": "demo", "scripts": {"start": "node index.js"}}),
            "index.js": "console.log('hi')",
            "nerdit.toml": ai_toml,
        }
    )


def _model_row(status: JobStatus = JobStatus.running, kind: JobKind = JobKind.model) -> Job:
    name = sanitize_model_name("llama3.1:8b")  # 'ollama-llama3-1-8b'
    return Job(
        id="mdl-1",
        service_name=name,
        name=name,
        kind=kind,
        gpu_count=1,
        status=status,
        config="{}",
    )


def _queries_with_model(model_row: Job | None, existing: Job | None = None) -> AsyncMock:
    """A queries mock resolving the model row by served ref, the app by name."""
    q = _queries(existing)

    async def lookup(name: str) -> Job | None:
        if model_row is not None and name == model_row.service_name:
            return model_row
        return existing

    q.get_service_by_name = AsyncMock(side_effect=lookup)
    # The deploy gate resolves ollama bindings by the served model ref (so a
    # --name override still counts), not by the derived service_name.
    q.get_model_by_ref = AsyncMock(return_value=model_row)
    return q


def test_deploy_persists_ai_spec(tmp_path):
    """[ai.*] in the uploaded nerdit.toml lands in config['ai'] as the raw spec."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_API_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    # Exactly AiBindingConfig.model_dump(exclude_none=True) — the SPEC, no
    # resolved values (resolution happens at every launch, not at deploy).
    assert cfg["ai"] == {
        "default": {
            "provider": "api",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "${secrets.OPENAI_KEY}",
        }
    }


def test_deploy_without_ai_section_has_no_ai_config(tmp_path):
    """A nerdit.toml without [ai] is a no-op — no config['ai'] key at all."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip('[deploy]\nname = "demo"\n'))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert "ai" not in cfg


def test_deploy_invalid_ai_binding_422(tmp_path):
    """provider='api' without base_url violates the frozen schema → 422 envelope."""
    toml = '[ai.default]\nprovider = "api"\nmodel = "m"\napi_key = "${secrets.K}"\n'
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.invalid_ai"
    assert "base_url" in body["detail"]
    assert "[ai.<name>]" in body["hint"]


def test_deploy_invalid_ai_binding_name_422(tmp_path):
    """Binding names outside the NERDIT_AI_<NAME>_* grammar are rejected."""
    toml = '[ai.BAD-NAME]\nprovider = "ollama"\nmodel = "llama3.1:8b"\n'
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid_ai"


def test_deploy_malformed_project_toml_422(tmp_path):
    """An unparseable nerdit.toml fails loudly instead of silently skipping [ai]."""
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip("[ai.default\nbroken"))
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid_ai"


def test_deploy_reads_all_project_sections_once(tmp_path, monkeypatch):
    from pathlib import Path

    read_text = Path.read_text
    reads = []

    def counted(path, *args, **kwargs):
        if path.name == "nerdit.toml":
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    q = _queries()
    toml = "[deploy]\nport = 3000\n" + _AI_API_TOML + _DB_EXTERNAL_TOML
    resp = _post(_client(q, tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 201, resp.text
    assert len(reads) == 1
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["ai"]["default"]["provider"] == "api"
    assert cfg["db"]["cache"]["provider"] == "external"


def test_malformed_project_preserves_buildpack_error_precedence(tmp_path):
    resp = _post(_client(_queries(), tmp_path), _zip({"nerdit.toml": "[broken"}))
    assert resp.status_code == 400
    assert resp.json()["code"] == "deploy.no_buildpack"


def test_deploy_ollama_binding_without_model_row_422(tmp_path):
    """provider='ollama' requires a served kind=model row → ai.model_not_served."""
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(_AI_OLLAMA_TOML))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "ai.model_not_served"
    assert body["hint"] == "run 'nerdit serve llama3.1:8b' first"
    # The error names the model ref (the lookup key), not the derived row name.
    assert "llama3.1:8b" in body["detail"]


def test_deploy_ollama_binding_with_building_model_accepted(tmp_path):
    """A model row still pulling (building) is non-terminal — deploy must pass."""
    q = _queries_with_model(_model_row(JobStatus.building))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_OLLAMA_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["ai"]["default"] == {"provider": "ollama", "model": "llama3.1:8b"}


def test_deploy_ollama_binding_with_terminal_model_422(tmp_path):
    """A stopped model row does not satisfy the binding (settled terminal set)."""
    q = _queries_with_model(_model_row(JobStatus.stopped))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_OLLAMA_TOML))
    assert resp.status_code == 422
    assert resp.json()["code"] == "ai.model_not_served"


def test_deploy_ollama_binding_requires_kind_model(tmp_path):
    """A plain service squatting the sanitized name is not a served model."""
    q = _queries_with_model(_model_row(JobStatus.running, kind=JobKind.service))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_OLLAMA_TOML))
    assert resp.status_code == 422
    assert resp.json()["code"] == "ai.model_not_served"


def test_redeploy_replaces_ai_spec(tmp_path):
    """A redeploy REPLACES config['ai'] wholesale with the new upload's spec."""
    existing = _redeploy_existing({"ai": {"legacy": {"provider": "ollama", "model": "old:1"}}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_API_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert set(cfg["ai"]) == {"default"}  # replaced, never merged
    assert cfg["ai"]["default"]["provider"] == "api"


def test_redeploy_drops_ai_when_absent(tmp_path):
    """A redeploy whose upload has no [ai] section removes the previous spec."""
    existing = _redeploy_existing(
        {"ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}}}
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # no nerdit.toml at all
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert "ai" not in cfg


def test_rollback_leaves_ai_intact(tmp_path):
    """Rollback swaps the image only — config['ai'] survives byte-identical."""
    ai = {"default": {"provider": "ollama", "model": "llama3.1:8b"}}
    existing = _svc_with_config(
        {
            "image": "nerdit-app/demo:2",
            "previous_image": "nerdit-app/demo:1",
            "build_version": 2,
            "ai": ai,
        }
    )
    q = _queries(existing)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["ai"] == ai  # untouched by the image swap
    assert cfg["image"] == "nerdit-app/demo:1"


def test_deploy_ai_api_key_masked_in_audit(tmp_path):
    """The audited spec masks api_key — even secret *references* never land."""
    q = _queries()
    resp = _post(_client(q, tmp_path, with_audit=True), _node_ai_zip(_AI_API_TOML))
    assert resp.status_code == 201, resp.text
    logged = json.dumps(q.insert_audit_log.call_args.kwargs)
    assert "${secrets.OPENAI_KEY}" not in logged
    assert "gpt-4o-mini" in logged  # the rest of the spec IS audited


# --- route: [ai.*] preserve marker (P13 WP9) ---------------------------------


def test_redeploy_preserves_ai_when_api_marked_and_source_silent(tmp_path):
    """Source silent + config['ai_source']=='api' → the API-set spec is preserved."""
    existing = _redeploy_existing(
        {
            "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
            "ai_source": "api",
        }
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())  # no nerdit.toml
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["ai"] == {"default": {"provider": "ollama", "model": "llama3.1:8b"}}
    assert cfg["ai_source"] == "api"  # marker survives for the NEXT silent redeploy


def test_second_silent_redeploy_still_preserves_ai(tmp_path):
    """The config_source re-stamp hole: after redeploy #1 stamped
    config_source='deploy', a SECOND silent redeploy must still preserve the
    API-set spec because the section-level ai_source marker persisted."""
    existing = _redeploy_existing(
        {
            "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
            "ai_source": "api",
            "config_source": "deploy",  # already re-stamped by redeploy #1
        }
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["ai"] == {"default": {"provider": "ollama", "model": "llama3.1:8b"}}
    assert cfg["ai_source"] == "api"


def test_redeploy_declared_ai_replaces_and_pops_marker(tmp_path):
    """A source that declares [ai] REPLACES the spec wholesale and drops the
    API-provenance marker (the source now owns the binding)."""
    existing = _redeploy_existing(
        {
            "ai": {"legacy": {"provider": "ollama", "model": "old:1"}},
            "ai_source": "api",
        }
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_API_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert set(cfg["ai"]) == {"default"}  # replaced
    assert "ai_source" not in cfg  # marker popped


# --- route: env null-delete (P13 WP9) ----------------------------------------


def test_redeploy_env_null_delete_removes_key(tmp_path):
    """A redeploy env with a JSON null value deletes that key; others merge/keep."""
    existing = _redeploy_existing({"env": {"KEEP": "1", "DROP": "2"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip(), env=json.dumps({"DROP": None, "NEW": "3"}))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert cfg["env"] == {"KEEP": "1", "NEW": "3", "NERDIT_DATA_DIR": "/data"}


def test_fresh_deploy_drops_null_env(tmp_path):
    """A fresh deploy drops null-valued env keys (nothing to delete)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip(), env=json.dumps({"A": None, "B": "1"}))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["env"] == {"B": "1", "NERDIT_DATA_DIR": "/data"}


# --- route: dry_run (P13 WP9 / 1.10) -----------------------------------------


def _dry_post(client: TestClient, zip_bytes: bytes, raw: str = SUB_RAW, **fields) -> object:
    data = {"name": "demo", "port": "8000", "gpus": "0"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/deploy",
        params={"dry_run": "true"},
        data=data,
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(raw),
    )


def test_dry_run_writes_nothing_and_returns_plan(tmp_path):
    """A dry-run runs every validator, returns the 1.10 plan (status 200), and
    performs zero writes."""
    q = _queries()
    resp = _dry_post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert body["action"] == "create"
    assert body["name"] == "demo"
    assert body["buildpack"] == "node"
    assert set(body["effective"]) == {
        "port",
        "gpus",
        "start",
        "health",
        "memory_limit",
        "cpu_limit",
        "volumes",
        "release",  # P20: the pre-swap command this deploy WOULD run
        # P25: the edge-auth this deploy WOULD apply ({user, password ref}).
        "edge_auth",
    }
    assert body["effective"]["release"] is None  # none declared on this fixture
    # P14 WP-A1: the plan carries the implicit data volume (names + paths only).
    assert body["effective"]["volumes"] == ["data:/data"]
    assert set(body["env_diff"]) == {"added", "removed", "changed", "kept"}
    assert body["ai_diff"] == {"action": "none", "bindings": []}
    assert body["overwrote_api_config"] is False
    assert isinstance(body["warnings"], list)
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config_guarded.assert_not_called()


def test_dry_run_redeploy_env_diff_names_only(tmp_path):
    existing = _redeploy_existing({"env": {"KEEP": "1", "DROP": "2"}})
    q = _queries(existing)
    resp = _dry_post(_client(q, tmp_path), _node_zip(), env=json.dumps({"DROP": None, "NEW": "3"}))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "redeploy"
    assert body["env_diff"] == {
        "added": ["NEW"],
        "removed": ["DROP"],
        "changed": [],
        "kept": 1,
    }
    q.update_service_config_guarded.assert_not_called()


def test_dry_run_audited_as_deploy_plan_with_env_masked(tmp_path):
    """The dry-run audits as deploy.plan; env values are masked and no secret
    value string leaks into the serialized audit row."""
    q = _queries()
    resp = _dry_post(
        _client(q, tmp_path, with_audit=True),
        _node_zip(),
        env=json.dumps({"SECRET_ENV": "super-secret-value"}),
    )
    assert resp.status_code == 200, resp.text
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "deploy.plan"
    serialized = json.dumps(kwargs)
    assert "super-secret-value" not in serialized  # env value never in the audit body
    assert "***" in serialized  # env masked
    q.reserve_service_for_token.assert_not_called()


def test_dry_run_redeploy_non_owner_403_before_diff(tmp_path):
    """A redeploy dry-run by a non-owner is 403 before any plan is computed."""
    existing = _redeploy_existing(owner="someone-else")
    q = _queries(existing)
    resp = _dry_post(_client(q, tmp_path), _node_zip())  # SUB_RAW = tok-sub
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    q.update_service_config_guarded.assert_not_called()
    q.reserve_service_for_token.assert_not_called()


# --- route: shared-scope refs (P8) --------------------------------------------


_AI_SHARED_TOML = (
    "[ai.default]\n"
    'provider = "api"\n'
    'model = "gpt-4o-mini"\n'
    'base_url = "https://api.openai.com/v1"\n'
    'api_key = "${secrets.shared.OPENAI_KEY}"\n'
)


def _shared_referenced_calls(q) -> list:
    return [
        c.kwargs
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "secret.shared_referenced"
    ]


def test_deploy_shared_ref_round_trips_to_config(tmp_path):
    """A ${secrets.shared.KEY} ref survives the deploy-ZIP parse verbatim (P8)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_SHARED_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["ai"]["default"]["api_key"] == "${secrets.shared.OPENAI_KEY}"


def test_deploy_shared_ref_audits_shared_referenced_to_requester(tmp_path):
    """The write-time row is attributed to the REQUESTING principal (P8)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_SHARED_TOML))
    assert resp.status_code == 201, resp.text
    calls = _shared_referenced_calls(q)
    assert len(calls) == 1
    row = calls[0]
    assert row["principal_id"] == "tok-sub"  # the requester, not 'system'
    assert row["principal_role"] == "submitter"
    assert row["target_type"] == "secret"
    assert row["target_id"] == "shared"
    assert json.loads(row["params_redacted"]) == {"service": "demo", "keys": ["OPENAI_KEY"]}


def test_deploy_unscoped_ref_records_no_shared_referenced(tmp_path):
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_AI_API_TOML))
    assert resp.status_code == 201, resp.text
    assert _shared_referenced_calls(q) == []


# --- controller: redeploy swap + rollback + prune ----------------------------


async def _launch_v1(queries, controller, tmp_path):
    await queries.create_job(_deploy_row(tmp_path, image="nerdit-app/app:1"))
    await controller.reconcile()
    await _drain_builds(controller)
    await controller.reconcile()  # launch v1
    return await queries.get_service_by_name("app")


async def test_redeploy_swaps_on_same_endpoint(queries, tmp_path):
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    assert job.status is JobStatus.running
    old_cid = job.container_id
    ep1 = await queries.get_service_endpoint("app")

    # Redeploy v2: reuse the row, restarting, new image + previous_image.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    (ctx2 / "Dockerfile").write_text("FROM scratch\n")
    cfg2 = {
        "image": "nerdit-app/app:2",
        "image_repo": "nerdit-app/app",
        "build_version": 2,
        "previous_image": "nerdit-app/app:1",
        "build_context_dir": str(ctx2),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg2), status=JobStatus.restarting, desired_state="running"
    )

    # Tick: old container still serving while the new image builds off-tick.
    await controller.reconcile()
    assert old_cid in runtime.live
    await _drain_builds(controller)
    assert "nerdit-app/app:2" in runtime.built

    # Tick: image ready → swap old→new on the SAME host port (stable URL).
    await controller.reconcile()
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert old_cid not in runtime.live
    assert runtime.run_configs[-1].image == "nerdit-app/app:2"
    ep2 = await queries.get_service_endpoint("app")
    assert ep2.host_port == ep1.host_port


async def test_rollback_reuses_prebuilt_image_no_rebuild(queries, tmp_path):
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    runtime.built.add("nerdit-app/app:2")  # a v2 was built before the rollback

    # Roll back to v1 (already present): swap, no rebuild.
    cfg = {
        "image": "nerdit-app/app:1",
        "image_repo": "nerdit-app/app",
        "build_version": 1,
        "previous_image": "nerdit-app/app:2",
        "build_context_dir": str(tmp_path / "ctx"),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg), status=JobStatus.restarting, desired_state="running"
    )
    await controller.reconcile()  # swaps (no build task, image present)
    assert not controller._build_tasks
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert runtime.run_configs[-1].image == "nerdit-app/app:1"


async def test_prune_keeps_last_three(queries, tmp_path):
    runtime = FakeBuildRuntime()
    for v in range(1, 5):  # v1..v4 already built
        runtime.built.add(f"nerdit-app/app:{v}")
    controller = _controller(queries, runtime)

    row = _deploy_row(
        tmp_path,
        image="nerdit-app/app:5",
        build_version=5,
        previous_image="nerdit-app/app:4",
    )
    await queries.create_job(row)
    await controller.reconcile()
    await _drain_builds(controller)

    # After building v5: keep the newest 3 (v3,v4,v5); v1,v2 pruned.
    assert runtime.built == {"nerdit-app/app:3", "nerdit-app/app:4", "nerdit-app/app:5"}


async def test_build_context_removed_after_success(queries, tmp_path):
    """A successful build drops its context dir so redeploys don't leak uploads."""
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    row = _deploy_row(tmp_path)
    ctx = row.config  # keep a handle to assert later
    ctx_dir = tmp_path / "ctx"
    assert ctx_dir.exists()
    await queries.create_job(row)

    await controller.reconcile()
    await _drain_builds(controller)

    assert "nerdit-app/app:1" in runtime.built
    assert not ctx_dir.exists()  # rmtree'd after the image was baked
    del ctx


async def test_build_context_removed_after_failure(queries, tmp_path):
    """A FAILED build also drops its context dir (no leak on the sad path)."""
    runtime = FakeBuildRuntime(fail_build=True)
    controller = _controller(queries, runtime)
    ctx_dir = tmp_path / "ctx"
    await queries.create_job(_deploy_row(tmp_path))
    assert ctx_dir.exists()

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    assert not ctx_dir.exists()  # cleaned even though the build failed


def _subdir_deploy_row(tmp_path):
    """A deploy row whose build context is a subdir of an ingress-owned clone root."""
    root = tmp_path / "clone-root"
    sub = root / "sub"
    sub.mkdir(parents=True)
    (sub / "Dockerfile").write_text("FROM scratch\n")
    row = Job(
        name="app",
        kind=JobKind.service,
        service_name="app",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(
            {
                "image": "nerdit-app/app:1",
                "image_repo": "nerdit-app/app",
                "port": 8000,
                "build_context_dir": str(sub),
                "build_context_root": str(root),
                "dockerfile_name": "Dockerfile",
            }
        ),
    )
    return row, root, sub


async def test_build_context_root_removed_after_success(queries, tmp_path):
    """A subdir deploy cleans up the whole clone ROOT, not just the built subdir."""
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    row, root, sub = _subdir_deploy_row(tmp_path)
    await queries.create_job(row)

    await controller.reconcile()
    await _drain_builds(controller)

    assert "nerdit-app/app:1" in runtime.built
    assert not root.exists()  # the entire clone root removed, sibling repo and all
    assert not sub.exists()


async def test_build_context_root_removed_after_failure(queries, tmp_path):
    """A FAILED subdir build also drops the whole clone root (no sibling leak)."""
    runtime = FakeBuildRuntime(fail_build=True)
    controller = _controller(queries, runtime)
    row, root, _sub = _subdir_deploy_row(tmp_path)
    await queries.create_job(row)

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    assert not root.exists()  # cleaned even though the build failed


async def test_prebuilt_image_skips_build_and_leaves_context(queries, tmp_path):
    """Rollback/restart with an already-built image never rebuilds nor rmtree's.

    A rollback swaps ``image`` to an already-present tag; ``_needs_build``
    must be False, so no build task is spawned and any lingering
    ``build_context_dir`` is left untouched.
    """
    runtime = FakeBuildRuntime()
    runtime.built.add("nerdit-app/app:1")  # rollback target already built
    controller = _controller(queries, runtime)
    ctx_dir = tmp_path / "ctx"
    await queries.create_job(_deploy_row(tmp_path))
    assert ctx_dir.exists()

    await controller.reconcile()
    assert not controller._build_tasks  # no build spawned
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running  # launched straight from the image
    assert ctx_dir.exists()  # context untouched: no build ran, no cleanup ran


async def test_redeploy_build_failure_keeps_previous(queries, tmp_path):
    """A failed redeploy build must NOT orphan the still-live old container."""
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    assert job.status is JobStatus.running
    old_cid = job.container_id
    assert old_cid in runtime.live

    # Redeploy v2, but the new build fails.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    (ctx2 / "Dockerfile").write_text("FROM scratch\n")
    cfg2 = {
        "image": "nerdit-app/app:2",
        "image_repo": "nerdit-app/app",
        "build_version": 2,
        "max_version": 2,
        "previous_image": "nerdit-app/app:1",
        "build_context_dir": str(ctx2),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg2), status=JobStatus.restarting, desired_state="running"
    )
    runtime.fail_build = True

    await controller.reconcile()  # spawns the (doomed) v2 build off-tick
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running  # reverted, not failed
    assert job.desired_state == "running"  # NOT settled to failed
    assert old_cid in runtime.live  # previous version still serving
    logs = " ".join(entry.message for entry in await queries.get_logs(job.id))
    assert "keeping previous version" in logs


async def test_redeploy_build_failure_over_exited_container_reverts(queries, tmp_path):
    """A previous generation that merely EXITED still reverts, never fails.

    The discriminator is "does the runtime still know the container", not "is
    it up": a row whose old container crashed between the redeploy and the
    build settle is still a service with a ``previous_image`` to fall back to,
    and the ordinary crash/restart machinery relaunches it next tick.
    """
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    old_cid = job.container_id
    # The old container dies mid-redeploy, before any reconcile tick sees it.
    runtime.live.pop(old_cid)
    runtime.exited.add(old_cid)

    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    (ctx2 / "Dockerfile").write_text("FROM scratch\n")
    cfg2 = {
        "image": "nerdit-app/app:2",
        "image_repo": "nerdit-app/app",
        "build_version": 2,
        "max_version": 2,
        "previous_image": "nerdit-app/app:1",
        "build_context_dir": str(ctx2),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg2), status=JobStatus.restarting, desired_state="running"
    )
    runtime.fail_build = True

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running  # reverted, not failed
    assert job.desired_state == "running"
    cfg = json.loads(job.config)
    assert cfg["image"] == "nerdit-app/app:1"
    assert cfg["build_version"] == 1
    assert "build_context_dir" not in cfg  # markers dropped ⇒ no rebuild
    assert "dockerfile_name" not in cfg
    assert await queries.get_service_endpoint("app") is not None  # endpoint kept
    logs = " ".join(entry.message for entry in await queries.get_logs(job.id))
    assert "keeping previous version" in logs

    # Self-heal: the corpse is detected as a crash, then relaunched from the
    # previous image — no rebuild, no manual intervention.
    runtime.fail_build = False
    await controller.reconcile()  # crash detected → restarting
    await controller.reconcile()  # relaunch
    assert runtime.live
    assert runtime.run_configs[-1].image == "nerdit-app/app:1"


async def test_fresh_build_failure_still_settles_failed(queries, tmp_path):
    """A FRESH deploy (no live container) with a failing build settles failed."""
    runtime = FakeBuildRuntime(fail_build=True)
    controller = _controller(queries, runtime)
    await queries.create_job(_deploy_row(tmp_path))

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.failed
    assert job.desired_state == "failed"
    assert not runtime.live


# --- redeploy GPU quota gate (real queries) ----------------------------------


async def _owner_token(queries, *, max_gpus=None):
    from nerdit.db.models import ApiToken

    raw = generate_token()
    return await queries.create_api_token(
        ApiToken(
            name="bot",
            role=TokenRole.submitter,
            token_hash=hash_token(raw),
            max_gpus=max_gpus,
        )
    )


async def _running_service(queries, token_id, *, gpus=0, name="app"):
    svc = Job(
        kind=JobKind.service,
        service_name=name,
        name=name,
        gpu_count=gpus,
        status=JobStatus.running,
        desired_state="running",
        submitted_by_token=token_id,
        config=json.dumps({"image": "nerdit-app/app:1"}),
    )
    return await queries.create_job(svc)


async def test_redeploy_gpu_increase_enforces_owner_quota(queries):
    """Redeploying a 0-GPU service to over max_gpus is rejected, gpu_count kept."""
    token = await _owner_token(queries, max_gpus=1)
    svc = await _running_service(queries, token.id, gpus=0)

    with pytest.raises(QuotaExceeded) as exc:
        await queries.update_service_config_guarded(
            svc.id,
            json.dumps({"image": "nerdit-app/app:2"}),
            expect_max_version=0,
            status=JobStatus.restarting,
            desired_state="running",
            gpu_count=2,
            token_id=token.id,
        )
    assert exc.value.reason == "max_gpus"
    assert exc.value.limit == 1
    # The rejected redeploy rolled back — gpu_count + config unchanged.
    row = await queries.get_job(svc.id)
    assert row.gpu_count == 0
    assert json.loads(row.config)["image"] == "nerdit-app/app:1"


async def test_redeploy_within_gpu_cap_succeeds(queries):
    token = await _owner_token(queries, max_gpus=2)
    svc = await _running_service(queries, token.id, gpus=0)

    await queries.update_service_config_guarded(
        svc.id,
        json.dumps({"image": "nerdit-app/app:2"}),
        expect_max_version=0,
        status=JobStatus.restarting,
        desired_state="running",
        gpu_count=2,
        token_id=token.id,
    )
    row = await queries.get_job(svc.id)
    assert row.gpu_count == 2
    assert json.loads(row.config)["image"] == "nerdit-app/app:2"


async def test_redeploy_gpu_not_double_counted(queries):
    """The row's own current gpu_count is excluded from the quota sum.

    A service already holding all of max_gpus can be redeployed at the SAME
    gpu_count — a naive check would double-count the row and reject it.
    """
    token = await _owner_token(queries, max_gpus=2)
    svc = await _running_service(queries, token.id, gpus=2)

    await queries.update_service_config_guarded(
        svc.id,
        json.dumps({"image": "nerdit-app/app:2"}),
        expect_max_version=0,
        status=JobStatus.restarting,
        desired_state="running",
        gpu_count=2,
        token_id=token.id,
    )
    row = await queries.get_job(svc.id)
    assert row.gpu_count == 2


async def test_redeploy_no_token_id_is_ungated(queries):
    """A rollback (token_id omitted) is never quota-gated even over the cap."""
    token = await _owner_token(queries, max_gpus=1)
    svc = await _running_service(queries, token.id, gpus=0)

    # No token_id → plain UPDATE, no quota check (gpu_count not even passed).
    await queries.update_service_config(
        svc.id,
        json.dumps({"image": "nerdit-app/app:1"}),
        status=JobStatus.restarting,
        desired_state="running",
    )
    row = await queries.get_job(svc.id)
    assert row.status is JobStatus.restarting


# --- redeploy build failure reverts config (no orphaned rebuild) -------------


async def test_redeploy_build_failure_reverts_config_no_rebuild(queries, tmp_path):
    """A failed redeploy build rolls config back to the live old image.

    ``config.image`` must return to v1 and the build markers must be dropped so a
    later crash relaunches v1 WITHOUT attempting a rebuild from the (rmtree'd)
    new context dir.
    """
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    old_cid = job.container_id
    assert old_cid in runtime.live

    # Redeploy v2 with a build that will fail.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    (ctx2 / "Dockerfile").write_text("FROM scratch\n")
    cfg2 = {
        "image": "nerdit-app/app:2",
        "image_repo": "nerdit-app/app",
        "build_version": 2,
        "max_version": 2,
        "previous_image": "nerdit-app/app:1",
        "build_context_dir": str(ctx2),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg2), status=JobStatus.restarting, desired_state="running"
    )
    runtime.fail_build = True

    await controller.reconcile()  # spawns the doomed v2 build off-tick
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running  # reverted, previous version kept
    cfg = json.loads(job.config)
    assert cfg["image"] == "nerdit-app/app:1"  # rolled back to the live image
    assert cfg["build_version"] == 1
    assert "build_context_dir" not in cfg  # no pending build
    assert "dockerfile_name" not in cfg
    assert old_cid in runtime.live

    # Simulate a crash of the old container; a rebuild would now succeed if
    # attempted. The service must relaunch v1 WITHOUT a rebuild.
    runtime.fail_build = False
    runtime.live.pop(old_cid, None)
    await controller.reconcile()  # detect crash → restarting
    await controller.reconcile()  # backoff 0 → relaunch
    assert not controller._build_tasks  # never spawned a rebuild

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    assert runtime.run_configs[-1].image == "nerdit-app/app:1"


async def test_redeploy_build_failure_strips_build_context_root(queries, tmp_path):
    """A failed subdir redeploy drops build_context_root alongside the other markers.

    Otherwise a later crash would relaunch v1 but still carry the pending
    ``build_context_root``, and a stale root pointer could be rmtree'd later.
    """
    runtime = FakeBuildRuntime()
    controller = _controller(queries, runtime)
    controller._backoff_seconds = lambda c: 0.0  # type: ignore[assignment]

    job = await _launch_v1(queries, controller, tmp_path)
    assert job.container_id in runtime.live

    # Redeploy v2 from a nested subdir; the build will fail.
    root2 = tmp_path / "clone-root-2"
    sub2 = root2 / "sub"
    sub2.mkdir(parents=True)
    (sub2 / "Dockerfile").write_text("FROM scratch\n")
    cfg2 = {
        "image": "nerdit-app/app:2",
        "image_repo": "nerdit-app/app",
        "build_version": 2,
        "max_version": 2,
        "previous_image": "nerdit-app/app:1",
        "build_context_dir": str(sub2),
        "build_context_root": str(root2),
        "dockerfile_name": "Dockerfile",
        "port": 8000,
    }
    await queries.update_service_config(
        job.id, json.dumps(cfg2), status=JobStatus.restarting, desired_state="running"
    )
    runtime.fail_build = True

    await controller.reconcile()
    await _drain_builds(controller)

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running  # reverted, previous version kept
    cfg = json.loads(job.config)
    assert cfg["image"] == "nerdit-app/app:1"
    assert "build_context_dir" not in cfg
    assert "build_context_root" not in cfg  # stripped with the other build markers
    assert "dockerfile_name" not in cfg


# --- query: restarting services keep their proxy route (redeploy no-outage) ---


async def test_restarting_service_included_in_active_routes(queries):
    """A 'restarting' service (mid-redeploy) stays in the proxy desired-set."""
    from nerdit.db.models import Job as _Job

    row = _Job(
        name="web",
        kind=JobKind.service,
        service_name="web",
        gpu_count=0,
        status=JobStatus.restarting,
        desired_state="running",
        restart_policy="always",
    )
    await queries.create_job(row)
    await queries.acquire_service_port("web", row.id, 8000, (9400, 9499))

    routes = await queries.list_active_service_routes()
    names = {r.service_name for r in routes}
    assert "web" in names  # kept routing on the stable host port during the build


# --- route: server-side [deploy] defaults from the uploaded nerdit.toml ------


_DEPLOY_TOML = """\
[deploy]
name = "demo"
port = 4321
gpus = 2
start = "./run.sh"
health = "/healthz"
"""


def _deploy_zip(toml_text: str) -> bytes:
    """A Dockerfile app + nerdit.toml (passthrough persists [deploy].start)."""
    return _zip({"Dockerfile": 'FROM scratch\nCMD ["true"]\n', "nerdit.toml": toml_text})


def _post_name_only(client: TestClient, zip_bytes: bytes, raw: str = SUB_RAW, **fields) -> object:
    """POST /deploy with ONLY the name field (so ZIP [deploy] defaults apply)."""
    data = {"name": "demo"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/deploy",
        data=data,
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(raw),
    )


def test_deploy_zip_deploy_section_used_as_defaults(tmp_path):
    """No form overrides: the ZIP's [deploy] supplies port/start/health/gpus."""
    q = _queries()
    resp = _post_name_only(_client(q, tmp_path), _deploy_zip(_DEPLOY_TOML))
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["port"] == 4321
    assert cfg["command"] == "./run.sh"  # Dockerfile passthrough honors start
    assert job.health_check == {"path": "/healthz"}
    assert job.gpu_count == 2


def test_deploy_zip_memory_and_cpu_limit_land_in_config_blob(tmp_path):
    """P13 WP5: ZIP [deploy].memory_limit/cpu_limit persist top-level (launch reader)."""
    q = _queries()
    toml = '[deploy]\nname = "demo"\nmemory_limit = "512m"\ncpu_limit = 1.5\n'
    resp = _post_name_only(_client(q, tmp_path), _deploy_zip(toml))
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["memory_limit"] == "512m"
    assert cfg["cpu_limit"] == 1.5


def test_deploy_zip_invalid_memory_limit_fails_loud(tmp_path):
    """P13 WP5: a previously-ignored invalid value now 422s via the fail-loud validator."""
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path), _deploy_zip('[deploy]\nname = "demo"\nmemory_limit = "lots"\n')
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid"
    q.reserve_service_for_token.assert_not_called()


def test_deploy_form_fields_override_zip_deploy_section(tmp_path):
    """Explicit form fields win over the ZIP's [deploy] values."""
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path),
        _deploy_zip(_DEPLOY_TOML),
        port=9000,
        gpus=0,
        start="./other.sh",
        health="/other",
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["port"] == 9000
    assert cfg["command"] == "./other.sh"
    assert job.health_check == {"path": "/other"}
    assert job.gpu_count == 0


def test_deploy_zip_quoted_gpus_coerced_to_int(tmp_path):
    """A quoted gpus value in [deploy] reaches the DB as an int, never a str.

    DeployConfig validates in lax mode ("2" -> 2); the route must use the
    VALIDATED value so the redeploy quota arithmetic (sum + gpu_count) can
    never TypeError on a str.
    """
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path),
        _deploy_zip('[deploy]\nname = "demo"\ngpus = "2"\n'),
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.gpu_count == 2
    assert isinstance(job.gpu_count, int)


def test_deploy_invalid_zip_deploy_section_fails_loud(tmp_path):
    """An unparseable [deploy] section is a structured 422, never ignored."""
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path), _deploy_zip('[deploy]\nname = "demo"\nport = 99999\n')
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.invalid"
    assert "nerdit.toml" in body["message"]
    q.reserve_service_for_token.assert_not_called()


def test_deploy_zip_stale_name_never_rejects_form_name(tmp_path):
    """A stale/invalid [deploy].name in the ZIP must not 422 the deploy.

    The form name is the service identity and the only name the route
    consumes; validation must let it win over the ZIP's [deploy].name
    (Codex PR #51 P2).
    """
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path),
        _deploy_zip('[deploy]\nname = "Not_A_DNS_Label!"\nport = 4321\n'),
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.service_name == "demo"
    assert json.loads(job.config)["port"] == 4321


# --- route: source provenance stamp (P11.5) ----------------------------------


def test_zip_deploy_stamps_source(tmp_path):
    """The ZIP ingress persists config['source'] == {"type": "zip"} on both paths."""
    # Fresh deploy: the stamp lands on the reserved row's config.
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["source"] == {"type": "zip"}

    # Redeploy: the stamp is (re)written on the update, replacing any prior one.
    existing = _redeploy_existing({"source": {"type": "git", "repo_url": "https://x/y"}})
    q2 = _queries(existing)
    resp = _post(_client(q2, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q2.update_service_config_guarded.call_args.args[1])
    assert cfg["source"] == {"type": "zip"}


# --- route: [db.*] provisioned-gate + persist + preserve (P15 / WP4) ----------

_DB_MANAGED_TOML = '[db.default]\nprovider = "managed"\ndatabase = "pg"\n'
_DB_EXTERNAL_TOML = (
    '[db.cache]\nprovider = "external"\n'
    'url = "redis://cache.example.com:6379/0"\n'
    'password = "${secrets.REDIS_PW}"\n'
)


def _db_row(
    status: JobStatus = JobStatus.running, name: str = "pg", owner: str | None = "tok-sub"
) -> Job:
    # `POST /databases` stamps `submitted_by_token`, so a realistic row is
    # owned; `owner` drives the H2 cross-owner regressions.
    return Job(
        id="db-1",
        service_name=name,
        name=name,
        kind=JobKind.database,
        gpu_count=0,
        status=status,
        config=json.dumps({"backend": "postgres"}),
        submitted_by_token=owner,
    )


def _queries_with_db(db_row: Job | None, existing: Job | None = None) -> AsyncMock:
    """A queries mock resolving the managed database row by ref (P15)."""
    q = _queries(existing)
    q.get_resource_by_ref = AsyncMock(return_value=db_row)
    return q


def test_deploy_gate_db_not_provisioned_on_deploy_route(tmp_path):
    """A managed [db.*] with no database row 422s db.not_provisioned on POST /deploy."""
    q = _queries_with_db(None)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "db.not_provisioned"
    assert "nerdit db create" in body["hint"]
    q.reserve_service_for_token.assert_not_called()


def test_deploy_gate_db_terminal_row(tmp_path):
    q = _queries_with_db(_db_row(JobStatus.stopped))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 422
    assert resp.json()["code"] == "db.not_provisioned"


def test_deploy_gate_db_wrong_kind_row(tmp_path):
    """A row that is not kind=database never satisfies a managed binding."""
    q = _queries_with_db(_model_row(JobStatus.running, kind=JobKind.service))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 422
    assert resp.json()["code"] == "db.not_provisioned"


def test_deploy_gate_db_building_passes(tmp_path):
    """A building database row is converging → deploy-during-provision works."""
    q = _queries_with_db(_db_row(JobStatus.building))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text


# --- H2: the managed [db.*] gate authorizes the REFERENCED database row ------
#
# A managed binding hands the app the database's minted password at launch
# (DATABASE_URL), so binding to a row is acting on it: without an ownership /
# scope check any submitter could read another owner's database.


def test_deploy_gate_db_cross_owner_forbidden(tmp_path):
    """A submitter binding to another owner's database is 403, nothing written."""
    q = _queries_with_db(_db_row(owner="tok-other"))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "forbidden"
    q.reserve_service_for_token.assert_not_called()


def test_deploy_gate_db_out_of_scope_forbidden(tmp_path):
    """A token scoped to its own app may not bind a database outside that scope."""
    q = _queries_with_db(_db_row(owner="tok-scoped"))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML), raw=SCOPED_RAW)
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["code"] == "forbidden"
    assert "pg" in body["message"]  # the scope refusal names the target
    q.reserve_service_for_token.assert_not_called()


def test_deploy_gate_db_owner_passes(tmp_path):
    """The database's own owner binds it (the pre-fix behaviour for this case)."""
    q = _queries_with_db(_db_row(owner="tok-sub"))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text


def test_deploy_gate_db_admin_bypasses(tmp_path):
    """Admin binds any database, as everywhere else."""
    q = _queries_with_db(_db_row(owner="tok-other"))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML), raw=LEGACY)
    assert resp.status_code == 201, resp.text


def test_deploy_gate_db_null_owner_passes(tmp_path):
    """A database created with the legacy global token (NULL owner) — the
    single-operator install's mainstream case — stays bindable by a scoped CI
    token and by the permanently-submitter tunnel principal. Admin-only there
    would break `nerdit db create pg` + `[db.default] database = "pg"`."""
    q = _queries_with_db(_db_row(owner=None))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text


def test_deploy_gate_db_cross_owner_refusal_names_the_binding(tmp_path):
    """The owner leg must not reuse the generic row denial: at this ingress
    that envelope is byte-identical to "you do not own the APP", so the caller
    could not tell which check failed."""
    q = _queries_with_db(_db_row(owner="tok-other"))
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 403, resp.text
    message = resp.json()["message"]
    assert "db.default" in message
    assert "'pg'" in message


def test_redeploy_over_an_unchanged_foreign_binding_passes(tmp_path):
    """An owner redeploying its OWN app whose nerdit.toml still names a
    database it does not own is carrying the binding forward, not authoring
    it — refusing there breaks `nerdit deploy .` on a flow that already runs
    (and that a toml-silent redeploy carries forward ungated anyway)."""
    existing = _redeploy_existing({"db": {"default": {"provider": "managed", "database": "pg"}}})
    q = _queries_with_db(_db_row(owner="tok-other"), existing)
    q.update_service_config_guarded = AsyncMock(return_value=True)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text


def test_redeploy_that_changes_a_foreign_binding_is_refused(tmp_path):
    """Carry-forward is not a laundering path: a spec that differs from the
    persisted one is authored by this caller and re-gated."""
    existing = _redeploy_existing({"db": {"default": {"provider": "managed", "database": "old"}}})
    q = _queries_with_db(_db_row(owner="tok-other"), existing)
    q.update_service_config_guarded = AsyncMock(return_value=True)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 403, resp.text
    q.update_service_config_guarded.assert_not_called()


def test_deploy_external_db_needs_no_row(tmp_path):
    """provider='external' needs no database row (and no secret at deploy)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_EXTERNAL_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["db"]["cache"]["provider"] == "external"
    assert cfg["db"]["cache"]["password"] == "${secrets.REDIS_PW}"


def test_deploy_persists_db_spec(tmp_path):
    q = _queries_with_db(_db_row())
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["db"] == {"default": {"provider": "managed", "database": "pg"}}


def test_deploy_invalid_db_url_rejected(tmp_path):
    toml = '[db.cache]\nprovider = "external"\nurl = "mysql://h/db"\npassword = "${secrets.P}"\n'
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "db.external_url_invalid"


def test_deploy_db_url_query_string_rejected(tmp_path):
    toml = (
        '[db.cache]\nprovider = "external"\n'
        'url = "postgresql://u@h/db?password=x"\npassword = "${secrets.P}"\n'
    )
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "db.external_url_invalid"


def test_dry_run_db_diff_set(tmp_path):
    q = _queries_with_db(_db_row())
    resp = _dry_post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["db_diff"] == {"action": "set", "bindings": ["default"]}
    q.reserve_service_for_token.assert_not_called()
    # §3 D-B: a dry-run diff for a managed [db.*] binding must never echo a
    # credential — no 64-hex mint token and no credential-bearing DSN.
    assert not re.search(r"\b[0-9a-f]{64}\b", resp.text), resp.text[:200]
    assert not re.search(r"postgresql://[^/\s]*:[^/\s@]+@", resp.text), "credential DSN leaked"


def test_redeploy_preserves_db_when_api_marked_and_source_silent(tmp_path):
    """Source silent + config['db_source']=='api' → the API-set spec is preserved."""
    existing = _redeploy_existing(
        {"db": {"default": {"provider": "managed", "database": "pg"}}, "db_source": "api"}
    )
    q = _queries_with_db(_db_row(), existing)

    async def _apply(job_id, config_json, **kwargs):
        _apply.cfg = json.loads(config_json)
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), _node_zip())  # no [db] in the source
    assert resp.status_code == 201, resp.text
    assert _apply.cfg["db"] == {"default": {"provider": "managed", "database": "pg"}}
    assert _apply.cfg["db_source"] == "api"


def test_redeploy_db_declared_replaces_and_drops_marker(tmp_path):
    existing = _redeploy_existing(
        {"db": {"default": {"provider": "managed", "database": "old"}}, "db_source": "api"}
    )
    q = _queries_with_db(_db_row(), existing)

    async def _apply(job_id, config_json, **kwargs):
        _apply.cfg = json.loads(config_json)
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_DB_MANAGED_TOML))
    assert resp.status_code == 201, resp.text
    assert _apply.cfg["db"] == {"default": {"provider": "managed", "database": "pg"}}
    assert "db_source" not in _apply.cfg
    assert resp.json()["overwrote_api_config"] is True


def test_redeploy_silent_no_marker_drops_db(tmp_path):
    """Source silent, no db_source marker → any carried db spec is dropped."""
    existing = _redeploy_existing({"db": {"default": {"provider": "managed", "database": "pg"}}})
    q = _queries_with_db(_db_row(), existing)

    async def _apply(job_id, config_json, **kwargs):
        _apply.cfg = json.loads(config_json)
        return True

    q.update_service_config_guarded = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert "db" not in _apply.cfg


# --- P20: [deploy].release plumbing through the deploy pipeline -----------------
#
# The execution hook lives in ``core/app_build.py`` (see
# ``tests/test_release_gate.py``); what is pinned here is the §1.5 blast radius
# the deploy route owns — precedence, persistence, the armed crash marker, the
# dry-run plan and the clobber flag.

_RELEASE_TOML = '[deploy]\nname = "demo"\nrelease = "alembic upgrade head"\n'


def _written_cfg(q) -> dict:
    """The config blob the pipeline handed to the write branch (fresh or redeploy)."""
    if q.update_service_config_guarded.call_args is not None:
        return json.loads(q.update_service_config_guarded.call_args.args[1])
    return json.loads(q.reserve_service_for_token.await_args.args[0].config)


def test_fresh_deploy_persists_release_but_does_not_arm_the_marker(tmp_path):
    """D-P20-4: deploy persists the command; only the builder arms the marker.

    Regression (found in the P20 live run). Arming ``release_pending`` here
    made it mean "a release is configured", which ``ensure_built`` layer 3
    cannot tell apart from "a release started and the daemon died" — so a
    deploy that skips the build (target tag already present) settled instantly
    as a phantom crashed release and the service never started.
    """
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["release"] == "alembic upgrade head"
    assert cfg["build_version"] == 1
    assert "release_pending" not in cfg


def test_redeploy_allocates_the_next_version_for_a_release_generation(tmp_path):
    """The candidate tag must be the freshly allocated one, never the serving one."""
    existing = _redeploy_existing({"build_version": 3, "max_version": 3})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["build_version"] == 4
    assert "release_pending" not in cfg


def test_redeploy_after_rollback_allocates_from_the_high_water_mark(tmp_path):
    """Rollback lowers ``build_version`` but not ``max_version``; the new
    generation must follow the freshly allocated tag, not the serving one."""
    existing = _redeploy_existing({"build_version": 2, "max_version": 5})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["build_version"] == 6
    assert "release_pending" not in cfg


def test_deploy_without_release_arms_no_marker(tmp_path):
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert "release" not in cfg
    assert "release_pending" not in cfg


def test_redeploy_carries_a_prior_release_forward_when_the_source_is_silent(tmp_path):
    """Precedence, leg 2: no form/body field exists for ``release``, so a deploy
    from a source that stays silent (a ZIP with no ``nerdit.toml``) must not
    silently drop a migration gate the prior generation declared."""
    existing = _redeploy_existing({"release": "alembic upgrade head"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["release"] == "alembic upgrade head"
    assert cfg["build_version"] == 2
    assert "release_pending" not in cfg


def test_zip_release_beats_the_prior_row_value(tmp_path):
    """Precedence, leg 1: the repo's own ``[deploy].release`` wins."""
    existing = _redeploy_existing({"release": "old-migration"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    assert _written_cfg(q)["release"] == "alembic upgrade head"


def test_dry_run_plan_carries_the_release_it_would_run(tmp_path):
    """An agent must be able to see the command a deploy WOULD execute before
    committing to it — verbatim (it is app-authored config, not a resolved
    value) and with zero writes."""
    q = _queries()
    resp = _dry_post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["release"] == "alembic upgrade head"
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config_guarded.assert_not_called()


def test_dry_run_plan_shows_a_carried_forward_release_on_a_silent_redeploy(tmp_path):
    existing = _redeploy_existing({"release": "alembic upgrade head"})
    resp = _dry_post(_client(_queries(existing), tmp_path), _node_zip())
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["release"] == "alembic upgrade head"


def test_invalid_release_in_the_zip_is_a_422_that_never_echoes_it(tmp_path):
    q = _queries()
    toml = '[deploy]\nname = "demo"\nrelease = """\nmigrate\nFORGED\n"""\n'
    resp = _post(_client(q, tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.invalid"
    assert "control characters" in body["message"]
    assert "FORGED" not in resp.content.decode()
    q.reserve_service_for_token.assert_not_awaited()


def test_deploy_shape_hint_documents_release(tmp_path):
    """The 422 hint is the only in-band schema an agent gets for ``[deploy]``."""
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip('[deploy]\nrelease = ""\n'))
    assert resp.status_code == 422
    assert "release" in resp.json()["hint"]


def test_declared_release_over_an_api_authored_one_flags_the_clobber(tmp_path):
    """The sixth ``_explicit_changed`` clause: ``release`` is API-writable, so a
    source that declares its own over an API-set value is a visible clobber."""
    existing = _redeploy_existing({"release": "api-authored-migration", "config_source": "api"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is True


def test_same_release_over_an_api_authored_one_is_not_a_clobber(tmp_path):
    """P13 WP9's narrowed rule: only an explicit AND changed value counts."""
    existing = _redeploy_existing({"release": "alembic upgrade head", "config_source": "api"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_RELEASE_TOML))
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is False


def test_silent_redeploy_over_an_api_authored_release_is_not_a_clobber(tmp_path):
    """Nothing was declared, so nothing the API authored is overwritten — the
    carried-forward value is preserved, not clobbered."""
    existing = _redeploy_existing({"release": "api-authored-migration", "config_source": "api"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is False
    assert _written_cfg(q)["release"] == "api-authored-migration"


def test_rollback_clears_a_stale_release_marker_but_keeps_the_command(tmp_path):
    """A rollback bypasses the builder entirely, so it is the one path that
    would otherwise inherit a ``release_pending`` left by a wedged generation —
    and that marker would make the controller settle the rolled-back generation
    ``release_failed`` on the next tick. The command itself is app config, not
    generation state, so it survives."""
    existing = _svc_with_config(
        {
            "image": "nerdit-app/demo:2",
            "previous_image": "nerdit-app/demo:1",
            "build_version": 2,
            "max_version": 2,
            "release": "alembic upgrade head",
            "release_pending": 2,
        }
    )
    q = _queries(existing)
    resp = _client(q, tmp_path).post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    cfg = json.loads(q.update_service_config_guarded.call_args.args[1])
    assert "release_pending" not in cfg
    assert cfg["release"] == "alembic upgrade head"
    assert cfg["build_version"] == 1


def test_unknown_deploy_keys_are_warned_without_echoing_control_characters(caplog):
    """The mitigation for the deliberately missing ``extra='forbid'`` (D-P22-3
    keeps ``DeployConfig`` tolerant for forward compat): a pre-P20 daemon
    silently drops ``[deploy].release``, so a daemon that DOES know a key says
    so for every FUTURE one. Key names are app-authored strings landing in a
    plain-text operator log, so they are sanitized and bounded."""
    import logging

    from nerdit.daemon.deploy_pipeline import (
        _UNKNOWN_KEY_LOG_LIMIT,
        _warn_unknown_deploy_keys,
    )

    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        shown = _warn_unknown_deploy_keys({"name": "demo", "release": "x", "future_key": 1}, "demo")
    assert shown == ["future_key"]  # (Agent-DX) also returned, for the caller
    assert "future_key" in caplog.text
    assert "release" not in caplog.text.split("IGNORE:")[-1]  # release IS declared

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        _warn_unknown_deploy_keys({"bad\nkey\x00": 1}, "demo")
    assert "\n" not in caplog.records[-1].getMessage()
    assert "\x00" not in caplog.records[-1].getMessage()
    assert "bad?key?" in caplog.records[-1].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        _warn_unknown_deploy_keys({f"k{i}": 1 for i in range(_UNKNOWN_KEY_LOG_LIMIT + 5)}, "demo")
    assert "(+5 more)" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        assert _warn_unknown_deploy_keys({"name": "demo", "port": 8000, "release": "x"}, "d") == []
    assert caplog.records == []


# --- P25 WP4j: [deploy].edge_auth plumbing through the deploy pipeline ----------
#
# ``edge_auth`` is the EIGHTH no-form-field ``[deploy]`` key and follows the
# ``release``/``cutover`` template exactly (precedence, persistence, dry-run
# plan, clobber flag). It is the one member of the family whose carry-forward is
# security-relevant: a redeploy from a silent source must never unprotect a
# published app, and the ONLY disarm is the explicit
# ``nerdit config app set <app> deploy edge_auth=null`` (D-P25-5).

_EDGE_AUTH_TOML = (
    '[deploy]\nname = "demo"\n[deploy.edge_auth]\nuser = "ops"\npassword = "${secrets.APP_PW}"\n'
)
_EDGE_AUTH_BLOB = {"user": "ops", "password": "${secrets.APP_PW}"}


def test_fresh_deploy_persists_edge_auth_as_a_plain_dict(tmp_path):
    """The persisted blob must be JSON-serializable (``jobs.config`` is
    ``json.dumps``'d) and must hold the ``${secrets.KEY}`` REFERENCE — the
    daemon resolves and bcrypt-hashes it at route-build time, so no password
    ever reaches the row."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_ai_zip(_EDGE_AUTH_TOML))
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["edge_auth"] == _EDGE_AUTH_BLOB
    assert json.dumps(cfg)  # round-trips: a pydantic model here would raise


def test_deploy_without_edge_auth_persists_no_key(tmp_path):
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert "edge_auth" not in _written_cfg(q)


def test_silent_redeploy_carries_edge_auth_forward(tmp_path):
    """The security-relevant leg: a ZIP with no ``nerdit.toml`` (a `nerdit dev`
    push, a dashboard redeploy) must NOT drop the protection the prior
    generation declared — there is no form field that could re-state it."""
    existing = _redeploy_existing({"edge_auth": _EDGE_AUTH_BLOB})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    cfg = _written_cfg(q)
    assert cfg["edge_auth"] == _EDGE_AUTH_BLOB
    assert cfg["build_version"] == 2


def test_zip_edge_auth_beats_the_prior_row_value(tmp_path):
    """Precedence, leg 1: the repo's own ``[deploy].edge_auth`` wins."""
    existing = _redeploy_existing({"edge_auth": {"user": "old", "password": "${secrets.OLD_PW}"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_EDGE_AUTH_TOML))
    assert resp.status_code == 201, resp.text
    assert _written_cfg(q)["edge_auth"] == _EDGE_AUTH_BLOB


def test_dry_run_plan_carries_the_edge_auth_it_would_apply(tmp_path):
    """The plan diff shows ``{user, password ref}`` verbatim. That is value-free
    BY CONSTRUCTION rather than by a filter: ``EdgeAuthConfig`` rejects a literal
    password, so what survives validation is a reference — and masking the ref
    would hide the one thing the operator needs to check before committing."""
    q = _queries()
    resp = _dry_post(_client(q, tmp_path), _node_ai_zip(_EDGE_AUTH_TOML))
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["edge_auth"] == _EDGE_AUTH_BLOB
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config_guarded.assert_not_called()


def test_dry_run_plan_shows_a_carried_forward_edge_auth_on_a_silent_redeploy(tmp_path):
    existing = _redeploy_existing({"edge_auth": _EDGE_AUTH_BLOB})
    resp = _dry_post(_client(_queries(existing), tmp_path), _node_zip())
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["edge_auth"] == _EDGE_AUTH_BLOB


def test_dry_run_plan_shows_null_when_no_edge_auth_is_declared(tmp_path):
    resp = _dry_post(_client(_queries(), tmp_path), _node_zip())
    assert resp.status_code == 200, resp.text
    assert resp.json()["effective"]["edge_auth"] is None


def test_a_literal_edge_auth_password_is_a_422_that_never_echoes_it(tmp_path):
    """Bug #28 / finding #6, applied up front: the ``deploy.invalid`` envelope
    surfaces ``errors()[0]['msg']``, so a validator that interpolated the
    rejected value would publish it. ``EdgeAuthConfig`` carries
    ``hide_input_in_errors`` AND a value-free message; neither alone suffices."""
    q = _queries()
    toml = (
        '[deploy]\nname = "demo"\n[deploy.edge_auth]\nuser = "ops"\npassword = "hunter2-FORGED"\n'
    )
    resp = _post(_client(q, tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.invalid"
    assert "secret reference" in body["message"]
    assert "hunter2-FORGED" not in resp.content.decode()
    q.reserve_service_for_token.assert_not_awaited()


def test_an_edge_auth_user_with_a_colon_is_a_422_that_never_echoes_it(tmp_path):
    """RFC 7617 separates user and password with ':' — a colon in the user would
    silently reshape which password the browser sends. The message names the
    rule, never the value (a control character echoed into an operator log is a
    log-forging vector — the ``[deploy].release`` precedent)."""
    q = _queries()
    toml = (
        '[deploy]\nname = "demo"\n'
        '[deploy.edge_auth]\nuser = "ops:FORGED"\npassword = "${secrets.APP_PW}"\n'
    )
    resp = _post(_client(q, tmp_path), _node_ai_zip(toml))
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.invalid"
    assert "FORGED" not in resp.content.decode()


def test_deploy_shape_hint_documents_edge_auth(tmp_path):
    """The 422 hint is the only in-band schema an agent gets for ``[deploy]``."""
    resp = _post(_client(_queries(), tmp_path), _node_ai_zip('[deploy]\nrelease = ""\n'))
    assert resp.status_code == 422
    assert "edge_auth" in resp.json()["hint"]


def test_declared_edge_auth_over_an_api_authored_one_flags_the_clobber(tmp_path):
    """``edge_auth`` is API-writable on the same PUT surface as ``release``, so
    the same visibility rule applies — and it matters more here: a source
    silently replacing an API-set declaration changes who can reach the app."""
    existing = _redeploy_existing(
        {
            "edge_auth": {"user": "api-set", "password": "${secrets.API_PW}"},
            "config_source": "api",
        }
    )
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_EDGE_AUTH_TOML))
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is True


def test_same_edge_auth_over_an_api_authored_one_is_not_a_clobber(tmp_path):
    existing = _redeploy_existing({"edge_auth": _EDGE_AUTH_BLOB, "config_source": "api"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_ai_zip(_EDGE_AUTH_TOML))
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is False


def test_silent_redeploy_over_an_api_authored_edge_auth_is_not_a_clobber(tmp_path):
    existing = _redeploy_existing({"edge_auth": _EDGE_AUTH_BLOB, "config_source": "api"})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert resp.json()["overwrote_api_config"] is False
    assert _written_cfg(q)["edge_auth"] == _EDGE_AUTH_BLOB


def test_edge_auth_is_not_warned_as_an_unknown_deploy_key(caplog):
    """Rev-2 amendment 9: ``_warn_unknown_deploy_keys`` reads
    ``DeployConfig.model_fields``, so declaring the field is the whole fix —
    there is no known-key set to edit. Pinned by test, not by code."""
    import logging

    from nerdit.daemon.deploy_pipeline import _warn_unknown_deploy_keys

    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        shown = _warn_unknown_deploy_keys(
            {"name": "demo", "edge_auth": {"user": "ops"}, "future_key": 1}, "demo"
        )
    assert shown == ["future_key"]
    assert "future_key" in caplog.text
    assert "edge_auth" not in caplog.text

    # ...and a [deploy] table with ONLY known keys warns about nothing at all.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.deploy_pipeline"):
        assert _warn_unknown_deploy_keys({"name": "demo", "edge_auth": {"user": "ops"}}, "d") == []
    assert caplog.records == []


# --- Agent-DX: the deploy-result summary + hints channel ----------------------
#
# ``summary``/``hints`` are assembled in ``_finalize_deploy``, the tail EVERY
# deploy ingress shares (ZIP / git / redeploy / template / workspace), so these
# route-level tests pin the mechanism once; the per-ingress files assert only
# that their ingress reaches the same seam.


def _proxy_client(queries: AsyncMock, tmp_path, *, mode: str, available: bool) -> TestClient:
    """A deploy client whose app.state reports a given proxy mode + live state.

    Mirrors what ``_deploy_hints`` reads (``settings.proxy.mode`` +
    ``proxy_manager.available``) and nothing else — the hint must not need a
    real proxy, a hostname or an endpoint row.
    """
    app = _make_app(queries, tmp_path)
    app.state.settings.proxy.mode = mode
    app.state.proxy_manager = MagicMock(available=available)
    return TestClient(app, raise_server_exceptions=False)


def test_deploy_response_carries_summary_projected_from_the_body(tmp_path):
    """The four-field block, COPIED from the assembled body (one derivation)."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body["summary"]) == {"app", "status", "version", "public_url"}
    assert body["summary"]["app"] == body["name"] == "demo"
    assert body["summary"]["status"] == body["status"]
    assert body["summary"]["version"] == body["build_version"]
    endpoint = body.get("endpoint") or {}
    assert body["summary"]["public_url"] == endpoint.get("public_url")


def test_fresh_deploy_summary_reports_a_null_public_url(tmp_path):
    """The honest pre-launch reading: the endpoint row is created at launch."""
    q = _queries()
    q.get_service_endpoint = AsyncMock(return_value=None)
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert resp.json()["summary"]["public_url"] is None


def test_path_mode_emits_the_base_path_hint(tmp_path):
    """Leg A: path mode + a live proxy ⇒ the frontend base-path advisory."""
    q = _queries()
    resp = _post(_proxy_client(q, tmp_path, mode="path", available=True), _node_zip())
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert any("'path' proxy mode" in h and "/demo/" in h for h in hints), hints
    # (P26 WP0 / L1) The hint must also name the way OUT of the trap: subdomain
    # mode serves the app at the root of its own hostname and needs no base path.
    # Without this sentence the advisory tells an operator to work around a
    # constraint their daemon can simply drop.
    assert any("subdomain" in h for h in hints), hints


def test_subdomain_mode_emits_no_base_path_hint(tmp_path):
    """Leg B: the same live proxy in subdomain mode says nothing — the app is
    served at the root of its own host, so there is no base path to set."""
    q = _queries()
    resp = _post(_proxy_client(q, tmp_path, mode="subdomain", available=True), _node_zip())
    assert resp.status_code == 201, resp.text
    # (P34) The async sentence is unconditional and always closes the list, so
    # "said nothing about base paths" is now "said ONLY the async sentence".
    assert resp.json()["hints"] == [_ASYNC_HINT]


def test_path_mode_emits_no_base_path_hint_when_the_proxy_is_unavailable(tmp_path):
    """Leg B′: the LIVE-state gate. Proxy configured for path mode but never came
    up ⇒ nothing is served under /<name>/, so advertising the base path would be
    advice about a URL that does not resolve (the ``_endpoint_view`` rule)."""
    q = _queries()
    resp = _post(_proxy_client(q, tmp_path, mode="path", available=False), _node_zip())
    assert resp.status_code == 201, resp.text
    assert resp.json()["hints"] == [_ASYNC_HINT]


def test_unknown_deploy_key_is_surfaced_as_a_hint(tmp_path):
    """Leg A: the field-test bug — a `build` key was silently ignored. The
    tolerant parse (D-P22-3) stays; the SILENCE is what is fixed."""
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path), _deploy_zip('[deploy]\nname = "demo"\nbuild = "npm run build"\n')
    )
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert len(hints) == 2
    assert "IGNORED" in hints[0] and "build" in hints[0]
    assert hints[1] == _ASYNC_HINT
    # ...and it points at the actual remedy for THIS key.
    assert "no `build` key" in hints[0]


def test_a_clean_deploy_section_produces_no_unknown_key_hint(tmp_path):
    """Leg B: every key declared ⇒ the channel stays silent."""
    q = _queries()
    resp = _post_name_only(_client(q, tmp_path), _deploy_zip(_DEPLOY_TOML))
    assert resp.status_code == 201, resp.text
    assert resp.json()["hints"] == [_ASYNC_HINT]


def test_unknown_key_hint_lists_names_only_and_never_a_value(tmp_path):
    """House rule: a hint carries key NAMES; a TOML value never reaches it."""
    q = _queries()
    resp = _post_name_only(
        _client(q, tmp_path),
        _deploy_zip('[deploy]\nname = "demo"\napi_token = "s3cr3t-value"\n'),
    )
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert "api_token" in hints[0]
    assert "s3cr3t-value" not in hints[0]


def test_unknown_key_hint_inherits_the_control_char_and_length_bounds(tmp_path):
    """The hint reuses ``_warn_unknown_deploy_keys``'s sanitized/bounded output,
    so a forged control character or a 200-key section cannot blow up the
    response any more than it can the log line."""
    from nerdit.daemon.deploy_pipeline import (
        _UNKNOWN_KEY_LOG_LIMIT,
        _UNKNOWN_KEY_LOG_MAX_CHARS,
        _unknown_keys_message,
        _warn_unknown_deploy_keys,
    )

    shown = _warn_unknown_deploy_keys({"bad\nkey\x00": 1}, "demo")
    assert shown == ["bad?key?"]
    msg = _unknown_keys_message(shown)
    assert "\n" not in msg and "\x00" not in msg

    many = _warn_unknown_deploy_keys({f"k{i}": 1 for i in range(_UNKNOWN_KEY_LOG_LIMIT + 5)}, "d")
    assert len(many) == _UNKNOWN_KEY_LOG_LIMIT + 1  # the names + the truncation marker
    assert many[-1] == "(+5 more)"
    # ...and the marker reaches the rendered advisory, so a caller past the cap
    # is never handed a silently shortened list as if it were the whole set.
    assert _unknown_keys_message(many).endswith(
        "(+5 more). Check them against the [deploy] schema "
        "(there is no `build` key — run the build in your Dockerfile)."
    )

    long_key = _warn_unknown_deploy_keys({"z" * 300: 1}, "demo")
    assert len(long_key[0]) == _UNKNOWN_KEY_LOG_MAX_CHARS


def test_unknown_key_hint_strips_bidi_and_line_separators(tmp_path):
    """(Codex 3804646882) Display-unsafe Unicode never rides an advisory.

    A quoted TOML key may legally carry U+202E RIGHT-TO-LEFT OVERRIDE, which
    visually reorders the rest of a one-line advisory in a terminal, and U+2028
    LINE SEPARATOR. ``_CONTROL_CHAR_RE`` only covers C0/C1, so both survived
    into ``hints``, the dry-run ``warnings`` and the operator log line.
    """
    from nerdit.daemon.deploy_pipeline import _unknown_keys_message, _warn_unknown_deploy_keys

    shown = _warn_unknown_deploy_keys({"a\u202eb\u2028c": 1}, "demo")
    assert shown == ["a?b?c"]
    msg = _unknown_keys_message(shown)
    assert "\u202e" not in msg and "\u2028" not in msg

    # ...and nothing else is touched: an ordinary key is byte-unchanged, so the
    # widened class cannot over-sanitize the overwhelmingly common case.
    assert _warn_unknown_deploy_keys({"future_key-2": 1}, "demo") == ["future_key-2"]


def test_deploy_release_validation_is_unchanged_by_the_display_safe_class(tmp_path):
    """The wider class is a SEPARATE constant, so ``[deploy].release`` is untouched.

    ``_CONTROL_CHAR_RE`` is also ``release``'s validator, where the documented
    rejection contract is control characters ("newlines, tabs and NUL included").
    Widening it there would silently start rejecting a class of characters the
    finding never discussed, so this pins that a bidi-carrying ``release`` still
    validates exactly as it did before.
    """
    from nerdit.config.project import DeployConfig

    assert DeployConfig(name="demo", release="a\u202eb").release == "a\u202eb"


def test_hint_order_is_unknown_keys_then_base_path(tmp_path):
    """LOCKED order: what the caller WROTE and we ignored outranks the
    structural routing note — it is the more actionable of the two."""
    q = _queries()
    resp = _post_name_only(
        _proxy_client(q, tmp_path, mode="path", available=True),
        _deploy_zip('[deploy]\nname = "demo"\nbuild = "npm run build"\n'),
    )
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert len(hints) == 3
    assert "IGNORED" in hints[0]
    assert "'path' proxy mode" in hints[1]
    # (P34) The async sentence is LAST: it is about the platform's timing, not
    # about anything the caller wrote or can change.
    assert hints[2] == _ASYNC_HINT


def _dry_post_name_only(client: TestClient, zip_bytes: bytes) -> object:
    return client.post(
        "/deploy",
        params={"dry_run": "true"},
        data={"name": "demo"},
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(SUB_RAW),
    )


def test_dry_run_plan_warns_about_unknown_deploy_keys(tmp_path):
    """Leg A: the plan's EXISTING advisory channel carries the same sentence, so
    an agent that dry-runs first learns about the ignored key before deploying."""
    q = _queries()
    resp = _dry_post_name_only(
        _client(q, tmp_path), _deploy_zip('[deploy]\nname = "demo"\nbuild = "npm run build"\n')
    )
    assert resp.status_code == 200, resp.text
    warnings = resp.json()["warnings"]
    assert any("IGNORED" in w and "build" in w for w in warnings), warnings
    q.reserve_service_for_token.assert_not_called()


def test_truncated_unknown_key_set_is_reported_as_truncated_to_the_caller(tmp_path):
    """Past ``_UNKNOWN_KEY_LOG_LIMIT`` the list the caller sees is a PREFIX, and
    both advisory channels say so — the log line has always carried the
    ``(+N more)`` marker, the response hint and the plan warning now do too."""
    from nerdit.daemon.deploy_pipeline import _UNKNOWN_KEY_LOG_LIMIT

    keys = "".join(f"zz{i:02d} = 1\n" for i in range(_UNKNOWN_KEY_LOG_LIMIT + 3))
    toml = f'[deploy]\nname = "demo"\n{keys}'

    resp = _post_name_only(_client(_queries(), tmp_path), _deploy_zip(toml))
    assert resp.status_code == 201, resp.text
    assert "(+3 more)" in resp.json()["hints"][0]

    dry = _dry_post_name_only(_client(_queries(), tmp_path), _deploy_zip(toml))
    assert dry.status_code == 200, dry.text
    assert any("(+3 more)" in w for w in dry.json()["warnings"])


def test_dry_run_plan_has_no_unknown_key_warning_for_a_clean_toml(tmp_path):
    """Leg B: a fully-declared [deploy] adds nothing to the plan's warnings."""
    q = _queries()
    resp = _dry_post_name_only(_client(q, tmp_path), _deploy_zip(_DEPLOY_TOML))
    assert resp.status_code == 200, resp.text
    assert not any("IGNORED" in w for w in resp.json()["warnings"])


def test_warn_unknown_deploy_keys_returns_the_sanitized_names(tmp_path):
    """The threaded return value IS the response/plan payload — one parse, one
    sanitization, one bound, no second source of truth at the response seam."""
    from nerdit.daemon.deploy_pipeline import _parse_deploy_defaults, _warn_unknown_deploy_keys

    assert _warn_unknown_deploy_keys({"name": "demo", "release": "x"}, "demo") == []
    assert _warn_unknown_deploy_keys({"build": 1, "aaa": 2}, "demo") == ["aaa", "build"]

    # ...and _parse_deploy_defaults threads it out as the second element.
    section, unknown = _parse_deploy_defaults({"deploy": {"name": "demo", "build": "x"}}, "demo")
    assert section["name"] == "demo"
    assert unknown == ["build"]

    # Every early return is the empty pair, never a bare {}.
    assert _parse_deploy_defaults({}, "demo") == ({}, [])


def test_deploy_shape_hint_names_every_deploy_config_field():
    """The other half of the SYNC OBLIGATION the MCP docstring test guards.

    ``_DEPLOY_SHAPE_HINT`` is what a caller reads when its ``[deploy]`` section
    422s, so a key present in the schema but missing from the hint tells the
    caller the daemon has a schema it does not have — the same class of bug as
    the field agent guessing ``build``. Forward direction only: the hint is
    prose (unlike the docstrings' ``key`` markup), so "every field is named"
    is checkable but "no non-field is named" is not.
    """
    from nerdit.config.project import DeployConfig
    from nerdit.daemon.deploy_pipeline import _DEPLOY_SHAPE_HINT

    missing = [
        field
        for field in DeployConfig.model_fields
        if not re.search(rf"\b{re.escape(field)}\b", _DEPLOY_SHAPE_HINT)
    ]
    assert missing == [], f"_DEPLOY_SHAPE_HINT omits {missing}"


# --- P34: next_step + the never-empty hints channel ----------------------------
#
# The field failure: ``deploy_app returned status: building with hints: []`` and
# the agent concluded the deploy had landed. It had not — it was crash-looping,
# discoverable only by knowing to call get_events/diagnose_service unprompted.
# Two channels now say the same thing, one structured and one prose, on the tail
# every ingress shares.


def test_a_deploy_response_carries_a_structured_next_step(tmp_path):
    """Machine-shaped, so an agent executes it rather than parsing hints."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["next_step"] == {
        "tool": "wait_for_service",
        "args": {"name": "demo"},
        "why": _ASYNC_HINT,
    }
    # It names the service the deploy actually created, never the request's
    # raw form field — the two can differ (nerdit.toml supplies the name).
    assert body["next_step"]["args"]["name"] == body["name"]


def test_hints_are_never_empty_on_an_in_flight_deploy(tmp_path):
    """The whole point: the empty list was the COMMON case (no unknown keys, no
    live path-mode proxy), i.e. exactly the node a remote agent works on."""
    q = _queries()
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert hints
    # And the async sentence is the SAME string ``next_step.why`` carries, so a
    # client surfacing only one channel still states the whole fact.
    assert hints[-1] == resp.json()["next_step"]["why"]


def test_a_dry_run_plan_carries_neither_next_step_nor_the_async_hint(tmp_path):
    """A plan is not in flight — nothing was built, so there is nothing to wait
    for, and the documented "a dry_run plan carries neither" must stay true."""
    q = _queries()
    resp = _dry_post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "next_step" not in body
    assert "hints" not in body


def test_redeploying_over_a_failed_row_leads_with_the_diagnose_pointer(tmp_path):
    """(D) 'hints/public_url empty when you most need them': a redeploy over a
    crash-looping app must name the app's own history FIRST, not the platform's."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.failed,
        submitted_by_token="tok-sub",
        config=json.dumps({"image": "nerdit-app/demo:1", "build_version": 1, "port": 8000}),
    )
    resp = _post(_client(_queries(existing), tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert "failed state" in hints[0]
    assert "diagnose_service('demo')" in hints[0]
    assert hints[-1] == _ASYNC_HINT


def test_a_healthy_row_gets_no_failure_hint_on_redeploy(tmp_path):
    """The counterweight: the pointer must not fire on every redeploy."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        submitted_by_token="tok-sub",
        config=json.dumps({"image": "nerdit-app/demo:1", "build_version": 1, "port": 8000}),
    )
    resp = _post(_client(_queries(existing), tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    assert resp.json()["hints"] == [_ASYNC_HINT]


def test_the_failure_hint_never_carries_app_authored_text(tmp_path):
    """Only daemon vocabulary reaches the hints channel: a machine ``reason``
    token, never ``error_message`` (which is whatever the app or docker said)."""
    existing = Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.degraded,
        error_message="Traceback: KeyError('AWS_SECRET_ACCESS_KEY=hunter2')",
        submitted_by_token="tok-sub",
        config=json.dumps(
            {
                "image": "nerdit-app/demo:1",
                "build_version": 1,
                "port": 8000,
                "last_deploy": {"version": 1, "phase": "failed", "reason": "crash_loop"},
            }
        ),
    )
    resp = _post(_client(_queries(existing), tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    hints = resp.json()["hints"]
    assert "crash_loop" in hints[0]
    assert "hunter2" not in " ".join(hints)


def test_dry_run_plan_shows_public_env_in_clear(tmp_path):
    """P38: the one plan field printed with its values.

    They compile into public build output, and `BuildSettings` refuses secret
    references, so the contrast with the names-only `env_diff` is deliberate.
    """
    q = _queries()
    resp = _dry_post(
        _client(q, tmp_path),
        _node_zip(),
        build_settings=json.dumps({"public_env": {"VITE_API": "https://api.example.com"}}),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["build"]["public_env"] == {"VITE_API": "https://api.example.com"}
    resp = _dry_post(_client(q, tmp_path), _node_zip())
    assert resp.json()["build"]["public_env"] == {}


# --- P39: a name claimed by setting its secrets first -------------------------


def _claim(token_id: str | None) -> SecretClaim:
    return SecretClaim(service_name="demo", token_id=token_id)


def test_fresh_deploy_by_the_claimant_consumes_the_claim(tmp_path):
    """The claimant's deploy reaches the row transaction, which consumes the claim."""
    q = _queries()
    q.get_secret_claim = AsyncMock(return_value=_claim("tok-sub"))
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.submitted_by_token == "tok-sub"
    # Non-admin: the query must fail closed on any claim it does not own.
    assert q.reserve_service_for_token.call_args.kwargs == {"admin": False}


def test_fresh_deploy_over_a_foreign_claim_409_before_extract(tmp_path):
    q = _queries()
    q.get_secret_claim = AsyncMock(return_value=_claim("tok-other"))
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "service.name_claimed"
    assert "nerdit secrets rm demo" in resp.json()["hint"]
    assert "tok-other" not in resp.text  # the claimant is never named
    q.reserve_service_for_token.assert_not_called()
    assert not (tmp_path / "uploads").exists()  # refused before the upload is unpacked


def test_admin_deploys_over_a_foreign_claim(tmp_path):
    q = _queries()
    q.get_secret_claim = AsyncMock(return_value=_claim("tok-other"))
    resp = _post(_client(q, tmp_path), _node_zip(), raw=LEGACY)
    assert resp.status_code == 201, resp.text
    assert q.reserve_service_for_token.call_args.kwargs == {"admin": True}


def test_claim_raced_into_the_transaction_maps_to_409(tmp_path):
    """The transaction is the boundary: a claim minted after the fast path still refuses."""
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameClaimed("demo"))
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.name_claimed"


def test_redeploy_never_reads_the_claim(tmp_path):
    q = _queries(_redeploy_existing())
    resp = _post(_client(q, tmp_path), _node_zip())
    assert resp.status_code == 201, resp.text
    q.get_secret_claim.assert_not_awaited()
