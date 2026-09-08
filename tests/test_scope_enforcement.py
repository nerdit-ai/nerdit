"""Test token scope enforcement through real routers and auth middleware.

Each protected route accepts in-scope and unscoped tokens, and rejects an
out-of-scope token with 403 naming only the target and caller's own scope.
Cover owner/admin checks, NULL-service fail-closed rows, derived-name creates,
intentionally unfiltered reads and refusal to mint scoped admins. Queries are
mocked; no database, network or Docker is used.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.config.settings import DatabasesSettings, ModelsSettings, ServicesSettings
from nerdit.core.data.backend import PostgresBackend
from nerdit.core.data.controller import DataController
from nerdit.core.gitsource import GitSourceInfo
from nerdit.core.models.backend import OllamaBackend
from nerdit.core.models.controller import ModelController
from nerdit.core.secrets import SecretManager
from nerdit.daemon import deploy_pipeline
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import app_templates as app_templates_route
from nerdit.daemon.routes import deploy as deploy_route
from nerdit.daemon.routes.app_config import router as app_config_router
from nerdit.daemon.routes.app_templates import router as app_templates_router
from nerdit.daemon.routes.databases import router as databases_router
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.routes.events import router as events_router
from nerdit.daemon.routes.models import router as models_router
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.rows import Event

LEGACY = "legacy-global"

# The target service every case acts on. The out-of-scope token is scoped
# elsewhere (rather than the target being renamed) so a single set of row
# fixtures serves all three legs of the matrix.
NAME = "demo"
OTHER_NAME = "other-app"

ADMIN_RAW = "admin-raw"
UNSCOPED_RAW = "unscoped-raw"
IN_RAW = "in-scope-raw"
OUT_RAW = "out-of-scope-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(UNSCOPED_RAW): ApiToken(
        id="tok-unscoped",
        name="u",
        role=TokenRole.submitter,
        token_hash=hash_token(UNSCOPED_RAW),
        scope_services=None,
    ),
    hash_token(IN_RAW): ApiToken(
        id="tok-in",
        name="i",
        role=TokenRole.submitter,
        token_hash=hash_token(IN_RAW),
        scope_services=[NAME],
    ),
    hash_token(OUT_RAW): ApiToken(
        id="tok-out",
        name="o",
        role=TokenRole.submitter,
        token_hash=hash_token(OUT_RAW),
        scope_services=[OTHER_NAME],
    ),
}
_TOKEN_IDS = {
    ADMIN_RAW: "tok-admin",
    UNSCOPED_RAW: "tok-unscoped",
    IN_RAW: "tok-in",
    OUT_RAW: "tok-out",
}


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


# --- rows ---------------------------------------------------------------------


def _service_row(owner: str | None, *, config: dict | None = None, name: str = NAME) -> Job:
    return Job(
        id="svc-1",
        service_name=name,
        name=name,
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(config or {"image": "nerdit-app/demo:1", "port": 8000}),
        submitted_by_token=owner,
    )


_GIT_SOURCE = {
    "type": "git",
    "repo_url": "https://github.com/acme/demo",
    "ref": "main",
    "commit_sha": "0" * 40,
}


def _rollback_row(owner: str | None) -> Job:
    return _service_row(
        owner,
        config={
            "image": "nerdit-app/demo:2",
            "previous_image": "nerdit-app/demo:1",
            "build_version": 2,
            "max_version": 2,
            "port": 8000,
        },
    )


def _git_row(owner: str | None) -> Job:
    return _service_row(
        owner,
        config={
            "image": "nerdit-app/demo:1",
            "build_version": 1,
            "max_version": 1,
            "port": 8000,
            "source": dict(_GIT_SOURCE),
        },
    )


# --- harness ------------------------------------------------------------------


def _queries(existing: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_by_name = AsyncMock(return_value=existing)
    q.get_job = AsyncMock(return_value=existing)
    q.get_model_by_ref = AsyncMock(return_value=None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    q.update_service_config = AsyncMock()
    q.update_app_config = AsyncMock()
    q.set_desired_state = AsyncMock()
    q.bump_restart_count = AsyncMock()
    q.list_services = AsyncMock(return_value=([], None))
    return q


def _make_app(queries: AsyncMock, tmp_path: Path) -> FastAPI:
    """One app carrying every router the sweep touches (state is per-router)."""
    app = FastAPI()
    register_error_handlers(app)
    for router in (
        services_router,
        deploy_router,
        app_templates_router,
        secrets_router,
        app_config_router,
        models_router,
        databases_router,
        events_router,
    ):
        app.include_router(router)

    app.state.queries = queries
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime

    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    settings.services = ServicesSettings()
    settings.models = ModelsSettings()
    settings.databases = DatabasesSettings()
    settings.git.enabled = True
    settings.git.allowed_hosts = ["github.com"]
    settings.git.clone_timeout_s = 30.0
    settings.git.max_clone_bytes = 10 * 1024 * 1024
    app.state.settings = settings

    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.model_controller = ModelController(
        OllamaBackend(), MagicMock(), queries, default_backend="ollama"
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
    controller = MagicMock()
    controller.has_active_run = MagicMock(return_value=False)
    controller.has_active_cutover = MagicMock(return_value=False)
    app.state.service_controller = controller

    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path: Path) -> TestClient:
    return TestClient(_make_app(queries, tmp_path), raise_server_exceptions=False)


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "demo", "scripts": {"start": "node i.js"}}))
        zf.writestr("i.js", "console.log('hi')")
    return buf.getvalue()


class _FakeClone:
    """A ``clone_source`` stand-in materializing a real, buildable app tree."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, repo_url, **kwargs):
        self.calls.append(repo_url)
        dest = Path(kwargs["dest_dir"])
        ctx = dest / kwargs["subdir"] if kwargs.get("subdir") else dest
        ctx.mkdir(parents=True, exist_ok=True)
        (ctx / "requirements.txt").write_text("fastapi\n")
        (ctx / "main.py").write_text("app = object()\n")
        return GitSourceInfo(
            commit_sha="1" * 40, resolved_ref=kwargs.get("ref") or "main", context_dir=ctx
        )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No case in this file may reach the network: every clone site is faked."""
    clone = _FakeClone()
    monkeypatch.setattr(deploy_route, "clone_source", clone)
    monkeypatch.setattr(app_templates_route, "clone_source", clone)
    monkeypatch.setattr(deploy_pipeline, "clone_source", clone)
    return clone


# --- the matrix ---------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    """One enforced route: how to seed its row/state and how to call it."""

    id: str
    row: Callable[[str | None], Job | None]
    call: Callable[[TestClient, str], object]
    ok_status: int
    seed: Callable[[Path], None] | None = None


def _seed_secret(tmp_path: Path) -> None:
    """Pre-store the key the delete-one-key case removes (out of band, no route)."""
    SecretManager(tmp_path / "secrets").set(NAME, {"K": "v"})


def _fresh(_owner: str | None) -> None:
    """Create paths see no pre-existing row."""
    return


_CASES: tuple[_Case, ...] = (
    _Case(
        "post_services",
        _fresh,
        lambda c, raw: c.post(
            "/services",
            json={"name": NAME, "image": "nerdit-runtime:0.1", "port": 8000, "gpus": 0},
            headers=_auth(raw),
        ),
        201,
    ),
    _Case(
        "post_deploy_zip",
        _fresh,
        lambda c, raw: c.post(
            "/deploy",
            data={"name": NAME, "port": "8000", "gpus": "0"},
            files={"archive": ("app.zip", _zip_bytes(), "application/zip")},
            headers=_auth(raw),
        ),
        201,
    ),
    _Case(
        "post_deploy_git",
        _fresh,
        lambda c, raw: c.post(
            "/deploy/git",
            json={"repo_url": "https://github.com/acme/demo", "name": NAME},
            headers=_auth(raw),
        ),
        201,
    ),
    _Case(
        "post_deploy_rollback",
        _rollback_row,
        lambda c, raw: c.post(f"/deploy/{NAME}/rollback", headers=_auth(raw)),
        200,
    ),
    _Case(
        "post_deploy_redeploy",
        _git_row,
        lambda c, raw: c.post(f"/deploy/{NAME}/redeploy", headers=_auth(raw)),
        201,
    ),
    _Case(
        "post_app_template_deploy",
        _fresh,
        lambda c, raw: c.post(
            "/app-templates/node-starter/deploy",
            json={"name": NAME},
            headers=_auth(raw),
        ),
        201,
    ),
    _Case(
        "post_secrets",
        _service_row,
        lambda c, raw: c.post(f"/secrets/{NAME}", json={"values": {"K": "v"}}, headers=_auth(raw)),
        200,
    ),
    _Case(
        "get_secrets",
        _service_row,
        lambda c, raw: c.get(f"/secrets/{NAME}", headers=_auth(raw)),
        200,
    ),
    _Case(
        "delete_secrets_all",
        _service_row,
        lambda c, raw: c.delete(f"/secrets/{NAME}", headers=_auth(raw)),
        200,
    ),
    _Case(
        "delete_secrets_key",
        _service_row,
        lambda c, raw: c.delete(f"/secrets/{NAME}/K", headers=_auth(raw)),
        200,
        seed=_seed_secret,
    ),
    _Case(
        "get_app_config",
        _service_row,
        lambda c, raw: c.get(f"/config/apps/{NAME}", headers=_auth(raw)),
        200,
    ),
    _Case(
        "put_app_config",
        _service_row,
        lambda c, raw: c.put(
            f"/config/apps/{NAME}/deploy",
            json={"gpus": 0},
            headers={**_auth(raw), "Idempotency-Key": "ik-1"},
        ),
        200,
    ),
    _Case(
        "post_models",
        _fresh,
        lambda c, raw: c.post(
            "/models", json={"model": "llama3.1:8b", "name": NAME, "gpus": 0}, headers=_auth(raw)
        ),
        201,
    ),
    _Case(
        "post_databases",
        _fresh,
        lambda c, raw: c.post(
            "/databases", json={"backend": "postgres", "name": NAME}, headers=_auth(raw)
        ),
        201,
    ),
)

_CASE_IDS = [case.id for case in _CASES]


def _run(case: _Case, tmp_path: Path, raw: str):
    if case.seed is not None:
        case.seed(tmp_path)
    queries = _queries(case.row(_TOKEN_IDS[raw]))
    return case.call(_client(queries, tmp_path), raw), queries


def _assert_scope_denied(resp, *, target: str = NAME, scope: str = OTHER_NAME) -> None:
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["code"] == "forbidden"
    assert body["message"] == f"This token's scope does not include service '{target}'."
    assert body["hint"] == f"Scoped to: {scope}."


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_in_scope_token_is_allowed(case, tmp_path):
    resp, _ = _run(case, tmp_path, IN_RAW)
    assert resp.status_code == case.ok_status, resp.text


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_out_of_scope_token_is_denied(case, tmp_path):
    resp, _ = _run(case, tmp_path, OUT_RAW)
    _assert_scope_denied(resp)


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_unscoped_token_is_unaffected(case, tmp_path):
    """The zero-breaking-change regression: NULL scope keeps today's behaviour."""
    resp, _ = _run(case, tmp_path, UNSCOPED_RAW)
    assert resp.status_code == case.ok_status, resp.text


@pytest.mark.parametrize("case", _CASES, ids=_CASE_IDS)
def test_out_of_scope_denial_writes_nothing(case, tmp_path):
    """A scope denial fires before every persistent side effect on the route."""
    resp, queries = _run(case, tmp_path, OUT_RAW)
    assert resp.status_code == 403, resp.text
    queries.reserve_service_for_token.assert_not_awaited()
    queries.update_service_config.assert_not_awaited()
    queries.update_app_config.assert_not_awaited()


# --- the denial lands BEFORE the expensive/observable work --------------------


def test_out_of_scope_zip_deploy_never_extracts_the_upload(tmp_path):
    """Rev-2 item 7a: the form ``name`` is authoritative, so the check is pre-extract."""
    queries = _queries(None)
    resp = _client(queries, tmp_path).post(
        "/deploy",
        data={"name": NAME, "port": "8000", "gpus": "0"},
        files={"archive": ("app.zip", _zip_bytes(), "application/zip")},
        headers=_auth(OUT_RAW),
    )
    _assert_scope_denied(resp)
    assert not (tmp_path / "uploads").exists()


def test_out_of_scope_git_deploy_never_clones_nor_resolves_a_credential(tmp_path, _no_network):
    """The check precedes ``_resolve_token_ref`` — which can audit a shared-ref hit."""
    queries = _queries(None)
    resp = _client(queries, tmp_path).post(
        "/deploy/git",
        json={
            "repo_url": "https://github.com/acme/demo",
            "name": NAME,
            "token_ref": "${secrets.shared.GH}",
        },
        headers=_auth(OUT_RAW),
    )
    _assert_scope_denied(resp)
    assert _no_network.calls == []
    queries.insert_audit_log.assert_not_awaited()


def test_out_of_scope_template_deploy_never_clones_nor_writes_secrets(tmp_path, _no_network):
    queries = _queries(None)
    resp = _client(queries, tmp_path).post(
        "/app-templates/node-starter/deploy",
        json={"name": NAME, "secrets": {"API_KEY": "canary-value"}},
        headers=_auth(OUT_RAW),
    )
    _assert_scope_denied(resp)
    assert _no_network.calls == []
    assert SecretManager(tmp_path / "secrets").list_keys(NAME) == []
    assert "canary-value" not in resp.text


# --- leg (a): require_owner_or_admin ------------------------------------------


def test_owner_outside_a_narrowed_scope_is_denied(tmp_path):
    """A scope narrowed AFTER the token created the row still binds (D-P25-3)."""
    # The row is owned by the caller, so only the scope leg can refuse it.
    queries = _queries(_service_row("tok-out"))
    resp = _client(queries, tmp_path).post(f"/services/{NAME}/stop", headers=_auth(OUT_RAW))
    _assert_scope_denied(resp)
    queries.set_desired_state.assert_not_awaited()


def test_owner_inside_scope_still_stops_its_service(tmp_path):
    queries = _queries(_service_row("tok-in"))
    resp = _client(queries, tmp_path).post(f"/services/{NAME}/stop", headers=_auth(IN_RAW))
    assert resp.status_code == 200, resp.text
    queries.set_desired_state.assert_awaited()


def test_null_service_name_row_is_denied_for_any_scoped_token(tmp_path):
    """A row with no ``service_name`` is outside EVERY scope — fail closed."""
    row = _service_row("tok-in")
    row.service_name = None
    queries = _queries(row)
    resp = _client(queries, tmp_path).post("/services/svc-1/stop", headers=_auth(IN_RAW))
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["code"] == "forbidden"
    assert body["message"] == "This token's scope does not include service '(unnamed)'."
    assert body["hint"] == f"Scoped to: {NAME}."
    queries.set_desired_state.assert_not_awaited()


def test_null_service_name_row_still_serves_an_unscoped_owner(tmp_path):
    row = _service_row("tok-unscoped")
    row.service_name = None
    queries = _queries(row)
    resp = _client(queries, tmp_path).post("/services/svc-1/stop", headers=_auth(UNSCOPED_RAW))
    assert resp.status_code == 200, resp.text


def test_admin_bypasses_scope_on_a_foreign_row(tmp_path):
    """Admin is unscopable by construction, so it bypasses both legs."""
    queries = _queries(_service_row("tok-in"))
    resp = _client(queries, tmp_path).post(f"/services/{NAME}/stop", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200, resp.text


def test_diagnose_inherits_the_scope_leg(tmp_path):
    """A read, but already owner-gated — excluding it would be the inconsistency."""
    queries = _queries(_service_row("tok-out"))
    resp = _client(queries, tmp_path).get(f"/services/{NAME}/diagnose", headers=_auth(OUT_RAW))
    _assert_scope_denied(resp)


# --- derived names ------------------------------------------------------------


def test_model_scope_is_checked_on_the_derived_name(tmp_path):
    """No ``name`` override: the scope must cover ``ollama-llama3-1-8b``."""
    queries = _queries(None)
    client = _client(queries, tmp_path)
    resp = client.post("/models", json={"model": "llama3.1:8b", "gpus": 0}, headers=_auth(IN_RAW))
    # ``demo`` is in scope but the DERIVED row name is not.
    _assert_scope_denied(resp, target="ollama-llama3-1-8b", scope=NAME)
    queries.reserve_service_for_token.assert_not_awaited()


def test_model_derived_name_inside_scope_is_allowed(tmp_path):
    queries = _queries(None)
    token = ApiToken(
        id="tok-mdl",
        name="m",
        role=TokenRole.submitter,
        token_hash=hash_token("mdl-raw"),
        scope_services=["ollama-llama3-1-8b"],
    )
    _TOKENS[hash_token("mdl-raw")] = token
    try:
        resp = _client(queries, tmp_path).post(
            "/models", json={"model": "llama3.1:8b", "gpus": 0}, headers=_auth("mdl-raw")
        )
        assert resp.status_code == 201, resp.text
        assert queries.reserve_service_for_token.await_args.args[0].service_name == (
            "ollama-llama3-1-8b"
        )
    finally:
        _TOKENS.pop(hash_token("mdl-raw"))


def test_database_scope_is_checked_on_the_derived_name(tmp_path):
    """No ``name`` override: the backend prefix (``postgres``) is the row name."""
    queries = _queries(None)
    resp = _client(queries, tmp_path).post(
        "/databases", json={"backend": "postgres"}, headers=_auth(IN_RAW)
    )
    _assert_scope_denied(resp, target=PostgresBackend().name_prefix, scope=NAME)
    queries.reserve_service_for_token.assert_not_awaited()


def test_database_derived_name_inside_scope_is_allowed(tmp_path):
    prefix = PostgresBackend().name_prefix
    queries = _queries(None)
    _TOKENS[hash_token("db-raw")] = ApiToken(
        id="tok-db",
        name="d",
        role=TokenRole.submitter,
        token_hash=hash_token("db-raw"),
        scope_services=[prefix],
    )
    try:
        resp = _client(queries, tmp_path).post(
            "/databases", json={"backend": "postgres"}, headers=_auth("db-raw")
        )
        assert resp.status_code == 201, resp.text
        assert queries.reserve_service_for_token.await_args.args[0].service_name == prefix
    finally:
        _TOKENS.pop(hash_token("db-raw"))


# --- reads stay unfiltered (the D-P25-3 sub-ruling) ---------------------------


def test_service_list_is_not_scope_filtered(tmp_path):
    """No list endpoint changes shape in v1 — pinned so a later change is deliberate."""
    queries = _queries(None)
    queries.list_services = AsyncMock(return_value=([_service_row("tok-in")], None))
    resp = _client(queries, tmp_path).get("/services", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text
    assert [s["name"] for s in resp.json()["items"]] == [NAME]


def test_service_detail_is_not_scope_filtered(tmp_path):
    queries = _queries(_service_row("tok-in"))
    resp = _client(queries, tmp_path).get(f"/services/{NAME}", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text


def test_service_logs_are_not_scope_filtered(tmp_path):
    queries = _queries(_service_row("tok-in"))
    queries.get_job_logs = AsyncMock(return_value=[])
    resp = _client(queries, tmp_path).get(f"/services/{NAME}/logs", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text


def test_service_stats_are_not_scope_filtered(tmp_path):
    """P24a read, any-authenticated: a scoped token still observes counters."""
    queries = _queries(_service_row("tok-in"))
    resp = _client(queries, tmp_path).get(f"/services/{NAME}/stats", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text


def test_events_feed_is_not_scope_filtered(tmp_path):
    """The durable feed is service-named yet any-authenticated (rev-2 item 7e)."""
    queries = _queries(None)
    queries.list_events = AsyncMock(
        return_value=([Event(id=1, type="service.healthy", service_name=NAME)], None)
    )
    resp = _client(queries, tmp_path).get("/events", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text
    assert [e["service_name"] for e in resp.json()["items"]] == [NAME]


def test_shared_secret_scope_is_not_a_service_scope(tmp_path):
    """``shared`` is admin-only for writes and open for reads — scope skips it."""
    SecretManager(tmp_path / "secrets").set("_shared", {"K": "v"})
    queries = _queries(None)
    resp = _client(queries, tmp_path).get("/secrets/shared", headers=_auth(OUT_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["keys"] == ["K"]


# --- creation refuses a scoped admin ------------------------------------------


async def test_scoped_admin_is_refused_at_creation(queries):
    """ "Admin bypasses scope" is only coherent because a scoped admin cannot exist."""
    from fastapi import FastAPI as _FastAPI
    from httpx import ASGITransport, AsyncClient

    from nerdit.daemon.routes.tokens import router as tokens_router

    admin = ApiToken(
        name="root", role=TokenRole.admin, token_hash=hash_token("root-raw"), max_gpus=None
    )
    await queries.create_api_token(admin)

    app = _FastAPI()
    register_error_handlers(app)
    app.include_router(tokens_router)
    app.state.queries = queries
    app.state.settings = MagicMock()
    app.state.settings.security.token_default_ttl_s = None
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/tokens",
            json={"name": "agent", "role": "admin", "scope_services": [NAME]},
            headers=_auth("root-raw"),
        )
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "token.scope_not_allowed"


# --- A3: one decision tree, four byte-identical envelopes ---------------------


def _fake_request(principal):  # noqa: ANN001, ANN202
    from types import SimpleNamespace

    return SimpleNamespace(state=SimpleNamespace(principal=principal))


def _envelope(exc) -> tuple:  # noqa: ANN001
    return (exc.status_code, exc.code, exc.message, exc.hint)


def test_require_owner_or_admin_envelopes_are_pinned():
    """``require_owner_or_admin`` now wraps the shared ``_check_owner`` tree; all
    four outcomes must stay byte-identical to the hand-rolled version."""
    from unittest.mock import MagicMock

    from nerdit.daemon.auth import Principal, require_owner_or_admin
    from nerdit.daemon.errors import NerditError
    from nerdit.db.models import TokenRole

    def _job(owner, name="demo"):  # noqa: ANN001, ANN202
        return MagicMock(submitted_by_token=owner, service_name=name)

    admin = Principal(token_id="tok-a", name="a", role=TokenRole.admin)
    owner = Principal(token_id="tok-o", name="o", role=TokenRole.submitter)
    scoped = Principal(
        token_id="tok-o", name="o", role=TokenRole.submitter, scope_services=frozenset({"other"})
    )

    # 1. admin bypass — including the NULL-owner row.
    assert require_owner_or_admin(_fake_request(admin), _job(None)) is admin
    # 2. owner match, unscoped.
    assert require_owner_or_admin(_fake_request(owner), _job("tok-o")) is owner
    # 3. NULL owner is admin-only.
    with pytest.raises(NerditError) as null_owner:
        require_owner_or_admin(_fake_request(owner), _job(None))
    assert _envelope(null_owner.value) == (
        403,
        "forbidden",
        "You do not have permission to act on this job.",
        "Only the submitting token or an admin may manage this job.",
    )
    # 3b. a foreign owner gets the very same envelope.
    with pytest.raises(NerditError) as foreign:
        require_owner_or_admin(_fake_request(owner), _job("tok-x"))
    assert _envelope(foreign.value) == _envelope(null_owner.value)
    # 4. owner match, scope miss — the shared ``_scope_denial`` shape.
    with pytest.raises(NerditError) as miss:
        require_owner_or_admin(_fake_request(scoped), _job("tok-o"))
    assert _envelope(miss.value) == (
        403,
        "forbidden",
        "This token's scope does not include service 'demo'.",
        "Scoped to: other.",
    )


def test_workspace_owner_denial_envelope_is_pinned():
    """The workspace route calls the same primitive with its own refusal text."""
    from nerdit.daemon.auth import Principal
    from nerdit.daemon.errors import NerditError
    from nerdit.daemon.routes.workspaces import _require_workspace_owner
    from nerdit.db.models import TokenRole

    stranger = Principal(token_id="tok-x", name="x", role=TokenRole.submitter)
    with pytest.raises(NerditError) as exc:
        _require_workspace_owner(_fake_request(stranger), "demo", {"owner_token_id": "tok-o"})
    assert _envelope(exc.value) == (
        403,
        "forbidden",
        "You do not have permission to act on workspace 'demo'.",
        "Only the token that created the workspace or an admin may use it.",
    )

    scoped = Principal(
        token_id="tok-o", name="o", role=TokenRole.submitter, scope_services=frozenset({"other"})
    )
    with pytest.raises(NerditError) as miss:
        _require_workspace_owner(_fake_request(scoped), "demo", {"owner_token_id": "tok-o"})
    assert _envelope(miss.value) == (
        403,
        "forbidden",
        "This token's scope does not include service 'demo'.",
        "Scoped to: other.",
    )


# --- NC-0: the non-raising twin must never disagree with the raising gate -----


def test_may_manage_job_agrees_with_require_owner_or_admin_on_every_leg():
    """Pin identical row-field extraction in the UI predicate and authorization gate.

    Both share _owner_or_admin_allows, so agreement does not test that decision
    tree; the leg-(a) and envelope tests do. This matrix catches different
    submitted_by_token/service_name attributes, defaults or argument order.
    Cover admin, NULL owner, owner/foreign owner, scoped/unscoped access, and
    nameless legacy rows so the UI never offers an action the daemon forbids.
    """
    from nerdit.daemon.auth import Principal, may_manage_job, require_owner_or_admin
    from nerdit.daemon.errors import NerditError
    from nerdit.db.models import TokenRole

    def _job(owner, name="demo"):  # noqa: ANN001, ANN202
        return MagicMock(submitted_by_token=owner, service_name=name)

    admin = Principal(token_id="tok-a", name="a", role=TokenRole.admin)
    owner = Principal(token_id="tok-o", name="o", role=TokenRole.submitter)
    scoped_out = Principal(
        token_id="tok-o", name="o", role=TokenRole.submitter, scope_services=frozenset({"other"})
    )
    scoped_in = Principal(
        token_id="tok-o", name="o", role=TokenRole.submitter, scope_services=frozenset({"demo"})
    )
    # The tunnel principal is a plain submitter whose id is synthetic — nothing
    # about the predicate special-cases it, which is exactly the point: the
    # console's verdict is the ordinary ownership verdict.
    tunnel = Principal(token_id="link:node-golden", name="node-link", role=TokenRole.submitter)

    cases = [
        (admin, _job(None)),
        (admin, _job("tok-x")),
        (owner, _job("tok-o")),
        (owner, _job("tok-x")),
        (owner, _job(None)),
        (scoped_out, _job("tok-o")),
        (scoped_in, _job("tok-o")),
        (scoped_in, _job("tok-o", name=None)),
        (tunnel, _job("link:node-golden")),
        (tunnel, _job(None)),
        (tunnel, _job("tok-o")),
    ]
    for principal, job in cases:
        request = _fake_request(principal)
        try:
            require_owner_or_admin(request, job)
            gate_allows = True
        except NerditError:
            gate_allows = False
        assert may_manage_job(request, job) is gate_allows, (
            f"predicate disagrees with the gate for {principal.name} "
            f"on owner={job.submitted_by_token!r} name={job.service_name!r}"
        )


def test_may_manage_job_never_grants_a_null_owner_row_to_a_null_token_id_principal():
    """The fail-*open* this predicate is one ``is not None`` away from.

    ``ANONYMOUS`` — the fail-closed default ``current_principal`` returns when
    no middleware attached one — carries ``token_id=None``, and so does every
    NULL-owner (local/legacy) service row. A bare ``owner == principal.token_id``
    would answer ``True`` here and the console would show live action buttons
    for services nobody may touch. The raising gate is protected from the same
    mistake only because its admin branch fires first for LOCAL/LEGACY_ADMIN;
    ANONYMOUS is readonly, so nothing catches it there either.
    """
    from nerdit.daemon.auth import ANONYMOUS, may_manage_job

    assert may_manage_job(_fake_request(ANONYMOUS), MagicMock(submitted_by_token=None)) is False
