"""Tests for ``POST /deploy/{name}/redeploy`` (P24b WP6 / D-P24-9).

The route is bodyless: every coordinate is read back off ``config['source']``,
so the interesting surface is the guard order (authz BEFORE any clone), the two
"someone else owns this generation" 409s, the ZIP-row and missing-credential
refusals, and the dry-run's zero-write + no-key-poisoning contract.

Harness: the ``test_deploy_route.py`` shape (real router, ``AsyncMock`` queries,
auth middleware) plus the ``test_routes_services.py`` ``_idem_store`` for the
replay tests. ``clone_source`` is faked at the ``deploy_pipeline`` import site —
no network, no git binary — and writes a real Python app tree so the buildpack
``detect()`` downstream is the real one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.core.gitsource import GitSourceError, GitSourceInfo
from nerdit.daemon import deploy_pipeline
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import NerditError, RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.views.service import _cutover_in_progress_error, _run_in_progress_error
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
OTHER_RAW = "other-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(OTHER_RAW): ApiToken(
        id="tok-other", name="o", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}

_GIT_SOURCE = {
    "type": "git",
    "repo_url": "https://github.com/acme/demo",
    "ref": "main",
    "commit_sha": "0" * 40,
}


def _svc(config: dict, *, owner: str | None = "tok-sub") -> Job:
    return Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        submitted_by_token=owner,
        config=json.dumps(config),
    )


def _git_row(*, source: dict | None = None, owner: str | None = "tok-sub") -> Job:
    return _svc(
        {
            "image": "nerdit-app/demo:1",
            "build_version": 1,
            "max_version": 1,
            "source": dict(_GIT_SOURCE if source is None else source),
        },
        owner=owner,
    )


def _queries(existing: Job | None) -> AsyncMock:
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


def _settings(tmp_path) -> MagicMock:
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    settings.git.enabled = True
    settings.git.allowed_hosts = ["github.com"]
    settings.git.github_host = "github.com"
    settings.git.clone_timeout_s = 30.0
    settings.git.max_clone_bytes = 10 * 1024 * 1024
    return settings


def _make_app(
    queries: AsyncMock,
    tmp_path,
    *,
    with_audit: bool = False,
    with_idempotency: bool = False,
    controller=None,
    secret_manager=None,
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.state.queries = queries
    app.state.settings = _settings(tmp_path)
    if controller is not None:
        app.state.service_controller = controller
    if secret_manager is not None:
        app.state.secret_manager = secret_manager
    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path, **kw) -> TestClient:
    return TestClient(_make_app(queries, tmp_path, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _quiet_controller(*, run: bool = False, cutover: bool = False) -> MagicMock:
    """A controller whose two race probes answer explicitly.

    ``has_active_cutover`` is lane-1 (WP5) machinery; the route only ever calls
    it, so the seam is exercised here against an explicit stub rather than a
    MagicMock's truthy auto-attribute.
    """
    controller = MagicMock()
    controller.has_active_run = MagicMock(return_value=run)
    controller.has_active_cutover = MagicMock(return_value=cutover)
    return controller


class _FakeClone:
    """A ``clone_source`` stand-in that writes a real, buildable app tree."""

    def __init__(self, *, error: GitSourceError | None = None, commit_sha: str = "1" * 40):
        self.error = error
        self.commit_sha = commit_sha
        self.calls: list[dict] = []

    async def __call__(self, repo_url, **kwargs):
        self.calls.append({"repo_url": repo_url, **kwargs})
        if self.error is not None:
            raise self.error
        dest = Path(kwargs["dest_dir"])
        sub = dest / kwargs["subdir"] if kwargs.get("subdir") else dest
        sub.mkdir(parents=True, exist_ok=True)
        (sub / "requirements.txt").write_text("fastapi\n")
        (sub / "main.py").write_text("app = object()\n")
        return GitSourceInfo(
            commit_sha=self.commit_sha,
            resolved_ref=kwargs.get("ref") or "main",
            context_dir=sub,
        )


@pytest.fixture
def fake_clone(monkeypatch):
    clone = _FakeClone()
    monkeypatch.setattr(deploy_pipeline, "clone_source", clone)
    return clone


def _post(client: TestClient, raw: str = SUB_RAW, *, dry_run: bool = False, **headers):
    return client.post(
        "/deploy/demo/redeploy",
        params={"dry_run": "true"} if dry_run else None,
        headers={**_auth(raw), **headers},
    )


# --- guard order --------------------------------------------------------------


def test_readonly_is_blocked(tmp_path, fake_clone):
    resp = _post(_client(_queries(_git_row()), tmp_path), RO_RAW)
    assert resp.status_code == 403
    assert not fake_clone.calls


def test_unknown_service_is_404(tmp_path, fake_clone):
    resp = _post(_client(_queries(None), tmp_path))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert not fake_clone.calls


def test_non_owner_submitter_is_403_before_any_clone(tmp_path, fake_clone):
    """The R5 ordering: an unauthorized redeploy must not consume clone
    timeout/bytes or resolve a private-repo credential."""
    resp = _post(_client(_queries(_git_row()), tmp_path), OTHER_RAW)
    assert resp.status_code == 403
    assert not fake_clone.calls, "the ownership gate must run BEFORE the clone"


def test_admin_may_redeploy_another_owners_service(tmp_path, fake_clone):
    resp = _post(_client(_queries(_git_row(owner="tok-other")), tmp_path), LEGACY)
    assert resp.status_code == 201, resp.text


def test_git_disabled_is_403(tmp_path, fake_clone):
    q = _queries(_git_row())
    app = _make_app(q, tmp_path)
    app.state.settings.git.enabled = False
    resp = _post(TestClient(app, raise_server_exceptions=False))
    assert resp.status_code == 403
    assert resp.json()["code"] == "deploy.git_disabled"
    assert not fake_clone.calls


def test_active_run_is_409(tmp_path, fake_clone):
    q = _queries(_git_row())
    client = _client(q, tmp_path, controller=_quiet_controller(run=True))
    resp = _post(client)
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    # The shared factory's default hint reaches the wire, naming the bound.
    assert "[services].release_timeout_s" in resp.json()["hint"]
    q.update_service_config.assert_not_awaited()
    assert not fake_clone.calls


def test_a_run_registering_during_the_clone_is_409(tmp_path, fake_clone):
    """The pre-clone check only proves the row was quiet when the clone STARTED.

    A `nerdit services run` (or a release) registering inside the clone window —
    bounded only by ``[git].clone_timeout_s`` — would otherwise have its image
    swapped out from under it by the finalize. The re-check after the clone is
    what makes the guard cover the whole window, on the HTTP route and the
    GitWatch path alike (both go through the same primitive)."""
    q = _queries(_git_row())
    controller = _quiet_controller()
    # Quiet before the clone, active by the time it returns.
    controller.has_active_run = MagicMock(side_effect=lambda _id: bool(fake_clone.calls))
    resp = _post(_client(q, tmp_path, controller=controller))

    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    assert fake_clone.calls, "the refusal is the POST-clone one"
    q.update_service_config.assert_not_awaited()
    # The cloned context is cleaned up on refusal, like every other late failure.
    assert not list((tmp_path / "uploads").iterdir())


def test_active_cutover_is_409(tmp_path, fake_clone):
    """A redeploy landing mid-cutover would overwrite the generation the green
    container is being probed against (WP5.5)."""
    q = _queries(_git_row())
    client = _client(q, tmp_path, controller=_quiet_controller(cutover=True))
    resp = _post(client)
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.cutover_in_progress"
    assert "[services].cutover_verify_timeout_s" in resp.json()["hint"]
    q.update_service_config.assert_not_awaited()
    assert not fake_clone.calls


def test_guards_are_skipped_when_no_controller_is_wired(tmp_path, fake_clone):
    """None-tolerant, the same posture as the rollback/DELETE hooks."""
    q = _queries(_git_row())
    app = _make_app(q, tmp_path)
    assert not hasattr(app.state, "service_controller")
    resp = _post(TestClient(app, raise_server_exceptions=False))
    assert resp.status_code == 201, resp.text


def test_rollback_is_refused_while_a_cutover_is_in_flight(tmp_path):
    """The same 409 lands on the EXISTING rollback route (plan WP5.5)."""
    q = _queries(_svc({"image": "nerdit-app/demo:2", "previous_image": "nerdit-app/demo:1"}))
    client = _client(q, tmp_path, controller=_quiet_controller(cutover=True))
    resp = client.post("/deploy/demo/rollback", headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.cutover_in_progress"
    q.update_service_config.assert_not_awaited()


# --- source provenance --------------------------------------------------------


def test_zip_row_is_409_no_source(tmp_path, fake_clone):
    q = _queries(_svc({"image": "nerdit-app/demo:1", "source": {"type": "zip"}}))
    resp = _post(_client(q, tmp_path))
    assert resp.status_code == 409
    assert resp.json()["code"] == "deploy.no_source"
    assert not fake_clone.calls


def test_row_without_any_source_is_409_no_source(tmp_path, fake_clone):
    """A pre-P11.5 row carries no ``source`` block at all."""
    resp = _post(_client(_queries(_svc({"image": "nerdit-app/demo:1"})), tmp_path))
    assert resp.status_code == 409
    assert resp.json()["code"] == "deploy.no_source"


def test_redeploy_workspace_row_409_with_workspace_hint(tmp_path, fake_clone):
    """(P29 D-P29-7) A workspace row keeps the ``deploy.no_source`` code but must
    point at the verb that DOES work — the generic "re-upload with nerdit deploy"
    hint sends an agent down a path its topology cannot take."""
    q = _queries(_svc({"image": "nerdit-app/demo:1", "source": {"type": "workspace"}}))
    resp = _post(_client(q, tmp_path))
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "deploy.no_source"
    assert "deploy_app" in body["hint"]
    assert "/api/workspaces/demo/deploy" in body["hint"]
    assert "nerdit deploy <path>" not in body["hint"]
    assert not fake_clone.calls


def test_read_git_source_gives_the_workspace_hint_on_the_poller_path():
    """The same branch in ``deploy_pipeline._read_git_source`` — the copy the
    GitWatch poller hits. A route-only fix would leave this one stale, so it is
    pinned separately rather than through the HTTP surface."""
    job = _svc({"image": "nerdit-app/demo:1", "source": {"type": "workspace"}})
    with pytest.raises(NerditError) as exc:
        deploy_pipeline._read_git_source(job, "demo")
    assert exc.value.code == "deploy.no_source"
    assert "deploy_app" in (exc.value.hint or "")

    # Falsified the other way: a ZIP row still gets the git-only wording.
    zip_job = _svc({"image": "nerdit-app/demo:1", "source": {"type": "zip"}})
    with pytest.raises(NerditError) as zip_exc:
        deploy_pipeline._read_git_source(zip_job, "demo")
    assert "nerdit deploy <path>" in (zip_exc.value.hint or "")


@pytest.mark.parametrize(
    "source",
    [
        {"type": "workspace"},
        {"type": "zip"},
        None,  # a pre-P11.5 row with no ``source`` block at all
        {"type": "git"},  # git provenance with no repo_url
    ],
)
def test_redeploy_refusal_shared_with_the_poller(tmp_path, fake_clone, source):
    """(P29 D13) Route and poller raise ONE refusal, not two copies of it.

    The route used to inline its own three-way branch beside
    ``deploy_pipeline._read_git_source``; both are now the single primitive, so
    every ``deploy.no_source`` envelope must be byte-identical on both paths.
    This is the guard against a future re-fork.
    """
    cfg: dict = {"image": "nerdit-app/demo:1"}
    if source is not None:
        cfg["source"] = source

    resp = _post(_client(_queries(_svc(cfg)), tmp_path))
    assert resp.status_code == 409
    body = resp.json()

    with pytest.raises(NerditError) as exc:
        deploy_pipeline._read_git_source(_svc(cfg), "demo")

    assert body["code"] == exc.value.code
    assert body["detail"] == exc.value.message
    assert body["hint"] == exc.value.hint
    assert not fake_clone.calls


def test_recorded_coordinates_drive_the_clone(tmp_path, fake_clone):
    source = {**_GIT_SOURCE, "ref": "release", "subdir": "apps/web"}
    q = _queries(_git_row(source=source))
    resp = _post(_client(q, tmp_path))
    assert resp.status_code == 201, resp.text
    call = fake_clone.calls[0]
    assert call["repo_url"] == "https://github.com/acme/demo"
    assert call["ref"] == "release"
    assert call["subdir"] == "apps/web"
    assert call["token"] is None
    # The freshly resolved commit replaces the recorded one.
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["source"]["commit_sha"] == "1" * 40
    assert cfg["source"]["subdir"] == "apps/web"


def test_a_host_removed_from_the_allowlist_blocks_the_redeploy(tmp_path, fake_clone):
    """The allowlist is re-checked against the CURRENT config, not the one that
    was in force at the original deploy."""
    q = _queries(_git_row())
    app = _make_app(q, tmp_path)
    app.state.settings.git.allowed_hosts = ["gitlab.com"]
    resp = _post(TestClient(app, raise_server_exceptions=False))
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.git_host_forbidden"
    assert not fake_clone.calls


# --- private-repo credential (D-P24-14) ---------------------------------------


def _secret_manager(store: dict[str, dict[str, str]]) -> MagicMock:
    mgr = MagicMock()
    mgr.load = MagicMock(side_effect=lambda scope: dict(store.get(scope, {})))
    return mgr


def test_recorded_token_ref_is_resolved_and_carried_forward(tmp_path, fake_clone):
    source = {**_GIT_SOURCE, "token_ref": "${secrets.GH_TOKEN}"}
    q = _queries(_git_row(source=source))
    client = _client(
        q, tmp_path, secret_manager=_secret_manager({"demo": {"GH_TOKEN": "ghp-live"}})
    )
    resp = _post(client)
    assert resp.status_code == 201, resp.text
    assert fake_clone.calls[0]["token"] == "ghp-live"
    cfg = json.loads(q.update_service_config.call_args.args[1])
    # The REFERENCE survives so the next redeploy is still unattended; the raw
    # token never reaches the persisted blob.
    assert cfg["source"]["token_ref"] == "${secrets.GH_TOKEN}"
    assert "ghp-live" not in json.dumps(cfg)


def test_private_repo_without_a_recorded_token_ref_is_409(tmp_path, monkeypatch):
    """A pre-P24b row records no ``token_ref``, so its unauthenticated re-clone
    comes back as an auth challenge — reported as the missing reference, not as
    a raw git failure."""
    clone = _FakeClone(
        error=GitSourceError(
            400,
            "deploy.git_clone_failed",
            "git clone failed: fatal: could not read Username for 'https://github.com'",
            hint="repository not found or private — ...",
        )
    )
    monkeypatch.setattr(deploy_pipeline, "clone_source", clone)
    q = _queries(_git_row())
    resp = _post(_client(q, tmp_path))
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "deploy.no_source_credential"
    assert "--token-ref" in body["hint"]
    # The hinted command must be one the CLI accepts: the service name rides
    # behind --name (a positional path + --repo is rejected by the git-mode
    # guard).
    assert "--name" in body["hint"]
    q.update_service_config.assert_not_awaited()


def test_a_non_auth_clone_failure_keeps_its_own_code(tmp_path, monkeypatch):
    clone = _FakeClone(
        error=GitSourceError(
            400, "deploy.git_clone_failed", "git clone failed: fatal: Remote branch not found"
        )
    )
    monkeypatch.setattr(deploy_pipeline, "clone_source", clone)
    resp = _post(_client(_queries(_git_row()), tmp_path))
    assert resp.status_code == 400
    assert resp.json()["code"] == "deploy.git_clone_failed"


def test_an_unresolvable_recorded_reference_is_409_before_the_clone(tmp_path, fake_clone):
    source = {**_GIT_SOURCE, "token_ref": "${secrets.GH_TOKEN}"}
    q = _queries(_git_row(source=source))
    client = _client(q, tmp_path, secret_manager=_secret_manager({}))
    resp = _post(client)
    assert resp.status_code == 409
    assert resp.json()["code"] == "deploy.no_source_credential"
    assert not fake_clone.calls


def test_the_git_route_stamps_token_ref_into_source_meta(tmp_path, monkeypatch):
    """WP6.3: the original ``POST /deploy/git`` records the reference NAME."""
    from nerdit.daemon.routes import deploy as deploy_routes

    clone = _FakeClone()
    monkeypatch.setattr(deploy_routes, "clone_source", clone)
    q = _queries(_git_row(source={"type": "zip"}))
    client = _client(
        q, tmp_path, secret_manager=_secret_manager({"demo": {"GH_TOKEN": "ghp-live"}})
    )
    resp = client.post(
        "/deploy/git",
        json={
            "repo_url": "https://github.com/acme/demo",
            "name": "demo",
            "token_ref": "${secrets.GH_TOKEN}",
        },
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["source"]["token_ref"] == "${secrets.GH_TOKEN}"
    assert "ghp-live" not in json.dumps(cfg)


# --- dry run ------------------------------------------------------------------


def test_dry_run_writes_nothing_and_returns_the_plan(tmp_path, fake_clone):
    q = _queries(_git_row())
    resp = _post(_client(q, tmp_path), dry_run=True)
    assert resp.status_code == 200, resp.text
    assert resp.json()["dry_run"] is True
    assert fake_clone.calls, "the clone still runs — the plan is computed from the real tree"
    q.update_service_config.assert_not_awaited()
    q.reserve_service_for_token.assert_not_awaited()


def test_dry_run_is_audited_as_the_plan_action(tmp_path, fake_clone):
    q = _queries(_git_row())
    resp = _post(_client(q, tmp_path, with_audit=True), dry_run=True)
    assert resp.status_code == 200, resp.text
    assert q.insert_audit_log.await_args.kwargs["action"] == "deploy.redeploy_plan"


def test_a_real_redeploy_is_audited_as_deploy_redeploy(tmp_path, fake_clone):
    q = _queries(_git_row())
    resp = _post(_client(q, tmp_path, with_audit=True))
    assert resp.status_code == 201, resp.text
    row = q.insert_audit_log.await_args.kwargs
    assert row["action"] == "deploy.redeploy"
    # The target comes from path group 1 — no ``_TARGET_STAMP_ALLOWED`` entry.
    assert row["target_id"] == "demo"
    assert row["target_type"] == "service"


def test_a_refused_redeploy_is_audited_as_denied(tmp_path, fake_clone):
    q = _queries(_git_row())
    assert _post(_client(q, tmp_path, with_audit=True), OTHER_RAW).status_code == 403
    rows = [
        c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == "deploy.redeploy"
    ]
    assert rows and rows[-1].kwargs["result"] == "denied"


# --- the system-principal shim (P24c WP10) ------------------------------------


async def test_system_shim_redeploys_without_a_request(tmp_path, fake_clone):
    """The GitWatch poller has no ``Request``. The shim supplies the four facets
    the pipeline reads and runs as an admin ``system`` principal, so a row owned
    by someone else still redeploys unattended — in-process, no HTTP self-call."""
    job = _git_row(owner="tok-other")
    config = json.loads(job.config)
    config["build_overrides"] = {"build": False, "start": "python custom.py"}
    job.config = json.dumps(config)
    q = _queries(job)
    app = _make_app(q, tmp_path)

    result = await deploy_pipeline.redeploy_from_source(
        request_or_none=None,
        app=app,
        queries=q,
        settings=app.state.settings,
        secrets=None,
        job=job,
        principal="system",
    )

    assert result["name"] == "demo"
    assert fake_clone.calls, "the recorded source is re-cloned"
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["source"]["commit_sha"] == "1" * 40
    assert cfg["build_overrides"] == config["build_overrides"]
    assert result["build"]["build"] is False
    assert result["build"]["start"] == "python custom.py"
    # No middleware runs on this path: the poller's own deploy.auto_redeploy row
    # is the audit record, not a route-middleware row.
    q.insert_audit_log.assert_not_awaited()


async def test_system_shim_is_refused_while_a_run_is_active(tmp_path, fake_clone):
    """The run/release guard lives in the PRIMITIVE, so the poller inherits it.

    A P20 run keeps the row ``running`` — inside the poller's candidate set — so
    without this the one ingress that has no human behind it would be the one
    that bumps the generation a live release owns."""
    job = _git_row()
    q = _queries(job)
    app = _make_app(q, tmp_path, controller=_quiet_controller(run=True))

    with pytest.raises(NerditError) as exc:
        await deploy_pipeline.redeploy_from_source(
            request_or_none=None,
            app=app,
            queries=q,
            settings=app.state.settings,
            secrets=None,
            job=job,
            principal="system",
        )

    assert (exc.value.status_code, exc.value.code) == (409, "service.run_in_progress")
    q.update_service_config.assert_not_awaited()
    assert not fake_clone.calls


async def test_system_shim_requires_an_app(tmp_path, fake_clone):
    job = _git_row()
    with pytest.raises(RuntimeError):
        await deploy_pipeline.redeploy_from_source(
            request_or_none=None,
            app=None,
            queries=_queries(job),
            settings=MagicMock(),
            secrets=None,
            job=job,
            principal="system",
        )
    assert not fake_clone.calls


# --- the shared 409 factories -------------------------------------------------
# Every deploy-surface guard (rollback, redeploy, plain deploy-over, delete)
# builds its refusal here, so the wording is pinned once instead of drifting
# per call site.


def test_run_in_progress_factory_shape():
    exc = _run_in_progress_error("demo", "roll back")
    assert (exc.status_code, exc.code) == (409, "service.run_in_progress")
    assert "cannot roll back." in exc.message
    assert "demo" in exc.message
    assert "[services].release_timeout_s" in exc.hint


def test_cutover_in_progress_factory_shape():
    exc = _cutover_in_progress_error("demo", "redeploy")
    assert (exc.status_code, exc.code) == (409, "service.cutover_in_progress")
    assert "cannot redeploy." in exc.message
    assert "demo" in exc.message
    assert "[services].cutover_verify_timeout_s" in exc.hint


def test_factory_hint_override_wins():
    """DELETE keeps its ?force hatch hint while sharing the message stem."""
    assert (
        _run_in_progress_error("demo", "delete", hint="use ?force=true").hint == "use ?force=true"
    )
    assert (
        _cutover_in_progress_error("demo", "delete", hint="use ?force=true").hint
        == "use ?force=true"
    )


# --- idempotency --------------------------------------------------------------


def _idem_store(q: AsyncMock) -> dict:
    """A principal-scoped in-memory idempotency store on the queries mock."""
    records: dict[tuple[str, str], SimpleNamespace] = {}

    async def _insert(*, principal_id, idem_key, method, path, expires_at, body_hash=None):
        if (principal_id, idem_key) in records:
            return False
        records[(principal_id, idem_key)] = SimpleNamespace(
            state="in_progress",
            method=method,
            path=path,
            response_status=None,
            response_body=None,
            content_type=None,
            resource_id=None,
            body_hash=body_hash,
        )
        return True

    async def _get(principal_id, idem_key):
        return records.get((principal_id, idem_key))

    async def _complete(
        *, principal_id, idem_key, response_status, response_body, content_type, resource_id
    ):
        rec = records[(principal_id, idem_key)]
        rec.state = "completed"
        rec.response_status = response_status
        rec.response_body = response_body
        rec.content_type = content_type
        rec.resource_id = resource_id

    async def _delete(principal_id, idem_key):
        records.pop((principal_id, idem_key), None)

    q.insert_idempotency_inprogress = AsyncMock(side_effect=_insert)
    q.get_idempotency_record = AsyncMock(side_effect=_get)
    q.complete_idempotency_record = AsyncMock(side_effect=_complete)
    q.delete_idempotency_record = AsyncMock(side_effect=_delete)
    return records


def test_same_key_replays_instead_of_rebuilding(tmp_path, fake_clone):
    q = _queries(_git_row())
    _idem_store(q)
    client = _client(q, tmp_path, with_idempotency=True)
    headers = {"Idempotency-Key": "redeploy-1"}

    first = _post(client, **headers)
    assert first.status_code == 201, first.text
    second = _post(client, **headers)
    assert second.status_code == 201
    assert second.headers.get("Idempotent-Replay") == "true"
    assert len(fake_clone.calls) == 1, "a replay must not re-clone or rebuild"
    assert q.update_service_config.await_count == 1


def test_a_stray_key_on_a_dry_run_does_not_poison_the_later_real_redeploy(tmp_path, fake_clone):
    """The P13 §1.10 regression: a dry run bypasses the key claim entirely, so
    reusing that key for the real redeploy must still execute."""
    q = _queries(_git_row())
    records = _idem_store(q)
    client = _client(q, tmp_path, with_idempotency=True)
    headers = {"Idempotency-Key": "shared-key"}

    plan = _post(client, dry_run=True, **headers)
    assert plan.status_code == 200, plan.text
    assert records == {}, "a dry run must not claim the key"

    real = _post(client, **headers)
    assert real.status_code == 201, real.text
    assert q.update_service_config.await_count == 1


def test_a_key_in_flight_is_409(tmp_path, fake_clone):
    import asyncio

    q = _queries(_git_row())
    _idem_store(q)
    asyncio.run(
        q.insert_idempotency_inprogress(
            principal_id="tok-sub",
            idem_key="redeploy-2",
            method="POST",
            path="/deploy/demo/redeploy",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    )
    resp = _post(_client(q, tmp_path, with_idempotency=True), **{"Idempotency-Key": "redeploy-2"})
    assert resp.status_code == 409
    assert resp.json()["code"] == "idempotency_in_progress"
    assert not fake_clone.calls


def test_a_failed_redeploy_does_not_pin_the_key(tmp_path, monkeypatch):
    """A non-2xx releases the claim so the same key can be retried."""
    clone = _FakeClone(error=GitSourceError(400, "deploy.git_clone_failed", "boom"))
    monkeypatch.setattr(deploy_pipeline, "clone_source", clone)
    q = _queries(_git_row())
    records = _idem_store(q)
    resp = _post(_client(q, tmp_path, with_idempotency=True), **{"Idempotency-Key": "redeploy-3"})
    assert resp.status_code == 400
    assert records == {}


# --- surface parity: client + CLI verb (WP6.4) --------------------------------


@pytest.mark.asyncio
async def test_client_mints_a_key_for_a_real_redeploy_and_stays_keyless_on_a_dry_run():
    import httpx

    from nerdit.cli.client import NerditClient

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "path": request.url.path,
                "query": dict(request.url.params),
                "idem": request.headers.get("idempotency-key"),
            }
        )
        return httpx.Response(201, json={"name": "demo"})

    client = NerditClient(
        host="localhost", port=9321, token=None, transport=httpx.MockTransport(handler)
    )
    await client.redeploy_app("demo")
    await client.redeploy_app("demo", dry_run=True)
    await client.redeploy_app("demo", idempotency_key="explicit")

    assert [c["path"] for c in seen] == ["/api/deploy/demo/redeploy"] * 3
    assert seen[0]["idem"] and "dry_run" not in seen[0]["query"]
    assert seen[1]["idem"] is None and seen[1]["query"]["dry_run"] == "true"
    assert seen[2]["idem"] == "explicit"


def test_cli_redeploy_verb_renders_and_exits_zero(monkeypatch):
    from typer.testing import CliRunner

    from nerdit.cli import client as client_mod
    from nerdit.cli.app import app

    monkeypatch.setenv("COLUMNS", "300")
    fake = MagicMock()
    fake.redeploy_app = AsyncMock(
        return_value={"name": "demo", "status": "restarting", "endpoint": {"url": "http://x"}}
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    result = CliRunner().invoke(app, ["services", "redeploy", "demo"])
    assert result.exit_code == 0, result.output
    assert "Redeploy accepted" in result.output
    fake.redeploy_app.assert_awaited_once_with("demo", dry_run=False)


# --- P25: edge auth survives a push-triggered redeploy -------------------------


def test_redeploy_carries_edge_auth_forward(tmp_path, fake_clone):
    """The bodyless redeploy is the ingress the P24c GitWatch poller drives with
    NO human in the loop: a push must never be able to unprotect a published app.
    ``edge_auth`` has no form field, so only the carry-forward keeps it — and the
    only disarm is an explicit ``deploy edge_auth=null``."""
    edge_auth = {"user": "ops", "password": "${secrets.APP_PW}"}
    row = _git_row()
    cfg = json.loads(row.config)
    cfg["edge_auth"] = edge_auth
    row.config = json.dumps(cfg)
    q = _queries(row)
    resp = _post(_client(q, tmp_path))
    assert resp.status_code == 201, resp.text
    written = json.loads(q.update_service_config.await_args.args[1])
    assert written["edge_auth"] == edge_auth


# --- P33 WP-R: a recorded ${github.installation} (D-GH-3 / D-GH-9) -----------


def _link_manager(
    tokens: dict[str, str],
    *,
    link_state: str = "connected",
    pro_mirror: str = "never",
) -> MagicMock:
    """As in ``tests/test_deploy_git_route.py``: both link reads spelled out.

    P34 D3 branches on ``status().state`` and ``pro_mirror_state()``; leaving
    them as bare ``MagicMock`` attributes would make the generic hint an
    accident of mock identity rather than a decision. Defaults are the ordinary
    P33 node: online, nothing asserted about the plan.
    """
    mgr = MagicMock()
    mgr.github_token_for_repo = MagicMock(side_effect=lambda slug: tokens.get(slug))
    # The auth middleware also consults a wired manager for relay capabilities:
    # an explicit "no" keeps the test bearers resolving as ordinary tokens.
    mgr.validate_capability = MagicMock(return_value=False)
    mgr.status = MagicMock(return_value=SimpleNamespace(state=link_state))
    mgr.pro_mirror_state = MagicMock(return_value=pro_mirror)
    return mgr


def test_the_redeploy_raise_site_is_tier_aware_too(tmp_path, fake_clone):
    """(P34 D3) The pipeline's ``_resolve_source_token`` is the SECOND raise
    site of ``github_token_absent_error`` — the redeploy path a user triggers by
    hand — and it takes the same one decision function as the git route, so the
    two cannot drift into telling the same account two different stories.

    Everything but the hint stays the P33 envelope: same status, same code.
    """
    source = {**_GIT_SOURCE, "token_ref": "${github.installation}"}
    q = _queries(_git_row(source=source))
    app = _make_app(q, tmp_path, secret_manager=_secret_manager({}))
    app.state.link_manager = _link_manager({}, link_state="connected", pro_mirror="fresh_false")

    resp = TestClient(app, raise_server_exceptions=False).post(
        "/deploy/demo/redeploy", headers=_auth(SUB_RAW)
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.github_token_absent"
    assert body["hint"] == (
        "this account's plan does not include GitHub deploys — see the Nerdit console"
    )
    assert not fake_clone.calls


def test_the_redeploy_hint_stays_generic_on_a_stale_mirror(tmp_path, fake_clone):
    """The failure D-X16-37 exists to prevent, pinned on the redeploy path too:
    a day of cloud silence must not turn into "your plan is too small"."""
    source = {**_GIT_SOURCE, "token_ref": "${github.installation}"}
    q = _queries(_git_row(source=source))
    app = _make_app(q, tmp_path, secret_manager=_secret_manager({}))
    app.state.link_manager = _link_manager({}, link_state="connected", pro_mirror="stale")

    resp = TestClient(app, raise_server_exceptions=False).post(
        "/deploy/demo/redeploy", headers=_auth(SUB_RAW)
    )

    assert resp.status_code == 422
    assert resp.json()["hint"] == (
        "link this node and install the Nerdit GitHub App, or pass a `${secrets.*}` token_ref"
    )
    assert not fake_clone.calls


def test_recorded_github_ref_resolves_by_repo_and_is_carried_forward(tmp_path, fake_clone):
    sentinel = "ghs-zqxjkw-ZQXJKW"
    source = {**_GIT_SOURCE, "token_ref": "${github.installation}"}
    q = _queries(_git_row(source=source))
    mgr = _secret_manager({})
    app = _make_app(q, tmp_path, with_audit=True, secret_manager=mgr)
    app.state.link_manager = _link_manager({"acme/demo": sentinel})
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/deploy/demo/redeploy", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 201, resp.text
    assert fake_clone.calls[0]["token"] == sentinel
    app.state.link_manager.github_token_for_repo.assert_called_once_with("acme/demo")
    # The secret store is never consulted for the literal.
    mgr.load.assert_not_called()
    # The reference NAME rides forward; the value never lands anywhere.
    cfg = json.loads(q.update_service_config.await_args.args[1])
    assert cfg["source"]["token_ref"] == "${github.installation}"
    assert sentinel not in json.dumps(cfg)
    assert sentinel not in json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list])
    assert sentinel not in resp.text


def test_recorded_github_ref_without_a_token_is_422_before_the_clone(tmp_path, fake_clone):
    source = {**_GIT_SOURCE, "token_ref": "${github.installation}"}
    q = _queries(_git_row(source=source))
    client = _client(q, tmp_path, secret_manager=_secret_manager({}))
    resp = _post(client)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.github_token_absent"
    assert "Nerdit GitHub App" in body["hint"]
    assert "${secrets.*}" in body["hint"]
    assert not fake_clone.calls
    q.update_service_config.assert_not_awaited()


def test_recorded_github_ref_for_a_repo_outside_every_installation_is_422(tmp_path, fake_clone):
    source = {**_GIT_SOURCE, "token_ref": "${github.installation}"}
    q = _queries(_git_row(source=source))
    app = _make_app(q, tmp_path, secret_manager=_secret_manager({}))
    app.state.link_manager = _link_manager({"other/repo": "ghs-x"})
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/deploy/demo/redeploy", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.github_token_absent"
    assert not fake_clone.calls
