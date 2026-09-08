"""Test git deployment with real validation, middleware and buildpack detection.

Mock only outbound cloning and queries; materialize a real Node context. Cover
SSRF and credential rejection, initial deployment, redeploy source replacement,
secret-reference resolution and redaction, error envelopes, and authorization
before cloning.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import nerdit.daemon.routes.deploy as deploy_mod
from nerdit.core.gitsource import GitSourceError, GitSourceInfo
from nerdit.daemon.audit import AuditMiddleware, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"
REPO = "https://github.com/owner/repo"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


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


def _secret_manager(store: dict[str, dict[str, str]] | None = None) -> MagicMock:
    """A SecretManager double whose ``load(scope)`` returns *store*[scope] or {}."""
    mgr = MagicMock()
    data = store or {}
    mgr.load = MagicMock(side_effect=lambda scope: data.get(scope, {}))
    return mgr


def _make_app(
    queries: AsyncMock,
    tmp_path,
    *,
    with_audit: bool = False,
    git_enabled: bool = True,
    secret_manager: MagicMock | None = None,
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    settings.git.enabled = git_enabled
    settings.git.allowed_hosts = ["github.com"]
    settings.git.github_host = "github.com"
    settings.git.clone_timeout_s = 120
    settings.git.max_clone_bytes = 10 * 1024 * 1024
    app.state.settings = settings
    app.state.secret_manager = secret_manager or _secret_manager()
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path, **kw) -> TestClient:
    return TestClient(_make_app(queries, tmp_path, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _fake_clone(commit: str = "a" * 40, resolved: str = "main") -> AsyncMock:
    """An AsyncMock clone that materializes a real Node context dir under dest_dir."""

    async def _run(repo_url, *, dest_dir, ref=None, subdir=None, **_kw) -> GitSourceInfo:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        # The context dir is the subdir when given; materialize the Node app there.
        ctx = dest / subdir if subdir else dest
        ctx.mkdir(parents=True, exist_ok=True)
        (ctx / "package.json").write_text(
            json.dumps({"name": "demo", "scripts": {"start": "node index.js"}})
        )
        (ctx / "index.js").write_text("console.log('hi')")
        return GitSourceInfo(commit_sha=commit, resolved_ref=ref or resolved, context_dir=ctx)

    return AsyncMock(side_effect=_run)


def _post(client: TestClient, monkeypatch, clone: AsyncMock, raw: str = SUB_RAW, **body):
    monkeypatch.setattr(deploy_mod, "clone_source", clone)
    payload = {"repo_url": REPO, "name": "demo"}
    payload.update(body)
    return client.post("/deploy/git", json=payload, headers=_auth(raw))


def _existing(cfg: dict | None = None, *, owner: str = "tok-sub") -> Job:
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


# --- R1: fresh 201 + source stamp --------------------------------------------


def test_r1_fresh_deploy_stamps_git_source(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "demo"
    assert body["status"] == JobStatus.building.value

    job = q.reserve_service_for_token.call_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["source"] == {
        "type": "git",
        "repo_url": REPO,
        "repo": "github.com/owner/repo",  # P33 D-GH-6 canonical identity
        "ref": "main",
        "commit_sha": "a" * 40,
    }
    # (P33 D-GH-10) The generation carries the same provenance, resolved at
    # clone time, plus a null remediation_code until it settles.
    ld = cfg["last_deploy"]
    assert (ld["repo"], ld["ref"], ld["sha"], ld["remediation_code"]) == (
        "github.com/owner/repo",
        "main",
        "a" * 40,
        None,
    )
    assert "subdir" not in cfg["source"]
    assert cfg["image"] == "nerdit-app/demo:1"
    clone.assert_awaited_once()


def test_r1_idempotency_key_persisted(tmp_path, monkeypatch):
    q = _queries()
    monkeypatch.setattr(deploy_mod, "clone_source", _fake_clone())
    resp = _client(q, tmp_path).post(
        "/deploy/git",
        json={"repo_url": REPO, "name": "demo"},
        headers={**_auth(SUB_RAW), "Idempotency-Key": "key-123"},
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.idempotency_key == "key-123"


def test_r1_fresh_git_deploy_stamps_queued_last_deploy(tmp_path, monkeypatch):
    """P13 WP2: the git ingress seeds ``config['last_deploy']`` = queued through the
    shared ``_finalize_deploy`` (create, version 1, forensics-clear) and the 201
    body projects it."""
    q = _queries()
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    ld = cfg["last_deploy"]
    assert ld["phase"] == "queued"
    assert ld["action"] == "create"
    assert ld["version"] == 1
    assert ld["reason"] is None
    assert ld["image"] == "nerdit-app/demo:1"
    # No stale crash forensics on a fresh generation.
    for key in ("last_exit_code", "oom_killed", "last_crash_at"):
        assert key not in cfg
    # The 201 body carries the stamp (ServiceResponse.last_deploy projection).
    assert resp.json()["last_deploy"] == ld


def test_r2_git_redeploy_stamps_queued_and_clears_forensics(tmp_path, monkeypatch):
    """P13 WP2: a git redeploy re-seeds queued at the bumped version and drops the
    previous generation's crash forensics carried by the dict(prev_cfg) copy."""
    existing = _existing(
        {
            "last_exit_code": 137,
            "oom_killed": True,
            "last_crash_at": "2026-07-09T00:00:00+00:00",
            "last_deploy": {"version": 1, "action": "create", "phase": "failed"},
        }
    )
    q = _queries(existing)
    written: dict = {}

    async def _apply(job_id, config_json, **kwargs):
        written["cfg"] = json.loads(config_json)
        existing.config = config_json

    q.update_service_config = AsyncMock(side_effect=_apply)
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone(commit="b" * 40))
    assert resp.status_code == 201, resp.text
    cfg = written["cfg"]
    for key in ("last_exit_code", "oom_killed", "last_crash_at"):
        assert key not in cfg
    ld = cfg["last_deploy"]
    assert ld["phase"] == "queued"
    assert ld["action"] == "redeploy"
    assert ld["version"] == 2
    assert ld["reason"] is None
    assert resp.json()["last_deploy"] == ld


def test_r1_subdir_recorded_in_source(tmp_path, monkeypatch):
    q = _queries()
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone(), subdir="services/api")
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["source"]["subdir"] == "services/api"


# --- edge-case #10: DNS-label name validated PRE-CLONE -----------------------


def test_bad_name_422_before_clone(tmp_path, monkeypatch):
    """A bad service name is a pydantic validation_error and never burns a clone."""
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, name="Bad_Name")
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "validation_error"
    clone.assert_not_awaited()


def test_long_name_422_before_clone(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, name="a" * 64)
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "validation_error"
    clone.assert_not_awaited()


def test_valid_name_reaches_clone(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, name="good-name")
    assert resp.status_code == 201, resp.text
    clone.assert_awaited_once()


# --- edge-case #1: kind=model name rejected as a redeploy target -------------


def test_git_deploy_rejects_model_name_kind_mismatch(tmp_path, monkeypatch):
    model_row = Job(
        id="mdl-1",
        service_name="demo",
        name="demo",
        kind=JobKind.model,
        gpu_count=0,
        status=JobStatus.running,
        config="{}",
    )
    q = _queries(model_row)
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone)
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "deploy.kind_mismatch"
    clone.assert_not_awaited()


# --- R2: redeploy version bump + stamp replacement ---------------------------


def test_r2_redeploy_bumps_version_and_replaces_stamp(tmp_path, monkeypatch):
    existing = _existing({"source": {"type": "zip"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone(commit="b" * 40))
    assert resp.status_code == 201, resp.text
    q.reserve_service_for_token.assert_not_called()
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["image"] == "nerdit-app/demo:2"
    assert cfg["build_version"] == 2
    assert cfg["previous_image"] == "nerdit-app/demo:1"
    # Source stamp replaced wholesale (git over the prior zip stamp).
    assert cfg["source"]["type"] == "git"
    assert cfg["source"]["commit_sha"] == "b" * 40


# --- R3: token_ref resolution ------------------------------------------------


def _shared_referenced(q) -> list:
    return [
        c.kwargs
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "secret.shared_referenced"
    ]


def test_r3a_shared_token_ref_resolves_and_stays_secret(tmp_path, monkeypatch):
    q = _queries()
    mgr = _secret_manager({"demo": {}, "_shared": {"GITHUB_TOKEN": "tok-value"}})
    clone = _fake_clone()
    client = _client(q, tmp_path, with_audit=True, secret_manager=mgr)
    resp = _post(client, monkeypatch, clone, token_ref="${secrets.shared.GITHUB_TOKEN}")
    assert resp.status_code == 201, resp.text
    # clone_source received the resolved raw token via its kwarg.
    assert clone.await_args.kwargs["token"] == "tok-value"
    # A shared-referenced audit row was recorded for the requester.
    calls = _shared_referenced(q)
    assert len(calls) == 1
    assert json.loads(calls[0]["params_redacted"]) == {"service": "demo", "keys": ["GITHUB_TOKEN"]}
    # The raw value never leaks: not in the persisted row, audit inserts, or body.
    row_cfg = json.dumps(json.loads(q.reserve_service_for_token.call_args.args[0].config))
    assert "tok-value" not in row_cfg
    assert "tok-value" not in json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list])
    assert "tok-value" not in resp.text


def test_r3a_per_service_scope_takes_precedence(tmp_path, monkeypatch):
    """A per-service KEY overrides the shared store (P8 escape-hatch precedence).

    The per-service scope is only consulted for an OWNED existing service (a
    redeploy), so this exercises the precedence on a row owned by the poster.
    """
    q = _queries(_existing())  # owned by tok-sub (the SUB_RAW poster)
    mgr = _secret_manager({"demo": {"GITHUB_TOKEN": "svc-tok"}, "_shared": {"GITHUB_TOKEN": "sh"}})
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, secret_manager=mgr),
        monkeypatch,
        clone,
        token_ref="${secrets.shared.GITHUB_TOKEN}",
    )
    assert resp.status_code == 201, resp.text
    assert clone.await_args.kwargs["token"] == "svc-tok"
    # Resolved from the per-service scope, so no shared-referenced row.
    assert _shared_referenced(q) == []


def test_r3a_unscoped_token_ref_resolves_on_owned_redeploy(tmp_path, monkeypatch):
    """An unscoped ${secrets.KEY} resolves the service scope only for an owner."""
    q = _queries(_existing())  # owned by tok-sub → per-service scope is legitimate
    mgr = _secret_manager({"demo": {"GH": "svc-value"}})
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, secret_manager=mgr), monkeypatch, clone, token_ref="${secrets.GH}"
    )
    assert resp.status_code == 201, resp.text
    assert clone.await_args.kwargs["token"] == "svc-value"
    assert _shared_referenced(q) == []


def test_r3_fresh_deploy_unscoped_token_ref_forbidden(tmp_path, monkeypatch):
    """A per-service token_ref on a FRESH deploy is rejected before the clone:
    the row was never owner-checked, and secret files outlive a deleted service."""
    q = _queries()  # fresh — no existing row
    mgr = _secret_manager({"demo": {"GH": "svc-value"}})
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, secret_manager=mgr), monkeypatch, clone, token_ref="${secrets.GH}"
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "deploy.git_token_scope"
    clone.assert_not_awaited()


def test_r3_fresh_deploy_ignores_leftover_service_scope(tmp_path, monkeypatch):
    """A fresh deploy reusing a deleted service's name must NOT resolve the
    previous owner's leftover per-service secret; a shared ref reads the shared
    store ALONE (the credential-reuse escalation Codex flagged on PR #65)."""
    q = _queries()  # fresh — name reused, no row
    mgr = _secret_manager(
        {"demo": {"GITHUB_TOKEN": "leftover-prev-owner"}, "_shared": {"GITHUB_TOKEN": "sh-value"}}
    )
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, secret_manager=mgr),
        monkeypatch,
        clone,
        token_ref="${secrets.shared.GITHUB_TOKEN}",
    )
    assert resp.status_code == 201, resp.text
    # Resolved from the shared store, NEVER the leftover per-service scope.
    assert clone.await_args.kwargs["token"] == "sh-value"


def test_r3_token_ref_redacted_in_audit(tmp_path, monkeypatch):
    """token_ref is masked in the deploy.git_create audit row: neither the raw
    value nor the reference string (which reveals the secret scope/key) leaks."""
    q = _queries(_existing())  # owned redeploy → unscoped ref allowed, no shared row
    mgr = _secret_manager({"demo": {"GH": "svc-value"}})
    clone = _fake_clone()
    client = _client(q, tmp_path, with_audit=True, secret_manager=mgr)
    resp = _post(client, monkeypatch, clone, token_ref="${secrets.GH}")
    assert resp.status_code == 201, resp.text
    audit = [c.kwargs for c in q.insert_audit_log.await_args_list]
    dumped = json.dumps(audit)
    assert "svc-value" not in dumped  # the resolved value never
    assert "${secrets.GH}" not in dumped  # nor the reference string
    create = [c for c in audit if c.get("action") == "deploy.git_create"]
    assert create, "expected a deploy.git_create audit row"
    assert json.loads(create[0]["params_redacted"])["token_ref"] == "***"


def test_r3b_bad_token_ref_422_before_clone(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, token_ref="${secrets.lower}")
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.git_token_ref_invalid"
    clone.assert_not_awaited()


def test_r3c_missing_token_ref_key_422(tmp_path, monkeypatch):
    q = _queries()
    mgr = _secret_manager({"demo": {}, "_shared": {}})
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, secret_manager=mgr),
        monkeypatch,
        clone,
        token_ref="${secrets.shared.MISSING}",
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.git_token_unresolved"
    clone.assert_not_awaited()


# --- R4: GitSourceError → structured envelope --------------------------------


def test_r4_git_unavailable_maps_to_envelope(tmp_path, monkeypatch):
    q = _queries()
    clone = AsyncMock(side_effect=GitSourceError(503, "deploy.git_unavailable", "no git binary"))
    resp = _post(_client(q, tmp_path), monkeypatch, clone)
    assert resp.status_code == 503
    assert resp.json()["code"] == "deploy.git_unavailable"


def test_r4_host_forbidden_before_clone(tmp_path, monkeypatch):
    """A non-allowed host is rejected by the REAL validator, clone never called."""
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, repo_url="https://evil.com/o/r")
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.git_host_forbidden"
    clone.assert_not_awaited()


# --- R5: pre-clone authz ordering + guard hygiene ----------------------------


def test_r5_redeploy_non_owner_forbidden_before_clone(tmp_path, monkeypatch):
    existing = _existing(owner="someone-else")
    q = _queries(existing)
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone)  # SUB_RAW = tok-sub
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    clone.assert_not_called()
    q.update_service_config.assert_not_called()


def test_r5_userinfo_url_422_leaves_no_audit_trace(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(
        _client(q, tmp_path, with_audit=True),
        monkeypatch,
        clone,
        repo_url="https://user:tok@github.com/o/r",
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.git_url_credentials"
    clone.assert_not_awaited()
    # The guard runs before audit_params is set: no repo_url in any audit row.
    logged = json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list])
    assert "repo_url" not in logged
    assert "github.com/o/r" not in logged


def test_r5_git_disabled_forbidden(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path, git_enabled=False), monkeypatch, clone)
    assert resp.status_code == 403
    assert resp.json()["code"] == "deploy.git_disabled"
    clone.assert_not_awaited()


def test_r5_reserved_name_422(tmp_path, monkeypatch):
    q = _queries()
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone(), name="shared")
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"


def test_r5_readonly_blocked(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, raw=RO_RAW)
    assert resp.status_code == 403
    clone.assert_not_awaited()


# --- R6: build_context_root stamping + clone-root cleanup (F3) ---------------


def test_r6_subdir_stamps_build_context_root(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone, subdir="services/api")
    assert resp.status_code == 201, resp.text
    dest = Path(clone.await_args.kwargs["dest_dir"])
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    # The clone ROOT is stamped so the controller cleans up the whole tree,
    # while the build context stays the nested subdir docker builds from.
    assert cfg["build_context_root"] == str(dest)
    assert cfg["build_context_dir"] == str(dest / "services/api")


def test_r6_subdir_finalize_failure_removes_clone_root(tmp_path, monkeypatch):
    q = _queries()

    async def _run(repo_url, *, dest_dir, ref=None, subdir=None, **_kw) -> GitSourceInfo:
        dest = Path(dest_dir)
        ctx = dest / subdir if subdir else dest
        ctx.mkdir(parents=True, exist_ok=True)  # clone root exists before finalize
        # No buildpack files → detect() raises → _finalize_deploy 400s.
        (ctx / "README.md").write_text("nothing to build here")
        return GitSourceInfo(commit_sha="a" * 40, resolved_ref=ref or "main", context_dir=ctx)

    clone = AsyncMock(side_effect=_run)
    resp = _post(_client(q, tmp_path), monkeypatch, clone, subdir="services/api")
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "deploy.no_buildpack"
    dest = Path(clone.await_args.kwargs["dest_dir"])
    assert not dest.exists()  # whole clone root removed, not just the subdir


# --- WP8: generated .dockerignore at the git ingress -------------------------


def test_git_deploy_generates_dockerignore(tmp_path, monkeypatch):
    """P13 WP8: a git buildpack deploy gets a generated .dockerignore in the
    build context — the guard the plan singles out (git ships node_modules/.venv
    into images today), pinned at the git route so it never regresses out of the
    shared ``_finalize_deploy``."""
    q = _queries()
    clone = _fake_clone()
    resp = _post(_client(q, tmp_path), monkeypatch, clone)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    ignore = Path(cfg["build_context_dir"]) / ".dockerignore"
    assert ignore.is_file()
    body = ignore.read_text()
    for pat in ("node_modules", ".git", "__pycache__", ".venv"):
        assert pat in body


def test_git_deploy_preserves_repo_dockerignore(tmp_path, monkeypatch):
    """A .dockerignore committed in the repo is preserved, never clobbered."""
    q = _queries()

    async def _run(repo_url, *, dest_dir, ref=None, subdir=None, **_kw) -> GitSourceInfo:
        ctx = Path(dest_dir)
        ctx.mkdir(parents=True, exist_ok=True)
        (ctx / "package.json").write_text(
            json.dumps({"name": "demo", "scripts": {"start": "node index.js"}})
        )
        (ctx / "index.js").write_text("console.log('hi')")
        (ctx / ".dockerignore").write_text("# repo file\nsecret.txt\n")
        return GitSourceInfo(commit_sha="a" * 40, resolved_ref=ref or "main", context_dir=ctx)

    clone = AsyncMock(side_effect=_run)
    resp = _post(_client(q, tmp_path), monkeypatch, clone)
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    ignore = Path(cfg["build_context_dir"]) / ".dockerignore"
    assert ignore.read_text() == "# repo file\nsecret.txt\n"


# --- env null-delete + dry_run (P13 WP9) -------------------------------------


def test_git_redeploy_env_null_delete(tmp_path, monkeypatch):
    """A git redeploy body with a null env value deletes that key (P13 WP9)."""
    existing = _existing({"env": {"KEEP": "1", "DROP": "2"}})
    q = _queries(existing)
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone(), env={"DROP": None, "NEW": "3"})
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config.call_args.args[1])
    # P14 WP-A1: the implicit /data wedge injects NERDIT_DATA_DIR via setdefault.
    assert cfg["env"] == {"KEEP": "1", "NEW": "3", "NERDIT_DATA_DIR": "/data"}


def test_git_dry_run_writes_nothing_and_cleans_clone(tmp_path, monkeypatch):
    """A git dry-run returns the plan (200), writes nothing, and removes the clone."""
    q = _queries()
    clone = _fake_clone()
    monkeypatch.setattr(deploy_mod, "clone_source", clone)
    resp = _client(q, tmp_path).post(
        "/deploy/git",
        params={"dry_run": "true"},
        json={"repo_url": REPO, "name": "demo"},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dry_run"] is True
    assert body["action"] == "create"
    assert body["buildpack"] == "node"
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config.assert_not_called()
    clone.assert_awaited_once()
    # The clone ROOT (dest_dir under upload_dir) is removed on the dry-run path.
    upload_dir = Path(tmp_path) / "uploads"
    leftover = [p for p in upload_dir.iterdir()] if upload_dir.exists() else []
    assert leftover == []


def test_git_dry_run_audited_as_deploy_git_plan(tmp_path, monkeypatch):
    q = _queries()
    monkeypatch.setattr(deploy_mod, "clone_source", _fake_clone())
    resp = _client(q, tmp_path, with_audit=True).post(
        "/deploy/git",
        params={"dry_run": "true"},
        json={"repo_url": REPO, "name": "demo", "env": {"SECRET_ENV": "super-secret"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 200, resp.text
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "deploy.git_plan"
    assert "super-secret" not in json.dumps(kwargs)  # env value never audited
    q.reserve_service_for_token.assert_not_called()


def test_git_dry_run_token_ref_redacted_under_deploy_git_plan(tmp_path, monkeypatch):
    """The 1.10 redaction pin under the OVERRIDDEN action name: a dry-run with a
    token_ref audits as ``deploy.git_plan`` with the reference masked exactly as
    under ``deploy.git_create`` — neither the resolved raw token nor the
    reference string (which reveals the secret scope/key) ever lands."""
    q = _queries()  # fresh deploy → shared-scope ref (unscoped is 403 pre-clone)
    mgr = _secret_manager({"demo": {}, "_shared": {"GH_TOKEN": "raw-token-value"}})
    monkeypatch.setattr(deploy_mod, "clone_source", _fake_clone())
    resp = _client(q, tmp_path, with_audit=True, secret_manager=mgr).post(
        "/deploy/git",
        params={"dry_run": "true"},
        json={
            "repo_url": REPO,
            "name": "demo",
            "token_ref": "${secrets.shared.GH_TOKEN}",
            "env": {"SECRET_ENV": "super-secret"},
        },
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 200, resp.text
    audit = [c.kwargs for c in q.insert_audit_log.await_args_list]
    dumped = json.dumps(audit)
    assert "raw-token-value" not in dumped  # the resolved value never
    assert "${secrets.shared.GH_TOKEN}" not in dumped  # nor the reference string
    assert "super-secret" not in dumped  # env values masked too
    plan = [c for c in audit if c.get("action") == "deploy.git_plan"]
    assert plan, "expected a deploy.git_plan audit row"
    assert json.loads(plan[0]["params_redacted"])["token_ref"] == "***"
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config.assert_not_called()


# --- audit action mapping ----------------------------------------------------


def test_derive_action_deploy_git():
    action, target_type, target_id = derive_action("POST", "/deploy/git")
    assert action == "deploy.git_create"
    assert target_type == "service"
    assert target_id is None
    # /api mount collapses to the same action.
    assert derive_action("POST", "/api/deploy/git")[0] == "deploy.git_create"


# --- PR1: audit target attribution -------------------------------------------


def test_git_deploy_attributes_audit_target_to_service_name(tmp_path, monkeypatch):
    """derive_action leaves deploy.git_create's target null; the route stamps the
    validated body.name so the git event is attributable to the service."""
    q = _queries()
    resp = _post(_client(q, tmp_path, with_audit=True), monkeypatch, _fake_clone())
    assert resp.status_code == 201, resp.text
    create = [
        c.kwargs
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "deploy.git_create"
    ]
    assert create, "expected a deploy.git_create audit row"
    assert create[0]["target_type"] == "service"
    assert create[0]["target_id"] == "demo"


def test_git_deploy_response_carries_summary_and_hints(tmp_path, monkeypatch):
    """(Agent-DX) The git ingress reaches the SHARED ``_finalize_deploy`` tail,
    so the DX channel arrives with no per-ingress code — and the route declares
    no ``response_model``, so the additive keys survive to the wire."""
    q = _queries()
    resp = _post(_client(q, tmp_path), monkeypatch, _fake_clone())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body["summary"]) == {"app", "status", "version", "public_url"}
    assert body["summary"]["app"] == body["name"] == "demo"
    assert body["summary"]["status"] == body["status"]
    assert isinstance(body["hints"], list)


# --- P33 WP-R: token_ref = ${github.installation} (D-GH-3 / D-GH-9) ----------


def _link_manager(
    tokens: dict[str, str],
    *,
    link_state: str = "connected",
    pro_mirror: str = "never",
) -> MagicMock:
    """A stand-in for the two reads the P33/P34 GitHub paths make of the link.

    ``link_state`` and ``pro_mirror`` are spelled out rather than left as bare
    ``MagicMock`` attributes because P34 D3 branches on both: a mock whose
    ``status().state`` merely happens not to equal ``"connected"`` would give
    the generic hint by accident, and an accident is not a pin. The defaults are
    the ordinary P33 node — online, with the cloud never having asserted
    anything about the plan — which is the generic-hint case.
    """
    mgr = MagicMock()
    mgr.github_token_for_repo = MagicMock(side_effect=lambda slug: tokens.get(slug))
    # The auth middleware also consults a wired manager for relay capabilities:
    # an explicit "no" keeps the test bearers resolving as ordinary tokens.
    mgr.validate_capability = MagicMock(return_value=False)
    mgr.status = MagicMock(return_value=SimpleNamespace(state=link_state))
    mgr.pro_mirror_state = MagicMock(return_value=pro_mirror)
    return mgr


def _gh_client(
    q,
    tmp_path,
    tokens: dict[str, str] | None,
    mgr=None,
    *,
    link_state: str = "connected",
    pro_mirror: str = "never",
    **kw,
) -> TestClient:
    app = _make_app(q, tmp_path, secret_manager=mgr, **kw)
    if tokens is not None:
        app.state.link_manager = _link_manager(tokens, link_state=link_state, pro_mirror=pro_mirror)
    return TestClient(app, raise_server_exceptions=False)


def test_github_ref_resolves_by_repo_on_a_fresh_deploy(tmp_path, monkeypatch):
    """A fresh deploy may use the literal: no per-service scope is read, so the
    ``deploy.git_token_scope`` carve-out does not apply; the lookup is by the
    canonical owner/name of the request URL."""
    sentinel = "ghs-zqxjkw-ZQXJKW"
    q = _queries()
    mgr = _secret_manager({"demo": {}, "_shared": {}})
    clone = _fake_clone()
    client = _gh_client(q, tmp_path, {"owner/repo": sentinel}, mgr, with_audit=True)
    resp = _post(
        client,
        monkeypatch,
        clone,
        repo_url="https://GitHub.com/Owner/Repo.git",
        token_ref="${github.installation}",
    )
    assert resp.status_code == 201, resp.text
    assert clone.await_args.kwargs["token"] == sentinel
    client.app.state.link_manager.github_token_for_repo.assert_called_once_with("owner/repo")
    mgr.load.assert_not_called()
    assert _shared_referenced(q) == []
    row_cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert row_cfg["source"]["token_ref"] == "${github.installation}"
    assert sentinel not in json.dumps(row_cfg)
    assert sentinel not in json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list])
    assert sentinel not in resp.text


def test_github_ref_without_a_link_manager_is_422_with_the_hint(tmp_path, monkeypatch):
    """D-GH-9: the explicit user action is the one loud surface."""
    q = _queries()
    clone = _fake_clone()
    resp = _post(
        _gh_client(q, tmp_path, None), monkeypatch, clone, token_ref="${github.installation}"
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "deploy.github_token_absent"
    assert body["hint"] == (
        "link this node and install the Nerdit GitHub App, or pass a `${secrets.*}` token_ref"
    )
    clone.assert_not_awaited()


def test_github_ref_for_a_repo_outside_every_installation_is_422(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    resp = _post(
        _gh_client(q, tmp_path, {"someone/else": "ghs-x"}),
        monkeypatch,
        clone,
        token_ref="${github.installation}",
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.github_token_absent"
    clone.assert_not_awaited()


def test_github_ref_is_never_offered_to_a_non_github_host(tmp_path, monkeypatch):
    q = _queries()
    clone = _fake_clone()
    client = _gh_client(q, tmp_path, {"owner/repo": "ghs-x"}, git_enabled=True)
    client.app.state.settings.git.allowed_hosts = ["github.com", "gitlab.com"]
    resp = _post(
        client,
        monkeypatch,
        clone,
        repo_url="https://gitlab.com/owner/repo",
        token_ref="${github.installation}",
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "deploy.github_token_absent"
    client.app.state.link_manager.github_token_for_repo.assert_not_called()
    clone.assert_not_awaited()


# --- P34 D3: the hint (and ONLY the hint) is tier-aware (D-X16-37) ----------

#: The wire contract, pinned byte-for-byte here rather than imported from the
#: source: an import would follow the source wherever it moved and prove
#: nothing. Clients, the CLI renderer and the poller's ``(422, code)`` match all
#: key off these two, so they must be identical in every mirror state.
_ABSENT_DETAIL = (
    "token_ref '${github.installation}' resolves to no installation token for this repository."
)
_GENERIC_HINT = (
    "link this node and install the Nerdit GitHub App, or pass a `${secrets.*}` token_ref"
)
_TIER_HINT = "this account's plan does not include GitHub deploys — see the Nerdit console"


def _absent_envelope(q, tmp_path, monkeypatch, *, link_state: str, pro_mirror: str) -> dict:
    """Provoke the 422 on a node whose link reads ``link_state``/``pro_mirror``.

    Asserts the invariant half of the envelope — status, ``code``, ``detail``,
    and that nothing was cloned — and hands back the body so each case has one
    thing left to say: which hint it got.
    """
    clone = _fake_clone()
    client = _gh_client(q, tmp_path, {}, link_state=link_state, pro_mirror=pro_mirror)
    resp = _post(client, monkeypatch, clone, token_ref="${github.installation}")

    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["code"] == "deploy.github_token_absent"
    assert body["detail"] == _ABSENT_DETAIL
    assert body["message"] == _ABSENT_DETAIL
    clone.assert_not_awaited()
    return body


def test_github_absent_hint_fresh_negative_tier(tmp_path, monkeypatch):
    """The one state that earns the tier hint: online link, recent cloud "no".

    Nothing is refused that was not already refused — the status and code are
    the P33 ones — and the hint carries no URL, slug or account identifier
    (D-X16-O11).
    """
    body = _absent_envelope(
        _queries(), tmp_path, monkeypatch, link_state="connected", pro_mirror="fresh_false"
    )

    assert body["hint"] == _TIER_HINT
    assert "http" not in body["hint"]
    assert "upgrade" not in body["hint"]


def test_github_absent_hint_fresh_positive_generic(tmp_path, monkeypatch):
    """A Pro account that has not installed the App is the GENERIC case.

    Telling a subscriber their plan lacks GitHub deploys when the cloud has just
    said the opposite would be the worst reading of all, so ``fresh_true``
    falls through to the wording that actually describes what is missing.
    """
    body = _absent_envelope(
        _queries(), tmp_path, monkeypatch, link_state="connected", pro_mirror="fresh_true"
    )

    assert body["hint"] == _GENERIC_HINT


def test_github_absent_hint_stale_generic(tmp_path, monkeypatch):
    """A mirror older than the 24 h TTL is evidence of nothing (D-X16-37).

    This is the case the asymmetry exists for: a paying customer whose node has
    been out of touch with the cloud for a day must not be told their plan is
    too small on the strength of a value nobody has re-asserted.
    """
    body = _absent_envelope(
        _queries(), tmp_path, monkeypatch, link_state="connected", pro_mirror="stale"
    )

    assert body["hint"] == _GENERIC_HINT


def test_github_absent_hint_never_asserted_generic(tmp_path, monkeypatch):
    """The cloud has never spoken about this account — silence is not a "no".

    The state a freshly-linked node sits in for its first seconds, and the one
    a node whose tunnel has just been cleared falls back to.
    """
    body = _absent_envelope(
        _queries(), tmp_path, monkeypatch, link_state="connected", pro_mirror="never"
    )

    assert body["hint"] == _GENERIC_HINT


def test_github_absent_hint_link_down_generic(tmp_path, monkeypatch):
    """Every not-``connected`` state is generic, even holding a fresh negative.

    The independent connection check is defence in depth over the mirror clear
    on the offline edge; ``backoff`` is the state that makes it observable,
    since a reconnecting node is exactly the one whose mirror is least worth
    quoting back at its owner.
    """
    for state in ("connecting", "backoff", "displaced", "terminal"):
        body = _absent_envelope(
            _queries(), tmp_path, monkeypatch, link_state=state, pro_mirror="fresh_false"
        )
        assert body["hint"] == _GENERIC_HINT, state


def test_github_absent_hint_without_a_link_manager_never_consults_a_tier(tmp_path, monkeypatch):
    """``[link].enabled=false`` reaches the generic hint without asking anything.

    D-X16-O15/D-ENT-2: an unlinked daemon is not a degraded daemon, and D3 adds
    no entitlement read to a path that had none — the tier decision short-
    circuits on ``manager is None`` before any status or mirror call exists.
    """
    clone = _fake_clone()
    resp = _post(
        _gh_client(_queries(), tmp_path, None),
        monkeypatch,
        clone,
        token_ref="${github.installation}",
    )

    assert resp.status_code == 422
    assert resp.json()["hint"] == _GENERIC_HINT


def test_the_tier_decision_reads_the_mirror_only_after_the_link_state(tmp_path):
    """The two reads are ordered and short-circuited, not just ANDed.

    ``status()`` is the cheap field read and the one that fails safe; a node
    whose link is down must reach the generic hint without the mirror being
    consulted at all, so a future mirror read that acquired a lock or did I/O
    could never be reached from an offline node.
    """
    from nerdit.daemon.deploy_pipeline import github_absent_is_tier_gated

    mgr = _link_manager({}, link_state="backoff", pro_mirror="fresh_false")
    app = SimpleNamespace(state=SimpleNamespace(link_manager=mgr))

    assert github_absent_is_tier_gated(app) is False
    mgr.pro_mirror_state.assert_not_called()


def test_github_ref_variants_are_not_the_literal(tmp_path, monkeypatch):
    """Only the exact literal is grammar: ``${github.other}`` is a bad ref."""
    q = _queries()
    clone = _fake_clone()
    for bad in ("${github.other}", "${GITHUB.installation}", "${github.installation} "):
        resp = _post(
            _gh_client(q, tmp_path, {"owner/repo": "ghs-x"}), monkeypatch, clone, token_ref=bad
        )
        assert resp.status_code == 422, bad
        assert resp.json()["code"] == "deploy.git_token_ref_invalid", bad
    clone.assert_not_awaited()


def test_secret_refs_are_untouched_by_a_wired_link_manager(tmp_path, monkeypatch):
    """The no-cloud path is byte-identical for ``${secrets.*}``: a wired link
    manager is never consulted and the secret-store resolution is the same."""
    q = _queries()
    mgr = _secret_manager({"demo": {}, "_shared": {"GITHUB_TOKEN": "tok-value"}})
    clone = _fake_clone()
    client = _gh_client(q, tmp_path, {"owner/repo": "ghs-must-not-be-used"}, mgr)
    resp = _post(client, monkeypatch, clone, token_ref="${secrets.shared.GITHUB_TOKEN}")
    assert resp.status_code == 201, resp.text
    assert clone.await_args.kwargs["token"] == "tok-value"
    client.app.state.link_manager.github_token_for_repo.assert_not_called()
