"""Tests for the P7 per-app config-as-API routes (plan §2.2 / §2.6 item 3).

Uses the ``test_deploy_route`` harness: the real router under the auth/audit
middleware with ``AsyncMock`` queries (no lifespan, no aiosqlite — the sandbox
caveat), plus the deploy-route stamping tests for ``config_source`` /
``config_revision`` / ``overwrote_api_config``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.app_config import router as app_config_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"
OTHER_RAW = "other-raw"
SCOPED_RAW = "scoped-raw"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
    hash_token(OTHER_RAW): ApiToken(
        id="tok-other", name="o", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
    # A submitter narrowed to the app it owns — the H2 scope leg.
    hash_token(SCOPED_RAW): ApiToken(
        id="tok-scoped",
        name="c",
        role=TokenRole.submitter,
        token_hash=hash_token(SCOPED_RAW),
        scope_services=["demo"],
    ),
}


def _app_job(**overrides) -> Job:
    config = {
        "image": "nerdit-app/demo:2",
        "build_version": 2,
        "port": 8000,
        "command": "npm start",
        "env": {"FOO": "bar", "BAZ": "qux"},
        "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
        "config_source": "deploy",
        "config_revision": 3,
    }
    config.update(overrides.pop("config_extra", {}))
    fields = dict(
        id="job-demo-0001",
        kind=JobKind.service,
        service_name="demo",
        name="demo",
        gpu_count=1,
        status=JobStatus.running,
        desired_state="running",
        health_check={"path": "/health"},
        config=json.dumps(config),
        submitted_by_token="tok-sub",
    )
    fields.update(overrides)
    return Job(**fields)


def _model_row() -> Job:
    return Job(
        kind=JobKind.model,
        service_name="ollama-llama3-1-8b",
        status=JobStatus.running,
        gpu_count=1,
        config=json.dumps({"model": "llama3.1:8b"}),
    )


def _database_row() -> Job:
    return Job(
        kind=JobKind.database,
        service_name="pg-demo",
        status=JobStatus.running,
        gpu_count=0,
        config=json.dumps({"backend": "postgres"}),
    )


def _queries(job: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_by_name = AsyncMock(return_value=job)
    # (P20) The write path re-reads the row to graft the build-tier keys off a
    # FRESH blob rather than the handler's snapshot; a mock that answers this
    # with anything other than the same row models a lost update, not a steady
    # state. Per-test overrides drive the concurrent-writer cases.
    q.get_job = AsyncMock(return_value=job)
    q.get_model_by_ref = AsyncMock(return_value=_model_row())
    q.update_app_config = AsyncMock()
    q.set_desired_state = AsyncMock()
    q.update_job_status = AsyncMock()
    q.bump_restart_count = AsyncMock()
    return q


def _client(queries: AsyncMock, *, with_audit: bool = False) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(app_config_router)
    app.state.queries = queries
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _put(client, section="deploy", body=None, raw=SUB_RAW, headers=None, **params):
    hdrs = {**_auth(raw), "Idempotency-Key": "ik-1", **(headers or {})}
    return client.put(f"/config/apps/demo/{section}", json=body or {}, headers=hdrs, params=params)


# --- GET ----------------------------------------------------------------------


def test_get_view_projection():
    client = _client(_queries(_app_job()))
    r = client.get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    view = r.json()
    assert view["service_name"] == "demo"
    assert view["deploy"] == {
        "name": "demo",
        "port": 8000,
        "gpus": 1,
        "start": "npm start",
        "health": "/health",
        "memory_limit": None,
        "cpu_limit": None,
        "volumes": [],  # P14 WP-A1: none declared on this fixture
        "release": None,  # P20: none declared on this fixture
        # P24b: tri-state — None is "not declared" (daemon default posture),
        # distinct from an explicit False opt-out.
        "cutover": None,
        "auto_deploy": None,
        # P25: the edge-auth declaration — none on this fixture.
        "edge_auth": None,
    }
    assert view["ai"]["default"]["model"] == "llama3.1:8b"
    assert view["env_keys"] == ["BAZ", "FOO"]  # names only, sorted, no values
    assert view["source"] == "deploy"
    assert view["revision"] == 3
    assert view["etag"] and r.headers["ETag"] == view["etag"]


def test_get_unknown_app_404():
    client = _client(_queries(None))
    r = client.get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 404
    assert r.json()["code"] == "not_found"


def test_get_model_row_is_not_an_app():
    # Finding #13: a kind=model row gets a distinct config.not_an_app 404 (not
    # the generic "No deployed app" not_found) so an agent stops retrying deploy.
    client = _client(_queries(_model_row()))
    r = client.get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "config.not_an_app"
    assert "model" in body["message"]


def test_put_model_row_is_not_an_app():
    # Both GET and PUT flow through _get_app: the model 404 fires before authz.
    client = _client(_queries(_model_row()))
    r = _put(client, section="deploy", body={"gpus": 0})
    assert r.status_code == 404
    assert r.json()["code"] == "config.not_an_app"


def test_get_database_row_is_not_an_app():
    # Plan §1.4: a kind=database row gets the same distinct config.not_an_app
    # 404 (not the generic not_found), pointing at /databases + /services.
    client = _client(_queries(_database_row()))
    r = client.get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 404
    body = r.json()
    assert body["code"] == "config.not_an_app"
    assert "database" in body["message"]


def test_put_database_row_is_not_an_app():
    # Both GET and PUT flow through _get_app: the database 404 fires before authz.
    client = _client(_queries(_database_row()))
    r = _put(client, section="deploy", body={"gpus": 0})
    assert r.status_code == 404
    assert r.json()["code"] == "config.not_an_app"


# --- PUT: sections + immutability ----------------------------------------------


def test_unknown_section_422():
    client = _client(_queries(_app_job()))
    for section in ("run", "data", "scheduling", "env", "nope"):
        r = _put(client, section=section, body={"x": 1})
        assert r.status_code == 422
        assert r.json()["code"] == "config.unknown_section"
        assert "deploy, ai" in r.json()["hint"]


@pytest.mark.parametrize("key,value", [("name", "other"), ("port", 9000)])
def test_immutable_keys_422(key, value):
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={key: value})
    assert r.status_code == 422
    assert r.json()["code"] == "config.immutable_key"
    q.update_app_config.assert_not_awaited()


def test_unknown_deploy_key_422():
    client = _client(_queries(_app_job()))
    r = _put(client, body={"image": "evil:latest"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "config.invalid"
    assert body["diagnostics"][0]["loc"] == ["deploy", "image"]


def test_unknown_deploy_key_hint_lists_all_writable_keys():
    """Edge-case #18: the hint is generated from _DEPLOY_MUTABLE_KEYS (no drift)."""
    client = _client(_queries(_app_job()))
    r = _put(client, body={"bogus": 1})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "config.invalid"
    hint = body["hint"]
    assert "memory_limit" in hint
    assert "volumes" in hint
    assert "cpu_limit" in hint
    # diagnostics list shape unchanged.
    assert body["diagnostics"][0]["loc"] == ["deploy", "bogus"]


# --- PUT deploy: apply + revision bump ------------------------------------------


def test_put_deploy_gpus_applies_and_bumps_revision():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is True
    # GPU allocation only changes on launch/restart paths (Codex review, PR #52).
    assert body["requires_restart"] is True
    assert body["restarted"] is False
    q.update_app_config.assert_awaited_once()
    args, kwargs = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert new_cfg["config_source"] == "api"
    assert new_cfg["config_revision"] == 4
    assert kwargs["gpu_count"] == 2
    assert kwargs["token_id"] == "tok-sub"  # quota charged to the OWNER


def test_put_deploy_volumes_applies_and_requires_restart():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"volumes": ["data:/data", "cache:/var/cache"]})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is True
    assert body["requires_restart"] is True  # volumes mount at launch
    q.update_app_config.assert_awaited_once()
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["volumes"] == ["data:/data", "cache:/var/cache"]


def test_put_deploy_volumes_invalid_422():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"volumes": ["Bad Name:/x"]})
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid"
    q.update_app_config.assert_not_awaited()


def test_put_deploy_volumes_null_deletes():
    q = _queries(_app_job(config_extra={"volumes": ["data:/data"]}))
    client = _client(q)
    r = _put(client, body={"volumes": None})
    assert r.status_code == 200
    assert r.json()["applied"] is True
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "volumes" not in new_cfg  # popped ⇒ no named volumes


def test_put_deploy_data_volume_retarget_moves_nerdit_data_dir():
    # Codex P2 (PR #73): retargeting the implicit data volume via the API must
    # move the wedge-persisted NERDIT_DATA_DIR with it, or apps following the
    # documented convention write to an unmounted path after the restart.
    q = _queries(
        _app_job(
            config_extra={
                "volumes": ["data:/data"],
                "env": {"NERDIT_DATA_DIR": "/data", "OTHER": "x"},
            }
        )
    )
    client = _client(q)
    r = _put(client, body={"volumes": ["data:/var/lib/app"]})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["volumes"] == ["data:/var/lib/app"]
    assert new_cfg["env"]["NERDIT_DATA_DIR"] == "/var/lib/app"
    assert new_cfg["env"]["OTHER"] == "x"


def test_put_deploy_data_volume_retarget_preserves_user_override():
    q = _queries(
        _app_job(
            config_extra={
                "volumes": ["data:/data"],
                "env": {"NERDIT_DATA_DIR": "/custom"},  # user's own value
            }
        )
    )
    client = _client(q)
    r = _put(client, body={"volumes": ["data:/var/lib/app"]})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["env"]["NERDIT_DATA_DIR"] == "/custom"  # not ours to move


def test_put_noop_write_is_idempotent():
    """Same-value retry: no persist, no revision bump, same ETag (Codex, PR #52)."""
    q = _queries(_app_job())
    client = _client(q)
    etag = client.get("/config/apps/demo", headers=_auth(SUB_RAW)).headers["ETag"]
    r = _put(client, body={"gpus": 1, "start": "npm start"})  # current values
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is True
    assert body["requires_restart"] is False
    assert body["view"]["revision"] == 3
    assert body["view"]["source"] == "deploy"  # provenance untouched
    assert r.headers["ETag"] == etag
    q.update_app_config.assert_not_awaited()


def test_put_deploy_start_requires_restart_and_null_deletes():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"start": None, "health": None})
    assert r.status_code == 200
    body = r.json()
    assert body["requires_restart"] is True
    args, kwargs = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert "command" not in new_cfg
    assert kwargs["clear_health"] is True
    assert kwargs["gpu_count"] is None  # unchanged -> column untouched


def test_put_deploy_memory_and_cpu_limit_apply_and_require_restart():
    """P13 WP5: both caps are writable, restart-required, persisted top-level."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"memory_limit": "512m", "cpu_limit": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is True
    assert body["requires_restart"] is True
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert new_cfg["memory_limit"] == "512m"
    assert new_cfg["cpu_limit"] == 1.5


def test_put_deploy_limits_null_delete_falls_back_to_defaults():
    """P13 WP5: null pops the top-level key so [containers] defaults apply."""
    q = _queries(_app_job(config_extra={"memory_limit": "512m", "cpu_limit": 2.0}))
    client = _client(q)
    r = _put(client, body={"memory_limit": None, "cpu_limit": None})
    assert r.status_code == 200
    assert r.json()["requires_restart"] is True
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert "memory_limit" not in new_cfg
    assert "cpu_limit" not in new_cfg


def test_put_deploy_invalid_memory_limit_422():
    """P13 WP5: bad format rides the existing deploy.invalid envelope."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"memory_limit": "lots"})
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid"
    q.update_app_config.assert_not_awaited()


def test_put_deploy_invalid_cpu_limit_422():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"cpu_limit": 0})
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid"
    q.update_app_config.assert_not_awaited()


def test_put_deploy_health_change_preserves_tuning_fields():
    """A health path change must not squash /services-set tuning fields."""
    q = _queries(
        _app_job(health_check={"path": "/health", "timeout_s": 9.0, "unhealthy_threshold": 5})
    )
    client = _client(q)
    r = _put(client, body={"health": "/live"})
    assert r.status_code == 200
    _, kwargs = q.update_app_config.await_args
    assert kwargs["health_check"] == {"path": "/live", "timeout_s": 9.0, "unhealthy_threshold": 5}
    assert kwargs["clear_health"] is False


# --- PUT deploy.release (P20) -----------------------------------------------------
#
# ``release`` is the ONE asymmetric deploy key: writable and tracked in the
# `changed` diff, but deliberately absent from ``_RESTART_DEPLOY_KEYS`` — it is a
# build-time gate consumed by the NEXT deploy, so restarting the service would
# not run it.


def test_put_deploy_release_applies_without_requiring_a_restart():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"release": "alembic upgrade head"})
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is True
    assert body["requires_restart"] is False, "release only takes effect at the next deploy"
    assert body["restarted"] is False
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["release"] == "alembic upgrade head"
    # It IS a tracked change, so provenance still moves (otherwise a later
    # redeploy clobber would be invisible).
    assert new_cfg["config_source"] == "api"
    assert new_cfg["config_revision"] == 4


def test_put_deploy_release_is_in_the_tracked_diff_not_the_restart_set():
    """Pins the asymmetry directly at the two frozensets, so a future editor who
    "tidies" them into one set breaks a test rather than silently making every
    release write bounce the service."""
    from nerdit.daemon.routes.app_config import (
        _DEPLOY_MUTABLE_KEYS,
        _DEPLOY_TRACKED_KEYS,
        _RESTART_DEPLOY_KEYS,
    )

    assert "release" in _DEPLOY_MUTABLE_KEYS
    assert "release" in _DEPLOY_TRACKED_KEYS
    assert "release" not in _RESTART_DEPLOY_KEYS
    # ...and the set of keys with that shape is CLOSED: P24b's `cutover` /
    # `auto_deploy` join it for the same reason (next-deploy keys a restart
    # would not apply), P25's `edge_auth` for its own (it is materialized into
    # the Caddy route object, not the container env, so the proxy reconcile
    # applies it on the next tick and a restart would only cause an outage),
    # and nothing else may.
    assert set(_DEPLOY_TRACKED_KEYS) - _RESTART_DEPLOY_KEYS == {
        "release",
        "cutover",
        "auto_deploy",
        "edge_auth",
    }


def test_put_deploy_release_noop_write_does_not_bump_provenance():
    q = _queries(_app_job(config_extra={"release": "alembic upgrade head"}))
    client = _client(q)
    r = _put(client, body={"release": "alembic upgrade head"})
    assert r.status_code == 200
    assert r.json()["requires_restart"] is False
    q.update_app_config.assert_not_awaited()


def test_put_deploy_release_null_deletes_the_command():
    q = _queries(_app_job(config_extra={"release": "alembic upgrade head"}))
    client = _client(q)
    r = _put(client, body={"release": None})
    assert r.status_code == 200
    assert r.json()["applied"] is True
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "release" not in new_cfg


def test_put_deploy_release_dry_run_previews_without_writing():
    """The dry-run preview view is built from the would-be blob, so it is where
    the round-trip through ``_build_view``'s ``release`` projection is visible."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"release": "alembic upgrade head"}, dry_run="true")
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is False
    assert body["requires_restart"] is False
    assert body["view"]["deploy"]["release"] == "alembic upgrade head"
    q.update_app_config.assert_not_awaited()

    # ...and the delete direction round-trips back to None.
    q2 = _queries(_app_job(config_extra={"release": "alembic upgrade head"}))
    r2 = _put(_client(q2), body={"release": None}, dry_run="true")
    assert r2.status_code == 200
    assert r2.json()["view"]["deploy"]["release"] is None
    q2.update_app_config.assert_not_awaited()


def test_put_deploy_release_null_delete_leaves_an_armed_crash_marker_intact():
    """Deleting the command must NOT disarm the crash marker (D-P20-4).

    ``release_pending`` is platform-owned: the builder arms it in the instant
    before the release container starts, so a row carrying it is either
    mid-migration or one whose release died with the daemon. Popping it from a
    config write would abort the builder's restart-mid-release recovery
    (``AppImageBuilder.ensure_built`` layer 3) while the migration is still
    running — and silently, since the app-config ETag does not cover the marker.
    """
    q = _queries(_app_job(config_extra={"release": "alembic upgrade head", "release_pending": 2}))
    client = _client(q)
    r = _put(client, body={"release": None})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "release" not in new_cfg  # the user-owned command IS deleted...
    assert new_cfg["release_pending"] == 2  # ...the platform-owned marker is not
    assert new_cfg["build_version"] == 2  # nor any other build-tier key


def test_put_deploy_release_write_leaves_a_live_marker_alone():
    """Setting a NEW command must not disarm a generation that is mid-release —
    neither branch of the release write touches the platform-owned marker."""
    q = _queries(_app_job(config_extra={"release": "old-cmd", "release_pending": 2}))
    client = _client(q)
    r = _put(client, body={"release": "new-cmd"})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["release"] == "new-cmd"
    assert new_cfg["release_pending"] == 2


def test_put_carries_platform_owned_build_state_through_untouched():
    """Generalises the marker pin: the PUT is an RMW of the WHOLE persisted blob.

    Every key here is written by the build tier alone and read back by the
    reconcile loop to decide whether to rebuild, revert or settle a generation.
    The merge must carry them forward verbatim — rebuilding the blob from a
    whitelist of user-writable keys would drop them silently, since none of them
    is covered by the app-config ETag.
    """
    platform = {
        "image": "nerdit-app/demo:2",
        "build_version": 2,
        "previous_image": "nerdit-app/demo:1",
        "build_context_dir": "/var/lib/nerdit/build/demo-2",
        "dockerfile_name": "Dockerfile.nerdit",
        "last_deploy": {"phase": "healthy"},
        "release_pending": 2,
    }
    q = _queries(_app_job(config_extra=platform))
    r = _put(_client(q), body={"gpus": 2})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert {key: new_cfg.get(key) for key in platform} == platform


def test_put_does_not_clobber_a_marker_the_builder_armed_mid_request():
    """The handler's snapshot is stale by the time it writes the whole blob.

    ``update_app_config`` is an unconditional ``UPDATE jobs SET config = ?``
    with no CAS, and the builder arms ``release_pending`` from an off-tick task
    that takes no lock this route could hold. Writing the snapshot back would
    silently disarm the pre-swap gate for a migration that is running RIGHT
    NOW — and the next daemon restart would then converge the unmigrated
    candidate. The build-tier keys are therefore grafted off a fresh read.
    """
    snapshot = _app_job(config_extra={"build_version": 4})
    armed = _app_job(config_extra={"build_version": 4, "release_pending": 4})
    q = _queries(snapshot)
    q.get_job = AsyncMock(return_value=armed)  # the builder wrote between read and write

    r = _put(_client(q), body={"gpus": 2})

    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["release_pending"] == 4
    assert new_cfg["config_source"] == "api"  # ...and the caller's write still landed


def test_put_honours_a_build_tier_key_deleted_mid_request():
    """The graft is a re-read, not a merge: a marker the builder POPPED between
    the read and the write must stay popped, or a settled generation would be
    re-armed and re-settled on the next restart."""
    snapshot = _app_job(config_extra={"build_version": 4, "release_pending": 4})
    settled = _app_job(config_extra={"build_version": 4})
    q = _queries(snapshot)
    q.get_job = AsyncMock(return_value=settled)

    r = _put(_client(q), body={"gpus": 2})

    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "release_pending" not in new_cfg


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "migrate\nrm -rf /", "migrate\x00x", "x" * 4097],
    ids=["empty", "blank", "newline", "nul", "oversize"],
)
def test_put_deploy_invalid_release_422(bad):
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"release": bad})
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid"
    q.update_app_config.assert_not_awaited()


def test_put_deploy_invalid_release_422_never_echoes_the_value():
    """End-to-end twin of the validator-tier pin: the rejected value carries the
    control characters that make it unsafe to interpolate, and this envelope is
    exactly the surface an agent (and the audit trail) reads back."""
    q = _queries(_app_job())
    client = _client(q)
    payload = "migrate\x1b[2K\x00FORGED-AUDIT-LINE"
    r = _put(client, body={"release": payload})
    assert r.status_code == 422
    raw = r.content.decode()
    assert "FORGED-AUDIT-LINE" not in raw
    assert "\\u0000" not in raw and "\\u001b" not in raw


def test_put_untouched_health_leaves_column_alone():
    """An ai/gpus write that never mentions health must not rewrite the column."""
    q = _queries(_app_job(health_check={"path": "/health", "timeout_s": 9.0}))
    client = _client(q)
    r = _put(client, section="ai", body={"default": {"provider": "ollama", "model": "phi3"}})
    assert r.status_code == 200
    _, kwargs = q.update_app_config.await_args
    assert kwargs["health_check"] is None  # column untouched
    assert kwargs["clear_health"] is False


# --- PUT ai: binding merge + null-delete + validation parity ---------------------


def test_put_ai_binding_merge_and_delete():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(
        client,
        section="ai",
        body={
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            },
            "default": None,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["requires_restart"] is True
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert "default" not in new_cfg["ai"]
    assert new_cfg["ai"]["cheap"]["model"] == "gpt-4o-mini"


def test_put_ai_literal_api_key_422_parity_with_deploy():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(
        client,
        section="ai",
        body={"cheap": {"provider": "api", "model": "m", "base_url": "u", "api_key": "sk-123"}},
    )
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid_ai"
    q.update_app_config.assert_not_awaited()


def test_put_ai_ollama_gated_on_served_model():
    q = _queries(_app_job())
    q.get_model_by_ref = AsyncMock(return_value=None)
    client = _client(q)
    r = _put(client, section="ai", body={"default": {"provider": "ollama", "model": "mistral:7b"}})
    assert r.status_code == 422
    assert r.json()["code"] == "ai.model_not_served"


# --- PUT ai: shared-scope refs (P8) ----------------------------------------------


_SHARED_BINDING = {
    "provider": "api",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "api_key": "${secrets.shared.OPENAI_KEY}",
}


def _shared_referenced_calls(q) -> list:
    return [
        c.kwargs
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "secret.shared_referenced"
    ]


def test_put_ai_shared_ref_round_trips_and_audits_requester():
    """A shared ref persists verbatim and fires secret.shared_referenced (P8)."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, section="ai", body={"cheap": _SHARED_BINDING})
    assert r.status_code == 200
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert new_cfg["ai"]["cheap"]["api_key"] == "${secrets.shared.OPENAI_KEY}"
    calls = _shared_referenced_calls(q)
    assert len(calls) == 1
    row = calls[0]
    assert row["principal_id"] == "tok-sub"  # the requester, not 'system'
    assert row["principal_role"] == "submitter"
    assert row["target_type"] == "secret"
    assert row["target_id"] == "shared"
    assert json.loads(row["params_redacted"]) == {"service": "demo", "keys": ["OPENAI_KEY"]}


def test_put_ai_unscoped_ref_records_no_shared_referenced():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(
        client,
        section="ai",
        body={
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            }
        },
    )
    assert r.status_code == 200
    assert _shared_referenced_calls(q) == []


def test_put_ai_dry_run_records_no_shared_referenced():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, section="ai", body={"cheap": _SHARED_BINDING}, dry_run="true")
    assert r.status_code == 200
    assert r.json()["applied"] is False
    assert _shared_referenced_calls(q) == []


def test_put_ai_noop_shared_rewrite_records_no_shared_referenced():
    # The spec already carries the shared ref; rewriting it unchanged is a
    # no-op write (changed == []) and must not re-fire the attribution row.
    q = _queries(_app_job(config_extra={"ai": {"cheap": dict(_SHARED_BINDING)}}))
    client = _client(q)
    r = _put(client, section="ai", body={"cheap": _SHARED_BINDING})
    assert r.status_code == 200
    q.update_app_config.assert_not_awaited()
    assert _shared_referenced_calls(q) == []


# --- PUT ceremony: If-Match / Idempotency-Key / dry-run / authz ------------------


def test_if_match_stale_409():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2}, headers={"If-Match": "bogus"})
    assert r.status_code == 409
    body = r.json()
    assert body["code"] == "config.stale"
    assert body["current_etag"]
    q.update_app_config.assert_not_awaited()


def test_if_match_current_ok():
    q = _queries(_app_job())
    client = _client(q)
    etag = client.get("/config/apps/demo", headers=_auth(SUB_RAW)).headers["ETag"]
    r = _put(client, body={"gpus": 2}, headers={"If-Match": etag})
    assert r.status_code == 200


def test_if_match_wildcard_writes_if_exists():
    """RFC 7232 ``If-Match: *`` (write-if-exists) applies without config.stale —
    the app row was resolved, so the representation exists."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2}, headers={"If-Match": "*"})
    assert r.status_code == 200
    assert r.json()["applied"] is True
    q.update_app_config.assert_awaited()


def test_idempotency_key_required_on_real_write():
    q = _queries(_app_job())
    client = _client(q)
    r = client.put("/config/apps/demo/deploy", json={"gpus": 2}, headers=_auth(SUB_RAW))
    assert r.status_code == 400
    assert r.json()["code"] == "idempotency_key_required"
    q.update_app_config.assert_not_awaited()


def test_dry_run_no_write_and_preview():
    q = _queries(_app_job())
    client = _client(q)
    r = client.put(
        "/config/apps/demo/deploy",
        json={"gpus": 2},
        headers=_auth(SUB_RAW),  # no Idempotency-Key: dry-run is exempt
        params={"dry_run": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["applied"] is False
    assert body["view"]["deploy"]["gpus"] == 2
    assert body["view"]["source"] == "api"
    assert body["view"]["revision"] == 4
    q.update_app_config.assert_not_awaited()
    q.set_desired_state.assert_not_awaited()


def test_readonly_blocked_on_put():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2}, raw=RO_RAW)
    assert r.status_code == 403
    q.update_app_config.assert_not_awaited()


def test_foreign_submitter_403_admin_ok():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2}, raw=OTHER_RAW)
    assert r.status_code == 403
    r = _put(client, body={"gpus": 2}, raw=LEGACY)  # legacy global token = admin
    assert r.status_code == 200


# --- restart wiring ---------------------------------------------------------------


def test_restart_opt_in_triggers_desired_state_path():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, section="ai", body={"default": None}, restart="true")
    assert r.status_code == 200
    assert r.json()["restarted"] is True
    q.set_desired_state.assert_awaited_once_with("job-demo-0001", "running")
    q.update_job_status.assert_awaited_once_with("job-demo-0001", JobStatus.restarting)
    q.bump_restart_count.assert_awaited_once_with("job-demo-0001", 0, None)


def test_no_restart_by_default():
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, body={"gpus": 2})
    assert r.status_code == 200
    assert r.json()["restarted"] is False
    q.set_desired_state.assert_not_awaited()
    q.update_job_status.assert_not_awaited()


# --- audit ------------------------------------------------------------------------


# --- deploy stamping (step 7): config_source / config_revision / clobber flag ----


def _deploy_client(queries: AsyncMock, tmp_path) -> TestClient:
    from unittest.mock import MagicMock

    from nerdit.daemon.routes.deploy import router as deploy_router

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    app.state.settings = settings
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _node_zip() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "demo", "scripts": {"start": "node i"}}))
        zf.writestr("index.js", "console.log('hi')")
    return buf.getvalue()


def _deploy(client: TestClient):
    return client.post(
        "/deploy",
        data={"name": "demo", "port": "8000", "gpus": "0"},
        files={"archive": ("app.zip", _node_zip(), "application/zip")},
        headers=_auth(SUB_RAW),
    )


def test_fresh_deploy_stamps_source_and_revision(tmp_path):
    q = _queries(None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    client = _deploy_client(q, tmp_path)
    r = _deploy(client)
    assert r.status_code == 201
    assert r.json()["overwrote_api_config"] is False
    job = q.reserve_service_for_token.await_args.args[0]
    cfg = json.loads(job.config)
    assert cfg["config_source"] == "deploy"
    assert cfg["config_revision"] == 1


def _ai_api_zip() -> bytes:
    import io
    import zipfile

    toml = (
        '[ai.default]\nprovider = "api"\nmodel = "gpt-4o-mini"\n'
        'base_url = "https://api.openai.com/v1"\napi_key = "${secrets.OPENAI_KEY}"\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "demo", "scripts": {"start": "node i"}}))
        zf.writestr("index.js", "console.log('hi')")
        zf.writestr("nerdit.toml", toml)
    return buf.getvalue()


def _deploy_zip(client: TestClient, zip_bytes: bytes, **fields):
    data = {"name": "demo"}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/deploy",
        data=data,
        files={"archive": ("app.zip", zip_bytes, "application/zip")},
        headers=_auth(SUB_RAW),
    )


def test_redeploy_over_api_config_no_explicit_change_does_not_flag_clobber(tmp_path):
    """P13 WP9 narrowed rule: a redeploy over API-authored config does NOT flag a
    clobber merely because the API was the prior writer — only an explicit + changed
    field (or a declared [ai] over an API-set spec) counts."""
    # gpu_count matches the form gpus, no health/start declared, no [ai] on either
    # side → nothing the API authored is actually overwritten.
    existing = _app_job(
        gpu_count=1,
        health_check=None,
        config_extra={"config_source": "api", "config_revision": 5},
    )
    existing_cfg = json.loads(existing.config)
    existing_cfg.pop("ai", None)  # no prior [ai] so a silent redeploy is a no-op there
    existing.config = json.dumps(existing_cfg)
    q = _queries(existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.update_service_config_guarded = AsyncMock()
    client = _deploy_client(q, tmp_path)
    r = _deploy_zip(client, _node_zip(), gpus=1)
    assert r.status_code == 201, r.text
    assert r.json()["overwrote_api_config"] is False
    cfg = json.loads(q.update_service_config_guarded.await_args.args[1])
    assert cfg["config_source"] == "deploy"
    assert cfg["config_revision"] == 6


def _node_zip_with_toml(toml: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "demo", "scripts": {"start": "node i"}}))
        zf.writestr("index.js", "console.log('hi')")
        zf.writestr("nerdit.toml", toml)
    return buf.getvalue()


def test_redeploy_buildpack_toml_start_over_api_does_not_flag_clobber(tmp_path):
    """B1: a Node buildpack row bakes its CMD into the Dockerfile and never
    persists ``command``, so a repo [deploy].start on an identical redeploy has no
    API-authored start to clobber — the flag must stay False even though the prior
    writer was the API."""
    # Buildpack row: config_source='api', gpus match, NO "command" key (buildpack).
    existing = _app_job(
        gpu_count=1,
        health_check=None,
        config_extra={"config_source": "api", "config_revision": 5},
    )
    existing_cfg = json.loads(existing.config)
    existing_cfg.pop("ai", None)  # silent redeploy is a no-op on ai
    existing_cfg.pop("command", None)  # buildpack: start baked into the Dockerfile
    existing.config = json.dumps(existing_cfg)
    q = _queries(existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.update_service_config_guarded = AsyncMock()
    client = _deploy_client(q, tmp_path)
    zip_bytes = _node_zip_with_toml('[deploy]\nname = "demo"\nstart = "npm run prod"\n')
    r = _deploy_zip(client, zip_bytes, gpus=1)
    assert r.status_code == 201, r.text
    assert r.json()["overwrote_api_config"] is False


def test_redeploy_changed_start_over_api_command_baseline_flags_clobber(tmp_path):
    """B1 true-positive kept green: when the API DID author a start (persisted as
    ``command``), an explicit + changed start still fires the clobber flag."""
    existing = _app_job(
        gpu_count=1,
        health_check=None,
        config_extra={"config_source": "api", "config_revision": 5, "command": "npm start"},
    )
    existing_cfg = json.loads(existing.config)
    existing_cfg.pop("ai", None)
    existing.config = json.dumps(existing_cfg)
    q = _queries(existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.update_service_config_guarded = AsyncMock()
    client = _deploy_client(q, tmp_path)
    zip_bytes = _node_zip_with_toml('[deploy]\nname = "demo"\nstart = "npm run prod"\n')
    r = _deploy_zip(client, zip_bytes, gpus=1)
    assert r.status_code == 201, r.text
    assert r.json()["overwrote_api_config"] is True


def test_redeploy_explicit_gpu_change_over_api_flags_clobber(tmp_path):
    """An explicit gpus change over config_source='api' fires the narrowed flag."""
    existing = _app_job(gpu_count=1, config_extra={"config_source": "api"})
    q = _queries(existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.update_service_config_guarded = AsyncMock()
    client = _deploy_client(q, tmp_path)
    r = _deploy_zip(client, _node_zip(), gpus=5)
    assert r.status_code == 201, r.text
    assert r.json()["overwrote_api_config"] is True


def test_redeploy_declared_ai_over_api_marker_flags_clobber(tmp_path):
    """A redeploy that declares [ai] over an API-set ai_source fires the flag and
    drops the marker (the API no longer owns the spec)."""
    existing = _app_job(
        gpu_count=1,
        config_extra={
            "ai_source": "api",
            "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
        },
    )
    q = _queries(existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.update_service_config_guarded = AsyncMock()
    client = _deploy_client(q, tmp_path)
    r = _deploy_zip(client, _ai_api_zip(), gpus=1)
    assert r.status_code == 201, r.text
    assert r.json()["overwrote_api_config"] is True
    cfg = json.loads(q.update_service_config_guarded.await_args.args[1])
    assert cfg["ai"]["default"]["provider"] == "api"  # replaced
    assert "ai_source" not in cfg  # marker popped — API no longer owns it


def test_put_ai_stamps_ai_source_marker():
    """The origin of the P13 WP9 preserve chain: a changed ``ai`` PUT stamps the
    section-level ``config['ai_source'] = 'api'`` provenance marker beside the
    row-level ``config_source`` stamp. Without this leg every preserve test
    (which fabricates the marker) would stay green while the stamp regressed —
    re-opening the exact agent-session bug the todo.md §0 refinement closes."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(
        client,
        section="ai",
        body={
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            },
        },
    )
    assert r.status_code == 200, r.text
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert new_cfg["ai_source"] == "api"
    assert new_cfg["config_source"] == "api"


def test_put_deploy_does_not_stamp_ai_source():
    """The marker is section-scoped: a changed ``deploy`` PUT must NOT claim
    API provenance over [ai.*] (pins the ``section == 'ai'`` check)."""
    q = _queries(_app_job())
    client = _client(q)
    r = _put(client, section="deploy", body={"gpus": 2})
    assert r.status_code == 200, r.text
    args, _ = q.update_app_config.await_args
    new_cfg = json.loads(args[1])
    assert "ai_source" not in new_cfg
    assert new_cfg["config_source"] == "api"


def test_put_ai_then_silent_redeploys_preserve_chain(tmp_path):
    """End-to-end PUT → marker → preserve chain, no fabricated marker: the real
    ``ai`` PUT stamps ``ai_source``, then TWO consecutive silent ZIP redeploys
    (the second over the ``config_source='deploy'`` re-stamp — the §0-refinement
    hole) both carry the API-set spec + marker forward."""
    # Leg 1: the real ai PUT persists the spec + marker.
    q_put = _queries(_app_job())
    r = _put(
        _client(q_put),
        section="ai",
        body={
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            },
        },
    )
    assert r.status_code == 200, r.text
    cfg_after_put = json.loads(q_put.update_app_config.await_args.args[1])
    assert cfg_after_put["ai_source"] == "api"

    # Legs 2 + 3: silent redeploys over the PUT-persisted blob, then over the
    # first redeploy's output (config_source re-stamped to 'deploy').
    cfg = cfg_after_put
    for leg in (1, 2):
        q = _queries(_app_job(config=json.dumps(cfg)))
        q.get_service_endpoint = AsyncMock(return_value=None)
        q.get_job_gpus = AsyncMock(return_value=[])
        q.update_service_config_guarded = AsyncMock()
        r = _deploy_zip(_deploy_client(q, tmp_path), _node_zip())  # no nerdit.toml
        assert r.status_code == 201, f"redeploy #{leg}: {r.text}"
        cfg = json.loads(q.update_service_config_guarded.await_args.args[1])
        assert cfg["ai"]["cheap"]["model"] == "gpt-4o-mini", f"redeploy #{leg}"
        assert cfg["ai_source"] == "api", f"redeploy #{leg}"
        assert cfg["config_source"] == "deploy", f"redeploy #{leg}"


def test_audit_row_config_app_update():
    q = _queries(_app_job())
    client = _client(q, with_audit=True)
    r = _put(client, section="ai", body={"default": None})
    assert r.status_code == 200
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "config.app_update"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "demo"
    assert kwargs["result"] == "ok"
    params = json.loads(kwargs["params_redacted"])
    assert params["changed_keys"] == ["ai.default"]
    assert params["requires_restart"] is True
    assert params["restarted"] is False


# --- section: db (P15 / WP4) --------------------------------------------------


def _db_row(status: JobStatus = JobStatus.running, owner: str | None = "tok-sub") -> Job:
    # `POST /databases` stamps `submitted_by_token` via new_workload_row, so a
    # realistic row is owned; `owner` drives the H2 cross-owner regressions.
    return Job(
        kind=JobKind.database,
        service_name="pg",
        name="pg",
        status=status,
        gpu_count=0,
        config=json.dumps({"backend": "postgres"}),
        submitted_by_token=owner,
    )


def _queries_db(job: Job | None, db_row: Job | None = None) -> AsyncMock:
    q = _queries(job)
    q.get_resource_by_ref = AsyncMock(return_value=db_row)
    return q


def test_db_section_write_persists_and_marks_source():
    q = _queries_db(_app_job(), _db_row())
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 200, r.text
    assert r.json()["requires_restart"] is True
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["db"] == {"main": {"provider": "managed", "database": "pg"}}
    assert new_cfg["db_source"] == "api"


def test_db_section_not_provisioned_rejected_on_config_put():
    q = _queries_db(_app_job(), None)
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 422
    assert r.json()["code"] == "db.not_provisioned"
    q.update_app_config.assert_not_awaited()


# --- H2: the managed [db.*] gate authorizes the REFERENCED database row ------
#
# A managed binding hands the app the database's minted password at launch
# (DATABASE_URL), so binding to a row is acting on it: without an ownership /
# scope check any submitter could read another owner's database.


def test_db_section_cross_owner_database_forbidden():
    """A submitter binding its own app to another owner's database is 403."""
    q = _queries_db(_app_job(), _db_row(owner="tok-other"))
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 403, r.text
    assert r.json()["code"] == "forbidden"
    q.update_app_config.assert_not_awaited()


def test_db_section_out_of_scope_database_forbidden():
    """A token scoped to its own app may not bind a database outside that scope."""
    job = _app_job(submitted_by_token="tok-scoped")
    q = _queries_db(job, _db_row(owner="tok-scoped"))
    r = _put(
        _client(q),
        section="db",
        body={"main": {"provider": "managed", "database": "pg"}},
        raw=SCOPED_RAW,
    )
    assert r.status_code == 403, r.text
    body = r.json()
    assert body["code"] == "forbidden"
    assert "pg" in body["message"]  # the scope refusal names the target
    q.update_app_config.assert_not_awaited()


def test_db_section_admin_binds_any_database():
    """Admin bypasses both legs, as everywhere else."""
    q = _queries_db(_app_job(), _db_row(owner="tok-other"))
    r = _put(
        _client(q),
        section="db",
        body={"main": {"provider": "managed", "database": "pg"}},
        raw=LEGACY,
    )
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["db"] == {"main": {"provider": "managed", "database": "pg"}}


def test_db_section_external_needs_no_row():
    q = _queries_db(_app_job(), None)
    r = _put(
        _client(q),
        section="db",
        body={"cache": {"provider": "external", "url": "redis://h/0", "password": "${secrets.P}"}},
    )
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["db"]["cache"]["provider"] == "external"


def test_db_section_null_deletes_binding():
    job = _app_job(
        config_extra={"db": {"main": {"provider": "managed", "database": "pg"}}, "db_source": "api"}
    )
    q = _queries_db(job, _db_row())
    r = _put(_client(q), section="db", body={"main": None})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "db" not in new_cfg


def test_db_invalid_url_rejected_on_config_put():
    q = _queries_db(_app_job(), None)
    r = _put(
        _client(q),
        section="db",
        body={"cache": {"provider": "external", "url": "mysql://h", "password": "${secrets.P}"}},
    )
    assert r.status_code == 422
    assert r.json()["code"] == "db.external_url_invalid"


def test_db_view_masks_external_password():
    job = _app_job(
        config_extra={
            "db": {
                "cache": {"provider": "external", "url": "redis://h/0", "password": "${secrets.P}"}
            }
        }
    )
    r = _client(_queries_db(job, None)).get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    body = r.json()
    assert body["db"]["cache"]["password"] == "***"
    assert body["db"]["cache"]["url"] == "redis://h/0"


# --- PUT deploy.edge_auth (P25 WP4j) ------------------------------------------
#
# ``edge_auth`` follows the ``release``/``cutover`` no-form-field template, with
# two properties of its own: it is the ONLY way to disarm edge auth (deleting the
# key from nerdit.toml carries forward instead — D-P25-5), and it is deliberately
# absent from the restart set because the credential is materialized into the
# Caddy route object, not the container env (§3.4.6 / D-P25-8).

_EDGE_AUTH = {"user": "ops", "password": "${secrets.APP_PW}"}


def test_put_deploy_edge_auth_applies_without_requiring_a_restart():
    q = _queries(_app_job())
    r = _put(_client(q), body={"edge_auth": _EDGE_AUTH})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is True
    assert body["requires_restart"] is False, "edge auth is applied by the proxy, not a relaunch"
    assert body["restarted"] is False
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["edge_auth"] == _EDGE_AUTH
    assert new_cfg["config_source"] == "api"
    assert new_cfg["config_revision"] == 4


def test_put_deploy_edge_auth_is_in_the_tracked_diff_not_the_restart_set():
    from nerdit.daemon.routes.app_config import (
        _DEPLOY_MUTABLE_KEYS,
        _DEPLOY_TRACKED_KEYS,
        _RESTART_DEPLOY_KEYS,
    )

    assert "edge_auth" in _DEPLOY_MUTABLE_KEYS
    assert "edge_auth" in _DEPLOY_TRACKED_KEYS
    assert "edge_auth" not in _RESTART_DEPLOY_KEYS


def test_put_deploy_edge_auth_noop_write_does_not_bump_provenance():
    """Dict inequality drives the `changed` diff for a dict value exactly as it
    does for a scalar — so a re-PUT of the same declaration is a no-op."""
    q = _queries(_app_job(config_extra={"edge_auth": _EDGE_AUTH}))
    r = _put(_client(q), body={"edge_auth": _EDGE_AUTH})
    assert r.status_code == 200
    q.update_app_config.assert_not_awaited()


def test_put_deploy_edge_auth_null_disarms_it():
    """The documented — and only — disarm. After this write the row carries no
    ``edge_auth`` at all, so the next proxy reconcile publishes an auth-free
    route (D-P25-8's "absent" leg, not its "malformed" leg)."""
    q = _queries(_app_job(config_extra={"edge_auth": _EDGE_AUTH}))
    r = _put(_client(q), body={"edge_auth": None})
    assert r.status_code == 200, r.text
    assert r.json()["applied"] is True
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "edge_auth" not in new_cfg


def test_put_deploy_edge_auth_rejects_a_literal_password_without_echoing_it():
    """The API path is held to exactly the grammar the ZIP path enforces, and
    the 422 never carries the rejected value (bug #28 / finding #6 channel)."""
    q = _queries(_app_job())
    r = _put(_client(q), body={"edge_auth": {"user": "ops", "password": "hunter2-FORGED"}})
    assert r.status_code == 422
    assert r.json()["code"] == "deploy.invalid"
    assert "hunter2-FORGED" not in r.content.decode()
    q.update_app_config.assert_not_awaited()


def test_put_deploy_edge_auth_rejects_an_unknown_subkey():
    """``EdgeAuthConfig`` is ``extra='forbid'``, so a typo fails loudly rather
    than being silently ignored (and then silently NOT enforced)."""
    q = _queries(_app_job())
    r = _put(_client(q), body={"edge_auth": {"user": "ops", "passwd": "${secrets.APP_PW}"}})
    assert r.status_code == 422
    q.update_app_config.assert_not_awaited()


def test_get_view_masks_the_edge_auth_password_and_surfaces_the_user():
    """The GET view reads a PERSISTED blob, not a validated model, so it may not
    assume the grammar ever held: ``redact_section`` masks the ``password`` leaf
    exactly as it does for ``[ai.*].api_key``. ``user`` is an identifier, not a
    credential, and rides through verbatim."""
    q = _queries(_app_job(config_extra={"edge_auth": _EDGE_AUTH}))
    r = _client(q).get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    assert r.json()["deploy"]["edge_auth"] == {"user": "ops", "password": "***"}


def test_get_view_never_surfaces_a_hand_forged_literal_password():
    """The reason the defensive mask exists: a blob that reached the row around
    ``EdgeAuthConfig`` (a direct DB edit, a future write path) must not be able
    to publish a real password through an authenticated read."""
    q = _queries(_app_job(config_extra={"edge_auth": {"user": "ops", "password": "hunter2"}}))
    r = _client(q).get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    assert "hunter2" not in r.content.decode()
    assert r.json()["deploy"]["edge_auth"]["password"] == "***"


def test_get_view_masks_a_non_table_edge_auth_blob_wholesale():
    """A blob whose SHAPE cannot be reasoned about surfaces its existence only —
    it could be anything, including a bare password string."""
    q = _queries(_app_job(config_extra={"edge_auth": "hunter2"}))
    r = _client(q).get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    assert "hunter2" not in r.content.decode()
    assert r.json()["deploy"]["edge_auth"] == "***"


@pytest.mark.parametrize(
    "blob",
    [
        # A secret held under a key the shallow denylist does not mask.
        {"user": "ops", "credential": "hunter2"},
        {"user": "ops", "password": "${secrets.APP_PW}", "note": "hunter2"},
        # A secret nested one level below the shallow redaction.
        {"metadata": {"password": "hunter2"}},
        {"user": "ops", "password": {"literal": "hunter2"}},
    ],
)
def test_get_view_masks_a_malformed_edge_auth_mapping_wholesale(blob):
    """A persisted mapping that did NOT go through ``EdgeAuthConfig`` (a hand
    edit, a legacy import, a future bypassing writer) must not reflect a secret
    held under an unexpected key or nested one level down. The projection is an
    ALLOWLIST — only ``{user?, password?}`` string mappings are surfaced
    field-by-field; anything else is masked wholesale (Codex PR #111 review)."""
    q = _queries(_app_job(config_extra={"edge_auth": blob}))
    r = _client(q).get("/config/apps/demo", headers=_auth(RO_RAW))
    assert r.status_code == 200
    assert "hunter2" not in r.content.decode()
    assert r.json()["deploy"]["edge_auth"] == "***"


def test_put_deploy_edge_auth_dry_run_previews_without_writing():
    q = _queries(_app_job())
    r = _put(_client(q), body={"edge_auth": _EDGE_AUTH}, dry_run="true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applied"] is False
    assert body["requires_restart"] is False
    assert body["view"]["deploy"]["edge_auth"] == {"user": "ops", "password": "***"}
    q.update_app_config.assert_not_awaited()

    # ...and the disarm direction round-trips back to None.
    q2 = _queries(_app_job(config_extra={"edge_auth": _EDGE_AUTH}))
    r2 = _put(_client(q2), body={"edge_auth": None}, dry_run="true")
    assert r2.status_code == 200
    assert r2.json()["view"]["deploy"]["edge_auth"] is None
    q2.update_app_config.assert_not_awaited()


def test_put_deploy_edge_auth_leaves_platform_owned_build_state_alone():
    q = _queries(
        _app_job(
            config_extra={"edge_auth": _EDGE_AUTH, "release_pending": 2, "cutover_pending": {}}
        )
    )
    r = _put(_client(q), body={"edge_auth": None})
    assert r.status_code == 200
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert "edge_auth" not in new_cfg
    assert new_cfg["release_pending"] == 2
    assert new_cfg["build_version"] == 2


def test_edge_auth_audit_params_mask_the_password_ref():
    """A rejected credential-shaped declaration reaches the audit row as key
    NAMES only. Leaf-name masking (``password`` is in ``_REDACT_KEYS``) was the
    old defence; the stamp now carries no submitted values at all, so a refactor
    that turned the ref into a resolved VALUE still could not persist it."""
    # A REJECTED write is where the up-front stamp is what gets recorded (the
    # success path replaces it with the changed-keys summary) — and it is the
    # case that matters: the rejected value is the credential-shaped one.
    q = _queries(_app_job())
    r = _put(
        _client(q, with_audit=True),
        body={"edge_auth": {"user": "ops", "password": "hunter2-FORGED"}},
    )
    assert r.status_code == 422
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "config.app_update"
    params = json.loads(kwargs["params_redacted"])
    assert params == {"section": "deploy", "submitted_keys": ["edge_auth"], "restart": False}
    assert "hunter2-FORGED" not in kwargs["params_redacted"]

    # The accepted write records the changed key by NAME, never the declaration.
    q2 = _queries(_app_job())
    r2 = _put(_client(q2, with_audit=True), body={"edge_auth": _EDGE_AUTH})
    assert r2.status_code == 200, r2.text
    kwargs2 = q2.insert_audit_log.await_args.kwargs
    params2 = json.loads(kwargs2["params_redacted"])
    assert params2["changed_keys"] == ["deploy.edge_auth"]
    assert params2["requires_restart"] is False
    assert "APP_PW" not in kwargs2["params_redacted"]


# --- no submitted value ever reaches the audit trail --------------------------
#
# The pre-validation stamp is what a 422 or a dry run records, and leaf-name
# masking cannot see a credential embedded IN a value: an external `[db.*]` url
# carries its password in the userinfo, which `config/project.py` already treats
# as a leak hazard on the 422 envelope, `BindingNotReady`, `job_logs` and
# diagnose. The audit row and the admin `audit.*` stream were the gap.

_DB_CANARY = "P4ssw0rdCANARY"


class _RecordingBus:
    """Event bus stand-in that keeps every published frame."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    def publish(self, frame: dict) -> None:
        self.frames.append(frame)


def _bus_client(queries: AsyncMock, bus: _RecordingBus) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(app_config_router)
    app.state.queries = queries
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


def test_db_url_password_never_reaches_the_audit_row_on_rejection():
    q = _queries_db(_app_job(), None)
    bus = _RecordingBus()
    r = _put(
        _bus_client(q, bus),
        section="db",
        body={
            "default": {
                "provider": "external",
                "url": f"postgresql://user:{_DB_CANARY}@db.example.com:5432/app",
                "password": "${secrets.DB_PW}",
            }
        },
    )
    assert r.status_code == 422
    assert _DB_CANARY not in r.text
    row, frame = _recorded(q, bus)
    assert _DB_CANARY not in row and _DB_CANARY not in frame
    assert json.loads(row) == {"section": "db", "submitted_keys": ["default"], "restart": False}


def test_db_binding_dry_run_records_names_only():
    q = _queries_db(_app_job(), None)
    bus = _RecordingBus()
    r = _put(
        _bus_client(q, bus),
        section="db",
        body={
            "cache": {
                "provider": "external",
                "url": f"redis://{_DB_CANARY}.example.com/0",
                "password": "${secrets.P}",
            }
        },
        dry_run="true",
    )
    assert r.status_code == 200, r.text
    assert r.json()["applied"] is False
    row, frame = _recorded(q, bus)
    assert _DB_CANARY not in row and _DB_CANARY not in frame
    assert json.loads(row) == {"section": "db", "submitted_keys": ["cache"], "restart": False}


def test_db_binding_commit_records_changed_key_names_only():
    q = _queries_db(_app_job(), None)
    bus = _RecordingBus()
    r = _put(
        _bus_client(q, bus),
        section="db",
        body={
            "cache": {
                "provider": "external",
                "url": f"redis://{_DB_CANARY}.example.com/0",
                "password": "${secrets.P}",
            }
        },
    )
    assert r.status_code == 200, r.text
    row, frame = _recorded(q, bus)
    assert _DB_CANARY not in row and _DB_CANARY not in frame
    params = json.loads(row)
    assert params["changed_keys"] == ["db.cache"]
    assert params["section"] == "db"
    assert params["requires_restart"] is True


# --- M9 second ingress: the config API folds the implicit data wedge ----------
#
# `resolve_effective_fields` folds `data:/data` in BEFORE validating, so the
# validated list is the persisted list. This route validated the DECLARED list
# and persisted it verbatim, so it accepted MAX_VOLUMES without a `data` entry
# — a row that then launched with no data mount under a live NERDIT_DATA_DIR
# and 422'd on every later silent redeploy.

_WEDGED = {"config_extra": {"volumes": ["data:/data"], "env": {"NERDIT_DATA_DIR": "/data"}}}


def test_put_deploy_volumes_folds_the_data_wedge():
    q = _queries(_app_job(**_WEDGED))
    r = _put(_client(q), body={"volumes": ["cache:/var/cache"]})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    # The persisted list is the VALIDATED list, wedge included.
    assert new_cfg["volumes"] == ["cache:/var/cache", "data:/data"]


def test_put_deploy_volumes_declared_cap_counts_the_wedge():
    """8 declared without a `data` entry is 9 persisted — refused at the ingress.

    Pre-fix this returned 200 and poisoned the row: the launch mounted 8
    volumes with no `/data`, and the next `nerdit deploy .` / `nerdit dev` /
    workspace deploy 422'd on `at most 8 volumes allowed, got 9`.
    """
    q = _queries(_app_job(**_WEDGED))
    r = _put(_client(q), body={"volumes": [f"v{i}:/m{i}" for i in range(8)]})
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["code"] == "deploy.invalid"
    assert "at most 8 volumes allowed, got 9" in body["message"]
    q.update_app_config.assert_not_awaited()


def test_put_deploy_volumes_seven_declared_plus_wedge_is_accepted():
    q = _queries(_app_job(**_WEDGED))
    r = _put(_client(q), body={"volumes": [f"v{i}:/m{i}" for i in range(7)]})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert len(new_cfg["volumes"]) == 8
    assert new_cfg["volumes"][-1] == "data:/data"


def test_put_deploy_volumes_no_wedge_on_a_row_that_never_had_one():
    """A `POST /services` row gets no implicit data volume from the deploy path,
    so the config API must not invent one for it either."""
    q = _queries(_app_job())  # no `volumes` in config
    r = _put(_client(q), body={"volumes": ["cache:/var/cache"]})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["volumes"] == ["cache:/var/cache"]


# --- H2 follow-ups: who the [db.*] gate may refuse ----------------------------


def test_db_section_null_owner_database_is_bindable_by_its_app_owner():
    """A database created with the legacy global token (NULL owner) is the
    single-operator install's normal case; admin-only there would break
    `nerdit db create pg` + a scoped CI/tunnel token deploying against it."""
    q = _queries_db(_app_job(), _db_row(owner=None))
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["db"] == {"main": {"provider": "managed", "database": "pg"}}


def test_db_cross_owner_refusal_names_the_binding_and_the_database():
    """The owner leg must not reuse the generic row denial: at the deploy
    ingress that envelope is byte-identical to "you do not own the APP"."""
    q = _queries_db(_app_job(), _db_row(owner="tok-other"))
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 403, r.text
    body = r.json()
    assert "db.main" in body["message"]
    assert "'pg'" in body["message"]


def test_unrelated_section_write_survives_a_carried_forward_foreign_binding():
    """An admin-authored (or pre-fix) foreign binding must not lock the app's
    own owner out of every later `deploy`/`ai` edit — the carried-forward spec
    is not an authoring act."""
    job = _app_job(config_extra={"db": {"main": {"provider": "managed", "database": "pg"}}})
    q = _queries_db(job, _db_row(owner="tok-other"))
    r = _put(_client(q), body={"memory_limit": "512m"})
    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["memory_limit"] == "512m"
    assert new_cfg["db"] == {"main": {"provider": "managed", "database": "pg"}}


def test_restating_a_foreign_binding_unchanged_is_still_allowed():
    """An idempotent re-PUT of the very spec already persisted changes nothing,
    so it is not a new authorization either."""
    job = _app_job(config_extra={"db": {"main": {"provider": "managed", "database": "pg"}}})
    q = _queries_db(job, _db_row(owner="tok-other"))
    r = _put(_client(q), section="db", body={"main": {"provider": "managed", "database": "pg"}})
    assert r.status_code == 200, r.text


def test_changing_a_foreign_binding_is_still_refused():
    """Carry-forward is not a laundering path: any edit to the spec re-gates."""
    job = _app_job(config_extra={"db": {"main": {"provider": "managed", "database": "pg"}}})
    q = _queries_db(job, _db_row(owner="tok-other"))
    r = _put(
        _client(q),
        section="db",
        body={"other": {"provider": "managed", "database": "pg"}},
    )
    assert r.status_code == 403, r.text
    q.update_app_config.assert_not_awaited()
