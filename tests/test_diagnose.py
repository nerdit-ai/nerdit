"""Tests for ``GET /api/services/{ident}/diagnose`` + the remediation classifier (P13 WP4).

Two layers:

* a pure unit table over :func:`derive_remediation` (no HTTP) pinning the locked
  §1.3 precedence, including the forensics-freshness gate; and
* a pattern-2 route harness (``httpx.ASGITransport`` over a real in-memory DB
  under the full middleware stack) pinning the assembled payload, the
  owner-or-admin gate, and the value-leak contract.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import NerditSettings
from nerdit.core.data.backend import PostgresBackend
from nerdit.core.data.controller import DataController
from nerdit.core.models.backend import OllamaBackend, VllmBackend
from nerdit.core.models.controller import ModelController
from nerdit.core.secrets import SecretManager
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.remediation import (
    AcmeWait,
    BindingWait,
    RemediationCode,
    derive_remediation,
)
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.database import Database
from nerdit.db.models import (
    ApiToken,
    Job,
    JobKind,
    JobStatus,
    LogStream,
    TokenRole,
)
from nerdit.db.queries import Queries

_STARTED = "2026-07-10T12:00:00+00:00"
_CRASH_FRESH = "2026-07-10T12:05:00+00:00"
_CRASH_STALE = "2026-07-10T11:00:00+00:00"


# --- pure unit table over derive_remediation ---------------------------------


def _job(**over) -> Job:
    fields = dict(
        id="svc-1",
        service_name="my-app",
        name="my-app",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.failed,
        desired_state="running",
        restart_policy="always",
        restart_count=0,
    )
    fields.update(over)
    return Job(**fields)


def _last_deploy(**over) -> dict:
    ld = dict(version=4, action="redeploy", phase="failed", started_at=_STARTED)
    ld.update(over)
    return ld


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hc, expected_type, probe_fn",
    [
        ({"type": "tcp"}, "tcp", "check_tcp"),
        ({"path": "/health"}, "http", "check_health"),
        ({"type": "grpc", "path": "/health"}, "http", "check_health"),  # junk → http
    ],
)
async def test_fresh_health_probe_reports_probe_type(monkeypatch, hc, expected_type, probe_fn):
    """P14 WP-C1: ``_fresh_health_probe`` branches on type and reports the probe kind."""
    from nerdit.daemon.routes import service_diagnose as svc_routes
    from nerdit.db.models import ServiceEndpoint

    called: list[str] = []

    async def _fake_tcp(host_port, timeout):
        called.append("check_tcp")
        return True

    async def _fake_http(host_port, path, timeout):
        called.append("check_health")
        return 200

    monkeypatch.setattr(svc_routes, "check_tcp", _fake_tcp)
    monkeypatch.setattr(svc_routes, "check_health", _fake_http)

    job = _job(status=JobStatus.running, container_id="c1", health_check=hc)
    endpoint = ServiceEndpoint(service_name="my-app", container_port=8000, host_port=9400)
    probe = await svc_routes._fresh_health_probe(job, endpoint)
    assert probe is not None
    assert probe["type"] == expected_type
    assert probe["status_code"] == 200
    assert called == [probe_fn]


def test_unit_oom_fresh_raises_memory_limit():
    cfg = {"last_deploy": _last_deploy(reason="crash_loop"), "memory_limit": "512m"}
    forensics = {"last_exit_code": 137, "oom_killed": True, "last_crash_at": _CRASH_FRESH}
    code, detail = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.raise_memory_limit
    assert "512m" in detail


@pytest.mark.parametrize("reason", ["volume_invalid", "volume_conflict"])
def test_unit_volume_failure_maps_to_volume_invalid(reason):
    cfg = {"last_deploy": _last_deploy(reason=reason)}
    code, detail = derive_remediation(_job(), cfg, {}, BindingWait(), None)
    assert code is RemediationCode.volume_invalid
    assert "volumes" in detail


def test_unit_stale_oom_does_not_outrank_fresh_build_failure():
    # Old OOM (stale) + a fresh redeploy build_failed with a rollback target →
    # rollback, NOT raise_memory_limit (the freshness gate).
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed"),
        "previous_image": "nerdit-app/my-app:3",
    }
    forensics = {"last_exit_code": 137, "oom_killed": True, "last_crash_at": _CRASH_STALE}
    code, _ = derive_remediation(
        _job(status=JobStatus.running), cfg, forensics, BindingWait(), None
    )
    assert code is RemediationCode.rollback


def test_unit_model_wait_serves_missing_model():
    cfg = {"last_deploy": _last_deploy(phase="launching", reason=None)}
    wait = BindingWait(
        waiting=True, model_wait=True, messages=("model 'llama3.1:8b' is not served",)
    )
    code, detail = derive_remediation(_job(status=JobStatus.building), cfg, {}, wait, None)
    assert code is RemediationCode.serve_missing_model
    assert "llama3.1:8b" in detail


def test_unit_secret_wait_sets_missing_secret():
    wait = BindingWait(waiting=True, secret_wait=True, messages=("secret 'OPENAI_KEY' not set",))
    code, _ = derive_remediation(_job(status=JobStatus.building), {}, {}, wait, None)
    assert code is RemediationCode.set_missing_secret


def test_unit_model_wait_precedes_secret_wait():
    # Both blocked at once → the model wait wins (rule 2 before rule 3).
    wait = BindingWait(waiting=True, model_wait=True, secret_wait=True, messages=("m",))
    code, _ = derive_remediation(_job(), {}, {}, wait, None)
    assert code is RemediationCode.serve_missing_model


def test_unit_redeploy_build_fail_with_previous_image_rolls_back():
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed"),
        "previous_image": "nerdit-app/my-app:3",
    }
    code, detail = derive_remediation(_job(status=JobStatus.running), cfg, {}, BindingWait(), None)
    assert code is RemediationCode.rollback
    assert "rollback" in detail.lower()


def test_unit_platform_build_error_outranks_rollback():
    """(BUG-1) Rule 3e: a HOST-fault build outranks Rule 4's rollback.

    Rolling back restores service but says nothing about why every future build
    on this node fails the same way — so the platform advice wins, and the
    rollback pointer is MERGED into its detail rather than lost.
    """
    from nerdit.db.models import ErrorClass

    cfg = {
        "last_deploy": _last_deploy(reason="build_failed"),
        "previous_image": "nerdit-app/my-app:3",
    }
    job = _job(status=JobStatus.running, error_class=ErrorClass.platform_error)
    code, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.platform_build_unavailable
    assert "buildx" in detail
    assert "rollback" in detail  # the Rule 4 pointer survives the re-ranking
    assert "DOCKER_CONFIG" in detail
    assert "submit the deploy again" in detail


def test_unit_platform_detail_covers_both_toolchain_faults():
    """(Codex 3804646811 coupling) Rule 3e now has TWO setters, so say both.

    ``BuildPlatformError`` is raised both when the BuildKit builder is missing
    and when the daemon has no ``docker`` CLI at all. The detail was written
    when buildx was the only setter; inheriting buildx-only advice for a
    missing-CLI fault would send the operator to install a plugin for a CLI
    that is not there.
    """
    from nerdit.db.models import ErrorClass

    cfg = {"last_deploy": _last_deploy(reason="build_failed", action="create")}
    job = _job(error_class=ErrorClass.platform_error)
    _, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert "buildx" in detail
    assert "docker CLI" in detail


def test_unit_build_failed_without_platform_class_still_rolls_back():
    """Leg B: the ordinary build failure is untouched by Rule 3e."""
    from nerdit.db.models import ErrorClass

    cfg = {
        "last_deploy": _last_deploy(reason="build_failed"),
        "previous_image": "nerdit-app/my-app:3",
    }
    job = _job(status=JobStatus.running, error_class=ErrorClass.user_error)
    code, _ = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.rollback


def test_unit_platform_class_from_last_deploy_on_a_reverted_redeploy_row():
    """Leg C (sign-off BLOCKER): the REDEPLOY leg, where the row column is NULL.

    A build that fails on a host fault over a still-live previous container is
    settled by ``_settle_failed_generation``'s revert branch: the row goes back
    to ``running`` and ``update_service_config`` /
    ``revert_service_config_guarded`` deliberately NULL ``error_class``
    (F2-STALE-ERR), so the class survives ONLY inside
    ``config['last_deploy']``. Gating Rule 3e on the row column alone therefore
    let every redeploy — the exact case the rule was written for — fall through
    to Rule 4's rollback. Mirrors the row-column-then-last_deploy precedence the
    diagnose route already uses.
    """
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed", error_class="PLATFORM_ERROR"),
        "previous_image": "nerdit-app/my-app:3",
    }
    job = _job(status=JobStatus.running, error_class=None)
    code, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.platform_build_unavailable
    assert "buildx" in detail
    # ...and the previous_image merge branch is reachable from here, so the
    # rollback pointer the docstring promises really does survive.
    assert f"POST /deploy/{job.service_name}/rollback" in detail
    # The rollback pointer is a stopgap, not the fix: the resubmit step rides
    # this leg too, otherwise the redeploy case is left serving v(n-1) forever.
    assert "submit the deploy again" in detail


def test_unit_user_error_from_last_deploy_on_a_reverted_redeploy_row_still_rolls_back():
    """Leg D: same reverted shape, an ordinary source-side build failure — the
    ``last_deploy`` fallback must not widen Rule 3e beyond PLATFORM_ERROR."""
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed", error_class="USER_ERROR"),
        "previous_image": "nerdit-app/my-app:3",
    }
    job = _job(status=JobStatus.running, error_class=None)
    code, _ = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.rollback


def test_unit_platform_build_error_without_previous_image_omits_the_rollback_pointer():
    from nerdit.db.models import ErrorClass

    cfg = {"last_deploy": _last_deploy(reason="build_failed", action="create")}
    job = _job(error_class=ErrorClass.platform_error)
    code, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.platform_build_unavailable
    assert "rollback" not in detail


def test_unit_platform_build_error_tells_the_caller_to_submit_the_deploy_again():
    """(Codex 3804646875) Repairing the host does NOT resume the failed build.

    ``core/app_build.py``'s ``BuildPlatformError`` branch settles the generation
    terminally and its enclosing ``finally`` drops the uploaded build context on
    either outcome, so nothing re-enters the builder once the operator fixes the
    machine. Without a resubmit step the prescribed remediation leaves the
    service failed forever while /diagnose keeps describing the fault in the
    present tense. Asserted on the fresh-create shape, where there is no
    rollback pointer to hide behind.
    """
    from nerdit.db.models import ErrorClass

    cfg = {"last_deploy": _last_deploy(reason="build_failed", action="create")}
    job = _job(error_class=ErrorClass.platform_error)
    _, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert "submit the deploy again" in detail
    assert "rollback" not in detail  # still no rollback pointer without a previous image


def test_unit_fresh_crash_loop_fixes_start_command():
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {"last_exit_code": 1, "oom_killed": False, "last_crash_at": _CRASH_FRESH}
    code, _ = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.fix_start_command


# --- (P33) Rule 4b: the image needs privileges the sandbox drops -------------


def test_unit_priv_denied_crash_loop_says_image_needs_privileges():
    """The field failure, 2026-08-23: a stock ``FROM nginx`` static site.

    Its root entrypoint dies on ``chown(...) Operation not permitted``, the
    restart budget burns, and rule 5 used to answer ``fix_start_command`` — a
    key the operator can edit forever without effect.
    """
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "priv_denied": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, detail = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.image_needs_privileges
    # Value-free, and it names the actual config section (``[containers]``, not
    # ``[security]``) plus the drop-in for the case that produced it.
    assert "drop_all_caps" in detail
    assert "nginx-unprivileged" in detail
    assert "[deploy].start cannot fix this" in detail


def test_unit_unrelated_crash_loop_still_fixes_start_command():
    """The negative: no ``priv_denied`` flag ⇒ rule 5 is untouched."""
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "priv_denied": False,
        "last_crash_at": _CRASH_FRESH,
    }
    code, detail = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.fix_start_command
    assert "exit 1" in detail


def test_unit_priv_denied_loses_to_a_fresh_cgroup_oom():
    """Precedence: rule 1b (OOM kill) outranks rule 4b, as the table says.

    A container killed by the cgroup is the bigger, likelier-current problem;
    a stale privilege refusal earlier in the same tail must not outrank it.
    """
    cfg = {"last_deploy": _last_deploy(reason="crash_loop"), "memory_limit": "512m"}
    forensics = {
        "last_exit_code": 137,
        "oom_killed": True,
        "priv_denied": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.raise_memory_limit


def test_unit_priv_denied_loses_to_a_fresh_gpu_oom_on_a_model_row():
    """Precedence: rule 1 (GPU OOM on a model row) still wins outright."""
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "gpu_oom": True,
        "priv_denied": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(kind=JobKind.model), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.model_gpu_oom


def test_unit_stale_priv_denied_does_not_fire():
    """The freshness gate covers rule 4b exactly as it covers rule 5."""
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed", action="redeploy"),
        "previous_image": "nerdit-app/my-app:3",
    }
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "priv_denied": True,
        "last_crash_at": _CRASH_STALE,
    }
    code, _ = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.rollback


def test_unit_priv_denied_outside_the_user_error_band_falls_through():
    """Rule 4b inherits rule 5's exit-code band — 137 is not a user error."""
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 137,
        "oom_killed": False,
        "priv_denied": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.inspect_logs


async def test_priv_denied_diagnose_payload(harness):
    """The route projects the persisted flag and the classifier reads it."""
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-site:1",
        "build_version": 1,
        "port": 80,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "failed",
            "reason": "crash_loop",
            "started_at": _STARTED,
        },
        "last_exit_code": 1,
        "oom_killed": False,
        "priv_denied": True,
        "last_crash_at": _CRASH_FRESH,
    }
    await _seed_service(queries, config=cfg)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["forensics"]["priv_denied"] is True
    assert body["remediation"]["code"] == "image_needs_privileges"


def test_unit_degraded_fixes_health_check():
    code, _ = derive_remediation(_job(status=JobStatus.degraded), {}, {}, BindingWait(), None)
    assert code is RemediationCode.fix_health_check


def test_unit_transient_pull_failure_retries_later():
    cfg = {
        "last_deploy": _last_deploy(
            reason="image_pull_failed", error_message="connection reset by peer"
        )
    }
    code, _ = derive_remediation(_job(), cfg, {}, BindingWait(), None)
    assert code is RemediationCode.retry_later


def test_unit_permanent_pull_failure_inspects_logs():
    cfg = {
        "last_deploy": _last_deploy(
            reason="image_pull_failed", error_message="manifest unknown: not found"
        )
    }
    code, _ = derive_remediation(_job(), cfg, {}, BindingWait(), None)
    assert code is RemediationCode.inspect_logs


def test_unit_healthy_running_needs_nothing():
    cfg = {"last_deploy": _last_deploy(phase="healthy", reason=None)}
    code, _ = derive_remediation(_job(status=JobStatus.running), cfg, {}, BindingWait(), None)
    assert code is RemediationCode.none


def test_unit_model_permanent_pull_failure_maps_to_model_pull_failed():
    # A kind=model row whose weights pull failed permanently: the bare server
    # answers its liveness probe, so without the model rule it would diagnose
    # ``none`` (finding #4).
    job = _job(
        kind=JobKind.model,
        status=JobStatus.running,
        error_message="Model pull failed: manifest not found",
    )
    code, detail = derive_remediation(job, {}, {}, BindingWait(), None)
    assert code is RemediationCode.model_pull_failed
    assert "restart" in detail


def test_unit_model_transient_pull_failure_retries_later():
    job = _job(
        kind=JobKind.model,
        status=JobStatus.running,
        error_message="Model pull failed: connection reset by peer",
    )
    code, _ = derive_remediation(job, {}, {}, BindingWait(), None)
    assert code is RemediationCode.retry_later


def test_unit_model_pulled_with_stale_error_falls_through_to_none():
    # A successful pull (model_pulled) with a stale error_message must not fire
    # the model rule.
    job = _job(
        kind=JobKind.model,
        status=JobStatus.running,
        error_message="Model pull failed: transient blip earlier",
    )
    code, _ = derive_remediation(job, {"model_pulled": True}, {}, BindingWait(), None)
    assert code is RemediationCode.none


def test_unit_service_row_with_pull_message_unchanged():
    # Identical inputs but kind=service: the model rule must not fire — a running
    # healthy service stays ``none``.
    job = _job(
        kind=JobKind.service,
        status=JobStatus.running,
        error_message="Model pull failed: manifest not found",
    )
    code, _ = derive_remediation(job, {}, {}, BindingWait(), None)
    assert code is RemediationCode.none


# --- route harness (real in-memory DB + full middleware) ---------------------

LEGACY = "legacy-global"
ADMIN_RAW = "admin-raw"
OWNER_RAW = "owner-raw"
OTHER_RAW = "other-raw"

_TOKENS = {
    "tok-admin": ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    "tok-owner": ApiToken(
        id="tok-owner", name="o", role=TokenRole.submitter, token_hash=hash_token(OWNER_RAW)
    ),
    "tok-other": ApiToken(
        id="tok-other", name="x", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
}


def _diagnose_app(queries, secrets) -> FastAPI:
    """The pattern-2 route app the fixtures below share.

    Factored out of ``harness`` (pure motion) so a second fixture can hand a
    test the FastAPI object itself — the P26 WP2 legs need to install a stub
    ``proxy_manager`` on ``app.state``, which the client alone cannot reach.
    """
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router, prefix="/api")
    app.state.queries = queries
    app.state.settings = NerditSettings()
    app.state.secret_manager = secrets
    app.state.model_controller = ModelController(
        OllamaBackend(),
        MagicMock(),
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
    )
    app.state.data_controller = DataController(
        PostgresBackend(),
        MagicMock(),
        queries,
    )
    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=LEGACY,
        get_queries=lambda: queries,
    )
    app.add_middleware(RequestIdMiddleware)
    return app


@pytest_asyncio.fixture
async def harness(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for tok in _TOKENS.values():
        await queries.create_api_token(tok)

    secrets = SecretManager(tmp_path / "secrets")
    app = _diagnose_app(queries, secrets)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield client, queries, secrets
    finally:
        await client.aclose()
        await db.close()


@pytest_asyncio.fixture
async def app_harness(tmp_path):
    """``harness`` plus the app object, for tests that stub ``app.state``."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for tok in _TOKENS.values():
        await queries.create_api_token(tok)

    secrets = SecretManager(tmp_path / "secrets")
    app = _diagnose_app(queries, secrets)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield client, queries, app
    finally:
        await client.aclose()
        await db.close()


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


async def _seed_service(queries, *, config: dict, owner="tok-owner", **over) -> Job:
    fields = dict(
        service_name="my-app",
        name="my-app",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.failed,
        desired_state="running",
        restart_policy="always",
        submitted_by_token=owner,
        config=json.dumps(config),
    )
    fields.update(over)
    return await queries.create_job(Job(**fields))


async def test_oom_crash_yields_raise_memory_limit(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:4",
        "build_version": 4,
        "port": 8000,
        "memory_limit": "512m",
        "last_deploy": {
            "version": 4,
            "action": "redeploy",
            "phase": "launching",
            "started_at": _STARTED,
        },
        "last_exit_code": 137,
        "oom_killed": True,
        "last_crash_at": _CRASH_FRESH,
    }
    await _seed_service(queries, config=cfg)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["remediation"]["code"] == "raise_memory_limit"
    assert body["forensics"] == {
        "last_exit_code": 137,
        "oom_killed": True,
        "gpu_oom": False,
        "priv_denied": False,
        "last_crash_at": _CRASH_FRESH,
    }


async def test_unserved_ollama_binding_yields_serve_missing_model(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "launching",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.building)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["remediation"]["code"] == "serve_missing_model"
    assert body["bindings"]["waiting"] is True
    assert any("llama3.1:8b" in m for m in body["bindings"]["messages"])


async def test_missing_secret_binding_yields_set_missing_secret(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "ai": {
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            }
        },
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "launching",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.building)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["remediation"]["code"] == "set_missing_secret"


async def test_unprovisioned_managed_db_binding_is_outcome_readable(harness):
    # (P15 WP4 review R2) An app whose ONLY binding is a managed [db.default]
    # pointing at an unprovisioned database sits in the launch retry loop. Before
    # the fix, _classify_bindings early-returned on absent [ai.*] and diagnose
    # reported waiting=false; and _pending_env_key_names omitted the db kind, so
    # the recomputed injected keys dropped DATABASE_URL. Both must be fixed.
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "db": {"default": {"provider": "managed", "database": "pg"}},
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "launching",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.building)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Gap 1: the managed [db.*] wait is now surfaced with an actionable message.
    assert body["bindings"]["waiting"] is True
    assert any(
        "not provisioned" in m and "nerdit db create" in m for m in body["bindings"]["messages"]
    )
    # Gap 2: recomputed injected keys include the db env contract.
    assert body["injected_env_keys_source"] == "recomputed"
    assert "DATABASE_URL" in body["injected_env_keys"]
    assert "NERDIT_DB_DEFAULT_URL" in body["injected_env_keys"]
    # No 64-hex minted value ever rides the diagnose body (D-B).
    assert not re.search(r"[0-9a-f]{64}", resp.text)


async def test_redeploy_build_fail_with_previous_image_yields_rollback(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:3",
        "build_version": 4,
        "previous_image": "nerdit-app/my-app:3",
        "port": 8000,
        "last_deploy": {
            "version": 4,
            "action": "redeploy",
            "phase": "failed",
            "reason": "build_failed",
            "started_at": _STARTED,
        },
    }
    # Old image still serving → status running even though the deploy failed.
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["remediation"]["code"] == "rollback"
    assert body["build"]["last_result"] == "failed"


async def test_degraded_yields_fix_health_check(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "healthy",
            "started_at": _STARTED,
        },
    }
    await _seed_service(
        queries,
        config=cfg,
        status=JobStatus.degraded,
        health_check={"path": "/health", "timeout_s": 1.0},
    )
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "fix_health_check"


async def test_healthy_running_yields_none(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "healthy",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "none"


async def test_f2_stale_error_cleared_on_redeploy_healthy(harness):
    """F2-STALE-ERR: a fail-v1 → redeploy-v2-healthy service reports no error.

    The redeploy write (update_service_config) must NULL the row error columns,
    so neither /diagnose nor the converged /wait body surfaces the prior
    generation's build failure through the row-column-precedence fallback.
    """
    from nerdit.db.models import ErrorClass

    client, queries, _ = harness
    failed_cfg = {
        "image": "nerdit-app/my-app:1",
        "build_version": 1,
        "port": 8000,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "failed",
            "reason": "build_failed",
            "error_class": "user_error",
            "error_message": "Build failed: boom",
            "started_at": _STARTED,
        },
    }
    job = await _seed_service(
        queries,
        config=failed_cfg,
        status=JobStatus.failed,
        error_class=ErrorClass.user_error,
        error_message="Build failed: boom",
    )

    # The redeploy converges to a healthy v2 through the shared service writer.
    healthy_cfg = {
        "image": "nerdit-app/my-app:2",
        "build_version": 2,
        "port": 8000,
        "last_deploy": {
            "version": 2,
            "action": "redeploy",
            "phase": "healthy",
            "reason": None,
            "error_class": None,
            "error_message": None,
            "started_at": _CRASH_FRESH,
        },
    }
    await queries.update_service_config(
        job.id,
        json.dumps(healthy_cfg),
        status=JobStatus.running,
        desired_state="running",
    )

    diag = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert diag.status_code == 200, diag.text
    dbody = diag.json()
    assert dbody["remediation"]["code"] == "none"
    assert dbody["error"]["class"] is None
    assert dbody["error"]["message"] is None

    wait = await client.get(
        "/api/services/my-app/wait",
        params={"version": 2, "timeout": 5},
        headers=_auth(OWNER_RAW),
    )
    assert wait.status_code == 200, wait.text
    wbody = wait.json()
    assert wbody["outcome"] == "converged"
    assert wbody["error_class"] is None
    assert wbody["error_message"] is None


async def test_stale_forensics_do_not_outrank_fresh_build_failure(harness):
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:3",
        "build_version": 4,
        "previous_image": "nerdit-app/my-app:3",
        "port": 8000,
        "last_deploy": {
            "version": 4,
            "action": "redeploy",
            "phase": "failed",
            "reason": "build_failed",
            "started_at": _STARTED,
        },
        # A STALE OOM from the previous generation (predates started_at).
        "last_exit_code": 137,
        "oom_killed": True,
        "last_crash_at": _CRASH_STALE,
    }
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "rollback"


async def test_post_launch_secret_shows_in_pending_not_injected(harness):
    client, queries, secrets = harness
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "last_launch_env_keys": ["OPENAI_BASE_URL", "PORT"],
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "healthy",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    # A secret added AFTER the launch — visible in pending, absent from injected.
    secrets.set("my-app", {"ADDED_AFTER_LAUNCH": "value-not-leaked"})
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["injected_env_keys_source"] == "launch"
    assert "ADDED_AFTER_LAUNCH" not in body["injected_env_keys"]
    assert "ADDED_AFTER_LAUNCH" in body["pending_env_keys"]


async def test_serialized_body_never_leaks_a_secret_value(harness):
    client, queries, secrets = harness
    leak = "sk-super-secret-value-abc123"
    secrets.set("my-app", {"OPENAI_KEY": leak})
    cfg = {
        "image": "nerdit-app/my-app:1",
        "port": 8000,
        "ai": {
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": "${secrets.OPENAI_KEY}",
            }
        },
        "last_launch_env_keys": ["OPENAI_KEY", "PORT"],
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "healthy",
            "started_at": _STARTED,
        },
    }
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    # The whole serialized body must carry names only — never the value.
    assert leak not in resp.text
    assert "OPENAI_KEY" in resp.json()["injected_env_keys"]


async def test_log_tail_is_clamped_to_200(harness):
    client, queries, _ = harness
    cfg = {"image": "nerdit-app/my-app:1", "port": 8000}
    job = await _seed_service(queries, config=cfg)
    for i in range(250):
        await queries.append_log(job.id, f"line {i}", LogStream.stdout)
    resp = await client.get(
        "/api/services/my-app/diagnose", params={"log_tail": 1000}, headers=_auth(OWNER_RAW)
    )
    assert resp.status_code == 200, resp.text
    logs = resp.json()["logs"]
    assert len(logs) == 200
    assert logs[-1]["line"] == "line 249"


async def test_non_owner_submitter_is_forbidden(harness):
    client, queries, _ = harness
    cfg = {"image": "nerdit-app/my-app:1", "port": 8000}
    await _seed_service(queries, config=cfg, owner="tok-owner")
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OTHER_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


async def test_admin_may_diagnose_any_service(harness):
    client, queries, _ = harness
    cfg = {"image": "nerdit-app/my-app:1", "port": 8000}
    await _seed_service(queries, config=cfg, owner="tok-owner")
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200, resp.text


async def test_unknown_service_is_404(harness):
    client, _, _ = harness
    resp = await client.get("/api/services/nope/diagnose", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def test_model_row_lists_backend_keys_only(harness):
    client, queries, _ = harness
    cfg = {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "healthy",
            "started_at": _STARTED,
        },
    }
    await _seed_service(
        queries,
        config=cfg,
        service_name="ollama-llama3-1-8b",
        name="ollama-llama3-1-8b",
        kind=JobKind.model,
        status=JobStatus.running,
    )
    resp = await client.get("/api/services/ollama-llama3-1-8b/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Models discard the service env / [ai.*] assembly — no OPENAI_* surface.
    assert "OPENAI_BASE_URL" not in body["pending_env_keys"]
    assert body["bindings"]["waiting"] is False


# --------------------------------------------------------------------------- #
# CLI renderer — Rich markup crash class (P6 lesson)                           #
# --------------------------------------------------------------------------- #


def test_cli_renders_bracketed_log_lines_without_markup_crash(monkeypatch):
    """The ``nerdit diagnose`` renderer must not crash on bracketed server text.

    Log lines, error messages, binding waits and remediation detail routinely
    carry ``[INFO]`` / closing-tag shapes like ``done [/uvicorn]`` — passed raw
    to ``console.print`` these raise ``rich.errors.MarkupError`` (or silently
    swallow the token). Every server-derived string must be escaped.
    """
    import asyncio

    from nerdit.cli.commands import diagnose as diag_mod

    payload = {
        "service_name": "my-app [prod]",
        "status": "failed [/oops]",
        "kind": "service",
        "last_deploy": {"version": 4, "action": "redeploy", "phase": "failed"},
        "error": {"class": "OOM", "message": "killed [/container] at 512m [limit]"},
        "forensics": {
            "last_exit_code": 137,
            "oom_killed": True,
            "last_crash_at": "2026-07-10T03:12:00+00:00",
        },
        "restarts": {"count": 3, "max_restarts": 3, "next_retry_in_s": None},
        "bindings": {
            "waiting": True,
            "messages": ["binding 'default': model '[llama3.1:8b]' is not served [/run it]"],
        },
        "remediation": {"code": "inspect_logs", "detail": "see logs [/tail] for [details]"},
        "logs": [
            {"stream": "stdout", "line": "INFO [/uvicorn] startup complete"},
            {"stream": "stderr", "line": "[ERROR] boom ]["},
        ],
    }

    class _FakeClient:
        async def diagnose_service(self, name, log_tail):
            return payload

    monkeypatch.setattr(diag_mod, "get_configured_client", lambda: _FakeClient(), raising=False)
    # Import path used inside the function is ``nerdit.cli.client`` — patch there too.
    import nerdit.cli.client as client_mod

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _FakeClient())

    # Must complete without raising rich.errors.MarkupError.
    asyncio.run(diag_mod._diagnose_async("my-app", 50))


# --------------------------------------------------------------------------- #
# CLI renderer — the P20 last_run block                                        #
# --------------------------------------------------------------------------- #

# A run's output is never written to job_logs (D-P14-6), so `config['last_run']`
# — surfaced by this verb — is its only human surface once the `nerdit services
# run` terminal is gone. Both that verb's help and the MCP `run_command` tool
# name `nerdit diagnose` as the place to recover it.

_NO_RUN_PAYLOAD = {
    "service_name": "api",
    "status": "running",
    "kind": "service",
    "last_deploy": {"version": 4, "action": "redeploy", "phase": "healthy"},
    "error": {"class": None, "message": None},
    "forensics": {},
    "restarts": {"count": 0, "max_restarts": 3, "next_retry_in_s": None},
    "bindings": {"waiting": False, "messages": []},
    "remediation": {"code": "none", "detail": "Nothing to do."},
    "logs": [{"stream": "stdout", "line": "INFO [uvicorn] started"}],
}

# Captured from the renderer BEFORE the last_run block existed: a service that
# has never had a run must still print exactly this, byte for byte.
_NO_RUN_OUTPUT = (
    "api  status: running  kind: service\n"
    "  deploy: v4 redeploy → phase healthy\n"
    "  restarts: 0/3  next retry in: -s\n"
    "\n"
    "remediation: none\n"
    "  Nothing to do.\n"
    "\n"
    "--- log tail ---\n"
    "  INFO [uvicorn] started\n"
)


def _diagnose_cli(monkeypatch, payload: dict):
    """Drive the real verb through CliRunner with the client stubbed."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import typer
    from typer.testing import CliRunner

    from nerdit.cli.commands.diagnose import diagnose

    app = typer.Typer()
    app.command()(diagnose)
    stub = SimpleNamespace(diagnose_service=AsyncMock(return_value=payload))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    return CliRunner().invoke(app, ["api"])


def test_cli_without_last_run_renders_exactly_as_before(monkeypatch):
    """No run has ever executed → the pre-P20 output, unchanged."""
    res = _diagnose_cli(monkeypatch, _NO_RUN_PAYLOAD)
    assert res.exit_code == 0, res.output
    assert res.output == _NO_RUN_OUTPUT


def test_cli_renders_last_run_with_bracketed_output_verbatim(monkeypatch):
    """The run block renders every field, and bracketed stdout survives intact.

    A run's log tail is raw container output: `INFO [uvicorn] started` and
    `[release] x` are exactly the shapes that raise ``rich.errors.MarkupError``
    (or get silently swallowed) when printed unescaped — the P6 lesson.
    """
    payload = dict(
        _NO_RUN_PAYLOAD,
        last_run={
            "run_id": "abc123def456",
            "command": ["alembic", "upgrade", "head"],
            "exit_code": 1,
            "timed_out": False,
            "oom_killed": False,
            "started_at": "2026-08-04T10:00:00+00:00",
            "finished_at": "2026-08-04T10:00:12+00:00",
            "duration_s": 12.3,
            "env_override_keys": ["APP_TOKEN"],
            "log_tail": ["INFO [uvicorn] started", "[release] x", "boom ]["],
        },
    )
    res = _diagnose_cli(monkeypatch, payload)
    assert res.exit_code == 0, res.output
    # Pre-existing blocks are untouched; the run block is appended.
    assert res.output.startswith(_NO_RUN_OUTPUT)
    assert "--- last run ---" in res.output
    # argv as a vector, not a shell line (D-P20-5: it was never shell-parsed).
    assert '["alembic", "upgrade", "head"]' in res.output
    assert "exit 1" in res.output
    assert "timed_out=False" in res.output
    assert "oom=False" in res.output
    assert "in 12.3s" in res.output
    assert "2026-08-04T10:00:12+00:00" in res.output
    # Bracketed stdout renders verbatim — no MarkupError, nothing swallowed.
    assert "INFO [uvicorn] started" in res.output
    assert "[release] x" in res.output
    assert "boom ][" in res.output


def test_cli_diagnose_renders_health_observations(monkeypatch):
    """(Codex 3804646869) The CLI must not discard ``health.observations``.

    The route populates it (an implicit ``/`` probe answering 404 is the case
    this branch added), and ``nerdit diagnose`` is the surface users are pointed
    at after a deploy — a renderer that never reads ``health`` silently drops
    the only actionable advice on that path. Server-derived text, so it goes
    through ``_plain`` like every other such string (the P6 markup lesson).
    """
    payload = dict(
        _NO_RUN_PAYLOAD,
        health={
            "spec": None,
            "probe": None,
            "observations": [
                "no [deploy].health is set and the implicit / probe answered 404 [INFO]"
            ],
        },
    )
    res = _diagnose_cli(monkeypatch, payload)
    assert res.exit_code == 0, res.output
    assert "health observations:" in res.output
    # Escaped, not swallowed: the bracketed token survives literally.
    assert "the implicit / probe answered 404 [INFO]" in res.output


def test_cli_diagnose_without_health_observations_renders_exactly_as_before(monkeypatch):
    """Leg B: an empty/absent observations list adds no block at all."""
    res = _diagnose_cli(monkeypatch, dict(_NO_RUN_PAYLOAD, health={"observations": []}))
    assert res.exit_code == 0, res.output
    assert "health observations" not in res.output
    assert res.output == _NO_RUN_OUTPUT


def test_cli_last_run_with_non_list_log_tail_does_not_explode(monkeypatch):
    """``last_run`` is an untyped passthrough — a stray scalar must not be looped."""
    payload = dict(
        _NO_RUN_PAYLOAD,
        last_run={"command": "alembic upgrade head", "exit_code": 0, "log_tail": "oops"},
    )
    res = _diagnose_cli(monkeypatch, payload)
    assert res.exit_code == 0, res.output
    assert "--- last run ---" in res.output
    assert "command: alembic upgrade head" in res.output
    assert "\n    o\n" not in res.output  # not iterated character by character


# --------------------------------------------------------------------------- #
# (P21 D2) gpu_oom → model.gpu_oom                                             #
# --------------------------------------------------------------------------- #


def test_unit_gpu_oom_model_maps_model_gpu_oom():
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "gpu_oom": True,
        "last_crash_at": _CRASH_FRESH,
    }
    job = _job(kind=JobKind.model, service_name="ollama-qwen", name="ollama-qwen")
    code, detail = derive_remediation(job, cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.model_gpu_oom
    assert code.value == "model.gpu_oom"
    assert "--max-model-len" in detail
    assert "gpu_memory_utilization" in detail
    assert "vllm_extra_args" in detail
    # The wrong fix must be named as wrong, never offered.
    assert "memory_limit" in detail and "NOT fix" in detail


def test_unit_gpu_oom_stale_not_fired():
    # A gpu_oom from a PREVIOUS deploy generation must not outrank the fresh
    # build failure (the same freshness gate as Rule 1).
    cfg = {
        "last_deploy": _last_deploy(reason="build_failed"),
        "previous_image": "nerdit-app/my-app:3",
    }
    forensics = {"gpu_oom": True, "last_crash_at": _CRASH_STALE}
    job = _job(kind=JobKind.model, status=JobStatus.running)
    code, _ = derive_remediation(job, cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.rollback


def test_unit_gpu_oom_outranks_cgroup_oom_on_model_row():
    # Both flags set on a model row → the GPU OOM wins (architect ruling
    # 2026-08-06): the allocator failure is the root cause, and this keeps
    # error.class GPU_OOM ↔ model.gpu_oom one-to-one with
    # _classify_service_exit and the troubleshooting table.
    cfg = {"last_deploy": _last_deploy(reason="crash_loop"), "memory_limit": "512m"}
    forensics = {
        "last_exit_code": 137,
        "oom_killed": True,
        "gpu_oom": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(kind=JobKind.model), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.model_gpu_oom


def test_unit_both_oom_flags_on_service_row_stay_cgroup():
    # The same both-flags forensics on a kind=service row: the model rule is
    # scoped out (D2), so the cgroup rule stays authoritative.
    cfg = {"last_deploy": _last_deploy(reason="crash_loop"), "memory_limit": "512m"}
    forensics = {
        "last_exit_code": 137,
        "oom_killed": True,
        "gpu_oom": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(kind=JobKind.service), cfg, forensics, BindingWait(), None)
    assert code is RemediationCode.raise_memory_limit


def test_unit_gpu_oom_on_service_row_does_not_fire_model_rule():
    # The mapping is scoped to model rows (locked); a service row falls through.
    cfg = {"last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {
        "last_exit_code": 1,
        "oom_killed": False,
        "gpu_oom": True,
        "last_crash_at": _CRASH_FRESH,
    }
    code, _ = derive_remediation(_job(kind=JobKind.service), cfg, forensics, BindingWait(), None)
    assert code is not RemediationCode.model_gpu_oom


async def test_gpu_oom_model_row_diagnose_payload(harness):
    client, queries, _ = harness
    cfg = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend": "vllm",
        "port": 8000,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": "failed",
            "reason": "crash_loop",
            "started_at": _STARTED,
        },
        "last_exit_code": 1,
        "oom_killed": False,
        "gpu_oom": True,
        "last_crash_at": _CRASH_FRESH,
    }
    job = await _seed_service(
        queries,
        config=cfg,
        kind=JobKind.model,
        error_class="GPU_OOM",
        error_message="Restart budget exhausted (4 restarts in 300s); last exit_code=1",
    )
    await queries.replace_crash_tail(
        job.id, ["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"]
    )

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["forensics"]["gpu_oom"] is True
    assert body["forensics"]["oom_killed"] is False
    assert body["error"]["class"] == "GPU_OOM"
    assert body["remediation"]["code"] == "model.gpu_oom"
    # The captured tail is what lights up the previously-empty log_tail (D1).
    crash_lines = [entry for entry in body["logs"] if entry["stream"] == "crash"]
    assert crash_lines and "CUDA out of memory" in crash_lines[0]["line"]


async def test_diagnose_projects_last_dump_for_a_database_row(harness):
    """(P37 D-P37-11) ``last_dump`` is the tail's ONE surface outside the 500 hint.

    A dump or restore writes nothing to ``job_logs`` (like a run), and the route
    that produced it returns only the tail's LAST line as a hint — so this
    owner-gated projection is where an operator reads what the tool actually
    said. The audit row and the durable event carry neither, deliberately: a
    ``pg_restore`` diagnostic prints row values.
    """
    client, queries, _ = harness
    last_dump = {
        "run_id": "abc123def456",
        "kind": "restore",
        "started_at": _STARTED,
        "finished_at": _STARTED,
        # Secret-free by construction (D-P37-4: the password travels in the
        # environment), which is why the argv is safe to keep verbatim.
        "command": ["pg_restore", "-h", "172.17.0.1", "-U", "nerdit", "-d", "nerdit"],
        "exit_code": 1,
        "timed_out": False,
        "reason": "exit_nonzero",
        "dump": "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz",
        "log_tail": ["pg_restore: error: could not execute query", "DETAIL:  Key (id)=(1)"],
    }
    await _seed_service(
        queries,
        config={"backend": "postgres", "image": "postgres:16", "port": 5432, "db_ready": True},
        kind=JobKind.database,
        service_name="pg",
        name="pg",
        status=JobStatus.running,
    )
    job = await queries.get_service_by_name("pg")
    cfg = json.loads(job.config)
    cfg["last_dump"] = last_dump
    await queries.set_last_dump(job.id, json.dumps(last_dump))

    resp = await client.get("/api/services/pg/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["last_dump"] == last_dump
    # ``last_run`` stays independent — a database that has never had a one-off
    # run must not borrow the dump's record.
    assert body["last_run"] is None


async def test_diagnose_last_dump_is_null_on_a_service_row(harness):
    """Only a managed database ever stamps the key; every other row projects null."""
    client, queries, _ = harness
    await _seed_service(queries, config={"image": "nerdit-app/my-app:1", "port": 8000})
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200
    assert resp.json()["last_dump"] is None


async def test_captured_tail_never_reaches_the_audit_log(harness):
    """Negative contract (plan §2.3): a crash tail lives in job_logs only —
    no audit row ever carries its content."""
    client, queries, _ = harness
    job = await _seed_service(queries, config={"image": "nerdit-app/my-app:1", "port": 8000})
    marker = "torch.OutOfMemoryError: CUDA out of memory"
    await queries.replace_crash_tail(job.id, [marker])
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text

    rows, _ = await queries.list_audit_log(limit=100)
    assert all(marker not in json.dumps(r.model_dump(), default=str) for r in rows)


# --- P25 WP4k: [deploy].edge_auth classification -------------------------------
#
# ``derive_remediation`` is pure by contract, so the ROUTE computes the edge-auth
# outcome (grammar parse + a read-only ref resolve) and hands it in. These legs
# pin both halves: the precedence slot (rules 3b/3c, adjacent to
# ``set_missing_secret``) and the value-free contract — the detail names a secret
# KEY or the offending FIELD names, never anything the blob or the store holds.

_EDGE_AUTH_CFG = {
    "image": "nerdit-app/my-app:1",
    "port": 8000,
    "last_deploy": {
        "version": 1,
        "action": "create",
        "phase": "healthy",
        "started_at": _STARTED,
    },
}


async def test_declared_edge_auth_with_an_unset_secret_names_the_key_not_a_value(harness):
    """The D-P25-8 fail-closed case seen from the operator's side: the container
    is RUNNING and healthy, and only the route is withheld — so the diagnose
    answer is the only place that explains why the URL 404s."""
    client, queries, _ = harness
    cfg = {**_EDGE_AUTH_CFG, "edge_auth": {"user": "ops", "password": "${secrets.APP_PW}"}}
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "edge_auth.secret_missing"
    assert "APP_PW" in rem["detail"]
    assert "nerdit secrets set my-app APP_PW=" in rem["detail"]
    # An edge-auth wait is NOT a binding wait: the launch is not blocked.
    assert resp.json()["bindings"]["waiting"] is False


async def test_edge_auth_secret_present_falls_through_to_the_existing_codes(harness):
    """Positive control for the rule above AND the precedence pin: once the
    reference resolves, the classifier is byte-for-byte what it was pre-P25."""
    client, queries, secrets = harness
    secrets.set("my-app", {"APP_PW": "s3cr3t-never-printed"})
    cfg = {**_EDGE_AUTH_CFG, "edge_auth": {"user": "ops", "password": "${secrets.APP_PW}"}}
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "none"
    assert "s3cr3t-never-printed" not in resp.text


async def test_edge_auth_over_length_secret_reports_invalid_with_the_secret_fix(harness):
    """Review-upheld P25 finding: a resolved secret over bcrypt's 72-byte input
    limit makes ``hash_password`` refuse on every route build (fail-closed, the
    route is withheld) — diagnose must mirror that refusal via the SAME shared
    predicate, and the detail must name the real fix (a shorter secret), never
    the declaration."""
    client, queries, secrets = harness
    long_pw = "hunter2-CANARY-0000" + "x" * 72
    secrets.set("my-app", {"APP_PW": long_pw})
    cfg = {**_EDGE_AUTH_CFG, "edge_auth": {"user": "ops", "password": "${secrets.APP_PW}"}}
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "edge_auth.invalid"
    assert "72-byte" in rem["detail"]
    assert "nerdit secrets set my-app APP_PW=" in rem["detail"]
    # The fix is the SECRET, so the declaration-repair path must not appear.
    assert "edge_auth=" not in rem["detail"]
    # The over-length value is still a credential: it never rides the payload.
    assert long_pw not in resp.text
    assert "hunter2-CANARY-0000" not in resp.text


async def test_edge_auth_shared_ref_points_at_the_shared_scope(harness):
    client, queries, _ = harness
    cfg = {
        **_EDGE_AUTH_CFG,
        "edge_auth": {"user": "ops", "password": "${secrets.shared.EDGE_PW}"},
    }
    await _seed_service(queries, config=cfg, status=JobStatus.running)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "edge_auth.secret_missing"
    assert "nerdit secrets set --shared EDGE_PW=" in rem["detail"]


@pytest.mark.parametrize(
    "blob, expected_fields",
    [
        ({"user": "", "password": "${secrets.APP_PW}"}, ["user"]),
        ({"user": "ops"}, ["password"]),
        ({"user": "ops", "password": "hunter2-LITERAL"}, ["password"]),
        ({"user": "o:ps", "password": "nope"}, ["user", "password"]),
        ("not-a-table", ["edge_auth"]),
    ],
)
async def test_malformed_edge_auth_names_fields_only(harness, blob, expected_fields):
    """D-P25-8 tri-state, diagnose side: a declared-but-broken blob is its own
    outcome (never "no auth"), and the detail carries FIELD names only — a
    hand-forged literal password is one of the malformed cases, so echoing the
    blob would leak exactly the credential the grammar re-check exists to
    reject."""
    client, queries, _ = harness
    await _seed_service(
        queries, config={**_EDGE_AUTH_CFG, "edge_auth": blob}, status=JobStatus.running
    )
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "edge_auth.invalid"
    for field_name in expected_fields:
        assert field_name in rem["detail"]
    assert "nerdit config app set my-app deploy edge_auth=" in rem["detail"]
    # Nothing of the blob's values travels — not through any envelope field.
    assert "hunter2-LITERAL" not in resp.text
    assert "not-a-table" not in resp.text


async def test_no_edge_auth_leaves_the_existing_codes_untouched(harness):
    """Regression guard for every pre-P25 row: an absent blob classifies as
    ``None`` and the ladder behaves exactly as before."""
    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:4",
        "build_version": 4,
        "port": 8000,
        "memory_limit": "512m",
        "last_deploy": {
            "version": 4,
            "action": "redeploy",
            "phase": "launching",
            "started_at": _STARTED,
        },
        "last_exit_code": 137,
        "oom_killed": True,
        "last_crash_at": _CRASH_FRESH,
    }
    await _seed_service(queries, config=cfg)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "raise_memory_limit"


def test_edge_auth_rules_slot_below_the_binding_waits():
    """The precedence pin (pure, no HTTP): rules 2/3 outrank 3b/3c.

    A blocked ``[ai.*]`` binding means the container never STARTS; a broken
    edge_auth means it runs and only its route is withheld. The launch blocker is
    the more urgent answer, so the two new rules slot immediately AFTER
    ``set_missing_secret`` — and never before it."""
    from nerdit.daemon.remediation import EdgeAuthWait

    job = _job(status=JobStatus.running)
    cfg = {"last_deploy": _last_deploy(phase="launching")}
    edge = EdgeAuthWait(missing_key="APP_PW")

    model_wait = BindingWait(waiting=True, model_wait=True, messages=("serve it",))
    code, _ = derive_remediation(job, cfg, {}, model_wait, None, edge)
    assert code is RemediationCode.serve_missing_model

    secret_wait = BindingWait(waiting=True, secret_wait=True, messages=("set it",))
    code, _ = derive_remediation(job, cfg, {}, secret_wait, None, edge)
    assert code is RemediationCode.set_missing_secret

    # ...and with no binding wait in play, the edge-auth rule is what fires —
    # ahead of rule 4 (rollback), whose preconditions are satisfied here too.
    code, _ = derive_remediation(
        job,
        {"last_deploy": _last_deploy(reason="build_failed"), "previous_image": "nerdit-app/x:1"},
        {},
        BindingWait(),
        None,
        edge,
    )
    assert code is RemediationCode.edge_auth_secret_missing


def test_derive_remediation_without_an_edge_auth_input_is_unchanged():
    """The parameter is optional and defaults to "not classified", so every
    existing caller (and every pre-P25 test) keeps its exact behaviour."""
    job = _job(status=JobStatus.degraded)
    code, _ = derive_remediation(job, {}, {}, BindingWait(), None)
    assert code is RemediationCode.fix_health_check


def test_invalid_and_missing_are_mutually_exclusive_with_invalid_first():
    """Belt-and-braces: a blob that fails the grammar has no reference to
    resolve, but if a caller ever hands in both, the declaration repair is the
    prerequisite fix and must win."""
    from nerdit.daemon.remediation import EdgeAuthWait

    both = EdgeAuthWait(invalid_fields=("password",), missing_key="APP_PW")
    job = _job(status=JobStatus.running)
    code, detail = derive_remediation(job, {}, {}, BindingWait(), None, both)
    assert code is RemediationCode.edge_auth_invalid
    assert "APP_PW" not in detail


async def test_undecryptable_scope_classifies_as_edge_auth_secret_missing(harness, monkeypatch):
    """An undecryptable store is the same OUTCOME as an unset key: the route
    plane cannot materialize the credential either, so the route stays withheld.
    The KEY name is still known — it comes from the reference's own grammar, not
    from the store — so the detail stays actionable."""
    from nerdit.core.secrets import SecretDecryptError

    client, queries, secrets = harness
    cfg = {**_EDGE_AUTH_CFG, "edge_auth": {"user": "ops", "password": "${secrets.APP_PW}"}}
    await _seed_service(queries, config=cfg, status=JobStatus.running)

    def _boom(_service):
        raise SecretDecryptError("scope unreadable")

    monkeypatch.setattr(secrets, "load", _boom)
    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "edge_auth.secret_missing"
    assert "APP_PW" in rem["detail"]


# --- Agent-DX: the implicit-probe-404 observation ------------------------------
#
# It lands HERE and not on the deploy result by mechanism, not preference: a
# deploy 201 is written before any container exists (the endpoint row is created
# at launch), so there is nothing to probe at the deploy seam — and an implicit
# root probe only ever runs in ``_fresh_health_probe``, which is a diagnose-only
# code path. Advisory only: a 404 root is legitimate liveness for an API-only
# app, so it must never move ``remediation.code``.


def _obs(job, probe):
    from nerdit.daemon.routes.service_diagnose import _health_observations

    return _health_observations(job, probe)


def test_diagnose_observes_an_implicit_probe_404():
    """Leg A: no health declared + the default '/' probe answered 404."""
    obs = _obs(_job(health_check=None), {"type": "http", "status_code": 404})
    assert len(obs) == 1
    assert "No [deploy].health is declared" in obs[0]
    assert "health = '/path'" in obs[0]


def test_diagnose_has_no_observation_when_health_is_declared():
    """Leg B: the probe hit the app's OWN declared path, so a 404 there is a
    real health failure the remediation classifier already speaks to — not a
    'you never configured this' observation."""
    assert _obs(_job(health_check={"path": "/healthz"}), {"type": "http", "status_code": 404}) == []


def test_diagnose_has_no_observation_when_the_implicit_probe_answers_200():
    """Leg B′: an app that serves its root needs no advice."""
    assert _obs(_job(health_check=None), {"type": "http", "status_code": 200}) == []


def test_diagnose_observation_is_empty_when_there_is_no_container():
    """``probe is None`` means 'no container to probe' — nothing was observed,
    so there is nothing to say. A tcp probe is likewise out of scope: there is
    no status code and no implicit path."""
    assert _obs(_job(health_check=None), None) == []
    assert _obs(_job(health_check=None), {"type": "tcp", "status_code": None}) == []


async def test_implicit_probe_404_does_not_change_the_remediation_code(harness, monkeypatch):
    """Semantics pin: the observation is advisory. The SAME row diagnoses to the
    same ``remediation.code`` with and without the 404 implicit probe — nothing
    in ``derive_remediation`` reads ``observations``, and it must stay that way."""
    from nerdit.daemon.routes import service_diagnose as svc_routes

    client, queries, _ = harness
    cfg = {
        "image": "nerdit-app/my-app:4",
        "build_version": 4,
        "port": 8000,
        "memory_limit": "512m",
        "last_deploy": {
            "version": 4,
            "action": "redeploy",
            "phase": "launching",
            "started_at": _STARTED,
        },
        "last_exit_code": 137,
        "oom_killed": True,
        "last_crash_at": _CRASH_FRESH,
    }
    await _seed_service(queries, config=cfg)

    async def _no_probe(job, endpoint):
        return None

    async def _probe_404(job, endpoint):
        return {"type": "http", "status_code": 404, "checked_at": _CRASH_FRESH}

    monkeypatch.setattr(svc_routes, "_fresh_health_probe", _no_probe)
    baseline = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert baseline.status_code == 200, baseline.text
    assert baseline.json()["health"]["observations"] == []

    monkeypatch.setattr(svc_routes, "_fresh_health_probe", _probe_404)
    observed = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert observed.status_code == 200, observed.text
    body = observed.json()
    assert len(body["health"]["observations"]) == 1
    assert body["remediation"] == baseline.json()["remediation"]
    assert body["remediation"]["code"] == "raise_memory_limit"


# --- P26 WP2: ACME certificate rules 7c/7d -------------------------------------
#
# ``derive_remediation`` is pure, so the ROUTE reads the domain table and the
# proxy's certificate storage and hands the outcome in. These legs pin both
# halves: the precedence slot (below every genuine failure, above
# ``inspect_logs``) and the value-free contract — the detail names public DNS
# the operator chose, never the ACME account email and never the directory URL.

_ACME_EMAIL = "ops@example.test"
_ACME_DIRECTORY = "https://acme-staging-v02.api.letsencrypt.org/directory"

_HEALTHY_CFG = {
    "image": "nerdit-app/my-app:1",
    "port": 8000,
    "last_deploy": {
        "version": 1,
        "action": "create",
        "phase": "healthy",
        "started_at": _STARTED,
    },
}


def test_unit_acme_pending_fires_on_a_perfectly_healthy_service():
    """The whole point of rule 7c: nothing else in the table would fire.

    The container is running and the route is published; only the certificate
    is missing — so without this rule the answer would be ``none``, "no
    remediation needed", on a node whose operator is staring at a failed
    handshake.
    """
    acme = AcmeWait(pending_domains=("a.example.com", "b.example.com"))
    code, detail = derive_remediation(
        _job(status=JobStatus.running),
        dict(_HEALTHY_CFG),
        {},
        BindingWait(),
        None,
        acme=acme,
    )

    assert code is RemediationCode.acme_cert_pending
    assert "2 ACME domain(s)" in detail
    assert "a.example.com" in detail and "b.example.com" in detail
    assert "acme_http_port" in detail
    # (review round 2) The detail used to say those names "are served with this
    # node's internal CA until the CA issues". Measured against real Caddy, they
    # are served with NOTHING: the acme-only automation policy claims the
    # subject, so the internal catch-all behind it is never consulted and the
    # handshake fails. Pinned so the sentence cannot drift back.
    assert "do NOT complete TLS handshakes" in detail
    assert "served with this node's internal CA" not in detail


def test_unit_acme_disabled_fires_and_names_the_two_ways_out():
    """Rule 7d is a *setting*, so its detail must name both fixes.

    Turning ACME on is one; dropping the request (``acme=false``) is the other,
    and an operator who never meant to ask for a public certificate needs the
    second one spelled out or the row stays wrong forever.
    """
    code, detail = derive_remediation(
        _job(status=JobStatus.running),
        dict(_HEALTHY_CFG),
        {},
        BindingWait(),
        None,
        acme=AcmeWait(disabled_domains=("a.example.com",)),
    )

    assert code is RemediationCode.acme_disabled
    assert "1 domain(s)" in detail
    assert "[proxy.acme].enabled is false" in detail
    assert "acme=false" in detail
    # (review round 1) The path has to be one that EXISTS. ``PUT /config/proxy``
    # is a 404 — the only config write route is ``PUT /config/daemon/{section}``
    # — and an agent is the stated consumer of these details, so the one
    # "actionable" ACME code must not send it to a dead URL.
    assert "PUT /config/daemon/proxy" in detail
    assert "PUT /config/proxy " not in detail


def test_unit_acme_rules_never_carry_the_email_or_the_directory():
    """S-W2-7's value-free contract, asserted on both codes at once.

    The email is PII-adjacent and useless as a fix; the directory URL only says
    which CA (staging or production) and nothing an operator does with this
    message needs it. ``@`` and ``://`` are the two shapes either one would
    take, so neither may appear.
    """
    for acme in (
        AcmeWait(pending_domains=("a.example.com",)),
        AcmeWait(disabled_domains=("a.example.com",)),
    ):
        _, detail = derive_remediation(
            _job(status=JobStatus.running), dict(_HEALTHY_CFG), {}, BindingWait(), None, acme=acme
        )
        assert "@" not in detail
        assert "://" not in detail
        assert _ACME_EMAIL not in detail
        assert _ACME_DIRECTORY not in detail
        assert "a.example.com" in detail


def test_unit_a_fresh_crash_outranks_a_pending_certificate():
    """Precedence pin: rule 5 wins.

    A crashed container is the bigger problem and has its own actionable fix; a
    certificate that has not been issued yet is what the operator looks at once
    the app runs. Getting this backwards would answer "check your DNS" to
    someone whose app exits on boot.
    """
    cfg = {**_HEALTHY_CFG, "last_deploy": _last_deploy(reason="crash_loop")}
    forensics = {"last_exit_code": 1, "oom_killed": False, "last_crash_at": _CRASH_FRESH}
    code, detail = derive_remediation(
        _job(status=JobStatus.restarting, restart_count=3),
        cfg,
        forensics,
        BindingWait(),
        None,
        acme=AcmeWait(pending_domains=("a.example.com",)),
    )

    assert code is RemediationCode.fix_start_command
    assert "a.example.com" not in detail


def test_unit_acme_pending_outranks_a_disabled_row_and_inspect_logs():
    """7c before 7d, and both before rule 8.

    The ordering between the two is a tiebreak that cannot actually occur
    (``cert_status`` answers ``disabled`` only while the setting is off, which
    is the same condition that fills the other tuple) — pinned so the
    fall-through stays deterministic if that ever changes. Against rule 8 the
    ranking is load-bearing: ``inspect_logs`` would send an operator to a log
    tail that says nothing about the CA.
    """
    both = AcmeWait(pending_domains=("a.example.com",), disabled_domains=("b.example.com",))
    code, _ = derive_remediation(
        _job(status=JobStatus.running), dict(_HEALTHY_CFG), {}, BindingWait(), None, acme=both
    )
    assert code is RemediationCode.acme_cert_pending

    failed = {**_HEALTHY_CFG, "last_deploy": _last_deploy(phase="failed", reason="unknown")}
    code, _ = derive_remediation(
        _job(status=JobStatus.failed),
        failed,
        {},
        BindingWait(),
        None,
        acme=AcmeWait(disabled_domains=("b.example.com",)),
    )
    assert code is RemediationCode.acme_disabled


def test_unit_derive_remediation_without_an_acme_input_is_unchanged():
    """``acme`` defaults to "not classified", so pre-WP2 callers are untouched.

    It is keyword-only for the same reason: the positional tail was already
    load-bearing in a dozen call sites, and a fourth optional positional would
    let a misordered call type-check silently.
    """
    code, _ = derive_remediation(
        _job(status=JobStatus.running), dict(_HEALTHY_CFG), {}, BindingWait(), None
    )
    assert code is RemediationCode.none

    with pytest.raises(TypeError):
        derive_remediation(  # type: ignore[misc]
            _job(), {}, {}, BindingWait(), None, None, AcmeWait()
        )


def test_unit_an_empty_acme_wait_changes_nothing():
    """A service with no ACME domains (the overwhelming majority) falls through."""
    code, _ = derive_remediation(
        _job(status=JobStatus.running),
        dict(_HEALTHY_CFG),
        {},
        BindingWait(),
        None,
        acme=AcmeWait(),
    )
    assert code is RemediationCode.none


class _StubCertManager:
    """A ``ProxyManager`` stand-in exposing only ``cert_status`` (S-W2-6).

    The route reaches it through ``getattr``, exactly as ``views/hosted.py``
    does, so a stub carrying one method is enough — and this test file must not
    depend on the real manager landing.
    """

    def __init__(self, states: dict[str, str]) -> None:
        self.states = states
        self.seen: list[str] = []

    def cert_status(self, row):  # noqa: ANN001 - a ServiceDomain row
        self.seen.append(row.domain)
        return SimpleNamespace(state=self.states.get(row.domain, "pending"), not_after=None)


async def _seed_domain(queries, job, domain: str, *, acme: bool) -> None:
    outcome, _ = await queries.add_service_domain(
        job.service_name, domain, acme=acme, job_id=job.id
    )
    assert outcome == "inserted"


async def test_diagnose_reports_a_pending_certificate_on_a_healthy_app(app_harness):
    """Route wiring end to end: table + ``cert_status`` → rule 7c.

    The service is RUNNING with a healthy last deploy, so this answer exists
    only because the route classified the certificate plane.
    """
    client, queries, app = app_harness
    job = await _seed_service(queries, config=dict(_HEALTHY_CFG), status=JobStatus.running)
    await _seed_domain(queries, job, "a.example.com", acme=True)
    app.state.proxy_manager = _StubCertManager({"a.example.com": "pending"})

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    rem = resp.json()["remediation"]
    assert rem["code"] == "acme.cert_pending"
    assert "a.example.com" in rem["detail"]
    assert _ACME_EMAIL not in resp.text


async def test_diagnose_counts_an_expired_leaf_as_pending(app_harness):
    """An expired leaf is a renewal that is not happening.

    Same operator problem, same causes (DNS, reachability), same fix — so it
    rides rule 7c rather than earning a third code nobody would act on
    differently.
    """
    client, queries, app = app_harness
    job = await _seed_service(queries, config=dict(_HEALTHY_CFG), status=JobStatus.running)
    await _seed_domain(queries, job, "a.example.com", acme=True)
    app.state.proxy_manager = _StubCertManager({"a.example.com": "expired"})

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.json()["remediation"]["code"] == "acme.cert_pending"


async def test_diagnose_reports_a_disabled_acme_row(app_harness):
    """``acme=1`` on a node whose ``[proxy.acme]`` is off ⇒ rule 7d."""
    client, queries, app = app_harness
    job = await _seed_service(queries, config=dict(_HEALTHY_CFG), status=JobStatus.running)
    await _seed_domain(queries, job, "a.example.com", acme=True)
    app.state.proxy_manager = _StubCertManager({"a.example.com": "disabled"})

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    rem = resp.json()["remediation"]
    assert rem["code"] == "acme.disabled"
    assert "a.example.com" in rem["detail"]


async def test_diagnose_ignores_issued_and_non_acme_domains(app_harness):
    """Positive control. An ``acme=0`` row never touches the disk at all.

    It is served with the internal CA on purpose — the default, not a defect —
    so it must not be classified, and ``cert_status`` must not even be called
    for it.
    """
    client, queries, app = app_harness
    job = await _seed_service(queries, config=dict(_HEALTHY_CFG), status=JobStatus.running)
    await _seed_domain(queries, job, "a.example.com", acme=True)
    await _seed_domain(queries, job, "plain.example.com", acme=False)
    manager = _StubCertManager({"a.example.com": "issued"})
    app.state.proxy_manager = manager

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.json()["remediation"]["code"] == "none"
    assert manager.seen == ["a.example.com"]


async def test_diagnose_without_a_proxy_manager_keeps_the_pre_wp2_answer(app_harness):
    """A daemon with the proxy off must degrade to the old answer, never 500.

    Both hops are ``getattr``-guarded; this is the one that is absent on every
    node running without an embedded proxy.
    """
    client, queries, _ = app_harness
    job = await _seed_service(queries, config=dict(_HEALTHY_CFG), status=JobStatus.running)
    await _seed_domain(queries, job, "a.example.com", acme=True)

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["remediation"]["code"] == "none"


async def test_diagnose_a_fresh_crash_still_outranks_the_certificate_at_the_route(app_harness):
    """The precedence pin, asserted through the assembled payload too."""
    client, queries, app = app_harness
    cfg = {**_HEALTHY_CFG, "last_deploy": _last_deploy(reason="crash_loop")}
    cfg |= {"last_exit_code": 1, "oom_killed": False, "last_crash_at": _CRASH_FRESH}
    job = await _seed_service(queries, config=cfg, status=JobStatus.restarting, restart_count=3)
    await _seed_domain(queries, job, "a.example.com", acme=True)
    app.state.proxy_manager = _StubCertManager({"a.example.com": "pending"})

    resp = await client.get("/api/services/my-app/diagnose", headers=_auth(OWNER_RAW))
    assert resp.json()["remediation"]["code"] == "fix_start_command"


# --- P34: the failed wait folds the diagnosis in -------------------------------
#
# The field failure: an agent that called ``wait_for_service`` and got
# ``outcome: failed`` had, at that instant, exactly the question ``/diagnose``
# answers — and only an agent who already knew the tool existed ever asked it.
# The bundle now rides the failure. The owner gate is the whole security
# argument: ``/wait`` stays any-authenticated, so a caller who may not read a
# log tail simply gets ``diagnosis: null``.

_P34_FAILED_CFG = {
    "image": "nerdit-app/my-app:2",
    "build_version": 2,
    "port": 8000,
    "last_deploy": {
        "version": 2,
        "action": "create",
        "phase": "failed",
        "reason": "crash_loop",
        "started_at": _STARTED,
    },
    "last_exit_code": 1,
    "oom_killed": False,
    "priv_denied": True,
    "last_crash_at": _CRASH_FRESH,
}


async def test_a_failed_wait_carries_the_full_diagnosis(harness):
    client, queries, _ = harness
    await _seed_service(queries, config=_P34_FAILED_CFG)

    resp = await client.get("/api/services/my-app/wait?timeout=1", headers=_auth(OWNER_RAW))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outcome"] == "failed"
    diagnosis = body["diagnosis"]
    assert diagnosis is not None
    # The three things the agent needed and had to make a second call for.
    assert diagnosis["remediation"]["code"] == "image_needs_privileges"
    assert diagnosis["remediation"]["detail"]
    assert isinstance(diagnosis["logs"], list)
    # ...and it describes the SAME service the wait resolved.
    assert diagnosis["service_name"] == body["service_name"]


async def test_the_inline_diagnosis_matches_the_diagnose_route(harness):
    """One implementation, delegated — never a second copy to drift."""
    client, queries, _ = harness
    await _seed_service(queries, config=_P34_FAILED_CFG)

    waited = await client.get("/api/services/my-app/wait?timeout=1", headers=_auth(OWNER_RAW))
    direct = await client.get("/api/services/my-app/diagnose?log_tail=25", headers=_auth(OWNER_RAW))

    assert waited.json()["diagnosis"] == direct.json()


async def test_a_non_owner_gets_the_failure_without_the_owner_gated_bundle(harness):
    """``/wait`` is any-authenticated and ``/diagnose`` is owner-or-admin; folding
    one into the other must not widen the second's audience by a single line."""
    client, queries, _ = harness
    await _seed_service(queries, config=_P34_FAILED_CFG)

    resp = await client.get("/api/services/my-app/wait?timeout=1", headers=_auth(OTHER_RAW))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The failure itself is NOT secret — only the bundle is.
    assert body["outcome"] == "failed"
    assert body["diagnosis"] is None


async def test_a_converged_wait_carries_no_diagnosis(harness):
    """Nothing to diagnose, and no extra reads paid for on the happy path."""
    client, queries, _ = harness
    await _seed_service(
        queries,
        config={"image": "nerdit-app/my-app:1", "build_version": 1, "port": 8000},
        status=JobStatus.running,
    )

    resp = await client.get("/api/services/my-app/wait?timeout=1", headers=_auth(OWNER_RAW))

    body = resp.json()
    assert body["outcome"] == "converged"
    assert body["diagnosis"] is None


# --- (P33 D-GH-10) settle-time subset <-> LOCKED classifier consistency ------
#
# ``settle_remediation_code`` lives in ``core`` (``deploy_state`` is its caller
# and ``core`` never imports ``daemon``), so it emits string literals rather than
# ``RemediationCode`` members. These pins keep the two halves from drifting.

_FRESH_CRASH = "2030-01-01T00:00:01+00:00"
_STALE_CRASH = "2029-01-01T00:00:00+00:00"


def test_unit_settle_literals_are_locked_remediation_codes():
    from nerdit.daemon import remediation

    for literal in remediation._SETTLE_LITERALS:
        assert RemediationCode(literal).value == literal


@pytest.mark.parametrize(
    ("forensics", "reason", "expected"),
    [
        (
            {"last_exit_code": 1, "oom_killed": True, "last_crash_at": _FRESH_CRASH},
            "crash_loop",
            "raise_memory_limit",
        ),
        (
            {"last_exit_code": 1, "priv_denied": True, "last_crash_at": _FRESH_CRASH},
            "crash_loop",
            "image_needs_privileges",
        ),
        ({"last_exit_code": 1, "last_crash_at": _FRESH_CRASH}, "crash_loop", "fix_start_command"),
        ({"last_exit_code": 137, "last_crash_at": _FRESH_CRASH}, "crash_loop", None),
        ({"last_exit_code": 1, "last_crash_at": _STALE_CRASH}, "crash_loop", None),
        ({"last_exit_code": 1, "last_crash_at": _FRESH_CRASH}, "build_failed", None),
        ({}, "build_failed", None),
    ],
)
def test_unit_settle_code_matches_the_forensics_rules(forensics, reason, expected):
    """The core settle subset ranks exactly like rules 1b / 4b / 5 of the daemon table."""
    from nerdit.daemon.remediation import (
        _crash_loop_rules,
        _forensics_fresh,
        settle_remediation_code,
    )

    last_deploy = {"phase": "failed", "reason": reason, "started_at": "2030-01-01T00:00:00+00:00"}
    assert settle_remediation_code(dict(forensics), last_deploy) == expected

    if expected not in (None, "raise_memory_limit"):
        fresh = _forensics_fresh(forensics, last_deploy)
        verdict = _crash_loop_rules("x", forensics, fresh=fresh, crash_loop=reason == "crash_loop")
        assert verdict is not None
        assert verdict[0].value == expected
