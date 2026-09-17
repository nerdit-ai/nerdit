"""Route-level tests for the ``/services`` surface (P2 / S6).

Copies the ``test_routes_authz`` harness: the real services router under the
``ScopedTokenAuthMiddleware`` with ``AsyncMock`` state (no real aiosqlite
connection crossing event loops). Asserts the write-quartet authorization
(readonly blocked, owner/admin gates, 404 envelope), idempotency-key persistence
+ submitting-token capture, the structured ``service.name_taken``/``quota``
errors, and the redacted ``service.create`` audit row. A separate direct-DB
asyncio smoke proves ``GET /jobs`` excludes service rows.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nerdit.daemon.service_purge as purge_mod
from nerdit.config.settings import ServicesSettings
from nerdit.core.runtime.protocol import (
    ContainerNotFoundError,
    ContainerRuntimeError,
    ContainerStartError,
    SandboxViolationError,
)
from nerdit.core.runtime.stub import StubRuntime
from nerdit.core.services import (
    LaunchEnvNotReady,
    RunInterruptedError,
    RunPreconditionError,
    RunResult,
)
from nerdit.core.volumes import VolumeSpecError
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import QuotaExceeded, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.schemas.services import ServiceResponse
from nerdit.db.models import (
    ApiToken,
    Job,
    JobKind,
    JobStatus,
    ServiceEndpoint,
    TokenRole,
)
from nerdit.db.queries import ServiceNameTaken

LEGACY = "legacy-global"

ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
OTHER_RAW = "other-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
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


def _service(owner: str | None = None, **over) -> Job:
    """A service-kind row (config carries the image the response surfaces)."""
    fields = dict(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "nerdit-runtime:0.1", "port": 8000}),
        submitted_by_token=owner,
    )
    fields.update(over)
    return Job(**fields)


def _queries(service: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # Read/resolve
    q.get_job = AsyncMock(return_value=service)
    q.get_service_by_name = AsyncMock(return_value=None)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    # (P26 D-P26-14) The ONE batched hosted-share read every service
    # projection makes; empty here, so ``public_urls`` degenerates to the
    # ``default`` entry (or nothing) exactly as it did pre-P26.
    q.list_service_shares = AsyncMock(return_value={})
    # Writes
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    q.set_desired_state = AsyncMock()
    q.bump_restart_count = AsyncMock()
    q.release_gpus = AsyncMock()
    q.release_service_endpoint = AsyncMock()
    q.delete_service_checked = AsyncMock(return_value=[])
    return q


def _make_app(
    queries: AsyncMock, *, with_audit: bool = False, with_idempotency: bool = False
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)

    app.state.queries = queries
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime
    app.state.settings = MagicMock()
    # (P20) Real ``[services]`` values, not MagicMock attributes: the run route
    # compares ``timeout_s > run_timeout_max_s`` and formats
    # ``max_concurrent_runs`` into a hint, and a MagicMock ``>`` raises TypeError
    # instead of comparing. Individual tests override with a narrower cap.
    app.state.settings.services = ServicesSettings()
    # P14b WP-B1: the default purge=secrets DELETE touches only the SecretManager.
    secret_manager = MagicMock()
    secret_manager.delete = MagicMock(return_value=True)
    app.state.secret_manager = secret_manager

    # Innermost first (add_middleware prepends): Idempotency → Audit → Auth →
    # RequestId, matching the daemon's own order so a replay short-circuited by
    # the idempotency layer still returns up through Audit.
    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_audit:
        app.add_middleware(
            AuditMiddleware,
            get_queries=lambda: queries,
            get_event_bus=lambda: None,
        )
    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=LEGACY,
        get_queries=lambda: queries,
    )
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, **kw) -> TestClient:
    return TestClient(_make_app(queries, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _body(**over) -> dict:
    base = {"name": "demo", "image": "nerdit-runtime:0.1", "port": 8000, "gpus": 0}
    base.update(over)
    return base


# --- create: role gate -------------------------------------------------------


def test_submitter_can_create_service():
    resp = _client(_queries()).post("/services", json=_body(), headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    assert resp.json()["name"] == "demo"


def test_create_accepts_tcp_health_type():
    """P14 WP-C1: POST /services accepts a ``HealthCheck.type = "tcp"``."""
    q = _queries()
    resp = _client(q).post(
        "/services", json=_body(health_check={"type": "tcp"}), headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.await_args.args[0]
    assert job.health_check["type"] == "tcp"


def test_create_rejects_junk_health_type():
    """An unknown probe type is rejected at the request boundary (422)."""
    resp = _client(_queries()).post(
        "/services", json=_body(health_check={"type": "grpc"}), headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 422


def test_create_rejects_unknown_field_with_the_structured_envelope():
    """P22 WP-C: a typo'd field 422s (was silently dropped) and names the key."""
    q = _queries()
    resp = _client(q).post("/services", json=_body(gpu=2), headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    assert any("gpu" in str(d.get("loc")) for d in body["diagnostics"])
    q.reserve_service_for_token.assert_not_awaited()


def test_admin_can_create_service():
    assert (
        _client(_queries()).post("/services", json=_body(), headers=_auth(ADMIN_RAW)).status_code
        == 201
    )


def test_legacy_token_can_create_service():
    assert (
        _client(_queries()).post("/services", json=_body(), headers=_auth(LEGACY)).status_code
        == 201
    )


def test_readonly_blocked_on_create():
    # Coarse readonly gate fires in the middleware before the route.
    resp = _client(_queries()).post("/services", json=_body(), headers=_auth(RO_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_non_admin_script_path_forbidden():
    # SANDBOX-1: script_path bind-mounts a host dir; non-admin tokens have no
    # upload path for it in P2 → rejected at submit with a structured 403.
    resp = _client(_queries()).post(
        "/services", json=_body(script_path="/home/victim/.ssh/id_rsa"), headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "sandbox.script_path_forbidden"


def test_admin_script_path_allowed():
    resp = _client(_queries()).post(
        "/services", json=_body(script_path="/srv/app/main.py"), headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 201


def test_missing_image_is_422():
    body = {"name": "demo", "port": 8000}
    resp = _client(_queries()).post("/services", json=body, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_invalid_name_is_422():
    resp = _client(_queries()).post(
        "/services", json=_body(name="Bad_Name"), headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422


# --- create: idempotency + ownership capture ---------------------------------


def test_create_persists_idempotency_key():
    q = _queries()
    app = _make_app(q)
    TestClient(app, raise_server_exceptions=False).post(
        "/services", json=_body(), headers={**_auth(SUB_RAW), "Idempotency-Key": "idem-xyz"}
    )
    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.idempotency_key == "idem-xyz"


def test_create_records_submitting_token():
    q = _queries()
    app = _make_app(q)
    TestClient(app, raise_server_exceptions=False).post(
        "/services", json=_body(), headers=_auth(SUB_RAW)
    )
    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.submitted_by_token == "tok-sub"
    assert job_arg.kind == JobKind.service
    assert job_arg.desired_state == "running"


# --- create: structured errors -----------------------------------------------


def test_duplicate_name_returns_409():
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameTaken("demo"))
    resp = _client(q).post("/services", json=_body(), headers=_auth(ADMIN_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.name_taken"


def test_quota_exceeded_returns_403():
    q = _queries()
    q.reserve_service_for_token = AsyncMock(
        side_effect=QuotaExceeded("max_concurrent_jobs", limit=1, current=1)
    )
    resp = _client(q).post("/services", json=_body(), headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "quota_exceeded"


def test_missing_image_locally_returns_400():
    q = _queries()
    app = _make_app(q)
    app.state.runtime.image_exists = AsyncMock(return_value=False)
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/services", json=_body(), headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "image_not_found"


# --- write quartet: owner gate (stop / restart / delete) ---------------------


def test_submitter_can_stop_own_service():
    resp = _client(_queries(_service("tok-sub"))).post(
        "/services/svc-1/stop", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200


def test_submitter_cannot_stop_others_service():
    resp = _client(_queries(_service("tok-other"))).post(
        "/services/svc-1/stop", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


def test_null_owner_service_is_admin_only():
    resp = _client(_queries(_service(None))).post("/services/svc-1/stop", headers=_auth(SUB_RAW))
    assert resp.status_code == 403


def test_admin_can_stop_any_service():
    assert (
        _client(_queries(_service("tok-other")))
        .post("/services/svc-1/stop", headers=_auth(ADMIN_RAW))
        .status_code
        == 200
    )


def test_restart_owner_gate():
    resp = _client(_queries(_service("tok-other"))).post(
        "/services/svc-1/restart", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 403


def test_restart_clears_backoff():
    q = _queries(_service("tok-sub"))
    _client(q).post("/services/svc-1/restart", headers=_auth(SUB_RAW))
    q.set_desired_state.assert_awaited_with("svc-1", "running")
    q.bump_restart_count.assert_awaited_with("svc-1", 0, None)


def test_delete_owner_gate():
    resp = _client(_queries(_service("tok-other"))).delete(
        "/services/svc-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 403


def test_admin_can_delete_service():
    q = _queries(_service("tok-other"))
    resp = _client(q).delete("/services/svc-1", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    # The endpoint row is released INSIDE delete_service_checked's transaction
    # (FK to jobs(id) + stable-port preservation on a blocked re-check) — the
    # route never calls release_service_endpoint on the delete path anymore.
    q.release_service_endpoint.assert_not_awaited()
    assert q.delete_service_checked.await_args.args[0] == "svc-1"


def test_delete_service_deregisters_proxy_route():
    # P3: the DELETE handler must deregister the proxy route explicitly (it does
    # not funnel through the controller's _release_endpoint), before the endpoint
    # is released, so the URL stops resolving immediately.
    q = _queries(_service("tok-other"))
    app = _make_app(q)
    proxy = AsyncMock()
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    proxy.deregister.assert_awaited_with("demo")


def test_get_service_returns_public_url():
    # P3: GET surfaces the proxy-resolved HTTPS URL when the proxy is enabled and
    # the endpoint carries a route.
    from nerdit.config.settings import ProxySettings

    q = _queries(_service("tok-sub"))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="demo",
            job_id="svc-1",
            container_port=8000,
            host_port=9400,
            route="/demo",
        )
    )
    app = _make_app(q)
    app.state.settings.proxy = ProxySettings(enabled=True)
    app.state.hostname = "box"
    # public_url is gated on the LIVE proxy state, not the config flag.
    proxy = MagicMock()
    proxy.available = True
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["endpoint"]["public_url"] == "https://box/demo/"


def test_get_service_returns_public_url_subdomain_mode():
    # P3.5 subdomain twin: the empty-string route projection resolves to the
    # host-shaped URL when the proxy is enabled/available in subdomain mode.
    from nerdit.config.settings import ProxySettings

    q = _queries(_service("tok-sub"))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="demo",
            job_id="svc-1",
            container_port=8000,
            host_port=9400,
            route="",
        )
    )
    app = _make_app(q)
    app.state.settings.proxy = ProxySettings(
        enabled=True, mode="subdomain", base_domain="lan.local"
    )
    app.state.hostname = "box"
    proxy = MagicMock()
    proxy.available = True
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["endpoint"]["public_url"] == "https://demo.lan.local/"


def test_get_service_no_public_url_when_proxy_unavailable_subdomain_mode():
    # Same URL gating (availability, not just config) holds in subdomain mode.
    from nerdit.config.settings import ProxySettings

    q = _queries(_service("tok-sub"))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="demo",
            job_id="svc-1",
            container_port=8000,
            host_port=9400,
            route="",
        )
    )
    app = _make_app(q)
    app.state.settings.proxy = ProxySettings(
        enabled=True, mode="subdomain", base_domain="lan.local"
    )
    app.state.hostname = "box"
    proxy = MagicMock()
    proxy.available = False  # Caddy down
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["endpoint"]["public_url"] is None


def test_get_service_no_public_url_when_proxy_unavailable():
    # P3 regression: proxy enabled but Caddy never came up (available=False) →
    # no dead URL advertised even though the route is persisted.
    from nerdit.config.settings import ProxySettings

    q = _queries(_service("tok-sub"))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="demo",
            job_id="svc-1",
            container_port=8000,
            host_port=9400,
            route="/demo",
        )
    )
    app = _make_app(q)
    app.state.settings.proxy = ProxySettings(enabled=True)
    app.state.hostname = "box"
    proxy = MagicMock()
    proxy.available = False  # Caddy down (e.g. :443 bind failed)
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["endpoint"]["public_url"] is None


# --- PR1: ServiceResponse.source provenance projection -----------------------


def _svc_with_source(source: object) -> Job:
    """A service row whose config carries the given ``config['source']`` value."""
    return _service(
        "tok-sub", config=json.dumps({"image": "img:1", "port": 8000, "source": source})
    )


def test_detail_projects_zip_source():
    resp = _client(_queries(_svc_with_source({"type": "zip"}))).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == {"type": "zip"}


def test_list_projects_zip_source():
    svc = _svc_with_source({"type": "zip"})
    q = _queries(svc)
    q.list_services = AsyncMock(return_value=([svc], None))
    resp = _client(q).get("/services", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["items"][0]["source"] == {"type": "zip"}


def test_detail_projects_full_git_source():
    src = {
        "type": "git",
        "repo_url": "https://github.com/o/r",
        "ref": "main",
        "commit_sha": "a" * 40,
        "subdir": "svc",
        "template_id": "t1",
    }
    resp = _client(_queries(_svc_with_source(src))).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] == src


def test_source_projection_drops_non_allowlisted_keys():
    """The redaction assertion the plan demands: a poisoned source dict (a future
    ingress leaking a credential) can never surface a key outside the allowlist."""
    svc = _svc_with_source(
        {
            "type": "git",
            "repo_url": "https://github.com/o/r",
            "token_ref": "${secrets.GH}",
            "secret": "leak",
            "password": "pw",
        }
    )
    resp = _client(_queries(svc)).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    src = resp.json()["source"]
    assert set(src) == {"type", "repo_url"}
    assert "token_ref" not in resp.text


def test_non_dict_source_projects_none():
    resp = _client(_queries(_svc_with_source("not-a-dict"))).get(
        "/services/demo", headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] is None


def test_service_without_source_projects_none():
    # A POST /services-created service (no config['source']) → None.
    resp = _client(_queries(_service("tok-sub"))).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["source"] is None


# --- 404 envelope on every write target --------------------------------------


def test_stop_missing_service_404():
    resp = _client(_queries(None)).post("/services/nope/stop", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_restart_missing_service_404():
    assert (
        _client(_queries(None)).post("/services/nope/restart", headers=_auth(ADMIN_RAW)).status_code
        == 404
    )


def test_delete_missing_service_404():
    assert (
        _client(_queries(None)).delete("/services/nope", headers=_auth(ADMIN_RAW)).status_code
        == 404
    )


def test_batch_id_not_resolvable_as_service():
    # A batch row id must never be actionable through /services → 404.
    batch = Job(id="job-1", script_path="/x.py", gpu_count=1, kind=JobKind.batch)
    resp = _client(_queries(batch)).post("/services/job-1/stop", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 404


# --- DELETE ?purge / ?force + model reference guard (P14b WP-B1) --------------


def _model(owner: str | None = "tok-sub", **over) -> Job:
    """A ``kind=model`` row serving ``config['model']`` (the binding ref)."""
    fields = dict(
        id="mdl-1",
        service_name="llama",
        name="llama",
        kind=JobKind.model,
        gpu_count=1,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"model": "llama3.1:8b", "backend": "ollama"}),
        submitted_by_token=owner,
    )
    fields.update(over)
    return Job(**fields)


def _dependent(
    *, owner="tok-sub", provider="ollama", model="llama3.1:8b", name="app", jid="svc-app"
):
    """A workload-config row for a service binding a model via ``[ai.default]``."""
    return {
        "id": jid,
        "kind": "service",
        "service_name": name,
        "status": "running",
        "config": {"ai": {"default": {"provider": provider, "model": model}}},
        "submitted_by_token": owner,
    }


def test_delete_invalid_purge_token_422():
    resp = _client(_queries(_service("tok-sub"))).delete(
        "/services/svc-1?purge=secrets,bogus", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "service.invalid_purge"
    assert "secrets" in body["hint"] and "data" in body["hint"] and "images" in body["hint"]


def test_delete_model_blocked_by_ollama_dependent():
    """A vLLM-or-ollama model bound by an app's ollama binding → 409 resource.in_use."""
    model = _model(config=json.dumps({"model": "meta/llama-3", "backend": "vllm"}))
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[_dependent(model="meta/llama-3")])
    resp = _client(q).delete("/services/mdl-1", headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "resource.in_use"
    assert body["dependents"] == [{"service": "app", "id": "svc-app", "binding": "default"}]
    q.delete_service_checked.assert_not_awaited()


def test_delete_model_name_override_dependent_detected():
    """The binding matches by served ref, not by the model's ``--name`` override."""
    model = _model(service_name="custom-name", config=json.dumps({"model": "llama3.1:8b"}))
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[_dependent()])
    resp = _client(q).delete("/services/custom-name", headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "resource.in_use"


def test_delete_model_duplicate_ref_shadow_both_409():
    """Two model rows serving one ref, an app bound to it → deleting EITHER 409s.

    Ref equality (not resolver identity): a stale duplicate is protected too, so
    neither row can be silently deleted out from under the dependent.
    """
    rows_for = lambda target_id: [  # noqa: E731 — test-local
        {
            "id": "mdl-a",
            "kind": "model",
            "service_name": "llama-a",
            "status": "running",
            "config": {"model": "llama3.1:8b"},
        },
        {
            "id": "mdl-b",
            "kind": "model",
            "service_name": "llama-b",
            "status": "running",
            "config": {"model": "llama3.1:8b"},
        },
        _dependent(),
    ]
    for target in ("mdl-a", "mdl-b"):
        model = _model(id=target, service_name=f"llama-{target[-1]}")
        q = _queries(model)
        q.list_workload_configs = AsyncMock(return_value=rows_for(target))
        resp = _client(q).delete(f"/services/{target}", headers=_auth(SUB_RAW))
        assert resp.status_code == 409, target
        assert resp.json()["code"] == "resource.in_use"


def test_delete_model_api_provider_never_matches():
    """An ``api`` binding is external — it never protects a local model row."""
    model = _model()
    q = _queries(model)
    q.list_workload_configs = AsyncMock(
        return_value=[_dependent(provider="api", model="llama3.1:8b")]
    )
    resp = _client(q).delete("/services/mdl-1", headers=_auth(SUB_RAW))
    assert resp.status_code == 200
    assert q.delete_service_checked.await_args.args[0] == "mdl-1"


def test_delete_model_force_cross_owner_requires_admin():
    """force by a non-admin is refused when a dependent is owned by another token."""
    model = _model(owner="tok-sub")
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[_dependent(owner="tok-other")])

    def _get_job(jid):
        return model if jid == "mdl-1" else _service("tok-other", id="svc-app")

    q.get_job = AsyncMock(side_effect=_get_job)
    resp = _client(q).delete("/services/mdl-1?force=true", headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    q.delete_service_checked.assert_not_awaited()


def test_delete_model_force_cross_owner_admin_succeeds():
    """An admin may force-delete a model even with foreign dependents (stamped)."""
    model = _model(owner="tok-sub")
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[_dependent(owner="tok-other")])

    def _get_job(jid):
        return model if jid == "mdl-1" else _service("tok-other", id="svc-app")

    q.get_job = AsyncMock(side_effect=_get_job)
    resp = _client(q, with_audit=True).delete(
        "/services/mdl-1?force=true", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert q.delete_service_checked.await_args.args[0] == "mdl-1"
    # The main audit row records the force + dependents.
    forced_row = [
        c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == "service.delete"
    ]
    assert forced_row, "expected a service.delete audit row"
    params = json.loads(forced_row[-1].kwargs["params_redacted"])
    assert params["forced"] is True
    assert params["dependents"] == ["app"]


def test_delete_model_guard_rerun_restores_running():
    """A dependent appearing during teardown → 409 AND desired_state restored.

    Teardown already happened, so the atomic re-check (inside
    ``delete_service_checked``) reporting a fresh dependent must set
    desired_state back to ``running`` (the reconciler relaunches the model),
    never leaving it stopped — and the row survives.
    """
    model = _model()
    q = _queries(model)
    # Initial guard sees no dependents; the atomic in-lock re-check sees one.
    q.list_workload_configs = AsyncMock(return_value=[])
    q.delete_service_checked = AsyncMock(return_value=[_dependent()])
    app = _make_app(q)
    proxy = AsyncMock()
    app.state.proxy_manager = proxy
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/mdl-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "resource.in_use"
    assert "relaunch" in body["message"].lower()
    # The atomic delete was invoked WITH a checker (model kind, non-force) —
    # the re-check runs inside the write lock, not at the route level.
    args = q.delete_service_checked.await_args.args
    assert args[0] == "mdl-1"
    assert callable(args[1])
    # desired_state was set 'stopped' (teardown) then restored to 'running'.
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert ("mdl-1", "stopped") in states
    assert states[-1] == ("mdl-1", "running")
    # FINDING 2: a REFUSED delete never surrenders the stable host port — the
    # endpoint release + proxy deregister run only AFTER a committed delete, so a
    # relaunched model reclaims exactly the port it held.
    q.release_service_endpoint.assert_not_awaited()
    proxy.deregister.assert_not_awaited()


def test_delete_model_force_cross_owner_atomic_recheck_403():
    """A foreign dependent committed DURING teardown still 403s a non-admin force.

    The pre-teardown snapshot is clean (fast path passes), but the atomic re-check
    inside ``delete_service_checked`` reports a FOREIGN-owned dependent → 403
    ``forbidden`` AND ``desired_state`` restored to ``running`` (the M2 TOCTOU the
    pre-teardown snapshot leaves open).
    """
    model = _model(owner="tok-sub")
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[])  # pre-teardown: clean
    q.delete_service_checked = AsyncMock(return_value=[_dependent(owner="tok-other")])
    resp = _client(q).delete("/services/mdl-1?force=true", headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    # force + non-admin + model → the CROSS-OWNER checker (not None): it filters
    # same-owner dependents out but keeps foreign ones.
    checker = q.delete_service_checked.await_args.args[1]
    assert callable(checker)
    assert checker([_dependent(owner="tok-sub")]) == []
    assert checker([_dependent(owner="tok-other")]) == [
        {"service": "app", "id": "svc-app", "binding": "default"}
    ]
    # A NULL owner is fail-closed (treated as foreign).
    assert checker([_dependent(owner=None)]) == [
        {"service": "app", "id": "svc-app", "binding": "default"}
    ]
    # Teardown already happened → desired_state restored to running.
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert states[-1] == ("mdl-1", "running")


def test_delete_model_passes_ref_checker_and_admin_force_skips_it():
    """Non-force model deletes carry a ref-equality checker; ADMIN force passes None."""
    model = _model()
    q = _queries(model)
    q.list_workload_configs = AsyncMock(return_value=[])
    assert _client(q).delete("/services/mdl-1", headers=_auth(SUB_RAW)).status_code == 200
    checker = q.delete_service_checked.await_args.args[1]
    assert callable(checker)
    # Pure function over workload rows: detects a dependent, ignores others.
    dep_rows = [_dependent()]
    assert checker(dep_rows) == [{"service": "app", "id": "svc-app", "binding": "default"}]
    assert checker([]) == []

    # Admin force intends to delete regardless → no atomic checker at all. (A
    # NON-admin force carries the cross-owner checker instead — see
    # ``test_delete_model_force_cross_owner_atomic_recheck_403``.)
    model2 = _model()
    q2 = _queries(model2)
    q2.list_workload_configs = AsyncMock(return_value=[])
    assert (
        _client(q2).delete("/services/mdl-1?force=true", headers=_auth(ADMIN_RAW)).status_code
        == 200
    )
    assert q2.delete_service_checked.await_args.args[1] is None


def test_delete_run_race_hook_blocks(monkeypatch):
    """The None-tolerant run-race hook 409s when a controller reports an active run."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    app.state.service_controller = MagicMock()
    app.state.service_controller.has_active_run = lambda jid: True
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    q.delete_service_checked.assert_not_awaited()


def _real_controller(queries) -> object:
    """A REAL ``ServiceController`` backing the purge hook (P20).

    The lambda-stubbed test above pins the route's reaction; this pins the whole
    seam — the controller's registry really is what the route consults, so a
    change to either side breaks a test.
    """
    from nerdit.core.services import ServiceController

    return ServiceController(queries=queries, runtime=StubRuntime())


def test_delete_during_a_run_409s_against_the_real_controller():
    """(P20) The registry is the only record of a rowless run container, so the
    DELETE route must refuse rather than tear the service down underneath it."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    controller = _real_controller(q)
    app.state.service_controller = controller
    controller._register_run("svc-1", "run-1", is_release=False)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.delete("/services/svc-1", headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    q.delete_service_checked.assert_not_awaited()

    # Once the run finishes, the same DELETE goes through.
    controller._discard_run("svc-1", "run-1")
    assert client.delete("/services/svc-1", headers=_auth(SUB_RAW)).status_code == 200


def test_delete_during_a_release_409s():
    """Documented behaviour change (reality R16): a delete during a *release*
    now 409s, where a delete during a plain build used to succeed. A release
    container holds the service's data volume mid-migration — tearing the
    service down underneath it is a torn teardown."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    controller = _real_controller(q)
    app.state.service_controller = controller
    controller._register_run("svc-1", "rel-1", is_release=True)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    q.delete_service_checked.assert_not_awaited()


def test_delete_without_a_controller_is_unaffected():
    """The hook is None-tolerant: a bare app (no ``service_controller`` on
    ``app.state``) still deletes — the seam must not become a hard dependency."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    assert getattr(app.state, "service_controller", None) is None
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200


def test_delete_default_purges_secrets_only():
    """Default purge=secrets calls SecretManager.delete and skips data/images/workspace."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1", headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    app.state.secret_manager.delete.assert_called_once_with("demo")
    body = resp.json()
    # (P29) ``workspace`` joined the report as a fourth opt-in member; every
    # unasked member stays ``None``, so the default set still purges only secrets.
    assert body["purged"] == {
        "secrets": True,
        "data": None,
        "images": None,
        "workspace": None,
    }
    # A service.purge_secrets out-of-band audit row was inserted (key names only).
    purge_rows = [
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "service.purge_secrets"
    ]
    assert purge_rows
    params = json.loads(purge_rows[-1].kwargs["params_redacted"])
    assert params["key"] == "services/demo"


def test_delete_purge_skipped_when_service_name_none():
    """A null-service_name row (never happens for services) skips every purge step."""
    svc = _service("tok-sub", service_name=None)
    q = _queries(svc)
    app = _make_app(q)
    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=secrets,data,images", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["purged"] is None
    app.state.secret_manager.delete.assert_not_called()


# --- F12: never rmtree data after an UNCONFIRMED teardown --------------------


def _teardown_runtime(*, fail: bool, running: object, err: Exception | None = None) -> AsyncMock:
    """A runtime whose stop/kill/remove optionally raise, with a scripted probe.

    ``running`` is either a tri-state return value for ``container_running``
    (``True`` still writing / ``False`` inert or gone / ``None`` the runtime cannot
    answer) or an exception instance to raise from the probe. ``err`` overrides the
    teardown exception (default: a generic ``ContainerRuntimeError``).
    """
    runtime = AsyncMock()
    if fail:
        exc = err or ContainerRuntimeError("docker is wedged")
        runtime.stop = AsyncMock(side_effect=exc)
        runtime.kill = AsyncMock(side_effect=exc)
        runtime.remove = AsyncMock(side_effect=exc)
    if isinstance(running, BaseException):
        runtime.container_running = AsyncMock(side_effect=running)
    else:
        runtime.container_running = AsyncMock(return_value=running)
    return runtime


def _data_purge_app(tmp_path, monkeypatch, q: AsyncMock, runtime: AsyncMock | StubRuntime):
    """App wired to a REAL data_dir with ``services/demo/blob`` seeded, rmtree recorded."""
    app = _make_app(q)
    app.state.runtime = runtime
    app.state.settings = SimpleNamespace(data_dir=str(tmp_path))
    root = tmp_path / "services" / "demo"
    root.mkdir(parents=True)
    (root / "blob").write_text("precious")

    calls: list[str] = []
    real_rmtree = shutil.rmtree

    def _recording_rmtree(path, *a, **kw):
        calls.append(str(path))
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(purge_mod.shutil, "rmtree", _recording_rmtree)
    return app, root, calls


def _purge_data_audit(q: AsyncMock) -> dict:
    rows = [
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "service.purge_data"
    ]
    assert rows, "no service.purge_data audit row"
    return json.loads(rows[-1].kwargs["params_redacted"])


def test_delete_data_purge_skipped_when_container_still_alive(tmp_path, monkeypatch):
    """Teardown raised AND the container is still present ⇒ the rmtree is SKIPPED."""
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=True)
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == []  # rmtree never ran
    assert (root / "blob").read_text() == "precious"
    assert resp.json()["purged"]["data"] is False
    params = _purge_data_audit(q)
    assert params["purged"] is False
    assert params["reason"] == "container_alive"


def test_delete_data_purge_runs_when_reconciler_removed_the_container_first(tmp_path, monkeypatch):
    """The reconcile race: ``stop`` raises NotFound ⇒ that is CONFIRMATION, not failure.

    Regression test for a live-run bug. This route sets ``desired_state=stopped``
    BEFORE tearing the container down, so the ServiceController reconcile loop
    routinely wins the race and removes the container first. ``DockerRuntime.stop``
    then raises ``ContainerNotFoundError`` — which subclasses ``ContainerRuntimeError``
    — so the teardown block counted a perfectly healthy delete as "unconfirmed" and
    the F12 gate silently skipped the purge (audited ``reason: container_alive`` while
    Docker was in perfect health, data dir left behind). "Already gone" is the
    strongest confirmation there is that nothing is writing.
    """
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=False, err=ContainerNotFoundError("gone"))
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    runtime.container_running.assert_not_awaited()  # NotFound needs no probe at all
    assert calls == [str(root)]  # the rmtree RAN
    assert not root.exists()
    assert resp.json()["purged"]["data"] is True
    assert "reason" not in _purge_data_audit(q)


def test_delete_data_purge_runs_when_container_present_but_exited(tmp_path, monkeypatch):
    """An exited-but-not-yet-removed container writes nothing ⇒ the rmtree still runs.

    The probe asks "is it still WRITING", not "does it still EXIST" — gating on mere
    presence would skip the purge every time a teardown raced a reconcile tick that
    had already stopped (but not yet reaped) the container.
    """
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=False)  # present, but inert
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == [str(root)]
    assert resp.json()["purged"]["data"] is True


def test_delete_data_purge_skipped_when_runtime_cannot_answer(tmp_path, monkeypatch):
    """The Docker-outage case: the probe returns ``None`` ⇒ unknown is NOT gone.

    This is the scenario the whole guard exists for — the daemon is unreachable,
    which is *why* the teardown raised, and containerd keeps the container (and its
    writes to the bind-mounted dir) alive underneath. ``status``/``inspect_state``
    would both report ``None`` here and be misread as "gone"; the tri-state
    ``container_present`` is what makes the distinction available.
    """
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=None)
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == []  # rmtree never ran
    assert (root / "blob").read_text() == "precious"
    assert resp.json()["purged"]["data"] is False
    assert _purge_data_audit(q)["reason"] == "container_alive"


def test_delete_data_purge_runs_when_teardown_failed_but_container_gone(tmp_path, monkeypatch):
    """Teardown raised but the probe confirms the container is gone ⇒ the rmtree runs."""
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=False)
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == [str(root)]
    assert not root.exists()
    assert resp.json()["purged"]["data"] is True
    params = _purge_data_audit(q)
    assert params["purged"] is True
    assert "reason" not in params  # happy-path audit shape is unchanged


def test_delete_data_purge_skipped_when_probe_itself_raises(tmp_path, monkeypatch):
    """An unanswerable probe is not a confirmation ⇒ fail closed (documented choice)."""
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=True, running=RuntimeError("docker unreachable"))
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == []
    assert (root / "blob").read_text() == "precious"
    assert resp.json()["purged"]["data"] is False
    assert _purge_data_audit(q)["reason"] == "container_alive"


def test_delete_data_purge_unaffected_on_clean_teardown(tmp_path, monkeypatch):
    """Happy path: teardown succeeded ⇒ no probe at all, rmtree runs as before."""
    q = _queries(_service("tok-sub", container_id="c1"))
    runtime = _teardown_runtime(fail=False, running=True)
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, runtime)

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    runtime.container_running.assert_not_awaited()  # the happy path never probes
    assert calls == [str(root)]
    assert not root.exists()
    assert resp.json()["purged"]["data"] is True
    assert "reason" not in _purge_data_audit(q)


def test_delete_data_purge_skipped_under_the_real_stub_runtime(tmp_path, monkeypatch):
    """C1: the daemon fell back to ``StubRuntime`` (Docker down) ⇒ liveness is UNKNOWN.

    Deliberately not a mock: ``server.py`` installs the REAL ``StubRuntime`` when Docker
    cannot be reached at startup, so this is a production configuration, not a test
    double. Its stop/kill/remove all raise ``ContainerRuntimeError`` — the teardown is
    therefore unconfirmed — and a Docker we cannot reach cannot testify that a container
    from a previous daemon run has stopped: containerd may still be running it, still
    writing into the bind-mounted data dir. The stub's probe must answer ``None``
    (cannot tell), never ``False``, or this delete ``rmtree``s a live writer's data.
    """
    q = _queries(_service("tok-sub", container_id="c1"))
    app, root, calls = _data_purge_app(tmp_path, monkeypatch, q, StubRuntime())

    resp = TestClient(app, raise_server_exceptions=False).delete(
        "/services/svc-1?purge=data", headers=_auth(ADMIN_RAW)
    )
    assert resp.status_code == 200
    assert calls == []  # rmtree never ran
    assert (root / "blob").read_text() == "precious"  # the data SURVIVES
    assert resp.json()["purged"]["data"] is False
    assert _purge_data_audit(q)["reason"] == "container_alive"


# --- reads -------------------------------------------------------------------


def test_get_service_by_name():
    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=_service("tok-sub"))
    resp = _client(q).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    assert resp.json()["image"] == "nerdit-runtime:0.1"


def test_detail_route_computes_data_dir_bytes(tmp_path):
    """The detail route walks the service data dir; the list route never does."""
    # Populate <data_dir>/services/demo with a byte or two.
    svc_dir = tmp_path / "services" / "demo"
    svc_dir.mkdir(parents=True)
    (svc_dir / "app.db").write_bytes(b"x" * 512)

    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=_service("tok-sub"))
    q.list_services = AsyncMock(return_value=([_service("tok-sub")], None))
    app = _make_app(q)
    app.state.settings.data_dir = str(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    detail = client.get("/services/demo", headers=_auth(RO_RAW))
    assert detail.status_code == 200
    assert detail.json()["data_dir_bytes"] == 512

    listing = client.get("/services", headers=_auth(RO_RAW))
    assert listing.status_code == 200
    assert listing.json()["items"][0]["data_dir_bytes"] is None


async def test_data_dir_walk_is_coalesced_under_concurrency(tmp_path, monkeypatch):
    """M4: N concurrent detail reads cost at most one walk per TTL.

    Unguarded, each read dispatched its own recursive `du` onto the loop's
    DEFAULT executor — the pool every `DockerRuntime` call and the reconciler
    share (measured: 42 concurrent readonly GETs delayed a pooled call 13.8 s).
    """
    import asyncio
    import time

    from nerdit.daemon.routes import services as svc_mod

    svc_dir = tmp_path / "services" / "demo"
    svc_dir.mkdir(parents=True)
    (svc_dir / "app.db").write_bytes(b"x" * 512)

    calls: list[str] = []
    real_du = svc_mod.du_bytes

    def counting_du(path):
        calls.append(str(path))
        time.sleep(0.05)  # widen the window so an unguarded fan-out is certain
        return real_du(path)

    monkeypatch.setattr(svc_mod, "du_bytes", counting_du)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(data_dir=str(tmp_path))))
    )
    job = _service("tok-sub")

    results = await asyncio.gather(
        *[svc_mod._compute_data_dir_bytes(request, job) for _ in range(20)]
    )
    assert results == [512] * 20
    assert len(calls) == 1, f"expected one coalesced walk, got {len(calls)}"

    # Still one within the TTL window — a follow-up read is served from cache.
    assert await svc_mod._compute_data_dir_bytes(request, job) == 512
    assert len(calls) == 1


async def test_a_slow_walk_does_not_block_another_service(tmp_path, monkeypatch):
    """The single-flight is per NAME, not daemon-wide.

    A daemon-wide lock coalesces a hammer on one service at the price of making
    a slow tree hold up the detail read of every OTHER service for a whole
    budget — cross-service head-of-line blocking the route never had before it
    was serialized.
    """
    import asyncio
    import time

    from nerdit.daemon.routes import services as svc_mod

    for name, payload in (("big", b"x" * 64), ("small", b"y" * 8)):
        d = tmp_path / "services" / name
        d.mkdir(parents=True)
        (d / "blob").write_bytes(payload)

    _real_du = svc_mod.du_bytes

    def slow_du(path):
        if path.name == "big":
            time.sleep(0.4)
        return _real_du(path)

    monkeypatch.setattr(svc_mod, "du_bytes", slow_du)
    monkeypatch.setattr(svc_mod, "_DATA_DIR_BUDGET_S", 2.0)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(data_dir=str(tmp_path))))
    )
    big = _service("tok-sub", service_name="big")
    small = _service("tok-sub", service_name="small")

    started = time.monotonic()
    big_task = asyncio.create_task(svc_mod._compute_data_dir_bytes(request, big))
    await asyncio.sleep(0.05)
    assert await svc_mod._compute_data_dir_bytes(request, small) == 8
    assert time.monotonic() - started < 0.35, "the small read waited on big's walk"
    assert await big_task == 64


async def test_an_overrun_walk_lands_in_the_cache_instead_of_being_thrown_away(
    tmp_path, monkeypatch
):
    """An overrunning walk used to be cancelled and its result dropped, so a
    tree slower than the budget reported ``null`` forever WHILE the daemon
    started a complete new walker every TTL. Now it finishes once and fills the
    cache, and a second read inside the TTL never starts a second walker."""
    import asyncio
    import time

    from nerdit.daemon.routes import services as svc_mod

    d = tmp_path / "services" / "demo"
    d.mkdir(parents=True)
    (d / "blob").write_bytes(b"z" * 32)

    calls: list[str] = []
    real_du = svc_mod.du_bytes

    def slow_du(path):
        calls.append(str(path))
        time.sleep(0.3)
        return real_du(path)

    monkeypatch.setattr(svc_mod, "du_bytes", slow_du)
    monkeypatch.setattr(svc_mod, "_DATA_DIR_BUDGET_S", 0.05)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=SimpleNamespace(data_dir=str(tmp_path))))
    )
    job = _service("tok-sub")

    assert await svc_mod._compute_data_dir_bytes(request, job) is None  # overran
    # A read while it is still running joins it rather than spawning a second.
    assert await svc_mod._compute_data_dir_bytes(request, job) is None
    assert len(calls) == 1

    await asyncio.sleep(0.4)  # let the detached walk land
    assert await svc_mod._compute_data_dir_bytes(request, job) == 32
    assert len(calls) == 1  # served from the cache the walk itself filled


def test_detail_route_data_dir_bytes_null_when_dir_absent(tmp_path):
    """A row whose data dir was never created reports null, not 0 (#6)."""
    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=_service("tok-sub"))
    app = _make_app(q)
    app.state.settings.data_dir = str(tmp_path)  # no services/demo dir created
    client = TestClient(app, raise_server_exceptions=False)

    detail = client.get("/services/demo", headers=_auth(RO_RAW))
    assert detail.status_code == 200
    assert detail.json()["data_dir_bytes"] is None


def test_get_service_surfaces_error_class():
    """(P13) A failed service projects its error_class onto the GET response body."""
    from nerdit.db.models import ErrorClass

    svc = _service(
        "tok-sub",
        status=JobStatus.failed,
        error_class=ErrorClass.oom,
        error_message="Restart budget exhausted",
    )
    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=svc)
    resp = _client(q).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    assert resp.json()["error_class"] == ErrorClass.oom.value


def test_fresh_service_has_no_rollback_fields():
    """A service never deployed through /deploy exposes the additive defaults (P6, D3)."""
    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=_service("tok-sub"))
    resp = _client(q).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["rollback_available"] is False
    assert body["build_version"] is None


def test_deployed_service_surfaces_rollback_fields():
    """A config blob carrying previous_image/build_version projects onto the response."""
    svc = _service(
        "tok-sub",
        config=json.dumps(
            {
                "image": "nerdit-app/demo:2",
                "previous_image": "nerdit-app/demo:1",
                "build_version": 2,
                "port": 8000,
            }
        ),
    )
    q = _queries()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=svc)
    resp = _client(q).get("/services/demo", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["rollback_available"] is True
    assert body["build_version"] == 2


def test_delete_response_wire_shape_unchanged():
    """DELETE keeps the legacy keys byte-identical + adds the P14b ``purged`` field.

    Deliberately amended by WP-B1 (its stated purpose): the strict ``==`` pin now
    asserts the legacy-key subset is unchanged and the new ``purged`` object is
    present (default purge=secrets ⇒ a per-service SecretManager.delete).
    """
    q = _queries(_service("tok-sub"))
    resp = _client(q).delete("/services/svc-1", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    body = resp.json()
    # Legacy keys byte-identical.
    assert body["id"] == "svc-1"
    assert body["name"] == "demo"
    assert body["deleted"] is True
    # New field: purge report for the default secrets purge. (P29) The report
    # gained a fourth member, ``workspace`` — additive and ``None`` when not
    # asked for, exactly like ``data``/``images``; the legacy keys above are
    # what this test pins byte-identical.
    assert body["purged"] == {
        "secrets": True,
        "data": None,
        "images": None,
        "workspace": None,
    }


def test_list_services_paginated():
    q = _queries()
    q.list_services = AsyncMock(return_value=([_service("tok-sub")], None))
    resp = _client(q).get("/services", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1 and body["next_cursor"] is None


# --- audit row is redacted ---------------------------------------------------


def test_create_audit_row_redacts_secret():
    q = _queries()
    client = _client(q, with_audit=True)
    resp = client.post(
        "/services",
        json=_body(env={"API_KEY": "topsecret"}),
        headers=_auth(ADMIN_RAW),
    )
    assert resp.status_code == 201
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "service.create"
    assert kwargs["target_type"] == "service"
    assert "topsecret" not in (kwargs["params_redacted"] or "")
    assert "***" in (kwargs["params_redacted"] or "")


# --- GET /services/{ident}/wait — the converge primitive (P13 WP3) ------------


def _deploy_cfg(*, version: int, phase: str, **over) -> dict:
    """A service config blob carrying a P13 ``last_deploy`` generation object."""
    ld = {
        "version": version,
        "action": "deploy",
        "phase": phase,
        "image": "nerdit-app/demo:%d" % version,
        "started_at": "2026-07-10T00:00:00+00:00",
        "updated_at": "2026-07-10T00:00:00+00:00",
        "reason": None,
        "error_class": None,
        "error_message": None,
    }
    ld.update(over)
    return {"image": ld["image"], "port": 8000, "build_version": version, "last_deploy": ld}


def test_wait_converges_immediately_when_healthy():
    svc = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=2, phase="healthy")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 2, "timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "converged"
    assert body["version"] == 2
    assert body["phase"] == "healthy"
    assert body["status"] == "running"


def test_wait_redeploy_build_fail_resolves_failed_while_running():
    # Redeploy build failure: last_deploy is ``failed`` but the old image still
    # serves (status ``running``) — the response must carry both.
    cfg = _deploy_cfg(
        version=3,
        phase="failed",
        reason="build_failed",
        error_class="USER_ERROR",
        error_message="npm ci failed",
    )
    svc = _service(status=JobStatus.running, config=json.dumps(cfg))
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 3, "timeout": 5}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "failed"
    assert body["status"] == "running"
    assert body["reason"] == "build_failed"
    assert body["error_class"] == "USER_ERROR"
    assert body["error_message"] == "npm ci failed"


def test_wait_superseded_resolves_immediately_without_burning_timeout():
    # Requested v4, but a concurrent redeploy already stamped v5.
    svc = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=5, phase="building")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 4, "timeout": 300}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "superseded"
    assert body["version"] == 4
    assert body["waited_s"] == 0.0


def _driven_get_job(rows: list):
    """A ``get_job`` side_effect returning ``rows`` in order, holding the last.

    Each ``_resolve_service`` re-read (the poll-refresh path) consumes one call,
    so a row that TRANSITIONS mid-wait is only observed if that re-read is live.
    """
    state = {"i": 0}

    def _f(_ident):
        i = state["i"]
        if i < len(rows) - 1:
            state["i"] += 1
        return rows[i]

    return _f


def test_wait_converges_after_transition_mid_wait(monkeypatch):
    # A driven controller: the row is ``building`` for the first two reads, then
    # flips to ``healthy``. Convergence must come from the poll-refresh re-read,
    # not iteration 0 — deleting that re-read makes this time out.
    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_POLL_INTERVAL", 0.01)
    building = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=2, phase="building")),
    )
    healthy = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=2, phase="healthy")),
    )
    q = _queries(building)
    q.get_job = AsyncMock(side_effect=_driven_get_job([building, building, healthy]))
    resp = _client(q).get(
        "/services/demo/wait", params={"version": 2, "timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "converged"
    assert body["phase"] == "healthy"
    # The poll-refresh re-read is load-bearing: it took >=3 reads (2 building +
    # the healthy flip), so convergence came from a later poll, not iteration 0.
    assert q.get_job.await_count >= 3


def test_wait_fails_after_transition_mid_wait(monkeypatch):
    # Driven controller: ``building`` → ``failed`` phase mid-wait resolves failed
    # via the poll-refresh re-read.
    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_POLL_INTERVAL", 0.01)
    building = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=3, phase="building")),
    )
    failed = _service(
        status=JobStatus.running,
        config=json.dumps(
            _deploy_cfg(
                version=3,
                phase="failed",
                reason="build_failed",
                error_class="USER_ERROR",
                error_message="npm ci failed",
            )
        ),
    )
    q = _queries(building)
    q.get_job = AsyncMock(side_effect=_driven_get_job([building, building, failed]))
    resp = _client(q).get(
        "/services/demo/wait", params={"version": 3, "timeout": 5}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "failed"
    assert body["reason"] == "build_failed"
    assert q.get_job.await_count >= 3  # resolved via the poll-refresh re-read


def test_wait_superseded_stamped_mid_wait_resolves_immediately(monkeypatch):
    # Wait on v4 while the row is still v4 building; a concurrent redeploy stamps
    # v5 mid-wait. The wait must resolve ``superseded`` on the re-read (not burn
    # the 300 s timeout) — the plan's named superseded case.
    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_POLL_INTERVAL", 0.01)
    v4 = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=4, phase="building")),
    )
    v5 = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=5, phase="building")),
    )
    q = _queries(v4)
    q.get_job = AsyncMock(side_effect=_driven_get_job([v4, v4, v5]))
    resp = _client(q).get(
        "/services/demo/wait", params={"version": 4, "timeout": 300}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "superseded"
    assert body["version"] == 4
    assert body["waited_s"] < 300.0  # never burned the timeout
    # Resolved on the re-read (v5 stamped mid-wait), not iteration 0.
    assert q.get_job.await_count >= 3


def test_wait_status_only_defers_within_start_period(monkeypatch):
    # §1.2 converged definition: a ``running`` status-only row with a health
    # check still inside its start period must NOT converge (health unverified).
    from datetime import datetime, timezone

    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_POLL_INTERVAL", 0.01)
    # Started "now" with a 3600 s start period → always inside the grace window.
    svc = _service(
        status=JobStatus.running,
        config=json.dumps({"image": "x:1", "port": 8000}),
        started_at=datetime.now(timezone.utc),
        health_check={"path": "/health", "start_period_s": 3600.0},
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"timeout": 1}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    # Health unverified within the start period → does not converge → timeout.
    assert resp.json()["outcome"] == "timeout"


def test_wait_old_container_running_during_redeploy_does_not_converge():
    # The new generation (v4) is still ``building`` while the old image serves
    # (status ``running``). A status-``running`` row must NOT converge on v4.
    svc = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=4, phase="building")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 4, "timeout": 1}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "timeout"
    assert body["phase"] == "building"


def test_wait_times_out_when_never_healthy():
    svc = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=2, phase="building")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 2, "timeout": 1}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "timeout"


def test_wait_status_only_mode_converges_on_running():
    # Pre-P13 / POST-/services row: no ``last_deploy`` object → status-only mode.
    svc = _service(status=JobStatus.running, config=json.dumps({"image": "x:1", "port": 8000}))
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "converged"
    assert body["phase"] is None


def test_wait_model_running_and_pulled_converges_immediately():
    # Finding #14: readiness for a kind=model row IS the ``model_pulled`` flag.
    # A running+pulled model must resolve ``converged`` in <1s, not burn the
    # timeout through the status-only start-period deferral.
    mdl = _model(
        status=JobStatus.running,
        config=json.dumps({"model": "tiny", "backend": "ollama", "model_pulled": True}),
    )
    resp = _client(_queries(mdl)).get(
        "/services/llama/wait", params={"timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "converged"
    assert body["waited_s"] < 1


def test_wait_model_pull_failed_resolves_failed():
    # A running model whose weights pull failed resolves ``failed`` (the bare
    # server answers its liveness probe, so status stays ``running``).
    mdl = _model(
        status=JobStatus.running,
        config=json.dumps({"model": "tiny", "backend": "ollama"}),
        error_message="Model pull failed: not found",
    )
    resp = _client(_queries(mdl)).get(
        "/services/llama/wait", params={"timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "failed"
    assert body["error_message"] == "Model pull failed: not found"


def test_wait_model_still_pulling_times_out():
    # A model whose image/weights are still pulling keeps polling → honest timeout.
    mdl = _model(
        status=JobStatus.building,
        config=json.dumps({"model": "tiny", "backend": "ollama"}),
    )
    resp = _client(_queries(mdl)).get(
        "/services/llama/wait", params={"timeout": 1}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "timeout"


def test_wait_saturation_resolves_timeout_zero(monkeypatch):
    # A fully-saturated wait semaphore resolves the request immediately as
    # ``timeout, waited_s=0`` (never holding an unbounded connection).
    import asyncio as _asyncio

    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_SEMAPHORE", _asyncio.Semaphore(0))
    svc = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=2, phase="healthy")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": 2, "timeout": 5}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "timeout"
    assert body["waited_s"] == 0.0


def test_wait_unknown_service_404():
    q = _queries(None)
    resp = _client(q).get("/services/nope/wait", headers=_auth(RO_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_wait_unparsable_version_400():
    svc = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=1, phase="healthy")),
    )
    resp = _client(_queries(svc)).get(
        "/services/demo/wait", params={"version": "notanint"}, headers=_auth(RO_RAW)
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_wait_superseded_by_rollback_lower_version(monkeypatch):
    # PR-review fix (Codex P2): a ROLLBACK stamps a *lower* version by design
    # and supersedes a pending wait exactly like a redeploy. The old ``>``-only
    # predicate burned the full timeout here.
    from nerdit.daemon.routes import service_wait as wait_mod

    monkeypatch.setattr(wait_mod, "_WAIT_POLL_INTERVAL", 0.01)
    v5 = _service(
        status=JobStatus.building,
        config=json.dumps(_deploy_cfg(version=5, phase="building")),
    )
    v4 = _service(
        status=JobStatus.running,
        config=json.dumps(_deploy_cfg(version=4, phase="queued")),
    )
    q = _queries(v5)
    q.get_job = AsyncMock(side_effect=_driven_get_job([v5, v5, v4]))
    resp = _client(q).get(
        "/services/demo/wait", params={"version": 5, "timeout": 300}, headers=_auth(SUB_RAW)
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "superseded"
    assert body["version"] == 5
    assert body["waited_s"] < 300.0  # resolved immediately, never burned the timeout


# --- POST /services/{ident}/run — the one-off run primitive (P20 WP4+WP5) -----
#
# The route is the gatekeeper, the controller is the engine: everything below
# pins the *gate* (kind, drain, caps, quota, authz, audit redaction,
# idempotency) against a REAL ``ServiceController`` whose ``run_once`` is the
# only stubbed member. The registry, the caps and the per-token run slots are
# therefore the production ones — a change to either side of that seam breaks a
# test. Execution mechanics (env precedence, scrub, timeout, volumes) live at
# the controller tier in ``tests/test_run_primitive.py``.


def _database(owner: str | None = "tok-sub", **over) -> Job:
    """A ``kind=database`` row (managed Postgres) — never runnable (§1.2)."""
    fields = dict(
        id="db-1",
        service_name="pg",
        name="pg",
        kind=JobKind.database,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "postgres:16", "engine": "postgres"}),
        submitted_by_token=owner,
    )
    fields.update(over)
    return Job(**fields)


def _run_result(**over) -> RunResult:
    """A settled :class:`RunResult` — what a stubbed ``run_once`` hands back."""
    fields = dict(
        run_id="run-abc123",
        exit_code=0,
        timed_out=False,
        oom_killed=False,
        duration_s=1.25,
        started_at="2026-08-05T10:00:00+00:00",
        finished_at="2026-08-05T10:00:01+00:00",
        log_tail=["running migrations", "done"],
    )
    fields.update(over)
    return RunResult(**fields)


def _run_controller(queries, *, services: ServicesSettings | None = None):
    """A REAL ``ServiceController`` for the run route (registry + slots live).

    Sibling of :func:`_real_controller` (which the purge tests use) with the
    ``[services]`` knobs exposed, since the D-P20-2 caps are read once at
    construction.
    """
    from nerdit.core.services import ServiceController

    return ServiceController(queries=queries, runtime=StubRuntime(), services_settings=services)


def _run_env(
    job: Job | None = None,
    *,
    run_once=None,
    services: ServicesSettings | None = None,
    cap: int | None = None,
    active: int = 0,
    **app_kw,
):
    """Wire an app + queries + real controller for ``POST /…/run``.

    ``job`` defaults to the tok-sub-owned service row. ``run_once`` becomes an
    ``AsyncMock`` returning :func:`_run_result`, or — when given — one whose
    ``side_effect`` is the supplied exception instance / coroutine function, to
    drive a failure path. ``cap``/``active`` script ``count_active_jobs_public``
    (the route's quota read). Returns ``(app, queries, controller)``.
    """
    q = _queries(_service("tok-sub") if job is None else job)
    q.count_active_jobs_public = AsyncMock(return_value=(active, cap))
    app = _make_app(q, **app_kw)
    if services is not None:
        app.state.settings.services = services
    controller = _run_controller(q, services=services)
    controller.run_once = (
        AsyncMock(return_value=_run_result())
        if run_once is None
        else AsyncMock(side_effect=run_once)
    )
    app.state.service_controller = controller
    return app, q, controller


def _run_body(**over) -> dict:
    base = {"command": ["alembic", "upgrade", "head"]}
    base.update(over)
    return base


def _post_run(app, raw: str = SUB_RAW, ident: str = "svc-1", **kw):
    headers = {**_auth(raw), **kw.pop("headers", {})}
    return TestClient(app, raise_server_exceptions=False).post(
        f"/services/{ident}/run", json=kw.pop("json", _run_body()), headers=headers, **kw
    )


# --- run: happy path ----------------------------------------------------------


def test_run_happy_path_returns_the_settled_outcome():
    """A submitter-owner run returns 200 (nothing is created) with the full
    outcome, and the controller is awaited with exactly the caller's argv."""
    app, q, controller = _run_env()
    resp = _post_run(app, json={"command": ["alembic", "upgrade", "head"], "timeout_s": 120})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "service_name": "demo",
        "run_id": "run-abc123",
        "exit_code": 0,
        "timed_out": False,
        "oom_killed": False,
        "duration_s": 1.25,
        "started_at": "2026-08-05T10:00:00+00:00",
        "finished_at": "2026-08-05T10:00:01+00:00",
        "log_tail": ["running migrations", "done"],
    }
    call = controller.run_once.await_args
    assert call.args[0].id == "svc-1"
    assert call.kwargs["command"] == ["alembic", "upgrade", "head"]
    assert call.kwargs["timeout_s"] == 120
    assert call.kwargs["env_overrides"] is None
    # Defaults ride through: schema default 200 == _MAX_RUN_LOG_TAIL, so the
    # route's clamp is a no-op here (its teeth are pinned below).
    assert call.kwargs["log_tail"] == 200


def test_run_nonzero_exit_is_a_200_not_an_error():
    """A failing command is DATA, not an HTTP error — the run happened."""
    app, _q, controller = _run_env()
    controller.run_once = AsyncMock(return_value=_run_result(exit_code=1, log_tail=["boom"]))
    resp = _post_run(app)
    assert resp.status_code == 200
    body = resp.json()
    assert body["exit_code"] == 1
    assert body["log_tail"] == ["boom"]


def test_run_clamps_log_tail_to_the_server_maximum(monkeypatch):
    """The schema's ``le`` is not the last line of defence: the route re-clamps
    against ``_MAX_RUN_LOG_TAIL`` before handing the value to the controller."""
    from nerdit.daemon.routes import service_run as run_mod

    monkeypatch.setattr(run_mod, "_MAX_RUN_LOG_TAIL", 5)
    app, _q, controller = _run_env()
    resp = _post_run(app, json=_run_body(log_tail=200))
    assert resp.status_code == 200
    assert controller.run_once.await_args.kwargs["log_tail"] == 5


def test_run_resolves_by_service_name_and_forwards_env_overrides():
    """``{ident}`` accepts the name, and the caller's env overlay is forwarded
    verbatim to the controller (the filtering is the controller's contract)."""
    q = _queries(None)
    q.get_service_by_name = AsyncMock(return_value=_service("tok-sub"))
    q.count_active_jobs_public = AsyncMock(return_value=(0, None))
    app = _make_app(q)
    controller = _run_controller(q)
    controller.run_once = AsyncMock(return_value=_run_result())
    app.state.service_controller = controller
    resp = _post_run(app, ident="demo", json=_run_body(env={"DRY_RUN": "1"}))
    assert resp.status_code == 200
    assert controller.run_once.await_args.kwargs["env_overrides"] == {"DRY_RUN": "1"}


# --- run: authorization -------------------------------------------------------


def test_run_readonly_blocked_by_the_coarse_gate():
    """A readonly token never reaches the route: a run executes code."""
    app, _q, controller = _run_env()
    resp = _post_run(app, RO_RAW)
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    controller.run_once.assert_not_awaited()


def test_run_non_owner_submitter_forbidden():
    """Owner-or-admin, like every other service write — a foreign submitter
    must not execute commands inside someone else's image."""
    app, _q, controller = _run_env()
    resp = _post_run(app, OTHER_RAW)
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    controller.run_once.assert_not_awaited()


def test_run_admin_may_run_against_another_owners_service():
    app, _q, controller = _run_env()
    assert _post_run(app, ADMIN_RAW).status_code == 200
    controller.run_once.assert_awaited_once()


def test_run_unknown_ident_404():
    app, q, controller = _run_env()
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(return_value=None)
    resp = _post_run(app, ident="nope")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    controller.run_once.assert_not_awaited()


# --- run: refusals decidable from the row alone (§1.2) ------------------------


def test_run_on_a_model_row_is_not_supported():
    """A model's image/command are platform-managed, never the caller's."""
    app, _q, controller = _run_env(job=_model())
    resp = _post_run(app, ident="mdl-1")
    assert resp.status_code == 422
    assert resp.json()["code"] == "run.not_supported"
    controller.run_once.assert_not_awaited()


def test_run_on_a_database_row_is_not_supported():
    """The database leg of the kind guard — a P20 narrowing the p14 plan was
    silent on. Tested separately from the model leg on purpose: a guard written
    as ``kind is JobKind.model`` would pass the model test and let a caller
    exec into a managed Postgres with its minted credentials in env."""
    app, _q, controller = _run_env(job=_database())
    resp = _post_run(app, ident="db-1")
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "run.not_supported"
    assert "database" in body["message"]
    controller.run_once.assert_not_awaited()


def test_run_timeout_above_the_server_cap_is_422():
    app, _q, controller = _run_env(services=ServicesSettings(run_timeout_max_s=60))
    resp = _post_run(app, json=_run_body(timeout_s=61))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "run.timeout_too_large"
    # The hint carries the cap AND the config key, so the caller can act on it.
    assert "60" in body["hint"] and "run_timeout_max_s" in body["hint"]
    controller.run_once.assert_not_awaited()


def test_run_at_exactly_the_server_cap_is_accepted():
    """The cap is inclusive (``>``, not ``>=``) — a boundary flip would make the
    documented maximum itself unusable."""
    app, _q, _controller = _run_env(services=ServicesSettings(run_timeout_max_s=60))
    assert _post_run(app, json=_run_body(timeout_s=60)).status_code == 200


def test_run_without_an_image_is_409():
    """A never-deployed service has nothing to run."""
    app, _q, controller = _run_env(job=_service("tok-sub", config=json.dumps({"port": 8000})))
    resp = _post_run(app)
    assert resp.status_code == 409
    assert resp.json()["code"] == "run.no_image"
    controller.run_once.assert_not_awaited()


def test_run_while_draining_is_refused():
    """A run started mid-drain is orphaned by the re-exec that follows it."""
    app, _q, controller = _run_env()
    controller.draining = True
    resp = _post_run(app)
    assert resp.status_code == 409
    assert resp.json()["code"] == "daemon.restart_in_progress"
    controller.run_once.assert_not_awaited()


def test_run_is_single_flight_per_service():
    """The route consults the REAL registry, so a live run (or release) refuses
    the next one before any container work — and the refusal lifts the moment
    the slot is discarded."""
    app, _q, controller = _run_env()
    controller._register_run("svc-1", "run-1", is_release=False)
    resp = _post_run(app)
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    controller.run_once.assert_not_awaited()

    controller._discard_run("svc-1", "run-1")
    assert _post_run(app).status_code == 200


# --- run: everything the controller raises maps to the §1.2 table -------------


def test_run_not_ready_puts_not_ready_kind_at_the_envelope_top_level():
    """``not_ready_kind`` is an EXTRA envelope key, never nested under ``detail``.

    Past-review finding: passing it as ``detail=`` would replace the
    backward-compat ``detail`` alias ``_envelope`` always populates, so the
    agent-facing contract is pinned on both sides — the kind is readable at the
    top level AND ``detail`` still carries the message string.
    """
    app, _q, _controller = _run_env(
        run_once=LaunchEnvNotReady("secrets", "secret 'DB_PASSWORD' is not set")
    )
    resp = _post_run(app)
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "run.not_ready"
    assert body["not_ready_kind"] == "secrets"
    # The alias is intact and is the message, not a nested object.
    assert body["detail"] == body["message"]
    assert isinstance(body["detail"], str)


def test_run_not_ready_kind_is_the_exceptions_own_kind():
    """The kind is forwarded verbatim (binding | secrets | shared_secrets), so
    an agent can branch on it."""
    app, _q, _controller = _run_env(
        run_once=LaunchEnvNotReady("binding", "[ai.default] model is not running")
    )
    assert _post_run(app).json()["not_ready_kind"] == "binding"


def test_run_volume_spec_failure_is_422():
    app, _q, _controller = _run_env(run_once=VolumeSpecError("bad spec"))
    resp = _post_run(app)
    assert resp.status_code == 422
    assert resp.json()["code"] == "run.volume_invalid"


def test_run_runtime_failure_is_503_and_path_free():
    """A wedged runtime is a 503 — and the runtime's own message (which
    routinely carries host paths and socket locations) never reaches the
    caller."""
    app, _q, _controller = _run_env(
        run_once=ContainerRuntimeError("cannot connect to /var/run/docker.sock")
    )
    resp = _post_run(app)
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "run.runtime_unavailable"
    assert "/var/run/docker.sock" not in json.dumps(body)


def test_run_sandbox_denial_is_422_never_503():
    """LOAD-BEARING subclass-ordering guard — do not delete.

    ``SandboxViolationError`` ⊂ ``ContainerStartError`` ⊂
    ``ContainerRuntimeError``, so if a future edit moves the
    ``except ContainerRuntimeError`` handler above the ``except
    SandboxViolationError`` one, the sandbox clause becomes dead code and every
    policy denial is answered with a misleading 503 ("the runtime is down")
    instead of the 422 that tells the caller their request was refused. Nothing
    else in the suite catches that reordering: both handlers raise a structured
    envelope, so only the code + status distinguish them.
    """
    app, _q, _controller = _run_env(
        run_once=SandboxViolationError("host path denied", reason="tier_a", path="/etc")
    )
    resp = _post_run(app)
    assert resp.status_code == 422, "sandbox denial fell through to the ContainerRuntimeError arm"
    assert resp.json()["code"] == "run.sandbox_denied"

    # Control: the immediate PARENT class has no handler of its own, so it lands
    # on the 503 arm. That is what proves the 422 above comes from the dedicated
    # subclass clause and not from some broader start-error special case.
    app2, _q2, _c2 = _run_env(run_once=ContainerStartError("image entrypoint missing"))
    sibling = _post_run(app2)
    assert sibling.status_code == 503
    assert sibling.json()["code"] == "run.runtime_unavailable"


def test_run_interrupted_is_distinguishable_from_a_runtime_that_never_started():
    """The SECOND load-bearing subclass-ordering guard — do not delete.

    ``RunInterruptedError`` ⊂ ``ContainerRuntimeError`` too, and it means the
    container **did** start and the runtime then lost it. If a future edit moves
    the base handler above this one, a migration that ran for two minutes is
    reported as "the container runtime could not execute the run" — the
    operator concludes nothing happened and runs it again against a
    half-migrated schema. ``core/app_build.py`` already orders it this way on
    the release path for exactly this reason.

    The scrubbed tail rides along as an extra envelope key because it is the
    only surviving evidence of how far the command got.
    """
    app, _q, _controller = _run_env(
        run_once=RunInterruptedError(
            "container vanished mid-wait",
            log_tail=["Running upgrade abc -> def", "INFO  [alembic] done"],
        )
    )
    resp = _post_run(app)
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "run.interrupted", "interrupted run collapsed into the generic arm"
    assert body["container_started"] is True
    assert body["log_tail"] == ["Running upgrade abc -> def", "INFO  [alembic] done"]
    # The message must not claim the run never happened.
    assert "PARTIALLY" in body["hint"]
    # And it must NOT send the operator to `last_run`: run_once raises before
    # reaching set_last_run, so /diagnose still shows the PREVIOUS run. This
    # envelope's log_tail is the only surviving record.
    assert "last_run" not in body["hint"]


def test_run_argv_with_a_nul_byte_is_rejected():
    """NUL in argv breaks the audit fingerprint's injectivity — reject it.

    ``_command_digest`` joins argv on NUL precisely because execve cannot carry
    one, which is what makes the digest able to prove WHICH argv ran. But JSON
    happily encodes ``"a\\u0000b"``, so without this check ``["a\\0b"]`` and
    ``["a", "b"]`` hash identically and the property quietly dies. Other C0
    controls stay legal: a newline inside one argument is ordinary
    (``python -c "line1\\nline2"``).
    """
    app, _q, controller = _run_env()
    resp = _post_run(app, json={"command": ["a\x00b"], "timeout_s": 30})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    controller.run_once.assert_not_awaited()

    # A newline is NOT rejected — banning it would break a normal migration.
    ok = _post_run(app, json={"command": ["python", "-c", "print(1)\nprint(2)"], "timeout_s": 30})
    assert ok.status_code == 200


def test_run_too_many_in_flight_is_409_with_the_cap_in_the_hint():
    app, _q, _controller = _run_env(
        run_once=RunPreconditionError("too_many_runs"),
        services=ServicesSettings(max_concurrent_runs=2),
    )
    resp = _post_run(app)
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "run.too_many_in_flight"
    assert "2" in body["hint"] and "max_concurrent_runs" in body["hint"]


def test_run_precondition_service_gone_is_a_404():
    """The row vanished (or was terminally stopped) between resolve and launch."""
    app, _q, _controller = _run_env(run_once=RunPreconditionError("service_gone"))
    resp = _post_run(app)
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


def test_run_precondition_no_image_is_409():
    """The controller's authoritative re-check (image pruned since the fast
    pre-flight) lands on the same code the pre-flight uses."""
    app, _q, _controller = _run_env(run_once=RunPreconditionError("no_image"))
    resp = _post_run(app)
    assert resp.status_code == 409
    assert resp.json()["code"] == "run.no_image"


def test_run_unknown_precondition_reason_is_still_structured():
    """Invariant #3: a reason added at the controller tier and not yet mapped
    must degrade to a structured 409, never escape as a bare 500."""
    app, _q, _controller = _run_env(run_once=RunPreconditionError("brand_new_reason"))
    resp = _post_run(app)
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"


def test_run_without_a_controller_is_503():
    """No run subsystem ⇒ the honest answer is 503, not a 500 traceback."""
    q = _queries(_service("tok-sub"))
    app = _make_app(q)
    assert getattr(app.state, "service_controller", None) is None
    resp = _post_run(app)
    assert resp.status_code == 503
    assert resp.json()["code"] == "run.runtime_unavailable"


# --- run: quota (the row-less claim, D-C) -------------------------------------


def test_run_quota_exceeded_releases_the_claimed_slot():
    """A run is charged against ``max_concurrent_jobs`` without inserting a row,
    so the claim is the route's — and the refusal path must release it. A leaked
    claim would permanently shrink the token's effective quota."""
    app, q, controller = _run_env(cap=1, active=1)
    resp = _post_run(app)
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "quota_exceeded"
    assert body["limit"] == 1 and body["current"] == 2
    controller.run_once.assert_not_awaited()
    q.count_active_jobs_public.assert_awaited_once_with("tok-sub")
    # The slot was released on the SAME finally that covers success — the
    # counter is back to empty, not stuck at 1.
    assert controller._run_counts == {}


def test_run_releases_the_slot_on_success_and_on_a_controller_raise():
    """One release, in one finally, covering every exit."""
    app, _q, controller = _run_env()
    assert _post_run(app).status_code == 200
    assert controller._run_counts == {}

    controller.run_once = AsyncMock(side_effect=ContainerRuntimeError("docker is wedged"))
    assert _post_run(app).status_code == 503
    assert controller._run_counts == {}


def test_run_uncapped_token_never_refused_on_quota():
    """``max_concurrent_jobs = None`` (uncapped) short-circuits the comparison."""
    app, _q, _controller = _run_env(cap=None, active=99)
    assert _post_run(app).status_code == 200


async def test_run_two_concurrent_requests_against_cap_one_yield_exactly_one_403():
    """The claim is SYNC and lands before the DB read, so a second concurrent
    request sees the first one's pending run.

    Without the pre-read increment both requests read ``active=0``, both compare
    ``0 + 1 <= 1`` and both execute — one token running twice its cap. Scheduled
    with ``asyncio.gather`` (the ``test_truly_concurrent_submits_respect_cap``
    idiom) so they really interleave at the quota read's await.
    """
    import asyncio

    from httpx import ASGITransport, AsyncClient

    app, q, controller = _run_env(cap=1, active=0, job=_service("tok-sub"))

    async def _slow_count(_token_id):
        await asyncio.sleep(0.02)  # both requests have claimed by now
        return (0, 1)

    q.count_active_jobs_public = AsyncMock(side_effect=_slow_count)
    # Two runs on one service would also trip single-flight, so the stub keeps
    # the registry out of it: this test is about the quota claim alone.
    controller.run_once = AsyncMock(return_value=_run_result())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        responses = await asyncio.gather(
            *(
                client.post("/services/svc-1/run", json=_run_body(), headers=_auth(SUB_RAW))
                for _ in range(2)
            )
        )
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 403], [r.text for r in responses]
    refused = [r for r in responses if r.status_code == 403][0]
    assert refused.json()["code"] == "quota_exceeded"
    assert controller.run_once.await_count == 1
    assert controller._run_counts == {}


def test_run_tokenless_principal_skips_the_claim_but_not_the_global_cap():
    """The legacy global token is not a row in ``api_tokens``, so there is no
    per-token cap to charge against and nothing to claim — but the D-P20-2
    daemon-wide cap is what keeps it from being unbounded, so it must still
    bind. Exercised through the REAL registry, not a stubbed refusal."""
    app, q, controller = _run_env(services=ServicesSettings(max_concurrent_runs=1))
    # Someone else's run already fills the daemon-wide budget of 1.
    controller._register_run("other-job", "run-elsewhere", is_release=False)

    async def _really_register(job, **_kw):
        controller._register_run(job.id, "run-mine", is_release=False)
        return _run_result()  # pragma: no cover — the register above raises

    controller.run_once = AsyncMock(side_effect=_really_register)

    resp = _post_run(app, LEGACY)
    assert resp.status_code == 409
    assert resp.json()["code"] == "run.too_many_in_flight"
    # No per-token quota read, no per-token slot: nothing to claim or release.
    q.count_active_jobs_public.assert_not_awaited()
    assert controller._run_counts == {}


# --- run: audit (D-P20-5 — the argv NEVER enters the trail) -------------------


def test_run_audit_records_the_command_fingerprint_never_the_argv():
    """D-P20-5, the core redaction contract — pin it hard.

    Audit params are kept forever (archive-first retention) and ride into every
    ``POST /system/backup`` tar, so the audit row carries a *fingerprint* trio
    (argv0 + length + SHA-256 over the NUL-joined argv) and never the argv
    itself: a credential mistakenly passed as an argument must not become a
    permanent at-rest copy. The verbatim argv lives only in the owner-gated
    ``config['last_run']``.
    """
    import hashlib

    command = ["psql", "-c", "GRANT ALL TO bob PASSWORD 'hunter2'"]
    app, q, _controller = _run_env(with_audit=True)
    resp = _post_run(app, json={"command": command, "env": {"PGPASSWORD": "s3cr3t-value"}})
    assert resp.status_code == 200

    rows = [
        c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == "service.run"
    ]
    assert rows, "expected one service.run audit row from the middleware"
    kwargs = rows[-1].kwargs
    assert kwargs["result"] == "ok"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "svc-1"
    params = json.loads(kwargs["params_redacted"])

    # The fingerprint trio is present and correct...
    assert params["command_argv0"] == "psql"
    assert params["command_len"] == 3
    assert params["command_sha256"] == hashlib.sha256("\0".join(command).encode()).hexdigest()
    # ...and the argv itself is absent, in every shape.
    assert "command" not in params
    assert "hunter2" not in kwargs["params_redacted"]
    assert "GRANT ALL" not in kwargs["params_redacted"]
    # Env KEY names survive (an auditor needs them); values are masked.
    assert params["env"] == {"PGPASSWORD": "***"}
    assert "s3cr3t-value" not in kwargs["params_redacted"]
    assert params["timeout_s"] == 300


def test_run_started_row_is_written_before_the_container_runs():
    """Security S7a: a run that dies with the daemon (kill -9, host OOM) leaves
    no response and no ``last_run`` stamp — but it DID execute, so the durable
    evidence that a command was run must not depend on the run finishing.

    Asserted from inside the controller call, so an edit moving the
    ``record_out_of_band`` below ``run_once`` fails here rather than passing on
    a post-hoc ordering check.
    """
    app, q, controller = _run_env(with_audit=True)
    seen_at_exec: list[str] = []

    async def _observe(_job, **_kw):
        seen_at_exec.extend(c.kwargs.get("action") for c in q.insert_audit_log.await_args_list)
        return _run_result()

    controller.run_once = AsyncMock(side_effect=_observe)
    resp = _post_run(app, json=_run_body(env={"MODE": "dry"}))
    assert resp.status_code == 200
    assert "service.run_started" in seen_at_exec

    started = [
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "service.run_started"
    ]
    assert len(started) == 1
    params = json.loads(started[-1].kwargs["params_redacted"])
    assert params["command_argv0"] == "alembic"
    assert params["command_len"] == 3
    assert params["command_sha256"]
    assert params["timeout_s"] == 300
    # Override KEY names only — no values, and again no argv. Named
    # *_submitted deliberately: this row is written before the controller runs,
    # so the route cannot yet know which overrides survive the D-P14-5
    # protected-key filter. The applied/dropped split lands in `last_run`.
    assert params["env_override_keys_submitted"] == ["MODE"]
    assert "env_override_keys" not in params, "the route must not imply these were applied"
    assert "command" not in params


def test_run_denial_is_audited_as_denied():
    """A refused run is still a recorded attempt (Invariant #3)."""
    app, q, _controller = _run_env(with_audit=True)
    assert _post_run(app, OTHER_RAW).status_code == 403
    rows = [
        c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == "service.run"
    ]
    assert rows and rows[-1].kwargs["result"] == "denied"


# --- run: idempotency (D-P20-6 — a run body is never cached) ------------------


def _idem_store(q: AsyncMock) -> dict:
    """A principal-scoped in-memory idempotency store on the queries mock.

    Mirrors the real table's key — ``(principal_id, idem_key)`` — which is the
    whole point of the second-principal test below: a shared key must not be a
    cross-tenant replay.
    """
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


def test_run_replay_returns_the_envelope_and_never_reexecutes():
    """D-P20-6: ``service.run`` is in ``NO_BODY_CACHE_ACTIONS``.

    A run body carries arbitrary container stdout; caching it would make the
    idempotency store a THIRD at-rest copy of that output, kept for 24 h and
    captured by every backup tar. So the replay hands back the non-secret
    ``idempotent_replay`` envelope — and, being a replay, does not run the
    command a second time.
    """
    app, q, controller = _run_env(with_idempotency=True)
    records = _idem_store(q)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {**_auth(SUB_RAW), "Idempotency-Key": "run-key-1"}

    first = client.post("/services/svc-1/run", json=_run_body(), headers=headers)
    assert first.status_code == 200
    assert first.json()["log_tail"] == ["running migrations", "done"]

    second = client.post("/services/svc-1/run", json=_run_body(), headers=headers)
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json()["code"] == "idempotent_replay"
    assert "log_tail" not in second.json()
    assert controller.run_once.await_count == 1, "a replay must not execute the command again"
    # The envelope must explain the RUN reason, not the minted-credential one:
    # "secret values are shown only once" sends a run caller hunting for a
    # secret that never existed. It must also say the command did not re-run.
    replay = second.json()
    assert "secret" not in replay["message"].lower()
    assert "never cached" in replay["message"]
    assert "NOT re-run" in replay["hint"] and "last_run" in replay["hint"]
    # The replay must NAME the run it replayed. `_extract_resource_id` looked
    # only for `id`, which a run body does not have — so the envelope was
    # anonymous and the caller's only recovery was "the most recent run on this
    # service", which a concurrent run has already made wrong. The hint must
    # therefore also warn that `last_run` is the latest run, not necessarily
    # this one.
    assert replay["resource_id"] == _run_result().run_id
    assert "MOST RECENT" in replay["hint"]

    # Nothing of the run output was persisted at rest.
    stored = records[("tok-sub", "run-key-1")]
    assert stored.response_body is None
    assert "running migrations" not in json.dumps(vars(stored), default=str)


def test_run_in_progress_key_is_409():
    """A second request under a key still executing is refused, not queued."""
    app, q, controller = _run_env(with_idempotency=True)
    _idem_store(q)
    # Seed the claim exactly as the middleware would for an in-flight request.
    import asyncio

    asyncio.run(
        q.insert_idempotency_inprogress(
            principal_id="tok-sub",
            idem_key="run-key-2",
            method="POST",
            path="/services/svc-1/run",
            expires_at="2099-01-01T00:00:00+00:00",
        )
    )
    resp = TestClient(app, raise_server_exceptions=False).post(
        "/services/svc-1/run",
        json=_run_body(),
        headers={**_auth(SUB_RAW), "Idempotency-Key": "run-key-2"},
    )
    assert resp.status_code == 409
    assert resp.json()["code"] == "idempotency_in_progress"
    controller.run_once.assert_not_awaited()


def test_run_same_key_from_a_second_principal_executes_independently():
    """Idempotency records are PRINCIPAL-scoped: one caller's key must never
    replay (or block) another caller's run. Two agents minting ``uuid``s is the
    happy case; a shared literal key is the one that would collide."""
    app, q, controller = _run_env(with_idempotency=True)
    records = _idem_store(q)
    client = TestClient(app, raise_server_exceptions=False)
    key = {"Idempotency-Key": "same-key"}

    owner = client.post("/services/svc-1/run", json=_run_body(), headers={**_auth(SUB_RAW), **key})
    admin = client.post(
        "/services/svc-1/run", json=_run_body(), headers={**_auth(ADMIN_RAW), **key}
    )
    assert owner.status_code == 200 and admin.status_code == 200
    assert "Idempotent-Replay" not in admin.headers
    assert controller.run_once.await_count == 2
    assert set(records) == {("tok-sub", "same-key"), ("tok-admin", "same-key")}


def test_run_failure_is_not_pinned_so_the_key_can_be_retried():
    """A non-2xx never pins the key: the row is deleted, so the caller may retry
    the same key once the cause is fixed."""
    app, q, controller = _run_env(
        with_idempotency=True, run_once=ContainerRuntimeError("docker is wedged")
    )
    records = _idem_store(q)
    client = TestClient(app, raise_server_exceptions=False)
    headers = {**_auth(SUB_RAW), "Idempotency-Key": "run-key-3"}
    assert client.post("/services/svc-1/run", json=_run_body(), headers=headers).status_code == 503
    assert records == {}

    controller.run_once = AsyncMock(return_value=_run_result())
    assert client.post("/services/svc-1/run", json=_run_body(), headers=headers).status_code == 200


# --- run: the request boundary (schema caps + the S16 env-key grammar) -------


@pytest.mark.parametrize(
    "body, why",
    [
        ({"command": []}, "empty argv"),
        ({"command": ["alembic", ""]}, "empty argument"),
        ({"command": ["a"] * 257}, "argv cardinality cap"),
        ({"command": ["a"], "timeout_s": 0}, "timeout floor"),
        ({"command": ["a"], "log_tail": 0}, "log_tail floor"),
        ({"command": ["a"], "log_tail": 201}, "log_tail ceiling"),
        ({"command": ["a"], "env": {"A=B": "C"}}, "S16: '=' in an env key"),
        ({"command": ["a"], "env": {"PATH\n": "x"}}, "S16: newline in an env key"),
        ({"command": ["a"], "env": {"9LIVES": "x"}}, "S16: leading digit"),
        ({"command": ["a"], "env": {"": "x"}}, "S16: empty key"),
    ],
)
def test_run_request_boundary_rejects_before_any_container_work(body, why):
    """Every schema cap 422s at the boundary, controller untouched.

    The env-key cases are the S16 security control, not cosmetics: the overlay
    reaches the runtime as ``KEY=VALUE`` strings, so ``{"A=B": "C"}`` would
    serialize to ``A=B=C`` and set ``A`` to ``B=C`` — smuggling a value past
    the controller's by-name protected-key filter (``PORT``,
    ``NERDIT_RUN_ID``, every actually-injected ``[ai.*]``/``[db.*]`` key).
    """
    app, _q, controller = _run_env()
    resp = _post_run(app, json=body)
    assert resp.status_code == 422, why
    assert resp.json()["code"] == "validation_error"
    controller.run_once.assert_not_awaited()


def test_run_invalid_env_key_422_does_not_echo_the_env_value():
    """A 422 must not hand the caller's own submitted secret back.

    Was an xfail; closed by the Codex round. A ``field_validator`` rejection
    makes pydantic attach the offending ``input`` — here the whole ``env`` map,
    values included — and ``_validation_exception_handler`` renders
    ``exc.errors()`` into BOTH ``detail`` and ``diagnostics``. Same-caller and
    never persisted, but an agent or CLI that logs the envelope writes the
    secret into its transcript, which is exactly what the credential gate
    forbids: no secret value in a response body or an error message. The
    handler now masks ``input`` for secret-bearing field names
    while keeping type/loc/msg intact.

    The REJECTED key is masked too, and that supersedes the earlier round's
    "the key name is not a secret, keep it" pin. It held while ``msg`` went
    only to the same caller over HTTP; it stopped holding when
    ``mcp/errors.py`` began copying ``msg`` into an agent transcript — and
    the pin's premise is false on this branch anyway, because a key that
    FAILED the name check is by definition not a name (an agent that swaps a
    key and a value lands exactly here). The ordinal replaces it: value-free,
    and more than ``loc``'s bare ``env`` gives.
    """
    app, _q, _controller = _run_env()
    resp = _post_run(app, json={"command": ["a"], "env": {"BAD=KEY": "s3cr3t-value"}})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    assert "s3cr3t-value" not in resp.text
    # The diagnostic itself must survive — masking the value, not the error.
    body = resp.json()
    assert body["diagnostics"], "masking must not empty the pydantic error list"
    assert any("env" in (e.get("loc") or []) for e in body["diagnostics"])
    assert "BAD=KEY" not in resp.text
    assert "position 1" in resp.text


def test_run_sibling_missing_field_422_does_not_echo_the_env_value():
    """A ``missing`` error on a SIBLING field must not leak ``env`` values.

    Security-review follow-up on the loc-based mask: pydantic attaches the
    WHOLE parent dict as the ``input`` of a ``missing``-type error, and its
    ``loc`` names only the missing field (``command``) — no secret-bearing
    segment, so the loc test alone let the submitted ``env`` map through
    verbatim. The handler now also walks mapping-shaped inputs and masks any
    value under a secret-bearing key, keeping the non-secret siblings echoed.
    """
    app, _q, _controller = _run_env()
    resp = _post_run(app, json={"env": {"STRIPE_API_KEY": "sk_live_leak"}, "timeout_s": 5})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    assert "sk_live_leak" not in resp.text
    body = resp.json()
    missing = [e for e in body["diagnostics"] if e.get("type") == "missing"]
    assert missing, "the missing-command error itself must survive the mask"
    # The non-secret sibling stays echoed — the mask is key-scoped, not wholesale.
    assert any((e.get("input") or {}).get("timeout_s") == 5 for e in missing)
    assert all((e.get("input") or {}).get("env") == "***" for e in missing)


def test_run_unknown_field_422_does_not_echo_its_value():
    """A mistyped secret-bearing field name must not leak its map.

    P22 WP-C made every route body strict, which adds an ``extra_forbidden``
    error carrying the unknown field's full ``input``. The loc-based mask cannot
    help: the whole point is that ``envv`` is NOT a name the model knows, so it
    matches nothing in ``_SECRET_INPUT_FIELDS``. ``extra_forbidden`` inputs are
    therefore masked unconditionally — the ``loc`` names the rejected key, which
    is the entire diagnostic value of that error.
    """
    app, _q, _controller = _run_env()
    resp = _post_run(app, json={"command": ["a"], "envv": {"STRIPE_API_KEY": "sk_live_leak"}})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"
    assert "sk_live_leak" not in resp.text
    extra = [e for e in resp.json()["diagnostics"] if e.get("type") == "extra_forbidden"]
    assert extra, "the unknown-field error itself must survive the mask"
    assert all("envv" in (e.get("loc") or []) for e in extra)


def test_run_a_non_secret_field_still_echoes_its_input():
    """The mask is field-scoped — it must not blind every other 422.

    ``input`` is what tells a caller which value was rejected; dropping it
    wholesale would degrade every validation error in the daemon to guesswork.
    Only the secret-bearing field names are masked.
    """
    app, _q, _controller = _run_env()
    resp = _post_run(app, json={"command": ["a"], "timeout_s": 0})
    assert resp.status_code == 422
    assert any(
        e.get("input") == 0
        for e in resp.json()["diagnostics"]
        if "timeout_s" in (e.get("loc") or [])
    ), "a non-secret field must still report the rejected value"


# --- run: the /diagnose passthrough ------------------------------------------


def _last_run_blob(**over) -> dict:
    blob = {
        "run_id": "run-abc123",
        "command": ["alembic", "upgrade", "head"],
        "exit_code": 0,
        "timed_out": False,
        "oom_killed": False,
        "started_at": "2026-08-05T10:00:00+00:00",
        "finished_at": "2026-08-05T10:00:01+00:00",
        "duration_s": 1.25,
        "env_override_keys": ["MODE"],
        "log_tail": ["running migrations", "done"],
    }
    blob.update(over)
    return blob


def _diagnose(job: Job):
    q = _queries(job)
    q.get_logs = AsyncMock(return_value=[])
    return TestClient(_make_app(q), raise_server_exceptions=False).get(
        "/services/svc-1/diagnose", headers=_auth(SUB_RAW)
    )


def test_diagnose_passes_last_run_through():
    """``/diagnose`` is the recovery path for a run whose response was lost — and
    the only place the verbatim argv is readable (owner-or-admin gated)."""
    blob = _last_run_blob()
    resp = _diagnose(_service("tok-sub", config=json.dumps({"image": "x:1", "last_run": blob})))
    assert resp.status_code == 200, resp.text
    assert resp.json()["last_run"] == blob


def test_diagnose_last_run_is_null_when_absent_or_not_a_dict():
    """Absent → null. A non-dict blob (a legacy string, a corrupted config)
    ALSO projects null rather than 500ing the whole diagnostic: the passthrough
    guards on ``isinstance(..., dict)``, which is why the fixture above stamps a
    dict and not a string."""
    plain = _diagnose(_service("tok-sub", config=json.dumps({"image": "x:1"})))
    assert plain.status_code == 200
    assert plain.json()["last_run"] is None

    junk = _diagnose(
        _service("tok-sub", config=json.dumps({"image": "x:1", "last_run": "run-abc123"}))
    )
    assert junk.status_code == 200
    assert junk.json()["last_run"] is None


# --- run: the round-5 adversarial review fixes ---------------------------------


def test_run_against_a_stopped_service_is_an_honest_409_not_a_404():
    """A stopped-but-listed service must not answer "No service".

    ``_TERMINAL_DESIRED_STATES`` includes plain ``stopped``, and the controller
    refuses those as ``service_gone`` — which §1.2 maps to 404 for the DELETE
    race. Applied to a row the user merely stopped, that 404 is a lie the
    operator disproves in one command: the canonical maintenance flow
    ``services stop app`` then ``run app -- migrate`` answered "No service
    'app'" while ``services list`` showed it, and an agent seeing 404 from run
    but 200 from GET /services concludes resolution is broken.
    """
    app, _q, controller = _run_env(job=_service("tok-sub", desired_state="stopped"))
    resp = _post_run(app)
    assert resp.status_code == 409, "a stopped service must not 404"
    body = resp.json()
    assert body["code"] == "run.not_ready"
    assert body["not_ready_kind"] == "service_stopped"
    assert "stopped" in body["message"]
    controller.run_once.assert_not_awaited()


def test_run_audit_bounds_argv0_so_a_one_element_argv_cannot_leak_a_command_line():
    """argv0 is a program NAME in the trail — never a whole command line.

    ``command_sha256`` carries the correlation value precisely so the argv
    itself stays out of rows that outlive the retention prune and ride into
    every backup tar (D-P20-5). A caller who passes one shell string instead of
    a vector makes argv[0] the entire command line, credentials included, which
    silently converts the fingerprint design back into verbatim storage.
    """
    leaky = "psql postgresql://app:P@ssw0rd@db/app -c 'select 1'"
    app, q, _c = _run_env(with_audit=True)
    resp = _post_run(app, json={"command": [leaky], "timeout_s": 30})
    assert resp.status_code == 200

    recorded = json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list], default=str)
    assert "P@ssw0rd" not in recorded, "a one-element argv put a credential in the audit trail"

    rows = [
        c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == "service.run"
    ]
    params = json.loads(rows[-1].kwargs["params_redacted"])
    # Here the basename pass alone removes it (the DSN password precedes the
    # last separator), and what survives is bounded.
    assert "P@ssw0rd" not in params["command_argv0"]
    assert len(params["command_argv0"]) <= 80
    # The fingerprint still identifies the exact argv that ran.
    assert params["command_sha256"] == hashlib.sha256(leaky.encode()).hexdigest()

    # A long separator-free argv is the case basename cannot help with: the
    # hard cap is what bounds it, and the marker tells an auditor it was cut
    # rather than letting them read a clamped value as a whole program name.
    app2, q2, _c2 = _run_env(with_audit=True)
    flat = "mytool --password=hunter2-" + "x" * 200  # secret near the FRONT
    assert _post_run(app2, json={"command": [flat], "timeout_s": 30}).status_code == 200
    rows2 = [
        c for c in q2.insert_audit_log.await_args_list if c.kwargs.get("action") == "service.run"
    ]
    argv0 = json.loads(rows2[-1].kwargs["params_redacted"])["command_argv0"]
    # THE assertion that matters, and the one a length-only check misses: a
    # length cap alone keeps a secret that sits near the FRONT of the string.
    # Caught live — the first fix clamped to 64 chars and preserved the
    # password verbatim. Only dropping the argument tail removes it.
    assert "hunter2" not in argv0
    assert argv0.startswith("mytool")
    assert "argv-tail dropped" in argv0
    assert len(argv0) < len(flat)


def test_run_from_a_tunnel_principal_skips_the_row_backed_quota_read():
    """PR #114 review: a synthetic node-link principal (``link:<node_id>``) has
    no ``api_tokens`` row, and ``count_active_jobs_public`` is fail-closed
    (missing row => cap 0) — charging it there refused EVERY tunneled run.
    The route must route link principals around the row-backed read; the
    daemon-wide D-P20-2 cap still binds."""
    from types import SimpleNamespace

    app, q, controller = _run_env(job=_service("link:node-golden"), cap=0, active=0)
    app.state.link_manager = SimpleNamespace(
        validate_capability=lambda token, role: token == "cap-tunnel-token" and role == "submitter",
        status=lambda: SimpleNamespace(node_id="node-golden"),
    )

    resp = TestClient(app, raise_server_exceptions=False).post(
        "/services/svc-1/run",
        json=_run_body(),
        headers={"Authorization": "Bearer cap-tunnel-token"},
    )

    assert resp.status_code == 200, resp.text
    # The fail-closed row read was never consulted for the synthetic principal.
    assert q.count_active_jobs_public.await_count == 0
    controller.run_once.assert_awaited()


# --- NC-0: the caller-relative ``manageable`` projection ---------------------
#
# The cloud console lists a node's services and offers restart/stop/redeploy/
# rollback on the ones it may actually act on. Over the tunnel it is a plain
# ``submitter`` whose ``token_id`` is ``link:<node_id>``, so the actionable set
# is exactly "what arrived over the tunnel" — its own deploys plus anything a
# remote MCP call deployed, which shares the same principal. It cannot compute
# that itself (it does not know who owns a row, and ``submitted_via`` is only
# 'cli'/'dashboard'), so the daemon answers with a boolean. These tests drive
# the REAL routes through the real middleware, because the value depends on the
# resolved principal and a unit test of the mapper would not exercise that.

TUNNEL_CAP = "cap-tunnel-token"
TUNNEL_NODE = "node-golden"
TUNNEL_OWNER = f"link:{TUNNEL_NODE}"


def _tunnel_app(queries: AsyncMock) -> FastAPI:
    """The services app with a link manager that accepts exactly one bearer.

    Mirrors ``test_run_from_a_tunnel_principal_skips_the_row_backed_quota_read``
    — the established way to reach the synthetic principal without standing up
    a relay: ``_link_principal`` only needs ``validate_capability`` to say yes
    and ``status()`` to name the node.
    """
    app = _make_app(queries)
    app.state.link_manager = SimpleNamespace(
        validate_capability=lambda token, role: token == TUNNEL_CAP and role == "submitter",
        # ``state`` is here because the service surfaces also load the P26
        # hosted context off this same duck-typed manager; ``None`` reads as
        # "link not ready", which keeps ``public_urls`` empty and leaves this
        # test about ownership and nothing else.
        status=lambda: SimpleNamespace(node_id=TUNNEL_NODE, state=None),
    )
    return app


def _tunnel_get(queries: AsyncMock, path: str) -> dict:
    client = TestClient(_tunnel_app(queries), raise_server_exceptions=False)
    resp = client.get(path, headers={"Authorization": f"Bearer {TUNNEL_CAP}"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_manageable_is_true_over_the_tunnel_for_a_tunnel_deployed_service():
    """The console's own deploys (and remote-MCP ones) are actionable."""
    assert _tunnel_get(_queries(_service(TUNNEL_OWNER)), "/services/demo")["manageable"] is True


def test_manageable_is_false_over_the_tunnel_for_a_locally_deployed_service():
    """A NULL-owner row (``nerdit deploy`` on the machine, or local mode) is
    admin-only and can never be acted on over the permanently-``submitter``
    tunnel. The console must LIST it — D-NC13 collapses it, never hides it —
    with its actions off, which is what this boolean buys."""
    assert _tunnel_get(_queries(_service(None)), "/services/demo")["manageable"] is False


def test_manageable_is_false_over_the_tunnel_for_another_tokens_service():
    """A service deployed by an operator's own scoped token is equally off-limits."""
    assert _tunnel_get(_queries(_service("tok-sub")), "/services/demo")["manageable"] is False


def test_manageable_is_projected_per_row_on_the_list_route():
    """The list route is the one the console actually renders, and it must carry
    a PER-ROW verdict — the single mapper is what makes that true without the
    route knowing anything about ownership."""
    mine = _service(TUNNEL_OWNER)
    theirs = _service(None, id="svc-2", service_name="local-app", name="local-app")
    q = _queries(mine)
    q.list_services = AsyncMock(return_value=([mine, theirs], None))
    items = _tunnel_get(q, "/services")["items"]
    assert [i["manageable"] for i in items] == [True, False]


def test_manageable_is_true_for_an_admin_on_a_row_it_does_not_own():
    """Admin bypass, same as the write gate — the local dashboard sees every
    service as actionable, which is the pre-NC-0 behaviour it must keep."""
    resp = _client(_queries(_service(None))).get("/services/demo", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["manageable"] is True


def test_manageable_never_exposes_the_owning_token():
    """A boolean, never an identifier. ``GET /services`` is open to every
    authenticated principal, so projecting ``submitted_by_token`` would publish
    one caller's provenance to every other one for no gain (cloud D-NC5).

    The verdict is asserted alongside the two absences on purpose. Absence
    assertions alone pass on a tree with no projection at all — they would have
    gone green before this field existed — so they pin nothing about NC-0 by
    themselves. Pairing them with the boolean makes this a test of *what is
    projected instead of the identifier*, which is the actual claim, and makes
    it fail if the field is ever removed or swapped back for an owner id."""
    body = _tunnel_get(_queries(_service("tok-sub")), "/services/demo")
    assert body["manageable"] is False
    assert "submitted_by_token" not in body
    assert "tok-sub" not in json.dumps(body)


SCOPED_RAW = "scoped-raw"


@contextlib.contextmanager
def _scoped_token(scope: list[str]):
    """Register a scoped ``ApiToken`` in ``_TOKENS`` for the body of one test.

    ``_TOKENS`` is module-global and read by every client in this file, so the
    registration is undone in a ``finally``: a leaked scoped token would
    silently re-scope any later test that reused the raw value.
    """
    _TOKENS[hash_token(SCOPED_RAW)] = ApiToken(
        id="tok-scoped",
        name="sc",
        role=TokenRole.submitter,
        token_hash=hash_token(SCOPED_RAW),
        scope_services=scope,
    )
    try:
        yield
    finally:
        del _TOKENS[hash_token(SCOPED_RAW)]


def test_manageable_honours_a_narrowed_scope_like_the_write_gate_does():
    """Owning the row is not enough: a scope narrowed after creation still binds,
    so a scoped token that owns a row outside its scope would be refused by
    ``require_owner_or_admin`` — and must therefore read ``manageable: false``.
    Computed through ``_owner_or_admin_allows``, so the two cannot drift."""
    with _scoped_token(["other"]):
        resp = _client(_queries(_service("tok-scoped"))).get(
            "/services/demo", headers=_auth(SCOPED_RAW)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["manageable"] is False


def test_manageable_is_true_for_a_scoped_owner_acting_inside_its_scope():
    """The fail-CLOSED half of the scope leg, with a middleware-resolved principal.

    Every other scoped case at route level is the refusal, and a projection
    that over-narrowed — dropping the ``scope_services is None`` short-circuit,
    matching the scope against the row *id* rather than its ``service_name`` —
    would satisfy all of them while silently stripping a scoped operator's own
    buttons from the console. Nothing above catches that; this does.

    It is deliberately not the agreement matrix's ``scoped_in`` case: there the
    ``Principal`` is hand-built, so it proves the predicate but proves nothing
    about the scope surviving token resolution in ``ScopedTokenAuthMiddleware``
    and reaching the mapper intact.
    """
    with _scoped_token(["demo"]):
        resp = _client(_queries(_service("tok-scoped"))).get(
            "/services/demo", headers=_auth(SCOPED_RAW)
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["manageable"] is True


def test_manageable_predicts_the_outcome_of_a_real_restart_over_the_tunnel():
    """The flag against an actual WRITE — the one assertion here that cannot go
    tautological.

    Every other test compares the projection against a predicate, and the
    predicate is now literally the one the gate calls, so agreement is
    structural. This compares it against what ``POST /services/{ident}/restart``
    really does to the same row for the same principal through the same
    middleware stack. If the read surface and the write gate ever stop being
    the same decision — a new ownership concept honoured in one and not the
    other, a route that stops calling ``require_owner_or_admin`` — the console
    starts lying and this is the test that says so.
    """
    for owner, manageable, status in ((TUNNEL_OWNER, True, 200), (None, False, 403)):
        client = TestClient(_tunnel_app(_queries(_service(owner))), raise_server_exceptions=False)
        headers = {"Authorization": f"Bearer {TUNNEL_CAP}"}
        read = client.get("/services/demo", headers=headers)
        assert read.status_code == 200, read.text
        assert read.json()["manageable"] is manageable
        assert client.post("/services/svc-1/restart", headers=headers).status_code == status


def test_manageable_has_no_default_so_a_future_call_site_cannot_omit_it():
    """The field is REQUIRED, and that is the claim the schema comment argues.

    Everything above drives the one mapper that always passes the value, so
    adding ``default=True`` to the field breaks NOTHING in this file — verified
    by mutation. The comment's whole case (a default can only ever fire for a
    future construction site that forgot, and both candidate defaults are wrong
    there: ``True`` invents permission nobody checked, ``False`` silently
    strips the owner's own actions) would therefore be an unbacked assertion.
    This is the assertion that backs it, so the argument and the code cannot
    part company.

    Asserted on ``model_fields`` rather than by omitting the kwarg at a call
    site: the requirement is a property of the SCHEMA — it must bind for a
    construction site that does not exist yet, which no call-site test can
    reach.
    """
    field = ServiceResponse.model_fields["manageable"]
    assert field.is_required(), "manageable must stay required — see the field's comment"
    with pytest.raises(ValidationError):
        # Every OTHER required field supplied, so the only thing this can be
        # failing on is the missing verdict.
        ServiceResponse(
            id="svc-1",
            name="demo",
            status=JobStatus.running,
            gpu_count=0,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
