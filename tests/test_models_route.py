"""Route-level tests for the ``/models`` surface (P5 / S4+S6).

Pure-async harness (``httpx.ASGITransport`` — never the Starlette
``TestClient``, per the sandbox convention): the real models router under the
``ScopedTokenAuthMiddleware`` with ``AsyncMock`` state for the authz / envelope
/ audit assertions, plus a real in-memory database (the ``test_idempotency``
pattern) for the Idempotency-Key replay and the cursor-paginated read.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import ModelsSettings, NerditSettings
from nerdit.core.models.backend import OllamaBackend, VllmBackend
from nerdit.core.models.controller import ModelController
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import QuotaExceeded, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.models import router as models_router
from nerdit.db.database import Database
from nerdit.db.models import (
    ApiToken,
    GpuMetrics,
    Job,
    JobKind,
    JobStatus,
    TokenRole,
)
from nerdit.db.queries import Queries, ServiceNameClaimed, ServiceNameTaken

LEGACY = "legacy-global"

ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


def _model_job(**over) -> Job:
    """A model-kind row shaped like the POST handler writes it."""
    fields = dict(
        id="mdl-1",
        service_name="ollama-llama3-1-8b",
        name="ollama-llama3-1-8b",
        kind=JobKind.model,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(
            {"model": "llama3.1:8b", "backend": "ollama", "image": "ollama/ollama", "port": 11434}
        ),
    )
    fields.update(over)
    return Job(**fields)


def _queries() -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job, **kw: job)
    q.get_secret_claim = AsyncMock(return_value=None)
    q.list_services = AsyncMock(return_value=([], None))
    return q


def _make_app(queries: AsyncMock, *, with_audit: bool = False) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(models_router)

    app.state.queries = queries
    settings = MagicMock()
    settings.models = ModelsSettings()
    app.state.settings = settings
    # (P11) The route resolves the serving backend from the controller registry.
    app.state.model_controller = ModelController(
        OllamaBackend(),
        MagicMock(),
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
    )

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


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1")


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


# --- POST /models: create + config blob shape ---------------------------------


@pytest.mark.asyncio
async def test_serve_creates_model_row_with_sanitized_name():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "ollama-llama3-1-8b"
    assert body["model"] == "llama3.1:8b"
    assert body["backend"] == "ollama"
    assert body["status"] == "building"
    assert body["model_pulled"] is False
    assert body["endpoint"] is None

    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.kind == JobKind.model
    assert job_arg.status == JobStatus.building
    assert job_arg.desired_state == "running"
    assert job_arg.restart_policy == "on-failure"
    assert job_arg.service_name == "ollama-llama3-1-8b"
    assert job_arg.submitted_by_token == "tok-sub"
    # Health probe covers liveness; grace period comes from [models] settings.
    assert job_arg.health_check == {"path": "/", "start_period_s": 300}
    cfg = json.loads(job_arg.config)
    assert cfg == {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
    }


@pytest.mark.asyncio
async def test_serve_vllm_backend_shapes_row_from_backend():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "meta-llama/Llama-3.1-8B", "gpus": 1, "backend": "vllm"},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "vllm-meta-llama-llama-3-1-8b"
    assert body["backend"] == "vllm"
    job_arg = q.reserve_service_for_token.await_args.args[0]
    # vLLM's liveness path + backend-supplied image/port land in the row.
    assert job_arg.health_check == {"path": "/health", "start_period_s": 300}
    cfg = json.loads(job_arg.config)
    assert cfg["backend"] == "vllm"
    assert cfg["port"] == 8000
    assert cfg["image"] == "vllm/vllm-openai:latest"


@pytest.mark.asyncio
async def test_serve_vllm_without_gpu_is_422():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "meta-llama/Llama-3.1-8B", "gpus": 0, "backend": "vllm"},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "model.gpu_required"
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_serve_unknown_backend_is_422():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "llama3.1:8b", "backend": "nope"},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "model.unknown_backend"


# --- P21 D4: per-serve typed engine bounds (vLLM only) --------------------------


@pytest.mark.asyncio
async def test_serve_vllm_persists_engine_overrides_on_row():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpus": 1,
                "backend": "vllm",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.8,
            },
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 201
    cfg = json.loads(q.reserve_service_for_token.await_args.args[0].config)
    assert cfg["max_model_len"] == 2048
    assert cfg["gpu_memory_utilization"] == 0.8
    # The response projects them back so an agent can verify the effective serve.
    body = resp.json()
    assert body["max_model_len"] == 2048
    assert body["gpu_memory_utilization"] == 0.8


@pytest.mark.asyncio
async def test_serve_without_overrides_leaves_config_clean():
    """An absent override must stay ABSENT so the D3 VRAM defaults still apply."""
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "gpus": 1, "backend": "vllm"},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 201
    cfg = json.loads(q.reserve_service_for_token.await_args.args[0].config)
    assert "max_model_len" not in cfg
    assert "gpu_memory_utilization" not in cfg
    assert resp.json()["max_model_len"] is None
    assert resp.json()["gpu_memory_utilization"] is None


@pytest.mark.asyncio
async def test_serve_engine_override_on_ollama_is_422():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "llama3.1:8b", "backend": "ollama", "max_model_len": 2048},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "model.backend_param"
    assert "max_model_len" in resp.json()["message"]
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_serve_engine_override_on_implicit_default_backend_is_422():
    """No ``backend`` named → resolves to the ollama default → still a 422."""
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "llama3.1:8b", "gpu_memory_utilization": 0.5},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "model.backend_param"
    assert "gpu_memory_utilization" in body["message"]
    assert "ollama" in body["message"]
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"max_model_len": 255},
        {"max_model_len": 262145},
        {"gpu_memory_utilization": 0.05},
        {"gpu_memory_utilization": 0.96},
    ],
)
async def test_serve_engine_override_bounds_are_422(payload):
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "m", "gpus": 1, "backend": "vllm", **payload},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_serve_engine_overrides_audited_unmasked():
    """D4: the two bounds are not secrets — they ride the audit params in clear."""
    q = _queries()
    async with _client(_make_app(q, with_audit=True)) as client:
        resp = await client.post(
            "/models",
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "gpus": 1,
                "backend": "vllm",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.8,
            },
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 201
    params = q.insert_audit_log.await_args.kwargs["params_redacted"] or ""
    assert "2048" in params
    assert "0.8" in params
    assert "***" not in params


@pytest.mark.asyncio
async def test_serve_honors_explicit_name_override():
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "llama3.1:8b", "name": "chat-model", "gpus": 1},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 201
    assert resp.json()["name"] == "chat-model"
    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.service_name == "chat-model"
    assert job_arg.gpu_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["shared", "rotate-key"])
async def test_serve_rejects_reserved_names(name):
    """P8: a model workload must not squat the reserved service names."""
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post(
            "/models",
            json={"model": "llama3.1:8b", "name": name},
            headers=_auth(SUB_RAW),
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_serve_persists_idempotency_key_on_row():
    q = _queries()
    async with _client(_make_app(q)) as client:
        await client.post(
            "/models",
            json={"model": "llama3.1:8b"},
            headers={**_auth(SUB_RAW), "Idempotency-Key": "idem-xyz"},
        )
    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.idempotency_key == "idem-xyz"


@pytest.mark.asyncio
async def test_serve_does_not_require_local_image():
    # Unlike POST /services there is no image_exists gate: the image is pulled
    # off-tick while the row sits `building` (S5). No runtime on app.state at all.
    q = _queries()
    async with _client(_make_app(q)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201


# --- POST /models: authz + validation + structured errors ----------------------


@pytest.mark.asyncio
async def test_readonly_blocked_on_serve():
    async with _client(_make_app(_queries())) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(RO_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"


@pytest.mark.asyncio
async def test_legacy_token_can_serve():
    async with _client(_make_app(_queries())) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(LEGACY))
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_empty_model_ref_is_422():
    async with _client(_make_app(_queries())) as client:
        resp = await client.post("/models", json={"model": ""}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


@pytest.mark.asyncio
async def test_bad_model_ref_is_422():
    async with _client(_make_app(_queries())) as client:
        resp = await client.post("/models", json={"model": "bad ref!:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_bad_name_override_is_422():
    async with _client(_make_app(_queries())) as client:
        resp = await client.post(
            "/models", json={"model": "llama3.1:8b", "name": "Bad_Name"}, headers=_auth(SUB_RAW)
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_duplicate_name_returns_409_envelope():
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameTaken("ollama-llama3-1-8b"))
    async with _client(_make_app(q)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "service.name_taken"
    assert "ollama-llama3-1-8b" in body["message"]
    assert body.get("hint")


@pytest.mark.asyncio
async def test_quota_exceeded_returns_403():
    q = _queries()
    q.reserve_service_for_token = AsyncMock(
        side_effect=QuotaExceeded("max_concurrent_jobs", limit=1, current=1)
    )
    async with _client(_make_app(q)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "quota_exceeded"


# --- audit ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serve_audited_as_model_serve():
    q = _queries()
    async with _client(_make_app(q, with_audit=True)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    q.insert_audit_log.assert_awaited()
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "model.serve"
    assert kwargs["target_type"] == "model"
    assert kwargs["result"] == "ok"
    assert "llama3.1:8b" in (kwargs["params_redacted"] or "")


# --- GET /models: projection ----------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_projects_endpoint_pulled_and_utilization():
    from nerdit.db.models import ServiceEndpoint

    q = _queries()
    job = _model_job(
        status=JobStatus.running,
        gpu_count=1,
        config=json.dumps(
            {
                "model": "llama3.1:8b",
                "backend": "ollama",
                "image": "ollama/ollama",
                "port": 11434,
                "model_pulled": True,
            }
        ),
    )
    q.list_services = AsyncMock(return_value=([job], None))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="ollama-llama3-1-8b",
            job_id="mdl-1",
            container_port=11434,
            host_port=9500,
        )
    )
    q.get_job_gpus = AsyncMock(return_value=["GPU-1"])

    app = _make_app(q)
    monitor = MagicMock()
    monitor.get_metrics.return_value = {"GPU-1": GpuMetrics(gpu_id="GPU-1", utilization_percent=42)}
    app.state.monitor = monitor

    async with _client(app) as client:
        resp = await client.get("/models", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    (item,) = resp.json()["items"]
    assert item["endpoint"] == "http://127.0.0.1:9500/v1"
    assert item["model_pulled"] is True
    assert item["gpu_ids"] == ["GPU-1"]
    assert item["gpu_utilization"] == {"GPU-1": 42}
    # kinds filter forwarded: only model rows are read.
    assert q.list_services.await_args.kwargs["kinds"] == (JobKind.model,)


@pytest.mark.asyncio
async def test_list_models_none_safe_without_monitor():
    # No monitor on app.state (route-level harness) → utilization is null, not 500.
    q = _queries()
    q.list_services = AsyncMock(return_value=([_model_job()], None))
    q.get_job_gpus = AsyncMock(return_value=["GPU-1"])
    async with _client(_make_app(q)) as client:
        resp = await client.get("/models", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    (item,) = resp.json()["items"]
    assert item["gpu_utilization"] == {"GPU-1": None}


@pytest.mark.asyncio
async def test_list_models_limit_is_bounded():
    async with _client(_make_app(_queries())) as client:
        too_low = await client.get("/models?limit=0", headers=_auth(RO_RAW))
        too_high = await client.get("/models?limit=201", headers=_auth(RO_RAW))
    assert too_low.status_code == 422
    assert too_high.status_code == 422


# --- real-DB flows: idempotent replay + cursor pagination + kinds default -------


async def _make_db_env() -> tuple[FastAPI, Database, Queries]:
    """Full middleware stack over a real in-memory DB (test_idempotency pattern)."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(models_router)
    app.state.queries = queries
    app.state.settings = NerditSettings()
    app.state.model_controller = ModelController(
        OllamaBackend(),
        MagicMock(),
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
    )

    # inner → outer: Idempotency (innermost), Audit, Auth, RequestId.
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, db, queries


@pytest.mark.asyncio
async def test_idempotency_key_replays_same_model_row():
    app, db, queries = await _make_db_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "K-model"}
            first = await client.post("/models", json={"model": "llama3.1:8b"}, headers=headers)
            assert first.status_code == 201
            assert first.headers.get("Idempotent-Replay") is None

            second = await client.post("/models", json={"model": "llama3.1:8b"}, headers=headers)
            assert second.status_code == 201
            assert second.headers.get("Idempotent-Replay") == "true"
            assert second.json()["id"] == first.json()["id"]

        rows, _ = await queries.list_services(kinds=(JobKind.model,))
        assert len(rows) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_list_models_paginates_and_excludes_services():
    app, db, queries = await _make_db_env()
    try:
        # A plain service row must never appear on the /models read.
        await queries.reserve_service_for_token(
            Job(
                id="svc-1",
                kind=JobKind.service,
                service_name="webapp",
                gpu_count=0,
                status=JobStatus.running,
                desired_state="running",
            )
        )
        async with _client(app) as client:
            for ref in ("llama3.1:8b", "phi3:mini", "qwen2:0.5b"):
                resp = await client.post("/models", json={"model": ref})
                assert resp.status_code == 201

            page1 = (await client.get("/models?limit=2")).json()
            assert len(page1["items"]) == 2
            assert page1["next_cursor"]

            page2 = (await client.get(f"/models?limit=2&cursor={page1['next_cursor']}")).json()
            assert len(page2["items"]) == 1
            assert page2["next_cursor"] is None

        names = {i["name"] for i in page1["items"]} | {i["name"] for i in page2["items"]}
        assert names == {"ollama-llama3-1-8b", "ollama-phi3-mini", "ollama-qwen2-0-5b"}
        assert "webapp" not in names

        # The default kinds filter preserves the shipped service+model behavior.
        both, _ = await queries.list_services()
        assert {j.service_name for j in both} == {
            "webapp",
            "ollama-llama3-1-8b",
            "ollama-phi3-mini",
            "ollama-qwen2-0-5b",
        }
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_claimed_name_returns_409_envelope():
    """P39: a foreign secret claim on the derived name refuses the serve."""
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameClaimed("ollama-llama3-1-8b"))
    async with _client(_make_app(q)) as client:
        resp = await client.post("/models", json={"model": "llama3.1:8b"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.name_claimed"
    assert q.reserve_service_for_token.await_args.kwargs == {"admin": False}
