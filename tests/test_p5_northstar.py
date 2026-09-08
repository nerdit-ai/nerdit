"""Exercise model serving and AI-bound app deployment in one controller scenario.

Use real model routes, middleware and shared in-memory database with FakeRuntime;
no Docker, Ollama or Starlette TestClient. Pin off-tick image pull, port 11434 and
bridge bind, persisted model_pulled state and running projection.

An app with Ollama and API bindings acquires nothing until its model is ready.
Verify the exact OPENAI_*, NERDIT_AI_DEFAULT_* and NERDIT_AI_CHEAP_* environment.
Fresh controllers over the same database re-adopt containers without pulling
weights again; a post-reboot app launch resolves the identical environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import ModelsSettings, NerditSettings, ServicesSettings
from nerdit.core.models.backend import sanitize_model_name
from nerdit.core.models.controller import ModelController
from nerdit.core.secrets import SecretManager
from nerdit.core.services import ServiceController
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.models import router as models_router
from nerdit.db.models import Job, JobKind, JobStatus
from nerdit.db.queries import Queries
from tests.test_services_reconcile import FakeRuntime, RecordingBackend, _settle_models

pytestmark = pytest.mark.asyncio

MODEL_REF = "llama3.1:8b"
MODEL_NAME = sanitize_model_name(MODEL_REF)  # 'ollama-llama3-1-8b'
BRIDGE_HOST = "172.17.0.1"  # pinned Linux value (default "auto" is platform-resolved)
_MODELS_SETTINGS = ModelsSettings(bridge_host=BRIDGE_HOST)

APP_NAME = "ai-app"
APP_PORT = 3000  # Node buildpack default — matches examples/ai-app
CHEAP_BASE_URL = "https://api.openai.com/v1"
CHEAP_MODEL = "gpt-4o-mini"
CHEAP_SECRET_VALUE = "sk-cheap-secret-value"

# Exactly the plan's north-star example: [ai.default] ollama + [ai.cheap] api,
# persisted as the deploy route (S8) would write it — the SPEC, never resolved
# values (resolution happens at every launch).
AI_SPEC = {
    "default": {"provider": "ollama", "model": MODEL_REF},
    "cheap": {
        "provider": "api",
        "model": CHEAP_MODEL,
        "base_url": CHEAP_BASE_URL,
        "api_key": "${secrets.OPENAI_KEY}",
    },
}


def _app_row() -> Job:
    """The ``kind=service`` row a ``nerdit deploy examples/ai-app`` produces."""
    config = {
        "image": f"nerdit-app/{APP_NAME}:1",
        "image_repo": f"nerdit-app/{APP_NAME}",
        "build_version": 1,
        "max_version": 1,
        "port": APP_PORT,
        "ai": AI_SPEC,
    }
    return Job(
        name=APP_NAME,
        kind=JobKind.service,
        service_name=APP_NAME,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(config),
    )


def _route_app(queries: Queries) -> FastAPI:
    """The real /models router under the full production middleware stack."""
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(models_router)
    app.state.queries = queries
    app.state.settings = NerditSettings()
    # (P11) The route resolves the serving backend from the controller registry;
    # its descriptors are stateless so a fresh controller is fine for the route.
    from unittest.mock import MagicMock

    from nerdit.core.models.backend import OllamaBackend, VllmBackend

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
    return app


def _controller(
    queries: Queries,
    runtime: FakeRuntime,
    mc: ModelController,
    secrets: SecretManager,
) -> ServiceController:
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        secrets=secrets,
        model_controller=mc,
    )
    controller._backoff_seconds = lambda count: 0.0  # type: ignore[assignment]

    # The model row carries a real health_check (from the route); stub the HTTP
    # probe so no test ever opens a socket.
    async def healthy(host_port: int, path: str, timeout: float) -> int:
        return 200

    controller._check_health = healthy  # type: ignore[assignment]
    return controller


@dataclass
class Scenario:
    """Shared north-star run: state handles + observations recorded per phase."""

    queries: Queries
    runtime: FakeRuntime
    secrets: SecretManager
    tmp_path: Path
    backend: RecordingBackend
    mc: ModelController
    controller: ServiceController
    client: AsyncClient

    serve_response: dict = field(default_factory=dict)
    # Phase A (model building, image pulling): the app must hold NOTHING.
    app_status_during_pull: JobStatus | None = None
    app_endpoint_during_pull: object = None
    app_gpus_during_pull: list = field(default_factory=list)
    live_containers_during_pull: int = 0
    # Phase B (model running, weights not pulled yet): still gated.
    app_status_before_pulled: JobStatus | None = None
    # Phase C (running + pulled): the app launches with the contract env.
    model_host_port: int = 0
    model_launch_config: object = None
    app_launch_config: object = None
    app_first_env: dict = field(default_factory=dict)
    app_container_id: str = ""

    async def run_happy_path(self) -> None:
        # --- nerdit serve llama3.1:8b --gpus 1 (through the real route) ------
        resp = await self.client.post(
            "/models",
            json={"model": MODEL_REF, "gpus": 1},
            headers={"Idempotency-Key": "northstar-serve-1"},
        )
        assert resp.status_code == 201, resp.text
        self.serve_response = resp.json()

        # --- nerdit secrets set ai-app OPENAI_KEY=... + nerdit deploy --------
        self.secrets.set(APP_NAME, {"OPENAI_KEY": CHEAP_SECRET_VALUE})
        app_row = await self.queries.reserve_service_for_token(_app_row())

        # Tick 1: model image absent → off-tick pull, no launch anywhere; the
        # app's [ai.default] binding is not ready → nothing acquired.
        await self.controller.reconcile()
        row = await self.queries.get_service_by_name(APP_NAME)
        self.app_status_during_pull = row.status
        self.app_endpoint_during_pull = await self.queries.get_service_endpoint(APP_NAME)
        self.app_gpus_during_pull = await self.queries.get_job_gpus(app_row.id)
        self.live_containers_during_pull = len(self.runtime.live)
        await _settle_models(self.mc)  # image pull lands

        # Tick 2: model launches (backend shape); the app still waits — the
        # container is up but the weights are not marked pulled yet.
        await self.controller.reconcile()
        self.model_launch_config = self.runtime.run_configs[-1]
        ep = await self.queries.get_service_endpoint(MODEL_NAME)
        self.model_host_port = ep.host_port
        row = await self.queries.get_service_by_name(APP_NAME)
        self.app_status_before_pulled = row.status
        await _settle_models(self.mc)  # ensure_model persists model_pulled

        # Tick 3: running + pulled → the app launches with the contract env.
        await self.controller.reconcile()
        row = await self.queries.get_service_by_name(APP_NAME)
        assert row.status is JobStatus.running
        self.app_launch_config = self.runtime.run_configs[-1]
        self.app_first_env = dict(self.app_launch_config.env or {})
        self.app_container_id = row.container_id

    async def shutdown(self) -> None:
        await self.controller.shutdown()
        await self.mc.shutdown()
        await self.client.aclose()


@pytest.fixture
async def scenario(queries, sample_gpus, tmp_path) -> Scenario:
    runtime = FakeRuntime()
    runtime.missing_images.add("ollama/ollama")  # forces the off-tick pull
    backend = RecordingBackend()
    mc = ModelController(
        backend, runtime, queries, models_settings=_MODELS_SETTINGS, data_dir=str(tmp_path)
    )
    secrets = SecretManager(tmp_path / "secrets")
    controller = _controller(queries, runtime, mc, secrets)
    client = AsyncClient(transport=ASGITransport(app=_route_app(queries)), base_url="http://test")

    s = Scenario(
        queries=queries,
        runtime=runtime,
        secrets=secrets,
        tmp_path=tmp_path,
        backend=backend,
        mc=mc,
        controller=controller,
        client=client,
    )
    await s.run_happy_path()
    yield s
    await s.shutdown()


# --- 1. serve: route → pull → launch shape → ensure_model → running -----------


async def test_model_served_via_route_reaches_running_and_pulled(scenario: Scenario):
    body = scenario.serve_response
    assert body["name"] == MODEL_NAME
    assert body["model"] == MODEL_REF
    assert body["status"] == "building"
    assert body["model_pulled"] is False

    # The backend image was pulled off-tick before any container ran.
    assert scenario.runtime.pulled == ["ollama/ollama"]

    # Launch shape: Ollama's fixed 11434 mapped to the stable host port, dual
    # bound on loopback + the bridge gateway (reachable from app containers,
    # never the LAN), with the system-owned weights volume.
    cfg = scenario.model_launch_config
    assert cfg.image == "ollama/ollama"
    assert cfg.ports == {11434: scenario.model_host_port}
    assert cfg.extra_port_bind_ips == [BRIDGE_HOST]
    assert cfg.network_mode == "bridge"
    assert cfg.env["OLLAMA_HOST"] == "0.0.0.0:11434"
    assert cfg.volumes == {str(scenario.tmp_path / "models" / "ollama"): "/root/.ollama"}
    assert cfg.gpu_ids  # --gpus 1 honored through the placement path

    # ensure_model fired once, on loopback (the daemon is on the host), and
    # persisted the readiness flag the binding resolver keys on.
    assert scenario.backend.ensured == [(f"http://127.0.0.1:{scenario.model_host_port}", MODEL_REF)]
    row = await scenario.queries.get_service_by_name(MODEL_NAME)
    assert row.status is JobStatus.running
    assert json.loads(row.config)["model_pulled"] is True

    # The GET /models projection surfaces the OpenAI-contract endpoint.
    resp = await scenario.client.get("/models")
    assert resp.status_code == 200
    (item,) = resp.json()["items"]
    assert item["name"] == MODEL_NAME
    assert item["model_pulled"] is True
    assert item["endpoint"] == f"http://127.0.0.1:{scenario.model_host_port}/v1"


# --- 2. the app is gated (with nothing acquired) until running + pulled --------


async def test_app_acquires_nothing_until_model_is_ready(scenario: Scenario):
    # While the model image was still pulling: untouched, no port, no GPU, and
    # the only live container later is the model's.
    assert scenario.app_status_during_pull is JobStatus.building
    assert scenario.app_endpoint_during_pull is None
    assert scenario.app_gpus_during_pull == []
    assert scenario.live_containers_during_pull == 0

    # Model container up but weights not marked pulled: still gated (strict
    # readiness, Decision #5).
    assert scenario.app_status_before_pulled is JobStatus.building

    # The wait was logged with an actionable line (retry, never terminal).
    app_row = await scenario.queries.get_service_by_name(APP_NAME)
    logs = [entry.message for entry in await scenario.queries.get_logs(app_row.id)]
    assert any("Waiting on AI binding" in line for line in logs)

    # And on the tick after running+pulled it actually launched.
    assert app_row.status is JobStatus.running
    assert app_row.container_id in scenario.runtime.live
    assert await scenario.queries.get_service_endpoint(APP_NAME) is not None


# --- 3. the launched env is EXACTLY the frozen contract (Invariant #1) ---------


async def test_app_env_carries_exact_binding_contract(scenario: Scenario):
    env = scenario.app_first_env
    local_url = f"http://{BRIDGE_HOST}:{scenario.model_host_port}/v1"

    # [ai.default] provider=ollama → the plain OPENAI_* triple an unmodified
    # OpenAI SDK picks up, plus its NERDIT_AI_DEFAULT_* mirror.
    assert env["OPENAI_BASE_URL"] == local_url
    assert env["OPENAI_API_KEY"] == "nerdit-local"
    assert env["OPENAI_MODEL"] == MODEL_REF
    assert env["NERDIT_AI_DEFAULT_URL"] == local_url
    assert env["NERDIT_AI_DEFAULT_KEY"] == "nerdit-local"
    assert env["NERDIT_AI_DEFAULT_MODEL"] == MODEL_REF

    # [ai.cheap] provider=api → spec base_url + the secret VALUE resolved from
    # the app's write-only secrets (${secrets.OPENAI_KEY}).
    assert env["NERDIT_AI_CHEAP_URL"] == CHEAP_BASE_URL
    assert env["NERDIT_AI_CHEAP_KEY"] == CHEAP_SECRET_VALUE
    assert env["NERDIT_AI_CHEAP_MODEL"] == CHEAP_MODEL

    # The raw user secret still flows through, and the app got its $PORT.
    assert env["OPENAI_KEY"] == CHEAP_SECRET_VALUE
    assert env["PORT"] == str(APP_PORT)


# --- 4. simulated reboot: re-adoption, no re-pull, identical re-resolution -----


async def test_reboot_readopts_both_and_reresolves_identical_env(scenario: Scenario):
    queries, runtime = scenario.queries, scenario.runtime
    launched_before = runtime.counter

    # A daemon reboot = FRESH controller pair over the SAME DB, containers alive.
    backend2 = RecordingBackend()
    mc2 = ModelController(
        backend2,
        runtime,
        queries,
        models_settings=_MODELS_SETTINGS,
        data_dir=str(scenario.tmp_path),
    )
    controller2 = _controller(queries, runtime, mc2, scenario.secrets)

    await controller2.reconcile()
    await _settle_models(mc2)

    # Both rows re-adopted in place: nothing relaunched, and ensure_model is
    # NOT re-fired because model_pulled is already persisted.
    assert runtime.counter == launched_before
    assert backend2.ensured == []
    for name in (MODEL_NAME, APP_NAME):
        row = await queries.get_service_by_name(name)
        assert row.status is JobStatus.running
        assert row.container_id in runtime.live

    # The app crashes post-reboot → the fresh controller relaunches it and the
    # bindings re-resolve to an IDENTICAL env (stable ports, same secret).
    runtime.live.pop(scenario.app_container_id)
    for _ in range(5):
        row = await queries.get_service_by_name(APP_NAME)
        if row.status is JobStatus.running and row.container_id in runtime.live:
            break
        await controller2.reconcile()
    assert row.status is JobStatus.running
    assert row.container_id != scenario.app_container_id

    relaunch_env = dict(runtime.run_configs[-1].env or {})
    assert relaunch_env == scenario.app_first_env
    # Stable endpoints: the model URL baked into the env still points at the
    # same host port the model endpoint row holds.
    ep = await queries.get_service_endpoint(MODEL_NAME)
    assert ep.host_port == scenario.model_host_port
    # Weights were never re-pulled across the whole reboot + relaunch.
    assert backend2.ensured == []

    await controller2.shutdown()
    await mc2.shutdown()
