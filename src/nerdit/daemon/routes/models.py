"""Create and read model workloads; lifecycle actions use /services.

Creation is authorized, idempotent and audited as model.serve. The controller
pulls missing backend images asynchronously while rows remain building, then
reconciles health, restart policy and stable ports. Mounted under /api only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request

from nerdit.core.jobconfig import parse_job_config
from nerdit.core.models.backend import sanitize_model_name
from nerdit.daemon.auth import require_service_scope
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes._resources import (
    authorize_create,
    new_workload_row,
    reject_reserved_name,
    reserve_or_conflict,
    resolve_backend_or_422,
)
from nerdit.db.models import (
    Job,
    JobKind,
    ModelListPage,
    ModelResponse,
    ModelServeRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Helpers -----------------------------------------------------------------


def _gpu_utilization(request: Request, gpu_ids: list[str]) -> dict[str, int | None]:
    """Join live per-GPU utilization from the in-memory resource monitor.

    Same source as `GET /gpus` (`app.state.monitor`). None-safe on both
    axes: a missing/unwired monitor and a GPU with no metrics snapshot yet
    both project as `None` rather than failing the read.
    """
    monitor = getattr(request.app.state, "monitor", None)
    try:
        metrics = monitor.get_metrics() if monitor is not None else {}
    except Exception:  # pragma: no cover - defensive; a read must never 500 here
        metrics = {}
    out: dict[str, int | None] = {}
    for gpu_id in gpu_ids:
        m = metrics.get(gpu_id)
        out[gpu_id] = m.utilization_percent if m is not None else None
    return out


async def _model_response(request: Request, job: Job) -> ModelResponse:
    """Map a `kind=model` Job to the dedicated model-shaped projection."""
    queries = request.app.state.queries
    cfg = parse_job_config(job)
    name = job.service_name or job.name or job.id

    endpoint_url: str | None = None
    endpoint = await queries.get_service_endpoint(name)
    if endpoint is not None:
        # Invariant #1: the OpenAI API is the contract — the loopback base URL
        # is always the /v1 API root, whatever backend serves the model.
        endpoint_url = f"http://127.0.0.1:{endpoint.host_port}/v1"

    # (P21 D4) Read-back of the per-serve engine bounds. `config` is a tolerant
    # JSON parse, so project only well-typed values (bools are ints in Python).
    raw_len = cfg.get("max_model_len")
    raw_util = cfg.get("gpu_memory_utilization")
    max_model_len = raw_len if isinstance(raw_len, int) and not isinstance(raw_len, bool) else None
    gpu_memory_utilization = (
        float(raw_util)
        if isinstance(raw_util, (int, float)) and not isinstance(raw_util, bool)
        else None
    )

    gpu_ids = await queries.get_job_gpus(job.id)
    return ModelResponse(
        id=job.id,
        name=name,
        model=cfg.get("model"),
        backend=cfg.get("backend"),
        status=job.status,
        desired_state=job.desired_state,
        model_pulled=bool(cfg.get("model_pulled", False)),
        gpu_count=job.gpu_count,
        gpu_ids=gpu_ids,
        gpu_utilization=_gpu_utilization(request, gpu_ids),
        endpoint=endpoint_url,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        created_at=job.created_at,
    )


# --- Write (desired-state; controller converges) ------------------------------


@router.post("/models", response_model=ModelResponse, status_code=201, operation_id="serve_model")
async def serve_model(request: Request, body: ModelServeRequest) -> ModelResponse:
    """Serve a local model: register a `kind=model` desired-state workload.

    Writes a stable row with `desired_state='running'` and status
    `building`; the controller pulls the backend image off-tick (S5),
    allocates the port/GPUs and launches the server on a later tick. The image
    is deliberately **not** required to exist locally (unlike `POST
    /services`). Authorized to `submitter`/`admin`, idempotent, audited
    (`model.serve`).
    """
    authorize_create(request, body)
    queries = request.app.state.queries
    settings = request.app.state.settings
    controller = request.app.state.model_controller

    # Resolve the serving backend. An explicit unknown name is a 422, not
    # a silent default. The backend supplies the name prefix, image, container
    # port and health path so the row is backend-shaped from creation.
    backend = resolve_backend_or_422(
        controller.get_backend,
        body.backend,
        code="model.unknown_backend",
        message=f"Unknown model backend '{body.backend}'.",
        hint="Supported backends: ollama, vllm.",
    )
    if backend.requires_gpu and body.gpus < 1:
        raise NerditError(
            422,
            "model.gpu_required",
            f"The '{backend.name}' backend requires at least one GPU.",
            hint="Pass --gpus N, or use --backend ollama for CPU serving.",
        )

    # (P21 D4) The typed engine bounds are vLLM-only knobs. Reject them on any
    # other resolved backend — including the implicit default when no backend
    # was named — rather than silently dropping them.
    requested_params = [
        param
        for param, value in (
            ("max_model_len", body.max_model_len),
            ("gpu_memory_utilization", body.gpu_memory_utilization),
        )
        if value is not None
    ]
    if requested_params and backend.name != "vllm":
        raise NerditError(
            422,
            "model.backend_param",
            f"{'/'.join(requested_params)} only apply to the 'vllm' backend "
            f"(resolved backend: '{backend.name}').",
            hint="Drop the parameter(s), or pass backend='vllm' (--backend vllm).",
        )

    service_name = body.name or sanitize_model_name(body.model, prefix=backend.name_prefix)
    reject_reserved_name(service_name)
    # (P25 D-P25-3 leg b) Checked on the DERIVED row name, never on `body.name`:
    # a scoped token must name the model into its scope up front (the default is
    # backend-shaped, e.g. 'ollama-…'). Pre-row-write, pre-reservation.
    require_service_scope(request, service_name)
    config: dict[str, object] = {
        "model": body.model,
        "backend": backend.name,
        "image": backend.image,
        "port": backend.container_port,
    }
    # Persist only what was asked for — an absent override must stay absent from
    # the row so the VRAM-aware defaults (D3) still apply on launch.
    if body.max_model_len is not None:
        config["max_model_len"] = body.max_model_len
    if body.gpu_memory_utilization is not None:
        config["gpu_memory_utilization"] = body.gpu_memory_utilization

    job = new_workload_row(
        request,
        kind=JobKind.model,
        service_name=service_name,
        gpu_count=body.gpus,
        # The backend's liveness path answers as soon as the server is up (even
        # mid-pull for Ollama; at /health once the vLLM engine is ready), so the
        # probe covers liveness; readiness is the separate model_pulled flag.
        health_check={
            "path": backend.health_path,
            "start_period_s": settings.models.start_period_s,
        },
        config=config,
    )

    job = await reserve_or_conflict(
        request,
        queries,
        job,
        service_name=service_name,
        name_taken_hint="It may already be serving this model; check `nerdit models list`, "
        "or pass a different --name.",
    )

    return await _model_response(request, job)


# --- Bounded read --------------------------------------------------------------


@router.get("/models", response_model=ModelListPage, operation_id="list_models")
async def list_models(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
) -> ModelListPage:
    """Cursor-paginated, model-shaped list of `kind=model` rows (newest first).

    Read is intentionally open to any authenticated principal (readonly and up).
    """
    queries = request.app.state.queries
    try:
        jobs, next_cursor = await queries.list_services(
            cursor=cursor, limit=limit, kinds=(JobKind.model,)
        )
    except ValueError as exc:
        raise NerditError(400, "bad_request", str(exc)) from exc
    items = [await _model_response(request, job) for job in jobs]
    return ModelListPage(items=items, next_cursor=next_cursor)
