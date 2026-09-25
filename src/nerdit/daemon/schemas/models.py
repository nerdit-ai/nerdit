"""Request/response schemas for the `/models` surface."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.enums import JobStatus
from nerdit.utils.names import DNS_LABEL_PATTERN


class ModelServeRequest(StrictRequestModel):
    """Serve a backend model reference, such as llama3.1:8b.

    Name defaults to sanitize_model_name(model) and follows service DNS-label
    rules. Zero GPUs enables CPU inference; allocated GPUs are shared.
    """

    model: str = Field(
        min_length=1,
        pattern=r"^[\w.\-/]+(:[\w.\-]+)?$",
        description="Model reference to serve, e.g. 'llama3.1:8b'",
    )
    gpus: int = Field(default=0, ge=0, description="GPUs the model server needs (0 = CPU; shared)")
    name: str | None = Field(
        default=None,
        pattern=DNS_LABEL_PATTERN,
        description="Optional service-name override (default: sanitized model ref)",
    )
    backend: str | None = Field(
        default=None,
        description="Serving backend ('ollama' | 'vllm'); None uses the daemon default (P11)",
    )
    # (P21 D4) Narrow typed engine bounds — deliberately NOT a raw argv
    # passthrough: a generic extra_args field would let any submitter token
    # inject arbitrary flags into the vLLM server process. The backend-vs-param
    # check needs the resolved backend object, so it lives in the route (the
    # `model.gpu_required` precedent), not in a pydantic validator.
    max_model_len: int | None = Field(
        default=None,
        ge=256,
        le=262144,
        description="vLLM only: cap the context window (--max-model-len). "
        "422 model.backend_param on other backends",
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        ge=0.1,
        le=0.95,
        description="vLLM only: VRAM fraction the engine may claim "
        "(--gpu-memory-utilization). 422 model.backend_param on other backends",
    )


class ModelResponse(BaseModel):
    """API projection of a `kind=model` row for the `/models` surface.

    Model-shaped read beside the generic `ServiceResponse` (Decision #7:
    model rows keep appearing in `GET /services` too). `endpoint` is the
    loopback OpenAI-compatible base URL (Invariant #1); `model_pulled` flips
    once the weights finished pulling (`ensure_model` success), so "container
    up, weights absent" is distinguishable from ready.
    """

    id: str = Field(description="Unique 12-character row identifier")
    name: str = Field(description="Stable service name (identity)")
    model: str | None = Field(default=None, description="Model reference, e.g. 'llama3.1:8b'")
    backend: str | None = Field(default=None, description="Serving backend (e.g. 'ollama')")
    status: JobStatus = Field(description="Current lifecycle state")
    desired_state: str | None = Field(
        default=None, description="Reconciler target ('running' | 'stopped')"
    )
    model_pulled: bool = Field(
        default=False, description="Whether the model weights finished pulling"
    )
    gpu_count: int = Field(default=0, description="GPUs requested (0 = CPU inference)")
    gpu_ids: list[str] = Field(
        default_factory=list, description="Nerdit inventory IDs allocated to the model server"
    )
    gpu_utilization: dict[str, int | None] = Field(
        default_factory=dict,
        description="Live compute utilization (0-100%) per allocated GPU; null when unknown",
    )
    endpoint: str | None = Field(
        default=None,
        description="Loopback OpenAI-compatible base URL (http://127.0.0.1:{port}/v1); "
        "null until the host port is published",
    )
    # (P21 D4) Read-back of the per-serve engine overrides so an agent can verify
    # what a serve actually took effect with; null when unset (vLLM-only knobs).
    max_model_len: int | None = Field(
        default=None, description="vLLM context-window cap set at serve time (null when unset)"
    )
    gpu_memory_utilization: float | None = Field(
        default=None, description="vLLM VRAM fraction set at serve time (null when unset)"
    )
    created_at: datetime = Field(description="Timestamp when the model workload was created")


class ModelListPage(BaseModel):
    """Cursor-paginated page of models for the bounded `GET /models` read."""

    items: list[ModelResponse]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page; null when the list is exhausted",
    )
