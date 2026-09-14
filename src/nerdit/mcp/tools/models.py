"""Model-serving MCP tool implementations (P5).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import (
    DEFAULT_MODEL_LIMIT,
    MAX_MODEL_LIMIT,
    Cursor,
    IdempotencyKey,
)
from nerdit.mcp.transport import _request_client

# Per-argument rules an agent needs before it calls (A24, D-A24-1). Shared
# vocabulary comes from ``_shared``; everything below is model-specific, so it
# is annotated here rather than stretching an alias.
ModelRef = Annotated[
    str,
    Field(
        description="Model reference to pull and serve, e.g. ``llama3.1:8b`` (ollama) "
        "or a Hugging Face repo id (vLLM): letters, digits, ``. - _ /`` with an "
        "optional ``:tag``."
    ),
]
ModelGpus = Annotated[
    int,
    Field(
        description="GPUs to reserve for the model server; 0 (the default) is CPU "
        "inference, and allocated GPUs are shared with other workloads. The ``vllm`` "
        "backend needs at least 1 and refuses 0 with ``model.gpu_required``."
    ),
]
ModelName = Annotated[
    str | None,
    Field(
        description="Service name for the model (lowercase DNS label). Omit and the "
        "name is the backend-prefixed, sanitized model reference (``llama3.1:8b`` on "
        "ollama → ``ollama-llama3-1-8b``)."
    ),
]
ModelBackend = Annotated[
    str | None,
    Field(
        description="Serving backend: ``ollama`` (CPU or GPU) or ``vllm`` (GPU-only, "
        "one model per server, weights from Hugging Face). Omit to use the daemon's "
        "configured default; an unknown name is refused with ``model.unknown_backend``."
    ),
]
ModelMaxLen = Annotated[
    int | None,
    Field(
        description="vLLM only: context-window cap in tokens (``--max-model-len``), "
        "256 to 262144. Omit and the daemon decides: an operator flag in "
        "``[models].vllm_extra_args``, else 4096 on a GPU under 16 GB VRAM, else "
        "vLLM's own default. Any other backend refuses it with ``model.backend_param``."
    ),
]
ModelGpuMemory = Annotated[
    float | None,
    Field(
        description="vLLM only: fraction of VRAM the engine may claim "
        "(``--gpu-memory-utilization``), 0.1 to 0.95. Omit and the daemon decides: an "
        "operator flag in ``[models].vllm_extra_args``, else 0.85 on a GPU under 16 GB "
        "VRAM, else vLLM's own default. Any other backend refuses it with "
        "``model.backend_param``."
    ),
]
ModelLimit = Annotated[
    int | None,
    Field(
        description="Models per page; a value outside 1–200 is clamped into it "
        "(0 becomes 1, 10000 becomes 200). Omit for 50."
    ),
]


# The signature mirrors POST /models one-for-one (the sibling tool modules
# carry the same ledger note); collapsing any pair would hide a REST field.
async def _serve_model_impl(  # noqa: PLR0913 - mirrors the REST payload
    client: NerditClient,
    model: str,
    *,
    gpus: int = 0,
    name: str | None = None,
    backend: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Serve a local model (kind=model workload), auto-minting an idempotency key.

    Like :func:`_serve_impl`, the agent rarely supplies its own key, so we mint
    a UUID to make every serve safe to retry: a replayed call collapses to the
    same model row rather than failing on the unique ``service_name``.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.serve_model(
            model,
            gpus=gpus,
            name=name,
            backend=backend,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            idempotency_key=idempotency_key,
        )
    )


async def _list_models_impl(
    client: NerditClient,
    *,
    limit: int | None = None,
    cursor: str | None = None,
) -> Any:
    """List served models, bounded to ``limit`` per page.

    The bound is enforced server-side (``?limit=``) so the daemon never returns
    an unbounded page; the cursor is passed straight through for paging.
    """
    bound = _clamp(limit if limit is not None else DEFAULT_MODEL_LIMIT, MAX_MODEL_LIMIT)
    return await _call(client.list_models(limit=bound, cursor=cursor))


# fmt: off
async def serve_model(  # noqa: PLR0913 - mirrors the REST payload
        model: ModelRef,
        gpus: ModelGpus = 0,
        name: ModelName = None,
        backend: ModelBackend = None,
        max_model_len: ModelMaxLen = None,
        gpu_memory_utilization: ModelGpuMemory = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Serve a local AI model as an OpenAI-compatible endpoint.

        Serving is asynchronous: the call returns immediately with the
        model row in ``building`` while the daemon pulls the image and weights
        off-tick; poll ``list_models`` (or ``get_service``) until it is
        ``running``. Lifecycle (stop / restart / delete) goes through the
        existing service tools (``stop_service`` / ``restart_service`` /
        ``remove_service``).

        An app connects to this model by declaring an ``[ai.<name>]`` binding
        (provider ``ollama``, model = this model); at each launch it receives
        ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the ``default`` binding also
        sets ``OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL`` so an unmodified
        OpenAI SDK reaches this endpoint with no code change.
        """
        return await _serve_model_impl(
            _request_client(),
            model,
            gpus=gpus,
            name=name,
            backend=backend,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            idempotency_key=idempotency_key,
        )

async def list_models(limit: ModelLimit = None, cursor: Cursor = None) -> Any:
        """List served AI models (kind=model workloads), bounded to ``limit`` per page.

        Returns a cursor-paginated page; pass ``cursor`` from the previous page
        to continue. A newly served model shows ``building`` until its off-tick
        pull completes; lifecycle actions go through the service tools.
        """
        return await _list_models_impl(_request_client(), limit=limit, cursor=cursor)
# fmt: on


TOOLS = (
    serve_model,
    list_models,
)
