"""Model-serving MCP tool implementations (P5).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import DEFAULT_MODEL_LIMIT, MAX_MODEL_LIMIT
from nerdit.mcp.transport import _request_client


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
        model: str,
        gpus: int = 0,
        name: str | None = None,
        backend: str | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Serve a local AI model as an OpenAI-compatible endpoint.

        ``backend`` selects the serving backend: ``ollama`` (default; CPU or
        GPU) or ``vllm`` (GPU-only, one model per server, weights from Hugging
        Face — pass ``gpus>=1``). Omit it to use the daemon's configured
        default. Serving is asynchronous: the call returns immediately with the
        model row in ``building`` while the daemon pulls the image and weights
        off-tick; poll ``list_models`` (or ``get_service``) until it is
        ``running``. Lifecycle (stop / restart / delete) goes through the
        existing service tools (``stop_service`` / ``restart_service`` /
        ``remove_service``). An idempotency key is auto-generated if omitted.

        ``max_model_len`` / ``gpu_memory_utilization`` are vLLM-only engine
        bounds (context window cap [256, 262144]; VRAM fraction [0.1, 0.95]);
        other backends reject them (``model.backend_param``).

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

async def list_models(limit: int | None = None, cursor: str | None = None) -> Any:
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
