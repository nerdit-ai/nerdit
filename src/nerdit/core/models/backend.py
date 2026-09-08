"""Configure Ollama and vLLM containers behind an OpenAI-compatible endpoint.

Backends supply container shapes, `/v1` endpoint URLs, health probes, and
weight/readiness checks. Controllers own lifecycle. App bindings keep the same
`OPENAI_BASE_URL` and `OPENAI_API_KEY` contract across backends. Sanitized,
backend-prefixed model names share the service-name namespace with apps.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from nerdit.core.runtime.container import ContainerConfig
from nerdit.db.enums import GpuVendor

# Container-side port Ollama listens on (fixed; the host port comes from the
# service_endpoints stable-port allocation, exactly like an app service).
OLLAMA_PORT = 11434

# Container-side port the vLLM OpenAI server listens on (its default; the host
# port is allocated identically to Ollama's).
VLLM_PORT = 8000

DEFAULT_OLLAMA_IMAGE = "ollama/ollama"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:latest"

# vLLM needs more shared memory than Docker's 64 MB default (worker IPC / NCCL),
# even single-GPU; overridable via `[models].vllm_shm_size`.
DEFAULT_VLLM_SHM_SIZE = "1g"

# Under this much VRAM on the smallest allocated GPU, vLLM's launch
# defaults (0.9 utilization, 32k context, full CUDA-graph capture) OOM during
# engine warmup even for a 0.5B model — reproduced on an 8 GB RTX 4060. Below
# the threshold the backend injects conservative bounds; at or above it, and
# whenever the VRAM is unknown, nothing is injected. `--enforce-eager` is
# deliberately NOT injected (it trades throughput; operators add it explicitly).
LOW_VRAM_THRESHOLD_MB = 16_384
_LOW_VRAM_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("--max-model-len", "4096"),
    ("--gpu-memory-utilization", "0.85"),
)


# Weight pulls are long (an 8B model is ~5 GB); overridable via the constructor
# (wired to `[models].pull_timeout_s` by the caller).
DEFAULT_PULL_TIMEOUT_S = 1800.0

# `service_name` grammar (must match ServiceCreateRequest.name in db/models.py):
# DNS label, lowercase, 1-63 chars, no leading/trailing hyphen.
_SERVICE_NAME_MAX_LEN = 63
_INVALID_NAME_RUN = re.compile(r"[^a-z0-9]+")


class ModelPullError(Exception):
    """Raised when pulling model weights fails.

    Covers non-200 responses **and** in-stream errors: Ollama reports pull
    failures as `{"error": ...}` NDJSON lines under an HTTP 200, so a
    successful status code alone never means success. Connection-level
    failures (server not listening yet) raise the `ModelServerUnreachableError`
    subclass instead, so the caller can retry rather than give up.
    """


class ModelServerUnreachableError(ModelPullError):
    """The model server could not be reached (connect refused / timeout).

    Distinct from a genuine pull failure: during a normal slow container start
    the server is not listening yet, so the first `ensure_model` attempt hits
    a transport error. That is **transient** — the controller re-arms and
    retries on a later tick rather than marking the one bounded attempt spent.
    """


def sanitize_model_name(model: str, prefix: str = "ollama") -> str:
    """Map a model ref to a DNS-label-safe `service_name` (backend-prefixed).

    `'llama3.1:8b'` → `'ollama-llama3-1-8b'`;
    `('meta-llama/Llama-3.1-8B', prefix='vllm')` →
    `'vllm-meta-llama-llama-3-1-8b'`. Lowercases, collapses every run of
    invalid characters (dots, colons, slashes, ...) to a single `-`, prefixes
    `<prefix>-` (namespace-separating model rows from app deploys *and* one
    backend's rows from another's), and deterministically truncates to the
    63-char `service_name` cap without leaving a trailing hyphen. The result
    always matches the `ServiceCreateRequest.name` validation pattern.
    """
    slug = _INVALID_NAME_RUN.sub("-", model.lower()).strip("-")
    return f"{prefix}-{slug}"[:_SERVICE_NAME_MAX_LEN].rstrip("-")


def _flag_present(args: Sequence[str], flag: str) -> bool:
    """True when *flag* appears in *args*, as `--flag value` or `--flag=value`.

    Presence-gating, not order-based: never rely on vLLM's argparse last-wins
    behavior to decide whether an injected default is in effect (D3).
    """
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _strip_flag(args: Sequence[str], flag: str) -> list[str]:
    """Remove every occurrence of *flag* (and its value token) from *args*.

    Handles both argv forms: `--flag=value` is one token; `--flag value`
    also drops the following token unless it looks like another flag (so a
    value-less trailing flag never swallows an unrelated argument).
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == flag:
            i += 1
            # Only the flag's own value is consumed — a value-less flag at the
            # end (or followed by another flag) must not swallow a neighbour.
            if i < len(args) and not args[i].startswith("-"):
                i += 1
            continue
        if arg.startswith(flag + "="):
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


@runtime_checkable
class ModelBackend(Protocol):
    """Interface for model-serving backends (Ollama first, vLLM later).

    Implementations are stateless-per-call: `base_url` is passed in (the
    server **root**, e.g. `http://127.0.0.1:18080` — not the `/v1` API
    base) so one backend instance serves every model row.
    """

    #: Backend id stored in `config['backend']` and shown by `GET /models`.
    name: str
    #: `service_name` namespace prefix (`sanitize_model_name` uses it) so a
    #: model row can never collide across backends or with an app deploy.
    name_prefix: str
    #: Container-side port the server listens on (host port is allocated by the
    #: `service_endpoints` stable-port pool, backend-independent).
    container_port: int
    #: HTTP path the `ServiceController` bounded liveness probe hits (200 == up).
    health_path: str
    #: Shared-scope secret keys to inject into the model container env at
    #: launch when present (e.g. `("HF_TOKEN",)` for gated HF repos). Empty for
    #: backends that need none. Never auto-injected beyond these declared keys.
    launch_env_secret_keys: tuple[str, ...]
    #: Whether this backend needs at least one GPU (`serve_model` rejects
    #: `gpus=0` when True — e.g. vLLM has no CPU serving path).
    requires_gpu: bool

    @property
    def image(self) -> str:
        """The container image this backend serves models with."""
        ...

    # One arg per launch-shape input; P21 D3/D4 widened it from 5 to 8. The
    # signature IS the ModelBackend contract — collapsing it would hide the
    # vLLM-only knobs behind an opaque dict, exactly what D4 forbids.
    def container_config(  # noqa: PLR0913 - the ModelBackend contract, see above
        self,
        model: str,
        gpu_ids: list[str],
        vendor: GpuVendor,
        host_port: int,
        data_dir: str,
        gpu_memory_mb: int | None = None,
        *,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> ContainerConfig:
        """Build the launch config for a model server container.

        *gpu_memory_mb* is the total VRAM of the **smallest** allocated GPU
        (`None` when no GPU is allocated or the size is unknown); a backend
        may use it to pick safer engine defaults. *max_model_len* /
        *gpu_memory_utilization* are the per-serve typed overrides —
        backends that have no such knob ignore both.
        """
        ...

    def endpoint(self, host: str, host_port: int) -> str:
        """OpenAI-compatible base URL for the served model (Invariant #1)."""
        ...

    async def health(self, base_url: str) -> bool:
        """Return `True` when the model server at *base_url* answers."""
        ...

    async def ensure_model(self, base_url: str, model: str) -> None:
        """Pull *model*'s weights if absent (idempotent when already present).

        Raises `ModelPullError` on failure.
        """
        ...


class OllamaBackend:
    """The first `ModelBackend`: an `ollama/ollama` container.

    Weights persist across container restarts via the system-owned
    `<data_dir>/models/ollama → /root/.ollama` volume. Tests inject an
    `httpx.MockTransport` via `transport` (the `CaddyAdmin` pattern) so
    the HTTP surface is exercised with no real Ollama.
    """

    name = "ollama"
    name_prefix = "ollama"
    container_port = OLLAMA_PORT
    health_path = "/"  # Ollama answers "Ollama is running" at the root.
    launch_env_secret_keys: tuple[str, ...] = ()  # weights come from the Ollama registry.
    requires_gpu = False  # Ollama serves on CPU too (gpus=0 is allowed).

    def __init__(
        self,
        *,
        image: str = DEFAULT_OLLAMA_IMAGE,
        pull_timeout_s: float = DEFAULT_PULL_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._image = image
        self._pull_timeout_s = pull_timeout_s
        self._transport = transport

    @property
    def image(self) -> str:
        return self._image

    # One arg per launch-shape input; P21 D3/D4 widened it from 5 to 8. The
    # signature IS the ModelBackend contract — collapsing it would hide the
    # vLLM-only knobs behind an opaque dict, exactly what D4 forbids.
    def container_config(  # noqa: PLR0913 - the ModelBackend contract, see above
        self,
        model: str,
        gpu_ids: list[str],
        vendor: GpuVendor,
        host_port: int,
        data_dir: str,
        gpu_memory_mb: int | None = None,
        *,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> ContainerConfig:
        """Launch config for the Ollama server (one server serves all models).

        *model* is part of the Protocol signature but unused here: weights are
        pulled after start via `ensure_model`, not baked into the launch.
        The three sizing params are likewise Protocol-only: Ollama picks its own
        VRAM budget and exposes no equivalent knobs, so the returned config is
        byte-identical with or without them (P21 D3/D4 — no Ollama behavior
        change; the route rejects the overrides for non-vLLM backends anyway).
        """
        weights_dir = str(Path(data_dir) / "models" / "ollama")
        return ContainerConfig(
            image=self._image,
            gpu_ids=gpu_ids,
            vendor=vendor,
            env={"OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}"},
            ports={OLLAMA_PORT: host_port},
            volumes={weights_dir: "/root/.ollama"},
        )

    def endpoint(self, host: str, host_port: int) -> str:
        """OpenAI-compatible base URL — Ollama serves the OpenAI API under `/v1`."""
        return f"http://{host}:{host_port}/v1"

    async def health(self, base_url: str) -> bool:
        """`GET <root>/` → 200 (Ollama answers `"Ollama is running"`)."""
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=5.0) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_model(self, base_url: str, model: str) -> None:
        """Pull *model* via the streaming `POST /api/pull` (idempotent).

        A model already present streams a short success and returns — pulls are
        safe to re-fire on every launch/adoption. Errors surface three ways and
        all raise `ModelPullError`: a non-200 response and `{"error": ...}`
        NDJSON lines (which Ollama emits under an HTTP 200) raise the base class;
        a connect-level transport failure (server not listening yet) raises the
        `ModelServerUnreachableError` subclass so the caller can retry.
        """
        root = base_url.rstrip("/")
        timeout = httpx.Timeout(self._pull_timeout_s, connect=10.0)
        try:
            async with (
                httpx.AsyncClient(transport=self._transport, timeout=timeout) as client,
                client.stream("POST", f"{root}/api/pull", json={"model": model}) as resp,
            ):
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise ModelPullError(
                        f"pull of {model!r} failed: HTTP {resp.status_code}: {body[:200]}"
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except ValueError:
                        continue  # tolerate non-JSON noise in the stream
                    if isinstance(payload, dict) and payload.get("error"):
                        raise ModelPullError(f"pull of {model!r} failed: {payload['error']}")
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ) as exc:
            # Server not listening yet — expected during a slow container start.
            # With Docker's userland proxy the host port ACCEPTS the connection
            # before the server inside is up, then resets it: that surfaces as a
            # Read/Write/RemoteProtocol error (often with an empty message), not
            # a ConnectError. All of these are transient — the pull is
            # idempotent, so the controller retries on a later tick.
            raise ModelServerUnreachableError(
                f"pull of {model!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelPullError(f"pull of {model!r} failed: {type(exc).__name__}: {exc}") from exc


class VllmBackend:
    """The second `ModelBackend`: a `vllm/vllm-openai` container.

    Unlike Ollama (one server, many models pulled on demand), vLLM serves
    **exactly one model per process**, baked into the launch via `--model
    <hf-ref>` and downloaded from Hugging Face *during* startup — there is no
    runtime pull API. This fits the `kind=model` row model (one row = one
    container = one served model) with no schema change; it only reshapes two
    Protocol members relative to Ollama:

    * `ensure_model` is a bounded **readiness confirmation** (poll
      `GET /v1/models` until the model is loaded), not a pull. It reuses the
      exact exception contract so the `nerdit.core.models.controller.ModelController`
      retry machinery is untouched: still-starting / still-downloading raises
      `ModelServerUnreachableError` (transient → retry next tick); a
      listening-but-erroring server raises `ModelPullError`.
    * `name_prefix` is `"vllm"` so a vLLM row can never collide with an
      Ollama row for the "same" model in the shared `service_name` index.

    `endpoint()` is byte-identical to Ollama — vLLM serves the OpenAI API
    under `/v1` natively, so Invariant #1 holds with no special-casing.

    Weights persist across restarts via the system-owned
    `<data_dir>/models/huggingface → /root/.cache/huggingface` volume. Gated
    HF repos need `HF_TOKEN`; it is injected from the P8 shared secrets scope
    only when present (`launch_env_secret_keys`), never baked here. Tests
    inject an `httpx.MockTransport` via `transport` exactly as for Ollama.
    """

    name = "vllm"
    name_prefix = "vllm"
    container_port = VLLM_PORT
    health_path = "/health"  # vLLM returns 200 at /health once the engine is ready.
    launch_env_secret_keys: tuple[str, ...] = ("HF_TOKEN",)
    requires_gpu = True  # vLLM has no supported CPU serving path.

    def __init__(
        self,
        *,
        image: str = DEFAULT_VLLM_IMAGE,
        shm_size: str = DEFAULT_VLLM_SHM_SIZE,
        extra_args: list[str] | None = None,
        pull_timeout_s: float = DEFAULT_PULL_TIMEOUT_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._image = image
        self._shm_size = shm_size
        self._extra_args = list(extra_args or [])
        self._pull_timeout_s = pull_timeout_s
        self._transport = transport

    @property
    def image(self) -> str:
        return self._image

    # One arg per launch-shape input; P21 D3/D4 widened it from 5 to 8. The
    # signature IS the ModelBackend contract — collapsing it would hide the
    # vLLM-only knobs behind an opaque dict, exactly what D4 forbids.
    def container_config(  # noqa: PLR0913 - the ModelBackend contract, see above
        self,
        model: str,
        gpu_ids: list[str],
        vendor: GpuVendor,
        host_port: int,
        data_dir: str,
        gpu_memory_mb: int | None = None,
        *,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> ContainerConfig:
        """Launch config for the vLLM OpenAI server (one server = one *model*).

        Unlike Ollama, *model* IS consumed here — it becomes `--model` on the
        launch command; the image's entrypoint is the OpenAI API server. The HF
        cache volume persists downloaded weights across restarts.

        Engine sizing, highest precedence first:

        * **per-serve overrides** (D4) — `max_model_len` /
          `gpu_memory_utilization` come from the `POST /models` typed fields
          stored on the row. They replace any same-named operator flag (stripped
          from `vllm_extra_args`) so the flag appears **exactly once**;
        * **operator flags** — `[models].vllm_extra_args`, verbatim;
        * **VRAM-aware defaults** (D3) — under
          `LOW_VRAM_THRESHOLD_MB` on the smallest allocated GPU, each
          bound is injected only when the operator has not already set it and no
          per-serve override supersedes it. Nothing is injected at or above the
          threshold, nor when *gpu_memory_mb* is `None` (unknown VRAM).
        """
        cache_dir = str(Path(data_dir) / "models" / "huggingface")

        overrides: list[tuple[str, str]] = []
        if max_model_len is not None:
            overrides.append(("--max-model-len", str(max_model_len)))
        if gpu_memory_utilization is not None:
            overrides.append(("--gpu-memory-utilization", str(gpu_memory_utilization)))
        override_flags = {flag for flag, _ in overrides}

        injected: list[str] = []
        if gpu_memory_mb is not None and gpu_memory_mb < LOW_VRAM_THRESHOLD_MB:
            for flag, default in _LOW_VRAM_DEFAULTS:
                if flag in override_flags or _flag_present(self._extra_args, flag):
                    continue
                injected += [flag, default]

        extra = list(self._extra_args)
        for flag in override_flags:
            extra = _strip_flag(extra, flag)

        command = [
            "--model",
            model,
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            *injected,
            *extra,
            *[token for pair in overrides for token in pair],
        ]
        return ContainerConfig(
            image=self._image,
            gpu_ids=gpu_ids,
            vendor=vendor,
            command=command,
            env={"HF_HOME": "/root/.cache/huggingface"},
            ports={VLLM_PORT: host_port},
            volumes={cache_dir: "/root/.cache/huggingface"},
            shm_size=self._shm_size,
        )

    def endpoint(self, host: str, host_port: int) -> str:
        """OpenAI-compatible base URL — vLLM serves the OpenAI API under `/v1`."""
        return f"http://{host}:{host_port}/v1"

    async def health(self, base_url: str) -> bool:
        """`GET <root>/health` → 200 once the vLLM engine is ready."""
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=5.0) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/health")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_model(self, base_url: str, model: str) -> None:
        """Confirm *model* is loaded by polling `GET /v1/models` (bounded, one shot).

        vLLM downloads the weights during container startup (no pull API), so
        this is a *readiness confirmation*, not a pull. It returns cleanly once
        the served model appears in `/v1/models`, flipping the row's
        `model_pulled` flag true. Failure modes map onto the shared contract:

        * server not listening yet, an empty model list, or a non-200 response
          (still downloading / compiling CUDA graphs) → `ModelServerUnreachableError`
          (transient → the controller retries on a later tick, exactly like
          Ollama's slow-start path);
        * a listening server whose response is otherwise a hard HTTP/transport
          error → `ModelPullError`.
        """
        root = base_url.rstrip("/")
        timeout = httpx.Timeout(self._pull_timeout_s, connect=10.0)
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=timeout) as client:
                resp = await client.get(f"{root}/v1/models")
                if resp.status_code != 200:
                    # Engine still bringing up the API — treat as not-ready-yet.
                    raise ModelServerUnreachableError(
                        f"vLLM for {model!r} not ready: HTTP {resp.status_code}"
                    )
                try:
                    served = resp.json().get("data") or []
                except ValueError as exc:
                    raise ModelPullError(
                        f"vLLM for {model!r} returned a non-JSON /v1/models body"
                    ) from exc
                if not served:
                    # API up but no model registered yet — still loading.
                    raise ModelServerUnreachableError(
                        f"vLLM for {model!r} has not registered a model yet"
                    )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ) as exc:
            # Server not listening yet / reset during slow start — transient,
            # identical rationale to OllamaBackend.ensure_model.
            raise ModelServerUnreachableError(
                f"vLLM for {model!r} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelPullError(
                f"readiness check for {model!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
