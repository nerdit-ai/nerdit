"""Tests for the P5 model backend (S2): ``ModelBackend`` Protocol + ``OllamaBackend``.

Pure-async over an ``httpx.MockTransport`` standing in for the Ollama HTTP API
(the ``FakeCaddy`` pattern from ``test_proxy.py``) — no real Ollama, no
TestClient. Covers ``ensure_model`` (success / in-stream error under HTTP 200 /
HTTP failure / transport failure), ``health``, the OpenAI-contract endpoint
shape, the container launch config, ``sanitize_model_name`` edge cases, and a
parameterized Protocol-conformance suite (a future ``VllmBackend`` joins as one
parameter row).
"""

from __future__ import annotations

import inspect
import json
import re

import httpx
import pytest

from nerdit.core.models import (
    DEFAULT_OLLAMA_IMAGE,
    DEFAULT_VLLM_IMAGE,
    OLLAMA_PORT,
    VLLM_PORT,
    ModelBackend,
    ModelPullError,
    ModelServerUnreachableError,
    OllamaBackend,
    VllmBackend,
    sanitize_model_name,
)
from nerdit.db.models import GpuVendor

pytestmark = pytest.mark.asyncio

# The service_name grammar from db/models.py (ServiceCreateRequest.name) —
# every sanitized model name must fit the shared UNIQUE namespace.
SERVICE_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


class FakeOllama:
    """An in-memory Ollama HTTP API for ``httpx.MockTransport``.

    ``pull_lines``: NDJSON payloads streamed by ``POST /api/pull`` (Ollama
    reports pull errors as ``{"error": ...}`` lines under an HTTP 200).
    ``pull_status`` / ``root_status``: override the response codes.
    """

    def __init__(
        self,
        *,
        pull_lines: list[dict] | None = None,
        pull_status: int = 200,
        root_status: int = 200,
    ):
        self.pull_lines = pull_lines if pull_lines is not None else [{"status": "success"}]
        self.pull_status = pull_status
        self.root_status = root_status
        self.requests: list[tuple[str, str, bytes]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/" and request.method == "GET":
            return httpx.Response(self.root_status)
        if request.url.path == "/api/pull" and request.method == "POST":
            if self.pull_status != 200:
                return httpx.Response(self.pull_status, text="boom")
            body = "".join(json.dumps(line) + "\n" for line in self.pull_lines)
            return httpx.Response(200, content=body.encode())
        return httpx.Response(404)


def _backend(fake: FakeOllama, **kwargs) -> OllamaBackend:
    return OllamaBackend(transport=httpx.MockTransport(fake.handler), **kwargs)


BASE_URL = "http://127.0.0.1:18080"


# --- 1. ensure_model ------------------------------------------------------------


async def test_ensure_model_success_posts_api_pull():
    fake = FakeOllama(pull_lines=[{"status": "pulling manifest"}, {"status": "success"}])
    await _backend(fake).ensure_model(BASE_URL, "llama3.1:8b")
    assert len(fake.requests) == 1
    method, path, content = fake.requests[0]
    assert (method, path) == ("POST", "/api/pull")
    assert json.loads(content) == {"model": "llama3.1:8b"}


async def test_ensure_model_is_idempotent_when_present():
    # An already-present model streams a short success — no error, safe to re-fire.
    fake = FakeOllama(pull_lines=[{"status": "success"}])
    backend = _backend(fake)
    await backend.ensure_model(BASE_URL, "llama3.1:8b")
    await backend.ensure_model(BASE_URL, "llama3.1:8b")
    assert len(fake.requests) == 2


async def test_ensure_model_error_line_under_http_200_raises():
    fake = FakeOllama(
        pull_lines=[
            {"status": "pulling manifest"},
            {"error": "pull model manifest: file does not exist"},
        ]
    )
    with pytest.raises(ModelPullError, match="file does not exist"):
        await _backend(fake).ensure_model(BASE_URL, "nosuch:model")


async def test_ensure_model_http_failure_raises():
    fake = FakeOllama(pull_status=500)
    with pytest.raises(ModelPullError, match="HTTP 500"):
        await _backend(fake).ensure_model(BASE_URL, "llama3.1:8b")


async def test_ensure_model_transport_failure_raises():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OllamaBackend(transport=httpx.MockTransport(explode))
    with pytest.raises(ModelPullError, match="llama3.1:8b"):
        await backend.ensure_model(BASE_URL, "llama3.1:8b")


async def test_ensure_model_connection_reset_is_retryable():
    # P5-runbook regression: Docker's userland proxy ACCEPTS the TCP connect on
    # the published port before the server inside the container is listening,
    # then resets the request — that surfaces as ReadError (often with an empty
    # message), not ConnectError. It must classify as the retryable subclass or
    # a fresh `nerdit serve` strands the row at model_pulled=false.
    def reset(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("", request=request)

    backend = OllamaBackend(transport=httpx.MockTransport(reset))
    with pytest.raises(ModelServerUnreachableError):
        await backend.ensure_model(BASE_URL, "llama3.1:8b")


async def test_ensure_model_tolerates_non_json_noise():
    class NoisyOllama(FakeOllama):
        def handler(self, request: httpx.Request) -> httpx.Response:
            self.requests.append((request.method, request.url.path, request.content))
            return httpx.Response(200, content=b'not-json\n{"status": "success"}\n\n')

    await _backend(NoisyOllama()).ensure_model(BASE_URL, "llama3.1:8b")


# --- 2. health -------------------------------------------------------------------


async def test_health_200_is_true():
    fake = FakeOllama()
    assert await _backend(fake).health(BASE_URL) is True
    assert fake.requests[0][:2] == ("GET", "/")


async def test_health_500_is_false():
    assert await _backend(FakeOllama(root_status=500)).health(BASE_URL) is False


async def test_health_connect_error_is_false():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = OllamaBackend(transport=httpx.MockTransport(explode))
    assert await backend.health(BASE_URL) is False


# --- 3. endpoint (the OpenAI contract) --------------------------------------------


async def test_endpoint_is_openai_v1_url():
    backend = OllamaBackend()
    assert backend.endpoint("172.17.0.1", 18080) == "http://172.17.0.1:18080/v1"
    assert backend.endpoint("127.0.0.1", 9000) == "http://127.0.0.1:9000/v1"


# --- 4. container_config -----------------------------------------------------------


async def test_container_config_shape():
    backend = OllamaBackend()
    cfg = backend.container_config(
        "llama3.1:8b", ["0", "1"], GpuVendor.nvidia, 18080, "/var/lib/nerdit"
    )
    assert cfg.image == DEFAULT_OLLAMA_IMAGE
    assert cfg.gpu_ids == ["0", "1"]
    assert cfg.vendor == GpuVendor.nvidia
    assert cfg.env == {"OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}"}
    assert cfg.ports == {OLLAMA_PORT: 18080}
    # Weights survive container restarts via the system-owned volume.
    assert cfg.volumes == {"/var/lib/nerdit/models/ollama": "/root/.ollama"}
    # None defers to the image's own CMD (the ollama server entrypoint).
    assert cfg.command is None


async def test_container_config_custom_image_and_amd_vendor():
    backend = OllamaBackend(image="ollama/ollama:0.5")
    cfg = backend.container_config("m:1", ["amd-0"], GpuVendor.amd, 19000, "/data")
    assert cfg.image == "ollama/ollama:0.5"
    assert cfg.vendor == GpuVendor.amd
    assert cfg.ports == {OLLAMA_PORT: 19000}


# --- 5. sanitize_model_name --------------------------------------------------------


async def test_sanitize_the_north_star_ref():
    assert sanitize_model_name("llama3.1:8b") == "ollama-llama3-1-8b"


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("Mixtral:8x7B", "ollama-mixtral-8x7b"),  # uppercase
        ("library/llama3", "ollama-library-llama3"),  # slashes
        ("a..b::c", "ollama-a-b-c"),  # invalid runs collapse to one hyphen
        ("8b:latest", "ollama-8b-latest"),  # leading digit stays valid
        (":tag:", "ollama-tag"),  # leading/trailing separators stripped
    ],
)
async def test_sanitize_edge_cases(ref: str, expected: str):
    assert sanitize_model_name(ref) == expected


async def test_sanitize_truncates_long_refs_deterministically():
    ref = "org/" + "a" * 40 + "." + "b" * 40 + ":8b"
    name = sanitize_model_name(ref)
    assert len(name) <= 63
    assert not name.endswith("-")
    assert name == sanitize_model_name(ref)  # deterministic


@pytest.mark.parametrize(
    "ref",
    ["llama3.1:8b", "Mixtral:8x7B", "library/llama3", "8b:latest", "x" * 100, "a..b::c"],
)
async def test_sanitize_always_matches_service_name_grammar(ref: str):
    name = sanitize_model_name(ref)
    assert SERVICE_NAME_RE.fullmatch(name), name
    assert name.startswith("ollama")


# --- 5b. VllmBackend (P11): one model per server, readiness poll not a pull ---------


class FakeVllm:
    """An in-memory vLLM OpenAI server for ``httpx.MockTransport``.

    ``models``: ids reported by ``GET /v1/models`` (empty ⇒ still loading).
    ``models_status`` / ``health_status``: override the response codes.
    ``models_body``: raw override to exercise the non-JSON path.
    """

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        models_status: int = 200,
        health_status: int = 200,
        models_body: bytes | None = None,
    ):
        self.models = models if models is not None else ["meta-llama/Llama-3.1-8B"]
        self.models_status = models_status
        self.health_status = health_status
        self.models_body = models_body
        self.requests: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if request.url.path == "/health" and request.method == "GET":
            return httpx.Response(self.health_status)
        if request.url.path == "/v1/models" and request.method == "GET":
            if self.models_status != 200:
                return httpx.Response(self.models_status, text="not ready")
            if self.models_body is not None:
                return httpx.Response(200, content=self.models_body)
            data = {"data": [{"id": m} for m in self.models]}
            return httpx.Response(200, json=data)
        return httpx.Response(404)


def _vllm(fake: FakeVllm, **kwargs) -> VllmBackend:
    return VllmBackend(transport=httpx.MockTransport(fake.handler), **kwargs)


async def test_vllm_ensure_model_ready_when_model_registered():
    fake = FakeVllm(models=["meta-llama/Llama-3.1-8B"])
    await _vllm(fake).ensure_model(BASE_URL, "meta-llama/Llama-3.1-8B")
    assert ("GET", "/v1/models") in fake.requests


async def test_vllm_ensure_model_empty_list_is_retryable():
    # API is up but no model registered yet — still loading weights.
    fake = FakeVllm(models=[])
    with pytest.raises(ModelServerUnreachableError):
        await _vllm(fake).ensure_model(BASE_URL, "meta-llama/Llama-3.1-8B")


async def test_vllm_ensure_model_non_200_is_retryable():
    fake = FakeVllm(models_status=503)
    with pytest.raises(ModelServerUnreachableError):
        await _vllm(fake).ensure_model(BASE_URL, "meta-llama/Llama-3.1-8B")


async def test_vllm_ensure_model_connect_error_is_retryable():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = VllmBackend(transport=httpx.MockTransport(explode))
    with pytest.raises(ModelServerUnreachableError):
        await backend.ensure_model(BASE_URL, "m")


async def test_vllm_ensure_model_connection_reset_is_retryable():
    # Same slow-start rationale as Ollama: userland proxy accepts then resets.
    def reset(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("", request=request)

    backend = VllmBackend(transport=httpx.MockTransport(reset))
    with pytest.raises(ModelServerUnreachableError):
        await backend.ensure_model(BASE_URL, "m")


async def test_vllm_ensure_model_non_json_body_raises_pull_error():
    fake = FakeVllm(models_body=b"not-json")
    with pytest.raises(ModelPullError):
        await _vllm(fake).ensure_model(BASE_URL, "m")


async def test_vllm_health_hits_slash_health():
    fake = FakeVllm()
    assert await _vllm(fake).health(BASE_URL) is True
    assert fake.requests[0] == ("GET", "/health")


async def test_vllm_health_500_is_false():
    assert await _vllm(FakeVllm(health_status=500)).health(BASE_URL) is False


async def test_vllm_endpoint_is_openai_v1_url():
    assert VllmBackend().endpoint("172.17.0.1", 8000) == "http://172.17.0.1:8000/v1"


async def test_vllm_container_config_shape():
    backend = VllmBackend()
    cfg = backend.container_config(
        "meta-llama/Llama-3.1-8B", ["0"], GpuVendor.nvidia, 18080, "/var/lib/nerdit"
    )
    assert cfg.image == DEFAULT_VLLM_IMAGE
    assert cfg.gpu_ids == ["0"]
    assert cfg.ports == {VLLM_PORT: 18080}
    # The model IS baked into the launch (unlike Ollama's post-start pull).
    assert cfg.command == [
        "--model",
        "meta-llama/Llama-3.1-8B",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    # HF weights persist across restarts; shm sized above Docker's 64MB default.
    assert cfg.volumes == {"/var/lib/nerdit/models/huggingface": "/root/.cache/huggingface"}
    assert cfg.shm_size == "1g"


async def test_vllm_container_config_threads_extra_args():
    backend = VllmBackend(extra_args=["--gpu-memory-utilization", "0.9"], shm_size="2g")
    cfg = backend.container_config("m", ["0"], GpuVendor.nvidia, 9000, "/data")
    assert cfg.command[-2:] == ["--gpu-memory-utilization", "0.9"]
    assert cfg.shm_size == "2g"


async def test_vllm_requires_gpu_flag():
    assert VllmBackend().requires_gpu is True
    assert OllamaBackend().requires_gpu is False


async def test_sanitize_with_vllm_prefix():
    name = sanitize_model_name("meta-llama/Llama-3.1-8B", prefix="vllm")
    assert name == "vllm-meta-llama-llama-3-1-8b"
    assert SERVICE_NAME_RE.fullmatch(name)


# --- 5c. VRAM-aware vLLM defaults (P21 WP2/D3) + per-serve overrides (WP3/D4) -------


def _vllm_cmd(backend: VllmBackend, **kwargs) -> list[str]:
    """The launch command for a canonical vLLM config, with *kwargs* threaded."""
    cfg = backend.container_config("m", ["0"], GpuVendor.nvidia, 9000, "/data", **kwargs)
    return list(cfg.command or [])


def _count_flag(command: list[str], flag: str) -> int:
    return sum(1 for token in command if token == flag or token.startswith(flag + "="))


def _flag_value(command: list[str], flag: str) -> str | None:
    for i, token in enumerate(command):
        if token == flag and i + 1 < len(command):
            return command[i + 1]
    return None


# --- unit: the pure argv helpers ---


@pytest.mark.parametrize(
    ("args", "flag", "expected"),
    [
        (["--max-model-len", "8192"], "--max-model-len", True),
        (["--max-model-len=8192"], "--max-model-len", True),
        (["--enforce-eager"], "--max-model-len", False),
        ([], "--max-model-len", False),
        # A longer flag sharing the prefix must NOT count as present.
        (["--max-model-length", "4"], "--max-model-len", False),
    ],
)
async def test_unit_flag_present(args, flag, expected):
    from nerdit.core.models.backend import _flag_present

    assert _flag_present(args, flag) is expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--max-model-len", "8192"], []),
        (["--max-model-len=8192"], []),
        (["--a", "1", "--max-model-len", "8192", "--b"], ["--a", "1", "--b"]),
        # Value-less flag at the end: nothing left to swallow.
        (["--enforce-eager", "--max-model-len"], ["--enforce-eager"]),
        # Value-less flag followed by another flag: the neighbour survives.
        (["--max-model-len", "--enforce-eager"], ["--enforce-eager"]),
        (["--enforce-eager"], ["--enforce-eager"]),
    ],
)
async def test_unit_strip_flag(args, expected):
    from nerdit.core.models.backend import _strip_flag

    assert _strip_flag(args, "--max-model-len") == expected


# --- D3: VRAM-gated injection ---


async def test_vllm_injects_bounds_under_16gb():
    command = _vllm_cmd(VllmBackend(), gpu_memory_mb=8192)
    assert _count_flag(command, "--max-model-len") == 1
    assert _count_flag(command, "--gpu-memory-utilization") == 1
    assert _flag_value(command, "--max-model-len") == "4096"
    assert _flag_value(command, "--gpu-memory-utilization") == "0.85"
    # Injected right after the fixed prefix (--model/--host/--port), so operator
    # extra args and per-serve overrides still land later in argv.
    assert command[:6] == ["--model", "m", "--host", "0.0.0.0", "--port", str(VLLM_PORT)]
    assert command[6:8] == ["--max-model-len", "4096"]


async def test_vllm_no_injection_at_threshold():
    from nerdit.core.models.backend import LOW_VRAM_THRESHOLD_MB

    baseline = _vllm_cmd(VllmBackend())
    assert LOW_VRAM_THRESHOLD_MB == 16_384
    assert _vllm_cmd(VllmBackend(), gpu_memory_mb=LOW_VRAM_THRESHOLD_MB) == baseline
    assert _vllm_cmd(VllmBackend(), gpu_memory_mb=24576) == baseline


async def test_vllm_no_injection_when_vram_unknown():
    # Unknown/absent VRAM (CPU-less allocation, unreported memory_mb) → no guess.
    assert _vllm_cmd(VllmBackend(), gpu_memory_mb=None) == _vllm_cmd(VllmBackend())


async def test_vllm_flag_presence_gates_per_flag():
    backend = VllmBackend(extra_args=["--max-model-len", "8192"])
    command = _vllm_cmd(backend, gpu_memory_mb=8192)
    # The operator's value survives untouched; only the *absent* bound is injected.
    assert _count_flag(command, "--max-model-len") == 1
    assert _flag_value(command, "--max-model-len") == "8192"
    assert "4096" not in command
    assert _flag_value(command, "--gpu-memory-utilization") == "0.85"


async def test_vllm_flag_presence_equals_form():
    backend = VllmBackend(extra_args=["--max-model-len=8192"])
    command = _vllm_cmd(backend, gpu_memory_mb=8192)
    assert _count_flag(command, "--max-model-len") == 1
    assert "--max-model-len=8192" in command
    assert "4096" not in command


# --- D4: per-serve overrides beat everything, exactly once ---


async def test_vllm_override_beats_injected_default():
    command = _vllm_cmd(VllmBackend(), gpu_memory_mb=8192, max_model_len=2048)
    assert _count_flag(command, "--max-model-len") == 1
    assert _flag_value(command, "--max-model-len") == "2048"
    assert "4096" not in command
    # The other bound is still injected (independent gating).
    assert _flag_value(command, "--gpu-memory-utilization") == "0.85"


async def test_vllm_override_beats_extra_args():
    backend = VllmBackend(extra_args=["--max-model-len", "8192", "--enforce-eager"])
    command = _vllm_cmd(backend, max_model_len=2048)
    assert _count_flag(command, "--max-model-len") == 1
    assert _flag_value(command, "--max-model-len") == "2048"
    assert "8192" not in command
    # Unrelated operator flags are never stripped.
    assert "--enforce-eager" in command


async def test_vllm_override_beats_extra_args_equals_form():
    backend = VllmBackend(extra_args=["--gpu-memory-utilization=0.9"])
    command = _vllm_cmd(backend, gpu_memory_utilization=0.5)
    assert _count_flag(command, "--gpu-memory-utilization") == 1
    assert _flag_value(command, "--gpu-memory-utilization") == "0.5"
    assert "--gpu-memory-utilization=0.9" not in command


async def test_vllm_util_float_formatting():
    command = _vllm_cmd(VllmBackend(), gpu_memory_utilization=0.5)
    assert _flag_value(command, "--gpu-memory-utilization") == "0.5"


async def test_vllm_both_overrides_without_vram_hint():
    command = _vllm_cmd(VllmBackend(), max_model_len=1024, gpu_memory_utilization=0.7)
    assert command == [
        "--model",
        "m",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.7",
    ]


async def test_ollama_container_config_byte_identical_with_params():
    # D3: signature widening only — Ollama's launch shape must not move.
    backend = OllamaBackend()
    plain = backend.container_config("llama3.1:8b", ["0"], GpuVendor.nvidia, 18080, "/data")
    with_params = backend.container_config(
        "llama3.1:8b",
        ["0"],
        GpuVendor.nvidia,
        18080,
        "/data",
        8192,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    assert with_params == plain


# --- 6. Protocol conformance (Ollama + vLLM) ----------------------------------------


BACKENDS = [OllamaBackend, VllmBackend]


@pytest.mark.parametrize("backend_cls", BACKENDS, ids=lambda c: c.__name__)
class TestModelBackendConformance:
    """Structural contract every backend must satisfy (Invariant #1)."""

    async def test_satisfies_protocol(self, backend_cls: type):
        backend: ModelBackend = backend_cls()
        assert isinstance(backend, ModelBackend)

    async def test_name_is_nonempty_str(self, backend_cls: type):
        assert isinstance(backend_cls().name, str)
        assert backend_cls().name

    async def test_sync_method_signatures(self, backend_cls: type):
        params = list(inspect.signature(backend_cls.container_config).parameters)
        assert params[1:] == [
            "model",
            "gpu_ids",
            "vendor",
            "host_port",
            "data_dir",
            # (P21 D3/D4) VRAM hint + the two typed engine overrides; every
            # backend accepts them, Ollama ignores all three.
            "gpu_memory_mb",
            "max_model_len",
            "gpu_memory_utilization",
        ]
        params = list(inspect.signature(backend_cls.endpoint).parameters)
        assert params[1:] == ["host", "host_port"]

    async def test_async_method_signatures(self, backend_cls: type):
        assert inspect.iscoroutinefunction(backend_cls.health)
        assert list(inspect.signature(backend_cls.health).parameters)[1:] == ["base_url"]
        assert inspect.iscoroutinefunction(backend_cls.ensure_model)
        assert list(inspect.signature(backend_cls.ensure_model).parameters)[1:] == [
            "base_url",
            "model",
        ]

    async def test_endpoint_speaks_the_openai_contract(self, backend_cls: type):
        url = backend_cls().endpoint("127.0.0.1", 12345)
        assert url == "http://127.0.0.1:12345/v1"
