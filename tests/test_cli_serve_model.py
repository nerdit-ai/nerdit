"""Unit tests for the S7 `nerdit serve <model>` positional heuristic (P5).

Drives ``_serve_async`` directly (pure-async, no TestClient) with a real
``NerditClient`` over ``httpx.MockTransport`` (the ``test_cli_models`` pattern)
so the classification is asserted end-to-end: model refs hit ``POST
/api/models`` with an ``Idempotency-Key``, existing directories keep the
shipped ``POST /api/services`` app path byte-identical, and ambiguous
positionals error before any HTTP call — and before the ``find_project_config``
ancestor walk-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import typer

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.serve import _is_model_ref, _serve_async


def _mock_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


def _patch_client(monkeypatch, client) -> None:
    import nerdit.cli.client as client_mod

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: client)


# _serve_async's positional signature, in order, with the defaults the CLI
# would pass. Keeping it as one mapping keeps the helper under ruff's max-args.
_SERVE_DEFAULTS = {
    "image": None,
    "name": None,
    "port": None,
    "gpus": None,
    "backend": None,
    "restart_policy": None,
    "command": None,
    "script": None,
    "health_path": None,
    "max_model_len": None,
    "gpu_memory_utilization": None,
}


async def _run_serve(path, **overrides):
    """Invoke _serve_async with keyword-shaped defaults for readability."""
    assert not set(overrides) - set(_SERVE_DEFAULTS), f"unknown flag(s): {overrides}"
    args = {**_SERVE_DEFAULTS, **overrides}
    await _serve_async(path, *(args[key] for key in _SERVE_DEFAULTS))


# ---- classification unit ----


@pytest.mark.parametrize(
    "arg",
    [
        "llama3.1:8b",
        "phi3.5",
        "library/llama:tag",
        "llama3.1",
        "mistral:7b-instruct-q4_0",
        # HF repo ids with no tag or dot must classify as model refs (vLLM).
        "google/gemma-2-2b-it",
        "facebook/opt-125m",
        "library/llama3",
    ],
)
def test_is_model_ref_accepts_model_shapes(arg):
    assert _is_model_ref(arg)


@pytest.mark.parametrize(
    "arg",
    [
        "myapp",  # bare word: no ':' or '.' → ambiguous, never a model
        "./my-app",  # leading '.' signals path intent
        "../my-app",
        "/abs/path",
        "~/my-app",
        "my app:tag",  # space breaks the strict grammar
        "bad::ref",  # only one ':tag' group allowed
    ],
)
def test_is_model_ref_rejects_path_intent_and_bare_words(arg):
    assert not _is_model_ref(arg)


# ---- model path: POST /api/models ----


@pytest.mark.asyncio
async def test_model_ref_dispatches_to_models_endpoint(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "id": "mdl-1",
                "name": "ollama-llama3-1-8b",
                "model": "llama3.1:8b",
                "status": "building",
            },
        )

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("llama3.1:8b", gpus=1)

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/models"
    assert seen["idem"]  # a fresh Idempotency-Key was minted
    assert seen["body"] == {"model": "llama3.1:8b", "gpus": 1}

    out = capsys.readouterr().out
    assert "ollama-llama3-1-8b" in out  # sanitized service name
    assert "nerdit models list" in out  # watch hint
    assert "once the model is pulled" in out  # endpoint-readiness note


@pytest.mark.asyncio
async def test_model_ref_passes_name_override_and_defaults_gpus(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "chat", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("phi3:mini", name="chat")

    assert seen["body"] == {"model": "phi3:mini", "gpus": 0, "name": "chat"}


@pytest.mark.asyncio
async def test_model_ref_passes_backend(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "v", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("meta-llama/Llama-3.1-8B", gpus=1, backend="vllm")

    assert seen["body"] == {"model": "meta-llama/Llama-3.1-8B", "gpus": 1, "backend": "vllm"}


@pytest.mark.asyncio
async def test_model_ref_daemon_error_exits_nonzero(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "service.name_taken", "message": "taken"})

    _patch_client(monkeypatch, _mock_client(handler))
    with pytest.raises(typer.Exit) as excinfo:
        await _run_serve("llama3.1:8b")
    assert excinfo.value.exit_code == 1


# ---- model ref combined with app-only flags: loud error ----


@pytest.mark.asyncio
async def test_model_ref_with_app_only_flag_errors_loudly(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "200")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover — must not fire
        raise AssertionError("no HTTP call expected")

    _patch_client(monkeypatch, _mock_client(handler))
    with pytest.raises(typer.Exit) as excinfo:
        await _run_serve("llama3.1:8b", image="demo:1", port=9000)

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "--image" in out
    assert "--port" in out
    assert "model reference" in out


# ---- P21 D4: the vLLM engine bounds ride the model path, error on the app path ----


@pytest.mark.asyncio
async def test_engine_bounds_reach_the_models_endpoint(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "v", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve(
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpus=1,
        backend="vllm",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
    )

    assert seen["body"] == {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "gpus": 1,
        "backend": "vllm",
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.8,
    }


@pytest.mark.asyncio
async def test_engine_bounds_omitted_when_unset(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "v", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("llama3.1:8b")

    assert "max_model_len" not in seen["body"]
    assert "gpu_memory_utilization" not in seen["body"]


@pytest.mark.asyncio
async def test_engine_bounds_on_app_path_error_loudly(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("COLUMNS", "200")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover — must not fire
        raise AssertionError("no HTTP call expected")

    _patch_client(monkeypatch, _mock_client(handler))
    with pytest.raises(typer.Exit) as excinfo:
        await _run_serve(str(tmp_path), image="demo:1", name="demo", max_model_len=2048)

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "--max-model-len" in out
    assert "app services" in out


# ---- app path: existing directory keeps the shipped services path ----


@pytest.mark.asyncio
async def test_existing_directory_keeps_legacy_services_path(monkeypatch, tmp_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve(str(tmp_path), image="demo:1", name="demo")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/services"
    assert seen["idem"]
    assert seen["body"]["name"] == "demo"
    assert seen["body"]["image"] == "demo:1"
    assert seen["body"]["port"] == 8000  # P2 defaults unchanged


@pytest.mark.asyncio
async def test_existing_directory_wins_over_model_shape(monkeypatch, tmp_path):
    # A directory whose *name* looks like a model ref is still an app path.
    app_dir = tmp_path / "llama3.1:8b"
    app_dir.mkdir()
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve(str(app_dir), image="demo:1", name="demo")
    assert seen["path"] == "/api/services"


# ---- neither shape: explicit error, no ancestor nerdit.toml pickup ----


@pytest.mark.asyncio
async def test_bare_word_errors_without_ancestor_toml_pickup(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("COLUMNS", "200")
    # An ancestor nerdit.toml with a valid [deploy] would silently register
    # 'ancestor-app' if the walk-up ran for the nonexistent positional.
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "ancestor-app"\nport = 8000\n')
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)

    import nerdit.cli.client as client_mod

    def _no_client():  # pragma: no cover — must not fire
        raise AssertionError("no client expected: the error branch precedes any HTTP")

    monkeypatch.setattr(client_mod, "get_configured_client", _no_client)

    with pytest.raises(typer.Exit) as excinfo:
        await _run_serve("myapp")

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "myapp" in out
    assert "./path" in out
    assert "name:tag" in out
    assert "ancestor-app" not in out


@pytest.mark.asyncio
async def test_nonexistent_dot_path_is_never_a_model(monkeypatch, tmp_path, capsys):
    # './ghost-app' contains '.', but leading-dot path intent must not become a
    # model serve — it errors like any other nonexistent app path.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)

    import nerdit.cli.client as client_mod

    def _no_client():  # pragma: no cover — must not fire
        raise AssertionError("no client expected")

    monkeypatch.setattr(client_mod, "get_configured_client", _no_client)

    with pytest.raises(typer.Exit):
        await _run_serve("./ghost-app")
    out = capsys.readouterr().out
    assert "ghost-app" in out
    assert not Path("./ghost-app").exists()


# ---- P11: slash HF refs + explicit --backend reach the model path ----


@pytest.mark.asyncio
async def test_slash_hf_ref_with_backend_dispatches_to_model(monkeypatch):
    # Regression (Codex, PR #62): a HF repo id with no ':'/'.' must serve, not
    # error as a nonexistent app path — especially with an explicit --backend.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "v", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("google/gemma-2-2b-it", gpus=1, backend="vllm")

    assert seen["path"] == "/api/models"
    assert seen["body"] == {"model": "google/gemma-2-2b-it", "gpus": 1, "backend": "vllm"}


@pytest.mark.asyncio
async def test_explicit_backend_forces_model_for_bare_word(monkeypatch):
    # A bare word the shape heuristic can't classify still serves when --backend
    # makes the intent explicit.
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1", "name": "m", "status": "building"})

    _patch_client(monkeypatch, _mock_client(handler))
    await _run_serve("mymodel", gpus=1, backend="ollama")

    assert seen["path"] == "/api/models"
    assert seen["body"]["model"] == "mymodel"
    assert seen["body"]["backend"] == "ollama"


@pytest.mark.asyncio
async def test_path_intent_wins_over_explicit_backend(monkeypatch, tmp_path, capsys):
    # A leading './' is path intent even with --backend: never a silent model serve.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)

    import nerdit.cli.client as client_mod

    def _no_client():  # pragma: no cover — must not fire
        raise AssertionError("no client expected: path intent errors before any HTTP")

    monkeypatch.setattr(client_mod, "get_configured_client", _no_client)

    with pytest.raises(typer.Exit) as excinfo:
        await _run_serve("./ghost", backend="vllm")
    assert excinfo.value.exit_code == 1
    assert "ghost" in capsys.readouterr().out
