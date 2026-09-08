"""Unit tests for the P5 models CLI: client methods and table render.

Client methods are driven through ``httpx.MockTransport`` (the ``test_mcp``
pattern) so the request path, JSON body, query params and ``Idempotency-Key``
header are asserted without a daemon; the table render is a capsys smoke.
"""

from __future__ import annotations

import json

import httpx
import pytest

from nerdit.cli.client import NerditClient
from nerdit.cli.display import display_model_table


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- client method header / payload behavior ----


@pytest.mark.asyncio
async def test_serve_model_sends_body_and_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201, json={"id": "mdl-1", "name": "ollama-llama3-1-8b", "status": "building"}
        )

    client = _make_client(handler)
    out = await client.serve_model("llama3.1:8b", gpus=1, idempotency_key="key-123")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/models"
    assert seen["idem"] == "key-123"
    assert seen["body"] == {"model": "llama3.1:8b", "gpus": 1}
    assert out["name"] == "ollama-llama3-1-8b"


@pytest.mark.asyncio
async def test_serve_model_sends_name_override_and_omits_key_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1"})

    client = _make_client(handler)
    await client.serve_model("llama3.1:8b", name="chat-model")

    assert seen["idem"] is None
    assert seen["body"] == {"model": "llama3.1:8b", "gpus": 0, "name": "chat-model"}


@pytest.mark.asyncio
async def test_list_models_hits_api_prefix_with_params():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    page = await client.list_models(limit=10, cursor="abc")

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/models"
    assert seen["params"] == {"limit": "10", "cursor": "abc"}
    assert page == {"items": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_list_models_omits_cursor_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    await _make_client(handler).list_models()
    assert seen["params"] == {"limit": "50"}


# ---- table render ----


def test_display_model_table_renders_fields(capsys, monkeypatch):
    # Wide virtual terminal so Rich never wraps the endpoint URL mid-assertion.
    monkeypatch.setenv("COLUMNS", "200")
    models = [
        {
            "name": "ollama-llama3-1-8b",
            "model": "llama3.1:8b",
            "status": "running",
            "model_pulled": True,
            "gpu_count": 1,
            "gpu_utilization": {"GPU-1": 42},
            "endpoint": "http://127.0.0.1:9500/v1",
        },
        {
            "name": "ollama-phi3-mini",
            "model": "phi3:mini",
            "status": "building",
            "model_pulled": False,
            "gpu_count": 0,
            "gpu_utilization": {},
            "endpoint": None,
        },
    ]
    display_model_table(models)
    out = capsys.readouterr().out
    assert "llama3.1:8b" in out
    assert "running" in out
    assert "yes" in out
    assert "42" in out
    assert "9500" in out
    assert "phi3:mini" in out
    assert "building" in out
    assert "no" in out


def test_display_model_table_renders_unmapped_status(capsys, monkeypatch):
    # P5-runbook regression: a status outside the style map (e.g. a terminal
    # 'completed' row) used to render as "[]completed[/]" and crash Rich with
    # a MarkupError — the whole `nerdit models list` command died on it.
    monkeypatch.setenv("COLUMNS", "200")
    display_model_table(
        [
            {
                "name": "ollama-old",
                "model": "llama3.1:8b",
                "status": "completed",
                "model_pulled": True,
                "gpu_count": 0,
                "gpu_utilization": {},
                "endpoint": None,
            }
        ]
    )
    assert "completed" in capsys.readouterr().out


# ---- command wiring ----


@pytest.mark.asyncio
async def test_models_list_command_renders_page(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import models as models_mod

    class _C:
        async def list_models(self, limit: int = 50, cursor: str | None = None):
            return {
                "items": [
                    {
                        "name": "ollama-llama3-1-8b",
                        "model": "llama3.1:8b",
                        "status": "running",
                        "model_pulled": True,
                        "gpu_count": 0,
                        "gpu_utilization": {},
                        "endpoint": "http://127.0.0.1:9500/v1",
                    }
                ],
                "next_cursor": None,
            }

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    await models_mod._list_async()
    out = capsys.readouterr().out
    assert "llama3.1:8b" in out


@pytest.mark.asyncio
async def test_models_list_command_empty_hint(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import models as models_mod

    class _C:
        async def list_models(self, limit: int = 50, cursor: str | None = None):
            return {"items": [], "next_cursor": None}

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    await models_mod._list_async()
    out = capsys.readouterr().out
    assert "No models" in out


def test_models_app_registered():
    from nerdit.cli.app import app

    names = {t.name for t in app.registered_groups}
    assert "models" in names
