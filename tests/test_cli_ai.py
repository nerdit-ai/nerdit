"""`nerdit ai` CLI + the AI gateway client methods and MCP tools: verb, path, body, key."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

import nerdit.cli.client as client_mod
from nerdit.cli.client import NerditClient
from nerdit.cli.commands.ai import ai_app
from nerdit.mcp.tools import ai_gateway as mcp_ai

_runner = CliRunner()
REF = "${secrets.shared.OPENROUTER_API_KEY}"


def _recording_client(seen: list, response: dict | None = None) -> NerditClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.method,
                request.url.path,
                dict(request.url.params),
                json.loads(request.content) if request.content else None,
                request.headers.get("idempotency-key"),
            )
        )
        return httpx.Response(200, json=response or {})

    return NerditClient(
        host="localhost", port=9321, token=None, transport=httpx.MockTransport(handler)
    )


@pytest.fixture
def seen(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        client_mod,
        "get_configured_client",
        lambda: _recording_client(calls, {"created": True, "enabled": False, "routes": []}),
    )
    return calls


def test_routes_set_sends_a_reference_and_an_idempotency_key(seen):
    args = ["routes", "set", "fast", "--model", "openai/gpt-4o-mini"]
    args += ["--base-url", "https://openrouter.ai/api/v1", "--key-ref", REF]
    result = _runner.invoke(ai_app, args)
    assert result.exit_code == 0, result.output
    method, path, _, body, idem = seen[0]
    assert (method, path) == ("PUT", "/api/ai-gateway/routes/fast")
    assert body == {
        "provider": "api",
        "model": "openai/gpt-4o-mini",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_ref": REF,
    }
    assert idem
    assert "AI gateway is off" in result.output and "[ai_gateway] enabled" in result.output


def test_read_and_delete_verbs_hit_their_routes(seen):
    for args in (
        ["routes", "list"],
        ["routes", "rm", "fast", "--force"],
        ["keys", "list", "--all"],
        ["keys", "revoke", "web"],
        ["usage", "--service", "web", "--days", "3"],
    ):
        assert _runner.invoke(ai_app, args).exit_code == 0
    assert [(m, p, q) for m, p, q, _, _ in seen] == [
        ("GET", "/api/ai-gateway/routes", {}),
        ("DELETE", "/api/ai-gateway/routes/fast", {"force": "true"}),
        ("GET", "/api/ai-gateway/keys", {"include_revoked": "true"}),
        ("DELETE", "/api/ai-gateway/keys/web", {}),
        ("GET", "/api/ai-gateway/usage", {"days": "3", "service": "web"}),
    ]
    assert all(idem for m, _, _, _, idem in seen if m == "DELETE")


def test_ai_group_is_registered_on_the_root_app():
    from nerdit.cli.app import app

    assert _runner.invoke(app, ["ai", "--help"]).exit_code == 0


@pytest.mark.asyncio
async def test_mcp_tools_project_onto_the_routes():
    seen: list = []
    client = _recording_client(seen)
    await mcp_ai._list_ai_routes_impl(client)
    await mcp_ai._set_ai_route_impl(
        client, "local", provider="ollama", model="llama3:8b", idempotency_key="k1"
    )
    await mcp_ai._remove_ai_route_impl(client, "local")
    await mcp_ai._get_ai_usage_impl(client, days=500)
    assert seen[0][:2] == ("GET", "/api/ai-gateway/routes")
    assert seen[1][1:] == (
        "/api/ai-gateway/routes/local",
        {},
        {"provider": "ollama", "model": "llama3:8b"},
        "k1",
    )
    assert seen[2][0] == "DELETE" and seen[2][4]  # a key is minted when absent
    assert seen[3][2] == {"days": "90"}  # clamped to the route's bound
