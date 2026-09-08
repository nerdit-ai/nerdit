"""Unit tests for the P11.5 ``nerdit store`` CLI (list / show / deploy)."""

from __future__ import annotations

import httpx
import pytest
import typer

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.store import _deploy_async, _list_async, _show_async


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- client methods ----


@pytest.mark.asyncio
async def test_client_list_app_templates_hits_api_prefix():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        return httpx.Response(200, json=[{"id": "node-starter"}])

    client = _make_client(handler)
    result = await client.list_app_templates()
    assert result == [{"id": "node-starter"}]
    assert captured == {"method": "GET", "path": "/api/app-templates"}


@pytest.mark.asyncio
async def test_client_get_app_template():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"id": "node-starter", "name": "Node Starter"})

    client = _make_client(handler)
    result = await client.get_app_template("node-starter")
    assert result["name"] == "Node Starter"
    assert captured["path"] == "/api/app-templates/node-starter"


@pytest.mark.asyncio
async def test_client_deploy_template_posts_json_and_idempotency_key():
    import json

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    client = _make_client(handler)
    await client.deploy_template(
        "node-starter",
        name="demo",
        env={"A": "1"},
        secrets={"K": "v"},
        port=3000,
        idempotency_key="key-tpl",
    )
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/app-templates/node-starter/deploy"
    assert captured["idem"] == "key-tpl"
    assert captured["body"] == {
        "name": "demo",
        "env": {"A": "1"},
        "secrets": {"K": "v"},
        "port": 3000,
    }


@pytest.mark.asyncio
async def test_client_deploy_template_omits_unset_optional_fields():
    import json

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.deploy_template("node-starter", name="demo", idempotency_key="k")
    assert captured["body"] == {"name": "demo"}


# ---- command flow (fake client) ----


class _FakeClient:
    def __init__(self):
        self.templates = [
            {
                "id": "node-starter",
                "name": "Node Starter",
                "category": "web",
                "description": "A minimal Node/Express web app",
                "repo_url": "https://github.com/o/tpl",
                "ref": "v1.0.0",
                "subdir": "node-starter",
                "deploy_defaults": {"port": 3000, "gpus": 0, "start": None, "health": None},
                "env_schema": [
                    {"name": "API_KEY", "description": "key", "required": True, "secret": True}
                ],
                "ai_hint": "expects ollama",
            }
        ]
        self.deploy_template_calls: list[dict] = []

    async def list_app_templates(self):
        return self.templates

    async def get_app_template(self, template_id):
        for tpl in self.templates:
            if tpl["id"] == template_id:
                return tpl
        raise AssertionError("unexpected id")

    async def deploy_template(self, template_id, **kwargs):
        self.deploy_template_calls.append({"template_id": template_id, **kwargs})
        return {
            "id": "svc-1",
            "name": kwargs["name"],
            "status": "building",
            "endpoint": {"public_url": f"https://host/{kwargs['name']}"},
        }


@pytest.fixture()
def fake_client(monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient()
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_store_list_renders_catalog(fake_client, capsys):
    await _list_async()
    out = capsys.readouterr().out
    assert "node-starter" in out
    assert "web" in out


@pytest.mark.asyncio
async def test_store_show_renders_detail(fake_client, capsys):
    await _show_async("node-starter")
    out = capsys.readouterr().out
    assert "https://github.com/o/tpl" in out
    assert "v1.0.0" in out
    assert "API_KEY" in out
    assert "expects ollama" in out


@pytest.mark.asyncio
async def test_store_deploy_forwards_env_and_secret_pairs(fake_client):
    await _deploy_async(
        "node-starter",
        "demo",
        ["A=1", "B=two"],
        ["K=secretval"],
        3000,
        None,
        None,
        None,
        None,
    )
    assert len(fake_client.deploy_template_calls) == 1
    call = fake_client.deploy_template_calls[0]
    assert call["template_id"] == "node-starter"
    assert call["name"] == "demo"
    assert call["env"] == {"A": "1", "B": "two"}
    assert call["secrets"] == {"K": "secretval"}
    assert call["port"] == 3000
    assert call["idempotency_key"]


@pytest.mark.asyncio
async def test_store_deploy_rejects_malformed_pair(fake_client):
    with pytest.raises(typer.Exit):
        await _deploy_async("node-starter", "demo", ["NOEQ"], [], None, None, None, None, None)
    assert fake_client.deploy_template_calls == []


@pytest.mark.asyncio
async def test_store_deploy_exits_on_client_error(monkeypatch, capsys):
    import nerdit.cli.client as client_mod

    class _ErrClient:
        async def deploy_template(self, template_id, **kwargs):
            request = httpx.Request("POST", "http://localhost:9321/api/app-templates/x/deploy")
            response = httpx.Response(
                403,
                request=request,
                json={"code": "deploy.git_disabled", "message": "Git deploy is disabled"},
            )
            raise httpx.HTTPStatusError("403", request=request, response=response)

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _ErrClient())

    with pytest.raises(typer.Exit):
        await _deploy_async("x", "demo", [], [], None, None, None, None, None)
    out = capsys.readouterr().out
    assert "Forbidden" in out
