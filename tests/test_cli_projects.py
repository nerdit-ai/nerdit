"""Unit tests for the P40b projects CLI: client methods and command dispatch."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.projects import (
    _create_async,
    _delete_async,
    _list_async,
    _show_async,
    projects_app,
)


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- client methods: verb / path / body ----


@pytest.mark.asyncio
async def test_list_projects_gets_bounded_page():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    await client.list_projects(limit=10, cursor="abc")
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/projects"
    assert seen["params"] == {"limit": "10", "cursor": "abc"}


@pytest.mark.asyncio
async def test_create_project_posts_name_with_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "prj_x", "name": "asso"})

    client = _make_client(handler)
    result = await client.create_project("asso", idempotency_key="key-1")
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/projects"
    assert seen["body"] == {"name": "asso"}
    assert seen["idem"] == "key-1"
    assert result["id"] == "prj_x"


@pytest.mark.asyncio
async def test_get_project_targets_name():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/projects/asso"
        return httpx.Response(200, json={"id": "prj_x", "name": "asso"})

    client = _make_client(handler)
    assert (await client.get_project("asso"))["name"] == "asso"


@pytest.mark.asyncio
async def test_delete_project_sends_purge_csv():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"name": "asso", "deleted": ["asso"]})

    client = _make_client(handler)
    await client.delete_project("asso", purge="secrets,data")
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/projects/asso"
    assert seen["params"] == {"purge": "secrets,data"}


# ---- command dispatch (fake client) ----


_PROJECT = {
    "id": "prj_x",
    "name": "asso",
    "services": [
        {"name": "asso", "status": "running", "restart_count": 0, "gpu_ids": [], "endpoint": None}
    ],
    "addresses": [{"kind": "default", "url": "http://asso.local", "state": "ready"}],
    "resources": [
        {
            "service": "asso",
            "type": "db",
            "binding": "default",
            "provider": "managed",
            "target": "pg1",
            "ready": True,
        }
    ],
    "home": {"hostname": "node-a", "node_id": None},
}


class _FakeClient:
    def __init__(self, *, delete_error: Exception | None = None):
        self.calls: list[tuple] = []
        self.delete_error = delete_error

    async def list_projects(self, limit=50, cursor=None):
        self.calls.append(("list_projects", limit, cursor))
        return {"items": [_PROJECT], "next_cursor": None}

    async def create_project(self, name, *, idempotency_key=None):
        self.calls.append(("create_project", name, idempotency_key))
        return {"id": "prj_x", "name": name, "services": [], "addresses": []}

    async def get_project(self, name):
        self.calls.append(("get_project", name))
        return _PROJECT

    async def delete_project(self, name, *, purge="secrets", idempotency_key=None):
        self.calls.append(("delete_project", name, purge, idempotency_key))
        if self.delete_error is not None:
            raise self.delete_error
        return {"name": name, "deleted": ["asso"]}


@pytest.fixture()
def fake_client(monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient()
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_list_renders_table(fake_client, capsys):
    await _list_async(json_out=False)
    assert fake_client.calls == [("list_projects", 50, None)]
    out = capsys.readouterr().out
    assert "asso" in out
    assert "http://asso.local" in out


@pytest.mark.asyncio
async def test_create_calls_client_with_minted_key(fake_client, capsys):
    await _create_async("asso")
    call = fake_client.calls[0]
    assert call[:2] == ("create_project", "asso")
    assert call[2]  # a minted idempotency key
    assert "prj_x" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_show_renders_services_resources_addresses(fake_client, capsys):
    await _show_async("asso", json_out=False)
    assert fake_client.calls == [("get_project", "asso")]
    out = capsys.readouterr().out
    for needle in ("asso", "node-a", "pg1", "http://asso.local"):
        assert needle in out


@pytest.mark.asyncio
async def test_delete_passes_purge_and_prints_deleted(fake_client, capsys):
    await _delete_async("asso", purge="secrets,images")
    call = fake_client.calls[0]
    assert call[:3] == ("delete_project", "asso", "secrets,images")
    assert call[3]
    assert "Deleted project asso" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_delete_incomplete_lists_failed_services(monkeypatch, capsys):
    import typer

    import nerdit.cli.client as client_mod

    response = httpx.Response(
        409,
        json={
            "code": "project.delete_incomplete",
            "message": "Project 'asso' was not deleted: 1 service(s) remain.",
            "hint": "Fix the failures listed under `failed` and re-run the delete.",
            "deleted": [],
            "failed": [{"name": "asso", "code": "resource.in_use", "message": "still bound"}],
        },
        request=httpx.Request("DELETE", "http://localhost:9321/api/projects/asso"),
    )
    error = httpx.HTTPStatusError("409", request=response.request, response=response)
    fake = _FakeClient(delete_error=error)
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    with pytest.raises(typer.Exit):
        await _delete_async("asso", purge="secrets")
    out = capsys.readouterr().out
    assert "resource.in_use" in out
    assert "still bound" in out


# ---- argv-level parsing (CliRunner) ----

_runner = CliRunner()


def test_cli_list_json_emits_parseable_json(fake_client):
    result = _runner.invoke(projects_app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["items"][0]["name"] == "asso"


def test_cli_show_json_emits_parseable_json(fake_client):
    result = _runner.invoke(projects_app, ["show", "asso", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["id"] == "prj_x"


def test_cli_delete_data_purge_confirms_unless_yes(fake_client):
    refused = _runner.invoke(projects_app, ["delete", "asso", "--purge", "data"], input="n\n")
    assert refused.exit_code != 0
    assert fake_client.calls == []
    accepted = _runner.invoke(projects_app, ["delete", "asso", "--purge", "data", "--yes"])
    assert accepted.exit_code == 0, accepted.output
    assert fake_client.calls[0][:3] == ("delete_project", "asso", "data")


def test_cli_create_parses_name(fake_client):
    result = _runner.invoke(projects_app, ["create", "asso"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:2] == ("create_project", "asso")


@pytest.mark.asyncio
async def test_rename_project_uses_id_patch_and_idempotency_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == "/api/projects/prj_test"
        assert request.headers["Idempotency-Key"] == "rename-key"
        assert json.loads(request.content) == {"name": "new-label"}
        return httpx.Response(200, json={"id": "prj_test", "name": "new-label"})

    result = await _make_client(handler).rename_project(
        "prj_test", "new-label", idempotency_key="rename-key"
    )
    assert result["name"] == "new-label"
