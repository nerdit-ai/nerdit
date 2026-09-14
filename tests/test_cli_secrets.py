"""Unit tests for the P4 secrets CLI: client methods and command dispatch."""

from __future__ import annotations

import json

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.secrets import (
    _list_async,
    _resolve_service,
    _rm_async,
    _rotate_key_async,
    _set_async,
    secrets_app,
    secrets_rm,
    secrets_rotate_key,
    secrets_set,
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
async def test_set_secrets_posts_values_envelope():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"service": "demo", "keys": ["A", "B"]})

    client = _make_client(handler)
    result = await client.set_secrets("demo", {"A": "1", "B": "2"})
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/secrets/demo"
    assert seen["body"] == {"values": {"A": "1", "B": "2"}}
    assert result["keys"] == ["A", "B"]


@pytest.mark.asyncio
async def test_list_secrets_gets_names_only():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/secrets/demo"
        return httpx.Response(200, json={"service": "demo", "keys": ["A"]})

    client = _make_client(handler)
    result = await client.list_secrets("demo")
    assert result == {"service": "demo", "keys": ["A"]}


@pytest.mark.asyncio
async def test_delete_secret_targets_single_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"service": "demo", "deleted": "A"})

    client = _make_client(handler)
    await client.delete_secret("demo", "A")
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/secrets/demo/A"


@pytest.mark.asyncio
async def test_delete_secrets_targets_whole_service():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"service": "demo", "deleted": True})

    client = _make_client(handler)
    await client.delete_secrets("demo")
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/secrets/demo"


# ---- command dispatch (fake client) ----


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    async def set_secrets(self, service, values, *, idempotency_key=None):
        self.calls.append(("set_secrets", service, values, idempotency_key))
        return {"service": service, "keys": sorted(values)}

    async def list_secrets(self, service):
        self.calls.append(("list_secrets", service))
        return {"service": service, "keys": ["API_KEY"]}

    async def delete_secret(self, service, key, *, idempotency_key=None):
        self.calls.append(("delete_secret", service, key, idempotency_key))
        return {"service": service, "deleted": key}

    async def delete_secrets(self, service, *, idempotency_key=None):
        self.calls.append(("delete_secrets", service, idempotency_key))
        return {"service": service, "deleted": True}

    async def rotate_secrets_key(self, *, idempotency_key=None):
        self.calls.append(("rotate_secrets_key", idempotency_key))
        return {"services_rewritten": 3}


@pytest.fixture()
def fake_client(monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient()
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_set_calls_client_and_prints_names_only(fake_client, capsys):
    await _set_async("demo", {"API_KEY": "s3cret", "OTHER": "v"})
    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call[:3] == ("set_secrets", "demo", {"API_KEY": "s3cret", "OTHER": "v"})
    out = capsys.readouterr().out
    assert "API_KEY" in out
    assert "s3cret" not in out  # values are never shown


def test_set_command_rejects_pair_without_equals(fake_client):
    with pytest.raises(typer.Exit):
        secrets_set("demo", ["NOEQUALS"])
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_list_calls_client_and_prints_keys(fake_client, capsys):
    await _list_async("demo")
    assert fake_client.calls == [("list_secrets", "demo")]
    assert "API_KEY" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_rm_with_key_deletes_single_secret(fake_client):
    await _rm_async("demo", "API_KEY")
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][:3] == ("delete_secret", "demo", "API_KEY")


@pytest.mark.asyncio
async def test_rm_without_key_deletes_all_secrets(fake_client):
    await _rm_async("demo", None)
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][:2] == ("delete_secrets", "demo")


# ---- P8: --shared sugar, rotate-key, minted Idempotency-Keys ----


def test_resolve_service_shared_flag_maps_to_shared_scope():
    assert _resolve_service(None, shared=True) == "shared"


def test_resolve_service_plain_service_passthrough():
    assert _resolve_service("demo", shared=False) == "demo"


def test_resolve_service_rejects_both_service_and_shared():
    with pytest.raises(typer.Exit):
        _resolve_service("demo", shared=True)


def test_resolve_service_rejects_neither_service_nor_shared():
    with pytest.raises(typer.Exit):
        _resolve_service(None, shared=False)


def test_secrets_set_shared_flag_targets_shared_scope(fake_client):
    secrets_set(None, ["A=1"], shared=True)
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][:2] == ("set_secrets", "shared")


@pytest.mark.asyncio
async def test_secrets_list_shared_flag_targets_shared_scope(fake_client, capsys):
    await _list_async(_resolve_service(None, shared=True))
    assert fake_client.calls == [("list_secrets", "shared")]


def test_secrets_rm_shared_flag_targets_shared_scope(fake_client):
    secrets_rm(None, key="API_KEY", shared=True)
    assert fake_client.calls[0][:3] == ("delete_secret", "shared", "API_KEY")


def test_secrets_set_shared_and_service_both_given_errors(fake_client):
    with pytest.raises(typer.Exit):
        secrets_set("demo", ["A=1"], shared=True)
    assert fake_client.calls == []


# ---- argv-level parsing (CliRunner: exercises click's positional binding) ----

_runner = CliRunner()


def test_cli_set_shared_single_pair_parses_through_argv(fake_client):
    # Click binds the first positional to [SERVICE]; the command must
    # redistribute it as a pair when --shared is given.
    result = _runner.invoke(secrets_app, ["set", "--shared", "FOO=bar"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == ("set_secrets", "shared", {"FOO": "bar"})


def test_cli_set_shared_multiple_pairs_parses_through_argv(fake_client):
    result = _runner.invoke(secrets_app, ["set", "--shared", "A=1", "B=2"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == ("set_secrets", "shared", {"A": "1", "B": "2"})


def test_cli_set_service_parses_through_argv(fake_client):
    result = _runner.invoke(secrets_app, ["set", "demo", "A=1"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == ("set_secrets", "demo", {"A": "1"})


@pytest.mark.parametrize("target", [["demo"], ["--shared"]])
def test_cli_set_prompts_without_echo_and_merges_pairs(fake_client, target):
    result = _runner.invoke(
        secrets_app,
        ["set", *target, "EXISTING=kept", "--prompt", "API_KEY", "--prompt", "OTHER"],
        input="hidden-secret\nvalue=with=equals\n",
    )
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == (
        "set_secrets",
        "shared" if target == ["--shared"] else "demo",
        {"EXISTING": "kept", "API_KEY": "hidden-secret", "OTHER": "value=with=equals"},
    )
    assert "hidden-secret" not in result.output
    assert "value=with=equals" not in result.output


def test_cli_set_prompt_without_pairs(fake_client):
    result = _runner.invoke(secrets_app, ["set", "demo", "--prompt", "API_KEY"], input="hidden\n")
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == ("set_secrets", "demo", {"API_KEY": "hidden"})
    assert "hidden" not in result.output


def test_cli_set_invalid_prompt_key_does_not_prompt_or_write(fake_client):
    result = _runner.invoke(secrets_app, ["set", "demo", "--prompt", "BAD=KEY"])
    assert result.exit_code == 1
    assert "Value for" not in result.output
    assert fake_client.calls == []


def test_cli_set_cancelled_prompt_does_not_write_partial_values(fake_client):
    result = _runner.invoke(secrets_app, ["set", "demo", "A=1", "--prompt", "API_KEY"])
    assert result.exit_code == 1
    assert fake_client.calls == []


def test_cli_set_shared_with_service_name_stays_mutually_exclusive(fake_client):
    # A real service name (no '=') alongside --shared is still an error.
    result = _runner.invoke(secrets_app, ["set", "--shared", "demo", "A=1"])
    assert result.exit_code == 1
    assert fake_client.calls == []


def test_cli_set_without_pairs_errors(fake_client):
    result = _runner.invoke(secrets_app, ["set", "demo"])
    assert result.exit_code == 1
    assert fake_client.calls == []


@pytest.mark.asyncio
async def test_set_async_mints_idempotency_key(fake_client):
    await _set_async("demo", {"A": "1"})
    assert len(fake_client.calls) == 1
    idem = fake_client.calls[0][3]
    assert isinstance(idem, str) and idem  # non-empty uuid4 hex


@pytest.mark.asyncio
async def test_rm_async_mints_idempotency_key(fake_client):
    await _rm_async("demo", None)
    idem = fake_client.calls[0][2]
    assert isinstance(idem, str) and idem


@pytest.mark.asyncio
async def test_rotate_key_calls_client_and_mints_idempotency_key(fake_client, capsys):
    await _rotate_key_async()
    assert len(fake_client.calls) == 1
    assert fake_client.calls[0][0] == "rotate_secrets_key"
    idem = fake_client.calls[0][1]
    assert isinstance(idem, str) and idem
    out = capsys.readouterr().out
    assert "3" in out


def test_rotate_key_command_aborts_without_confirm(fake_client, monkeypatch):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit):
        secrets_rotate_key()
    assert fake_client.calls == []


def test_rotate_key_command_proceeds_on_confirm(fake_client, monkeypatch, capsys):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    secrets_rotate_key()
    assert fake_client.calls[0][0] == "rotate_secrets_key"


@pytest.mark.asyncio
async def test_client_rotate_secrets_key_posts_and_parses_response():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("Idempotency-Key")
        return httpx.Response(200, json={"services_rewritten": 5})

    client = _make_client(handler)
    result = await client.rotate_secrets_key(idempotency_key="abc123")
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/secrets/rotate-key"
    assert seen["idem"] == "abc123"
    assert result == {"services_rewritten": 5}


@pytest.mark.asyncio
async def test_secrets_error_renders_structured_message(monkeypatch, capsys):
    import nerdit.cli.client as client_mod

    class _ErrClient:
        async def set_secrets(self, service, values, *, idempotency_key=None):
            request = httpx.Request("POST", f"http://localhost:9321/api/secrets/{service}")
            response = httpx.Response(
                422,
                request=request,
                json={
                    "code": "secret.invalid_service",
                    "message": "Invalid service name",
                    "detail": "Invalid service name 'Bad_Name'.",
                },
            )
            raise httpx.HTTPStatusError("422", request=request, response=response)

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _ErrClient())

    with pytest.raises(typer.Exit):
        await _set_async("Bad_Name", {"A": "1"})

    out = capsys.readouterr().out
    assert "Invalid request" in out
    assert "Invalid service name" in out
