"""Unit tests for the P40c variables CLI: client methods and command dispatch."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.vars import vars_app


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- client methods: verb / path / body ----


@pytest.mark.asyncio
async def test_set_variables_puts_values_flag_and_service():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["body"] = json.loads(request.content)
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"project": "asso", "scope": "p", "keys": [], "plain": 1})

    client = _make_client(handler)
    await client.set_variables(
        "asso", {"LOG": "debug"}, secret=False, service="web", idempotency_key="k1"
    )
    assert (seen["method"], seen["path"]) == ("PUT", "/api/projects/asso/variables")
    assert seen["params"] == {"service": "web"}
    assert seen["body"] == {"values": {"LOG": "debug"}, "secret": False}
    assert seen["idem"] == "k1"


@pytest.mark.asyncio
async def test_set_variables_defaults_to_secret_and_project_scope():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    await _make_client(handler).set_variables("asso", {"K": "v"})
    assert seen["params"] == {}  # no `service`, and never an `environment` (D-P40-11)
    assert seen["body"]["secret"] is True  # D-P40-16


@pytest.mark.asyncio
async def test_read_and_delete_variable_paths():
    seen: list[tuple] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={})

    client = _make_client(handler)
    await client.list_variables("asso")
    await client.resolve_variables("asso", service="api")
    await client.delete_variable("asso", "LOG", service="web")
    assert seen == [
        ("GET", "/api/projects/asso/variables", {}),
        ("GET", "/api/projects/asso/variables/resolve", {"service": "api"}),
        ("DELETE", "/api/projects/asso/variables/LOG", {"service": "web"}),
    ]


# ---- command dispatch (fake client, CliRunner: real argv binding) ----


class _FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []

    async def set_variables(self, project, values, *, secret=True, service=None, **_):
        self.calls.append(("set_variables", project, service, dict(values), secret))
        scope = "project" if service is None else f"production/{service}"
        return {"project": project, "scope": scope, "keys": sorted(values), "plain": not secret}

    async def set_secrets(self, service, values, *, idempotency_key=None):
        self.calls.append(("set_secrets", service, dict(values)))
        return {"service": service, "keys": sorted(values)}

    async def delete_variable(self, project, key, *, service=None, idempotency_key=None):
        self.calls.append(("delete_variable", project, service, key))
        return {"project": project, "scope": "project", "deleted": key}

    async def list_variables(self, project, *, service=None):
        self.calls.append(("list_variables", project, service))
        return {
            "project": project,
            "scope": "project",
            "variables": [
                {"key": "LOG_LEVEL", "scope": "project", "plain": True, "value": "debug"},
                {"key": "API_KEY", "scope": "project", "plain": False, "value": None},
            ],
        }

    async def resolve_variables(self, project, *, service=None):
        self.calls.append(("resolve_variables", project, service))
        return {
            "project": project,
            "service": service or "web",
            "variables": [{"key": "API_KEY", "scope": "production/web", "plain": False}],
        }


@pytest.fixture()
def fake_client(monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient()
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


_runner = CliRunner()


def test_set_pairs_is_a_plain_project_scope_write(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso", "LOG_LEVEL=debug", "REGION=eu"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [
        ("set_variables", "asso", None, {"LOG_LEVEL": "debug", "REGION": "eu"}, False)
    ]
    assert "plain" in result.output


def test_set_service_target_splits_on_slash(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso/web", "LOG_LEVEL=debug"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][:3] == ("set_variables", "asso", "web")


@pytest.mark.parametrize("target", ["/web", "asso/", ""])
def test_malformed_target_is_refused(fake_client, target):
    result = _runner.invoke(vars_app, ["set", target, "A=1"])
    assert result.exit_code != 0
    assert fake_client.calls == []


def test_secret_with_key_val_is_refused_and_value_not_echoed(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso", "--secret", "API_KEY=sk-live-123"])
    assert result.exit_code == 1
    assert fake_client.calls == []  # refused before anything is sent
    assert "--prompt KEY" in result.output
    assert "sk-live-123" not in result.output


def test_prompt_with_key_val_is_refused_without_prompting(fake_client):
    # --prompt implies --secret, so a pair beside it is the same refusal.
    result = _runner.invoke(vars_app, ["set", "asso", "A=1", "--prompt", "API_KEY"])
    assert result.exit_code == 1
    assert fake_client.calls == []
    assert "Value for" not in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["set", "--secret", "API_KEY=sk-live-123", "--prompt", "OTHER"],
        ["set", "--prompt", "OTHER", "API_KEY=sk-live-123"],
        ["set", "--secret", "API_KEY=sk-live-123"],
        ["set", "API_KEY=sk-live-123", "B=2"],
    ],
)
def test_a_forgotten_project_never_sends_the_pair_as_the_project(fake_client, argv):
    # Click binds the first KEY=VAL to <project>; it must never reach the URL
    # path / audit row, be echoed, or trigger the hidden prompt.
    result = _runner.invoke(vars_app, argv, input="hidden\n")
    assert result.exit_code == 1
    assert fake_client.calls == []
    assert "sk-live-123" not in result.output
    assert "Value for" not in result.output


def test_unset_refuses_a_key_val_without_echoing_it(fake_client):
    result = _runner.invoke(vars_app, ["unset", "asso", "API_KEY=sk-live-123"])
    assert result.exit_code == 1
    assert fake_client.calls == []
    assert "sk-live-123" not in result.output


def test_secret_prompt_reads_hidden_value_off_argv(fake_client):
    argv = ["set", "asso/web", "--secret", "--prompt", "API_KEY"]
    result = _runner.invoke(vars_app, argv, input="hunter2-value\n")
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [
        ("set_variables", "asso", "web", {"API_KEY": "hunter2-value"}, True)
    ]
    assert not any("hunter2-value" in arg for arg in argv)
    assert "hunter2-value" not in result.output  # hidden input, names-only result
    assert "API_KEY" in result.output


def test_prompt_alone_writes_secret(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso", "--prompt", "API_KEY"], input="v\n")
    assert result.exit_code == 0, result.output
    assert fake_client.calls[0][4] is True


def test_invalid_prompt_key_does_not_prompt_or_write(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso", "--secret", "--prompt", "BAD=KEY"])
    assert result.exit_code == 1
    assert fake_client.calls == []
    assert "Value for" not in result.output


def test_set_without_values_is_refused(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso"])
    assert result.exit_code == 1
    assert fake_client.calls == []


def test_machine_is_the_existing_shared_secrets_route(fake_client):
    result = _runner.invoke(
        vars_app, ["set", "--machine", "--prompt", "GH_TOKEN"], input="s3cr3t\n"
    )
    assert result.exit_code == 0, result.output
    # Never a project route: `_shared` is only reachable through /secrets/shared.
    assert fake_client.calls == [("set_secrets", "shared", {"GH_TOKEN": "s3cr3t"})]
    assert "s3cr3t" not in result.output


def test_machine_pair_without_target_is_rebound(fake_client):
    result = _runner.invoke(vars_app, ["set", "--machine", "REGION=eu"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [("set_secrets", "shared", {"REGION": "eu"})]


def test_machine_with_project_is_refused(fake_client):
    result = _runner.invoke(vars_app, ["set", "asso", "--machine", "REGION=eu"])
    assert result.exit_code == 1
    assert fake_client.calls == []


def test_unset_targets_scope_and_key(fake_client):
    result = _runner.invoke(vars_app, ["unset", "asso/web", "LOG_LEVEL"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [("delete_variable", "asso", "web", "LOG_LEVEL")]


def test_list_shows_plain_value_and_placeholder_for_secret(fake_client):
    result = _runner.invoke(vars_app, ["list", "asso"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [("list_variables", "asso", None)]
    assert "debug" in result.output
    assert "API_KEY" in result.output
    assert "None" not in result.output  # the withheld value renders as a placeholder


def test_list_json_is_the_raw_body(fake_client):
    result = _runner.invoke(vars_app, ["list", "asso", "--json"])
    assert result.exit_code == 0, result.output
    rows = {v["key"]: v for v in json.loads(result.output)["variables"]}
    assert rows["LOG_LEVEL"]["value"] == "debug"
    assert rows["API_KEY"]["value"] is None


def test_resolve_shows_winning_scope_and_no_value_column(fake_client):
    result = _runner.invoke(vars_app, ["resolve", "asso"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [("resolve_variables", "asso", None)]
    assert "production/web" in result.output
    assert "Value" not in result.output


def test_resolve_json(fake_client):
    result = _runner.invoke(vars_app, ["resolve", "asso/api", "--json"])
    assert result.exit_code == 0, result.output
    assert fake_client.calls == [("resolve_variables", "asso", "api")]
    assert json.loads(result.output)["variables"][0]["scope"] == "production/web"


def test_no_env_option_exists():
    # D-P40-11: there is deliberately no `--env` on any vars verb.
    for verb in ("set", "unset", "list", "resolve"):
        assert "--env" not in _runner.invoke(vars_app, [verb, "--help"]).output


def test_vars_group_is_registered_on_the_root_app():
    from nerdit.cli.app import app

    assert _runner.invoke(app, ["vars", "--help"]).exit_code == 0
