"""CLI ``nerdit config`` tests (P1 / S8).

Covers the value coercion of ``key=value`` arguments and the get/set commands
driven against a mocked :class:`NerditClient` (no live daemon).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.cli.commands.config import _coerce, _parse_pairs

runner = CliRunner()


# --- value parsing ------------------------------------------------------------


def test_coerce_scalars():
    assert _coerce("9") == 9
    assert _coerce("2.5") == 2.5
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("none") is None
    assert _coerce("hello") == "hello"


def test_parse_pairs_ok():
    assert _parse_pairs(["a=1", "b=x"]) == {"a": 1, "b": "x"}


def test_parse_pairs_rejects_bad():
    with pytest.raises(Exception):
        _parse_pairs(["noequals"])


# --- get / set commands -------------------------------------------------------


def _mock_client(**methods) -> AsyncMock:
    client = AsyncMock()
    for name, value in methods.items():
        setattr(client, name, AsyncMock(return_value=value))
    return client


def test_config_get_prints_values():
    client = _mock_client(
        get_config={"section": "monitor", "values": {"interval_seconds": 5}, "etag": "e1"}
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "get", "monitor"])
    assert result.exit_code == 0
    assert "monitor" in result.output
    assert "interval_seconds" in result.output


def test_config_set_sends_idempotency_and_if_match():
    client = _mock_client(
        get_config={"section": "monitor", "values": {}, "etag": "etag-123"},
        put_config={
            "applied": True,
            "diff": [{"key": "monitor.interval_seconds", "old": 5, "new": 9}],
            "requires_restart": False,
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "set", "monitor", "interval_seconds=9"])
    assert result.exit_code == 0
    # A real write generates an Idempotency-Key and carries the read ETag.
    kwargs = client.put_config.await_args.kwargs
    assert kwargs["dry_run"] is False
    assert kwargs["idempotency_key"]
    assert kwargs["if_match"] == "etag-123"
    args = client.put_config.await_args.args
    assert args[0] == "monitor"
    assert args[1] == {"interval_seconds": 9}


def test_config_set_dry_run_has_no_idempotency_key():
    client = _mock_client(
        get_config={"section": "monitor", "values": {}, "etag": "e1"},
        put_config={"applied": False, "diff": [], "requires_restart": False},
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "set", "monitor", "interval_seconds=9", "--dry-run"])
    assert result.exit_code == 0
    kwargs = client.put_config.await_args.kwargs
    assert kwargs["dry_run"] is True
    assert kwargs["idempotency_key"] is None


# --- apply (P7) ----------------------------------------------------------------


def test_config_apply_parses_toml_and_sends_if_match(tmp_path):
    toml_file = tmp_path / "cfg.toml"
    toml_file.write_text("[monitor]\ninterval_seconds = 9\n")
    client = _mock_client(
        get_config=[{"section": "monitor", "values": {}, "etag": "etag-abc"}],
        apply_config={
            "applied": True,
            "changed": True,
            "etag": "etag-new",
            "requires_restart": False,
            "restart_keys": [],
            "diff": [
                {
                    "key": "interval_seconds",
                    "old": 5,
                    "new": 9,
                    "section": "monitor",
                    "op": "change",
                }
            ],
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "apply", str(toml_file)])
    assert result.exit_code == 0
    kwargs = client.apply_config.await_args.kwargs
    assert kwargs["if_match"] == "etag-abc"
    assert kwargs["idempotency_key"]
    assert kwargs["dry_run"] is False
    args = client.apply_config.await_args.args
    assert args[0] == {"monitor": {"interval_seconds": 9}}
    assert "monitor" in result.output


def test_config_apply_dry_run_has_no_idempotency_key(tmp_path):
    toml_file = tmp_path / "cfg.toml"
    toml_file.write_text("[monitor]\ninterval_seconds = 9\n")
    client = _mock_client(
        get_config=[{"section": "monitor", "values": {}, "etag": "etag-abc"}],
        apply_config={
            "applied": False,
            "changed": True,
            "etag": "etag-abc",
            "requires_restart": False,
            "restart_keys": [],
            "diff": [],
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "apply", str(toml_file), "--dry-run"])
    assert result.exit_code == 0
    kwargs = client.apply_config.await_args.kwargs
    assert kwargs["dry_run"] is True
    assert kwargs["idempotency_key"] is None


def test_config_apply_prints_restart_footer(tmp_path):
    toml_file = tmp_path / "cfg.toml"
    toml_file.write_text("[daemon]\nport = 9322\n")
    client = _mock_client(
        get_config=[{"section": "daemon", "values": {}, "etag": "e1"}],
        apply_config={
            "applied": True,
            "changed": True,
            "etag": "e2",
            "requires_restart": True,
            "restart_keys": ["daemon.port"],
            "diff": [
                {"key": "port", "old": 9321, "new": 9322, "section": "daemon", "op": "change"}
            ],
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "apply", str(toml_file)])
    assert result.exit_code == 0
    assert "daemon.port" in result.output


def test_config_apply_bad_toml_exits_nonzero(tmp_path):
    toml_file = tmp_path / "bad.toml"
    toml_file.write_text("not = valid = toml")
    result = runner.invoke(app, ["config", "apply", str(toml_file)])
    assert result.exit_code == 1


# --- app get / set (P7) ---------------------------------------------------------


def test_app_config_get_prints_view():
    client = _mock_client(
        get_app_config={
            "service_name": "demo",
            "deploy": {"name": "demo", "port": 8000, "gpus": 0, "start": None, "health": None},
            "ai": {"default": {"provider": "ollama", "model": "llama3.1:8b"}},
            "env_keys": ["API_KEY"],
            "source": "deploy",
            "revision": 1,
            "etag": "e1",
        }
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "app", "get", "demo"])
    assert result.exit_code == 0
    assert "demo" in result.output
    assert "source=deploy" in result.output
    assert "llama3.1:8b" in result.output
    assert "API_KEY" in result.output


def test_app_config_set_deploy_section():
    client = _mock_client(
        get_app_config={"service_name": "demo", "etag": "etag-1"},
        put_app_config={
            "applied": True,
            "requires_restart": False,
            "restarted": False,
            "view": {"service_name": "demo"},
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(app, ["config", "app", "set", "demo", "deploy", "gpus=1"])
    assert result.exit_code == 0
    kwargs = client.put_app_config.await_args.kwargs
    assert kwargs["dry_run"] is False
    assert kwargs["idempotency_key"]
    assert kwargs["if_match"] == "etag-1"
    args = client.put_app_config.await_args.args
    assert args[0] == "demo"
    assert args[1] == "deploy"
    assert args[2] == {"gpus": 1}
    assert "Saved" in result.output


def test_app_config_set_ai_section_dotted_keys():
    client = _mock_client(
        get_app_config={"service_name": "demo", "etag": "etag-1"},
        put_app_config={
            "applied": True,
            "requires_restart": True,
            "restarted": False,
            "view": {"service_name": "demo"},
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(
            app,
            ["config", "app", "set", "demo", "ai", "default.model=llama3.1:8b"],
        )
    assert result.exit_code == 0
    args = client.put_app_config.await_args.args
    assert args[2] == {"default": {"model": "llama3.1:8b"}}
    assert "--restart" in result.output


def test_app_config_set_restart_flag_and_dry_run():
    client = _mock_client(
        get_app_config={"service_name": "demo", "etag": "etag-1"},
        put_app_config={
            "applied": False,
            "requires_restart": False,
            "restarted": False,
            "view": {"service_name": "demo"},
        },
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=client):
        result = runner.invoke(
            app,
            ["config", "app", "set", "demo", "deploy", "gpus=1", "--dry-run", "--restart"],
        )
    assert result.exit_code == 0
    kwargs = client.put_app_config.await_args.kwargs
    assert kwargs["dry_run"] is True
    assert kwargs["restart"] is True
    assert kwargs["idempotency_key"] is None
    assert "No write performed" in result.output
