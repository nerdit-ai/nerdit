"""Tests for the token command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.config.settings import ClientSettings, DaemonSettings, NerditSettings

runner = CliRunner()


def test_token_displays_daemon_token():
    settings = NerditSettings(daemon=DaemonSettings(auth_token="test-secret-123"))
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "test-secret-123" in result.output


def test_token_displays_client_token_when_remote():
    settings = NerditSettings(
        client=ClientSettings(remote_host="10.0.0.1", auth_token="remote-token-456"),
    )
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "remote-token-456" in result.output


def test_token_no_token():
    settings = NerditSettings()
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "No authentication token configured" in result.output
