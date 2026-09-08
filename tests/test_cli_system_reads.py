"""CLI tests for the P23 WP1 read verbs: ``capabilities`` / ``proxy status`` / ``routes``.

``CliRunner`` + an ``AsyncMock`` client patched at
``nerdit.cli.client.get_configured_client`` (the ``tests/test_cli_token.py``
pattern — every command body imports the factory lazily inside the function).

The load-bearing assertions, per the plan:

* each verb renders its key fields and exits 0; an unreachable daemon exits 1;
* ``capabilities --json`` emits parseable JSON (and only JSON);
* the admin-only ``proxy.admin_addr`` / ``paths`` projections render **only when
  present** — an absent key prints nothing, never ``-``;
* ``proxy status`` short-circuits the disabled case to one sentence;
* ``routes`` renders all three ``route`` tri-states (``null`` / ``""`` /
  ``"/name"``), an unknown ``live`` as ``-``, the ``live_table`` footer and the
  ``next_cursor`` re-invocation hint;
* a markup-hostile server string (``"[/x]"``) never raises ``MarkupError``
  (the discipline pinned by ``tests/test_cli_display_markup.py``).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app

runner = CliRunner()

# A value that is legal server output and illegal Rich markup.
MARKUP_BOMB = "[/x]"


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Keep Rich from truncating table cells under a narrow test terminal."""
    monkeypatch.setenv("COLUMNS", "300")


CAPABILITIES_BODY = {
    "version": "0.3.0",
    "uptime_s": 42,
    "caller": {
        "role": "submitter",
        "token_name": "ci",
        "quotas": {"max_gpus": 2, "max_concurrent_jobs": 4},
    },
    "proxy": {
        "enabled": True,
        "available": True,
        "mode": "path",
        "base_domain": None,
        "hostname": "gpu-box.local",
        "scheme": "https",
        "https_port": 443,
        "public_port": None,
        "url_shape": "https://gpu-box.local/<name>/",
        "dashboard_apex": False,
        "mdns": True,
    },
    "models": {"backends": ["ollama", "vllm"], "default_backend": "ollama"},
    "databases": {"backends": ["postgres", "redis"], "default_backend": "postgres"},
    "buildpacks": ["python", "node"],
    "deploy": {
        "git_enabled": True,
        "git_allowed_hosts": ["github.com"],
        "max_upload_bytes": 1048576,
        "dry_run": True,
        "max_concurrent_builds": 2,
    },
    "gpus": {"count": 1, "schedulable": 1, "vendors": ["nvidia"]},
    "mcp": {"http_enabled": False},
    "limits": {"page_limit_max": 200},
    "features": {"app_templates": True},
}


def _client(**methods) -> AsyncMock:
    fake = AsyncMock()
    for name, value in methods.items():
        setattr(fake, name, AsyncMock(**value))
    return fake


# --- nerdit capabilities ------------------------------------------------------


def test_capabilities_renders_key_fields():
    fake = _client(get_capabilities={"return_value": CAPABILITIES_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0
    assert "0.3.0" in result.output
    assert "submitter" in result.output
    assert "gpu-box.local" in result.output
    assert "ollama" in result.output
    assert "github.com" in result.output
    # max_upload_bytes goes through _fmt_bytes (1 MiB), never the raw int.
    assert "1.0 MiB" in result.output
    assert "1048576" not in result.output


def test_capabilities_omits_admin_only_fields_when_absent():
    fake = _client(get_capabilities={"return_value": CAPABILITIES_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0
    # Absent ⇒ the row is not printed at all (a `-` would read as "unset"
    # rather than "your role does not see this").
    assert "admin_addr" not in result.output
    assert "paths." not in result.output


def test_capabilities_renders_admin_only_fields_when_present():
    body = json.loads(json.dumps(CAPABILITIES_BODY))
    body["proxy"]["admin_addr"] = "127.0.0.1:2019"
    body["paths"] = {"data_dir": "/home/op/.nerdit", "db_path": "/home/op/.nerdit/nerdit.db"}
    fake = _client(get_capabilities={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0
    assert "admin_addr" in result.output
    assert "127.0.0.1:2019" in result.output
    assert "paths.data_dir" in result.output


def test_capabilities_json_flag_emits_parseable_json():
    fake = _client(get_capabilities={"return_value": CAPABILITIES_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed == CAPABILITIES_BODY


def test_capabilities_unreachable_daemon_exits_one():
    fake = _client(get_capabilities={"side_effect": httpx.ConnectError("refused")})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 1


def test_capabilities_survives_markup_hostile_strings():
    body = json.loads(json.dumps(CAPABILITIES_BODY))
    body["proxy"]["hostname"] = MARKUP_BOMB
    body["buildpacks"] = [MARKUP_BOMB]
    body["caller"]["token_name"] = MARKUP_BOMB
    fake = _client(get_capabilities={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["capabilities"])
    assert result.exit_code == 0, result.output
    assert MARKUP_BOMB in result.output


# --- nerdit proxy status ------------------------------------------------------


PROXY_BODY = {
    "state": "available",
    "enabled": True,
    "available": True,
    "mode": "path",
    "base_domain": None,
    "hostname": "gpu-box.local",
    "scheme": "https",
    "https_port": 443,
    "tls": {"subjects": ["*.gpu-box.local"], "synced": True},
    "ca": {"fingerprint": "AA:BB:CC", "present": True},
    "apex": {"enabled": False, "present": None, "is_last": None},
    "respawn": {"attempts": 0, "last_spawn_ago_s": 12.5, "next_retry_in_s": None},
    "routes": {"count": 3, "live_table": "readable"},
    "mdns": {"enabled": True, "address": "192.168.1.10", "registered": True},
}


def test_proxy_status_renders_projection_blocks():
    fake = _client(get_proxy_status={"return_value": PROXY_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["proxy", "status"])
    assert result.exit_code == 0
    assert "available" in result.output
    assert "gpu-box.local" in result.output
    assert "tls.synced" in result.output
    assert "*.gpu-box.local" in result.output
    assert "ca.fingerprint" in result.output
    assert "AA:BB:CC" in result.output
    assert "respawn.next_retry_in_s" in result.output
    assert "routes.live_table" in result.output
    assert "mdns.registered" in result.output


def test_proxy_status_disabled_short_circuits():
    body = dict(PROXY_BODY, state="disabled", enabled=False, available=False)
    fake = _client(get_proxy_status={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["proxy", "status"])
    assert result.exit_code == 0
    assert "Proxy off. Services stay on loopback." in result.output
    # No table of nulls.
    assert "tls.synced" not in result.output


def test_proxy_status_unreachable_daemon_exits_one():
    fake = _client(get_proxy_status={"side_effect": httpx.ConnectError("refused")})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["proxy", "status"])
    assert result.exit_code == 1


def test_proxy_status_survives_markup_hostile_strings():
    body = dict(PROXY_BODY, hostname=MARKUP_BOMB, tls={"subjects": [MARKUP_BOMB], "synced": False})
    fake = _client(get_proxy_status={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["proxy", "status"])
    assert result.exit_code == 0, result.output
    assert MARKUP_BOMB in result.output


# --- nerdit routes ------------------------------------------------------------


ROUTES_BODY = {
    "items": [
        {
            "service_name": "api",
            "kind": "service",
            "status": "running",
            "host_port": 30001,
            "container_port": 8000,
            "protocol": "tcp",
            "route": "/api",
            "public_url": "https://gpu-box.local/api/",
            "live": {"registered": True, "dial_matches": True},
        },
        {
            "service_name": "web",
            "kind": "service",
            "status": "running",
            "host_port": 30002,
            "container_port": 3000,
            "protocol": "tcp",
            # Subdomain mode: an EMPTY STRING is a real route, not a missing one.
            "route": "",
            "public_url": "https://web.gpu-box.local/",
            "live": None,
        },
        {
            "service_name": "ollama-llama3",
            "kind": "model",
            "status": "running",
            "host_port": 30003,
            "container_port": 11434,
            "protocol": "tcp",
            "route": None,
            "public_url": None,
            "live": None,
        },
    ],
    "next_cursor": None,
    "live_table": "unreadable",
}


def test_routes_renders_the_three_route_states_and_footer():
    fake = _client(list_routes={"return_value": ROUTES_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["routes"])
    assert result.exit_code == 0
    assert "unrouted" in result.output  # route is None
    assert "(subdomain)" in result.output  # route == ""
    assert "/api" in result.output  # a literal path route
    assert "live table: unreadable" in result.output
    fake.list_routes.assert_awaited_once()
    assert fake.list_routes.await_args.kwargs == {"cursor": None, "limit": 50}


def test_routes_unknown_live_renders_dash_not_no():
    fake = _client(list_routes={"return_value": ROUTES_BODY})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["routes"])
    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if "ollama-llama3" in line]
    assert lines
    cells = [cell.strip() for cell in lines[0].split("│") if cell.strip()]
    # Last column is `live`: an unreadable live table is `-` (unknown), not "no".
    assert cells[-1] == "-"


def test_routes_next_cursor_prints_reinvocation_hint():
    body = dict(ROUTES_BODY, next_cursor="abc123", live_table="readable")
    fake = _client(list_routes={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["routes", "--limit", "2", "--cursor", "prev"])
    assert result.exit_code == 0
    assert "nerdit routes --cursor abc123" in result.output
    assert fake.list_routes.await_args.kwargs == {"cursor": "prev", "limit": 2}


def test_routes_unreachable_daemon_exits_one():
    fake = _client(list_routes={"side_effect": httpx.ConnectError("refused")})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["routes"])
    assert result.exit_code == 1


def test_routes_survives_markup_hostile_strings():
    body = {
        "items": [
            {
                "service_name": MARKUP_BOMB,
                "kind": "service",
                "status": "running",
                "host_port": 30001,
                "container_port": 8000,
                "protocol": "tcp",
                "route": MARKUP_BOMB,
                "public_url": None,
                "live": None,
            }
        ],
        "next_cursor": MARKUP_BOMB,
        "live_table": "readable",
    }
    fake = _client(list_routes={"return_value": body})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["routes"])
    assert result.exit_code == 0, result.output
    assert MARKUP_BOMB in result.output
