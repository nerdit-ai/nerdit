"""Render daemon capabilities as tables or JSON.

Non-admin responses omit paths and proxy.admin_addr; omit their rows rather
than imply missing values. Escape server-derived Rich text. Unreachable
daemons produce exit code 1.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.markup import escape
from rich.table import Table

# Single source of truth for the byte formatter (shared with `nerdit disk`).
from nerdit.cli.display import console, fmt_bytes, render_client_error
from nerdit.cli.display import plain as _plain


def _join(value: object) -> str:
    """Render a list as a comma-joined escaped string; empty ⇒ a dim `none`."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(item) for item in value) if value else "[dim]none[/dim]"
    return _plain(value)


def _block(data: dict, key: str) -> dict:
    """Return `data[key]` when it is a mapping, else `{}` (shape-tolerant)."""
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def capabilities(
    json_out: bool = typer.Option(False, "--json", help="Print the raw JSON body."),
) -> None:
    """Show what this daemon can do, and what the current token may do.

    Exit code 1 when the daemon is unreachable, else 0.
    """
    code = asyncio.run(_capabilities_async(json_out=json_out))
    if code:
        raise typer.Exit(code)


def _kv_table(title: str) -> Table:
    table = Table(show_header=True, header_style="bold", title=title)
    table.add_column("key")
    table.add_column("value")
    return table


async def _capabilities_async(*, json_out: bool) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.get_capabilities()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    if json_out:
        # The scripting escape hatch: the raw body, stable key order. Emitted
        # through typer.echo so Rich never reflows or styles machine output.
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
        return 0

    _render_facts(data)
    _render_proxy(data)
    _render_capabilities(data)
    return 0


def _render_facts(data: dict) -> None:
    caller = _block(data, "caller")
    quotas = _block(caller, "quotas")
    table = _kv_table("daemon")
    table.add_row("version", _plain(data.get("version")))
    table.add_row("uptime_s", _plain(data.get("uptime_s")))
    table.add_row("caller.role", _plain(caller.get("role")))
    table.add_row("caller.token_name", _plain(caller.get("token_name")))
    table.add_row("caller.quotas.max_gpus", _plain(quotas.get("max_gpus")))
    table.add_row("caller.quotas.max_concurrent_jobs", _plain(quotas.get("max_concurrent_jobs")))
    console.print(table)


def _render_proxy(data: dict) -> None:
    proxy = _block(data, "proxy")
    table = _kv_table("proxy")
    for key in (
        "enabled",
        "available",
        "mode",
        "hostname",
        "scheme",
        "https_port",
        "url_shape",
        "dashboard_apex",
        "mdns",
    ):
        table.add_row(key, _plain(proxy.get(key)))
    # Admin-only projection: present ⇒ render it, absent ⇒ render NOTHING (a
    # `-` would read as "unset" instead of "your role does not see this").
    if "admin_addr" in proxy:
        table.add_row("admin_addr", _plain(proxy.get("admin_addr")))
    console.print(table)


def _render_capabilities(data: dict) -> None:
    models = _block(data, "models")
    databases = _block(data, "databases")
    deploy = _block(data, "deploy")
    gpus = _block(data, "gpus")
    mcp = _block(data, "mcp")

    table = _kv_table("capabilities")
    table.add_row("buildpacks", _join(data.get("buildpacks")))
    table.add_row("models.backends", _join(models.get("backends")))
    table.add_row("models.default_backend", _plain(models.get("default_backend")))
    table.add_row("databases.backends", _join(databases.get("backends")))
    table.add_row("deploy.git_enabled", _plain(deploy.get("git_enabled")))
    table.add_row("deploy.git_allowed_hosts", _join(deploy.get("git_allowed_hosts")))
    table.add_row("deploy.max_upload_bytes", fmt_bytes(deploy.get("max_upload_bytes")))
    table.add_row("gpus.count", _plain(gpus.get("count")))
    table.add_row("gpus.schedulable", _plain(gpus.get("schedulable")))
    table.add_row("mcp.http_enabled", _plain(mcp.get("http_enabled")))
    # (P33) The container sandbox every deployed image has to survive — the
    # operator-facing half of the ``image_needs_privileges`` remediation.
    for key, value in _block(data, "sandbox").items():
        table.add_row(f"sandbox.{escape(str(key))}", _plain(value))
    for key, value in _block(data, "limits").items():
        table.add_row(f"limits.{escape(str(key))}", _join(value))
    for key, value in _block(data, "features").items():
        table.add_row(f"features.{escape(str(key))}", _plain(value))
    # Admin-only block, same rule as `admin_addr`: omitted, never nulled.
    for key, value in _block(data, "paths").items():
        table.add_row(f"paths.{escape(str(key))}", _plain(value))
    console.print(table)
