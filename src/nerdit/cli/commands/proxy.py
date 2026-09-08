"""Render proxy status, including server-computed relative respawn times.

Disabled proxies render a loopback-only notice. Unreachable daemons exit 1.
"""

from __future__ import annotations

import asyncio

import typer
from rich.markup import escape
from rich.table import Table

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain

proxy_app = typer.Typer(
    name="proxy",
    help="Inspect the embedded reverse proxy.",
    no_args_is_help=True,
)

# The projection blocks of ProxyStatusResponse, in render order.
_BLOCKS = ("tls", "ca", "apex", "respawn", "routes", "mdns")


def _join(value: object) -> str:
    """Render a list as a comma-joined escaped string; empty ⇒ a dim `none`."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(item) for item in value) if value else "[dim]none[/dim]"
    return _plain(value)


@proxy_app.command("status")
def status() -> None:
    """Show the embedded proxy's state, TLS, apex, respawn and route counts.

    Exit code 1 when the daemon is unreachable, else 0.
    """
    code = asyncio.run(_proxy_status_async())
    if code:
        raise typer.Exit(code)


async def _proxy_status_async() -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.get_proxy_status()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    if not data.get("enabled"):
        console.print("Proxy off. Services stay on loopback.")
        return 0

    table = Table(show_header=True, header_style="bold", title="proxy")
    table.add_column("key")
    table.add_column("value")
    for key in (
        "state",
        "enabled",
        "available",
        "mode",
        "hostname",
        "base_domain",
        "scheme",
        "https_port",
    ):
        table.add_row(key, _plain(data.get(key)))
    console.print(table)

    details = Table(show_header=True, header_style="bold", title="details")
    details.add_column("key")
    details.add_column("value")
    for name in _BLOCKS:
        block = data.get(name)
        if not isinstance(block, dict):
            details.add_row(name, _plain(block))
            continue
        for key, value in block.items():
            details.add_row(f"{name}.{escape(str(key))}", _join(value))
    console.print(details)
    return 0
