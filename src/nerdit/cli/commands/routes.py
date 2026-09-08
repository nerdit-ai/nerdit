"""Render DB-owned endpoint inventory, including unrouted models.

Route None means unrouted, an empty string means subdomain, and `/name` means
path routing; test identity, not truthiness. Unknown live-Caddy state renders
`-`, with a footer explaining why. Unreachable daemons exit 1.
"""

from __future__ import annotations

import asyncio

import typer
from rich.table import Table

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain


def _route_cell(route: object) -> str:
    """Render the locked route tri-state (null / "" / "/name")."""
    if route is None:
        return "[dim]unrouted[/dim]"
    if route == "":
        return "(subdomain)"
    return _plain(route)


def _live_cell(live: object) -> str:
    """Render the live-Caddy annotation; `None` ⇒ `-` (unknown, not "no")."""
    if live is None:
        return "-"
    if not isinstance(live, dict):
        return _plain(live)
    if not live.get("registered"):
        return "[yellow]missing[/yellow]"
    return "ok" if live.get("dial_matches") else "[yellow]stale dial[/yellow]"


def routes(
    limit: int = typer.Option(50, "--limit", help="Max rows per page."),
    cursor: str | None = typer.Option(None, "--cursor", help="Opaque cursor from a previous page."),
) -> None:
    """List every registered route (models included, shown as unrouted).

    Exit code 1 when the daemon is unreachable, else 0.
    """
    code = asyncio.run(_routes_async(limit=limit, cursor=cursor))
    if code:
        raise typer.Exit(code)


async def _routes_async(*, limit: int, cursor: str | None) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.list_routes(cursor=cursor, limit=limit)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    table = Table(show_header=True, header_style="bold", title="routes")
    table.add_column("service")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("route")
    table.add_column("public_url")
    table.add_column("host_port", justify="right")
    table.add_column("live")
    for item in data.get("items", []):
        table.add_row(
            _plain(item.get("service_name")),
            _plain(item.get("kind")),
            _plain(item.get("status")),
            _route_cell(item.get("route")),
            _plain(item.get("public_url")),
            _plain(item.get("host_port")),
            _live_cell(item.get("live")),
        )
    console.print(table)

    console.print(f"live table: {_plain(data.get('live_table'))}")
    next_cursor = data.get("next_cursor")
    if next_cursor is not None:
        console.print(f"[dim]next page: nerdit routes --cursor {_plain(next_cursor)}[/dim]")
    return 0
