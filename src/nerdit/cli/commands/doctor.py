"""Render daemon health checks, falling back to local checks when unreachable.

Exit 1 for an unreachable daemon or a failing check; otherwise exit 0.
"""

from __future__ import annotations

import asyncio

import typer
from rich.markup import escape
from rich.table import Table

from nerdit.cli.display import console
from nerdit.cli.display import plain as _plain

_STATUS_STYLE = {
    "ok": "green",
    "warn": "yellow",
    "fail": "red",
    "skipped": "dim",
    "skip": "dim",  # the local registry (nerdit.cli.checks) uses 'skip'
}


def doctor() -> None:
    """Diagnose the daemon: run its structured health checks and print them.

    Exit code 1 when the daemon is unreachable or the worst check is `fail`,
    else 0 (a scripting convenience — not a locked contract).
    """
    code = asyncio.run(_doctor_async())
    if code:
        raise typer.Exit(code)


def _status_cell(status: str) -> str:
    style = _STATUS_STYLE.get(status, "white")
    return f"[{style}]{_plain(status)}[/{style}]"


async def _doctor_async() -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.get_doctor()
    except Exception as exc:  # noqa: BLE001 — a down daemon is the expected failure here
        # The daemon is the thing being diagnosed: render a single local fail
        # row instead of a traceback, then fall back to the shared local check
        # registry so the verb stays useful precisely when the daemon is down.
        table = _make_table()
        table.add_row("daemon", _status_cell("fail"), _plain(f"unreachable: {exc}"), "-")
        console.print(table)
        console.print("status: [red]fail[/red]")
        _render_local_registry(getattr(client, "host", None))
        return 1

    top = data.get("status", "?")
    table = _make_table()
    for check in data.get("checks", []):
        table.add_row(
            _plain(check.get("name")),
            _status_cell(check.get("status", "?")),
            _plain(check.get("detail")),
            _plain(check.get("latency_ms")),
        )
    console.print(table)
    top_style = _STATUS_STYLE.get(top, "white")
    console.print(f"status: [{top_style}]{_plain(top)}[/{top_style}]")
    return 1 if top == "fail" else 0


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1"})


def _render_local_registry(host: str | None = None) -> None:
    """Render local checks when the daemon cannot be reached, without tracebacks.

    For remote connections, label these as checks of the CLI host.
    """
    from nerdit.cli import checks

    try:
        results = checks.run_checks()
    except Exception as exc:  # noqa: BLE001 — the fallback must never itself crash
        console.print(_plain(f"local checks unavailable: {exc}"))
        return

    console.print()
    if host is not None and host not in _LOOPBACK_HOSTS:
        console.print(
            f"[yellow]The CLI is configured for a remote daemon ({escape(host)}) — "
            "the checks below describe THIS machine, not the daemon host.[/yellow]"
        )
    console.print("[bold]local checks (daemon unreachable)[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    table.add_column("remediation")
    for r in results:
        table.add_row(
            _plain(r.name),
            _status_cell(r.status),
            _plain(r.detail),
            _plain(r.remediation),
        )
    console.print(table)


def _make_table() -> Table:
    table = Table(show_header=True, header_style="bold")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    table.add_column("ms", justify="right")
    return table
