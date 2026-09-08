"""nerdit logs — Stream service logs."""

from __future__ import annotations

import asyncio

import typer

from nerdit.cli.display import _plain, console, display_logs, render_client_error

# Terminal statuses for ``--follow``. A service settles at
# ``stopped/failed/completed/cancelled`` (a clean ``on-failure``/``no`` exit
# reaches ``completed``); ``restarting/degraded/building`` are transient
# desired-state churn the follower must tail through, not stop on.
_SERVICE_TERMINAL = ("stopped", "failed", "completed", "cancelled")


def logs(
    target: str = typer.Argument(..., help="Service id or name"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow logs continuously"),
    grep: str | None = typer.Option(
        None,
        "--grep",
        "-g",
        help="Keep only lines containing this text (a literal substring, not a regex)",
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Only lines at or after this ISO-8601 UTC timestamp (e.g. 2026-08-07T10:00:00Z)",
    ),
) -> None:
    """Display logs for a service (resolved by id or name).

    `--grep` and `--since` filter **server-side**, so a search over a large
    log never pulls the non-matching remainder across the wire. `--grep` is a
    plain substring: `--grep '100%'` finds the literal `100%`.
    """
    asyncio.run(_logs_async(target, follow, grep=grep, since=since))


async def _logs_async(
    target: str, follow: bool, *, grep: str | None = None, since: str | None = None
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()

    try:
        # ``GET /api/services/{ident}[/logs]`` resolves by id *or* name, so the
        # raw identifier goes straight through — no client-side resolve step.
        # The filters go through as-is: the client already omits a falsy one
        # from the query string, so an unfiltered read stays byte-identical to
        # the pre-P24 request without a second omission rule here.
        if not follow:
            entries = await client.get_service_logs(target, since_id=0, grep=grep, since=since)
            if entries:
                display_logs(entries)
            else:
                console.print("[dim]No logs available[/dim]")
            return

        # Services follow via polling. The daemon also exposes an SSE stream
        # (``GET /services/{ident}/logs/stream``) for long-lived agent/dashboard
        # consumers; the CLI stays on the poll it has always used — same frames,
        # one fewer connection mode to get wrong on a flaky link, and the filter
        # params reach the server identically either way.
        await _follow_polling(client, target, grep=grep, since=since)

    except Exception as exc:
        render_client_error(exc)
        raise typer.Exit(1)


async def _follow_polling(
    client, ident: str, *, grep: str | None = None, since: str | None = None
) -> None:
    """Poll logs and status until the service becomes terminal.

    Advance to the larger of the displayed ID and scan watermark, including
    empty filtered pages, to avoid rescanning already-decided log ranges.
    """
    since_id = 0
    while True:
        entries, watermark = await client.get_service_logs_page(
            ident, since_id=since_id, grep=grep, since=since
        )
        if entries:
            display_logs(entries)
            since_id = entries[-1]["id"]
        if watermark is not None:
            since_id = max(since_id, watermark)

        service = await client.get_service(ident)
        if service["status"] in _SERVICE_TERMINAL:
            entries = await client.get_service_logs(
                ident, since_id=since_id, grep=grep, since=since
            )
            if entries:
                display_logs(entries)
            # ``_plain``: the status is server-derived, and every server-derived
            # string is escaped at every Rich sink (the four-times-shipped bug).
            console.print(f"\n[bold]Service finished: {_plain(service['status'])}[/bold]")
            break

        await asyncio.sleep(1.0)
