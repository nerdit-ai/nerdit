"""nerdit logs — Stream service logs."""

from __future__ import annotations

import asyncio

import typer

from nerdit.cli.display import _plain, console, display_logs, render_client_error

# The daemon's own forward-page cap. Read here only to recognise a FULL page —
# "there is more backlog" — so the follower can drain it without waiting a poll
# per page. A remote daemon with a different cap degrades to the old cadence,
# never to a wrong result.
from nerdit.daemon.limits import _MAX_LOG_TAIL

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
            if await _drain(client, target, grep=grep, since=since) == 0:
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


async def _drain(
    client, ident: str, *, since_id: int = 0, grep: str | None = None, since: str | None = None
) -> int:
    """Display every log page from *since_id* onward; return the new cursor.

    A forward page is server-capped, so one request is not the whole backlog.
    The loop terminates because every non-empty page strictly advances the
    cursor (ids ascend and the query is `id > since_id`), and an empty page
    means no matching row past the cursor at all — the cap is applied after the
    filters, inside SQL.
    """
    while True:
        entries = await client.get_service_logs(ident, since_id=since_id, grep=grep, since=since)
        if not entries:
            return since_id
        display_logs(entries)
        since_id = entries[-1]["id"]


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
            if len(entries) >= _MAX_LOG_TAIL:
                # The page filled the server cap, so more backlog is already
                # waiting: draining it at one page per poll would put a chatty
                # service's retained history minutes ahead of its first LIVE
                # line. Loop straight back without the status poll or the
                # sleep — ids ascend, so this terminates at the first short
                # page and the steady-state cadence below is unchanged. A
                # daemon with a smaller cap simply never takes this branch.
                continue
        if watermark is not None:
            since_id = max(since_id, watermark)

        service = await client.get_service(ident)
        if service["status"] in _SERVICE_TERMINAL:
            # Drain, not one last page: the dying lines of a chatty service can
            # exceed one server-capped page, and this is the caller's last read.
            await _drain(client, ident, since_id=since_id, grep=grep, since=since)
            # ``_plain``: the status is server-derived, and every server-derived
            # string is escaped at every Rich sink (the four-times-shipped bug).
            console.print(f"\n[bold]Service finished: {_plain(service['status'])}[/bold]")
            break

        await asyncio.sleep(1.0)
