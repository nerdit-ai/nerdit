"""Browse or follow the durable event feed.

Default reads are newest-first; `--since` reads forward and is exclusive.
Following replays before tailing. Report feed gaps with a resume cursor one
below the retained minimum ID, so the oldest retained row is included.
Escape all server values; an unreachable daemon exits 1.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.table import Table

from nerdit.cli.display import _plain, console, render_client_error

# Bound on the inline ``data`` rendering so one chatty payload cannot wrap the
# whole table off the screen. The full object is always one API call away.
_MAX_DATA_CHARS = 60


def _data_cell(data: object) -> str:
    """Render the machine-shaped `data` object as bounded `k=v` pairs."""
    if not isinstance(data, dict) or not data:
        return "-"
    rendered = " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in sorted(data.items()))
    if len(rendered) > _MAX_DATA_CHARS:
        rendered = rendered[: _MAX_DATA_CHARS - 1] + "…"
    return _plain(rendered)


def _resume_hint(retained_min_id: object) -> str:
    """Return max(0, retained_min_id - 1) for the exclusive replay cursor.

    Preserve non-integer values rather than fail while rendering an error.
    """
    if isinstance(retained_min_id, bool) or not isinstance(retained_min_id, int):
        return _plain(retained_min_id)
    return _plain(max(retained_min_id - 1, 0))


def _type_cell(event_type: object) -> str:
    """Colour the failure vocabulary; everything else renders plain."""
    text = _plain(event_type)
    if isinstance(event_type, str) and (
        event_type.endswith("_failed") or event_type.endswith(".failed")
    ):
        return f"[red]{text}[/red]"
    return text


def events(
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new events as they happen."),
    since: int | None = typer.Option(
        None,
        "--since",
        help=(
            "Replay forwards from this event id (id ASC, oldest first) — the "
            "resume mode. Without it the feed reads newest-first."
        ),
    ),
    type_filter: list[str] = typer.Option(
        [], "--type", help="Only this event type (repeatable, at most 10 honoured)."
    ),
    service: str | None = typer.Option(None, "--service", help="Only this service name."),
    limit: int = typer.Option(50, "--limit", help="Max rows (clamped server-side to 200)."),
) -> None:
    """Show what the daemon did on its own: deploys, health flaps, restarts, failures.

    Exit code 1 when the daemon is unreachable, else 0.
    """
    code = asyncio.run(
        _events_async(
            follow=follow,
            since=since,
            types=list(type_filter),
            service=service,
            limit=limit,
        )
    )
    if code:
        raise typer.Exit(code)


async def _events_async(
    *, follow: bool, since: int | None, types: list[str], service: str | None, limit: int
) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    joined = ",".join(t for t in types if t) or None
    if follow:
        return await _follow(client, since=since, types=set(types), service=service)

    try:
        data = await client.list_events(limit=limit, types=joined, service=service, since_id=since)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    items = data.get("items", [])
    table = Table(show_header=True, header_style="bold", title="events")
    table.add_column("id", justify="right")
    table.add_column("ts")
    table.add_column("type")
    table.add_column("service")
    table.add_column("reason")
    table.add_column("data")
    for item in items:
        table.add_row(
            _plain(item.get("id")),
            _plain(item.get("ts")),
            _type_cell(item.get("type")),
            _plain(item.get("service_name")),
            _plain(item.get("reason")),
            _data_cell(item.get("data")),
        )
    console.print(table)

    if not items:
        console.print("[dim]No events recorded yet.[/dim]")
    next_cursor = data.get("next_cursor")
    if next_cursor is not None:
        if since is not None:
            # Forward mode: next_cursor is the LARGEST id returned, so feeding
            # it back as --since walks strictly forward with no repeats.
            console.print(f"[dim]next page: nerdit events --since {_plain(next_cursor)}[/dim]")
        else:
            # Browse mode walks backwards into history; the useful next step
            # from the CLI is a wider page (there is deliberately no --cursor
            # flag — the forward mode is the one worth scripting).
            console.print("[dim]More events behind this page; raise --limit to see them.[/dim]")
    return 0


async def _follow(client, *, since: int | None, types: set[str], service: str | None) -> int:
    """Render the live stream until the user interrupts it.

    Filtering is client-side here on purpose: the stream carries the whole feed
    (plus legacy bus-only frames), and a server-side filter would silently hide
    the `feed.gap` frame that says the replay was incomplete.
    """
    console.print("[dim]Streaming events (Ctrl-C to stop)…[/dim]")
    try:
        async for frame in client.stream_events(last_event_id=since):
            event_type = frame.get("type")
            if not event_type:
                continue  # heartbeat
            if event_type == "feed.gap":
                retained = frame.get("retained_min_id")
                console.print(
                    f"[yellow]feed.gap[/yellow] reason={_plain(frame.get('reason'))} "
                    f"from={_plain(frame.get('from'))} "
                    f"retained_min_id={_plain(retained)} "
                    f"— replay was not exact; reconcile with "
                    f"nerdit events --since {_resume_hint(retained)}"
                )
                continue
            if event_type == "feed.saturated":
                console.print(
                    f"[yellow]feed.saturated[/yellow] the daemon is already serving its "
                    f"maximum of {_plain(frame.get('limit'))} streams; try again shortly"
                )
                return 0
            if types and event_type not in types:
                continue
            if service and frame.get("service_name") != service:
                continue
            console.print(
                f"{_plain(frame.get('id'))} {_plain(frame.get('ts'))} "
                f"{_type_cell(event_type)} {_plain(frame.get('service_name'))} "
                f"{_plain(frame.get('reason'))} {_data_cell(frame.get('data'))}"
            )
    except KeyboardInterrupt:  # pragma: no cover — interactive only
        return 0
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1
    return 0
