"""Publish apps through the existing cloud tunnel.

The daemon validates all eligibility and consent. Stored share intent survives
outages and unlinking; render link readiness separately from origin health,
using machine tokens and server hints. Escape all server-derived Rich text.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import typer

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain

#: Machine ``state`` token → the one-line explanation of why a stored share is
#: not answering yet. Keyed on the token from the daemon so an older/newer
#: vocabulary degrades to "no extra line", never to a wrong sentence.
_STATE_NOTES = {
    "link_down": (
        "the tunnel is down — the URL answers once the link reconnects (check: nerdit link)"
    ),
    "not_entitled": (
        "public sharing is free during the public beta; the share is stored, "
        "but cloud access is not confirmed — check nerdit link and the Nerdit console"
    ),
}


def _render_share(name: str, result: Any, *, verb: str) -> None:
    """Print one share as `<verb> <name> (<access>): <url>` + any state note."""
    result = result if isinstance(result, dict) else {}
    access = result.get("access")
    url = result.get("url")
    console.print(
        f"[green]{verb} [bold]{_plain(name)}[/bold] ({_plain(access)}):[/green] {_plain(url)}"
    )
    state = result.get("state")
    note = _STATE_NOTES.get(state) if isinstance(state, str) else None
    if note is not None:
        console.print(f"[yellow]{_plain(state)} — {note}[/yellow]")
    # (P34) The share can be provisioned and the app still be down, in which
    # case the URL 404s with share.not_shared and "Shared: <url>" alone reads as
    # a lie. Printed from the daemon's own ``origin.hint`` — the CLI keeps no
    # second copy of the wording, exactly as it keeps none of the refusal hints.
    origin = result.get("origin")
    if isinstance(origin, dict) and origin.get("answers") is False:
        hint = origin.get("hint")
        if isinstance(hint, str) and hint:
            console.print(f"[yellow]origin — {_plain(hint)}[/yellow]")


def share(
    name: str = typer.Argument(..., help="Name of the deployed app to share."),
    public: bool = typer.Option(
        False,
        "--public",
        help=(
            "Make the URL world-reachable instead of owner-only. Needs an "
            "active linked account AND either --consent or a \\[deploy].edge_auth "
            "block on the app."
        ),
    ),
    consent: bool = typer.Option(
        False,
        "--consent",
        help="Acknowledge that --public means anyone with the link can open the app.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Print the current share instead of changing it.",
    ),
) -> None:
    """Share a deployed app at a hosted URL through this node's cloud link."""
    asyncio.run(_share_async(name, public=public, consent=consent, show=show))


async def _share_async(name: str, *, public: bool, consent: bool, show: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()

    if show:
        try:
            result = await client.get_share(name)
        except Exception as exc:  # noqa: BLE001 — rendered for the user
            render_client_error(exc)
            raise typer.Exit(1) from exc
        _render_share(name, result, verb="Shared")
        return

    try:
        result = await client.set_share(
            name,
            access="public" if public else "private",
            consent=consent,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # The envelope carries the hint that tells the operator what to do next
        # ("pass --consent", "run nerdit link refresh", "check account status") —
        # there is deliberately no local guess layered on top of it.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    _render_share(name, result, verb="Shared")


def unshare(
    name: str = typer.Argument(..., help="Name of the app to stop sharing."),
) -> None:
    """Remove an app's hosted share; the URL stops answering on the next request."""
    asyncio.run(_unshare_async(name))


async def _unshare_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.remove_share(name, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    result = result if isinstance(result, dict) else {}
    if result.get("removed"):
        console.print(f"[green]Unshared [bold]{_plain(name)}[/bold].[/green]")
    else:
        # Idempotent by design — say so plainly rather than reporting success
        # for a share that never existed.
        console.print(f"[dim]{_plain(name)} was not shared — nothing to do.[/dim]")
