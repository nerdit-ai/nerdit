"""Manage direct domains served by this node's Caddy.

The daemon validates domain grammar, ownership, workload kind and ACME policy.
Stored domains survive stopped apps and disabled proxies. Render readiness and
certificate state separately, and escape all server-derived Rich text.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain

#: Machine ``state`` token → the one-line explanation of why a stored domain is
#: not answering yet. Keyed on the daemon's token (the ``share.py`` pattern) so
#: an older/newer vocabulary degrades to "no extra line", never a wrong sentence.
_STATE_NOTES = {
    "withheld": ("the proxy is off or the app is not routed yet; the domain answers once it is"),
}

#: Printed once after a successful add: the one thing the daemon can never do
#: for you. DNS is yours, always.
_DNS_NOTE = "Point DNS for {domain} at this machine."

#: Trust instructions apply only to internal-CA states, including disabled ACME.
#: Issued/expired public leaves and pending issuance do not use the internal root.
#: Unknown states retain the instruction; branch on cert_state, not rendered text.
_TRUST_CLAUSE = " Clients trust the internal CA via nerdit trust."
_NO_TRUST_CLAUSE_STATES = frozenset({"issued", "pending", "expired"})

#: Machine ``cert_state`` token → the one-line explanation, same contract as
#: `_STATE_NOTES`: keyed on the daemon's token, so ``internal`` (the
#: default binding, and not a defect) and ``issued`` (nothing to say) simply
#: have no entry. No square brackets in any value — these strings go through
#: ``console.print``, where ``[proxy.acme]`` would parse as Rich markup.
_CERT_NOTES = {
    "pending": (
        "the public certificate has not been issued yet; until it is, this name does not "
        "complete HTTPS handshakes at all, and nerdit trust does not help - the internal "
        "CA is not a fallback for a name that asks for a public certificate"
    ),
    "expired": (
        "the public certificate has expired and clients see the stale leaf; check the "
        "acme_http_port doctor row and the proxy log"
    ),
    "disabled": (
        "this name asks for a public certificate but ACME is off on this node; "
        "set enabled and email under proxy.acme in config.toml "
        "(nerdit config set cannot address a sub-table), then restart the daemon"
    ),
}

domains_app = typer.Typer(
    name="domains",
    help="Domains for a deployed app (served by this node's proxy).",
    no_args_is_help=True,
)


def _state_note(result: dict[str, Any]) -> None:
    """Print the yellow one-liner for a non-`ready` state, if any."""
    _note(result.get("state"), _STATE_NOTES)


def _cert_note(result: dict[str, Any]) -> None:
    """Print the yellow one-liner for a certificate state worth acting on."""
    _note(result.get("cert_state"), _CERT_NOTES)


def _note(token: object, notes: dict[str, str]) -> None:
    """Render a known state token's explanation; omit unknown explanations.

    Escape the token because it is server-derived.
    """
    note = notes.get(token) if isinstance(token, str) else None
    if note is not None:
        console.print(f"[yellow]{_plain(token)} — {note}[/yellow]")


@domains_app.command("list")
def domains_list(
    app: str = typer.Argument(..., help="Name of the deployed app."),
) -> None:
    """List an app's direct domains."""
    asyncio.run(_list_async(app))


async def _list_async(app: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.list_domains(app)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    result = result if isinstance(result, dict) else {}
    rows = result.get("domains") or []
    if not rows:
        console.print(f"[dim]{_plain(app)} has no direct domains.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", title=f"domains — {app}")
    table.add_column("domain")
    table.add_column("state")
    # The second axis beside ``state``: the route fact and the
    # certificate fact are independent, so the operator needs both columns to
    # tell "not routed yet" from "routed, but the browser will warn".
    table.add_column("cert")
    table.add_column("url")
    for row in rows:
        row = row if isinstance(row, dict) else {}
        table.add_row(
            _plain(row.get("domain")),
            _plain(row.get("state")),
            _plain(row.get("cert_state")),
            _plain(row.get("url")),
        )
    console.print(table)


@domains_app.command("add")
def domains_add(
    app: str = typer.Argument(..., help="Name of the deployed app."),
    domain: str = typer.Argument(..., help="Bare DNS name you control, e.g. app.example.com."),
    acme: bool | None = typer.Option(
        None,
        "--acme/--no-acme",
        # Escape the TOML section in Rich help so it remains visible.
        # Neither ACME flag preserves stored policy; --no-acme explicitly downgrades it.
        help=(
            "Request a public certificate via ACME HTTP-01. Needs an admin token "
            r"and \[proxy.acme].enabled on the daemon (409 domain.acme_disabled / "
            "403 domain.acme_forbidden otherwise). --no-acme serves the name with "
            "this node's internal CA; neither flag keeps an existing domain's "
            "current setting."
        ),
    ),
) -> None:
    """Add a direct domain to a deployed app."""
    asyncio.run(_add_async(app, domain, acme=acme))


async def _add_async(app: str, domain: str, *, acme: bool | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.add_domain(
            app,
            domain,
            acme=acme,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # The envelope carries the hint that tells the operator what to do next
        # (which grammar rule the name broke, that the name is taken, that
        # --acme is not available yet) — no local guess is layered on top.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    result = result if isinstance(result, dict) else {}
    # The daemon folds the name (case, trailing dot), so echo *its* value, not
    # the argv one — otherwise the operator sees a name that is not the row.
    stored = result.get("domain") or domain
    console.print(
        f"[green]Added [bold]{_plain(stored)}[/bold] → "
        f"[bold]{_plain(app)}[/bold]:[/green] {_plain(result.get('url'))}"
    )
    _state_note(result)
    _cert_note(result)
    note = _DNS_NOTE.format(domain=_plain(stored))
    cert_state = result.get("cert_state")
    if not (isinstance(cert_state, str) and cert_state in _NO_TRUST_CLAUSE_STATES):
        note += _TRUST_CLAUSE
    console.print(f"[dim]{note}[/dim]")


@domains_app.command("remove")
def domains_remove(
    app: str = typer.Argument(..., help="Name of the deployed app."),
    domain: str = typer.Argument(..., help="The direct domain to remove."),
) -> None:
    """Remove a direct domain; the route disappears on the next reconcile tick."""
    asyncio.run(_remove_async(app, domain))


async def _remove_async(app: str, domain: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.remove_domain(app, domain, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    result = result if isinstance(result, dict) else {}
    if result.get("removed"):
        console.print(
            f"[green]Removed [bold]{_plain(domain)}[/bold] from [bold]{_plain(app)}[/bold].[/green]"
        )
    else:
        # Idempotent by design — say so plainly rather than reporting success
        # for a domain that was never added.
        console.print(
            f"[dim]{_plain(domain)} was not a domain of {_plain(app)} — nothing to do.[/dim]"
        )
