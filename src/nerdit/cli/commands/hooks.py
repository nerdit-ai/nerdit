"""Manage an app's inbound webhooks (Stripe, GitHub, Slack, generic HMAC).

The daemon validates every field. The signing secret never rides argv: it is
read with a hidden prompt and stored through the existing secrets route.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.display import call_or_exit, console
from nerdit.cli.display import plain as _plain

hooks_app = typer.Typer(
    name="hooks",
    help="Inbound webhooks for a deployed app (from Stripe, GitHub, Slack...).",
    no_args_is_help=True,
)


@hooks_app.command("list")
def hooks_list(app: str = typer.Argument(..., help="Name of the deployed app.")) -> None:
    """List an app's webhooks and their URLs."""
    asyncio.run(_list_async(app))


async def _list_async(app: str) -> None:
    from nerdit.cli.client import get_configured_client

    result = await call_or_exit(get_configured_client().list_hooks(app))
    rows = (result if isinstance(result, dict) else {}).get("hooks") or []
    if not rows:
        console.print(f"[dim]{_plain(app)} has no webhooks.[/dim]")
        return
    table = Table(show_header=True, header_style="bold", title=f"webhooks — {app}")
    for column in ("name", "sender", "path", "url"):
        table.add_column(column)
    for row in rows:
        row = row if isinstance(row, dict) else {}
        table.add_row(
            _plain(row.get("name")),
            _plain(row.get("sender")),
            _plain(row.get("path")),
            _plain(row.get("url") or "-"),
        )
    console.print(table)


@hooks_app.command("add")
def hooks_add(  # noqa: PLR0913 - one option per request field
    app: str = typer.Argument(..., help="Name of the deployed app."),
    name: str = typer.Argument(..., help="Webhook name, e.g. stripe (lowercase, digits, '-')."),
    sender: str = typer.Option(..., "--sender", help="stripe | github | slack | hmac | none"),
    path: str = typer.Option(..., "--path", help="App path to POST to, e.g. /webhooks/stripe."),
    header: str | None = typer.Option(None, "--header", help="hmac only: the signature header."),
    prefix: str | None = typer.Option(None, "--prefix", help="hmac only: e.g. 'sha256='."),
    secret_prompt: bool = typer.Option(
        False, "--secret-prompt", help="Then read the provider's signing secret without echo."
    ),
) -> None:
    """Add an inbound webhook to a deployed app."""
    asyncio.run(_add_async(app, name, sender, path, header, prefix, secret_prompt=secret_prompt))


async def _add_async(  # noqa: PLR0913
    app: str,
    name: str,
    sender: str,
    path: str,
    header: str | None,
    prefix: str | None,
    *,
    secret_prompt: bool,
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result: Any = await call_or_exit(
        client.add_hook(
            app,
            name,
            sender=sender,
            path=path,
            header=header,
            prefix=prefix,
            idempotency_key=uuid4().hex,
        )
    )
    result = result if isinstance(result, dict) else {}
    console.print(
        f"[green]Webhook [bold]{_plain(name)}[/bold] on [bold]{_plain(app)}[/bold].[/green]"
    )
    key = result.get("secret_key")
    if isinstance(key, str) and key:
        if secret_prompt:
            value = typer.prompt("Secret", hide_input=True)
            await call_or_exit(client.set_secrets(app, {key: value}, idempotency_key=uuid4().hex))
            console.print(f"[green]Signing secret stored as {_plain(key)}.[/green]")
        else:
            console.print(
                f"[dim]Store the provider's signing secret: "
                f"nerdit secrets set {_plain(app)} --prompt {_plain(key)}[/dim]"
            )
    url = result.get("url")
    if url:
        console.print(f"Webhook URL: {_plain(url)}")
    else:
        console.print(
            f"[dim]link this machine and share {_plain(app)} to get its Webhook URL[/dim]"
        )


@hooks_app.command("rm")
def hooks_rm(
    app: str = typer.Argument(..., help="Name of the deployed app."),
    name: str = typer.Argument(..., help="The webhook to remove."),
) -> None:
    """Remove a webhook; its address stops answering. The signing secret is kept."""
    asyncio.run(_rm_async(app, name))


async def _rm_async(app: str, name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(client.remove_hook(app, name, idempotency_key=uuid4().hex))
    if (result if isinstance(result, dict) else {}).get("removed"):
        console.print(f"[green]Removed webhook [bold]{_plain(name)}[/bold].[/green]")
    else:
        console.print(f"[dim]{_plain(app)} has no webhook named {_plain(name)}.[/dim]")
