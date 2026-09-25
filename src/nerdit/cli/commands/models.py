"""List served models; use nerdit serve to create and nerdit services for lifecycle writes."""

from __future__ import annotations

import asyncio

import typer

from nerdit.cli.display import call_or_exit, console, display_model_table

models_app = typer.Typer(
    name="models",
    help="Inspect locally served models.",
    no_args_is_help=True,
)


@models_app.command("list")
def models_list() -> None:
    """List served models (bounded, newest first)."""
    asyncio.run(_list_async())


async def _list_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    page = await call_or_exit(client.list_models())

    items = page.get("items", [])
    if not items:
        console.print("[dim]No models. Serve one with `nerdit serve <model>`.[/dim]")
        return
    display_model_table(items)
