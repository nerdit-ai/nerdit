"""List served models; use nerdit serve to create and nerdit services for lifecycle writes."""

from __future__ import annotations

import asyncio

import typer

from nerdit.cli.display import console, display_model_table, render_client_error

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
    try:
        page = await client.list_models()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    items = page.get("items", [])
    if not items:
        console.print("[dim]No models. Serve one with `nerdit serve <model>`.[/dim]")
        return
    display_model_table(items)
