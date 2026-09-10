"""Browse embedded app templates and deploy their Git sources with optional env and secrets."""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.commands.deploy import parse_env_pairs
from nerdit.cli.display import console, display_deploy_result, render_client_error

store_app = typer.Typer(
    name="store",
    help="Browse and deploy from the app template store.",
    no_args_is_help=True,
)


@store_app.command("list")
def store_list() -> None:
    """List the app template catalog."""
    asyncio.run(_list_async())


async def _list_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        templates = await client.list_app_templates()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    if not templates:
        console.print("[dim]No templates.[/dim]")
        return

    table = Table(title="App templates")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Description")
    for tpl in templates:
        table.add_row(
            tpl.get("id", ""),
            tpl.get("name", ""),
            tpl.get("category", ""),
            tpl.get("description", ""),
        )
    console.print(table)


@store_app.command("show")
def store_show(
    template_id: str = typer.Argument(..., help="Template id"),
) -> None:
    """Show one template's coordinates, defaults and env schema."""
    asyncio.run(_show_async(template_id))


async def _show_async(template_id: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        tpl = await client.get_app_template(template_id)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    console.print(f"[cyan]{tpl.get('id')}[/cyan] — {tpl.get('name')}")
    console.print(f"  Category:    {tpl.get('category')}")
    console.print(f"  Description: {tpl.get('description')}")
    console.print(f"  Repo:        {tpl.get('repo_url')}")
    if tpl.get("ref"):
        console.print(f"  Ref:         {tpl['ref']}")
    if tpl.get("subdir"):
        console.print(f"  Subdir:      {tpl['subdir']}")

    defaults = tpl.get("deploy_defaults") or {}
    shown = {k: v for k, v in defaults.items() if v is not None}
    if shown:
        rendered = ", ".join(f"{k}={v}" for k, v in shown.items())
        console.print(f"  Defaults:    {rendered}")

    env_schema = tpl.get("env_schema") or []
    if env_schema:
        console.print("  Env schema:")
        for item in env_schema:
            flags = []
            if item.get("required"):
                flags.append("required")
            if item.get("secret"):
                flags.append("secret")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            desc = item.get("description") or ""
            console.print(f"    - {item.get('name')}{suffix} {desc}".rstrip())

    if tpl.get("ai_hint"):
        console.print(f"  AI hint:     {tpl['ai_hint']}")


@store_app.command("deploy")
def store_deploy(
    template_id: str = typer.Argument(..., help="Template id"),
    name: str = typer.Option(..., "--name", "-n", help="Service name (DNS label)"),
    env: list[str] = typer.Option(
        [], "--env", "-e", help="Environment variable KEY=VAL (repeatable)"
    ),
    secret: list[str] = typer.Option(
        [], "--secret", "-s", help="Secret KEY=VAL written write-only (repeatable)"
    ),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Container port override"),
    gpus: Optional[int] = typer.Option(None, "--gpus", "-g", help="GPUs the app needs"),
    start: Optional[str] = typer.Option(None, "--start", help="Start command override"),
    build_settings: Optional[str] = typer.Option(
        None,
        "--build-settings",
        help=(
            "JSON build overrides, including preset (node/nextjs/python/dockerfile); "
            "null resets, build:false skips compilation"
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview without building or writing secrets"
    ),
    health: Optional[str] = typer.Option(None, "--health", help="HTTP health path override"),
    vendor: Optional[str] = typer.Option(None, "--vendor", help="Force a GPU vendor"),
) -> None:
    """Deploy an app template: clone the catalog repo server-side, build, run."""
    from nerdit.cli.commands.deploy import parse_build_settings

    asyncio.run(
        _deploy_async(
            template_id,
            name,
            env,
            secret,
            port,
            gpus,
            start,
            health,
            vendor,
            build_settings=parse_build_settings(build_settings),
            dry_run=dry_run,
        )
    )


async def _deploy_async(
    template_id: str,
    name: str,
    env: list[str],
    secret: list[str],
    port: int | None,
    gpus: int | None,
    start: str | None,
    health: str | None,
    vendor: str | None,
    *,
    build_settings: dict | None = None,
    dry_run: bool = False,
) -> None:
    from nerdit.cli.client import get_configured_client

    try:
        env_values: dict[str, str | None] | None = dict(parse_env_pairs(env)) if env else None
        secret_values = parse_env_pairs(secret) if secret else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    client = get_configured_client()
    console.print(f"[dim]Deploying template {template_id}...[/dim]")
    try:
        service = await client.deploy_template(
            template_id,
            name=name,
            env=env_values,
            secrets=secret_values,
            port=port,
            gpus=gpus,
            start=start,
            build_settings=build_settings,
            dry_run=dry_run,
            health=health,
            vendor=vendor,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    if dry_run:
        from nerdit.cli.commands.deploy import _render_dry_run

        _render_dry_run(service)
    else:
        display_deploy_result(service)
