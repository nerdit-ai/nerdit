"""Manage projects: the grouping every deployed service belongs to (P40b).

A project row outlives its services and reserves the name for the token that
created it (D-P40-5); ``nerdit projects delete`` releases it. Each write mints
an idempotency key. ``--json`` on ``list``/``show`` prints the raw body through
``typer.echo`` so Rich never reflows machine output (the ``capabilities``
precedent). Wire identity stays the label: a service is addressed by its name
everywhere else in the CLI (D-P40-6).
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.display import call_or_exit, console, display_service_table, render_client_error
from nerdit.cli.display import plain as _plain

projects_app = typer.Typer(
    name="projects",
    help="Manage projects (a project groups an app's services and reserves its name).",
    no_args_is_help=True,
)


def _echo_json(data: dict) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True))


@projects_app.command("list")
def projects_list(
    json_out: bool = typer.Option(False, "--json", help="Print the raw JSON body."),
) -> None:
    """List the projects this token can see (bounded, first page)."""
    asyncio.run(_list_async(json_out=json_out))


async def _list_async(*, json_out: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    page = await call_or_exit(client.list_projects())
    if json_out:
        _echo_json(page)
        return
    items = page.get("items", [])
    if not items:
        console.print("[dim]No projects. Deploy an app or run `nerdit projects create`.[/dim]")
        return
    table = Table(title="Projects")
    table.add_column("Name", style="cyan")
    table.add_column("ID")
    table.add_column("Services")
    table.add_column("Addresses")
    for item in items:
        services = [svc.get("name") for svc in item.get("services", [])]
        addresses = [entry.get("url") for entry in item.get("addresses", [])]
        table.add_row(
            _plain(item.get("name")),
            _plain(item.get("id")),
            ", ".join(_plain(s) for s in services) if services else "[dim]none[/dim]",
            "\n".join(_plain(a) for a in addresses) if addresses else "[dim]none[/dim]",
        )
    console.print(table)


@projects_app.command("create")
def projects_create(
    name: str = typer.Argument(..., help="Project name (lowercase DNS label, ≤ 40, no '--')"),
) -> None:
    """Create an empty project, reserving its name for your token."""
    asyncio.run(_create_async(name))


async def _create_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(client.create_project(name, idempotency_key=uuid4().hex))
    console.print(
        f"[green]Created project {_plain(result.get('name'))}[/green] ({_plain(result.get('id'))})"
    )


@projects_app.command("rename")
def projects_rename(
    project_id: str = typer.Argument(..., help="Immutable project ID (prj_…)."),
    name: str = typer.Argument(..., help="New display name (lowercase DNS label, ≤ 40)."),
) -> None:
    """Rename a project's display label; service names and public URLs stay fixed."""
    asyncio.run(_rename_async(project_id, name))


async def _rename_async(project_id: str, name: str) -> None:
    from nerdit.cli.client import get_configured_client

    try:
        result = await get_configured_client().rename_project(
            project_id, name, idempotency_key=uuid4().hex
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    console.print(f"Renamed project {_plain(result.get('id'))} to {_plain(result.get('name'))}.")


@projects_app.command("show")
def projects_show(
    name: str = typer.Argument(..., help="Immutable project ID or original namespace"),
    json_out: bool = typer.Option(False, "--json", help="Print the raw JSON body."),
) -> None:
    """Show a project: services, referenced resources, addresses, home node."""
    asyncio.run(_show_async(name, json_out=json_out))


async def _show_async(name: str, *, json_out: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    project = await call_or_exit(client.get_project(name))
    if json_out:
        _echo_json(project)
        return
    home = project.get("home") or {}
    console.print(
        f"[bold]{_plain(project.get('name'))}[/bold] ({_plain(project.get('id'))}) "
        f"on {_plain(home.get('hostname'))}"
    )
    services = project.get("services", [])
    if services:
        display_service_table(services)
    else:
        console.print("[dim]No services yet.[/dim]")
    resources = project.get("resources", [])
    if resources:
        table = Table(title="Resources")
        table.add_column("Service", style="cyan")
        table.add_column("Type")
        table.add_column("Binding")
        table.add_column("Provider")
        table.add_column("Target")
        table.add_column("Ready")
        for res in resources:
            table.add_row(
                *(
                    _plain(res.get(key))
                    for key in ("service", "type", "binding", "provider", "target", "ready")
                )
            )
        console.print(table)
    for entry in project.get("addresses", []):
        console.print(f"  {_plain(entry.get('kind'))}: {_plain(entry.get('url'))}")


@projects_app.command("delete")
def projects_delete(
    name: str = typer.Argument(..., help="Immutable project ID or original namespace"),
    purge: str = typer.Option(
        "secrets", "--purge", help="CSV of purge targets per service: secrets,data,images,workspace"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the data-purge confirmation"),
) -> None:
    """Delete a project: every service in it, then the row (releases the name).

    A refused service leaves the project in place (`409 project.delete_incomplete`
    lists what went and what did not); re-run after fixing the failures.
    """
    targets = {t.strip().lower() for t in purge.split(",") if t.strip()}
    if "data" in targets and not yes:
        typer.confirm(
            f"Purge the data dir of every service in '{name}' permanently (irreversible)?",
            abort=True,
        )
    asyncio.run(_delete_async(name, purge=purge))


def _failed_entries(exc: Exception) -> list[dict]:
    """The `failed` list of a `project.delete_incomplete` envelope, else `[]`."""
    response = getattr(exc, "response", None)
    try:
        body = response.json() if response is not None else None
    except Exception:  # noqa: BLE001 — a non-JSON body has no entries
        return []
    failed = body.get("failed") if isinstance(body, dict) else None
    return [f for f in failed if isinstance(f, dict)] if isinstance(failed, list) else []


async def _delete_async(name: str, *, purge: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.delete_project(name, purge=purge, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        # `409 project.delete_incomplete` carries the per-service refusals under
        # `failed`; the generic renderer shows message + hint only, so list them.
        for entry in _failed_entries(exc):
            console.print(
                f"  [red]{_plain(entry.get('name'))}[/red]: {_plain(entry.get('code'))} — "
                f"{_plain(entry.get('message'))}"
            )
        raise typer.Exit(1) from exc
    deleted = result.get("deleted", [])
    console.print(
        f"[green]Deleted project {_plain(name)}[/green] "
        f"({len(deleted)} service(s): {', '.join(_plain(s) for s in deleted) or 'none'})."
    )
