"""Manage plain and secret variables at project, service, or machine scope."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Optional
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.commands.deploy import parse_env_pairs
from nerdit.cli.commands.secrets import SHARED_SERVICE, prompt_values
from nerdit.cli.commands.secrets import _set_async as _set_machine_async
from nerdit.cli.display import call_or_exit, console, render_client_error
from nerdit.cli.display import plain as _plain

vars_app = typer.Typer(
    name="vars",
    help="Manage project variables (plain or secret; per project or per service).",
    no_args_is_help=True,
)

_TARGET_HELP = "Project, or project/service for one service's scope"


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]{message}[/red]")
    return typer.Exit(1)


def _split_target(target: str) -> tuple[str, str | None]:
    """`asso` → the project scope; `asso/web` → that service's scope."""
    project, _, service = target.partition("/")
    if not project or (not service and "/" in target):
        raise _fail("Expected <project> or <project>/<service>.")
    return project, service or None


def _echo_json(data: dict) -> None:
    typer.echo(json.dumps(data, indent=2, sort_keys=True))


@vars_app.command("set")
def vars_set(
    target: Optional[str] = typer.Argument(None, help=_TARGET_HELP),
    pairs: Optional[list[str]] = typer.Argument(None, help="Plain KEY=VAL pairs"),
    secret: bool = typer.Option(
        False, "--secret", help="Write-only value; requires --prompt KEY (never KEY=VAL)."
    ),
    prompt_keys: Annotated[
        Optional[list[str]],
        typer.Option(
            "--prompt",
            metavar="KEY",
            help="Read a secret value without echo (repeatable; implies --secret).",
        ),
    ] = None,
    machine: bool = typer.Option(
        False, "--machine", help="Machine-wide scope: same as `nerdit secrets set --shared`."
    ),
) -> None:
    """Set/merge variables. KEY=VAL is plain; a secret is --secret --prompt KEY.

    On a project that does not exist yet, the project is created for your token.
    """
    pairs = list(pairs or [])
    prompt_keys = list(prompt_keys or [])
    # Same Click quirk as `secrets set --shared KEY=VAL`: with no positional
    # target the first pair lands in `target` (a target can never contain '=').
    # Rebound for EVERY mode, not just --machine: a forgotten <project> must
    # meet the refusals below, never travel as the project path segment (URL,
    # audit row, success line).
    if target is not None and "=" in target:
        pairs, target = [target, *pairs], None
    if machine and target is not None:
        raise _fail("--machine and <project> are mutually exclusive.")
    # --prompt implies --secret: the fail-safe direction (D-P40-16).
    secret = secret or bool(prompt_keys)
    if secret and pairs:
        # Refused before anything is parsed or sent; the pair is NOT echoed back.
        raise _fail(
            "A secret value never goes on the command line. "
            "Use --prompt KEY (hidden input) instead of KEY=VAL."
        )
    if not pairs and not prompt_keys:
        raise _fail("Provide KEY=VAL pairs, or --secret --prompt KEY.")
    try:
        values = parse_env_pairs(pairs)
        values.update(prompt_values(prompt_keys))
    except ValueError as exc:
        raise _fail(_plain(exc)) from exc
    if machine:
        asyncio.run(_set_machine_async(SHARED_SERVICE, values))
        return
    if target is None:
        raise _fail("Missing argument: <project> (or pass --machine).")
    project, service = _split_target(target)
    asyncio.run(_set_async(project, service, values, secret=secret))


async def _set_async(
    project: str, service: str | None, values: dict[str, str], *, secret: bool
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(
        client.set_variables(
            project, values, secret=secret, service=service, idempotency_key=uuid4().hex
        )
    )
    kind = "plain" if result.get("plain") else "secret"
    # Names of THIS write only, from the local map: never a value.
    console.print(
        f"[green]Set {kind} variable(s) in {_plain(project)} "
        f"({_plain(result.get('scope'))}):[/green] {', '.join(_plain(k) for k in sorted(values))}"
    )


@vars_app.command("unset")
def vars_unset(
    target: str = typer.Argument(..., help=_TARGET_HELP),
    key: str = typer.Argument(..., help="Variable key to delete"),
) -> None:
    """Delete one variable from a scope."""
    if "=" in target or "=" in key:
        # A pasted KEY=VAL would ride the DELETE path and the audit `keys`; not echoed.
        raise _fail("Expected <project> KEY: a key name, never KEY=VAL.")
    project, service = _split_target(target)
    asyncio.run(_unset_async(project, service, key))


async def _unset_async(project: str, service: str | None, key: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(
        client.delete_variable(project, key, service=service, idempotency_key=uuid4().hex)
    )
    console.print(
        f"[green]Deleted {_plain(key)} from {_plain(project)} "
        f"({_plain(result.get('scope'))}).[/green]"
    )


def _render(title: str, rows: list[dict], *, values: bool) -> None:
    if not rows:
        console.print("[dim]No variables.[/dim]")
        return
    table = Table(title=title)
    table.add_column("Key", style="cyan")
    table.add_column("Scope")
    table.add_column("Kind")
    if values:
        table.add_column("Value")
    for row in rows:
        cells = [
            _plain(row.get("key")),
            _plain(row.get("scope")),
            "plain" if row.get("plain") else "secret",
        ]
        if values:
            # The server withholds a secret's value (null); this is a placeholder.
            value = row.get("value")
            cells.append(
                _plain(value) if row.get("plain") and value is not None else "[dim]•••[/dim]"
            )
        table.add_row(*cells)
    console.print(table)


@vars_app.command("list")
def vars_list(
    target: str = typer.Argument(..., help=_TARGET_HELP),
    json_out: bool = typer.Option(False, "--json", help="Print the raw JSON body."),
) -> None:
    """List one scope's variables: plain values shown, secret values never."""
    project, service = _split_target(target)
    asyncio.run(_read_async(project, service, json_out=json_out, resolve=False))


@vars_app.command("resolve")
def vars_resolve(
    target: str = typer.Argument(..., help="Project, or project/service (default service: web)"),
    json_out: bool = typer.Option(False, "--json", help="Print the raw JSON body."),
) -> None:
    """Show, per key, the scope a launch would take it from. Never a value."""
    project, service = _split_target(target)
    asyncio.run(_read_async(project, service, json_out=json_out, resolve=True))


async def _read_async(project: str, service: str | None, *, json_out: bool, resolve: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        read = client.resolve_variables if resolve else client.list_variables
        result = await read(project, service=service)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    if json_out:
        _echo_json(result)
        return
    where = result.get("service") if resolve else result.get("scope")
    _render(f"{_plain(project)} ({_plain(where)})", result.get("variables", []), values=not resolve)
