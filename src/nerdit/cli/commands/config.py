"""Read and update daemon or app configuration.

Writes carry a fresh idempotency key and the current ETag for concurrency.
Dry runs show redacted diffs without writing; apply accepts multi-section TOML.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

import typer

from nerdit.cli.display import console, render_client_error

app_config_app = typer.Typer(
    name="app",
    help="Inspect and update a deployed app's config.",
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Inspect and update daemon configuration.",
    no_args_is_help=True,
)


def _coerce(value: str) -> object:
    """Best-effort scalar coercion for a `key=value` CLI argument."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_pairs(pairs: list[str]) -> dict[str, object]:
    """Parse `key=value` arguments into a dict, coercing scalar values."""
    out: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Expected key=value, got '{pair}'.")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Empty key in '{pair}'.")
        out[key] = _coerce(raw)
    return out


@config_app.command("get")
def config_get(
    section: str = typer.Argument(None, help="Section name (omit for all sections)."),
) -> None:
    """Show daemon configuration (secrets redacted)."""
    asyncio.run(_get_async(section))


async def _get_async(section: str | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.get_config(section)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    views = data if isinstance(data, list) else [data]
    for view in views:
        console.print(f"[bold]\\[{view.get('section')}][/bold]")
        for key, value in (view.get("values") or {}).items():
            console.print(f"  {key} = {value!r}")


@config_app.command("set")
def config_set(
    section: str = typer.Argument(..., help="Section name (e.g. services)."),
    pairs: list[str] = typer.Argument(..., help="One or more key=value assignments."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the diff without writing."),
) -> None:
    """Update daemon configuration keys (admin token required)."""
    values = _parse_pairs(pairs)
    asyncio.run(_set_async(section, values, dry_run))


async def _set_async(section: str, values: dict[str, object], dry_run: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        current = await client.get_config(section)
        if_match = current.get("etag") if isinstance(current, dict) else None
        idem = None if dry_run else uuid.uuid4().hex
        result = await client.put_config(
            section,
            values,
            dry_run=dry_run,
            idempotency_key=idem,
            if_match=if_match,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    diff = result.get("diff") or []
    if not diff:
        console.print("[dim]No changes.[/dim]")
    else:
        verb = "Would change" if not result.get("applied") else "Changed"
        console.print(f"[green]{verb}:[/green]")
        for entry in diff:
            console.print(f"  {entry.get('key')}: {entry.get('old')!r} -> {entry.get('new')!r}")
    if result.get("requires_restart"):
        console.print(
            "[yellow]A daemon restart is required for this change to take effect.[/yellow]"
        )


@config_app.command("apply")
def config_apply(
    path: str = typer.Argument(..., help="TOML file to apply, or '-' for stdin."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the diff without writing."),
) -> None:
    """Declaratively apply a multi-section daemon config document."""
    asyncio.run(_apply_async(path, dry_run))


async def _apply_async(path: str, dry_run: bool) -> None:
    import io
    import tomllib

    from nerdit.cli.client import get_configured_client

    try:
        if path == "-":
            raw = sys.stdin.buffer.read()
        else:
            with open(path, "rb") as fh:
                raw = fh.read()
        sections = tomllib.load(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        console.print(f"[red]Failed to parse '{path}': {exc}[/red]")
        raise typer.Exit(1) from exc

    client = get_configured_client()
    try:
        current = await client.get_config(None)
        if_match = None
        if isinstance(current, list) and current:
            if_match = current[0].get("etag")
        idem = None if dry_run else uuid.uuid4().hex
        result = await client.apply_config(
            sections,
            dry_run=dry_run,
            idempotency_key=idem,
            if_match=if_match,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    diff = result.get("diff") or []
    if not diff:
        console.print("[dim]No changes.[/dim]")
    else:
        by_section: dict[str, list[dict]] = {}
        for entry in diff:
            by_section.setdefault(entry.get("section") or "", []).append(entry)
        verb = "Would apply" if not result.get("applied") else "Applied"
        console.print(f"[green]{verb}:[/green]")
        for section, entries in by_section.items():
            console.print(f"[bold]\\[{section}][/bold]")
            for entry in entries:
                op = entry.get("op", "change")
                marker = {"add": "+", "change": "~", "delete": "-"}.get(op, "~")
                console.print(
                    f"  {marker} {entry.get('key')}: {entry.get('old')!r} -> {entry.get('new')!r}"
                )
    if result.get("requires_restart") and result.get("restart_keys"):
        keys = ", ".join(result["restart_keys"])
        console.print(f"[red]Restart nerditd to apply: {keys}[/red]")


config_app.add_typer(app_config_app)


@app_config_app.command("get")
def app_config_get(name: str = typer.Argument(..., help="Deployed app name.")) -> None:
    """Show a deployed app's config (deploy fields, ai bindings, source/revision)."""
    asyncio.run(_app_get_async(name))


async def _app_get_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        view = await client.get_app_config(name)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    console.print(
        f"[bold]{view.get('service_name')}[/bold]  "
        f"(source={view.get('source')}, revision={view.get('revision')})"
    )
    console.print("[bold]\\[deploy][/bold]")
    for key, value in (view.get("deploy") or {}).items():
        console.print(f"  {key} = {value!r}")
    ai = view.get("ai") or {}
    if ai:
        console.print("[bold]\\[ai][/bold]")
        for bname, spec in ai.items():
            console.print(f"  [{bname}]")
            for key, value in spec.items():
                console.print(f"    {key} = {value!r}")
    if view.get("env_keys"):
        console.print(f"[dim]env keys: {', '.join(view['env_keys'])}[/dim]")


@app_config_app.command("set")
def app_config_set(
    name: str = typer.Argument(..., help="Deployed app name."),
    section: str = typer.Argument(..., help="Section name (deploy or ai)."),
    pairs: list[str] = typer.Argument(..., help="One or more key=value assignments."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the diff without writing."),
    restart: bool = typer.Option(
        False, "--restart", help="Restart the service after a real write."
    ),
) -> None:
    """Update a deployed app's config section (admin/owner token required).

    For `ai`, use dotted keys to address a binding's fields, e.g.
    `default.model=llama3.1:8b`.
    """
    values: dict[str, object]
    if section == "ai":
        values = dict(_parse_ai_pairs(pairs))
    else:
        values = _parse_pairs(pairs)
    asyncio.run(_app_set_async(name, section, values, dry_run, restart))


def _parse_ai_pairs(pairs: list[str]) -> dict[str, dict[str, object]]:
    """Parse `binding.key=value` arguments into a nested `{binding: {key: value}}`."""
    out: dict[str, dict[str, object]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Expected binding.key=value, got '{pair}'.")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if "." not in key:
            raise typer.BadParameter(f"Expected binding.key=value, got '{pair}'.")
        binding, _, field = key.partition(".")
        value = _coerce(raw)
        out.setdefault(binding, {})[field] = None if value is None else value
    return out


async def _app_set_async(
    name: str, section: str, values: dict[str, object], dry_run: bool, restart: bool
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        current = await client.get_app_config(name)
        if_match = current.get("etag") if isinstance(current, dict) else None
        idem = None if dry_run else uuid.uuid4().hex
        result = await client.put_app_config(
            name,
            section,
            values,
            dry_run=dry_run,
            restart=restart,
            idempotency_key=idem,
            if_match=if_match,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    if not result.get("applied"):
        console.print("[dim]No write performed (dry-run).[/dim]")
        if result.get("requires_restart"):
            console.print("[dim]Would require a restart to apply (--restart).[/dim]")
    else:
        console.print("[green]Saved.[/green]")
        if result.get("restarted"):
            console.print("[green]Service restarted.[/green]")
        elif result.get("requires_restart"):
            console.print("[yellow]saved; restart to apply (--restart)[/yellow]")


# Bare ``config`` alias used by ``app.add_typer``.
config = config_app
