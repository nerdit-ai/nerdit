"""Manage the machine AI gateway: aliases, per-app virtual keys and usage.

An alias names a provider model and a `${secrets.shared.KEY}` reference to its
key; the key itself is set with `nerdit secrets set --shared` and never passes
through these commands. Writes take a submitter or admin token and mint an
idempotency key; a submitter revokes the keys of its own apps only.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import typer

from nerdit.cli.display import _plain, call_or_exit, console

ai_app = typer.Typer(
    name="ai",
    help="Manage the machine AI gateway (aliases, app keys, usage).",
    no_args_is_help=True,
)
routes_app = typer.Typer(name="routes", help="Model aliases apps call.", no_args_is_help=True)
keys_app = typer.Typer(
    name="keys", help="Per-app virtual keys (never shown).", no_args_is_help=True
)
ai_app.add_typer(routes_app, name="routes")
ai_app.add_typer(keys_app, name="keys")


def _client():
    from nerdit.cli.client import get_configured_client

    return get_configured_client()


def _off_note(enabled: object) -> None:
    if enabled is False:
        console.print(
            "[yellow]The AI gateway is off on this machine: set \\[ai_gateway] enabled = true "
            "in config.toml and restart the daemon.[/yellow]"
        )


@routes_app.command("list")
def routes_list() -> None:
    """List the machine's aliases (provider keys show as their reference only)."""
    result = asyncio.run(call_or_exit(_client().list_ai_routes()))
    routes = result.get("routes", [])
    if not routes:
        console.print("[dim]No AI gateway aliases.[/dim]")
    for route in routes:
        target = route.get("base_url") or "local ollama"
        ref = route.get("api_key_ref") or "-"
        console.print(
            f"  {_plain(route.get('alias'))}  {_plain(route.get('provider'))}  "
            f"{_plain(route.get('model'))}  {_plain(target)}  key={_plain(ref)}"
        )
    _off_note(result.get("enabled"))


@routes_app.command("set")
def routes_set(
    alias: str = typer.Argument(..., help="Alias apps put in [ai.*] model, e.g. 'fast'."),
    model: str = typer.Option(..., "--model", help="Provider model the alias maps to."),
    provider: str = typer.Option("api", "--provider", help="'api' or 'ollama'."),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Provider 'api': OpenAI-compatible base URL (https, …/v1)."
    ),
    key_ref: Optional[str] = typer.Option(
        None,
        "--key-ref",
        help="Provider 'api': a reference like ${secrets.shared.OPENROUTER_API_KEY}, "
        "never the key itself.",
    ),
) -> None:
    """Create or replace an alias (submitter or admin).

    Example: nerdit ai routes set fast --model openai/gpt-4o-mini
    --base-url https://openrouter.ai/api/v1 --key-ref '${secrets.shared.OPENROUTER_API_KEY}'
    """
    result = asyncio.run(
        call_or_exit(
            _client().set_ai_route(
                alias,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_ref=key_ref,
                idempotency_key=uuid4().hex,
            )
        )
    )
    verb = "Created" if result.get("created") else "Updated"
    console.print(f"[green]{verb} alias {_plain(alias)}.[/green]")
    _off_note(result.get("enabled"))


@routes_app.command("rm")
def routes_rm(
    alias: str = typer.Argument(..., help="Alias to remove."),
    force: bool = typer.Option(False, "--force", help="Remove even if a running app uses it."),
) -> None:
    """Remove an alias (submitter or admin). Refused while a running app uses it, unless --force."""
    asyncio.run(
        call_or_exit(_client().remove_ai_route(alias, force=force, idempotency_key=uuid4().hex))
    )
    console.print(f"[green]Removed alias {_plain(alias)}.[/green]")


@keys_app.command("list")
def keys_list(
    all_keys: bool = typer.Option(False, "--all", help="Include revoked keys."),
) -> None:
    """List which apps hold a virtual key (the key itself is never shown)."""
    result = asyncio.run(call_or_exit(_client().list_ai_gateway_keys(include_revoked=all_keys)))
    keys = result.get("keys", [])
    if not keys:
        console.print("[dim]No AI gateway keys.[/dim]")
    for key in keys:
        revoked = key.get("revoked_at")
        state = f"revoked {revoked}" if revoked else "active"
        console.print(
            f"  {_plain(key.get('service_name'))}  since {_plain(key.get('created_at'))}  "
            f"{_plain(state)}"
        )


@keys_app.command("revoke")
def keys_revoke(service: str = typer.Argument(..., help="App whose keys to revoke.")) -> None:
    """Revoke an app's virtual keys (its owner or admin); it gets a new one when it restarts."""
    result = asyncio.run(
        call_or_exit(_client().revoke_ai_gateway_keys(service, idempotency_key=uuid4().hex))
    )
    console.print(f"[green]Revoked {result.get('revoked', 0)} key(s) of {_plain(service)}.[/green]")


@ai_app.command("usage")
def usage(
    service: Optional[str] = typer.Option(None, "--service", help="Only this app."),
    days: int = typer.Option(7, "--days", min=1, max=90, help="UTC days back, today included."),
) -> None:
    """Show requests, tokens and upstream errors per day, app and alias."""
    result = asyncio.run(call_or_exit(_client().get_ai_usage(service=service, days=days)))
    rows = result.get("usage", [])
    if not rows:
        console.print("[dim]No AI gateway usage.[/dim]")
    for row in rows:
        console.print(
            f"  {_plain(row.get('day'))}  {_plain(row.get('service_name'))}  "
            f"{_plain(row.get('alias'))}  requests={row.get('requests', 0)}  "
            f"prompt={row.get('prompt_tokens', 0)}  completion={row.get('completion_tokens', 0)}  "
            f"errors={row.get('upstream_errors', 0)}"
        )
