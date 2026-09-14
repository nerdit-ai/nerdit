"""Manage write-only service or shared secrets and rotate the encryption key.

API responses contain key names only; values enter containers at launch.
Each write mints an idempotency key. Shared writes and key rotation require
admin access; rotation confirms before replacing the old key.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional
from uuid import uuid4

import typer

from nerdit.cli.commands.deploy import parse_env_pairs
from nerdit.cli.display import _plain, console, render_client_error
from nerdit.core.secrets import validate_secret_items

SHARED_SERVICE = "shared"

secrets_app = typer.Typer(
    name="secrets",
    help="Manage per-service secrets (write-only: values are never shown).",
    no_args_is_help=True,
)


def _print_keys(result: dict) -> None:
    keys = result.get("keys", [])
    if not keys:
        console.print(f"[dim]No secrets for service {result.get('service')}.[/dim]")
        return
    console.print(f"[green]Secrets for {result.get('service')}:[/green] (names only)")
    for key in keys:
        console.print(f"  {key}")


def _resolve_service(service: Optional[str], shared: bool) -> str:
    """Resolve the positional `service` against `--shared` sugar.

    `--shared` and `<service>` are mutually exclusive; exactly one must be
    given so the command always has an unambiguous target.
    """
    if shared and service:
        console.print("[red]--shared and <service> are mutually exclusive.[/red]")
        raise typer.Exit(1)
    if shared:
        return SHARED_SERVICE
    if not service:
        console.print("[red]Missing argument: <service> (or pass --shared).[/red]")
        raise typer.Exit(1)
    return service


@secrets_app.command("set")
def secrets_set(
    service: Optional[str] = typer.Argument(None, help="Service name"),
    pairs: Optional[list[str]] = typer.Argument(None, help="Secret KEY=VAL pairs"),
    shared: bool = typer.Option(
        False, "--shared", help="Target the global shared scope instead of a service."
    ),
    prompt_keys: Annotated[
        Optional[list[str]],
        typer.Option(
            "--prompt", metavar="KEY", help="Read a secret value without echo (repeatable)."
        ),
    ] = None,
) -> None:
    """Set/merge secrets for a service (values are write-only)."""
    pairs = list(pairs or [])
    # Click cannot know `secrets set --shared KEY=VAL` has no positional
    # service: the first pair lands in `service`. Redistribute it before
    # resolving (a value with '=' can never be a service name).
    if shared and service is not None and "=" in service:
        pairs = [service, *pairs]
        service = None
    target = _resolve_service(service, shared)
    if not pairs and not prompt_keys:
        console.print("[red]Provide KEY=VAL pairs or --prompt KEY.[/red]")
        raise typer.Exit(1)
    try:
        values = parse_env_pairs(pairs)
        validate_secret_items(dict.fromkeys(prompt_keys or [], ""))
        for key in prompt_keys or []:
            values[key] = typer.prompt(f"Value for {key}", hide_input=True)
    except ValueError as exc:
        console.print(f"[red]{_plain(exc)}[/red]")
        raise typer.Exit(1) from exc
    asyncio.run(_set_async(target, values))


async def _set_async(service: str, values: dict[str, str]) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.set_secrets(service, values, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    _print_keys(result)


@secrets_app.command("list")
def secrets_list(
    service: Optional[str] = typer.Argument(None, help="Service name"),
    shared: bool = typer.Option(
        False, "--shared", help="Target the global shared scope instead of a service."
    ),
) -> None:
    """List a service's secret key names (values are never shown)."""
    target = _resolve_service(service, shared)
    asyncio.run(_list_async(target))


async def _list_async(service: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.list_secrets(service)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    _print_keys(result)


@secrets_app.command("rm")
def secrets_rm(
    service: Optional[str] = typer.Argument(None, help="Service name"),
    key: Optional[str] = typer.Option(
        None, "--key", "-k", help="Delete a single key (default: all secrets)"
    ),
    shared: bool = typer.Option(
        False, "--shared", help="Target the global shared scope instead of a service."
    ),
) -> None:
    """Delete one secret key, or all of a service's secrets."""
    target = _resolve_service(service, shared)
    asyncio.run(_rm_async(target, key))


async def _rm_async(service: str, key: str | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        if key:
            await client.delete_secret(service, key, idempotency_key=uuid4().hex)
        else:
            await client.delete_secrets(service, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    if key:
        console.print(f"[green]Deleted secret {key} from {service}.[/green]")
    else:
        console.print(f"[green]Deleted all secrets for {service}.[/green]")


@secrets_app.command("rotate-key")
def secrets_rotate_key() -> None:
    """Rotate the secrets-at-rest encryption key (admin-only).

    Re-encrypts every stored secret file under a fresh key; the daemon
    returns a count of files rewritten (key material never leaves the
    daemon). Destructive/irreversible on the old key, so this confirms
    before proceeding.
    """
    if not typer.confirm("Rotate the secrets encryption key and re-encrypt all stored secrets?"):
        raise typer.Exit(0)
    asyncio.run(_rotate_key_async())


async def _rotate_key_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.rotate_secrets_key(idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    console.print(
        f"[green]Rotated secrets key — {result.get('services_rewritten', 0)} "
        "file(s) re-encrypted.[/green]"
    )
