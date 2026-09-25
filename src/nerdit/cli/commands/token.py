"""Display the configured token or manage scoped API tokens.

Create/list/revoke require admin access; whoami/rotate are self-service.
Creation and rotation show plaintext once; listing exposes no hashes or secrets.
"""

from __future__ import annotations

import asyncio
import re

import typer

from nerdit.cli.display import (
    call_or_exit,
    console,
    display_token_self,
    display_token_table,
)
from nerdit.config.settings import load_settings

# ``--expires-in`` grammar: a bare second count, or a number with an h/d/w
# suffix. Deliberately tiny — anything else is rejected with a clear message
# rather than silently reinterpreted (a mis-parsed TTL is a credential that
# outlives its intent).
_DURATION_RE = re.compile(r"^(\d+)([smhdw]?)$")
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

token_app = typer.Typer(
    name="token",
    help="Display or manage API tokens.",
)


@token_app.callback(invoke_without_command=True)
def token_main(ctx: typer.Context) -> None:
    """Display the current authentication token (when no sub-command is given)."""
    if ctx.invoked_subcommand is not None:
        return

    settings = load_settings()
    auth_token = settings.daemon.auth_token
    if settings.client.remote_host:
        auth_token = settings.client.auth_token

    if auth_token:
        console.print(auth_token)
    else:
        console.print("[yellow]No authentication token configured.[/yellow]")


def parse_duration(value: str) -> int:
    """Parse `3600` / `24h` / `30d` into seconds; raise on anything else."""
    match = _DURATION_RE.match(value.strip())
    if not match or match.group(1) == "":
        raise ValueError(
            f"Invalid duration '{value}': use seconds (3600) or a number with an "
            "s/m/h/d/w suffix (90m, 24h, 30d, 2w)."
        )
    return int(match.group(1)) * _DURATION_UNITS[match.group(2)]


@token_app.command("create")
def token_create(
    name: str = typer.Argument(..., help="Human-readable label for the token."),
    role: str = typer.Option("submitter", "--role", help="admin | submitter | readonly."),
    max_gpus: int | None = typer.Option(
        None, "--max-gpus", help="Cumulative GPU cap across active jobs."
    ),
    max_concurrent_jobs: int | None = typer.Option(
        None, "--max-concurrent-jobs", help="Concurrent active-job cap."
    ),
    expires_in: str | None = typer.Option(
        None,
        "--expires-in",
        help="Lifetime: seconds (3600) or a suffixed duration (24h, 30d). "
        "Omitted uses the daemon's [security].token_default_ttl_s.",
    ),
    scope: list[str] = typer.Option(
        [],
        "--scope",
        help="Restrict this token to a service (repeatable). Admin tokens cannot be scoped.",
    ),
) -> None:
    """Create a scoped API token (admin token required)."""
    expires_in_s: int | None = None
    if expires_in is not None:
        try:
            expires_in_s = parse_duration(expires_in)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
    asyncio.run(
        _create_async(name, role, max_gpus, max_concurrent_jobs, expires_in_s, list(scope) or None)
    )


async def _create_async(
    name: str,
    role: str,
    max_gpus: int | None,
    max_concurrent_jobs: int | None,
    expires_in_s: int | None = None,
    scope_services: list[str] | None = None,
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(
        client.create_token(
            name,
            role=role,
            max_gpus=max_gpus,
            max_concurrent_jobs=max_concurrent_jobs,
            expires_in_s=expires_in_s,
            scope_services=scope_services,
        )
    )

    raw = result.get("token", "")
    console.print(f"[green]Token created:[/green] {result.get('id')} ({result.get('role')})")
    console.print(raw)
    expires_at = result.get("expires_at")
    if expires_at:
        # Named even when the caller did not ask for a TTL: the daemon's
        # ``[security].token_default_ttl_s`` may have supplied one (P25 D-P25-1).
        console.print(f"[yellow]Expires:[/yellow] {expires_at}")
    scope = result.get("scope_services")
    if scope:
        console.print(f"[dim]Scoped to:[/dim] {', '.join(scope)}")
    console.print(
        "[yellow]Store this token now — it is shown only once and cannot be retrieved later."
        "[/yellow]"
    )


@token_app.command("list")
def token_list(
    include_revoked: bool = typer.Option(
        False, "--include-revoked", help="Also show revoked tokens."
    ),
) -> None:
    """List API tokens (admin token required). Hashes/plaintext are never shown."""
    asyncio.run(_list_async(include_revoked))


async def _list_async(include_revoked: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    tokens = await call_or_exit(client.list_tokens(include_revoked=include_revoked))

    if not tokens:
        console.print("[dim]No tokens.[/dim]")
        return
    display_token_table(tokens)


@token_app.command("whoami")
def token_whoami() -> None:
    """Show the token this CLI is authenticating with (any role)."""
    asyncio.run(_whoami_async())


async def _whoami_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    token = await call_or_exit(client.get_self_token())

    display_token_self(token)


@token_app.command("rotate")
def token_rotate(
    extend: bool = typer.Option(
        False, "--extend", help="Also push the expiry forward (rotation never extends silently)."
    ),
    expires_in: str | None = typer.Option(
        None,
        "--expires-in",
        help="New lifetime: seconds (3600) or a suffixed duration (24h, 30d). Implies --extend.",
    ),
) -> None:
    """Rotate this token's secret in place; the new value is shown once."""
    expires_in_s: int | None = None
    if expires_in is not None:
        try:
            expires_in_s = parse_duration(expires_in)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc
        # An explicit lifetime is an unambiguous request to move the clock; the
        # daemon reads it only under ``extend``, so sending one without the flag
        # would be silently ignored (P25 D-P25-4).
        extend = True
    asyncio.run(_rotate_async(extend, expires_in_s))


async def _rotate_async(extend: bool, expires_in_s: int | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(client.rotate_self_token(extend=extend, expires_in_s=expires_in_s))

    console.print(f"[green]Token rotated:[/green] {result.get('id')} ({result.get('role')})")
    console.print(result.get("token", ""))
    expires_at = result.get("expires_at")
    if expires_at:
        console.print(f"[yellow]Expires:[/yellow] {expires_at}")
    else:
        console.print("[dim]No expiry.[/dim]")
    console.print(
        "[yellow]Store this token now — it is shown only once. The previous value stopped "
        "working the moment this rotation committed.[/yellow]"
    )
    console.print(
        "[yellow]If this is the token the CLI is configured with, update "
        "~/.nerdit/config.toml (or your NERDIT_* environment) before the next command."
        "[/yellow]"
    )
    console.print(
        "[dim]Lose this output and the credential is unrecoverable: a replayed rotation "
        "returns no plaintext, so recovery is an admin re-mint ('nerdit token create')."
        "[/dim]"
    )


@token_app.command("revoke")
def token_revoke(
    token_id: str = typer.Argument(..., help="Token id to revoke."),
) -> None:
    """Revoke an API token (admin token required)."""
    asyncio.run(_revoke_async(token_id))


async def _revoke_async(token_id: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    await call_or_exit(client.revoke_token(token_id))

    console.print(f"[green]Revoked token {token_id}.[/green]")


# Backward-compatible alias used by ``app.add_typer``.
token = token_app
