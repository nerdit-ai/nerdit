"""Manage the daemon's offline product license, separate from repository licensing.

Read license blobs from a file or stdin, never argv; do not echo rejected
inputs or print secrets. Admin writes use idempotency keys and update live
license state immediately. Only a configured license path requires restart.
Status shows both the persisted path and verified live state. Escape server text.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import typer

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain

license_app = typer.Typer(
    name="license",
    help="Install, inspect and remove this daemon's product license.",
    no_args_is_help=True,
)


def _section_values(view: Any) -> dict:
    """Unwrap a section view or view list; return an empty mapping for unexpected shapes."""
    if isinstance(view, list):
        view = next((item for item in view if isinstance(item, dict)), {})
    if not isinstance(view, dict):
        return {}
    values = view.get("values")
    return values if isinstance(values, dict) else {}


def _read_blob(file: str | None) -> str:
    """Return the license blob from a FILE path or from stdin. Never from argv.

    `-` means stdin explicitly; an omitted argument means stdin **only when it
    is piped** — an interactive terminal gets usage instead of a silent hang.
    A non-existent path is refused WITHOUT echoing the value, because the most
    likely reason someone passed a non-path is that they pasted the blob.
    """
    if file is None or file == "-":
        if file is None and sys.stdin.isatty():
            console.print("[red]Nothing to install.[/red]")
            console.print(
                "[dim]Pass the license file: nerdit license install ./license.jws "
                "(or pipe it: cat license.jws | nerdit license install -)[/dim]"
            )
            raise typer.Exit(1)
        blob = sys.stdin.read().strip()
        if not blob:
            console.print("[red]No license blob on stdin.[/red]")
            raise typer.Exit(1)
        return blob

    path = Path(file).expanduser()
    if not path.is_file():
        # The value is NOT echoed: a pasted blob is the likeliest non-path, and
        # printing it would put a paid artifact into scrollback for nothing.
        console.print("[red]No such license file.[/red]")
        console.print(
            "[dim]Pass a path to the license file, or pipe the blob on stdin — "
            "'nerdit license install' never takes the blob as an argument "
            "(argv lands in shell history).[/dim]"
        )
        raise typer.Exit(1)
    try:
        blob = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        # ``exc`` names the path the operator typed, nothing else.
        console.print(f"[red]Could not read the license file: {_plain(exc)}[/red]")
        raise typer.Exit(1) from exc
    if not blob:
        console.print("[red]The license file is empty.[/red]")
        raise typer.Exit(1)
    return blob


@license_app.command("install")
def install(
    file: Optional[str] = typer.Argument(  # noqa: UP007 — Typer needs Optional[]
        None,
        metavar="[FILE]",
        help=(
            "Path to the license file, or '-' to read it from stdin. "
            "Omit it to read piped stdin. The blob itself is never accepted "
            "as an argument."
        ),
    ),
) -> None:
    """Install a signed license file on the daemon (admin)."""
    asyncio.run(_install_async(file))


async def _install_async(file: str | None) -> None:
    from nerdit.cli.client import get_configured_client

    blob = _read_blob(file)
    client = get_configured_client()
    try:
        result = await client.install_license(blob, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # The daemon's envelope carries a machine ``reason`` token and never the
        # blob; nothing about the blob may be added here.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    state = result.get("state")
    console.print(
        f"[green]License installed — plan [bold]{_plain(result.get('plan'))}[/bold] "
        f"(lid {_plain(result.get('lid'))}).[/green]"
    )
    console.print(f"  customer    = {_plain(result.get('customer_id'))}")
    console.print(f"  features    = {_features(result.get('features'))}")
    console.print(f"  expires_at  = {_plain(result.get('expires_at'))}")
    console.print(f"  state       = {_plain(state)}")
    if state == "expired_grace":
        console.print(
            "[yellow]This license is past its expiry but inside the grace "
            "period — renew it and reinstall.[/yellow]"
        )
    elif state != "valid":
        console.print(
            "[yellow]This license is installed but not currently valid — "
            "check 'nerdit doctor'.[/yellow]"
        )
    console.print("[dim]No restart needed — the daemon picked it up immediately.[/dim]")


def _features(value: object) -> str:
    """Render the feature list; empty ⇒ a dim `none` (the proxy-table idiom)."""
    if isinstance(value, (list, tuple)):
        return ", ".join(_plain(item) for item in value) if value else "[dim]none[/dim]"
    return _plain(value)


@license_app.command("status")
def status() -> None:
    """Show the persisted license path and what the running daemon verified."""
    asyncio.run(_status_async())


async def _status_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        config_view = await client.get_config("license")
        caps = await client.get_capabilities()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    values = _section_values(config_view)
    console.print("[bold]License (persisted)[/bold]")
    console.print(
        f"  file        = {_plain(values.get('file') or None)}"
        f"{'' if values.get('file') else '  [dim](<data_dir>/license.jws)[/dim]'}"
    )

    live = caps.get("license") if isinstance(caps, dict) else None
    live = live if isinstance(live, dict) else {}
    console.print("[bold]License (live)[/bold]")
    if not live.get("installed"):
        console.print("  installed   = no")
        console.print(
            "[dim]No license installed. This daemon runs the free local product; "
            "install one with: nerdit license install ./license.jws[/dim]"
        )
        return

    state = live.get("state")
    console.print("  installed   = yes")
    console.print(f"  state       = {_plain(state)}")
    if state == "invalid":
        # A machine token from the fixed vocabulary — safe to print verbatim.
        console.print(f"  reason      = {_plain(live.get('reason'))}")
        console.print("[red]The installed license did not verify — reinstall a valid one.[/red]")
        return
    console.print(f"  plan        = {_plain(live.get('plan'))}")
    console.print(f"  features    = {_features(live.get('features'))}")
    console.print(f"  lid         = {_plain(live.get('lid'))}")
    console.print(f"  expires_at  = {_plain(live.get('expires_at'))}")
    # customer_id is admin-only and OMITTED for other roles — print it only when
    # the daemon actually sent it, never a "-" placeholder implying it is unset.
    if "customer_id" in live:
        console.print(f"  customer    = {_plain(live.get('customer_id'))}")
    if state == "expired_grace":
        console.print("[yellow]Expired, inside the grace period — renew and reinstall.[/yellow]")
    elif state == "expired":
        console.print("[red]Expired and past the grace period — renew and reinstall.[/red]")


@license_app.command("remove")
def remove(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Do not prompt for confirmation.",
    ),
) -> None:
    """Delete the installed license from the daemon (admin)."""
    asyncio.run(_remove_async(yes))


async def _remove_async(yes: bool) -> None:
    from nerdit.cli.client import get_configured_client

    # Confirmation before anything is called: the license is a paid artifact and
    # the daemon keeps no copy of it once removed — reinstalling needs the file.
    if not yes and not typer.confirm(
        "Remove the installed license? Reinstalling it later needs the original file.",
        default=False,
    ):
        console.print("[red]Aborted — nothing was changed.[/red]")
        raise typer.Exit(1)

    client = get_configured_client()
    try:
        result = await client.remove_license(idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    if result.get("removed"):
        console.print("[green]License removed.[/green]")
    else:
        # Report what actually happened (the unlink idiom): a double remove is a
        # 200, not an error, and saying "removed" would be a lie.
        console.print("[dim]No license file was installed — nothing to remove.[/dim]")
