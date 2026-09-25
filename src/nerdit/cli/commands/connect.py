"""nerdit connect — Connect CLI to a remote nerditd daemon."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from nerdit.cli.display import console, render_client_error
from nerdit.config.defaults import DEFAULT_PORT
from nerdit.utils.fs import atomic_write


def connect(
    host: str = typer.Argument(..., help="Server IP address or hostname"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Daemon port"),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help=(
            "Authentication token. Avoid: argv is visible in the process list and "
            "shell history; omit it to read the token from a hidden prompt or stdin."
        ),
    ),
) -> None:
    """Connect the CLI to a remote daemon."""
    if token is None:
        token = _read_token()
    if not token:
        console.print("[red]No authentication token entered.[/red]")
        raise typer.Exit(1)
    asyncio.run(_connect_async(host, port, token))


def _read_token() -> str:
    """Read the token from a hidden terminal prompt, or one line of piped stdin."""
    if sys.stdin is not None and sys.stdin.isatty():
        return str(typer.prompt("Authentication token", hide_input=True)).strip()
    line = sys.stdin.readline() if sys.stdin is not None else ""
    return line.strip()


async def _connect_async(host: str, port: int, token: str) -> None:
    config_path = Path("~/.nerdit/config.toml").expanduser()

    # Read existing config or start fresh
    if config_path.exists():
        import tomllib

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    else:
        data = {}

    # Update [client] section
    data["client"] = {
        "remote_host": host,
        "remote_port": port,
        "auth_token": token,
    }

    # Write back
    import tomli_w

    config_path.parent.mkdir(parents=True, exist_ok=True)
    # 0600 and atomic: the file carries the auth token.
    atomic_write(config_path, tomli_w.dumps(data).encode("utf-8"))

    console.print("[green]Client configuration saved[/green]")
    console.print(f"  Server: {host}:{port}")
    if host.strip().strip("[]") not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            "[yellow]This connection uses plain HTTP: the token crosses the network "
            "unencrypted. Use it only on a trusted network or through an SSH tunnel/VPN."
            "[/yellow]"
        )

    # Test connection
    from nerdit.cli.client import NerditClient

    client = NerditClient(host=host, port=port, token=token)
    try:
        health = await client.health()
        console.print(
            f"[green]Connection successful[/green] — version {health['version']}, "
            f"{health['gpu_count']} GPU(s)"
        )
    except Exception as exc:
        render_client_error(exc)
        raise typer.Exit(1)
