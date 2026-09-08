"""nerdit connect — Connect CLI to a remote nerditd daemon."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from nerdit.cli.display import console, render_client_error
from nerdit.config.defaults import DEFAULT_PORT


def connect(
    host: str = typer.Argument(..., help="Server IP address or hostname"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Daemon port"),
    token: str = typer.Option(..., "--token", "-t", help="Authentication token"),
) -> None:
    """Connect the CLI to a remote daemon."""
    asyncio.run(_connect_async(host, port, token))


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
    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)

    console.print("[green]Client configuration saved[/green]")
    console.print(f"  Server: {host}:{port}")

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
