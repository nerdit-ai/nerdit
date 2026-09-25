"""Run the bundled MCP server, reporting incomplete installations before stdio starts."""

from __future__ import annotations

import importlib.util

import typer


def mcp() -> None:
    """Run the Nerdit MCP server over stdio."""
    if importlib.util.find_spec("mcp") is None:
        typer.echo(
            "The bundled MCP dependency is missing; this installation is incomplete.\n"
            "Repair it with:  python -m pip install --upgrade nerdit",
            err=True,
        )
        raise SystemExit(1)

    from nerdit.mcp.server import run

    run()
