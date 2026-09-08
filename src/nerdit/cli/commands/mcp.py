"""nerdit mcp — run the Model Context Protocol server (P1 / S10).

The base package stays MCP-free; this command only becomes usable with the
optional ``mcp`` extra. An import firewall checks for the dependency *before*
any stdio handshake so a missing extra fails loudly and actionably instead of
emitting a broken MCP transport an agent would hang on.
"""

from __future__ import annotations

import importlib.util

import typer


def mcp() -> None:
    """Run the Nerdit MCP server over stdio (requires the ``mcp`` extra)."""
    if importlib.util.find_spec("mcp") is None:
        typer.echo(
            "The MCP server requires the optional 'mcp' extra, which is not installed.\n"
            "Install it with:  pip install 'nerdit[mcp]'",
            err=True,
        )
        raise SystemExit(1)

    from nerdit.mcp.server import run

    run()
