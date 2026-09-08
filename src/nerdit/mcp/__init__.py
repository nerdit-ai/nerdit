"""Nerdit MCP server — a thin httpx projection of the agent-grade REST API.

This package is import-safe **without** the optional ``mcp`` extra: the pure
tool implementations and the :func:`_call` error normalizer live at module level
in :mod:`nerdit.mcp.server` and only :func:`nerdit.mcp.server.build_server`
imports FastMCP. The CLI entry point (``nerdit mcp``) enforces the extra via an
import firewall before any stdio handshake.
"""
