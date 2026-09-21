"""MCP tool implementations, split by domain (Track B WP24).

Each sibling module holds the pure, testable ``_*_impl`` coroutines for one
domain (cluster, services, models, databases, deploy, secrets, config,
system) plus the public tool functions themselves (``list_gpus``, ``deploy``,
...) that ``@mcp.tool()`` registers — FastMCP derives each tool's description
from the function's docstring and its ``inputSchema`` from the signature, so
both are byte-identical to the nested defs this collapse replaces (W24.3).
Each domain module ends with an ordered ``TOOLS`` tuple; :data:`ALL_TOOLS`
here concatenates them (domain-grouped order) for :func:`nerdit.mcp.server.
build_server` to register in one loop.
"""

from __future__ import annotations

from nerdit.mcp.tools import (
    cluster,
    config,
    databases,
    deploy,
    exposure,
    models,
    projects,
    secrets,
    services,
    system,
    workspaces,
)

ALL_TOOLS = (
    *cluster.TOOLS,
    *services.TOOLS,
    *models.TOOLS,
    *databases.TOOLS,
    *deploy.TOOLS,
    *workspaces.TOOLS,
    *exposure.TOOLS,
    *projects.TOOLS,
    *secrets.TOOLS,
    *config.TOOLS,
    *system.TOOLS,
)
