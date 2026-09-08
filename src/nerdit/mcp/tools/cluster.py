"""Cluster-wide MCP tool implementations (GPUs, aggregate stats).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.transport import _request_client


async def _list_gpus_impl(client: NerditClient) -> Any:
    """Return the daemon's GPU inventory with live metrics."""
    return await _call(client.list_gpus())


async def _cluster_stats_impl(client: NerditClient) -> Any:
    """Return aggregate cluster metrics (GPU counts, services up)."""
    return await _call(client.cluster_stats())


# fmt: off
async def list_gpus() -> Any:
        """List GPUs known to the daemon, with live utilization metrics."""
        return await _list_gpus_impl(_request_client())

async def cluster_stats() -> Any:
        """Get aggregate cluster metrics (GPUs in use/total, services up)."""
        return await _cluster_stats_impl(_request_client())
# fmt: on


TOOLS = (
    list_gpus,
    cluster_stats,
)
