"""Cluster overview endpoints for the dashboard."""

from __future__ import annotations

import socket
from datetime import UTC, datetime

from fastapi import APIRouter, Request

import nerdit
from nerdit.db.models import ClusterInfo, ClusterStats, GpuStatus

router = APIRouter()


def _started_at(request: Request) -> datetime:
    started_at = getattr(request.app.state, "started_at", None)
    if isinstance(started_at, datetime):
        return started_at
    return datetime.now(UTC)


@router.get("/cluster/stats", response_model=ClusterStats, operation_id="get_cluster_stats")
async def cluster_stats(request: Request) -> ClusterStats:
    """Return aggregate daemon, GPU, and service metrics."""
    queries = request.app.state.queries
    gpus = await queries.list_gpus()
    metrics = request.app.state.monitor.get_metrics()
    services_up = await queries.count_services_up()
    online_gpus = [gpu for gpu in gpus if gpu.status != GpuStatus.offline]

    utilization_values = [
        metric.utilization_percent
        for metric in metrics.values()
        if metric is not None and metric.utilization_percent is not None
    ]
    avg_util = sum(utilization_values) / len(utilization_values) if utilization_values else 0.0
    uptime = int((datetime.now(UTC) - _started_at(request)).total_seconds())
    return ClusterStats(
        gpus_total=len(online_gpus),
        gpus_in_use=sum(1 for gpu in online_gpus if gpu.status.value == "busy"),
        gpus_avg_utilization=avg_util,
        services_up=services_up,
        daemon_uptime_seconds=uptime,
        daemon_version=nerdit.__version__,
    )


@router.get("/cluster/info", response_model=ClusterInfo, operation_id="get_cluster_info")
async def cluster_info(request: Request) -> ClusterInfo:
    """Return daemon metadata for the dashboard."""
    settings = request.app.state.settings
    uptime = int((datetime.now(UTC) - _started_at(request)).total_seconds())
    # Emit PostHog config only when analytics are enabled AND a key is set; a
    # daemon with no key ships an inert dashboard (the key is a publishable
    # client-side token, safe to hand the browser).
    posthog = settings.posthog
    posthog_on = posthog.enabled and bool(posthog.project_key)
    return ClusterInfo(
        hostname=socket.gethostname(),
        version=nerdit.__version__,
        uptime_seconds=uptime,
        posthog_key=posthog.project_key if posthog_on else None,
        posthog_host=posthog.host if posthog_on else None,
    )
