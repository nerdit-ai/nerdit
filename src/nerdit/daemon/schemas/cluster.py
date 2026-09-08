"""Response schemas for the dashboard `/cluster/*` surface."""

from __future__ import annotations

from pydantic import BaseModel


class ClusterStats(BaseModel):
    """Aggregated metrics for the dashboard overview."""

    gpus_total: int
    gpus_in_use: int
    gpus_avg_utilization: float
    services_up: int = 0
    daemon_uptime_seconds: int
    daemon_version: str


class ClusterInfo(BaseModel):
    """Daemon metadata displayed in the dashboard."""

    hostname: str
    version: str
    uptime_seconds: int
    # Product-analytics config for the SPA (feat/posthog). Both are null unless
    # `[posthog].enabled` is true AND a `project_key` is set — a daemon with
    # no key ships an inert dashboard. The project key is a publishable
    # client-side token (`phc_…`), safe to emit to the browser.
    posthog_key: str | None = None
    posthog_host: str | None = None
