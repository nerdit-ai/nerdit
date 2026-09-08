"""Schemas for live service resource usage.

Unknown is null, never zero. Missing containers, unavailable Docker and failed
samples return available=false and stats=null, distinct from a live idle app.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContainerStatsView(BaseModel):
    """One point-in-time container resource sample (every counter optional)."""

    cpu_pct: float | None = Field(
        default=None,
        description=(
            "CPU percent, docker-CLI convention: 100.0 = one saturated core, so an "
            "N-core host can report up to N*100. null when the delta is not derivable "
            "(first sample / missing counters) — never 0.0, which would read as idle"
        ),
    )
    mem_used_bytes: int | None = Field(
        default=None, description="Working-set bytes (page cache excluded), or null"
    )
    mem_limit_bytes: int | None = Field(
        default=None,
        description=(
            "Memory limit in bytes — the [deploy].memory_limit when set, else the "
            "host total as docker reports it. null when unknown"
        ),
    )
    mem_pct: float | None = Field(
        default=None,
        description="mem_used_bytes / mem_limit_bytes as a percent; null when either is unknown",
    )
    net_rx_bytes: int | None = Field(
        default=None,
        description="Bytes received, summed over attached interfaces (null on host networking)",
    )
    net_tx_bytes: int | None = Field(
        default=None, description="Bytes transmitted, summed over attached interfaces"
    )
    pids: int | None = Field(
        default=None, description="Processes/threads in the container, or null"
    )


class GpuStatsView(BaseModel):
    """Per-GPU utilization joined from the monitor's EXISTING snapshot.

    No device is probed to answer this read (D-P24-10): the values come from
    the same in-memory `app.state.monitor` sample that backs `GET /gpus`,
    so a GPU the monitor has not sampled yet reports `null` rather than
    blocking the request on a driver call.
    """

    gpu_id: str = Field(description="GPU identifier as allocated to this workload")
    utilization_percent: int | None = Field(
        default=None, description="Live GPU utilization percent, or null when unsampled"
    )
    memory_used_mb: int | None = Field(
        default=None, description="Live GPU memory used (MB), or null when unsampled"
    )


class ServiceStatsResponse(BaseModel):
    """Live resource usage, readable by any authenticated principal.

    This contains numeric counters and a service name, not diagnose's confidential
    logs or env names. Available=false with stats=null means unavailable, not idle.
    """

    service_name: str = Field(description="Stable service name")
    container_id: str | None = Field(
        default=None, description="Container the sample was taken from (null when none is running)"
    )
    available: bool = Field(
        description="Whether a sample could be taken at all; false ⇒ stats is null"
    )
    stats: ContainerStatsView | None = Field(
        default=None, description="The sample, or null when available is false"
    )
    gpus: list[GpuStatsView] = Field(
        default_factory=list,
        description="Per-GPU utilization for the GPUs allocated to this workload (may be empty)",
    )
    sampled_at: str | None = Field(
        default=None, description="ISO-8601 timestamp of the sample (null when unavailable)"
    )
    cached: bool = Field(
        default=False,
        description=(
            "Whether this sample was served from the short TTL cache rather than freshly "
            "collected — sampling blocks ~1-2s per container, so repeated reads coalesce"
        ),
    )
