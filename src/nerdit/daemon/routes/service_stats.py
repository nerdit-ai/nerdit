"""Sample service resource usage on demand, outside the reconcile loop.

Docker sampling may take 1–2 seconds. A short container-ID cache and per-container
single-flight coalesce repeated and concurrent reads. Unavailable samples return
available=false, stats=null; never invent zero counters. Import shared views,
not the services router that includes this module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request

from nerdit.core.runtime.protocol import ContainerStats
from nerdit.daemon.schemas.service_stats import (
    ContainerStatsView,
    GpuStatsView,
    ServiceStatsResponse,
)
from nerdit.daemon.views.service import _not_found, _resolve_service

logger = logging.getLogger(__name__)

router = APIRouter()

#: Cache window, seconds. Short enough that a dashboard refresh still shows
#: movement, long enough that a burst of agent polls collapses to one docker
#: call. Deliberately not configurable: it is a coalescing detail, not policy.
_STATS_TTL_S = 2.0

#: `app.state` attribute holding the cache. Kept on the app (not a module
#: global) so parallel test apps never share entries and the cache dies with
#: the daemon process it belongs to.
_CACHE_ATTR = "service_stats_cache"

#: `app.state` attribute holding the in-flight sample per container id. Same
#: reasoning as the cache: per-app, never a module global.
_INFLIGHT_ATTR = "service_stats_inflight"

#: Hard bound on retained cache entries. Expired rows are dropped on every
#: read, so the steady-state size is "containers sampled in the last 2 s"; this
#: is the belt-and-braces cap in case a pathological caller cycles ids faster
#: than they expire.
_CACHE_MAX_ENTRIES = 256


#: One cache entry: `(monotonic_deadline, sample, sampled_at_iso)`. The ISO
#: stamp is the *sample's* wall clock, captured when the sample was taken —
#: a cache hit must report when the numbers were measured, not when the
#: response happened to be assembled.
_CacheEntry = tuple[float, ContainerStats | None, str | None]


def _cache(request: Request) -> dict[str, _CacheEntry]:
    """The per-app `{container_id: (deadline, sample, sampled_at)}` cache."""
    state = request.app.state
    cache = getattr(state, _CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(state, _CACHE_ATTR, cache)
    return cache


def _inflight(request: Request) -> dict[str, asyncio.Task[tuple[ContainerStats | None, str]]]:
    """The per-app `{container_id: in-flight sample task}` map."""
    state = request.app.state
    flight = getattr(state, _INFLIGHT_ATTR, None)
    if not isinstance(flight, dict):
        flight = {}
        setattr(state, _INFLIGHT_ATTR, flight)
    return flight


def _cache_get(
    cache: dict[str, _CacheEntry], container_id: str, now: float
) -> tuple[bool, ContainerStats | None, str | None]:
    """`(hit, sample, sampled_at)` — expired rows are pruned as a side effect.

    A `None` *sample* is cached exactly like a real one: an unreadable
    container is the case most likely to be polled hardest (an agent watching a
    crash loop), and re-asking docker every time would be the expensive answer
    to a question that just failed.
    """
    for key, entry in list(cache.items()):
        if entry[0] <= now:
            cache.pop(key, None)
    hit = cache.get(container_id)
    if hit is None:
        return False, None, None
    return True, hit[1], hit[2]


def _cache_put(
    cache: dict[str, _CacheEntry],
    container_id: str,
    sample: ContainerStats | None,
    now: float,
    sampled_at: str | None,
) -> None:
    if len(cache) >= _CACHE_MAX_ENTRIES and container_id not in cache:
        cache.clear()
    cache[container_id] = (now + _STATS_TTL_S, sample, sampled_at)


def _stats_view(sample: ContainerStats) -> ContainerStatsView:
    """Project a runtime sample, deriving `mem_pct` only when both sides exist."""
    mem_pct: float | None = None
    used = sample.mem_used_bytes
    limit = sample.mem_limit_bytes
    if used is not None and limit is not None and limit > 0:
        mem_pct = round(used / limit * 100.0, 2)
    return ContainerStatsView(
        cpu_pct=sample.cpu_pct,
        mem_used_bytes=used,
        mem_limit_bytes=limit,
        mem_pct=mem_pct,
        net_rx_bytes=sample.net_rx_bytes,
        net_tx_bytes=sample.net_tx_bytes,
        pids=sample.pids,
    )


async def _sample_and_cache(
    request: Request, container_id: str, name: str
) -> tuple[ContainerStats | None, str]:
    """Take ONE docker sample, cache it, and release the in-flight slot.

    The `finally` drops the in-flight entry *after* the cache is populated, so
    a request arriving in that instant finds the fresh entry rather than opening
    a second sample. It also runs on cancellation, so a client walking away can
    never leave a permanently-claimed container id behind.
    """
    try:
        runtime = getattr(request.app.state, "runtime", None)
        sample: ContainerStats | None = None
        if runtime is not None:
            try:
                sample = await runtime.stats(container_id)
            except Exception:  # noqa: BLE001 — the Protocol swallows; belt and braces
                logger.debug("stats sample failed for %s", name, exc_info=True)
                sample = None
        sampled_at = datetime.now(timezone.utc).isoformat()
        # The deadline is stamped from a FRESH monotonic read: the call above
        # blocks ~1-2 s collecting two CPU samples, so an entry dated from
        # before it would already be most of the way expired.
        _cache_put(_cache(request), container_id, sample, time.monotonic(), sampled_at)
        return sample, sampled_at
    finally:
        _inflight(request).pop(container_id, None)


async def _sample(
    request: Request, container_id: str, name: str
) -> tuple[ContainerStats | None, str]:
    """The single-flight entry point: sample once per container, share the result.

    N concurrent misses for the same container used to mean N blocking docker
    calls for an answer that is identical by construction. The first caller owns
    the sample; the rest await it. Both sides `shield` so one client
    disconnecting mid-sample does not cancel the work every other waiter is
    parked on.
    """
    flight = _inflight(request)
    task = flight.get(container_id)
    if task is None:
        task = asyncio.ensure_future(_sample_and_cache(request, container_id, name))
        flight[container_id] = task
    return await asyncio.shield(task)


async def _gpu_views(request: Request, job_id: str) -> list[GpuStatsView]:
    """Join per-GPU utilization from the monitor's EXISTING snapshot (no new probe).

    Same source as `GET /gpus` and the model list projection
    (`app.state.monitor`). `None`-safe on both axes: a missing/unwired
    monitor and a GPU the monitor has not sampled yet both project as `null`
    fields rather than failing the read.
    """
    try:
        gpu_ids = await request.app.state.queries.get_job_gpus(job_id)
    except Exception:  # noqa: BLE001 — a projection must never fail the stats read
        return []
    if not gpu_ids:
        return []
    monitor = getattr(request.app.state, "monitor", None)
    try:
        metrics = monitor.get_metrics() if monitor is not None else {}
    except Exception:  # noqa: BLE001 — defensive; a read must never 500 here
        metrics = {}
    views: list[GpuStatsView] = []
    for gpu_id in gpu_ids:
        m = metrics.get(gpu_id) if isinstance(metrics, dict) else None
        views.append(
            GpuStatsView(
                gpu_id=str(gpu_id),
                utilization_percent=getattr(m, "utilization_percent", None),
                memory_used_mb=getattr(m, "memory_used_mb", None),
            )
        )
    return views


@router.get(
    "/services/{ident}/stats",
    response_model=ServiceStatsResponse,
    operation_id="get_service_stats",
    tags=["Services"],
)
async def get_service_stats(request: Request, ident: str) -> ServiceStatsResponse:
    """Return live resource counters to any authenticated principal.

    No ownership gate, audit or Idempotency-Key: this contains no confidential logs
    or env names. Missing/unreadable containers return available=false, stats=null.
    A two-second cache avoids repeating slow Docker samples on polling reads.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)

    name = job.service_name or job.name or job.id
    gpus = await _gpu_views(request, job.id)
    container_id = job.container_id

    if not container_id:
        return ServiceStatsResponse(
            service_name=name, container_id=None, available=False, stats=None, gpus=gpus
        )

    cache = _cache(request)
    hit, sample, sampled_at = _cache_get(cache, container_id, time.monotonic())
    if not hit:
        sample, sampled_at = await _sample(request, container_id, name)

    if sample is None:
        return ServiceStatsResponse(
            service_name=name,
            container_id=container_id,
            available=False,
            stats=None,
            gpus=gpus,
            cached=hit,
        )
    return ServiceStatsResponse(
        service_name=name,
        container_id=container_id,
        available=True,
        stats=_stats_view(sample),
        gpus=gpus,
        # The SAMPLE's wall clock, replayed verbatim on a cache hit: stamping
        # "now" would describe the response, not the measurement.
        sampled_at=sampled_at,
        cached=hit,
    )
