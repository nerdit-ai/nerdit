"""Coerce persisted health settings and run consistent reconcile/diagnostic probes."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import httpx

# Defaults for the per-service health-check spec (JSON in `jobs.health_check`).
DEFAULT_HEALTH_PATH = "/"
DEFAULT_HEALTH_TIMEOUT_S = 2.0
DEFAULT_UNHEALTHY_THRESHOLD = 3
DEFAULT_START_PERIOD_S = 0.0

HttpCheck = Callable[[int, str, float], Awaitable[int | None]]
TcpCheck = Callable[[int, float], Awaitable[bool]]


def as_float(value: object, default: float) -> float:
    """Coerce a loosely-typed health-check value to float, falling back on error."""
    if value is None:
        return default
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int) -> int:
    """Coerce a loosely-typed health-check value to int, falling back on error."""
    if value is None:
        return default
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


async def check_health(
    host_port: int, path: str, timeout: float, *, client: httpx.AsyncClient | None = None
) -> int | None:
    """Probe `http://127.0.0.1:host_port{path}` with a strict per-check timeout.

    Returns the HTTP status code, or `None` on any connection/timeout error
    (CRIT-5: the timeout bounds the loop cost of a hanging endpoint). Shared by
    the reconcile health loop and the `/diagnose` fresh probe. `client` reuses
    a caller-owned client; without one a client is built for this call.
    """
    if not path.startswith("/"):
        path = "/" + path
    url = f"http://127.0.0.1:{host_port}{path}"
    try:
        if client is not None:
            resp = await client.get(url, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as own:
                resp = await own.get(url)
        return resp.status_code
    except Exception:
        return None


async def check_tcp(host_port: int, timeout: float) -> bool:
    """Probe raw TCP connectivity to `127.0.0.1:host_port` (P14 WP-C1).

    Returns `True` when a connection is established within `timeout`, `False`
    on any connect/timeout error. The reconcile loop synthesizes an HTTP `200`
    from a `True` result so the shared degraded/threshold/start-period machinery
    runs unchanged. Shared by the health loop and the `/diagnose` fresh probe.
    """
    writer = None
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", host_port), timeout=timeout
        )
        return True
    except Exception:
        return False
    finally:
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


def within_start_period(
    hc: dict | None,
    started_at: datetime | None,
    *,
    now: datetime | None = None,
) -> bool:
    """True when a health-checked row is still inside its start period.

    Tolerant coercion of `start_period_s` via `as_float` — the same
    §1.5-item-1 coercion-alignment family as `timeout_s`: a string value like
    `"3600"` now resolves the same way here as it does for the probe timeout,
    instead of silently degrading to the `0.0` default. A naive (tz-unaware)
    `started_at` is normalized to UTC before the elapsed-time comparison.
    """
    if not hc:
        return False
    if started_at is None:
        return False
    period_s = as_float(hc.get("start_period_s"), DEFAULT_START_PERIOD_S)
    if period_s <= 0:
        return False
    started = started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    elapsed = (reference - started).total_seconds()
    return elapsed < period_s


async def run_probe(
    hc: dict,
    host_port: int,
    timeout: float,
    *,
    http_check: HttpCheck,
    tcp_check: TcpCheck,
) -> tuple[int | None, str]:
    """Dispatch a single health probe on `hc["type"]` (P14 WP-C1).

    A junk/absent `type` falls through to http. A successful tcp connect
    synthesizes HTTP code `200` so a caller's shared degraded/threshold
    machinery stays oblivious to the probe kind. The probe callables are
    parameters (not module-level lookups) so both call sites keep their own
    monkeypatch seams. Returns `(status_code, kind)` where `kind` is the
    probe actually run (`"tcp"` or `"http"`), never the raw (possibly
    junk) blob value.
    """
    if str(hc.get("type") or "http") == "tcp":
        code = 200 if await tcp_check(host_port, timeout) else None
        return code, "tcp"
    path = str(hc.get("path") or DEFAULT_HEALTH_PATH)
    code = await http_check(host_port, path, timeout)
    return code, "http"
