"""Shared EventBus-backed SSE streaming with keepalives and disconnect cleanup.

The subscription predicate excludes events before enqueueing. Frame mappers
may return one frame, None, or asynchronously a list of frames. The preamble
runs after subscribing so backlog reads cannot leave a gap in the live feed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

from starlette.requests import Request

logger = logging.getLogger(__name__)

#: Heartbeat interval used to keep proxies from closing idle connections.
HEARTBEAT_SECONDS = 15.0

#: One event dict -> one SSE frame dict, or `None` to drop the event.
FrameMapper = Callable[[dict[str, Any]], dict[str, Any] | None]

#: One event dict -> **every** frame it expands to; `[]` drops it. The async,
#: fan-out form of `FrameMapper`, for a stream whose mapper has to read
#: the DB before it can answer (the durable feed's catch-up leg).
MultiFrameMapper = Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]

#: `preamble()` -> frames to emit before the live tail starts.
Preamble = Callable[[], AsyncIterator[dict[str, Any]]]


def ping_frame() -> dict[str, str]:
    """The keep-alive frame. A fresh dict per call — never a shared constant."""
    return {"event": "ping", "data": "{}"}


def parse_resume_cursor(request: Request, *, query_param: str | None = None) -> int | None:
    """Read Last-Event-ID, falling back to the named query parameter.

    Malformed, negative or out-of-signed-64-bit-range cursors return None to start
    live. The range protects SQLite binding inside streaming generators.
    """
    raw = request.headers.get("last-event-id")
    if not raw and query_param is not None:
        raw = request.query_params.get(query_param)
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.debug("Ignoring malformed resume cursor %r", raw)
        return None
    if not (0 <= value < 2**63):
        logger.debug("Ignoring out-of-range resume cursor %r", raw)
        return None
    return value


def saturated_frame(event_name: str, limit: int) -> dict[str, str]:
    """The "daemon is at its stream budget" frame — agent-branchable, then close.

    Refusing loudly beats accepting a connection the daemon cannot serve: an SSE
    connection is held open indefinitely, so an uncapped stream route is a cheap
    connection-exhaustion vector on a LAN daemon. Each route keeps its own
    semaphore and its own `event_name` (`feed.saturated` / `logs.saturated`
    — two independent resources); only the frame *shape* is shared.
    """
    return {
        "event": event_name,
        "data": json.dumps({"type": event_name, "reason": "concurrency_limit", "limit": limit}),
    }


def default_frame(event: dict[str, Any]) -> dict[str, Any]:
    """The historical mapping: event type as the SSE event name, JSON body."""
    return {"event": str(event.get("type", "event")), "data": json.dumps(event)}


async def heartbeat_only(
    request: Request, *, heartbeat_s: float = HEARTBEAT_SECONDS
) -> AsyncIterator[dict[str, Any]]:
    """Frames for a daemon with no event bus (tests / dev).

    Heartbeats only, so a client holds the connection instead of
    reconnect-looping against a route that will never speak.
    """
    while True:
        if await request.is_disconnected():
            return
        yield ping_frame()
        await asyncio.sleep(heartbeat_s)


def _as_multi(frame: FrameMapper | MultiFrameMapper) -> MultiFrameMapper:
    """Normalize either mapper shape into the async, list-returning one.

    The sync single-frame form is the historical contract and stays exactly
    that — one call per event, `None` drops — so the audit stream and the log
    stream keep their behaviour byte for byte. A mapper declared `async def`
    is taken as the fan-out form and used as-is. The loop below never has to
    know which shape it was handed, which is the point: the alternative was an
    `isinstance`/`isawaitable` ladder inside the hot path, re-evaluated for
    every event on every connection.
    """
    if inspect.iscoroutinefunction(frame):
        return cast("MultiFrameMapper", frame)

    single = cast("FrameMapper", frame)

    async def _one(event: dict[str, Any]) -> list[dict[str, Any]]:
        mapped = single(event)
        return [] if mapped is None else [mapped]

    return _one


async def sse_from_bus(
    request: Request,
    event_bus: Any,
    *,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
    frame: FrameMapper | MultiFrameMapper = default_frame,
    preamble: Preamble | None = None,
    heartbeat_s: float = HEARTBEAT_SECONDS,
) -> AsyncIterator[dict[str, Any]]:
    """Subscribe to the bus and yield SSE frames until the client goes away.

    The subscription is opened **first**, before `preamble` runs, and closed
    in a `finally` whatever ends the loop (disconnect, cancellation, or the
    caller closing the generator).
    """
    emit = _as_multi(frame)
    subscription = event_bus.subscribe(predicate)
    try:
        if preamble is not None:
            async for pre in preamble():
                yield pre
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(subscription.__anext__(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                yield ping_frame()
                continue
            for mapped in await emit(event):
                yield mapped
    finally:
        await subscription.aclose()
