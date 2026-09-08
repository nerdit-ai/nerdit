"""Stream resumable service logs from the database.

Poll job_logs by since_id; Docker follow streams would pin one worker thread
per client. Continue through degraded and restarting states. Finish only after
a terminal status and an exhausted cursor, draining pages and taking one settle
re-poll so trailing crash/system lines precede logs.end.

Use a separate daemon-wide semaphore and bounded row pages. Import shared views,
never the services router that includes this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from nerdit.daemon.limits import LOGS_STREAM_CONCURRENCY_MAX, valid_ts_filter
from nerdit.daemon.sse import (
    HEARTBEAT_SECONDS,
    parse_resume_cursor,
    ping_frame,
    saturated_frame,
)
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import JobStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Bounded concurrency, mirroring `EVENTS_STREAM_CONCURRENCY_MAX`: a single
# module-level object (never duplicated — two would double the cap).
_STREAM_SEMAPHORE = asyncio.Semaphore(LOGS_STREAM_CONCURRENCY_MAX)

#: Seconds between polls when the cursor is caught up. Fast enough that a tail
#: feels live, slow enough that 32 idle streams are ~32 keyset reads/s — each
#: one an `idx_job_logs_job` range scan from the connection's cursor, which is
#: why the filtered path must keep that cursor moving (see the watermark below).
_POLL_INTERVAL_S = 1.0

#: Rows per poll. A client attaching to a long-lived service drains the backlog
#: in bounded batches; consecutive non-empty polls skip the sleep, so catching
#: up is fast without ever holding an unbounded result set.
_POLL_CHUNK = 500

#: Statuses at which a service is genuinely finished. `degraded` and
#: `restarting` are deliberately ABSENT: they are transient desired-state
#: churn a follower must tail *through*. Mirrors `_SERVICE_TERMINAL` in
#: `cli/commands/logs.py` — the CLI poll-follower makes the same call.
_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.stopped,
        JobStatus.failed,
        JobStatus.completed,
        JobStatus.cancelled,
    }
)


def _log_frame(entry: Any) -> dict[str, str]:
    """One `job_logs` row → one SSE frame carrying its id as `id:`.

    The id is what makes the stream resumable: a client that drops reconnects
    with `?since_id=<last id>` (or `Last-Event-ID`) and resumes exactly.
    """
    return {
        "id": str(entry.id),
        "event": "log",
        "data": json.dumps(
            {
                "id": entry.id,
                "stream": entry.stream.value,
                "message": entry.message,
                "timestamp": entry.timestamp.isoformat(),
            }
        ),
    }


def _end_frame(status: str) -> dict[str, str]:
    """The terminal frame. Emitted only once the cursor is also exhausted."""
    return {
        "event": "logs.end",
        "data": json.dumps({"type": "logs.end", "status": status}),
    }


async def _drain_to_end(
    request: Request,
    poll: Callable[[int], Awaitable[tuple[list[Any], int]]],
    cursor: int,
    status: str,
) -> AsyncIterator[dict[str, Any]]:
    """Drain terminal-row logs to a short page, then emit logs.end.

    The terminal status may commit after the previous page read. Take one additional
    settle re-poll for the trailing system line; do not shorten its sleep. Later logs
    beyond that window are not covered. Disconnect ends the drain silently.
    """
    settled = False
    while True:
        if await request.is_disconnected():
            return
        entries, cursor = await poll(cursor)
        for entry in entries:
            yield _log_frame(entry)
        if len(entries) == _POLL_CHUNK:
            continue  # full page — more may remain, keep draining
        if settled:
            break
        settled = True
        await asyncio.sleep(_POLL_INTERVAL_S)
    yield _end_frame(status)


async def _log_frames(
    request: Request,
    queries: Any,
    job_id: str,
    *,
    since_id: int,
    grep: str | None,
    since_ts: str | None,
) -> AsyncIterator[dict[str, Any]]:
    """Poll-and-yield frames for one log-stream connection."""
    if _STREAM_SEMAPHORE.locked():
        yield saturated_frame("logs.saturated", LOGS_STREAM_CONCURRENCY_MAX)
        return

    filtered = bool(grep) or bool(since_ts)

    async def poll(cursor: int) -> tuple[list[Any], int]:
        """One bounded page from `cursor`, plus the cursor it leaves behind.

        WATERMARK, captured BEFORE the scan and only when a filter is active. A
        filter that matches nothing yields no row to advance the cursor with, so
        every poll would re-scan the whole remaining range — bounded rows
        returned, unbounded rows scanned, on the shared connection. Capturing the
        max id first is what makes skipping safe: anything written after this
        read gets a strictly larger id.
        """
        pre_max = await queries.max_log_id() if filtered else 0
        entries = await queries.get_logs(
            job_id, since_id=cursor, grep=grep, since_ts=since_ts, limit=_POLL_CHUNK
        )
        if entries:
            cursor = entries[-1].id
        if filtered and len(entries) < _POLL_CHUNK:
            # Short page ⇒ the scan reached the end of the range it was given,
            # so everything up to the pre-scan watermark is decided.
            cursor = max(cursor, pre_max)
        return entries, cursor

    async with _STREAM_SEMAPHORE:
        cursor = since_id
        idle_s = 0.0
        while True:
            if await request.is_disconnected():
                return

            entries, cursor = await poll(cursor)
            for entry in entries:
                yield _log_frame(entry)
            if entries:
                idle_s = 0.0
                continue  # more may be buffered — drain before sleeping

            # The row is read ONLY on an empty page — the hot path (a service
            # that is actually talking) costs one query per poll, not two. The
            # status read exists to decide termination, and termination can only
            # ever be decided on an empty page.
            job = await queries.get_job(job_id)
            if job is None:
                yield _end_frame("deleted")
                return
            if job.status in _TERMINAL_STATUSES:
                async for frame in _drain_to_end(request, poll, cursor, job.status.value):
                    yield frame
                return

            if idle_s >= HEARTBEAT_SECONDS:
                yield ping_frame()
                idle_s = 0.0
            await asyncio.sleep(_POLL_INTERVAL_S)
            idle_s += _POLL_INTERVAL_S


@router.get(
    "/services/{ident}/logs/stream",
    operation_id="stream_service_logs",
    tags=["Services"],
)
async def stream_service_logs(
    request: Request,
    ident: str,
    since_id: int = Query(
        0,
        description=(
            "Resume cursor: only lines with a higher id are sent. A Last-Event-ID "
            "header, if present, overrides this."
        ),
    ),
    grep: str | None = Query(
        None,
        description=(
            "Keep only lines CONTAINING this text. A literal substring, not a regex — "
            "'%' matches a percent sign. Clamped to 200 characters."
        ),
    ),
    since: str | None = Query(
        None, description="Only lines at or after this ISO-8601 UTC timestamp (inclusive)"
    ),
):
    """Stream logs to any authenticated principal with resumable row IDs.

    Resume through Last-Event-ID or since_id; grep/since filter in the bounded SQL
    query. Continue during degraded/restarting states, ending with logs.end only
    after terminal status and complete drain. At capacity emit logs.saturated and
    close.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    since_ts = valid_ts_filter(since, "since") if since is not None else None
    # `Last-Event-ID` wins over `?since_id=`: a browser `EventSource`
    # resends the last id it saw by itself and cannot be told to drop it. An
    # unusable header value falls back to `since_id` rather than 400-ing a
    # reconnect (the one shared parser owns that rule — and its 64-bit bound).
    resume = parse_resume_cursor(request)
    return EventSourceResponse(
        _log_frames(
            request,
            queries,
            job.id,
            since_id=since_id if resume is None else resume,
            grep=grep,
            since_ts=since_ts,
        )
    )
