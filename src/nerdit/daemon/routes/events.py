"""Read and stream the durable cluster event feed for authenticated callers.

Cursor pages backward (id < cursor, newest first); since_id replays forward
(id > since_id, oldest first). Supplying both returns 400 bad_request.

Streams replay Last-Event-ID before live events. Durable frames carry row IDs;
legacy bus-only frames are live-only. Inexact replay emits feed.gap with pruned
or overflow, telling clients to reconcile through GET /events?since_id=….
The live tail fills sequence gaps from the DB, covering bounded-queue drops and
out-of-order bus publication without inventing data loss.

Never expose audit events: the durable table excludes them and the bus predicate
filters them before enqueueing. Audit reads have a separate admin gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from nerdit.daemon.errors import NerditError
from nerdit.daemon.limits import _MAX_SSE_REPLAY, EVENTS_STREAM_CONCURRENCY_MAX
from nerdit.daemon.schemas.events import EventPage
from nerdit.daemon.sse import (
    default_frame,
    heartbeat_only,
    parse_resume_cursor,
    saturated_frame,
    sse_from_bus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Bounded concurrency, mirroring `WAIT_CONCURRENCY_MAX`: a single module-level
# object (never duplicated — two would double the cap).
_STREAM_SEMAPHORE = asyncio.Semaphore(EVENTS_STREAM_CONCURRENCY_MAX)

#: Bound on ONE live catch-up read (the hole a dropped subscriber queue left).
#: Deliberately smaller than `_MAX_SSE_REPLAY`: that bound governs a
#: once-per-connection preamble, this one a read a *single live frame* can
#: trigger. Comfortably above the bus subscriber queue's own depth, so the
#: ordinary "slow consumer, full queue" burst is recovered whole; anything
#: larger is pathological and gets the honest `feed.gap` answer instead.
_MAX_CATCHUP = 200


def _not_audit(event: dict[str, Any]) -> bool:
    """Exclude admin-only `audit.*` events from the general stream.

    They are served separately through the admin-gated `/api/audit/stream`.
    This GET route is reachable by readonly/submitter tokens, so filtering at
    the subscription source is the least-privilege guard (M2).
    """
    return not str(event.get("type", "")).startswith("audit.")


def _split_types(types: str | None) -> list[str] | None:
    """Parse the comma-separated `types` filter into a list.

    Splitting and stripping only: the **bound** on how many entries are honoured
    belongs to the query layer (`_MAX_TYPE_FILTERS`), which clamps every
    caller — including one reaching it without going through this route — rather
    than to a second clamp here that could only ever drift from it. Over-long
    lists are clamped, never rejected (D-P24-12), the same posture as `limit`.
    """
    if not types:
        return None
    parsed = [part.strip() for part in types.split(",") if part.strip()]
    return parsed or None


@router.get("/events", response_model=EventPage, operation_id="list_events")
async def list_events(
    request: Request,
    limit: int = Query(50, description="Max rows per page. Clamped to [1, 200] — never rejected."),
    cursor: str | None = Query(
        None,
        description=(
            "Browse mode: cursor browses backwards, id DESC (rows with id < cursor, "
            "newest first). Pass the previous page's next_cursor. Mutually exclusive "
            "with since_id."
        ),
    ),
    types: str | None = Query(
        None,
        description=(
            "Comma-separated event types to keep (e.g. service.failed,model.ready). "
            "At most 10 entries are honoured."
        ),
    ),
    service: str | None = Query(None, description="Filter by exact service name."),
    since_id: int | None = Query(
        None,
        description=(
            "Reconcile mode: since_id replays forwards, id ASC (rows with id > since_id, "
            "oldest first) — the mode a webhook consumer or a resuming agent uses. Feed "
            "back the returned next_cursor to page forward. Mutually exclusive with cursor."
        ),
    ),
) -> EventPage:
    """Bounded page of the durable event feed, in one of two directions.

    `cursor` browses backwards (id DESC, `id < cursor`) — the dashboard
    read. `since_id` replays forwards (id ASC, `id > since_id`) — the
    consumer read: a durable cursor reconciles forward, never into the past.
    Supplying both is `400 bad_request`. Read is open to any authenticated
    principal; `audit.*` is never in this table.
    """
    queries = request.app.state.queries
    try:
        items, next_cursor = await queries.list_events(
            limit=limit,
            cursor=cursor,
            types=_split_types(types),
            service=service,
            since_id=since_id,
        )
    except (ValueError, OverflowError) as exc:
        # OverflowError: sqlite binds 64-bit integers, so a since_id/cursor
        # beyond that range is a client error (400), never a 500.
        raise NerditError(400, "bad_request", str(exc) or "Value out of range") from exc
    return EventPage(items=items, next_cursor=next_cursor)


@router.get("/events/stream", operation_id="stream_cluster_events")
async def stream_cluster_events(request: Request):
    """Subscribe to cluster events as Server-Sent Events, resumably.

    Every durable frame carries its row id as the SSE `id:` field, so a
    reconnect with `Last-Event-ID` (header; `?last_event_id=` accepted as a
    fallback for clients that cannot set one) replays what was missed. Legacy
    `job.status_changed` frames keep flowing with **no** id — they are
    bus-only and never replayed. A resume that cannot be served exactly gets a
    `feed.gap` frame first; a saturated daemon gets one `feed.saturated`
    frame and the stream closes.
    """
    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus is None:
        # Surfaces in tests / dev without a bus — keep the connection alive
        # with heartbeats only so clients don't reconnect-loop.
        return EventSourceResponse(heartbeat_only(request))

    queries = getattr(request.app.state, "queries", None)
    resume = parse_resume_cursor(request, query_param="last_event_id")
    return EventSourceResponse(_event_frames(request, event_bus, queries, resume))


def _gap_frame(*, from_id: int, reason: str, retained_min_id: int) -> dict[str, str]:
    """The synthetic "your replay is not exact" frame.

    D-P24-3-clean by construction: two ints and one pinned literal, no free
    text. `reason` tells a consumer *why* it was cut and `retained_min_id`
    where the feed now starts.
    """
    return {
        "event": "feed.gap",
        "data": json.dumps(
            {
                "type": "feed.gap",
                "from": from_id,
                "reason": reason,
                "retained_min_id": retained_min_id,
            }
        ),
    }


def _row_frame(row: Any) -> dict[str, Any]:
    """One durable row -> its SSE frame, id included.

    Shared by the replay preamble and the live catch-up leg: both emit rows
    read straight from the table, and two copies of this projection could
    disagree about the `id:` line a client resumes from.
    """
    return {
        "id": str(row.id),
        "event": row.type,
        "data": json.dumps(row.model_dump(mode="json")),
    }


async def _catch_up_frames(
    queries: Any, event: dict[str, Any], rid: int, seen: int
) -> tuple[list[dict[str, Any]], int]:
    """Frames for a live id that is NOT contiguous with what was delivered.

    Reached only when `rid > seen + 1`, i.e. the bounded subscriber queue
    dropped something (or, rarely, two publishes crossed). The missing rows are
    read rather than guessed at — see the module docstring for why that read is
    exact — so the client receives every id once, in order, and the `feed.gap`
    escape hatch stays reserved for the one case a read cannot fix.

    Returns the frames to emit and the new watermark.
    """
    rows = await queries.replay_events_after(seen, limit=_MAX_CATCHUP + 1)
    if len(rows) > _MAX_CATCHUP:
        # Pathological: more rows are missing than one live frame is allowed to
        # drag off disk. Say so once and jump the watermark to the newest row
        # this bounded read did return, so the tail resumes from a real id
        # instead of re-reading the same hole on every subsequent frame.
        # `retained_min_id` is the OLDEST row the read returned, never the
        # watermark: the skipped rows are still on disk, and the CLI's
        # reconcile hint (`--since retained_min_id - 1`) is contractually
        # one below the oldest surviving row. A non-contiguous first row means
        # retention already ate part of the hole — then the cause is pruning.
        reason = "pruned" if rows[0].id > seen + 1 else "overflow"
        gap = _gap_frame(from_id=seen, reason=reason, retained_min_id=rows[0].id)
        return [gap], rows[-1].id
    if rows and rows[0].id > seen + 1:
        # The missing range was partly swept before this read: the surviving
        # rows are NOT an exact continuation. Same contiguity check as the
        # replay preamble — derived from the snapshot actually replayed, so a
        # sweep between reads cannot fake exactness.
        frames = [_gap_frame(from_id=seen, reason="pruned", retained_min_id=rows[0].id)]
        frames.extend(_row_frame(row) for row in rows)
    else:
        frames = [_row_frame(row) for row in rows]
    if rows:
        seen = rows[-1].id
    if rid > seen:
        # Defensive: the row was published before it was readable. Emit the bus
        # copy so the event is not lost, keeping the ids monotone.
        seen = rid
        frames.append({"id": str(rid), **default_frame(event)})
    return frames, seen


async def _event_frames(
    request: Request, event_bus: Any, queries: Any, last_id: int | None
) -> AsyncIterator[dict[str, Any]]:
    """Replay-then-tail frames for one `/events/stream` connection."""
    if _STREAM_SEMAPHORE.locked():
        yield saturated_frame("feed.saturated", EVENTS_STREAM_CONCURRENCY_MAX)
        return

    async with _STREAM_SEMAPHORE:
        # Highest durable id already delivered on this connection. The live tail
        # drops anything at or below it, which is what makes the replay window
        # and the subscription overlap harmless.
        seen = last_id

        async def replay() -> AsyncIterator[dict[str, Any]]:
            nonlocal seen
            if last_id is None or queries is None:
                return
            # Check before replay for retention beyond the cursor, more than _MAX_SSE_REPLAY
            # rows (limit + 1), or a cursor above the retained maximum. Clamp an oversized
            # cursor or future live rows would all be dropped. Report the existing pruned
            # reason only for nonempty tables. Read min/max in one statement so a retention
            # sweep cannot make the bounds disagree.
            min_id, max_id = await queries.feed_bounds()
            rows = await queries.replay_events_after(last_id, limit=_MAX_SSE_REPLAY + 1)
            overflow = len(rows) > _MAX_SSE_REPLAY
            if last_id > max_id:
                seen = max_id
            ahead = max_id > 0 and last_id > max_id
            if rows:
                # (a) is derived from the SNAPSHOT ACTUALLY REPLAYED, never
                # from the bounds read that preceded it: a retention sweep
                # landing between the two reads leaves `min_id` describing a
                # table that no longer exists, and the classic shape of that
                # race — resume from 0, read bounds (1, 6), rows 1..3 swept,
                # replay starts at 4 — reads perfectly healthy against the
                # stale minimum while three rows vanish. The first row the
                # client is actually about to receive cannot lie about where
                # the replay begins.
                pruned = rows[0].id > last_id + 1
                retained_min = rows[0].id
            else:
                # Nothing to replay, so there is no snapshot to derive from and
                # the bounds are the only evidence there is. Both the
                # empty-table rule (`min_id == 0` is never a gap) and the
                # cursor-above-the-feed clamp live on this side.
                pruned = min_id > 0 and last_id < min_id - 1
                retained_min = min_id
            if pruned or overflow or ahead:
                yield _gap_frame(
                    from_id=last_id,
                    reason="pruned" if (pruned or ahead) else "overflow",
                    retained_min_id=retained_min,
                )
            for row in rows[:_MAX_SSE_REPLAY]:
                seen = row.id
                yield _row_frame(row)

        async def live_frames(event: dict[str, Any]) -> list[dict[str, Any]]:
            nonlocal seen
            rid = event.get("id")
            if not isinstance(rid, int):
                # A legacy bus-only frame (`job.status_changed`). It flows,
                # but carries NO id: an empty `id:` line would reset the
                # client's Last-Event-ID and break the next resume.
                return [default_frame(event)]
            if seen is not None and rid <= seen:
                return []  # already covered by the replay window
            if seen is not None and queries is not None and rid > seen + 1:
                frames, seen = await _catch_up_frames(queries, event, rid, seen)
                return frames
            # `seen is None` — this connection started at live, so there is no
            # watermark a hole could be measured against; the first durable
            # frame simply establishes one.
            seen = rid
            return [{"id": str(rid), **default_frame(event)}]

        async for frame in sse_from_bus(
            request, event_bus, predicate=_not_audit, frame=live_frames, preamble=replay
        ):
            yield frame
