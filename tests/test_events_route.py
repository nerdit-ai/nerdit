"""Tests for the event read surface — ``GET /api/events`` + the resumable
stream (P24a WP2).

Two layers:

* a pattern-2 route harness (``httpx.ASGITransport`` over a real in-memory DB
  under the auth middleware) for the bounded read: both pagination directions,
  the filters, the clamp-don't-reject posture, and the readonly read; and
* direct drives of the stream's frame generator for the SSE contract, because
  the things worth pinning there are orderings — *subscribe before you read the
  backlog*, *gap frame before any replay frame*, *live frame dropped when the
  replay already carried it* — and an ordering is exactly what a
  connect-and-hope integration test cannot assert deterministically.

The **direction pin** (``cursor`` walks backwards, ``since_id`` walks forwards)
and the **retained-minimum gap** are the two regressions with teeth here: the
first silently reconciles a consumer into the past, the second silently drops
an unbounded number of events while every count-based check reads healthy.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request

from nerdit.core.events import EventBus
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.limits import _MAX_SSE_REPLAY, EVENTS_STREAM_CONCURRENCY_MAX
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.events import (
    _MAX_CATCHUP,
    _STREAM_SEMAPHORE,
    _event_frames,
    _not_audit,
    _split_types,
    stream_cluster_events,
)
from nerdit.daemon.routes.events import router as events_router
from nerdit.daemon.sse import parse_resume_cursor
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, TokenRole
from nerdit.db.queries import Queries

LEGACY = "legacy-admin-token"
READONLY_RAW = "readonly-raw-token"

_TOKENS = (
    ApiToken(
        id="tok-readonly",
        name="readonly",
        token_hash=hash_token(READONLY_RAW),
        role=TokenRole.readonly,
    ),
)


# --- fixtures ----------------------------------------------------------------


@pytest_asyncio.fixture
async def harness():
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for tok in _TOKENS:
        await queries.create_api_token(tok)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(events_router, prefix="/api")
    app.state.queries = queries
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield client, queries
    finally:
        await client.aclose()
        await db.close()


def _auth(raw: str = LEGACY) -> dict:
    return {"Authorization": f"Bearer {raw}"}


async def _seed(queries: Queries, rows: list[tuple[str, str | None]]) -> None:
    """Bulk-insert ``(type, service_name)`` rows; ids are 1..len in order."""
    await queries._db.conn.executemany(
        "INSERT INTO events (type, service_name) VALUES (?, ?)", rows
    )
    await queries._db.conn.commit()


async def _seed_n(queries: Queries, count: int, *, type_: str = "service.healthy") -> None:
    await _seed(queries, [(type_, "app") for _ in range(count)])


def _ids(body: dict) -> list[int]:
    return [item["id"] for item in body["items"]]


# --- pure units --------------------------------------------------------------


def test_split_types_splits_and_strips_only():
    """The route splits; the QUERY layer owns the 10-entry bound (one clamp)."""
    assert _split_types(None) is None
    assert _split_types("  ,  ") is None
    assert _split_types(" a , b ") == ["a", "b"]


@pytest.mark.asyncio
async def test_an_over_long_types_filter_is_clamped_never_rejected(queries):
    """An over-long filter degrades to its first 10 entries (D-P24-12).

    Asserted through the query layer, which is where the single bound lives.
    """
    await _seed(queries, [(f"t{i}", "app") for i in range(3)])
    items, _ = await queries.list_events(types=[f"t{i}" for i in range(25)])
    # Only the first 10 requested types are honoured, and no error is raised.
    assert [item.type for item in items] == ["t2", "t1", "t0"]


def test_not_audit_keeps_the_audit_trail_off_the_general_stream():
    assert _not_audit({"type": "service.failed"}) is True
    assert _not_audit({"type": "audit.service.create"}) is False
    assert _not_audit({}) is True


def _scope_request(headers: dict[str, str] | None = None, query: str = "") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/events/stream",
            "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
            "query_string": query.encode(),
        }
    )


def _parse(request) -> int | None:
    """The ONE shared parser, as ``/events/stream`` configures it."""
    return parse_resume_cursor(request, query_param="last_event_id")


def test_last_event_id_prefers_the_header_and_falls_back_to_the_query():
    assert _parse(_scope_request({"Last-Event-ID": "42"})) == 42
    assert _parse(_scope_request(query="last_event_id=7")) == 7
    # Header wins when both are present (it is the canonical carrier).
    assert _parse(_scope_request({"Last-Event-ID": "42"}, "last_event_id=7")) == 42


def test_malformed_last_event_id_is_ignored_not_rejected():
    """A bad resume cursor means "start live" — never a permanently broken stream."""
    assert _parse(_scope_request({"Last-Event-ID": "not-a-number"})) is None
    assert _parse(_scope_request({"Last-Event-ID": "-5"})) is None
    assert _parse(_scope_request()) is None


def test_an_out_of_range_last_event_id_is_unparseable_too():
    """Beyond sqlite's signed-64-bit binding range it can only be a client bug.

    Passing it through would raise ``OverflowError`` deep in the replay read,
    on a generator whose only failure mode is a dead stream.
    """
    assert _parse(_scope_request({"Last-Event-ID": str(2**70)})) is None
    assert _parse(_scope_request({"Last-Event-ID": str(2**63)})) is None
    assert _parse(_scope_request({"Last-Event-ID": str(2**63 - 1)})) == 2**63 - 1


# --- the payload's ``ts``: ONE shape across every surface ---------------------


def test_event_ts_from_a_stored_row_serializes_offset_aware():
    """The stored column is naive UTC; the live bus frame is offset-aware.

    Without normalization one event has two incompatible ``ts`` shapes
    depending on which surface a consumer read it from, and the D-P24-3 locked
    payload example pins the aware form.
    """
    from datetime import UTC, datetime

    from nerdit.db.rows import Event

    # Exactly what SQLite's ``datetime('now')`` default writes.
    stored = Event(id=1, ts="2026-08-07 12:34:56", type="service.failed")
    assert stored.ts is not None and stored.ts.tzinfo is not None
    assert stored.model_dump(mode="json")["ts"].endswith("+00:00")

    # Comparable with a live-frame ts, which is what a consumer merging the
    # replay window with the live tail actually does.
    live = datetime.fromisoformat(datetime.now(UTC).isoformat())
    assert datetime.fromisoformat(stored.model_dump(mode="json")["ts"]) < live

    # An already-aware value is passed through untouched.
    aware = Event(id=2, ts="2026-08-07T12:34:56+00:00", type="service.healthy")
    assert aware.model_dump(mode="json")["ts"] == "2026-08-07T12:34:56+00:00"
    assert Event(id=3, type="service.healthy").model_dump(mode="json")["ts"] is None


@pytest.mark.asyncio
async def test_the_read_route_serves_an_offset_aware_ts(harness):
    client, queries = harness
    await _seed_n(queries, 1)

    body = (await client.get("/api/events", headers=_auth())).json()
    assert body["items"][0]["ts"].endswith("+00:00")


# --- GET /api/events: pagination --------------------------------------------


@pytest.mark.asyncio
async def test_browse_pages_backwards_newest_first(harness):
    client, queries = harness
    await _seed_n(queries, 5)

    first = await client.get("/api/events?limit=2", headers=_auth())
    assert first.status_code == 200, first.text
    assert _ids(first.json()) == [5, 4]
    cursor = first.json()["next_cursor"]
    assert cursor == "4"

    second = await client.get(f"/api/events?limit=2&cursor={cursor}", headers=_auth())
    assert _ids(second.json()) == [3, 2]


@pytest.mark.asyncio
async def test_rows_pruned_under_an_open_cursor_degrade_to_fewer_results(harness):
    """Retention deleting rows behind a live cursor is *fewer rows*, never an error.

    The cursor is a plain id keyset, so a page that has been swept out from
    under a consumer must still answer 200 with whatever survives.
    """
    client, queries = harness
    await _seed_n(queries, 6)

    first = await client.get("/api/events?limit=3", headers=_auth())
    cursor = first.json()["next_cursor"]
    assert _ids(first.json()) == [6, 5, 4]

    # Sweep everything but the newest 4 (ids 1 and 2 go).
    while await queries.sweep_events(4):
        pass

    second = await client.get(f"/api/events?limit=3&cursor={cursor}", headers=_auth())
    assert second.status_code == 200, second.text
    assert _ids(second.json()) == [3]


@pytest.mark.asyncio
async def test_cursor_and_since_id_walk_opposite_directions_over_one_fixture(harness):
    """THE direction pin (D-P24-6).

    ``?cursor=N`` returns ids **below** N newest-first; ``?since_id=N`` returns
    ids **above** N oldest-first. Asserted as two disjoint, oppositely-ordered
    sets over the same rows, so the "consumers reconcile forward" contract
    cannot silently invert into "consumers walk into the past".
    """
    client, queries = harness
    await _seed_n(queries, 5)

    backwards = _ids((await client.get("/api/events?cursor=3", headers=_auth())).json())
    forwards = _ids((await client.get("/api/events?since_id=3", headers=_auth())).json())

    assert backwards == [2, 1]
    assert forwards == [4, 5]
    assert backwards == sorted(backwards, reverse=True)
    assert forwards == sorted(forwards)
    assert set(backwards).isdisjoint(forwards)


@pytest.mark.asyncio
async def test_forward_next_cursor_is_the_largest_id_and_pages_without_repeats(harness):
    client, queries = harness
    await _seed_n(queries, 6)

    first = (await client.get("/api/events?since_id=0&limit=2", headers=_auth())).json()
    assert _ids(first) == [1, 2]
    assert first["next_cursor"] == "2"  # the LARGEST id returned, not the smallest

    second = (
        await client.get(f"/api/events?since_id={first['next_cursor']}&limit=2", headers=_auth())
    ).json()
    assert _ids(second) == [3, 4]
    assert set(_ids(first)).isdisjoint(_ids(second))

    third = (
        await client.get(f"/api/events?since_id={second['next_cursor']}&limit=2", headers=_auth())
    ).json()
    assert _ids(third) == [5, 6]
    assert third["next_cursor"] is None  # exhausted


@pytest.mark.asyncio
async def test_cursor_and_since_id_together_are_a_400(harness):
    client, queries = harness
    await _seed_n(queries, 3)

    resp = await client.get("/api/events?cursor=3&since_id=1", headers=_auth())
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "bad_request"


@pytest.mark.asyncio
async def test_malformed_cursor_is_a_400_bad_request(harness):
    client, queries = harness
    await _seed_n(queries, 3)

    resp = await client.get("/api/events?cursor=not-an-id", headers=_auth())
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "bad_request"


@pytest.mark.asyncio
async def test_an_out_of_range_since_id_is_a_400_not_a_500(harness):
    """sqlite binds 64-bit integers; a bigger one is a client error, not a crash."""
    client, queries = harness
    await _seed_n(queries, 3)

    resp = await client.get(f"/api/events?since_id={2**70}", headers=_auth())
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "bad_request"


@pytest.mark.asyncio
async def test_an_out_of_range_cursor_is_a_400_not_a_500(harness):
    client, queries = harness
    await _seed_n(queries, 3)

    resp = await client.get(f"/api/events?cursor={2**70}", headers=_auth())
    assert resp.status_code == 400, resp.text
    assert resp.json()["code"] == "bad_request"


@pytest.mark.asyncio
async def test_limit_clamps_at_200_and_at_1_instead_of_rejecting(harness):
    """No new reject-vs-clamp divergence (D-P24-12) — both ends clamp."""
    client, queries = harness
    await _seed_n(queries, 5)

    high = await client.get("/api/events?limit=100000", headers=_auth())
    assert high.status_code == 200, high.text
    assert len(high.json()["items"]) == 5

    low = await client.get("/api/events?limit=0", headers=_auth())
    assert low.status_code == 200, low.text
    assert len(low.json()["items"]) == 1


# --- GET /api/events: filters + auth ----------------------------------------


@pytest.mark.asyncio
async def test_types_and_service_filters(harness):
    client, queries = harness
    await _seed(
        queries,
        [
            ("service.failed", "alpha"),
            ("service.healthy", "alpha"),
            ("model.ready", "beta"),
            ("service.failed", "beta"),
        ],
    )

    typed = (await client.get("/api/events?types=service.failed", headers=_auth())).json()
    assert _ids(typed) == [4, 1]

    multi = (
        await client.get("/api/events?types=model.ready,service.healthy", headers=_auth())
    ).json()
    assert _ids(multi) == [3, 2]

    scoped = (await client.get("/api/events?service=beta", headers=_auth())).json()
    assert _ids(scoped) == [4, 3]

    both = (
        await client.get("/api/events?types=service.failed&service=beta", headers=_auth())
    ).json()
    assert _ids(both) == [4]


@pytest.mark.asyncio
async def test_readonly_principal_can_read_the_feed(harness):
    """Any authenticated principal, unlike the admin-gated audit trail."""
    client, queries = harness
    await _seed_n(queries, 2)

    resp = await client.get("/api/events", headers=_auth(READONLY_RAW))
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["items"]) == 2


@pytest.mark.asyncio
async def test_the_feed_needs_a_token(harness):
    client, _ = harness
    assert (await client.get("/api/events")).status_code == 401


# --- the stream: harness -----------------------------------------------------


class _StubRequest:
    """The two things the frame generator asks of a request."""

    def __init__(self) -> None:
        self.disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


async def _drain(agen, count: int, *, timeout: float = 2.0) -> list[dict[str, Any]]:
    """Pull exactly ``count`` frames, skipping heartbeats."""
    frames: list[dict[str, Any]] = []
    while len(frames) < count:
        frame = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
        if frame.get("event") == "ping":
            continue
        frames.append(frame)
    return frames


def _payload(frame: dict[str, Any]) -> dict[str, Any]:
    return json.loads(frame["data"])


class _HookedQueries:
    """Queries proxy that fires a callback at a chosen point of the replay leg.

    The generator is lazy, so "publish something while the stream is live" is
    otherwise a race against a sleep. These hooks are the windows worth testing
    and all are reached strictly **after** the bus subscription exists:
    ``before_replay`` fires inside ``feed_bounds`` (the replay leg's first read,
    before the backlog query), ``after_bounds`` in the gap *between* the bounds
    read and the backlog query (the prune-vs-replay TOCTOU), and
    ``after_replay`` once ``replay_events_after`` has returned its rows.
    """

    def __init__(self, inner, *, before_replay=None, after_bounds=None, after_replay=None) -> None:
        self._inner = inner
        self._before = before_replay
        self._between = after_bounds
        self._after = after_replay
        self._fired: set[str] = set()

    async def _fire(self, name: str, hook) -> None:
        if hook is None or name in self._fired:
            return
        self._fired.add(name)
        await hook()

    async def feed_bounds(self):
        await self._fire("before", self._before)
        bounds = await self._inner.feed_bounds()
        await self._fire("between", self._between)
        return bounds

    async def replay_events_after(self, after_id, limit):
        rows = await self._inner.replay_events_after(after_id, limit)
        await self._fire("after", self._after)
        return rows

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest_asyncio.fixture
async def feed():
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    bus = EventBus()
    try:
        yield queries, bus, _StubRequest()
    finally:
        await db.close()


# --- the stream: replay, dedupe, ordering ------------------------------------


@pytest.mark.asyncio
async def test_replay_frames_carry_the_row_id_and_the_live_tail_follows(feed):
    queries, bus, request = feed
    await _seed_n(queries, 3)

    agen = _event_frames(request, bus, queries, 1)
    try:
        replayed = await _drain(agen, 2)
        assert [f["id"] for f in replayed] == ["2", "3"]
        assert _payload(replayed[0])["id"] == 2

        bus.publish({"type": "service.failed", "id": 4, "service_name": "app"})
        live = await _drain(agen, 1)
        assert live[0]["id"] == "4"
        assert live[0]["event"] == "service.failed"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_an_event_landing_between_subscribe_and_the_replay_read_is_delivered_once(feed):
    """The subscribe-before-you-read ordering, asserted at its exact seam.

    ``feed_bounds`` is the first read the replay leg makes, so a row inserted
    *and* published from inside it lands after the subscription exists but
    before the backlog query — the window a naive "read then subscribe" would
    lose it in, and the window a naive "subscribe then replay" would duplicate
    it in. Exactly one delivery is the only correct answer.
    """
    queries, bus, request = feed
    await _seed_n(queries, 2)

    async def inject() -> None:
        row_id = await queries.insert_event(type="service.failed", service_name="app")
        bus.publish({"type": "service.failed", "id": row_id, "service_name": "app"})

    agen = _event_frames(request, bus, _HookedQueries(queries, before_replay=inject), 2)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["id"] == "3"
        assert frames[0]["event"] == "service.failed"

        # The bus copy of the very same row must NOT come through again.
        bus.publish({"type": "service.healthy", "id": 4, "service_name": "app"})
        following = await _drain(agen, 1)
        assert following[0]["id"] == "4"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_an_event_published_after_the_replay_read_still_arrives(feed):
    """The other half of the seam: not in the backlog, so the live tail owns it."""
    queries, bus, request = feed
    await _seed_n(queries, 2)

    async def inject() -> None:
        bus.publish({"type": "service.failed", "id": 3, "service_name": "app"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 2)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["id"] == "3"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_legacy_bus_only_frames_flow_without_an_id(feed):
    """``job.status_changed`` is never durable — an id would break the next resume."""
    queries, bus, request = feed

    async def inject() -> None:
        bus.publish({"type": "job.status_changed", "job_id": "abc", "status": "running"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 0)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] == "job.status_changed"
        assert "id" not in frames[0]
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_audit_events_never_reach_the_general_stream(feed):
    queries, bus, request = feed

    async def inject() -> None:
        bus.publish({"type": "audit.service.create", "id": 1})
        bus.publish({"type": "service.failed", "id": 2})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 0)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] == "service.failed"
    finally:
        await agen.aclose()


# --- the stream: the live tail's DB-backed catch-up --------------------------


@pytest.mark.asyncio
async def test_a_dropped_bus_frame_is_read_back_off_disk_not_skipped(feed):
    """The burst-drop regression: the subscriber queue is bounded and DROPS.

    Ids 1 then 3 reach the tail while 2 only ever existed on disk — the exact
    shape a full subscriber queue produces. Accepting "anything above the
    watermark" delivers 1, 3 and breaks the stream's exact-or-gap promise
    silently; the catch-up read closes the hole with the real row.
    """
    queries, bus, request = feed
    await _seed_n(queries, 3)

    async def inject() -> None:
        # Two more rows land; only the SECOND survives the (simulated) queue.
        for _ in range(2):
            await queries.insert_event(type="service.failed", service_name="app")
        bus.publish({"type": "service.failed", "id": 5, "service_name": "app"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 3)
    try:
        frames = await _drain(agen, 2)
        assert [f["id"] for f in frames] == ["4", "5"]
        assert [_payload(f)["id"] for f in frames] == [4, 5]
        assert all(f["event"] != "feed.gap" for f in frames)

        # And the watermark moved: the dropped row is not replayed a second time.
        bus.publish({"type": "service.healthy", "id": 4, "service_name": "app"})
        row_id = await queries.insert_event(type="service.healthy", service_name="app")
        bus.publish({"type": "service.healthy", "id": row_id, "service_name": "app"})
        following = await _drain(agen, 1)
        assert following[0]["id"] == str(row_id)
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_an_out_of_order_publish_catches_up_and_drops_the_late_copy(feed):
    """Two publishes crossing is handled EXACTLY, not reported as a gap.

    The bus hands over 2 before 1 while both rows exist. The catch-up read
    triggered by 2 delivers 1 and 2 in id order; the late copy of 1 is then
    below the watermark and dropped, as a replay-window overlap would be.
    """
    queries, bus, request = feed
    await _seed_n(queries, 2)
    inserted: list[int] = []

    async def inject() -> None:
        for _ in range(2):
            inserted.append(await queries.insert_event(type="service.failed", service_name="b"))
        bus.publish({"type": "service.failed", "id": inserted[1], "service_name": "b"})
        bus.publish({"type": "service.failed", "id": inserted[0], "service_name": "b"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 2)
    try:
        frames = await _drain(agen, 2)
        assert [f["id"] for f in frames] == [str(inserted[0]), str(inserted[1])]
        assert all(f["event"] != "feed.gap" for f in frames)

        # The late copy of the lower id was dropped — the next frame is new.
        row_id = await queries.insert_event(type="service.healthy", service_name="b")
        bus.publish({"type": "service.healthy", "id": row_id, "service_name": "b"})
        following = await _drain(agen, 1)
        assert following[0]["id"] == str(row_id)
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_a_hole_larger_than_the_catchup_bound_gaps_then_resumes(feed):
    """One live frame may not drag an unbounded read off disk.

    Past ``_MAX_CATCHUP`` the honest answer is the pinned ``feed.gap`` frame
    plus a watermark jump — never silence, and never an unbounded query on the
    shared connection.
    """
    queries, bus, request = feed
    hole = _MAX_CATCHUP + 2

    async def inject() -> None:
        await _seed_n(queries, hole)
        bus.publish({"type": "service.healthy", "id": hole, "service_name": "app"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 0)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] == "feed.gap"
        body = _payload(frames[0])
        assert body["reason"] == "overflow"
        assert body["from"] == 0
        # ``retained_min_id`` is the OLDEST surviving row: the skipped rows are
        # still on disk, and the CLI's reconcile hint replays from one below
        # this value — pointing it at the watermark would silently skip them.
        assert body["retained_min_id"] == 1

        # ...and the tail resumes past the newest row the bounded read
        # returned, rather than going mute.
        row_id = await queries.insert_event(type="service.failed", service_name="app")
        bus.publish({"type": "service.failed", "id": row_id, "service_name": "app"})
        following = await _drain(agen, 2)
        assert [f["id"] for f in following] == [str(hole), str(row_id)]
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_a_sweep_inside_the_catchup_hole_gaps_before_the_surviving_rows(feed):
    """Retention eating part of a missed range must not fake exactness.

    The subscriber queue drops rows 2..4, and the sweep deletes 2 and 3 before
    the next surviving frame triggers catch-up. The read returns rows starting
    ABOVE ``seen + 1`` — the same contiguity check the replay preamble does
    must fire here too, or the surviving rows are emitted as an exact
    continuation and the loss is silent.
    """
    queries, bus, request = feed
    await _seed_n(queries, 1)

    async def inject() -> None:
        # Rows 2..5 land after the replay preamble; the sweep then keeps only
        # the last two, so the catch-up read finds a hole at 2..3.
        await _seed_n(queries, 4)
        await queries.sweep_events(2)
        bus.publish({"type": "service.healthy", "id": 5, "service_name": "app"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 1)
    try:
        frames = await _drain(agen, 3)
        assert frames[0]["event"] == "feed.gap"
        body = _payload(frames[0])
        assert body["reason"] == "pruned"
        assert body["from"] == 1
        assert body["retained_min_id"] == 4
        assert [f["id"] for f in frames[1:]] == ["4", "5"]
    finally:
        await agen.aclose()


# --- the stream: the two feed.gap causes -------------------------------------


@pytest.mark.asyncio
async def test_a_sweep_between_the_bounds_read_and_the_replay_still_gaps(feed):
    """The prune-vs-replay TOCTOU: the bounds read is stale by the time it is used.

    ``feed_bounds()`` says the feed starts at 1, retention then sweeps 1..3,
    and the replay begins at 4 — a bounds-derived check compares the cursor
    against a minimum that no longer exists and reports a clean resume while
    three rows are gone. Deriving the decision from the replay snapshot itself
    cannot go stale: the first row the client receives IS where the feed starts.
    """
    queries, bus, request = feed
    await _seed_n(queries, 6)

    async def sweep() -> None:
        while await queries.sweep_events(3):  # keep the newest 3 → ids 4..6
            pass

    agen = _event_frames(request, bus, _HookedQueries(queries, after_bounds=sweep), 0)
    try:
        frames = await _drain(agen, 2)
        assert frames[0]["event"] == "feed.gap"
        body = _payload(frames[0])
        assert body["reason"] == "pruned"
        assert body["from"] == 0
        assert body["retained_min_id"] == 4
        assert frames[1]["id"] == "4"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_gap_frame_on_overflow_only_past_the_boundary(feed):
    """The ``limit + 1`` boundary, not a "a full page means probably more" guess.

    Exactly ``_MAX_SSE_REPLAY`` rows behind the cursor is a *complete* replay
    and must not raise a gap; one more row must.
    """
    queries, bus, request = feed
    await _seed_n(queries, _MAX_SSE_REPLAY)

    agen = _event_frames(request, bus, queries, 0)
    try:
        first = await _drain(agen, 1)
        assert first[0]["event"] != "feed.gap"
        assert first[0]["id"] == "1"
    finally:
        await agen.aclose()

    await _seed_n(queries, 1)  # now _MAX_SSE_REPLAY + 1 rows behind the cursor
    agen = _event_frames(request, bus, queries, 0)
    try:
        first = await _drain(agen, 1)
        assert first[0]["event"] == "feed.gap"
        body = _payload(first[0])
        assert body["reason"] == "overflow"
        assert body["from"] == 0
        assert body["retained_min_id"] == 1
        # And the replay that follows is truncated to the cap, not unbounded.
        following = await _drain(agen, 1)
        assert following[0]["id"] == "1"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_gap_frame_on_a_pruned_cursor_even_with_a_tiny_backlog(feed):
    """The finding-4 regression pin.

    Retention swept past the client's cursor, leaving **far fewer** than
    ``_MAX_SSE_REPLAY`` newer rows — so every count-based check reads this as a
    healthy, complete replay while the client silently loses everything between
    its cursor and the retained minimum. Only the retained-minimum comparison
    catches it, and the frame must precede the replay.
    """
    queries, bus, request = feed
    await _seed_n(queries, 10)
    while await queries.sweep_events(6):  # keep ids 5..10
        pass
    assert await queries.min_event_id() == 5

    agen = _event_frames(request, bus, queries, 2)
    try:
        frames = await _drain(agen, 2)
        assert frames[0]["event"] == "feed.gap"
        body = _payload(frames[0])
        assert body["reason"] == "pruned"
        assert body["from"] == 2
        assert body["retained_min_id"] == 5
        # The gap frame comes FIRST; the surviving rows follow it.
        assert frames[1]["id"] == "5"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_a_cursor_exactly_one_below_the_retained_minimum_is_not_a_gap(feed):
    """``last_id + 1 == min_event_id()`` is a perfectly contiguous resume."""
    queries, bus, request = feed
    await _seed_n(queries, 10)
    while await queries.sweep_events(6):
        pass

    agen = _event_frames(request, bus, queries, 4)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] != "feed.gap"
        assert frames[0]["id"] == "5"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_a_cursor_above_the_retained_maximum_gaps_and_then_flows(feed):
    """A cursor ABOVE every retained id must not mute the stream forever.

    Left unclamped, the live tail's ``rid <= seen`` dedupe drops every new row
    (their ids are all below the bogus cursor) and the client sees silence. The
    fix gaps once — reusing the pinned ``pruned`` literal, the remediation is
    identical — and clamps the dedupe cursor down to the real maximum.
    """
    queries, bus, request = feed
    await _seed_n(queries, 3)  # ids 1..3

    published: list[int] = []

    async def inject() -> None:
        row_id = await queries.insert_event(type="service.healthy", service_name="later")
        published.append(row_id)
        bus.publish({"type": "service.healthy", "id": row_id, "service_name": "later"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 999)
    try:
        frames = await _drain(agen, 2)
        assert frames[0]["event"] == "feed.gap"
        body = _payload(frames[0])
        assert body["reason"] == "pruned"
        assert body["from"] == 999
        # The event published after the clamp IS delivered — that is the bug.
        assert frames[1]["event"] == "service.healthy"
        assert frames[1]["id"] == str(published[0])
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_an_empty_table_is_never_a_gap(feed):
    """``min_event_id() == 0`` means "nothing retained", not "you were pruned"."""
    queries, bus, request = feed

    async def inject() -> None:
        bus.publish({"type": "service.healthy", "id": 100})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 99)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] == "service.healthy"
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_an_empty_table_still_clamps_so_the_first_row_is_not_muted(feed):
    """No gap frame here (the plan's empty-table rule) — but the clamp still runs.

    A fresh feed's first durable row gets id 1, which is far below a stale
    ``Last-Event-ID``; without the clamp ``rid <= seen`` would drop it and
    every row after it for the life of the connection.
    """
    queries, bus, request = feed

    async def inject() -> None:
        row_id = await queries.insert_event(type="service.healthy", service_name="first")
        bus.publish({"type": "service.healthy", "id": row_id, "service_name": "first"})

    agen = _event_frames(request, bus, _HookedQueries(queries, after_replay=inject), 99)
    try:
        frames = await _drain(agen, 1)
        assert frames[0]["event"] == "service.healthy"
        assert frames[0]["id"] == "1"
    finally:
        await agen.aclose()


# --- the stream: bounded concurrency ----------------------------------------


@pytest.mark.asyncio
async def test_a_saturated_daemon_answers_one_frame_and_closes(feed):
    queries, bus, request = feed
    held = [
        asyncio.ensure_future(_STREAM_SEMAPHORE.acquire())
        for _ in range(EVENTS_STREAM_CONCURRENCY_MAX)
    ]
    await asyncio.gather(*held)
    try:
        agen = _event_frames(request, bus, queries, None)
        frames = [frame async for frame in agen]
        assert len(frames) == 1
        body = _payload(frames[0])
        assert body["type"] == "feed.saturated"
        assert body["reason"] == "concurrency_limit"
        assert body["limit"] == EVENTS_STREAM_CONCURRENCY_MAX
    finally:
        for _ in range(EVENTS_STREAM_CONCURRENCY_MAX):
            _STREAM_SEMAPHORE.release()


# --- the stream: route wiring ------------------------------------------------
#
# There is deliberately no end-to-end HTTP read of ``/events/stream`` here:
# httpx's ``ASGITransport`` buffers a response body to completion before
# handing it back, so an endless SSE stream over it never returns — the test
# would hang, not fail. The route wiring is covered by the two checks below
# plus the ``list_events``/``stream_cluster_events`` operationId pins in
# ``tests/test_openapi.py``; the frame contract is covered above, where it is
# assertable.


@pytest.mark.asyncio
async def test_the_stream_route_is_mounted_and_gated(harness):
    """Auth is answered by the middleware, before the handler ever streams."""
    client, _ = harness
    assert (await client.get("/api/events/stream")).status_code == 401


@pytest.mark.asyncio
async def test_stream_route_degrades_to_an_sse_response_without_a_bus():
    """A daemon with no event bus still answers SSE (heartbeats only)."""

    class _App:
        state = SimpleNamespace(event_bus=None, queries=None)

    request = _scope_request()
    request.scope["app"] = _App()
    response = await stream_cluster_events(request)
    assert isinstance(response, EventSourceResponse)
