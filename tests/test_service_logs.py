"""Test literal substring filters and resumable service-log streams.

Percent signs stay literal. Apply filters and tail limits in one bounded query,
never by fetching all rows. End only after exhausting a terminal service's cursor;
degraded and restarting services keep streaming.
Drive the frame generator directly: ASGITransport buffers endless SSE responses.
Pin route wiring through OpenAPI and a direct route call.
"""

from __future__ import annotations

import json

import pytest
from fastapi import Response
from sse_starlette.sse import EventSourceResponse

from nerdit.daemon.routes import service_logs_stream as stream_mod
from nerdit.daemon.routes.service_logs_stream import _log_frames, stream_service_logs
from nerdit.daemon.sse import parse_resume_cursor
from nerdit.db.models import Job, JobKind, JobStatus, LogStream
from nerdit.db.queries.logs import _escape_like


async def _service(queries, name="svc", status=JobStatus.running):
    return await queries.create_job(
        Job(
            name=name,
            kind=JobKind.service,
            service_name=name,
            gpu_count=0,
            status=status,
            desired_state="running",
            config='{"image": "demo:latest", "port": 8000}',
        )
    )


# --- grep: a literal substring, not a pattern ---------------------------------


async def test_grep_matches_a_literal_substring(queries):
    job = await _service(queries)
    await queries.append_log(job.id, "starting up")
    await queries.append_log(job.id, "Traceback (most recent call last)")
    await queries.append_log(job.id, "listening on 8000")

    entries = await queries.get_logs(job.id, grep="Traceback")
    assert [e.message for e in entries] == ["Traceback (most recent call last)"]


async def test_grep_percent_is_literal_not_a_wildcard(queries):
    """The pin. ``%`` is LIKE's "match anything" — here it must match a ``%``."""
    job = await _service(queries)
    await queries.append_log(job.id, "progress 100% done")
    await queries.append_log(job.id, "no percent sign here")

    entries = await queries.get_logs(job.id, grep="100%")
    assert [e.message for e in entries] == ["progress 100% done"]

    # A bare '%' finds the one line that contains a percent sign — not both.
    entries = await queries.get_logs(job.id, grep="%")
    assert [e.message for e in entries] == ["progress 100% done"]


async def test_grep_underscore_is_literal_not_a_single_char_wildcard(queries):
    job = await _service(queries)
    await queries.append_log(job.id, "db_url resolved")
    await queries.append_log(job.id, "dbXurl resolved")

    entries = await queries.get_logs(job.id, grep="db_url")
    assert [e.message for e in entries] == ["db_url resolved"]


async def test_grep_backslash_is_literal(queries):
    """``\\`` is the ESCAPE character, so it has to be escaped first of all three.

    Escaping ``%``/``_`` before ``\\`` would double-escape the backslashes this
    code adds and the needle would stop matching itself.
    """
    job = await _service(queries)
    await queries.append_log(job.id, r"path C:\temp\build")
    await queries.append_log(job.id, "path /tmp/build")

    entries = await queries.get_logs(job.id, grep=r"C:\temp")
    assert [e.message for e in entries] == [r"path C:\temp\build"]


def test_escape_like_clamps_the_needle():
    """200 chars, clamped not rejected — and clamped before escaping.

    Clamping after escaping could truncate between a backslash and the
    character it escapes, producing a pattern SQLite rejects outright.
    """
    assert len(_escape_like("x" * 500)) == 200
    assert _escape_like("%" * 500) == r"\%" * 200


# --- LIMIT-first --------------------------------------------------------------


async def test_filters_are_applied_in_sql_under_the_limit(queries, monkeypatch):
    """The bound and the filters must be in ONE query (WP3.2).

    "Fetch the tail, then filter in Python" would silently mean "the last N
    lines, of which some match" — and "fetch everything, then filter" would
    materialize an unbounded set. Both are ruled out by asserting the executed
    SQL carries the grep clause and the LIMIT together.
    """
    job = await _service(queries)
    await queries.append_log(job.id, "boom")

    seen: list[str] = []
    real_execute = queries._db.conn.execute

    async def spy(sql, params=()):
        seen.append(" ".join(str(sql).split()))
        return await real_execute(sql, params)

    monkeypatch.setattr(queries._db.conn, "execute", spy)
    await queries.get_logs(job.id, tail=10, grep="boom", since_ts="2020-01-01 00:00:00")

    assert len(seen) == 1, "the filtered read must be a single query"
    sql = seen[0]
    assert "message LIKE '%' || ? || '%' ESCAPE" in sql
    assert "timestamp >= ?" in sql
    assert sql.rstrip().endswith("LIMIT ?")


async def test_tail_counts_matching_lines(queries):
    """``tail`` bounds the FILTERED set, which is the useful reading of it."""
    job = await _service(queries)
    for i in range(10):
        await queries.append_log(job.id, f"noise {i}")
        await queries.append_log(job.id, f"ERROR {i}")

    entries = await queries.get_logs(job.id, tail=3, grep="ERROR")
    assert [e.message for e in entries] == ["ERROR 7", "ERROR 8", "ERROR 9"]


async def test_ascending_branch_takes_an_explicit_limit(queries):
    """The stream's per-poll chunk: the resume path is bounded too."""
    job = await _service(queries)
    for i in range(10):
        await queries.append_log(job.id, f"line {i}")

    entries = await queries.get_logs(job.id, since_id=0, limit=4)
    assert [e.message for e in entries] == ["line 0", "line 1", "line 2", "line 3"]


async def test_since_ts_bounds_by_stored_timestamp_format(queries):
    job = await _service(queries)
    await queries.append_log(job.id, "old line")
    await queries.append_log(job.id, "new line")
    await queries._db.conn.execute(
        "UPDATE job_logs SET timestamp = '2026-01-01 00:00:00' WHERE message = 'old line'"
    )
    await queries._db.conn.execute(
        "UPDATE job_logs SET timestamp = '2026-06-01 00:00:00' WHERE message = 'new line'"
    )
    await queries._db.conn.commit()

    entries = await queries.get_logs(job.id, since_ts="2026-03-01 00:00:00")
    assert [e.message for e in entries] == ["new line"]


async def test_unfiltered_read_is_unchanged(queries):
    """The pre-P24 shape must survive verbatim — this is the hot path."""
    job = await _service(queries)
    await queries.append_log(job.id, "a")
    await queries.append_log(job.id, "b", stream=LogStream.stderr)

    entries = await queries.get_logs(job.id)
    assert [(e.message, e.stream) for e in entries] == [
        ("a", LogStream.stdout),
        ("b", LogStream.stderr),
    ]


# --- the stream ---------------------------------------------------------------


class _StubRequest:
    """Minimal ``Request`` surface the frame generator touches.

    ``disconnect_after`` lets a test end an otherwise endless stream
    deterministically instead of racing a sleep.
    """

    def __init__(self, *, disconnect_after: int | None = None, headers: dict | None = None):
        self.headers = headers or {}
        self._checks = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._checks += 1
        if self._disconnect_after is None:
            return False
        return self._checks > self._disconnect_after


class _App:
    """The one ``app.state`` attribute the paged log route reaches for."""

    class state:  # noqa: N801
        pass


async def _no_sleep(_seconds: float) -> None:
    """Collapse the follower's 1 s poll interval so a test does not wait on it."""


async def _drain(gen, limit=50):
    frames = []
    try:
        async for frame in gen:
            frames.append(frame)
            if len(frames) >= limit:
                break
    finally:
        await gen.aclose()
    return frames


@pytest.fixture(autouse=True)
def _no_poll_sleep(monkeypatch):
    """Idle polls cost nothing, so a multi-iteration test is instant."""
    monkeypatch.setattr(stream_mod, "_POLL_INTERVAL_S", 0)


async def test_stream_emits_a_frame_per_line_then_ends_on_a_terminal_row(queries):
    job = await _service(queries, status=JobStatus.running)
    await queries.append_log(job.id, "hello")
    await queries.append_log(job.id, "goodbye", stream=LogStream.stderr)
    await queries.update_job_status(job.id, JobStatus.failed)

    frames = await _drain(
        _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
    )

    assert [f["event"] for f in frames] == ["log", "log", "logs.end"]
    assert json.loads(frames[0]["data"])["message"] == "hello"
    assert json.loads(frames[1]["data"])["stream"] == "stderr"
    assert json.loads(frames[-1]["data"]) == {"type": "logs.end", "status": "failed"}
    # Every log frame carries its row id, which is what makes a resume exact.
    assert [f["id"] for f in frames[:2]] == [str(e.id) for e in await queries.get_logs(job.id)]


async def test_stream_delivers_the_dying_lines_before_the_end_frame(queries):
    """Terminal status alone must not end the stream — the cursor must be empty too.

    This is the crash case: a container's last words are written as the row goes
    terminal, and a stream that hangs up on the status change loses exactly the
    lines the operator opened it for.
    """
    job = await _service(queries, status=JobStatus.failed)
    for i in range(5):
        await queries.append_log(job.id, f"dying {i}")

    frames = await _drain(
        _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
    )
    assert [f["event"] for f in frames] == ["log"] * 5 + ["logs.end"]


async def test_final_drain_loops_past_a_full_page(queries, monkeypatch):
    """The dying lines can outnumber one page, and a single drain drops the rest.

    The gap between the empty page and the status read is not bounded by
    ``_POLL_CHUNK``: a crash can flush thousands of buffered lines into it. The
    drain therefore keeps paging until a short page instead of reading once.
    """
    monkeypatch.setattr(stream_mod, "_POLL_CHUNK", 5)
    job = await _service(queries, status=JobStatus.failed)

    inner = queries.get_job
    injected = {"done": False}

    async def racing_get_job(job_id):
        if not injected["done"]:
            injected["done"] = True
            # Written between the (empty) page and the status read — twelve
            # lines against a five-row page, so one drain cannot carry them.
            for i in range(12):
                await queries.append_log(job_id, f"late {i}")
        return await inner(job_id)

    queries.get_job = racing_get_job  # type: ignore[method-assign]
    try:
        frames = await _drain(
            _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
        )
    finally:
        queries.get_job = inner  # type: ignore[method-assign]

    assert [f["event"] for f in frames] == ["log"] * 12 + ["logs.end"]
    assert [json.loads(f["data"])["message"] for f in frames[:-1]] == [
        f"late {i}" for i in range(12)
    ]
    assert json.loads(frames[-1]["data"])["status"] == "failed"


async def test_terminal_settle_catches_a_line_written_after_the_status_read(queries):
    """Every settle path logs its one trailing line AFTER committing the status.

    ``_handle_crash``/``_teardown_to_stopped``/``_count_restart`` all flip the row
    terminal and only then append "Service exited…"/"Service stopped by user"/
    "Service exhausted N restarts…", a few awaits later on the same loop. Without
    a settle re-poll that line is never delivered — the very line that says why.
    """
    job = await _service(queries, status=JobStatus.failed)

    calls = {"n": 0}
    inner = queries.get_logs

    async def settling_get_logs(job_id, **kw):
        calls["n"] += 1
        entries = await inner(job_id, **kw)
        if calls["n"] == 2:
            await queries.append_log(job_id, "Service exited (code=1)")
        return entries

    queries.get_logs = settling_get_logs  # type: ignore[method-assign]
    try:
        frames = await _drain(
            _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
        )
    finally:
        queries.get_logs = inner  # type: ignore[method-assign]

    assert [f["event"] for f in frames] == ["log", "logs.end"]
    assert json.loads(frames[0]["data"])["message"] == "Service exited (code=1)"


async def test_stream_keeps_streaming_on_degraded(queries):
    """``degraded`` is not terminal — the batch-shaped stream got this wrong."""
    job = await _service(queries, status=JobStatus.degraded)
    await queries.append_log(job.id, "health check failed")

    frames = await _drain(
        _log_frames(
            # Two idle polls, then the client "goes away" — without the
            # disconnect this generator would poll forever, which is the point.
            _StubRequest(disconnect_after=3),
            queries,
            job.id,
            since_id=0,
            grep=None,
            since_ts=None,
        )
    )
    assert [f["event"] for f in frames] == ["log"]
    assert not any(f["event"] == "logs.end" for f in frames)


async def test_stream_keeps_streaming_on_restarting(queries):
    job = await _service(queries, status=JobStatus.restarting)
    frames = await _drain(
        _log_frames(
            _StubRequest(disconnect_after=2), queries, job.id, since_id=0, grep=None, since_ts=None
        )
    )
    assert frames == []


async def test_stream_ends_when_the_row_disappears(queries):
    """A deleted service must not strand the connection polling a ghost."""
    frames = await _drain(
        _log_frames(_StubRequest(), queries, "gone-id", since_id=0, grep=None, since_ts=None)
    )
    assert frames == [
        {"event": "logs.end", "data": json.dumps({"type": "logs.end", "status": "deleted"})}
    ]


async def test_stream_resumes_from_since_id(queries):
    job = await _service(queries, status=JobStatus.stopped)
    await queries.append_log(job.id, "first")
    await queries.append_log(job.id, "second")
    entries = await queries.get_logs(job.id)

    frames = await _drain(
        _log_frames(
            _StubRequest(), queries, job.id, since_id=entries[0].id, grep=None, since_ts=None
        )
    )
    assert [json.loads(f["data"])["message"] for f in frames if f["event"] == "log"] == ["second"]


async def test_stream_applies_grep(queries):
    job = await _service(queries, status=JobStatus.stopped)
    await queries.append_log(job.id, "boring")
    await queries.append_log(job.id, "ERROR boom")

    frames = await _drain(
        _log_frames(_StubRequest(), queries, job.id, since_id=0, grep="ERROR", since_ts=None)
    )
    assert [json.loads(f["data"])["message"] for f in frames if f["event"] == "log"] == [
        "ERROR boom"
    ]


async def test_a_never_matching_grep_advances_the_cursor(queries):
    """A filter that matches nothing must not re-scan the backlog every poll.

    No matching row means no row to advance the cursor with, so before the
    watermark the ``since_id`` of every consecutive poll was identical and each
    1 s tick re-walked the whole remaining range — bounded rows returned,
    unbounded rows scanned, on the shared connection, times 32 streams.
    """
    job = await _service(queries, status=JobStatus.degraded)
    for i in range(20):
        await queries.append_log(job.id, f"boring {i}")
    entries = await queries.get_logs(job.id)

    seen_since: list[int] = []
    inner = queries.get_logs

    async def spy(job_id, since_id=0, **kw):
        seen_since.append(since_id)
        return await inner(job_id, since_id=since_id, **kw)

    queries.get_logs = spy  # type: ignore[method-assign]
    try:
        frames = await _drain(
            _log_frames(
                _StubRequest(disconnect_after=3),
                queries,
                job.id,
                since_id=0,
                grep="NEVER-MATCHES",
                since_ts=None,
            )
        )
    finally:
        queries.get_logs = inner  # type: ignore[method-assign]

    assert frames == []
    assert len(seen_since) >= 2, seen_since
    # First poll starts where the client asked; every later one starts at the
    # watermark, so the scanned range collapses instead of repeating.
    assert seen_since[0] == 0
    assert seen_since[1] == entries[-1].id
    assert seen_since[1:] == [entries[-1].id] * (len(seen_since) - 1)


async def test_a_row_written_after_the_watermark_is_still_delivered(queries):
    """The watermark is captured BEFORE the scan, so nothing can be skipped.

    A row inserted between the ``MAX(id)`` read and the ``get_logs`` call gets a
    strictly larger id than the watermark, so the next poll still sees it.
    """
    job = await _service(queries, status=JobStatus.degraded)
    await queries.append_log(job.id, "boring")

    inner = queries.max_log_id
    injected = {"done": False}

    async def racing_max_log_id():
        value = await inner()
        if not injected["done"]:
            injected["done"] = True
            # Written strictly AFTER the watermark was read.
            await queries.append_log(job.id, "ERROR late arrival")
        return value

    queries.max_log_id = racing_max_log_id  # type: ignore[method-assign]
    try:
        frames = await _drain(
            _log_frames(
                _StubRequest(disconnect_after=4),
                queries,
                job.id,
                since_id=0,
                grep="ERROR",
                since_ts=None,
            )
        )
    finally:
        queries.max_log_id = inner  # type: ignore[method-assign]

    assert [json.loads(f["data"])["message"] for f in frames if f["event"] == "log"] == [
        "ERROR late arrival"
    ]


async def test_an_unfiltered_stream_never_reads_the_watermark(queries):
    """The extra ``MAX(id)`` read is filtered-path only — no cost on the hot path."""
    job = await _service(queries, status=JobStatus.stopped)
    await queries.append_log(job.id, "hello")

    calls = []
    inner = queries.max_log_id

    async def spy():
        calls.append(1)
        return await inner()

    queries.max_log_id = spy  # type: ignore[method-assign]
    try:
        await _drain(
            _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
        )
    finally:
        queries.max_log_id = inner  # type: ignore[method-assign]

    assert calls == []


async def test_stream_refuses_once_saturated(queries, monkeypatch):
    """A saturated daemon answers one branchable frame and closes.

    An SSE connection is held open indefinitely, so an uncapped stream route is
    a cheap connection-exhaustion vector on a LAN daemon.
    """
    import asyncio

    monkeypatch.setattr(stream_mod, "_STREAM_SEMAPHORE", asyncio.Semaphore(1))
    job = await _service(queries, status=JobStatus.stopped)

    held = _log_frames(_StubRequest(), queries, job.id, since_id=0, grep=None, since_ts=None)
    async with stream_mod._STREAM_SEMAPHORE:
        frames = await _drain(held)

    assert len(frames) == 1
    body = json.loads(frames[0]["data"])
    assert frames[0]["event"] == "logs.saturated"
    assert body["reason"] == "concurrency_limit"


def test_last_event_id_header_parses_through_the_one_shared_parser():
    """A browser ``EventSource`` resends the header by itself; it must win.

    Both stream routes read it through :func:`parse_resume_cursor`; an unusable
    value comes back ``None`` so the route falls back to its own ``since_id``.
    """
    assert parse_resume_cursor(_StubRequest(headers={"last-event-id": "42"})) == 42
    # A malformed value is IGNORED, not rejected: 400-ing a reconnect turns a
    # cosmetic client bug into a permanently broken stream.
    assert parse_resume_cursor(_StubRequest(headers={"last-event-id": "junk"})) is None
    assert parse_resume_cursor(_StubRequest(headers={"last-event-id": "-1"})) is None
    assert parse_resume_cursor(_StubRequest()) is None


async def test_an_oversized_last_event_id_is_treated_as_absent(queries, monkeypatch):
    """Regression: the log stream used to hand a >64-bit cursor to the sqlite bind.

    The events stream had always rejected it; this one only checked ``>= 0``, so
    a bogus ``Last-Event-ID`` reached ``get_logs``' keyset bind and raised
    ``OverflowError`` inside a generator whose only failure mode is a dead
    stream. Both routes now share one parser, so the bound cannot drift again:
    the header is treated as absent and the explicit ``since_id`` stands.
    """

    class _App:
        class state:  # noqa: N801 — mimics ``app.state``
            pass

    request = _StubRequest(headers={"last-event-id": str(2**70)})
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    await _service(queries, name="huge")

    seen: dict = {}

    def spy(request_, queries_, job_id, **kw):
        seen.update(kw)

        async def _empty():
            return
            yield  # pragma: no cover — never reached

        return _empty()

    monkeypatch.setattr(stream_mod, "_log_frames", spy)
    await stream_service_logs(request, "huge", since_id=11, grep=None, since=None)
    assert seen["since_id"] == 11


async def test_stream_route_returns_an_event_source_response(queries):
    """Route wiring, without opening an endless body over HTTP."""

    class _App:
        class state:  # noqa: N801 — mimics ``app.state``
            pass

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    job = await _service(queries, name="wired")

    resp = await stream_service_logs(request, "wired", since_id=0, grep=None, since=None)
    assert isinstance(resp, EventSourceResponse)
    assert job.service_name == "wired"


async def test_paged_route_forwards_grep_and_normalized_since(queries, monkeypatch):
    """The params must reach the query layer, and ``since`` must be normalized.

    An unwired param makes the read look filtered when it is not — the same
    failure mode as an unwidened MCP tool, one layer down. And an ISO-8601
    ``since`` passed through raw would compare a 'T'-separated string against a
    space-separated column and quietly answer nonsense.
    """
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    await _service(queries, name="paged")

    seen: dict = {}

    async def spy(job_id, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(queries, "get_logs", spy)
    await get_service_logs(
        request,
        Response(),
        "paged",
        since_id=0,
        tail=50,
        grep="100%",
        since="2026-08-07T10:00:00Z",
    )
    assert seen["grep"] == "100%"
    assert seen["since_ts"] == "2026-08-07 10:00:00"
    assert seen["tail"] == 50


# --- the paged route's scan watermark (the --follow --grep re-scan) ----------


async def test_a_filtered_page_returns_the_pre_scan_watermark(queries, monkeypatch):
    """A grep that matches nothing must still tell the follower where to resume.

    No matching row means no id to advance ``since_id`` with, so every 1 s poll
    re-scanned the same growing range server-side — bounded rows returned,
    unbounded rows scanned. The watermark is the pre-scan ``MAX(id)``: anything
    written after that read has a strictly larger id, so skipping to it is safe.
    """
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    job = await _service(queries, name="wm")
    for i in range(5):
        await queries.append_log(job.id, f"line {i}")

    scanned: list[int] = []
    inner = queries.get_logs

    async def spy(job_id, **kw):
        scanned.append(kw.get("since_id", 0))
        return await inner(job_id, **kw)

    monkeypatch.setattr(queries, "get_logs", spy)

    response = Response()
    entries = await get_service_logs(
        request, response, "wm", since_id=0, tail=None, grep="nope", since=None
    )
    assert entries == []
    watermark = int(response.headers["X-Nerdit-Scan-Watermark"])
    assert watermark == await queries.max_log_id() == 5

    # The follower resumes from it, so the second poll re-scans nothing old.
    await get_service_logs(
        request, Response(), "wm", since_id=watermark, tail=None, grep="nope", since=None
    )
    assert scanned == [0, 5]


async def test_an_unfiltered_page_carries_no_watermark_header(queries):
    """The no-filter response stays byte-identical, header included (absent)."""
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    job = await _service(queries, name="plain")
    await queries.append_log(job.id, "hello")

    response = Response()
    entries = await get_service_logs(
        request, response, "plain", since_id=0, tail=None, grep=None, since=None
    )
    assert [e.message for e in entries] == ["hello"]
    assert "X-Nerdit-Scan-Watermark" not in response.headers

    # ``tail`` walks backwards from the newest row, so no watermark there either.
    tailed = Response()
    await get_service_logs(request, tailed, "plain", since_id=0, tail=10, grep="hello", since=None)
    assert "X-Nerdit-Scan-Watermark" not in tailed.headers


async def _bulk_logs(queries, job_id: str, count: int) -> None:
    """Insert *count* stdout lines in one statement (20k append_log calls is a minute)."""
    await queries._db.conn.executemany(
        "INSERT INTO job_logs (job_id, stream, message) VALUES (?, 'stdout', ?)",
        [(job_id, f"line {i}") for i in range(count)],
    )
    await queries._db.conn.commit()


async def test_a_parameterless_page_is_capped_and_resumable(queries):
    """M3: the bare documented call must be bounded, and the rest reachable.

    With no ``tail`` the forward branch used to run without a LIMIT, so one
    ~50-byte GET materialized the whole retained history (measured: 55.9 MB of
    JSON at 200k rows). The cap is the page size, and ``since_id`` is the way
    past it — the same contract the SSE twin drains on.
    """
    from nerdit.daemon.limits import _MAX_LOG_TAIL
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    job = await _service(queries, name="chatty")
    await _bulk_logs(queries, job.id, 20_000)

    first = await get_service_logs(
        request, Response(), "chatty", since_id=0, tail=None, grep=None, since=None
    )
    assert len(first) == _MAX_LOG_TAIL
    assert first[0].message == "line 0"

    second = await get_service_logs(
        request, Response(), "chatty", since_id=first[-1].id, tail=None, grep=None, since=None
    )
    assert len(second) == _MAX_LOG_TAIL
    assert second[0].id == first[-1].id + 1  # the remainder, not a re-read


async def test_a_capped_filtered_page_withholds_the_watermark(queries):
    """A full page stopped at the BOUND, so the pre-scan max is not decided yet.

    Handing it to the follower would jump the cursor past the undrained
    remainder — the cap would silently eat the backlog it was added to bound.
    """
    from nerdit.daemon.limits import _MAX_LOG_TAIL
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    job = await _service(queries, name="grepped")
    await _bulk_logs(queries, job.id, _MAX_LOG_TAIL + 100)

    full = Response()
    entries = await get_service_logs(
        request, full, "grepped", since_id=0, tail=None, grep="line", since=None
    )
    assert len(entries) == _MAX_LOG_TAIL
    assert "X-Nerdit-Scan-Watermark" not in full.headers

    # The short page that finishes the range does carry it again.
    short = Response()
    rest = await get_service_logs(
        request, short, "grepped", since_id=entries[-1].id, tail=None, grep="line", since=None
    )
    assert len(rest) == 100
    assert "X-Nerdit-Scan-Watermark" in short.headers


async def test_the_cli_drains_every_capped_page(monkeypatch):
    """The CLI half of M3: one bounded page is no longer the whole log."""
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    pages = [
        [
            {"id": i, "stream": "stdout", "message": f"a{i}", "timestamp": "2026-01-01T00:00:00"}
            for i in range(1, 6)
        ],
        [
            {"id": i, "stream": "stdout", "message": f"b{i}", "timestamp": "2026-01-01T00:00:00"}
            for i in range(6, 9)
        ],
        [],
    ]
    cursors: list[int] = []

    class _Fake:
        async def get_service_logs(self, ident, since_id=0, **kw):
            cursors.append(since_id)
            return pages.pop(0)

    shown: list[int] = []
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _Fake())
    monkeypatch.setattr(
        logs_mod, "display_logs", lambda entries: shown.extend(e["id"] for e in entries)
    )

    await logs_mod._logs_async("demo", False)
    assert cursors == [0, 5, 8]  # resumed from the last id of each page
    assert shown == list(range(1, 9))  # nothing dropped at a page boundary


async def test_the_cli_follower_advances_to_the_scan_watermark(monkeypatch):
    """The CLI half of the fix: an all-filtered page still moves the cursor."""
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    cursors: list[int] = []

    class _Fake:
        def __init__(self) -> None:
            self._polls = 0

        async def get_service_logs_page(self, ident, *, since_id=0, grep=None, since=None):
            cursors.append(since_id)
            self._polls += 1
            return [], 42 + self._polls

        async def get_service_logs(self, ident, since_id=0, **kw):
            cursors.append(since_id)
            return []

        async def get_service(self, ident):
            # Terminal on the SECOND poll, so exactly one advance is observable.
            return {"status": "running" if self._polls < 2 else "stopped"}

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _Fake())
    monkeypatch.setattr(logs_mod.asyncio, "sleep", _no_sleep)
    await logs_mod._logs_async("demo", True, grep="nope")
    # 0 → 43 (first watermark) → 44 (second), then the terminal drain reuses it.
    assert cursors == [0, 43, 44]


async def test_the_follower_drains_a_full_backlog_without_a_status_poll_between_pages(
    monkeypatch,
):
    """A page that fills the server cap means "there is more backlog".

    Taking one capped page per 1 s status poll put a chatty service's retained
    history minutes ahead of its first LIVE line, so a full page must loop
    straight back into the next read.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod
    from nerdit.daemon.limits import _MAX_LOG_TAIL

    full = [
        {"id": i, "stream": "stdout", "message": "backlog", "timestamp": "2026-01-01T00:00:00"}
        for i in range(1, _MAX_LOG_TAIL + 1)
    ]
    tail = [
        {"id": 99999, "stream": "stdout", "message": "live", "timestamp": "2026-01-01T00:00:00"}
    ]
    calls: list[str] = []
    shown: list[str] = []

    class _Fake:
        def __init__(self) -> None:
            self.pages = [full, tail]

        async def get_service_logs_page(self, ident, *, since_id=0, grep=None, since=None):
            calls.append("page")
            return (self.pages.pop(0) if self.pages else []), None

        async def get_service_logs(self, ident, since_id=0, **kw):
            return []

        async def get_service(self, ident):
            calls.append("status")
            return {"status": "stopped"}

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _Fake())
    monkeypatch.setattr(
        logs_mod, "display_logs", lambda entries: shown.extend(e["message"] for e in entries)
    )
    await logs_mod._logs_async("demo", True)

    # Two reads back to back, no status poll wedged between them: the capped
    # page was drained straight into the next read.
    assert calls == ["page", "page", "status"]
    # And the live line arrived through the follower itself.
    assert shown[-1] == "live"


# --- CLI + client surface parity ----------------------------------------------


async def test_cli_logs_forwards_the_filter_flags(monkeypatch):
    """``--grep``/``--since`` must reach the server, not be filtered client-side.

    Filtering in the CLI would pull the whole log across the wire to throw most
    of it away — the exact cost the SQL-side filter exists to avoid.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    seen: list[dict] = []

    class _Fake:
        async def get_service_logs(self, ident, since_id=0, **kw):
            seen.append({"ident": ident, **kw})
            return []

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _Fake())
    await logs_mod._logs_async("demo", False, grep="100%", since="2026-08-07T10:00:00Z")
    assert seen == [{"ident": "demo", "grep": "100%", "since": "2026-08-07T10:00:00Z"}]


async def test_cli_logs_sends_no_filter_params_when_unset(monkeypatch):
    """An unfiltered read stays byte-identical to the pre-P24 request.

    The CLI passes the flags straight through; the ONE omission rule lives in
    the client (pinned by ``test_client_get_service_logs_omits_empty_filters``
    below), so an unset flag never reaches the query string.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import logs as logs_mod

    seen: list[dict] = []

    class _Fake:
        async def get_service_logs(self, ident, since_id=0, **kw):
            seen.append(kw)
            return []

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _Fake())
    await logs_mod._logs_async("demo", False)
    assert seen == [{"grep": None, "since": None}]


async def test_client_get_service_logs_omits_empty_filters():
    from nerdit.cli.client import NerditClient

    captured: dict = {}

    class _Resp:
        headers: dict = {}

        def raise_for_status(self):
            pass

        def json(self):
            return []

    class _HTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None, timeout=None):
            captured.update(params or {})
            return _Resp()

    client = NerditClient("http://x")
    object.__setattr__(client, "_client", lambda: _HTTP())
    await client.get_service_logs("svc", since_id=3, tail=10, grep="boom")
    assert captured == {"since_id": 3, "tail": 10, "grep": "boom"}


async def test_mcp_service_logs_forwards_the_filters():
    """The WS3 lesson: FastMCP silently drops unknown kwargs.

    An impl that accepts ``grep`` but never passes it on would advertise a
    filter in its schema and answer unfiltered rows — worse than not having it,
    because the agent believes the narrowing happened.
    """
    from nerdit.mcp.tools.services import _service_logs_impl

    seen: dict = {}

    class _Fake:
        async def get_service_logs(self, ident, since_id=0, tail=None, **kw):
            seen.update({"ident": ident, "tail": tail, **kw})
            return []

    await _service_logs_impl(_Fake(), "svc", grep="Traceback", since="2026-08-07T10:00:00Z")
    assert seen["grep"] == "Traceback"
    assert seen["since"] == "2026-08-07T10:00:00Z"


async def test_mcp_get_audit_forwards_the_filters():
    from nerdit.mcp.tools.system import _get_audit_impl

    seen: dict = {}

    class _Fake:
        async def get_audit(self, **kw):
            seen.update(kw)
            return {}

    await _get_audit_impl(
        _Fake(),
        principal_id="tok-1",
        action_prefix="deploy.",
        since="2026-08-07T10:00:00Z",
        until="2026-08-07T12:00:00Z",
    )
    assert seen["principal_id"] == "tok-1"
    assert seen["action_prefix"] == "deploy."
    assert seen["since"] == "2026-08-07T10:00:00Z"
    assert seen["until"] == "2026-08-07T12:00:00Z"


async def test_stream_route_rejects_an_unparseable_since(queries):
    from nerdit.daemon.errors import NerditError

    class _App:
        class state:  # noqa: N801
            pass

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    await _service(queries, name="wired2")

    with pytest.raises(NerditError) as exc:
        await stream_service_logs(request, "wired2", since_id=0, grep=None, since="yesterday")
    assert exc.value.code == "bad_request"


# --- P34: source — build output and app output are different streams ---------
#
# The field failure: an agent grepping a deployed app's logs kept matching
# BuildKit layer lines ("404B", "CACHED", "sha256:…") because the build wrote
# to ``stdout``, the same stream the app writes to. The fix is a stream value of
# its own plus a filter that can express "the app, not the build".


async def _mixed_logs(queries, name="mixed"):
    """One service carrying every stream the daemon writes."""
    job = await _service(queries, name=name)
    await queries.append_log(job.id, "Building image nerdit-app/mixed:1 ...", LogStream.system)
    await queries.append_log(job.id, " => transferring context: 404B", LogStream.build)
    await queries.append_log(job.id, " => CACHED [2/5] RUN pip install", LogStream.build)
    await queries.append_log(job.id, "listening on 8000", LogStream.stdout)
    await queries.append_log(job.id, "Traceback (most recent call last)", LogStream.stderr)
    await queries.append_log(job.id, "final gasp", LogStream.crash)
    return job


async def test_source_runtime_excludes_the_build_output(queries):
    """The reported bug, expressed as a test: 404B must not be reachable."""
    job = await _mixed_logs(queries)

    entries = await queries.get_logs(
        job.id, streams=(LogStream.stdout, LogStream.stderr, LogStream.crash)
    )

    messages = [e.message for e in entries]
    assert messages == ["listening on 8000", "Traceback (most recent call last)", "final gasp"]
    assert not any("404B" in m for m in messages)


async def test_source_build_returns_only_the_builder_output(queries):
    job = await _mixed_logs(queries)

    entries = await queries.get_logs(job.id, streams=(LogStream.build,))

    assert [e.message for e in entries] == [
        " => transferring context: 404B",
        " => CACHED [2/5] RUN pip install",
    ]


async def test_no_stream_filter_is_the_pre_p34_read(queries):
    """``streams=None`` must not add a clause — every line, in id order."""
    job = await _mixed_logs(queries)

    assert len(await queries.get_logs(job.id, streams=None)) == 6


async def test_an_empty_stream_set_matches_nothing_rather_than_everything(queries):
    """``IN ()`` is not valid SQLite; the fail-closed spelling must not widen."""
    job = await _mixed_logs(queries)

    assert await queries.get_logs(job.id, streams=()) == []


async def test_the_stream_filter_composes_with_grep_inside_the_tail_bound(queries):
    """The point of the pair: "the last N matching lines from the APP"."""
    job = await _service(queries, name="composed")
    for i in range(5):
        await queries.append_log(job.id, f" => CACHED error {i}", LogStream.build)
        await queries.append_log(job.id, f"error {i} from the app", LogStream.stdout)

    entries = await queries.get_logs(
        job.id, grep="error", tail=2, streams=(LogStream.stdout, LogStream.stderr, LogStream.crash)
    )

    assert [e.message for e in entries] == ["error 3 from the app", "error 4 from the app"]


async def test_route_maps_source_to_the_stream_set(queries, monkeypatch):
    """The route owns the vocabulary; the query layer only sees stream values."""
    from nerdit.daemon.routes.services import get_service_logs

    request = _StubRequest()
    request.app = _App()  # type: ignore[attr-defined]
    request.app.state.queries = queries  # type: ignore[attr-defined]
    await _service(queries, name="sourced")

    seen: dict = {}

    async def spy(job_id, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(queries, "get_logs", spy)

    for source, expected in (
        ("all", None),
        ("build", (LogStream.build,)),
        ("runtime", (LogStream.stdout, LogStream.stderr, LogStream.crash)),
    ):
        await get_service_logs(
            request,
            Response(),
            "sourced",
            since_id=0,
            tail=50,
            grep=None,
            since=None,
            source=source,
        )
        assert seen["streams"] == expected, source


async def test_route_rejects_an_unknown_source_at_the_http_boundary(queries):
    """FastAPI's Literal owns the validation — a bad value never reaches SQL."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from nerdit.daemon.routes.services import router as services_router

    app = FastAPI()
    app.include_router(services_router, prefix="/api")
    app.state.queries = queries
    job = await _service(queries, name="bounded")
    await queries.append_log(job.id, "hi")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        bad = await client.get("/api/services/bounded/logs?source=everything")
        ok = await client.get("/api/services/bounded/logs?source=runtime")

    assert bad.status_code == 422
    assert ok.status_code == 200
