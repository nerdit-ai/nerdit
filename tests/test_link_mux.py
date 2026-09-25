"""Test inbound stream limits, credentials, HTTP replay and streaming.

Validate relay Authorization and X-Nerdit-Role before serving each stream.
Use real httpx with in-process transports to exercise headers and body handling.
ASGITransport buffers complete responses: streaming, cancellation and live
concurrency tests must use _ParkedTransport, which yields then waits on a
controlled event. No sockets are opened.
Validate emitted frames against independent extra-forbid oracle models.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from nerdit.core.link import mux as mux_module
from nerdit.core.link.client import AppTarget, CloseInfo, LinkClosedError, MuxContext
from nerdit.core.link.frames import (
    ERROR_CODE_AUTH,
    ERROR_CODE_INTERNAL,
    CloseStream,
    OpenStream,
    StreamBody,
    StreamEnd,
    StreamLimits,
    encode_body,
    parse_downlink,
)
from nerdit.core.link.mux import (
    APP_HEADER,
    CLOUD_CONTROL_HEADER,
    HOP_BY_HOP,
    RESPONSE_CHUNK_BYTES,
    StreamMux,
)
from tests.link_fake_relay import FIXTURE_NODE_ID, load_fixture, open_stream_frame
from tests.node_link_frames import parse_outbound

#: The capability the manager would have proven on the live connection.
TOKEN = "live-capability-token-not-a-secret"  # noqa: S105 - a test literal

#: The §8 limits, read from the frozen ``hello_ack`` rather than hardcoded —
#: the daemon obeys what the relay advertised, and so does this harness.
ACK_LIMITS = StreamLimits.from_payload(load_fixture("hello_ack.json")["limits"])


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class Collector:
    """Records uplink frames, validating each under the conformance oracle."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, frame: Any) -> None:
        if self.closed:
            raise LinkClosedError(
                CloseInfo(code=1000, reason="gone", error_code=None, cause="local")
            )
        payload = json.loads(json.dumps(frame))
        parse_outbound(payload)  # the relay would refuse anything this rejects
        self.frames.append(payload)

    def types(self) -> list[str]:
        return [frame["type"] for frame in self.frames]

    def of(self, frame_type: str) -> list[dict[str, Any]]:
        return [frame for frame in self.frames if frame["type"] == frame_type]

    def body(self) -> bytes:
        return b"".join(
            base64.b64decode(frame["body_b64"]) for frame in self.of("stream_response_body")
        )

    def only_error(self) -> dict[str, Any]:
        errors = self.of("stream_error")
        assert len(errors) == 1, f"expected exactly one stream_error, got {self.types()}"
        return errors[0]

    async def wait(self, predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"timed out; frames so far: {self.types()}")
            await asyncio.sleep(0.001)

    async def wait_types(self, *types: str, timeout: float = 3.0) -> None:
        await self.wait(lambda: self.types()[: len(types)] == list(types), timeout=timeout)

    async def settle(self) -> None:
        """Yield enough for one stream's task chain to finish."""
        for _ in range(50):
            await asyncio.sleep(0)


class _FakeMonotonic:
    """A monotonic clock the pacer's sleeps advance, so pacing costs no time."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


def _ctx(  # noqa: PLR0913 - one keyword per injected seam, by design
    transport: httpx.AsyncBaseTransport,
    *,
    limits: StreamLimits | None = None,
    validate: Callable[[str, str], bool] | None = None,
    collector: Collector | None = None,
    monotonic: _FakeMonotonic | None = None,
    resolve_app: Callable[[str], Any] | None = None,
    app_transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[MuxContext, Collector]:
    sink = collector or Collector()
    clock = monotonic or _FakeMonotonic()
    return (
        MuxContext(
            # Uplink frames carry OUR handshake-authoritative node id, not an
            # echo of whatever the downlink frame claimed.
            node_id=FIXTURE_NODE_ID,
            limits=limits or ACK_LIMITS,
            send=sink.send,
            validate_capability=validate
            or (lambda token, role: token == TOKEN and role == "submitter"),
            loopback_base_url="http://127.0.0.1:9321",
            http_client_factory=lambda: httpx.AsyncClient(
                transport=transport, base_url="http://127.0.0.1:9321"
            ),
            resolve_app=resolve_app,
            # (P26) The app client gets NO base_url, exactly like production:
            # an app stream's authority is per-stream, so a relative target is
            # not even expressible on this client.
            app_client_factory=(
                None
                if app_transport is None
                else (lambda: httpx.AsyncClient(transport=app_transport))
            ),
            monotonic=clock,
            sleep=clock.sleep,
        ),
        sink,
    )


def _frame(**kw: Any) -> OpenStream:
    """Build an ``open_stream`` and parse it, so the frame under test is real."""
    kw.setdefault("token", TOKEN)
    kw.setdefault("stream_id", "s1")
    parsed = parse_downlink(json.dumps(open_stream_frame(**kw)))
    assert isinstance(parsed, OpenStream)
    return parsed


def _raw_frame(payload: dict[str, Any]) -> OpenStream:
    parsed = parse_downlink(json.dumps(payload))
    assert isinstance(parsed, OpenStream)
    return parsed


# --- transports ------------------------------------------------------------


class _CountingApp:
    """An ASGI app that records every request it is asked to serve.

    ``hits`` is the whole point of the credential tests: an unvalidated stream
    must leave it at zero.
    """

    def __init__(
        self,
        status: int = 200,
        body: bytes = b'{"ok":true}',
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        self.hits = 0
        self.requests: list[dict[str, Any]] = []
        self._status = status
        self._body = body
        self._headers = headers or [(b"content-type", b"application/json")]

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        self.hits += 1
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        self.requests.append(
            {
                "method": scope["method"],
                "path": scope["path"],
                "query": scope["query_string"].decode(),
                "headers": [(name.decode(), value.decode()) for name, value in scope["headers"]],
                "body": body,
            }
        )
        await send(
            {"type": "http.response.start", "status": self._status, "headers": self._headers}
        )
        await send({"type": "http.response.body", "body": self._body})

    def header(self, name: str) -> str | None:
        for key, value in self.requests[-1]["headers"]:
            if key.lower() == name.lower():
                return value
        return None


class _ParkedStream(httpx.AsyncByteStream):
    """A body that yields one chunk, then waits for the test to let it finish."""

    def __init__(self, first: bytes, rest: list[bytes], gate: asyncio.Event) -> None:
        self._first = first
        self._rest = rest
        self._gate = gate
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        await self._gate.wait()
        for chunk in self._rest:
            yield chunk

    async def aclose(self) -> None:
        self.closed.set()


class _ParkedTransport(httpx.AsyncBaseTransport):
    """Returns a response whose body is parked mid-stream.

    The only way to observe "the head and the first chunk left before the
    origin finished", which ``ASGITransport`` structurally cannot show.
    """

    def __init__(
        self,
        first: bytes = b"event: a\ndata: 1\n\n",
        rest: list[bytes] | None = None,
        headers: list[tuple[str, str]] | None = None,
        status: int = 200,
    ) -> None:
        self.first = first
        self.rest = rest if rest is not None else [b"event: b\ndata: 2\n\n"]
        self.headers = headers or [("content-type", "text/event-stream")]
        self.status = status
        self.gates: list[asyncio.Event] = []
        self.streams: list[_ParkedStream] = []
        self.hits = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.hits += 1
        gate = asyncio.Event()
        stream = _ParkedStream(self.first, self.rest, gate)
        self.gates.append(gate)
        self.streams.append(stream)
        return httpx.Response(self.status, headers=self.headers, stream=stream)

    def release(self, index: int = 0) -> None:
        self.gates[index].set()

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()


class _BytesTransport(httpx.AsyncBaseTransport):
    """Returns exactly the status/headers/body given — no httpx synthesis.

    Used where the *head frame* must equal a fixture byte for byte, which an
    ASGI app cannot do (Starlette adds its own ``content-length``).
    """

    def __init__(self, status: int, headers: list[tuple[str, str]], chunks: list[bytes]) -> None:
        self.status = status
        self.headers = headers
        self.chunks = chunks
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        outer = self

        class _Stream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                for chunk in outer.chunks:
                    yield chunk

            async def aclose(self) -> None:
                return None

        return httpx.Response(self.status, headers=self.headers, stream=_Stream())


class _BrokenTransport(httpx.AsyncBaseTransport):
    """The daemon's own listener is not answering."""

    def __init__(self) -> None:
        self.hits = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.hits += 1
        raise httpx.ConnectError("connection refused", request=request)


# ---------------------------------------------------------------------------
# checklist item 3 — the credential gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        (
            "no authorization header at all",
            open_stream_frame(stream_id="s1"),
        ),
        (
            "no x-nerdit-role header",
            open_stream_frame(stream_id="s1", headers=[("authorization", f"Bearer {TOKEN}")]),
        ),
        (
            "wrong x-nerdit-role",
            open_stream_frame(
                stream_id="s1",
                headers=[
                    ("authorization", f"Bearer {TOKEN}"),
                    ("x-nerdit-role", "admin"),
                ],
            ),
        ),
        (
            "wrong auth scheme",
            open_stream_frame(
                stream_id="s1",
                headers=[("authorization", f"Token {TOKEN}"), ("x-nerdit-role", "submitter")],
            ),
        ),
        (
            "lowercase bearer prefix",
            open_stream_frame(
                stream_id="s1",
                headers=[("authorization", f"bearer {TOKEN}"), ("x-nerdit-role", "submitter")],
            ),
        ),
        (
            "empty bearer",
            open_stream_frame(
                stream_id="s1",
                headers=[("authorization", "Bearer "), ("x-nerdit-role", "submitter")],
            ),
        ),
        (
            "a token that is not the live capability",
            open_stream_frame(stream_id="s1", token="stale-or-forged"),
        ),
    ],
)
async def test_an_unvalidated_stream_never_touches_the_listener(
    label: str, payload: dict[str, Any]
) -> None:
    """Checklist item 3, the load-bearing half: refuse *before* serving.

    ``app.hits == 0`` is the assertion that matters. Validating after the
    request had already been replayed would mean the daemon executed a
    tunnelled call it then declined to report — the exact hole the check exists
    to close.
    """
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_raw_frame(payload))
    await sink.settle()

    assert app.hits == 0, f"the listener was contacted for: {label}"
    error = sink.only_error()
    assert error["code"] == ERROR_CODE_AUTH
    assert error["message"] == "the tunnel capability is invalid"
    assert error["stream_id"] == "s1"
    assert error["node_id"] == FIXTURE_NODE_ID
    # The refusal says nothing about *which* check failed.
    assert TOKEN not in json.dumps(error)
    assert mux.active_streams() == 0
    await mux.aclose()


async def test_a_non_submitter_frame_role_is_refused() -> None:
    """The relay refuses a role mismatch (``control.py:684``), so a frame that
    claims another role did not come from a conforming relay — and it is
    *answered*, not dropped, so the requester learns why."""
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    payload = open_stream_frame(stream_id="s1", token=TOKEN)
    payload["role"] = "admin"
    await mux.handle(_raw_frame(payload))
    await sink.settle()

    assert app.hits == 0
    assert sink.only_error()["code"] == ERROR_CODE_AUTH
    await mux.aclose()


async def test_validate_capability_is_consulted_with_the_submitter_ceiling() -> None:
    """The mux never asks for a role other than ``submitter`` (D-R2)."""
    asked: list[tuple[str, str]] = []

    def spy(token: str, role: str) -> bool:
        asked.append((token, role))
        return token == TOKEN

    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app), validate=spy)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait_types("stream_response_head")
    await sink.settle()

    assert asked == [(TOKEN, "submitter")]
    await mux.aclose()


# ---------------------------------------------------------------------------
# item 2 — the loopback round trip
# ---------------------------------------------------------------------------


async def test_a_validated_stream_is_replayed_and_answered() -> None:
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="GET", path="/api/status", query="verbose=1"))
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
    await sink.settle()

    assert app.hits == 1
    assert app.requests[0]["method"] == "GET"
    assert app.requests[0]["path"] == "/api/status"
    assert app.requests[0]["query"] == "verbose=1"

    head = sink.of("stream_response_head")[0]
    assert head["status"] == 200
    assert head["stream_id"] == "s1"
    assert head["node_id"] == FIXTURE_NODE_ID
    assert ["content-type", "application/json"] in head["headers"]
    assert sink.body() == b'{"ok":true}'
    assert sink.types() == [
        "stream_response_head",
        "stream_response_body",
        "stream_response_end",
    ]
    assert mux.active_streams() == 0
    await mux.aclose()


async def test_the_injected_credentials_are_forwarded_verbatim() -> None:
    """Item 3's *other* half: this is the audit-provenance path.

    The middleware resolves the same bearer into the ``link:<node_id>``
    submitter principal, so stripping these would silently turn a tunnelled
    request into an anonymous local one — and erase its audit provenance.
    """
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_frame(headers=[("x-nerdit-project-id", "prj_aaaaaaaaaaaaaaaa")]))
    await sink.wait_types("stream_response_head")
    await sink.settle()

    assert app.header("authorization") == f"Bearer {TOKEN}"
    assert app.header("x-nerdit-role") == "submitter"
    assert app.header("x-nerdit-project-id") == "prj_aaaaaaaaaaaaaaaa"
    await mux.aclose()


async def test_hop_by_hop_host_and_content_length_are_stripped_from_the_request() -> None:
    """RFC 7230 §6.1: these describe *this* hop and are wrong on a tunnelled one."""
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    hop_headers = [(name, "x") for name in sorted(HOP_BY_HOP)]
    frame = _frame(
        method="POST",
        path="/api/echo",
        headers=[
            ("accept", "*/*"),
            ("host", "evil.example"),
            ("content-length", "999"),
            *hop_headers,
        ],
        body_b64=encode_body(b"payload"),
    )
    await mux.handle(frame)
    await sink.wait_types("stream_response_head")
    await sink.settle()

    # Not "no hop-by-hop header reaches the app" — httpx sets its own
    # ``connection: keep-alive`` for the loopback hop, which is exactly right.
    # What must not survive is the *tunnelled* hop's values, marked "x" here.
    forwarded = [
        (name.lower(), value)
        for name, value in app.requests[0]["headers"]
        if name.lower() in HOP_BY_HOP
    ]
    assert all(value != "x" for _, value in forwarded), forwarded
    assert {name for name, _ in forwarded} <= {"connection"}
    assert app.header("host") != "evil.example"
    # content-length is recomputed from the body actually assembled.
    assert app.header("content-length") == "7"
    assert app.requests[0]["body"] == b"payload"
    await mux.aclose()


async def test_response_hop_by_hop_is_stripped_but_content_length_is_kept() -> None:
    """Re-chunking into frames does not change the byte count, and the browser
    at the far end wants it."""
    transport = _BytesTransport(
        200,
        [
            ("content-type", "text/plain"),
            ("content-length", "5"),
            ("connection", "keep-alive"),
            ("transfer-encoding", "chunked"),
        ],
        [b"hello"],
    )
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")

    head = sink.of("stream_response_head")[0]
    names = {name.lower() for name, _ in head["headers"]}
    assert "content-length" in names
    assert names & HOP_BY_HOP == set()
    assert sink.body() == b"hello"
    await mux.aclose()


@pytest.mark.parametrize("status", [201, 302, 404, 422, 500])
async def test_any_status_is_relayed_as_an_ordinary_response(status: int) -> None:
    """A 404 is a *response*, not a mux failure — ``stream_error`` is reserved
    for the tunnel being unable to produce one at all."""
    app = _CountingApp(status=status, body=b"body")
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait_types("stream_response_head")
    await sink.settle()

    assert sink.of("stream_response_head")[0]["status"] == status
    assert sink.of("stream_error") == []
    assert "stream_response_end" in sink.types()
    await mux.aclose()


async def test_redirects_are_not_followed() -> None:
    """A 302 must reach the requester; following it would make the daemon a
    proxy for whatever the Location header names."""
    transport = _BytesTransport(302, [("location", "http://evil.example/")], [b""])
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait_types("stream_response_head")
    await sink.settle()
    assert sink.of("stream_response_head")[0]["status"] == 302
    assert len(transport.requests) == 1
    await mux.aclose()


# ---------------------------------------------------------------------------
# request-body completion semantics (reference-daemon pinned)
# ---------------------------------------------------------------------------


async def test_an_inline_body_is_complete_immediately() -> None:
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)
    await mux.handle(_frame(method="POST", path="/x", body_b64=encode_body(b"inline")))
    await sink.wait_types("stream_response_head")
    await sink.settle()
    assert app.requests[0]["body"] == b"inline"
    await mux.aclose()


async def test_an_inline_empty_body_is_complete_not_pending() -> None:
    """``body_b64: ""`` and an absent ``body_b64`` are both legal and differ:
    the empty string says "this is the whole body"."""
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)
    await mux.handle(_frame(method="POST", path="/x", body_b64=""))
    await sink.wait_types("stream_response_head")
    await sink.settle()
    assert app.hits == 1
    assert app.requests[0]["body"] == b""
    await mux.aclose()


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "DELETE"])
async def test_bodiless_methods_serve_without_waiting(method: str) -> None:
    """The four exempt methods (reference parity, ``fake_daemon.py:806``)."""
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)
    await mux.handle(_frame(method=method, path="/x"))
    await sink.wait_types("stream_response_head")
    await sink.settle()
    assert app.hits == 1
    await mux.aclose()


async def test_a_chunked_body_is_assembled_across_frames() -> None:
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="POST", path="/x"))
    await asyncio.sleep(0)
    assert app.hits == 0, "serving must wait for the body to be complete"

    for chunk in (b"one-", b"two-", b"three"):
        await mux.handle(
            StreamBody(stream_id="s1", node_id=FIXTURE_NODE_ID, body_b64=encode_body(chunk))
        )
    await mux.handle(StreamEnd(stream_id="s1", node_id=FIXTURE_NODE_ID))
    await sink.wait_types("stream_response_head")
    await sink.settle()

    assert app.requests[0]["body"] == b"one-two-three"
    await mux.aclose()


async def test_an_incomplete_body_times_out_and_is_never_partially_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated POST that the handler accepts is silent data corruption."""
    monkeypatch.setattr(mux_module, "REQUEST_BODY_TIMEOUT_S", 0.02)
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="POST", path="/x"))
    await mux.handle(
        StreamBody(stream_id="s1", node_id=FIXTURE_NODE_ID, body_b64=encode_body(b"half"))
    )
    await sink.wait(lambda: bool(sink.of("stream_error")))

    assert app.hits == 0
    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "request body was never completed"
    assert "stream_response_head" not in sink.types()
    await mux.aclose()


async def test_a_request_body_over_the_advertised_limit_is_refused() -> None:
    """Defense in depth: the relay enforces §8 first, but a relay bug must not
    make the daemon buffer without bound."""
    limits = dataclasses.replace(ACK_LIMITS, max_body_bytes=16)
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app), limits=limits)
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="POST", path="/x"))
    for _ in range(4):
        await mux.handle(
            StreamBody(stream_id="s1", node_id=FIXTURE_NODE_ID, body_b64=encode_body(b"0123456789"))
        )
    await sink.wait(lambda: bool(sink.of("stream_error")))

    assert app.hits == 0
    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "the request body exceeded the relay limit"
    await mux.aclose()


async def test_an_oversized_inline_body_is_refused_too() -> None:
    limits = dataclasses.replace(ACK_LIMITS, max_body_bytes=4)
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app), limits=limits)
    mux = StreamMux(ctx)
    await mux.handle(_frame(method="POST", path="/x", body_b64=encode_body(b"far too long")))
    await sink.wait(lambda: bool(sink.of("stream_error")))
    assert app.hits == 0
    assert sink.only_error()["message"] == "the request body exceeded the relay limit"
    await mux.aclose()


# ---------------------------------------------------------------------------
# streaming — the requirement the whole design turns on
# ---------------------------------------------------------------------------


async def test_sse_chunks_reach_the_relay_before_the_source_finishes() -> None:
    """Item 2: *SSE and long-poll must STREAM, never buffer.*

    The origin is parked after its first event. If the mux accumulated — or if
    it used httpx's ``aiter_bytes(chunk_size)``, whose ``ByteChunker`` withholds
    bytes until the chunk size is reached — nothing would be on the wire yet.
    """
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="GET", path="/events/stream"))
    await sink.wait_types("stream_response_head", "stream_response_body")

    # The source has NOT finished — and the first event is already delivered.
    assert not transport.gates[0].is_set()
    assert sink.body() == b"event: a\ndata: 1\n\n"
    assert "stream_response_end" not in sink.types()
    assert mux.active_streams() == 1
    head = sink.of("stream_response_head")[0]
    assert ["content-type", "text/event-stream"] in head["headers"]

    transport.release()
    await sink.wait_types(
        "stream_response_head",
        "stream_response_body",
        "stream_response_body",
        "stream_response_end",
    )
    assert sink.body() == b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"
    assert mux.active_streams() == 0
    await mux.aclose()


async def test_each_read_becomes_its_own_frame() -> None:
    """One frame per arrival — never joined, which is what makes a feed a feed."""
    transport = _ParkedTransport(first=b"a", rest=[b"b", b"c", b"d"])
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait_types("stream_response_head", "stream_response_body")
    transport.release()
    await sink.wait(lambda: "stream_response_end" in sink.types())

    assert [base64.b64decode(frame["body_b64"]) for frame in sink.of("stream_response_body")] == [
        b"a",
        b"b",
        b"c",
        b"d",
    ]
    await mux.aclose()


async def test_a_large_read_is_sliced_to_the_frame_ceiling() -> None:
    """``RESPONSE_CHUNK_BYTES`` is a ceiling, not a fill target."""
    payload = b"x" * (RESPONSE_CHUNK_BYTES * 2 + 17)
    transport = _BytesTransport(200, [("content-type", "application/octet-stream")], [payload])
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait(lambda: "stream_response_end" in sink.types())

    sizes = [len(base64.b64decode(f["body_b64"])) for f in sink.of("stream_response_body")]
    assert sizes == [RESPONSE_CHUNK_BYTES, RESPONSE_CHUNK_BYTES, 17]
    assert sink.body() == payload
    await mux.aclose()


async def test_an_empty_response_still_terminates() -> None:
    transport = _BytesTransport(204, [], [b""])
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame())
    await sink.wait(lambda: "stream_response_end" in sink.types())
    assert sink.of("stream_response_body") == []
    assert sink.types() == ["stream_response_head", "stream_response_end"]
    await mux.aclose()


# ---------------------------------------------------------------------------
# §8 limits
# ---------------------------------------------------------------------------


async def test_a_response_over_the_limit_stops_reading_and_sends_no_end() -> None:
    """§8: the relay would kill the stream here anyway.

    No ``stream_response_end``, deliberately — an incomplete response must not
    look complete to the far end.
    """
    limits = dataclasses.replace(ACK_LIMITS, max_response_body_bytes=10)
    transport = _BytesTransport(200, [("content-type", "text/plain")], [b"0123456789ABCDEF"])
    ctx, sink = _ctx(transport, limits=limits)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: bool(sink.of("stream_error")))
    await sink.settle()

    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "response exceeded the relay limit"
    assert "stream_response_end" not in sink.types()
    # The head went out first, so the failure is reported as a truncation.
    assert sink.types()[0] == "stream_response_head"
    await mux.aclose()


async def test_the_concurrency_cap_refuses_the_extra_stream_while_others_are_live() -> None:
    """§8 ``max_concurrent_streams_per_node``, asserted with streams genuinely
    open — a cap that only holds once everything has finished is not a cap."""
    limits = dataclasses.replace(ACK_LIMITS, max_concurrent_streams_per_node=2)
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport, limits=limits)
    mux = StreamMux(ctx)

    await mux.handle(_frame(stream_id="a"))
    await mux.handle(_frame(stream_id="b"))
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 2)
    assert mux.active_streams() == 2

    await mux.handle(_frame(stream_id="c"))
    await sink.settle()

    assert transport.hits == 2, "the refused stream never reached the listener"
    error = sink.only_error()
    assert error["stream_id"] == "c"
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "stream limit exceeded"

    transport.release_all()
    await sink.wait(lambda: mux.active_streams() == 0)
    await mux.aclose()


async def test_a_duplicate_stream_id_is_dropped_not_answered() -> None:
    """The relay rejects duplicates control-side (``control.py:693-700``), so
    this is not a legal wire state — and answering it would terminate the LIVE
    stream of the same name."""
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame(stream_id="s1"))
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 1)
    await mux.handle(_frame(stream_id="s1"))
    await sink.settle()

    assert transport.hits == 1
    assert sink.of("stream_error") == []
    assert mux.active_streams() == 1

    transport.release_all()
    await sink.wait(lambda: mux.active_streams() == 0)
    await mux.aclose()


async def test_the_absolute_lifetime_watchdog_fails_a_wedged_stream() -> None:
    """§8: a wedged local handler must release its slot, not hold one of eight
    until the socket dies."""
    limits = dataclasses.replace(ACK_LIMITS, stream_absolute_timeout_s=0.05)
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport, limits=limits)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: mux.active_streams() == 0, timeout=3.0)
    await sink.settle()

    # The head already went out, and the stream is failed *anyway*: a late
    # ``stream_error`` is the terminator, not a retraction. The relay pops the
    # stream on any ``stream_response_end | stream_error``, so this frees the
    # concurrent-stream slot and fails the waiting requester at once instead of
    # leaving both ends hanging until the 60 s idle sweep. Reference-daemon
    # parity: ``dev/fake_daemon.py`` errors on any serve failure, head or no head.
    # What is NOT sent is ``stream_response_end`` — the response is incomplete
    # and must never look complete.
    assert sink.types() == ["stream_response_head", "stream_response_body", "stream_error"]
    assert sink.frames[-1]["code"] == ERROR_CODE_INTERNAL
    assert "stream_response_end" not in sink.types()
    assert transport.streams[0].closed.is_set()
    await mux.aclose()


async def test_a_watchdog_timeout_before_the_head_is_reported() -> None:
    """Pre-head, the wire still allows an error — and the requester deserves one."""

    class _HangingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10)
            raise AssertionError("unreachable")  # pragma: no cover

    limits = dataclasses.replace(ACK_LIMITS, stream_absolute_timeout_s=0.05)
    ctx, sink = _ctx(_HangingTransport(), limits=limits)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: bool(sink.of("stream_error")), timeout=3.0)

    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "the daemon could not serve the request"
    assert "stream_response_head" not in sink.types()
    await mux.aclose()


async def test_the_pacer_honours_the_advertised_bandwidth() -> None:
    """§8: the relay kills a stream that outruns its budget, so pacing is
    survival, not politeness. Driven by an injected monotonic, so the whole
    rolling window costs zero real time."""
    limits = dataclasses.replace(ACK_LIMITS, max_bandwidth_bytes_per_second=RESPONSE_CHUNK_BYTES)
    chunks = [b"x" * RESPONSE_CHUNK_BYTES for _ in range(4)]
    transport = _BytesTransport(200, [("content-type", "application/octet-stream")], chunks)
    clock = _FakeMonotonic()
    ctx, sink = _ctx(transport, limits=limits, monotonic=clock)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: "stream_response_end" in sink.types())

    assert len(sink.of("stream_response_body")) == 4
    # Four full windows' worth of bytes cannot leave inside one window.
    assert len(clock.sleeps) == 3
    assert all(0 < delay <= 1.0 for delay in clock.sleeps)
    assert clock.now == pytest.approx(3.0)
    await mux.aclose()


# ---------------------------------------------------------------------------
# cancellation + teardown
# ---------------------------------------------------------------------------


async def test_close_stream_cancels_the_task_and_closes_the_response() -> None:
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 1)
    assert mux.active_streams() == 1

    await mux.handle(CloseStream(stream_id="s1", node_id=FIXTURE_NODE_ID, reason="gone"))
    await sink.wait(lambda: transport.streams[0].closed.is_set())

    assert mux.active_streams() == 0
    # A cancelled stream produces no terminal frame: the requester asked us to
    # stop, so there is nothing to report back to it.
    assert "stream_response_end" not in sink.types()
    assert sink.of("stream_error") == []
    await mux.aclose()


@pytest.mark.parametrize(
    "frame",
    [
        StreamBody(stream_id="ghost", node_id=FIXTURE_NODE_ID, body_b64=""),
        StreamEnd(stream_id="ghost", node_id=FIXTURE_NODE_ID),
        CloseStream(stream_id="ghost", node_id=FIXTURE_NODE_ID, reason=None),
    ],
)
async def test_frames_for_an_unknown_stream_are_dropped(frame: Any) -> None:
    ctx, sink = _ctx(httpx.ASGITransport(app=_CountingApp()))
    mux = StreamMux(ctx)
    await mux.handle(frame)
    await sink.settle()
    assert sink.frames == []
    await mux.aclose()


async def test_aclose_cancels_live_streams_and_is_idempotent() -> None:
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame(stream_id="a"))
    await mux.handle(_frame(stream_id="b"))
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 2)

    await mux.aclose()
    assert mux.active_streams() == 0
    assert all(stream.closed.is_set() for stream in transport.streams)
    # Called on every close path by the manager (renewal included).
    await mux.aclose()
    await mux.aclose()


async def test_a_dead_tunnel_ends_a_stream_quietly() -> None:
    """``LinkClosedError`` from the uplink: nowhere to report, and the manager
    is already reconnecting."""
    transport = _ParkedTransport()
    sink = Collector()
    ctx, _ = _ctx(transport, collector=sink)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 1)
    sink.closed = True
    transport.release_all()
    await sink.wait(lambda: mux.active_streams() == 0)
    assert sink.of("stream_error") == []
    await mux.aclose()


async def test_a_listener_failure_is_reported_as_a_stream_error() -> None:
    transport = _BrokenTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: bool(sink.of("stream_error")))

    assert transport.hits == 1
    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "the daemon could not serve the request"
    assert mux.active_streams() == 0
    await mux.aclose()


# ---------------------------------------------------------------------------
# shutdown hygiene
# ---------------------------------------------------------------------------


async def test_aclose_reaps_every_stream_task() -> None:
    """A task still inside ``client.stream`` when the client closes raises
    noisily, so ``aclose`` cancels *and* awaits before releasing the client."""
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame(stream_id="s1"))
    await mux.handle(_frame(stream_id="s2"))
    await sink.wait(lambda: mux.active_streams() == 2)
    tasks = [stream.task for stream in mux._streams.values()]  # noqa: SLF001

    await mux.aclose()

    assert mux.active_streams() == 0
    assert all(task is not None and task.done() for task in tasks)


async def test_aclose_keeps_a_cancellation_aimed_at_itself() -> None:
    """``aclose`` must reap its children without swallowing its own cancellation.

    The old per-task ``suppress(CancelledError, Exception)`` loop could not tell
    "the child I just cancelled finished" from "somebody cancelled *me*": it
    absorbed the second case and returned normally, so a caller cancelling a
    shutdown got a coroutine that quietly claimed to have completed. ``gather``
    collects child cancellations as results while letting ours through.
    """
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)
    await mux.handle(_frame(stream_id="s1"))
    await sink.wait(lambda: mux.active_streams() == 1)

    async def stubborn() -> None:
        """A child that takes a moment to honour its cancellation."""
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            raise

    stream = next(iter(mux._streams.values()))  # noqa: SLF001
    assert stream.task is not None
    stream.task.cancel()
    await asyncio.gather(stream.task, return_exceptions=True)
    stream.task = asyncio.create_task(stubborn())

    closing = asyncio.create_task(mux.aclose())
    await asyncio.sleep(0)  # let aclose reach the gather
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    transport.release_all()


# ---------------------------------------------------------------------------
# fixture replays — the daemon half of the frozen exchanges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "exchange_status.json",
        "exchange_dashboard.json",
        "exchange_mcp.json",
        "exchange_mcp_tools_list.json",
    ],
)
async def test_a_frozen_exchange_replays_frame_for_frame(name: str) -> None:
    """Feed the fixture's ``open_stream`` in; expect its exact uplink frames out.

    Note the fixtures predate the relay's credential injection (spec §4.2), so
    the request has to be re-signed with the two injected headers before it can
    legally be served — the security gate correctly refuses the verbatim frame,
    which is itself worth knowing.
    """
    exchange = load_fixture(name)
    request = exchange["request"]
    expected = exchange["response"]

    head = next(f for f in expected if f["type"] == "stream_response_head")
    body = b"".join(
        base64.b64decode(f["body_b64"]) for f in expected if f["type"] == "stream_response_body"
    )
    transport = _BytesTransport(
        head["status"], [(name, value) for name, value in head["headers"]], [body]
    )
    # The fixture node id, or the uplink frames differ on that key alone.
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    # The verbatim fixture frame is refused — no credentials on it.
    await mux.handle(_raw_frame(dict(request)))
    await sink.settle()
    assert sink.only_error()["code"] == ERROR_CODE_AUTH

    sink.frames.clear()
    credentialed = dict(request)
    credentialed["headers"] = [
        *[list(pair) for pair in request["headers"]],
        ["authorization", f"Bearer {TOKEN}"],
        ["x-nerdit-role", "submitter"],
    ]
    await mux.handle(_raw_frame(credentialed))
    await sink.wait(lambda: "stream_response_end" in sink.types())

    assert sink.frames == expected
    assert len(transport.requests) == 1
    assert transport.requests[0].method == request["method"]
    assert transport.requests[0].url.path == request["path"]
    await mux.aclose()


async def test_the_sse_exchange_replays_as_one_frame_per_event() -> None:
    """``exchange_events.json`` pins two events as two frames — the shape that
    makes the gateway able to stream incrementally."""
    exchange = load_fixture("exchange_events.json")
    expected = exchange["response"]
    head = expected[0]
    events = [
        base64.b64decode(f["body_b64"]) for f in expected if f["type"] == "stream_response_body"
    ]
    transport = _BytesTransport(
        head["status"], [(name, value) for name, value in head["headers"]], events
    )
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    credentialed = dict(exchange["request"])
    credentialed["headers"] = [
        *[list(pair) for pair in exchange["request"]["headers"]],
        ["authorization", f"Bearer {TOKEN}"],
        ["x-nerdit-role", "submitter"],
    ]
    await mux.handle(_raw_frame(credentialed))
    await sink.wait(lambda: "stream_response_end" in sink.types())

    assert sink.frames == expected
    assert len(sink.of("stream_response_body")) == 2
    await mux.aclose()


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------


async def test_no_log_record_carries_a_token_header_body_or_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The discipline is stated in the module docstring; this is the enforcement.

    A "just log the frame" edit is exactly the kind of change that breaks it
    silently, and the frame's headers hold the injected bearer.
    """
    caplog.set_level(logging.DEBUG, logger="nerdit.link.mux")
    secret_query = "apikey=SUPERSECRETQUERY"
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    # A refused stream, a duplicate, an unknown stream and a served one.
    await mux.handle(_frame(stream_id="bad", token="forged", query=secret_query))
    await mux.handle(
        _frame(stream_id="ok", query=secret_query, body_b64=encode_body(b"SECRETBODY"))
    )
    await mux.handle(StreamEnd(stream_id="ghost", node_id=FIXTURE_NODE_ID))
    await sink.wait(lambda: "stream_response_end" in sink.types())
    await sink.settle()

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged  # the mux really did log something
    for forbidden in (TOKEN, "forged", "SUPERSECRETQUERY", "SECRETBODY", "Bearer"):
        assert forbidden not in logged, forbidden
    await mux.aclose()


async def test_no_log_record_carries_an_app_stream_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(P26 WP-H, PR review) The path is redacted on the APP path.

    A hosted app's URL path is third-party input and routinely carries
    capability material in its segments — a password-reset link, a signed
    download, a magic login. Both app log sites are exercised: the 404 refusal
    and the pre-dispatch credential refusal (an app stream reaches that one
    before the mux ever looks at the carrier).
    """
    caplog.set_level(logging.DEBUG, logger="nerdit.link.mux")
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app), resolve_app=_resolver(None))
    mux = StreamMux(ctx)

    # A refused app stream (the 404 log line) and a refused-credential app
    # stream (the pre-dispatch log line) — both carry the sensitive path.
    await mux.handle(_app_frame(path="/reset/MAGICLINKTOKEN"))
    await mux.handle(_app_frame(stream_id="bad", token="forged", path="/reset/MAGICLINKTOKEN"))
    await sink.wait(lambda: "stream_response_end" in sink.types())
    await sink.settle()

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "MAGICLINKTOKEN" not in logged
    assert "/reset/" not in logged
    assert "<app>" in logged  # the redaction marker, so the line is still readable
    await mux.aclose()


async def test_a_narrowed_bandwidth_slices_chunks_to_the_budget() -> None:
    """PR #114 review: the pacer sleeps a window out at most once per chunk,
    so a chunk larger than the negotiated per-second budget would still be
    sent whole — and the relay kills a stream that outruns its budget. Chunks
    are sliced to min(frame ceiling, budget)."""
    budget = 1024
    limits = dataclasses.replace(ACK_LIMITS, max_bandwidth_bytes_per_second=budget)
    transport = _BytesTransport(
        200, [("content-type", "application/octet-stream")], [b"x" * (budget * 4)]
    )
    clock = _FakeMonotonic()
    ctx, sink = _ctx(transport, limits=limits, monotonic=clock)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: "stream_response_end" in sink.types())

    bodies = sink.of("stream_response_body")
    assert len(bodies) == 4  # sliced to the budget, not the 64 KiB frame ceiling
    assert all(len(base64.b64decode(b["body_b64"])) <= budget for b in bodies)
    # Four full windows' worth cannot leave inside one window.
    assert len(clock.sleeps) == 3
    await mux.aclose()


async def test_a_reused_stream_id_survives_the_predecessors_teardown() -> None:
    """PR #114 review: close_stream pops-and-cancels, so the relay may legally
    reuse the id at once — and the cancelled task's ``finally`` ran AFTER the
    replacement registered, deregistering it (untracked: frames dropped,
    capacity bypassed, aclose blind). The pop is now guarded by identity."""
    transport = _ParkedTransport()
    ctx, sink = _ctx(transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame())
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 1)
    await mux.handle(CloseStream(stream_id="s1", node_id=FIXTURE_NODE_ID, reason="gone"))
    # Reuse the id IMMEDIATELY — before the cancelled task's finally has run.
    await mux.handle(_frame())
    await sink.wait(lambda: transport.streams[0].closed.is_set())
    for _ in range(5):  # let the predecessor's teardown fully unwind
        await asyncio.sleep(0)

    assert mux.active_streams() == 1  # the replacement is still tracked
    await mux.aclose()


# ---------------------------------------------------------------------------
# P26 WP-H — app-stream dispatch (D-P26-H2, plan §4 "Mux dispatch")
# ---------------------------------------------------------------------------


def _app_frame(name: str = "demo", **kw: Any) -> OpenStream:
    """An ``open_stream`` carrying the P26 carrier header.

    Built through :func:`_frame` so the relay-injected credentials are on it
    exactly as they are on every other stream — app traffic rides the SAME
    capability (D-P26-H4); the header selects the upstream, not the privilege.
    """
    headers = list(kw.pop("headers", [("accept", "*/*")]))
    headers.append((APP_HEADER, name))
    return _frame(headers=headers, **kw)


def _resolver(target: AppTarget | None) -> Callable[[str], Any]:
    """A resolver double that records every name it was asked about."""

    async def resolve(
        name: str, authority: str | None, job_id: str | None, access: str | None
    ) -> AppTarget | None:
        calls.append(name)
        return target

    calls: list[str] = []
    resolve.calls = calls  # type: ignore[attr-defined]
    return resolve


async def test_header_less_stream_is_unchanged_when_a_resolver_is_installed() -> None:
    """The non-regression that the whole phase rests on.

    An armed node must serve the daemon API exactly as an unarmed one does:
    the loopback app is hit once, the app transport never, and the frame
    sequence is the pre-P26 happy path frame for frame.
    """
    app = _CountingApp()
    app_transport = _BytesTransport(200, [("content-type", "text/plain")], [b"nope"])
    resolve = _resolver(AppTarget("demo", 4321, "demo--gpu-box.nodes.test"))
    ctx, sink = _ctx(httpx.ASGITransport(app=app), resolve_app=resolve, app_transport=app_transport)
    mux = StreamMux(ctx)

    await mux.handle(_frame(method="GET", path="/api/status", query="verbose=1"))
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
    await sink.settle()

    assert app.hits == 1
    assert app_transport.requests == []
    assert resolve.calls == []  # type: ignore[attr-defined]
    assert sink.body() == b'{"ok":true}'
    assert sink.types() == [
        "stream_response_head",
        "stream_response_body",
        "stream_response_end",
    ]
    await mux.aclose()


async def test_app_stream_dials_the_live_port_with_host_rewritten_and_path_unchanged() -> None:
    """D-P26-H2 in one assertion block.

    ``127.0.0.1:<live_port>`` is the ONLY upstream; the path stays
    origin-relative (the app is served at the root of its hosted name, which is
    the base-path class this layer exists to dissolve); ``Host`` becomes the
    hosted authority; and the relay-injected daemon credential plus both
    relay-asserted carriers (``x-nerdit-app`` and the P32
    ``x-nerdit-cloud-control``) are stripped, while the cloud's
    ``X-Forwarded-*`` survive untouched.
    """
    app = _CountingApp()
    app_transport = _BytesTransport(200, [("content-type", "text/html")], [b"<h1>hi</h1>"])
    resolve = _resolver(AppTarget("demo", 4321, "demo--gpu-box.nodes.test"))
    ctx, sink = _ctx(httpx.ASGITransport(app=app), resolve_app=resolve, app_transport=app_transport)
    mux = StreamMux(ctx)

    await mux.handle(
        _app_frame(
            method="GET",
            path="/x/y",
            query="q=1",
            headers=[
                ("accept", "*/*"),
                ("x-forwarded-proto", "https"),
                ("host", "evil.example"),
                # (P32, review round 1) Unreachable in production — the cloud
                # gateway refuses this header inbound and the entitlement pusher
                # never sets ``x-nerdit-app`` — so it is pinned here as the
                # defence-in-depth it is: relay-asserted control metadata never
                # reaches a user's container, whatever framed the stream.
                (CLOUD_CONTROL_HEADER, "entitlement"),
                ("x-nerdit-project-id", "prj_aaaaaaaaaaaaaaaa"),
            ],
        )
    )
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
    await sink.settle()

    assert app.hits == 0, "an app stream must never reach the daemon API"
    assert resolve.calls == ["demo"]  # type: ignore[attr-defined]
    request = app_transport.requests[0]
    assert str(request.url) == "http://127.0.0.1:4321/x/y?q=1"
    assert request.headers["host"] == "demo--gpu-box.nodes.test"
    assert request.headers["x-forwarded-proto"] == "https"
    for stripped in (
        "authorization",
        "x-nerdit-role",
        APP_HEADER,
        CLOUD_CONTROL_HEADER,
        "x-nerdit-project-id",
    ):
        assert stripped not in request.headers, stripped
    assert sink.body() == b"<h1>hi</h1>"
    await mux.aclose()


def test_the_cloud_control_carrier_matches_the_daemon_auth_constant() -> None:
    """``core`` may not import ``daemon``, so the header name is duplicated.

    That duplication is the whole risk: if the auth gate ever renames its
    carrier, the mux's strip set would silently stop matching and the header
    would start reaching app containers. Pinned rather than refactored, because
    the alternative — ``core.link.mux`` importing ``nerdit.daemon.auth`` —
    inverts the layering that keeps the tunnel client importable without the
    daemon package.
    """
    from nerdit.daemon.auth import CLOUD_CONTROL_HEADER as AUTH_HEADER  # noqa: PLC0415

    assert CLOUD_CONTROL_HEADER == AUTH_HEADER


@pytest.mark.parametrize(
    "name",
    ["demo", 'evil"><b'],
    ids=["ordinary", "hostile"],
)
async def test_app_stream_404_when_the_resolver_says_none(name: str) -> None:
    """A ``None`` from the resolver is the SAME answer whatever was asked for,
    and nothing upstream is contacted.

    This test's subject is the **mux** branch, not the resolver: it proves that
    a refusal costs no connection and echoes no name. *Why* a given name is
    refused — unshared, unknown, a model, a database, an endpoint-less service,
    a malformed label rejected before any DB read — is falsified in
    ``tests/test_link_app_streams.py`` against the real
    :class:`~nerdit.core.link.apps.AppStreamResolver` and a real database; the
    two files are complementary and neither subsumes the other.

    The hostile parameter is what makes the "never echoed" assertion bite: were
    the name interpolated into the body, that value would both appear in it and
    break its JSON.
    """
    app = _CountingApp()
    app_transport = _BytesTransport(200, [("content-type", "text/plain")], [b"never"])
    resolve = _resolver(None)
    ctx, sink = _ctx(httpx.ASGITransport(app=app), resolve_app=resolve, app_transport=app_transport)
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(name, method="GET", path="/"))
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
    await sink.settle()

    assert app.hits == 0
    assert app_transport.requests == [], "no upstream connection may be attempted"
    head = sink.of("stream_response_head")[0]
    assert head["status"] == 404
    assert ["content-type", "application/json"] in head["headers"]
    assert ["cache-control", "no-store"] in head["headers"]
    body = json.loads(sink.body())
    assert body["code"] == "share.not_shared"
    # The attacker-chosen name is never echoed back.
    assert name not in sink.body().decode()
    assert "stream_error" not in sink.types()
    await mux.aclose()


async def test_app_stream_404_when_there_is_no_resolver() -> None:
    """The pre-P26 / unlinked / domain-less node: ``resolve_app is None``.

    The dangerous failure mode would be treating "no resolver" as "not an app
    stream" and replaying the request on loopback with a live daemon
    credential. It is a 404 instead.
    """
    app = _CountingApp()
    ctx, sink = _ctx(httpx.ASGITransport(app=app))
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(method="POST", path="/", body_b64=encode_body(b"x")))
    await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
    await sink.settle()

    assert app.hits == 0
    assert json.loads(sink.body())["code"] == "share.not_shared"
    await mux.aclose()


async def test_app_stream_sse_streams_chunk_for_chunk() -> None:
    """The app path reuses ``_pump``, so a user's SSE feed streams too."""
    app_transport = _ParkedTransport()
    resolve = _resolver(AppTarget("demo", 4321, "demo--gpu-box.nodes.test"))
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()), resolve_app=resolve, app_transport=app_transport
    )
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(method="GET", path="/stream"))
    await sink.wait_types("stream_response_head", "stream_response_body")

    assert not app_transport.gates[0].is_set()
    assert sink.body() == b"event: a\ndata: 1\n\n"
    assert "stream_response_end" not in sink.types()

    app_transport.release()
    await sink.wait(lambda: "stream_response_end" in sink.types())
    assert sink.body() == b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"
    await mux.aclose()


async def test_app_stream_limits_apply() -> None:
    """§8's response cap is the loopback path's, unmodified — one pump."""
    limits = dataclasses.replace(ACK_LIMITS, max_response_body_bytes=10)
    app_transport = _BytesTransport(200, [("content-type", "text/plain")], [b"0123456789ABCDEF"])
    resolve = _resolver(AppTarget("demo", 4321, "demo--gpu-box.nodes.test"))
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()),
        limits=limits,
        resolve_app=resolve,
        app_transport=app_transport,
    )
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(method="GET", path="/big"))
    await sink.wait(lambda: bool(sink.of("stream_error")))
    await sink.settle()

    error = sink.only_error()
    assert error["code"] == ERROR_CODE_INTERNAL
    assert error["message"] == "response exceeded the relay limit"
    assert "stream_response_end" not in sink.types()
    await mux.aclose()


async def test_share_deleted_mid_session_closes_the_next_stream() -> None:
    """D-P26-H1: no cache, so an unshare closes the path on the NEXT stream."""
    app_transport = _BytesTransport(200, [("content-type", "text/plain")], [b"served"])
    target: AppTarget | None = AppTarget("demo", 4321, "demo--gpu-box.nodes.test")

    async def resolve(
        name: str, authority: str | None, job_id: str | None, access: str | None
    ) -> AppTarget | None:
        nonlocal target
        answer, target = target, None  # the operator unshares between streams
        return answer

    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()), resolve_app=resolve, app_transport=app_transport
    )
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(stream_id="s1", method="GET", path="/"))
    await sink.wait(lambda: "stream_response_end" in sink.types())
    assert sink.of("stream_response_head")[0]["status"] == 200

    await mux.handle(_app_frame(stream_id="s2", method="GET", path="/"))
    await sink.wait(lambda: len(sink.of("stream_response_head")) == 2)
    await sink.settle()

    assert sink.of("stream_response_head")[1]["status"] == 404
    assert len(app_transport.requests) == 1
    await mux.aclose()


async def test_the_credential_gate_precedes_app_dispatch() -> None:
    """Order is unchanged: duplicate → capacity → credential → dispatch.

    A forged bearer with a carrier header must die at the gate, before the
    resolver is asked anything at all — otherwise the header would be a free
    read oracle on the share table for anyone who can reach the relay.
    """
    app_transport = _BytesTransport(200, [("content-type", "text/plain")], [b"never"])
    resolve = _resolver(AppTarget("demo", 4321, "demo--gpu-box.nodes.test"))
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()), resolve_app=resolve, app_transport=app_transport
    )
    mux = StreamMux(ctx)

    await mux.handle(_app_frame(token="forged"))
    await sink.settle()

    assert sink.only_error()["code"] == ERROR_CODE_AUTH
    assert resolve.calls == []  # type: ignore[attr-defined]
    assert app_transport.requests == []
    await mux.aclose()


async def test_aclose_closes_both_clients() -> None:
    """Two clients, two pools: leaking the app one would leak a connection
    pool per capability lifetime (~13 min), for the life of the process."""
    loopback = httpx.AsyncClient(transport=httpx.ASGITransport(app=_CountingApp()))
    app_client = httpx.AsyncClient(transport=_BytesTransport(200, [], [b""]))
    sink = Collector()
    clock = _FakeMonotonic()
    ctx = MuxContext(
        node_id=FIXTURE_NODE_ID,
        limits=ACK_LIMITS,
        send=sink.send,
        validate_capability=lambda token, role: True,
        loopback_base_url="http://127.0.0.1:9321",
        http_client_factory=lambda: loopback,
        app_client_factory=lambda: app_client,
        monotonic=clock,
        sleep=clock.sleep,
    )
    mux = StreamMux(ctx)

    await mux.aclose()

    assert loopback.is_closed
    assert app_client.is_closed


async def test_neither_default_client_honours_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loopback dials must never be routed through ``HTTP_PROXY`` (PR #128 review).

    ``httpx`` has no implicit ``127.0.0.1`` bypass: with a proxy in the
    environment and no matching ``NO_PROXY`` entry, a default client mounts the
    loopback URL on an ``AsyncHTTPProxy`` pool. ``nerditd`` runs with the
    operator's environment, so a corporate proxy would silently carry the
    tunnelled request — a third party's cookies, headers and body on the app
    path — off the machine, and the share would not reach the local service
    either. This is the only test that builds the mux through the PRODUCTION
    constructors: both factories are ``None``, which is exactly why the seams
    hid the defect.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    sink = Collector()
    clock = _FakeMonotonic()
    ctx = MuxContext(
        node_id=FIXTURE_NODE_ID,
        limits=ACK_LIMITS,
        send=sink.send,
        validate_capability=lambda token, role: True,
        loopback_base_url="http://127.0.0.1:9321",
        monotonic=clock,
        sleep=clock.sleep,
    )
    mux = StreamMux(ctx)

    try:
        # No proxy MOUNTS at all is the strong form of the claim: not "the
        # bypass list happened to cover this URL", but "env discovery is off".
        assert mux._client._mounts == {}
        assert mux._app_client._mounts == {}
        for client, url in (
            (mux._client, "http://127.0.0.1:9321/api/services"),
            (mux._app_client, "http://127.0.0.1:4321/"),
        ):
            # …and the URL actually resolves to the client's own direct
            # transport rather than a proxy-backed mount.
            assert client._transport_for_url(httpx.URL(url)) is client._transport
    finally:
        await mux.aclose()
