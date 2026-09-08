"""Stream validated node-link requests to daemon or shared-app loopback targets.

Validate injected authorization and role before creating an HTTP task. Rejected
credentials produce `node_authentication_failed` without touching a listener.
Daemon API requests retain those headers so middleware records the synthetic
`link:<node_id>` principal and tunnel audit provenance.

App streams use a separate client with no daemon base URL. Resolve only shared
app ports; all misses return the same 404 without connecting. Strip capability,
role, app-carrier, and cloud-control headers before forwarding to the app, and
rewrite Host to its hosted name. Hop-by-hop headers are dropped; this protocol
does not support WebSockets.

Honor advertised limits and stream unbuffered `aiter_raw` chunks immediately,
splitting only at `RESPONSE_CHUNK_BYTES`. No read timeout may truncate live SSE
or long-poll responses. The frozen node-link spec defines body completion and
stream errors.

Logs may contain only stream ID, method, safe path, and status: never headers,
bodies, queries, or tokens. App URL paths may contain credentials and must use
`_log_path` redaction; daemon API paths may be logged.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field

import httpx

from nerdit.core.link.client import LinkClosedError, MuxContext
from nerdit.core.link.frames import (
    ERROR_CODE_AUTH,
    ERROR_CODE_INTERNAL,
    CloseStream,
    Header,
    OpenStream,
    StreamBody,
    StreamDownlink,
    StreamEnd,
    response_body_frame,
    response_end_frame,
    response_head_frame,
    stream_error_frame,
)

logger = logging.getLogger("nerdit.link.mux")

#: How long a chunked request body may stay incomplete before the stream is
#: failed. Matches the reference daemon's `REQUEST_BODY_TIMEOUT_SECONDS`
#: (`dev/fake_daemon.py`). It is a module global rather than a constructor
#: argument on purpose: it is not a relay-advertised limit (it is not in
#: `hello_ack.limits`), so it does not belong on `MuxContext`, and a
#: test that needs it short monkeypatches this name — it is read at call time.
REQUEST_BODY_TIMEOUT_S = 30.0

#: Maximum decoded bytes per `stream_response_body` frame — a **ceiling**,
#: not a target. Base64 inflates by 4/3, so 64 KiB decoded is ~87 KiB on the
#: wire: comfortably inside the frame budget while keeping the frame count for
#: a large download sane. A read that yields less is sent immediately (see
#: `StreamMux._pump`); waiting to fill the ceiling is what would break
#: SSE.
RESPONSE_CHUNK_BYTES = 64 * 1024

#: RFC 7230 §6.1 connection-scoped headers. They describe *this* hop and are
#: meaningless — or actively wrong — on a tunneled one. `transfer-encoding`
#: is the sharp one: re-chunking into `stream_response_body` frames is our
#: framing, not the origin's.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

#: Additionally dropped from the *request*: `host` is this hop's authority
#: (httpx derives it from the loopback base URL) and `content-length` is
#: recomputed from the assembled body.
_REQUEST_DROPPED = frozenset({"host", "content-length"})

#: The carrier header (P26 D-P26-H2). The cloud gateway **strips any inbound
#: `x-nerdit-app`** from the browser and sets its own from the hosted name it
#: resolved, so its presence here is a routing instruction from the relay, never
#: user input. The daemon still treats the *value* as hostile: it is validated
#: as a service-name label and looked up against the share table before
#: anything is dialled, and any value that does not name a shared service is a
#: 404 (`_NOT_SHARED_BODY`).
APP_HEADER = "x-nerdit-app"

#: The cloud's control-plane carrier (P32 D-P32-2) — the header that marks a
#: stream as framed by the cloud's own server-side code rather than by a user,
#: and the second conjunct of `daemon/auth.py::require_cloud_principal`.
#: Mirrored here as a literal for the same reason `x-nerdit-role` is: `core`
#: must not import `daemon`. `tests/test_link_mux.py` pins it equal to
#: `nerdit.daemon.auth.CLOUD_CONTROL_HEADER` so the two cannot drift.
CLOUD_CONTROL_HEADER = "x-nerdit-cloud-control"

#: Dropped from an APP-bound request on top of `HOP_BY_HOP` and
#: `_REQUEST_DROPPED`. `authorization`/`x-nerdit-role` are the
#: relay-injected capability: on the loopback path they are the whole point
#: (they carry audit provenance), on the app path they would hand a user's
#: container a live daemon-API credential. The two carriers are consumed here
#: and never forwarded either — an app has no business reading its own routing
#: instruction, and echoing it invites a confused-deputy loop.
#:
#: `CLOUD_CONTROL_HEADER` joins them for the same reason and not because
#: it leaks anything: it is a constant, the cloud gateway already refuses it
#: inbound, and the entitlement pusher never sets `x-nerdit-app`, so no app
#: stream can carry it today. It is dropped here because it is *defined* as
#: relay-asserted control metadata, and a header whose whole meaning is "the
#: cloud said so" must not reach a third party's container just because two
#: unrelated invariants currently keep it out. Kept in lockstep with the
#: reference daemon (`nerdit-cloud` `dev/fake_daemon.py`).
_APP_REQUEST_DROPPED = frozenset(
    {
        "host",
        "content-length",
        "authorization",
        "x-nerdit-role",
        APP_HEADER,
        CLOUD_CONTROL_HEADER,
    }
)

#: The one answer an unroutable app stream ever gets: the daemon's ordinary
#: error envelope (`{code, message, detail}`), pre-encoded because it is a
#: constant. Identical for an unknown name, an unshared service, a model, a
#: database, an endpoint-less service and a node with no resolver at all — the
#: hosted edge must not be an enumeration oracle. A 404 *response* rather than a
#: `stream_error` because a browser sits at the far end: a structured page
#: beats a relay-level abort.
_NOT_SHARED_BODY = (
    b'{"code":"share.not_shared",'
    b'"message":"No shared service answers at this address.",'
    b'"detail":"No shared service answers at this address."}'
)

#: What a log line is allowed to say about an APP stream's path (P26 WP-H, PR
#: review). Every warning in this module names `method` and `path` and
#: deliberately omits the query string. On the loopback path the path is a
#: daemon-API path with a known grammar, so printing it is free; on the app path
#: it is a third-party application's URL path, which routinely carries
#: capability material *in its segments* — password-reset links, signed
#: download URLs, magic-login tokens. Redacted whole rather than truncated to
#: the first segment, because the first segment is exactly where a bare
#: magic-link token sits. The stream id remains the correlation handle, and the
#: cloud edge has the request line if an operator needs it.
_APP_PATH_REDACTED = "<app>"


def _log_path(frame: OpenStream) -> str:
    """The path this frame may contribute to a log record (`_APP_PATH_REDACTED`)."""
    return _APP_PATH_REDACTED if frame.header(APP_HEADER) is not None else frame.path


#: Methods whose absent `body_b64` means "there is no body", rather than
#: "the body arrives in `stream_body` frames" (reference-daemon parity,
#: `dev/fake_daemon.py:806`). The distinction cannot be inferred from the
#: frame alone, so it is pinned to the same method set on both sides.
_BODILESS_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})

#: The only capability role the wire admits (spec §5 step 5; the relay refuses
#: any other at hello and refuses a role mismatch at `open_stream`,
#: `relay/control.py:319` and `:684`). Remote admin is permanently out of
#: the protocol (D-R2), so this is a constant, not a policy knob.
_SUBMITTER = "submitter"

_BEARER_PREFIX = "Bearer "


@dataclass(slots=True)
class _Stream:
    """Per-stream state: the request being assembled and the task serving it."""

    frame: OpenStream
    complete: asyncio.Event
    body_parts: list[bytes] = field(default_factory=list)
    body_bytes: int = 0
    #: Set when the accumulated request body passed `max_body_bytes`. The
    #: stream is not failed inline — it is marked and woken, so exactly one
    #: place (the serving task) owns answering.
    body_overflow: bool = False
    task: asyncio.Task[None] | None = None
    response: httpx.Response | None = None

    def body(self) -> bytes:
        return b"".join(self.body_parts)


class _BandwidthWindow:
    """Rolling one-second uplink budget (spec §8, "Per-stream bandwidth").

    The relay enforces 10 MiB/s per stream in *both* directions and answers an
    overrun by killing the stream. Pacing ourselves is therefore not politeness
    — an unpaced large download would be terminated mid-flight.

    Time comes from `MuxContext` (`monotonic`/`sleep`) rather than
    `asyncio` so the whole limits matrix is testable without real clocks.
    """

    __slots__ = ("_limit", "_monotonic", "_sent", "_sleep", "_window_start")

    def __init__(
        self,
        limit: int,
        monotonic: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._limit = limit
        self._monotonic = monotonic
        self._sleep = sleep
        self._window_start = monotonic()
        self._sent = 0

    async def consume(self, count: int) -> None:
        """Account `count` bytes, sleeping out the window if it is full."""
        if self._limit <= 0:  # pragma: no cover - a relay never advertises 0
            return
        now = self._monotonic()
        if now - self._window_start >= 1.0:
            self._window_start = now
            self._sent = 0
        if self._sent + count > self._limit:
            remaining = 1.0 - (now - self._window_start)
            if remaining > 0:
                await self._sleep(remaining)
            self._window_start = self._monotonic()
            self._sent = 0
        self._sent += count


class StreamMux:
    """Serve relay-forwarded requests against the daemon's own listener.

    Implements `nerdit.core.link.client.InboundStreamHandler`:
    `handle` is called from the connection's receive loop and must return
    promptly, so everything that can block lives in a per-stream task.
    """

    def __init__(self, ctx: MuxContext) -> None:
        self._ctx = ctx
        self._streams: dict[str, _Stream] = {}
        self._closed = False
        if ctx.http_client_factory is not None:
            self._client = ctx.http_client_factory()
        else:
            self._client = httpx.AsyncClient(
                base_url=ctx.loopback_base_url,
                follow_redirects=False,
                # Loopback targets must NEVER be routed through an environment
                # proxy: httpx has no built-in `127.0.0.1` bypass, so on a
                # node with `HTTP_PROXY`/`ALL_PROXY` set and no matching
                # `NO_PROXY` entry the dial would be mounted on an
                # `AsyncHTTPProxy` and the tunnelled request would leave the
                # machine. Env proxy discovery is disabled explicitly rather
                # than by hoping the operator wrote a bypass (PR #128 review).
                trust_env=False,
                # `read=None` is the whole point: SSE and long-poll responses
                # (`/events/stream`, `/services/{id}/wait`, `/api/mcp`)
                # are *supposed* to stay quiet for minutes. A read timeout here
                # would truncate them into corrupt tunnels. The stream's own
                # absolute lifetime (spec §8) is the bound instead.
                timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            )
        # The SECOND client — app streams only. Separate from
        # the loopback one so the two paths cannot be confused by construction:
        # this one has **no** `base_url`, so a relative target is not even
        # expressible on it, and a bug that reached it with a daemon path would
        # fail to build a URL rather than quietly hit the control plane. Same
        # `read=None` reasoning as above — a user's app streams SSE too.
        if ctx.app_client_factory is not None:
            self._app_client = ctx.app_client_factory()
        else:
            self._app_client = httpx.AsyncClient(
                follow_redirects=False,
                # Same reason as the loopback client above, and it matters more
                # here: this is the path carrying a third party's cookies,
                # headers and body to a local app port.
                trust_env=False,
                timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
            )

    # -- InboundStreamHandler ------------------------------------------------

    async def handle(self, frame: StreamDownlink) -> None:
        """Apply one inbound stream frame. Never blocks on serving."""
        if isinstance(frame, OpenStream):
            await self._open_stream(frame)
        elif isinstance(frame, StreamBody):
            self._append_body(frame)
        elif isinstance(frame, StreamEnd):
            self._end_body(frame)
        elif isinstance(frame, CloseStream):
            self._close_stream(frame)

    def active_streams(self) -> int:
        """How many streams are being served right now."""
        return len(self._streams)

    async def aclose(self) -> None:
        """Cancel every stream and release the loopback client.

        Called on every close path by the manager (renewal included), so it has
        to be idempotent and it has to *await* the cancellations: a task still
        inside `client.stream` when the client closes would raise noisily.
        """
        self._closed = True
        streams, self._streams = self._streams, {}
        tasks = [stream.task for stream in streams.values() if stream.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            # `gather` rather than a per-task `suppress(CancelledError)`
            # loop: it collects each *child's* cancellation as a result, while
            # a cancellation delivered to `aclose` itself still propagates.
            # The suppressing loop swallowed our own cancellation and then kept
            # awaiting the remaining tasks — shutdown-hygiene sand in the gears.
            await asyncio.gather(*tasks, return_exceptions=True)
        with suppress(Exception):
            await self._client.aclose()
        # Suppressed independently: a failure closing one client must not leave
        # the other's connection pool open for the life of the process.
        with suppress(Exception):
            await self._app_client.aclose()

    # -- inbound frames ------------------------------------------------------

    async def _open_stream(self, frame: OpenStream) -> None:
        """Admit (or refuse) one forwarded request.

        Order is deliberate and matches the relay's own control-side ordering:
        duplicate → capacity → credential. The first two are answered from
        state we already hold; only after them does anything look at a header.
        """
        if self._closed:  # pragma: no cover - the manager stops feeding first
            return
        if frame.stream_id in self._streams:
            # The relay rejects a duplicate `stream_id` control-side
            # (`relay/control.py:693-700`) and never forwards it, so this is
            # not a legal wire state — log it, do not answer it (answering
            # would risk terminating the *live* stream of the same name).
            logger.warning("Ignoring a duplicate open_stream for %s", frame.stream_id)
            return

        limits = self._ctx.limits
        if len(self._streams) >= limits.max_concurrent_streams_per_node:
            # Defense in depth: the relay caps concurrency first (spec §8), so
            # reaching this means the relay and we disagree about what is open.
            # Refusing is still cheaper than serving an unbounded fan-out.
            logger.warning(
                "Refusing stream %s: %d concurrent streams already open",
                frame.stream_id,
                len(self._streams),
            )
            await self._send_error(frame, ERROR_CODE_INTERNAL, "stream limit exceeded")
            return

        if not self._credential_is_valid(frame):
            # Checklist item 3: this returns *before* any task exists, so the
            # daemon's own listener is never contacted for an unvalidated
            # stream. The message is the reference daemon's, verbatim
            # (`dev/fake_daemon.py:797`) — and deliberately says nothing
            # about which of the four checks failed.
            logger.warning(
                "Refusing stream %s %s %s: the injected capability did not validate",
                frame.stream_id,
                frame.method,
                _log_path(frame),
            )
            await self._send_error(frame, ERROR_CODE_AUTH, "the tunnel capability is invalid")
            return

        stream = _Stream(frame=frame, complete=asyncio.Event())
        if frame.body_b64 is not None:
            # An inline body is the *complete* body, even when empty: a sender
            # that chunks leaves `body_b64` absent and finishes with
            # `stream_end` (reference parity, `dev/fake_daemon.py:802-805`).
            chunk = frame.body()
            stream.body_parts.append(chunk)
            stream.body_bytes = len(chunk)
            stream.complete.set()
        elif frame.method.upper() in _BODILESS_METHODS:
            stream.complete.set()
        if stream.body_bytes > limits.max_body_bytes:
            stream.body_overflow = True

        self._streams[frame.stream_id] = stream
        stream.task = asyncio.create_task(
            self._serve(stream), name=f"nerdit-link-stream-{frame.stream_id}"
        )

    def _append_body(self, frame: StreamBody) -> None:
        """Accumulate one request-body continuation chunk."""
        stream = self._streams.get(frame.stream_id)
        if stream is None:
            logger.debug("Dropping stream_body for unknown stream %s", frame.stream_id)
            return
        if stream.body_overflow:
            return
        chunk = frame.body()
        stream.body_parts.append(chunk)
        stream.body_bytes += len(chunk)
        if stream.body_bytes > self._ctx.limits.max_body_bytes:
            # The relay enforces this first (spec §8, `control.py:818-834`);
            # we enforce it too so a relay bug cannot make the daemon buffer
            # without bound. Mark and wake — the serving task answers, so
            # there is exactly one place that can send a terminal frame.
            stream.body_overflow = True
            stream.body_parts.clear()
            stream.complete.set()

    def _end_body(self, frame: StreamEnd) -> None:
        """Mark a chunked request body complete."""
        stream = self._streams.get(frame.stream_id)
        if stream is None:
            logger.debug("Dropping stream_end for unknown stream %s", frame.stream_id)
            return
        stream.complete.set()

    def _close_stream(self, frame: CloseStream) -> None:
        """Abandon one stream: the requester is gone, or a relay limit tripped."""
        stream = self._streams.pop(frame.stream_id, None)
        if stream is None:
            logger.debug("Dropping close_stream for unknown stream %s", frame.stream_id)
            return
        logger.debug("Closing stream %s on relay request", frame.stream_id)
        if stream.task is not None:
            # Cancellation unwinds the `async with client.stream(...)` block,
            # which is what actually closes the httpx response and releases the
            # connection. Awaiting the task here would block the receive loop,
            # so the task's own `finally` owns the teardown.
            stream.task.cancel()

    # -- credential gate -----------------------------------------------------

    def _credential_is_valid(self, frame: OpenStream) -> bool:
        """Validate the relay-injected capability (spec §4.2, checklist item 3).

        Four conditions, all required, mirroring the reference daemon's single
        composite check (`dev/fake_daemon.py:783-793`):

        1. the frame's own `role` is `submitter` — the relay refuses a role
           mismatch in either direction (`relay/control.py:684-692`), so any
           other value means the frame did not come from a conforming relay;
        2. the injected `X-Nerdit-Role` header agrees;
        3. an `Authorization` header is present with the exact `Bearer`
           prefix;
        4. the token *is* the capability currently proven on this connection
           — `LinkManager.validate_capability` compares it in constant
           time and re-checks expiry (spec §9: expiry is reactive, so a
           capability can lapse under a live socket).
        """
        if frame.role != _SUBMITTER:
            return False
        if frame.header("x-nerdit-role") != _SUBMITTER:
            return False
        authorization = frame.header("authorization")
        if authorization is None or not authorization.startswith(_BEARER_PREFIX):
            return False
        token = authorization[len(_BEARER_PREFIX) :]
        if not token:
            return False
        return bool(self._ctx.validate_capability(token, _SUBMITTER))

    # -- serving -------------------------------------------------------------

    async def _serve(self, stream: _Stream) -> None:
        """Own one stream end to end, under its absolute lifetime bound."""
        frame = stream.frame
        try:
            # Spec §8: the relay kills a stream that outlives
            # `stream_absolute_timeout_s` (3600 s by default). Bounding
            # ourselves means a wedged local handler releases its slot instead
            # of holding one of the eight until the socket dies.
            await asyncio.wait_for(
                self._serve_request(stream), self._ctx.limits.stream_absolute_timeout_s
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning(
                "Stream %s (%s %s) exceeded its absolute lifetime",
                frame.stream_id,
                frame.method,
                _log_path(frame),
            )
            await self._fail_late(stream)
        except LinkClosedError:
            # The tunnel went away under an uplink send. There is nowhere to
            # report to and the manager is already reconnecting; end quietly.
            logger.debug("Stream %s ended with the tunnel", frame.stream_id)
        except httpx.HTTPError as exc:
            # Never log `exc` verbatim beyond its type: an httpx error can
            # carry the request URL, and the URL carries the query string.
            logger.warning(
                "Stream %s (%s %s) failed: %s",
                frame.stream_id,
                frame.method,
                _log_path(frame),
                type(exc).__name__,
            )
            await self._fail_late(stream)
        except Exception:
            logger.exception("Stream %s (%s) failed unexpectedly", frame.stream_id, frame.method)
            await self._fail_late(stream)
        finally:
            # Remove ONLY this task's stream: after `close_stream` cancels
            # us, the relay may legally reuse the id, and the replacement can
            # be registered before this `finally` runs — an unconditional
            # pop would deregister the replacement (untracked: capacity
            # accounting bypassed, its frames dropped, `aclose` blind to it).
            if self._streams.get(frame.stream_id) is stream:
                del self._streams[frame.stream_id]
            response = stream.response
            if response is not None:
                with suppress(Exception):
                    await response.aclose()

    async def _serve_request(self, stream: _Stream) -> None:
        """Assemble the request, replay it on loopback, stream the answer back."""
        frame = stream.frame

        if not stream.complete.is_set():
            try:
                # Read the timeout from the module at call time so a test can
                # monkeypatch it; see the constant's own note.
                await asyncio.wait_for(stream.complete.wait(), REQUEST_BODY_TIMEOUT_S)
            except TimeoutError:
                # Never serve a partial body: a truncated POST that the handler
                # accepts is a silent data-corruption bug, and the requester
                # cannot tell it apart from success.
                logger.warning(
                    "Stream %s (%s %s): request body was never completed",
                    frame.stream_id,
                    frame.method,
                    _log_path(frame),
                )
                await self._send_error(
                    frame, ERROR_CODE_INTERNAL, "request body was never completed"
                )
                return

        if stream.body_overflow:
            logger.warning(
                "Stream %s (%s %s): request body exceeded the advertised limit",
                frame.stream_id,
                frame.method,
                _log_path(frame),
            )
            await self._send_error(
                frame, ERROR_CODE_INTERNAL, "the request body exceeded the relay limit"
            )
            return

        app_name = frame.header(APP_HEADER)
        if app_name is not None:
            # Everything below this branch is the daemon-API
            # path and must stay byte-for-byte what the conformance fixtures
            # pin; app traffic never falls through into it.
            await self._serve_app(stream, app_name)
            return

        target = f"{frame.path}?{frame.query}" if frame.query else frame.path
        headers = [
            (name, value)
            for name, value in frame.headers
            if name.lower() not in HOP_BY_HOP and name.lower() not in _REQUEST_DROPPED
        ]
        async with self._client.stream(
            frame.method, target, headers=headers, content=stream.body()
        ) as response:
            stream.response = response
            await self._pump(stream, response)

    async def _serve_app(self, stream: _Stream, app_name: str) -> None:
        """Serve one hosted app stream (P26 D-P26-H2), or answer 404.

        The resolver is consulted on **every** stream, so an unshare closes the
        path on the next request. A `None` answer — including "there is no
        resolver", which is what an unlinked or domain-less node has — returns
        before any connection is attempted: the daemon must never dial on
        behalf of a name it has not authorised.
        """
        frame = stream.frame
        resolver = self._ctx.resolve_app
        target = await resolver(app_name) if resolver is not None else None
        if target is None:
            await self._send_not_shared(frame)
            return

        headers = [
            (name, value)
            for name, value in frame.headers
            if name.lower() not in HOP_BY_HOP and name.lower() not in _APP_REQUEST_DROPPED
        ]
        # httpx honours an explicit `Host` over the URL authority
        # (`Request._prepare` uses `setdefault`), so the app sees the name
        # the browser typed while the socket still goes to loopback.
        headers.append(("host", target.host))

        # `frame.path` is origin-relative by grammar (`frames.py`), so the
        # app is served at the ROOT of its hosted name — no base path, which is
        # the whole point of the hosted layer for path-mode nodes.
        url = f"http://127.0.0.1:{target.port}{frame.path}"
        if frame.query:
            url = f"{url}?{frame.query}"

        async with self._app_client.stream(
            frame.method, url, headers=headers, content=stream.body()
        ) as response:
            stream.response = response
            # The SAME pump: limits table, pacing, chunk-for-chunk SSE and the
            # overflow rule are the loopback path's, unmodified. There is no
            # second response policy to keep in sync.
            await self._pump(stream, response)

    async def _send_not_shared(self, frame: OpenStream) -> None:
        """Answer an unroutable app stream with the one structured 404.

        The requested name is deliberately absent from both the body and the
        log line: it is attacker-chosen input, and repeating it would turn the
        refusal into an echo surface. The *path* is redacted for the adjacent
        reason (`_log_path`) — a refused app request is exactly the one
        whose URL is most likely to be a bare magic link somebody mistyped.
        """
        logger.warning(
            "Stream %s (%s %s): no shared service answers this app stream",
            frame.stream_id,
            frame.method,
            _log_path(frame),
        )
        ctx = self._ctx
        await ctx.send(
            response_head_frame(
                stream_id=frame.stream_id,
                node_id=ctx.node_id,
                status=404,
                headers=[
                    ("content-type", "application/json"),
                    ("cache-control", "no-store"),
                    ("content-length", str(len(_NOT_SHARED_BODY))),
                ],
            )
        )
        await ctx.send(
            response_body_frame(
                stream_id=frame.stream_id, node_id=ctx.node_id, body=_NOT_SHARED_BODY
            )
        )
        await ctx.send(response_end_frame(stream_id=frame.stream_id, node_id=ctx.node_id))

    async def _pump(self, stream: _Stream, response: httpx.Response) -> None:
        """Send the head, then one frame per chunk **as it arrives**."""
        frame = stream.frame
        ctx = self._ctx
        limits = ctx.limits

        head: list[Header] = [
            (name, value)
            for name, value in response.headers.multi_items()
            # `content-length` is kept: re-chunking into
            # `stream_response_body` frames does not change the byte count,
            # and the browser at the far end wants it.
            if name.lower() not in HOP_BY_HOP
        ]
        await ctx.send(
            response_head_frame(
                stream_id=frame.stream_id,
                node_id=ctx.node_id,
                status=response.status_code,
                headers=head,
            )
        )

        pacer = _BandwidthWindow(limits.max_bandwidth_bytes_per_second, ctx.monotonic, ctx.sleep)
        total = 0
        # `aiter_raw()` with **no** chunk size, deliberately. httpx's
        # `aiter_bytes(n)`/`aiter_raw(n)` run the body through a
        # `ByteChunker` that *withholds* bytes until `n` have accumulated
        # (`httpx/_models.py`, `ByteChunker.decode`) — an SSE feed emitting
        # 40-byte events would sit unsent until 64 KiB of them piled up, which
        # is exactly the buffering this mux exists to avoid. So we take each
        # read as it arrives and do the *capping* ourselves below: emit-as-it-
        # arrives, at most `RESPONSE_CHUNK_BYTES` per frame.
        #
        # `aiter_raw` rather than `aiter_bytes` because this is a proxy, not
        # a client: the origin's bytes are forwarded untranscoded, so a
        # `content-encoding`/`content-length` kept in the head above stays
        # true of what the far end actually receives.
        # A chunk must also fit ONE bandwidth window: `_BandwidthWindow`
        # sleeps a window out at most once per `consume`, so a chunk larger
        # than the negotiated per-second budget would still be sent whole and
        # the relay would kill the stream for the overrun. Slicing to the
        # smaller of the frame ceiling and the budget keeps every frame
        # payable within a single window.
        chunk_cap = min(RESPONSE_CHUNK_BYTES, limits.max_bandwidth_bytes_per_second)
        async for raw in response.aiter_raw():
            if not raw:
                continue
            for offset in range(0, len(raw), chunk_cap):
                chunk = raw[offset : offset + chunk_cap]
                total += len(chunk)
                if total > limits.max_response_body_bytes:
                    # Spec §8: the relay would kill the stream at this point
                    # anyway. Stop *reading* (leaving the local handler to be
                    # cancelled by the `async with` unwind) and report — with
                    # no `stream_response_end`, because the response is not
                    # complete and must not look like it is.
                    logger.warning(
                        "Stream %s (%s %s): response exceeded the advertised limit",
                        frame.stream_id,
                        frame.method,
                        _log_path(frame),
                    )
                    await self._send_error(
                        frame, ERROR_CODE_INTERNAL, "response exceeded the relay limit"
                    )
                    return
                await pacer.consume(len(chunk))
                # One frame per chunk, sent here rather than collected: this
                # line is what makes `/events/stream` an event *stream* at
                # the far end instead of a long pause and a wall of text.
                await ctx.send(
                    response_body_frame(stream_id=frame.stream_id, node_id=ctx.node_id, body=chunk)
                )
        await ctx.send(response_end_frame(stream_id=frame.stream_id, node_id=ctx.node_id))

    # -- uplink helpers ------------------------------------------------------

    async def _fail_late(self, stream: _Stream) -> None:
        """Report a serving failure — **always**, head sent or not.

        A `stream_error` after `stream_response_head` is not a protocol lie,
        it is the terminator: the relay pops the stream on *any*
        `stream_response_end | stream_error` (`relay/control.py`), so a late
        error frame promptly frees the concurrent-stream slot and fails the
        waiting requester. Staying silent instead leaves both ends hanging until
        the 60 s idle sweep. This matches the reference daemon
        (`dev/fake_daemon.py` sends `stream_error internal_error` on any
        serve failure, including after response frames went out) and this mux's
        own response-overflow path, which already errors after the head.
        """
        await self._send_error(
            stream.frame, ERROR_CODE_INTERNAL, "the daemon could not serve the request"
        )

    async def _send_error(self, frame: OpenStream, code: str, message: str) -> None:
        """Send one `stream_error`; a dead tunnel is not an error to report."""
        try:
            await self._ctx.send(
                stream_error_frame(
                    stream_id=frame.stream_id,
                    node_id=self._ctx.node_id,
                    code=code,
                    message=message,
                )
            )
        except LinkClosedError:
            logger.debug("Could not report stream %s: the tunnel is gone", frame.stream_id)


__all__ = [
    "APP_HEADER",
    "HOP_BY_HOP",
    "REQUEST_BODY_TIMEOUT_S",
    "RESPONSE_CHUNK_BYTES",
    "StreamMux",
]
