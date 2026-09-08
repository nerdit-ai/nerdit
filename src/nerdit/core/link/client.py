"""Dial, authenticate, heartbeat, and classify closure for one node-link connection.

WSS uses `node-link/v1`: receive `node_challenge`, send signed `hello`, then
receive `hello_ack` and use its heartbeat cadence. Classify pre-upgrade HTTP
403 refusals as well as socket closes, including ambiguous 4401 expiry/drain.
The frozen `nerdit-cloud/docs/node-link-v1.md` contract defines these semantics.

The manager owns renewal/reconnect policy and events; the mux owns inbound
request handling. Shared mux interfaces live here to keep imports acyclic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from nerdit.core.link.frames import (
    PROTOCOL_VERSION,
    Downlink,
    ErrorFrame,
    FrameError,
    HelloAck,
    NodeChallenge,
    StreamDownlink,
    StreamLimits,
    dump_frame,
    heartbeat_frame,
    hello_frame,
    parse_downlink,
)
from nerdit.core.link.identity import NodeIdentity, proof_message

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

    from nerdit.core.link.capability import MintedCapability

logger = logging.getLogger("nerdit.link.client")

#: Bound for `LinkConnection._expect`. Written as a `TypeVar` rather
#: than PEP 695 syntax because the package floor is Python 3.11.
_FrameT = TypeVar("_FrameT", bound="Downlink")

#: The relay's own handshake deadline is 10 s (spec §5 step 3,
#: `node_handshake_timeout_seconds`). Ours mirrors it **per phase** rather
#: than over the whole handshake: a relay that answers the challenge and then
#: stalls before `hello_ack` must not be able to hold the dial open for
#: twice the budget it granted us.
HANDSHAKE_TIMEOUT_S = 10.0

#: The relay forwards request bodies up to 10 MiB (spec §8) and base64 inflates
#: them by a third, so the `websockets` default 1 MiB frame cap would reject
#: legal traffic. Sized with headroom for the JSON envelope; a frame larger than
#: this cannot be a legal `node-link/v1` message.
MAX_FRAME_BYTES = 16 * 1024 * 1024

#: The path the relay mounts the node link on (spec §1, `control.py:1015`).
CONNECT_PATH = "/v1/connect"

#: Floor on the `hello_ack`-advertised heartbeat cadence. The relay's own
#: default is 5 s (spec §8); anything below a second is not a cadence, it is an
#: uplink flood the relay would be asking us to inflict on ourselves.
MIN_HEARTBEAT_INTERVAL_S = 1.0

#: Connection-scoped `error` frames the relay emits *in answer to one uplink
#: frame* while the connection stays perfectly healthy (`relay/control.py`
#: answers an unknown `stream_id` and an unparseable frame this way and keeps
#: serving). They are stream-scoped noise wearing a connection-scoped envelope,
#: so they must not be retained as `CloseInfo.error_code`: the expiry
#: sweep's 4401 arrives with *no* preceding error frame, and pairing it with a
#: stale `relay_stream_unknown` would misattribute the closure in
#: `nerdit.core.link.manager.LinkStatus` and in the durable
#: `link.disconnected` payload.
MID_SESSION_ERROR_CODES = frozenset({"relay_stream_unknown", "relay_frame_invalid"})


class LinkClosedError(Exception):
    """The socket is gone; `close` carries what we know about why."""

    def __init__(self, close: CloseInfo) -> None:
        super().__init__(close.reason or f"link closed (code={close.code})")
        self.close = close


class DialRejected(Exception):  # noqa: N818 - names the wire event, not a class of error
    """The WebSocket upgrade itself was refused with an HTTP status.

    Spec §5 step 1: the relay's capacity/drain refusal calls `close(1013)`
    **before** `accept()`, and the ASGI server delivers that as an **HTTP 403
    rejection of the upgrade** — the 1013 close code never reaches the client.
    So a 403 here is capacity/drain with the same retry semantics as 1013, and
    it must be a distinguishable outcome rather than an opaque dial failure.
    """

    def __init__(self, status: int) -> None:
        super().__init__(f"relay refused the tunnel upgrade with HTTP {status}")
        self.status = status


class LinkHandshakeError(Exception):
    """The handshake did not reach `hello_ack`.

    `phase` is `"challenge"`, `"hello"` or `"ack"`; `close` is the
    closure the relay followed its refusal with (spec: the `error` frame
    precedes the close, so both are captured); `error_code` is that frame's
    stable code, which is what the manager classifies on.
    """

    def __init__(
        self,
        phase: str,
        message: str,
        *,
        close: CloseInfo | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(f"handshake failed at {phase}: {message}")
        self.phase = phase
        self.close = close
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class CloseInfo:
    """Everything one connection knew about its own ending.

    `error_code` is the code of the last **close-adjacent** downlink `error`
    frame seen on this connection. The relay always sends the error frame
    *before* the close (spec §5 steps 4–6), so pairing them here is what lets
    the manager tell an entitlement refusal from an ordinary 4401 — the close
    code alone cannot. The mid-session, stream-scoped codes in
    `MID_SESSION_ERROR_CODES` are excluded from the pairing: they arrive
    on a healthy connection and would misattribute a later, unrelated close.
    """

    code: int | None
    reason: str
    error_code: str | None
    cause: str  # "remote" | "local" | "transport"

    def with_error_code(self, error_code: str | None) -> CloseInfo:
        """Return the same closure annotated with a connection-scoped code."""
        if error_code is None or self.error_code == error_code:
            return self
        return CloseInfo(
            code=self.code, reason=self.reason, error_code=error_code, cause=self.cause
        )


class CloseClass(Enum):
    """What a closure means for the reconnect policy (spec §6)."""

    RETRY_BACKOFF = "retry_backoff"
    AUTH_RETRY = "auth_retry"
    TERMINAL_PROTOCOL = "terminal_protocol"
    TERMINAL_REVOKED = "terminal_revoked"
    TERMINAL_ENTITLEMENT = "terminal_entitlement"
    DISPLACED = "displaced"


#: WP-C4 interim seam (plan §7 "WP-C4 — entitlement (thin in v1)"): relay-side
#: subscription admission is already enforced cloud-side, and the daemon's job
#: in v1 is to map the resulting refusal to a terminal-with-hint state rather
#: than to reconnect-loop against a subscription that will not be granted by
#: retrying. These codes are not in the frozen §4.3 downlink set, so they are
#: matched tolerantly — an unrecognised code stays an ordinary 4401.
ENTITLEMENT_ERROR_CODES = frozenset({"entitlement_required", "subscription_required"})

_CLOSE_PROTOCOL_UNSUPPORTED = 4400
_CLOSE_UNAUTHORIZED = 4401
_CLOSE_REVOKED = 4403
_CLOSE_REPLACED = 4409


def classify_close(close: CloseInfo) -> CloseClass:
    """Map a closure onto its retry policy (spec §6, with the §7 caveat).

    * **4400** — never retry the same protocol version.
    * **4401** — identity/authorization *or* a relay that started draining
      mid-handshake: the wire cannot tell them apart (§7). Normative guidance
      is bounded retries with freshly minted material, so this is
      `CloseClass.AUTH_RETRY` and the manager counts. The one
      distinguishable case is an entitlement refusal, which retrying cannot fix.
    * **4403** — revoked/unlinked: stop until re-linked.
    * **4409** — routine displacement; the displaced socket must not
      reconnect-fight.
    * everything else (1000, 1001, 1013, an absent code, a code a newer relay
      invents) — back off and retry.
    """
    if close.code == _CLOSE_PROTOCOL_UNSUPPORTED:
        return CloseClass.TERMINAL_PROTOCOL
    if close.code == _CLOSE_UNAUTHORIZED:
        if close.error_code in ENTITLEMENT_ERROR_CODES:
            return CloseClass.TERMINAL_ENTITLEMENT
        return CloseClass.AUTH_RETRY
    if close.code == _CLOSE_REVOKED:
        return CloseClass.TERMINAL_REVOKED
    if close.code == _CLOSE_REPLACED:
        return CloseClass.DISPLACED
    return CloseClass.RETRY_BACKOFF


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class LinkSocket(Protocol):
    """Minimal duplex text socket the connection needs.

    A `Protocol` so the whole handshake/heartbeat/close matrix is drivable
    in-process by a test double, and so the manager can be handed any
    connector (the in-process `FakeRelay` uses a plain loopback `ws://`).
    """

    async def send(self, message: str) -> None:
        """Send one JSON text frame; raise `LinkClosedError` if gone."""

    async def recv(self) -> str:
        """Return the next text frame; raise `LinkClosedError` if gone."""

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the socket, tolerating an already-closed one."""

    def close_info(self) -> CloseInfo | None:
        """Return the closure once closed, else `None`."""


Connector = Callable[[], Awaitable[LinkSocket]]


def dial_url(relay_url: str) -> str:
    """Translate a configured relay URL into the websocket dial URL.

    `[link].relay_url` is validated as `wss`/`https` (a relay fronted by
    an HTTP/2 endpoint is configured with its `https` origin, D-R3), and the
    node link is mounted at `/v1/connect` (spec §1). Both spellings therefore
    have to end up at the same socket, and appending the path twice must be
    impossible for an operator who already wrote it out.
    """
    parts = urlsplit(relay_url)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    if not path.endswith(CONNECT_PATH):
        path = f"{path}{CONNECT_PATH}"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


class _WebSocketLinkSocket:
    """`websockets`-backed `LinkSocket`.

    Folds `OSError` in with `ConnectionClosed` for the same reason the
    cloud's transport does: an `ECONNRESET` on a write *is* the peer being
    gone, and a bare socket error escaping the receive loop would end it for
    good, leaving a daemon with no tunnel and no attempt to get one back.
    """

    __slots__ = ("_closed", "_connection", "_local_close")

    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection
        self._closed: CloseInfo | None = None
        self._local_close = False

    def _capture(self, exc: BaseException | None = None) -> CloseInfo:
        if self._closed is not None:
            return self._closed
        code = self._connection.close_code
        reason = self._connection.close_reason or ""
        if code is None:
            cause = "transport"
            reason = reason or (str(exc) if exc is not None else "connection lost")
        elif self._local_close:
            cause = "local"
        else:
            cause = "remote"
        self._closed = CloseInfo(code=code, reason=reason, error_code=None, cause=cause)
        return self._closed

    async def send(self, message: str) -> None:
        """Send one text frame."""
        try:
            await self._connection.send(message)
        except (ConnectionClosed, OSError) as exc:
            raise LinkClosedError(self._capture(exc)) from exc

    async def recv(self) -> str:
        """Return the next text frame, refusing binary traffic."""
        try:
            message = await self._connection.recv()
        except (ConnectionClosed, OSError) as exc:
            raise LinkClosedError(self._capture(exc)) from exc
        if not isinstance(message, str):
            # Every node-link frame is JSON text (spec §1). A binary frame is
            # either a different protocol or an attack surface; neither is
            # something to guess at.
            await self.close(1003, "frames must be text")
            raise LinkClosedError(self._capture())
        return message

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the socket; an already-closed one is not an error."""
        self._local_close = True
        with suppress(ConnectionClosed, OSError):  # an already-closed socket is fine
            await self._connection.close(code, reason)
        self._capture()

    def close_info(self) -> CloseInfo | None:
        """Return the closure once the socket has ended."""
        if self._closed is not None:
            return self._closed
        if self._connection.close_code is not None:
            return self._capture()
        return None


async def dial_relay(relay_url: str) -> LinkSocket:
    """Open one outbound WSS connection to the relay.

    An `InvalidStatus` becomes `DialRejected` because the pre-accept
    refusal is an HTTP status, not a close frame (spec §5 step 1).

    No `Sec-WebSocket-Protocol` is offered — deliberately, and it is worth
    stating why, because the WP-C1 checklist reads "with subprotocol
    `node-link/v1`".

    `node-link/v1` is **not a legal subprotocol token**. RFC 6455 §4.1 defines
    the header's values as RFC 7230 `token`\\ s, and `/` is a separator, not
    a token character. `websockets` enforces that grammar on both ends:
    `validate_subprotocols` raises `ValueError` before a single byte leaves
    the client, and a server parsing the header answers **HTTP 400**. So the
    literal reading of the checklist item is not merely unimplementable, it is
    actively harmful — it turns every dial into either a local crash or a
    refused upgrade.

    The wire itself never negotiates one either: the relay calls
    `websocket.accept()` with no subprotocol (`relay/control.py:1022`) and
    the cloud's own reference daemon offers none. Version negotiation on this
    link is **in-band**, by the `hello.protocol` field, and its refusal is the
    `protocol_version_unsupported` error frame plus close 4400 (spec §5 step
    4, §6) — which is exactly what `LinkConnection.handshake` sends and
    what `classify_close` maps to
    `CloseClass.TERMINAL_PROTOCOL`. The checklist's intent — one pinned
    protocol version, refused loudly rather than guessed at — is therefore
    honoured where the protocol actually puts it.
    """
    url = dial_url(relay_url)
    try:
        connection = await ws_connect(
            url,
            open_timeout=HANDSHAKE_TIMEOUT_S,
            max_size=MAX_FRAME_BYTES,
        )
    except InvalidStatus as exc:
        raise DialRejected(exc.response.status_code) from exc
    return _WebSocketLinkSocket(connection)


# ---------------------------------------------------------------------------
# The connection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandshakeResult:
    """What `hello_ack` told us to obey for the life of this connection."""

    node_id: str
    heartbeat_interval_s: float
    limits: StreamLimits
    relay_id: str


class LinkConnection:
    """The lifecycle of one accepted node-link connection."""

    def __init__(  # noqa: PLR0913 - identity + metadata are the hello's own fields
        self,
        socket: LinkSocket,
        *,
        identity: NodeIdentity,
        node_id: str,
        node_name: str,
        daemon_version: str,
        uptime_s: Callable[[], int],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._socket = socket
        self._identity = identity
        self._node_id = node_id
        self._node_name = node_name
        self._daemon_version = daemon_version
        self._uptime_s = uptime_s
        self._sleep = sleep
        self._send_lock = asyncio.Lock()
        self._error_code: str | None = None
        self._result: HandshakeResult | None = None

    # -- handshake ---------------------------------------------------------

    async def handshake(self, capability: MintedCapability) -> HandshakeResult:
        """Run `node_challenge` → signed `hello` → `hello_ack`.

        The capability is proven, never persisted: it lives in the caller's
        memory for this connection only (ADR-W1, D-R2), and nothing in this
        method logs or returns it.
        """
        challenge = await self._expect(NodeChallenge, "challenge")

        # ONE uptime sample, used for both the signed bytes and the frame. The
        # proof binds all ten fields (spec §5 step 5), so a second read here —
        # a second is easily crossed between the two — would produce a proof
        # the relay verifies against a hello it did not sign.
        uptime = self._uptime_s()
        proof = self._identity.sign(
            proof_message(
                challenge=challenge.challenge,
                relay_id=challenge.relay_id,
                protocol=PROTOCOL_VERSION,
                node_id=self._node_id,
                node_name=self._node_name,
                daemon_version=self._daemon_version,
                uptime_s=uptime,
                capability_token=capability.token,
                capability_expires_at=capability.expires_at,
                capability_role="submitter",
            )
        )
        try:
            await self.send_frame(
                hello_frame(
                    node_id=self._node_id,
                    node_name=self._node_name,
                    daemon_version=self._daemon_version,
                    uptime_s=uptime,
                    capability=capability,
                    proof=proof,
                )
            )
        except LinkClosedError as exc:
            raise LinkHandshakeError(
                "hello",
                "the relay closed before the hello was sent",
                close=exc.close.with_error_code(self._error_code),
                error_code=self._error_code,
            ) from exc

        ack = await self._expect(HelloAck, "ack")
        if ack.node_id != self._node_id:
            await self.close(1002, "hello_ack named a different node")
            raise LinkHandshakeError(
                "ack",
                "hello_ack named a different node_id",
                close=self._socket.close_info(),
            )
        result = HandshakeResult(
            node_id=ack.node_id,
            heartbeat_interval_s=ack.heartbeat_interval_s,
            limits=ack.limits,
            relay_id=challenge.relay_id,
        )
        self._result = result
        return result

    async def _expect(self, expected: type[_FrameT], phase: str) -> _FrameT:
        """Receive one frame and require it to be of `expected` type."""
        try:
            raw = await asyncio.wait_for(self._socket.recv(), HANDSHAKE_TIMEOUT_S)
        except LinkClosedError as exc:
            raise LinkHandshakeError(
                phase,
                "the relay closed the connection",
                close=exc.close.with_error_code(self._error_code),
                error_code=self._error_code,
            ) from exc
        except TimeoutError as exc:
            await self.close(1002, "handshake timed out")
            raise LinkHandshakeError(
                phase, f"the relay did not answer within {HANDSHAKE_TIMEOUT_S:.0f}s"
            ) from exc

        try:
            frame = parse_downlink(raw)
        except FrameError as exc:
            await self.close(1002, "unparseable handshake frame")
            raise LinkHandshakeError(phase, str(exc)) from exc

        if isinstance(frame, ErrorFrame):
            # The relay always sends the error frame *before* the close (spec
            # §5 steps 4–6), so waiting for the close here is what pairs the
            # stable code with the close code the manager classifies on.
            self._error_code = frame.code
            close = await self._await_close()
            raise LinkHandshakeError(
                phase,
                f"the relay refused the handshake ({frame.code})",
                close=close,
                error_code=frame.code,
            )
        if not isinstance(frame, expected):
            await self.close(1002, "unexpected handshake frame")
            raise LinkHandshakeError(
                phase, f"expected {expected.__name__}, got {type(frame).__name__}"
            )
        return frame

    async def _await_close(self) -> CloseInfo | None:
        """Drain until the relay's close lands, bounded by the same deadline."""
        try:
            async with asyncio.timeout(HANDSHAKE_TIMEOUT_S):
                while True:
                    await self._socket.recv()
        except LinkClosedError as exc:
            return exc.close.with_error_code(self._error_code)
        except TimeoutError:
            # The relay said no and then held the socket open; do not wait on
            # it any longer than it gave us.
            await self.close(1002, "refusal without a close")
            info = self._socket.close_info()
            return info.with_error_code(self._error_code) if info else None

    # -- session -----------------------------------------------------------

    async def send_frame(self, frame: Mapping[str, object]) -> None:
        """Serialize and send one uplink frame.

        The lock is not decoration: the heartbeat task and every stream task
        share this socket, and `websockets` interleaving two concurrent sends
        would produce a frame no `extra="forbid"` model can parse.
        """
        message = dump_frame(frame)
        async with self._send_lock:
            await self._socket.send(message)

    async def run(self, handler: InboundStreamHandler) -> CloseInfo:
        """Serve the connection until it closes; return why it did.

        `handler.handle` must return promptly — it spawns per-stream tasks.
        Blocking it would stall every other stream *and* the close detection,
        which is why the mux owns tasks and this loop owns only dispatch.
        """
        if self._result is None:
            raise RuntimeError("handshake() must succeed before run()")
        heartbeat = asyncio.create_task(self._heartbeat_loop(self._result.heartbeat_interval_s))
        try:
            while True:
                raw = await self._socket.recv()
                try:
                    frame = parse_downlink(raw)
                except FrameError as exc:
                    # Mid-session invalids are non-fatal on the relay side too
                    # (spec §5 "Mid-session"): it answers `relay_frame_invalid`
                    # and keeps the connection open. Dropping one bad frame
                    # must not cost a live tunnel and every stream on it.
                    logger.warning("Dropping an invalid downlink frame: %s", exc)
                    continue
                await self._dispatch(frame, handler)
        except LinkClosedError as exc:
            # The only ordinary end of a session: the socket went away, either
            # under the receive or under a handler's own uplink send.
            return exc.close.with_error_code(self._error_code)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

    async def _dispatch(self, frame: Downlink, handler: InboundStreamHandler) -> None:
        """Route one parsed downlink frame."""
        if isinstance(frame, ErrorFrame):
            # Connection-scoped; the close normally follows. Remember the code
            # — it is the only thing that distinguishes the §7 4401 cases —
            # unless it is one of the mid-session, stream-scoped answers the
            # relay sends while staying healthy (see MID_SESSION_ERROR_CODES):
            # retaining one of those would pair it with an unrelated later
            # close. The warning is logged for every code either way.
            logger.warning("Relay reported %s: %s", frame.code, frame.message)
            if frame.code not in MID_SESSION_ERROR_CODES:
                self._error_code = frame.code
            return
        if isinstance(frame, (NodeChallenge, HelloAck)):
            logger.warning("Ignoring a mid-session %s frame", type(frame).__name__)
            return
        try:
            await handler.handle(frame)
        except LinkClosedError:
            raise
        except Exception:
            # A mux failure is one stream's problem; the tunnel keeps serving
            # the other seven. The frame itself is never logged (headers carry
            # the injected bearer).
            logger.exception("The stream handler failed on a %s frame", type(frame).__name__)

    async def _heartbeat_loop(self, interval_s: float) -> None:
        """Beat at the cadence `hello_ack` asked for (spec §5, §8).

        Floored at `MIN_HEARTBEAT_INTERVAL_S`. The cadence is relay-chosen
        and therefore peer-controlled: `0.001` is a well-formed float that
        would turn this loop into a 1 kHz uplink flood against our own send
        lock, starving every stream on the connection. A relay may slow us down;
        it may not speed us up past a sane floor.
        """
        interval = max(interval_s, MIN_HEARTBEAT_INTERVAL_S)
        while True:
            await self._sleep(interval)
            try:
                await self.send_frame(
                    heartbeat_frame(node_id=self._node_id, uptime_s=self._uptime_s())
                )
            except LinkClosedError:
                return

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the underlying socket."""
        await self._socket.close(code, reason)


# ---------------------------------------------------------------------------
# manager <-> mux seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppTarget:
    """Where an app stream dials, and the authority it is told it serves.

    The resolved answer to one `x-nerdit-app` header: a service that has a
    share row, a live job row of `kind=service` and an endpoint. Everything a
    `nerdit.core.link.mux.StreamMux` app stream needs, and deliberately
    nothing else — the resolver, not the mux, owns the policy that produced it.

    `port` is `nerdit.db.rows.ServiceEndpoint.live_port`, never
    `host_port`: during a health-gated cutover the serving container is
    on the transient port, and dialling the reserved one would hand the browser
    the blue generation the daemon has already stopped advertising.

    `host` is the hosted authority `<app>--<slug>.<nodes_base_domain>` the
    mux rewrites `Host` to, so the app builds absolute URLs against the name
    the browser typed rather than a loopback port.
    """

    service_name: str
    port: int
    host: str


@dataclass(frozen=True, slots=True)
class MuxContext:
    """Everything the stream mux needs, and nothing it should reach around for.

    Defined in this module — not in `mux` — so `manager` can build one
    without importing `mux` at module scope. Timing and HTTP are injected
    because the whole limits/pacing/timeout matrix has to be testable without
    real clocks or real sockets.
    """

    node_id: str
    limits: StreamLimits
    #: `LinkConnection.send_frame`; raises `LinkClosedError`.
    send: Callable[[Mapping[str, object]], Awaitable[None]]
    #: `(token, role) -> bool`; `LinkManager.validate_capability`.
    validate_capability: Callable[[str, str], bool]
    #: e.g. `"http://127.0.0.1:9321"` — the daemon's own listener.
    loopback_base_url: str
    http_client_factory: Callable[[], httpx.AsyncClient] | None = None
    #: `(service_name) -> AppTarget | None` (P26 D-P26-H1/H2), built by
    #: `bootstrap.build_link_manager` from the share table. `None` — the
    #: default, and every pre-P26 context — means NO app stream can be served:
    #: a stream carrying `x-nerdit-app` is answered 404 and is **never**
    #: replayed on loopback. Fail-closed is the whole posture: an unlinked,
    #: unshared or domain-less node must not turn a routing header into a
    #: request against its own control plane.
    resolve_app: Callable[[str], Awaitable[AppTarget | None]] | None = None
    #: Test seam for the SECOND client — the app dial. Same shape as
    #: `http_client_factory`; production builds an `httpx.AsyncClient` with
    #: no `base_url`, because an app stream's authority is per-stream.
    app_client_factory: Callable[[], httpx.AsyncClient] | None = None
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


class InboundStreamHandler(Protocol):
    """What `LinkConnection.run` needs from the stream mux."""

    async def handle(self, frame: StreamDownlink) -> None:
        """Accept one stream frame; must not block the receive loop."""

    def active_streams(self) -> int:
        """Return how many streams are being served right now."""

    async def aclose(self) -> None:
        """Cancel every stream and release the HTTP client."""


__all__ = [
    "CONNECT_PATH",
    "ENTITLEMENT_ERROR_CODES",
    "HANDSHAKE_TIMEOUT_S",
    "MAX_FRAME_BYTES",
    "MID_SESSION_ERROR_CODES",
    "MIN_HEARTBEAT_INTERVAL_S",
    "AppTarget",
    "CloseClass",
    "CloseInfo",
    "Connector",
    "DialRejected",
    "HandshakeResult",
    "InboundStreamHandler",
    "LinkClosedError",
    "LinkConnection",
    "LinkHandshakeError",
    "LinkSocket",
    "MuxContext",
    "classify_close",
    "dial_relay",
    "dial_url",
]
