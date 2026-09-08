"""Provide a loopback node-link/v1 relay double for protocol tests.

Serve plain ws on 127.0.0.1:0, replay frozen challenge/ack fixtures, verify hello
with the cloud-pinned proof and four authorization checks, then run the scripted
scenario: frames, close codes, HTTP 403 upgrade refusal or an open connection.

The injected connector bypasses relay_url validation; dial_url supports http to
ws for this harness. TLS is outside these protocol tests. Validate outbound
frames with the independent extra-forbid models in tests/node_link_frames.py.
Production code must not import this module.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from nerdit.core.link.identity import proof_message, verify_node_proof
from tests.node_link_frames import HelloFrame, parse_outbound

FIXTURES = Path(__file__).parent / "data" / "node_link_v1"

#: The node id every vendored fixture carries. Tests that compare a produced
#: frame against a fixture frame must use it, or the comparison differs on that
#: key alone.
FIXTURE_NODE_ID = "00000000-0000-4000-8000-00000000000a"

#: Close codes, named so a scenario reads like the §6 table.
CLOSE_PROTOCOL_UNSUPPORTED = 4400
CLOSE_UNAUTHORIZED = 4401
CLOSE_REVOKED = 4403
CLOSE_REPLACED = 4409

#: The relay's own error codes for the refusals a scenario can stage
#: (spec §4.3 downlink; ``relay/control.py``).
ERROR_AUTH = "node_authentication_failed"
ERROR_PROTOCOL = "protocol_version_unsupported"
ERROR_REVOKED = "node_revoked"

#: ADR-W2 look-ahead ceiling the relay enforces at hello (§5 step 5).
MAX_CAPABILITY_LOOKAHEAD = timedelta(minutes=15)


def load_fixture(name: str) -> Any:
    """Read one vendored fixture as JSON."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Scenario:
    """What the relay does on **one** connection.

    Every field is a distinct point on the §5/§6 matrix. The default is the
    happy path: challenge, verify, ack, then hold the socket open until the
    client closes it (which is what a renewal or a shutdown looks like from
    here).
    """

    #: Refuse the upgrade *before* ``accept()`` with this HTTP status. Spec §5
    #: step 1: the relay's pre-accept ``close(1013)`` reaches the client as an
    #: HTTP 403 upgrade rejection, never as a close frame.
    refuse_upgrade: int | None = None
    #: Send this ``error`` frame + close instead of ``hello_ack``. The tuple is
    #: ``(code, message, close_code)``.
    refuse_hello: tuple[str, str, int] | None = None
    #: Answer a *valid* hello as though it were invalid (drives the §7
    #: drain-vs-credential ambiguity without a bad proof).
    force_unauthorized: bool = False
    #: Downlink frames to inject after ``hello_ack``, in order.
    inject: Sequence[dict[str, Any]] = ()
    #: Close the connection with ``(code, reason)`` after the injections.
    close_with: tuple[int, str] | None = None
    #: Run arbitrary relay-side behaviour after ``hello_ack`` (streaming tests).
    after_ack: Callable[[Session], Awaitable[None]] | None = None
    #: Override the fixture nonce so two attempts sign different challenges.
    challenge: str | None = None


@dataclass(slots=True)
class Session:
    """Everything one connection attempt did, for assertions afterwards."""

    scenario: Scenario
    #: Every uplink frame, as the parsed JSON dict the client actually sent.
    uplink: list[dict[str, Any]] = field(default_factory=list)
    #: The hello, validated under the oracle model. ``None`` if none arrived.
    hello: HelloFrame | None = None
    #: The exact bytes the relay reconstructed and verified the proof over.
    proof_message: bytes | None = None
    proof_valid: bool | None = None
    #: All four §5-step-5 authorization checks together.
    authorized: bool | None = None
    challenge: str = ""
    refused_upgrade: bool = False
    acked: bool = False
    #: The live server socket, for a scenario that wants to drive it directly.
    connection: ServerConnection | None = None
    #: Set once the relay's handler for this connection has returned.
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    def frames(self, frame_type: str) -> list[dict[str, Any]]:
        """Every uplink frame of one type, in arrival order."""
        return [frame for frame in self.uplink if frame.get("type") == frame_type]

    @property
    def capability_token(self) -> str:
        """The capability the client proved on this connection."""
        assert self.hello is not None, "no hello was received on this connection"
        return self.hello.capability.token


# ---------------------------------------------------------------------------
# the relay
# ---------------------------------------------------------------------------


class FakeRelay:
    """A scripted ``node-link/v1`` relay listening on loopback.

    Usage::

        async with FakeRelay(verifier=identity.verifier) as relay:
            relay.script(Scenario(close_with=(4409, "replaced")))
            ...

    Scenarios are consumed one per connection attempt, in order; once the list
    is exhausted the ``default`` scenario is reused indefinitely (so a reconnect
    loop under test keeps finding a relay rather than a refused port).
    """

    def __init__(
        self,
        *,
        verifier: str | None = None,
        node_id: str = FIXTURE_NODE_ID,
        default: Scenario | None = None,
        heartbeat_interval_s: float | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._verifier = verifier
        self._node_id = node_id
        self._default = default or Scenario()
        self._scenarios: list[Scenario] = []
        self._clock = clock or (lambda: datetime.now(UTC))
        self._server: Server | None = None
        self._cm: Any = None
        self._challenge_fixture = load_fixture("node_challenge.json")
        self._ack_fixture = load_fixture("hello_ack.json")
        if heartbeat_interval_s is not None:
            self._ack_fixture = {
                **self._ack_fixture,
                "heartbeat_interval_s": heartbeat_interval_s,
            }
        #: One entry per connection attempt, including refused upgrades.
        self.sessions: list[Session] = []

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> FakeRelay:
        self._cm = serve(
            self._handle,
            "127.0.0.1",
            0,
            process_request=self._process_request,
            # A 10 MiB body base64-inflates past the 1 MiB default, exactly as
            # on the real link (see ``client.MAX_FRAME_BYTES``).
            max_size=16 * 1024 * 1024,
        )
        self._server = await self._cm.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(None, None, None)
        self._cm = None
        self._server = None
        # Give the loop a few turns before the server object becomes garbage.
        # asyncio finalizes a socket transport in ``__del__``, which calls back
        # into the (already closed) ``Server`` — on 3.13 that raises inside the
        # finalizer and pytest reports it as an unraisable-exception warning
        # from whichever unrelated test happened to trigger the collection.
        # Draining here keeps the noise out of other people's test output.
        for _ in range(3):
            await asyncio.sleep(0)

    @property
    def url(self) -> str:
        """The ``http://`` origin to configure — ``dial_url`` maps it to ``ws``."""
        assert self._server is not None, "the relay is not running"
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"http://{host}:{port}"

    def script(self, *scenarios: Scenario) -> None:
        """Queue what the relay does on the next ``len(scenarios)`` dials."""
        self._scenarios.extend(scenarios)

    # -- assertions helpers ------------------------------------------------

    async def wait_sessions(self, count: int, *, timeout: float = 5.0) -> None:
        """Block until ``count`` connection attempts have been made."""
        await self._wait(lambda: len(self.sessions) >= count, timeout, f"{count} sessions")

    async def wait_hello(self, index: int = 0, *, timeout: float = 5.0) -> Session:
        """Block until session ``index`` has received (and judged) its hello."""
        await self.wait_sessions(index + 1, timeout=timeout)
        session = self.sessions[index]
        await self._wait(lambda: session.hello is not None, timeout, "a hello")
        return session

    async def wait_frame(
        self, index: int, frame_type: str, count: int = 1, *, timeout: float = 5.0
    ) -> list[dict[str, Any]]:
        """Block until session ``index`` has received ``count`` frames of a type."""
        await self.wait_sessions(index + 1, timeout=timeout)
        session = self.sessions[index]
        await self._wait(
            lambda: len(session.frames(frame_type)) >= count, timeout, f"{count}x {frame_type}"
        )
        return session.frames(frame_type)

    @staticmethod
    async def _wait(predicate: Callable[[], bool], timeout: float, what: str) -> None:
        """Poll a condition at millisecond granularity (never a real delay)."""
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(f"timed out waiting for {what}")
            await asyncio.sleep(0.001)

    # -- server plumbing ---------------------------------------------------

    def _next_scenario(self) -> Scenario:
        if self._scenarios:
            return self._scenarios.pop(0)
        return self._default

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        """Claim this attempt's scenario, and refuse the upgrade if it says so.

        Runs *before* ``accept()``, which is the only place a pre-accept refusal
        can be staged — the whole point of spec §5 step 1 is that the client
        never sees a close frame for it.
        """
        scenario = self._next_scenario()
        session = Session(scenario=scenario)
        self.sessions.append(session)
        connection.nerdit_session = session  # type: ignore[attr-defined]
        if scenario.refuse_upgrade is not None:
            session.refused_upgrade = True
            session.finished.set()
            return connection.respond(scenario.refuse_upgrade, "relay handshake capacity reached\n")
        return None

    async def _handle(self, connection: ServerConnection) -> None:
        session: Session = connection.nerdit_session  # type: ignore[attr-defined]
        try:
            await self._converse(connection, session)
        except (ConnectionClosed, asyncio.CancelledError):
            pass
        finally:
            session.finished.set()

    async def _converse(self, connection: ServerConnection, session: Session) -> None:
        scenario = session.scenario

        # 1. node_challenge (spec §5 step 2), from the vendored fixture.
        challenge_frame = dict(self._challenge_fixture)
        if scenario.challenge is not None:
            challenge_frame["challenge"] = scenario.challenge
        session.challenge = str(challenge_frame["challenge"])
        await connection.send(json.dumps(challenge_frame))

        # 2. hello (spec §5 steps 4-5).
        hello = await self._read(connection, session)
        if hello is None:
            return
        session.hello = parse_outbound(hello)  # type: ignore[assignment]
        assert isinstance(session.hello, HelloFrame), "the first uplink frame must be a hello"
        session.authorized = self._authorize(session)

        if scenario.refuse_hello is not None:
            code, message, close_code = scenario.refuse_hello
            await self._error_and_close(connection, code, message, close_code)
            return
        if not session.authorized or scenario.force_unauthorized:
            await self._error_and_close(
                connection, ERROR_AUTH, "node authentication failed", CLOSE_UNAUTHORIZED
            )
            return

        # 3. hello_ack (spec §5 step 8), fixture-derived.
        ack = {**self._ack_fixture, "node_id": self._node_id}
        await connection.send(json.dumps(ack))
        session.acked = True

        # 4. whatever this scenario is actually about.
        for frame in scenario.inject:
            await connection.send(json.dumps(frame))
        if scenario.after_ack is not None:
            session.connection = connection
            await scenario.after_ack(session)
        if scenario.close_with is not None:
            code, reason = scenario.close_with
            await connection.close(code, reason)
            return

        # Hold: drain uplink until the client goes away (renewal, shutdown).
        while True:
            if await self._read(connection, session) is None:
                return

    def _authorize(self, session: Session) -> bool:
        """The four §5-step-5 checks, in the relay's own order."""
        hello = session.hello
        assert hello is not None
        now = self._clock()
        if hello.capability.expires_at <= now:
            return False
        if hello.capability.expires_at > now + MAX_CAPABILITY_LOOKAHEAD:
            return False
        if hello.capability.role != "submitter":
            return False
        message = proof_message(
            challenge=session.challenge,
            relay_id=str(self._challenge_fixture["relay_id"]),
            protocol=hello.protocol,
            node_id=hello.node_id,
            node_name=hello.node_name,
            daemon_version=hello.daemon_version,
            uptime_s=hello.uptime_s,
            capability_token=hello.capability.token,
            capability_expires_at=hello.capability.expires_at,
            capability_role=hello.capability.role,
        )
        session.proof_message = message
        if self._verifier is None:
            session.proof_valid = None
            return True
        session.proof_valid = verify_node_proof(self._verifier, message, hello.proof)
        return session.proof_valid

    async def _read(self, connection: ServerConnection, session: Session) -> dict[str, Any] | None:
        """Receive one uplink frame, recording it. ``None`` once closed."""
        try:
            raw = await connection.recv()
        except ConnectionClosed:
            return None
        assert isinstance(raw, str), "node-link frames are JSON text (spec §1)"
        frame = json.loads(raw)
        session.uplink.append(frame)
        # Oracle check on the way in: a frame the relay's own ``extra="forbid"``
        # models would refuse must fail here, not in a live run.
        parse_outbound(frame)
        return frame

    @staticmethod
    async def _error_and_close(
        connection: ServerConnection, code: str, message: str, close_code: int
    ) -> None:
        """Error frame **then** close — the ordering the client relies on."""
        with suppress(ConnectionClosed):
            await connection.send(json.dumps({"type": "error", "code": code, "message": message}))
            await connection.close(close_code, message)


# ---------------------------------------------------------------------------
# downlink frame builders (relay side, for ``Scenario.inject``)
# ---------------------------------------------------------------------------


def open_stream_frame(  # noqa: PLR0913 - one keyword per wire field, by design
    *,
    stream_id: str,
    node_id: str = FIXTURE_NODE_ID,
    method: str = "GET",
    path: str = "/api/status",
    query: str = "",
    headers: Sequence[tuple[str, str]] | None = None,
    body_b64: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Build an ``open_stream`` with the relay's injected credentials (§4.2)."""
    injected = list(headers or [("accept", "*/*")])
    if token is not None:
        injected += [("authorization", f"Bearer {token}"), ("x-nerdit-role", "submitter")]
    frame: dict[str, Any] = {
        "type": "open_stream",
        "stream_id": stream_id,
        "node_id": node_id,
        "method": method,
        "path": path,
        "query": query,
        "headers": [list(pair) for pair in injected],
        "role": "submitter",
    }
    if body_b64 is not None:
        frame["body_b64"] = body_b64
    return frame


def error_frame(code: str, message: str = "refused") -> dict[str, Any]:
    """Build a connection-scoped ``error`` frame."""
    return {"type": "error", "code": code, "message": message}
