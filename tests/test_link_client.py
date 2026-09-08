"""Test a node-link connection against a real loopback FakeRelay.

Replay frozen challenge/ack fixtures and verify hello with the cloud-pinned
proof reconstructed from its ten wire fields. Pin handshake, acknowledged
heartbeat cadence, close-code classification and golden fixtures. Inject timing
to avoid sleeps longer than milliseconds.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from nerdit.core.link import client as client_module
from nerdit.core.link.capability import MintedCapability, mint_capability
from nerdit.core.link.client import (
    CONNECT_PATH,
    ENTITLEMENT_ERROR_CODES,
    HANDSHAKE_TIMEOUT_S,
    MAX_FRAME_BYTES,
    MID_SESSION_ERROR_CODES,
    MIN_HEARTBEAT_INTERVAL_S,
    CloseClass,
    CloseInfo,
    DialRejected,
    LinkClosedError,
    LinkConnection,
    LinkHandshakeError,
    classify_close,
    dial_relay,
    dial_url,
)
from nerdit.core.link.frames import PROTOCOL_VERSION, StreamDownlink
from nerdit.core.link.identity import NodeIdentity, proof_message, verify_node_proof
from tests.link_fake_relay import (
    CLOSE_PROTOCOL_UNSUPPORTED,
    CLOSE_REPLACED,
    CLOSE_REVOKED,
    CLOSE_UNAUTHORIZED,
    ERROR_AUTH,
    ERROR_PROTOCOL,
    FIXTURE_NODE_ID,
    FakeRelay,
    Scenario,
    error_frame,
    load_fixture,
    open_stream_frame,
)
from tests.node_link_frames import HeartbeatFrame, HelloFrame

NODE_NAME = "Dev Node"
DAEMON_VERSION = "0.1.0"
UPTIME = 4242


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity() -> NodeIdentity:
    """A fresh Ed25519 identity — never a fixture key, never persisted."""
    raw = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    return NodeIdentity(base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii"))


class _NullHandler:
    """An :class:`InboundStreamHandler` that records and never blocks."""

    def __init__(self) -> None:
        self.seen: list[StreamDownlink] = []

    async def handle(self, frame: StreamDownlink) -> None:
        self.seen.append(frame)

    def active_streams(self) -> int:
        return 0

    async def aclose(self) -> None:
        return None


def _connection(socket: object, identity: NodeIdentity, **kw: object) -> LinkConnection:
    return LinkConnection(
        socket,  # type: ignore[arg-type]
        identity=identity,
        node_id=kw.pop("node_id", FIXTURE_NODE_ID),  # type: ignore[arg-type]
        node_name=NODE_NAME,
        daemon_version=DAEMON_VERSION,
        uptime_s=kw.pop("uptime_s", lambda: UPTIME),  # type: ignore[arg-type]
        **kw,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# dial_url — the one pure function on the dial path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        # The two spellings the settings validator admits (D-R3).
        ("wss://relay.example.com", "wss://relay.example.com/v1/connect"),
        ("https://relay.example.com", "wss://relay.example.com/v1/connect"),
        # Trailing slashes must not produce ``//v1/connect``.
        ("https://relay.example.com/", "wss://relay.example.com/v1/connect"),
        ("wss://relay.example.com///", "wss://relay.example.com/v1/connect"),
        # An operator who already wrote the path must not get it twice.
        ("wss://relay.example.com/v1/connect", "wss://relay.example.com/v1/connect"),
        ("https://relay.example.com/v1/connect/", "wss://relay.example.com/v1/connect"),
        # A relay behind a path prefix keeps it.
        ("https://edge.example.com/relay", "wss://edge.example.com/relay/v1/connect"),
        # Ports survive; so does an IPv6 literal.
        ("wss://relay.example.com:8443", "wss://relay.example.com:8443/v1/connect"),
        ("https://[::1]:9000", "wss://[::1]:9000/v1/connect"),
        # http→ws exists only so an in-process loopback relay is reachable
        # through the same translation; the settings validator refuses it.
        ("http://127.0.0.1:1234", "ws://127.0.0.1:1234/v1/connect"),
    ],
)
def test_dial_url_translation(configured: str, expected: str) -> None:
    assert dial_url(configured) == expected


def test_dial_url_never_forwards_a_query_or_fragment() -> None:
    """The relay mounts one path; anything else is operator paste noise."""
    assert dial_url("wss://relay.example.com/?token=leaked#frag") == (
        "wss://relay.example.com/v1/connect"
    )


def test_connect_path_matches_the_spec() -> None:
    assert CONNECT_PATH == "/v1/connect"


def test_frame_cap_admits_a_base64_inflated_max_body() -> None:
    """Spec §8 allows a 10 MiB body; base64 makes that ~13.3 MiB on the wire.

    The ``websockets`` default is 1 MiB, so an explicit cap is not tuning — it
    is what keeps a legal request from being refused as an oversized frame.
    """
    assert MAX_FRAME_BYTES > 10 * 1024 * 1024 * 4 / 3


# ---------------------------------------------------------------------------
# the golden handshake
# ---------------------------------------------------------------------------


async def test_golden_handshake_against_the_vendored_fixtures(identity: NodeIdentity) -> None:
    """The WP-C1 item-6 mandate, end to end on a real socket.

    Asserted here: the relay's fixture ``node_challenge`` is answered by a
    ``hello`` that (a) validates under the **oracle** ``HelloFrame`` model —
    i.e. the relay's own ``extra="forbid"`` shape — (b) carries the pinned
    protocol token, (c) proves out under ``verify_node_proof`` against a
    ``proof_message`` the relay reconstructed from the wire fields alone, and
    (d) is answered with the fixture ``hello_ack`` whose seven §8 limits reach
    the caller unchanged.
    """
    capability = mint_capability(600)
    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        result = await connection.handshake(capability)
        await connection.close()

    session = relay.sessions[0]

    # (a) the hello is exactly the frozen shape
    hello = session.hello
    assert isinstance(hello, HelloFrame)
    assert hello.protocol == PROTOCOL_VERSION == "node-link/v1"
    assert hello.node_id == FIXTURE_NODE_ID
    assert hello.node_name == NODE_NAME
    assert hello.daemon_version == DAEMON_VERSION
    assert hello.uptime_s == UPTIME
    assert hello.capability.token == capability.token
    assert hello.capability.expires_at == capability.expires_at
    assert hello.capability.role == "submitter"

    # (b)+(c) the relay verified the proof over ten fields it read off the wire
    assert session.proof_valid is True
    assert session.authorized is True
    assert session.proof_message is not None
    # The same bytes, reconstructed here a third time, still verify: this is
    # what makes the assertion about the *contract* and not about the relay
    # double's own bookkeeping.
    expected = proof_message(
        challenge=session.challenge,
        relay_id=load_fixture("node_challenge.json")["relay_id"],
        protocol=PROTOCOL_VERSION,
        node_id=FIXTURE_NODE_ID,
        node_name=NODE_NAME,
        daemon_version=DAEMON_VERSION,
        uptime_s=UPTIME,
        capability_token=capability.token,
        capability_expires_at=capability.expires_at,
        capability_role="submitter",
    )
    assert session.proof_message == expected
    assert verify_node_proof(identity.verifier, expected, hello.proof) is True

    # (d) the acked limits are the seven §8 numbers, unmodified
    assert result.node_id == FIXTURE_NODE_ID
    assert result.relay_id == load_fixture("node_challenge.json")["relay_id"]
    assert result.heartbeat_interval_s == 5.0
    ack_limits = load_fixture("hello_ack.json")["limits"]
    for field, value in ack_limits.items():
        assert getattr(result.limits, field) == value, field
    assert result.limits.max_concurrent_streams_per_node == 8
    assert result.limits.max_body_bytes == 10 * 1024 * 1024
    assert result.limits.max_response_body_bytes == 50 * 1024 * 1024
    assert result.limits.max_bandwidth_bytes_per_second == 10 * 1024 * 1024


async def test_the_proof_binds_a_single_uptime_sample(identity: NodeIdentity) -> None:
    """A second ``uptime_s()`` read between signing and sending breaks the proof.

    The relay verifies over ``hello.uptime_s``; if the frame carried a different
    sample than the signed bytes the proof would fail. Driving ``uptime_s`` from
    a counter that advances on every call is the direct way to pin "read once".
    """
    calls = 0

    def ticking_uptime() -> int:
        nonlocal calls
        calls += 1
        return calls

    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity, uptime_s=ticking_uptime)
        await connection.handshake(mint_capability(600))
        await connection.close()

    session = relay.sessions[0]
    assert session.proof_valid is True
    assert calls == 1, "the handshake must sample uptime exactly once"
    assert session.hello is not None
    assert session.hello.uptime_s == 1


async def test_a_tampered_proof_is_refused_by_the_relay(identity: NodeIdentity) -> None:
    """Negative control: the relay double really is checking the signature."""
    other = NodeIdentity(
        base64.urlsafe_b64encode(
            Ed25519PrivateKey.generate().private_bytes(
                Encoding.Raw, PrivateFormat.Raw, NoEncryption()
            )
        )
        .rstrip(b"=")
        .decode("ascii")
    )
    async with FakeRelay(verifier=other.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError) as excinfo:
            await connection.handshake(mint_capability(600))

    assert relay.sessions[0].proof_valid is False
    error = excinfo.value
    assert error.error_code == ERROR_AUTH
    assert error.close is not None
    assert error.close.code == CLOSE_UNAUTHORIZED
    assert classify_close(error.close) is CloseClass.AUTH_RETRY


async def test_an_expired_capability_is_refused_at_hello(identity: NodeIdentity) -> None:
    """§5 step 5: ``expires_at > now``. Our own clock is not the relay's."""
    stale = MintedCapability(
        token="stale", expires_at=datetime.now(UTC) - timedelta(seconds=1), role="submitter"
    )
    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError):
            await connection.handshake(stale)
    assert relay.sessions[0].authorized is False


async def test_an_overlong_capability_is_refused_at_hello(identity: NodeIdentity) -> None:
    """§5 step 5 / ADR-W2: ``expires_at <= now + 15 minutes``."""
    overlong = MintedCapability(
        token="too-long", expires_at=datetime.now(UTC) + timedelta(hours=1), role="submitter"
    )
    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError):
            await connection.handshake(overlong)
    assert relay.sessions[0].authorized is False


# ---------------------------------------------------------------------------
# handshake failure surfaces
# ---------------------------------------------------------------------------


async def test_dial_rejected_on_a_pre_accept_403(identity: NodeIdentity) -> None:
    """Spec §5 step 1: capacity/drain refusal is an HTTP status, not a close.

    A daemon that only classified close frames would see this as an opaque dial
    failure; it has to be a distinguishable, retryable outcome.
    """
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(Scenario(refuse_upgrade=403))
        with pytest.raises(DialRejected) as excinfo:
            await dial_relay(relay.url)
    assert excinfo.value.status == 403
    assert "403" in str(excinfo.value)
    assert relay.sessions[0].refused_upgrade is True
    # The refusal never reached the frame layer: no hello was ever sent.
    assert relay.sessions[0].hello is None


async def test_protocol_refusal_surfaces_the_4400_close(identity: NodeIdentity) -> None:
    """§5 step 4 / §6: unsupported version → error frame then close 4400."""
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(
            Scenario(
                refuse_hello=(ERROR_PROTOCOL, "unsupported protocol", CLOSE_PROTOCOL_UNSUPPORTED)
            )
        )
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError) as excinfo:
            await connection.handshake(mint_capability(600))

    error = excinfo.value
    assert error.phase == "ack"
    assert error.error_code == ERROR_PROTOCOL
    assert error.close is not None
    assert error.close.code == CLOSE_PROTOCOL_UNSUPPORTED
    # The error frame and the close are paired — the whole reason the client
    # waits for the close after an error frame.
    assert error.close.error_code == ERROR_PROTOCOL
    assert classify_close(error.close) is CloseClass.TERMINAL_PROTOCOL


async def test_an_entitlement_refusal_is_distinguishable_from_an_ordinary_4401(
    identity: NodeIdentity,
) -> None:
    """WP-C4 seam: 4401 + an entitlement code is terminal, not a retry.

    The close code alone cannot tell them apart — that is exactly why the error
    frame's code is carried onto :class:`CloseInfo`.
    """
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(
            Scenario(refuse_hello=("entitlement_required", "subscribe", CLOSE_UNAUTHORIZED))
        )
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError) as excinfo:
            await connection.handshake(mint_capability(600))

    close = excinfo.value.close
    assert close is not None
    assert close.code == CLOSE_UNAUTHORIZED
    assert close.error_code == "entitlement_required"
    assert classify_close(close) is CloseClass.TERMINAL_ENTITLEMENT


async def test_hello_ack_naming_another_node_is_refused(identity: NodeIdentity) -> None:
    """A relay that acks a different ``node_id`` is not talking to us."""
    async with FakeRelay(verifier=identity.verifier, node_id="somebody-else") as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(LinkHandshakeError) as excinfo:
            await connection.handshake(mint_capability(600))
    assert excinfo.value.phase == "ack"
    assert "node_id" in str(excinfo.value)


async def test_a_closed_socket_during_the_challenge_phase_is_a_handshake_error(
    identity: NodeIdentity,
) -> None:
    class _DeadSocket:
        async def send(self, message: str) -> None:  # pragma: no cover - never reached
            raise AssertionError("nothing may be sent before the challenge")

        async def recv(self) -> str:
            raise LinkClosedError(
                CloseInfo(code=1013, reason="capacity", error_code=None, cause="remote")
            )

        async def close(self, code: int = 1000, reason: str = "") -> None:
            return None

        def close_info(self) -> CloseInfo | None:
            return None

    connection = _connection(_DeadSocket(), identity)
    with pytest.raises(LinkHandshakeError) as excinfo:
        await connection.handshake(mint_capability(600))
    assert excinfo.value.phase == "challenge"
    assert excinfo.value.close is not None
    assert excinfo.value.close.code == 1013
    assert classify_close(excinfo.value.close) is CloseClass.RETRY_BACKOFF


async def test_a_stalled_relay_times_the_handshake_out(
    identity: NodeIdentity, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay's own deadline is 10 s; ours mirrors it **per phase**.

    The constant is the contract; the ten seconds are not something to spend in
    CI, so the deadline is shortened at the module global the client reads at
    call time.
    """
    assert HANDSHAKE_TIMEOUT_S == 10.0
    monkeypatch.setattr(client_module, "HANDSHAKE_TIMEOUT_S", 0.01)
    never = asyncio.Event()

    class _SilentSocket:
        def __init__(self) -> None:
            self.closed_with: tuple[int, str] | None = None

        async def send(self, message: str) -> None:  # pragma: no cover
            return None

        async def recv(self) -> str:
            await never.wait()
            raise AssertionError("unreachable")

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed_with = (code, reason)
            never.set()

        def close_info(self) -> CloseInfo | None:
            return None

    socket = _SilentSocket()
    connection = _connection(socket, identity)
    with pytest.raises(LinkHandshakeError) as excinfo:
        await connection.handshake(mint_capability(600))
    assert excinfo.value.phase == "challenge"
    assert socket.closed_with == (1002, "handshake timed out")


# ---------------------------------------------------------------------------
# the session: heartbeat, dispatch, close
# ---------------------------------------------------------------------------


async def test_heartbeat_beats_at_the_acked_cadence(identity: NodeIdentity) -> None:
    """Item 1: *heartbeat at the ACKED cadence* — 5.0 s, from ``hello_ack``.

    Injected sleep, so the cadence is asserted as a **number the relay chose**
    rather than as elapsed wall-clock time.
    """
    requested: list[float] = []
    release = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        requested.append(delay)
        await release.wait()
        release.clear()

    async with FakeRelay(verifier=identity.verifier, heartbeat_interval_s=17.5) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity, sleep=fake_sleep)
        result = await connection.handshake(mint_capability(600))
        assert result.heartbeat_interval_s == 17.5

        session_task = asyncio.create_task(connection.run(_NullHandler()))
        # Three beats, each gated on the injected sleep returning.
        for beat in range(1, 4):
            await FakeRelay._wait(
                lambda expected=beat: len(requested) >= expected, 2.0, "a heartbeat sleep"
            )
            release.set()
            await relay.wait_frame(0, "heartbeat", beat)
        await connection.close()
        close = await session_task

    assert requested[:3] == [17.5, 17.5, 17.5]
    beats = relay.sessions[0].frames("heartbeat")
    assert len(beats) >= 3
    for beat in beats:
        parsed = HeartbeatFrame.model_validate(beat)
        assert parsed.node_id == FIXTURE_NODE_ID
        assert parsed.uptime_s == UPTIME
        # v1 omits it deliberately: free text about node activity, no consumer.
        assert parsed.workload_state is None
        assert "workload_state" not in beat
    assert close.cause == "local"


@pytest.mark.parametrize("advertised", [0.001, 0.0, -5.0, 0.999])
async def test_an_absurd_acked_cadence_is_floored(
    identity: NodeIdentity, advertised: float
) -> None:
    """The cadence is relay-chosen, so it is peer-controlled.

    ``0.001`` is a perfectly well-formed float that would turn the heartbeat
    loop into a 1 kHz uplink flood against the connection's own send lock,
    starving every stream on it. A relay may slow us down; it may not speed us
    up past a sane floor.
    """
    requested: list[float] = []
    gate = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        requested.append(delay)
        await gate.wait()

    async with FakeRelay(verifier=identity.verifier, heartbeat_interval_s=advertised) as relay:
        relay.script(Scenario(close_with=(1001, "draining")))
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity, sleep=fake_sleep)
        result = await connection.handshake(mint_capability(600))
        assert result.heartbeat_interval_s == advertised  # obeyed on the wire…
        await connection.run(_NullHandler())

    assert requested == [MIN_HEARTBEAT_INTERVAL_S]  # …floored where it costs
    gate.set()
    await asyncio.sleep(0)


async def test_the_heartbeat_task_is_cancelled_when_the_session_ends(
    identity: NodeIdentity,
) -> None:
    """A leaked beat task would outlive its socket and log forever."""
    sleeps = 0
    gate = asyncio.Event()

    async def fake_sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        await gate.wait()

    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(Scenario(close_with=(1001, "draining")))
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity, sleep=fake_sleep)
        await connection.handshake(mint_capability(600))
        close = await connection.run(_NullHandler())

    assert close.code == 1001
    assert sleeps == 1
    # If the task had leaked, releasing the gate would make it send on a dead
    # socket; it is already cancelled, so nothing happens.
    gate.set()
    await asyncio.sleep(0)


async def test_stream_frames_reach_the_handler_and_invalid_ones_do_not_kill_the_session(
    identity: NodeIdentity,
) -> None:
    """Spec §5 "Mid-session": a bad frame costs the frame, never the tunnel."""
    handler = _NullHandler()
    injections = [
        {"type": "open_tunnel"},  # unknown type
        {"type": "stream_end", "stream_id": "s1", "node_id": FIXTURE_NODE_ID, "bogus": 1},
        # A mid-session challenge/ack is nonsense but must not be fatal either.
        load_fixture("node_challenge.json"),
        load_fixture("hello_ack.json"),
        open_stream_frame(stream_id="s1", token="tok"),
        {"type": "stream_end", "stream_id": "s1", "node_id": FIXTURE_NODE_ID},
    ]
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(Scenario(inject=injections, close_with=(1000, "done")))
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(handler)

    assert [type(frame).__name__ for frame in handler.seen] == ["OpenStream", "StreamEnd"]
    assert close.code == 1000


async def test_a_handler_exception_is_contained(identity: NodeIdentity) -> None:
    """One stream's failure must not end the other seven."""

    class _AngryHandler(_NullHandler):
        async def handle(self, frame: StreamDownlink) -> None:
            self.seen.append(frame)
            raise RuntimeError("boom")

    handler = _AngryHandler()
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(
            Scenario(
                inject=[
                    open_stream_frame(stream_id="s1", token="t"),
                    open_stream_frame(stream_id="s2", token="t"),
                ],
                close_with=(1000, "done"),
            )
        )
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(handler)

    assert len(handler.seen) == 2  # the second frame was still dispatched
    assert close.code == 1000


async def test_a_mid_session_error_frame_annotates_the_close(identity: NodeIdentity) -> None:
    """The manager classifies on the pair, not on the close code alone."""
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(
            Scenario(
                inject=[error_frame("entitlement_required", "subscription lapsed")],
                close_with=(CLOSE_UNAUTHORIZED, "unauthorized"),
            )
        )
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(_NullHandler())

    assert close.code == CLOSE_UNAUTHORIZED
    assert close.error_code == "entitlement_required"
    assert close.cause == "remote"
    assert classify_close(close) is CloseClass.TERMINAL_ENTITLEMENT


@pytest.mark.parametrize("code", sorted(MID_SESSION_ERROR_CODES))
async def test_a_stream_scoped_error_frame_does_not_annotate_a_later_close(
    identity: NodeIdentity, code: str
) -> None:
    """The pairing invariant, protected from noise.

    The relay answers an unknown ``stream_id`` and an unparseable uplink frame
    with a *connection-scoped* ``error`` and then keeps serving. The expiry
    sweep's 4401, by contrast, arrives with no preceding error frame at all —
    so retaining one of these would pair an unrelated stream-scoped code with a
    later close, surfacing it as ``last_error_code`` in ``LinkStatus`` and in
    the durable ``link.disconnected`` payload. Worse, ``classify_close`` reads
    that field: a retained ``entitlement_required``-shaped code would flip a
    retryable 4401 to terminal.
    """
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(
            Scenario(
                inject=[error_frame(code, "no such stream")],
                close_with=(CLOSE_UNAUTHORIZED, "capability expired"),
            )
        )
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(_NullHandler())

    assert close.code == CLOSE_UNAUTHORIZED
    assert close.error_code is None
    # Still an ordinary retryable 4401 — bounded retries with fresh material.
    assert classify_close(close) is CloseClass.AUTH_RETRY


async def test_run_before_handshake_is_a_programming_error(identity: NodeIdentity) -> None:
    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        with pytest.raises(RuntimeError, match="handshake"):
            await connection.run(_NullHandler())
        await connection.close()


async def test_send_frame_on_a_dead_socket_raises_link_closed(identity: NodeIdentity) -> None:
    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(Scenario(close_with=(1000, "done")))
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(_NullHandler())
        assert close.code == 1000
        with pytest.raises(LinkClosedError) as excinfo:
            await connection.send_frame({"type": "heartbeat", "node_id": "n", "uptime_s": 1})
    assert excinfo.value.close.code == 1000


async def test_concurrent_sends_never_interleave(identity: NodeIdentity) -> None:
    """The send lock: two interleaved sends would produce an unparseable frame."""
    async with FakeRelay(verifier=identity.verifier) as relay:
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        session_task = asyncio.create_task(connection.run(_NullHandler()))
        await asyncio.gather(
            *(
                connection.send_frame(
                    {"type": "heartbeat", "node_id": FIXTURE_NODE_ID, "uptime_s": index}
                )
                for index in range(30)
            )
        )
        await relay.wait_frame(0, "heartbeat", 30)
        await connection.close()
        await session_task

    beats = relay.sessions[0].frames("heartbeat")
    assert sorted(beat["uptime_s"] for beat in beats[:30]) == list(range(30))


async def test_a_binary_frame_is_refused(identity: NodeIdentity) -> None:
    """Spec §1: every node-link frame is JSON text.

    A binary frame is either a different protocol or an attack surface; the
    socket closes 1003 rather than guessing at a decoding.
    """

    async def send_binary(session) -> None:  # noqa: ANN001
        await session.connection.send(b"\x00\x01\x02")
        await asyncio.sleep(0.2)

    async with FakeRelay(verifier=identity.verifier) as relay:
        relay.script(Scenario(after_ack=send_binary))
        socket = await dial_relay(relay.url)
        connection = _connection(socket, identity)
        await connection.handshake(mint_capability(600))
        close = await connection.run(_NullHandler())

    assert close.cause == "local"
    assert close.code == 1003


# ---------------------------------------------------------------------------
# classify_close — every §6 row
# ---------------------------------------------------------------------------


def _close(code: int | None, error_code: str | None = None) -> CloseInfo:
    return CloseInfo(code=code, reason="", error_code=error_code, cause="remote")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # §6 table, row by row.
        (CLOSE_PROTOCOL_UNSUPPORTED, CloseClass.TERMINAL_PROTOCOL),
        (CLOSE_UNAUTHORIZED, CloseClass.AUTH_RETRY),
        (CLOSE_REVOKED, CloseClass.TERMINAL_REVOKED),
        (CLOSE_REPLACED, CloseClass.DISPLACED),
        (1000, CloseClass.RETRY_BACKOFF),
        (1001, CloseClass.RETRY_BACKOFF),
        (1013, CloseClass.RETRY_BACKOFF),
        # Not in the table: a transport death, and a code a newer relay invents.
        (None, CloseClass.RETRY_BACKOFF),
        (1006, CloseClass.RETRY_BACKOFF),
        (4999, CloseClass.RETRY_BACKOFF),
    ],
)
def test_classify_close_covers_every_spec_row(code: int | None, expected: CloseClass) -> None:
    assert classify_close(_close(code)) is expected


@pytest.mark.parametrize("error_code", sorted(ENTITLEMENT_ERROR_CODES))
def test_4401_with_an_entitlement_code_is_terminal(error_code: str) -> None:
    assert classify_close(_close(CLOSE_UNAUTHORIZED, error_code)) is CloseClass.TERMINAL_ENTITLEMENT


def test_4401_with_the_ordinary_auth_code_stays_retryable() -> None:
    """§7: drain and a bad credential are indistinguishable — so retry, bounded."""
    assert classify_close(_close(CLOSE_UNAUTHORIZED, ERROR_AUTH)) is CloseClass.AUTH_RETRY
    assert classify_close(_close(CLOSE_UNAUTHORIZED, "something_new")) is CloseClass.AUTH_RETRY


def test_an_entitlement_code_on_another_close_code_changes_nothing() -> None:
    """The pair is (4401, entitlement); the code alone must not be terminal."""
    assert classify_close(_close(1013, "entitlement_required")) is CloseClass.RETRY_BACKOFF
    assert classify_close(_close(CLOSE_REPLACED, "entitlement_required")) is CloseClass.DISPLACED


def test_close_info_with_error_code_is_a_pure_annotation() -> None:
    base = _close(4401)
    annotated = base.with_error_code("node_authentication_failed")
    assert base.error_code is None  # frozen: the original is untouched
    assert annotated.error_code == "node_authentication_failed"
    assert (annotated.code, annotated.reason, annotated.cause) == (
        base.code,
        base.reason,
        base.cause,
    )
    # A no-op annotation returns the same object rather than a copy.
    assert annotated.with_error_code(None) is annotated
    assert annotated.with_error_code("node_authentication_failed") is annotated


def test_capability_material_never_renders(identity: NodeIdentity) -> None:
    """Defense in depth for every log line in this module's call graph."""
    capability = mint_capability(600)
    assert capability.token not in repr(capability)
    assert capability.token not in str(capability)
    assert identity.verifier in repr(identity)
    assert "_private_key_b64" not in repr(identity)


def test_the_hello_never_serializes_a_null(identity: NodeIdentity) -> None:
    """Canonical serialization (§3): absent, never ``null``."""
    from nerdit.core.link.frames import dump_frame, hello_frame

    frame = hello_frame(
        node_id=FIXTURE_NODE_ID,
        node_name=NODE_NAME,
        daemon_version=DAEMON_VERSION,
        uptime_s=UPTIME,
        capability=mint_capability(600),
        proof=identity.sign(b"x"),
    )
    assert "null" not in dump_frame(frame)
    assert json.loads(dump_frame(frame))["protocol"] == PROTOCOL_VERSION
