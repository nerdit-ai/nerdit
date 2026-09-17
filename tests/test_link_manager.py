"""The connect / renew / reconnect state machine (P27 WP-C1, items 4-6).

WP-C1 "tunnel client" item 4 (renewal is the daemon's job (D-R8) … close-code
classification) and item 5 (durable ``link.connected``/``link.disconnected``
events), driven per item 6 by the in-process FakeRelay.

**No test here sleeps.** Every timer the manager owns — the renewal lead, the
backoff ladder, the fixed displacement delay, the heartbeat cadence — is an
injected ``sleep`` that *parks* until the test releases it, and releasing it
advances an injected clock. That inverts the usual bargain: instead of
approximating "about a second passed", each test asserts the exact number of
seconds the manager asked to wait, and gets to choose when that wait ends. A
reconnect policy whose timings cannot be asserted is one that silently rots
(and one that, tested with real sleeps, makes CI slow *and* flaky).

The relay is a real loopback socket, so the frames crossing it are the frames a
real relay would see: two connects really do mint two capabilities, and "the
second hello carries a different token" is read off the wire, not off a mock.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from pydantic import ValidationError

from nerdit.config.settings import LinkSettings
from nerdit.core.link.client import MuxContext, dial_relay
from nerdit.core.link.identity import NodeIdentity
from nerdit.core.link.manager import (
    AUTH_FAILURE_LIMIT,
    BACKOFF_FACTOR,
    BACKOFF_INITIAL_S,
    BACKOFF_JITTER,
    BACKOFF_MAX_S,
    BACKOFF_STABLE_S,
    DISCONNECT_REASONS,
    DISPLACED_RETRY_DELAY_S,
    ENTITLEMENT_TTL_S,
    GITHUB_INSTALLATION_MIRROR_MAX,
    MIN_RENEW_LEAD_S,
    LinkManager,
    LinkStatus,
)
from nerdit.core.link.mux import StreamMux
from tests.link_fake_relay import (
    CLOSE_PROTOCOL_UNSUPPORTED,
    CLOSE_REPLACED,
    CLOSE_REVOKED,
    CLOSE_UNAUTHORIZED,
    ERROR_AUTH,
    ERROR_PROTOCOL,
    ERROR_REVOKED,
    FIXTURE_NODE_ID,
    FakeRelay,
    Scenario,
    open_stream_frame,
)

#: TTL 600 / margin 120 ⇒ the renewal timer asks for exactly 480 s, a value no
#: other timer in the machine can collide with (heartbeat 5, backoff 1/2/4/…,
#: displacement 60). Timer identification by delay is only sound because of that.
TTL_S = 600
MARGIN_S = 120
RENEW_LEAD_S = float(TTL_S - MARGIN_S)
HEARTBEAT_S = 5.0


# ---------------------------------------------------------------------------
# injected time
# ---------------------------------------------------------------------------


class FakeClock:
    """A UTC clock the test moves by hand, plus a monotonic in lockstep.

    The two move together deliberately. The manager measures wall-clock things
    (capability expiry, ``retry_at``) against ``clock`` and elapsed things
    (uptime, the :data:`BACKOFF_STABLE_S` session-stability gate) against
    ``monotonic``; a harness where only one of them advanced would make
    "the session lasted 31 s" unexpressible.
    """

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        self.elapsed = 100.0

    def __call__(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.elapsed

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)
        self.elapsed += seconds


@dataclass(slots=True)
class _Pending:
    delay: float
    event: asyncio.Event = field(default_factory=asyncio.Event)


class Clockwork:
    """An injected ``sleep`` that parks every waiter until the test says go.

    ``requested`` is the full ledger of delays asked for, in order — which is
    what the backoff-ladder assertions read. Releasing a wait advances the
    :class:`FakeClock` by exactly that delay, so ``clock()`` inside the manager
    stays consistent with the waits it performed.
    """

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.requested: list[float] = []
        self.pending: list[_Pending] = []

    async def sleep(self, delay: float) -> None:
        entry = _Pending(delay)
        self.requested.append(delay)
        self.pending.append(entry)
        try:
            await entry.event.wait()
        finally:
            if entry in self.pending:
                self.pending.remove(entry)

    async def wait_for(self, delay: float, *, timeout: float = 5.0) -> _Pending:
        """Block until somebody is parked on exactly ``delay`` seconds."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            for entry in self.pending:
                if entry.delay == pytest.approx(delay):
                    return entry
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    f"no sleep({delay}) is pending; pending={[e.delay for e in self.pending]} "
                    f"requested={self.requested}"
                )
            await asyncio.sleep(0.001)

    async def fire(self, delay: float, *, timeout: float = 5.0) -> None:
        """Wait for a ``sleep(delay)``, advance the clock by it, and release it."""
        entry = await self.wait_for(delay, timeout=timeout)
        self._clock.advance(delay)
        entry.event.set()
        await asyncio.sleep(0)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    type: str
    kind: str | None
    reason: str | None
    data: dict[str, object] | None


class FakeRecorder:
    """Collects durable rows without a database."""

    def __init__(self) -> None:
        self.events: list[RecordedEvent] = []

    async def record(
        self,
        type: str,  # noqa: A002 - mirrors the locked D-P24-3 field name
        *,
        kind: str | None = None,
        service_name: str | None = None,
        reason: str | None = None,
        build_version: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        assert service_name is None, "link events are not service-scoped"
        assert build_version is None
        self.events.append(RecordedEvent(type=type, kind=kind, reason=reason, data=data))

    def types(self) -> list[str]:
        return [event.type for event in self.events]


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@pytest.fixture
def identity() -> NodeIdentity:
    raw = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    return NodeIdentity(base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii"))


class _RecordingMux:
    """A mux stand-in: this file tests the manager, not stream serving."""

    instances: list[_RecordingMux] = []

    def __init__(self, ctx: MuxContext) -> None:
        self.ctx = ctx
        self.closed = 0
        self.streams = 0
        _RecordingMux.instances.append(self)

    async def handle(self, frame: object) -> None:  # pragma: no cover - unused here
        return None

    def active_streams(self) -> int:
        return self.streams

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def _reset_mux_instances() -> None:
    _RecordingMux.instances = []


@dataclass(slots=True)
class Harness:
    manager: LinkManager
    relay: FakeRelay
    clock: FakeClock
    clockwork: Clockwork
    events: FakeRecorder

    def status(self) -> LinkStatus:
        return self.manager.status()

    async def wait_state(self, state: str, *, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while self.manager.status().state != state:
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError(
                    f"state stayed {self.manager.status().state!r}, wanted {state!r}"
                )
            await asyncio.sleep(0.001)


def _settings(**kw: object) -> LinkSettings:
    return LinkSettings(
        enabled=True,
        relay_url="wss://relay.example.test/link",
        capability_ttl_s=TTL_S,
        renew_margin_s=MARGIN_S,
        node_id=FIXTURE_NODE_ID,
        slug="dev-node",
        **kw,  # type: ignore[arg-type]
    )


def _build(  # noqa: PLR0913 - the injected clocks are the point of this harness
    identity: NodeIdentity,
    relay: FakeRelay,
    clock: FakeClock,
    clockwork: Clockwork,
    *,
    rng: float = 0.5,
    link: LinkSettings | None = None,
    mux_factory: Callable[[MuxContext], Any] | None = None,
    wrap_socket: Callable[[Any], Any] | None = None,
) -> Harness:
    """Wire a manager onto the loopback relay with every clock injected.

    ``rng=0.5`` makes the jitter factor exactly ``1.0``, so the backoff ladder
    is 1, 2, 4, 8… on the nose and can be asserted as equality rather than as a
    range. The jitter *bounds* get their own test.

    ``wrap_socket`` decorates the dialled :class:`LinkSocket`. It exists for the
    one behaviour that is otherwise untestable in-process: the ordering race
    between ``stop()``'s close and its ``task.cancel()``, which on loopback the
    cancel always wins but over a real relay the close-handshake RTT does not.
    """
    events = FakeRecorder()

    async def connect() -> Any:
        socket = await dial_relay(relay.url)
        return socket if wrap_socket is None else wrap_socket(socket)

    manager = LinkManager(
        link or _settings(),
        identity,
        node_id=FIXTURE_NODE_ID,
        node_name="dev-node",
        daemon_version="0.1.0",
        loopback_base_url="http://127.0.0.1:9321",
        events=events,  # type: ignore[arg-type]
        connector=connect,
        mux_factory=mux_factory or _RecordingMux,
        clock=clock,
        sleep=clockwork.sleep,
        monotonic=clock.monotonic,
        rng=lambda: rng,
    )
    return Harness(manager=manager, relay=relay, clock=clock, clockwork=clockwork, events=events)


@asynccontextmanager
async def link_harness(
    identity: NodeIdentity,
    *,
    scenarios: Sequence[Scenario] = (),
    rng: float = 0.5,
    link: LinkSettings | None = None,
    mux_factory: Callable[[MuxContext], Any] | None = None,
    wrap_socket: Callable[[Any], Any] | None = None,
) -> AsyncIterator[Harness]:
    """A running relay + a manager wired to it, sharing ONE injected clock.

    The shared clock is load-bearing, not tidiness: the relay enforces the §5
    step 5 capability window (``now < expires_at <= now + 15 min``) exactly as
    the real one does, so a relay reading wall-clock time while the manager
    mints against a fake one would refuse every hello the moment a test moved
    time — which is what every renewal test does.
    """
    clock = FakeClock()
    clockwork = Clockwork(clock)
    async with FakeRelay(verifier=identity.verifier, clock=clock) as relay:
        relay.script(*scenarios)
        harness = _build(
            identity,
            relay,
            clock,
            clockwork,
            rng=rng,
            link=link,
            mux_factory=mux_factory,
            wrap_socket=wrap_socket,
        )
        try:
            yield harness
        finally:
            await harness.manager.stop()


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_policy_constants_match_the_recorded_decisions() -> None:
    assert (BACKOFF_INITIAL_S, BACKOFF_FACTOR, BACKOFF_MAX_S) == (1.0, 2.0, 60.0)
    assert BACKOFF_JITTER == 0.2
    # Spec §7 asks for "a bounded number of times"; five spans a relay-restart
    # drain window without hiding a genuinely dead credential for long.
    assert AUTH_FAILURE_LIMIT == 5
    # §6 4409: fixed, NOT exponential — an architect judgment call (risk note 2).
    assert DISPLACED_RETRY_DELAY_S == 60.0
    assert MIN_RENEW_LEAD_S == 5.0


def test_the_disconnect_reason_vocabulary_is_closed() -> None:
    """D-P24-3: durable payloads are machine tokens, never free text."""
    assert {
        "error",
        "displaced",
        "revoked",
        "auth_failed",
        "protocol_unsupported",
        "entitlement_required",
        "shutdown",
    } == DISCONNECT_REASONS


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


async def test_connect_emits_one_durable_event_and_installs_the_capability(
    identity: NodeIdentity,
) -> None:
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        session = await relay.wait_hello()
        await harness.wait_state("connected")

        status = harness.status()
        assert status.state == "connected"
        assert status.node_id == FIXTURE_NODE_ID
        # host[:port] only — never the URL, never the path.
        assert status.relay_host == "relay.example.test"
        assert "/" not in status.relay_host
        assert status.connected_at == harness.clock.now
        assert status.capability_expires_at == harness.clock.now + timedelta(seconds=TTL_S)
        assert (status.attempt, status.auth_failures) == (0, 0)
        assert status.terminal_reason is None and status.retry_at is None
        assert status.active_streams == 0

        assert harness.events.events == [
            RecordedEvent(
                type="link.connected",
                kind="link",
                reason=None,
                data={"node_id": FIXTURE_NODE_ID},
            )
        ]
        # Nothing in the payload is the capability, and nothing is free text.
        assert session.capability_token not in str(harness.events.events)
        # And nothing in it is the relay endpoint: GET /events is readable by
        # any authenticated principal, while /capabilities gates ``relay_host``
        # behind admin (the proxy.admin_addr precedent). The weaker gate must
        # not defeat the stronger one for a readonly token.
        assert "relay_host" not in (harness.events.events[0].data or {})
        assert status.relay_host not in str(harness.events.events)

        await harness.manager.stop()


async def test_the_mux_is_built_from_the_acked_limits(identity: NodeIdentity) -> None:
    """§8: the daemon obeys what the relay advertised, never a hardcoded table."""
    async with link_harness(identity) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")

        assert len(_RecordingMux.instances) == 1
        ctx = _RecordingMux.instances[0].ctx
        assert ctx.node_id == FIXTURE_NODE_ID
        assert ctx.limits.max_concurrent_streams_per_node == 8
        assert ctx.limits.max_body_bytes == 10 * 1024 * 1024
        assert ctx.limits.max_response_body_bytes == 50 * 1024 * 1024
        assert ctx.loopback_base_url == "http://127.0.0.1:9321"
        # The mux validates through the manager, so expiry and the submitter
        # ceiling are enforced in one place.
        assert ctx.validate_capability == harness.manager.validate_capability

        await harness.manager.stop()


async def test_start_is_idempotent(identity: NodeIdentity) -> None:
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        await harness.manager.start()
        await harness.wait_state("connected")
        await relay.wait_sessions(1)
        await asyncio.sleep(0.02)
        assert len(relay.sessions) == 1, "a second start() must not open a second tunnel"
        await harness.manager.stop()


# ---------------------------------------------------------------------------
# renewal (D-R8) — the checklist's headline behaviour
# ---------------------------------------------------------------------------


async def test_renewal_reconnects_with_freshly_minted_material_and_no_event_churn(
    identity: NodeIdentity,
) -> None:
    """D-R8: renew at ``expires_at − renew_margin_s`` with *fresh* material.

    Three things are asserted together because they are one behaviour:
    the timer asks for exactly the lead time; the second hello carries a
    **different** capability token (a re-proved connection, not a resumed one);
    and the durable feed stays silent — ``online`` spans the renewal, so a
    healthy daemon does not write two rows every ~13 minutes forever.
    """
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        first = await relay.wait_hello(0)
        await harness.wait_state("connected")

        # The renewal timer asked for exactly expires_at − margin.
        await harness.clockwork.wait_for(RENEW_LEAD_S)
        assert RENEW_LEAD_S == 480.0

        await harness.clockwork.fire(RENEW_LEAD_S)
        second = await relay.wait_hello(1)
        await harness.wait_state("connected")

        assert first.capability_token != second.capability_token
        assert second.hello is not None and second.hello.proof != first.hello.proof
        # A fresh nonce would also change the proof; pin the token explicitly.
        assert second.proof_valid is True

        # No backoff was involved: the reconnect is immediate.
        assert BACKOFF_INITIAL_S not in harness.clockwork.requested

        # And the feed still shows exactly one connect, no disconnect.
        assert harness.events.types() == ["link.connected"]
        status = harness.status()
        assert status.state == "connected"
        assert status.capability_expires_at == harness.clock.now + timedelta(seconds=TTL_S)

        # The old connection's mux was torn down before the new one was built.
        assert len(_RecordingMux.instances) == 2
        assert _RecordingMux.instances[0].closed == 1

        await harness.manager.stop()


async def test_renewal_closes_the_old_socket_cleanly(identity: NodeIdentity) -> None:
    """Close-then-reconnect, not dial-first (risk note 3).

    Dialling a second socket would self-displace and race our own 4409, so the
    old socket is closed with 1000 first. The cost — in-flight streams die once
    per capability lifetime — is accepted for v1 and is what this pins.
    """
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        await relay.wait_hello(0)
        await harness.wait_state("connected")
        await harness.clockwork.fire(RENEW_LEAD_S)
        await relay.wait_hello(1)

        # The relay saw the first socket end before the second one opened.
        await FakeRelay._wait(lambda: relay.sessions[0].finished.is_set(), 2.0, "a clean close")
        assert len(relay.sessions) == 2
        await harness.manager.stop()


async def test_a_dial_failure_at_the_renewal_boundary_records_the_outage(
    identity: NodeIdentity,
) -> None:
    """The renewal boundary is the one moment the link is up but socketless.

    An outage that begins exactly there used to be invisible twice over: the
    manager stayed ``online`` through arbitrarily many refused dials (so the
    durable feed never showed the link going down), and the capability minted
    for the *previous* socket stayed installed (so ``validate_capability`` kept
    honouring a bearer whose connection no longer existed).
    """
    scenarios = [Scenario(), *[Scenario(refuse_upgrade=403)] * 3]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        old_token = (await harness.relay.wait_hello(0)).capability_token
        assert harness.manager.validate_capability(old_token, "submitter") is True

        # Renewal closes the socket and immediately re-dials; the re-dial is
        # refused pre-accept (spec §5 step 1) and the relay is now down.
        await harness.clockwork.fire(RENEW_LEAD_S)
        await harness.wait_state("backoff")

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.events.events[-1].reason == "error"
        assert harness.manager.validate_capability(old_token, "submitter") is False
        assert harness.status().capability_expires_at is None

        # …and the edge is recorded exactly once, however long the outage runs.
        for delay in (BACKOFF_INITIAL_S, BACKOFF_INITIAL_S * BACKOFF_FACTOR):
            await harness.clockwork.fire(delay)
            await harness.wait_state("backoff")
        assert harness.events.types().count("link.disconnected") == 1
        await harness.manager.stop()


async def test_the_renewal_gap_leaves_nothing_to_validate_against(
    identity: NodeIdentity,
) -> None:
    """Even on the happy renewal, the old capability dies with its socket.

    ADR-W1/D-R2: the capability is proven per-connection. Between the renewal
    close and the next ``hello_ack`` there is no live proof, so nothing may
    validate — and the fresh one must be installed by the time the link reports
    ``connected`` again.
    """
    release = asyncio.Event()

    async def hold_the_second_socket(session: object) -> None:
        await release.wait()

    scenarios = [Scenario(), Scenario(after_ack=hold_the_second_socket)]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        old_token = (await harness.relay.wait_hello(0)).capability_token

        await harness.clockwork.fire(RENEW_LEAD_S)
        second = await harness.relay.wait_hello(1)

        # The relay has the new hello but has not acked it yet.
        assert harness.manager.validate_capability(old_token, "submitter") is False
        release.set()

        await harness.wait_state("connected")
        assert harness.manager.validate_capability(second.capability_token, "submitter") is True
        assert harness.manager.validate_capability(old_token, "submitter") is False
        # Still no event churn: ``online`` spanned the whole renewal.
        assert harness.events.types() == ["link.connected"]
        await harness.manager.stop()


async def test_the_renewal_lead_has_a_floor(identity: NodeIdentity) -> None:
    """A margin at/above the TTL must degrade to a slow reconnect, not a spin.

    ``LinkSettings`` refuses that combination outright — asserted here first, so
    this test cannot quietly stop covering the config layer — which is why the
    settings object has to be built with ``model_construct``. What remains is
    defense in depth for direct callers: without the floor, ``renew_delay`` goes
    negative and the manager reconnect-loops as fast as the relay will accept.
    """
    with pytest.raises(ValidationError, match="renew_margin_s"):
        LinkSettings(
            enabled=True,
            relay_url="wss://relay.example.test/link",
            capability_ttl_s=60,
            renew_margin_s=600,
        )
    link = LinkSettings.model_construct(
        enabled=True,
        relay_url="wss://relay.example.test/link",
        key_file=None,
        capability_ttl_s=60,
        renew_margin_s=600,
        node_id=FIXTURE_NODE_ID,
        slug=None,
    )
    async with link_harness(identity, link=link) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        await harness.clockwork.wait_for(MIN_RENEW_LEAD_S)
        assert MIN_RENEW_LEAD_S in harness.clockwork.requested


# ---------------------------------------------------------------------------
# the close-code matrix
# ---------------------------------------------------------------------------


async def test_bounded_auth_retries_then_terminal_auth_failed(identity: NodeIdentity) -> None:
    """Spec §7, the normative guidance, verbatim.

    Five refused handshakes, each retried with **freshly minted** material and
    a real backoff between them; the sixth never happens because the link goes
    terminal. Nothing is emitted durably, because the link was never online —
    the doctor check and ``LinkStatus`` carry a never-connected failure, not the
    event feed.
    """
    refuse = Scenario(refuse_hello=(ERROR_AUTH, "node authentication failed", CLOSE_UNAUTHORIZED))
    async with link_harness(identity, scenarios=[*[refuse] * AUTH_FAILURE_LIMIT]) as harness:
        relay = harness.relay
        await harness.manager.start()

        for attempt in range(1, AUTH_FAILURE_LIMIT):
            # Backoff between attempts: 1, 2, 4, 8 seconds. Waiting for the
            # parked sleep (rather than for the hello) is also the synchronizing
            # edge — the manager has fully classified the close by then.
            delay = BACKOFF_INITIAL_S * BACKOFF_FACTOR ** (attempt - 1)
            await harness.clockwork.wait_for(delay)
            assert harness.status().auth_failures == attempt
            assert harness.status().state == "backoff"
            await harness.clockwork.fire(delay)

        await relay.wait_hello(AUTH_FAILURE_LIMIT - 1)
        await harness.wait_state("terminal")

        status = harness.status()
        assert status.terminal_reason == "auth_failed"
        assert status.auth_failures == AUTH_FAILURE_LIMIT
        assert status.last_close_code == CLOSE_UNAUTHORIZED
        assert status.last_error_code == ERROR_AUTH
        assert status.retry_at is None

        # Every attempt minted its own capability — never a replayed credential.
        tokens = [session.capability_token for session in relay.sessions]
        assert len(tokens) == AUTH_FAILURE_LIMIT
        assert len(set(tokens)) == AUTH_FAILURE_LIMIT
        # …and its own proof, over the relay's own fresh nonce.
        proofs = [session.hello.proof for session in relay.sessions if session.hello]
        assert len(set(proofs)) == AUTH_FAILURE_LIMIT

        assert harness.events.events == []

        # Terminal means terminal: no sixth dial, ever.
        await asyncio.sleep(0.02)
        assert len(relay.sessions) == AUTH_FAILURE_LIMIT
        await harness.manager.stop()


async def test_a_first_4401_does_not_alarm(identity: NodeIdentity) -> None:
    """§7 explicitly: do not alarm on the first 4401 — it may be a drain."""
    async with link_harness(
        identity,
        scenarios=[
            Scenario(refuse_hello=(ERROR_AUTH, "no", CLOSE_UNAUTHORIZED)),
            Scenario(),
        ],
    ) as harness:
        relay = harness.relay
        await harness.manager.start()
        await relay.wait_hello(0)
        assert harness.status().terminal_reason is None
        await harness.clockwork.fire(BACKOFF_INITIAL_S)
        await harness.wait_state("connected")

        # The retry succeeded, so the counter resets and the feed shows one
        # connect — a transient drain leaves no scar.
        assert harness.status().auth_failures == 0
        assert harness.events.types() == ["link.connected"]
        await harness.manager.stop()


@pytest.mark.parametrize(
    ("close_code", "error_code", "reason"),
    [
        (CLOSE_PROTOCOL_UNSUPPORTED, ERROR_PROTOCOL, "protocol_unsupported"),
        (CLOSE_REVOKED, ERROR_REVOKED, "revoked"),
        (CLOSE_UNAUTHORIZED, "entitlement_required", "entitlement_required"),
    ],
)
async def test_terminal_closes_stop_the_loop_and_name_their_reason(
    identity: NodeIdentity, close_code: int, error_code: str, reason: str
) -> None:
    """§6: 4400 / 4403 / entitlement are terminal until reconfigured.

    Each is staged **after** a successful connect so the online→offline edge is
    real and the durable row is emitted with the right machine token.
    """
    async with link_harness(
        identity,
        scenarios=[
            Scenario(
                inject=[{"type": "error", "code": error_code, "message": "refused"}],
                close_with=(close_code, "refused"),
            )
        ],
    ) as harness:
        relay = harness.relay
        await harness.manager.start()
        await harness.wait_state("terminal")

        status = harness.status()
        assert status.terminal_reason == reason
        assert status.last_close_code == close_code
        assert status.last_error_code == error_code
        assert status.capability_expires_at is None
        assert status.connected_at is None

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        disconnected = harness.events.events[-1]
        assert disconnected.reason == reason
        assert disconnected.reason in DISCONNECT_REASONS
        assert disconnected.kind == "link"
        assert disconnected.data == {"close_code": close_code, "error_code": error_code}

        # No further dials, and the mux was torn down.
        await asyncio.sleep(0.02)
        assert len(relay.sessions) == 1
        assert _RecordingMux.instances[0].closed == 1
        await harness.manager.stop()


async def test_a_plain_4401_without_an_error_frame_still_retries_bounded(
    identity: NodeIdentity,
) -> None:
    """(P27 WP-C4) The entitlement mapping must not swallow the ordinary 4401.

    4401 is overloaded: with an ``entitlement_required`` error frame it is
    terminal (the test above), with ``node_authentication_failed`` — or with no
    error frame at all — it is the bounded retry ladder, because a relay that
    starts draining mid-handshake closes exactly like this. Only the *pair*
    decides. That distinction is pinned at the classification layer
    (``test_classify_close_covers_every_spec_row``); this pins it where it is
    actually consumed, on a manager driven by a real relay. The full five-strike
    bound stays with :func:`test_bounded_auth_retries_then_terminal_auth_failed`,
    which this cites rather than duplicates.
    """
    async with link_harness(
        identity,
        scenarios=[Scenario(close_with=(CLOSE_UNAUTHORIZED, "unauthorized")), Scenario()],
    ) as harness:
        relay = harness.relay
        await harness.manager.start()
        await harness.wait_state("backoff")

        status = harness.status()
        assert status.state == "backoff"
        assert status.terminal_reason is None
        assert status.auth_failures == 1
        assert status.last_close_code == CLOSE_UNAUTHORIZED
        # No error frame arrived, so there is no error code to blame it on —
        # and the close alone is never enough to call it terminal.
        assert status.last_error_code is None
        assert harness.events.events[-1].reason == "error"

        # The retry really happens, and it re-proves rather than replaying.
        await harness.clockwork.fire(BACKOFF_INITIAL_S)
        await harness.wait_state("connected")
        assert len(relay.sessions) == 2
        assert relay.sessions[0].capability_token != relay.sessions[1].capability_token
        assert harness.status().auth_failures == 0
        await harness.manager.stop()


async def test_a_boot_time_entitlement_refusal_is_terminal_but_not_durable(
    identity: NodeIdentity,
) -> None:
    """(P27 WP-C4) A never-online entitlement refusal is terminal, and silent.

    The test above stages the refusal *after* a successful connect, so the
    online→offline edge is real and a durable ``link.disconnected`` row is
    written. A subscription that was already lapsed when the daemon booted takes
    the other path: ``_go_terminal`` → ``_go_offline`` early-returns on ``not
    self._online``, so there is no edge and no row — only the terminal state.
    That is the WP-C1 decision, not an oversight, and it is the same shape
    :func:`test_bounded_auth_retries_then_terminal_auth_failed` pins for
    ``auth_failed``: a never-connected failure is carried by the doctor check and
    ``LinkStatus``, never by the durable feed. Pinned here so the WP-C4 hint work
    cannot quietly grow a boot-time event.
    """
    async with link_harness(
        identity,
        scenarios=[
            Scenario(
                refuse_hello=(
                    "entitlement_required",
                    "no active subscription",
                    CLOSE_UNAUTHORIZED,
                )
            )
        ],
    ) as harness:
        await harness.manager.start()
        await harness.wait_state("terminal")

        status = harness.status()
        assert status.state == "terminal"
        assert status.terminal_reason == "entitlement_required"
        assert status.last_close_code == CLOSE_UNAUTHORIZED
        assert harness.events.events == []

        # Terminal means terminal: the refusal is not retried.
        await asyncio.sleep(0.02)
        assert len(harness.relay.sessions) == 1

        # …and shutting down a link that was never online writes nothing either.
        await harness.manager.stop()
        assert harness.events.events == []


async def test_displacement_waits_a_fixed_minute_and_retries(identity: NodeIdentity) -> None:
    """§6 4409: routine, and the displaced socket must not reconnect-fight.

    Fixed 60 s, deliberately not exponential (risk note 2): a lingering stale
    socket heals on the next try, while a genuine second daemon shows up as slow
    *visible* flapping in the durable feed rather than an invisible retreat.
    """
    scenarios = [Scenario(close_with=(CLOSE_REPLACED, "replaced")), Scenario()]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("displaced")

        status = harness.status()
        assert status.last_close_code == CLOSE_REPLACED
        assert status.retry_at == harness.clock.now + timedelta(seconds=DISPLACED_RETRY_DELAY_S)
        # Not the backoff ladder: the attempt counter is untouched.
        assert status.attempt == 0
        assert harness.clockwork.requested.count(DISPLACED_RETRY_DELAY_S) == 1
        assert BACKOFF_INITIAL_S not in harness.clockwork.requested

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.events.events[-1].reason == "displaced"
        assert harness.events.events[-1].data == {
            "close_code": CLOSE_REPLACED,
            "error_code": None,
        }

        await harness.clockwork.fire(DISPLACED_RETRY_DELAY_S)
        await harness.wait_state("connected")
        assert harness.events.types() == [
            "link.connected",
            "link.disconnected",
            "link.connected",
        ]
        await harness.manager.stop()


@pytest.mark.parametrize("close_code", [1000, 1001, 1013])
async def test_ordinary_closes_back_off_and_reconnect(
    identity: NodeIdentity, close_code: int
) -> None:
    """§6: 1000/1001/1013 all mean "try again, politely"."""
    release = asyncio.Event()

    async def hold_until_released(session: object) -> None:
        await release.wait()

    scenarios = [
        Scenario(close_with=(close_code, "bye")),
        Scenario(after_ack=hold_until_released, close_with=(close_code, "bye")),
        Scenario(),
    ]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("backoff")

        status = harness.status()
        assert status.attempt == 1
        assert status.last_close_code == close_code
        assert status.retry_at == harness.clock.now + timedelta(seconds=BACKOFF_INITIAL_S)
        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.events.events[-1].reason == "error"
        assert harness.events.events[-1].data == {"close_code": close_code, "error_code": None}

        await harness.clockwork.fire(BACKOFF_INITIAL_S)
        await harness.wait_state("connected")
        # Reaching ``hello_ack`` does NOT by itself clear the ladder — see the
        # flap test below for why. The credential doubt is cleared (§7), the
        # connection doubt is not.
        assert harness.status().attempt == 1
        assert harness.status().auth_failures == 0

        # Let this session prove itself, then drop it: the ladder restarts at
        # the bottom rung because the last failure really was a blip.
        harness.clock.advance(BACKOFF_STABLE_S)
        release.set()
        await harness.wait_state("backoff")
        assert harness.status().attempt == 1
        assert harness.status().retry_at == harness.clock.now + timedelta(seconds=BACKOFF_INITIAL_S)
        await harness.manager.stop()


async def test_a_connect_then_drop_flap_climbs_the_ladder(identity: NodeIdentity) -> None:
    """A relay that accepts the handshake and immediately drops must not be
    redialled at ~1 Hz forever.

    This is the shape the ladder used to be blind to: every cycle reached
    ``hello_ack``, so resetting ``_attempt`` there put the next delay back at
    1 s no matter how many times it had happened. Five consecutive
    accept-then-drop cycles must produce five *increasing* delays.
    """
    cycles = 5
    scenarios = [Scenario(close_with=(1001, "drop")) for _ in range(cycles + 1)]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()

        delays: list[float] = []
        for index in range(cycles):
            expected = BACKOFF_INITIAL_S * BACKOFF_FACTOR**index
            await harness.clockwork.wait_for(expected)
            delays.append(expected)
            assert harness.status().attempt == index + 1
            await harness.clockwork.fire(expected)

        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]
        assert delays == sorted(delays) and len(set(delays)) == cycles
        await harness.manager.stop()


async def test_the_backoff_ladder_doubles_to_its_ceiling(identity: NodeIdentity) -> None:
    """1, 2, 4, 8, 16, 32, 60, 60 — capped, never unbounded."""
    async with link_harness(identity, scenarios=[*[Scenario(refuse_upgrade=403)] * 9]) as harness:
        relay = harness.relay
        await harness.manager.start()

        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
        for index, delay in enumerate(expected):
            await relay.wait_sessions(index + 1)
            await harness.clockwork.fire(delay)

        assert harness.clockwork.requested[: len(expected)] == expected
        assert max(harness.clockwork.requested) <= BACKOFF_MAX_S
        await harness.manager.stop()


@pytest.mark.parametrize(("rng", "factor"), [(0.0, 0.8), (1.0, 1.2), (0.5, 1.0)])
async def test_backoff_jitter_stays_within_twenty_percent(
    identity: NodeIdentity, rng: float, factor: float
) -> None:
    """Full ±20 % jitter, so a relay restart does not bring every node back in
    lockstep. The bounds are asserted at the extremes of the rng."""
    scenarios = [Scenario(refuse_upgrade=403), Scenario(refuse_upgrade=403)]
    async with link_harness(identity, scenarios=scenarios, rng=rng) as harness:
        await harness.manager.start()
        await harness.relay.wait_sessions(1)
        entry = await harness.clockwork.wait_for(BACKOFF_INITIAL_S * factor)
        assert entry.delay == pytest.approx(BACKOFF_INITIAL_S * factor)
        assert 0.8 <= entry.delay / BACKOFF_INITIAL_S <= 1.2


async def test_a_pre_accept_403_backs_off_without_a_durable_event(
    identity: NodeIdentity,
) -> None:
    """§5 step 1: the capacity/drain refusal that is an HTTP status.

    It is retryable, and — because the link was never online — it writes nothing
    durable. Pre-first-connect failures belong to ``/doctor``, not to the feed.
    """
    scenarios = [Scenario(refuse_upgrade=403), Scenario(refuse_upgrade=403), Scenario()]
    async with link_harness(identity, scenarios=scenarios) as harness:
        relay = harness.relay
        await harness.manager.start()

        await relay.wait_sessions(1)
        await harness.wait_state("backoff")
        assert harness.events.events == []
        await harness.clockwork.fire(BACKOFF_INITIAL_S)

        await relay.wait_sessions(2)
        await harness.clockwork.fire(BACKOFF_INITIAL_S * BACKOFF_FACTOR)

        await harness.wait_state("connected")
        assert harness.events.types() == ["link.connected"]
        assert all(session.hello is None for session in relay.sessions[:2])
        await harness.manager.stop()


async def test_a_dead_relay_is_an_ordinary_backoff(identity: NodeIdentity) -> None:
    """A refused TCP connect must not end the loop — that would need a restart."""
    clock = FakeClock()
    clockwork = Clockwork(clock)
    events = FakeRecorder()
    attempts = 0

    async def broken_connector() -> object:
        nonlocal attempts
        attempts += 1
        raise OSError("connection refused")

    manager = LinkManager(
        _settings(),
        identity,
        node_id=FIXTURE_NODE_ID,
        node_name="dev-node",
        daemon_version="0.1.0",
        loopback_base_url="http://127.0.0.1:9321",
        events=events,  # type: ignore[arg-type]
        connector=broken_connector,  # type: ignore[arg-type]
        mux_factory=_RecordingMux,
        clock=clock,
        sleep=clockwork.sleep,
        monotonic=lambda: 0.0,
        rng=lambda: 0.5,
    )
    await manager.start()
    await clockwork.fire(BACKOFF_INITIAL_S)
    await clockwork.fire(BACKOFF_INITIAL_S * BACKOFF_FACTOR)
    assert attempts >= 3
    assert manager.status().state == "backoff"
    assert events.events == []
    await manager.stop()


# ---------------------------------------------------------------------------
# validate_capability — the middleware's authorization primitive
# ---------------------------------------------------------------------------


async def test_validate_capability_accepts_only_the_live_proven_token(
    identity: NodeIdentity,
) -> None:
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        session = await relay.wait_hello()
        await harness.wait_state("connected")
        token = session.capability_token
        validate = harness.manager.validate_capability

        assert validate(token, "submitter") is True
        # Wrong token, near-miss token, empty token.
        assert validate("nope", "submitter") is False
        assert validate(token[:-1], "submitter") is False
        assert validate(token + "x", "submitter") is False
        assert validate("", "submitter") is False
        # The submitter ceiling is permanent (D-R2 as amended by ADR-W1):
        # remote admin is out of the protocol, not merely unused.
        for role in ("admin", "readonly", "Submitter", ""):
            assert validate(token, role) is False

        await harness.manager.stop()


async def test_validate_capability_rechecks_expiry_under_a_live_socket(
    identity: NodeIdentity,
) -> None:
    """Spec §9: expiry is *reactive* — the relay never refreshes for us.

    A capability can therefore lapse while the socket is still up, and a check
    that trusted "the link is connected" would keep honouring a dead credential.
    """
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        session = await relay.wait_hello()
        await harness.wait_state("connected")
        token = session.capability_token

        harness.clock.advance(TTL_S - 1)
        assert harness.manager.validate_capability(token, "submitter") is True
        harness.clock.advance(1)  # exactly at expires_at
        assert harness.manager.validate_capability(token, "submitter") is False
        assert harness.status().state == "connected"  # still connected, just stale

        await harness.manager.stop()


async def test_validate_capability_is_false_while_offline(identity: NodeIdentity) -> None:
    async with link_harness(identity, scenarios=[Scenario(close_with=(1001, "bye"))]) as harness:
        # Before start: nothing is proven.
        assert harness.manager.validate_capability("anything", "submitter") is False

        await harness.manager.start()
        session = await harness.relay.wait_hello()
        token = session.capability_token
        await harness.wait_state("backoff")
        # Between sockets there is no capability at all — a request arriving on
        # a stale bearer must not authenticate.
        assert harness.manager.validate_capability(token, "submitter") is False
        assert harness.status().capability_expires_at is None

        await harness.manager.stop()


async def test_validate_capability_survives_a_non_ascii_token(identity: NodeIdentity) -> None:
    """``secrets.compare_digest`` raises on mixed str/bytes-ish input.

    The middleware feeds it whatever a caller put in the header, so a bearer of
    arbitrary text must answer ``False`` rather than 500 the request.
    """
    async with link_harness(identity) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        assert harness.manager.validate_capability("héllo-wörld", "submitter") is False
        await harness.manager.stop()


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


async def test_stop_closes_the_tunnel_and_records_the_shutdown_edge(
    identity: NodeIdentity,
) -> None:
    async with link_harness(identity) as harness:
        relay = harness.relay
        await harness.manager.start()
        session = await relay.wait_hello()
        await harness.wait_state("connected")
        token = session.capability_token

        await harness.manager.stop()

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        shutdown = harness.events.events[-1]
        assert shutdown.reason == "shutdown"
        assert shutdown.data == {"close_code": None, "error_code": None}
        # The capability dies with the connection (ADR-W1): a request that
        # arrives during the drain must not authenticate on it.
        assert harness.manager.validate_capability(token, "submitter") is False
        assert harness.status().capability_expires_at is None
        assert harness.status().connected_at is None
        assert _RecordingMux.instances[0].closed >= 1
        # The relay saw the socket go.
        await FakeRelay._wait(lambda: session.finished.is_set(), 2.0, "the socket to close")


class _SlowClosingSocket:
    """A socket whose ``close()`` yields until the manager task has finished.

    This is the whole point of the ``wrap_socket`` seam. ``stop()`` closes the
    connection and *then* cancels the task; on loopback the cancel always wins,
    but over a real relay the close-handshake RTT sits in that window and the
    manager task observes its own local 1000 close first. Holding the loop
    inside ``close()`` makes that ordering deterministic.
    """

    def __init__(self, inner: Any, cell: dict[str, Any]) -> None:
        self._inner = inner
        self._cell = cell

    async def send(self, message: str) -> None:
        await self._inner.send(message)

    async def recv(self) -> str:
        return await self._inner.recv()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._inner.close(code, reason)
        task = self._cell.get("task")
        deadline = asyncio.get_running_loop().time() + 2.0
        while task is not None and not task.done():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("the manager task never observed the close")
            await asyncio.sleep(0.001)

    def close_info(self) -> Any:
        return self._inner.close_info()


async def test_stop_says_shutdown_even_when_the_close_wins_the_race(
    identity: NodeIdentity,
) -> None:
    """The shutdown race, driven deterministically.

    When the manager task classifies ``stop()``'s own 1000 close before
    ``task.cancel()`` lands, the §6 policy would read it as an ordinary closure
    — writing ``link.disconnected`` with reason ``"error"`` and a spurious
    backoff transition for what is a clean shutdown.
    """
    cell: dict[str, Any] = {}
    async with link_harness(
        identity, wrap_socket=lambda socket: _SlowClosingSocket(socket, cell)
    ) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        cell["task"] = harness.manager._task  # noqa: SLF001 - the race is between these two

        await harness.manager.stop()

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.events.events[-1].reason == "shutdown"
        # Exactly one edge: stop()'s own ``if self._online`` block sees it done.
        assert harness.events.types().count("link.disconnected") == 1
        # And no backoff was scheduled for a shutdown.
        assert harness.clockwork.pending == []
        assert harness.status().state != "backoff"


async def test_an_unexpected_failure_goes_terminal_instead_of_killing_the_task(
    identity: NodeIdentity,
) -> None:
    """Nothing observes the ``nerdit-link`` task, so nothing may escape its loop.

    The mux factory eagerly builds an ``httpx.AsyncClient``, so a malformed
    loopback origin raises right here — after ``_on_connected`` has already
    reported ``connected``. Without the last-resort handler the task dies
    silently and ``status()`` is frozen at ``"connected"`` forever, with no live
    task behind it: the exact opposite of "every failure mode degrades to
    doctor-visible".
    """

    def exploding_mux(ctx: MuxContext) -> Any:
        raise httpx.InvalidURL("Invalid port: :1:9321")

    async with link_harness(identity, mux_factory=exploding_mux) as harness:
        await harness.manager.start()
        await harness.wait_state("terminal")

        status = harness.status()
        assert status.state == "terminal"
        assert status.terminal_reason == "error"
        assert status.terminal_reason in DISCONNECT_REASONS
        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.events.events[-1].reason == "error"

        # The task ended by *returning*, not by raising into the void.
        task = harness.manager._task  # noqa: SLF001 - the point of the test
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), 2.0)
        assert task.exception() is None
        # Terminal means terminal: nothing redials into the same failure.
        await asyncio.sleep(0.02)
        assert len(harness.relay.sessions) == 1
        await harness.manager.stop()


async def test_stop_is_safe_before_start_and_twice(identity: NodeIdentity) -> None:
    async with link_harness(identity) as harness:
        await harness.manager.stop()
        assert harness.events.events == []

        await harness.manager.start()
        await harness.wait_state("connected")
        await harness.manager.stop()
        await harness.manager.stop()
        # The second stop must not double-record the edge.
        assert harness.events.types().count("link.disconnected") == 1


async def test_stop_while_backing_off_emits_nothing(identity: NodeIdentity) -> None:
    """A daemon that never connected has no online→offline edge to report."""
    async with link_harness(identity, scenarios=[Scenario(refuse_upgrade=403)]) as harness:
        await harness.manager.start()
        await harness.wait_state("backoff")
        await harness.manager.stop()
        assert harness.events.events == []
        assert harness.status().state == "backoff"


async def test_stop_cancels_a_parked_reconnect_promptly(identity: NodeIdentity) -> None:
    """Shutdown must not wait out a 60 s displacement delay."""
    scenarios = [Scenario(close_with=(CLOSE_REPLACED, "replaced"))]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("displaced")
        await harness.clockwork.wait_for(DISPLACED_RETRY_DELAY_S)
        await asyncio.wait_for(harness.manager.stop(), 2.0)
        assert harness.clockwork.pending == []


# ---------------------------------------------------------------------------
# end to end — manager + the REAL mux + a real socket
# ---------------------------------------------------------------------------


class _StatusApp:
    """The daemon's own listener, in-process."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        self.requests.append(
            {
                "path": scope["path"],
                "headers": {
                    name.decode().lower(): value.decode() for name, value in scope["headers"]
                },
            }
        )
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"node":"up"}'})


async def test_the_whole_tunnel_serves_one_request_end_to_end(identity: NodeIdentity) -> None:
    """Every WP-C1 piece at once, over a real socket.

    ``LinkManager`` mints and proves a capability; the **real** ``StreamMux``
    (not the stand-in used elsewhere in this file) receives the relay's
    ``open_stream``, validates the injected bearer against ``LinkManager``'s
    live capability, replays the request on the loopback client and streams the
    answer back up the same socket.

    The seam this covers exists nowhere else: the other suites test the manager
    with a fake mux and the mux with a fake manager, so only this one can catch
    a ``MuxContext`` the manager fills in wrongly — and in particular that the
    token the relay injects is the token the manager proved.
    """
    app = _StatusApp()

    def real_mux(ctx: MuxContext) -> Any:
        # The manager's own MuxContext, with only the HTTP transport swapped —
        # node_id, limits, send and validate_capability stay exactly as the
        # manager built them.
        return StreamMux(
            dataclasses.replace(
                ctx,
                http_client_factory=lambda: httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url=ctx.loopback_base_url
                ),
            )
        )

    async def forward_one_request(session: Any) -> None:
        assert session.connection is not None
        # The relay injects the capability it just verified (spec §4.2).
        await session.connection.send(
            json.dumps(
                open_stream_frame(
                    stream_id="e2e",
                    method="GET",
                    path="/api/status",
                    token=session.capability_token,
                )
            )
        )

    scenarios = [Scenario(after_ack=forward_one_request)]
    async with link_harness(identity, scenarios=scenarios, mux_factory=real_mux) as harness:
        await harness.manager.start()
        await harness.relay.wait_frame(0, "stream_response_end", timeout=5.0)

        session = harness.relay.sessions[0]
        uplink = [frame for frame in session.uplink if frame["type"].startswith("stream_")]
        assert [frame["type"] for frame in uplink] == [
            "stream_response_head",
            "stream_response_body",
            "stream_response_end",
        ]
        assert uplink[0]["status"] == 200
        assert uplink[0]["node_id"] == FIXTURE_NODE_ID
        assert base64.b64decode(uplink[1]["body_b64"]) == b'{"node":"up"}'

        # The listener really was reached, with the credentials the middleware
        # resolves into the ``link:<node_id>`` principal.
        assert len(app.requests) == 1
        assert app.requests[0]["path"] == "/api/status"
        assert app.requests[0]["headers"]["authorization"] == f"Bearer {session.capability_token}"
        assert app.requests[0]["headers"]["x-nerdit-role"] == "submitter"
        assert harness.status().state == "connected"


async def test_a_forged_capability_is_refused_by_the_live_manager(
    identity: NodeIdentity,
) -> None:
    """The same path, negative: the mux asks the manager, and the manager says no."""
    app = _StatusApp()

    def real_mux(ctx: MuxContext) -> Any:
        return StreamMux(
            dataclasses.replace(
                ctx,
                http_client_factory=lambda: httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url=ctx.loopback_base_url
                ),
            )
        )

    async def forward_a_forgery(session: Any) -> None:
        assert session.connection is not None
        await session.connection.send(
            json.dumps(open_stream_frame(stream_id="e2e", token="not-the-live-capability"))
        )

    scenarios = [Scenario(after_ack=forward_a_forgery)]
    async with link_harness(identity, scenarios=scenarios, mux_factory=real_mux) as harness:
        await harness.manager.start()
        errors = await harness.relay.wait_frame(0, "stream_error", timeout=5.0)

        assert app.requests == [], "the daemon's listener must never be contacted"
        assert errors[0]["code"] == "node_authentication_failed"
        assert errors[0]["stream_id"] == "e2e"


async def test_backoff_caps_the_exponent_before_it_overflows(identity):
    """PR #114 review: ``2.0 ** 1024`` raises OverflowError, so an
    uninterrupted ~17 h outage would turn a retryable condition terminal on
    the 1025th dial. The exponent is capped before exponentiation; the delay
    stays pinned at the ceiling."""
    slept: list[float] = []

    async def sleep(delay: float) -> None:
        slept.append(delay)

    async def connect() -> None:  # pragma: no cover - never dialled here
        raise AssertionError("this test never dials")

    clock = FakeClock()
    manager = LinkManager(
        _settings(),
        identity,
        node_id=FIXTURE_NODE_ID,
        node_name="dev-node",
        daemon_version="0.1.0",
        loopback_base_url="http://127.0.0.1:9321",
        events=FakeRecorder(),  # type: ignore[arg-type]
        connector=connect,
        mux_factory=_RecordingMux,
        clock=clock,
        sleep=sleep,
        monotonic=clock.monotonic,
        rng=lambda: 0.5,  # jitter factor exactly 1.0
    )
    manager._attempt = 5000  # deep into the outage
    await manager._backoff()  # must not raise
    assert slept == [BACKOFF_MAX_S]


# ---------------------------------------------------------------------------
# the entitlement mirror (P32)
# ---------------------------------------------------------------------------


def _seam_manager(identity: NodeIdentity, clock: FakeClock) -> LinkManager:
    """A manager wired to nothing but a clock and a recorder.

    The entitlement seam is deliberately pure — a synchronous setter and a
    synchronous read, both driven by the injected clock — so the TTL matrix
    needs no relay, no socket and no sleeps. That is the whole point of the
    design: expiry is evaluated on READ, so a test can prove the 24 h boundary
    in microseconds and there is no timer to leak.
    """

    async def never_dialled() -> Any:  # pragma: no cover - the seam never dials
        raise AssertionError("the entitlement seam does not touch the socket")

    manager = LinkManager(
        _settings(),
        identity,
        node_id=FIXTURE_NODE_ID,
        node_name="dev-node",
        daemon_version="0.1.0",
        loopback_base_url="http://127.0.0.1:9321",
        events=FakeRecorder(),  # type: ignore[arg-type]
        connector=never_dialled,
        mux_factory=_RecordingMux,
        clock=clock,
        sleep=_never_sleep,
        monotonic=clock.monotonic,
        rng=lambda: 0.5,
    )
    # The seam is reachable only over a live link (Codex round 1): a push that
    # lands after the offline edge must not undo the clear. The matrix runs
    # 'online' unless a test flips it.
    manager._online = True  # noqa: SLF001
    return manager


async def _never_sleep(delay: float) -> None:  # pragma: no cover - never awaited
    raise AssertionError("the entitlement seam does not sleep")


class TestEntitlementSeam:
    """``set_hosted_public_entitled`` + the TTL read, on an injected clock."""

    def test_a_fresh_manager_mirrors_nothing(self, identity: NodeIdentity) -> None:
        """Boot state is the fail-closed one: not entitled, never asserted."""
        status = _seam_manager(identity, FakeClock()).status()
        assert status.hosted_public_entitled is False
        assert status.hosted_public_entitled_at is None

    def test_now_reports_the_injected_clock(self, identity: NodeIdentity) -> None:
        """One clock governs the seam AND the route's future-skew check.

        ``daemon/routes/link.py`` calls :meth:`LinkManager.now` rather than
        ``datetime.now(UTC)`` (review round 1) so an injected clock cannot make
        the route and the seam disagree about which stamps are in the future.
        """
        clock = FakeClock()
        manager = _seam_manager(identity, clock)

        assert manager.now() == clock.now
        clock.advance(3600)
        assert manager.now() == clock.now

    def test_a_true_push_is_applied_and_is_a_change(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)

        update = manager.set_hosted_public_entitled(True, clock.now)

        assert (update.applied, update.changed, update.effective) == (True, True, True)
        assert update.received_at == clock.now
        assert update.expires_at == clock.now + timedelta(seconds=ENTITLEMENT_TTL_S)
        assert manager.status().hosted_public_entitled is True
        assert manager.status().hosted_public_entitled_at == clock.now

    def test_re_asserting_the_same_value_is_applied_but_not_a_change(
        self, identity: NodeIdentity
    ) -> None:
        """The cloud re-asserts every few minutes; only edges are worth a row."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)

        clock.advance(300)
        update = manager.set_hosted_public_entitled(True, clock.now)

        assert (update.applied, update.changed, update.effective) == (True, False, True)
        # …and the refresh really did move the clock the TTL runs against.
        assert update.received_at == clock.now

    def test_the_mirror_survives_to_the_last_second_of_the_ttl(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        asserted_at = clock.now
        manager.set_hosted_public_entitled(True, clock.now)

        clock.advance(ENTITLEMENT_TTL_S - 1)
        assert manager.status().hosted_public_entitled is True

        clock.advance(1)
        expired = manager.status()
        assert expired.hosted_public_entitled is False
        # The timestamp is NOT cleared by expiry: "false because it aged out"
        # and "false because the cloud said no" are different facts, and the
        # CLI renders them differently.
        assert expired.hosted_public_entitled_at == asserted_at

    def test_a_true_push_after_expiry_is_a_change_again(self, identity: NodeIdentity) -> None:
        """The stored bool never moved, but the EFFECTIVE read did — twice."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)

        clock.advance(ENTITLEMENT_TTL_S)
        assert manager.status().hosted_public_entitled is False

        update = manager.set_hosted_public_entitled(True, clock.now)
        assert (update.applied, update.changed, update.effective) == (True, True, True)

    def test_a_false_push_revokes_the_mirror(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)

        clock.advance(60)
        update = manager.set_hosted_public_entitled(False, clock.now)

        assert (update.applied, update.changed, update.effective) == (True, True, False)
        assert manager.status().hosted_public_entitled is False
        assert manager.status().hosted_public_entitled_at == clock.now

    def test_pro_mirror_state_three_way(self, identity: NodeIdentity) -> None:
        """(P34 D3, D-X16-37) All four readings, on the one injected clock.

        A stale assertion cannot confirm public sharing, in either direction.
        """
        clock = FakeClock()
        manager = _seam_manager(identity, clock)

        # Nothing received: a fresh link, or a tunnel that never came up.
        assert manager.pro_mirror_state() == "never"

        manager.set_hosted_public_entitled(True, clock.now)
        assert manager.pro_mirror_state() == "fresh_true"

        clock.advance(60)
        manager.set_hosted_public_entitled(False, clock.now)
        assert manager.pro_mirror_state() == "fresh_false"

        # The last believable second of the lease is still a real assertion…
        clock.advance(ENTITLEMENT_TTL_S - 1)
        assert manager.pro_mirror_state() == "fresh_false"
        # …and one second later the cloud has been quiet for the whole TTL, so
        # the value stops being evidence in EITHER direction.
        clock.advance(1)
        assert manager.pro_mirror_state() == "stale"

        # A positive assertion ages out to exactly the same ambiguity: "stale"
        # never leaks which value it went stale on.
        manager.set_hosted_public_entitled(True, clock.now)
        clock.advance(ENTITLEMENT_TTL_S)
        assert manager.pro_mirror_state() == "stale"

    def test_pro_mirror_state_and_the_effective_read_can_never_disagree(
        self, identity: NodeIdentity
    ) -> None:
        """One TTL evaluation site, projected two ways (P34 D3).

        ``_hosted_public_effective`` is defined as ``state == "fresh_true"``, so
        this asserts the property across the whole matrix rather than trusting
        two independent comparisons to keep agreeing as the code moves.
        """
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        for value, age in (
            (None, 0),  # never asserted
            (True, 0),
            (True, ENTITLEMENT_TTL_S - 1),
            (True, ENTITLEMENT_TTL_S),
            (False, 0),
            (False, ENTITLEMENT_TTL_S),
        ):
            manager.clear_hosted_public_entitlement()
            if value is not None:
                manager.set_hosted_public_entitled(value, clock.now)
                clock.advance(age)
            state = manager.pro_mirror_state()
            assert manager.status().hosted_public_entitled == (state == "fresh_true"), (
                value,
                age,
                state,
            )

    def test_a_disconnect_leaves_the_mirror_reading_never_not_fresh_false(
        self, identity: NodeIdentity
    ) -> None:
        """The clear is a real forgetting, not a push of ``False`` (D-P32-3).

        A disconnect is not a new cloud decision. Clearing drops the stamp,
        so the state falls back to the ambiguous ``"never"``.
        """
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)
        assert manager.pro_mirror_state() == "fresh_true"

        manager.clear_hosted_public_entitlement()

        assert manager.pro_mirror_state() == "never"

    def test_an_older_assertion_is_ignored_rather_than_applied(
        self, identity: NodeIdentity
    ) -> None:
        """Two pushes can race down two streams; ``issued_at`` breaks the tie.

        Without this the retry of an older ``false`` would silently revoke a
        paying customer's share until the next re-assert.
        """
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        newest = clock.now
        manager.set_hosted_public_entitled(True, newest)
        received_at = manager.status().hosted_public_entitled_at

        clock.advance(30)
        update = manager.set_hosted_public_entitled(False, newest - timedelta(seconds=1))

        assert (update.applied, update.changed, update.effective) == (False, False, True)
        assert manager.status().hosted_public_entitled is True
        # Nothing moved — not the value, and not the TTL's own start.
        assert manager.status().hosted_public_entitled_at == received_at

    def test_an_equal_issued_at_is_still_applied(self, identity: NodeIdentity) -> None:
        """Only STRICTLY older is dropped: a re-assert carrying the same stamp
        is the periodic sweep's ordinary shape and must refresh the TTL."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        issued_at = clock.now
        manager.set_hosted_public_entitled(True, issued_at)

        clock.advance(600)
        update = manager.set_hosted_public_entitled(True, issued_at)

        assert update.applied is True
        assert manager.status().hosted_public_entitled_at == clock.now

    def test_a_naive_issued_at_is_a_contract_violation(self, identity: NodeIdentity) -> None:
        """The route 422s first; the assert is depth for direct callers."""
        manager = _seam_manager(identity, FakeClock())
        with pytest.raises(AssertionError):
            manager.set_hosted_public_entitled(True, datetime(2026, 1, 1, 12, 0))  # noqa: DTZ001

    def test_clearing_forgets_the_mirror_entirely(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)

        manager.clear_hosted_public_entitlement()

        status = manager.status()
        assert status.hosted_public_entitled is False
        assert status.hosted_public_entitled_at is None
        # The ordering key is forgotten too, so the cloud's next assertion —
        # whatever it stamps — is accepted rather than dropped as "older".
        update = manager.set_hosted_public_entitled(True, clock.now - timedelta(days=7))
        assert update.applied is True

    def test_a_push_landing_after_the_offline_edge_is_ignored(self, identity: NodeIdentity) -> None:
        """The route authenticated while the link was up; the handler finished
        after ``_go_offline`` cleared the mirror. Applying it would leave an
        offline node entitled for up to the TTL."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)
        manager.clear_hosted_public_entitlement()
        manager._online = False  # noqa: SLF001

        update = manager.set_hosted_public_entitled(True, clock.now)

        assert update.applied is False
        assert manager.status().hosted_public_entitled is False
        assert manager.status().hosted_public_entitled_at is None

    def test_reading_the_mirror_never_awaits(self, identity: NodeIdentity) -> None:
        """``status()`` is a pure snapshot; the TTL must not have changed that."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_hosted_public_entitled(True, clock.now)
        assert not asyncio.iscoroutine(manager.status())


async def test_a_disconnect_clears_the_entitlement_mirror(identity: NodeIdentity) -> None:
    """A dead link asserts nothing (D-P32-3).

    The forwarder already treats a down tunnel as not-public, so a daemon that
    kept the mirror through an outage would keep accepting ``PUT share
    access=public`` for a cloud it can no longer hear from. The cost of
    forgetting is one push on the next node-online edge.
    """
    scenarios = [Scenario(), *[Scenario(refuse_upgrade=403)] * 3]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        harness.manager.set_hosted_public_entitled(True, harness.clock.now)
        assert harness.status().hosted_public_entitled is True

        await harness.clockwork.fire(RENEW_LEAD_S)
        await harness.wait_state("backoff")

        assert harness.events.types() == ["link.connected", "link.disconnected"]
        assert harness.status().hosted_public_entitled is False
        assert harness.status().hosted_public_entitled_at is None
        await harness.manager.stop()


async def test_stop_clears_the_entitlement_mirror(identity: NodeIdentity) -> None:
    """Boot ``False`` is only honest if shutdown leaves nothing behind."""
    async with link_harness(identity) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        harness.manager.set_hosted_public_entitled(True, harness.clock.now)

        await harness.manager.stop()

        assert harness.status().hosted_public_entitled is False
        assert harness.status().hosted_public_entitled_at is None


async def test_a_change_publishes_one_durable_row_carrying_only_the_bool(
    identity: NodeIdentity,
) -> None:
    """The durable feed is POSTed to third-party webhook hosts (D-P24-3).

    So the payload is one key: no account id, no plan name, no node identity
    beyond what the row already carries, and certainly no capability token.
    """
    clock = FakeClock()
    manager = _seam_manager(identity, clock)
    recorder = manager._events

    await manager.record_entitlement_change(True)

    assert recorder.types() == ["link.entitlement"]  # type: ignore[union-attr]
    event = recorder.events[-1]  # type: ignore[union-attr]
    assert event.kind == "link"
    assert event.reason is None
    assert event.data == {"hosted_public_entitled": True}


# ---------------------------------------------------------------------------
# the GitHub installation-token mirror (P33 D-GH-2 / D-GH-3)
# ---------------------------------------------------------------------------

_TOKEN_A = "ghs_installation_token_A_not_a_secret"  # noqa: S105 - a test literal
_TOKEN_B = "ghs_installation_token_B_not_a_secret"  # noqa: S105 - a test literal


def _push(  # noqa: PLR0913 - the push body, field for field
    manager: LinkManager,
    clock: FakeClock,
    *,
    installation_id: int = 11,
    token: str = _TOKEN_A,
    repos: Sequence[str] = ("acme/web",),
    ttl_s: float = 3600,
    issued_at: datetime | None = None,
) -> Any:
    return manager.set_github_token(
        installation_id=installation_id,
        token=token,
        expires_at=clock.now + timedelta(seconds=ttl_s),
        issued_at=issued_at or clock.now,
        repos=repos,
    )


class TestGithubTokenSeam:
    """``set_github_token`` + ``github_token_for_repo``, on an injected clock.

    The seam reuses the entitlement discipline — pure, synchronous, ordered by
    the cloud's ``issued_at``, expiry on READ — so the matrix needs no socket.
    """

    def test_a_fresh_manager_mirrors_nothing(self, identity: NodeIdentity) -> None:
        manager = _seam_manager(identity, FakeClock())
        assert manager.status().github_installations == ()
        assert manager.github_token_for_repo("acme/web") is None

    def test_a_push_is_applied_and_resolves_by_repo(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)

        update = _push(manager, clock, repos=["acme/web", "acme/api"])

        assert (update.applied, update.changed) == (True, True)
        assert update.installation_id == 11
        assert update.repos_count == 2
        assert update.expires_at == clock.now + timedelta(seconds=3600)
        assert manager.github_token_for_repo("acme/web") == _TOKEN_A
        assert manager.github_token_for_repo("acme/api") == _TOKEN_A
        # By repo, not by installation: an unlisted repo resolves to nothing
        # even though a live token exists (D-GH-3, §6 Q3).
        assert manager.github_token_for_repo("acme/other") is None
        summary = manager.status().github_installations
        assert len(summary) == 1
        assert (summary[0].installation_id, summary[0].repos_count) == (11, 2)

    def test_repo_matching_is_canonical_and_case_insensitive(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, repos=["Acme/Web "])
        assert manager.github_token_for_repo("acme/web") == _TOKEN_A
        assert manager.github_token_for_repo("ACME/WEB") == _TOKEN_A

    def test_re_pushing_the_same_token_is_applied_but_not_a_change(
        self, identity: NodeIdentity
    ) -> None:
        """The cloud re-pushes until it re-mints; only edges are worth a row."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        expires = clock.now + timedelta(seconds=3600)
        manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now,
            repos=["a/b"],
        )
        clock.advance(300)
        update = manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now,
            repos=["a/b"],
        )
        assert (update.applied, update.changed) == (True, False)

    def test_a_re_mint_is_a_change(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock)
        clock.advance(1800)
        update = _push(manager, clock, token=_TOKEN_B)
        assert (update.applied, update.changed) == (True, True)
        assert manager.github_token_for_repo("acme/web") == _TOKEN_B

    def test_a_widened_repo_list_is_a_change(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        expires = clock.now + timedelta(seconds=3600)
        manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now,
            repos=["a/b"],
        )
        update = manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now + timedelta(seconds=1),
            repos=["a/b", "a/c"],
        )
        assert update.changed is True
        assert update.repos_count == 2

    def test_swapping_the_repo_set_at_the_same_expiry_is_a_change(
        self, identity: NodeIdentity
    ) -> None:
        """(D6) Only the repos differ — same token, same ``expires_at``, so
        ``repos_count`` is unchanged — yet the effective view moved (org/a is
        no longer resolvable, org/b now is), which the ``changed`` return must
        report so consumers re-run."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        expires = clock.now + timedelta(seconds=3600)
        manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now,
            repos=["org/a"],
        )
        update = manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now + timedelta(seconds=1),
            repos=["org/b"],
        )
        assert update.changed is True
        assert update.repos_count == 1
        assert manager.github_token_for_repo("org/a") is None
        assert manager.github_token_for_repo("org/b") == _TOKEN_A

    def test_an_expired_push_never_evicts_a_live_entry(self, identity: NodeIdentity) -> None:
        """(D3) A newer push whose own token is ALREADY expired at ``now`` must
        not replace a still-live entry — that would silently blank a working
        token until the cloud's next re-mint. It is refused as a no-op and the
        live incumbent is kept, echoing ITS summary; a newer valid token would
        still replace."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        expires = clock.now + timedelta(seconds=3600)
        manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=expires,
            issued_at=clock.now,
            repos=["acme/web"],
        )
        clock.advance(60)  # 11 is still live for another ~59 minutes

        update = manager.set_github_token(
            installation_id=11,
            token=_TOKEN_B,
            expires_at=clock.now - timedelta(seconds=1),  # dead on arrival
            issued_at=clock.now,  # newer than the incumbent
            repos=["acme/web"],
        )

        assert (update.applied, update.changed) == (False, False)
        assert manager.github_token_for_repo("acme/web") == _TOKEN_A
        assert update.repos_count == 1
        assert update.expires_at == expires

    def test_a_still_valid_push_replaces_a_live_entry(self, identity: NodeIdentity) -> None:
        """(D3, the other side) The guard is narrow: a newer, still-valid token
        replaces the live incumbent as before."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.set_github_token(
            installation_id=11,
            token=_TOKEN_A,
            expires_at=clock.now + timedelta(seconds=600),
            issued_at=clock.now,
            repos=["acme/web"],
        )
        clock.advance(60)

        update = manager.set_github_token(
            installation_id=11,
            token=_TOKEN_B,
            expires_at=clock.now + timedelta(seconds=3600),
            issued_at=clock.now,
            repos=["acme/web"],
        )

        assert (update.applied, update.changed) == (True, True)
        assert manager.github_token_for_repo("acme/web") == _TOKEN_B

    def test_an_older_issued_at_is_ignored(self, identity: NodeIdentity) -> None:
        """Two pushes race down two streams; the cloud's stamp orders them."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        newest = clock.now
        _push(manager, clock, token=_TOKEN_B, issued_at=newest)

        update = _push(manager, clock, token=_TOKEN_A, issued_at=newest - timedelta(seconds=1))

        assert (update.applied, update.changed) == (False, False)
        assert update.repos_count == 1
        assert manager.github_token_for_repo("acme/web") == _TOKEN_B

    def test_ordering_is_per_installation(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11, issued_at=clock.now)
        older = clock.now - timedelta(hours=1)
        update = _push(
            manager, clock, installation_id=22, token=_TOKEN_B, repos=["x/y"], issued_at=older
        )
        assert update.applied is True
        assert manager.github_token_for_repo("x/y") == _TOKEN_B

    def test_expiry_is_evaluated_on_read(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, ttl_s=3600)

        clock.advance(3599)
        assert manager.github_token_for_repo("acme/web") == _TOKEN_A
        clock.advance(1)
        assert manager.github_token_for_repo("acme/web") is None
        assert manager.status().github_installations == ()

    def test_the_held_latch_is_empty_on_a_fresh_manager(self, identity: NodeIdentity) -> None:
        """(P33 doctor D2) Never linked ⇒ ``(0, None)`` ⇒ the doctor row is
        ``skipped``, not a warn."""
        manager = _seam_manager(identity, FakeClock())
        assert manager.github_mirror_held() == (0, None)

    def test_the_held_latch_survives_expiry_for_the_doctor(self, identity: NodeIdentity) -> None:
        """The bug fix: once the last token lapses the live surfaces go empty,
        but the raw mirror still HOLDS the lapsed entry, so the doctor can tell
        a silent-pusher node from a never-linked one."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, ttl_s=3600)
        expiry = clock.now + timedelta(seconds=3600)

        clock.advance(3700)  # well past expiry, no re-mint

        # Live surfaces (capabilities / clone) treat it as absent…
        assert manager.status().github_installations == ()
        assert manager.github_token_for_repo("acme/web") is None
        # …but the doctor latch still reports it, with the lapsed expiry.
        held_count, last_expiry = manager.github_mirror_held()
        assert held_count == 1
        assert last_expiry == expiry
        assert last_expiry < clock.now  # overdue

    def test_the_held_latch_reports_the_latest_expiry_of_several(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11, ttl_s=100)
        _push(manager, clock, installation_id=22, token=_TOKEN_B, repos=["x/y"], ttl_s=300)
        latest = clock.now + timedelta(seconds=300)

        clock.advance(400)

        held_count, last_expiry = manager.github_mirror_held()
        assert held_count == 2
        assert last_expiry == latest

    def test_clearing_empties_the_held_latch(self, identity: NodeIdentity) -> None:
        """The offline edge / unlink clears the mirror, so the doctor falls
        back to ``skipped`` — a dead link asserts nothing."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, ttl_s=10)
        clock.advance(60)
        assert manager.github_mirror_held()[0] == 1

        manager.clear_github_tokens()

        assert manager.github_mirror_held() == (0, None)

    def test_a_fresh_push_onto_an_expired_entry_is_a_change(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, ttl_s=10)
        clock.advance(60)
        update = _push(manager, clock, token=_TOKEN_B)
        assert (update.applied, update.changed) == (True, True)

    def test_multiple_installations_resolve_independently(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=22, token=_TOKEN_B, repos=["org/two"])
        _push(manager, clock, installation_id=11, token=_TOKEN_A, repos=["org/one"])

        assert manager.github_token_for_repo("org/one") == _TOKEN_A
        assert manager.github_token_for_repo("org/two") == _TOKEN_B
        ids = [s.installation_id for s in manager.status().github_installations]
        assert ids == [11, 22]

    def test_the_first_installation_listing_a_repo_wins(self, identity: NodeIdentity) -> None:
        """Deterministic: ``installation_id`` order, not arrival order."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=22, token=_TOKEN_B, repos=["org/shared"])
        _push(manager, clock, installation_id=11, token=_TOKEN_A, repos=["org/shared"])
        assert manager.github_token_for_repo("org/shared") == _TOKEN_A

    def test_an_expired_first_match_falls_through_to_the_next(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11, token=_TOKEN_A, repos=["org/shared"], ttl_s=10)
        _push(manager, clock, installation_id=22, token=_TOKEN_B, repos=["org/shared"])
        clock.advance(60)
        assert manager.github_token_for_repo("org/shared") == _TOKEN_B

    def test_clearing_forgets_every_installation_and_its_ordering(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11)
        _push(manager, clock, installation_id=22, repos=["x/y"])

        manager.clear_github_tokens()

        assert manager.status().github_installations == ()
        assert manager.github_token_for_repo("acme/web") is None
        # The ordering key is forgotten too: the next push, whatever its
        # stamp, is accepted rather than dropped as "older".
        update = _push(manager, clock, issued_at=clock.now - timedelta(days=7))
        assert update.applied is True

    def test_a_push_landing_after_the_offline_edge_is_ignored(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager.clear_github_tokens()
        manager._online = False  # noqa: SLF001

        update = _push(manager, clock)

        assert update.applied is False
        assert manager.github_token_for_repo("acme/web") is None

    def test_an_ignored_push_with_nothing_live_echoes_its_own_stamp(
        self, identity: NodeIdentity
    ) -> None:
        """The contract says ``expires_at: str`` — never null, even for a
        no-op (review round): the offline-edge case echoes the push."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        manager._online = False  # noqa: SLF001

        update = _push(manager, clock, ttl_s=1800)

        assert (update.applied, update.repos_count) == (False, 0)
        assert update.expires_at == clock.now + timedelta(seconds=1800)

    def test_an_out_of_order_push_after_expiry_echoes_its_own_stamp(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        newest = clock.now
        _push(manager, clock, ttl_s=60, issued_at=newest)
        clock.advance(120)

        update = _push(manager, clock, ttl_s=3600, issued_at=newest - timedelta(seconds=1))

        assert (update.applied, update.repos_count) == (False, 0)
        assert update.expires_at == clock.now + timedelta(seconds=3600)

    def test_a_push_already_expired_on_arrival_echoes_its_own_stamp(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)

        update = _push(manager, clock, ttl_s=-1)

        assert (update.applied, update.repos_count) == (True, 0)
        assert update.expires_at == clock.now - timedelta(seconds=1)
        assert manager.github_token_for_repo("acme/web") is None

    def test_a_hosted_public_false_push_does_not_clear_the_mirror(
        self, identity: NodeIdentity
    ) -> None:
        """The tokens must survive a ``hosted_public_entitled=false`` push.

        Not because the flag is unrelated (post-X16 it is the Pro bit, so it
        does mean "no GitHub deploys"), but because the cloud quiesces the
        mirror separately with a ``repos: []`` push and then revokes the token
        upstream (D-X16-30, D-X16-40): clearing here would race a downgrade
        the cloud already completes. A lapse that ends hosted access reaches
        the daemon as the relay closing the link, i.e. the offline edge (D-GH-2
        as amended in the review round)."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock)

        update = manager.set_hosted_public_entitled(False, clock.now)

        assert update.applied is True
        assert manager.github_token_for_repo("acme/web") == _TOKEN_A

    def test_a_naive_stamp_is_a_contract_violation(self, identity: NodeIdentity) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        with pytest.raises(AssertionError):
            manager.set_github_token(
                installation_id=1,
                token=_TOKEN_A,
                expires_at=clock.now + timedelta(hours=1),
                issued_at=datetime(2026, 1, 1, 12, 0),  # noqa: DTZ001
                repos=["a/b"],
            )

    def test_the_token_is_absent_from_every_projection(self, identity: NodeIdentity) -> None:
        """``status()``, the update and the manager's own ``repr`` carry no
        token — the only way out is ``github_token_for_repo``."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        update = _push(manager, clock)
        haystack = "\n".join(
            [repr(update), repr(manager.status()), repr(manager._github_tokens)]  # noqa: SLF001
        )
        assert _TOKEN_A not in haystack

    # -- the bounded mirror (security review F1) ----------------------------

    def test_an_accepted_push_evicts_expired_entries(self, identity: NodeIdentity) -> None:
        """Expiry is filtered on READ, but the dict itself must not hoard
        dead tokens forever: the write choke point sweeps them (F1)."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11, ttl_s=60)
        _push(manager, clock, installation_id=22, repos=["x/y"], ttl_s=3600)
        clock.advance(120)  # 11 is now expired, 22 still live

        _push(manager, clock, installation_id=33, repos=["z/w"], ttl_s=3600)

        held = set(manager._github_tokens)  # noqa: SLF001
        assert held == {22, 33}

    def test_a_push_already_expired_on_arrival_is_evicted_by_the_next(
        self, identity: NodeIdentity
    ) -> None:
        """Stored-but-dead on arrival (the ``changed`` honesty choice) is
        still evictable: the next accepted push sweeps it out."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        _push(manager, clock, installation_id=11, ttl_s=-1)
        assert 11 in manager._github_tokens  # noqa: SLF001

        _push(manager, clock, installation_id=22, repos=["x/y"], ttl_s=3600)

        assert set(manager._github_tokens) == {22}  # noqa: SLF001

    def test_the_mirror_caps_distinct_installations(self, identity: NodeIdentity) -> None:
        """At the cap a NEW installation evicts the oldest ``received_at``
        entry — the gitwatch nudge-dedupe precedent for oldest-out."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        for iid in range(1, GITHUB_INSTALLATION_MIRROR_MAX + 1):
            _push(manager, clock, installation_id=iid, repos=[f"org/r{iid}"])
            clock.advance(1)
        assert len(manager._github_tokens) == GITHUB_INSTALLATION_MIRROR_MAX  # noqa: SLF001

        _push(manager, clock, installation_id=9999, repos=["org/new"])

        held = manager._github_tokens  # noqa: SLF001
        assert len(held) == GITHUB_INSTALLATION_MIRROR_MAX
        assert 9999 in held
        assert 1 not in held  # the oldest received went first
        assert 2 in held

    def test_updating_an_existing_installation_at_cap_evicts_nothing(
        self, identity: NodeIdentity
    ) -> None:
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        for iid in range(1, GITHUB_INSTALLATION_MIRROR_MAX + 1):
            _push(manager, clock, installation_id=iid, repos=[f"org/r{iid}"])
            clock.advance(1)

        update = _push(manager, clock, installation_id=1, token=_TOKEN_B, repos=["org/r1"])

        held = manager._github_tokens  # noqa: SLF001
        assert update.applied is True
        assert len(held) == GITHUB_INSTALLATION_MIRROR_MAX
        assert set(range(1, GITHUB_INSTALLATION_MIRROR_MAX + 1)) == set(held)
        assert manager.github_token_for_repo("org/r1") == _TOKEN_B

    def test_an_ignored_older_push_evicts_nothing(self, identity: NodeIdentity) -> None:
        """The older-``issued_at`` no-op stays a full no-op: no sweep, no
        cap eviction, no write."""
        clock = FakeClock()
        manager = _seam_manager(identity, clock)
        newest = clock.now
        _push(manager, clock, installation_id=22, repos=["x/y"], issued_at=newest)
        _push(manager, clock, installation_id=11, ttl_s=-1)  # dead weight

        update = _push(
            manager,
            clock,
            installation_id=22,
            token=_TOKEN_B,
            repos=["x/y"],
            issued_at=newest - timedelta(seconds=1),
        )

        assert update.applied is False
        assert 11 in manager._github_tokens  # noqa: SLF001


async def test_a_disconnect_clears_the_github_token_mirror(identity: NodeIdentity) -> None:
    """A dead link cannot be re-minted for (D-GH-2); the cloud re-pushes on
    the next online edge."""
    scenarios = [Scenario(), *[Scenario(refuse_upgrade=403)] * 3]
    async with link_harness(identity, scenarios=scenarios) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        _push(harness.manager, harness.clock)
        assert harness.manager.github_token_for_repo("acme/web") == _TOKEN_A

        await harness.clockwork.fire(RENEW_LEAD_S)
        await harness.wait_state("backoff")

        assert harness.manager.github_token_for_repo("acme/web") is None
        assert harness.status().github_installations == ()
        await harness.manager.stop()


async def test_stop_clears_the_github_token_mirror(identity: NodeIdentity) -> None:
    """``DELETE /link`` stops the manager, so unlink rides this edge too."""
    async with link_harness(identity) as harness:
        await harness.manager.start()
        await harness.wait_state("connected")
        _push(harness.manager, harness.clock)

        await harness.manager.stop()

        assert harness.manager.github_token_for_repo("acme/web") is None
        assert harness.status().github_installations == ()
