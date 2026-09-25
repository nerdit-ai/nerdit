"""Renew and reconnect node links, recording durable connection events.

Renew at `expires_at - renew_margin_s` by closing before reconnecting: a second
socket would displace the first. In-flight streams therefore end on renewal.
Expiry-related 4401 closes retry with fresh material and bounded backoff;
revocation, verifier mismatch, or bad proof require reconfiguration. Displacement
(4409) is routine, 1001/1013 back off, and 4400 never retries the same protocol
version. The frozen node-link specification defines the 4401 drain ambiguity.

Injected clocks, sleep, and randomness make this policy testable. Import the
mux lazily when starting the manager.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from random import random
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from nerdit.core.link.capability import MintedCapability, mint_capability
from nerdit.core.link.client import (
    AppTarget,
    CloseClass,
    CloseInfo,
    Connector,
    DialRejected,
    InboundStreamHandler,
    LinkConnection,
    LinkHandshakeError,
    MuxContext,
    classify_close,
    dial_relay,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nerdit.config.settings import LinkSettings
    from nerdit.core.eventlog import EventRecorder
    from nerdit.core.link.identity import NodeIdentity

logger = logging.getLogger("nerdit.link.manager")

#: Reconnect ladder: 1 s doubling to a 60 s ceiling, with full ±20 % jitter so
#: a relay restart does not bring every node back in lockstep.
BACKOFF_INITIAL_S = 1.0
BACKOFF_FACTOR = 2.0
BACKOFF_MAX_S = 60.0
BACKOFF_JITTER = 0.2

#: Consecutive handshake-4401s before the link gives up and reports
#: `auth_failed`. Spec §7 asks for "a bounded number of times … minting
#: freshly generated capability material and a new proof for each attempt,
#: before surfacing a credential failure to the operator" — five attempts spans
#: the drain window a relay restart opens without hiding a genuinely dead
#: credential for long.
AUTH_FAILURE_LIMIT = 5

#: Displacement (4409) is routine, and the displaced socket "must **not**
#: reconnect-fight" (spec §6). A fixed, non-exponential delay is the shape that
#: says both things at once: a lingering stale socket heals on the next try,
#: while a genuine second daemon shows up as slow, visible flapping in the
#: durable feed instead of an invisible exponential retreat.
DISPLACED_RETRY_DELAY_S = 60.0

#: Floor on the renewal timer so a misconfigured margin (`>= ttl`) degrades
#: into a slow reconnect rather than a busy loop. The settings model already
#: refuses that combination; this is defense in depth for direct callers.
MIN_RENEW_LEAD_S = 5.0

#: How long a session must last before it counts as *stable* and clears the
#: reconnect ladder. Reaching `hello_ack` is not evidence of a working
#: tunnel: a relay that accepts the Ed25519 handshake and then immediately
#: drops the socket would, if the ladder reset on the handshake alone, be
#: redialled at ~1 Hz forever — a flap that is invisible in the backoff numbers
#: and expensive for both ends. Only a session that survived this long is
#: allowed to say "the last failure was a blip, start the ladder over".
BACKOFF_STABLE_S = 30.0

#: How long an accepted entitlement assertion stays believable.
#: The cloud re-asserts on every node-online edge and on its own periodic sweep
#: (default 300 s), so 24 h is ~288 missed pushes of slack — long enough that a
#: transient cloud outage never revokes a paying customer's public share, short
#: enough that a node the cloud has genuinely stopped talking to drifts back to
#: `False` within a day. Expiry is evaluated **on read** (:meth:`LinkManager.
#: status`), never by a timer: a timer would be a second clock to test and could
#: fire while the daemon is suspended.
ENTITLEMENT_TTL_S = 86400

#: Internal mirror states; only a fresh positive assertion permits publication.
ProMirrorState = Literal["fresh_true", "fresh_false", "stale", "never"]

#: How far ahead of the daemon's clock a push's `issued_at` may sit
#: before the daemon calls it malformed. Two machines' clocks drift; five
#: minutes absorbs ordinary NTP skew without letting a wildly future stamp
#: pin the mirror open against later, correctly-ordered pushes.
ENTITLEMENT_MAX_FUTURE_SKEW_S = 300

#: The fixed machine-token vocabulary of `link.disconnected.reason` (D-P24-3:
#: durable payloads are machine-shaped, never free text).
DISCONNECT_REASONS = frozenset(
    {
        "error",
        "displaced",
        "revoked",
        "auth_failed",
        "protocol_unsupported",
        "entitlement_required",
        "shutdown",
    }
)

_TERMINAL_REASONS: dict[CloseClass, str] = {
    CloseClass.TERMINAL_PROTOCOL: "protocol_unsupported",
    CloseClass.TERMINAL_REVOKED: "revoked",
    CloseClass.TERMINAL_ENTITLEMENT: "entitlement_required",
}


@dataclass(frozen=True, slots=True)
class EntitlementUpdate:
    """What one `PUT /api/link/entitlement` push did to the mirror.

    `applied` is `False` for a push the manager *ignored*: one whose
    `issued_at` is older than the one already recorded, or one that landed
    after the link went offline (the mirror was cleared; a dead link asserts
    nothing, D-P32-3). `changed` is about the **effective** read, not the stored bool:
    a `True` push that merely refreshes a live `True` is applied but not a
    change, while a `True` push landing on an EXPIRED `True` is both. Only
    `changed` writes an audit row and a durable event.
    """

    applied: bool
    changed: bool
    effective: bool
    received_at: datetime | None
    expires_at: datetime | None


#: Below this much remaining life the doctor `github_token` row
#: reads `warn`: GitHub installation tokens live at most an hour and the
#: cloud re-mints ahead of expiry, so a token this close to the edge means
#: the pusher has gone quiet and the next private clone is about to fail.
GITHUB_TOKEN_WARN_LEAD_S = 600

#: Hard cap on DISTINCT installations the mirror
#: will hold. The writer is the authenticated cloud, but the mirror must
#: still be bounded — a compromised or buggy pusher inventing installation
#: ids must not grow daemon memory without limit. 100 is far beyond any
#: real account (installations are per-GitHub-org); at the cap the OLDEST
#: `received_at` entry is evicted first, the gitwatch nudge-dedupe
#: precedent for a bounded oldest-out structure.
GITHUB_INSTALLATION_MIRROR_MAX = 100


@dataclass(frozen=True, slots=True)
class GithubTokenUpdate:
    """What one `PUT /api/link/github-token` push did to the mirror.

    `applied` is `False` for a push the manager *ignored*: one whose
    `issued_at` is older than the one already recorded for that
    installation, or one that landed after the link went offline (D-GH-2).
    `changed` is about the **effective** read — the live (non-expired) entry
    for the installation before versus after — so a re-push of the same token
    is applied but not a change, while a fresh token (new `expires_at`) or a
    widened `repos` list is. Only `changed` writes an audit row.
    Deliberately carries no token: this object is what the route projects.

    `expires_at` is never `None` (the cross-repo contract says `str`):
    it is the live entry's stamp when one exists, else the push's own — an
    ignored or already-expired push echoes what it carried, and
    `repos_count` is then `0` for "nothing live".
    """

    applied: bool
    changed: bool
    installation_id: int
    expires_at: datetime
    repos_count: int


@dataclass(frozen=True, slots=True)
class GithubInstallation:
    """One mirrored installation token (D-GH-2). In memory only.

    `repos` is the cloud's repo-filtered list for THIS node (§6 Q3), in
    canonical `owner/name` lower-case form; resolution is by repo
    (D-GH-3). `token` never leaves the manager except through
    `LinkManager.github_token_for_repo` — `repr=False` keeps it out
    of any accidental `%r`/`str()`.
    """

    installation_id: int
    token: str = field(repr=False)
    expires_at: datetime
    issued_at: datetime
    repos: tuple[str, ...]
    received_at: datetime


@dataclass(frozen=True, slots=True)
class GithubInstallationSummary:
    """The secret-free projection of one live installation: what
    `/capabilities`, the doctor row and `nerdit link` see. Never the
    token, never the repo names — a count is the whole of what a status
    surface needs (D-GH-7)."""

    installation_id: int
    expires_at: datetime
    repos_count: int


@dataclass(frozen=True, slots=True)
class LinkStatus:
    """Synchronous snapshot of the link, safe for any surface to project.

    Deliberately free of secrets and of the relay URL: `relay_host` is
    `host[:port]` only, and the capability token appears nowhere — this
    object is what `/capabilities` and `/doctor` render.
    """

    state: str  # "connecting" | "connected" | "backoff" | "displaced" | "terminal"
    node_id: str
    relay_host: str
    connected_at: datetime | None
    capability_expires_at: datetime | None
    last_close_code: int | None
    last_error_code: str | None
    terminal_reason: str | None
    retry_at: datetime | None
    attempt: int
    auth_failures: int
    active_streams: int
    #: May this node's account expose a PUBLIC hosted share?
    #: A trailing defaulted field so every existing constructor call stays
    #: valid. Written by `PUT /api/link/entitlement` — the cloud's
    #: entitlement push — and read here through a 24 h TTL
    #: (`ENTITLEMENT_TTL_S`), so it is `False` on a node the cloud has
    #: stopped asserting for. It is a UX mirror only: the cloud forwarder
    #: remains the sole judge of anonymous traffic. Locally it
    #: decides whether `PUT share access=public` is accepted and how
    #: `share.state` renders.
    hosted_public_entitled: bool = False
    #: When the last accepted push landed on this daemon's clock, or
    #: `None` while the cloud has never asserted. Trailing and defaulted for
    #: the same constructor-compatibility reason as the field above. It is a
    #: timestamp, not a secret: `/capabilities` projects it for every role so
    #: `nerdit link` can say how stale the mirror is.
    hosted_public_entitled_at: datetime | None = None
    #: The live (non-expired) GitHub installation tokens, as secret-free
    #: summaries — `()` while the cloud has pushed none. Trailing and
    #: defaulted like the two fields above.
    github_installations: tuple[GithubInstallationSummary, ...] = ()


class LinkManager:
    """Owns the tunnel across connections: minting, renewal, retry, events."""

    def __init__(  # noqa: PLR0913 - identity, metadata and four injected clocks
        self,
        link: LinkSettings,
        identity: NodeIdentity,
        *,
        node_id: str,
        node_name: str,
        daemon_version: str,
        loopback_base_url: str,
        events: EventRecorder | None = None,
        connector: Connector | None = None,
        mux_factory: Callable[[MuxContext], InboundStreamHandler] | None = None,
        resolve_app: Callable[
            [str, str | None, str | None, str | None], Awaitable[AppTarget | None]
        ]
        | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        monotonic: Callable[[], float] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self._link = link
        self._identity = identity
        self._node_id = node_id
        self._node_name = node_name
        self._daemon_version = daemon_version
        self._loopback_base_url = loopback_base_url
        self._events = events
        self._connector: Connector = connector or (lambda: dial_relay(link.relay_url))
        self._mux_factory = mux_factory
        #: The share lookup handed to every `MuxContext`.
        #: `None` — an unlinked, slug-less or domain-less node — means the mux
        #: serves no app stream at all.
        self._resolve_app = resolve_app
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep or asyncio.sleep
        self._monotonic = monotonic or time.monotonic
        self._rng = rng or random
        self._relay_host = urlsplit(link.relay_url).netloc

        self._started_at = self._monotonic()
        self._stopping = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        # Session state — all of it read synchronously by `status`.
        self._state = "connecting"
        self._online = False
        self._connected_at: datetime | None = None
        self._capability: MintedCapability | None = None
        self._last_close_code: int | None = None
        self._last_error_code: str | None = None
        self._terminal_reason: str | None = None
        self._retry_at: datetime | None = None
        self._attempt = 0
        self._auth_failures = 0
        self._connection: LinkConnection | None = None
        self._mux: InboundStreamHandler | None = None
        #: Monotonic stamp of the current session's `hello_ack`, or `None`
        #: between sockets. Read by `_handle_close` to decide whether the
        #: session lasted long enough to clear the ladder (`BACKOFF_STABLE_S`).
        self._connected_monotonic: float | None = None
        #: The entitlement mirror: what the cloud last
        #: asserted about this account's right to publish a PUBLIC hosted
        #: share, when the daemon received it, and the cloud's own stamp for
        #: that assertion. Written ONLY by
        #: `set_hosted_public_entitled` (the `PUT /api/link/entitlement`
        #: route) and cleared by `clear_hosted_public_entitlement`.
        #: In memory only — never persisted, so a restart boots `False` and
        #: waits for the cloud to re-assert.
        self._entitlement_value: bool = False
        self._entitlement_received_at: datetime | None = None
        self._entitlement_issued_at: datetime | None = None
        #: The GitHub installation-token mirror, keyed by
        #: `installation_id`. Written ONLY by `set_github_token` (the
        #: `PUT /api/link/github-token` route), read by
        #: `github_token_for_repo`, cleared by `clear_github_tokens`.
        #: In memory only — never persisted, never logged, never audited; a
        #: restart boots empty and waits for the cloud to push again.
        self._github_tokens: dict[int, GithubInstallation] = {}

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Start the connect loop. Idempotent."""
        if self._task is not None:
            return
        if self._mux_factory is None:
            # Lazy on purpose: `core.link.client`/`manager` import cleanly
            # without `mux`, and a mux import error belongs to whoever asked
            # for a tunnel, not to daemon boot.
            from nerdit.core.link.mux import StreamMux  # noqa: PLC0415

            self._mux_factory = StreamMux
        self._stopping.clear()
        self._state = "connecting"
        self._task = asyncio.create_task(self._run(), name="nerdit-link")

    async def stop(self) -> None:
        """Stop the loop, close the socket and drain the mux."""
        self._stopping.set()
        task, self._task = self._task, None
        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(Exception):
                await connection.close(1000, "daemon shutdown")
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        mux, self._mux = self._mux, None
        if mux is not None:
            with suppress(Exception):
                await mux.aclose()
        # Unconditional and ordered BEFORE the durable row: a manager
        # that never came online still leaves nothing for a re-`start()` to
        # inherit, and a consumer reacting to `link.disconnected` by reading
        # `/capabilities` can never observe a stale `True`.
        self.clear_hosted_public_entitlement()
        self.clear_github_tokens()
        if self._online:
            self._online = False
            await self._record_disconnected("shutdown", None)
        self._connected_at = None
        self._capability = None
        self._connected_monotonic = None

    # -- surface -----------------------------------------------------------

    def now(self) -> datetime:
        """This manager's notion of "now".

        The entitlement seam has exactly one clock: `received_at`, the TTL and
        the ordering comparison in `set_hosted_public_entitled` all read
        `self._clock`. The route's future-skew check must read the same one,
        or an injected clock makes the route and the seam disagree about which
        stamps are "in the future" — in production both are the wall clock, so
        this is a testability seam, not a behaviour change.
        """
        return self._clock()

    def status(self) -> LinkStatus:
        """Return a pure snapshot — no I/O, no locks, no awaits."""
        mux = self._mux
        return LinkStatus(
            state=self._state,
            node_id=self._node_id,
            relay_host=self._relay_host,
            connected_at=self._connected_at,
            capability_expires_at=(
                self._capability.expires_at if self._capability is not None else None
            ),
            last_close_code=self._last_close_code,
            last_error_code=self._last_error_code,
            terminal_reason=self._terminal_reason,
            retry_at=self._retry_at,
            attempt=self._attempt,
            auth_failures=self._auth_failures,
            active_streams=mux.active_streams() if mux is not None else 0,
            hosted_public_entitled=self._hosted_public_effective(),
            hosted_public_entitled_at=self._entitlement_received_at,
            github_installations=self.github_installations(),
        )

    # -- entitlement mirror -----------------------------------------

    def pro_mirror_state(self) -> ProMirrorState:
        """The mirror read as one of four machine tokens (D-X16-37).

        `_hosted_public_effective` collapses four distinguishable facts
        into one bool, which is right for "may this node serve a public share"
        and wrong for anything that has to tell a user *why*. Splitting them
        out here — rather than in the caller — keeps the TTL on one clock:
        `ENTITLEMENT_TTL_S` is evaluated in **exactly
        this method**, against `self._clock()`, the same injected clock that
        stamped `_entitlement_received_at` on receipt. A second reader that
        re-derived freshness from `LinkStatus.hosted_public_entitled_at` and
        `datetime.now(UTC)` would be a second clock to test and would drift
        the moment either side rounded differently.

        * `"never"` — nothing received: a fresh link, or a tunnel that has
          never come up. Says nothing whatever about the account.
        * `"stale"` — something *was* received, longer than the 24 h TTL ago.
          The value is no longer believable in either direction.
        * `"fresh_true"` — a recent positive assertion; the only state
          `_hosted_public_effective` answers `True` for.
        * `"fresh_false"` — a recent *negative* assertion: the cloud looked at
          this account and said no. The only state a caller may read as
          evidence about hosted access; the other three are ambiguous
          and every caller must fall back to saying nothing specific.

        Read-only and synchronous, like every other reader of the mirror, and
        advisory everywhere: the mirror is a UX mirror, never an authority —
        the cloud remains the only judge of entitlement (D-ENT-2).
        """
        received_at = self._entitlement_received_at
        if received_at is None:
            return "never"
        if (self._clock() - received_at) >= timedelta(seconds=ENTITLEMENT_TTL_S):
            return "stale"
        return "fresh_true" if self._entitlement_value else "fresh_false"

    def _hosted_public_effective(self) -> bool:
        """The mirror as every reader sees it: the value, aged out.

        Fail closed on three counts — a `False` value, a value never
        received, and a value older than `ENTITLEMENT_TTL_S`. Pure and
        synchronous like `status`, which is its only caller in the
        daemon; expiry therefore needs no timer.

        Expressed in terms of `pro_mirror_state` so the TTL keeps
        having exactly one evaluation site: the two readings cannot drift apart
        because there is only one of them, projected two ways.
        """
        return self.pro_mirror_state() == "fresh_true"

    def set_hosted_public_entitled(self, value: bool, issued_at: datetime) -> EntitlementUpdate:
        """Record one cloud entitlement assertion. Pure, synchronous, no I/O.

        `issued_at` orders assertions **at the source**, because the tunnel
        does not: two pushes racing down two streams can arrive in either
        order, and the daemon's own receipt time cannot tell which value the
        cloud decided last. An assertion older than the one already recorded is
        therefore dropped (`applied=False`) rather than applied and then
        overwritten by a retry of the older one.

        The caller (`daemon/routes/link.py`) has already refused a naive or
        implausibly-future stamp; the assert here is a contract check for
        direct callers, not input validation.
        """
        assert issued_at.tzinfo is not None, "issued_at must be timezone-aware"
        before = self._hosted_public_effective()
        previous_issued_at = self._entitlement_issued_at
        # A push is only meaningful on a live link: the route authenticated it
        # while the tunnel was up, but an in-flight handler can outlive the
        # mux's cancellation and finish AFTER `_go_offline` cleared the
        # mirror. Applying it then would undo the clear for up to the TTL.
        # Checked here, in the same synchronous step as the write, so the
        # offline edge and the apply cannot interleave.
        stale = not self._online or (
            previous_issued_at is not None and issued_at < previous_issued_at
        )
        if stale:
            return EntitlementUpdate(
                applied=False,
                changed=False,
                effective=before,
                received_at=self._entitlement_received_at,
                expires_at=self._entitlement_expires_at(),
            )
        self._entitlement_value = value
        self._entitlement_received_at = self._clock()
        self._entitlement_issued_at = issued_at
        after = self._hosted_public_effective()
        return EntitlementUpdate(
            applied=True,
            changed=after != before,
            effective=after,
            received_at=self._entitlement_received_at,
            expires_at=self._entitlement_expires_at(),
        )

    def _entitlement_expires_at(self) -> datetime | None:
        received_at = self._entitlement_received_at
        if received_at is None:
            return None
        return received_at + timedelta(seconds=ENTITLEMENT_TTL_S)

    def clear_hosted_public_entitlement(self) -> None:
        """Forget the mirror — a dead link asserts nothing.

        Called on every online→offline edge and on `stop`. The cloud
        re-asserts on the next node-online, so the cost of forgetting is one
        push; the cost of *not* forgetting would be a daemon that keeps
        accepting `PUT share access=public` for up to 24 h after the account
        that entitled it stopped being reachable at all.
        """
        self._entitlement_value = False
        self._entitlement_received_at = None
        self._entitlement_issued_at = None

    async def record_entitlement_change(self, value: bool) -> None:
        """Publish the durable `link.entitlement` row for a value CHANGE.

        Separate from `set_hosted_public_entitled` on purpose: the setter
        is pure and synchronous (`status()`'s discipline), while the durable
        feed is I/O. The route calls this only when the update reported
        `changed` — the feed carries edges, not pushes.
        """
        await self._record("link.entitlement", data={"hosted_public_entitled": value})

    # -- GitHub installation-token mirror (D-GH-2 / D-GH-3) -----------------

    def _live_github_installations(self) -> list[GithubInstallation]:
        """Every mirrored installation whose token has not expired, in
        `installation_id` order. Expiry is evaluated on READ against the
        injected clock (the entitlement precedent) — no timer, no second clock.
        """
        now = self._clock()
        return [entry for _, entry in sorted(self._github_tokens.items()) if entry.expires_at > now]

    def github_installations(self) -> tuple[GithubInstallationSummary, ...]:
        """Secret-free summaries of the live installations (status surfaces)."""
        return tuple(
            GithubInstallationSummary(
                installation_id=entry.installation_id,
                expires_at=entry.expires_at,
                repos_count=len(entry.repos),
            )
            for entry in self._live_github_installations()
        )

    def github_mirror_held(self) -> tuple[int, datetime | None]:
        """The doctor's overdue latch: the count of mirrored
        installations *regardless of expiry*, and the latest `expires_at`
        among them (`(0, None)` when the mirror is empty).

        Every traffic-bearing surface reads live-only —
        `github_installations` for `/capabilities` and `nerdit link`,
        `github_token_for_repo` for a clone — so an expired entry is
        unusable and invisible there, unchanged by this method. But once the
        LAST live token lapses those go empty, and the doctor could no longer
        tell a node whose cloud pusher has gone SILENT (every private
        auto-deploy stopped — the D-GH-9 quiet-backoff blind spot) from one
        that was never linked: both read as "nothing mirrored", a deceptively
        green row. Expired entries linger in the mirror (they are swept only by
        the next accepted push, and the online→offline edge empties it via
        `clear_github_tokens`), so a non-zero held count with nothing
        live means exactly "we held a token and the cloud let it lapse while
        the link stayed up". This read inspects what already lingers — it adds
        no retention, so the mirror's memory bound and the live-only resolution
        are both untouched.
        """
        if not self._github_tokens:
            return 0, None
        latest = max(entry.expires_at for entry in self._github_tokens.values())
        return len(self._github_tokens), latest

    def github_token_for_repo(self, canonical_repo: str) -> str | None:
        """The live token for `owner/name` (lower-case), or `None`.

        Resolution is BY REPO (D-GH-3): the first live installation, in
        `installation_id` order, whose `repos` list carries the canonical
        name wins; no match — or a match whose token expired — is `None`,
        which the callers treat as "absent" (quiet in GitWatch, a loud 422 on
        `POST /deploy/git`; D-GH-9). The one method that hands the secret
        out: its only callers are the git credential seam (`GIT_ASKPASS`).
        """
        wanted = canonical_repo.strip().lower()
        for entry in self._live_github_installations():
            if wanted in entry.repos:
                return entry.token
        return None

    def set_github_token(  # noqa: PLR0913 - the push body, field for field
        self,
        *,
        installation_id: int,
        token: str,
        expires_at: datetime,
        issued_at: datetime,
        repos: Sequence[str],
    ) -> GithubTokenUpdate:
        """Record one cloud token push. Pure, synchronous, no I/O (D-GH-2).

        Same discipline as `set_hosted_public_entitled`: ordered by the
        cloud's `issued_at` per installation (an older push is dropped,
        `applied=False`), refused after the offline edge (a dead link
        asserts nothing, and an in-flight handler must not undo the clear),
        and the route has already refused naive or far-future stamps. A token
        that is already expired on arrival is still stored WHEN there is no
        live incumbent — it simply never reads as live — so `changed` stays
        honest about the effective view, and it is then garbage for the next
        accepted push's expiry sweep, exactly like an entry that expired while
        mirrored. The one exception: an already-expired push must NOT
        evict a currently-live entry — that would silently blank a working
        token until the cloud's next re-mint — so it is refused as a no-op
        (`applied=False`) and the live entry is kept; a newer, still-valid
        token still replaces.

        The mirror is bounded: every accepted push first
        evicts entries whose `expires_at` has passed, then — only when the
        push introduces a NEW `installation_id` — enforces
        `GITHUB_INSTALLATION_MIRROR_MAX` by dropping the oldest
        `received_at` entry. Updating an already-mirrored installation never
        counts against the cap.
        """
        assert issued_at.tzinfo is not None, "issued_at must be timezone-aware"
        assert expires_at.tzinfo is not None, "expires_at must be timezone-aware"
        before = self._github_summary(installation_id)
        previous = self._github_tokens.get(installation_id)
        stale = not self._online or (previous is not None and issued_at < previous.issued_at)
        if stale:
            return GithubTokenUpdate(
                applied=False,
                changed=False,
                installation_id=installation_id,
                expires_at=before.expires_at if before is not None else expires_at,
                repos_count=before.repos_count if before is not None else 0,
            )
        now = self._clock()
        # An already-expired push never replaces a live incumbent (see above).
        if expires_at <= now and previous is not None and previous.expires_at > now:
            return GithubTokenUpdate(
                applied=False,
                changed=False,
                installation_id=installation_id,
                expires_at=before.expires_at if before is not None else expires_at,
                repos_count=before.repos_count if before is not None else 0,
            )
        # Bound the mirror before the write: expired
        # entries are dead weight the read path already filters, so evict them
        # here — the one write choke point — against the same injected clock.
        for expired_id in [
            iid for iid, entry in self._github_tokens.items() if entry.expires_at <= now
        ]:
            del self._github_tokens[expired_id]
        # Cap DISTINCT installations; an update-in-place never evicts.
        if installation_id not in self._github_tokens:
            while len(self._github_tokens) >= GITHUB_INSTALLATION_MIRROR_MAX:
                oldest = min(
                    self._github_tokens.values(),
                    key=lambda entry: (entry.received_at, entry.installation_id),
                )
                del self._github_tokens[oldest.installation_id]
        new_repos = tuple(sorted({repo.strip().lower() for repo in repos}))
        # `changed` drives consumers, so it must track the EFFECTIVE read
        # — the live `(expires_at, repos)` pair — not merely presence. A
        # same-expiry push that only swaps the repo set (org/a → org/b, a
        # narrow or a widen) is a real change even though `repos_count` and
        # `expires_at` are identical; comparing the repos tuple catches what
        # the live-filtered summary's count alone cannot.
        before_effective = (
            (previous.expires_at, previous.repos)
            if previous is not None and previous.expires_at > now
            else None
        )
        after_effective = (expires_at, new_repos) if expires_at > now else None
        self._github_tokens[installation_id] = GithubInstallation(
            installation_id=installation_id,
            token=token,
            expires_at=expires_at,
            issued_at=issued_at,
            repos=new_repos,
            received_at=now,
        )
        after = self._github_summary(installation_id)
        return GithubTokenUpdate(
            applied=True,
            changed=before_effective != after_effective,
            installation_id=installation_id,
            expires_at=after.expires_at if after is not None else expires_at,
            repos_count=after.repos_count if after is not None else 0,
        )

    def _github_summary(self, installation_id: int) -> GithubInstallationSummary | None:
        for summary in self.github_installations():
            if summary.installation_id == installation_id:
                return summary
        return None

    def clear_github_tokens(self) -> None:
        """Forget every mirrored token — on the offline edge, on `stop`
        (which is what `DELETE /link` calls), and therefore on unlink. The
        cloud re-pushes on the next node-online edge, so the cost of
        forgetting is one push per installation.

        Independent of `set_hosted_public_entitled`: repository authorization
        is separate from public sharing. The cloud quiesces a revoked installation
        with `repos: []` before revoking its token upstream. Clearing here on a
        public-share assertion would race that sequence. Account standing is
        enforced by the relay closing the link, which clears both mirrors.
        """
        self._github_tokens.clear()

    def validate_capability(self, token: str, role: str) -> bool:
        """Return whether `token` is the capability proven on the LIVE link.

        Three conditions, all required: a session is up, the role is the
        permanent `"submitter"` ceiling (D-R2 as amended by ADR-W1), and the
        capability has not lapsed against the injected clock — expiry is
        reactive on the relay (spec §9), so the daemon rechecks rather than
        trusting that a live socket implies a live capability.

        The token comparison is `secrets.compare_digest`: this is called
        from the daemon's auth middleware on every tunnelled request, so a
        short-circuiting `==` would make it a timing oracle for a bearer.

        It compares **bytes**, not `str`: `compare_digest` raises
        `TypeError` on a `str` holding any non-ASCII character, and the
        argument here comes straight out of an attacker-controlled
        `Authorization` header (Starlette decodes headers as latin-1, so
        `Bearer é` is a perfectly reachable input). Comparing on the encoded
        form turns a 500 on every such request into the `False` it always
        meant. The minted token is `secrets.token_urlsafe` — ASCII by
        construction — so no legitimate credential is affected.
        """
        capability = self._capability
        if capability is None or not self._online:
            return False
        if role != "submitter":
            return False
        if self._clock() >= capability.expires_at:
            return False
        return secrets.compare_digest(
            token.encode("utf-8", "replace"), capability.token.encode("ascii")
        )

    # -- the loop ----------------------------------------------------------

    def _uptime_s(self) -> int:
        return max(0, int(self._monotonic() - self._started_at))

    async def _run(self) -> None:
        """Drive `_connect_once` until terminal or stopped, whatever happens.

        The last-resort handler is the load-bearing part. This coroutine is the
        `nerdit-link` task and **nothing observes it**: an exception escaping
        the loop would kill the task silently, freezing `status()` at whatever
        it last said — `"connected"`, with no live task behind it — which is
        precisely the "every failure mode degrades to doctor-visible" promise
        inverted. Anything unexpected therefore ends as a doctor-visible
        terminal state instead of as a dead task. (It is reachable: the mux
        factory eagerly builds an `httpx.AsyncClient`, which raises
        `httpx.InvalidURL` on a malformed loopback origin.)
        """
        while not self._stopping.is_set():
            try:
                if await self._connect_once():
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the task must never die unobserved
                logger.exception("The node link failed unexpectedly and is giving up")
                await self._go_terminal(
                    "error",
                    CloseInfo(code=None, reason=str(exc), error_code=None, cause="local"),
                )
                return

    async def _connect_once(self) -> bool:
        """One connect/serve/close cycle. Returns `True` when terminal."""
        capability = mint_capability(self._link.capability_ttl_s, clock=self._clock)

        self._state = "connecting"
        try:
            socket = await self._connector()
        except DialRejected as exc:
            # Spec §5 step 1 / §7: a pre-accept refusal is capacity or
            # drain, with 1013's semantics. Retry with backoff.
            logger.warning("Relay refused the tunnel upgrade (HTTP %d)", exc.status)
            await self._dial_failed(exc)
            return False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any dial failure is a retry
            logger.warning("Relay dial failed: %s", exc)
            await self._dial_failed(exc)
            return False

        connection = LinkConnection(
            socket,
            identity=self._identity,
            node_id=self._node_id,
            node_name=self._node_name,
            daemon_version=self._daemon_version,
            uptime_s=self._uptime_s,
            sleep=self._sleep,
        )
        self._connection = connection
        try:
            handshake = await connection.handshake(capability)
        except LinkHandshakeError as exc:
            with suppress(Exception):
                await connection.close(1000, "handshake failed")
            self._connection = None
            close = exc.close or CloseInfo(
                code=None, reason=str(exc), error_code=exc.error_code, cause="transport"
            )
            return await self._handle_close(close)

        await self._on_connected(capability, handshake.node_id)
        factory = self._mux_factory
        if factory is None:  # pragma: no cover - start() always installs one
            raise RuntimeError("start() must install a mux factory before connecting")
        mux = factory(
            MuxContext(
                node_id=handshake.node_id,
                limits=handshake.limits,
                send=connection.send_frame,
                validate_capability=self.validate_capability,
                loopback_base_url=self._loopback_base_url,
                resolve_app=self._resolve_app,
            )
        )
        self._mux = mux

        renewed, close = await self._serve(connection, mux, capability)

        self._connection = None
        self._mux = None
        with suppress(Exception):
            await mux.aclose()

        if renewed:
            # A successful in-place renewal is not an outage: `online` spans
            # it, no durable event is emitted (that would be ~13-min event
            # spam), and there is no backoff — reconnect immediately with
            # freshly minted material (D-R8).
            #
            # The capability, though, dies with the socket it was proven on:
            # between this close and the next `hello_ack` there is nothing
            # live to validate against, and leaving it installed would keep the
            # middleware honouring a bearer whose connection is gone — for as
            # long as the re-dial takes, which is unbounded when the relay has
            # just become unreachable. `_on_connected` reinstalls the fresh
            # one on success, so the no-event fast path is preserved.
            self._capability = None
            self._connected_at = None
            self._connected_monotonic = None
            return False
        return await self._handle_close(close)

    async def _dial_failed(self, exc: BaseException) -> None:
        """Record the outage a failed dial represents, then back off.

        A dial can fail at a renewal boundary — the moment the link is between
        two sockets while still `online`. Without this, an outage that starts
        exactly there is never written to the durable feed: the manager retries
        forever, reporting `online` through arbitrarily many refused dials,
        and the operator's event feed shows a link that never went down.
        `_go_offline` is idempotent, so the ordinary "never connected yet"
        case still writes nothing.
        """
        await self._go_offline(
            "error", CloseInfo(code=None, reason=str(exc), error_code=None, cause="transport")
        )
        await self._backoff()

    async def _serve(
        self,
        connection: LinkConnection,
        mux: InboundStreamHandler,
        capability: MintedCapability,
    ) -> tuple[bool, CloseInfo]:
        """Race the session against the renewal timer.

        Returns `(renewed, close)`: `renewed` is `True` when the timer
        won and we closed cleanly to re-prove fresh material.
        """
        run_task = asyncio.create_task(connection.run(mux), name="nerdit-link-session")
        renew_task = asyncio.create_task(
            self._renewal_timer(self._renew_delay_s(capability)), name="nerdit-link-renewal"
        )
        try:
            await asyncio.wait({run_task, renew_task}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # Cancel *and reap*. Cancelling without awaiting leaves the
            # `nerdit-link-session` task (and the heartbeat child it owns in a
            # `finally`) pending on a wedged `recv` — asyncio then reports
            # "Task was destroyed but it is pending" at shutdown, and the socket
            # is not actually released.
            run_task.cancel()
            renew_task.cancel()
            await asyncio.gather(run_task, renew_task, return_exceptions=True)
            raise

        renewed = run_task.done() is False
        if renewed:
            logger.info("Renewing the tunnel capability (reconnecting with fresh material)")
            with suppress(Exception):
                await connection.close(1000, "capability renewal")
        renew_task.cancel()
        with suppress(asyncio.CancelledError):
            await renew_task

        try:
            close = await run_task
        except asyncio.CancelledError:
            # Same reaping obligation as above: a cancellation delivered while
            # parked here must not orphan the session task.
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            raise
        except Exception as exc:  # noqa: BLE001 - a session crash is a closure
            logger.warning("The tunnel session ended abnormally: %s", exc)
            close = CloseInfo(code=None, reason=str(exc), error_code=None, cause="transport")
        return renewed, close

    async def _renewal_timer(self, delay_s: float) -> None:
        """Sleep until renewal is due (a coroutine so it can be a task)."""
        await self._sleep(delay_s)

    def _renew_delay_s(self, capability: MintedCapability) -> float:
        """Seconds until `expires_at − renew_margin_s` (floored)."""
        due = capability.expires_at - timedelta(seconds=self._link.renew_margin_s)
        return max((due - self._clock()).total_seconds(), MIN_RENEW_LEAD_S)

    async def _on_connected(self, capability: MintedCapability, node_id: str) -> None:
        """Install the proven capability and announce an offline→online edge.

        `_auth_failures` resets here and `_attempt` deliberately does not:
        spec §7 says an accepted fresh proof clears the *credential* doubt, but
        it says nothing about the connection being usable. Clearing the
        reconnect ladder is `_handle_close`'s job, and only once the
        session has lasted `BACKOFF_STABLE_S`.
        """
        self._auth_failures = 0
        self._retry_at = None
        self._capability = capability
        self._state = "connected"
        self._connected_at = self._clock()
        self._connected_monotonic = self._monotonic()
        self._terminal_reason = None
        if not self._online:
            self._online = True
            # `relay_host` is deliberately NOT in the payload: GET /events is
            # readable by any authenticated principal, while /capabilities gates
            # the same value behind admin (the proxy.admin_addr precedent). The
            # weaker gate must not leak past the stronger one, and `node_id`
            # alone identifies the session.
            await self._record("link.connected", data={"node_id": node_id})

    async def _handle_close(self, close: CloseInfo) -> bool:
        """Apply the §6 policy to one closure. Returns `True` when terminal."""
        if self._stopping.is_set():
            # Shutdown race: `stop()` closes the socket and *then* cancels this
            # task, so over a real relay the close handshake can land in between
            # and be classified here as an ordinary closure — writing a durable
            # `link.disconnected` with reason `"error"` plus a spurious
            # backoff transition for what is a clean shutdown. `stop()`'s own
            # `if self._online` block then correctly skips the duplicate.
            await self._go_offline("shutdown", close)
            return True

        self._last_close_code = close.code
        self._last_error_code = close.error_code
        # A session that lasted long enough to be *useful* clears the ladder;
        # one that dropped the instant it was accepted does not (that is the
        # flap, and climbing 1→60 s is the only thing that keeps it from
        # becoming a ~1 Hz redial loop forever).
        connected_monotonic, self._connected_monotonic = self._connected_monotonic, None
        if (
            connected_monotonic is not None
            and self._monotonic() - connected_monotonic >= BACKOFF_STABLE_S
        ):
            self._attempt = 0
        # The capability died with the connection; nothing may validate against
        # it while we are between sockets.
        self._capability = None
        self._connected_at = None
        close_class = classify_close(close)

        if close_class is CloseClass.AUTH_RETRY:
            self._auth_failures += 1
            if self._auth_failures >= AUTH_FAILURE_LIMIT:
                await self._go_terminal("auth_failed", close)
                return True
            # Spec §7: do not alarm on the first 4401 — it is as likely to be a
            # relay that started draining mid-handshake as a bad credential.
            logger.info(
                "The relay refused the tunnel credential (attempt %d/%d); "
                "retrying with freshly minted material",
                self._auth_failures,
                AUTH_FAILURE_LIMIT,
            )
            await self._go_offline("error", close)
            await self._backoff()
            return False

        terminal_reason = _TERMINAL_REASONS.get(close_class)
        if terminal_reason is not None:
            await self._go_terminal(terminal_reason, close)
            return True

        if close_class is CloseClass.DISPLACED:
            logger.info("The tunnel was displaced by a newer connection for this node")
            await self._go_offline("displaced", close)
            self._state = "displaced"
            self._retry_at = self._clock() + timedelta(seconds=DISPLACED_RETRY_DELAY_S)
            await self._sleep(DISPLACED_RETRY_DELAY_S)
            return False

        await self._go_offline("error", close)
        await self._backoff()
        return False

    async def _go_offline(self, reason: str, close: CloseInfo) -> None:
        """Record an online→offline edge exactly once."""
        if not self._online:
            return
        self._online = False
        # The mirror follows the link down. Ordered before the
        # durable row only so a consumer that reacts to `link.disconnected`
        # by reading `/capabilities` can never observe a stale `True`.
        self.clear_hosted_public_entitlement()
        # So does the token mirror: a dead link cannot be
        # re-minted for, and the cloud re-pushes on the next online edge.
        self.clear_github_tokens()
        await self._record_disconnected(reason, close)

    async def _go_terminal(self, reason: str, close: CloseInfo) -> None:
        """Stop reconnecting until the daemon is restarted or reconfigured."""
        logger.error("The node link is terminal: %s", reason)
        await self._go_offline(reason, close)
        self._state = "terminal"
        self._terminal_reason = reason
        self._retry_at = None

    async def _backoff(self) -> None:
        """Sleep the jittered exponential delay before the next dial."""
        self._attempt += 1
        # Cap the EXPONENT, not just the product: `2.0 ** 1024` raises
        # OverflowError, so an uninterrupted outage (~17 h at the capped
        # cadence) would turn a retryable condition terminal. 32 doublings
        # already exceed BACKOFF_MAX_S by orders of magnitude.
        delay = min(
            BACKOFF_INITIAL_S * (BACKOFF_FACTOR ** min(self._attempt - 1, 32)),
            BACKOFF_MAX_S,
        )
        delay *= 1.0 + (self._rng() * 2.0 - 1.0) * BACKOFF_JITTER
        self._state = "backoff"
        self._retry_at = self._clock() + timedelta(seconds=delay)
        await self._sleep(delay)

    # -- durable events ----------------------------------------------------

    async def _record_disconnected(self, reason: str, close: CloseInfo | None) -> None:
        await self._record(
            "link.disconnected",
            reason=reason,
            data={
                "close_code": close.code if close is not None else None,
                "error_code": close.error_code if close is not None else None,
            },
        )

    async def _record(
        self,
        event_type: str,
        *,
        reason: str | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        """Emit one durable row. Machine-shaped payloads only, never a token."""
        if self._events is None:
            return
        with suppress(Exception):  # the recorder is tolerant; be doubly so here
            await self._events.record(event_type, kind="link", reason=reason, data=data)


__all__ = [
    "AUTH_FAILURE_LIMIT",
    "BACKOFF_FACTOR",
    "BACKOFF_INITIAL_S",
    "BACKOFF_JITTER",
    "BACKOFF_MAX_S",
    "BACKOFF_STABLE_S",
    "DISCONNECT_REASONS",
    "DISPLACED_RETRY_DELAY_S",
    "ENTITLEMENT_MAX_FUTURE_SKEW_S",
    "ENTITLEMENT_TTL_S",
    "GITHUB_INSTALLATION_MIRROR_MAX",
    "GITHUB_TOKEN_WARN_LEAD_S",
    "MIN_RENEW_LEAD_S",
    "EntitlementUpdate",
    "GithubInstallation",
    "GithubInstallationSummary",
    "GithubTokenUpdate",
    "LinkManager",
    "LinkStatus",
]
