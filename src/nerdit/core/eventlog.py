"""Record durable domain events and publish best-effort wakeups.

Write the DB row before publishing to the lossy event bus. Recording failures
must never interrupt reconciliation. Payloads contain machine tokens, numbers,
booleans, and env key names only: authenticated readers and external webhook
targets must never receive free-text errors, container output, or secrets.

Consecutive identical `(type, service_name, reason, build_version)` keys within
`_COALESCE_WINDOW_S` are suppressed by a bounded in-memory map; restart may
re-emit them. Legacy `job.status_changed` remains bus-only.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nerdit.core.jobconfig import parse_job_config

if TYPE_CHECKING:
    from nerdit.core.events import EventBus
    from nerdit.db.queries import Queries
    from nerdit.db.rows import Job

logger = logging.getLogger(__name__)

#: The complete D-P24-3 emission vocabulary, pinned by test. Entries whose
#: emitter ships in a later sub-phase are declared here up front so the
#: vocabulary is reviewed once, as a whole, rather than grown ad hoc.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        # --- deploy generation (WP1) ---
        "service.deploy_started",
        "service.deploy_succeeded",
        "service.deploy_failed",
        # --- autonomous runtime transitions (WP1) — service.failed is THE
        # killer event: the 03:00 crash-loop settle nobody was ever told about.
        "service.healthy",
        "service.degraded",
        "service.restarting",
        "service.failed",
        "service.stopped",
        # --- health-gated cutover (P24b WP5, core/cutover.py) ---
        "service.cutover_started",
        "service.cutover_succeeded",
        "service.cutover_failed",
        # --- resource planes (WP1). `database.failed` has NO live firing site
        # in v1: its only emitter is the `DataProvisionError` branch, which is
        # reserved for the D-P15-1 dump/restore substrate and which the v1
        # readiness probes never raise (they raise the transient
        # `DataNotReadyError`, which emits nothing). Declared so nobody hunts
        # for the missing emitter.
        "model.ready",
        "model.failed",
        "database.ready",
        "database.failed",
        # --- managed-database dumps (P37 WP4, daemon/routes/databases.py) ---
        # ``database.dump_succeeded`` carries ``{dump, bytes}`` (a tar basename,
        # never served over HTTP), ``database.restore_succeeded`` ``{dump}``. The
        # two failure types carry a fixed ``reason`` token and nothing else
        # (D-P37-11): never the tool's tail, which prints row values and would
        # reach third-party webhook hosts; the tail lives on ``/diagnose`` only.
        "database.dump_succeeded",
        "database.dump_failed",
        "database.restore_succeeded",
        "database.restore_failed",
        # --- unattended redeploy poller (P24c WP10, core/gitwatch.py) ---
        "gitwatch.redeploy_triggered",
        "gitwatch.skipped_no_cutover",
        "gitwatch.poll_failed",
        "gitwatch.redeploy_failed",
        # A cloud nudge matched a git-sourced service that opted
        # OUT of `auto_deploy` (plan §6 Q2: record, don't act). `data` is
        # `{"to_sha"}` — the pushed head, a public commit id.
        "gitwatch.nudge_ignored",
        # --- node link (P27 WP-C1, core/link/manager.py) --- the remote-access
        # tunnel's session bookends. Emitted ONLY on an offline→online /
        # online→offline transition, so a successful in-place capability
        # renewal (every ~13 min at the default TTL) writes nothing: the
        # session spans the renewal. `reason` on the disconnect side is a
        # fixed machine-token set (D-P24-3, see manager.DISCONNECT_REASONS) —
        # error / displaced / revoked / auth_failed / protocol_unsupported /
        # entitlement_required / shutdown — and `data` carries the close
        # code + relay error code only, never the capability token.
        "link.connected",
        "link.disconnected",
        # The account entitlement mirror moved. Emitted on a value CHANGE
        # only — never on the cloud's periodic re-assert — with `data` =
        # `{"hosted_public_entitled": bool}` and nothing else. No account id,
        # no plan name, no header or token: this feed is POSTed to third-party
        # webhook hosts, and "may this node publish publicly" is the whole fact
        # a consumer needs.
        "link.entitlement",
        # --- hosted share (P26 WP-H, daemon/routes/share.py + service_purge.py)
        # --- the owner's exposure decisions. `share.ready`: a share was set
        # or changed — `data` carries `{access}` and, for a PRIVATE share
        # only, `url`. A private hosted URL is not a bearer capability (the
        # cloud edge still demands an owner session), while a PUBLIC one is
        # world-reachable and this feed is POSTed to third-party webhook hosts,
        # so the public URL is omitted. `share.removed`: the share was deleted
        # (`reason` = `unshared` | `service_deleted`). Never the capability
        # token, never a navigation ticket.
        "share.ready",
        "share.removed",
        # --- custom domains (P26 WP1, daemon/routes/domains.py +
        # --- service_purge.py) --- the owner's OTHER exposure decision.
        # `domain.added`: a name was bound — `data` carries
        # `{domain, acme}`. `domain.removed`: it was released (`reason` =
        # `removed` | `service_deleted`), `data` carries `{domain}`.
        # The domain name travels (S-W9): unlike a hosted URL it is not a
        # bearer capability at all — it is public DNS the operator chose to
        # point at this node, and a webhook consumer cannot act on
        # "a domain was bound" without knowing which one.
        "domain.added",
        "domain.removed",
        # --- offline product license (P17d WP-D1, core/license.py) --- the
        # admin-action and boot edges of the installed entitlement document.
        # Payloads are machine tokens ONLY: `license.installed` carries
        # `{lid, plan, state}`, `license.removed` `{lid}`,
        # `license.rejected` `{reason}` (an INVALID_REASONS member). Never
        # the blob and never `customer_id` — this feed is POSTed to
        # third-party webhook hosts. There are deliberately NO temporal events
        # (`license.expiring`): v1 has no license poller, doctor owns the nag,
        # and a timer loop for a once-a-year edge is not worth the moving part.
        # `license.rejected` fires at boot on a present-but-invalid file; an
        # interactive install refusal is a 422 + audit denial, because the
        # unattended case is the one webhooks need.
        "license.installed",
        "license.removed",
        "license.rejected",
        # --- lifespan bookends: they bound a consumer's resume gap ---
        "daemon.started",
        "daemon.stopping",
    }
)

#: Consecutive identical `(type, service_name, reason, build_version)` keys
#: inside this many seconds collapse to one row.
_COALESCE_WINDOW_S = 60.0

#: The coalescing key. `build_version` is part of it so the flap guard cannot
#: swallow a *distinct deploy generation* settling inside the window; same-
#: generation flaps still share a key because `record_job_event` projects
#: `build_version` straight off the row.
_CoalesceKey = tuple[str, str | None, str | None, int | None]

#: Bound on the coalescing map. Oldest key evicted first — an unbounded map
#: would grow with every distinct service name the daemon ever saw.
_MAX_RECENT = 512

#: Types exempt from coalescing: session EDGES whose every occurrence is the
#: signal. All `link.connected` rows share one key (no service_name, no
#: reason, no build_version), so the flap guard would swallow the reconnect in
#: a connect → error → reconnect cycle inside one window and a consumer could
#: never reconstruct the actual session state (P27 WP-C1 PR #114 review). The
#: manager already rate-bounds emission structurally: an edge requires a real
#: offline↔online transition, and reconnects sit behind the backoff ladder.
#:
#: The three `license.*` types join for the same structural reason (P17d
#: D-LIC6): they are admin-action / boot edges whose EVERY occurrence is the
#: signal, and they share a coalescing key (no service_name, no build_version —
#: and `license.rejected` repeats one `reason`), so an install → remove →
#: install sequence inside one window would otherwise collapse to a single row
#: and leave a consumer unable to reconstruct what the operator did.
#:
#: **Both** `share.*` types join for the third instance of the same shape
#: : each is an owner-action EDGE whose every occurrence is the
#: signal, and all rows of one type for one service share a coalescing key
#: (`reason` is fixed per emitter and there is no `build_version`), so a
#: share → unshare → share sequence inside one window would collapse and leave a
#: consumer believing the app is still exposed — or, worse, still PRIVATE.
#:
#: **Both** `domain.*` types join for the same reason as `share.*` (P26
#: WP1): all rows of one type for one service share a coalescing key (`reason`
#: is fixed per emitter, `data` is not part of the key and there is no
#: `build_version`), so an add → remove → add of the same name inside one
#: window — or, worse, adds of two DIFFERENT names — would collapse to a single
#: row and leave a consumer with a wrong picture of what the node now answers
#: for.
#:
#: `share.ready` needs the exemption even though it reads like a steady-state
#: signal, because the fact it carries lives in `data` and `data` is *not*
#: part of the coalescing key (PR review, P26 WP-H): a private→public flip
#: inside the window is a different fact under an identical key, and dropping it
#: leaves the feed saying "private" about a world-reachable app. Keying on
#: `reason=access` instead would fix that one case and still swallow an
#: unshare→re-share of the same access, so the honest fix is the frozenset.
_NEVER_COALESCED = frozenset(
    {
        "link.connected",
        "link.disconnected",
        # An entitlement EDGE, for the `link.*` reason above: every row
        # is already a change (the route suppresses re-asserts), all rows share
        # one coalescing key, and a revoke → restore inside one window would
        # otherwise collapse and leave a consumer believing the wrong thing
        # about a world-reachable app.
        "link.entitlement",
        "license.installed",
        "license.removed",
        "license.rejected",
        "share.ready",
        "share.removed",
        "domain.added",
        "domain.removed",
        # (P37 D-P37-11) The four dump types, for the ``share.*``/``domain.*``
        # reason: each occurrence is the signal, and coalescing would collapse
        # two dumps (or two restores of different tars) into one row.
        "database.dump_succeeded",
        "database.dump_failed",
        "database.restore_succeeded",
        "database.restore_failed",
    }
)


class EventRecorder:
    """The single writer of the durable feed.

    `queries` is the persistence handle; `bus` is the optional in-process
    fan-out the dashboard live view and the P24c dispatcher both read. Neither
    failure mode propagates: `record` catches everything and returns.
    """

    def __init__(
        self,
        queries: Queries,
        bus: EventBus | None = None,
        *,
        coalesce_window_s: float = _COALESCE_WINDOW_S,
    ) -> None:
        self._queries = queries
        self._bus = bus
        self._window = coalesce_window_s
        # Insertion-ordered so eviction is O(1) from the oldest end. Values are
        # `time.monotonic()` samples — never wall clock, which a clock step
        # could move backwards and wedge the window open.
        self._recent: OrderedDict[_CoalesceKey, float] = OrderedDict()

    async def record(
        self,
        type: str,  # noqa: A002 — the LOCKED D-P24-3 field name
        *,
        kind: str | None = None,
        service_name: str | None = None,
        reason: str | None = None,
        build_version: int | None = None,
        data: dict[str, object] | None = None,
    ) -> None:
        """Persist one feed row, then tee it to the bus. Never raises.

        `type` must be in `EVENT_TYPES`; an unknown value is refused
        (logged, nothing written) rather than silently minting a vocabulary a
        consumer's filter can never match.
        """
        try:
            if type not in EVENT_TYPES:
                logger.error("Refusing to record unknown event type %r", type)
                return
            key = (type, service_name, reason, build_version)
            if type not in _NEVER_COALESCED and self._coalesce_suppressed(key):
                return
            # ONE instant per event, minted here and used for BOTH the stored
            # row and the bus frame. Two clock reads (the column default plus a
            # `now()` for the frame) made a live frame and its own replay
            # describe the same event at two different times, which is exactly
            # what a consumer reconciling across a reconnect must not see.
            # Truncated to whole seconds because that is the resolution the
            # stored column has, so the two shapes are the same instant:
            # `2026-08-07 10:00:00` on disk, `2026-08-07T10:00:00+00:00` on
            # the wire. `isoformat(sep=" ")` on the naive UTC value IS the
            # `datetime('now')` space format — no second format literal.
            now = datetime.now(UTC).replace(microsecond=0)
            row_id = await self._queries.insert_event(
                type=type,
                kind=kind,
                service_name=service_name,
                reason=reason,
                build_version=build_version,
                data=data,
                ts=now.replace(tzinfo=None).isoformat(sep=" "),
            )
            if row_id is None:
                return
            # The window is stamped only once the row EXISTS. Reserving the key
            # before the insert meant a transient write failure (a locked DB, a
            # dropped connection) silently suppressed every retry of that same
            # event for the rest of the window — the flap guard swallowing the
            # very transition it was meant to report once.
            if type not in _NEVER_COALESCED:
                self._coalesce_commit(key)
            if self._bus is None:
                return
            # The bus payload carries the full row shape INCLUDING `id`: the
            # WP2 SSE resume leg dedups a live frame against its replay window
            # by id, so an id-less frame would be undeduplicable.
            self._bus.publish(
                {
                    "type": type,
                    "id": row_id,
                    "ts": now.isoformat(),
                    "kind": kind,
                    "service_name": service_name,
                    "reason": reason,
                    "build_version": build_version,
                    "data": data,
                }
            )
        except Exception:
            # D-P24-2: a feed write must never abort a reconcile tick.
            logger.warning("Durable event write failed (%s); continuing", type, exc_info=True)

    def _coalesce_suppressed(self, key: _CoalesceKey) -> bool:
        """Whether this key was already **written** inside the window. Read-only.

        Deliberately stamps nothing: the window opens in
        `_coalesce_commit`, after the row is on disk. The cost of that
        split is a race — two concurrent identical records can both read "not
        suppressed" and both insert — and that is the acceptable side of the
        trade. The feed is at-least-once by contract (a consumer already
        dedupes by row id), so a duplicate row is a visible, recoverable
        annoyance, whereas an event swallowed because a write happened to fail
        is invisible and unrecoverable.
        """
        last = self._recent.get(key)
        return last is not None and (time.monotonic() - last) < self._window

    def _coalesce_commit(self, key: _CoalesceKey) -> None:
        """Open the coalescing window for `key`; evict past `_MAX_RECENT`."""
        self._recent[key] = time.monotonic()
        self._recent.move_to_end(key)
        while len(self._recent) > _MAX_RECENT:
            self._recent.popitem(last=False)


#: Process-wide recorder, set at daemon startup. It lets deep call sites that
#: no controller owns — notably `core/deploy_state.py::stamp_last_deploy`,
#: reached from
#: `core/app_build.py` and `core/launch.py` as well as the controllers —
#: emit without threading a recorder through every signature. `None` (the
#: default, and every unit test that never boots a daemon) makes every emit
#: inert.
_recorder: EventRecorder | None = None


def set_recorder(recorder: EventRecorder | None) -> None:
    """Install (or clear, with `None`) the process-wide recorder."""
    global _recorder
    _recorder = recorder


def get_recorder() -> EventRecorder | None:
    """The process-wide recorder, or `None` outside a running daemon."""
    return _recorder


async def record_job_event(
    recorder: EventRecorder | None,
    type: str,  # noqa: A002 — the LOCKED D-P24-3 field name
    job: Job,
    *,
    reason: str | None = None,
    data: dict[str, object] | None = None,
) -> None:
    """Record a feed row derived from a workload row. No-ops without a recorder.

    Projects `kind`/`service_name`/`build_version` off the row so every
    controller call site stays a one-liner and cannot disagree about the shape.
    Every caller passes a real `nerdit.db.models.Job`, so the projection
    is plain attribute access — the tolerance that matters lives in
    `EventRecorder.record`, which swallows everything the I/O can raise.
    """
    if recorder is None:
        return
    version = parse_job_config(job).get("build_version")
    await recorder.record(
        type,
        kind=job.kind.value if job.kind else None,
        service_name=job.service_name,
        reason=reason,
        build_version=version if isinstance(version, int) else None,
        data=data,
    )
