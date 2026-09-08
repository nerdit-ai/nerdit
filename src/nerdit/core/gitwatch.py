"""Poll opted-in Git services and redeploy changed refs through the local primitive.

Only cutover-eligible services may auto-deploy; others emit a machine-readable
skip reason. Reconcile schedules semaphore-bounded tasks without blocking the
tick on network I/O. Poll failures emit codes, never stderr, and back off to
30 minutes. Non-409 redeploy failures pin the failed SHA until the ref changes.

Polling resolves secret references without access audit noise; actual redeploys
retain audited resolution. `${github.installation}` uses the injected repo-token
lookup. Missing or expired installation tokens back off quietly, including
`deploy.github_token_absent` during redeploy, without pinning the SHA.

Cloud nudges run the same guarded poll immediately for matching services.
Their SHA is a hint, never the checkout target. Bounded `(service, sha)` dedupe
suppresses webhook redelivery; non-auto-deploy matches only record an ignored
event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nerdit.config.settings import GitSettings
from nerdit.core.bindings.secretref import walk_secret_ref
from nerdit.core.eventlog import record_job_event
from nerdit.core.gitsource import (
    GITHUB_INSTALLATION_REF,
    GitSourceError,
    canonical_repo,
    github_repo_slug,
    installation_token_allowed_for_host,
    ls_remote_head,
)
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError, SecretManager

if TYPE_CHECKING:
    from nerdit.core.eventlog import EventRecorder
    from nerdit.db.queries import Queries
    from nerdit.db.rows import Job

logger = logging.getLogger(__name__)

#: Ceiling on the exponential poll backoff.
_POLL_BACKOFF_CAP_S = 1800.0
#: A permanently ineligible service must not fill the feed: at most one
#: `skipped_no_cutover` per `(service, reason)` per hour, on top of the
#: recorder's own 60 s coalescing.
_SKIP_EMIT_INTERVAL_S = 3600.0
#: Concurrent remote probes, the `max_concurrent_builds` posture.
_MAX_CONCURRENT_POLLS = 4
#: Bound on the doubling exponent so a long-dead target cannot compute a
#: pathologically large intermediate before the cap clamps it.
_MAX_BACKOFF_EXPONENT = 10
#: Only a live service is worth polling: a stopped or failed row has nothing to
#: cut over TO.
_POLLABLE_STATUSES = frozenset({"running", "degraded"})
#: The one redeploy refusal that means "credential not here YET", not "bad
#: commit": backed off like a 409, never pinned, never a durable event (D-GH-9).
_GITHUB_TOKEN_ABSENT = (422, "deploy.github_token_absent")
#: How many `(service, sha)` nudge pairs are remembered for the second-layer
#: dedupe (D-GH-4). Oldest-out: a webhook redelivery lands within minutes, so
#: a few hundred entries outlast any realistic replay window by orders of
#: magnitude while bounding memory on a node that is nudged for years.
_NUDGE_DEDUPE_MAX = 512
#: The host a nudge's `owner/name` lives on (default). Nudges come from
#: GitHub's push webhook (D-GH-C3); a source on any other allow-listed host
#: can never match. The effective host follows `[git].github_clone_base_url`
#: (the D-GH-3 pin), so the cloud dev stack's git fixture can play GitHub in
#: `make e2e-real`.
_NUDGE_HOST = "github.com"
_REFS_HEADS = "refs/heads/"


@dataclass(frozen=True)
class NudgeResult:
    """What one nudge did, by service name (the `202` body, D-GH-4).

    `matched`: `auto_deploy` services whose immediate poll was started;
    `ignored`: matched the source but opted out of unattended redeploy
    (recorded as `gitwatch.nudge_ignored`); `deduped`: already nudged for
    this exact sha, nothing re-fired.
    """

    matched: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    deduped: list[str] = field(default_factory=list)


def normalise_nudge_ref(ref: str) -> str:
    """`refs/heads/main` → `main`; a short name passes through unchanged."""
    ref = ref.strip()
    return ref[len(_REFS_HEADS) :] if ref.startswith(_REFS_HEADS) else ref


def normalise_nudge_repo(repo: str, github_host: str = _NUDGE_HOST) -> str:
    """A nudge's `owner/name` as the D-GH-6 canonical `host/owner/name`.

    Lower-cased, `.git` stripped, a host prefix accepted if already present
    (`github.com/owner/name`) and added otherwise — so the result compares
    equal to what `nerdit.core.gitsource.canonical_repo` recorded for
    a `https://github.com/Owner/Name.git` deploy. `github_host` is the
    daemon's `[git].github_host` (D-GH-3): the one host GitHub identities
    live on for this daemon.
    """
    repo = repo.strip().strip("/").lower().removesuffix(".git")
    if repo.count("/") == 1:
        return f"{github_host.strip().lower()}/{repo}"
    return repo


class GitWatchController:
    """Polls each `auto_deploy` git-sourced service and redeploys on SHA drift."""

    def __init__(  # noqa: PLR0913 - one seam per collaborator, all keyword-only
        self,
        queries: Queries,
        git_settings: GitSettings,
        secrets: SecretManager,
        *,
        redeploy: Callable[[Job], Awaitable[None]],
        skip_reason: Callable[[Job, dict], str | None],
        events: EventRecorder | None = None,
        github_token: Callable[[str], str | None] | None = None,
    ) -> None:
        self._queries = queries
        self._git = git_settings
        self._secrets = secrets
        self._redeploy = redeploy
        # `owner/name` → live installation token, or `None` (P33 D-GH-3).
        # Absent (no link manager wired) reads as "always absent" — the no-cloud
        # daemon never consults anything for a `${secrets.*}` ref.
        self._github_token = github_token
        # `None` means eligible; anything else is the pinned reason.
        self._skip_reason = skip_reason
        self._events = events
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT_POLLS)
        self._inflight: dict[str, asyncio.Task] = {}
        # Set at the top of `shutdown()` BEFORE `_inflight` is cleared: a
        # poll that is past its last await runs its `finally` with
        # `cancelled=False` and would otherwise register a fresh follow-up
        # task that `shutdown()` has already stopped waiting for (D5). The
        # finally re-spawn is guarded on `not self._stopping`.
        self._stopping = False
        # Every deadline below is `time.monotonic()`-based: a wall-clock jump
        # (NTP, suspend/resume) must not turn into a poll storm.
        self._next_due: dict[str, float] = {}
        self._failures: dict[str, int] = {}
        # SHA that permanently failed to redeploy, per service: one durable
        # signal per bad commit, never a re-trigger until the remote moves or a
        # manual redeploy advances `config['source'].commit_sha`. In-memory by
        # design — a daemon restart re-fires at most one trigger/fail pair.
        self._failed_sha: dict[str, str] = {}
        self._skip_last: dict[tuple[str, str], float] = {}
        # `(service, sha)` pairs already nudged — the D-GH-4 second-layer
        # dedupe behind the route's `Idempotency-Key`. Insertion-ordered and
        # bounded (oldest out) so a redelivered webhook with a fresh delivery
        # id still fires at most once per commit.
        self._nudged: OrderedDict[tuple[str, str], None] = OrderedDict()
        # `service → sha` a nudge asked for but whose poll has not yet
        # deployed or confirmed it (D7). The commit into `_nudged` happens
        # only on that success (`_commit_nudge`), so a nudge that could not
        # act yet — the installation token not mirrored on a cold reconnect, a
        # quiet backoff — is retried by a later nudge or tick, never burned by
        # a dedupe entry laid down before the sha was ever handled.
        self._pending_nudge: dict[str, str] = {}
        # Services nudged WHILE a poll of theirs was in flight. That poll may
        # have probed the remote before the push landed, and every exit of
        # `_poll_once` re-arms a full interval out — so the running task
        # re-spawns itself once when it finishes (see `_poll`) instead of
        # the nudge being answered "matched" and then waiting for the next
        # regular tick.
        self._renudge: set[str] = set()
        # Deadline for the next candidate scan: the DB read is worth one visit
        # per poll interval, not one per 2 s tick.
        self._scan_due: float = 0.0

    # --- reconcile ------------------------------------------------------------

    async def reconcile(self) -> None:
        """Spawn a poll task for every candidate that is due. Never blocks."""
        for name, task in list(self._inflight.items()):
            if task.done():
                self._inflight.pop(name, None)

        # The candidate scan is a full workload read: it belongs on the poll
        # cadence, not on the shared 2 s tick.
        now = time.monotonic()
        if now < self._scan_due:
            return

        rows = await self._queries.list_workload_configs()
        candidates = {str(row["service_name"]): row for row in rows if _is_candidate(row)}
        self._prune(candidates.keys())

        for name, row in candidates.items():
            if name in self._inflight:
                continue
            if now < self._next_due.get(name, 0.0):
                continue
            self._inflight[name] = asyncio.create_task(self._poll(row, name))

        # Next scan: the earliest still-future poll deadline, bounded above by
        # one plain interval so a newly-enabled `auto_deploy` service — or a
        # row that just became pollable — is discovered within ~watch_interval_s.
        # Deadlines armed by the tasks spawned above always land at or beyond
        # that horizon; the `min` only catches earlier scans' deadlines.
        horizon = now + self._git.watch_interval_s
        pending = [due for due in self._next_due.values() if due > now]
        self._scan_due = min([horizon, *pending])

    def _prune(self, live: Iterable[str]) -> None:
        """Drop per-service state for rows that are no longer candidates."""
        names = set(live)
        for tracked in (self._next_due, self._failures, self._failed_sha, self._pending_nudge):
            for name in [n for n in tracked if n not in names]:
                tracked.pop(name, None)
        self._renudge.intersection_update(names)
        for key in [k for k in self._skip_last if k[0] not in names]:
            self._skip_last.pop(key, None)

    async def shutdown(self) -> None:
        """Cancel every in-flight poll and wait for it to unwind."""
        # Set BEFORE the clear so a poll finishing its `finally` after
        # shutdown began (cancelled=False, past its last await) sees it and
        # refuses to register a follow-up task we would then orphan (D5).
        self._stopping = True
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._inflight.clear()

    # --- nudge (P33 D-GH-4) ---------------------------------------------------

    async def nudge(self, repo: str, ref: str, sha: str) -> NudgeResult:
        """Make the poll immediate for every service sourced from `(repo, ref)`.

        `repo` is GitHub's `owner/name` (any case, `.git` tolerated),
        `ref` a short branch name or `refs/heads/<name>`, `sha` the
        pushed head — a HINT only: the poll it triggers resolves the recorded
        ref itself and records whatever it finds (D-GH-4), so a branch that
        moved on between webhook and clone deploys its newer head. The match
        is on the D-GH-6 canonical identity (`config['source'].repo`, with a
        fallback derivation from `repo_url` for rows recorded before it
        existed) plus the recorded ref; a source recorded without a ref (the
        default branch) is never matched — the daemon does not know which
        branch that is. Never raises; a service it cannot touch is simply not
        in the result.
        """
        want_repo = normalise_nudge_repo(repo, self._git.github_host)
        want_ref = normalise_nudge_ref(ref)
        result = NudgeResult()
        rows = await self._queries.list_workload_configs()
        for row in rows:
            name = row.get("service_name")
            if not name or not _source_matches(row.get("config"), want_repo, want_ref):
                continue
            name = str(name)
            if str(row.get("status")) not in _POLLABLE_STATUSES:
                # Nothing live to cut over to — the same rule the tick applies.
                continue
            cfg = row.get("config")
            if not (isinstance(cfg, dict) and cfg.get("auto_deploy") is True):
                await self._nudge_ignored(str(row["id"]), name, sha)
                result.ignored.append(name)
                continue
            if (name, sha) in self._nudged:
                result.deduped.append(name)
                continue
            # Tentative only: the dedupe is finalized when a poll actually
            # deploys or confirms this sha (`_commit_nudge`), never here —
            # a nudge that cannot act yet (token not mirrored, quiet backoff)
            # must stay retryable (D7).
            self._pending_nudge[name] = sha
            # Fresh information beats a backoff armed on stale failures: a
            # push is the cloud telling us the remote moved, so the next probe
            # is due now — and spawned now, rather than on the next 2 s tick.
            self._failures.pop(name, None)
            self._next_due[name] = 0.0
            self._scan_due = 0.0
            if name in self._inflight:
                # The running poll owns the follow-up: it re-spawns once on
                # exit, so a push that landed after its probe still deploys
                # now rather than one interval (or one backoff) later.
                self._renudge.add(name)
            else:
                self._inflight[name] = asyncio.create_task(self._poll(row, name))
            result.matched.append(name)
        return result

    def _remember_nudge(self, name: str, sha: str) -> None:
        self._nudged[(name, sha)] = None
        while len(self._nudged) > _NUDGE_DEDUPE_MAX:
            self._nudged.popitem(last=False)

    def _commit_nudge(self, name: str) -> None:
        """Finalize a pending nudge's dedupe once its poll handled the sha (D7).

        Called only on the two success terminals of a poll — the remote head
        already deployed, or a fresh redeploy that shipped it. A poll that
        could not act leaves the pending entry untouched, so a later nudge or
        tick retries the sha instead of finding it burned.
        """
        sha = self._pending_nudge.pop(name, None)
        if sha is not None:
            self._remember_nudge(name, sha)

    async def _nudge_ignored(self, job_id: str, name: str, sha: str) -> None:
        """Record that a push reached a service that opted out (plan §6 Q2)."""
        job = await self._queries.get_job(job_id)
        if job is None:
            return
        await record_job_event(self._events, "gitwatch.nudge_ignored", job, data={"to_sha": sha})

    # --- one poll -------------------------------------------------------------

    async def _poll(self, row: dict, name: str) -> None:
        """Task body: bounded by the semaphore, never lets an exception escape.

        A nudge that arrived while this poll was running (`_renudge`) is
        honoured on exit by spawning exactly one follow-up poll — after the
        `_schedule`/`_backoff` of the finished run, so the follow-up's
        `_next_due = 0` is the deadline that survives. Never on cancellation:
        `shutdown` must not leave a freshly spawned task behind.
        """
        cancelled = False
        try:
            async with self._sem:
                await self._poll_once(row, name)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            logger.exception("GitWatch poll failed for service '%s'", name)
            self._backoff(name)
        finally:
            self._inflight.pop(name, None)
            pending = name in self._renudge
            self._renudge.discard(name)
            if pending and not cancelled and not self._stopping:
                self._failures.pop(name, None)
                self._next_due[name] = 0.0
                self._scan_due = 0.0
                self._inflight[name] = asyncio.create_task(self._poll(row, name))

    async def _poll_once(self, row: dict, name: str) -> None:
        # Two gates, both on a FRESH read, never on the tick-time snapshot: a
        # task can sit behind the semaphore for a whole `clone_timeout_s`, and
        # the probe itself is a network round trip. `row` stays only as the
        # tick-time candidate handle (id + name).
        #
        # Gate 1 (pre-probe): candidacy + cutover eligibility. An opt-out
        # (`auto_deploy = false`), a stop/delete or a source rewrite landing
        # in that window must cost zero egress.
        fresh = await self._refresh(str(row["id"]))
        if fresh is None:
            # No longer a candidate; re-arm so a revived service is not polled
            # instantly on every subsequent tick.
            self._schedule(name)
            return
        job, cfg, source = fresh

        # The cutover gate runs BEFORE any remote touch: an ineligible service
        # is not going to be redeployed whatever the remote says, so probing it
        # would be pure egress for nothing.
        reason = self._skip_reason(job, cfg)
        if reason is not None:
            await self._emit_skip(job, name, reason)
            self._schedule(name)
            return

        token, usable = await self._credential(job, name, source)
        if not usable:
            return

        probed = (source.get("repo_url"), source.get("ref"))
        try:
            sha = await ls_remote_head(
                str(source.get("repo_url") or ""),
                ref=source.get("ref"),
                token=token,
                timeout_s=float(self._git.clone_timeout_s),
                allowed_hosts=self._git.allowed_hosts,
            )
        except GitSourceError as exc:
            await self._poll_failed(job, name, exc.code)
            return

        final = await self._post_probe_gate(job.id, name, probed)
        if final is None:
            return
        job, cfg, source = final

        if sha == source.get("commit_sha"):
            self._failures.pop(name, None)
            self._commit_nudge(name)
            self._schedule(name)
            return

        if sha == self._failed_sha.get(name):
            # This exact commit already failed to redeploy and was reported
            # once. Keep polling at the plain interval so the fix commit is
            # picked up promptly, but never re-trigger the same SHA.
            self._failures.pop(name, None)
            self._schedule(name)
            return

        # Record BEFORE the redeploy (at-least-once): a daemon that dies mid
        # redeploy must still have said that it decided to fire.
        await self._audit_redeploy(job, name, source, sha)
        await record_job_event(
            self._events,
            "gitwatch.redeploy_triggered",
            job,
            data={"from_sha": source.get("commit_sha"), "to_sha": sha},
        )
        try:
            await self._redeploy(job)
        except Exception as exc:
            code = getattr(exc, "code", None)
            status = getattr(exc, "status_code", None)
            if isinstance(code, str) and isinstance(status, int):
                # A machine-coded failure — `GitSourceError` or the daemon's
                # `NerditError`, matched structurally on the shared
                # `(status_code, code)` pair: `core` never imports
                # `daemon`, so the daemon error type cannot be named here.
                await self._redeploy_failed(job, name, sha, status, code)
                return
            # Unclassified (a bug, a cancelled shim): back off and retry — never
            # pin on an exception that carries no machine code.
            logger.exception("Auto-redeploy failed for service '%s'", name)
            self._backoff(name)
            return
        self._failures.pop(name, None)
        self._failed_sha.pop(name, None)
        self._commit_nudge(name)
        self._schedule(name)

    async def _post_probe_gate(
        self, job_id: str, name: str, probed: tuple[object, object]
    ) -> tuple[Job, dict, dict] | None:
        """Gate 2 (post-probe): re-read once more before acting.

        A manual redeploy landing during the probe already shipped the SHA we
        are about to "discover", so comparing against the pre-probe snapshot
        would fire a duplicate auto-redeploy of the identical commit; an
        opt-out or a `cutover = false` write landing there must likewise win
        over a decision taken before it. And when the write rewrote the SOURCE
        itself (`probed` is the pre-probe `(repo_url, ref)`), the probed
        SHA belongs to a repo/ref this row no longer points at — comparing it
        against the new source's `commit_sha` would trigger a redeploy off a
        foreign probe and stamp a `to_sha` that repo never had. `None`
        means "do not act"; the poll is already re-armed.
        """
        final = await self._refresh(job_id)
        if final is None:
            self._schedule(name)
            return None
        job, cfg, source = final
        reason = self._skip_reason(job, cfg)
        if reason is not None:
            await self._emit_skip(job, name, reason)
            self._schedule(name)
            return None
        if (source.get("repo_url"), source.get("ref")) != probed:
            self._schedule(name)
            return None
        return final

    async def _refresh(self, job_id: str) -> tuple[Job, dict, dict] | None:
        """Re-read the row and re-apply the candidacy gate.

        `None` means "not a candidate any more" — deleted, no longer live, or
        opted out of `auto_deploy` / no longer git-sourced. The caller re-arms
        and returns; it never probes, audits or redeploys on a stale snapshot.
        Returns `(job, config, source)` — `source` is a dict by the gate.
        """
        job = await self._queries.get_job(job_id)
        if job is None:
            return None
        cfg = parse_job_config(job)
        if not _is_candidate_state(job.status.value, cfg):
            return None
        source = cfg.get("source")
        return job, cfg, source if isinstance(source, dict) else {}

    async def _redeploy_failed(self, job: Job, name: str, sha: str, status: int, code: str) -> None:
        """Classify a post-trigger redeploy failure: contention, or a bad commit.

        The poll itself succeeded, so this is never a `poll_failed`. A **409**
        is contention (an active run/release, a cutover in flight) already
        recorded by its own machinery: back off and retry the *same* SHA —
        pinning there would permanently drop a good commit that merely landed
        during a run. Anything else is permanent for this commit.
        """
        if status == 409 or (status, code) == _GITHUB_TOKEN_ABSENT:
            logger.info("Auto-redeploy for '%s' refused (%s); backing off", name, code)
            self._backoff(name)
            return
        # Permanent for THIS commit (no buildpack, clone size cap, a removed
        # allowlist host, ...): one machine-shaped durable signal, then suppress
        # the SHA until it moves. The plain interval, not a backoff — the fix
        # commit must be picked up promptly.
        logger.warning("Auto-redeploy for '%s' failed (%s); suppressing SHA", name, code)
        self._failed_sha[name] = sha
        await record_job_event(
            self._events,
            "gitwatch.redeploy_failed",
            job,
            reason=code,
            data={"to_sha": sha},
        )
        self._schedule(name)

    # --- credentials ----------------------------------------------------------

    def _resolve_token(self, name: str, token_ref: str) -> str | None:
        """Resolve a recorded `token_ref`, quietly (D-BP-3 — no audit row).

        Same precedence as the request-bound resolver (per-service scope, then
        the shared scope only when the reference names it), minus the
        `secret.shared_referenced` row: a poll repeats every minute and is not
        an access event. `None` means "no such stored secret" — the caller
        turns that into a `deploy.no_source_credential` poll failure. The
        value lives in a local and never reaches a log, an event or argv.
        """
        res = walk_secret_ref(
            token_ref,
            service_env=lambda: self._secrets.load(name),
            shared_env=lambda: self._secrets.load(SHARED_SCOPE),
        )
        return res.value

    async def _credential(self, job: Job, name: str, source: dict) -> tuple[str | None, bool]:
        """`(token, usable)` for the recorded `token_ref`; the poll is
        already re-armed when `usable` is `False`.

        Two absences, two postures: a `${secrets.*}` reference that does not
        resolve is a real `deploy.no_source_credential` poll failure (the
        operator recorded something that is not there), whereas a
        `${github.installation}` that resolves to nothing is transient by
        construction (D-GH-9) — backoff only, no event, no audit row.
        """
        token_ref = source.get("token_ref")
        if token_ref is None:
            return None, True
        if token_ref == GITHUB_INSTALLATION_REF:
            token = self._resolve_github_token(source)
            if token is None:
                # D-GH-9: quiet. No event, no audit row — the doctor row says it.
                logger.debug(
                    "GitWatch: no installation token for service '%s' yet; backing off", name
                )
                self._backoff(name)
                return None, False
            return token, True
        try:
            token = self._resolve_token(name, str(token_ref))
        except SecretDecryptError:
            await self._poll_failed(job, name, "secret.decrypt_failed")
            return None, False
        if token is None:
            await self._poll_failed(job, name, "deploy.no_source_credential")
            return None, False
        return token, True

    def _resolve_github_token(self, source: dict) -> str | None:
        """Resolve `${github.installation}` by repo (P33 D-GH-3), quietly.

        The recorded `repo_url` is reduced to its `owner/name` (GitHub
        only) and looked up in the cloud-pushed mirror. `None` for a missing
        resolver, a non-GitHub host, a repo outside every installation's list
        or an expired token — all of which the caller treats as transient.

        The same host gate as `resolve_github_installation_token` (security
        review F2 / D9): a configured `[git].github_host` other than
        `github.com` offers nothing unless the `NERDIT_DEV_GITHUB_CLONE_BASE`
        env guard is set — the poller is an unattended, repeating egress, so it
        must not hand the cloud credential to a host a clone could not.
        """
        if self._github_token is None:
            return None
        if not installation_token_allowed_for_host(self._git.github_host):
            return None
        slug = github_repo_slug(str(source.get("repo_url") or ""), self._git.github_host)
        if slug is None:
            return None
        return self._github_token(slug)

    # --- records --------------------------------------------------------------

    async def _audit_redeploy(self, job: Job, name: str, source: dict, sha: str) -> None:
        """One `deploy.auto_redeploy` row attributed to `system`.

        Best-effort (the `CutoverManager._audit` posture): failing to audit
        must not stop the redeploy the operator asked for. Commit SHAs and the
        ref are not secrets — the repo URL is deliberately absent, it is already
        on the row.
        """
        try:
            await self._queries.insert_audit_log(
                action="deploy.auto_redeploy",
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=job.id,
                params_redacted=json.dumps(
                    {
                        "service": name,
                        "ref": source.get("ref"),
                        "from_sha": source.get("commit_sha"),
                        "to_sha": sha,
                    },
                    sort_keys=True,
                ),
            )
        except Exception:
            logger.warning("Failed to audit deploy.auto_redeploy for %s", name, exc_info=True)

    async def _emit_skip(self, job: Job, name: str, reason: str) -> None:
        """Emit `gitwatch.skipped_no_cutover`, at most hourly per (service, reason)."""
        now = time.monotonic()
        last = self._skip_last.get((name, reason))
        if last is not None and now - last < _SKIP_EMIT_INTERVAL_S:
            return
        self._skip_last[(name, reason)] = now
        await record_job_event(self._events, "gitwatch.skipped_no_cutover", job, reason=reason)

    async def _poll_failed(self, job: Job, name: str, code: str) -> None:
        """Emit `gitwatch.poll_failed` with the machine code, then back off."""
        await record_job_event(self._events, "gitwatch.poll_failed", job, reason=code)
        self._backoff(name)

    # --- scheduling -----------------------------------------------------------

    def _schedule(self, name: str) -> None:
        """Arm the next poll one plain interval out."""
        self._next_due[name] = time.monotonic() + self._git.watch_interval_s

    def _backoff(self, name: str) -> None:
        """Arm the next poll one doubled (capped) interval out."""
        failures = self._failures.get(name, 0) + 1
        self._failures[name] = failures
        delay = self._git.watch_interval_s * (2 ** min(failures, _MAX_BACKOFF_EXPONENT))
        self._next_due[name] = time.monotonic() + min(delay, _POLL_BACKOFF_CAP_S)


def _source_matches(cfg: object, want_repo: str, want_ref: str) -> bool:
    """Whether a row's git source is `(want_repo, want_ref)` canonically."""
    if not isinstance(cfg, dict):
        return False
    source = cfg.get("source")
    if not isinstance(source, dict) or source.get("type") != "git":
        return False
    recorded = source.get("repo")
    if not isinstance(recorded, str) or not recorded:
        # Recorded before D-GH-6 stored the identity: derive it the same way.
        try:
            recorded = canonical_repo(str(source.get("repo_url") or ""))
        except ValueError:
            return False
    ref = source.get("ref")
    if not isinstance(ref, str) or not ref:
        return False
    return recorded == want_repo and normalise_nudge_ref(ref) == want_ref


def _is_candidate_state(status: str, cfg: object) -> bool:
    """The candidacy rule itself: live status, opted in, git-sourced.

    Lives here once and is applied three times — on the tick-time snapshot and
    again on each of the two fresh reads inside a poll — so the tick's notion of
    a candidate and the poll's can never drift apart.
    """
    if status not in _POLLABLE_STATUSES:
        return False
    if not isinstance(cfg, dict) or cfg.get("auto_deploy") is not True:
        return False
    source = cfg.get("source")
    return isinstance(source, dict) and source.get("type") == "git"


def _is_candidate(row: dict) -> bool:
    """Whether a workload row opted into unattended redeploy from a git source."""
    if not row.get("service_name"):
        return False
    return _is_candidate_state(str(row.get("status")), row.get("config"))


__all__ = ["GitWatchController", "NudgeResult", "normalise_nudge_ref", "normalise_nudge_repo"]
