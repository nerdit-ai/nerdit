"""Reap orphaned nerdit containers across service, model, and database workloads."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from nerdit.core.runtime.protocol import (
    ContainerNotFoundError,
    ContainerRuntime,
    ContainerRuntimeError,
)
from nerdit.db.enums import JobStatus
from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)

# Skip containers younger than this threshold — they may belong to a workload
# whose DB row has not yet transitioned from `scheduled` to `running`.
ZOMBIE_AGE_THRESHOLD_SECONDS = 30.0

# Pause between listing the boot-time run orphans and killing them, so a
# container the first reconcile tick launched microseconds before the listing
# has time to bind its RunSlot (and therefore appear in `extra_protected`)
# before the kill loop reaches it. Read at call time, never bound at import, so
# tests can flatten it to 0.
BOOT_KILL_SETTLE_SECONDS = 1.0


class ZombieSweeper:
    """Periodic orphan-container reaper for every workload kind."""

    def __init__(
        self,
        queries: Queries,
        runtime: ContainerRuntime,
        *,
        extra_protected: Callable[[], set[str]],
        unbound_run_probe: Callable[[], bool] = lambda: False,
    ) -> None:
        """Build the sweeper.

        `extra_protected` is a **required** keyword argument on purpose:
        it supplies the container ids the DB cannot know about — one-off runs
        and `[deploy].release` executions are rowless containers, so every
        query in `cleanup_zombies` is blind to them. Making it required
        means a future construction site cannot silently omit the wiring and
        turn the sweep back into a killer of live runs. Callers that genuinely
        have nothing to protect pass `lambda: set()` explicitly.

        `unbound_run_probe` reports whether any run/release slot is claimed
        but not yet bound to a container id
        (`ServiceController.has_any_unbound_run`); it lets
        `kill_boot_run_orphans` stand down rather than race a starting
        container. Unlike `extra_protected` it is **optional**, and the
        asymmetry is deliberate: omitting `extra_protected` would kill live
        runs, while omitting this one merely restores the fixed-settle-delay
        residual — the reap is still correct, just occasionally early.
        """
        self._queries = queries
        self._runtime = runtime
        self._extra_protected = extra_protected
        self._unbound_run_probe = unbound_run_probe

    async def cleanup_zombies(self) -> int:
        """Kill nerdit-managed containers not attached to a live job row.

        The protected set covers `running` and `scheduled` rows: a workload
        briefly sits in `scheduled` between `runtime.run()` returning and
        the subsequent `update_job_status(running, ...)` write, so a sweep
        landing in that window must not kill the fresh container. As
        defence-in-depth, containers younger than
        `ZOMBIE_AGE_THRESHOLD_SECONDS` are also skipped regardless of
        which state the row is in.

        CRIT-2: the status loop only covers those two states, so a managed row
        parked in `building`/`degraded`/`restarting` would fall out of it.
        Every managed row's live `container_id` is therefore re-protected
        through a *separate*, status-free query — both layers live in
        `_row_backed_container_ids`, shared with the boot-side kill. The `extra_protected` hook is
        consulted **first** and is
        **fail-closed**: any exception skips the entire tick (return 0, kill
        nothing) instead of sweeping with an incomplete protected set. For a
        destructive sweep that is the cheap direction — a missed 5-minute tick
        leaves an orphan around a little longer, whereas a wrong kill destroys
        a live migration mid-flight. A hook returning an empty set leaves the
        behaviour identical to the pre-P20 sweep.
        """
        try:
            # The copy is INSIDE the guard on purpose: a hook that returns
            # something un-iterable (e.g. a coroutine, if the wiring ever went
            # async) must take the fail-closed branch too, not raise past it.
            protected_ids: set[str] = set(self._extra_protected())
        except Exception:
            # Fail closed: an unknown protected set means an unsafe sweep.
            logger.warning(
                "Could not resolve extra protected containers; skipping zombie sweep",
                exc_info=True,
            )
            return 0

        protected_ids |= await self._row_backed_container_ids()

        try:
            # Instance-scoped: the sweep KILLS, so it must only ever see this
            # daemon's own containers — never a co-located sibling daemon's.
            containers = await self._runtime.list_own_managed_containers()
        except Exception:
            logger.warning("Could not list managed containers for zombie sweep", exc_info=True)
            return 0

        cleaned = 0
        now = datetime.now(UTC)
        for container_id, created_at in containers:
            if container_id in protected_ids:
                continue
            if (now - created_at).total_seconds() < ZOMBIE_AGE_THRESHOLD_SECONDS:
                continue
            logger.warning("Zombie container found: %s, killing", container_id[:12])
            try:
                await self._runtime.kill(container_id)
            except ContainerRuntimeError:
                logger.warning("Failed to kill zombie %s", container_id[:12], exc_info=True)
                continue
            try:
                await self._runtime.remove(container_id, force=True)
            except ContainerRuntimeError:
                logger.warning("Failed to remove zombie %s", container_id[:12], exc_info=True)
                continue
            cleaned += 1

        return cleaned

    async def _row_backed_container_ids(self) -> set[str]:
        """Collect container IDs protected by live workload rows.

        Combine running/scheduled rows with a separate status-free managed-service
        query, which also protects building, degraded, and restarting containers.
        Stale IDs are harmless because callers intersect with live containers.
        """
        ids: set[str] = set()
        for status in (JobStatus.running, JobStatus.scheduled):
            for job in await self._queries.list_jobs(status=status):
                if job.container_id:
                    ids.add(job.container_id)
        ids |= await self._queries.get_service_container_ids()
        return ids

    async def kill_boot_run_orphans(self) -> int:
        """Kill this instance's orphaned run/release containers once at boot, without an age gate.

        Crash loses the in-memory run registry. Prompt cleanup prevents old and retried
        commands writing the same volume until the periodic sweep catches up.
        Protect both registry and row-backed IDs: image-inherited `nerdit-run` labels
        can also appear on live services and are not proof of orphanhood.

        Fail closed on protected-set or runtime errors. Defer the whole pass while any
        run slot is unbound: Docker exposes containers before runtime.run returns their
        IDs, so a fixed delay cannot safely distinguish a new migration from an orphan.
        The periodic sweep remains the fallback.
        """
        try:
            # Same shape (and same reason) as the cleanup_zombies guard: the
            # copy is INSIDE the try so a hook returning something un-iterable
            # takes the fail-closed branch too.
            protected_ids: set[str] = set(self._extra_protected())
        except Exception:
            # Fail closed: an unknown protected set means an unsafe kill.
            logger.warning(
                "Could not resolve extra protected containers; skipping boot run-orphan kill",
                exc_info=True,
            )
            return 0

        try:
            # Instance-scoped: this KILLS, so it must only ever see this
            # daemon's own containers — never a co-located sibling daemon's.
            orphans = await self._runtime.list_own_run_containers()
        except Exception:
            logger.warning("Could not list run containers for the boot orphan kill", exc_info=True)
            return 0

        if not orphans:
            return 0

        # THE RACE GUARD. This pass runs from the zombie-sweep task, which the
        # lifespan starts AFTER `workload_manager.start()` — so the first
        # reconcile tick may already have launched a `[deploy].release`
        # container, which carries the same `nerdit-run` label. Such a
        # container becomes visible to docker (and hence to the listing above)
        # slightly BEFORE its RunSlot binds the container id (the bind happens
        # right after `runtime.run()` returns, an executor hop later), so it
        # could be listed while still unprotected. The settle delay lets any
        # in-flight bind complete before the kill loop looks at the registry
        # again. Containers created AFTER the listing are not in it and can
        # therefore never be killed by this pass at all.
        await asyncio.sleep(BOOT_KILL_SETTLE_SECONDS)

        # ...and the delay is a shrink, not a fix, so it is backed by the probe:
        # if any slot is STILL unbound, a container we might be about to kill
        # could be its. Stand down for this boot; the periodic sweep collects
        # genuine orphans regardless. Fail closed here too — an unreadable
        # probe is an unknown answer, and the safe unknown is "do not kill".
        try:
            unbound = self._unbound_run_probe()
        except Exception:
            logger.warning(
                "Could not probe for unbound runs; skipping boot run-orphan kill",
                exc_info=True,
            )
            return 0
        if unbound:
            logger.info(
                "Boot run-orphan kill stood down: a run/release is starting "
                "(the periodic sweep will collect any orphans)"
            )
            return 0

        # Read AFTER the settle delay, for the same reason the delay exists: a
        # row whose container id was written between the listing and now is
        # then already visible here. One read for the whole pass — the loop
        # below only re-reads the (sync, cheap) registry hook.
        try:
            row_backed = await self._row_backed_container_ids()
        except Exception:
            # Fail closed: a partial row-backed set could cost a live service.
            logger.warning(
                "Could not resolve row-backed containers; skipping boot run-orphan kill",
                exc_info=True,
            )
            return 0

        killed = 0
        for container_id in orphans:
            try:
                # Re-evaluated FRESH per iteration (sync and cheap): the whole
                # point of the settle delay is that the registry may have gained
                # the id since the listing.
                protected_ids = set(self._extra_protected()) | row_backed
            except Exception:
                # Fail closed mid-loop too: stop, keeping whatever was reaped.
                logger.warning(
                    "Could not resolve extra protected containers; "
                    "aborting the boot run-orphan kill",
                    exc_info=True,
                )
                break
            if container_id in protected_ids:
                continue
            killed += await self._reap_boot_orphan(container_id)

        if killed:
            logger.info("Boot run-orphan kill: reaped %d container(s)", killed)
        return killed

    async def _reap_boot_orphan(self, container_id: str) -> int:
        """Kill and remove one boot orphan best-effort, reporting whether it was reaped."""
        try:
            await self._runtime.kill(container_id)
        except ContainerNotFoundError:
            # Already gone — most often reaped a moment earlier by
            # _settle_crashed_release, which runs the same crash recovery
            # from the build side. Silent: announcing a kill that did not
            # happen is worse than saying nothing during crash recovery.
            return 0
        except ContainerRuntimeError:
            # Most often "container not running": the listing is
            # running-only, but the settle delay and the loop give a
            # short-lived orphan time to exit on its own. That is the
            # outcome we wanted, not a failure — so it is not counted as a
            # kill and not warned about, but the remove below still runs,
            # or the exited container would leak (every listing this class
            # has is running-only, so nothing else would ever collect it).
            logger.debug(
                "Boot run orphan %s could not be killed (likely already exited); removing anyway",
                container_id[:12],
                exc_info=True,
            )
            reaped = 0
        else:
            # Counted at the KILL, not after the remove. What this pass
            # guarantees — and what the boot log line reports — is that the
            # orphan has stopped writing to the service's data volume; the
            # remove is best-effort tidying of an already-inert exited
            # container, so a failure there must not erase a real kill from
            # the count.
            reaped = 1
            logger.warning("Boot run orphan killed: %s", container_id[:12])

        try:
            await self._runtime.remove(container_id, force=True)
        except ContainerNotFoundError:
            pass
        except ContainerRuntimeError:
            logger.warning("Failed to remove run orphan %s", container_id[:12], exc_info=True)
        return reaped
