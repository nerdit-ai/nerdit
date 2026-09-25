"""Background cleanup and reconciliation loops for the daemon lifespan.

Sweep intervals remain on daemon.server, which passes them to these loops.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from nerdit.core.backup import DUMP_TAR_GLOB, DUMP_TAR_RE
from nerdit.core.remediation_settle import parse_iso
from nerdit.core.workspaces import WorkspaceError, read_meta, settled_to_thread, workspace_lock
from nerdit.daemon.imagegc import _live_service_names
from nerdit.daemon.limits import STORED_TS_FORMAT
from nerdit.utils.disk import du_bytes, resolve_archive_dir
from nerdit.utils.names import DNS_LABEL_RE

if TYPE_CHECKING:
    from nerdit.config.settings import RetentionSettings
    from nerdit.core.proxy import ProxyManager
    from nerdit.core.sweeper import ZombieSweeper
    from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)


async def _zombie_sweep_loop(sweeper: ZombieSweeper, interval: float) -> None:
    """Periodically sweep orphan `managed-by=nerdit` containers. Runs one single-shot
    `nerdit-run` orphan kill FIRST: the run
    registry is in-memory only, so a crashed daemon leaves rowless one-off-run /
    `[deploy].release` containers that the periodic sweep would only reap up
    to ~5 minutes later (30 s age gate + this loop's interval) — long enough for
    the orphan and a retry to write the same volume concurrently. Guarded so a
    failure never keeps the periodic sweep from starting; the loop body below is
    unchanged, and its 30 s age gate means the first `cleanup_zombies` cannot
    double-kill what this pass just reaped.
    """
    try:
        # No summary log here: kill_boot_run_orphans already emits one, and two
        # lines for one event is exactly the crash-recovery log noise this pass
        # was tightened to remove.
        await sweeper.kill_boot_run_orphans()
    except Exception:
        logger.exception("Boot run-orphan kill failed — the periodic sweep will collect them")

    while True:
        try:
            cleaned = await sweeper.cleanup_zombies()
            if cleaned:
                logger.info("Zombie sweep: cleaned %d orphan container(s)", cleaned)
        except Exception:
            logger.exception("Zombie sweep failed")
        await asyncio.sleep(interval)


async def _idempotency_sweep_loop(queries: Queries, interval: float) -> None:
    """Periodically delete expired idempotency keys so the table stays bounded."""
    while True:
        try:
            removed = await queries.sweep_expired_idempotency()
            if removed:
                logger.info("Idempotency sweep: removed %d expired key(s)", removed)
        except Exception:
            logger.exception("Idempotency sweep failed")
        await asyncio.sleep(interval)


def _prune_tars(
    data_dir: Path, keep_last: int, glob: str, name_re: re.Pattern[str] | None = None
) -> int:
    """Delete the oldest `<data_dir>/backups/<glob>` tars beyond `keep_last`.

    With `name_re` the tars are grouped by its `service` group (non-matches are
    skipped) and `keep_last` applies per service; without it, to all of them.
    Runs in a worker thread (blocking stat/unlink).
    """
    backups_dir = data_dir / "backups"
    if not backups_dir.is_dir():
        return 0
    groups: dict[str, list[tuple[float, Path]]] = {}
    for p in backups_dir.glob(glob):
        key = ""
        if name_re is not None:
            match = name_re.match(p.name)
            if match is None:
                continue
            key = match.group("service")
        # Per-file stat tolerance: a file deleted between glob and stat must not
        # abort the pass (and thereby skip the sweep summary row — the sha anchor).
        with contextlib.suppress(OSError):
            groups.setdefault(key, []).append((p.stat().st_mtime, p))
    removed = 0
    for stamped in groups.values():
        for _, stale in sorted(stamped)[:-keep_last]:
            with contextlib.suppress(OSError):
                stale.unlink()
                removed += 1
    return removed


# The three tar flavours have pairwise-disjoint globs, so no phase ever prunes
# another flavour's tars (pinned by test).
# `nerdit-volumes-<service>-<UTC %Y%m%dT%H%M%SZ>-<6 hex>.tar.gz`. The
# service segment is greedy `.+` but the anchored stamp + 6-hex + extension pin
# it exactly, so a hyphenated DNS-label service name parses correctly. The
# dump-tar grammar lives in `core/backup.py` beside the packer that mints it.
_VOLUME_TAR_RE = re.compile(r"^nerdit-volumes-(?P<service>.+)-\d{8}T\d{6}Z-[0-9a-f]{6}\.tar\.gz$")


def _prune_backups(data_dir: Path, keep_last: int) -> int:
    """Control-plane `nerdit-backup-*` tars, one group."""
    return _prune_tars(data_dir, keep_last, "nerdit-backup-*.tar.gz")


def _prune_volume_backups(data_dir: Path, keep_last: int) -> int:
    """Per-service `nerdit-volumes-<service>-*` tars."""
    return _prune_tars(data_dir, keep_last, "nerdit-volumes-*.tar.gz", _VOLUME_TAR_RE)


def _prune_dump_backups(data_dir: Path, keep_last: int) -> int:
    """Per-service `nerdit-dump-<service>-*` tars (`dump_keep_last`, default 5)."""
    return _prune_tars(data_dir, keep_last, DUMP_TAR_GLOB, DUMP_TAR_RE)


async def _chunked_delete(sweep_call: Callable[[], Awaitable[int]]) -> int:
    """Loop a chunked `@_serialized` sweep until it returns 0; sum the counts.

    The write lock is released between chunks (`@_serialized` is per-call) so a
    long sweep never blocks live writers for its whole duration.
    """
    total = 0
    while True:
        removed = await sweep_call()
        total += removed
        if removed == 0:
            break
        await asyncio.sleep(0.05)
    return total


async def _sweep_event_feed(queries: Queries, keep_last: int) -> int:
    """Prune durable events in chunks against one fixed keep-last watermark.

    Use plain deletes, without audit-style archives or anchors. Disabled/empty
    prunes are handled by the query layer. Report events_deleted in the retention
    summary; log failures without preventing later sweep phases.
    """
    try:
        watermark = await queries.event_sweep_watermark(keep_last)
        if watermark is None:
            return 0
        return await _chunked_delete(lambda: queries.sweep_events(keep_last, watermark=watermark))
    except Exception:
        logger.exception("Retention event-feed phase failed (sweep continues)")
        return 0


def _workspace_written_at(data_dir: Path, name: str, mtime_source: Path) -> datetime | None:
    """Age of one workspace: `meta.json`'s `last_written_at`, else dir mtime.

    A missing or unparseable sidecar falls back to the directory's own mtime, so
    a corrupted meta can neither exempt a workspace forever nor age a fresh one
    out. `None` means "could not tell" — the caller skips such an entry.

    Pure reads, so it is safe both in a worker thread (the scan) and on the loop
    via `asyncio.to_thread` (the per-removal re-check).
    """
    written: datetime | None = None
    try:
        meta = read_meta(data_dir, name)
    except WorkspaceError:
        # A dir the write path could never have created (bad name / symlinked
        # component), or a sidecar that is present but corrupt/unreadable.
        # Tolerance lives HERE, not inside `read_meta` (which fails closed for
        # ownership checks): a corrupt sidecar must not exempt a workspace from
        # the sweep forever, and no ownership decision is made here. It is
        # still an orphan; fall through to the mtime rule.
        meta = None
    if isinstance(meta, dict):
        raw = meta.get("last_written_at")
        if isinstance(raw, str):
            written = parse_iso(raw)
    if written is None:
        try:
            written = datetime.fromtimestamp(mtime_source.stat().st_mtime, UTC)
        except OSError:
            return None
    return written


def _orphan_still_aged(data_dir: Path, name: str, cutoff: datetime) -> bool:
    """Re-run the age rule for one candidate (the loop-side TOCTOU re-check).

    A workspace rewritten between the scan and its removal is no longer aged
    out, and must survive the sweep.
    """
    written = _workspace_written_at(data_dir, name, data_dir / "workspaces" / name)
    return written is not None and written < cutoff


def _scan_workspace_orphans(
    data_dir: Path, live: set[str], cutoff: datetime
) -> tuple[list[str], list[str], int]:
    """Find aged workspace orphans in a worker thread.

    Any workload row, including stopped rows, or a project row protects its
    workspace (a project's workspace is named after the project). Legal names
    are returned for locked removal and fresh checks on the event loop; remove only
    junk the write path cannot create in this thread. The live set is a prefilter.

    Returns:
        A tuple of candidates, junk directories removed and junk bytes removed.
    """
    root = data_dir / "workspaces"
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return [], [], 0

    candidates: list[str] = []
    junk: list[str] = []
    junk_bytes = 0
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        if entry.name in live:
            continue
        written = _workspace_written_at(data_dir, entry.name, Path(entry.path))
        if written is None or written >= cutoff:
            continue
        if DNS_LABEL_RE.fullmatch(entry.name):
            candidates.append(entry.name)
            continue
        path = Path(entry.path)
        size = du_bytes(path)
        try:
            shutil.rmtree(path)
        except OSError:
            continue
        junk.append(entry.name)
        junk_bytes += size
    return candidates, junk, junk_bytes


async def _live_workspace_names(queries: Queries) -> set[str]:
    """Names whose workspace is live.

    Any workload row (any kind, any desired_state — a stopped service is still
    live) or any project row protects it.
    """
    return _live_service_names(await queries.list_workload_configs()) | (
        await queries.list_project_names()
    )


async def _sweep_workspace_phase(
    queries: Queries, orphan_days: int, data_dir: Path, now: datetime
) -> int:
    """Remove aged workspace orphans and return the count, or zero on failure.

    A service row or a project row protects the workspace. Scan in a worker; lock
    each candidate on the event loop and recheck row existence and age before
    removal. Log failures without stopping later phases.
    Audit removed names/bytes only, with no paths, contents or SSE publication.
    """
    if orphan_days <= 0:
        return 0
    try:
        live = await _live_workspace_names(queries)
        cutoff = now - timedelta(days=orphan_days)
        candidates, names, total_bytes = await asyncio.to_thread(
            _scan_workspace_orphans, data_dir, live, cutoff
        )
        for name in candidates:
            async with workspace_lock(name):
                if name in await _live_workspace_names(queries):
                    continue  # a row appeared mid-sweep
                # Every `to_thread` under a `workspace_lock` goes through
                # `settled_to_thread` — the sweep task is a RAW asyncio task,
                # precisely the case a bare anyio shield would not defer, so an
                # abandoned rmtree here would race a fresh first write with the
                # lock already free. The two read-only re-checks take it too, so
                # the invariant is one grep and not a judgement call.
                if not await settled_to_thread(_orphan_still_aged, data_dir, name, cutoff):
                    continue  # rewritten mid-sweep
                path = data_dir / "workspaces" / name
                size = await settled_to_thread(du_bytes, path)
                try:
                    await settled_to_thread(shutil.rmtree, path)
                except OSError:
                    continue
                names.append(name)
                total_bytes += size
        if not names:
            return 0
        names.sort()
        await queries.insert_audit_log(
            action="workspace.purge_orphan",
            result="ok",
            principal_id="system",
            principal_role="system",
            params_redacted=json.dumps(
                {"names": names, "total_bytes": total_bytes}, sort_keys=True
            ),
        )
        return len(names)
    except Exception:
        logger.exception("Retention workspace-orphan phase failed (sweep continues)")
        return 0


def retention_sweep_enabled(retention: RetentionSettings) -> bool:
    """True when at least one retention phase would prune something; the lifespan
    skips the sweep loop otherwise. Every phase's threshold must be represented
    here: a phase missing from the predicate that gates its own loop never runs.
    """
    return bool(
        retention.job_log_days
        or retention.audit_days
        or retention.backup_keep_last
        or retention.volume_backup_keep_last
        # Phase 3e, default-ON at 5 per service.
        or retention.dump_keep_last
        # Phase 3d.
        or retention.workspace_orphan_days
        # The durable feed's keep-last-N is default-ON (10000), so it is
        # normally what keeps this loop alive at all.
        or retention.events_keep_last
    )


async def _run_retention_sweep(
    queries: Queries,
    retention: RetentionSettings,
    data_dir: Path,
    archive_dir: Path | None,
) -> None:
    """One retention sweep pass (all phases). Each phase is skippable at 0."""
    from nerdit.core.retention import AuditArchiveResult, archive_and_prune_audit

    now = datetime.now(UTC)
    counts: dict[str, object] = {}

    # Phase 1: job_logs, every kind. Cutoff in the `datetime('now')` space
    # format (job_logs.timestamp column) — NOT isoformat.
    if retention.job_log_days > 0:
        cutoff = (now - timedelta(days=retention.job_log_days)).strftime(STORED_TS_FORMAT)
        try:
            removed = await _chunked_delete(lambda: queries.sweep_job_logs(cutoff))
            if removed:
                counts["job_logs"] = removed
        except Exception:
            logger.exception("Retention job-log phase failed (sweep continues)")

    # Phase 3: control-plane backup-keep. Runs
    # BEFORE the audit prune so its count is known when the anchor row (below)
    # is written; isolated so a failure never blocks the anchor.
    if retention.backup_keep_last > 0:
        try:
            removed = await asyncio.to_thread(_prune_backups, data_dir, retention.backup_keep_last)
            if removed:
                counts["backups_removed"] = removed
        except Exception:
            logger.exception("Retention backup-keep phase failed (sweep continues)")

    # Phase 3b: per-database volume-tar keep (no-op at the default 0).
    # Isolated like phase 3 so a failure never blocks the anchor/summary row.
    if retention.volume_backup_keep_last > 0:
        try:
            removed = await asyncio.to_thread(
                _prune_volume_backups, data_dir, retention.volume_backup_keep_last
            )
            if removed:
                counts["volume_backups_removed"] = removed
        except Exception:
            logger.exception("Retention volume-backup-keep phase failed (sweep continues)")

    # Phase 3c: durable event feed. Runs BEFORE the audit phase
    # so the count is already in `counts` when the anchor row is written.
    removed = await _sweep_event_feed(queries, retention.events_keep_last)
    if removed:
        counts["events_deleted"] = removed

    # Phase 3d: orphan agent workspaces. Isolated like every
    # other phase so a scan failure never blocks the anchor/summary row. The
    # live-name set is read fresh here (any kind, any desired_state) — the
    # DELETE-time `?purge=workspace` member is the other, disjoint reclaim
    # path (`daemon/service_purge.py::_purge_workspace`): that one only ever
    # sees workspaces whose row existed, this one only those whose row does not.
    swept = await _sweep_workspace_phase(queries, retention.workspace_orphan_days, data_dir, now)
    if swept:
        counts["workspace_orphans_removed"] = swept

    # Phase 3e: per-database dump-tar keep (default 5). The phase letters are
    # load-bearing in the retention suite, so new phases append, never
    # renumber. Isolated like phases 3/3b so a failure never blocks the anchor/summary row.
    if retention.dump_keep_last > 0:
        try:
            removed = await asyncio.to_thread(
                _prune_dump_backups, data_dir, retention.dump_keep_last
            )
            if removed:
                counts["dump_backups_removed"] = removed
        except Exception:
            logger.exception("Retention dump-backup-keep phase failed (sweep continues)")

    async def _write_sweep_row(params: dict[str, object]) -> None:
        # DB-only, no SSE publish — the ServiceController._audit system-
        # principal norm. Called strictly between (never inside)
        # @_serialized frames.
        await queries.insert_audit_log(
            action="retention.sweep",
            result="ok",
            principal_id="system",
            principal_role="system",
            params_redacted=json.dumps(params, sort_keys=True),
        )

    # Phase 4 (LAST): archive-first audit prune. Cutoff in the
    # `datetime('now')` space format (audit_log.ts column). An archiving pass
    # writes TWO `retention.sweep` rows:
    #   * anchor row  — written by `_anchor` (`on_archived`) after the
    #     archive is fsynced+hashed but BEFORE the first row delete: carries
    #     audit_archived + basename + sha256, the tamper-evidence anchor. It
    #     survives even if the daemon crashes mid-prune.
    #   * summary row — written below AFTER `archive_and_prune_audit` returns
    #     successfully: the completion record, carrying the full counts
    #     (audit_archived AND audit_deleted) + basename (+ sha256) for
    #     correlation.
    # An anchor row with NO matching summary row therefore reliably signals an
    # aborted prune (a crash or delete failure between the two writes).
    if retention.audit_days > 0 and archive_dir is None:
        # Misconfigured archive dir disabled archiving at loop start; repeat
        # the error every pass so it cannot scroll away once and hide that
        # audit rows are silently accumulating unpruned.
        logger.error(
            "audit_archive_dir is misconfigured (resolves under a protected "
            "directory); audit archiving/pruning remains DISABLED."
        )
    if retention.audit_days > 0 and archive_dir is not None:
        cutoff = (now - timedelta(days=retention.audit_days)).strftime(STORED_TS_FORMAT)

        async def _anchor(meta: AuditArchiveResult) -> None:
            params: dict[str, object] = dict(counts)
            params["audit_archived"] = meta.archived
            params["archive"] = meta.basename
            params["sha256"] = meta.sha256
            await _write_sweep_row(params)

        try:
            result = await archive_and_prune_audit(
                queries, archive_dir, cutoff, on_archived=_anchor
            )
        except Exception:
            # Phases 1-2 may have ALREADY deleted rows; an archive/anchor/prune
            # failure must not skip auditing them (Invariant #3). Log and fall
            # through so the summary row below still records whatever phase
            # counts exist. No audit_deleted is added to `counts` here, so the
            # anchor row (if the callback already committed) stays orphaned —
            # the aborted-prune signature.
            logger.exception("Retention audit-prune phase failed (sweep continues)")
        else:
            if result.archived or result.deleted:
                counts["audit_archived"] = result.archived
                counts["audit_deleted"] = result.deleted
                counts["archive"] = result.basename
                counts["sha256"] = result.sha256

    if not counts:
        return

    # Completion summary row: an archiving pass reaches here with the full
    # counts (audit_archived + audit_deleted + basename), a non-archiving pass
    # with whatever phases 1-3 pruned. A pass that pruned nothing writes nothing.
    await _write_sweep_row(dict(counts))


async def _retention_sweep_loop(
    queries: Queries, retention: RetentionSettings, data_dir: Path
) -> None:
    """Off-tick data-retention sweep: `job_logs` (every kind),
    archive-first audit prune, tar keep-last-N, event feed and orphan workspaces.

    Sleeps **first** (unlike the zombie/idempotency sweepers, which run their
    body immediately at boot): an archive-writing sweep at boot is the wrong
    first tick. The interval comes from `[retention].sweep_interval_seconds`
    (settings, not a module constant — a deliberate deviation from the sweeper
    precedent). Each pass is self-contained (never raises out of the loop).
    """
    interval = float(retention.sweep_interval_seconds)
    archive_dir = resolve_archive_dir(retention, data_dir)
    while True:
        await asyncio.sleep(interval)
        try:
            await _run_retention_sweep(queries, retention, data_dir, archive_dir)
        except Exception:
            logger.exception("Retention sweep failed")


async def _proxy_reconcile_loop(proxy: ProxyManager, interval: float) -> None:
    """Periodically converge Caddy's routes to the live service set.

    Runs on its own task — *separate* from the WorkloadManager loop — so Caddy
    readiness polling / respawn backoff never stalls the workload reconcile.
    Each tick is self-contained (never raises) so the loop survives a flaky Caddy.
    """
    while True:
        try:
            await proxy.reconcile()
        except Exception:
            logger.exception("Proxy reconcile failed")
        await asyncio.sleep(interval)
