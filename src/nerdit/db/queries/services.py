"""Service, model and database lookups plus transactional checked deletion."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from datetime import datetime
from types import SimpleNamespace

from nerdit.core.jobconfig import parse_job_config
from nerdit.db.models import (
    MANAGED_KINDS,
    MANAGED_KINDS_SQL,
    ActiveServiceRoute,
    Job,
    JobKind,
    JobStatus,
)

from ._base import _JOB_SELECT, QueriesBase, _decode_cursor, _encode_cursor, _serialized


def _parse_config_blob(raw: str | None) -> dict:
    """Wrap a raw config column for the shared parser; invalid or absent config becomes an empty
    dict.
    """
    return parse_job_config(SimpleNamespace(config=raw))


def _row_edge_auth(service_name: str, raw: str | None) -> dict | None:
    """Read edge auth from persisted config without raising on malformed JSON.

    An absent declaration returns None. A declared non-table returns an empty dict
    so edge-auth validation fails closed rather than exposing an intended protected route.
    """
    cfg = parse_job_config(SimpleNamespace(config=raw, id=service_name), warn=True)
    value = cfg.get("edge_auth")
    if value is None:
        return None
    return value if isinstance(value, dict) else {}


class ServiceQueries(QueriesBase):
    """Service/model/database reconcile/list/lookup reads + the checked delete."""

    # --- Service operations (P2) ---

    async def get_reconcilable_services(self) -> list[Job]:
        """Return service/model rows the `ServiceController` must converge.

        A row needs reconciliation when it is not in a settled terminal state
        (`completed`/`cancelled`/`stopped`/`failed`) **or** when its
        `desired_state` no longer matches its `status` (e.g. a user asked a
        `stopped` service to run again, or a running one to stop).
        """
        cursor = await self._db.conn.execute(
            f"{_JOB_SELECT} WHERE {MANAGED_KINDS_SQL} "
            "AND (status NOT IN ('completed', 'cancelled', 'stopped', 'failed') "
            "OR desired_state != status)"
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(r) for r in rows]

    async def list_active_service_routes(self) -> list[ActiveServiceRoute]:
        """Return routed service candidates and their effective live ports.

        Include running, degraded and restarting services with endpoints, preserving
        URLs while old containers serve during redeploy. Models remain unrouted.
        Use active_host_port when set, otherwise the stable reservation, so reconcile
        honors cutover. Include raw edge_auth for per-tick secret/repair convergence.
        """
        cursor = await self._db.conn.execute(
            "SELECT j.service_name AS service_name, "
            "COALESCE(e.active_host_port, e.host_port) AS host_port, "
            "j.status AS status, e.route AS route, j.config AS config, "
            "j.project_id AS project_id FROM jobs j "
            "JOIN service_endpoints e ON e.service_name = j.service_name "
            "WHERE j.kind = 'service' "
            "AND j.status IN ('running', 'degraded', 'restarting')"
        )
        rows = await cursor.fetchall()
        return [
            ActiveServiceRoute(
                service_name=r["service_name"],
                host_port=r["host_port"],
                status=r["status"],
                route=r["route"],
                project_id=r["project_id"],
                edge_auth=_row_edge_auth(r["service_name"], r["config"]),
            )
            for r in rows
        ]

    async def get_service_container_ids(self) -> set[str]:
        """Return managed container IDs regardless of status for zombie protection.

        Building, degraded and restarting workloads may still own live containers;
        stale IDs are harmless because they no longer match live containers.
        """
        cursor = await self._db.conn.execute(
            f"SELECT container_id FROM jobs WHERE {MANAGED_KINDS_SQL} AND container_id IS NOT NULL"
        )
        rows = await cursor.fetchall()
        return {r["container_id"] for r in rows}

    async def get_service_by_name(self, service_name: str) -> Job | None:
        """Fetch the single service row owning `service_name`, or `None`."""
        cursor = await self._db.conn.execute(
            f"{_JOB_SELECT} WHERE service_name = ?", (service_name,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    async def get_resource_by_ref(self, ref: str, kind: JobKind) -> Job | None:
        """Find the best resource row: models by config.model, databases by service name.

        Prefer running, then other non-terminal, then terminal rows; ties use newest
        creation. Model name overrides do not change the served reference.
        """
        terminal = {
            JobStatus.completed,
            JobStatus.cancelled,
            JobStatus.stopped,
            JobStatus.failed,
        }
        cursor_obj = await self._db.conn.execute(
            f"{_JOB_SELECT} WHERE kind = ? ORDER BY jobs.created_at DESC, jobs.id DESC",
            (kind.value,),
        )
        rows = await cursor_obj.fetchall()
        best: Job | None = None
        best_rank = -1
        for r in rows:
            job = self._row_to_job(r)
            if kind is JobKind.database:
                if job.service_name != ref:
                    continue
            else:
                cfg = parse_job_config(job)
                if cfg.get("model") != ref:
                    continue
            rank = 2 if job.status is JobStatus.running else (0 if job.status in terminal else 1)
            # Rows are newest-first, so a strict ``>`` keeps the newest among
            # rows of equal rank.
            if rank > best_rank:
                best, best_rank = job, rank
        return best

    async def get_model_by_ref(self, model: str) -> Job | None:
        """Find the best model row by its served config.model reference."""
        return await self.get_resource_by_ref(model, JobKind.model)

    async def list_workload_configs(self) -> list[dict[str, object]]:
        """Return every managed row's (service/model/database) identity + parsed config.

        Plain read (NOT `@_serialized`): a lightweight projection used by the
        delete reference guard, the image protected-set, GC's TOCTOU re-checks,
        and the C2 tombstone startup sweep. Each entry is
        `{id, kind, service_name, status, desired_state, config}` with
        `config` parsed to a dict (`{}` on a malformed/absent blob).
        """
        cursor = await self._db.conn.execute(
            f"SELECT id, kind, service_name, status, desired_state, config "
            f"FROM jobs WHERE {MANAGED_KINDS_SQL}"
        )
        rows = await cursor.fetchall()
        result: list[dict[str, object]] = []
        for r in rows:
            cfg = _parse_config_blob(r["config"])
            result.append(
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "service_name": r["service_name"],
                    "status": r["status"],
                    "desired_state": r["desired_state"],
                    "config": cfg,
                }
            )
        return result

    async def list_services(
        self,
        *,
        status: JobStatus | None = None,
        cursor: str | None = None,
        limit: int = 50,
        kinds: tuple[JobKind, ...] = MANAGED_KINDS,
    ) -> tuple[list[Job], str | None]:
        """Cursor-paginated list of service/model rows, newest first.

        Uses a `(created_at, id)` cursor so pagination is stable across
        inserts; `limit` is clamped to `[1, 200]`.
        Backs the bounded `GET /services` read surface; `GET /models`
        narrows `kinds` to `(JobKind.model,)` and `GET /databases`
        to `(JobKind.database,)` — the default (`MANAGED_KINDS`) lists every
        managed kind.
        """
        limit = max(1, min(limit, 200))
        if not kinds:
            raise ValueError("kinds must name at least one workload kind")
        placeholders = ", ".join("?" for _ in kinds)
        clauses: list[str] = [f"kind IN ({placeholders})"]
        params: list[object] = [k.value for k in kinds]
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if cursor:
            cursor_ts, cursor_id = _decode_cursor(cursor)
            clauses.append("(jobs.created_at < ? OR (jobs.created_at = ? AND jobs.id < ?))")
            params.extend([cursor_ts.isoformat(), cursor_ts.isoformat(), cursor_id])
        where = " WHERE " + " AND ".join(clauses)
        sql = f"{_JOB_SELECT}{where} ORDER BY jobs.created_at DESC, jobs.id DESC LIMIT ?"
        params.append(limit + 1)
        cursor_obj = await self._db.conn.execute(sql, params)
        rows = list(await cursor_obj.fetchall())
        services = [self._row_to_job(r) for r in rows[:limit]]
        next_cursor: str | None = None
        if len(rows) > limit and services:
            last = services[-1]
            next_cursor = _encode_cursor(last.created_at, last.id)
        return services, next_cursor

    async def count_services_up(self) -> int:
        """Count running/degraded managed workloads; degraded containers are still up."""
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) AS n FROM jobs "
            f"WHERE {MANAGED_KINDS_SQL} AND status IN ('running', 'degraded')"
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    @_serialized
    async def set_desired_state(self, job_id: str, desired_state: str) -> None:
        """Set a service's reconciler target (`running` | `stopped`).

        The router writes desired state directly; the controller only converges
        toward it (decision #5 — desired-state writes bypass the controller).
        """
        await self._db.conn.execute(
            "UPDATE jobs SET desired_state = ? WHERE id = ?",
            (desired_state, job_id),
        )
        await self._db.conn.commit()

    @_serialized
    async def record_service_exit(self, job_id: str, exit_at: datetime) -> None:
        """Stamp a service's `last_exit_at` (the backoff anchor; decision #4)."""
        await self._db.conn.execute(
            "UPDATE jobs SET last_exit_at = ? WHERE id = ?",
            (exit_at.isoformat(), job_id),
        )
        await self._db.conn.commit()

    @_serialized
    async def bump_restart_count(
        self,
        job_id: str,
        restart_count: int,
        restart_window_start: datetime | None,
    ) -> None:
        """Write a service's restart counter and rate-window anchor.

        The controller computes both — resetting `restart_count` to `1` with
        a fresh `restart_window_start` when the window has elapsed, or
        incrementing within the window — and persists them here so the backoff /
        rate-cap state survives a daemon restart (statelessness, decision #4).
        """
        await self._db.conn.execute(
            "UPDATE jobs SET restart_count = ?, restart_window_start = ? WHERE id = ?",
            (
                restart_count,
                restart_window_start.isoformat() if restart_window_start else None,
                job_id,
            ),
        )
        await self._db.conn.commit()

    @_serialized
    async def delete_service_checked(
        self,
        job_id: str,
        find_dependents: Callable[[list[dict[str, object]]], list[dict[str, str | None]]]
        | None = None,
        *,
        on_share_removed: Callable[[bool], None] | None = None,
        on_domains_removed: Callable[[list[str]], None] | None = None,
        on_secrets_reclaimed: Callable[[bool], None] | None = None,
    ) -> list[dict[str, str | None]] | None:
        """Atomically check dependents and delete a managed service, model or database.

        Hold the write lock and BEGIN IMMEDIATE across the check and delete, preventing
        new bindings from racing the guard. A nonempty pure find_dependents result rolls
        back; None skips checking. Dependency rows include submitted_by_token for owner
        checks. Delete FK children and name-keyed shares/domains in the same transaction;
        a refusal preserves endpoints and their stable ports.

        Args:
            on_share_removed: Called once after successful commit with whether a share
                was deleted, determined inside the transaction for accurate events.
            on_domains_removed: Called once after successful commit with sorted removed
                domain names, including domains added before this transaction acquired its lock.
            on_secrets_reclaimed: When given, re-mint the name's secret claim for the
                row's owner in this transaction (D-P39-6) and report after commit
                whether it was minted. Inside the transaction because a stranger's
                fresh row landing between the commit and a later mint would launch
                with the kept secret file.

        Returns:
            None if absent (no name-keyed purge is authorized), an empty list after
            deletion, or the nonempty dependent list when refused. Distinguish absence
            from success so stale deletes cannot purge a same-name replacement.
        """
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.conn.execute(
                f"SELECT service_name, submitted_by_token FROM jobs "
                f"WHERE id = ? AND {MANAGED_KINDS_SQL}",
                (job_id,),
            )
            target = await cursor.fetchone()
            if target is None:
                await self._db.conn.execute("ROLLBACK")
                return None
            if find_dependents is not None:
                cursor = await self._db.conn.execute(
                    "SELECT id, kind, service_name, status, config, submitted_by_token "
                    f"FROM jobs WHERE {MANAGED_KINDS_SQL}"
                )
                raw = await cursor.fetchall()
                rows: list[dict[str, object]] = []
                for r in raw:
                    cfg = _parse_config_blob(r[4])
                    rows.append(
                        {
                            "id": r[0],
                            "kind": r[1],
                            "service_name": r[2],
                            "status": r[3],
                            "config": cfg,
                            "submitted_by_token": r[5],
                        }
                    )
                deps = find_dependents(rows)
                if deps:
                    await self._db.conn.execute("ROLLBACK")
                    return deps
            await self._db.conn.execute("DELETE FROM job_logs WHERE job_id = ?", (job_id,))
            await self._db.conn.execute("DELETE FROM gpu_allocations WHERE job_id = ?", (job_id,))
            await self._db.conn.execute(
                "DELETE FROM service_endpoints WHERE job_id = ? "
                "OR service_name = (SELECT service_name FROM jobs WHERE id = ?)",
                (job_id, job_id),
            )
            # Cascade shares transactionally so replacements never inherit exposure.
            # Use raw SQL: the public writer would reacquire this non-reentrant lock.
            share_cursor = await self._db.conn.execute(
                "DELETE FROM service_shares "
                "WHERE service_name = (SELECT service_name FROM jobs WHERE id = ?)",
                (job_id,),
            )
            share_removed = (share_cursor.rowcount or 0) > 0
            # Capture domain names for events, then cascade in this transaction.
            # Replacements must not inherit them; public writers would reacquire the lock.
            domain_cursor = await self._db.conn.execute(
                "SELECT domain FROM service_domains WHERE service_name = "
                "(SELECT service_name FROM jobs WHERE id = ?) ORDER BY domain",
                (job_id,),
            )
            removed_domains = [r[0] for r in await domain_cursor.fetchall()]
            await self._db.conn.execute(
                "DELETE FROM service_domains WHERE service_name = "
                "(SELECT service_name FROM jobs WHERE id = ?)",
                (job_id,),
            )
            await self._db.conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            secrets_reclaimed = False
            if on_secrets_reclaimed is not None and target["service_name"] is not None:
                claim_cursor = await self._db.conn.execute(
                    "INSERT OR IGNORE INTO secret_claims (service_name, token_id) VALUES (?, ?)",
                    (target["service_name"], target["submitted_by_token"]),
                )
                secrets_reclaimed = (claim_cursor.rowcount or 0) > 0
            # (P40b / D-P40-5) No project prune here any more: a project outlives
            # its services and keeps reserving its name for the owner token
            # (the D-P39-6 posture one scope up); `delete_project_checked`
            # releases it on request.
            await self._db.conn.commit()
            if on_share_removed is not None:
                on_share_removed(share_removed)
            if on_domains_removed is not None:
                on_domains_removed(removed_domains)
            if on_secrets_reclaimed is not None:
                on_secrets_reclaimed(secrets_reclaimed)
            return []
        except BaseException:
            with contextlib.suppress(Exception):
                await self._db.conn.execute("ROLLBACK")
            raise
