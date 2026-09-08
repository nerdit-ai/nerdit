"""Workload row queries and atomic quota-checked reservations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import cast

from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

from ._base import _ACTIVE_STATUSES, QueriesBase, ServiceNameTaken, _serialized


class WorkloadQueries(QueriesBase):
    """Job-row CRUD + the atomic quota-checked reserve path."""

    # --- Job operations ---

    async def _exec_job_insert(self, job: Job) -> None:
        """Execute the `jobs` INSERT for *job* (no commit / transaction control).

        Shared by `create_job` (which commits immediately) and
        `reserve_service_for_token` (which runs it inside an open
        `BEGIN IMMEDIATE` quota transaction).
        """
        await self._db.conn.execute(
            """INSERT INTO jobs
               (id, name, script_path, gpu_count, status,
                container_id, created_at, started_at, finished_at,
                exit_code, config,
                error_class, error_message, submitted_via, kind,
                submitted_by_token, idempotency_key,
                desired_state, restart_policy, restart_count, health_check,
                service_name, last_exit_at, restart_window_start)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.id,
                job.name,
                job.script_path,
                job.gpu_count,
                job.status.value,
                job.container_id,
                job.created_at.isoformat(),
                job.started_at.isoformat() if job.started_at else None,
                job.finished_at.isoformat() if job.finished_at else None,
                job.exit_code,
                job.config,
                job.error_class.value if job.error_class else None,
                job.error_message,
                job.submitted_via,
                job.kind.value,
                job.submitted_by_token,
                job.idempotency_key,
                job.desired_state,
                job.restart_policy,
                job.restart_count,
                json.dumps(job.health_check) if job.health_check else None,
                job.service_name,
                job.last_exit_at.isoformat() if job.last_exit_at else None,
                job.restart_window_start.isoformat() if job.restart_window_start else None,
            ),
        )

    @_serialized
    async def create_job(self, job: Job) -> Job:
        """Persist a job without scheduling or quota checks; daemon submissions use
        reserve_service_for_token.
        """
        await self._exec_job_insert(job)
        await self._db.conn.commit()
        return job

    async def _check_quota_locked(
        self, token_id: str, gpu_count: int, statuses: tuple[str, ...]
    ) -> None:
        """Check caps against all active workload kinds inside the caller's BEGIN IMMEDIATE
        transaction.

        Include building, degraded and restarting states. Raise QuotaExceeded on breach;
        the caller owns commit and rollback.
        """
        from nerdit.daemon.auth import QuotaExceeded

        cursor = await self._db.conn.execute(
            "SELECT max_gpus, max_concurrent_jobs FROM api_tokens WHERE id = ? AND revoked = 0",
            (token_id,),
        )
        caps = await cursor.fetchone()
        max_gpus = caps["max_gpus"] if caps else None
        max_concurrent = caps["max_concurrent_jobs"] if caps else None

        placeholders = ",".join("?" for _ in statuses)
        cursor = await self._db.conn.execute(
            f"SELECT COUNT(*) AS jobs, COALESCE(SUM(gpu_count), 0) AS gpus "
            f"FROM jobs WHERE submitted_by_token = ? AND status IN ({placeholders})",
            (token_id, *statuses),
        )
        counts = cast(sqlite3.Row, await cursor.fetchone())
        active_jobs, active_gpus = counts["jobs"], counts["gpus"]

        if max_concurrent is not None and active_jobs + 1 > max_concurrent:
            raise QuotaExceeded("max_concurrent_jobs", limit=max_concurrent, current=active_jobs)
        if max_gpus is not None and active_gpus + gpu_count > max_gpus:
            raise QuotaExceeded("max_gpus", limit=max_gpus, current=active_gpus)

    @_serialized
    async def reserve_service_for_token(self, job: Job) -> Job:
        """Check token quotas and insert a workload atomically under BEGIN IMMEDIATE.

        Count all active kinds, including transient service states. Legacy/local
        submitted_by_token=None is uncapped. Do not call inside another transaction.

        Raises:
            QuotaExceeded: The token's active footprint exceeds a cap.
            ServiceNameTaken: The unique service name collided; the transaction rolls back.
        """
        import sqlite3

        token_id = job.submitted_by_token
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            if token_id is not None:
                await self._check_quota_locked(token_id, job.gpu_count, _ACTIVE_STATUSES)
            try:
                await self._exec_job_insert(job)
            except sqlite3.IntegrityError as exc:
                if "service_name" in str(exc):
                    raise ServiceNameTaken(job.service_name) from exc
                raise
            await self._db.conn.execute("COMMIT")
            return job
        except Exception:
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    async def count_active_jobs_public(self, token_id: str) -> tuple[int, int | None]:
        """Read active workload count and concurrency cap without taking the write lock.

        Include all active kinds; missing or revoked tokens receive cap zero. Only a
        live token with a NULL cap is uncapped. One-off runs add their in-memory pending
        count because they create no row. This is a snapshot: callers own race control
        and must account for the remaining cross-pool race.
        """
        cursor = await self._db.conn.execute(
            "SELECT max_concurrent_jobs FROM api_tokens WHERE id = ? AND revoked = 0",
            (token_id,),
        )
        caps = await cursor.fetchone()
        max_concurrent: int | None = caps["max_concurrent_jobs"] if caps else 0

        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        cursor = await self._db.conn.execute(
            f"SELECT COUNT(*) AS jobs FROM jobs "
            f"WHERE submitted_by_token = ? AND status IN ({placeholders})",
            (token_id, *_ACTIVE_STATUSES),
        )
        counts = cast(sqlite3.Row, await cursor.fetchone())
        return int(counts["jobs"]), max_concurrent

    async def get_job(self, job_id: str) -> Job | None:
        """Fetch a single job by ID, or `None` if not found."""
        cursor = await self._db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        *,
        kind: JobKind | None = None,
    ) -> list[Job]:
        """List workloads newest-first; omitted kind includes all kinds."""
        clauses: list[str] = []
        params: list[str] = []
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await self._db.conn.execute(
            f"SELECT * FROM jobs{where} ORDER BY created_at DESC", params
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(r) for r in rows]

    @_serialized
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        container_id: str | None = None,
        exit_code: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        error_class: ErrorClass | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a job's status and optional fields (container_id, timestamps, etc.)."""
        updates = ["status = ?"]
        params: list = [status.value]

        if container_id is not None:
            updates.append("container_id = ?")
            params.append(container_id)
        if exit_code is not None:
            updates.append("exit_code = ?")
            params.append(exit_code)
        if started_at is not None:
            updates.append("started_at = ?")
            params.append(started_at.isoformat())
        if finished_at is not None:
            updates.append("finished_at = ?")
            params.append(finished_at.isoformat())
        if error_class is not None:
            updates.append("error_class = ?")
            params.append(error_class.value)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(job_id)
        await self._db.conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await self._db.conn.commit()

    @_serialized
    async def set_job_error_message(self, job_id: str, error_message: str | None) -> None:
        """Set or clear the error message without regressing concurrent lifecycle changes."""
        await self._db.conn.execute(
            "UPDATE jobs SET error_message = ? WHERE id = ?",
            (error_message, job_id),
        )
        await self._db.conn.commit()
