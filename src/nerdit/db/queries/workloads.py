"""Workload row queries and atomic quota-checked reservations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import cast

from nerdit.core.project_identity import DEFAULT_SERVICE, PRODUCTION, mint_project_id
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

from ._base import (
    _ACTIVE_STATUSES,
    _JOB_SELECT,
    ProjectOwned,
    QueriesBase,
    ServiceNameClaimed,
    ServiceNameTaken,
    _serialized,
)


class WorkloadQueries(QueriesBase):
    """Job-row CRUD + the atomic quota-checked reserve path."""

    # --- Job operations ---

    async def _stamp_project(self, job: Job) -> None:
        """Give a tripleless service row its project triple before the INSERT (P40a).

        The legacy mapping: ``service_name`` is the project name and the row is
        ``(project, production, web)`` -- the label rule's identity case
        (D-P40-2), the same mapping ``_migrate``'s backfill applies. The
        ``projects`` row is find-or-created by name with the job's token as
        owner and no judgment (D-P40-5: P40b adds the refusals). A row that
        arrives with ``project_id`` set (P40d's composed rows) is left alone.

        Runs inside whatever transaction the caller holds: ``BEGIN IMMEDIATE``
        under ``reserve_service_for_token``, the implicit DML transaction under
        ``create_job``. Either way a failing jobs INSERT rolls the fresh
        ``projects`` row back with it, so no orphan can outlive the statement.
        """
        # ponytail: stamps the caller's Job in place, so a refused INSERT leaves
        # a rolled-back id on the object; no caller re-inserts the same Job
        # today -- return the triple instead if one ever does.
        if job.kind is not JobKind.service or job.service_name is None or job.project_id:
            return
        cursor = await self._db.conn.execute(
            "SELECT id FROM projects WHERE name = ?", (job.service_name,)
        )
        row = await cursor.fetchone()
        if row is None:
            job.project_id = mint_project_id()
            await self._db.conn.execute(
                "INSERT INTO projects (id, name, submitted_by_token) VALUES (?, ?, ?)",
                (job.project_id, job.service_name, job.submitted_by_token),
            )
        else:
            job.project_id = row["id"]
        job.environment = PRODUCTION
        job.service = DEFAULT_SERVICE
        # The in-memory row is what the create routes project straight back
        # (no re-read), so the JOIN-derived name is filled here too (P40b).
        job.project = job.service_name

    async def _exec_job_insert(self, job: Job) -> None:
        """Execute the `jobs` INSERT for *job* (no commit / transaction control).

        Shared by `create_job` (which commits immediately) and
        `reserve_service_for_token` (which runs it inside an open
        `BEGIN IMMEDIATE` quota transaction). Both stamp the project triple
        through `_stamp_project` first.
        """
        await self._stamp_project(job)
        await self._db.conn.execute(
            """INSERT INTO jobs
               (id, name, script_path, gpu_count, status,
                container_id, created_at, started_at, finished_at,
                exit_code, config,
                error_class, error_message, kind,
                submitted_by_token, idempotency_key,
                desired_state, restart_policy, restart_count, health_check,
                service_name, last_exit_at, restart_window_start,
                project_id, environment, service)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?)""",
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
                job.project_id,
                job.environment,
                job.service,
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
    async def reserve_service_for_token(self, job: Job, *, admin: bool = False) -> Job:
        """Check token quotas, consume the name's secret claim and insert a workload
        atomically under BEGIN IMMEDIATE.

        Count all active kinds, including transient service states. Legacy/local
        submitted_by_token=None is uncapped. Do not call inside another transaction.

        This transaction is the security boundary for P39 claims: the routes'
        pre-ingress claim checks are optimizations, and a claim minted between
        that check and this insert is still honoured here. `admin` bypasses the
        claim the way it bypasses row ownership; it defaults to `False` so a
        route that forgets it fails closed. A NULL-token claim (LOCAL/LEGACY_ADMIN
        claimant, or a claim whose token was revoked) is foreign to every
        non-admin, so a revoked claimant cannot be impersonated by another
        tokenless principal.

        The project judgment (P40b / D-P40-5 rule 1) runs after the claim step,
        under the same lock, for every kind: the row's project name is its
        label (kind=service: the legacy triple `_stamp_project` resolves;
        models and databases: the label itself), and a `projects` row of that
        name owned by another token refuses the insert before `_stamp_project`
        could adopt it. A NULL-owner project is foreign to every non-admin.
        A declared service (P40d: `project_id` preset by `apply_project`) is
        judged on the project it joins, by id, as well; a preset id whose
        project is gone refuses every caller.

        Raises:
            QuotaExceeded: The token's active footprint exceeds a cap.
            ServiceNameClaimed: Another token set this name's secrets first (D-P39-3).
            ProjectOwned: Another token owns the project this row would join (D-P40-5).
            ServiceNameTaken: The unique service name collided; the transaction rolls back.
        """
        import sqlite3

        token_id = job.submitted_by_token
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            if token_id is not None:
                await self._check_quota_locked(token_id, job.gpu_count, _ACTIVE_STATUSES)
            cursor = await self._db.conn.execute(
                "SELECT token_id FROM secret_claims WHERE service_name = ?", (job.service_name,)
            )
            claim = await cursor.fetchone()
            if claim is not None:
                claimant = claim["token_id"]
                if not admin and (claimant is None or claimant != token_id):
                    raise ServiceNameClaimed(job.service_name)
                await self._db.conn.execute(
                    "DELETE FROM secret_claims WHERE service_name = ?", (job.service_name,)
                )
            # (P40d) A row arriving with `project_id` preset is a declared
            # service: the project it JOINS is judged by that id, never by
            # parsing its label (D-P40-2). The project NAMED after the label is
            # still judged for every row: a composed label `api--asso` must not
            # squat a foreign implicit project literally named `api--asso`.
            if job.project_id:
                cursor = await self._db.conn.execute(
                    "SELECT submitted_by_token FROM projects WHERE id = ?", (job.project_id,)
                )
                joined = await cursor.fetchone()
                # A project deleted since the route judged it: `jobs.project_id`
                # carries no FK, so an admin is refused too rather than left
                # with a row pointing at nothing. Re-applying re-creates it.
                if joined is None or (
                    not admin
                    and (
                        joined["submitted_by_token"] is None
                        or joined["submitted_by_token"] != token_id
                    )
                ):
                    raise ProjectOwned(job.service_name)
            if not admin and job.service_name is not None:
                cursor = await self._db.conn.execute(
                    "SELECT submitted_by_token FROM projects WHERE name = ?", (job.service_name,)
                )
                project = await cursor.fetchone()
                if project is not None and (
                    project["submitted_by_token"] is None
                    or project["submitted_by_token"] != token_id
                ):
                    raise ProjectOwned(job.service_name)
            try:
                await self._exec_job_insert(job)
            except sqlite3.IntegrityError as exc:
                # D-P40-3: the label index OR the triple index (SQLite names it
                # ``jobs.project_id, jobs.environment, jobs.service``) is a
                # name collision; the PK error ``jobs.id`` matches neither.
                msg = str(exc)
                if "service_name" in msg or "jobs.project_id" in msg:
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
        cursor = await self._db.conn.execute(f"{_JOB_SELECT} WHERE jobs.id = ?", (job_id,))
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
            f"{_JOB_SELECT}{where} ORDER BY jobs.created_at DESC", params
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
        desired_state: str | None = None,
    ) -> None:
        """Update a job's status and optional fields (container_id, timestamps, etc.).

        Pass ``desired_state`` to settle both columns in one statement, so a crash
        cannot leave a terminal status with ``desired_state = running``.
        """
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
        if desired_state is not None:
            updates.append("desired_state = ?")
            params.append(desired_state)

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
