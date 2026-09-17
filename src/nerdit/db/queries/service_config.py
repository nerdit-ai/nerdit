"""Service config writes with shared transactional GPU-quota checks."""

from __future__ import annotations

import json
from datetime import datetime

from nerdit.db.models import ErrorClass, JobStatus

from ._base import _ACTIVE_STATUSES, QueriesBase, _serialized

_LAST_DUMP_PATCH_PATHS = {"dump": "'$.last_dump.dump'", "reason": "'$.last_dump.reason'"}

#: The row's current build-version allocator, read at UPDATE time. Mirrors the
#: deploy pipeline's ``int(prev_cfg.get("max_version", prev_cfg.get("build_version", 0)))``
#: exactly, including the tolerant fallbacks: a NULL/invalid config blob parses
#: to ``{}`` there and must compare as 0 here, and a rollback lowers
#: ``build_version`` while leaving ``max_version`` as the allocator. CAST keeps a
#: version stored as text comparing equal to the integer the caller read.
_MAX_VERSION_EXPR = (
    "CASE WHEN json_valid(config) THEN CAST(COALESCE("
    "json_extract(config, '$.max_version'), json_extract(config, '$.build_version'), 0"
    ") AS INTEGER) ELSE 0 END"
)

#: 1 when the row carries no cutover marker, 0 when it does, read at UPDATE
#: time. An invalid blob reads as "absent" so it matches the tolerant `{}` a
#: caller's `parse_job_config` produced for the same row.
_CUTOVER_ABSENT_EXPR = (
    "CASE WHEN json_valid(config) AND json_extract(config, '$.cutover_pending') "
    "IS NOT NULL THEN 0 ELSE 1 END"
)


def _redeploy_sets(
    *,
    config: str,
    status: JobStatus,
    desired_state: str,
    gpu_count: int | None,
    health_check: dict | None,
) -> tuple[list[str], list[object]]:
    """Build the SET clause shared by the guarded and unguarded config rewrites.

    Rewrite GPU/health columns only when supplied, so a caller that passes
    neither leaves them untouched.
    """
    sets = [
        "config = ?",
        "status = ?",
        "desired_state = ?",
        "restart_count = 0",
        "last_exit_at = NULL",
        "restart_window_start = NULL",
        "finished_at = NULL",
        "exit_code = NULL",
        # F2-STALE-ERR: a redeploy/rollback that converges healthy must not
        # surface a prior generation's error through the row-column-precedence
        # reads in _build_wait_response / diagnose_service.
        "error_class = NULL",
        "error_message = NULL",
    ]
    params: list[object] = [config, status.value, desired_state]
    if gpu_count is not None:
        sets.append("gpu_count = ?")
        params.append(gpu_count)
    if health_check is not None:
        sets.append("health_check = ?")
        params.append(json.dumps(health_check))
    return sets, params


class ServiceConfigQueries(QueriesBase):
    """Service/app config rewrites, including the guarded revert and quota re-check."""

    async def _recheck_gpu_quota_locked(self, token_id: str, job_id: str, gpu_count: int) -> None:
        """Check the owner's GPU cap inside the caller's open BEGIN IMMEDIATE transaction.

        Exclude this job's existing footprint to avoid double charging. Raise
        QuotaExceeded on breach; never open, commit or roll back the transaction here.
        """
        from nerdit.daemon.auth import QuotaExceeded

        cursor = await self._db.conn.execute(
            "SELECT max_gpus FROM api_tokens WHERE id = ? AND revoked = 0",
            (token_id,),
        )
        caps = await cursor.fetchone()
        max_gpus = caps["max_gpus"] if caps else None
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        cursor = await self._db.conn.execute(
            f"SELECT COALESCE(SUM(gpu_count), 0) AS gpus FROM jobs "
            f"WHERE submitted_by_token = ? AND id != ? "
            f"AND status IN ({placeholders})",
            (token_id, job_id, *_ACTIVE_STATUSES),
        )
        row = await cursor.fetchone()
        sum_excl = row["gpus"] if row else 0
        if max_gpus is not None and sum_excl + gpu_count > max_gpus:
            raise QuotaExceeded("max_gpus", limit=max_gpus, current=sum_excl)

    @_serialized
    async def update_service_config(
        self,
        job_id: str,
        config: str,
        *,
        status: JobStatus,
        desired_state: str,
        gpu_count: int | None = None,
        health_check: dict | None = None,
        token_id: str | None = None,
    ) -> None:
        """Replace deploy config and clear restart bookkeeping for redeploy or rollback.

        Preserve row identity and its durable endpoint. Rewrite GPU/health columns only
        when supplied. When token_id and gpu_count are set, recheck the owner's GPU
        quota and update in one BEGIN IMMEDIATE transaction, excluding this row's old
        footprint. Concurrency count does not change. Do not nest in another transaction.
        """
        sets, params = _redeploy_sets(
            config=config,
            status=status,
            desired_state=desired_state,
            gpu_count=gpu_count,
            health_check=health_check,
        )
        params.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?"

        if token_id is not None and gpu_count is not None:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._recheck_gpu_quota_locked(token_id, job_id, gpu_count)
                await self._db.conn.execute(sql, params)
                await self._db.conn.execute("COMMIT")
            except Exception:
                try:
                    await self._db.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            return

        await self._db.conn.execute(sql, params)
        await self._db.conn.commit()

    @_serialized
    async def update_service_config_guarded(
        self,
        job_id: str,
        config: str,
        *,
        expect_max_version: int,
        expect_cutover_pending: bool = False,
        status: JobStatus,
        desired_state: str,
        gpu_count: int | None = None,
        health_check: dict | None = None,
        token_id: str | None = None,
    ) -> bool:
        """Rewrite deploy config only while the expected build-version allocator owns the row.

        The redeploy twin of :meth:`update_service_config`: the new generation's
        version is derived from a config read that happens well before this write,
        and nothing serializes the two, so two overlapping redeploys would both
        stamp the same version and the loser's image tag would name code that is
        never built. Return False on that supersession; the caller must refuse its
        own deploy and clean up its build context, never retry unguarded.

        `expect_cutover_pending` states whether the caller's own read saw a
        cutover marker. The route's in-flight 409 is a task-registry check, and
        `CutoverManager._arm` commits the marker, releases the write lock and
        awaits before it registers the task — so a redeploy that passed the 409
        can still take the lock next, pop a LIVE marker and leave the verify
        probing a superseded row. Guarding on "the marker is in the state I read
        it in" refuses exactly that interleaving, and still lets a redeploy land
        over a stale marker left by an abandoned verify.

        Raise QuotaExceeded exactly as the unguarded twin does; a quota breach is a
        different answer from a lost race and must not be flattened into False.
        """
        sets, params = _redeploy_sets(
            config=config,
            status=status,
            desired_state=desired_state,
            gpu_count=gpu_count,
            health_check=health_check,
        )
        params.extend((job_id, expect_max_version, 0 if expect_cutover_pending else 1))
        sql = (
            f"UPDATE jobs SET {', '.join(sets)} "
            f"WHERE id = ? AND {_MAX_VERSION_EXPR} = ? AND {_CUTOVER_ABSENT_EXPR} = ?"
        )

        if token_id is not None and gpu_count is not None:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._recheck_gpu_quota_locked(token_id, job_id, gpu_count)
                cursor = await self._db.conn.execute(sql, params)
                await self._db.conn.execute("COMMIT")
            except Exception:
                try:
                    await self._db.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            return cursor.rowcount == 1

        cursor = await self._db.conn.execute(sql, params)
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def revert_service_config_guarded(
        self,
        job_id: str,
        config: str,
        *,
        expect_build_version: int,
        status: JobStatus,
        desired_state: str,
    ) -> bool:
        """Revert config and restart bookkeeping only while the expected build version still owns
        the row.

        Return False on supersession; the caller must abandon the stale revert.
        """
        sets = [
            "config = ?",
            "status = ?",
            # Preserve operator stops during autonomous revert with one atomic CASE.
            # Explicit redeploy differs: it intentionally sets desired_state to running.
            "desired_state = CASE WHEN desired_state IN ('stopped', 'failed')"
            " THEN desired_state ELSE ? END",
            "restart_count = 0",
            "last_exit_at = NULL",
            "restart_window_start = NULL",
            "finished_at = NULL",
            "exit_code = NULL",
            # F2-STALE-ERR: a revert to the previous healthy image must not leave
            # the failed-build error columns set (they would leak through the
            # row-column-precedence reads).
            "error_class = NULL",
            "error_message = NULL",
        ]
        params: list[object] = [
            config,
            status.value,
            desired_state,
            job_id,
            expect_build_version,
        ]
        sql = (
            f"UPDATE jobs SET {', '.join(sets)} "
            "WHERE id = ? AND json_extract(config, '$.build_version') = ?"
        )
        cursor = await self._db.conn.execute(sql, params)
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def update_app_config(
        self,
        job_id: str,
        config: str,
        *,
        gpu_count: int | None = None,
        health_check: dict | None = None,
        clear_health: bool = False,
        token_id: str | None = None,
    ) -> None:
        """Update app config without resetting lifecycle or implicitly restarting it.

        Rewrite GPU/health only when supplied; clear_health sets health to NULL.
        GPU quota recheck and update are atomic and exclude the row's current footprint.
        """
        sets = ["config = ?"]
        params: list[object] = [config]
        if gpu_count is not None:
            sets.append("gpu_count = ?")
            params.append(gpu_count)
        if health_check is not None:
            sets.append("health_check = ?")
            params.append(json.dumps(health_check))
        elif clear_health:
            sets.append("health_check = NULL")
        params.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?"

        if token_id is not None and gpu_count is not None:
            await self._db.conn.execute("BEGIN IMMEDIATE")
            try:
                await self._recheck_gpu_quota_locked(token_id, job_id, gpu_count)
                await self._db.conn.execute(sql, params)
                await self._db.conn.execute("COMMIT")
            except Exception:
                try:
                    await self._db.conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            return

        await self._db.conn.execute(sql, params)
        await self._db.conn.commit()

    @_serialized
    async def update_job_config(self, job_id: str, config: str) -> None:
        """Rewrite only the config blob, preserving lifecycle and restart bookkeeping."""
        await self._db.conn.execute("UPDATE jobs SET config = ? WHERE id = ?", (config, job_id))
        await self._db.conn.commit()

    @_serialized
    async def set_last_run(self, job_id: str, last_run: str) -> bool:
        """Set only config.last_run, preserving concurrent deploy and release markers.

        Use a literal json_set path against the current row rather than stale whole-blob
        writes. Invalid JSON skips the update and returns False; the run result still
        reaches its caller.
        """
        cursor = await self._db.conn.execute(
            "UPDATE jobs SET config = json_set(config, '$.last_run', json(?)) "
            "WHERE id = ? AND json_valid(config)",
            (last_run, job_id),
        )
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def set_last_dump(self, job_id: str, last_dump: str) -> bool:
        """Write ``config['last_dump']`` WITHOUT rewriting the rest of the blob.

        (P37 / D-P37-11) The exact twin of :meth:`set_last_run`, for the same
        reason: a dump or a restore is long-lived (an hour is a legal
        ``timeout_s``) and nothing stops a redeploy, a release or a reconcile
        from committing to the SAME row while the sibling container is still
        streaming. A whole-blob read-modify-write would read the pre-commit
        blob, await, and write it back over the newer one — erasing ``image`` /
        ``build_version`` / ``release_pending``. ``json_set`` rewrites one path
        inside the value the row holds AT UPDATE TIME, so there is no read step
        to go stale; the path is a literal, never caller-derived.

        ``json_valid`` guards the one case ``json_set`` would answer NULL and
        blank the column. ``False`` therefore means "the row is gone, or its
        blob is not valid JSON" — the dump's own result still reaches the
        caller either way, because this stamp is bookkeeping, not the outcome.
        """
        cursor = await self._db.conn.execute(
            "UPDATE jobs SET config = json_set(config, '$.last_dump', json(?)) "
            "WHERE id = ? AND json_valid(config)",
            (last_dump, job_id),
        )
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def patch_last_dump_field(
        self, job_id: str, *, field: str, value: str, expect_run_id: str
    ) -> bool:
        """Set one ``config['last_dump']`` key, only while the blob still belongs
        to ``expect_run_id`` (D-P37-11).

        The route patches after :meth:`ServiceController.dump_database` released
        the row's slot, so a later dump or restore may own ``last_dump`` by then;
        the run-id guard makes that a no-op instead of an overwrite. ``field`` is
        looked up in a literal path table, never interpolated from the caller.
        Returns ``False`` when nothing matched.
        """
        path = _LAST_DUMP_PATCH_PATHS[field]
        cursor = await self._db.conn.execute(
            f"UPDATE jobs SET config = json_set(config, {path}, ?) "
            "WHERE id = ? AND json_valid(config) "
            "AND json_extract(config, '$.last_dump.run_id') = ?",
            (value, job_id, expect_run_id),
        )
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def update_job_config_guarded(
        self, job_id: str, config: str, *, expect_build_version: int
    ) -> bool:
        """Update only the config blob while its build version matches the expected version.

        Return False on missing/mismatched versions. Callers must abandon superseded
        writes, never retry them unguarded; lifecycle columns remain unchanged.
        """
        cursor = await self._db.conn.execute(
            "UPDATE jobs SET config = ? "
            "WHERE id = ? AND json_extract(config, '$.build_version') = ?",
            (config, job_id, expect_build_version),
        )
        await self._db.conn.commit()
        return cursor.rowcount == 1

    @_serialized
    async def settle_failed_guarded(
        self,
        job_id: str,
        *,
        expect_build_version: int,
        finished_at: datetime,
        exit_code: int,
        error_class: ErrorClass,
        error_message: str,
    ) -> bool:
        """Atomically mark one deploy generation failed with its forensic fields.

        Return False if a newer build version owns the row. Missing build_version is
        accepted for prebuilt/legacy rows whose expected version comes from the image
        tag; every redeploy writes a version, so absence cannot hide a newer generation.
        Leave the config blob and release_pending crash marker intact.
        """
        cursor = await self._db.conn.execute(
            "UPDATE jobs SET status = ?, desired_state = ?, finished_at = ?, exit_code = ?, "
            "error_class = ?, error_message = ? "
            "WHERE id = ? AND (json_extract(config, '$.build_version') = ? "
            "OR json_extract(config, '$.build_version') IS NULL)",
            (
                JobStatus.failed.value,
                JobStatus.failed.value,
                finished_at.isoformat(),
                exit_code,
                error_class.value,
                error_message,
                job_id,
                expect_build_version,
            ),
        )
        await self._db.conn.commit()
        return cursor.rowcount == 1
