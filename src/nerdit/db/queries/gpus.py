"""GPU inventory and refcount-allocation queries."""

from __future__ import annotations

import sqlite3
from typing import cast

from nerdit.db.models import Gpu, GpuDiscoveryBackend, GpuStatus, GpuVendor

from ._base import QueriesBase, _serialized


class GpuQueries(QueriesBase):
    """GPU discovery, refcount allocation, and service-placement queries."""

    async def _write_gpu(self, gpu: Gpu) -> None:
        await self._db.conn.execute(
            """INSERT INTO gpus
                 (id, name, memory_mb, compute_cap, vendor, device_index, runtime_id,
                  discovery_backend, schedulable, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name,
                 memory_mb=excluded.memory_mb,
                 compute_cap=excluded.compute_cap,
                 vendor=excluded.vendor,
                 device_index=excluded.device_index,
                 runtime_id=excluded.runtime_id,
                 discovery_backend=excluded.discovery_backend,
                 schedulable=excluded.schedulable,
                 status=excluded.status""",
            (
                gpu.id,
                gpu.name,
                gpu.memory_mb,
                gpu.compute_cap,
                gpu.vendor.value,
                gpu.device_index,
                gpu.runtime_id or gpu.id,
                gpu.discovery_backend.value,
                int(gpu.schedulable),
                gpu.status.value,
            ),
        )

    @staticmethod
    def _gpu_from_row(row) -> Gpu:
        return Gpu(
            id=row["id"],
            name=row["name"],
            memory_mb=row["memory_mb"],
            compute_cap=row["compute_cap"],
            vendor=GpuVendor(row["vendor"]),
            device_index=row["device_index"],
            runtime_id=row["runtime_id"],
            discovery_backend=GpuDiscoveryBackend(row["discovery_backend"]),
            schedulable=bool(row["schedulable"]),
            status=GpuStatus(row["status"]),
        )

    async def list_gpus(self) -> list[Gpu]:
        """Return all registered GPUs."""
        cursor = await self._db.conn.execute("SELECT * FROM gpus")
        rows = await cursor.fetchall()
        return [self._gpu_from_row(row) for row in rows]

    @_serialized
    async def reconcile_gpus(self, gpus: list[Gpu]) -> None:
        """Upsert detected GPUs and mark devices missing from this scan offline."""
        detected_ids = {gpu.id for gpu in gpus}
        cursor = await self._db.conn.execute("SELECT rowid, * FROM gpus ORDER BY rowid")
        existing_rows = await cursor.fetchall()
        existing_ids = {row["id"] for row in existing_rows}
        legacy_rows = [
            row
            for row in existing_rows
            if row["device_index"] is None and row["id"] not in detected_ids
        ]
        legacy_by_vendor_index = {
            (vendor, index): row
            for vendor in {row["vendor"] for row in legacy_rows}
            for index, row in enumerate(
                [candidate for candidate in legacy_rows if candidate["vendor"] == vendor]
            )
        }

        for gpu in gpus:
            if gpu.id not in existing_ids:
                matching_runtime = next(
                    (
                        row
                        for row in legacy_rows
                        if row["vendor"] == gpu.vendor.value and row["runtime_id"] == gpu.runtime_id
                    ),
                    None,
                )
                legacy = matching_runtime
                if legacy is None and gpu.device_index is not None:
                    candidate = legacy_by_vendor_index.get((gpu.vendor.value, gpu.device_index))
                    legacy = candidate if candidate in legacy_rows else None
                if legacy is not None:
                    await self._write_gpu(gpu)
                    await self._db.conn.execute(
                        "UPDATE gpu_allocations SET gpu_id = ? WHERE gpu_id = ?",
                        (gpu.id, legacy["id"]),
                    )
                    await self._db.conn.execute("DELETE FROM gpus WHERE id = ?", (legacy["id"],))
                    legacy_rows.remove(legacy)
                    existing_ids.add(gpu.id)
                    continue
            await self._write_gpu(gpu)
        if detected_ids:
            placeholders = ",".join("?" for _ in detected_ids)
            await self._db.conn.execute(
                f"UPDATE gpus SET status = 'offline' WHERE id NOT IN ({placeholders})",
                tuple(sorted(detected_ids)),
            )
        else:
            await self._db.conn.execute("UPDATE gpus SET status = 'offline'")
        await self._db.conn.commit()

    async def _derive_gpu_status(self, gpu_id: str) -> None:
        """Recompute a GPU's status from its live allocations.

        Status is *derived*, never written as a literal by callers:
        `busy` = at least one exclusive allocation, `shared` = one or more
        non-exclusive allocations and no exclusive one, `idle` = no
        allocations. `error` and `offline` are out-of-pool states owned by
        discovery/repair and are left untouched.

        Must be called inside an open `BEGIN IMMEDIATE` transaction.
        """
        cursor = await self._db.conn.execute("SELECT status FROM gpus WHERE id = ?", (gpu_id,))
        row = await cursor.fetchone()
        if row is None or row["status"] in ("error", "offline"):
            return
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN exclusive = 1 THEN 1 ELSE 0 END), 0) AS excl "
            "FROM gpu_allocations WHERE gpu_id = ?",
            (gpu_id,),
        )
        counts = cast(sqlite3.Row, await cursor.fetchone())
        total, excl = counts["total"], counts["excl"]
        if total == 0:
            status = GpuStatus.idle.value
        elif excl >= 1:
            status = GpuStatus.busy.value
        else:
            status = GpuStatus.shared.value
        await self._db.conn.execute(
            "UPDATE gpus SET status = ? WHERE id = ?",
            (status, gpu_id),
        )

    @_serialized
    async def allocate_gpus(
        self, job_id: str, gpu_ids: list[str], *, exclusive: bool = True
    ) -> None:
        """Reserve GPUs atomically under BEGIN IMMEDIATE.

        Exclusive allocation requires idle devices; shared allocation also accepts
        shared devices. Derive status from the resulting allocation set.

        Raises:
            RuntimeError: Any requested GPU is not allocatable.
        """
        eligible = "('idle')" if exclusive else "('idle', 'shared')"
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            # Re-check that requested GPUs are still allocatable inside the transaction.
            placeholders = ",".join("?" for _ in gpu_ids)
            cursor = await self._db.conn.execute(
                f"SELECT id FROM gpus WHERE id IN ({placeholders}) "
                f"AND schedulable = 1 AND status IN {eligible}",
                gpu_ids,
            )
            available = {row["id"] for row in await cursor.fetchall()}
            if not all(gid in available for gid in gpu_ids):
                await self._db.conn.execute("ROLLBACK")
                raise RuntimeError(
                    f"GPU allocation conflict: some GPUs are not allocatable for job {job_id}"
                )

            for gpu_id in gpu_ids:
                await self._db.conn.execute(
                    "INSERT INTO gpu_allocations (job_id, gpu_id, exclusive) VALUES (?, ?, ?)",
                    (job_id, gpu_id, 1 if exclusive else 0),
                )
                await self._derive_gpu_status(gpu_id)
            await self._db.conn.execute("COMMIT")
        except Exception:
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    @_serialized
    async def release_gpus(self, job_id: str) -> None:
        """Release a job's GPU allocations and recompute each device's status.

        Uses `BEGIN IMMEDIATE` for atomicity. A GPU returns to `idle` only
        when its *last* allocation is removed; if another (non-exclusive)
        allocation remains it stays `shared`.
        """
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.conn.execute(
                "SELECT gpu_id FROM gpu_allocations WHERE job_id = ?", (job_id,)
            )
            gpu_ids = [row["gpu_id"] for row in await cursor.fetchall()]
            await self._db.conn.execute("DELETE FROM gpu_allocations WHERE job_id = ?", (job_id,))
            for gpu_id in gpu_ids:
                await self._derive_gpu_status(gpu_id)
            await self._db.conn.execute("COMMIT")
        except Exception:
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    async def get_job_gpus(self, job_id: str) -> list[str]:
        """Return the list of GPU IDs currently allocated to a job."""
        cursor = await self._db.conn.execute(
            "SELECT gpu_id FROM gpu_allocations WHERE job_id = ?", (job_id,)
        )
        rows = await cursor.fetchall()
        return [row["gpu_id"] for row in rows]

    async def get_service_placement_gpus(self, vendor: GpuVendor | None = None) -> list[Gpu]:
        """Return schedulable GPUs eligible for a *shared* service placement.

        Candidates are devices in `idle` or `shared` state (services use the
        non-exclusive refcount path), returned least-loaded first — fewest live
        allocations, `idle` (0) ahead of `shared` — so service placement
        prefers empty devices and does not starve batch jobs of idle GPUs.
        """
        clauses = ["g.status IN ('idle', 'shared')", "g.schedulable = 1"]
        params: list[object] = []
        if vendor is not None:
            clauses.append("g.vendor = ?")
            params.append(vendor.value)
        where = " WHERE " + " AND ".join(clauses)
        sql = (
            f"SELECT g.* FROM gpus g{where} "
            "ORDER BY (SELECT COUNT(*) FROM gpu_allocations a WHERE a.gpu_id = g.id) ASC, g.id ASC"
        )
        cursor = await self._db.conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._gpu_from_row(r) for r in rows]
