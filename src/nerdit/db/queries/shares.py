"""Store whether and how services are shared; the cloud edge controls audience.

Read each stream uncached for immediate fail-closed unsharing; list views batch
all rows. Partial service-name uniqueness prevents a foreign key, so writes
check the authorized live job and checked deletion cascades under the write lock.
"""

from __future__ import annotations

import sqlite3

from nerdit.db.rows import ServiceShare

from ._base import QueriesBase, _serialized


class ShareQueries(QueriesBase):
    """Read / upsert / delete the `service_shares` rows."""

    async def get_service_share(self, service_name: str) -> ServiceShare | None:
        """The service's share row, or `None` when it is not shared.

        `None` is the fail-closed answer the mux's app-stream resolver acts
        on, so this must stay a plain, uncached point read: an unshare closes
        the hosted path on the very next stream.
        """
        cursor = await self._db.conn.execute(
            "SELECT service_name, access, created_at FROM service_shares WHERE service_name = ?",
            (service_name,),
        )
        row = await cursor.fetchone()
        return self._row_to_share(row) if row is not None else None

    async def list_service_shares(self) -> dict[str, ServiceShare]:
        """Return all shares keyed by service name in one unpaginated read for list projections."""
        cursor = await self._db.conn.execute(
            "SELECT service_name, access, created_at FROM service_shares"
        )
        rows = await cursor.fetchall()
        shares = [self._row_to_share(r) for r in rows]
        return {share.service_name: share for share in shares}

    @_serialized
    async def set_service_share(
        self, service_name: str, access: str, *, job_id: str, preserve_existing: bool = False
    ) -> ServiceShare | None:
        """Upsert access while preserving the share's original creation time.

        Require the authorized live job_id inside the locked statement: a deleted or
        same-name replacement service must never inherit stale sharing intent. Return
        None when that job vanished; route validation and the column CHECK enforce access.
        preserve_existing retains access on conflict inside the same statement.
        """
        cursor = await self._db.conn.execute(
            """INSERT INTO service_shares (service_name, access)
               SELECT ?, ? WHERE EXISTS (
                   SELECT 1 FROM jobs
                   WHERE id = ? AND service_name = ? AND kind = 'service'
               )
               ON CONFLICT(service_name) DO UPDATE SET access =
                   CASE WHEN ? THEN service_shares.access ELSE excluded.access END""",
            (service_name, access, job_id, service_name, preserve_existing),
        )
        wrote = (cursor.rowcount or 0) > 0
        await self._db.conn.commit()
        if not wrote:
            # Zero changed rows means the ``EXISTS`` was false. Answered from
            # the statement's own count rather than by re-reading the table: a
            # pre-existing orphan row would make a re-read report success for a
            # write that never landed.
            return None
        cursor = await self._db.conn.execute(
            "SELECT service_name, access, created_at FROM service_shares WHERE service_name = ?",
            (service_name,),
        )
        row = await cursor.fetchone()
        # Committed a moment ago under the write lock this method holds.
        assert row is not None
        return self._row_to_share(row)

    @_serialized
    async def delete_service_share(self, service_name: str) -> bool:
        """Drop the share row; `True` when one was there.

        The boolean is the route's idempotency signal (`removed: false` is a
        200, never a 404) and the gate on the `share.removed` event — a
        no-op delete must not mint an edge that never happened.
        """
        cursor = await self._db.conn.execute(
            "DELETE FROM service_shares WHERE service_name = ?", (service_name,)
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @staticmethod
    def _row_to_share(row: sqlite3.Row) -> ServiceShare:
        """Convert an aiosqlite Row to a `ServiceShare`."""
        return ServiceShare(
            service_name=row["service_name"],
            access=row["access"],
            created_at=row["created_at"],
        )
