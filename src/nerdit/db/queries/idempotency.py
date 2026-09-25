"""Idempotency-key queries."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from nerdit.db.models import IdempotencyRecord

from ._base import QueriesBase, _serialized


class IdempotencyQueries(QueriesBase):
    """Idempotency-key claim/complete/sweep queries."""

    @staticmethod
    def _row_to_idempotency(row: sqlite3.Row) -> IdempotencyRecord:
        """Convert an aiosqlite Row to an `IdempotencyRecord`."""
        r = row
        return IdempotencyRecord(
            principal_id=r["principal_id"],
            idem_key=r["idem_key"],
            method=r["method"],
            path=r["path"],
            state=r["state"],
            response_status=r["response_status"],
            response_body=r["response_body"],
            content_type=r["content_type"],
            resource_id=r["resource_id"],
            body_hash=r["body_hash"],
            created_at=r["created_at"],
            expires_at=r["expires_at"],
        )

    @_serialized
    async def insert_idempotency_inprogress(
        self,
        *,
        principal_id: str,
        idem_key: str,
        method: str,
        path: str,
        expires_at: str,
        body_hash: str | None = None,
    ) -> bool:
        """Claim a principal/key pair using BEGIN IMMEDIATE and INSERT OR IGNORE.

        Do not call inside another transaction. Store only the optional bounded request
        body hash, never the body. An expired row never blocks a fresh claim.

        Returns:
            True if claimed; False if the caller must re-read the existing record.
        """
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            # Same cutoff as the sweep, enforced in-claim so the TTL does not
            # depend on the sweep interval.
            await self._db.conn.execute(
                """DELETE FROM idempotency_keys
                   WHERE principal_id = ? AND idem_key = ?
                     AND expires_at IS NOT NULL AND expires_at < ?""",
                (principal_id, idem_key, datetime.now(UTC).isoformat()),
            )
            cursor = await self._db.conn.execute(
                """INSERT OR IGNORE INTO idempotency_keys
                   (principal_id, idem_key, method, path, state, expires_at, body_hash)
                   VALUES (?, ?, ?, ?, 'in_progress', ?, ?)""",
                (principal_id, idem_key, method, path, expires_at, body_hash),
            )
            inserted = cursor.rowcount > 0
            await self._db.conn.execute("COMMIT")
            return inserted
        except Exception:
            try:
                await self._db.conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    async def get_idempotency_record(
        self, principal_id: str, idem_key: str
    ) -> IdempotencyRecord | None:
        """Return the stored idempotency record, or `None` if absent."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM idempotency_keys WHERE principal_id = ? AND idem_key = ?",
            (principal_id, idem_key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_idempotency(row)

    @_serialized
    async def complete_idempotency_record(
        self,
        *,
        principal_id: str,
        idem_key: str,
        response_status: int,
        response_body: str | None,
        content_type: str | None,
        resource_id: str | None,
    ) -> None:
        """Pin a successful (2xx) response so future replays return it.

        `response_body` is `None` for secret-returning routes (only the
        status and `resource_id` are kept); the body is otherwise already
        run through the secret-redaction denylist by the caller.
        """
        await self._db.conn.execute(
            """UPDATE idempotency_keys
               SET state = 'completed', response_status = ?, response_body = ?,
                   content_type = ?, resource_id = ?
               WHERE principal_id = ? AND idem_key = ?""",
            (response_status, response_body, content_type, resource_id, principal_id, idem_key),
        )
        await self._db.conn.commit()

    @_serialized
    async def mark_idempotency_interrupted(self, principal_id: str, idem_key: str) -> None:
        """Mark an in-progress claim interrupted after cancellation; never demote a completed
        record.
        """
        await self._db.conn.execute(
            """UPDATE idempotency_keys SET state = 'interrupted'
               WHERE principal_id = ? AND idem_key = ? AND state = 'in_progress'""",
            (principal_id, idem_key),
        )
        await self._db.conn.commit()

    @_serialized
    async def delete_idempotency_record(self, principal_id: str, idem_key: str) -> None:
        """Remove an idempotency row (used to un-pin a non-2xx outcome)."""
        await self._db.conn.execute(
            "DELETE FROM idempotency_keys WHERE principal_id = ? AND idem_key = ?",
            (principal_id, idem_key),
        )
        await self._db.conn.commit()

    @_serialized
    async def sweep_expired_idempotency(self, now: str | None = None) -> int:
        """Delete idempotency rows past their `expires_at`. Returns the count."""
        cutoff = now if now is not None else datetime.now(UTC).isoformat()
        cursor = await self._db.conn.execute(
            "DELETE FROM idempotency_keys WHERE expires_at IS NOT NULL AND expires_at < ?",
            (cutoff,),
        )
        await self._db.conn.commit()
        return cursor.rowcount or 0
