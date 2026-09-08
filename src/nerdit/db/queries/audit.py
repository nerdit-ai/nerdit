"""Audit-log queries and archive-first retention."""

from __future__ import annotations

import json
import sqlite3

from nerdit.db.models import AuditLogEntry

from ._base import _RETENTION_SWEEP_CHUNK, QueriesBase, _serialized


class AuditQueries(QueriesBase):
    """Append and browse audit rows, then prune only archived ranges."""

    @_serialized
    async def insert_audit_log(
        self,
        *,
        action: str,
        result: str,
        principal_id: str | None = None,
        principal_role: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        params_redacted: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        """Append an immutable audit row. Secret values must already be redacted."""
        await self._db.conn.execute(
            """INSERT INTO audit_log
               (principal_id, principal_role, action, target_type, target_id,
                params_redacted, result, status_code, request_id, idempotency_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                principal_id,
                principal_role,
                action,
                target_type,
                target_id,
                params_redacted,
                result,
                status_code,
                request_id,
                idempotency_key,
            ),
        )
        await self._db.conn.commit()

    async def list_audit_log(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        action: str | None = None,
        result: str | None = None,
        target: str | None = None,
        target_type: str | None = None,
        principal_id: str | None = None,
        action_prefix: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> tuple[list[AuditLogEntry], str | None]:
        """Page newest-first audit rows with an integer ID cursor stable across inserts.

        Filters combine with AND in SQL. Action is exact; action_prefix uses GLOB,
        whose wildcards the route grammar rejects, preserving literal % and _.
        Timestamp bounds are inclusive, space-separated UTC (YYYY-MM-DD HH:MM:SS);
        callers must normalize ISO timestamps before lexical comparison.
        """
        limit = max(1, min(limit, 200))
        clauses: list[str] = []
        params: list[object] = []
        if action:
            clauses.append("action = ?")
            params.append(action)
        if action_prefix:
            clauses.append("action GLOB ? || '*'")
            params.append(action_prefix)
        if principal_id:
            clauses.append("principal_id = ?")
            params.append(principal_id)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if until:
            clauses.append("ts <= ?")
            params.append(until)
        if result:
            clauses.append("result = ?")
            params.append(result)
        if target:
            clauses.append("target_id = ?")
            params.append(target)
        if target_type:
            clauses.append("target_type = ?")
            params.append(target_type)
        if cursor:
            try:
                cursor_id = int(cursor)
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid cursor") from exc
            clauses.append("id < ?")
            params.append(cursor_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM audit_log{where} ORDER BY id DESC LIMIT ?"
        params.append(limit + 1)
        cursor_obj = await self._db.conn.execute(sql, params)
        rows = list(await cursor_obj.fetchall())
        items = [self._row_to_audit(r) for r in rows[:limit]]
        next_cursor: str | None = None
        if len(rows) > limit and items:
            next_cursor = str(items[-1].id)
        return items, next_cursor

    async def audit_archive_range(self, cutoff: str) -> tuple[int | None, int | None, int]:
        """Return min ID, max ID and count before the space-format UTC cutoff for an archive
        snapshot.
        """
        cursor = await self._db.conn.execute(
            "SELECT MIN(id), MAX(id), COUNT(*) FROM audit_log WHERE ts < ?",
            (cutoff,),
        )
        row = await cursor.fetchone()
        if row is None:
            return (None, None, 0)
        return (row[0], row[1], row[2] or 0)

    async def fetch_audit_for_archive(
        self,
        cutoff: str,
        after_id: int = 0,
        max_id: int | None = None,
        limit: int = _RETENTION_SWEEP_CHUNK,
    ) -> list[AuditLogEntry]:
        """Read ascending audit rows after after_id and before the cutoff, without a write lock.

        max_id pins the pre-stream snapshot so concurrent inserts survive this sweep.
        """
        clauses = ["ts < ?", "id > ?"]
        params: list[object] = [cutoff, after_id]
        if max_id is not None:
            clauses.append("id <= ?")
            params.append(max_id)
        params.append(limit)
        cursor = await self._db.conn.execute(
            f"SELECT * FROM audit_log WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_audit(r) for r in rows]

    @_serialized
    async def delete_audit_range(self, max_id: int, cutoff: str) -> int:
        """Delete at most 500 rows before the cutoff and at or below the fsynced watermark.

        Call only after archive fsync. The UTC cutoff uses YYYY-MM-DD HH:MM:SS;
        higher concurrent IDs survive. Return the deleted count for chunked sweeping.
        """
        cursor = await self._db.conn.execute(
            """DELETE FROM audit_log WHERE id IN (
                   SELECT id FROM audit_log
                   WHERE id <= ? AND ts < ?
                   LIMIT ?)""",
            (max_id, cutoff, _RETENTION_SWEEP_CHUNK),
        )
        await self._db.conn.commit()
        return cursor.rowcount or 0

    @staticmethod
    def _row_to_audit(row: sqlite3.Row) -> AuditLogEntry:
        """Convert an aiosqlite Row to an `AuditLogEntry`.

        `params_redacted` is stored as JSON text (already secret-masked at the
        write site); it is parsed back to a structured value, degrading to the
        raw string if it is not valid JSON.
        """
        r = row
        raw_params = r["params_redacted"]
        params: object | None
        if raw_params is None:
            params = None
        else:
            try:
                params = json.loads(raw_params)
            except (TypeError, ValueError):
                params = raw_params
        return AuditLogEntry(
            id=r["id"],
            ts=r["ts"],
            principal_id=r["principal_id"],
            principal_role=r["principal_role"],
            action=r["action"],
            target_type=r["target_type"],
            target_id=r["target_id"],
            params_redacted=params,
            result=r["result"],
            status_code=r["status_code"],
            request_id=r["request_id"],
            idempotency_key=r["idempotency_key"],
        )
