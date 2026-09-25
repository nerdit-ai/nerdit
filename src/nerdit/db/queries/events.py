"""Durable events with backwards browsing, forward replay and keep-last-N retention."""

from __future__ import annotations

import json
import logging
import sqlite3

from nerdit.db.rows import Event

from ._base import _RETENTION_SWEEP_CHUNK, QueriesBase, _serialized

logger = logging.getLogger(__name__)

#: Hard cap on the ``types`` filter list. A caller asking for more is clamped
#: (never rejected) so a chatty client degrades instead of erroring — the same
#: posture as the ``limit`` clamp.
_MAX_TYPE_FILTERS = 10


class EventQueries(QueriesBase):
    """Insert / read / prune the durable `events` feed."""

    @_serialized
    async def insert_event(
        self,
        *,
        type: str,  # noqa: A002 — the LOCKED D-P24-3 field name
        kind: str | None = None,
        service_name: str | None = None,
        reason: str | None = None,
        build_version: int | None = None,
        data: dict[str, object] | None = None,
        ts: str | None = None,
    ) -> int | None:
        """Append an event and return its cursor ID.

        Use the caller's space-format UTC timestamp so live and durable events agree;
        None uses SQLite's UTC clock. The caller enforces secret-free payload rules.
        """
        cursor = await self._db.conn.execute(
            """INSERT INTO events (type, kind, service_name, reason, build_version, data, ts)
               VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, datetime('now')))""",
            (
                type,
                kind,
                service_name,
                reason,
                build_version,
                json.dumps(data, sort_keys=True) if data is not None else None,
                ts,
            ),
        )
        await self._db.conn.commit()
        return cursor.lastrowid

    async def list_events(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        types: list[str] | None = None,
        service: str | None = None,
        since_id: int | None = None,
    ) -> tuple[list[Event], str | None]:
        """Page events backwards by cursor or forwards by since_id.

        Browse uses descending IDs below cursor; replay uses ascending IDs above since_id.
        next_cursor is the last returned ID in either direction.

        Raises:
            ValueError: Both modes were supplied, or the cursor is malformed.
        """
        limit = max(1, min(limit, 200))
        if cursor and since_id is not None:
            raise ValueError("cursor and since_id are mutually exclusive")
        clauses, params = self._event_filter_clauses(types, service)
        if since_id is not None:
            clauses.append("id > ?")
            params.append(int(since_id))
            order = "ASC"
        else:
            order = "DESC"
            if cursor:
                try:
                    cursor_id = int(cursor)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Invalid cursor") from exc
                clauses.append("id < ?")
                params.append(cursor_id)
        rows = await self._select_events(clauses, params, order, limit + 1)
        items = rows[:limit]
        next_cursor: str | None = None
        if len(rows) > limit and items:
            # DESC: the last row IS the smallest id returned; ASC: the largest.
            next_cursor = str(items[-1].id)
        return items, next_cursor

    async def replay_events_after(self, after_id: int, limit: int) -> list[Event]:
        """Read ascending events after the ID; request limit + 1 to detect replay overflow."""
        return await self._select_events(["id > ?"], [int(after_id)], "ASC", limit)

    async def last_event_id(self) -> int:
        """Highest feed id, or `0` on an empty table (the new-target cursor)."""
        return await self._scalar_int("SELECT COALESCE(MAX(id), 0) FROM events")

    async def feed_bounds(self) -> tuple[int, int]:
        """Return min/max retained IDs in one consistent statement, or (0, 0) for an empty feed."""
        cursor = await self._db.conn.execute(
            "SELECT COALESCE(MIN(id), 0), COALESCE(MAX(id), 0) FROM events"
        )
        row = await cursor.fetchone()
        return (int(row[0]), int(row[1])) if row is not None else (0, 0)

    async def event_sweep_watermark(self, keep_last: int) -> int | None:
        """Find the highest deletable ID while retaining the newest keep_last rows.

        Return None when retention is disabled or no rows qualify. Resolve once per
        sweep; holding the bound across chunks preserves concurrent inserts.
        """
        if keep_last <= 0:
            return None
        cursor = await self._db.conn.execute(
            "SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET ?", (keep_last,)
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else None

    @_serialized
    async def sweep_events(self, keep_last: int, *, watermark: int | None = None) -> int:
        """Delete at most 500 events below the keep-last-N watermark; return the count.

        No archive is required. keep_last <= 0 disables pruning. Supply the fixed
        once-per-sweep watermark, or omit it to resolve a bound for this call.
        """
        if watermark is None:
            watermark = await self.event_sweep_watermark(keep_last)
            if watermark is None:
                return 0
        cursor = await self._db.conn.execute(
            """DELETE FROM events WHERE id IN (
                   SELECT id FROM events WHERE id <= ? LIMIT ?)""",
            (watermark, _RETENTION_SWEEP_CHUNK),
        )
        await self._db.conn.commit()
        return cursor.rowcount or 0

    # --- webhook delivery cursors -----------------------------------------

    async def get_notification_cursor(self, target_id: str) -> int | None:
        """Return the durable target cursor, or None to seed a new target at the feed head."""
        cursor = await self._db.conn.execute(
            "SELECT last_id FROM notification_cursors WHERE target_id = ?", (target_id,)
        )
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else None

    @_serialized
    async def set_notification_cursor(self, target_id: str, last_id: int) -> None:
        """Persist one batch cursor, advanced on success or retry exhaustion to bound redelivery."""
        await self._db.conn.execute(
            """INSERT INTO notification_cursors (target_id, last_id, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(target_id) DO UPDATE SET
                   last_id = excluded.last_id, updated_at = excluded.updated_at""",
            (target_id, int(last_id)),
        )
        await self._db.conn.commit()

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _event_filter_clauses(
        types: list[str] | None, service: str | None
    ) -> tuple[list[str], list[object]]:
        """Shared WHERE fragments for both orderings (never fork the filters)."""
        clauses: list[str] = []
        params: list[object] = []
        if types:
            capped = list(types)[:_MAX_TYPE_FILTERS]
            placeholders = ", ".join("?" * len(capped))
            clauses.append(f"type IN ({placeholders})")
            params.extend(capped)
        if service:
            clauses.append("service_name = ?")
            params.append(service)
        return clauses, params

    async def _select_events(
        self, clauses: list[str], params: list[object], order: str, limit: int
    ) -> list[Event]:
        """Run the one bounded SELECT both read modes share."""
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM events{where} ORDER BY id {order} LIMIT ?"
        cursor = await self._db.conn.execute(sql, [*params, limit])
        return [self._row_to_event(r) for r in await cursor.fetchall()]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        """Parse event JSON; missing, invalid or non-object data becomes None without failing the
        feed.
        """
        raw = row["data"]
        data: dict[str, object] | None = None
        if raw is not None:
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                logger.debug("Event %s carries non-JSON data; degrading to null", row["id"])
            else:
                if isinstance(parsed, dict):
                    data = parsed
        return Event(
            id=row["id"],
            ts=row["ts"],
            type=row["type"],
            kind=row["kind"],
            service_name=row["service_name"],
            reason=row["reason"],
            build_version=row["build_version"],
            data=data,
        )
