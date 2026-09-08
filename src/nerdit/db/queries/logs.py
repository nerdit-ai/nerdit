"""Append, filter and prune workload logs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from nerdit.db.models import LogEntry, LogStream

from ._base import _RETENTION_SWEEP_CHUNK, QueriesBase, _serialized

#: Hard clamp on the ``grep`` needle. The route clamps too; this is the
#: last line, so a direct query-layer caller cannot smuggle an unbounded
#: pattern past it either.
_MAX_GREP_LEN = 200


def _escape_like(needle: str) -> str:
    """Escape a literal LIKE needle, truncating input to 200 characters first.

    Escape backslashes before % and _ so added escapes are not escaped again.
    Truncate before escaping to avoid splitting an escape pair.
    """
    clamped = needle[:_MAX_GREP_LEN]
    return clamped.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")


class LogQueries(QueriesBase):
    """Workload logs and chunked retention."""

    @_serialized
    async def append_log(
        self, job_id: str, message: str, stream: LogStream = LogStream.stdout
    ) -> None:
        """Append a log line (stdout, stderr, or system) for a job."""
        await self._db.conn.execute(
            "INSERT INTO job_logs (job_id, stream, message) VALUES (?, ?, ?)",
            (job_id, stream.value, message),
        )
        await self._db.conn.commit()

    @_serialized
    async def replace_crash_tail(self, job_id: str, lines: list[str]) -> None:
        """Atomically replace the latest crash capture, preserving ordinary log history.

        Serialized writes roll back exceptions. Callers tolerate a foreign-key failure
        when the job was concurrently deleted.
        """
        await self._db.conn.execute(
            "DELETE FROM job_logs WHERE job_id = ? AND stream = ?",
            (job_id, LogStream.crash.value),
        )
        await self._db.conn.executemany(
            "INSERT INTO job_logs (job_id, stream, message) VALUES (?, ?, ?)",
            [(job_id, LogStream.crash.value, line) for line in lines],
        )
        await self._db.conn.commit()

    async def get_logs(  # noqa: PLR0913 — one parameter per orthogonal log filter
        self,
        job_id: str,
        since_id: int = 0,
        *,
        tail: int | None = None,
        since_ts: str | None = None,
        grep: str | None = None,
        limit: int | None = None,
        streams: Sequence[LogStream] | None = None,
    ) -> list[LogEntry]:
        """Read filtered logs in ascending order, bounding results in SQL.

        Args:
            since_id: Exclusive forward cursor, ignored when tail is supplied.
            tail: Most recent N matching entries; takes precedence over since_id/limit.
            limit: Maximum entries for forward reads.
            since_ts: Inclusive space-format UTC lower bound (YYYY-MM-DD HH:MM:SS).
            grep: Literal substring; escape LIKE wildcards rather than interpret regex.
            streams: Bound stream-value filter, or None for all streams.

        Apply every filter before the bound; never materialize unfiltered logs in Python.
        """
        clauses = ["job_id = ?"]
        params: list[object] = [job_id]
        if streams is not None:
            placeholders = ", ".join("?" for _ in streams)
            # An EMPTY sequence is honoured as "nothing matches" (``IN ()`` is
            # not valid SQLite, so it is spelled as an always-false clause)
            # rather than silently widened to "everything" — a caller that
            # computed an empty stream set must not get the whole log back.
            clauses.append(f"stream IN ({placeholders})" if placeholders else "0")
            params.extend(st.value for st in streams)
        if since_ts:
            clauses.append("timestamp >= ?")
            params.append(since_ts)
        if grep:
            clauses.append(r"message LIKE '%' || ? || '%' ESCAPE '\'")
            params.append(_escape_like(grep))
        where = " AND ".join(clauses)

        if tail is not None and tail > 0:
            cursor = await self._db.conn.execute(
                f"""SELECT id, stream, message, timestamp FROM job_logs
                    WHERE {where}
                    ORDER BY id DESC LIMIT ?""",
                (*params, tail),
            )
            rows = list(reversed(list(await cursor.fetchall())))
        else:
            bound = "" if limit is None else " LIMIT ?"
            tail_params: tuple[object, ...] = () if limit is None else (limit,)
            cursor = await self._db.conn.execute(
                f"""SELECT id, stream, message, timestamp FROM job_logs
                    WHERE {where} AND id > ?
                    ORDER BY id ASC{bound}""",
                (*params, since_id, *tail_params),
            )
            rows = list(await cursor.fetchall())
        return [
            LogEntry(
                id=r["id"],
                stream=LogStream(r["stream"]),
                message=r["message"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    async def max_log_id(self) -> int:
        """Return the maximum log ID, or zero; filtered followers use it as a scan watermark."""
        cursor = await self._db.conn.execute("SELECT COALESCE(MAX(id), 0) FROM job_logs")
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    @_serialized
    async def sweep_job_logs(self, cutoff: str) -> int:
        """Delete at most 500 logs older than a space-format UTC cutoff, across all workload
        kinds.

        Return the count so callers can release the write lock between chunks.
        The retention setting's zero value disables pruning at the caller.
        """
        cursor = await self._db.conn.execute(
            """DELETE FROM job_logs WHERE id IN (
                   SELECT id FROM job_logs
                   WHERE timestamp < ?
                   LIMIT ?)""",
            (cutoff, _RETENTION_SWEEP_CHUNK),
        )
        await self._db.conn.commit()
        return cursor.rowcount or 0
