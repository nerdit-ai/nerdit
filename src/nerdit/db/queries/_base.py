"""Shared query exceptions, cursor codecs, write serialization and row conversion."""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from nerdit.db.database import Database
from nerdit.db.models import (
    ErrorClass,
    Job,
    JobKind,
    JobStatus,
)

# Active statuses that count against a token's quota (legacy ``jobs`` rows).
# ``retrying`` is included so a retry that never re-reserves keeps counting
# toward ``max_concurrent_jobs``; terminal states are out.
_ACTIVE_JOB_STATUSES = ("pending", "scheduled", "running", "paused", "retrying")

# Active statuses that count a *service* against a token's quota. This set
# INCLUDES the service-mode transient states ``building``/``degraded``/
# ``restarting`` (CRIT-3): they each hold a live row that the reconciler is
# converging, so omitting them would let a token cycle services through those
# states to slip past ``max_concurrent_jobs``. Terminal states
# (completed/failed/cancelled/stopped) are out.
_ACTIVE_SERVICE_STATUSES = (
    "building",
    "scheduled",
    "running",
    "degraded",
    "restarting",
)

# Union of every non-terminal status across BOTH kinds, used for the per-token
# quota count (QUOTA-1). The count query has no ``kind`` predicate, so it must
# span both sets: counting only the reserving kind's statuses let a token park a
# workload of the *other* kind in a kind-exclusive status (e.g. a ``pending``
# batch job invisible to a service reserve) and exceed max_concurrent_jobs /
# max_gpus. Terminal states (completed/failed/cancelled/stopped) stay out.
_ACTIVE_STATUSES = tuple(dict.fromkeys((*_ACTIVE_JOB_STATUSES, *_ACTIVE_SERVICE_STATUSES)))


class ServiceNameTaken(Exception):  # noqa: N818 — domain error mapped to a 409 envelope
    """Raised when a service INSERT collides with the unique `service_name` index.

    Surfaced by `Queries.reserve_service_for_token`; routes map it to a
    `service.name_taken` 409 (S6). One stable row per service name is the P2
    identity invariant (port stability + `nerdit logs <name>` resolution).
    """

    def __init__(self, service_name: str | None) -> None:
        super().__init__(f"Service name already in use: {service_name}")
        self.service_name = service_name


class PortRangeExhausted(Exception):  # noqa: N818 — domain error mapped to a 503 envelope
    """Raised when no free host port remains in the configured service range.

    Surfaced by `Queries.acquire_service_port` after every candidate in
    `[lo, hi]` (minus the daemon port and ports already reserved or, with the
    CRIT-4 bindability hook, not bindable) is taken.
    """

    def __init__(self, lo: int, hi: int) -> None:
        super().__init__(f"No free host port in range {lo}-{hi}")
        self.lo = lo
        self.hi = hi


def _encode_cursor(created_at: datetime, job_id: str) -> str:
    return base64.urlsafe_b64encode(f"{created_at.isoformat()}|{job_id}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        created_at_iso, job_id = raw.split("|", 1)
        return datetime.fromisoformat(created_at_iso), job_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid cursor") from exc


def _encode_name_cursor(service_name: str) -> str:
    """Opaque keyset cursor over `service_name` (P13b, `GET /routes`)."""
    return base64.urlsafe_b64encode(service_name.encode()).decode()


def _decode_name_cursor(cursor: str) -> str:
    try:
        return base64.urlsafe_b64decode(cursor.encode()).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid cursor") from exc


_T = TypeVar("_T")

#: Rows deleted per chunk by the P14b retention sweeps. Each ``@_serialized``
#: sweep call removes at most this many rows and returns the count so the loop
#: releases the write lock between chunks (bare ``DELETE … LIMIT`` is a syntax
#: error on this sqlite3 build, so the sweeps use the portable id-subquery form).
_RETENTION_SWEEP_CHUNK = 500


@dataclass
class RequestWriteMarker:
    """Mutable commit marker shared across a claimed request's middleware tasks.

    No committed effect allows claim deletion after cancellation; a committed or
    uncertain effect pins it interrupted. Mutation crosses the child-task context
    boundary, where ContextVar assignment alone would not propagate back.
    """

    committed: bool = False


#: The in-flight request's write marker, or ``None`` outside a claimed request.
#: Background work (reconcile loops, retention sweeps) is created outside any
#: request context, so its writers see ``None`` and the flip is a no-op.
REQUEST_WRITE_MARKER: ContextVar[RequestWriteMarker | None] = ContextVar(
    "nerdit_request_write_marker", default=None
)


def mark_request_side_effect() -> None:
    """Pin a claimed request before starting a non-database durable effect.

    Worker threads can finish after coroutine cancellation. Mark before dispatch
    so cancellation cannot free the key and permit duplicate backups or files.
    No-op outside a claimed request.
    """
    marker = REQUEST_WRITE_MARKER.get()
    if marker is not None:
        marker.committed = True


def _serialized(
    method: Callable[..., Awaitable[_T]],
) -> Callable[..., Awaitable[_T]]:
    """Serialize a complete write transaction on the shared connection.

    The non-reentrant lock prevents another coroutine committing an open transaction;
    decorated writers must not call each other. Mark request effects on success and
    conservatively on cancellation: aiosqlite's worker may still commit queued work.
    Other exceptions roll back without marking the request.
    """

    @functools.wraps(method)
    async def wrapper(self: QueriesBase, *args: object, **kwargs: object) -> _T:
        async with self._db.write_lock:
            try:
                result = await method(self, *args, **kwargs)
            except BaseException as exc:
                # Roll back open transactions before unlocking, including on cancellation,
                # so the next writer can begin. Check in_transaction and preserve the original
                # exception if rollback also fails.
                if self._db.conn.in_transaction:
                    try:
                        await self._db.conn.rollback()
                    except BaseException:
                        pass
                if isinstance(exc, asyncio.CancelledError):
                    # The rollback above is best-effort against a cancellation:
                    # aiosqlite may still execute the statements (and the
                    # ``commit()``) this coroutine already submitted to its
                    # worker thread. Whether anything landed is unknowable here,
                    # so pin the claim rather than free it for a re-execution.
                    cancel_marker = REQUEST_WRITE_MARKER.get()
                    if cancel_marker is not None:
                        cancel_marker.committed = True
                raise
            marker = REQUEST_WRITE_MARKER.get()
            if marker is not None:
                marker.committed = True
            return result

    return wrapper


class QueriesBase:
    """Shared connection and row conversion for domain query mixins."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def _scalar_int(self, sql: str, params: Sequence[object] = ()) -> int:
        """Return a one-column integer aggregate, defaulting to zero when no row exists."""
        cursor = await self._db.conn.execute(sql, params)
        row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        """Convert an aiosqlite Row to a Job model (deserializes JSON and ISO timestamps)."""
        r = row  # aiosqlite.Row
        health_raw = r["health_check"]
        health_check = json.loads(health_raw) if health_raw else None
        last_exit_raw = r["last_exit_at"]
        window_start_raw = r["restart_window_start"]
        return Job(
            id=r["id"],
            name=r["name"],
            script_path=r["script_path"],
            gpu_count=r["gpu_count"],
            status=JobStatus(r["status"]),
            container_id=r["container_id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            started_at=datetime.fromisoformat(r["started_at"]) if r["started_at"] else None,
            finished_at=datetime.fromisoformat(r["finished_at"]) if r["finished_at"] else None,
            exit_code=r["exit_code"],
            config=r["config"],
            error_class=ErrorClass(r["error_class"]) if r["error_class"] else None,
            error_message=r["error_message"],
            submitted_via=r["submitted_via"] or "cli",
            kind=JobKind(r["kind"]),
            submitted_by_token=r["submitted_by_token"],
            idempotency_key=r["idempotency_key"],
            desired_state=r["desired_state"],
            restart_policy=r["restart_policy"],
            restart_count=r["restart_count"],
            health_check=health_check,
            service_name=r["service_name"],
            last_exit_at=datetime.fromisoformat(last_exit_raw) if last_exit_raw else None,
            restart_window_start=(
                datetime.fromisoformat(window_start_raw) if window_start_raw else None
            ),
        )
