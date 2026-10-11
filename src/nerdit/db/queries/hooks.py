"""Read and write the `hooks` rows (inbound webhooks).

The table is the truth. A revoke keeps the row, so a revoked id never resolves
again; writes verify the authorized job id under the write lock, the
`DomainQueries` posture.
"""

from __future__ import annotations

import contextlib
import sqlite3
from datetime import UTC, datetime
from typing import Any, Literal

from nerdit.core.hooks import new_hook_id
from nerdit.db.rows import Hook

from ._base import QueriesBase, _serialized

AddHookOutcome = Literal["inserted", "exists", "conflict", "no_service"]

_COLUMNS = "hook_id, service_name, name, sender, path, header, prefix, secret_ref, created_at"
_FIELDS = ("sender", "path", "header", "prefix", "secret_ref")


class HookQueries(QueriesBase):
    """Live hooks of a service: list, resolve, add, revoke, apply declarations."""

    async def list_hooks(self, service_name: str, job_id: str | None = None) -> list[Hook]:
        """Return a service's live hooks ordered by name.

        With `job_id`, only while that job is still the service: a route that
        authorized a row must not read a successor's hooks (each url is a capability).
        """
        if job_id is not None and not await _job_matches(self._db.conn, job_id, service_name):
            return []
        cursor = await self._db.conn.execute(
            f"SELECT {_COLUMNS} FROM hooks WHERE service_name = ? AND revoked_at IS NULL "
            "ORDER BY name",
            (service_name,),
        )
        return [_row_to_hook(r) for r in await cursor.fetchall()]

    async def get_live_hook(
        self, hook_id: str, service_name: str
    ) -> tuple[Hook, str | None] | None:
        """Return a live hook of a live service with that service's project id, else None."""
        cursor = await self._db.conn.execute(
            "SELECT h.hook_id, h.service_name, h.name, h.sender, h.path, h.header, h.prefix, "
            "h.secret_ref, h.created_at, j.project_id FROM hooks h "
            "JOIN jobs j ON j.service_name = h.service_name AND j.kind = 'service' "
            "WHERE h.hook_id = ? AND h.service_name = ? AND h.revoked_at IS NULL",
            (hook_id, service_name),
        )
        row = await cursor.fetchone()
        return None if row is None else (_row_to_hook(row), row["project_id"])

    @_serialized
    async def add_hook(  # noqa: PLR0913 - the row's own fields, all keyword-only
        self,
        service_name: str,
        *,
        job_id: str,
        name: str,
        sender: str,
        path: str,
        header: str | None,
        prefix: str | None,
        secret_ref: str | None,
    ) -> tuple[AddHookOutcome, Hook | None]:
        """Add a hook; an identical live one is `exists`, a different one `conflict`."""
        fields = dict(zip(_FIELDS, (sender, path, header, prefix, secret_ref), strict=True))
        conn = self._db.conn
        await conn.execute("BEGIN IMMEDIATE")
        try:
            if not await _job_matches(conn, job_id, service_name):
                await conn.execute("ROLLBACK")
                return "no_service", None
            existing = await _live(conn, service_name, name)
            if existing is not None:
                await conn.execute("ROLLBACK")
                same = all(getattr(existing, f) == fields[f] for f in _FIELDS)
                return ("exists" if same else "conflict"), existing
            created = await _insert(conn, service_name, name, fields)
            await conn.commit()
            return "inserted", created
        except BaseException:
            with contextlib.suppress(Exception):
                await conn.execute("ROLLBACK")
            raise

    @_serialized
    async def remove_hook(self, service_name: str, name: str, job_id: str | None = None) -> bool:
        """Revoke a live hook by name; False when none was live.

        With `job_id`, only while that job is still the service (same reason as
        `list_hooks`: the authorized row, not a successor by the same name).
        """
        if job_id is not None and not await _job_matches(self._db.conn, job_id, service_name):
            return False
        cursor = await self._db.conn.execute(
            "UPDATE hooks SET revoked_at = ? "
            "WHERE service_name = ? AND name = ? AND revoked_at IS NULL",
            (datetime.now(UTC).isoformat(), service_name, name),
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @_serialized
    async def apply_declared_hooks(
        self, service_name: str, job_id: str, specs: dict[str, dict[str, Any]]
    ) -> None:
        """Upsert nerdit.toml declarations; ids survive field changes, undeclared rows stay."""
        if not specs:
            return
        conn = self._db.conn
        await conn.execute("BEGIN IMMEDIATE")
        try:
            if not await _job_matches(conn, job_id, service_name):
                await conn.execute("ROLLBACK")
                return
            for name, spec in specs.items():
                existing = await _live(conn, service_name, name)
                if existing is None:
                    await _insert(conn, service_name, name, spec)
                elif any(getattr(existing, f) != spec.get(f) for f in _FIELDS):
                    # Update in place: the address pasted at the provider survives.
                    await conn.execute(
                        "UPDATE hooks SET sender = ?, path = ?, header = ?, prefix = ?, "
                        "secret_ref = ? WHERE hook_id = ?",
                        (*(spec.get(f) for f in _FIELDS), existing.hook_id),
                    )
            await conn.commit()
        except BaseException:
            with contextlib.suppress(Exception):
                await conn.execute("ROLLBACK")
            raise


async def _job_matches(conn: Any, job_id: str, service_name: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM jobs WHERE id = ? AND service_name = ? AND kind = 'service'",
        (job_id, service_name),
    )
    return await cursor.fetchone() is not None


async def _live(conn: Any, service_name: str, name: str) -> Hook | None:
    cursor = await conn.execute(
        f"SELECT {_COLUMNS} FROM hooks WHERE service_name = ? AND name = ? AND revoked_at IS NULL",
        (service_name, name),
    )
    row = await cursor.fetchone()
    return None if row is None else _row_to_hook(row)


async def _insert(conn: Any, service_name: str, name: str, spec: dict[str, Any]) -> Hook:
    hook_id = new_hook_id()
    await conn.execute(
        "INSERT INTO hooks (hook_id, service_name, name, sender, path, header, prefix, "
        "secret_ref) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (hook_id, service_name, name, *(spec.get(f) for f in _FIELDS)),
    )
    cursor = await conn.execute(f"SELECT {_COLUMNS} FROM hooks WHERE hook_id = ?", (hook_id,))
    row = await cursor.fetchone()
    assert row is not None  # inserted a statement ago inside this transaction
    return _row_to_hook(row)


def _row_to_hook(row: sqlite3.Row) -> Hook:
    return Hook(**{key: row[key] for key in Hook.model_fields})
