"""Claim custom domains exclusively for services.

The case-insensitive primary key prevents takeover. This table is the sole
domain authority; project config must not recreate removed names. Partial
service-name uniqueness and endpoint reallocation preclude foreign keys, so
writes verify the authorized job ID under lock and checked deletion cascades
in its transaction. Never reveal a conflicting service's name to another owner.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import Literal

from nerdit.db.rows import ServiceDomain

from ._base import QueriesBase, _serialized

#: Outcome of `DomainQueries.add_service_domain` — four distinct facts the
#: route maps to four distinct answers (200 created / 200 idempotent / 409 / 404).
AddDomainOutcome = Literal["inserted", "exists", "taken", "no_service"]

_SELECT = "SELECT domain, service_name, acme, kind, created_at FROM service_domains"


class DomainQueries(QueriesBase):
    """Read / claim / release the `service_domains` rows."""

    async def list_service_domains(self) -> list[ServiceDomain]:
        """Return all domains ordered by service and domain in one batched read."""
        cursor = await self._db.conn.execute(f"{_SELECT} ORDER BY service_name, domain")
        return [self._row_to_domain(r) for r in await cursor.fetchall()]

    async def get_service_domains(self, service_name: str) -> list[ServiceDomain]:
        """Return one service’s domains sorted by name, or an empty list."""
        cursor = await self._db.conn.execute(
            f"{_SELECT} WHERE service_name = ? ORDER BY domain", (service_name,)
        )
        return [self._row_to_domain(r) for r in await cursor.fetchall()]

    async def get_service_domain(self, domain: str) -> ServiceDomain | None:
        """Return the domain claimant or None; case-insensitive collation prevents case-based
        takeover.
        """
        cursor = await self._db.conn.execute(f"{_SELECT} WHERE domain = ?", (domain,))
        row = await cursor.fetchone()
        return self._row_to_domain(row) if row is not None else None

    @_serialized
    async def add_service_domain(
        self, service_name: str, domain: str, *, acme: bool | None, job_id: str
    ) -> tuple[AddDomainOutcome, ServiceDomain | None]:
        """Claim a domain under the write lock and BEGIN IMMEDIATE, never overwriting another
        owner.

        Match the authorized job_id, not just its name, to reject deletion or same-name
        replacement races. acme=None preserves an existing flag and defaults to False
        on insertion, avoiding accidental certificate downgrade.

        Returns:
            ("inserted", row), ("exists", row), ("taken", row), or
            ("no_service", None). A taken response may name the domain, never the
            other service; the returned row is for internal comparison only.
        """
        await self._db.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await self._db.conn.execute(
                "SELECT 1 FROM jobs WHERE id = ? AND service_name = ? AND kind = 'service'",
                (job_id, service_name),
            )
            if await cursor.fetchone() is None:
                await self._db.conn.execute("ROLLBACK")
                return "no_service", None

            cursor = await self._db.conn.execute(f"{_SELECT} WHERE domain = ?", (domain,))
            row = await cursor.fetchone()
            if row is not None:
                existing = self._row_to_domain(row)
                if existing.service_name != service_name:
                    await self._db.conn.execute("ROLLBACK")
                    return "taken", existing
                # ``None`` = "unspecified": the stored value IS the resolution,
                # so the comparison below can never fire and the row is returned
                # untouched.
                resolved = existing.acme if acme is None else acme
                if existing.acme != resolved:
                    # Only column the row has that a re-PUT may legitimately
                    # change; ``created_at`` survives (the ``set_service_share``
                    # rule — "bound since" is the fact an operator reads).
                    await self._db.conn.execute(
                        "UPDATE service_domains SET acme = ? WHERE domain = ?",
                        (int(resolved), existing.domain),
                    )
                    existing = existing.model_copy(update={"acme": resolved})
                await self._db.conn.commit()
                return "exists", existing

            await self._db.conn.execute(
                "INSERT INTO service_domains (domain, service_name, acme, kind) "
                "VALUES (?, ?, ?, 'domain')",
                (domain, service_name, int(bool(acme))),
            )
            cursor = await self._db.conn.execute(f"{_SELECT} WHERE domain = ?", (domain,))
            inserted = await cursor.fetchone()
            # Inserted a statement ago inside this transaction.
            assert inserted is not None
            created = self._row_to_domain(inserted)
            await self._db.conn.commit()
            return "inserted", created
        except BaseException:
            with contextlib.suppress(Exception):
                await self._db.conn.execute("ROLLBACK")
            raise

    @_serialized
    async def remove_service_domain(self, service_name: str, domain: str) -> bool:
        """Remove a domain only from its owning service; return whether a row was deleted.

        No-op removal is idempotent success and must not emit a removal event.
        """
        cursor = await self._db.conn.execute(
            "DELETE FROM service_domains WHERE service_name = ? AND domain = ?",
            (service_name, domain),
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @staticmethod
    def _row_to_domain(row: sqlite3.Row) -> ServiceDomain:
        """Convert an aiosqlite Row to a `ServiceDomain`."""
        return ServiceDomain(
            domain=row["domain"],
            service_name=row["service_name"],
            acme=bool(row["acme"]),
            kind=row["kind"],
            created_at=row["created_at"],
        )
