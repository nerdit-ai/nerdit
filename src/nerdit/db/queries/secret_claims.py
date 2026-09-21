"""Secret claims: a service name reserved by the token that set its secrets first (P39).

A claim exists precisely while a name has secrets but no ``jobs`` row. It is
minted by the secrets route, consumed by ``reserve_service_for_token`` inside
the row-insert transaction, and deleted by a delete-all of the name's secrets.
No TTL, no sweep: a stale claim is visible to its owner and deletable.
(ponytail: no retention sweep; add one keyed on ``created_at`` if stale
claims ever pile up.)
"""

from __future__ import annotations

import sqlite3

from nerdit.db.rows import SecretClaim

from ._base import QueriesBase, _serialized


class SecretClaimQueries(QueriesBase):
    """Read / mint / delete the `secret_claims` rows."""

    async def get_secret_claim(self, service_name: str) -> SecretClaim | None:
        """The name's claim, or `None` when unclaimed."""
        cursor = await self._db.conn.execute(
            "SELECT service_name, token_id, created_at FROM secret_claims WHERE service_name = ?",
            (service_name,),
        )
        row = await cursor.fetchone()
        return self._row_to_claim(row) if row is not None else None

    async def name_retaken(
        self, service_name: str, token_id: str | None, *, any_claim: bool = False
    ) -> bool:
        """Whether the name now has a row, or a claim that is not `token_id`'s (`any_claim`: any).

        ONE statement, so one snapshot: `reserve_service_for_token` turns a
        claim into a row in a single transaction, and two separate reads could
        see neither the claim it consumed nor the row it inserted.
        """
        cursor = await self._db.conn.execute(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE service_name = ?1) OR "
            "EXISTS(SELECT 1 FROM secret_claims WHERE service_name = ?1 "
            "AND (?3 OR token_id IS NOT ?2))",
            (service_name, token_id, any_claim),
        )
        row = await cursor.fetchone()
        return row is None or bool(row[0])  # no row cannot happen; fail closed

    @_serialized
    async def mint_secret_claim(
        self, service_name: str, token_id: str | None, *, admin: bool = False
    ) -> bool:
        """Claim the name for `token_id`; `True` when this call won it.

        One statement arbitrates every race under the write lock: the primary
        key against another claim, the `jobs` predicate against a row that
        landed after the caller's unlocked row read (`reserve_service_for_token`
        commits under this same lock), and -- for a non-admin -- the `projects`
        predicate against a project of that name owned by another token
        (P40b / D-P40-5 rule 2; a NULL owner is foreign to every non-admin).
        `False` means "a claim, a row or a foreign project owns the name now"
        -- the route re-reads and judges the owner (`_landed_is_callers`), so
        the refusal wears the same 403 as a foreign claim and leaks nothing.
        `admin` defaults to `False` so a caller that forgets it fails closed.
        """
        # ponytail: the label's project is the label itself in phase 1
        # (D-P40-2); a composed label (P40d) carries no parsable project and a
        # claim exists precisely when no row maps it -- pass the project name
        # explicitly if P40d ever needs the judgment on `api--asso`.
        sql = (
            "INSERT OR IGNORE INTO secret_claims (service_name, token_id) "
            "SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE service_name = ?)"
        )
        params: tuple[object, ...] = (service_name, token_id, service_name)
        if not admin:
            sql += (
                " AND NOT EXISTS (SELECT 1 FROM projects WHERE name = ?"
                " AND (submitted_by_token IS NULL OR submitted_by_token IS NOT ?))"
            )
            params += (service_name, token_id)
        cursor = await self._db.conn.execute(sql, params)
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @_serialized
    async def delete_secret_claim(self, service_name: str) -> bool:
        """Drop the claim; `True` when one was there."""
        cursor = await self._db.conn.execute(
            "DELETE FROM secret_claims WHERE service_name = ?", (service_name,)
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> SecretClaim:
        """Convert an aiosqlite Row to a `SecretClaim`."""
        return SecretClaim(
            service_name=row["service_name"],
            token_id=row["token_id"],
            created_at=row["created_at"],
        )
