"""Store whether and how services are shared; the cloud edge controls audience.

Read each stream uncached for immediate fail-closed unsharing; list views batch
all rows. Partial service-name uniqueness prevents a foreign key, so writes
check the authorized live job and checked deletion cascades under the write lock.
"""

from __future__ import annotations

import hashlib
import sqlite3

from nerdit.core.link.hosted import PUBLIC_ADDRESS_DOMAIN, hosted_host, hosted_label_fits
from nerdit.db.rows import ActiveServicePublicAddress, ServicePublicAddress, ServiceShare

from ._base import QueriesBase, _serialized


class ShareQueries(QueriesBase):
    """Read / upsert / delete the `service_shares` rows."""

    async def list_service_public_addresses(
        self, node_id: str
    ) -> dict[str, ActiveServicePublicAddress]:
        """Read assignments still matching the linked node and current job/project."""
        async with self._db.write_lock:
            cursor = await self._db.conn.execute(
                "SELECT a.* FROM service_public_addresses a JOIN jobs j ON j.id = a.job_id "
                "JOIN projects p ON p.id = a.project_id "
                "WHERE a.node_id = ? AND j.project_id = a.project_id "
                "AND j.service_name = a.service_name AND j.kind = 'service'",
                (node_id,),
            )
            return {
                row["service_name"]: ActiveServicePublicAddress(**dict(row))
                for row in await cursor.fetchall()
            }

    async def get_service_public_address(
        self, service_name: str, node_id: str, job_id: str
    ) -> ActiveServicePublicAddress | None:
        """Resolve a live binding without reusing an address across job replacement."""
        async with self._db.write_lock:
            cursor = await self._db.conn.execute(
                "SELECT a.* FROM service_public_addresses a JOIN jobs j ON j.id = a.job_id "
                "JOIN projects p ON p.id = a.project_id "
                "WHERE a.node_id = ? AND a.service_name = ? AND a.job_id = ? "
                "AND j.project_id = a.project_id AND j.service_name = a.service_name "
                "AND j.kind = 'service'",
                (node_id, service_name, job_id),
            )
            row = await cursor.fetchone()
            return ActiveServicePublicAddress(**dict(row)) if row is not None else None

    async def list_service_hosted_aliases(self, node_id: str) -> dict[str, str | None]:
        """Read alias hashes with live labels; a tombstone deliberately has no name."""
        async with self._db.write_lock:
            cursor = await self._db.conn.execute(
                "SELECT a.host_hash, j.service_name FROM service_hosted_aliases a "
                "LEFT JOIN jobs j ON j.id = a.job_id WHERE a.node_id = ?",
                (node_id,),
            )
            return {row["host_hash"]: row["service_name"] for row in await cursor.fetchall()}

    @_serialized
    async def set_service_public_address(
        self,
        address: ServicePublicAddress,
        *,
        activate: bool = False,
        legacy_host: str | None = None,
    ) -> bool | None:
        """Return changed, unchanged, or None for stale identity; refuse reassignment.

        Check the job in the same transaction as the insert. A different assignment
        raises ValueError; deletion cascades under SQLite's foreign-key constraint.
        """
        await self._db.conn.execute("BEGIN IMMEDIATE")
        cursor = await self._db.conn.execute(
            "SELECT 1 FROM jobs j JOIN projects p ON p.id = j.project_id "
            "WHERE j.id = ? AND j.project_id = ? AND j.service_name = ? AND j.kind = 'service'",
            (address.job_id, address.project_id, address.service_name),
        )
        if await cursor.fetchone() is None:
            await self._db.conn.rollback()
            return None
        if activate:
            cursor = await self._db.conn.execute(
                "SELECT 1 FROM service_shares WHERE service_name = ?", (address.service_name,)
            )
            if await cursor.fetchone() is None:
                await self._db.conn.rollback()
                return None
        cursor = await self._db.conn.execute(
            "SELECT * FROM service_public_addresses WHERE job_id = ? OR service_name = ? "
            "OR slug = ?",
            (address.job_id, address.service_name, address.slug),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            if ServicePublicAddress(**dict(existing)) != address:
                raise ValueError("Public address assignments cannot change")
            if activate and not existing["active"]:
                await self._db.conn.execute(
                    "UPDATE service_public_addresses SET active = 1 WHERE job_id = ?",
                    (address.job_id,),
                )
                if legacy_host is not None:
                    await self._pin_hosted_alias(address.node_id, legacy_host, address.job_id)
                await self._db.conn.commit()
                return True
            await self._db.conn.rollback()
            return False
        await self._db.conn.execute(
            "INSERT INTO service_public_addresses "
            "(node_id, project_id, job_id, service_name, slug, active) VALUES (?, ?, ?, ?, ?, ?)",
            (
                address.node_id,
                address.project_id,
                address.job_id,
                address.service_name,
                address.slug,
                activate,
            ),
        )
        if activate and legacy_host is not None:
            await self._pin_hosted_alias(address.node_id, legacy_host, address.job_id)
        await self._db.conn.commit()
        return True

    async def _pin_hosted_alias(self, node_id: str, host: str, job_id: str) -> None:
        """Claim a host once; nullable job tombstones deliberately survive deletion."""
        host_hash = hashlib.sha256(host.encode("ascii")).hexdigest()
        await self._db.conn.execute(
            "INSERT OR IGNORE INTO service_hosted_aliases(node_id, host_hash, job_id) "
            "VALUES (?, ?, ?)",
            (node_id, host_hash, job_id),
        )

    @_serialized
    async def pin_legacy_hosted_aliases(self, node_id: str, slug: str, domain: str) -> None:
        """Pin preexisting service authorities before deletion can reuse their labels."""
        cursor = await self._db.conn.execute(
            "SELECT id, service_name FROM jobs WHERE kind = 'service'"
        )
        for row in await cursor.fetchall():
            name = row["service_name"]
            if name and hosted_label_fits(name, slug):
                await self._pin_hosted_alias(node_id, hosted_host(name, slug, domain), row["id"])
        await self._db.conn.commit()

    @_serialized
    async def resolve_shared_app(
        self,
        node_id: str,
        service_name: str,
        authority: str,
        job_id: str | None,
        access: str | None,
        legacy_host: str | None,
    ) -> tuple[str, int, str] | None:
        """Validate a single committed incarnation, audience, endpoint and authority."""
        cursor = await self._db.conn.execute(
            "SELECT j.id, s.access, COALESCE(e.active_host_port, e.host_port) AS port, "
            "a.slug, a.active FROM jobs j "
            "JOIN service_shares s ON s.service_name = j.service_name "
            "JOIN service_endpoints e ON e.service_name = j.service_name AND e.job_id = j.id "
            "LEFT JOIN service_public_addresses a ON a.job_id = j.id "
            "AND a.project_id = j.project_id AND a.service_name = j.service_name AND a.node_id = ? "
            "WHERE j.service_name = ? AND j.kind = 'service' "
            "AND COALESCE(j.desired_state, 'running') = 'running'",
            (node_id, service_name),
        )
        row = await cursor.fetchone()
        if row is None or (job_id is not None and row["id"] != job_id):
            return None
        if access == "public" and row["access"] != "public":
            return None
        if authority.partition(":")[0].endswith(f".{PUBLIC_ADDRESS_DOMAIN}"):
            if (
                job_id is None
                or access is None
                or not row["active"]
                or authority != f"{row['slug']}.{PUBLIC_ADDRESS_DOMAIN}"
            ):
                return None
        else:
            host_hash = hashlib.sha256(authority.encode("ascii")).hexdigest()
            cursor = await self._db.conn.execute(
                "SELECT job_id FROM service_hosted_aliases WHERE node_id = ? AND host_hash = ?",
                (node_id, host_hash),
            )
            alias = await cursor.fetchone()
            if alias is not None:
                if alias["job_id"] != row["id"]:
                    return None
                # Historical aliases need cloud proof of the registered incarnation.
                if authority != legacy_host and (job_id is None or access is None):
                    return None
            elif authority == legacy_host:
                await self._pin_hosted_alias(node_id, authority, row["id"])
                await self._db.conn.commit()
            else:
                return None
        return row["id"], row["port"], access or row["access"]

    @_serialized
    async def clear_service_public_addresses(self) -> None:
        """Forget all cloud assignments when the node is unlinked."""
        await self._db.conn.execute("DELETE FROM service_public_addresses")
        await self._db.conn.commit()

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
        self,
        service_name: str,
        access: str,
        *,
        job_id: str,
        preserve_existing: bool = False,
        alias_node_id: str | None = None,
        alias_host: str | None = None,
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
        if wrote and alias_node_id is not None and alias_host is not None:
            await self._pin_hosted_alias(alias_node_id, alias_host, job_id)
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
