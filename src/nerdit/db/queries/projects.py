"""Project rows: the grouping every ``kind=service`` job points at (P40a / D-P40-5).

A row is created by the row insert (``_stamp_project``), the boot backfill or
``create_project``; it outlives its services (P40b) and reserves its name for
the owner token until ``delete_project_checked`` releases it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Collection, Iterable

from nerdit.core.project_identity import DEFAULT_SERVICE, PRODUCTION, mint_project_id, service_label
from nerdit.db.models import Job
from nerdit.db.rows import LinkedProjectService, Project, VariableFlag

from ._base import (
    _JOB_SELECT,
    ProjectExists,
    QueriesBase,
    ServiceNameClaimed,
    ServiceNameTaken,
    _decode_name_cursor,
    _encode_name_cursor,
    _serialized,
)

_PROJECT_SELECT = "SELECT id, name, display_name, submitted_by_token, created_at FROM projects"


class ProjectQueries(QueriesBase):
    """Reads, the judged create and the guarded delete over the `projects` rows."""

    async def get_project(self, project_id: str) -> Project | None:
        """The project with this id, or `None`."""
        cursor = await self._db.conn.execute(f"{_PROJECT_SELECT} WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return self._row_to_project(row) if row is not None else None

    async def get_committed_project_services(
        self, project_id: str
    ) -> tuple[Project, list[LinkedProjectService]] | None:
        """Read a complete identity/publication snapshot after pending writes settle."""
        async with self._db.write_lock:
            project = await self.get_project(project_id)
            if project is None:
                return None
            cursor = await self._db.conn.execute(
                "SELECT j.id AS job_id, j.service_name, j.environment, j.service, s.access, "
                "EXISTS(SELECT 1 FROM service_endpoints e WHERE e.service_name = j.service_name "
                "AND e.job_id = j.id) AS has_endpoint "
                "FROM jobs j LEFT JOIN service_shares s ON s.service_name = j.service_name "
                "WHERE j.project_id = ? AND j.kind = 'service' ORDER BY j.id",
                (project_id,),
            )
            return project, [LinkedProjectService(**dict(row)) for row in await cursor.fetchall()]

    async def get_project_by_name(self, name: str) -> Project | None:
        """The project with this unique name, or `None`."""
        cursor = await self._db.conn.execute(f"{_PROJECT_SELECT} WHERE name = ?", (name,))
        row = await cursor.fetchone()
        return self._row_to_project(row) if row is not None else None

    async def list_project_names(self) -> set[str]:
        """Every project name (the workspace sweep's project-side live set)."""
        cursor = await self._db.conn.execute("SELECT name FROM projects")
        return {row[0] for row in await cursor.fetchall()}

    async def page_projects(
        self, *, names: Collection[str] | None = None, cursor: str | None = None, limit: int = 50
    ) -> tuple[list[Project], str | None]:
        """Keyset page over `projects` by name, optionally restricted to `names`.

        Backs `GET /projects` (P40b): a scoped token passes its `scope_services`
        as `names` so the filter rides the SQL and pages stay full (D-P40-15).
        A scope entry matches a project by NAME only (D-P40-7): a token scoped
        to the composed label `api--asso` does not list project `asso`, the
        same answer `get_project`'s scope gate gives it. `limit` is clamped to
        `[1, 200]` like `list_services`.

        Args:
            names: Only projects with these names; `None` for every project.
            cursor: An opaque cursor from a previous page.
            limit: Page size.

        Returns:
            The page and the cursor for the next one (`None` when exhausted).

        Raises:
            ValueError: On a malformed cursor.
        """
        limit = max(1, min(limit, 200))
        clauses: list[str] = []
        params: list[object] = []
        if names is not None:
            if not names:
                return [], None
            clauses.append(f"name IN ({', '.join('?' for _ in names)})")
            params.extend(sorted(names))
        if cursor:
            clauses.append("name > ?")
            params.append(_decode_name_cursor(cursor))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit + 1)
        cursor_obj = await self._db.conn.execute(
            f"{_PROJECT_SELECT}{where} ORDER BY name LIMIT ?", params
        )
        rows = list(await cursor_obj.fetchall())
        items = [self._row_to_project(r) for r in rows[:limit]]
        next_cursor = _encode_name_cursor(items[-1].name) if len(rows) > limit else None
        return items, next_cursor

    async def list_project_services(self, project_id: str) -> list[Job]:
        """Every `jobs` row stamped with this project, by service name."""
        cursor = await self._db.conn.execute(
            f"{_JOB_SELECT} WHERE jobs.project_id = ? ORDER BY jobs.service, jobs.service_name",
            (project_id,),
        )
        return [self._row_to_job(r) for r in await cursor.fetchall()]

    @_serialized
    async def create_project(
        self, name: str, token_id: str | None, *, admin: bool = False
    ) -> Project:
        """Create the project `name` owned by `token_id` (P40b / D-P40-5 rule 3).

        One statement under the write lock: for a non-admin the INSERT is
        predicated on no foreign `secret_claims` row AND no foreign `jobs` row
        for the project's default label (`service_label(name, production,
        web)`, which equals the name), so a stranger's pre-deploy secrets on
        `blog` -- or a stranger's live model/database row `blog`, which owns no
        `projects` row (`_stamp_project` skips non-service kinds) -- refuse
        `create_project("blog")` with the same 409 the deploy would meet.
        Without the row predicate a stranger could squat the name and lock the
        row's owner out at its next re-create (rule 1), the admin-only tangle
        rules 2-3 exist to prevent. The caller's own claim survives untouched:
        the deploy consumes it (D-P39-3). A NULL claimant or owner is foreign
        to every non-admin. `admin` defaults to `False` so a route that
        forgets it fails closed.

        Args:
            name: The project name (grammar enforced by the route, D-P40-14).
            token_id: The owner's token id; `None` for a LOCAL/LEGACY_ADMIN
                owner (admin-only thereafter).
            admin: Whether the caller bypasses the claim, as on row inserts.

        Returns:
            The created row.

        Raises:
            ProjectExists: A `projects` row already carries the name.
            ServiceNameClaimed: Another token set the label's secrets first.
            ServiceNameTaken: Another token's live row already carries the label.
        """
        label = service_label(name, PRODUCTION, DEFAULT_SERVICE)
        project_id = mint_project_id()
        sql = "INSERT INTO projects (id, name, submitted_by_token) SELECT ?, ?, ?"
        params: tuple[object, ...] = (project_id, name, token_id)
        if not admin:
            sql += (
                " WHERE NOT EXISTS (SELECT 1 FROM secret_claims WHERE service_name = ?"
                " AND (token_id IS NULL OR token_id IS NOT ?))"
                " AND NOT EXISTS (SELECT 1 FROM jobs WHERE service_name = ?"
                " AND (submitted_by_token IS NULL OR submitted_by_token IS NOT ?))"
            )
            params += (label, token_id, label, token_id)
        try:
            cursor = await self._db.conn.execute(sql, params)
        except sqlite3.IntegrityError as exc:  # UNIQUE(name); `_serialized` rolls back
            raise ProjectExists(name) from exc
        await self._db.conn.commit()
        if (cursor.rowcount or 0) == 0:
            # Still under the lock: one value-free re-read picks the envelope
            # (both are 409s that name only the label -- no new oracle).
            cursor = await self._db.conn.execute(
                "SELECT 1 FROM jobs WHERE service_name = ?", (label,)
            )
            if await cursor.fetchone() is not None:
                raise ServiceNameTaken(label)
            raise ServiceNameClaimed(label)
        project = await self.get_project(project_id)
        assert project is not None  # just inserted under the lock
        return project

    @_serialized
    async def rename_project(self, project_id: str, name: str) -> Project | None:
        """Change the display label without moving any operational namespace."""
        await self._db.conn.execute(
            "UPDATE projects SET display_name = ? WHERE id = ?", (name, project_id)
        )
        await self._db.conn.commit()
        return await self.get_project(project_id)

    @_serialized
    async def delete_project_checked(self, project_id: str) -> list[str] | None:
        """Delete the `projects` row once nothing references it (P40b / D-P40-17).

        The service rows go first through the per-row route cascade
        (`service_purge.delete_service`: teardown, `delete_service_checked`,
        purge flags); this is the last step, and it refuses while any `jobs`
        row still points at the project -- including one that landed between
        the route's per-row deletes and this call -- so a stale delete can
        never orphan a live service's project. Same return contract as
        `delete_service_checked`: distinguish absence from success so the
        route reports the right thing.

        Returns:
            `None` when no such project, `[]` after deletion, or the sorted
            labels of the rows still referencing it when refused.
        """
        # `_serialized` rolls back on any exception; the refusal paths roll back by hand.
        await self._db.conn.execute("BEGIN IMMEDIATE")
        cursor = await self._db.conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,))
        if await cursor.fetchone() is None:
            await self._db.conn.execute("ROLLBACK")
            return None
        cursor = await self._db.conn.execute(
            "SELECT service_name FROM jobs WHERE project_id = ? ORDER BY service_name",
            (project_id,),
        )
        remaining = [r[0] for r in await cursor.fetchall()]
        if remaining:
            await self._db.conn.execute("ROLLBACK")
            return remaining
        await self._db.conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await self._db.conn.commit()
        return []

    # --- Variable flags (P40c / D-P40-1): plain-or-secret per key, never a value ---

    @staticmethod
    def _variable_scope(service: str | None) -> tuple[str, str]:
        """The `(environment, service)` storage pair for a scope.

        The project scope is `('', '')`, not NULLs: NULLs never collide in a
        SQLite primary key, so a NULL pair would let one key hold two flag
        rows. A service scope is `('production', <service>)` (D-P40-13: the
        only environment phase 1 writes).
        """
        return ("", "") if service is None else (PRODUCTION, service)

    @_serialized
    async def upsert_variable_flags(
        self, project_id: str, service: str | None, keys: Iterable[str], plain: bool
    ) -> None:
        """Record `keys` as plain or secret in one scope, flipping any existing flag.

        Args:
            project_id: The owning project's id (FK, D-P40-17).
            service: The service name, or `None` for the project scope.
            keys: Key names just written to the scope's encrypted file.
            plain: Whether the values may be shown to the owner.
        """
        environment, scope_service = self._variable_scope(service)
        await self._db.conn.executemany(
            "INSERT INTO variables (project_id, environment, service, key, plain) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (project_id, environment, service, key) "
            "DO UPDATE SET plain = excluded.plain, updated_at = datetime('now')",
            [(project_id, environment, scope_service, key, int(plain)) for key in keys],
        )
        await self._db.conn.commit()

    @_serialized
    async def delete_variable_flag(self, project_id: str, service: str | None, key: str) -> bool:
        """Drop one key's flag row; `True` when one was there."""
        environment, scope_service = self._variable_scope(service)
        cursor = await self._db.conn.execute(
            "DELETE FROM variables WHERE project_id = ? AND environment = ? "
            "AND service = ? AND key = ?",
            (project_id, environment, scope_service, key),
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    async def list_variable_flags(self, project_id: str) -> list[VariableFlag]:
        """Every flag row of the project, all scopes, by service then key."""
        cursor = await self._db.conn.execute(
            "SELECT key, service, plain FROM variables WHERE project_id = ? ORDER BY service, key",
            (project_id,),
        )
        return [
            VariableFlag(key=r["key"], service=r["service"], plain=bool(r["plain"]))
            for r in await cursor.fetchall()
        ]

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        """Convert an aiosqlite Row to a `Project`."""
        return Project(
            id=row["id"],
            name=row["name"],
            display_name=row["display_name"],
            submitted_by_token=row["submitted_by_token"],
            created_at=row["created_at"],
        )
