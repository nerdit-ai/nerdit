"""AI gateway rows: machine-level aliases, hashed per-service virtual keys, daily usage.

No secret value is ever stored here: a route holds a `${secrets[.shared].KEY}`
reference and a key row holds the SHA-256 of the virtual key.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from nerdit.db.rows import AiGatewayKey, AiGatewayRoute, AiGatewayUsage

from ._base import QueriesBase, _serialized


def _now() -> str:
    return datetime.now(UTC).isoformat()


SQLITE_MAX_INT = 2**63 - 1


def _sat(col: str) -> str:
    """SQL for `col + excluded.col` saturating at SQLite's max integer."""
    return (
        f"CASE WHEN {col} > {SQLITE_MAX_INT} - excluded.{col} THEN {SQLITE_MAX_INT} "
        f"ELSE {col} + excluded.{col} END"
    )


class AiGatewayQueries(QueriesBase):
    """Read / write the `ai_gateway_routes`, `ai_gateway_keys` and `ai_gateway_usage` rows."""

    async def list_ai_routes(self) -> list[AiGatewayRoute]:
        """Every alias, sorted by name."""
        cursor = await self._db.conn.execute("SELECT * FROM ai_gateway_routes ORDER BY alias")
        return [self._row_to_ai_route(row) for row in await cursor.fetchall()]

    async def get_ai_route(self, alias: str) -> AiGatewayRoute | None:
        """The alias's route, or `None`."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM ai_gateway_routes WHERE alias = ?", (alias,)
        )
        row = await cursor.fetchone()
        return self._row_to_ai_route(row) if row is not None else None

    @_serialized
    async def upsert_ai_route(
        self,
        alias: str,
        *,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key_ref: str | None = None,
    ) -> bool:
        """Create or replace an alias (keeps `created_at`); `True` when it was created.

        The caller validates every field; the `provider` CHECK is the backstop.
        """
        now = _now()
        cursor = await self._db.conn.execute(
            "SELECT 1 FROM ai_gateway_routes WHERE alias = ?", (alias,)
        )
        created = await cursor.fetchone() is None
        await self._db.conn.execute(
            "INSERT INTO ai_gateway_routes "
            "(alias, provider, base_url, model, api_key_ref, services, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, ?) "
            "ON CONFLICT(alias) DO UPDATE SET provider = excluded.provider, "
            "base_url = excluded.base_url, model = excluded.model, "
            "api_key_ref = excluded.api_key_ref, updated_at = excluded.updated_at",
            (alias, provider, base_url, model, api_key_ref, now, now),
        )
        await self._db.conn.commit()
        return created

    @_serialized
    async def delete_ai_route(self, alias: str) -> bool:
        """Drop the alias; `True` when one was there."""
        cursor = await self._db.conn.execute(
            "DELETE FROM ai_gateway_routes WHERE alias = ?", (alias,)
        )
        await self._db.conn.commit()
        return (cursor.rowcount or 0) > 0

    @_serialized
    async def insert_ai_gateway_key(
        self, service_name: str, key_hash: str, *, revoke_others: bool
    ) -> None:
        """Record a freshly minted key's hash; optionally revoke the service's other keys.

        One transaction under the write lock, so a relaunch never leaves two
        generations of the main container's key valid.
        """
        now = _now()
        if revoke_others:
            await self._db.conn.execute(
                "UPDATE ai_gateway_keys SET revoked_at = ? "
                "WHERE service_name = ? AND revoked_at IS NULL",
                (now, service_name),
            )
        await self._db.conn.execute(
            "INSERT INTO ai_gateway_keys (key_hash, service_name, created_at) VALUES (?, ?, ?)",
            (key_hash, service_name, now),
        )
        await self._db.conn.commit()

    async def ai_gateway_key_service(self, key_hash: str) -> tuple[str, str] | None:
        """`(stored_hash, service_name)` of the ACTIVE key with this hash, or `None`."""
        cursor = await self._db.conn.execute(
            "SELECT key_hash, service_name FROM ai_gateway_keys "
            "WHERE key_hash = ? AND revoked_at IS NULL",
            (key_hash,),
        )
        row = await cursor.fetchone()
        return (row["key_hash"], row["service_name"]) if row is not None else None

    async def list_ai_gateway_keys(self, *, include_revoked: bool = False) -> list[AiGatewayKey]:
        """Key metadata (never the key nor its hash), by service then age."""
        sql = "SELECT service_name, created_at, revoked_at FROM ai_gateway_keys"
        if not include_revoked:
            sql += " WHERE revoked_at IS NULL"
        cursor = await self._db.conn.execute(sql + " ORDER BY service_name, created_at")
        return [AiGatewayKey(**dict(row)) for row in await cursor.fetchall()]

    @_serialized
    async def revoke_ai_gateway_keys(self, service_name: str) -> int:
        """Revoke every active key of the service; the count revoked."""
        cursor = await self._db.conn.execute(
            "UPDATE ai_gateway_keys SET revoked_at = ? "
            "WHERE service_name = ? AND revoked_at IS NULL",
            (_now(), service_name),
        )
        await self._db.conn.commit()
        return cursor.rowcount or 0

    @_serialized
    async def record_ai_usage(
        self,
        service_name: str,
        alias: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        upstream_error: bool = False,
    ) -> None:
        """Count one forwarded request against today's (UTC) row."""
        await self._db.conn.execute(
            "INSERT INTO ai_gateway_usage (day, service_name, alias, requests, prompt_tokens, "
            "completion_tokens, upstream_errors) VALUES (?, ?, ?, 1, ?, ?, ?) "
            "ON CONFLICT(day, service_name, alias) DO UPDATE SET "
            "requests = requests + 1, "
            # Saturating adds: an integer overflow would turn the column REAL
            # and the row unreadable. `excluded.*` is already clamped to MAX.
            f"prompt_tokens = {_sat('prompt_tokens')}, "
            f"completion_tokens = {_sat('completion_tokens')}, "
            "upstream_errors = upstream_errors + excluded.upstream_errors",
            (
                datetime.now(UTC).date().isoformat(),
                service_name,
                alias,
                prompt_tokens,
                completion_tokens,
                int(upstream_error),
            ),
        )
        await self._db.conn.commit()

    async def list_ai_usage(
        self, *, service_name: str | None = None, days: int = 7
    ) -> list[AiGatewayUsage]:
        """Usage rows of the last `days` UTC days (today included), newest first."""
        since = (datetime.now(UTC).date() - timedelta(days=max(days, 1) - 1)).isoformat()
        sql = "SELECT * FROM ai_gateway_usage WHERE day >= ?"
        params: tuple[object, ...] = (since,)
        if service_name is not None:
            sql += " AND service_name = ?"
            params += (service_name,)
        sql += " ORDER BY day DESC, service_name, alias"
        cursor = await self._db.conn.execute(sql, params)
        return [AiGatewayUsage(**dict(row)) for row in await cursor.fetchall()]

    @staticmethod
    def _row_to_ai_route(row: sqlite3.Row) -> AiGatewayRoute:
        data = dict(row)
        data["services"] = json.loads(data["services"]) if data["services"] else None
        return AiGatewayRoute(**data)
