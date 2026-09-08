"""API token CRUD queries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from nerdit.db.models import ApiToken, TokenRole

from ._base import QueriesBase, _serialized


class TokenQueries(QueriesBase):
    """Scoped-token CRUD."""

    @staticmethod
    def _decode_scope(raw: str | None) -> list[str] | None:
        """Decode the `scope_services` JSON column, **failing closed**.

        (P25 D-P25-3) NULL is the only value meaning "unscoped". Anything that
        does not parse as a JSON array of strings decodes to `[]` — an empty
        scope grants nothing — so a corrupted or hand-edited column can never
        widen a token's authority.
        """
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(value, list):
            return []
        return [s for s in value if isinstance(s, str)]

    @staticmethod
    def _decode_expires_at(raw: str | None) -> datetime | None:
        """Parse token expiry as aware UTC, treating naive stored timestamps as UTC."""
        if not raw:
            return None
        parsed = datetime.fromisoformat(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    @staticmethod
    def _row_to_api_token(row: sqlite3.Row) -> ApiToken:
        """Convert an aiosqlite Row to an `ApiToken` model."""
        r = row
        return ApiToken(
            id=r["id"],
            name=r["name"],
            role=TokenRole(r["role"]),
            token_hash=r["token_hash"],
            max_gpus=r["max_gpus"],
            max_concurrent_jobs=r["max_concurrent_jobs"],
            created_at=datetime.fromisoformat(r["created_at"]),
            last_used_at=datetime.fromisoformat(r["last_used_at"]) if r["last_used_at"] else None,
            revoked=bool(r["revoked"]),
            expires_at=TokenQueries._decode_expires_at(r["expires_at"]),
            scope_services=TokenQueries._decode_scope(r["scope_services"]),
        )

    @_serialized
    async def create_api_token(self, token: ApiToken) -> ApiToken:
        """Persist a new scoped API token (only its hash is stored)."""
        await self._db.conn.execute(
            """INSERT INTO api_tokens
               (id, name, role, token_hash, max_gpus, max_concurrent_jobs,
                created_at, last_used_at, revoked, expires_at, scope_services)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                token.id,
                token.name,
                token.role.value,
                token.token_hash,
                token.max_gpus,
                token.max_concurrent_jobs,
                token.created_at.isoformat(),
                token.last_used_at.isoformat() if token.last_used_at else None,
                1 if token.revoked else 0,
                token.expires_at.isoformat() if token.expires_at else None,
                json.dumps(token.scope_services) if token.scope_services is not None else None,
            ),
        )
        await self._db.conn.commit()
        return token

    async def get_api_token_by_id(self, token_id: str) -> ApiToken | None:
        """Return the token row for `token_id` (revoked included), or `None`.

        Used by the scheduler to resolve a job's submitting-token role for
        sandbox tiering; revoked tokens are still resolvable so already-queued
        jobs keep their original (non-admin) treatment.
        """
        cursor = await self._db.conn.execute(
            "SELECT * FROM api_tokens WHERE id = ?",
            (token_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_api_token(row)

    async def get_api_token_by_hash(self, token_hash: str) -> ApiToken | None:
        """Return the active (non-revoked) token matching `token_hash`."""
        cursor = await self._db.conn.execute(
            "SELECT * FROM api_tokens WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_api_token(row)

    async def list_api_tokens(self, *, include_revoked: bool = False) -> list[ApiToken]:
        """List API tokens, newest first. Revoked tokens are hidden by default."""
        where = "" if include_revoked else " WHERE revoked = 0"
        cursor = await self._db.conn.execute(
            f"SELECT * FROM api_tokens{where} ORDER BY created_at DESC, id DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_api_token(r) for r in rows]

    @_serialized
    async def revoke_api_token(self, token_id: str) -> bool:
        """Soft-delete a token (`revoked = 1`). Returns True if a row changed."""
        cursor = await self._db.conn.execute(
            "UPDATE api_tokens SET revoked = 1 WHERE id = ? AND revoked = 0",
            (token_id,),
        )
        await self._db.conn.commit()
        return bool(cursor.rowcount)

    @_serialized
    async def rotate_api_token_hash(
        self,
        token_id: str,
        new_hash: str,
        *,
        expires_at: datetime | None,
        set_expiry: bool,
    ) -> bool:
        """Atomically replace the token secret; return False if missing or revoked.

        Preserve identity, role, quotas and scope; clear last_used_at for the new secret.
        Only set_expiry changes or clears expiry, preventing silent lifetime extension.
        Concurrent rotations serialize (last wins); write marking pins cancelled claims
        so retries cannot mint another secret.
        """
        sets = "token_hash = ?, last_used_at = NULL"
        params: list[object] = [new_hash]
        if set_expiry:
            sets += ", expires_at = ?"
            params.append(expires_at.isoformat() if expires_at else None)
        params.append(token_id)
        cursor = await self._db.conn.execute(
            f"UPDATE api_tokens SET {sets} WHERE id = ? AND revoked = 0",
            params,
        )
        await self._db.conn.commit()
        return bool(cursor.rowcount)

    @_serialized
    async def touch_api_token(self, token_id: str) -> None:
        """Stamp a token's `last_used_at` (throttled by the caller)."""
        await self._db.conn.execute(
            "UPDATE api_tokens SET last_used_at = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), token_id),
        )
        await self._db.conn.commit()
