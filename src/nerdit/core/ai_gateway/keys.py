"""Per-service virtual keys: `nk_<32 hex>`, stored as SHA-256, compared in constant time."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

from nerdit.db.queries import Queries

_KEY_RE = re.compile(r"^nk_[0-9a-f]{32}$")


def hash_key(key: str) -> str:
    """The stored form of a virtual key."""
    return hashlib.sha256(key.encode()).hexdigest()


async def mint_key(queries: Queries, service_name: str, *, revoke_others: bool) -> str:
    """Mint a virtual key for the service and return the plaintext (container env only).

    `revoke_others` is set by a main-container launch: the relaunch revokes the
    service's previous keys. A one-off run, a release or a cutover green mints
    alongside, so the serving container keeps working.
    """
    key = f"nk_{secrets.token_hex(16)}"
    await queries.insert_ai_gateway_key(service_name, hash_key(key), revoke_others=revoke_others)
    return key


async def service_for_key(queries: Queries, key: str) -> str | None:
    """The service an ACTIVE virtual key belongs to, or `None`."""
    if not _KEY_RE.fullmatch(key):
        return None
    digest = hash_key(key)
    found = await queries.ai_gateway_key_service(digest)
    if found is None or not hmac.compare_digest(found[0], digest):
        return None
    return found[1]
