"""Secrets MCP tool implementations (P4).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.transport import _request_client


async def _set_secret_impl(
    client: NerditClient,
    service: str,
    values: dict[str, str],
    *,
    idempotency_key: str | None = None,
) -> Any:
    """Set/merge write-only secrets for a service; returns key names only.

    Like the other write tools, a UUID idempotency key is minted when absent so
    a retried write collapses to a single apply.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.set_secrets(service, values, idempotency_key=idempotency_key))


async def _list_secret_names_impl(client: NerditClient, service: str) -> Any:
    """List a service's secret key names (values are never returned)."""
    return await _call(client.list_secrets(service))


async def _rm_secret_impl(
    client: NerditClient,
    service: str,
    *,
    key: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Delete one secret key, or all of a service's secrets when ``key`` is omitted.

    A UUID idempotency key is minted when absent so a retried delete collapses
    to a single apply.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    if key:
        return await _call(client.delete_secret(service, key, idempotency_key=idempotency_key))
    return await _call(client.delete_secrets(service, idempotency_key=idempotency_key))


# fmt: off
async def set_secret(
        service: str, values: dict[str, str], idempotency_key: str | None = None
    ) -> Any:
        """Set/merge write-only secrets for a service; returns key names only.

        An idempotency key is auto-generated if omitted, so a retried write
        collapses to a single apply. ``service="shared"`` targets the global
        shared scope (admin-only; readable by every service via
        ``${secrets.shared.KEY}``) — use it sparingly. WARNING: the values you
        pass here transit this agent's transcript in plaintext; for
        high-value shared keys (broadly-scoped API keys, credentials shared
        across every app) prefer ``nerdit secrets set --shared`` from the CLI
        instead of routing them through an agent.
        """
        return await _set_secret_impl(
            _request_client(), service, values, idempotency_key=idempotency_key
        )

async def list_secret_names(service: str) -> Any:
        """List a service's secret key names (values are never returned).

        ``service="shared"`` lists the global shared scope's key names;
        readable by any authenticated principal (owners need the
        referenceable list), unlike shared writes which are admin-only.
        """
        return await _list_secret_names_impl(_request_client(), service)

async def remove_secret(
        service: str, key: str | None = None, idempotency_key: str | None = None
    ) -> Any:
        """Delete one secret key, or all of a service's secrets when key is omitted.

        An idempotency key is auto-generated if omitted, so a retried delete
        collapses to a single apply. ``service="shared"`` deletes from the
        global shared scope (admin-only) — deleting a shared key can break
        every app still referencing it via ``${secrets.shared.KEY}``.
        """
        return await _rm_secret_impl(
            _request_client(), service, key=key, idempotency_key=idempotency_key
        )
# fmt: on


TOOLS = (
    set_secret,
    list_secret_names,
    remove_secret,
)
