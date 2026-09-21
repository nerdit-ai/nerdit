"""Secrets MCP tool implementations (P4).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.tools._shared import IdempotencyKey
from nerdit.mcp.transport import _request_client

_SecretsScopeWrite = Annotated[
    str,
    Field(
        description="Service (app, model or database) whose secrets this call modifies — "
        "need not exist yet: a set on an undeployed name reserves it for this token until "
        "deployed — or ``shared`` for the global scope: admin-only to write, read by a "
        "service only where a ``${secrets.shared.KEY}`` reference is honoured "
        "(``ai``/``db``/``edge_auth`` refs, a git ``token_ref``), never through ``env``, "
        "which stays literal."
    ),
]
_SecretsScopeRead = Annotated[
    str,
    Field(
        description="Service (app, model or database) whose secret key names to read, "
        "or ``shared`` for the global scope, whose names any authenticated principal "
        "may read."
    ),
]


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
        service: _SecretsScopeWrite,
        values: Annotated[
            dict[str, str],
            Field(
                description="KEY → value pairs merged into the stored set: a listed key is "
                "overwritten, an unlisted one kept. Key names are env-var identifiers "
                "(``[A-Za-z_][A-Za-z0-9_]*``); a value may be multi-line but may contain no "
                "NUL and no control character other than tab/newline/CR. An empty map is "
                "refused."
            ),
        ],
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Set/merge write-only secrets for a service; returns key names only.

        WARNING: the values you pass here transit this agent's transcript in
        plaintext; for high-value shared keys (broadly-scoped API keys,
        credentials shared across every app) prefer ``nerdit secrets set --shared`` from the CLI
        instead of routing them through an agent. The leak-free path for any
        value: a human runs ``nerdit secrets set <service> --prompt KEY``
        (hidden input) and you only ever name ``KEY``; it is injected at
        launch, overriding an ``env`` key of the same name. You may set secrets
        before the service exists; the name is then reserved for your token
        until you deploy it (another token's deploy of that name gets 409
        ``service.name_claimed``).
        """
        return await _set_secret_impl(
            _request_client(), service, values, idempotency_key=idempotency_key
        )

async def list_secret_names(service: _SecretsScopeRead) -> Any:
        """List a service's secret key names (values are never returned)."""
        return await _list_secret_names_impl(_request_client(), service)

async def remove_secret(
        service: _SecretsScopeWrite,
        key: Annotated[
            str | None,
            Field(
                description="The one secret key to delete; omit it — null or an empty "
                "string — to delete ALL of the scope's secrets at once. Deleting a key "
                "that does not exist is a 404."
            ),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Delete one secret key, or all of a service's secrets when key is omitted.

        Deleting a shared key can break every app still referencing it via
        ``${secrets.shared.KEY}``.
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
