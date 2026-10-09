"""AI gateway MCP tools: machine aliases and their usage.

Thin loopback projections of ``/api/ai-gateway/routes`` and
``/api/ai-gateway/usage`` (Invariant #3): every refusal (role, alias grammar,
base_url SSRF gate, secret-ref grammar, alias in use) is the daemon's
structured envelope. No tool takes or returns a key: an alias names a
``${secrets.shared.KEY}`` reference, and virtual keys exist only in app envs.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.tools._shared import IdempotencyKey
from nerdit.mcp.transport import _request_client

_Alias = Annotated[
    str,
    Field(
        description="Machine alias an app names in ``[ai.<name>] model`` with "
        '``provider = "gateway"``: a lowercase letter, then up to 31 of a-z, 0-9, '
        "'_' or '-'."
    ),
]


async def _list_ai_routes_impl(client: NerditClient) -> Any:
    return await _call(client.list_ai_routes())


async def _set_ai_route_impl(  # noqa: PLR0913 - one wire field per keyword
    client: NerditClient,
    alias: str,
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    api_key_ref: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Create or replace an alias, minting an idempotency key when absent."""
    return await _call(
        client.set_ai_route(
            alias,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_ref=api_key_ref,
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )
    )


async def _remove_ai_route_impl(
    client: NerditClient,
    alias: str,
    *,
    force: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    return await _call(
        client.remove_ai_route(
            alias, force=force, idempotency_key=idempotency_key or str(uuid.uuid4())
        )
    )


async def _get_ai_usage_impl(
    client: NerditClient, *, service: str | None = None, days: int = 7
) -> Any:
    return await _call(client.get_ai_usage(service=service, days=max(1, min(days, 90))))


# fmt: off
async def list_ai_routes() -> Any:
        """List the machine AI gateway's aliases and whether the gateway is enabled.

        Each alias shows its provider, provider model, base_url and the
        ``${secrets.shared.KEY}`` reference of its key: never a key value.
        """
        return await _list_ai_routes_impl(_request_client())

async def set_ai_route(
        alias: _Alias,
        model: Annotated[
            str,
            Field(description="Provider model the alias maps to (an ollama model name for "
                  "provider 'ollama')."),
        ],
        provider: Annotated[
            str,
            Field(description="``api`` (an OpenAI-compatible provider) or ``ollama`` (a "
                  "model served on this machine)."),
        ] = "api",
        base_url: Annotated[
            str | None,
            Field(description="Provider ``api`` only: OpenAI-compatible base URL, usually "
                  "ending in ``/v1``; https to a public host, or http to loopback or the "
                  "models bridge."),
        ] = None,
        api_key_ref: Annotated[
            str | None,
            Field(description="Provider ``api`` only: a ``${secrets.shared.KEY}`` reference "
                  "to the provider key, set beforehand with ``set_secret`` on scope "
                  "``shared``. Never the key itself."),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Create or replace a machine AI gateway alias (submitter or admin).

        Apps then declare ``[ai.default] provider = "gateway"`` and
        ``model = "<alias>"``: the daemon injects the gateway URL and a per-app
        virtual key, and the provider key never enters a container. Answers
        ``enabled: false`` while ``[ai_gateway]`` is off (the alias is stored
        and serves once it is enabled). Refusals: 422 ``ai_gateway.alias_invalid``,
        ``ai_gateway.base_url_invalid`` (not https, private address, userinfo),
        ``ai_gateway.secret_ref_invalid``.
        """
        return await _set_ai_route_impl(
            _request_client(),
            alias,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_ref=api_key_ref,
            idempotency_key=idempotency_key,
        )

async def remove_ai_route(
        alias: _Alias,
        force: Annotated[
            bool,
            Field(description="true = remove even while a running app uses the alias (its "
                  "AI calls then fail with 404 ``alias_not_found``)."),
        ] = False,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Remove a machine AI gateway alias (submitter or admin).

        Refused with 409 ``ai_gateway.alias_in_use`` (naming the apps) while a
        running app's ``[ai.*]`` uses it, unless ``force``.
        """
        return await _remove_ai_route_impl(
            _request_client(), alias, force=force, idempotency_key=idempotency_key
        )

async def get_ai_usage(
        service: Annotated[
            str | None,
            Field(description="Only this app's usage; omit for every app."),
        ] = None,
        days: Annotated[
            int,
            Field(description="UTC days back, today included (1-90)."),
        ] = 7,
    ) -> Any:
        """AI gateway usage per (UTC day, app, alias): requests, prompt and
        completion tokens, upstream errors. No cost estimate, no prompt trace."""
        return await _get_ai_usage_impl(_request_client(), service=service, days=days)
# fmt: on


TOOLS = (
    list_ai_routes,
    set_ai_route,
    remove_ai_route,
    get_ai_usage,
)
