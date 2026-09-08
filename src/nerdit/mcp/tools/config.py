"""Config-as-API MCP tool implementations (P6/P7).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.transport import _request_client


async def _get_config_impl(client: NerditClient, *, section: str | None = None) -> Any:
    """Read daemon config: one section view, or all sections when omitted.

    Secret-bearing values are redacted server-side; this is a read-only
    projection of the validated config store.
    """
    return await _call(client.get_config(section))


async def _set_config_impl(
    client: NerditClient,
    section: str,
    values: dict[str, Any],
    *,
    dry_run: bool = False,
    if_match: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Write a daemon config section, auto-minting an idempotency key.

    ``dry_run=True`` validates and returns the would-be view without writing;
    ``if_match`` carries the optimistic-concurrency ``If-Match`` header (a
    plain string, no auto-fetch — see :func:`_apply_config_impl` /
    :func:`_set_app_config_impl` for the P7 tools that auto-fetch it). Like
    the other write tools, a UUID key is minted when absent so a retried write
    collapses to a single apply.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.put_config(
            section,
            values,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )
    )


async def _apply_config_impl(
    client: NerditClient,
    sections: dict[str, dict[str, Any]],
    *,
    dry_run: bool = False,
    if_match: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Declaratively apply a multi-section daemon config document (all-or-nothing).

    ``If-Match`` is mandatory on a real apply; when ``if_match`` is omitted
    here, the current ETag is auto-fetched (``GET /config/daemon``) and sent,
    so an agent can apply without a manual read-then-write round trip. Passing
    ``if_match`` explicitly disables the auto-fetch and uses that value
    as-is. ``dry_run=True`` is exempt from both ``If-Match`` and the
    idempotency key. An idempotency key is minted when absent (real apply
    only), like the other write tools.
    """
    if not dry_run and if_match is None:
        current = await _call(client.get_config())
        if isinstance(current, dict) and "error" in current:
            return current
        if_match = current[0]["etag"] if current else None
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.apply_config(
            sections,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )
    )


async def _get_app_config_impl(client: NerditClient, name: str) -> Any:
    """Read a deployed app's daemon-persisted config (``[deploy]`` + ``[ai.*]`` spec).

    Secret values are never returned — only ``${secrets.X}`` refs and env key
    names.
    """
    return await _call(client.get_app_config(name))


async def _set_app_config_impl(
    client: NerditClient,
    name: str,
    section: str,
    values: dict[str, Any],
    *,
    dry_run: bool = False,
    restart: bool = False,
    if_match: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Write an app's ``deploy`` or ``ai`` config section (admin/owner-scoped).

    Like :func:`_apply_config_impl`, ``if_match`` is auto-fetched (``GET
    /config/apps/{name}``) when omitted so a retried/first-time write does not
    need a manual read first; pass ``if_match`` explicitly to disable the
    auto-fetch. ``restart=True`` restarts the underlying service after a
    successful real write when the change requires it. ``dry_run=True``
    validates and previews the resulting view without writing. An idempotency
    key is minted when absent (real write only).
    """
    if not dry_run and if_match is None:
        current = await _call(client.get_app_config(name))
        if isinstance(current, dict) and "error" in current:
            return current
        if_match = current.get("etag")
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.put_app_config(
            name,
            section,
            values,
            dry_run=dry_run,
            restart=restart,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )
    )


# fmt: off
async def get_config(section: str | None = None) -> Any:
        """Read daemon config: one section view, or all sections when omitted.

        Secret-bearing values are redacted server-side.
        """
        return await _get_config_impl(_request_client(), section=section)

async def set_config(
        section: str,
        values: dict[str, Any],
        dry_run: bool = False,
        if_match: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Write a daemon config section (admin-scoped, validated, audited).

        ``dry_run=True`` validates and returns the would-be section view
        without writing; ``if_match`` enables optimistic concurrency against
        the section's current version. An idempotency key is auto-generated if
        omitted.
        """
        return await _set_config_impl(
            _request_client(),
            section,
            values,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )

async def apply_config(
        sections: dict[str, dict[str, Any]],
        dry_run: bool = False,
        if_match: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Declaratively apply a multi-section daemon config document (all-or-nothing).

        ``If-Match`` is required by the daemon on a real apply; when omitted
        here it is auto-fetched (current ETag) so an agent can apply without
        a manual read-then-write round trip — pass ``if_match`` explicitly to
        disable the auto-fetch. ``dry_run=True`` validates and previews
        without writing (exempt from ``If-Match``/idempotency key). An
        idempotency key is auto-generated if omitted.
        """
        return await _apply_config_impl(
            _request_client(),
            sections,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )

async def get_app_config(name: str) -> Any:
        """Read a deployed app's daemon-persisted config (deploy fields + ai bindings).

        Secret values are never returned — only ``${secrets.X}`` refs and env
        key names. Each ``[ai.<name>]`` binding shown here is injected into the
        app at launch as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the
        ``default`` binding also sets ``OPENAI_BASE_URL/OPENAI_API_KEY/
        OPENAI_MODEL`` (drop-in OpenAI SDK).
        """
        return await _get_app_config_impl(_request_client(), name)

async def set_app_config(
        name: str,
        section: str,
        values: dict[str, Any],
        dry_run: bool = False,
        restart: bool = False,
        if_match: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Write an app's ``deploy`` or ``ai`` config section (admin/owner-scoped).

        ``if_match`` is auto-fetched (current app-config ETag) when omitted,
        same convention as ``apply_config`` — pass it explicitly to disable
        the auto-fetch. ``restart=True`` restarts the service after a
        successful real write that needs it. ``dry_run=True`` previews
        without writing. An idempotency key is auto-generated if omitted.

        Writing the ``ai`` section re-wires the app's bindings: on the next
        launch/restart each ``[ai.<name>]`` binding is injected as
        ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the ``default`` binding also
        sets ``OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL`` — pass
        ``restart=True`` to apply it immediately.
        """
        return await _set_app_config_impl(
            _request_client(),
            name,
            section,
            values,
            dry_run=dry_run,
            restart=restart,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )
# fmt: on


TOOLS = (
    get_config,
    set_config,
    apply_config,
    get_app_config,
    set_app_config,
)
