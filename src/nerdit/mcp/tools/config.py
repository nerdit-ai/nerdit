"""Config-as-API MCP tool implementations (P6/P7).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.tools._shared import DryRun, IdempotencyKey
from nerdit.mcp.transport import _request_client

_ConfigSectionRead = Annotated[
    str | None,
    Field(
        description="Config section to read (e.g. ``proxy``, ``services``, ``retention``); "
        "omit to return every section. An unknown name is a 404 "
        "``config.unknown_section`` listing the known ones."
    ),
]
_ConfigSectionWrite = Annotated[
    str,
    Field(
        description="Config section to write, one per call (e.g. ``proxy``, ``services``, "
        "``retention``). An unknown name is a 404 ``config.unknown_section``."
    ),
]
_ConfigValues = Annotated[
    dict[str, Any],
    Field(
        description="Keys to merge into that section: an omitted key keeps its stored "
        "value, a null value deletes a NULLABLE key (reverting it to its default; a null "
        "on a non-nullable key is a 422 ``config.invalid``), and a nested table is merged "
        "one level deep. An unknown key is refused (422 ``config.invalid``) rather than "
        "silently dropped; ``auth_token`` is refused (403)."
    ),
]
_ConfigIfMatch = Annotated[
    str | None,
    Field(
        description="ETag from a previous read of this section, sent as ``If-Match``: the "
        "write is refused with 409 ``config.stale`` if the config changed since. Omit to "
        "write unconditionally (no ETag is fetched for you); ``*`` always matches."
    ),
]
_AppName = Annotated[
    str,
    Field(
        description="Name of a deployed app (``kind=service``). A served model or a managed "
        "database is a 404 ``config.not_an_app``; an unknown name a 404 ``not_found``."
    ),
]


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
async def get_config(section: _ConfigSectionRead = None) -> Any:
        """Read daemon config: one section view, or all sections when omitted.

        Secret-bearing values are redacted server-side.
        """
        return await _get_config_impl(_request_client(), section=section)

async def set_config(
        section: _ConfigSectionWrite,
        values: _ConfigValues,
        dry_run: DryRun = False,
        if_match: _ConfigIfMatch = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Write a daemon config section (admin-scoped, validated, audited)."""
        return await _set_config_impl(
            _request_client(),
            section,
            values,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )

async def apply_config(
        sections: Annotated[
            dict[str, dict[str, Any]],
            Field(
                description="Desired state as ``{section: {key: value}}``. Only the "
                "sections named are touched; each is merged and validated like "
                "``set_config``'s ``values`` (an omitted key keeps its value, null "
                "deletes a NULLABLE key — a null on a non-nullable one is a 422 — and an "
                "unknown key is a 422). One invalid key rejects the whole document."
            ),
        ],
        dry_run: DryRun = False,
        if_match: Annotated[
            str | None,
            Field(
                description="ETag of the whole config, sent as ``If-Match`` (the daemon "
                "requires it on a real apply; 409 ``config.stale`` when it no longer "
                "matches). Omit and the current ETag is read and used for you; ``*`` "
                "matches unconditionally. Not needed when ``dry_run`` is true."
            ),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Declaratively apply a multi-section daemon config document (all-or-nothing).

        A dry run needs neither an ``If-Match`` nor an idempotency key.
        """
        return await _apply_config_impl(
            _request_client(),
            sections,
            dry_run=dry_run,
            if_match=if_match,
            idempotency_key=idempotency_key,
        )

async def get_app_config(name: _AppName) -> Any:
        """Read a deployed app's daemon-persisted config (deploy fields + ai bindings).

        Secret values are never returned — only ``${secrets.X}`` refs and env
        key names. Each ``[ai.<name>]`` binding shown here is injected into the
        app at launch as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the
        ``default`` binding also sets ``OPENAI_BASE_URL/OPENAI_API_KEY/
        OPENAI_MODEL`` (drop-in OpenAI SDK).

        ``deploy.release`` is the command run once inside the new image before
        traffic moves (a failure aborts the deploy and reverts the image,
        never the data — write idempotent migrations); ``deploy.cutover`` null =
        zero-downtime swap when eligible, false = same-port swap. ``env_keys``
        are literal runtime env names; secret refs are accepted only by
        ``ai.*.api_key``, ``db.*.password``, ``edge_auth.password`` and a git
        ``token_ref`` — a ``${secrets.shared.KEY}`` there resolves at launch for
        any owner, so a human can place the secret and the agent only names it.
        """
        return await _get_app_config_impl(_request_client(), name)

async def set_app_config(
        name: _AppName,
        section: Annotated[
            str,
            Field(
                description="Section to write: ``deploy``, ``ai`` or ``db`` (anything "
                "else is a 422 ``config.unknown_section``). ``env`` is not writable "
                "here — it belongs to the deploy tools and ``set_secret``."
            ),
        ],
        values: Annotated[
            dict[str, Any],
            Field(
                description="Keys to merge into that section: an omitted key keeps its "
                "value, a null deletes it (a whole binding, or a ``deploy`` key back to "
                "its default). In ``ai``/``db`` the unit is the "
                "binding — each ``[ai.<name>]``/``[db.<name>]`` table REPLACES the stored "
                "one wholesale, so send the whole binding. In ``deploy`` name and port are "
                "immutable (422); ``get_app_config`` lists the writable keys."
            ),
        ],
        dry_run: DryRun = False,
        restart: Annotated[
            bool,
            Field(
                description="true = restart the service right after a real write, whether "
                "or not the change needed one. false leaves the running container alone "
                "and the response's ``requires_restart`` says whether it should be "
                "restarted. No effect when ``dry_run`` is true."
            ),
        ] = False,
        if_match: Annotated[
            str | None,
            Field(
                description="ETag from a previous read of this app's config, sent as "
                "``If-Match`` (409 ``config.stale`` when it no longer matches). Omit and "
                "the current ETag is read and used for you; ``*`` matches unconditionally."
            ),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Write an app's ``deploy`` or ``ai`` config section (admin/owner-scoped).

        Writing the ``ai`` section re-wires the app's bindings: on the next
        launch/restart each ``[ai.<name>]`` binding is injected as
        ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the ``default`` binding also
        sets ``OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL`` — pass
        ``restart=True`` to apply it immediately. A dry run needs neither an
        ``If-Match`` nor an idempotency key.
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
