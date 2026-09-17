"""Managed-database MCP tool implementations (P15; dumps P37).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).

(P37, D-P37-12) The dump surface reaches MCP as a **pair** — ``dump_database``
and ``list_database_dumps`` — and stops there. ``POST /databases/{name}/restore``
gets no tool: the custody precedent (P14c's backup, P17d's license) keeps a tool
off any surface whose *response* carries key material, and this one is kept off
for the sibling reason — the response carries none, but the act replaces every
row of a live database, and an agent that can dump before a migration has the
whole agent story without also holding the destructive half. Restore stays CLI
and REST, where a human types the word ``restore`` at a confirm prompt.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import (
    DEFAULT_DATABASE_LIMIT,
    DEFAULT_DUMP_TIMEOUT_S,
    MAX_DATABASE_LIMIT,
    Cursor,
    IdempotencyKey,
)
from nerdit.mcp.transport import _request_client


async def _create_database_impl(
    client: NerditClient,
    *,
    backend: str | None = None,
    name: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Provision a managed database (kind=database workload), auto-minting a key.

    Like :func:`_serve_model_impl`, the agent rarely supplies its own key, so we
    mint a UUID to make every create safe to retry: a replayed call collapses to
    the same database row rather than failing on the unique ``service_name``. The
    minted password never rides the response — key names only (D-B).
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.create_database(backend=backend, name=name, idempotency_key=idempotency_key)
    )


async def _list_databases_impl(
    client: NerditClient,
    *,
    limit: int | None = None,
    cursor: str | None = None,
) -> Any:
    """List managed databases, bounded to ``limit`` per page.

    The bound is enforced server-side (``?limit=``) so the daemon never returns
    an unbounded page; the cursor is passed straight through for paging.
    """
    bound = _clamp(limit if limit is not None else DEFAULT_DATABASE_LIMIT, MAX_DATABASE_LIMIT)
    return await _call(client.list_databases(limit=bound, cursor=cursor))


async def _dump_database_impl(
    client: NerditClient,
    *,
    name: str,
    timeout_s: int | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Dump a managed database, auto-minting an Idempotency-Key (P37, §1.7).

    The key is minted here for the reason ``create_database`` mints one: the
    route requires one in-route (400 ``idempotency_key_required`` without it)
    and an agent rarely carries its own. It is honest about what that buys —
    exactly what the ``run_command`` docstring says about the same auto-mint: a
    *single* call is safe against a middleware-level duplicate, but the tool
    exposes no ``idempotency_key`` argument (the locked §1.7 signature), so a
    fresh call mints a fresh key and really does run a second dump. That is why
    both descriptions send an agent to ``list_database_dumps`` before retrying a
    call its client gave up on: the first dump most likely finished.

    ``timeout_s`` is floored at 1 and otherwise passed through unclamped: its
    ceiling is ``[services].dump_timeout_max_s``, so a client-side clamp would
    silently rewrite the request whenever an operator raises the cap and would
    keep the authoritative ``422 dump.timeout_too_large`` from ever reaching an
    agent (the ``_run_command_impl`` precedent).
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    budget = max(1, int(timeout_s if timeout_s is not None else DEFAULT_DUMP_TIMEOUT_S))
    return await _call(
        client.create_database_dump(name, timeout_s=budget, idempotency_key=idempotency_key)
    )


async def _list_database_dumps_impl(client: NerditClient, *, name: str) -> Any:
    """List one database's dump tars — a bounded read, no paging.

    Bounded by the daemon's own cap on the route (the newest 500), reported as
    ``truncated`` on the page: retention is NOT what bounds it, because
    ``[retention].dump_keep_last = 0`` is a supported "never prune". There is no
    cursor to project, so the projection is the response verbatim.
    """
    return await _call(client.list_database_dumps(name))


DatabaseBackend = Annotated[
    str | None,
    Field(
        description="Data backend to provision: ``postgres`` or ``redis``. Omit to "
        "use the daemon's ``[databases].default_backend`` (``postgres`` out of the "
        "box); an unknown value is refused, never silently defaulted."
    ),
]
DatabaseName = Annotated[
    str | None,
    Field(
        description="Service name for the new database (lowercase DNS label, ≤ 63 "
        "chars). Omit to use the backend's own name prefix. An existing name is a "
        "conflict, not a redeploy."
    ),
]
DatabaseLimit = Annotated[
    int | None,
    Field(
        description="Databases per page; omit for 50. Clamped to 200, the daemon's own page cap."
    ),
]
DatabaseTarget = Annotated[
    str,
    Field(
        description="Name of an existing managed database (``kind=database``); an "
        "app or model row is refused."
    ),
]
DumpTimeout = Annotated[
    int,
    Field(
        description="Seconds the dump tool may run server-side; omit for 300. "
        "Floored at 1 and otherwise passed straight through, so a value above the "
        "daemon's ``[services].dump_timeout_max_s`` gets the authoritative 422 "
        "``dump.timeout_too_large`` instead of being silently clamped."
    ),
]


# fmt: off
async def create_database(
        backend: DatabaseBackend = None,
        name: DatabaseName = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Provision a managed database (kind=database workload).

        Provisioning is asynchronous: the call returns immediately with the
        database row in ``building`` while the daemon pulls the image off-tick
        and launches the server; poll ``list_databases`` (or ``get_service``)
        until ``db_ready`` is true. Lifecycle (stop / restart / delete) goes
        through the existing service tools.

        The daemon mints the password server-side and stores it write-only — it
        is NEVER returned by this tool (key names only). An app connects to this
        database by declaring a ``[db.<name>]`` binding (``provider="managed"``,
        ``database`` = this database's name — the ``database`` field is
        required, there is no implicit default); at each launch it receives
        ``NERDIT_DB_<NAME>_URL``, and the ``default`` binding also sets
        ``DATABASE_URL`` (Postgres) or ``REDIS_URL`` (Redis) with the composed
        DSN so an unmodified client library connects with no code change.
        """
        return await _create_database_impl(
            _request_client(),
            backend=backend,
            name=name,
            idempotency_key=idempotency_key,
        )

async def list_databases(limit: DatabaseLimit = None, cursor: Cursor = None) -> Any:
        """List managed databases (kind=database workloads), bounded to ``limit`` per page.

        Returns a cursor-paginated page. A newly created database shows
        ``building`` until its off-tick image pull and readiness probe complete
        (``db_ready`` true).
        The ``endpoint`` is password-free by construction; lifecycle actions go
        through the service tools.
        """
        return await _list_databases_impl(_request_client(), limit=limit, cursor=cursor)

async def dump_database(
        name: DatabaseTarget,
        timeout_s: DumpTimeout = DEFAULT_DUMP_TIMEOUT_S,
    ) -> Any:
        """Capture a logical, application-consistent dump of a managed database.

        The daemon runs the engine's own dump tool — ``pg_dump --format=custom``
        for Postgres, ``redis-cli --rdb`` for Redis — from a throwaway container
        built from the database's OWN image, against the running server. This is
        the dump to take before a migration or a destructive change: unlike the
        volume backup (``nerdit backup --volume``, no tool), it is consistent by
        the engine's rules rather than by the filesystem's.

        CUSTODY: the artifact is a server-side tar under the daemon's backups
        directory holding **every row** of ``name``; it is never served over
        HTTP and nothing but metadata comes back here — basename, size, sha256,
        engine, format, duration. Treat the basename as the handle: it is what
        ``nerdit db restore`` takes.

        RETENTION: the daemon keeps the last ``[retention].dump_keep_last``
        dumps per database (default 5) and prunes older ones on its hourly
        sweep, so a dump is not an archive. Copy one off the box if it must
        outlive that window.

        TIMEOUT: the default matches the usual MCP client read budget, not
        the route's own 900: if your client gives up first, **a longer dump still completes
        server-side and appears in ``list_database_dumps``**. So a read timeout
        here is not a failure — call ``list_database_dumps`` and take the newest
        entry. Do NOT blind-retry: this tool mints its own idempotency key per
        call, so a retry is a second real dump (and the first one may still be
        running, which answers ``409 dump.in_progress``).

        Authorization is **owner-or-admin plus token scope**: a scoped submitter
        can dump the databases it created, which is what makes this usable over
        the (submitter-capped) cloud link. Refusals you should branch on: ``409
        dump.database_not_ready`` (not running, or still doing initdb — poll
        ``list_databases`` for ``db_ready``), ``409 dump.in_progress`` (this
        database already has one running; distinct from the daemon-wide
        ``backup.in_progress``), ``409 dump.too_many_in_flight``, ``409
        dump.insufficient_disk`` (the error object carries top-level
        ``required_bytes`` / ``free_bytes``), ``422 dump.not_a_database``, and
        ``500 dump.failed``
        whose ``hint`` is the tool's own last output line. There is deliberately
        **no restore tool** — restoring replaces live data and stays on the CLI
        (``nerdit db restore``) and REST.
        """
        return await _dump_database_impl(_request_client(), name=name, timeout_s=timeout_s)

async def list_database_dumps(name: DatabaseTarget) -> Any:
        """List the dump tars this daemon holds for one managed database.

        Newest first, ``{dump, size_bytes, created_at}`` per row, where
        ``created_at`` is the tar's mtime (the listing never opens an archive).
        Metadata only, like ``dump_database``: the tars themselves stay under
        the daemon's backups directory and are never served over HTTP, and each
        one holds every row of ``name``.

        Two things this read is for. It is the recovery path when a
        ``dump_database`` call outran your client's read budget — **the dump
        still completes server-side and shows up here**, so list before you
        retry. And it is how you check retention: the daemon keeps the last
        ``[retention].dump_keep_last`` per database (default 5) and prunes the
        rest hourly, so on a default daemon what is listed here is all there is.
        Retention can be turned off (``dump_keep_last = 0``), so the listing is
        capped at the newest 500 instead: ``truncated`` is true when older tars
        exist on the box that this page does not name.

        Owner-or-admin plus token scope — the same gate as the dump that
        produced them, because knowing which dumps a database has is knowing
        something about the data.
        """
        return await _list_database_dumps_impl(_request_client(), name=name)
# fmt: on


TOOLS = (
    create_database,
    list_databases,
    dump_database,
    list_database_dumps,
)
