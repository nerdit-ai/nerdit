"""Audit + events + self-knowledge + observability + disk/GC MCP tool
implementations.

``get_events`` joins ``get_audit`` here rather than getting a module of its own:
they are the two bounded read surfaces over the daemon's two durable trails —
the admin-gated record of *who asked for what*, and the any-authenticated
record of *what the daemon then did on its own*.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import (
    DEFAULT_AUDIT_LIMIT,
    DEFAULT_EVENT_LIMIT,
    DEFAULT_ROUTE_LIMIT,
    MAX_AUDIT_LIMIT,
    MAX_EVENT_LIMIT,
    MAX_ROUTE_LIMIT,
    Cursor,
)
from nerdit.mcp.transport import _request_client


async def _get_audit_impl(
    client: NerditClient,
    *,
    action: str | None = None,
    result: str | None = None,
    target: str | None = None,
    target_type: str | None = None,
    principal_id: str | None = None,
    action_prefix: str | None = None,
    since: str | None = None,
    until: str | None = None,
    cursor: str | None = None,
    limit: int = DEFAULT_AUDIT_LIMIT,
) -> Any:
    """Return a page of the audit log (admin-only), bounded to ``limit`` entries.

    The bound is enforced server-side (``?limit=``, daemon caps at ≤200) and
    clamped here too so an agent can never request an unbounded page. A
    non-admin token gets the structured ``forbidden`` (403) error envelope.

    Every filter is forwarded verbatim; a malformed ``action_prefix`` or
    timestamp comes back as the daemon's ``bad_request`` envelope rather than
    being silently dropped — which is the whole reason this signature has to
    grow alongside the route's (FastMCP drops unknown kwargs, so an unwidened
    tool looks filtered but is not).
    """
    bound = _clamp(limit, MAX_AUDIT_LIMIT)
    return await _call(
        client.get_audit(
            action=action,
            result=result,
            target=target,
            target_type=target_type,
            principal_id=principal_id,
            action_prefix=action_prefix,
            since=since,
            until=until,
            cursor=cursor,
            limit=bound,
        )
    )


async def _get_events_impl(
    client: NerditClient,
    *,
    types: str | None = None,
    service: str | None = None,
    cursor: str | None = None,
    since_id: int | None = None,
    limit: int = DEFAULT_EVENT_LIMIT,
) -> Any:
    """Return a page of the durable event feed, bounded to ``limit`` rows.

    Both pagination modes are passed straight through; the daemon rejects
    ``cursor`` + ``since_id`` together with ``400 bad_request`` rather than
    silently picking one, and that error is what the agent should see.
    """
    bound = _clamp(limit, MAX_EVENT_LIMIT)
    return await _call(
        client.list_events(
            limit=bound,
            cursor=cursor,
            types=types,
            service=service,
            since_id=since_id,
        )
    )


async def _capabilities_impl(client: NerditClient) -> Any:
    """Return the daemon's role-aware self-knowledge projection."""
    return await _call(client.get_capabilities())


async def _doctor_impl(client: NerditClient) -> Any:
    """Return the daemon's timeout-bounded environment health checks."""
    return await _call(client.get_doctor())


async def _proxy_status_impl(client: NerditClient) -> Any:
    """Return the embedded proxy's typed state projection."""
    return await _call(client.get_proxy_status())


async def _list_routes_impl(
    client: NerditClient, *, cursor: str | None = None, limit: int = DEFAULT_ROUTE_LIMIT
) -> Any:
    """Return a page of the route inventory, bounded to ``limit`` rows."""
    bound = _clamp(limit, MAX_ROUTE_LIMIT)
    return await _call(client.list_routes(cursor=cursor, limit=bound))


async def _system_disk_impl(client: NerditClient) -> Any:
    """Return the daemon's disk-usage report (docker + bind-mount data trees)."""
    return await _call(client.get_system_disk())


async def _system_gc_impl(
    client: NerditClient,
    *,
    include_orphan_data: bool = False,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Garbage-collect orphan images (+ opt-in data dirs).

    A real run auto-mints an idempotency key when omitted so a retry collapses
    to one gc; a ``dry_run`` mints **no** key (the ``_deploy_impl`` guard) — it
    writes nothing, and a claimed key would poison a later real gc.
    """
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.run_system_gc(
            include_orphan_data=include_orphan_data,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )
    )


async def _restart_daemon_impl(
    client: NerditClient, *, drain_timeout_s: int = 60, idempotency_key: str | None = None
) -> Any:
    """Restart the daemon (admin-only, audited).

    Unlike ``_system_gc_impl`` there is no dry run here, so the key is
    **always** minted when omitted: the daemon requires one in-route, and a
    minted key is what makes a retry after a dropped response collapse to a
    ``409 daemon.restart_in_progress`` instead of a second restart.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.restart_daemon(drain_timeout_s=drain_timeout_s, idempotency_key=idempotency_key)
    )


# fmt: off
async def get_audit(
        action: Annotated[
            str | None,
            Field(
                description="Keep only entries whose action equals this exactly (e.g. "
                "``deploy.create``); omit for every action, or use ``action_prefix`` for a family."
            ),
        ] = None,
        result: Annotated[
            str | None,
            Field(
                description="Keep only entries with this outcome — ``ok``, ``error``, ``denied`` "
                "or ``replay``; omit for all four."
            ),
        ] = None,
        target: Annotated[
            str | None,
            Field(
                description="Keep only entries acting on this exact target id (e.g. a service "
                "name); omit for every target."
            ),
        ] = None,
        target_type: Annotated[
            str | None,
            Field(
                description="Keep only entries whose target is of this type (e.g. ``service``, "
                "``model``); omit for every type."
            ),
        ] = None,
        principal_id: Annotated[
            str | None,
            Field(
                description="Keep only entries written by this token id — everything that caller "
                "did; omit for every principal."
            ),
        ] = None,
        action_prefix: Annotated[
            str | None,
            Field(
                description="Keep only actions starting with this **literal** prefix, never a "
                "pattern (``deploy.`` matches deploy.create, deploy.git_create, deploy.plan). "
                "Grammar ``[a-z0-9._]``, 1-40 chars, else ``bad_request``; omit for every family."
            ),
        ] = None,
        since: Annotated[
            str | None,
            Field(
                description="Keep only entries at or after this inclusive ISO-8601 UTC timestamp "
                "(``2026-08-07T10:00:00Z``); omit for no lower bound."
            ),
        ] = None,
        until: Annotated[
            str | None,
            Field(
                description="Keep only entries at or before this inclusive ISO-8601 UTC timestamp "
                "(``2026-08-07T10:00:00Z``); omit for no upper bound."
            ),
        ] = None,
        cursor: Cursor = None,
        limit: Annotated[
            int,
            Field(
                description="Max entries in one page, clamped to [1, 200] before the call (the "
                "daemon rejects anything outside that range); omit for 50."
            ),
        ] = DEFAULT_AUDIT_LIMIT,
    ) -> Any:
        """Read the audit log (admin-scoped), bounded to ``limit`` entries per page.

        Returns an ``AuditLogPage`` dict. Non-admin tokens get a structured
        ``forbidden`` (403) error.

        All filters compose with AND, and all are applied in SQL — narrowing is
        cheap, so prefer a filter over paging.
        """
        return await _get_audit_impl(
            _request_client(),
            action=action,
            result=result,
            target=target,
            target_type=target_type,
            principal_id=principal_id,
            action_prefix=action_prefix,
            since=since,
            until=until,
            cursor=cursor,
            limit=limit,
        )

async def get_events(
        types: Annotated[
            str | None,
            Field(
                description="Comma-separated event types to keep (e.g. "
                "``service.failed,model.ready``); only the first 10 entries are honoured. Omit "
                "for every type."
            ),
        ] = None,
        service: Annotated[
            str | None,
            Field(
                description="Keep only events for this exact service name; omit for every "
                "service."
            ),
        ] = None,
        cursor: Annotated[
            str | None,
            Field(
                description="Browse **backwards** from here (id DESC, newest first): pass the "
                "previous page's ``next_cursor`` to see what just happened. Omit for the newest "
                "page; passing it together with ``since_id`` is a 400."
            ),
        ] = None,
        since_id: Annotated[
            int | None,
            Field(
                description="Replay **forwards** from this event id (id ASC, rows with id > it) — "
                "the resume mode after processing up to that id. Omit to browse backwards "
                "instead; passing it together with ``cursor`` is a 400. To keep replaying "
                "forward, pass the page's ``next_cursor`` back as ``since_id``, never as "
                "``cursor``."
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                description="Max rows in one page, clamped to [1, 200] here and server-side; omit "
                "for 50."
            ),
        ] = DEFAULT_EVENT_LIMIT,
    ) -> Any:
        """Use when: you need the daemon's own account of what it just did.

        The durable feed of autonomous transitions (``service.failed``,
        ``service.degraded``, ``service.deploy_succeeded``, ``model.ready``,
        ...), readable by any authenticated token. Two directions, and picking
        the wrong one walks the wrong way through history.
        """
        return await _get_events_impl(
            _request_client(),
            types=types,
            service=service,
            cursor=cursor,
            since_id=since_id,
            limit=limit,
        )

async def capabilities() -> Any:
        """Describe this daemon: version, caller role/quotas, URL grammar, backends.

        Role-aware, pure in-memory read — call it first to learn what the
        daemon can do (buildpacks, model backends, proxy mode and public-URL
        shape, deploy limits, git allowlist, feature flags) instead of probing
        endpoints. Admin callers additionally receive ``proxy.admin_addr`` and
        the ``paths`` block.
        """
        return await _capabilities_impl(_request_client())

async def doctor() -> Any:
        """Run the daemon's bounded environment health checks (docker/gpu/proxy/db/...).

        Returns per-check ``{status, detail}`` plus a worst-of top-level
        status. Every check is timeout-bounded server-side; details carry key
        *names* only, never paths or secret values — safe for every role.
        """
        return await _doctor_impl(_request_client())

async def proxy_status() -> Any:
        """Get the embedded proxy's typed state (TLS, apex, respawn, live routes).

        Authenticated read: proxy state, TLS subjects + CA fingerprint, apex
        registration, respawn backoff, and a tri-state live route table read
        from the Caddy admin API. Complements ``list_routes`` (the
        DB-authoritative inventory).
        """
        return await _proxy_status_impl(_request_client())

async def list_routes(
        cursor: Cursor = None,
        limit: Annotated[
            int,
            Field(
                description="Max rows in one page, clamped to [1, 200] here and server-side; omit "
                "for 50."
            ),
        ] = DEFAULT_ROUTE_LIMIT,
    ) -> Any:
        """List the DB-authoritative route inventory, annotated with live Caddy state.

        Models appear with ``route: null`` (loopback-only by design); each row
        carries its live-table annotation.
        """
        return await _list_routes_impl(_request_client(), cursor=cursor, limit=limit)

async def system_disk() -> Any:
        """Report disk usage: docker totals + named-volume/model/archive/backup trees.

        Any authenticated read. Names and byte counts only (no absolute paths);
        lists orphan images and orphan data dirs (reclaim them with
        ``system_gc``). Concurrent ``du`` walks run under a soft budget — an
        overrun returns a partial body with a ``scan_timeout`` warning.
        """
        return await _system_disk_impl(_request_client())

async def system_gc(
        include_orphan_data: Annotated[
            bool,
            Field(
                description="true also deletes service data directories with no live workload row "
                "— IRREVERSIBLE. false (the default) reclaims orphan images only."
            ),
        ] = False,
        dry_run: Annotated[
            bool,
            Field(
                description="true = enumerate what would be reclaimed and remove nothing; false "
                "(the default) actually removes."
            ),
        ] = False,
        idempotency_key: Annotated[
            str | None,
            Field(
                description="Caller-chosen key: a retry with the same key replays the first "
                "result instead of running a second gc. Omit and a real run mints a fresh key per "
                "call, so each call acts; a dry run mints none and needs none."
            ),
        ] = None,
    ) -> Any:
        """Garbage-collect orphan app images (admin-scoped, audited).

        Reclaims ``nerdit-app/*`` image repos with no live workload row, verified
        by re-listing so an in-use tag is reported skipped, never removed.
        """
        return await _system_gc_impl(
            _request_client(),
            include_orphan_data=include_orphan_data,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

async def restart_daemon(
        drain_timeout_s: Annotated[
            int,
            Field(
                description="Seconds to let in-flight builds, runs and releases drain before the "
                "daemon self-execs, clamped server-side to [0, 300]; omit for 60. At 0 the "
                "SIGTERM can beat the 202 response."
            ),
        ] = 60,
        idempotency_key: Annotated[
            str | None,
            Field(
                description="Caller-chosen key; the daemon REQUIRES one and the MCP tool mints a "
                "fresh uuid4 when you omit it. A retry with the same key is safe: it answers "
                "``409 daemon.restart_in_progress`` (or replays the first 202), never a second "
                "restart mid-drain."
            ),
        ] = None,
    ) -> Any:
        """Restart the daemon to apply restart-required config (admin-scoped, audited).

        Returns 202 with drain counts. **Over the HTTP MCP transport this kills
        the connection serving this call**: a dropped response after issuing it
        means the restart most likely started. Confirm with ``capabilities``
        (``uptime_s`` resets); retrying with the same idempotency key answers
        ``409 daemon.restart_in_progress`` or a fresh 202, never a second
        restart mid-drain.
        """
        return await _restart_daemon_impl(
            _request_client(),
            drain_timeout_s=drain_timeout_s,
            idempotency_key=idempotency_key,
        )
# fmt: on


TOOLS = (
    get_audit,
    get_events,
    capabilities,
    doctor,
    proxy_status,
    list_routes,
    system_disk,
    system_gc,
    restart_daemon,
)
