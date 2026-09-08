"""Audit log read + stream endpoints.

`GET /api/audit` is an **admin-only**, cursor-paginated view of the
append-only audit trail; `GET /api/audit/stream` is the SSE projection of the
live `audit.*` events published by `nerdit.daemon.audit.AuditMiddleware`.
Both are mounted under `/api` only (no bare-root legacy alias).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

from nerdit.daemon.auth import require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.limits import _ACTION_PREFIX_MAX, valid_ts_filter
from nerdit.daemon.sse import default_frame, heartbeat_only, sse_from_bus
from nerdit.db.models import AuditLogPage, TokenRole

logger = logging.getLogger(__name__)

router = APIRouter()

#: The full alphabet of an audit action. Enforced here as well as relied on in
#: the query layer's `GLOB` matching, so `*`/`?`/`[` — the characters
#: GLOB treats as wildcards — can never reach the pattern.
_ACTION_PREFIX_RE = re.compile(rf"^[a-z0-9._]{{1,{_ACTION_PREFIX_MAX}}}$")


def _valid_action_prefix(value: str) -> str:
    """Validate `action_prefix` against the action grammar, or 400.

    A malformed **filter** is rejected rather than clamped (D-P24-12 clamps
    *bounds*; a prefix has no meaningful truncation — silently searching for
    something other than what was asked is worse than an error).
    """
    if not _ACTION_PREFIX_RE.match(value):
        raise NerditError(
            400,
            "bad_request",
            "Invalid action_prefix.",
            hint=(
                f"An action prefix is lowercase letters, digits, '.' and '_', "
                f"1-{_ACTION_PREFIX_MAX} characters (e.g. 'deploy.' or 'service')."
            ),
        )
    return value


@router.get("/audit", response_model=AuditLogPage, operation_id="list_audit")
async def list_audit(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
    action: str | None = Query(None, description="Filter by exact action"),
    result: str | None = Query(None, description="Filter by result (ok/error/denied/replay)"),
    target: str | None = Query(None, description="Filter by exact target id (e.g. a service name)"),
    target_type: str | None = Query(
        None, description="Filter by target type (e.g. service, model)"
    ),
    principal_id: str | None = Query(
        None, description="Filter by the acting principal's token id (exact match)"
    ),
    action_prefix: str | None = Query(
        None,
        description=(
            "Filter by action family — matched as a literal prefix, never a pattern "
            "(e.g. 'deploy.' returns deploy.create, deploy.git_create, deploy.plan). "
            "Grammar: lowercase letters, digits, '.' and '_', 1-40 chars."
        ),
    ),
    since: str | None = Query(
        None, description="Only entries at or after this ISO-8601 UTC timestamp (inclusive)"
    ),
    until: str | None = Query(
        None, description="Only entries at or before this ISO-8601 UTC timestamp (inclusive)"
    ),
) -> AuditLogPage:
    """Admin-only, cursor-paginated audit log (newest first).

    Filters compose. `principal_id` answers "everything this token did";
    `action_prefix` answers "everything in this family" without the caller
    enumerating action names; `since`/`until` bound the window. All four
    are applied in SQL, so narrowing never costs a full page walk.
    """
    require_role(request, TokenRole.admin)
    if action_prefix is not None:
        action_prefix = _valid_action_prefix(action_prefix)
    if since is not None:
        since = valid_ts_filter(since, "since")
    if until is not None:
        until = valid_ts_filter(until, "until")
    queries = request.app.state.queries
    try:
        items, next_cursor = await queries.list_audit_log(
            limit=limit,
            cursor=cursor,
            action=action,
            result=result,
            target=target,
            target_type=target_type,
            principal_id=principal_id,
            action_prefix=action_prefix,
            since=since,
            until=until,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuditLogPage(items=items, next_cursor=next_cursor)


@router.get("/audit/stream", operation_id="stream_audit")
async def stream_audit(request: Request):
    """Admin-only SSE stream of live `audit.*` events."""
    require_role(request, TokenRole.admin)
    event_bus = getattr(request.app.state, "event_bus", None)
    if event_bus is None:
        return EventSourceResponse(heartbeat_only(request))

    def is_audit(event: dict[str, Any]) -> bool:
        """Whether an event belongs on this stream at all."""
        return str(event.get("type", "")).startswith("audit.")

    def audit_only(event: dict[str, Any]) -> dict[str, Any] | None:
        """Keep `audit.*` frames, drop everything else.

        Belt on top of the subscription predicate's braces: the mapper is the
        last thing between the bus and the wire, and this stream is the one
        surface where a non-`audit.*` frame leaking would be a *correctness*
        bug rather than noise.
        """
        if not is_audit(event):
            return None
        return default_frame(event)

    # Filter at the SUBSCRIPTION source too, not only in the mapper: since P24a
    # every durable feed row is teed to the same bus, so a mapper-only filter
    # would enqueue the entire feed per admin stream just to throw it away.
    return EventSourceResponse(
        sse_from_bus(request, event_bus, predicate=is_audit, frame=audit_only)
    )
