"""Error normalization + bounds helpers shared by every MCP tool implementation.

Holds the transport/HTTP error → structured-dict translation, plus the small
bounds/validation helpers that ride alongside it. A leaf module — imports
nothing from ``nerdit.mcp`` itself, so every domain module under
``mcp/tools/`` can depend on it without risking an import cycle back to
``server``.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import httpx

# Stable error codes derived from HTTP status for responses that lack a daemon
# envelope (e.g. an upstream proxy or a non-Nerdit error page).
_CODE_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    500: "internal_error",
}

# Envelope fields already folded into the fixed error shape (``detail`` is the
# fallback source for ``message``). Everything else the daemon attached is
# merged through verbatim — see :func:`_call`.
#
# ``diagnostics`` is dropped for the same reason as ``detail``, and the reason is
# a leak, not tidiness: ``_validation_exception_handler`` (``daemon/errors.py``)
# writes the pydantic error list into BOTH fields, and a pydantic v2 error entry
# carries the offending ``input`` — i.e. the caller's own submitted values. On
# HTTP that echo is accepted as a same-caller, never-persisted limitation. It
# must NOT be widened to MCP, where the value would land in an agent's
# transcript: a 422 on ``run_command(env=…)`` or ``set_secret`` would
# otherwise hand the rejected secret straight back. The two must be dropped
# together — adding one without the other reopens the hole.
_CONSUMED_ENVELOPE_KEYS = frozenset(
    {"code", "message", "detail", "diagnostics", "request_id", "status"}
)


def _clamp(value: int, hi: int) -> int:
    """Clamp ``value`` into ``[1, hi]``."""
    return max(1, min(int(value), hi))


async def _call(coro: Awaitable[Any]) -> Any:
    """Await a client call, normalizing transport/HTTP errors to a dict.

    Success passes the parsed JSON through unchanged. An HTTP error becomes a
    structured ``{"error": {code, message, request_id, status, ...}}`` dict that
    reuses the daemon's envelope fields when present; a connection-level failure
    becomes a ``connection_error``.

    Those four keys are fixed, but the envelope's *extras* ride along instead of
    being dropped, because they are the actionable half of the error: ``hint``
    carries the remediation text ("raise ``[services].max_concurrent_runs``",
    "deploy it first"), and several codes attach a key an agent is meant to
    branch on — ``not_ready_kind`` on ``run.not_ready``, ``dependents`` on
    ``resource.in_use``, ``log_tail``/``container_started`` on
    ``run.interrupted``. Only keys the daemon actually sent are copied, and
    never over one of the four, so the shape an existing caller reads is stable.
    """
    try:
        return await coro
    except httpx.HTTPStatusError as exc:
        response = exc.response
        body: dict[str, Any] = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:  # noqa: BLE001 — non-JSON error body
            body = {}
        code = body.get("code") or _CODE_BY_STATUS.get(response.status_code, "error")
        message = body.get("message") or body.get("detail") or response.reason_phrase
        error: dict[str, Any] = {
            "code": code,
            "message": message,
            "request_id": body.get("request_id"),
            "status": response.status_code,
        }
        for key, value in body.items():
            if key not in _CONSUMED_ENVELOPE_KEYS and key not in error:
                error[key] = value
        return {"error": error}
    except httpx.HTTPError as exc:
        return {
            "error": {
                "code": "connection_error",
                "message": str(exc),
                "request_id": None,
                "status": None,
            }
        }


def _bad_request(message: str) -> dict[str, Any]:
    """A structured error dict matching :func:`_call`'s envelope shape."""
    return {
        "error": {"code": "bad_request", "message": message, "request_id": None, "status": None}
    }
