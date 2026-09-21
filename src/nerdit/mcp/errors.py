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

from nerdit.cli.client import QualifiedNameError

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

# Envelope fields already folded into the fixed error shape (a ``detail`` that
# is a plain STRING is the fallback source for ``message``; a list-shaped one
# is dropped, see :func:`_call`). Everything else the daemon attached is
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
# together — adding one without the other reopens the hole. What an agent needs
# to fix the call travels instead through :func:`_sanitized_validation_errors`,
# which copies field/type/message and never the value.
_CONSUMED_ENVELOPE_KEYS = frozenset(
    {"code", "message", "detail", "diagnostics", "request_id", "status"}
)

# The only pydantic-entry keys allowed out to an agent. ``input`` and ``ctx``
# both carry submitted values; ``url`` is a docs link of no use to a tool caller.
_VALIDATION_ENTRY_KEYS = ("loc", "type", "msg")


def _clamp(value: int, hi: int) -> int:
    """Clamp ``value`` into ``[1, hi]``."""
    return max(1, min(int(value), hi))


def _sanitized_validation_errors(detail: Any) -> list[dict[str, Any]]:
    """Project a request-model 422's pydantic error list down to leak-free entries.

    Keeps only ``loc``/``type``/``msg`` per entry, so an agent learns which
    field was rejected and why without the offending value riding along.

    Scoped to ``validation_error`` — the request-model family. The daemon's
    other 422, ``config.invalid``, carries a ``diagnostics`` list instead and
    is deliberately NOT projected: those messages are validator-authored and
    routinely interpolate the submitted value (``admin_addr '<value>' must
    be…``), and a config section is exactly where a credential lives.
    """
    if not isinstance(detail, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in detail:
        if not isinstance(item, dict):
            continue
        entry = {key: item[key] for key in _VALIDATION_ENTRY_KEYS if key in item}
        if entry:
            entries.append(entry)
    return entries


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

    A ``validation_error`` additionally gets an ``errors`` list — the daemon's
    pydantic entries stripped to ``loc``/``type``/``msg`` — because otherwise a
    request-model 422 reaches the agent as a bare "Request validation failed."
    with nothing to act on. The value-bearing keys stay behind (see
    ``_CONSUMED_ENVELOPE_KEYS``), including a list-shaped ``detail``, which is
    never stringified into ``message``.
    """
    try:
        return await coro
    except QualifiedNameError as exc:
        # (D-P40-6) `NerditClient.wire_name` runs inside the awaited client
        # method, before any request; its message is value-free.
        return _bad_request(str(exc))
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
        # ``detail`` is a fallback for ``message`` ONLY when it is a string. A
        # stock FastAPI 422 (no daemon envelope — an upstream proxy, or any
        # non-Nerdit 422) puts the pydantic error LIST there, ``input`` and all;
        # stringifying it into ``message`` would hand the agent the very values
        # ``_CONSUMED_ENVELOPE_KEYS`` exists to keep out.
        detail = body.get("detail")
        message = (
            body.get("message")
            or (detail if isinstance(detail, str) else None)
            or response.reason_phrase
        )
        error: dict[str, Any] = {
            "code": code,
            "message": message,
            "request_id": body.get("request_id"),
            "status": response.status_code,
        }
        if code == "validation_error":
            entries = _sanitized_validation_errors(detail)
            if entries:
                # Set before the merge below, which never overwrites an existing
                # key — so a body-supplied ``errors`` cannot displace this one.
                error["errors"] = entries
        for key, value in body.items():
            if key not in _CONSUMED_ENVELOPE_KEYS and key not in error:
                error[key] = value
        return {"error": error}
    except httpx.HTTPError as exc:
        return {
            "error": {
                "code": "connection_error",
                # Several httpx timeouts stringify to "", which would hand the
                # agent a contentless error; the class name at least names it.
                "message": str(exc) or type(exc).__name__,
                "request_id": None,
                "status": None,
            }
        }


def _bad_request(message: str) -> dict[str, Any]:
    """A structured error dict matching :func:`_call`'s envelope shape."""
    return {
        "error": {"code": "bad_request", "message": message, "request_id": None, "status": None}
    }
