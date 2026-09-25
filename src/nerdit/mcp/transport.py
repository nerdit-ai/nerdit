"""Streamable-HTTP MCP transport security wrapper + mode state (P13c §3).

Split out of :mod:`nerdit.mcp.server` (Track B WP24) — the HTTP-mount
security wrapper (:class:`_TransportGuard`), the per-request bearer/role
contextvars, the two HTTP-only tool-body refusals they back, and the mutable
transport-mode globals the daemon flips at ``build_http_app()`` time.

**One-directional dependency**: this module imports nothing from
``nerdit.mcp.server`` — ``server.py`` imports this module (and re-exports the
tool-facing names), never the reverse, so the two modules stay acyclic.
"""

from __future__ import annotations

import contextvars
import ipaddress
import json
import socket
import urllib.parse
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse

from nerdit.cli.client import NerditClient, get_configured_client

if TYPE_CHECKING:
    from nerdit.config.settings import NerditSettings

# --- streamable-HTTP transport state (P13c §3) --------------------------------
# The per-request bearer captured by the ASGI wrapper. Under HTTP the tool
# closures MUST use it (fail closed): FastMCP's session manager runs handlers
# in a lifespan-started task group, so a body executing outside the request
# context would otherwise silently fall back to the config-file token — i.e.
# LEGACY_ADMIN — recreating the escalation this design exists to prevent.
_REQUEST_TOKEN: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nerdit_mcp_request_token", default=None
)
# The caller's token role for this request, read off the auth middleware's scope
# state by the ASGI wrapper. Lets a tool body that touches the daemon host BEFORE
# its inner hop refuse a readonly caller up front; `None` = not resolved here.
_REQUEST_ROLE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nerdit_mcp_request_role", default=None
)
_REQUEST_PROJECT_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "nerdit_mcp_request_project_id", default=None
)
# A coroutine the ASGI wrapper builds from the scope, recording one
# `result='denied'` audit row (and its admin `audit.*` bus frame) for a refusal
# the tool body makes on its own. The two refusals below short-circuit BEFORE
# the inner loopback hop, and that hop is what used to write the denial row, so
# without this the daemon's "auth denials are themselves audit-logged"
# invariant would not hold for the one path they cover. `None` under stdio (no
# daemon app in the scope) — there is nothing to record against.
_REQUEST_DENIAL: contextvars.ContextVar[Callable[[str, int], Awaitable[None]] | None] = (
    contextvars.ContextVar("nerdit_mcp_request_denial", default=None)
)
# `TokenRole.readonly.value`, spelled literally rather than imported so this
# module keeps its single `nerdit` dependency (the client).
_READONLY_ROLE = "readonly"
# Set (never reset) by build_http_app(): the daemon process serves HTTP; the
# stdio entrypoint (`nerdit mcp`) is a different process and never sets it.
_HTTP_MODE: bool = False
# Inner hop target — the daemon's own listener. Loopback by default; a wildcard
# bind (0.0.0.0/::) still resolves to loopback, but a specific-interface
# `[daemon].host` (e.g. 192.168.1.50) is the *only* address uvicorn listens on,
# so the hop must dial it or every tool call fails with connection_error. The
# host is trusted config (never request-derived), so this is not the bind-address
# redirect the transport guards against.
_HTTP_HOST: str = "127.0.0.1"
_HTTP_PORT: int = 9321
# Wildcard binds cover loopback; prefer 127.0.0.1 for the inner hop there.
# Compared after bracket-stripping, so `::` and `[::]` are one value.
_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "::0", ""})


def _inner_hop_host(daemon_host: str) -> str:
    """Resolve the inner-hop host from the daemon's configured bind address."""
    host = daemon_host.strip().strip("[]")
    return "127.0.0.1" if host in _WILDCARD_BIND_HOSTS else host


# JSON-RPC framing never legitimately approaches this; uvicorn does not cap
# request bodies, so the wrapper enforces it (plan §3 item 5).
MAX_MCP_BODY_BYTES = 1_048_576

_HTTP_AUTH_ERROR = {
    "error": {
        "code": "unauthenticated",
        "message": (
            "The MCP HTTP transport did not capture a bearer token for this "
            "call; refusing the config-token fallback."
        ),
        "request_id": None,
        "status": 401,
    }
}


class McpHttpAuthError(RuntimeError):
    """Fail-closed guard: an HTTP-mode tool call ran without a captured bearer."""

    def __init__(self) -> None:
        super().__init__(json.dumps(_HTTP_AUTH_ERROR))


def _request_client() -> NerditClient:
    """Build the client for the current tool call.

    Contextvar token set (HTTP request path) → a client pinned to the daemon's
    own listener (``_HTTP_HOST:<port>`` — loopback unless a specific-interface
    bind forces otherwise) carrying the caller's own bearer, so role/quota/
    ownership/audit on the inner REST hop are all the real caller's.
    Unset under HTTP mode → raise (structured 401 envelope as the message) —
    never fall back to the config-file token. Unset in stdio mode → the
    config-file client, exactly today's behavior.
    """
    token = _REQUEST_TOKEN.get()
    if token is not None:
        return NerditClient(
            host=_HTTP_HOST, port=_HTTP_PORT, token=token, project_id=_REQUEST_PROJECT_ID.get()
        )
    if _HTTP_MODE:
        raise McpHttpAuthError()
    return get_configured_client()


def _role_from_scope(scope: Any) -> str | None:
    """Read the caller's token role off the ASGI scope, or ``None``.

    ``ScopedTokenAuthMiddleware`` attaches the resolved ``Principal`` to
    ``request.state``, which is the very dict the mounted sub-app's scope
    carries, so the role costs no second hop and no DB read. ``None`` whenever
    the state is absent or unresolved — the inner loopback hop stays the
    authoritative gate either way.
    """
    principal = (scope.get("state") or {}).get("principal")
    return getattr(getattr(principal, "role", None), "value", None)


def _denial_recorder(scope: Any, request_id: str) -> Callable[[str, int], Awaitable[None]] | None:
    """Build the "record this refusal" coroutine for one HTTP request.

    Closes over the daemon app and the resolved principal off the ASGI scope —
    both already there, so recording costs no extra hop and no DB read beyond
    the insert itself. The import is function-local on purpose: this module is
    imported by the stdio ``nerdit mcp`` process too, and it owes the daemon
    package nothing at import time.
    """
    app = scope.get("app")
    if app is None:
        return None
    principal = (scope.get("state") or {}).get("principal")

    async def record(rest_path: str, status_code: int) -> None:
        from nerdit.daemon.audit import derive_action, record_denial

        # Derive the action from the REST path the refused tool would have
        # called, so the row is the one the inner hop's own denial would write.
        action, target_type, target_id = derive_action("POST", rest_path)
        await record_denial(
            queries=getattr(app.state, "queries", None),
            bus=getattr(app.state, "event_bus", None),
            action=action,
            target_type=target_type,
            target_id=target_id,
            status_code=status_code,
            principal_id=getattr(principal, "token_id", None),
            principal_role=getattr(getattr(principal, "role", None), "value", None),
            request_id=request_id,
        )

    return record


async def _record_tool_denial(rest_path: str, status_code: int) -> None:
    """Record a tool-body refusal, if this request carries a recorder."""
    recorder = _REQUEST_DENIAL.get()
    if recorder is not None:
        await recorder(rest_path, status_code)


async def _readonly_write_refusal(rest_path: str) -> dict[str, Any] | None:
    """Refuse a write tool body up front when the HTTP caller is ``readonly``.

    The coarse readonly write gate exempts ``/api/mcp`` (JSON-RPC framing is not
    a mutation), so a readonly caller is normally refused by the inner loopback
    hop — but only *after* the tool body has run. A body that reads the daemon
    host before that hop must therefore check first. Returns the same envelope
    the hop would, so the caller sees exactly one refusal; ``None`` under stdio
    or when the role was not resolved. *rest_path* is the REST path this tool
    would have called, so the denial row the skipped hop used to write is still
    written.
    """
    if not _HTTP_MODE or _REQUEST_ROLE.get() != _READONLY_ROLE:
        return None
    await _record_tool_denial(rest_path, 403)
    return {
        "error": {
            "code": "forbidden",
            "message": "This token is read-only and cannot perform this operation.",
            "request_id": None,
            "status": 403,
        }
    }


async def _local_path_refusal(rest_path: str) -> dict[str, Any] | None:
    """Refuse a caller-supplied daemon-host path when serving over HTTP.

    Under the HTTP transport the tool body runs *inside the daemon*, so a
    ``path`` argument names the daemon host's filesystem, not the caller's —
    and the remote caller (the tunnel principal included) is permanently
    ``submitter``, never an operator of that box. The local ``nerdit mcp``
    stdio process is the one place the two filesystems coincide, so it keeps
    today's behavior and gets ``None``. Recorded as a denial against
    *rest_path*: a submitter probing the daemon host's paths is precisely the
    attempt an admin reading `/audit` needs to see.
    """
    if not _HTTP_MODE:
        return None
    await _record_tool_denial(rest_path, 403)
    return {
        "error": {
            "code": "mcp.local_path_unavailable",
            "message": (
                "This MCP server runs inside the daemon, so 'path' would read the "
                "daemon host's filesystem rather than yours. Path-based deploy is "
                "available only to the local 'nerdit mcp' process."
            ),
            "hint": (
                "Upload the source with write_app_files then deploy_app, or deploy "
                "from a repository with deploy_git or a starter with deploy_template."
            ),
            "request_id": None,
            "status": None,
        }
    }


def _host_from_header(value: str) -> str | None:
    """Extract the bare hostname from a ``Host``/``Origin`` authority.

    Strips a ``[v6]`` bracket form first, else drops a trailing ``:port`` only
    when the remainder is all digits (so an IPv6 literal without brackets is
    left intact). Returns ``None`` for an empty authority.
    """
    value = value.strip()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end != -1:
            return value[1:end]
    head, sep, tail = value.rpartition(":")
    if sep and head and tail.isdigit():
        return head
    return value


def _hostname_allowed(hostname: str, allowed: frozenset[str]) -> bool:
    """A Host/Origin hostname is allowed if it is an IP literal or allowlisted.

    An IP-literal authority is not a DNS-rebinding vector — rebinding needs an
    attacker *name* resolving to the victim — and auth is mandatory here anyway.
    """
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return hostname.lower() in allowed


def _allowed_hostnames(settings: NerditSettings) -> frozenset[str]:
    """Host/Origin name allowlist: loopback + every name this daemon answers to."""
    names = {"localhost"}
    host = settings.daemon.host
    if host and host not in {"0.0.0.0", "::"}:
        names.add(host.lower())  # non-wildcard bind name/IP
    if settings.proxy.hostname_override:
        names.add(settings.proxy.hostname_override.lower())
    else:
        names.add(socket.gethostname().lower())
        if settings.proxy.mdns:
            from nerdit.core.dns import default_local_hostname

            names.add(default_local_hostname().lower())
    return frozenset(names)


class _TransportGuard:
    """ASGI wrapper around the mounted streamable-HTTP app (P13c §3).

    Owns the transport-level security the generic middleware stack does not:
    Host/Origin validation (DNS-rebinding defense — deterministic across the
    mcp>=1.9 floor, unlike FastMCP's version-dependent native check), the
    request body cap, the trailing-slash normalization the reference MCP
    client requires, and per-request capture of the bearer — and of the role
    the auth middleware resolved from it — into the contextvars.
    """

    def __init__(
        self,
        inner: Any,
        *,
        allowed_hostnames: frozenset[str],
        max_body_bytes: int = MAX_MCP_BODY_BYTES,
    ) -> None:
        self._inner = inner
        self._allowed = allowed_hostnames
        # (P29) Bound once at mount time from ``[mcp].max_body_bytes``; the
        # module constant stays the default so a guard built without the kwarg
        # behaves exactly as it did pre-P29.
        self._max_body_bytes = max_body_bytes

    @staticmethod
    def _reject(status: int, code: str, message: str, request_id: str) -> JSONResponse:
        """Refuse at the transport, correlatable like every other daemon error."""
        return JSONResponse(
            status_code=status,
            content={"code": code, "message": message, "detail": {}, "request_id": request_id},
            headers={"X-Request-Id": request_id},
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._inner(scope, receive, send)
            return

        # 1. Path normalization (R5): both /api/mcp and /api/mcp/ reach the
        # inner Route "/" — no 307 the reference client hard-fails on.
        if scope.get("path") == "":
            scope = dict(scope)
            scope["path"] = "/"

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        # A transport refusal never reaches a route, so it is the one daemon
        # error with no correlation id unless the guard carries one. Reuse the
        # id `RequestIdMiddleware` already put on the scope (it wraps this
        # mount); fall back to its own header-or-mint rule when it did not run.
        request_id = (
            (scope.get("state") or {}).get("request_id")
            or headers.get("x-request-id")
            or uuid.uuid4().hex
        )

        # 2. Host: mandatory; IP literal or allowlisted.
        host_header = headers.get("host")
        if host_header is None:
            await self._reject(400, "bad_request", "Host header required.", request_id)(
                scope, receive, send
            )
            return
        hostname = _host_from_header(host_header)
        if hostname is None or not _hostname_allowed(hostname, self._allowed):
            await self._reject(
                421, "invalid_host", "Host not permitted for this transport.", request_id
            )(scope, receive, send)
            return

        # 3. Origin (R4): absent → allow (the reference Python MCP client sends
        # none); present → must pass the same rule as Host.
        origin = headers.get("origin")
        if origin is not None:
            parsed = urllib.parse.urlsplit(origin)
            origin_host = parsed.hostname
            if origin_host is None or not _hostname_allowed(origin_host, self._allowed):
                await self._reject(403, "origin_forbidden", "Origin not permitted.", request_id)(
                    scope, receive, send
                )
                return

        # 4. Body cap (R11): POST framing always carries Content-Length; a
        # chunked body has none and is refused rather than cap-while-read.
        if scope.get("method") == "POST":
            raw_len = headers.get("content-length")
            try:
                length = int(raw_len) if raw_len is not None else None
            except ValueError:
                length = None
            if length is None:
                await self._reject(411, "length_required", "Content-Length required.", request_id)(
                    scope, receive, send
                )
                return
            if length > self._max_body_bytes:
                await self._reject(413, "payload_too_large", "Request body too large.", request_id)(
                    scope, receive, send
                )
                return

        # 5. Bearer + role capture: set the contextvars for the tool bodies. A
        # missing or malformed bearer is left unset (the outer auth middleware
        # already gated it; _request_client() fails closed if one ever slipped
        # through), as is a role the auth middleware did not resolve.
        bound: list[tuple[contextvars.ContextVar[str | None], contextvars.Token[str | None]]] = []
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token:
            bound.append((_REQUEST_TOKEN, _REQUEST_TOKEN.set(token)))
        role = _role_from_scope(scope)
        if role is not None:
            bound.append((_REQUEST_ROLE, _REQUEST_ROLE.set(role)))
        principal = (scope.get("state") or {}).get("principal")
        bound.append(
            (_REQUEST_PROJECT_ID, _REQUEST_PROJECT_ID.set(getattr(principal, "project_id", None)))
        )
        recorder = _denial_recorder(scope, request_id)
        recorder_ctx = _REQUEST_DENIAL.set(recorder)
        try:
            receive = await self._project_receive(scope, receive, send, request_id)
            if receive is None:
                return
            await self._inner(scope, receive, send)
        finally:
            _REQUEST_DENIAL.reset(recorder_ctx)
            for var, ctx in reversed(bound):
                var.reset(ctx)

    async def _project_receive(self, scope, receive, send, request_id):
        # A project endpoint has no resource/prompt/global protocol
        # surface. Check framing before any registered handler runs.
        if _REQUEST_PROJECT_ID.get() is None:
            return receive
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return None
            body.extend(message.get("body", b""))
            if len(body) > self._max_body_bytes:
                await self._reject(413, "payload_too_large", "Request body too large.", request_id)(
                    scope, receive, send
                )
                return None
            if not message.get("more_body", False):
                break
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            payload = None
        method = payload.get("method") if isinstance(payload, dict) else None
        if not isinstance(method, str) or method not in {
            "initialize",
            "notifications/initialized",
            "notifications/cancelled",
            "ping",
            "tools/list",
            "tools/call",
        }:
            await _record_tool_denial("/api/project-mcp", 403)
            await self._reject(
                403,
                "project.delegation_forbidden",
                "This MCP method is not available under project-only delegation.",
                request_id,
            )(scope, receive, send)
            return None
        replayed = False

        async def replay_receive() -> Any:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        return replay_receive
