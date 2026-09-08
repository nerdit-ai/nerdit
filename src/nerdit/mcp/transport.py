"""Streamable-HTTP MCP transport security wrapper + mode state (P13c §3).

Split out of :mod:`nerdit.mcp.server` (Track B WP24) — the HTTP-mount
security wrapper (:class:`_TransportGuard`), the per-request bearer
contextvar, and the mutable transport-mode globals the daemon flips at
``build_http_app()`` time.

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
_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "[::]", ""})


def _inner_hop_host(daemon_host: str) -> str:
    """Resolve the inner-hop host from the daemon's configured bind address."""
    return "127.0.0.1" if daemon_host.strip() in _WILDCARD_BIND_HOSTS else daemon_host


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
        return NerditClient(host=_HTTP_HOST, port=_HTTP_PORT, token=token)
    if _HTTP_MODE:
        raise McpHttpAuthError()
    return get_configured_client()


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
    client requires, and per-request bearer capture into the contextvar.
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
    def _reject(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={"code": code, "message": message, "detail": {}, "request_id": None},
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

        # 2. Host: mandatory; IP literal or allowlisted.
        host_header = headers.get("host")
        if host_header is None:
            await self._reject(400, "bad_request", "Host header required.")(scope, receive, send)
            return
        hostname = _host_from_header(host_header)
        if hostname is None or not _hostname_allowed(hostname, self._allowed):
            await self._reject(421, "invalid_host", "Host not permitted for this transport.")(
                scope, receive, send
            )
            return

        # 3. Origin (R4): absent → allow (the reference Python MCP client sends
        # none); present → must pass the same rule as Host.
        origin = headers.get("origin")
        if origin is not None:
            parsed = urllib.parse.urlsplit(origin)
            origin_host = parsed.hostname
            if origin_host is None or not _hostname_allowed(origin_host, self._allowed):
                await self._reject(403, "origin_forbidden", "Origin not permitted.")(
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
                await self._reject(411, "length_required", "Content-Length required.")(
                    scope, receive, send
                )
                return
            if length > self._max_body_bytes:
                await self._reject(413, "payload_too_large", "Request body too large.")(
                    scope, receive, send
                )
                return

        # 5. Bearer capture: set the contextvar for the tool bodies. A missing
        # or malformed bearer is left unset (the outer auth middleware already
        # gated it; _request_client() fails closed if one ever slipped through).
        auth = headers.get("authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token:
            ctx = _REQUEST_TOKEN.set(token)
            try:
                await self._inner(scope, receive, send)
            finally:
                _REQUEST_TOKEN.reset(ctx)
        else:
            await self._inner(scope, receive, send)
