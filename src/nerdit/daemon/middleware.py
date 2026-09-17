"""Resolve request principals before audit, idempotency and routing.

Auth runs outside ExceptionMiddleware and records its own denied mutations.
Resolve the live link capability first as a link:<node_id> submitter, preserving
tunnel provenance even in local bypass mode. Otherwise token=None yields LOCAL,
the legacy global token yields LEGACY_ADMIN, and public paths stay public.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from nerdit.daemon.audit import derive_action, is_mcp_path, record_denial
from nerdit.daemon.auth import (
    LEGACY_ADMIN,
    LINK_TOKEN_PREFIX,
    LOCAL,
    Principal,
    hash_token,
)
from nerdit.daemon.errors import _envelope, request_id_of
from nerdit.db.models import TokenRole

logger = logging.getLogger(__name__)

# Paths that do not require authentication
_PUBLIC_PATHS = {
    "/health",
    "/api/health",
    "/docs",
    "/openapi.json",
    "/api/docs",
    "/api/openapi.json",
    # trust bootstrap — the internal-CA root PEM is served pre-auth by
    # design (the client cannot authenticate before it trusts the daemon; the
    # root cert is non-secret). GET-only route, mounted under /api only.
    "/api/proxy/ca",
}

_PUBLIC_STATIC_ROOT_FILES = {
    "/favicon.ico",
    "/manifest.webmanifest",
    "/site.webmanifest",
    "/robots.txt",
}

_PUBLIC_SHELL_PATHS = {"/", "/login"}


def _is_public(path: str, method: str) -> bool:
    """Whether this request needs no credential at all (grants no authority)."""
    return (
        path in _PUBLIC_PATHS
        or path.startswith("/assets/")
        or path in _PUBLIC_STATIC_ROOT_FILES
        or (method == "GET" and path in _PUBLIC_SHELL_PATHS)
    )


# Methods that never mutate state: skipped by the readonly gate and the
# `last_used_at` throttle (avoids serializing safe reads on the shared conn).
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

# Throttle window for the per-token `last_used_at` write (hot-path guard).
_TOUCH_INTERVAL_SECONDS = 60.0


def _expired(expires_at: datetime) -> bool:
    """Whether `expires_at` is in the past.

    A tz-naive value is only reachable through a hand-edited DB; coerce it to
    UTC rather than comparing it against an aware `now`, which would raise
    `TypeError` here — i.e. a 500 on every authenticated request.
    """
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= datetime.now(UTC)


class ScopedTokenAuthMiddleware(BaseHTTPMiddleware):
    """Resolve the bearer token to a `Principal` on `request.state`.

    When *token* is `None` (v0.1 compat / no auth configured) every request
    passes through as the `LOCAL` admin principal. Otherwise the header is
    validated, the legacy global token maps to `LEGACY_ADMIN`, and any
    other value is looked up (by SHA-256 hash) against `api_tokens`.
    """

    def __init__(
        self,
        app,  # noqa: ANN001
        token: str | None,
        get_queries: Callable[[], object | None] | None = None,
    ) -> None:
        super().__init__(app)
        self.token = token
        self._get_queries = get_queries
        self._last_touch: dict[str, float] = {}

    def _queries(self):  # noqa: ANN202
        if self._get_queries is None:
            return None
        return self._get_queries()

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        """Attach a principal and gate the request, hand-building any envelope."""
        path = request.url.path

        # (P27 WP-C1 item 3) Tunnel capability → the synthetic node-link
        # principal. This branch is FIRST by security requirement, not by
        # taste:
        #
        # (i) ORDERING. In local mode (`self.token is None`) the next branch
        #     grants LOCAL — an *admin* principal. A request arriving through
        #     the relay must never become a local admin: the tunnel's role is
        #     permanently `submitter` (D-R2 as amended by ADR-W1) and that
        #     ceiling is structural, enforced here at the single resolution
        #     choke point. Moving this branch below either the local shortcut
        #     or the legacy compare silently re-opens admin over the tunnel.
        # (ii) PROVENANCE. `token_id = "link:<node_id>"` IS the checklist's
        #     tunnel-provenance audit field — no schema change: every
        #     AuditMiddleware row and every `_deny` row for a tunnelled
        #     request already carries it in `principal_id` (+ `submitter`
        #     in `principal_role`), filterable via GET /audit?principal_id=.
        # (iii) COST. `validate_capability` is `secrets.compare_digest`
        #     against the capability proven on the LIVE connection (plus the
        #     role ceiling and the reactive-expiry recheck), so this pre-check
        #     adds no timing oracle; a bearer that is not the capability falls
        #     through to the normal resolution chain completely unchanged.
        #
        # `_touch` is skipped deliberately — there is no `api_tokens` row
        # to stamp — and so is the readonly gate (submitter mutates).
        auth_header = request.headers.get("authorization", "")
        if auth_header:
            link_principal = self._link_principal(request, auth_header)
            if link_principal is not None:
                request.state.principal = link_principal
                return await call_next(request)
            denial = await self._refuse_stale_tunnel_bearer(request, auth_header)
            if denial is not None:
                return denial

        # No token configured → local principal, skip auth (v0.1 backward compat).
        if self.token is None:
            request.state.principal = LOCAL
            return await call_next(request)

        # Public paths are always accessible (attach LOCAL defensively).
        if _is_public(path, request.method):
            request.state.principal = LOCAL
            return await call_next(request)

        rid = request_id_of(request)
        if not auth_header:
            return await self._deny(
                request,
                401,
                "unauthenticated",
                "Authentication required. Send an Authorization: Bearer <token> header.",
                rid,
            )

        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return await self._deny(
                request,
                401,
                "invalid_auth_format",
                "Invalid Authorization header format. Use: Bearer <token>.",
                rid,
            )
        presented = parts[1]

        # Legacy global token → admin principal (no breaking change).
        #
        # Compared as **bytes**, never as `str`: `compare_digest` raises
        # `TypeError` on a `str` holding any non-ASCII character, and
        # `presented` comes straight out of an attacker-controlled
        # `Authorization` header (Starlette decodes headers as latin-1, so
        # `Bearer é` is a reachable input). Raised here the error would escape
        # the FastAPI exception handlers — auth is outermost — as a bare 500
        # with no envelope and no denial audit row. On the encoded form it is
        # the `False` it always meant, so such a bearer falls through to the
        # ordinary `invalid_token` denial. `LinkManager.validate_capability`
        # fixes the same hazard on the tunnel path. `[daemon].auth_token` is
        # ASCII-validated at settings load, so no legitimate credential is
        # affected.
        if secrets.compare_digest(
            presented.encode("utf-8", "replace"), self.token.encode("utf-8", "replace")
        ):
            request.state.principal = LEGACY_ADMIN
            return await call_next(request)

        # Scoped token: resolve by hash against api_tokens.
        principal = await self._scoped_principal(presented)
        if principal is None:
            return await self._deny(
                request,
                403,
                "invalid_token",
                "Invalid token. Check your authentication token.",
                rid,
            )

        # Expiry at the single resolution choke point. Placed
        # BEFORE `request.state.principal` is attached, BEFORE `_touch` (an
        # expired token must not stamp `last_used_at`) and BEFORE the readonly
        # gate (expiry applies to every method, including safe reads, and to the
        # MCP mount — the readonly gate's boundary-exact MCP exemption is about
        # framing vs mutation and does NOT extend to expiry). The principal is
        # passed to `_deny` so the denial row keeps real attribution.
        if principal.expires_at is not None and _expired(principal.expires_at):
            return await self._deny(
                request,
                403,
                "token_expired",
                f"This token expired on {principal.expires_at.isoformat()}. An expired "
                "token cannot rotate itself — ask an admin for a new one "
                "('nerdit token create'). Rotate tokens BEFORE they expire with "
                "'nerdit token rotate'.",
                rid,
                principal=principal,
            )

        request.state.principal = principal
        await self._touch(principal, request.method)

        # Coarse role gate: readonly tokens may not perform mutating methods.
        # Fine-grained owner/admin checks stay in the routes (S4).
        # (P13c §3) The MCP mount is exempted boundary-exactly: an MCP POST is
        # transport framing, not a mutation — every actual mutation happens on
        # the inner loopback hop carrying the same token, where this gate + the
        # route owner/admin checks + audit all run normally. Authentication is
        # untouched (/api/mcp is NOT public); denials elsewhere are unaffected.
        if (
            principal.role == TokenRole.readonly
            and request.method not in _SAFE_METHODS
            and not is_mcp_path(path)
        ):
            return await self._deny(
                request,
                403,
                "forbidden",
                "This token is read-only and cannot perform this operation.",
                rid,
                principal=principal,
            )

        return await call_next(request)

    async def _scoped_principal(self, presented: str) -> Principal | None:
        """Resolve a presented token against `api_tokens` by SHA-256 hash."""
        queries = self._queries()
        if queries is None:
            return None
        row = await queries.get_api_token_by_hash(hash_token(presented))
        if row is None:
            return None
        return Principal(
            token_id=row.id,
            name=row.name,
            role=row.role,
            max_gpus=row.max_gpus,
            max_concurrent_jobs=row.max_concurrent_jobs,
            scope_services=(
                frozenset(row.scope_services) if row.scope_services is not None else None
            ),
            expires_at=row.expires_at,
        )

    async def _refuse_stale_tunnel_bearer(
        self, request: Request, auth_header: str
    ) -> JSONResponse | None:
        """Close the tunnel→LOCAL-admin fall-through.

        In local mode (`self.token is None`) the branch after this one grants
        LOCAL — an **admin** principal — to anything, bearer or not. On a
        link-enabled daemon that is a real window, not a theoretical one: the
        mux validates the injected capability at `open_stream` and this
        middleware re-checks it up to `REQUEST_BODY_TIMEOUT_S` (~30 s) later
        once a chunked body has been assembled, so a capability that lapses in
        between arrives here as a `Bearer` that just failed
        `_link_principal` — and would be promoted to admin.

        Returns a 403 for that case and `None` for everything else. Public
        paths are exempt: they grant no authority, and a monitor polling
        `/api/health` must not break because a caller still carries an old
        token.
        """
        if self.token is not None:
            return None
        if getattr(request.app.state, "link_manager", None) is None:
            return None
        if _is_public(request.url.path, request.method):
            return None
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return await self._deny(
            request,
            403,
            "invalid_token",
            "Invalid token. Check your authentication token.",
            request_id_of(request),
        )

    @staticmethod
    def _link_principal(request: Request, auth_header: str) -> Principal | None:
        """Resolve a presented bearer against the live tunnel capability.

        Returns the synthetic node-link principal when the header carries
        exactly the capability the daemon proved on its CURRENT relay
        connection, `None` for everything else (no link manager, a
        non-`Bearer` header, any other token). `None` is not a denial — the
        caller falls through to the ordinary resolution chain.
        """
        manager = getattr(request.app.state, "link_manager", None)
        if manager is None:
            return None
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        if not manager.validate_capability(parts[1], "submitter"):
            return None
        # The node id is the audit provenance; `status()` is a pure snapshot.
        # `LINK_TOKEN_PREFIX` marks the id as synthetic (no `api_tokens`
        # row) so row-backed quota reads route around it (`is_link_token_id`).
        return Principal(
            token_id=f"{LINK_TOKEN_PREFIX}{manager.status().node_id}",
            name="node-link",
            role=TokenRole.submitter,
        )

    async def _deny(
        self,
        request: Request,
        status_code: int,
        code: str,
        message: str,
        rid: str | None,
        *,
        principal: Principal | None = None,
    ) -> JSONResponse:
        """Record a denial audit row (best-effort) and return the envelope."""
        await self._record_denial(request, status_code, principal)
        # Pre-route denials carry `detail` as an empty OBJECT, matching the MCP
        # transport guard's shape (mcp/server.py). Route-level NerditError
        # envelopes deliberately keep the message-string `detail` alias
        # (errors.py:_envelope) for backward compatibility.
        return JSONResponse(
            status_code=status_code,
            content=_envelope(code, message, request_id=rid, detail={}),
        )

    async def _record_denial(
        self, request: Request, status_code: int, principal: Principal | None
    ) -> None:
        """Append a `result='denied'` audit row for a rejected request.

        Auth is outermost, so AuditMiddleware never sees the requests Auth
        rejects — recording here keeps denied attempts on the record. Best
        effort: an audit failure must never mask the original denial.
        """
        action, target_type, target_id = derive_action(request.method, request.url.path)
        # One writer for both halves (row + the admin-gated `audit.*` mirror an
        # admin watching live depends on), shared with the MCP transport's
        # pre-body refusals — see `audit.record_denial`.
        await record_denial(
            queries=self._queries(),
            bus=getattr(request.app.state, "event_bus", None),
            action=action,
            target_type=target_type,
            target_id=target_id,
            status_code=status_code,
            principal_id=principal.token_id if principal else None,
            principal_role=principal.role.value if principal else None,
            request_id=request_id_of(request),
        )

    async def _touch(self, principal: Principal, method: str) -> None:
        """Throttled `last_used_at` update; skipped on safe methods.

        A per-request `UPDATE … + commit` serializes against the GPU/quota
        `BEGIN IMMEDIATE` transactions on the single shared connection, so it
        is throttled to ~60s and never fires for GET/HEAD/OPTIONS.
        """
        if method in _SAFE_METHODS or principal.token_id is None:
            return
        now = time.monotonic()
        last = self._last_touch.get(principal.token_id)
        # First touch for this token always fires; only *subsequent* touches are
        # throttled. (Comparing against a 0.0 default is wrong on a freshly
        # booted host where `monotonic()` is itself < the interval.)
        if last is not None and now - last < _TOUCH_INTERVAL_SECONDS:
            return
        self._last_touch[principal.token_id] = now
        queries = self._queries()
        if queries is None:
            return
        try:
            await queries.touch_api_token(principal.token_id)
        except Exception:
            logger.warning("Failed to update token last_used_at", exc_info=True)


# Backward-compatible alias: existing imports of `BearerAuthMiddleware` keep
# working. The class was renamed to reflect its scoped-token behaviour.
BearerAuthMiddleware = ScopedTokenAuthMiddleware
