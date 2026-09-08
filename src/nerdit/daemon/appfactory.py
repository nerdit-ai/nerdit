"""Assemble the FastAPI application from settings and a lifespan.

`server.create_app` retains the settings/lifespan patch points. Imports flow
from server to appfactory to routes; never import daemon.server here.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Match, Mount, get_route_path
from starlette.types import Scope

from nerdit import __version__
from nerdit.config.settings import NerditSettings
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.bodylimit import BodyLimitMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.app_config import router as app_config_router
from nerdit.daemon.routes.app_templates import router as app_templates_router
from nerdit.daemon.routes.audit import router as audit_router
from nerdit.daemon.routes.auth import router as auth_router
from nerdit.daemon.routes.cluster import router as cluster_router
from nerdit.daemon.routes.config import router as config_router
from nerdit.daemon.routes.databases import router as databases_router
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.routes.domains import router as domains_router
from nerdit.daemon.routes.events import router as events_router
from nerdit.daemon.routes.gpus import router as gpus_router
from nerdit.daemon.routes.health import router as health_router
from nerdit.daemon.routes.images import router as images_router
from nerdit.daemon.routes.license import router as license_router
from nerdit.daemon.routes.link import router as link_router
from nerdit.daemon.routes.models import router as models_router
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.routes.share import router as share_router
from nerdit.daemon.routes.system import router as system_router
from nerdit.daemon.routes.tokens import router as tokens_router
from nerdit.daemon.routes.workspaces import router as workspaces_router


class DashboardHtmlMiddleware(BaseHTTPMiddleware):
    """Serve the SPA shell for browser navigations before legacy API routes."""

    async def dispatch(self, request: Request, call_next):
        accept = request.headers.get("accept", "")
        path = request.url.path
        if (
            request.method == "GET"
            and "text/html" in accept
            and not path.startswith(("/api", "/assets"))
            and "." not in Path(path).name
        ):
            from importlib.resources import files

            index = files("nerdit.daemon") / "web" / "dist" / "index.html"
            if index.is_file():
                from fastapi.responses import FileResponse

                return FileResponse(str(index))
        return await call_next(request)


def mount_dashboard(app: FastAPI) -> None:
    """Mount the pre-built React bundle or a clear 503 fallback."""
    from importlib.resources import files

    bundle_dir = files("nerdit.daemon") / "web" / "dist"
    if bundle_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(bundle_dir), html=True), name="dashboard")
        return

    async def dashboard_not_built() -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Dashboard bundle not found. Run the frontend build before packaging."
            },
        )

    app.add_api_route("/", dashboard_not_built, methods=["GET"], include_in_schema=False)
    app.add_api_route("/{path:path}", dashboard_not_built, methods=["GET"], include_in_schema=False)


class _McpMount(Mount):
    """A `Mount` that also serves its bare mount path without a 307 (P13c §3).

    Starlette's `Mount` path regex is `^/api/mcp/(?P<path>.*)$` — it requires
    the trailing slash, so a bare `POST /api/mcp` matches no route and the
    router `redirect_slashes` issues a 307 to `/api/mcp/`. Not every MCP
    client follows that redirect, so we match the bare path here too and hand
    the sub-app the slash form (root_path `/api/mcp`, path `/api/mcp/`) so
    the inner streamable-HTTP `Route("/")` matches directly. The subtree match
    (`/api/mcp/...`) is delegated to the stock `Mount` unchanged; a sibling
    like `/api/mcpX` still matches nothing here.
    """

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        match, child_scope = super().matches(scope)
        if match is not Match.NONE:
            return match, child_scope
        if scope["type"] in ("http", "websocket") and get_route_path(scope) == self.path:
            root_path = scope.get("root_path", "")
            raw_path = scope.get("raw_path") or scope["path"].encode()
            return Match.FULL, {
                "path_params": dict(scope.get("path_params", {})),
                "app_root_path": scope.get("app_root_path", root_path),
                "root_path": root_path + self.path,
                "path": scope["path"] + "/",
                "raw_path": raw_path + b"/",
                "endpoint": self.app,
            }
        return Match.NONE, {}


def build_app(
    settings: NerditSettings,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
) -> FastAPI:
    """Assemble the FastAPI application from resolved settings + a lifespan.

    `settings`/`lifespan` are parameters (not read from module globals) so
    this module stays a pure downstream of `routes/*` — it never needs to
    import `daemon.server` for `load_settings` or the `lifespan`
    coroutine, which is what keeps the `server` → `appfactory` → `routes`
    direction acyclic.
    """
    # (P13c §3) MCP-over-HTTP preconditions — refuse to boot rather than serve
    # a misconfigured surface. token=None maps EVERY caller to the LOCAL admin
    # before any check (middleware.py:103-105) — with the HTTP transport on,
    # that is an unauthenticated admin tool-calling surface. No escape hatch.
    if settings.mcp.http_enabled:
        if settings.daemon.auth_token is None:
            raise RuntimeError(
                "[mcp].http_enabled requires [daemon].auth_token: without a token "
                "every caller is an unauthenticated local admin. Set a token or "
                "disable [mcp].http_enabled."
            )
        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError(
                "[mcp].http_enabled requires the optional 'mcp' extra. Install it "
                "with: pip install 'nerdit[mcp]' — or disable [mcp].http_enabled."
            )

    app = FastAPI(
        title="nerditd",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    register_error_handlers(app)
    # Custom middleware stack assembled inner→outer (`add_middleware` prepends),
    # so the runtime order is:
    #   RequestId → Dashboard → Auth → Audit → BodyLimit → Idempotency →
    #   Exception → router
    # - Auth (outermost of the trio) resolves the principal before Audit and
    #   Idempotency read it, and logs its own denials (those never reach Audit).
    # - Audit is outer of Idempotency, so a replay short-circuited by Idempotency
    #   still returns up through Audit and is recorded `result='replay'`.
    # - BodyLimit sits BETWEEN them (P29 review round-1): outside Idempotency so
    #   an over-large request never claims an Idempotency-Key — a body the daemon
    #   refuses unread must not poison a later legal retry, the dry-run-bypass
    #   spirit — and inside Audit so the refusal is still recorded.
    # - Idempotency is innermost: it claims/replays the key around the router,
    #   reading the request body only under the bounded D-P22-2 gate (small
    #   non-multipart, non-secret-carrying bodies) to digest it; the router still
    #   reads the body normally via Starlette's _CachedRequest.
    app.add_middleware(
        IdempotencyMiddleware,
        get_queries=lambda: getattr(app.state, "queries", None),
        require_idempotency_key=settings.security.require_idempotency_key,
    )
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(
        AuditMiddleware,
        get_queries=lambda: getattr(app.state, "queries", None),
        get_event_bus=lambda: getattr(app.state, "event_bus", None),
    )
    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=settings.daemon.auth_token,
        # Lazily read the queries layer attached to app.state during lifespan
        # startup, so scoped-token lookups resolve against the live DB.
        get_queries=lambda: getattr(app.state, "queries", None),
    )
    app.add_middleware(DashboardHtmlMiddleware)
    # Outermost custom middleware: assigns request_id before auth/handlers read it.
    app.add_middleware(RequestIdMiddleware)

    # Curated OpenAPI (P6, D4): one tag per router, applied at include time so
    # the route modules stay tag-agnostic. Every route also carries an explicit,
    # stable `operation_id` — generated agent clients pin to those ids.
    api_router = APIRouter(prefix="/api")
    for router, tag in (
        (health_router, "Health"),
        (auth_router, "Auth"),
        (gpus_router, "GPUs"),
        (cluster_router, "Cluster"),
        (events_router, "Events"),
        (images_router, "Images"),
        (audit_router, "Audit"),
        (config_router, "Config"),
        (app_config_router, "Config"),
        (tokens_router, "Tokens"),
        (services_router, "Services"),
        (deploy_router, "Deploy"),
        (secrets_router, "Secrets"),
        (models_router, "Models"),
        (databases_router, "Databases"),
        (proxy_router, "Proxy"),
        (app_templates_router, "Store"),
        (system_router, "System"),
        # The product-license install pair. Under the existing
        # System tag rather than a tag of its own: it is daemon-level custody
        # like /system/backup, and a two-operation tag would be noise in a
        # generated client. `/api` only — no legacy CLI to serve.
        (license_router, "System"),
        # The agent-workspace surface, `/api` only — a brand-new
        # content-bearing surface with no legacy CLI to serve, so no bare-root
        # alias (the `link_router` precedent).
        (workspaces_router, "Workspaces"),
        # (P27 WP-C2) Registered LAST, `/api` only — a brand-new surface with
        # no legacy CLI to serve, so it gets no bare-root alias.
        (link_router, "Link"),
        # (P26 WP-H) The hosted-share trio, `/api` only for the same reason.
        # Its own tag rather than folding into Services: exposure is the
        # surface custom domains and ACME will join, and a
        # generated client should find all of it under one heading.
        (share_router, "Exposure"),
        # The custom-domain trio, same tag and same `/api`-only
        # mount: it is the second way an operator publishes an app, and a
        # generated client should find both under one heading.
        (domains_router, "Exposure"),
    ):
        api_router.include_router(router, tags=[tag])
    app.include_router(api_router)

    # Legacy v0.2 routes stay mounted for CLI/backward compatibility. They are
    # exact aliases of the /api/* operations above (same handlers, same explicit
    # operation_ids), so they are hidden from the OpenAPI schema — documenting
    # them would duplicate every operationId. Behavior is unchanged.
    app.include_router(health_router, include_in_schema=False)
    app.include_router(gpus_router, include_in_schema=False)

    # (P13c §3) Streamable-HTTP MCP transport: opt-in, wrapped, auth mandatory
    # (NOT in _PUBLIC_PATHS). Flag off ⇒ no import, no mount, 404 — zero new
    # surface. The FastMCP instance rides app.state to the lifespan, which
    # drives its session manager (a Mount never runs a sub-app lifespan).
    if settings.mcp.http_enabled:
        from nerdit.mcp.server import build_http_app

        mcp_asgi, mcp_server = build_http_app(settings)
        # `_McpMount` (not `app.mount`) so a bare `/api/mcp` is served
        # directly instead of 307-redirecting to `/api/mcp/` (see the class
        # docstring); both spellings then reach the wrapped transport.
        app.router.routes.append(_McpMount("/api/mcp", app=mcp_asgi))
        app.state.mcp_server = mcp_server

    mount_dashboard(app)
    return app
