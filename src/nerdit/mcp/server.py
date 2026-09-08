"""FastMCP server exposing Nerdit as agent tools — a thin REST projection (P1 / S10).

Design notes
------------
* **Import-safe without the extra.** This module imports neither ``mcp`` nor
  FastMCP at top level. Only :func:`build_server` does, lazily, so the rest of
  the package (and its tests) load even when ``nerdit[mcp]`` is not installed.
* **Testable tool logic.** The real work lives in the ``_impl`` coroutines,
  which take an explicit :class:`~nerdit.cli.client.NerditClient` so they can be
  driven by an ``httpx`` test transport. The FastMCP-registered tools are thin
  wrappers that build a configured client and expose clean, agent-facing
  signatures (no ``client`` argument leaks into the tool schema). Split by
  domain under ``mcp/tools/`` (Track B WP24, pure motion); this module keeps
  its identity as the façade + registrar — every name below stays importable
  at this path.
* **Structured errors.** :func:`_call` (``mcp/errors.py``) normalizes
  ``httpx.HTTPStatusError`` into a flat ``{"error": {...}}`` dict that passes
  through the daemon's ``code``/``message``/``request_id`` envelope, so agents
  get machine-readable failures instead of a raised exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nerdit.mcp import transport
from nerdit.mcp.errors import _CODE_BY_STATUS as _CODE_BY_STATUS
from nerdit.mcp.errors import _bad_request as _bad_request
from nerdit.mcp.errors import _call as _call
from nerdit.mcp.errors import _clamp as _clamp
from nerdit.mcp.tools import ALL_TOOLS
from nerdit.mcp.tools._shared import DEFAULT_AUDIT_LIMIT as DEFAULT_AUDIT_LIMIT
from nerdit.mcp.tools._shared import DEFAULT_DATABASE_LIMIT as DEFAULT_DATABASE_LIMIT
from nerdit.mcp.tools._shared import DEFAULT_DIAGNOSE_LOG_TAIL as DEFAULT_DIAGNOSE_LOG_TAIL
from nerdit.mcp.tools._shared import DEFAULT_DUMP_TIMEOUT_S as DEFAULT_DUMP_TIMEOUT_S
from nerdit.mcp.tools._shared import DEFAULT_EVENT_LIMIT as DEFAULT_EVENT_LIMIT
from nerdit.mcp.tools._shared import DEFAULT_LOG_TAIL as DEFAULT_LOG_TAIL
from nerdit.mcp.tools._shared import DEFAULT_MODEL_LIMIT as DEFAULT_MODEL_LIMIT
from nerdit.mcp.tools._shared import DEFAULT_ROUTE_LIMIT as DEFAULT_ROUTE_LIMIT
from nerdit.mcp.tools._shared import DEFAULT_RUN_LOG_TAIL as DEFAULT_RUN_LOG_TAIL
from nerdit.mcp.tools._shared import DEFAULT_RUN_TIMEOUT_S as DEFAULT_RUN_TIMEOUT_S
from nerdit.mcp.tools._shared import DEFAULT_SERVICE_LIMIT as DEFAULT_SERVICE_LIMIT
from nerdit.mcp.tools._shared import MAX_AUDIT_LIMIT as MAX_AUDIT_LIMIT
from nerdit.mcp.tools._shared import MAX_DATABASE_LIMIT as MAX_DATABASE_LIMIT
from nerdit.mcp.tools._shared import MAX_DIAGNOSE_LOG_TAIL as MAX_DIAGNOSE_LOG_TAIL
from nerdit.mcp.tools._shared import MAX_EVENT_LIMIT as MAX_EVENT_LIMIT
from nerdit.mcp.tools._shared import MAX_LOG_TAIL as MAX_LOG_TAIL
from nerdit.mcp.tools._shared import MAX_MODEL_LIMIT as MAX_MODEL_LIMIT
from nerdit.mcp.tools._shared import MAX_ROUTE_LIMIT as MAX_ROUTE_LIMIT
from nerdit.mcp.tools._shared import MAX_RUN_LOG_TAIL as MAX_RUN_LOG_TAIL
from nerdit.mcp.tools._shared import MAX_SERVICE_LIMIT as MAX_SERVICE_LIMIT
from nerdit.mcp.tools.cluster import _cluster_stats_impl as _cluster_stats_impl
from nerdit.mcp.tools.cluster import _list_gpus_impl as _list_gpus_impl
from nerdit.mcp.tools.config import _apply_config_impl as _apply_config_impl
from nerdit.mcp.tools.config import _get_app_config_impl as _get_app_config_impl
from nerdit.mcp.tools.config import _get_config_impl as _get_config_impl
from nerdit.mcp.tools.config import _set_app_config_impl as _set_app_config_impl
from nerdit.mcp.tools.config import _set_config_impl as _set_config_impl
from nerdit.mcp.tools.databases import _create_database_impl as _create_database_impl
from nerdit.mcp.tools.databases import _dump_database_impl as _dump_database_impl
from nerdit.mcp.tools.databases import _list_database_dumps_impl as _list_database_dumps_impl
from nerdit.mcp.tools.databases import _list_databases_impl as _list_databases_impl
from nerdit.mcp.tools.deploy import _deploy_git_impl as _deploy_git_impl
from nerdit.mcp.tools.deploy import _deploy_impl as _deploy_impl
from nerdit.mcp.tools.deploy import _deploy_template_impl as _deploy_template_impl
from nerdit.mcp.tools.deploy import _list_app_templates_impl as _list_app_templates_impl
from nerdit.mcp.tools.deploy import _redeploy_service_impl as _redeploy_service_impl
from nerdit.mcp.tools.models import _list_models_impl as _list_models_impl
from nerdit.mcp.tools.models import _serve_model_impl as _serve_model_impl
from nerdit.mcp.tools.secrets import _list_secret_names_impl as _list_secret_names_impl
from nerdit.mcp.tools.secrets import _rm_secret_impl as _rm_secret_impl
from nerdit.mcp.tools.secrets import _set_secret_impl as _set_secret_impl
from nerdit.mcp.tools.services import _diagnose_service_impl as _diagnose_service_impl
from nerdit.mcp.tools.services import _get_service_impl as _get_service_impl
from nerdit.mcp.tools.services import _list_services_impl as _list_services_impl
from nerdit.mcp.tools.services import _remove_service_impl as _remove_service_impl
from nerdit.mcp.tools.services import _restart_service_impl as _restart_service_impl
from nerdit.mcp.tools.services import _run_command_impl as _run_command_impl
from nerdit.mcp.tools.services import _serve_impl as _serve_impl
from nerdit.mcp.tools.services import _service_logs_impl as _service_logs_impl
from nerdit.mcp.tools.services import _stop_service_impl as _stop_service_impl
from nerdit.mcp.tools.services import _wait_for_service_impl as _wait_for_service_impl
from nerdit.mcp.tools.system import _capabilities_impl as _capabilities_impl
from nerdit.mcp.tools.system import _doctor_impl as _doctor_impl
from nerdit.mcp.tools.system import _get_audit_impl as _get_audit_impl
from nerdit.mcp.tools.system import _get_events_impl as _get_events_impl
from nerdit.mcp.tools.system import _list_routes_impl as _list_routes_impl
from nerdit.mcp.tools.system import _proxy_status_impl as _proxy_status_impl
from nerdit.mcp.tools.system import _restart_daemon_impl as _restart_daemon_impl
from nerdit.mcp.tools.system import _system_disk_impl as _system_disk_impl
from nerdit.mcp.tools.system import _system_gc_impl as _system_gc_impl
from nerdit.mcp.transport import McpHttpAuthError as McpHttpAuthError

if TYPE_CHECKING:
    from nerdit.config.settings import NerditSettings


# --- FastMCP wiring (requires the optional extra) ---------------------------


def build_server(*, http: bool = False) -> Any:
    """Build and return the FastMCP server with all tools registered.

    Imports FastMCP lazily so importing this module never requires the extra.
    ``http=True`` configures the stateless streamable-HTTP transport (mounted
    by the daemon); the default stdio configuration is unchanged.
    """
    from mcp.server.fastmcp import FastMCP

    kwargs: dict[str, Any] = {}
    if http:
        # Stateless: no session resumption/event store — request/response tools
        # only (smaller attack + state surface, plan §3). streamable_http_path
        # "/" because the daemon mounts the sub-app at /api/mcp itself.
        kwargs = {"stateless_http": True, "streamable_http_path": "/"}
        try:
            # Disable FastMCP's native DNS-rebinding allowlist so our wrapper is
            # the single Host/Origin authority (R4/R8): the native list is
            # loopback-pinned (421s legitimate LAN clients) and absent on older
            # builds inside the >=1.9 floor — two layers must never disagree.
            from mcp.server.transport_security import TransportSecuritySettings

            kwargs["transport_security"] = TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            )
        except ImportError:  # pre-transport_security builds have no native check
            pass
    mcp = FastMCP("nerdit", **kwargs)

    # Registered from the domain-grouped ALL_TOOLS tuple (mcp/tools/__init__.py) —
    # each function's docstring becomes the tool description and its signature
    # the inputSchema (Track B WP24 registry collapse; byte-stable vs. the
    # nested defs this replaced, per the B0.2 golden).
    for fn in ALL_TOOLS:
        mcp.tool()(fn)

    return mcp


def build_http_app(settings: NerditSettings) -> tuple[Any, Any]:
    """Build the wrapped streamable-HTTP ASGI app and its FastMCP instance.

    Called by the daemon's ``create_app()`` when ``[mcp].http_enabled``. Flips
    the module into HTTP mode (``_request_client()`` fails closed from here
    on) and pins the inner-hop host+port. Returns ``(asgi_app, mcp_server)`` — the
    daemon mounts the former at ``/api/mcp`` and enters the latter's
    ``session_manager.run()`` in its lifespan (a Mount never runs a sub-app
    lifespan, so the daemon must).
    """
    server = build_server(http=True)
    transport._HTTP_MODE = True
    transport._HTTP_PORT = settings.daemon.port
    transport._HTTP_HOST = transport._inner_hop_host(settings.daemon.host)
    inner = server.streamable_http_app()
    guard = transport._TransportGuard(
        inner,
        allowed_hostnames=transport._allowed_hostnames(settings),
        # (P29) The one construction site — the cap is bound here, not read
        # per-request, so it is restart-required like ``http_enabled``.
        max_body_bytes=settings.mcp.max_body_bytes,
    )
    return guard, server


def run() -> None:
    """Run the MCP server over stdio (blocking)."""
    build_server().run()
