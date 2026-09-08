"""Pin application routes, middleware order, aliases and mount placement.

Cover the pinned REST operation table, MCP enablement and dashboard placement.
Patch server.load_settings to avoid operator config, and access middleware via
the server facade so extraction into appfactory preserves the checks.
"""

from __future__ import annotations

import itertools

import pytest
from fastapi.routing import APIRoute
from starlette.routing import Mount

import nerdit.daemon.server as server_module
from nerdit.config.settings import NerditSettings
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.bodylimit import BodyLimitMiddleware
from nerdit.daemon.errors import RequestIdMiddleware
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware

# The router-inclusion order from server.py's ``api_router`` tag table,
# collapsed to unique consecutive tags. Mirrors test_openapi.py's
# ``_EXPECTED_TAGS`` set: 19 routers share 18 distinct tags because
# ``config_router`` and ``app_config_router`` are adjacent and both "Config".
_EXPECTED_TAG_ORDER = [
    "Health",
    "Auth",
    "GPUs",
    "Cluster",
    "Events",
    "Images",
    "Audit",
    "Config",
    "Tokens",
    "Services",
    "Deploy",
    "Secrets",
    "Models",
    "Databases",
    "Proxy",
    "Store",
    "System",
    "Workspaces",
    "Link",
    # (P26 WP-H) The hosted-share trio, registered last on its own tag.
    "Exposure",
]

# The pinned {operation_id: (path, sorted methods)} table under /api.
# Dropped, renamed or duplicated operations and path/method drift fail here.
_EXPECTED_OPERATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "apply_daemon_config": ("/api/config/daemon/apply", ("POST",)),
    "check_auth": ("/api/auth/check", ("GET",)),
    "claim_link": ("/api/link/claim", ("POST",)),
    "create_backup": ("/api/system/backup", ("POST",)),
    "create_database": ("/api/databases", ("POST",)),
    "create_database_dump": ("/api/databases/{name}/dump", ("POST",)),
    "create_service": ("/api/services", ("POST",)),
    "create_token": ("/api/tokens", ("POST",)),
    "get_self_token": ("/api/tokens/self", ("GET",)),
    "create_volume_backup": ("/api/system/backup/volumes", ("POST",)),
    "delete_service": ("/api/services/{ident}", ("DELETE",)),
    "delete_service_secret": ("/api/secrets/{service}/{key}", ("DELETE",)),
    "delete_service_secrets": ("/api/secrets/{service}", ("DELETE",)),
    "deploy_app": ("/api/deploy", ("POST",)),
    "deploy_app_template": ("/api/app-templates/{template_id}/deploy", ("POST",)),
    "deploy_git_app": ("/api/deploy/git", ("POST",)),
    "diagnose_service": ("/api/services/{ident}/diagnose", ("GET",)),
    "get_app_config": ("/api/config/apps/{name}", ("GET",)),
    "get_app_template": ("/api/app-templates/{template_id}", ("GET",)),
    "get_capabilities": ("/api/capabilities", ("GET",)),
    "get_cluster_info": ("/api/cluster/info", ("GET",)),
    "get_cluster_stats": ("/api/cluster/stats", ("GET",)),
    "get_daemon_config": ("/api/config/daemon", ("GET",)),
    "get_daemon_config_section": ("/api/config/daemon/{section}", ("GET",)),
    "get_doctor": ("/api/doctor", ("GET",)),
    "get_health": ("/api/health", ("GET",)),
    "get_proxy_ca": ("/api/proxy/ca", ("GET",)),
    "get_proxy_status": ("/api/proxy/status", ("GET",)),
    "get_service": ("/api/services/{ident}", ("GET",)),
    "get_service_logs": ("/api/services/{ident}/logs", ("GET",)),
    "get_service_stats": ("/api/services/{ident}/stats", ("GET",)),
    "get_system_disk": ("/api/system/disk", ("GET",)),
    "install_license": ("/api/license", ("POST",)),
    "list_app_templates": ("/api/app-templates", ("GET",)),
    "list_audit": ("/api/audit", ("GET",)),
    "list_databases": ("/api/databases", ("GET",)),
    "list_database_dumps": ("/api/databases/{name}/dumps", ("GET",)),
    "list_events": ("/api/events", ("GET",)),
    "list_gpus": ("/api/gpus", ("GET",)),
    "list_images": ("/api/images", ("GET",)),
    "list_models": ("/api/models", ("GET",)),
    "list_routes": ("/api/routes", ("GET",)),
    "list_secret_names": ("/api/secrets/{service}", ("GET",)),
    "list_services": ("/api/services", ("GET",)),
    "list_tokens": ("/api/tokens", ("GET",)),
    "put_app_config_section": ("/api/config/apps/{name}/{section}", ("PUT",)),
    "put_daemon_config_section": ("/api/config/daemon/{section}", ("PUT",)),
    "redeploy_app": ("/api/deploy/{name}/redeploy", ("POST",)),
    "restart_daemon": ("/api/daemon/restart", ("POST",)),
    "restart_service": ("/api/services/{ident}/restart", ("POST",)),
    "revoke_token": ("/api/tokens/{token_id}", ("DELETE",)),
    "rotate_self_token": ("/api/tokens/self/rotate", ("POST",)),
    "remove_license": ("/api/license", ("DELETE",)),
    "rollback_deploy": ("/api/deploy/{name}/rollback", ("POST",)),
    "restore_database_dump": ("/api/databases/{name}/restore", ("POST",)),
    "rotate_secrets_key": ("/api/secrets/rotate-key", ("POST",)),
    "run_service_command": ("/api/services/{ident}/run", ("POST",)),
    "run_system_gc": ("/api/system/gc", ("POST",)),
    "serve_model": ("/api/models", ("POST",)),
    "set_service_secrets": ("/api/secrets/{service}", ("POST",)),
    "stop_service": ("/api/services/{ident}/stop", ("POST",)),
    "unlink_node": ("/api/link", ("DELETE",)),
    "stream_audit": ("/api/audit/stream", ("GET",)),
    "stream_cluster_events": ("/api/events/stream", ("GET",)),
    "stream_service_logs": ("/api/services/{ident}/logs/stream", ("GET",)),
    "wait_for_service": ("/api/services/{ident}/wait", ("GET",)),
    "write_workspace_files": ("/api/workspaces/{name}/files", ("PUT",)),
    "get_workspace": ("/api/workspaces/{name}", ("GET",)),
    "read_workspace_file": ("/api/workspaces/{name}/files/{path:path}", ("GET",)),
    "deploy_workspace": ("/api/workspaces/{name}/deploy", ("POST",)),
    # (P26 WP-H) The hosted-share trio (tag Exposure) plus the hosted-metadata
    # re-read, which stays a link-custody op under the Link tag.
    "refresh_link": ("/api/link/refresh", ("POST",)),
    # (P32) The cloud's entitlement push — the one route written by the cloud
    # rather than by an operator. Registered inside the link router, so it sits
    # in the same contiguous Link tag group and the tag ORDER is unchanged.
    "start_device_link": ("/api/link/device", ("POST",)),
    "poll_device_link": ("/api/link/device/poll", ("POST",)),
    "push_link_entitlement": ("/api/link/entitlement", ("PUT",)),
    "push_link_github_token": ("/api/link/github-token", ("PUT",)),
    "nudge_git": ("/api/link/git-nudge", ("POST",)),
    "get_share": ("/api/services/{name}/share", ("GET",)),
    "set_share": ("/api/services/{name}/share", ("PUT",)),
    "remove_share": ("/api/services/{name}/share", ("DELETE",)),
    # (P26 WP1) The custom-domain trio, same Exposure tag, two paths (the
    # domain is the resource id).
    "list_domains": ("/api/services/{name}/domains", ("GET",)),
    # ``:path`` so a pasted URL reaches the validator (422 not_bare), not a 404.
    "add_domain": ("/api/services/{name}/domains/{domain:path}", ("PUT",)),
    "remove_domain": ("/api/services/{name}/domains/{domain:path}", ("DELETE",)),
}


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    """A ``create_app()`` build against a stock ``NerditSettings()`` — MCP-over-HTTP
    off (the ``McpSettings.http_enabled`` default), so no config file, real or
    otherwise, is ever read here."""
    monkeypatch.setattr(server_module, "load_settings", lambda: NerditSettings())
    return server_module.create_app()


def _iter_api_routes(built_app):
    """Yield ``(absolute_path, APIRoute)`` pairs in registration order.

    fastapi >= 0.141 made ``include_router`` lazy: ``app.routes`` holds
    ``_IncludedRouter`` wrappers (no ``APIRoute`` instances), and the include
    context — not the route — carries prefix/tags/include_in_schema.
    ``iter_route_contexts`` resolves the wrappers to absolute paths; older
    fastapi flattens eagerly, so ``app.routes`` itself is the table.
    """
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:  # fastapi < 0.141
        for r in built_app.routes:
            if isinstance(r, APIRoute):
                yield r.path, r
        return
    for ctx in iter_route_contexts(built_app.routes):
        if isinstance(ctx.route, APIRoute):
            yield ctx.path, ctx.route


def _api_routes(built_app) -> list[tuple[str, APIRoute]]:
    return [(path, r) for path, r in _iter_api_routes(built_app) if path.startswith("/api/")]


def test_api_router_and_tag_table(app):
    """The /api operationId set is exact, and the 19-router tag order survives
    (collapsed to 18 unique consecutive tags — see _EXPECTED_TAG_ORDER)."""
    routes = _api_routes(app)
    ids = [r.operation_id for _, r in routes]
    # 79 = the Track B 57 + the P25 self-token pair (get_self_token,
    # rotate_self_token) + the P27 WP-C2 link pair (claim_link, unlink_node)
    # + the P17d license pair (install_license, remove_license) + the P29
    # workspace quartet (write_workspace_files, get_workspace,
    # read_workspace_file, deploy_workspace) + the P26 WP-H exposure trio
    # (get_share, set_share, remove_share) and refresh_link + the P26 WP1
    # custom-domain trio (list_domains, add_domain, remove_domain) + the P32
    # cloud-written push_link_entitlement + the P33 cloud-written pair
    # (push_link_github_token, nudge_git) + P34 D1's device pair
    # (start_device_link, poll_device_link); the pre-auth key adds none — it
    # rides the claim route as a second grant (D-P34-6).
    #
    # P33 and P34 were built in parallel off the same 75-operation baseline and
    # each independently pinned 77, so this literal is the one number the merge
    # had to derive from the built app rather than from either branch's
    # arithmetic (P34 plan §4, "whichever lands second reconciles").
    #
    # P37's managed-database dump trio (create_database_dump,
    # list_database_dumps, restore_database_dump — all under the existing
    # Databases tag) takes it to 82.
    assert len(ids) == len(set(ids)) == 82, "operation_id set drifted from 82"

    found = {r.operation_id: (path, tuple(sorted(r.methods))) for path, r in routes}
    assert found == _EXPECTED_OPERATIONS

    # fastapi >= 0.141 keeps include-context tags off the route object; the
    # OpenAPI schema is the version-stable projection of the merged tags.
    tag_of = {
        op["operationId"]: op["tags"][0]
        for ops in app.openapi()["paths"].values()
        for op in ops.values()
        if op.get("operationId") and op.get("tags")
    }
    tags_seq = [tag_of[r.operation_id] for _, r in routes if r.operation_id in tag_of]
    collapsed = [tag for tag, _ in itertools.groupby(tags_seq)]
    assert collapsed == _EXPECTED_TAG_ORDER

    # Config is shared by two adjacent routers (config_router + app_config_router);
    # assert both landed under the tag, not just that the tag string appears once.
    config_paths = {path for path, r in routes if tag_of.get(r.operation_id) == "Config"}
    assert any(p.startswith("/api/config/daemon") for p in config_paths)
    assert any(p.startswith("/api/config/apps") for p in config_paths)


def test_middleware_order(app):
    """``add_middleware`` prepends; the runtime order is the reverse of
    registration — pinned per the server.py comment block.

    ``BodyLimitMiddleware`` sits between Audit and Idempotency (P29 review
    round-1): inside Audit so a refusal is still recorded, outside Idempotency
    so an over-large body never claims a key it would then poison.
    """
    assert [m.cls for m in app.user_middleware] == [
        RequestIdMiddleware,
        server_module.DashboardHtmlMiddleware,
        ScopedTokenAuthMiddleware,
        AuditMiddleware,
        BodyLimitMiddleware,
        IdempotencyMiddleware,
    ]


def test_legacy_bare_root_aliases(app):
    """``/health`` and ``/gpus`` stay mounted bare (CLI back-compat), hidden
    from the OpenAPI schema, same operation_id as their /api twins."""
    # The dashboard bundle is CI-built, not committed: a source checkout runs
    # mount_dashboard's fallback branch, which registers the 503 "not built"
    # handler at "/" and "/{path:path}" as plain API routes. Those are the
    # dashboard, not legacy aliases — exclude them either way.
    bare = {
        path: r
        for path, r in _iter_api_routes(app)
        if not path.startswith("/api/")
        and getattr(r.endpoint, "__name__", "") != "dashboard_not_built"
    }
    assert set(bare) == {"/health", "/gpus"}
    schema_paths = set(app.openapi()["paths"])
    for path, expected_op in (("/health", "get_health"), ("/gpus", "list_gpus")):
        route = bare[path]
        # fastapi >= 0.141 keeps ``include_in_schema=False`` on the include
        # context, not the route — schema absence is the version-stable pin.
        assert path not in schema_paths
        assert route.operation_id == expected_op


def test_mcp_mount_absent_when_http_disabled(app):
    """[mcp].http_enabled defaults False: no ``_McpMount``, ``/api/mcp`` is
    unmounted (the enabled path is test_mcp_http.py's, needs the mcp extra)."""
    assert not any(isinstance(r, server_module._McpMount) for r in app.routes)


def test_dashboard_mount_is_last_route(app):
    """The dashboard mount — or its 503 "not built" fallback — is registered
    last, so no later route can shadow it."""
    last = app.routes[-1]
    if isinstance(last, Mount):
        assert last.name == "dashboard"
    else:
        assert isinstance(last, APIRoute)
        assert last.path == "/{path:path}"
        assert last.endpoint.__name__ == "dashboard_not_built"
        assert last.include_in_schema is False
