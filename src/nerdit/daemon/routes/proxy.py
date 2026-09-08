"""Distribute the public internal-CA root and report proxy status.

GET /proxy/ca is unauthenticated to bootstrap trust and contains no private key.
Clients verify its fingerprint out of band, as nerdit trust instructs.
"""

from fastapi import APIRouter, Request, Response

from nerdit.core.proxy import _route_id, public_url_for
from nerdit.daemon.errors import NerditError
from nerdit.daemon.views.hosted import load_hosted_context, public_urls_for
from nerdit.db.models import ProxyStatusResponse, RouteItem, RouteListPage
from nerdit.utils.certs import ca_fingerprint

router = APIRouter()

FINGERPRINT_HEADER = "X-Nerdit-Ca-Fingerprint"


@router.get(
    "/proxy/ca",
    operation_id="get_proxy_ca",
    response_class=Response,
    responses={200: {"content": {"application/x-pem-file": {}}}},
)
async def get_proxy_ca(request: Request) -> Response:
    """Serve the internal-CA root certificate (PEM) with its SHA-256 fingerprint."""
    proxy_manager = getattr(request.app.state, "proxy_manager", None)
    pem = await proxy_manager.ca_root_pem() if proxy_manager is not None else None
    if pem is None:
        raise NerditError(
            404,
            "proxy.ca_unavailable",
            "No internal CA certificate exists on this daemon.",
            hint=(
                "Enable the URL layer ([proxy].enabled = true) and restart the "
                "daemon; the CA is created when the embedded Caddy first starts."
            ),
        )
    try:
        fingerprint = ca_fingerprint(pem)
    except ValueError as exc:
        raise NerditError(
            500,
            "proxy.ca_invalid",
            "The stored internal CA certificate could not be parsed.",
            hint="Inspect <data_dir>/caddy/pki/authorities/local/root.crt on the daemon host.",
        ) from exc
    return Response(
        content=pem,
        media_type="application/x-pem-file",
        headers={FINGERPRINT_HEADER: fingerprint},
    )


@router.get("/proxy/status", operation_id="get_proxy_status", response_model=ProxyStatusResponse)
async def get_proxy_status(request: Request) -> ProxyStatusResponse:
    """Project the embedded-proxy state for agents (1.8, P13b).

    Authenticated (NOT public like `/proxy/ca`): this exposes upstream ports
    and respawn internals. A pure read — the manager composes the snapshot from
    typed accessors + `CaddyAdmin` typed reads only (Invariant #2, zero route
    writes); the `mdns` block is composed here from `[proxy].mdns`/
    `mdns_address` and the advertiser's live registration state.
    """
    manager = request.app.state.proxy_manager
    settings = getattr(request.app.state, "settings", None)
    advertiser = getattr(request.app.state, "mdns_advertiser", None)
    proxy_settings = getattr(settings, "proxy", None)

    # (P26 WP1 / S-W11) The domain rows are passed IN so the manager can
    # annotate each with Caddy-factual liveness off the live table it already
    # read — a second admin read on a status path would be one more way for a
    # slow Caddy to make an observability endpoint hang.
    queries = getattr(request.app.state, "queries", None)
    lister = getattr(queries, "list_service_domains", None)
    rows = await lister() if lister is not None else []
    snapshot = await manager.status_snapshot(domains=rows)
    snapshot.setdefault("domains", [])
    snapshot["mdns"] = {
        "enabled": bool(getattr(proxy_settings, "mdns", False)),
        "address": getattr(proxy_settings, "mdns_address", None),
        "registered": bool(getattr(advertiser, "registered", False)),
    }
    return ProxyStatusResponse(**snapshot)


@router.get("/routes", operation_id="list_routes", response_model=RouteListPage)
async def list_routes(
    request: Request, cursor: str | None = None, limit: int = 50
) -> RouteListPage:
    """List every service endpoint + its live Caddy annotation (1.9, P13b).

    DB-authoritative, live-advisory (Invariant #4): the desired set is the
    bounded `list_service_endpoints` query (models + terminal rows included);
    `public_url` is composed via the `public_url_for` seam (never
    hand-built), NULL when unrouted. The live table is read ONCE through the
    typed `CaddyAdmin` wrapper and annotates each row; on an unreadable read
    (or a disabled/down proxy) `live` stays NULL and `live_table` reflects
    the tri-state.
    """
    queries = request.app.state.queries
    try:
        endpoints, next_cursor = await queries.list_service_endpoints(cursor=cursor, limit=limit)
    except ValueError as exc:
        raise NerditError(400, "bad_request", str(exc)) from exc

    manager = getattr(request.app.state, "proxy_manager", None)
    settings = getattr(request.app.state, "settings", None)
    proxy_settings = getattr(settings, "proxy", None)
    hostname = getattr(request.app.state, "hostname", None)

    enabled = bool(getattr(manager, "enabled", False))
    available = bool(getattr(manager, "available", False))
    live_map = None
    if manager is not None and enabled and available:
        live_map = await manager.live_routes()
        live_table = "unreadable" if live_map is None else "readable"
    else:
        live_table = "disabled"

    # ONE share/link snapshot for the whole page — the same
    # batching rule the `/services` list follows.
    hosted = await load_hosted_context(request)

    items: list[RouteItem] = []
    for endpoint in endpoints:
        public_url: str | None = None
        if proxy_settings is not None and hostname:
            # `public_url_for` returns None for an unrouted row (`route` is
            # None — e.g. a model), so this is a no-op there.
            public_url = public_url_for(
                endpoint.service_name,
                endpoint.route,
                mode=proxy_settings.mode,
                hostname=hostname,
                base_domain=proxy_settings.base_domain,
                scheme=proxy_settings.scheme,
                https_port=proxy_settings.https_port,
                public_port=proxy_settings.public_port,
            )
        live: dict[str, object] | None = None
        if live_table == "readable" and manager is not None:
            # (P25, rev. 2 amendment 2) The id seam, NOT `build_route`: since
            # P25 the latter resolves the row's edge-auth secret and can raise,
            # and a READ path must never be able to trigger secret resolution.
            caddy_id = _route_id(endpoint.service_name)
            live_route = live_map.get(caddy_id) if live_map else None
            # (P24b, D-P24-4b) The dial is compared against the LIVE port, not
            # the stable allocation: during a promoted cutover the proxy dials
            # the green's ephemeral port, which is the correct converged state —
            # comparing to `host_port` would report drift for the whole window.
            live = {
                "registered": live_route is not None,
                "dial_matches": bool(
                    live_route is not None and live_route.dial == f"127.0.0.1:{endpoint.live_port}"
                ),
            }
        items.append(
            RouteItem(
                service_name=endpoint.service_name,
                kind=endpoint.kind,
                status=endpoint.status,
                host_port=endpoint.host_port,
                effective_host_port=endpoint.live_port,
                container_port=endpoint.container_port,
                protocol=endpoint.protocol,
                route=endpoint.route,
                public_url=public_url,
                public_urls=public_urls_for(hosted, endpoint.service_name, public_url),
                live=live,
            )
        )
    return RouteListPage(items=items, next_cursor=next_cursor, live_table=live_table)
