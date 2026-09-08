"""Response schemas for the `/routes` and `/proxy/status` surfaces."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.db.enums import JobKind, JobStatus


class RouteEndpoint(BaseModel):
    """An endpoint joined to its job kind and status for route inventory.

    Include models, unrouted and terminal rows; inventory describes registrations,
    not just active routes. The route layer adds public_url and live Caddy state.
    """

    service_name: str
    kind: JobKind
    status: JobStatus
    container_port: int
    host_port: int
    active_host_port: int | None = Field(
        default=None,
        description=(
            "(P24b / D-P24-4b) Port the currently serving container is actually "
            "bound to; NULL = same as host_port. Read `live_port`, never this."
        ),
    )
    protocol: str = "tcp"
    route: str | None = Field(
        default=None,
        description=(
            "Proxy route contract: NULL = unrouted (proxy off, or a kind=model "
            'row); "/name" = path route; "" = subdomain route. Test '
            "`route is not None`, never truthiness."
        ),
    )

    @property
    def live_port(self) -> int:
        """Where this endpoint's container actually answers (D-P24-4b).

        The twin of `nerdit.db.rows.ServiceEndpoint.live_port`; the
        `/routes` projection surfaces it as `effective_host_port` beside the
        stable `host_port`.
        """
        return self.active_host_port or self.host_port


class RouteItem(BaseModel):
    """One row of `GET /routes` (1.9): a desired endpoint + live annotation.

    DB-authoritative, live-advisory (Invariant #4). `route` follows the locked
    contract (NULL = unrouted, `""` = subdomain shape, `"/name"` = path);
    `public_url` is composed via `nerdit.core.proxy.public_url_for` (the
    seam, never hand-built) and is NULL when unrouted. `live` carries the
    per-route Caddy annotation (`registered`/`dial_matches`) or is NULL when
    the live table could not be read / the proxy is off.

    (P24b, D-P24-4b surface ruling) `effective_host_port` is the port the live
    container actually answers on, and `live.dial_matches` is computed against
    it — comparing the live dial to the stable `host_port` would read a
    correctly-repointed cutover route as drifted for the whole window.
    """

    service_name: str
    kind: JobKind
    status: JobStatus
    host_port: int
    effective_host_port: int = Field(
        description=(
            "(P24b / D-P24-4b) Port the live container is ACTUALLY bound to = "
            "COALESCE(active_host_port, host_port); equal to host_port on every "
            "ordinary row. `live.dial_matches` compares against this."
        )
    )
    container_port: int
    protocol: str = "tcp"
    route: str | None = None
    public_url: str | None = None
    public_urls: list[PublicUrlEntry] = Field(
        default_factory=list,
        description=(
            "(P26 D-P26-14) Every advertised URL with its kind and state; "
            "`public_url` stays the default-kind scalar"
        ),
    )
    live: dict[str, object] | None = None


class RouteListPage(BaseModel):
    """Cursor-paginated page of routes for `GET /routes` (1.9)."""

    items: list[RouteItem]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page; null when the list is exhausted",
    )
    live_table: str = Field(
        description="Live-read state: 'readable' | 'unreadable' | 'disabled' (tri-state)"
    )


def _acme_disabled_block() -> dict[str, object]:
    """Return the complete disabled ACME status block.

    Keep acme.enabled readable even when disabled. Omit the account email from this
    any-authenticated status response; it belongs in the configuration view.
    """
    return {
        "enabled": False,
        "directory": None,
        "http_port": None,
        "http_redirect": None,
        "listening": None,
    }


class ProxyStatusResponse(BaseModel):
    """Response body for `GET /proxy/status` (1.8).

    A pure projection of `nerdit.core.proxy.ProxyManager` state through
    typed accessors + `CaddyAdmin` typed reads only (Invariant #2). Monotonic
    timers are exposed as relative seconds; `routes.live_table` and
    `apex`/`routes.count` follow tri-state discipline (null, never 0, on an
    unreadable tick).
    """

    state: str = Field(
        description="ProxyState value (disabled|no_binary|starting|backoff|"
        "foreign_conflict|available)"
    )
    enabled: bool
    available: bool
    mode: str
    base_domain: str | None = None
    hostname: str
    scheme: str
    https_port: int
    tls: dict[str, object]
    ca: dict[str, object]
    apex: dict[str, object]
    respawn: dict[str, object]
    routes: dict[str, object]
    mdns: dict[str, object]
    domains: list[dict[str, object]] = Field(
        default_factory=list,
        description=(
            "(P26 WP1 / S-W11) Every bound custom domain as CADDY sees it: "
            "{domain, service_name, acme, live, cert_state}. `live` is tri-state — "
            "true/false when the live route table was readable, null when it was "
            "not (or the proxy is off), never a state-inferred guess. `cert_state` "
            "(P26 WP2) is the certificate fact for the same row, read off Caddy's "
            "storage tree — internal|disabled|pending|issued|expired."
        ),
    )
    acme: dict[str, object] = Field(
        default_factory=_acme_disabled_block,
        description=(
            "(P26 WP2) {enabled, directory, http_port, http_redirect, listening}. "
            "The three config values are null while ACME is disabled; `listening` "
            "is the Caddy-bind fact for the HTTP-01 listener (null when unknown). "
            "The ACME account email is never projected here."
        ),
    )
