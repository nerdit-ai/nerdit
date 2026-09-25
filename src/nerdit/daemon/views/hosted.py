"""Project service exposure from one request-scoped snapshot.

The scalar public_url remains the LAN/proxy URL. The public_urls list includes
hosted and domain rows even when unreachable; state reports reachability.
Load the share and domain tables once per request, then project synchronously
to avoid per-service queries. Missing app state degrades safely to no exposure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from fastapi import Request

from nerdit.core.link.hosted import (
    hosted_host,
    hosted_label_fits,
    hosted_url,
    public_address_url,
)
from nerdit.core.proxy import _domain_route_id, domain_url_for
from nerdit.core.proxy.certs import CertStatus
from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.db.rows import ActiveServicePublicAddress, ServiceDomain, ServiceShare

#: The machine tokens a hosted share can report, shared with
#: `nerdit.daemon.schemas.exposure.PublicUrlEntry` and `ShareView` so the
#: projection and the wire contract cannot drift.
HostedState = Literal["ready", "link_down", "not_entitled", "pending"]

#: The two a custom domain can report (P26 WP1, S-W8). Deliberately NOT a
#: superset of `HostedState`: a domain is served by this node's own proxy
#: and never traverses the tunnel, so `link_down`/`not_entitled` are not
#: facts about it and would send an operator to debug the wrong layer.
DomainState = Literal["ready", "withheld"]


@dataclass(frozen=True, slots=True)
class HostedContext:
    """Everything the hosted projection needs, loaded once per request.

    Frozen because it is a snapshot: the projections below must never be able
    to mutate a fact mid-render and make two services in one list disagree
    about the state of the same link.
    """

    #: `[link].slug` — the claim result. `None` on an unlinked node.
    slug: str | None
    #: `[link].nodes_base_domain` — the cloud's hosted base domain, learned at
    #: claim time or via `nerdit link refresh`. `None` on a node linked
    #: before P26 that has not refreshed yet.
    nodes_base_domain: str | None
    #: `LinkStatus.state`, or `None` when there is no link manager at all.
    link_state: str | None
    #: Whether this node's account may serve PUBLIC hosted shares. Always
    #: `False` in WP-H: the cloud sends no per-account entitlement to the
    #: daemon yet (S5), and the enforcement point does not move when it does.
    hosted_public_entitled: bool
    #: The whole `service_shares` table, keyed by service name — the first of
    #: the two batched reads (D-P26-14, S-W1).
    shares: Mapping[str, ServiceShare]
    #: The whole `service_domains` table grouped by service name — the second
    #: batched read. Every field below it has a default, so a caller
    #: constructing a pre-WP1 context keyword-wise is unchanged.
    domains: Mapping[str, tuple[ServiceDomain, ...]] = field(default_factory=dict)
    addresses: Mapping[str, ActiveServicePublicAddress] = field(default_factory=dict)
    aliases: Mapping[str, str | None] = field(default_factory=dict)
    #: Whether the embedded proxy is up AND serving right now
    #: (`ProxyManager.available`). `False` on a daemon with the URL layer
    #: off, which is exactly when a bound domain cannot answer.
    proxy_available: bool = False
    #: Services whose route the proxy is WITHHOLDING — today, the ones whose
    #: `[deploy].edge_auth` secret does not resolve (`_edge_auth_failed`).
    #: Fail-closed: the default empty set means "nothing withheld", and a proxy
    #: manager that does not expose the property reads as such.
    withheld: frozenset[str] = frozenset()
    #: `ProxyManager.live_domain_route_ids` — the custom-domain `@id`s the
    #: proxy last confirmed live in Caddy. THE readiness fact for a domain
    #: (Codex round 1): a row exists from the moment of the PUT, the Host route
    #: only from the reconcile tick that writes it — and never while
    #: `_converge_tls` keeps failing. Fail-closed like `withheld`: empty (or
    #: a manager predating the property) means no domain answers yet.
    live_domain_ids: frozenset[str] = frozenset()
    #: `ProxyManager.cert_states` over the domain rows already read
    #: — one entry per domain, keyed by name. Empty when the manager cannot
    #: answer (a stub, or a daemon with the URL layer off), which
    #: `domain_cert_state` reads fail-closed: never `issued`.
    cert_states: Mapping[str, CertStatus] = field(default_factory=dict)
    #: `[proxy].scheme`/`https_port`/`public_port` — the three inputs
    #: `nerdit.core.proxy.domain_url_for` needs, snapshotted so the
    #: projections stay synchronous and settings-free.
    scheme: str = "https"
    https_port: int = 443
    public_port: int | None = None

    @property
    def addressable(self) -> bool:
        """Can a hosted URL even be COMPOSED for this node?

        Distinct from "does it answer": a node that was unlinked after sharing
        keeps its rows (the share goes dormant rather than being
        deleted) but has no slug/domain to build an address from, so its hosted
        entries carry `url=None` instead of a guessed string.
        """
        return bool(self.slug and self.nodes_base_domain)


#: The "nothing is hosted" snapshot. The default for every projection argument,
#: so a call site that has not (or cannot) load a context renders exactly what it
#: rendered before P26.
EMPTY_HOSTED = HostedContext(
    slug=None, nodes_base_domain=None, link_state=None, hosted_public_entitled=False, shares={}
)


async def load_hosted_context(request: Request) -> HostedContext:
    """Load exposure facts in batched reads, without per-service queries.

    Missing queries yields EMPTY_HOSTED; a missing link manager or domains accessor
    produces an unavailable link or empty domain mapping.
    """
    state = request.app.state
    queries = getattr(state, "queries", None)
    if queries is None:
        return EMPTY_HOSTED
    shares = await queries.list_service_shares()
    lister = getattr(queries, "list_service_domains", None)
    rows = await lister() if lister is not None else []
    domains: dict[str, tuple[ServiceDomain, ...]] = {}
    for row in rows:
        domains[row.service_name] = (*domains.get(row.service_name, ()), row)

    settings = getattr(state, "settings", None)
    link = getattr(settings, "link", None) if settings is not None else None
    node_id = getattr(link, "node_id", None)
    address_lister = getattr(queries, "list_service_public_addresses", None)
    addresses = await address_lister(node_id) if node_id and address_lister else {}
    alias_lister = getattr(queries, "list_service_hosted_aliases", None)
    aliases = await alias_lister(node_id) if node_id and alias_lister else {}
    manager = getattr(state, "link_manager", None)
    link_state: str | None = None
    entitled = False
    if manager is not None:
        status = manager.status()
        link_state = status.state
        # Read through `getattr` because `app.state.link_manager` is
        # duck-typed (route tests install stubs); a status object without the
        # field means "not entitled", which is the fail-closed answer. The real
        # value is `False` on every node in this WP — the cloud sends no
        # per-account entitlement to the daemon yet, so PUBLIC is refused by
        # default and WP-HC only has to flip this one boolean.
        entitled = bool(getattr(status, "hosted_public_entitled", False))

    # The proxy half, read through `getattr` for the same reason and
    # with the same fail-closed default: an unknown manager is "not available",
    # which renders every bound domain `withheld` rather than promising an
    # operator a URL that answers nothing.
    proxy_manager = getattr(state, "proxy_manager", None)
    proxy = getattr(settings, "proxy", None) if settings is not None else None
    # One call over the rows the domain read already produced — the
    # manager stats one small file per `acme=1` row and nothing at all for the
    # rest, so this adds no query and no admin round-trip to the batched
    # contract. `callable` rather than a bare `is not None` because
    # `app.state.proxy_manager` is duck-typed: a stub that happens to carry a
    # non-callable attribute of this name must degrade to "no facts", not 500.
    cert_reader = getattr(proxy_manager, "cert_states", None)
    cert_states: Mapping[str, CertStatus] = cert_reader(rows) if callable(cert_reader) else {}
    return HostedContext(
        slug=getattr(link, "slug", None),
        nodes_base_domain=getattr(link, "nodes_base_domain", None),
        link_state=link_state,
        hosted_public_entitled=entitled,
        shares=shares,
        addresses=addresses,
        aliases=aliases,
        domains=domains,
        proxy_available=bool(getattr(proxy_manager, "available", False)),
        withheld=frozenset(getattr(proxy_manager, "withheld_services", ()) or ()),
        live_domain_ids=frozenset(getattr(proxy_manager, "live_domain_route_ids", ()) or ()),
        cert_states=cert_states,
        scheme=getattr(proxy, "scheme", None) or "https",
        https_port=getattr(proxy, "https_port", None) or 443,
        public_port=getattr(proxy, "public_port", None),
    )


def _hosted_url(ctx: HostedContext, service_name: str) -> str | None:
    address = ctx.addresses.get(service_name)
    if address is not None and address.active:
        return public_address_url(address.slug)
    if ctx.addressable and hosted_label_fits(service_name, ctx.slug or ""):
        host = hosted_host(service_name, ctx.slug or "", ctx.nodes_base_domain or "")
        host_hash = hashlib.sha256(host.encode("ascii")).hexdigest()
        if host_hash in ctx.aliases and ctx.aliases[host_hash] != service_name:
            return None
        return hosted_url(service_name, ctx.slug or "", ctx.nodes_base_domain or "")
    return None


def hosted_state(ctx: HostedContext, share: ServiceShare) -> HostedState:
    """Project entitlement, address activation and current link reachability."""
    if share.access == "public" and not ctx.hosted_public_entitled:
        return "not_entitled"
    url = _hosted_url(ctx, share.service_name)
    if url is None and ctx.addressable:
        return "pending"
    if ctx.link_state == "connected" and url is not None:
        return "ready"
    return "link_down"


def hosted_entry(ctx: HostedContext, service_name: str) -> PublicUrlEntry | None:
    """Keep a shared canonical address stable across downtime and access changes."""
    share = ctx.shares.get(service_name)
    if share is None:
        return None
    return PublicUrlEntry(
        url=_hosted_url(ctx, service_name),
        kind="hosted",
        state=hosted_state(ctx, share),
        access=share.access,
    )


def domain_state(
    ctx: HostedContext, service_name: str, domain: str, public_url: str | None
) -> DomainState:
    """Return ready only for an observed, converged domain route.

    Require an available proxy, a routed service, no credential-based withholding
    and a live domain route matching the desired dial and authentication. Merely
    having a route ID is insufficient after a failed update or TLS reconciliation.
    """
    if not ctx.proxy_available or not public_url or service_name in ctx.withheld:
        return "withheld"
    if _domain_route_id(service_name, domain) not in ctx.live_domain_ids:
        return "withheld"
    return "ready"


def domain_cert_state(ctx: HostedContext, row: ServiceDomain) -> CertStatus:
    """Classify certificate trust independently of route reachability.

    Prefer the manager's observation. Without one, non-ACME domains are internal
    and ACME domains are pending, never optimistically issued.
    """
    status = ctx.cert_states.get(row.domain)
    if status is not None:
        return status
    return CertStatus("pending") if row.acme else CertStatus("internal")


def domain_entries(
    ctx: HostedContext, service_name: str, public_url: str | None
) -> list[PublicUrlEntry]:
    """One `kind='domain'` entry per bound domain, in stored (sorted) order.

    Presence tracks the ROW, exactly like `hosted_entry`: a domain bound
    while the proxy is off still appears, carrying `state='withheld'`. The
    URL is always composable (a domain IS its own address — there is no
    node-identity input that could be missing), so `url` is never `None`
    here and `access` stays `None`: exposure of a domain is decided by the
    app's own `edge_auth`, not by a hosted access mode.
    """
    rows = ctx.domains.get(service_name, ())
    if not rows:
        return []
    return [
        PublicUrlEntry(
            url=domain_url_for(
                row.domain,
                scheme=ctx.scheme,
                https_port=ctx.https_port,
                public_port=ctx.public_port,
            ),
            kind="domain",
            state=domain_state(ctx, service_name, row.domain, public_url),
            domain=row.domain,
            cert_state=domain_cert_state(ctx, row).state,
        )
        for row in rows
    ]


def public_urls_for(
    ctx: HostedContext, service_name: str, public_url: str | None
) -> list[PublicUrlEntry]:
    """List advertised URLs in stable order: default, hosted, then domains.

    Include a default entry only when public_url exists.
    """
    entries: list[PublicUrlEntry] = []
    if public_url:
        entries.append(PublicUrlEntry(url=public_url, kind="default", state="ready"))
    hosted = hosted_entry(ctx, service_name)
    if hosted is not None:
        entries.append(hosted)
    entries.extend(domain_entries(ctx, service_name, public_url))
    return entries
