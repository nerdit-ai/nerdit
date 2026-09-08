"""The ``public_urls`` seam (P26 WP-H, D-P26-14).

Two things are under test, and they are the two things the decision actually
promises:

1. **The projection matrix.** ``public_url`` (the scalar) keeps today's meaning
   everywhere, and ``public_urls`` is the additive list beside it. Every cell of
   the (share? proxy? linked? entitled?) grid is pinned here rather than in the
   route tests, because the route tests would only ever exercise one cell each.
2. **One batched query per request.** A list surface renders N services; the
   share table must be read exactly ONCE. The await-count assertions on
   ``GET /services`` and ``GET /routes`` are the whole reason
   :func:`~nerdit.daemon.views.hosted.load_hosted_context` exists as a separate
   async step in front of a pile of synchronous projections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.proxy import _domain_route_id
from nerdit.core.proxy.certs import CertStatus
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.daemon.schemas.proxy import RouteEndpoint
from nerdit.daemon.views.hosted import (
    EMPTY_HOSTED,
    HostedContext,
    domain_cert_state,
    domain_entries,
    domain_state,
    hosted_entry,
    hosted_state,
    load_hosted_context,
    public_urls_for,
)
from nerdit.db.models import (
    ApiToken,
    Job,
    JobKind,
    JobStatus,
    ServiceDomain,
    ServiceEndpoint,
    ServiceShare,
    TokenRole,
)

RO_RAW = "ro-raw"
_TOKENS = {
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    )
}

LAN_URL = "https://box/demo/"
HOSTED_URL = "https://demo--gpu-box.nodes.test/"


def _ctx(  # noqa: PLR0913 - one keyword per fact in the projection matrix
    *,
    slug: str | None = "gpu-box",
    domain: str | None = "nodes.test",
    link_state: str | None = "connected",
    entitled: bool = False,
    shares: dict[str, ServiceShare] | None = None,
    domains: dict[str, tuple[ServiceDomain, ...]] | None = None,
    live_domains: frozenset[str] | None = None,
    proxy_available: bool = True,
    withheld: frozenset[str] = frozenset(),
    https_port: int = 443,
    public_port: int | None = None,
    cert_states: dict[str, CertStatus] | None = None,
) -> HostedContext:
    """A hosted snapshot; the defaults are "linked, connected, free plan".

    ``live_domains`` is the proxy's confirmed-live domain route id set. Left
    unset it is DERIVED from ``domains`` — i.e. "a converged proxy, every bound
    domain routed" — so the tests that are about the other three dimensions read
    unchanged. The tests that are about this dimension pass it explicitly
    (Codex round 1, P2 #3831777097).
    """
    rows = domains if domains is not None else {}
    return HostedContext(
        slug=slug,
        nodes_base_domain=domain,
        link_state=link_state,
        hosted_public_entitled=entitled,
        shares=shares if shares is not None else {},
        domains=rows,
        live_domain_ids=(
            live_domains
            if live_domains is not None
            else frozenset(
                _domain_route_id(svc, row.domain) for svc, group in rows.items() for row in group
            )
        ),
        proxy_available=proxy_available,
        withheld=withheld,
        cert_states=cert_states if cert_states is not None else {},
        https_port=https_port,
        public_port=public_port,
    )


def _domain(name: str, service: str = "demo", *, acme: bool = False) -> ServiceDomain:
    return ServiceDomain(service_name=service, domain=name, acme=acme)


def _share(access: str = "private") -> ServiceShare:
    return ServiceShare(service_name="demo", access=access)


def _fake_link_manager(*, state: str = "connected", entitled: bool = False) -> SimpleNamespace:
    """A link manager stub the auth middleware also tolerates.

    ``validate_capability`` is part of the contract, not decoration: the
    ScopedTokenAuthMiddleware asks every manager on ``app.state`` whether the
    presented bearer is the live tunnel capability before falling through to the
    token table, so a stub without it turns an authorised read into a 500.
    """
    return SimpleNamespace(
        status=lambda: SimpleNamespace(state=state, hosted_public_entitled=entitled),
        validate_capability=lambda token, role: False,
    )


# --- The projection matrix ----------------------------------------------------


def test_no_share_and_no_proxy_advertises_nothing():
    """The pre-P26 default: an unrouted, unshared service has no URLs at all —
    an EMPTY list, never a list carrying a null-url entry nobody can open."""
    assert public_urls_for(_ctx(), "demo", None) == []
    assert public_urls_for(EMPTY_HOSTED, "demo", None) == []


def test_proxy_only_yields_one_default_entry_equal_to_the_scalar():
    """``public_url`` is re-projected, never recomputed: the ``default`` entry's
    url is the same string, so the two can never disagree about the LAN URL."""
    entries = public_urls_for(_ctx(), "demo", LAN_URL)
    assert len(entries) == 1
    assert entries[0].url == LAN_URL
    assert entries[0].kind == "default"
    assert entries[0].state == "ready"
    # ``access`` is a hosted-only field; a default entry must not claim one.
    assert entries[0].access is None


def test_share_on_a_connected_addressable_node_is_ready():
    entries = public_urls_for(_ctx(shares={"demo": _share()}), "demo", LAN_URL)
    assert [e.kind for e in entries] == ["default", "hosted"]
    hosted = entries[1]
    assert hosted.url == HOSTED_URL
    assert hosted.state == "ready"
    assert hosted.access == "private"


def test_share_without_a_link_manager_is_link_down_but_keeps_its_url():
    """No manager (``link_state=None``) means the tunnel is not running, but the
    node still knows its own name — so the URL is shown with an honest state
    rather than hidden. An operator must be able to see WHERE it will answer."""
    entries = public_urls_for(_ctx(link_state=None, shares={"demo": _share()}), "demo", None)
    assert len(entries) == 1
    assert entries[0].url == HOSTED_URL
    assert entries[0].state == "link_down"


def test_share_on_an_unlinked_node_is_link_down_with_a_null_url():
    """The dormant share (S8): unlinking does NOT delete the row, so the intent
    survives a re-link — but with no slug there is no address, and the entry
    carries ``url=None`` instead of a guessed string."""
    ctx = _ctx(slug=None, shares={"demo": _share()})
    assert ctx.addressable is False
    entry = hosted_entry(ctx, "demo")
    assert entry is not None
    assert entry.url is None
    assert entry.state == "link_down"


def test_share_with_no_hosted_domain_is_link_down_with_a_null_url():
    """A node linked before P26 has a slug but no ``nodes_base_domain`` until it
    refreshes — the other half of ``addressable``."""
    entry = hosted_entry(_ctx(domain=None, shares={"demo": _share()}), "demo")
    assert entry is not None
    assert entry.url is None
    assert entry.state == "link_down"


def test_public_share_without_entitlement_is_not_entitled_even_when_connected():
    """The ordering rule, pinned: ``not_entitled`` outranks ``link_down`` and
    survives a perfectly healthy tunnel. Reporting ``link_down`` here would send
    an operator to debug the wrong layer for a state the reconnect cannot fix."""
    entry = hosted_entry(_ctx(shares={"demo": _share("public")}), "demo")
    assert entry is not None
    assert entry.state == "not_entitled"
    assert entry.access == "public"
    # The URL is still composed: the share exists, it simply will not serve.
    assert entry.url == HOSTED_URL


def test_public_share_on_an_unlinked_node_still_reports_not_entitled_first():
    """Both refusals at once ⇒ the entitlement one wins (the ordering rule again,
    this time where ``link_down`` would also have been true)."""
    ctx = _ctx(slug=None, link_state=None, shares={"demo": _share("public")})
    assert hosted_state(ctx, _share("public")) == "not_entitled"
    entry = hosted_entry(ctx, "demo")
    assert entry is not None and entry.url is None


def test_entitled_public_share_on_a_connected_node_is_ready():
    """The WP-HC future: the daemon-side enforcement point does not move when
    the cloud starts sending the entitlement — only this one boolean flips."""
    entry = hosted_entry(_ctx(entitled=True, shares={"demo": _share("public")}), "demo")
    assert entry is not None
    assert entry.state == "ready"
    assert entry.access == "public"


def test_a_share_for_another_service_is_not_projected_onto_this_one():
    """The mapping is keyed by name — a page of services must not smear one
    service's share across its neighbours."""
    ctx = _ctx(shares={"other": ServiceShare(service_name="other")})
    assert hosted_entry(ctx, "demo") is None
    assert public_urls_for(ctx, "demo", LAN_URL) == [
        e for e in public_urls_for(ctx, "demo", LAN_URL) if e.kind == "default"
    ]


def test_a_disconnected_tunnel_is_link_down_not_ready():
    """``ready`` means "answers now": only the ``connected`` state qualifies, so
    a backing-off or displaced link reports honestly."""
    for state in ("connecting", "backoff", "displaced", "terminal"):
        ctx = _ctx(link_state=state, shares={"demo": _share()})
        assert hosted_state(ctx, _share()) == "link_down", state


# --- Custom domains (P26 WP1, S-W8) -------------------------------------------


def test_a_bound_domain_on_a_serving_proxy_is_ready():
    """The happy cell: the proxy is available, the app is routed, nothing is
    withheld — so the domain answers now and carries its own root URL."""
    ctx = _ctx(domains={"demo": (_domain("app.example.com"),)})
    entries = public_urls_for(ctx, "demo", LAN_URL)
    assert [e.kind for e in entries] == ["default", "domain"]
    entry = entries[1]
    assert entry.url == "https://app.example.com/"
    assert entry.state == "ready"
    assert entry.domain == "app.example.com"
    # ``access`` is a hosted-only field: a domain's exposure is decided by the
    # app's own edge_auth, not by a hosted access mode.
    assert entry.access is None


def test_a_bound_domain_with_the_proxy_off_is_withheld_but_still_listed():
    """The row IS the presence (the ``hosted`` rule): a domain bound while the
    URL layer is off must still be visible, with an honest state — an operator
    has to be able to see what the node will answer for once it is on."""
    ctx = _ctx(proxy_available=False, domains={"demo": (_domain("app.example.com"),)})
    entries = public_urls_for(ctx, "demo", None)
    assert [e.kind for e in entries] == ["domain"]
    assert entries[0].url == "https://app.example.com/"
    assert entries[0].state == "withheld"


def test_an_unrouted_service_withholds_its_domains():
    """A stopped or never-routed app has no default route for a Host matcher to
    sit beside, so a domain pointed at it cannot answer — even with the proxy
    up. NULL ``public_url`` is exactly that fact."""
    ctx = _ctx(domains={"demo": (_domain("app.example.com"),)})
    assert domain_state(ctx, "demo", "app.example.com", None) == "withheld"
    assert domain_entries(ctx, "demo", None)[0].state == "withheld"


def test_an_edge_auth_withheld_service_withholds_its_domains_too():
    """(S-W5) The fail-closed edge-auth rule reaches this surface intact: when
    the proxy withholds a service's routes because its credential reference does
    not resolve, EVERY route goes — default and domain alike — so a domain entry
    claiming ``ready`` would advertise a URL answering 404."""
    ctx = _ctx(withheld=frozenset({"demo"}), domains={"demo": (_domain("app.example.com"),)})
    assert domain_state(ctx, "demo", "app.example.com", LAN_URL) == "withheld"
    # A neighbour on the same page is unaffected — the set is keyed by name.
    other = _ctx(
        withheld=frozenset({"demo"}), domains={"other": (_domain("x.example.com", "other"),)}
    )
    assert domain_state(other, "other", "x.example.com", LAN_URL) == "ready"


def test_two_domains_keep_their_stored_order():
    """N rows, N entries, stored order preserved — both ``ready`` here because
    the proxy has confirmed both routes live."""
    ctx = _ctx(
        domains={"demo": (_domain("a.example.com"), _domain("b.example.com"))},
    )
    entries = domain_entries(ctx, "demo", LAN_URL)
    assert [e.domain for e in entries] == ["a.example.com", "b.example.com"]
    assert {e.state for e in entries} == {"ready"}


def test_a_bound_domain_whose_host_route_is_not_live_yet_is_withheld():
    """(Codex round 1, P2 #3831777097) The three facts the state used to be
    derived from are ALL true here — proxy available, app routed, nothing
    edge-auth-withheld — and the domain still does not answer, because its Host
    route has not been written. Two reachable states have that shape: the window
    between the ``PUT`` and the next reconcile tick, and — for as long as it
    lasts — a tick whose ``_converge_tls`` failed, which withholds every domain
    spec and prunes the ones already live."""
    ctx = _ctx(
        domains={"demo": (_domain("app.example.com"),)},
        live_domains=frozenset(),
    )
    assert domain_state(ctx, "demo", "app.example.com", LAN_URL) == "withheld"
    assert [e.state for e in domain_entries(ctx, "demo", LAN_URL)] == ["withheld"]


def test_only_the_domain_with_a_live_route_is_ready():
    """Per ROW, not one verdict per service: ``register`` and ``reconcile`` both
    ``continue`` past a failed per-route upsert, so a service really can have
    one live domain route and one absent."""
    ctx = _ctx(
        domains={"demo": (_domain("a.example.com"), _domain("b.example.com"))},
        live_domains=frozenset({_domain_route_id("demo", "a.example.com")}),
    )
    assert [e.state for e in domain_entries(ctx, "demo", LAN_URL)] == ["ready", "withheld"]


def test_a_live_route_for_another_service_never_makes_this_one_ready():
    """The id carries the service half, so the sets cannot smear across a page
    of services the way a bare domain set would."""
    ctx = _ctx(
        domains={"demo": (_domain("app.example.com"),)},
        live_domains=frozenset({_domain_route_id("other", "app.example.com")}),
    )
    assert domain_state(ctx, "demo", "app.example.com", LAN_URL) == "withheld"


def test_a_domain_url_follows_the_public_url_port_rule():
    """The ``public_url_for`` suffix rule verbatim: 443 is suppressed, a
    non-default ``https_port`` shows, and ``public_port`` (a port-forwarded
    daemon) wins over the bound one."""
    rows = {"demo": (_domain("app.example.com"),)}
    assert domain_entries(_ctx(domains=rows), "demo", LAN_URL)[0].url == "https://app.example.com/"
    ctx = _ctx(domains=rows, https_port=8443)
    assert domain_entries(ctx, "demo", LAN_URL)[0].url == "https://app.example.com:8443/"
    ctx = _ctx(domains=rows, https_port=8443, public_port=443)
    assert domain_entries(ctx, "demo", LAN_URL)[0].url == "https://app.example.com/"


def test_hosted_and_domain_entries_coexist_in_the_documented_order():
    """default -> hosted -> domain..., appended so every existing positional
    reading of ``public_urls`` (and every dashboard test) stays valid."""
    ctx = _ctx(
        shares={"demo": _share()},
        domains={"demo": (_domain("a.example.com"), _domain("b.example.com"))},
    )
    entries = public_urls_for(ctx, "demo", LAN_URL)
    assert [e.kind for e in entries] == ["default", "hosted", "domain", "domain"]
    assert [e.url for e in entries[2:]] == ["https://a.example.com/", "https://b.example.com/"]


def test_another_services_domains_are_not_projected_onto_this_one():
    """The mapping is keyed by name — a page of services must not smear one
    app's domains across its neighbours."""
    ctx = _ctx(domains={"other": (_domain("x.example.com", "other"),)})
    assert domain_entries(ctx, "demo", LAN_URL) == []
    assert public_urls_for(ctx, "demo", LAN_URL) == [
        PublicUrlEntry(url=LAN_URL, kind="default", state="ready")
    ]


# --- Certificate state (P26 WP2, S-W2-6) --------------------------------------


def test_a_domain_entry_carries_the_managers_cert_state():
    """The manager's reading wins whenever it has one — the projection never
    re-derives a certificate fact it was handed."""
    rows = {"demo": (_domain("app.example.com", acme=True),)}
    ctx = _ctx(domains=rows, cert_states={"app.example.com": CertStatus("issued")})

    assert domain_entries(ctx, "demo", LAN_URL)[0].cert_state == "issued"


def test_a_plain_row_is_internal_and_an_unreported_acme_row_is_pending():
    """The fail-closed fallback, both directions. ``acme=false`` is the DEFAULT
    binding and reports ``internal`` — a fact, not a defect. An ``acme=true``
    row the manager could not report on reports ``pending``, never ``issued``:
    promising a certificate nobody has seen would tell an operator to skip
    ``nerdit trust`` and leave every client on a warning page."""
    ctx = _ctx(
        domains={"demo": (_domain("plain.example.com"), _domain("pub.example.com", acme=True))}
    )

    assert [e.cert_state for e in domain_entries(ctx, "demo", LAN_URL)] == ["internal", "pending"]


def test_domain_cert_state_returns_the_expiry_for_an_issued_leaf():
    """``not_after`` rides the status object, so the pure helper is what the
    route reads for ``cert_not_after`` — one derivation, two surfaces."""
    expiry = datetime(2026, 11, 1, tzinfo=UTC)
    row = _domain("app.example.com", acme=True)
    ctx = _ctx(cert_states={"app.example.com": CertStatus("issued", expiry)})

    assert domain_cert_state(ctx, row) == CertStatus("issued", expiry)


def test_the_two_axes_are_independent():
    """``state`` is the ROUTE fact, ``cert_state`` the CERTIFICATE fact, and a
    client composes them. A domain with a public certificate whose Host route is
    not live yet must report both truths at once — fusing them into one token
    would have to invent a word for every combination."""
    row = _domain("app.example.com", acme=True)
    ctx = _ctx(
        domains={"demo": (row,)},
        live_domains=frozenset(),
        cert_states={"app.example.com": CertStatus("issued")},
    )

    entry = domain_entries(ctx, "demo", LAN_URL)[0]
    assert (entry.state, entry.cert_state) == ("withheld", "issued")


def test_default_and_hosted_entries_carry_no_cert_state():
    """The field is a domain-entry bolt-on, exactly like ``access`` and
    ``domain``: a LAN URL and a tunnelled hosted URL have no certificate this
    node issues, so ``null`` is the honest answer, not ``internal``."""
    ctx = _ctx(shares={"demo": _share()}, domains={"demo": (_domain("app.example.com"),)})

    entries = public_urls_for(ctx, "demo", LAN_URL)
    assert [e.cert_state for e in entries] == [None, None, "internal"]


def test_a_context_with_no_cert_states_at_all_still_projects():
    """The empty mapping is the pre-WP2 / proxy-off snapshot; every row falls
    back rather than raising on a read path."""
    ctx = _ctx(domains={"demo": (_domain("app.example.com"),)})
    assert ctx.cert_states == {}
    assert domain_entries(ctx, "demo", LAN_URL)[0].cert_state == "internal"


# --- One batched query per request --------------------------------------------


def _job(name: str, idx: int) -> Job:
    return Job(
        id=f"svc-{idx}",
        service_name=name,
        name=name,
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config="{}",
    )


def _services_client(queries) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)
    app.state.queries = queries
    app.state.settings = MagicMock()
    app.state.proxy_manager = None
    app.state.hostname = "box"
    app.state.link_manager = None
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="legacy-global", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {RO_RAW}"}


def test_services_list_reads_the_share_table_exactly_once_for_three_services():
    """The D-P26-14 budget. Three services on the page, ONE
    ``list_service_shares`` await and ZERO per-row ``get_service_share`` awaits —
    the point of loading a request-scoped context in front of sync projections."""
    jobs = [_job(f"app{i}", i) for i in range(3)]
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.touch_api_token = AsyncMock()
    q.list_services = AsyncMock(return_value=(jobs, None))
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.list_service_shares = AsyncMock(return_value={})
    q.get_service_share = AsyncMock(return_value=None)
    q.list_service_domains = AsyncMock(return_value=[])
    q.get_service_domains = AsyncMock(return_value=[])

    resp = _services_client(q).get("/services", headers=_auth())
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["items"]) == 3
    assert q.list_service_shares.await_count == 1
    assert q.get_service_share.await_count == 0
    # (P26 WP1 / S-W1) The second whole-table read, and only ONE of it; the
    # per-service point read is never reached from a list surface.
    assert q.list_service_domains.await_count == 1
    assert q.get_service_domains.await_count == 0


def test_services_list_projects_the_hosted_entry_from_the_batched_read():
    """The batching must not cost correctness: the one read still lands on the
    right row, and only that row."""
    jobs = [_job("demo", 1), _job("other", 2)]
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.touch_api_token = AsyncMock()
    q.list_services = AsyncMock(return_value=(jobs, None))
    q.get_service_endpoint = AsyncMock(
        side_effect=lambda name: ServiceEndpoint(
            service_name=name, job_id="svc-1", container_port=8000, host_port=9400, route=f"/{name}"
        )
    )
    q.get_job_gpus = AsyncMock(return_value=[])
    q.list_service_shares = AsyncMock(return_value={"demo": _share()})
    q.list_service_domains = AsyncMock(return_value=[])

    client = _services_client(q)
    client.app.state.settings.link = SimpleNamespace(slug="gpu-box", nodes_base_domain="nodes.test")
    client.app.state.link_manager = _fake_link_manager()

    items = client.get("/services", headers=_auth()).json()["items"]
    by_name = {i["name"]: i for i in items}
    demo = by_name["demo"]["endpoint"]["public_urls"]
    assert [e["kind"] for e in demo] == ["hosted"]
    assert demo[0]["url"] == HOSTED_URL
    assert demo[0]["state"] == "ready"
    assert by_name["other"]["endpoint"]["public_urls"] == []
    # The scalar is untouched by all of this (the D-P26-14 pin).
    assert by_name["demo"]["endpoint"]["public_url"] is None
    assert q.list_service_shares.await_count == 1


def _routes_client(queries) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(proxy_router)
    app.include_router(api)
    app.state.proxy_manager = None
    app.state.queries = queries
    app.state.settings = SimpleNamespace(
        proxy=SimpleNamespace(
            mode="path", base_domain=None, scheme="https", https_port=443, public_port=None
        ),
        link=SimpleNamespace(slug="gpu-box", nodes_base_domain="nodes.test"),
    )
    app.state.hostname = "box"
    app.state.link_manager = _fake_link_manager()
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token")
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def _route_endpoint(name: str) -> RouteEndpoint:
    return RouteEndpoint(
        service_name=name,
        kind=JobKind.service,
        status=JobStatus.running,
        container_port=8000,
        host_port=9400,
        route=f"/{name}",
    )


def test_routes_reads_the_share_table_exactly_once_and_projects_it():
    """Same budget on ``GET /routes``, plus the projection: the LAN URL and the
    hosted URL sit side by side, default first."""
    q = SimpleNamespace(
        list_service_endpoints=AsyncMock(
            return_value=([_route_endpoint("demo"), _route_endpoint("other")], None)
        ),
        list_service_shares=AsyncMock(return_value={"demo": _share()}),
        get_service_share=AsyncMock(return_value=None),
        list_service_domains=AsyncMock(return_value=[]),
        get_service_domains=AsyncMock(return_value=[]),
    )
    resp = _routes_client(q).get("/api/routes", headers={"Authorization": "Bearer admin-raw-token"})
    assert resp.status_code == 200, resp.text
    items = {i["service_name"]: i for i in resp.json()["items"]}
    assert [e["kind"] for e in items["demo"]["public_urls"]] == ["default", "hosted"]
    assert items["demo"]["public_urls"][0]["url"] == items["demo"]["public_url"]
    assert items["demo"]["public_urls"][1]["url"] == HOSTED_URL
    assert [e["kind"] for e in items["other"]["public_urls"]] == ["default"]
    assert q.list_service_shares.await_count == 1
    assert q.get_service_share.await_count == 0
    assert q.list_service_domains.await_count == 1
    assert q.get_service_domains.await_count == 0


def test_routes_reads_the_domain_table_exactly_once_and_projects_it():
    """(P26 WP1) The same budget and the same projection for domains. The proxy
    stub is available and both rows are routed, so both domains are ``ready``
    and land after the hosted entry."""
    q = SimpleNamespace(
        list_service_endpoints=AsyncMock(
            return_value=([_route_endpoint("demo"), _route_endpoint("other")], None)
        ),
        list_service_shares=AsyncMock(return_value={}),
        list_service_domains=AsyncMock(
            return_value=[
                ServiceDomain(service_name="demo", domain="a.example.com"),
                ServiceDomain(service_name="demo", domain="b.example.com"),
                ServiceDomain(service_name="other", domain="c.example.com"),
            ]
        ),
        get_service_domains=AsyncMock(return_value=[]),
    )
    client = _routes_client(q)
    client.app.state.proxy_manager = SimpleNamespace(
        enabled=True,
        available=True,
        withheld_services=frozenset(),
        live_domain_route_ids=frozenset(
            _domain_route_id(svc, dom)
            for svc, dom in (
                ("demo", "a.example.com"),
                ("demo", "b.example.com"),
                ("other", "c.example.com"),
            )
        ),
        live_routes=AsyncMock(return_value={}),
    )
    resp = client.get("/api/routes", headers={"Authorization": "Bearer admin-raw-token"})
    assert resp.status_code == 200, resp.text
    items = {i["service_name"]: i for i in resp.json()["items"]}
    demo = items["demo"]["public_urls"]
    assert [e["kind"] for e in demo] == ["default", "domain", "domain"]
    assert [e["domain"] for e in demo[1:]] == ["a.example.com", "b.example.com"]
    assert {e["state"] for e in demo[1:]} == {"ready"}
    assert [e["domain"] for e in items["other"]["public_urls"][1:]] == ["c.example.com"]
    assert q.list_service_domains.await_count == 1
    assert q.get_service_domains.await_count == 0


def test_hosted_context_without_queries_is_the_empty_snapshot():
    """The defensive read (the ``_endpoint_view`` precedent): a route test whose
    ``app.state`` has no ``queries`` must render, not explode."""
    assert EMPTY_HOSTED.addressable is False
    assert EMPTY_HOSTED.shares == {}
    assert hosted_entry(EMPTY_HOSTED, "demo") is None


def _request(**state) -> SimpleNamespace:
    """The two attributes ``load_hosted_context`` reads, and nothing else."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


async def test_load_hosted_context_without_queries_returns_empty():
    ctx = await load_hosted_context(_request())
    assert ctx is EMPTY_HOSTED


async def test_load_hosted_context_tolerates_a_pre_wp1_queries_stub():
    """(P26 WP1) The domain read is behind a ``getattr`` probe: a stub queries
    object that predates the table projects an empty domain map rather than
    raising on a read path."""
    q = SimpleNamespace(list_service_shares=AsyncMock(return_value={}))
    ctx = await load_hosted_context(_request(queries=q))
    assert ctx.domains == {}
    # And the proxy half fails closed with no manager on the state at all.
    assert ctx.proxy_available is False
    assert ctx.withheld == frozenset()
    assert ctx.live_domain_ids == frozenset()
    assert ctx.cert_states == {}


async def test_load_hosted_context_groups_domains_and_reads_the_proxy():
    """The two batched reads and the proxy snapshot, in one pass."""
    q = SimpleNamespace(
        list_service_shares=AsyncMock(return_value={}),
        list_service_domains=AsyncMock(
            return_value=[
                ServiceDomain(service_name="demo", domain="a.example.com"),
                ServiceDomain(service_name="demo", domain="b.example.com"),
                ServiceDomain(service_name="other", domain="c.example.com"),
            ]
        ),
    )
    ctx = await load_hosted_context(
        _request(
            queries=q,
            settings=SimpleNamespace(
                proxy=SimpleNamespace(scheme="https", https_port=8443, public_port=None)
            ),
            proxy_manager=SimpleNamespace(
                available=True,
                withheld_services={"demo"},
                live_domain_route_ids={_domain_route_id("other", "c.example.com")},
            ),
        )
    )
    assert [d.domain for d in ctx.domains["demo"]] == ["a.example.com", "b.example.com"]
    assert [d.domain for d in ctx.domains["other"]] == ["c.example.com"]
    assert ctx.proxy_available is True
    assert ctx.withheld == frozenset({"demo"})
    assert ctx.live_domain_ids == frozenset({_domain_route_id("other", "c.example.com")})
    assert ctx.https_port == 8443


async def test_load_hosted_context_reads_cert_states_once_over_the_rows_it_already_has():
    """(P26 WP2) The certificate read costs NO extra query: it is a synchronous
    call over the rows ``list_service_domains`` already returned, made once for
    the whole request — the batched contract this module exists to keep."""
    rows = [
        ServiceDomain(service_name="demo", domain="a.example.com", acme=True),
        ServiceDomain(service_name="other", domain="c.example.com"),
    ]
    q = SimpleNamespace(
        list_service_shares=AsyncMock(return_value={}),
        list_service_domains=AsyncMock(return_value=rows),
    )
    seen: list[list[str]] = []

    def _cert_states(given):  # noqa: ANN001, ANN202
        seen.append([row.domain for row in given])
        return {"a.example.com": CertStatus("issued")}

    ctx = await load_hosted_context(
        _request(queries=q, proxy_manager=SimpleNamespace(cert_states=_cert_states))
    )

    assert seen == [["a.example.com", "c.example.com"]]
    assert ctx.cert_states == {"a.example.com": CertStatus("issued")}
    assert q.list_service_domains.await_count == 1


async def test_load_hosted_context_tolerates_a_manager_whose_cert_states_is_not_callable():
    """``app.state.proxy_manager`` is duck-typed. A stub carrying a non-callable
    attribute of this name must degrade to "no facts" — the fail-closed
    fallback — never a 500 on a read path."""
    q = SimpleNamespace(
        list_service_shares=AsyncMock(return_value={}),
        list_service_domains=AsyncMock(return_value=[_domain("a.example.com")]),
    )
    ctx = await load_hosted_context(
        _request(queries=q, proxy_manager=SimpleNamespace(cert_states={"a.example.com": "issued"}))
    )
    assert ctx.cert_states == {}


async def test_load_hosted_context_reads_link_settings_and_status():
    q = SimpleNamespace(
        list_service_shares=AsyncMock(return_value={"demo": _share()}),
        list_service_domains=AsyncMock(return_value=[]),
    )
    ctx = await load_hosted_context(
        _request(
            queries=q,
            settings=SimpleNamespace(
                link=SimpleNamespace(slug="gpu-box", nodes_base_domain="nodes.test")
            ),
            link_manager=_fake_link_manager(),
        )
    )
    assert ctx.slug == "gpu-box"
    assert ctx.nodes_base_domain == "nodes.test"
    assert ctx.link_state == "connected"
    assert ctx.hosted_public_entitled is False
    assert set(ctx.shares) == {"demo"}


async def test_load_hosted_context_defaults_entitlement_off_when_the_status_omits_it():
    """(S5) The daemon must default to REFUSING public: a link status that does
    not carry the field yields ``False``, never a truthy accident."""
    q = SimpleNamespace(list_service_shares=AsyncMock(return_value={}))
    manager = SimpleNamespace(status=lambda: SimpleNamespace(state="connected"))
    ctx = await load_hosted_context(_request(queries=q, settings=None, link_manager=manager))
    assert ctx.hosted_public_entitled is False
    assert ctx.slug is None and ctx.nodes_base_domain is None
