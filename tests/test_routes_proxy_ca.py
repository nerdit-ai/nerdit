"""Route tests for the public ``GET /api/proxy/ca`` trust endpoint (P9, part B).

Same harness as ``test_routes_services.py``: a hand-built FastAPI app with
mocked state on ``app.state`` (no real aiosqlite crossing event loops) plus
the real auth middleware — the load-bearing assertion here is that the
endpoint is reachable **without** credentials while the rest of the API
stays gated.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core.proxy import LiveRoute
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.proxy import FINGERPRINT_HEADER
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.db.models import JobKind, JobStatus, RouteEndpoint
from nerdit.utils.certs import ca_fingerprint, normalize_fingerprint

_AUTH = {"Authorization": "Bearer admin-raw-token"}


def _self_signed_pem() -> str:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Nerdit Test CA")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def _make_app(pem: str | None) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(proxy_router)
    app.include_router(api)
    proxy_manager = SimpleNamespace(ca_root_pem=AsyncMock(return_value=pem))
    app.state.proxy_manager = proxy_manager
    # token set => auth is enforced for every non-public path
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token")
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def test_proxy_ca_is_public_and_serves_pem_with_fingerprint():
    pem = _self_signed_pem()
    client = _make_app(pem)
    resp = client.get("/api/proxy/ca")  # no Authorization header
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-pem-file")
    assert resp.text == pem
    assert resp.headers[FINGERPRINT_HEADER] == ca_fingerprint(pem)


def test_proxy_ca_404_when_no_ca_exists():
    client = _make_app(None)
    resp = client.get("/api/proxy/ca")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "proxy.ca_unavailable"
    assert "hint" in body


def test_proxy_ca_500_on_unparseable_cert():
    client = _make_app("not a certificate")
    resp = client.get("/api/proxy/ca")
    assert resp.status_code == 500
    assert resp.json()["code"] == "proxy.ca_invalid"


def test_other_paths_stay_authenticated():
    """The public allowlist must be exactly /api/proxy/ca, not a prefix match."""
    client = _make_app(_self_signed_pem())
    resp = client.get("/api/proxy/ca/../../tokens")
    assert resp.status_code in (401, 404)
    resp = client.get("/api/proxy")
    assert resp.status_code == 401


# --- P13b WP7: GET /proxy/status + GET /routes --------------------------------


def _proxy_app(
    *,
    proxy_manager,
    queries=None,
    settings=None,
    hostname: str | None = None,
    mdns_advertiser=None,
) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(proxy_router)
    app.include_router(api)
    app.state.proxy_manager = proxy_manager
    app.state.queries = queries
    app.state.settings = settings
    app.state.hostname = hostname
    app.state.mdns_advertiser = mdns_advertiser
    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token")
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def _snapshot(**overrides) -> dict:
    snap = {
        "state": "available",
        "enabled": True,
        "available": True,
        "mode": "path",
        "base_domain": None,
        "hostname": "box",
        "scheme": "https",
        "https_port": 443,
        "tls": {"subjects": ["box"], "synced": True},
        "ca": {"fingerprint": "sha256:deadbeef", "present": True},
        "apex": {"enabled": True, "present": True, "is_last": True},
        "respawn": {"attempts": 0, "last_spawn_ago_s": None, "next_retry_in_s": None},
        "routes": {"count": 1, "live_table": "readable"},
    }
    snap.update(overrides)
    return snap


def test_proxy_status_requires_auth():
    # Inverse of the /proxy/ca public pin: /proxy/status is any-authenticated.
    mgr = SimpleNamespace(status_snapshot=AsyncMock(return_value=_snapshot()))
    client = _proxy_app(proxy_manager=mgr)
    assert client.get("/api/proxy/status").status_code == 401


def test_proxy_status_projects_snapshot_and_mdns():
    mgr = SimpleNamespace(status_snapshot=AsyncMock(return_value=_snapshot()))
    settings = SimpleNamespace(proxy=SimpleNamespace(mdns=True, mdns_address=None))
    advertiser = SimpleNamespace(registered=True)
    client = _proxy_app(proxy_manager=mgr, settings=settings, mdns_advertiser=advertiser)

    resp = client.get("/api/proxy/status", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "available"
    assert body["apex"] == {"enabled": True, "present": True, "is_last": True}
    assert body["mdns"] == {"enabled": True, "address": None, "registered": True}


def test_proxy_status_ca_absent_projects_present_false():
    snap = _snapshot(ca={"fingerprint": None, "present": False})
    mgr = SimpleNamespace(status_snapshot=AsyncMock(return_value=snap))
    settings = SimpleNamespace(proxy=SimpleNamespace(mdns=False, mdns_address=None))
    client = _proxy_app(proxy_manager=mgr, settings=settings, mdns_advertiser=None)

    body = client.get("/api/proxy/status", headers=_AUTH).json()
    assert body["ca"]["present"] is False
    assert body["ca"]["fingerprint"] is None
    assert body["mdns"] == {"enabled": False, "address": None, "registered": False}


def test_proxy_status_projects_the_san_registry_verbatim():
    # P25 WP5 item 4: the SAN registry reaches /proxy/status through the
    # snapshot's existing ``tls.subjects`` projection — the route needed no
    # change, which this pins rather than assumes.
    snap = _snapshot(tls={"subjects": ["box", "192.168.1.50", "nerd-box.local"], "synced": True})
    mgr = SimpleNamespace(status_snapshot=AsyncMock(return_value=snap))
    settings = SimpleNamespace(proxy=SimpleNamespace(mdns=False, mdns_address=None))
    client = _proxy_app(proxy_manager=mgr, settings=settings, mdns_advertiser=None)

    body = client.get("/api/proxy/status", headers=_AUTH).json()
    assert body["tls"]["subjects"] == ["box", "192.168.1.50", "nerd-box.local"]


def _routes_app(*, endpoints, next_cursor=None, available=True, live_map=None, side_effect=None):
    q = SimpleNamespace(
        list_service_endpoints=AsyncMock(
            return_value=(endpoints, next_cursor), side_effect=side_effect
        ),
        # (P26 D-P26-14) The one batched share read the route makes before its
        # loop; empty here, so ``public_urls`` carries the default entry only.
        list_service_shares=AsyncMock(return_value={}),
    )
    mgr = SimpleNamespace(
        enabled=True,
        available=available,
        live_routes=AsyncMock(return_value=live_map),
        build_route=lambda name, port: SimpleNamespace(caddy_id=f"nerdit-route-{name}"),
    )
    settings = SimpleNamespace(
        proxy=SimpleNamespace(
            mode="path", base_domain=None, scheme="https", https_port=443, public_port=None
        )
    )
    return _proxy_app(proxy_manager=mgr, queries=q, settings=settings, hostname="box")


def test_routes_requires_auth():
    client = _routes_app(endpoints=[])
    assert client.get("/api/routes").status_code == 401


def test_routes_projects_public_url_and_live_annotation():
    web = RouteEndpoint(
        service_name="web",
        kind=JobKind.service,
        status=JobStatus.running,
        container_port=8000,
        host_port=8101,
        route="/web",
    )
    model = RouteEndpoint(
        service_name="ollama-x",
        kind=JobKind.model,
        status=JobStatus.running,
        container_port=11434,
        host_port=8102,
        route=None,
    )
    live_map = {"nerdit-route-web": LiveRoute(dial="127.0.0.1:8101", shape="path")}
    client = _routes_app(endpoints=[web, model], live_map=live_map)

    body = client.get("/api/routes", headers=_AUTH).json()
    assert body["live_table"] == "readable"
    items = {i["service_name"]: i for i in body["items"]}

    assert items["web"]["route"] == "/web"
    assert items["web"]["public_url"] == "https://box/web/"
    assert items["web"]["live"] == {"registered": True, "dial_matches": True}

    # Models are loopback-only: route null, no public_url; still annotated live.
    assert items["ollama-x"]["route"] is None
    assert items["ollama-x"]["public_url"] is None
    assert items["ollama-x"]["live"] == {"registered": False, "dial_matches": False}


def test_routes_live_table_disabled_when_proxy_down():
    web = RouteEndpoint(
        service_name="web",
        kind=JobKind.service,
        status=JobStatus.running,
        container_port=8000,
        host_port=8101,
        route="/web",
    )
    client = _routes_app(endpoints=[web], available=False)

    body = client.get("/api/routes", headers=_AUTH).json()
    assert body["live_table"] == "disabled"
    # DB-authoritative public_url still projected; live annotation is null.
    item = body["items"][0]
    assert item["public_url"] == "https://box/web/"
    assert item["live"] is None


def test_routes_bad_cursor_maps_to_400():
    client = _routes_app(endpoints=[], side_effect=ValueError("Invalid cursor"))
    resp = client.get("/api/routes", params={"cursor": "bogus"}, headers=_AUTH)
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_normalize_fingerprint_accepts_openssl_shapes():
    fp = ca_fingerprint(_self_signed_pem())
    hexpart = fp.removeprefix("sha256:")
    colons = ":".join(hexpart[i : i + 2] for i in range(0, len(hexpart), 2)).upper()
    assert normalize_fingerprint(colons) == fp
    assert normalize_fingerprint(fp.upper()) == fp
    assert normalize_fingerprint(hexpart) == fp
