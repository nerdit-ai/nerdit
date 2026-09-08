"""Exercise the live Caddy admin API; skip when caddy is unavailable.

Pin path-prefix stripping, absent-route deletion and pruning of removed services.
Verify one wildcard internal-CA certificate with SNI for a multi-label base domain.
Adopting path-mode Caddy after a mode flip must converge host routes, empty route
projections and the TLS automate list. Mock Caddy cannot validate these contracts.
Use ephemeral listener ports; the admin API uses localhost:2019.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import shutil
import socket
import ssl
import threading

import httpx
import pytest

from nerdit.config.settings import ProxySettings
from nerdit.core.proxy import ProxyManager
from nerdit.db.models import ActiveServiceRoute

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("caddy") is None, reason="caddy binary not installed"),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RecordingStub:
    """A loopback HTTP server that records the request paths it receives."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.port = _free_port()
        paths = self.paths

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                paths.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args: object) -> None:  # silence
                pass

        self._srv = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self) -> _RecordingStub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._srv.shutdown()


def _manager(
    tmp_path,
    *,
    mode: str = "path",
    base_domain: str | None = None,
    queries: object | None = None,
    admin_addr: str | None = None,
    https_port: int | None = None,
) -> ProxyManager:
    settings = ProxySettings(
        enabled=True,
        mode=mode,
        # A free admin port so the smoke is isolated from any system Caddy on the
        # default :2019 (the apt package ships an active caddy.service). The flip
        # smoke passes an explicit admin_addr/https_port so two managers can talk
        # to the SAME live Caddy.
        admin_addr=admin_addr or f"localhost:{_free_port()}",
        https_port=https_port or _free_port(),
        scheme="https",
        base_domain=base_domain,
        hostname_override="localhost",
    )
    # queries is unused on the paths these tests hit except audit; a no-DB run is
    # fine — audit failures are swallowed. Pass a stub with the one attribute used.
    return ProxyManager(queries or _NoAudit(), settings, hostname="localhost", data_dir=tmp_path)


def _subdomain_manager(tmp_path, **kw) -> ProxyManager:
    """The P3.5 variant: Host-matched routes + ``*.dev.localhost`` wildcard TLS.

    The base MUST be multi-label: RFC 6125 (enforced by OpenSSL, so curl,
    Python, browsers) rejects a wildcard with fewer than two labels after the
    ``*`` — ``*.localhost`` never validates ``myapp.localhost``. Caddy ≤2.7
    masked this by also minting an exact-name leaf per Host-matched route;
    ≥2.8 skips per-name issuance when a wildcard covers the name and serves
    the wildcard itself, so a single-label base fails strict clients.
    ``*.dev.localhost`` still resolves to loopback (RFC 6761 ``.localhost``
    handling is suffix-based) and satisfies the two-label rule everywhere.
    """
    return _manager(tmp_path, mode="subdomain", base_domain="dev.localhost", **kw)


def _resolves(host: str) -> bool:
    """``.localhost`` subdomain resolution is common but not universal — guard."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


class _NoAudit:
    """Minimal queries stub: route audit + DB reads no-op (smoke is proxy-only).

    (P26 WP1) ``domains`` is the in-memory stand-in for ``service_domains``:
    the reconcile tick reads the whole table once and the TLS convergence reads
    it through the same method, so one list drives both.
    """

    domains: list = []

    async def insert_audit_log(self, **_kw: object) -> None:
        return None

    async def set_endpoint_route(self, *_a: object) -> None:
        return None

    async def list_active_service_routes(self) -> list:
        return []

    async def list_service_domains(self) -> list:
        return list(self.domains)

    async def get_service_domains(self, service_name: str) -> list:
        return [d for d in self.domains if d.service_name == service_name]


class _FlipQueries(_NoAudit):
    """Queries stub for the flip smoke: a fixed desired set + recorded route writes."""

    def __init__(self, entries: list[ActiveServiceRoute]) -> None:
        self.entries = entries
        self.route_writes: list[tuple[str, str]] = []

    async def list_active_service_routes(self) -> list[ActiveServiceRoute]:
        return list(self.entries)

    async def set_endpoint_route(self, service_name: str, route: str) -> None:
        self.route_writes.append((service_name, route))


async def _wait_proxy(client: httpx.AsyncClient, url: str, attempts: int = 20) -> httpx.Response:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return await client.get(url, timeout=5.0)
        except Exception as exc:  # TLS provisioning / connect races
            last = exc
            await asyncio.sleep(0.5)
    raise AssertionError(f"proxy never answered {url}: {last}")


async def _wait_file(path, attempts: int = 40):
    """Wait for Caddy to materialize a storage file (e.g. the internal root CA)."""
    for _ in range(attempts):
        if path.exists():
            return path
        await asyncio.sleep(0.25)
    raise AssertionError(f"{path} never appeared")


async def test_smoke_register_strips_prefix(tmp_path):
    mgr = _manager(tmp_path)
    https_port = mgr._settings.https_port
    with _RecordingStub() as stub:
        try:
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.register("myapp", stub.port)
            async with httpx.AsyncClient(verify=False) as client:
                resp = await _wait_proxy(client, f"https://localhost:{https_port}/myapp/sub")
            assert resp.status_code == 200
            # handle_path / strip_path_prefix: the upstream sees "/sub", not "/myapp/sub".
            assert "/sub" in stub.paths
            assert "/myapp/sub" not in stub.paths
        finally:
            await mgr.stop()


async def test_smoke_delete_route_is_idempotent(tmp_path):
    mgr = _manager(tmp_path)
    with _RecordingStub() as stub:
        try:
            await mgr.start()
            await mgr.register("myapp", stub.port)
            assert await mgr._admin.delete_route("nerdit-route-myapp") is True
            # Second delete: Caddy reports an unknown id; we treat it as success/False.
            assert await mgr._admin.delete_route("nerdit-route-myapp") is False
        finally:
            await mgr.stop()


async def test_smoke_reconcile_prunes_orphan(tmp_path):
    mgr = _manager(tmp_path)
    try:
        await mgr.start()
        # A nerdit route with no backing service (e.g. a stale leftover).
        ghost = mgr._caddy_route_obj(mgr.build_route("ghost", 65000))
        await mgr._admin.upsert_route(ghost)
        assert "nerdit-route-ghost" in await mgr._admin.live_routes()
        # Desired set is empty (_NoAudit.list_active_service_routes → []) → prune.
        await mgr.reconcile()
        assert "nerdit-route-ghost" not in await mgr._admin.live_routes()
    finally:
        await mgr.stop()


# ---------------------------------------------------------------------------- #
# P3.5 — subdomain mode (D1 wildcard TLS, D2 adopt-path TLS sync, flip UX).     #
# ---------------------------------------------------------------------------- #


async def test_smoke_subdomain_host_routing_and_wildcard_tls(tmp_path):
    """Host-matched routing + ONE wildcard internal-CA cert, proven via real SNI.

    Requests ``https://myapp.dev.localhost:<port>/sub`` by its REAL hostname —
    not a Host-header-against-127.0.0.1 trick — so wildcard cert selection is
    genuinely driven by SNI, and verifies against the internal root CA.
    """
    if not _resolves("myapp.dev.localhost"):
        pytest.skip("myapp.dev.localhost does not resolve on this host")
    mgr = _subdomain_manager(tmp_path)
    https_port = mgr._settings.https_port
    with _RecordingStub() as stub:
        try:
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.register("myapp", stub.port)
            root_crt = await _wait_file(
                tmp_path / "caddy" / "pki" / "authorities" / "local" / "root.crt"
            )
            # Verify against root.crt: the handshake only succeeds if Caddy issued
            # (and SNI-selected) a ``*.localhost`` cert chained to the internal CA.
            ctx = ssl.create_default_context(cafile=str(root_crt))
            async with httpx.AsyncClient(verify=ctx) as client:
                resp = await _wait_proxy(client, f"https://myapp.dev.localhost:{https_port}/sub")
            assert resp.status_code == 200
            # Subdomain mode never strips: the upstream sees "/sub" untouched.
            assert "/sub" in stub.paths
        finally:
            await mgr.stop()


async def test_smoke_flip_adoption_reshapes_route_and_syncs_tls(tmp_path):
    """Mode flip over a still-running Caddy: adopt, reshape, rewrite, TLS-sync.

    Emulates the real flip: a path-mode daemon dies (Caddy keeps running), the
    restarted daemon comes up in subdomain mode on the same admin_addr, adopts,
    and one reconcile tick converges everything (D2 + D3 end to end).
    """
    admin_addr = f"localhost:{_free_port()}"
    https_port = _free_port()
    upstream = _free_port()
    path_mgr = _manager(tmp_path, admin_addr=admin_addr, https_port=https_port)
    sub_queries = _FlipQueries(
        [
            ActiveServiceRoute(
                service_name="myapp", host_port=upstream, status="running", route="/myapp"
            )
        ]
    )
    sub_mgr = _subdomain_manager(
        tmp_path, queries=sub_queries, admin_addr=admin_addr, https_port=https_port
    )
    try:
        await path_mgr.start()
        assert path_mgr.available, "Caddy did not come up"
        await path_mgr.register("myapp", upstream)
        live = await path_mgr._admin.live_routes()
        assert live["nerdit-route-myapp"].shape == "path"
        tls = await path_mgr._admin.get_tls_config()
        assert tls["certificates"]["automate"] == ["localhost"]
        # "Daemon restart": the manager goes away, Caddy stays up. Close only the
        # admin client — mgr.stop() would kill the Caddy process itself.
        await path_mgr._admin.aclose()

        await sub_mgr.start()
        assert sub_mgr.available, "adoption failed"
        await sub_mgr.reconcile()

        # Route object is now host-shaped (Host matcher, direct proxy, no strip)
        # and — critically — REPLACED, not duplicated: real Caddy 2.6.2 appends
        # a same-@id sibling on POST /id (the P3.5 runbook bug), so the upsert
        # must PATCH. count==1 is the end-to-end proof.
        live = await sub_mgr._admin.live_routes()
        assert live["nerdit-route-myapp"].shape == "host"
        assert live["nerdit-route-myapp"].count == 1
        resp = await sub_mgr._admin._client.get("/id/nerdit-route-myapp")
        obj = resp.json()
        assert obj["match"] == [{"host": ["myapp.dev.localhost"]}]
        assert obj["handle"][0]["handler"] == "reverse_proxy"
        # The persisted projection was rewritten "/myapp" → "" (subdomain contract).
        assert sub_queries.route_writes == [("myapp", "")]
        # D2: the adopted Caddy's LIVE automate list now carries the wildcard.
        tls = await sub_mgr._admin.get_tls_config()
        assert set(tls["certificates"]["automate"]) == {"localhost", "*.dev.localhost"}
    finally:
        try:
            await path_mgr._admin.aclose()
        except Exception:
            pass
        await sub_mgr.stop()


# ---------------------------------------------------------------------------- #
# P9.5 — dashboard apex (G4): one HTTPS name serves dashboard + services.       #
# ---------------------------------------------------------------------------- #


class _DashboardStub:
    """A loopback HTTP server standing in for the daemon at the apex.

    Records request paths like :class:`_RecordingStub`, plus a ``/sse`` path that
    streams a fixed number of ``text/event-stream`` events (then closes) so the
    smoke can prove the apex forwards SSE through Caddy.
    """

    def __init__(self, events: int = 3) -> None:
        self.paths: list[str] = []
        self.port = _free_port()
        paths = self.paths
        n = events

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                paths.append(self.path)
                if self.path == "/sse":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for i in range(n):
                        self.wfile.write(f"data: tick-{i}\n\n".encode())
                        self.wfile.flush()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"dashboard")

            def log_message(self, *args: object) -> None:  # silence
                pass

        self._srv = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self) -> _DashboardStub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._srv.shutdown()


async def test_smoke_apex_serves_dashboard_and_service_wins(tmp_path):
    """G4: apex → dashboard (200); service prefix still wins; SSE streams through.

    A real Caddy fronts two upstreams: the apex catch-all reverse-proxies the
    daemon stub, while ``/myapp/*`` reverse-proxies the service stub. Because the
    service route precedes the apex catch-all, ``/myapp/page`` reaches the
    service (path stripped) and ``/`` falls through to the dashboard. An SSE
    request through the apex is forwarded and delivers its events.
    """
    with _DashboardStub() as dash, _RecordingStub() as svc:
        queries = _FlipQueries(
            [
                ActiveServiceRoute(
                    service_name="myapp", host_port=svc.port, status="running", route="/myapp"
                )
            ]
        )
        settings = ProxySettings(
            enabled=True,
            mode="path",
            admin_addr=f"localhost:{_free_port()}",
            https_port=_free_port(),
            scheme="https",
            hostname_override="localhost",
            dashboard_apex=True,
        )
        mgr = ProxyManager(
            queries,
            settings,
            hostname="localhost",
            data_dir=tmp_path,
            dashboard_upstream=f"127.0.0.1:{dash.port}",
        )
        https_port = settings.https_port
        try:
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.reconcile()  # registers the service route AND the apex (last)
            async with httpx.AsyncClient(verify=False) as client:
                root = await _wait_proxy(client, f"https://localhost:{https_port}/")
                assert root.status_code == 200
                assert root.content == b"dashboard"  # apex → daemon stub
                page = await _wait_proxy(client, f"https://localhost:{https_port}/myapp/page")
                assert page.status_code == 200  # service prefix wins over the catch-all
                sse = await _wait_proxy(client, f"https://localhost:{https_port}/sse")
                assert sse.status_code == 200
                assert "tick-0" in sse.text and "tick-2" in sse.text  # SSE forwarded
            # the service upstream saw the stripped path, never "/" or the SSE path
            assert "/page" in svc.paths
            assert "/myapp/page" not in svc.paths
            assert "/" not in svc.paths
            # the dashboard stub served the apex "/" and the "/sse" stream
            assert "/" in dash.paths
            assert "/sse" in dash.paths
        finally:
            await mgr.stop()


async def test_smoke_subdomain_bare_host_matches_no_route(tmp_path):
    """Pins the documented UX change: bare-host requests match NO route.

    In path mode any Host reaches ``/myapp/*``; in subdomain mode only the exact
    ``myapp.<base>`` Host matches — a bare-hostname request handshakes (the bare
    subject stays in the automate list) but falls through every route.
    """
    mgr = _subdomain_manager(tmp_path)
    https_port = mgr._settings.https_port
    with _RecordingStub() as stub:
        try:
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.register("myapp", stub.port)
            async with httpx.AsyncClient(verify=False) as client:
                resp = await _wait_proxy(client, f"https://localhost:{https_port}/myapp/sub")
            # Caddy answers (empty default response) but the upstream is never hit.
            assert resp.content != b"ok"
            assert stub.paths == []
        finally:
            await mgr.stop()


# ---------------------------------------------------------------------------- #
# P26 WP1 — custom domains: internal-only issuance + Host-first ordering.       #
# ---------------------------------------------------------------------------- #


def _peer_cert(sni: str, port: int) -> bytes:
    """The DER leaf a real Caddy serves for *sni* on the loopback listener."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with (
        socket.create_connection(("127.0.0.1", port), timeout=5) as sock,
        ctx.wrap_socket(sock, server_hostname=sni) as tls,
    ):
        der = tls.getpeercert(binary_form=True)
    assert der is not None
    return der


async def _wait_peer_cert(sni: str, port: int, attempts: int = 20) -> bytes:
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return _peer_cert(sni, port)
        except Exception as exc:  # on-demand issuance / listener races
            last = exc
            await asyncio.sleep(0.5)
    raise AssertionError(f"no leaf for {sni}: {last}")


async def _reconcile_recording_admin_traffic(mgr) -> list[tuple[str, str]]:
    """One ``reconcile()`` with every admin request logged as ``(method, path)``.

    The only way to prove "this tick issued no write" against a REAL Caddy.
    Comparing the live route array before and after cannot: a ``PATCH /id/<id>``
    re-emitting an identical object, or the ordering pass deleting and
    re-inserting at the same index, both leave the array byte-identical.
    """
    log: list[tuple[str, str]] = []

    async def _hook(request: httpx.Request) -> None:
        log.append((request.method, request.url.path))

    mgr._admin._client.event_hooks["request"] = [_hook]
    try:
        await mgr.reconcile()
    finally:
        mgr._admin._client.event_hooks["request"] = []
    return log


async def test_smoke_domain_route_is_internal_only_and_wins_in_path_mode(tmp_path):
    """The whole WP1 proxy contract against a REAL Caddy, in path mode.

    ``FakeCaddy`` answers 200 to everything and never evaluates a matcher, so
    none of this is proven by the unit suite: that ``PUT .../routes/0`` really
    inserts, that a *terminal* Host route ordered first beats a path route that
    also matches the request, that Caddy issues the domain's leaf from the
    INTERNAL CA (the trailing catch-all policy — no ACME anywhere), and that
    the daemon's Caddy is not autosaving its config.
    """
    domain = "app.dev.localhost"
    if not _resolves(domain):
        pytest.skip(f"{domain} does not resolve on this host")
    from datetime import UTC, datetime

    from cryptography import x509
    from cryptography.x509.oid import ExtensionOID

    from nerdit.db.rows import ServiceDomain

    with _RecordingStub() as app_a, _RecordingStub() as app_other:
        queries = _FlipQueries(
            [
                ActiveServiceRoute(
                    service_name="a", host_port=app_a.port, status="running", route="/a"
                ),
                ActiveServiceRoute(
                    service_name="other",
                    host_port=app_other.port,
                    status="running",
                    route="/other",
                ),
            ]
        )
        queries.domains = [
            ServiceDomain(domain=domain, service_name="a", created_at=datetime.now(UTC))
        ]
        mgr = _manager(tmp_path, queries=queries)
        https_port = mgr._settings.https_port
        try:
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.reconcile()

            # (i) the Host route is FIRST in the live array.
            resp = await mgr._admin._client.get("/config/apps/http/servers/nerdit/routes")
            ids = [obj.get("@id") for obj in resp.json()]
            assert ids[0] == f"nerdit-route-a@{domain}"
            assert set(ids[1:]) == {"nerdit-route-a", "nerdit-route-other"}
            before = resp.json()

            # (iii) the leaf Caddy serves for the domain is INTERNAL-CA issued
            # and names the domain. This is the F6 proof: the name is in no
            # ``automate`` policy's subjects, only the trailing catch-all.
            leaf = x509.load_der_x509_certificate(await _wait_peer_cert(domain, https_port))
            assert "Caddy Local Authority" in leaf.issuer.rfc4514_string()
            san = leaf.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            assert san.value.get_values_for_type(x509.DNSName) == [domain]

            # (ii) a path another app owns, requested under the custom domain,
            # reaches app "a" with the path UNSTRIPPED (terminal + first).
            root_crt = tmp_path / "caddy" / "pki" / "authorities" / "local" / "root.crt"
            await _wait_file(root_crt)
            ctx = ssl.create_default_context(cafile=str(root_crt))
            async with httpx.AsyncClient(verify=ctx) as client:
                page = await _wait_proxy(client, f"https://{domain}:{https_port}/other/x")
            assert page.status_code == 200
            assert "/other/x" in app_a.paths
            assert app_other.paths == []

            # (iv) the live automation policy list ends with the catch-all, and
            # no policy anywhere names an ACME issuer.
            tls = await mgr._admin.get_tls_config()
            policies = tls["automation"]["policies"]
            assert "subjects" not in policies[-1]
            assert all(p["issuers"] == [{"module": "internal"}] for p in policies)
            assert domain in tls["certificates"]["automate"]
            assert "acme" not in json.dumps(tls).lower()

            # (v) the tick after is write-free. Array equality alone cannot
            # prove that — a PATCH re-emitting an identical object, or the
            # ordering pass deleting and re-inserting at the same index, both
            # leave the array byte-identical. So COUNT the admin traffic: the
            # second reconcile must issue GETs and nothing else, and must not
            # touch ``/config/apps/tls`` at all (the D-P26-4 hash latch,
            # re-proven against a real Caddy rather than the fake).
            admin_log = await _reconcile_recording_admin_traffic(mgr)
            assert [r for r in admin_log if r[0] != "GET"] == [], admin_log
            assert "/config/apps/tls" not in {p for _, p in admin_log}
            after = (await mgr._admin._client.get("/config/apps/http/servers/nerdit/routes")).json()
            assert after == before

            # (vi) S-W7/F5: this Caddy does not autosave its config.
            admin_cfg = (await mgr._admin._client.get("/config/admin")).json()
            assert admin_cfg["config"]["persist"] is False
        finally:
            await mgr.stop()
