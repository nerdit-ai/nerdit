"""Exercise real ACME HTTP-01 with Caddy, Pebble and pebble-challtestsrv.

Skip unless all three binaries exist. Use real DNS resolution and challenge GETs;
PEBBLE_VA_ALWAYS_VALID would bypass the listener, redirects and catch-all under
test. Disable only artificial latency and random nonce rejection.

Keep these distinct: challtestsrv's -dnsserver binds a server; Pebble's selects
its resolver. Pebble listen addresses host its APIs, while httpPort/tlsPort target
Caddy; httpPort must match proxy.acme.http_port. The generated API CA belongs in
ca_root_file; Pebble's independently generated issuance root, fetched from its
management API, belongs in the test client's trust store.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtensionOID, NameOID

from nerdit.config.settings import ProxyAcmeSettings, ProxySettings
from nerdit.core.proxy import ProxyManager
from nerdit.core.proxy.certs import acme_cert_path
from nerdit.db.models import ActiveServiceRoute
from nerdit.db.rows import ServiceDomain

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        any(shutil.which(b) is None for b in ("caddy", "pebble", "pebble-challtestsrv")),
        reason="needs caddy + pebble + pebble-challtestsrv on PATH",
    ),
]

#: The name the whole loop is about. ``.test`` is reserved by RFC 6761 and
#: resolves nowhere real — only Pebble, pointed at the challtest DNS server,
#: ever looks it up.
DOMAIN = "app.example.test"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RecordingStub:
    """A loopback HTTP upstream, so the Host route dials something real."""

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

            def log_message(self, *args: object) -> None:
                pass

        self._srv = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self) -> _RecordingStub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._srv.shutdown()


class _SmokeQueries:
    """The three reads the proxy takes per tick, plus a no-op audit sink."""

    def __init__(self, entries: list[ActiveServiceRoute], domains: list[ServiceDomain]) -> None:
        self.entries = entries
        self.domains = domains

    async def insert_audit_log(self, **_kw: object) -> None:
        return None

    async def set_endpoint_route(self, *_a: object) -> None:
        return None

    async def list_active_service_routes(self) -> list[ActiveServiceRoute]:
        return list(self.entries)

    async def list_service_domains(self) -> list[ServiceDomain]:
        return list(self.domains)

    async def get_service_domains(self, service_name: str) -> list[ServiceDomain]:
        return [d for d in self.domains if d.service_name == service_name]


def _write_pebble_pki(dest: Path) -> tuple[Path, Path, Path]:
    """A throwaway CA + ``localhost`` leaf for Pebble's own API listener.

    Generated per run rather than vendored: nothing about it is a contract, and
    a checked-in key — even a test one — is a credential in the repo. It is
    ONLY the trust anchor for ``https://127.0.0.1:<dir>/dir``; the certificates
    Pebble *issues* chain to a different, per-run root (see the module docstring).
    """
    dest.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "nerdit smoke pebble CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        # OpenSSL (so Python, curl and Go) refuses a chain whose leaf carries no
        # Authority Key Identifier, which needs this on the issuer first.
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    ca_pem = dest / "ca.pem"
    ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_pem = dest / "cert.pem"
    cert_pem.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_pem = dest / "key.pem"
    key_pem.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return ca_pem, cert_pem, key_pem


class _Pebble:
    """The two Pebble processes, their ports, and a hard kill-by-pid teardown."""

    def __init__(self, root: Path, *, http_port: int) -> None:
        self.root = root
        self.http_port = http_port
        self.dns_port = _free_port()
        self.challtest_mgmt_port = _free_port()
        self.dir_port = _free_port()
        self.mgmt_port = _free_port()
        self.ca_pem, cert, key = _write_pebble_pki(root)
        self.config = root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "pebble": {
                        "listenAddress": f"127.0.0.1:{self.dir_port}",
                        "managementListenAddress": f"127.0.0.1:{self.mgmt_port}",
                        "certificate": str(cert),
                        "privateKey": str(key),
                        # The VA connects to THIS port on the target — it must
                        # be the daemon's [proxy.acme].http_port.
                        "httpPort": http_port,
                        "tlsPort": _free_port(),
                        "ocspResponderURL": "",
                        "externalAccountBindingRequired": False,
                    }
                }
            )
        )
        self.procs: list[subprocess.Popen[bytes]] = []

    @property
    def directory(self) -> str:
        return f"https://127.0.0.1:{self.dir_port}/dir"

    def _api_trust(self) -> ssl.SSLContext:
        """Trust the generated CA for Pebble's own API — never ``verify=False``.

        Every call this harness makes to the directory or the management API
        verifies, so a broken chain fails here instead of being tunnelled past
        with ``-k``.
        """
        return ssl.create_default_context(cafile=str(self.ca_pem))

    def _spawn(self, argv: list[str], name: str, env: dict[str, str] | None = None) -> None:
        log = (self.root / f"{name}.log").open("wb")
        self.procs.append(
            subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
            )
        )

    async def start(self) -> None:
        # DNS only: every challenge solver is disabled because Caddy's own
        # solver is the thing under test. ``-dnsserver`` here is a BIND address.
        self._spawn(
            [
                "pebble-challtestsrv",
                "-http01",
                "",
                "-https01",
                "",
                "-tlsalpn01",
                "",
                "-dnsserver",
                f"127.0.0.1:{self.dns_port}",
                "-management",
                f"127.0.0.1:{self.challtest_mgmt_port}",
            ],
            "challtestsrv",
        )
        # ``-dnsserver`` here is a CLIENT override: resolve through the box above.
        self._spawn(
            [
                "pebble",
                "-config",
                str(self.config),
                "-dnsserver",
                f"127.0.0.1:{self.dns_port}",
            ],
            "pebble",
            env={"PEBBLE_VA_NOSLEEP": "1", "PEBBLE_WFE_NONCEREJECT": "0"},
        )
        async with httpx.AsyncClient(verify=self._api_trust()) as client:
            for _ in range(60):
                try:
                    resp = await client.get(self.directory, timeout=2.0)
                    if resp.status_code == 200:
                        return
                except Exception:  # noqa: BLE001 — still booting
                    pass
                await asyncio.sleep(0.25)
        raise AssertionError(f"pebble never served {self.directory}")

    async def chain_pem(self, dest: Path) -> Path:
        """Pebble's per-run root + intermediate, for verifying the ISSUED leaf."""
        async with httpx.AsyncClient(verify=self._api_trust()) as client:
            root = await client.get(f"https://127.0.0.1:{self.mgmt_port}/roots/0", timeout=5.0)
            inter = await client.get(
                f"https://127.0.0.1:{self.mgmt_port}/intermediates/0", timeout=5.0
            )
        assert root.status_code == 200 and inter.status_code == 200
        dest.write_text(root.text.strip() + "\n" + inter.text.strip() + "\n")
        return dest

    def stop(self) -> None:
        """Kill exactly the pids we spawned — never a pattern-matched sweep."""
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def _acme_manager(
    tmp_path: Path,
    *,
    http_port: int,
    directory: str,
    ca_root_file: Path | None,
    queries: object,
    http_redirect: bool = True,
) -> ProxyManager:
    settings = ProxySettings(
        enabled=True,
        mode="path",
        admin_addr=f"localhost:{_free_port()}",
        https_port=_free_port(),
        hostname_override="localhost",
        acme=ProxyAcmeSettings(
            enabled=True,
            email="smoke@example.test",
            directory=directory,
            http_port=http_port,
            http_redirect=http_redirect,
            ca_root_file=str(ca_root_file) if ca_root_file is not None else None,
        ),
    )
    return ProxyManager(queries, settings, hostname="localhost", data_dir=tmp_path)


async def _wait_file(path: Path, attempts: int = 240) -> Path:
    for _ in range(attempts):
        if path.exists():
            return path
        await asyncio.sleep(0.25)
    raise AssertionError(f"{path} never appeared")


def _peer_leaf(domain: str, port: int, cafile: Path) -> x509.Certificate:
    """Handshake to ``127.0.0.1:port`` as *domain*, VERIFYING against *cafile*.

    Verification is the assertion: a leaf that does not chain to Pebble's roots,
    or whose SAN is not the domain, raises here rather than being inspected and
    waved through.
    """
    ctx = ssl.create_default_context(cafile=str(cafile))
    with (
        socket.create_connection(("127.0.0.1", port), timeout=10) as sock,
        ctx.wrap_socket(sock, server_hostname=domain) as tls,
    ):
        der = tls.getpeercert(binary_form=True)
    assert der is not None
    return x509.load_der_x509_certificate(der)


async def _reconcile_recording_admin_traffic(mgr: ProxyManager) -> list[tuple[str, str]]:
    log: list[tuple[str, str]] = []

    async def _hook(request: httpx.Request) -> None:
        log.append((request.method, request.url.path))

    mgr._admin._client.event_hooks["request"] = [_hook]
    try:
        await mgr.reconcile()
    finally:
        mgr._admin._client.event_hooks["request"] = []
    return log


async def test_acme_http01_issues_a_public_leaf_end_to_end(tmp_path):
    """The whole WP2 loop against a real CA: DNS → solver → leaf → served."""
    http_port = _free_port()
    pebble = _Pebble(tmp_path / "pebble", http_port=http_port)
    mgr: ProxyManager | None = None
    try:
        await pebble.start()
        with _RecordingStub() as app:
            queries = _SmokeQueries(
                [
                    ActiveServiceRoute(
                        service_name="a", host_port=app.port, status="running", route="/a"
                    )
                ],
                [
                    ServiceDomain(
                        domain=DOMAIN,
                        service_name="a",
                        acme=True,
                        created_at=datetime.now(UTC),
                    )
                ],
            )
            mgr = _acme_manager(
                tmp_path,
                http_port=http_port,
                directory=pebble.directory,
                ca_root_file=pebble.ca_pem,
                queries=queries,
            )
            https_port = mgr._settings.https_port

            # (i) the config Caddy is SPAWNED with carries both servers and the
            # HTTP role — the listener cannot be added to a running Caddy.
            boot = mgr._bootstrap_config()["apps"]["http"]
            assert boot["http_port"] == http_port
            assert set(boot["servers"]) == {"nerdit", "nerdit-acme-http"}

            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.reconcile()

            # (ii) the LIVE policy list, read back from Caddy: public first,
            # node subjects next, subject-less catch-all still last.
            tls = await mgr._admin.get_tls_config()
            policies = tls["automation"]["policies"]
            assert policies[0]["subjects"] == [DOMAIN]
            assert policies[0]["issuers"][0]["module"] == "acme"
            assert policies[0]["issuers"][0]["ca"] == pebble.directory
            # EXACTLY one issuer (review round 2). An internal fallback behind
            # it was built and measured here: the route upserts that follow the
            # TLS push reload Caddy, the in-flight order dies with "stopping
            # apps", certmagic falls through to the internal CA in milliseconds
            # and THIS TEST never saw a public leaf again. The single issuer is
            # the shape; see test_a_pending_acme_name_serves_no_leaf_at_all.
            assert len(policies[0]["issuers"]) == 1
            assert policies[1]["subjects"] == ["localhost"]
            assert "subjects" not in policies[-1]
            assert policies[-1]["issuers"] == [{"module": "internal"}]

            # (iii) a real issuance lands at the DERIVED storage path — this is
            # what pins ``acme_storage_key`` against a certmagic change.
            cert_path = acme_cert_path(mgr._storage_root, pebble.directory, DOMAIN)
            await _wait_file(cert_path)
            row = queries.domains[0]
            assert mgr.cert_status(row).state == "issued"
            assert mgr.cert_status(row).not_after is not None

            # (iv) and it is the leaf Caddy actually SERVES: verified against
            # Pebble's own per-run roots, so this cannot pass on an internal
            # leaf that merely happens to name the domain.
            chain = await pebble.chain_pem(tmp_path / "pebble" / "chain.pem")
            leaf = _peer_leaf(DOMAIN, https_port, chain)
            san = leaf.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            assert san.value.get_values_for_type(x509.DNSName) == [DOMAIN]
            assert "Caddy Local Authority" not in leaf.issuer.rfc4514_string()

            # (v) the :80 role: a Host-less 308 naming the advertised port.
            async with httpx.AsyncClient(follow_redirects=False) as client:
                resp = await client.get(
                    f"http://127.0.0.1:{http_port}/x",
                    headers={"Host": DOMAIN},
                    timeout=5.0,
                )
            assert resp.status_code == 308
            assert resp.headers["Location"] == f"https://{DOMAIN}:{https_port}/x"

            # (vi) steady state is still zero writes with an ACME domain bound.
            admin_log = await _reconcile_recording_admin_traffic(mgr)
            assert [r for r in admin_log if r[0] != "GET"] == [], admin_log
            assert "/config/apps/tls" not in {p for _, p in admin_log}

            # (vii) the proof the CA's GET hit CADDY's solver on OUR listener,
            # rather than the order being waved through.
            assert "served key authentication" in (tmp_path / "caddy.log").read_text()
    finally:
        if mgr is not None:
            await mgr.stop()
        pebble.stop()


def _try_handshake(domain: str, port: int) -> x509.Certificate:
    """Handshake as *domain* WITHOUT verifying, returning the leaf Caddy served.

    Deliberately unverified: the question is which CA signed what came back (or
    whether anything came back at all), not whether this test's trust store
    holds the node's internal root.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with (
        socket.create_connection(("127.0.0.1", port), timeout=10) as sock,
        ctx.wrap_socket(sock, server_hostname=domain) as tls,
    ):
        der = tls.getpeercert(binary_form=True)
    assert der is not None
    return x509.load_der_x509_certificate(der)


async def test_a_pending_acme_name_serves_no_leaf_at_all(tmp_path):
    """Pin failed TLS handshakes while an ACME-only name has no certificate.

    Caddy uses the first matching policy, so its internal catch-all cannot serve
    the name. Changing an internal-CA name to ACME also evicts its cached leaf.
    Do not add an internal fallback: route reloads cancel ACME orders, certmagic
    falls through to the fast internal issuer, and public issuance stalls until
    renewal. Keep a loopback directory port closed to hold the name pending.
    """
    http_port = _free_port()
    dead_ca = f"https://127.0.0.1:{_free_port()}/dir"
    mgr: ProxyManager | None = None
    try:
        with _RecordingStub() as app:
            queries = _SmokeQueries(
                [
                    ActiveServiceRoute(
                        service_name="a", host_port=app.port, status="running", route="/a"
                    )
                ],
                [
                    ServiceDomain(
                        domain=DOMAIN,
                        service_name="a",
                        acme=True,
                        created_at=datetime.now(UTC),
                    )
                ],
            )
            mgr = _acme_manager(
                tmp_path,
                http_port=http_port,
                directory=dead_ca,
                ca_root_file=None,
                queries=queries,
            )
            await mgr.start()
            assert mgr.available, "Caddy did not come up"
            await mgr.reconcile()

            # (i) the pending name does not complete a handshake at all.
            with pytest.raises(ssl.SSLError) as caught:
                _try_handshake(DOMAIN, mgr._settings.https_port)
            assert "INTERNAL_ERROR" in str(caught.value).upper()

            # (ii) while the node's OWN name, on the internal policy in the very
            # same config, serves normally — so this is the acme policy's doing,
            # not a broken listener.
            node_leaf = _try_handshake("localhost", mgr._settings.https_port)
            assert "Caddy Local Authority" in node_leaf.issuer.rfc4514_string()

            # (iii) and ``cert_state`` reports it honestly: nothing at the ACME
            # storage key, so S-W2-6 reads ``pending`` (never ``issued``).
            assert not acme_cert_path(mgr._storage_root, dead_ca, DOMAIN).exists()
            assert mgr.cert_status(queries.domains[0]).state == "pending"
    finally:
        if mgr is not None:
            await mgr.stop()


async def test_the_acme_listener_answers_404_when_the_redirect_is_off(tmp_path):
    """``http_redirect=false`` leaves the solver working and nothing else."""
    http_port = _free_port()
    pebble = _Pebble(tmp_path / "pebble", http_port=http_port)
    mgr: ProxyManager | None = None
    try:
        # No CA needs to run for this leg, but the directory must be a real
        # https URL for the settings to validate — reuse the same generated PKI
        # rather than inventing a second shape.
        queries = _SmokeQueries([], [])
        mgr = _acme_manager(
            tmp_path,
            http_port=http_port,
            directory=pebble.directory,
            ca_root_file=pebble.ca_pem,
            queries=queries,
            http_redirect=False,
        )
        await mgr.start()
        assert mgr.available, "Caddy did not come up"

        async with httpx.AsyncClient(follow_redirects=False) as client:
            resp = await client.get(
                f"http://127.0.0.1:{http_port}/x", headers={"Host": DOMAIN}, timeout=5.0
            )
        # 404, not the bare 200 an empty Caddy route table would answer.
        assert resp.status_code == 404
    finally:
        if mgr is not None:
            await mgr.stop()
        pebble.stop()
