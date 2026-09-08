"""Tests for the P3 URL layer / ProxyManager (S3–S5).

Pure-async over an in-memory DB and an ``httpx.MockTransport`` standing in for
the Caddy admin API — no real Caddy, no TestClient (so no Starlette/aiosqlite
cross-loop hazard). They cover the mode-aware seams, the ``CaddyAdmin`` HTTP
shapes (incl. idempotent delete semantics), reconcile drift/prune/idempotency,
self-heal adoption, graceful degradation, and audit de-dup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import httpx
import pytest

from nerdit.config.settings import ProxyAcmeSettings, ProxySettings
from nerdit.core.proxy import (
    _AUTH_UNPARSABLE,
    CaddyAdmin,
    EdgeAuthInvalid,
    EdgeAuthSpec,
    LiveRoute,
    ProxyManager,
    ProxyState,
    _extract_auth_fingerprint,
    _extract_dial,
    _route_shape,
    auth_fingerprint,
    edgeauth,
    generate_route,
    load_edge_auth,
    ordering_violations,
    partition_domains,
    public_url_for,
)
from nerdit.core.proxy.certs import CertStatus, acme_cert_path
from nerdit.db.models import Job, JobKind, JobStatus

pytestmark = pytest.mark.asyncio


# --- 1. seams -----------------------------------------------------------------


async def test_generate_route_path_and_subdomain():
    assert generate_route("myapp") == "/myapp"
    assert generate_route("myapp", mode="path") == "/myapp"
    assert generate_route("myapp", mode="subdomain") == ""


async def test_public_url_for_path_mode():
    # default 443 → no port suffix
    url = public_url_for(
        "myapp",
        "/myapp",
        mode="path",
        hostname="box",
        base_domain=None,
        scheme="https",
        https_port=443,
    )
    assert url == "https://box/myapp/"
    # non-default port → suffix present
    url2 = public_url_for(
        "myapp",
        "/myapp",
        mode="path",
        hostname="box",
        base_domain=None,
        scheme="https",
        https_port=8443,
    )
    assert url2 == "https://box:8443/myapp/"


async def test_public_url_for_subdomain_mode():
    url = public_url_for(
        "myapp",
        "",
        mode="subdomain",
        hostname="box",
        base_domain="lan.local",
        scheme="https",
        https_port=443,
    )
    assert url == "https://myapp.lan.local/"
    # base_domain falls back to hostname
    url2 = public_url_for(
        "myapp",
        "",
        mode="subdomain",
        hostname="box",
        base_domain=None,
        scheme="https",
        https_port=443,
    )
    assert url2 == "https://myapp.box/"


async def test_public_url_for_none_route_is_none():
    assert (
        public_url_for(
            "myapp",
            None,
            mode="path",
            hostname="box",
            base_domain=None,
            scheme="https",
            https_port=443,
        )
        is None
    )


async def test_public_url_for_subdomain_mode_non_default_port():
    # Only path mode exercised the non-default port suffix before P3.5.
    url = public_url_for(
        "myapp",
        "",
        mode="subdomain",
        hostname="box",
        base_domain="lan.local",
        scheme="https",
        https_port=8443,
    )
    assert url == "https://myapp.lan.local:8443/"


async def test_public_url_for_public_port_override():
    # ``public_port`` overrides the ADVERTISED port only: an external proxy
    # fronts the embedded Caddy (public 443 → loopback https_port), so the URL
    # must show the reachable port, not the bound one.
    url = public_url_for(
        "myapp",
        "/myapp",
        mode="path",
        hostname="box",
        base_domain=None,
        scheme="https",
        https_port=9443,
        public_port=443,
    )
    assert url == "https://box/myapp/"
    # Non-default override keeps its suffix (and wins over https_port).
    url2 = public_url_for(
        "myapp",
        "",
        mode="subdomain",
        hostname="box",
        base_domain="lan.local",
        scheme="https",
        https_port=9443,
        public_port=8443,
    )
    assert url2 == "https://myapp.lan.local:8443/"
    # ``None`` (the default) preserves the historical https_port behaviour.
    url3 = public_url_for(
        "myapp",
        "/myapp",
        mode="path",
        hostname="box",
        base_domain=None,
        scheme="https",
        https_port=9443,
        public_port=None,
    )
    assert url3 == "https://box:9443/myapp/"


async def test_public_url_for_empty_string_route_is_not_none():
    # D7 contract pin: route="" (the subdomain projection) is falsy but a real,
    # routed value — consumers must gate on ``route is not None``, never
    # truthiness. Distinguish it explicitly from the ``route=None`` case above.
    url = public_url_for(
        "myapp",
        "",
        mode="subdomain",
        hostname="box",
        base_domain="lan.local",
        scheme="https",
        https_port=443,
    )
    assert url is not None
    assert url == "https://myapp.lan.local/"


# --- 2. route object shape ----------------------------------------------------


def _manager(mode="path", base_domain=None, hostname="box", *, acme=None, **settings_kw):
    settings = ProxySettings(
        enabled=True,
        mode=mode,
        base_domain=base_domain,
        **({"acme": acme} if acme is not None else {}),
        **settings_kw,
    )
    # queries=None: route-shaping methods never touch the DB.
    return ProxyManager(None, settings, hostname=hostname, data_dir="/tmp/nerdit-test")


async def test_caddy_route_obj_path_mode():
    mgr = _manager()
    spec = mgr.build_route("myapp", 9401)
    obj = mgr._caddy_route_obj(spec)
    assert obj["@id"] == "nerdit-route-myapp"
    assert obj["match"] == [{"path": ["/myapp", "/myapp/*"]}]
    inner = obj["handle"][0]["routes"][0]["handle"]
    assert inner[0] == {"handler": "rewrite", "strip_path_prefix": "/myapp"}
    assert inner[1]["handler"] == "reverse_proxy"
    assert _extract_dial(obj) == "127.0.0.1:9401"


async def test_caddy_route_obj_subdomain_mode():
    mgr = _manager(mode="subdomain", base_domain="lan.local")
    spec = mgr.build_route("myapp", 9401)
    obj = mgr._caddy_route_obj(spec)
    assert obj["match"] == [{"host": ["myapp.lan.local"]}]
    # no strip_path_prefix in subdomain mode
    assert all(h.get("handler") != "rewrite" for h in obj["handle"])
    assert _extract_dial(obj) == "127.0.0.1:9401"


async def test_caddy_route_obj_subdomain_mode_base_domain_none_falls_back_to_hostname():
    # Contract pin (currently untested before P3.5): the Host matcher itself
    # falls back to the hostname, mirroring public_url_for's fallback.
    mgr = _manager(mode="subdomain", base_domain=None, hostname="myhost")
    spec = mgr.build_route("myapp", 9401)
    obj = mgr._caddy_route_obj(spec)
    assert obj["match"] == [{"host": ["myapp.myhost"]}]


# --- 2b. TLS bootstrap subjects (P3.5 S2 — the fourth, TLS-only mode-aware site)


async def test_tls_subjects_path_mode_single_subject():
    # Regression pin: path mode still automates exactly one subject. Since P26
    # WP1 the policy list carries a SECOND, subject-less catch-all entry (F4)
    # so no Host-derived name can fall through to public issuance (F6).
    mgr = _manager()
    assert mgr._tls_subjects() == ["box"]
    tls = mgr._bootstrap_config()["apps"]["tls"]
    assert tls["certificates"]["automate"] == ["box"]
    assert tls["automation"]["policies"] == [
        {"subjects": ["box"], "issuers": [{"module": "internal"}]},
        {"issuers": [{"module": "internal"}]},
    ]


async def test_tls_subjects_subdomain_adds_wildcard():
    mgr = _manager(mode="subdomain", base_domain="lan.local")
    assert mgr._tls_subjects() == ["box", "*.lan.local"]
    tls = mgr._bootstrap_config()["apps"]["tls"]
    assert tls["certificates"]["automate"] == ["box", "*.lan.local"]
    assert tls["automation"]["policies"][0]["subjects"] == ["box", "*.lan.local"]


async def test_tls_subjects_subdomain_base_domain_falls_back_to_hostname():
    mgr = _manager(mode="subdomain", base_domain=None, hostname="box")
    assert mgr._tls_subjects() == ["box", "*.box"]


async def test_tls_subjects_deduped_when_base_domain_equals_hostname():
    mgr = _manager(mode="subdomain", base_domain="box", hostname="box")
    subjects = mgr._tls_subjects()
    assert subjects == ["box", "*.box"]
    assert len(subjects) == len(set(subjects))


# --- 2c. SAN registry — [proxy].extra_hostnames (P25 WP5 / D-P25-9) -----------


def _san_manager(extras, *, mode="path", base_domain=None, hostname="box", override=None):
    settings = ProxySettings(
        enabled=True,
        mode=mode,
        base_domain=base_domain,
        hostname_override=override,
        extra_hostnames=extras,
    )
    return ProxyManager(None, settings, hostname=hostname, data_dir="/tmp/nerdit-test")


async def test_tls_subjects_path_mode_zero_extras_is_unchanged():
    # Non-regression: the empty registry must leave the P3 single-subject
    # bootstrap byte-identical.
    assert _san_manager([])._tls_subjects() == ["box"]


async def test_tls_subjects_path_mode_one_extra():
    mgr = _san_manager(["192.168.1.50"])
    assert mgr._tls_subjects() == ["box", "192.168.1.50"]
    tls = mgr._bootstrap_config()["apps"]["tls"]
    # Both TLS subtrees come from the one _tls_subjects() call.
    assert tls["certificates"]["automate"] == ["box", "192.168.1.50"]
    assert tls["automation"]["policies"][0]["subjects"] == ["box", "192.168.1.50"]


async def test_tls_subjects_path_mode_three_extras_keep_primary_first():
    mgr = _san_manager(["192.168.1.50", "nerd-box.local", "nerdit.lan"])
    assert mgr._tls_subjects() == ["box", "192.168.1.50", "nerd-box.local", "nerdit.lan"]


async def test_tls_subjects_subdomain_mode_extras_precede_the_wildcard():
    # The derived wildcard stays LAST (it is appended after the registry), and
    # extras keep their configured order.
    mgr = _san_manager(["192.168.1.50", "nerdit.lan"], mode="subdomain", base_domain="lan.local")
    assert mgr._tls_subjects() == ["box", "192.168.1.50", "nerdit.lan", "*.lan.local"]
    tls = mgr._bootstrap_config()["apps"]["tls"]
    assert tls["certificates"]["automate"] == [
        "box",
        "192.168.1.50",
        "nerdit.lan",
        "*.lan.local",
    ]


async def test_tls_subjects_hostname_override_duplicating_an_extra_is_deduped():
    # The order-preserving dict.fromkeys dedupe must survive the splice: the
    # primary wins its slot and the duplicate extra vanishes (not doubled).
    mgr = _san_manager(["nerdit.lan", "192.168.1.50"], override="nerdit.lan")
    subjects = mgr._tls_subjects()
    assert subjects == ["nerdit.lan", "192.168.1.50"]
    assert len(subjects) == len(set(subjects))


async def test_tls_subjects_subdomain_dedupe_survives_extras():
    # An extra that repeats the bare subject in subdomain mode collapses, and
    # the wildcard is still appended exactly once.
    mgr = _san_manager(["box", "nerdit.lan"], mode="subdomain", base_domain="box")
    subjects = mgr._tls_subjects()
    assert subjects == ["box", "nerdit.lan", "*.box"]
    assert len(subjects) == len(set(subjects))


#: The internal-CA issuer block, the ONLY issuer WP1 ever emits (P26 F4/F6).
_INTERNAL = [{"module": "internal"}]


#: The ``[proxy.acme]`` block the WP2 tests configure. A loopback ``http``
#: directory is the validator's test carve-out, so no network name appears here.
_ACME_DIRECTORY = "https://127.0.0.1:14000/dir"
_ACME_EMAIL = "ops@example.test"
_ACME_HTTP_PORT = 18080


def _acme_settings(**kw) -> ProxyAcmeSettings:
    """An enabled ``[proxy.acme]`` block; ``kw`` overrides one field at a time."""
    return ProxyAcmeSettings(
        **{
            "enabled": True,
            "email": _ACME_EMAIL,
            "directory": _ACME_DIRECTORY,
            "http_port": _ACME_HTTP_PORT,
            **kw,
        }
    )


def _acme_issuer(*, ca_root_file: str | None = None, http_port: int = _ACME_HTTP_PORT) -> dict:
    """The ``acme`` issuer block, built by hand (P26 S-W2-3).

    Hand-built for the same reason ``_tls_app`` is: keys Caddy 2.11.4 actually
    accepts (``ca``, not ``directory``; ``trusted_roots_pem_files``, a list of
    PATHS, not inline PEM) are the assertion, not whatever the manager emits.
    """
    issuer: dict = {
        "module": "acme",
        "ca": _ACME_DIRECTORY,
        "email": _ACME_EMAIL,
        "challenges": {
            "tls-alpn": {"disabled": True},
            "http": {"alternate_port": http_port},
        },
    }
    if ca_root_file is not None:
        issuer["trusted_roots_pem_files"] = [ca_root_file]
    return issuer


def _tls_app(
    subjects: list[str],
    domains: list[str] | None = None,
    acme: list[str] | None = None,
    issuer: dict | None = None,
) -> dict:
    """The ``apps.tls`` subtree the proxy converges to (P26 S-W6 + S-W2-3), by hand.

    Deliberately NOT ``mgr._tls_desired_app(...)``: a golden the tests compute
    themselves is what makes the catch-all policy's presence and position an
    assertion rather than a tautology.

    *acme* (WP2) is the subset issued publicly. It leads the policy list — first
    match wins in Caddy — while the node-subjects policy and the subject-less
    catch-all keep their WP1 positions, so a name that is not in *acme* still
    resolves to the internal CA. That policy carries exactly ONE issuer: an
    internal fallback behind it was measured and rejected (review round 2) — see
    ``test_a_pending_acme_name_serves_no_leaf_at_all``.
    """
    policies: list[dict] = []
    if acme:
        policies.append({"subjects": sorted(acme), "issuers": [issuer or _acme_issuer()]})
    policies.append({"subjects": subjects, "issuers": _INTERNAL})
    policies.append({"issuers": _INTERNAL})  # the catch-all, LAST and subject-less
    return {
        "certificates": {"automate": [*subjects, *sorted(domains or [])]},
        "automation": {"policies": policies},
    }


# --- 3. CaddyAdmin over MockTransport -----------------------------------------


class FakeCaddy:
    """An in-memory Caddy admin API for httpx.MockTransport.

    ``has_nerdit``: whether the ``nerdit`` server block exists (a *foreign* Caddy
    has none). ``routes_status``: override the status for ``GET .../routes`` to
    simulate a transient read failure. ``tls``/``tls_status``: the in-memory
    ``tls`` app subtree (defaults to the P3 single-subject bootstrap for
    hostname ``box``) and its ``GET`` status override. ``fail_upserts``: fail
    the next N route upserts (``POST /id/...`` and route appends) with a
    transient 503 — the whole route objects are stored, so shape-aware tests
    can assert the live matcher, not just the dial.
    """

    def __init__(
        self,
        *,
        has_nerdit: bool = True,
        has_acme_server: bool = False,
        routes_status: int = 200,
        tls: dict | None = None,
        tls_status: int = 200,
        fail_routes_get_after: int | None = None,
    ):
        self.routes: list[dict] = []  # the nerdit server's route list (full objects)
        self.requests: list[tuple[str, str]] = []
        self.has_nerdit = has_nerdit
        # (P26 WP2) Whether the ``nerdit-acme-http`` server block exists. A Caddy
        # spawned before ``[proxy.acme]`` was turned on has the ``nerdit`` server
        # (so adoption succeeds) and NOT this one.
        self.has_acme_server = has_acme_server
        # (review round 2) Set post-construction: answer the FIRST
        # ``fail_acme_probe_times`` probes inconclusively — with
        # ``acme_probe_status`` (a non-200 that is not 404), or ``-1`` for a
        # transport error. Everything after answers factually.
        self.fail_acme_probe_times = 0
        self.acme_probe_status = 503
        self.acme_probe_calls = 0
        self.routes_status = routes_status
        self.fail_upserts = 0
        # Once this many GET .../routes calls have succeeded, fail the rest with
        # a 503. Lets a test fail ONLY the apex_state read (a later routes GET)
        # while the earlier live_routes read succeeds — both hit the same URL.
        self.fail_routes_get_after = fail_routes_get_after
        self.routes_get_calls = 0
        # (P26) Permanently 503 every PATCH/PUT/POST carrying one of these
        # ``@id``s — one route of a set can then fail while its siblings land.
        self.fail_ids: set[str] = set()
        # The default is the CONVERGED WP1 subtree for hostname ``box`` (P26
        # S-W6: automate + a subjects policy + the trailing catch-all), so a
        # path-mode adoption is a no-op exactly as it was pre-WP1.
        self.tls: dict = tls if tls is not None else _tls_app(["box"])
        self.tls_status = tls_status

    def _acme_probe(self, request: httpx.Request) -> httpx.Response:
        """Answer ``GET .../servers/nerdit-acme-http`` (review round 2)."""
        self.acme_probe_calls += 1
        if self.acme_probe_calls <= self.fail_acme_probe_times:
            if self.acme_probe_status < 0:
                raise httpx.ConnectError("boom", request=request)
            return httpx.Response(self.acme_probe_status)
        if not self.has_acme_server:
            # Caddy-factual: an absent config key answers 200 with a JSON ``null``
            # body, measured on v2.11.4 — NOT a 404. Spelled as raw content
            # because httpx treats ``json=None`` as "no body at all".
            return httpx.Response(
                200, content=b"null", headers={"content-type": "application/json"}
            )
        return httpx.Response(200, json={"listen": [":80"], "routes": []})

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))
        if path == "/config/" and method == "GET":
            return httpx.Response(200, json={})
        if path == "/config/apps/http/servers/nerdit" and method == "GET":
            if not self.has_nerdit:
                return httpx.Response(404)
            return httpx.Response(200, json={"listen": [":443"], "routes": self.routes})
        if path == "/config/apps/http/servers/nerdit-acme-http" and method == "GET":
            return self._acme_probe(request)
        if path.startswith("/config/apps/http/servers/nerdit/routes/") and method == "PUT":
            # (F1) PUT at an index INSERTS before the element there — also on an
            # empty array, where index 0 is the only legal position.
            idx = int(path.rsplit("/", 1)[1])
            obj = json.loads(request.content)
            if self.fail_ids and obj.get("@id") in self.fail_ids:
                return httpx.Response(503, text="transient failure")
            if idx > len(self.routes):
                return httpx.Response(500, json={"error": "index out of range"})
            self.routes.insert(idx, obj)
            return httpx.Response(200)
        if path == "/config/apps/http/servers/nerdit/routes":
            if method == "GET":
                if self.routes_status != 200:
                    return httpx.Response(self.routes_status)
                self.routes_get_calls += 1
                if (
                    self.fail_routes_get_after is not None
                    and self.routes_get_calls > self.fail_routes_get_after
                ):
                    return httpx.Response(503)
                return httpx.Response(200, json=self.routes)
            if method == "POST":
                obj = json.loads(request.content)
                if self.fail_upserts > 0 or (self.fail_ids and obj.get("@id") in self.fail_ids):
                    self.fail_upserts = max(0, self.fail_upserts - 1)
                    return httpx.Response(503, text="transient failure")
                self.routes.append(obj)
                return httpx.Response(200)
        if path == "/config/apps/tls":
            if method == "GET":
                if self.tls_status != 200:
                    return httpx.Response(self.tls_status)
                return httpx.Response(200, json=self.tls)
            if method == "POST":
                self.tls = json.loads(request.content)
                return httpx.Response(200)
        if path == "/load" and method == "POST":
            return httpx.Response(200)
        if path.startswith("/id/"):
            # Mirror REAL Caddy 2.6.2 semantics (observed in the P3.5 manual
            # runbook): the id map resolves to the NEWEST (last) occurrence;
            # PATCH replaces it; POST on an existing id APPENDS a sibling (the
            # duplication bug upsert_route must avoid); absent id → 404.
            rid = path[len("/id/") :]
            matches = [i for i, r in enumerate(self.routes) if r.get("@id") == rid]
            if method in ("PATCH", "POST"):
                if self.fail_upserts > 0 or rid in self.fail_ids:
                    self.fail_upserts = max(0, self.fail_upserts - 1)
                    return httpx.Response(503, text="transient failure")
                if not matches:
                    return httpx.Response(404, json={"error": f"unknown object ID '{rid}'"})
                if method == "PATCH":
                    self.routes[matches[-1]] = json.loads(request.content)
                else:
                    self.routes.append(json.loads(request.content))
                return httpx.Response(200)
            if method == "DELETE":
                if not matches:
                    return httpx.Response(500, json={"error": f"unknown object ID '{rid}'"})
                del self.routes[matches[-1]]
                return httpx.Response(200)
        if path.startswith("/config/apps/http/servers/nerdit/routes/") and method == "DELETE":
            idx = int(path.rsplit("/", 1)[1])
            if 0 <= idx < len(self.routes):
                del self.routes[idx]
                return httpx.Response(200)
            return httpx.Response(500, json={"error": "index out of range"})
        return httpx.Response(404)


def _admin(fake: FakeCaddy) -> CaddyAdmin:
    return CaddyAdmin("localhost:2019", transport=httpx.MockTransport(fake.handler))


async def test_caddy_admin_upsert_appends_then_replaces():
    fake = FakeCaddy()
    admin = _admin(fake)
    obj = {"@id": "nerdit-route-a", "match": [], "handle": []}
    await admin.upsert_route(obj)  # absent → append
    assert len(fake.routes) == 1
    obj2 = {"@id": "nerdit-route-a", "match": [{"x": 1}], "handle": []}
    await admin.upsert_route(obj2)  # present → PATCH /id replaces, no duplicate
    assert len(fake.routes) == 1
    assert fake.routes[0]["match"] == [{"x": 1}]
    # POST /id must never be used on an existing id: on a real Caddy 2.6.2 it
    # APPENDS a sibling with the same @id instead of replacing (P3.5 runbook).
    assert not any(m == "POST" and p.startswith("/id/") for m, p in fake.requests)
    await admin.aclose()


async def test_caddy_admin_live_routes_counts_duplicates_last_wins():
    """Duplicate @ids are counted; dial/shape reflect the newest (id-map target)."""
    fake = FakeCaddy()
    fake.routes = [
        {
            "@id": "nerdit-route-a",
            "match": [{"path": ["/a", "/a/*"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:1111"}]}],
        },
        {
            "@id": "nerdit-route-a",
            "match": [{"host": ["a.lan.local"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:2222"}]}],
        },
    ]
    admin = _admin(fake)
    live = await admin.live_routes()
    # (P26 S-W4) ``index`` is the LAST occurrence's array position, for the same
    # reason dial/shape are: that is the object the id map resolves to.
    assert live == {
        "nerdit-route-a": LiveRoute(dial="127.0.0.1:2222", shape="host", count=2, index=1)
    }
    await admin.aclose()


async def test_caddy_admin_dedupe_route_deletes_stale_twins_by_index():
    """dedupe_route removes all but the newest copy, leaving other ids alone."""
    fake = FakeCaddy()
    fake.routes = [
        {"@id": "nerdit-route-a", "match": [{"path": ["/a", "/a/*"]}], "handle": []},
        {"@id": "nerdit-route-b", "match": [{"path": ["/b", "/b/*"]}], "handle": []},
        {"@id": "nerdit-route-a", "match": [{"host": ["a.lan.local"]}], "handle": []},
    ]
    admin = _admin(fake)
    removed = await admin.dedupe_route("nerdit-route-a")
    assert removed == 1
    assert [r["@id"] for r in fake.routes] == ["nerdit-route-b", "nerdit-route-a"]
    assert fake.routes[1]["match"] == [{"host": ["a.lan.local"]}]  # newest kept
    assert await admin.dedupe_route("nerdit-route-a") == 0  # idempotent
    await admin.aclose()


async def test_caddy_admin_delete_idempotent():
    fake = FakeCaddy()
    fake.routes.append({"@id": "nerdit-route-a", "match": [], "handle": []})
    admin = _admin(fake)
    assert await admin.delete_route("nerdit-route-a") is True  # existed
    assert await admin.delete_route("nerdit-route-a") is False  # gone → 500 unknown obj
    await admin.aclose()


async def test_caddy_admin_live_routes_parses_dials():
    fake = FakeCaddy()
    fake.routes.append(
        {
            "@id": "nerdit-route-a",
            "match": [{"path": ["/a", "/a/*"]}],
            "handle": [
                {
                    "handler": "subroute",
                    "routes": [
                        {
                            "handle": [
                                {"handler": "rewrite", "strip_path_prefix": "/a"},
                                {
                                    "handler": "reverse_proxy",
                                    "upstreams": [{"dial": "127.0.0.1:9400"}],
                                },
                            ]
                        }
                    ],
                }
            ],
        }
    )
    fake.routes.append({"@id": "operator-added", "handle": []})  # foreign → ignored
    admin = _admin(fake)
    live = await admin.live_routes()
    assert live == {"nerdit-route-a": LiveRoute(dial="127.0.0.1:9400", shape="path", index=0)}
    await admin.aclose()


async def test_caddy_admin_live_routes_reports_matcher_shape():
    # P3.5 S4: live_routes classifies the matcher shape so reconcile can detect
    # a stale matcher whose dial still matches (mode-flip drift, D3).
    fake = FakeCaddy()
    fake.routes.append(
        {
            "@id": "nerdit-route-a",
            "match": [{"path": ["/a", "/a/*"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9400"}]}],
        }
    )
    fake.routes.append(
        {
            "@id": "nerdit-route-b",
            "match": [{"host": ["b.lan.local"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9401"}]}],
        }
    )
    # a dial-less nerdit route is still reported (prunable / upsertable)
    fake.routes.append({"@id": "nerdit-route-c", "handle": []})
    admin = _admin(fake)
    live = await admin.live_routes()
    assert live["nerdit-route-a"] == LiveRoute(dial="127.0.0.1:9400", shape="path", index=0)
    assert live["nerdit-route-b"] == LiveRoute(dial="127.0.0.1:9401", shape="host", index=1)
    assert live["nerdit-route-c"] == LiveRoute(dial=None, shape="path", index=2)
    # _route_shape is defensive about a missing/malformed match
    assert _route_shape({"handle": []}) == "path"
    assert _route_shape({"match": "garbage"}) == "path"
    await admin.aclose()


# --- 3b. B21: retry-once on RemoteProtocolError for idempotent verbs only -----


async def test_caddy_admin_delete_route_retries_once_on_remote_protocol_error():
    """The B21 flake: a dead keep-alive on the shared client raises on reuse;
    ``delete_route`` (idempotent) retries exactly once and succeeds."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("connection closed")
        return httpx.Response(200)

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    assert await admin.delete_route("nerdit-route-a") is True
    assert calls["n"] == 2
    await admin.aclose()


async def test_caddy_admin_delete_route_propagates_after_one_retry():
    """A second RemoteProtocolError propagates — never more than one retry."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.RemoteProtocolError("connection closed")

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.RemoteProtocolError):
        await admin.delete_route("nerdit-route-a")
    assert calls["n"] == 2
    await admin.aclose()


async def test_caddy_admin_upsert_route_patch_leg_retries_once():
    """upsert_route's PATCH-replace leg (id present) gets the same retry — a
    replayed PATCH just re-replaces the same object, so it is safe."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("connection closed")
        return httpx.Response(200)

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    obj = {"@id": "nerdit-route-a", "match": [], "handle": []}
    await admin.upsert_route(obj)
    assert calls["n"] == 2
    await admin.aclose()


async def test_caddy_admin_upsert_route_append_leg_never_retries():
    """Negative pin: the POST append leg (id absent) must NOT be retried — a
    replay would create a duplicate ``@id`` twin (the hazard upsert_route's own
    docstring documents), so a RemoteProtocolError there propagates untouched."""
    post_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            return httpx.Response(404, json={"error": "unknown object ID 'nerdit-route-a'"})
        if request.method == "POST":
            post_calls["n"] += 1
            raise httpx.RemoteProtocolError("connection closed")
        return httpx.Response(404)

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    obj = {"@id": "nerdit-route-a", "match": [], "handle": []}
    with pytest.raises(httpx.RemoteProtocolError):
        await admin.upsert_route(obj)
    assert post_calls["n"] == 1
    await admin.aclose()


# --- 4-7. ProxyManager reconcile / adopt / degrade / audit --------------------


def _svc(name, status=JobStatus.running):
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state="running",
        config='{"image": "x", "port": 8000}',
    )


async def _seed_running(queries, name, port_seed_job_id=None):
    job = _svc(name)
    await queries.create_job(job)
    await queries.acquire_service_port(name, job.id, 8000, (9400, 9499))
    return job


def _proxy(queries, fake, mode="path", base_domain=None, *, acme=None, data_dir=None):
    settings = ProxySettings(
        enabled=True,
        mode=mode,
        base_domain=base_domain,
        **({"acme": acme} if acme is not None else {}),
    )
    mgr = ProxyManager(
        queries,
        settings,
        hostname="box",
        data_dir=data_dir or "/tmp/nerdit-test",
        admin=_admin(fake),
    )
    mgr._binary = "/usr/bin/caddy"  # pretend the binary exists (enabled gate)
    mgr._available = True  # pretend Caddy is up (skip process management)
    # _ensure_alive would hit pid/ping; force it to "alive" for reconcile tests.
    mgr._pid_alive = lambda: True  # type: ignore[method-assign]
    return mgr


async def test_reconcile_registers_then_idempotent(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await _seed_running(queries, "beta")

    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert set(live) == {"nerdit-route-alpha", "nerdit-route-beta"}
    # shape-aware pin (P3.5 S4): our own path-mode objects read back as "path",
    # so shape comparison never turns a converged path-mode state into writes.
    assert live["nerdit-route-alpha"].shape == "path"

    # routes were persisted as the projection
    ep = await queries.get_service_endpoint("alpha")
    assert ep.route == "/alpha"

    # second reconcile on a converged state → no mutating writes
    fake.requests.clear()
    await mgr.reconcile()
    assert not any(p == "/load" for _, p in fake.requests)
    assert not any(
        p.startswith("/id/") and m in ("POST", "PATCH", "DELETE") for m, p in fake.requests
    )
    assert not any(
        p == "/config/apps/http/servers/nerdit/routes" and m == "POST" for m, p in fake.requests
    )


async def test_reconcile_registers_then_idempotent_subdomain_mode(queries):
    # P3.5 subdomain twin: fresh services register with the host-shaped matcher
    # and the empty-string route projection; steady state stays zero-write.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain", base_domain="lan.local")
    await _seed_running(queries, "alpha")
    await _seed_running(queries, "beta")

    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert set(live) == {"nerdit-route-alpha", "nerdit-route-beta"}
    assert live["nerdit-route-alpha"].shape == "host"

    ep = await queries.get_service_endpoint("alpha")
    assert ep.route == ""

    # second reconcile on a converged state → no mutating writes
    fake.requests.clear()
    await mgr.reconcile()
    assert not any(p == "/load" for _, p in fake.requests)
    assert not any(
        p.startswith("/id/") and m in ("POST", "PATCH", "DELETE") for m, p in fake.requests
    )
    assert not any(
        p == "/config/apps/http/servers/nerdit/routes" and m == "POST" for m, p in fake.requests
    )


async def test_reconcile_drift_repoints_and_prunes(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")
    # a stale nerdit route whose service no longer exists → must be pruned
    fake.routes.append({"@id": "nerdit-route-ghost", "handle": []})
    # alpha already present but pointing at the WRONG port → must be re-pointed
    fake.routes.append(
        {
            "@id": "nerdit-route-alpha",
            "match": [{"path": ["/alpha", "/alpha/*"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:1111"}]}],
        }
    )

    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert "nerdit-route-ghost" not in live  # pruned
    # alpha re-pointed to its real allocated port
    ep = await queries.get_service_endpoint("alpha")
    assert live["nerdit-route-alpha"].dial == f"127.0.0.1:{ep.host_port}"


async def test_reconcile_adopts_running_services_without_route(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    # 3 running services, none registered in Caddy, route IS NULL (enable-flip / restart)
    for n in ("a", "b", "c"):
        await _seed_running(queries, n)
    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert set(live) == {"nerdit-route-a", "nerdit-route-b", "nerdit-route-c"}


async def test_disabled_proxy_noops(queries):
    fake = FakeCaddy()
    settings = ProxySettings(enabled=False)
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/x", admin=_admin(fake))
    mgr._binary = None  # binary missing
    mgr._pid_alive = lambda: False  # no leftover process
    assert mgr.enabled is False
    await _seed_running(queries, "alpha")
    await mgr.start()
    await mgr.reconcile()
    await mgr.register("alpha", 9400)
    assert fake.requests == []  # never touched Caddy


async def test_disabled_start_reaps_leftover_caddy(queries):
    # A leftover nerdit-owned Caddy (ungraceful crash) must be reaped when the
    # daemon restarts with the proxy disabled, so it can't keep serving LAN URLs.
    settings = ProxySettings(enabled=False)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/x", admin=_admin(FakeCaddy())
    )
    mgr._pid_alive = lambda: True  # a leftover process is tracked
    killed: list[bool] = []

    async def fake_kill():
        killed.append(True)

    mgr._kill = fake_kill
    await mgr.start()
    assert killed == [True]
    assert mgr.available is False


async def test_stop_reaps_when_pid_alive(queries):
    # stop() must reap our process even when disabled (pid file is ours).
    settings = ProxySettings(enabled=False)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/x", admin=_admin(FakeCaddy())
    )
    mgr._pid_alive = lambda: True
    killed: list[bool] = []

    async def fake_kill():
        killed.append(True)

    mgr._kill = fake_kill
    await mgr.stop()
    assert killed == [True]


async def test_audit_dedup_on_register_and_deregister(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")

    await mgr.register("alpha", 9400)
    rows, _ = await queries.list_audit_log(limit=50)
    reg = [r for r in rows if r.action == "proxy.route_registered"]
    assert len(reg) == 1

    # re-register same port → no new audit row (de-dup)
    await mgr.register("alpha", 9400)
    rows, _ = await queries.list_audit_log(limit=50)
    assert len([r for r in rows if r.action == "proxy.route_registered"]) == 1

    # deregister → one deregistered row; second is a no-op (already gone)
    await mgr.deregister("alpha")
    await mgr.deregister("alpha")
    rows, _ = await queries.list_audit_log(limit=50)
    assert len([r for r in rows if r.action == "proxy.route_deregistered"]) == 1


# --- 8. review fixes: CaddyAdmin error semantics + foreign-Caddy refusal ------


async def test_live_routes_returns_none_on_error():
    # Fix #5: a transient read failure must be distinguishable from "no routes"
    # so reconcile/register don't treat everything as missing and re-push all.
    fake = FakeCaddy(routes_status=503)
    admin = _admin(fake)
    assert await admin.live_routes() is None
    await admin.aclose()


async def test_upsert_does_not_append_on_non_absent_error():
    # Fix #4: a non-absent failure (transient 5xx on an existing id) must raise,
    # NOT fall through to append a duplicate @id.
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.startswith("/id/"):
            return httpx.Response(503)  # transient, not "absent"
        return httpx.Response(200)

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await admin.upsert_route({"@id": "nerdit-route-x", "handle": []})
    assert not any(p == "/config/apps/http/servers/nerdit/routes" for _, p in calls)
    await admin.aclose()


async def test_reconcile_refuses_foreign_caddy(queries):
    # Fix #1: a foreign Caddy answers the admin port (no nerdit server) and we own
    # no process → reconcile must NOT push nerdit routes into it.
    fake = FakeCaddy(has_nerdit=False)
    settings = ProxySettings(enabled=True)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False  # we hold no live process
    await _seed_running(queries, "alpha")

    await mgr.reconcile()

    assert mgr.available is False
    assert not any(p.startswith("/id/") for _, p in fake.requests)  # no route push
    assert not any(
        p == "/config/apps/http/servers/nerdit/routes" and m == "POST" for m, p in fake.requests
    )


async def test_ensure_alive_adopts_nerdit_caddy_without_pid(queries):
    # Fix #1 interaction regression: a live nerdit Caddy whose pid we don't track
    # (stale/missing pid file) must be ADOPTED (content-first, like start()), not
    # misclassified as dead → doomed respawn. It must register routes normally.
    fake = FakeCaddy(has_nerdit=True)  # our server is present (ours by content)
    settings = ProxySettings(enabled=True)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False  # pid not tracked, but Caddy is up with our server

    await _seed_running(queries, "alpha")
    await mgr.reconcile()

    assert mgr.available is True
    assert "nerdit-route-alpha" in (await mgr._admin.live_routes())


async def test_reconcile_skips_tick_when_live_unreadable(queries):
    # Fix #5: if the live route set can't be read, skip the tick rather than
    # re-pushing every route (which would also spam audit rows).
    fake = FakeCaddy(routes_status=503)
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")

    await mgr.reconcile()

    # no upsert attempted (the only writes would be PATCH/POST /id or append)
    assert not any(
        m in ("POST", "PATCH") and (p.startswith("/id/") or p.endswith("/routes"))
        for m, p in fake.requests
    )


# --- 9. P3.5 S3: adopt-path TLS sync (D2) --------------------------------------


def _adopt_manager(queries, fake, mode="path", base_domain=None):
    """A manager poised to ADOPT the FakeCaddy (fresh state — start() untested-yet)."""
    settings = ProxySettings(enabled=True, mode=mode, base_domain=base_domain)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False  # adoption is content-first, no tracked process
    return mgr


def _acme_adopt_manager(queries, fake):
    """A manager with ``[proxy.acme]`` on, poised to ADOPT the FakeCaddy."""
    settings = ProxySettings(enabled=True, acme=_acme_settings())
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False  # adoption is content-first, no tracked process
    return mgr


async def test_adopting_a_caddy_without_the_acme_listener_is_not_listening(queries, caplog):
    """(review round 1) Adoption is content-checked on the ``nerdit`` server
    ALONE. A Caddy that outlived an ungraceful daemon death — and was spawned
    before ``[proxy.acme]`` was enabled — is adopted on that evidence while
    carrying no ``:80`` listener at all. ``listening`` must report the FACT, not
    infer it from availability, and the operator must be told once."""
    fake = FakeCaddy(has_acme_server=False)
    mgr = _acme_adopt_manager(queries, fake)

    with caplog.at_level(logging.WARNING):
        await mgr.start()

    assert mgr.available is True
    assert mgr.acme_listener_live is False
    assert (await mgr.status_snapshot())["acme"]["listening"] is False
    assert any("carries no 'nerdit-acme-http' server" in r.getMessage() for r in caplog.records)


async def test_adopting_a_caddy_with_the_acme_listener_is_listening(queries):
    """The same adoption with the listener present latches ``True`` — and the
    latch means exactly ONE extra admin read, not one per reconcile tick."""
    fake = FakeCaddy(has_acme_server=True)
    mgr = _acme_adopt_manager(queries, fake)

    await mgr.start()
    assert mgr.acme_listener_live is True

    probes = [p for _, p in fake.requests if p.endswith("/nerdit-acme-http")]
    fake.requests.clear()
    await mgr._ensure_alive()
    await mgr._ensure_alive()
    assert len(probes) == 1
    assert not any(p.endswith("/nerdit-acme-http") for _, p in fake.requests)


async def test_the_acme_listener_latch_is_rechecked_after_a_respawn(queries):
    """The listener belongs to a PROCESS. Invalidating convergence — which every
    availability drop does — must also drop what we believed about it, or a
    respawned Caddy would inherit the previous one's answer."""
    fake = FakeCaddy(has_acme_server=True)
    mgr = _acme_adopt_manager(queries, fake)
    await mgr.start()
    assert mgr.acme_listener_live is True

    mgr._invalidate_convergence()
    assert mgr._acme_listener_live is None
    # Fails closed until the next successful adopt/spawn re-reads it.
    assert mgr.acme_listener_live is False

    fake.has_acme_server = False
    await mgr._ensure_alive()
    assert mgr.acme_listener_live is False


@pytest.mark.parametrize("status", [503, -1, 200])
async def test_an_inconclusive_acme_listener_probe_is_retried_not_latched(queries, caplog, status):
    """(review round 2) The latch is only cleared when an availability episode
    ENDS, so latching a blip pins it for the daemon's whole life.

    One 503 / transport error / unparseable body during ``start()`` used to
    write ``False`` into the latch of a Caddy that was serving :80 perfectly —
    and stayed there, because a healthy Caddy never drops and the guard skips
    every later probe. ``/proxy/status.acme.listening`` then read ``false``
    forever and the ``acme_http_port`` doctor row told the operator to restart
    the daemon, which is the one action that would have "fixed" a state that
    was never real. Only a CONCLUSIVE answer may be latched.

    ``status=200`` is the unparseable-body leg: a 200 whose payload is not JSON.
    """
    fake = FakeCaddy(has_acme_server=True)
    fake.fail_acme_probe_times = 1
    fake.acme_probe_status = status
    if status == 200:
        # A 200 that does not parse — "we could not tell", not "it is absent".
        real = fake.handler

        def handler(request: httpx.Request) -> httpx.Response:
            if (
                request.url.path.endswith("/nerdit-acme-http")
                and fake.acme_probe_calls == 0
                and request.method == "GET"
            ):
                fake.acme_probe_calls += 1
                return httpx.Response(200, text="<html>not json</html>")
            return real(request)

        fake.handler = handler  # type: ignore[method-assign]
        fake.fail_acme_probe_times = 0
    mgr = _acme_adopt_manager(queries, fake)

    with caplog.at_level(logging.WARNING):
        await mgr.start()

    # Nothing latched, and the operator was NOT told the listener is missing.
    assert mgr._acme_listener_live is None
    assert mgr.acme_listener_live is False  # fail-closed for the window
    assert not any("carries no 'nerdit-acme-http' server" in r.getMessage() for r in caplog.records)

    # The next healthy tick re-probes and latches the truth; the one after does not.
    await mgr._ensure_alive()
    assert mgr.acme_listener_live is True
    fake.requests.clear()
    await mgr._ensure_alive()
    assert not any(p.endswith("/nerdit-acme-http") for _, p in fake.requests)


async def test_a_conclusively_absent_acme_listener_still_latches_false(queries, caplog):
    """The other half of the tri-state: Caddy's factual "that key is absent"
    answer is a ``200`` with a JSON ``null`` body (measured on v2.11.4), and it
    must still latch ``False`` and warn once — the round-1 adoption story is
    unchanged, it just no longer shares a code path with "we could not tell"."""
    fake = FakeCaddy(has_acme_server=False)
    mgr = _acme_adopt_manager(queries, fake)

    with caplog.at_level(logging.WARNING):
        await mgr.start()

    assert mgr._acme_listener_live is False
    assert any("carries no 'nerdit-acme-http' server" in r.getMessage() for r in caplog.records)
    fake.requests.clear()
    await mgr._ensure_alive()
    assert not any(p.endswith("/nerdit-acme-http") for _, p in fake.requests)


async def test_server_present_separates_absent_from_unreadable(queries):
    """The primitive itself: ``server_present`` is tri-state, ``has_server``
    stays the fail-closed boolean every pre-WP2 caller was written against."""
    fake = FakeCaddy(has_acme_server=True)
    admin = _admin(fake)

    assert await admin.server_present("nerdit-acme-http") is True
    assert await admin.has_server("nerdit-acme-http") is True

    fake.has_acme_server = False
    assert await admin.server_present("nerdit-acme-http") is False
    assert await admin.has_server("nerdit-acme-http") is False

    fake.fail_acme_probe_times = 2
    fake.acme_probe_calls = 0
    assert await admin.server_present("nerdit-acme-http") is None
    assert await admin.has_server("nerdit-acme-http") is False

    # A 404 is read as "absent" too — tolerance for a Caddy that answers that way.
    assert await admin.server_present("nerdit") is True
    fake.has_nerdit = False
    assert await admin.server_present("nerdit") is False


async def test_a_disabled_acme_block_never_probes_for_the_listener(queries):
    """With ACME off there is no listener to have: no admin read, and the
    projection is ``None`` rather than a synthesized boolean."""
    fake = FakeCaddy()
    mgr = _adopt_manager(queries, fake)

    await mgr.start()

    assert mgr.acme_listener_live is None
    assert not any(p.endswith("/nerdit-acme-http") for _, p in fake.requests)


_LE_PRODUCTION_DIRECTORY = "https://acme-v02.api.letsencrypt.org/directory"


def _production_manager(queries, fake, tmp_path):
    settings = ProxySettings(enabled=True, acme=_acme_settings(directory=_LE_PRODUCTION_DIRECTORY))
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir=str(tmp_path), admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False
    return mgr


def _acme_log(caplog):
    return [r for r in caplog.records if "PRODUCTION directory" in r.getMessage()]


async def test_the_production_directory_warns_before_anything_is_issued(queries, tmp_path, caplog):
    """The staging GUARD: a node pointed at Let's Encrypt production with nothing
    on disk yet is one wrong DNS record away from burning the duplicate-cert
    limit, so it is told — loudly — to validate against staging first."""
    fake = FakeCaddy(has_acme_server=True)
    mgr = _production_manager(queries, fake, tmp_path)

    with caplog.at_level(logging.INFO):
        await mgr.start()

    rows = _acme_log(caplog)
    assert len(rows) == 1
    assert rows[0].levelno == logging.WARNING
    assert "acme-staging-v02" in rows[0].getMessage()


async def test_the_production_directory_is_info_once_a_leaf_exists(queries, tmp_path, caplog):
    """(review round 1) …and stops being a warning the moment this node has
    issued from that directory. The default directory IS production, so the old
    unconditional WARNING fired on every boot of every correctly configured
    node — including after the operator did validate against staging. A warning
    on the correct steady state teaches operators to ignore warnings."""
    fake = FakeCaddy(has_acme_server=True)
    (tmp_path / "caddy" / "certificates" / "acme-v02.api.letsencrypt.org-directory").mkdir(
        parents=True
    )
    mgr = _production_manager(queries, fake, tmp_path)

    with caplog.at_level(logging.INFO):
        await mgr.start()

    rows = _acme_log(caplog)
    assert len(rows) == 1
    assert rows[0].levelno == logging.INFO
    assert "acme-staging-v02" not in rows[0].getMessage()


async def test_caddy_admin_tls_config_get_set_and_error_tolerance():
    fake = FakeCaddy()
    admin = _admin(fake)
    cfg = await admin.get_tls_config()
    assert cfg["certificates"]["automate"] == ["box"]
    await admin.set_tls_config({"certificates": {"automate": ["box", "*.lan.local"]}})
    assert fake.tls == {"certificates": {"automate": ["box", "*.lan.local"]}}
    await admin.aclose()
    # same tolerance as live_routes: unreadable → None (NOT {}), so callers can
    # tell "retry later" apart from "genuinely different config".
    fake2 = FakeCaddy(tls_status=503)
    admin2 = _admin(fake2)
    assert await admin2.get_tls_config() is None
    await admin2.aclose()


async def test_start_adopt_syncs_stale_tls_and_leaves_routes_untouched(queries):
    # Flip-to-subdomain + daemon restart over a still-running Caddy: the adopted
    # TLS app still automates only the bare host → replaced with the wildcard
    # set. Routes are NOT touched by the TLS sync (subtree POST, never /load).
    fake = FakeCaddy()  # tls automate == ["box"] (stale for subdomain mode)
    seeded = {"@id": "nerdit-route-alpha", "match": [], "handle": []}
    fake.routes.append(dict(seeded))
    mgr = _adopt_manager(queries, fake, mode="subdomain", base_domain="lan.local")

    await mgr.start()

    assert mgr.available is True
    assert fake.tls == _tls_app(["box", "*.lan.local"])
    assert fake.routes == [seeded]  # untouched
    assert not any(p == "/load" for _, p in fake.requests)


async def test_start_adopt_with_the_desired_subtree_skips_post(queries):
    # The live subtree already IS the desired one → adoption performs zero TLS
    # writes and latches on the hash it verified.
    fake = FakeCaddy(tls=_tls_app(["box", "*.lan.local"]))
    mgr = _adopt_manager(queries, fake, mode="subdomain", base_domain="lan.local")

    await mgr.start()

    assert mgr.available is True
    assert ("POST", "/config/apps/tls") not in fake.requests
    assert mgr._tls_synced is True
    assert mgr._tls_hash == mgr._tls_hash_of(_tls_app(["box", "*.lan.local"]))


async def test_converge_tls_rewrites_a_reordered_live_subtree_once(queries):
    # (P26 D-P26-4) Convergence moved from a SET comparison of ``automate`` to a
    # canonical-JSON hash of the whole subtree, because the policy list is
    # ORDERED and the catch-all must be last — an order-insensitive comparison
    # could not see a catch-all that had drifted ahead of the subjects policy.
    # The cost is that a cosmetically reordered live list is now drift: it is
    # rewritten EXACTLY ONCE, and the tick after that is write-free.
    fake = FakeCaddy(tls=_tls_app(["nerdit.lan", "box", "192.168.1.50"]))
    settings = ProxySettings(enabled=True, extra_hostnames=["192.168.1.50", "nerdit.lan"])
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )

    await mgr._converge_tls()

    assert mgr._tls_synced is True
    assert fake.tls == _tls_app(["box", "192.168.1.50", "nerdit.lan"])
    fake.requests.clear()
    await mgr._converge_tls()
    assert not any(p == "/config/apps/tls" for _, p in fake.requests)


async def test_converge_tls_pushes_a_newly_added_extra_hostname(queries):
    # The drift twin of the test above: a name added to the registry IS drift,
    # and the pushed subtree carries the full subject list in registry order.
    fake = FakeCaddy()  # automate == ["box"]
    settings = ProxySettings(enabled=True, extra_hostnames=["192.168.1.50"])
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )

    await mgr._converge_tls()

    assert mgr._tls_synced is True
    assert fake.tls == _tls_app(["box", "192.168.1.50"])


async def test_path_mode_adopt_never_writes_tls(queries):
    # Path-mode pin: the P3 single-subject config IS current, so adoption stays
    # byte-identical (no TLS write) — S3 must not perturb path mode.
    fake = FakeCaddy()  # default tls == path-mode bootstrap for "box"
    mgr = _adopt_manager(queries, fake)

    await mgr.start()
    await mgr.reconcile()

    assert mgr.available is True
    assert ("POST", "/config/apps/tls") not in fake.requests


async def test_reconcile_ticks_converge_tls_exactly_once(queries):
    # _ensure_alive's adopt branch runs on EVERY healthy reconcile tick (~5s);
    # the _tls_synced flag must keep the TLS write a one-shot, not a per-tick one.
    fake = FakeCaddy()  # stale single-subject tls for a subdomain manager
    mgr = _adopt_manager(queries, fake, mode="subdomain", base_domain="lan.local")

    for _ in range(3):
        await mgr.reconcile()

    posts = [r for r in fake.requests if r == ("POST", "/config/apps/tls")]
    assert len(posts) == 1
    assert fake.tls["certificates"]["automate"] == ["box", "*.lan.local"]
    # once synced, later ticks don't even re-read the TLS subtree
    fake.requests.clear()
    await mgr.reconcile()
    assert not any(p == "/config/apps/tls" for _, p in fake.requests)


async def test_converge_tls_retries_after_unreadable_then_converges(queries):
    # An unreadable TLS subtree must NOT set the flag (nothing was verified);
    # the next tick re-reads and converges.
    fake = FakeCaddy(tls_status=503)
    mgr = _adopt_manager(queries, fake, mode="subdomain", base_domain="lan.local")

    await mgr.reconcile()
    assert mgr._tls_synced is False
    assert ("POST", "/config/apps/tls") not in fake.requests

    fake.tls_status = 200  # blip over
    await mgr.reconcile()
    assert mgr._tls_synced is True
    assert fake.tls["certificates"]["automate"] == ["box", "*.lan.local"]


# --- 10. P3.5 S4: shape-aware drift detection + write-order hardening (D3) -----


async def test_reconcile_failed_upsert_keeps_db_route_then_converges(queries):
    # Staleness scenario (a): the DB rewrite must run AFTER a successful upsert.
    # If the upsert fails on the flip tick, the row must keep the OLD path route
    # so route_changed re-fires and the next tick converges — writing the DB
    # first would freeze reconcile on "converged" while Caddy never got the
    # new-shape object.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain")
    await _seed_running(queries, "alpha")
    await queries.set_endpoint_route("alpha", "/alpha")  # path-mode leftover (pre-flip)
    fake.fail_upserts = 1

    await mgr.reconcile()
    ep = await queries.get_service_endpoint("alpha")
    assert ep.route == "/alpha"  # failed upsert → DB untouched, will retry
    assert fake.routes == []  # nothing landed in Caddy either

    await mgr.reconcile()  # blip over → converges
    ep = await queries.get_service_endpoint("alpha")
    assert ep.route == ""
    live = await mgr._admin.live_routes()
    assert live["nerdit-route-alpha"].shape == "host"


async def test_reconcile_upserts_on_shape_drift_when_dial_and_db_match(queries):
    # Staleness scenario (b): the DB already holds the subdomain projection ("")
    # and the live dial matches, but Caddy still serves the old PATH-shaped
    # matcher under the same @id. Dial-only comparison calls this converged
    # forever; shape-awareness must force the upsert.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain")
    await _seed_running(queries, "alpha")
    await queries.set_endpoint_route("alpha", "")  # DB already converged
    ep = await queries.get_service_endpoint("alpha")
    fake.routes.append(
        {
            "@id": "nerdit-route-alpha",
            "match": [{"path": ["/alpha", "/alpha/*"]}],
            "handle": [
                {"handler": "reverse_proxy", "upstreams": [{"dial": f"127.0.0.1:{ep.host_port}"}]}
            ],
        }
    )

    await mgr.reconcile()

    (route_obj,) = [r for r in fake.routes if r.get("@id") == "nerdit-route-alpha"]
    assert route_obj["match"] == [{"host": ["alpha.box"]}]  # base_domain→hostname fallback
    assert len(fake.routes) == 1  # replaced under the same @id, never duplicated


async def test_register_upserts_on_shape_drift(queries):
    # The inline fast-path uses the same shape-aware comparison: a matching dial
    # under the wrong matcher shape is drift, not "already current".
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain")
    fake.routes.append(
        {
            "@id": "nerdit-route-alpha",
            "match": [{"path": ["/alpha", "/alpha/*"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9400"}]}],
        }
    )

    await mgr.register("alpha", 9400)

    (route_obj,) = fake.routes
    assert route_obj["match"] == [{"host": ["alpha.box"]}]
    # and once the shape is current, re-register is a no-op (audit de-dup holds)
    fake.requests.clear()
    await mgr.register("alpha", 9400)
    assert not any(m in ("POST", "DELETE") for m, _ in fake.requests)


async def test_flip_migration_path_to_subdomain_rewrites_route_and_audits_once(queries):
    # Full flip-migration pin (D3 + Invariant #2): pre-existing running services
    # carry path-mode routes end-to-end (DB row AND live Caddy object); a single
    # subdomain-mode tick must migrate both, in place, exactly once per service.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain", base_domain="lan.local")
    for name in ("alpha", "beta"):
        await _seed_running(queries, name)
        await queries.set_endpoint_route(name, f"/{name}")
        ep = await queries.get_service_endpoint(name)
        fake.routes.append(
            {
                "@id": f"nerdit-route-{name}",
                "match": [{"path": [f"/{name}", f"/{name}/*"]}],
                "handle": [
                    {
                        "handler": "subroute",
                        "routes": [
                            {
                                "handle": [
                                    {"handler": "rewrite", "strip_path_prefix": f"/{name}"},
                                    {
                                        "handler": "reverse_proxy",
                                        "upstreams": [{"dial": f"127.0.0.1:{ep.host_port}"}],
                                    },
                                ]
                            }
                        ],
                    }
                ],
            }
        )

    await mgr.reconcile()

    for name in ("alpha", "beta"):
        ep = await queries.get_service_endpoint(name)
        assert ep.route == ""
        (route_obj,) = [r for r in fake.routes if r.get("@id") == f"nerdit-route-{name}"]
        assert route_obj["match"] == [{"host": [f"{name}.lan.local"]}]
        assert all(h.get("handler") != "rewrite" for h in route_obj["handle"])

    rows, _ = await queries.list_audit_log(limit=50)
    reg = [r for r in rows if r.action == "proxy.route_registered"]
    assert len(reg) == 2  # exactly one per service on the flip tick, not global


async def test_flip_migration_subdomain_to_path_rewrites_route_and_audits_once(queries):
    # The reverse direction: a subdomain deployment flipped back to path mode.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="path")
    for name in ("alpha", "beta"):
        await _seed_running(queries, name)
        await queries.set_endpoint_route(name, "")
        ep = await queries.get_service_endpoint(name)
        fake.routes.append(
            {
                "@id": f"nerdit-route-{name}",
                "match": [{"host": [f"{name}.lan.local"]}],
                "handle": [
                    {
                        "handler": "reverse_proxy",
                        "upstreams": [{"dial": f"127.0.0.1:{ep.host_port}"}],
                    }
                ],
            }
        )

    await mgr.reconcile()

    for name in ("alpha", "beta"):
        ep = await queries.get_service_endpoint(name)
        assert ep.route == f"/{name}"
        (route_obj,) = [r for r in fake.routes if r.get("@id") == f"nerdit-route-{name}"]
        assert route_obj["match"] == [{"path": [f"/{name}", f"/{name}/*"]}]
        inner = route_obj["handle"][0]["routes"][0]["handle"]
        assert inner[0] == {"handler": "rewrite", "strip_path_prefix": f"/{name}"}

    rows, _ = await queries.list_audit_log(limit=50)
    reg = [r for r in rows if r.action == "proxy.route_registered"]
    assert len(reg) == 2


async def test_path_mode_steady_state_zero_writes_with_shape_awareness(queries):
    # Path-mode pin (hard constraint): shape-aware drift detection must not
    # perturb a converged path-mode deployment — repeated ticks stay pure reads.
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()  # converge (route object + DB projection)

    fake.requests.clear()
    for _ in range(2):
        await mgr.reconcile()
    assert not any(m in ("POST", "PATCH", "DELETE") for m, _ in fake.requests)
    ep = await queries.get_service_endpoint("alpha")
    assert ep.route == "/alpha"


async def test_reconcile_self_heals_duplicate_route_objects(queries):
    """A stale twin left by the pre-PATCH upsert is pruned; the newest survives.

    Regression for the P3.5 manual runbook finding: on Caddy 2.6.2 the old
    ``POST /id`` upsert appended instead of replacing, so a mode flip left the
    old path-shaped object in FRONT of the new host-shaped one — and the stale
    matcher kept serving the old URL shape.
    """
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain", base_domain="lan.local")
    await _seed_running(queries, "alpha")
    await mgr.reconcile()  # converge: one host-shaped route, DB route == ""

    # Inject the stale path-shaped twin ahead of the converged object.
    fake.routes.insert(
        0,
        {
            "@id": "nerdit-route-alpha",
            "match": [{"path": ["/alpha", "/alpha/*"]}],
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9400"}]}],
        },
    )
    fake.requests.clear()
    await mgr.reconcile()

    assert [r["@id"] for r in fake.routes] == ["nerdit-route-alpha"]
    assert fake.routes[0]["match"] == [{"host": ["alpha.lan.local"]}]
    # The survivor was already current — dedupe is not an upsert, no route push.
    assert not any(m in ("POST", "PATCH") and p.startswith("/id/") for m, p in fake.requests)


async def test_reconcile_prunes_duplicated_orphan_completely(queries):
    """An orphaned id with duplicate copies is fully removed in one tick."""
    fake = FakeCaddy()
    fake.routes = [
        {"@id": "nerdit-route-ghost", "match": [{"path": ["/ghost", "/ghost/*"]}], "handle": []},
        {"@id": "nerdit-route-ghost", "match": [{"host": ["ghost.lan.local"]}], "handle": []},
    ]
    mgr = _proxy(queries, fake)
    await mgr.reconcile()
    assert fake.routes == []


# --- 11. P9.5: dashboard apex route (U1–U7) -----------------------------------


def _apex_proxy(
    queries, fake, *, dashboard_apex=True, mode="path", upstream="127.0.0.1:9321", base_domain=None
):
    """A reconcile-ready manager with the dashboard-apex feature configured."""
    settings = ProxySettings(
        enabled=True, mode=mode, base_domain=base_domain, dashboard_apex=dashboard_apex
    )
    mgr = ProxyManager(
        queries,
        settings,
        hostname="box",
        data_dir="/tmp/nerdit-test",
        admin=_admin(fake),
        dashboard_upstream=upstream,
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._available = True
    mgr._pid_alive = lambda: True  # type: ignore[method-assign]
    return mgr


async def test_apex_registered_when_enabled_path_mode(queries):
    # U1: enabled ∧ dashboard_apex ∧ path → the routes array ENDS with a
    # catch-all (no matcher, no strip) apex reverse-proxying the daemon.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await mgr.reconcile()
    apex = fake.routes[-1]
    assert apex["@id"] == "nerdit-apex"
    assert "match" not in apex  # catch-all
    handler = apex["handle"][0]
    assert handler["handler"] == "reverse_proxy"  # direct, no rewrite/strip
    assert handler["upstreams"] == [{"dial": "127.0.0.1:9321"}]
    assert _extract_dial(apex) == "127.0.0.1:9321"
    # de-dup: a converged steady tick performs no writes.
    fake.requests.clear()
    await mgr.reconcile()
    assert not any(m in ("POST", "PATCH", "DELETE") for m, _ in fake.requests)


async def test_apex_is_last_after_service_route(queries):
    # U2: the shadowing guarantee — the service route PRECEDES the apex catch-all
    # so Caddy's first-terminal-match evaluation reaches the service first.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    ids = [r["@id"] for r in fake.routes]
    assert ids[-1] == "nerdit-apex"
    assert "nerdit-route-alpha" in ids
    assert ids.index("nerdit-route-alpha") < ids.index("nerdit-apex")


async def test_apex_reorder_self_heal_then_zero_writes(queries):
    # U3: a live array with the apex NOT last (a service appended after it, here
    # simulated by moving the apex to the front) → reconcile deletes+re-appends
    # the apex last; a follow-up steady tick performs ZERO writes.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    assert fake.routes[-1]["@id"] == "nerdit-apex"

    apex_obj = fake.routes.pop()  # the apex
    fake.routes.insert(0, apex_obj)  # now NOT last
    assert fake.routes[-1]["@id"] != "nerdit-apex"

    fake.requests.clear()
    await mgr.reconcile()  # heal
    assert fake.routes[-1]["@id"] == "nerdit-apex"
    assert any(m == "DELETE" and p == "/id/nerdit-apex" for m, p in fake.requests)
    assert any(
        m == "POST" and p == "/config/apps/http/servers/nerdit/routes" for m, p in fake.requests
    )


async def test_apex_dedupes_duplicate_catchalls_before_converging(queries):
    # Finding 2 (Codex P2): two _reconcile_apex calls (the reconcile tick and the
    # inline register() fast-path) can each append a catch-all after both read the
    # apex as absent. apex_state's is_last/dial reflect the LAST copy, so a naive
    # "last copy looks converged" check would leave an EARLIER duplicate catch-all
    # that shadows every service ordered after it. reconcile must collapse twins.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    assert [r["@id"] for r in fake.routes] == ["nerdit-route-alpha", "nerdit-apex"]

    # Inject a duplicate apex at the FRONT: it shadows the service, yet the LAST
    # apex copy is array-last with the right dial (the exact trap for a
    # count-blind convergence check).
    fake.routes.insert(0, dict(fake.routes[-1]))
    assert [r["@id"] for r in fake.routes] == [
        "nerdit-apex",
        "nerdit-route-alpha",
        "nerdit-apex",
    ]

    await mgr.reconcile()
    ids = [r["@id"] for r in fake.routes]
    assert ids.count("nerdit-apex") == 1  # duplicate collapsed
    assert ids[-1] == "nerdit-apex"  # single apex, last
    assert ids.index("nerdit-route-alpha") < ids.index("nerdit-apex")  # service wins

    fake.requests.clear()
    await mgr.reconcile()
    assert not any(m in ("POST", "PATCH", "DELETE") for m, _ in fake.requests)


async def test_apex_reanchored_when_new_service_added_organic(queries):
    # U3 (organic): apex present+last from a prior tick; a genuinely NEW service
    # is seeded and one reconcile appends its route AFTER the apex (upsert_route's
    # absent-id path lands it last), then _reconcile_apex re-anchors the apex
    # below it — same-tick self-correction, no manual array surgery.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    assert fake.routes[-1]["@id"] == "nerdit-apex"

    await _seed_running(queries, "beta")
    await mgr.reconcile()

    ids = [r["@id"] for r in fake.routes]
    assert ids[-1] == "nerdit-apex"  # still last after the new service
    assert ids.index("nerdit-route-beta") < ids.index("nerdit-apex")
    assert ids.index("nerdit-route-alpha") < ids.index("nerdit-apex")


async def test_apex_reanchored_inline_register_keeps_apex_last(queries):
    # The inline fast-path must NOT leave a fresh service shadowed by the apex:
    # register() appends the new service route (absent id) AFTER the apex, then
    # re-anchors the apex last within the SAME call, so the URL is live now, not
    # after the next reconcile tick.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    await mgr.reconcile()  # apex established, last
    assert fake.routes[-1]["@id"] == "nerdit-apex"

    await mgr.register("alpha", 9400)

    ids = [r["@id"] for r in fake.routes]
    assert ids[-1] == "nerdit-apex"  # re-anchored last by register(), same tick
    assert "nerdit-route-alpha" in ids
    assert ids.index("nerdit-route-alpha") < ids.index("nerdit-apex")


async def test_apex_absent_when_flag_off_and_pruned(queries):
    # U4a: dashboard_apex=false → no apex is created; a pre-existing one (flag
    # flipped off) is pruned.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake, dashboard_apex=False)
    await mgr.reconcile()
    assert not any(r["@id"] == "nerdit-apex" for r in fake.routes)

    fake.routes.append(
        {
            "@id": "nerdit-apex",
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9321"}]}],
        }
    )
    await mgr.reconcile()
    assert not any(r["@id"] == "nerdit-apex" for r in fake.routes)
    rows, _ = await queries.list_audit_log(limit=50)
    assert any(r.action == "proxy.apex_deregistered" for r in rows)


async def test_apex_absent_in_subdomain_mode(queries):
    # U4b: subdomain mode leaves the apex routeless regardless of the flag; a
    # pre-existing apex left by a mode flip is pruned.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake, dashboard_apex=True, mode="subdomain")
    fake.routes.append(
        {
            "@id": "nerdit-apex",
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9321"}]}],
        }
    )
    await mgr.reconcile()
    assert not any(r["@id"] == "nerdit-apex" for r in fake.routes)


async def test_apex_dial_drift_converges_and_audits_once(queries):
    # U5: a stale-dial apex (already last) → single converge to the real daemon
    # upstream, audited once; the converged steady tick adds no more audit rows.
    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake, upstream="127.0.0.1:9999")
    fake.routes.append(
        {
            "@id": "nerdit-apex",
            "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:1111"}]}],
        }
    )

    await mgr.reconcile()
    apex = [r for r in fake.routes if r["@id"] == "nerdit-apex"]
    assert len(apex) == 1  # replaced, not duplicated
    assert _extract_dial(apex[0]) == "127.0.0.1:9999"
    rows, _ = await queries.list_audit_log(limit=50)
    assert len([r for r in rows if r.action == "proxy.apex_registered"]) == 1

    await mgr.reconcile()  # converged → no per-tick audit spam
    rows, _ = await queries.list_audit_log(limit=50)
    assert len([r for r in rows if r.action == "proxy.apex_registered"]) == 1


async def test_apex_not_reconciled_when_proxy_disabled(queries):
    # U6: proxy disabled → reconcile early-returns before the apex step; nothing
    # is pushed and no apex lingers.
    fake = FakeCaddy()
    settings = ProxySettings(enabled=False, dashboard_apex=True)
    mgr = ProxyManager(
        queries,
        settings,
        hostname="box",
        data_dir="/tmp/x",
        admin=_admin(fake),
        dashboard_upstream="127.0.0.1:9321",
    )
    mgr._binary = None
    mgr._pid_alive = lambda: False
    await mgr.reconcile()
    assert fake.requests == []
    assert not any(r.get("@id") == "nerdit-apex" for r in fake.routes)


async def test_apex_skips_tick_when_state_unreadable(queries):
    # An unreadable apex_state must be skipped, NOT misread as "apex absent"
    # (which would re-append + re-audit every blip). live_routes and apex_state
    # read the SAME endpoint, so a blanket routes 503 makes reconcile bail at
    # `live is None` BEFORE ever reaching _reconcile_apex (the state-is-None
    # guard would then be untested). Instead fail ONLY the apex_state read: the
    # first routes GET (live_routes) succeeds so reconcile proceeds, the second
    # (apex_state) returns 503 → None → _reconcile_apex must skip, appending
    # nothing and auditing nothing.
    fake = FakeCaddy(fail_routes_get_after=1)
    mgr = _apex_proxy(queries, fake)
    await mgr.reconcile()
    # apex_state saw None → no apex appended, no audit, despite the flag being on.
    assert not any(r.get("@id") == "nerdit-apex" for r in fake.routes)
    assert not any(
        m == "POST" and p == "/config/apps/http/servers/nerdit/routes" for m, p in fake.requests
    )
    rows, _ = await queries.list_audit_log(limit=50)
    assert not any(r.action == "proxy.apex_registered" for r in rows)


async def test_dashboard_apex_setting_default_and_restart_key():
    # U7: default is False (O1 locked — no behavior change for existing users);
    # it round-trips and is flagged restart-required.
    from nerdit.config.store import _RESTART_KEYS

    assert ProxySettings().dashboard_apex is False
    assert ProxySettings(dashboard_apex=True).dashboard_apex is True
    assert "dashboard_apex" in _RESTART_KEYS["proxy"]


async def test_dashboard_apex_config_store_roundtrip(tmp_path):
    # U7 (store leg): staging the key surfaces requires_restart and it persists.
    from nerdit.config.store import ConfigStore

    store = ConfigStore(tmp_path / "config.toml")
    staged = store.stage("proxy", {"dashboard_apex": True})
    assert staged.requires_restart is True
    assert "proxy.dashboard_apex" in staged.restart_keys
    store.commit(staged)
    assert store.effective_section("proxy")["dashboard_apex"] is True


# --- 12. P13b WP7: state enum + status_snapshot -------------------------------


async def _never_ready(*args, **kwargs) -> bool:
    return False


def _snap_manager(fake, *, mode="path", base_domain=None):
    """A manager wired to a FakeCaddy for status_snapshot reads (proxy 'available')."""
    settings = ProxySettings(enabled=True, mode=mode, base_domain=base_domain)
    mgr = ProxyManager(
        None, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    return mgr


async def test_state_disabled_when_proxy_off():
    settings = ProxySettings(enabled=False)
    mgr = ProxyManager(None, settings, hostname="box", data_dir="/tmp/nerdit-test")
    mgr._binary = "/usr/bin/caddy"  # binary present, but proxy off → disabled wins
    assert mgr.state is ProxyState.disabled


async def test_state_no_binary_when_enabled_but_missing():
    mgr = _manager()  # enabled=True
    mgr._binary = None  # shutil.which failed
    assert mgr.state is ProxyState.no_binary


async def test_state_starting_before_first_tick():
    mgr = _manager()
    mgr._binary = "/usr/bin/caddy"
    mgr._available = False
    mgr._spawn_attempts = 0
    mgr._conflict = False
    assert mgr.state is ProxyState.starting


async def test_state_backoff_after_failed_respawn(queries):
    # Drive _ensure_alive to a failed respawn (nothing answers the admin port,
    # spawn does nothing) → attempts > 0, not available → backoff.
    def dead(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(dead))
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False
    mgr._spawn = lambda: None  # respawn is a no-op; nothing ever answers
    mgr._wait_ready = _never_ready  # never ready

    alive = await mgr._ensure_alive()
    assert alive is False
    assert mgr._spawn_attempts > 0
    assert mgr.available is False
    assert mgr.state is ProxyState.backoff
    await admin.aclose()


async def test_state_foreign_conflict_sets_and_clears(queries):
    # foreign_conflict: reuse the refuses-foreign scaffolding — a Caddy answers
    # the admin port with NO nerdit server and we own no process.
    fake = FakeCaddy(has_nerdit=False)
    settings = ProxySettings(enabled=True)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False
    await _seed_running(queries, "alpha")

    await mgr.reconcile()

    assert mgr._conflict is True
    assert mgr.state is ProxyState.foreign_conflict


async def test_conflict_clears_when_foreign_process_gone(queries):
    # Regression: a stale foreign_conflict flag must not outlive the foreign
    # process. Tick 1 sees a foreign Caddy (answers ping, no nerdit server, we
    # own no process) → _conflict=True → foreign_conflict. Then the foreign
    # process exits (nothing answers the admin ping) and our respawn keeps
    # failing for an unrelated reason: state must fall through to backoff, NOT
    # keep reporting a stale foreign_conflict that masks the real spawn failure.
    world = {"foreign": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if not world["foreign"]:
            return httpx.Response(503)  # process gone → nothing answers the port
        # A foreign Caddy answers ping but carries no nerdit server block.
        if request.url.path == "/config/" and request.method == "GET":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False  # we hold no live process
    mgr._spawn = lambda: None  # respawn is a no-op; nothing ever answers
    mgr._wait_ready = _never_ready

    # Tick 1: foreign process present → conflict.
    assert await mgr._ensure_alive() is False
    assert mgr._conflict is True
    assert mgr.state is ProxyState.foreign_conflict

    # Foreign process exits; our respawn keeps failing → the stale conflict must
    # clear and the real backoff state become visible.
    world["foreign"] = False
    assert await mgr._ensure_alive() is False
    assert mgr._conflict is False
    assert mgr.state is ProxyState.backoff
    await admin.aclose()


async def test_conflict_cleared_on_adopt(queries):
    # A prior foreign conflict must clear the moment a nerdit-owned Caddy is
    # adopted (content-first, like start()).
    fake = FakeCaddy(has_nerdit=True)
    mgr = _adopt_manager(queries, fake)
    mgr._conflict = True  # stale conflict from an earlier tick

    await mgr.reconcile()

    assert mgr._conflict is False
    assert mgr.state is ProxyState.available


async def test_state_available_when_up():
    fake = FakeCaddy()
    mgr = _snap_manager(fake)
    mgr._available = True
    mgr._conflict = False
    assert mgr.state is ProxyState.available


async def test_relative_seconds_projection():
    mgr = _manager()
    mgr._binary = "/usr/bin/caddy"
    # never spawned → both timers null
    assert mgr.last_spawn_ago_s is None
    assert mgr.next_retry_in_s is None
    # spawned, not available → ago >= 0 (relative), next_retry in [0, backoff]
    mgr._available = False
    mgr._spawn_attempts = 2
    mgr._last_spawn_at = time.monotonic()
    ago = mgr.last_spawn_ago_s
    assert ago is not None and ago >= 0.0 and ago < 1.0
    nxt = mgr.next_retry_in_s
    assert nxt is not None and 0.0 <= nxt <= mgr._backoff_seconds()
    # available → next_retry null even after a spawn
    mgr._available = True
    assert mgr.next_retry_in_s is None


async def test_status_snapshot_readable_counts_live_routes():
    fake = FakeCaddy()
    fake.routes.append({"@id": "nerdit-route-alpha", "match": [], "handle": []})
    fake.routes.append({"@id": "nerdit-route-beta", "match": [], "handle": []})
    mgr = _snap_manager(fake)
    mgr._available = True

    snap = await mgr.status_snapshot()

    assert snap["state"] == "available"
    assert snap["enabled"] is True
    assert snap["available"] is True
    assert snap["routes"]["live_table"] == "readable"
    assert snap["routes"]["count"] == 2
    assert snap["tls"]["subjects"] == ["box"]
    # no /pki/ca/local handler + no on-disk CA → CA absent
    assert snap["ca"]["present"] is False
    assert snap["ca"]["fingerprint"] is None


async def test_status_snapshot_projects_extra_hostnames_with_no_proxy_change():
    # P25 WP5 item 4: /proxy/status surfaces the SAN registry for free because
    # the snapshot already projects _tls_subjects() — pinned rather than assumed,
    # so a future refactor cannot quietly drop the alternates from the surface.
    fake = FakeCaddy()
    settings = ProxySettings(enabled=True, extra_hostnames=["192.168.1.50", "nerd-box.local"])
    mgr = ProxyManager(
        None, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._available = True

    snap = await mgr.status_snapshot()

    assert snap["tls"]["subjects"] == ["box", "192.168.1.50", "nerd-box.local"]


async def test_status_snapshot_unreadable_live_table_is_tri_state():
    # Reuse the transient-503 scaffolding: live_routes/apex_state return None →
    # live_table 'unreadable' with a NULL count (never 0).
    fake = FakeCaddy(routes_status=503)
    mgr = _snap_manager(fake)
    mgr._available = True

    snap = await mgr.status_snapshot()

    assert snap["routes"]["live_table"] == "unreadable"
    assert snap["routes"]["count"] is None
    # apex_state also unreadable → null fields, never "absent"
    assert snap["apex"]["present"] is None
    assert snap["apex"]["is_last"] is None


async def test_status_snapshot_disabled_skips_admin_reads():
    fake = FakeCaddy()
    settings = ProxySettings(enabled=False)
    mgr = ProxyManager(
        None, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"

    snap = await mgr.status_snapshot()

    assert snap["state"] == "disabled"
    assert snap["routes"]["live_table"] == "disabled"
    assert snap["routes"]["count"] is None
    # never touched the routes admin endpoint
    assert not any(p.endswith("/routes") for _, p in fake.requests)


# --- 13. WP22.T (Batch 5, R-B5) characterisation gap-fill --------------------
#
# Pins on the least-covered quarter of the file BEFORE the core/proxy → package
# split (W22.R/A/S/L/M): the exact respawn backoff ladder, the backoff-window
# gate, kill-then-spawn ordering + state reset on a successful respawn, the
# foreign-Caddy guard re-checked AFTER a respawn, TLS-sync reset-then-resync
# across a foreign-conflict bounce, two `start()` branches no prior test drove
# (fresh spawn, foreign refusal), two `_reconcile_apex` exception/dedupe edges,
# `_wait_ready`'s own timeout return, the REAL `_pid_alive` implementation (every
# other test in this file intercepts it via instance-attribute assignment, so
# without these it would never run at all), and a few small CaddyAdmin/manager
# exception-swallowing edges. All against FakeCaddy / httpx.MockTransport —
# no real Caddy, no subprocess spawns.


async def _always_ready(*args, **kwargs) -> bool:
    return True


class _BoomAuditQueries:
    """A queries stand-in whose ``insert_audit_log`` always raises."""

    async def insert_audit_log(self, **kwargs):
        raise RuntimeError("db down")


# -- backoff ladder + _ensure_alive respawn mechanics -------------------------


async def test_backoff_seconds_exponential_ladder_capped_at_60():
    mgr = _manager()
    for attempts, expected in [
        (0, 1.0),
        (1, 2.0),
        (2, 4.0),
        (3, 8.0),
        (4, 16.0),
        (5, 32.0),
        (6, 60.0),  # 2**6 == 64, already past the cap
        (7, 60.0),
        (10, 60.0),
    ]:
        mgr._spawn_attempts = attempts
        assert mgr._backoff_seconds() == expected


async def test_ensure_alive_backoff_window_defers_respawn_without_spawning(queries):
    # Within the backoff window, _ensure_alive must return False WITHOUT ever
    # calling _spawn (a hot-loop respawn is exactly what the window prevents).
    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin(
        "localhost:2019", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False
    mgr._spawn_attempts = 2
    mgr._last_spawn_at = time.monotonic()  # just attempted — still inside the window
    spawned: list[bool] = []
    mgr._spawn = lambda: spawned.append(True)

    alive = await mgr._ensure_alive()

    assert alive is False
    assert spawned == []
    assert mgr._spawn_attempts == 2  # unchanged
    await admin.aclose()


async def test_ensure_alive_respawns_once_backoff_window_elapses(queries):
    # Once the window has elapsed, a tick DOES respawn and increments the
    # attempt counter (even though nothing ever answers, in this case).
    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin(
        "localhost:2019", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False
    mgr._spawn_attempts = 1
    mgr._last_spawn_at = time.monotonic() - 100  # long past any backoff cap
    spawned: list[bool] = []
    mgr._spawn = lambda: spawned.append(True)
    mgr._wait_ready = _never_ready

    alive = await mgr._ensure_alive()

    assert alive is False  # nothing ever answers back
    assert spawned == [True]  # but a respawn WAS attempted
    assert mgr._spawn_attempts == 2
    await admin.aclose()


async def test_ensure_alive_successful_respawn_resets_state_and_kills_before_spawning(queries):
    # A stale live pid must be killed BEFORE the respawn, and a successful
    # respawn (our nerdit server answers) resets the attempt counter and clears
    # the foreign-warned/conflict latches — none of which a doomed respawn may
    # touch.
    order: list[str] = []
    state = {"up": False}

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/config/" and method == "GET":
            return httpx.Response(200, json={}) if state["up"] else httpx.Response(503)
        if path == "/config/apps/http/servers/nerdit" and method == "GET":
            if state["up"]:
                return httpx.Response(200, json={"listen": [], "routes": []})
            return httpx.Response(404)
        return httpx.Response(404)

    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: True  # a stale live process from a previous run
    mgr._foreign_warned = True  # stale latches a real respawn must clear
    mgr._conflict = True

    async def fake_kill() -> None:
        order.append("kill")

    def fake_spawn() -> None:
        order.append("spawn")
        state["up"] = True  # our new process comes up

    mgr._kill = fake_kill
    mgr._spawn = fake_spawn

    # Latches from the process we are about to replace (Codex round 1, P1
    # #3831777092): a fresh Caddy loads the bootstrap config, which carries no
    # custom domains and no observed routes, so none of these may survive.
    mgr._tls_synced = True
    mgr._tls_hash = "stale-digest"
    mgr._live_domain_ids = frozenset({"nerdit-route-demo@app.example.com"})

    alive = await mgr._ensure_alive()

    assert alive is True
    assert order == ["kill", "spawn"]  # kill-then-spawn ordering (:1320-1322)
    assert mgr._spawn_attempts == 0
    assert mgr._foreign_warned is False
    assert mgr._conflict is False
    assert mgr.available is True
    assert mgr._tls_synced is False
    assert mgr._tls_hash is None
    assert mgr._live_domain_ids == frozenset()
    await admin.aclose()


async def test_a_respawn_forces_the_next_converge_tls_to_re_push_the_domains(queries):
    """(Codex round 1, P1 #3831777092) The D-P26-4 gate, end to end across a
    respawn.

    ``_ensure_alive`` cleared the TLS latch on the foreign-conflict, backoff and
    FAILED-respawn branches, but not on the successful one — so after Caddy died
    and was respawned, ``_converge_tls`` short-circuited on the previous
    process's digest with zero admin I/O and returned ``True``. That opened the
    route gate against a brand-new process whose live ``apps.tls`` is
    ``_bootstrap_config``'s — the node's own subjects, no custom domains — which
    is exactly the state in which a Host route lands ahead of its
    internal-issuer policy.
    """
    state = {"up": False}
    writes: list[dict] = []
    bootstrap: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/config/" and method == "GET":
            return httpx.Response(200, json={}) if state["up"] else httpx.Response(503)
        if path == "/config/apps/http/servers/nerdit" and method == "GET":
            if state["up"]:
                return httpx.Response(200, json={"listen": [], "routes": []})
            return httpx.Response(404)
        if path == "/config/apps/tls" and method == "GET":
            # What the FRESH process is actually serving: the bootstrap subtree.
            return httpx.Response(200, json=bootstrap)
        if path == "/config/apps/tls" and method == "POST":
            writes.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.Response(404)

    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: True
    bootstrap.update(mgr._tls_desired_app(()))
    # Converged for one domain against the process that is about to die.
    mgr._tls_synced = True
    mgr._tls_hash = mgr._tls_hash_of(mgr._tls_desired_app(["app.example.com"]))

    async def fake_kill() -> None:
        pass

    def fake_spawn() -> None:
        state["up"] = True

    mgr._kill = fake_kill
    mgr._spawn = fake_spawn

    assert await mgr._ensure_alive() is True
    assert await mgr._converge_tls(["app.example.com"]) is True

    # The tick after the respawn re-pushed the subtree instead of trusting a
    # digest that describes a process which no longer exists.
    assert len(writes) == 1
    assert "app.example.com" in writes[0]["certificates"]["automate"]
    await admin.aclose()


async def test_ensure_alive_respawn_foreign_probe_after_spawn_stays_unavailable(queries):
    # The respawn's OWN readiness check must re-verify content, not just
    # reachability: _wait_ready answering True is not enough if the process
    # that answers carries no nerdit server (a foreign Caddy grabbed the port
    # the instant ours died) — must NOT declare available.
    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin(
        "localhost:2019", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: True

    async def fake_kill() -> None:
        return None

    mgr._kill = fake_kill
    mgr._spawn = lambda: None
    mgr._wait_ready = _always_ready  # the port answers...

    alive = await mgr._ensure_alive()

    assert alive is False  # ...but never with OUR nerdit server loaded (still 404)
    assert mgr.available is False
    assert mgr._spawn_attempts == 1  # attempted once, never "succeeded"
    await admin.aclose()


# -- TLS-sync-on-adoption: reset on the down/foreign path, bounce re-syncs ----


async def test_tls_synced_resets_on_foreign_conflict_then_resyncs_on_readopt(queries):
    # A synced adoption that later loses the admin port to a foreign process
    # must clear _tls_synced (never stay stuck "synced" against a process we no
    # longer own); when a nerdit Caddy answers again, the flag is NOT sticky —
    # it re-verifies and converges again.
    fake = FakeCaddy()  # stale single-subject tls
    mgr = _adopt_manager(queries, fake, mode="subdomain", base_domain="lan.local")

    await mgr.reconcile()
    assert mgr._tls_synced is True
    assert fake.tls["certificates"]["automate"] == ["box", "*.lan.local"]

    # A foreign process takes over the admin port.
    fake.has_nerdit = False
    await mgr.reconcile()
    assert mgr._tls_synced is False
    assert mgr.state is ProxyState.foreign_conflict

    # It leaves, our nerdit Caddy answers again (content-first re-adopt); the
    # live tls is stale again (simulating a config bounce) → resync fires anew.
    fake.tls = {
        "certificates": {"automate": ["box"]},
        "automation": {"policies": [{"subjects": ["box"], "issuers": [{"module": "internal"}]}]},
    }
    fake.has_nerdit = True
    await mgr.reconcile()
    assert mgr._tls_synced is True
    assert fake.tls["certificates"]["automate"] == ["box", "*.lan.local"]
    assert mgr.state is ProxyState.available


# -- start(): fresh-spawn success + foreign refusal (uncovered branches) -----


async def test_start_fresh_spawn_then_becomes_available(queries):
    # A genuinely fresh boot: nothing answers the admin port yet. start() must
    # spawn, wait for readiness, and — since the fake spawn doesn't actually
    # load our config — push the bootstrap config once ready (the "defensive"
    # comment at the has_nerdit_server check in start()).
    state = {"up": False}
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        requests.append((method, path))
        if path == "/config/" and method == "GET":
            return httpx.Response(200, json={}) if state["up"] else httpx.Response(503)
        if path == "/config/apps/http/servers/nerdit" and method == "GET":
            return httpx.Response(404)  # never loaded by our fake spawn
        if path == "/load" and method == "POST":
            return httpx.Response(200)
        return httpx.Response(404)

    settings = ProxySettings(enabled=True)
    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    mgr = ProxyManager(queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=admin)
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False

    def fake_spawn() -> None:
        state["up"] = True

    mgr._spawn = fake_spawn
    await mgr.start()

    assert mgr.available is True
    assert mgr._spawn_attempts == 0
    assert ("POST", "/load") in requests
    await admin.aclose()


async def test_start_refuses_foreign_caddy_when_pid_not_ours(queries):
    # start()'s OWN foreign-refusal branch (distinct from _ensure_alive's): the
    # admin port answers, carries no nerdit server, and we track no pid of our
    # own → refuse rather than clobber a foreign config with load().
    fake = FakeCaddy(has_nerdit=False)
    settings = ProxySettings(enabled=True)
    mgr = ProxyManager(
        queries, settings, hostname="box", data_dir="/tmp/nerdit-test", admin=_admin(fake)
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._pid_alive = lambda: False

    await mgr.start()

    assert mgr.available is False
    assert mgr._conflict is True
    assert mgr._tls_synced is False
    assert not any(p == "/load" for _, p in fake.requests)


# -- _reconcile_apex: unreadable state + dedupe-before-remove when disabled --


async def test_apex_state_exception_skips_tick(queries):
    # An apex_state() that raises outright (not just a bad status) must be
    # swallowed and skip the tick — never misread as "apex absent".
    class _BoomAdmin(CaddyAdmin):
        async def apex_state(self):
            raise RuntimeError("boom")

    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    mgr._admin = _BoomAdmin("localhost:2019", transport=httpx.MockTransport(fake.handler))

    await mgr._reconcile_apex()  # must not raise

    assert not any(r.get("@id") == "nerdit-apex" for r in fake.routes)
    await mgr._admin.aclose()


async def test_reconcile_apex_converge_exception_is_swallowed(queries):
    # The SECOND try/except in _reconcile_apex (the converge step itself, as
    # opposed to the apex_state() read covered above) must likewise swallow an
    # unexpected failure rather than propagate out of the reconcile loop.
    class _BoomAdmin(CaddyAdmin):
        async def append_route(self, obj):
            raise RuntimeError("boom")

    fake = FakeCaddy()
    mgr = _apex_proxy(queries, fake)
    mgr._admin = _BoomAdmin("localhost:2019", transport=httpx.MockTransport(fake.handler))

    await mgr._reconcile_apex()  # must not raise despite append_route failing

    assert not any(r.get("@id") == "nerdit-apex" for r in fake.routes)
    await mgr._admin.aclose()


async def test_apex_disabled_dedupes_duplicate_twins_before_removing(queries):
    # dashboard_apex=false with duplicate leftover twins (a race between the
    # reconcile tick and the inline register() fast-path) must collapse the
    # twins before deleting — deleting only the id-map target would leave a
    # stale duplicate catch-all behind.
    fake = FakeCaddy()
    apex_obj = {
        "@id": "nerdit-apex",
        "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9321"}]}],
    }
    fake.routes = [dict(apex_obj), dict(apex_obj)]
    mgr = _apex_proxy(queries, fake, dashboard_apex=False)

    await mgr._reconcile_apex()

    assert not any(r.get("@id") == "nerdit-apex" for r in fake.routes)


# -- _wait_ready's own timeout path -------------------------------------------


async def test_wait_ready_returns_false_on_timeout():
    admin = CaddyAdmin(
        "localhost:2019", transport=httpx.MockTransport(lambda r: httpx.Response(503))
    )
    mgr = ProxyManager(
        None, ProxySettings(enabled=True), hostname="box", data_dir="/tmp/nerdit-test", admin=admin
    )
    ready = await mgr._wait_ready(timeout=0.05, interval=0.01)
    assert ready is False
    await admin.aclose()


# -- the REAL _pid_alive implementation ---------------------------------------
#
# Every other test in this file intercepts `_pid_alive` via instance-attribute
# assignment (§1.3) — deliberately, so reconcile/register/start tests don't
# depend on real PIDs. That means the real body has NO coverage anywhere else
# in the suite; these are its only exercise.


async def test_pid_alive_false_when_no_pid_file(tmp_path):
    mgr = ProxyManager(None, ProxySettings(enabled=True), hostname="box", data_dir=tmp_path)
    assert mgr._pid_alive() is False


async def test_pid_alive_true_for_a_live_process(tmp_path):
    mgr = ProxyManager(None, ProxySettings(enabled=True), hostname="box", data_dir=tmp_path)
    mgr._pid_file.write_text(str(os.getpid()))  # our own process is definitely alive
    assert mgr._pid_alive() is True


async def test_pid_alive_false_for_garbage_or_dead_pid(tmp_path):
    mgr = ProxyManager(None, ProxySettings(enabled=True), hostname="box", data_dir=tmp_path)
    mgr._pid_file.write_text("not-a-pid")
    assert mgr._pid_alive() is False  # ValueError branch
    mgr._pid_file.write_text("999999999")  # astronomically unlikely to be live
    assert mgr._pid_alive() is False  # ProcessLookupError branch


# -- small CaddyAdmin / manager exception-swallowing edges --------------------


async def test_delete_route_raises_on_non_absent_error():
    # A genuine (non-"unknown object") failure must propagate, not be read as
    # "already gone".
    admin = CaddyAdmin(
        "localhost:2019",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, text="disk full")),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await admin.delete_route("nerdit-route-x")
    await admin.aclose()


async def test_stop_tolerates_admin_close_failure(queries):
    class _BoomAdmin(CaddyAdmin):
        async def aclose(self) -> None:
            raise RuntimeError("boom")

    admin = _BoomAdmin(
        "localhost:2019", transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    mgr = ProxyManager(
        queries, ProxySettings(enabled=False), hostname="box", data_dir="/tmp/x", admin=admin
    )
    mgr._pid_alive = lambda: False
    await mgr.stop()  # must not raise despite the close failure


async def test_audit_route_exception_is_swallowed():
    mgr = ProxyManager(
        _BoomAuditQueries(),
        ProxySettings(enabled=True),
        hostname="box",
        data_dir="/tmp/nerdit-test",
    )
    await mgr._audit_route("proxy.route_registered", "alpha")  # must not raise


async def test_register_returns_when_live_routes_unreadable(queries):
    # A transient read failure inside register()'s fast-path must be a silent
    # no-op — the reconcile loop is the guarantee, this is best-effort only.
    fake = FakeCaddy(routes_status=503)
    mgr = _proxy(queries, fake)
    await mgr.register("alpha", 9400)
    assert fake.routes == []


async def test_manager_live_routes_passthrough(queries):
    # The public passthrough behind `GET /routes` (Invariant #2: observability
    # goes through the typed wrapper, never raw admin JSON) — never exercised
    # via `mgr.live_routes()` itself elsewhere in this file (tests read
    # `mgr._admin.live_routes()` directly).
    fake = FakeCaddy()
    fake.routes.append({"@id": "nerdit-route-alpha", "match": [], "handle": []})
    mgr = _proxy(queries, fake)
    live = await mgr.live_routes()
    assert live is not None and "nerdit-route-alpha" in live


async def test_mode_property_reflects_settings():
    assert _manager(mode="path").mode == "path"
    assert _manager(mode="subdomain").mode == "subdomain"


# --- 8. (P24b / D-P24-4b) the cutover dial swap is a DIAL change, nothing more


async def test_reconcile_dials_the_active_port_when_set(queries):
    """The COALESCE is what the reconcile reads — Invariant #2 stays clean.

    Writing ``active_host_port`` IS the authoritative repoint: the manager needs
    no edit at all, because ``list_active_service_routes`` already feeds
    ``entry.host_port`` into ``new_dial``.
    """
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    job = await _seed_running(queries, "alpha")
    endpoint = await queries.get_service_endpoint("alpha")

    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert live["nerdit-route-alpha"].dial == f"127.0.0.1:{endpoint.host_port}"

    await queries.set_endpoint_active_port("alpha", 51234)
    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert live["nerdit-route-alpha"].dial == "127.0.0.1:51234"

    # And the unwind converges straight back to the stable port.
    await queries.set_endpoint_active_port("alpha", None)
    await mgr.reconcile()
    live = await mgr._admin.live_routes()
    assert live["nerdit-route-alpha"].dial == f"127.0.0.1:{endpoint.host_port}"
    assert job.service_name == "alpha"


async def test_a_cutover_dial_change_creates_no_new_route_object(queries):
    """Invariant #2 pin: one route object, one @id, only the dial moves."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    before = json.loads(json.dumps(fake.routes))
    assert len(before) == 1

    await queries.set_endpoint_active_port("alpha", 51235)
    await mgr.reconcile()

    assert len(fake.routes) == 1, "a dial change must never add a route object"
    after = fake.routes[0]
    assert after["@id"] == before[0]["@id"]
    assert after["match"] == before[0]["match"], "the matcher shape is untouched"
    live = await mgr._admin.live_routes()
    assert live["nerdit-route-alpha"].shape == before_shape(mgr)
    assert live["nerdit-route-alpha"].dial == "127.0.0.1:51235"


def before_shape(mgr) -> str:
    """The manager's own expected matcher shape (path vs host)."""
    return mgr._expected_shape


# --- 9. (P25 WP4) edge auth: emission, the drift dimension, fail-closed -------
#
# The D-P25-7 gate: an added or removed ``authentication`` handler must be as
# visible to the drift classifier as a moved dial, and a declaration that cannot
# be materialized must take the route AWAY (D-P25-8) rather than serve it open.

PW = "s3cr3t-pw"
PW_REF = "${secrets.APP_PW}"
# bcrypt output is salted per call and therefore never byte-stable, so the
# route-object goldens pin a deterministic stand-in and assert STRUCTURE +
# PLACEMENT. The real function is covered by tests/test_edgeauth.py.
GOLDEN_HASH = "$2b$12$GOLDENGOLDENGOLDENGOLDENuBqf5xJp1qYc0m4t7lQeS2rV6oO"


def _stub_hash(plaintext: str) -> str:
    """Deterministic stand-in for ``edgeauth.hash_password`` (§5.4).

    Distinct plaintexts still hash differently — the secret-rotation leg needs a
    fingerprint that actually moves — while the canonical fixture password maps
    onto the literal the golden dicts pin.
    """
    if plaintext == PW:
        return GOLDEN_HASH
    return "$2b$12$" + hashlib.sha256(plaintext.encode()).hexdigest()[:31]


@pytest.fixture
def hash_calls(monkeypatch):
    """Patch ``edgeauth.hash_password`` with a counting deterministic spy.

    Patchable precisely because ``manager.py`` calls it THROUGH the module
    (never imported by value) — the memoization assertions read this list.
    """
    calls: list[str] = []

    def spy(plaintext: str) -> str:
        calls.append(plaintext)
        return _stub_hash(plaintext)

    monkeypatch.setattr(edgeauth, "hash_password", spy)
    return calls


def _auth_handler(user: str = "alice", pw_hash: str = GOLDEN_HASH) -> dict:
    """The exact ``authentication`` handler the manager must emit."""
    return {
        "handler": "authentication",
        "providers": {
            "http_basic": {
                "hash": {"algorithm": "bcrypt"},
                "accounts": [{"username": user, "password": pw_hash}],
            }
        },
    }


class FakeSecrets:
    """A resolver stand-in: ``(service_name, ref) -> plaintext | None``.

    A CALLABLE is all the proxy is given (D-P25-8) — it can resolve the one
    reference a row declares and cannot enumerate anything.
    """

    def __init__(self, values: dict[str, str] | None = None):
        self.values = dict(values or {})
        self.calls: list[tuple[str, str]] = []

    def __call__(self, service_name: str, ref: str) -> str | None:
        self.calls.append((service_name, ref))
        return self.values.get(service_name)


def _protected_proxy(queries, fake, secrets, mode="path", base_domain=None):
    mgr = _proxy(queries, fake, mode=mode, base_domain=base_domain)
    mgr.set_secret_resolver(secrets)
    return mgr


def _protected_svc(name, *, user="alice", ref=PW_REF, blob=..., status=JobStatus.running):
    """A service row declaring ``[deploy].edge_auth`` (persisted verbatim)."""
    cfg = {"image": "x", "port": 8000}
    cfg["edge_auth"] = {"user": user, "password": ref} if blob is ... else blob
    job = _svc(name, status)
    return job.model_copy(update={"config": json.dumps(cfg)})


async def _seed_protected(queries, name, **kw):
    job = _protected_svc(name, **kw)
    await queries.create_job(job)
    await queries.acquire_service_port(name, job.id, 8000, (9400, 9499))
    return job


def _route_writes(fake) -> list[tuple[str, str]]:
    """Every mutating route request the fake saw (upserts, appends, deletes)."""
    return [
        (m, p)
        for m, p in fake.requests
        if (p.startswith("/id/") and m in ("POST", "PATCH", "DELETE"))
        or (
            p.startswith("/config/apps/http/servers/nerdit/routes")
            # PUT is the P26 insert-at-index leg; without it an ordering
            # re-anchor would read as "no writes".
            and m in ("POST", "PUT", "DELETE")
        )
    ]


def _live_auth(route_obj: dict) -> dict | None:
    """The auth handler inside an emitted route object, or ``None``."""

    def walk(node):
        if isinstance(node, dict):
            if node.get("handler") == "authentication":
                return node
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = walk(item)
                if found is not None:
                    return found
        return None

    return walk(route_obj)


# --- 9a. route-object goldens (both modes x {no auth, auth}) ------------------


async def test_route_object_golden_path_mode_no_auth():
    """Pre-P25 emission byte-identical: no ``edge_auth`` ⇒ no handler at all."""
    mgr = _manager()
    obj = mgr._caddy_route_obj(mgr.build_route("myapp", 9401))
    assert obj == {
        "@id": "nerdit-route-myapp",
        "match": [{"path": ["/myapp", "/myapp/*"]}],
        "handle": [
            {
                "handler": "subroute",
                "routes": [
                    {
                        "handle": [
                            {"handler": "rewrite", "strip_path_prefix": "/myapp"},
                            {
                                "handler": "reverse_proxy",
                                "upstreams": [{"dial": "127.0.0.1:9401"}],
                            },
                        ]
                    }
                ],
            }
        ],
    }


async def test_route_object_golden_path_mode_with_auth(hash_calls):
    """The handler leads the INNER subroute — auth gates before any rewriting.

    The shape is the one ``caddy adapt`` emits for a ``basic_auth`` block
    (captured against Caddy v2.11.4) minus its ``hash_cache: {}`` key, which we
    deliberately do not write (Caddy provisions the default cache and the admin
    API returns config as stored, so the read-back stays symmetric).
    """
    mgr = _manager()
    mgr.set_secret_resolver(FakeSecrets({"myapp": PW}))
    spec = mgr.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    obj = mgr._caddy_route_obj(spec)
    assert obj == {
        "@id": "nerdit-route-myapp",
        "match": [{"path": ["/myapp", "/myapp/*"]}],
        "handle": [
            {
                "handler": "subroute",
                "routes": [
                    {
                        "handle": [
                            _auth_handler(),
                            {"handler": "rewrite", "strip_path_prefix": "/myapp"},
                            {
                                "handler": "reverse_proxy",
                                "upstreams": [{"dial": "127.0.0.1:9401"}],
                            },
                        ]
                    }
                ],
            }
        ],
    }
    # ``hash_cache`` is NOT emitted (rev. 2 amendment 6).
    assert "hash_cache" not in json.dumps(obj)
    assert hash_calls == [PW]


async def test_route_object_golden_subdomain_mode_no_auth():
    mgr = _manager(mode="subdomain", base_domain="lan.local")
    obj = mgr._caddy_route_obj(mgr.build_route("myapp", 9401))
    assert obj == {
        "@id": "nerdit-route-myapp",
        "match": [{"host": ["myapp.lan.local"]}],
        "handle": [{"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9401"}]}],
    }


async def test_route_object_golden_subdomain_mode_with_auth(hash_calls):
    """Subdomain mode: the handler precedes ``reverse_proxy`` in the top-level chain."""
    mgr = _manager(mode="subdomain", base_domain="lan.local")
    mgr.set_secret_resolver(FakeSecrets({"myapp": PW}))
    spec = mgr.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    assert mgr._caddy_route_obj(spec) == {
        "@id": "nerdit-route-myapp",
        "match": [{"host": ["myapp.lan.local"]}],
        "handle": [
            _auth_handler(),
            {"handler": "reverse_proxy", "upstreams": [{"dial": "127.0.0.1:9401"}]},
        ],
    }


async def test_auth_handler_precedes_rewrite_and_proxy_by_position(hash_calls):
    """Placement, stated as positions rather than as a whole-object compare."""
    mgr = _manager()
    mgr.set_secret_resolver(FakeSecrets({"myapp": PW}))
    spec = mgr.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    inner = mgr._caddy_route_obj(spec)["handle"][0]["routes"][0]["handle"]
    assert [h["handler"] for h in inner] == ["authentication", "rewrite", "reverse_proxy"]

    sub = _manager(mode="subdomain", base_domain="lan.local")
    sub.set_secret_resolver(FakeSecrets({"myapp": PW}))
    sub_spec = sub.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    assert [h["handler"] for h in sub._caddy_route_obj(sub_spec)["handle"]] == [
        "authentication",
        "reverse_proxy",
    ]


# --- 9b. the drift classifier gains an auth dimension (D-P25-7) ---------------


async def test_extract_auth_fingerprint_reads_our_own_emission(hash_calls):
    """The desired and live fingerprints are ONE function over the same inputs."""
    mgr = _manager()
    mgr.set_secret_resolver(FakeSecrets({"myapp": PW}))
    spec = mgr.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    obj = mgr._caddy_route_obj(spec)
    assert _extract_auth_fingerprint(obj) == auth_fingerprint("alice", GOLDEN_HASH)
    # And an auth-free route reads as None, never as the sentinel.
    assert _extract_auth_fingerprint(mgr._caddy_route_obj(mgr.build_route("myapp", 9401))) is None


async def test_extract_auth_fingerprint_tolerates_caddys_hash_cache_key(hash_calls):
    """``caddy adapt`` emits ``hash_cache: {}``; an unknown sibling is not drift."""
    mgr = _manager()
    mgr.set_secret_resolver(FakeSecrets({"myapp": PW}))
    obj = mgr._caddy_route_obj(
        mgr.build_route("myapp", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
    )
    adapted = json.loads(json.dumps(obj))
    _live_auth(adapted)["providers"]["http_basic"]["hash_cache"] = {}
    assert _extract_auth_fingerprint(adapted) == auth_fingerprint("alice", GOLDEN_HASH)


@pytest.mark.parametrize(
    "handler",
    [
        {"handler": "authentication"},  # no providers at all
        {"handler": "authentication", "providers": {"http_basic": {"accounts": []}}},
        {
            "handler": "authentication",
            "providers": {
                "http_basic": {"accounts": [{"username": "a", "password": "h"}, {"username": "b"}]}
            },
        },
        {"handler": "authentication", "providers": {"http_basic": {"accounts": [{"user": "a"}]}}},
        {"handler": "authentication", "providers": {"some_future_provider": {}}},
        {"handler": "authentication", "providers": "not-a-table"},
        {"handler": "authentication", "providers": {"http_basic": "not-a-table"}},
        {
            "handler": "authentication",
            "providers": {"http_basic": {"accounts": [{"username": "", "password": ""}]}},
        },
    ],
)
async def test_unparsable_auth_handler_reads_as_drift_not_as_no_auth(handler):
    """An auth handler we cannot reproduce is DRIFT — never "no auth"."""
    obj = {"@id": "nerdit-route-a", "match": [], "handle": [handler]}
    assert _extract_auth_fingerprint(obj) == _AUTH_UNPARSABLE
    assert _extract_auth_fingerprint(obj) is not None


async def test_live_routes_populates_the_auth_fingerprint(hash_calls):
    """The classifier's live half is read in ``CaddyAdmin.live_routes`` or nowhere."""
    mgr = _manager()
    mgr.set_secret_resolver(FakeSecrets({"alpha": PW}))
    fake = FakeCaddy()
    fake.routes = [
        mgr._caddy_route_obj(
            mgr.build_route("alpha", 9401, EdgeAuthSpec(user="alice", password_ref=PW_REF))
        )
    ]
    admin = _admin(fake)
    live = await admin.live_routes()
    assert live["nerdit-route-alpha"].auth_fingerprint == auth_fingerprint("alice", GOLDEN_HASH)
    await admin.aclose()


# --- 9c. reconcile: round-trip, churn, strip, sensitivity --------------------


async def test_reconcile_round_trips_an_authed_route_with_zero_writes(queries, hash_calls):
    """THE D-P25-7 gate: a converged authed route costs no upsert and no audit.

    Warm ``_auth_cache`` (one process), per D-P25-6: bcrypt salts per call, so
    the cross-restart claim is deliberately NOT made here — see the churn test.
    """
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    await _seed_protected(queries, "alpha")

    await mgr.reconcile()
    assert _live_auth(fake.routes[0]) == _auth_handler()

    fake.requests.clear()
    before, _ = await queries.list_audit_log(limit=50)
    await mgr.reconcile()
    assert _route_writes(fake) == []
    after, _ = await queries.list_audit_log(limit=50)
    assert len(after) == len(before)
    # The route is still protected after the no-op tick.
    assert _live_auth(fake.routes[0]) == _auth_handler()


async def test_cold_cache_costs_exactly_one_corrective_upsert_then_settles(queries, hash_calls):
    """(D-P25-6) Restart churn is bounded and convergent, never a strip.

    A fresh process re-hashes with a fresh salt, so the desired fingerprint
    cannot equal the live one and the first tick corrects it — once.
    """
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()

    # A "restarted" daemon: same DB, same live Caddy, a cold manager whose
    # hash for the same plaintext differs (as a real salted bcrypt would).
    mgr2 = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(edgeauth, "hash_password", lambda plaintext: GOLDEN_HASH + "2")
        fake.requests.clear()
        await mgr2.reconcile()
        upserts = [w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]
        assert len(upserts) == 1, "exactly one corrective upsert per authed service"
        assert _live_auth(fake.routes[0]) == _auth_handler(pw_hash=GOLDEN_HASH + "2")

        fake.requests.clear()
        await mgr2.reconcile()
        assert _route_writes(fake) == [], "and then it settles — the churn is bounded"


async def test_a_stripped_auth_handler_is_drift_and_is_restored(queries, hash_calls):
    """The regression this phase exists to prevent: dial+shape match, auth gone."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()

    # Someone (or an older daemon) removes the handler by hand.
    inner = fake.routes[0]["handle"][0]["routes"][0]["handle"]
    del inner[0]
    assert _live_auth(fake.routes[0]) is None

    fake.requests.clear()
    await mgr.reconcile()
    upserts = [w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]
    assert len(upserts) == 1
    assert _live_auth(fake.routes[0]) == _auth_handler()


async def test_changing_only_the_username_moves_the_fingerprint(queries, hash_calls):
    """(D-P25-6 cache key) The memo key carries the user, so renaming it drifts.

    The regression pinned: a cache keyed ``(service, digest)`` returns stale
    material, the fingerprint never moves, and Caddy keeps accepting the
    obsolete username until the next daemon restart.
    """
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert (
        _live_auth(fake.routes[0])["providers"]["http_basic"]["accounts"][0]["username"] == "alice"
    )

    # Same secret (warm cache), new username only.
    cfg = json.loads(job.config)
    cfg["edge_auth"] = {"user": "bob", "password": PW_REF}
    await queries.update_job_config(job.id, json.dumps(cfg))

    fake.requests.clear()
    await mgr.reconcile()
    upserts = [w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]
    assert len(upserts) == 1
    account = _live_auth(fake.routes[0])["providers"]["http_basic"]["accounts"][0]
    assert account["username"] == "bob"

    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == []


async def test_rotating_the_secret_propagates_on_the_next_tick(queries, hash_calls):
    """A rotated secret is picked up by the reconcile loop — no redeploy needed."""
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    first = _extract_auth_fingerprint(fake.routes[0])

    secrets.values["alpha"] = "rotated-pw"
    fake.requests.clear()
    await mgr.reconcile()
    upserts = [w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]
    assert len(upserts) == 1
    assert _extract_auth_fingerprint(fake.routes[0]) != first
    assert _live_auth(fake.routes[0]) == _auth_handler(pw_hash=_stub_hash("rotated-pw"))


async def test_removing_edge_auth_upserts_an_auth_free_route(queries, hash_calls):
    """The disarm direction: dropping the declaration takes the handler away."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert _live_auth(fake.routes[0]) is not None

    await queries.update_job_config(job.id, json.dumps({"image": "x", "port": 8000}))
    fake.requests.clear()
    await mgr.reconcile()
    assert len([w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]) == 1
    assert _live_auth(fake.routes[0]) is None
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == []


async def test_bcrypt_is_memoized_across_ticks(queries, hash_calls):
    """(D-P25-6) bcrypt is ~100 ms by design; N ticks must cost ONE hash."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await _seed_protected(queries, "alpha")
    for _ in range(5):
        await mgr.reconcile()
    assert hash_calls == [PW]


# --- 9d. fail closed (D-P25-8) ----------------------------------------------


async def test_an_unset_secret_prunes_even_a_previously_open_route(queries, hash_calls):
    """Never serve an app its owner asked to protect.

    The transition that matters: the route was live and UNAUTHENTICATED, the
    owner then declared ``edge_auth`` but has not set the secret yet.
    """
    fake = FakeCaddy()
    secrets = FakeSecrets()
    mgr = _protected_proxy(queries, fake, secrets)
    job = await _seed_running(queries, "alpha")  # open route first
    await mgr.reconcile()
    assert len(fake.routes) == 1
    assert _live_auth(fake.routes[0]) is None

    cfg = json.loads(job.config)
    cfg["edge_auth"] = {"user": "alice", "password": PW_REF}
    await queries.update_job_config(job.id, json.dumps(cfg))
    await mgr.reconcile()

    assert fake.routes == [], "the open route is torn down, not left serving"
    # And nothing open is ever emitted while the secret stays unset.
    await mgr.reconcile()
    assert fake.routes == []
    assert hash_calls == []

    # Setting the secret converges the protected route on the next tick.
    secrets.values["alpha"] = PW
    await mgr.reconcile()
    assert _live_auth(fake.routes[0]) == _auth_handler()


@pytest.mark.parametrize(
    "blob",
    [
        {"user": ""},
        {"user": "u"},
        {"user": "u", "password": "hunter2"},
    ],
)
async def test_a_malformed_blob_prunes_and_never_serves_open(queries, hash_calls, blob):
    """(D-P25-8 tri-state) "declared but broken" is NOT "no auth"."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await _seed_running(queries, "alpha")
    await mgr.reconcile()
    assert len(fake.routes) == 1

    job = await queries.get_service_by_name("alpha")
    cfg = json.loads(job.config)
    cfg["edge_auth"] = blob
    await queries.update_job_config(job.id, json.dumps(cfg))
    await mgr.reconcile()

    assert fake.routes == []
    # The negative, asserted explicitly: no route was emitted WITHOUT auth
    # (a bare "no route" check cannot tell a prune from a crash mid-loop).
    assert not any(_live_auth(r) is None for r in fake.routes)
    assert hash_calls == []


async def test_a_non_table_edge_auth_blob_is_malformed_not_absent(queries, hash_calls):
    """A corrupt row (``edge_auth: "yes"``) fails closed, and never raises."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha", blob="yes")
    await mgr.reconcile()
    assert fake.routes == []
    assert job.service_name == "alpha"


async def test_an_unparsable_config_blob_never_raises_in_the_query(queries):
    """The desired-set query feeds the reconcile loop: it degrades, never raises."""
    job = await _seed_running(queries, "alpha")
    await queries.update_job_config(job.id, "{not json")
    rows = await queries.list_active_service_routes()
    assert [r.service_name for r in rows] == ["alpha"]
    assert rows[0].edge_auth is None


async def test_the_desired_set_projects_the_blob_verbatim(queries):
    """The query decides nothing: ``load_edge_auth`` owns the tri-state."""
    await _seed_protected(queries, "alpha")
    await _seed_running(queries, "beta")
    rows = {r.service_name: r for r in await queries.list_active_service_routes()}
    assert rows["alpha"].edge_auth == {"user": "alice", "password": PW_REF}
    assert rows["beta"].edge_auth is None
    # The P24b COALESCE still governs the port this projection carries.
    await queries.set_endpoint_active_port("alpha", 51238)
    rows = {r.service_name: r for r in await queries.list_active_service_routes()}
    assert rows["alpha"].host_port == 51238
    assert rows["alpha"].edge_auth == {"user": "alice", "password": PW_REF}


async def test_a_non_table_blob_projects_as_empty_not_absent(queries):
    """``None`` means "serve openly"; a broken declaration must never reach it."""
    await _seed_protected(queries, "alpha", blob=["not", "a", "table"])
    rows = await queries.list_active_service_routes()
    assert rows[0].edge_auth == {}
    with pytest.raises(EdgeAuthInvalid):
        load_edge_auth(rows[0].edge_auth)


async def test_an_unwired_resolver_fails_closed(queries, hash_calls):
    """A daemon that somehow booted without the wiring withholds, never serves."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)  # no set_secret_resolver
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert fake.routes == []


async def test_a_raising_resolver_fails_closed(queries, hash_calls):
    """Injected code that blows up must not abort the tick — or open the route."""

    def boom(service_name, ref):
        raise RuntimeError("secret store on fire")

    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, boom)
    await _seed_protected(queries, "alpha")
    await _seed_running(queries, "beta")
    await mgr.reconcile()
    assert [r["@id"] for r in fake.routes] == ["nerdit-route-beta"]
    assert _live_auth(fake.routes[0]) is None, "beta declares no edge auth"


async def test_the_withheld_route_is_logged_once_per_transition(queries, hash_calls, caplog):
    """A permanent misconfiguration must not log a line every five seconds."""
    fake = FakeCaddy()
    secrets = FakeSecrets()
    mgr = _protected_proxy(queries, fake, secrets)
    await _seed_protected(queries, "alpha")
    with caplog.at_level("WARNING", logger="nerdit.core.proxy.manager"):
        for _ in range(4):
            await mgr.reconcile()
    withheld = [r for r in caplog.records if "cannot be materialized" in r.getMessage()]
    assert len(withheld) == 1
    # The line names the secret KEY and never a value.
    assert "APP_PW" in withheld[0].getMessage()
    assert PW not in withheld[0].getMessage()


async def test_the_resolver_receives_the_service_and_the_ref_verbatim(queries, hash_calls):
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    await _seed_protected(queries, "alpha", ref="${secrets.shared.APP_PW}")
    await mgr.reconcile()
    assert secrets.calls == [("alpha", "${secrets.shared.APP_PW}")]


async def test_no_plaintext_ever_reaches_the_emitted_route_object(queries, hash_calls):
    """Only the bcrypt hash rides the admin API — never the resolved password."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert PW not in json.dumps(fake.routes)


# --- 9e. the inline register fast-path + its call sites -----------------------


async def test_register_emits_the_handler_inline(queries, hash_calls):
    """The inline fast-path is auth-aware: no unprotected window, not even a tick."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert _live_auth(fake.routes[0]) == _auth_handler()


async def test_register_short_circuits_only_when_the_fingerprint_also_matches(queries, hash_calls):
    """(rev. 2 amendment 5) The already-current conjunction gained the auth term."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    fake.requests.clear()
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert _route_writes(fake) == [], "converged: dial, shape, count AND fingerprint"

    fake.requests.clear()
    await mgr.register("alpha", 9401, edge_auth={"user": "bob", "password": PW_REF})
    assert len([w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]) == 1


async def test_register_deregisters_rather_than_falling_back_to_an_open_route(queries, hash_calls):
    """An unresolvable secret at register time takes the route AWAY."""
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert len(fake.routes) == 1

    secrets.values.clear()
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert fake.routes == []


async def test_register_never_raises_on_a_malformed_blob(queries, hash_calls):
    """A container launch can never be turned into an exception by a bad blob."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await mgr.register("alpha", 9401, edge_auth={"user": "", "password": "hunter2"})
    assert fake.routes == []


async def test_deregister_evicts_the_auth_cache(queries, hash_calls):
    """Credential material goes with the route (D-P25-6 eviction)."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert len(mgr._auth_cache) == 1
    await mgr.deregister("alpha")
    assert mgr._auth_cache == {}
    # A re-register re-hashes rather than reusing an evicted entry.
    await mgr.register("alpha", 9401, edge_auth={"user": "alice", "password": PW_REF})
    assert hash_calls == [PW, PW]


async def test_a_successful_prune_evicts_the_auth_cache(queries, hash_calls):
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert len(mgr._auth_cache) == 1

    await queries.update_job_status(job.id, JobStatus.stopped)
    await mgr.reconcile()
    assert fake.routes == []
    assert mgr._auth_cache == {}


async def test_the_launch_fast_path_carries_the_blob(queries, hash_calls, tmp_path):
    """(D-P25-7) ``core/launch.py``'s inline register passes the row's blob.

    Asserted on the object handed to ``upsert_route`` BY THE LAUNCH ITSELF — no
    proxy reconcile tick runs in this test — so a protected route is never
    briefly replaced by an auth-free one.
    """
    from nerdit.config.settings import ServicesSettings
    from nerdit.core.services import ServiceController
    from tests.test_services_reconcile import FakeRuntime

    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    runtime = FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        proxy=mgr,
    )
    await queries.create_job(_protected_svc("alpha", status=JobStatus.building))
    await controller.reconcile()
    await controller.shutdown()

    assert len(fake.routes) == 1
    assert _live_auth(fake.routes[0]) == _auth_handler()


async def test_the_launch_fast_path_withholds_when_the_secret_is_unset(
    queries, hash_calls, tmp_path
):
    """No ``upsert_route`` call at all — and the launch still succeeds."""
    from nerdit.config.settings import ServicesSettings
    from nerdit.core.services import ServiceController
    from tests.test_services_reconcile import FakeRuntime

    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets())
    runtime = FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        proxy=mgr,
    )
    await queries.create_job(_protected_svc("alpha", status=JobStatus.building))
    await controller.reconcile()
    await controller.shutdown()

    assert fake.routes == []
    assert not [w for w in _route_writes(fake) if w[0] in ("POST", "PATCH")]
    row = await queries.get_service_by_name("alpha")
    assert row.status is JobStatus.running, "the launch itself still succeeds"


async def test_a_cutover_commit_never_strips_the_auth_handler(queries, hash_calls):
    """(rev. 2 amendment 1) The cutover repoint REBUILDS the whole route object.

    Un-amended, every commit of a protected service would replace it with an
    auth-free one — and ``_await_dial`` compares the DIAL ONLY, so the strip
    would still "verify".
    """
    from types import SimpleNamespace

    from nerdit.config.settings import ServicesSettings
    from nerdit.core.cutover import CutoverManager

    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    green_port = 51236

    settled: list[str] = []
    controller = SimpleNamespace(
        _queries=queries,
        _proxy=mgr,
        _services_settings=ServicesSettings(service_port_range="9400-9499"),
        _is_container_live=lambda cid: _false(),
    )
    cutover = CutoverManager(controller)
    cutover._settle_repoint_failure = lambda *a, **kw: _record(settled, "repoint_failure")

    await cutover._commit(job, 2, green_id="c-green", green_port=green_port, name="alpha")

    assert _extract_dial(fake.routes[0]) == f"127.0.0.1:{green_port}"
    assert _live_auth(fake.routes[0]) == _auth_handler(), "the repoint kept the handler"
    # The commit diverted at the post-verify liveness re-check (f2b), which is
    # all this test drives — the dial verify itself passed.
    assert settled == ["repoint_failure"]


async def _false() -> bool:
    return False


async def _record(sink: list[str], value: str) -> None:
    sink.append(value)


async def test_a_cutover_with_an_unresolvable_secret_fails_the_repoint(queries, hash_calls):
    """The accepted corollary: posture beats availability (rev. 2 amendment 1)."""
    from types import SimpleNamespace

    from nerdit.config.settings import ServicesSettings
    from nerdit.core import cutover as cutover_mod
    from nerdit.core.cutover import CutoverManager

    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    job = await _seed_protected(queries, "alpha")
    await mgr.reconcile()

    secrets.values.clear()  # the secret becomes unresolvable mid-cutover
    settled: list[str] = []
    controller = SimpleNamespace(
        _queries=queries,
        _proxy=mgr,
        _services_settings=ServicesSettings(service_port_range="9400-9499"),
    )
    cutover = CutoverManager(controller)
    cutover._settle_repoint_failure = lambda *a, **kw: _record(settled, "repoint_failure")
    # Virtual time: the dial read-back must be allowed to burn its budget.
    clock = {"t": 0.0}

    async def _sleep(seconds):
        clock["t"] += seconds

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cutover_mod, "_SLEEP", _sleep)
        mp.setattr(cutover_mod, "_MONOTONIC", lambda: clock["t"])
        await cutover._commit(job, 2, green_id="c-green", green_port=51237, name="alpha")

    assert settled == ["repoint_failure"], "the dial never converges — fail closed"
    assert fake.routes == [], "and the route is withheld, never served open"


# --- 9f. adoption/respawn + the untouched neighbours --------------------------


async def test_a_post_respawn_tick_rebuilds_authed_routes(queries, hash_calls):
    """(rev. 2 amendment 5) Respawn loads a bootstrap whose routes array is []."""
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()
    assert _live_auth(fake.routes[0]) is not None

    fake.routes.clear()  # a respawned Caddy: full route reset
    await mgr.reconcile()
    assert len(fake.routes) == 1
    assert _live_auth(fake.routes[0]) == _auth_handler()


async def test_the_apex_route_is_never_given_an_auth_handler(queries, hash_calls):
    """(non-goal 4) The dashboard apex has its own bearer auth; it stays untouched."""
    fake = FakeCaddy()
    settings = ProxySettings(enabled=True, dashboard_apex=True)
    mgr = ProxyManager(
        queries,
        settings,
        hostname="box",
        data_dir="/tmp/nerdit-test",
        admin=_admin(fake),
        secret_resolver=FakeSecrets({"alpha": PW}),
    )
    mgr._binary = "/usr/bin/caddy"
    mgr._available = True
    mgr._pid_alive = lambda: True  # type: ignore[method-assign]
    await _seed_protected(queries, "alpha")
    await mgr.reconcile()

    apex = [r for r in fake.routes if r["@id"] == "nerdit-apex"]
    assert len(apex) == 1
    assert _live_auth(apex[0]) is None
    assert fake.routes[-1]["@id"] == "nerdit-apex", "and it is still held LAST"


async def test_the_ca_endpoint_stays_public_with_edge_auth_in_play():
    """(non-goal 4) Edge auth protects APPS; the trust-bootstrap read is untouched.

    ``GET /api/proxy/ca`` is public BY DESIGN (P9) — a client that cannot yet
    trust the CA cannot present a credential either.
    """
    from nerdit.daemon.middleware import _PUBLIC_PATHS

    assert "/api/proxy/ca" in _PUBLIC_PATHS


# --- 9g. the injected resolver (daemon/bootstrap.py) -------------------------


async def test_the_edge_auth_resolver_walks_the_p8_precedence(tmp_path):
    """The closure is the P8 kernel, not a second implementation."""
    from nerdit.core.secrets import SHARED_SCOPE, SecretManager
    from nerdit.daemon.bootstrap import build_edge_auth_resolver

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("alpha", {"APP_PW": "per-service"})
    secrets.set(SHARED_SCOPE, {"SHARED_PW": "shared-value", "ONLY_SHARED": "x"})
    resolve = build_edge_auth_resolver(secrets)

    assert resolve("alpha", "${secrets.APP_PW}") == "per-service"
    assert resolve("alpha", "${secrets.shared.SHARED_PW}") == "shared-value"
    # An unset key is the ordinary fail-closed signal.
    assert resolve("alpha", "${secrets.MISSING}") is None
    # An UNSCOPED ref never falls back to a same-named shared key (P8 precedence).
    assert resolve("alpha", "${secrets.ONLY_SHARED}") is None
    # A service with no store at all resolves to nothing, never raises.
    assert resolve("ghost", "${secrets.APP_PW}") is None


async def test_the_edge_auth_resolver_turns_an_unreadable_store_into_none(tmp_path):
    """A corrupted/undecryptable file must fail closed, not abort a reconcile tick."""
    from nerdit.core.secrets import SecretManager
    from nerdit.daemon.bootstrap import build_edge_auth_resolver

    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("alpha", {"APP_PW": "per-service"})
    (tmp_path / "secrets" / "alpha.enc").write_bytes(b"not a valid envelope")
    assert build_edge_auth_resolver(secrets)("alpha", "${secrets.APP_PW}") is None


# --- 10. (P26 WP1) custom domains: a second route per service ----------------
#
# The whole WP1 proxy contract in one section. Five things have to hold at once
# and each has its own test below:
#
#   * a domain route is Host-matched and ``terminal`` in EITHER proxy mode, and
#     the two default shapes are byte-identical to P3.5's;
#   * it is held AHEAD of every default route, which in path mode is the
#     difference between the operator's domain reaching their app and reaching
#     whichever app happens to own the requested path prefix;
#   * ordering is a DRIFT dimension, not a one-shot placement;
#   * the TLS subtree converges BEFORE any route write, and what it emits can
#     never trigger public issuance;
#   * the steady state is still zero writes.

ROUTES_PATH = "/config/apps/http/servers/nerdit/routes"


def _route_ids(fake) -> list[str]:
    """The live route array's ``@id``s, in array order."""
    return [obj.get("@id") for obj in fake.routes]


def _path_matches(pattern: str, path: str) -> bool:
    if pattern.endswith("/*"):
        return path.startswith(pattern[:-1])
    return path == pattern


def fake_dispatch(routes: list[dict], host: str, path: str) -> str | None:
    """Caddy's own first-match rule over a live route array → the winning dial.

    A faithful-enough router to make shadowing *demonstrable* rather than
    asserted: walk the array in order, a ``host`` matcher matches on equality, a
    ``path`` matcher on an exact hit or a ``prefix/*`` hit, an object with no
    ``match`` key is a catch-all, and the FIRST match wins. That last clause is
    the whole reason domain routes are prepended.
    """
    for obj in routes:
        matchers = obj.get("match")
        if matchers is None:
            return _extract_dial(obj)
        for matcher in matchers:
            hosts = matcher.get("host")
            if hosts is not None and host not in hosts:
                continue
            paths = matcher.get("path")
            if paths is not None and not any(_path_matches(p, path) for p in paths):
                continue
            return _extract_dial(obj)
    return None


async def _seed_domain(queries, job, domain: str, *, acme: bool = False) -> None:
    outcome, _row = await queries.add_service_domain(
        job.service_name, domain, acme=acme, job_id=job.id
    )
    assert outcome == "inserted"


def _domain_row(service_name: str, domain: str, *, acme: bool = False):
    from datetime import UTC, datetime

    from nerdit.db.rows import ServiceDomain

    return ServiceDomain(
        domain=domain, service_name=service_name, acme=acme, created_at=datetime.now(UTC)
    )


# -- 10a. the admin-API primitive (F1/F2/F3) ----------------------------------


async def test_insert_route_first_inserts_and_the_id_then_patches_in_place():
    """F1 + F2 + F3, the three facts the ordering machinery rests on."""
    fake = FakeCaddy()
    admin = _admin(fake)
    # F1: insert into an EMPTY array, then ahead of an existing element.
    await admin.upsert_route({"@id": "nerdit-route-a", "match": [], "handle": []})
    await admin.insert_route_first({"@id": "nerdit-route-a@app.example.com", "handle": []})
    assert _route_ids(fake) == ["nerdit-route-a@app.example.com", "nerdit-route-a"]
    # F2: the inserted object's @id is resolvable — so F3 applies to it.
    live = await admin.live_routes()
    assert live["nerdit-route-a@app.example.com"].index == 0
    # F3: a later upsert PATCHes in place; the array position does not move and
    # no twin appears, even with prepend=True.
    await admin.upsert_route(
        {"@id": "nerdit-route-a@app.example.com", "handle": ["x"]}, prepend=True
    )
    assert _route_ids(fake) == ["nerdit-route-a@app.example.com", "nerdit-route-a"]
    assert fake.routes[0]["handle"] == ["x"]
    await admin.aclose()


async def test_insert_route_first_is_not_retried_on_a_dead_keepalive():
    """A replayed insert is a TWIN — the same rule that keeps POST off the retry."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.RemoteProtocolError("server disconnected", request=request)

    admin = CaddyAdmin("localhost:2019", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.RemoteProtocolError):
        await admin.insert_route_first({"@id": "nerdit-route-a@x.example", "handle": []})
    assert calls["n"] == 1
    await admin.aclose()


# -- 10b. the emitted object --------------------------------------------------


async def test_domain_route_object_shape(hash_calls):
    """Host-matched + terminal in BOTH modes, same auth handler as the default.

    And the load-bearing non-regression: the two DEFAULT shapes are unchanged,
    so nothing an existing deployment serves moves because domains exist.
    """
    for mode, base in (("path", None), ("subdomain", "lan.local")):
        mgr = _manager(mode=mode, base_domain=base)
        mgr.set_secret_resolver(FakeSecrets({"alpha": PW}))
        default = mgr.build_route("alpha", 9400, EdgeAuthSpec(user="alice", password_ref=PW_REF))
        obj = mgr._caddy_route_obj(
            mgr.build_domain_route("alpha", 9400, "app.example.com", auth=default.auth)
        )
        assert obj["@id"] == "nerdit-route-alpha@app.example.com"
        assert obj["match"] == [{"host": ["app.example.com"]}]
        # Terminal: a matched custom domain stops route evaluation.
        assert obj["terminal"] is True
        # No strip in either mode — the domain serves the app at its own root.
        assert all(h.get("handler") != "rewrite" for h in obj["handle"])
        assert _extract_dial(obj) == "127.0.0.1:9400"
        # S-W5: the SAME resolved material as the default route, hashed once.
        assert _live_auth(obj) == _auth_handler()
        assert obj["handle"][0]["handler"] == "authentication"

        default_obj = mgr._caddy_route_obj(default)
        assert "terminal" not in default_obj
        if mode == "path":
            assert default_obj["match"] == [{"path": ["/alpha", "/alpha/*"]}]
        else:
            assert default_obj["match"] == [{"host": ["alpha.lan.local"]}]
    # Two managers x one service = two resolutions; the domain rode along free.
    assert hash_calls == [PW, PW]


async def test_a_subdomain_default_route_is_never_labelled_a_custom_domain(queries):
    """``_spec_domain``: a Host-shaped DEFAULT carries a host but no domain."""
    mgr = _manager(mode="subdomain", base_domain="lan.local")
    spec = mgr.build_route("alpha", 9400)
    assert spec.shape == "host" and spec.host == "alpha.lan.local"
    assert spec.kind == "default"
    from nerdit.core.proxy.manager import _spec_domain

    assert _spec_domain(spec) is None
    assert (
        _spec_domain(mgr.build_domain_route("alpha", 9400, "x.example", auth=None)) == "x.example"
    )


# -- 10c. reconcile: placement, ordering, steady state ------------------------


async def test_reconcile_inserts_domain_routes_at_index_zero_and_pins_order(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    b = await _seed_running(queries, "b")
    await _seed_domain(queries, a, "a.example.com")
    await _seed_domain(queries, b, "b.example.com")

    await mgr.reconcile()

    ids = _route_ids(fake)
    # Order among the domains is free; ALL of them precede the first default.
    domains = {"nerdit-route-a@a.example.com", "nerdit-route-b@b.example.com"}
    assert set(ids) == domains | {"nerdit-route-a", "nerdit-route-b"}
    assert set(ids[:2]) == domains
    live = await mgr._admin.live_routes()
    assert ordering_violations(live) == []


async def test_the_apex_stays_last_with_domain_routes_present(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    mgr._settings.dashboard_apex = True
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")

    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a", "nerdit-apex"]


async def test_second_tick_is_zero_writes_with_domains(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await _seed_domain(queries, a, "b.example.com")

    await mgr.reconcile()
    fake.requests.clear()
    await mgr.reconcile()

    # Reads only — no route write, no TLS write, and not even a TLS read (the
    # convergence latch is keyed on the desired subtree's hash).
    assert _route_writes(fake) == []
    assert {m for m, _ in fake.requests} == {"GET"}
    assert not any(p == "/config/apps/tls" for _, p in fake.requests)


async def test_ordering_is_a_drift_dimension(queries):
    """A domain route dragged behind a default (a foreign edit) is re-anchored."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()

    # Simulate an operator (or an older daemon) appending the domain route last.
    domain_obj = fake.routes.pop(0)
    fake.routes.append(domain_obj)
    assert _route_ids(fake) == ["nerdit-route-a", "nerdit-route-a@a.example.com"]
    before_audits = len((await queries.list_audit_log(limit=200))[0])

    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a"]
    # Position is not content: a re-anchor writes no audit row.
    assert len((await queries.list_audit_log(limit=200))[0]) == before_audits
    # And it converges — the tick after is write-free.
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == []


async def test_path_mode_shadowing_is_fixed_by_ordering(queries):
    """The reason ordering exists, proven both ways with a faithful router."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_running(queries, "b")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()

    a_dial = (await queries.get_service_endpoint("a")).host_port
    b_dial = (await queries.get_service_endpoint("b")).host_port
    request = ("a.example.com", "/b/x")

    # As reconcile leaves it: the Host route is first, so the operator's own
    # domain reaches THEIR app even for a path another app claims.
    assert fake_dispatch(fake.routes, *request) == f"127.0.0.1:{a_dial}"
    # Appended instead, the very same objects serve app "b" under a's domain.
    shadowed = [obj for obj in fake.routes if obj["@id"] != "nerdit-route-a@a.example.com"]
    shadowed.append(next(o for o in fake.routes if o["@id"] == "nerdit-route-a@a.example.com"))
    assert fake_dispatch(shadowed, *request) == f"127.0.0.1:{b_dial}"


async def test_subdomain_mode_domain_routes_precede_host_defaults(queries):
    """One ordering rule, no mode branch — and the emitted shapes still differ."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, mode="subdomain", base_domain="lan.local")
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")

    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a"]
    assert fake.routes[0]["match"] == [{"host": ["a.example.com"]}]
    assert fake.routes[0]["terminal"] is True
    assert fake.routes[1]["match"] == [{"host": ["a.lan.local"]}]
    assert "terminal" not in fake.routes[1]
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == []


async def test_a_domain_route_follows_a_port_change(queries):
    """CRIT-4's twin at the proxy: the domain rides the service's new dial."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()

    await queries.set_endpoint_active_port("a", 51234)
    await mgr.reconcile()

    live = await mgr._admin.live_routes()
    assert live["nerdit-route-a@a.example.com"].dial == "127.0.0.1:51234"
    assert live["nerdit-route-a"].dial == "127.0.0.1:51234"


async def test_prune_handles_domain_ids(queries):
    """A live domain route whose row is gone is pruned and audited by name."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()

    assert await queries.remove_service_domain("a", "a.example.com") is True
    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a"]
    items, _ = await queries.list_audit_log(action="proxy.route_deregistered", limit=10)
    assert items[0].params_redacted == {"service_name": "a", "domain": "a.example.com"}
    assert items[0].target_id == "a", "the audit target stays the SERVICE"


async def test_pruning_one_domain_does_not_rewrite_the_surviving_routes(queries, monkeypatch):
    """D-P25-6 scoping: a DOMAIN prune must not evict the service's auth cache.

    bcrypt salts per call, so a needless eviction re-hashes the same credential
    into a NEW hash; the fingerprint then differs from the live handler's and
    reconcile PATCHes — and re-audits — the default route plus every surviving
    domain route of an app the operator did not touch. The stand-in here is
    per-call unique precisely so that is visible: the shared ``hash_calls``
    stub is deterministic and cannot see it.
    """
    counter = iter(range(1000))

    def _salted(plaintext: str) -> str:
        return f"$2b$12${next(counter):031d}"

    monkeypatch.setattr(edgeauth, "hash_password", _salted)
    fake = FakeCaddy()
    mgr = _protected_proxy(queries, fake, FakeSecrets({"alpha": PW}))
    job = await _seed_protected(queries, "alpha")
    await _seed_domain(queries, job, "a.example.com")
    await _seed_domain(queries, job, "b.example.com")
    await mgr.reconcile()
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == [], "converged, salted hashes and all"

    assert await queries.remove_service_domain("alpha", "b.example.com") is True
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == [("DELETE", "/id/nerdit-route-alpha@b.example.com")]

    before = len((await queries.list_audit_log(limit=500))[0])
    fake.requests.clear()
    await mgr.reconcile()
    assert _route_writes(fake) == [], "the survivors are not re-salted into drift"
    assert len((await queries.list_audit_log(limit=500))[0]) == before


async def test_withheld_edge_auth_withholds_domain_routes_too(queries, hash_calls):
    """S-W5: an unresolvable secret takes the default AND every domain away."""
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    job = await _seed_protected(queries, "alpha")
    await _seed_domain(queries, job, "a.example.com")
    await mgr.reconcile()
    assert set(_route_ids(fake)) == {"nerdit-route-alpha", "nerdit-route-alpha@a.example.com"}

    secrets.values.clear()  # the secret is deleted under a live, protected app
    await mgr.reconcile()

    assert fake.routes == [], "a withheld service serves NOTHING, domains included"
    assert mgr.withheld_services == frozenset({"alpha"})


# -- 10d. TLS convergence (D-P26-4, S-W6) -------------------------------------


async def test_tls_is_pushed_before_the_first_route_write(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")

    await mgr.reconcile()

    tls_push = fake.requests.index(("POST", "/config/apps/tls"))
    first_write = min(fake.requests.index(w) for w in _route_writes(fake))
    assert tls_push < first_write, "the certificate must exist before the Host route"
    assert fake.tls == _tls_app(["box"], ["a.example.com"])

    # A removal changes the hash, so the next tick pushes again — and again
    # BEFORE the route work (here, the prune).
    assert await queries.remove_service_domain("a", "a.example.com") is True
    fake.requests.clear()
    await mgr.reconcile()
    assert fake.tls == _tls_app(["box"])
    tls_push = fake.requests.index(("POST", "/config/apps/tls"))
    first_write = min(fake.requests.index(w) for w in _route_writes(fake))
    assert tls_push < first_write


async def test_domain_routes_are_withheld_until_the_tls_subtree_converges(queries):
    """The WP1 invariant is ENFORCED, not merely sequenced.

    ``_converge_tls`` is best-effort and swallows its failures, so "TLS first"
    alone proved nothing: on an ADOPTED pre-WP1 Caddy — whose live policy list
    carries only ``[subjects → internal]`` and no catch-all — a Host route
    landing after a failed convergence is handed to Caddy's DEFAULT issuers,
    i.e. a real public ACME registration. So a failed convergence withholds
    every domain spec for the tick; the default route still flows.
    """
    fake = FakeCaddy(tls_status=503)
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")

    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a"]
    assert not [w for w in _route_writes(fake) if "@" in w[1]], "no Host route was written"

    # A withholding, not a refusal: the first tick that converges binds it.
    fake.tls_status = 200
    await mgr.reconcile()
    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a"]


async def test_a_live_domain_route_is_torn_down_when_tls_stops_converging(queries):
    """Fail-closed is the safe direction: no Host route without its policy."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()
    assert _route_ids(fake)[0] == "nerdit-route-a@a.example.com"

    # The subtree becomes unreadable AND the latch is invalidated — what a
    # respawn into an unhealthy admin API looks like.
    fake.tls_status = 503
    mgr._tls_synced = False
    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a"]


async def test_converge_tls_reports_whether_the_latch_holds(queries):
    """The return value the two gates read, pinned on its own."""
    fake = FakeCaddy(tls_status=503)
    mgr = _proxy(queries, fake)
    assert await mgr._converge_tls([]) is False
    assert mgr._tls_synced is False

    fake.tls_status = 200
    assert await mgr._converge_tls([]) is True
    assert await mgr._converge_tls([]) is True, "the latched short-circuit is also True"


async def test_no_acme_issuer_anywhere(queries):
    """The WP1 scope guard, asserted on the bytes actually sent to Caddy."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()

    for blob in (mgr._bootstrap_config(), fake.tls):
        text = json.dumps(blob)
        assert "acme" not in text.lower()
        policies = (
            blob["apps"]["tls"]["automation"]["policies"]
            if "apps" in blob
            else (blob["automation"]["policies"])
        )
        # The catch-all is LAST and has no ``subjects`` key: every name Caddy
        # manages, including one it derives from a Host matcher, is internal.
        assert "subjects" not in policies[-1]
        assert all(p["issuers"] == [{"module": "internal"}] for p in policies)
    # There is no :80 listener either — nothing an HTTP-01 challenge could use.
    servers = mgr._bootstrap_config()["apps"]["http"]["servers"]
    assert all(":80" not in listen for s in servers.values() for listen in s["listen"])


async def test_bootstrap_disables_admin_config_persistence(queries):
    """S-W7 / F5: Caddy must not autosave a config carrying bcrypt material."""
    mgr = _manager()
    assert mgr._bootstrap_config()["admin"] == {
        "listen": mgr._settings.admin_addr,
        "config": {"persist": False},
    }


async def test_tls_latch_skips_admin_io_when_converged(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await mgr._converge_tls([])
    assert mgr._tls_synced is True
    fake.requests.clear()
    await mgr._converge_tls([])
    assert not any(p == "/config/apps/tls" for _, p in fake.requests)

    # An unlatched manager re-READS before deciding, and writes nothing when the
    # live subtree already matches.
    mgr._tls_synced = False
    fake.requests.clear()
    await mgr._converge_tls([])
    assert ("GET", "/config/apps/tls") in fake.requests
    assert ("POST", "/config/apps/tls") not in fake.requests
    assert mgr._tls_synced is True


async def test_tls_desired_app_dedupes_domains_against_the_subjects(queries):
    mgr = _manager()
    app = mgr._tls_desired_app(["z.example", "box", "a.example", "a.example"])
    # Node names keep their order and lead; domains are sorted and deduped
    # against them, so the subtree is stable input for the hash.
    assert app["certificates"]["automate"] == ["box", "a.example", "z.example"]
    assert app["automation"]["policies"][0]["subjects"] == ["box"]


async def test_load_domain_names_is_total_over_a_stubless_queries():
    """No queries, a stub without the method, and a raising one all read []."""

    class _Boom:
        async def list_service_domains(self):
            raise RuntimeError("db down")

    mgr = _manager()  # queries is None
    assert await mgr._load_domain_names() == []
    mgr._queries = object()  # a stub lacking the method
    assert await mgr._load_domain_names() == []
    mgr._queries = _Boom()
    assert await mgr._load_domain_names() == []


# -- 10e. register / deregister / status --------------------------------------


async def test_register_is_default_only_unless_with_domains(queries):
    """S-W3: the launch path stays one route write; the domain lags one tick."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    port = (await queries.get_service_endpoint("a")).host_port

    await mgr.register("a", port)

    assert _route_ids(fake) == ["nerdit-route-a"]


async def test_register_with_domains_prepends_the_whole_set(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    port = (await queries.get_service_endpoint("a")).host_port

    await mgr.register("a", port, with_domains=True)

    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a"]
    items, _ = await queries.list_audit_log(action="proxy.route_registered", limit=10)
    assert {tuple(sorted(i.params_redacted.items())) for i in items} == {
        (("service_name", "a"),),
        (("domain", "a.example.com"), ("service_name", "a")),
    }
    # Idempotent: a converged re-register writes nothing.
    fake.requests.clear()
    await mgr.register("a", port, with_domains=True)
    assert _route_writes(fake) == []


async def test_register_with_domains_withholds_them_when_tls_is_unreadable(queries):
    """The cutover commit carries the same D-P26-4 gate reconcile does."""
    fake = FakeCaddy(tls_status=503)
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    port = (await queries.get_service_endpoint("a")).host_port

    await mgr.register("a", port, with_domains=True)

    assert _route_ids(fake) == ["nerdit-route-a"], "default only; domains follow a later tick"


async def test_live_domain_route_ids_track_what_caddy_actually_serves(queries):
    """(Codex round 1, P2 #3831777097) The fact ``views/hosted.py`` derives a
    domain's ``ready`` from. It must be EMPTY before the tick that writes the
    Host route — the row exists from the PUT, the route does not — and it must
    name exactly the ids Caddy is serving afterwards."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")

    assert mgr.live_domain_route_ids == frozenset(), "bound, not yet routed"

    await mgr.reconcile()
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-a@a.example.com"})

    # A second domain bound between ticks is not ready until its tick runs.
    await _seed_domain(queries, a, "b.example.com")
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-a@a.example.com"})
    await mgr.reconcile()
    assert mgr.live_domain_route_ids == {
        "nerdit-route-a@a.example.com",
        "nerdit-route-a@b.example.com",
    }


async def test_a_drifted_domain_route_whose_patch_failed_is_not_live(queries):
    """(Codex round 2, #3835632983) PRESENT is not CONVERGED.

    The snapshot used to be "desired AND present in Caddy", so a route whose
    corrective PATCH raised stayed in it: the container respawned on a new port,
    the upsert 503'd, and Caddy kept dialling the DEAD port while the domains
    API and the dashboard's Share card called the name ``ready`` and offered it
    as the app's URL. The verdict must be Caddy-factual — same dial, same
    matcher shape, same auth handler — not a presence check."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-a@a.example.com"})

    # The service respawns on a new port; only the DOMAIN route's PATCH fails,
    # so the default route converges and the tick is a partial success.
    await queries.set_endpoint_active_port("a", 51234)
    fake.fail_ids = {"nerdit-route-a@a.example.com"}
    await mgr.reconcile()

    live = await mgr._admin.live_routes()
    assert live["nerdit-route-a"].dial == "127.0.0.1:51234", "the default converged"
    assert live["nerdit-route-a@a.example.com"].dial == "127.0.0.1:9400", "stale, still present"
    assert mgr.live_domain_route_ids == frozenset(), "present, but not what we asked for"

    # And it comes back the moment the PATCH lands.
    fake.fail_ids = set()
    await mgr.reconcile()
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-a@a.example.com"})


async def test_a_domain_route_serving_obsolete_credentials_is_not_live(queries, hash_calls):
    """The auth dimension of the same bug: the route dials the right port and
    matches the right Host, but its ``authentication`` handler is the one from
    before the secret was rotated. A presence check cannot see that — the
    fingerprint comparison can, and it is the same one ``needs_upsert`` uses."""
    fake = FakeCaddy()
    secrets = FakeSecrets({"alpha": PW})
    mgr = _protected_proxy(queries, fake, secrets)
    job = await _seed_protected(queries, "alpha")
    await _seed_domain(queries, job, "a.example.com")
    await mgr.reconcile()
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-alpha@a.example.com"})

    secrets.values["alpha"] = "rotated-pw"
    fake.fail_ids = {"nerdit-route-alpha@a.example.com"}
    await mgr.reconcile()

    live = await mgr._admin.live_routes()
    assert "nerdit-route-alpha@a.example.com" in live, "still present, with the OLD handler"
    assert mgr.live_domain_route_ids == frozenset()

    fake.fail_ids = set()
    await mgr.reconcile()
    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-alpha@a.example.com"})


async def test_a_failed_reanchor_drops_the_route_from_the_live_snapshot(queries):
    """The ordering repair deletes then re-inserts. If the insert fails the
    route is GONE — and the snapshot was computed BEFORE that loop ran, so it
    still named the id. Computing it after, minus the ids the repair could not
    put back, is what makes ``ready`` mean "answers right now"."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()
    assert _route_ids(fake) == ["nerdit-route-a@a.example.com", "nerdit-route-a"]

    # Shove the Host route BEHIND the path route (what an out-of-band append or
    # an adopted Caddy leaves), and make the re-insert fail. ``fail_ids`` gates
    # the PUT-at-index leg; the DELETE that precedes it still lands.
    fake.routes.reverse()
    fake.fail_ids = {"nerdit-route-a@a.example.com"}
    await mgr.reconcile()

    assert _route_ids(fake) == ["nerdit-route-a"], "deleted, never re-inserted"
    assert mgr.live_domain_route_ids == frozenset()


async def test_a_tls_withheld_tick_reports_no_live_domain_routes(queries):
    """The condition the old three-fact reading could not see at all: the proxy
    is available and the app is routed, but every domain spec is withheld
    because the TLS subtree will not converge — indefinitely, on an adopted
    pre-WP1 Caddy. The surface must say ``withheld``, not feature the URL."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()
    assert mgr.live_domain_route_ids

    fake.tls_status = 503
    mgr._tls_synced = False
    await mgr.reconcile()

    assert mgr.available is True, "the default route is still served"
    assert mgr.live_domain_route_ids == frozenset()


async def test_register_with_domains_widens_the_live_snapshot(queries):
    """The cutover fast-path is a writer too: what it lands is live now, not one
    tick later."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    port = (await queries.get_service_endpoint("a")).host_port

    await mgr.register("a", port, with_domains=True)

    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-a@a.example.com"})


async def test_deregister_narrows_the_live_snapshot_to_the_other_services(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    b = await _seed_running(queries, "b")
    await _seed_domain(queries, a, "a.example.com")
    await _seed_domain(queries, b, "b.example.com")
    await mgr.reconcile()

    await mgr.deregister("a")

    assert mgr.live_domain_route_ids == frozenset({"nerdit-route-b@b.example.com"})


async def test_deregister_removes_every_route_of_the_service(queries):
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_running(queries, "b")
    await _seed_domain(queries, a, "a1.example.com")
    await _seed_domain(queries, a, "a2.example.com")
    await mgr.reconcile()

    await mgr.deregister("a")

    assert _route_ids(fake) == ["nerdit-route-b"], "b's route is untouched"
    items, _ = await queries.list_audit_log(action="proxy.route_deregistered", limit=10)
    assert {i.params_redacted.get("domain") for i in items} == {
        None,
        "a1.example.com",
        "a2.example.com",
    }


async def test_deregister_with_an_unreadable_live_table_removes_the_default_only(queries):
    """The prune loop is the backstop — never guess at ids we cannot see."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "a.example.com")
    await mgr.reconcile()
    fake.routes_status = 503

    await mgr.deregister("a")

    fake.routes_status = 200
    assert _route_ids(fake) == ["nerdit-route-a@a.example.com"]
    # The prune loop is the backstop: once the service leaves the desired set,
    # the next tick removes what deregister could not see.
    await queries.update_job_status(a.id, JobStatus.stopped)
    await mgr.reconcile()
    assert _route_ids(fake) == []


async def test_deregister_never_touches_a_prefix_neighbour(queries):
    """``nerdit-route-ab`` must not be swept up by ``deregister('a')``."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)
    await _seed_running(queries, "a")
    await _seed_running(queries, "ab")
    await mgr.reconcile()

    await mgr.deregister("a")

    assert _route_ids(fake) == ["nerdit-route-ab"]


async def test_status_snapshot_domains_live_tristate(queries):
    fake = FakeCaddy()
    mgr = _snap_manager(fake)
    mgr._available = True
    fake.routes.append({"@id": "nerdit-route-a@a.example.com", "handle": []})
    rows = [_domain_row("a", "a.example.com"), _domain_row("b", "b.example.com")]

    snap = await mgr.status_snapshot(domains=rows)

    assert snap["domains"] == [
        {
            "domain": "a.example.com",
            "service_name": "a",
            "acme": False,
            "live": True,
            "cert_state": "internal",
        },
        {
            "domain": "b.example.com",
            "service_name": "b",
            "acme": False,
            "live": False,
            "cert_state": "internal",
        },
    ]
    # Unreadable live table → ``None``, never ``False`` (tri-state discipline).
    fake.routes_status = 503
    snap = await mgr.status_snapshot(domains=rows)
    assert [d["live"] for d in snap["domains"]] == [None, None]
    # Proxy off → the same, and no admin read is attempted at all.
    mgr._available = False
    snap = await mgr.status_snapshot(domains=rows)
    assert snap["routes"]["live_table"] == "disabled"
    assert [d["live"] for d in snap["domains"]] == [None, None]
    # Default: no rows passed, no key surprise for pre-WP1 callers.
    assert (await mgr.status_snapshot())["domains"] == []


# --- 20. P26 WP2: [proxy.acme] — the :80 server, the acme policy, cert_state --
#
# The negative contract comes first and stays first: with the block at its
# defaults every byte below is WP1's. Everything after it is what turning the
# block on adds, and each addition is asserted on the JSON actually handed to
# Caddy, never on a manager method calling itself.


def _partition_rows(*specs):
    return [_domain_row(svc, dom, acme=acme) for svc, dom, acme in specs]


async def test_partition_domains_is_the_one_place_the_rule_lives():
    rows = _partition_rows(
        ("a", "b.example.com", True),
        ("a", "a.example.com", False),
        ("b", "c.example.com", True),
    )

    names, acme = partition_domains(rows, acme_enabled=True)
    # Both halves sorted (the digest is order-sensitive) and the acme half a
    # subset of the whole.
    assert names == ["a.example.com", "b.example.com", "c.example.com"]
    assert acme == ["b.example.com", "c.example.com"]
    assert set(acme) <= set(names)

    # The block off: the rows keep their names, nobody gets a public issuer.
    names_off, acme_off = partition_domains(rows, acme_enabled=False)
    assert names_off == names
    assert acme_off == []

    # Duplicates (two services can never own one domain, but the partition is
    # pure and must not depend on that) collapse in both halves.
    dupes = _partition_rows(("a", "x.example.com", True), ("b", "x.example.com", True))
    assert partition_domains(dupes, acme_enabled=True) == (["x.example.com"], ["x.example.com"])
    assert partition_domains([], acme_enabled=True) == ([], [])


async def test_bootstrap_with_acme_off_is_byte_identical_to_wp1():
    """The WP2 negative contract, pinned against a literal — not a helper call."""
    mgr = _manager()

    assert mgr._bootstrap_config() == {
        "admin": {"listen": "localhost:2019", "config": {"persist": False}},
        "storage": {"module": "file_system", "root": "/tmp/nerdit-test/caddy"},
        "apps": {
            "pki": {"certificate_authorities": {"local": {"install_trust": False}}},
            "http": {
                "https_port": 443,
                "servers": {
                    "nerdit": {
                        "listen": [":443"],
                        "routes": [],
                        "automatic_https": {"disable_redirects": True},
                    }
                },
            },
            "tls": _tls_app(["box"]),
        },
    }
    # No HTTP role at all: no ``http_port`` key, one server, nothing on :80.
    assert "http_port" not in mgr._bootstrap_config()["apps"]["http"]


async def test_bootstrap_with_acme_on_adds_the_http_port_and_the_solver_server():
    mgr = _manager(acme=_acme_settings())

    http = mgr._bootstrap_config()["apps"]["http"]

    # (a) the HTTP role is declared — this is what makes the listener the one
    # Caddy answers ``/.well-known/acme-challenge/`` on.
    assert http["http_port"] == _ACME_HTTP_PORT
    assert http["https_port"] == 443
    assert set(http["servers"]) == {"nerdit", "nerdit-acme-http"}
    # (b) the WP1 server is untouched by the flip.
    assert http["servers"]["nerdit"] == {
        "listen": [":443"],
        "routes": [],
        "automatic_https": {"disable_redirects": True},
    }
    # (c) the solver server, whole and exact. No route for the challenge itself
    # (Caddy answers it ahead of the route table), a Host-less 308, and the
    # explicit 404 LAST — a Caddy server whose routes match nothing answers 200
    # with an empty body, so "else 404" has to be written down.
    assert http["servers"]["nerdit-acme-http"] == {
        "listen": [f":{_ACME_HTTP_PORT}"],
        "routes": [
            {
                "handle": [
                    {
                        "handler": "static_response",
                        "status_code": 308,
                        "headers": {"Location": ["https://{http.request.host}{http.request.uri}"]},
                    }
                ],
                "terminal": True,
            },
            {"handle": [{"handler": "static_response", "status_code": 404}]},
        ],
        "automatic_https": {"disable_redirects": True},
    }


async def test_the_acme_listener_reaches_no_upstream():
    """The one new public listener has no route to anything (S-W2-15)."""
    mgr = _manager(acme=_acme_settings())

    server = mgr._bootstrap_config()["apps"]["http"]["servers"]["nerdit-acme-http"]

    handlers = [h for route in server["routes"] for h in route["handle"]]
    assert handlers, "the server must carry routes, or it answers 200 to everything"
    assert all(h["handler"] == "static_response" for h in handlers)
    assert "reverse_proxy" not in json.dumps(server)
    assert "upstreams" not in json.dumps(server)
    # …and it is the LAST route that has no matcher, i.e. the catch-all 404.
    assert "match" not in server["routes"][-1]
    assert server["routes"][-1]["handle"][0]["status_code"] == 404


async def test_the_redirect_names_the_advertised_port_when_it_is_not_443():
    """``{http.request.host}`` strips the port, so a non-443 node must add it."""
    default = _manager(acme=_acme_settings())._acme_http_server()
    assert default["routes"][0]["handle"][0]["headers"]["Location"] == [
        "https://{http.request.host}{http.request.uri}"
    ]

    odd = _manager(acme=_acme_settings(), https_port=8443)._acme_http_server()
    assert odd["routes"][0]["handle"][0]["headers"]["Location"] == [
        "https://{http.request.host}:8443{http.request.uri}"
    ]

    # An external proxy fronting us: the ADVERTISED port wins over the bound
    # one, exactly like ``public_url_for``.
    fronted = _manager(acme=_acme_settings(), https_port=8443, public_port=443)
    assert fronted._acme_http_server()["routes"][0]["handle"][0]["headers"]["Location"] == [
        "https://{http.request.host}{http.request.uri}"
    ]
    fronted_odd = _manager(acme=_acme_settings(), https_port=8443, public_port=9443)
    assert fronted_odd._acme_http_server()["routes"][0]["handle"][0]["headers"]["Location"] == [
        "https://{http.request.host}:9443{http.request.uri}"
    ]


async def test_the_acme_listener_is_404_only_without_the_redirect():
    mgr = _manager(acme=_acme_settings(http_redirect=False))

    server = mgr._bootstrap_config()["apps"]["http"]["servers"]["nerdit-acme-http"]

    # One route, the 404 — the solver still works (it is not a route), so the
    # listener keeps its whole reason to exist while answering nothing else.
    assert server["routes"] == [{"handle": [{"handler": "static_response", "status_code": 404}]}]
    assert "308" not in json.dumps(server)


async def test_the_acme_policy_leads_and_the_catch_all_still_trails(queries):
    """The enabled sibling of ``test_no_acme_issuer_anywhere``."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, acme=_acme_settings())
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "public.example.com", acme=True)
    await _seed_domain(queries, a, "internal.example.com")

    await mgr.reconcile()

    assert fake.tls == _tls_app(
        ["box"], ["internal.example.com", "public.example.com"], ["public.example.com"]
    )
    policies = fake.tls["automation"]["policies"]
    # Order is the contract: the public name is claimed FIRST, the node's own
    # names next, and the subject-less catch-all is still last — so a name in
    # neither list still gets an internal leaf, never a public issuance.
    assert policies[0]["issuers"][0]["module"] == "acme"
    assert policies[0]["subjects"] == ["public.example.com"]
    # (review round 2) EXACTLY one issuer. An internal fallback behind the ACME
    # one was built and measured: certmagic falls through on a cancelled attempt,
    # our own route pushes cancel it, and the public certificate then never
    # arrives. The pinned shape is the single issuer; the cost — a pending name
    # serves no leaf — is what the user-facing surfaces now state.
    assert len(policies[0]["issuers"]) == 1
    assert policies[1]["subjects"] == ["box"]
    assert "subjects" not in policies[-1]
    assert policies[-1]["issuers"] == [{"module": "internal"}]
    # The ACME name still rides ``automate`` like any other: WP2 changes HOW a
    # name is issued, never whether it is managed.
    assert "public.example.com" in fake.tls["certificates"]["automate"]
    # And the internal name is claimed by nothing but the catch-all.
    assert "internal.example.com" not in json.dumps(policies[:2])


async def test_trusted_roots_is_present_only_with_a_ca_root_file(queries):
    """A private CA is opt-in; the key is absent otherwise, never null/empty."""
    plain = _manager(acme=_acme_settings())._tls_desired_app(["p.example.com"], ["p.example.com"])
    issuer = plain["automation"]["policies"][0]["issuers"][0]
    assert "trusted_roots_pem_files" not in issuer
    assert issuer == _acme_issuer()

    private = _manager(acme=_acme_settings(ca_root_file="/etc/nerdit/pebble.pem"))
    app = private._tls_desired_app(["p.example.com"], ["p.example.com"])
    # A list of PATHS — the only shape Caddy 2.11.4 accepts (there is no
    # inline-PEM variant on this module).
    assert app["automation"]["policies"][0]["issuers"][0] == _acme_issuer(
        ca_root_file="/etc/nerdit/pebble.pem"
    )


async def test_an_acme_row_under_a_disabled_block_is_an_ordinary_internal_name(queries):
    """A node that turned ACME off keeps serving its domains — internally."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake)  # [proxy.acme] at its default: off
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "public.example.com", acme=True)

    await mgr.reconcile()

    # Byte-identical to the WP1 subtree for the same row set…
    assert fake.tls == _tls_app(["box"], ["public.example.com"])
    assert "acme" not in json.dumps(fake.tls).lower()
    # …and the Host route is live, not withheld: the discrepancy is reported as
    # a cert_state, never as an outage.
    assert _route_ids(fake)[0] == "nerdit-route-a@public.example.com"
    assert mgr.cert_status(_domain_row("a", "public.example.com", acme=True)).state == "disabled"


async def test_flipping_a_rows_acme_pushes_the_subtree_once_then_settles(queries):
    """The digest carries the flag; no second latch, no per-tick rewrite."""
    fake = FakeCaddy()
    mgr = _proxy(queries, fake, acme=_acme_settings())
    a = await _seed_running(queries, "a")
    await _seed_domain(queries, a, "app.example.com")
    await mgr.reconcile()
    assert fake.tls == _tls_app(["box"], ["app.example.com"])
    before = mgr._tls_hash

    # The re-PUT that flips the column (the route layer's accepted re-PUT).
    outcome, row = await queries.add_service_domain("a", "app.example.com", acme=True, job_id=a.id)
    assert (outcome, row.acme) == ("exists", True)

    fake.requests.clear()
    await mgr.reconcile()

    assert fake.tls == _tls_app(["box"], ["app.example.com"], ["app.example.com"])
    assert mgr._tls_hash != before
    assert [p for m, p in fake.requests if m == "POST" and p == "/config/apps/tls"] == [
        "/config/apps/tls"
    ], "exactly one whole-subtree push"
    # The route itself did not move — only its issuer did.
    assert _route_writes(fake) == []

    # And the tick after is write-free, TLS read included (the hash latch).
    fake.requests.clear()
    await mgr.reconcile()
    assert {m for m, _ in fake.requests} == {"GET"}
    assert not any(p == "/config/apps/tls" for _, p in fake.requests)


async def test_status_snapshot_projects_the_acme_block(queries):
    fake = FakeCaddy()

    off = _snap_manager(fake)
    off._available = True
    # Disabled: the three configuration values are ABSENT (null), not defaults,
    # and ``listening`` is null rather than a synthesized False.
    assert (await off.status_snapshot())["acme"] == {
        "enabled": False,
        "directory": None,
        "http_port": None,
        "http_redirect": None,
        "listening": None,
    }

    on = ProxyManager(
        None,
        ProxySettings(enabled=True, acme=_acme_settings()),
        hostname="box",
        data_dir="/tmp/nerdit-test",
        admin=_admin(fake),
    )
    on._binary = "/usr/bin/caddy"
    on._available = True
    # Enabled and up, with the listener latched by the spawn/adopt path: Caddy
    # refuses to start when a listener fails to bind, so a live
    # ``nerdit-acme-http`` server IS the bind fact for the HTTP-01 port.
    on._acme_listener_live = True
    assert (await on.status_snapshot())["acme"] == {
        "enabled": True,
        "directory": _ACME_DIRECTORY,
        "http_port": _ACME_HTTP_PORT,
        "http_redirect": True,
        "listening": True,
    }
    # (review round 1) Up, but the running Caddy carries no ACME listener — an
    # adopted process from a pre-ACME run. Availability is NOT the bind fact
    # there, and the projection must say so rather than infer ``true``.
    on._acme_listener_live = False
    assert (await on.status_snapshot())["acme"]["listening"] is False
    # Unread latch fails closed too — never an invented ``true``.
    on._acme_listener_live = None
    assert (await on.status_snapshot())["acme"]["listening"] is False
    # Enabled and down: False, not null — enablement is known, the bind is not
    # happening.
    on._acme_listener_live = True
    on._available = False
    assert (await on.status_snapshot())["acme"]["listening"] is False
    # The account email never reaches this body.
    assert _ACME_EMAIL not in json.dumps(await on.status_snapshot())


def _leaf_pem(domain: str, *, days: int) -> bytes:
    """A self-signed leaf for *domain*, valid (or expired) by *days*."""
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID

    key = ed25519.Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domain)])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=30))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(domain)]), critical=False)
        # Ed25519 carries its own hash — the algorithm argument must be None.
        .sign(key, None)
    )
    return cert.public_bytes(Encoding.PEM)


async def test_cert_status_reads_the_four_states_off_storage(tmp_path):
    domain = "app.example.com"
    mgr = ProxyManager(
        None,
        ProxySettings(enabled=True, acme=_acme_settings()),
        hostname="box",
        data_dir=tmp_path,
    )

    # (i) an ordinary row: this node's own CA, which is the default and not a
    # defect — and it costs no file read at all.
    assert mgr.cert_status(_domain_row("a", domain)) == CertStatus("internal")

    row = _domain_row("a", domain, acme=True)
    # (ii) requested but nothing on disk yet. ``pending`` is honest about
    # covering "failing" too — Caddy records no failure in storage.
    assert mgr.cert_status(row) == CertStatus("pending")

    # (iii) a real leaf at the DERIVED path — the storage key is part of the
    # contract, so writing it anywhere else must keep reading ``pending``.
    path = acme_cert_path(tmp_path / "caddy", _ACME_DIRECTORY, domain)
    assert path == (
        tmp_path / "caddy" / "certificates" / "127.0.0.1-14000-dir" / domain / f"{domain}.crt"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(_leaf_pem(domain, days=30))
    status = mgr.cert_status(row)
    assert status.state == "issued"
    assert status.not_after is not None

    # (iv) the same file past its notAfter.
    path.write_bytes(_leaf_pem(domain, days=-1))
    assert mgr.cert_status(row).state == "expired"

    # (v) the block turned off after the fact: the row's request is what is
    # disabled, and the answer never depends on the file still being there.
    off = ProxyManager(None, ProxySettings(enabled=True), hostname="box", data_dir=tmp_path)
    assert off.cert_status(row) == CertStatus("disabled")


async def test_cert_states_keys_a_batch_by_domain(tmp_path):
    mgr = ProxyManager(
        None,
        ProxySettings(enabled=True, acme=_acme_settings()),
        hostname="box",
        data_dir=tmp_path,
    )
    rows = _partition_rows(("a", "a.example.com", True), ("b", "b.example.com", False))

    states = mgr.cert_states(rows)

    assert states == {
        "a.example.com": CertStatus("pending"),
        "b.example.com": CertStatus("internal"),
    }
    assert mgr.cert_states([]) == {}
