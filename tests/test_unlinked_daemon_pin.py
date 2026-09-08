"""Keep all local daemon functionality available without a cloud link.

Link-disabled daemons must still deploy, proxy, reconcile, serve models/databases
and expose every route indefinitely. Cloud outages may cost remote access only.

Compare linked/unlinked operation tables, exercise representative routes with no
link manager, and scan for entitlement calls outside the sole advisory check.
Doctor's link result may be warn or skipped but must never cause failure.
Use real routers/auth/error middleware with self-contained mocked state.
"""

from __future__ import annotations

import ast
import re
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

import nerdit.daemon.server as server_module
from nerdit.config.settings import LinkSettings, NerditSettings
from nerdit.core.proxy.domains import ReservedNames
from nerdit.core.proxy.manager import ProxyState
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.daemon.routes.domains import router as domains_router
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.routes.system import router as system_router
from nerdit.db.models import Job, JobKind, JobStatus, ServiceEndpoint

_ADMIN = {"Authorization": "Bearer admin-raw-token"}

#: The repository's ``src/nerdit`` tree, resolved from this file so the source
#: pin below works from any working directory.
_SRC = Path(__file__).resolve().parent.parent / "src" / "nerdit"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _real_db_dir() -> str:
    """A temp dir holding a real, integrity-clean sqlite file.

    The doctor's ``db`` probe opens its own read-only connection against the
    file on disk rather than the mocked shared one, so it needs a real database
    to ``quick_check``. Built once per module, like the neighbouring suite's.
    """
    d = tempfile.mkdtemp(prefix="nerdit-unlinked-pin-db-")
    conn = sqlite3.connect(str(Path(d) / "nerdit.db"))
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    return d


_DB_DIR = _real_db_dir()


def _key_file() -> Path:
    """A real secrets key file — the ``secrets_key`` probe classifies from
    ``lstat`` mode bits, which a stub answering ``is_file()`` cannot satisfy."""
    path = Path(tempfile.mkdtemp(prefix="nerdit-unlinked-pin-key-")) / "secrets.key"
    path.write_text("ab" * 32 + "\n", encoding="utf-8")
    return path


_KEY_PATH = _key_file()


def _unlinked_settings() -> SimpleNamespace:
    """Settings for a node that has **never** been linked.

    ``LinkSettings()`` is used unmodified rather than a hand-rolled namespace
    precisely so this pin tracks the real defaults: ``enabled = False`` and
    ``node_id = None`` are the two facts the whole test turns on, and if a
    future release changed either default this harness would start lying.
    """
    link = LinkSettings()
    assert link.enabled is False
    assert link.node_id is None
    return SimpleNamespace(
        data_dir=_DB_DIR,
        link=link,
        proxy=SimpleNamespace(
            enabled=True,
            mode="path",
            base_domain=None,
            hostname_override=None,
            scheme="https",
            https_port=443,
            public_port=None,
            dashboard_apex=True,
            mdns=False,
            mdns_address=None,
            admin_addr="localhost:2019",
            extra_hostnames=[],
        ),
        models=SimpleNamespace(default_backend="ollama", bridge_host="172.17.0.1"),
        git=SimpleNamespace(enabled=False, allowed_hosts=["github.com"]),
        daemon=SimpleNamespace(max_upload_bytes=524288000),
        services=SimpleNamespace(max_concurrent_builds=2, service_port_range="8100-8199"),
        containers=SimpleNamespace(
            drop_all_caps=True, no_new_privileges=True, read_only_rootfs=False
        ),
    )


#: One deployed, running, routed app. The node under test is not a bare install
#: — it is a box doing real work with no cloud account attached, which is the
#: configuration D-ENT-2 is actually about.
_APP = Job(
    name="demo",
    kind=JobKind.service,
    service_name="demo",
    gpu_count=0,
    status=JobStatus.running,
    desired_state="running",
    restart_policy="always",
    config='{"image": "nerdit-app/demo:1", "port": 8000}',
)
_ENDPOINT = ServiceEndpoint(
    service_name="demo", job_id=_APP.id, container_port=8000, host_port=9400, route="/demo"
)


def _queries() -> AsyncMock:
    """A queries stand-in holding one deployed app and no exposure rows.

    The legacy ``admin-raw-token`` short-circuits to the LEGACY_ADMIN principal
    without ever reaching the token lookup, so the token methods only need to
    exist. ``get_job`` must answer a real ``None`` — an ``AsyncMock``'s default
    return is a truthy ``MagicMock``, which would resolve every ident and hide a
    404 — and the two exposure readers answer empty: this node has no hosted
    share and no direct domains, the true state of a fresh unlinked box.
    """
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(return_value=None)
    q.get_api_token_by_id = AsyncMock(return_value=None)
    q.touch_api_token = AsyncMock()
    q.insert_audit_log = AsyncMock()
    q.list_services = AsyncMock(return_value=([_APP], None))
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(
        side_effect=lambda n: _APP if n == _APP.service_name else None
    )
    q.get_service_endpoint = AsyncMock(
        side_effect=lambda n: _ENDPOINT if n == _APP.service_name else None
    )
    q.get_job_gpus = AsyncMock(return_value=[])
    q.list_service_shares = AsyncMock(return_value={})
    q.list_service_domains = AsyncMock(return_value=[])
    return q


class _Proxy:
    """``app.state.proxy_manager`` for a node whose proxy is up and healthy.

    The proxy being *available* while the link is *off* is the combination that
    matters: it is what "proxies" in the invariant means, and it is the state
    of every LAN-only Nerdit box in the field.
    """

    state = ProxyState.available
    available = True
    withheld_services: frozenset[str] = frozenset()
    live_domain_route_ids: frozenset[str] = frozenset()

    def cert_states(self, rows: object) -> dict[str, object]:
        return {}

    async def status_snapshot(self, *, domains: object = ()) -> dict[str, object]:
        """The shape ``GET /proxy/status`` projects — a converged, empty node.

        Field set mirrors ``tests/test_domains_route.py::_snapshot``; the route
        composes ``mdns`` itself and defaults ``domains`` when absent.
        """
        return {
            "state": "available",
            "enabled": True,
            "available": True,
            "mode": "path",
            "base_domain": None,
            "hostname": "box",
            "scheme": "https",
            "https_port": 443,
            "tls": {},
            "ca": {},
            "apex": {},
            "respawn": {},
            "routes": {},
        }

    def validate_capability(self, token: str, role: str) -> bool:
        return False


def _unlinked_app() -> TestClient:
    """The daemon's read + deploy surface, built for a never-linked node.

    Note what is *absent*: ``app.state.link_manager``. Not a stub reporting
    "disconnected" — absent entirely, which is what the boot seam leaves behind
    when ``[link].enabled`` is false. Every surface below therefore takes its
    no-link branch for real.
    """
    q = _queries()
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    api.include_router(services_router)
    api.include_router(deploy_router)
    api.include_router(proxy_router)
    api.include_router(domains_router)
    app.include_router(api)

    settings = _unlinked_settings()
    app.state.settings = settings
    app.state.queries = q
    app.state.started_at = datetime.now(UTC) - timedelta(seconds=12)
    app.state.hostname = "box"
    app.state.proxy_manager = _Proxy()
    app.state.mdns_advertiser = None
    app.state.domain_reserved = ReservedNames(names=frozenset({"box"}))
    app.state.shared_scope_blocked = False
    app.state.gpu_snapshot = {"count": 1, "schedulable": 1, "vendors": ["nvidia"]}
    app.state.model_controller = SimpleNamespace(
        backends=["ollama", "vllm"],
        default_backend_name="ollama",
        bridge_host="172.17.0.1",
    )
    app.state.data_controller = SimpleNamespace(
        backends=["postgres", "redis"],
        default_backend_name="postgres",
        bridge_host="172.17.0.1",
    )
    app.state.runtime = SimpleNamespace(
        list_images=AsyncMock(return_value=[]),
        buildx_available=AsyncMock(return_value="present"),
    )
    app.state.secret_manager = SimpleNamespace(
        key_path=_KEY_PATH,
        has_ciphertexts=lambda: True,
        self_check=lambda: True,
    )
    db_cursor = SimpleNamespace(fetchone=AsyncMock(return_value=("ok",)))
    app.state.db = SimpleNamespace(conn=SimpleNamespace(execute=AsyncMock(return_value=db_cursor)))

    app.add_middleware(ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: q)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _check(body: dict, name: str) -> dict | None:
    return next((c for c in body["checks"] if c["name"] == name), None)


#: Any of these appearing in a REFUSAL body would mean the daemon declined for a
#: link/entitlement/licence reason — the one class of refusal D-ENT-2 forbids on
#: a local surface. Matched against the whole serialized body so a nested
#: ``detail`` or ``hint`` cannot hide one.
#:
#: Scanned on refusals only, never on a 200. A successful body may legitimately
#: *describe* account state — ``/capabilities`` carries both a ``link`` and a
#: ``license`` block, and reporting a fact is the opposite of gating on it. What
#: is forbidden is a refusal that cites one.
_FORBIDDEN_REFUSAL_MARKERS = (
    "entitlement",
    "not_entitled",
    "subscription",
    "link_required",
    "license",
    "licence",
    "unlicensed",
    "upgrade",
)


def _assert_no_account_reason(response: object, where: str) -> None:
    """Fail if a refusal cites an account, entitlement or licence reason."""
    body = response.text  # type: ignore[attr-defined]
    lowered = body.lower()
    for marker in _FORBIDDEN_REFUSAL_MARKERS:
        assert marker not in lowered, (
            f"{where} refused citing {marker!r} on an unlinked node — a local "
            f"surface may never gate on account state (D-ENT-2). Body: {body}"
        )


def _assert_served(response: object, where: str) -> None:
    """Assert *response* was served, and that any refusal was not account-shaped.

    Two failures, deliberately ordered: if the surface 4xx'd we first say
    *whether it was a gate* — the specific wrong this programme exists to
    prevent — and only then that it refused at all.
    """
    status = response.status_code  # type: ignore[attr-defined]
    if status >= 400:
        _assert_no_account_reason(response, where)
    assert status < 400, (
        f"{where} refused with {status} on an unlinked node: {response.text}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------


def test_unlinked_daemon_serves_everything():
    """A daemon with ``[link].enabled = false`` and no ``node_id`` serves, deploys,
    proxies and reconciles — the D-ENT-2 / D-X16-O15 / D-X16-57 invariant, as a
    test rather than a sentence (P34 D5).

    If a future change adds a runtime entitlement gate anywhere on these paths,
    this goes red. Deleting it to make a gate green is the thing a reviewer is
    supposed to notice.
    """
    client = _unlinked_app()

    # --- it reads: self-knowledge, and it is honest about the link being off.
    caps = client.get("/api/capabilities", headers=_ADMIN)
    _assert_served(caps, "GET /api/capabilities")
    body = caps.json()
    # The link block reports the truth without any of it being a gate: the
    # proxy grammar, the buildpacks, the model backends and the deploy limits
    # are all still there for an agent to plan against.
    assert body.get("link", {}).get("enabled") is False
    assert body["proxy"]["enabled"] is True
    assert body["models"]["backends"]

    # --- it self-diagnoses, and the link row is advisory, never a gate.
    doctor = client.get("/api/doctor", headers=_ADMIN)
    _assert_served(doctor, "GET /api/doctor")
    doc = doctor.json()
    link_row = _check(doc, "link")
    assert link_row is not None, "the doctor lost its link row"
    # The ceiling, not the exact word (P34 D4 owns whether this is warn or
    # skipped): doctor is advisory and never a gate, so being unlinked may never
    # read as a failure. The top-level verdict is deliberately NOT asserted here
    # — it is worst-of over thirteen rows, several of which (disk headroom,
    # Docker) depend on the machine the suite runs on, and pinning it would make
    # this pin fail for reasons that have nothing to do with the link.
    assert link_row["status"] in {"ok", "warn", "skipped"}, link_row
    # (D-ENT-3 / D-X16-O21) And the same ceiling for P17d, which ships dormant:
    # a daemon with no licence installed is a normal daemon, not a broken one.
    license_row = _check(doc, "license")
    if license_row is not None:
        assert license_row["status"] in {"ok", "warn", "skipped"}, license_row
    # (D-X16-O11) And the detail leaks nothing on the way past: no URL, no
    # filesystem path, no code and no key material — the standing rule for every
    # row on this surface, which a new never-linked branch must not be the first
    # to break.
    detail = link_row["detail"]
    assert "http" not in detail.lower(), detail
    assert _DB_DIR not in detail, detail
    assert str(_KEY_PATH) not in detail, detail

    # --- it serves services, with the exposure projection taking its no-link
    #     branch for real (no link_manager on app.state at all).
    services = client.get("/api/services", headers=_ADMIN)
    _assert_served(services, "GET /api/services")
    listed = services.json()["items"]
    assert [s["name"] for s in listed] == ["demo"]
    # The app's LAN URL is still composed and still handed out — the local
    # exposure path never consults the account.
    assert listed[0]["endpoint"]["public_url"] == "https://box/demo/"

    # --- it proxies: the proxy status surface answers with the proxy up while
    #     the link is off, which is the state of every LAN-only node.
    status = client.get("/api/proxy/status", headers=_ADMIN)
    _assert_served(status, "GET /api/proxy/status")
    assert status.json()["available"] is True

    # --- it serves direct domains: the node-served exposure path is explicitly
    #     cloud-free (docs/guide/exposure.md §7), so an unlinked node lists them
    #     like any other.
    domains = client.get("/api/services/demo/domains", headers=_ADMIN)
    _assert_served(domains, "GET /api/services/demo/domains")
    assert domains.json() == {"service_name": "demo", "domains": []}

    # --- it deploys: the ingress is reached and refuses only for CONTENT
    #     reasons. A bodyless POST is a 422 validation error from FastAPI —
    #     proof the route was served and evaluated, not withheld. A link gate
    #     would answer 402/403/409 before the body was ever parsed.
    deploy = client.post("/api/deploy", headers=_ADMIN)
    _assert_no_account_reason(deploy, "POST /api/deploy")
    assert deploy.status_code == 422, deploy.text


def test_no_api_route_is_registered_behind_the_link_being_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    """The ``/api`` operation table is identical with the link off and on.

    A route that only exists on a linked node would be a gate by another name —
    the agent surface would silently shrink for a free user, and the REST
    operation count would depend on account state. ``create_app``
    reads ``load_settings`` as a module global, so both builds are driven by
    monkeypatching it (the ``tests/test_create_app.py`` idiom); neither touches
    the operator's real ``~/.nerdit/config.toml``.
    """

    def _ops(settings: NerditSettings) -> set[str]:
        monkeypatch.setattr(server_module, "load_settings", lambda: settings)
        schema = server_module.create_app().openapi()
        return {
            op["operationId"]
            for methods in schema["paths"].values()
            for op in methods.values()
            if isinstance(op, dict) and "operationId" in op
        }

    off = NerditSettings()
    assert off.link.enabled is False and off.link.node_id is None

    on = NerditSettings()
    on.link.enabled = True
    on.link.node_id = "00000000-0000-4000-8000-00000000000a"
    on.link.slug = "corvid-mesa"

    assert _ops(off) == _ops(on)


def test_the_daemon_has_exactly_one_entitlement_call_site_and_it_stays_advisory():
    """``require_entitlement`` is called once in ``src/nerdit``, and it only warns.

    This is the assertion that reaches where the request sweep above cannot: a
    gate added to a reconcile tick, the builder, a model backend, the proxy
    manager or a route nobody thought to dial would be invisible to a behaviour
    test and obvious here. P17d's D-LIC2 made the seam advisory by decision
    (the relay is the authoritative online judge, so a second offline judge
    could only ever disagree with it), and D-X16-O15 / D-X16-57 froze that: one
    call site, still advisory, for the whole X16 programme.

    ``core/license.py`` itself is excluded — it *defines* the method, and P17d
    stays shipped-but-dormant (D-X16-O21 / D-ENT-3): this pin guards against a
    second **caller**, never against the feature existing.
    """
    call_sites: list[tuple[Path, int]] = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.relative_to(_SRC).as_posix() == "core/license.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "require_entitlement" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "require_entitlement"
            ):
                call_sites.append((path, node.lineno))

    assert len(call_sites) == 1, (
        "the daemon must have exactly ONE entitlement-shaped call site "
        f"(D-X16-O15 / D-X16-57); found {[(str(p), n) for p, n in call_sites]}"
    )
    path, lineno = call_sites[0]
    assert path.relative_to(_SRC).as_posix() == "daemon/server.py", path

    # Advisory means: the decision is consumed by a log line, and nothing in
    # the block it guards raises, exits, or refuses. Read the enclosing
    # ``if decision.reason is not None:`` block and prove it contains a warning
    # and no control-flow escape.
    lines = path.read_text(encoding="utf-8").splitlines()
    window = "\n".join(lines[lineno - 1 : lineno + 12])
    assert "logger.warning" in window, window
    assert not re.search(r"\braise\b|\bsys\.exit\b|HTTPException|NerditError", window), (
        "the entitlement seam must stay advisory — it may warn, never refuse "
        f"(D-LIC2, D-X16-O15). Found in:\n{window}"
    )
