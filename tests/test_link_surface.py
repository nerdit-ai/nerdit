"""Test tunnel authorization, provenance, status and disabled-link behavior.

Authorize tunnel capabilities before local-mode shortcuts, which otherwise grant
admin and violate the submitter ceiling. Use link:<node_id> as the audit principal.
Capabilities and doctor read synchronous status only, never expose capability
tokens, and restrict relay_host to admins. Disabled links have no manager or
outbound dial, report enabled=false and skip the link check.
Use real auth/audit/error middleware with mocked application state.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient

from nerdit.config.settings import DaemonSettings, LinkSettings
from nerdit.config.store import _RESTART_KEYS
from nerdit.core.eventlog import EVENT_TYPES
from nerdit.core.link.identity import NodeIdentity
from nerdit.core.link.manager import (
    DISCONNECT_REASONS,
    GITHUB_TOKEN_WARN_LEAD_S,
    GithubInstallationSummary,
    LinkStatus,
)
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import current_principal, hash_token
from nerdit.daemon.bootstrap import _loopback_base_url, build_link_manager
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.system import router as system_router
from nerdit.db.models import ApiToken, TokenRole
from tests.link_fake_relay import CLOSE_UNAUTHORIZED, Scenario
from tests.test_link_manager import link_harness

NODE_ID = "00000000-0000-4000-8000-00000000000a"
CAPABILITY = "live-capability-token-not-a-secret"  # noqa: S105 - a test literal
LINK_PRINCIPAL_ID = f"link:{NODE_ID}"

_ADMIN = {"Authorization": "Bearer admin-raw-token"}
_TUNNEL = {"Authorization": f"Bearer {CAPABILITY}"}


# ---------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------


def _status(**kw: object) -> LinkStatus:
    base: dict[str, object] = {
        "state": "connected",
        "node_id": NODE_ID,
        "relay_host": "relay.example.test",
        "connected_at": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        "capability_expires_at": datetime.now(UTC) + timedelta(seconds=300),
        "last_close_code": None,
        "last_error_code": None,
        "terminal_reason": None,
        "retry_at": None,
        "attempt": 0,
        "auth_failures": 0,
        "active_streams": 2,
    }
    base.update(kw)
    return LinkStatus(**base)  # type: ignore[arg-type]


class FakeLinkManager:
    """What ``app.state.link_manager`` looks like to the daemon surfaces.

    Only the two things the surfaces touch: a constant-time-ish validator and a
    synchronous status snapshot.
    """

    def __init__(
        self,
        *,
        token: str | None = CAPABILITY,
        status: LinkStatus | None = None,
        held: tuple[int, datetime | None] | None = None,
    ):
        self._token = token
        self._status = status or _status()
        # (P33 doctor D2) The raw-held mirror the doctor's overdue latch reads.
        # Defaults to the live count implied by the status snapshot with no
        # expired stragglers; a test wanting the had-one-now-expired branch
        # passes ``held=(count, past_expiry)`` alongside an empty
        # ``github_installations``.
        self._held = held
        self.validations: list[tuple[str, str]] = []

    def validate_capability(self, token: str, role: str) -> bool:
        self.validations.append((token, role))
        return self._token is not None and token == self._token and role == "submitter"

    def status(self) -> LinkStatus:
        return self._status

    def github_mirror_held(self) -> tuple[int, datetime | None]:
        if self._held is not None:
            return self._held
        live = self._status.github_installations
        if not live:
            return 0, None
        return len(live), max(inst.expires_at for inst in live)

    def now(self) -> datetime:
        return datetime.now(UTC)


def _queries(token_row: ApiToken | None = None) -> AsyncMock:
    queries = AsyncMock()
    queries.get_api_token_by_hash = AsyncMock(return_value=token_row)
    queries.insert_audit_log = AsyncMock()
    queries.touch_api_token = AsyncMock()
    return queries


def _scoped(role: TokenRole = TokenRole.submitter, raw: str = "scoped-raw") -> ApiToken:
    return ApiToken(
        id="tok-1",
        name="ci-bot",
        role=role,
        token_hash=hash_token(raw),
        max_gpus=1,
        max_concurrent_jobs=1,
    )


def _app(
    *,
    token: str | None = "admin-raw-token",
    link_manager: object | None = None,
    queries: AsyncMock | None = None,
    audit: bool = False,
) -> tuple[TestClient, AsyncMock]:
    app = FastAPI()
    register_error_handlers(app)
    resolved = queries if queries is not None else _queries()

    @app.get("/whoami")
    def whoami(request: Request) -> dict:
        principal = current_principal(request)
        return {
            "name": principal.name,
            "role": principal.role.value,
            "token_id": principal.token_id,
        }

    @app.post("/services")
    def mutate(request: Request) -> dict:
        principal = current_principal(request)
        return {"token_id": principal.token_id, "role": principal.role.value}

    app.state.link_manager = link_manager
    if audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: resolved)
    app.add_middleware(ScopedTokenAuthMiddleware, token=token, get_queries=lambda: resolved)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False), resolved


# ---------------------------------------------------------------------------
# middleware: the tunnel principal
# ---------------------------------------------------------------------------


def test_a_live_capability_resolves_to_the_node_link_submitter_principal() -> None:
    manager = FakeLinkManager()
    client, queries = _app(link_manager=manager)

    body = client.get("/whoami", headers=_TUNNEL).json()

    assert body == {"name": "node-link", "role": "submitter", "token_id": LINK_PRINCIPAL_ID}
    assert manager.validations == [(CAPABILITY, "submitter")]
    # There is no ``api_tokens`` row to stamp, and none was looked for.
    queries.get_api_token_by_hash.assert_not_awaited()
    queries.touch_api_token.assert_not_awaited()


def test_the_tunnel_principal_may_mutate_but_is_only_a_submitter() -> None:
    """D-R2 as amended by ADR-W1: the submitter ceiling is permanent."""
    client, _ = _app(link_manager=FakeLinkManager())
    body = client.post("/services", headers=_TUNNEL).json()
    assert body == {"token_id": LINK_PRINCIPAL_ID, "role": "submitter"}
    assert body["role"] != "admin"


def test_local_mode_does_not_promote_a_tunnelled_request() -> None:
    """The ordering regression, stated as directly as it can be.

    With ``token=None`` the next branch in ``dispatch`` grants LOCAL — an
    **admin** principal. If the capability branch were moved below it, a request
    that arrived over the relay would become a local admin and the tunnel's role
    ceiling would be gone. Both halves are asserted so the test cannot pass by
    the branch simply never running.
    """
    client, _ = _app(token=None, link_manager=FakeLinkManager())

    tunnelled = client.get("/whoami", headers=_TUNNEL).json()
    assert tunnelled["name"] == "node-link"
    assert tunnelled["role"] == "submitter"
    assert tunnelled["token_id"] == LINK_PRINCIPAL_ID

    # A genuinely local caller is still the local admin — nothing regressed.
    local = client.get("/whoami").json()
    assert local["name"] == "local"
    assert local["role"] == "admin"


def test_the_legacy_admin_token_is_unaffected_by_the_link_branch() -> None:
    client, _ = _app(link_manager=FakeLinkManager())
    body = client.get("/whoami", headers=_ADMIN).json()
    assert body["name"] == "legacy-admin"
    assert body["role"] == "admin"


def test_a_scoped_token_still_resolves_normally() -> None:
    queries = _queries(_scoped())
    client, _ = _app(link_manager=FakeLinkManager(), queries=queries)
    body = client.get("/whoami", headers={"Authorization": "Bearer scoped-raw"}).json()
    assert body["token_id"] == "tok-1"
    assert body["name"] == "ci-bot"


def test_a_bearer_that_is_not_the_capability_falls_through_to_a_denial() -> None:
    """``None`` from the link branch is a fall-through, never a denial of its own."""
    manager = FakeLinkManager()
    client, _ = _app(link_manager=manager)
    response = client.get("/whoami", headers={"Authorization": "Bearer forged-or-stale"})
    assert response.status_code == 403
    assert response.json()["code"] == "invalid_token"
    assert manager.validations == [("forged-or-stale", "submitter")]


def test_a_stale_capability_is_refused_once_the_link_is_down() -> None:
    """The manager answers ``False`` between sockets; the daemon must too."""
    client, _ = _app(link_manager=FakeLinkManager(token=None))
    assert client.get("/whoami", headers=_TUNNEL).status_code == 403


def test_a_non_bearer_authorization_header_never_reaches_the_validator() -> None:
    manager = FakeLinkManager()
    client, _ = _app(link_manager=manager)
    response = client.get("/whoami", headers={"Authorization": f"Basic {CAPABILITY}"})
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_auth_format"
    assert manager.validations == []


def test_local_mode_refuses_a_bearer_that_is_not_the_live_capability() -> None:
    """The tunnel→LOCAL-admin fall-through, closed.

    The window is real, not theoretical: the mux validates the injected
    capability at ``open_stream``, and this middleware re-checks it up to
    ``REQUEST_BODY_TIMEOUT_S`` (~30 s) later once a chunked body has been
    assembled. A capability that lapses in between arrives here as a Bearer
    that failed ``_link_principal`` — and on a tokenless daemon the next branch
    would hand it the LOCAL **admin** principal, which is the exact ceiling
    breach the branch ordering exists to prevent.
    """
    manager = FakeLinkManager(token=None)  # the capability has lapsed
    client, _ = _app(token=None, link_manager=manager)

    response = client.get("/whoami", headers=_TUNNEL)

    assert response.status_code == 403
    assert response.json()["code"] == "invalid_token"
    assert manager.validations == [(CAPABILITY, "submitter")]


def test_local_mode_refuses_a_mutation_carrying_a_stale_capability() -> None:
    """The same, on the method that actually costs something."""
    client, _ = _app(token=None, link_manager=FakeLinkManager(token="a-different-capability"))
    response = client.post("/services", headers=_TUNNEL)
    assert response.status_code == 403
    assert "admin" not in response.text


def test_local_mode_still_serves_an_unauthenticated_caller_as_the_local_admin() -> None:
    """Negative control: the denial is about the *bearer*, not about the link.

    A genuinely local caller sends no Authorization header at all and is
    unaffected — closing the fall-through must not turn a link-enabled daemon
    into one that refuses its own operator.
    """
    client, _ = _app(token=None, link_manager=FakeLinkManager(token=None))
    body = client.get("/whoami").json()
    assert (body["name"], body["role"]) == ("local", "admin")


def test_local_mode_leaves_public_paths_reachable_with_a_stale_bearer() -> None:
    """A public path grants no authority, so it is exempt.

    ``/api/proxy/ca`` is the trust bootstrap and ``/api/health`` is what every
    monitor polls; refusing them because a caller still carries an old token
    would break liveness checking for no security gain.
    """
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    app.state.link_manager = FakeLinkManager(token=None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: _queries())
    app.add_middleware(RequestIdMiddleware)

    client = TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False)
    assert client.get("/api/health", headers=_TUNNEL).status_code == 200


def test_no_link_manager_means_no_link_branch() -> None:
    """``absent`` and ``None`` must mean the same inert thing."""
    for manager in (None,):
        client, _ = _app(link_manager=manager)
        assert client.get("/whoami", headers=_TUNNEL).status_code == 403
    client, _ = _app(token=None, link_manager=None)
    assert client.get("/whoami", headers=_TUNNEL).json()["name"] == "local"


def test_an_unauthenticated_request_is_unaffected() -> None:
    client, _ = _app(link_manager=FakeLinkManager())
    response = client.get("/whoami")
    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


# ---------------------------------------------------------------------------
# audit provenance (checklist item 3, "cheap, do not defer")
# ---------------------------------------------------------------------------


def test_a_tunnelled_mutation_is_audited_as_the_node() -> None:
    """The whole provenance mechanism, with zero per-route work.

    One principal, stamped at the single auth choke point, so every mutation
    that ever arrives through the relay is attributable to the node.
    """
    manager = FakeLinkManager()
    client, queries = _app(link_manager=manager, audit=True)

    assert client.post("/services", headers=_TUNNEL).status_code == 200

    queries.insert_audit_log.assert_awaited()
    row = queries.insert_audit_log.await_args.kwargs
    assert row["principal_id"] == LINK_PRINCIPAL_ID
    assert row["principal_role"] == "submitter"
    assert row["result"] == "ok"
    # The capability itself is nowhere in the row.
    assert CAPABILITY not in str(row)


def test_a_local_mutation_is_not_attributed_to_the_node() -> None:
    """Negative control: the provenance is the tunnel's, not everyone's."""
    client, queries = _app(link_manager=FakeLinkManager(), audit=True)
    assert client.post("/services", headers=_ADMIN).status_code == 200
    row = queries.insert_audit_log.await_args.kwargs
    assert row["principal_id"] != LINK_PRINCIPAL_ID
    assert row["principal_role"] == "admin"


# ---------------------------------------------------------------------------
# GET /capabilities — the folded link block
# ---------------------------------------------------------------------------


def _caps_settings(
    link_enabled: bool = True,
    *,
    slug: str | None = "gpu-box",
    nodes_base_domain: str | None = "nodes.test",
    acme_enabled: bool = False,
    node_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_dir="/home/u/.nerdit",
        proxy=SimpleNamespace(
            enabled=False,
            mode="path",
            base_domain=None,
            hostname_override=None,
            scheme="https",
            https_port=443,
            public_port=None,
            dashboard_apex=False,
            mdns=False,
            admin_addr="localhost:2019",
            # (P26 WP2) The nested [proxy.acme] section, which ``exposure.acme``
            # projects. Only ``enabled`` is read by the capabilities route —
            # neither the email nor the directory is ever projected.
            acme=SimpleNamespace(enabled=acme_enabled),
        ),
        models=SimpleNamespace(default_backend="ollama", bridge_host="172.17.0.1"),
        databases=SimpleNamespace(default_backend="postgres"),
        git=SimpleNamespace(enabled=False, allowed_hosts=[]),
        daemon=SimpleNamespace(max_upload_bytes=1),
        services=SimpleNamespace(max_concurrent_builds=1, service_port_range="8100-8199"),
        link=SimpleNamespace(
            enabled=link_enabled,
            # (P26 WP-H) The two halves of a hosted name. Present on the
            # settings namespace for every capabilities test; individual cases
            # override them to exercise the unlinked / un-refreshed shapes.
            slug=slug,
            nodes_base_domain=nodes_base_domain,
            # (P34 D4) The claim's persisted node identity. ``None`` — the
            # default — is a node that has NEVER linked, which is exactly the
            # state the doctor ``link`` check now separates from a deliberate
            # opt-out; a case that wants the opt-out passes a value.
            node_id=node_id,
        ),
    )


def _surface_app(
    *,
    link_manager: object | None,
    role: TokenRole | None = None,
    settings: SimpleNamespace | None = None,
) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.settings = settings or _caps_settings()
    app.state.started_at = datetime.now(UTC) - timedelta(seconds=5)
    app.state.hostname = "nerd-box.local"
    app.state.link_manager = link_manager
    app.state.gpu_snapshot = {"count": 0, "schedulable": 0, "vendors": []}
    app.state.shared_scope_blocked = False
    row = (
        SimpleNamespace(
            id="tok-1",
            name="ci",
            role=role,
            max_gpus=1,
            max_concurrent_jobs=1,
            expires_at=None,
            scope_services=None,
        )
        if role is not None
        else None
    )
    queries = SimpleNamespace(
        get_api_token_by_hash=AsyncMock(return_value=row),
        touch_api_token=AsyncMock(),
        insert_audit_log=AsyncMock(),
    )
    app.add_middleware(
        ScopedTokenAuthMiddleware, token="admin-raw-token", get_queries=lambda: queries
    )
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app)


def test_capabilities_projects_the_full_link_block_for_an_admin() -> None:
    manager = FakeLinkManager()
    client = _surface_app(link_manager=manager)
    block = client.get("/api/capabilities", headers=_ADMIN).json()["link"]

    assert block == {
        "enabled": True,
        "state": "connected",
        "node_id": NODE_ID,
        "slug": "gpu-box",
        "nodes_base_domain": "nodes.test",
        "hosted_public_entitled": False,
        # (P32) Never asserted on this stand-in status, so the age is null —
        # the key is present regardless, so a reader can tell "the cloud says
        # no" from "the cloud has not spoken".
        "hosted_public_entitled_at": None,
        # (P33) No installation token mirrored on the stand-in; the key is
        # present regardless so ``nerdit link`` can print ``github = none``.
        "github_installations": [],
        "connected_at": "2026-01-01T12:00:00+00:00",
        "capability_expires_at": manager.status().capability_expires_at.isoformat(),
        "last_close_code": None,
        "terminal_reason": None,
        "active_streams": 2,
        "relay_host": "relay.example.test",
    }


def test_capabilities_dates_the_entitlement_mirror_for_every_role() -> None:
    """(P32) ``false`` alone is ambiguous between "the plan says no" and "the
    cloud has gone quiet"; the timestamp is what separates them, and it is a
    timestamp, not a secret, so every authenticated role sees it."""
    asserted_at = datetime(2026, 2, 3, 9, 30, tzinfo=UTC)
    manager = FakeLinkManager(
        status=_status(hosted_public_entitled=True, hosted_public_entitled_at=asserted_at)
    )
    client = _surface_app(link_manager=manager, role=TokenRole.submitter)

    block = client.get("/api/capabilities", headers={"Authorization": "Bearer scoped"}).json()[
        "link"
    ]

    assert block["hosted_public_entitled"] is True
    assert block["hosted_public_entitled_at"] == asserted_at.isoformat()


def _installation(expires_in_s: float, *, installation_id: int = 11, repos_count: int = 3):
    return GithubInstallationSummary(
        installation_id=installation_id,
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_s),
        repos_count=repos_count,
    )


def test_capabilities_projects_the_github_installations_as_summaries() -> None:
    """(P33 D-GH-7) Id, expiry, repo COUNT — for every role, because none of
    it is a secret; the token and the repo names are nowhere in the payload."""
    manager = FakeLinkManager(
        status=_status(github_installations=(_installation(2400, repos_count=3),))
    )
    client = _surface_app(link_manager=manager, role=TokenRole.submitter)

    response = client.get("/api/capabilities", headers={"Authorization": "Bearer scoped"})
    block = response.json()["link"]

    assert len(block["github_installations"]) == 1
    inst = block["github_installations"][0]
    assert set(inst) == {"installation_id", "expires_at", "repos_count"}
    assert inst["installation_id"] == 11
    assert inst["repos_count"] == 3
    assert "repos" not in inst
    assert CAPABILITY not in response.text
    assert "ghs_" not in response.text


def test_capabilities_lists_no_installations_by_default() -> None:
    block = (
        _surface_app(link_manager=FakeLinkManager())
        .get("/api/capabilities", headers=_ADMIN)
        .json()["link"]
    )
    assert block["github_installations"] == []


def test_the_relay_host_is_admin_only() -> None:
    """The ``proxy.admin_addr``/``paths`` precedent: omitted, not nulled."""
    client = _surface_app(link_manager=FakeLinkManager(), role=TokenRole.submitter)
    block = client.get("/api/capabilities", headers={"Authorization": "Bearer scoped"}).json()[
        "link"
    ]
    assert "relay_host" not in block
    # The rest is not a secret and stays visible to every authenticated role.
    assert block["state"] == "connected"
    assert block["node_id"] == NODE_ID


def test_the_capability_token_is_nowhere_in_the_capabilities_payload() -> None:
    client = _surface_app(link_manager=FakeLinkManager())
    body = client.get("/api/capabilities", headers=_ADMIN).text
    assert CAPABILITY not in body
    # Nor the relay URL's path, nor any filesystem path of the link.
    assert "wss://" not in body
    assert "node.key" not in body


def test_capabilities_says_disabled_when_there_is_no_manager() -> None:
    """No manager means no TUNNEL state — and says so without hiding enrolment.

    The block was ``{"enabled": False}`` alone until P34 widened it: collapsing
    everything made a node that never linked indistinguishable from one linked
    and then disabled, which is a real distinction consumers need (see
    ``test_capabilities_separates_a_never_linked_node_from_a_deliberate_opt_out``).
    What stays absent is everything that belongs to a manager that does not
    exist — state, timestamps, stream counts, the entitlement mirror.
    """
    client = _surface_app(link_manager=None)
    block = client.get("/api/capabilities", headers=_ADMIN).json()["link"]
    assert block == {"enabled": False, "node_id": None, "slug": "gpu-box"}
    assert not {"state", "connected_at", "active_streams", "hosted_public_entitled"} & set(block)


# ---------------------------------------------------------------------------
# GET /capabilities — the P26 WP-H exposure block
# ---------------------------------------------------------------------------


def _exposure(**kw: object) -> dict:
    client = _surface_app(**kw)  # type: ignore[arg-type]
    return client.get("/api/capabilities", headers=_ADMIN).json()["exposure"]


def test_exposure_reports_the_hosted_url_grammar_on_a_ready_node() -> None:
    """A linked node that knows its domain advertises the SHAPE, not a promise.

    ``hosted_url_shape`` is what an agent reads to learn the grammar before it
    shares anything, so it must be the real composition (``<app>--<slug>``, one
    label, https, root-relative) rather than a hand-written example.
    """
    block = _exposure(link_manager=FakeLinkManager())

    assert block == {
        "hosted": True,
        "hosted_public": False,
        # (P26 WP1) A plain capability flag, not a live conjunction like
        # ``hosted``: a custom domain is served by this node's own Caddy and
        # needs no link, so nothing at runtime can make the operation
        # unavailable. (P26 WP2) ``acme`` is now a second such flag — it
        # mirrors ``[proxy.acme].enabled``, which the harness leaves off by
        # default; the two tests below pin that it follows the setting.
        "domains": True,
        "acme": False,
        "nodes_base_domain": "nodes.test",
        "hosted_url_shape": "https://<name>--gpu-box.nodes.test/",
    }


def test_exposure_is_not_hosted_without_a_tunnel() -> None:
    """No manager ⇒ no hosted path, even with both name halves configured."""
    block = _exposure(link_manager=None)

    assert block["hosted"] is False
    assert block["hosted_url_shape"] is None
    # The configured suffix is still reported: it is what the operator set, and
    # hiding it would make "why is hosted false?" unanswerable from this read.
    assert block["nodes_base_domain"] == "nodes.test"


def test_exposure_is_not_hosted_before_the_domain_is_learned() -> None:
    """A node linked before P26 has a slug but no domain until it refreshes."""
    block = _exposure(
        link_manager=FakeLinkManager(),
        settings=_caps_settings(nodes_base_domain=None),
    )

    assert block["hosted"] is False
    assert block["hosted_url_shape"] is None
    assert block["nodes_base_domain"] is None


def test_exposure_is_not_hosted_on_an_unlinked_node() -> None:
    """No slug ⇒ nothing to compose the second half of the label from."""
    block = _exposure(link_manager=FakeLinkManager(), settings=_caps_settings(slug=None))

    assert block["hosted"] is False
    assert block["hosted_url_shape"] is None


def test_hosted_public_is_false_with_the_default_link_status() -> None:
    """S5: nothing in this WP sets the entitlement, so public is refused.

    Pinned against the REAL :class:`LinkStatus` default rather than a stub, so
    the day WP-HC starts feeding the flag this test is the thing that notices.
    """
    assert _status().hosted_public_entitled is False
    assert _exposure(link_manager=FakeLinkManager())["hosted_public"] is False


def test_hosted_public_follows_the_link_status_seam() -> None:
    """The one boolean WP-HC will flip reaches the surface unchanged."""
    manager = FakeLinkManager(status=_status(hosted_public_entitled=True))

    assert _exposure(link_manager=manager)["hosted_public"] is True
    # ``hosted`` stays independent: entitlement says whether PUBLIC is allowed,
    # not whether the node can serve a hosted URL at all.
    assert _exposure(link_manager=manager)["hosted"] is True


def test_the_link_block_carries_the_two_hosted_name_halves() -> None:
    """An agent must not have to read /config to learn how to address a share."""
    block = (
        _surface_app(link_manager=FakeLinkManager())
        .get("/api/capabilities", headers=_ADMIN)
        .json()["link"]
    )

    assert block["slug"] == "gpu-box"
    assert block["nodes_base_domain"] == "nodes.test"
    assert block["hosted_public_entitled"] is False


def test_exposure_acme_follows_the_proxy_acme_setting() -> None:
    """(P26 WP2) ``acme`` answers "will ``PUT {acme: true}`` be accepted here?".

    That is precisely the 409 ``domain.acme_disabled`` gate the domains route
    applies, so the flag and the refusal read the same setting. It is a plain
    capability flag like ``domains`` — no live conjunction — because
    ``[proxy.acme].enabled`` already implies ``[proxy].enabled`` (refused at
    load otherwise), and whether a leaf has actually been issued is a
    per-domain fact (``cert_state``), not a capability.
    """
    assert _exposure(link_manager=None, settings=_caps_settings(acme_enabled=True))["acme"] is True
    assert (
        _exposure(link_manager=None, settings=_caps_settings(acme_enabled=False))["acme"] is False
    )


def test_exposure_acme_is_false_on_a_settings_object_without_the_section() -> None:
    """A lightweight settings namespace predating [proxy.acme] must not 500.

    The ``[link]``/``[mcp]`` tolerance precedent: capabilities is a pure
    projection that every authenticated role reads, so a missing nested section
    means "off", never an exception.
    """
    settings = _caps_settings()
    del settings.proxy.acme

    assert _exposure(link_manager=None, settings=settings)["acme"] is False


def test_exposure_never_carries_the_acme_email_or_directory() -> None:
    """The block names capabilities, not configuration (S-W2-10).

    ``exposure`` is visible to every authenticated role, so the ACME account
    email (PII-adjacent) and the directory URL (which CA, staging or prod) stay
    out of it — ``nerdit config show`` is where configuration comes back.
    """
    settings = _caps_settings(acme_enabled=True)
    settings.proxy.acme = SimpleNamespace(
        enabled=True,
        email="ops@example.test",
        directory="https://acme-staging-v02.api.letsencrypt.org/directory",
    )
    body = _surface_app(link_manager=None, settings=settings).get(
        "/api/capabilities", headers=_ADMIN
    )

    assert body.json()["exposure"]["acme"] is True
    assert "ops@example.test" not in body.text
    assert "acme-staging-v02" not in body.text


def test_exposure_is_visible_to_every_authenticated_role() -> None:
    """Capabilities and a public DNS suffix are not admin-only facts."""
    client = _surface_app(link_manager=FakeLinkManager(), role=TokenRole.readonly)
    block = client.get("/api/capabilities", headers={"Authorization": "Bearer scoped"}).json()[
        "exposure"
    ]

    assert block["hosted"] is True
    assert block["hosted_url_shape"] == "https://<name>--gpu-box.nodes.test/"


def test_a_terminal_link_is_visible_in_capabilities() -> None:
    manager = FakeLinkManager(
        status=_status(
            state="terminal",
            terminal_reason="revoked",
            last_close_code=4403,
            connected_at=None,
            capability_expires_at=None,
            active_streams=0,
        )
    )
    block = (
        _surface_app(link_manager=manager).get("/api/capabilities", headers=_ADMIN).json()["link"]
    )
    assert block["state"] == "terminal"
    assert block["terminal_reason"] == "revoked"
    assert block["last_close_code"] == 4403
    assert block["connected_at"] is None
    assert block["capability_expires_at"] is None


def test_a_terminal_entitlement_link_is_machine_readable_in_capabilities() -> None:
    """(P27 WP-C4) The machine contract for the plan's ``link.entitlement_required``.

    No such literal exists anywhere, and deliberately so: the state is already
    machine-shaped as the pair below (plus ``CloseClass.TERMINAL_ENTITLEMENT``
    on the wire and the durable ``link.disconnected`` reason), so a fourth
    spelling would be a second vocabulary for one state. A machine consumer
    reads these fields; the doctor detail's hint is for humans.
    """
    manager = FakeLinkManager(
        status=_status(
            state="terminal",
            terminal_reason="entitlement_required",
            last_close_code=4401,
            connected_at=None,
            capability_expires_at=None,
            active_streams=0,
        )
    )
    block = (
        _surface_app(link_manager=manager).get("/api/capabilities", headers=_ADMIN).json()["link"]
    )
    assert block["state"] == "terminal"
    assert block["terminal_reason"] == "entitlement_required"
    assert block["last_close_code"] == 4401
    assert block["terminal_reason"] in DISCONNECT_REASONS
    # Never a free-text hint on the machine surface.
    assert "hint" not in block
    assert "subscription" not in str(block)


# ---------------------------------------------------------------------------
# GET /doctor — the eleventh check
# ---------------------------------------------------------------------------


def _doctor_link(client: TestClient) -> dict:
    body = client.get("/api/doctor", headers=_ADMIN).json()
    checks = {check["name"]: check for check in body["checks"]}
    assert "link" in checks, sorted(checks)
    return checks["link"]


def test_a_node_that_has_never_linked_warns_instead_of_skipping_the_link_check() -> None:
    """(P34 D4) ``[link].enabled = false`` with no ``node_id`` is an anomaly.

    Once ``install.sh`` drives the link, a daemon carrying no Nerdit account at
    all is the state worth surfacing — so the row that used to read ``skipped``
    for every disabled node now warns for the never-linked half of them. The
    detail names the missing thing and the one command that supplies it.
    """
    client = _surface_app(link_manager=None, settings=_caps_settings(link_enabled=False))
    check = _doctor_link(client)

    assert check["status"] == "warn"
    assert check["detail"] == (
        "not linked — no Nerdit account attached; run 'nerdit link --device' (free)"
    )


def test_the_never_linked_warn_neither_gates_nor_implies_a_degraded_local_plane() -> None:
    """(P34 D4, D-ENT-2 / D-X16-O11) The two properties the copy must keep.

    ``warn`` and never ``fail``: doctor is advisory, never a gate, and an
    unlinked daemon serves every route forever — so the row must not describe
    local function as broken, and must not push the report to ``fail`` on a
    node whose only "problem" is having no cloud account. And like every other
    detail on this surface it carries no URL, no code and no key material: a
    doctor body is read by every role and lands in transcripts.
    """
    client = _surface_app(link_manager=None, settings=_caps_settings(link_enabled=False))
    check = _doctor_link(client)
    detail = check["detail"]

    # The row itself, not the worst-of top status: this harness deliberately
    # runs without a container runtime, so the report's top is dominated by the
    # docker check and would say nothing about the link row's severity.
    assert check["status"] != "fail"
    assert "/" not in detail  # no URL, no path — the shipped discipline
    assert CAPABILITY not in client.get("/api/doctor", headers=_ADMIN).text
    for degraded in ("degraded", "broken", "disabled", "offline"):
        assert degraded not in detail.lower()


def test_a_node_linked_once_and_then_disabled_keeps_the_quiet_skipped_link_check() -> None:
    """(P34 D4, OD-P34-5) The opt-out stays silent — and IS the escape hatch.

    ``[link].node_id`` survives ``enabled = false`` (only ``nerdit unlink``
    clears it), so a node the owner deliberately unplugged from the cloud — an
    air-gapped box, a machine moved off the account — reads ``skipped`` exactly
    as it did before D4. That is why v1 ships no suppression knob for the warn
    above: linking once is the supported way to silence it forever.
    """
    client = _surface_app(
        link_manager=None,
        settings=_caps_settings(link_enabled=False, node_id="a1b2c3d4e5f6"),
    )
    check = _doctor_link(client)

    # ``skipped`` is ranked below ``ok`` by ``_STATUS_RANK``, so it can never
    # worsen the worst-of top status — which is the whole point of keeping the
    # deliberate opt-out on this branch rather than warning about it forever.
    assert check["status"] == "skipped"
    assert check["detail"] == "link disabled"


@pytest.mark.parametrize("node_id", [None, "a1b2c3d4e5f6"])
def test_the_enabled_but_unlinked_link_row_is_untouched_by_the_never_linked_split(
    node_id: str | None,
) -> None:
    """(P34 D4) The ``enabled = true`` branch reads no ``node_id`` at all.

    D4 splits only the disabled branch. An enabled node with no live manager is
    either mid-claim or has an unreadable identity key, which is the P27 state
    with its own copy — and it must render identically whether or not a claim
    has already persisted ``node_id``, since the manager's absence is the fact
    being reported.
    """
    client = _surface_app(
        link_manager=None, settings=_caps_settings(link_enabled=True, node_id=node_id)
    )
    check = _doctor_link(client)

    assert check["status"] == "warn"
    assert check["detail"] == "link enabled but not linked or identity unavailable"


def test_doctor_warns_when_enabled_but_not_linked() -> None:
    """The ordinary state between flipping the flag and running the claim."""
    check = _doctor_link(_surface_app(link_manager=None))
    assert check["status"] == "warn"
    assert "not linked" in check["detail"]


def test_doctor_reports_a_connected_link_with_its_renewal_lead() -> None:
    manager = FakeLinkManager(
        status=_status(capability_expires_at=datetime.now(UTC) + timedelta(seconds=300))
    )
    check = _doctor_link(_surface_app(link_manager=manager))
    assert check["status"] == "ok"
    assert check["detail"].startswith("connected; capability renews in ")
    seconds = int(check["detail"].split(" in ")[1].removesuffix("s"))
    assert 290 <= seconds <= 300


@pytest.mark.parametrize("state", ["connecting", "backoff", "displaced"])
def test_doctor_warns_while_reconnecting(state: str) -> None:
    manager = FakeLinkManager(status=_status(state=state, attempt=3, last_close_code=1013))
    check = _doctor_link(_surface_app(link_manager=manager))
    assert check["status"] == "warn"
    assert "attempt 3" in check["detail"]
    assert "1013" in check["detail"]


def test_doctor_fails_on_a_terminal_link_and_names_the_reason() -> None:
    manager = FakeLinkManager(status=_status(state="terminal", terminal_reason="auth_failed"))
    check = _doctor_link(_surface_app(link_manager=manager))
    assert check["status"] == "fail"
    assert check["detail"] == "terminal: auth_failed"


def test_doctor_hints_the_entitlement_fix_on_a_terminal_entitlement_link() -> None:
    """(P27 WP-C4, recopy P34 D-X16-23) A refusal told with its real remedy.

    A refused entitlement is the terminal state an operator can actually fix,
    and the fix is not "wait" — the link task exits on terminal, so only a
    daemon restart re-dials. The detail therefore keeps its machine-parsable
    ``terminal: <reason>`` prefix (so the shipped parsers and the pinned
    ``auth_failed`` detail are untouched) and appends the next step.

    Post-collapse the relay admits on account standing alone, so the hint must
    name standing and not a subscription: this asserts the money wording is
    gone, because a hint that sends a suspended customer to the billing page
    is worse than the bare reason it replaced.
    """
    manager = FakeLinkManager(
        status=_status(
            state="terminal",
            terminal_reason="entitlement_required",
            last_close_code=4401,
            connected_at=None,
            capability_expires_at=None,
            active_streams=0,
        )
    )
    check = _doctor_link(_surface_app(link_manager=manager))

    assert check["status"] == "fail"
    assert check["detail"].startswith("terminal: entitlement_required")
    assert "not active" in check["detail"]
    assert "subscription" not in check["detail"]
    assert "restart the daemon" in check["detail"]
    # The URL/path/secret discipline of every other doctor detail, extended to
    # the hinted state: the remedy is named in words, never as an origin a
    # reader could be walked to.
    assert "/" not in check["detail"]
    assert CAPABILITY not in check["detail"]


def test_doctor_hints_the_relink_command_on_a_revoked_link() -> None:
    """Tell revoked nodes to unlink before starting device linking.

    Cloud revocation, including dormancy sweeps, leaves local node_id intact.
    Relinking first therefore returns already_linked; unlink must clear that
    state. Include --device because bare nerdit link only displays status.
    """
    manager = FakeLinkManager(
        status=_status(state="terminal", terminal_reason="revoked", last_close_code=4403)
    )
    check = _doctor_link(_surface_app(link_manager=manager))

    assert check["status"] == "fail"
    detail = check["detail"]
    assert detail.startswith("terminal: revoked")
    assert "nerdit unlink" in detail
    assert "nerdit link --device" in detail
    assert detail.index("nerdit unlink") < detail.index("nerdit link --device")
    # Doctor discipline, same as every other detail: names a command, never a
    # path, an origin or a capability.
    assert "/" not in detail
    assert CAPABILITY not in detail


@pytest.mark.parametrize("reason", ["protocol_unsupported", "auth_failed", "error"])
def test_other_terminal_reasons_keep_the_bare_detail(reason: str) -> None:
    """The smallest-honest-slice decision, pinned as an absence.

    ``protocol_unsupported`` ("upgrade the daemon") would be a one-line row in
    the same table, but its copy encodes a version-skew story that is not this
    table's line, and ``auth_failed``/``error`` have no single known fix to
    name. Adding one later must be a conscious edit, and this test is what
    makes it one. ``revoked`` left this list in P34 for exactly that reason: the
    cloud can now stamp it on a node nobody touched, so the operator needs the
    re-link command, and the removal here is the record of that decision.
    """
    manager = FakeLinkManager(status=_status(state="terminal", terminal_reason=reason))
    check = _doctor_link(_surface_app(link_manager=manager))
    assert check["status"] == "fail"
    assert check["detail"] == f"terminal: {reason}"


def test_the_doctor_link_detail_never_carries_a_url_path_or_token() -> None:
    manager = FakeLinkManager()
    body = _surface_app(link_manager=manager).get("/api/doctor", headers=_ADMIN).text
    assert CAPABILITY not in body
    assert "relay.example.test" not in body
    assert "/" not in _doctor_link(_surface_app(link_manager=manager))["detail"]


# ---------------------------------------------------------------------------
# GET /doctor — the github_token row (P33 D-GH-7)
# ---------------------------------------------------------------------------


def _doctor_github(client: TestClient) -> dict:
    body = client.get("/api/doctor", headers=_ADMIN).json()
    return next(check for check in body["checks"] if check["name"] == "github_token")


def test_doctor_skips_the_github_row_without_a_manager() -> None:
    check = _doctor_github(_surface_app(link_manager=None))
    assert check["status"] == "skipped"


def test_doctor_skips_the_github_row_while_nothing_is_mirrored() -> None:
    """Absence is quiet (D-GH-9): no cloud, cold reconnect, expiry — all
    ``skipped``, which never worsens the top status."""
    check = _doctor_github(_surface_app(link_manager=FakeLinkManager()))
    assert check["status"] == "skipped"
    assert "no GitHub installation token" in check["detail"]


def test_doctor_reports_a_live_github_token_with_its_lead() -> None:
    manager = FakeLinkManager(status=_status(github_installations=(_installation(2400),)))
    check = _doctor_github(_surface_app(link_manager=manager))
    assert check["status"] == "ok"
    assert check["detail"].startswith("1 installation; token expires in ")


def test_doctor_warns_when_the_soonest_expiry_is_under_the_lead() -> None:
    """The cloud re-mints ahead of expiry; a token this close to the edge
    means the pusher has gone quiet. The soonest of several decides."""
    manager = FakeLinkManager(
        status=_status(
            github_installations=(
                _installation(3000, installation_id=11),
                _installation(GITHUB_TOKEN_WARN_LEAD_S - 60, installation_id=22),
            )
        )
    )
    check = _doctor_github(_surface_app(link_manager=manager))
    assert check["status"] == "warn"
    assert check["detail"].startswith("2 installations; token expires in ")
    assert "overdue" in check["detail"]


def test_doctor_warns_when_the_mirror_held_a_token_that_has_since_lapsed() -> None:
    """(P33 doctor D2) The silent-pusher case: no LIVE installation, but the
    mirror HELD one that expired with no re-mint. Skipping would show an
    all-green row while every private auto-deploy has stopped, so it is
    ``warn`` with an overdue age — distinct from a never-linked box."""
    manager = FakeLinkManager(
        status=_status(github_installations=()),
        held=(1, datetime.now(UTC) - timedelta(seconds=120)),
    )
    check = _doctor_github(_surface_app(link_manager=manager))
    assert check["status"] == "warn"
    assert check["detail"].startswith("1 installation held; token expired ")
    assert "ago" in check["detail"]
    assert "overdue" in check["detail"]


def test_doctor_pluralises_and_hides_ids_on_the_lapsed_mirror_detail() -> None:
    manager = FakeLinkManager(
        status=_status(github_installations=()),
        held=(3, datetime.now(UTC) - timedelta(seconds=45)),
    )
    detail = _doctor_github(_surface_app(link_manager=manager))["detail"]
    assert detail.startswith("3 installations held; token expired ")
    assert "/" not in detail
    assert CAPABILITY not in detail


def test_a_genuinely_empty_mirror_is_skipped_not_warned() -> None:
    """The other half of D2: held count 0 stays ``skipped`` (never-linked),
    which never worsens the top status."""
    manager = FakeLinkManager(status=_status(github_installations=()), held=(0, None))
    check = _doctor_github(_surface_app(link_manager=manager))
    assert check["status"] == "skipped"
    assert "no GitHub installation token" in check["detail"]


def test_the_doctor_github_detail_never_carries_an_id_name_or_token() -> None:
    manager = FakeLinkManager(
        status=_status(github_installations=(_installation(2400, installation_id=987654),))
    )
    detail = _doctor_github(_surface_app(link_manager=manager))["detail"]
    assert "987654" not in detail
    assert "/" not in detail
    assert CAPABILITY not in detail


def test_the_link_check_is_the_eleventh_and_does_not_displace_the_others() -> None:
    # (P17d D-LIC6) ``license`` joined as the twelfth check; (P26 WP2)
    # ``acme_http_port`` joined as the thirteenth; (P33 D-GH-7)
    # ``github_token`` joined as the fourteenth and now sits last. ``link``
    # keeps its position, which is what this test is about — a new check must
    # ADD a row, never displace one.
    body = _surface_app(link_manager=None).get("/api/doctor", headers=_ADMIN).json()
    names = [check["name"] for check in body["checks"]]
    assert names[10] == "link"
    assert names[-2] == "acme_http_port"
    assert names[-1] == "github_token"
    assert len(names) == 14
    assert len(set(names)) == 14


# ---------------------------------------------------------------------------
# end to end: a refused entitlement, from the wire to both surfaces (WP-C4)
# ---------------------------------------------------------------------------


def _node_identity() -> NodeIdentity:
    raw = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    return NodeIdentity(base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii"))


async def test_an_entitlement_refusal_reaches_doctor_and_capabilities_end_to_end() -> None:
    """The WP-C4 signature test: one real refusal, both surfaces, no stand-in.

    Every leg of the mapping is already pinned on its own — the client's
    close classification, the manager's terminal reason, the doctor render, the
    capabilities projection — but nothing yet pins them *composed*. Here a real
    :class:`LinkManager` is refused by the relay with the real frames (error
    ``entitlement_required`` + close 4401), and THAT manager is the one mounted
    on the surface app, so a break anywhere along the chain lands here.

    The last assertion is the honesty check behind the hint's wording: after the
    refusal the manager never dials again, which is why the hint says "restart
    the daemon" rather than "wait".
    """
    async with link_harness(
        _node_identity(),
        scenarios=[
            Scenario(
                inject=[{"type": "error", "code": "entitlement_required", "message": "refused"}],
                close_with=(CLOSE_UNAUTHORIZED, "refused"),
            )
        ],
    ) as harness:
        await harness.manager.start()
        await harness.wait_state("terminal")

        client = _surface_app(link_manager=harness.manager)

        block = client.get("/api/capabilities", headers=_ADMIN).json()["link"]
        assert block["enabled"] is True
        assert block["state"] == "terminal"
        assert block["terminal_reason"] == "entitlement_required"
        assert block["last_close_code"] == 4401

        check = _doctor_link(client)
        assert check["status"] == "fail"
        assert check["detail"].startswith("terminal: entitlement_required")
        assert "not active" in check["detail"]
        assert "subscription" not in check["detail"]
        assert "restart the daemon" in check["detail"]

        # Terminal means terminal: the refusal bought exactly one session.
        await asyncio.sleep(0.02)
        assert len(harness.relay.sessions) == 1
        await harness.manager.stop()


# ---------------------------------------------------------------------------
# durable event vocabulary
# ---------------------------------------------------------------------------


def test_the_link_event_types_are_registered() -> None:
    """``EventRecorder.record`` refuses an unregistered type outright, so an
    unregistered link event would be silently dropped, not loudly broken."""
    assert "link.connected" in EVENT_TYPES
    assert "link.disconnected" in EVENT_TYPES
    # (P32) The entitlement mirror's change edge joins the family.
    assert "link.entitlement" in EVENT_TYPES
    assert len([t for t in EVENT_TYPES if t.startswith("link.")]) == 3


def test_every_disconnect_reason_is_a_machine_token() -> None:
    """D-P24-3: a fixed vocabulary a consumer's filter can actually match."""
    assert all(reason.replace("_", "").isalpha() for reason in DISCONNECT_REASONS)
    assert all(reason == reason.lower() for reason in DISCONNECT_REASONS)


# ---------------------------------------------------------------------------
# bootstrap: the start gate and the loopback origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        # All-interfaces spellings are bind addresses, not destinations.
        ("0.0.0.0", "http://127.0.0.1:9321"),
        ("::", "http://127.0.0.1:9321"),
        ("::0", "http://127.0.0.1:9321"),
        ("[::]", "http://127.0.0.1:9321"),
        ("", "http://127.0.0.1:9321"),
        # An explicit host is dialled verbatim: a daemon bound to one interface
        # must not be missed by a hardcoded loopback.
        ("127.0.0.1", "http://127.0.0.1:9321"),
        ("192.168.1.50", "http://192.168.1.50:9321"),
        # An IPv6 literal must come back bracketed, in either spelling:
        # ``httpx.URL("http://::1:9321")`` raises ``InvalidURL("Invalid port:
        # :1:9321")``, so the unbracketed form would blow up the mux's eager
        # ``AsyncClient`` construction — after the link already reported
        # ``connected``.
        ("::1", "http://[::1]:9321"),
        ("[::1]", "http://[::1]:9321"),
        ("fe80::1", "http://[fe80::1]:9321"),
    ],
)
def test_loopback_base_url_normalization(host: str, expected: str) -> None:
    assert _loopback_base_url(SimpleNamespace(host=host, port=9321)) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]", "127.0.0.1", "::1", "[::1]", "fe80::1"])
def test_every_loopback_base_url_is_a_dialable_origin(host: str) -> None:
    """The property that actually matters: httpx can parse what we build.

    The mux builds its ``AsyncClient`` from this string eagerly, so a value
    httpx refuses is not a bad URL — it is a link that reports ``connected``
    and then dies on the first stream.
    """
    import httpx

    url = httpx.URL(_loopback_base_url(SimpleNamespace(host=host, port=9321)))  # type: ignore[arg-type]
    assert url.port == 9321


def _settings(link: LinkSettings, tmp_path: object) -> SimpleNamespace:
    return SimpleNamespace(
        link=link,
        data_dir=str(tmp_path),
        daemon=DaemonSettings(host="127.0.0.1", port=9321),
    )


async def test_the_manager_is_not_built_when_the_link_is_disabled(tmp_path) -> None:  # noqa: ANN001
    settings = _settings(LinkSettings(), tmp_path)
    assert await build_link_manager(settings, AsyncMock()) is None  # type: ignore[arg-type]
    # Nothing was created — not even a key file.
    assert not (tmp_path / "link").exists()


async def test_an_enabled_but_unclaimed_daemon_warns_once_and_starts_nothing(
    tmp_path,  # noqa: ANN001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reachable half of the gate: WP-C2 has not run yet.

    (``enabled`` without ``relay_url`` is refused by ``LinkSettings`` itself, so
    the only reachable "enabled but not linked" state is a missing ``node_id``.)
    """
    caplog.set_level(logging.WARNING)
    link = LinkSettings(enabled=True, relay_url="wss://relay.example.test", node_id=None)
    settings = _settings(link, tmp_path)

    assert await build_link_manager(settings, AsyncMock()) is None  # type: ignore[arg-type]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "nerdit link" in warnings[0].getMessage()
    assert not (tmp_path / "link").exists()


async def test_a_claimed_daemon_gets_a_manager_pointed_at_its_own_listener(
    tmp_path,  # noqa: ANN001
) -> None:
    link = LinkSettings(
        enabled=True,
        relay_url="wss://relay.example.test/link",
        node_id=NODE_ID,
        slug="my-node",
    )
    manager = await build_link_manager(_settings(link, tmp_path), AsyncMock())  # type: ignore[arg-type]

    assert manager is not None
    status = manager.status()
    assert status.node_id == NODE_ID
    assert status.relay_host == "relay.example.test"
    assert status.state == "connecting"
    # The identity was generated on the node, owner-only, and nothing dialled.
    key_file = tmp_path / "link" / "node.key"
    assert key_file.is_file()
    assert oct(key_file.stat().st_mode & 0o777) == "0o600"


async def test_a_tokenless_daemon_is_warned_that_local_bearers_now_fail(
    tmp_path,  # noqa: ANN001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirrors the [mcp].http_enabled precedent, one notch softer.

    With the tunnel→LOCAL-admin fall-through closed, a link-enabled tokenless
    daemon refuses every Bearer that is not the live capability. That is a
    visible behaviour change for an operator whose CLI still carries an old
    token, so it is announced at boot rather than discovered as a 403.
    """
    caplog.set_level(logging.WARNING)
    link = LinkSettings(enabled=True, relay_url="wss://relay.example.test", node_id=NODE_ID)
    settings = _settings(link, tmp_path)
    assert settings.daemon.auth_token is None

    assert await build_link_manager(settings, AsyncMock()) is not None  # type: ignore[arg-type]

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "auth_token" in warnings[0]
    # Never the URL, never a token value.
    assert "relay.example.test" not in warnings[0]


async def test_a_daemon_with_an_auth_token_is_not_warned(
    tmp_path,  # noqa: ANN001
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    link = LinkSettings(enabled=True, relay_url="wss://relay.example.test", node_id=NODE_ID)
    settings = SimpleNamespace(
        link=link,
        data_dir=str(tmp_path),
        daemon=DaemonSettings(host="127.0.0.1", port=9321, auth_token="a-real-token"),  # noqa: S106
    )

    assert await build_link_manager(settings, AsyncMock()) is not None  # type: ignore[arg-type]
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_a_corrupt_node_key_disables_the_link_without_aborting_boot(
    tmp_path,  # noqa: ANN001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Remote access is an opt-in accessory; it must never take the daemon down.

    A corrupt key is also never regenerated — that would change the node's
    enrolled identity and permanently break the ADR-W2 claim binding.
    """
    caplog.set_level(logging.ERROR)
    key_dir = tmp_path / "link"
    key_dir.mkdir()
    (key_dir / "node.key").write_text("this-is-not-a-key")
    link = LinkSettings(enabled=True, relay_url="wss://relay.example.test", node_id=NODE_ID)

    assert await build_link_manager(_settings(link, tmp_path), AsyncMock()) is None  # type: ignore[arg-type]

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "node identity" in errors[0].getMessage()
    # Untouched: the operator's file is theirs to fix.
    assert (key_dir / "node.key").read_text() == "this-is-not-a-key"


async def test_the_node_name_falls_back_to_the_hostname_when_unslugged(
    tmp_path,  # noqa: ANN001
) -> None:
    import socket

    link = LinkSettings(enabled=True, relay_url="wss://relay.example.test", node_id=NODE_ID)
    manager = await build_link_manager(_settings(link, tmp_path), AsyncMock())  # type: ignore[arg-type]
    assert manager is not None
    assert manager._node_name == socket.gethostname()  # noqa: SLF001


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------


def test_the_claim_keys_are_restart_required() -> None:
    """Whole-section precedent: the manager captures both at boot."""
    assert {"node_id", "slug"} <= _RESTART_KEYS["link"]


def test_the_claim_keys_default_to_unset_and_reject_whitespace() -> None:
    link = LinkSettings()
    assert link.node_id is None and link.slug is None
    for bad in ("", "has space", "tab\there", " lead", "trail "):
        with pytest.raises(ValueError, match="whitespace|empty"):
            LinkSettings(node_id=bad)
        with pytest.raises(ValueError, match="whitespace|empty"):
            LinkSettings(slug=bad)


def test_the_section_still_carries_no_role_or_admin_knob() -> None:
    """D-R2: the submitter ceiling is structural, not a setting."""
    fields = set(LinkSettings.model_fields)
    assert "role" not in fields
    assert "allow_admin" not in fields
    assert fields == {
        "enabled",
        "relay_url",
        "key_file",
        "capability_ttl_s",
        "renew_margin_s",
        "node_id",
        "slug",
        # (P26 WP-H) The cloud's hosted base domain — a claim/refresh result,
        # restart-required like every other key here. Still no role knob.
        "nodes_base_domain",
    }


def test_capabilities_separates_a_never_linked_node_from_a_deliberate_opt_out() -> None:
    """The disabled link block carries the ENROLMENT fact, not just the flag.

    Both nodes here have no live manager and both report ``enabled: false`` —
    one never linked, one linked and then set ``[link].enabled = false``. The
    flag cannot tell them apart, and a consumer that read it as enrolment put a
    permanent, non-dismissable "not linked" banner on top of a deliberate
    opt-out (and on top of a node whose key had merely become unreadable).
    ``node_id`` is the discriminator the doctor ``link`` row already used.
    """
    never = _surface_app(
        link_manager=None,
        settings=_caps_settings(link_enabled=False, slug=None, node_id=None),
    )
    assert never.get("/api/capabilities", headers=_ADMIN).json()["link"] == {
        "enabled": False,
        "node_id": None,
        "slug": None,
    }

    opted_out = _surface_app(
        link_manager=None,
        settings=_caps_settings(link_enabled=False, slug="gpu-box", node_id=NODE_ID),
    )
    assert opted_out.get("/api/capabilities", headers=_ADMIN).json()["link"] == {
        "enabled": False,
        "node_id": NODE_ID,
        "slug": "gpu-box",
    }
