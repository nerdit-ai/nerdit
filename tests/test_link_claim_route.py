"""Test daemon-owned claim and hot unlink with real config and middleware.

Claim codes travel to the cloud once and never enter durable records, responses
or body hashes. Sweep successes and refusals for leaks. Cloud refusals leave
config unchanged; already-linked and missing-relay refusals make no cloud call.

Unlink stops the live tunnel immediately, clears claim state while preserving
relay_url/key_file, and is a successful no-op when already unlinked. Use temporary
ConfigStore files, mocked queries and MockTransport for the cloud hop.
"""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import tomli_w
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.config.settings import LinkSettings
from nerdit.config.store import ConfigStore
from nerdit.core.link.identity import public_reference
from nerdit.daemon.audit import AuditMiddleware, audit_params, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import NO_BODY_CACHE_ACTIONS, NO_BODY_HASH_ACTIONS
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import router as link_router
from nerdit.daemon.schemas.link import LinkClaimRequest
from nerdit.db.models import ApiToken, TokenRole

ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SUBMITTER_TOKEN = "scoped-raw"  # noqa: S105 - a test literal
CODE = "PASTE-ME-1234"  # noqa: S105 - the plaintext link code under test
NODE_ID = "00000000-0000-4000-8000-00000000000a"
SLUG = "tower"
RELAY = "wss://relay.example.test/link"
DOMAIN = "nodes.example"
#: A marker the cloud puts in a refusal body; nothing may reflect it.
CLOUD_NOISE = {"detail": "cloud-body-marker-9f2a"}

_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "idem-1"}
_SUBMITTER = {"Authorization": f"Bearer {SUBMITTER_TOKEN}", "Idempotency-Key": "idem-1"}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class FakeCloud:
    """A scripted ``POST /api/link/claim`` endpoint plus a request recorder."""

    def __init__(
        self,
        *,
        status: int = 200,
        payload: Any = None,
        error_code: str | None = None,
        raises: bool = False,
    ) -> None:
        self.status = status
        self.payload = payload if payload is not None else {"node_id": NODE_ID, "node_slug": SLUG}
        self.error_code = error_code
        self.raises = raises
        self.requests: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.requests.append(json.loads(request.content))
        if self.raises:
            raise httpx.ConnectError("nothing listening", request=request)
        if self.status >= 400:
            body = {"error": {"code": self.error_code}} if self.error_code else {"detail": "no"}
            return httpx.Response(self.status, json=body)
        return httpx.Response(self.status, json=self.payload)

    @property
    def called(self) -> bool:
        return bool(self.requests)


class FakeMetadata:
    """A scripted ``GET /api/link/metadata`` endpoint (P26 WP-H).

    Separate from :class:`FakeCloud` because the refresh hop is a GET with no
    body — reusing the claim fake would have it try to JSON-decode an empty
    request. Same recorder shape so a test can assert what (if anything) left
    the machine.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        payload: Any = None,
        error_code: str | None = None,
        raises: bool = False,
    ) -> None:
        self.status = status
        self.payload = payload if payload is not None else {"nodes_base_domain": DOMAIN}
        self.error_code = error_code
        self.raises = raises
        self.urls: list[str] = []
        self.bodies: list[bytes] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.bodies.append(request.content)
        if self.raises:
            raise httpx.ConnectError("nothing listening", request=request)
        if self.status >= 400:
            body: Any = {"error": {"code": self.error_code}} if self.error_code else CLOUD_NOISE
            return httpx.Response(self.status, json=body)
        return httpx.Response(self.status, json=self.payload)

    @property
    def called(self) -> bool:
        return bool(self.urls)


class FakeLinkManager:
    """What ``app.state.link_manager`` looks like to the daemon.

    ``validate_capability`` is here because the auth middleware consults the
    manager on EVERY bearer (the WP-C1 tunnel-principal branch); it answers
    ``False`` so the admin token still resolves normally.
    """

    def __init__(self) -> None:
        self.stop = AsyncMock()

    def validate_capability(self, token: str, role: str) -> bool:
        return False


def _queries() -> AsyncMock:
    queries = AsyncMock()
    queries.get_api_token_by_hash = AsyncMock(
        return_value=ApiToken(
            id="tok-1",
            name="ci-bot",
            role=TokenRole.submitter,
            token_hash=hash_token(SUBMITTER_TOKEN),
            max_gpus=1,
            max_concurrent_jobs=1,
        )
    )
    queries.insert_audit_log = AsyncMock()
    queries.touch_api_token = AsyncMock()
    return queries


class Harness(SimpleNamespace):
    """The built app plus the handles a test needs to assert against."""

    client: TestClient
    store: ConfigStore
    config_path: Path
    queries: AsyncMock
    cloud: FakeCloud | FakeMetadata | None
    app: FastAPI
    key_file: Path

    def raw(self) -> dict[str, Any]:
        with open(self.config_path, "rb") as handle:
            return tomllib.load(handle)

    def link_section(self) -> dict[str, Any]:
        return self.raw().get("link", {})

    def audit_rows(self) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.queries.insert_audit_log.await_args_list]


def _build(
    tmp_path: Path,
    *,
    link: dict[str, Any] | None = None,
    cloud: FakeCloud | FakeMetadata | None = None,
    link_manager: object | None = None,
    config: dict[str, Any] | None = None,
) -> Harness:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.toml"
    document: dict[str, Any] = dict(config or {})
    if link:
        document["link"] = link
    config_path.write_bytes(tomli_w.dumps(document).encode("utf-8"))

    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(link_router)
    app.include_router(api)

    data_dir = tmp_path / "data"
    app.state.config_store = ConfigStore(config_path)
    app.state.settings = SimpleNamespace(
        data_dir=str(data_dir),
        link=LinkSettings(),
        daemon=SimpleNamespace(host="127.0.0.1", port=9321),
    )
    app.state.link_manager = link_manager
    app.state.link_claim_transport = httpx.MockTransport(cloud.handler) if cloud else None

    queries = _queries()
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=ADMIN_TOKEN, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    return Harness(
        client=TestClient(app, raise_server_exceptions=False),
        store=app.state.config_store,
        config_path=config_path,
        queries=queries,
        cloud=cloud,
        app=app,
        key_file=data_dir / "link" / "node.key",
    )


def _claim_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "code": CODE,
        "api_url": "https://app.example.test",
        "relay_url": RELAY,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# claim — the happy path
# ---------------------------------------------------------------------------


def test_claim_happy_path_persists_and_enables(tmp_path: Path) -> None:
    """One claim: identity enrolled, config written, restart announced."""
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["node_id"] == NODE_ID
    assert body["slug"] == SLUG
    assert body["relay_url"] == RELAY
    assert body["enabled"] is True
    assert body["requires_restart"] is True
    # Every [link] key is restart-keyed, so the claim's own keys must be listed.
    assert {"link.node_id", "link.slug"} <= set(body["restart_keys"])

    # What actually reached the cloud: the code once, and the PUBLIC verifier —
    # never the private key, never the fingerprint as a stand-in.
    assert h.cloud is not None
    sent = h.cloud.requests[0]
    assert sent["code"] == CODE
    private_key_b64 = h.key_file.read_text().strip()
    assert sent["credential_reference"] == public_reference(private_key_b64)
    assert set(sent) == {"code", "credential_reference"}
    assert h.cloud.urls == ["https://app.example.test/api/link/claim"]
    # The fingerprint is the non-secret audit form of the key, not the key.
    assert len(body["verifier_fingerprint"]) == 64
    assert private_key_b64 not in response.text

    # The key was generated on the node, owner-only.
    assert oct(h.key_file.stat().st_mode & 0o777) == "0o600"

    # Persisted, and still a valid [link] section on the next boot.
    section = h.link_section()
    assert section["node_id"] == NODE_ID
    assert section["slug"] == SLUG
    assert section["enabled"] is True
    assert section["relay_url"] == RELAY
    # (D5) ``api_url`` is per-invocation and NEVER persisted — not as a [link]
    # key, not anywhere else in the file. It is where the operator pasted from,
    # not what the daemon is.
    assert "api_url" not in section
    assert "app.example.test" not in h.config_path.read_text()
    settings = LinkSettings(**h.store.effective_section("link"))
    assert (settings.node_id, settings.slug, settings.enabled) == (NODE_ID, SLUG, True)

    row = h.audit_rows()[-1]
    assert row["action"] == "link.created"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    params = json.loads(row["params_redacted"])
    assert params["node_id"] == NODE_ID
    assert params["slug"] == SLUG
    # The HOST only — an accepted relay URL may carry a path, and paths are
    # where credentials get parked; the config file holds the full value.
    assert params["relay_host"] == "relay.example.test"
    assert "relay_url" not in params
    assert params["enabled"] is True
    assert params["verifier_fingerprint"] == body["verifier_fingerprint"]


def test_claim_enables_the_mcp_transport_when_a_token_exists(tmp_path: Path) -> None:
    """A linked node serves the cloud's remote MCP gateway, so an enabled claim
    also stages ``[mcp].http_enabled`` — provided the bearer precondition the
    daemon enforces at boot is already met (owner decision 2026-08-23)."""
    h = _build(tmp_path, cloud=FakeCloud(), config={"daemon": {"auth_token": "t" * 32}})

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mcp_http_enabled"] is True
    assert body["mcp_skipped_reason"] is None
    # The whole mutation, not just the [link] half (Codex round 1).
    assert "mcp.http_enabled" in body["restart_keys"]
    assert h.store.effective_section("mcp")["http_enabled"] is True
    params = json.loads(h.audit_rows()[-1]["params_redacted"])
    assert params["mcp_http_enabled"] is True
    # The token itself never travels: not in the body, not in the audit row.
    assert "t" * 32 not in response.text
    assert "t" * 32 not in h.audit_rows()[-1]["params_redacted"]


def test_claim_skips_the_mcp_transport_without_a_token(tmp_path: Path) -> None:
    """No ``[daemon].auth_token`` → the transport would be an unauthenticated
    admin surface and the daemon would refuse to boot; the claim must never
    turn its own required restart into that refusal."""
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mcp_http_enabled"] is False
    assert body["mcp_skipped_reason"].startswith("no_auth_token:")
    assert "mcp.http_enabled" not in body["restart_keys"]
    assert h.store.effective_section("mcp").get("http_enabled", False) is False


def test_claim_no_enable_leaves_the_mcp_transport_alone(tmp_path: Path) -> None:
    h = _build(tmp_path, cloud=FakeCloud(), config={"daemon": {"auth_token": "t" * 32}})

    response = h.client.post("/api/link/claim", json=_claim_body(enable=False), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["mcp_http_enabled"] is False
    assert h.store.effective_section("mcp").get("http_enabled", False) is False


def test_claim_no_enable_keeps_disabled(tmp_path: Path) -> None:
    """``--no-enable`` persists the identity without arming the tunnel (D6)."""
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(enable=False), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    section = h.link_section()
    assert section["node_id"] == NODE_ID
    assert section.get("enabled", False) is False
    assert LinkSettings(**h.store.effective_section("link")).enabled is False


def test_claim_reuses_stored_relay_url(tmp_path: Path) -> None:
    """An operator who already configured ``[link].relay_url`` need not repeat it."""
    h = _build(tmp_path, link={"relay_url": RELAY}, cloud=FakeCloud())

    body = _claim_body()
    del body["relay_url"]
    response = h.client.post("/api/link/claim", json=body, headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["relay_url"] == RELAY
    assert h.link_section()["relay_url"] == RELAY


# ---------------------------------------------------------------------------
# claim — local refusals, all BEFORE the cloud is contacted
# ---------------------------------------------------------------------------


def test_claim_without_any_relay_url_422(tmp_path: Path) -> None:
    """Never burn a code to produce a node that cannot start (D5)."""
    h = _build(tmp_path, cloud=FakeCloud())

    body = _claim_body()
    del body["relay_url"]
    response = h.client.post("/api/link/claim", json=body, headers=_ADMIN)

    assert response.status_code == 422
    assert response.json()["code"] == "link.relay_url_required"
    assert h.cloud is not None and not h.cloud.called
    assert h.link_section() == {}


class _RelayClearingCloud(FakeCloud):
    """A cloud whose round trip is when an admin clears ``[link].relay_url``.

    The claim resolves its relay before the exchange and writes it after, and
    the config routes do not share ``_LINK_MUTATION_LOCK``, so the exchange is
    a window a concurrent ``PUT /config/daemon/link`` can land in. Rewriting
    the file from inside the handler is that write, at the widest point of the
    window.
    """

    config_path: Path | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert self.config_path is not None  # noqa: S101 - a test rig invariant
        self.config_path.write_bytes(tomli_w.dumps({"link": {}}).encode("utf-8"))
        return super().handler(request)


@pytest.mark.parametrize("enable", [True, False])
def test_claim_refuses_when_the_relay_vanishes_inside_the_cloud_exchange(
    tmp_path: Path, enable: bool
) -> None:
    """Re-resolve the stored relay at commit after the cloud exchange.

    With no relay in the request, clearing the stored value must refuse both
    enable modes. Previously, enable=True raised a generic config error after
    spending the code; enable=False persisted node_id/slug without a relay and
    returned a stale URL. Explicit request relays are persisted and cannot reach
    this race. Device polling has a separate test for its credential-check await.
    """
    cloud = _RelayClearingCloud()
    h = _build(tmp_path, link={"relay_url": RELAY}, cloud=cloud)
    cloud.config_path = h.config_path

    body = _claim_body(enable=enable)
    del body["relay_url"]
    response = h.client.post("/api/link/claim", json=body, headers=_ADMIN)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "link.relay_url_required"
    # Refusing without committing is what keeps recovery real: restore the
    # relay, claim again, and the cloud's idempotent grant returns the same
    # identity for the same enrolled verifier.
    section = h.link_section()
    assert section.get("node_id") is None
    assert section.get("slug") is None
    assert "relay_url" not in section


def test_claim_while_linked_409(tmp_path: Path) -> None:
    """D8: a daemon that still holds its identity must not re-claim silently."""
    stored = {"relay_url": RELAY, "node_id": "old-node", "slug": "old"}
    h = _build(tmp_path, link=stored, cloud=FakeCloud())
    before = h.config_path.read_bytes()

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "link.already_linked"
    assert "unlink" in payload["hint"]
    assert h.cloud is not None and not h.cloud.called
    assert h.config_path.read_bytes() == before


def test_claim_with_an_unreadable_node_key_500s_before_the_cloud(tmp_path: Path) -> None:
    """A corrupt ``node.key`` is a structured 500, never a silent regeneration.

    ``load_or_create_identity`` refuses to mint a NEW key over a malformed one
    (regenerating would break the ADR-W2 claim binding of an already-enrolled
    node), so the route's job is to turn that into a code an agent can act on —
    with the failing path in the log and nowhere else (the P14c M3 posture).
    """
    h = _build(tmp_path, cloud=FakeCloud())
    h.key_file.parent.mkdir(parents=True)
    # Decodes cleanly but to the wrong length: NodeIdentity raises rather than
    # regenerating, which is exactly the branch under test.
    h.key_file.write_text("not-a-real-node-key\n")
    before = h.config_path.read_bytes()

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 500, response.text
    assert response.json()["code"] == "link.identity_unavailable"
    # The code is never spent on a node that cannot present a verifier.
    assert h.cloud is not None and not h.cloud.called
    assert h.config_path.read_bytes() == before
    assert h.link_section() == {}
    # The path belongs in the log and nowhere else.
    assert "node.key" not in response.text
    assert str(h.key_file) not in response.text
    assert CODE not in response.text


def test_claim_missing_idempotency_key_400(tmp_path: Path) -> None:
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post(
        "/api/link/claim",
        json=_claim_body(),
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    assert h.cloud is not None and not h.cloud.called


def test_claim_requires_admin(tmp_path: Path) -> None:
    """D2: linking is custody, and the tunnel principal is only a submitter."""
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_SUBMITTER)

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert h.cloud is not None and not h.cloud.called
    assert h.link_section() == {}


@pytest.mark.parametrize(
    ("bad_url", "reason"),
    [
        ("http://cloud.example.test", "plain http off-box"),
        ("ftp://cloud.example.test", "unsupported scheme"),
        ("https://user:pw@cloud.example.test", "userinfo"),
        ("https://cloud.example.test?token=x", "query string"),
        ("https://cloud.example.test#frag", "fragment"),
        ("https://cloud example.test", "whitespace"),
        ("https://", "no host"),
    ],
)
def test_claim_rejects_a_dirty_api_url(tmp_path: Path, bad_url: str, reason: str) -> None:
    """The schema refuses before the route body runs — so before the cloud hop."""
    h = _build(tmp_path, cloud=FakeCloud())
    response = h.client.post("/api/link/claim", json=_claim_body(api_url=bad_url), headers=_ADMIN)
    assert response.status_code == 422, reason
    assert h.cloud is not None and not h.cloud.called


def test_claim_accepts_a_loopback_dev_stack_over_http(tmp_path: Path) -> None:
    """RFC 6761 loopback names keep the dev stack usable without TLS."""
    h = _build(tmp_path, cloud=FakeCloud())
    response = h.client.post(
        "/api/link/claim", json=_claim_body(api_url="http://app.localhost"), headers=_ADMIN
    )
    assert response.status_code == 200, response.text
    assert h.cloud is not None
    assert h.cloud.urls == ["http://app.localhost/api/link/claim"]


def test_claim_rejects_an_invalid_relay_url_before_the_cloud_hop(tmp_path: Path) -> None:
    """The relay rules are ``LinkSettings``' own — validated pre-flight (D5)."""
    h = _build(tmp_path, cloud=FakeCloud())
    response = h.client.post(
        "/api/link/claim", json=_claim_body(relay_url="http://relay.example.test"), headers=_ADMIN
    )
    assert response.status_code == 422
    assert h.cloud is not None and not h.cloud.called


# ---------------------------------------------------------------------------
# claim — cloud refusals (D11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "cloud_code", "expected_status", "expected_code"),
    [
        (404, "link_code_invalid", 409, "link.claim_refused"),
        (409, "link_code_consumed", 409, "link.claim_refused"),
        (410, "link_code_expired", 409, "link.claim_refused"),
        (429, "link_code_attempts_exceeded", 429, "link.claim_rate_limited"),
        (400, None, 409, "link.claim_refused"),
        (503, "internal", 502, "link.cloud_error"),
    ],
)
def test_claim_refusal_mapping(
    tmp_path: Path,
    status: int,
    cloud_code: str | None,
    expected_status: int,
    expected_code: str,
) -> None:
    """Stable daemon codes wrap the cloud's vocabulary; the code value never rides along."""
    h = _build(tmp_path, cloud=FakeCloud(status=status, error_code=cloud_code))
    before = h.config_path.read_bytes()

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == expected_status
    payload = response.json()
    assert payload["code"] == expected_code
    assert payload["cloud_status"] == status
    assert payload["cloud_code"] == (cloud_code or f"http_{status}")
    assert payload["hint"]
    # The code was spent (or not) cloud-side; locally nothing moved.
    assert h.config_path.read_bytes() == before
    assert CODE not in response.text
    assert h.link_section() == {}


def test_claim_cloud_unreachable_502(tmp_path: Path) -> None:
    """A transport failure names the netloc — never the URL, never the code."""
    h = _build(tmp_path, cloud=FakeCloud(raises=True))

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502
    payload = response.json()
    assert payload["code"] == "link.cloud_unreachable"
    assert "app.example.test" in payload["message"]
    assert "/api/link/claim" not in payload["message"]
    assert CODE not in response.text
    assert h.link_section() == {}


def test_claim_malformed_200_502(tmp_path: Path) -> None:
    """A 200 that is not a node identity is a protocol error, not a claim."""
    h = _build(tmp_path, cloud=FakeCloud(payload={"node_id": NODE_ID}))

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_protocol"
    assert h.link_section() == {}


def test_claim_empty_identity_200_is_also_a_protocol_error(tmp_path: Path) -> None:
    h = _build(tmp_path, cloud=FakeCloud(payload={"node_id": NODE_ID, "node_slug": ""}))
    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_protocol"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"node_id": None, "node_slug": None}, "JSON nulls"),
        ({"node_id": NODE_ID, "node_slug": None}, "a null slug beside a real id"),
        ({"node_id": 5, "node_slug": {}}, "an int id and an object slug"),
        ({"node_id": [NODE_ID], "node_slug": SLUG}, "a list id"),
    ],
)
def test_claim_non_string_identity_200_is_a_protocol_error(
    tmp_path: Path, payload: dict[str, Any], reason: str
) -> None:
    """A 200 whose identity is not two strings never reaches the config store.

    The guard is a type check, not an emptiness check, precisely because
    ``str(None)`` is the truthy ``"None"``: coercion would have persisted
    ``[link].node_id = "None"`` and left the daemon "linked" to nothing.
    """
    h = _build(tmp_path, cloud=FakeCloud(payload=payload))

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502, reason
    assert response.json()["code"] == "link.cloud_protocol"
    assert "node_id" not in h.link_section()
    assert h.link_section() == {}


def test_claim_idempotent_reclaim_after_wipe(tmp_path: Path) -> None:
    """The lost-local-state path the cloud's replay idempotency exists for.

    A daemon whose ``[link]`` claim seam was cleared (a restore from an older
    config, a hand edit) re-presents the SAME verifier and gets the SAME
    identity back — which is why D8's already-linked 409 costs nothing.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    first = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert first.status_code == 200, first.text

    # Simulate the lost state: the claim seam is gone, the key file is not.
    h.store.commit(h.store.stage("link", {"node_id": None, "slug": None}))
    assert "node_id" not in h.link_section()

    second = h.client.post(
        "/api/link/claim",
        json=_claim_body(),
        headers={**_ADMIN, "Idempotency-Key": "idem-2"},
    )

    assert second.status_code == 200, second.text
    assert second.json()["node_id"] == NODE_ID
    assert h.link_section()["node_id"] == NODE_ID
    assert h.cloud is not None
    verifiers = {sent["credential_reference"] for sent in h.cloud.requests}
    assert len(verifiers) == 1, "the node re-claimed under a fresh identity"


# ---------------------------------------------------------------------------
# the never-leak contract
# ---------------------------------------------------------------------------


def test_never_leak_sweep(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The literal code appears in no audit row, no log record, no response.

    Run over the happy path AND every refusal shape, because a leak is most
    likely on an error path — an exception message quoting the request, a debug
    log of the outgoing body.
    """
    caplog.set_level(logging.DEBUG)
    clouds = [
        FakeCloud(),
        FakeCloud(status=404, error_code="link_code_invalid"),
        FakeCloud(status=410, error_code="link_code_expired"),
        FakeCloud(status=429, error_code="link_code_attempts_exceeded"),
        FakeCloud(status=500, error_code="boom"),
        FakeCloud(raises=True),
        FakeCloud(payload={"nope": 1}),
    ]
    for index, cloud in enumerate(clouds):
        h = _build(tmp_path / f"run-{index}", cloud=cloud)
        response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
        assert CODE not in response.text
        assert CODE not in json.dumps(h.audit_rows(), default=str)
        assert CODE not in h.config_path.read_text()

    assert CODE not in caplog.text

    # The denylist entry is the second line of defense (the route never puts a
    # ``code`` key in its params at all).
    assert audit_params({"code": CODE}) == {"code": "***"}
    # And the structural one: no repr of the request model can spill it.
    model = LinkClaimRequest(code=CODE, api_url="https://app.example.test", relay_url=RELAY)
    assert CODE not in repr(model)
    assert CODE not in str(model)
    assert model.code == CODE


def test_validation_errors_never_echo_the_code(tmp_path: Path) -> None:
    """A 422 on a *malformed* body must not hand the code back either.

    ``never_leak_sweep`` only sends well-formed bodies, so it misses the one
    place the plaintext can escape without the route ever running: FastAPI's
    422, which echoes the offending ``input``. Two live shapes, and
    ``repr=False`` on the field stops neither — both raise before the model is
    constructed, so the echo is the raw submitted value:

    * a sibling ``missing`` (``api_url`` omitted) whose ``input`` is the WHOLE
      body dict, ``code`` key included;
    * ``string_too_long`` on ``code`` itself, whose ``input`` IS the code —
      duplicated into both ``detail`` and ``diagnostics``.

    Covered by the ``"code"`` entry in ``errors._SECRET_INPUT_FIELDS``.
    """
    h = _build(tmp_path)

    missing_sibling = h.client.post("/api/link/claim", json={"code": CODE}, headers=_ADMIN)
    assert missing_sibling.status_code == 422
    assert CODE not in missing_sibling.text

    long_code = "L" * 200
    too_long = h.client.post(
        "/api/link/claim",
        json=_claim_body(code=long_code),
        headers=_ADMIN,
    )
    assert too_long.status_code == 422
    assert long_code not in too_long.text


# ---------------------------------------------------------------------------
# unlink
# ---------------------------------------------------------------------------


def test_unlink_drops_live_manager(tmp_path: Path) -> None:
    """The one hot action (D7) — plus the exact scope of the wipe (D10)."""
    manager = FakeLinkManager()
    h = _build(
        tmp_path,
        link={"enabled": True, "relay_url": RELAY, "node_id": NODE_ID, "slug": SLUG},
        link_manager=manager,
    )
    # The BOOT snapshot a running daemon would be holding — read paths project
    # the hosted identity off this, never off the TOML (D-P26-14).
    h.app.state.settings.link = LinkSettings(
        enabled=True, relay_url=RELAY, node_id=NODE_ID, slug=SLUG, nodes_base_domain=DOMAIN
    )
    h.key_file.parent.mkdir(parents=True)
    h.key_file.write_text("not-a-real-key\n")

    response = h.client.delete("/api/link", headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "was_linked": True,
        "tunnel_stopped": True,
        "key_removed": True,
        "pending_wipes": 0,
        "requires_restart": True,
    }
    manager.stop.assert_awaited_once()
    assert h.app.state.link_manager is None
    # Second half of the hot action: the in-process identity the stopped tunnel
    # served goes with it, or every read path keeps advertising the revoked
    # node's hosted address until someone restarts (PR #128 review).
    live = h.app.state.settings.link
    assert live.node_id is None
    assert live.slug is None
    assert live.nodes_base_domain is None
    assert live.enabled is False
    assert live.relay_url == RELAY

    section = h.link_section()
    assert "node_id" not in section
    assert "slug" not in section
    assert section["enabled"] is False
    # Operator-supplied endpoint config survives — a re-link reuses it.
    assert section["relay_url"] == RELAY
    assert not h.key_file.exists()

    row = h.audit_rows()[-1]
    assert row["action"] == "link.revoked"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    params = json.loads(row["params_redacted"])
    assert params == {
        "was_linked": True,
        "tunnel_stopped": True,
        "key_removed": True,
        "pending_wipes": 0,
    }
    # The key PATH is never in the response or the row (the identity.py posture).
    assert "node.key" not in response.text
    assert "node.key" not in row["params_redacted"]


def test_unlink_idempotent_when_unlinked(tmp_path: Path) -> None:
    """D9: retries and cleanup scripts must be safe."""
    h = _build(tmp_path)

    response = h.client.delete("/api/link", headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "was_linked": False,
        "tunnel_stopped": False,
        "key_removed": False,
        "pending_wipes": 0,
        "requires_restart": True,
    }
    row = h.audit_rows()[-1]
    assert row["action"] == "link.revoked"
    assert row["target_id"] is None
    assert json.loads(row["params_redacted"])["was_linked"] is False


def test_unlink_requires_admin(tmp_path: Path) -> None:
    h = _build(tmp_path, link={"relay_url": RELAY, "node_id": NODE_ID}, link_manager=None)
    response = h.client.delete("/api/link", headers=_SUBMITTER)
    assert response.status_code == 403
    assert h.link_section()["node_id"] == NODE_ID


def test_unlink_requires_an_idempotency_key(tmp_path: Path) -> None:
    h = _build(tmp_path, link={"relay_url": RELAY, "node_id": NODE_ID})
    response = h.client.delete("/api/link", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"})
    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    assert h.link_section()["node_id"] == NODE_ID


def test_unlinked_config_gates_manager(tmp_path: Path) -> None:
    """After an unlink the documented start gate is false.

    ``build_link_manager`` starts only on ``enabled AND relay_url AND node_id``;
    the manager-side half of that is owned by ``tests/test_link_surface.py``
    (``test_an_enabled_but_unclaimed_daemon_warns_once_and_starts_nothing``).
    What this pins is the *config* the gate will read on the next boot.
    """
    h = _build(
        tmp_path,
        link={"enabled": True, "relay_url": RELAY, "node_id": NODE_ID, "slug": SLUG},
    )

    assert h.client.delete("/api/link", headers=_ADMIN).status_code == 200

    settings = LinkSettings(**h.store.effective_section("link"))
    assert settings.enabled is False
    assert settings.node_id is None
    assert settings.slug is None
    assert settings.relay_url == RELAY
    assert not (settings.enabled and settings.relay_url and settings.node_id)


# ---------------------------------------------------------------------------
# audit / idempotency wiring
# ---------------------------------------------------------------------------


def test_derive_action_pins() -> None:
    """Both mounts collapse to one action, and neither body is hashed."""
    assert derive_action("POST", "/api/link/claim") == ("link.created", "link", None)
    assert derive_action("DELETE", "/link") == ("link.revoked", "link", None)
    # The claim body carries a low-entropy human-typed secret: no digest at rest.
    assert "link.created" in NO_BODY_HASH_ACTIONS
    # The RESPONSE is the non-secret node identity, so replay caching is correct.
    assert "link.created" not in NO_BODY_CACHE_ACTIONS
    assert "link.revoked" not in NO_BODY_CACHE_ACTIONS


# ---------------------------------------------------------------------------
# PR #115 review regressions
# ---------------------------------------------------------------------------


def test_unlink_wipes_the_boot_time_key_not_a_restaged_path(tmp_path: Path) -> None:
    """A ``key_file`` re-staged through the config API is NOT what unlink wipes.

    The effective ``[link].key_file`` can be repointed at any time without a
    restart, so at unlink it may name a pre-provisioned key the enrolled
    identity never lived in. The wipe targets the key the running daemon
    actually loaded (the boot-time path); the staged path is operator property
    and survives (PR #115 P1).
    """
    h = _build(
        tmp_path,
        link={"enabled": True, "relay_url": RELAY, "node_id": NODE_ID, "slug": SLUG},
    )
    h.key_file.parent.mkdir(parents=True)
    h.key_file.write_text("enrolled-key\n")
    decoy = tmp_path / "provisioned.key"
    decoy.write_text("pre-provisioned-key\n")
    h.store.commit(h.store.stage("link", {"key_file": str(decoy)}))

    response = h.client.delete("/api/link", headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["key_removed"] is True
    # The enrolled key is gone; the staged, never-enrolled key is untouched.
    assert not h.key_file.exists()
    assert decoy.read_text() == "pre-provisioned-key\n"
    # D10 still holds: the staged path itself survives in config.
    assert h.link_section()["key_file"] == str(decoy)


def test_claim_stamps_the_enrolled_path_and_unlink_prefers_it(tmp_path: Path) -> None:
    """The claim records exactly which file it enrolled; unlink wipes that one.

    Between a same-process claim and unlink the effective ``key_file`` is
    repointed at a decoy — the wipe must still hit the file whose verifier the
    cloud actually holds, and the stamp must not survive the unlink.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert response.status_code == 200, response.text
    assert h.app.state.link_enrolled_key_file == str(h.key_file)
    assert h.key_file.exists()

    decoy = tmp_path / "provisioned.key"
    decoy.write_text("pre-provisioned-key\n")
    h.store.commit(h.store.stage("link", {"key_file": str(decoy)}))

    response = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "idem-2"})
    assert response.status_code == 200, response.text
    assert response.json()["key_removed"] is True
    assert not h.key_file.exists()
    assert decoy.exists()
    assert h.app.state.link_enrolled_key_file is None


async def test_unlink_serializes_behind_an_in_flight_claim(tmp_path: Path) -> None:
    """The claim/unlink lock: an unlink cannot slip inside a claim's cloud hop.

    Without it, the unlink observes "not linked", deletes the key the claim
    just minted, and lets the claim commit a cloud identity whose private key
    is gone — a daemon that re-keys on restart and fails relay auth terminally
    (PR #115 P1). With it, the unlink parks until the claim commits, then
    tears down the *committed* link (``was_linked: true``).
    """
    import asyncio

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"node_id": NODE_ID, "node_slug": SLUG})

    h = _build(tmp_path)
    h.app.state.link_claim_transport = httpx.MockTransport(handler)

    transport = httpx.ASGITransport(app=h.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        claim = asyncio.create_task(
            client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        # The claim holds the lock, parked on the cloud, its fresh key on disk.
        unlink = asyncio.create_task(
            client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "idem-2"})
        )
        await asyncio.sleep(0.05)
        assert not unlink.done()
        assert h.key_file.exists()
        release.set()
        claim_response = await asyncio.wait_for(claim, timeout=5)
        unlink_response = await asyncio.wait_for(unlink, timeout=5)

    assert claim_response.status_code == 200, claim_response.text
    # The unlink saw the COMMITTED claim — never the pre-claim emptiness under
    # which it would have deleted the in-flight key.
    assert unlink_response.status_code == 200, unlink_response.text
    assert unlink_response.json()["was_linked"] is True
    assert not h.key_file.exists()


def test_a_nonconforming_cloud_code_is_never_reflected(tmp_path: Path) -> None:
    """``cloud_code`` is a bounded machine token, never untrusted echo text.

    An operator-supplied ``--api-url`` endpoint is untrusted; one that echoes
    the submitted link code inside ``{"error": {"code": …}}`` must not see it
    reflected into the daemon's refusal envelope (PR #115 P2).
    """
    h = _build(tmp_path, cloud=FakeCloud(status=404, error_code=CODE))

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 409
    assert CODE not in response.text
    assert response.json()["cloud_code"] == "http_404"

    # A conforming machine token still travels (the vocabulary keeps working).
    h2 = _build(tmp_path / "b", cloud=FakeCloud(status=404, error_code="link_code_invalid"))
    response = h2.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert response.json()["cloud_code"] == "link_code_invalid"


def test_both_commits_are_pinned_as_request_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mark_request_side_effect`` runs BEFORE each config commit (PR #115 P2).

    The commit is a non-DB durable effect, so without the pin a
    graceful-shutdown cancellation lets the idempotency middleware delete the
    claim key and a same-key retry re-executes into ``link.already_linked``
    instead of replaying the interrupted outcome.
    """
    h = _build(
        tmp_path,
        cloud=FakeCloud(),
        link_manager=FakeLinkManager(),
    )
    sections_at_call: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "nerdit.daemon.routes.link.mark_request_side_effect",
        lambda: sections_at_call.append(h.link_section()),
    )

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert response.status_code == 200, response.text
    # Pinned before the commit: the claim seam was not yet on disk.
    assert len(sections_at_call) == 1
    assert "node_id" not in sections_at_call[0]
    assert h.link_section()["node_id"] == NODE_ID

    response = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "idem-2"})
    assert response.status_code == 200, response.text
    # Pinned before the null-delete: the seam was still on disk at call time.
    assert len(sections_at_call) == 2
    assert sections_at_call[1].get("node_id") == NODE_ID
    assert "node_id" not in h.link_section()


# ---------------------------------------------------------------------------
# PR #115 review — round 2 regressions
# ---------------------------------------------------------------------------


def test_unlink_never_wipes_a_key_on_a_never_linked_daemon(tmp_path: Path) -> None:
    """A no-op unlink must not destroy a pre-provisioned key (round-2 P1).

    An initially unlinked daemon booted with a custom ``[link].key_file``
    pointing at an operator-provisioned key: the idempotent-no-op unlink used
    to treat that boot path as enrolled and delete it.
    """
    provisioned = tmp_path / "provisioned.key"
    provisioned.write_text("operator-property\n")
    h = _build(tmp_path, link={"key_file": str(provisioned)})
    h.app.state.settings.link = LinkSettings(key_file=str(provisioned))

    response = h.client.delete("/api/link", headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["was_linked"] is False
    assert response.json()["key_removed"] is False
    assert provisioned.read_text() == "operator-property\n"


def test_a_200_echoing_the_code_as_identity_is_refused(tmp_path: Path) -> None:
    """Success-path fields are shape-validated, not merely null-checked (P2).

    A nonconforming endpoint answering 200 with the submitted code in
    ``node_id``/``node_slug`` must not see it persisted, returned, audited, or
    logged — a link code can pass neither the UUID nor the DNS-label shape.
    """
    cloud = FakeCloud(payload={"node_id": CODE, "node_slug": CODE})
    h = _build(tmp_path, cloud=cloud)

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_protocol"
    assert CODE not in response.text
    assert h.link_section() == {}
    assert CODE not in json.dumps(h.audit_rows(), default=str)

    # The frozen shapes still pass end to end.
    ok = _build(tmp_path / "ok", cloud=FakeCloud())
    assert ok.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200


def test_a_slug_shaped_like_anything_else_is_refused(tmp_path: Path) -> None:
    """Uppercase, dots, or over-length slugs all fail the DNS-label shape."""
    for bad in ("Tower", "a.b", "-lead", "x" * 64):
        h = _build(
            tmp_path / bad.replace(".", "_"),
            cloud=FakeCloud(payload={"node_id": NODE_ID, "node_slug": bad}),
        )
        response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
        assert response.status_code == 502, bad


def test_the_exchange_has_a_wall_clock_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow-dripping endpoint cannot hold the mutation lock forever (P2).

    The scalar httpx timeout is per-read; the asyncio deadline bounds the whole
    hop. With the deadline shrunk, a transport that stalls beyond it maps to
    ``link.cloud_unreachable`` instead of hanging the claim.
    """
    import asyncio as _asyncio

    monkeypatch.setattr("nerdit.daemon.routes.link._CLAIM_TOTAL_TIMEOUT_S", 0.05)

    async def stall(request: httpx.Request) -> httpx.Response:
        await _asyncio.sleep(30)
        return httpx.Response(200, json={"node_id": NODE_ID, "node_slug": SLUG})

    h = _build(tmp_path)
    h.app.state.link_claim_transport = httpx.MockTransport(stall)

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_unreachable"


def test_audit_records_the_api_host_never_the_full_url(tmp_path: Path) -> None:
    """A path-embedded credential in --api-url stays out of every surface (P2).

    Round 3 tightened this from "audit the netloc" to "refuse any path at the
    schema": the claim endpoint is always ``{origin}/api/link/claim``, and a
    path is where operators park bearer tokens — which would otherwise ride
    into the audit row AND httpx's INFO request log.
    """
    h = _build(tmp_path, cloud=FakeCloud(status=404, error_code="link_code_invalid"))
    secret_path_url = "https://app.example.test/hook-bearer-t0ken"

    response = h.client.post(
        "/api/link/claim",
        json=_claim_body(api_url=secret_path_url),
        headers=_ADMIN,
    )

    assert response.status_code == 422
    assert "hook-bearer-t0ken" not in json.dumps(h.audit_rows(), default=str)
    assert not h.cloud.called

    # A refused claim (the row that keeps the initial params) audits the
    # HOST, never a full URL; a successful claim replaces them with the
    # identity set, which carries no URL at all.
    refused = _build(
        tmp_path / "refused", cloud=FakeCloud(status=404, error_code="link_code_invalid")
    )
    assert (
        refused.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code
        == 409
    )
    row = json.loads(refused.audit_rows()[0]["params_redacted"])
    assert row.get("api_host") == "app.example.test"
    assert "api_url" not in row


def test_relay_url_with_an_unusable_port_is_a_422_before_the_cloud(tmp_path: Path) -> None:
    """``wss://relay:notaport/link`` must never survive to a committed claim (P2)."""
    cloud = FakeCloud()
    h = _build(tmp_path, cloud=cloud)

    for bad in ("wss://relay.example.test:notaport/link", "wss://relay.example.test:0/link"):
        response = h.client.post("/api/link/claim", json=_claim_body(relay_url=bad), headers=_ADMIN)
        assert response.status_code == 422, bad
    assert not cloud.called
    assert h.link_section() == {}


# ---------------------------------------------------------------------------
# PR #115 review — round 3 regressions
# ---------------------------------------------------------------------------


def test_a_shape_conforming_identity_equal_to_the_code_is_refused(tmp_path: Path) -> None:
    """The shapes overlap the code space; equality with the code is fatal (P2).

    ``secret-code`` is a valid DNS label and a code can be a lowercase UUID —
    shape checks alone would let a hostile endpoint round-trip the secret.
    """
    for code, payload in (
        ("secret-code", {"node_id": NODE_ID, "node_slug": "secret-code"}),
        (NODE_ID, {"node_id": NODE_ID, "node_slug": SLUG}),  # code IS the uuid
    ):
        h = _build(tmp_path / code[:8], cloud=FakeCloud(payload=payload))
        response = h.client.post("/api/link/claim", json=_claim_body(code=code), headers=_ADMIN)
        assert response.status_code == 502, code
        assert code not in response.text
        assert h.link_section() == {}
        assert code not in json.dumps(h.audit_rows(), default=str)


def test_a_machine_shaped_code_echo_in_cloud_code_is_degraded(tmp_path: Path) -> None:
    """A lowercase code passes the token regex; equality still refuses it (P2)."""
    h = _build(tmp_path, cloud=FakeCloud(status=404, error_code="secretcode"))

    response = h.client.post("/api/link/claim", json=_claim_body(code="secretcode"), headers=_ADMIN)

    assert response.status_code == 409
    assert response.json()["cloud_code"] == "http_404"
    assert "secretcode" not in json.dumps(h.audit_rows(), default=str)


def test_a_failed_key_wipe_is_retryable_via_the_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient wipe failure must not orphan the enrolled key forever (P2).

    The first unlink clears ``node_id`` but the wipe fails; without the stamp
    surviving, the retry sees ``was_linked: false`` and skips the wipe path.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    assert h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200
    assert h.key_file.exists()

    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == h.key_file and calls["n"] == 0:
            calls["n"] += 1
            raise OSError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    first = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-1"})
    assert first.status_code == 200
    assert first.json() == {
        "was_linked": True,
        "tunnel_stopped": False,
        "key_removed": False,
        "pending_wipes": 1,
        "requires_restart": True,
    }
    assert h.key_file.exists()
    # Round 5 moved the retry marker from the stamp to the pending set (a
    # subsequent claim overwrites the stamp; the set survives it).
    assert h.app.state.link_pending_key_wipes == {str(h.key_file)}

    second = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-2"})
    assert second.status_code == 200
    assert second.json()["was_linked"] is False
    assert second.json()["key_removed"] is True  # the retry finished the wipe
    assert not h.key_file.exists()
    assert h.app.state.link_pending_key_wipes == set()


def test_a_pending_data_dir_move_refuses_the_claim(tmp_path: Path) -> None:
    """A claim must not straddle a staged data_dir change (P2).

    The key would enroll under the OLD directory while the required restart
    resolves the NEW one — a different verifier and terminal relay auth.
    """
    cloud = FakeCloud()
    h = _build(tmp_path, cloud=cloud)
    h.store.commit(h.store.stage("nerdit", {"data_dir": str(tmp_path / "elsewhere")}))

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 409
    assert response.json()["code"] == "link.restart_required"
    assert not cloud.called
    assert not h.key_file.exists()  # no key was minted for a doomed enrollment

    # Storing the data_dir the daemon actually booted with is NOT a pending
    # move — the explicit-value path must compare, not blanket-refuse.
    ok = _build(tmp_path / "ok", cloud=FakeCloud())
    ok.store.commit(ok.store.stage("nerdit", {"data_dir": str(ok.key_file.parent.parent)}))
    assert ok.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200


# ---------------------------------------------------------------------------
# PR #115 review — round 4 regressions
# ---------------------------------------------------------------------------


def test_the_code_alphabet_is_pinned_and_normalized(tmp_path: Path) -> None:
    """The anti-reflection keystone: codes are UPPERCASE, echoes are lowercase.

    A lowercased paste is normalized (Crockford is case-insensitive), so the
    cloud receives the canonical uppercase form; anything outside the issued
    alphabet is a 422 that never reaches the cloud and never echoes the value.
    """
    cloud = FakeCloud()
    h = _build(tmp_path, cloud=cloud)

    response = h.client.post(
        "/api/link/claim", json=_claim_body(code="dev-link-code-000001"), headers=_ADMIN
    )
    assert response.status_code == 200, response.text
    assert cloud.requests[0]["code"] == "DEV-LINK-CODE-000001"

    for bad in ("short", "with space CODE", "code_with_underscores", "päste-1"):
        refused = _build(tmp_path / str(hash(bad)), cloud=FakeCloud())
        response = refused.client.post(
            "/api/link/claim", json=_claim_body(code=bad), headers=_ADMIN
        )
        assert response.status_code == 422, bad
        assert bad not in response.text
        assert not refused.cloud.called


def test_an_identity_or_cloud_code_embedding_the_code_is_refused(tmp_path: Path) -> None:
    """Containment, not equality: an EMBEDDED lowercased echo is still the secret.

    ``x-secret-code`` carries ``secret-code`` verbatim; uppercasing it back is
    trivial, so embedding must refuse exactly like equality (round 4).
    """
    slug_echo = _build(
        tmp_path / "slug",
        cloud=FakeCloud(payload={"node_id": NODE_ID, "node_slug": "x-secret-code"}),
    )
    response = slug_echo.client.post(
        "/api/link/claim", json=_claim_body(code="SECRET-CODE"), headers=_ADMIN
    )
    assert response.status_code == 502
    assert "secret-code" not in response.text.lower() or "SECRET-CODE" not in response.text
    assert slug_echo.link_section() == {}

    token_echo = _build(tmp_path / "token", cloud=FakeCloud(status=404, error_code="x_secretcode"))
    response = token_echo.client.post(
        "/api/link/claim", json=_claim_body(code="SECRETCODE"), headers=_ADMIN
    )
    assert response.status_code == 409
    assert response.json()["cloud_code"] == "http_404"


def test_rejected_urls_are_scrubbed_from_422_echoes(tmp_path: Path) -> None:
    """A REJECTED credential-bearing URL must not ride back in the 422 input echo."""
    h = _build(tmp_path)

    response = h.client.post(
        "/api/link/claim",
        json=_claim_body(api_url="https://app.example.test/hook-bearer-t0ken"),
        headers=_ADMIN,
    )
    assert response.status_code == 422
    assert "hook-bearer-t0ken" not in response.text

    response = h.client.post(
        "/api/link/claim",
        json=_claim_body(relay_url="wss://user:s3cretpw@relay.example.test/link"),
        headers=_ADMIN,
    )
    assert response.status_code == 422
    assert "s3cretpw" not in response.text


async def test_a_mid_claim_key_path_config_change_refuses_the_commit(tmp_path: Path) -> None:
    """The round-4 P1: config PUTs do not share the link lock.

    A ``[link].key_file`` re-stage landing while the claim awaits the cloud
    must refuse with ``link.config_changed`` and commit NOTHING — committing
    would bind the cloud identity to a config whose next restart resolves a
    different key than the enrolled verifier.
    """
    import asyncio

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"node_id": NODE_ID, "node_slug": SLUG})

    h = _build(tmp_path)
    h.app.state.link_claim_transport = httpx.MockTransport(handler)

    transport = httpx.ASGITransport(app=h.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        claim = asyncio.create_task(
            client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        # Another admin re-stages the key path while the claim is parked.
        h.store.commit(h.store.stage("link", {"key_file": str(tmp_path / "other.key")}))
        release.set()
        response = await asyncio.wait_for(claim, timeout=5)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.config_changed"
    section = h.link_section()
    assert "node_id" not in section  # nothing was committed
    assert section["key_file"] == str(tmp_path / "other.key")  # the PUT survives


def test_a_boot_loaded_identity_wipe_failure_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 4: the stamp is SET on a failed wipe, not merely preserved.

    Post-restart there is no claim stamp; the first unlink clears ``node_id``
    and its wipe fails, so without stamping the target the retry would skip
    the block and orphan the boot-loaded key forever.
    """
    h = _build(
        tmp_path,
        link={"enabled": True, "relay_url": RELAY, "node_id": NODE_ID, "slug": SLUG},
    )
    h.key_file.parent.mkdir(parents=True)
    h.key_file.write_text("boot-loaded-key\n")
    assert getattr(h.app.state, "link_enrolled_key_file", None) is None  # post-restart

    real_unlink = Path.unlink
    calls = {"n": 0}

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == h.key_file and calls["n"] == 0:
            calls["n"] += 1
            raise OSError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    first = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-1"})
    assert first.status_code == 200
    assert first.json()["key_removed"] is False
    assert h.app.state.link_pending_key_wipes == {str(h.key_file)}  # marker SET

    second = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-2"})
    assert second.status_code == 200
    assert second.json()["key_removed"] is True
    assert not h.key_file.exists()
    assert h.app.state.link_pending_key_wipes == set()


# ---------------------------------------------------------------------------
# PR #115 review — round 5 regressions
# ---------------------------------------------------------------------------


def test_a_hex_code_embedded_in_the_node_uuid_is_refused(tmp_path: Path) -> None:
    """Containment covers node_id too: hex codes can embed in UUID hex (P2)."""
    h = _build(
        tmp_path,
        cloud=FakeCloud(
            payload={"node_id": "abcdef00-0000-4000-8000-000000000000", "node_slug": SLUG}
        ),
    )

    response = h.client.post("/api/link/claim", json=_claim_body(code="ABCDEF"), headers=_ADMIN)

    assert response.status_code == 502
    assert "abcdef" not in response.text.lower()
    assert h.link_section() == {}


def test_no_enable_persists_false_over_a_preexisting_true(tmp_path: Path) -> None:
    """``--no-enable`` must WRITE false, not merely skip writing true (P2).

    An enabled-but-unlinked daemon (the documented ordinary state) claiming
    with ``--no-enable`` used to keep ``enabled = true``, so the required
    restart started the tunnel the operator declined.
    """
    h = _build(tmp_path, link={"enabled": True, "relay_url": RELAY}, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(enable=False), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert h.link_section()["enabled"] is False


def test_a_pending_wipe_survives_an_intervening_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed-wipe marker outlives a re-stage + re-link to another key (P2).

    Fail the wipe of key A, re-stage ``key_file`` to B, claim again (which
    overwrites the enrollment stamp with B), then unlink: BOTH B and the
    still-pending A must go.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    assert h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200
    key_a = h.key_file

    real_unlink = Path.unlink
    fail_a = {"on": True}

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == key_a and fail_a["on"]:
            raise OSError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    first = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-1"})
    assert first.json()["key_removed"] is False
    assert h.app.state.link_pending_key_wipes == {str(key_a)}

    key_b = h.config_path.parent / "b.key"
    h.store.commit(h.store.stage("link", {"key_file": str(key_b)}))
    assert h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200
    assert h.app.state.link_pending_key_wipes == {str(key_a)}  # claim left it alone
    assert key_b.exists()

    fail_a["on"] = False
    final = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-2"})
    assert final.status_code == 200
    assert final.json()["key_removed"] is True  # B, the enrolled primary
    assert not key_b.exists()
    assert not key_a.exists()  # the pending retry finally landed
    assert h.app.state.link_pending_key_wipes == set()


# ---------------------------------------------------------------------------
# PR #115 review — round 6 regressions
# ---------------------------------------------------------------------------


def test_rejected_schemes_do_not_echo_in_the_message(tmp_path: Path) -> None:
    """A credential pasted AS the scheme must not ride back in the 422 msg (P2).

    The input-field redaction already scrubs ``input``; this pins the
    validator-generated MESSAGE, which used to interpolate the scheme.
    """
    h = _build(tmp_path)
    for field, url in (
        ("api_url", "sk-s3cret-token://app.example.test"),
        ("relay_url", "sk-s3cret-token://relay.example.test/link"),
    ):
        response = h.client.post(
            "/api/link/claim", json=_claim_body(**{field: url}), headers=_ADMIN
        )
        assert response.status_code == 422, field
        assert "sk-s3cret-token" not in response.text, field


def test_a_live_manager_gates_the_wipe_even_when_config_says_unlinked(tmp_path: Path) -> None:
    """A staged ``node_id = null`` must not shield the live identity's key (P2).

    The booted manager proves an enrolled identity is live; unlink must wipe
    its boot-loaded key even though the effective config already reads
    unlinked.
    """
    manager = FakeLinkManager()
    h = _build(tmp_path, link={"enabled": True, "relay_url": RELAY}, link_manager=manager)
    h.key_file.parent.mkdir(parents=True)
    h.key_file.write_text("live-but-config-unlinked\n")

    response = h.client.delete("/api/link", headers=_ADMIN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["was_linked"] is False
    assert body["tunnel_stopped"] is True
    assert body["key_removed"] is True
    assert not h.key_file.exists()
    manager.stop.assert_awaited_once()


def test_key_removed_true_does_not_conceal_stuck_pending_wipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``pending_wipes`` reports what "key deleted" alone would conceal (P2)."""
    h = _build(tmp_path, cloud=FakeCloud())
    assert h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200
    key_a = h.key_file

    real_unlink = Path.unlink

    def stubborn_a(self: Path, missing_ok: bool = False) -> None:
        if self == key_a:
            raise OSError(13, "Permission denied")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stubborn_a)

    first = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-1"})
    assert first.json()["pending_wipes"] == 1

    key_b = h.config_path.parent / "b.key"
    h.store.commit(h.store.stage("link", {"key_file": str(key_b)}))
    assert h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN).status_code == 200

    # B wipes cleanly, A stays stuck: the response must say BOTH truths.
    second = h.client.delete("/api/link", headers={**_ADMIN, "Idempotency-Key": "u-2"})
    body = second.json()
    assert body["key_removed"] is True
    assert body["pending_wipes"] == 1
    assert key_a.exists()
    row = json.loads(h.audit_rows()[-1]["params_redacted"])
    assert row["pending_wipes"] == 1
    assert str(key_a) not in json.dumps(row)  # the count, never the path


# ---------------------------------------------------------------------------
# P26 WP-H — the hosted base domain: claim side
# ---------------------------------------------------------------------------


def test_claim_persists_the_hosted_domain_when_the_cloud_sends_one(tmp_path: Path) -> None:
    """The claim is the first chance to learn the suffix; it takes it.

    ``nodes_base_domain`` is additive on the cloud's claim response, so a node
    claimed against a P26 cloud can share immediately without a second
    round-trip through ``nerdit link refresh``.
    """
    cloud = FakeCloud(payload={"node_id": NODE_ID, "node_slug": SLUG, "nodes_base_domain": DOMAIN})
    h = _build(tmp_path, cloud=cloud)

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["nodes_base_domain"] == DOMAIN
    assert h.link_section()["nodes_base_domain"] == DOMAIN
    assert LinkSettings(**h.store.effective_section("link")).nodes_base_domain == DOMAIN
    assert json.loads(h.audit_rows()[-1]["params_redacted"])["nodes_base_domain"] == DOMAIN


def test_claim_against_a_pre_p26_cloud_leaves_the_domain_untouched(tmp_path: Path) -> None:
    """Absence is tolerated, and it never NULLS a domain the node already had.

    A cloud that stopped sending the key (a rollback, a misconfigured origin)
    must not silently disable every hosted share on the node — forgetting the
    suffix is indistinguishable from unsharing everything.
    """
    h = _build(tmp_path, link={"nodes_base_domain": DOMAIN}, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert h.link_section()["nodes_base_domain"] == DOMAIN
    # The view reports what the daemon HOLDS, not merely what this hop returned.
    assert response.json()["nodes_base_domain"] == DOMAIN
    assert json.loads(h.audit_rows()[-1]["params_redacted"])["nodes_base_domain"] is None


def test_unlink_clears_the_hosted_domain_so_a_re_claim_cannot_inherit_it(tmp_path: Path) -> None:
    """The domain is a claim result, not endpoint config: unlink drops it (D10).

    Otherwise a node unlinked from a P26 cloud and re-claimed against one whose
    claim omits ``nodes_base_domain`` would pair the NEW slug with the OLD
    cloud's suffix and advertise hosted URLs that cannot route (PR #128 review).
    """
    h = _build(
        tmp_path,
        link={**_LINKED, "nodes_base_domain": DOMAIN},
        cloud=FakeCloud(),
    )
    h.key_file.parent.mkdir(parents=True, exist_ok=True)
    h.key_file.write_text("not-a-real-key\n")

    unlinked = h.client.delete("/api/link", headers=_ADMIN)
    assert unlinked.status_code == 200, unlinked.text
    section = h.link_section()
    assert "nodes_base_domain" not in section
    assert "slug" not in section
    assert section["relay_url"] == RELAY

    reclaimed = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert reclaimed.status_code == 200, reclaimed.text
    assert reclaimed.json()["nodes_base_domain"] is None
    assert "nodes_base_domain" not in h.link_section()


def test_claim_on_a_fresh_node_without_a_domain_reports_null(tmp_path: Path) -> None:
    """Nothing is invented: the key is simply absent until a refresh writes it."""
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 200, response.text
    assert response.json()["nodes_base_domain"] is None
    assert "nodes_base_domain" not in h.link_section()


@pytest.mark.parametrize(
    ("domain", "why"),
    [
        ("Nodes.Example", "uppercase is refused, never normalized"),
        ("*.nodes.example", "a wildcard is not a concrete name"),
        ("nodes.example:443", "a port is not part of a DNS name"),
        ("https://nodes.example", "a URL is not a DNS name"),
        ("", "empty is not an omission"),
        (17, "not even a string"),
        # (PR review, P26 WP-H) The shared ``[proxy]`` grammar accepts an IPv4
        # literal on purpose (``hostname_override`` takes a "name/IP"), so this
        # field refuses one itself: no label can be delegated below an address,
        # and the cloud must not be able to make this node advertise
        # ``https://demo--gpu-box.10.0.0.1/``.
        ("10.0.0.1", "an IP literal carries no label below it"),
        (f"{CODE.lower()}.example", "an echo of the submitted link code"),
    ],
)
def test_claim_refuses_a_malformed_hosted_domain(tmp_path: Path, domain: Any, why: str) -> None:
    """Present-but-malformed is a protocol error, never a silent drop.

    A claim that persisted a bad suffix would advertise URLs nothing resolves,
    and the last case is the reflection arm: a nonconforming ``--api-url``
    endpoint echoing the link code back as a "domain" must not see that secret
    written into config.
    """
    cloud = FakeCloud(payload={"node_id": NODE_ID, "node_slug": SLUG, "nodes_base_domain": domain})
    h = _build(tmp_path, cloud=cloud)

    response = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)

    assert response.status_code == 502, why
    assert response.json()["code"] == "link.cloud_protocol"
    # The refusal never echoes the offending value, and nothing was persisted.
    # (Only the string cases can echo. The empty one is vacuous — there is
    # nothing to reflect — and the non-string one would false-positive against
    # the request id, but both still have to be REFUSED rather than coerced.)
    if isinstance(domain, str) and domain:
        assert domain not in response.text
    assert h.link_section() == {}


# ---------------------------------------------------------------------------
# P26 WP-H — POST /link/refresh
# ---------------------------------------------------------------------------

_LINKED = {"node_id": NODE_ID, "slug": SLUG, "relay_url": RELAY}


def _refresh(h: Harness, *, key: str = "r-1") -> Any:
    return h.client.post(
        "/api/link/refresh",
        json={"api_url": "https://app.example.test"},
        headers={**_ADMIN, "Idempotency-Key": key},
    )


def test_refresh_learns_and_persists_the_hosted_domain(tmp_path: Path) -> None:
    """The whole point: a node linked before P26 can learn its suffix."""
    meta = FakeMetadata()
    h = _build(tmp_path, link=dict(_LINKED), cloud=meta)

    response = _refresh(h)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["nodes_base_domain"] == DOMAIN
    assert body["node_id"] == NODE_ID
    assert body["slug"] == SLUG
    assert body["changed"] is True
    assert body["requires_restart"] is True
    assert "link.nodes_base_domain" in body["restart_keys"]

    assert meta.urls == ["https://app.example.test/api/link/metadata"]
    # A read, not a write: nothing left the machine in the body.
    assert meta.bodies == [b""]
    assert h.link_section()["nodes_base_domain"] == DOMAIN
    # ``api_url`` stays per-invocation (D5) — never persisted, here either.
    assert "app.example.test" not in h.config_path.read_text()

    row = h.audit_rows()[-1]
    assert row["action"] == "link.refreshed"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    params = json.loads(row["params_redacted"])
    # The host survives the success path (the claim's api_host rule): the
    # per-request params set before the fetch are merged, not replaced.
    assert params == {
        "api_host": "app.example.test",
        "node_id": NODE_ID,
        "nodes_base_domain": DOMAIN,
        "changed": True,
    }


def test_refresh_is_idempotent_and_does_not_ask_for_a_restart(tmp_path: Path) -> None:
    """A refresh that rewrote the same value must not bounce a healthy daemon.

    ``[link]`` is restart-required as a whole section, so ``requires_restart``
    has to follow ``changed`` rather than being hardcoded true like the claim's
    — otherwise every routine refresh costs a tunnel drop.
    """
    h = _build(tmp_path, link={**_LINKED, "nodes_base_domain": DOMAIN}, cloud=FakeMetadata())

    body = _refresh(h).json()

    assert body["changed"] is False
    assert body["requires_restart"] is False
    assert body["restart_keys"] == []
    assert h.link_section()["nodes_base_domain"] == DOMAIN


def test_refresh_on_an_unlinked_node_409s_before_the_cloud(tmp_path: Path) -> None:
    """The suffix is only meaningful beside a slug (and the code is never spent)."""
    meta = FakeMetadata()
    h = _build(tmp_path, cloud=meta)

    response = _refresh(h)

    assert response.status_code == 409
    assert response.json()["code"] == "link.not_linked"
    assert not meta.called
    assert h.link_section() == {}


def test_refresh_against_a_pre_p26_cloud_is_a_structured_refusal(tmp_path: Path) -> None:
    """A 404 is "your console is older than your daemon", not a generic 502."""
    h = _build(tmp_path, link=dict(_LINKED), cloud=FakeMetadata(status=404))

    response = _refresh(h)

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "link.refresh_unsupported"
    assert "hosted metadata" in body["message"]
    assert "nodes_base_domain" in body["hint"]
    assert "nodes_base_domain" not in h.link_section()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_refresh_maps_every_other_cloud_status_to_a_502(tmp_path: Path, status: int) -> None:
    """One code for "the cloud could not answer", with its token attached."""
    h = _build(tmp_path, link=dict(_LINKED), cloud=FakeMetadata(status=status, error_code="nope"))

    response = _refresh(h)

    assert response.status_code == 502
    body = response.json()
    assert body["code"] == "link.cloud_error"
    # The cloud's machine token travels (there is no submitted secret on this
    # hop to guard against); the cloud's BODY never does.
    assert body["cloud_code"] == "nope"
    assert body["cloud_status"] == status
    assert "nodes_base_domain" not in h.link_section()


def test_refresh_when_the_cloud_is_unreachable_502s(tmp_path: Path) -> None:
    h = _build(tmp_path, link=dict(_LINKED), cloud=FakeMetadata(raises=True))

    response = _refresh(h)

    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_unreachable"
    assert "app.example.test" in response.json()["message"]
    assert "nodes_base_domain" not in h.link_section()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"nodes_base_domain": None},
        {"nodes_base_domain": "Nodes.Example"},
        {"nodes_base_domain": "*.nodes.example"},
        {"nodes_base_domain": "10.0.0.1"},
        {"nodes_base_domain": ""},
        {"nodes_base_domain": 17},
        ["nodes.example"],
        "nodes.example",
    ],
)
def test_refresh_refuses_a_response_that_is_not_a_hosted_domain(
    tmp_path: Path, payload: Any
) -> None:
    """A 200 with the wrong shape is a protocol error, and nothing is written."""
    h = _build(tmp_path, link=dict(_LINKED), cloud=FakeMetadata(payload=payload))

    response = _refresh(h)

    assert response.status_code == 502
    assert response.json()["code"] == "link.cloud_protocol"
    assert "nodes_base_domain" not in h.link_section()


def test_refresh_requires_admin(tmp_path: Path) -> None:
    """D2: refreshing commits ``[link]`` config, so it is custody like the claim."""
    meta = FakeMetadata()
    h = _build(tmp_path, link=dict(_LINKED), cloud=meta)

    response = h.client.post(
        "/api/link/refresh",
        json={"api_url": "https://app.example.test"},
        headers={**_SUBMITTER, "Idempotency-Key": "r-1"},
    )

    assert response.status_code == 403
    assert not meta.called


def test_refresh_requires_an_idempotency_key(tmp_path: Path) -> None:
    """D3: it commits config, so the in-route gate applies (before the cloud hop)."""
    meta = FakeMetadata()
    h = _build(tmp_path, link=dict(_LINKED), cloud=meta)

    response = h.client.post(
        "/api/link/refresh",
        json={"api_url": "https://app.example.test"},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )

    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    assert not meta.called


def test_refresh_rejects_a_dirty_api_url_with_the_claim_rules(tmp_path: Path) -> None:
    """One rule set for both ops: a URL the claim refuses, the refresh refuses."""
    meta = FakeMetadata()
    h = _build(tmp_path, link=dict(_LINKED), cloud=meta)

    response = h.client.post(
        "/api/link/refresh",
        json={"api_url": "https://u:p@app.example.test"},
        headers={**_ADMIN, "Idempotency-Key": "r-1"},
    )

    assert response.status_code == 422
    assert not meta.called


def test_refresh_never_reflects_the_cloud_body(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The never-leak sweep, extended to the refresh path (P26 WP-H).

    There is no link code on this hop, so what must not escape is the CLOUD's
    body: an operator-supplied ``--api-url`` is untrusted in both directions,
    and a refusal that quoted its payload would be the same reflection channel
    the claim's ``_cloud_code`` guard exists to close.
    """
    caplog.set_level(logging.DEBUG)
    marker = CLOUD_NOISE["detail"]
    clouds = [
        FakeMetadata(status=404, payload=CLOUD_NOISE),
        FakeMetadata(status=500),
        FakeMetadata(status=403),
        FakeMetadata(payload={"nodes_base_domain": marker + " not a domain"}),
        FakeMetadata(payload=CLOUD_NOISE),
        FakeMetadata(raises=True),
    ]
    for index, cloud in enumerate(clouds):
        h = _build(tmp_path / f"refresh-{index}", link=dict(_LINKED), cloud=cloud)
        response = _refresh(h)
        assert response.status_code >= 400
        assert marker not in response.text
        assert marker not in json.dumps(h.audit_rows(), default=str)
        assert marker not in h.config_path.read_text()

    assert marker not in caplog.text
