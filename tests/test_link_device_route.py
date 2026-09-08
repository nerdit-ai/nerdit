"""Test device-code linking and pre-auth claims against the shipped cloud contract.

Pin field names, status codes, Crockford alphabet, 900-second TTL and 5/10-second
poll cadence. Device codes are exchanged only with the cloud, never exposed to
CLI callers; user codes are display material. Neither code nor pre-auth keys
may appear in durable records.

Preserve the four-step poll preflight order: retries after commit return
already_linked, and concurrent starts cannot cross-talk. Approved polls reuse
the claim staging tail, including config, MCP enablement and audit changes.

Use real auth/audit/error middleware and a temporary ConfigStore, with mocked
queries and cloud transport. Sweep successful and refused flows for leaks.
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
from nerdit.core.link.identity import credential_fingerprint, load_or_create_identity
from nerdit.daemon.audit import AuditMiddleware, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import NO_BODY_CACHE_ACTIONS, NO_BODY_HASH_ACTIONS
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import _DeviceMint, _validated_verification_uri
from nerdit.daemon.routes.link import router as link_router
from nerdit.db.models import ApiToken, TokenRole

ADMIN_TOKEN = "admin-raw-token"  # noqa: S105 - a test literal
SUBMITTER_TOKEN = "scoped-raw"  # noqa: S105 - a test literal
API_URL = "https://app.example.test"
RELAY = "wss://relay.example.test/link"
NODE_ID = "00000000-0000-4000-8000-00000000000a"
SLUG = "corvid-mesa"
DOMAIN = "nodes.example"

#: 32 Crockford symbols — the shipped ``generate_device_code`` shape. This is
#: the value that must never appear anywhere but the outbound poll body.
DEVICE_CODE = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # noqa: S105 - a test literal
#: 12 Crockford symbols in the canonical ``XXXX-XXXX-XXXX`` grouping.
USER_CODE = "7Q3M-X2VF-9KHT"  # noqa: S105 - a test literal
#: ``nk_`` + 32 Crockford symbols — the shipped ``valid_preauth_key`` shape.
PREAUTH_KEY = "nk_0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # noqa: S105 - a test literal
#: A marker the cloud puts in a refusal body; nothing may reflect it.
CLOUD_NOISE = {"detail": "cloud-body-marker-9f2a"}

_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": "idem-1"}
_SUBMITTER = {"Authorization": f"Bearer {SUBMITTER_TOKEN}", "Idempotency-Key": "idem-1"}


def _headers(nonce: str) -> dict[str, str]:
    """Admin headers with a fresh Idempotency-Key — the CLI mints one per poll."""
    return {"Authorization": f"Bearer {ADMIN_TOKEN}", "Idempotency-Key": nonce}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class FakeCloud:
    """A scripted stand-in for the four ``/api/link/*`` endpoints a daemon dials.

    One transport handler for all of them, dispatching on path, because a
    device flow touches two of them in one test and the pre-auth path shares
    the claim route's plumbing. Every request is recorded so a test can assert
    what left the machine — and, just as often, that nothing did.

    ``credential_fingerprint`` is COMPUTED from the submitted reference rather
    than being a constant, exactly as the cloud computes it: the daemon refuses
    a mint whose fingerprint does not match its own identity, and a constant
    here would make that check untestable in both directions.
    """

    def __init__(  # noqa: PLR0913 - one knob per scripted endpoint answer
        self,
        *,
        mint_status: int = 201,
        mint_payload: dict[str, Any] | None = None,
        polls: list[httpx.Response | Exception] | None = None,
        claim_status: int = 200,
        claim_payload: Any = None,
        error_code: str | None = None,
        on_mint: Any = None,
        on_poll: Any = None,
    ) -> None:
        self.mint_status = mint_status
        self.mint_payload = mint_payload
        self.polls = list(polls or [])
        self.claim_status = claim_status
        self.claim_payload = (
            claim_payload
            if claim_payload is not None
            else {"node_id": NODE_ID, "node_slug": SLUG, "nodes_base_domain": DOMAIN}
        )
        self.error_code = error_code
        self.on_mint = on_mint
        self.on_poll = on_poll
        self.mints: list[dict[str, Any]] = []
        self.poll_bodies: list[dict[str, Any]] = []
        self.claims: list[dict[str, Any]] = []
        self.preauths: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        body = json.loads(request.content)
        path = request.url.path
        if path.endswith("/api/link/device"):
            self.mints.append(body)
            if self.on_mint is not None:
                self.on_mint()
            if self.mint_status >= 400:
                return httpx.Response(self.mint_status, json=self._error_body())
            return httpx.Response(self.mint_status, json=self._mint_body(body))
        if path.endswith("/api/link/device/poll"):
            self.poll_bodies.append(body)
            if self.on_poll is not None:
                self.on_poll()
            scripted = self.polls.pop(0) if self.polls else self._approved()
            if isinstance(scripted, Exception):
                raise scripted
            return scripted
        if path.endswith("/api/link/preauth"):
            self.preauths.append(body)
        else:
            self.claims.append(body)
        if self.claim_status >= 400:
            return httpx.Response(self.claim_status, json=self._error_body())
        return httpx.Response(self.claim_status, json=self.claim_payload)

    def _error_body(self) -> Any:
        return {"error": {"code": self.error_code}} if self.error_code else CLOUD_NOISE

    def _mint_body(self, submitted: dict[str, Any]) -> dict[str, Any]:
        if self.mint_payload is not None:
            return self.mint_payload
        return {
            "device_code": DEVICE_CODE,
            "user_code": USER_CODE,
            "verification_uri": f"{API_URL}/link",
            "verification_uri_complete": f"{API_URL}/link#{USER_CODE}",
            "credential_fingerprint": credential_fingerprint(submitted["credential_reference"]),
            "expires_in": 900,
            "interval": 5,
        }

    def _approved(self) -> httpx.Response:
        return httpx.Response(200, json=self.claim_payload)

    @property
    def polled(self) -> bool:
        return bool(self.poll_bodies)


def pending(interval: int = 5) -> httpx.Response:
    """The shipped ``202 authorization_pending`` body."""
    return httpx.Response(202, json={"status": "authorization_pending", "interval": interval})


def slow_down(interval: int = 10) -> httpx.Response:
    """The shipped ``202 slow_down`` body — RFC 8628's back-off answer."""
    return httpx.Response(202, json={"status": "slow_down", "interval": interval})


def refusal(status: int, code: str) -> httpx.Response:
    """One cloud refusal, with the noise a daemon must never reflect."""
    return httpx.Response(status, json={"error": {"code": code}, **CLOUD_NOISE})


class FakeLinkManager:
    """What ``app.state.link_manager`` looks like to the auth middleware."""

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
    cloud: FakeCloud
    app: FastAPI
    key_file: Path

    def raw(self) -> dict[str, Any]:
        with open(self.config_path, "rb") as handle:
            return tomllib.load(handle)

    def link_section(self) -> dict[str, Any]:
        return self.raw().get("link", {})

    def audit_rows(self) -> list[dict[str, Any]]:
        return [call.kwargs for call in self.queries.insert_audit_log.await_args_list]

    def audit_text(self) -> str:
        return json.dumps(self.audit_rows(), default=str)

    def link_config(self, **values: Any) -> None:
        """Write ``[link]`` keys behind the daemon's back — a concurrent writer.

        A ``None`` value REMOVES the key rather than writing one, because TOML
        has no null and "the operator re-staged this key away" is exactly the
        mid-flight change the commit-time re-checks exist for.
        """
        document = self.raw()
        section = document.setdefault("link", {})
        for name, value in values.items():
            if value is None:
                section.pop(name, None)
            else:
                section[name] = value
        self.config_path.write_bytes(tomli_w.dumps(document).encode("utf-8"))

    def start(self, **overrides: Any) -> httpx.Response:
        body: dict[str, Any] = {"api_url": API_URL, "relay_url": RELAY}
        body.update(overrides)
        return self.client.post("/api/link/device", json=body, headers=_ADMIN)

    def poll(self, session: str, *, nonce: str = "poll-1") -> httpx.Response:
        return self.client.post(
            "/api/link/device/poll", json={"session": session}, headers=_headers(nonce)
        )


def _build(
    tmp_path: Path,
    *,
    link: dict[str, Any] | None = None,
    cloud: FakeCloud | None = None,
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
    app.state.link_manager = None
    app.state.link_device_pending = None
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
    body: dict[str, Any] = {"api_url": API_URL, "relay_url": RELAY}
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# device start
# ---------------------------------------------------------------------------


def test_device_start_requires_admin_and_an_idempotency_key_and_refuses_when_already_linked(
    tmp_path: Path,
) -> None:
    """The three refusals that must all precede the first cloud hop.

    Admin-only is structural custody (D2): the tunnel principal is a permanent
    ``submitter``, so a compromised relay cannot start a link flow through the
    tunnel it is already talking on. The in-route ``Idempotency-Key`` is D3's,
    because a mint has a durable remote effect. And already-linked is D8's,
    refused before any cloud contact so a live approval screen is never put in
    front of a human for a node that could not consume it.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    denied = h.client.post("/api/link/device", json=_claim_body(), headers=_SUBMITTER)
    assert denied.status_code == 403
    assert denied.json()["code"] == "forbidden"

    keyless = h.client.post(
        "/api/link/device",
        json=_claim_body(),
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert keyless.status_code == 400
    assert keyless.json()["code"] == "idempotency_key_required"

    # Not one of the three reached the cloud, and none of them wrote config.
    assert not h.cloud.urls
    assert h.link_section() == {}

    linked = _build(
        tmp_path / "linked",
        link={"relay_url": RELAY, "node_id": NODE_ID, "slug": SLUG},
        cloud=FakeCloud(),
    )
    before = linked.config_path.read_bytes()
    response = linked.start()
    assert response.status_code == 409
    payload = response.json()
    assert payload["code"] == "link.already_linked"
    assert "unlink" in payload["hint"]
    assert not linked.cloud.urls
    assert linked.config_path.read_bytes() == before


def test_device_start_without_any_relay_url_422s_before_the_cloud(tmp_path: Path) -> None:
    """D5, applied to the slower flow: never mint a code for a node that cannot start."""
    h = _build(tmp_path, cloud=FakeCloud())

    body = _claim_body()
    del body["relay_url"]
    response = h.client.post("/api/link/device", json=body, headers=_ADMIN)

    assert response.status_code == 422
    assert response.json()["code"] == "link.relay_url_required"
    assert not h.cloud.urls
    assert h.app.state.link_device_pending is None


def test_device_start_returns_the_user_code_and_fragment_uri_but_never_the_device_code(
    tmp_path: Path,
) -> None:
    """The response is display material plus a grantless selector — nothing else.

    The ``user_code`` and the fragment-carrying one-click URL are what the
    terminal prints (D-P34-4). The 160-bit ``device_code`` is what the daemon
    polls with, and it stays in this process: not in the body, not in the audit
    row, not on any wire the CLI can see. The three machine facts are
    daemon-local — the cloud never echoes them back — so they are asserted
    against what the mint REPORTED, not against a cloud field.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.start()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["user_code"] == USER_CODE
    # D-X16-O24: the code rides in the FRAGMENT, which never reaches a server.
    assert body["verification_uri_complete"] == f"{API_URL}/link#{USER_CODE}"
    assert "#" in body["verification_uri_complete"]
    assert "?" not in body["verification_uri_complete"]
    assert body["verification_uri"] == f"{API_URL}/link"
    assert body["expires_in"] == 900
    assert body["interval"] == 5
    # The full 64 hex; the CLI truncates to the console's preview length of 8.
    assert len(body["credential_fingerprint"]) == 64

    # The device code is nowhere the CLI can reach it.
    assert DEVICE_CODE not in response.text
    assert DEVICE_CODE not in h.audit_text()
    assert USER_CODE not in h.audit_text()

    # ...but the slot holds it, and the outbound mint reported this machine.
    slot = h.app.state.link_device_pending
    assert slot is not None
    assert slot.device_code == DEVICE_CODE
    assert slot.session == body["session"]
    # Redacted by construction: a dataclass repr would put the code into any
    # traceback frame or log record that touched the slot.
    assert DEVICE_CODE not in repr(slot)

    sent = h.cloud.mints[0]
    assert set(sent) == {"credential_reference", "hostname_hint", "daemon_version", "os"}
    private_key_b64 = h.key_file.read_text().strip()
    assert private_key_b64 not in response.text
    assert sent["daemon_version"] == body["daemon_version"]
    assert sent["os"] == body["os"]
    assert sent["hostname_hint"] == body["hostname"]

    # Nothing is persisted by a start: the flow commits at the poll, not here.
    assert h.link_section() == {}


def test_device_start_asserts_the_cloud_fingerprint_matches_the_local_identity(
    tmp_path: Path,
) -> None:
    """A fingerprint that is not this node's means the peer is not our protocol peer.

    Both sides compute ``sha256`` over the full ``ed25519:…`` verifier, so a
    mismatch is not a formatting disagreement — and the eight characters the
    terminal is about to tell an operator to compare against the console would
    be a lie. It is a cloud fault, so ``502 link.cloud_error``, and no slot is
    parked for a flow that cannot be trusted.
    """
    cloud = FakeCloud(
        mint_payload={
            "device_code": DEVICE_CODE,
            "user_code": USER_CODE,
            "verification_uri": f"{API_URL}/link",
            "verification_uri_complete": f"{API_URL}/link#{USER_CODE}",
            "credential_fingerprint": "f" * 64,
            "expires_in": 900,
            "interval": 5,
        }
    )
    h = _build(tmp_path, cloud=cloud)

    response = h.start()

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "link.cloud_error"
    assert h.app.state.link_device_pending is None
    assert h.link_section() == {}


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("user_code", "not a code", "the terminal prints this for a human to retype"),
        ("verification_uri_complete", "javascript:alert(1)", "the operator clicks this"),
        ("verification_uri", "http://evil.example.test/link", "plain http off-box"),
        ("interval", 0, "a zero interval is a busy loop, not a cadence"),
        ("expires_in", 10**9, "a TTL no attended flow could ever honour"),
        ("device_code", "", "an empty poll credential is not a credential"),
    ],
)
def test_device_start_refuses_a_mint_it_cannot_safely_render(
    tmp_path: Path, field: str, value: Any, why: str
) -> None:
    """``--api-url`` is operator-supplied and therefore untrusted.

    Everything a start returns is either printed in a terminal or clicked by a
    human, so a nonconforming endpoint must not be able to choose those strings
    freely. A refusal here costs one un-minted code; the alternative is a
    terminal wearing the console's copy around somebody else's address.
    """
    payload = {
        "device_code": DEVICE_CODE,
        "user_code": USER_CODE,
        "verification_uri": f"{API_URL}/link",
        "verification_uri_complete": f"{API_URL}/link#{USER_CODE}",
        "credential_fingerprint": None,
        "expires_in": 900,
        "interval": 5,
    }
    payload[field] = value

    class _Cloud(FakeCloud):
        def _mint_body(self, submitted: dict[str, Any]) -> dict[str, Any]:
            out = dict(payload)
            if out["credential_fingerprint"] is None:
                out["credential_fingerprint"] = credential_fingerprint(
                    submitted["credential_reference"]
                )
            return out

    h = _build(tmp_path, cloud=_Cloud())

    response = h.start()

    assert response.status_code == 502, why
    assert response.json()["code"] == "link.cloud_protocol"
    assert h.app.state.link_device_pending is None


def test_device_start_maps_a_cloud_refusal_without_echoing_the_body(tmp_path: Path) -> None:
    """The mint sends no secret, so the cloud's machine token travels as-is.

    Its BODY still never does (D11).
    """
    h = _build(tmp_path, cloud=FakeCloud(mint_status=429, error_code="rate_limited"))

    response = h.start()

    assert response.status_code == 429, response.text
    payload = response.json()
    assert payload["code"] == "link.claim_rate_limited"
    assert payload["cloud_code"] == "rate_limited"
    assert CLOUD_NOISE["detail"] not in response.text
    assert h.app.state.link_device_pending is None


# ---------------------------------------------------------------------------
# device poll — the four-step pre-flight
# ---------------------------------------------------------------------------


def test_device_poll_after_a_daemon_restart_answers_device_not_started(tmp_path: Path) -> None:
    """The slot is process memory, and the answer says so rather than pretending.

    A restarted daemon has no slot and no ``device_code``; the cloud row it
    abandoned dies at its own TTL. Step (3) of the pre-flight, and it must never
    reach the cloud — there is nothing to poll WITH.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.poll("a-session-from-a-previous-life")

    assert response.status_code == 409, response.text
    payload = response.json()
    assert payload["code"] == "link.device_not_started"
    assert "nerdit link --device" in payload["hint"]
    assert not h.cloud.polled


def test_device_poll_from_a_superseded_start_is_refused_not_crosstalked(tmp_path: Path) -> None:
    """Two terminals, one slot — and the older one is told, never silently re-pointed.

    Without the session binding the second start's code would be polled by the
    FIRST terminal, which is displaying a different code. Approving what is on
    screen would then link nothing while still enrolling an orphan cloud node
    (device rows enroll at approval), and approving the other would make both
    terminals render "Linked" and both fire a service restart.

    The refusal leaves the slot alone: the newer terminal still owns the flow.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    first = h.start().json()["session"]
    second = h.start().json()["session"]
    assert first != second

    response = h.poll(first)

    assert response.status_code == 409, response.text
    payload = response.json()
    assert payload["code"] == "link.device_superseded"
    assert "newer terminal" in payload["hint"]
    # The whole point: flow #2's device code was NOT forwarded on #1's behalf.
    assert not h.cloud.polled
    # And #2 still owns the slot, so its own poll works.
    assert h.app.state.link_device_pending.session == second
    assert h.poll(second, nonce="poll-2").status_code == 200


def test_device_poll_refuses_a_locally_expired_request_without_a_cloud_hop(
    tmp_path: Path,
) -> None:
    """The daemon already knows the request is dead; spending a poll to prove it is waste."""
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]
    # Monotonic, not a wall clock: a clock step during an attended approval must
    # not expire a live request or resurrect a dead one.
    h.app.state.link_device_pending.expires_at = 0.0

    response = h.poll(session)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.device_expired"
    assert not h.cloud.polled
    # Terminal: the slot is gone, so the next poll says "not started".
    assert h.app.state.link_device_pending is None


def test_device_poll_after_a_successful_link_answers_already_linked(tmp_path: Path) -> None:
    """Step (1), which is also the lost-response recovery path (D-P34-3).

    The commit clears the slot, so a poll retried after a link that actually
    succeeded would otherwise fall through to ``device_not_started`` — a
    failure-shaped answer for a flow that worked. Answering the D8 ``409`` with
    the stored slug instead is what lets the CLI print "Already linked as
    <slug>." and exit 0, and it stays honest: a node linked concurrently by a
    *different* flow gets the same truthful answer rather than being mislabelled
    as this flow's success.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]
    assert h.poll(session).status_code == 200
    assert h.app.state.link_device_pending is None

    replay = h.poll(session, nonce="poll-retry")

    assert replay.status_code == 409, replay.text
    payload = replay.json()
    assert payload["code"] == "link.already_linked"
    assert NODE_ID in payload["message"]
    assert SLUG in payload["message"]


def test_device_poll_requires_admin_and_an_idempotency_key(tmp_path: Path) -> None:
    """Same custody gate as the start and the claim (D2/D3)."""
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]

    denied = h.client.post("/api/link/device/poll", json={"session": session}, headers=_SUBMITTER)
    assert denied.status_code == 403

    keyless = h.client.post(
        "/api/link/device/poll",
        json={"session": session},
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
    )
    assert keyless.status_code == 400
    assert keyless.json()["code"] == "idempotency_key_required"
    assert not h.cloud.polled


# ---------------------------------------------------------------------------
# device poll — the cloud hop
# ---------------------------------------------------------------------------


def test_device_poll_pending_and_slow_down_pass_the_interval_through_and_touch_no_config(
    tmp_path: Path,
) -> None:
    """Cadence belongs to the cloud, and a waiting poll is not a write.

    ``slow_down`` returns the LONGER interval the daemon passes straight
    through (RFC 8628 semantics): the shipped cadence gate fires before any
    database read and is oracle-free, so racing it is both rude and useless.
    Neither answer may write a byte of config or clear the slot.
    """
    h = _build(tmp_path, cloud=FakeCloud(polls=[pending(5), slow_down(10)]))
    session = h.start().json()["session"]
    before = h.config_path.read_bytes()

    first = h.poll(session, nonce="poll-1")
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "pending"
    assert first.json()["interval"] == 5
    assert first.json()["node_id"] is None
    assert first.json()["requires_restart"] is False

    second = h.poll(session, nonce="poll-2")
    assert second.status_code == 200, second.text
    assert second.json()["status"] == "slow_down"
    assert second.json()["interval"] == 10

    assert h.config_path.read_bytes() == before
    assert h.app.state.link_device_pending is not None
    # What the daemon actually put on the wire, both times.
    assert h.cloud.poll_bodies[0] == {
        "device_code": DEVICE_CODE,
        "credential_reference": h.app.state.link_device_pending.verifier,
    }
    assert len(h.cloud.poll_bodies) == 2

    # A pending poll audits as itself, never as a link creation.
    actions = [row["action"] for row in h.audit_rows()]
    assert actions[-1] == "link.device_poll"
    assert "link.created" not in actions


def test_device_poll_approved_commits_the_same_staging_tail_and_mcp_flip_as_claim(
    tmp_path: Path,
) -> None:
    """The approved branch runs the CLAIM's tail, verbatim — it calls the same function.

    Same ``[link]`` keys, the same 2026-08-23 ``[mcp].http_enabled`` flip, the
    same ``link_enrolled_key_file`` stamp, the same ``link.created`` audit row
    with the same param set. If somebody ever re-implements the tail for the
    device path, this is what notices.
    """
    h = _build(tmp_path, cloud=FakeCloud(), config={"daemon": {"auth_token": "t" * 32}})
    session = h.start().json()["session"]

    response = h.poll(session)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "linked"
    assert body["node_id"] == NODE_ID
    assert body["slug"] == SLUG
    assert body["relay_url"] == RELAY
    assert body["enabled"] is True
    assert body["nodes_base_domain"] == DOMAIN
    assert body["requires_restart"] is True
    assert body["mcp_http_enabled"] is True
    assert body["mcp_skipped_reason"] is None
    assert {"link.node_id", "link.slug", "mcp.http_enabled"} <= set(body["restart_keys"])
    assert len(body["verifier_fingerprint"]) == 64

    section = h.link_section()
    assert section["node_id"] == NODE_ID
    assert section["slug"] == SLUG
    assert section["relay_url"] == RELAY
    assert section["enabled"] is True
    assert section["nodes_base_domain"] == DOMAIN
    # (D5) ``api_url`` is per-invocation and NEVER persisted — anywhere.
    assert "api_url" not in section
    assert "app.example.test" not in h.config_path.read_text()
    assert LinkSettings(**h.store.effective_section("link")).node_id == NODE_ID
    assert h.store.effective_section("mcp")["http_enabled"] is True
    assert h.app.state.link_enrolled_key_file == str(h.key_file)
    # The flow is over: the slot goes, so a stray later poll cannot re-commit.
    assert h.app.state.link_device_pending is None

    # A device link IS a link creation, so it audits as one — same action, same
    # target, the claim's exact param set (D-P34-1).
    row = h.audit_rows()[-1]
    assert row["action"] == "link.created"
    assert row["target_type"] == "link"
    assert row["target_id"] == NODE_ID
    params = json.loads(row["params_redacted"])
    assert params["node_id"] == NODE_ID
    assert params["slug"] == SLUG
    assert params["relay_host"] == "relay.example.test"
    assert params["enabled"] is True
    assert params["nodes_base_domain"] == DOMAIN
    assert params["mcp_http_enabled"] is True
    assert params["verifier_fingerprint"] == body["verifier_fingerprint"]
    # Three ways in now, so the durable row says which one this was.
    assert params["grant"] == "device"


def test_device_poll_no_enable_persists_the_identity_without_arming_the_tunnel(
    tmp_path: Path,
) -> None:
    """``enable`` is captured at the START and honoured at the commit (D6)."""
    h = _build(tmp_path, cloud=FakeCloud(), config={"daemon": {"auth_token": "t" * 32}})
    session = h.start(enable=False).json()["session"]

    body = h.poll(session).json()

    assert body["enabled"] is False
    assert body["mcp_http_enabled"] is False
    section = h.link_section()
    assert section["node_id"] == NODE_ID
    assert section.get("enabled", False) is False
    assert h.store.effective_section("mcp").get("http_enabled", False) is False


def test_device_poll_approved_recheck_refuses_when_a_concurrent_claim_won_the_lock(
    tmp_path: Path,
) -> None:
    """The approval is not lost — but this daemon already has an identity.

    A claim or a pre-auth enroll can commit while this poll is parked on its
    cloud hop, so the commit branch re-reads ``[link]`` under the lock. Writing
    over a live identity would be a silent re-link; refusing keeps the cloud's
    forever-replaying 200 available to whoever actually needs it, and clears the
    slot because this flow is over either way.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]
    # The concurrent writer, landing WHILE the poll is in flight — which is the
    # only window the commit-time re-check exists for.
    h.cloud.on_poll = lambda: h.link_config(node_id="another-node", slug="elsewhere")

    response = h.poll(session)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.already_linked"
    assert "another-node" in response.json()["message"]
    # Not overwritten by the answer this poll was holding.
    assert h.link_section()["node_id"] == "another-node"
    assert h.app.state.link_device_pending is None


def test_device_poll_recheck_refuses_when_the_relay_endpoint_vanished(tmp_path: Path) -> None:
    """D5 re-checked at commit, as the start promised.

    The window the start's fast-fail leaves open is a human — up to fifteen
    minutes — and ``[link].relay_url`` can be re-staged through the config API
    inside it. Committing anyway would produce exactly what D5 exists to
    prevent: a linked node that cannot start.
    """
    h = _build(tmp_path, link={"relay_url": RELAY}, cloud=FakeCloud())
    body = _claim_body()
    del body["relay_url"]
    session = h.client.post("/api/link/device", json=body, headers=_ADMIN).json()["session"]
    h.cloud.on_poll = lambda: h.link_config(relay_url=None)

    response = h.poll(session)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "link.relay_url_required"
    assert h.link_section().get("node_id") is None


@pytest.mark.parametrize("enable", [True, False])
def test_device_poll_refuses_when_the_relay_vanishes_inside_the_credential_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable: bool
) -> None:
    """Re-resolve the stored relay after the commit tail's final credential-check await.

    Clearing the relay during that await must refuse the commit for both enable
    values. Previously, enable=True returned a generic config error after approval;
    enable=False persisted node_id/slug without a relay and reported a stale URL.
    Resolve beside the write so both refusals identify the missing relay.
    """
    h = _build(tmp_path, link={"relay_url": RELAY}, cloud=FakeCloud())
    # No ``relay_url`` in the body: this bug needs the STORED-relay shape, since
    # an explicitly supplied one is carried in the slot and persisted regardless.
    body = _claim_body(enable=enable)
    del body["relay_url"]
    session = h.client.post("/api/link/device", json=body, headers=_ADMIN).json()["session"]
    assert h.app.state.link_device_pending.relay_url is None

    # The concurrent writer, fired from inside the identity load — the only
    # await between the poll's config read and the commit. Delegating to the
    # real loader leaves the credential itself untouched, so the refusal under
    # test can only be the relay one.
    real_loader = load_or_create_identity

    def _clear_the_relay_then_load(path: Path) -> Any:
        h.link_config(relay_url=None)
        return real_loader(path)

    monkeypatch.setattr(
        "nerdit.daemon.routes.link.load_or_create_identity", _clear_the_relay_then_load
    )

    response = h.poll(session)

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "link.relay_url_required"
    # Refusing WITHOUT committing is what keeps recovery real: restore the relay
    # and poll again, and the cloud's forever-replaying approval still lands.
    # ``node_id`` is the one that mattered on the ``--no-enable`` arm, where the
    # old code persisted an identity with no endpoint behind it.
    section = h.link_section()
    assert section.get("node_id") is None
    assert section.get("slug") is None
    assert "relay_url" not in section


@pytest.mark.parametrize(
    ("status", "cloud_code", "expected", "clears"),
    [
        (403, "link_code_denied", "link.device_denied", True),
        (410, "link_code_expired", "link.device_expired", True),
        (404, "link_code_invalid", "link.device_invalid", True),
        (409, "node_authentication_failed", "link.device_credential_mismatch", True),
        # The 400 arm exists in the SHIPPED cloud though the cloud plan's
        # response table omits it — this daemon follows the code.
        (400, "node_authentication_failed", "link.device_credential_mismatch", True),
        (429, "rate_limited", "link.claim_rate_limited", False),
        (503, "service_unavailable", "link.cloud_error", False),
    ],
)
def test_device_poll_maps_denied_expired_invalid_and_mismatch_without_echoing_the_cloud_body(
    tmp_path: Path, status: int, cloud_code: str, expected: str, clears: bool
) -> None:
    """One table, and the slot outcome is half of it.

    A terminal answer means this request is dead, so the slot goes and the next
    poll honestly says "not started". A rate limit or a cloud outage says
    nothing about the request's fate, so the slot stays and the CLI keeps
    polling. The cloud's machine token rides as ``cloud_code`` so an agent can
    act on the real reason; the cloud's BODY never does (D11).
    """
    h = _build(tmp_path, cloud=FakeCloud(polls=[refusal(status, cloud_code)]))
    session = h.start().json()["session"]
    before = h.config_path.read_bytes()

    response = h.poll(session)

    payload = response.json()
    assert payload["code"] == expected, response.text
    assert payload["cloud_code"] == cloud_code
    assert payload["cloud_status"] == status
    assert CLOUD_NOISE["detail"] not in response.text
    assert DEVICE_CODE not in response.text
    # A refusal never burns state.
    assert h.config_path.read_bytes() == before
    assert (h.app.state.link_device_pending is None) is clears


@pytest.mark.parametrize(
    ("sent", "expected"),
    [
        ("17", "17"),
        (" 17 ", "17"),
        ("0", None),
        ("-5", None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        ("soon", None),
        (None, None),
    ],
)
def test_device_poll_forwards_the_clouds_retry_after_header(
    tmp_path: Path, sent: str | None, expected: str | None
) -> None:
    """(D-X16-O26) The cloud's backoff has to reach the client to be a knob.

    The cloud computes ``Retry-After`` from its live rate-limit window so the
    cadence can be tuned without shipping a daemon; dropped here, the CLI falls
    back to doubling its own interval and the server-side number means nothing.
    Asserted on the HTTP header, not on the envelope, because the header is
    what the CLI reads: a body field would look like a fix and change nothing.

    The value is re-derived rather than echoed, so the negative rows matter as
    much as the first: a nonconforming endpoint must not be able to put an
    unreadable, zero or negative ``Retry-After`` on this daemon's response,
    where it would displace the fallback the CLI would otherwise use.
    """
    headers = {} if sent is None else {"Retry-After": sent}
    poll = httpx.Response(
        429, json={"error": {"code": "rate_limited"}, **CLOUD_NOISE}, headers=headers
    )
    h = _build(tmp_path, cloud=FakeCloud(polls=[poll]))
    session = h.start().json()["session"]

    response = h.poll(session)

    assert response.status_code == 429, response.text
    assert response.json()["code"] == "link.claim_rate_limited"
    assert response.headers.get("retry-after") == expected
    # The header is the whole forwarding surface: nothing leaked into the body,
    # and no other header of the cloud's rode along.
    assert "headers" not in response.json()
    assert CLOUD_NOISE["detail"] not in response.text
    # The slot survives a rate limit: the request's fate is still undecided.
    assert h.app.state.link_device_pending is not None


def test_device_poll_survives_a_transport_failure_and_keeps_the_slot(tmp_path: Path) -> None:
    """An unreachable cloud says nothing about the request's fate (D-ENT-2's posture)."""
    h = _build(
        tmp_path,
        cloud=FakeCloud(
            polls=[httpx.ConnectError("nothing listening", request=httpx.Request("POST", API_URL))]
        ),
    )
    session = h.start().json()["session"]

    response = h.poll(session)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "link.cloud_unreachable"
    # The netloc the operator supplied, never the full URL, never the code.
    assert "app.example.test" in response.json()["message"]
    assert DEVICE_CODE not in response.text
    assert h.app.state.link_device_pending is not None
    # ...so the next poll simply carries on, and links.
    assert h.poll(session, nonce="poll-2").json()["status"] == "linked"


def test_device_poll_terminal_refusal_clears_only_its_own_flow_never_a_replacement(
    tmp_path: Path,
) -> None:
    """The slot-clear is identity-guarded against the lock-free window (D-P34-1).

    The poll's cloud hop runs without the mutation lock for up to 30 s, and a
    concurrent ``nerdit link --device`` can replace the slot inside that
    window — the session binding already refused the OLD flow's *next* poll,
    but a poll that passed pre-flight *before* the replacement still settles
    afterwards. Its terminal refusal must settle only its own flow: clearing
    whatever slot is current would kill the newer terminal's live code, answer
    its next poll ``device_not_started``, and leave its approval to enroll an
    orphan cloud node — the exact cross-talk the binding exists to prevent.
    """
    replacement = object()
    cloud = FakeCloud(polls=[refusal(410, "link_code_expired")])
    h = _build(tmp_path, cloud=cloud)
    # The replacement lands while the first flow's poll is parked on the cloud:
    # the transport handler runs inside the daemon's own hop, which is exactly
    # the lock-free window a second start would exploit.
    cloud.on_poll = lambda: setattr(h.app.state, "link_device_pending", replacement)
    session = h.start().json()["session"]

    response = h.poll(session)

    assert response.json()["code"] == "link.device_expired", response.text
    # The refused flow is settled; the replacement's slot survives untouched.
    assert h.app.state.link_device_pending is replacement


def test_device_poll_refuses_a_reflected_device_code_in_the_success_shape(
    tmp_path: Path,
) -> None:
    """The containment guard is not depth on this path — it is the guard.

    A link code is uppercase and every reflectable field is lowercase-only, so
    the two spaces are disjoint. A ``device_code`` is 32 Crockford symbols,
    whose lowercase form is a perfectly legal DNS label — so a nonconforming
    endpoint answering 200 with the submitted code as the ``node_slug`` would
    otherwise see that secret persisted into config, echoed, audited and logged.
    """
    h = _build(
        tmp_path,
        cloud=FakeCloud(
            polls=[httpx.Response(200, json={"node_id": NODE_ID, "node_slug": DEVICE_CODE.lower()})]
        ),
    )
    session = h.start().json()["session"]

    response = h.poll(session)

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "link.cloud_protocol"
    assert DEVICE_CODE.lower() not in response.text
    assert h.link_section() == {}


# ---------------------------------------------------------------------------
# the pre-auth grant on the claim route (D-P34-6)
# ---------------------------------------------------------------------------


def test_claim_dispatches_an_nk_prefixed_secret_to_the_preauth_endpoint_with_the_declared_field_set(
    tmp_path: Path,
) -> None:
    """One route, two grants: only the URL and the payload branch.

    The cloud declares all three link request models ``extra="forbid"``, so the
    daemon sends exactly the declared fields and no more. Everything after the
    hop — the staging tail, the ``[mcp]`` flip, the ``link.created`` row — is the
    claim's, unchanged, which is the whole argument for not minting a third
    route.
    """
    h = _build(tmp_path, cloud=FakeCloud(), config={"daemon": {"auth_token": "t" * 32}})

    response = h.client.post("/api/link/claim", json=_claim_body(key=PREAUTH_KEY), headers=_ADMIN)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["node_id"] == NODE_ID
    assert body["slug"] == SLUG
    assert body["mcp_http_enabled"] is True

    # The pre-auth endpoint, not the claim endpoint.
    assert h.cloud.urls == [f"{API_URL}/api/link/preauth"]
    assert not h.cloud.claims
    sent = h.cloud.preauths[0]
    assert set(sent) == {"key", "credential_reference", "hostname_hint"}
    assert sent["key"] == PREAUTH_KEY
    assert sent["credential_reference"].startswith("ed25519:")

    # The identical staging tail: the config a code claim would have written.
    section = h.link_section()
    assert section["node_id"] == NODE_ID
    assert section["enabled"] is True
    assert h.store.effective_section("mcp")["http_enabled"] is True

    row = h.audit_rows()[-1]
    assert row["action"] == "link.created"
    params = json.loads(row["params_redacted"])
    assert params["grant"] == "key"
    assert PREAUTH_KEY not in row["params_redacted"]


def test_a_claim_naming_both_grants_or_neither_is_refused_before_the_cloud(
    tmp_path: Path,
) -> None:
    """Exactly-one-of, in the route, before anything reaches the config store.

    In-route rather than a model validator (the ``/events``
    ``cursor``/``since_id`` precedent) so the answer is this daemon's structured
    envelope: a pydantic ``value_error`` would echo the model's own field values
    back at the caller, and one of those fields is a secret.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    both = h.client.post(
        "/api/link/claim",
        json=_claim_body(code="NL-PASTEME1234", key=PREAUTH_KEY),
        headers=_ADMIN,
    )
    assert both.status_code == 422, both.text
    assert both.json()["code"] == "validation_error"
    assert PREAUTH_KEY not in both.text

    neither = h.client.post("/api/link/claim", json=_claim_body(), headers=_ADMIN)
    assert neither.status_code == 422, neither.text
    assert neither.json()["code"] == "validation_error"

    assert not h.cloud.urls
    assert h.link_section() == {}


def test_preauth_refusals_reuse_the_refusal_mapping_and_one_key_invalid_message(
    tmp_path: Path,
) -> None:
    """The cloud collapses every key death into one code; the daemon adds no oracle.

    Unknown, malformed, revoked, expired, exhausted and race-lost all arrive as
    ``404 link_key_invalid`` (D-X16-O9), and the daemon offers one message for
    all of them. Distinguishing deaths the cloud deliberately merged would
    reopen the oracle on this machine, for whoever holds a former key.
    """
    h = _build(tmp_path, cloud=FakeCloud(claim_status=404, error_code="link_key_invalid"))

    response = h.client.post("/api/link/claim", json=_claim_body(key=PREAUTH_KEY), headers=_ADMIN)

    assert response.status_code == 409, response.text
    payload = response.json()
    assert payload["code"] == "link.claim_refused"
    assert payload["cloud_code"] == "link_key_invalid"
    assert "revoked, expired, or exhausted" in payload["hint"]
    assert CLOUD_NOISE["detail"] not in response.text
    assert PREAUTH_KEY not in response.text
    assert h.link_section() == {}


@pytest.mark.parametrize(
    ("cloud_code", "fragment"),
    [
        ("node_already_linked", "another account"),
        ("plan_limit_exceeded", "node limit"),
        ("entitlement_required", "not active"),
    ],
)
def test_preauth_hints_cover_the_rest_of_the_shipped_key_vocabulary(
    tmp_path: Path, cloud_code: str, fragment: str
) -> None:
    """Each cloud refusal an operator can act on gets its own line — and no more.

    ``entitlement_required`` in particular says nothing about THIS daemon's
    runtime: no route, deploy, proxy, model or database is gated on it, in this
    phase or in this programme (D-X16-O15 / D-X16-57 / D-ENT-2).

    Its fragment is ``not active`` and not ``not entitled`` on purpose (P34, X16
    reconciliation §3.3): post-collapse the cloud raises this code from one
    predicate only, account standing (D-X16-O25), so the hint names standing.
    The node-count refusal is the ``plan_limit_exceeded`` row above, which is
    where a plan legitimately belongs.
    """
    h = _build(tmp_path, cloud=FakeCloud(claim_status=409, error_code=cloud_code))

    response = h.client.post("/api/link/claim", json=_claim_body(key=PREAUTH_KEY), headers=_ADMIN)

    assert response.status_code == 409, response.text
    assert response.json()["cloud_code"] == cloud_code
    assert fragment in response.json()["hint"]


@pytest.mark.parametrize(
    "bad_key",
    [
        "nk_short",
        "NK_0123456789ABCDEFGHJKMNPQRSTVWXYZ",
        "nk_0123456789abcdefghjkmnpqrstvwxyz",
        "nk_0123456789ABCDEFGHJKMNPQRSTVWXYZI",
        "0123456789ABCDEFGHJKMNPQRSTVWXYZ",
    ],
)
def test_a_malformed_key_is_refused_by_the_schema_before_the_cloud(
    tmp_path: Path, bad_key: str
) -> None:
    """The shipped ``valid_preauth_key`` shape, mirrored so no round trip is spent.

    Exact prefix, exact length, uppercase Crockford body (no ``I``, ``L``, ``O``
    or ``U``). Strict on purpose: this artifact is never retyped by a human, so
    every spelling that is not the issued one is a mistake worth failing on
    locally rather than a rate-limit budget spent proving it remotely.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    response = h.client.post("/api/link/claim", json=_claim_body(key=bad_key), headers=_ADMIN)

    assert response.status_code == 422
    assert not h.cloud.urls


# ---------------------------------------------------------------------------
# the never-leak sweep, and the surface pins
# ---------------------------------------------------------------------------


def test_device_flow_audit_rows_carry_hosts_and_fingerprint_never_a_code(
    tmp_path: Path,
) -> None:
    """What the durable trail records, positively — and what it must not.

    Hosts (netloc only: operators park bearer tokens in URL *paths*) and the
    verifier fingerprint, which is the documented non-secret audit form of the
    node key. Never the ``user_code``, never the ``device_code``, never the URL.
    """
    h = _build(tmp_path, cloud=FakeCloud(polls=[pending(), refusal(403, "link_code_denied")]))

    h.start()
    start_row = h.audit_rows()[-1]
    assert start_row["action"] == "link.device_started"
    assert start_row["target_type"] == "link"
    start_params = json.loads(start_row["params_redacted"])
    assert start_params["api_host"] == "app.example.test"
    assert len(start_params["verifier_fingerprint"]) == 64
    assert start_params["expires_in"] == 900
    assert set(start_params) == {"api_host", "verifier_fingerprint", "enable", "expires_in"}

    session = h.app.state.link_device_pending.session
    h.poll(session, nonce="poll-1")
    poll_row = h.audit_rows()[-1]
    assert poll_row["action"] == "link.device_poll"
    poll_params = json.loads(poll_row["params_redacted"])
    assert poll_params == {
        "api_host": "app.example.test",
        "verifier_fingerprint": start_params["verifier_fingerprint"],
        "status": "pending",
    }

    h.poll(session, nonce="poll-2")
    denied_row = h.audit_rows()[-1]
    assert denied_row["action"] == "link.device_poll"
    # A refusal still records what it was asked to do; a refusal never records
    # a status it did not reach.
    assert "status" not in json.loads(denied_row["params_redacted"])


def test_no_secret_of_these_routes_reaches_an_audit_row_a_log_record_or_a_response(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The sweep: three secrets, every shape they could escape through.

    The ``device_code`` is the daemon's poll credential, the ``user_code`` is
    what a human carries, and the pre-auth ``key`` is the unattended grant — and
    none of the three may appear in a response body, an audit row, a log record
    or the config file, on ANY path through these routes (D-X16-O11 and the
    standing convention).

    Run against the happy path and every refusal shape, because the refusals are
    where reflection actually happens: a nonconforming ``--api-url`` endpoint
    controls the error body, and an unguarded mapping would hand it straight
    back.
    """
    caplog.set_level(logging.DEBUG)
    secrets = (DEVICE_CODE, DEVICE_CODE.lower(), USER_CODE, PREAUTH_KEY)

    def _sweep(h: Harness, *responses: httpx.Response) -> None:
        haystacks = [r.text for r in responses]
        haystacks.append(h.audit_text())
        haystacks.append(h.config_path.read_text())
        for secret in secrets:
            for haystack in haystacks:
                assert secret not in haystack

    happy = _build(tmp_path / "happy", cloud=FakeCloud())
    started = happy.start()
    session = started.json()["session"]
    # The user code IS in the start response — that is its job — so the sweep
    # runs against everything else the flow produced.
    _sweep(happy, happy.poll(session))

    for index, scripted in enumerate(
        [
            refusal(403, "link_code_denied"),
            refusal(410, "link_code_expired"),
            refusal(404, "link_code_invalid"),
            refusal(409, "node_authentication_failed"),
            refusal(429, "rate_limited"),
            refusal(500, "internal_error"),
            # The reflection arm: the endpoint echoes the submitted device code
            # as its own machine token.
            httpx.Response(400, json={"error": {"code": DEVICE_CODE.lower()}}),
        ]
    ):
        h = _build(tmp_path / f"refusal-{index}", cloud=FakeCloud(polls=[scripted]))
        session = h.start().json()["session"]
        _sweep(h, h.poll(session))

    for index, (status, code) in enumerate(
        [(404, "link_key_invalid"), (409, "node_already_linked"), (500, "internal_error")]
    ):
        h = _build(
            tmp_path / f"preauth-{index}", cloud=FakeCloud(claim_status=status, error_code=code)
        )
        _sweep(
            h,
            h.client.post("/api/link/claim", json=_claim_body(key=PREAUTH_KEY), headers=_ADMIN),
        )

    for secret in secrets:
        assert secret not in caplog.text


def test_the_two_device_actions_are_mapped_and_deliberately_not_body_hash_exempt() -> None:
    """The route-table registration and the ``NO_BODY_HASH_ACTIONS`` decision.

    Named actions rather than the derived ``POST /link/device`` fallback, so an
    operator filtering the trail for ``link.*`` sees the whole of how a node
    arrived. And neither joins ``NO_BODY_HASH_ACTIONS``, deliberately: the start
    body is three non-secret config fields and the poll body is one grantless
    opaque ``session``, so the soft body-hash is harmless on both — while
    ``link.created`` stays a member, which is what keeps the pre-auth key's
    digest off disk.
    """
    assert derive_action("POST", "/api/link/device") == ("link.device_started", "link", None)
    assert derive_action("POST", "/api/link/device/poll") == ("link.device_poll", "link", None)
    # The literal /device/poll rule must not be shadowed by /device.
    assert derive_action("POST", "/link/device/poll")[0] == "link.device_poll"

    assert "link.device_started" not in NO_BODY_HASH_ACTIONS
    assert "link.device_poll" not in NO_BODY_HASH_ACTIONS
    assert "link.created" in NO_BODY_HASH_ACTIONS


def test_the_device_start_response_is_never_cached_for_replay() -> None:
    """``link.device_started`` is in ``NO_BODY_CACHE_ACTIONS`` — the other set.

    The request-body-HASH question above was decided one way (three non-secret
    config fields, harmless digest); the response-body-CACHE question is the
    opposite one and decided the opposite way: the start response carries the
    ``user_code`` twice (bare, and in ``verification_uri_complete``'s
    fragment), and that code is a live grant for its 900 s TTL — a DB reader
    or a fresh backup tar could approve the victim's pending flow onto their
    own account. The route's contract says the user_code is displayed by the
    CLI and recorded nowhere; the durable idempotency row must not become the
    place it is recorded. The poll stays cacheable: its pending answer is
    grantless, and its linked answer is the claim view — the same non-secret
    body ``link.created`` has always cached.
    """
    assert "link.device_started" in NO_BODY_CACHE_ACTIONS
    assert "link.device_poll" not in NO_BODY_CACHE_ACTIONS


def test_a_validation_error_never_echoes_the_pre_auth_key(tmp_path: Path) -> None:
    """A 422 on a malformed body must not hand the pre-auth key back.

    ``repr=False`` on the field stops neither live shape, because both raise
    BEFORE the model is constructed and the echoed ``input`` is FastAPI's raw
    value, not the model repr — the lesson ``code``/``blob``/``relay_url``
    already taught in ``errors._SECRET_INPUT_FIELDS``:

    * ``value_error`` from the shape validator, whose ``input`` **is** the key —
      and the keys most likely to be malformed are the ones pasted wrong, i.e.
      real keys;
    * ``string_too_long``, same shape.

    The key is a live grant that enrolls a machine, and the whole D-X16-O11
    custody chain (stdin → CLI memory → JSON body → JSON body) exists so it
    never lands in a transcript. A 422 that echoes it undoes all of it in one
    line, so the guarantee lives in that frozenset's ``"key"`` entry — this
    test is the pin that keeps the one word from being tidied away.
    """
    h = _build(tmp_path, cloud=FakeCloud())

    malformed = h.client.post(
        "/api/link/claim",
        json=_claim_body(key=PREAUTH_KEY.lower()),
        headers=_ADMIN,
    )
    assert malformed.status_code == 422
    assert PREAUTH_KEY.lower() not in malformed.text

    too_long = h.client.post("/api/link/claim", json=_claim_body(key="K" * 200), headers=_ADMIN)
    assert too_long.status_code == 422
    assert "K" * 200 not in too_long.text


# ---------------------------------------------------------------------------
# the credential the cloud enrolled must still be the credential on disk
# ---------------------------------------------------------------------------


def test_device_poll_refuses_to_commit_when_the_node_key_vanished_mid_flow(
    tmp_path: Path,
) -> None:
    """The unlink-mid-flow hole, closed at the commit rather than only upstream.

    ``nerdit link --device`` → ``nerdit unlink`` → approve in the browser used to
    walk every pre-flight: ``node_id`` had just been cleared, the slot and its
    session were untouched, and the cloud confirmed the approval for the verifier
    minted from the DELETED key. The commit then wrote
    ``node_id``/``slug``/``enabled = true`` naming a credential that no longer
    existed; the next boot minted a fresh keypair and the tunnel could never
    authenticate as the enrolled node.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]
    assert h.key_file.exists()
    h.key_file.unlink()

    response = h.poll(session)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.node_credential_changed"
    # Nothing enrolled, and the dead flow does not linger.
    assert h.store.effective_section("link").get("node_id") is None
    assert h.app.state.link_device_pending is None


def test_device_poll_refuses_to_commit_when_the_node_key_was_replaced_mid_flow(
    tmp_path: Path,
) -> None:
    """A DIFFERENT key in the same place is refused too, not just a missing one.

    A restored backup or an operator swapping identity files leaves a perfectly
    readable key that the cloud never enrolled. Committing it would look like a
    success and produce the same unauthenticatable node, so the fingerprint —
    not merely the file's existence — is what the commit compares.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    session = h.start().json()["session"]
    minted = h.key_file.read_bytes()
    h.key_file.unlink()
    replacement = load_or_create_identity(h.key_file)
    assert h.key_file.read_bytes() != minted

    response = h.poll(session)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.node_credential_changed"
    assert h.store.effective_section("link").get("node_id") is None
    # The replacement is left exactly as found — refusing is not repairing.
    assert load_or_create_identity(h.key_file).fingerprint == replacement.fingerprint


def test_unlink_disarms_an_in_flight_device_request(tmp_path: Path) -> None:
    """Unlink tears down the whole enrolment, so every slot goes with it.

    Not ``_clear_pending``: that helper is identity-guarded so one flow's
    terminal answer cannot settle another's. Unlink is not one flow ending.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    h.start()
    assert h.app.state.link_device_pending is not None

    response = h.client.request(
        "DELETE",
        "/api/link",
        headers=_headers("unlink-1"),
    )

    assert response.status_code == 200, response.text
    assert h.app.state.link_device_pending is None


def test_device_poll_refuses_to_commit_for_a_flow_superseded_mid_hop(tmp_path: Path) -> None:
    """The session binding closed the pre-flight window; this is the commit one.

    ``_poll_device_code`` runs its cloud hop lock-free, so a concurrent start
    can replace the slot inside it. A poll returning approved must re-establish
    that it is still the daemon's current flow — checking ``node_id`` alone is
    not that, because ``node_id`` is unset in exactly this race, so the
    superseded flow would take custody of the daemon and leave the newer
    terminal polling a slot whose approval can never land.
    """
    h = _build(tmp_path, cloud=FakeCloud())
    first = h.start().json()["session"]

    # Replace the slot while the first flow's poll is in flight: the cloud
    # handler is the only place that runs mid-hop.
    replaced: list[str] = []

    original = h.cloud.handler

    def _supersede(request: httpx.Request) -> httpx.Response:
        response = original(request)
        if request.url.path.endswith("/device/poll") and not replaced:
            replaced.append(h.start().json()["session"])
        return response

    h.app.state.link_claim_transport = httpx.MockTransport(_supersede)

    response = h.poll(first)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "link.device_superseded"
    # The daemon stayed unlinked, and the REPLACEMENT flow is untouched.
    assert h.store.effective_section("link").get("node_id") is None
    assert h.app.state.link_device_pending is not None
    assert h.app.state.link_device_pending.session == replaced[0]


def test_a_verification_uri_carrying_the_code_outside_its_fragment_is_refused(
    tmp_path: Path,
) -> None:
    """D-X16-O24 is about WHERE in the URL the code sits, so the daemon checks.

    A fragment is never transmitted; a query string reaches access logs, CDN
    logs, ``Referer`` headers and shared-link history, and a live user code is
    adoptable by anyone who reads one for its whole TTL. An endpoint answering
    ``…/link?user_code=…`` would otherwise be accepted and printed for the
    operator to click, undoing the decision on the daemon's side.
    """

    class QueryCloud(FakeCloud):
        def handler(self, request: httpx.Request) -> httpx.Response:
            response = super().handler(request)
            if request.url.path.endswith("/api/link/device"):
                body = response.json()
                body["verification_uri_complete"] = (
                    f"https://app.example.test/link?user_code={body['user_code']}"
                )
                return httpx.Response(response.status_code, json=body)
            return response

    h = _build(tmp_path, cloud=QueryCloud())
    response = h.start()

    assert response.status_code == 502, response.text
    assert response.json()["code"] == "link.cloud_protocol"
    assert h.app.state.link_device_pending is None


@pytest.mark.parametrize(
    ("complete", "leak"),
    [
        (
            f"{API_URL}/link?user_code={USER_CODE.lower()}",
            "the same code, lower-cased, in the query",
        ),
        (
            f"{API_URL}/link/{USER_CODE.replace('-', '')}#{USER_CODE}",
            "the same code, de-hyphenated, in the path",
        ),
        (
            f"{API_URL}/link?user_code={USER_CODE.replace('-', '').lower()}#{USER_CODE}",
            "the same code, de-hyphenated AND lower-cased, in the query",
        ),
    ],
)
def test_a_normalised_copy_of_the_code_outside_the_fragment_is_refused_too(
    tmp_path: Path, complete: str, leak: str
) -> None:
    """The console forgives case and hyphens on entry, so the guard must too.

    A substring test against the canonical spelling alone would pass every URI
    below, and each of them puts a *live, redeemable* credential into access
    logs, CDN logs, ``Referer`` headers and shared-link history — which is the
    entire content of D-X16-O24. Normalising both sides (hyphens stripped,
    case-folded) before the containment test closes that gap without touching
    the fragment requirement, which stays exact: what the terminal prints there
    must be the code it printed, verbatim.
    """

    class NormalisedLeakCloud(FakeCloud):
        def handler(self, request: httpx.Request) -> httpx.Response:
            response = super().handler(request)
            if request.url.path.endswith("/api/link/device"):
                body = response.json()
                body["verification_uri_complete"] = complete
                return httpx.Response(response.status_code, json=body)
            return response

    h = _build(tmp_path, cloud=NormalisedLeakCloud())
    response = h.start()

    assert response.status_code == 502, f"{leak}: {response.text}"
    assert response.json()["code"] == "link.cloud_protocol"
    assert h.app.state.link_device_pending is None


def test_a_clean_fragment_only_verification_uri_still_passes_the_guard() -> None:
    """The other half of the normalisation change: it refuses nothing new.

    A code appears in the canonical URI exactly once, in the fragment, and the
    normalised containment test must not read that fragment as a leak.
    """
    uri = f"{API_URL}/link#{USER_CODE}"

    assert _validated_verification_uri(uri, user_code=USER_CODE) == uri


@pytest.mark.parametrize(
    ("complete", "regression"),
    [
        (f"{API_URL}/link", "the fragment dropped entirely"),
        (f"{API_URL}/link#7Q3M-X2VF-0000", "some other flow's code in the fragment"),
    ],
)
def test_a_one_click_verification_uri_without_this_flows_code_is_refused(
    tmp_path: Path, complete: str, regression: str
) -> None:
    """Rejecting the wrong places never established the right one.

    The test above pins where the code must NOT be. This pins where it must be,
    and the gap between the two was real: both URIs below carry no query, no
    path segment holding a code, https and no userinfo, so both passed every
    rule the daemon had and got printed as the one-click address. Neither is
    one. The operator clicks, lands on an empty or wrong approval form, and the
    flow the terminal just promised was a single click degrades into a retype
    with no diagnostic anywhere — the failure a fragment-dropping cloud
    regression would produce, and the reason this assertion is worth its one
    comparison.

    Refused at the mint, so the slot is never armed: a request whose printable
    material is wrong has nothing to poll for.
    """

    class RegressedCloud(FakeCloud):
        def handler(self, request: httpx.Request) -> httpx.Response:
            response = super().handler(request)
            if request.url.path.endswith("/api/link/device"):
                body = response.json()
                body["verification_uri_complete"] = complete
                return httpx.Response(response.status_code, json=body)
            return response

    h = _build(tmp_path, cloud=RegressedCloud())
    response = h.start()

    assert response.status_code == 502, f"{regression}: {response.text}"
    assert response.json()["code"] == "link.cloud_protocol"
    assert h.app.state.link_device_pending is None
    # The refusal names the endpoint's mistake, never the material it mangled.
    assert USER_CODE not in response.text


def test_the_device_mint_never_renders_either_code_in_its_repr() -> None:
    """The slot redacts its ``__repr__``; the mint holds BOTH codes and must too.

    No call site logs it today — that is a fact about today's call sites, not a
    property of the type.
    """
    mint = _DeviceMint(
        device_code="D" * 32,
        user_code="ABCD-EFGH-JKMN",
        verification_uri="https://app.example.test/link",
        verification_uri_complete="https://app.example.test/link#ABCD-EFGH-JKMN",
        credential_fingerprint="f" * 64,
        expires_in=900,
        interval=5,
    )
    rendered = repr(mint)
    assert "D" * 32 not in rendered
    assert "ABCD-EFGH-JKMN" not in rendered
    assert "expires_in=900" in rendered
