"""Test hosted-share gates with real middleware and a stateful query double.

Public access requires cloud entitlement plus edge_auth intent or explicit consent.
edge_auth does not protect hosted paths that bypass Caddy. Require node ID, slug,
base domain and a live tunnel; enforce the 63-character DNS label limit on writes.

Pin roles, ownership, scope, mandatory in-route idempotency and non-secret audit
fields. Durable events may carry private URLs, never public ones.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import tomli_w
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient

from nerdit.config.settings import LinkSettings
from nerdit.config.store import ConfigStore
from nerdit.core import eventlog
from nerdit.daemon.audit import AuditMiddleware, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.link import router as link_router
from nerdit.daemon.routes.share import router as share_router
from nerdit.daemon.views.hosted import HostedContext, hosted_entry, load_hosted_context
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.rows import ServiceShare

LEGACY = "legacy-global"  # noqa: S105 - a test literal
SUB_RAW = "sub-raw"  # noqa: S105 - a test literal
OTHER_RAW = "other-raw"  # noqa: S105 - a test literal
RO_RAW = "ro-raw"  # noqa: S105 - a test literal
ADMIN_RAW = "admin-raw"  # noqa: S105 - a test literal
SCOPED_RAW = "scoped-raw"  # noqa: S105 - a test literal

NODE_ID = "00000000-0000-4000-8000-00000000000a"
SLUG = "gpu-box"
DOMAIN = "nodes.test"
HOSTED_URL = f"https://demo--{SLUG}.{DOMAIN}/"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(OTHER_RAW): ApiToken(
        id="tok-other", name="o", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SCOPED_RAW): ApiToken(
        id="tok-sub",
        name="sc",
        role=TokenRole.submitter,
        token_hash=hash_token(SCOPED_RAW),
        scope_services=["other-app"],
    ),
}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def _svc(
    *,
    owner: str | None = "tok-sub",
    kind: JobKind = JobKind.service,
    name: str = "demo",
    config: dict[str, Any] | None = None,
    status: JobStatus = JobStatus.running,
) -> Job:
    return Job(
        id="svc-1",
        service_name=name,
        name=name,
        kind=kind,
        gpu_count=0,
        status=status,
        submitted_by_token=owner,
        config=json.dumps({"image": f"nerdit-app/{name}:1", **(config or {})}),
    )


class FakeLinkManager:
    """``app.state.link_manager`` as the share route and the hosted context see it.

    Three things only: the auth middleware's capability probe, the ``state`` the
    hosted projection classifies on, and the one entitlement boolean the cloud's
    P32 push writes (already TTL-resolved — ``LinkStatus`` exposes the effective
    read, so a route never re-derives expiry).
    """

    def __init__(self, *, state: str = "connected", entitled: bool = False) -> None:
        self._status = SimpleNamespace(
            state=state,
            hosted_public_entitled=entitled,
            hosted_public_entitled_at=None,
        )

    def validate_capability(self, token: str, role: str) -> bool:
        return False

    def status(self) -> SimpleNamespace:
        return self._status


def _queries(job: Job | None = None, shares: dict[str, ServiceShare] | None = None) -> AsyncMock:
    """Queries with a REAL little share table behind the four share methods."""
    table: dict[str, ServiceShare] = dict(shares or {})
    q = AsyncMock()
    q.share_table = table
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # ``_resolve_service`` tries the row id first; these tests always address by
    # name, so the id lookup must answer a real ``None`` (an AsyncMock's default
    # return value is a truthy MagicMock, which would resolve every ident).
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(
        side_effect=lambda n: job if job is not None and job.service_name == n else None
    )
    q.list_service_shares = AsyncMock(side_effect=lambda: dict(table))

    async def _set(
        name: str, access: str, *, job_id: str, preserve_existing: bool = False
    ) -> ServiceShare:
        existing = table.get(name)
        row = ServiceShare(
            service_name=name,
            access=existing.access if existing and preserve_existing else access,
            # The upsert preserves ``created_at`` — "shared since" is a fact
            # about the share, not about the last access flip.
            created_at=existing.created_at if existing else datetime.now(UTC),
        )
        table[name] = row
        return row

    async def _delete(name: str) -> bool:
        return table.pop(name, None) is not None

    q.set_service_share = AsyncMock(side_effect=_set)
    q.delete_service_share = AsyncMock(side_effect=_delete)
    return q


def _idem_store(q: AsyncMock) -> dict:
    """A principal-scoped in-memory idempotency store (the workspaces harness)."""
    records: dict[tuple[str, str], SimpleNamespace] = {}

    async def _insert(*, principal_id, idem_key, method, path, expires_at, body_hash=None):  # noqa: ANN001, ANN202
        if (principal_id, idem_key) in records:
            return False
        records[(principal_id, idem_key)] = SimpleNamespace(
            state="in_progress",
            method=method,
            path=path,
            response_status=None,
            response_body=None,
            content_type=None,
            resource_id=None,
            body_hash=body_hash,
        )
        return True

    async def _get(principal_id, idem_key):  # noqa: ANN001, ANN202
        return records.get((principal_id, idem_key))

    async def _complete(  # noqa: ANN202
        *, principal_id, idem_key, response_status, response_body, content_type, resource_id
    ):  # noqa: ANN001
        rec = records[(principal_id, idem_key)]
        rec.state = "completed"
        rec.response_status = response_status
        rec.response_body = response_body
        rec.content_type = content_type
        rec.resource_id = resource_id

    async def _delete(principal_id, idem_key):  # noqa: ANN001, ANN202
        records.pop((principal_id, idem_key), None)

    q.insert_idempotency_inprogress = AsyncMock(side_effect=_insert)
    q.get_idempotency_record = AsyncMock(side_effect=_get)
    q.complete_idempotency_record = AsyncMock(side_effect=_complete)
    q.delete_idempotency_record = AsyncMock(side_effect=_delete)
    return records


def _link(
    *,
    node_id: str | None = NODE_ID,
    slug: str | None = SLUG,
    domain: str | None = DOMAIN,
    enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(node_id=node_id, slug=slug, nodes_base_domain=domain, enabled=enabled)


def _make_app(
    queries: AsyncMock,
    *,
    link: SimpleNamespace | None = None,
    manager: object | None = "default",
    with_audit: bool = False,
    with_idempotency: bool = False,
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    # Mounted under ``/api`` exactly as production does, so the audit matcher
    # sees the same path it will see live.
    api = APIRouter(prefix="/api")
    api.include_router(share_router)
    app.include_router(api)
    app.state.queries = queries
    app.state.settings = SimpleNamespace(link=link if link is not None else _link())
    app.state.link_manager = FakeLinkManager() if manager == "default" else manager
    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, **kw) -> TestClient:  # noqa: ANN003
    return TestClient(_make_app(queries, **kw), raise_server_exceptions=False)


async def _hosted_context_of(app: FastAPI) -> HostedContext:
    """The ONE seam every list/detail surface projects through (D-P26-14)."""
    return await load_hosted_context(Request({"type": "http", "app": app, "headers": []}))


def _auth(raw: str = SUB_RAW, *, key: str | None = "idem-1") -> dict[str, str]:
    headers = {"Authorization": f"Bearer {raw}"}
    if key is not None:
        headers["Idempotency-Key"] = key
    return headers


def _put(client: TestClient, raw: str = SUB_RAW, *, name: str = "demo", **body: Any):  # noqa: ANN201
    return client.put(f"/api/services/{name}/share", json=body, headers=_auth(raw))


@pytest.fixture
def recorder() -> Any:
    """Install a recording event recorder for the duration of one test."""
    fake = AsyncMock()
    eventlog.set_recorder(fake)
    yield fake
    eventlog.set_recorder(None)


def _events(recorder: AsyncMock) -> list[tuple[tuple, dict]]:
    return [(c.args, c.kwargs) for c in recorder.record.await_args_list]


# ---------------------------------------------------------------------------
# authorization
# ---------------------------------------------------------------------------


def test_a_readonly_token_can_neither_share_nor_unshare() -> None:
    """The coarse role gate first: a read-only principal never reaches the row."""
    q = _queries(_svc())
    client = _client(q)

    assert _put(client, RO_RAW).status_code == 403
    assert client.delete("/api/services/demo/share", headers=_auth(RO_RAW)).status_code == 403
    q.set_service_share.assert_not_awaited()
    q.delete_service_share.assert_not_awaited()


def test_a_non_owner_submitter_is_refused() -> None:
    q = _queries(_svc(owner="tok-sub"))
    client = _client(q)

    response = _put(client, OTHER_RAW)

    assert response.status_code == 403
    q.set_service_share.assert_not_awaited()


def test_an_admin_may_share_someone_elses_app() -> None:
    q = _queries(_svc(owner="tok-other"))

    response = _put(_client(q), ADMIN_RAW)

    assert response.status_code == 200, response.text
    q.set_service_share.assert_awaited_once_with(
        "demo", "private", job_id="svc-1", preserve_existing=False
    )


def test_a_scoped_token_may_not_share_a_service_outside_its_scope() -> None:
    """D-P25-3: scope binds the owner too — the row-shaped gate enforces it."""
    q = _queries(_svc(owner="tok-sub"))

    response = _put(_client(q), SCOPED_RAW)

    assert response.status_code == 403
    assert "other-app" in response.text
    q.set_service_share.assert_not_awaited()


def test_an_unknown_service_is_a_404_before_anything_else() -> None:
    q = _queries(None)
    client = _client(q)

    for response in (
        client.get("/api/services/ghost/share", headers=_auth()),
        _put(client, name="ghost"),
        client.delete("/api/services/ghost/share", headers=_auth()),
    ):
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"


# ---------------------------------------------------------------------------
# PUT — the happy path
# ---------------------------------------------------------------------------


def test_sharing_privately_writes_the_row_and_computes_the_url(recorder: AsyncMock) -> None:
    """The core act: one row, one computed URL, one durable event."""
    q = _queries(_svc())

    response = _put(_client(q, with_audit=True))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["service_name"] == "demo"
    assert body["access"] == "private"
    assert body["url"] == HOSTED_URL
    assert body["state"] == "ready"
    assert body["created_at"]
    q.set_service_share.assert_awaited_once_with(
        "demo", "private", job_id="svc-1", preserve_existing=False
    )

    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "share.set"
    assert row["target_type"] == "service"
    assert row["target_id"] == "demo"
    assert json.loads(row["params_redacted"]) == {
        "service": "demo",
        "access": "private",
        "consent": False,
    }

    ((_, kwargs),) = _events(recorder)
    assert kwargs["service_name"] == "demo"
    assert kwargs["kind"] == "service"
    # A private hosted URL is not a bearer capability — the cloud edge still
    # demands an owner session — so the feed may carry it.
    assert kwargs["data"] == {"access": "private", "url": HOSTED_URL}
    assert recorder.record.await_args_list[-1].args == ("share.ready",)


def test_resharing_keeps_created_at_and_repoints_access(recorder: AsyncMock) -> None:
    """An access flip is not a new share: "shared since" must survive it."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})
    manager = FakeLinkManager(entitled=True)
    client = _client(q, manager=manager)
    first = client.get("/api/services/demo/share", headers=_auth()).json()

    response = _put(client, access="public", consent=True)

    assert response.status_code == 200, response.text
    assert response.json()["access"] == "public"
    assert response.json()["created_at"] == first["created_at"]


@pytest.mark.parametrize("existing", [None, "private", "public"])
def test_private_preview_preserves_existing_access(existing, recorder: AsyncMock) -> None:
    shares = {"demo": ServiceShare(service_name="demo", access=existing)} if existing else {}
    q = _queries(_svc(), shares=shares)
    response = _put(_client(q, with_audit=True), access="private", preserve_existing=True)
    assert response.status_code == 200
    assert response.json()["access"] == (existing or "private")
    q.set_service_share.assert_awaited_once_with(
        "demo", "private", job_id="svc-1", preserve_existing=True
    )
    audit = json.loads(q.insert_audit_log.await_args.kwargs["params_redacted"])
    assert audit["access"] == (existing or "private")
    if existing == "public":
        assert "url" not in recorder.record.await_args.kwargs["data"]


def test_a_public_share_never_carries_the_url_into_the_event(recorder: AsyncMock) -> None:
    """The feed is POSTed to third-party webhook hosts; a public URL stays home."""
    q = _queries(_svc())

    response = _put(
        _client(q, manager=FakeLinkManager(entitled=True)), access="public", consent=True
    )

    assert response.status_code == 200, response.text
    assert response.json()["url"] == HOSTED_URL
    ((_, kwargs),) = _events(recorder)
    assert kwargs["data"] == {"access": "public"}


def test_a_declared_edge_auth_stands_in_for_consent() -> None:
    """D-P26-H3: advisory intent, accepted in place of consent — never as a gate."""
    job = _svc(config={"edge_auth": {"user": "ops", "password": "${secrets.EDGE_PW}"}})
    q = _queries(job)

    response = _put(_client(q, manager=FakeLinkManager(entitled=True)), access="public")

    assert response.status_code == 200, response.text
    assert response.json()["access"] == "public"


def test_a_malformed_edge_auth_does_not_stand_in_for_consent() -> None:
    """Fail closed: a typo must not silently authorize a world-reachable URL."""
    q = _queries(_svc(config={"edge_auth": {"user": "ops"}}))

    response = _put(_client(q, manager=FakeLinkManager(entitled=True)), access="public")

    assert response.status_code == 409
    assert response.json()["code"] == "share.unprotected"


# ---------------------------------------------------------------------------
# PUT — every refusal in the WP-H error vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database])
def test_only_services_can_be_shared(kind: JobKind) -> None:
    """S9: models and databases have no HTTP app semantics on the hosted path."""
    q = _queries(_svc(kind=kind))

    response = _put(_client(q))

    assert response.status_code == 422
    assert response.json()["code"] == "share.kind_unsupported"
    q.set_service_share.assert_not_awaited()


@pytest.mark.parametrize(
    ("link", "manager", "expected_hint"),
    [
        (_link(node_id=None, slug=None, domain=None), "default", "nerdit link <code>"),
        (_link(domain=None), "default", "nerdit link refresh"),
        (_link(enabled=False), None, "config set link enabled=true"),
        (_link(), None, "restart the daemon"),
    ],
    ids=["unlinked", "domain_unknown", "link_disabled", "tunnel_down"],
)
def test_a_node_that_cannot_compose_a_url_refuses_the_write(
    link: SimpleNamespace, manager: object, expected_hint: str
) -> None:
    """D-P26-H5: one code, four hints — never a row the daemon cannot address.

    The ``link_disabled`` case is the one a PR review found missing:
    ``build_link_manager`` returns ``None`` before anything else when
    ``[link].enabled`` is false, so a claimed-but-disabled node landed in the
    ``tunnel_down`` branch and was told to restart — advice a restart cannot
    satisfy. It is checked BEFORE the manager for exactly that reason, which is
    why this row passes ``manager=None``: the disabled node's real state.
    """
    q = _queries(_svc())

    response = _put(_client(q, link=link, manager=manager))

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "share.link_required"
    assert expected_hint in body["hint"]
    q.set_service_share.assert_not_awaited()


def test_a_service_deleted_under_the_write_is_a_404_not_an_orphan_share() -> None:
    """The TOCTOU the upsert closes (PR review, P26 WP-H).

    Every check above runs without the DB write lock, so a
    ``DELETE /services/{name}`` can commit between the resolve and the write.
    ``set_service_share`` re-checks under that lock and answers ``None``; the
    route must report the deletion, not a share for a name with no service —
    an orphan row nothing clears would expose the NEXT app deployed under it.
    """
    q = _queries(_svc())
    q.set_service_share = AsyncMock(return_value=None)

    response = _put(_client(q))

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_a_name_that_cannot_fit_one_dns_label_is_refused() -> None:
    """Checked at SHARE time: a row that can never resolve is not a share."""
    long_name = "a" * 60
    q = _queries(_svc(name=long_name))

    response = _put(_client(q), name=long_name)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "share.name_too_long"
    assert "63" in body["message"]
    # The message reports the LENGTH, never the composed label — the node slug
    # and the hosted name are not things to scatter through bug reports.
    assert SLUG not in json.dumps(body)
    assert DOMAIN not in json.dumps(body)
    q.set_service_share.assert_not_awaited()


def test_public_is_refused_without_the_account_entitlement() -> None:
    """The default, and the state a node drifts back to (P32 D-P32-3).

    ``hosted_public_entitled`` is ``False`` on a node the cloud has never
    asserted for and on one whose last assertion aged past the 24 h TTL — the
    route cannot tell the two apart and must not: both mean "no".
    """
    q = _queries(_svc())

    response = _put(_client(q), access="public", consent=True)

    assert response.status_code == 409
    assert response.json()["code"] == "share.not_entitled"
    assert "Pro plan" in response.json()["hint"]
    q.set_service_share.assert_not_awaited()


def test_public_is_accepted_once_the_cloud_has_pushed_the_entitlement() -> None:
    """The positive half of the same gate (P32).

    This is the ONLY thing the daemon's mirror decides: whether a local ``PUT
    share access=public`` is accepted. Anonymous traffic is still judged
    entirely by the cloud's hosted forwarder (D-P32-5), so a daemon that says
    yes here has opened nothing by itself.
    """
    q = _queries(_svc())

    response = _put(
        _client(q, manager=FakeLinkManager(entitled=True)), access="public", consent=True
    )

    assert response.status_code == 200
    assert response.json()["access"] == "public"
    q.set_service_share.assert_awaited()


def test_public_without_edge_auth_or_consent_is_refused() -> None:
    """ "Public means public" is a decision the caller has to make explicitly."""
    q = _queries(_svc())

    response = _put(_client(q, manager=FakeLinkManager(entitled=True)), access="public")

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "share.unprotected"
    assert "consent=true" in body["hint"]
    q.set_service_share.assert_not_awaited()


def test_the_body_refuses_unknown_fields_and_an_unknown_access() -> None:
    """``StrictRequestModel``: a typo'd knob is a 422, never a silent default."""
    client = _client(_queries(_svc()))

    assert _put(client, access="world").status_code == 422
    assert _put(client, acess="public").status_code == 422


# ---------------------------------------------------------------------------
# PUT — idempotency
# ---------------------------------------------------------------------------


def test_a_share_write_without_an_idempotency_key_is_a_400() -> None:
    q = _queries(_svc())

    response = _client(q).put(
        "/api/services/demo/share", json={}, headers={"Authorization": f"Bearer {SUB_RAW}"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    q.set_service_share.assert_not_awaited()
    # The same call WITH a key goes through — the header is the only difference.
    assert _put(_client(q)).status_code == 200


def test_a_replayed_share_write_returns_the_same_body_and_writes_once() -> None:
    q = _queries(_svc())
    _idem_store(q)
    client = _client(q, with_idempotency=True)

    first = _put(client)
    second = _put(client)

    assert first.status_code == 200, first.text
    assert second.status_code == 200
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    q.set_service_share.assert_awaited_once()


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


def test_an_unshared_service_reports_absence_rather_than_a_private_default() -> None:
    """A synthetic ``access: private`` body would erase a real distinction."""
    q = _queries(_svc())

    response = _client(q).get("/api/services/demo/share", headers=_auth())

    assert response.status_code == 404
    assert response.json()["code"] == "share.not_shared"


def test_a_readonly_token_may_read_a_share() -> None:
    """The read is open to any authenticated principal, like ``GET /services``."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q).get("/api/services/demo/share", headers=_auth(RO_RAW))

    assert response.status_code == 200, response.text
    assert response.json()["url"] == HOSTED_URL


def test_a_share_survives_a_down_tunnel_and_says_so() -> None:
    """The row records intent; ``state`` carries whether it answers right now."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q, manager=FakeLinkManager(state="reconnecting")).get(
        "/api/services/demo/share", headers=_auth()
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "link_down"
    # The address is still composable, so it is still reported — an operator
    # must be able to see WHAT will answer once the tunnel is back.
    assert body["url"] == HOSTED_URL


def test_a_share_on_a_node_unlinked_since_sharing_has_no_url() -> None:
    """The intent survives an unlink (S8); the address does not."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q, link=_link(node_id=None, slug=None, domain=None), manager=None).get(
        "/api/services/demo/share", headers=_auth()
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["access"] == "private"
    # No slug and no domain left: the address cannot be COMPOSED, so it is
    # reported as null rather than guessed from a half-remembered claim.
    assert body["url"] is None
    assert body["state"] == "link_down"


def test_share_reads_go_dark_in_the_unlink_before_restart_window(tmp_path: Path) -> None:
    """The gap the projection tests could not see: unlink → the SAME process.

    ``app.state.settings`` is the boot snapshot and nothing re-reads it, so
    before this fix a ``DELETE /api/link`` left every share surface composing
    ``https://demo--<old-slug>.<old-domain>/`` — an address belonging to a node
    the cloud has just revoked — for the rest of the daemon's life. The other
    unlinked-share tests hand the context a ``slug=None`` link directly, i.e.
    the state AFTER a restart; only driving the real route proves the window in
    between (PR #128 review).
    """
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})
    app = _make_app(q, with_idempotency=True, with_audit=True)
    # The link router beside the share one, on the same app: this is a
    # cross-route invariant, so a stub unlink would prove nothing.
    link_api = APIRouter(prefix="/api")
    link_api.include_router(link_router)
    app.include_router(link_api)
    config_path = tmp_path / "config.toml"
    config_path.write_bytes(
        tomli_w.dumps(
            {"link": {"enabled": True, "relay_url": "wss://relay.test", "node_id": NODE_ID}}
        ).encode("utf-8")
    )
    app.state.config_store = ConfigStore(config_path)
    app.state.settings = SimpleNamespace(
        data_dir=str(tmp_path / "data"),
        link=LinkSettings(
            enabled=True,
            relay_url="wss://relay.test",
            node_id=NODE_ID,
            slug=SLUG,
            nodes_base_domain=DOMAIN,
        ),
    )
    manager = AsyncMock()
    manager.status = lambda: SimpleNamespace(state="connected", hosted_public_entitled=False)
    manager.validate_capability = lambda token, role: False
    app.state.link_manager = manager
    client = TestClient(app, raise_server_exceptions=False)

    before = client.get("/api/services/demo/share", headers=_auth())
    assert before.json()["url"] == HOSTED_URL
    assert before.json()["state"] == "ready"

    unlinked = client.delete(
        "/api/link", headers={"Authorization": f"Bearer {ADMIN_RAW}", "Idempotency-Key": "unlink-1"}
    )
    assert unlinked.status_code == 200, unlinked.text

    after = client.get("/api/services/demo/share", headers=_auth())
    assert after.status_code == 200, after.text
    # The row survives — the intent is not what the unlink revoked — but the
    # address is null, per the ``PublicUrlEntry.url`` contract.
    assert after.json()["access"] == "private"
    assert after.json()["url"] is None
    assert after.json()["state"] == "link_down"

    # The list surface reads the same seam, so it goes dark with it.
    ctx = asyncio.run(_hosted_context_of(app))
    assert ctx.addressable is False
    assert hosted_entry(ctx, "demo").url is None

    # And the WRITE now names the command that fixes it — before the fix the
    # slug still looked live, so the operator was told to restart the daemon,
    # advice no restart can satisfy.
    refused = _put(client, ADMIN_RAW, access="private")
    assert refused.status_code == 409, refused.text
    assert refused.json()["code"] == "share.link_required"
    assert "nerdit link" in refused.json()["hint"]


def test_an_unentitled_public_share_reads_not_entitled_even_while_connected() -> None:
    """Ordering pin: ``not_entitled`` outranks ``link_down``.

    An operator told ``link_down`` would go debug the tunnel, when the thing
    that will still be true after it reconnects is the missing entitlement.
    """
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="public")})

    response = _client(q).get("/api/services/demo/share", headers=_auth())

    assert response.json()["state"] == "not_entitled"


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_unsharing_removes_the_row_and_records_the_edge(recorder: AsyncMock) -> None:
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q, with_audit=True).delete("/api/services/demo/share", headers=_auth())

    assert response.status_code == 200, response.text
    assert response.json() == {"service_name": "demo", "removed": True}
    assert q.share_table == {}

    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "share.removed"
    assert row["target_id"] == "demo"
    assert json.loads(row["params_redacted"]) == {"service": "demo"}

    ((args, kwargs),) = _events(recorder)
    assert args == ("share.removed",)
    # ``reason`` is what separates an owner's unshare from the purge route's
    # ``service_deleted`` edge on the same event type.
    assert kwargs["reason"] == "unshared"
    assert "data" not in kwargs or not kwargs.get("data")


def test_unsharing_twice_is_a_200_no_op_and_emits_nothing(recorder: AsyncMock) -> None:
    """A retrying agent converges instead of parsing "already gone" from a 404."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})
    client = _client(q)

    assert client.delete("/api/services/demo/share", headers=_auth()).status_code == 200
    second = client.delete("/api/services/demo/share", headers=_auth())

    assert second.status_code == 200
    assert second.json() == {"service_name": "demo", "removed": False}
    assert len(_events(recorder)) == 1


def test_delete_needs_no_idempotency_key() -> None:
    """The ``DELETE /services/{ident}`` precedent: a delete is its own replay."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q).delete(
        "/api/services/demo/share", headers={"Authorization": f"Bearer {SUB_RAW}"}
    )

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# audit-matcher pins (kept here so file ownership stays clean)
# ---------------------------------------------------------------------------


def test_derive_action_pins_the_p26_routes() -> None:
    """The matcher, not the router, decides what a row is called.

    Both share patterns are anchored on ``/share``, so neither can be shadowed
    by the generic ``DELETE /services/{ident}`` rule — and dropping either would
    silently downgrade the row to the space-bearing fallback action, which is
    also excluded from the idempotency body hash.
    """
    assert derive_action("PUT", "/api/services/demo/share") == ("share.set", "service", "demo")
    assert derive_action("DELETE", "/api/services/demo/share") == (
        "share.removed",
        "service",
        "demo",
    )
    # The bare-root spelling collapses to the same action (the ``/api`` strip).
    assert derive_action("PUT", "/services/demo/share")[0] == "share.set"
    # The generic service delete is untouched by the two new rules.
    assert derive_action("DELETE", "/api/services/demo") == ("service.delete", "service", "demo")
    assert derive_action("POST", "/api/link/refresh")[0] == "link.refreshed"


def test_a_share_read_is_never_audited() -> None:
    """Reads are not mutations; the middleware must not manufacture a row."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    _client(q, with_audit=True).get("/api/services/demo/share", headers=_auth())

    q.insert_audit_log.assert_not_awaited()


# ---------------------------------------------------------------------------
# P34: origin — the app behind the share
# ---------------------------------------------------------------------------
#
# The field failure: ``state: ready`` was read as "your app is live". It is not
# — it is "the hosted path is provisioned". A crash-looping app behind a ready
# share answers ``share.not_shared`` at the edge (the resolver refuses a service
# with no live endpoint), which is byte-identical to what an UNSHARED name
# returns. ``origin`` is the second, orthogonal fact, and the two are asserted
# together in every test below precisely because reading one without the other
# is the bug.


def test_a_running_app_reports_an_answering_origin_with_no_hint() -> None:
    """The happy path: link ready AND origin answering, and no advice to give."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="private")})

    response = _client(q).get("/api/services/demo/share", headers=_auth())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "ready"
    assert body["origin"] == {"status": "running", "answers": True, "hint": None}


def test_a_ready_share_over_a_dead_app_says_the_origin_is_not_answering() -> None:
    """The exact reported failure: ``state`` says ready, the URL 404s anyway."""
    q = _queries(
        _svc(status=JobStatus.failed),
        shares={"demo": ServiceShare(service_name="demo", access="private")},
    )

    response = _client(q).get("/api/services/demo/share", headers=_auth())

    assert response.status_code == 200, response.text
    body = response.json()
    # ``state`` is UNCHANGED — the vocabulary other code and the cloud read is
    # not widened, and the link really is ready.
    assert body["state"] == "ready"
    assert body["url"] == HOSTED_URL
    assert body["origin"]["answers"] is False
    assert body["origin"]["status"] == "failed"
    # The hint names the status, the symptom and the next call — an agent that
    # reads only this line still knows what to do.
    hint = body["origin"]["hint"]
    assert "failed" in hint
    assert "share.not_shared" in hint
    assert "diagnose_service" in hint


def test_degraded_counts_as_not_answering() -> None:
    """``degraded`` keeps its container but its health check is failing — which
    is precisely the "link up, app broken" state this field exists to name."""
    q = _queries(
        _svc(status=JobStatus.degraded),
        shares={"demo": ServiceShare(service_name="demo", access="private")},
    )

    body = _client(q).get("/api/services/demo/share", headers=_auth()).json()

    assert body["origin"]["answers"] is False
    assert body["origin"]["status"] == "degraded"


def test_the_share_write_response_carries_the_origin_too() -> None:
    """PUT and GET return the same view, so an agent that only ever shares
    (never re-reads) still learns its app is not up."""
    q = _queries(_svc(status=JobStatus.building))
    _idem_store(q)

    response = _put(_client(q, with_idempotency=True))

    assert response.status_code == 200, response.text
    assert response.json()["origin"]["answers"] is False


def test_origin_is_additive_and_leaves_every_pre_p34_field_intact() -> None:
    """The cloud's ``share_lookup`` reads this body; the contract may only grow."""
    q = _queries(_svc(), shares={"demo": ServiceShare(service_name="demo", access="public")})

    body = (
        _client(q, manager=FakeLinkManager(entitled=True))
        .get("/api/services/demo/share", headers=_auth())
        .json()
    )

    assert set(body) == {"service_name", "access", "url", "state", "created_at", "origin"}
    assert body["access"] == "public"
    assert body["state"] == "ready"
