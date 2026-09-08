"""Test domain ownership, validation and certificate policy through real middleware.

The mocked query table preserves DomainQueries outcomes: repeated PUTs succeed,
but another service receives domain.taken without learning the owner's identity.
Reject wildcards, IPs, IDNs, single labels and boot-reserved node names with
machine-readable reasons.

Refuse ACME before writes when disabled on the node. Otherwise persist the flag
and expose observed certificate state, defaulting to pending. Pin role, owner,
scope and in-route idempotency checks, audit fields, and distinct durable events
for add/remove/add cycles.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from urllib.parse import quote

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from nerdit.core import eventlog
from nerdit.core.proxy import _domain_route_id
from nerdit.core.proxy.certs import CertStatus
from nerdit.core.proxy.domains import ReservedNames, normalize_domain
from nerdit.daemon.audit import AuditMiddleware, _templated_path, derive_action
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.domains import router as domains_router
from nerdit.daemon.routes.proxy import router as proxy_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, ServiceEndpoint, TokenRole
from nerdit.db.rows import ServiceDomain
from tests.test_delete_purge import FakeRuntime, _app
from tests.test_delete_purge import _auth as _purge_auth
from tests.test_delete_purge import _queries as _purge_queries
from tests.test_delete_purge import _svc as _purge_svc

LEGACY = "legacy-global"  # noqa: S105 - a test literal
SUB_RAW = "sub-raw"  # noqa: S105 - a test literal
OTHER_RAW = "other-raw"  # noqa: S105 - a test literal
RO_RAW = "ro-raw"  # noqa: S105 - a test literal
ADMIN_RAW = "admin-raw"  # noqa: S105 - a test literal
SCOPED_RAW = "scoped-raw"  # noqa: S105 - a test literal

DOMAIN = "app.example.com"

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
    job_id: str = "svc-1",
) -> Job:
    return Job(
        id=job_id,
        service_name=name,
        name=name,
        kind=kind,
        gpu_count=0,
        status=JobStatus.running,
        submitted_by_token=owner,
        config=json.dumps({"image": f"nerdit-app/{name}:1"}),
    )


def _queries(*jobs: Job, domains: list[ServiceDomain] | None = None) -> AsyncMock:
    """Queries with a REAL little domain table behind the three domain methods.

    The table reproduces ``DomainQueries.add_service_domain``'s four-valued
    contract — including the ``job_id``-pinned existence check — because every
    route branch under test is a projection of one of those four outcomes.
    """
    table: dict[str, ServiceDomain] = {d.domain: d for d in (domains or [])}
    by_name = {j.service_name: j for j in jobs}
    by_id = {j.id: j for j in jobs}
    q = AsyncMock()
    q.domain_table = table
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    # ``_resolve_service`` tries the row id first; these tests address by name,
    # so the id lookup must answer a real ``None`` (an AsyncMock's default
    # return value is a truthy MagicMock, which would resolve every ident).
    q.get_job = AsyncMock(return_value=None)
    q.get_service_by_name = AsyncMock(side_effect=lambda n: by_name.get(n))
    q.list_service_shares = AsyncMock(return_value={})
    q.list_service_domains = AsyncMock(
        side_effect=lambda: sorted(table.values(), key=lambda d: (d.service_name, d.domain))
    )
    q.get_service_endpoint = AsyncMock(
        side_effect=lambda n: (
            ServiceEndpoint(
                service_name=n, job_id="svc-1", container_port=8000, host_port=9400, route=f"/{n}"
            )
            if n in by_name
            else None
        )
    )

    async def _add(
        service_name: str, domain: str, *, acme: bool | None, job_id: str
    ) -> tuple[str, ServiceDomain | None]:
        # Mirrors ``DomainQueries.add_service_domain``'s tri-state ``acme``:
        # ``None`` keeps an existing row's flag and inserts ``False``.
        job = by_id.get(job_id)
        if job is None or job.service_name != service_name or job.kind is not JobKind.service:
            return "no_service", None
        existing = table.get(domain)
        if existing is not None:
            if existing.service_name != service_name:
                return "taken", existing
            resolved = existing.acme if acme is None else acme
            if resolved != existing.acme:
                existing = existing.model_copy(update={"acme": resolved})
                table[domain] = existing
            return "exists", existing
        row = ServiceDomain(
            service_name=service_name,
            domain=domain,
            acme=bool(acme),
            created_at=datetime.now(UTC),
        )
        table[domain] = row
        return "inserted", row

    async def _remove(service_name: str, domain: str) -> bool:
        row = table.get(domain)
        if row is None or row.service_name != service_name:
            return False
        del table[domain]
        return True

    q.add_service_domain = AsyncMock(side_effect=_add)
    q.remove_service_domain = AsyncMock(side_effect=_remove)
    return q


def _idem_store(q: AsyncMock) -> dict:
    """A principal-scoped in-memory idempotency store (the share harness)."""
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


class FakeProxy:
    """``app.state.proxy_manager`` as the domain projection sees it.

    ``live`` is the set of DOMAIN NAMES whose Host route the proxy has confirmed
    live for ``service``; the manager reports route ids, so they are composed
    here through the real ``_domain_route_id`` grammar. Default EMPTY, which is
    the true state of a freshly bound domain and of every ``_queries``-only test
    that never runs a reconcile tick (Codex round 1, P2 #3831777097).
    """

    def __init__(  # noqa: PLR0913 - one knob per proxy fact under test
        self,
        *,
        available: bool = True,
        withheld: frozenset[str] = frozenset(),
        live: tuple[str, ...] = (),
        service: str = "demo",
        certs: dict[str, CertStatus] | None = None,
    ) -> None:
        self.available = available
        self.withheld_services = withheld
        self.live_domain_route_ids = frozenset(_domain_route_id(service, d) for d in live)
        self._certs = dict(certs or {})

    def cert_states(self, rows: Any) -> dict[str, CertStatus]:
        """(P26 WP2) The manager's certificate reading over the rows it is given.

        A domain the map does not mention is simply absent — exactly what the
        real manager does for a name whose leaf it could not read — so the
        route exercises the fail-closed default in
        ``views/hosted.py::domain_cert_state`` rather than a stub's guess.
        """
        return {row.domain: self._certs[row.domain] for row in rows if row.domain in self._certs}

    def validate_capability(self, token: str, role: str) -> bool:
        return False


def _make_app(  # noqa: PLR0913 - a test harness knob per fact under test
    queries: AsyncMock,
    *,
    proxy: object | None = "default",
    reserved: ReservedNames | None = None,
    https_port: int = 443,
    public_port: int | None = None,
    with_audit: bool = False,
    with_idempotency: bool = False,
    acme_enabled: bool | None = None,
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    # Mounted under ``/api`` exactly as production does, so the audit matcher
    # sees the same path it will see live.
    api = APIRouter(prefix="/api")
    api.include_router(domains_router)
    app.include_router(api)
    app.state.queries = queries
    proxy_settings = SimpleNamespace(
        mode="path",
        base_domain=None,
        scheme="https",
        https_port=https_port,
        public_port=public_port,
        hostname_override=None,
        extra_hostnames=[],
    )
    # ``None`` leaves the block ABSENT — the pre-WP2 settings shape, which is
    # what pins the ``getattr`` chain's fail-closed default. ``False``/``True``
    # install a real block.
    if acme_enabled is not None:
        proxy_settings.acme = SimpleNamespace(enabled=acme_enabled)
    app.state.settings = SimpleNamespace(proxy=proxy_settings)
    app.state.hostname = "box"
    app.state.proxy_manager = FakeProxy() if proxy == "default" else proxy
    # The boot-computed set (S-W10); the default reserves this node's own name.
    app.state.domain_reserved = (
        reserved if reserved is not None else ReservedNames(names=frozenset({"box"}))
    )
    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, **kw) -> TestClient:  # noqa: ANN003
    return TestClient(_make_app(queries, **kw), raise_server_exceptions=False)


def _auth(raw: str = SUB_RAW, *, key: str | None = "idem-1") -> dict[str, str]:
    headers = {"Authorization": f"Bearer {raw}"}
    if key is not None:
        headers["Idempotency-Key"] = key
    return headers


def _put(  # noqa: ANN201
    client: TestClient,
    raw: str = SUB_RAW,
    *,
    name: str = "demo",
    domain: str = DOMAIN,
    key: str | None = "idem-1",
    **body: Any,
):
    return client.put(
        f"/api/services/{name}/domains/{domain}", json=body, headers=_auth(raw, key=key)
    )


def _delete(client: TestClient, raw: str = SUB_RAW, *, name: str = "demo", domain: str = DOMAIN):  # noqa: ANN201
    return client.delete(
        f"/api/services/{name}/domains/{domain}", headers={"Authorization": f"Bearer {raw}"}
    )


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


def test_a_readonly_token_can_neither_bind_nor_release() -> None:
    """The coarse role gate first: a read-only principal never reaches the row."""
    q = _queries(_svc())
    client = _client(q)

    assert _put(client, RO_RAW).status_code == 403
    assert _delete(client, RO_RAW).status_code == 403
    q.add_service_domain.assert_not_awaited()
    q.remove_service_domain.assert_not_awaited()


def test_a_non_owner_submitter_is_refused() -> None:
    q = _queries(_svc(owner="tok-sub"))

    response = _put(_client(q), OTHER_RAW)

    assert response.status_code == 403
    q.add_service_domain.assert_not_awaited()


def test_an_admin_may_bind_a_domain_to_someone_elses_app() -> None:
    q = _queries(_svc(owner="tok-other"))

    response = _put(_client(q), ADMIN_RAW)

    assert response.status_code == 200, response.text
    q.add_service_domain.assert_awaited_once_with("demo", DOMAIN, acme=None, job_id="svc-1")


def test_a_scoped_token_may_not_bind_outside_its_scope() -> None:
    """D-P25-3: scope binds the owner too — the row-shaped gate enforces it."""
    q = _queries(_svc(owner="tok-sub"))

    response = _put(_client(q), SCOPED_RAW)

    assert response.status_code == 403
    assert "other-app" in response.text
    q.add_service_domain.assert_not_awaited()


def test_an_unknown_service_is_a_404_on_every_verb() -> None:
    q = _queries()
    client = _client(q)

    assert client.get("/api/services/ghost/domains", headers=_auth()).status_code == 404
    assert _put(client, name="ghost").status_code == 404
    assert _delete(client, name="ghost").status_code == 404


def test_a_domain_read_is_open_to_any_authenticated_principal() -> None:
    """The ``get_share`` rule: the body carries hostnames the operator chose to
    publish, not credentials, so a readonly token may read it."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    response = _client(q).get(
        "/api/services/demo/domains", headers={"Authorization": f"Bearer {RO_RAW}"}
    )

    assert response.status_code == 200, response.text
    assert [d["domain"] for d in response.json()["domains"]] == [DOMAIN]


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


def test_a_service_with_no_domains_lists_an_empty_array_not_a_404() -> None:
    """Absence of domains is a normal state; only an unknown SERVICE is a 404,
    so an agent enumerating exposure never has to read a status code to tell
    "none" from "no such app"."""
    response = _client(_queries(_svc())).get("/api/services/demo/domains", headers=_auth())

    assert response.status_code == 200, response.text
    assert response.json() == {"service_name": "demo", "domains": []}


def test_listing_projects_url_and_state_from_the_live_proxy() -> None:
    q = _queries(
        _svc(),
        domains=[
            ServiceDomain(service_name="demo", domain="a.example.com"),
            ServiceDomain(service_name="demo", domain="b.example.com"),
            ServiceDomain(service_name="other", domain="c.example.com"),
        ],
    )
    proxy = FakeProxy(live=("a.example.com", "b.example.com"))

    body = _client(q, proxy=proxy).get("/api/services/demo/domains", headers=_auth()).json()

    assert [d["domain"] for d in body["domains"]] == ["a.example.com", "b.example.com"]
    assert [d["url"] for d in body["domains"]] == [
        "https://a.example.com/",
        "https://b.example.com/",
    ]
    assert {d["state"] for d in body["domains"]} == {"ready"}
    # Another service's row is never smeared onto this one.
    assert all(d["service_name"] == "demo" for d in body["domains"])


def test_a_bound_domain_whose_host_route_is_not_live_yet_is_withheld() -> None:
    """(Codex round 1, P2 #3831777097) The proxy is up, the app is routed and
    nothing is edge-auth-withheld — and the domain still does not answer,
    because its Host route lands on the next reconcile tick (and never at all
    while ``_converge_tls`` keeps failing, which withholds every domain spec).
    Readiness is the ROUTE, not the row plus a healthy default."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    body = _client(q).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["state"] == "withheld"
    assert body["domains"][0]["url"] == f"https://{DOMAIN}/"


def test_only_the_domain_whose_route_is_live_is_ready() -> None:
    """Per ROW, not per service: one failed upsert can leave a service with one
    live domain route and one absent, so a single verdict for the whole service
    would advertise a URL that answers nothing."""
    q = _queries(
        _svc(),
        domains=[
            ServiceDomain(service_name="demo", domain="a.example.com"),
            ServiceDomain(service_name="demo", domain="b.example.com"),
        ],
    )
    proxy = FakeProxy(live=("a.example.com",))

    body = _client(q, proxy=proxy).get("/api/services/demo/domains", headers=_auth()).json()

    assert [d["state"] for d in body["domains"]] == ["ready", "withheld"]


def test_a_bound_domain_is_withheld_while_the_proxy_is_off() -> None:
    """The row is the presence, the state is the reachability: a domain bound on
    a daemon with the URL layer off is listed, honestly, as not answering."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    body = _client(q, proxy=None).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["state"] == "withheld"
    assert body["domains"][0]["url"] == f"https://{DOMAIN}/"


def test_an_edge_auth_withheld_service_reports_withheld_domains() -> None:
    """(S-W5) The proxy withholds EVERY route of a service whose ``edge_auth``
    secret does not resolve — domain routes included — so the surface must not
    promise a URL that answers 404."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])
    proxy = FakeProxy(withheld=frozenset({"demo"}))

    body = _client(q, proxy=proxy).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["state"] == "withheld"


def test_the_url_follows_the_public_url_port_rule() -> None:
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    body = _client(q, https_port=8443).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["url"] == f"https://{DOMAIN}:8443/"


# ---------------------------------------------------------------------------
# PUT — the happy path and its idempotency
# ---------------------------------------------------------------------------


def test_binding_a_domain_returns_the_row_the_url_and_created_true(recorder: AsyncMock) -> None:
    q = _queries(_svc())

    response = _put(_client(q))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["service_name"] == "demo"
    assert body["domain"] == DOMAIN
    assert body["url"] == f"https://{DOMAIN}/"
    # NOT ``ready``: the PUT writes the row, the reconcile tick writes the Host
    # route (S-W3 registers the default route only). The documented ~5 s lag is
    # now visible on the surface instead of being papered over (Codex round 1).
    assert body["state"] == "withheld"
    assert body["created"] is True
    assert body["acme"] is False
    assert body["kind"] == "domain"
    assert set(q.domain_table) == {DOMAIN}

    ((args, kwargs),) = _events(recorder)
    assert args == ("domain.added",)
    assert kwargs["service_name"] == "demo"
    # Public DNS the operator chose — safe to carry to a webhook host, and the
    # only thing a consumer can act on.
    assert kwargs["data"] == {"domain": DOMAIN, "acme": False}


def test_re_putting_the_same_domain_is_created_false_and_mints_no_event(
    recorder: AsyncMock,
) -> None:
    """A re-PUT is a converging no-op, not a re-announcement: the exposure did
    not change, so the feed must not say it did."""
    q = _queries(_svc())
    client = _client(q)

    first = _put(client)
    second = _put(client)

    assert first.json()["created"] is True
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    assert len(q.domain_table) == 1
    assert len(_events(recorder)) == 1


def test_the_domain_is_folded_and_the_unfolded_spelling_removes_it() -> None:
    """``App.Example.COM.`` and ``app.example.com`` are ONE resource: the id is
    normalised (trim → case-fold → drop one trailing dot) on both verbs, so a
    caller cannot end up with two rows for one name — or a row it cannot
    delete because it typed the other spelling."""
    q = _queries(_svc())
    client = _client(q)

    stored = _put(client, domain="App.Example.COM.")

    assert stored.status_code == 200, stored.text
    assert stored.json()["domain"] == DOMAIN
    assert set(q.domain_table) == {DOMAIN}

    removed = _delete(client, domain="APP.example.com")
    assert removed.json() == {"service_name": "demo", "domain": DOMAIN, "removed": True}
    assert q.domain_table == {}


def test_a_replayed_idempotency_key_returns_the_same_body() -> None:
    q = _queries(_svc())
    _idem_store(q)
    client = _client(q, with_idempotency=True)

    first = _put(client)
    second = _put(client)

    assert first.status_code == 200, first.text
    assert second.status_code == 200
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    q.add_service_domain.assert_awaited_once()


def test_a_put_without_an_idempotency_key_is_refused_before_any_write() -> None:
    q = _queries(_svc())

    response = _put(_client(q), key=None)

    assert response.status_code == 400
    assert response.json()["code"] == "idempotency_key_required"
    q.add_service_domain.assert_not_awaited()


# ---------------------------------------------------------------------------
# PUT — refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "reason"),
    [
        ("*.example.com", "wildcard"),
        ("192.168.1.50", "ip_literal"),
        ("bücher.example", "idn"),
        ("xn--bcher-kva.example", "idn"),
        ("intranet", "single_label"),
        ("x.example:8443", "has_port"),
        ("-a.example", "grammar"),
        # ``box`` itself is refused one check earlier as ``single_label`` — the
        # order is load-bearing, and a bare LAN hostname is not a domain
        # whichever way you look at it. Suffix membership is what makes the
        # reserved set an ownership gate.
        ("api.box", "reserved"),
    ],
)
def test_every_invalid_domain_carries_its_machine_reason(domain: str, reason: str) -> None:
    """One assertion per refusal token: the message is for a human, the token in
    ``detail.reason`` is what an agent branches on. Nothing is written.

    The domain is percent-encoded into the path exactly as the client sends it
    (``quote(domain, safe="")``): it is a path SEGMENT, so a caller pasting a
    URL must not be able to split it into extra segments.
    """
    q = _queries(_svc())

    response = _put(_client(q), domain=quote(domain, safe=""))

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["code"] == "domain.invalid"
    assert body["detail"]["reason"] == reason
    assert "wildcards, IP literals" in body["hint"]
    assert q.domain_table == {}


def test_a_url_shaped_input_is_refused_as_not_bare_not_lost_as_a_404() -> None:
    """The domain parameter is a ``:path`` converter: a pasted URL (or a
    ``%2F``-encoded one, which the ASGI layer decodes before routing) reaches
    the validator and is refused ``not_bare`` with the agent-facing hint —
    the live run found the plain 404 told the caller nothing. Nothing is
    written either way.
    """
    q = _queries(_svc())

    for spelled in ("https://x.example", quote("https://x.example", safe="")):
        response = _put(_client(q), domain=spelled)
        assert response.status_code == 422, response.text
        assert response.json()["detail"]["reason"] == "not_bare"
    assert q.domain_table == {}


def test_the_refusal_reason_rides_at_the_top_level_for_the_mcp_projection() -> None:
    """``mcp/errors.py`` drops ``detail`` by design and copies every other
    envelope key through, so ``reason`` must also be a top-level extra or an
    agent never sees the token."""
    q = _queries(_svc())

    body = _put(_client(q), domain="*.example.com").json()

    assert body["reason"] == "wildcard"
    assert body["detail"]["reason"] == "wildcard"


def test_the_nodes_own_bare_hostname_is_refused_one_check_earlier() -> None:
    """``box`` is in the reserved set AND a single label. The single-label check
    wins by design — it is the more explanatory refusal, and every public-cert
    path (WP2) refuses such a name anyway."""
    q = _queries(_svc())

    response = _put(_client(q), domain="box")

    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "single_label"
    assert q.domain_table == {}


def test_a_name_under_the_nodes_base_domain_is_reserved() -> None:
    """Suffix membership, not equality: in subdomain mode every service's own
    Host lives under ``base_domain``, so a custom domain landing inside that
    zone could shadow one (or be shadowed by one)."""
    q = _queries(_svc())
    reserved = ReservedNames(names=frozenset({"box", "dev.lan"}))

    response = _put(_client(q, reserved=reserved), domain="x.dev.lan")

    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "reserved"


def test_acme_true_is_refused_before_any_write_when_the_node_has_acme_off() -> None:
    """A stored row claiming a public certificate the operator will never get is
    worse than an honest refusal — so ``acme`` is judged before the insert.

    The envelope carries ``reason`` as a TOP-LEVEL extra, not only in
    ``detail``: the MCP mapper drops ``detail`` by design, so an agent that has
    to distinguish "this node does not do ACME" from every other 409 needs the
    token where it survives the projection.
    """
    q = _queries(_svc())

    response = _put(_client(q, acme_enabled=False), ADMIN_RAW, acme=True)

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "domain.acme_disabled"
    assert body["reason"] == "acme_disabled"
    assert "proxy.acme" in body["hint"]
    q.add_service_domain.assert_not_awaited()
    assert q.domain_table == {}


def test_acme_true_is_refused_on_a_settings_object_that_has_no_acme_block() -> None:
    """Fail-closed on the shape, not only on the value: a daemon whose settings
    predate ``[proxy.acme]`` must refuse, never accept-and-store. The opposite
    default would let a mis-wired app factory write rows asking for certificates
    no Caddy on this box will ever request."""
    q = _queries(_svc())

    response = _put(_client(q), ADMIN_RAW, acme=True)  # no ``acme`` attribute at all

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "domain.acme_disabled"
    q.add_service_domain.assert_not_awaited()


def test_acme_true_is_stored_as_sent_when_the_node_enables_acme(recorder: AsyncMock) -> None:
    """(P26 WP2) With ``[proxy.acme].enabled`` the flag is DATA: it reaches the
    insert unchanged and comes back on the row, and the certificate fact beside
    it is the proxy manager's own reading — here ``pending``, the honest answer
    for a name whose leaf is not on disk yet."""
    q = _queries(_svc())
    proxy = FakeProxy(certs={DOMAIN: CertStatus("pending")})

    response = _put(_client(q, proxy=proxy, acme_enabled=True), ADMIN_RAW, acme=True)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["acme"] is True
    assert body["created"] is True
    assert body["cert_state"] == "pending"
    # Only issued/expired carry an expiry; ``pending`` has nothing to report.
    assert body["cert_not_after"] is None
    q.add_service_domain.assert_awaited_once_with("demo", DOMAIN, acme=True, job_id="svc-1")
    assert q.domain_table[DOMAIN].acme is True
    ((_args, kwargs),) = _events(recorder)
    assert kwargs["data"] == {"domain": DOMAIN, "acme": True}


def test_an_issued_domain_reports_its_expiry() -> None:
    """``cert_not_after`` is the one datum an operator cannot derive: it says
    when the automation has to have worked again."""
    expiry = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN, acme=True)])
    proxy = FakeProxy(certs={DOMAIN: CertStatus("issued", expiry)})

    body = (
        _client(q, proxy=proxy, acme_enabled=True)
        .get("/api/services/demo/domains", headers=_auth())
        .json()
    )

    row = body["domains"][0]
    assert row["cert_state"] == "issued"
    assert datetime.fromisoformat(row["cert_not_after"]) == expiry


def test_an_acme_row_the_manager_cannot_report_on_is_pending_never_issued() -> None:
    """Fail-closed: an ``acme=1`` row with no reading from the proxy reads
    ``pending``. Reporting ``issued`` for a certificate nobody has seen would
    tell an operator to skip ``nerdit trust`` and leave every client on a
    warning page."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN, acme=True)])

    body = _client(q, acme_enabled=True).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["cert_state"] == "pending"


def test_a_plain_row_is_internal_not_pending() -> None:
    """``acme=false`` is the DEFAULT binding, not a defect: it reports
    ``internal`` (this node's own CA), which is a fact an operator acts on with
    ``nerdit trust`` — never a state that looks like a stuck issuance."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    body = _client(q).get("/api/services/demo/domains", headers=_auth()).json()

    assert body["domains"][0]["cert_state"] == "internal"
    assert body["domains"][0]["cert_not_after"] is None


def test_re_putting_to_flip_acme_is_created_false_and_mints_no_event(
    recorder: AsyncMock,
) -> None:
    """The issuer changed; the EXPOSURE did not. A flip is a converging write —
    ``created: false``, the column updated, and no ``domain.added`` on the feed
    to re-announce a URL that has been advertised all along."""
    q = _queries(_svc())
    client = _client(q, acme_enabled=True)

    first = _put(client, acme=False)
    second = _put(client, ADMIN_RAW, acme=True)

    assert first.json()["created"] is True
    assert second.status_code == 200, second.text
    assert second.json()["created"] is False
    assert len(q.domain_table) == 1
    assert len(_events(recorder)) == 1


def test_the_add_audit_row_records_the_acme_decision_that_was_stored() -> None:
    """The audit row is the only durable record of a flip (it mints no event),
    so it has to carry the decision as sent."""
    q = _queries(_svc())

    _put(_client(q, with_audit=True, acme_enabled=True), ADMIN_RAW, acme=True)

    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "domain.added"
    assert json.loads(row["params_redacted"]) == {
        "service": "demo",
        "domain": DOMAIN,
        "acme": True,
    }


def test_a_re_put_with_no_body_keeps_an_issued_certificate(recorder: AsyncMock) -> None:
    """(P26 WP2 review round 1) The downgrade footgun, closed at the route.

    ``add_domain(app, name)`` is documented as an idempotent PUT and is exactly
    what an agent re-runs to read a URL back; every client defaults ``acme`` to
    false. With a two-valued flag that re-run flipped the column, the next tick
    dropped the acme policy, Caddy re-keyed the name to the internal issuer and
    a browser that worked yesterday showed a warning — with a 200, no event, and
    only an audit row to show for it.
    """
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN, acme=True)])
    client = _client(q, acme_enabled=True)

    response = _put(client)  # no ``acme`` key at all

    assert response.status_code == 200, response.text
    assert response.json()["created"] is False
    assert response.json()["acme"] is True
    q.add_service_domain.assert_awaited_once_with("demo", DOMAIN, acme=None, job_id="svc-1")
    assert q.domain_table[DOMAIN].acme is True


def test_an_explicit_acme_false_still_downgrades(recorder: AsyncMock) -> None:
    """Tri-state is not one-way: saying ``acme: false`` out loud is still one
    call, and still takes the name back to the internal CA. Only the SILENT
    downgrade is gone."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN, acme=True)])

    response = _put(_client(q, acme_enabled=True), acme=False)

    assert response.status_code == 200, response.text
    assert response.json()["acme"] is False
    assert q.domain_table[DOMAIN].acme is False


def test_a_submitter_may_not_request_a_public_certificate() -> None:
    """(review round 1) The node holds ONE ACME account and the CA counts its
    rate limits against it, so a per-app write must not be able to spend a
    node-wide budget: ``acme: true`` is admin-only.

    Without this a submitter token (or a buggy agent holding one) could park
    dozens of names it does not control as ``acme=1``; each becomes a subject of
    the first automation policy, Caddy orders on the next tick, and the failed
    validations starve renewals for the admin's real public domains.
    """
    q = _queries(_svc())

    response = _put(_client(q, acme_enabled=True), acme=True)

    assert response.status_code == 403, response.text
    body = response.json()
    assert body["code"] == "domain.acme_forbidden"
    assert body["reason"] == "acme_admin_only"
    q.add_service_domain.assert_not_awaited()
    assert q.domain_table == {}


def test_a_submitter_acme_denial_is_audited_by_digest() -> None:
    """(Codex round 2) The ACME gate is the one refusal a submitter can trigger
    by hand, and it exists to make probing visible — so the denial has to carry
    the same digest every other refusal carries.

    The gate shipped ABOVE the ``audit_params`` stamp, so the row landed with
    ``params_redacted = NULL``: the service still rode the path into
    ``target_id``, but the intent (``acme: true``) and the per-name digest were
    both lost, and an operator could not tell one token retrying one name from
    one token spraying a hundred. The stamp now precedes the gate.
    """
    q = _queries(_svc())

    response = _put(_client(q, with_audit=True, acme_enabled=True), acme=True)

    assert response.status_code == 403, response.text
    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "domain.added"
    assert row["result"] == "denied"
    assert row["target_id"] == "demo"
    params = json.loads(row["params_redacted"])
    assert params == {
        "service": "demo",
        "domain": "<unvalidated>",
        "domain_sha256": _digest(DOMAIN),
        "acme": True,
    }
    # Still a digest, never the bytes as sent — the gate fires before validation.
    assert DOMAIN not in json.dumps(params)
    assert q.domain_table == {}


def test_a_submitter_binds_an_internal_ca_domain_exactly_as_before() -> None:
    """The gate is on the FLAG, not on the surface: everything a submitter could
    do before still works, including an explicit ``acme: false``."""
    q = _queries(_svc())
    client = _client(q, acme_enabled=True)

    assert _put(client).status_code == 200
    assert _put(client, key="idem-2", acme=False).status_code == 200
    assert q.domain_table[DOMAIN].acme is False


def test_a_domain_bound_to_another_service_is_taken_and_never_names_it() -> None:
    """The takeover story: a name is a global claim, and the refusal must not
    become a way for an owner-scoped principal to enumerate another owner's app
    names by probing."""
    q = _queries(
        _svc(name="demo", job_id="svc-1"),
        _svc(name="secret-app", job_id="svc-2", owner="tok-other"),
        domains=[ServiceDomain(service_name="secret-app", domain=DOMAIN)],
    )

    response = _put(_client(q), ADMIN_RAW)

    assert response.status_code == 409, response.text
    body = response.json()
    assert body["code"] == "domain.taken"
    assert DOMAIN in body["message"]
    assert "secret-app" not in json.dumps(body)
    # The row is untouched — a second service never silently takes a name.
    assert q.domain_table[DOMAIN].service_name == "secret-app"


def test_a_model_cannot_carry_a_custom_domain() -> None:
    """A model is loopback-only by design; there is no HTTP app for a Host route
    to carry (the ``share.kind_unsupported`` rule)."""
    q = _queries(_svc(kind=JobKind.model))

    response = _put(_client(q))

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "domain.kind_unsupported"
    q.add_service_domain.assert_not_awaited()


def test_a_service_deleted_mid_request_is_a_404_and_writes_nothing() -> None:
    """The insert re-checks the AUTHORIZED job id under the DB write lock; when
    it is gone the honest answer is the 404 the resolve would have given a
    moment later, not a row for a name with no service behind it."""
    q = _queries(_svc())
    q.add_service_domain = AsyncMock(return_value=("no_service", None))

    response = _put(_client(q))

    assert response.status_code == 404
    assert q.domain_table == {}


def test_the_request_body_rejects_unknown_fields() -> None:
    """``StrictRequestModel``: a typo'd key is a 422, never a silently ignored
    intent (P22 #2/#24)."""
    q = _queries(_svc())

    response = _put(_client(q), acme=False, cert="please")

    assert response.status_code == 422
    q.add_service_domain.assert_not_awaited()


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_removing_a_domain_drops_the_row_and_records_the_edge(recorder: AsyncMock) -> None:
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    response = _delete(_client(q, with_audit=True))

    assert response.status_code == 200, response.text
    assert response.json() == {"service_name": "demo", "domain": DOMAIN, "removed": True}
    assert q.domain_table == {}

    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "domain.removed"
    assert row["target_id"] == "demo"
    assert json.loads(row["params_redacted"]) == {"service": "demo", "domain": DOMAIN}

    ((args, kwargs),) = _events(recorder)
    assert args == ("domain.removed",)
    # ``reason`` separates an owner's removal from the purge route's cascade.
    assert kwargs["reason"] == "removed"
    assert kwargs["data"] == {"domain": DOMAIN}


def test_removing_twice_is_a_200_no_op_and_emits_nothing(recorder: AsyncMock) -> None:
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])
    client = _client(q)

    assert _delete(client).status_code == 200
    second = _delete(client)

    assert second.status_code == 200
    assert second.json()["removed"] is False
    assert len(_events(recorder)) == 1


def test_delete_needs_no_idempotency_key() -> None:
    """The ``DELETE /services/{ident}`` precedent: a delete is its own replay."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    assert _delete(_client(q)).status_code == 200


def test_a_malformed_domain_on_delete_simply_removes_nothing() -> None:
    """No grammar check on the way out: validation is an ownership gate on the
    way IN, and applying it here would leave a domain bound by an earlier, laxer
    release with no way to unbind it."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    response = _delete(_client(q), domain="*.example.com")

    assert response.status_code == 200, response.text
    assert response.json()["removed"] is False
    assert set(q.domain_table) == {DOMAIN}


def test_deleting_a_domain_another_service_holds_is_a_no_op() -> None:
    """The delete surface must not be a way around ``domain.taken``."""
    q = _queries(
        _svc(name="demo", job_id="svc-1"),
        _svc(name="other", job_id="svc-2"),
        domains=[ServiceDomain(service_name="other", domain=DOMAIN)],
    )

    response = _delete(_client(q))

    assert response.status_code == 200
    assert response.json()["removed"] is False
    assert q.domain_table[DOMAIN].service_name == "other"


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_the_add_audit_row_carries_the_decision_and_nothing_else() -> None:
    q = _queries(_svc())

    _put(_client(q, with_audit=True))

    row = q.insert_audit_log.await_args_list[-1].kwargs
    assert row["action"] == "domain.added"
    assert row["target_type"] == "service"
    assert row["target_id"] == "demo"
    assert json.loads(row["params_redacted"]) == {
        "service": "demo",
        "domain": DOMAIN,
        "acme": False,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def test_a_refused_write_is_audited_by_digest_never_by_the_name_as_sent() -> None:
    """``audit_params`` is stamped BEFORE the refusals, so a denied attempt to
    bind a name is recorded — which is what makes probing visible.

    What it records is the DIGEST of the bytes as sent, not the bytes (Codex
    round 1, P1 #3831777111). Both properties have to hold at once: a refused
    homograph must not be recorded under the ASCII name it resembles (a
    different, bindable resource), and an arbitrary path segment must never be
    persisted verbatim because it can be a mis-pasted credential. A digest over
    the submitted bytes is exactly the fact that separates the two spellings
    without keeping either. Once validation accepts, the canonical name — a
    well-formed DNS name by then — is stamped in the clear.
    """
    q = _queries(_svc())

    _put(_client(q, with_audit=True), ADMIN_RAW, domain="App.Example.COM.", acme=True)
    row = q.insert_audit_log.await_args_list[-1].kwargs
    # acme=true is refused AFTER validation, so the canonical name is recorded.
    assert json.loads(row["params_redacted"])["domain"] == DOMAIN
    assert json.loads(row["params_redacted"])["acme"] is True

    homograph = "\u017fervice.example.com"  # LATIN SMALL LETTER LONG S
    _put(_client(q, with_audit=True), domain=quote(homograph, safe=""))
    params = json.loads(q.insert_audit_log.await_args_list[-1].kwargs["params_redacted"])
    assert params["domain"] == "<refused>"
    assert params["reason"] == "idn"
    # The homograph is recorded as ITSELF and never collapses onto the ASCII
    # name it looks like — the whole point of hashing the bytes as sent.
    assert params["domain_sha256"] == _digest(homograph)
    assert params["domain_sha256"] != _digest("service.example.com")
    assert homograph not in json.dumps(params)
    assert q.domain_table == {}


def test_a_token_shaped_domain_never_reaches_the_audit_row_or_the_response() -> None:
    """(Codex round 1, P1 #3831777111) The ``domain`` path segment is arbitrary
    caller input, so a mis-paste can be a bearer token — and it used to land
    verbatim in ``params_redacted`` (``audit_params`` masks by KEY name, and
    "domain" is not a secret key) and be interpolated into the 422 message,
    which ``mcp/errors.py`` copies into the agent transcript. Neither may
    carry it, whichever refusal fires."""
    secret = "nrd_" + "a1b2c3d4" * 6  # noqa: S105 - a test literal, token-shaped
    for spelled, reason in (
        (secret, "single_label"),
        (f"Bearer {secret}", "whitespace"),
        (f"{secret}.example.com:8443", "has_port"),
        # (Codex round 2, #3835632982) The URL-shaped paste — the one spelling
        # that carries SLASHES. The route is ``{domain:path}`` (WP1 F1) and the
        # middleware sees the DECODED path, so this arrives with real slashes:
        # ``params_redacted`` was never the only sink, ``action`` is one too.
        (f"https://app.example.com/{secret}", "not_bare"),
    ):
        q = _queries(_svc())
        response = _put(_client(q, with_audit=True), domain=quote(spelled, safe=""))

        assert response.status_code == 422, response.text
        assert response.json()["detail"]["reason"] == reason
        assert secret not in response.text
        row = q.insert_audit_log.await_args_list[-1].kwargs
        params = json.loads(row["params_redacted"])
        assert params["domain"] == "<refused>"
        assert params["domain_sha256"] == _digest(spelled)
        assert secret not in json.dumps(params)
        # The ACTION is a sink too: it is echoed by ``GET /api/audit`` (and its
        # ``action_prefix`` filter), by MCP ``get_audit``, and verbatim as both
        # the ``type`` and the ``action`` of the admin ``audit.*`` SSE frame. A
        # slash-bearing paste must therefore still hit the MAPPED rule instead
        # of falling through to the ``f"{method} {path}"`` fallback.
        assert row["action"] == "domain.added"
        assert row["target_type"] == "service"
        assert row["target_id"] == "demo"
        assert secret not in row["action"]
        assert q.domain_table == {}


def test_a_token_shaped_domain_on_delete_is_redacted_too() -> None:
    """The DELETE path stamped ``normalize_domain(domain)`` with no validation
    at all, so it leaked the same way — and echoed it back in the body. The
    delete itself stays deliberately lax (it must be able to unbind whatever is
    stored); only what is RECORDED changes."""
    secret = "nrd_" + "f00dcafe" * 6  # noqa: S105 - a test literal, token-shaped
    # Both spellings: the bare token, and the slash-bearing URL paste that the
    # ``{domain:path}`` route accepts verbatim (Codex round 2, #3835632982).
    for spelled in (secret, f"https://app.example.com/{secret}"):
        q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

        response = _delete(_client(q, with_audit=True), domain=quote(spelled, safe=""))

        assert response.status_code == 200, response.text
        assert response.json() == {"service_name": "demo", "domain": "<refused>", "removed": False}
        assert secret not in response.text
        row = q.insert_audit_log.await_args_list[-1].kwargs
        params = json.loads(row["params_redacted"])
        assert params == {
            "service": "demo",
            "domain": "<refused>",
            "domain_sha256": _digest(spelled),
        }
        # The action is the second sink and carries no input either.
        assert row["action"] == "domain.removed"
        assert row["target_type"] == "service"
        assert row["target_id"] == "demo"
        assert secret not in row["action"]
        # And the lax removal is untouched: the real row is still there.
        assert set(q.domain_table) == {DOMAIN}


def test_the_refusal_message_names_the_reason_not_the_input() -> None:
    """The human half of a 422 is derived from the reason token alone
    (``DomainInvalid.public_message``). ``message`` — which quotes the input —
    stays inside the daemon."""
    q = _queries(_svc())

    body = _put(_client(q), domain=quote("*.example.com", safe="")).json()

    assert body["message"] == "A domain must not be a wildcard; bind each name explicitly."
    assert "*.example.com" not in body["message"]


def test_derive_action_pins_the_wp1_routes() -> None:
    """The matcher, not the router, decides what a row is called. Both patterns
    are anchored on ``/domains/<name>``, so neither can be shadowed by the
    generic ``DELETE /services/{ident}`` rule — and dropping either would
    silently downgrade the row to the space-bearing fallback action."""
    assert derive_action("PUT", f"/api/services/demo/domains/{DOMAIN}") == (
        "domain.added",
        "service",
        "demo",
    )
    assert derive_action("DELETE", f"/api/services/demo/domains/{DOMAIN}") == (
        "domain.removed",
        "service",
        "demo",
    )
    # The bare-root spelling collapses to the same action (the ``/api`` strip).
    assert derive_action("PUT", f"/services/demo/domains/{DOMAIN}")[0] == "domain.added"
    # The generic service delete and the share pair are untouched.
    assert derive_action("DELETE", "/api/services/demo") == ("service.delete", "service", "demo")
    assert derive_action("DELETE", "/api/services/demo/share")[0] == "share.removed"


def test_a_slash_bearing_domain_still_matches_the_mapped_rule() -> None:
    """(Codex round 2, #3835632982) The routes are ``{domain:path}`` and the
    middleware sees the DECODED path, so ``%2F`` in a mis-pasted URL is a real
    slash by the time ``derive_action`` runs. With a ``[^/]+`` domain group the
    rule missed and the whole raw path became the action — echoed by
    ``GET /api/audit`` and by the admin ``audit.*`` SSE frame — and the row lost
    its service attribution as collateral."""
    pasted = "https://app.example.com/ghp_" + "0" * 36
    assert derive_action("PUT", f"/api/services/demo/domains/{pasted}") == (
        "domain.added",
        "service",
        "demo",
    )
    assert derive_action("DELETE", f"/api/services/demo/domains/{pasted}") == (
        "domain.removed",
        "service",
        "demo",
    )
    # Defence in depth: even if a future ``/domains/`` mutation is NOT mapped,
    # the fallback templates the whole remainder, not one slash-split segment.
    assert _templated_path("/services/x/domains/a/ghp_x") == "/services/{id}/domains/{domain}"
    assert _templated_path("/services/x/domains/app.example.com") == (
        "/services/{id}/domains/{domain}"
    )
    # Paths without a ``domains`` segment keep the id-collection templating.
    assert _templated_path("/services/x/logs") == "/services/{id}/logs"


def test_a_domain_read_is_never_audited() -> None:
    """Reads are not mutations; the middleware must not manufacture a row."""
    q = _queries(_svc(), domains=[ServiceDomain(service_name="demo", domain=DOMAIN)])

    _client(q, with_audit=True).get("/api/services/demo/domains", headers=_auth())

    q.insert_audit_log.assert_not_awaited()


# ---------------------------------------------------------------------------
# the reserved set is the one computed at boot
# ---------------------------------------------------------------------------


def test_the_reserved_set_is_read_from_app_state_not_recomputed() -> None:
    """(S-W10) The route reads ``app.state.domain_reserved``. Recomputing per
    request could only ever read a ``[proxy]`` change the running proxy has not
    applied — a window in which a racing PUT could bind a name the live Caddy is
    about to claim."""
    q = _queries(_svc())
    app = _make_app(q, reserved=ReservedNames(names=frozenset({"pinned.example"})))
    client = TestClient(app, raise_server_exceptions=False)

    # ``box`` is the live hostname but NOT in the stamped set, so it passes; the
    # stamped name is refused. Both facts together prove which set was used.
    assert (
        client.put("/api/services/demo/domains/a.box", json={}, headers=_auth()).status_code == 200
    )
    refused = client.put(
        "/api/services/demo/domains/x.pinned.example", json={}, headers=_auth(key="idem-2")
    )
    assert refused.status_code == 422
    assert refused.json()["detail"]["reason"] == "reserved"


def test_a_bare_app_state_falls_back_to_grammar_only_validation() -> None:
    """A route test with no stamped set (and no settings) still validates the
    grammar; the fallback narrows what is refused, it never widens it.

    Said plainly, because the ``_reserved`` docstring used to claim the
    opposite: the bare-state reserved set is EMPTY, so the ownership check does
    not run — that is permissive, not fail-closed. It is safe only because
    ``server.py`` stamps the real set before any request is served.
    """
    q = _queries(_svc())
    app = _make_app(q)
    app.state.domain_reserved = None
    app.state.settings = None
    client = TestClient(app, raise_server_exceptions=False)

    assert (
        client.put(f"/api/services/demo/domains/{DOMAIN}", json={}, headers=_auth()).status_code
        == 200
    )
    bad = client.put(
        "/api/services/demo/domains/*.example.com", json={}, headers=_auth(key="idem-2")
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["reason"] == "wildcard"


def test_normalize_domain_is_what_the_route_stores() -> None:
    """The route and the validator agree on one canonical form — pinned here so
    a later change to either cannot make the stored name and the audited name
    diverge."""
    assert normalize_domain("  App.Example.COM.  ") == DOMAIN


# ---------------------------------------------------------------------------
# the purge cascade — one edge per name
# ---------------------------------------------------------------------------


def _delete_app(monkeypatch, tmp_path, *, removed: list[str]):
    """The real ``DELETE /services/{ident}`` route with a recorder double.

    ``delete_service_checked`` is a mock here (the ``test_service_shares_db``
    harness verbatim), so it stands in for the real transaction by invoking
    ``on_domains_removed`` the way that transaction does — which is the ONLY
    channel the route learns the names through: they are unrecoverable after
    the commit and unreadable before it without a race.
    """
    queries = _purge_queries(_purge_svc(), workloads=[])

    async def _delete(*_args, on_share_removed=None, on_domains_removed=None, **_kw):  # noqa: ANN001, ANN202
        if on_share_removed is not None:
            on_share_removed(False)
        if on_domains_removed is not None:
            on_domains_removed(list(removed))
        return []

    queries.delete_service_checked = AsyncMock(side_effect=_delete)
    fake = AsyncMock()
    monkeypatch.setattr("nerdit.daemon.service_purge.get_recorder", lambda: fake)
    return _app(runtime=FakeRuntime([]), queries=queries, data_dir=tmp_path), fake


def test_deleting_a_service_emits_one_domain_removed_per_name(monkeypatch, tmp_path) -> None:
    """The cascade an operator must be able to see in the feed: the rows go with
    the service, so a consumer that heard ``domain.added`` has to hear the
    matching removal for every name — not one summary edge."""
    app, fake = _delete_app(monkeypatch, tmp_path, removed=["a.example.com", "b.example.com"])

    with TestClient(app) as client:
        assert client.delete("/services/a", headers=_purge_auth()).status_code == 200

    calls = [(c.args, c.kwargs) for c in fake.record.await_args_list]
    assert [args[0] for args, _ in calls] == ["domain.removed", "domain.removed"]
    assert [kwargs["data"]["domain"] for _, kwargs in calls] == ["a.example.com", "b.example.com"]
    # ``reason`` is what separates the cascade from an owner's explicit removal.
    assert {kwargs["reason"] for _, kwargs in calls} == {"service_deleted"}
    assert {kwargs["service_name"] for _, kwargs in calls} == {"a"}


def test_deleting_a_service_with_no_domains_records_nothing(monkeypatch, tmp_path) -> None:
    app, fake = _delete_app(monkeypatch, tmp_path, removed=[])

    with TestClient(app) as client:
        assert client.delete("/services/a", headers=_purge_auth()).status_code == 200

    fake.record.assert_not_awaited()


def test_the_delete_route_never_pre_reads_the_domain_table(monkeypatch, tmp_path) -> None:
    """The names come from the transaction, never from a read before it: a
    ``PUT .../domains/x`` committing in that window would be cascaded away with
    nobody told (the ``share.removed`` PR-review finding, same shape)."""
    app, _fake = _delete_app(monkeypatch, tmp_path, removed=["a.example.com"])
    queries = app.state.queries

    with TestClient(app) as client:
        assert client.delete("/services/a", headers=_purge_auth()).status_code == 200

    queries.get_service_domains.assert_not_awaited()
    queries.list_service_domains.assert_not_awaited()


# ---------------------------------------------------------------------------
# GET /proxy/status — the Caddy-factual domain rows (S-W11)
# ---------------------------------------------------------------------------


def _status_client(manager: object, queries: AsyncMock) -> TestClient:
    """The real ``/proxy/status`` route over a manager double."""
    app = FastAPI()
    register_error_handlers(app)
    api = APIRouter(prefix="/api")
    api.include_router(proxy_router)
    app.include_router(api)
    app.state.queries = queries
    app.state.proxy_manager = manager
    app.state.settings = SimpleNamespace(proxy=SimpleNamespace(mdns=False, mdns_address=None))
    app.state.mdns_advertiser = None
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return TestClient(app, raise_server_exceptions=False)


def _snapshot(**extra: Any) -> dict[str, Any]:
    """The pre-WP1 ``status_snapshot`` body, minus the route-composed ``mdns``."""
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
        **extra,
    }


def test_proxy_status_passes_the_domain_rows_to_the_manager() -> None:
    """(S-W11) The rows are passed IN so the manager annotates liveness off the
    live table it already read — a second admin read on an observability path is
    one more way for a slow Caddy to make it hang."""
    rows = [ServiceDomain(service_name="demo", domain=DOMAIN)]
    q = _queries(_svc(), domains=rows)
    seen: dict[str, Any] = {}

    async def _status_snapshot(*, domains=()):  # noqa: ANN001, ANN202
        seen["domains"] = list(domains)
        return _snapshot(
            domains=[
                {"domain": d.domain, "service_name": d.service_name, "acme": d.acme, "live": True}
                for d in domains
            ]
        )

    manager = SimpleNamespace(status_snapshot=_status_snapshot)
    body = (
        _status_client(manager, q)
        .get("/api/proxy/status", headers={"Authorization": f"Bearer {LEGACY}"})
        .json()
    )

    assert [d.domain for d in seen["domains"]] == [DOMAIN]
    assert body["domains"] == [
        {"domain": DOMAIN, "service_name": "demo", "acme": False, "live": True}
    ]


def test_proxy_status_always_carries_a_domains_key() -> None:
    """The field is never absent, so a client can read ``domains`` without
    branching on the daemon's build — an empty list is the honest answer for a
    node with no bound names."""
    q = _queries(_svc())

    async def _status_snapshot(*, domains=()):  # noqa: ANN001, ANN202
        return _snapshot()

    manager = SimpleNamespace(status_snapshot=_status_snapshot)
    body = (
        _status_client(manager, q)
        .get("/api/proxy/status", headers={"Authorization": f"Bearer {LEGACY}"})
        .json()
    )

    assert body["domains"] == []


def test_proxy_status_defaults_the_acme_block_when_the_manager_omits_it() -> None:
    """(P26 WP2) The key is never absent, so a client reads ``acme.enabled``
    without branching on the daemon's build — the rule ``domains`` already
    follows. A manager that predates WP2 (or a node with the block off and a
    snapshot that says nothing) projects the DISABLED shape, never ``{}``, and
    every config value is null rather than a guessed default."""
    q = _queries(_svc())

    async def _status_snapshot(*, domains=()):  # noqa: ANN001, ANN202
        return _snapshot()

    manager = SimpleNamespace(status_snapshot=_status_snapshot)
    body = (
        _status_client(manager, q)
        .get("/api/proxy/status", headers={"Authorization": f"Bearer {LEGACY}"})
        .json()
    )

    assert body["acme"] == {
        "enabled": False,
        "directory": None,
        "http_port": None,
        "http_redirect": None,
        "listening": None,
    }


def test_proxy_status_passes_an_enabled_acme_block_through_untouched() -> None:
    """The manager owns the fact; the schema only guarantees the key exists.
    ``listening`` is the Caddy-bind fact and rides through as sent — including
    ``None``, which means "not known", not "not listening"."""
    q = _queries(_svc())
    block = {
        "enabled": True,
        "directory": "https://acme-v02.api.letsencrypt.org/directory",
        "http_port": 80,
        "http_redirect": True,
        "listening": True,
    }

    async def _status_snapshot(*, domains=()):  # noqa: ANN001, ANN202
        return _snapshot(acme=block)

    manager = SimpleNamespace(status_snapshot=_status_snapshot)
    body = (
        _status_client(manager, q)
        .get("/api/proxy/status", headers={"Authorization": f"Bearer {LEGACY}"})
        .json()
    )

    assert body["acme"] == block
    # The ACME account contact reaches the CA and nothing else; this body is
    # readable by every authenticated principal.
    assert "email" not in json.dumps(body)


def test_proxy_status_domain_rows_carry_the_cert_state() -> None:
    """The same certificate vocabulary on the Caddy-factual surface: an
    operator debugging a name should not have to hold two words for one fact."""
    rows = [ServiceDomain(service_name="demo", domain=DOMAIN, acme=True)]
    q = _queries(_svc(), domains=rows)

    async def _status_snapshot(*, domains=()):  # noqa: ANN001, ANN202
        return _snapshot(
            domains=[
                {
                    "domain": d.domain,
                    "service_name": d.service_name,
                    "acme": d.acme,
                    "live": True,
                    "cert_state": "issued",
                }
                for d in domains
            ]
        )

    manager = SimpleNamespace(status_snapshot=_status_snapshot)
    body = (
        _status_client(manager, q)
        .get("/api/proxy/status", headers={"Authorization": f"Bearer {LEGACY}"})
        .json()
    )

    assert body["domains"][0]["cert_state"] == "issued"


# ``GET /capabilities`` advertises ``exposure.domains: true`` from WP1 and
# mirrors ``[proxy.acme].enabled`` as ``exposure.acme`` from WP2. That pin lives
# with the rest of the exposure block in ``tests/test_link_surface.py``, where
# the capabilities harness already stands up the whole link/proxy state the
# route reads — duplicating that harness here would be a second thing to keep
# in step.
