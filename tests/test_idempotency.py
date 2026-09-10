"""Test idempotency through real middleware and an in-memory database.

Use one event loop for httpx and aiosqlite to exercise BEGIN IMMEDIATE and UNIQUE
constraints. Cover replay, in-progress 409, method/path/body mismatch 422,
retryable failures, 24-hour expiry, key requirements and distinct mount scopes.
Secret-returning routes cache only status/resource ID; cached bodies are redacted.

Pin claim races and cancellation: release keys before writes, mark committed
writes interrupted, and shield successful finalization from disconnects and raw
Task.cancel(). Multipart, missing-length, oversized and secret-bearing requests
keep a NULL body digest and skip strict body comparison.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
from starlette.responses import Response, StreamingResponse

from nerdit.config.settings import NerditSettings
from nerdit.core.events import EventBus
from nerdit.daemon.audit import AuditMiddleware, derive_action
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import (
    _MAX_HASH_BODY_BYTES,
    NO_BODY_CACHE_ACTIONS,
    NO_BODY_HASH_ACTIONS,
    IdempotencyMiddleware,
    _extract_resource_id,
    _hashable_body,
    _honors_dry_run,
    _redacted_body_text,
)
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.db.database import Database
from nerdit.db.models import Job, JobKind
from nerdit.db.queries import Queries
from nerdit.db.queries._base import (
    REQUEST_WRITE_MARKER,
    RequestWriteMarker,
    mark_request_side_effect,
)

# ``token=None`` resolves to the LOCAL admin principal; its idempotency scope is
# ``anon:local`` (token_id is None → sentinel name).
LOCAL_SCOPE = "anon:local"

_SECRET = "nrd_PLAINTEXTSECRET_must_never_leak"


class _CreateBody(BaseModel):
    """Minimal stand-in for ``ServiceCreateRequest`` — enough shape to make a
    malformed body a 422 (the non-2xx-is-not-pinned case)."""

    name: str
    gpus: int = 0


def _create_router() -> APIRouter:
    """A generic authenticated write route standing in for ``POST /services``.

    The idempotency middleware is route-agnostic; these tests only need *a*
    mutating route that persists something. Using a local stub (rather than the
    real services router) keeps the middleware suite free of the service
    controller's app.state requirements.
    """
    r = APIRouter()

    @r.post("/services", status_code=201)
    async def _create(request: Request, body: _CreateBody) -> dict:
        job = Job(
            name=body.name,
            kind=JobKind.service,
            gpu_count=body.gpus,
            config='{"image": "demo:latest"}',
        )
        # ``service_name`` is UNIQUE; derive it from the generated id so two
        # keyless creates of the same body stay independent rows.
        job.service_name = f"{body.name}-{job.id}"
        await request.app.state.scheduler.submit_job(job)
        return {"id": job.id}

    return r


def _stub_router() -> APIRouter:
    """Extra routes: a secret-returning create, a redaction probe, and a real
    ``?dry_run``-honoring write (``POST /deploy``) so the dry-run bypass can be
    exercised on a route the registry actually recognizes."""
    r = APIRouter()
    calls = {"deploy": 0}

    @r.post("/tokens", status_code=201)
    async def _create_token() -> JSONResponse:
        # Mirrors S9's POST /tokens: returns a one-time plaintext secret.
        return JSONResponse(
            status_code=201,
            content={"id": "tok-x", "name": "n", "token": _SECRET},
        )

    @r.post("/echo", status_code=200)
    async def _echo() -> dict:
        # Not a NO_BODY_CACHE route → body is cached but must be redacted.
        return {"id": "e", "api_key": "sk-should-be-masked", "ok": 1}

    @r.post("/deploy", status_code=201)
    async def _deploy(dry_run: bool = False) -> dict:
        # Mirrors POST /deploy: a real ``?dry_run``-honoring write. Each call
        # yields a fresh id so a replayed (poisoned) response is detectable.
        calls["deploy"] += 1
        return {"id": f"dep-{calls['deploy']}"}

    @r.post("/databases", status_code=201)
    async def _echo_body(request: Request) -> dict:
        # Reports what the ROUTER received, after the middleware may already
        # have consumed the stream — the D-P22-7 read-twice probe. Mounted on a
        # path ``derive_action`` MAPS (``database.create``): only mapped actions
        # are digested, so a synthetic path would test the refusal instead.
        raw = await request.body()
        return {"id": hashlib.sha256(raw).hexdigest(), "len": len(raw)}

    @r.post("/secrets/{service}", status_code=200)
    async def _set_secret(service: str, request: Request) -> dict:
        # ``derive_action`` maps this path to ``secret.set``, the action whose
        # body must never be digested.
        await request.body()
        return {"id": service}

    return r


def _blocking_router() -> tuple[APIRouter, dict]:
    """(P20/P22) Writes that stay in flight until the test releases them.

    Stands in for the longest-running route in the product — ``POST
    /services/{ident}/run`` — so the client-disconnect / cancellation paths can
    be driven deterministically. ``/slow`` commits nothing before it blocks;
    ``/slow-write`` commits a real ``@_serialized`` write first, which is the
    whole difference between releasing the claim and pinning it ``interrupted``.
    """
    r = APIRouter()
    state: dict = {
        "calls": 0,
        "writes": 0,
        "entered": asyncio.Event(),
        "release": asyncio.Event(),
    }

    @r.post("/slow", status_code=201)
    async def _slow() -> dict:
        state["calls"] += 1
        state["entered"].set()
        await state["release"].wait()
        return {"id": f"slow-{state['calls']}"}

    @r.post("/slow-write", status_code=201)
    async def _slow_write(request: Request) -> dict:
        state["calls"] += 1
        state["writes"] += 1
        await request.app.state.queries.insert_audit_log(
            action="test.write", result="ok", principal_id=LOCAL_SCOPE
        )
        state["entered"].set()
        await state["release"].wait()
        return {"id": f"slow-write-{state['calls']}"}

    return r, state


def _seam_request(idem_key: str) -> Request:
    """A bare ``POST /services`` request for driving ``dispatch`` directly.

    Two cancellation shapes are not reachable through the ASGI client — a
    cancellation landing between response-complete and the finalize, and a
    ``CancelledError`` surfacing out of ``call_next`` (``BaseHTTPMiddleware``
    converts a route-raised one into ``RuntimeError: No response returned.``,
    i.e. the *Exception* branch). Both are driven at the middleware seam, where
    the branch under test is entered by construction.
    """
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/services",
            "raw_path": b"/services",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"idempotency-key", idem_key.encode())],
            "state": {},
        }
    )


# No auth middleware runs at the seam, so ``current_principal`` fails closed to
# ANONYMOUS and the claim is scoped to its sentinel name.
SEAM_SCOPE = "anon:anonymous"


async def _make_env(
    *, require_idempotency_key: bool = False, extra_router: APIRouter | None = None
) -> tuple[FastAPI, Database, Queries, AsyncMock]:
    """Build an app over a real in-memory DB with the full middleware stack."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    app = FastAPI()
    register_error_handlers(app)
    create_router = _create_router()
    app.include_router(create_router)
    app.include_router(create_router, prefix="/api")
    app.include_router(_stub_router())
    if extra_router is not None:
        app.include_router(extra_router)

    app.state.queries = queries
    app.state.event_bus = EventBus()
    app.state.settings = NerditSettings()

    scheduler = AsyncMock()

    async def _submit(job):  # noqa: ANN001, ANN202
        await queries.create_job(job)
        return job

    scheduler.submit_job = AsyncMock(side_effect=_submit)
    app.state.scheduler = scheduler
    runtime = AsyncMock()
    runtime.image_exists = AsyncMock(return_value=True)
    app.state.runtime = runtime

    # inner → outer: Idempotency (innermost), Audit, Auth, RequestId.
    app.add_middleware(
        IdempotencyMiddleware,
        get_queries=lambda: queries,
        require_idempotency_key=require_idempotency_key,
    )
    app.add_middleware(
        AuditMiddleware,
        get_queries=lambda: queries,
        get_event_bus=lambda: app.state.event_bus,
    )
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, db, queries, scheduler


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _create_body() -> dict:
    return {"name": "demo", "gpus": 1}


async def _await_record_state(queries: Queries, key: str, state: str, *, scope: str = LOCAL_SCOPE):
    """Poll until a DETACHED bookkeeping task has landed ``state`` on ``key``.

    The finalize/cleanup runs in its own task (so a raw ``Task.cancel()`` on the
    request cannot abort it), which means the caller's cancellation is observed
    *before* the bookkeeping completes — there is nothing to await but the row.
    """
    for _ in range(200):
        record = await queries.get_idempotency_record(scope, key)
        if record is not None and record.state == state:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"idempotency record {key!r} never reached state {state!r}")


# --- pure unit helpers --------------------------------------------------------


def test_extract_resource_id_from_json():
    assert _extract_resource_id(b'{"id": "abc", "x": 1}') == "abc"
    assert _extract_resource_id(b'{"x": 1}') is None
    assert _extract_resource_id(b"not json") is None


def test_redacted_body_masks_secret_keys():
    out = _redacted_body_text(b'{"id": "e", "api_key": "sk-1", "ok": 1}')
    assert "sk-1" not in out
    assert "***" in out
    assert '"ok": 1' in out or '"ok":1' in out


def test_token_create_is_a_no_body_cache_route():
    assert "token.create" in NO_BODY_CACHE_ACTIONS


# --- replay / dedupe ----------------------------------------------------------


@pytest.mark.asyncio
async def test_same_key_replays_without_duplicate_work():
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "K1"}
            first = await client.post("/services", json=_create_body(), headers=headers)
            assert first.status_code == 201
            assert first.headers.get("Idempotent-Replay") is None
            first_id = first.json()["id"]

            second = await client.post("/services", json=_create_body(), headers=headers)
            assert second.status_code == 201
            assert second.headers.get("Idempotent-Replay") == "true"
            # Body replayed verbatim (same job id), and the route ran only once.
            assert second.json()["id"] == first_id
        assert scheduler.submit_job.await_count == 1
        assert len(await queries.list_jobs(kind=JobKind.service)) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_keyless_requests_are_independent():
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            r1 = await client.post("/services", json=_create_body())
            r2 = await client.post("/services", json=_create_body())
            assert r1.status_code == 201 and r2.status_code == 201
            assert r1.json()["id"] != r2.json()["id"]
        assert scheduler.submit_job.await_count == 2
    finally:
        await db.close()


# --- in-progress / mismatch ---------------------------------------------------


@pytest.mark.asyncio
async def test_in_progress_key_returns_409():
    app, db, queries, scheduler = await _make_env()
    try:
        # Simulate a concurrent original still running.
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        inserted = await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="K2",
            method="POST",
            path="/services",
            expires_at=future,
        )
        assert inserted is True
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": "K2"}
            )
        assert resp.status_code == 409
        assert resp.json()["code"] == "idempotency_in_progress"
        # The blocked request never ran the route.
        assert scheduler.submit_job.await_count == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_method_path_mismatch_returns_422():
    app, db, queries, scheduler = await _make_env()
    try:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        # Same key, but originally used on a different path.
        await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="K3",
            method="POST",
            path="/services/other/stop",
            expires_at=future,
        )
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": "K3"}
            )
        assert resp.status_code == 422
        assert resp.json()["code"] == "idempotency_key_conflict"
    finally:
        await db.close()


# --- non-2xx is not pinned ----------------------------------------------------


@pytest.mark.asyncio
async def test_non_2xx_is_not_pinned_and_retryable():
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "K4"}
            bad = await client.post(
                "/services", json={"name": "demo", "gpus": "not-an-int"}, headers=headers
            )
            assert bad.status_code == 422  # validation error, not pinned
            # The failed outcome left no record behind.
            assert await queries.get_idempotency_record(LOCAL_SCOPE, "K4") is None

            good = await client.post("/services", json=_create_body(), headers=headers)
            assert good.status_code == 201
            assert good.headers.get("Idempotent-Replay") is None
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


# --- sweep --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_removes_expired_records():
    app, db, queries, scheduler = await _make_env()
    try:
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="old",
            method="POST",
            path="/services",
            expires_at=past,
        )
        await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="new",
            method="POST",
            path="/services",
            expires_at=future,
        )
        removed = await queries.sweep_expired_idempotency()
        assert removed == 1
        assert await queries.get_idempotency_record(LOCAL_SCOPE, "old") is None
        assert await queries.get_idempotency_record(LOCAL_SCOPE, "new") is not None
    finally:
        await db.close()


# --- key length bound ---------------------------------------------------------


@pytest.mark.asyncio
async def test_over_long_key_rejected_before_claim():
    """A key longer than the bound is 422'd BEFORE the claim — nothing persisted,
    the route never runs."""
    app, db, queries, scheduler = await _make_env()
    try:
        long_key = "k" * 6000
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": long_key}
            )
        assert resp.status_code == 422
        assert resp.json()["code"] == "idempotency_key_too_long"
        # Nothing was claimed/persisted and the route never executed.
        assert await queries.get_idempotency_record(LOCAL_SCOPE, long_key) is None
        assert scheduler.submit_job.await_count == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_max_length_key_still_claims():
    """A key at exactly the bound claims normally."""
    from nerdit.daemon.idempotency import _MAX_IDEMPOTENCY_KEY_LEN

    app, db, queries, scheduler = await _make_env()
    try:
        key = "k" * _MAX_IDEMPOTENCY_KEY_LEN
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": key}
            )
        assert resp.status_code == 201
        record = await queries.get_idempotency_record(LOCAL_SCOPE, key)
        assert record is not None
        assert record.state == "completed"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_empty_key_header_passes_through_as_absent():
    """An empty ``Idempotency-Key`` header is treated as absent (falsy check
    precedes the length bound)."""
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": ""}
            )
        assert resp.status_code == 201
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


# --- require flag -------------------------------------------------------------


@pytest.mark.asyncio
async def test_require_idempotency_key_rejects_keyless():
    app, db, queries, scheduler = await _make_env(require_idempotency_key=True)
    try:
        async with _client(app) as client:
            resp = await client.post("/services", json=_create_body())
        assert resp.status_code == 400
        assert resp.json()["code"] == "idempotency_key_required"
        assert scheduler.submit_job.await_count == 0
    finally:
        await db.close()


# --- secret-at-rest -----------------------------------------------------------


@pytest.mark.asyncio
async def test_secret_returning_route_is_not_cached():
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "Ksec"}
            first = await client.post("/tokens", headers=headers)
            assert first.status_code == 201
            assert _SECRET in first.text  # original response still shows the secret

            record = await queries.get_idempotency_record(LOCAL_SCOPE, "Ksec")
            assert record is not None
            assert record.state == "completed"
            assert record.response_body is None  # body NEVER stored
            assert record.resource_id == "tok-x"  # only the id is kept

            replay = await client.post("/tokens", headers=headers)
            assert replay.status_code == 201
            assert replay.headers.get("Idempotent-Replay") == "true"
            assert _SECRET not in replay.text  # non-secret envelope on replay
            assert replay.json()["code"] == "idempotent_replay"
            assert replay.json()["resource_id"] == "tok-x"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cached_body_is_redacted_defense_in_depth():
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "Kecho"}
            first = await client.post("/echo", headers=headers)
            assert first.status_code == 200
            record = await queries.get_idempotency_record(LOCAL_SCOPE, "Kecho")
            assert record is not None
            assert record.response_body is not None
            assert "sk-should-be-masked" not in record.response_body
            assert "***" in record.response_body
    finally:
        await db.close()


# --- mount-scope distinctness -------------------------------------------------


@pytest.mark.asyncio
async def test_no_cross_mount_dedupe():
    """Same key on ``/services`` then ``/api/services`` must NOT replay across mounts.

    The key is scoped by the concrete ``url.path`` (it is part of the key per the
    plan), while the committed table PK is ``(principal_id, idem_key)`` with
    ``path`` stored as a guard. Reusing one key on a different path is therefore
    a conflict (422) — the ``/api`` request never receives the ``/services``
    response. This is the intended "no cross-mount dedupe in P1" behavior.
    """
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "Kmount"}
            bare = await client.post("/services", json=_create_body(), headers=headers)
            api = await client.post("/api/services", json=_create_body(), headers=headers)
            assert bare.status_code == 201
            assert bare.headers.get("Idempotent-Replay") is None
            # The second mount is a conflict, NOT a silent replay of the first.
            assert api.status_code == 422
            assert api.json()["code"] == "idempotency_key_conflict"
        rec = await queries.get_idempotency_record(LOCAL_SCOPE, "Kmount")
        assert rec is not None
        assert rec.path == "/services"
        # Only the bare-root request reached the router.
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


# --- dry_run bypass: the key is ignored entirely (P13 WP9) --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run_value", ["true", "1", "yes", "on", "t", "y"])
async def test_dry_run_key_is_not_claimed_and_does_not_poison_real_request(dry_run_value):
    """A ``?dry_run=<truthy>`` write on a route that honors it must neither claim
    nor record its Idempotency-Key.

    Otherwise a key sent on the dry-run would be pinned for the bare path and the
    subsequent *real* POST with the same key would replay the cached dry-run
    response — a silent no-op deploy (the exact trap this fix closes; the deploy +
    config-apply routes both ride this one middleware change). Driven against a
    *real* ``dry_run``-honoring route (``POST /deploy``, in ``_DRY_RUN_ROUTES``),
    NOT a stand-in like ``/services`` which does not implement ``dry_run`` and so must
    keep claiming its key (see the sibling test below). Parametrized over pydantic
    v2's full str→bool truthy set (notably ``t``/``y``) so the middleware bypass
    grammar and the route's coercion can never drift apart.
    """
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KP"}
            first = await client.post(f"/deploy?dry_run={dry_run_value}", headers=headers)
            assert first.status_code == 201
            assert first.headers.get("Idempotent-Replay") is None
            # The middleware ignored the key entirely — nothing pinned.
            assert await queries.get_idempotency_record(LOCAL_SCOPE, "KP") is None

            # The real POST with the SAME key EXECUTES (fresh claim), never replays.
            second = await client.post("/deploy", headers=headers)
            assert second.status_code == 201
            assert second.headers.get("Idempotent-Replay") is None
            assert second.json()["id"] != first.json()["id"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_dry_run_on_non_dry_run_route_still_claims_key():
    """``?dry_run=true`` on a route that does NOT declare a ``dry_run`` arg (e.g.
    ``POST /services``) must NOT trigger the bypass.

    FastAPI silently ignores the undeclared query param, so the request performs
    the REAL write; skipping idempotency there would let a retry duplicate the
    mutation. So the key is claimed/recorded, and a replay with the same key
    returns the cached response instead of executing again (the regression this
    scoping fix guards — the pre-fix middleware bypassed on the query string alone).
    """
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KND"}
            first = await client.post(
                "/services?dry_run=true", json=_create_body(), headers=headers
            )
            assert first.status_code == 201
            assert first.headers.get("Idempotent-Replay") is None
            # The key WAS claimed and pinned (the route really executed).
            record = await queries.get_idempotency_record(LOCAL_SCOPE, "KND")
            assert record is not None
            assert record.state == "completed"
            assert record.method == "POST" and record.path == "/services"

            # A replay with the same key returns the cached response — no re-run.
            replay = await client.post(
                "/services?dry_run=true", json=_create_body(), headers=headers
            )
            assert replay.status_code == 201
            assert replay.headers.get("Idempotent-Replay") == "true"
            assert replay.json()["id"] == first.json()["id"]
        # The route ran exactly once despite two requests.
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_config_apply_dry_run_key_does_not_poison_real_apply(tmp_path):
    """The same bypass fixes config-apply's identical latent trap: a dry-run apply
    carrying a key must not poison the real apply into a cached no-op replay."""
    from nerdit.config.store import ConfigStore
    from nerdit.daemon.routes.config import router as config_router

    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[monitor]\ninterval_seconds = 5\n")

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(config_router, prefix="/api")
    app.state.queries = queries
    app.state.config_store = ConfigStore(cfg_path)
    app.state.event_bus = EventBus()
    app.state.settings = NerditSettings()
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(
        AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: app.state.event_bus
    )
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    try:
        async with _client(app) as client:
            etag = (await client.get("/api/config/daemon")).headers["ETag"]
            dry = await client.post(
                "/api/config/daemon/apply?dry_run=true",
                json={"sections": {"monitor": {"interval_seconds": 9}}},
                headers={"Idempotency-Key": "CK"},
            )
            assert dry.status_code == 200, dry.text
            assert dry.json()["applied"] is False
            assert await queries.get_idempotency_record(LOCAL_SCOPE, "CK") is None

            real = await client.post(
                "/api/config/daemon/apply",
                json={"sections": {"monitor": {"interval_seconds": 9}}},
                headers={"Idempotency-Key": "CK", "If-Match": etag},
            )
            assert real.status_code == 200, real.text
            assert real.headers.get("Idempotent-Replay") is None
            assert real.json()["applied"] is True
            assert ConfigStore(cfg_path).effective_section("monitor")["interval_seconds"] == 9
    finally:
        await db.close()


def test_honors_dry_run_registry_matches_both_mounts():
    """The bypass registry must recognize every ``dry_run``-declaring route in
    BOTH path forms (bare legacy root and the ``/api`` mount — the daemon mounts
    each router twice), and nothing else. Keep the positive list in lockstep with
    the handlers that declare ``dry_run: bool = Query(...)``."""
    positives = [
        ("POST", "/deploy"),
        ("POST", "/deploy/git"),
        ("POST", "/app-templates/synth/deploy"),
        ("POST", "/config/daemon/apply"),
        ("PUT", "/config/daemon/proxy"),
        ("PUT", "/config/apps/demo/ai"),
        ("POST", "/system/gc"),  # P14b WP-A2 — dry-run gc must skip the key claim
    ]
    for method, path in positives:
        assert _honors_dry_run(method, path), (method, path)
        assert _honors_dry_run(method, f"/api{path}"), (method, f"/api{path}")

    negatives = [
        ("POST", "/models"),  # no dry_run handler arg — FastAPI ignores the param
        ("POST", "/services"),
        ("POST", "/app-templates/synth/deploy/extra"),
        ("GET", "/deploy"),  # wrong method never bypasses
        ("POST", "/deploy/git/extra"),  # anchored patterns: no prefix match
        ("PUT", "/config/apps/demo"),  # whole-app PUT does not exist / no dry_run
    ]
    for method, path in negatives:
        assert not _honors_dry_run(method, path), (method, path)
        assert not _honors_dry_run(method, f"/api{path}"), (method, f"/api{path}")


# --- (P22 / D-P22-1) the claim race, driven through the real stack ------------


@pytest.mark.asyncio
async def test_concurrent_same_key_claims_serialize():
    """Two LIVE same-key requests: one claims, the other gets a clean 409.

    The pre-existing 409 test pre-seeds the row, so it proves the read path but
    never exercises the claim itself. Here request B's ``INSERT OR IGNORE``
    really races A's while A holds the key, which is the shape that used to fail
    with ``cannot start a transaction within a transaction`` before the P14b
    write-lock fix — and the shape nothing covered end to end until now.
    """
    router, state = _blocking_router()
    app, db, queries, _ = await _make_env(extra_router=router)
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KRACE"}
            first = asyncio.create_task(client.post("/slow", headers=headers))
            await asyncio.wait_for(state["entered"].wait(), timeout=5)

            blocked = await client.post("/slow", headers=headers)
            assert blocked.status_code == 409
            assert blocked.json()["code"] == "idempotency_in_progress"

            state["release"].set()
            done = await first
            assert done.status_code == 201
            assert done.headers.get("Idempotent-Replay") is None

            replay = await client.post("/slow", headers=headers)
            assert replay.status_code == 201
            assert replay.headers.get("Idempotent-Replay") == "true"
            assert replay.json()["id"] == done.json()["id"]
        # The blocked request and the replay never re-entered the route.
        assert state["calls"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_simultaneous_same_key_posts_never_5xx():
    """Two same-key requests fired together: exactly one executes, and the other
    is an *answered* outcome (409 or replay) — never a 500 from a poisoned
    transaction."""
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KGATHER"}
            r1, r2 = await asyncio.gather(
                client.post("/services", json=_create_body(), headers=headers),
                client.post("/services", json=_create_body(), headers=headers),
            )
        statuses = sorted([r1.status_code, r2.status_code])
        assert all(s < 500 for s in statuses), statuses
        executed = [
            r for r in (r1, r2) if r.status_code == 201 and not r.headers.get("Idempotent-Replay")
        ]
        assert len(executed) == 1, [(r.status_code, dict(r.headers)) for r in (r1, r2)]
        other = r2 if executed[0] is r1 else r1
        assert other.status_code == 409 or other.headers.get("Idempotent-Replay") == "true"
        # Whatever the interleaving, the route ran exactly once.
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


# --- (P22 / D-P22-4) cancel-safe cleanup --------------------------------------


@pytest.mark.asyncio
async def test_cancelled_request_unpins_its_idempotency_key():
    """Release an uncommitted request's key after a disconnect so retries can run.

    CancelledError inherits BaseException, so Exception cleanup misses it. Test
    through the real ASGI stack: BaseHTTPMiddleware cancels the child task's anyio
    scope, which cancels every subsequent await unless cleanup is shielded. The
    route blocks before committing; deleting the key is therefore safe.
    """
    router, state = _blocking_router()
    app, db, queries, _ = await _make_env(extra_router=router)
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KCANCEL"}
            inflight = asyncio.create_task(client.post("/slow", headers=headers))
            await asyncio.wait_for(state["entered"].wait(), timeout=5)

            # Precondition: the key really is claimed while the request runs.
            record = await queries.get_idempotency_record(LOCAL_SCOPE, "KCANCEL")
            assert record is not None
            assert record.state == "in_progress"

            inflight.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inflight

            # The claim is released — nothing pinned for a request that never
            # completed and never wrote.
            leftover = await queries.get_idempotency_record(LOCAL_SCOPE, "KCANCEL")
            assert leftover is None, (
                "cancelled request left its idempotency claim behind "
                f"(state={getattr(leftover, 'state', None)!r}) — the key is "
                "unusable for 24 h; the cleanup await needs a shielded cancel scope"
            )

            # ...and the same key is immediately reusable: the retry EXECUTES
            # (no 409, no replay of a response that was never produced).
            state["release"].set()
            retry = await client.post("/slow", headers=headers)
            assert retry.status_code == 201
            assert retry.headers.get("Idempotent-Replay") is None
            assert retry.json()["id"] == "slow-2"

        # Two real executions: the cancelled one and the retry.
        assert state["calls"] == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_cancel_after_committed_write_pins_interrupted():
    """A cancellation AFTER a write committed must keep the claim, honestly.

    This is what the P20 revert (``77f566f``) demanded: an unconditional delete
    lets a retry re-execute a write that already landed (a cancelled
    ``POST /tokens`` mints a second admin token). The ``RequestWriteMarker``
    supplies the distinction, and this test is also the empirical proof that the
    ContextVar crosses the ``BaseHTTPMiddleware`` child-task boundary — the route
    runs two tasks below the middleware that set it.
    """
    router, state = _blocking_router()
    app, db, queries, _ = await _make_env(extra_router=router)
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KWRITE"}
            inflight = asyncio.create_task(client.post("/slow-write", headers=headers))
            await asyncio.wait_for(state["entered"].wait(), timeout=5)

            inflight.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inflight

            record = await queries.get_idempotency_record(LOCAL_SCOPE, "KWRITE")
            assert record is not None, "a committed write must not have its claim deleted"
            assert record.state == "interrupted"

            # A retry is told the truth: never a silent re-execution, never a
            # bare 24 h in-progress 409.
            retry = await client.post("/slow-write", headers=headers)
            assert retry.status_code == 409
            assert retry.json()["code"] == "idempotency_interrupted"

            # A FRESH key executes normally.
            state["release"].set()
            fresh = await client.post("/slow-write", headers={"Idempotency-Key": "KWRITE2"})
            assert fresh.status_code == 201
            assert fresh.headers.get("Idempotent-Replay") is None
        assert state["writes"] == 2  # the cancelled one and the fresh-key one
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_cancellederror_from_call_next_takes_the_baseexception_branch(committed):
    """The uvicorn shutdown cancel and a client disconnect are ONE code path.

    The daemon's graceful shutdown cancels in-flight requests; whichever task the
    cancellation is delivered to, it reaches this middleware as a
    ``CancelledError`` out of ``call_next`` — the same branch the disconnect of
    the test above takes. Asserted by construction at the seam, both with and
    without a committed write, because the ASGI client can only produce the
    disconnect shape.
    """
    app, db, queries, _ = await _make_env()
    try:
        middleware = IdempotencyMiddleware(app=None, get_queries=lambda: queries)
        key = f"KCANCEL-{committed}"

        async def _call_next(_req: Request) -> Response:
            if committed:
                # Stands in for any ``@_serialized`` writer having committed.
                marker = REQUEST_WRITE_MARKER.get()
                assert marker is not None, "the middleware must install a marker"
                marker.committed = True
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await middleware.dispatch(_seam_request(key), _call_next)

        record = await queries.get_idempotency_record(SEAM_SCOPE, key)
        if committed:
            assert record is not None and record.state == "interrupted"
        else:
            assert record is None
    finally:
        await db.close()


# --- (P22 / D-P22-6) the 2xx finalize is shielded -----------------------------


@pytest.mark.asyncio
async def test_finalize_is_shielded_from_a_late_cancellation():
    """A cancellation arriving after the route succeeded must still pin the key.

    The disconnect-between-response-and-finalize window is milliseconds and not
    deterministically reachable through the ASGI client, so it is driven at the
    middleware seam instead: ``call_next`` cancels the surrounding scope and
    *then* returns a 2xx. Without ``anyio.CancelScope(shield=True)`` around the
    finalize, the ``complete_idempotency_record`` await is re-cancelled and the
    claim is left ``in_progress`` — a 24 h 409 for a request that succeeded.
    """
    app, db, queries, _ = await _make_env()
    try:
        middleware = IdempotencyMiddleware(app=None, get_queries=lambda: queries)

        with anyio.CancelScope() as scope:

            async def _call_next(_req: Request) -> StreamingResponse:
                async def _body():
                    yield b'{"id": "late-1"}'

                scope.cancel()
                return StreamingResponse(_body(), status_code=201)

            await middleware.dispatch(_seam_request("KLATE"), _call_next)

        record = await queries.get_idempotency_record(SEAM_SCOPE, "KLATE")
        assert record is not None
        assert record.state == "completed"
        assert record.response_status == 201
        assert record.resource_id == "late-1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_finalize_survives_a_raw_task_cancel():
    """A RAW ``Task.cancel()`` during the finalize must still pin the claim.

    ``anyio.CancelScope(shield=True)`` is not a general answer: it only defers a
    cancellation for a task anyio *tracks* (today: the one starlette's
    ``BaseHTTPMiddleware`` task group registers). A raw ``asyncio.Task.cancel()``
    on any other task — which is the shape uvicorn's graceful-shutdown timeout
    uses — goes straight through, aborting ``complete_idempotency_record`` and
    stranding a SUCCEEDED request's claim ``in_progress`` for the full 24 h TTL.
    The bookkeeping therefore also runs as its own task (``_detached``), which is
    not the cancellation target.

    Driven at the middleware seam precisely because there is no task group there:
    the raw cancel really is delivered, so this fails without ``_detached``.
    """
    app, db, queries, _ = await _make_env()
    parked = asyncio.Event()
    release = asyncio.Event()
    real_complete = queries.complete_idempotency_record

    async def _blocking_complete(**kwargs):  # noqa: ANN003, ANN202
        parked.set()
        await release.wait()
        return await real_complete(**kwargs)

    queries.complete_idempotency_record = _blocking_complete
    try:
        middleware = IdempotencyMiddleware(app=None, get_queries=lambda: queries)

        async def _call_next(_req: Request) -> StreamingResponse:
            async def _body():
                yield b'{"id": "raw-1"}'

            return StreamingResponse(_body(), status_code=201)

        inflight = asyncio.create_task(middleware.dispatch(_seam_request("KRAW"), _call_next))
        await asyncio.wait_for(parked.wait(), timeout=5)

        inflight.cancel()
        with pytest.raises(asyncio.CancelledError):
            await inflight

        # The detached finalize is still alive and completes the record.
        release.set()
        record = await _await_record_state(queries, "KRAW", "completed", scope=SEAM_SCOPE)
        assert record.response_status == 201
        assert record.resource_id == "raw-1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_interrupt_cleanup_survives_a_raw_task_cancel():
    """Same hazard on the BaseException branch: the cleanup must not be abortable.

    ``call_next`` is cancelled with a write already committed, so the middleware
    starts pinning the claim ``interrupted``; a raw cancel landing on THAT await
    used to abandon it, leaving the claim ``in_progress`` although a write had
    landed — the exact state D-P22-4 exists to prevent.
    """
    app, db, queries, _ = await _make_env()
    parked = asyncio.Event()
    release = asyncio.Event()
    real_interrupt = queries.mark_idempotency_interrupted

    async def _blocking_interrupt(principal_id, idem_key):  # noqa: ANN001, ANN202
        parked.set()
        await release.wait()
        return await real_interrupt(principal_id, idem_key)

    queries.mark_idempotency_interrupted = _blocking_interrupt
    try:
        middleware = IdempotencyMiddleware(app=None, get_queries=lambda: queries)

        async def _call_next(_req: Request) -> Response:
            marker = REQUEST_WRITE_MARKER.get()
            assert marker is not None, "the middleware must install a marker"
            marker.committed = True
            raise asyncio.CancelledError

        inflight = asyncio.create_task(middleware.dispatch(_seam_request("KRAW2"), _call_next))
        await asyncio.wait_for(parked.wait(), timeout=5)

        inflight.cancel()  # raw cancel DURING the cleanup
        with pytest.raises(asyncio.CancelledError):
            await inflight

        release.set()
        await _await_record_state(queries, "KRAW2", "interrupted", scope=SEAM_SCOPE)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_mark_interrupted_only_touches_in_progress_rows():
    """The state guard makes a late cancellation racing the finalize a no-op: a
    ``completed`` record must stay replayable, never be demoted."""
    app, db, queries, _ = await _make_env()
    try:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        for key in ("KA", "KB"):
            await queries.insert_idempotency_inprogress(
                principal_id=LOCAL_SCOPE,
                idem_key=key,
                method="POST",
                path="/services",
                expires_at=future,
            )
        await queries.complete_idempotency_record(
            principal_id=LOCAL_SCOPE,
            idem_key="KB",
            response_status=201,
            response_body='{"id": "x"}',
            content_type="application/json",
            resource_id="x",
        )

        await queries.mark_idempotency_interrupted(LOCAL_SCOPE, "KA")
        await queries.mark_idempotency_interrupted(LOCAL_SCOPE, "KB")

        assert (await queries.get_idempotency_record(LOCAL_SCOPE, "KA")).state == "interrupted"
        completed = await queries.get_idempotency_record(LOCAL_SCOPE, "KB")
        assert completed.state == "completed"
        assert completed.response_body == '{"id": "x"}'

        # A key that no longer exists is a silent no-op, not an error.
        await queries.mark_idempotency_interrupted(LOCAL_SCOPE, "KGONE")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_interrupted_rows_are_swept_like_any_other():
    """The 24 h sweep reaps ``interrupted`` rows — the honest 409 is bounded."""
    app, db, queries, _ = await _make_env()
    try:
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="KOLD",
            method="POST",
            path="/services",
            expires_at=past,
        )
        await queries.mark_idempotency_interrupted(LOCAL_SCOPE, "KOLD")
        assert await queries.sweep_expired_idempotency() == 1
        assert await queries.get_idempotency_record(LOCAL_SCOPE, "KOLD") is None
    finally:
        await db.close()


def test_write_marker_flips_only_inside_a_request_context():
    """Background work (reconcile, sweeps) runs with no marker installed, so the
    flip is a no-op there; a marker only exists for a claimed request."""
    assert REQUEST_WRITE_MARKER.get() is None
    marker = RequestWriteMarker()
    assert marker.committed is False
    token = REQUEST_WRITE_MARKER.set(marker)
    try:
        assert REQUEST_WRITE_MARKER.get() is marker
    finally:
        REQUEST_WRITE_MARKER.reset(token)
    assert REQUEST_WRITE_MARKER.get() is None


def test_mark_request_side_effect_flips_only_with_a_marker_installed():
    """The escape hatch for routes whose durable effect is NOT a DB write.

    Outside a claimed request there is no marker, and the call must be a silent
    no-op (background work must never raise here); inside one it pins the claim
    exactly as a ``@_serialized`` commit would.
    """
    assert REQUEST_WRITE_MARKER.get() is None
    mark_request_side_effect()  # no marker installed → no-op, no raise

    marker = RequestWriteMarker()
    token = REQUEST_WRITE_MARKER.set(marker)
    try:
        assert marker.committed is False
        mark_request_side_effect()
        assert marker.committed is True
    finally:
        REQUEST_WRITE_MARKER.reset(token)


# --- (P22 / D-P22-2) the request-body digest ----------------------------------


def _body_request(
    body: bytes,
    *,
    content_type: str | None = "application/json",
    content_length: str | None = None,
    path: str = "/services",
) -> Request:
    """A request with a real receive channel, for driving ``_hashable_body``.

    ``content_length`` defaults to the true length; pass an explicit value (or
    ``None``) to drive the chunked / mis-declared shapes the ASGI client cannot
    produce on demand.
    """
    headers: list[tuple[bytes, bytes]] = []
    if content_type is not None:
        headers.append((b"content-type", content_type.encode()))
    declared = str(len(body)) if content_length is None else content_length
    if declared != "":
        headers.append((b"content-length", declared.encode()))

    async def _receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": headers,
            "state": {},
        },
        _receive,
    )


@pytest.mark.asyncio
async def test_hashable_body_digests_a_small_json_body():
    body = b'{"name": "demo"}'
    # ``database.create`` is the reference DIGESTED action: its credentials are
    # daemon-minted, so the body carries no caller-supplied secret material.
    digest = await _hashable_body(_body_request(body), "database.create")
    assert digest == hashlib.sha256(body).hexdigest()


@pytest.mark.asyncio
async def test_hashable_body_refuses_secret_carrying_actions():
    """D-P22-2: a SHA-256 of a low-entropy secret is an offline-dictionary
    fingerprint of it, persisted 24 h and captured by every backup tar. Every
    action whose body carries a caller-supplied env/secrets map therefore stores
    no digest at all — the same posture as ``NO_BODY_CACHE_ACTIONS`` on the
    response side."""
    body = b'{"API_KEY": "hunter2"}'
    for action in sorted(NO_BODY_HASH_ACTIONS):
        assert await _hashable_body(_body_request(body), action) is None


@pytest.mark.asyncio
async def test_hashable_body_refuses_unmapped_action_spellings():
    """A non-canonical spelling must not route AROUND the secret exclusion.

    ``derive_action`` matches anchored patterns, so ``POST /secrets/demo/``
    (which Starlette answers with a 307) derives the ``"METHOD /path"`` fallback
    rather than ``secret.set`` — and the denylist, keyed on the mapped name,
    would not have caught it. The claim row is written BEFORE the router
    redirects, so the digest of a secret body would briefly be at rest, and a
    concurrent ``POST /system/backup`` snapshots it into a key-bearing tar.
    Only mapped actions are hashed; fallbacks are the ones carrying a space.
    """
    body = b'{"values": {"API_KEY": "hunter2"}}'
    for path in ("/secrets/demo/", "/app-templates/tpl/deploy/"):
        action = derive_action("POST", path)[0]
        assert action not in NO_BODY_HASH_ACTIONS, "precondition: an unmapped spelling"
        assert await _hashable_body(_body_request(body, path=path), action) is None


def test_every_real_mutating_route_derives_a_mapped_action():
    """The other half of the mapped-action-only rule: it must cost no coverage.

    Refusing to digest fallback actions is what makes the secret exclusion
    spelling-proof, but it silently skips the body check for any route
    ``derive_action`` does not map. Today that set is empty; a new mutating
    route without a ``_ROUTE_RULES`` entry must fail HERE rather than quietly
    losing its same-key-different-body 422 (and its audit action name with it).
    """
    from nerdit.daemon.server import create_app

    app = create_app()
    unmapped = {
        (method, route.path)
        for route in app.routes
        for method in (getattr(route, "methods", None) or ())
        if method in {"POST", "PUT", "PATCH", "DELETE"}
        and " " in derive_action(method, route.path)[0]
    }
    assert not unmapped, f"mutating routes with no audit rule: {sorted(unmapped)}"


def test_no_body_hash_actions_match_the_real_routes():
    """The exclusion set is keyed on ``derive_action`` output, so a route-rule
    rename must not silently start fingerprinting secrets.

    Membership rule: any action whose request body carries a caller-supplied
    env/secrets map — the two secret maps plus the two env-carrying service
    writes (``ServiceCreateRequest.env`` and the run route's documented env
    overlay). P27 WP-C2 extends it to the one body carrying a BARE secret
    rather than a map of them: ``link.created``'s plaintext link code, a
    low-entropy human-typed value whose SHA-256 would be an offline-dictionary
    fingerprint of it. P17d D-LIC5 extends the same bare-secret branch to
    ``license.install``: the body is the compact JWS blob, a paid artifact
    handled as a secret in transit that must never live outside the license file
    in ANY derived form — high entropy, so the digest is a confirmation oracle
    rather than a dictionary, but the posture does not change with entropy.
    """
    assert derive_action("POST", "/secrets/demo")[0] == "secret.set"
    assert derive_action("POST", "/api/secrets/demo")[0] == "secret.set"
    assert derive_action("POST", "/app-templates/tpl/deploy")[0] == "template.deploy"
    assert derive_action("POST", "/services")[0] == "service.create"
    assert derive_action("POST", "/services/demo/run")[0] == "service.run"
    assert derive_action("POST", "/link/claim")[0] == "link.created"
    assert derive_action("POST", "/api/license")[0] == "license.install"
    assert sorted(NO_BODY_HASH_ACTIONS) == [
        "license.install",
        "link.created",
        "secret.set",
        "service.create",
        "service.run",
        "template.deploy",
    ]
    # rotate-key carries no secret VALUE (its literal rule sits above the
    # generic one), so it is deliberately NOT exempt.
    assert derive_action("POST", "/secrets/rotate-key")[0] not in NO_BODY_HASH_ACTIONS
    # Deliberate non-members: managed-database credentials are daemon-minted
    # (the request carries none), and a config value like ``[posthog].api_key``
    # already sits in plaintext TOML on the same disk — a digest adds nothing.
    assert derive_action("POST", "/databases")[0] not in NO_BODY_HASH_ACTIONS
    assert derive_action("POST", "/config/daemon/apply")[0] not in NO_BODY_HASH_ACTIONS


@pytest.mark.asyncio
async def test_hashable_body_refuses_multipart():
    """The ``/deploy`` ZIP streams up to ``[daemon].max_upload_bytes`` through the
    early-reject spool; buffering it here to digest it would defeat that."""
    request = _body_request(b"--x--", content_type="multipart/form-data; boundary=x")
    assert await _hashable_body(request, "deploy.create") is None


@pytest.mark.asyncio
async def test_hashable_body_refuses_absent_or_unparseable_content_length():
    """Absent CL (chunked) ⇒ skip: the declared length is what bounds the read."""
    assert await _hashable_body(_body_request(b"{}", content_length=""), "database.create") is None
    bad = _body_request(b"{}", content_length="not-a-number")
    assert await _hashable_body(bad, "database.create") is None


@pytest.mark.asyncio
async def test_hashable_body_bound_is_inclusive():
    at_bound = _body_request(b"x", content_length=str(_MAX_HASH_BODY_BYTES))
    assert await _hashable_body(at_bound, "database.create") is not None
    over = _body_request(b"x", content_length=str(_MAX_HASH_BODY_BYTES + 1))
    assert await _hashable_body(over, "database.create") is None


@pytest.mark.asyncio
async def test_same_key_different_body_is_a_422_conflict():
    """One key means one operation: a different body is a different request.

    Before P22 the second call replayed the FIRST body's response, so an agent
    that reused a key across two distinct creates was told its second create had
    succeeded when nothing had run.

    Driven on ``POST /databases`` (``database.create``): it is the write whose
    body carries no caller-supplied secret material, so it is digested. The
    env-carrying creates are exempt — see the test below.
    """
    app, db, queries, _ = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KBODY"}
            first = await client.post("/databases", content=b'{"name": "one"}', headers=headers)
            assert first.status_code == 201

            conflict = await client.post("/databases", content=b'{"name": "two"}', headers=headers)
            assert conflict.status_code == 422
            assert conflict.json()["code"] == "idempotency_key_conflict"
            assert "body" in conflict.json()["message"]
            assert "resend the original body" in conflict.json()["hint"]

            # The identical body still replays — the digest gates the conflict,
            # it does not break the contract it protects.
            replay = await client.post("/databases", content=b'{"name": "one"}', headers=headers)
            assert replay.headers.get("Idempotent-Replay") == "true"
            assert replay.json()["id"] == first.json()["id"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_env_carrying_create_body_is_never_digested():
    """``service.create`` carries ``ServiceCreateRequest.env`` — caller-supplied
    values of the same credential class as a secrets map — so its body is not
    digested, and the honest consequence is pinned here: a reused key with a
    DIFFERENT body replays instead of 422-ing (the ``secret.set`` trade).
    """
    app, db, queries, scheduler = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KENV"}
            first = await client.post("/services", json={"name": "one", "gpus": 1}, headers=headers)
            assert first.status_code == 201
            record = await queries.get_idempotency_record(LOCAL_SCOPE, "KENV")
            assert record is not None
            assert record.body_hash is None

            second = await client.post(
                "/services", json={"name": "two", "gpus": 1}, headers=headers
            )
            assert second.headers.get("Idempotent-Replay") == "true"
            assert second.json()["id"] == first.json()["id"]
        assert scheduler.submit_job.await_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_claim_stores_the_digest_not_the_body():
    """Only the hex digest is persisted — never the request body itself."""
    app, db, queries, _ = await _make_env()
    try:
        async with _client(app) as client:
            resp = await client.post(
                "/databases", content=b'{"secretish": "value"}', headers={"Idempotency-Key": "KD"}
            )
            assert resp.status_code == 201
        record = await queries.get_idempotency_record(LOCAL_SCOPE, "KD")
        assert record is not None
        assert record.body_hash == hashlib.sha256(b'{"secretish": "value"}').hexdigest()
        assert "secretish" not in (record.response_body or "")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_router_still_reads_the_body_the_middleware_consumed():
    """D-P22-7: the middleware reads the body pre-``call_next`` and relies on
    Starlette's ``_CachedRequest`` replaying it downstream.

    The declared ``fastapi>=0.110`` floor did NOT guarantee that; an environment
    resolving an unsafe Starlette would hand the router an EMPTY body — every
    write silently 422-ing on a missing field, or worse, parsing as ``{}``. This
    pins it: the router's own digest of what it read must equal the middleware's.
    """
    app, db, queries, _ = await _make_env()
    body = b'{"payload": "' + b"a" * 4096 + b'"}'
    try:
        async with _client(app) as client:
            resp = await client.post(
                "/databases", content=body, headers={"Idempotency-Key": "KTWICE"}
            )
            assert resp.status_code == 201
            assert resp.json()["len"] == len(body)
            assert resp.json()["id"] == hashlib.sha256(body).hexdigest()
        record = await queries.get_idempotency_record(LOCAL_SCOPE, "KTWICE")
        assert record is not None
        assert record.body_hash == resp.json()["id"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_multipart_upload_is_not_digested_but_still_reaches_the_route():
    app, db, queries, _ = await _make_env()
    try:
        async with _client(app) as client:
            resp = await client.post(
                "/databases",
                files={"file": ("app.zip", b"PK\x03\x04payload", "application/zip")},
                headers={"Idempotency-Key": "KMULTI"},
            )
            assert resp.status_code == 201
            assert resp.json()["len"] > 0  # the route got the real upload
        record = await queries.get_idempotency_record(LOCAL_SCOPE, "KMULTI")
        assert record is not None
        assert record.body_hash is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_oversized_body_is_not_digested():
    app, db, queries, _ = await _make_env()
    try:
        async with _client(app) as client:
            resp = await client.post(
                "/databases",
                content=b"x" * (_MAX_HASH_BODY_BYTES + 1),
                headers={"Idempotency-Key": "KBIG"},
            )
            assert resp.status_code == 201
        record = await queries.get_idempotency_record(LOCAL_SCOPE, "KBIG")
        assert record is not None
        assert record.body_hash is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_secret_route_body_is_never_digested_even_across_bodies():
    """The honest consequence of the exclusion, pinned: on ``secret.set`` a
    reused key still dedupes on method+path only, so a DIFFERENT body replays
    instead of 422-ing. Storing a fingerprint of the value is the worse trade."""
    app, db, queries, _ = await _make_env()
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "KSECRET"}
            first = await client.post("/secrets/demo", json={"API_KEY": "one"}, headers=headers)
            assert first.status_code == 200
            record = await queries.get_idempotency_record(LOCAL_SCOPE, "KSECRET")
            assert record is not None
            assert record.body_hash is None

            second = await client.post("/secrets/demo", json={"API_KEY": "two"}, headers=headers)
            assert second.headers.get("Idempotent-Replay") == "true"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_legacy_null_digest_still_replays():
    """A row claimed before P22 carries no digest; the comparison is soft, so an
    upgraded daemon must keep replaying it rather than 422-ing every retry."""
    app, db, queries, _ = await _make_env()
    try:
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        await queries.insert_idempotency_inprogress(
            principal_id=LOCAL_SCOPE,
            idem_key="KLEGACY",
            method="POST",
            path="/services",
            expires_at=future,
        )
        await queries.complete_idempotency_record(
            principal_id=LOCAL_SCOPE,
            idem_key="KLEGACY",
            response_status=201,
            response_body='{"id": "pre-p22"}',
            content_type="application/json",
            resource_id="pre-p22",
        )
        async with _client(app) as client:
            resp = await client.post(
                "/services", json=_create_body(), headers={"Idempotency-Key": "KLEGACY"}
            )
        assert resp.status_code == 201
        assert resp.headers.get("Idempotent-Replay") == "true"
        assert resp.json()["id"] == "pre-p22"
    finally:
        await db.close()
