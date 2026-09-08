"""Route-level tests for the ``/databases`` surface (P15 WP2+WP3, P37 WP4).

Pure-async harness (``httpx.ASGITransport`` — never the Starlette ``TestClient``,
per the sandbox convention): the real databases router under the
``ScopedTokenAuthMiddleware`` with ``AsyncMock`` state for the authz / envelope /
audit assertions, plus a real ``SecretManager`` so the credential mint (WP2) is
exercised end-to-end, and a real in-memory database for the Idempotency-Key
replay and the cursor-paginated read (the ``test_models_route`` pattern).

P37 appends the dump trio (``POST …/dump``, ``GET …/dumps``, ``POST …/restore``)
in the same shape, with a fake ``ServiceController`` for the container half and
the REAL packer/extractor underneath it.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import stat
import tarfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import DatabasesSettings, ModelsSettings, NerditSettings
from nerdit.core import eventlog
from nerdit.core.backup import BackupError, create_dump_backup
from nerdit.core.data.backend import PostgresBackend, RedisBackend
from nerdit.core.data.controller import DataController
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.core.secrets import SecretManager
from nerdit.core.services import DumpError, DumpResult, RunPreconditionError
from nerdit.core.volumes import create_dump_staging_dir
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import QuotaExceeded, hash_token
from nerdit.daemon.errors import NerditError, RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import databases as databases_routes
from nerdit.daemon.routes.databases import router as databases_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries, ServiceNameTaken

LEGACY = "legacy-global"
ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
RO_RAW = "ro-raw"

_HEX64 = re.compile(r"\b[0-9a-f]{64}\b")

_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
}


def _db_job(**over) -> Job:
    fields = dict(
        id="db-1",
        service_name="pg",
        name="pg",
        kind=JobKind.database,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(
            {
                "backend": "postgres",
                "image": "postgres:16",
                "port": 5432,
                "volumes": ["data:/var/lib/postgresql/data"],
            }
        ),
    )
    fields.update(over)
    return Job(**fields)


def _queries() -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job: job)
    q.list_services = AsyncMock(return_value=([], None))
    return q


def _data_controller(queries, tmp_path) -> DataController:
    # Register both backends exactly like the daemon lifespan (P15.5 registry
    # proof) — Redis rides in as an extra_backends entry with no other change.
    redis_backend = RedisBackend()
    return DataController(
        PostgresBackend(),
        runtime=MagicMock(),
        queries=queries,
        extra_backends={redis_backend.name: redis_backend},
        default_backend="postgres",
        databases_settings=DatabasesSettings(),
        models_settings=ModelsSettings(),
        data_dir=str(tmp_path),
    )


def _make_app(queries, tmp_path, *, with_audit: bool = False) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(databases_router)

    app.state.queries = queries
    settings = MagicMock()
    settings.databases = DatabasesSettings()
    app.state.settings = settings
    app.state.data_controller = _data_controller(queries, tmp_path)
    app.state.secret_manager = SecretManager(tmp_path / "secrets")

    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


# --- POST /databases: create + row shape + mint -------------------------------


@pytest.mark.asyncio
async def test_create_defaults_to_postgres_name_prefix(tmp_path):
    q = _queries()
    app = _make_app(q, tmp_path)
    async with _client(app) as client:
        resp = await client.post("/databases", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "pg"
    assert body["backend"] == "postgres"
    assert body["status"] == "building"
    assert body["db_ready"] is False
    assert body["endpoint"] is None

    job_arg = q.reserve_service_for_token.await_args.args[0]
    assert job_arg.kind == JobKind.database
    assert job_arg.gpu_count == 0
    assert job_arg.desired_state == "running"
    assert job_arg.restart_policy == "on-failure"
    assert job_arg.submitted_by_token == "tok-sub"
    assert job_arg.health_check == {"type": "tcp", "start_period_s": 30}
    cfg = json.loads(job_arg.config)
    assert cfg == {
        "backend": "postgres",
        "image": "postgres:16",
        "port": 5432,
        "volumes": ["data:/var/lib/postgresql/data"],
    }

    # WP2: the credential was minted into the row's own secret scope, names-only
    # visible; the stored value is a 64-hex token (never returned).
    sm = app.state.secret_manager
    assert sm.list_keys("pg") == ["POSTGRES_PASSWORD"]
    assert _HEX64.fullmatch(sm.load("pg")["POSTGRES_PASSWORD"])


@pytest.mark.asyncio
async def test_create_honors_name_override(tmp_path):
    q = _queries()
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post(
            "/databases", json={"backend": "postgres", "name": "orders-db"}, headers=_auth(SUB_RAW)
        )
    assert resp.status_code == 201
    assert resp.json()["name"] == "orders-db"
    assert q.reserve_service_for_token.await_args.args[0].service_name == "orders-db"


@pytest.mark.asyncio
async def test_create_unknown_backend_is_422(tmp_path):
    q = _queries()
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post("/databases", json={"backend": "nope"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "db.unknown_backend"
    # The hint lists every registered backend (both after the P15.5 registry proof).
    assert "postgres" in body["hint"]
    assert "redis" in body["hint"]
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["shared", "rotate-key"])
async def test_create_rejects_reserved_names(name, tmp_path):
    q = _queries()
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post("/databases", json={"name": name}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"
    q.reserve_service_for_token.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_bad_name_is_422(tmp_path):
    async with _client(_make_app(_queries(), tmp_path)) as client:
        resp = await client.post("/databases", json={"name": "Bad_Name"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_duplicate_name_returns_409(tmp_path):
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameTaken("pg"))
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post("/databases", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.name_taken"


@pytest.mark.asyncio
async def test_create_quota_exceeded_returns_403(tmp_path):
    q = _queries()
    q.reserve_service_for_token = AsyncMock(
        side_effect=QuotaExceeded("max_concurrent_jobs", limit=1, current=1)
    )
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post("/databases", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 403
    assert resp.json()["code"] == "quota_exceeded"


@pytest.mark.asyncio
async def test_readonly_cannot_create(tmp_path):
    q = _queries()
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.post("/databases", json={}, headers=_auth(RO_RAW))
    assert resp.status_code == 403
    q.reserve_service_for_token.assert_not_awaited()


# --- audit --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_audited_with_names_only(tmp_path):
    q = _queries()
    async with _client(_make_app(q, tmp_path, with_audit=True)) as client:
        resp = await client.post("/databases", json={"backend": "postgres"}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201
    actions = {c.kwargs["action"] for c in q.insert_audit_log.await_args_list}
    assert "database.create" in actions
    assert "database.credential_minted" in actions
    # The minted-credential audit row carries the key NAME only — never a value.
    minted = next(
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs["action"] == "database.credential_minted"
    )
    params = json.loads(minted.kwargs["params_redacted"])
    assert params["keys"] == ["POSTGRES_PASSWORD"]
    assert not _HEX64.search(minted.kwargs["params_redacted"])


# --- GET /databases: projection + password-free endpoint ----------------------


@pytest.mark.asyncio
async def test_list_projects_password_free_endpoint(tmp_path):
    from nerdit.db.models import ServiceEndpoint

    q = _queries()
    job = _db_job(
        status=JobStatus.running,
        config=json.dumps(
            {
                "backend": "postgres",
                "image": "postgres:16",
                "port": 5432,
                "volumes": ["data:/var/lib/postgresql/data"],
                "db_ready": True,
            }
        ),
    )
    q.list_services = AsyncMock(return_value=([job], None))
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name="pg", job_id="db-1", container_port=5432, host_port=9500
        )
    )
    async with _client(_make_app(q, tmp_path)) as client:
        resp = await client.get("/databases", headers=_auth(RO_RAW))
    assert resp.status_code == 200
    (item,) = resp.json()["items"]
    assert item["db_ready"] is True
    # Password-free host:port — never a DSN with a credential.
    assert ":9500" in item["endpoint"]
    assert "postgresql://" not in item["endpoint"]
    assert q.list_services.await_args.kwargs["kinds"] == (JobKind.database,)


@pytest.mark.asyncio
async def test_list_limit_is_bounded(tmp_path):
    async with _client(_make_app(_queries(), tmp_path)) as client:
        too_low = await client.get("/databases?limit=0", headers=_auth(RO_RAW))
        too_high = await client.get("/databases?limit=201", headers=_auth(RO_RAW))
    assert too_low.status_code == 422
    assert too_high.status_code == 422


# --- real-DB flow: idempotent replay + cursor pagination ----------------------


async def _make_db_env(tmp_path) -> tuple[FastAPI, Database, Queries]:
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(databases_router)
    app.state.queries = queries
    app.state.settings = NerditSettings()
    app.state.data_controller = _data_controller(queries, tmp_path)
    app.state.secret_manager = SecretManager(tmp_path / "secrets")

    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, db, queries


@pytest.mark.asyncio
async def test_idempotency_key_replays_same_database_row(tmp_path):
    app, db, queries = await _make_db_env(tmp_path)
    try:
        async with _client(app) as client:
            headers = {"Idempotency-Key": "K-db"}
            first = await client.post("/databases", json={}, headers=headers)
            assert first.status_code == 201
            assert first.headers.get("Idempotent-Replay") is None

            second = await client.post("/databases", json={}, headers=headers)
            assert second.status_code == 201
            assert second.headers.get("Idempotent-Replay") == "true"
            assert second.json()["id"] == first.json()["id"]

        rows, _ = await queries.list_services(kinds=(JobKind.database,))
        assert len(rows) == 1
    finally:
        await db.close()


# --- pure branch tests: /wait, /diagnose, remediation -------------------------


def test_evaluate_wait_database_branch():
    from nerdit.daemon.routes.service_wait import _evaluate_wait

    running_ready = _db_job(
        status=JobStatus.running,
        config=json.dumps({"backend": "postgres", "db_ready": True}),
    )
    assert _evaluate_wait(running_ready, None, False) == ("converged", None)

    running_not_ready = _db_job(
        status=JobStatus.running, config=json.dumps({"backend": "postgres"})
    )
    assert _evaluate_wait(running_not_ready, None, False) == (None, None)

    failed = _db_job(status=JobStatus.failed, config=json.dumps({"backend": "postgres"}))
    assert _evaluate_wait(failed, None, False) == ("failed", None)


def test_database_env_key_names_is_allowlisted(tmp_path):
    from types import SimpleNamespace

    from nerdit.daemon.routes.service_diagnose import _database_env_key_names

    dc = _data_controller(_queries(), tmp_path)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(data_controller=dc)))
    job = _db_job()
    cfg = json.loads(job.config)
    names = _database_env_key_names(request, job, cfg)
    # Exactly the four allowlisted backend statics + the minted key NAME — never
    # PORT, never the full secret scope.
    assert set(names) == {"POSTGRES_USER", "POSTGRES_DB", "PGDATA", "POSTGRES_PASSWORD"}


def test_remediation_db_not_ready():
    """Rule 7b keys on STATE, not an error-message prefix (nothing writes one in
    v1): kind=database + running + db_ready falsy + past the start-period grace.
    """
    from datetime import datetime, timedelta, timezone

    from nerdit.daemon.remediation import BindingWait, RemediationCode, derive_remediation

    # A stuck row: running, no db_ready, no error message, started well before now.
    job = _db_job(
        status=JobStatus.running,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    cfg = json.loads(job.config)  # no db_ready
    code, detail = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is RemediationCode.db_not_ready
    assert "recreate" in detail.lower()


def test_remediation_db_not_ready_negative_within_start_period():
    """A database still inside its start-period grace is NOT flagged stuck — it
    is expected to be un-ready while initdb runs (falls through to inspect_logs).
    """
    from datetime import datetime, timezone

    from nerdit.daemon.remediation import BindingWait, RemediationCode, derive_remediation

    job = _db_job(
        status=JobStatus.running,
        started_at=datetime.now(timezone.utc),
        health_check={"type": "tcp", "start_period_s": 300.0},
    )
    cfg = json.loads(job.config)  # no db_ready
    code, _ = derive_remediation(job, cfg, {}, BindingWait(), None)
    assert code is not RemediationCode.db_not_ready


@pytest.mark.asyncio
async def test_list_paginates_and_excludes_services(tmp_path):
    app, db, queries = await _make_db_env(tmp_path)
    try:
        await queries.reserve_service_for_token(
            Job(
                id="svc-1",
                kind=JobKind.service,
                service_name="webapp",
                gpu_count=0,
                status=JobStatus.running,
                desired_state="running",
            )
        )
        async with _client(app) as client:
            for name in ("pg-a", "pg-b", "pg-c"):
                resp = await client.post("/databases", json={"name": name})
                assert resp.status_code == 201
            page1 = (await client.get("/databases?limit=2")).json()
            assert len(page1["items"]) == 2
            assert page1["next_cursor"]
            page2 = (await client.get(f"/databases?limit=2&cursor={page1['next_cursor']}")).json()
            assert len(page2["items"]) == 1
            assert page2["next_cursor"] is None
        names = {i["name"] for i in page1["items"]} | {i["name"] for i in page2["items"]}
        assert names == {"pg-a", "pg-b", "pg-c"}
        assert "webapp" not in names
    finally:
        await db.close()


# --- (P37) Managed-database dumps ---------------------------------------------
#
# The dump trio is route-level machinery over WP1-WP3: the controller and the
# packer have their own suites (``test_db_dump_controller.py``,
# ``test_backup_dumps.py``), so what is pinned HERE is the gatekeeping — the
# D-P37-10 ladder in its locked order with the §1.4 codes, the auth split, the
# audit params, the durable events and the custody posture.
#
# The controller is a fake, but a HONEST one: the dump fake really writes a real
# payload into a real staging dir, so the success path exercises
# ``create_dump_backup`` end to end (tar on disk, ``0o600``, manifest, sha256)
# rather than asserting against a mock's return value.

_SCOPED_RAW = "scoped-raw"

_TOKENS[hash_token(_SCOPED_RAW)] = ApiToken(
    id="tok-scoped",
    name="scoped",
    role=TokenRole.submitter,
    token_hash=hash_token(_SCOPED_RAW),
    scope_services=["other-db"],
)


def _ready_db_job(**over) -> Job:
    """A ``kind=database`` row in the one state a dump accepts: running + db_ready."""
    fields = dict(
        status=JobStatus.running,
        submitted_by_token="tok-sub",
        config=json.dumps(
            {
                "backend": "postgres",
                "image": "postgres:16",
                "port": 5432,
                "volumes": ["data:/var/lib/postgresql/data"],
                "db_ready": True,
            }
        ),
    )
    fields.update(over)
    return _db_job(**fields)


class _FakeDumpController:
    """Stand-in for the ``ServiceController`` halves the dump routes call.

    Reproduces the two contracts the route depends on and nothing else:

    * ``dump_database`` creates a real staging dir through the real
      ``create_dump_staging_dir`` and writes a real payload into it, then awaits
      the route's ``on_captured`` packer with the result before returning it —
      exactly the hand-off the shipped method makes (the packer runs while the
      slot is held), so the route's packing leg runs for real;
    * ``restore_database`` asserts the staging dir it is handed is already
      POPULATED (the route extracts before it calls), which is the half of the
      hand-off a mock would otherwise let regress silently.

    ``raises`` lets a test make either half fail with the real exception type.
    """

    def __init__(
        self, data_dir, *, raises: Exception | None = None, payload: bytes = b"PGDMP\x00x"
    ):
        self.data_dir = Path(data_dir)
        self.draining = False
        self.raises = raises
        self.payload = payload
        self.dump_calls: list[str] = []
        self.captured_under_slot: list[str] = []
        self.restore_calls: list[str] = []
        self.restore_saw_payload: list[bool] = []
        self.slot_events: list[tuple[str, str]] = []
        self.staging_empty_at_reserve: list[bool] = []

    def reserve_dump_slot(self, job_id, run_id):
        root = self.data_dir / "dump-staging"
        self.staging_empty_at_reserve.append(not root.exists() or not any(root.iterdir()))
        self.slot_events.append(("reserve", run_id))
        if isinstance(self.raises, RunPreconditionError):
            raise self.raises

    def release_dump_slot(self, job_id, run_id):
        self.slot_events.append(("release", run_id))

    def _capture(self, run_id) -> DumpResult:
        """Stage what the dump tool would have written and describe it.

        Overridden by the tests that plant something the packer must refuse.
        """
        staging = create_dump_staging_dir(self.data_dir, run_id)
        output = staging / "dump.pgdump"
        output.write_bytes(self.payload)
        return DumpResult(
            exit_code=0,
            timed_out=False,
            log_tail=["pg_dump: done"],
            output_path=str(output),
            started_at="2026-09-07T10:00:00+00:00",
            finished_at="2026-09-07T10:00:01+00:00",
        )

    async def dump_database(self, job, *, run_id, timeout_s, on_captured=None):
        self.dump_calls.append(run_id)
        if self.raises is not None:
            raise self.raises
        captured = self._capture(run_id)
        if on_captured is not None:
            self.captured_under_slot.append(run_id)
            await on_captured(captured)
        return captured

    async def restore_database(self, job, *, run_id, staging, timeout_s, slot_held=False):
        assert slot_held is True, "the route must own the slot it reserved"
        self.restore_calls.append(run_id)
        self.restore_saw_payload.append((Path(staging) / "dump.pgdump").is_file())
        if self.raises is not None:
            raise self.raises
        return DumpResult(
            exit_code=0,
            timed_out=False,
            log_tail=["pg_restore: done"],
            output_path=None,
            started_at="2026-09-07T10:00:00+00:00",
            finished_at="2026-09-07T10:00:01+00:00",
        )


class _RecordedEvents:
    """Minimal ``EventRecorder`` stand-in capturing every ``record`` call."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def record(self, type, *, kind=None, service_name=None, reason=None, data=None, **_):  # noqa: A002
        self.rows.append(
            {
                "type": type,
                "kind": kind,
                "service_name": service_name,
                "reason": reason,
                "data": data,
            }
        )


def _dump_settings(tmp_path, *, dump_timeout_max_s: int = 3600, max_concurrent_dumps: int = 2):
    """Real-enough settings for the dump routes (data_dir + the two [services] caps)."""
    settings = MagicMock()
    settings.databases = DatabasesSettings()
    settings.data_dir = str(tmp_path / "data")
    settings.services.dump_timeout_max_s = dump_timeout_max_s
    settings.services.max_concurrent_dumps = max_concurrent_dumps
    return settings


def _dump_app(
    queries,
    tmp_path,
    *,
    controller=None,
    with_audit: bool = False,
    with_idempotency: bool = False,
    settings=None,
) -> FastAPI:
    """The databases router with everything the dump trio reads off app.state."""
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(databases_router)

    app.state.queries = queries
    app.state.settings = settings or _dump_settings(tmp_path)
    app.state.data_controller = _data_controller(queries, tmp_path)
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.service_controller = controller

    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _dump_queries(job: Job | None = None, *, endpoint_port: int = 9500) -> AsyncMock:
    """``_queries()`` plus a resolvable database row with a live endpoint."""
    from nerdit.db.models import ServiceEndpoint

    q = _queries()
    row = job if job is not None else _ready_db_job()
    q.get_job = AsyncMock(side_effect=lambda ident: row if ident == row.id else None)
    q.get_service_by_name = AsyncMock(
        side_effect=lambda ident: row if ident == row.service_name else None
    )
    q.get_service_endpoint = AsyncMock(
        return_value=ServiceEndpoint(
            service_name=row.service_name,
            job_id=row.id,
            container_port=5432,
            host_port=endpoint_port,
        )
    )
    q.list_workload_configs = AsyncMock(return_value=[])
    q.set_last_dump = AsyncMock(return_value=True)
    q.patch_last_dump_field = AsyncMock(return_value=True)
    return q


def _key(name: str = "K-dump") -> dict:
    return {"Idempotency-Key": name}


@pytest.fixture
def _no_disk_pressure(monkeypatch):
    """Make the disk pre-flight pass without touching a real volume tree.

    Both patched at the ROUTE module's namespace (never the global ``shutil`` /
    ``utils.disk``): a test that silently patched a process-wide primitive would
    also silence the retention sweep and the ``/system/disk`` walk running in
    the same session.
    """
    monkeypatch.setattr(databases_routes, "du_bytes", lambda _p: 1024)
    monkeypatch.setattr(databases_routes, "_free_bytes_for", lambda _p: 10 * 1024**3)


# --- auth matrix --------------------------------------------------------------


@pytest.mark.asyncio
async def test_dump_readonly_is_403(tmp_path, _no_disk_pressure):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post("/databases/pg/dump", json={}, headers={**_auth(RO_RAW), **_key()})
    assert resp.status_code == 403
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_dump_non_owner_submitter_is_403(tmp_path, _no_disk_pressure):
    """A submitter who did NOT create the database cannot dump it (D-P37-8)."""
    q = _dump_queries(_ready_db_job(submitted_by_token="tok-someone-else"))
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(SUB_RAW), **_key()}
        )
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_dump_owner_submitter_succeeds(tmp_path, _no_disk_pressure):
    """The whole point of the auth split: the submitter who owns it may dump it."""
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(SUB_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["service_name"] == "pg"
    assert body["engine"] == "postgres"
    assert body["format"] == "pg_custom"
    assert re.fullmatch(r"nerdit-dump-pg-\d{8}T\d{6}Z-[0-9a-f]{6}\.tar\.gz", body["dump"])
    # The tar really exists, ``0o600``, under the daemon's own backups dir — and
    # the staging dir it was packed from is gone.
    tar = tmp_path / "data" / "backups" / body["dump"]
    assert tar.is_file()
    assert stat.S_IMODE(tar.stat().st_mode) == 0o600
    assert not list((tmp_path / "data" / "dump-staging").iterdir())
    # Custody: metadata only — no path, and the tar is never served.
    assert not any(isinstance(v, str) and "/" in v for k, v in body.items() if k != "sha256")


@pytest.mark.asyncio
async def test_dump_stamps_the_tar_basename_onto_last_dump(tmp_path, _no_disk_pressure):
    """The controller stamps ``dump: None`` (no tar yet); the route fills it in.

    Without this the ``/diagnose`` projection would say "a dump ran and produced
    nothing", which is exactly the shape of a FAILED dump — the one distinction
    an operator reading that panel needs.
    """
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    # The row the stamp reads back carries the controller's own ``dump: None``
    # blob, as it does live.
    stamped = _ready_db_job()
    cfg = json.loads(stamped.config)
    cfg["last_dump"] = {"run_id": "abc123def456", "kind": "dump", "dump": None, "log_tail": []}
    stamped.config = json.dumps(cfg)
    q.get_job = AsyncMock(return_value=stamped)

    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    (job_id,), kw = q.patch_last_dump_field.await_args
    assert job_id == stamped.id
    assert kw["field"] == "dump"
    assert kw["value"] == resp.json()["dump"]
    # One guarded json_set on the sub-key: scoped to the run that made the tar.
    assert kw["expect_run_id"] == controller.dump_calls[0]


def _job_with_controller_stamp() -> Job:
    """A ready row already carrying the blob ``dump_database``'s ``finally`` writes.

    The controller decides "success" the moment the tool's output verifies and
    stamps ``reason: None`` / ``dump: None`` before the route's packer has
    returned. Every re-stamp assertion below is about correcting THAT blob, so
    the tests have to start from it rather than from an absent key.
    """
    job = _ready_db_job()
    cfg = json.loads(job.config)
    cfg["last_dump"] = {
        "run_id": "abc123def456",
        "kind": "dump",
        "exit_code": 0,
        "timed_out": False,
        "reason": None,
        "dump": None,
        "log_tail": ["pg_dump: done"],
    }
    job.config = json.dumps(cfg)
    return job


@pytest.mark.asyncio
async def test_dump_response_created_at_matches_the_manifest(tmp_path, _no_disk_pressure):
    """The tar and the response must not disagree about when the dump was taken."""
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    tar = tmp_path / "data" / "backups" / resp.json()["dump"]
    with tarfile.open(tar) as archive:
        manifest = json.loads(archive.extractfile("dump-manifest.json").read())
    assert datetime.fromisoformat(resp.json()["created_at"]) == datetime.fromisoformat(
        manifest["created_at"]
    )
    assert resp.json()["sha256"] == manifest["sha256"]
    assert manifest["engine"] == "postgres"
    assert manifest["tool"] == {"argv0": "pg_dump"}


@pytest.mark.asyncio
async def test_dump_scoped_token_miss_is_403(tmp_path, _no_disk_pressure):
    """A scoped token that owns the row but not the NAME is still refused (P25)."""
    q = _dump_queries(_ready_db_job(submitted_by_token="tok-scoped"))
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(_SCOPED_RAW), **_key()}
        )
    assert resp.status_code == 403
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_list_dumps_non_owner_is_403(tmp_path):
    """Reads deviate from role-only here for the P29 reason: it is data (D-P37-8)."""
    q = _dump_queries(_ready_db_job(submitted_by_token="tok-someone-else"))
    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.get("/databases/pg/dumps", headers=_auth(RO_RAW))
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_restore_submitter_is_403_even_when_owner(tmp_path):
    """Restore is admin-only: destructiveness, not ownership, decides (D-P37-8)."""
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz"},
            headers={**_auth(SUB_RAW), **_key()},
        )
    assert resp.status_code == 403
    assert controller.restore_calls == []


@pytest.mark.asyncio
async def test_dump_without_idempotency_key_is_400(tmp_path, _no_disk_pressure):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post("/databases/pg/dump", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 400
    assert resp.json()["code"] == "idempotency_key_required"
    assert controller.dump_calls == []


# --- the D-P37-10 precondition ladder, in order -------------------------------


@pytest.mark.asyncio
async def test_dump_unknown_row_is_404(tmp_path):
    q = _dump_queries()
    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.post(
            "/databases/nope/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_dump_of_a_service_row_is_422_not_a_database(tmp_path):
    """A service row is refused with the dump code, NOT ``run.not_supported`` (frozen)."""
    q = _dump_queries(_ready_db_job(kind=JobKind.service, service_name="pg"))
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 422
    assert resp.json()["code"] == "dump.not_a_database"
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_dump_without_a_data_plane_is_503(tmp_path):
    q = _dump_queries()
    app = _dump_app(q, tmp_path, controller=_FakeDumpController(tmp_path / "data"))
    app.state.data_controller = None
    async with _client(app) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 503
    assert resp.json()["code"] == "db.unavailable"


@pytest.mark.asyncio
async def test_dump_while_draining_is_409(tmp_path):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    controller.draining = True
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "daemon.restart_in_progress"
    assert controller.dump_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "over",
    [
        {"status": JobStatus.building},
        {"config": json.dumps({"backend": "postgres", "image": "postgres:16", "port": 5432})},
    ],
    ids=["not-running", "running-but-not-db_ready"],
)
async def test_dump_of_a_not_ready_database_is_409(over, tmp_path):
    """Status AND ``db_ready`` are a conjunction — either half missing refuses."""
    q = _dump_queries(_ready_db_job(**over))
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "dump.database_not_ready"
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_dump_without_a_published_port_is_409(tmp_path):
    q = _dump_queries()
    q.get_service_endpoint = AsyncMock(return_value=None)
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "dump.database_not_ready"


@pytest.mark.asyncio
async def test_dump_over_the_timeout_cap_is_422_naming_the_key(tmp_path):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    settings = _dump_settings(tmp_path, dump_timeout_max_s=120)
    async with _client(_dump_app(q, tmp_path, controller=controller, settings=settings)) as client:
        resp = await client.post(
            "/databases/pg/dump",
            json={"timeout_s": 900},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "dump.timeout_too_large"
    # The hint names the cap AND the key, so the operator can raise it.
    assert "120" in body["hint"]
    assert "[services].dump_timeout_max_s" in body["hint"]
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_dump_insufficient_disk_is_409_with_detail(tmp_path, monkeypatch):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    monkeypatch.setattr(databases_routes, "du_bytes", lambda _p: 5 * 1024**3)
    monkeypatch.setattr(databases_routes, "_free_bytes_for", lambda _p: 1024)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "dump.insufficient_disk"
    assert body["detail"]["free_bytes"] == 1024
    assert body["detail"]["required_bytes"] == 5 * 1024**3 + 64 * 1024 * 1024
    # A refused dump claims no slot and starts no container.
    assert controller.dump_calls == []


@pytest.mark.asyncio
async def test_disk_preflight_runs_off_the_event_loop(tmp_path, monkeypatch):
    """The walk is blocking; running it on the loop stalls every other request.

    ``du_bytes`` is a full ``os.walk`` of a PGDATA that can be multi-GB, so the
    pre-flight is the one check that MUST be on a worker thread — and the only
    way to prove it is to ask, from inside, whether a loop is running.
    """
    seen: list[str] = []

    def _probe(_path):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            seen.append("off-loop")
            return 1024
        raise AssertionError("du_bytes ran on the event loop")

    monkeypatch.setattr(databases_routes, "du_bytes", _probe)
    monkeypatch.setattr(databases_routes, "_free_bytes_for", lambda _p: 10 * 1024**3)
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    assert seen == ["off-loop"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "code"),
    [("run_in_progress", "dump.in_progress"), ("too_many_dumps", "dump.too_many_in_flight")],
)
async def test_dump_slot_refusals_map_to_their_codes(reason, code, tmp_path, _no_disk_pressure):
    """``_register_run``'s authoritative refusals, projected per §1.4."""
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data", raises=RunPreconditionError(reason))
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == code
    # The two codes must be legible as different scopes from the hint alone.
    if code == "dump.in_progress":
        assert "backup.in_progress" in resp.json()["hint"]
    else:
        assert "[services].max_concurrent_dumps" in resp.json()["hint"]


@pytest.mark.asyncio
async def test_dump_failure_is_500_with_the_last_tail_line_as_hint(tmp_path, _no_disk_pressure):
    """D-P37-11: loud, path-free, and the tail's LAST line is the actionable bit."""
    q = _dump_queries()
    controller = _FakeDumpController(
        tmp_path / "data",
        raises=DumpError("exit_nonzero", log_tail=["connecting…", "pg_dump: error: no such db"]),
    )
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "dump.failed"
    assert body["reason"] == "exit_nonzero"
    assert body["hint"] == "pg_dump: error: no such db"
    # Nothing is left behind and no tar was written.
    assert not (tmp_path / "data" / "backups").exists()


@pytest.mark.asyncio
async def test_dump_runtime_refusal_is_500_dump_failed_with_the_no_outcome_reason(
    tmp_path, monkeypatch, _no_disk_pressure
):
    """A sibling that never started answers with the SAME code as one that failed.

    Deliberately not the 503 ``run.runtime_unavailable`` the P20 run route uses:
    §1.4's registry is the locked list of what this surface may say, so the
    distinction rides ``reason`` — the one field an agent branches on — rather
    than a code the registry does not contain.
    """
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)
    q = _dump_queries()
    controller = _FakeDumpController(
        tmp_path / "data", raises=ContainerRuntimeError("docker socket at /var/run/docker.sock")
    )
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "dump.failed"
    assert body["reason"] == "interrupted"
    # Path-free by contract: the runtime's own message carries host paths.
    assert "/var/run" not in json.dumps(body)
    assert [row["type"] for row in events.rows] == ["database.dump_failed"]


@pytest.mark.asyncio
async def test_dump_output_refused_by_the_packer_is_500_and_leaves_no_tar(
    tmp_path, monkeypatch, _no_disk_pressure
):
    """The sibling exited 0 and wrote a symlink — the real WP3 refusal, end to end.

    Not a mocked packer: the fake controller plants the link where the tool's
    output belongs and ``create_dump_backup``'s own ``lstat`` refuses to follow
    it, which is the property D-P37-7 exists for.
    """
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)

    class _PlantsASymlink(_FakeDumpController):
        def _capture(self, run_id):
            staging = create_dump_staging_dir(self.data_dir, run_id)
            (self.data_dir / "target").write_bytes(b"secret")
            (staging / "dump.pgdump").symlink_to(self.data_dir / "target")
            return DumpResult(
                exit_code=0,
                timed_out=False,
                log_tail=["pg_dump: done"],
                output_path=str(staging / "dump.pgdump"),
                started_at="2026-09-07T10:00:00+00:00",
                finished_at="2026-09-07T10:00:01+00:00",
            )

    q = _dump_queries(_job_with_controller_stamp())
    controller = _PlantsASymlink(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "dump.failed"
    assert resp.json()["reason"] == "output_not_regular"
    assert not list((tmp_path / "data" / "backups").glob("nerdit-dump-*.tar.gz"))
    assert not list((tmp_path / "data" / "dump-staging").iterdir())
    (row,) = events.rows
    assert row["reason"] == "output_not_regular"
    # The row must not keep the controller's optimistic ``reason: null``: with
    # ``dump: null`` beside it, /diagnose renders a FAILED dump as a successful
    # one that produced nothing.
    assert _stamped_last_dump(q)["reason"] == "output_not_regular"


def _stamped_last_dump(q) -> dict:
    """The ``last_dump`` keys the route patched (field -> value), run-scoped."""
    patched: dict = {}
    for call in q.patch_last_dump_field.await_args_list:
        assert call.kwargs["expect_run_id"]
        patched[call.kwargs["field"]] = call.kwargs["value"]
    # The route never rewrites the blob: nothing but the named keys can change.
    assert not q.set_last_dump.await_args_list
    return patched


@pytest.mark.asyncio
async def test_dump_pack_io_failure_stamps_interrupted_onto_last_dump(
    tmp_path, monkeypatch, _no_disk_pressure
):
    """The daemon's own I/O failed while packing: no artifact, so ``interrupted``."""
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)

    def _boom(*_a, **_kw):
        raise BackupError("could not write the archive")

    monkeypatch.setattr(databases_routes, "create_dump_backup", _boom)

    q = _dump_queries(_job_with_controller_stamp())
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "dump.failed"
    assert resp.json()["reason"] == "interrupted"
    assert _stamped_last_dump(q)["reason"] == "interrupted"
    # No tar, so no basename patch; ``exit_code`` is the controller's and untouched.
    assert "dump" not in _stamped_last_dump(q)
    assert [row["reason"] for row in events.rows] == ["interrupted"]


@pytest.mark.asyncio
async def test_dump_disk_recheck_refusal_stamps_interrupted_onto_last_dump(
    tmp_path, monkeypatch, _no_disk_pressure
):
    """The post-capture 409 is the one packer refusal with no reason of its own,
    and it still lands on the feed as ``database.dump_failed``."""
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)

    async def _refuse(*_a, **_kw):
        raise NerditError(
            409,
            "dump.insufficient_disk",
            "There is not enough free disk to pack this dump.",
            detail={"required_bytes": 10, "free_bytes": 1},
        )

    monkeypatch.setattr(databases_routes, "_recheck_disk", _refuse)

    q = _dump_queries(_job_with_controller_stamp())
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "dump.insufficient_disk"
    assert _stamped_last_dump(q)["reason"] == "interrupted"
    assert not list((tmp_path / "data" / "dump-staging").iterdir())
    assert [(row["type"], row["reason"]) for row in events.rows] == [
        ("database.dump_failed", "interrupted")
    ]


@pytest.mark.asyncio
async def test_a_cancelled_pack_unlinks_the_tar_the_worker_published(tmp_path, monkeypatch):
    """A cancellation mid-pack settles the worker and removes what it wrote:
    a run recorded ``interrupted`` leaves no listable tar (D-P37-11)."""
    data_dir = tmp_path / "data"
    backups = data_dir / "backups"
    backups.mkdir(parents=True)
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _slow_pack(*_a, **_kw):
        loop.call_soon_threadsafe(started.set)
        import time as _t

        _t.sleep(0.2)
        tar = backups / "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz"
        tar.write_bytes(b"gz")
        from nerdit.core.backup import DumpBackupResult

        return DumpBackupResult(
            path=str(tar),
            basename=tar.name,
            size_bytes=2,
            sha256="0" * 64,
            service="pg",
            engine="postgres",
            manifest=None,  # type: ignore[arg-type]
        )

    async def _ok(*_a, **_kw):
        return None

    monkeypatch.setattr(databases_routes, "create_dump_backup", _slow_pack)
    monkeypatch.setattr(databases_routes, "_recheck_disk", _ok)
    staging = create_dump_staging_dir(data_dir, "abc123def456")
    output = staging / "dump.pgdump"
    output.write_bytes(b"PGDMP")
    captured = DumpResult(
        exit_code=0,
        timed_out=False,
        log_tail=[],
        output_path=str(output),
        started_at="2026-09-07T10:00:00+00:00",
        finished_at="2026-09-07T10:00:01+00:00",
    )
    task = asyncio.create_task(
        databases_routes._pack_or_fail(
            data_dir, captured, service_name="pg", image="postgres:16", backend=PostgresBackend()
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(backups.iterdir()) == []
    assert not staging.exists()


@pytest.mark.asyncio
async def test_the_route_packs_while_the_controller_still_holds_the_slot(
    tmp_path, _no_disk_pressure
):
    """(D-P37-9) Packing is handed to ``dump_database``, not run after it.

    Released before the pack, the slot would leave ``DELETE /services/{name}``,
    a second dump and the restart drain all open across the disk-heaviest half
    of the operation. The controller-tier pin that the slot is genuinely still
    held lives in ``tests/test_db_dump_controller.py``; what this asserts is the
    half the route owns — that it hands its packer over rather than calling it
    itself.
    """
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    assert controller.captured_under_slot == controller.dump_calls
    assert (tmp_path / "data" / "backups" / resp.json()["dump"]).is_file()


# --- events -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dump_success_records_the_durable_event(tmp_path, monkeypatch, _no_disk_pressure):
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    (row,) = events.rows
    assert row["type"] == "database.dump_succeeded"
    assert row["kind"] == "database"
    assert row["service_name"] == "pg"
    assert row["reason"] is None
    assert set(row["data"]) == {"dump", "bytes"}
    assert row["data"]["dump"] == resp.json()["dump"]


@pytest.mark.asyncio
async def test_dump_failure_event_carries_a_reason_and_no_log_line(
    tmp_path, monkeypatch, _no_disk_pressure
):
    """The tail never reaches the feed: it is POSTed to third-party hosts (D-P37-11).

    The tail here is a ``pg_restore``-shaped diagnostic printing a row value —
    the exact payload the decision exists to keep out of a webhook body.
    """
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)
    q = _dump_queries()
    controller = _FakeDumpController(
        tmp_path / "data",
        raises=DumpError("timed_out", log_tail=["DETAIL:  Key (id)=(1) already exists."]),
    )
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    (row,) = events.rows
    assert row["type"] == "database.dump_failed"
    assert row["reason"] == "timed_out"
    assert row["data"] is None
    assert "DETAIL:" not in json.dumps(row)


@pytest.mark.asyncio
async def test_all_four_dump_event_types_are_declared_and_never_coalesced():
    """The vocabulary is refused at the recorder if it is not declared (P24a)."""
    expected = {
        "database.dump_succeeded",
        "database.dump_failed",
        "database.restore_succeeded",
        "database.restore_failed",
    }
    assert expected <= eventlog.EVENT_TYPES
    assert expected <= eventlog._NEVER_COALESCED


# --- audit --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dump_audit_params_shape(tmp_path, _no_disk_pressure):
    """``{service, dump, engine, bytes}`` — never argv, never a path, never a tail."""
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    app = _dump_app(q, tmp_path, controller=controller, with_audit=True)
    async with _client(app) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 200, resp.text
    row = next(
        c for c in q.insert_audit_log.await_args_list if c.kwargs["action"] == "database.dump"
    )
    assert row.kwargs["target_type"] == "database"
    assert row.kwargs["target_id"] == "pg"
    params = json.loads(row.kwargs["params_redacted"])
    assert set(params) == {"service", "dump", "engine", "bytes"}
    assert params["service"] == "pg"
    assert params["engine"] == "postgres"
    assert params["dump"] == resp.json()["dump"]
    assert params["bytes"] == resp.json()["size_bytes"]


@pytest.mark.asyncio
async def test_failed_dump_is_audited_as_error(tmp_path, _no_disk_pressure):
    """A dump that never produced a tar still leaves an audit row, ``result='error'``."""
    q = _dump_queries()
    controller = _FakeDumpController(
        tmp_path / "data", raises=ContainerRuntimeError("docker socket at /var/run/docker.sock")
    )
    app = _dump_app(q, tmp_path, controller=controller, with_audit=True)
    async with _client(app) as client:
        resp = await client.post(
            "/databases/pg/dump", json={}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 500
    row = next(
        c for c in q.insert_audit_log.await_args_list if c.kwargs["action"] == "database.dump"
    )
    assert row.kwargs["result"] == "error"
    assert row.kwargs["target_id"] == "pg"
    assert "/var/run" not in json.dumps(row.kwargs)


@pytest.mark.asyncio
async def test_restore_audit_params_shape_and_denials_are_recorded(tmp_path):
    """A refused restore still audits the database it was aimed at (domains posture)."""
    q = _dump_queries()
    app = _dump_app(q, tmp_path, controller=_FakeDumpController(tmp_path / "data"), with_audit=True)
    async with _client(app) as client:
        resp = await client.post(
            "/databases/pg/restore?force=true",
            json={"dump": "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz"},
            headers={**_auth(SUB_RAW), **_key()},
        )
    assert resp.status_code == 403
    row = next(
        c for c in q.insert_audit_log.await_args_list if c.kwargs["action"] == "database.restore"
    )
    params = json.loads(row.kwargs["params_redacted"])
    assert set(params) == {"service", "dump", "engine", "force"}
    assert params["force"] is True
    # Refused before the row's backend was resolved, so the engine is honestly null.
    assert params["engine"] is None


# --- GET /databases/{name}/dumps ----------------------------------------------


@pytest.mark.asyncio
async def test_list_dumps_filters_by_service_and_sorts_newest_first(tmp_path):
    q = _dump_queries()
    backups = tmp_path / "data" / "backups"
    backups.mkdir(parents=True)
    names = [
        "nerdit-dump-pg-20260901T100000Z-aaaaaa.tar.gz",
        "nerdit-dump-pg-20260902T100000Z-bbbbbb.tar.gz",
        # Another database, the v1 control-plane tar and the v2 volume tar are
        # all in the same directory and must never be listed here.
        "nerdit-dump-cache-20260903T100000Z-cccccc.tar.gz",
        "nerdit-backup-20260903T100000Z.tar.gz",
        "nerdit-volumes-pg-20260903T100000Z.tar.gz",
    ]
    for i, name in enumerate(names):
        path = backups / name
        path.write_bytes(b"x" * (i + 1))
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))

    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.get("/databases/pg/dumps", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    body = resp.json()
    assert body["service_name"] == "pg"
    assert [d["dump"] for d in body["dumps"]] == [
        "nerdit-dump-pg-20260902T100000Z-bbbbbb.tar.gz",
        "nerdit-dump-pg-20260901T100000Z-aaaaaa.tar.gz",
    ]
    assert body["dumps"][0]["size_bytes"] == 2


@pytest.mark.asyncio
async def test_list_dumps_is_capped_and_reports_truncation(tmp_path, monkeypatch):
    """Invariant #3: the read is bounded by a module constant, not by retention.

    ``[retention].dump_keep_last = 0`` ("never prune") is a supported operator
    choice, so a listing that is only small because retention happens to be on
    is not a bounded read. The cap is lowered here rather than writing 501 tars.
    """
    monkeypatch.setattr(databases_routes, "DUMP_LIST_MAX", 3)
    q = _dump_queries()
    backups = tmp_path / "data" / "backups"
    backups.mkdir(parents=True)
    for i in range(5):
        path = backups / f"nerdit-dump-pg-2026090{i + 1}T100000Z-aaaaa{i}.tar.gz"
        path.write_bytes(b"x")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))

    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.get("/databases/pg/dumps", headers=_auth(ADMIN_RAW))
    body = resp.json()
    assert body["truncated"] is True
    # The NEWEST three, in newest-first order: a cap that dropped the newest
    # would hide exactly the dump a caller is looking for.
    assert [d["dump"] for d in body["dumps"]] == [
        "nerdit-dump-pg-20260905T100000Z-aaaaa4.tar.gz",
        "nerdit-dump-pg-20260904T100000Z-aaaaa3.tar.gz",
        "nerdit-dump-pg-20260903T100000Z-aaaaa2.tar.gz",
    ]

    monkeypatch.setattr(databases_routes, "DUMP_LIST_MAX", 500)
    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.get("/databases/pg/dumps", headers=_auth(ADMIN_RAW))
    assert resp.json()["truncated"] is False
    assert len(resp.json()["dumps"]) == 5


@pytest.mark.asyncio
async def test_list_dumps_with_no_backups_dir_is_an_empty_list(tmp_path):
    """ "No dumps" is an empty list, never a 404 — the caller's next step is the same."""
    q = _dump_queries()
    async with _client(_dump_app(q, tmp_path)) as client:
        resp = await client.get("/databases/pg/dumps", headers=_auth(ADMIN_RAW))
    assert resp.status_code == 200
    assert resp.json()["dumps"] == []


# --- POST /databases/{name}/restore -------------------------------------------


def _write_dump_tar(
    data_dir: Path,
    *,
    service="pg",
    engine="postgres",
    payload=b"PGDMP\x00x",
    dump_format: str | None = None,
):
    """Pack a real dump tar through the shipped packer and return its basename.

    ``dump_format`` overrides the manifest's ``format`` token without touching
    its ``engine`` or ``file`` — the shape of a hand-repacked archive that
    contradicts itself, which is what the third consistency check exists for.
    """
    run_id = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12))
    staging = create_dump_staging_dir(data_dir, run_id)
    filename = "dump.pgdump" if engine == "postgres" else "dump.rdb"
    (staging / filename).write_bytes(payload)
    result = create_dump_backup(
        data_dir,
        service=service,
        engine=engine,
        image=f"{engine}:16",
        staging=staging,
        backend_dump_filename=filename,
        backend_dump_format=dump_format or ("pg_custom" if engine == "postgres" else "rdb"),
        tool_argv0="pg_dump" if engine == "postgres" else "redis-cli",
    )
    return result.basename


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "nerdit-backup-20260903T100000Z.tar.gz",
        "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz.evil",
        "nerdit-dump-PG-20260907T100000Z-abcdef.tar.gz",
    ],
)
async def test_restore_refuses_a_bad_basename_before_any_filesystem_access(
    bad, tmp_path, monkeypatch
):
    """The grammar is a field pattern, so it is checked before the route body runs.

    ``_is_regular_file`` is the route's ONE filesystem touch for a caller-supplied
    name; making it explode proves the refusal happened before it — a stronger
    statement than "the traversal did not escape", because no path was ever
    composed at all.
    """

    def _boom(_path):
        raise AssertionError("the dump basename reached the filesystem")

    monkeypatch.setattr(databases_routes, "_is_regular_file", _boom)
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore", json={"dump": bad}, headers={**_auth(ADMIN_RAW), **_key()}
        )
    assert resp.status_code == 422
    assert controller.restore_calls == []


@pytest.mark.asyncio
async def test_restore_absent_dump_is_404(tmp_path):
    q = _dump_queries()
    controller = _FakeDumpController(tmp_path / "data")
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz"},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 404
    assert resp.json()["code"] == "restore.not_found"
    assert controller.restore_calls == []


@pytest.mark.asyncio
async def test_restore_happy_path_hands_a_populated_staging_dir_to_the_controller(tmp_path):
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir)
    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "service_name": "pg",
        "dump": basename,
        "engine": "postgres",
        "duration_s": body["duration_s"],
    }
    # The route extracted BEFORE handing off, and nothing is left behind.
    assert controller.restore_saw_payload == [True]
    assert not list((data_dir / "dump-staging").iterdir())


@pytest.mark.asyncio
async def test_restore_engine_mismatch_is_422(tmp_path):
    """A Redis dump into a Postgres row is refused before any container."""
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir, engine="redis", payload=b"REDIS0011x")
    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "restore.engine_mismatch"
    assert body["detail"] == {"expected": "postgres", "found": "redis"}
    assert body["reason"] == "engine_mismatch"
    assert controller.restore_calls == []
    assert not list((data_dir / "dump-staging").iterdir())


@pytest.mark.asyncio
async def test_restore_format_mismatch_is_422_manifest_invalid(tmp_path):
    """An archive whose manifest contradicts ITSELF is refused before the container.

    ``engine`` and ``file`` both say Postgres, ``format`` says ``rdb``. Nothing
    on the restore path reads ``format`` — the tool is chosen by the ROW's
    backend — so without this check the archive reaches ``pg_restore`` and the
    schema violation surfaces as an opaque non-zero exit. It is also what makes
    ``DataBackend.dump_format``'s own docstring true.
    """
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir, dump_format="rdb")
    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "restore.manifest_invalid"
    assert body["detail"] == {"reason": "schema"}
    assert body["reason"] == "schema"
    assert controller.restore_calls == []
    assert not list((data_dir / "dump-staging").iterdir())


@pytest.mark.asyncio
async def test_restore_of_a_tampered_archive_is_422_manifest_invalid(tmp_path):
    """An archive that is not what this daemon packed is refused with a reason."""
    data_dir = tmp_path / "data"
    backups = data_dir / "backups"
    backups.mkdir(parents=True)
    basename = "nerdit-dump-pg-20260907T100000Z-abcdef.tar.gz"
    with tarfile.open(backups / basename, "w:gz") as tar:
        for member in ("dump.pgdump", "dump-manifest.json", "extra.txt"):
            info = tarfile.TarInfo(member)
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))

    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "restore.manifest_invalid"
    assert body["detail"]["reason"] == "members"
    assert body["reason"] == "members"
    assert controller.restore_calls == []
    assert not list((data_dir / "dump-staging").iterdir())


@pytest.mark.asyncio
async def test_restore_refuses_while_a_live_app_is_bound_then_force_proceeds(tmp_path):
    """Live is ``running`` OR ``degraded`` — a stopped dependent is not a blocker.

    ``degraded`` is the one that is easy to get wrong: Decision #2 leaves such a
    container RUNNING (it is failing its health check, not stopped), so it holds
    exactly the Postgres locks ``pg_restore --clean`` waits on and the Redis
    connections the quiesce drops. Counting only ``running`` would let the
    guard's own hazard through.
    """
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir)
    q = _dump_queries()
    q.list_workload_configs = AsyncMock(
        return_value=[
            {
                "id": "svc-1",
                "kind": "service",
                "service_name": "webapp",
                "status": "running",
                "config": {"db": {"default": {"provider": "managed", "database": "pg"}}},
            },
            # A STOPPED dependent is deliberately not a blocker: a restore only
            # hurts a client holding connections right now (D-P37-6).
            {
                "id": "svc-2",
                "kind": "service",
                "service_name": "worker",
                "status": "stopped",
                "config": {"db": {"default": {"provider": "managed", "database": "pg"}}},
            },
            {
                "id": "svc-3",
                "kind": "service",
                "service_name": "flaky",
                "status": "degraded",
                "config": {"db": {"default": {"provider": "managed", "database": "pg"}}},
            },
        ]
    )
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        refused = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key("K1")},
        )
        assert refused.status_code == 409
        body = refused.json()
        assert body["code"] == "restore.in_use"
        assert sorted(d["service"] for d in body["dependents"]) == ["flaky", "webapp"]
        assert body["detail"]["dependents"] == body["dependents"]
        # Asserted INSIDE the block, before the forced retry: the guard's whole
        # job is that no container ran, and checking after both requests would
        # only ever see the forced one's call.
        assert controller.restore_calls == []

        forced = await client.post(
            "/databases/pg/restore?force=true",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key("K2")},
        )
    assert forced.status_code == 200, forced.text
    assert len(controller.restore_calls) == 1
    assert not list((data_dir / "dump-staging").iterdir())


@pytest.mark.asyncio
async def test_restore_failure_cleans_up_staging_and_records_the_event(tmp_path, monkeypatch):
    """Every refusal path removes the staging dir the route populated (D-P37-11)."""
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir)
    q = _dump_queries()
    # A slot refusal lands BEFORE the archive is touched: no staging dir is
    # ever created, nothing is extracted, and the controller is never called.
    controller = _FakeDumpController(data_dir, raises=RunPreconditionError("run_in_progress"))
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 409
    assert resp.json()["code"] == "dump.in_progress"
    root = data_dir / "dump-staging"
    assert not root.exists() or not list(root.iterdir())
    assert controller.restore_calls == []
    # A slot refusal is not a restore attempt, so it records no failure event.
    assert events.rows == []

    # The post-extraction failure path still cleans up through the route.
    controller = _FakeDumpController(data_dir, raises=DumpError("interrupted"))
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 500
    assert controller.restore_saw_payload == [True]
    assert not list(root.iterdir())
    assert controller.slot_events == [
        ("reserve", controller.restore_calls[0]),
        ("release", controller.restore_calls[0]),
    ]


@pytest.mark.asyncio
async def test_restore_claims_the_dump_slot_before_extracting(tmp_path):
    """(D-P37-9) The slot is held for the whole operation, from before the
    first byte is extracted until after the controller returns."""
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir)
    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 200, resp.text
    (run_id,) = controller.restore_calls
    assert controller.slot_events == [("reserve", run_id), ("release", run_id)]
    assert controller.staging_empty_at_reserve == [True]

    # Every refusal after the reservation releases it: a bad engine (422).
    basename = _write_dump_tar(data_dir, engine="redis", payload=b"REDIS0011x")
    q = _dump_queries()
    controller = _FakeDumpController(data_dir)
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 422, resp.text
    assert [kind for kind, _ in controller.slot_events] == ["reserve", "release"]


@pytest.mark.asyncio
async def test_restore_tool_failure_is_500_and_records_the_event(tmp_path, monkeypatch):
    events = _RecordedEvents()
    monkeypatch.setattr(eventlog, "_recorder", events)
    data_dir = tmp_path / "data"
    basename = _write_dump_tar(data_dir)
    q = _dump_queries()
    controller = _FakeDumpController(
        data_dir,
        raises=DumpError("quiesce_timeout", log_tail=["DETAIL:  Key (id)=(1) already exists."]),
    )
    async with _client(_dump_app(q, tmp_path, controller=controller)) as client:
        resp = await client.post(
            "/databases/pg/restore",
            json={"dump": basename},
            headers={**_auth(ADMIN_RAW), **_key()},
        )
    assert resp.status_code == 500
    assert resp.json()["code"] == "restore.failed"
    assert resp.json()["reason"] == "quiesce_timeout"
    (row,) = events.rows
    assert row["type"] == "database.restore_failed"
    assert row["reason"] == "quiesce_timeout"
    assert "DETAIL:" not in json.dumps(row)
    assert not list((data_dir / "dump-staging").iterdir())


# --- idempotent replay --------------------------------------------------------


@pytest.mark.asyncio
async def test_dump_idempotent_replay_returns_the_same_basename_without_a_second_container(
    tmp_path, _no_disk_pressure
):
    """The key pins the OUTCOME, not the act: a retry must not dump twice."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    try:
        job = _ready_db_job(submitted_by_token=None)
        await queries.create_job(job)
        await queries.acquire_service_port("pg", job.id, 9500, (9400, 9599))

        controller = _FakeDumpController(tmp_path / "data")
        app = _dump_app(queries, tmp_path, controller=controller, with_idempotency=True)
        # A tokenless local admin: the middleware's LOCAL principal, so the
        # owner gate passes without an api_tokens row to seed.
        app.user_middleware = [
            m for m in app.user_middleware if m.cls is not ScopedTokenAuthMiddleware
        ]
        app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
        app.middleware_stack = app.build_middleware_stack()

        async with _client(app) as client:
            first = await client.post("/databases/pg/dump", json={}, headers=_key("K-replay"))
            second = await client.post("/databases/pg/dump", json={}, headers=_key("K-replay"))

        assert first.status_code == 200, first.text
        assert second.status_code == 200
        assert second.headers.get("Idempotent-Replay") == "true"
        assert second.json()["dump"] == first.json()["dump"]
        # One container, one tar.
        assert len(controller.dump_calls) == 1
        assert len(list((tmp_path / "data" / "backups").glob("nerdit-dump-*.tar.gz"))) == 1
    finally:
        await db.close()
