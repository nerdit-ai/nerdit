"""Tests for the consolidated P1/S1 schema migration.

Covers three things:
1. A fresh database has every new P1 table/column.
2. A *legacy* database (old NOT NULL ``script_path`` ``jobs`` table without the
   new columns) survives ``init_schema``/``_migrate`` — in particular the
   ``submitted_by_token`` and ``idempotency_key`` columns survive the
   ``jobs_new`` rebuild in ``_ensure_nullable_script_path`` (the dual-literal
   hazard guard).
3. A ``Job`` round-trips with both new fields populated.
"""

from __future__ import annotations

import pytest

from nerdit.db.database import Database
from nerdit.db.models import Job, JobKind, JobStatus
from nerdit.db.queries import Queries


async def _jobs_columns(db: Database) -> set[str]:
    cursor = await db.conn.execute("PRAGMA table_info(jobs)")
    return {row[1] for row in await cursor.fetchall()}


async def _table_names(db: Database) -> set[str]:
    cursor = await db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in await cursor.fetchall()}


async def _index_names(db: Database) -> set[str]:
    cursor = await db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    return {row[0] for row in await cursor.fetchall()}


@pytest.mark.asyncio
async def test_fresh_db_has_new_job_columns(db):
    cols = await _jobs_columns(db)
    assert "submitted_by_token" in cols
    assert "idempotency_key" in cols
    # P0 column still present (no regression on the precedent).
    assert "kind" in cols


@pytest.mark.asyncio
async def test_fresh_db_has_new_tables_and_indexes(db):
    tables = await _table_names(db)
    assert {"api_tokens", "idempotency_keys", "audit_log"} <= tables
    indexes = await _index_names(db)
    assert {"idx_api_tokens_hash", "idx_idem_expires", "idx_audit_ts"} <= indexes


@pytest.mark.asyncio
async def test_api_tokens_table_shape(db):
    cursor = await db.conn.execute("PRAGMA table_info(api_tokens)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {
        "id",
        "name",
        "role",
        "token_hash",
        "max_gpus",
        "max_concurrent_jobs",
        "created_at",
        "last_used_at",
        "revoked",
        # P25 D-P25-1/D-P25-3 — expiry + scope ride the same row.
        "expires_at",
        "scope_services",
    } <= cols


@pytest.mark.asyncio
async def test_idempotency_keys_table_shape(db):
    cursor = await db.conn.execute("PRAGMA table_info(idempotency_keys)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {
        "principal_id",
        "idem_key",
        "method",
        "path",
        "state",
        "response_status",
        "response_body",
        "content_type",
        "resource_id",
        "body_hash",
        "created_at",
        "expires_at",
    } <= cols


@pytest.mark.asyncio
async def test_audit_log_table_shape(db):
    cursor = await db.conn.execute("PRAGMA table_info(audit_log)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {
        "id",
        "ts",
        "principal_id",
        "principal_role",
        "action",
        "target_type",
        "target_id",
        "params_redacted",
        "result",
        "status_code",
        "request_id",
        "idempotency_key",
    } <= cols


# --- P2 service columns / tables / indexes ---


# The 7 service columns added to ``jobs`` in P2/S1 (exact set).
_SERVICE_JOB_COLS = {
    "desired_state",
    "restart_policy",
    "restart_count",
    "health_check",
    "service_name",
    "last_exit_at",
    "restart_window_start",
}


@pytest.mark.asyncio
async def test_fresh_db_has_service_columns(db):
    cols = await _jobs_columns(db)
    assert _SERVICE_JOB_COLS <= cols
    # P1 columns still present (no regression on the precedent).
    assert {"kind", "submitted_by_token", "idempotency_key"} <= cols


@pytest.mark.asyncio
async def test_fresh_db_has_service_endpoints_table_and_indexes(db):
    tables = await _table_names(db)
    assert "service_endpoints" in tables
    indexes = await _index_names(db)
    assert {"idx_service_endpoints_name", "idx_jobs_service_name"} <= indexes


@pytest.mark.asyncio
async def test_service_endpoints_table_shape(db):
    cursor = await db.conn.execute("PRAGMA table_info(service_endpoints)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {
        "id",
        "service_name",
        "job_id",
        "container_port",
        "host_port",
        "protocol",
        "route",
        "created_at",
    } <= cols


@pytest.mark.asyncio
async def test_service_name_index_is_partial(db):
    """idx_jobs_service_name must be a partial unique index (WHERE … IS NOT NULL).

    Without the partial predicate, multiple batch rows (all ``service_name``
    NULL) would still be allowed by SQLite, but the index must stay scoped to
    service rows. Two service rows sharing a name must collide; many batch rows
    with NULL ``service_name`` must not.
    """
    sql = await db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_jobs_service_name'"
    )
    row = await sql.fetchone()
    assert row is not None
    assert "WHERE service_name IS NOT NULL" in row[0]


# --- Legacy-DB upgrade: the dual-literal rebuild guard ---


# A pre-P1 ``jobs`` table: NOT NULL ``script_path`` (forces the ``jobs_new``
# rebuild) and *without* the new columns (kind/submitted_by_token/idempotency).
_LEGACY_JOBS_DDL = """
CREATE TABLE jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    script_path TEXT NOT NULL,
    gpu_count   INTEGER DEFAULT 1,
    priority    INTEGER DEFAULT 5,
    status      TEXT DEFAULT 'pending',
    container_id TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    started_at  TEXT,
    finished_at TEXT,
    paused_at   TEXT,
    exit_code   INTEGER,
    retries     INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    config      TEXT,
    tags        TEXT,
    preemptible INTEGER DEFAULT 0,
    max_runtime_seconds INTEGER,
    time_window TEXT,
    error_class TEXT,
    error_message TEXT,
    submitted_via TEXT NOT NULL DEFAULT 'cli',
    template_id TEXT
)
"""


@pytest.mark.asyncio
async def test_legacy_db_upgrade_retains_new_columns():
    """A legacy jobs table → init_schema must keep the new columns after rebuild.

    This guards the dual-literal hazard: the ``CREATE TABLE jobs_new`` literal
    and the ``INSERT INTO jobs_new SELECT`` list in
    ``_ensure_nullable_script_path`` must both carry ``submitted_by_token`` and
    ``idempotency_key`` in identical positional order after ``kind``.
    """
    db = Database(":memory:")
    await db.connect()
    # Stand up the legacy schema by hand, before any P1 migration runs.
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.execute(
        "INSERT INTO jobs (id, name, script_path, gpu_count, status) "
        "VALUES ('legacyjob001', 'old', 'train.py', 2, 'completed')"
    )
    await db.conn.commit()

    # Run the real schema init + migrations (executescript is IF NOT EXISTS, so
    # the legacy jobs table is preserved and then migrated/rebuilt).
    await db.init_schema()

    cols = await _jobs_columns(db)
    assert "submitted_by_token" in cols, "column dropped during jobs_new rebuild"
    assert "idempotency_key" in cols, "column dropped during jobs_new rebuild"
    assert "kind" in cols

    # script_path is now nullable (rebuild happened) and the legacy row survived.
    queries = Queries(db)
    job = await queries.get_job("legacyjob001")
    assert job is not None
    assert job.name == "old"
    assert job.gpu_count == 2
    assert job.submitted_by_token is None
    assert job.idempotency_key is None

    # New tables created on a legacy DB too.
    tables = await _table_names(db)
    assert {"api_tokens", "idempotency_keys", "audit_log", "service_endpoints"} <= tables
    await db.close()


@pytest.mark.asyncio
async def test_legacy_db_upgrade_retains_service_columns():
    """The 7 P2 service columns survive the dual-literal ``jobs_new`` rebuild.

    Same hazard as the P1 columns: both the ``CREATE TABLE jobs_new`` literal
    and the ``INSERT … SELECT`` list must carry ``desired_state`` …
    ``restart_window_start`` in identical positional order.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.execute(
        "INSERT INTO jobs (id, name, script_path, gpu_count, status) "
        "VALUES ('legacyjob002', 'old', 'train.py', 1, 'completed')"
    )
    await db.conn.commit()

    await db.init_schema()

    cols = await _jobs_columns(db)
    assert _SERVICE_JOB_COLS <= cols, "a service column was dropped during jobs_new rebuild"

    # restart_count carries its NOT NULL DEFAULT 0 onto the migrated legacy row.
    queries = Queries(db)
    job = await queries.get_job("legacyjob002")
    assert job is not None
    assert job.restart_count == 0
    assert job.service_name is None
    await db.close()


@pytest.mark.asyncio
async def test_service_name_index_survives_legacy_rebuild():
    """idx_jobs_service_name is created AFTER ``_ensure_nullable_script_path``.

    The rebuild does ``DROP TABLE jobs``; an index declared in ``_SCHEMA`` would
    be dropped with it. Creating it at the end of ``_migrate()`` means it must
    still exist after a legacy upgrade that triggers the rebuild.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.commit()

    await db.init_schema()

    indexes = await _index_names(db)
    assert "idx_jobs_service_name" in indexes, "index dropped by the jobs_new DROP TABLE rebuild"
    await db.close()


# A pre-P22 ``idempotency_keys`` table: everything except ``body_hash``.
_LEGACY_IDEMPOTENCY_DDL = """
CREATE TABLE idempotency_keys (
    principal_id    TEXT NOT NULL,
    idem_key        TEXT NOT NULL,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'in_progress',
    response_status INTEGER,
    response_body   TEXT,
    content_type    TEXT,
    resource_id     TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    expires_at      TEXT,
    PRIMARY KEY (principal_id, idem_key)
)
"""


@pytest.mark.asyncio
async def test_legacy_db_gains_body_hash_column():
    """P22: an existing install must GAIN ``body_hash`` via ``_migrate``.

    ``_SCHEMA`` is ``CREATE TABLE IF NOT EXISTS``, so editing the literal alone
    silently no-ops on every database that already has the table — the ALTER
    block is what actually ships the column. The legacy row must survive with a
    NULL digest (which the soft comparison reads as "no opinion", so it keeps
    replaying).
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_IDEMPOTENCY_DDL)
    await db.conn.execute(
        "INSERT INTO idempotency_keys (principal_id, idem_key, method, path, state) "
        "VALUES ('anon:local', 'legacy-key', 'POST', '/services', 'completed')"
    )
    await db.conn.commit()

    await db.init_schema()

    cursor = await db.conn.execute("PRAGMA table_info(idempotency_keys)")
    assert "body_hash" in {row[1] for row in await cursor.fetchall()}

    record = await Queries(db).get_idempotency_record("anon:local", "legacy-key")
    assert record is not None
    assert record.state == "completed"
    assert record.body_hash is None
    await db.close()


@pytest.mark.asyncio
async def test_body_hash_migration_is_idempotent():
    """``init_schema`` runs on every boot: the ALTER must be presence-checked."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    await db.init_schema()

    cursor = await db.conn.execute("PRAGMA table_info(idempotency_keys)")
    cols = [row[1] for row in await cursor.fetchall()]
    assert cols.count("body_hash") == 1
    await db.close()


# --- P24a durable event feed ---


@pytest.mark.asyncio
async def test_events_table_shape(db):
    """The D-P24-1 columns, exactly as locked."""
    cursor = await db.conn.execute("PRAGMA table_info(events)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {
        "id",
        "ts",
        "type",
        "kind",
        "service_name",
        "reason",
        "build_version",
        "data",
    } <= cols


@pytest.mark.asyncio
async def test_events_id_is_autoincrement_and_ts_defaults(db, queries):
    """The monotonic id IS the cursor; ``ts`` comes from the column default."""
    first = await queries.insert_event(type="daemon.started")
    second = await queries.insert_event(type="daemon.stopping")
    assert second > first

    cursor = await db.conn.execute("SELECT ts FROM events WHERE id = ?", (first,))
    assert (await cursor.fetchone())[0]


@pytest.mark.asyncio
async def test_events_table_and_indexes_appear_on_a_legacy_db():
    """A table added only to ``_SCHEMA`` IS created on every existing install.

    ``executescript`` runs the whole ``CREATE TABLE IF NOT EXISTS`` literal on
    every boot, which is why D-P24-1 needs no ``_migrate`` block — but that is
    exactly the kind of claim that is worth failing loudly if it stops being
    true, since the feed would then simply never exist on an upgraded daemon.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.execute(
        "INSERT INTO jobs (id, name, script_path, gpu_count, status) "
        "VALUES ('legacyjob003', 'old', 'train.py', 1, 'completed')"
    )
    await db.conn.commit()

    await db.init_schema()

    assert "events" in await _table_names(db)
    assert {"idx_events_service", "idx_events_type"} <= await _index_names(db)
    # The legacy row survived the jobs rebuild that ran in the same pass.
    assert await Queries(db).get_job("legacyjob003") is not None
    await db.close()


@pytest.mark.asyncio
async def test_events_indexes_survive_the_legacy_jobs_rebuild():
    """D-P24-1's safety argument: only ``jobs`` is ever rebuilt.

    The ``jobs`` indexes are (deliberately) not in ``_SCHEMA`` because the
    rebuild drops them; the ``events`` ones are, and must still be there after a
    boot that performed the rebuild.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.commit()

    await db.init_schema()
    await db.init_schema()  # idempotent: a second boot must not duplicate either

    indexes = [
        name
        for name in (
            row[0]
            for row in await (
                await db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
            ).fetchall()
        )
        if name.startswith("idx_events_")
    ]
    assert sorted(indexes) == ["idx_events_service", "idx_events_type"]
    await db.close()


@pytest.mark.asyncio
async def test_job_logs_keyset_index_appears_on_a_legacy_db():
    """``idx_job_logs_job`` rides ``_SCHEMA`` for the same reason the events ones do.

    ``job_logs`` is never rebuilt, so no ``DROP TABLE`` can remove it. It backs
    every per-job keyset read (``WHERE job_id = ? AND id > ? ORDER BY id``) —
    the paged read, ``/diagnose``, and the P24a log stream's 1 s poll, which
    without it walks other jobs' rows on every tick.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.commit()

    await db.init_schema()
    await db.init_schema()  # idempotent

    names = await _index_names(db)
    assert {"idx_job_logs_job", "idx_job_logs_ts"} <= names

    cursor = await db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_job_logs_job'"
    )
    ddl = (await cursor.fetchone())[0]
    assert "job_logs(job_id, id)" in ddl.replace('"', "")
    await db.close()


@pytest.mark.asyncio
async def test_job_status_new_service_states():
    """The 4 P2 service states are constructible (``_row_to_job`` builds them)."""
    assert JobStatus("building") is JobStatus.building
    assert JobStatus("degraded") is JobStatus.degraded
    assert JobStatus("restarting") is JobStatus.restarting
    assert JobStatus("stopped") is JobStatus.stopped


@pytest.mark.asyncio
async def test_job_model_roundtrip_with_service_fields():
    """A service-shaped ``Job`` round-trips through model dump/validate.

    Exercises ``gpu_count=0`` (relaxed to ``ge=0`` on the Job model only — a
    service may request no GPUs) and the new service fields.
    """
    job = Job(
        name="svc",
        gpu_count=0,
        kind="service",
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        restart_count=2,
        health_check={"path": "/healthz", "interval_seconds": 5},
        service_name="my-svc",
    )
    restored = Job.model_validate(job.model_dump())
    assert restored.gpu_count == 0
    assert restored.status is JobStatus.building
    assert restored.desired_state == "running"
    assert restored.restart_policy == "always"
    assert restored.restart_count == 2
    assert restored.health_check == {"path": "/healthz", "interval_seconds": 5}
    assert restored.service_name == "my-svc"


@pytest.mark.asyncio
async def test_job_roundtrip_with_new_fields(queries):
    job = Job(
        script_path="train.py",
        gpu_count=1,
        submitted_by_token="tok123456789",
        idempotency_key="idem-key-abc",
    )
    await queries.create_job(job)
    fetched = await queries.get_job(job.id)
    assert fetched is not None
    assert fetched.submitted_by_token == "tok123456789"
    assert fetched.idempotency_key == "idem-key-abc"


@pytest.mark.asyncio
async def test_job_roundtrip_defaults_null(queries):
    job = Job(script_path="train.py")
    await queries.create_job(job)
    fetched = await queries.get_job(job.id)
    assert fetched is not None
    assert fetched.submitted_by_token is None
    assert fetched.idempotency_key is None


# A pre-P24b ``service_endpoints`` table: everything except ``active_host_port``.
_LEGACY_ENDPOINTS_DDL = """
CREATE TABLE service_endpoints (
    id             TEXT PRIMARY KEY,
    service_name   TEXT NOT NULL,
    job_id         TEXT,
    container_port INTEGER NOT NULL,
    host_port      INTEGER NOT NULL UNIQUE,
    protocol       TEXT DEFAULT 'tcp',
    route          TEXT,
    created_at     TEXT DEFAULT (datetime('now'))
)
"""


@pytest.mark.asyncio
async def test_legacy_db_gains_active_host_port_column():
    """P24b: an existing install must GAIN ``active_host_port`` via ``_migrate``.

    Same hazard as the P22 ``body_hash`` column: ``_SCHEMA`` is
    ``CREATE TABLE IF NOT EXISTS``, so the literal edit alone silently no-ops
    on every database that already has the table. NULL on the legacy row is the
    "same as host_port" reading, which is exactly what a pre-cutover row means.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_ENDPOINTS_DDL)
    await db.conn.execute(
        "INSERT INTO service_endpoints (id, service_name, job_id, container_port, host_port) "
        "VALUES ('ep0000000001', 'legacy-svc', NULL, 8000, 9401)"
    )
    await db.conn.commit()

    await db.init_schema()

    cursor = await db.conn.execute("PRAGMA table_info(service_endpoints)")
    assert "active_host_port" in {row[1] for row in await cursor.fetchall()}

    endpoint = await Queries(db).get_service_endpoint("legacy-svc")
    assert endpoint is not None
    assert endpoint.active_host_port is None
    assert endpoint.live_port == 9401  # NULL reads as "same as host_port"
    await db.close()


@pytest.mark.asyncio
async def test_active_route_coalesce_falls_back_to_host_port_on_a_null_row(queries):
    """The ONE COALESCE: a NULL ``active_host_port`` yields the stable port."""
    job = Job(
        name="svc",
        kind=JobKind.service,
        service_name="svc",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config='{"image": "x", "port": 8000}',
    )
    await queries.create_job(job)
    endpoint = await queries.acquire_service_port("svc", job.id, 8000, (9400, 9499))

    routes = await queries.list_active_service_routes()
    assert [r.host_port for r in routes] == [endpoint.host_port]

    await queries.set_endpoint_active_port("svc", 54321)
    routes = await queries.list_active_service_routes()
    assert [r.host_port for r in routes] == [54321]

    await queries.set_endpoint_active_port("svc", None)
    routes = await queries.list_active_service_routes()
    assert [r.host_port for r in routes] == [endpoint.host_port]


# --- P24c webhook delivery cursors ---


@pytest.mark.asyncio
async def test_notification_cursors_table_shape(db):
    """The WP8 columns, exactly as locked."""
    cursor = await db.conn.execute("PRAGMA table_info(notification_cursors)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert cols == {"target_id", "last_id", "updated_at"}


@pytest.mark.asyncio
async def test_notification_cursors_appear_on_a_legacy_db():
    """A table added only to ``_SCHEMA`` IS created on every existing install.

    The ``events`` argument verbatim (D-BP-7): ``executescript`` runs the whole
    ``CREATE TABLE IF NOT EXISTS`` literal on every boot, which is why this
    table needs no ``_migrate`` block — but an upgraded daemon whose cursors
    table never appeared would simply stop delivering, silently.
    """
    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_JOBS_DDL)
    await db.conn.commit()

    await db.init_schema()
    await db.init_schema()  # idempotent: a second boot must not duplicate it

    tables = [name for name in await _table_names(db) if name == "notification_cursors"]
    assert tables == ["notification_cursors"]
    await db.close()


@pytest.mark.asyncio
async def test_notification_cursor_upsert_round_trip(queries):
    """Absent ⇒ ``None`` (the seed signal); the writer upserts in place."""
    assert await queries.get_notification_cursor("deadbeefdeadbeef") is None

    await queries.set_notification_cursor("deadbeefdeadbeef", 7)
    assert await queries.get_notification_cursor("deadbeefdeadbeef") == 7

    await queries.set_notification_cursor("deadbeefdeadbeef", 42)
    assert await queries.get_notification_cursor("deadbeefdeadbeef") == 42


# --- P25 token lifecycle (expires_at + scope_services) ---

# A pre-P25 ``api_tokens`` table: the nine original columns, no expiry, no scope.
_LEGACY_API_TOKENS_DDL = """
CREATE TABLE api_tokens (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    role                TEXT NOT NULL DEFAULT 'submitter',
    token_hash          TEXT NOT NULL UNIQUE,
    max_gpus            INTEGER,
    max_concurrent_jobs INTEGER,
    created_at          TEXT DEFAULT (datetime('now')),
    last_used_at        TEXT,
    revoked             INTEGER NOT NULL DEFAULT 0
)
"""


@pytest.mark.asyncio
async def test_legacy_db_gains_token_lifecycle_columns():
    """P25: an existing install must GAIN both columns via ``_migrate``.

    Same hazard as the P22 ``body_hash`` and P24b ``active_host_port`` columns:
    ``_SCHEMA`` is ``CREATE TABLE IF NOT EXISTS``, so the literal edit alone
    silently no-ops on every database that already has the table. NULL on the
    legacy row is the anti-bricking reading (D-P25-1): never expires, unscoped.
    """
    from nerdit.daemon.auth import hash_token

    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_API_TOKENS_DDL)
    await db.conn.execute(
        "INSERT INTO api_tokens (id, name, role, token_hash, created_at) "
        "VALUES ('tok000000001', 'legacy-ci', 'submitter', ?, '2026-01-01T00:00:00+00:00')",
        (hash_token("legacy-raw"),),
    )
    await db.conn.commit()

    await db.init_schema()

    cursor = await db.conn.execute("PRAGMA table_info(api_tokens)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert {"expires_at", "scope_services"} <= cols

    token = await Queries(db).get_api_token_by_id("tok000000001")
    assert token is not None
    assert token.expires_at is None  # never expires
    assert token.scope_services is None  # unscoped
    await db.close()


@pytest.mark.asyncio
async def test_legacy_token_still_authenticates_after_migration():
    """The anti-bricking gate: a pre-P25 token keeps resolving post-upgrade."""
    from nerdit.daemon.auth import hash_token

    db = Database(":memory:")
    await db.connect()
    await db.conn.execute(_LEGACY_API_TOKENS_DDL)
    await db.conn.execute(
        "INSERT INTO api_tokens (id, name, role, token_hash, created_at) "
        "VALUES ('tok000000002', 'legacy-ci', 'admin', ?, '2026-01-01T00:00:00+00:00')",
        (hash_token("legacy-raw-2"),),
    )
    await db.conn.commit()

    await db.init_schema()

    resolved = await Queries(db).get_api_token_by_hash(hash_token("legacy-raw-2"))
    assert resolved is not None
    assert resolved.id == "tok000000002"
    assert resolved.expires_at is None
    await db.close()


@pytest.mark.asyncio
async def test_token_lifecycle_migration_is_idempotent():
    """``init_schema`` runs on every boot: both ALTERs must be presence-checked."""
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    await db.init_schema()

    cursor = await db.conn.execute("PRAGMA table_info(api_tokens)")
    cols = [row[1] for row in await cursor.fetchall()]
    assert cols.count("expires_at") == 1
    assert cols.count("scope_services") == 1
    await db.close()


@pytest.mark.asyncio
async def test_token_lifecycle_columns_round_trip(queries):
    """Both columns survive a create/read cycle with their typed shapes."""
    from datetime import UTC, datetime

    from nerdit.db.models import ApiToken

    expiry = datetime(2030, 6, 1, 12, 0, tzinfo=UTC)
    await queries.create_api_token(
        ApiToken(
            id="tok000000003",
            name="scoped",
            token_hash="hash-3",
            expires_at=expiry,
            scope_services=["api", "worker"],
        )
    )

    token = await queries.get_api_token_by_id("tok000000003")
    assert token is not None
    assert token.expires_at == expiry
    assert token.scope_services == ["api", "worker"]


@pytest.mark.asyncio
async def test_garbage_scope_column_decodes_to_an_empty_scope(queries):
    """(P25 D-P25-3) Fail closed: an unparsable scope grants NOTHING, not everything.

    ``None`` is the ONLY spelling of "unscoped"; a corrupted or hand-edited
    column must never widen a token's authority.
    """
    from nerdit.db.models import ApiToken

    await queries.create_api_token(ApiToken(id="tok000000004", name="x", token_hash="hash-4"))
    await queries._db.conn.execute(
        "UPDATE api_tokens SET scope_services = ? WHERE id = 'tok000000004'",
        ("{not json at all",),
    )
    await queries._db.conn.commit()

    token = await queries.get_api_token_by_id("tok000000004")
    assert token is not None
    assert token.scope_services == []


@pytest.mark.asyncio
async def test_non_array_and_non_string_scope_entries_fail_closed(queries):
    """A JSON object, and a list with non-string members, both decode narrowly."""
    from nerdit.db.models import ApiToken

    await queries.create_api_token(ApiToken(id="tok000000005", name="x", token_hash="hash-5"))
    await queries._db.conn.execute(
        "UPDATE api_tokens SET scope_services = ? WHERE id = 'tok000000005'", ('{"a": 1}',)
    )
    await queries._db.conn.commit()
    token = await queries.get_api_token_by_id("tok000000005")
    assert token is not None and token.scope_services == []

    await queries._db.conn.execute(
        "UPDATE api_tokens SET scope_services = ? WHERE id = 'tok000000005'", ('["api", 7, null]',)
    )
    await queries._db.conn.commit()
    token = await queries.get_api_token_by_id("tok000000005")
    assert token is not None and token.scope_services == ["api"]


@pytest.mark.asyncio
async def test_naive_expires_at_column_is_coerced_to_utc(queries):
    """A hand-edited tz-naive timestamp must not TypeError in the auth middleware."""
    from datetime import UTC

    from nerdit.db.models import ApiToken

    await queries.create_api_token(ApiToken(id="tok000000006", name="x", token_hash="hash-6"))
    await queries._db.conn.execute(
        "UPDATE api_tokens SET expires_at = ? WHERE id = 'tok000000006'", ("2030-06-01T12:00:00",)
    )
    await queries._db.conn.commit()

    token = await queries.get_api_token_by_id("tok000000006")
    assert token is not None
    assert token.expires_at is not None
    assert token.expires_at.tzinfo is not None
    assert token.expires_at.utcoffset() == UTC.utcoffset(None)
