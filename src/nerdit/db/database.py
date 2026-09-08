"""SQLite database connection and schema management."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gpus (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    memory_mb         INTEGER NOT NULL,
    compute_cap       TEXT,
    vendor            TEXT NOT NULL DEFAULT 'nvidia',
    device_index      INTEGER,
    runtime_id        TEXT,
    discovery_backend TEXT NOT NULL DEFAULT 'nvml',
    schedulable       INTEGER NOT NULL DEFAULT 1,
    status            TEXT DEFAULT 'idle'
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    script_path TEXT,
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
    template_id TEXT,
    kind        TEXT NOT NULL DEFAULT 'batch',
    submitted_by_token TEXT,
    idempotency_key TEXT,
    desired_state TEXT,
    restart_policy TEXT,
    restart_count INTEGER NOT NULL DEFAULT 0,
    health_check TEXT,
    service_name TEXT,
    last_exit_at TEXT,
    restart_window_start TEXT
);

CREATE TABLE IF NOT EXISTS gpu_allocations (
    job_id    TEXT REFERENCES jobs(id),
    gpu_id    TEXT REFERENCES gpus(id),
    exclusive INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (job_id, gpu_id)
);

CREATE TABLE IF NOT EXISTS job_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT REFERENCES jobs(id),
    stream     TEXT DEFAULT 'stdout',
    message    TEXT,
    timestamp  TEXT DEFAULT (datetime('now'))
);

-- Backs the P14b retention sweep's ``timestamp < cutoff`` predicate. Safe in
-- ``_SCHEMA`` (unlike the jobs-table indexes): ``job_logs`` is never rebuilt,
-- so no ``DROP TABLE`` removes it.
CREATE INDEX IF NOT EXISTS idx_job_logs_ts ON job_logs(timestamp);

-- Backs every per-job keyset read (``WHERE job_id = ? AND id > ? ORDER BY id``)
-- — the paged log read, the ``/diagnose`` tail, and the P24a log stream's
-- 1 s poll. Without it a poll on a busy daemon walks *other* jobs' rows.
-- Same ``_SCHEMA`` safety as ``idx_job_logs_ts``.
CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id, id);

CREATE TABLE IF NOT EXISTS file_uploads (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    uploaded_at  TEXT DEFAULT (datetime('now')),
    used_by_jobs TEXT
);

CREATE TABLE IF NOT EXISTS api_tokens (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    role                TEXT NOT NULL DEFAULT 'submitter',
    token_hash          TEXT NOT NULL UNIQUE,
    max_gpus            INTEGER,
    max_concurrent_jobs INTEGER,
    created_at          TEXT DEFAULT (datetime('now')),
    last_used_at        TEXT,
    revoked             INTEGER NOT NULL DEFAULT 0,
    expires_at          TEXT,
    scope_services      TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_tokens_hash ON api_tokens(token_hash);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    principal_id    TEXT NOT NULL,
    idem_key        TEXT NOT NULL,
    method          TEXT NOT NULL,
    path            TEXT NOT NULL,
    state           TEXT NOT NULL DEFAULT 'in_progress',
    response_status INTEGER,
    response_body   TEXT,
    content_type    TEXT,
    resource_id     TEXT,
    body_hash       TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    expires_at      TEXT,
    PRIMARY KEY (principal_id, idem_key)
);

CREATE INDEX IF NOT EXISTS idx_idem_expires ON idempotency_keys(expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT DEFAULT (datetime('now')),
    principal_id    TEXT,
    principal_role  TEXT,
    action          TEXT NOT NULL,
    target_type     TEXT,
    target_id       TEXT,
    params_redacted TEXT,
    result          TEXT,
    status_code     INTEGER,
    request_id      TEXT,
    idempotency_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_id);

-- "Everything token X did" is the second question an operator asks
-- after "everything that happened to service Y", and it was a full scan. The
-- trailing ``id`` makes the index cover the ORDER BY id DESC / cursor walk too,
-- so a filtered page is a range scan rather than a scan-then-sort.
CREATE INDEX IF NOT EXISTS idx_audit_principal ON audit_log(principal_id, id);

CREATE TABLE IF NOT EXISTS service_endpoints (
    id             TEXT PRIMARY KEY,
    service_name   TEXT NOT NULL,
    job_id         TEXT REFERENCES jobs(id),
    container_port INTEGER NOT NULL,
    host_port      INTEGER NOT NULL UNIQUE,
    protocol       TEXT DEFAULT 'tcp',
    route          TEXT,
    -- (P24b / D-P24-4b) The port the CURRENTLY SERVING container is bound to.
    -- NULL => "same as host_port", the value on every pre-P24 row and after
    -- every ordinary launch. Deliberately NOT UNIQUE and NOT range-allocated:
    -- ``host_port`` keeps its meaning as the reserved, stable, UNIQUE identity,
    -- and this is a transient pointer at the green container's ephemeral port
    -- for the length of a cutover window.
    active_host_port INTEGER,
    created_at     TEXT DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_service_endpoints_name
    ON service_endpoints(service_name);

-- Durable event feed (P24a / D-P24-1). A whole new table, so no ``_migrate``
-- block is needed: ``executescript`` creates it on every existing install too.
-- Safe in ``_SCHEMA`` for the same reason ``job_logs`` is — only ``jobs`` is
-- ever rebuilt (``_ensure_nullable_script_path``), so no ``DROP TABLE`` can
-- remove these indexes. The monotonic ``INTEGER PRIMARY KEY AUTOINCREMENT``
-- **is** the cursor (the ``list_audit_log`` id-keyset pattern).
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT DEFAULT (datetime('now')),
    type          TEXT NOT NULL,
    kind          TEXT,
    service_name  TEXT,
    reason        TEXT,
    build_version INTEGER,
    data          TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_service ON events(service_name, id);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, id);

-- Per-target webhook delivery cursors. A wholly new table, so it
-- rides ``_SCHEMA`` unconditionally for the same reason ``events`` does (see
-- the note above) — no ``_migrate`` block. ``target_id`` is a 16-hex canonical
-- hash of the target's delivery identity (``NotificationTarget.cursor_id``):
-- the URL itself (which may carry a topic token) never enters the database.
CREATE TABLE IF NOT EXISTS notification_cursors (
    target_id  TEXT PRIMARY KEY,
    last_id    INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Hosted-share state (P26 D-P26-H1): one row per shared service. The daemon
-- is the source of truth for WHETHER and HOW a service is shared; the cloud
-- edge decides WHO may reach it. No FK: ``jobs.service_name`` is only
-- partially unique (see the index note below), so an FK to it raises
-- ``foreign key mismatch`` -- the ``service_endpoints`` posture. The row is
-- deleted inside ``delete_service_checked``'s transaction instead. A wholly
-- new table, so it rides ``_SCHEMA`` unconditionally for the same reason
-- ``events``/``notification_cursors`` do -- no ``_migrate`` block.
CREATE TABLE IF NOT EXISTS service_shares (
    service_name TEXT PRIMARY KEY,
    access       TEXT NOT NULL CHECK (access IN ('private', 'public')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Custom domains (P26 D-P26-1): one row per (domain -> service). The table is
-- the ONLY truth -- there is no ``[deploy].domains`` mirror (two writers would
-- resurrect a removed domain, and a string list cannot carry ``acme``). No FK,
-- for the ``service_shares`` reason above (the partial unique index on
-- ``jobs.service_name``); the cascade lives in ``delete_service_checked``. An
-- FK to ``service_endpoints`` was rejected too: the CRIT-4 port reallocation
-- deletes and recreates that row and would take the domains with it. ``domain``
-- is stored case-folded by the write path; ``COLLATE NOCASE`` is the backstop
-- that keeps a stray case variant from claiming the same name twice. ``acme``
-- is dormant DATA in WP1 (every write path forces it to 0; WP2 acts on it) and
-- ``kind`` is a dormant discriminator for a future hosted-CNAME kind. A wholly
-- new table, so it rides ``_SCHEMA`` unconditionally -- no ``_migrate`` block.
CREATE TABLE IF NOT EXISTS service_domains (
    domain       TEXT PRIMARY KEY COLLATE NOCASE,
    service_name TEXT NOT NULL,
    acme         INTEGER NOT NULL DEFAULT 0 CHECK (acme IN (0, 1)),
    kind         TEXT NOT NULL DEFAULT 'domain',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_service_domains_service ON service_domains(service_name);
"""


class Database:
    """Async SQLite database wrapper."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        # Lock whole write transactions, not individual statements: the shared connection
        # otherwise lets a foreign commit break quota/idempotency atomicity. Reads stay unlocked.
        self._write_lock = asyncio.Lock()

    @property
    def write_lock(self) -> asyncio.Lock:
        """The single-writer lock; acquired by every mutating query."""
        return self._write_lock

    async def connect(self) -> None:
        """Open the database connection and configure pragmas."""
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

    async def init_schema(self) -> None:
        """Create tables if they don't exist, then apply migrations."""
        assert self._conn is not None, "Database not connected"
        await self._conn.executescript(_SCHEMA)
        await self._migrate()
        await self._conn.execute("PRAGMA user_version = 1")
        await self._conn.commit()

    @property
    def path(self) -> str:
        """Filesystem path of the database ('' for :memory:)."""
        return self._db_path

    @property
    def conn(self) -> aiosqlite.Connection:
        assert self._conn is not None, "Database not connected"
        return self._conn

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _migrate(self) -> None:
        """Add columns introduced after the initial schema if missing."""
        assert self._conn is not None
        jobs_table_info = await self._get_jobs_table_info()
        existing = {row[1] for row in jobs_table_info}
        migrations = [
            ("tags", "TEXT"),
            ("preemptible", "INTEGER DEFAULT 0"),
            ("max_runtime_seconds", "INTEGER"),
            ("time_window", "TEXT"),
            ("paused_at", "TEXT"),
            ("error_class", "TEXT"),
            ("error_message", "TEXT"),
            ("submitted_via", "TEXT NOT NULL DEFAULT 'cli'"),
            ("template_id", "TEXT"),
            ("kind", "TEXT NOT NULL DEFAULT 'batch'"),
            ("submitted_by_token", "TEXT"),
            ("idempotency_key", "TEXT"),
            ("desired_state", "TEXT"),
            ("restart_policy", "TEXT"),
            ("restart_count", "INTEGER NOT NULL DEFAULT 0"),
            ("health_check", "TEXT"),
            ("service_name", "TEXT"),
            ("last_exit_at", "TEXT"),
            ("restart_window_start", "TEXT"),
        ]
        for col, typedef in migrations:
            if col not in existing:
                await self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typedef}")
        gpu_cursor = await self._conn.execute("PRAGMA table_info(gpus)")
        gpu_existing = {row[1] for row in await gpu_cursor.fetchall()}
        gpu_migrations = [
            ("vendor", "TEXT NOT NULL DEFAULT 'nvidia'"),
            ("device_index", "INTEGER"),
            ("runtime_id", "TEXT"),
            ("discovery_backend", "TEXT NOT NULL DEFAULT 'nvml'"),
            ("schedulable", "INTEGER NOT NULL DEFAULT 1"),
        ]
        for col, typedef in gpu_migrations:
            if col not in gpu_existing:
                await self._conn.execute(f"ALTER TABLE gpus ADD COLUMN {col} {typedef}")
        await self._conn.execute("UPDATE gpus SET runtime_id = id WHERE runtime_id IS NULL")
        alloc_cursor = await self._conn.execute("PRAGMA table_info(gpu_allocations)")
        alloc_existing = {row[1] for row in await alloc_cursor.fetchall()}
        if "exclusive" not in alloc_existing:
            # Existing allocations predate the refcount model; treat them as
            # exclusive (the legacy batch semantics).
            await self._conn.execute(
                "ALTER TABLE gpu_allocations ADD COLUMN exclusive INTEGER NOT NULL DEFAULT 1"
            )
        idem_cursor = await self._conn.execute("PRAGMA table_info(idempotency_keys)")
        idem_existing = {row[1] for row in await idem_cursor.fetchall()}
        if "body_hash" not in idem_existing:
            # NULL on every pre-P22 row; the body comparison is soft precisely so
            # those rows keep replaying instead of 422-ing after an upgrade.
            await self._conn.execute("ALTER TABLE idempotency_keys ADD COLUMN body_hash TEXT")
        endpoint_cursor = await self._conn.execute("PRAGMA table_info(service_endpoints)")
        endpoint_existing = {row[1] for row in await endpoint_cursor.fetchall()}
        if "active_host_port" not in endpoint_existing:
            # (P24b / D-P24-4b) NULL on every pre-P24 row, read as "same as
            # host_port" by the COALESCE in ``list_active_service_routes`` and by
            # ``ServiceEndpoint.live_port``. The ``_SCHEMA`` edit alone would
            # silently no-op on every existing install (the P22 WP-B lesson).
            await self._conn.execute(
                "ALTER TABLE service_endpoints ADD COLUMN active_host_port INTEGER"
            )
        tok_cursor = await self._conn.execute("PRAGMA table_info(api_tokens)")
        tok_existing = {row[1] for row in await tok_cursor.fetchall()}
        tok_migrations = [
            ("expires_at", "TEXT"),
            ("scope_services", "TEXT"),
        ]
        for col, typedef in tok_migrations:
            if col not in tok_existing:
                # (P25 / D-P25-1) NULL on every pre-P25 row: NULL expiry means
                # "never expires", NULL scope means "unscoped". An upgrade must
                # never lock an operator out of their own daemon. The ``_SCHEMA``
                # edit alone would silently no-op on every existing install (the
                # P22 WP-B lesson).
                await self._conn.execute(f"ALTER TABLE api_tokens ADD COLUMN {col} {typedef}")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_uploads (
                id           TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                path         TEXT NOT NULL,
                size_bytes   INTEGER NOT NULL,
                uploaded_at  TEXT DEFAULT (datetime('now')),
                used_by_jobs TEXT
            )
            """
        )
        await self._ensure_nullable_script_path()
        # Created LAST: ``_ensure_nullable_script_path`` does ``DROP TABLE jobs``,
        # so an index declared in ``_SCHEMA`` would be dropped with it. The
        # partial WHERE keeps batch rows (``service_name IS NULL``) out of the
        # unique constraint while enforcing one stable row per service name.
        await self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_service_name "
            "ON jobs(service_name) WHERE service_name IS NOT NULL"
        )
        await self._conn.commit()

    async def _get_jobs_table_info(self) -> list[aiosqlite.Row]:
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(jobs)")
        return list(await cursor.fetchall())

    async def _ensure_nullable_script_path(self) -> None:
        assert self._conn is not None
        table_info = await self._get_jobs_table_info()
        script_path_col = next((row for row in table_info if row[1] == "script_path"), None)
        if script_path_col is None or script_path_col[3] == 0:
            return

        await self._conn.execute("PRAGMA foreign_keys=OFF")
        await self._conn.execute(
            """
            CREATE TABLE jobs_new (
                id          TEXT PRIMARY KEY,
                name        TEXT,
                script_path TEXT,
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
                template_id TEXT,
                kind        TEXT NOT NULL DEFAULT 'batch',
                submitted_by_token TEXT,
                idempotency_key TEXT,
                desired_state TEXT,
                restart_policy TEXT,
                restart_count INTEGER NOT NULL DEFAULT 0,
                health_check TEXT,
                service_name TEXT,
                last_exit_at TEXT,
                restart_window_start TEXT
            )
            """
        )
        # Migrations above add every required column before rebuilding.
        # Keep this list positionally identical to CREATE TABLE or legacy upgrades lose data.
        await self._conn.execute(
            """
            INSERT INTO jobs_new
            SELECT id, name, script_path, gpu_count, priority, status, container_id, created_at,
                   started_at, finished_at, paused_at, exit_code, retries, max_retries, config,
                   tags, preemptible, max_runtime_seconds, time_window, error_class, error_message,
                   submitted_via, template_id, kind, submitted_by_token, idempotency_key,
                   desired_state, restart_policy, restart_count, health_check, service_name,
                   last_exit_at, restart_window_start
            FROM jobs
            """
        )
        await self._conn.execute("DROP TABLE jobs")
        await self._conn.execute("ALTER TABLE jobs_new RENAME TO jobs")
        await self._conn.execute("PRAGMA foreign_keys=ON")

    async def __aenter__(self) -> Database:
        await self.connect()
        await self.init_schema()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
