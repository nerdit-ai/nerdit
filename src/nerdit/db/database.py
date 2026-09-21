"""SQLite database connection and schema management."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite

# Stdlib-only module (no ``nerdit.db`` import), so this cannot cycle.
from nerdit.core.project_identity import mint_project_id

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
    restart_window_start TEXT,
    -- (P40a / D-P40-4) The project triple; NULL on model/database/batch rows.
    -- ``service_name`` stays the wire/filesystem/proxy key (D-P40-2); these
    -- three only group rows. Existing installs gain them through the second
    -- ``ADD COLUMN`` loop at the END of ``_migrate`` (after the ``jobs_new``
    -- rebuild, which must never learn them), and the partial unique index on
    -- the triple lives there too, for the ``idx_jobs_service_name`` reason.
    project_id  TEXT,
    environment TEXT,
    service     TEXT
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

-- Secret claims (P39 D-P39-1): one row per service name that has secrets but
-- no ``jobs`` row yet. The first ``POST /secrets/{name}`` on a rowless name
-- mints it for the caller's token; ``reserve_service_for_token`` consumes it
-- inside the row-insert transaction, so a stranger can never deploy over a
-- name whose secrets someone else set. No FK to ``jobs``, for the
-- ``service_shares`` reason above -- and because a claim exists precisely
-- when no row does. No FK to ``api_tokens`` either: a claimant token may be
-- revoked, and the claim must then survive as an admin-only row rather than
-- vanish and reopen the name. ``token_id`` NULL = a LOCAL/LEGACY_ADMIN
-- claimant (admin anyway). A wholly new table, so it rides ``_SCHEMA``
-- unconditionally -- no ``_migrate`` block.
CREATE TABLE IF NOT EXISTS secret_claims (
    service_name TEXT PRIMARY KEY,
    token_id     TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Projects (P40a / D-P40-5): one row per project name, the project-scope
-- reservation. ``submitted_by_token`` is the owner's token id (the ``jobs``
-- precedent); NULL = a LOCAL/LEGACY_ADMIN owner, admin-only. No FK to
-- ``api_tokens``, for the ``secret_claims`` reason above: a revoked owner
-- must leave an admin-only row, never a vanished one that reopens the name.
-- ``jobs.project_id`` points here without an FK either -- it is added by
-- ``ALTER TABLE`` on every existing install, and P40a backfills it. A wholly
-- new table, so it rides ``_SCHEMA`` unconditionally -- no ``_migrate`` block.
CREATE TABLE IF NOT EXISTS projects (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    submitted_by_token TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Variable flags (P40c / D-P40-1): one row per key set through the variables
-- API, recording only whether it is plain. There is NO value column -- every
-- value, plain or secret, lives in the scope's encrypted file; a key with no
-- row here is secret. ``environment`` / ``service`` use '' (never NULL) for
-- the project scope because NULLs never collide in a SQLite primary key.
-- The FK is legal because ``projects.id`` is a real PK (D-P40-17): deleting
-- the project row removes its flags. Rides ``_SCHEMA`` with no ``_migrate``
-- block, like ``projects``.
CREATE TABLE IF NOT EXISTS variables (
    project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    environment TEXT NOT NULL DEFAULT '',
    service     TEXT NOT NULL DEFAULT '',
    key         TEXT NOT NULL,
    plain       INTEGER NOT NULL CHECK (plain IN (0, 1)),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, environment, service, key)
);
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
        # (P40a / D-P40-4) Placed AFTER the rebuild and its index, on purpose
        # -- see the method for why the order is load-bearing.
        await self._migrate_project_identity()
        await self._conn.commit()

    async def _migrate_project_identity(self) -> None:
        """Add the P40a project triple, its index and the name-keyed backfill.

        Runs last in ``_migrate``: the ``jobs_new`` rebuild fires only on a
        pre-P1 database, which has never seen these columns, so placing this
        after it means the 33-column literals can never drop them (D-P40-4).
        """
        assert self._conn is not None
        # ``PRAGMA table_info`` is re-read: the rebuild may have replaced the
        # table since ``_migrate`` read it at the top.
        existing = {row[1] for row in await self._get_jobs_table_info()}
        for col in ("project_id", "environment", "service"):
            if col not in existing:
                await self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
        # (D-P40-3) One row per triple, beside ``idx_jobs_service_name`` -- never
        # instead of it. Partial so untripled rows (models, databases, batch)
        # stay out of the constraint.
        await self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_project_service "
            "ON jobs(project_id, environment, service) WHERE project_id IS NOT NULL"
        )
        # Backfill: every tripleless service row is ``(name, production, web)``
        # in a project named after it (D-P40-2, the legacy label is the
        # project name), owned by the row's token. Find-or-create by name keeps
        # ids stable across boots; ``project_id IS NULL`` makes a second boot a
        # no-op. A row with ``service_name`` NULL (the P1-P2 window) stays
        # tripleless: unreachable by name today, fails closed in auth.
        cursor = await self._conn.execute(
            "SELECT id, service_name, submitted_by_token FROM jobs "
            "WHERE kind = 'service' AND service_name IS NOT NULL AND project_id IS NULL"
        )
        for job_id, name, owner in await cursor.fetchall():
            row = await (
                await self._conn.execute("SELECT id FROM projects WHERE name = ?", (name,))
            ).fetchone()
            project_id = row[0] if row else mint_project_id()
            if row is None:
                await self._conn.execute(
                    "INSERT INTO projects (id, name, submitted_by_token) VALUES (?, ?, ?)",
                    (project_id, name, owner),
                )
            await self._conn.execute(
                "UPDATE jobs SET project_id = ?, environment = 'production', service = 'web' "
                "WHERE id = ?",
                (project_id, job_id),
            )

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

        # ``PRAGMA foreign_keys`` is a silent no-op inside a transaction, and
        # ``_migrate`` has already opened Python's implicit one. Commit first,
        # or ``DROP TABLE jobs`` runs its implicit DELETE with FKs on and any
        # job_logs / gpu_allocations / service_endpoints child row fails the
        # boot. The explicit BEGIN keeps jobs_new/DROP/RENAME atomic.
        await self._conn.commit()
        await self._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            await self._conn.execute("BEGIN")
            await self._rebuild_jobs_nullable_script_path()
            await self._conn.commit()
        except BaseException:
            await self._conn.rollback()
            raise
        finally:
            await self._conn.execute("PRAGMA foreign_keys=ON")

    async def _rebuild_jobs_nullable_script_path(self) -> None:
        assert self._conn is not None
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

    async def __aenter__(self) -> Database:
        await self.connect()
        await self.init_schema()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
