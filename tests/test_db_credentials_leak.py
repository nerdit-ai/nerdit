"""D-B minted-credential leak assertions (P15 WP2 — the security-review core).

The daemon mints a database password server-side (``secrets.token_hex(32)``) and
stores it write-only. It must NEVER ride a response body, an idempotency cache
row, an audit param, or a launched container's env beyond the single allowlisted
``POSTGRES_PASSWORD`` key. One shared leak-assert helper is applied across every
surface; a launch-level test pins that a user-set ``POSTGRES_HOST_AUTH_METHOD``
secret in the row's scope can never reach ``ContainerConfig.env`` (the §1.3
allowlist).
"""

from __future__ import annotations

import json
import os
import re

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import (
    ContainerSettings,
    DatabasesSettings,
    ModelsSettings,
    NerditSettings,
    ServicesSettings,
)
from nerdit.core.data.backend import PostgresBackend
from nerdit.core.data.controller import DataController
from nerdit.core.secrets import SecretManager
from nerdit.core.services import ServiceController
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.databases import router as databases_router
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.database import Database
from nerdit.db.models import Job, JobKind, JobStatus
from nerdit.db.queries import Queries

_HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def assert_no_credential(text: str | None, password: str | None = None) -> None:
    """The shared leak assertion: no 64-hex token, no credential-bearing DSN.

    Applied to every response body / cache row / audit param / diagnose payload
    that could conceivably carry the minted value. Fails on a bare 64-hex token,
    on the exact minted ``password`` when supplied, and on a ``postgresql://``
    URL that embeds a userinfo password.
    """
    if text is None:
        return
    assert not _HEX64.search(text), f"64-hex token leaked: {text[:200]}"
    if password is not None:
        assert password not in text, "the exact minted password leaked"
    # A DSN with an embedded userinfo password (user:pw@host).
    assert not re.search(r"postgresql://[^/\s]*:[^/\s@]+@", text or ""), "credential DSN leaked"


# --- route surfaces: response / replay / cache row / audit --------------------


async def _db_env(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(databases_router)
    app.include_router(services_router)
    app.state.queries = queries
    app.state.settings = NerditSettings()
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.data_controller = DataController(
        PostgresBackend(),
        runtime=__import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(),
        queries=queries,
        default_backend="postgres",
        databases_settings=DatabasesSettings(),
        models_settings=ModelsSettings(),
        data_dir=str(tmp_path),
    )
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app, db, queries


@pytest.mark.asyncio
async def test_no_credential_on_response_replay_cache_or_audit(tmp_path):
    app, db, queries = await _db_env(tmp_path)
    try:
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1")
        async with client:
            headers = {"Idempotency-Key": "K-leak"}
            first = await client.post("/databases", json={}, headers=headers)
            assert first.status_code == 201
            replay = await client.post("/databases", json={}, headers=headers)
            assert replay.headers.get("Idempotent-Replay") == "true"

        # The minted value exists only in the encrypted store.
        password = app.state.secret_manager.load("pg")["POSTGRES_PASSWORD"]
        assert _HEX64.fullmatch(password)

        # 1) the fresh create response body
        assert_no_credential(first.text, password)
        # 2) the idempotent-replay response body
        assert_no_credential(replay.text, password)
        # 3) the cached idempotency_keys row (same protection must hold)
        cur = await db.conn.execute(
            "SELECT response_body FROM idempotency_keys WHERE idem_key = ?", ("K-leak",)
        )
        cached = await cur.fetchone()
        assert cached is not None
        assert_no_credential(cached["response_body"], password)
        # 4) every audit row's params
        rows, _ = await queries.list_audit_log(limit=200)
        for row in rows:
            params = row.params_redacted
            as_text = params if isinstance(params, str) else json.dumps(params)
            assert_no_credential(as_text, password)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_credential_in_diagnose_payload(tmp_path):
    """§3 D-B: ``GET /services/{db}/diagnose`` on a provisioned row leaks nothing.

    The diagnose payload recomputes the database row's injected env-var NAMES
    (the allowlisted backend statics + the minted key name) — never the value.
    Closes the §3 D-B surface list (dry-run + diagnose) with the same 64-hex/DSN
    regex the four other surfaces use, not only a structural names check.
    """
    app, db, queries = await _db_env(tmp_path)
    try:
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1")
        async with client:
            created = await client.post("/databases", json={})
            assert created.status_code == 201
            diag = await client.get("/services/pg/diagnose")
            assert diag.status_code == 200, diag.text
        password = app.state.secret_manager.load("pg")["POSTGRES_PASSWORD"]
        assert_no_credential(diag.text, password)
        # The name (never the value) is the point of the injected-env-keys block.
        assert "POSTGRES_PASSWORD" in diag.text
    finally:
        await db.close()


# --- launch allowlist: a stray user secret never reaches the container --------


class _CapturingRuntime:
    """Minimal runtime that records the ContainerConfig passed to ``run``."""

    def __init__(self) -> None:
        self.run_configs: list = []

    async def run(self, config) -> str:
        self.run_configs.append(config)
        return "cid-1"

    async def image_exists(self, image: str) -> bool:
        return True

    async def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
        since: int | None = None,
    ):
        return
        yield  # pragma: no cover — empty async generator

    async def stop(self, container_id: str, timeout: int = 10) -> None: ...
    async def remove(self, container_id: str, force: bool = False) -> None: ...


@pytest.mark.asyncio
async def test_host_auth_method_secret_never_reaches_container_env(tmp_path, queries):
    """§1.3 allowlist: a user-set ``POSTGRES_HOST_AUTH_METHOD`` never launches.

    The database launch branch assembles an ALLOWLISTED env (backend statics +
    the minted key only), never the full secret scope — so first-boot ``pg_hba``
    stays the image's scram default and cannot be flipped to ``trust`` via a
    stray secret in the row's scope.
    """
    secrets = SecretManager(tmp_path / "secrets")
    # Both the minted credential AND a hostile user-set secret share the scope.
    secrets.set("pg", {"POSTGRES_PASSWORD": "a" * 64, "POSTGRES_HOST_AUTH_METHOD": "trust"})

    data_controller = DataController(
        PostgresBackend(),
        runtime=_CapturingRuntime(),
        queries=queries,
        default_backend="postgres",
        databases_settings=DatabasesSettings(),
        models_settings=ModelsSettings(),
        data_dir=str(tmp_path),
    )
    runtime = _CapturingRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        container_settings=ContainerSettings(),
        secrets=secrets,
        data_controller=data_controller,
        data_dir=str(tmp_path),
    )

    job = Job(
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
    await queries.create_job(job)
    await controller._launch(job)
    await data_controller.shutdown()

    assert runtime.run_configs, "the database container was never launched"
    env = runtime.run_configs[-1].env
    # Exactly the four allowlisted keys — the stray secret is absent.
    assert set(env) == {"POSTGRES_USER", "POSTGRES_DB", "PGDATA", "POSTGRES_PASSWORD"}
    assert "POSTGRES_HOST_AUTH_METHOD" not in env
    assert env["POSTGRES_PASSWORD"] == "a" * 64
    # Non-root by construction (D-P15-6), as the daemon uid (D-P15-7) + the
    # sandbox overlay intact.
    assert runtime.run_configs[-1].user == f"{os.getuid()}:{os.getgid()}"
    assert runtime.run_configs[-1].cap_drop == ["ALL"]
    assert runtime.run_configs[-1].no_new_privileges is True


# --- external [db] url shape errors never echo the rejected credential (#6) ---


def test_redact_db_url_strips_userinfo_and_query():
    from nerdit.config.project import _redact_db_url

    # A userinfo password (even one carrying a literal '@') is dropped, keeping
    # scheme/host/path; the query string (which may hold ?password=) is dropped.
    assert _redact_db_url("postgresql://user:SUPERSECRETpw@host:5432/db") == (
        "postgresql://host:5432/db"
    )
    assert _redact_db_url("postgresql://user:pa@ss@host/db") == "postgresql://host/db"
    # Unencoded whitespace inside the userinfo still redacts: urlsplit's netloc
    # boundary is the first '/', so parts.password spans the space — the regex
    # must too (only '/' is excluded from its class).
    assert _redact_db_url("postgresql://user:pa ss@host/db") == "postgresql://host/db"
    assert _redact_db_url("redis://:pw@host:6379/0") == "redis://host:6379/0"
    # A bare username userinfo is dropped too (harmless), and the ?password= query
    # is stripped — the credential never survives.
    assert _redact_db_url("postgresql://user@host/db?password=LEAKME") == ("postgresql://host/db")
    assert _redact_db_url("postgresql://host/db") == "postgresql://host/db"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:SUPERSECRETpw@host:5432/db",  # userinfo password
        "postgresql://user@host:5432/db?password=SUPERSECRETpw",  # query password
        "postgresql://user:SUPERSECRETpw nope@host/db",  # unencoded space in password
    ],
)
def test_bad_external_url_error_never_echoes_the_credential(url):
    """The shape-check message flows into the validation error, the deploy 422
    body (``validate_db_section``) and the persisted-spec ``BindingNotReady``
    string (job_logs / diagnose). None of them may echo the rejected credential.
    """
    import asyncio

    from pydantic import ValidationError

    from nerdit.config.app_config import validate_db_section
    from nerdit.config.project import DbBindingConfig
    from nerdit.core.data.binding import BindingNotReady, resolve_db_binding
    from nerdit.daemon.errors import NerditError

    secret = "SUPERSECRETpw"
    spec = {"provider": "external", "url": url, "password": "${secrets.PW}"}

    # 1) the raw ValidationError (what pydantic surfaces to callers)
    with pytest.raises(ValidationError) as vi:
        DbBindingConfig(**spec)
    assert_no_credential(str(vi.value), secret)

    # 2) the deploy 422 envelope body (validate_db_section interpolates {exc})
    with pytest.raises(NerditError) as ne:
        validate_db_section({"default": dict(spec)}, source="nerdit.toml")
    assert_no_credential(ne.value.message, secret)

    # 3) the persisted-spec BindingNotReady (job_logs-bound; also diagnose)
    async def _resolve() -> None:
        await resolve_db_binding(
            "default",
            spec,
            queries=None,
            secrets={},
            bridge_host="172.17.0.1",
            backend_for=lambda cfg: None,
            load_scope=lambda name: {},
        )

    with pytest.raises(BindingNotReady) as bn:
        asyncio.run(_resolve())
    assert_no_credential(str(bn.value), secret)
