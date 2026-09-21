"""``[db.*]`` binding resolution + env injection (P15 / D-A).

The executable form of the ``DATABASE_URL``-is-the-contract analog. The injected
env var names are asserted LITERALLY; the managed-readiness ladder mirrors the
frozen ``[ai.*]`` resolver. Pure-async unit tests over the in-memory DB fixture
(no TestClient), plus a direct ``_resolve_launch_env`` merge-order check.
"""

from __future__ import annotations

import json

import pytest

from nerdit.config.settings import ServicesSettings
from nerdit.core.data.backend import PostgresBackend
from nerdit.core.data.binding import (
    ResolvedDbBinding,
    inject_db_env,
    inject_db_env_key_names,
    resolve_db_binding,
    resolve_db_bindings,
)
from nerdit.core.models.binding import BindingNotReady, inject_env_key_names
from nerdit.core.secrets import SecretManager, project_storage_name
from nerdit.core.services import ServiceController
from nerdit.db.models import Job, JobKind, JobStatus

BRIDGE_HOST = "172.17.0.1"
PW = "deadbeef" * 8  # 64-hex, the mint shape

MANAGED_SPEC = {"provider": "managed", "database": "pg"}
EXTERNAL_PG_SPEC = {
    "provider": "external",
    "url": "postgresql://appuser@db.example.com:5432/app",
    "password": "${secrets.DB_PW}",
}
EXTERNAL_REDIS_SPEC = {
    "provider": "external",
    "url": "redis://cache.example.com:6379/0",
    "password": "${secrets.REDIS_PW}",
}

_BACKEND = PostgresBackend()


def _backend_for(_cfg: dict) -> PostgresBackend:
    return _BACKEND


async def _serve_db(queries, *, name="pg", ready=True, status=JobStatus.running, endpoint=True):
    cfg = {"backend": "postgres", "image": "postgres:16", "port": 5432}
    if ready:
        cfg["db_ready"] = True
    job = Job(
        name=name,
        kind=JobKind.database,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
    )
    await queries.create_job(job)
    if endpoint:
        await queries.acquire_service_port(name, job.id, 5432, (9400, 9499))
    return job


# --- Injected env grammar (LITERAL contract) -----------------------------------


def test_inject_db_env_names_managed_default():
    resolved = {"default": ResolvedDbBinding(url="postgresql://x", default_alias="DATABASE_URL")}
    env = inject_db_env(resolved)
    assert env["NERDIT_DB_DEFAULT_URL"] == "postgresql://x"
    assert env["DATABASE_URL"] == "postgresql://x"
    assert set(env) == {"NERDIT_DB_DEFAULT_URL", "DATABASE_URL"}


def test_inject_db_env_redis_alias():
    resolved = {"default": ResolvedDbBinding(url="redis://x", default_alias="REDIS_URL")}
    env = inject_db_env(resolved)
    assert env["REDIS_URL"] == "redis://x"
    assert "DATABASE_URL" not in env


def test_inject_db_env_non_default_has_no_alias():
    resolved = {"cache": ResolvedDbBinding(url="redis://x", default_alias="REDIS_URL")}
    env = inject_db_env(resolved)
    assert env == {"NERDIT_DB_CACHE_URL": "redis://x"}


def test_inject_db_env_key_names_match_injector():
    specs = {
        "default": {"provider": "managed", "database": "pg"},
        "cache": {"provider": "external", "url": "redis://h/0", "password": "${secrets.P}"},
    }
    names = inject_db_env_key_names(specs)
    assert "NERDIT_DB_DEFAULT_URL" in names
    assert "NERDIT_DB_CACHE_URL" in names
    assert "DATABASE_URL" in names  # managed default → postgres alias


def test_inject_db_env_key_names_external_redis_default_alias():
    specs = {"default": {"provider": "external", "url": "rediss://h/0", "password": "${secrets.P}"}}
    assert inject_db_env_key_names(specs) == {"NERDIT_DB_DEFAULT_URL", "REDIS_URL"}


def test_names_only_managed_default_reports_database_url_even_for_redis():
    """Accepted softness (`_default_alias_for_spec`): a managed ``default`` binding
    whose target row is a Redis backend injects ``REDIS_URL`` at launch, but the
    names-only path (no row/backend in hand) reports ``DATABASE_URL`` — the only
    key-name projection a live resolve corrects. ``NERDIT_DB_<NAME>_URL`` stays
    exact. Pinned so the divergence is a decision, not a silent regression.
    """
    specs = {"default": {"provider": "managed", "database": "redis"}}
    assert inject_db_env_key_names(specs) == {"NERDIT_DB_DEFAULT_URL", "DATABASE_URL"}


# --- Collision-freedom: AI ∪ DB key sets are disjoint --------------------------


@pytest.mark.parametrize("names", [{"default"}, {"default", "cache", "analytics"}, {"a", "b"}])
def test_ai_and_db_key_sets_are_disjoint(names):
    ai_keys = inject_env_key_names(names)
    db_specs = {n: {"provider": "managed", "database": "pg"} for n in names}
    db_keys = inject_db_env_key_names(db_specs)
    assert ai_keys.isdisjoint(db_keys)


# --- External resolution + password quoting ------------------------------------


async def test_external_postgres_resolves_and_inserts_password():
    resolved = await resolve_db_binding(
        "default", EXTERNAL_PG_SPEC, None, {"DB_PW": "s3cret"}, BRIDGE_HOST, _backend_for, dict
    )
    assert resolved.url == "postgresql://appuser:s3cret@db.example.com:5432/app"
    assert resolved.default_alias == "DATABASE_URL"


async def test_external_redis_alias_and_empty_user():
    resolved = await resolve_db_binding(
        "cache", EXTERNAL_REDIS_SPEC, None, {"REDIS_PW": "pw"}, BRIDGE_HOST, _backend_for, dict
    )
    assert resolved.url == "redis://:pw@cache.example.com:6379/0"
    assert resolved.default_alias == "REDIS_URL"


@pytest.mark.parametrize(
    "raw, quoted",
    [
        ("p@ss", "p%40ss"),
        ("a:b", "a%3Ab"),
        ("a/b", "a%2Fb"),
        ("a#b", "a%23b"),
        ("a?b", "a%3Fb"),
        ("a b", "a%20b"),
    ],
)
async def test_external_password_special_chars_quoted(raw, quoted):
    resolved = await resolve_db_binding(
        "default", EXTERNAL_PG_SPEC, None, {"DB_PW": raw}, BRIDGE_HOST, _backend_for, dict
    )
    assert f"appuser:{quoted}@" in resolved.url


async def test_external_username_with_raw_at_splits_at_last_at():
    """A username containing a raw ``@`` (parse admits it — ``parts.password`` is
    None) composes a DSN whose userinfo/host boundary is the LAST ``@``
    (``rpartition``), matching ``urlsplit``/libpq — never the first.
    """
    spec = {
        "provider": "external",
        "url": "postgresql://weird@user@db.example.com:5432/app",
        "password": "${secrets.DB_PW}",
    }
    resolved = await resolve_db_binding(
        "default", spec, None, {"DB_PW": "pw"}, BRIDGE_HOST, _backend_for, dict
    )
    assert resolved.url == "postgresql://weird@user:pw@db.example.com:5432/app"


async def test_external_missing_secret_is_not_ready():
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", EXTERNAL_PG_SPEC, None, {}, BRIDGE_HOST, _backend_for, dict
        )
    assert "DB_PW" in str(exc.value)
    assert "nerdit secrets set" in str(exc.value)


async def test_external_shared_ref_resolves_and_precedence():
    spec = {**EXTERNAL_PG_SPEC, "password": "${secrets.shared.DB_PW}"}
    # shared store
    r = await resolve_db_binding(
        "default", spec, None, {}, BRIDGE_HOST, _backend_for, dict, shared_env={"DB_PW": "shared"}
    )
    assert ":shared@" in r.url
    # per-service override wins
    r2 = await resolve_db_binding(
        "default",
        spec,
        None,
        {"DB_PW": "own"},
        BRIDGE_HOST,
        _backend_for,
        dict,
        shared_env={"DB_PW": "shared"},
    )
    assert ":own@" in r2.url


# --- Managed readiness ladder ---------------------------------------------------


async def test_managed_running_ready_resolves_dsn(queries):
    await _serve_db(queries)
    endpoint = await queries.get_service_endpoint("pg")
    resolved = await resolve_db_binding(
        "default",
        MANAGED_SPEC,
        queries,
        {},
        BRIDGE_HOST,
        _backend_for,
        lambda s: {"POSTGRES_PASSWORD": PW},
    )
    assert resolved.url == f"postgresql://nerdit:{PW}@{BRIDGE_HOST}:{endpoint.host_port}/nerdit"
    assert resolved.default_alias == "DATABASE_URL"


async def test_managed_missing_row_is_not_ready(queries):
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", MANAGED_SPEC, queries, {}, BRIDGE_HOST, _backend_for, dict
        )
    assert "not provisioned" in str(exc.value)
    assert "nerdit db create" in str(exc.value)


@pytest.mark.parametrize("status", [JobStatus.building, JobStatus.stopped, JobStatus.restarting])
async def test_managed_non_running_is_not_ready(queries, status):
    await _serve_db(queries, status=status)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", MANAGED_SPEC, queries, {}, BRIDGE_HOST, _backend_for, dict
        )
    assert status.value in str(exc.value)


async def test_managed_not_ready_flag_is_not_ready(queries):
    await _serve_db(queries, ready=False)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", MANAGED_SPEC, queries, {}, BRIDGE_HOST, _backend_for, dict
        )
    assert "starting up" in str(exc.value)


async def test_managed_no_endpoint_is_not_ready(queries):
    await _serve_db(queries, endpoint=False)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", MANAGED_SPEC, queries, {}, BRIDGE_HOST, _backend_for, dict
        )
    assert "no live endpoint" in str(exc.value)


async def test_managed_missing_credential_is_not_ready(queries):
    await _serve_db(queries)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_db_binding(
            "default", MANAGED_SPEC, queries, {}, BRIDGE_HOST, _backend_for, lambda s: {}
        )
    assert "no minted credential" in str(exc.value)


async def test_resolve_db_bindings_all_or_nothing(queries):
    await _serve_db(queries)
    specs = {"default": MANAGED_SPEC, "cache": EXTERNAL_REDIS_SPEC}
    # cache secret missing → whole resolve raises
    with pytest.raises(BindingNotReady):
        await resolve_db_bindings(
            specs, queries, {}, BRIDGE_HOST, _backend_for, lambda s: {"POSTGRES_PASSWORD": PW}
        )
    resolved = await resolve_db_bindings(
        specs,
        queries,
        {"REDIS_PW": "pw"},
        BRIDGE_HOST,
        _backend_for,
        lambda s: {"POSTGRES_PASSWORD": PW},
    )
    assert set(resolved) == {"default", "cache"}


# --- Launch-env merge order: injected binding beats a same-named user secret ----


class _FakeData:
    bridge_host = BRIDGE_HOST

    def backend_for(self, _cfg: dict) -> PostgresBackend:
        return _BACKEND


async def test_injected_database_url_beats_user_secret(queries, tmp_path):
    """Decision #4 extended: a DATABASE_URL in either variable scope must not win (D-P40-9)."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("pg", {"POSTGRES_PASSWORD": PW})
    mgr.set("app", {"DATABASE_URL": "postgresql://user-should-lose@nowhere/x"})
    project_id = "prj_" + "a" * 16
    mgr.set(
        project_storage_name(project_id),
        {
            "DATABASE_URL": "postgresql://project-should-lose@nowhere/x",
            "NERDIT_DB_DEFAULT_URL": "x",
        },
    )
    await _serve_db(queries)
    endpoint = await queries.get_service_endpoint("pg")

    controller = ServiceController(
        queries=queries,
        runtime=None,
        services_settings=ServicesSettings(service_port_range="9500-9599"),
        secrets=mgr,
        data_controller=_FakeData(),
    )
    # Without a model_controller the bridge host falls back to the
    # platform-resolved default (172.17.0.1 on Linux, host.docker.internal on
    # macOS); pin it so the expectation below holds on both.
    controller._bridge_host = BRIDGE_HOST
    app = Job(
        name="app",
        kind=JobKind.service,
        service_name="app",
        project_id=project_id,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "demo:1", "port": 8000, "db": {"default": MANAGED_SPEC}}),
    )
    resolved_env = await controller._resolve_launch_env(
        app, json.loads(app.config), 8000, None, {"default": MANAGED_SPEC}
    )
    env = resolved_env.env
    assert (
        env["DATABASE_URL"] == f"postgresql://nerdit:{PW}@{BRIDGE_HOST}:{endpoint.host_port}/nerdit"
    )
    assert env["NERDIT_DB_DEFAULT_URL"] == env["DATABASE_URL"]
