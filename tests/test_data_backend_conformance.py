"""Conformance suite for the P15 ``DataBackend`` Protocol.

Parameterized over every registered backend (the P11 ``FakeVllm`` precedent):
Postgres and Redis, sharing every assertion — including the P37 dump/restore
members (``dump_format``/``dump_filename``/``dump_argv``/``dump_env``/
``restore_argv``), whose contract is pinned once here for both engines. Each
case bundles the backend instance, its
expected static attrs, a sample password, and a pair of asyncio TCP fakes —
``serve_ready`` (answers the readiness probe) and ``serve_starting`` (accepts
then resets, the transient case) — so ``ensure_ready`` is exercised end-to-end
over the real wire handshake with no real database.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from nerdit.core.data import DataBackend, DataNotReadyError, PostgresBackend, RedisBackend
from nerdit.core.data.backend import MANAGED_DB_NAME, MANAGED_DB_ROLE
from nerdit.db.models import ContainerConfig

# NB: no module-level asyncio mark — the suite mixes sync (shape/attr) and async
# (probe) tests; ``--asyncio-mode=auto`` collects the async ones automatically.


# --- asyncio TCP fake -----------------------------------------------------------


@asynccontextmanager
async def _fake_server(handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable]):
    """Run a throwaway TCP server on 127.0.0.1:0; yield its (host, port)."""
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    async with server:
        await server.start_serving()
        try:
            yield host, port
        finally:
            server.close()
            await server.wait_closed()


async def _reset_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Accept then immediately close without responding — the 'still starting' case."""
    writer.close()


async def _pg_ready_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Read the SSLRequest and answer a single ``N`` byte (SSL not offered, but alive)."""
    try:
        await reader.read(8)
        writer.write(b"N")
        await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


async def _redis_ready_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Read the inline PING and answer ``-NOAUTH ...`` (a live, password-set server)."""
    try:
        await reader.read(6)
        writer.write(b"-NOAUTH Authentication required.\r\n")
        await writer.drain()
    except OSError:
        pass
    finally:
        writer.close()


# --- backend cases (WP8 appends a Redis case) -----------------------------------


@dataclass
class _BackendCase:
    backend: DataBackend
    name: str
    name_prefix: str
    container_port: int
    minted_secret_key: str
    default_env_alias: str
    volume_spec: str
    sample_password: str
    static_env_keys: set[str]
    ready_handler: Callable
    dsn_scheme: str
    # P37 (§1.1): the dump substrate's per-backend statics.
    dump_format: str
    dump_filename: str
    dump_env_key: str
    dump_argv0: str
    #: ``None`` for a backend whose restore is a file install under quiesce
    #: (Redis, D-P37-6); otherwise the expected ``restore_argv`` argv0.
    restore_argv0: str | None


_POSTGRES_CASE = _BackendCase(
    backend=PostgresBackend(ready_timeout_s=1.0),
    name="postgres",
    name_prefix="pg",
    container_port=5432,
    minted_secret_key="POSTGRES_PASSWORD",
    default_env_alias="DATABASE_URL",
    volume_spec="data:/var/lib/postgresql/data",
    # Distinctive per backend and shaped like the real thing (``token_hex(32)``)
    # so the P37 "never on argv" pins below cannot pass by coincidence.
    sample_password="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0",
    static_env_keys={"POSTGRES_USER", "POSTGRES_DB", "PGDATA", "POSTGRES_PASSWORD"},
    ready_handler=_pg_ready_handler,
    dsn_scheme="postgresql://",
    dump_format="pg_custom",
    dump_filename="dump.pgdump",
    dump_env_key="PGPASSWORD",
    dump_argv0="pg_dump",
    restore_argv0="pg_restore",
)

_REDIS_CASE = _BackendCase(
    backend=RedisBackend(ready_timeout_s=1.0),
    name="redis",
    name_prefix="redis",
    container_port=6379,
    minted_secret_key="REDIS_PASSWORD",
    default_env_alias="REDIS_URL",
    volume_spec="data:/data",
    sample_password="a9b8c7d6e5f40312a9b8c7d6e5f40312a9b8c7d6e5f40312a9b8c7d6e5f40312",
    static_env_keys={"REDIS_PASSWORD"},
    ready_handler=_redis_ready_handler,
    dsn_scheme="redis://",
    dump_format="rdb",
    dump_filename="dump.rdb",
    dump_env_key="REDISCLI_AUTH",
    dump_argv0="redis-cli",
    # D-P37-6: Redis has no client-drivable load command — restore is a file
    # install under quiesce, so the Protocol hook is ``None`` by design.
    restore_argv0=None,
)

CASES = [
    pytest.param(_POSTGRES_CASE, id=_POSTGRES_CASE.name),
    pytest.param(_REDIS_CASE, id=_REDIS_CASE.name),
]


# --- Protocol attrs -------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_backend_satisfies_protocol(case: _BackendCase) -> None:
    assert isinstance(case.backend, DataBackend)


@pytest.mark.parametrize("case", CASES)
def test_static_attrs(case: _BackendCase) -> None:
    b = case.backend
    assert b.name == case.name
    assert b.name_prefix == case.name_prefix
    assert b.container_port == case.container_port
    assert b.minted_secret_key == case.minted_secret_key
    assert b.default_env_alias == case.default_env_alias
    # D-P15-7: both v1 backends declare ``None`` → daemon-uid at launch.
    assert b.run_as_user is None
    assert b.volume_spec == case.volume_spec
    assert isinstance(b.image, str) and b.image


# --- container shape ------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_container_config_shape(case: _BackendCase) -> None:
    """Non-root user set, port published, gpu-free, env is the allowlist."""
    env = {case.minted_secret_key: case.sample_password}
    config = case.backend.container_config("db1", 17000, env)
    assert isinstance(config, ContainerConfig)
    # D-P15-6/D-P15-7: runs non-root, as the daemon's own uid:gid (composed at
    # launch time), so the on-disk data tree is daemon-owned.
    assert config.user == f"{os.getuid()}:{os.getgid()}"
    # No GPU parameter — databases are GPU-free.
    assert config.gpu_ids == []
    # The container port is published to the allocated host port.
    assert config.ports == {case.container_port: 17000}
    # Image comes from the backend.
    assert config.image == case.backend.image


@pytest.mark.parametrize("case", CASES)
def test_container_config_env_allowlist(case: _BackendCase) -> None:
    """Given only the minted credential, the env is EXACTLY the static allowlist.

    A stray key must not smuggle in — the launch branch (WP3) allowlists *env*
    to the minted key; the backend adds only its own image-native statics.
    """
    env = {case.minted_secret_key: case.sample_password}
    config = case.backend.container_config("db1", 17000, env)
    assert set(config.env) == case.static_env_keys
    assert config.env[case.minted_secret_key] == case.sample_password
    # POSTGRES_HOST_AUTH_METHOD (or any auth-shaping key) is never present.
    assert "POSTGRES_HOST_AUTH_METHOD" not in config.env


# --- dsn / public_endpoint ------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_dsn_carries_password_and_scheme(case: _BackendCase) -> None:
    dsn = case.backend.dsn("172.17.0.1", 17000, case.sample_password)
    assert dsn.startswith(case.dsn_scheme)
    assert case.sample_password in dsn
    assert "172.17.0.1:17000" in dsn


@pytest.mark.parametrize("case", CASES)
def test_public_endpoint_is_password_free(case: _BackendCase) -> None:
    endpoint = case.backend.public_endpoint("172.17.0.1", 17000)
    assert endpoint == "172.17.0.1:17000"
    assert case.sample_password not in endpoint


# --- ensure_ready wire probe ----------------------------------------------------


@pytest.mark.parametrize("case", CASES)
async def test_ensure_ready_success(case: _BackendCase) -> None:
    """A backend answering the readiness handshake returns cleanly (no raise)."""
    async with _fake_server(case.ready_handler) as (host, port):
        await case.backend.ensure_ready(host, port)


@pytest.mark.parametrize("case", CASES)
async def test_ensure_ready_accept_then_reset_is_transient(case: _BackendCase) -> None:
    """Accept-then-reset (the docker userland-proxy false-accept) → transient."""
    async with _fake_server(_reset_handler) as (host, port):
        with pytest.raises(DataNotReadyError):
            await case.backend.ensure_ready(host, port)


@pytest.mark.parametrize("case", CASES)
async def test_ensure_ready_unreachable_is_transient(case: _BackendCase) -> None:
    """Connect refused (nothing listening) → transient, never a permanent error."""
    # Port 1 is reserved and never listening in the test sandbox.
    with pytest.raises(DataNotReadyError):
        await case.backend.ensure_ready("127.0.0.1", 1)


# --- Redis-specific: credential rides argv by NAME only (D-P15-2) ---------------


def test_redis_command_carries_password_by_name_only() -> None:
    """The ``sh -c`` command expands ``$REDIS_PASSWORD`` in-container only.

    Host-visible argv / ``docker inspect`` must carry the unexpanded variable
    name, never the minted value (D-P15-2, §4 security checklist item 5).
    """
    config = _REDIS_CASE.backend.container_config(
        "cache", 17000, {"REDIS_PASSWORD": _REDIS_CASE.sample_password}
    )
    assert config.command == [
        "sh",
        "-c",
        'exec redis-server --requirepass "$REDIS_PASSWORD" --appendonly yes',
    ]
    joined = " ".join(config.command or [])
    assert "$REDIS_PASSWORD" in joined
    assert _REDIS_CASE.sample_password not in joined


# --- P37 dump/restore substrate (§1.1) ------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_dump_statics(case: _BackendCase) -> None:
    """``dump_format``/``dump_filename`` are the manifest + tar-member contract.

    Both are fixed per backend and never caller-supplied: ``dump_filename`` is
    the basename the tool writes inside the daemon-composed staging mount and
    the tar's first member, ``dump_format`` is what the manifest records and a
    restore compares against the target row's backend (D-P37-7/D-P37-10).
    """
    b = case.backend
    assert b.dump_format == case.dump_format
    assert b.dump_filename == case.dump_filename


@pytest.mark.parametrize("case", CASES)
def test_dump_env_is_exactly_one_credential_key(case: _BackendCase) -> None:
    """D-P37-4: the sibling's ENTIRE env is one key, whose value is the password.

    This is a **value** pin, not a name pin: ``REDISCLI_AUTH`` matches no
    ``_CREDENTIAL_KEY_PARTS`` entry and that list is deliberately not widened
    (§7), so what protects the tail is the controller passing this same value as
    ``scrub_values``.
    """
    env = case.backend.dump_env(case.sample_password)
    assert env == {case.dump_env_key: case.sample_password}


@pytest.mark.parametrize("case", CASES)
def test_dump_argv_shape(case: _BackendCase) -> None:
    """The dump argv names the tool, dials host:port, and writes the staged path."""
    argv = case.backend.dump_argv("172.17.0.1", 17000, "/nerdit-dump/" + case.dump_filename)
    assert argv[0] == case.dump_argv0
    assert "172.17.0.1" in argv
    assert "17000" in argv  # the port is stringified, never an int
    assert "/nerdit-dump/" + case.dump_filename in argv


@pytest.mark.parametrize("case", CASES)
def test_argv_never_carries_the_password(case: _BackendCase) -> None:
    """D-P37-4: no argv element contains the password, as a whole or a substring.

    The argv is recorded verbatim in ``config['last_dump']`` and is visible to
    ``docker inspect`` / the host process table, so it must be secret-free by
    construction — the credential travels only in :meth:`dump_env`.
    """
    pw = case.sample_password
    argvs = [case.backend.dump_argv("172.17.0.1", 17000, "/nerdit-dump/out")]
    restore = case.backend.restore_argv("172.17.0.1", 17000, "/nerdit-dump/in")
    if restore is not None:
        argvs.append(restore)
    for argv in argvs:
        assert all(pw not in element for element in argv)


def test_postgres_argvs_always_name_the_managed_role() -> None:
    """``-U <MANAGED_DB_ROLE>`` in EVERY Postgres argv (verified-live requirement).

    The sibling runs as the daemon's uid (D-P37-3) and that uid has no passwd
    entry in the postgres image, so without ``-U`` libpq fails ``local user with
    ID 501 does not exist`` before it dials at all — this is a correctness pin,
    not a style one.
    """
    backend = _POSTGRES_CASE.backend
    dump = backend.dump_argv("172.17.0.1", 17000, "/nerdit-dump/dump.pgdump")
    restore = backend.restore_argv("172.17.0.1", 17000, "/nerdit-dump/dump.pgdump")
    assert restore is not None
    for argv in (dump, restore):
        assert "-U" in argv
        assert argv[argv.index("-U") + 1] == MANAGED_DB_ROLE
        assert "-d" in argv
        assert argv[argv.index("-d") + 1] == MANAGED_DB_NAME


def test_postgres_dump_and_restore_flags() -> None:
    """The load-bearing flags: custom format out, single-transaction clean in."""
    backend = _POSTGRES_CASE.backend
    dump = backend.dump_argv("172.17.0.1", 17000, "/nerdit-dump/dump.pgdump")
    assert dump[0] == "pg_dump"
    # ``--format=custom`` is the only format ``pg_restore`` can drive, and
    # ``--file`` keeps the artifact off stdout (the tail is a scrubbed 16 KiB).
    assert "--format=custom" in dump
    assert dump[-2:] == ["--file", "/nerdit-dump/dump.pgdump"]
    assert {"--no-owner", "--no-privileges"} <= set(dump)

    restore = backend.restore_argv("172.17.0.1", 17000, "/nerdit-dump/dump.pgdump")
    assert restore is not None
    assert restore[0] == "pg_restore"
    # One transaction + exit-on-error is what makes a failed live restore leave
    # the database exactly as it was (D-P37-6).
    assert {
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        "--single-transaction",
        "--exit-on-error",
    } <= set(restore)
    assert restore[-1] == "/nerdit-dump/dump.pgdump"  # positional archive, last


def test_redis_dump_argv_uses_rdb_and_no_auth_flag() -> None:
    """``redis-cli --rdb`` (D-P37-5) and never ``-a`` (that would be argv)."""
    argv = _REDIS_CASE.backend.dump_argv("172.17.0.1", 17000, "/nerdit-dump/dump.rdb")
    assert argv == [
        "redis-cli",
        "-h",
        "172.17.0.1",
        "-p",
        "17000",
        "--rdb",
        "/nerdit-dump/dump.rdb",
    ]
    assert "-a" not in argv
    assert "--pass" not in argv


@pytest.mark.parametrize("case", CASES)
def test_restore_argv_is_none_exactly_for_redis(case: _BackendCase) -> None:
    """``None`` means "file install under quiesce", never "unsupported" (D-P37-6).

    An RDB dropped into ``/data`` of an ``appendonly yes`` server is ignored —
    verified live, data-loss shaped — so Redis takes the quiesce + AOF-base
    branch and runs no sibling at all.
    """
    restore = case.backend.restore_argv("172.17.0.1", 17000, "/nerdit-dump/in")
    if case.name == "redis":
        assert restore is None
    else:
        assert restore is not None and restore[0] == case.restore_argv0
