"""Define Postgres and Redis container configuration and readiness probes.

Managed and external databases share app-facing connection env variables.
Backends provide container shapes, password-free display endpoints, launch-time
DSNs, and bounded wire probes; controllers own lifecycle.

The daemon stores passwords as scoped secrets and injects them at launch.
Official image entrypoints initialize credentials; the daemon runs no SQL
provisioning client. Containers use the daemon's UID:GID, computed at launch,
with all capabilities dropped and no privilege escalation. Data stays
daemon-owned for unprivileged backup, restore, and purge. A backend may instead
specify a fixed `run_as_user`.
"""

from __future__ import annotations

import asyncio
import os
import struct
from typing import Protocol, runtime_checkable

from nerdit.config.defaults import (
    DEFAULT_DB_READY_TIMEOUT_S,
    DEFAULT_POSTGRES_IMAGE,
    DEFAULT_REDIS_IMAGE,
)
from nerdit.core.runtime.container import ContainerConfig

# Container-side ports each backend listens on (fixed; the host port comes from
# the service_endpoints stable-port allocation, exactly like an app service).
POSTGRES_PORT = 5432
REDIS_PORT = 6379

# The single logical role + database every managed Postgres row exposes in v1
# (per-app roles / multiple logical DBs are post-P14.5 work). `nerdit` is a
# safe URL-userinfo token and a valid identifier.
MANAGED_DB_ROLE = "nerdit"
MANAGED_DB_NAME = "nerdit"

# The Postgres SSLRequest startup message (protocol 3): Int32 length (8) +
# Int32 request code (1234 << 16 | 5679 == 80877103). The postmaster answers a
# single byte 'S' (SSL available) or 'N' (not) even before it accepts logins —
# a liveness signal that requires a RESPONSE byte, so it also defeats the docker
# userland-proxy false-accept (a bare connect is accepted before the server is
# up; this probe is not).
_PG_SSL_REQUEST = struct.pack("!ii", 8, 80877103)

# The Redis inline PING command (protocol-agnostic: works before or after AUTH).
# A live server answers with a RESP reply whose first byte is one of the five
# RESP type markers — `+PONG` when unauthenticated is allowed, or
# `-NOAUTH Authentication required.` when a password is set (both count as
# ready). Like the Postgres probe this requires a RESPONSE byte, so it defeats
# the docker userland-proxy false-accept.
_REDIS_PING = b"PING\r\n"
_RESP_TYPE_BYTES = frozenset((b"+", b"-", b":", b"$", b"*"))


class DataProvisionError(Exception):
    """Raised when **provisioning** a managed database fails permanently.

    Permanent failures record `database.failed`; wire probes raise the transient
    DataNotReadyError subclass instead. Dump and restore failures use DumpError
    from nerdit.core.services and never settle the database row.
    """


class DataNotReadyError(DataProvisionError):
    """The database server is not answering the readiness probe yet.

    Distinct from a permanent failure: during a normal container start the
    server is still running `initdb` / WAL recovery, so the probe connects and
    resets (or times out). That is **transient** — the controller re-arms and
    retries on a later tick rather than marking the one bounded attempt spent,
    exactly like `nerdit.core.models.backend.ModelServerUnreachableError`.
    """


@runtime_checkable
class DataBackend(Protocol):
    """Interface for managed-data backends (Postgres first, Redis in P15.5).

    Implementations are stateless-per-call: `host`/`host_port` are passed in
    so one backend instance serves every database row. There is deliberately no
    `endpoint()` (unlike `ModelBackend`): a DSN needs a password that must
    never be persisted, so it is composed at binding-resolution/launch time only
    via `dsn`, while `public_endpoint` supplies a password-free
    display string for reads (D7). `requires_gpu` is deliberately absent —
    databases take no GPU parameter at all.
    """

    #: Backend id stored in `config['backend']` and shown by `GET /databases`.
    name: str
    #: `service_name` namespace + default instance name (`"pg"` | `"redis"`)
    #: so a database row can never collide across backends or with an app deploy.
    name_prefix: str
    #: Container-side port the server listens on (host port is allocated by the
    #: `service_endpoints` stable-port pool, backend-independent).
    container_port: int
    #: The scoped-secret key name the minted password is stored under — the
    #: image-native env name (`"POSTGRES_PASSWORD"` | `"REDIS_PASSWORD"`) so
    #: the ordinary per-service secret scope delivers it at launch (D-B (b)).
    minted_secret_key: str
    #: Default-binding env alias for this backend's scheme
    #: (`"DATABASE_URL"` | `"REDIS_URL"`).
    default_env_alias: str
    #: The P14a named-volume spec the create route stamps into
    #: `config['volumes']` so PGDATA rides the shipped named-volume machinery.
    volume_spec: str
    #: The user the container runs as. `None` (both v1
    #: backends) means the daemon's own `uid:gid`, composed at launch time by
    #: `container_config`, so the on-disk data tree is daemon-owned. A
    #: named string is reserved for a future backend needing a fixed in-image
    #: user.
    run_as_user: str | None
    #: P37 (D-P37-7): the logical format the dump tool writes, recorded verbatim
    #: in the dump manifest and compared to the row's backend on restore
    #: (``"pg_custom"`` | ``"rdb"``). A free-form label, not a file extension.
    dump_format: str
    #: P37 (D-P37-7): the fixed basename the dump tool writes inside the staging
    #: mount and the tar's first member (``"dump.pgdump"`` | ``"dump.rdb"``).
    #: Fixed per backend, never caller-supplied — the daemon composes the whole
    #: path from :func:`~nerdit.core.volumes.dump_staging_dir` (D-P37-2).
    dump_filename: str

    @property
    def image(self) -> str:
        """The container image this backend serves databases with."""
        ...

    def container_config(
        self, service_name: str, host_port: int, env: dict[str, str]
    ) -> ContainerConfig:
        """Build the launch config for a database container.

        *env* carries the minted credential (allowlisted by the launch branch to
        the backend's `minted_secret_key` — never the full secret scope); the
        backend adds its own image-native statics and sets `user` to
        `run_as_user or f"{os.getuid()}:{os.getgid()}"` (D-P15-7, daemon uid).
        """
        ...

    def dsn(self, host: str, host_port: int, password: str) -> str:
        """Compose the credential-bearing DSN (binding-resolution/launch time only)."""
        ...

    def public_endpoint(self, host: str, host_port: int) -> str:
        """Password-free `host:port` display string for `GET /databases`."""
        ...

    async def ensure_ready(self, host: str, host_port: int) -> None:
        """Bounded wire-protocol readiness probe.

        Raises `DataNotReadyError` while the server is still starting.
        """
        ...

    def dump_argv(self, host: str, port: int, out_path: str) -> list[str]:
        """Argv for the application-consistent dump, run in a sibling container built
        from the row's own image, dialing ``host:port`` (D-P37-1). *out_path* is
        inside the staged bind mount (D-P37-2). Secret-free by construction
        (D-P37-4): the password travels only in :meth:`dump_env`.
        """
        ...

    def dump_env(self, password: str) -> dict[str, str]:
        """The sibling's **entire** environment for a dump or a restore (D-P37-4):
        exactly one key, the tool's password variable. The caller passes the same
        *password* as ``scrub_values``; ``REDISCLI_AUTH`` matches no
        ``_CREDENTIAL_KEY_PARTS`` entry and that list is not widened (§7).
        """
        ...

    def restore_argv(self, host: str, port: int, in_path: str) -> list[str] | None:
        """Argv for the live restore, or ``None`` when the engine has no client-drivable
        load command (D-P37-6): ``None`` means "file install under quiesce",
        never "unsupported".
        """
        ...


class PostgresBackend:
    """The first `DataBackend`: an official `postgres` container.

    The image mints the role + database from `POSTGRES_USER`/`POSTGRES_DB`/
    `POSTGRES_PASSWORD` at first `initdb` (empty `PGDATA` only), so there
    is no SQL provisioning path. The launch env is **allowlisted** to exactly
    `{POSTGRES_USER, POSTGRES_DB, POSTGRES_PASSWORD, PGDATA}` — a user-set
    secret in the row's scope (e.g. `POSTGRES_HOST_AUTH_METHOD=trust`) can
    therefore never reach the container, so first-boot `pg_hba` stays the
    image's scram default.

    `PGDATA` points at a **subdirectory** of the mount so `initdb`'s
    `0700` data dir lives inside the world-writable P14a sticky volume leaf
    (the sticky bit permits the non-root `mkdir`; Postgres's perm check
    applies to `PGDATA` itself, not the mount parent).
    """

    name = "postgres"
    name_prefix = "pg"
    container_port = POSTGRES_PORT
    minted_secret_key = "POSTGRES_PASSWORD"
    default_env_alias = "DATABASE_URL"
    volume_spec = "data:/var/lib/postgresql/data"
    run_as_user: str | None = None
    # P37 (D-P37-7): ``--format=custom`` is pg_dump's compressed, selective,
    # version-tolerant archive — the only format ``pg_restore`` can drive with
    # ``--single-transaction``, which is what makes a failed restore leave the
    # database untouched (D-P37-6). ``.pgdump`` is a bare label; the archive is
    # self-describing and the manifest carries ``format`` anyway.
    dump_format = "pg_custom"
    dump_filename = "dump.pgdump"

    # `initdb`'s 0700 data dir, kept as a SUBDIR of the mount (never the mount
    # root, whose sticky-bit `0o1777` perms Postgres would reject).
    _PGDATA = "/var/lib/postgresql/data/pgdata"

    def __init__(
        self,
        *,
        image: str = DEFAULT_POSTGRES_IMAGE,
        ready_timeout_s: float = DEFAULT_DB_READY_TIMEOUT_S,
    ) -> None:
        self._image = image
        self._ready_timeout_s = ready_timeout_s

    @property
    def image(self) -> str:
        return self._image

    def container_config(
        self, service_name: str, host_port: int, env: dict[str, str]
    ) -> ContainerConfig:
        """Launch config for the Postgres server (one server = one logical db).

        The image-native statics are owned here; *env* supplies only the minted
        `POSTGRES_PASSWORD` (the launch branch allowlists it). The union is
        EXACTLY the four allowlisted keys — a stray key in *env* would ride
        through, which is why the launch branch, not this shape builder, is
        responsible for restricting *env* to the minted credential (§1.3).
        """
        full_env = {
            "POSTGRES_USER": MANAGED_DB_ROLE,
            "POSTGRES_DB": MANAGED_DB_NAME,
            "PGDATA": self._PGDATA,
            **env,
        }
        return ContainerConfig(
            image=self._image,
            gpu_ids=[],
            env=full_env,
            ports={self.container_port: host_port},
            # D-P15-7: composed at launch time so a daemon restarted under a
            # different uid stamps the CURRENT one (never import time).
            user=self.run_as_user or f"{os.getuid()}:{os.getgid()}",
        )

    def dsn(self, host: str, host_port: int, password: str) -> str:
        """`postgresql://nerdit:<password>@host:port/nerdit` (managed shape).

        The managed password is minted as `secrets.token_hex(32)` — pure hex,
        so no URL-encoding hazard. External DSNs (with user-chosen passwords)
        are composed by the binding resolver, not here.
        """
        return f"postgresql://{MANAGED_DB_ROLE}:{password}@{host}:{host_port}/{MANAGED_DB_NAME}"

    def public_endpoint(self, host: str, host_port: int) -> str:
        """Password-free `host:port` display string (never a DSN)."""
        return f"{host}:{host_port}"

    def dump_argv(self, host: str, port: int, out_path: str) -> list[str]:
        """``pg_dump --format=custom`` into the staging mount (D-P37-1/4). ``-U`` is
        mandatory (the daemon's uid has no passwd entry in the image, D-P37-3);
        ``--no-owner``/``--no-privileges`` keep cross-name restore possible
        (D-P37-10); ``--file`` so the artifact never rides the log stream.
        """
        return [
            "pg_dump",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            MANAGED_DB_ROLE,
            "-d",
            MANAGED_DB_NAME,
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            "--file",
            out_path,
        ]

    def dump_env(self, password: str) -> dict[str, str]:
        """The sibling's **entire** environment for a dump or a restore (D-P37-4):
        exactly one key, the tool's password variable. The caller passes the same
        *password* as ``scrub_values``; ``REDISCLI_AUTH`` matches no
        ``_CREDENTIAL_KEY_PARTS`` entry and that list is not widened (§7).
        """
        return {"PGPASSWORD": password}

    def restore_argv(self, host: str, port: int, in_path: str) -> list[str] | None:
        """``pg_restore --clean --if-exists --single-transaction --exit-on-error`` into
        the running database (D-P37-6): one transaction, a failure rolls back
        whole, post-dump objects survive, no CASCADE. Same ``-U`` and
        ``--no-owner``/``--no-privileges`` as :meth:`dump_argv`.
        """
        return [
            "pg_restore",
            "-h",
            host,
            "-p",
            str(port),
            "-U",
            MANAGED_DB_ROLE,
            "-d",
            MANAGED_DB_NAME,
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
            "--single-transaction",
            "--exit-on-error",
            in_path,
        ]

    async def ensure_ready(self, host: str, host_port: int) -> None:
        """Probe readiness with the Postgres SSLRequest handshake (bounded).

        Open a connection, send the 8-byte SSLRequest, and expect exactly one
        byte `S` or `N`. Anything else — connect refused, reset, timeout, a
        short read, or an unexpected byte — is transient
        (`DataNotReadyError`): the controller retries on a later tick.

        Known softness (accepted, documented §1.3): the postmaster answers
        SSLRequest while still in WAL crash-recovery, so this can stamp
        `db_ready` moments before logins succeed — bound apps retry.
        """
        writer = None
        try:
            # One whole-attempt budget (§1.3, locked as PER-ATTEMPT): connect +
            # send + read share a single deadline, so an attempt can never take
            # up to 3× `ready_timeout_s`.
            async with asyncio.timeout(self._ready_timeout_s):
                reader, writer = await asyncio.open_connection(host, host_port)
                writer.write(_PG_SSL_REQUEST)
                await writer.drain()
                reply = await reader.readexactly(1)
        except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            raise DataNotReadyError(
                f"postgres at {host}:{host_port} not ready: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
        if reply not in (b"S", b"N"):
            raise DataNotReadyError(
                f"postgres at {host}:{host_port} answered an unexpected SSLRequest byte {reply!r}"
            )


class RedisBackend:
    """P15.5 `DataBackend`: an official `redis` container.

    This backend is the **registry proof** (the vLLM role in the model plane):
    it slots in as an `extra_backends` entry with zero controller, route or CLI
    change — the `DataController` registry, `POST /databases` write path,
    `GET /databases`, `/capabilities` and the CLI/MCP surface are all
    backend-name-driven, so adding Redis is one settings image key + this class.

    Credentials are minted image-natively: Redis takes
    `--requirepass`, but the official entrypoint reads no `REDIS_PASSWORD`
    env, so the launch `command` expands the variable **inside the container**
    via `sh -c` — host-visible argv / `docker inspect` carry only the
    unexpanded `"$REDIS_PASSWORD"` name, never the value.

    The `sh -c` override bypasses the entrypoint's own root→`redis`
    step-down branch, so **without D-P15-6/D-P15-7 this would silently run Redis
    as root**. With the daemon-uid `user` the shell starts non-root, the env
    expansion works unchanged, and the AOF files land daemon-owned under the
    P14a `0o1777` sticky volume leaf.
    """

    name = "redis"
    name_prefix = "redis"
    container_port = REDIS_PORT
    minted_secret_key = "REDIS_PASSWORD"
    default_env_alias = "REDIS_URL"
    volume_spec = "data:/data"
    run_as_user: str | None = None
    # P37 (D-P37-5): a fresh point-in-time RDB pulled over the replication
    # protocol is the only *logical*, application-consistent capture Redis
    # offers — there is no ``pg_dump`` equivalent, and copying the live AOF dir
    # is a physical capture (which is what the P15 volume tar already does).
    dump_format = "rdb"
    dump_filename = "dump.rdb"

    # `exec` replaces the shell so redis-server is PID 1 (clean signal
    # forwarding); `--requirepass "$REDIS_PASSWORD"` expands inside the
    # container only; `--appendonly yes` backs the volume backup story (WP7 —
    # Redis reloads its AOF on restart, improving live-tar recoverability, though
    # a stop-first capture is still the guaranteed-clean one).
    _COMMAND = ["sh", "-c", 'exec redis-server --requirepass "$REDIS_PASSWORD" --appendonly yes']

    def __init__(
        self,
        *,
        image: str = DEFAULT_REDIS_IMAGE,
        ready_timeout_s: float = DEFAULT_DB_READY_TIMEOUT_S,
    ) -> None:
        self._image = image
        self._ready_timeout_s = ready_timeout_s

    @property
    def image(self) -> str:
        return self._image

    def container_config(
        self, service_name: str, host_port: int, env: dict[str, str]
    ) -> ContainerConfig:
        """Launch config for the Redis server (`--requirepass` via `sh -c`).

        *env* supplies only the minted `REDIS_PASSWORD` (the launch branch
        allowlists it); there are no image-native statics for Redis, so the
        container env is EXACTLY `{REDIS_PASSWORD}` and the password reaches
        redis-server only through the in-container variable expansion.
        """
        return ContainerConfig(
            image=self._image,
            gpu_ids=[],
            env=dict(env),
            ports={self.container_port: host_port},
            command=list(self._COMMAND),
            # D-P15-7: composed at launch time so a daemon restarted under a
            # different uid stamps the CURRENT one (never import time).
            user=self.run_as_user or f"{os.getuid()}:{os.getgid()}",
        )

    def dsn(self, host: str, host_port: int, password: str) -> str:
        """`redis://:<password>@host:port/0` (managed shape).

        The managed password is minted as `secrets.token_hex(32)` — pure hex,
        so no URL-encoding hazard. External DSNs (with user-chosen passwords)
        are composed by the binding resolver, not here.
        """
        return f"redis://:{password}@{host}:{host_port}/0"

    def public_endpoint(self, host: str, host_port: int) -> str:
        """Password-free `host:port` display string (never a DSN)."""
        return f"{host}:{host_port}"

    def dump_argv(self, host: str, port: int, out_path: str) -> list[str]:
        """``redis-cli --rdb`` — the client acts as a one-shot replica and receives a
        fresh RDB (D-P37-5). No ``-a``: the password rides ``REDISCLI_AUTH``
        (D-P37-4).
        """
        return ["redis-cli", "-h", host, "-p", str(port), "--rdb", out_path]

    def dump_env(self, password: str) -> dict[str, str]:
        """The sibling's **entire** environment for a dump or a restore (D-P37-4):
        exactly one key, the tool's password variable. The caller passes the same
        *password* as ``scrub_values``; ``REDISCLI_AUTH`` matches no
        ``_CREDENTIAL_KEY_PARTS`` entry and that list is not widened (§7).
        """
        return {"REDISCLI_AUTH": password}

    def restore_argv(self, host: str, port: int, in_path: str) -> list[str] | None:
        """``None`` — Redis restores by file install under quiesce (D-P37-6): an RDB
        dropped beside an ``appendonly yes`` server is ignored (verified live),
        so the controller installs it as the AOF base instead.
        """
        return None

    async def ensure_ready(self, host: str, host_port: int) -> None:
        """Probe readiness with an inline `PING` (bounded).

        Open a connection, send `PING\\r\\n`, and expect a RESP reply whose
        first byte is a RESP type marker (`+PONG` or `-NOAUTH ...` both
        count as ready). Anything else — connect refused, reset, timeout, a short
        read, or a non-RESP byte — is transient (`DataNotReadyError`): the
        controller retries on a later tick.
        """
        writer = None
        try:
            # One whole-attempt budget (§1.3, locked as PER-ATTEMPT): connect +
            # send + read share a single deadline, so an attempt can never take
            # up to 3× `ready_timeout_s`.
            async with asyncio.timeout(self._ready_timeout_s):
                reader, writer = await asyncio.open_connection(host, host_port)
                writer.write(_REDIS_PING)
                await writer.drain()
                reply = await reader.readexactly(1)
        except (OSError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            raise DataNotReadyError(
                f"redis at {host}:{host_port} not ready: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if writer is not None:
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
        if reply not in _RESP_TYPE_BYTES:
            raise DataNotReadyError(
                f"redis at {host}:{host_port} answered an unexpected reply byte {reply!r}"
            )
