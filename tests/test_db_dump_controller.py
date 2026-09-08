"""(P37 WP2) The controller tier of managed-database dumps and restores.

Contract freeze for :meth:`~nerdit.core.services.ServiceController.dump_database`,
:meth:`~nerdit.core.services.ServiceController.restore_database` and
:meth:`~nerdit.core.services.ServiceController.quiesce_row`, plus the additive
``RunMode`` widening of the P20 run registry that carries them (D-P37-9).

Controller tier throughout — the real in-memory ``queries`` fixture, the shared
``FakeRuntime`` (extended here into a runtime that behaves like the dump tool:
it writes the file the argv names), a real ``ServiceController``, a real
``SecretManager``. No HTTP: the route half is WP4.

What is pinned here, and why each of these is a hazard rather than a detail:

* the sibling's container shape — the P20 run shape plus exactly two changes
  (the daemon's uid:gid, and the staging bind mount as the ONLY mount), because
  a mount that is not the staging dir is a database volume the dump has no
  business touching, and a published port would contend with the live server;
* the argv is secret-free and the env is EXACTLY ``dump_env(password)``
  (D-P37-4): the argv is persisted verbatim in ``config['last_dump']`` and is
  visible to ``docker inspect``;
* the scrub really covers this path — the password must not survive in the
  tail, in ``last_dump`` or in the error;
* every failure leaves NOTHING (D-P37-11): staging removed, slot released;
* the three caps compose as D-P37-9 says — a dump excluded from the run cap and
  a run excluded from the dump cap, both single-flight per row;
* the Redis restore's ORDER (D-P37-6): the replacement AOF dir is complete
  before the row is quiesced, both renames happen, the previous data is kept,
  and the desired state comes back to ``running`` on every path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import time
from pathlib import Path
from typing import NamedTuple

import pytest

from nerdit.core.data.backend import DataBackend, PostgresBackend, RedisBackend
from nerdit.core.secrets import SecretManager
from nerdit.core.services import (
    DumpError,
    RunMode,
    RunPreconditionError,
    RunSlot,
    ServiceController,
)
from nerdit.core.volumes import VolumeSpecError, dump_staging_dir, dump_staging_root
from nerdit.db.models import JobStatus
from tests.test_run_primitive import _PW, _FakeData, _serve_managed_db, _vol_controller
from tests.test_services_reconcile import FakeRuntime, _controller, _deploy_svc, _svc

_BRIDGE = "172.17.0.1"
_MOUNT = "/nerdit-dump"


class _DumpRuntime(FakeRuntime):
    """A ``FakeRuntime`` that behaves like the dump tool it is standing in for.

    The real ``pg_dump`` / ``redis-cli --rdb`` write the file the last argv
    element names, inside the one bind mount. Without that the success path
    would always fail its output verification and every "it packed" assertion
    would be testing the refusal branch instead.

    The seams model the failure modes the plan calls out by name: ``payload``
    empty reproduces the 0-byte file ``pg_dump`` leaves when the connection is
    refused (it creates its output BEFORE it connects); ``as_symlink`` plants
    the link a hostile image could put at the output path; ``extra`` drops a
    second file in the staging dir.
    """

    def __init__(
        self,
        *,
        payload: bytes = b"PGDMP-payload",
        as_symlink: bool = False,
        extra: str | None = None,
        echo: list[str] | None = None,
        write_output: bool = True,
    ) -> None:
        super().__init__()
        self.exit_code = 0
        self.payload = payload
        self.as_symlink = as_symlink
        self.extra = extra
        self.write_output = write_output
        self.log_lines = list(echo or [])

    async def run(self, config) -> str:
        cid = await super().run(config)
        if not self.write_output or not config.volumes:
            return cid
        staging = Path(next(h for h, c in config.volumes.items() if c == _MOUNT))
        out = staging / Path(config.command[-1]).name
        if self.as_symlink:
            out.symlink_to(staging / "elsewhere")
        else:
            out.write_bytes(self.payload)
        if self.extra:
            (staging / self.extra).write_bytes(b"x")
        return cid


def _pg_controller(queries, runtime, tmp_path, secrets, **kw) -> ServiceController:
    """A controller with every dump seam wired: data plane, data_dir, secrets."""
    return _vol_controller(
        queries,
        runtime,
        tmp_path=tmp_path,
        secrets=secrets,
        data_controller=_FakeData(PostgresBackend(), _BRIDGE),
        service_port_range="9500-9599",
        **kw,
    )


def _redis_controller(queries, runtime, tmp_path, secrets, **kw) -> ServiceController:
    return _vol_controller(
        queries,
        runtime,
        tmp_path=tmp_path,
        secrets=secrets,
        data_controller=_FakeData(RedisBackend(), _BRIDGE),
        service_port_range="9500-9599",
        **kw,
    )


class _Engine(NamedTuple):
    """One managed engine, in the shape the per-engine parametrized tests need."""

    backend: DataBackend
    name: str
    backend_key: str
    port: int
    secret_key: str
    payload: bytes


#: Anything asserted for one engine and not the other is a half-tested contract —
#: the scrub especially, since ``REDISCLI_AUTH`` matches no
#: ``_CREDENTIAL_KEY_PARTS`` entry (which D-P37-4 forbids widening) while
#: ``PGPASSWORD`` does, so the two reach the scrub by different routes and only
#: the explicit ``scrub_values`` set covers both.
_ENGINES = [
    pytest.param(
        _Engine(PostgresBackend(), "pg", "postgres", 5432, "POSTGRES_PASSWORD", b"PGDMP-payload"),
        id="postgres",
    ),
    pytest.param(
        _Engine(RedisBackend(), "cache", "redis", 6379, "REDIS_PASSWORD", b"REDIS0011"),
        id="redis",
    ),
]


def _engine_controller(queries, runtime, tmp_path, engine: _Engine) -> ServiceController:
    """A dump-wired controller for *engine*, with that row's minted credential."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(engine.name, {engine.secret_key: _PW})
    return _vol_controller(
        queries,
        runtime,
        tmp_path=tmp_path,
        secrets=mgr,
        data_controller=_FakeData(engine.backend, _BRIDGE),
        service_port_range="9500-9599",
    )


def _pg_secrets(tmp_path) -> SecretManager:
    """The row's minted credential, at rest exactly as the daemon stores it."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("pg", {"POSTGRES_PASSWORD": _PW})
    return mgr


def _redis_secrets(tmp_path) -> SecretManager:
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("cache", {"REDIS_PASSWORD": _PW})
    return mgr


async def _last_dump(queries, job_id: str) -> dict:
    row = await queries.get_job(job_id)
    return json.loads(row.config)["last_dump"]


# --- the sibling's shape (D-P37-1/2/3/4) --------------------------------------


async def test_dump_sibling_is_the_run_shape_plus_uid_and_one_mount(queries, tmp_path):
    """The two deliberate deltas from a P20 run container, and nothing else.

    ``user`` is the daemon's CURRENT uid:gid so the artifact is daemon-owned
    (D-P37-3), and the staging dir is the ONLY mount (D-P37-2) — the database's
    own volume is deliberately absent, because a logical dump goes through the
    server, not around it. Everything else (portless, GPU-less, cap_drop,
    no_new_privileges, forced bridge) is the run shape verbatim.
    """
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    endpoint = await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")

    result = await controller.dump_database(job, run_id="dumpaaaaaaaa", timeout_s=30)

    config = runtime.run_configs[-1]
    assert config.user == f"{os.getuid()}:{os.getgid()}"
    staging = dump_staging_dir(tmp_path, "dumpaaaaaaaa")
    assert config.volumes == {str(staging): _MOUNT}
    assert config.ports is None
    assert config.gpu_ids == []
    assert config.cap_drop == ["ALL"]
    assert config.no_new_privileges is True
    assert config.network_mode == "bridge"
    assert config.image == "postgres:latest"  # the ROW's image, never a default
    # The artifact is left in place for the route's packer, which owns removing
    # the staging dir (create_dump_backup rmtree's it in its own finally).
    assert result.output_path == str(staging / "dump.pgdump")
    assert Path(result.output_path).read_bytes() == b"PGDMP-payload"
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700
    # The sibling dials the RESERVED host port, never the container-side one.
    assert endpoint.host_port != 5432
    await controller.shutdown()


async def test_dump_argv_is_secret_free_and_env_is_exactly_the_one_credential(queries, tmp_path):
    """(D-P37-4) The password travels in the env and NOWHERE else.

    The argv is persisted verbatim in ``config['last_dump']`` and is readable
    through ``docker inspect``, so a credential on it would be at rest in two
    places the daemon cannot scrub. The env is the backend's ``dump_env``
    EXACTLY — no resolved launch env, no ``PORT``, no ``NERDIT_RUN_ID``: a
    sibling the daemon spawns on the operator's behalf carries one secret and
    one only.
    """
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    endpoint = await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")

    await controller.dump_database(job, run_id="dumpbbbbbbbb", timeout_s=30)

    config = runtime.run_configs[-1]
    assert config.env == {"PGPASSWORD": _PW}
    assert _PW not in " ".join(config.command)
    assert config.command == PostgresBackend().dump_argv(
        _BRIDGE, endpoint.host_port, f"{_MOUNT}/dump.pgdump"
    )
    assert "-U" in config.command  # libpq cannot resolve the sibling's uid
    stamped = await _last_dump(queries, job.id)
    assert stamped["command"] == config.command
    assert _PW not in json.dumps(stamped)


@pytest.mark.parametrize("engine", _ENGINES)
async def test_dump_scrubs_the_password_out_of_the_tail_and_last_dump(queries, tmp_path, engine):
    """(D-P20-1 reused) A tool that echoes its own env must not publish it.

    ``redis-cli`` prints a warning naming what it read; a chatty image could
    print more, and ``pg_dump``/``pg_restore`` echo connection diagnostics. The
    tail reaches the response hint and ``config['last_dump']`` at rest, so the
    ONE scrub choke point in ``_execute_container_once`` has to cover this path
    with the value the controller resolved.

    Both engines, deliberately: the scrub is driven by the explicit
    ``scrub_values={password}`` set rather than by env-key heuristics, so a
    name-driven regression would be invisible on one leg alone.
    """
    runtime = _DumpRuntime(payload=engine.payload, echo=[f"connecting with {_PW}", "done"])
    controller = _engine_controller(queries, runtime, tmp_path, engine)
    endpoint = await _serve_managed_db(
        queries, name=engine.name, backend_key=engine.backend_key, port=engine.port
    )
    job = await queries.get_service_by_name(engine.name)

    result = await controller.dump_database(job, run_id="dumpcccccccc", timeout_s=30)

    assert result.log_tail == ["connecting with ***", "done"]
    stamped = await _last_dump(queries, job.id)
    assert stamped["log_tail"] == ["connecting with ***", "done"]
    assert _PW not in json.dumps(stamped)
    assert runtime.run_configs[-1].command == engine.backend.dump_argv(
        _BRIDGE, endpoint.host_port, f"{_MOUNT}/{engine.backend.dump_filename}"
    )


# --- failure leaves nothing (D-P37-11) ---------------------------------------


async def _failing_dump(queries, tmp_path, runtime, run_id: str = "dumpddddddddd"[:12]):
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    with pytest.raises(DumpError) as excinfo:
        await controller.dump_database(job, run_id=run_id, timeout_s=30)
    return controller, job, excinfo.value


async def test_a_nonzero_exit_removes_the_staging_dir_and_stamps_the_failure(queries, tmp_path):
    """The whole D-P37-11 contract on one path: reason, tail, no residue, no slot."""
    runtime = _DumpRuntime(write_output=False, echo=["FATAL:  password authentication failed"])
    runtime.exit_code = 1
    controller, job, exc = await _failing_dump(queries, tmp_path, runtime)

    assert exc.reason == "exit_nonzero"
    assert exc.log_tail == ["FATAL:  password authentication failed"]
    assert not dump_staging_dir(tmp_path, "dumpdddddddd").exists()
    assert list(dump_staging_root(tmp_path).iterdir()) == []
    assert controller.has_active_run(job.id) is False
    stamped = await _last_dump(queries, job.id)
    assert stamped["reason"] == "exit_nonzero"
    assert stamped["kind"] == "dump"
    assert stamped["exit_code"] == 1
    assert stamped["dump"] is None  # the tar does not exist yet; the route restamps


async def test_a_timeout_is_its_own_reason(queries, tmp_path):
    runtime = _DumpRuntime(write_output=False)
    runtime.wait_error = TimeoutError()
    _, job, exc = await _failing_dump(queries, tmp_path, runtime)
    assert exc.reason == "timed_out"
    assert (await _last_dump(queries, job.id))["timed_out"] is True


async def test_an_empty_output_is_refused_rather_than_packed(queries, tmp_path):
    """``pg_dump`` creates its file BEFORE connecting, so a refused connection
    leaves exactly 0 bytes — packing that would ship an unusable "backup"."""
    _, _, exc = await _failing_dump(queries, tmp_path, _DumpRuntime(payload=b""))
    assert exc.reason == "empty_output"
    assert not dump_staging_root(tmp_path).joinpath("dumpdddddddd").exists()


async def test_a_symlink_planted_at_the_output_path_is_never_followed(queries, tmp_path):
    """A process in the sibling can plant a symlink inside the bind mount
    (verified live, plan §0). The verifier ``lstat``s and refuses."""
    _, _, exc = await _failing_dump(queries, tmp_path, _DumpRuntime(as_symlink=True))
    assert exc.reason == "output_not_regular"


async def test_an_extra_file_in_staging_fails_loudly(queries, tmp_path):
    """Staging is packed wholesale, so an unexpected entry would ride along."""
    _, _, exc = await _failing_dump(queries, tmp_path, _DumpRuntime(extra="core.dump"))
    assert exc.reason == "unexpected_output"


async def test_a_symlinked_staging_root_is_refused_before_it_is_created_or_chmoded(
    queries, tmp_path
):
    """(D-P37-2) The single owner of the grammar refuses FIRST.

    ``mkdir(exist_ok=True)`` succeeds against a symlink-to-directory and
    ``os.chmod`` follows the link, so creating the root before validating it
    would ``chmod 0o700`` whatever an attacker pointed ``dump-staging`` at —
    a permission change on someone else's directory, made by the daemon,
    before the refusal it was entitled to.
    """
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    target = tmp_path / "elsewhere"
    target.mkdir()
    os.chmod(target, 0o755)
    dump_staging_root(tmp_path).symlink_to(target)

    with pytest.raises(VolumeSpecError):
        await controller.dump_database(job, run_id="dumpgggggggg", timeout_s=30)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755  # never chmod'ed through the link
    assert list(target.iterdir()) == []
    assert runtime.run_configs == []
    assert controller.has_active_run(job.id) is False


async def test_a_controller_without_a_data_plane_refuses_before_claiming_a_slot(queries):
    """``self._data is None`` is the ONE precondition the controller owns."""
    controller = _controller(queries, FakeRuntime())
    job = _svc("pg")
    await queries.create_job(job)
    with pytest.raises(DumpError) as excinfo:
        await controller.dump_database(job, run_id="dumpeeeeeeee", timeout_s=30)
    assert excinfo.value.reason == "data_plane_unavailable"
    assert controller.has_active_run(job.id) is False


# --- the packer hand-off runs under the slot (D-P37-9) -----------------------


async def test_the_captured_hand_off_runs_while_the_slot_is_still_held(queries, tmp_path):
    """(D-P37-9) The route's packer is awaited BEFORE the slot is released.

    Packing hashes and gzips the artifact on a worker thread, so the loop serves
    other requests for its whole duration. Released first, the slot would leave
    ``DELETE /services/{name}``, a second dump and the restart drain — all of
    which read this registry, and all of which the surface advertises as covering
    the operation — open across the disk-heaviest half of the work, and would put
    the packing phase outside ``[services].max_concurrent_dumps`` entirely.
    """
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    seen: list[object] = []

    async def _pack(captured):
        seen.append(controller.has_active_run(job.id))
        # The artifact is still where the controller left it: the packer, not
        # the ``finally``, owns removing the staging dir on the success path.
        seen.append(Path(captured.output_path).read_bytes())

    result = await controller.dump_database(
        job, run_id="dumphhhhhhhh", timeout_s=30, on_captured=_pack
    )

    assert seen == [True, b"PGDMP-payload"]
    assert controller.has_active_run(job.id) is False
    assert result.output_path == str(dump_staging_dir(tmp_path, "dumphhhhhhhh") / "dump.pgdump")
    # A success is still a success: the hand-off returning cleanly changes
    # nothing about the record the controller stamps.
    assert (await _last_dump(queries, job.id))["reason"] is None
    await controller.shutdown()


async def test_a_failing_hand_off_releases_the_slot_and_records_no_outcome(queries, tmp_path):
    """The packer raised: no artifact survives, so the row must not say success.

    ``reason: None`` beside ``dump: None`` is exactly the shape of a FAILED dump
    on ``/diagnose`` — a run that produced nothing — so leaving the optimistic
    record in place would make the two indistinguishable. ``interrupted`` is the
    D-P37-11 floor ("no outcome was observed"); the route re-stamps its packer's
    own token once the exception reaches it.
    """
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")

    async def _pack(_captured):
        raise RuntimeError("the packer refused")

    with pytest.raises(RuntimeError):
        await controller.dump_database(job, run_id="dumpiiiiiiii", timeout_s=30, on_captured=_pack)

    assert controller.has_active_run(job.id) is False
    stamped = await _last_dump(queries, job.id)
    assert stamped["reason"] == "interrupted"
    assert stamped["dump"] is None
    assert stamped["exit_code"] == 0  # the TOOL exited 0; the daemon lost its output
    assert stamped["log_tail"] == []
    # Nothing is left behind: the packer owns the staging dir on the success
    # path, so when it fails the controller's own ``finally`` clears it.
    assert list(dump_staging_root(tmp_path).iterdir()) == []
    await controller.shutdown()


# --- the registry widening (D-P37-9) -----------------------------------------


def test_the_is_release_spelling_still_selects_the_release_mode():
    """(D-P37-9) The widening is additive: ~45 existing ``is_release=`` call
    sites, ``core/app_build.py`` included, must keep meaning what they meant."""
    assert RunSlot(is_release=True).mode is RunMode.release
    assert RunSlot(is_release=True).is_release is True
    assert RunSlot().mode is RunMode.run
    assert RunSlot().is_release is False
    assert RunSlot(mode=RunMode.dump).is_release is False


async def test_a_dump_is_single_flight_per_row(queries, tmp_path):
    """Two dumps of one database would both stream it and both write the disk."""
    runtime = _DumpRuntime()
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    controller._register_run(job.id, "holdaaaaaaaa", mode=RunMode.dump)

    with pytest.raises(RunPreconditionError) as excinfo:
        await controller.dump_database(job, run_id="dumpffffffff", timeout_s=30)

    assert excinfo.value.reason == "run_in_progress"
    assert runtime.run_configs == []  # refused before any container


async def test_dumps_have_their_own_cap_and_do_not_count_against_runs(queries, tmp_path):
    """(D-P37-9) Two separate pools. A dump neither consumes a run slot nor is
    bounded by one: they contend for different resources (a run for the image's
    CPU, a dump for daemon disk and the database's replication bandwidth), and
    letting either starve the other would make an operator's backup fail
    because an agent happened to be running a migration elsewhere."""
    controller = _pg_controller(
        queries,
        _DumpRuntime(),
        tmp_path,
        _pg_secrets(tmp_path),
        max_concurrent_runs=1,
        max_concurrent_dumps=1,
    )
    controller._register_run("job-a", "dumpaaaaaaaa", mode=RunMode.dump)

    # The dump cap is full — a second dump, on ANOTHER row, is refused.
    with pytest.raises(RunPreconditionError) as excinfo:
        controller._register_run("job-b", "dumpbbbbbbbb", mode=RunMode.dump)
    assert excinfo.value.reason == "too_many_dumps"

    # ...but the run cap is untouched by it.
    controller._register_run("job-b", "runbbbbbbbbb", is_release=False)
    # ...and the run now holding the only run slot does not close the dump pool
    # for a row that has none (the dump above is what does).
    controller._discard_run("job-a", "dumpaaaaaaaa")
    controller._register_run("job-c", "dumpccccccc1", mode=RunMode.dump)


async def test_a_release_still_bypasses_every_cap(queries, tmp_path):
    """The P20 exemption survives the widening: a deploy must never fail
    because unrelated runs or dumps are in flight."""
    controller = _pg_controller(
        queries,
        _DumpRuntime(),
        tmp_path,
        _pg_secrets(tmp_path),
        max_concurrent_runs=1,
        max_concurrent_dumps=1,
    )
    controller._register_run("job-a", "dumpaaaaaaaa", mode=RunMode.dump)
    controller._register_run("job-a", "relaaaaaaaaa", is_release=True)
    assert controller.has_active_run("job-a") is True


# --- restore: Postgres runs the sibling (D-P37-6) -----------------------------


def _seed_staging(tmp_path: Path, run_id: str, filename: str, payload: bytes = b"x") -> Path:
    """Stand in for the route's ``extract_dump_tar``: a populated staging dir."""
    staging = dump_staging_dir(tmp_path, run_id)
    staging.mkdir(parents=True, mode=0o700)
    (staging / filename).write_bytes(payload)
    return staging


async def test_postgres_restore_runs_the_sibling_and_consumes_the_staging_dir(queries, tmp_path):
    """``pg_restore`` into the RUNNING database: no quiesce, one transaction,
    and the extracted payload is spent by the time the method returns."""
    runtime = _DumpRuntime(write_output=False)
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    endpoint = await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    staging = _seed_staging(tmp_path, "restaaaaaaaa", "dump.pgdump")

    result = await controller.restore_database(
        job, run_id="restaaaaaaaa", staging=staging, timeout_s=30
    )

    assert runtime.run_configs[-1].command == PostgresBackend().restore_argv(
        _BRIDGE, endpoint.host_port, f"{_MOUNT}/dump.pgdump"
    )
    assert runtime.run_configs[-1].env == {"PGPASSWORD": _PW}
    assert result.output_path is None
    assert not staging.exists()
    stamped = await _last_dump(queries, job.id)
    assert stamped["kind"] == "restore"
    assert stamped["reason"] is None
    # The row is never stopped for a Postgres restore.
    assert (await queries.get_job(job.id)).desired_state == "running"


async def test_a_failed_postgres_restore_still_removes_the_payload_and_scrubs_its_tail(
    queries, tmp_path
):
    """The restore leg of the scrub, on the path that actually publishes a tail.

    A failed restore is exactly where the tail travels furthest — it becomes the
    500's ``hint`` and is stamped in ``config['last_dump']`` at rest — so the
    same ``scrub_values`` coverage the dump has is asserted here too. (The Redis
    restore runs no container at all, so it has no tail to scrub: it is a file
    install under a quiesce, pinned below.)
    """
    runtime = _DumpRuntime(
        write_output=False,
        echo=[f"pg_restore: connecting to database with {_PW}", "ERROR:  relation does not exist"],
    )
    runtime.exit_code = 1
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")
    staging = _seed_staging(tmp_path, "restbbbbbbbb", "dump.pgdump")

    with pytest.raises(DumpError) as excinfo:
        await controller.restore_database(job, run_id="restbbbbbbbb", staging=staging, timeout_s=30)

    assert excinfo.value.reason == "exit_nonzero"
    assert excinfo.value.log_tail == [
        "pg_restore: connecting to database with ***",
        "ERROR:  relation does not exist",
    ]
    stamped = await _last_dump(queries, job.id)
    assert stamped["log_tail"] == excinfo.value.log_tail
    assert _PW not in json.dumps(stamped)
    assert not staging.exists()
    assert controller.has_active_run(job.id) is False


# --- restore: Redis installs an AOF base under a quiesce (D-P37-6) ------------


async def _redis_row(queries, tmp_path, name: str = "cache"):
    """A ready managed-Redis row that declares its ``data`` named volume."""
    await _serve_managed_db(queries, name=name, backend_key="redis", port=6379)
    job = await queries.get_service_by_name(name)
    cfg = json.loads(job.config)
    cfg["volumes"] = [RedisBackend.volume_spec]
    await queries.update_job_config(job.id, json.dumps(cfg))
    volume_dir = tmp_path / "services" / name / "data"
    volume_dir.mkdir(parents=True)
    return await queries.get_job(job.id), volume_dir


class _FakeReconciler:
    """The convergence the real reconcile loop provides, in ~20 lines.

    ``quiesce_row`` drives the row through the ORDINARY desired-state
    machinery, so a controller-tier test has to stand in for the loop that
    converges it. Faithful to production on the three points the quiesce
    depends on, because a fake that is kind on any of them would let a broken
    wait pass:

    * the STOP does not touch ``config`` at all — ``_teardown_to_stopped``
      (``core/services.py``) destroys the container, releases the endpoint and
      flips the status, and never clears ``db_ready``. So a quiesced row really
      does still carry the previous container's ``True``, and a ready wait that
      looked at the flag alone would return before anything had restarted;
    * the RELAUNCH clears ``db_ready`` in the same breath as the flip to
      ``running`` — that is the P15/C5 clear in ``core/launch.py::on_launched``,
      which is awaited BEFORE the status flip precisely so no
      ``running ∧ stale-flag`` state is observable;
    * the readiness probe sets the flag back only some polls LATER
      (``ready_delay_polls``), the window in which a Redis is replaying the AOF
      base this restore just installed — up, but not yet a database. A ready
      wait that looked at the status alone would return inside it.

    ``events`` records that sequence so a test can assert the quiesce returned
    after the post-relaunch flag, not on the stale one.
    """

    def __init__(
        self,
        queries,
        job_id: str,
        *,
        converge_stop: bool = True,
        ready_delay_polls: int = 2,
    ) -> None:
        self._queries = queries
        self._job_id = job_id
        self._converge_stop = converge_stop
        self._ready_delay_polls = ready_delay_polls
        self._polls_since_relaunch = 0
        self.events: list[str] = []
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        while True:
            row = await self._queries.get_job(self._job_id)
            cfg = json.loads(row.config)
            if row.desired_state == "stopped" and row.status is not JobStatus.stopped:
                if self._converge_stop:
                    # Status only — the real teardown leaves config alone, and
                    # the stale ``db_ready`` it leaves behind is the hazard.
                    await self._queries.update_job_status(self._job_id, JobStatus.stopped)
                    self.events.append("stopped")
            elif row.desired_state == "running" and row.status is JobStatus.stopped:
                cfg.pop("db_ready", None)
                await self._queries.update_job_config(self._job_id, json.dumps(cfg))
                await self._queries.update_job_status(self._job_id, JobStatus.running)
                self._polls_since_relaunch = 0
                self.events.append("relaunched")
            elif row.desired_state == "running" and not cfg.get("db_ready"):
                if self._polls_since_relaunch < self._ready_delay_polls:
                    self._polls_since_relaunch += 1
                else:
                    cfg["db_ready"] = True
                    await self._queries.update_job_config(self._job_id, json.dumps(cfg))
                    self.events.append("ready")
            await asyncio.sleep(0.005)

    async def __aenter__(self) -> _FakeReconciler:
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


@pytest.fixture(autouse=True)
def _fast_quiesce_polls(monkeypatch):
    """The 0.5 s production cadence would make every quiesce test a stopwatch."""
    monkeypatch.setattr("nerdit.core.services._QUIESCE_POLL_INTERVAL_S", 0.005)


async def test_redis_restore_prepares_the_dir_before_quiescing_then_swaps(queries, tmp_path):
    """(D-P37-6) The ORDER is the safety property.

    The replacement ``appendonlydir`` is written COMPLETE, next to the live
    one, while the server is still up; only then is the row quiesced and the
    two swapped by two renames. A bare ``dump.rdb`` in ``/data`` is never
    written — verified live, a server booted ``appendonly yes`` IGNORES it and
    serves an empty dataset.
    """
    runtime = _DumpRuntime(write_output=False)
    controller = _redis_controller(queries, runtime, tmp_path, _redis_secrets(tmp_path))
    job, volume_dir = await _redis_row(queries, tmp_path)
    (volume_dir / "appendonlydir").mkdir()
    (volume_dir / "appendonlydir" / "appendonly.aof.manifest").write_text("old\n")
    staging = _seed_staging(tmp_path, "restccccccc1", "dump.rdb", b"REDIS0011payload")

    async with _FakeReconciler(queries, job.id):
        result = await controller.restore_database(
            job, run_id="restccccccc1", staging=staging, timeout_s=5
        )

    live = volume_dir / "appendonlydir"
    assert (live / "appendonly.aof.1.base.rdb").read_bytes() == b"REDIS0011payload"
    assert (live / "appendonly.aof.manifest").read_text() == (
        "file appendonly.aof.1.base.rdb seq 1 type b\n"
    )
    assert stat.S_IMODE((live / "appendonly.aof.1.base.rdb").stat().st_mode) == 0o600
    # The previous data is renamed aside and KEPT — never deleted.
    kept = [p for p in volume_dir.iterdir() if p.name.startswith("appendonlydir.pre-restore-")]
    assert len(kept) == 1
    assert (kept[0] / "appendonly.aof.manifest").read_text() == "old\n"
    # No prepared dir is left behind, and no container ran at all.
    assert not any(p.name.startswith("appendonlydir.restore-") for p in volume_dir.iterdir())
    assert runtime.run_configs == []
    assert result.exit_code is None
    row = await queries.get_job(job.id)
    assert row.desired_state == "running"
    assert not staging.exists()


async def test_redis_restore_installs_when_there_is_no_live_aof_dir_yet(queries, tmp_path):
    """A database that has never written an AOF dir still restores — the first
    rename is simply skipped, and nothing is invented to rename aside."""
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, volume_dir = await _redis_row(queries, tmp_path)
    staging = _seed_staging(tmp_path, "restddddddd1", "dump.rdb", b"REDIS0011")

    async with _FakeReconciler(queries, job.id):
        await controller.restore_database(job, run_id="restddddddd1", staging=staging, timeout_s=5)

    assert (volume_dir / "appendonlydir" / "appendonly.aof.1.base.rdb").exists()
    assert [p.name for p in volume_dir.iterdir()] == ["appendonlydir"]


async def test_restore_with_slot_held_neither_registers_nor_discards(queries, tmp_path):
    """(D-P37-9) ``slot_held=True`` leaves the caller's reservation alone on
    every path; the default still claims and releases its own."""
    runtime = _DumpRuntime(write_output=False)
    controller = _pg_controller(queries, runtime, tmp_path, _pg_secrets(tmp_path))
    await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)
    job = await queries.get_service_by_name("pg")

    controller.reserve_dump_slot(job.id, "restsloteeee")
    assert controller.has_active_run(job.id) is True
    with pytest.raises(RunPreconditionError):
        controller.reserve_dump_slot(job.id, "restslotffff")
    staging = _seed_staging(tmp_path, "restsloteeee", "dump.pgdump")
    await controller.restore_database(
        job, run_id="restsloteeee", staging=staging, timeout_s=30, slot_held=True
    )
    # Still held: the controller did not discard what it did not register.
    assert controller.has_active_run(job.id) is True
    controller.release_dump_slot(job.id, "restsloteeee")
    assert controller.has_active_run(job.id) is False

    staging = _seed_staging(tmp_path, "restslotgggg", "dump.pgdump")
    await controller.restore_database(job, run_id="restslotgggg", staging=staging, timeout_s=30)
    assert controller.has_active_run(job.id) is False


async def test_a_failed_swap_still_puts_the_row_back_to_running(queries, tmp_path):
    """The row must never be left administratively stopped because a restore
    failed: that is an outage the operator did not ask for and cannot tell
    apart from a crash."""
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, volume_dir = await _redis_row(queries, tmp_path)
    staging = _seed_staging(tmp_path, "resteeeeeee1", "dump.rdb", b"REDIS0011")

    def _boom(*_args, **_kwargs):
        raise OSError("rename failed")

    async with _FakeReconciler(queries, job.id):
        with pytest.raises(DumpError) as excinfo:
            import nerdit.core.services as services_module

            original = services_module._swap_aof_dir
            services_module._swap_aof_dir = _boom
            try:
                await controller.restore_database(
                    job, run_id="resteeeeeee1", staging=staging, timeout_s=5
                )
            finally:
                services_module._swap_aof_dir = original

    assert excinfo.value.reason == "interrupted"
    row = await queries.get_job(job.id)
    assert row.desired_state == "running"
    assert controller.has_active_run(job.id) is False
    assert not staging.exists()


async def test_a_swap_that_fails_after_renaming_aside_rolls_the_live_dir_back(
    queries, tmp_path, monkeypatch
):
    """(D-P37-6) A second rename that fails unwinds the first.

    The quiesce restarts the row unconditionally, and an append-only Redis with
    no live ``appendonlydir`` creates an empty base and serves an EMPTY dataset
    — so the failure has to hand the ORIGINAL dir back before that restart.
    """
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, volume_dir = await _redis_row(queries, tmp_path)
    (volume_dir / "appendonlydir").mkdir()
    (volume_dir / "appendonlydir" / "appendonly.aof.manifest").write_text("old\n")
    staging = _seed_staging(tmp_path, "restggggggg1", "dump.rdb", b"REDIS0011")

    real_rename = os.rename
    calls: list[int] = []

    def _fail_on_the_second_rename(src, dst):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("rename failed")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _fail_on_the_second_rename)

    async with _FakeReconciler(queries, job.id):
        with pytest.raises(DumpError) as excinfo:
            await controller.restore_database(
                job, run_id="restggggggg1", staging=staging, timeout_s=5
            )

    assert excinfo.value.reason == "interrupted"
    # The original data is LIVE again, so the relaunch serves it, not an empty set.
    assert (volume_dir / "appendonlydir" / "appendonly.aof.manifest").read_text() == "old\n"
    assert [p.name for p in volume_dir.iterdir()] == ["appendonlydir"]
    row = await queries.get_job(job.id)
    assert row.desired_state == "running"
    assert controller.has_active_run(job.id) is False
    assert not staging.exists()


async def test_a_swap_whose_rollback_also_fails_leaves_both_copies_on_the_volume(
    queries, tmp_path, monkeypatch
):
    """(D-P37-6) Nothing is ever deleted: when the unwind fails too, the
    previous data and the restore candidate both stay for the operator."""
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, volume_dir = await _redis_row(queries, tmp_path)
    (volume_dir / "appendonlydir").mkdir()
    (volume_dir / "appendonlydir" / "appendonly.aof.manifest").write_text("old\n")
    staging = _seed_staging(tmp_path, "resthhhhhhh1", "dump.rdb", b"REDIS0011")

    real_rename = os.rename
    calls: list[int] = []

    def _fail_on_the_swap_and_the_unwind(src, dst):
        calls.append(1)
        if len(calls) in (2, 3):
            raise OSError("rename failed")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", _fail_on_the_swap_and_the_unwind)

    async with _FakeReconciler(queries, job.id):
        with pytest.raises(DumpError) as excinfo:
            await controller.restore_database(
                job, run_id="resthhhhhhh1", staging=staging, timeout_s=5
            )

    assert excinfo.value.reason == "interrupted"
    assert not (volume_dir / "appendonlydir").exists()
    kept = [p for p in volume_dir.iterdir() if p.name.startswith("appendonlydir.pre-restore-")]
    assert len(kept) == 1
    assert (kept[0] / "appendonly.aof.manifest").read_text() == "old\n"
    prepared = [p for p in volume_dir.iterdir() if p.name.startswith("appendonlydir.restore-")]
    assert len(prepared) == 1
    row = await queries.get_job(job.id)
    assert row.desired_state == "running"


async def test_a_quiesce_that_never_stops_times_out_and_restores_the_desired_state(
    queries, tmp_path
):
    """(D-P37-11) ``quiesce_timeout`` is a real reason, and the row is handed
    back ``running`` even though it never converged."""
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, volume_dir = await _redis_row(queries, tmp_path)
    staging = _seed_staging(tmp_path, "restfffffff1", "dump.rdb", b"REDIS0011")

    async with _FakeReconciler(queries, job.id, converge_stop=False):
        with pytest.raises(DumpError) as excinfo:
            await controller.restore_database(
                job, run_id="restfffffff1", staging=staging, timeout_s=0
            )

    assert excinfo.value.reason == "quiesce_timeout"
    row = await queries.get_job(job.id)
    assert row.desired_state == "running"
    # Nothing was swapped, and the prepared dir did not survive the refusal.
    assert not (volume_dir / "appendonlydir").exists()
    assert list(volume_dir.iterdir()) == []
    assert (await _last_dump(queries, job.id))["reason"] == "quiesce_timeout"


async def test_the_ready_wait_takes_the_flag_set_after_the_relaunch_not_the_stale_one(
    queries, tmp_path
):
    """(D-P37-6) The restart wait is a CONJUNCTION, and each half is load-bearing.

    The sequence the fake reconciler reproduces from production is
    ``stopped`` (config untouched, so ``db_ready`` is still the previous
    container's ``True``) → ``relaunched`` (``on_launched`` clears the flag, then
    the status flips) → ``ready`` (the probe sets it again, some polls later).
    Asserting where the quiesce returned INSIDE that sequence is what pins the
    conjunction: a ``db_ready``-only wait would have returned on the stale flag
    before ``relaunched``, and a ``running``-only wait would have returned
    before ``ready``, while the Redis this restore just fed was still replaying
    the AOF base.
    """
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, _ = await _redis_row(queries, tmp_path)

    async with _FakeReconciler(queries, job.id) as loop:
        async with controller.quiesce_row(job, timeout_s=5):
            row = await queries.get_job(job.id)
            assert row.status is JobStatus.stopped
            # The hazard, made explicit: the stop really does leave the flag
            # behind, so it cannot be the whole readiness signal.
            assert json.loads(row.config)["db_ready"] is True
        loop.events.append("quiesce-returned")

    assert loop.events == ["stopped", "relaunched", "ready", "quiesce-returned"]
    row = await queries.get_job(job.id)
    assert row.status is JobStatus.running
    assert json.loads(row.config)["db_ready"] is True


async def test_the_two_quiesce_phases_share_one_deadline(queries, tmp_path, monkeypatch):
    """(D-P37-6/8) The WHOLE quiesce fits *timeout_s*, not each half of it.

    The MCP tool holds a ~300 s caller-side read budget (the CLI no longer
    bounds the read at all), so a stop wait and a restart wait each allowed a
    full *timeout_s* would let a slow Redis swap run for ``2 × timeout_s``
    server-side and blow through that budget — the daemon still working while the
    caller has already been handed a transport error. Pinned on the arithmetic
    rather than on a stopwatch: both phases receive the SAME absolute deadline,
    and it is one *timeout_s* away from the call.
    """
    seen: list[float] = []
    original_stop = ServiceController._await_row_stopped
    original_ready = ServiceController._await_db_ready

    async def _record_stop(self, job_id: str, deadline: float) -> bool:
        seen.append(deadline)
        return await original_stop(self, job_id, deadline)

    async def _record_ready(self, job_id: str, deadline: float) -> bool:
        seen.append(deadline)
        return await original_ready(self, job_id, deadline)

    monkeypatch.setattr(ServiceController, "_await_row_stopped", _record_stop)
    monkeypatch.setattr(ServiceController, "_await_db_ready", _record_ready)

    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, _ = await _redis_row(queries, tmp_path)

    before = time.monotonic()
    async with _FakeReconciler(queries, job.id), controller.quiesce_row(job, timeout_s=5):
        pass

    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert before + 5 <= seen[0] <= time.monotonic() + 5


async def test_quiesce_row_restores_running_even_when_the_body_raises(queries, tmp_path):
    """The context manager's own contract, tested without a restore around it."""
    controller = _redis_controller(
        queries,
        _DumpRuntime(write_output=False),
        tmp_path,
        _redis_secrets(tmp_path),
    )
    job, _ = await _redis_row(queries, tmp_path)

    async with _FakeReconciler(queries, job.id):
        with pytest.raises(RuntimeError):
            async with controller.quiesce_row(job, timeout_s=5):
                assert (await queries.get_job(job.id)).status is JobStatus.stopped
                raise RuntimeError("body failed")

    assert (await queries.get_job(job.id)).desired_state == "running"


# --- the P20 surface is untouched ---------------------------------------------


async def test_run_once_still_claims_a_plain_run_slot(queries, tmp_path):
    """The widening must not have changed what a one-off run registers."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _vol_controller(queries, runtime, tmp_path=tmp_path)
    job = _deploy_svc("app")
    await queries.create_job(job)

    async def _peek() -> RunMode:
        await asyncio.sleep(0)
        return next(iter(controller._active_runs[job.id].values())).mode

    task = asyncio.create_task(controller.run_once(job, command=["true"], timeout_s=5))
    mode = await _peek()
    await task
    assert mode is RunMode.run
    await controller.shutdown()
