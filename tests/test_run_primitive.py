"""Test one-off runs, secret scrubbing and service-equivalent hardening.

Runs receive the full resolved environment: scrub runtime-bounded output before
clamping lines. Pin this order directly and through container execution. Run
containers have no ports or GPUs, reuse the selected image, and match service
limits and hardening fields.

Exercise run_once with real controllers/database and FakeRuntime to pin environment
precedence, preconditions, the active-run registry and config.last_run. Controller
helpers own the byte clamp and composed scrub/truncate operation.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nerdit.config.settings import ContainerSettings, RetentionSettings, ServicesSettings
from nerdit.core.app_build import _sensitive_env_values
from nerdit.core.data.backend import PostgresBackend, RedisBackend
from nerdit.core.launch import (
    _RUN_LINE_MAX_BYTES,
    _RUN_TAIL_MAX_BYTES,
    _SCRUB_MIN_VALUE_LEN,
    AppContainerSpec,
    apply_platform_overlay,
    build_app_container_config,
    build_run_container_config,
    finalize_container_config,
    scrub_secret_values,
)
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.core.secrets import SecretManager
from nerdit.core.services import (
    LaunchEnvNotReady,
    ResolvedLaunchEnv,
    RunInterruptedError,
    RunPreconditionError,
    ServiceController,
    _scrub_and_truncate,
    _truncate_line,
)
from nerdit.core.volumes import VolumeSpecError
from nerdit.db.models import ContainerConfig, GpuVendor, Job, JobKind, JobStatus
from tests.test_services_reconcile import (
    FakeRuntime,
    FakeSecrets,
    _controller,
    _deploy_svc,
    _run_config,
    _svc,
)

# --- scrub_secret_values ------------------------------------------------------


def test_scrub_replaces_every_occurrence_on_every_line():
    lines = ["key=hunter2000", "again hunter2000 and hunter2000", "clean"]
    out = scrub_secret_values(lines, {"hunter2000"})
    assert out == ["key=***", "again *** and ***", "clean"]


def test_scrub_returns_a_new_list_and_does_not_mutate_the_input():
    lines = ["key=hunter2000"]
    out = scrub_secret_values(lines, {"hunter2000"})
    assert lines == ["key=hunter2000"]
    assert out is not lines


def test_scrub_masks_every_length_because_the_floor_lives_at_the_builder():
    """(PR #96 review F1) The consumer masks whatever it is handed, at any
    length. By this tier the two provenances are one flat set of strings, so a
    length test here cannot tell a DECLARED secret from a name-heuristic GUESS
    — it just re-erases the distinction ``_sensitive_env_values`` drew one hop
    earlier, and publishes every declared secret below the floor."""
    short = "a" * (_SCRUB_MIN_VALUE_LEN - 1)
    assert scrub_secret_values([f"pw={short} data={short}b"], {short}) == ["pw=*** data=***b"]
    long = "a" * _SCRUB_MIN_VALUE_LEN
    assert scrub_secret_values([f"pw={long}"], {long}) == ["pw=***"]


def test_the_length_floor_applies_to_guesses_and_never_to_declared_secrets():
    """(PR #96 review F1) The builder is the only tier that still knows which
    is which, so it is the only tier that may apply the floor.

    A declared secret is in the set at ANY length — ``POST /secrets/{service}``
    enforces no minimum, and a 4-char production key is still a production key.
    A value caught only by the spelling of its key stays floored, so a
    ``DB_PASSWORD=1`` cannot turn every ``1`` in a migration log into a mask.
    """
    short = "ab12"
    assert len(short) < _SCRUB_MIN_VALUE_LEN
    resolved = ResolvedLaunchEnv(
        env={"DB_PASSWORD": "1", "STRIPE_API_KEY": "sk_live_long_enough"},
        secret_env={"API_KEY": short},
    )

    values = _sensitive_env_values(resolved)

    assert short in values, "a declared secret must not be floored"
    assert "1" not in values, "a short heuristic guess must stay floored"
    assert "sk_live_long_enough" in values


def test_an_empty_declared_secret_is_dropped_by_the_builder():
    """``"".replace()`` splices ``***`` between every character, so an empty
    value has to be dropped before it reaches the scrub — and now that the
    consumer no longer floors, the builder is where that happens."""
    resolved = ResolvedLaunchEnv(env={}, secret_env={"UNSET": ""})
    assert _sensitive_env_values(resolved) == set()


def test_scrub_ignores_empty_values():
    """An empty secret value would otherwise turn every character boundary into
    ``***``. Callers do not have to pre-filter."""
    assert scrub_secret_values(["nothing to hide"], {""}) == ["nothing to hide"]


def test_scrub_masks_the_widest_value_first_when_values_nest():
    """A password nested inside the DSN built from it: masking the short one
    first would blank out the middle of the long one and leave its surrounding
    structure unmatched and visible."""
    password = "s3cr3t-pass"
    dsn = f"postgres://u:{password}@db:5432/app"
    out = scrub_secret_values([f"DATABASE_URL={dsn}"], {password, dsn})
    assert out == ["DATABASE_URL=***"]
    assert password not in out[0]


def test_scrub_result_is_independent_of_set_iteration_order():
    """Ties break on the value itself, so two equal-length secrets can never
    produce two different tails across runs."""
    a, b = "aaaaaaaa", "bbbbbbbb"
    line = f"{a} then {b}"
    assert scrub_secret_values([line], {a, b}) == scrub_secret_values([line], {b, a})


def test_scrub_leaves_non_secret_env_values_alone():
    """The scrub set is secret VALUES, not env keys: an injected
    ``OPENAI_BASE_URL`` is platform-computed, not sensitive, and must survive
    verbatim so a failing run stays diagnosable."""
    lines = ["OPENAI_BASE_URL=http://172.17.0.1:11434/v1", "OPENAI_API_KEY=sk-live-abcdef"]
    out = scrub_secret_values(lines, {"sk-live-abcdef"})
    assert out[0] == "OPENAI_BASE_URL=http://172.17.0.1:11434/v1"
    assert out[1] == "OPENAI_API_KEY=***"


def test_scrub_with_no_targets_is_a_copy():
    lines = ["a", "b"]
    out = scrub_secret_values(lines, [])
    assert out == lines
    assert out is not lines


# --- _truncate_line -----------------------------------------------------------


def test_truncate_line_leaves_short_lines_untouched():
    assert _truncate_line("short") == "short"
    exact = "x" * _RUN_LINE_MAX_BYTES
    assert _truncate_line(exact) == exact


def test_truncate_line_clamps_an_oversized_line_in_bytes():
    line = "x" * (_RUN_LINE_MAX_BYTES + 10)
    out = _truncate_line(line)
    assert out.endswith("…[truncated]")
    assert out[: -len("…[truncated]")] == "x" * _RUN_LINE_MAX_BYTES


def test_truncate_line_measures_bytes_not_characters():
    """2 KiB is a BYTE budget: a line of three-byte characters is clamped well
    before it reaches 2048 *characters*, and a cut landing inside a multi-byte
    sequence drops that character rather than producing a mojibake row.

    ``_RUN_LINE_MAX_BYTES`` is not a multiple of 3, so the cut here really does
    fall mid-sequence.
    """
    assert _RUN_LINE_MAX_BYTES % 3 != 0  # the mid-sequence cut is genuine
    line = "日" * _RUN_LINE_MAX_BYTES  # 3 bytes each ⇒ 3× the budget
    out = _truncate_line(line)

    assert out.endswith("…[truncated]")
    body = out[: -len("…[truncated]")]
    assert set(body) == {"日"}  # no replacement character, no partial byte
    assert len(body) == _RUN_LINE_MAX_BYTES // 3
    assert len(body.encode()) <= _RUN_LINE_MAX_BYTES


def test_run_byte_caps_are_the_pinned_values():
    """These bound the response body, ``config['last_run']`` and a release's
    ``job_logs``; they live in ``core`` because the enforcement site is the
    controller tier and ``core`` must never import ``daemon``."""
    assert _RUN_TAIL_MAX_BYTES == 16384
    assert _RUN_LINE_MAX_BYTES == 2048


# --- the redaction pipeline ORDER (PR #96 review F2) --------------------------
#
# Clamp-then-scrub cut a secret longer than `_RUN_LINE_MAX_BYTES` mid-value, so
# the literal match in `scrub_secret_values` missed and the clamped 2 KiB PREFIX
# of the secret survived into the run response and — for a `[deploy].release` —
# into `job_logs`, which ANY authenticated principal (including `readonly`) can
# read. A 100-char prefix is the assertion: it is far past the point where the
# leak stops being deniable, and it holds whatever the clamp does at the margin.

_LONG_SECRET = "s" * 3000  # > _RUN_LINE_MAX_BYTES: the clamp lands INSIDE it


def test_scrub_and_truncate_scrubs_first_then_clamps():
    """The composition point itself: scrub the full line, THEN clamp it."""
    line = f"DB_PASSWORD={_LONG_SECRET} " + "tail" * 250  # ~4 KiB, secret inside

    out = _scrub_and_truncate([line], {_LONG_SECRET})

    assert "***" in out[0]
    assert _LONG_SECRET[:100] not in out[0]
    # Scrubbing first also SHRINKS the line below the cap — nothing to clamp.
    assert len(out[0].encode()) <= _RUN_LINE_MAX_BYTES + len("…[truncated]".encode())
    # ...and the clamp is still there for a line that is genuinely oversized
    # (the reorder must not quietly drop truncation).
    oversized = "x" * (_RUN_LINE_MAX_BYTES + 500)
    assert _scrub_and_truncate([oversized], {_LONG_SECRET})[0].endswith("…[truncated]")


async def test_execute_container_once_scrubs_a_secret_longer_than_the_line_cap(queries):
    """End-to-end through the run path: a 3000-char secret is masked whole."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = [f"connecting with DB_PASSWORD={_LONG_SECRET} ok"]
    controller = _controller(queries, runtime)
    job = _svc("svc")

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        result = await controller._execute_container_once(
            job,
            run_id="run-1",
            config=_run_config(),
            timeout_s=5,
            log_tail=10,
            scrub_values=[_LONG_SECRET],
        )
    finally:
        controller._discard_run(job.id, "run-1")

    assert "***" in "\n".join(result.log_tail)
    assert _LONG_SECRET[:100] not in "\n".join(result.log_tail)


async def test_run_interrupted_tail_is_scrubbed_before_truncation(queries):
    """The lost-container branch carries a tail too, through the same pipeline —
    a runtime hiccup must not be the one path that leaks the prefix."""
    runtime = FakeRuntime()
    runtime.wait_error = ContainerRuntimeError("docker went away")
    runtime.log_lines = [f"connecting with DB_PASSWORD={_LONG_SECRET} ok"]
    controller = _controller(queries, runtime)
    job = _svc("svc")

    controller._register_run(job.id, "run-1", is_release=False)
    try:
        with pytest.raises(RunInterruptedError) as exc:
            await controller._execute_container_once(
                job,
                run_id="run-1",
                config=_run_config(),
                timeout_s=5,
                log_tail=10,
                scrub_values=[_LONG_SECRET],
            )
    finally:
        controller._discard_run(job.id, "run-1")

    assert "***" in "\n".join(exc.value.log_tail)
    assert _LONG_SECRET[:100] not in "\n".join(exc.value.log_tail)


# --- build_run_container_config ----------------------------------------------


def _cfg() -> dict:
    return {"image": "nerdit-app/demo:3", "port": 8000, "memory_limit": "512m", "cpu_limit": 1.5}


def test_run_container_is_portless_and_gpuless():
    """A run must never contend for (or transiently steal) the live service's
    stable host port, and v1 runs are CPU-only (D-C)."""
    config = build_run_container_config(
        _cfg(),
        ContainerSettings(),
        image="nerdit-app/demo:3",
        command=["python", "migrate.py"],
        env={"A": "1"},
        workdir=None,
    )
    assert config.ports is None
    assert config.gpu_ids == []
    assert config.volumes is None
    assert config.command == ["python", "migrate.py"]
    assert config.env == {"A": "1"}


def test_run_container_image_is_the_caller_s_never_the_default():
    """``run_once``'s image gate resolves and existence-checks the row's
    deployed image first; a run must never silently fall back to
    ``nerdit-runtime``."""
    cs = ContainerSettings(default_image="nerdit-runtime:0.1")
    config = build_run_container_config(
        {},  # no image in the row config at all
        cs,
        image="nerdit-app/demo:7",
        command=["sh", "-c", "true"],
        env=None,
        workdir=None,
    )
    assert config.image == "nerdit-app/demo:7"


def test_run_container_forces_bridge_networking():
    """A run inherits the SERVICE's network posture (so ``[models].bridge_host``
    stays reachable), not an agent token's default."""
    config = build_run_container_config(
        _cfg(),
        ContainerSettings(),
        image="nerdit-app/demo:3",
        command=["true"],
        env=None,
        workdir=None,
    )
    assert config.network_mode == "bridge"


# --- security S10: structural drift pin --------------------------------------

_SHARED_FIELDS = (
    "cap_drop",
    "no_new_privileges",
    "read_only",
    "network_mode",
    "memory_limit",
    "cpu_limit",
    "pids_limit",
    "log_config",
)


@pytest.mark.parametrize(
    "cs",
    [
        ContainerSettings(),
        ContainerSettings(drop_all_caps=False, no_new_privileges=False),
        ContainerSettings(read_only_rootfs=True),
        ContainerSettings(default_memory_limit="256m", default_cpu_limit=0.5),
        ContainerSettings(pids_limit=None),
    ],
    ids=["defaults", "caps-off", "read-only-rootfs", "daemon-defaults", "pids-unlimited"],
)
@pytest.mark.parametrize("cfg", [{}, {"memory_limit": "1g", "cpu_limit": 2.0}], ids=["bare", "row"])
def test_run_container_hardening_is_field_equal_to_the_service_container(cs, cfg):
    """Security S10: a one-off run gets EXACTLY the service's limits + sandbox.

    Compared after ``finalize_container_config`` because that is where the P14b
    ``log_config`` caps are set (both are ``None`` before it). If a new limit or
    hardening field is added to the service branch alone, this fails.
    """
    retention = RetentionSettings(container_log_max_size="10m", container_log_max_file=3)

    app_config = build_app_container_config(
        cfg,
        cs,
        AppContainerSpec(
            command=["npm", "start"],
            volumes=None,
            env={"PORT": "8000"},
            workdir=None,
            gpu_ids=[],
            vendor=GpuVendor.nvidia,
            container_port=8000,
            host_port=9400,
        ),
    )
    run_config = build_run_container_config(
        cfg,
        cs,
        image="nerdit-app/demo:3",
        command=["python", "migrate.py"],
        env={"PORT": "8000"},
        workdir=None,
    )
    finalize_container_config(app_config, {}, retention)
    finalize_container_config(run_config, {}, retention)

    for field in _SHARED_FIELDS:
        assert getattr(run_config, field) == getattr(app_config, field), field

    assert run_config.pids_limit == cs.pids_limit

    # ...and the two differ ONLY where they are meant to.
    assert app_config.ports is not None
    assert run_config.ports is None


def test_pids_limit_defaults_to_4096_and_reaches_the_overlay():
    """Security S3: every launched container gets the fork-bomb cap by default."""
    assert ContainerSettings().pids_limit == 4096
    config = ContainerConfig(image="postgres:16", gpu_ids=[])
    apply_platform_overlay(config, {}, ContainerSettings(pids_limit=128), 5432, 9500)
    assert config.pids_limit == 128


# --- where the bounds constants live -----------------------------------------


def test_route_facing_tail_cap_lives_in_daemon_limits():
    """The response-shaping cap belongs on the route side, beside its
    ``/diagnose`` twin; only the capture-time BYTE budgets live in ``core``."""
    from nerdit.daemon.limits import _MAX_DIAGNOSE_TAIL, _MAX_RUN_LOG_TAIL

    assert _MAX_RUN_LOG_TAIL == 200
    assert _MAX_RUN_LOG_TAIL == _MAX_DIAGNOSE_TAIL


def test_core_never_imports_daemon():
    """The import-direction invariant that decided where the byte caps are homed.

    ``_RUN_TAIL_MAX_BYTES``/``_RUN_LINE_MAX_BYTES`` are enforced at the
    controller tier, so putting them in ``daemon/limits.py`` (the plan's first
    guess) would have made ``core`` import ``daemon``. Scanned across the whole
    package so a later move is caught rather than silently accepted.
    """
    import re
    from pathlib import Path

    import nerdit.core

    pattern = re.compile(r"^\s*(?:from|import)\s+nerdit\.daemon\b", re.M)
    offenders = [
        str(path.relative_to(Path(nerdit.core.__file__).parent))
        for path in Path(nerdit.core.__file__).parent.rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


# =============================================================================
# WP3 — run_once
# =============================================================================
#
# The D-P14-5 contract-freeze block. `run_once` is the controller half of
# `POST /services/{ident}/run` (the route is WP4), so everything here is driven
# directly against the method: real in-memory `queries`, the shared
# `FakeRuntime`, a real `ServiceController`. The four env-precedence tests
# (a)-(d) + (c-bis) are the ones that must never be relaxed — they encode
# "a caller override beats config env and secrets, and LOSES to anything the
# platform computed".


def _vol_controller(
    queries, runtime, *, tmp_path=None, secrets=None, data_controller=None, **kw
) -> ServiceController:
    """A controller with the container/data_dir seams ``run_once`` needs.

    The ``test_services_reconcile._controller`` helper wires neither
    ``container_settings`` nor ``data_dir``, so a volume assertion built on it
    would silently exercise the *no-volume* branch and pass for the wrong
    reason (``run_once`` guards on ``self._data_dir is not None``). Same shape
    as ``test_release_gate._controller``.
    """
    kw.setdefault("service_port_range", "9400-9499")
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(**kw),
        container_settings=ContainerSettings(),
        secrets=secrets,
        data_controller=data_controller,
        data_dir=str(tmp_path) if tmp_path is not None else None,
    )


class _GatedRuntime(FakeRuntime):
    """A runtime whose run container blocks inside ``wait`` until ``gate`` is set."""

    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__()
        self.gate = gate
        self.started = asyncio.Event()

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        self.started.set()
        await self.gate.wait()
        return await super().wait(container_id, timeout_s=timeout_s)


_PW = "deadbeef" * 8  # 64-hex: the managed-credential mint shape

_API_SPEC = {
    "provider": "api",
    "model": "gpt-4o-mini",
    "base_url": "https://api.example.com/v1",
    "api_key": "${secrets.OPENAI_KEY}",
}


async def _run(controller, job, **kw):
    """``run_once`` with the boring arguments filled in."""
    kw.setdefault("command", ["python", "migrate.py"])
    kw.setdefault("timeout_s", 30)
    return await controller.run_once(job, **kw)


# --- §1.1 (a)-(d): the D-P14-5 protected-key overlay --------------------------


async def test_run_once_override_of_a_secret_provided_key_wins(queries):
    """(a) A caller override beats config env AND a per-service secret.

    Neither is platform-computed: they are the app owner's own values, and the
    owner is exactly who is allowed to say "run this migration with a different
    APP_TOKEN". Only the binding resolvers' output is untouchable.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime, secrets=FakeSecrets({"APP_TOKEN": "from-secrets"}))
    job = _deploy_svc("app", env={"APP_TOKEN": "from-config", "OTHER": "keep"})
    await queries.create_job(job)

    await _run(controller, job, env_overrides={"APP_TOKEN": "from-caller"})

    env = runtime.run_configs[-1].env
    assert env["APP_TOKEN"] == "from-caller"
    assert env["OTHER"] == "keep"  # untouched keys survive the overlay
    await controller.shutdown()


async def test_run_once_drops_an_openai_base_url_override_when_a_binding_resolves(queries):
    """(b) Invariant #1: an ``[ai.*]`` binding's injected env is platform-computed,
    so a caller ``--env OPENAI_BASE_URL=evil`` is dropped, not honoured."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime, secrets=FakeSecrets({"OPENAI_KEY": "sk-live-xyz"}))
    job = _deploy_svc("app", ai={"default": _API_SPEC})
    await queries.create_job(job)

    await _run(
        controller,
        job,
        env_overrides={"OPENAI_BASE_URL": "https://evil.example/v1", "OPENAI_API_KEY": "sk-evil"},
    )

    env = runtime.run_configs[-1].env
    assert env["OPENAI_BASE_URL"] == "https://api.example.com/v1"
    assert env["OPENAI_API_KEY"] == "sk-live-xyz"
    await controller.shutdown()


class _FakeData:
    """Minimal ``DataController`` stand-in: one fixed backend for every row."""

    def __init__(self, backend, bridge_host: str) -> None:
        self._backend = backend
        self.bridge_host = bridge_host

    def backend_for(self, _cfg: dict) -> object:
        return self._backend


async def _serve_managed_db(queries, *, name: str, backend_key: str, port: int):
    """A ready ``kind=database`` row with a stable endpoint (P15 recipe)."""
    job = Job(
        name=name,
        kind=JobKind.database,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(
            {
                "backend": backend_key,
                "image": f"{backend_key}:latest",
                "port": port,
                "db_ready": True,
            }
        ),
    )
    await queries.create_job(job)
    await queries.acquire_service_port(name, job.id, port, (9400, 9499))
    return await queries.get_service_endpoint(name)


async def test_run_once_drops_database_url_overrides_for_a_managed_postgres_binding(
    queries, tmp_path
):
    """(c) The managed-DB half of (b): both the alias and the per-binding var."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("pg", {"POSTGRES_PASSWORD": _PW})
    endpoint = await _serve_managed_db(queries, name="pg", backend_key="postgres", port=5432)

    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _vol_controller(
        queries,
        runtime,
        secrets=mgr,
        data_controller=_FakeData(PostgresBackend(), "172.17.0.1"),
        service_port_range="9500-9599",
    )
    job = _deploy_svc("app", db={"default": {"provider": "managed", "database": "pg"}})
    await queries.create_job(job)

    await _run(
        controller,
        job,
        env_overrides={
            "DATABASE_URL": "postgresql://evil@nowhere/x",
            "NERDIT_DB_DEFAULT_URL": "postgresql://evil@nowhere/x",
        },
    )

    env = runtime.run_configs[-1].env
    expected = f"postgresql://nerdit:{_PW}@{controller._bridge_host}:{endpoint.host_port}/nerdit"
    assert env["DATABASE_URL"] == expected
    assert env["NERDIT_DB_DEFAULT_URL"] == expected
    await controller.shutdown()


async def test_run_once_drops_a_redis_url_override_for_a_managed_redis_binding(queries, tmp_path):
    """(c-bis) Invariants I1: the protected set is what was ACTUALLY injected.

    ``inject_db_env_key_names`` reports ``DATABASE_URL`` for a managed
    ``default`` spec whatever the engine, while a Redis backend really injects
    ``REDIS_URL``. A names-only protected set would therefore let
    ``--env REDIS_URL=…`` win — which is why ``run_once`` derives ``protected``
    from ``resolved.injected_keys``.
    """
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("cache", {"REDIS_PASSWORD": _PW})
    endpoint = await _serve_managed_db(queries, name="cache", backend_key="redis", port=6379)

    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _vol_controller(
        queries,
        runtime,
        secrets=mgr,
        data_controller=_FakeData(RedisBackend(), "172.17.0.1"),
        service_port_range="9500-9599",
    )
    job = _deploy_svc("app", db={"default": {"provider": "managed", "database": "cache"}})
    await queries.create_job(job)

    await _run(controller, job, env_overrides={"REDIS_URL": "redis://evil.example:6379/0"})

    env = runtime.run_configs[-1].env
    assert env["REDIS_URL"] == f"redis://:{_PW}@{controller._bridge_host}:{endpoint.host_port}/0"
    assert "DATABASE_URL" not in env  # the alias really is engine-derived

    # And the RECORD must not claim the drop never happened. Reporting the
    # submitted set here would tell an operator (and a forensic reader) that
    # REDIS_URL was honoured, when the command in fact talked to the managed
    # instance — the silent drop is locked (D-P14-5), confirming it is not.
    cfg = json.loads((await queries.get_job(job.id)).config)
    last_run = cfg["last_run"]
    assert last_run["env_override_keys"] == []
    assert last_run["env_override_keys_dropped"] == ["REDIS_URL"]
    await controller.shutdown()


async def test_run_once_port_and_nerdit_run_id_are_platform_values_always(queries):
    """(d) ``PORT``/``NERDIT_RUN_ID`` are plain assignments AFTER the overlay.

    Stricter than ``_launch``, deliberately — and this is the ONE place a run's
    env diverges from a launch's rather than merely being narrower. The resolver
    *setdefaults* ``PORT`` (``services.py`` ``_resolve_launch_env``), so at
    launch the config ``PORT`` below (4444) would win; ``run_once`` overwrites it
    with the row's ``[deploy].port``. Per the locked §1.1 contract. A
    ``setdefault`` here would be a silent contract break — this test is what
    catches it.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(
        queries,
        runtime,
        secrets=FakeSecrets({"PORT": "5555", "NERDIT_RUN_ID": "from-secrets"}),
    )
    job = _deploy_svc("app", env={"PORT": "4444", "NERDIT_RUN_ID": "from-config"})
    await queries.create_job(job)

    result = await _run(
        controller, job, env_overrides={"PORT": "3333", "NERDIT_RUN_ID": "from-caller"}
    )

    env = runtime.run_configs[-1].env
    assert env["PORT"] == "8000"  # the row's container port, nothing else
    assert env["NERDIT_RUN_ID"] == result.run_id
    await controller.shutdown()


# --- preconditions ------------------------------------------------------------


async def test_run_once_service_gone_when_the_row_vanished(queries):
    """The fresh re-read is the point: the route's row is stale by the time the
    slot is held, so a deleted service must not start a container."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")  # never inserted

    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)

    assert exc.value.reason == "service_gone"
    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    await controller.shutdown()


@pytest.mark.parametrize("desired", ["completed", "cancelled", "stopped", "failed"])
async def test_run_once_service_gone_on_a_terminal_desired_state(queries, desired):
    """A row the user has terminally stopped is refused even though it still
    exists — the same four-value settled set ``get_reconcilable_services`` uses."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    job.desired_state = desired
    await queries.create_job(job)

    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)

    assert exc.value.reason == "service_gone"
    assert controller.has_active_run(job.id) is False
    await controller.shutdown()


@pytest.mark.parametrize("mode", ["no-key", "absent"], ids=["config-lacks-image", "image-absent"])
async def test_run_once_no_image(queries, mode):
    """A run must never fall back to ``[containers].default_image``: it runs the
    app's own code or it does not run."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    if mode == "no-key":
        cfg = json.loads(job.config)
        cfg.pop("image")
        job.config = json.dumps(cfg)
    else:
        runtime.missing_images.add("nerdit-app/app:1")
    await queries.create_job(job)

    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)

    assert exc.value.reason == "no_image"
    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    await controller.shutdown()


async def test_run_once_refuses_a_non_service_row(queries):
    """The kind guard is a real refusal, not an ``assert``.

    The route's 422 ``run.not_supported`` is WP4, so until it ships this is the
    only gate a model or database row meets — and an ``assert`` is the one form
    of defence in depth that ``python -O`` deletes. Nothing may launch a model's
    or a database's image with that row's env through the run primitive.
    """
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    job.kind = JobKind.model
    await queries.create_job(job)

    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)

    assert exc.value.reason == "service_gone"
    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    await controller.shutdown()


# --- D-P20-2 single-flight + the S1 registry ledger ---------------------------


async def test_run_once_registers_the_slot_before_its_first_await(queries, monkeypatch):
    """Security S1 ordering pin: the slot exists BEFORE the first suspension.

    ``_register_run`` must run synchronously, before ``run_once`` yields to the
    loop at all — otherwise a DELETE landing inside that window sees
    ``has_active_run() is False``, proceeds, and a run container is then started
    against a row the delete path has already begun tearing down. Moving the
    registration one line down (below the fresh re-read) is invisible to every
    other test in this block, which is exactly why this one exists: it spies on
    ``get_job``, the first await in the method, and asserts the slot is already
    held when it is entered.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    held: list[bool] = []
    original = queries.get_job

    async def spy(job_id: str):
        held.append(controller.has_active_run(job_id))
        return await original(job_id)

    monkeypatch.setattr(queries, "get_job", spy)

    await _run(controller, job)

    assert held, "run_once never re-read the row — the ordering pin is vacuous"
    assert held[0] is True
    await controller.shutdown()


async def test_run_once_is_single_flight_and_registry_empties_after(queries):
    """One run per service, and ``has_active_run`` is true for the WHOLE run —
    that is what makes the DELETE route's 409 meaningful."""
    gate = asyncio.Event()
    runtime = _GatedRuntime(gate)
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    task = asyncio.create_task(_run(controller, job))
    await asyncio.wait_for(runtime.started.wait(), timeout=2)

    assert controller.has_active_run(job.id) is True
    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)
    assert exc.value.reason == "run_in_progress"

    gate.set()
    await task
    assert controller.has_active_run(job.id) is False
    assert len(runtime.run_configs) == 1  # the refused run started nothing
    await controller.shutdown()


async def test_run_once_respects_the_daemon_wide_cap(queries):
    """``[services].max_concurrent_runs`` is checked before any awaited work."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime, max_concurrent_runs=1)
    controller._register_run("someone-else", "r1", is_release=False)
    job = _deploy_svc("app")
    await queries.create_job(job)

    with pytest.raises(RunPreconditionError) as exc:
        await _run(controller, job)

    assert exc.value.reason == "too_many_runs"
    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    await controller.shutdown()


async def test_run_once_launch_env_not_ready_propagates_and_frees_the_slot(queries):
    """Unlike ``_launch`` (retry next tick) and the release (settle the
    generation), a run is caller-facing: the exception travels out so the route
    can map ``.kind``. No wait-line, no dedupe-map write, no swallow."""
    runtime = FakeRuntime()
    controller = _controller(queries, runtime)  # no SecretManager => ref unresolvable
    job = _deploy_svc("app", ai={"default": _API_SPEC})
    await queries.create_job(job)

    with pytest.raises(LaunchEnvNotReady) as exc:
        await _run(controller, job)

    assert exc.value.kind == "binding"
    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    assert await queries.get_logs(job.id) == []
    await controller.shutdown()


async def test_run_once_volume_spec_error_propagates_before_any_start(queries, tmp_path):
    """A bad ``[deploy].volumes`` blob is the caller's 422, never a settle: the
    row is not ``run_once``'s to mark failed."""
    runtime = FakeRuntime()
    controller = _vol_controller(queries, runtime, tmp_path=tmp_path)
    job = _deploy_svc("app", volumes=["bad name:/x"])
    await queries.create_job(job)

    with pytest.raises(VolumeSpecError):
        await _run(controller, job)

    assert controller.has_active_run(job.id) is False
    assert runtime.run_configs == []
    await controller.shutdown()


async def test_run_once_runtime_start_failure_propagates_and_frees_the_slot(queries):
    """Security S1: a raise out of ``runtime.run()`` must still empty the
    registry, or DELETE is wedged for this service until the daemon restarts."""
    runtime = FakeRuntime()
    runtime.run_error = ContainerRuntimeError("no such image")
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    with pytest.raises(ContainerRuntimeError):
        await _run(controller, job)

    assert controller.has_active_run(job.id) is False
    assert "remove" not in runtime.calls  # nothing started, nothing to remove
    await controller.shutdown()


# --- container shape ----------------------------------------------------------


async def test_run_once_container_is_portless_gpuless_bridge_and_labelled(queries):
    """D-C, plus the WP3 attribution labels that let a crash-orphaned container
    be reaped by ``nerdit-job`` instead of waiting out the zombie sweep."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    result = await _run(controller, job, command=["python", "-m", "alembic", "upgrade", "head"])

    config = runtime.run_configs[-1]
    assert config.ports is None
    assert config.gpu_ids == []
    assert config.network_mode == "bridge"
    assert config.command == ["python", "-m", "alembic", "upgrade", "head"]
    assert config.extra_labels == {"nerdit-run": result.run_id, "nerdit-job": job.id}
    await controller.shutdown()


async def test_run_once_mounts_named_volumes_and_no_tier_b_mounts(queries, tmp_path):
    """A migration-style run sees the LIVE data (same daemon-computed host dirs);
    Tier-B sandbox script mounts are deliberately not inherited."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _vol_controller(queries, runtime, tmp_path=tmp_path)
    job = _deploy_svc("app", volumes=["data:/data"])
    await queries.create_job(job)

    await _run(controller, job)

    host = (tmp_path / "services" / "app" / "data").resolve()
    assert runtime.run_configs[-1].volumes == {str(host): "/data"}
    await controller.shutdown()


async def test_run_once_with_no_data_dir_does_no_volume_work(queries):
    """R10: a bare-constructed controller has no ``data_dir``; the guard comes
    FIRST (``_ensure_volume_dirs`` asserts on it) and the run still succeeds."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime)  # data_dir is None
    job = _deploy_svc("app", volumes=["data:/data"])
    await queries.create_job(job)

    result = await _run(controller, job)

    assert result.exit_code == 0
    assert runtime.run_configs[-1].volumes is None
    await controller.shutdown()


# --- persistence: job_logs never, config['last_run'] only ---------------------


async def test_run_once_writes_no_job_logs(queries):
    """D-P14-6. ``GET /services/{ident}/logs`` is served to EVERY authenticated
    principal, readonly included — a run's stdout is owner-only, so it must not
    reach that stream at all. The release is the deliberate exception."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = [f"line {i}" for i in range(50)]
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    await _run(controller, job)

    assert await queries.get_logs(job.id) == []
    await controller.shutdown()


async def test_run_once_stamps_last_run_with_a_scrubbed_tail(queries):
    """The one thing a run persists — and the tail it mirrors is the ALREADY
    scrubbed-and-clamped list ``_execute_container_once`` produced (one choke
    point, never a second scrub with a different set)."""
    runtime = FakeRuntime()
    runtime.exit_code = 3
    runtime.log_lines = ["using APP_TOKEN=hunter2000-long-value now"]
    controller = _controller(
        queries, runtime, secrets=FakeSecrets({"APP_TOKEN": "hunter2000-long-value"})
    )
    job = _deploy_svc("app")
    await queries.create_job(job)

    result = await _run(
        controller, job, command=["sh", "-c", "true"], env_overrides={"B": "2", "A": "1"}
    )

    cfg = json.loads((await queries.get_job(job.id)).config)
    last_run = cfg["last_run"]
    assert set(last_run) == {
        "run_id",
        "command",
        "exit_code",
        "timed_out",
        "oom_killed",
        "started_at",
        "finished_at",
        "duration_s",
        "env_override_keys",
        "env_override_keys_dropped",
        "log_tail",
    }
    assert last_run["run_id"] == result.run_id
    assert last_run["command"] == ["sh", "-c", "true"]  # verbatim (D-P20-5)
    assert last_run["exit_code"] == 3  # a non-zero exit is NOT an error
    assert last_run["timed_out"] is False
    assert last_run["env_override_keys"] == ["A", "B"]  # names only, sorted
    assert last_run["env_override_keys_dropped"] == []
    assert last_run["log_tail"] == result.log_tail
    assert "***" in last_run["log_tail"][0]
    assert "hunter2000-long-value" not in json.dumps(cfg)
    await controller.shutdown()


async def test_run_once_scrub_set_is_sensitive_env_values(queries):
    """The scrub set is ``app_build._sensitive_env_values(resolved)`` verbatim —
    never re-derived, never ``secret_env``-only. Two provenances prove it: a
    declared secret (in unconditionally) and a credential-NAMED plain config
    env value (in via the name heuristic + length floor)."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["declared=declared-secret-value plain=sk_live_abcdefgh"]
    controller = _controller(
        queries, runtime, secrets=FakeSecrets({"PLAIN": "declared-secret-value"})
    )
    job = _deploy_svc("app", env={"STRIPE_API_KEY": "sk_live_abcdefgh"})
    await queries.create_job(job)

    result = await _run(controller, job)

    tail = "\n".join(result.log_tail)
    assert "declared-secret-value" not in tail
    assert "sk_live_abcdefgh" not in tail
    assert tail.count("***") == 2
    await controller.shutdown()


# --- bounds, timeout, mid-run deletion, shared-secret audit -------------------


async def test_run_once_timeout_reports_and_frees_the_slot(queries):
    """The server cap fires inside the runtime's bounded wait; the container is
    killed, the outcome is a normal ``RunResult`` and the slot is freed."""
    runtime = FakeRuntime()
    runtime.wait_error = TimeoutError()
    runtime.exit_code = 137
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    result = await _run(controller, job, timeout_s=7)

    assert result.timed_out is True
    assert "kill" in runtime.calls
    assert "remove" in runtime.calls
    assert runtime.wait_calls == [("c1", 7)]
    assert controller.has_active_run(job.id) is False
    await controller.shutdown()


async def test_run_once_byte_budget_reaches_the_runtime(queries):
    """The tail is bounded twice at READ time — line count AND total bytes."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    await _run(controller, job, log_tail=25)

    assert runtime.log_calls[-1] == ("c1", False, 25, _RUN_TAIL_MAX_BYTES)
    await controller.shutdown()


async def test_run_once_survives_a_row_deleted_mid_run(queries):
    """The stamp is a fresh-read RMW that silently skips a vanished row: the
    container already ran, so the caller must still get its result."""
    gate = asyncio.Event()
    runtime = _GatedRuntime(gate)
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    task = asyncio.create_task(_run(controller, job))
    await asyncio.wait_for(runtime.started.wait(), timeout=2)
    await queries.delete_service_checked(job.id, None)
    gate.set()

    result = await task
    assert result.exit_code == 0
    assert await queries.get_job(job.id) is None
    assert controller.has_active_run(job.id) is False
    await controller.shutdown()


async def test_run_once_audits_shared_resolved_keys(queries):
    """Invariants I2: a run consuming ``${secrets.shared.KEY}`` must not be the
    one launch-shaped resolution that leaves no trail. Key NAMES only."""
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(
        queries, runtime, secrets=FakeSecrets({"OPENAI_KEY": "sk-live-shared-value"})
    )
    job = _deploy_svc(
        "app",
        ai={"default": {**_API_SPEC, "api_key": "${secrets.shared.OPENAI_KEY}"}},
    )
    await queries.create_job(job)

    await _run(controller, job)

    items, _ = await queries.list_audit_log(action="secret.shared_resolved", limit=200)
    rows = [item.params_redacted or {} for item in items]
    assert len(rows) == 1
    assert rows[0]["service_name"] == "app"
    assert rows[0]["keys"] == ["OPENAI_KEY"]
    assert "sk-live-shared-value" not in json.dumps(rows)
    await controller.shutdown()


# --- PR #97 review round: the two Codex P1 findings ---------------------------


async def test_run_once_scrubs_a_caller_supplied_credential_override(queries):
    """A credential the CALLER passed is masked too (PR #97 Codex P1).

    ``_sensitive_env_values`` is computed on the PRE-overlay resolver output, so
    an ``--env APP_TOKEN=<live token>`` echoed by the command used to survive
    verbatim into the returned tail AND into ``config['last_run']``, which
    persists at rest in ``jobs.config`` and rides into ``POST /system/backup``.
    The applied overrides now get the same name heuristic and the same floor.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["auth header: APP_TOKEN=caller-live-token-value"]
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    result = await _run(controller, job, env_overrides={"APP_TOKEN": "caller-live-token-value"})

    tail = "\n".join(result.log_tail)
    assert "caller-live-token-value" not in tail
    assert "***" in tail
    # ...and the persisted mirror carries the masked copy, not the raw one.
    cfg = json.loads((await queries.get_job(job.id)).config)
    assert "caller-live-token-value" not in json.dumps(cfg)
    await controller.shutdown()


async def test_run_once_does_not_scrub_a_dropped_override(queries):
    """Only APPLIED overrides join the scrub set.

    An override of a protected key never reaches the container env, so it cannot
    be echoed — masking it could only mangle unrelated output that happens to
    contain the same string.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["port marker: 3333-not-a-secret"]
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    # PORT is protected, so this override is dropped before it reaches env.
    result = await _run(controller, job, env_overrides={"PORT": "3333-not-a-secret"})

    assert "3333-not-a-secret" in "\n".join(result.log_tail)
    await controller.shutdown()


async def test_run_once_stamp_never_rewrites_the_whole_config_blob(queries):
    """The ``last_run`` stamp must not be a read-modify-write (PR #97 Codex P1).

    A release is exempt from the run single-flight check (D-P20-2) and a
    redeploy can land at any moment, so a whole-blob writer would read the
    pre-release blob, await, and write it back — erasing ``image`` /
    ``build_version`` / the ``release_pending`` crash marker that D-P20-4 needs
    to refuse a silent swap after a crash mid-migration. Structural pin: the
    blob-rewriting seam is never touched by a run.
    """
    runtime = FakeRuntime()
    runtime.exit_code = 0
    controller = _controller(queries, runtime)
    job = _deploy_svc("app")
    await queries.create_job(job)

    calls: list[str] = []
    original = queries.update_job_config

    async def _spy(job_id, config):
        calls.append(job_id)
        return await original(job_id, config)

    queries.update_job_config = _spy  # type: ignore[method-assign]
    try:
        await _run(controller, job)
    finally:
        queries.update_job_config = original  # type: ignore[method-assign]

    assert calls == []
    assert json.loads((await queries.get_job(job.id)).config)["last_run"]
    await controller.shutdown()


async def test_last_run_stamp_preserves_siblings_written_after_the_caller_read(queries):
    """The ``last_run`` stamp is one statement over the value the row holds NOW.

    Reproduces the lost update the read-modify-write helper allowed: the caller
    snapshots the blob, a release commits ``release_pending`` into the same row,
    and only then does the stamp land. With ``json_set`` there is no read step
    to go stale, so the marker survives.
    """
    job = _deploy_svc("app")
    await queries.create_job(job)

    snapshot = json.loads((await queries.get_job(job.id)).config)  # the stale read
    assert "release_pending" not in snapshot

    # A concurrent release arms its crash marker against the same row.
    concurrent = dict(snapshot)
    concurrent["release_pending"] = 7
    await queries.update_job_config(job.id, json.dumps(concurrent))

    assert await queries.patch_job_config(job.id, {"last_run": {"run_id": "abc", "exit_code": 0}})

    cfg = json.loads((await queries.get_job(job.id)).config)
    assert cfg["release_pending"] == 7  # NOT erased by the stamp
    assert cfg["last_run"] == {"run_id": "abc", "exit_code": 0}
    assert cfg["image"] == snapshot["image"]  # every other sibling intact


async def test_last_run_stamp_reports_a_miss_instead_of_blanking_the_blob(queries):
    """A vanished row (or a non-JSON blob) is a ``False`` return, never a write.

    ``json_set`` answers NULL on an invalid document, and an unguarded UPDATE
    would then blank the column outright.
    """
    assert await queries.patch_job_config("no-such-job", {"last_run": {"run_id": "x"}}) is False

    job = _deploy_svc("app")
    await queries.create_job(job)
    await queries.update_job_config(job.id, "not json at all")

    assert await queries.patch_job_config(job.id, {"last_run": {"run_id": "x"}}) is False
    row = await queries.get_job(job.id)
    assert row.config == "not json at all"  # untouched, not blanked
