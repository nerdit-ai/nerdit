"""Reconcile long-running workloads toward persisted desired state.

Service rows and endpoint reservations preserve names and host ports across
restarts. Reconcile reads current rows and containers to adopt or relaunch
workloads after daemon restart. Persisted exit/backoff timestamps keep restart
budgets intact across reboot.

Failed health checks mark a live container degraded without killing it; later
success restores running. Only dead containers auto-restart. Autonomous state
transitions audit as `system` and publish status changes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import sqlite3
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from nerdit.config.project import shared_secret_keys_for
from nerdit.config.settings import (
    ContainerSettings,
    ModelsSettings,
    RetentionSettings,
    ServicesSettings,
)
from nerdit.core.app_build import (
    AppImageBuilder,
    _sensitive_env_values,
    _sensitive_override_values,
)
from nerdit.core.backup import DumpOutputError, _rmtree_staging, _verify_dump_output
from nerdit.core.cutover import CutoverManager
from nerdit.core.data.backend import DataBackend
from nerdit.core.data.binding import inject_db_env, resolve_db_bindings
from nerdit.core.data.controller import DataController
from nerdit.core.deploy_state import stamp_last_deploy
from nerdit.core.eventlog import EventRecorder, record_job_event
from nerdit.core.events import EventBus
from nerdit.core.health import (
    DEFAULT_HEALTH_TIMEOUT_S as _DEFAULT_HEALTH_TIMEOUT_S,
)
from nerdit.core.health import (
    DEFAULT_UNHEALTHY_THRESHOLD as _DEFAULT_UNHEALTHY_THRESHOLD,
)
from nerdit.core.health import (
    as_float as _as_float,
)
from nerdit.core.health import (
    as_int as _as_int,
)
from nerdit.core.health import (
    check_health,
    check_tcp,
    run_probe,
    within_start_period,
)
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.launch import (
    _RUN_LINE_MAX_BYTES,
    _RUN_TAIL_MAX_BYTES,
    AppContainerSpec,
    apply_platform_overlay,
    build_app_container_config,
    build_run_container_config,
    finalize_container_config,
    plan_gpu_placement,
    require_minted_credential,
    resolve_host_port,
    scrub_secret_values,
    settle_started,
    stamp_launching,
)
from nerdit.core.models.binding import (
    BindingNotReady,
    ResolvedBinding,
    inject_env,
    resolve_bindings,
)
from nerdit.core.models.controller import ModelController
from nerdit.core.proxy import ProxyManager
from nerdit.core.runtime.container import ContainerConfig
from nerdit.core.runtime.protocol import (
    ContainerNotFoundError,
    ContainerRuntime,
    ContainerRuntimeError,
    SandboxViolationError,
)
from nerdit.core.sandbox import enforce_mount_allowlist, resolve_role
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError, SecretManager
from nerdit.core.volumes import VolumeSpecError, create_dump_staging_dir, service_volumes
from nerdit.db.enums import ErrorClass, GpuVendor, JobKind, JobStatus, LogStream, TokenRole
from nerdit.db.queries import Queries
from nerdit.db.rows import Job
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

# Upper bound on health checks performed per reconcile tick (CRIT-5). Inline
# checks run inside the single sequential WorkloadManager loop, so an unbounded
# fan-out of slow endpoints would delay the reconcile cadence. The per-check
# httpx timeout bounds each probe; this bounds their cumulative cost per tick.
_MAX_HEALTH_CHECKS_PER_TICK = 20

# Hard ceiling on the exponential restart backoff (seconds).
_MAX_BACKOFF_S = 60.0


# --- (P37) Managed-database dump/restore constants ---------------------------

#: Where the dump staging dir is bind-mounted inside the sibling container
#: (D-P37-2). The daemon's ONLY mount into that container, and the only path
#: the backends' argv builders ever name.
_DUMP_MOUNT = "/nerdit-dump"

#: Tail budget for a dump/restore sibling. Same 200 lines a one-off run keeps,
#: bounded at read time and scrubbed before it is stored anywhere (D-P37-11).
_DUMP_LOG_TAIL = 200

#: Cadence of the two :meth:`ServiceController.quiesce_row` polls. Slow enough
#: to be free next to a container stop, fast enough that a Redis restore does
#: not visibly wait; monkeypatched down by the controller tests.
_QUIESCE_POLL_INTERVAL_S = 0.5

#: The Redis multi-part-AOF layout (verified live, plan §0). ``appendonlydir/``
#: is what a server booted ``--appendonly yes`` reads at startup; a bare
#: ``dump.rdb`` beside it is IGNORED, which is why the restore installs the RDB
#: as the base file of a hand-written one-entry manifest (D-P37-6).
_AOF_DIR_NAME = "appendonlydir"
_AOF_BASE_NAME = "appendonly.aof.1.base.rdb"
_AOF_MANIFEST_NAME = "appendonly.aof.manifest"
_AOF_MANIFEST_LINE = f"file {_AOF_BASE_NAME} seq 1 type b\n"

#: Stream chunk for copying an RDB into the prepared AOF dir (bounded RSS).
_AOF_COPY_CHUNK = 1024 * 1024


def _verify_dump_output_path(staging: Path, backend: "DataBackend") -> Path:
    """Verify the sibling's output. One verifier shared with the packer, so both tiers
    refuse a symlink, a 0-byte file or an extra entry with the same reason token.
    """
    try:
        return _verify_dump_output(staging, backend.dump_filename)
    except DumpOutputError as exc:
        raise DumpError(exc.reason) from exc


def _last_dump_payload(
    *,
    run_id: str,
    kind: str,
    started: datetime,
    command: list[str],
    outcome: dict,
) -> dict:
    """Build the ``config['last_dump']`` blob (D-P37-11).

    ``command`` is the argv verbatim (secret-free by D-P37-4); ``log_tail`` is the
    scrubbed tail and this blob is the only place the whole of it is kept. ``dump``
    is ``None`` here: the route packs the tar afterwards and patches the basename in.
    """
    return {
        "run_id": run_id,
        "kind": kind,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "command": list(command),
        "exit_code": outcome.get("exit_code"),
        "timed_out": bool(outcome.get("timed_out", False)),
        "reason": outcome.get("reason"),
        "dump": None,
        "log_tail": list(outcome.get("log_tail") or []),
    }


def _prepare_aof_dir(prepared: Path, source: Path) -> None:
    """Write a COMPLETE replacement ``appendonlydir`` next to the live one (D-P37-6).

    Blocking (worker thread). Files ``0o600`` under a ``0o700`` dir, all opened
    ``O_CREAT|O_EXCL|O_NOFOLLOW``; the RDB is read ``O_NOFOLLOW``. Complete before
    it is named, so the swap never exposes a half-written base.
    """
    prepared.mkdir(mode=0o700, exist_ok=False)
    src_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        dst_fd = os.open(
            prepared / _AOF_BASE_NAME,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            with (
                os.fdopen(src_fd, "rb", closefd=False) as src,
                os.fdopen(dst_fd, "wb", closefd=False) as dst,
            ):
                shutil.copyfileobj(src, dst, _AOF_COPY_CHUNK)
                dst.flush()
                os.fsync(dst_fd)
        finally:
            os.close(dst_fd)
    finally:
        os.close(src_fd)

    manifest_fd = os.open(
        prepared / _AOF_MANIFEST_NAME,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(manifest_fd, _AOF_MANIFEST_LINE.encode())
        os.fsync(manifest_fd)
    finally:
        os.close(manifest_fd)


def _swap_aof_dir(volume_dir: Path, prepared: Path) -> None:
    """Rename the live AOF dir aside, then the prepared one into its place.

    Blocking, under the quiesce. The previous data is renamed to
    ``appendonlydir.pre-restore-<stamp>``, never deleted (D-P37-6). A second rename
    that fails renames the aside back (the quiesce restarts the row
    unconditionally, and an append-only Redis with no live dir serves an empty
    dataset); post-state is either the live dir intact or both copies on the volume.
    """
    live = volume_dir / _AOF_DIR_NAME
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    aside: Path | None = None
    try:
        os.lstat(live)
    except FileNotFoundError:
        pass
    else:
        aside = volume_dir / f"{_AOF_DIR_NAME}.pre-restore-{stamp}"
        os.rename(live, aside)
    try:
        os.rename(prepared, live)
    except OSError:
        if aside is not None:
            with contextlib.suppress(OSError):
                os.rename(aside, live)
        raise


class LaunchEnvNotReady(Exception):  # noqa: N818 — domain condition, not an error (retry signal)
    """Launch env (secrets / shared / [ai.*] bindings) is not yet resolvable.

    Non-terminal, exactly like the return-based retry-next-tick exits it
    replaces (P14 WP-0 C2.1 factoring): the caller logs the wait once and defers
    the launch. `kind` selects the caller's dedupe/log channel:
    `"binding"` → `ServiceController._log_binding_wait`;
    `"secrets"` / `"shared_secrets"` → `_log_secret_wait`.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class ResolvedLaunchEnv:
    """Resolved container env plus inputs for caller-owned secret auditing.

    `env` combines config, secrets, bindings, and the PORT default. `secret_env`
    and `shared_keys` support auditing without side effects during resolution.
    `injected_keys` records actual binding output and protects run overrides;
    a names-only projection would miss engine-specific aliases such as Redis's
    `REDIS_URL`.
    """

    env: dict[str, str]
    secret_env: dict[str, str] = field(default_factory=dict)
    shared_keys: list[str] = field(default_factory=list)
    injected_keys: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RunResult:
    """Outcome of one bounded, rowless container execution.

    Produced by `ServiceController._execute_container_once` for both a
    one-off `run_once` and a `[deploy].release`. `exit_code` is `None`
    only when neither the bounded wait nor the post-mortem inspect could
    produce one; `timed_out` means the server-side cap fired and the
    container was killed. `log_tail` is already bounded at read time,
    **scrubbed** and only then per-line truncated (that order — see
    `_scrub_and_truncate`) — it is safe to return to the owner, to mirror
    into `config['last_run']` and (release only) to append to `job_logs`.
    """

    run_id: str
    exit_code: int | None
    timed_out: bool
    oom_killed: bool
    duration_s: float
    started_at: str
    finished_at: str
    log_tail: list[str]


class RunPreconditionError(Exception):
    """A one-off run cannot start; `reason` selects the caller's error code.

    `reason ∈ {"service_gone", "no_image", "run_in_progress",
    "too_many_runs"}` — the run route maps them to 404 `not_found`, 409
    `run.no_image`, 409 `service.run_in_progress` and 409
    `run.too_many_in_flight` respectively (P20 §1.2). Raised **before** any
    container exists, so there is never anything to clean up beyond the
    caller's registry slot.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# The settled-terminal desired_state values. `set_desired_state`
# (db/queries/services.py) documents the live domain as `running | stopped`, so
# only `stopped` is reachable today; the other three are carried deliberately as
# a superset (they are the terminal `status` values `get_reconcilable_services`
# filters on) so a future widening of the desired-state domain cannot silently
# let a run through. A run against a row the user has terminally stopped is
# refused as "service_gone" even though the row still physically exists.
_TERMINAL_DESIRED_STATES = frozenset({"completed", "cancelled", "stopped", "failed"})


class RunInterruptedError(ContainerRuntimeError):
    """Report a container that started but was lost during wait or forensics.

    `container_started` distinguishes this from start failure without importing
    controller types. `log_tail` is bounded, scrubbed, and clamped before exposure.
    The runtime-error base preserves existing caller exception handling.
    """

    container_started: bool = True

    def __init__(self, message: str, *, log_tail: list[str]) -> None:
        super().__init__(message)
        self.log_tail = log_tail


class RunMode(StrEnum):
    """The kind of rowless container execution a slot tracks (P37, D-P37-9).

    ``dump`` is single-flight per row like ``run``, exempt from
    ``max_concurrent_runs`` like ``release``, and capped by its own
    ``[services].max_concurrent_dumps``. Additive: ``is_release=`` stays valid on
    every P20 call site. A restore takes a ``dump`` slot; ``last_dump.kind`` tells
    them apart.
    """

    run = "run"
    release = "release"
    dump = "dump"


@dataclass(init=False)
class RunSlot:
    """One in-flight rowless container execution, tracked per job.

    `container_id` is None until runtime.run() returns; registration makes this
    window sweep-safe. `mode` selects the execution kind. `is_release` remains
    a derived property and constructor keyword for existing callers.
    Mutable by design: the container id is bound after the slot is registered.

    The explicit ``__init__`` (rather than the generated one) is what lets
    ``RunSlot(is_release=True)`` and ``RunSlot(mode=RunMode.dump)`` both be
    spelled: a dataclass field named ``is_release`` would collide with the
    property that replaced it.
    """

    container_id: str | None = None
    mode: RunMode = RunMode.run

    def __init__(
        self,
        container_id: str | None = None,
        mode: RunMode | None = None,
        *,
        is_release: bool = False,
    ) -> None:
        self.container_id = container_id
        if mode is None:
            mode = RunMode.release if is_release else RunMode.run
        self.mode = mode

    @property
    def is_release(self) -> bool:
        """``True`` for a ``[deploy].release`` slot (the P20 spelling, derived)."""
        return self.mode is RunMode.release


class DumpError(Exception):
    """A managed-database dump or restore failed (P37, D-P37-11).

    ``reason`` is one of the locked tokens shared by the response, the event and
    ``config['last_dump']``: ``data_plane_unavailable``, ``exit_nonzero``,
    ``timed_out``, ``empty_output``, ``output_not_regular``, ``unexpected_output``,
    ``interrupted``, ``quiesce_timeout``. ``log_tail`` is already scrubbed; the
    event payload carries ``reason`` only. Message is path-free (M3).
    """

    def __init__(self, reason: str, *, log_tail: list[str] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.log_tail = list(log_tail or [])


@dataclass(frozen=True)
class DumpResult:
    """Outcome of one SUCCESSFUL dump or restore (P37, §1.2); every failure is a
    :class:`DumpError`. ``exit_code`` is ``0`` (``None`` on the Redis restore path,
    which runs no container); ``output_path`` is the verified dump inside the
    staging dir, ``None`` for a restore.
    """

    exit_code: int | None
    timed_out: bool
    log_tail: list[str]
    output_path: str | None
    started_at: str
    finished_at: str


def _coerce_exit_code(state: object) -> int | None:
    """Best-effort read of `exit_code` off a runtime `inspect_state` result.

    Defensive: a `None` state, a missing attribute, or a non-`int` value
    (an un-specced mock's coroutine-of-MagicMock, a malformed `attrs`) degrade
    to `None` so forensics persistence never raises inside the crash path.
    """
    if state is None:
        return None
    value = getattr(state, "exit_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _coerce_oom(state: object) -> bool:
    """Best-effort read of `oom_killed` off a runtime `inspect_state` result.

    Non-`bool` / missing / garbage returns degrade to `False` (see
    `_coerce_exit_code`).
    """
    if state is None:
        return False
    value = getattr(state, "oom_killed", None)
    return value if isinstance(value, bool) else False


# Crash-tail capture bounds — enforced at READ time by the runtime, so a
# pathological container can never materialize more than this into the daemon.
_CRASH_TAIL_MAX_LINES = 100
_CRASH_TAIL_MAX_BYTES = 64 * 1024

# Markers that identify a CUDA/HIP *allocator* OOM in a captured tail.
# Deliberately narrow — a bare "out of memory" is cgroup-kill / user-error
# territory and must keep mapping to `ErrorClass.oom`/`user_error`.
_GPU_OOM_MARKERS = (
    "torch.outofmemoryerror",
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "hip out of memory",
    "hiperroroutofmemory",
    # vLLM's startup preflight refusal (observed live on vllm v0.25,
    # 2026-08-06): the engine refuses to claim more VRAM than is free BEFORE
    # the allocator can OOM — same GPU-sizing root cause, same fix, so it
    # belongs to gpu_oom (D2 amendment, plan §0).
    "is less than desired gpu memory utilization",
)


def _tail_indicates_gpu_oom(lines: Iterable[str]) -> bool:
    """True when a captured crash tail carries a CUDA/HIP allocator OOM marker.

    Case-insensitive substring match on the bounded captured tail only — never a
    streaming cost.
    """
    for line in lines:
        low = line.lower()
        if any(marker in low for marker in _GPU_OOM_MARKERS):
            return True
    return False


# Markers that identify a container whose
# ENTRYPOINT needed root privileges the sandbox drops (`cap_drop=["ALL"]` +
# `no_new_privileges`). Two shapes, both observed on stock root images:
#   1. a privileged syscall refused outright — ``chown(...) failed (1: Operation
#      not permitted)` (nginx's entrypoint on `/var/cache/nginx/client_temp``),
#      `setgid`/`setuid`/`capset` on the drop-privileges path;
#   2. a root-owned runtime path refused to a non-root uid — ``can't open
#      /var/run/nginx.pid ... Permission denied``.
# Deliberately narrow on BOTH halves: a bare "permission denied" is ordinary
# application/user error (a bad bind mount, an app writing into its own tree)
# and must keep mapping to `fix_start_command`.
_PRIV_SYSCALL_MARKERS = (
    "chown",
    "chmod",
    "setuid",
    "setgid",
    "setgroups",
    "setcap",
    "capset",
    "cannot set uid",
    "cannot set gid",
)
# Root-owned runtime roots a dropped-capability container cannot write. An app's
# own tree (`/app`, `/srv`, a mounted volume) is deliberately absent.
#
# These are FILESYSTEM ROOTS, so they must only match a path token that actually
# STARTS there: a plain substring test made ``Permission denied:
# '/app/run/state.pid'` match `/run/`` and misreport an ordinary app bug as
# `image_needs_privileges` (Codex, PR #138). The regex therefore requires a
# boundary immediately before the leading slash — start of line, whitespace
# (which covers the `: ` and `, ` separators), a quote, or an opening paren —
# i.e. the position where a path token begins in every observed error shape:
# `open("/var/run/nginx.pid")`, `... open /run/service: Permission denied`,
# `Permission denied: '/var/lib/x'`.
_PRIV_PATH_MARKERS = (
    "/var/run/",
    "/run/",
    "/var/cache/",
    "/var/lib/",
    "/usr/local/",
    "/etc/",
)
_PRIV_PATH_RE = re.compile(
    r"""(?:^|[\s"'(])(?:""" + "|".join(re.escape(p) for p in _PRIV_PATH_MARKERS) + r")"
)


def _tail_indicates_priv_denied(lines: Iterable[str]) -> bool:
    """True when a captured crash tail carries a dropped-privilege refusal.

    Case-insensitive substring match on the bounded captured tail only — never a
    streaming cost, exactly like `_tail_indicates_gpu_oom`. Both shapes
    require TWO independent markers (the refusal AND either a privileged syscall
    or a root-owned path) so an ordinary `Permission denied` on the app's own
    files never claims the sandbox is at fault. The path half is anchored to the
    START of a path token (`_PRIV_PATH_RE`), so `/app/run/state.pid` or
    `/srv/etc/config` — an app's own tree that merely *contains* a root name —
    stays `fix_start_command`.
    """
    for line in lines:
        low = line.lower()
        if "operation not permitted" in low and any(m in low for m in _PRIV_SYSCALL_MARKERS):
            return True
        if "permission denied" in low and _PRIV_PATH_RE.search(low):
            return True
    return False


def _classify_service_exit(
    exit_code: int | None, oom_killed: bool, gpu_oom: bool = False
) -> ErrorClass:
    """Map a crashing service/model exit to an `ErrorClass`
    (137→oom, 1..126→user_error, else container_fail; None→unknown;
    OOM-killer wins over the code map). `gpu_oom` — a CUDA/HIP allocator OOM matched on the
    captured crash
    tail — outranks everything, including the cgroup kill: the two have different
    fixes (GPU sizing vs `[deploy].memory_limit`), and in the pathological
    both-flags case the CUDA marker names the allocator that actually failed.
    `derive_remediation` (`daemon/remediation.py`) ranks the same way for
    model rows (architect ruling 2026-08-06), so a both-flags model crash reports
    class GPU_OOM with remediation `model.gpu_oom`; service rows have no
    gpu_oom remediation by D2's scoping and fall through to the cgroup rule.
    """
    if gpu_oom:
        return ErrorClass.gpu_oom
    if oom_killed:
        return ErrorClass.oom
    if exit_code is None:
        return ErrorClass.unknown
    if exit_code == 137:
        return ErrorClass.oom
    if 1 <= exit_code <= 126:
        return ErrorClass.user_error
    return ErrorClass.container_fail


def _truncate_line(line: str) -> str:
    """Clamp one captured run/release log line to `_RUN_LINE_MAX_BYTES`.

    The runtime's `max_bytes` budget bounds the tail as a whole; this bounds
    any single pathological line (a minified bundle, a base64 blob) so the
    per-line cap holds in the response, in `config['last_run']` and in
    `job_logs`. Measured and cut in **bytes**, then decoded with `ignore` so
    a cut inside a multi-byte character degrades to dropping that character
    rather than producing a mojibake row.
    """
    raw = line.encode("utf-8", "replace")
    if len(raw) <= _RUN_LINE_MAX_BYTES:
        return line
    return raw[:_RUN_LINE_MAX_BYTES].decode("utf-8", "ignore") + "…[truncated]"


def _scrub_and_truncate(lines: list[str], values: Iterable[str]) -> list[str]:
    """The D-P20-1 redaction pipeline for a captured tail: scrub, THEN clamp.

    The order is load-bearing, and it was the wrong way round once (PR #96
    review F2): clamping first cut a secret longer than `_RUN_LINE_MAX_BYTES`
    mid-value, so `scrub_secret_values`' literal match no longer found it
    and the clamped 2 KiB PREFIX of the secret reached the run response and —
    for a `[deploy].release` — `job_logs`, which ANY authenticated
    principal (`readonly` included) can read. The `core/gitsource.py`
    `_scrub` precedent is the template: scrub, then slice.

    Scrubbing the un-clamped lines stays bounded — the tail is capped at
    `_RUN_TAIL_MAX_BYTES` at READ time by the runtime, so this scans ~16 KiB
    however pathological the output was. Every captured tail composes the two
    steps HERE and nowhere else, so the order cannot drift per call site.
    """
    return [_truncate_line(line) for line in scrub_secret_values(lines, values)]


class ServiceController:
    """Reconciles desired vs. actual state for service/model workloads.

    The reconcile loop is owned by `nerdit.core.workload.WorkloadManager`,
    which calls `reconcile` on every tick — never concurrently with
    itself, so no two passes hold overlapping `BEGIN IMMEDIATE` locks on the
    shared connection.
    """

    def __init__(
        self,
        queries: Queries,
        runtime: ContainerRuntime,
        event_bus: EventBus | None = None,
        *,
        services_settings: ServicesSettings | None = None,
        container_settings: ContainerSettings | None = None,
        proxy: ProxyManager | None = None,
        proxy_mode: str = "path",
        secrets: SecretManager | None = None,
        model_controller: ModelController | None = None,
        data_controller: DataController | None = None,
        data_dir: str | None = None,
        retention_settings: RetentionSettings | None = None,
        events: EventRecorder | None = None,
    ) -> None:
        self._queries = queries
        self._runtime = runtime
        self._event_bus = event_bus
        # Durable event feed. Optional so a bare-constructed
        # controller in a unit test keeps working unchanged (the `event_bus`
        # precedent); `None` ⇒ every `record_job_event` call is a no-op.
        self._events = events
        # Root for daemon-computed per-service data dirs (named volumes,
        # `<data_dir>/services/<name>`) and the retention config the sweep
        # loop reads. Both None-defaulted so bare-constructed controllers are
        # unaffected; wired from `server.py`.
        self._data_dir = Path(data_dir).expanduser() if data_dir else None
        self._retention_settings = retention_settings
        # Per-service secret env. Injected into the container at launch;
        # None => no secret injection (services run with only their config env).
        self._secrets = secrets
        # Model-side hooks: off-tick image pull, backend container shape,
        # ensure_model. None => kind=model rows launch through the plain
        # service path (image must pre-exist), exactly as before P5.
        self._models = model_controller
        # Data-side hooks: off-tick image pull, backend container shape,
        # ensure_ready (wire-protocol readiness probe). None => kind=database rows
        # cannot be reconciled (the /databases route is only wired when a
        # DataController exists), mirroring the model_controller posture.
        self._data = data_controller
        # Host [ai.*] ollama bindings resolve their base_url on (platform-
        # resolved advertise host when [models].bridge_host = "auto").
        self._bridge_host = (
            model_controller.bridge_host
            if model_controller is not None
            else ModelsSettings().bridge_advertise_host
        )
        # URL layer: registered on RUNNING, deregistered on every terminal
        # transition. `None` (or disabled) => services run on loopback as in P2.
        self._proxy = proxy
        self._proxy_mode = proxy_mode
        # Defaults match production settings so bare-constructed controllers
        # (tests) exercise the same policy as the daemon.
        self._services_settings = services_settings or ServicesSettings()
        self._container_settings = container_settings or ContainerSettings()
        self._port_range = self._parse_port_range(self._services_settings.service_port_range)
        self._max_restarts = self._services_settings.service_max_restarts
        self._restart_window_seconds = self._services_settings.restart_window_seconds
        # Daemon-wide cap on concurrent route-initiated runs.
        # Restart-required key, so snapshotting at construction is exact.
        self._max_concurrent_runs = self._services_settings.max_concurrent_runs
        # (P37 D-P37-9) The dump/restore pool, kept SEPARATE from the run cap
        # above: the two bound different resources (a run burns the image's
        # CPU, a dump burns daemon disk and the database's own replication
        # bandwidth), so neither borrows from the other. Same restart-required
        # snapshot discipline.
        self._max_concurrent_dumps = self._services_settings.max_concurrent_dumps
        # Background log-collection tasks keyed by container_id, so a specific
        # service's stream can be cancelled on stop and re-adopted after a reboot.
        self._log_tasks: dict[str, asyncio.Task] = {}
        # In-flight image builds keyed by job id. A deploy build can take
        # minutes, so it runs off the shared reconcile tick as a background task;
        # this guards against spawning a second build for the same row while one
        # is still running.
        self._build_tasks: dict[str, asyncio.Task] = {}
        # In-flight ROWLESS container executions keyed by job id, then by
        # run id: one-off runs (`run_once`) AND `[deploy].release` executions.
        # Neither has a `jobs` row (D-C), so this registry is the ONLY thing
        # that (a) blocks a concurrent DELETE, (b) enforces the D-P20-2 caps and
        # (c) protects the container from the zombie sweep once it ages past
        # ZOMBIE_AGE_THRESHOLD_SECONDS. In-memory only: a daemon restart forgets
        # every slot, which is correct — the restart kills the runs, and the
        # boot-side sweep is then *meant* to reap their now-unprotected
        # containers.
        self._active_runs: dict[str, dict[str, RunSlot]] = {}
        # Per-token in-flight run counter backing the route's quota claim
        # against `max_concurrent_jobs`. Runs have no row, so `count_active_jobs`
        # cannot see them; this is the pending-side addend. Same in-memory
        # caveat as `_active_runs` (a restart forgets, and kills, both).
        self._run_counts: dict[str, int] = {}
        # In-flight health-gated cutovers keyed by job id: the verify
        # task, and the GREEN container it launched. Both live here rather than
        # on `CutoverManager` for the same reason `_build_tasks` does — the
        # sweeper hook, the DELETE guard and the restart drain all reach the
        # controller, not the manager. The green is rowless for the length of
        # the window (no `jobs` row names it), so this registry is the ONLY
        # thing protecting it from the zombie sweep. In-memory only: a daemon
        # restart forgets both, which is correct — the persisted
        # `cutover_pending` marker is what settles the generation afterwards.
        self._cutover_tasks: dict[str, asyncio.Task] = {}
        self._cutover_containers: dict[str, str] = {}
        # Construct after services settings, which the builder snapshots.
        self._builder = AppImageBuilder(self)
        # The health-gated cutover, beside the builder and reaching
        # this controller through `self._c` (the same D-T-2 patch-point
        # contract). Snapshots the `[services]` cutover trio at construction.
        self._cutover = CutoverManager(self)
        # Consecutive health-check failures per job id (in-memory: resets on
        # reboot, which is fine — degraded is observable, never load-bearing).
        self._health_failures: dict[str, int] = {}
        # Last BindingNotReady message logged per job id: the wait line is
        # written to job_logs once per DISTINCT message, not once per tick.
        self._binding_wait_msgs: dict[str, str] = {}
        # Last (keys, overridden) tuple audited as secret.shared_resolved per
        # job id: a crash-looping service re-audits only when the tuple
        # actually changes, never once per relaunch.
        self._shared_resolved_sigs: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        # Drain gate: while a `POST /daemon/restart` is draining, no new
        # image build is *spawned* — the reconcile tick keeps accepting deploys
        # but parks them at phase "queued" (row stays `building`), and the fresh
        # daemon re-detects `_needs_build` after the re-exec. Gating at the
        # spawn point (never registering the task) avoids the build-context
        # rmtree in `_build_app_image`'s `finally` that an early in-build return
        # would trigger.
        self._draining = False

    @property
    def draining(self) -> bool:
        """Whether a graceful restart drain is in progress."""
        return self._draining

    @draining.setter
    def draining(self, value: bool) -> None:
        self._draining = bool(value)

    def busy_builds(self) -> int:
        """Return the number of image builds still running for restart drain."""
        return self._builder.busy()

    # --- rowless-run registry -----------------------------------------
    # One-off runs and `[deploy].release` executions are containers with no
    # `jobs` row (D-C), so nothing else in the system knows they exist. Every
    # method here is SYNC and mutates only in-process dicts, which makes each
    # one atomic on the event loop — the property `_register_run` relies on to
    # enforce D-P20-2 without a lock.

    def has_active_run(self, job_id: str) -> bool:
        """Whether a run or release is currently executing for `job_id`.

        Called synchronously by the DELETE route (`daemon/service_purge.py`),
        which 409s `service.run_in_progress` rather than tearing a service
        down under a live migration. Releases count: deleting a service whose
        release container is mid-migration is exactly as unsafe as deleting one
        mid-run (behaviour change vs pre-P20, where a delete during a build
        succeeded).
        """
        return bool(self._active_runs.get(job_id))

    def active_run_container_ids(self) -> set[str]:
        """Container ids of every bound run/release slot (zombie-sweep hook).

        Passed to `nerdit.core.sweeper.ZombieSweeper` at construction as
        `extra_protected`: these containers are managed-by-nerdit and older
        than the sweeper's age gate within seconds, but have no row for the
        sweep to match them against, so without this they would be killed
        mid-run.
        """
        return {
            cid for job_id in self._active_runs for cid in self.active_run_container_ids_for(job_id)
        }

    async def kill_active_run_containers(self) -> int:
        """Kill bound run/release containers at the restart drain deadline.

        Kill without removing so blocked waiters can collect scrubbed logs and own
        cleanup. Unbound slots have no container ID; graceful-shutdown timeout bounds
        that residual. Killed releases settle normally or through their persisted
        crash marker at boot. Skip runtime failures per container, logging IDs only.

        Returns:
            Number of containers killed.
        """
        container_ids = self.active_run_container_ids()
        killed = 0
        for container_id in container_ids:
            try:
                await self._runtime.kill(container_id)
            except ContainerNotFoundError:
                continue  # already gone: nothing to kill
            except ContainerRuntimeError as exc:
                # Usually "container not running" — the run exited on its own
                # between busy_runs() reporting it and this kill, which is a
                # clean shutdown, not a failure. Debug, so an ordinary restart
                # does not log a warning it cannot act on. Id truncated to match
                # every sibling kill path.
                logger.debug(
                    "Drain deadline: could not kill run container %s (%s)", container_id[:12], exc
                )
                continue
            killed += 1
        if killed:
            logger.warning(
                "Drain deadline expired — killed %d in-flight run/release container(s)", killed
            )
        return killed

    def active_run_container_ids_for(self, job_id: str) -> set[str]:
        """Container ids of the bound run/release slots of ONE job.

        The escape hatch behind `DELETE /services/{ident}?force=true`: a
        wedged run or migration must not pin a service undeletable until the
        daemon restarts, so the forced path kills exactly *this* service's run
        containers — the daemon-wide set above would take unrelated services'
        runs down with it.

        Unbound slots (`container_id is None`) are omitted: there is nothing
        to kill yet. They are NOT harmless, though — see
        `has_unbound_run`, which the forced-delete path pairs with this
        one so a container that is *about* to start cannot race a data purge.
        """
        return {
            slot.container_id
            for slot in self._active_runs.get(job_id, {}).values()
            if slot.container_id is not None
        }

    def has_unbound_run(self, job_id: str) -> bool:
        """Whether `job_id` holds a run/release slot with no container yet.

        A slot is claimed BEFORE any awaited work (so a DELETE is refused for
        the whole operation) and bound only once `runtime.run()` returns a
        container id. For a release, everything between the two is real work:
        arming the crash marker, resolving the launch env (DB reads + secret
        decryption), materializing volume dirs and the container create itself
        — precisely the window a hung `runtime.run()` sits in, which is one
        of the reasons `?force` exists.

        An unbound slot cannot be killed, so a forced delete must treat it as a
        writer it could not stop: the container may bind-mount the service data
        tree microseconds after the purge rmtree'd it, leaving a root-owned
        orphan the operator can only reclaim through `nerdit gc`.
        """
        return any(slot.container_id is None for slot in self._active_runs.get(job_id, {}).values())

    def has_any_unbound_run(self) -> bool:
        """Return whether any run/release slot is waiting for its container ID.

        A container becomes Docker-visible before `runtime.run` returns. The boot
        sweeper must defer its whole pass during this gap or it could kill a live
        migration as an orphan. A fixed delay cannot close that race.
        """
        return any(
            slot.container_id is None
            for slots in self._active_runs.values()
            for slot in slots.values()
        )

    def busy_runs(self) -> int:
        """Number of run/release containers still executing (restart-drain probe).

        Companion to `busy_builds`: the `POST /daemon/restart` drain
        predicate must count BOTH, or a re-exec kills a live migration the
        drain was supposed to wait out. Releases are included precisely because
        they are the dangerous half. The drain timeout still bounds the wait,
        and the bound is real: at the deadline the drain calls
        `kill_active_run_containers`, so a run longer than the operator's
        budget is killed *there* (bounded restart > unbounded wait) rather than
        left to outlive the shutdown. This count is also reported verbatim as
        the 202 body's `in_flight_runs`.
        """
        return sum(len(slots) for slots in self._active_runs.values())

    # --- cutover registry ----------------------------------------
    # A cutover GREEN is a second live container for a row that already has
    # one, so it is invisible to every DB query the sweep makes and must not be
    # torn down by a concurrent DELETE. Same posture, and the same three hooks,
    # as the P20 rowless-run registry above.

    def has_active_cutover(self, job_id: str) -> bool:
        """Whether a cutover verify is in flight for `job_id`.

        Called synchronously by the DELETE route and the rollback route, which
        409 `service.cutover_in_progress` rather than tearing the service down
        (or lowering `build_version`) under a live verify.
        """
        return self._cutover.has_active(job_id)

    def has_unbound_cutover(self, job_id: str) -> bool:
        """An in-flight verify not yet bound to a green container id.

        The `has_unbound_run` twin — the forced-delete data purge fails
        closed on it (see `CutoverManager.has_unbound`).
        """
        return self._cutover.has_unbound(job_id)

    def cutover_container_ids(self) -> set[str]:
        """Green container ids of every in-flight cutover, daemon-wide."""
        return self._cutover.container_ids()

    def protected_container_ids(self) -> set[str]:
        """Every rowless container the zombie sweep must not kill.

        The `extra_protected` hook wired into
        `nerdit.core.sweeper.ZombieSweeper`: run/release containers ∪ cutover greens.
        `active_run_container_ids` stays as-is
        for its other callers.
        """
        return self.active_run_container_ids() | self.cutover_container_ids()

    def busy_cutovers(self) -> int:
        """Number of cutover verifies still running (restart-drain probe)."""
        return self._cutover.busy()

    async def cancel_cutover(self, job_id: str) -> set[str]:
        """Cancel an in-flight verify; unwind it or finish it, depending on side.

        Three callers share ONE semantic — never a settle: the
        `DELETE /services/{ident}?force=true` escape hatch, the
        `desired_state == "stopped"` branch of the reconcile, and the
        `POST /daemon/restart` drain deadline. It has two branches, split on
        the commit point: **pre-commit** the green is destroyed, the endpoint
        pointer goes back to blue's own port and the marker is popped;
        **post-commit** the green IS the row's own live container, so it is
        never destroyed and the pointer never rewound — a marker still armed
        means the tail is unfinished, and the cancel completes it. Neither
        branch stamps the generation `failed`: an operator-initiated cancel
        is not a verification verdict. Returns the green container ids it
        destroyed (the drain counts them), empty once promoted.
        """
        return await self._cutover.cancel(job_id)

    async def kill_transient_containers(self) -> int:
        """Cancel cutovers and kill rowless containers at the drain deadline.

        Cancel verification before killing green: a racing commit could otherwise
        promote a dead container and destroy blue. Pre-commit cancellation restores
        blue without marking failure, allowing a post-restart retry. Promoted green
        survives; raw label-based kills cover only candidates with no live task.

        Returns:
            Number of transient containers killed.
        """
        killed = await self.kill_active_run_containers()
        for job_id in list(self._cutover_tasks):
            killed += len(await self.cancel_cutover(job_id))
        for container_id in self.cutover_container_ids():
            try:
                await self._runtime.kill(container_id)
            except ContainerNotFoundError:
                continue
            except ContainerRuntimeError as exc:
                logger.debug(
                    "Drain deadline: could not kill cutover container %s (%s)",
                    container_id[:12],
                    exc,
                )
                continue
            killed += 1
        return killed

    def _register_run(
        self,
        job_id: str,
        run_id: str,
        *,
        is_release: bool = False,
        mode: RunMode | None = None,
    ) -> None:
        """Claim a run slot atomically before the first await.

        Runs and dumps require no existing slot for the row and obey separate
        daemon-wide caps. Releases bypass both. `is_release` derives the mode
        for existing callers. Callers must discard the slot in `finally` or
        deletion remains blocked.

        Raises:
            RunPreconditionError: A per-row or per-mode cap is exceeded.
        """
        if mode is None:
            mode = RunMode.release if is_release else RunMode.run
        if mode is not RunMode.release:
            if self._active_runs.get(job_id):
                raise RunPreconditionError("run_in_progress")
            cap = self._max_concurrent_dumps if mode is RunMode.dump else self._max_concurrent_runs
            in_flight = sum(
                1
                for slots in self._active_runs.values()
                for slot in slots.values()
                if slot.mode is mode
            )
            if in_flight >= cap:
                raise RunPreconditionError(
                    "too_many_dumps" if mode is RunMode.dump else "too_many_runs"
                )
        self._active_runs.setdefault(job_id, {})[run_id] = RunSlot(mode=mode)

    def _bind_run_container(self, job_id: str, run_id: str, container_id: str) -> None:
        """Attach the started container id to an already-registered slot.

        No-op when the slot is gone (the caller's `finally` already discarded
        it): binding must never resurrect a released slot.
        """
        slot = self._active_runs.get(job_id, {}).get(run_id)
        if slot is not None:
            slot.container_id = container_id

    def reserve_dump_slot(self, job_id: str, run_id: str) -> None:
        """Claim the row's dump slot for a caller that will pass ``slot_held=True``
        to :meth:`restore_database` and release it in its own ``finally``
        (D-P37-9). Raises :class:`RunPreconditionError`."""
        self._register_run(job_id, run_id, mode=RunMode.dump)

    def release_dump_slot(self, job_id: str, run_id: str) -> None:
        """Release a slot taken by :meth:`reserve_dump_slot`. Idempotent."""
        self._discard_run(job_id, run_id)

    def _discard_run(self, job_id: str, run_id: str) -> None:
        """Release a run slot. Idempotent — safe to call from a `finally`."""
        slots = self._active_runs.get(job_id)
        if slots is None:
            return
        slots.pop(run_id, None)
        if not slots:
            self._active_runs.pop(job_id, None)

    def claim_run_slot(self, token_id: str) -> int:
        """Increment and return the token's in-flight run count (quota claim).

        Sync so the increment lands before the caller's `await` on the DB
        read, making a concurrent claimer see it. The returned value is the
        pending count *including* this claim — the route adds it to the
        DB-visible active-job count before comparing against the token's
        `max_concurrent_jobs`. Always paired with `release_run_slot` in
        a `finally`.
        """
        pending = self._run_counts.get(token_id, 0) + 1
        self._run_counts[token_id] = pending
        return pending

    def release_run_slot(self, token_id: str) -> None:
        """Decrement the token's in-flight run count (floor 0, drop key at 0)."""
        pending = self._run_counts.get(token_id)
        if pending is None:
            return
        if pending <= 1:
            self._run_counts.pop(token_id, None)
        else:
            self._run_counts[token_id] = pending - 1

    async def run_once(
        self,
        job: Job,
        *,
        command: list[str],
        env_overrides: dict[str, str] | None = None,
        timeout_s: int,
        log_tail: int = 200,
    ) -> RunResult:
        """Run a command in the service's current image and remove its temporary container.

        The container has no ports, GPUs, endpoint, or proxy route. It shares resolved
        env, sandbox limits, and live named volumes, but no host-mounted scripts.
        Persistence is limited to `last_run` and shared-secret auditing.

        Pass command argv verbatim, without shell wrapping: Docker replaces CMD and
        appends the argv to any image ENTRYPOINT. Only owner-visible `last_run` stores
        argv; audit records use hashes. Route authorization and timeout clamping are
        the caller's responsibility.

        Applied env overrides beat config and secrets except actual injected binding
        keys, PORT, and NERDIT_RUN_ID. These platform values always win; unlike normal
        launch, a run overwrites PORT with the deploy port. Derive protected keys from
        live resolution so engine-specific aliases such as REDIS_URL are protected.

        Claim the run slot before awaiting and release it on every exit. Scrub tails
        with the release's sensitive-value set plus applied overrides, then clamp.
        Literal scrubbing cannot mask transformed values or runtime head-clip fragments;
        unrecognized credential names need explicit secret declaration.

        Args:
            command: Argv to execute in the deployed image.
            timeout_s: Hard execution timeout in seconds, already clamped by the route.

        Returns:
            RunResult, including nonzero exits as outcomes rather than exceptions.

        Raises:
            RunPreconditionError: Service/image absent or a run cap is exceeded.
            LaunchEnvNotReady: Launch bindings or secrets cannot resolve.
            VolumeSpecError: Persisted named-volume configuration is invalid.
            ContainerRuntimeError: The container could not start.
            RunInterruptedError: The container started but was lost afterward.
        """
        run_id = generate_id()
        # Sync, before the first await: blocks DELETE and enforces D-P20-2
        # from this instant. Raises run_in_progress / too_many_runs with NO
        # slot inserted, so those two paths need no cleanup.
        self._register_run(job.id, run_id, is_release=False)
        try:
            # Fresh re-read: the route's row is stale by the time we hold the
            # slot; this closes the resolve->launch delete window to near-zero.
            fresh = await self._queries.get_job(job.id)
            if fresh is None or fresh.desired_state in _TERMINAL_DESIRED_STATES:
                raise RunPreconditionError("service_gone")
            # Defence in depth behind the route's kind guard (WP4 answers a
            # non-service row with 422 `run.not_supported`). A real refusal
            # rather than an `assert`: an optimised interpreter (``python
            # -O``) deletes an assert, and until that route ships this is the
            # ONLY gate — a model or database row would otherwise launch its
            # own image with its own env. Reuses the locked `service_gone`
            # reason instead of widening the error set.
            if fresh.kind is not JobKind.service:
                raise RunPreconditionError("service_gone")
            cfg = parse_job_config(fresh, warn=True)

            image = cfg.get("image")
            if not isinstance(image, str) or not image:
                raise RunPreconditionError("no_image")
            if not await self._runtime.image_exists(image):
                raise RunPreconditionError("no_image")

            # Resolver inputs, verbatim from `_launch` / `_execute_release`.
            # The model/data suppression is defence in depth for a kind=service
            # row.
            container_port = int(cfg.get("port") or 8000)
            is_model = fresh.kind is JobKind.model and self._models is not None
            is_data = fresh.kind is JobKind.database and self._data is not None
            ai_specs = cfg.get("ai") if not (is_model or is_data) else None
            db_specs = cfg.get("db") if not (is_model or is_data) else None
            # LaunchEnvNotReady PROPAGATES (no _log_binding_wait, no dedupe-map
            # pop, no swallow): a run is caller-facing, not retry-next-tick.
            resolved = await self._resolve_launch_env(
                fresh, cfg, container_port, ai_specs, db_specs
            )
            scrub = _sensitive_env_values(resolved)
            # Shared-scope audit parity — same helper, same positional
            # shape as `_launch`; dedupes per row.
            if resolved.shared_keys:
                await self._audit_shared_resolved(fresh, resolved.shared_keys, resolved.secret_env)

            # --- the §1.1 protected-key overlay, verbatim ---
            env = dict(resolved.env)  # copy: never mutate the resolver's dict
            protected = resolved.injected_keys | {"PORT", "NERDIT_RUN_ID"}
            overrides = env_overrides or {}
            applied: dict[str, str] = {}
            for key, value in overrides.items():
                if key not in protected:
                    env[key] = value  # override beats config env AND secrets
                    applied[key] = value
            env["PORT"] = str(container_port)  # plain update — platform wins
            env["NERDIT_RUN_ID"] = run_id  # (stricter than _launch's setdefault)
            # The scrub set is computed on the PRE-overlay resolver output, so a
            # credential the CALLER supplied is not in it. Same name heuristic,
            # same floor — an `--env APP_TOKEN=…` is as dangerous as a
            # deploy-supplied one, and `last_run` persists at rest.
            scrub |= _sensitive_override_values(applied)

            # Named volumes, exactly as the release mounts them; the
            # _data_dir-None guard comes FIRST (bare controllers have none).
            # VolumeSpecError propagates (route -> 422 run.volume_invalid) —
            # never _settle_launch_user_error (the row is not ours to settle).
            named_volumes: dict[str, str] = {}
            if self._data_dir is not None and fresh.service_name and cfg.get("volumes"):
                named_volumes = service_volumes(self._data_dir, fresh.service_name, cfg)
                if named_volumes:
                    self._ensure_volume_dirs(fresh.service_name, named_volumes)

            config = build_run_container_config(
                cfg,
                self._container_settings,
                image=image,  # the gated row image — never cs.default_image
                command=list(command),  # exec form, no shell wrap
                env=env,
                workdir=None,  # the image's own WORKDIR, as the release
            )
            finalize_container_config(config, named_volumes, self._retention_settings)

            result = await self._execute_container_once(
                fresh,
                run_id=run_id,
                config=config,
                timeout_s=timeout_s,
                log_tail=log_tail,
                scrub_values=scrub,
            )

            # The ONLY persistence: config['last_run'], written by the
            # single-statement `set_last_run` — NOT `_stamp_config`. A run is
            # long-lived and a release is exempt from the run single-flight
            # check, so a redeploy or a release legitimately commits
            # to this row while the container is still going: a whole-blob
            # read-modify-write would erase `image`/`build_version`/
            # `release_pending` from under it. The guarded CAS twin is wrong
            # here too — its expectation is generation-scoped, and `last_run` is
            # not, so a redeploy would silently drop the stamp instead of
            # merging it. A DB error must not eat the result the container
            # already produced; a `False` return just means the row vanished (or
            # its blob is not valid JSON) mid-run.
            try:
                stamped = await self._queries.set_last_run(
                    fresh.id,
                    json.dumps(
                        {
                            "run_id": run_id,
                            "command": list(command),  # verbatim — owner-gated
                            "exit_code": result.exit_code,
                            "timed_out": result.timed_out,
                            "oom_killed": result.oom_killed,
                            "started_at": result.started_at,
                            "finished_at": result.finished_at,
                            "duration_s": result.duration_s,
                            # APPLIED, not submitted (names only). The D-P14-5
                            # overlay silently drops any override that collides
                            # with a platform-computed key, so recording the
                            # submitted set would confirm a caller's false
                            # belief: an `--env DATABASE_URL=…` aimed at a
                            # scratch DB is dropped, the command runs against
                            # the live one, and `last_run` would say the
                            # override was honoured. The dropped set is
                            # recorded beside it so the divergence is visible
                            # rather than merely absent.
                            "env_override_keys": sorted(applied),
                            "env_override_keys_dropped": sorted(set(overrides) - set(applied)),
                            "log_tail": result.log_tail,  # already scrubbed+clamped
                        }
                    ),
                )
                if not stamped:
                    logger.warning(
                        "last_run stamp for service %s (run %s) matched no row",
                        fresh.service_name or fresh.id,
                        run_id,
                    )
            except Exception:  # noqa: BLE001 — bookkeeping must not mask the result
                logger.warning(
                    "Could not stamp last_run for service %s after run %s",
                    fresh.service_name or fresh.id,
                    run_id,
                    exc_info=True,
                )
            return result
        finally:
            # Unconditional (S1): every raise path above leaves the registry
            # empty, or DELETE is wedged for this service until restart.
            self._discard_run(job.id, run_id)

    # --- (P37) Managed-database dumps and restores --------------------------

    def _dump_backend(self, cfg: dict) -> DataBackend:
        """Return the row's data backend, or raise ``data_plane_unavailable``.

        The one precondition the controller owns (§1.2, D-P37-10); every other check
        is the route's. Sync and before :meth:`_register_run`, so no slot is claimed.
        """
        if self._data is None or self._data_dir is None:
            raise DumpError("data_plane_unavailable")
        return self._data.backend_for(cfg)

    @property
    def _dump_bridge_host(self) -> str:
        """Host the dump sibling dials (D-P15-4): the ``DataController``'s bridge host,
        the same path every ``[db.*]`` binding takes.
        """
        assert self._data is not None  # guaranteed by _dump_backend
        return self._data.bridge_host

    async def _dump_endpoint_port(self, job: Job) -> int:
        """Return the row's published host port, or raise ``data_plane_unavailable``.

        The route already checked the endpoint exists (D-P37-10); this is the
        fresh re-read that closes the window between that check and the launch
        — a delete or a terminal transition releases the reservation, and a
        sibling dialing a freed port would reach whatever took it next.
        """
        endpoint = await self._queries.get_service_endpoint(job.service_name or "")
        if endpoint is None:
            raise DumpError("data_plane_unavailable")
        # ``live_port``, never ``host_port``: a database row never cuts over
        # today, but reading the reserved port directly is exactly the D-P24-4b
        # mistake — the sibling must dial whatever the serving container is
        # actually bound to.
        return endpoint.live_port

    async def _managed_password(self, job: Job, backend: DataBackend) -> str:
        """Load the row's minted credential off the loop (D-P37-4). It travels only into
        the sibling's env and the ``scrub_values`` set — never argv, response, audit
        or ``config['last_dump']``.
        """
        if self._secrets is None or not job.service_name:
            raise DumpError("data_plane_unavailable")
        try:
            secrets = await asyncio.to_thread(self._secrets.load, job.service_name)
        except SecretDecryptError as exc:
            raise DumpError("data_plane_unavailable") from exc
        password = secrets.get(backend.minted_secret_key)
        if not password:
            # Same posture as ``require_minted_credential`` at launch: a purged
            # or corrupt secret scope is honestly "this database is not
            # reachable", never a dump run without authentication.
            raise DumpError("data_plane_unavailable")
        return password

    def _create_staging(self, run_id: str) -> Path:
        """Create this run's ``0o700`` staging dir via
        :func:`nerdit.core.volumes.create_dump_staging_dir` (D-P37-2), which owns the
        grammar and the symlink refusals. Sync, between the slot claim and the launch.
        """
        assert self._data_dir is not None  # guaranteed by _dump_backend
        return create_dump_staging_dir(self._data_dir, run_id)

    def _dump_container_config(
        self,
        cfg: dict,
        *,
        image: str,
        argv: list[str],
        env: dict[str, str],
        staging: Path,
    ) -> ContainerConfig:
        """Build the sibling's container config (D-P37-1/2/3): the P20 run shape
        (portless, GPU-less, ``cap_drop=ALL``, ``no_new_privileges``, bridge) plus
        ``user`` = the daemon's uid:gid and the staging dir as the ONLY mount, at
        ``/nerdit-dump``. The data volume is never mounted.
        """
        config = build_run_container_config(
            cfg,
            self._container_settings,
            image=image,
            command=argv,
            env=env,
            workdir=None,
        )
        config.user = f"{os.getuid()}:{os.getgid()}"
        finalize_container_config(config, {str(staging): _DUMP_MOUNT}, self._retention_settings)
        return config

    async def _stamp_last_dump(self, job: Job, payload: dict) -> None:
        """Write ``config['last_dump']`` best-effort via the single-statement
        ``json_set`` (:meth:`Queries.set_last_dump`) — a redeploy may commit to the
        row meanwhile. Failures are logged and swallowed (D-P37-11).
        """
        try:
            stamped = await self._queries.set_last_dump(job.id, json.dumps(payload))
            if not stamped:
                logger.warning(
                    "last_dump stamp for database %s (run %s) matched no row",
                    job.service_name or job.id,
                    payload.get("run_id"),
                )
        except Exception:  # noqa: BLE001 — bookkeeping must not mask the result
            logger.warning(
                "Could not stamp last_dump for database %s after run %s",
                job.service_name or job.id,
                payload.get("run_id"),
                exc_info=True,
            )

    async def dump_database(
        self,
        job: Job,
        *,
        run_id: str,
        timeout_s: int,
        on_captured: Callable[[DumpResult], Awaitable[None]] | None = None,
    ) -> DumpResult:
        """Capture a logical, application-consistent dump of a managed database
        (P37 WP2, D-P37-1/2/3/4/9/11).

        A rowless sibling from the row's own image runs the backend's dump tool
        against ``bridge_host:<published port>`` and writes into a fresh ``0o700``
        staging dir bind-mounted at ``/nerdit-dump``. No ``docker exec``, no host
        tool, no volume read. Returns the verified output path; it does NOT pack.

        ``on_captured`` is the route's packer, awaited while the slot is still held
        (D-P37-9): the callback runs inside the try with ``outcome["reason"]`` already
        ``None``, the ``finally`` leaves staging to the packer, and if it raises the
        row is stamped ``interrupted`` and the caller re-stamps the precise token.
        On every failure here the staging dir is removed.

        Credentials (D-P37-4): the minted password is the sibling's entire env and a
        ``scrub_values`` entry; the argv is secret-free and recorded verbatim.

        The ``dump`` slot is claimed before the first await; the drain and the boot
        orphan kill see the sibling like any run.

        Raises :class:`DumpError` (``reason`` per D-P37-11) and
        :class:`RunPreconditionError` (``run_in_progress`` | ``too_many_dumps``, no
        slot claimed); a start-time ``ContainerRuntimeError`` propagates as-is, the
        staging dir still removed and the slot still released.
        """
        cfg = parse_job_config(job, warn=True)
        backend = self._dump_backend(cfg)
        image = cfg.get("image")
        if not isinstance(image, str) or not image:
            raise DumpError("data_plane_unavailable")

        # Sync, before the first await (D-P37-9): blocks DELETE and enforces the
        # dump caps from this instant. A refusal claims no slot, so there is
        # nothing to release on either raise.
        self._register_run(job.id, run_id, mode=RunMode.dump)
        started = datetime.now(UTC)
        staging: Path | None = None
        argv: list[str] = []
        outcome: dict = {"reason": "interrupted"}
        try:
            host_port = await self._dump_endpoint_port(job)
            password = await self._managed_password(job, backend)
            staging = self._create_staging(run_id)
            argv = backend.dump_argv(
                self._dump_bridge_host, host_port, f"{_DUMP_MOUNT}/{backend.dump_filename}"
            )
            config = self._dump_container_config(
                cfg,
                image=image,
                argv=argv,
                env=backend.dump_env(password),
                staging=staging,
            )
            try:
                result = await self._execute_container_once(
                    job,
                    run_id=run_id,
                    config=config,
                    timeout_s=timeout_s,
                    log_tail=_DUMP_LOG_TAIL,
                    scrub_values={password},
                )
            except RunInterruptedError as exc:
                # The sibling started and the runtime then lost it. Its tail is
                # already scrubbed through the same pipeline as the happy path.
                raise DumpError("interrupted", log_tail=exc.log_tail) from exc
            # ``update``, never a rebind: dropping the pre-seeded ``reason``
            # would make an unexpected exception below look like a success to
            # the ``finally`` and leave the staging dir behind.
            outcome.update(
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "log_tail": result.log_tail,
                }
            )
            if result.timed_out:
                raise DumpError("timed_out", log_tail=result.log_tail)
            if result.exit_code != 0:
                raise DumpError("exit_nonzero", log_tail=result.log_tail)
            output = await asyncio.to_thread(_verify_dump_output_path, staging, backend)
            outcome["reason"] = None
            captured = DumpResult(
                exit_code=result.exit_code,
                timed_out=False,
                log_tail=result.log_tail,
                output_path=str(output),
                started_at=result.started_at,
                finished_at=result.finished_at,
            )
            if on_captured is not None:
                try:
                    await on_captured(captured)
                except BaseException:
                    # Capture succeeded, hand-off did not: no artifact survives, so
                    # ``interrupted`` is the floor (D-P37-11); the route re-stamps
                    # its packer's own token.
                    outcome["reason"] = "interrupted"
                    raise
            return captured
        except DumpError as exc:
            outcome["reason"] = exc.reason
            if not outcome.get("log_tail"):
                outcome["log_tail"] = exc.log_tail
            raise
        finally:
            # (D-P37-11) A failed dump leaves NOTHING: the staging dir goes on
            # every non-success path. On success the route's packer owns it (it
            # removes it in its own ``finally``), so removing it here would
            # destroy the artifact before it is packed.
            if staging is not None and outcome.get("reason") is not None:
                await asyncio.to_thread(_rmtree_staging, staging)
            await self._stamp_last_dump(
                job,
                _last_dump_payload(
                    run_id=run_id,
                    kind="dump",
                    started=started,
                    command=argv,
                    outcome=outcome,
                ),
            )
            self._discard_run(job.id, run_id)

    async def restore_database(
        self,
        job: Job,
        *,
        run_id: str,
        staging: Path,
        timeout_s: int,
        slot_held: bool = False,
    ) -> DumpResult:
        """Restore a managed database from an already-extracted dump (D-P37-6).

        *staging* was populated by the route (``extract_dump_tar``, sha256 and engine
        already verified) and is consumed and removed on every path. Two shapes,
        chosen by the backend: ``restore_argv`` is a list (Postgres — a sibling runs
        ``pg_restore --single-transaction --exit-on-error`` against the running
        server) or ``None`` (Redis — no container; a complete AOF dir is prepared next
        to the live one, then the row is quiesced and the two are swapped by rename,
        the old one kept as ``appendonlydir.pre-restore-<stamp>``).

        *slot_held*: the caller reserved the dump slot and releases it.

        Raises :class:`DumpError` (``exit_nonzero`` | ``timed_out`` | ``interrupted``
        | ``quiesce_timeout``), :class:`RunPreconditionError` before any slot is
        claimed; ``ContainerRuntimeError`` propagates as-is.
        """
        cfg = parse_job_config(job, warn=True)
        backend = self._dump_backend(cfg)
        image = cfg.get("image")
        if not isinstance(image, str) or not image:
            raise DumpError("data_plane_unavailable")

        # A restore takes a DUMP slot (D-P37-9); ``slot_held`` means the caller
        # reserved it (``reserve_dump_slot``) and releases it.
        if not slot_held:
            self._register_run(job.id, run_id, mode=RunMode.dump)
        started = datetime.now(UTC)
        argv: list[str] = []
        outcome: dict = {"reason": "interrupted"}
        try:
            host_port = await self._dump_endpoint_port(job)
            in_path = f"{_DUMP_MOUNT}/{backend.dump_filename}"
            maybe_argv = backend.restore_argv(self._dump_bridge_host, host_port, in_path)
            if maybe_argv is None:
                installed = await self._restore_by_aof_install(
                    job, cfg, backend, run_id=run_id, staging=staging, timeout_s=timeout_s
                )
                outcome.update({"exit_code": None, "timed_out": False, "reason": None})
                return installed
            argv = maybe_argv
            password = await self._managed_password(job, backend)
            config = self._dump_container_config(
                cfg,
                image=image,
                argv=argv,
                env=backend.dump_env(password),
                staging=staging,
            )
            try:
                result = await self._execute_container_once(
                    job,
                    run_id=run_id,
                    config=config,
                    timeout_s=timeout_s,
                    log_tail=_DUMP_LOG_TAIL,
                    scrub_values={password},
                )
            except RunInterruptedError as exc:
                raise DumpError("interrupted", log_tail=exc.log_tail) from exc
            # ``update``, never a rebind: dropping the pre-seeded ``reason``
            # would make an unexpected exception below look like a success to
            # the ``finally`` and leave the staging dir behind.
            outcome.update(
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "log_tail": result.log_tail,
                }
            )
            if result.timed_out:
                raise DumpError("timed_out", log_tail=result.log_tail)
            if result.exit_code != 0:
                raise DumpError("exit_nonzero", log_tail=result.log_tail)
            outcome["reason"] = None
            return DumpResult(
                exit_code=result.exit_code,
                timed_out=False,
                log_tail=result.log_tail,
                output_path=None,
                started_at=result.started_at,
                finished_at=result.finished_at,
            )
        except DumpError as exc:
            outcome["reason"] = exc.reason
            if not outcome.get("log_tail"):
                outcome["log_tail"] = exc.log_tail
            raise
        finally:
            # The extracted payload is spent by now on every path — success,
            # refusal or crash — so it goes with the same "leaves nothing"
            # discipline as a failed dump's staging dir (D-P37-11).
            await asyncio.to_thread(_rmtree_staging, staging)
            await self._stamp_last_dump(
                job,
                _last_dump_payload(
                    run_id=run_id,
                    kind="restore",
                    started=started,
                    command=argv,
                    outcome=outcome,
                ),
            )
            if not slot_held:
                self._discard_run(job.id, run_id)

    async def _restore_by_aof_install(
        self,
        job: Job,
        cfg: dict,
        backend: DataBackend,
        *,
        run_id: str,
        staging: Path,
        timeout_s: int,
    ) -> DumpResult:
        """Install an RDB as the Redis multi-part-AOF base under a quiesce (D-P37-6).

        A ``dump.rdb`` dropped into ``/data`` of an ``--appendonly yes`` server is
        ignored (verified live, plan §0); placed as
        ``appendonlydir/appendonly.aof.1.base.rdb`` beside a manifest it loads. The
        replacement dir is written complete while the server is up; only then is the
        row quiesced and the dirs swapped by two renames.
        """
        assert self._data_dir is not None  # guaranteed by _dump_backend
        volume_dir = self._managed_volume_dir(job, cfg, backend)
        prepared = volume_dir / f"{_AOF_DIR_NAME}.restore-{run_id}"
        source = staging / backend.dump_filename
        started = datetime.now(UTC)
        try:
            await asyncio.to_thread(_prepare_aof_dir, prepared, source)
        except OSError as exc:
            # The vocabulary is locked (D-P37-11) and ``interrupted`` is its
            # member for "started and did not finish". Nothing has been swapped
            # at this point, so the live data is untouched.
            logger.warning("Preparing the Redis AOF base for %s failed", job.id, exc_info=True)
            await asyncio.to_thread(_rmtree_staging, prepared)
            raise DumpError("interrupted") from exc

        try:
            async with self.quiesce_row(job, timeout_s=timeout_s):
                await asyncio.to_thread(_swap_aof_dir, volume_dir, prepared)
        except DumpError:
            # A quiesce that never reached ``stopped`` never swapped anything;
            # the prepared dir is the only residue and it is ours to remove.
            await asyncio.to_thread(_rmtree_staging, prepared)
            raise
        except OSError as exc:
            logger.warning("Installing the Redis AOF base for %s failed", job.id, exc_info=True)
            # Live dir still in place ⇒ nothing moved or the swap unwound: the
            # prepared dir is residue, remove it. Live dir missing ⇒ the rollback
            # failed too: leave both for the operator.
            if (volume_dir / _AOF_DIR_NAME).exists():
                await asyncio.to_thread(_rmtree_staging, prepared)
            raise DumpError("interrupted") from exc
        finished = datetime.now(UTC)
        return DumpResult(
            exit_code=None,
            timed_out=False,
            log_tail=[],
            output_path=None,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
        )

    def _managed_volume_dir(self, job: Job, cfg: dict, backend: DataBackend) -> Path:
        """Return the host dir behind the row's ``data`` named volume, through
        :func:`~nerdit.core.volumes.service_volumes` (the one resolver every launch
        uses); the volume name comes from the backend's ``volume_spec``.
        """
        assert self._data_dir is not None
        volname, _, _ = backend.volume_spec.partition(":")
        volumes = service_volumes(self._data_dir, job.service_name or "", cfg)
        for host_path in volumes:
            if Path(host_path).name == volname:
                return Path(host_path)
        raise VolumeSpecError(f"database row declares no {volname!r} volume")

    @contextlib.asynccontextmanager
    async def quiesce_row(self, job: Job, *, timeout_s: int) -> AsyncIterator[None]:
        """Stop one row, yield while it is down, then bring it back (D-P37-6).

        Drives the ordinary desired-state machinery (``set_desired_state`` + wait for
        the reconcile loop), so every stop invariant holds. Both waits share ONE
        deadline, so the whole quiesce is bounded by *timeout_s*; a stop that eats
        the budget still ends ``running`` and reports ``quiesce_timeout`` honestly.
        A stop that never converges raises ``DumpError("quiesce_timeout")`` AFTER
        putting the desired state back to ``running``. The restart wait needs
        ``running`` AND ``config['db_ready']``; if it expires the body has already
        completed, so the raise cannot mask an error from inside the block. The
        caller holds a ``dump`` slot throughout.
        """
        name = job.service_name or job.id
        deadline = time.monotonic() + timeout_s
        await self._queries.set_desired_state(job.id, "stopped")
        if not await self._await_row_stopped(job.id, deadline):
            logger.warning("Quiesce of %s timed out waiting for it to stop", name)
            await self._queries.set_desired_state(job.id, "running")
            raise DumpError("quiesce_timeout")
        logger.info("Quiesced %s for a restore", name)
        ready = False
        try:
            yield
        finally:
            # ALWAYS, even when the body raised: a row left ``stopped`` because
            # an install failed is an outage the operator did not ask for.
            await self._queries.set_desired_state(job.id, "running")
            ready = await self._await_db_ready(job.id, deadline)
        if not ready:
            logger.warning("Quiesce of %s timed out waiting for it to come back", name)
            raise DumpError("quiesce_timeout")

    async def _await_row_stopped(self, job_id: str, deadline: float) -> bool:
        """Poll until the row reports ``stopped``; ``True`` on success.

        *deadline* is an absolute :func:`time.monotonic` instant, not a
        duration, because :meth:`quiesce_row` computes ONE for both of its
        phases — see its docstring for why the whole quiesce, and not each half
        of it, is what has to fit the caller's timeout.
        """
        while True:
            row = await self._queries.get_job(job_id)
            if row is None:
                # The row went away under us — nothing left to quiesce, and the
                # caller's next step (a rename under its volume dir) would be
                # against a deleted service. Report it as a timeout rather than
                # inventing a reason outside the locked vocabulary.
                return False
            if row.status is JobStatus.stopped:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_QUIESCE_POLL_INTERVAL_S)

    async def _await_db_ready(self, job_id: str, deadline: float) -> bool:
        """Poll until the row is ``running`` AND ``config['db_ready']``.

        BOTH conditions, and neither alone. ``db_ready`` alone is unsound here:
        the teardown does not clear the flag (only the relaunch does, see
        ``core/launch.py``), so a row that has just been quiesced still carries
        the PREVIOUS container's ``True`` and a flag-only wait would return
        instantly, before anything had restarted. ``running`` alone is unsound
        the other way: a container whose Redis is still replaying the AOF base
        this restore just installed is up but not yet a database, which is
        precisely the moment a caller must not be told the restore is done.

        *deadline* is an absolute :func:`time.monotonic` instant shared with
        :meth:`_await_row_stopped` (see :meth:`quiesce_row`).
        """
        while True:
            row = await self._queries.get_job(job_id)
            if (
                row is not None
                and row.status is JobStatus.running
                and parse_job_config(row, warn=False).get("db_ready") is True
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_QUIESCE_POLL_INTERVAL_S)

    @staticmethod
    def _parse_port_range(raw: str) -> tuple[int, int]:
        """Parse a validated `"lo-hi"` range into a `(lo, hi)` tuple."""
        lo_str, hi_str = raw.split("-")
        return int(lo_str.strip()), int(hi_str.strip())

    # --- Loop ---------------------------------------------------------------

    async def reconcile(self) -> None:
        """Run one reconciliation tick over every service/model row.

        Two-pass and stateless: fetch the rows needing convergence,
        snapshot the live managed-container set once,
        then drive each row through its state machine under its own
        `try/except` so one bad service never kills the tick.
        """
        services = await self._queries.get_reconcilable_services()
        if not services:
            return
        try:
            live = await self._runtime.list_managed_containers()
        except Exception:
            logger.warning("Could not list managed containers for service tick", exc_info=True)
            return
        live_ids = {cid for cid, _ in live}
        # Health-check budget for this tick (CRIT-5), shared across services.
        budget = [_MAX_HEALTH_CHECKS_PER_TICK]
        for job in services:
            try:
                await self._reconcile_one(job, live_ids, budget)
            except Exception:
                logger.exception("Error reconciling service %s", job.service_name or job.id)

    async def _reconcile_one(self, job: Job, live_ids: set[str], budget: list[int]) -> None:
        """Drive one service row toward its `desired_state`."""
        desired = job.desired_state or "running"
        live = job.container_id is not None and job.container_id in live_ids

        if desired == "stopped":
            # A stop landing mid-cutover must not leave the green
            # running behind the teardown's back: the verify task holds a
            # SECOND live container that no DB query names, so
            # `_teardown_to_stopped` (which only knows `job.container_id`)
            # would tear blue down and leave the green orphaned — still
            # probing, still able to reach its commit and promote a container
            # onto a row the operator just stopped. Cancel FIRST: pre-commit it
            # is the clean-unwind path (green destroyed, pointer back to blue's
            # own port, marker popped, generation NOT stamped failed), so the
            # row settles plainly `stopped`; post-commit it completes the
            # residual commit tail instead and leaves the promoted green alone
            # — the fresh re-read below is what tears that one down.
            if self.has_active_cutover(job.id):
                await self.cancel_cutover(job.id)
            # Tear down the FRESH row, not this tick's snapshot (PR #108
            # review): a verify that committed between the tick's fetch and
            # the check above promoted the row to the green and finished its
            # task, so `has_active_cutover` is honestly False — but this
            # `job` still names the destroyed blue. Tearing that down would
            # mark the row stopped while the freshly promoted green keeps
            # running forever (row-backed ids are sweep-protected regardless
            # of status). The re-read happens AFTER the cancel, so any commit
            # that raced the check is visible in `container_id` here.
            fresh = await self._queries.get_job(job.id)
            await self._teardown_to_stopped(fresh if fresh is not None else job)
            return

        # Build phase — build a pending target image BEFORE converging. This
        # covers both a fresh deploy (no container yet) and a redeploy (the old
        # container is still live and keeps serving): the build runs OFF this tick
        # (it can take minutes; the reconcile loop is shared with every other
        # workload) as a tracked background task, and we return without touching
        # any live container. Once the image exists the row falls through to the
        # launch/swap path below. Only checked when a build could be pending —
        # a healthy running service never pays the image_exists probe.
        if await self._builder.ensure_built(job, live):
            return  # build in flight — keep any live container serving meanwhile

        # Model image gate — the backend image (ollama/ollama) is pulled
        # OFF-tick while the row sits `building`, mirroring the deploy build
        # phase above: a dead model row never reaches _launch until its image
        # is present locally. A live model never pays the image_exists probe.
        if (
            job.kind is JobKind.model
            and self._models is not None
            and not live
            and not await self._models.ensure_image(job, parse_job_config(job, warn=True))
        ):
            return  # pull in flight — retry on a later tick

        # Database image gate — the backend image (postgres:16) is pulled
        # OFF-tick while the row sits `building`, mirroring the model gate above.
        if (
            job.kind is JobKind.database
            and self._data is not None
            and not live
            and not await self._data.ensure_image(job, parse_job_config(job, warn=True))
        ):
            return  # pull in flight — retry on a later tick

        # Health-gated cutover — the swap gate for an eligible
        # service. Placed AFTER the build gate and BEFORE the destroy-first
        # branch below, which it replaces for eligible rows: the new generation
        # is launched on a transient port and verified healthy before anything
        # touches the live container, and the route is repointed (and the
        # repoint VERIFIED) before blue is destroyed. `True` ⇒ a verify is in
        # flight (or just settled) and blue keeps serving; `False` ⇒ the row
        # is not eligible and the pre-P24 destroy-first path below runs
        # unchanged (pinned byte-identical by test_cutover.py).
        if await self._cutover.maybe_cutover(job, live):
            return

        # Explicit restart of a *still-running* service: the /restart route set
        # status='restarting' while the container is still alive. Replace it —
        # tear the live container down so the dead path relaunches a fresh one
        # (without this, a restart of a healthy service would be a no-op).
        if live and job.status == JobStatus.restarting and job.container_id:
            await self._destroy_container(job.container_id)
            await self._queries.release_gpus(job.id)
            live = False

        if live:
            await self._reconcile_live(job, budget)
        else:
            await self._reconcile_dead(job)

    async def _destroy_container(self, container_id: str) -> None:
        """Best-effort stop→kill→remove of a container + cancel its log task."""
        try:
            await self._runtime.stop(container_id)
        except ContainerRuntimeError:
            try:
                await self._runtime.kill(container_id)
            except ContainerRuntimeError:
                pass
        try:
            await self._runtime.remove(container_id, force=True)
        except ContainerRuntimeError:
            pass
        self._cancel_log_task(container_id)

    # --- (A) desired = stopped ---------------------------------------------

    async def _append_log_tolerant(
        self, job_id: str, message: str, stream: LogStream = LogStream.system
    ) -> None:
        """Append a log line, ignoring foreign-key failure if the job was deleted.

        Serialized rollback keeps that failed insert from poisoning the connection.
        The post-launch append intentionally handles this itself: a vanished job there
        requires tearing down the newly started orphan container.
        """
        try:
            await self._queries.append_log(job_id, message, stream)
        except sqlite3.IntegrityError:
            logger.debug("append_log skipped for vanished job %s", job_id)

    async def _teardown_to_stopped(self, job: Job) -> None:
        """Tear a service down to the terminal `stopped` state.

        The host-port endpoint **is** released here (PORTS-1): retaining it on
        every stop while the quota slot frees let a token loop create→stop with
        fresh names and drain the shared `9400-9499` pool (a cross-tenant
        availability DoS). A later `restart` re-acquires from the pool — usually
        the same lowest-free port, so the URL is stable in practice — and the
        crash→`restarting` path (which never reaches a terminal state) keeps its
        port across automatic restarts unchanged.
        """
        if job.status == JobStatus.stopped:
            return  # already terminal — no-op
        if job.container_id:
            await self._destroy_container(job.container_id)
        await self._queries.release_gpus(job.id)
        await self._release_endpoint(job)
        self._health_failures.pop(job.id, None)
        self._binding_wait_msgs.pop(job.id, None)
        self._shared_resolved_sigs.pop(job.id, None)
        await self._queries.update_job_status(
            job.id, JobStatus.stopped, finished_at=datetime.now(UTC)
        )
        await self._append_log_tolerant(job.id, "Service stopped by user", LogStream.system)
        self._emit_status_change(job.id, JobStatus.stopped)
        # Durable feed, BESIDE the bus-only legacy event — never
        # replacing it (D-P24-3 rule 3).
        await record_job_event(self._events, "service.stopped", job)
        logger.info("Service %s stopped", job.service_name or job.id)

    async def _release_endpoint(self, job: Job) -> None:
        """Release the service's host-port reservation back to the pool (PORTS-1).

        Called on every *terminal* transition (stopped / failed / completed) so a
        non-running service never holds a port. Best-effort and idempotent — the
        DELETE route also releases. The proxy route is deregistered first so
        the URL stops resolving the moment the service stops; the reconcile loop
        would prune it on the next tick regardless.
        """
        if job.service_name:
            if self._proxy:
                await self._proxy.deregister(job.service_name)
            await self._queries.release_service_endpoint(job.service_name)

    # --- (B) desired = running, container dead -----------------------------

    async def _reconcile_dead(self, job: Job) -> None:
        """Converge a service whose container is not running.

        Branches on the current status: a service we believed up
        (`running`/`degraded`) just crashed; a `restarting` one is in
        backoff; a `failed` one stays terminal unless its cap was externally
        cleared (the restart route); everything else is a fresh/clean launch.
        """
        now = datetime.now(UTC)
        status = job.status

        if status == JobStatus.restarting:
            delay = self._backoff_seconds(job.restart_count)
            if job.last_exit_at and (now - job.last_exit_at).total_seconds() < delay:
                return  # still backing off — retry on a later tick
            await self._launch(job)
            return

        if status == JobStatus.failed:
            # Terminal: only relaunch when the restart route cleared the cap.
            if job.restart_count >= self._max_restarts:
                return
            await self._launch(job)
            return

        if status in (JobStatus.running, JobStatus.degraded):
            await self._handle_crash(job, now)
            return

        # building / scheduled / stopped→running / completed / cancelled → launch.
        await self._launch(job)

    async def _crash_scrub_values(self, job: Job) -> set[str] | None:
        """Best-effort D-P20-1 scrub set for a crashed container's tail (PR #102 review).

        Re-runs the same launch-env resolution the dead container was started
        with (the `_launch` prelude, verbatim — incl. the model/database
        binding suppression), so the set matches what the container could
        actually have printed. The service WAS running, so resolution normally
        succeeds; `None` means it could not be computed (a binding target or
        secret vanished mid-crash) and the caller must keep the previous
        capture rather than persist unscrubbed lines.
        """
        try:
            cfg = parse_job_config(job, warn=False)
            container_port = int(cfg.get("port") or 8000)
            is_model = job.kind is JobKind.model and self._models is not None
            is_data = job.kind is JobKind.database and self._data is not None
            ai_specs = cfg.get("ai") if not (is_model or is_data) else None
            db_specs = cfg.get("db") if not (is_model or is_data) else None
            resolved = await self._resolve_launch_env(job, cfg, container_port, ai_specs, db_specs)
        except Exception:  # noqa: BLE001 - forensics must never break the crash path
            return None
        return _sensitive_env_values(resolved)

    async def _capture_crash_tail(self, job: Job) -> list[str]:
        """Capture the latest crash between inspection and container removal.

        Bound reads to 100 lines/64 KiB, scrub, then replace the `crash` log stream.
        A successful empty read clears prior output. Read/scrub-resolution failures
        preserve the prior capture; never persist output without scrubbing.
        Tolerate job-deletion FK races; other DB errors let reconcile retry next tick.

        Returns:
            Raw lines for boolean classification only, never direct persistence or output.
        """
        if not job.container_id:
            return []
        raw: list[str] = []
        try:
            async for line in self._runtime.logs(
                job.container_id, tail=_CRASH_TAIL_MAX_LINES, max_bytes=_CRASH_TAIL_MAX_BYTES
            ):
                raw.append(line)
        except Exception:  # noqa: BLE001 - forensics must never break the crash path
            logger.warning(
                "Could not read the crash log tail of container %s", job.container_id, exc_info=True
            )
            return []
        scrubbed: list[str] = []
        if raw:
            scrub = await self._crash_scrub_values(job)
            if scrub is None:
                logger.warning(
                    "Crash tail of %s not persisted: launch env unresolvable, cannot scrub",
                    job.id,
                )
                return raw
            scrubbed = _scrub_and_truncate(raw, scrub)
        try:
            await self._queries.replace_crash_tail(job.id, scrubbed)
        except sqlite3.IntegrityError:
            logger.debug("crash-tail capture skipped for vanished job %s", job.id)
        return raw

    async def _handle_crash(self, job: Job, now: datetime) -> None:
        """Handle a service container that died while we believed it up.

        Reads the exit code, captures the bounded crash tail, releases
        GPUs, removes the dead container, then applies the restart policy and the
        rate-cap / backoff accounting.
        """
        exit_code: int | None = None
        oom_killed = False
        if job.container_id:
            try:
                exit_code = await self._runtime.wait(job.container_id)
            except ContainerRuntimeError:
                exit_code = None
            # Inspect the dead container BEFORE removing it (load-bearing order):
            # once removed, the OOM flag / exit code are gone. The result is
            # coerced defensively so a garbage/un-specced mock return degrades to
            # no-forensics rather than crashing the reconcile tick.
            state = None
            try:
                state = await self._runtime.inspect_state(job.container_id)
            except Exception:  # noqa: BLE001 - forensics must never break the crash path
                state = None
            oom_killed = _coerce_oom(state)
            if exit_code is None:
                exit_code = _coerce_exit_code(state)
            self._cancel_log_task(job.container_id)
        # Capture the crash tail while the container still exists — the
        # live-follow task is already cancelled above so the two cannot interleave.
        # `remove()` below is what makes this window the only one.
        crash_tail = await self._capture_crash_tail(job)
        gpu_oom = _tail_indicates_gpu_oom(crash_tail)
        # The same bounded tail also answers "did the image's entrypoint
        # need root capabilities this sandbox drops?" — a diagnosis nothing else
        # in the crash path can make once the container is removed.
        priv_denied = _tail_indicates_priv_denied(crash_tail)
        # Persist forensics on EVERY crash (terminal AND restart) so /diagnose
        # reads them regardless of the restart-policy branch below.
        await self._persist_forensics(
            job.id, exit_code, oom_killed, now, gpu_oom=gpu_oom, priv_denied=priv_denied
        )
        await self._queries.release_gpus(job.id)
        if job.container_id:
            try:
                await self._runtime.remove(job.container_id, force=True)
            except ContainerRuntimeError:
                pass

        policy = job.restart_policy or "always"
        clean_exit = exit_code == 0
        if policy == "no" or (policy == "on-failure" and clean_exit):
            final = JobStatus.completed if clean_exit else JobStatus.failed
            error_class = (
                None if clean_exit else _classify_service_exit(exit_code, oom_killed, gpu_oom)
            )
            error_message = None if clean_exit else f"Service exited (code={exit_code})"
            await self._queries.update_job_status(
                job.id,
                final,
                finished_at=now,
                exit_code=exit_code if exit_code is not None else -1,
                error_class=error_class,
                error_message=error_message,
            )
            # Settle desired_state to the terminal status so the row leaves
            # get_reconcilable_services and is NOT relaunched every tick — honoring
            # restart_policy='no' / a clean 'on-failure' exit. The /restart route
            # flips desired_state back to 'running' to revive it explicitly.
            await self._queries.set_desired_state(job.id, final.value)
            await self._append_log_tolerant(
                job.id,
                f"Service exited (code={exit_code}); restart_policy={policy} → {final.value}",
                LogStream.system,
            )
            await self._release_endpoint(job)  # terminal → free the host port (PORTS-1)
            self._emit_status_change(job.id, final)
            if final == JobStatus.failed:
                await self._audit("service.failed", job)
                # The OTHER terminal-failure path: a restart_policy
                # of "no" settles failed without ever reaching _count_restart,
                # so the crash-loop emitter there never fires. Same machine-
                # shaped payload shape as that site — an ErrorClass member and
                # an int, never the error_message built above.
                await record_job_event(
                    self._events,
                    "service.failed",
                    job,
                    reason="exited",
                    data={
                        "exit_code": exit_code,
                        "error_class": error_class.value if error_class is not None else None,
                    },
                )
            logger.info("Service %s exited → %s", job.service_name or job.id, final.value)
            return

        await self._count_restart(job, now, exit_code, oom_killed, gpu_oom)

    async def _count_restart(
        self,
        job: Job,
        now: datetime,
        exit_code: int | None,
        oom_killed: bool = False,
        gpu_oom: bool = False,
    ) -> None:
        """Apply the restart rate-cap, persist the counters, set the next state.

        Resets the window (`restart_count = 1`) when it has elapsed, otherwise
        increments within it. Exceeding `service_max_restarts` within the
        window is terminal `failed`; otherwise the service goes `restarting`
        and `last_exit_at` anchors the exponential backoff.
        """
        window = self._restart_window_seconds
        window_start = job.restart_window_start
        if window_start is None or (now - window_start).total_seconds() > window:
            new_count = 1
            window_start = now
        else:
            new_count = job.restart_count + 1
        await self._queries.bump_restart_count(job.id, new_count, window_start)
        await self._queries.record_service_exit(job.id, now)

        if new_count > self._max_restarts:
            await self._queries.update_job_status(
                job.id,
                JobStatus.failed,
                finished_at=now,
                exit_code=exit_code if exit_code is not None else -1,
                error_class=_classify_service_exit(exit_code, oom_killed, gpu_oom),
                error_message=(
                    f"Restart budget exhausted ({new_count} restarts in {window}s); "
                    f"last exit_code={exit_code}"
                ),
            )
            await self._queries.set_desired_state(
                job.id, JobStatus.failed.value
            )  # settle (see _handle_crash)
            # (F5-CRASHLOOP) Stamp crash_loop only onto the generation that
            # actually crashed — the tick-start `job` snapshot the controller
            # observed exhausting its budget, captured BEFORE a redeploy could
            # bump the version. A redeploy may seed a newer queued generation in
            # this same tick; `expect_version` on the snapshot's version keeps
            # the failed stamp off it (matching the build-fail idiom above). When
            # the snapshot carries no last_deploy (pre-P13 / POST /services row),
            # skip the phase stamp entirely rather than risk stamping a
            # generation seeded mid-tick. The row-level `failed` transition
            # above stays unguarded (the plan's "any → failed" row semantics).
            snap_ld = parse_job_config(job, warn=True).get("last_deploy")
            crashed_version = snap_ld.get("version") if isinstance(snap_ld, dict) else None
            if crashed_version is not None:
                await stamp_last_deploy(
                    self._queries,
                    job.id,
                    expect_version=crashed_version,
                    phase="failed",
                    reason="crash_loop",
                    error_class=_classify_service_exit(exit_code, oom_killed, gpu_oom).value,
                    error_message=(
                        f"Restart budget exhausted ({new_count} restarts in {window}s); "
                        f"last exit_code={exit_code}"
                    ),
                )
            await self._append_log_tolerant(
                job.id,
                f"Service exhausted {self._max_restarts} restarts within {window}s → failed",
                LogStream.system,
            )
            await self._release_endpoint(job)  # terminal → free the host port (PORTS-1)
            self._emit_status_change(job.id, JobStatus.failed)
            await self._audit("service.failed", job)
            # THE killer event — the 03:00 crash-loop settle that
            # released the endpoint and 404'd the public URL. Payload is
            # machine-shaped only: an ErrorClass member and two integers, never
            # the error_message built above (it interpolates runtime values).
            await record_job_event(
                self._events,
                "service.failed",
                job,
                reason="crash_loop",
                data={
                    "restart_count": new_count,
                    "exit_code": exit_code,
                    "error_class": _classify_service_exit(exit_code, oom_killed, gpu_oom).value,
                },
            )
            logger.warning(
                "Service %s exhausted restarts (%d) → failed",
                job.service_name or job.id,
                new_count,
            )
            return

        await self._queries.update_job_status(job.id, JobStatus.restarting)
        await self._append_log_tolerant(
            job.id,
            f"Service restarting (attempt {new_count}/{self._max_restarts})",
            LogStream.system,
        )
        self._emit_status_change(job.id, JobStatus.restarting)
        await self._audit("service.restart", job)
        # Coalesced on (type, service_name, reason): a fast flapping
        # loop inside the 60 s window collapses to one row, but a *changed*
        # error class passes through immediately.
        await record_job_event(
            self._events,
            "service.restarting",
            job,
            reason=_classify_service_exit(exit_code, oom_killed, gpu_oom).value,
            data={"restart_count": new_count, "exit_code": exit_code},
        )
        logger.info(
            "Service %s restarting (%d/%d)",
            job.service_name or job.id,
            new_count,
            self._max_restarts,
        )

    @staticmethod
    def _backoff_seconds(restart_count: int) -> float:
        """Exponential backoff `min(60, 2**n)` seconds before the next launch."""
        return min(_MAX_BACKOFF_S, float(2 ** max(restart_count, 1)))

    async def _stamp_config(
        self,
        job_id: str,
        updates: dict[str, object] | None = None,
        *,
        pop: list[str] | None = None,
    ) -> None:
        """Fresh read-modify-write of a service row's `config` blob.

        Re-reads the row via `get_job` *now* (never a tick-start snapshot) so a
        redeploy landing mid-tick is never clobbered, applies `pop` then
        `updates`, and persists via `update_job_config` — a blob-only write
        that touches neither `status` nor the restart bookkeeping. Shared with
        the crash-forensics writer here; WP2 routes its `last_deploy` phase
        stamps through the same helper.
        """
        row = await self._queries.get_job(job_id)
        if row is None:
            return
        cfg = parse_job_config(row, warn=True)
        for key in pop or ():
            cfg.pop(key, None)
        if updates:
            cfg.update(updates)
        await self._queries.update_job_config(job_id, json.dumps(cfg))

    async def _persist_forensics(
        self,
        job_id: str,
        exit_code: int | None,
        oom_killed: bool,
        now: datetime,
        gpu_oom: bool = False,
        priv_denied: bool = False,
    ) -> None:
        """Stamp crash forensics (exit code, `oom_killed`, `gpu_oom`, `priv_denied`, crash time).

        Called on **every** crash (terminal and restart paths). `exit_code` is
        only written when it is a real `int` so a lost wait/inspect leaves the
        prior value untouched rather than nulling it. `gpu_oom` and
        `priv_denied` are written unconditionally, like `oom_killed`:
        every crash overwrites them, so a later unrelated crash clears a stale
        flag.
        """
        updates: dict[str, object] = {
            "oom_killed": bool(oom_killed),
            "gpu_oom": bool(gpu_oom),
            "priv_denied": bool(priv_denied),
            "last_crash_at": now.isoformat(),
        }
        if isinstance(exit_code, int):
            updates["last_exit_code"] = exit_code
        await self._stamp_config(job_id, updates)

    # --- Launch sequence ----------------------------------------------------

    async def _resolve_launch_env(
        self,
        job: Job,
        cfg: dict,
        container_port: int,
        ai_specs: object,
        db_specs: object,
    ) -> ResolvedLaunchEnv:
        """Acquire the full launch env or raise `LaunchEnvNotReady`.

        Pure of audit side effects (P14 WP-0 C2.1): the caller owns the wait-log
        and shared-resolved audit. Loads per-service secrets ONCE, lazily
        loads the shared scope only when a `${secrets.shared.KEY}` ref is
        actually unresolved, resolves `[ai.*]` then `[db.*]`
        bindings, and merges the container env (config env ⊕ secrets ⊕ injected
        ai ⊕ injected db ⊕ `PORT`). A decrypt failure or a not-ready binding
        raises rather than returning — the caller re-establishes the exact
        retry-next-tick semantics. Also reports `injected_keys` — the keys the binding resolvers
        actually produced — for `run_once`'s protected-key overlay. Purely
        additive: `_launch`'s behaviour is unchanged.
        """
        # Per-service secrets, loaded ONCE — shared between binding
        # resolution and the container env. A decrypt failure is
        # NON-terminal: raise, the admin restores the key/file, launch converges.
        secret_env: dict[str, str] = {}
        if self._secrets is not None and job.service_name:
            try:
                secret_env = self._secrets.load(job.service_name)
            except SecretDecryptError as exc:
                raise LaunchEnvNotReady("secrets", f"Secrets unavailable: {exc}") from exc

        ai_dict = ai_specs if isinstance(ai_specs, dict) else {}
        db_dict = db_specs if isinstance(db_specs, dict) else {}
        resolved_ai: dict[str, ResolvedBinding] = {}
        resolved_db: dict = {}
        shared_keys: list[str] = []
        if ai_dict or db_dict:
            # LAZY shared load: _shared is read only when a spec carries a
            # ${secrets.shared.KEY} ref, so one corrupt shared file blocks only
            # the services that reference it — never the whole node. A ref whose
            # KEY is already a per-service override resolves from secret_env first
            # (the documented escape hatch), so it does NOT force a shared
            # decrypt: a fully-overridden service must launch even when _shared is
            # corrupt/unreadable. Shared refs come from BOTH kinds — the
            # [ai.*] api_key and the [db.*] password.
            shared_keys = sorted(
                set(shared_secret_keys_for(ai_dict, ("api_key",)))
                | set(shared_secret_keys_for(db_dict, ("password",)))
            )
            unresolved_shared = [k for k in shared_keys if k not in secret_env]
            shared_env: dict[str, str] = {}
            if unresolved_shared and self._secrets is not None:
                try:
                    shared_env = self._secrets.load(SHARED_SCOPE)
                except SecretDecryptError as exc:
                    raise LaunchEnvNotReady(
                        "shared_secrets", f"Shared secrets unavailable: {exc}"
                    ) from exc
            # AI resolves BEFORE DB so a mixed spec's first BindingNotReady
            # message is deterministic (the log dedupe keys on the message).
            try:
                if ai_dict:
                    resolved_ai = await resolve_bindings(
                        ai_dict, self._queries, secret_env, self._bridge_host, shared_env=shared_env
                    )
                if db_dict:
                    if self._data is None or self._secrets is None:
                        raise LaunchEnvNotReady(
                            "binding",
                            "[db.*] bindings require the database subsystem, "
                            "which is not enabled on this daemon",
                        )
                    resolved_db = await resolve_db_bindings(
                        db_dict,
                        self._queries,
                        secret_env,
                        self._bridge_host,
                        self._data.backend_for,
                        self._secrets.load,
                        shared_env=shared_env,
                    )
            except BindingNotReady as exc:
                raise LaunchEnvNotReady("binding", str(exc)) from exc

        # Env: config env first, then per-service secrets override it so a
        # rotated/removed secret takes effect on the next launch without the app
        # re-declaring anything. Resolved [ai.*] then [db.*] bindings
        # inject AFTER the secrets merge so platform-computed values override
        # same-named user secrets (Decision #4: users must not hand-override
        # OPENAI_BASE_URL / DATABASE_URL).
        env = dict(cfg.get("env") or {})
        env.update(secret_env)
        # Materialize the injected pairs once so their KEYS can be handed
        # to the caller as `injected_keys` — the D-P14-5 protected set derives
        # from what was actually injected, never from a spec-shape projection.
        injected: dict[str, str] = {}
        if resolved_ai:
            injected.update(inject_env(resolved_ai))
        if resolved_db:
            injected.update(inject_db_env(resolved_db))
        env.update(injected)
        # Tell the app which port it was wired to (Heroku/Cloud-Run convention).
        # An explicit config/secret PORT still wins.
        env.setdefault("PORT", str(container_port))
        # (edge-case #5b) Path-mode Caddy terminates TLS and forwards plain HTTP
        # with X-Forwarded-Proto; stock uvicorn trusts only loopback peers while
        # Caddy's hop arrives from the docker-bridge gateway, so apps silently
        # emit http:// links. uvicorn reads this env var when the CLI flag is
        # absent (proxy-headers is default-on), so both the buildpack fallback
        # and a [deploy].start uvicorn trust the proxy with zero app changes.
        # Only for services (models and databases are unrouted); an explicit
        # user env value always wins. Positive form so a fourth kind can never
        # leak the proxy-trust var in again.
        if (
            job.kind is JobKind.service
            and self._proxy is not None
            and self._proxy.enabled
            and self._proxy_mode == "path"
        ):
            env.setdefault("FORWARDED_ALLOW_IPS", "*")
        return ResolvedLaunchEnv(
            env=env,
            secret_env=secret_env,
            shared_keys=shared_keys,
            injected_keys=set(injected),
        )

    async def _launch(self, job: Job) -> None:
        """Resolve env and mounts, allocate resources, and launch a service container.

        Share optional GPU allocations, reserve a stable port, and force bridge
        networking so published ports work. Validate sandbox/named-volume boundaries
        before launch; transient placement conflicts leave the row for a later retry.
        Shared launch helpers assemble configuration and settle startup.
        """
        cfg = parse_job_config(job, warn=True)
        now = datetime.now(UTC)
        is_model = job.kind is JobKind.model and self._models is not None
        is_data = job.kind is JobKind.database and self._data is not None

        # Hoisted above the env resolve (pure, side-effect-free) so
        # the resolver can seed the PORT default. An explicit config/secret PORT
        # still wins (setdefault).
        container_port = int(cfg.get("port") or 8000)
        # [ai.*] AND [db.*] bindings are suppressed for model AND database rows:
        # both are bound-TO resources (a binding consumer's target), never
        # binding consumers themselves (P15, mirror of the model suppression).
        ai_specs = cfg.get("ai") if not (is_model or is_data) else None
        db_specs = cfg.get("db") if not (is_model or is_data) else None

        # --- (P5 / S9) resolve the launch env (per-service secrets + shared +
        # [ai.*]/[db.*] bindings) BEFORE any resource acquisition. A not-ready
        # binding or an undecryptable secret raises LaunchEnvNotReady — NOTHING
        # acquired (no GPU, no host port, no endpoint row) — and the next tick
        # retries, so "serve the model/database afterwards" self-heals and a
        # reboot's resource-before-app ordering converges on its own. Never a
        # terminal failure.
        try:
            resolved_env = await self._resolve_launch_env(
                job, cfg, container_port, ai_specs, db_specs
            )
        except LaunchEnvNotReady as exc:
            if exc.kind == "binding":
                await self._log_binding_wait(job, exc.message)
            else:
                await self._log_secret_wait(job, exc.message)
            return  # nothing acquired — retry next tick
        self._binding_wait_msgs.pop(job.id, None)
        # Shared-scope usage is audited (key NAMES only) after a successful
        # resolve — even when every ref was overridden by a per-service key,
        # since that row is the admin's only bypass signal. Kept caller-side (the
        # resolver is audit-side-effect-free) so run_once can audit differently.
        if resolved_env.shared_keys:
            await self._audit_shared_resolved(
                job, resolved_env.shared_keys, resolved_env.secret_env
            )

        # The minted-credential precondition (see require_minted_credential):
        # a database row's launch env is the ALLOWLISTED minted credential, so it
        # must be present in resolved_env.secret_env before any resource
        # acquisition — absent => nothing to release, retry next tick.
        data_backend = None
        if is_data:
            data_backend = await require_minted_credential(self, job, cfg, resolved_env.secret_env)
            if data_backend is None:
                return  # nothing acquired — retry next tick

        # --- (optional) GPU placement ---
        vendor = GpuVendor.nvidia
        runtime_gpu_ids: list[str] = []
        gpu_memory_mb: int | None = None  # smallest allocated card's VRAM
        if job.gpu_count > 0:
            requested = self._requested_vendor(cfg)
            candidates = await self._queries.get_service_placement_gpus(requested)
            placement = plan_gpu_placement(candidates, job.gpu_count, requested)
            if placement is None:
                logger.info(
                    "Service %s waiting for %d GPU(s)",
                    job.service_name or job.id,
                    job.gpu_count,
                )
                return  # retry next tick
            try:
                await self._queries.allocate_gpus(job.id, placement.gpu_ids, exclusive=False)
            except RuntimeError:
                logger.warning(
                    "GPU allocation conflict for service %s, will retry",
                    job.service_name or job.id,
                )
                return
            vendor = placement.vendor
            runtime_gpu_ids = placement.runtime_gpu_ids
            gpu_memory_mb = placement.min_memory_mb

        # --- stable host port (CRIT-4 bindability fallback; deferral / exhaustion
        # release any GPU acquired above and return None — see resolve_host_port) ---
        assert job.service_name is not None  # service rows always carry a name
        endpoint = await resolve_host_port(self, job, container_port, now)
        if endpoint is None:
            return  # deferred or retried next tick — resources already released

        # --- command / workspace ---
        command, volumes, workdir = self._build_command(job, cfg)

        # Tier-B mount allowlist for non-admin tokens (SANDBOX-1) so a
        # scoped/agent token cannot get an out-of-allowlist host path
        # bind-mounted via `script_path`. Tier-A
        # (docker.sock/etc/root/...) is enforced unconditionally by the runtime.
        # A violation is the submitter's fault and deterministic — fail the
        # service terminally rather than retry-loop it every tick.
        if volumes:
            role = await resolve_role(self._queries, job)
            if role != TokenRole.admin:
                try:
                    enforce_mount_allowlist(volumes, self._container_settings.allowed_mount_roots)
                except SandboxViolationError as exc:
                    await self._settle_launch_user_error(job, "sandbox_denied", str(exc))
                    return

        # Named volumes: re-validate the blob at launch (trust NO
        # producer — TOML, app-config PUT, a hand-written config) and materialize
        # the leaf dirs under <data_dir>/services/<name>. A grammar violation or a
        # forged host path is deterministic → settle terminal (never retry-loop).
        # A collision with a Tier-B user mount (normalized prefix, not equality)
        # is likewise a user error. These daemon-computed paths BYPASS the Tier-B
        # allowlist (the ModelBackend precedent — they are always under data_dir,
        # never user-supplied), so they are merged into the config AFTER the
        # allowlist enforcement above.
        named_volumes: dict[str, str] = {}
        if self._data_dir is not None and job.service_name and cfg.get("volumes"):
            try:
                named_volumes = service_volumes(self._data_dir, job.service_name, cfg)
            except VolumeSpecError as exc:
                await self._settle_launch_user_error(job, "volume_invalid", str(exc))
                return
            conflict = self._volume_path_conflict(named_volumes.values(), (volumes or {}).values())
            if conflict is not None:
                await self._settle_launch_user_error(job, "volume_conflict", conflict)
                return
            if named_volumes:
                self._ensure_volume_dirs(job.service_name, named_volumes)

        # The container env was fully assembled by _resolve_launch_env
        # above (config env ⊕ per-service secrets ⊕ injected [ai.*] ⊕ PORT
        # default). Secret values never touch logs or the audit row.
        env = resolved_env.env

        if is_model:
            assert self._models is not None  # is_model implies a controller
            # The backend supplies the launch shape (image, OLLAMA_HOST env,
            # the system-owned /root/.ollama weights volume, container port) and
            # the ModelController adds the bridge-gateway extra port bind. The
            # backend-composed volume deliberately BYPASSES the Tier-B user-mount
            # allowlist — it is daemon-owned (under data_dir), never a user
            # supplied path; the user-mount path above stays fully enforced.
            # Platform policy (limits, caps, forced bridge) is applied on top so
            # a model container is hardened exactly like an app service.
            # The allocated VRAM drives the backend's conservative
            # bounds injection; the per-serve overrides persisted on the row win
            # over both it and [models].vllm_extra_args. `cfg` is a tolerant JSON
            # parse, so both are type-checked here rather than trusted.
            raw_len = cfg.get("max_model_len")
            raw_util = cfg.get("gpu_memory_utilization")
            config = self._models.build_container_config(
                str(cfg.get("model") or ""),
                runtime_gpu_ids,
                vendor,
                endpoint.host_port,
                backend_name=cfg.get("backend"),
                gpu_memory_mb=gpu_memory_mb,
                max_model_len=(
                    raw_len if isinstance(raw_len, int) and not isinstance(raw_len, bool) else None
                ),
                gpu_memory_utilization=(
                    float(raw_util)
                    if isinstance(raw_util, (int, float)) and not isinstance(raw_util, bool)
                    else None
                ),
            )
            # Inject shared-scope launch secrets the backend declares
            # (e.g. HF_TOKEN for gated vLLM repos) — only keys actually present
            # in the shared store, never auto-injected beyond the declared set.
            await self._inject_model_launch_secrets(job, cfg, config)
            apply_platform_overlay(
                config, cfg, self._container_settings, container_port, endpoint.host_port
            )
        elif is_data:
            assert self._data is not None
            assert data_backend is not None  # is_data set it above
            # Allowlisted launch env: EXACTLY the minted credential from
            # the row's own secret scope. The backend adds its image-native
            # statics (POSTGRES_USER/DB/PGDATA) and sets user to the daemon's
            # uid:gid; a stray user-set secret in the scope (e.g.
            # POSTGRES_HOST_AUTH_METHOD) can therefore never reach the container
            # (§1.3). PGDATA rides the GENERIC named-volume block below (the
            # row's config['volumes'] was stamped at create) — no branch-local
            # volume set. The DataController adds the bridge-gateway extra port
            # bind so bound app containers can reach the port.
            minted_env = {
                data_backend.minted_secret_key: resolved_env.secret_env[
                    data_backend.minted_secret_key
                ]
            }
            config = self._data.build_container_config(
                cfg.get("backend"), job.service_name or "", endpoint.host_port, minted_env
            )
            # Apply the platform sandbox overlay on top so a database container
            # is hardened exactly like an app service (cap_drop=ALL +
            # no_new_privileges kept INTACT — the non-root entrypoint path needs
            # no capability carve-out, D-P15-6).
            apply_platform_overlay(
                config, cfg, self._container_settings, container_port, endpoint.host_port
            )
        else:
            config = build_app_container_config(
                cfg,
                self._container_settings,
                AppContainerSpec(
                    command=command,
                    volumes=volumes,
                    env=env,
                    workdir=workdir,
                    gpu_ids=runtime_gpu_ids,
                    vendor=vendor,
                    container_port=container_port,
                    host_port=endpoint.host_port,
                ),
            )

        finalize_container_config(config, named_volumes, self._retention_settings)

        await stamp_launching(self, job, config, named_volumes)
        await self._queries.update_job_status(job.id, JobStatus.scheduled)
        self._emit_status_change(job.id, JobStatus.scheduled)
        try:
            container_id = await self._runtime.run(config)
        except ContainerRuntimeError as exc:
            logger.error("Failed to launch service %s: %s", job.service_name, exc)
            await self._queries.release_gpus(job.id)
            await self._append_log_tolerant(
                job.id, f"Service container launch failed: {exc}", LogStream.system
            )
            # A failed start counts against the restart budget (always retry the
            # start itself; the policy gate only governs *exit* handling).
            # NOTE (WP3/WP4): a launch failure rides the restart loop and lands
            # as reason="crash_loop" once the budget is exhausted — there is NO
            # reason="launch_failed" write site (that locked enum value is
            # deliberately unused), so /diagnose must never branch on it.
            await self._count_restart(job, datetime.now(UTC), exit_code=-1)
            return

        await settle_started(self, job, container_id, endpoint, now=now)

    async def _log_binding_wait(self, job: Job, message: str) -> None:
        """Log a BindingNotReady wait once per DISTINCT message.

        The launch path retries every tick (~2 s); without dedupe the same
        actionable line would flood `job_logs`. A *changed* message (e.g.
        "not served" → "still pulling") is logged again, so the log reads as a
        progress trail.
        """
        if self._binding_wait_msgs.get(job.id) == message:
            return
        self._binding_wait_msgs[job.id] = message
        await self._append_log_tolerant(
            job.id, f"Waiting on AI binding: {message}", LogStream.system
        )
        logger.info("Service %s waiting on AI binding: %s", job.service_name or job.id, message)

    async def _log_secret_wait(self, job: Job, message: str) -> None:
        """Log a `SecretDecryptError` launch deferral once per DISTINCT message.

        Mirrors `_log_binding_wait` (and shares its dedupe map): a decrypt
        failure is deterministic until the admin restores the matching key/file,
        so the launch is deferred and retried every tick — never a terminal
        failure — without flooding `job_logs`.
        """
        if self._binding_wait_msgs.get(job.id) == message:
            return
        self._binding_wait_msgs[job.id] = message
        await self._append_log_tolerant(job.id, message, LogStream.system)
        logger.warning("Service %s launch deferred: %s", job.service_name or job.id, message)

    async def _audit_shared_resolved(
        self, job: Job, keys: list[str], secret_env: dict[str, str]
    ) -> None:
        """Audit which shared keys a launch resolved, as `secret.shared_resolved`.

        `overridden` is the subset shadowed by a same-named per-service secret
        (the precedence escape hatch) — the row fires even when EVERY ref was
        overridden, since it is the admin's only bypass signal. Deduped like
        `_binding_wait_msgs`: a crash-looping service re-audits only when the
        `(keys, overridden)` tuple actually changes. Key *names* only.
        """
        overridden = [key for key in keys if key in secret_env]
        signature = (tuple(keys), tuple(overridden))
        if self._shared_resolved_sigs.get(job.id) == signature:
            return
        self._shared_resolved_sigs[job.id] = signature
        try:
            await self._queries.insert_audit_log(
                action="secret.shared_resolved",
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="secret",
                target_id="shared",
                params_redacted=json.dumps(
                    {"service_name": job.service_name, "keys": keys, "overridden": overridden}
                ),
            )
        except Exception:
            logger.warning(
                "Failed to audit secret.shared_resolved for service %s", job.id, exc_info=True
            )

    async def _inject_model_launch_secrets(
        self, job: Job, cfg: dict, config: ContainerConfig
    ) -> None:
        """Inject the shared-scope secrets a model backend declares.

        A backend may declare launch env secret keys (vLLM: `HF_TOKEN` for
        gated Hugging Face repos). Only keys *actually present* in the shared
        store are injected — never auto-injected beyond the declared set, never
        per-service (model rows carry no `[ai.*]` refs). An absent or
        undecryptable shared store is non-terminal: skip injection (public
        models still serve) and let a gated repo fail loudly inside the
        container. The audit records key **names** only (values never logged).
        """
        if self._models is None or self._secrets is None:
            return
        backend = self._models.backend_for(cfg)
        wanted = tuple(getattr(backend, "launch_env_secret_keys", ()))
        if not wanted:
            return
        try:
            shared_env = self._secrets.load(SHARED_SCOPE)
        except SecretDecryptError as exc:
            logger.warning(
                "Shared secrets unavailable for model %s launch secrets: %s",
                job.service_name or job.id,
                exc,
            )
            return
        present = [key for key in wanted if key in shared_env]
        if not present:
            return
        env = config.env if config.env is not None else {}
        for key in present:
            env[key] = shared_env[key]
        config.env = env
        # Reuse the P8 shared-resolved audit (names only; no per-service override
        # concept for models, so `overridden` is always empty here).
        await self._audit_shared_resolved(job, present, {})

    @staticmethod
    def _volume_path_conflict(named_paths, mount_paths) -> str | None:  # noqa: ANN001
        """Return a message if a named volume nests with a Tier-B mount, else None.

        Normalizes both sides and checks *prefix* containment in either direction
        (not just equality): `data:/data` collides with a user mount at
        `/data/sub` (and the trailing-slash variants), because bind-mounting one
        under the other shadows it (D-P14-4 / critique L2).
        """

        # `os.path.normpath` preserves an exactly-two-slash prefix (`//data`
        # stays `//data`) that the kernel collapses to `/data`; collapse it
        # here too so `//data` and `/data` are recognised as the same mount.
        def _canon(p: str) -> str:
            return "/" + os.path.normpath(p).lstrip("/")

        norm_mounts = [_canon(p) for p in mount_paths]
        for raw in named_paths:
            npath = _canon(raw)
            for mount in norm_mounts:
                if npath == mount or npath.startswith(mount + "/") or mount.startswith(npath + "/"):
                    return f"named volume {raw!r} conflicts with mount {mount!r}"
        return None

    def _ensure_volume_dirs(self, service_name: str, named_volumes: dict[str, str]) -> None:
        """Create the named-volume leaf dirs and enforce the D-P14-3 perms scheme.

        The per-service root (`<data_dir>/services` and `.../<name>`) is held
        `0o700` (daemon-owned), while each volume leaf is `0o1777` (sticky, the
        docker named-volume norm) so a non-root container user can write into it.
        The parent 0o700 is *enforced* — not assumed — on every launch, which is
        what makes the world-writable leaf safe. `self._data_dir` is
        guaranteed non-None by the caller.
        """
        assert self._data_dir is not None
        services_root = self._data_dir / "services"
        service_root = services_root / service_name
        for parent in (services_root, service_root):
            parent.mkdir(parents=True, exist_ok=True)
            os.chmod(parent, 0o700)
        for host in named_volumes:
            leaf = Path(host)
            leaf.mkdir(parents=True, exist_ok=True)
            os.chmod(leaf, 0o1777)

    async def _settle_launch_user_error(self, job: Job, reason: str, message: str) -> None:
        """Settle a service terminally on a deterministic launch-time user error.

        Used for volume-spec violations (`volume_invalid`/`volume_conflict`):
        a corrupted/forged or conflicting `volumes` blob is deterministic, so
        the row is failed (never retry-looped). Releases any resources acquired
        earlier this tick (GPUs + endpoint), mirrors the sandbox-denied settle
        path, and stamps `last_deploy.reason` for `/diagnose`.
        """
        await self._queries.release_gpus(job.id)
        await self._release_endpoint(job)
        await self._queries.update_job_status(
            job.id,
            JobStatus.failed,
            finished_at=datetime.now(UTC),
            exit_code=-1,
            error_class=ErrorClass.user_error,
            error_message=message,
        )
        await self._queries.set_desired_state(job.id, JobStatus.failed.value)
        await stamp_last_deploy(
            self._queries,
            job.id,
            phase="failed",
            reason=reason,
            error_class=ErrorClass.user_error.value,
            error_message=message,
        )
        await self._append_log_tolerant(
            job.id, f"Service launch failed ({reason}): {message}", LogStream.system
        )
        self._emit_status_change(job.id, JobStatus.failed)
        await self._audit("service.failed", job)
        # `reason` is the caller's fixed token
        # (volume_invalid/volume_conflict) — never `message`, which quotes the
        # offending spec.
        await record_job_event(
            self._events,
            "service.failed",
            job,
            reason=reason,
            data={"error_class": ErrorClass.user_error.value},
        )
        logger.warning("Service %s launch failed (%s): %s", job.service_name, reason, message)

    def _build_command(
        self, job: Job, cfg: dict
    ) -> tuple[list[str] | None, dict[str, str], str | None]:
        """Resolve the container command + workspace mount for a service.

        Command mode (`cfg['command']`) runs a shell command with no workspace
        mount; script mode mounts the script's directory at `/workspace`.
        With neither, the command is `None` so the prebuilt image's own
        `CMD` runs (the P2 register-only path) — an empty list would instead
        override `CMD` with nothing. Folder builds arrive in P4.
        """
        custom_command = cfg.get("command")
        if custom_command:
            return shlex.split(custom_command), {}, None
        if job.script_path:
            script_path = Path(job.script_path)
            volumes = {str(script_path.parent): "/workspace"}
            return ["python", f"/workspace/{script_path.name}"], volumes, "/workspace"
        return None, {}, None

    @staticmethod
    def _requested_vendor(cfg: dict) -> GpuVendor | None:
        """Return the GPU vendor pinned in the service config, or `None` for any.

        A malformed vendor is tolerated (treated as `None`) rather than wedging
        the service — placement simply falls back to any available vendor.
        """
        raw = cfg.get("vendor")
        if not raw:
            return None
        try:
            return GpuVendor(raw)
        except ValueError:
            logger.warning("Service config carries unknown vendor %r; ignoring", raw)
            return None

    @staticmethod
    def _is_port_bindable(port: int) -> bool:
        """Whether `port` can currently be bound on loopback (CRIT-4 probe).

        A held endpoint port grabbed by a foreign process while the daemon was
        down would otherwise make every restart fail on the same port and burn
        the restart budget. Probing lets `Queries.acquire_service_port`
        drop the stale reservation and reallocate.

        `SO_REUSEADDR` is set to mirror what Docker's port binding does: right
        after a `docker kill` the service's own host port lingers in
        `TIME_WAIT` while the proxy tears down, and a plain `bind` would
        report it unbindable for up to a minute — a false negative that would
        needlessly drift the service URL to a new port. Docker binds such a port
        fine (it uses `SO_REUSEADDR`), so the probe must too, or the stable-port
        guarantee breaks on every hard restart.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    # --- (C) desired = running, container live -----------------------------

    async def _stamp_healthy_from_launching(self, job: Job) -> None:
        """Advance `config['last_deploy']` phase `launching` → `healthy`.

        Shared by both liveness paths in `_reconcile_live`: the
        no-health-check branch (a live container is healthy by definition) and
        the first-2xx-probe branch. A `kind=model` row additionally requires
        `config['model_pulled']` and a `kind=database` row
        `config['db_ready']` — neither is "healthy" until its async
        readiness step lands even though its container is already live. `verifying` joins
        `launching` as an origin: the
        promotion edge is `verifying|launching → healthy`. The cutover's own
        commit tail normally stamps it (f5), so this only fires for a row left
        parked at `verifying` — it can never race an in-flight verify, since
        `maybe_cutover`'s layer 1 returns before `_reconcile_live` is
        reached.
        """
        stamped = await stamp_last_deploy(
            self._queries,
            job.id,
            only_from=("launching", "verifying"),
            require_ready_flag=(
                "model_pulled"
                if job.kind is JobKind.model
                else "db_ready"
                if job.kind is JobKind.database
                else None
            ),
            phase="healthy",
        )
        # Edge-triggered on the actual write: this method runs on
        # EVERY reconcile tick of a live service, so emitting unconditionally
        # would put one row per coalescing window on the feed forever.
        if stamped:
            await record_job_event(self._events, "service.healthy", job)

    async def _reconcile_live(self, job: Job, budget: list[int]) -> None:
        """Health-check a running service (and re-adopt its log stream).

        Daemon-restart recovery rides here: a still-running container adopted on
        the first post-reboot tick gets its log stream re-spawned and its health
        loop resumed. Per Decision #2 a failing check only downgrades to
        `degraded` — the container is never killed on health alone.
        """
        now = datetime.now(UTC)
        # Re-adopt the log stream if we are not already collecting it (reboot).
        if job.container_id and job.container_id not in self._log_tasks:
            self._spawn_log_task(job.id, job.container_id)

        # Model re-adoption: a daemon reboot mid-weights-pull leaves a live
        # container with config['model_pulled'] unset — re-fire the idempotent
        # ensure_model. needs_ensure is a cheap in-memory pre-check, so a model
        # that is pulled (or already attempted on this container) never pays
        # the endpoint query per tick.
        if (
            job.kind is JobKind.model
            and self._models is not None
            and job.container_id
            and self._models.needs_ensure(job)
        ):
            model_ep = await self._queries.get_service_endpoint(job.service_name or "")
            if model_ep is not None:
                self._models.on_running(job, job.container_id, model_ep.host_port)

        # Database re-adoption: a daemon reboot mid-initdb/WAL-recovery
        # leaves a live container with config['db_ready'] unset — re-fire the
        # idempotent ensure_ready. needs_ensure is the same cheap in-memory
        # pre-check as the model path.
        if (
            job.kind is JobKind.database
            and self._data is not None
            and job.container_id
            and self._data.needs_ensure(job)
        ):
            db_ep = await self._queries.get_service_endpoint(job.service_name or "")
            if db_ep is not None:
                self._data.on_running(job, job.container_id, db_ep.host_port)

        hc = job.health_check
        if not hc:
            # Liveness-only: a live container is healthy by definition.
            # Advance launching → healthy once (guarded); a kind=model
            # row additionally waits for config['model_pulled'] and a kind=database
            # row for config['db_ready'].
            await self._stamp_healthy_from_launching(job)
            if job.status != JobStatus.running:
                await self._queries.update_job_status(job.id, JobStatus.running)
                self._emit_status_change(job.id, JobStatus.running)
            return

        if within_start_period(hc, job.started_at, now=now):
            return  # within the grace period — defer health judgement

        if budget[0] <= 0:
            return  # tick health budget spent (CRIT-5) — check on a later tick
        budget[0] -= 1

        endpoint = await self._queries.get_service_endpoint(job.service_name or "")
        if endpoint is None:
            return  # no published port to probe (should not happen for a live svc)
        timeout = _as_float(hc.get("timeout_s"), _DEFAULT_HEALTH_TIMEOUT_S)
        # Dispatch on probe kind via the shared health.run_probe; a
        # junk/absent type falls through to http. A tcp connect synthesizes code
        # 200 so the failure/degraded machine below is oblivious to the probe kind.
        code, _kind = await run_probe(
            hc,
            # `live_port`, not `host_port`: after a
            # successful cutover the live container is bound to the transient
            # green port until the next ordinary launch clears the pointer.
            # Probing the stable port there would flip a perfectly healthy
            # generation to `degraded` — and a degraded row never relaunches.
            endpoint.live_port,
            timeout,
            http_check=self._check_health,
            tcp_check=self._check_tcp,
        )

        if code is not None and 200 <= code < 300:
            self._health_failures.pop(job.id, None)
            # First health 2xx flips launching → healthy (guarded; a
            # kind=model row also requires config['model_pulled'], a kind=database
            # row config['db_ready'] — P15).
            await self._stamp_healthy_from_launching(job)
            if job.status != JobStatus.running:
                await self._queries.update_job_status(job.id, JobStatus.running)
                self._emit_status_change(job.id, JobStatus.running)
                logger.info("Service %s recovered → running", job.service_name or job.id)
            return

        threshold = _as_int(hc.get("unhealthy_threshold"), _DEFAULT_UNHEALTHY_THRESHOLD)
        failures = self._health_failures.get(job.id, 0) + 1
        self._health_failures[job.id] = failures
        if failures >= threshold and job.status != JobStatus.degraded:
            # Decision #2: degraded STAYS RUNNING — never auto-killed on health.
            await self._queries.update_job_status(job.id, JobStatus.degraded)
            await self._append_log_tolerant(
                job.id,
                f"Service unhealthy ({failures} consecutive failures) → degraded "
                "(container left running)",
                LogStream.system,
            )
            self._emit_status_change(job.id, JobStatus.degraded)
            await self._audit("service.degraded", job)
            # Already edge-triggered by the status guard above, and
            # coalesced on top of it — the degraded↔running flap guard.
            await record_job_event(
                self._events,
                "service.degraded",
                job,
                reason="health_check_failed",
                data={"failures": failures},
            )
            logger.warning("Service %s degraded (health)", job.service_name or job.id)

    async def _check_health(self, host_port: int, path: str, timeout: float) -> int | None:
        """Delegate to the module-level `check_health`.

        Kept as a method so the reconcile call site (`self._check_health`) and
        the tests that monkeypatch it stay stable (F6-PROBE-DUP dedup).
        """
        return await check_health(host_port, path, timeout)

    async def _check_tcp(self, host_port: int, timeout: float) -> bool:
        """Delegate to the module-level `check_tcp` (P14 WP-C1).

        Kept as a method for the same monkeypatch stability as `_check_health`.
        """
        return await check_tcp(host_port, timeout)

    # --- one-off container execution ----------------------------------

    async def _read_run_tail(self, container_id: str, log_tail: int) -> list[str]:
        """Read a runtime-bounded RAW log tail off a finished container.

        Bounded twice, both at READ time: the runtime caps the number of lines
        (`tail`) and the total bytes it will materialize (`max_bytes`).
        That pair — not the per-line clamp — is what keeps a chatty migration
        from ballooning the daemon's memory, so the lines can safely come back
        un-clamped. They come back un-scrubbed too: redaction and the per-line
        clamp are the caller's, in that order (`_scrub_and_truncate`),
        because clamping here would cut a long secret mid-value and defeat the
        literal match (PR #96 review F2).

        Tolerant on purpose: the container has already run, so a failed log
        read must degrade to a partial (or empty) tail, never turn a completed
        run into a runtime error the route would report as 503.
        """
        lines: list[str] = []
        try:
            async for line in self._runtime.logs(
                container_id, tail=log_tail, max_bytes=_RUN_TAIL_MAX_BYTES
            ):
                lines.append(line)
        except Exception:  # noqa: BLE001 - a lost tail must not fail a finished run
            logger.warning(
                "Could not read the full log tail of run container %s", container_id, exc_info=True
            )
        return lines

    async def _execute_container_once(
        self,
        job: Job,
        *,
        run_id: str,
        config: ContainerConfig,
        timeout_s: int,
        log_tail: int,
        scrub_values: Iterable[str],
    ) -> RunResult:
        """Start, wait with a hard timeout, capture scrubbed output, and remove a container.

        Callers register a slot before awaiting and discard it in `finally`; this
        method only binds the container ID. Callers also own all DB bookkeeping.
        Kill on timeout, inspect the outcome, scrub bounded raw lines, then clamp.
        All response, last-run, and release-log tails consume this same scrubbed result.

        Raises:
            ContainerRuntimeError: Start failed.
            RunInterruptedError: The runtime lost a container after it started.
        """
        started = datetime.now(UTC)
        started_monotonic = time.monotonic()
        # Attribution labels, stamped at the ONE choke point both
        # run_once and the release share, so every rowless container carries
        # them. This is what lets `_settle_crashed_release` reap a crash-
        # orphaned migration by `nerdit-job=<job_id>` instead of waiting out
        # the 30 s + 300 s zombie sweep — after a crash the in-memory registry
        # (and the run_id) is gone, so the JOB id is the only recoverable key.
        # Platform labels still win in DockerRuntime's merge.
        config.extra_labels = {
            **(config.extra_labels or {}),
            "nerdit-run": run_id,
            "nerdit-job": job.id,
        }
        container_id = await self._runtime.run(config)
        # Sweep-safety of the window between `run()` returning and this bind:
        # `ZombieSweeper` only kills containers older than
        # ZOMBIE_AGE_THRESHOLD_SECONDS (30 s, core/sweeper.py) and the bind is
        # the very next statement — a just-created container is never eligible,
        # so the unprotected window cannot be observed by a sweep tick.
        self._bind_run_container(job.id, run_id, container_id)

        exit_code: int | None = None
        timed_out = False
        try:
            try:
                exit_code = await self._runtime.wait(container_id, timeout_s=timeout_s)
            except TimeoutError:
                # The cap is enforced INSIDE the runtime's bounded wait, never by
                # an `asyncio.wait_for` over the raw wait: cancelling the
                # coroutine would leave the blocking thread pinned, and a few
                # timed-out runs would exhaust the shared executor (the
                # post-P15 pool-pinning bug class).
                timed_out = True
                logger.warning(
                    "Run %s of service %s exceeded %ss; killing container",
                    run_id,
                    job.service_name or job.id,
                    timeout_s,
                )
                try:
                    await self._runtime.kill(container_id)
                except ContainerRuntimeError:
                    logger.warning(
                        "Could not kill timed-out run container %s", container_id, exc_info=True
                    )
            # Inspect BEFORE the removal below (load-bearing order, as on the
            # crash path): once removed, the OOM flag and exit code are gone.
            state = None
            try:
                state = await self._runtime.inspect_state(container_id)
            except Exception:  # noqa: BLE001 - forensics must never break the run path
                state = None
            oom_killed = _coerce_oom(state)
            if exit_code is None:
                # Fallback for the timed-out branch (and for a lost wait): the
                # kill above produced a real exit code on the container.
                exit_code = _coerce_exit_code(state)
            tail = _scrub_and_truncate(
                await self._read_run_tail(container_id, log_tail), scrub_values
            )
        except ContainerRuntimeError as exc:
            # Everything above happens AFTER `runtime.run()` returned, so the
            # container did start: a raise here is the runtime losing a live
            # container (a docker hiccup, a lost wait), not a start failure.
            # Re-raised as the subclass so the caller can tell those apart and
            # still show what the container printed — `_read_run_tail` swallows
            # its own failures, so this best-effort tail can never mask `exc`,
            # and it goes through the SAME scrub-then-clamp pipeline as the
            # happy path.
            raise RunInterruptedError(
                str(exc),
                log_tail=_scrub_and_truncate(
                    await self._read_run_tail(container_id, log_tail), scrub_values
                ),
            ) from exc
        finally:
            # Best-effort: a leaked run container is reaped by the zombie sweep
            # once the caller's `finally` drops its registry slot.
            try:
                await self._runtime.remove(container_id, force=True)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                logger.warning(
                    "Could not remove run container %s after run %s",
                    container_id,
                    run_id,
                    exc_info=True,
                )
        finished = datetime.now(UTC)
        return RunResult(
            run_id=run_id,
            exit_code=exit_code,
            timed_out=timed_out,
            oom_killed=oom_killed,
            duration_s=round(time.monotonic() - started_monotonic, 3),
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            log_tail=tail,
        )

    # --- Shared helpers -----------------------------------------------------

    def _emit_status_change(self, job_id: str, status: JobStatus) -> None:
        """Best-effort emit on the event bus. Silent if no bus configured."""
        if self._event_bus is None:
            return
        self._event_bus.publish(
            {
                "type": "job.status_changed",
                "job_id": job_id,
                "status": status.value,
                "ts": datetime.now(UTC).isoformat(),
            }
        )

    async def _audit(self, action: str, job: Job) -> None:
        """Record an autonomous transition as a `principal='system'` audit row.

        Decision #3: `ServiceController` audits its own `service.restart` /
        `service.failed` / `service.degraded` transitions, on top of the
        `job.status_changed` SSE — these have no human principal, so the audit
        row is the only durable record of *who* (the controller) acted. Secrets
        never reach here; only the service name and restart counter are recorded.
        """
        try:
            await self._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=job.id,
                params_redacted=json.dumps(
                    {"service_name": job.service_name, "restart_count": job.restart_count}
                ),
            )
        except Exception:
            logger.warning("Failed to audit %s for service %s", action, job.id, exc_info=True)

    # --- deploy build -----------------------------------------------
    # Keep controller-level hooks overridable by callers and tests.

    async def _needs_build(self, job: Job) -> bool:
        return await self._builder.needs_build(job)

    def _spawn_build_task(
        self, job: Job, context_dir: str, image: str, dockerfile: str | None, cleanup_dir: str
    ) -> None:
        self._builder.spawn(job, context_dir, image, dockerfile, cleanup_dir)

    async def _build_app_image(
        self, job: Job, context_dir: str, image: str, dockerfile: str | None, cleanup_dir: str
    ) -> None:
        await self._builder.build(job, context_dir, image, dockerfile, cleanup_dir)

    async def _is_container_live(self, container_id: str) -> bool:
        """Best-effort probe: is `container_id` a container that is actually up?

        `status()` returns a docker state STRING (`"exited"`, `"dead"`, …)
        for a stopped-but-present container, so a bare truthiness check reads
        every corpse as live — the P24b live run caught a crashed cutover green
        burning the whole verify budget that way instead of failing
        immediately. Only `running` (and a docker-native `restarting`,
        which is "about to be up") count.
        """
        try:
            return await self._runtime.status(container_id) in ("running", "restarting")
        except ContainerRuntimeError:
            return False

    async def _container_exists(self, container_id: str) -> bool:
        """Best-effort probe: does the runtime still know `container_id` at all?

        Truthy for ANY present container — including a stopped/exited corpse
        (`status()` returns the docker state string for those, `None` only
        once the container is gone). This is the redeploy-over-a-previous-
        generation discriminator (`AppImageBuilder._settle_failed_generation`):
        a previous container that merely exited still means the row was a live
        service with a `previous_image` to fall back to, so a failed build
        must REVERT (self-heal on the next reconcile), never terminally fail
        the whole service. Distinct from `_is_container_live`, which the
        cutover verify path uses and which counts only running/restarting.
        """
        try:
            return bool(await self._runtime.status(container_id))
        except ContainerRuntimeError:
            return False

    async def _prune_old_images(self, job: Job) -> None:
        await self._builder.prune_old_images(job)

    def _spawn_log_task(self, job_id: str, container_id: str) -> None:
        """Start (and track) a background log-collection task for a container."""
        if container_id in self._log_tasks:
            return
        task = asyncio.create_task(self._collect_logs(job_id, container_id))
        self._log_tasks[container_id] = task
        task.add_done_callback(self._discard_log_task)

    def _discard_log_task(self, task: asyncio.Task) -> None:
        """Drop a finished log task from the tracking map (by identity)."""
        for cid, tracked in list(self._log_tasks.items()):
            if tracked is task:
                self._log_tasks.pop(cid, None)
                break

    def _cancel_log_task(self, container_id: str | None) -> None:
        """Cancel the tracked log task for `container_id` if any."""
        if not container_id:
            return
        task = self._log_tasks.pop(container_id, None)
        if task is not None:
            task.cancel()

    async def _collect_logs(self, job_id: str, container_id: str) -> None:
        """Background task to collect container logs into the database."""
        try:
            async for line in self._runtime.logs(container_id, follow=True):
                await self._append_log_tolerant(job_id, line, LogStream.stdout)
        except ContainerRuntimeError as exc:
            logger.info("Log stream closed for service %s: %s", job_id, exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Log collection failed for service %s", job_id)

    async def shutdown(self) -> None:
        """Cancel log/health tasks; **leave service containers running**.

        Desired state is persisted, so the next daemon boot re-adopts a still-live
        container (or relaunches a dead one on its stable host port). Killing
        containers here would defeat survive-a-reboot.
        """
        for task in list(self._log_tasks.values()) + list(self._build_tasks.values()):
            task.cancel()
        for task in list(self._log_tasks.values()) + list(self._build_tasks.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._log_tasks.clear()
        self._build_tasks.clear()
        logger.info("ServiceController stopped (containers left running)")
