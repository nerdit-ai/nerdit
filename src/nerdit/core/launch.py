"""Shared launch, one-off run, sandbox, and log-scrubbing helpers.

`ServiceController._launch` orders GPU placement, stable-port acquisition,
container assembly, sandbox overlay, config finalization, and settlement.
It retains env resolution and mount/volume validation. Helpers take the
controller explicitly when they need its state.

Database readiness is cleared for each new container before marking it running;
model readiness survives because pulled weights are durable. One-off runs
share sandbox/finalization rules but expose no ports or GPUs. Secret scrubbing
and log byte caps live here so core callers need no daemon imports.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from nerdit.config.settings import ContainerSettings, RetentionSettings
from nerdit.core.data.backend import DataBackend
from nerdit.core.deploy_state import stamp_last_deploy
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.proxy import generate_route
from nerdit.core.runtime.container import ContainerConfig
from nerdit.db.enums import GpuVendor, JobKind, JobStatus, LogStream
from nerdit.db.queries import PortRangeExhausted
from nerdit.db.rows import Gpu, Job, ServiceEndpoint

if TYPE_CHECKING:
    from nerdit.core.services import ServiceController

logger = logging.getLogger(__name__)

# Grace for a service's own host port to be released by a just-removed
# container's docker-proxy before the CRIT-4 fallback treats it as foreign and
# reallocates (which would drift the URL). Enforced by DEFERRING the launch to a
# later reconcile tick (never a blocking in-tick sleep — RECON-1), measured from
# the recorded exit. Past the grace a genuinely-occupied port falls through to
# reallocation.
_PORT_RELEASE_GRACE_S = 4.0

# Byte budget for a one-off run's (or a release's) captured stdout tail:
# the whole tail is capped at 16 KiB and any single line at 2 KiB before it
# reaches the run response, `config['last_run']` or a release's `job_logs`.
# The tail cap is enforced at READ time by the runtime's bounded log reader
# (never post-materialization), so a chatty migration can never balloon the
# daemon's memory or the row's config blob. Homed here rather than in
# `daemon/limits.py` (where the route-facing `_MAX_RUN_LOG_TAIL` lives)
# because the enforcement site is the controller tier and `core` must never
# import `daemon`.
_RUN_TAIL_MAX_BYTES = 16384
_RUN_LINE_MAX_BYTES = 2048

# Shortest value the credential-NAME heuristic will redact. A 1-5 char
# value matched only by the spelling of its key (a dummy dev key, "x", "test",
# a port number sitting under `DB_PASSWORD`) occurs all over unrelated
# output, so scrubbing it would mangle the tail into noise — and the mangling
# itself would advertise where the value appears.
#
# The floor is a guard on the platform's own GUESS, so it lives at the set
# builder (`app_build._sensitive_env_values`), where provenance is still
# known — NOT at `scrub_secret_values`, which masks whatever it is
# handed. Applied at the consumer, it would let a DECLARED secret shorter
# than this (the secrets API enforces no minimum length) reach `job_logs`
# verbatim, readable by any authenticated principal.
_SCRUB_MIN_VALUE_LEN = 6


def scrub_secret_values(lines: list[str], values: Iterable[str]) -> list[str]:
    """Mask every nonempty sensitive value, longest first, before line truncation.

    Callers decide sensitivity and any heuristic length floor; this function masks
    even one-character declared secrets. Longest-first replacement prevents an
    embedded password from breaking the match for its whole DSN. Lexical tie-breaks
    keep output deterministic; ignore empty values rather than inserting masks
    between characters.

    Literal matching cannot mask transformed/split secrets or fragments already
    clipped by the runtime's raw-byte tail budget. Per-line clipping must happen
    after this scrub; closing the runtime residual requires redaction before its
    byte budget is applied.
    """
    targets = sorted(
        {value for value in values if value},
        key=lambda value: (-len(value), value),
    )
    if not targets:
        return list(lines)
    scrubbed: list[str] = []
    for line in lines:
        masked = line
        for value in targets:
            masked = masked.replace(value, "***")
        scrubbed.append(masked)
    return scrubbed


class GpuPlacement(NamedTuple):
    """A resolved GPU placement: raw DB ids (for the allocation write) plus
    runtime selectors (for the container config) and the winning vendor. `min_memory_mb` is the
    smallest allocated card's total VRAM — the
    signal `VllmBackend` keys its conservative-bounds injection off. `None`
    when the discovery backend reported no usable size, and the injection is
    then skipped (known VRAM only).
    """

    gpu_ids: list[str]
    runtime_gpu_ids: list[str]
    vendor: GpuVendor
    min_memory_mb: int | None = None


def plan_gpu_placement(
    candidates: list[Gpu], gpu_count: int, requested: GpuVendor | None
) -> GpuPlacement | None:
    """Pick `gpu_count` GPUs of one vendor from `candidates`, or `None`.

    Pure selection only: group the candidates by vendor,
    try the requested vendor first (else NVIDIA then AMD), and take the first
    pool with enough capacity. The DB fetch of `candidates` and the
    `allocate_gpus` write (with its `RuntimeError` conflict handling) stay
    in the caller — this function never touches the database.
    """
    by_vendor: dict[GpuVendor, list[Gpu]] = {}
    for gpu in candidates:
        by_vendor.setdefault(gpu.vendor, []).append(gpu)
    order = [requested] if requested else [GpuVendor.nvidia, GpuVendor.amd]
    chosen: list[Gpu] | None = None
    for cand in order:
        pool = by_vendor.get(cand, [])
        if len(pool) >= gpu_count:
            chosen = pool[:gpu_count]
            break
    if chosen is None:
        return None
    sizes = [g.memory_mb for g in chosen if isinstance(g.memory_mb, int) and g.memory_mb > 0]
    return GpuPlacement(
        gpu_ids=[gpu.id for gpu in chosen],
        runtime_gpu_ids=[gpu.runtime_id or gpu.id for gpu in chosen],
        vendor=chosen[0].vendor,
        min_memory_mb=min(sizes) if sizes else None,
    )


async def require_minted_credential(
    sc: "ServiceController", job: Job, cfg: dict, secret_env: dict[str, str]
) -> DataBackend | None:
    """Return the database backend only when its scoped minted credential exists.

    Read the already-resolved secret env before acquiring resources. Missing or
    corrupt credentials defer launch with a restore/recreate hint, never trigger
    trust-auth initialization. Database env remains allowlisted to backend statics
    and its minted credential; decryption errors are handled during resolution.
    """
    assert sc._data is not None
    backend = sc._data.backend_for(cfg)
    if backend.minted_secret_key not in secret_env:
        await sc._log_secret_wait(
            job,
            "minted credential missing — the database cannot start; "
            "delete and recreate it, or restore a backup",
        )
        return None
    return backend


async def resolve_host_port(
    sc: "ServiceController", job: Job, container_port: int, now: datetime
) -> ServiceEndpoint | None:
    """Acquire (or re-confirm) the service's stable host port, or defer/retry.

    CRIT-4 bindability fallback. Returns `None` when the launch must be
    DEFERRED (the just-removed container's docker-proxy grace window) or
    RETRIED next tick (the port range is exhausted) — the caller returns
    immediately in either case; any GPU acquired earlier this tick is released
    here so nothing is held across the wait.
    """
    assert job.service_name is not None  # service rows always carry a name
    # A hard `docker kill` releases the container's port asynchronously: the
    # docker-proxy for the just-removed container can still hold the host
    # port for a moment. If the service's *own* held port is not yet
    # bindable, DEFER this launch to the next reconcile tick rather than
    # blocking the shared loop with a sleep (RECON-1) — the 2s loop_interval
    # paces the retry. `last_exit_at` bounds the deferral: past the grace we
    # fall through so the CRIT-4 fallback reallocates a *persistently*
    # occupied port. (With SO_REUSEADDR the own port is usually bindable
    # immediately, so this rarely defers at all.)
    held = await sc._queries.get_service_endpoint(job.service_name)
    if held is not None and not sc._is_port_bindable(held.host_port):
        waited = (
            (now - job.last_exit_at).total_seconds() if job.last_exit_at else _PORT_RELEASE_GRACE_S
        )
        if waited < _PORT_RELEASE_GRACE_S:
            if job.gpu_count > 0:
                await sc._queries.release_gpus(job.id)  # don't hold GPUs while deferring
            logger.debug(
                "Service %s host port %d not yet released; deferring launch",
                job.service_name,
                held.host_port,
            )
            return None
    try:
        return await sc._queries.acquire_service_port(
            job.service_name,
            job.id,
            container_port,
            sc._port_range,
            is_bindable=sc._is_port_bindable,
        )
    except PortRangeExhausted:
        logger.warning(
            "No free host port for service %s, will retry",
            job.service_name,
        )
        await sc._queries.release_gpus(job.id)
        return None


class AppContainerSpec(NamedTuple):
    """The service-branch shape inputs to `build_app_container_config`.

    Bundled (rather than 8 loose keyword args) to keep this module's ruff
    `PLR0913` (max-args=6) clean with zero per-file-ignore entry.
    """

    command: list[str] | None
    volumes: dict[str, str] | None
    env: dict[str, str] | None
    workdir: str | None
    gpu_ids: list[str]
    vendor: GpuVendor
    container_port: int
    host_port: int


class _SandboxProfile(NamedTuple):
    """The resource limits + sandbox hardening one launch resolves from the
    row config and the daemon's container settings."""

    memory_limit: str | None
    cpu_limit: float | None
    pids_limit: int | None
    cap_drop: list[str] | None
    no_new_privileges: bool
    read_only: bool


def _sandbox_profile(cfg: dict, cs: ContainerSettings) -> _SandboxProfile:
    """Resolve shared service/run limits and hardening so one-off containers are no weaker."""
    return _SandboxProfile(
        memory_limit=(cfg.get("memory_limit") or cs.default_memory_limit),
        cpu_limit=cfg.get("cpu_limit") or cs.default_cpu_limit,
        pids_limit=cs.pids_limit,
        cap_drop=["ALL"] if cs.drop_all_caps else None,
        no_new_privileges=cs.no_new_privileges,
        read_only=cs.read_only_rootfs,
    )


def build_app_container_config(
    cfg: dict, cs: ContainerSettings, spec: AppContainerSpec
) -> ContainerConfig:
    """Build the plain-service `ContainerConfig` (the non-model, non-data branch).

    Verbatim motion of `_launch`'s `else` branch: a service row's shape is
    entirely config-driven (image/command/volumes/env from the deploy or
    `nerdit.toml`), unlike the model/data branches which get their shape
    from a backend. The limits + hardening block now reads from the
    shared `_sandbox_profile` — same values, one source.
    """
    sandbox = _sandbox_profile(cfg, cs)
    return ContainerConfig(
        image=cfg.get("image") or cs.default_image,
        gpu_ids=spec.gpu_ids,
        vendor=spec.vendor,
        command=spec.command,
        volumes=spec.volumes or None,
        env=spec.env or None,
        workdir=spec.workdir,
        memory_limit=sandbox.memory_limit,
        cpu_limit=sandbox.cpu_limit,
        pids_limit=sandbox.pids_limit,
        cap_drop=sandbox.cap_drop,
        no_new_privileges=sandbox.no_new_privileges,
        read_only=sandbox.read_only,
        # Forced bridge (NOT _resolve_network_mode): published ports require
        # a routable namespace; an agent token's default 'none' would drop
        # them.
        network_mode="bridge",
        ports={spec.container_port: spec.host_port},
    )


def build_run_container_config(
    cfg: dict,
    cs: ContainerSettings,
    *,
    image: str,
    command: list[str],
    env: dict[str, str] | None,
    workdir: str | None,
) -> ContainerConfig:
    """Build a portless, GPU-free run/release container from the supplied deployed image.

    Share service sandbox limits, capability drops, privilege/read-only settings,
    and bridge networking. Never fall back to a default image or claim a service
    port. Leave volumes unset here; finalization adds validated named volumes and
    log caps. Runs inherit no host-mounted scripts.
    """
    sandbox = _sandbox_profile(cfg, cs)
    return ContainerConfig(
        image=image,
        gpu_ids=[],
        command=command,
        volumes=None,
        env=env or None,
        workdir=workdir,
        memory_limit=sandbox.memory_limit,
        cpu_limit=sandbox.cpu_limit,
        pids_limit=sandbox.pids_limit,
        cap_drop=sandbox.cap_drop,
        no_new_privileges=sandbox.no_new_privileges,
        read_only=sandbox.read_only,
        # Same forced bridge as the service branch (NOT _resolve_network_mode):
        # a run inherits the service's network posture, not an agent token's.
        network_mode="bridge",
        ports=None,
    )


def apply_platform_overlay(
    config: ContainerConfig,
    cfg: dict,
    cs: ContainerSettings,
    container_port: int,
    host_port: int,
) -> None:
    """Harden a backend-built model/database config exactly like a plain app service."""
    sandbox = _sandbox_profile(cfg, cs)
    config.memory_limit = sandbox.memory_limit
    config.cpu_limit = sandbox.cpu_limit
    config.pids_limit = sandbox.pids_limit
    config.cap_drop = sandbox.cap_drop
    config.no_new_privileges = sandbox.no_new_privileges
    config.read_only = sandbox.read_only
    config.network_mode = "bridge"
    config.ports = {container_port: host_port}


def finalize_container_config(
    config: ContainerConfig,
    named_volumes: dict[str, str],
    retention: RetentionSettings | None,
) -> None:
    """Merge the daemon-computed named volumes and cap the container's json-file logs.

    Named volumes merge onto whatever the branch built (a service's Tier-B
    workspace mount, or a model's backend weights volume); a row that declares
    none is left untouched.

    Logs are capped (service + model kinds) when retention is configured. An
    empty `container_log_max_size` is the operator opt-out: set NO log_config at
    all, so DockerRuntime never forces LogConfig(type="json-file") and the
    container inherits the daemon's configured default driver
    (journald/local/fluentd/…) untouched. The runtime guard is a truthiness
    check, so a None log_config leaves the kwarg absent entirely.
    """
    if named_volumes:
        config.volumes = {**(config.volumes or {}), **named_volumes}

    if retention is not None and retention.container_log_max_size:
        config.log_config = {
            "max-size": retention.container_log_max_size,
            "max-file": str(retention.container_log_max_file),
        }


async def stamp_launching(
    sc: "ServiceController",
    job: Job,
    config: ContainerConfig,
    named_volumes: dict[str, str],
    *,
    phase: str | None = "launching",
) -> None:
    """Advance the current deploy generation and record launched env key names.

    Fresh guarded reads prevent an old relaunch from advancing newer work. Never
    persist env values. With phase=None, record provenance only: cutover already
    stamped verifying and must not re-enter launching.
    """
    if phase is not None:
        await stamp_last_deploy(
            sc._queries,
            job.id,
            only_from=("queued", "building"),
            require_version_match=True,
            phase=phase,
        )
    await sc._queries.patch_job_config(
        job.id,
        {
            "last_launch_env_keys": sorted((config.env or {}).keys()),
            # Provenance: the volume NAMES actually mounted this
            # launch (host paths are daemon-computed and never persisted).
            # Derived from the resolved `named_volumes` — not the raw config
            # list — so a launch that skipped mounting (no `data_dir`) never
            # claims volumes it did not mount. Host path is
            # `<root>/<volname>`, so the basename is the volume name.
            "last_launch_volumes": sorted(Path(host).name for host in named_volumes),
        },
    )


async def settle_started(
    sc: "ServiceController",
    job: Job,
    container_id: str,
    endpoint: ServiceEndpoint,
    *,
    now: datetime,
) -> None:
    """Post-run settle tail: clear stale flags, flip to `running`, wire it up.

    `model_pulled` is NEVER touched here — only the model error-message clear
    and the `db_ready` clear are launch-time clears.
    `is_model`/`is_data` are re-derived from `job`/`sc` (identical to
    the orchestrator's own top-of-`_launch` computation) rather than passed
    in, to keep this module's ruff `PLR0913` (max-args=6) clean with zero
    per-file-ignore entry.
    """
    is_model = job.kind is JobKind.model and sc._models is not None
    is_data = job.kind is JobKind.database and sc._data is not None
    # A fresh model container supersedes any prior weights-pull attempt, so
    # clear a now-stale "Model pull failed" message BEFORE the row goes
    # `running` — the off-tick ensure_model re-runs on this new container
    # and re-sets it if this attempt also fails. Without the clear, /wait +
    # /diagnose would read the old message on the newly-running row and
    # report `failed` while the retry is actively pulling and may succeed.
    # Cleared first (not after) so no (running ∧ stale-error) state is ever
    # observable to a concurrent /wait poll.
    if is_model and job.error_message:
        await sc._queries.set_job_error_message(job.id, None)
    # A fresh database container has never been probed, so any
    # persisted config['db_ready'] is stale — it asserted the PREVIOUS
    # container answered the wire probe. Clear it (and re-arm the probe for
    # this new container) BEFORE the row goes `running`, mirroring the model
    # error-message clear above: no (running ∧ stale db_ready) state is ever
    # observable to a concurrent /wait or binding resolution. on_launched
    # (unlike on_running, the re-adoption hook) does not gate on the stale
    # flag. Awaited so the clear is persisted before the status flip below.
    if is_data:
        assert sc._data is not None
        await sc._data.on_launched(job, container_id, endpoint.host_port)
    # An ordinary launch ALWAYS binds the stable `host_port`,
    # so any leftover cutover pointer is stale the instant this container
    # starts. Cleared unconditionally and BEFORE the status flip, with the other
    # pre-flip clears, so no (running ∧ stale active_host_port) state is ever
    # observable to a concurrent proxy reconcile or /routes read. This closes
    # the transient window at the first crash, restart, rollback or non-cutover
    # deploy — but it is deliberately NOT the only backstop: a `degraded` row
    # stays running and never relaunches, so the cutover crash settle clears the
    # column explicitly rather than trusting this one.
    if job.service_name:
        await sc._queries.set_endpoint_active_port(job.service_name, None)
    await sc._queries.update_job_status(
        job.id, JobStatus.running, container_id=container_id, started_at=now
    )
    # POST-LAUNCH FK PROBE — the one append site that must NOT simply
    # swallow. An IntegrityError here IS the deleted-row signal: the DELETE
    # route removed the `jobs` row (cascading `job_logs`) between run() and
    # this write, so the status update above hit zero rows and the fresh
    # container has no DB owner. Completing the wiring would spawn a log task
    # and (proxy on) register an HTTPS route for a service the user already
    # deleted, leaving it reachable until the proxy reconcile prunes the route
    # and the zombie sweep reaps the container. Re-read the row to tell a real
    # delete from a spurious FK: gone ⇒ tear the orphan container down and abort
    # the rest of the wiring (release NOTHING — the DELETE already released the
    # GPUs and the endpoint, and releasing the port here could free one a
    # recreated same-named service has since acquired); still there ⇒ keep the
    # tolerate-and-continue of every other append site.
    try:
        await sc._queries.append_log(
            job.id, f"Service container started: {container_id[:12]}", LogStream.system
        )
    except sqlite3.IntegrityError:
        if await sc._queries.get_job(job.id) is None:
            logger.info(
                "Service %s was deleted mid-launch; removing orphan container %s",
                job.service_name or job.id,
                container_id[:12],
            )
            await sc._destroy_container(container_id)
            return
        logger.debug("append_log skipped for job %s (spurious FK)", job.id)
    sc._emit_status_change(job.id, JobStatus.running)
    sc._health_failures.pop(job.id, None)
    sc._spawn_log_task(job.id, container_id)
    # Model post-launch hook: fire the off-tick, idempotent
    # ensure_model (weights pull) unless config['model_pulled'] already
    # says the weights are present.
    if is_model:
        assert sc._models is not None
        sc._models.on_running(job, container_id, endpoint.host_port)
    # The database readiness hook fired BEFORE the status flip above
    # (on_launched) so a fresh container's stale db_ready never surfaces as
    # ready; nothing to do here.
    # URL layer: persist the route projection + best-effort register so
    # the HTTPS URL is live this tick. The proxy reconcile loop is the actual
    # guarantee; this inline call (which never raises) just removes the lag.
    # kind=model rows are loopback-only: they never get a
    # Caddy route and their endpoint route projection stays NULL. kind=database
    # rows share that posture — positive `is service` form so
    # a fourth kind can never leak a route in again.
    if sc._proxy and sc._proxy.enabled and job.service_name and job.kind is JobKind.service:
        route = generate_route(job.service_name, mode=sc._proxy_mode)
        await sc._queries.set_endpoint_route(job.service_name, route)
        # The raw `edge_auth` blob rides along: without it this
        # inline rebuild would replace a protected route with an auth-free one
        # for a whole reconcile interval on every launch and every redeploy.
        # `register` parses/resolves it and, when it cannot, withholds the
        # route entirely rather than registering an open fallback — the launch
        # itself still succeeds (`register` never raises).
        await sc._proxy.register(
            job.service_name,
            endpoint.host_port,
            edge_auth=parse_job_config(job).get("edge_auth"),
            project_id=job.project_id,
        )
    logger.info(
        "Service %s running in container %s on host port %d",
        job.service_name,
        container_id[:12],
        endpoint.host_port,
    )
