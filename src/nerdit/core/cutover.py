"""Swap eligible services only after the replacement passes health and route checks.

Build and release run first. Green starts on a transient loopback port while
blue keeps serving on its existing port. Failed probes destroy green and revert the image;
an unverifiable proxy dial also unwinds the route, keeping blue alive.
`cutover_pending = {version, blue}` persists the gate. A crash before promotion
fails the generation; only an explicit redeploy retries it.

Commit order is required for recovery:

1. Persist green's active port before registering routes, so concurrent proxy
   reconciliation cannot restore blue's dial.
2. Read back every route (default and custom domains) in one live snapshot.
   All must dial green; a normal `register` return or unreadable route set is
   insufficient. On failure, unwind and verify the whole set against blue.
3. Recheck green's liveness after the bounded route wait, then promote its row.
4. Write the audit/event/healthy records before removing the crash marker.
   A restart replays this tail at least once if promotion already committed.
5. Remove the marker, then destroy blue. Destroying blue before promotion
   would leave crash recovery pointing to a dead container.

Before promotion, recovery restores blue's recorded port, reaps green, and
reverts while preserving blue. Clear the override only if blue used the stable
port (or an older marker lacks its port). If routes had reached green, a brief 502 can last until
one proxy tick restores blue. After promotion, recovery keeps green and replays
the commit tail; after marker removal, the zombie sweeper collects orphan blue.
The failure unwind is idempotent across crashes.

Controller hooks remain overridable through `self._c`; the controller owns
cutover task/container registries used by sweep, drain, and delete paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
import time
from collections.abc import Collection, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nerdit.core.deploy_state import stamp_last_deploy
from nerdit.core.eventlog import record_job_event
from nerdit.core.health import DEFAULT_HEALTH_TIMEOUT_S, as_float, run_probe
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.launch import (
    AppContainerSpec,
    build_app_container_config,
    finalize_container_config,
    stamp_launching,
)
from nerdit.core.proxy import _domain_route_id, _route_id
from nerdit.core.runtime.protocol import (
    ContainerNotFoundError,
    ContainerRuntimeError,
    SandboxViolationError,
)
from nerdit.core.sandbox import enforce_mount_allowlist, resolve_role
from nerdit.core.volumes import VolumeSpecError, service_volumes
from nerdit.db.enums import (
    TERMINAL_STATUSES,
    ErrorClass,
    GpuVendor,
    JobKind,
    JobStatus,
    LogStream,
    TokenRole,
)
from nerdit.db.rows import Job

if TYPE_CHECKING:  # pragma: no cover — import cycle (services imports this module)
    from nerdit.config.settings import ServicesSettings
    from nerdit.core.proxy import ProxyManager
    from nerdit.core.services import ServiceController

logger = logging.getLogger(__name__)

#: Poll cadence for the verify probe loop and the dial read-back. Module-level
#: so a test can monkeypatch it to something instant instead of waiting out real
#: seconds (`fake_clock` only patches `core.services`' `datetime` name).
_SLEEP = asyncio.sleep

#: The monotonic clock the budgets are measured against. Module-level for the
#: same reason as `_SLEEP`: a test patches the pair together (an instant
#: `_SLEEP` that advances a virtual clock) so the verify/repoint budgets are
#: exercised deterministically instead of by waiting out real seconds.
_MONOTONIC = time.monotonic

#: Seconds between probe attempts / dial read-backs. The budgets come from
#: `[services]`; this is just how often we look.
_POLL_INTERVAL_S = 1.0

#: Label stamped on the green container so a crash settle can reap it by job id
#: after the in-memory registry is gone. Deliberately DISJOINT from the
#: `nerdit-run` / `nerdit-job` pair: a run/release container can never
#: appear in a `nerdit-cutover` listing, so the crash reap needs no
#: abort-on-active-run belt (it needs — and has — the never-kill-the-row's-own
#: -container belt instead).
_CUTOVER_LABEL = "nerdit-cutover"

#: The two-value `stage` literal carried on a `service.cutover_failed`
#: audit row / event (D-P24-3 rule 1): "the new version is broken" vs "the proxy
#: could not be repointed" need opposite fixes.
_STAGE_PROBE = "probe"
_STAGE_REPOINT = "repoint"

_FAILED_MESSAGE = (
    "the new version failed its health verification; the previous version is "
    "still serving. Data changes are NOT rolled back."
)
_CRASH_MESSAGE = (
    "daemon restarted during the cutover verification; the new version was "
    "never promoted — redeploy to retry. Data changes are NOT rolled back."
)


def _cutover_audit_params(
    job: Job, version: int | None, *, data_rollback: bool, stage: str | None = None
) -> dict[str, object]:
    """Audit params for a `service.cutover` / `service.cutover_failed` row.

    Outcome facts only — never a port, never a container id, never a log line.
    `data_rollback` is always `False`: the platform reverts the image, never
    the data (the green mounts the SAME named volumes blue is writing).
    """
    params: dict[str, object] = {
        "service": job.service_name,
        "version": version,
        "data_rollback": data_rollback,
    }
    if stage is not None:
        params["stage"] = stage
    return params


def _as_port(value: object) -> int | None:
    """A marker field as a port int, or `None` (bools are not ports)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reserve_ephemeral_port(excluded: Collection[int] = ()) -> int:
    """Choose a free port outside all persisted stable and cutover reservations.

    Hold rejected sockets until selection completes so the OS cannot repeatedly
    return the same excluded port. This is still a hint: Docker detects a lost
    external bind race and the cutover keeps blue unchanged.
    """
    sockets: list[socket.socket] = []
    try:
        for _ in range(32):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port not in excluded:
                return port
        raise OSError("No unreserved ephemeral port available for cutover")
    finally:
        for sock in sockets:
            sock.close()


def cutover_skip_reason(
    job: Job, cfg: dict, settings: "ServicesSettings", proxy: "ProxyManager | None"
) -> str | None:
    """`None` when cutover-eligible; else the machine reason it is not.

    The reason vocabulary is pinned by D-P24-8: `proxy_off` | `gpu_bound` |
    `no_verify_signal` | `disabled`. It exists because the GitWatch poller must
    *tell* the operator why an unattended redeploy refused to fire, and a bare
    predicate cannot carry that — `_eligible` is this function's boolean
    face, so the two can never disagree.

    `disabled` is also the residual bucket for the two mechanical
    preconditions with no operator-facing story of their own (see below).
    """
    # Both of the following are structurally unreachable for a GitWatch
    # candidate — the poller only ever sees `kind=service` rows deployed
    # from git, which always carry an integer `build_version` — and are kept
    # as the residual `disabled` so the reason is never `None` by accident.
    if job.kind is not JobKind.service:
        return "disabled"
    if job.gpu_count > 0:
        return "gpu_bound"
    if proxy is None or not proxy.enabled or not proxy.available:
        return "proxy_off"
    if cfg.get("cutover") is False:
        return "disabled"
    version = cfg.get("build_version")
    if not isinstance(version, int) or isinstance(version, bool):
        return "disabled"
    if not job.health_check and settings.cutover_grace_s <= 0:
        return "no_verify_signal"
    return None


def _eligible(
    job: Job, cfg: dict, settings: "ServicesSettings", proxy: "ProxyManager | None"
) -> bool:
    """Check whether health-gated cutover can safely own this generation.

    Require an enabled, available proxy, a GPU-free service, no explicit cutover
    opt-out, and an integer build version for crash-safe CAS writes. Verification
    uses a health spec or a positive grace interval. Rows without a version keep
    the same-port swap path; models/databases are ineligible.
    """
    return cutover_skip_reason(job, cfg, settings, proxy) is None


class CutoverManager:
    """Owns the health-gated cutover: the arm, the verify task and both settles.

    Constructed by `nerdit.core.services.ServiceController` next to
    `nerdit.core.app_build.AppImageBuilder`, reaching the controller
    through `self._c`.
    """

    def __init__(self, controller: "ServiceController") -> None:
        self._c = controller
        # Restart-required, like every `[services]` bound: snapshot at construction
        # so a config PUT only rewrites TOML until the daemon restarts.
        settings = controller._services_settings
        self._grace_s = settings.cutover_grace_s
        self._verify_timeout_s = settings.cutover_verify_timeout_s
        self._repoint_timeout_s = settings.cutover_repoint_timeout_s

    # --- registries ---------------------------------------------------------

    def _task_in_flight(self, job_id: str) -> bool:
        """Whether a verify task is still running for `job_id`.

        `not task.done()` rather than bare membership, for the reason
        `AppImageBuilder._task_in_flight` documents: a task cancelled
        before its first step never runs its `finally`, and bare membership
        would wedge the row forever.
        """
        task = self._c._cutover_tasks.get(job_id)
        return task is not None and not task.done()

    def has_active(self, job_id: str) -> bool:
        """Whether a cutover verify is in flight for `job_id` (DELETE guard)."""
        return self._task_in_flight(job_id)

    def has_unbound(self, job_id: str) -> bool:
        """An in-flight verify whose green has no container id yet.

        The `has_unbound_run` twin: everything between the
        task spawn and `runtime.run()` returning — env resolution, volume
        materialization, the container create itself — is a window in which a
        cancel cannot kill anything (there is no id) yet the docker thread may
        still start a container that bind-mounts the service data tree. The
        forced-delete data purge fails closed on it.
        """
        return self._task_in_flight(job_id) and job_id not in self._c._cutover_containers

    def container_ids(self) -> set[str]:
        """Green container ids, daemon-wide (zombie-sweep + drain hook)."""
        return {cid for cid in self._c._cutover_containers.values() if cid}

    def busy(self) -> int:
        """Number of verify tasks still running (restart-drain probe)."""
        return sum(1 for task in self._c._cutover_tasks.values() if not task.done())

    def _discard(self, job_id: str) -> None:
        """Drop this row's green from the registry (idempotent)."""
        self._c._cutover_containers.pop(job_id, None)

    # --- entry point --------------------------------------------------------

    async def maybe_cutover(self, job: Job, live: bool) -> bool:
        """Arm, wait for, or recover a health-gated swap.

        Check active tasks first, then persisted crash markers before applicability.
        Blue may die with the daemon; testing liveness first would launch an unverified
        candidate on the stable port. Eligible restarting live services arm a guarded
        marker and verify off-tick; other rows use the ordinary swap path.

        Returns:
            True if the caller must not touch containers this tick.
        """
        # --- layer 1: an in-flight verify task owns the row ---
        if self._task_in_flight(job.id):
            return True

        cfg = parse_job_config(job, warn=True)

        # --- layer 4: a cutover this daemon never finished ---
        pending = cfg.get("cutover_pending")
        # `is not None` + the version match is load-bearing: the key is absent
        # on the vast majority of rows, and a stale marker from an older
        # generation belongs to that generation's settle, not this one's.
        if isinstance(pending, dict) and pending.get("version") == cfg.get("build_version"):
            # ...but `job` is the tick's SNAPSHOT, and a verify that finished
            # between the fetch and this row's turn has already committed and
            # popped. Acting on the stale blob would run 4a against a
            # just-promoted generation, and the never-kill belt (which compares
            # the STALE `container_id` — blue) would happily SIGKILL the
            # promoted green. Re-read and re-check before dispatching; settle
            # on the FRESH row so both branches read a coherent state.
            fresh_row = await self._c._queries.get_job(job.id)
            if fresh_row is None:
                return True
            fresh_cfg = parse_job_config(fresh_row, warn=True)
            fresh_pending = fresh_cfg.get("cutover_pending")
            if not (
                isinstance(fresh_pending, dict)
                and fresh_pending.get("version") == fresh_cfg.get("build_version")
            ):
                return True  # settled under us — hands off, nothing to do
            await self._settle_crashed_cutover(fresh_row, fresh_pending)
            return True

        # --- layer 2: not applicable — today's path, byte-identical ---
        if not live or job.status is not JobStatus.restarting or not job.container_id:
            return False
        if not _eligible(job, cfg, self._c._services_settings, self._c._proxy):
            return False

        # --- layer 3: arm ---
        return await self._arm(job, cfg)

    async def _arm(self, job: Job, cfg: dict) -> bool:
        """Stamp `verifying`, write the marker, spawn the verify task."""
        version = cfg.get("build_version")
        assert isinstance(version, int)  # _eligible guarantees it

        row = await self._c._queries.get_job(job.id)
        if row is None:
            return False
        # A stop/delete that landed between the tick's row fetch and here must
        # not get a green launched behind it: the reconcile's own
        # `desired == "stopped"` branch already ran against the STALE row, so
        # nothing else in this tick will notice. Same fresh-row refusal
        # `run_once` makes for the identical reason. Checked BEFORE the phase
        # stamp so a refused arm cannot leave a stopped row displaying a
        # `verifying` phase no verify ever ran.
        if (row.desired_state or "running") in TERMINAL_STATUSES:
            return False

        # The ONLY site that stamps `verifying`: the launch inside
        # the verify task is provenance-only and must not re-stamp it.
        # Best-effort, NOT a gate: on a manual restart
        # the phase is typically `healthy`, the guard rejects the
        # write, and the cutover proceeds anyway — the marker write + its CAS is
        # the gate. The promotion/regression edges then no-op symmetrically.
        await stamp_last_deploy(
            self._c._queries,
            job.id,
            only_from=("queued", "building"),
            phase="verifying",
        )

        # Version-CAS'd keyed write, as `AppImageBuilder._arm_release_pending`
        # does: a redeploy committing between this read and the write mints a
        # new build_version and the marker write misses instead of arming a
        # generation that no longer owns the row.
        row = await self._c._queries.get_job(job.id)
        if row is None:
            return False
        fresh = parse_job_config(row, warn=True)
        if fresh.get("build_version") != version:
            return False  # a newer generation owns the row
        # Record blue's OWN serving port beside its id: blue may itself be a
        # cutover-promoted green on an ephemeral port (back-to-back cutovers),
        # and every unwind must restore THAT port, not clear to NULL: NULL means
        # "the stable port", where in that composition nothing listens, and a
        # degraded row never relaunches — a permanent 502.
        blue_port: int | None = None
        if job.service_name:
            ep = await self._c._queries.get_service_endpoint(job.service_name)
            blue_port = ep.active_host_port if ep is not None else None
        marker = {"version": version, "blue": job.container_id, "blue_port": blue_port}
        if not await self._c._queries.patch_job_config(
            job.id, {"cutover_pending": marker}, expect_build_version=version
        ):
            return False  # CAS miss — supersession, hands off

        await record_job_event(self._c._events, "service.cutover_started", job)
        task = asyncio.create_task(self._run_verify(job, cfg, version))
        self._c._cutover_tasks[job.id] = task
        task.add_done_callback(self._discard_task)
        return True

    def _discard_task(self, task: asyncio.Task) -> None:
        """Drop a finished verify task from the registry (the log-task idiom)."""
        for job_id, tracked in list(self._c._cutover_tasks.items()):
            if tracked is task:
                self._c._cutover_tasks.pop(job_id, None)
                break

    # --- the verify task ----------------------------------------------------

    async def _run_verify(self, job: Job, cfg: dict, version: int) -> None:
        """Outer catch-all around `_verify` — an escape here wedges the row.

        A bug that raised out of the verify would otherwise leave the marker
        armed, no task registered and nothing settled. It does not: the marker
        stays armed on purpose, so the NEXT tick's layer 4 settles the
        generation (4a if the row was never promoted, 4b if it was). The
        registry entry is dropped either way.
        """
        try:
            await self._verify(job, cfg, version)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Cutover verify failed unexpectedly for service %s; the persisted "
                "marker will settle it on the next tick",
                job.service_name or job.id,
            )
        finally:
            self._discard(job.id)

    async def _verify(self, job: Job, cfg: dict, version: int) -> None:
        """Launch the green, probe it, and run the step-(f) commit on success."""
        from nerdit.core.services import LaunchEnvNotReady

        name = job.service_name or ""
        container_port = int(cfg.get("port") or 8000)

        # (a) resolve env. RETRY-NEXT-TICK, not terminal (unlike a release
        # failure): a binding wait is indefinite
        # by design, so pop the marker, regress the phase and let the next tick
        # re-arm. Blue keeps serving throughout; nothing was launched.
        try:
            resolved = await self._c._resolve_launch_env(
                job, cfg, container_port, cfg.get("ai"), cfg.get("db")
            )
        except LaunchEnvNotReady as exc:
            await self._pop_marker(job.id, version)
            await stamp_last_deploy(
                self._c._queries, job.id, only_from=("verifying",), phase="building"
            )
            if exc.kind == "binding":
                await self._c._log_binding_wait(job, exc.message)
            else:
                await self._c._log_secret_wait(job, exc.message)
            return
        self._c._binding_wait_msgs.pop(job.id, None)
        if resolved.shared_keys:
            await self._c._audit_shared_resolved(job, resolved.shared_keys, resolved.secret_env)

        # (b) a transient OS-assigned loopback port.
        green_port = _reserve_ephemeral_port(await self._c._queries.get_reserved_service_ports())

        # (c) the green container config: the service's own shape on the green
        # port, the SAME named volumes blue is writing (a new generation must
        # see the real data), plus the reap label.
        command, volumes, workdir = self._c._build_command(job, cfg)
        if volumes and not await self._check_mounts(job, volumes, version):
            return
        try:
            named_volumes = self._named_volumes(job, cfg)
        except VolumeSpecError as exc:
            await self._settle_probe_failure(job, version, message=str(exc))
            return
        # `_launch`'s companion check, mirrored (D-P14-4): a
        # named volume nesting with a Tier-B user mount shadows one of them.
        # The green becomes the live container on success, so `_launch`
        # never re-checks it — skipping it here would make a cutover the one
        # path that lands a shadowed mount. Blue is untouched, so this is an
        # ordinary probe-stage failure, settled before anything is launched.
        conflict = self._c._volume_path_conflict(named_volumes.values(), (volumes or {}).values())
        if conflict is not None:
            await self._settle_probe_failure(job, version, message=conflict)
            return
        if named_volumes and job.service_name:
            self._c._ensure_volume_dirs(job.service_name, named_volumes)
        config = build_app_container_config(
            cfg,
            self._c._container_settings,
            AppContainerSpec(
                command=command,
                volumes=volumes,
                env=resolved.env,
                workdir=workdir,
                # Eligible rows are CPU-only (D-P24-5 rule 2): two live
                # containers cannot both hold a GPU allocation.
                gpu_ids=[],
                vendor=GpuVendor.nvidia,
                container_port=container_port,
                host_port=green_port,
            ),
        )
        config.extra_labels = {**(config.extra_labels or {}), _CUTOVER_LABEL: job.id}
        finalize_container_config(config, named_volumes, self._c._retention_settings)
        # PROVENANCE ONLY — `phase=None` skips the phase stamp entirely (the
        # phase already IS `verifying`, stamped by the layer-3 arm, and a
        # re-stamp would be rejected by its own guard while reading as
        # protection that does not exist).
        await stamp_launching(self._c, job, config, named_volumes, phase=None)

        # (d) run it.
        try:
            green_id = await self._c._runtime.run(config)
        except ContainerRuntimeError as exc:
            await self._settle_probe_failure(job, version, message=str(exc))
            return
        self._c._cutover_containers[job.id] = green_id

        # (e) probe.
        if not await self._probe_green(job, green_id, green_port):
            await self._settle_probe_failure(job, version, green_id=green_id)
            return

        # (f) success.
        await self._commit(job, version, green_id=green_id, green_port=green_port, name=name)

    async def _check_mounts(self, job: Job, volumes: dict[str, str], version: int) -> bool:
        """Re-run `_launch`'s Tier-B mount allowlist for the green container.

        The green becomes the live container on success, so `_launch` never
        re-checks it: skipping the check here would make a cutover the one path
        that lands an out-of-allowlist host mount for a non-admin token.
        """
        role = await resolve_role(self._c._queries, job)
        if role == TokenRole.admin:
            return True
        try:
            enforce_mount_allowlist(volumes, self._c._container_settings.allowed_mount_roots)
        except SandboxViolationError as exc:
            await self._settle_probe_failure(job, version, message=str(exc))
            return False
        return True

    def _named_volumes(self, job: Job, cfg: dict) -> dict[str, str]:
        """The daemon-computed named volumes — the SAME set blue has mounted.

        Resolution only: materializing the leaf dirs is the caller's, after the
        Tier-B conflict check has passed (`_launch`'s order).
        """
        if self._c._data_dir is None or not job.service_name or not cfg.get("volumes"):
            return {}
        return service_volumes(self._c._data_dir, job.service_name, cfg)

    async def _probe_green(self, job: Job, green_id: str, green_port: int) -> bool:
        """Verify the green within `[services].cutover_verify_timeout_s`.

        A health spec drives `core/health.py`'s `run_probe` (first 2xx
        wins); without one, the green must simply stay `running` for
        `[services].cutover_grace_s`. **A container that exits fails
        immediately** — never wait out the budget for a dead candidate.

        A declared `start_period_s` WIDENS the budget rather than being
        ignored: the app has told the platform it needs that long before its
        first probe can pass, and the ordinary launch path honours it
        (`core/health.py::within_start_period` suppresses the degraded flip).
        A cutover that timed out inside a declared warm-up would settle a
        perfectly good generation `cutover_failed` on every deploy.
        """
        hc = job.health_check
        timeout = as_float((hc or {}).get("timeout_s"), DEFAULT_HEALTH_TIMEOUT_S)
        budget = float(self._verify_timeout_s)
        if hc:
            start_period = as_float(hc.get("start_period_s"), 0.0)
            if start_period > 0:
                budget = max(budget, start_period + self._grace_s)
        else:
            # The same widening for the no-health-check
            # fallback: nothing schema-validates grace <= verify budget, and a
            # `cutover_grace_s` above `cutover_verify_timeout_s` would
            # otherwise make every healthy no-spec redeploy deterministically
            # settle `cutover_failed` — it cannot pass before the grace
            # elapses and cannot outlive the budget.
            budget = max(budget, self._grace_s + 1.0)
        started = _MONOTONIC()
        deadline = started + budget
        while True:
            if not await self._c._is_container_live(green_id):
                logger.warning(
                    "Cutover candidate for service %s exited during verification",
                    job.service_name or job.id,
                )
                return False
            if hc:
                code, _kind = await run_probe(
                    hc,
                    green_port,
                    timeout,
                    http_check=self._c._check_health,
                    tcp_check=self._c._check_tcp,
                )
                if code is not None and 200 <= code < 300:
                    return True
            elif (_MONOTONIC() - started) >= self._grace_s:
                return True
            if _MONOTONIC() >= deadline:
                return False
            await _SLEEP(_POLL_INTERVAL_S)

    # --- (f) the commit -----------------------------------------------------

    async def _commit(
        self, job: Job, version: int, *, green_id: str, green_port: int, name: str
    ) -> None:
        """Step (f) in its locked order — see the module docstring."""
        queries = self._c._queries
        # f1 — the DB is authoritative FIRST, so a proxy reconcile firing in the
        # gap converges the dial TO the green rather than reverting it to blue.
        await queries.set_endpoint_active_port(name, green_port)

        # f2 — VERIFIED repoint. The inline register is a latency fast-path; the
        # read-back is the only proof.
        proxy = self._c._proxy
        assert proxy is not None  # _eligible required an available proxy
        # The ids only — `_route_id` / `_domain_route_id`, not
        # `build_route`: the latter resolves secrets and can raise,
        # and this call needs nothing but the ids. The domain rows are read
        # here, at commit time, so a domain bound during the verify window is
        # included and one removed during it is not.
        domain_rows = await queries.get_service_domains(name)
        caddy_ids = (_route_id(name), *(_domain_route_id(name, r.domain) for r in domain_rows))
        # The repoint REBUILDS the whole route object,
        # so the row's `edge_auth` must ride along or every cutover commit of a
        # protected service would strip its auth handler — and `_await_dial`
        # compares the DIAL ONLY, so the strip would still "verify". ACCEPTED
        # corollary: a secret that becomes unresolvable mid-cutover makes
        # `register` refuse (fail closed, D-P25-8), the dial never converges,
        # and the cutover settles `repoint_failure` with blue restored — the
        # route is then withheld until the secret resolves. Posture beats
        # availability.
        await proxy.register(
            name,
            green_port,
            edge_auth=parse_job_config(job).get("edge_auth"),
            project_id=job.project_id,
            with_domains=True,
            # The SAME snapshot `caddy_ids` was derived from, never a second
            # read: a `DELETE` landing between two reads would leave a removed
            # id in the awaited set that nothing ever writes, and
            # `_await_dials` would burn its whole budget and unwind a healthy
            # green. The removed route is torn down by the next reconcile tick.
            domains=[r.domain for r in domain_rows],
        )
        if not await self._await_dials(caddy_ids, green_port):
            await self._settle_repoint_failure(
                job, version, green_id=green_id, name=name, caddy_ids=caddy_ids
            )
            return

        # f2b — the residual window. `_await_dial` can take up to
        # `cutover_repoint_timeout_s`, and the green is NOT probed during it;
        # a drain-deadline kill or a crash-loop exit in that gap would promote
        # a dead container onto the row and then destroy blue at f7, leaving
        # the route dialling nothing. Re-check liveness immediately before the
        # commit point and divert to the (g') unwind — the pointer already
        # names the green, so g' is the settle that puts it back.
        if not await self._c._is_container_live(green_id):
            logger.warning(
                "Cutover candidate for service %s died between the dial verify and the "
                "promotion; unwinding",
                job.service_name or job.id,
            )
            await self._settle_repoint_failure(
                job,
                version,
                green_id=green_id,
                name=name,
                caddy_ids=caddy_ids,
                message=(
                    "the new version exited after its health verification and before it "
                    "could be promoted; the previous version is still serving. Data "
                    "changes are NOT rolled back."
                ),
            )
            return

        # f3 — THE COMMIT POINT.
        await queries.update_job_status(
            job.id,
            JobStatus.running,
            container_id=green_id,
            started_at=datetime.now(UTC),
        )
        self._c._emit_status_change(job.id, JobStatus.running)

        # f4-f6 — the commit tail, replayed verbatim by layer 4b after a crash.
        await self._complete_commit_tail(job, {"version": version})

        # f7 — destroy BLUE (+ cancel its log task); adopt the green's stream.
        if job.container_id:
            await self._c._destroy_container(job.container_id)
        self._c._spawn_log_task(job.id, green_id)
        await self._c._append_log_tolerant(
            job.id,
            f"Cutover complete: new container {green_id[:12]} is serving",
            LogStream.system,
        )

        # f8 — keep-last-3.
        await self._c._prune_old_images(job)
        logger.info(
            "Cutover for service %s version %s promoted container %s",
            job.service_name or job.id,
            version,
            green_id[:12],
        )

    async def _complete_commit_tail(self, job: Job, pending: dict) -> None:
        """f4-f6: durable records FIRST, `healthy` stamp, then pop the marker.

        The ONE implementation of the tail: the online path calls it at f4 and
        layer 4b replays it verbatim after a crash, so the two can never drift.
        Because it is a replay, the audit row and the event are
        **at-least-once** across a crash — a duplicated record of a transition
        that really happened is strictly better than the pop-first ordering's
        silent loss of it.
        """
        version = pending.get("version")
        target = version if isinstance(version, int) and not isinstance(version, bool) else None
        # f4 — durable records BEFORE the pop.
        await self._audit(
            "service.cutover", job, _cutover_audit_params(job, target, data_rollback=False)
        )
        await record_job_event(self._c._events, "service.cutover_succeeded", job)
        # f5 — the promotion edge (a no-op when the crash landed after it,
        # which is correct).
        await stamp_last_deploy(self._c._queries, job.id, only_from=("verifying",), phase="healthy")
        # f6 — LAST: this is what disarms layer 4.
        await self._pop_marker(job.id, target)

    async def _await_dials(self, caddy_ids: Sequence[str], port: int) -> bool:
        """Read the live dials back until **every** id names `port` (or time is up).

        Asserts the **end state**, not the writer: a dial converged by a proxy
        reconcile tick passes exactly the same as one written by our inline
        `register`. An unreadable live set (`None`) is deliberately NOT
        success — it is the admin wrapper's "could not read" signal, distinct
        from `{}`.

        (D-P26-2) All-or-nothing across ONE live read: a partially converged
        set is a failure, not a partial success, and the ids must agree on the
        same snapshot rather than each being satisfied on a different poll —
        otherwise a route that flipped back between two reads would still count.
        One deadline covers the whole set (`cutover_repoint_timeout_s`), not
        one per id.
        """
        proxy = self._c._proxy
        assert proxy is not None
        expected = f"127.0.0.1:{port}"
        deadline = _MONOTONIC() + self._repoint_timeout_s
        while True:
            live = None
            try:
                live = await proxy.live_routes()
            except Exception:  # noqa: BLE001 — an unreadable admin API is a "no"
                logger.debug("[cutover] live_routes read failed", exc_info=True)
            if live is not None:
                current = [live.get(rid) for rid in caddy_ids]
                if all(route is not None and route.dial == expected for route in current):
                    return True
            if _MONOTONIC() >= deadline:
                return False
            await _SLEEP(_POLL_INTERVAL_S)

    # --- (g) / (g') failure settles ----------------------------------------

    async def _settle_probe_failure(
        self,
        job: Job,
        version: int,
        *,
        green_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """(g) The green never verified healthy. Blue, the route and the pointer
        are all untouched — `active_host_port` was never written."""
        if green_id is not None:
            await self._destroy_green(green_id)
        self._discard(job.id)
        await self._settle_failed(job, version, stage=_STAGE_PROBE, message=message)

    async def _settle_repoint_failure(
        self,
        job: Job,
        version: int,
        *,
        green_id: str,
        name: str,
        caddy_ids: Sequence[str],
        message: str | None = None,
    ) -> None:
        """(g') The green IS healthy but the live dial could not be confirmed.

        f1 already wrote `active_host_port`, so the unwind has one extra step
        and a strict order: the DB goes back to BLUE'S OWN serving port FIRST
        (the marker's `blue_port` — the stable port unless blue itself is a
        cutover-promoted green on an ephemeral port), then a bounded re-verify (proceed anyway if
        unreadable — blue is listening there and the proxy reconcile converges
        on its own), and only then is the green destroyed. The marker stays
        armed until this completes, so a crash during the unwind is caught by
        layer 4a.
        """
        queries = self._c._queries
        endpoint = await queries.get_service_endpoint(name)
        blue_port = await self._marker_blue_port(job.id, version)
        # g'1 — DB back to blue's port, FIRST, always.
        await queries.set_endpoint_active_port(name, blue_port)
        # g'2 — bounded re-verify; unreadable or timed out => proceed anyway.
        if endpoint is not None:
            await self._await_dials(caddy_ids, blue_port or endpoint.host_port)
        # g'3 — only now.
        await self._destroy_green(green_id)
        self._discard(job.id)
        # g'4 + g'5 + g'6 — the settle pops the marker as its LAST step (see
        # _settle_failed): the marker stays armed until the failure is durably
        # settled, so a crash anywhere in this unwind resumes into layer 4a.
        await self._settle_failed(
            job,
            version,
            stage=_STAGE_REPOINT,
            message=message
            or (
                "the new version was healthy but the proxy dial could not be confirmed; "
                "the previous version is still serving. Data changes are NOT rolled back."
            ),
        )

    async def _settle_failed(
        self,
        job: Job,
        version: int,
        *,
        stage: str | None,
        message: str | None = None,
        error_class: ErrorClass = ErrorClass.user_error,
    ) -> None:
        """Settle failure and its event before removing the cutover marker.

        Probe failures are user errors; unobserved crash outcomes use unknown. Keeping
        the marker through settlement makes failures/crashes replayable without an
        automatic verification rerun. Pop by marker version so a reverted row clears
        its old marker without disarming a newer generation.
        """
        text = message or _FAILED_MESSAGE
        await self._c._builder._settle_failed_generation(
            job,
            reason="cutover_failed",
            message=text,
            error_class=error_class,
            target_version=version,
            audit_action="service.cutover_failed",
            audit_params=_cutover_audit_params(job, version, data_rollback=False, stage=stage),
        )
        await self._pop_marker(job.id, version)
        await record_job_event(
            self._c._events,
            "service.cutover_failed",
            job,
            reason="cutover_failed",
            data=None if stage is None else {"stage": stage},
        )

    async def _destroy_green(self, green_id: str) -> None:
        """Tear the green down. It has no log task (f7 is what spawns one)."""
        await self._c._destroy_container(green_id)

    # --- layer 4: the crash settles ----------------------------------------

    async def _settle_crashed_cutover(self, job: Job, pending: dict) -> None:
        """Settle a cutover whose daemon died mid-flight (D-P24-4 layer 4).

        Which SIDE of the commit point it died on is read off the marker's
        `blue` field, and the two branches do opposite things. The
        discriminator is **total** because the marker's `blue` never changes
        once armed: at every instant the row's `container_id` either equals it
        (not committed ⇒ 4a) or does not (committed ⇒ 4b).
        """
        raw = pending.get("version")
        version = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
        blue = pending.get("blue")

        if job.container_id is not None and job.container_id != blue:
            # 4b — already committed: the green passed its probe, its dial was
            # verified, AND the row was written. Reverting would take down a
            # healthy live container and dial a destroyed blue. Replay the tail
            # instead; kill NOTHING (the orphaned blue is the zombie sweep's).
            await self._complete_commit_tail(job, pending)
            return

        # 4a — not committed (the mid-verify case), in this exact order.
        # 1. Restore `active_host_port` to BLUE'S OWN port — MANDATORY, and
        #    the whole reason this branch is ordered. A crash in the f1->f3
        #    window (or an abandoned g' unwind) leaves the column naming the
        #    green this branch is about to kill; without the restore the
        #    `COALESCE` keeps dialling the dead green forever, the surviving
        #    blue is probed on a dead port, flips `degraded` (which stays
        #    running and is never relaunched, so `settle_started`'s clear is
        #    never reached) and the public URL 502s **permanently**. The target
        #    is the marker's `blue_port`, not NULL: blue may itself be a
        #    promoted green on an ephemeral port (back-to-back cutovers), and
        #    NULL would dial the stable port where nothing listens — the same
        #    permanent 502 from the other direction. `blue_port` absent (an
        #    older marker) degrades to NULL. Idempotent.
        if job.service_name:
            await self._c._queries.set_endpoint_active_port(
                job.service_name, _as_port(pending.get("blue_port"))
            )
        # 2. Reap the labelled green.
        await self._reap_green(job)
        # 3. Settle — which pops the marker as its LAST step, so a crash
        #    anywhere above re-enters this branch idempotently. A marker with
        #    no usable version cannot be settled; blind-pop it instead.
        if version is None:
            await self._pop_marker(job.id, None)
        else:
            await self._settle_failed(
                job,
                version,
                stage=None,
                message=_CRASH_MESSAGE,
                error_class=ErrorClass.unknown,
            )

    async def _reap_green(self, job: Job) -> None:
        """Kill any container labelled `nerdit-cutover=<job.id>` (best-effort).

        The `_settle_crashed_release` reap, minus its abort-on-active-run belt
        (the label is disjoint from `nerdit-run`/`nerdit-job`, so a run
        container can never appear in this listing) and **plus one extra belt:
        never kill a container id equal to `job.container_id`**, so a
        mislabelled or already-promoted container can never be SIGKILLed out
        from under a live row.
        """
        try:
            orphans = await self._c._runtime.list_own_labeled_containers(_CUTOVER_LABEL, job.id)
            for cid in orphans:
                if cid == job.container_id:
                    logger.warning(
                        "Skipping the cutover reap of %s for service %s: it is the row's "
                        "own live container",
                        cid[:12],
                        job.service_name or job.id,
                    )
                    continue
                try:
                    await self._c._runtime.kill(cid)
                    await self._c._runtime.remove(cid, force=True)
                    logger.warning(
                        "Reaped crash-orphaned cutover container %s of service %s",
                        cid,
                        job.service_name or job.id,
                    )
                except ContainerNotFoundError:
                    continue
                except ContainerRuntimeError:
                    logger.warning(
                        "Could not reap orphaned cutover container %s", cid, exc_info=True
                    )
        except Exception:  # noqa: BLE001 — the settle must never fail on the reap
            logger.warning(
                "Cutover orphan reap failed for service %s; the zombie sweep will collect it",
                job.service_name or job.id,
                exc_info=True,
            )

    # --- cancellation (DELETE ?force) --------------------------------------

    async def cancel(self, job_id: str) -> set[str]:
        """Cancel verification without destroying a committed replacement.

        Before commit, destroy green, restore blue's port, and clear the marker without
        stamping failure: an operator cancellation is not a health verdict. After
        commit, preserve green and its route, completing any still-marked audit/event
        tail. Force-delete, stop, and drain share this behavior.

        Returns:
            Destroyed candidate IDs, empty once green is promoted.
        """
        # Snapshot the green BEFORE cancelling: the verify task's `finally`
        # drops its own registry entry, so reading after the await would find
        # nothing and leave the container alive. Re-read afterwards too, in case
        # the task bound one between the snapshot and the cancel.
        greens = {cid for cid in (self._c._cutover_containers.get(job_id),) if cid}
        task = self._c._cutover_tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        late = self._c._cutover_containers.pop(job_id, None)
        if late:
            greens.add(late)

        # The row is read BEFORE anything is destroyed: the task is already dead
        # here (cancelled and awaited), so no commit can land after this read,
        # and a green that reached f3 is now the row's own container — killing
        # it first would take the live service down with it.
        row = await self._c._queries.get_job(job_id)
        row_cid = row.container_id if row is not None else None
        destroyed: set[str] = set()
        for green_id in greens:
            if green_id == row_cid:
                logger.warning(
                    "Skipping the cutover-cancel destroy of %s for service %s: it is the "
                    "row's own live container",
                    green_id[:12],
                    (row.service_name if row else None) or job_id,
                )
                continue
            with contextlib.suppress(Exception):
                await self._destroy_green(green_id)
            destroyed.add(green_id)
        if row is None:
            return destroyed

        cfg = parse_job_config(row, warn=True)
        pending = cfg.get("cutover_pending")
        if row_cid is not None and row_cid in greens:
            # Post-commit. A marker still armed for THIS generation means the
            # task died inside the f4-f6 tail: finish it, so the durable records
            # land at-least-once (the layer-4b contract, replayed verbatim —
            # kill NOTHING, an f3-f6 blue is the zombie sweep's exactly as 4b
            # leaves it). The version + `blue` pair mirrors layer 4's dispatch
            # and 4b's discriminator, so a marker armed by a NEWER generation is
            # never replayed and the failure mode is hands-off. Marker already
            # popped => the cutover is fully recorded: touch nothing.
            if (
                isinstance(pending, dict)
                and pending.get("version") == cfg.get("build_version")
                and pending.get("blue") != row_cid
            ):
                await self._complete_commit_tail(row, pending)
            return destroyed

        if isinstance(pending, dict):
            if row.service_name:
                # Back to blue's OWN port (see the 4a comment): NULL only when
                # blue was on the stable port or the marker predates
                # `blue_port`.
                await self._c._queries.set_endpoint_active_port(
                    row.service_name, _as_port(pending.get("blue_port"))
                )
            raw = pending.get("version")
            version = raw if isinstance(raw, int) and not isinstance(raw, bool) else None
            await self._pop_marker(job_id, version)
        # No marker => nothing owned to unwind: the cutover
        # either never armed or ALREADY COMMITTED — and on a committed row the
        # pointer names the promoted green's live port, so "restoring" NULL
        # here would redirect the dial to the unused stable port and 502 the
        # service. Reachable via the drain's sequential cancels (a later
        # verify commits while an earlier cancel is awaited) and the stop
        # branch's check-to-cancel gap. Touch nothing.
        return destroyed

    # --- shared helpers -----------------------------------------------------

    async def _marker_blue_port(self, job_id: str, version: int | None) -> int | None:
        """Read the armed marker's `blue_port` — the unwind restore target.

        `None` when the row/marker is gone, names another generation, or
        predates the field: NULL = "the stable port".
        """
        row = await self._c._queries.get_job(job_id)
        if row is None:
            return None
        pending = parse_job_config(row, warn=True).get("cutover_pending")
        if not isinstance(pending, dict):
            return None
        if version is not None and pending.get("version") != version:
            return None
        return _as_port(pending.get("blue_port"))

    async def _pop_marker(self, job_id: str, version: int | None) -> None:
        """Disarm `cutover_pending` for `version` (version-CAS'd RMW).

        Guarded rather than a blind pop, for `_clear_release_pending`'s exact
        reason: if a redeploy landed while the verify ran, the marker on the row
        now names ITS generation, and clearing it would silently disarm
        crash-safety for a cutover that has not run yet.
        """
        row = await self._c._queries.get_job(job_id)
        if row is None:
            return
        cfg = parse_job_config(row, warn=True)
        pending = cfg.get("cutover_pending")
        if not isinstance(pending, dict):
            return
        # The identity guard keys on the MARKER's version, not the row's
        # current build_version: after the settle's revert branch the row is
        # back on the previous generation while the marker still names the
        # failed one — that marker is settled and must pop. What must never
        # pop is a marker naming a DIFFERENT version than the caller's: that
        # one belongs to a newer generation.
        if version is not None and pending.get("version") != version:
            return
        current = cfg.get("build_version")
        # CAS on the row's CURRENT version: a redeploy landing between the
        # read and this write mints a new build_version, misses the CAS, and
        # its own write_redeploy pop owns the marker instead.
        await self._c._queries.patch_job_config(
            job_id,
            remove=["cutover_pending"],
            expect_build_version=(
                current if isinstance(current, int) and not isinstance(current, bool) else None
            ),
        )

    async def _audit(self, action: str, job: Job, params: dict[str, object]) -> None:
        """Write one `principal='system'` audit row (the `_audit_settle` shape).

        A verify task cannot use the request-bound `record_out_of_band`, and
        the controller's `_audit` helper takes no params. Best-effort: an
        audit failure never fails a settle.
        """
        try:
            await self._c._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=job.id,
                params_redacted=json.dumps(params),
            )
        except Exception:
            logger.warning("Failed to audit %s for service %s", action, job.id, exc_info=True)


__all__: list[str] = ["CutoverManager", "_eligible", "cutover_skip_reason"]
