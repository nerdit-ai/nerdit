"""FastAPI daemon server for Nerdit."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import socket
import stat
import sys
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from nerdit import __version__
from nerdit.config.defaults import DEFAULT_DB_NAME, DEFAULT_PROXY_RECONCILE_INTERVAL
from nerdit.config.settings import (
    DaemonSettings,
    NerditSettings,
    ProxySettings,
    load_settings,
)
from nerdit.config.store import ConfigStore
from nerdit.core.backup import sweep_staging_orphans
from nerdit.core.cutover import cutover_skip_reason
from nerdit.core.dns import MdnsAdvertiser
from nerdit.core.eventlog import set_recorder
from nerdit.core.events import EventBus
from nerdit.core.gitsource import git_available
from nerdit.core.gitwatch import GitWatchController
from nerdit.core.license import FEATURE_REMOTE_LINK
from nerdit.core.monitor import ResourceMonitor
from nerdit.core.notify import WebhookDispatcher
from nerdit.core.proxy.domains import host_aliases, reserved_names
from nerdit.core.runtime.docker import DockerRuntime
from nerdit.core.runtime.protocol import ContainerRuntime
from nerdit.core.runtime.stub import StubRuntime
from nerdit.core.sweeper import ZombieSweeper
from nerdit.core.volumes import dump_staging_root

# Track B WP19.b moved the app-assembly body (router imports, middleware
# stack, tag table, legacy aliases, MCP mount, dashboard mount) to
# `daemon/appfactory.py`; `create_app()` below stays a thin wrapper.
# `DashboardHtmlMiddleware`/`_McpMount` are re-exported here (rather than
# imported directly) so `test_create_app.py`/`test_mcp_http.py` — which
# read/import them off this module — survive the split unmodified.
from nerdit.daemon.appfactory import (  # noqa: F401
    DashboardHtmlMiddleware,
    _McpMount,
    build_app,
)
from nerdit.daemon.bootstrap import (
    build_controllers,
    build_edge_auth_resolver,
    build_license_state,
    build_link_manager,
    build_proxy_manager,
    build_secret_manager,
    discover_gpus,
    normalize_mount_roots,
    resolve_dashboard_upstream,
    resolve_hostname,
)
from nerdit.daemon.deploy_pipeline import redeploy_from_source
from nerdit.daemon.service_purge import _sweep_data_tombstones

# Re-exported for the retention/backup suites (Track B WP19.a moved the sweep
# family to daemon/sweeps.py; the four loop functions are also lifespan's real
# call sites, but `_prune_backups`/`_prune_volume_backups`/
# `_run_retention_sweep` exist ONLY as bindings here, read by
# tests/test_backup_volumes.py:32 and tests/test_retention.py (direct
# `server_module.<name>` calls, no monkeypatch involved).
from nerdit.daemon.sweeps import (  # noqa: F401
    _idempotency_sweep_loop,
    _proxy_reconcile_loop,
    _prune_backups,
    _prune_dump_backups,
    _prune_volume_backups,
    _retention_sweep_loop,
    _run_retention_sweep,
    _zombie_sweep_loop,
    retention_sweep_enabled,
)
from nerdit.db.database import Database
from nerdit.db.models import GpuVendor, Job
from nerdit.db.queries import Queries
from nerdit.utils.certs import ca_fingerprint

# The archive-dir guard now lives in `utils.disk` so `daemon/routes/system.py`
# can walk the SAME dir the sweep writes to (F14) without importing this module
# (a route importing `daemon.server` would be a cycle). The alias keeps the old
# private name bound in this namespace: WP19.a moved its call site to
# `daemon/sweeps.py` (which imports the real name directly), so this binding
# is now a pure re-export, reached as `server_module._resolve_archive_dir` by
# tests/test_retention.py:437/:446/:457.
from nerdit.utils.disk import resolve_archive_dir as _resolve_archive_dir  # noqa: F401
from nerdit.utils.frozen import restore_host_loader_env
from nerdit.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

ZOMBIE_SWEEP_INTERVAL_SECONDS = 300

# Idempotency keys are retained ~24h (see `IDEMPOTENCY_TTL_HOURS`); the daemon
# sweeps expired rows hourly so the table stays bounded.
IDEMPOTENCY_SWEEP_INTERVAL_SECONDS = 3600

# Hard bound on uvicorn's graceful shutdown. Without it uvicorn waits
# for in-flight requests INDEFINITELY, which turned the `POST /daemon/restart`
# drain budget into a lie: a live `POST /services/{ident}/run` held the
# shutdown open for up to `[services].run_timeout_max_s` (30 min by default)
# after the drain had already given up. The drain now kills run/release
# containers at its own deadline, so those requests complete promptly; this
# constant only bounds whatever straggler connection is left after
# `should_exit` (notably a run slot claimed but not yet bound to a container,
# which has nothing to kill). It equally bounds a plain-SIGTERM shutdown with a
# long request in flight — deliberate. A constant, not a config key.
#
# The bound is a CANCEL. Since P22 that cancellation is handled honestly:
# `IdempotencyMiddleware` catches `BaseException` and either releases the
# straggler's claim (nothing committed) or pins it `interrupted` (a write
# committed — replays answer `409 idempotency_interrupted` instead of
# silently re-executing). Live-validated: SIGTERM mid-run → restart → the
# retry sees `interrupted`, persisted across the reboot.
GRACEFUL_SHUTDOWN_TIMEOUT_S = 30


def _truncate_if_regular_file(fd: int) -> None:
    """Truncate `fd` to 0 iff it refers to a regular file (P14b WP-B2c).

    Bounds `nerditd.boot.log` across `POST /daemon/restart` self-execs: the
    inherited stderr fd survives every `os.execv` with its offset, so the
    CLI-managed `open(..., "w")` truncate only fires on a fresh start. A
    foreground tty / pipe run reports a non-regular fstat and is left untouched.
    Best-effort — any `OSError` is swallowed.
    """
    try:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.ftruncate(fd, 0)
            # ftruncate does not move the file offset; the execv-inherited fd
            # still points at the old end, so the next write would re-extend
            # the file with a NUL hole and st_size would keep growing across
            # restarts. Rewind so the bound actually holds.
            os.lseek(fd, 0, os.SEEK_SET)
    except OSError:
        pass


def _acquire_restore_lock(data_dir: Path) -> int:
    """Hold a shared restore lock until daemon shutdown and return its file descriptor.

    First acquire LOCK_EX|LOCK_NB to refuse boot during offline restore, then
    downgrade to LOCK_SH. Restore cannot run against any live daemon, even without
    a pidfile. Process exit releases the lock; file existence alone never blocks.
    """
    fd = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            "A `nerdit restore` is in progress on this data dir; the daemon "
            "refuses to start until it completes. If a restore crashed, the lock "
            "is already released — retry."
        ) from exc
    fcntl.flock(fd, fcntl.LOCK_SH)
    return fd


def _warn_if_unauthenticated_exposure(
    daemon: DaemonSettings, proxy: ProxySettings | None = None
) -> None:
    """Log a warning when the daemon is exposed to the network without auth.

    The middleware silently bypasses auth when `auth_token` is `None`
    (v0.1 compat). Two ways that becomes a network-reachable admin surface:

    * The daemon is bound to a non-loopback host — every endpoint is directly
      reachable without authentication.
    * **The dashboard apex re-exposes the daemon.** With
      `[proxy].dashboard_apex` on, the catch-all apex forwards `/api/*` from
      Caddy's LAN-facing HTTPS listener to the daemon (`127.0.0.1:<port>`)
      *even when the daemon itself is bound to loopback* — so the host check
      above misses it. Warn on this combination regardless of `daemon.host`.
    """
    if daemon.auth_token is None and daemon.host not in _LOOPBACK_HOSTS:
        logger.warning(
            "Daemon is bound to %s with no auth_token configured. "
            "All endpoints are reachable without authentication. "
            "Set auth_token in ~/.nerdit/config.toml or bind to 127.0.0.1.",
            daemon.host,
        )
    apex_on = bool(
        proxy is not None and proxy.enabled and proxy.dashboard_apex and proxy.mode == "path"
    )
    if daemon.auth_token is None and apex_on:
        logger.warning(
            "[proxy] dashboard_apex is enabled with no auth_token configured: "
            "the apex route forwards the daemon API (/api/*) from the proxy's "
            "LAN-facing :%d HTTPS listener to the daemon, so every endpoint is "
            "reachable without authentication over the network — even though the "
            "daemon is bound to %s. Set auth_token in ~/.nerdit/config.toml or "
            "disable [proxy].dashboard_apex.",
            proxy.https_port,  # type: ignore[union-attr]  # apex_on ⇒ proxy is not None
            daemon.host,
        )


def _github_token_for_repo(app: FastAPI, slug: str) -> str | None:
    """GitWatch's `${github.installation}` resolver: the live
    installation token for `owner/name` from the link manager's mirror, or
    `None` when no manager is wired or no installation covers the repo.
    """
    manager = getattr(app.state, "link_manager", None)
    if manager is None:
        return None
    token = manager.github_token_for_repo(slug)
    return token if isinstance(token, str) and token else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage daemon startup and shutdown; constructors live in daemon/bootstrap.py."""
    settings = load_settings()

    # Resolve the data dir BEFORE logging so `setup_logging` can target
    # the rotating `<data_dir>/nerditd.log`.
    data_dir = Path(settings.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(data_dir / DEFAULT_DB_NAME)

    # Bound `nerditd.boot.log` across restart self-execs (see helper),
    # then tee root logs to the rotating file. `daemon_log_max_bytes=0` ⇒ no
    # file handler (daemon logs stay on stderr → boot.log only). Gated on the
    # lifecycle spawn marker: stderr may be a regular file in foreign embeddings
    # too (pytest capture, `2>>file` runs) and must never be truncated there.
    if os.environ.get("NERDIT_BOOT_LOG"):
        _truncate_if_regular_file(2)
    _retention = getattr(settings, "retention", None)
    setup_logging(
        settings.log_level,
        log_file=str(data_dir / "nerditd.log"),
        max_bytes=_retention.daemon_log_max_bytes if _retention is not None else 0,
        backup_count=_retention.daemon_log_backups if _retention is not None else 0,
    )

    # Take and hold a shared flock on `.restore.lock` for the
    # daemon's whole life, BEFORE any DB open. An offline `nerdit restore`
    # holds this exclusively; refusing to boot here (rather than a boot-instant
    # probe) means the daemon never opens the WAL while a restore is swapping
    # files underneath it — the pid/health heuristics become UX, not the safety
    # boundary. Released at lifespan shutdown.
    app.state.restore_lock_fd = _acquire_restore_lock(data_dir)

    # Upload directory
    upload_dir = Path(settings.daemon.upload_dir).expanduser()
    upload_dir.mkdir(parents=True, exist_ok=True)

    db = Database(db_path)
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    # bootstrap.py: GPU discovery/feedback + mount-root normalization.
    gpu_backend, gpus, gpu_count = await discover_gpus(settings, queries)
    normalize_mount_roots(settings)
    # Container runtime. Only client construction may fall back to the stub:
    # readiness checks must never downgrade a working Docker to StubRuntime.
    # `docker_runtime` is the narrowed handle used for the Docker-only
    # readiness checks below; `runtime` is the Protocol-typed one handed to
    # every consumer (and may be the stub).
    runtime: ContainerRuntime
    docker_runtime: DockerRuntime | None = None
    try:
        docker_runtime = DockerRuntime(
            denied_mount_paths=settings.containers.denied_mount_paths,
            allowed_mount_roots=settings.containers.allowed_mount_roots,
            # The Ollama weights cache lives under `<data_dir>/models`
            # (i.e. under the denied `~/.nerdit`). Exempt that daemon-owned
            # dir from Tier-A so `nerdit serve <model>` launches — but keep it
            # OUT of the Tier-B allowlist above so scoped tokens still cannot
            # bind-mount the shared weights cache.
            # (P14 WP-A1) <data_dir>/services holds the per-service named-volume
            # leaf dirs (daemon-computed host paths, never user-supplied). Carve
            # them out of Tier-A too — a Tier-A carve-out ONLY, kept out of the
            # Tier-B allowlist so a scoped token still cannot bind another
            # service's data dir via a user mount.
            # (P37 D-P37-2) One 0o700 dir per in-flight dump — the one path a
            # sibling bind-mounts. A DEDICATED root: ``_under_allowed`` is
            # prefix-based, so carving out ``backups/`` would expose the v1 tars.
            system_mount_roots=[
                str(Path(settings.data_dir).expanduser() / "models"),
                str(Path(settings.data_dir).expanduser() / "services"),
                str(dump_staging_root(Path(settings.data_dir).expanduser())),
            ],
            # Scopes container ownership so co-located daemons sharing a Docker
            # host don't reap each other's containers in the zombie sweep.
            instance_id=settings.daemon.instance_id,
        )
    except Exception:
        logger.warning("Docker not available, using stub runtime")
        runtime = StubRuntime()
    else:
        runtime = docker_runtime

    if docker_runtime is not None:
        checks: dict[str, Awaitable[bool]] = {}
        if any(gpu.schedulable and gpu.vendor == GpuVendor.nvidia for gpu in gpus):
            checks["nvidia runtime"] = docker_runtime.check_nvidia_runtime()
        if any(gpu.schedulable and gpu.vendor == GpuVendor.amd for gpu in gpus):
            checks["rocm runtime"] = docker_runtime.check_rocm_runtime()
        if checks:
            results = dict(
                zip(
                    checks,
                    await asyncio.gather(*checks.values(), return_exceptions=True),
                    strict=True,
                )
            )
            for name, result in results.items():
                if isinstance(result, BaseException):
                    logger.warning("Startup %s check failed: %s", name, result)

    # Deploy-from-git needs a git binary; warn non-fatally at startup so
    # the failure is obvious before the first POST /deploy/git (mirrors the
    # check_nvidia_runtime tone). Gated on [git].enabled.
    if settings.git.enabled and not git_available():
        logger.warning(
            "git binary not found; POST /deploy/git and the template store will "
            "return deploy.git_unavailable. Install git or set [git].enabled = false."
        )

    # In-process event bus for dashboard SSE.
    event_bus = EventBus()
    # bootstrap.py: hostname/proxy, secrets, and the model/data/
    # service controllers + WorkloadManager.
    hostname = resolve_hostname(settings)
    dashboard_upstream = resolve_dashboard_upstream(settings)
    proxy_manager = build_proxy_manager(queries, settings, hostname, dashboard_upstream, data_dir)
    secret_manager, shared_scope_blocked, rotate_key_blocked = await build_secret_manager(
        settings, queries
    )
    # (P25 §3.4.5) The proxy is constructed above, the secret manager here, so
    # the edge-auth resolver is wired with a late setter rather than by
    # reordering boot — a much smaller blast radius, and nothing has started
    # yet: the proxy is first used by build_controllers below and only started
    # further down the lifespan.
    proxy_manager.set_secret_resolver(build_edge_auth_resolver(secret_manager))
    (
        model_controller,
        data_controller,
        service_controller,
        workload_manager,
        event_recorder,
    ) = build_controllers(settings, runtime, queries, event_bus, proxy_manager, secret_manager)

    # Cross-plane orphan-container sweep (services, models, databases).
    # Constructed AFTER build_controllers so the run-registry hook can be
    # passed at construction: one-off runs and [deploy].release executions are
    # rowless containers, invisible to every DB query the sweep makes, and the
    # required kwarg has no post-construction assignment path.
    # The hook is now `protected_container_ids` — runs ∪ cutover
    # greens. A green is managed-by=nerdit but is no row's `container_id`, so
    # without it the sweep would kill the candidate mid-verification.
    zombie_sweeper = ZombieSweeper(
        queries=queries,
        runtime=runtime,
        extra_protected=service_controller.protected_container_ids,
        unbound_run_probe=service_controller.has_any_unbound_run,
    )

    # Unattended redeploy-on-push. A second controller on the same
    # reconcile tick, registered BEFORE the loop starts so no tick ever runs a
    # half-assembled controller set. Off entirely when [git] is disabled; a
    # service still opts in individually with [deploy].auto_deploy.
    if settings.git.enabled and settings.git.watch_interval_s > 0:

        async def _auto_redeploy(job: Job) -> None:
            """WP6's redeploy primitive, in-process (never an HTTP self-call)."""
            await redeploy_from_source(
                request_or_none=None,
                app=app,
                queries=queries,
                settings=settings,
                secrets=secret_manager,
                job=job,
                principal="system",
            )

        gitwatch = GitWatchController(
            queries,
            settings.git,
            secret_manager,
            redeploy=_auto_redeploy,
            # D-P24-8: the poller refuses to fire for a service the cutover
            # machinery would not protect, and says WHY in the feed.
            skip_reason=lambda job, cfg: cutover_skip_reason(
                job, cfg, settings.services, proxy_manager
            ),
            events=event_recorder,
            # `${github.installation}` resolves by repo through
            # the link manager's mirror. Read lazily off `app.state`: the
            # manager is built further down this lifespan and may be None for
            # the whole life of an unlinked daemon — which reads as "absent".
            github_token=lambda slug: _github_token_for_repo(app, slug),
        )
        workload_manager.register(gitwatch)
        app.state.gitwatch = gitwatch

    # URL layer: adopt-or-spawn Caddy, then drive route reconciliation on a
    # dedicated task (no-op when [proxy].enabled is false / the binary is absent).
    # Started BEFORE workload_manager so the first reconcile tick (which runs
    # before its first sleep) never sees proxy.available == False on a healthy
    # boot — a pending cutover-eligible redeploy would otherwise take the
    # destroy-first branch and GitWatch would emit a spurious
    # gitwatch.skipped_no_cutover(proxy_off) event.
    await proxy_manager.start()
    proxy_task = asyncio.create_task(
        _proxy_reconcile_loop(proxy_manager, DEFAULT_PROXY_RECONCILE_INTERVAL)
    )

    # (C2) Restore any database data tombstone left by a crash between the
    # pre-delete rename-aside and the checked row delete — BEFORE the first
    # reconcile tick, so a relaunch never boots on an empty data root.
    await _sweep_data_tombstones(queries, data_dir)
    await workload_manager.start()

    # Monitor
    monitor = ResourceMonitor(
        backend=gpu_backend,
        interval_seconds=settings.monitor.interval_seconds,
        temp_warning=settings.monitor.gpu_temp_warning,
        temp_critical=settings.monitor.gpu_temp_critical,
    )
    await monitor.start()

    # Display the internal-CA fingerprint whenever the URL layer is on —
    # this is the out-of-band value operators verify against 'nerdit trust'
    # on client machines, to establish trust-bootstrap authenticity.
    if proxy_manager.enabled:
        ca_pem = await proxy_manager.ca_root_pem()
        if ca_pem is not None:
            try:
                logger.info(
                    "Internal CA fingerprint: %s — clients should verify this "
                    "value when running 'nerdit trust'.",
                    ca_fingerprint(ca_pem),
                )
            except ValueError:
                logger.warning("Internal CA root.crt exists but could not be parsed.")

    # LAN name resolution. Advertises the effective hostname over mDNS;
    # self-gated on [proxy].mdns and degrades to a warning when the zeroconf
    # extra is missing or the hostname is not a single-label .local name.
    mdns_advertiser = MdnsAdvertiser(settings.proxy, hostname=hostname)
    await mdns_advertiser.start()

    # (P17d WP-D1) The offline product license. Loaded BEFORE the link manager
    # because the [link] seam below is its one v1 consumer. A missing file is
    # the ordinary unlicensed state and costs one stat; a broken one degrades to
    # doctor-visible and never aborts boot (D-LIC2). Published on app.state for
    # the doctor / capabilities / install-route surfaces, which refresh THIS
    # holder in place — the [license].file *path* is restart-keyed, its content
    # deliberately is not.
    app.state.license = await build_license_state(settings, event_recorder)

    # (P27 WP-C1) Outbound node link. Ships DARK: with [link].enabled false —
    # or enabled but not yet claimed — `build_link_manager` returns None and
    # nothing at all is constructed or started, so a fresh install dials
    # nowhere. When it is on, the manager owns its own connect/renew/reconnect
    # task; the daemon opens no inbound port for it (D-R3).
    link_manager = await build_link_manager(settings, event_recorder, queries=queries)
    # Publish on app.state BEFORE starting: the middleware's capability
    # resolution and the stale-bearer refusal both read this attribute, so it
    # must be set before any tunneled loopback request can possibly arrive
    # (uvicorn serves nothing until the lifespan yields, but the ordering
    # should not depend on that — /security-review PR #114 defense-in-depth).
    app.state.link_manager = link_manager
    if link_manager is not None:
        # (P17d WP-D2) The ONE v1 entitlement call site, evaluated only when a
        # tunnel is actually about to start. The posture is fail-open-with-
        # advisory and the caller owns it: W-D11 makes the relay the
        # authoritative online judge of `remote_link`, so a second offline
        # judge could only ever disagree with it — and under manual issuance a
        # stale file on disk is certain. One WARNING, then the tunnel starts as
        # it always has.
        #
        # The guard is ANY advisory reason, not `not allowed`: the D-LIC2
        # state matrix requires a boot warning for FOUR states, two of which are
        # allowed-with-advisory — `invalid` and `expired_grace` — beside the
        # two refusals (`expired`, `feature_not_licensed`). Silence is
        # correct for exactly two rows: no license at all, and valid-with-the-
        # feature. Every `allowed=False` decision carries a reason, so this
        # guard is a superset of the old one and never drops a warning.
        decision = app.state.license.require_entitlement(FEATURE_REMOTE_LINK)
        if decision.reason is not None:
            logger.warning(
                "License advisory for remote_link (%s). The tunnel still starts "
                "— the relay is the authoritative check — but renew or "
                "reinstall the license; see 'nerdit doctor'.",
                decision.reason,
            )
        await link_manager.start()

    # Outbound webhook delivery for the durable event feed. Ships
    # dark: nothing is constructed — so a fresh install POSTs nowhere — until
    # [notifications].enabled is on AND at least one target is configured.
    webhook_dispatcher: WebhookDispatcher | None = None
    if settings.notifications.enabled and settings.notifications.targets:
        webhook_dispatcher = WebhookDispatcher(
            queries,
            settings.notifications,
            secret_manager,
            bus=event_bus,
            version=__version__,
            instance_id=settings.daemon.instance_id,
        )
        await webhook_dispatcher.start()

    # Periodic orphan-container sweep
    zombie_task = asyncio.create_task(
        _zombie_sweep_loop(zombie_sweeper, ZOMBIE_SWEEP_INTERVAL_SECONDS)
    )

    # (P37 D-P37-2) The filesystem twin of the boot run-orphan kill: no slot
    # can exist at boot, so every entry under <data_dir>/dump-staging and every
    # <data_dir>/backups/.staging-* or .tmp-nerdit-* leftover is an orphan. A
    # symlinked staging root is refused, not swept through. Awaited: a handful
    # of rmtrees on a tree that is empty on every clean boot.
    try:
        swept = await asyncio.to_thread(sweep_staging_orphans, data_dir)
        if swept:
            logger.info("Boot sweep: removed %d orphan staging entr(ies)", swept)
    except Exception:
        logger.exception("Boot staging sweep failed — leftover staging dirs remain on disk")

    # Periodic expired-idempotency-key sweep
    idempotency_task = asyncio.create_task(
        _idempotency_sweep_loop(queries, IDEMPOTENCY_SWEEP_INTERVAL_SECONDS)
    )

    # Periodic data-retention sweep: skipped entirely when every phase is
    # off (nothing to prune). Sleeps before its first pass, so no boot-time
    # archive write.
    retention_settings = getattr(settings, "retention", None)
    retention_task: asyncio.Task[None] | None = None
    if retention_settings is not None and retention_sweep_enabled(retention_settings):
        retention_task = asyncio.create_task(
            _retention_sweep_loop(queries, retention_settings, Path(settings.data_dir).expanduser())
        )

    # Store in app state
    app.state.db = db
    app.state.queries = queries
    app.state.config_store = ConfigStore(Path("~/.nerdit/config.toml").expanduser())
    app.state.workload_manager = workload_manager
    app.state.runtime = runtime
    app.state.monitor = monitor
    app.state.gpu_backend = gpu_backend
    app.state.settings = settings
    app.state.event_bus = event_bus
    app.state.proxy_manager = proxy_manager
    app.state.mdns_advertiser = mdns_advertiser
    # (P27 WP-C1 item 5) `app.state.link_manager` is published above, before
    # the manager starts — explicitly None when the link is off/unclaimed: the
    # auth middleware, the doctor `link` check and the /capabilities `link`
    # block all `getattr` it, and "absent" and "None" must mean the same
    # inert thing.
    app.state.webhook_dispatcher = webhook_dispatcher
    app.state.secret_manager = secret_manager
    app.state.shared_scope_blocked = shared_scope_blocked
    app.state.rotate_key_blocked = rotate_key_blocked
    app.state.model_controller = model_controller
    app.state.data_controller = data_controller
    # The reconcile controller — restart drains its build registry;
    # and a boot-time GPU snapshot so GET /capabilities/GET /doctor project the
    # inventory without any live discovery I/O.
    app.state.service_controller = service_controller
    app.state.gpu_snapshot = {
        "count": gpu_count,
        "schedulable": sum(1 for gpu in gpus if gpu.schedulable),
        "vendors": sorted({gpu.vendor.value for gpu in gpus}),
    }
    app.state.hostname = hostname
    # (P26 WP1 / S-W10) The custom-domain reserved set, computed ONCE here.
    # Every `[proxy]` key feeding it is restart-required, so recomputing it
    # per request could only ever read a config change the running proxy has
    # not applied — a window in which a racing PUT could bind a name the live
    # Caddy is about to claim.
    # The raw socket name (and its short/.local forms) are reserved too: in
    # path mode the proxy answers on any Host, so they are names this box
    # serves even though nothing advertises them.
    app.state.domain_reserved = reserved_names(
        settings.proxy, hostname, aliases=host_aliases(socket.gethostname())
    )
    app.state.zombie_task = zombie_task
    app.state.idempotency_task = idempotency_task
    app.state.retention_task = retention_task
    app.state.proxy_task = proxy_task
    app.state.started_at = datetime.now(UTC)

    # (P13c §3) Drive the mounted MCP sub-app's session manager: its Starlette
    # lifespan never runs under Mount, so its run() context is entered here and
    # exited in the `finally` below — same task as the enter (the same-task
    # requirement of anyio task groups), and unconditionally, so a startup step
    # that raises after this point still tears the session manager down.
    # It is NOT the first thing exited on shutdown: the tunnel is stopped
    # before it — not to protect in-flight `/api/mcp` streams (uvicorn has
    # already closed the listener and drained every connection by the time a
    # lifespan shutdown runs, so the mux's loopback dial is refused long
    # before this line), but for the reason recorded below the `yield`: the
    # manager's `link.disconnected` row must land while the event recorder and
    # the DB are still open.
    mcp_session_cm = None
    mcp_server = getattr(app.state, "mcp_server", None)
    if mcp_server is not None:
        mcp_session_cm = mcp_server.session_manager.run()
        await mcp_session_cm.__aenter__()

    try:
        _warn_if_unauthenticated_exposure(settings.daemon, settings.proxy)
        logger.info("nerditd started on %s:%d", settings.daemon.host, settings.daemon.port)

        # Durable bookend. A consumer resuming from a cursor uses the
        # started/stopping pair to bound the window in which the daemon was not
        # observing anything at all — and the in-memory coalescing map is reset by
        # the same restart, so a repeated condition is allowed to re-emit.
        await event_recorder.record("daemon.started")
        yield
        await event_recorder.record("daemon.stopping")

        # (P27 WP-C1) Stop the tunnel FIRST, immediately after the stopping
        # bookend and before any task cancellation: closing the relay socket is
        # what stops tunnelled ingress, so it must happen before the controllers
        # drain — and doing it here means the manager's `link.disconnected`
        # (reason "shutdown") row still lands while the event recorder is
        # installed and the DB is open.
        if link_manager is not None:
            await link_manager.stop()
    finally:
        if mcp_session_cm is not None:
            await mcp_session_cm.__aexit__(None, None, None)

    # Shutdown
    zombie_task.cancel()
    idempotency_task.cancel()
    if retention_task is not None:
        retention_task.cancel()
    proxy_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await zombie_task
    with contextlib.suppress(asyncio.CancelledError):
        await idempotency_task
    if retention_task is not None:
        with contextlib.suppress(asyncio.CancelledError):
            await retention_task
    with contextlib.suppress(asyncio.CancelledError):
        await proxy_task
    await workload_manager.stop()
    # Cancel in-flight model pull/ensure tasks; containers are left running
    # (same survive-a-reboot posture as ServiceController.shutdown).
    await model_controller.shutdown()
    # Cancel in-flight database image-pull/ensure_ready tasks likewise.
    await data_controller.shutdown()
    await monitor.stop()
    await mdns_advertiser.stop()
    await proxy_manager.stop()
    # Stop the webhook drain before the DB closes — an in-flight
    # cursor write against a closed connection would be the only way this
    # best-effort path can raise into the shutdown.
    if webhook_dispatcher is not None:
        await webhook_dispatcher.stop()
    # Same posture for the durable feed's process singleton: clear it
    # before the DB closes so a post-shutdown emit is inert rather than writing
    # against a closed connection.
    set_recorder(None)
    await db.close()
    # Release the held `.restore.lock` — an offline restore can now
    # take the exclusive lock. Crash would auto-release too; this is the clean path.
    with contextlib.suppress(OSError):
        os.close(app.state.restore_lock_fd)
    logger.info("nerditd stopped")


def create_app() -> FastAPI:
    """Build the application while preserving server settings and lifespan patch points."""
    return build_app(load_settings(), lifespan=lifespan)


def _build_uvicorn_config(app: FastAPI, settings: NerditSettings) -> uvicorn.Config:
    """Build the uvicorn config for `main()` (extracted so it is testable).

    The only thing here that is not a straight settings passthrough is
    `timeout_graceful_shutdown` — see `GRACEFUL_SHUTDOWN_TIMEOUT_S`.
    """
    return uvicorn.Config(
        app,
        host=settings.daemon.host,
        port=settings.daemon.port,
        log_level=settings.log_level,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_S,
    )


def main() -> None:
    """Entry point for nerditd.

    Snapshots the boot invocation up front so `POST /daemon/restart` can re-exec it
    verbatim after uvicorn's graceful shutdown returns —
    same PID, so `~/.nerdit/nerditd.pid` stays valid. Falls back to the
    `-m nerdit.daemon.server` form (exactly the CLI-managed spawn) when the
    captured `argv[0]` is not an on-disk file.
    """
    # Children must see the host's loader path, not the bundle's (frozen only).
    restore_host_loader_env()
    from nerdit.daemon.routes.system import is_restart_requested, set_uvicorn_server

    boot_argv = list(sys.argv)
    settings = load_settings()
    app = create_app()
    # An explicit Server (not uvicorn.run) so the restart drain can stop it via
    # should_exit: uvicorn ≥0.29 replays captured signals after run() returns
    # (default handlers restored), so a SIGTERM-driven shutdown would kill the
    # process before the re-exec branch below ever executes.
    server = uvicorn.Server(_build_uvicorn_config(app, settings))
    set_uvicorn_server(server)
    server.run()

    if not is_restart_requested():
        return
    # Graceful shutdown has completed; re-exec the boot-captured invocation
    # (a console-script launch or `-m` form re-runs verbatim). Falls back to
    # the `-m` spawn — exactly the CLI-managed launch — when argv[0] is not a
    # usable on-disk file.
    if boot_argv and Path(boot_argv[0]).is_file():
        # Under a frozen daemon `sys.executable` IS the program, not an
        # interpreter: prepending it would append one more copy of the binary
        # path to argv on every restart, growing without bound.
        argv = list(boot_argv) if getattr(sys, "frozen", False) else [sys.executable, *boot_argv]
    else:
        argv = [sys.executable, "-m", "nerdit.daemon.server"]
    logger.info("Restart requested — re-execing %s", argv)
    try:
        os.execv(sys.executable, argv)
    except OSError as exc:
        # execv failed → exit normally; the pidfile is now stale. `nerdit
        # doctor` (unreachable daemon) + the CLI connect error cover discovery.
        logger.error("Re-exec failed (%s); exiting. The pidfile may be stale.", exc)


if __name__ == "__main__":
    main()
