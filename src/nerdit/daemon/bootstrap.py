"""Construct daemon subsystems from settings.

Functions return values for `lifespan()` to wire up; they keep no module state.
DockerRuntime construction stays in `lifespan()` to preserve its patch point.
Imports flow from server to bootstrap to core/db; never import daemon.server.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from nerdit.core.bindings.secretref import BindingNotReady, resolve_secret_ref
from nerdit.core.data import DataController, PostgresBackend, RedisBackend
from nerdit.core.dns import default_local_hostname
from nerdit.core.eventlog import EventRecorder, set_recorder
from nerdit.core.license import (
    REASON_MALFORMED,
    STATE_INVALID,
    TRUSTED_LICENSE_KEYS,
    LicenseError,
    LicenseState,
    LicenseVerdict,
    read_license_file,
    resolve_license_file,
    verify_license,
)
from nerdit.core.license import (
    utcnow as license_utcnow,
)
from nerdit.core.link.apps import AppStreamResolver
from nerdit.core.link.identity import (
    LinkIdentityError,
    load_or_create_identity,
    resolve_key_file,
)
from nerdit.core.link.manager import LinkManager
from nerdit.core.models import ModelController, OllamaBackend, VllmBackend
from nerdit.core.proxy import ProxyManager
from nerdit.core.secrets import (
    SHARED_SCOPE,
    InvalidServiceName,
    SecretDecryptError,
    SecretManager,
)
from nerdit.core.services import ServiceController
from nerdit.core.variables import load_scoped
from nerdit.core.workload import WorkloadManager
from nerdit.db.models import GpuVendor

if TYPE_CHECKING:
    from datetime import datetime

    from nerdit.config.settings import DaemonSettings, NerditSettings
    from nerdit.core.discovery import GpuBackend
    from nerdit.core.events import EventBus
    from nerdit.core.runtime.protocol import ContainerRuntime
    from nerdit.db.models import Gpu
    from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)


async def discover_gpus(
    settings: NerditSettings, queries: Queries
) -> tuple[GpuBackend | None, list[Gpu], int]:
    """Discover GPUs, reconcile them into the DB, and log the feedback."""
    gpu_count = 0
    gpu_backend = None
    gpus: list[Gpu] = []
    try:
        from nerdit.core.discovery import discover_gpu_system

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="nerdit-discovery") as executor:
            discovery = await loop.run_in_executor(
                executor,
                partial(
                    discover_gpu_system,
                    settings.monitor.zml_smi_path,
                    amd_schedulable=settings.monitor.enable_amd,
                ),
            )
        gpu_backend = discovery.backend
        gpus = discovery.gpus
        await queries.reconcile_gpus(gpus)
        gpu_count = len(gpus)
        logger.info("Discovered %d GPU(s) via %s", gpu_count, gpu_backend.name)
    except Exception as exc:
        await queries.reconcile_gpus([])
        logger.warning("GPU discovery failed, continuing without GPUs: %s", exc)

    # GPU feedback
    amd_present = any(gpu.vendor == GpuVendor.amd for gpu in gpus)
    amd_inventory_only = amd_present and not settings.monitor.enable_amd
    amd_hint = (
        "AMD GPUs are inventory-only. Set [monitor] enable_amd = true in "
        "~/.nerdit/config.toml to make them allocatable."
    )
    if gpu_count == 0:
        logger.warning("No GPU detected. Jobs requiring GPUs will stay pending.")
    elif not any(gpu.schedulable for gpu in gpus):
        if amd_inventory_only:
            logger.warning(amd_hint)
        else:
            logger.warning(
                "GPUs were detected for inventory and monitoring, but none are schedulable."
            )
    elif amd_inventory_only:
        logger.info(amd_hint)

    return gpu_backend, gpus, gpu_count


def normalize_mount_roots(settings: NerditSettings) -> None:
    """Always allow mounting from the *configured* upload dir.

    Uploaded job workspaces are bind-mounted as /workspace, so a non-default
    daemon.upload_dir would otherwise be rejected by the Tier-B allowlist
    (and, if it sits under ~/.nerdit, the Tier-A denylist) for scoped-token
    jobs. Normalizing the shared settings object covers both the runtime
    (Tier-A exemption) and the service sandbox (Tier-B), which read the same
    list.
    """
    if settings.daemon.upload_dir not in settings.containers.allowed_mount_roots:
        settings.containers.allowed_mount_roots = [
            *settings.containers.allowed_mount_roots,
            settings.daemon.upload_dir,
        ]


def resolve_hostname(settings: NerditSettings) -> str:
    """Resolve the effective hostname (P3/P9).

    The resolved hostname feeds both the public URL and the internal-CA cert
    SAN. P9: with the proxy + mDNS on and no explicit override, default the
    effective hostname to '<short-host>.local' so the public URL, the TLS
    cert SAN and the advertised mDNS name agree with zero configuration.
    Gated on enabled too: mdns without the proxy advertises nothing (the
    advertiser warns), so the hostname must not silently change either.
    """
    if settings.proxy.enabled and settings.proxy.mdns and settings.proxy.hostname_override is None:
        return default_local_hostname()
    return settings.proxy.hostname_override or socket.gethostname()


def resolve_dashboard_upstream(settings: NerditSettings) -> str:
    """Resolve the apex dial host:port.

    The dashboard-apex catch-all reverse-proxies to the daemon. Dial loopback
    when the daemon listens on all-interfaces/unset (reachable from the
    co-located Caddy); dial an explicit non-loopback host verbatim so
    `127.0.0.1` wouldn't miss the listener. Host/port policy lives here, out
    of the proxy module.
    """
    daemon_host = settings.daemon.host
    # Normalize the various all-interfaces spellings — incl. the bracketed IPv6
    # form `[::]` some tooling uses — before deciding to dial loopback; a raw
    # `[::]` upstream is not a reachable destination and would 502 every apex
    # request. A specific non-loopback host is dialed verbatim.
    daemon_host_norm = daemon_host.strip().strip("[]")
    apex_dial_host = (
        "127.0.0.1" if daemon_host_norm in ("0.0.0.0", "::", "::0", "") else daemon_host
    )
    return f"{apex_dial_host}:{settings.daemon.port}"


def build_proxy_manager(
    queries: Queries,
    settings: NerditSettings,
    hostname: str,
    dashboard_upstream: str,
    data_dir: Path,
) -> ProxyManager:
    """Construct the URL-layer `ProxyManager`.

    Opt-in via [proxy].enabled; off => services run on loopback exactly as in
    P2 (non-regression).
    """
    return ProxyManager(
        queries=queries,
        settings=settings.proxy,
        hostname=hostname,
        data_dir=data_dir,
        dashboard_upstream=dashboard_upstream,
    )


async def build_secret_manager(
    settings: NerditSettings, queries: Queries
) -> tuple[SecretManager, bool, bool]:
    """Construct the `SecretManager` and run its boot hooks.

    Construction runs the boot hooks: resume an interrupted key rotation,
    then migrate any legacy plaintext files — each migrated service gets its
    own audit row here. Also checks the two P8 reserved service names
    ('shared' / 'rotate-key') that predate the reserved-name rule.

    Returns `(secret_manager, shared_scope_blocked, rotate_key_blocked)`.
    """
    secret_manager = SecretManager(
        Path(settings.data_dir).expanduser() / "secrets",
        key_path=settings.security.secrets_key_file,
    )
    logger.info("Secrets encrypted at rest (key file: %s)", secret_manager.key_path)
    for migrated in secret_manager.migrated_services:
        try:
            await queries.insert_audit_log(
                action="secret.encrypt_migrate",
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="secret",
                target_id=migrated,
            )
        except Exception:
            logger.warning("Failed to audit secrets migration for %s", migrated, exc_info=True)
    # a pre-existing service literally named 'shared' predates the reserved
    # name. Never reinterpret its secrets as the shared scope — the secrets
    # routes refuse shared-scope operations while this flag is set.
    shared_scope_blocked = await queries.get_service_by_name("shared") is not None
    if shared_scope_blocked:
        logger.warning(
            "A service named 'shared' exists; the shared secrets scope is DISABLED "
            "until it is renamed or deleted ('shared' is a reserved name since P8)."
        )
    # a pre-existing service literally named 'rotate-key' predates the
    # reserved name, and the literal POST /secrets/rotate-key route shadows its
    # POST /secrets/{service} write path. Surface the collision (like 'shared')
    # and refuse the global rotation so it never silently rotates the key when
    # the admin meant to write that service's secrets.
    rotate_key_blocked = await queries.get_service_by_name("rotate-key") is not None
    if rotate_key_blocked:
        logger.warning(
            "A service named 'rotate-key' exists; secrets key rotation is DISABLED "
            "and that service's secrets cannot be set via POST until it is renamed "
            "or deleted ('rotate-key' is a reserved name since P8)."
        )
    return secret_manager, shared_scope_blocked, rotate_key_blocked


def build_edge_auth_resolver(
    secret_manager: SecretManager,
) -> Callable[[str, str | None, str], str | None]:
    """Build a single-reference resolver without exposing secret-store enumeration.

    Use the shared secretref precedence for local/shared references; the local
    map is the merged project < service reader over the ROW's `project_id`
    (D-P40-9 -- the proxy materializes only owner-declared config). Any failure
    returns None, causing the proxy to withhold the route. Plaintext stays local;
    never log, audit, persist or return it elsewhere.
    """

    def resolve(service_name: str, project_id: str | None, ref: str) -> str | None:
        try:
            return resolve_secret_ref(
                ref,
                load_scoped(secret_manager, service_name, project_id)[0],
                secret_manager.load(SHARED_SCOPE),
                label="deploy.edge_auth",
                field="password",
            )
        except (BindingNotReady, SecretDecryptError, InvalidServiceName, OSError):
            # `BindingNotReady` is the ordinary "not set (yet)" signal; the
            # others are a corrupted/unreadable store or a malformed project id
            # (`InvalidServiceName`). All of them mean the
            # same thing here — the handler cannot be materialized — and the
            # exception messages (which name keys and paths) stay unlogged.
            return None

    return resolve


def build_controllers(
    settings: NerditSettings,
    runtime: ContainerRuntime,
    queries: Queries,
    event_bus: EventBus,
    proxy_manager: ProxyManager,
    secret_manager: SecretManager,
) -> tuple[ModelController, DataController, ServiceController, WorkloadManager, EventRecorder]:
    """Construct backend-registered controllers and the workload manager.

    Rows select backends through config['backend']; lifecycle hooks are backend
    agnostic. Model and database bridge reachability both use models.bridge_host.
    """
    # One EventRecorder per daemon, built beside the EventBus: it
    # inserts the durable `events` row and then tees to the (lossy) bus.
    # `set_recorder` additionally installs it as the process singleton, so
    # `core/deploy_state.py`'s tail emitter fires for the
    # `core/app_build.py` / `core/launch.py` call sites too, which no
    # controller owns.
    event_recorder = EventRecorder(queries, event_bus)
    set_recorder(event_recorder)
    model_backend = OllamaBackend(
        image=settings.models.ollama_image,
        pull_timeout_s=float(settings.models.pull_timeout_s),
    )
    vllm_backend = VllmBackend(
        image=settings.models.vllm_image,
        shm_size=settings.models.vllm_shm_size,
        extra_args=list(settings.models.vllm_extra_args),
        pull_timeout_s=float(settings.models.pull_timeout_s),
    )
    model_controller = ModelController(
        backend=model_backend,
        extra_backends={vllm_backend.name: vllm_backend},
        default_backend=settings.models.default_backend,
        runtime=runtime,
        queries=queries,
        models_settings=settings.models,
        data_dir=str(Path(settings.data_dir).expanduser()),
        event_bus=event_bus,
        events=event_recorder,
    )
    postgres_backend = PostgresBackend(
        image=settings.databases.postgres_image,
        ready_timeout_s=float(settings.databases.ready_timeout_s),
    )
    # P15.5 registry proof: Redis joins as an extra_backends entry with zero
    # controller/route/CLI change — everything is backend-name-driven.
    redis_backend = RedisBackend(
        image=settings.databases.redis_image,
        ready_timeout_s=float(settings.databases.ready_timeout_s),
    )
    data_controller = DataController(
        postgres_backend,
        runtime=runtime,
        queries=queries,
        extra_backends={redis_backend.name: redis_backend},
        default_backend=settings.databases.default_backend,
        databases_settings=settings.databases,
        models_settings=settings.models,
        data_dir=str(Path(settings.data_dir).expanduser()),
        event_bus=event_bus,
        events=event_recorder,
    )
    service_controller = ServiceController(
        queries=queries,
        runtime=runtime,
        event_bus=event_bus,
        services_settings=settings.services,
        container_settings=settings.containers,
        proxy=proxy_manager,
        proxy_mode=settings.proxy.mode,
        secrets=secret_manager,
        model_controller=model_controller,
        data_controller=data_controller,
        data_dir=str(Path(settings.data_dir).expanduser()),
        retention_settings=getattr(settings, "retention", None),
        events=event_recorder,
    )
    workload_manager = WorkloadManager(
        service_controller,
        loop_interval=settings.services.loop_interval,
    )
    return (
        model_controller,
        data_controller,
        service_controller,
        workload_manager,
        event_recorder,
    )


def _loopback_base_url(daemon: DaemonSettings) -> str:
    """Build a dialable URL for replaying tunnel streams to this daemon.

    Use loopback for wildcard bind addresses, preserve explicit hosts and bracket
    IPv6 literals, including already-bracketed input.
    """
    host_norm = daemon.host.strip().strip("[]")
    if host_norm in ("0.0.0.0", "::", "::0", ""):
        return f"http://127.0.0.1:{daemon.port}"
    if ":" in host_norm:
        return f"http://[{host_norm}]:{daemon.port}"
    return f"http://{host_norm}:{daemon.port}"


async def build_license_state(
    settings: NerditSettings,
    event_recorder: EventRecorder,
    *,
    trusted_keys: Mapping[str, str] = TRUSTED_LICENSE_KEYS,
    now: Callable[[], datetime] = license_utcnow,
) -> LicenseState:
    """Load the product license without aborting daemon boot.

    Absent licenses produce an empty holder, no log or event. Unreadable files
    produce an error and invalid/malformed state. Invalid licenses log only the
    reason token and emit license.rejected, never the blob. Failures remain visible
    to doctor; remote-link enforcement belongs to the relay.
    """
    path = resolve_license_file(settings.license.file, str(settings.data_dir))
    try:
        # Blocking file I/O: off the loop, the `load_or_create_identity` idiom.
        blob = await asyncio.to_thread(read_license_file, path)
    except LicenseError as exc:
        logger.error(
            "Installed license could not be read (%s); the daemon continues "
            "unlicensed-with-advisory — see 'nerdit doctor'.",
            exc,
        )
        verdict = LicenseVerdict(STATE_INVALID, REASON_MALFORMED)
        await event_recorder.record("license.rejected", reason=REASON_MALFORMED)
        return LicenseState(verdict, now=now)

    if blob is None:
        return LicenseState(now=now)

    verdict = verify_license(blob, trusted_keys=trusted_keys, now=now)
    if verdict.state == STATE_INVALID:
        # The reason is a fixed machine token by construction — nothing
        # server-derived, nothing from the blob, reaches this log line.
        logger.error(
            "Installed license is not valid (%s); the daemon continues "
            "unlicensed-with-advisory — see 'nerdit doctor'.",
            verdict.reason,
        )
        await event_recorder.record("license.rejected", reason=verdict.reason)
    return LicenseState(verdict, now=now)


async def build_link_manager(
    settings: NerditSettings,
    event_recorder: EventRecorder,
    *,
    queries: Queries | None = None,
) -> LinkManager | None:
    """Build a tunnel manager when enabled, relay_url and node_id are all set.

    Unclaimed configuration warns; missing/corrupt keys log an error. Neither aborts
    boot, and doctor reports the failure. Without queries, slug or hosted domain,
    app streams have no resolver and fail closed with 404.
    """
    link = settings.link
    if not link.enabled:
        return None
    if not link.relay_url or not link.node_id:
        logger.warning(
            "[link] is enabled but this daemon is not linked yet — "
            "run 'nerdit link' to claim it (WP-C2). The tunnel stays down."
        )
        return None

    key_file = resolve_key_file(link.key_file, str(settings.data_dir))
    try:
        # Blocking file I/O + an Ed25519 keygen on first boot: off the loop.
        identity = await asyncio.to_thread(load_or_create_identity, key_file)
    except LinkIdentityError as exc:
        logger.error(
            "Node link disabled: the node identity key could not be loaded (%s). "
            "Fix or remove the key file and restart.",
            exc,
        )
        return None

    if not getattr(settings.daemon, "auth_token", None):
        # Mirrors the [mcp].http_enabled precedent, one notch softer (that one
        # hard-errors; this one must not take a local control plane down for an
        # opt-in accessory). Without a daemon token the middleware runs in v0.1
        # local mode, where every non-tunnel bearer is now REFUSED rather than
        # silently promoted to the LOCAL admin — so an operator whose CLI still
        # carries an old token needs to know why it stopped working.
        logger.warning(
            "[link] is enabled but [daemon].auth_token is unset. The daemon is in "
            "local (tokenless) mode: any request carrying a Bearer token that is "
            "not the live tunnel capability is refused. Set [daemon].auth_token "
            "('nerdit token') so local callers authenticate normally."
        )

    node_name = link.slug or socket.gethostname()

    # (P26 D-P26-H1/H5) The hosted URL is computed, never guessed: without BOTH
    # the claim's slug and the cloud's own base domain there is no authority to
    # rewrite `Host` to, so no app stream can be served — and the resolver
    # stays `None` rather than being handed a half-known name.
    resolve_app = None
    if queries is not None and link.slug and link.nodes_base_domain:
        resolve_app = AppStreamResolver(
            queries, slug=link.slug, nodes_base_domain=link.nodes_base_domain
        ).resolve
    elif queries is not None:
        logger.info(
            "Hosted shares inactive: [link].nodes_base_domain is unset — "
            "run 'nerdit link refresh'. The tunnel still serves the daemon API."
        )
    from nerdit import __version__  # noqa: PLC0415 - avoids a package-import cycle

    logger.info(
        "Node link enabled (node %s, relay %s) — outbound only, submitter role.",
        link.node_id,
        urlsplit(link.relay_url).netloc,
    )
    return LinkManager(
        link,
        identity,
        node_id=link.node_id,
        node_name=node_name,
        daemon_version=__version__,
        loopback_base_url=_loopback_base_url(settings.daemon),
        events=event_recorder,
        resolve_app=resolve_app,
    )
