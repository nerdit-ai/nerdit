"""Report daemon capabilities and health, and perform maintenance.

Capabilities is a role-aware, I/O-free projection. Non-admins never receive
paths, proxy admin addresses, relay hosts or license customer IDs. Tokens and
license blobs are never exposed. Link and exposure blocks describe live state.

Doctor runs bounded concurrent checks and reports the worst status. Details
contain no filesystem paths or secret values. Restart is admin-only: commit its
audit row, drain builds within the configured timeout, then signal the process
to re-execute its captured boot invocation.
"""

from __future__ import annotations

import asyncio
import errno
import fnmatch
import json
import logging
import os
import shutil
import signal
import socket
import sqlite3
import stat
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Query, Request
from fastapi.routing import APIRouter
from pydantic import Field

from nerdit.config.defaults import DEFAULT_DB_NAME
from nerdit.config.settings import load_settings
from nerdit.config.store import _RESTART_KEYS, restart_section_holder
from nerdit.core.backup import (
    DUMP_TAR_GLOB,
    BackupError,
    create_backup,
    create_volume_backup,
)
from nerdit.core.dns import MDNS_REASON_ZEROCONF_MISSING
from nerdit.core.gitsource import git_available
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.license import (
    FEATURE_REMOTE_LINK,
    LICENSE_GRACE_S,
    STATE_EXPIRED,
    STATE_EXPIRED_GRACE,
    STATE_INVALID,
    resolve_license_file,
)
from nerdit.core.link.hosted import hosted_url
from nerdit.core.link.identity import resolve_key_file
from nerdit.core.link.manager import GITHUB_TOKEN_WARN_LEAD_S
from nerdit.core.node_runtime import SUPPORTED_NODE_VERSIONS
from nerdit.core.proxy import ProxyState, generate_route, public_url_for
from nerdit.core.runtime.stub import StubRuntime
from nerdit.core.secrets import SecretDecryptError, SecretRotationInProgress
from nerdit.core.volumes import VolumeSpecError, dump_staging_root, tombstone_service_name
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import current_principal, require_role
from nerdit.daemon.bootstrap import normalize_mount_roots
from nerdit.daemon.deploy_pipeline import APEX_RESERVED_NAMES, apex_shadow_active
from nerdit.daemon.errors import NerditError, request_id_of
from nerdit.daemon.imagegc import (
    DEFAULT_INSTANCE_ID,
    _attribute_images,
    _live_repos,
    _live_service_names,
    _orphan_app_repos,
    _orphan_data_dir_names,
    _orphan_data_dir_path,
    _protected_image_refs,
    _repo_in_progress,
    _repo_size_estimate,
    _repo_tags,
)
from nerdit.daemon.limits import (
    _MAX_DIAGNOSE_TAIL,
    _MAX_LOG_TAIL,
    _WAIT_TIMEOUT_MAX,
    EVENTS_STREAM_CONCURRENCY_MAX,
    LOGS_STREAM_CONCURRENCY_MAX,
    MAX_SSE_REPLAY,
    WAIT_CONCURRENCY_MAX,
)
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.daemon.schemas.tokens import seconds_until
from nerdit.daemon.secret_scope import variable_write_lock
from nerdit.db.models import BackupResponse, JobKind, TokenRole, VolumeBackupResponse
from nerdit.db.queries._base import mark_request_side_effect
from nerdit.utils.disk import du_bytes, resolve_archive_dir, spawn_walk

logger = logging.getLogger(__name__)

router = APIRouter()

# The buildpack precedence order the Builder tries (Dockerfile passthrough wins).
_BUILDPACKS = ["dockerfile", "python", "node"]
# The services-list page-size clamp (mirrors the `le=200` on GET /services).
_PAGE_LIMIT_MAX = 200
# Restart-drain poll cadence + clamp bounds.
_DRAIN_POLL_SECONDS = 1.0
_DRAIN_TIMEOUT_MAX = 300


# --- restart flag (module-level, read via the accessors below) ----------------
#
# A restart is requested by setting a module global; `main()` reads it after
# `uvicorn.run()` returns to decide whether to re-exec. Callers read/write it
# through the accessors (never `from … import restart_requested` — that binds a
# stale snapshot). The check-and-set in the route has no `await` between the
# 409 read and the set, so asyncio single-thread atomicity makes it race-free.
restart_requested: bool = False

# The fire-and-forget drain task is retained here so the event loop cannot GC it
# out from under us (the loop holds only weak refs to bare `create_task`
# handles). Set beside the flag when the restart commits; cleared by the
# test-only reset helper.
_drain_task: asyncio.Task[None] | None = None

# The running uvicorn.Server, registered by `main()` so the drain can stop it
# programmatically. Uvicorn ≥0.29 REPLAYS captured signals after `run()`
# completes (uvicorn/server.py: `signal.raise_signal` with default handlers
# restored), so a SIGTERM-triggered shutdown kills the process before the
# re-exec branch in `main()` ever runs. Setting `should_exit` drives the
# identical graceful shutdown with no captured signal, so `run()` returns.
# SIGTERM stays as the fallback when no server handle is registered.
_uvicorn_server: Any | None = None


def set_uvicorn_server(server: Any) -> None:
    """Register the running uvicorn.Server for programmatic shutdown."""
    global _uvicorn_server
    _uvicorn_server = server


def request_restart() -> None:
    """Set the module-level restart flag (idempotent)."""
    global restart_requested
    restart_requested = True


def is_restart_requested() -> bool:
    """Whether a graceful restart has been requested this process lifetime."""
    return restart_requested


def reset_restart_requested() -> None:
    """Clear the restart flag + drain-task ref.

    Used by tests between cases, and by the restart route's rollback path when
    the pre-drain audit insert fails (so a wedged flag never leaves the daemon
    in a permanent 409 / spurious re-exec state).
    """
    global restart_requested, _drain_task, _uvicorn_server
    restart_requested = False
    _drain_task = None
    _uvicorn_server = None


# --- GET /capabilities --------------------------------------------------------


@router.get("/capabilities", operation_id="get_capabilities")
async def get_capabilities(request: Request) -> dict[str, Any]:
    """Project the daemon's static capabilities + the caller's role (1.5).

    A pure `app.state` read (no I/O). Role-aware: the admin-only
    `proxy.admin_addr` field and the whole `paths` block are **omitted** for
    non-admin callers (absent keys, not nulls); `[git].allowed_hosts` stays
    visible to every role so an agent knows which hosts it may deploy from, and
    `sandbox` projects the container hardening every deployed image has
    to run under.
    """
    state = request.app.state
    settings = state.settings
    principal = current_principal(request)
    is_admin = principal.role == TokenRole.admin

    started_at = getattr(state, "started_at", None)
    uptime_s = (
        int((datetime.now(UTC) - started_at).total_seconds()) if started_at is not None else 0
    )

    proxy_settings = settings.proxy
    proxy_manager = getattr(state, "proxy_manager", None)
    mode = proxy_settings.mode
    effective_hostname = proxy_settings.hostname_override or getattr(state, "hostname", None)
    # url_shape is composed through the route seams (never hand-built), so a
    # mode flip changes the grammar without touching this projection.
    url_shape = public_url_for(
        "<name>",
        generate_route("<name>", mode=mode),
        mode=mode,
        hostname=effective_hostname or "",
        base_domain=proxy_settings.base_domain,
        scheme=proxy_settings.scheme,
        https_port=proxy_settings.https_port,
        public_port=proxy_settings.public_port,
    )
    proxy_block: dict[str, Any] = {
        "enabled": proxy_settings.enabled,
        "available": bool(proxy_manager.available) if proxy_manager is not None else False,
        "mode": mode,
        "base_domain": proxy_settings.base_domain,
        "hostname": effective_hostname,
        "scheme": proxy_settings.scheme,
        "https_port": proxy_settings.https_port,
        "public_port": proxy_settings.public_port,
        "url_shape": url_shape,
        "dashboard_apex": proxy_settings.dashboard_apex,
        "mdns": proxy_settings.mdns,
    }
    if is_admin:
        proxy_block["admin_addr"] = proxy_settings.admin_addr

    model_controller = getattr(state, "model_controller", None)
    if model_controller is not None:
        models_block = {
            "backends": model_controller.backends,
            "default_backend": model_controller.default_backend_name,
            "bridge_host": model_controller.bridge_host,
        }
    else:
        models_block = {
            "backends": [],
            "default_backend": settings.models.default_backend,
            "bridge_host": settings.models.bridge_host,
        }

    # Managed-data plane self-knowledge, mirroring the models block.
    data_controller = getattr(state, "data_controller", None)
    if data_controller is not None:
        databases_block = {
            "backends": data_controller.backends,
            "default_backend": data_controller.default_backend_name,
            "bridge_host": data_controller.bridge_host,
        }
    else:
        databases_block = {
            "backends": [],
            "default_backend": settings.databases.default_backend,
            "bridge_host": settings.models.bridge_host,
        }

    # Node-link session state. Folded in here rather than given a GET /link of
    # its own: /capabilities is THE role-aware self-knowledge surface and a pure
    # `app.state` projection, which `LinkStatus` is by construction (no I/O, no
    # locks). Health-shaped consumption belongs to the doctor `link` check.
    # Visible to every authenticated role (a state name and two timestamps
    # are not secrets); `relay_host` is admin-only per
    # the `proxy.admin_addr`/`paths` precedent, and the capability token
    # appears nowhere at all.
    #
    # The block also carries the three hosted-share facts an agent needs before
    # it calls `PUT /services/{name}/share`: the node `slug` and
    # `nodes_base_domain` are the two halves of every hosted name, and
    # `hosted_public_entitled` says whether `access='public'` can succeed at
    # all. This block IS the link status surface; there is no GET /link/status.
    link_manager = getattr(state, "link_manager", None)
    link_settings = getattr(settings, "link", None)
    slug = getattr(link_settings, "slug", None)
    nodes_base_domain = getattr(link_settings, "nodes_base_domain", None)
    hosted_public = False
    if link_manager is None:
        # Warn only when no node_id indicates the node is unlinked. Disabled linking with
        # an existing identity is a deliberate opt-out and stays quiet. Read config only;
        # doctor is advisory and local services remain fully usable. Name missing remote
        # features and the recovery command without URLs or key material.
        link_block: dict[str, Any] = {
            "enabled": False,
            "node_id": getattr(link_settings, "node_id", None),
            "slug": slug,
        }
    else:
        link_status = link_manager.status()
        hosted_public = bool(getattr(link_status, "hosted_public_entitled", False))
        link_block = {
            "enabled": True,
            "state": link_status.state,
            "node_id": link_status.node_id,
            "slug": slug,
            "nodes_base_domain": nodes_base_domain,
            "hosted_public_entitled": hosted_public,
            # When the cloud last asserted the line above, on this
            # daemon's clock — `null` while it never has. A timestamp, not a
            # secret, so every role sees it: without it "false" is ambiguous
            # between "the plan says no" and "the cloud has gone quiet", and
            # `nerdit link` could not tell an operator which.
            "hosted_public_entitled_at": (
                link_status.hosted_public_entitled_at.isoformat()
                if getattr(link_status, "hosted_public_entitled_at", None) is not None
                else None
            ),
            # The live GitHub installation tokens, as secret-free
            # summaries: id, expiry, repo COUNT. `nerdit link` renders its
            # `github =` line from this; the token and the repo names are
            # nowhere in the payload.
            "github_installations": [
                {
                    "installation_id": inst.installation_id,
                    "expires_at": inst.expires_at.isoformat(),
                    "repos_count": inst.repos_count,
                }
                for inst in getattr(link_status, "github_installations", ())
            ],
            "connected_at": (
                link_status.connected_at.isoformat()
                if link_status.connected_at is not None
                else None
            ),
            "capability_expires_at": (
                link_status.capability_expires_at.isoformat()
                if link_status.capability_expires_at is not None
                else None
            ),
            "last_close_code": link_status.last_close_code,
            "terminal_reason": link_status.terminal_reason,
            "active_streams": link_status.active_streams,
        }
        if is_admin:
            link_block["relay_host"] = link_status.relay_host

    # Expose capabilities and public URL grammar, never credentials or CA account data.
    # Hosted requires live link state and known metadata, using the share route's URL
    # builder. Domains is a static capability independent of linking; ACME reflects
    # enabled node policy, not whether any particular certificate has been issued.
    # Missing settings degrade to disabled.
    if link_manager is not None and slug and nodes_base_domain:
        hosted_url_shape: str | None = hosted_url("<name>", slug, nodes_base_domain)
    else:
        hosted_url_shape = None
    exposure_block: dict[str, Any] = {
        "hosted": hosted_url_shape is not None,
        "hosted_public": hosted_public,
        "domains": True,
        "acme": bool(getattr(getattr(settings.proxy, "acme", None), "enabled", False)),
        "nodes_base_domain": nodes_base_domain,
        "hosted_url_shape": hosted_url_shape,
    }

    # The offline product license. Folded in beside `link` for
    # the same reason: a pure `app.state` projection off the boot/install-time
    # holder (no file I/O here — the verdict was computed when the file was
    # read), so the existing `nerdit capabilities` + MCP `capabilities`
    # paths surface it with zero new REST operations. Health-shaped consumption
    # belongs to the doctor `license` check. The **temporal** half is
    # recomputed live by `LicenseState.state`, so a grace transition shows up
    # with no poller and no restart. `customer_id` is admin-only per the
    # `proxy.admin_addr`/`link.relay_host` precedent — OMITTED, not nulled —
    # and the blob appears nowhere at all.
    license_state = getattr(state, "license", None)
    if license_state is None or not license_state.installed:
        license_block: dict[str, Any] = {"installed": False}
    else:
        claims = license_state.claims
        license_block = {"installed": True, "state": license_state.state}
        if claims is None:
            # Invalid: there are no trustworthy claims to project, only the
            # machine reason token.
            license_block["reason"] = license_state.reason
        else:
            license_block.update(
                {
                    "lid": claims.lid,
                    "plan": claims.plan,
                    "features": list(claims.features),
                    "expires_at": claims.expires_at.isoformat(),
                    # Rendered on the HOLDER's clock, so this number can never
                    # disagree with the `state` beside it. `seconds_until`
                    # clamps at 0, so an expired license reads `0` rather than
                    # a negative — `state` carries that half of the truth, and
                    # the clamp keeps one shape for both consumers of the
                    # token/license expiry idiom.
                    "expires_in_s": seconds_until(claims.expires_at, license_state.clock()),
                }
            )
            if is_admin:
                license_block["customer_id"] = claims.customer_id

    # The container sandbox an agent's image has
    # to survive. Read through `getattr` so a lightweight settings object
    # without `[containers]` still projects the shipped defaults rather than
    # raising (the `[proxy.acme]` precedent above). Visible to every role: it
    # names a hardening POLICY, not a credential, and an agent that cannot read
    # it deploys a stock root image, watches the entrypoint die on
    # `chown(...) Operation not permitted`, and burns the restart budget —
    # exactly the loop `image_needs_privileges` was added to close.
    container_settings = getattr(settings, "containers", None)
    sandbox_block = {
        "drop_all_caps": bool(getattr(container_settings, "drop_all_caps", True)),
        "no_new_privileges": bool(getattr(container_settings, "no_new_privileges", True)),
        "read_only_rootfs": bool(getattr(container_settings, "read_only_rootfs", False)),
    }

    gpu_snapshot = getattr(state, "gpu_snapshot", None) or {
        "count": 0,
        "schedulable": 0,
        "vendors": [],
    }
    mcp_settings = getattr(settings, "mcp", None)

    body: dict[str, Any] = {
        "version": _daemon_version(),
        "uptime_s": uptime_s,
        "caller": {
            "role": principal.role.value,
            "token_name": principal.name,
            "quotas": {
                "max_gpus": principal.max_gpus,
                "max_concurrent_jobs": principal.max_concurrent_jobs,
            },
        },
        # The agent's self-knowledge path: with no MCP
        # tool for the token surface, this block is how an agent learns its own
        # lifetime and reach BEFORE the clock runs out — after it does, every
        # route including `/tokens/self` answers 403 `token_expired`.
        # Projected straight off the `Principal` (which already carries both
        # `expires_at` and `scope_services`), so this route stays the pure
        # `app.state` read its docstring promises — no DB access, ever.
        "token": {
            "role": principal.role.value,
            "expires_at": (
                principal.expires_at.isoformat() if principal.expires_at is not None else None
            ),
            "expires_in_s": seconds_until(principal.expires_at),
            "scope_services": (
                sorted(principal.scope_services) if principal.scope_services is not None else None
            ),
            "rotatable": principal.token_id is not None,
        },
        "proxy": proxy_block,
        "link": link_block,
        "exposure": exposure_block,
        "license": license_block,
        "models": models_block,
        "databases": databases_block,
        "buildpacks": list(_BUILDPACKS),
        "deploy": {
            "git_enabled": settings.git.enabled,
            "git_allowed_hosts": list(settings.git.allowed_hosts),
            "max_upload_bytes": settings.daemon.max_upload_bytes,
            "dry_run": True,
            "build_settings": {
                "version": 1,
                "node_versions": list(SUPPORTED_NODE_VERSIONS),
                "presets": ["node", "nextjs", "python", "dockerfile"],
                "fields": [
                    "preset",
                    "install",
                    "build",
                    "start",
                    "node_version",
                    "package_manager",
                    "subdir",
                    "public_env",
                ],
                "public_env": True,
                "secret_mounts": False,
            },
            "max_concurrent_builds": settings.services.max_concurrent_builds,
        },
        "sandbox": sandbox_block,
        "gpus": gpu_snapshot,
        "mcp": {"http_enabled": bool(mcp_settings is not None and mcp_settings.http_enabled)},
        "limits": {
            "wait_timeout_max_s": _WAIT_TIMEOUT_MAX,
            "wait_concurrency_max": WAIT_CONCURRENCY_MAX,
            # The stream budgets: an agent that plans N concurrent
            # followers needs to know the daemon-wide cap BEFORE it opens the
            # N+1st and gets a `*.saturated` frame back.
            "events_stream_concurrency_max": EVENTS_STREAM_CONCURRENCY_MAX,
            "logs_stream_concurrency_max": LOGS_STREAM_CONCURRENCY_MAX,
            "events_stream_replay_max": MAX_SSE_REPLAY,
            "log_tail_max": _MAX_LOG_TAIL,
            "diagnose_log_tail_max": _MAX_DIAGNOSE_TAIL,
            "page_limit_max": _PAGE_LIMIT_MAX,
            "service_port_range": settings.services.service_port_range,
        },
        "features": {
            "secrets_shared_scope": not getattr(state, "shared_scope_blocked", False),
            "app_templates": True,
            # Constant true on this build: lets a frontend distinguish this
            # daemon from an older one, since FastAPI silently ignores unknown
            # GET /audit ?target= / ?target_type= params (no 422 to sniff).
            "audit_target_filter": True,
            # Enabled-flag ONLY — never the target list, a target count,
            # or any URL: targets carry operator-chosen endpoints (and, when an
            # operator ignores the credential-placement warning, tokens).
            "notifications": getattr(getattr(settings, "notifications", None), "enabled", False),
            # Constant ``True`` on this build: an unknown path is a plain 404,
            # so a capability flag is how an agent tells "predates dumps" from
            # "no such database". The surface exists; dumpability is the route's.
            "database_dumps": True,
            # Constant ``True`` on this build: a 404 on `POST /secrets/{name}`
            # is how an older daemon says "deploy first", so the flag is how an
            # agent tells "predates secrets-before-deploy" from "no such service".
            "secrets_before_deploy": True,
            # Constant ``True`` on this build: an unknown `/projects`
            # path is a plain 404, so the flag is how an agent tells "predates
            # the project noun" from "no such project".
            "projects": True,
            "project_ids": True,
            "project_rename_v1": True,
            "project_delegation": True,
            "public_address_bindings": True,
            "public_address_routing": True,
            # Constant ``True``: `/projects/{p}/variables` is a plain 404 on an
            # older daemon, indistinguishable from "no such project".
            "variables": True,
            # Constant ``True``: `POST /projects/{p}/apply` is a plain 404/405
            # on an older daemon, and a legacy deploy of a `[project]` file
            # answers `deploy.use_apply` only from this build on.
            "project_apply": True,
        },
    }
    if is_admin:
        data_dir = Path(settings.data_dir).expanduser()
        body["paths"] = {
            "data_dir": str(data_dir),
            "db_path": str(data_dir / DEFAULT_DB_NAME),
        }
    return body


def _daemon_version() -> str:
    from nerdit import __version__

    return __version__


# --- GET /doctor --------------------------------------------------------------

_STATUS_RANK = {"ok": 0, "warn": 1, "fail": 2}

# Keep stable terminal reason tokens and add path/URL-free operator hints.
# Unauthorized means suspended/deleting account. Revocation retains local node_id,
# so recovery must unlink before linking again; otherwise already-linked guards
# short-circuit. Unlink needs no cloud access and wipes identity, so the next link
# enrolls a new verifier rather than resurrecting the old node.
_LINK_TERMINAL_HINTS: dict[str, str] = {
    "entitlement_required": (
        "the relay refused the tunnel: this Nerdit account is not active "
        "(suspended, or being deleted). Restore the account with support, then "
        "restart the daemon; retrying cannot help."
    ),
    "revoked": (
        "the cloud revoked this node's link (unlinked in the console, or swept "
        "after long inactivity). This daemon still holds the revoked identity, "
        "and linking is refused while it does: run 'nerdit unlink' here (it "
        "wipes the node identity key), then 'nerdit link --device'. Restarting "
        "the daemon cannot help."
    ),
}


def _whole_days(seconds: float) -> int:
    """Whole days in `seconds`, floored at 0 — the doctor day-count idiom.

    Floored rather than rounded up so the count never overstates the time an
    operator has: `0d` honestly means "less than a day left", which is the
    reading that makes someone act today.
    """
    return max(0, int(seconds // 86400))


def _key_path_shape(path: Path) -> str:
    """Classify a key path as absent, regular or other using metadata only.

    A FIFO read would wedge its worker beyond the async timeout. Use lstat to see
    dangling symlinks as unusable, then stat to accept links to regular files.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        # Nothing at the path — not even a symlink entry. The lazy-creation
        # case; the caller decides via has_ciphertexts() whether that is fine.
        return "absent"
    except OSError:
        # ENOTDIR / EACCES on a parent / ELOOP / ENAMETOOLONG: the path cannot
        # even be interrogated, so the daemon will not be able to use it.
        return "other"
    if stat.S_ISLNK(st.st_mode):
        try:
            st = os.stat(path)  # one hop; `stat` on a FIFO does not block
        except OSError:
            return "other"  # dangling or otherwise unresolvable
    return "regular" if stat.S_ISREG(st.st_mode) else "other"


async def _run_check(name: str, probe) -> dict[str, Any]:  # noqa: ANN001
    """Run one check under a 2 s budget; any error/timeout ⇒ a generic fail.

    `latency_ms` is always reported. Details never carry filesystem paths or
    secret values, so the same body is safe for every role.
    """
    start = time.monotonic()
    try:
        status, detail = await asyncio.wait_for(probe(), 2.0)
    except Exception:  # noqa: BLE001 — a diagnostic must never propagate
        status, detail = "fail", "check failed or timed out"
    latency_ms = int((time.monotonic() - start) * 1000)
    return {"name": name, "status": status, "detail": detail, "latency_ms": latency_ms}


@router.get("/doctor", operation_id="get_doctor")
async def get_doctor(request: Request) -> dict[str, Any]:
    """Run the timeout-bounded structured health checks (1.6).

    All checks run concurrently; the top `status` is the worst-of (`skipped`
    never worsens it). Details are path-free and secret-free for every role.
    """
    state = request.app.state
    settings = state.settings

    async def _docker() -> tuple[str, str]:
        runtime = getattr(state, "runtime", None)
        # The stub is indistinguishable from a healthy-empty Docker via a probe
        # (its list_images() returns [] without raising), so detect it by type.
        if isinstance(runtime, StubRuntime):
            return "fail", "Docker unreachable at startup — daemon running on the stub runtime"
        if runtime is None:
            return "fail", "container runtime not initialised"
        await runtime.list_images()
        # (BUG-1) The runtime answering says nothing about whether an image can
        # be BUILT. `warn`, not `fail`: a node that only runs pre-built
        # images (POST /services, nerdit serve, databases) is fully functional
        # without buildx, and `fail` would set doctor's exit code and the
        # worst-of top status. The hard signals live where they convert to
        # action — check-deps FAILS, and the build itself classifies
        # PLATFORM_ERROR with a remediation. No paths in the detail (doctor
        # discipline); the path-bearing guidance is check-deps' remediation.
        buildx = await runtime.buildx_available()
        if buildx == "missing":
            return "warn", (
                "Docker runtime responding; the BuildKit builder (docker buildx) "
                "is NOT available to this daemon — every image build will fail. "
                "Install the buildx CLI plugin system-wide, or set DOCKER_CONFIG "
                "in the daemon's service unit"
            )
        if buildx == "no_cli":
            # Distinct from `unknown`: the daemon reached
            # the socket but has no `docker` CLI to build with — concluded,
            # not inconclusive. `warn` for the same recorded reason as the
            # buildx branch above: a node serving only pre-built images is
            # healthy. Path-free, like every doctor detail.
            return "warn", (
                "Docker runtime responding; the daemon cannot find the docker CLI on "
                "its PATH — every image build will fail. Install the Docker CLI where "
                "the daemon service can see it, or set PATH in the daemon's service unit"
            )
        if buildx == "present":
            return "ok", "Docker runtime responding; BuildKit builder available"
        # `unknown` never worsens the status — the probe is bounded well
        # inside the 2 s per-check budget precisely so it cannot turn the
        # existing list_images signal into "check failed or timed out".
        return "ok", "Docker runtime responding; BuildKit builder state unknown"

    async def _gpu() -> tuple[str, str]:
        snap = getattr(state, "gpu_snapshot", None)
        if snap is None:
            return "skipped", "GPU inventory unavailable"
        count = int(snap.get("count", 0))
        schedulable = int(snap.get("schedulable", 0))
        if count == 0:
            return "warn", "no GPUs detected"
        return "ok", f"{count} GPU(s), {schedulable} schedulable"

    async def _proxy() -> tuple[str, str]:
        manager = getattr(state, "proxy_manager", None)
        if manager is None:
            return "skipped", "proxy manager not initialised"
        st = manager.state
        if st == ProxyState.foreign_conflict:
            return "fail", "the proxy admin port is answered by a non-nerdit process"
        if st == ProxyState.backoff:
            return "warn", "proxy is respawning (backoff)"
        if st == ProxyState.no_binary:
            # Enabled but the caddy binary is unresolved: a real misconfiguration
            # the operator asked for (proxy on) but cannot satisfy — warn, don't
            # bury it in `skipped`. No path in the detail.
            return "warn", "proxy enabled but the caddy binary is missing — install caddy"
        if st == ProxyState.disabled:
            return "skipped", "proxy disabled"
        if st == ProxyState.available:
            # A row predating the apex reservation still shadows the dashboard.
            if apex_shadow_active(settings):
                for name in sorted(APEX_RESERVED_NAMES):
                    if await state.queries.get_service_by_name(name) is not None:
                        return "warn", (
                            f"service '{name}' shadows the dashboard at the proxy apex"
                            " — rename it or disable [proxy].dashboard_apex"
                        )
            return "ok", "proxy up"
        return "warn", "proxy starting (no successful tick yet)"

    async def _mdns() -> tuple[str, str]:
        if not settings.proxy.mdns:
            return "skipped", "mDNS advertising disabled"
        advertiser = getattr(state, "mdns_advertiser", None)
        if advertiser is not None and getattr(advertiser, "registered", False):
            return "ok", "mDNS advertising active"
        # Distinguish a missing optional dependency from any other registration
        # failure so the operator gets an actionable hint (the advertiser stamps
        # a machine-readable reason at each early-return in start()).
        if getattr(advertiser, "reason", None) == MDNS_REASON_ZEROCONF_MISSING:
            return (
                "warn",
                'mDNS enabled but the zeroconf package is missing — pip install "nerdit[mdns]"',
            )
        return "warn", "mDNS enabled but not registered"

    async def _secrets_key() -> tuple[str, str]:
        manager = getattr(state, "secret_manager", None)
        if manager is None:
            return "fail", "secret manager not initialised"

        # The stat calls, `has_ciphertexts()` and `self_check()` are
        # blocking sync I/O — run them off the event loop so `wait_for` gets
        # a real cancellation point.
        def _probe() -> tuple[str, str]:
            # Classify the path FIRST — never call anything that would create it.
            shape = _key_path_shape(manager.key_path)
            if shape == "other":
                # Something occupies the key path but is not a key file: a
                # directory, a dangling symlink, a FIFO, an entry that cannot
                # be stat'd. `is_file()` answers False for every one of them,
                # so the lazy-creation branch below would call this reassuring
                # `skipped` on a node where the very first secret write is
                # guaranteed to fail (O_EXCL raises EEXIST on any of them).
                # It is a misconfigured path, not a fresh install — `fail`.
                detail = (
                    "the secrets key path exists but is not a regular file "
                    "(a directory, a dangling symlink, a FIFO, or an entry "
                    "the daemon cannot stat) — the key can be neither read nor "
                    "created there, so secret writes and injection will fail; fix "
                    "[security].secrets_key_file or clear what sits at that path"
                )
                # Same misconfiguration, strictly worse node: material already
                # at rest is unreadable until the path is fixed. Say the more
                # urgent half too rather than choosing between them.
                if manager.has_ciphertexts():
                    detail += (
                        "; encrypted secrets already at rest cannot be decrypted "
                        "until that path is fixed"
                    )
                return "fail", detail
            if shape == "absent":
                # An absent key is only a fault when there is encrypted material
                # it was meant to open. The key is minted lazily on the first
                # secret write, so a node that has never stored a secret has no
                # key by design — reporting that as `fail` told every fresh
                # install that something was broken. `skipped`, like the other
                # "this facility has nothing to report yet" rows, so it also
                # never worsens the top status.
                #
                # The detail says "encrypted secrets", not "secrets": legacy
                # plaintext is deliberately not counted (it is
                # recoverable without the key), so claiming "no secrets stored"
                # would be false on a node whose migration left a straggler.
                if not manager.has_ciphertexts():
                    return "skipped", (
                        "no encrypted secrets at rest yet — the key is created "
                        "on the first secret write"
                    )
                # Encrypted material with no key to open it: real, unrecoverable
                # loss. Stays `fail`, and says so. `has_ciphertexts` also
                # answers True when the store exists but cannot be listed — an
                # unenumerable store is never reported as an empty one.
                return "fail", (
                    "encrypted secrets exist (or the store is unreadable) but the "
                    "key file is missing — they cannot be decrypted; restore the "
                    "key from a backup"
                )
            # A regular file: reading it cannot block, so `self_check` is safe.
            try:
                ok = manager.self_check()
            except SecretDecryptError:
                # Unreadable (EACCES), empty, truncated or non-hex. Handled here
                # rather than left to `_run_check`'s blanket handler, whose
                # "check failed or timed out" teaches the operator nothing and
                # reads like a daemon bug. The exception message carries the
                # path, so it is deliberately NOT echoed (doctor discipline).
                return "fail", (
                    "the secrets key file exists but could not be read as a key — "
                    "it is unreadable (permissions), empty, or malformed; every "
                    "secret operation will fail; restore the key from a backup"
                )
            if ok:
                return "ok", "secrets key present and usable"
            return "fail", "secrets key round-trip failed"

        return await asyncio.to_thread(_probe)

    async def _disk() -> tuple[str, str]:
        data_dir = Path(settings.data_dir).expanduser()

        def _probe() -> tuple[str, str]:
            usage = shutil.disk_usage(data_dir)
            free_pct = (usage.free / usage.total) if usage.total else 0.0
            pct = round(free_pct * 100, 1)
            if free_pct < 0.05:
                return "fail", f"{pct}% free"
            if free_pct < 0.10:
                return "warn", f"{pct}% free"
            return "ok", f"{pct}% free"

        return await asyncio.to_thread(_probe)

    async def _data_dir_perms() -> tuple[str, str]:
        # The named-volume leaf dirs are created world-
        # writable+sticky (0o1777) so non-root container users can write; that is
        # safe ONLY because `data_dir` and `<data_dir>/services` are meant to
        # be private (0o700). Warn when `data_dir` itself is group/world
        # accessible, so a mis-permissioned host dir cannot silently expose the
        # sticky leaves to other local users. Mode bits only — no secret values.
        data_dir = Path(settings.data_dir).expanduser()

        def _probe() -> tuple[str, str]:
            try:
                mode = os.stat(data_dir).st_mode & 0o777
            except OSError:
                return "skipped", "data_dir does not exist yet"
            if mode & 0o077:
                return "warn", f"data_dir is group/world-accessible (mode {mode:04o})"
            return "ok", f"data_dir is private (mode {mode:04o})"

        return await asyncio.to_thread(_probe)

    async def _db() -> tuple[str, str]:
        db = getattr(state, "db", None)
        if db is None:
            return "fail", "database not initialised"
        db_file = Path(settings.data_dir).expanduser() / DEFAULT_DB_NAME

        # Run `quick_check` on a short-lived DEDICATED read-only connection off
        # the event loop — never on the single shared aiosqlite connection, whose
        # O(db-size) integrity scan would monopolise it past the 2 s budget and
        # stall every other query in the daemon.
        def _probe() -> tuple[str, str]:
            if not db_file.is_file():
                return "fail", "database file missing"
            size = db_file.stat().st_size
            conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
            try:
                row = conn.execute("PRAGMA quick_check").fetchone()
            finally:
                conn.close()
            result = row[0] if row else None
            if result == "ok":
                return "ok", f"{size} bytes, quick_check ok"
            return "fail", "quick_check reported integrity errors"

        return await asyncio.to_thread(_probe)

    async def _git() -> tuple[str, str]:
        if not settings.git.enabled:
            return "skipped", "git deploy disabled"
        # `git_available()` shells out to `shutil.which` — run it off the loop.
        if await asyncio.to_thread(git_available):
            return "ok", "git binary on PATH"
        return "warn", "git enabled but the git binary is not on PATH"

    async def _config_restart_pending() -> tuple[str, str]:
        # Key NAMES only — config values (paths, hosts, addresses) must not leak.
        # `load_settings()` is file I/O + a full pydantic build, so run it off
        # the event loop.
        #
        # Normalize the fresh copy the way the lifespan normalized the boot one:
        # `normalize_mount_roots` appends `[daemon].upload_dir` IN PLACE to
        # `containers.allowed_mount_roots` before that object becomes
        # `app.state.settings`. Comparing it against an un-normalized re-read
        # would report a permanent, restart-proof drift on every install whose
        # upload dir is not already in the list — untruthful in exactly the
        # direction this check exists to avoid, and it would train the operator
        # to ignore the row that flags a real `auth_token`/`link.*` change.
        fresh = await asyncio.to_thread(load_settings)
        # Tolerant like the `link` check below: a lightweight settings object
        # (test harnesses, a stubbed loader) may not carry either section, and
        # a doctor check must never turn that into a `fail`.
        if hasattr(fresh, "daemon") and hasattr(fresh, "containers"):
            normalize_mount_roots(fresh)
        pending: list[str] = []
        for section, keys in _RESTART_KEYS.items():
            old_sec = restart_section_holder(state.settings, section)
            new_sec = restart_section_holder(fresh, section)
            for key in keys:
                if getattr(old_sec, key, None) != getattr(new_sec, key, None):
                    pending.append(f"{section}.{key}")
        if pending:
            return "warn", "pending restart: " + ", ".join(sorted(pending))
        return "ok", "no restart-required config drift"

    async def _link() -> tuple[str, str]:
        # The remote-access tunnel's health surface.
        # `status()` is a pure in-memory snapshot, so this check does no I/O
        # and lands well inside the 2 s budget. Details carry state, counters
        # and close codes only — never the relay URL, the key path, or the
        # capability.
        manager = getattr(state, "link_manager", None)
        if manager is None:
            # Tolerant like the mcp flag: a lightweight settings
            # object may not carry [link] at all; absence means off.
            link_settings = getattr(settings, "link", None)
            if not getattr(link_settings, "enabled", False):
                # Warn only when no node_id indicates the node is unlinked. Disabled linking with
                # an existing identity is a deliberate opt-out and stays quiet. Read config only;
                # doctor is advisory and local services remain fully usable. Name missing remote
                # features and the recovery command without URLs or key material.
                if getattr(link_settings, "node_id", None) is None:
                    return (
                        "warn",
                        "not linked — no Nerdit account attached; "
                        "run 'nerdit link --device' (free)",
                    )
                return "skipped", "link disabled"
            return "warn", "link enabled but not linked or identity unavailable"
        snapshot = manager.status()
        if snapshot.state == "connected":
            expires_at = snapshot.capability_expires_at
            renews_in = 0
            if expires_at is not None:
                renews_in = max(0, int((expires_at - datetime.now(UTC)).total_seconds()))
            return "ok", f"connected; capability renews in {renews_in}s"
        if snapshot.state == "terminal":
            reason = snapshot.terminal_reason or "unknown"
            hint = _LINK_TERMINAL_HINTS.get(reason)
            if hint is not None:
                return "fail", f"terminal: {reason}; {hint}"
            return "fail", f"terminal: {reason}"
        return (
            "warn",
            f"reconnecting (attempt {snapshot.attempt}, last close {snapshot.last_close_code})",
        )

    async def _github_token() -> tuple[str, str]:
        # The mirrored GitHub installation tokens. A pure
        # `status()` / manager read like the link row. `ok` while a token
        # is live, `warn` when the soonest LIVE expiry is inside
        # `GITHUB_TOKEN_WARN_LEAD_S` (the pusher is falling behind) OR
        # when the mirror HELD a token that has since lapsed with no re-mint —
        # GitWatch's quiet-backoff case, which must not read as a green
        # never-linked daemon while every private auto-deploy has silently
        # stopped — and `skipped` only when the mirror is genuinely empty (no
        # cloud, not linked, cold reconnect, or the online→offline edge cleared
        # it — absence is quiet). The detail carries a count and a lead/age —
        # never the token, never an installation id, never a repo name.
        manager = getattr(state, "link_manager", None)
        if manager is None:
            return "skipped", "no GitHub installation token mirrored"
        installations = tuple(getattr(manager.status(), "github_installations", ()))
        if installations:
            soonest = min(inst.expires_at for inst in installations)
            lead = int((soonest - manager.now()).total_seconds())
            count = len(installations)
            noun = "installation" if count == 1 else "installations"
            if lead < GITHUB_TOKEN_WARN_LEAD_S:
                return (
                    "warn",
                    f"{count} {noun}; token expires in {max(lead, 0)}s — cloud push overdue",
                )
            return "ok", f"{count} {noun}; token expires in {lead}s"
        # Nothing live. Distinguish a mirror that HELD a token and let it lapse
        # (silent pusher — surface it as `warn` so a stalled auto-deploy is
        # visible) from a genuinely empty one (`skipped`). The held read is
        # over the raw mirror, which the live surfaces never see.
        held_count, last_expiry = manager.github_mirror_held()
        if held_count == 0 or last_expiry is None:
            return "skipped", "no GitHub installation token mirrored"
        overdue = int((manager.now() - last_expiry).total_seconds())
        noun = "installation" if held_count == 1 else "installations"
        return (
            "warn",
            f"{held_count} {noun} held; token expired {max(overdue, 0)}s ago — cloud push overdue",
        )

    async def _license() -> tuple[str, str]:
        # The offline product license. A pure `app.state` read —
        # the file was read at boot (or at install), so this check does ZERO I/O
        # and lands trivially inside the 2 s budget — while the *temporal* state
        # is recomputed live from the stored claims and the holder's clock, so a
        # grace transition surfaces with no poller and no restart.
        #
        # Details carry the state, the machine reason token, the plan token and
        # day counts. Never a path, never `customer_id`, never a byte of the
        # blob: this body is safe for every role, like every other check here.
        # "Warn during grace, FAIL after" lands exactly here — the tunnel's
        # fate belongs to the relay, which is why the entitlement decision at
        # the [link] seam is advisory.
        holder = getattr(state, "license", None)
        if holder is None or not holder.installed:
            return "skipped", "no license installed"
        license_state = holder.state
        claims = holder.claims
        if license_state == STATE_INVALID or claims is None:
            return "fail", f"license invalid ({holder.reason}) — reinstall a valid license"
        if license_state == STATE_EXPIRED:
            return "fail", "expired — grace period over; renew and reinstall"

        seconds = (claims.expires_at - holder.clock()).total_seconds()
        if license_state == STATE_EXPIRED_GRACE:
            status = "warn"
            detail = (
                f"expired {_whole_days(-seconds)}d ago — grace ends in "
                f"{_whole_days(seconds + LICENSE_GRACE_S)}d; renew"
            )
        else:
            status = "ok"
            detail = f"valid — plan {claims.plan}, expires in {_whole_days(seconds)}d"

        # The feature gap is only meaningful when a tunnel is actually
        # configured, and it is APPENDED rather than substituted: an in-grace
        # license that also lacks the feature has two problems, and dropping
        # either half would send the operator to fix only one of them.
        link_enabled = getattr(state, "link_manager", None) is not None or getattr(
            getattr(settings, "link", None), "enabled", False
        )
        if link_enabled and FEATURE_REMOTE_LINK not in claims.features:
            status = "warn"
            detail = f"{detail}; does not cover {FEATURE_REMOTE_LINK}"
        return status, detail

    async def _acme_http_port() -> tuple[str, str]:
        # The one listener this daemon asks the OUTSIDE world
        # to reach: the ACME HTTP-01 challenge port. Every other doctor row
        # reports something local; this one is the difference between "the CA
        # can validate us" and a domain stuck `pending` forever with no
        # machine-readable reason anywhere (certmagic writes no failure record).
        #
        # `getattr` on both hops so a settings object without the nested
        # `[proxy.acme]` section reads as disabled rather than raising.
        acme = getattr(settings.proxy, "acme", None)
        if not getattr(acme, "enabled", False):
            return "skipped", "ACME disabled"
        port = int(getattr(acme, "http_port", 80))

        # An available proxy is *almost* the bind fact: Caddy refuses to start
        # when any configured listener fails to bind, and when ACME is on the
        # bootstrap carries the `:<http_port>` server. So there is nothing left
        # to probe — and probing would collide with our own listener and report
        # EADDRINUSE against ourselves. The "almost" is adoption: a Caddy this
        # process did not spawn can be adopted on the strength of its `nerdit`
        # server alone and carry no ACME listener at all, which is why the
        # manager latches the answer (`acme_listener_live`) instead of letting
        # two surfaces each infer it from availability (review round 1).
        manager = getattr(state, "proxy_manager", None)
        proxy_state = getattr(manager, "state", None)
        if proxy_state == ProxyState.available:
            if getattr(manager, "acme_listener_live", True) is False:
                return "fail", (
                    f"the running proxy carries no ACME listener on http port {port} "
                    "— it was started before [proxy.acme] was enabled; restart the daemon"
                )
            return "ok", f"http port {port} bound by the embedded proxy"

        if proxy_state == ProxyState.starting:
            # Caddy is spawned and has very likely already bound the port; a
            # probe here reports EADDRINUSE against OURSELVES and calls it
            # "another process", which is both wrong and a `fail` that drags
            # the top-level doctor status down on every boot with ACME on.
            return "warn", f"proxy is starting — http port {port} not yet confirmed"

        # The proxy is starting / in backoff / has no binary / lost the admin
        # port to a foreigner. Whether the port itself is available is now an
        # independent, actionable fact, and the common privileged-port failure
        # (`:80` without CAP_NET_BIND_SERVICE) is invisible in every other
        # row. A bind-and-close is not a filesystem write and lands far inside
        # the 2 s budget; no `SO_REUSEADDR`, so a socket another process holds
        # is reported as held rather than silently shared.
        def _probe() -> tuple[str, str]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", port))
            except PermissionError:
                # EACCES. The fix is a capability or a config change, never a
                # retry — name both, and name the config KEY, not a path.
                #
                # Carefully hedged (review round 1): the probe runs in the DAEMON
                # process and cannot see the caddy binary's file capabilities. On
                # a --user install where caddy already HAS
                # CAP_NET_BIND_SERVICE, this branch is reached for an unrelated
                # reason (the proxy is in backoff over a bad ca_root_file, an
                # admin-port clash…) and the old text sent the operator to fix a
                # capability that was already in place. Name what was actually
                # observed first, then both fixes.
                return "fail", (
                    f"this daemon cannot bind http port {port} (EACCES). If the caddy "
                    "binary already has CAP_NET_BIND_SERVICE, see the proxy check for "
                    "the real cause; otherwise grant it (setcap, or a system unit with "
                    "AmbientCapabilities) or set [proxy.acme].http_port above 1023"
                )
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    return "fail", f"http port {port} is held by another process"
                name = errno.errorcode.get(exc.errno or 0, "unknown error")
                return "fail", (
                    f"http port {port} cannot be bound ({name}) — check [proxy.acme].http_port"
                )
            finally:
                sock.close()
            return "warn", (
                f"http port {port} is bindable but the proxy is not up — see the proxy check"
            )

        return await asyncio.to_thread(_probe)

    checks = await asyncio.gather(
        _run_check("docker", _docker),
        _run_check("gpu", _gpu),
        _run_check("proxy", _proxy),
        _run_check("mdns", _mdns),
        _run_check("secrets_key", _secrets_key),
        _run_check("disk", _disk),
        _run_check("data_dir_perms", _data_dir_perms),
        _run_check("db", _db),
        _run_check("git", _git),
        _run_check("config_restart_pending", _config_restart_pending),
        _run_check("link", _link),
        _run_check("license", _license),
        _run_check("acme_http_port", _acme_http_port),
        _run_check("github_token", _github_token),
    )

    top = "ok"
    for check in checks:
        rank = _STATUS_RANK.get(check["status"])
        if rank is not None and rank > _STATUS_RANK[top]:
            top = check["status"]
    return {"status": top, "checks": checks}


# --- POST /daemon/restart -----------------------------------------------------


class RestartRequest(StrictRequestModel):
    """Optional body for `POST /daemon/restart`."""

    drain_timeout_s: int = Field(60, description="Seconds to drain in-flight builds ([0, 300]).")


@router.post("/daemon/restart", operation_id="restart_daemon", status_code=202)
async def restart_daemon(request: Request, body: RestartRequest | None = None) -> dict[str, Any]:
    """Gracefully drain in-flight builds, then self-exec (1.7).

    Admin-only, mandatory in-route `Idempotency-Key` (the config-PUT
    precedent). Two audit rows land: this route writes `daemon.restart`
    **directly before** spawning the drain task (so the row is committed even at
    `drain_timeout_s=0` before the SIGTERM the middleware's own post-response
    record cannot outrun), and `AuditMiddleware` fires the standard mutation
    row on the way out — deterministic ordering, two rows by design.
    """
    require_role(request, TokenRole.admin)

    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            "A daemon restart requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the restart is safe to retry.",
        )

    # Check-and-set with NO await in between (single-thread atomicity): a second
    # concurrent restart request loses the race and gets a 409.
    if is_restart_requested():
        raise NerditError(
            409,
            "daemon.restart_in_progress",
            "A daemon restart is already in progress.",
            hint="Wait for the daemon to come back up; poll GET /capabilities.uptime_s.",
        )
    request_restart()

    drain_timeout_s = 60 if body is None else max(0, min(_DRAIN_TIMEOUT_MAX, body.drain_timeout_s))

    controller = getattr(request.app.state, "service_controller", None)

    # Close the drain gate HERE — still inside the await-free window opened by
    # the check-and-set above, and BEFORE the counts are taken. `POST /run`
    # gates on `controller.draining`, not on the restart flag, so leaving the
    # gate open across the awaited audit insert below would admit a run into an
    # already-requested restart: it would be missing from both the 202 body and
    # the audit row (both computed from counts taken before it arrived), and at
    # `drain_timeout_s=0` the deadline path would kill it immediately while the
    # operator had just been told nothing was in flight. Gate first, then count,
    # and the numbers we report are the numbers we drain.
    if controller is not None:
        controller.draining = True

    in_flight = controller.busy_builds() if controller is not None else 0
    # Rowless run/release containers are not builds, so they need their own
    # count — the drain waits for both. Counts only, never a run id or argv.
    in_flight_runs = controller.busy_runs() if controller is not None else 0
    # A cutover verify is a third kind of in-flight work: a green
    # container is already running and about to be promoted. Counted (and
    # drained) beside the other two so a re-exec cannot land mid-promotion.
    in_flight_cutovers = controller.busy_cutovers() if controller is not None else 0

    # The 409 gate above is set now; everything up to the drain spawn is
    # transactional against that flag. If the fallible audit insert raises, clear
    # the flag (and any drain gate) and re-raise — otherwise a stuck flag would
    # wedge the daemon into a permanent 409 AND make main()'s re-exec branch fire
    # on the next clean shutdown, resurrecting it against the operator's intent.
    # The locked "audit committed before SIGTERM" contract is unaffected: the
    # SIGTERM happens in the drain task, spawned only after this section succeeds.
    global _drain_task
    principal = current_principal(request)
    queries = request.app.state.queries
    try:
        # Commit the audit row BEFORE spawning the drain — at drain_timeout_s=0
        # the SIGTERM fires immediately and the middleware's post-response record
        # cannot be guaranteed to win the race.
        await queries.insert_audit_log(
            action="daemon.restart",
            result="ok",
            principal_id=principal.token_id,
            principal_role=principal.role.value,
            target_type="daemon",
            params_redacted=json.dumps(
                {
                    "drain_timeout_s": drain_timeout_s,
                    "in_flight_builds": in_flight,
                    "in_flight_runs": in_flight_runs,
                    "in_flight_cutovers": in_flight_cutovers,
                }
            ),
            status_code=202,
            request_id=request_id_of(request),
            idempotency_key=request.headers.get("Idempotency-Key"),
        )

        # Retain the task handle (module-level) so it is not GC'd mid-drain.
        _drain_task = asyncio.create_task(_drain_and_restart(controller, drain_timeout_s))
    except Exception:
        reset_restart_requested()
        if controller is not None:
            controller.draining = False
        raise

    return {
        "restarting": True,
        "in_flight_builds": in_flight,
        "in_flight_runs": in_flight_runs,
        "in_flight_cutovers": in_flight_cutovers,
        "drain_timeout_s": drain_timeout_s,
    }


async def _drain_and_restart(controller, drain_timeout_s: int) -> None:  # noqa: ANN001
    """Drain builds and runs until the deadline, then request daemon re-execution.

    New deploys stay queued for the next daemon. At timeout kill active run/release
    containers but retain them for scrubbed log capture; leave internal builds for
    restart recovery. Uvicorn's graceful-shutdown bound covers remaining handlers,
    including run slots without containers. Report both build and run counts.

    Request shutdown in finally even if Docker cleanup fails; otherwise draining
    and restart_in_progress could permanently wedge the daemon. Prefer should_exit
    on the registered server, with SIGTERM as fallback.
    """
    deadline = time.monotonic() + drain_timeout_s
    try:
        if controller is not None:
            while (
                controller.busy_builds() + controller.busy_runs() + controller.busy_cutovers()
            ) > 0 and (time.monotonic() < deadline):
                await asyncio.sleep(_DRAIN_POLL_SECONDS)
            # Deadline expired with work still in flight → make the bound real.
            # Counts only in the log line; the kill loop logs container ids at most.
            remaining_builds = controller.busy_builds()
            remaining_runs = controller.busy_runs()
            remaining_cutovers = controller.busy_cutovers()
            if (remaining_builds + remaining_runs + remaining_cutovers) > 0:
                logger.warning(
                    "Restart drain deadline expired with %d build(s), %d run(s) and %d "
                    "cutover(s) in flight; killing rowless containers",
                    remaining_builds,
                    remaining_runs,
                    remaining_cutovers,
                )
                # Runs/releases ∪ cutover greens: a surviving green would
                # keep answering a repointed dial the row does not name.
                await controller.kill_transient_containers()
    except Exception:
        logger.exception("Restart drain failed before the shutdown step; restarting anyway")
    finally:
        if _uvicorn_server is not None:
            # Programmatic stop: same graceful shutdown as a signal, but with no
            # captured signal for uvicorn to replay after run() — so main()'s
            # re-exec branch actually runs (see the _uvicorn_server note above).
            _uvicorn_server.should_exit = True
        else:
            os.kill(os.getpid(), signal.SIGTERM)


# --- GET /system/disk + POST /system/gc ---------------------------------------
#
# Disk accounting composes docker's own `df` aggregate (the only source for
# docker-side totals) with `du` walks over the bind-mount trees docker cannot
# see (named-volume data dirs, model weights, archive, backups). GC reclaims
# only **orphan** `nerdit-app/*` image repos (repos with no live workload row)
# and, opt-in, orphan service data dirs — both behind a fresh TOCTOU re-check
# taken immediately before every destructive step, since a racing deploy can
# re-create a row/dir between the snapshot and the mutation.

# Soft budget for the concurrent du walks (own route, own budget — not doctor's
# 2 s regime). On overrun the body is returned partial with a `scan_timeout`
# warning rather than blocking on a slow/huge tree.
_DISK_SCAN_BUDGET_S = 10.0

# A single concurrent gc at a time (module-level, consulted via `.locked()`).
# The gc body awaits, so a plain bool check-and-set would not serialize it; an
# asyncio.Lock does. A different-key idempotency replay does not stop a second
# gc, so this guard is the real mutual exclusion.
_gc_lock = asyncio.Lock()
_disk_lock = asyncio.Lock()
# A single concurrent backup at a time (module-level, consulted via `.locked()`
# — the gc precedent). The `locked()` pre-check is best-effort: two requests can
# both pass it before either acquires, so the loser runs a second backup serially.
# "never a second concurrent VACUUM" holds; "always 409 on overlap" does not.
_backup_lock = asyncio.Lock()


def _service_dir_names(services_root: Path) -> list[str]:
    """Cheap (no-recursion) sorted list of the subdir names under *services_root*."""
    try:
        return sorted(e.name for e in os.scandir(services_root) if e.is_dir(follow_symlinks=False))
    except OSError:
        return []


def _walk_services(services_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Per-service `{name, bytes}` list + total for `<data_dir>/services`."""
    out: list[dict[str, Any]] = []
    total = 0
    try:
        entries = sorted(os.scandir(services_root), key=lambda e: e.name)
    except OSError:
        return out, total
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        size = du_bytes(Path(entry.path))
        out.append({"name": entry.name, "bytes": size})
        total += size
    return out, total


def _walk_workspaces(workspaces_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Per-workspace `{name, bytes}` list + total for `<data_dir>/workspaces`.

    A literal twin of `_walk_services`: the agent workspaces are a second
    daemon-owned tree that grows without a container ever mounting it, so it
    must not be invisible to disk observability. Absent dir ⇒ empty list + 0,
    never an error (the root is created lazily at the first write).
    """
    out: list[dict[str, Any]] = []
    total = 0
    try:
        entries = sorted(os.scandir(workspaces_root), key=lambda e: e.name)
    except OSError:
        return out, total
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        size = du_bytes(Path(entry.path))
        out.append({"name": entry.name, "bytes": size})
        total += size
    return out, total


def _walk_tars(backups_dir: Path, glob: str) -> dict[str, int]:
    """`{bytes, count}` for the `glob` tars in `backups_dir` (absent dir ⇒ 0/0).

    Each tar flavour (control-plane, volume, dump) is its own disk-report
    bucket; their globs are disjoint, so no bucket double-counts another's.
    """
    total = 0
    count = 0
    try:
        entries = list(os.scandir(backups_dir))
    except OSError:
        return {"bytes": 0, "count": 0}
    for entry in entries:
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            if not fnmatch.fnmatch(entry.name, glob):
                continue
            total += entry.stat(follow_symlinks=False).st_size
            count += 1
        except OSError:
            continue
    return {"bytes": total, "count": count}


def _backups_over_keep(backups_dir: Path, keep: int) -> int:
    """How many backup archives sit beyond `backup_keep_last` (0 when keep≤0)."""
    if keep <= 0:
        return 0
    return max(0, _walk_tars(backups_dir, "nerdit-backup-*.tar.gz")["count"] - keep)


def _effective_archive_dir(settings: Any, data_dir: Path) -> Path | None:
    """The audit-archive dir the retention SWEEP actually writes to.

    Delegates to the sweep's own guard (`utils.disk.resolve_archive_dir`) so the
    report can never diverge from the writer: a custom
    `[retention].audit_archive_dir` is the dir that gets walked, and a value the
    guard rejects means archiving is DISABLED (`None`) ⇒ the route reports
    `archive_bytes: 0` with no error, which is the honest count. Walking
    `<data_dir>/archive` unconditionally would report a
    stale/absent default while the real archive grew unwatched.
    """
    retention = getattr(settings, "retention", None)
    if retention is None:
        return data_dir / "archive"
    return resolve_archive_dir(retention, data_dir)


def _instance_id(settings: Any) -> str:
    """This daemon's `[daemon].instance_id` — the image-GC ownership scope.

    Mirrors `_effective_archive_dir`'s tolerance: a settings object with no
    `daemon` section (or an empty id, which the field validator forbids in
    production) falls back to the same `"default"` the setting itself does, so
    a single-daemon host behaves exactly as before.
    """
    daemon = getattr(settings, "daemon", None)
    value = getattr(daemon, "instance_id", None)
    return value if isinstance(value, str) and value else DEFAULT_INSTANCE_ID


async def _bounded_walks(
    walks: dict[str, Callable[[], Any]], budget: float
) -> tuple[dict[str, Any], bool]:
    """Run each blocking walk off-loop concurrently under a soft *budget*.

    Returns `(results, timed_out)`: a completed walk's value is keyed by name;
    a walk that overran the deadline (or raised) is `None` and flips
    `timed_out` (the caller surfaces a `scan_timeout` warning). Overrunning
    threads are detached daemon threads, not awaited — the budget is soft on
    purpose, and detachment must never touch the default executor (see
    `utils.disk.spawn_walk`).
    """
    futures = {name: spawn_walk(fn) for name, fn in walks.items()}
    if futures:
        await asyncio.wait(list(futures.values()), timeout=budget)
    results: dict[str, Any] = {}
    timed_out = False
    for name, fut in futures.items():
        if fut.done():
            try:
                results[name] = fut.result()
            except Exception:  # noqa: BLE001 — a diagnostic walk never propagates
                results[name] = None
                timed_out = True
        else:
            fut.cancel()
            results[name] = None
            timed_out = True
    return results, timed_out


@router.get("/system/disk", operation_id="get_system_disk")
async def get_system_disk(request: Request) -> dict[str, Any]:
    """Report disk usage across docker + the bind-mount data trees (P14b WP-A2).

    Any authenticated principal (doctor posture). The body carries **names and
    byte counts only — zero absolute paths** (readonly visibility is accepted).
    `docker` is the `df` aggregate (`None` when docker is unreachable — the
    bind-mount trees are still reported); `orphan_images` /`orphan_data_dirs`
    are report-only here (GC acts on them). `orphan_images` is instance-scoped
    exactly like the GC that consumes it, so the report can never advertise an
    image the gc would then refuse to touch. The du walks run under a soft
    `scan_timeout` budget ⇒ partial body + a warning rather than blocking.
    """
    state = request.app.state
    settings = state.settings
    runtime = state.runtime
    queries = state.queries
    data_dir = Path(settings.data_dir).expanduser()

    # Serialize the du walks: each request can detach up to five tree walkers
    # onto the shared default executor for the budget window; unserialized, a
    # readonly-token hammer could starve the pool DockerRuntime and the
    # reconciler depend on. Waiters queue briefly instead (budget-bounded).
    async with _disk_lock:
        return await _build_disk_report(
            runtime,
            queries,
            data_dir,
            _effective_archive_dir(settings, data_dir),
            _instance_id(settings),
        )


async def _build_disk_report(
    runtime: Any,
    queries: Any,
    data_dir: Path,
    archive_dir: Path | None,
    instance_id: str = DEFAULT_INSTANCE_ID,
) -> dict[str, Any]:
    warnings: list[str] = []
    docker = await runtime.disk_usage()
    detailed = await runtime.list_images_detailed()
    rows = await queries.list_workload_configs()

    by_repo, images_total = _attribute_images(detailed)
    tags = [e.get("repo_tag") or "" for e in detailed]
    protected = _protected_image_refs(rows, tags)
    live_repos = _live_repos(rows)
    orphan_images = _orphan_app_repos(detailed, live_repos, protected, instance_id)

    services_root = data_dir / "services"
    dir_names = await asyncio.to_thread(_service_dir_names, services_root)
    live_names = _live_service_names(rows)
    orphan_data_dirs = _orphan_data_dir_names(dir_names, live_names)

    walks: dict[str, Callable[[], Any]] = {
        "services": lambda: _walk_services(services_root),
        "workspaces": lambda: _walk_workspaces(data_dir / "workspaces"),
        "ollama": lambda: du_bytes(data_dir / "models" / "ollama"),
        "huggingface": lambda: du_bytes(data_dir / "models" / "huggingface"),
        "backups": lambda: _walk_tars(data_dir / "backups", "nerdit-backup-*.tar.gz"),
        "volume_backups": lambda: _walk_tars(data_dir / "backups", "nerdit-volumes-*.tar.gz"),
        "dumps": lambda: _walk_tars(data_dir / "backups", DUMP_TAR_GLOB),
        # Leftover staging dirs are invisible disk: a crash
        # between a dump's staging mkdir and its ``finally`` leaves one behind
        # until the next boot sweep. Counted so an operator can SEE that.
        "dump_staging": lambda: du_bytes(dump_staging_root(data_dir)),
    }
    # Only walk the archive when archiving is actually ENABLED. A guard-rejected
    # audit_archive_dir means nothing is written there, so `0` is the honest
    # count — and it must be 0, not `None`: a `None` there is the walk's
    # "timed out / failed" value and would be indistinguishable from a partial
    # scan to every consumer.
    if archive_dir is not None:
        walks["archive"] = lambda: du_bytes(archive_dir)
    results, timed_out = await _bounded_walks(walks, _DISK_SCAN_BUDGET_S)
    if timed_out:
        warnings.append("scan_timeout")

    services_result = results.get("services")
    if services_result is None:
        services_list: list[dict[str, Any]] = []
        services_total: int | None = None
    else:
        services_list, services_total = services_result

    # Same None-on-timeout handling as services: an empty list + a null
    # total says "the walk did not finish", never "there is nothing there".
    workspaces_result = results.get("workspaces")
    if workspaces_result is None:
        workspaces_list: list[dict[str, Any]] = []
        workspaces_total: int | None = None
    else:
        workspaces_list, workspaces_total = workspaces_result

    return {
        "docker": docker,
        "images": {"total_bytes": images_total, "by_repo": by_repo},
        "data_dir": {
            "services": services_list,
            "services_total_bytes": services_total,
            # The agent workspace trees, the services grammar exactly.
            "workspaces": workspaces_list,
            "workspaces_total_bytes": workspaces_total,
            "models": {
                "ollama": results.get("ollama"),
                "huggingface": results.get("huggingface"),
            },
            # Bytes only — never the resolved path (the no-absolute-paths posture
            # holds even though the dir is admin-configured).
            "archive_bytes": results.get("archive") if archive_dir is not None else 0,
            "backups": results.get("backups") or {"bytes": 0, "count": 0},
            # Per-database volume tars: a disjoint glob from `backups`,
            # default-0 accumulate-forever, so counted explicitly here.
            "volume_backups": results.get("volume_backups") or {"bytes": 0, "count": 0},
            # Logical dump tars — a third disjoint glob, swept per service
            # by ``[retention].dump_keep_last`` (default 5).
            "dumps": results.get("dumps") or {"bytes": 0, "count": 0},
            # Bytes still sitting in ``<data_dir>/dump-staging/``; ``null`` when
            # the walk timed out (the ``models``/``archive`` convention).
            "dump_staging_bytes": results.get("dump_staging"),
        },
        "orphan_images": orphan_images,
        "orphan_data_dirs": orphan_data_dirs,
        "warnings": warnings,
    }


class GcRequest(StrictRequestModel):
    """Optional body for `POST /system/gc`."""

    include_orphan_data: bool = Field(
        False,
        description="Also rmtree service data dirs with no live row (irreversible).",
    )


@router.post("/system/gc", operation_id="run_system_gc", status_code=200)
async def run_system_gc(
    request: Request,
    body: GcRequest | None = None,
    dry_run: bool = Query(
        False, description="Enumerate what would be reclaimed without removing anything."
    ),
) -> dict[str, Any]:
    """Reclaim orphan app images (+ opt-in orphan data dirs) (P14b WP-A2).

    Admin-only. `?dry_run=true` enumerates candidates with zero writes (audited
    `system.gc_plan`, Idempotency-Key ignored by the middleware); a real run is
    audited `system.gc` and honors an Idempotency-Key. A second concurrent gc
    gets `409 system.gc_in_progress`; a docker daemon that cannot report `df`
    gets `503 system.docker_unavailable` (dry-run included). Every destructive
    step re-reads the workload rows immediately beforehand (TOCTOU) and image
    removals are verified by re-listing — a tag still present ⇒ reported skipped,
    never claimed removed. Image reclaim is scoped to THIS daemon's
    `[daemon].instance_id`: a repo is a candidate only when EVERY one of its
    tags carries our own `nerdit-instance` label — a single foreign OR
    unlabelled tag refuses the whole repo — so co-located daemons never reclaim
    each other's images, and a repo carrying any pre-label tag is left alone
    entirely until those tags are removed by hand.
    """
    require_role(request, TokenRole.admin)
    include_orphan_data = bool(body.include_orphan_data) if body is not None else False
    if dry_run:
        request.state.audit_action = "system.gc_plan"

    if _gc_lock.locked():
        raise NerditError(
            409,
            "system.gc_in_progress",
            "A garbage-collection run is already in progress.",
            hint="Wait for the current gc to finish before starting another.",
        )

    async with _gc_lock:
        return await _run_gc(request, dry_run=dry_run, include_orphan_data=include_orphan_data)


async def _run_gc(request: Request, *, dry_run: bool, include_orphan_data: bool) -> dict[str, Any]:
    state = request.app.state
    settings = state.settings
    runtime = state.runtime
    queries = state.queries
    data_dir = Path(settings.data_dir).expanduser()

    # `df` returning None is the ONLY honest docker-unavailable signal: the
    # image-list methods degrade every error to `[]`, so a dead dockerd is
    # indistinguishable from an empty image list through them.
    docker = await runtime.disk_usage()
    if docker is None:
        raise NerditError(
            503,
            "system.docker_unavailable",
            "Docker is unavailable; cannot garbage-collect.",
            hint="Start Docker and retry.",
        )

    detailed = await runtime.list_images_detailed()
    rows = await queries.list_workload_configs()
    tags = [e.get("repo_tag") or "" for e in detailed]
    protected = _protected_image_refs(rows, tags)
    live_repos = _live_repos(rows)
    # Ownership scope: only images this daemon built are reclaimable. A
    # co-located sibling daemon's `nerdit-app/*` images look orphan through our
    # DB (their rows live in the sibling's DB) — the instance label is what stops
    # us deleting them (container scoping, extended to images).
    orphan_repos = _orphan_app_repos(detailed, live_repos, protected, _instance_id(settings))

    images_removed: list[str] = []
    images_skipped: list[dict[str, str]] = []
    reclaim_estimate = 0

    if dry_run:
        images_removed = list(orphan_repos)
        reclaim_estimate = sum(
            _repo_size_estimate(detailed, _repo_tags(detailed, repo, protected))
            for repo in orphan_repos
        )
    else:
        for repo in orphan_repos:
            # TOCTOU re-check immediately before removal: a racing deploy may have
            # created a row referencing this repo (or stamped it mid-build). Only
            # the *rows* are re-read — `candidate_tags` still comes from the
            # scan-time `detailed` listing, which is the set we are deciding
            # about; freshness that matters lives in `fresh_rows`.
            fresh_rows = await queries.list_workload_configs()
            candidate_tags = [e.get("repo_tag") or "" for e in detailed]
            fresh_protected = _protected_image_refs(fresh_rows, candidate_tags)
            if repo in _live_repos(fresh_rows) or _repo_in_progress(fresh_rows, repo):
                images_skipped.append({"repo": repo, "reason": "in_use_or_error"})
                continue
            repo_tags = _repo_tags(detailed, repo, fresh_protected)
            for tag in repo_tags:
                await runtime.remove_image(tag)
            # `remove_image` swallows every error, so verify by re-listing: a
            # tag still present means the removal was refused (image in use).
            after = {e.get("repo_tag") for e in await runtime.list_images_detailed()}
            if any(tag in after for tag in repo_tags):
                images_skipped.append({"repo": repo, "reason": "in_use_or_error"})
            else:
                images_removed.append(repo)
                reclaim_estimate += _repo_size_estimate(detailed, repo_tags)

    data_removed: list[str] = []
    data_skipped: list[dict[str, str]] = []
    if include_orphan_data:
        services_root = data_dir / "services"
        dir_names = await asyncio.to_thread(_service_dir_names, services_root)
        candidates = _orphan_data_dir_names(dir_names, _live_service_names(rows))
        if dry_run:
            data_removed = candidates
        else:
            for name in candidates:
                # TOCTOU: destroying a data dir is irreversible user-data loss, so
                # re-read the rows and skip a name whose service now has ANY row.
                # For a C2 tombstone the live check keys on its BASE service — a
                # tombstone whose row reappeared belongs to the startup sweep, not
                # to GC (removing it would be the data loss C2 closes).
                fresh_rows = await queries.list_workload_configs()
                base = tombstone_service_name(name) or name
                if base in _live_service_names(fresh_rows):
                    data_skipped.append({"name": name, "reason": "now_referenced"})
                    continue
                try:
                    root = _orphan_data_dir_path(data_dir, name)
                    await asyncio.to_thread(shutil.rmtree, root)
                    data_removed.append(name)
                except (VolumeSpecError, OSError) as exc:
                    data_skipped.append({"name": name, "reason": type(exc).__name__})

    keep = int(getattr(getattr(settings, "retention", None), "backup_keep_last", 0) or 0)
    # The report-only walks take the SAME detached-daemon-thread + soft-budget path
    # as /system/disk. `asyncio.to_thread` borrows the loop's *default* executor,
    # and a du walk still running there at `asyncio.run` teardown is JOINED —
    # under uvloop unboundedly — so a gc overlapping a /daemon/restart while a huge
    # weights tree is walked wedges the re-exec. Overrun ⇒
    # null sizes + a `scan_timeout` warning, never a blocked loop. These values
    # are report-only: no destructive step depends on them.
    report_walks: dict[str, Callable[[], Any]] = {
        "ollama": lambda: du_bytes(data_dir / "models" / "ollama"),
        "huggingface": lambda: du_bytes(data_dir / "models" / "huggingface"),
        "backups_over_keep": lambda: _backups_over_keep(data_dir / "backups", keep),
    }
    walk_results, walks_timed_out = await _bounded_walks(report_walks, _DISK_SCAN_BUDGET_S)
    warnings: list[str] = ["scan_timeout"] if walks_timed_out else []
    reports = {
        "weights": {
            "ollama": walk_results.get("ollama"),
            "huggingface": walk_results.get("huggingface"),
        },
        "build_cache_bytes": int(docker.get("build_cache_bytes", 0) or 0),
        "backups_over_keep": walk_results.get("backups_over_keep"),
    }

    request.state.audit_params = audit_params(
        {
            "dry_run": dry_run,
            "include_orphan_data": include_orphan_data,
            "images_removed": images_removed,
            "images_skipped": [s["repo"] for s in images_skipped],
            "data_removed": data_removed,
            "data_skipped": [s["name"] for s in data_skipped],
        }
    )

    return {
        "dry_run": dry_run,
        "images": {
            "removed": images_removed,
            "skipped": images_skipped,
            "reclaimed_bytes_estimate": reclaim_estimate,
        },
        "orphan_data": {
            "enabled": include_orphan_data,
            "removed": data_removed,
            "skipped": data_skipped,
        },
        "reports": reports,
        # Same shape as the disk route's: a soft-budget overrun (or a failed walk)
        # degrades the report-only sizes to null and says so here.
        "warnings": warnings,
    }


# --- POST /system/backup -----------------------------------------------------

_BACKUP_CUSTODY_HINT = (
    "This archive contains the secrets master key (secrets.key). Copy it off-box "
    "and delete the local file; anyone holding it can read every stored secret."
)


@router.post("/system/backup", operation_id="create_backup", response_model=BackupResponse)
async def create_backup_route(request: Request) -> BackupResponse:
    """Stage a control-plane backup tar (DB + secrets + CA identity) (P14c WP-B3).

    Admin-only. The tar lands under `<data_dir>/backups/` (`0o700` dir,
    `0o600` tar); it carries the secrets master key, so custody transfers with
    the file. Idempotent (`IdempotencyMiddleware`) and audited `system.backup`
    with the basename + `contains_master_key` only — never the path, kid, or
    size. A second concurrent request while one stages gets `409
    backup.in_progress` (best-effort — the loser may run serially); a staged key
    rotation gets `409 secret.rotation_in_progress` (path-free message); a
    staging failure gets `500 backup.failed` (staging always cleaned up).
    """
    require_role(request, TokenRole.admin)
    if _backup_lock.locked():
        raise NerditError(
            409,
            "backup.in_progress",
            "A backup is already being staged.",
            hint="Wait for the current backup to finish before starting another.",
        )
    async with _backup_lock:
        settings = request.app.state.settings
        data_dir = Path(settings.data_dir).expanduser()
        # The durable effect here is a staged tar written by worker threads, not
        # a `@_serialized` DB write, and cancelling this await does not stop
        # them. Pin the idempotency claim BEFORE staging starts so a cancellation
        # mid-capture answers `interrupted` instead of freeing the key and
        # letting a same-key retry publish a second archive.
        mark_request_side_effect()
        try:
            # The secrets and DB snapshots are sequenced, never atomic, and
            # (enc file, variables.plain) is a cross-store pair: a
            # secret->plain flip landing between them would restore as an old
            # secret flagged plain. No variable write runs while a backup stages.
            # ponytail: held for the whole staging, not just the two snapshots.
            async with variable_write_lock(request.app):
                result = await create_backup(
                    db=request.app.state.db,
                    secret_manager=request.app.state.secret_manager,
                    data_dir=data_dir,
                    # Neither the node-link private key nor the product license ever
                    # enters a backup tar, even when an operator-chosen
                    # [link].key_file / [license].file sits inside a captured tree
                    # (a tar that can impersonate a node changes the custody story;
                    # a license is re-issuable, and keeping it out keeps customer_id
                    # out of a tar that already demands custody for the master key).
                    exclude_paths=(
                        resolve_key_file(settings.link.key_file, str(data_dir)),
                        resolve_license_file(settings.license.file, str(data_dir)),
                    ),
                )
        except SecretRotationInProgress as exc:
            # Reuse the class+code, never str(exc) — its message embeds the
            # absolute staged-key path.
            raise NerditError(
                409,
                "secret.rotation_in_progress",
                "A key rotation is staged; complete it before backing up.",
            ) from exc
        except BackupError as exc:
            # BackupError messages are already path-free (exc class + errno text).
            raise NerditError(
                500,
                "backup.failed",
                str(exc),
                hint="Staging was cleaned up; check daemon logs and retry.",
            ) from exc
        request.state.audit_params = audit_params(
            {"backup": result.basename, "contains_master_key": True}
        )
        return BackupResponse(
            backup=result.basename,
            path=result.path,
            size_bytes=result.size_bytes,
            kid=result.kid,
            created_at=result.manifest["created_at"],
            contains_master_key=True,
            hint=_BACKUP_CUSTODY_HINT,
        )


# --- POST /system/backup/volumes ---------------------------------------------

_VOLUME_BACKUP_CUSTODY_HINT = (
    "This archive contains all database data and the SCRAM password verifiers "
    "(derived from the minted password) — for most users more sensitive than a "
    "control-plane tar. It does NOT contain the secrets master key or any "
    "plaintext secret. Store it securely."
)


class VolumeBackupRequest(StrictRequestModel):
    """Body for `POST /system/backup/volumes`."""

    service: str = Field(description="The kind=database service whose volumes to capture.")


def _row_backend(job: Any) -> str | None:
    """Read `config['backend']` off a workload row, degrading to `None`."""
    cfg = parse_job_config(job)
    backend = cfg.get("backend")
    return backend if isinstance(backend, str) else None


@router.post(
    "/system/backup/volumes",
    operation_id="create_volume_backup",
    response_model=VolumeBackupResponse,
)
async def create_volume_backup_route(
    request: Request, body: VolumeBackupRequest
) -> VolumeBackupResponse:
    """Stage a per-database volume tar (data + SCRAM verifiers) (P15 WP7 — backup v2).

    Admin-only. Only `kind=database` rows are accepted in v1 (`422
    backup.not_a_database` otherwise — app-volume backup is a later
    generalization). The tar lands under `<data_dir>/backups/` named
    `nerdit-volumes-<service>-*` — OUTSIDE the v1 control-plane glob, so the
    control-plane retention/disk walkers never sweep it. Idempotent and audited
    `system.backup_volume` with `{service, backup: basename}` only (never the
    path or size). Guarded by the shared `_backup_lock` so a volume capture and
    a control-plane capture never interleave; `409 backup.in_progress` /
    `500 backup.failed` reuse the control-plane path-free codes. Unlike the control-plane
    tar this NEVER contains the secrets master key (`contains_master_key` is
    always false).
    """
    require_role(request, TokenRole.admin)
    queries = request.app.state.queries
    job = await queries.get_service_by_name(body.service)
    if job is None:
        raise NerditError(
            404,
            "not_found",
            f"No database '{body.service}'.",
            hint="List databases with `nerdit db list`.",
        )
    if job.kind is not JobKind.database:
        raise NerditError(
            422,
            "backup.not_a_database",
            "Volume backup targets a managed database in v1.",
            hint="Only kind=database rows can be captured with a volume backup.",
        )
    backend_name = _row_backend(job)
    # Set an early names-only audit param so a staging failure still records the
    # target (enriched with the basename on success — never the path or size).
    request.state.audit_params = audit_params({"service": body.service})

    if _backup_lock.locked():
        raise NerditError(
            409,
            "backup.in_progress",
            "A backup is already being staged.",
            hint="Wait for the current backup to finish before starting another.",
        )
    async with _backup_lock:
        settings = request.app.state.settings
        data_dir = Path(settings.data_dir).expanduser()
        # Same non-DB durable effect as the control-plane tar: the worker thread
        # keeps staging after a cancellation, so pin the claim before it starts.
        mark_request_side_effect()
        try:
            # Pure filesystem work (walk + tar + fsync) — off the event loop.
            result = await asyncio.to_thread(
                create_volume_backup,
                data_dir=data_dir,
                service=body.service,
                backend=backend_name,
            )
        except BackupError as exc:
            # BackupError messages are already path-free (exc class + errno text).
            raise NerditError(
                500,
                "backup.failed",
                str(exc),
                hint="Staging was cleaned up; check daemon logs and retry.",
            ) from exc
    request.state.audit_params = audit_params({"service": body.service, "backup": result.basename})
    return VolumeBackupResponse(
        service=body.service,
        backend=backend_name,
        backup=result.basename,
        path=result.path,
        size_bytes=result.size_bytes,
        created_at=result.manifest["created_at"],
        contains_master_key=False,
        hint=_VOLUME_BACKUP_CUSTODY_HINT,
    )
