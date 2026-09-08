"""Docker container runtime implementation."""

from __future__ import annotations

import asyncio
import grp
import logging
import os
import shutil
import signal
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import docker.errors
import docker.models.containers
import docker.types
import requests.exceptions
import urllib3.exceptions
from docker.client import DockerClient

import docker
from nerdit.config.defaults import (
    DEFAULT_ALLOWED_MOUNT_ROOTS,
    DEFAULT_DENIED_MOUNT_PATHS,
    NVIDIA_LIB_DIRS,
)
from nerdit.core.runtime.container import ContainerConfig
from nerdit.core.runtime.protocol import (
    BuildError,
    BuildPlatformError,
    ContainerNotFoundError,
    ContainerRuntimeError,
    ContainerStartError,
    ContainerStateInfo,
    ContainerStats,
    SandboxViolationError,
)
from nerdit.db.enums import GpuVendor

logger = logging.getLogger(__name__)

_ROCM_DEVICE_PATHS: tuple[Path, ...] = (Path("/dev/kfd"), Path("/dev/dri"))
_ROCM_DEVICES = [str(p) for p in _ROCM_DEVICE_PATHS]
_ROCM_GROUPS = ("video", "render")

#: Docker container states in which a process can still be writing to a bind mount.
#: `paused` counts: the writes are frozen, not finished, and an unpause resumes
#: them. Every other state (`exited` / `created` / `dead` / `removing`) is
#: inert. Consumed by `DockerRuntime.container_running`.
_WRITER_STATES = frozenset({"running", "restarting", "paused"})

#: Ownership labels stamped on every container this daemon creates and on every
#: image it builds. `managed-by` marks the resource as nerdit's; the value of
#: `nerdit-instance` is the `[daemon].instance_id` of the daemon that made
#: it, so co-located daemons never reclaim each other's resources (PR #81).
_MANAGED_BY_LABEL = "managed-by"
_INSTANCE_LABEL = "nerdit-instance"

#: Concurrency bound on `DockerRuntime.stats`.
#: `container.stats(stream=False)` is a *bounded single request* — not a
#: stream — so `asyncio.to_thread`'s shared default executor is the right
#: place for it (unlike the log-*follow* streams that pinned that pool and
#: deadlocked the daemon on the playground, post-P15). It still parks a worker
#: for the ~1-2 s docker takes to collect two CPU samples, so a burst of
#: `/stats` reads is capped here at four in flight; the rest queue on the
#: semaphore instead of exhausting the pool that every other docker call
#: shares. Module-level (never per-instance): the bound is on the process's
#: thread pool, which is also process-wide.
_STATS_CONCURRENCY = 4
_STATS_SEMAPHORE = asyncio.Semaphore(_STATS_CONCURRENCY)


def _as_int(value: object) -> int | None:
    """Coerce a docker stats counter to `int`, or `None` when unusable.

    `bool` is rejected explicitly (it is an `int` subclass and a `True`
    silently becoming `1` byte would be a lie). Anything non-numeric — a
    missing key, a `None`, a string — projects as *unknown*, never as `0`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _derive_cpu_pct(raw: dict) -> float | None:
    """Docker-CLI CPU percentage from a `stats(stream=False)` payload.

    `(cpu_delta / system_delta) * online_cpus * 100` — so `100.0` is one
    saturated core and a 4-core host tops out at `400.0`.

    Returns `None` — never `0.0` — whenever the ratio is undefined: a
    missing counter, a **zero or negative system-CPU delta** (the first sample
    on a freshly started container reports `precpu_stats.system_cpu_usage ==
    0`, and cgroup v2 omits the field entirely), or a negative container
    delta (a counter reset). A zero here would read as "idle"; the truth is
    "not measurable yet".
    """
    cpu_stats = raw.get("cpu_stats")
    precpu_stats = raw.get("precpu_stats")
    if not isinstance(cpu_stats, dict) or not isinstance(precpu_stats, dict):
        return None
    usage = cpu_stats.get("cpu_usage")
    pre_usage = precpu_stats.get("cpu_usage")
    if not isinstance(usage, dict) or not isinstance(pre_usage, dict):
        return None

    total = _as_int(usage.get("total_usage"))
    pre_total = _as_int(pre_usage.get("total_usage"))
    system = _as_int(cpu_stats.get("system_cpu_usage"))
    pre_system = _as_int(precpu_stats.get("system_cpu_usage"))
    if total is None or pre_total is None or system is None or pre_system is None:
        return None

    cpu_delta = total - pre_total
    system_delta = system - pre_system
    if system_delta <= 0 or cpu_delta < 0:
        return None

    online = _as_int(cpu_stats.get("online_cpus"))
    if online is None or online <= 0:
        percpu = usage.get("percpu_usage")
        online = len(percpu) if isinstance(percpu, list) and percpu else 1
    return round((cpu_delta / system_delta) * online * 100.0, 2)


def _derive_memory(raw: dict) -> tuple[int | None, int | None]:
    """`(used_bytes, limit_bytes)` from a stats payload, page cache excluded.

    Mirrors the docker CLI: the reported *usage* includes the page cache, so
    the file-backed inactive pages are subtracted (`total_inactive_file` on
    cgroup v1 — looked up first, exactly as the CLI does — else
    `inactive_file` on v2) to give the working set an operator actually
    recognises. When neither key is present the raw usage is returned unchanged
    rather than nothing — an over-report is still information; a `None` would
    not be.
    """
    mem = raw.get("memory_stats")
    if not isinstance(mem, dict):
        return None, None
    used = _as_int(mem.get("usage"))
    limit = _as_int(mem.get("limit"))
    detail = mem.get("stats")
    if used is not None and isinstance(detail, dict):
        # v1 key FIRST, matching the docker CLI: a cgroup-v1 payload carries
        # both `total_inactive_file` (the hierarchy total, what docker
        # subtracts) and a bare per-cgroup `inactive_file`; preferring the
        # bare one there would under-subtract. A v2 payload has only
        # `inactive_file`, so the fallback still covers it.
        cache = _as_int(detail.get("total_inactive_file"))
        if cache is None:
            cache = _as_int(detail.get("inactive_file"))
        if cache is not None:
            used = max(0, used - cache)
    return used, limit


def _derive_network(raw: dict) -> tuple[int | None, int | None]:
    """`(rx_bytes, tx_bytes)` summed over every attached network interface.

    `None` when docker reported no `networks` block at all (a container on
    `network_mode=host` or with networking disabled), which is genuinely
    unknown — distinct from a container that has transferred zero bytes.
    """
    networks = raw.get("networks")
    if not isinstance(networks, dict) or not networks:
        return None, None
    rx = 0
    tx = 0
    seen = False
    for iface in networks.values():
        if not isinstance(iface, dict):
            continue
        iface_rx = _as_int(iface.get("rx_bytes"))
        iface_tx = _as_int(iface.get("tx_bytes"))
        if iface_rx is None and iface_tx is None:
            continue
        seen = True
        rx += iface_rx or 0
        tx += iface_tx or 0
    return (rx, tx) if seen else (None, None)


#: Trailing BuildKit chatter that carries no diagnostic value. The last line of
#: a failed `docker build` is normally a `docker-desktop://` deep link, so a
#: naive "last line" in the `BuildError` message would tell an agent
#: nothing about why its build failed.
_BUILD_NOISE_PREFIXES = ("View build details:",)

#: Per-line buffer for the build's stdout `StreamReader` (P29 review round-2,
#: Codex 3803596893). asyncio's default is 64 KiB, and `readline()` raises a
#: bare `ValueError` — NOT a `ContainerRuntimeError` — when a single line
#: overruns it. `AppBuildManager.build` catches only `ContainerRuntimeError`,
#: so the escape skipped `_settle_failed_generation` entirely and wedged the
#: row in `building` forever. BuildKit `--progress=plain` legitimately emits
#: lines past 64 KiB (echoed long `RUN` commands, inline-cache base64, big
#: `COPY` file lists), so raise the bound and handle the overrun besides.
_BUILD_STREAM_LIMIT = 1_048_576

#: What a caller sees in place of a line that overran even the raised bound.
_BUILD_LINE_TRUNCATED = "[nerdit] build output line exceeded 1 MiB; truncated"

#: (BUG-1) The buildx availability probe. `docker buildx version` exercises the
#: CLI's own plugin resolution — the exact mechanism `docker build` uses — so
#: the verdict cannot disagree with what a real build would find. A stderr regex
#: over the build output would instead depend on Docker's message wording.
_BUILDX_PROBE_ARGV = ("buildx", "version")
#: Bounded well under the 2 s `/doctor` per-check budget, so the buildx leg can
#: never turn an otherwise-successful docker check into "check failed or timed out".
_BUILDX_PROBE_TIMEOUT_S = 1.0
#: (Codex 3804646823) How long a POSITIVE buildx verdict may be reused. The
#: memo was permanent, which made the cache asymmetric in the wrong direction:
#: install→present needed no restart, but remove/break→missing needed one, and
#: until then /doctor claimed a builder that was gone while failed builds fell
#: back to BuildError/USER_ERROR. A bounded TTL restores the symmetry at one
#: 1 s-bounded subprocess per daemon per window.
_BUILDX_CACHE_TTL_S = 300.0


def _kill_process_tree(proc) -> None:  # noqa: ANN001 — asyncio.subprocess.Process
    """SIGKILL a spawned child's whole session, not just its head.

    (Codex 3804646864) `docker buildx version` runs the CLI's own
    `docker-buildx` plugin as a child process (the plugin is a separate
    executable under `cli-plugins`), and `docker build` leaves BuildKit work
    behind the same way. Killing only the direct `docker` process reparents
    the rest to init; because a timed-out probe verdict is never cached, every
    later /doctor call or failed build leaked another one.

    Both spawn sites pass `start_new_session=True`, so the child is a
    session/group leader and its pid IS its pgid — one `killpg` reaps the
    tree. The `returncode` check keeps us from ever signalling a possibly
    recycled pid; it leaves the same microscopic TOCTOU window the plain
    `proc.kill()` it replaces already had. Callers still `await proc.wait()`
    to reap.
    """
    if proc.returncode is not None:
        return
    try:
        # ProcessLookupError is an OSError, so an already-reaped group is covered.
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        # No process group (a platform or a test double without one): fall back
        # to the head-only kill rather than leaving the child running.
        proc.kill()


async def _probe_buildx(docker_bin: str) -> str:
    """Return `'present' | 'missing' | 'unknown'` for the buildx CLI plugin.

    `'unknown'` is the fail-open value: a probe that timed out or could not be
    spawned must never be read as "buildx is missing" (that would reclassify a
    genuine user build error as a platform fault). Module-level and
    dependency-free so both `DockerRuntime` and the tests have exactly
    one patch seam.
    """
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            docker_bin,
            *_BUILDX_PROBE_ARGV,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            # (Codex 3804646864) Its own session, so the kill below reaps the
            # CLI *and* the `docker-buildx` plugin it spawns.
            start_new_session=True,
        )
        rc = await asyncio.wait_for(proc.wait(), _BUILDX_PROBE_TIMEOUT_S)
    except (TimeoutError, OSError):
        # 3.11+ aliases asyncio.TimeoutError onto the builtin, so one clause
        # covers both the wait_for bound and a failed spawn. Fail open.
        return "unknown"
    finally:
        # Same discipline as `build_image`: never leave a detached child —
        # nor a detached grandchild.
        if proc is not None:
            _kill_process_tree(proc)
            await proc.wait()
    return "present" if rc == 0 else "missing"


def _build_failure_tail(tail: "deque[str] | list[str]") -> str:
    """The most informative line of a failed build's output tail.

    Prefers BuildKit's own `ERROR:` marker (searched from the end, so the
    innermost failure wins), then the last line that is not trailing chatter,
    and only then the raw last line. The full output has already been streamed
    to the caller's log capture — this picks the one line that goes into the
    `BuildError` message, `config['last_deploy'].reason`'s companion
    text and, ultimately, the agent's diagnosis.
    """
    lines = list(tail)
    for line in reversed(lines):
        if line.lstrip().upper().startswith("ERROR"):
            return line
    for line in reversed(lines):
        if not line.lstrip().startswith(_BUILD_NOISE_PREFIXES):
            return line
    return lines[-1] if lines else "no output"


def _image_instance_label(attrs: dict) -> str | None:
    """The `nerdit-instance` label of an image, or `None` when unlabelled.

    Tolerant of both shapes docker reports labels in (`Config.Labels` on an
    inspect payload, a flat `Labels` on a listing) and of an image with no
    labels at all. `None` means "no owning instance recorded" — either a
    non-nerdit image or a `nerdit-app/*` image built before the label shipped.
    """
    labels = None
    config = attrs.get("Config")
    if isinstance(config, dict):
        labels = config.get("Labels")
    if not isinstance(labels, dict):
        labels = attrs.get("Labels")
    if not isinstance(labels, dict):
        return None
    value = labels.get(_INSTANCE_LABEL)
    return value if isinstance(value, str) and value else None


def _is_read_timeout(exc: BaseException) -> bool:
    """Is this exception (or something it wraps) a urllib3 read timeout?

    `requests` surfaces a read timeout two different ways depending on where
    it expires. Waiting for the response headers gives a
    `requests.exceptions.ReadTimeout` (a `Timeout`); expiring while the
    body is being streamed gives a plain `requests.exceptions.ConnectionError`
    wrapping urllib3's `ReadTimeoutError` (`requests/models.py`
    `iter_content`) — NOT a `Timeout`, and that is the path docker-py's
    `container.wait(timeout=…)` takes over the unix socket.

    Classifying on the wrapped cause rather than the outer type matters: a real
    docker-daemon outage is also a `ConnectionError` and must keep
    propagating as one instead of being mistaken for an expired wait.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, urllib3.exceptions.ReadTimeoutError):
            return True
        # requests wraps via args[0]; `raise ... from ...` via __cause__.
        wrapped = current.args[0] if current.args else None
        candidates = (wrapped, current.__cause__, current.__context__)
        pending.extend(e for e in candidates if isinstance(e, BaseException))
    return False


def _rocm_group_ids() -> list[str]:
    """Resolve host GIDs for ROCm device access.

    Group *names* in `group_add` resolve against the container's
    /etc/group, whose GIDs rarely match the host's /dev/kfd ownership, so
    numeric host GIDs are passed instead. Groups missing on the host are
    omitted (with a warning) rather than passed by name.
    """
    ids = []
    for group_name in _ROCM_GROUPS:
        try:
            ids.append(str(grp.getgrnam(group_name).gr_gid))
        except KeyError:
            logger.warning(
                "Host group '%s' not found; omitting it from container group_add", group_name
            )
    return ids


class DockerRuntime:
    """Container runtime using docker-py SDK.

    All docker-py calls are wrapped in asyncio.to_thread() since
    the SDK is synchronous.
    """

    def __init__(
        self,
        client: DockerClient | None = None,
        denied_mount_paths: list[str] | None = None,
        allowed_mount_roots: list[str] | None = None,
        system_mount_roots: list[str] | None = None,
        instance_id: str = "default",
    ) -> None:
        self._client = client or DockerClient.from_env()
        self._nvidia_available: bool | None = None
        # (BUG-1) Memoizes ONLY the positive buildx verdict, and only until this
        # monotonic deadline — see `buildx_available` for why the negative
        # must always re-probe and why the positive is now bounded.
        self._buildx_ok_until: float | None = None
        # Labels every container this daemon creates (`nerdit-instance=<id>`)
        # and scopes the zombie sweep to the same label, so two daemons on one
        # Docker host never reap each other's containers.
        self._instance_id = instance_id
        # Host GIDs are static for the daemon's lifetime; resolved on the
        # first AMD launch (getgrnam can be a network round-trip on
        # LDAP/sssd-backed NSS) and never on NVIDIA-only hosts.
        self._rocm_group_add: list[str] | None = None
        # Tier-A sandbox denylist (always enforced, even constructed bare in
        # tests). Stored as `(original_str, expanded_Path)` pairs.
        raw_denied = (
            denied_mount_paths if denied_mount_paths is not None else DEFAULT_DENIED_MOUNT_PATHS
        )
        self._denied_mounts: list[tuple[str, Path]] = [
            (p, Path(p).expanduser()) for p in raw_denied
        ]
        # Daemon-managed roots carved out of Tier-A: uploaded job workspaces and
        # the cache live *under* `~/.nerdit` (a denied path), so without this
        # exemption every uploaded `nerdit run` would be blocked. Narrow by
        # default (uploads + cache); admins control any widening.
        # `system_mount_roots` are daemon-owned dirs the daemon itself binds
        # (e.g. the Ollama weights cache under `<data_dir>/models`). They are
        # carved out of Tier-A *only* — deliberately NOT part of the Tier-B
        # user-mount allowlist (that list lives in settings and gates scoped
        # tokens), so a non-admin cannot bind-mount the shared weights cache
        # while the daemon's own model volume still launches. The symlink
        # resolve-before-check in `_match_denied` still applies, so a symlink
        # planted under a system root that escapes to /etc is not waved through.
        raw_allowed = (
            allowed_mount_roots if allowed_mount_roots is not None else DEFAULT_ALLOWED_MOUNT_ROOTS
        )
        self._allowed_roots: list[Path] = []
        for root in [*raw_allowed, *(system_mount_roots or [])]:
            try:
                self._allowed_roots.append(Path(root).expanduser().resolve())
            except (OSError, RuntimeError):
                self._allowed_roots.append(Path(root).expanduser())

    @staticmethod
    def _path_forms(path: Path) -> set[Path]:
        """Return the literal and symlink-resolved forms of *path*.

        `resolve()` follows symlinks (catching `/tmp/x -> docker.sock`),
        but the unresolved form is kept too so a denied entry that is itself a
        symlink still matches.
        """
        forms = {path}
        try:
            forms.add(path.resolve())
        except (OSError, RuntimeError):
            pass
        return forms

    def _under_allowed(self, path: Path) -> bool:
        """Whether *path* is equal to or under a daemon-managed allowed root."""
        return any(path == allowed or allowed in path.parents for allowed in self._allowed_roots)

    def _match_denied(self, host_path: str) -> str | None:
        """Return the denied policy entry a host mount matches, or `None`.

        A mount is denied when any of its path forms:

        * equals a denied entry, or
        * is a **descendant** of one (mounting `/etc/foo` exposes `/etc`), or
        * is an **ancestor** of a denied non-root file/dir (mounting `/run` or
          `/var/run` exposes `docker.sock` — the ancestor case the original
          code missed).

        The filesystem root (`/`) only matches by equality, so it blocks
        mounting `/` itself without rejecting every absolute path.
        """
        host_forms = self._path_forms(Path(host_path).expanduser())
        # Daemon-managed roots are exempt (e.g. ~/.nerdit/uploads under the
        # denied ~/.nerdit) so legitimate uploaded workspaces still mount — but
        # only when *every* form (literal AND symlink-resolved) is under an
        # allowed root, so a symlink under uploads whose target escapes to
        # /etc or docker.sock is not waved through (resolve-before-check).
        if host_forms and all(self._under_allowed(h) for h in host_forms):
            return None
        root = Path(host_path).anchor or "/"
        for original, denied in self._denied_mounts:
            for d in self._path_forms(denied):
                is_root = str(d) == root
                for h in host_forms:
                    if h == d:
                        return original
                    if not is_root and d in h.parents:
                        return original  # mount is a descendant of a denied path
                    if not is_root and h in d.parents:
                        return original  # mount is a parent of a denied file
        return None

    def _validate_mounts(self, volumes: dict[str, str] | None) -> None:
        """Reject any bind mount whose host path hits the Tier-A denylist.

        Runs for *every* caller (admin included) — the last line of defense
        against escaping the sandbox via `/var/run/docker.sock`, `/etc`,
        `~/.nerdit`, etc.
        """
        if not volumes:
            return
        for host_path in volumes:
            denied = self._match_denied(host_path)
            if denied is not None:
                raise SandboxViolationError(
                    f"Mount of host path {host_path!r} is blocked by the sandbox "
                    f"policy (matches denied path {denied!r}).",
                    reason="denied_mount",
                    path=host_path,
                    denied=denied,
                )

    async def run(self, config: ContainerConfig) -> str:
        """Launch a container and return its ID."""
        try:
            container = await asyncio.to_thread(self._run_sync, config)
            return container.id
        except docker.errors.ImageNotFound as exc:
            raise ContainerStartError(f"Image not found: {config.image}") from exc
        except docker.errors.APIError as exc:
            error_msg = str(exc)
            # Vendor-specific diagnostics only apply to that vendor's jobs:
            # the substrings (especially "hsa") are too generic to classify
            # arbitrary Docker errors safely.
            if config.vendor == GpuVendor.nvidia:
                if "libnvidia-ml" in error_msg:
                    raise ContainerStartError(
                        "NVIDIA driver library (libnvidia-ml.so.1) not accessible. "
                        "This often happens with snap-installed Docker. "
                        f"Original error: {error_msg}"
                    ) from exc
                if "nvidia-container-cli" in error_msg:
                    raise ContainerStartError(
                        "NVIDIA Container Toolkit error: nvidia-container-cli failed. "
                        f"Original error: {error_msg}"
                    ) from exc
            elif config.vendor == GpuVendor.amd and (
                "/dev/kfd" in error_msg or "/dev/dri" in error_msg or "hsa" in error_msg.lower()
            ):
                raise ContainerStartError(
                    "AMD ROCm device access failed. Check that the amdgpu kernel driver "
                    "is loaded, /dev/kfd and /dev/dri exist, and a ROCm-capable image is "
                    f"used. Original error: {error_msg}"
                ) from exc
            raise ContainerStartError(f"Failed to start container: {exc}") from exc

    def _run_sync(self, config: ContainerConfig) -> docker.models.containers.Container:
        """Synchronous Docker SDK call to create and start a container."""
        # Tier-A sandbox enforcement runs before anything is built, for every
        # caller including admin.
        self._validate_mounts(config.volumes)

        volumes = {}
        if config.volumes:
            for host_path, container_path in config.volumes.items():
                volumes[host_path] = {"bind": container_path, "mode": "rw"}

        kwargs: dict = dict(
            image=config.image,
            command=config.command,
            volumes=volumes or None,
            environment=config.env,
            working_dir=config.workdir,
            mem_limit=config.memory_limit,
            detach=True,
            # `nerdit-instance` scopes ownership to this daemon so a
            # co-located sibling daemon's zombie sweep never reaps it. Platform
            # labels are spread LAST: extra_labels (P20 run attribution) can
            # never overwrite ownership.
            labels={
                **(config.extra_labels or {}),
                _MANAGED_BY_LABEL: "nerdit",
                _INSTANCE_LABEL: self._instance_id,
            },
        )
        # docker-py rejects `nano_cpus=None`; only set it when a limit is given.
        if config.cpu_limit:
            kwargs["nano_cpus"] = int(config.cpu_limit * 1_000_000_000)
        # /dev/shm sizing (P11 vLLM); only set when requested so the 64MB default
        # is otherwise untouched.
        if config.shm_size:
            kwargs["shm_size"] = config.shm_size
        # Run as a non-root in-image user (e.g. postgres/redis) so the
        # official database entrypoints take their non-root path under the
        # untouched cap_drop=ALL / no-new-privileges sandbox. Guarded
        # so every existing caller (user is None) keeps byte-identical kwargs.
        if config.user is not None:
            kwargs["user"] = config.user
        # Per-container json-file log caps (service/model only; callers
        # leaves it None). Bounds container stdout/stderr on disk.
        if config.log_config:
            kwargs["log_config"] = docker.types.LogConfig(
                type="json-file", config=config.log_config
            )

        # Sandbox hardening kwargs (None/False leaves the runtime defaults).
        if config.cap_drop:
            kwargs["cap_drop"] = list(config.cap_drop)
        if config.read_only:
            kwargs["read_only"] = True
        if config.network_mode is not None:
            kwargs["network_mode"] = config.network_mode
        # Service-mode published ports: bind each container port to its host port
        # on the loopback interface only (never 0.0.0.0 — the ProxyManager fronts
        # services in P3). Guarded so batch (ports is None) keeps byte-identical
        # kwargs; docker-py rejects `ports=None`.
        #
        # P5: `extra_port_bind_ips` adds *additional* host bind addresses
        # (e.g. the docker bridge gateway, so app containers can reach a model
        # endpoint) alongside loopback. When unset, the loopback-only kwargs
        # shape stays byte-identical to the pre-P5 form — the frozen batch and
        # plain-service paths must never change.
        if config.ports:
            extras = config.extra_port_bind_ips
            if extras:
                kwargs["ports"] = {
                    f"{cport}/tcp": [("127.0.0.1", hport)] + [(ip, hport) for ip in extras]
                    for cport, hport in config.ports.items()
                }
            else:
                kwargs["ports"] = {
                    f"{cport}/tcp": ("127.0.0.1", hport) for cport, hport in config.ports.items()
                }

        # Single security_opt list so per-workload privilege hardening and the
        # AMD seccomp override coexist instead of clobbering each other.
        sec_opt: list[str] = []
        if config.no_new_privileges:
            sec_opt.append("no-new-privileges:true")

        # Only touch GPU passthrough when GPUs were actually allocated. A
        # CPU-only workload (e.g. a default `gpus=0` service) must NOT get a
        # DeviceRequest — an empty device_ids + gpu capability would demand the
        # NVIDIA runtime and fail to start on a host without it. Batch jobs
        # always carry gpu_ids, so this is a no-op for them.
        if not config.gpu_ids:
            pass
        elif config.vendor == GpuVendor.amd:
            # Coarse ROCm passthrough: expose the KFD and DRI device nodes and
            # restrict visibility via ROCR_VISIBLE_DEVICES (per-renderD isolation
            # is future hardening).
            kwargs["devices"] = list(_ROCM_DEVICES)
            if self._rocm_group_add is None:
                self._rocm_group_add = _rocm_group_ids()
            kwargs["group_add"] = self._rocm_group_add
            # APPEND (never replace): ROCm needs an unconfined seccomp profile,
            # but no-new-privileges (if requested) must survive alongside it.
            sec_opt.append("seccomp=unconfined")
            env = dict(config.env or {})
            env["ROCR_VISIBLE_DEVICES"] = ",".join(config.gpu_ids)
            kwargs["environment"] = env
        # Only request NVIDIA GPU passthrough when the toolkit is available
        elif self._nvidia_available is not False:
            device_request = docker.types.DeviceRequest(
                device_ids=config.gpu_ids,
                capabilities=[["gpu"]],
            )
            kwargs["device_requests"] = [device_request]
        else:
            logger.warning(
                "Running container without GPU passthrough (NVIDIA toolkit not installed)"
            )

        if sec_opt:
            kwargs["security_opt"] = sec_opt

        return self._client.containers.run(**kwargs)

    async def check_nvidia_runtime(self) -> bool:
        """Check if the NVIDIA runtime is available in Docker.

        Returns True if present, False otherwise. Logs a warning if missing.
        """
        try:
            info = await asyncio.to_thread(self._client.info)
            runtimes = info.get("Runtimes", {})
            if "nvidia" in runtimes:
                logger.info("NVIDIA runtime detected in Docker")
                # Verify the driver library is actually accessible (includes the
                # WSL2 passthrough path `/usr/lib/wsl/lib` via NVIDIA_LIB_DIRS).
                lib_accessible = any(
                    (Path(d) / "libnvidia-ml.so.1").exists() for d in NVIDIA_LIB_DIRS
                )
                if lib_accessible:
                    self._nvidia_available = True
                    return True
                else:
                    self._nvidia_available = False
                    logger.warning(
                        "NVIDIA runtime registered but libnvidia-ml.so.1 not found. "
                        "GPU passthrough will likely fail. "
                        "This often happens with snap-installed Docker."
                    )
                    return False
            self._nvidia_available = False
            logger.warning(
                "NVIDIA runtime not found in Docker. "
                "Install it with: sudo apt install nvidia-container-toolkit && "
                "sudo systemctl restart docker"
            )
            return False
        except Exception:
            logger.warning("Could not check Docker runtimes", exc_info=True)
            return False

    async def check_rocm_runtime(self) -> bool:
        """Check that AMD ROCm container passthrough prerequisites are present.

        Returns True when /dev/kfd and /dev/dri exist, False otherwise. Access
        diagnostics are logged as warnings only: they probe the daemon user
        while dockerd typically runs as root, so they are a coarse proxy.
        """
        from nerdit.core.discovery import amd_gpu_access_diagnostic

        try:
            missing = [p for p in _ROCM_DEVICE_PATHS if not p.exists()]
            if missing:
                logger.warning(
                    "AMD ROCm passthrough unavailable: %s missing. "
                    "Ensure the amdgpu kernel driver and ROCm KFD support are loaded.",
                    missing[0],
                )
                return False
            diagnostic = await asyncio.to_thread(amd_gpu_access_diagnostic)
            if diagnostic:
                logger.warning("%s", diagnostic)
            logger.info("AMD ROCm device nodes detected (/dev/kfd, /dev/dri)")
            return True
        except Exception:
            logger.warning("Could not check ROCm passthrough prerequisites", exc_info=True)
            return False

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        """Stop a container gracefully with the given timeout (seconds)."""
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.stop, timeout=timeout)
        except docker.errors.NotFound as exc:
            raise ContainerNotFoundError(f"Container {container_id} not found") from exc
        except docker.errors.APIError as exc:
            raise ContainerRuntimeError(f"Failed to stop container: {exc}") from exc

    async def kill(self, container_id: str) -> None:
        """Force-kill a running container immediately."""
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.kill)
        except docker.errors.NotFound as exc:
            raise ContainerNotFoundError(f"Container {container_id} not found") from exc
        except docker.errors.APIError as exc:
            raise ContainerRuntimeError(f"Failed to kill container: {exc}") from exc

    @staticmethod
    def _read_log_tail(
        container: docker.models.containers.Container,
        tail: int | None,
        max_bytes: int | None,
    ) -> bytes:
        """Blocking bounded tail read — runs in a worker thread.

        Consumes `container.logs(stream=True, follow=False, tail=…)`, which
        docker-py returns as a generator of **byte chunks**: one per frame of
        the multiplexed stdout/stderr stream for a non-TTY container (every
        container this runtime creates is non-TTY — docker-py's TTY branch
        instead yields raw `iter_content` chunks). Frame boundaries are NOT
        line boundaries, so a chunk may hold several lines or half of one; the
        budget is therefore applied over bytes and the caller splits lines.

        `follow=False` keeps the stream finite — the daemon closes it once
        the backlog is drained — which is why a pooled `to_thread` is safe
        here where `logs`' follow branch needs a thread of its own.
        """
        chunks: deque[bytes] = deque()
        total = 0
        # docker-py coerces anything that is not a non-negative int to "all";
        # pass the sentinel explicitly rather than rely on that coercion.
        docker_tail: int | str = tail if tail is not None else "all"
        for chunk in container.logs(stream=True, follow=False, tail=docker_tail):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if max_bytes is None:
                continue
            # Drop from the HEAD as we consume: the oldest bytes go first, so
            # what is retained is always the most recent `max_bytes` and peak
            # memory is the budget plus the chunk just read. Slicing a
            # fully-materialized buffer afterwards would be unbounded.
            while total > max_bytes and chunks:
                head = chunks[0]
                excess = total - max_bytes
                if len(head) <= excess:
                    chunks.popleft()
                    total -= len(head)
                else:
                    chunks[0] = head[excess:]
                    total -= excess
        return b"".join(chunks)

    async def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream container logs as decoded lines.

        When *follow* is `True`, yields lines in real time via a
        background thread and an `asyncio.Queue`; *tail* and *max_bytes* are
        **ignored** on that branch — a follow stream is unbounded by
        construction and its consumer bounds it by disconnecting. Otherwise *tail* caps the
        trailing lines (server-side, `docker
        logs --tail`) and *max_bytes* the trailing bytes, enforced **while the
        stream is consumed** (see `_read_log_tail`). When the budget
        clips mid-line the first yielded line is a fragment: for a crash tail
        recency beats alignment. With both `None` — every pre-P20 caller —
        the legacy unbounded whole-buffer read is used unchanged.
        """
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
        except docker.errors.NotFound as exc:
            raise ContainerNotFoundError(f"Container {container_id} not found") from exc

        if follow:
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def _put(item: str | None) -> None:
                # asyncio.Queue is NOT thread-safe — hand the put to the loop.
                # A closed loop (daemon shutdown) just drops the line.
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                except RuntimeError:
                    pass

            def _stream_logs() -> None:
                try:
                    for chunk in container.logs(stream=True, follow=True):
                        _put(chunk.decode("utf-8", errors="replace").rstrip("\n"))
                except Exception:
                    pass
                finally:
                    _put(None)

            # A DEDICATED daemon thread — NEVER the shared default executor: a
            # follow-stream blocks for the container's entire lifetime, and the
            # default pool caps at min(32, cpus+4) workers, so a dozen running
            # services would pin every worker and DEADLOCK the whole daemon
            # (every asyncio.to_thread docker call — including deploys — queues
            # forever behind them; observed live on the playground box).
            threading.Thread(
                target=_stream_logs,
                name=f"docker-logs-{container_id[:12]}",
                daemon=True,
            ).start()

            while True:
                line = await queue.get()
                if line is None:
                    break
                yield line
        elif tail is None and max_bytes is None:
            raw = await asyncio.to_thread(container.logs, stream=False, follow=False)
            for line in raw.decode("utf-8", errors="replace").splitlines():
                yield line
        else:
            bounded = await asyncio.to_thread(self._read_log_tail, container, tail, max_bytes)
            for line in bounded.decode("utf-8", errors="replace").splitlines():
                yield line

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        """Wait for a container to exit and return its status code. *timeout_s* bounds the wait:
        the docker `/wait` request itself
        carries the timeout, so the **blocking call** ends when it expires, and
        the expiry surfaces to the awaiting caller as
        `asyncio.TimeoutError`. `asyncio.wait_for` over
        `to_thread(container.wait)` would NOT do: it cancels the coroutine
        only, leaving the worker blocked forever, so a handful of timed-out
        runs would pin the whole pool.

        Which is also why the call runs on a DEDICATED daemon thread and never
        on the shared default executor — same reasoning as `logs`' follow
        branch: a wait blocks for the container's entire lifetime, and the
        default pool caps at `min(32, cpus + 4)` workers (the post-P15
        pool-pinning deadlock, observed live).
        """
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
        except docker.errors.NotFound as exc:
            raise ContainerNotFoundError(f"Container {container_id} not found") from exc

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict] = loop.create_future()

        def _deliver(outcome: dict | Exception) -> None:
            # Runs on the loop thread. The future is already done when the
            # awaiting caller was cancelled — drop the late outcome.
            if future.done():
                return
            if isinstance(outcome, Exception):
                future.set_exception(outcome)
            else:
                future.set_result(outcome)

        def _wait_blocking() -> None:
            outcome: dict | Exception
            try:
                result = container.wait(timeout=timeout_s)
            except Exception as exc:
                outcome = exc
            else:
                outcome = result if isinstance(result, dict) else {}
            try:
                loop.call_soon_threadsafe(_deliver, outcome)
            except RuntimeError:
                pass  # loop closed (daemon shutdown) — nobody left to deliver to

        threading.Thread(
            target=_wait_blocking,
            name=f"docker-wait-{container_id[:12]}",
            daemon=True,
        ).start()

        try:
            result = await future
        except docker.errors.NotFound as exc:
            raise ContainerNotFoundError(f"Container {container_id} not found") from exc
        except requests.exceptions.Timeout as exc:
            # docker-py's DOCUMENTED expiry signal for `wait(timeout=…)` — a
            # requests read timeout, not a docker error. Re-raised as the
            # asyncio flavour so callers bound a run the same way they would
            # bound any other await.
            raise asyncio.TimeoutError(
                f"Timed out after {timeout_s}s waiting for container {container_id}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            # ...and its ACTUAL signal over the unix socket, which is not a
            # `requests.exceptions.Timeout` at all: the read expires while
            # requests is streaming the response body, and `iter_content`
            # re-wraps urllib3's `ReadTimeoutError` as a plain
            # `ConnectionError` (requests/models.py). Found live — the P20 WP3
            # run-primitive timeout leg raised `ConnectionError` straight out
            # of `run_once` instead of settling `timed_out=True`, and a
            # `[deploy].release` overrunning `release_timeout_s` took the
            # generic-Exception branch. A genuine docker outage is also a
            # `ConnectionError`, so classify on the wrapped cause, never on the
            # type alone.
            if not _is_read_timeout(exc):
                raise
            raise asyncio.TimeoutError(
                f"Timed out after {timeout_s}s waiting for container {container_id}"
            ) from exc
        return result.get("StatusCode", -1)

    async def remove(self, container_id: str, force: bool = False) -> None:
        """Remove a container. Silently ignores already-removed containers."""
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.remove, force=force)
        except docker.errors.NotFound:
            pass  # Already removed
        except docker.errors.APIError as exc:
            raise ContainerRuntimeError(f"Failed to remove container: {exc}") from exc

    async def status(self, container_id: str) -> str | None:
        """Return the container status string, or `None` if it no longer exists."""
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.reload)
            return container.status
        except docker.errors.NotFound:
            return None
        except docker.errors.APIError:
            return None

    async def container_running(self, container_id: str) -> bool | None:
        """Tri-state liveness probe (see the Protocol).

        Deliberately does NOT reuse `status`' `APIError → None` swallow: an
        `APIError` here is most often *the daemon is unreachable*, and a container
        we cannot see is not a container that has stopped — containerd keeps it
        running while dockerd is down. A caller taking an irreversible action on
        "nothing is writing here" (the DELETE data purge) must be able to tell
        "definitely inert" from "cannot say".
        """
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.reload)
        except docker.errors.NotFound:
            return False  # gone is definitively inert
        except docker.errors.APIError:
            return None  # unknown — NOT the same as inert
        return container.status in _WRITER_STATES

    async def inspect_state(self, container_id: str) -> ContainerStateInfo | None:
        """Return the container's `State` snapshot, or `None` if gone.

        Mirrors `status`' try/except-NotFound→None (and APIError→None)
        shape; reads `container.attrs['State']` after a `reload()`.
        """
        try:
            container = await asyncio.to_thread(self._client.containers.get, container_id)
            await asyncio.to_thread(container.reload)
            state = container.attrs.get("State") or {}
        except docker.errors.NotFound:
            return None
        except docker.errors.APIError:
            return None
        return ContainerStateInfo(
            exit_code=state.get("ExitCode"),
            oom_killed=bool(state.get("OOMKilled")),
            error=state.get("Error") or None,
            started_at=state.get("StartedAt") or None,
            finished_at=state.get("FinishedAt") or None,
        )

    async def stats(self, container_id: str) -> ContainerStats | None:
        """One bounded resource sample, or `None` if gone/unreadable.

        `container.stats(stream=False)` under `_STATS_SEMAPHORE`: a single request, never a
        stream, and never more than
        `_STATS_CONCURRENCY` of them parked in the shared thread pool at
        once. Same `NotFound`/`APIError` → `None` swallow as
        `inspect_state`, widened to any malformed payload — every derived
        field is independently `None`-safe, so a partial sample still reports
        the counters it does carry.

        Deliberately NOT called from the reconcile tick: the call blocks for as
        long as docker takes to collect its two CPU samples (~1-2 s), which
        would stall every service the tick has yet to visit.
        """
        try:
            async with _STATS_SEMAPHORE:
                container = await asyncio.to_thread(self._client.containers.get, container_id)
                raw = await asyncio.to_thread(container.stats, stream=False)
        except docker.errors.NotFound:
            return None
        except docker.errors.APIError:
            return None
        except (requests.exceptions.RequestException, OSError, ValueError):
            # A daemon that hung up mid-read, or a body that was not JSON. A
            # read-only projection never propagates: the caller's contract is
            # "None means unavailable".
            return None
        if not isinstance(raw, dict):
            return None

        mem_used, mem_limit = _derive_memory(raw)
        net_rx, net_tx = _derive_network(raw)
        pids_stats = raw.get("pids_stats")
        pids = _as_int(pids_stats.get("current")) if isinstance(pids_stats, dict) else None
        return ContainerStats(
            cpu_pct=_derive_cpu_pct(raw),
            mem_used_bytes=mem_used,
            mem_limit_bytes=mem_limit,
            net_rx_bytes=net_rx,
            net_tx_bytes=net_tx,
            pids=pids,
        )

    async def image_exists(self, image_name: str) -> bool:
        """Check if a Docker image exists locally."""
        try:
            await asyncio.to_thread(self._client.images.get, image_name)
            return True
        except docker.errors.ImageNotFound:
            return False
        except docker.errors.APIError:
            return False

    async def pull_image(self, image: str) -> None:
        """Pull *image* from its registry, blocking until complete.

        Wraps the synchronous `client.images.pull` in a thread executor like
        every other docker-py call. Idempotent when the image is already
        present locally (Docker no-ops the layers). Registry/daemon failures
        are mapped to `ContainerRuntimeError` so callers (the P5 model
        pull path) settle the workload cleanly.
        """
        try:
            await asyncio.to_thread(self._client.images.pull, image)
        except docker.errors.ImageNotFound as exc:
            raise ContainerRuntimeError(f"Image not found in registry: {image}") from exc
        except docker.errors.APIError as exc:
            raise ContainerRuntimeError(f"Failed to pull image {image}: {exc}") from exc

    async def list_images(self) -> list[str]:
        """List locally available image tags (`repo:tag`), sorted.

        Untagged (dangling) images are skipped. Errors degrade to an empty
        list — the consumer is a dashboard dropdown, not a health check.
        """
        try:
            images = await asyncio.to_thread(self._client.images.list)
        except Exception:
            logger.warning("Could not list Docker images", exc_info=True)
            return []
        tags: set[str] = set()
        for image in images:
            tags.update(tag for tag in (image.tags or []) if "<none>" not in tag)
        return sorted(tags)

    async def list_images_detailed(self) -> list[dict]:
        """List local images as `{repo_tag, id, size_bytes, instance}` dicts.

        One entry per `repo:tag` (dangling/`<none>` tags skipped). `id` is
        the short image id; `size_bytes` is the image's on-disk size.
        `instance` is the image's `nerdit-instance` label — the daemon that
        built it (`build_image` stamps it) — or `None` for an image with
        no such label (any non-nerdit image, and every `nerdit-app/*` image
        built before the label shipped). Errors degrade to an empty list, like
        `list_images`.
        """
        try:
            images = await asyncio.to_thread(self._client.images.list)
        except Exception:
            logger.warning("Could not list Docker images (detailed)", exc_info=True)
            return []
        result: list[dict] = []
        for image in images:
            attrs = image.attrs or {}
            size = attrs.get("Size")
            size_bytes = int(size) if isinstance(size, int) else 0
            short_id = image.short_id or image.id or ""
            instance = _image_instance_label(attrs)
            for tag in image.tags or []:
                if "<none>" in tag:
                    continue
                result.append(
                    {
                        "repo_tag": tag,
                        "id": short_id,
                        "size_bytes": size_bytes,
                        "instance": instance,
                    }
                )
        return result

    async def disk_usage(self) -> dict[str, int] | None:
        """Return docker's aggregate disk usage, or `None` on any error.

        Sums the layer sizes reported by `client.df()` into
        `{images_bytes, containers_bytes, volumes_bytes, build_cache_bytes}`.
        """
        try:
            df = await asyncio.to_thread(self._client.df)
        except Exception:
            logger.warning("Could not read Docker disk usage", exc_info=True)
            return None

        def _sum(items: object, key: str) -> int:
            total = 0
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        val = it.get(key)
                        if isinstance(val, (int, float)):
                            total += int(val)
            return total

        # Volume sizes live under each volume's `UsageData.Size` (docker df).
        volumes_bytes = 0
        vols = df.get("Volumes")
        if isinstance(vols, list):
            for v in vols:
                usage = v.get("UsageData") if isinstance(v, dict) else None
                if isinstance(usage, dict):
                    size = usage.get("Size")
                    if isinstance(size, (int, float)) and size > 0:
                        volumes_bytes += int(size)

        return {
            "images_bytes": _sum(df.get("Images"), "Size"),
            "containers_bytes": _sum(df.get("Containers"), "SizeRw"),
            "volumes_bytes": volumes_bytes,
            "build_cache_bytes": _sum(df.get("BuildCache"), "Size"),
        }

    async def list_managed_containers(self) -> list[tuple[str, datetime]]:
        """Return ALL nerdit-managed containers as `(id, created_at)` pairs.

        Deliberately UNSCOPED (`managed-by=nerdit` only, no instance filter):
        the service controller uses this to re-adopt its own live containers
        after a restart by matching their id against its DB rows. Scoping it to
        this daemon's `nerdit-instance` would hide the daemon's OWN
        pre-upgrade (unlabelled) containers from re-adoption, so a restart would
        see live services as dead and relaunch them into host-port conflicts.
        A sibling instance's containers appearing here are harmless — the
        controller only ever acts on ids present in its own DB. The zombie
        sweep, which KILLS, uses the instance-scoped variant below instead.

        Wraps the blocking docker-py call in `asyncio.to_thread`. Entries
        whose `Created` timestamp cannot be parsed are skipped.
        """
        return await self._list_managed(filters={"label": "managed-by=nerdit"})

    async def list_own_managed_containers(self) -> list[tuple[str, datetime]]:
        """Return only THIS daemon's managed containers (`(id, created_at)`).

        AND both labels (docker AND-combines a multi-value label filter), so a
        daemon only ever surfaces — and therefore the zombie sweep only ever
        KILLS — containers carrying its own `nerdit-instance`. That is what
        lets co-located daemons share one Docker host without reaping each
        other's services. Pre-upgrade containers (no `nerdit-instance` label)
        are excluded here too, so the sweep never kills them across the upgrade;
        they are still re-adopted via `list_managed_containers`, and get
        labelled on their next (re)deploy.
        """
        return await self._list_managed(
            filters={"label": ["managed-by=nerdit", f"nerdit-instance={self._instance_id}"]}
        )

    async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
        """Ids of THIS daemon's RUNNING containers carrying `label=value`.

        Instance-scoped like `list_own_managed_containers` — this feeds a
        KILL path (`_settle_crashed_release`), so it must never surface a
        co-located sibling daemon's containers (the PR #81 regression class).
        `containers.list` defaults to running-only, which is exactly the
        orphan-reaping set; an already-exited orphan is inert. No `Created`
        parsing (unlike `_list_managed`): a kill path must not skip a
        container because a timestamp failed to parse.
        """
        containers = await asyncio.to_thread(
            self._client.containers.list,
            filters={
                "label": [
                    "managed-by=nerdit",
                    f"nerdit-instance={self._instance_id}",
                    f"{label}={value}",
                ]
            },
        )
        return [c.id for c in containers]

    async def list_own_run_containers(self) -> list[str]:
        """Ids of THIS daemon's RUNNING containers carrying `nerdit-run` (ANY value).

        A bare label name in a docker filter list matches *presence* with any
        value, and docker AND-combines the list — so this is the exact-value
        sibling of `list_own_labeled_containers` with the value dropped.
        Instance-scoped for the same reason (this feeds the boot-side orphan
        kill, a KILL path — the PR #81 regression class), running-only because
        an already-exited orphan is inert, and with no `Created` parsing
        because a kill path must not skip a container over an unparsable
        timestamp.
        """
        containers = await asyncio.to_thread(
            self._client.containers.list,
            filters={
                "label": [
                    "managed-by=nerdit",
                    f"nerdit-instance={self._instance_id}",
                    "nerdit-run",
                ]
            },
        )
        return [c.id for c in containers]

    async def _list_managed(self, *, filters: dict) -> list[tuple[str, datetime]]:
        containers = await asyncio.to_thread(self._client.containers.list, filters=filters)
        result: list[tuple[str, datetime]] = []
        for container in containers:
            created_raw = container.attrs.get("Created") if container.attrs else None
            if not created_raw:
                continue
            try:
                # Docker returns RFC3339 with a trailing 'Z' and nanosecond
                # precision; trim to microseconds and swap 'Z' for '+00:00'
                # so fromisoformat accepts it on 3.11.
                normalized = created_raw.rstrip("Z")
                if "." in normalized:
                    head, frac = normalized.split(".", 1)
                    normalized = f"{head}.{frac[:6]}"
                created_at = datetime.fromisoformat(normalized).replace(tzinfo=UTC)
            except (ValueError, AttributeError):
                continue
            result.append((container.id, created_at))
        return result

    async def buildx_available(self) -> str:
        """Build-toolchain state, memoizing only the POSITIVE result, briefly.

        (BUG-1) Caching `'missing'` would force a daemon restart after the
        operator installs the plugin, so the missing→present transition always
        re-probes.

        (Codex 3804646823) A plugin CAN vanish (uninstall, a broken package
        upgrade, a lost `DOCKER_CONFIG`), so the positive verdict is memoized
        only for `_BUILDX_CACHE_TTL_S`. Concurrent callers (a /doctor check
        and a failed build) may probe twice inside one window — harmless, the
        probe is a read, so there is no lock. Both readers deliberately share
        ONE freshness window: a stale `present` on /doctor contradicting a
        fresh `missing` in a build's classification would put two first-party
        surfaces in direct disagreement.

        (Codex 3804646811) No docker CLI at all ⇒ `'no_cli'`, not
        `'unknown'`: that is a concluded, definite host fault (nothing on this
        node can build), while `unknown` is reserved for a probe that could
        not conclude and stays fail-open.
        """
        now = time.monotonic()
        if self._buildx_ok_until is not None and now < self._buildx_ok_until:
            return "present"
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            return "no_cli"
        verdict = await _probe_buildx(docker_bin)
        if verdict == "present":
            self._buildx_ok_until = time.monotonic() + _BUILDX_CACHE_TTL_S
        return verdict

    async def build_image(
        self, context_dir: str, tag: str, dockerfile: str | None = None
    ) -> AsyncIterator[str]:
        """Build an image from *context_dir*, streaming build-log lines.

        Shells out to the `docker` CLI with BuildKit rather than driving
        docker-py's `api.build`: the classic (non-BuildKit) `/build` endpoint
        that call targets is inert on **Docker >= 29** — it yields zero stream
        lines and never terminates, so every real deploy hung forever — and
        docker-py has no BuildKit support to move to. Never `shell=True`; the
        argv is fixed and the only caller-derived members are the context path,
        the tag and the Dockerfile name.

        `-f` is joined onto *context_dir* because the CLI resolves a relative
        `-f` against the CWD, not against the build context. `DOCKER_BUILDKIT`
        is forced on and the parent environment is inherited, so `DOCKER_HOST`
        / the active docker context still select the same daemon
        `docker.from_env()` talks to. BuildKit writes its `--progress=plain`
        output on stderr, which is merged into stdout here and yielded line by
        line — the caller's existing capture into `job_logs` is unchanged; only
        the line *format* differs from the classic stream.

        Every built image is stamped with the same ownership labels this runtime
        puts on containers (`managed-by=nerdit` + `nerdit-instance=<id>`),
        so the image GC can tell this daemon's images from a co-located sibling
        daemon's. Stamped here rather than as a `LABEL` in the generated
        Dockerfile so a passthrough (user-supplied) Dockerfile is labelled too.

        A non-zero exit raises `BuildError` carrying the exit code and the
        last output line. There is no wall-clock bound (the classic path had none
        either): the bound is consumer cancellation — if the generator is closed
        or cancelled (the P20 drain that abandons builds), the child process is
        killed rather than left orphaned.
        """
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            # (Codex 3804646811) A PLATFORM error, not a plain BuildError:
            # redeploying identical source cannot help, so the USER_ERROR this
            # used to settle sent every caller off to fix an app that is fine.
            # Path-free on purpose — this message is persisted into
            # `config['last_deploy']` and re-served over /diagnose.
            raise BuildPlatformError(
                "docker CLI not found on the daemon's PATH; the image build "
                "requires it. Install the Docker CLI where the daemon service "
                "can see it, or set PATH in the daemon's service unit."
            )

        argv = [
            docker_bin,
            "build",
            "--progress=plain",
            "-t",
            tag,
            "-f",
            os.path.join(context_dir, dockerfile or "Dockerfile"),
            "--label",
            f"{_MANAGED_BY_LABEL}=nerdit",
            "--label",
            f"{_INSTANCE_LABEL}={self._instance_id}",
            context_dir,
        ]

        try:
            # (BUG-1) DECISION: DOCKER_CONFIG is deliberately NOT pinned here.
            # The CLI already resolves plugins from $DOCKER_CONFIG/cli-plugins
            # (default $HOME/.docker/cli-plugins) and then the system-wide
            # plugin dirs, so a service-account HOME finds a system-wide install
            # on its own. Pinning to the daemon's own $HOME/.docker is a no-op;
            # pinning to a nerdit-owned dir would hide whatever plugin dir the
            # real HOME does have AND relocate config.json, silently dropping
            # registry credential helpers and breaking pull_image auth. And
            # because the parent environment is inherited verbatim, an operator
            # who sets DOCKER_CONFIG in the systemd Environment= / launchd
            # EnvironmentVariables line already gets it — hard-coding it would
            # override that and take away their only correct lever.
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "DOCKER_BUILDKIT": "1"},
                limit=_BUILD_STREAM_LIMIT,
                # (Codex 3804646864) Its own session, so abandoning the
                # generator reaps BuildKit's children too, not just the CLI
                # head. Verified safe: `daemon/lifecycle.py` stops the daemon
                # with a single-pid `os.kill`, never a group signal, so
                # nothing relies on builds sharing the daemon's process group.
                start_new_session=True,
            )
        except OSError as exc:
            raise BuildError(f"Image build failed: {exc}") from exc

        tail: deque[str] = deque(maxlen=30)
        try:
            assert proc.stdout is not None  # PIPE was requested
            while True:
                try:
                    raw = await proc.stdout.readline()
                except ValueError:
                    # Still longer than the raised bound. CPython's StreamReader
                    # CLEARS its buffer before raising, so the loop resumes on the
                    # next line rather than spinning: report the gap honestly and
                    # keep streaming. Never let this escape — it is not a
                    # `ContainerRuntimeError`, so the builder would not settle
                    # the generation and the row would wedge in `building`.
                    tail.append(_BUILD_LINE_TRUNCATED)
                    yield _BUILD_LINE_TRUNCATED
                    continue
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    tail.append(line)
                    yield line

            rc = await proc.wait()
            if rc != 0:
                tail_text = _build_failure_tail(tail)
                # (BUG-1) Classify the failure by PROBING for the BuildKit
                # builder, on the failure path only: a happy-path preflight
                # would add a subprocess to every build to answer a question
                # that only matters once one has already failed. The proactive
                # half of BUG-1 lives in `check-deps` and `/doctor`.
                # `== "missing"` and not `!= "present"`: `no_cli` is
                # unreachable here (the early guard above already raised on it)
                # and `unknown` must stay fail-open.
                if await self.buildx_available() == "missing":
                    raise BuildPlatformError(
                        "docker build failed: the BuildKit builder (docker buildx) "
                        "is not available to this daemon. Install the buildx CLI "
                        "plugin system-wide (apt: docker-buildx-plugin) or set "
                        "DOCKER_CONFIG in the daemon's service unit to a home that "
                        f"has it. Build output: {tail_text}"
                    )
                raise BuildError(f"docker build failed (exit {rc}): {tail_text}")
        finally:
            # Closed/cancelled generator (drain, client disconnect): never leave
            # a detached `docker build` — nor its BuildKit children — running
            # against the daemon. An abandoned build burns CPU and disk, so the
            # tree kill matters more here than at the probe.
            _kill_process_tree(proc)
            await proc.wait()

    async def remove_image(self, tag: str, force: bool = False) -> None:
        """Remove a local image by tag. Silently ignores a missing image.

        An `APIError` (typically "image is referenced in a stopped container"
        during prune) is logged and swallowed — pruning is best-effort.
        """
        try:
            await asyncio.to_thread(self._client.images.remove, tag, force=force)
        except docker.errors.ImageNotFound:
            pass  # Already gone
        except Exception as exc:
            # Best-effort: "image is referenced in a stopped container" during
            # prune, or dockerd hiccups, must not break the reconcile loop.
            logger.warning("Could not remove image %s: %s", tag, exc)
