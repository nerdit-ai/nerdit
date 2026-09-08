"""Shared daemon-free dependency checks for check-deps, init and doctor.

Checks return CheckResult with ok/warn/fail/skip status; only fail blocks.
Remediation names a platform-specific command or config setting. Access system
APIs through their module namespaces so probes remain independently mockable.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from nerdit.config.defaults import (
    DEFAULT_CADDY_BINARY,
    DEFAULT_DATA_DIR,
    DEFAULT_PORT,
    NVIDIA_LIB_DIRS,
)
from nerdit.config.settings import load_settings
from nerdit.core.discovery import amd_gpu_access_diagnostic, amd_gpu_present
from nerdit.utils import install_layout

# Includes the WSL2 passthrough path ``/usr/lib/wsl/lib`` (see NVIDIA_LIB_DIRS).
_LIB_DIRS = NVIDIA_LIB_DIRS

# ---------------------------------------------------------------------------
# Stable check names (the install map in the command keys on these)
# ---------------------------------------------------------------------------

NAME_PYTHON = "Python ≥3.11"
NAME_DOCKER = "Docker Engine"
NAME_BUILDX = "Docker Buildx (image builds)"
NAME_NVIDIA_DRIVER = "NVIDIA Driver (≥535)"
NAME_NVIDIA_LIB = "libnvidia-ml.so.1"
NAME_NVIDIA_TOOLKIT = "NVIDIA Container Toolkit"
NAME_RUNTIME_IMAGE = "nerdit-runtime:0.1"
NAME_ZML_SMI = "zml-smi JSON backend"
NAME_AMD_ACCESS = "AMD /dev/kfd access"
NAME_PORT = "Daemon port"
NAME_DISK = "Disk space"
NAME_CADDY = "Caddy (URL layer)"
NAME_GIT = "git binary"
NAME_ZEROCONF = "zeroconf (mDNS)"
NAME_CONFIG = "Config file"

_WSL_INTEGRATION_HINT = (
    "enable Docker Desktop → Settings → Resources → WSL integration for this distro"
)

# On WSL2 the NVIDIA GPU is driven by the *Windows* host driver (the passthrough
# libs live in /usr/lib/wsl/lib); installing a Linux display driver inside the
# distro overwrites the passthrough stub and breaks GPU support. NVIDIA's own
# WSL docs warn against it — so never emit an "apt install nvidia-driver-…" hint
# under WSL2.
_WSL_NVIDIA_HINT = (
    "update the NVIDIA driver on Windows (WSL2 GPU support comes from the host "
    "driver — do not install a Linux driver inside the distro)"
)


@dataclass
class CheckResult:
    """The outcome of one dependency check.

    `status` is one of `ok` / `warn` / `fail` / `skip`. `remediation`
    is an exact command or config line (or `None` when nothing is actionable,
    e.g. an `ok` row).
    """

    name: str
    status: str  # 'ok' | 'warn' | 'fail' | 'skip'
    detail: str
    remediation: str | None = None


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def detect_wsl2() -> bool:
    """Return True when running under WSL2 (`microsoft` in `/proc/version`).

    Cached (read once). False on any `OSError` or on a non-Linux platform.
    """
    if sys.platform != "linux":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def platform_label(platform: str | None = None) -> str:
    """Human-readable platform label: `macOS` / `Linux` / `WSL2`."""
    plat = platform if platform is not None else sys.platform
    if plat == "darwin":
        return "macOS"
    if plat == "linux":
        return "WSL2" if detect_wsl2() else "Linux"
    return plat


# ---------------------------------------------------------------------------
# Individual checks — each returns a CheckResult
# ---------------------------------------------------------------------------


def check_python(platform: str | None = None) -> CheckResult:
    """Python ≥ 3.11. Remediation is an exact, platform-resolved command."""
    plat = platform if platform is not None else sys.platform
    major, minor, micro = sys.version_info[0], sys.version_info[1], sys.version_info[2]
    ver = f"{major}.{minor}.{micro}"
    if major >= 3 and minor >= 11:
        return CheckResult(NAME_PYTHON, "ok", ver)
    remediation = "brew install python@3.11" if plat == "darwin" else "sudo apt install python3.11"
    return CheckResult(NAME_PYTHON, "fail", f"{ver} (need ≥3.11)", remediation)


def check_nvidia_driver(is_wsl2: bool = False) -> CheckResult:
    """NVIDIA driver via `nvidia-smi` (needs ≥535)."""
    remediation = (
        _WSL_NVIDIA_HINT if is_wsl2 else "sudo apt install nvidia-driver-535 (then reboot)"
    )
    if shutil.which("nvidia-smi") is None:
        return CheckResult(NAME_NVIDIA_DRIVER, "fail", "nvidia-smi not found", remediation)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return CheckResult(NAME_NVIDIA_DRIVER, "fail", "nvidia-smi failed", remediation)
        driver_ver = result.stdout.strip().splitlines()[0]
        major = int(driver_ver.split(".")[0])
        if major >= 535:
            return CheckResult(NAME_NVIDIA_DRIVER, "ok", driver_ver)
        return CheckResult(NAME_NVIDIA_DRIVER, "fail", f"{driver_ver} (need ≥535)", remediation)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        return CheckResult(NAME_NVIDIA_DRIVER, "fail", "nvidia-smi error", remediation)


def check_nvidia_lib(is_wsl2: bool = False) -> CheckResult:
    """`libnvidia-ml.so.1` on the host."""
    for d in _LIB_DIRS:
        p = Path(d) / "libnvidia-ml.so.1"
        if p.exists():
            return CheckResult(NAME_NVIDIA_LIB, "ok", str(p))
    remediation = (
        _WSL_NVIDIA_HINT if is_wsl2 else "provided by the NVIDIA driver — try: sudo ldconfig"
    )
    return CheckResult(NAME_NVIDIA_LIB, "fail", "not found", remediation)


def _docker_absent_remediation(platform: str, is_wsl2: bool) -> str:
    if is_wsl2:
        return _WSL_INTEGRATION_HINT
    if platform == "darwin":
        return (
            "install Docker Desktop (or OrbStack/colima) and start it: "
            "https://docs.docker.com/desktop/setup/install/mac-install/"
        )
    return "curl -fsSL https://get.docker.com | sudo sh"


def _docker_daemon_remediation(platform: str, is_wsl2: bool) -> str:
    if is_wsl2:
        return _WSL_INTEGRATION_HINT
    if platform == "darwin":
        return "start Docker Desktop"
    return "sudo systemctl start docker"


def _docker_version() -> str:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "running"


def check_docker(platform: str, is_wsl2: bool = False) -> CheckResult:
    """Docker Engine — distinguishes binary-absent / daemon-down / no-permission."""
    if shutil.which("docker") is None:
        return CheckResult(
            NAME_DOCKER,
            "fail",
            "docker not on PATH",
            _docker_absent_remediation(platform, is_wsl2),
        )
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult(
            NAME_DOCKER,
            "fail",
            "docker info failed",
            _docker_daemon_remediation(platform, is_wsl2),
        )
    if result.returncode == 0:
        return CheckResult(NAME_DOCKER, "ok", _docker_version())
    raw = result.stderr or ""
    stderr = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw).lower()
    if "permission denied" in stderr:
        return CheckResult(
            NAME_DOCKER,
            "fail",
            "permission denied on the Docker socket",
            "sudo usermod -aG docker $USER, then log out and back in",
        )
    if "cannot connect" in stderr or "is the docker daemon running" in stderr:
        return CheckResult(
            NAME_DOCKER,
            "fail",
            "Docker daemon not running",
            _docker_daemon_remediation(platform, is_wsl2),
        )
    return CheckResult(
        NAME_DOCKER,
        "fail",
        "docker info failed",
        _docker_daemon_remediation(platform, is_wsl2),
    )


def check_buildx() -> CheckResult:
    """Check BuildKit support, required for deployments.

    Fail if missing; skip when Docker is unusable because its check diagnoses that.
    """
    if shutil.which("docker") is None:
        return CheckResult(NAME_BUILDX, "skip", "Docker not installed")
    try:
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult(NAME_BUILDX, "skip", "Docker not available")
    if result.returncode == 0:
        version = (result.stdout or "").strip().splitlines()
        # (Codex 3804646855) State the scope of the claim: this probe runs with
        # the interactive user's HOME/DOCKER_CONFIG, while builds inherit the
        # daemon service's environment — a per-user ~/.docker/cli-plugins
        # install passes here and still fails every deploy. The qualifier lives
        # in ``detail`` because ``cli/commands/check_deps.py`` only prints
        # ``remediation`` for non-``ok`` rows.
        found = version[0][:64] if version else "available"
        return CheckResult(
            NAME_BUILDX,
            "ok",
            f"{found} (this user's environment; `nerdit doctor` reports the daemon's)",
        )
    stderr = (result.stderr or "").lower()
    if "cannot connect" in stderr or "permission denied" in stderr:
        return CheckResult(NAME_BUILDX, "skip", "docker unavailable (see the Docker Engine row)")
    return CheckResult(
        NAME_BUILDX,
        "fail",
        "buildx plugin not available",
        "install the buildx CLI plugin system-wide (apt: sudo apt-get install "
        "docker-buildx-plugin; macOS: Docker Desktop ships it) — a daemon running "
        "under a service-account HOME will not see a per-user ~/.docker/cli-plugins "
        "install",
    )


def check_nvidia_toolkit() -> CheckResult:
    """NVIDIA Container Toolkit registered as a Docker runtime."""
    if shutil.which("docker") is None:
        return CheckResult(NAME_NVIDIA_TOOLKIT, "skip", "Docker not installed")
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult(NAME_NVIDIA_TOOLKIT, "skip", "Docker not available")
    if result.returncode != 0:
        # Cause-neutral: the Docker Engine row already carries the exact
        # diagnosis (daemon down vs permission denied). Naming "daemon down?"
        # here misdirects in the permission-denied case.
        return CheckResult(
            NAME_NVIDIA_TOOLKIT, "skip", "docker unavailable (see the Docker Engine row)"
        )
    if "nvidia" in result.stdout:
        return CheckResult(NAME_NVIDIA_TOOLKIT, "ok", "registered")
    return CheckResult(
        NAME_NVIDIA_TOOLKIT,
        "fail",
        "not found in Docker runtimes",
        "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
    )


def check_runtime_image() -> CheckResult:
    """The `nerdit-runtime:0.1` default container image."""
    if shutil.which("docker") is None:
        return CheckResult(NAME_RUNTIME_IMAGE, "skip", "Docker not installed")
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "nerdit-runtime:0.1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return CheckResult(NAME_RUNTIME_IMAGE, "skip", "Docker not available")
    if result.returncode == 0:
        return CheckResult(NAME_RUNTIME_IMAGE, "ok", "available")
    # A non-zero return does not always mean "no such image": with the daemon
    # down or the socket permission-denied, inspect exits non-zero with a
    # connect error. Claiming "image not found" (unknowable) + a build hint that
    # itself fails would mislead. Degrade to skip like check_nvidia_toolkit.
    stderr = (result.stderr or "").lower()
    if "cannot connect" in stderr or "permission denied" in stderr:
        return CheckResult(
            NAME_RUNTIME_IMAGE, "skip", "docker unavailable (see the Docker Engine row)"
        )
    return CheckResult(
        NAME_RUNTIME_IMAGE,
        "fail",
        "image not found",
        "docker build -t nerdit-runtime:0.1 docker/",
    )


def check_zml_smi(executable: str = "zml-smi") -> CheckResult:
    """Optional vendor-neutral zml-smi JSON capability (never gates the run)."""
    resolved = executable if "/" in executable else shutil.which(executable)
    if not resolved:
        return CheckResult(NAME_ZML_SMI, "skip", f"{executable} not found (optional)")
    try:
        result = subprocess.run(
            [resolved, "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return CheckResult(NAME_ZML_SMI, "skip", "zml-smi --json failed")
        payload = json.loads(result.stdout)
        devices = payload.get("devices") if isinstance(payload, dict) else None
        if not isinstance(devices, list):
            return CheckResult(NAME_ZML_SMI, "skip", "invalid JSON response")
        targets = sorted(
            next(iter(device))
            for device in devices
            if isinstance(device, dict) and len(device) == 1
        )
        target_detail = f" ({', '.join(targets)})" if targets else ""
        return CheckResult(NAME_ZML_SMI, "ok", f"{len(devices)} device(s){target_detail}")
    except (json.JSONDecodeError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return CheckResult(NAME_ZML_SMI, "skip", "zml-smi error")


def check_amd_access() -> CheckResult:
    """Whether the current user can access AMD's KFD device (never gates)."""
    if not amd_gpu_present():
        return CheckResult(NAME_AMD_ACCESS, "skip", "no AMD display GPU detected")
    problem = amd_gpu_access_diagnostic()
    if problem:
        return CheckResult(
            NAME_AMD_ACCESS,
            "warn",
            problem,
            "add your user to the render/video groups (sudo usermod -aG render,video $USER)",
        )
    return CheckResult(NAME_AMD_ACCESS, "ok", "/dev/kfd is readable and writable")


def _port_process(port: int) -> str | None:
    """Best-effort listener process name for `port` via `lsof`. Tolerates
    lsof being absent.
    """
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for line in result.stdout.splitlines():
        if line.startswith("COMMAND"):
            continue
        parts = line.split()
        if parts:
            return parts[0]
    return None


def _nerditd_answers(port: int) -> bool:
    """True when `127.0.0.1:<port>/health` responds with a nerdit-shaped body."""
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(body, dict)
        and "version" in body
        and ("jobs_running" in body or "gpu_count" in body)
    )


def _port_listening(port: int) -> bool:
    """Check for a TCP listener on loopback.

    Use connect rather than bind: BSD SO_REUSEADDR allows a second bind over
    uvicorn's live listener and would falsely report the port free.
    """
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def check_port(port: int = DEFAULT_PORT) -> CheckResult:
    """Probe the configured daemon port. A live nerditd is `ok`; any other
    occupant is a `fail` with the process name (when discoverable).
    """
    # Connect-first: catches live listeners even where a SO_REUSEADDR bind
    # would falsely succeed (macOS, see _port_listening). The bind probe below
    # still catches non-loopback binds a loopback connect cannot see.
    occupied = _port_listening(port)
    if not occupied:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # SO_REUSEADDR models the daemon's own listener: a TIME_WAIT
            # socket left by a just-stopped nerditd must not read as a live
            # occupant (the daemon itself would bind fine over it).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
        except OSError:
            occupied = True
        finally:
            sock.close()
    if occupied:
        # Port is taken. Distinguish an already-running nerditd from a stranger.
        if _nerditd_answers(port):
            return CheckResult(NAME_PORT, "ok", f"nerditd already running on port {port}")
        proc = _port_process(port)
        by = f" (by {proc})" if proc else ""
        return CheckResult(
            NAME_PORT,
            "fail",
            f"port {port} is in use{by}",
            "stop it or set [daemon].port in ~/.nerdit/config.toml",
        )
    return CheckResult(NAME_PORT, "ok", f"port {port} is free")


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing directory at or above `path` (for disk probing a
    not-yet-created data dir).
    """
    p = path
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    return p


def check_disk(data_dir: str) -> CheckResult:
    """Free space on the data-dir filesystem. Thresholds mirror `/doctor`
    exactly: warn <10 %, fail <5 %.
    """
    target = _existing_ancestor(Path(data_dir).expanduser())
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return CheckResult(NAME_DISK, "skip", "could not stat the data dir filesystem")
    free_pct = (usage.free / usage.total) if usage.total else 0.0
    pct = round(free_pct * 100, 1)
    remediation = f"free up disk space on {target}"
    if free_pct < 0.05:
        return CheckResult(NAME_DISK, "fail", f"{pct}% free", remediation)
    if free_pct < 0.10:
        return CheckResult(NAME_DISK, "warn", f"{pct}% free", remediation)
    return CheckResult(NAME_DISK, "ok", f"{pct}% free")


def check_caddy(platform: str, configured: str | None = None) -> CheckResult:
    """Find optional Caddy using the proxy's own binary-resolution rules.

    Report the resolved path, including a bundled binary absent from PATH.
    """
    resolved = install_layout.resolve_caddy_binary(configured or DEFAULT_CADDY_BINARY)
    if resolved is None:
        if platform == "darwin":
            hint = "brew install caddy"
        else:
            hint = "sudo apt install caddy (or see https://caddyserver.com/docs/install)"
        return CheckResult(
            NAME_CADDY,
            "warn",
            "not found (URL layer degrades to loopback)",
            hint,
        )
    try:
        result = subprocess.run(
            [resolved, "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip().split()[0]
            return CheckResult(NAME_CADDY, "ok", f"{version} ({resolved})")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return CheckResult(NAME_CADDY, "warn", f"{resolved} present but 'caddy version' failed")


def check_git(platform: str) -> CheckResult:
    """git binary — informational ([git].enabled gates /deploy/git + the store)."""
    if shutil.which("git") is not None:
        return CheckResult(NAME_GIT, "ok", "available")
    hint = "brew install git" if platform == "darwin" else "sudo apt install git"
    return CheckResult(
        NAME_GIT,
        "warn",
        "not found (git deploy + template store disabled)",
        hint,
    )


def check_zeroconf(is_wsl2: bool = False) -> CheckResult:
    """The optional `zeroconf` extra used for mDNS advertising."""
    if is_wsl2:
        return CheckResult(
            NAME_ZEROCONF,
            "skip",
            "mDNS unsupported on WSL2 (zeroconf does not cross the NAT)",
        )
    if importlib.util.find_spec("zeroconf") is not None:
        return CheckResult(NAME_ZEROCONF, "ok", "installed")
    return CheckResult(
        NAME_ZEROCONF,
        "warn",
        "not installed (LAN mDNS advertising disabled)",
        'pip install "nerdit[mdns]"',
    )


def _nvidia_gpu_present() -> bool:
    """True when an NVIDIA GPU is detectable — `nvidia-smi` on PATH or a
    `/dev/nvidia*` device node. CPU-only hosts (a first-class supported mode)
    have neither, and must not be shown gate-capable GPU-tooling rows.
    """
    if shutil.which("nvidia-smi") is not None:
        return True
    try:
        return any(Path("/dev").glob("nvidia*"))
    except OSError:
        return False


def _cpu_only_gpu_rows() -> list[CheckResult]:
    """The GPU-tooling rows downgraded to `skip` on a CPU-only host, so a
    GPU-less Linux box never fails the run nor gets harmful driver-install
    advice (mirrors how macOS omits these rows entirely).
    """
    detail = "no NVIDIA GPU detected (CPU-only mode works)"
    return [
        CheckResult(NAME_NVIDIA_DRIVER, "skip", detail),
        CheckResult(NAME_NVIDIA_LIB, "skip", detail),
        CheckResult(NAME_NVIDIA_TOOLKIT, "skip", detail),
        CheckResult(NAME_RUNTIME_IMAGE, "skip", detail),
    ]


def _wsl2_hints() -> list[CheckResult]:
    """Informational rows surfacing the two WSL2 config gotchas."""
    return [
        CheckResult(
            "WSL2 note (bridge_host)",
            "skip",
            "keep [models].bridge_host on 'auto' (docker0 gateway 172.17.0.1)",
        ),
        CheckResult(
            "WSL2 note (mdns)",
            "skip",
            "leave [proxy].mdns off — zeroconf does not cross WSL2's NAT",
        ),
    ]


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------


def run_checks(
    platform: str | None = None,
    *,
    port: int | None = None,
    data_dir: str | None = None,
) -> list[CheckResult]:
    """Assemble the platform-appropriate list of checks.

    GPU-tooling rows appear only on Linux (macOS is CPU-only by construction).
    `port` / `data_dir` default to the locally-configured daemon settings
    (tolerant of a missing config file).
    """
    plat = platform if platform is not None else sys.platform
    is_wsl2 = detect_wsl2() if plat == "linux" else False

    # Loading the config must never crash the diagnostic — a hand-mangled
    # ~/.nerdit/config.toml is exactly the broken state check-deps exists to
    # surface (and the port row's own remediation sends users to edit that
    # file). On a parse error, fall back to the defaults and emit a config row.
    config_row: CheckResult | None = None
    # (P30) The Caddy row must honour an explicitly configured binary the same
    # way ProxyManager does; unset (the common case) it stays None and
    # check_caddy falls back to the bundled-then-PATH default resolution.
    configured_caddy: str | None = None
    if port is None or data_dir is None:
        try:
            settings = load_settings()
            resolved_port = settings.daemon.port
            resolved_data_dir = settings.data_dir
            configured_caddy = settings.proxy.caddy_binary
        except Exception as exc:  # noqa: BLE001 — any parse/validation error
            config_row = CheckResult(
                NAME_CONFIG,
                "fail",
                f"could not parse ~/.nerdit/config.toml ({type(exc).__name__})",
                "fix the TOML syntax or delete the file to regenerate defaults",
            )
            resolved_port = DEFAULT_PORT
            resolved_data_dir = str(DEFAULT_DATA_DIR)
        if port is None:
            port = resolved_port
        if data_dir is None:
            data_dir = resolved_data_dir

    # (BUG-1) buildx reads directly after the runtime it extends: it is a build
    # prerequisite, not a GPU-conditional row, so it runs on every platform.
    results: list[CheckResult] = [
        check_python(plat),
        check_docker(plat, is_wsl2),
        check_buildx(),
    ]
    if config_row is not None:
        results.append(config_row)

    if plat == "linux":
        if _nvidia_gpu_present():
            results.append(check_nvidia_driver(is_wsl2))
            results.append(check_nvidia_lib(is_wsl2))
            results.append(check_nvidia_toolkit())
            results.append(check_runtime_image())
        else:
            results.extend(_cpu_only_gpu_rows())
        results.append(check_zml_smi())
        results.append(check_amd_access())

    results.append(check_port(port))
    results.append(check_disk(data_dir))
    results.append(check_caddy(plat, configured_caddy))
    results.append(check_git(plat))
    results.append(check_zeroconf(is_wsl2))

    if is_wsl2:
        results.extend(_wsl2_hints())

    return results
