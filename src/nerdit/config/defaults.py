"""Default constants for Nerdit configuration."""

import fnmatch
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9321
DEFAULT_DATA_DIR = Path("~/.nerdit")
DEFAULT_DB_NAME = "nerdit.db"
DEFAULT_LOG_LEVEL: Final = "info"

DEFAULT_IMAGE = "nerdit-runtime:0.1"
# Not a [containers] key: the sandbox mount allow-list below is its only reader.
DEFAULT_CACHE_DIR = "~/.nerdit/cache"

# Shared NVIDIA-library probe paths, including WSL2 passthrough.
# Missing directories are skipped; daemon, init and check-deps use the same list.
NVIDIA_LIB_DIRS = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/lib/wsl/lib",
]

# --- Deploy / Builder (P4) ---
# Per-app images built by `nerdit deploy` are tagged ``nerdit-app/{name}:{n}``.
APP_IMAGE_REPO_PREFIX = "nerdit-app"


def app_image_repo(name: str) -> str:
    """Image repository for a deployed app, e.g. `nerdit-app/my-app`."""
    return f"{APP_IMAGE_REPO_PREFIX}/{name}"


def app_image_tag(name: str, version: int) -> str:
    """Full image tag for a deployed app build, e.g. `nerdit-app/my-app:3`."""
    return f"{APP_IMAGE_REPO_PREFIX}/{name}:{version}"


# --- Git deploy (P11.5) ---
# Deny-by-default egress posture (D4): a submitter can only make the daemon
# clone from these hosts, so it cannot be steered at arbitrary LAN/internal
# hosts. Self-hosted forges are one config line.
DEFAULT_GIT_ALLOWED_HOSTS = ["github.com"]
# Hard wall-clock cap on each clone subprocess. Kept below the client-side
# 180s (O3) so the structured ``deploy.git_timeout`` envelope wins over a raw
# client disconnect.
DEFAULT_GIT_CLONE_TIMEOUT_S = 120

DEFAULT_MONITOR_INTERVAL = 5  # seconds
DEFAULT_GPU_TEMP_WARNING = 80  # °C
DEFAULT_GPU_TEMP_CRITICAL = 90  # °C

# --- Service mode (P2) ---
# Host-port range services bind on (loopback). Must exclude DEFAULT_PORT 9321
# (the daemon's own port); the ``[services]`` validator enforces this.
DEFAULT_SERVICE_PORT_RANGE = "9400-9499"
DEFAULT_SERVICE_MAX_RESTARTS = 3  # restarts allowed within the rate window
DEFAULT_RESTART_WINDOW_SECONDS = 300  # 5 min rate window for restart counting

# --- One-off runs / release hook (P20) ---
# Server cap on a run request's ``timeout_s`` (D-C: the timeout is mandatory and
# server-enforced, never client-honored). 30 min covers a slow data migration
# while keeping a single run from pinning a container indefinitely.
DEFAULT_RUN_TIMEOUT_MAX_S = 1800
# Wall-clock bound on one ``[deploy].release`` execution. Deliberately tighter
# than a run: a release blocks the deploy it gates, so an unbounded migration
# would wedge the generation instead of settling it ``failed``.
DEFAULT_RELEASE_TIMEOUT_S = 300
# Daemon-wide cap on route-initiated runs (D-P20-2). Releases are exempt: a
# deploy must never fail because unrelated runs are in flight. Note this leaves
# releases bounded by NO cap at all — ``max_concurrent_builds`` is released
# before the release container starts (``core/app_build.py``), so concurrent
# migrations are bounded only by how many services deploy at the same moment.
DEFAULT_MAX_CONCURRENT_RUNS = 4

# --- Managed-database dumps (P37) ---
# Server cap on a dump/restore request's ``timeout_s`` (D-P37-8, the P20
# ``run_timeout_max_s`` mirror). An hour: a logical dump of a multi-GB database
# over the loopback bridge is minutes, but a restore into a busy Postgres can
# wait on locks, and the alternative to a generous cap is an operator who
# cannot finish a restore at all.
DEFAULT_DUMP_TIMEOUT_MAX_S = 3600
# Daemon-wide cap on concurrent dump/restore siblings (D-P37-9). Deliberately
# tighter than ``max_concurrent_runs``: each sibling streams a whole database
# through the loopback bridge and writes it to the daemon's own disk, so two is
# already enough to saturate a small box. Dumps are exempt from the run cap and
# runs are exempt from this one — the two pools never borrow from each other.
DEFAULT_MAX_CONCURRENT_DUMPS = 2

# --- URL layer / ProxyManager (P3) ---
# Embedded Caddy reverse proxy, driven via its admin API (loopback only).
DEFAULT_PROXY_ADMIN_ADDR = "localhost:2019"  # Caddy admin API (never file reload)
DEFAULT_PROXY_HTTPS_PORT = 443  # needs setcap/root; degrade-on-bind-failure
DEFAULT_CADDY_BINARY = "caddy"
DEFAULT_PROXY_RECONCILE_INTERVAL = 5.0  # seconds between proxy reconcile ticks

# --- Model serving (P5 / P11) ---
# Ollama is the first ModelBackend (default); vLLM is the second (P11). Models
# are served as kind=model workloads; a row's config['backend'] picks which.
DEFAULT_OLLAMA_IMAGE = "ollama/ollama"
DEFAULT_MODELS_BACKEND = "ollama"  # backend used when a serve omits --backend.
# vLLM's OpenAI server image; it serves exactly one model per container.
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:latest"
# vLLM needs more /dev/shm than Docker's 64MB default (worker IPC / NCCL).
DEFAULT_VLLM_SHM_SIZE = "1g"
# Auto model bridge: Linux binds/advertises 172.17.0.1; macOS adds no bind
# and advertises host.docker.internal through its VM proxy.
# Explicit bridge_host overrides both roles for custom/rootless networks.
DEFAULT_MODELS_BRIDGE_HOST = "auto"
LINUX_DOCKER_BRIDGE_GATEWAY = "172.17.0.1"
DOCKER_DESKTOP_HOST_ALIAS = "host.docker.internal"


def default_bridge_binding() -> tuple[str | None, str]:
    """Return the platform's default bind IP and advertised bridge host.

    Use sys.platform so resolution works before Docker is initialized.
    """
    if sys.platform == "darwin":
        # Binding 172.17.0.1 would fail outright (no such host interface);
        # binding nothing extra is strictly less exposure.
        return None, DOCKER_DESKTOP_HOST_ALIAS
    return LINUX_DOCKER_BRIDGE_GATEWAY, LINUX_DOCKER_BRIDGE_GATEWAY


DEFAULT_MODEL_PULL_TIMEOUT_S = 1800  # `ollama pull` of ~5 GB weights on slow links
DEFAULT_MODEL_START_PERIOD_S = 300  # health grace period covering model load

# --- Managed data (P15 / P15.5) ---
# Postgres is the first DataBackend (default); Redis is the second (P15.5). A
# database is served as a kind=database workload; a row's config['backend'] picks
# which. Images are pinned to a widely-exercised tag (the non-root/PGDATA
# entrypoint path D-P15-6 leans on the debian ``postgres:16`` image).
DEFAULT_POSTGRES_IMAGE = "postgres:16"
DEFAULT_REDIS_IMAGE = "redis:7"
DEFAULT_DATABASES_BACKEND = "postgres"  # backend used when a create omits --backend.
DEFAULT_DB_START_PERIOD_S = 30  # health grace period covering database startup
DEFAULT_DB_READY_TIMEOUT_S = 5  # per-attempt budget for the wire-protocol readiness probe

# Tunnel capabilities must expire within the relay's 900-second ceiling.
# The 600-second default leaves renewal slack.
DEFAULT_LINK_CAPABILITY_TTL_S = 600
# Renewal fires at ``expires_at - renew_margin_s`` (D-R8) — the daemon
# reconnects with freshly minted material before the relay's reactive expiry
# bites, so a live tunnel is never dropped for a merely stale capability.
DEFAULT_LINK_RENEW_MARGIN_S = 120

# Upload / ZIP constants (v0.2)
DEFAULT_UPLOAD_DIR = Path("~/.nerdit/uploads")
DEFAULT_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
UPLOAD_WARNING_BYTES = 100 * 1024 * 1024  # 100 MB
ZIP_EXCLUDE_PATTERNS = {".git", "data/", "*.pyc", "__pycache__", ".venv", "node_modules"}


def matches_exclude_pattern(parts: Sequence[str], *, casefold: bool = False) -> bool:
    """Check path components against ZIP_EXCLUDE_PATTERNS.

    Trailing-slash patterns match directory components exactly; other patterns use
    fnmatchcase. With casefold, lowercase components first. ZIP uploads preserve
    case; workspace writes fold it. Shared here to avoid core importing CLI.
    """
    folded = [part.casefold() for part in parts] if casefold else list(parts)
    for pattern in ZIP_EXCLUDE_PATTERNS:
        if pattern.endswith("/"):
            if pattern.rstrip("/") in folded:
                return True
        elif any(fnmatch.fnmatchcase(part, pattern) for part in folded):
            return True
    return False


# --- Agent workspaces (P29 / D-P29-5) ---
# Caps are module constants, not config knobs (the P17d grace-as-constant
# posture): no new knob until a real deployment needs one.
WORKSPACE_MAX_FILE_BYTES = 256 * 1024  # 256 KiB per file
WORKSPACE_MAX_TOTAL_BYTES = 10 * 1024 * 1024  # 10 MiB per workspace
WORKSPACE_MAX_FILES = 500

# Pre-parse workspace JSON cap: at most 3x decoded bytes for ensure_ascii
# expansion (astral surrogate pairs), plus 1 MiB for keys and structure.
# Reject larger bodies before reading because no legal batch can require them.
WORKSPACE_MAX_BODY_BYTES = 3 * WORKSPACE_MAX_TOTAL_BYTES + 1_048_576  # 32_505_856

# Host mounts denied to every caller, including admins. Check literal and
# symlink-resolved paths plus protected descendants. Root itself is denied
# without treating every absolute path as forbidden.
DEFAULT_DENIED_MOUNT_PATHS = [
    "/var/run/docker.sock",
    "/run/docker.sock",
    # Docker Desktop (macOS) keeps the real socket under the user's home.
    "~/.docker/run/docker.sock",
    "~/.docker/run",
    # The socket's parent dirs are denied too: mounting ``/run`` or ``/var/run``
    # exposes the real docker.sock (the ancestor-match in ``_match_denied`` also
    # rejects mounting a parent of any denied file, this makes it explicit).
    "/run",
    "/var/run",
    "~/.nerdit",
    "/etc",
    "/root",
    "/",
    "/proc",
    "/sys",
    "/boot",
    "/dev",
]

# Tier-B: roots under which non-admin (scoped-token) workloads may mount host
# paths. Only the daemon-managed upload/cache dirs are allowed; a caller's own
# workspace is never auto-appended.
DEFAULT_ALLOWED_MOUNT_ROOTS = [
    str(DEFAULT_UPLOAD_DIR),
    DEFAULT_CACHE_DIR,
]
