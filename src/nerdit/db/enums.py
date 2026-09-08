"""Shared workload enums and derived constants; stdlib-only to avoid import cycles."""

from __future__ import annotations

from enum import Enum

# --- Enums ---


class GpuStatus(str, Enum):
    """Lifecycle states for a GPU in the allocation pool."""

    idle = "idle"
    busy = "busy"
    shared = "shared"
    error = "error"
    offline = "offline"


class GpuVendor(str, Enum):
    """Hardware vendor reported by the discovery backend."""

    nvidia = "nvidia"
    amd = "amd"


class GpuDiscoveryBackend(str, Enum):
    """Backend that produced the current GPU inventory record."""

    zml_smi = "zml_smi"
    nvml = "nvml"


class JobStatus(str, Enum):
    """Lifecycle states for a job in the scheduler."""

    pending = "pending"
    scheduled = "scheduled"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    retrying = "retrying"
    # Service-mode states (P2). Batch never enters these.
    building = "building"
    degraded = "degraded"
    restarting = "restarting"
    stopped = "stopped"


class JobKind(str, Enum):
    """Workload kinds; batch remains only for loading legacy rows."""

    batch = "batch"
    service = "service"
    model = "model"
    database = "database"


# The desired-state ("managed") workload kinds the `ServiceController`
# reconciles — every kind except the legacy ``batch``. Widen this tuple, not the
# nine call sites, when a new managed kind lands (P15 added ``database``).
MANAGED_KINDS: tuple[JobKind, ...] = (JobKind.service, JobKind.model, JobKind.database)

# The same set as a raw SQL ``IN`` fragment, built from ``MANAGED_KINDS`` at
# import time so the kind list is never hand-written twice. The values are the
# enum ``.value`` strings quoted in the single-quote style the surrounding SQL
# literals use.
MANAGED_KINDS_SQL: str = "kind IN (" + ", ".join(f"'{k.value}'" for k in MANAGED_KINDS) + ")"


class TokenRole(str, Enum):
    """Authorization role carried by a scoped API token.

    `admin` may do anything (the legacy global token resolves to this).
    `submitter` may create and manage its own jobs. `readonly` may only
    read. Permissive-by-default: anonymous/legacy principals map to `admin`.
    """

    admin = "admin"
    submitter = "submitter"
    readonly = "readonly"


class ErrorClass(str, Enum):
    """Coarse failure category used by the dashboard for localized guidance."""

    oom = "OOM"
    gpu_fail = "GPU_FAIL"
    image_pull_fail = "IMAGE_PULL_FAIL"
    user_error = "USER_ERROR"
    timeout = "TIMEOUT"
    container_fail = "CONTAINER_FAIL"
    mount_missing = "MOUNT_MISSING"
    # (P21 D2) CUDA/HIP allocator OOM inside the container — distinct from the
    # cgroup-kill ``oom`` above: the fix is GPU sizing, not [deploy].memory_limit.
    gpu_oom = "GPU_OOM"
    # (BUG-1) The build/launch failed for a HOST reason — a missing BuildKit
    # builder, not the app's source. Distinct from user_error because the fix is
    # on the machine, and redeploying the same source cannot help.
    platform_error = "PLATFORM_ERROR"
    unknown = "UNKNOWN"


class LogStream(str, Enum):
    """Log origin: stdout, stderr, system, bounded crash capture or image build output."""

    stdout = "stdout"
    stderr = "stderr"
    system = "system"
    # Separate BuildKit output so runtime filtering excludes image-build logs.
    # Legacy build rows remain stdout; filtering cannot retroactively classify them.
    build = "build"
    # (P21 D1) The bounded tail read off a dead container just before it is
    # removed. Its own stream value is the delete-and-replace primitive: only the
    # latest crash's capture is kept, without touching the live-follow rows.
    crash = "crash"
