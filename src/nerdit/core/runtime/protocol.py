"""Container runtime protocol — abstraction layer for Docker/Podman."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Protocol

from nerdit.core.runtime.container import ContainerConfig


@dataclass(frozen=True)
class ContainerStateInfo:
    """Point-in-time snapshot of a container's `State` (crash forensics).

    Read from `container.attrs['State']` right after a service/model container
    dies, *before* it is removed, so the reconciler can persist the crashing
    `exit_code` and whether the OOM-killer fired. Fields mirror Docker's
    `State` object; all are best-effort and may be `None` on a malformed or
    partial inspect.
    """

    exit_code: int | None
    oom_killed: bool
    error: str | None
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class ContainerStats:
    """Point-in-time resource sample for one running container.

    Every field is `Optional` on purpose: a runtime reports whatever its
    cgroup driver exposes, and a missing counter must project as `None` —
    **never** as `0`, which an operator (or an agent) would read as "idle"
    rather than "unknown" (the P14b `container_running` tri-state lesson).

    `cpu_pct` is the docker-CLI convention: the container's CPU-time delta
    over the system CPU-time delta, scaled by the number of online CPUs, so
    `100.0` means one core saturated and a 4-core host can report up to
    `400.0`. It is `None` whenever the delta cannot be derived — in
    particular on a first sample, where the previous system CPU reading is
    zero and the ratio is undefined (never a divide-by-zero).
    """

    cpu_pct: float | None
    mem_used_bytes: int | None
    mem_limit_bytes: int | None
    net_rx_bytes: int | None
    net_tx_bytes: int | None
    pids: int | None


class ContainerRuntimeError(Exception):
    """Base exception for container runtime errors."""


class ContainerNotFoundError(ContainerRuntimeError):
    """Raised when a container is not found."""


class ContainerStartError(ContainerRuntimeError):
    """Raised when a container fails to start."""


class BuildError(ContainerRuntimeError):
    """Raised when an image build fails.

    Subclasses `ContainerRuntimeError` so the service controller's
    existing runtime-error handling settles the build cleanly. The failing
    build-log line is carried in the message so it can be surfaced to the user.
    """


class BuildPlatformError(BuildError):
    """Report a host build-toolchain fault, preserving the BuildError catch contract.

    Missing Docker CLI or buildx requires machine repair, not source redeployment.
    The exception type selects remediation without matching localized build text.
    Other platform faults need distinct guidance before using this classification.
    """


class SandboxViolationError(ContainerStartError):
    """Raised when a launch request violates the container sandbox policy.

    Carries a machine-readable `reason` plus the offending host `path` and,
    for Tier-A denials, the `denied` policy entry that matched. Subclasses
    `ContainerStartError` so the scheduler's existing
    `ContainerRuntimeError` handling fails the job cleanly (GPUs released).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "sandbox_violation",
        path: str | None = None,
        denied: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.path = path
        self.denied = denied


class ContainerRuntime(Protocol):
    """Interface for container runtimes (Docker, Podman, etc.)."""

    async def run(self, config: ContainerConfig) -> str:
        """Launch a container and return its ID."""
        ...

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        """Stop a container gracefully."""
        ...

    async def kill(self, container_id: str) -> None:
        """Force-kill a container."""
        ...

    def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream container logs, bounding completed-container reads while consuming them.

        This regular method returns an async iterator directly, not an awaitable.
        Enforce the raw byte budget during reads: a huge newline-free output defeats
        a line-only cap, and slicing after buffering does not bound memory.

        Args:
            tail: Maximum trailing lines for a one-shot read, or None for no line cap.
            max_bytes: Maximum trailing bytes retained, or None for no byte cap.
            follow: Stream until disconnected, ignoring both one-shot caps.
        """
        ...

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        """Wait for a container to finish and return exit code. *timeout_s* bounds the wait;
        expiry raises
        `asyncio.TimeoutError`. Implementations must bound the *blocking
        call itself* — wrapping an unbounded wait in `asyncio.wait_for` only
        cancels the coroutine and leaks the still-blocked worker.
        """
        ...

    async def remove(self, container_id: str, force: bool = False) -> None:
        """Remove a container."""
        ...

    async def status(self, container_id: str) -> str | None:
        """Get the status of a container. Returns None if not found."""
        ...

    async def container_running(self, container_id: str) -> bool | None:
        """Tri-state liveness probe: `True` running, `False` inert, `None` unknown.

        Answers "could this container still be *writing*?" — `True` for the states
        that can touch a bind mount (running / restarting / paused, the last because
        it can be unpaused), `False` when it is gone or inert (exited / created /
        dead / removing), `None` when the runtime could not answer at all.

        Distinct from `status` and `inspect_state`, which both collapse
        "not found" and "the runtime could not answer" into `None`. That is fine for
        a reconciler (an unanswerable probe just means "recheck next tick") but not
        for a caller about to take an irreversible action on the strength of "nothing
        is writing here" — hence the third state: `None` means *we do not know*, and
        such a caller must not assume absence.
        """
        ...

    async def inspect_state(self, container_id: str) -> ContainerStateInfo | None:
        """Return the container's `State` snapshot, or `None` if not found.

        Read-side companion to `status`; used for crash forensics
        (exit code / OOM). Returns `None` when the container no longer exists
        or on any inspect error (mirrors the `status` swallow convention).
        """
        ...

    async def stats(self, container_id: str) -> ContainerStats | None:
        """Return a one-shot resource sample for *container_id*, or `None`.

        Same swallow convention as `inspect_state`: `None` when the
        container no longer exists, when the runtime cannot answer, or when the
        payload is unreadable — a caller can never distinguish "gone" from
        "unavailable" here and must project both as *not available*, never as
        zeroes.

        Implementations MUST take a **single bounded sample** (docker's
        `stats(stream=False)`), never open a stream: the post-P15 lesson is
        that docker *follow* streams pin the shared thread pool and deadlock the
        daemon. The call still blocks ~1-2 s per container, so it is on-demand
        only and must NEVER be called from the reconcile tick;
        implementations additionally bound their own concurrency.
        """
        ...

    async def image_exists(self, image_name: str) -> bool:
        """Check if a Docker image exists locally."""
        ...

    async def pull_image(self, image: str) -> None:
        """Pull *image* from its registry (e.g. `ollama/ollama`).

        Blocks until the pull completes; idempotent when the image is already
        present. Raises `ContainerRuntimeError` on registry/daemon
        failures.
        """
        ...

    async def list_images(self) -> list[str]:
        """List locally available image tags (`repo:tag`), sorted."""
        ...

    async def list_images_detailed(self) -> list[dict]:
        """List local images as `{repo_tag, id, size_bytes, instance}` dicts.

        A richer projection than `list_images` (which stays a plain sorted
        `list[str]` and is untouched): one entry per `repo:tag`, carrying the
        image id and on-disk `size_bytes` for disk accounting / GC, plus
        `instance` — the id of the nerdit daemon that built the image (see
        `build_image`) or `None` when the image carries no such label.
        Consumers must treat a missing key as `None`. Errors degrade to an
        empty list.
        """
        ...

    async def disk_usage(self) -> dict[str, int] | None:
        """Return docker's aggregate disk usage, or `None` when unavailable.

        Wraps the daemon's `df` report into
        `{images_bytes, containers_bytes, volumes_bytes, build_cache_bytes}`.
        Returns `None` on any runtime/API error so the caller can fall back.
        """
        ...

    def build_image(
        self, context_dir: str, tag: str, dockerfile: str | None = None
    ) -> AsyncIterator[str]:
        """Build an image from *context_dir*, tagging it *tag*.

        Yields decoded build-log lines in real time. *dockerfile* is the name
        of the Dockerfile relative to *context_dir* (defaults to `Dockerfile`).
        Raises `BuildError` if the build fails.

        An implementation MUST stamp the building daemon's ownership labels on
        the resulting image (`managed-by` + `nerdit-instance`), so that
        `list_images_detailed` can report `instance` and the image GC
        never reclaims a co-located sibling daemon's images.

        Declared `def` for the same reason as `logs`: implementations
        are async generators, so the call returns an `AsyncIterator` and is
        never awaited.
        """
        ...

    async def buildx_available(self) -> str:
        """`'present' | 'missing' | 'no_cli' | 'unknown'` — build-toolchain state.

        (BUG-1) Multi-state on purpose: `unknown` is the fail-open value for a
        probe that could not conclude (a timeout, a failed spawn), and must
        never be read as "missing" — that would reclassify a genuine app build
        error as a host fault.

        (Codex 3804646811) `no_cli` is the opposite kind of answer: the daemon
        has no `docker` CLI on its `PATH`, which is a **definite host
        fault** — nothing on this node can build — and so must not be folded
        into the fail-open `unknown`.
        """
        ...

    async def remove_image(self, tag: str, force: bool = False) -> None:
        """Remove a local image by tag. Silently ignores a missing image."""
        ...

    async def list_managed_containers(self) -> list[tuple[str, datetime]]:
        """List ALL `managed-by=nerdit` containers as `(id, created_at)` pairs
        (unscoped — used for re-adoption; the controller matches ids to its DB)."""
        ...

    async def list_own_managed_containers(self) -> list[tuple[str, datetime]]:
        """List only THIS daemon's `nerdit-instance` containers (for the zombie
        sweep, which kills — it must never see a co-located sibling's containers)."""
        ...

    async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
        """Ids of THIS daemon's running containers carrying `label=value`
        (instance-scoped — feeds the crash-orphan kill path, so it must never
        see a co-located sibling's containers)."""
        ...

    async def list_own_run_containers(self) -> list[str]:
        """Ids of THIS daemon's RUNNING containers carrying `nerdit-run` (ANY value).

        Presence-of-label listing (unlike `list_own_labeled_containers`,
        which pins an exact value): `managed-by=nerdit` AND
        `nerdit-instance=<this daemon>` AND the bare `nerdit-run` key. Feeds
        the boot-side orphan kill, so — like every kill path — it must never
        surface a co-located sibling daemon's containers. Running-only is
        exactly the reap set (an already-exited orphan is inert), and there is
        deliberately NO `Created`-timestamp parsing: a kill path must not skip
        a container because a timestamp failed to parse.
        """
        ...
