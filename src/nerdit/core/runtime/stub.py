"""Stub container runtime — used when Docker is not available."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import AsyncIterator

from nerdit.core.runtime.container import ContainerConfig
from nerdit.core.runtime.protocol import (
    ContainerRuntimeError,
    ContainerStateInfo,
    ContainerStats,
)


class StubRuntime:
    """No-op runtime that raises on every operation.

    Activated automatically when Docker cannot be reached at daemon startup.
    """

    async def run(self, config: ContainerConfig) -> str:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def stop(self, container_id: str, timeout: int = 10) -> None:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def kill(self, container_id: str) -> None:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def logs(
        self,
        container_id: str,
        follow: bool = False,
        tail: int | None = None,
        max_bytes: int | None = None,
    ) -> AsyncIterator[str]:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")
        # Make this an async generator
        yield  # pragma: no cover

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def remove(self, container_id: str, force: bool = False) -> None:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def status(self, container_id: str) -> str | None:
        return None

    async def container_running(self, container_id: str) -> bool | None:
        # "Cannot tell" (`None`), never "definitively inert" (`False`) — this stub IS
        # the Docker-unreachable state (the daemon fell back to it because it could not
        # talk to Docker at startup), and an unreachable Docker cannot testify that a
        # container from a previous daemon run has stopped: containerd may still be
        # running it, still writing into its bind-mounted data dir. The stub holding no
        # container says nothing about what the HOST holds. Per the tri-state contract,
        # callers about to take an irreversible action on the strength of "nothing is
        # writing here" (the `DELETE ?purge=data` rmtree gate) fail closed on `None`.
        return None

    async def inspect_state(self, container_id: str) -> ContainerStateInfo | None:
        return None

    async def stats(self, container_id: str) -> ContainerStats | None:
        # There is no Docker to sample. `None` — never a zero-filled
        # `ContainerStats`, which the /stats route would surface as a live but
        # idle container instead of "stats unavailable" (tri-state posture).
        return None

    async def image_exists(self, image_name: str) -> bool:
        return True

    async def pull_image(self, image: str) -> None:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def list_images(self) -> list[str]:
        return []

    async def list_images_detailed(self) -> list[dict]:
        return []

    async def disk_usage(self) -> dict[str, int] | None:
        return None

    async def build_image(
        self,
        context_dir: str,
        tag: str,
        dockerfile: str | None = None,
        build_args: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")
        # Make this an async generator
        yield  # pragma: no cover

    async def buildx_available(self) -> str:
        # (BUG-1) There is no Docker to ask, so the honest answer is the
        # fail-open one: `unknown` never worsens a status and never accuses
        # the host of a missing plugin the stub simply cannot see. Deliberately
        # NOT `no_cli` either: the stub means Docker itself is unreachable
        # (doctor already `fail`s on that before this verdict is read), so it
        # must not claim a CLI fault it is in no position to observe.
        return "unknown"

    async def remove_image(self, tag: str, force: bool = False) -> None:
        raise ContainerRuntimeError("Docker is not available. Start Docker and restart the daemon.")

    async def list_managed_containers(self) -> list[tuple[str, datetime]]:
        return []

    async def list_own_managed_containers(self) -> list[tuple[str, datetime]]:
        return []

    async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
        return []

    async def list_own_run_containers(self) -> list[str]:
        return []
