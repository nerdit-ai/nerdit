"""Shared image-pull and readiness hooks for model and database services.

Subclasses supply the backend, readiness probe, config flag, and audit label.
Database readiness belongs to one container and is cleared on replacement;
model readiness records durable weights and survives replacement.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from nerdit.core.deploy_state import stamp_last_deploy
from nerdit.core.events import EventBus
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.resources.registry import BackendRegistry
from nerdit.core.runtime.protocol import ContainerRuntime, ContainerRuntimeError
from nerdit.db.enums import ErrorClass, JobStatus, LogStream
from nerdit.db.queries import Queries
from nerdit.db.rows import Job

logger = logging.getLogger(__name__)


class _NamedBackend(Protocol):
    """Backend attributes required for image selection."""

    name: str

    @property
    def image(self) -> str: ...


B = TypeVar("B", bound=_NamedBackend)


class ResourceController(ABC, Generic[B]):
    """Shared lifecycle hooks for model and database services.

    Readiness is persisted in `config[ready_flag]` so unfinished probes resume
    on daemon restart. Task maps prevent duplicate work within this process.
    """

    ready_flag: str
    kind_label: str

    def __init__(
        self,
        backend: B,
        runtime: ContainerRuntime,
        queries: Queries,
        *,
        extra_backends: dict[str, B] | None = None,
        default_backend: str | None = None,
        data_dir: str = "~/.nerdit",
        event_bus: EventBus | None = None,
    ) -> None:
        self._registry: BackendRegistry[B] = BackendRegistry(
            backend, extra_backends, default_backend, kind_label=self.kind_label.capitalize()
        )
        self._runtime = runtime
        self._queries = queries
        self._data_dir = str(Path(data_dir).expanduser())
        self._event_bus = event_bus
        # In-flight image pulls keyed by job id (idempotent per tick — the
        # reconcile loop calls ensure_image every tick while the row builds).
        self._pull_tasks: dict[str, asyncio.Task] = {}
        # In-flight ensure tasks keyed by job id.
        self._ensure_tasks: dict[str, asyncio.Task] = {}
        # Spent probe attempts, keyed by container so replacements can retry.
        self._ensure_attempted: set[str] = set()

    @property
    def backend(self) -> B:
        """The default backend; use backend_for for a persisted row."""
        return self._registry.default

    def get_backend(self, name: str | None) -> B | None:
        """Resolve an explicit name, returning None for unknown names.

        None selects the default. Write paths reject unknown names instead
        of using the reconcile path's fallback.
        """
        return self._registry.get(name)

    @property
    def default_backend_name(self) -> str:
        return self._registry.default_name

    @property
    def backends(self) -> list[str]:
        """Sorted registered backend names for capabilities."""
        return self._registry.names

    def backend_for(self, cfg: dict) -> B:
        """Resolve a row's backend, falling back to default if missing or unknown.

        Unknown names are logged so the reconcile loop can keep serving the row.
        """
        return self._registry.resolve(cfg)

    # --- Background image pull -----------------------------------------------

    async def ensure_image(self, job: Job, cfg: dict) -> bool:
        """Return whether the image is present; otherwise start one background pull.

        Use the row's explicit image or its resolved backend's image. The reconcile
        loop retries while the tracked pull is in flight.
        """
        image = str(cfg.get("image") or self.backend_for(cfg).image)
        if await self._runtime.image_exists(image):
            return True
        if job.id not in self._pull_tasks:
            self._spawn_pull_task(job, image)
        return False

    def _spawn_pull_task(self, job: Job, image: str) -> None:
        """Run an image pull off the reconcile tick, tracked by job id."""

        async def _run() -> None:
            try:
                await self._pull_image(job, image)
            finally:
                self._pull_tasks.pop(job.id, None)

        self._pull_tasks[job.id] = asyncio.create_task(_run())

    async def _pull_image(self, job: Job, image: str) -> None:
        """Pull the backend image, streaming progress/errors into `job_logs`.

        A pull failure (registry down, bad image ref) settles the row to a
        terminal `failed` exactly like a failed deploy build — never a retry
        loop; the `/restart` route revives it explicitly.
        """
        await self._queries.append_log(job.id, f"Pulling image {image} ...", LogStream.system)
        try:
            await self._runtime.pull_image(image)
        except ContainerRuntimeError as exc:
            await self._queries.append_log(job.id, f"Image pull failed: {exc}", LogStream.system)
            await self._queries.update_job_status(
                job.id,
                JobStatus.failed,
                finished_at=datetime.now(UTC),
                exit_code=-1,
                error_class=ErrorClass.image_pull_fail,
                error_message=f"Image pull failed: {exc}",
            )
            await self._queries.set_desired_state(job.id, JobStatus.failed.value)
            await self._stamp_last_deploy_failed(
                job.id,
                "image_pull_failed",
                ErrorClass.image_pull_fail.value,
                f"Image pull failed: {exc}",
            )
            self._emit_status_change(job.id, JobStatus.failed)
            await self._audit(f"{self.kind_label}.image_pull_failed", job)
            logger.warning(
                "Image pull failed for %s %s: %s",
                self.kind_label,
                job.service_name or job.id,
                exc,
            )
            return
        await self._queries.append_log(job.id, f"Image {image} pulled", LogStream.system)

    # --- ensure probe (weights pull / readiness probe) ------------------------

    def needs_ensure(self, job: Job) -> bool:
        """Whether a live row still needs an ensure-probe attempt.

        Cheap pre-check for the per-tick re-adoption path: false once
        `config[ready_flag]` is set, while an attempt is in flight, or after
        this container's one bounded attempt was already made.
        """
        if (
            not job.container_id
            or job.container_id in self._ensure_attempted
            or job.id in self._ensure_tasks
        ):
            return False
        return not bool(parse_job_config(job).get(self.ready_flag))

    def on_running(self, job: Job, container_id: str, host_port: int) -> None:
        """Start a background readiness probe for a launched or re-adopted row.

        Skip ready rows, in-flight probes, and containers with a spent attempt.
        Fresh database containers must first clear their per-container readiness
        through `DataController.on_launched`; model readiness is durable.
        """
        cfg = parse_job_config(job)
        if cfg.get(self.ready_flag):
            return
        if container_id in self._ensure_attempted or job.id in self._ensure_tasks:
            return
        self._ensure_attempted.add(container_id)

        async def _run() -> None:
            try:
                await self._ensure(job, cfg, host_port, container_id)
            finally:
                self._ensure_tasks.pop(job.id, None)

        self._ensure_tasks[job.id] = asyncio.create_task(_run())

    @abstractmethod
    async def _ensure(self, job: Job, cfg: dict, host_port: int, container_id: str) -> None:
        """Run the subclass readiness probe for one container."""

    # --- Shared helpers ---------------------------------------------------------

    async def _stamp_last_deploy_failed(
        self, job_id: str, reason: str, error_class: str | None, error_message: str
    ) -> None:
        """Mark the current deploy generation failed without overwriting newer config.

        Rows without `last_deploy` are unchanged.
        """
        await stamp_last_deploy(
            self._queries,
            job_id,
            phase="failed",
            reason=reason,
            error_class=error_class,
            error_message=error_message,
        )

    def _emit_status_change(self, job_id: str, status: JobStatus) -> None:
        """Best-effort emit on the event bus. Silent if no bus configured."""
        if self._event_bus is None:
            return
        self._event_bus.publish(
            {
                "type": "job.status_changed",
                "job_id": job_id,
                "status": status.value,
                "ts": datetime.now(UTC).isoformat(),
            }
        )

    async def _audit(self, action: str, job: Job) -> None:
        """Audit an autonomous transition as the system principal.

        Only the service name enters params; never include credentials.
        """
        try:
            await self._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type=self.kind_label,
                target_id=job.id,
                params_redacted=json.dumps({"service_name": job.service_name}),
            )
        except Exception:
            logger.warning(
                "Failed to audit %s for %s %s", action, self.kind_label, job.id, exc_info=True
            )

    async def shutdown(self) -> None:
        """Cancel in-flight pull/ensure tasks."""
        tasks = list(self._pull_tasks.values()) + list(self._ensure_tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._pull_tasks.clear()
        self._ensure_tasks.clear()
        logger.info("%s stopped", type(self).__name__)
