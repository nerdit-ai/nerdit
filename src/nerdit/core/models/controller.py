"""Model-specific launch configuration and readiness probes.

ServiceController owns container lifecycle; ResourceController pulls images.
Successful weight pulls persist `model_pulled`. Unreachable servers re-arm
the probe; permanent pull failures wait for container replacement or daemon
restart, avoiding repeated downloads for an invalid model.

"""

from __future__ import annotations

import logging

from nerdit.config.settings import ModelsSettings
from nerdit.core.eventlog import EventRecorder, record_job_event
from nerdit.core.events import EventBus
from nerdit.core.models.backend import (
    ModelBackend,
    ModelPullError,
    ModelServerUnreachableError,
)
from nerdit.core.resources.controller import ResourceController
from nerdit.core.runtime.container import ContainerConfig
from nerdit.core.runtime.protocol import ContainerRuntime
from nerdit.db.enums import GpuVendor, LogStream
from nerdit.db.queries import Queries
from nerdit.db.rows import Job

logger = logging.getLogger(__name__)


class ModelController(ResourceController[ModelBackend]):
    """Model companion to ServiceController, with persisted weight readiness."""

    ready_flag = "model_pulled"
    kind_label = "model"

    def __init__(
        self,
        backend: ModelBackend,
        runtime: ContainerRuntime,
        queries: Queries,
        *,
        extra_backends: dict[str, ModelBackend] | None = None,
        default_backend: str | None = None,
        models_settings: ModelsSettings | None = None,
        data_dir: str = "~/.nerdit",
        event_bus: EventBus | None = None,
        events: EventRecorder | None = None,
    ) -> None:
        super().__init__(
            backend,
            runtime,
            queries,
            extra_backends=extra_backends,
            default_backend=default_backend,
            data_dir=data_dir,
            event_bus=event_bus,
        )
        self._settings = models_settings or ModelsSettings()
        self._events = events

    # --- Launch-shape helpers (used by ServiceController._launch) ------------

    @property
    def bridge_host(self) -> str:
        """Host [ai.*] ollama bindings advertise to app containers.

        Platform-resolved when `[models].bridge_host = "auto"` (Linux:
        docker0 gateway; macOS: `host.docker.internal`).
        """
        return self._settings.bridge_advertise_host

    def build_container_config(
        self,
        model: str,
        gpu_ids: list[str],
        vendor: GpuVendor,
        host_port: int,
        backend_name: str | None = None,
        gpu_memory_mb: int | None = None,
        *,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> ContainerConfig:
        """Build the backend launch config with loopback and optional bridge binds.

        `None` selects the default backend. Pass the smallest GPU's total VRAM
        (or `None` if unknown) and per-serve overrides to the backend.
        System-owned weights volumes bypass `enforce_mount_allowlist`; never
        extend that bypass to user-supplied mounts.
        """
        backend = self.backend_for({"backend": backend_name})
        config = backend.container_config(
            model,
            gpu_ids,
            vendor,
            host_port,
            self._data_dir,
            gpu_memory_mb,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        # Loopback stays the primary bind; the bridge gateway makes the endpoint
        # reachable from app containers without any LAN exposure.
        # No extra bind on platforms where none is needed (macOS: the VM proxy
        # already exposes loopback-published ports as host.docker.internal).
        bind_ip = self._settings.bridge_bind_ip
        if bind_ip is not None:
            config.extra_port_bind_ips = [bind_ip]
        return config

    # --- ensure_model (weights) -----------------------------------------------

    async def _ensure(self, job: Job, cfg: dict, host_port: int, container_id: str) -> None:
        """Pull weights over loopback and persist `model_pulled` on success.

        An unreachable server releases the attempt guard for a later reconcile
        tick. A permanent pull failure is logged and audited with the attempt
        spent. App containers use the separate bridge-host URL.
        """
        model = str(cfg.get("model") or "")
        backend = self.backend_for(cfg)
        base_url = f"http://127.0.0.1:{host_port}"
        await self._queries.append_log(
            job.id, f"Pulling model weights for '{model}' (idempotent) ...", LogStream.system
        )
        try:
            await backend.ensure_model(base_url, model)
        except ModelServerUnreachableError as exc:
            # Transient: re-arm so the next reconcile tick retries once the
            # server is listening. Do not audit — this is not a real failure.
            self._ensure_attempted.discard(container_id)
            logger.debug(
                "Model %s server not reachable yet, will retry: %s",
                job.service_name or job.id,
                exc,
            )
            return
        except ModelPullError as exc:
            await self._queries.append_log(job.id, f"Model pull failed: {exc}", LogStream.system)
            # Update only the error: job is a pre-launch snapshot, so restoring
            # its status could regress the live row and trigger container churn.
            await self._queries.set_job_error_message(job.id, f"Model pull failed: {exc}")
            # Fail the deploy generation for /wait and /diagnose. error_class
            # stays None: weights failed, not the container image pull.
            await self._stamp_last_deploy_failed(
                job.id, "model_pull_failed", None, f"Model pull failed: {exc}"
            )
            await self._audit("model.pull_failed", job)
            # Emit a fixed reason; registry exception text must not enter events.
            await record_job_event(self._events, "model.failed", job, reason="model_pull_failed")
            logger.warning("Model pull failed for %s: %s", job.service_name or job.id, exc)
            return
        # Keyed write: a concurrent config update during the pull is never clobbered.
        await self._queries.patch_job_config(job.id, {"model_pulled": True})
        await self._queries.append_log(
            job.id, f"Model '{model}' ready (weights present)", LogStream.system
        )
        # Emit here: POST /models rows have no last_deploy to signal readiness.
        await record_job_event(self._events, "model.ready", job)
        logger.info("Model %s weights ready", job.service_name or job.id)
