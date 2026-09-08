"""Database-specific launch configuration and readiness probes.

ServiceController owns container lifecycle; ResourceController pulls images.
Successful probes persist `db_ready` for the current container. Replacement
containers clear that flag before running; transient failures re-arm the probe.

"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from nerdit.config.settings import DatabasesSettings, ModelsSettings
from nerdit.core.data.backend import (
    DataBackend,
    DataNotReadyError,
    DataProvisionError,
)
from nerdit.core.eventlog import EventRecorder, record_job_event
from nerdit.core.events import EventBus
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.resources.controller import ResourceController
from nerdit.core.runtime.container import ContainerConfig
from nerdit.core.runtime.protocol import ContainerRuntime
from nerdit.db.enums import LogStream
from nerdit.db.queries import Queries
from nerdit.db.rows import Job

logger = logging.getLogger(__name__)


class DataController(ResourceController[DataBackend]):
    """Database companion to ServiceController, with per-container readiness."""

    ready_flag = "db_ready"
    kind_label = "database"

    def __init__(
        self,
        backend: DataBackend,
        runtime: ContainerRuntime,
        queries: Queries,
        *,
        extra_backends: dict[str, DataBackend] | None = None,
        default_backend: str | None = None,
        databases_settings: DatabasesSettings | None = None,
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
        self._settings = databases_settings or DatabasesSettings()
        # Databases share the model binding's bridge reachability settings.
        self._models_settings = models_settings or ModelsSettings()
        self._events = events

    # --- Launch-shape helpers -----------------------------------------------

    @property
    def bridge_host(self) -> str:
        """Host advertised to app containers by managed database bindings.

        Uses `[models].bridge_host`: auto selects the Docker bridge on Linux
        and `host.docker.internal` on macOS.
        """
        return self._models_settings.bridge_advertise_host

    def build_container_config(
        self, backend_name: str | None, service_name: str, host_port: int, env: dict[str, str]
    ) -> ContainerConfig:
        """Build the backend config with loopback and optional bridge binds.

        `None` selects the default backend. `env` is the launch allowlist;
        the backend adds its static values. The bridge bind allows app-container
        access without LAN exposure.
        """
        backend = self.backend_for({"backend": backend_name})
        config = backend.container_config(service_name, host_port, env)
        bind_ip = self._models_settings.bridge_bind_ip
        if bind_ip is not None:
            config.extra_port_bind_ips = [bind_ip]
        return config

    # --- ensure_ready (wire-protocol readiness probe) -------------------------

    async def on_launched(self, job: Job, container_id: str, host_port: int) -> None:
        """Clear readiness and start probing before marking the row running.

        Unlike re-adoption, replacement invalidates the persisted `db_ready`
        flag. Await the clear before the caller marks the row running, so /wait
        and bindings cannot treat a database still initializing as ready.
        Cancel and reap the previous probe before clearing; otherwise its success
        could restore stale readiness or prevent the new probe from starting.
        """
        old_task = self._ensure_tasks.pop(job.id, None)
        if old_task is not None:
            old_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await old_task
        # Fresh read-modify-write (mirrors the success write below): drop the
        # stale flag if present, never clobbering a concurrent config update.
        row = await self._queries.get_job(job.id)
        cfg = parse_job_config(row) if row is not None else parse_job_config(job)
        if cfg.get("db_ready"):
            cfg.pop("db_ready", None)
            await self._queries.update_job_config(job.id, json.dumps(cfg))
        # Fire the probe unconditionally for the new container (no db_ready gate),
        # still bounded to one attempt per container.
        if container_id in self._ensure_attempted or job.id in self._ensure_tasks:
            return
        self._ensure_attempted.add(container_id)

        async def _run() -> None:
            try:
                await self._ensure_ready(job, cfg, host_port, container_id)
            finally:
                self._ensure_tasks.pop(job.id, None)

        self._ensure_tasks[job.id] = asyncio.create_task(_run())

    async def _ensure_ready(self, job: Job, cfg: dict, host_port: int, container_id: str) -> None:
        """Probe over loopback and persist readiness for the current container.

        App containers use the separate bridge-host DSN. DataNotReadyError releases
        the attempt guard for a later tick. DataProvisionError records a permanent
        failure without changing status or emitting an audit action; the current
        backends do not raise it. Dumps and restores run on ServiceController and
        report DumpError independently of readiness.
        """
        backend = self.backend_for(cfg)
        host = "127.0.0.1"
        await self._queries.append_log(
            job.id, "Waiting for the database to become ready ...", LogStream.system
        )
        try:
            await backend.ensure_ready(host, host_port)
        except DataNotReadyError as exc:
            # Transient: re-arm so the next reconcile tick retries once the
            # server answers. Do not audit — this is not a real failure.
            self._ensure_attempted.discard(container_id)
            logger.debug(
                "Database %s not ready yet, will retry: %s", job.service_name or job.id, exc
            )
            return
        except DataProvisionError as exc:
            await self._queries.append_log(
                job.id, f"Database provisioning failed: {exc}", LogStream.system
            )
            await self._queries.set_job_error_message(
                job.id, f"Database provisioning failed: {exc}"
            )
            # Emit a fixed reason; backend exception text must not enter events.
            await record_job_event(
                self._events, "database.failed", job, reason="db_provision_failed"
            )
            logger.warning(
                "Database provisioning failed for %s: %s", job.service_name or job.id, exc
            )
            return
        # Re-read the row before writing so a concurrent config update (e.g. a
        # status-side write during the probe) is never clobbered with stale data.
        row = await self._queries.get_job(job.id)
        # Ignore success from a replaced container. None means the new id has
        # not been recorded yet and does not prove replacement.
        if row is not None and row.container_id is not None and row.container_id != container_id:
            logger.debug(
                "Skipping db_ready stamp for %s: container moved %s -> %s during probe",
                job.service_name or job.id,
                container_id,
                row.container_id,
            )
            return
        latest = parse_job_config(row) if row is not None else dict(cfg)
        latest["db_ready"] = True
        await self._queries.update_job_config(job.id, json.dumps(latest))
        await self._queries.append_log(
            job.id, "Database ready (accepting connections)", LogStream.system
        )
        # Emit here: POST /databases rows have no last_deploy readiness signal.
        await record_job_event(self._events, "database.ready", job)
        logger.info("Database %s ready", job.service_name or job.id)

    _ensure = _ensure_ready
