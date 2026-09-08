"""Resource monitor for periodic vendor-neutral GPU metrics collection."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from nerdit.core.discovery import GpuBackend
from nerdit.db.rows import GpuMetrics

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Collect GPU metrics from the selected discovery backend."""

    def __init__(
        self,
        backend: GpuBackend | None,
        interval_seconds: float = 5.0,
        temp_warning: int = 80,
        temp_critical: int = 90,
    ) -> None:
        self._backend = backend
        self._interval = interval_seconds
        self._temp_warning = temp_warning
        self._temp_critical = temp_critical
        self._metrics: dict[str, GpuMetrics] = {}
        self._task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = False

    async def start(self) -> None:
        """Start the monitoring loop when a GPU backend is available."""
        if self._task is not None:
            return
        self._running = True
        if self._backend is None:
            logger.warning("No GPU metrics backend available; monitoring is disabled")
            return
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nerdit-gpu")
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "Resource monitor started (backend=%s, interval=%ss)",
            self._backend.name,
            self._interval,
        )

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        logger.info("Resource monitor stopped")

    def get_metrics(self) -> dict[str, GpuMetrics]:
        """Return the latest cached metrics keyed by inventory ID."""
        return dict(self._metrics)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._collect()
            except Exception:
                logger.exception("Error collecting GPU metrics")
            await asyncio.sleep(self._interval)

    async def _collect(self) -> None:
        if self._backend is None:
            return
        loop = asyncio.get_running_loop()
        snapshots = await loop.run_in_executor(self._executor, self._backend.collect_metrics)
        self._metrics = snapshots
        for gpu_id, metrics in snapshots.items():
            temp = metrics.temperature_c
            if temp is None:
                continue
            if temp >= self._temp_critical:
                logger.critical("GPU %s at %d°C - CRITICAL", gpu_id, temp)
            elif temp >= self._temp_warning:
                logger.warning("GPU %s at %d°C - warning", gpu_id, temp)
