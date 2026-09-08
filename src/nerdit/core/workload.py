"""Reconcile registered controllers on one periodic task.

Isolate each controller's failures so one cannot stall the others.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from nerdit.core.services import ServiceController

logger = logging.getLogger(__name__)


class WorkloadController(Protocol):
    """A workload controller reconciled by the manager on each loop tick."""

    async def reconcile(self) -> None: ...

    async def shutdown(self) -> None: ...


class WorkloadManager:
    """Drives `reconcile()` on the registered workload controllers from one loop."""

    def __init__(
        self,
        services: ServiceController,
        loop_interval: float = 2.0,
    ) -> None:
        self._controllers: list[WorkloadController] = [services]
        self._loop_interval = loop_interval
        self._task: asyncio.Task | None = None
        self._running = False

    def register(self, controller: WorkloadController) -> None:
        """Append a controller to the shared reconcile tick.

        Additive on purpose: `__init__` keeps its signature, so
        `bootstrap.build_controllers`'s return tuple is unchanged and a
        controller the lifespan only builds conditionally (the GitWatch poller)
        joins the loop without threading an optional through the bootstrap.
        The loop's per-controller `try/except` gives it crash isolation and
        `stop` shuts it down, both for free.
        """
        self._controllers.append(controller)

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start the reconcile loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("WorkloadManager started")

    async def stop(self) -> None:
        """Stop the loop and shut down every controller."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for controller in self._controllers:
            await controller.shutdown()
        logger.info("WorkloadManager stopped")

    async def _loop(self) -> None:
        """Main loop: reconcile each controller every `loop_interval` seconds."""
        while self._running:
            for controller in self._controllers:
                try:
                    await controller.reconcile()
                except Exception:
                    logger.exception("Error reconciling %s", type(controller).__name__)
            await asyncio.sleep(self._loop_interval)
