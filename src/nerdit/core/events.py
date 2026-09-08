"""Publish best-effort dashboard SSE events through bounded subscriber queues.

A slow subscriber loses events without blocking publishers. The daemon shares
one bus with workload controllers and the SSE route.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

_QUEUE_MAX = 100


class Subscription:
    """Async iterator backed by a bounded queue. Returned by `EventBus.subscribe`.

    An optional `predicate` filters which events reach this subscriber: events
    for which it returns `False` are never enqueued. This is how the general
    dashboard stream excludes admin-only `audit.*` events at the source,
    rather than relying on each route to re-filter (a least-privilege guard).
    """

    def __init__(
        self, bus: EventBus, predicate: Callable[[dict[str, Any]], bool] | None = None
    ) -> None:
        self._bus = bus
        self._predicate = predicate
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._closed = False

    def _push(self, event: dict[str, Any]) -> None:
        if self._predicate is not None and not self._predicate(event):
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Slow subscriber — drop the event rather than block the publisher.
            pass

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        return await self._queue.get()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            self._bus._remove(self)


class EventBus:
    """Fan-out async pub/sub. Best-effort delivery to bounded per-subscriber queues."""

    def __init__(self) -> None:
        self._subscribers: set[Subscription] = set()

    def publish(self, event: dict[str, Any]) -> None:
        """Fire-and-forget publish to every subscriber. Drops on overflow."""
        for sub in list(self._subscribers):
            sub._push(event)

    def subscribe(self, predicate: Callable[[dict[str, Any]], bool] | None = None) -> Subscription:
        """Register a new subscriber and return its iterator.

        `predicate` optionally restricts which events this subscriber sees
        (e.g. exclude `audit.*` from the general dashboard stream).
        """
        sub = Subscription(self, predicate)
        self._subscribers.add(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        self._subscribers.discard(sub)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
