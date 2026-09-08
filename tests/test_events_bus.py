"""Tests for the in-process EventBus used by the dashboard SSE."""

from __future__ import annotations

import asyncio

from nerdit.core.events import EventBus


async def test_subscriber_receives_published_event():
    bus = EventBus()
    received: list[dict] = []

    async def reader():
        sub = bus.subscribe()
        async for event in sub:
            received.append(event)
            if len(received) == 1:
                await sub.aclose()
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)  # let subscribe() run
    bus.publish({"type": "job.status_changed", "job_id": "abc", "status": "running"})
    await asyncio.wait_for(task, timeout=1.0)
    assert received == [{"type": "job.status_changed", "job_id": "abc", "status": "running"}]


async def test_multiple_subscribers_each_receive_event():
    bus = EventBus()
    out1: list[dict] = []
    out2: list[dict] = []

    async def reader(out: list[dict]):
        sub = bus.subscribe()
        async for event in sub:
            out.append(event)
            await sub.aclose()
            return

    t1 = asyncio.create_task(reader(out1))
    t2 = asyncio.create_task(reader(out2))
    await asyncio.sleep(0)
    bus.publish({"type": "ping"})
    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1.0)
    assert out1 == out2 == [{"type": "ping"}]


async def test_publish_with_no_subscribers_does_not_raise():
    bus = EventBus()
    bus.publish({"type": "nobody-home"})  # must not raise
    assert bus.subscriber_count == 0


async def test_unsubscribed_iterator_is_removed():
    bus = EventBus()

    async def short_reader():
        sub = bus.subscribe()
        await sub.aclose()

    await short_reader()
    assert bus.subscriber_count == 0


async def test_slow_subscriber_drops_events_instead_of_blocking():
    """When a subscriber's queue is full, publish must not stall the publisher."""
    bus = EventBus()
    sub = bus.subscribe()
    # Prime the subscriber so it exists in the set, but never consume.
    await asyncio.sleep(0)
    # Fill beyond the queue capacity (100) — none of these should block.
    for i in range(500):
        bus.publish({"i": i})
    # Publisher returned in finite time => OK.
    assert bus.subscriber_count == 1
    await sub.aclose()


async def test_reconcile_emits_job_status_changed_event(queries, mock_runtime):
    """End-to-end: reconciling a service to its desired 'stopped' state emits a
    job.status_changed event on the bus."""
    from nerdit.config.settings import ServicesSettings
    from nerdit.core.services import ServiceController
    from nerdit.db.models import Job, JobKind, JobStatus

    bus = EventBus()
    controller = ServiceController(
        queries=queries,
        runtime=mock_runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        event_bus=bus,
    )
    await queries.create_job(
        Job(
            id="evt-svc-1",
            name="evt-svc",
            kind=JobKind.service,
            service_name="evt-svc",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="stopped",
        )
    )

    received: list[dict] = []

    async def reader():
        sub = bus.subscribe()
        async for event in sub:
            received.append(event)
            if event.get("status") == "stopped":
                await sub.aclose()
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    await controller.reconcile()
    await asyncio.wait_for(task, timeout=1.0)
    await controller.shutdown()

    statuses = [e["status"] for e in received]
    assert "stopped" in statuses
    assert all(e["type"] == "job.status_changed" for e in received)
    assert all(e["job_id"] == "evt-svc-1" for e in received)


async def test_predicate_filters_events_at_source():
    """M2: a subscriber predicate drops non-matching events (audit.* isolation)."""
    bus = EventBus()
    received: list[dict] = []

    def not_audit(event: dict) -> bool:
        return not str(event.get("type", "")).startswith("audit.")

    async def reader():
        sub = bus.subscribe(not_audit)
        async for event in sub:
            received.append(event)
            if event.get("type") == "job.status_changed":
                await sub.aclose()
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    # An audit event must be dropped before reaching this subscriber...
    bus.publish({"type": "audit.token.create", "principal_id": "admin"})
    # ...while a normal cluster event is delivered.
    bus.publish({"type": "job.status_changed", "job_id": "j1", "status": "running"})
    await asyncio.wait_for(task, timeout=1.0)
    assert all(not e["type"].startswith("audit.") for e in received)
    assert received == [{"type": "job.status_changed", "job_id": "j1", "status": "running"}]
