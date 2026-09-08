"""Tests for vendor-neutral resource monitor behavior."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from nerdit.core.monitor import ResourceMonitor
from nerdit.db.models import GpuMetrics


def _backend(temp=55):
    backend = MagicMock()
    backend.name = "test"
    backend.collect_metrics = MagicMock(
        return_value={
            "gpu:amd:0": GpuMetrics(
                gpu_id="gpu:amd:0",
                utilization_percent=25,
                memory_used_mb=1024,
                memory_total_mb=81920,
                temperature_c=temp,
            )
        }
    )
    return backend


@pytest.mark.asyncio
async def test_start_creates_task_stop_cancels_it():
    monitor = ResourceMonitor(backend=_backend(), interval_seconds=0.01)

    await monitor.start()
    assert monitor._task is not None
    assert monitor._running is True

    await monitor.stop()
    assert monitor._task is None
    assert monitor._running is False


@pytest.mark.asyncio
async def test_start_without_backend_does_not_crash(caplog):
    monitor = ResourceMonitor(backend=None, interval_seconds=0.01)
    with caplog.at_level(logging.WARNING):
        await monitor.start()

    assert monitor._task is None
    assert any("monitoring is disabled" in record.message for record in caplog.records)
    await monitor.stop()


@pytest.mark.asyncio
async def test_metrics_are_cached_and_retrievable():
    monitor = ResourceMonitor(backend=_backend(), interval_seconds=0.01)

    await monitor.start()
    await asyncio.sleep(0.05)
    metrics = monitor.get_metrics()
    await monitor.stop()

    assert metrics["gpu:amd:0"].utilization_percent == 25
    assert metrics["gpu:amd:0"].temperature_c == 55


@pytest.mark.asyncio
async def test_collection_failure_does_not_stop_loop(caplog):
    backend = _backend()
    expected_metrics = backend.collect_metrics.return_value
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return expected_metrics

    backend.collect_metrics.side_effect = fail_once
    monitor = ResourceMonitor(backend=backend, interval_seconds=0.01)

    with caplog.at_level(logging.ERROR):
        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

    assert backend.collect_metrics.call_count >= 2
    assert any("Error collecting GPU metrics" in record.message for record in caplog.records)
    assert "gpu:amd:0" in monitor.get_metrics()


@pytest.mark.asyncio
async def test_warning_threshold_logs_warning(caplog):
    monitor = ResourceMonitor(
        backend=_backend(temp=85),
        interval_seconds=0.01,
        temp_warning=80,
        temp_critical=90,
    )

    with caplog.at_level(logging.WARNING):
        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

    assert any("warning" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_critical_threshold_logs_critical(caplog):
    monitor = ResourceMonitor(
        backend=_backend(temp=95),
        interval_seconds=0.01,
        temp_warning=80,
        temp_critical=90,
    )

    with caplog.at_level(logging.CRITICAL):
        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

    assert any("CRITICAL" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    monitor = ResourceMonitor(backend=_backend(), interval_seconds=0.01)
    await monitor.start()
    await monitor.stop()
    await monitor.stop()

    assert monitor._task is None


@pytest.mark.asyncio
async def test_interval_honored(monkeypatch):
    observed_intervals: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay):
        observed_intervals.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("nerdit.core.monitor.asyncio.sleep", recording_sleep)

    monitor = ResourceMonitor(backend=_backend(), interval_seconds=0.25)
    await monitor.start()
    await real_sleep(0.05)
    await monitor.stop()

    assert 0.25 in observed_intervals
