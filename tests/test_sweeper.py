"""Tests for ZombieSweeper — orphan managed-container sweep (all workload kinds).

Moved out of ``tests/test_repair.py`` when ``cleanup_zombies`` was split into
:mod:`nerdit.core.sweeper`.

(P20) ``extra_protected`` is a **required** keyword argument — the run registry
hook that keeps a rowless one-off run / ``[deploy].release`` container from
being reaped mid-flight. Every pre-P20 case passes the explicit no-op
:func:`_no_extra`, which must leave the sweep byte-identical; the P20 block at
the bottom covers the hook itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from nerdit.core import sweeper as sweeper_module
from nerdit.core.runtime.protocol import ContainerNotFoundError, ContainerRuntimeError
from nerdit.core.sweeper import ZOMBIE_AGE_THRESHOLD_SECONDS, ZombieSweeper
from nerdit.daemon.sweeps import _zombie_sweep_loop
from nerdit.db.models import Job, JobKind, JobStatus

# ---- helpers ----


def _no_extra() -> set[str]:
    """The explicit no-op run-registry hook (P20): nothing extra to protect."""
    return set()


def _no_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flatten the boot-kill settle delay (the release-bind race guard).

    The method reads the module constant at call time, so patching the module
    attribute is enough — no test has to sit through a real second.
    """
    monkeypatch.setattr(sweeper_module, "BOOT_KILL_SETTLE_SECONDS", 0)


def _old(age_seconds: float = ZOMBIE_AGE_THRESHOLD_SECONDS + 10) -> datetime:
    """Container created ``age_seconds`` ago — old enough to be swept."""
    return datetime.now(UTC) - timedelta(seconds=age_seconds)


def _young(age_seconds: float = 1.0) -> datetime:
    """Container created recently — inside the safety window."""
    return datetime.now(UTC) - timedelta(seconds=age_seconds)


# ---- cleanup_zombies ----


@pytest.mark.asyncio
async def test_cleanup_zombies_removes_unknown_containers(queries, mock_runtime):
    mock_runtime.list_own_managed_containers = AsyncMock(
        return_value=[("abcdef1234567890", _old())]
    )

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    mock_runtime.kill.assert_awaited_once_with("abcdef1234567890")
    mock_runtime.remove.assert_awaited_once_with("abcdef1234567890", force=True)


@pytest.mark.asyncio
async def test_cleanup_zombies_preserves_known_running_containers(queries, mock_runtime):
    job = Job(id="live-1", script_path="/tmp/train.py", gpu_count=1)
    await queries.create_job(job)
    await queries.update_job_status("live-1", JobStatus.running, container_id="known-container-id")

    mock_runtime.list_own_managed_containers = AsyncMock(
        return_value=[
            ("known-container-id", _old()),
            ("orphan-container-id", _old()),
        ]
    )

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    mock_runtime.kill.assert_awaited_once_with("orphan-container-id")
    mock_runtime.remove.assert_awaited_once_with("orphan-container-id", force=True)


@pytest.mark.asyncio
async def test_cleanup_zombies_protects_scheduled_jobs(queries, mock_runtime):
    """Regression for bug_001: a job briefly sits in ``scheduled`` with a
    container_id between ``runtime.run()`` returning and the DB write.
    """
    job = Job(id="sch-1", script_path="/tmp/train.py", gpu_count=1)
    await queries.create_job(job)
    await queries.update_job_status("sch-1", JobStatus.scheduled, container_id="fresh-cid")

    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[("fresh-cid", _old())])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    mock_runtime.kill.assert_not_awaited()
    mock_runtime.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_zombies_reaps_retrying_job_containers(queries, mock_runtime):
    """``retrying`` is NOT in the protected tuple (D-S-6).

    It was, historically, but the status is only ever produced by the batch
    auto-repair path that is on its way out; a container left behind by a
    retrying row is an orphan like any other and must be reaped.
    """
    job = Job(id="retry-1", script_path="/tmp/train.py", gpu_count=1)
    await queries.create_job(job)
    await queries.update_job_status("retry-1", JobStatus.retrying, container_id="retrying-cid")

    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[("retrying-cid", _old())])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    mock_runtime.kill.assert_awaited_once_with("retrying-cid")


@pytest.mark.asyncio
async def test_cleanup_zombies_protects_live_service_containers(queries, mock_runtime):
    """CRIT-2: ``list_jobs`` defaults to ``kind='batch'``, so a running service's
    container is protected through ``get_service_container_ids`` instead.
    """
    await queries.create_job(
        Job(
            name="demo",
            kind=JobKind.service,
            service_name="demo",
            gpu_count=0,
            status=JobStatus.running,
            container_id="service-cid",
            config='{"image": "demo:latest", "port": 8000}',
        )
    )

    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[("service-cid", _old())])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    mock_runtime.kill.assert_not_awaited()
    mock_runtime.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_zombies_skips_young_containers(queries, mock_runtime):
    """Defence-in-depth: containers younger than the age threshold are never
    swept, closing the window between ``runtime.run()`` and the status write.
    """
    mock_runtime.list_own_managed_containers = AsyncMock(
        return_value=[("young-orphan-cid", _young(5.0))]
    )

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_zombies_handles_docker_unreachable(queries, mock_runtime, caplog):
    mock_runtime.list_own_managed_containers = AsyncMock(
        side_effect=RuntimeError("docker unreachable")
    )

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    with caplog.at_level(logging.WARNING):
        cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    assert any("managed containers" in record.message.lower() for record in caplog.records)
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_cleanup_zombies_continues_after_kill_failure(queries, mock_runtime):
    """A single container that fails to kill must not abort the whole sweep."""
    mock_runtime.list_own_managed_containers = AsyncMock(
        return_value=[
            ("flaky-cid", _old()),
            ("healthy-cid", _old()),
        ]
    )

    async def flaky_kill(container_id: str) -> None:
        if container_id == "flaky-cid":
            raise ContainerRuntimeError("transient")

    mock_runtime.kill = AsyncMock(side_effect=flaky_kill)

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    assert mock_runtime.kill.await_count == 2
    mock_runtime.remove.assert_awaited_once_with("healthy-cid", force=True)


# ---- (P20) extra_protected: the rowless-run hook ----


@pytest.mark.asyncio
async def test_extra_protected_empty_is_byte_identical(queries, mock_runtime):
    """A hook returning an empty set leaves the pre-P20 sweep untouched."""
    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[("orphan-cid", _old())])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=lambda: set())
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    mock_runtime.kill.assert_awaited_once_with("orphan-cid")
    mock_runtime.remove.assert_awaited_once_with("orphan-cid", force=True)


@pytest.mark.asyncio
async def test_extra_protected_containers_survive(queries, mock_runtime):
    """A one-off run / release container has NO job row and ages past the 30 s
    gate within seconds — the hook is the only thing keeping the sweep off it."""
    mock_runtime.list_own_managed_containers = AsyncMock(
        return_value=[("run-cid", _old()), ("orphan-cid", _old())]
    )

    sweeper = ZombieSweeper(
        queries=queries, runtime=mock_runtime, extra_protected=lambda: {"run-cid"}
    )
    cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 1
    mock_runtime.kill.assert_awaited_once_with("orphan-cid")
    assert "run-cid" not in [c.args[0] for c in mock_runtime.kill.await_args_list]


@pytest.mark.asyncio
async def test_extra_protected_exception_skips_the_whole_tick(queries, mock_runtime, caplog):
    """Fail-closed (security S5): an unresolvable protected set means an unsafe
    sweep, so the tick kills NOTHING — a missed 5-minute sweep costs an orphan
    hanging around, a wrong kill destroys a live migration.

    The hook is consulted first, so the runtime is never even listed.
    """
    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[("orphan-cid", _old())])

    def _boom() -> set[str]:
        raise RuntimeError("controller exploded")

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_boom)
    with caplog.at_level(logging.WARNING):
        cleaned = await sweeper.cleanup_zombies()

    assert cleaned == 0
    mock_runtime.kill.assert_not_awaited()
    mock_runtime.remove.assert_not_awaited()
    mock_runtime.list_own_managed_containers.assert_not_awaited()
    assert any("extra protected" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_extra_protected_result_is_not_mutated(queries, mock_runtime):
    """The sweep copies the hook's set before unioning DB ids into it.

    ``active_run_container_ids`` returns a freshly-built set today, but the
    sweep must not depend on that — mutating a controller-owned collection
    would be a cross-component write.
    """
    await queries.create_job(
        Job(
            name="demo",
            kind=JobKind.service,
            service_name="demo",
            gpu_count=0,
            status=JobStatus.running,
            container_id="service-cid",
            config='{"image": "demo:latest", "port": 8000}',
        )
    )
    mock_runtime.list_own_managed_containers = AsyncMock(return_value=[])
    owned = {"run-cid"}

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=lambda: owned)
    await sweeper.cleanup_zombies()

    assert owned == {"run-cid"}


# ---- (P20 WP6) kill_boot_run_orphans: the boot-side run-orphan reap ----


@pytest.mark.asyncio
async def test_boot_kill_reaps_run_labelled_orphans(queries, mock_runtime, monkeypatch):
    """The registry is empty at boot, so every ``nerdit-run``-labelled container
    is an orphan of the previous process — and there is NO age gate (that is the
    whole point: the periodic sweep's 30 s gate is what leaves the window open)."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1", "o2"])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 2
    assert [c.args[0] for c in mock_runtime.kill.await_args_list] == ["o1", "o2"]
    assert [c.args[0] for c in mock_runtime.remove.await_args_list] == ["o1", "o2"]


@pytest.mark.asyncio
async def test_boot_kill_protects_registry_bound_containers(queries, mock_runtime, monkeypatch):
    """A run that survived into this process (an in-flight release launched by
    the first reconcile tick) is bound in the registry and must not be reaped."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1", "o2"])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=lambda: {"o1"})
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 1
    mock_runtime.kill.assert_awaited_once_with("o2")


@pytest.mark.asyncio
async def test_boot_kill_rechecks_protection_after_settle(queries, mock_runtime, monkeypatch):
    """The protected set is re-read FRESH on every iteration — the release-bind
    race guard: a container listed before its RunSlot bound the container id is
    protected by the time the kill loop reaches it."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1"])
    seen: list[int] = []

    def _binds_late() -> set[str]:
        seen.append(1)
        # First call (the pre-listing guard) sees nothing; the per-iteration
        # re-check, after the settle delay, sees the bound id.
        return set() if len(seen) == 1 else {"o1"}

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_binds_late)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_kill_protects_row_backed_service_containers(queries, mock_runtime, monkeypatch):
    """A LIVE, row-backed service must survive the boot kill even when the
    ``nerdit-run`` listing surfaces it.

    Docker merges a container's labels with its image's, so a service whose
    image bakes ``LABEL nerdit-run=...`` (passthrough Dockerfile, prebuilt image
    on ``POST /services``, template repo) matches the presence filter without
    ever having been a run — and its container id is absent from the run
    registry. Only the row-backed union keeps it alive.
    """
    _no_settle(monkeypatch)
    await queries.create_job(
        Job(
            name="demo",
            kind=JobKind.service,
            service_name="demo",
            gpu_count=0,
            status=JobStatus.running,
            container_id="service-cid",
            config='{"image": "demo:latest", "port": 8000}',
        )
    )
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["service-cid", "run-orphan-cid"])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 1
    mock_runtime.kill.assert_awaited_once_with("run-orphan-cid")
    mock_runtime.remove.assert_awaited_once_with("run-orphan-cid", force=True)


@pytest.mark.asyncio
async def test_boot_kill_protects_scheduled_row_containers(queries, mock_runtime, monkeypatch):
    """The row-backed union carries the status-scan layer too, not only the
    status-free service query: a row still in ``scheduled`` is protected."""
    _no_settle(monkeypatch)
    job = Job(id="sch-boot", script_path="/tmp/train.py", gpu_count=1)
    await queries.create_job(job)
    await queries.update_job_status("sch-boot", JobStatus.scheduled, container_id="fresh-cid")

    mock_runtime.list_own_run_containers = AsyncMock(return_value=["fresh-cid"])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_kill_db_failure_kills_nothing(queries, mock_runtime, monkeypatch, caplog):
    """Fail-closed on the row-backed query as well as on the hook: a partial
    protected set could cost a live service."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1"])
    monkeypatch.setattr(
        queries, "get_service_container_ids", AsyncMock(side_effect=RuntimeError("db gone"))
    )

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    with caplog.at_level(logging.WARNING):
        killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()
    mock_runtime.remove.assert_not_awaited()
    assert any("row-backed" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_boot_kill_hook_exception_kills_nothing(queries, mock_runtime, monkeypatch, caplog):
    """Fail-closed, exactly like the periodic sweep: an unresolvable protected
    set means an unsafe kill, so the pass reaps NOTHING."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1"])

    def _boom() -> set[str]:
        raise RuntimeError("controller exploded")

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_boom)
    with caplog.at_level(logging.WARNING):
        killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()
    mock_runtime.remove.assert_not_awaited()
    mock_runtime.list_own_run_containers.assert_not_awaited()
    assert any("extra protected" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_boot_kill_docker_unreachable_returns_zero(queries, mock_runtime, monkeypatch):
    """An unreachable runtime degrades to a no-op; the periodic sweep retries."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(side_effect=ContainerRuntimeError("no docker"))

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_kill_continues_after_kill_failure(queries, mock_runtime, monkeypatch):
    """One unkillable orphan must not strand the rest of the reap — and must
    still be removed.

    The commonest cause of a failed kill here is "container not running": the
    listing is running-only, but the settle delay gives a short-lived orphan
    time to exit on its own. That is the outcome the pass wanted, so it is not
    counted as a kill — but every listing this class has is running-only, so if
    the remove were skipped nothing would ever collect the exited container.
    """
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1", "o2"])
    mock_runtime.kill = AsyncMock(side_effect=[ContainerRuntimeError("not running"), None])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 1
    assert [c.args[0] for c in mock_runtime.remove.await_args_list] == ["o1", "o2"]


@pytest.mark.asyncio
async def test_boot_kill_stands_down_while_a_run_is_unbound(queries, mock_runtime, monkeypatch):
    """A claimed-but-unbound slot aborts the whole pass.

    A ``[deploy].release`` from the first reconcile tick is docker-visible from
    the container CREATE but binds its slot only when ``runtime.run()`` returns.
    While that gap is open the listing cannot be trusted, so the pass stands
    down and leaves the orphans to the periodic sweep.
    """
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1"])

    sweeper = ZombieSweeper(
        queries=queries,
        runtime=mock_runtime,
        extra_protected=_no_extra,
        unbound_run_probe=lambda: True,
    )
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_kill_probe_failure_kills_nothing(queries, mock_runtime, monkeypatch):
    """An unreadable probe is an unknown answer, and the safe unknown is 'do
    not kill' — same fail-closed posture as the hook and the row query."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1"])

    def _boom_probe() -> bool:
        raise RuntimeError("registry unreadable")

    sweeper = ZombieSweeper(
        queries=queries,
        runtime=mock_runtime,
        extra_protected=_no_extra,
        unbound_run_probe=_boom_probe,
    )
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 0
    mock_runtime.kill.assert_not_awaited()


@pytest.mark.asyncio
async def test_boot_kill_swallows_not_found(queries, mock_runtime, monkeypatch):
    """An orphan that vanished between the listing and the kill is not an error
    (and is not removed either — there is nothing left to remove)."""
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1", "o2"])
    mock_runtime.kill = AsyncMock(side_effect=[ContainerNotFoundError("gone"), None])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 1
    mock_runtime.remove.assert_awaited_once_with("o2", force=True)


@pytest.mark.asyncio
async def test_boot_kill_counts_killed_when_remove_fails(queries, mock_runtime, monkeypatch):
    """A failed remove must not erase a real kill from the count.

    The count feeds the "reaped %d" boot log line, whose claim is that those
    containers have stopped writing — which the kill alone establishes. The
    remove is best-effort tidying of an already-inert exited container.
    """
    _no_settle(monkeypatch)
    mock_runtime.list_own_run_containers = AsyncMock(return_value=["o1", "o2"])
    mock_runtime.remove = AsyncMock(side_effect=[ContainerRuntimeError("in use"), None])

    sweeper = ZombieSweeper(queries=queries, runtime=mock_runtime, extra_protected=_no_extra)
    killed = await sweeper.kill_boot_run_orphans()

    assert killed == 2
    assert [c.args[0] for c in mock_runtime.kill.await_args_list] == ["o1", "o2"]


# ---- (P20 WP6) the loop wiring ----


class _RecordingSweeper:
    """Sweeper double recording the ORDER of the loop's two calls."""

    def __init__(self, boot_exc: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._boot_exc = boot_exc
        self.swept = asyncio.Event()

    async def kill_boot_run_orphans(self) -> int:
        self.calls.append("kill_boot_run_orphans")
        if self._boot_exc is not None:
            raise self._boot_exc
        return 0

    async def cleanup_zombies(self) -> int:
        self.calls.append("cleanup_zombies")
        self.swept.set()
        return 0


async def _drive_loop(double: _RecordingSweeper) -> None:
    """Run ``_zombie_sweep_loop`` until its first sweep, then cancel it."""
    task = asyncio.create_task(_zombie_sweep_loop(double, 3600))  # type: ignore[arg-type]
    try:
        await asyncio.wait_for(double.swept.wait(), timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_zombie_sweep_loop_runs_boot_kill_once_before_first_sweep():
    """The boot kill is single-shot and lands BEFORE the first periodic pass —
    the 5-minute post-crash window only closes if it runs first."""
    double = _RecordingSweeper()

    await _drive_loop(double)

    assert double.calls == ["kill_boot_run_orphans", "cleanup_zombies"]
    assert double.calls.count("kill_boot_run_orphans") == 1


@pytest.mark.asyncio
async def test_zombie_sweep_loop_survives_a_failing_boot_kill():
    """A failing boot kill must never keep the periodic sweep from starting —
    it is the backstop for whatever the boot pass missed."""
    double = _RecordingSweeper(boot_exc=RuntimeError("docker exploded"))

    await _drive_loop(double)

    assert double.calls == ["kill_boot_run_orphans", "cleanup_zombies"]
