"""Test release commands as a pre-swap gate with real controllers and database.

Reuse FakeRuntime from test_services_reconcile. While release blocks, real
reconcile ticks must preserve the old container despite the tagged candidate
image. A persisted release_pending marker matching build_version without a
registered task fails closed after a crash and never swaps.

Pin success, failure, timeout, supersession and not-ready outcomes in rows, logs
and audit records. Scrub release secrets from job_logs, which all authenticated
principals can read.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nerdit.config.settings import ContainerSettings, ServicesSettings
from nerdit.core.launch import _SCRUB_MIN_VALUE_LEN
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.core.services import ServiceController
from nerdit.db.models import ErrorClass, JobStatus
from tests.test_services_reconcile import (
    FakeRuntime,
    FakeSecrets,
    _deploy_svc,
    _drain_builds,
)

pytestmark = pytest.mark.asyncio

RELEASE_CMD = "python manage.py migrate && echo ok"


# --- fakes -------------------------------------------------------------------


class ReleaseRuntime(FakeRuntime):
    """A build-capable runtime whose release container is fully controllable.

    ``build_image`` tags the image (a successful build), then the release
    container goes through the ordinary ``run``/``wait``/``logs``/``remove``
    path of the base fake. Two extras the base does not have:

    * ``remove_image`` — recorded, so the failure path's best-effort candidate
      tag reclaim is observable (and can be made to raise);
    * ``gate`` / ``build_gate`` — optional :class:`asyncio.Event`s the release's
      ``wait`` (resp. the image build) blocks on, which is what makes the
      swap-gate and supersession tests real races rather than assertions about
      a mock.
    """

    def __init__(
        self, *, gate: asyncio.Event | None = None, build_gate: asyncio.Event | None = None
    ) -> None:
        super().__init__()
        self.gate = gate
        self.build_gate = build_gate
        self.removed_images: list[str] = []
        self.remove_image_error: BaseException | None = None
        self.build_calls: list[tuple[str, str]] = []

    async def build_image(self, context_dir, image, dockerfile=None):
        self.calls.append("build_image")
        self.build_calls.append((str(context_dir), image))
        if self.build_gate is not None:
            await self.build_gate.wait()
        self.missing_images.discard(image)
        yield f"Built {image}"

    async def wait(self, container_id: str, timeout_s: float | None = None) -> int:
        if self.gate is not None:
            await self.gate.wait()
        return await super().wait(container_id, timeout_s=timeout_s)

    async def remove_image(self, tag: str, force: bool = False) -> None:
        self.removed_images.append(tag)
        if self.remove_image_error is not None:
            raise self.remove_image_error
        # Really untag it. A recording-only fake hides the whole hazard the
        # reclaim guards exist for: a removed tag makes ``_needs_build`` true
        # again on any row that kept its build markers.
        self.missing_images.add(tag)


def _controller(queries, runtime, *, tmp_path=None, secrets=None, **kw) -> ServiceController:
    """A real controller with the container/retention seams the release needs."""
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499", **kw),
        container_settings=ContainerSettings(),
        secrets=secrets,
        data_dir=str(tmp_path) if tmp_path is not None else None,
    )


def _release_svc(name: str = "app", *, version: int = 2, ctx=None, **cfg_extra):
    """A redeploy row over a still-live old container, with a release armed.

    ``release_pending`` is stamped at ``version`` exactly as the builder arms
    it immediately before starting the release container, so every crash/revert
    assertion here runs against the shape a real interrupted release leaves.
    (It is NOT armed at deploy time — see
    ``test_a_declared_release_that_never_started_is_not_a_crash``.)
    """
    cfg: dict = {
        "release": RELEASE_CMD,
        "release_pending": version,
        "previous_image": f"nerdit-app/{name}:{version - 1}",
    }
    cfg.update(cfg_extra)
    return _deploy_svc(
        name,
        version=version,
        action="redeploy",
        ctx=ctx,
        status=JobStatus.restarting,
        container_id="c-old",
        **cfg,
    )


async def _audit_rows(queries, action: str) -> list[dict]:
    """Every audit row for ``action`` (``params_redacted`` is parsed by the query)."""
    items, _ = await queries.list_audit_log(action=action, limit=200)
    return [item.params_redacted or {} for item in items]


async def _log_lines(queries, job_id: str) -> list[str]:
    return [entry.message for entry in await queries.get_logs(job_id)]


def _cfg(job) -> dict:
    return json.loads(job.config)


# =============================================================================
# 1. The swap gate — the hole the whole design exists to close
# =============================================================================


async def test_release_in_flight_holds_the_swap_across_real_reconcile_ticks(queries, tmp_path):
    """Reality R1: the candidate image is TAGGED before the release runs.

    Every tick between ``build_image`` returning and the release finishing
    would otherwise see ``_needs_build`` False, fall straight through to the
    launch/swap path, destroy the still-serving old container and start the new
    image on top of a half-applied migration. Driven with real
    ``controller.reconcile()`` calls against a release container blocked inside
    ``wait`` — not by asserting on a mock.
    """
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))

    # Tick 1 spawns the build task; let it reach the blocked release.
    await controller.reconcile()
    await _spin_until(lambda: runtime.run_configs, "the release container to start")

    # The candidate image is now present — the pre-P20 fall-through condition.
    assert await runtime.image_exists("nerdit-app/app:2")
    release_cid = "c1"
    assert release_cid in runtime.live

    # Ten real ticks while the migration runs. Not one may touch the old
    # container or launch a second app container.
    for _ in range(10):
        await controller.reconcile()

    assert "c-old" in runtime.live, "the old container was destroyed mid-release"
    assert len(runtime.run_configs) == 1, "a second container was launched mid-release"
    assert runtime.run_configs[0].command == ["/bin/sh", "-c", RELEASE_CMD]
    row = await queries.get_service_by_name("app")
    assert row.container_id == "c-old"  # never repointed
    assert _cfg(row)["image"] == "nerdit-app/app:2"  # generation still in flight

    # Release the migration → the task settles and the gate opens.
    gate.set()
    await _drain_builds(controller)
    assert controller._builder._task_in_flight(row.id) is False

    # ...and only NOW does the swap happen.
    await controller.reconcile()
    assert "c-old" not in runtime.live
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.running
    assert row.container_id != "c-old"
    await controller.shutdown()


async def test_ensure_built_gate_is_done_aware_not_bare_membership(queries, tmp_path):
    """A task cancelled before its first step never runs ``spawn``'s ``finally``,
    so the entry is left behind DONE. Bare ``job.id in _build_tasks`` would then
    make ``ensure_built`` return True forever and the row could never converge
    again — an unrecoverable wedge. The gate is ``not task.done()``."""
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app"))
    row = await queries.get_service_by_name("app")

    async def _noop() -> None:
        return None

    task = asyncio.create_task(_noop())
    await task
    controller._build_tasks[row.id] = task  # done, never popped

    assert controller._builder._task_in_flight(row.id) is False
    assert await controller._builder.ensure_built(row, live=False) is False
    await controller.shutdown()


async def test_release_holds_the_swap_even_for_a_fresh_deploy(queries, tmp_path):
    """No live container to protect, but the swap must still wait: launching the
    new image before the migration finishes is the same torn state."""
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc("app", version=1, ctx=ctx, release=RELEASE_CMD, release_pending=1)
    )

    await controller.reconcile()
    await _spin_until(lambda: runtime.run_configs, "the release container to start")
    for _ in range(5):
        await controller.reconcile()
    assert len(runtime.run_configs) == 1  # only the release

    gate.set()
    await _drain_builds(controller)
    await controller.reconcile()
    assert len(runtime.run_configs) == 2  # release, then the app
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.running
    await controller.shutdown()


# =============================================================================
# 2. The persisted crash marker (D-P20-4)
# =============================================================================


async def test_crashed_release_settles_failed_and_never_swaps(queries, tmp_path):
    """Image built, no task registered, ``release_pending == build_version``:
    the daemon died mid-migration. Its state is unknowable, so the platform
    refuses to decide — settle ``release_failed`` and keep the OLD image in
    front of the data. Never a silent swap, never an automatic re-run."""
    runtime = ReleaseRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc())  # image present, no build context

    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    cfg = _cfg(row)
    # Reverted to the previous image, still serving on the OLD container.
    assert row.status is JobStatus.running
    assert cfg["image"] == "nerdit-app/app:1"
    assert cfg["build_version"] == 1
    assert "release_pending" not in cfg
    assert "c-old" in runtime.live
    assert runtime.run_configs == []  # nothing launched, nothing re-run
    # The release command itself survives: it is app config, not generation state.
    assert cfg["release"] == RELEASE_CMD

    ld = cfg["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.unknown.value
    assert "daemon restarted during release" in ld["error_message"]

    rows = await _audit_rows(queries, "service.release_failed")
    assert rows == [
        {
            "service": "app",
            "version": 2,
            "exit_code": None,
            "timed_out": False,
            "data_rollback": False,
        }
    ]
    await controller.shutdown()


async def test_a_declared_release_that_never_started_is_not_a_crash(queries, tmp_path):
    """A release that was configured but never RAN must not settle as a crash.

    Regression from the P20 live run. The marker used to be armed at deploy
    time, so it meant "a release is configured" — indistinguishable from "a
    release started and the daemon died". Any deploy that skipped the build
    (its target tag already existed, e.g. redeploying a name whose images
    survived the default ``?purge=secrets`` delete) therefore settled
    instantly as a phantom "daemon restarted during release" and the service
    never started at all.

    The row here is exactly that shape: ``release`` declared, image present,
    no build task, and — the fix — no ``release_pending``. It must converge.
    """
    runtime = ReleaseRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    # _release_svc minus the marker: the builder never armed it because the
    # build (and therefore the release) never ran.
    await queries.create_job(_release_svc(release_pending=None))

    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    cfg = _cfg(row)
    assert row.status is not JobStatus.failed
    assert cfg.get("last_deploy", {}).get("reason") != "release_failed"
    assert await _audit_rows(queries, "service.release_failed") == []


async def test_a_deploy_over_a_surviving_tag_still_builds_and_runs_its_release(queries, tmp_path):
    """A present tag is not evidence that THIS generation was built.

    ``nerdit services rm app`` keeps the images under the default
    ``?purge=secrets``, and a fresh deploy always mints version 1 — so the
    target tag is already on the host. Reusing it launched the PREVIOUS code
    and, since the release only ever runs from inside a build task, silently
    skipped the migration the new generation declared. The phase object is the
    seam: ``queued`` means no build task ever reached this generation.
    """
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    # `missing_images` deliberately NOT populated: nerdit-app/app:1 survives.
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx, release=RELEASE_CMD))

    await controller.reconcile()
    await _drain_builds(controller)

    assert runtime.build_calls == [(str(ctx), "nerdit-app/app:1")]
    assert [c.command for c in runtime.run_configs] == [["/bin/sh", "-c", RELEASE_CMD]]
    assert len(await _audit_rows(queries, "service.release")) == 1

    await controller.reconcile()
    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    await controller.shutdown()


async def test_a_rollback_never_rebuilds_and_never_runs_a_release(queries, tmp_path):
    """The counterpart guard. A rollback is stamped ``queued`` too, but it
    deliberately re-points at an already-built tag — rebuilding it would also
    re-execute a migration, and a rollback runs none (§0.27)."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc(
            "app",
            version=1,
            action="rollback",
            ctx=ctx,
            status=JobStatus.restarting,
            release=RELEASE_CMD,
        )
    )

    await controller.reconcile()
    await _drain_builds(controller)

    assert runtime.build_calls == []
    assert [c for c in runtime.run_configs if c.command == ["/bin/sh", "-c", RELEASE_CMD]] == []
    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    await controller.shutdown()


async def test_crashed_release_on_a_fresh_deploy_settles_terminally(queries, tmp_path):
    """No live container to fall back on ⇒ the terminal branch: the row fails,
    and the audit action is the fixed ``service.failed`` (the action varies by
    BRANCH, not by reason) with ``reason=release_failed`` in ``last_deploy``."""
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, release=RELEASE_CMD, release_pending=1))

    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed
    assert row.error_class is ErrorClass.unknown
    cfg = _cfg(row)
    assert cfg["last_deploy"]["reason"] == "release_failed"
    assert await _audit_rows(queries, "service.release_failed") == []
    assert len(await _audit_rows(queries, "service.failed")) == 1
    await controller.shutdown()


async def test_a_restart_of_a_crashed_release_re_settles_rather_than_launching(queries, tmp_path):
    """A restart is not evidence the migration finished. Only a redeploy (which
    mints a new ``build_version`` and re-arms the marker) retries a release."""
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, release=RELEASE_CMD, release_pending=1))
    await controller.reconcile()
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed

    # A `nerdit restart`: desired stays running, status flips to restarting.
    await queries.update_job_status(row.id, JobStatus.restarting)
    await queries.set_desired_state(row.id, "running")
    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed  # re-settled, NOT launched
    assert runtime.run_configs == []
    await controller.shutdown()


async def test_a_restart_after_a_terminal_release_failure_never_launches_the_candidate(
    queries, tmp_path
):
    """The terminal branch deliberately does NOT pop ``release_pending``.

    Pins the asymmetry with the two revert blobs, which do pop it: those move
    the row OFF the failed generation (``image`` back to ``previous_image``, a
    live container still serving), so the marker has nothing left to protect.
    The terminal branch leaves the row ON the candidate image with a migration
    that failed part-way, and the marker is the only thing that makes layer 3
    re-settle a ``nerdit restart`` instead of launching it. Same rule as the
    crashed-release case above, reached here through an OBSERVED failure whose
    candidate tag survived the best-effort reclaim ("image is in use").
    """
    runtime = ReleaseRuntime()
    runtime.exit_code = 1
    runtime.remove_image_error = ContainerRuntimeError("image is in use")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx, release=RELEASE_CMD))

    await controller.reconcile()
    await _drain_builds(controller)
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed
    assert _cfg(row)["release_pending"] == 1
    assert await runtime.image_exists("nerdit-app/app:1")  # the reclaim lost

    # `nerdit restart`: the candidate is on disk and needs no build, so layer 3
    # is the ONLY thing standing between the user and a half-migrated launch.
    await queries.update_job_status(row.id, JobStatus.restarting)
    await queries.set_desired_state(row.id, "running")
    await controller.reconcile()

    after = await queries.get_service_by_name("app")
    assert after.status is JobStatus.failed  # re-settled, NOT launched
    assert len(runtime.run_configs) == 1  # only the release container, ever
    await controller.shutdown()


async def test_marker_for_another_generation_does_not_settle(queries, tmp_path):
    """The marker names a GENERATION. A stale one left by an older version must
    not settle the generation that owns the row today."""
    runtime = ReleaseRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(release_pending=1))  # row is build_version 2

    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.running
    assert _cfg(row)["image"] == "nerdit-app/app:2"  # converged, not reverted
    assert _cfg(row)["last_deploy"]["reason"] is None
    assert await _audit_rows(queries, "service.release_failed") == []
    await controller.shutdown()


async def test_no_marker_and_no_build_version_is_not_a_crashed_release(queries, tmp_path):
    """``pending is not None`` is load-bearing: both keys are absent on the vast
    majority of rows, and a bare ``==`` would settle every one of them."""
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    plain = _deploy_svc("app")
    cfg = _cfg(plain)
    cfg.pop("build_version")
    plain.config = json.dumps(cfg)
    await queries.create_job(plain)

    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.running  # launched normally
    assert await _audit_rows(queries, "service.release_failed") == []
    await controller.shutdown()


# =============================================================================
# 3. Release success
# =============================================================================


async def test_release_success_clears_the_marker_prunes_and_lets_the_swap_through(
    queries, tmp_path
):
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["applying 0001_initial", "OK"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    pruned: list[str] = []

    async def _prune(job) -> None:
        pruned.append(job.id)

    controller._prune_old_images = _prune  # type: ignore[assignment]
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    fresh = await queries.get_service_by_name("app")
    cfg = _cfg(fresh)
    assert "release_pending" not in cfg  # marker cleared
    assert cfg["image"] == "nerdit-app/app:2"  # NOT reverted
    assert cfg["last_deploy"]["phase"] == "building"  # still converging
    assert pruned == [row.id], "the keep-last-3 prune must be reached on success"

    lines = await _log_lines(queries, row.id)
    assert "[release] running (timeout 300s)" in lines
    assert "[release] succeeded (exit 0)" in lines
    assert "[release] applying 0001_initial" in lines
    assert "[release] OK" in lines

    audits = await _audit_rows(queries, "service.release")
    assert audits == [{"service": "app", "version": 2, "exit_code": 0, "timed_out": False}], (
        "exactly one success row, and data_rollback is omitted on success"
    )

    # The gate is open: the next tick performs the swap.
    await controller.reconcile()
    assert "c-old" not in runtime.live
    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    await controller.shutdown()


async def test_release_success_registers_a_cap_exempt_run_slot(queries, tmp_path):
    """The release rides the rowless-run registry so a DELETE is refused and the
    zombie sweep spares the migration container — and it must be exempt from
    the D-P20-2 caps (a deploy never fails because unrelated runs are busy)."""
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path, max_concurrent_runs=1)
    controller._register_run("someone-else", "r1", is_release=False)  # cap full
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _spin_until(lambda: runtime.run_configs, "the release container to start")

    assert controller.has_active_run(row.id) is True
    assert controller.active_run_container_ids() == {"c1"}

    gate.set()
    await _drain_builds(controller)
    assert controller.has_active_run(row.id) is False
    await controller.shutdown()


async def test_no_release_declared_runs_nothing_and_logs_nothing(queries, tmp_path):
    """The whole feature is inert for a deploy that declares no release."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    assert [c for c in runtime.run_configs if c.command == ["/bin/sh", "-c", RELEASE_CMD]] == []
    assert [line for line in await _log_lines(queries, row.id) if "[release]" in line] == []
    assert await _audit_rows(queries, "service.release") == []
    await controller.shutdown()


async def test_blank_release_string_is_treated_as_no_release(queries, tmp_path):
    """A forged/hand-written blob can carry a whitespace-only ``release`` that
    :class:`DeployConfig` would have rejected; the builder must not shell out
    to it."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc("app", version=1, ctx=ctx, release="   ", release_pending=1)
    )
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    assert runtime.run_configs == []
    assert [line for line in await _log_lines(queries, row.id) if "[release]" in line] == []
    await controller.shutdown()


# =============================================================================
# 4. Release failure
# =============================================================================


async def test_release_failure_reverts_the_image_and_keeps_the_old_container(queries, tmp_path):
    """exit 1 ⇒ the generation is settled failed, the row goes back to the
    previous image, the still-serving old container is NEVER destroyed, the
    candidate tag is reclaimed, and there is EXACTLY ONE audit row."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 1
    runtime.log_lines = ["django.db.utils.ProgrammingError: relation does not exist"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    pruned: list[str] = []
    controller._prune_old_images = lambda job: pruned.append(job.id)  # type: ignore[assignment]
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    fresh = await queries.get_service_by_name("app")
    cfg = _cfg(fresh)
    assert fresh.status is JobStatus.running
    assert cfg["image"] == "nerdit-app/app:1"  # reverted
    assert cfg["build_version"] == 1
    # Build markers AND the release marker are popped together.
    assert "release_pending" not in cfg
    assert "build_context_dir" not in cfg
    assert "build_context_root" not in cfg
    assert "dockerfile_name" not in cfg
    ld = cfg["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.user_error.value
    assert "exit 1" in ld["error_message"]
    assert "Data changes are not rolled back" in ld["error_message"]

    # The old container is untouched and nothing new was launched.
    assert "c-old" in runtime.live
    assert [c.image for c in runtime.run_configs] == ["nerdit-app/app:2"]  # only the release
    assert pruned == [], "a failed release must not reach the keep-last-3 prune"
    assert runtime.removed_images == ["nerdit-app/app:2"]  # candidate tag reclaimed

    audits = await _audit_rows(queries, "service.release_failed")
    assert audits == [
        {
            "service": "app",
            "version": 2,
            "exit_code": 1,
            "timed_out": False,
            "data_rollback": False,
        }
    ]
    assert await _audit_rows(queries, "service.release") == []
    assert await _audit_rows(queries, "service.build_failed") == []

    # The failure tail reached job_logs, prefixed, plus the settle's own line.
    lines = await _log_lines(queries, row.id)
    assert "[release] django.db.utils.ProgrammingError: relation does not exist" in lines
    assert "Release failed; keeping previous version." in lines

    # ...and the reverted row converges back onto the old image, not the new one.
    await controller.reconcile()
    after = await queries.get_service_by_name("app")
    assert _cfg(after)["image"] == "nerdit-app/app:1"
    await controller.shutdown()


async def test_release_failure_on_a_fresh_deploy_is_terminal(queries, tmp_path):
    """No live container ⇒ the terminal branch: row ``failed``, and the audit is
    the fixed ``service.failed`` (never ``service.release_failed``) with the
    cause carried in ``last_deploy.reason``."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 3
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc("app", version=1, ctx=ctx, release=RELEASE_CMD, release_pending=1)
    )

    await controller.reconcile()
    await _drain_builds(controller)

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed
    assert row.error_class is ErrorClass.user_error
    assert row.exit_code == -1
    cfg = _cfg(row)
    assert cfg["last_deploy"]["reason"] == "release_failed"
    assert "exit 3" in cfg["last_deploy"]["error_message"]

    assert await _audit_rows(queries, "service.release_failed") == []
    assert len(await _audit_rows(queries, "service.failed")) == 1
    # The candidate tag is deliberately NOT reclaimed here: the terminal branch
    # keeps ``build_context_dir``, so a missing tag would make ``_needs_build``
    # true and layer 2 (checked BEFORE layer 3) would answer the next restart
    # with a REBUILD — re-executing the migration the platform promised never
    # to re-run on its own. See the restart test below.
    assert runtime.removed_images == []
    await controller.shutdown()


async def test_a_restart_after_a_terminal_release_failure_never_rebuilds(queries, tmp_path):
    """The normal terminal path (nothing forced): the tag survives the settle,
    so a ``nerdit restart`` re-settles through layer 3 instead of rebuilding
    the image and running the migration a second time."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 3
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx, release=RELEASE_CMD))

    await controller.reconcile()
    await _drain_builds(controller)
    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.failed
    assert _cfg(row)["release_pending"] == 1
    assert await runtime.image_exists("nerdit-app/app:1")
    assert len(runtime.build_calls) == 1

    await queries.update_job_status(row.id, JobStatus.restarting)
    await queries.set_desired_state(row.id, "running")
    await controller.reconcile()
    await _drain_builds(controller)

    after = await queries.get_service_by_name("app")
    assert after.status is JobStatus.failed  # re-settled, NOT launched
    assert len(runtime.build_calls) == 1, "the image was rebuilt after a terminal release failure"
    assert len(runtime.run_configs) == 1, "the migration was re-executed by the platform"
    await controller.shutdown()


async def test_release_timeout_kills_the_container_and_fails_the_generation(queries, tmp_path):
    """Past ``[services].release_timeout_s`` the bounded wait fires, the runtime
    kills the container, and the outcome is a timeout-classed release failure."""
    runtime = ReleaseRuntime()
    runtime.wait_error = TimeoutError("bounded wait expired")
    runtime.exit_code = 137
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path, release_timeout_s=7)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    # The configured cap really reached the runtime's bounded wait.
    assert runtime.wait_calls == [("c1", 7)]
    assert "kill" in runtime.calls

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["image"] == "nerdit-app/app:1"
    ld = cfg["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.timeout.value
    assert "timed out after 7s" in ld["error_message"]

    assert await _audit_rows(queries, "service.release_failed") == [
        {
            "service": "app",
            "version": 2,
            "exit_code": 137,
            "timed_out": True,
            "data_rollback": False,
        }
    ]
    assert "[release] running (timeout 7s)" in await _log_lines(queries, row.id)
    await controller.shutdown()


async def test_a_release_lost_mid_run_is_not_reported_as_a_start_failure(queries, tmp_path):
    """The runtime loses a container that ALREADY STARTED (dockerd restarted, the
    container was removed under us): ``_execute_container_once`` re-raises
    :class:`RunInterruptedError`, and the release hook must tell those two
    stories apart. "Failed to start" would let an operator whose migration ran
    for two minutes conclude the schema is untouched and safely redeploy; and
    the output the exception carries is the ONLY evidence of how far it got, so
    it has to reach ``job_logs`` rather than being dropped."""
    runtime = ReleaseRuntime()
    runtime.wait_error = ContainerRuntimeError("docker daemon connection reset")
    runtime.log_lines = ["applying 001_add_column", "applying 002_backfill"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    ld = _cfg(await queries.get_service_by_name("app"))["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert "failed to start" not in ld["error_message"]
    assert "lost while running" in ld["error_message"]
    assert "PARTIALLY applied" in ld["error_message"]
    # The migration's own exit is unobserved — neither the app's fault nor a
    # container that never came up.
    assert ld["error_class"] == ErrorClass.unknown.value

    lines = await _log_lines(queries, row.id)
    assert "[release] applying 001_add_column" in lines
    assert "[release] applying 002_backfill" in lines
    await controller.shutdown()


async def test_a_lost_release_tail_is_still_scrubbed(queries, tmp_path):
    """The interrupted path feeds ``job_logs`` like any other, so D-P20-1 holds
    there too — the tail travels through the same scrub choke point."""
    secret = "pg-production-password"
    runtime = ReleaseRuntime()
    runtime.wait_error = ContainerRuntimeError("docker daemon connection reset")
    runtime.log_lines = [f"connecting as admin:{secret}@db"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"PGPASSWORD": secret})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    lines = await _log_lines(queries, row.id)
    assert secret not in "\n".join(lines)
    assert "[release] connecting as admin:***@db" in lines
    await controller.shutdown()


async def test_release_oom_is_classified_oom(queries, tmp_path):
    runtime = ReleaseRuntime()
    runtime.exit_code = 137
    runtime.oom_killed = True
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)

    ld = _cfg(await queries.get_service_by_name("app"))["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.oom.value
    assert "OOM-killed" in ld["error_message"]
    await controller.shutdown()


async def test_release_container_that_cannot_start_is_a_container_failure(queries, tmp_path):
    """A shell-less image / vanished tag / docker hiccup: the settle still runs
    and nothing escapes the hook."""
    runtime = ReleaseRuntime()
    runtime.run_error = ContainerRuntimeError("oci runtime exec failed: no /bin/sh")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    ld = _cfg(await queries.get_service_by_name("app"))["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.container_fail.value
    assert "no /bin/sh" in ld["error_message"]
    # Security S1: the registry slot is freed even though nothing ever started.
    assert controller.has_active_run(row.id) is False
    assert controller._active_runs == {}
    await controller.shutdown()


async def test_candidate_tag_removal_failure_never_breaks_the_settle(queries, tmp_path):
    """Reclaiming the tag is best-effort — an "in use" error must not turn a
    settled failure into an unhandled task exception."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 1
    runtime.remove_image_error = ContainerRuntimeError("image is in use")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)  # must not raise

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["image"] == "nerdit-app/app:1"
    assert cfg["last_deploy"]["reason"] == "release_failed"
    await controller.shutdown()


async def test_the_unmigrated_candidate_is_reclaimed_when_there_is_nothing_to_revert_to(
    queries, tmp_path
):
    """The 'stripped' branch (live old container, NO ``previous_image``) has no
    tag to point ``image`` back at, so the row keeps naming the failed
    candidate while it stays convergeable on the old container. Leaving the tag
    on disk converts a loud failure into a silent one: the moment the old
    container dies, the reconcile loop launches the UNMIGRATED image in front
    of half-migrated data — the exact outcome the pre-swap gate exists to
    prevent. So the tag is dropped and a later launch fails loudly instead. No
    rebuild can follow: this branch pops ``build_context_dir``, and
    ``needs_build`` is False the moment that key is absent."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 1
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx, previous_image=None))

    await controller.reconcile()
    await _drain_builds(controller)

    row = await queries.get_service_by_name("app")
    cfg = _cfg(row)
    assert row.status is JobStatus.running  # still serving on the old container
    assert cfg["image"] == "nerdit-app/app:2"  # no rollback target to revert to
    assert cfg["last_deploy"]["reason"] == "release_failed"
    assert runtime.removed_images == ["nerdit-app/app:2"]
    assert not await runtime.image_exists("nerdit-app/app:2")
    # ...and the row cannot quietly rebuild it either: the build markers went
    # with the settle, so ``needs_build`` stays False.
    assert await controller._builder.needs_build(row) is False
    await controller.shutdown()


async def test_candidate_tag_is_kept_when_a_newer_generation_made_it_the_rollback_target(
    queries, tmp_path
):
    """The reclaim must not delete the NEXT generation's rollback target.

    A redeploy landing mid-release writes ``previous_image = prev_cfg['image']``
    — over a still-building generation that is our never-launched candidate
    tag. The settle then takes the superseded branch and writes nothing, so an
    unguarded ``remove_image`` would leave gen-3 recording a rollback target
    that no longer exists on disk (``nerdit deploy --rollback`` would then try
    to rebuild from an already-rmtree'd context).
    """
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 1  # the gen-2 release fails
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx1))
    row = await queries.get_service_by_name("app")

    build = asyncio.create_task(
        controller._build_app_image(row, str(ctx1), "nerdit-app/app:2", None, str(ctx1))
    )
    await _spin_until(lambda: runtime.run_configs, "the gen-2 release to start")

    # gen-3 lands mid-release; `_release_svc(version=3)` records
    # previous_image = nerdit-app/app:2 — exactly the tag about to be reclaimed.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen3 = json.loads(_release_svc(version=3, ctx=ctx2).config)
    assert gen3["previous_image"] == "nerdit-app/app:2"
    await queries.update_service_config(
        row.id, json.dumps(gen3), status=JobStatus.restarting, desired_state="running"
    )
    gate.set()
    await build

    assert runtime.removed_images == [], "gen-3's rollback target was reclaimed"
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["previous_image"] == "nerdit-app/app:2"
    assert cfg["build_version"] == 3  # gen-3's blob still intact
    await controller.shutdown()


# =============================================================================
# 5. Supersession, env readiness, shared secrets
# =============================================================================


async def test_superseded_generation_runs_no_release_and_writes_no_revert(queries, tmp_path):
    """A redeploy landed while gen-N built: gen-N's release belongs to a dead
    generation. Running it would execute a migration on behalf of config nobody
    asked for; reverting would clobber gen-N+1's blob."""
    build_gate = asyncio.Event()
    runtime = ReleaseRuntime(build_gate=build_gate)
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx1))
    row = await queries.get_service_by_name("app")

    # Drive the gen-2 build directly, blocked inside build_image...
    build = asyncio.create_task(
        controller._build_app_image(row, str(ctx1), "nerdit-app/app:2", None, str(ctx1))
    )
    await _spin_until(lambda: "build_image" in runtime.calls, "the build to start")

    # ...and land a gen-3 blob on the row while it runs, so the release's own
    # fresh-row re-read sees a generation it does not own.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen3 = json.loads(_release_svc(version=3, ctx=ctx2).config)
    await queries.update_service_config(
        row.id, json.dumps(gen3), status=JobStatus.restarting, desired_state="running"
    )
    build_gate.set()
    await build

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["build_version"] == 3  # gen-3 blob intact
    assert cfg["build_context_dir"] == str(ctx2)  # no revert write landed
    assert cfg["release_pending"] == 3  # gen-3's marker untouched
    assert cfg["last_deploy"]["phase"] == "queued"

    # No release container ran, and nothing was audited as a release outcome.
    assert runtime.run_configs == []
    assert await _audit_rows(queries, "service.release") == []
    assert await _audit_rows(queries, "service.release_failed") == []
    assert "[release] skipped for superseded version 2; a newer deploy owns the row." in (
        await _log_lines(queries, row.id)
    )
    await controller.shutdown()


async def test_clearing_the_marker_never_disarms_a_newer_generations_release(queries, tmp_path):
    """The success path's marker clear is version-GUARDED, not a blind pop.

    A redeploy can land in the window between the release's own CAS re-read and
    its success write. The marker on the row then names the NEW generation, and
    an unguarded pop would silently disarm crash-safety for a release that has
    not run yet — the next daemon death mid-migration would swap in an image
    whose migration state is unknown, exactly what D-P20-4 exists to prevent.
    """
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 0
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx1))
    row = await queries.get_service_by_name("app")

    build = asyncio.create_task(
        controller._build_app_image(row, str(ctx1), "nerdit-app/app:2", None, str(ctx1))
    )
    # Wait until the gen-2 release is running — its CAS has already passed.
    await _spin_until(lambda: runtime.run_configs, "the gen-2 release to start")

    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen3 = json.loads(_release_svc(version=3, ctx=ctx2).config)
    await queries.update_service_config(
        row.id, json.dumps(gen3), status=JobStatus.restarting, desired_state="running"
    )
    gate.set()
    await build

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["build_version"] == 3
    assert cfg["release_pending"] == 3, "gen-3's crash marker was disarmed by gen-2's success"
    await controller.shutdown()


def _stale_read_once(monkeypatch, queries, snapshot) -> None:
    """Make the NEXT ``get_job`` return ``snapshot`` — a pre-redeploy row.

    Every generation-scoped marker write is a read-modify-write, and the read
    and the write are two separate ``@_serialized`` statements: a redeploy
    committing BETWEEN them makes the application-level ``build_version``
    compare pass against a snapshot the row no longer holds. The real window is
    microseconds wide, so faking the read is the only deterministic way to open
    it; the write that follows is the real, unpatched one, which is exactly the
    statement under test.
    """
    real = queries.get_job
    seen: list[str] = []

    async def _get_job(job_id: str):
        seen.append(job_id)
        return snapshot if len(seen) == 1 else await real(job_id)

    monkeypatch.setattr(queries, "get_job", _get_job)


async def test_arm_release_pending_is_a_version_cas_never_a_blind_overwrite(
    queries, tmp_path, monkeypatch
):
    """(PR #96 F4) Arming the marker is a DB CAS, not an app-level re-read.

    With an unguarded ``update_job_config`` the stale gen-1 blob is written
    back WHOLE: gen-2's image, build context and marker are reverted to a dead
    generation, and the row is left pointing at a tag whose context dir the
    finished build already rmtree'd — unrecoverable without a redeploy.
    """
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx1, release=RELEASE_CMD))
    row = await queries.get_service_by_name("app")
    stale = await queries.get_job(row.id)  # the gen-1 snapshot the read will see

    # ...and gen-2 lands on the row before the marker write does.
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen2 = json.loads(_deploy_svc("app", version=2, ctx=ctx2, release=RELEASE_CMD).config)
    await queries.update_service_config(
        row.id, json.dumps(gen2), status=JobStatus.restarting, desired_state="running"
    )

    _stale_read_once(monkeypatch, queries, stale)
    armed = await controller._builder._arm_release_pending(row.id, 1)

    assert armed is False, "a CAS miss must report un-armed so the caller aborts the release"
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["build_version"] == 2, "gen-1's stale blob was written over gen-2"
    assert cfg["image"] == "nerdit-app/app:2"
    assert cfg["build_context_dir"] == str(ctx2)
    assert "release_pending" not in cfg, "a dead generation's marker was armed on gen-2's row"
    await controller.shutdown()


async def test_release_aborts_when_the_marker_cas_misses(queries, tmp_path, monkeypatch):
    """(Codex P1, PR #100) An un-armed marker aborts the release, never runs it.

    The window: the release step's (a) fresh-read CAS passes for gen-2, then a
    redeploy lands gen-3 on the row before ``_arm_release_pending`` re-reads.
    The helper no-ops on the mismatch — correctly — but it used to no-op
    SILENTLY, and ``_release_impl`` proceeded to run gen-2's migration against
    the live volumes with no crash marker armed: an unobservable stale
    migration. The helper now reports whether it armed, and the caller treats
    ``False`` as the same supersession as the (a) early-return.
    """
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    await queries.create_job(_deploy_svc("app", version=2, ctx=ctx1, release=RELEASE_CMD))
    row = await queries.get_service_by_name("app")
    stale = await queries.get_job(row.id)  # the gen-2 snapshot the (a) read will see

    # gen-3 lands on the row before the release step's first read...
    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen3 = json.loads(_deploy_svc("app", version=3, ctx=ctx2, release=RELEASE_CMD).config)
    await queries.update_service_config(
        row.id, json.dumps(gen3), status=JobStatus.restarting, desired_state="running"
    )

    # ...so the (a) CAS passes against the stale snapshot and only the arm
    # step's own re-read can catch the supersession.
    _stale_read_once(monkeypatch, queries, stale)
    assert await controller._builder._release_impl(stale, "nerdit-app/app:2") is False

    assert runtime.run_configs == [], "a superseded generation's release container ran"
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["build_version"] == 3  # gen-3 blob intact, no revert write landed
    assert "release_pending" not in cfg, "gen-2's marker was armed on gen-3's row"
    assert not controller.has_active_run(row.id), "the abort leaked the run slot (wedges DELETE)"
    assert (
        "[release] skipped for superseded version 2; "
        "a newer deploy took the row before the release marker was armed."
    ) in await _log_lines(queries, row.id)
    await controller.shutdown()


async def test_clear_release_pending_is_a_version_cas_never_a_blind_overwrite(
    queries, tmp_path, monkeypatch
):
    """(PR #96 F4) Same window, the disarm side — and the worse outcome.

    Clearing writes the stale blob MINUS the marker, so an unguarded write both
    reverts gen-2's config and silently disarms crash-safety for a release that
    has not run yet: the next daemon death mid-migration swaps in an image whose
    migration state is unknown, exactly what D-P20-4 exists to prevent.
    """
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    await queries.create_job(
        _deploy_svc("app", version=1, ctx=ctx1, release=RELEASE_CMD, release_pending=1)
    )
    row = await queries.get_service_by_name("app")
    stale = await queries.get_job(row.id)

    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen2 = json.loads(
        _deploy_svc("app", version=2, ctx=ctx2, release=RELEASE_CMD, release_pending=2).config
    )
    await queries.update_service_config(
        row.id, json.dumps(gen2), status=JobStatus.restarting, desired_state="running"
    )

    _stale_read_once(monkeypatch, queries, stale)
    await controller._builder._clear_release_pending(row.id, 1)

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["build_version"] == 2, "gen-1's stale blob was written over gen-2"
    assert cfg["image"] == "nerdit-app/app:2"
    assert cfg["release_pending"] == 2, "gen-2's crash marker was disarmed by gen-1's clear"
    await controller.shutdown()


async def test_a_superseded_terminal_settle_never_wedges_a_newer_generation(queries, tmp_path):
    """(PR #96 F5) The terminal branch CAS's on ``build_version`` too.

    The live-container branches re-read + CAS, but the terminal branch wrote
    ``status``/``desired_state`` unconditionally from a tick-start snapshot. A
    redeploy that lands while a fresh deploy's build runs was therefore settled
    ``failed``/``failed`` on ITS behalf — and a row whose status AND
    desired_state are both terminal drops straight out of
    ``get_reconcilable_services``, so the new generation is never built, never
    launched and never converges again: a permanent wedge, no matter how many
    ticks run. A CAS miss is supersession — settle nothing, log, audit, done
    (D-P20-4 compatible: nothing is swapped and no release is re-run).
    """
    runtime = ReleaseRuntime()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    # A FRESH deploy (no container_id) — the snapshot that takes the terminal
    # branch rather than either revert branch.
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx1, release=RELEASE_CMD))
    row = await queries.get_service_by_name("app")
    stale = await queries.get_job(row.id)
    assert stale.container_id is None

    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    gen2 = json.loads(_deploy_svc("app", version=2, ctx=ctx2, release=RELEASE_CMD).config)
    await queries.update_service_config(
        row.id, json.dumps(gen2), status=JobStatus.restarting, desired_state="running"
    )

    settled = await controller._builder._settle_failed_generation(
        stale,
        reason="build_failed",
        message="Build failed: npm install failed",
        error_class=ErrorClass.user_error,
        target_version=1,
    )

    assert settled is False
    after = await queries.get_service_by_name("app")
    assert after.status is JobStatus.restarting, "gen-2 was settled failed on gen-1's behalf"
    assert after.desired_state == "running", "gen-2's desired state was overwritten"
    assert after.error_message is None
    assert _cfg(after)["build_version"] == 2  # gen-2 blob intact
    assert row.id in {j.id for j in await queries.get_reconcilable_services()}, (
        "gen-2 dropped out of the reconcile loop — permanently wedged"
    )
    # Supersession is loud but harmless: the log line + the caller's audit
    # action (never the terminal ``service.failed``).
    assert "Build failed for superseded version 1; a newer deploy owns the row." in (
        await _log_lines(queries, row.id)
    )
    assert len(await _audit_rows(queries, "service.build_failed")) == 1
    assert await _audit_rows(queries, "service.failed") == []
    await controller.shutdown()


async def test_guarded_queries_return_false_on_version_mismatch_and_true_on_match(queries):
    """(PR #96 F4/F5) The two CAS primitives, at the query tier.

    ``rowcount == 1`` is the whole contract — a miss must write NOTHING and
    return ``False`` rather than raise, because every caller treats it as
    supersession. ``settle_failed_guarded`` additionally never touches the
    config blob: the terminal branch deliberately KEEPS ``release_pending`` as
    the unlaunchable-generation flag.
    """
    await queries.create_job(_deploy_svc("app", version=2, release_pending=2))
    row = await queries.get_service_by_name("app")
    marker = {"release_pending": 99}  # a visible marker only a real write can land

    assert await queries.patch_job_config(row.id, marker, expect_build_version=1) is False
    assert _cfg(await queries.get_service_by_name("app"))["release_pending"] == 2
    assert await queries.patch_job_config(row.id, marker, expect_build_version=2) is True
    assert _cfg(await queries.get_service_by_name("app"))["release_pending"] == 99

    assert (
        await queries.settle_failed_guarded(
            row.id,
            expect_build_version=1,
            finished_at=_now(),
            exit_code=-1,
            error_class=ErrorClass.user_error,
            error_message="boom",
        )
        is False
    )
    missed = await queries.get_service_by_name("app")
    assert missed.status is JobStatus.building  # the _deploy_svc default, untouched
    assert missed.desired_state == "running"
    assert missed.error_message is None

    assert (
        await queries.settle_failed_guarded(
            row.id,
            expect_build_version=2,
            finished_at=_now(),
            exit_code=-1,
            error_class=ErrorClass.user_error,
            error_message="boom",
        )
        is True
    )
    hit = await queries.get_service_by_name("app")
    assert hit.status is JobStatus.failed
    assert hit.desired_state == "failed"  # one statement covers BOTH columns
    assert hit.exit_code == -1
    assert hit.error_class is ErrorClass.user_error
    assert hit.error_message == "boom"
    assert hit.finished_at is not None
    assert _cfg(hit)["release_pending"] == 99, "the settle rewrote the config blob"


async def test_settle_failed_guarded_still_settles_a_row_that_carries_no_version(queries):
    """``expect_build_version`` is derived from the IMAGE TAG, not the blob.

    A ``POST /services`` / pre-P13 row has a numeric tag and NO
    ``build_version`` in its config, so a strict CAS would read every one of
    them as superseded and leave a failed build stuck ``building`` forever with
    no forensics. A NULL version means no newer generation exists to protect (a
    redeploy always writes one), so it settles.
    """
    versionless = _deploy_svc("app", version=1)
    cfg = _cfg(versionless)
    cfg.pop("build_version")
    versionless.config = json.dumps(cfg)
    await queries.create_job(versionless)
    row = await queries.get_service_by_name("app")

    assert (
        await queries.settle_failed_guarded(
            row.id,
            expect_build_version=1,
            finished_at=_now(),
            exit_code=-1,
            error_class=ErrorClass.user_error,
            error_message="Build failed: npm ci failed",
        )
        is True
    )
    after = await queries.get_service_by_name("app")
    assert after.status is JobStatus.failed
    assert after.desired_state == "failed"


async def test_launch_env_not_ready_fails_the_release_instead_of_retrying(queries, tmp_path):
    """A deploy is a ONE-SHOT gate: an unresolvable ``[ai.*]`` binding must fail
    the generation (the old container keeps serving) rather than leaving the row
    half-converged in ``restarting`` forever, which is what the launch path's
    retry-next-tick semantics would do."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    # A [ai.*] binding pointing at a model row that does not exist.
    await queries.create_job(
        _release_svc(ctx=ctx, ai={"default": {"provider": "ollama", "model": "llama3.1:8b"}})
    )

    await controller.reconcile()
    await _drain_builds(controller)

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["image"] == "nerdit-app/app:1"  # reverted, not retried
    ld = cfg["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.user_error.value
    assert ld["error_message"].startswith("Release could not start:")
    assert runtime.run_configs == []  # no container ever started
    assert len(await _audit_rows(queries, "service.release_failed")) == 1
    await controller.shutdown()


async def test_invalid_volume_blob_fails_the_release_not_the_task(queries, tmp_path):
    """A forged volumes blob is re-validated at release time (the same
    fail-closed rule the launch path applies), and the raise is funnelled into
    the settle rather than escaping as an unhandled task exception."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx, volumes=["bad name:/x"]))

    await controller.reconcile()
    await _drain_builds(controller)

    ld = _cfg(await queries.get_service_by_name("app"))["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.user_error.value
    assert "invalid volume spec" in ld["error_message"]
    assert runtime.run_configs == []
    await controller.shutdown()


async def test_release_mounts_the_same_named_volumes_as_the_live_container(queries, tmp_path):
    """Deliberate: a migration must reach the REAL data, so the release mounts
    the same host dirs the old container is still writing to."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx, volumes=["data:/data"]))

    await controller.reconcile()
    await _drain_builds(controller)

    config = runtime.run_configs[-1]
    host = (tmp_path / "services" / "app" / "data").resolve()
    assert config.volumes == {str(host): "/data"}
    assert config.ports is None  # portless: never contends for the stable port
    assert config.gpu_ids == []
    await controller.shutdown()


async def test_shared_secret_release_leaves_a_shared_resolved_audit_row(queries, tmp_path):
    """Invariants I2: a release consuming ``${secrets.shared.KEY}`` must not be
    the one launch-shaped resolution that leaves no ``secret.shared_resolved``
    trail. Key NAMES only."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"OPENAI_KEY": "sk-live-shared-value"})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(
        _release_svc(
            ctx=ctx,
            ai={
                "default": {
                    "provider": "api",
                    "model": "gpt-4o-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "${secrets.shared.OPENAI_KEY}",
                }
            },
        )
    )

    await controller.reconcile()
    await _drain_builds(controller)

    rows = await _audit_rows(queries, "secret.shared_resolved")
    assert len(rows) == 1
    assert rows[0]["service_name"] == "app"
    assert rows[0]["keys"] == ["OPENAI_KEY"]
    assert "sk-live-shared-value" not in json.dumps(rows)
    await controller.shutdown()


async def test_release_runs_with_the_new_generations_resolved_env(queries, tmp_path):
    """The whole point of a PRE-swap release: the migration runs with the env the
    new code will see, inside the candidate image."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"DB_PASSWORD": "s3cr3t-value-long"})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(_release_svc(ctx=ctx, env={"FEATURE_FLAG": "on"}))

    await controller.reconcile()
    await _drain_builds(controller)

    config = runtime.run_configs[-1]
    assert config.image == "nerdit-app/app:2"  # the CANDIDATE, not previous_image
    assert config.command == ["/bin/sh", "-c", RELEASE_CMD]
    assert config.env["FEATURE_FLAG"] == "on"
    assert config.env["DB_PASSWORD"] == "s3cr3t-value-long"  # secrets injected
    assert config.env["PORT"] == "8000"
    await controller.shutdown()


# =============================================================================
# 6. The scrub — job_logs is readable by ANY authenticated principal
# =============================================================================


async def test_release_tail_in_job_logs_is_scrubbed(queries, tmp_path):
    """Security S3 / D-P20-1. ``GET /services/{ident}/logs`` is served to every
    authenticated principal, readonly included — a release running with the
    service's FULL production env must not print its secrets into that stream.
    This is a security assertion, not a nicety."""
    secret = "pg-production-password"
    runtime = ReleaseRuntime()
    runtime.exit_code = 1  # a failing migration is exactly when people echo env
    runtime.log_lines = [
        f"connecting as admin:{secret}@db",
        f"FATAL: password authentication failed for {secret}",
    ]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"PGPASSWORD": secret})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    lines = await _log_lines(queries, row.id)
    assert secret not in "\n".join(lines)
    assert "[release] connecting as admin:***@db" in lines
    assert "[release] FATAL: password authentication failed for ***" in lines
    # ...and it never leaked into last_deploy.error_message or the audit either.
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert secret not in json.dumps(cfg)


async def test_a_short_declared_secret_is_scrubbed_however_short_it_is(queries, tmp_path):
    """(PR #96 review F1) A declared secret under ``_SCRUB_MIN_VALUE_LEN``.

    The floor exists to keep the platform's credential-NAME *guess* from
    mangling a log, and it belongs at the tier that knows a guess from a
    declaration. Applied a second time at the consumer it caught declared
    secrets too, and ``POST /secrets/{service}`` enforces no minimum length —
    so a 4-character production key was written verbatim into ``job_logs``,
    readable by every authenticated principal including ``readonly``.
    """
    secret = "ab12"
    assert len(secret) < _SCRUB_MIN_VALUE_LEN
    runtime = ReleaseRuntime()
    runtime.exit_code = 1
    runtime.log_lines = [f"connecting with API_KEY={secret} to upstream"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"API_KEY": secret})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    lines = await _log_lines(queries, row.id)
    assert secret not in "\n".join(lines)
    assert "[release] connecting with API_KEY=*** to upstream" in lines
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert secret not in json.dumps(cfg)
    assert secret not in json.dumps(await _audit_rows(queries, "service.release_failed"))
    await controller.shutdown()


async def test_a_plain_deploy_env_credential_is_scrubbed_too(queries, tmp_path):
    """A user-supplied ``nerdit deploy --env STRIPE_API_KEY=…`` is ordinary
    ``config['env']``, not a ``SecretManager`` secret and not a binding-injected
    key — but the release container gets it exactly the same, and ``job_logs``
    is readable by ANY authenticated principal. Scoping the credential-name
    heuristic to the binding resolvers' own keys left the biggest class of
    user secrets in the clear. Base URLs stay readable: masking them would make
    a migration log useless and they are not secret."""
    api_key = "sk_live_51HxxQuietPlease"
    gh_token = "ghp_0123456789abcdefghij"
    runtime = ReleaseRuntime()
    runtime.exit_code = 1  # a failing migration is exactly when people echo env
    runtime.log_lines = [
        f"POST https://api.stripe.com/v1/charges key={api_key}",
        f"git fetch https://{gh_token}@github.com/acme/migrations",
        "callback https://hooks.example.test/deploy",
    ]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _release_svc(
            ctx=ctx,
            env={
                "STRIPE_API_KEY": api_key,
                "GH_TOKEN": gh_token,
                "CALLBACK_URL": "https://hooks.example.test/deploy",
            },
        )
    )
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    lines = await _log_lines(queries, row.id)
    joined = "\n".join(lines)
    assert api_key not in joined
    assert gh_token not in joined
    assert "[release] POST https://api.stripe.com/v1/charges key=***" in lines
    assert "[release] git fetch https://***@github.com/acme/migrations" in lines
    # Not credential-named ⇒ deliberately left readable.
    assert "[release] callback https://hooks.example.test/deploy" in lines
    # ``config['env']`` is where a plain deploy env legitimately lives (it is
    # replayed on every launch); what must not carry the value is the failure
    # message, which is mirrored into last_deploy and served far wider.
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert api_key not in cfg["last_deploy"]["error_message"]
    assert api_key not in json.dumps(await _audit_rows(queries, "service.release_failed"))
    await controller.shutdown()


async def test_the_credential_name_heuristic_covers_the_common_secret_families():
    """The heuristic is the ONLY thing standing between a plain
    ``nerdit deploy --env`` credential and ``job_logs``. A suffix-only test
    missed every family that puts the giveaway word in the middle of the name
    — which is most of them."""
    from nerdit.core.app_build import _is_credential_key

    for key in (
        "AWS_SECRET_ACCESS_KEY",
        "DJANGO_SECRET_KEY",
        "SECRET_KEY",
        "JWT_SECRET_KEY",
        "PRIVATE_KEY",
        "ENCRYPTION_KEY",
        "SUPABASE_KEY",
        "STRIPE_API_KEY",
        "GH_TOKEN",
        "PGPASSWORD",
        "SERVICE_ACCOUNT_CREDENTIALS",
        "DATABASE_URL",
        "NERDIT_DB_MAIN_URL",
    ):
        assert _is_credential_key(key) is True, key
    # Not credential-named ⇒ left readable, or a migration log becomes noise.
    for key in ("OPENAI_BASE_URL", "CALLBACK_URL", "LOG_LEVEL", "PORT", "KEYCLOAK_URL"):
        assert _is_credential_key(key) is False, key


async def test_the_credential_name_heuristic_is_case_insensitive():
    """(PR #96 F3) Env key casing is user-authored and unenforced. Nothing in
    the deploy path uppercases ``--env api_key=sk_live_…``, and it is exactly as
    dangerous as ``API_KEY`` — a case-sensitive heuristic simply never saw it.
    Uppercasing also widens the URL aliases and ``NERDIT_DB_*_URL`` to their
    lowercase spellings: the fail-safe direction, same as every other
    over-match here."""
    from nerdit.core.app_build import _is_credential_key

    for key in ("api_key", "Stripe_Api_Key", "database_url", "db_passwd", "nerdit_db_main_url"):
        assert _is_credential_key(key) is True, key
    # The suffix carve-outs must survive the normalisation, not just uppercase.
    for key in ("keycloak_url", "keyspace"):
        assert _is_credential_key(key) is False, key


async def test_a_lowercase_keyed_plain_env_credential_is_scrubbed_in_job_logs(queries, tmp_path):
    """(PR #96 F3) The end-to-end half of the casing fix: a lowercase-keyed
    credential must reach ``job_logs`` masked, exactly like its uppercase twin.
    ``job_logs`` is readable by ANY authenticated principal, readonly
    included."""
    api_key = "sk_live_51HxxLowerCasePlease"
    runtime = ReleaseRuntime()
    runtime.exit_code = 1  # a failing migration is exactly when people echo env
    runtime.log_lines = [f"POST https://api.stripe.com/v1/charges key={api_key}"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx, env={"stripe_api_key": api_key}))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    lines = await _log_lines(queries, row.id)
    assert api_key not in "\n".join(lines)
    assert "[release] POST https://api.stripe.com/v1/charges key=***" in lines
    await controller.shutdown()


async def test_a_short_credential_value_does_not_mask_the_whole_tail(queries, tmp_path):
    """Guard on the name heuristic: it matches on the KEY, so a one-character
    ``DB_PASSWORD`` would otherwise turn every ``1`` in a migration log into a
    mask — and the mangling would itself advertise where the value appears. A
    heuristic-matched value below the scrub floor is skipped."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["applied 1 of 3 migrations"]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx, env={"DB_PASSWORD": "1"}))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    assert "[release] applied 1 of 3 migrations" in await _log_lines(queries, row.id)
    await controller.shutdown()


async def test_release_tail_line_is_truncated_in_job_logs(queries, tmp_path):
    """An oversized line is clamped before it reaches ``job_logs`` — one runaway
    migration must not be able to inflate the log table by megabytes per line."""
    from nerdit.core.launch import _RUN_LINE_MAX_BYTES

    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    runtime.log_lines = ["x" * (_RUN_LINE_MAX_BYTES + 5000)]
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    tail = [line for line in await _log_lines(queries, row.id) if line.startswith("[release] x")]
    assert len(tail) == 1
    assert tail[0].endswith("…[truncated]")
    assert len(tail[0].encode()) < _RUN_LINE_MAX_BYTES + 100


async def test_a_secret_bearing_runtime_error_is_scrubbed_before_it_is_stored(queries, tmp_path):
    """The failure MESSAGE goes through the same choke point as the tail: a
    runtime error can echo part of the container spec (env included), and the
    message lands in ``job_logs``, in ``last_deploy.error_message`` and on the
    row — all readable well beyond the owner."""
    secret = "leaky-secret-value-123"
    runtime = ReleaseRuntime()
    runtime.run_error = ContainerRuntimeError(f"failed to create container with env {secret}")
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    secrets = FakeSecrets({"API_TOKEN": secret})
    controller = _controller(queries, runtime, tmp_path=tmp_path, secrets=secrets)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _drain_builds(controller)

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert secret not in cfg["last_deploy"]["error_message"]
    assert "***" in cfg["last_deploy"]["error_message"]
    assert secret not in "\n".join(await _log_lines(queries, row.id))


# =============================================================================
# 7. Nothing escapes the hook
# =============================================================================


def _escaping_release_svc(name: str = "app", *, version: int = 2, ctx=None, **cfg_extra):
    """The PRODUCTION shape of a redeploy with a release: **no** seeded marker.

    ``_release_svc`` hand-seeds ``release_pending`` because it models a row
    whose release already started. Nothing writes that key at deploy time any
    more — only the build task arms it, at the top of the release step — so an
    escape-path test that starts from the seeded shape asserts against a state
    production cannot produce, and silently passes even when the escape leaves
    the generation both unsettled and unmarked.
    """
    cfg: dict = {"release": RELEASE_CMD, "previous_image": f"nerdit-app/{name}:{version - 1}"}
    cfg.update(cfg_extra)
    return _deploy_svc(
        name,
        version=version,
        action="redeploy",
        ctx=ctx,
        status=JobStatus.restarting,
        container_id="c-old",
        **cfg,
    )


async def test_an_unexpected_exception_settles_the_generation_and_never_swaps(queries, tmp_path):
    """The success tail of ``build()`` sits OUTSIDE its ``except
    ContainerRuntimeError``, so an escape would surface as an unhandled task
    exception and wedge the row mid-generation. A non-enumerated failure is
    caught AND settled in place: swallowing it alone would pop the task from
    ``_build_tasks`` with the image already tagged, and the very next tick
    would open all three gates and swap the unmigrated image in."""
    runtime = ReleaseRuntime()
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)

    async def _boom(*args, **kwargs):
        raise RuntimeError("something nobody enumerated")

    controller._audit_shared_resolved = _boom  # type: ignore[assignment]
    secrets = FakeSecrets({"OPENAI_KEY": "sk-shared-value"})
    controller._secrets = secrets  # type: ignore[assignment]
    await queries.create_job(
        _escaping_release_svc(
            ctx=ctx,
            ai={
                "default": {
                    "provider": "api",
                    "model": "gpt-4o-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "${secrets.shared.OPENAI_KEY}",
                }
            },
        )
    )

    await controller.reconcile()
    await _drain_builds(controller)  # must not raise

    row = await queries.get_service_by_name("app")
    cfg = _cfg(row)
    # Settled immediately, with an accurate message — layer 3's "daemon
    # restarted during release" would be a lie for an in-process exception.
    assert cfg["image"] == "nerdit-app/app:1"
    assert "release_pending" not in cfg
    ld = cfg["last_deploy"]
    assert ld["reason"] == "release_failed"
    assert ld["error_class"] == ErrorClass.unknown.value
    assert ld["error_message"].startswith("Release failed unexpectedly;")
    assert "Data changes are not rolled back" in ld["error_message"]
    assert len(await _audit_rows(queries, "service.release_failed")) == 1
    # Slot released regardless (S1).
    assert controller.has_active_run(row.id) is False

    # The old container never stopped serving, and the next tick converges
    # back onto the OLD image rather than swapping the candidate in.
    assert "c-old" in runtime.live
    await controller.reconcile()
    assert "c-old" in runtime.live
    assert _cfg(await queries.get_service_by_name("app"))["image"] == "nerdit-app/app:1"
    await controller.shutdown()


async def test_an_escape_before_the_env_resolves_still_never_swaps(queries, tmp_path):
    """Regression: an escape raised EARLIER than the env resolve.

    The crash marker used to be armed deep inside ``_execute_release``, once
    the env, volumes and container spec were assembled. Anything raising before
    that point — a fresh ``get_job``, ``parse_job_config``, ``_register_run``,
    ``_resolve_launch_env``'s own unexpected errors, ``service_volumes`` — left
    the generation BOTH unsettled and unmarked, and the build task then
    returned: layer 1 saw no task, layer 2 saw the candidate already tagged,
    layer 3 saw no marker, and the UNMIGRATED image was swapped in on the next
    tick. The row here carries no seeded marker (production's shape), so only
    the code under test can arm or settle it.
    """
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)

    async def _boom(*args, **kwargs):
        raise RuntimeError("resolver blew up before the container existed")

    controller._resolve_launch_env = _boom  # type: ignore[assignment]
    await queries.create_job(_escaping_release_svc(ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)  # must not raise

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["image"] == "nerdit-app/app:1", "the unmigrated candidate was left in place"
    assert cfg["last_deploy"]["reason"] == "release_failed"
    assert runtime.run_configs == []  # no release container ever started

    await controller.reconcile()
    assert "c-old" in runtime.live, "the old container was swapped out for the unmigrated image"
    assert _cfg(await queries.get_service_by_name("app"))["image"] == "nerdit-app/app:1"
    await controller.shutdown()


async def test_a_failing_catch_all_settle_falls_back_on_the_layer_3_backstop(queries, tmp_path):
    """The marker is armed the moment we commit to running the release, so
    layer 3 backstops ANY escape — including a catch-all settle that itself
    raises. Without the arming, this row would swap the unmigrated image in on
    the very next tick."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)

    async def _boom(*args, **kwargs):
        raise RuntimeError("nothing on this path works today")

    controller._resolve_launch_env = _boom  # type: ignore[assignment]
    controller._builder._settle_unexpected_release_failure = _boom  # type: ignore[assignment]
    await queries.create_job(_escaping_release_svc(ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)  # must not raise

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["release_pending"] == 2, "the swap gate has no backstop left"
    assert cfg["image"] == "nerdit-app/app:2"  # not settled yet
    assert "c-old" in runtime.live

    # Next tick: layer 3 re-settles from the armed marker, still no swap.
    await controller.reconcile()
    settled = _cfg(await queries.get_service_by_name("app"))
    assert settled["image"] == "nerdit-app/app:1"
    assert settled["last_deploy"]["reason"] == "release_failed"
    assert "daemon restarted during release" in settled["last_deploy"]["error_message"]
    assert "c-old" in runtime.live
    await controller.shutdown()


# =============================================================================
# 8. The shared settle — the audit action varies by BRANCH, not by reason
# =============================================================================
#
# ``_settle_failed_generation`` was extracted out of ``build()`` so the release
# hook could reuse it. Collapsing "which action do we audit?" from BRANCH to
# REASON was the easy bug in that extraction, and nothing pinned it before P20:
# a release failure over a live container must audit ``service.release_failed``,
# while a fresh deploy that never had a container audits the FIXED terminal
# ``service.failed`` whatever killed it.


class FailingBuildRuntime(ReleaseRuntime):
    """``build_image`` always raises — a deterministic build failure."""

    async def build_image(self, context_dir, image, dockerfile=None):
        self.calls.append("build_image")
        if self.build_gate is not None:
            await self.build_gate.wait()
        raise ContainerRuntimeError("npm install failed")
        yield  # pragma: no cover — marks this an async generator


async def test_build_failure_over_a_live_container_audits_build_failed(queries, tmp_path):
    runtime = FailingBuildRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc(
            "app",
            version=2,
            action="redeploy",
            ctx=ctx,
            status=JobStatus.restarting,
            container_id="c-old",
            previous_image="nerdit-app/app:1",
        )
    )

    await controller.reconcile()
    await _drain_builds(controller)

    assert len(await _audit_rows(queries, "service.build_failed")) == 1
    assert await _audit_rows(queries, "service.failed") == []
    assert await _audit_rows(queries, "service.release_failed") == []
    await controller.shutdown()


async def test_fresh_build_failure_audits_the_fixed_terminal_action(queries, tmp_path):
    runtime = FailingBuildRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:1")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_deploy_svc("app", version=1, ctx=ctx))

    await controller.reconcile()
    await _drain_builds(controller)

    assert len(await _audit_rows(queries, "service.failed")) == 1
    assert await _audit_rows(queries, "service.build_failed") == []
    await controller.shutdown()


async def test_build_failure_audit_params_are_the_controller_helper_defaults(queries, tmp_path):
    """``audit_params=None`` (every build caller) still routes through
    ``ServiceController._audit``, so its ``{service_name, restart_count}``
    payload — and its interceptability — survive the extraction byte-for-byte."""
    runtime = FailingBuildRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(
        _deploy_svc(
            "app",
            version=2,
            action="redeploy",
            ctx=ctx,
            status=JobStatus.restarting,
            container_id="c-old",
            previous_image="nerdit-app/app:1",
        )
    )

    await controller.reconcile()
    await _drain_builds(controller)

    params = (await _audit_rows(queries, "service.build_failed"))[0]
    assert set(params) == {"service_name", "restart_count"}
    assert params["service_name"] == "app"
    await controller.shutdown()


# =============================================================================
# 9. KNOWN GAP — a `stop` racing the settle overrides the user's desired state
# =============================================================================


async def test_known_gap_stop_during_a_release_is_overridden_by_the_settle(queries, tmp_path):
    """Document a known bug; rewrite this test when the behavior is fixed.

    Unlike DELETE, stop bypasses the active-release gate and tears down the old
    container. Release failure then takes the terminal settlement branch:
    desired_state changes from stopped to failed, no revert blob is written,
    and the unused candidate image/context/release marker remain armed. Restarts
    repeatedly settle release_failed until redeployment. Plain build failures
    have the same race; long releases make it easy to reach.
    """
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 1  # the release will fail
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    await controller.reconcile()
    await _spin_until(lambda: runtime.run_configs, "the release container to start")

    # `nerdit stop` mid-migration: NOT refused (unlike DELETE), and it destroys
    # the still-serving old container.
    await queries.set_desired_state(row.id, "stopped")
    await controller.reconcile()
    assert "c-old" not in runtime.live
    stopped = await queries.get_service_by_name("app")
    assert stopped.status is JobStatus.stopped

    gate.set()
    await _drain_builds(controller)

    final = await queries.get_service_by_name("app")
    cfg = _cfg(final)
    # --- everything below is the BUG, asserted so a fix trips this test ---
    assert final.status is JobStatus.failed
    assert final.desired_state == "failed", "the user's explicit stop was overridden"
    assert cfg["image"] == "nerdit-app/app:2", "not reverted to previous_image"
    assert cfg["release_pending"] == 2, "crash marker left armed on a non-reverted row"
    assert "build_context_dir" in cfg, "build markers left armed"
    await controller.shutdown()


async def test_service_deleted_during_its_build_skips_the_release(queries, tmp_path):
    """The row is re-read fresh before the release: if it vanished mid-build
    there is no generation to gate and nothing to run."""
    runtime = ReleaseRuntime()
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc(ctx=ctx))
    row = await queries.get_service_by_name("app")

    async def _delete_then_build(context_dir, image, dockerfile=None):
        runtime.calls.append("build_image")
        runtime.missing_images.discard(image)
        await queries.delete_service_checked(row.id, None)
        yield f"Built {image}"

    runtime.build_image = _delete_then_build  # type: ignore[assignment]

    await controller._build_app_image(row, str(ctx), "nerdit-app/app:2", None, str(ctx))

    assert runtime.run_configs == []
    assert await queries.get_service_by_name("app") is None
    await controller.shutdown()


# --- small helpers ------------------------------------------------------------


def _now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


async def _spin_until(predicate, what: str, *, tries: int = 200) -> None:
    """Yield to the loop until ``predicate()`` is truthy (a real off-tick task)."""
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


# =============================================================================
# 8. (WP3) Run-container attribution labels + the crash-orphan reap
# =============================================================================
#
# `_settle_crashed_release` used to leave the orphaned migration container
# running until the zombie sweep collected it — worst case 30 s
# (ZOMBIE_AGE_THRESHOLD_SECONDS) + 300 s (ZOMBIE_SWEEP_INTERVAL_SECONDS) of a
# migration still writing while the settle puts the PREVIOUS image back in
# front of the same data. After a crash the in-memory run registry (and the
# run_id) is gone, so `nerdit-job=<job.id>` is the only recoverable key —
# which is why the labels are stamped at `_execute_container_once`, the one
# choke point a run and a release share.


async def test_release_container_carries_the_run_and_job_labels(queries, tmp_path):
    """Every rowless container is attributable: ``nerdit-run`` + ``nerdit-job``."""
    gate = asyncio.Event()
    runtime = ReleaseRuntime(gate=gate)
    runtime.exit_code = 0
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    runtime.missing_images.add("nerdit-app/app:2")
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    row = _release_svc(ctx=ctx)
    await queries.create_job(row)

    await controller.reconcile()
    await _spin_until(lambda: runtime.run_configs, "the release container to start")
    # Read the slot id off the registry WHILE it is held — it is discarded the
    # moment the release settles.
    (run_id,) = list(controller._active_runs[row.id])

    assert runtime.run_configs[-1].extra_labels == {"nerdit-run": run_id, "nerdit-job": row.id}

    gate.set()
    await _drain_builds(controller)
    await controller.shutdown()


async def test_crashed_release_settle_reaps_the_orphan_by_label(queries, tmp_path):
    """The migration must stop writing BEFORE the previous image goes back in
    front of the data — so the reap runs ahead of the settle, not after it."""
    runtime = ReleaseRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    row = _release_svc()
    await queries.create_job(row)
    runtime.labeled_containers[("nerdit-job", row.id)] = ["orphan1"]

    await controller.reconcile()

    assert runtime.killed == ["orphan1"]
    assert "remove" in runtime.calls
    # ...and the settle itself still lands, unchanged.
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["last_deploy"]["reason"] == "release_failed"
    assert cfg["image"] == "nerdit-app/app:1"
    await controller.shutdown()


async def test_crashed_release_settle_survives_a_reap_failure(queries, tmp_path):
    """The reap is best-effort by construction: it runs inside the reconcile
    tick, and a docker error there must never wedge the row."""

    class _ReapFailingRuntime(ReleaseRuntime):
        async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
            raise RuntimeError("docker went away")

    runtime = _ReapFailingRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(_release_svc())

    await controller.reconcile()  # must not raise

    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["last_deploy"]["reason"] == "release_failed"
    await controller.shutdown()


async def test_crashed_release_settle_skips_the_reap_while_a_run_is_active(queries, tmp_path):
    """Belt over ``ensure_built`` layer 1: that gate keys on the BUILD-task
    registry, while ``_active_runs`` is the authority on live containers. A
    route-initiated run against this row must not have its container killed by
    a settle that is only meant to reap a crash orphan."""
    runtime = ReleaseRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    row = _release_svc()
    await queries.create_job(row)
    runtime.labeled_containers[("nerdit-job", row.id)] = ["live-run"]
    controller._register_run(row.id, "r1", is_release=False)

    await controller.reconcile()

    assert runtime.killed == []
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["last_deploy"]["reason"] == "release_failed"
    controller._discard_run(row.id, "r1")
    await controller.shutdown()


async def test_crashed_release_settle_aborts_the_reap_when_a_run_lands_mid_pass(queries, tmp_path):
    """The ``has_active_run`` belt is re-checked after every await, not once.

    The pre-check and the kills are separated by real awaits (the docker
    listing, then each kill/remove), and a run claiming its slot inside that
    window IS returned by the listing — the labels are stamped for every
    rowless container, runs included. Killing it is exactly the harm the reap
    exists to prevent: a SIGKILL mid-write on the service data volume.
    """
    row = _release_svc()

    class _RacingRuntime(ReleaseRuntime):
        async def list_own_labeled_containers(self, label: str, value: str) -> list[str]:
            # The run route wins the race: it claims and binds while the
            # docker listing is in flight.
            controller._register_run(row.id, "r1", is_release=False)
            controller._bind_run_container(row.id, "r1", "live-run-cid")
            return await super().list_own_labeled_containers(label, value)

    runtime = _RacingRuntime()
    runtime.live["c-old"] = _now()
    controller = _controller(queries, runtime, tmp_path=tmp_path)
    await queries.create_job(row)
    runtime.labeled_containers[("nerdit-job", row.id)] = ["live-run-cid"]

    await controller.reconcile()

    assert runtime.killed == []
    assert controller.has_active_run(row.id) is True
    cfg = _cfg(await queries.get_service_by_name("app"))
    assert cfg["last_deploy"]["reason"] == "release_failed"
    controller._discard_run(row.id, "r1")
    await controller.shutdown()
