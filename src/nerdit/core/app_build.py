"""Build deploy images and run release commands before swapping containers.

`ensure_built` blocks swaps while a build or release is active. The persisted
`release_pending` marker also blocks swaps after a daemon crash; only an
explicit redeploy retries a crashed release. Build and release failures share
one generation-aware settlement path. Controller hooks remain overridable
through `self._c`, and task registries remain on the controller.

A failed release reverts the image, never data. Migrations use the same named
volumes as the live container and must tolerate concurrent access, partial
application, and the previous image's schema expectations. Audit params report
`data_rollback: false`.

Releases consume neither run quota nor run/build concurrency slots: the build
semaphore is released first. Their results go to `last_deploy.reason`,
`job_logs`, and audit records, never `last_run`. Rollback bypasses the builder
and never runs a release command.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nerdit.core.deploy_state import record_deploy_settled, stamp_last_deploy
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.launch import (
    _SCRUB_MIN_VALUE_LEN,
    build_run_container_config,
    finalize_container_config,
    scrub_secret_values,
)
from nerdit.core.runtime.protocol import (
    BuildPlatformError,
    ContainerNotFoundError,
    ContainerRuntimeError,
)
from nerdit.core.volumes import VolumeSpecError, service_volumes
from nerdit.db.enums import ErrorClass, JobKind, JobStatus, LogStream
from nerdit.db.rows import Job
from nerdit.utils.ids import generate_id

if TYPE_CHECKING:
    from nerdit.core.services import ResolvedLaunchEnv, RunResult, ServiceController

logger = logging.getLogger(__name__)

# Lines of release stdout/stderr captured into `job_logs`. The tail is
# bounded again by the runtime's byte budget and per-line clamp at read time
# (`core/launch.py`), so this only caps the line COUNT.
_RELEASE_LOG_TAIL = 200

# Env keys whose VALUES are credential-class and must be scrubbed out of a
# captured release tail. Base URLs (`OPENAI_BASE_URL`) are
# deliberately not scrubbed — they are not secret and masking them would make a
# migration log unreadable.
_SCRUB_URL_KEYS = frozenset({"DATABASE_URL", "REDIS_URL"})

# Name fragments that make a value credential-class. Matched as
# SUBSTRINGS, not suffixes: the biggest real-world families put the giveaway
# word in the middle (`AWS_SECRET_ACCESS_KEY`, `DJANGO_SECRET_KEY`,
# `JWT_SECRET_KEY`), and a suffix-only test let every one of them through.
# Over-matching (`TOKEN_BUDGET`, `SECRET_ROTATION_DAYS`) costs readability
# in a migration log, but those keys carry short numeric values that the
# `_SCRUB_MIN_VALUE_LEN` floor skips anyway — whereas an unmasked AWS secret
# in `job_logs` is readable by every authenticated principal.
_CREDENTIAL_KEY_PARTS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
# Trailing fragments that only mean "credential" at the END of a name: `KEY`
# catches `API_KEY`/`SECRET_KEY`/`SUPABASE_KEY`/`PRIVATE_KEY`, while a
# substring test on it would also swallow `KEYCLOAK_URL` and `KEYSPACE`.
_CREDENTIAL_KEY_SUFFIXES = ("KEY", "PWD")


def _is_credential_key(key: str) -> bool:
    """Identify credential-bearing env keys case-insensitively.

    Cover credential name fragments, managed-DB aliases, and NERDIT_DB_*_URL DSNs
    across the entire env, including plain deploy overrides. Prefer over-masking
    to leaking; the caller applies a minimum length only to heuristic matches.
    """
    key = key.upper()
    return (
        any(part in key for part in _CREDENTIAL_KEY_PARTS)
        or key.endswith(_CREDENTIAL_KEY_SUFFIXES)
        or key in _SCRUB_URL_KEYS
        or (key.startswith("NERDIT_DB_") and key.endswith("_URL"))
    )


def _build_never_ran(cfg: dict) -> bool:
    """Detect queued deploy generations even if an old image has the same tag.

    Delete/recreate can reuse version 1 while its old image remains. A queued
    phase requires a fresh build and release; build advances it when a slot is
    held. Exclude rollback, which intentionally uses an existing tag without a
    release. Rows without phase metadata retain tag-presence behavior.
    """
    ld = cfg.get("last_deploy")
    if not isinstance(ld, dict):
        return False
    return (
        ld.get("phase") == "queued"
        and ld.get("action") != "rollback"
        and ld.get("version") == cfg.get("build_version")
    )


def _version_of_tag(image: str) -> int | None:
    """Deploy generation encoded in an image tag (`repo:3` → `3`), else None.

    A non-numeric tag (a pre-P13 or hand-written row) yields `None`, which
    every consumer reads as "nothing to CAS on".
    """
    suffix = str(image).rsplit(":", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _scrub_text(text: str, values: set[str]) -> str:
    """Mask sensitive values in a single failure message.

    A release failure message can quote a runtime/binding error that echoed
    part of the container spec, and it lands in `job_logs`, in
    `last_deploy.error_message` and on the row — all readable well beyond the
    owner. Same choke point as the log tail, one line at a time.
    """
    return scrub_secret_values([text], values)[0]


def _release_audit_params(
    job: Job,
    version: int | None,
    *,
    exit_code: int | None,
    timed_out: bool,
    data_rollback: bool | None = None,
) -> dict[str, object]:
    """Audit params for a `service.release` / `service.release_failed` row (§1.3).

    Outcome facts only — never a log line, never an env key or value, never the
    release command itself (it is app-authored text that would then live
    forever in the archive-first audit trail and every backup tar; it is
    already readable via the per-app config API). `data_rollback` is omitted
    on success and `false` on every failure: the platform reverts the image,
    never the data.
    """
    params: dict[str, object] = {
        "service": job.service_name,
        "version": version,
        "exit_code": exit_code,
        "timed_out": timed_out,
    }
    if data_rollback is not None:
        params["data_rollback"] = data_rollback
    return params


def _sensitive_env_values(resolved: "ResolvedLaunchEnv") -> set[str]:
    """Collect declared secrets and credential-named env values for tail scrubbing.

    Scan the full resolved env, not only binding keys. Apply the minimum length to
    heuristic matches only; declared secrets are masked at every nonzero length.
    The downstream scrubber must not reapply that floor after provenance is lost.

    Literal matching cannot mask transformed values. Shared values embedded in a
    DSN are masked as the whole DSN because this tier receives shared key names,
    not their raw values. Both releases and one-off runs use this same set.
    """
    # Empty values are dropped here rather than downstream: a "" target would
    # splice `***` between every character of every line.
    values = {value for value in resolved.secret_env.values() if value}
    for key, value in resolved.env.items():
        if value and len(value) >= _SCRUB_MIN_VALUE_LEN and _is_credential_key(key):
            values.add(value)
    return values


def _sensitive_override_values(applied: dict[str, str]) -> set[str]:
    """Collect credential-named values from applied run overrides.

    Use the same heuristic and length floor as ordinary env values. Ignore rejected
    overrides: they never enter the container and masking them could hide unrelated
    output. Declared secrets retain their unconditional treatment in the base set.
    """
    return {
        value
        for key, value in applied.items()
        if value and len(value) >= _SCRUB_MIN_VALUE_LEN and _is_credential_key(key)
    }


class AppImageBuilder:
    """Build images and settle releases through controller-owned hooks and registries."""

    def __init__(self, controller: "ServiceController") -> None:
        self._c = controller
        # Global build-concurrency cap: a single semaphore bounds how
        # many app-image builds run at once across all services, so a fleet of
        # simultaneous deploys can't thrash the build host. The controller's
        # per-job `_build_tasks` map is a *dedupe* (one build per row), NOT a
        # cap — this is the cap. Restart-required (read once at construction).
        self.semaphore = asyncio.Semaphore(controller._services_settings.max_concurrent_builds)

    def busy(self) -> int:
        """Return the number of image builds still running for restart drain."""
        return sum(1 for task in self._c._build_tasks.values() if not task.done())

    async def needs_build(self, job: Job) -> bool:
        """Check whether an app's target image still needs building.

        Require build context and either a missing image or a queued generation that
        never reached a build task. Existing completed tags allow immediate launch.
        """
        cfg = parse_job_config(job, warn=True)
        context, image = cfg.get("build_context_dir"), cfg.get("image")
        if not context or not image:
            return False
        if not await self._c._runtime.image_exists(str(image)):
            return True
        return _build_never_ran(cfg)

    def _task_in_flight(self, job_id: str) -> bool:
        """Whether a tracked build/release task is still running for `job_id`.

        Deliberately `not task.done()` rather than bare dict membership. An
        entry is normally popped by `spawn`'s `finally`, but that `finally`
        only runs if the coroutine actually STARTED: a task cancelled between
        `create_task` and its first step (`shutdown`) never runs it, leaving
        a done entry behind. Bare membership would then make
        `ensure_built` return `True` forever and the row could never
        converge again — an unrecoverable wedge traded for a theoretical race.
        The done-aware read degrades the other way (no gate) in exactly the case
        where nothing is executing, and matches `busy`'s existing idiom.
        """
        task = self._c._build_tasks.get(job_id)
        return task is not None and not task.done()

    async def ensure_built(self, job: Job, live: bool) -> bool:
        """Block launch or swap while a build/release needs to finish or settle.

        Check active tasks before image existence: releases run after tagging, while
        the previous container must keep serving. Start pending builds off-tick.
        Then settle a current-generation release marker with no task as crashed,
        reverting when possible. Only redeploy may retry an unobserved migration.
        Healthy running services do not need an image probe.

        Returns:
            True if the caller must leave containers untouched this tick.
        """
        # --- layer 1: an in-flight build/release task owns the row ---
        if self._task_in_flight(job.id):
            return True

        converging = not live or job.status in (JobStatus.building, JobStatus.restarting)
        if not converging:
            # A healthy live service: no build can be pending and no release can
            # be mid-flight (a release only ever runs while the row sits
            # `building`/`restarting`), so skip both probes below entirely.
            return False

        # --- layer 2: a target image still to build ---
        # Through the controller delegate, not self.needs_build — the D-T-2
        # patch-point contract: an instance-assigned override on the CONTROLLER
        # must keep intercepting this predicate.
        if await self._c._needs_build(job):
            if not self._task_in_flight(job.id):
                # Boot touch-up: a row stuck at phase "building" with
                # no live build task (e.g. a daemon restart mid-build) regresses
                # to "queued" — the freshly spawned build re-stamps "building"
                # once it acquires a slot.
                await stamp_last_deploy(
                    self._c._queries, job.id, only_from=("building",), phase="queued"
                )
                cfg = parse_job_config(job, warn=True)
                # Docker builds from `build_context_dir` (a git ingress builds
                # from a nested subdir), but cleanup removes the ingress-owned
                # clone ROOT (`build_context_root`) so sibling repo contents
                # don't leak. ZIP rows have no separate root → fall back to the
                # context dir.
                cleanup_dir = str(cfg.get("build_context_root") or cfg["build_context_dir"])
                self._c._spawn_build_task(
                    job,
                    str(cfg["build_context_dir"]),
                    str(cfg["image"]),
                    cfg.get("dockerfile_name"),
                    cleanup_dir,
                )
            return True

        # --- layer 3: a release this daemon never finished ---
        cfg = parse_job_config(job, warn=True)
        pending = cfg.get("release_pending")
        # `is not None` is load-bearing: both keys are absent on the vast
        # majority of rows, and `None == None` would settle every one of them.
        if pending is not None and pending == cfg.get("build_version"):
            await self._settle_crashed_release(job, pending)
            return True
        return False

    async def _settle_crashed_release(self, job: Job, pending: object) -> None:
        """Fail an unobserved release without retrying its migration.

        Kill labelled migration containers best-effort before settling. Revert to the
        previous image if a live container exists; otherwise leave a terminal failure.
        Restart re-settles that marker, while redeploy mints a new generation to retry.
        Keep the candidate tag as crash evidence and to avoid triggering a rebuild.
        """
        # Reap the crash-orphaned migration container by label. After
        # a restart the in-memory registry (and the run_id) is gone, so
        # `nerdit-job=<job.id>` is the only recoverable key. Belt: skip the
        # whole pass while ANY live slot exists for this row — `ensure_built`
        # layer 1 already prevents reaching here during our OWN release, but
        # layer 1 keys on the build-task registry while `_active_runs` is the
        # authority on live containers (e.g. a route-initiated run against
        # this crashed row), and a just-started run container may not be bound
        # yet. Best-effort and swallow-everything: this settle runs inside the
        # reconcile tick and a docker error must never wedge the row.
        try:
            if not self._c.has_active_run(job.id):
                orphans = await self._c._runtime.list_own_labeled_containers("nerdit-job", job.id)
                for cid in orphans:
                    # Re-checked AFTER every await, not just once up front: a
                    # route-initiated run can claim its slot and bind its
                    # container while the docker listing (or a preceding
                    # kill/remove) is in flight, and it WOULD be in `orphans`
                    # — the labels are stamped for every rowless container,
                    # runs included. Killing it is precisely the harm this
                    # reap exists to prevent, a SIGKILL mid-write on the
                    # service data volume. An unbound slot has no id to
                    # subtract, so ANY live slot aborts the pass; the zombie
                    # sweep remains the backstop for whatever is left.
                    if self._c.has_active_run(job.id):
                        logger.warning(
                            "Aborting the orphan reap for service %s: a run "
                            "claimed a slot mid-pass; the zombie sweep will "
                            "collect any remaining orphan",
                            job.service_name or job.id,
                        )
                        break
                    try:
                        await self._c._runtime.kill(cid)
                        await self._c._runtime.remove(cid, force=True)
                        logger.warning(
                            "Reaped crash-orphaned release container %s of service %s",
                            cid,
                            job.service_name or job.id,
                        )
                    except ContainerNotFoundError:
                        continue
                    except ContainerRuntimeError:
                        logger.warning(
                            "Could not reap orphaned release container %s", cid, exc_info=True
                        )
        except Exception:  # noqa: BLE001 — the settle must never fail on the reap
            logger.warning(
                "Orphan-container reap failed for service %s; the zombie sweep will collect it",
                job.service_name or job.id,
                exc_info=True,
            )

        target_version = pending if isinstance(pending, int) else None
        await self._settle_failed_generation(
            job,
            reason="release_failed",
            message=(
                "daemon restarted during release; migration state unknown — redeploy to retry"
            ),
            # Not the user's fault and not a container failure: the outcome is
            # genuinely unobserved.
            error_class=ErrorClass.unknown,
            target_version=target_version,
            audit_action="service.release_failed",
            audit_params=_release_audit_params(
                job, target_version, exit_code=None, timed_out=False, data_rollback=False
            ),
        )

    def spawn(
        self, job: Job, context_dir: str, image: str, dockerfile: str | None, cleanup_dir: str
    ) -> None:
        """Run a deploy image build off the reconcile tick, tracked by job id."""
        # Drain gate: a restart is winding the daemon down — do not
        # start (or register) a new build. The row stays `building` with
        # `last_deploy.phase` at "queued" (the boot touch-up already regressed
        # it), the build context is left on disk untouched, and the freshly
        # re-exec'd daemon re-detects `_needs_build` and rebuilds. Returning here
        # (rather than inside the build) avoids the `_build_app_image` finally
        # cleanup that would rmtree the still-needed context.
        if self._c._draining:
            return

        async def _run() -> None:
            try:
                # Preserve controller-level overrides.
                await self._c._build_app_image(job, context_dir, image, dockerfile, cleanup_dir)
            finally:
                self._c._build_tasks.pop(job.id, None)

        self._c._build_tasks[job.id] = asyncio.create_task(_run())

    async def build(
        self, job: Job, context_dir: str, image: str, dockerfile: str | None, cleanup_dir: str
    ) -> None:
        """Build the app image for a deploy, streaming build logs into `job_logs`.

        On success the image is present locally and the row stays `building`;
        the next reconcile tick sees the image and launches the container. A
        build failure (bad Dockerfile / npm error) is deterministic, so the row
        is settled to a terminal `failed` — never a retry loop — mirroring the
        `SandboxViolationError` handling in `ServiceController._launch`.
        The one exception is a **redeploy** whose old container is still live:
        the failed build only means the *new* image isn't ready, so we revert
        the row to `running` and leave the previous version serving rather
        than orphaning a live container behind a `failed` row. Both branches
        live in `_settle_failed_generation`.
        """
        await self._c._append_log_tolerant(job.id, f"Building image {image} ...", LogStream.system)
        # The build context is only needed for the build itself; drop it on EITHER
        # outcome (success or a failed build) so successive deploys never leak
        # extracted upload dirs.
        try:
            try:
                # Hold the global build semaphore only for the actual image
                # build, so a burst of deploys queues here instead of thrashing
                # the build host. The revert/cleanup below runs outside the cap.
                async with self.semaphore:
                    # A build slot is now held: "queued" (waiting for a
                    # slot) → "building". Guarded so a re-entrant/adopted build
                    # never regresses a later phase.
                    await stamp_last_deploy(
                        self._c._queries, job.id, only_from=("queued",), phase="building"
                    )
                    async for line in self._c._runtime.build_image(
                        context_dir, image, dockerfile=dockerfile
                    ):
                        # `build`, not `stdout`: BuildKit's layer
                        # chatter and the app's own output are different
                        # streams, and an agent grepping a deployed app's logs
                        # must be able to exclude one. See `LogStream.build`.
                        await self._c._append_log_tolerant(job.id, line, LogStream.build)
            # (BUG-1) Most specific first: a build that failed because THIS NODE
            # has no BuildKit builder is a host fault, not the app's source.
            # `reason` stays `build_failed` on both branches — that
            # vocabulary feeds audit (`service.build_failed`) and the
            # message derivation; only the *class* carries the distinction.
            except BuildPlatformError as exc:
                await self._c._append_log_tolerant(job.id, f"Build failed: {exc}", LogStream.system)
                target_version = _version_of_tag(image)
                await self._settle_failed_generation(
                    job,
                    reason="build_failed",
                    # Captured eagerly: `exc` is cleared when the block exits.
                    message=f"Build failed: {exc}",
                    error_class=ErrorClass.platform_error,
                    target_version=target_version,
                )
                return
            except ContainerRuntimeError as exc:
                await self._c._append_log_tolerant(job.id, f"Build failed: {exc}", LogStream.system)
                # This build targeted the version encoded in `image`.
                # A last_deploy=failed stamp must only settle THAT generation —
                # never a newer redeploy whose config landed while we built.
                target_version = _version_of_tag(image)
                # Share generation guards and settlement with release failures.
                await self._settle_failed_generation(
                    job,
                    reason="build_failed",
                    # Captured eagerly: `exc` is cleared when the block exits.
                    message=f"Build failed: {exc}",
                    error_class=ErrorClass.user_error,
                    target_version=target_version,
                )
                return
            await self._c._append_log_tolerant(
                job.id, f"Image {image} built successfully", LogStream.system
            )
            # Pre-swap release gate, BEFORE the image is allowed to
            # serve: the swap itself is held back by `ensure_built` for as
            # long as this task is registered. It runs under NO concurrency
            # bound — the `async with self.semaphore` block above has already
            # closed, and D-P20-2 exempts a release from both run caps — so
            # simultaneous releases are bounded only by how many services
            # happen to be deploying at once. Deliberate: queueing a migration
            # behind a cap the deployer cannot see would wedge the deploy
            # behind unrelated work. A False return means the generation was
            # already settled failed — skip the prune so the candidate tag the
            # failure path just removed is not immediately re-listed, and leave
            # the row exactly as the settle left it.
            if not await self._run_release(job, image):
                return
            await self._c._prune_old_images(job)
        finally:
            await self.cleanup_context(job, cleanup_dir)

    async def _run_release(self, job: Job, image: str) -> bool:
        """Run the built generation's release and contain unexpected task failures.

        Settle unexpected exceptions immediately so a tagged but unmigrated image
        cannot pass the next tick's swap gate. If settlement itself fails, the armed
        crash marker supplies the fallback.

        Returns:
            True when no release is required or it succeeds; False when build must stop.
        """
        try:
            return await self._release_impl(job, image)
        except Exception:  # noqa: BLE001 - an escape here wedges the row (see docstring)
            logger.exception(
                "Release handling failed unexpectedly for service %s; settling the generation "
                "failed without swapping the image",
                job.service_name or job.id,
            )
            try:
                await self._settle_unexpected_release_failure(job, image)
            except Exception:
                logger.exception(
                    "Could not settle the failed release for service %s; the generation stays "
                    "unsettled and is re-settled from release_pending on the next tick",
                    job.service_name or job.id,
                )
            return False

    async def _release_impl(self, job: Job, image: str) -> bool:
        """Body of the release gate — see `_run_release` for the contract."""
        # Local import: `core/services` imports THIS module at load time (the
        # controller constructs an AppImageBuilder), so a controller-tier
        # exception type can only be reached lazily. A build task never runs at
        # import time, so the module is always fully loaded by now.
        from nerdit.core.services import LaunchEnvNotReady, RunInterruptedError

        target_version = _version_of_tag(image)
        # (a) Fresh row + version CAS. The tick-start snapshot is minutes stale
        # by now: a redeploy may have landed mid-build, and the release command
        # itself belongs to whichever generation owns the row.
        fresh = await self._c._queries.get_job(job.id)
        if fresh is None:
            logger.info(
                "Service %s vanished during its build; skipping release",
                job.service_name or job.id,
            )
            return False
        cfg = parse_job_config(fresh, warn=True)
        release = cfg.get("release")
        if not isinstance(release, str) or not release.strip():
            return True  # no release declared for this generation — nothing to gate
        if target_version is not None and cfg.get("build_version") != target_version:
            # A newer deploy owns the row and its own build task drives it
            # forward, release included. Running ours would execute a migration
            # on behalf of a superseded generation.
            await self._c._append_log_tolerant(
                job.id,
                f"[release] skipped for superseded version {target_version}; "
                "a newer deploy owns the row.",
                LogStream.system,
            )
            return False

        run_id = generate_id()
        # Claimed BEFORE any awaited work so a DELETE arriving mid-migration is
        # refused (409 service.run_in_progress) and the zombie sweeper protects
        # the container. A release is exempt from both D-P20-2 caps (and the
        # build semaphore was released before we got here), so this slot is a
        # registry entry, never a bound, and it never raises.
        self._c._register_run(job.id, run_id, is_release=True)
        # Filled in by `_execute_release` as soon as the env resolves; an
        # out-parameter because the failure branches below need the scrub set
        # even when the call raised past its own return.
        scrub: set[str] = set()
        try:
            # Arm the crash marker HERE — the first write after
            # the version CAS, i.e. the moment we know a release IS declared
            # for THIS generation and we are committed to running it. Arming it
            # at deploy time instead made it mean "a release was declared",
            # which layer 3 of `ensure_built` cannot distinguish from a
            # crash: a deploy that skipped the build (target tag already
            # present, e.g. redeploying a name whose images survived a default
            # `?purge=secrets` delete) never runs a release, yet settled
            # instantly as a phantom crashed one. Arming it any LATER (once the
            # env/volumes/spec are assembled) left every failure in between
            # both unsettled and unmarked, so layer 3 could not back the swap
            # gate up. Everything from here to the container exiting settles
            # `release_failed` on a restart — the fail-closed direction.
            if not await self._arm_release_pending(job.id, target_version):
                # The (a) CAS passed but a redeploy committed before the marker
                # write. Same supersession as the early-return above — abort,
                # or the OLD generation's migration runs against the live
                # volumes with no crash marker armed (Codex P1, PR #100). The
                # `finally` still frees the run slot claimed above.
                await self._c._append_log_tolerant(
                    job.id,
                    f"[release] skipped for superseded version {target_version}; "
                    "a newer deploy took the row before the release marker was armed.",
                    LogStream.system,
                )
                return False
            exit_code: int | None = None
            timed_out = False
            tail: list[str] = []
            try:
                result = await self._execute_release(fresh, cfg, image, run_id, release, scrub)
            except LaunchEnvNotReady as exc:
                # NOT the launch path's retry-next-tick: a deploy is a one-shot
                # gate, the old container keeps serving, and a half-converged
                # deploy must not sit in `restarting` forever waiting for a
                # binding. Fail the generation; a redeploy retries.
                message = f"Release could not start: {exc.message}"
                error_class = ErrorClass.user_error
            except VolumeSpecError as exc:
                message = f"Release could not start: invalid volume spec: {exc}"
                error_class = ErrorClass.user_error
            except RunInterruptedError as exc:
                # Checked BEFORE its `ContainerRuntimeError` base: the
                # container DID start and the runtime lost it mid-flight, so
                # "failed to start" would tell an operator whose migration ran
                # for two minutes exactly the wrong story — they would assume
                # the schema is untouched. The exception carries the output the
                # container managed to print (already bounded and scrubbed
                # through the same D-P20-1 choke point), which is the only
                # evidence left of how far the migration got.
                tail = exc.log_tail
                message = (
                    f"Release container was lost while running: {exc}; the image was NOT "
                    "swapped. The migration may have PARTIALLY applied — data changes are "
                    "not rolled back; check the release output above, make the migration "
                    "idempotent and redeploy."
                )
                # Not a start failure and not the app's fault: the container's
                # own exit is genuinely unobserved.
                error_class = ErrorClass.unknown
            except ContainerRuntimeError as exc:
                message = f"Release container failed to start: {exc}"
                error_class = ErrorClass.container_fail
            except TimeoutError:
                # Defensive: the executor enforces the cap internally and
                # reports it as `timed_out`, so this only fires if a runtime
                # ever lets one escape.
                message = f"Release timed out after {self._release_timeout_s}s"
                error_class = ErrorClass.timeout
            else:
                exit_code, timed_out, tail = result.exit_code, result.timed_out, result.log_tail
                if exit_code == 0 and not timed_out:
                    await self._finish_release_ok(fresh, target_version, tail)
                    return True
                message, error_class = self._release_failure_reason(result)
            await self._settle_release_failure(
                fresh,
                image,
                target_version,
                message=_scrub_text(message, scrub),
                error_class=error_class,
                exit_code=exit_code,
                timed_out=timed_out,
                tail=tail,
            )
            return False
        finally:
            # A leaked slot wedges DELETE for this service until the daemon
            # restarts, so the discard is unconditional (S1).
            self._c._discard_run(job.id, run_id)

    @property
    def _release_timeout_s(self) -> int:
        """Wall-clock cap on one release execution (`[services].release_timeout_s`)."""
        return self._c._services_settings.release_timeout_s

    async def _execute_release(
        self,
        job: Job,
        cfg: dict,
        image: str,
        run_id: str,
        release: str,
        scrub: set[str],
    ) -> "RunResult":
        """Resolve the env + volumes and run the release container to completion.

        Steps (b)–(d) of the gate. The container is the **candidate image** with
        the **new generation's** resolved env — the whole point of a pre-swap
        release is that the migration runs with what the new code will see — and
        it mounts the SAME named volumes the live old container is still writing
        to, since that is where the data to migrate lives.

        `scrub` is an out-parameter: it is populated with the D-P20-1
        sensitive-value set the moment the env resolves, so a failure raised
        later still has the set available for redacting its own message.
        """
        container_port = int(cfg.get("port") or 8000)
        # Spec derivation copied from `_launch`: both binding kinds are
        # suppressed for model/database rows (they are bound-TO resources, never
        # binding consumers). A release only ever runs for a deploy-created
        # kind=service row, so this is defence in depth, not a live branch.
        is_model = job.kind is JobKind.model and self._c._models is not None
        is_data = job.kind is JobKind.database and self._c._data is not None
        ai_specs = cfg.get("ai") if not (is_model or is_data) else None
        db_specs = cfg.get("db") if not (is_model or is_data) else None
        resolved = await self._c._resolve_launch_env(job, cfg, container_port, ai_specs, db_specs)
        scrub.update(_sensitive_env_values(resolved))
        # Shared-scope usage is audited on every launch-shaped resolution,
        # key NAMES only — a release consuming ${secrets.shared.KEY} must not be
        # the one resolution that leaves no trail. Same helper, same kwargs as
        # `_launch`; its per-row dedupe makes a repeat cheap.
        if resolved.shared_keys:
            await self._c._audit_shared_resolved(job, resolved.shared_keys, resolved.secret_env)

        # Named volumes, materialized exactly as `_launch` does. The
        # `_data_dir is not None` guard comes FIRST: a bare-constructed
        # controller has none, and `service_volumes` would fail on it.
        named_volumes: dict[str, str] = {}
        if self._c._data_dir is not None and job.service_name and cfg.get("volumes"):
            named_volumes = service_volumes(self._c._data_dir, job.service_name, cfg)
            if named_volumes:
                self._c._ensure_volume_dirs(job.service_name, named_volumes)

        config = build_run_container_config(
            cfg,
            self._c._container_settings,
            image=image,
            # Shell-wrapped, a deliberate divergence from
            # `[deploy].start` (which `_build_command` splits with
            # `shlex.split` into exec form): migrations want `&&`, pipes and
            # redirection. The string is app-owner-authored and runs only inside
            # the candidate container — the same trust domain as that repo's own
            # Dockerfile — never a host shell. An image without `/bin/sh`
            # fails the release with a clear tail.
            command=["/bin/sh", "-c", release],
            env=resolved.env,
            # No workdir override: the release runs in the image's own WORKDIR,
            # exactly where the app's own start command runs.
            workdir=None,
        )
        finalize_container_config(config, named_volumes, self._c._retention_settings)

        timeout_s = self._release_timeout_s
        await self._c._append_log_tolerant(
            job.id, f"[release] running (timeout {timeout_s}s)", LogStream.system
        )
        return await self._c._execute_container_once(
            job,
            run_id=run_id,
            config=config,
            timeout_s=timeout_s,
            log_tail=_RELEASE_LOG_TAIL,
            scrub_values=scrub,
        )

    def _release_failure_reason(self, result: "RunResult") -> tuple[str, ErrorClass]:
        """Message + `ErrorClass` for a release container that ran and lost."""
        if result.timed_out:
            return (
                f"Release command timed out after {self._release_timeout_s}s and was killed; "
                "the image was NOT swapped. Data changes are not rolled back — make the "
                "migration idempotent (or raise [services].release_timeout_s) and redeploy.",
                ErrorClass.timeout,
            )
        if result.oom_killed:
            return (
                "Release command was OOM-killed; the image was NOT swapped. Data changes are "
                "not rolled back — raise [deploy].memory_limit and redeploy.",
                ErrorClass.oom,
            )
        return (
            f"Release command failed (exit {result.exit_code}); the image was NOT swapped. "
            "Data changes are not rolled back — make the migration idempotent, fix it and "
            "redeploy.",
            ErrorClass.user_error,
        )

    async def _finish_release_ok(
        self, job: Job, target_version: int | None, tail: list[str]
    ) -> None:
        """Clear the crash marker, record the tail, audit — then let the swap happen.

        Order is load-bearing: the marker is cleared BEFORE this task finishes,
        so there is no window in which the task is gone AND the marker still
        says a release is pending (layer 3 of `ensure_built` would settle
        a perfectly good generation).
        """
        await self._clear_release_pending(job.id, target_version)
        await self._c._append_log_tolerant(job.id, "[release] succeeded (exit 0)", LogStream.system)
        await self._append_release_tail(job.id, tail)
        # The ONE canonical success row (§1.3). `record_out_of_band` is
        # request-bound and unusable from a build task, so this goes straight to
        # the audit table with the `principal='system'` shape every autonomous
        # controller transition uses.
        await self._audit_settle(
            "service.release",
            job,
            _release_audit_params(job, target_version, exit_code=0, timed_out=False),
        )
        logger.info(
            "Release for service %s version %s succeeded",
            job.service_name or job.id,
            target_version,
        )

    async def _settle_unexpected_release_failure(self, job: Job, image: str) -> None:
        """Settle a generation whose release raised something nobody enumerated.

        Re-reads the row so the settle branches on the CURRENT container/config
        state (this runs minutes after the tick-start snapshot), and derives the
        generation from the image tag rather than from the raised-past `cfg`,
        which may never have been read. A row that vanished mid-release has no
        generation to settle.
        """
        fresh = await self._c._queries.get_job(job.id)
        if fresh is None:
            return
        await self._settle_release_failure(
            fresh,
            image,
            _version_of_tag(image),
            message=(
                "Release failed unexpectedly; the image was NOT swapped. Data changes are not "
                "rolled back - check the daemon log, fix the cause and redeploy."
            ),
            # The migration's own outcome is unobserved: the failure is in the
            # platform's handling of it, not in the container's exit.
            error_class=ErrorClass.unknown,
            exit_code=None,
            timed_out=False,
            tail=[],
        )

    async def _settle_release_failure(
        self,
        job: Job,
        image: str,
        target_version: int | None,
        *,
        message: str,
        error_class: ErrorClass,
        exit_code: int | None,
        timed_out: bool,
        tail: list[str],
    ) -> None:
        """Record release output, settle failure, and reclaim only safe candidate tags.

        Settlement owns the single audit row and reports `data_rollback: false`.
        Live-container reverts clear build/release markers to prevent silent swaps.
        Reclaim only when settlement permits it and the image is not a newer
        rollback target; deleting terminal candidates could trigger migration reruns.
        """
        await self._append_release_tail(job.id, tail)
        reclaimable = await self._settle_failed_generation(
            job,
            reason="release_failed",
            message=message,
            error_class=error_class,
            target_version=target_version,
            audit_action="service.release_failed",
            audit_params=_release_audit_params(
                job,
                target_version,
                exit_code=exit_code,
                timed_out=timed_out,
                data_rollback=False,
            ),
        )
        # Best-effort: the candidate image is now unreferenced, and a
        # repeatedly-failing release must not accumulate one dangling tag per
        # attempt. An "in use" error is swallowed — the next successful
        # deploy's keep-last-3 prune reclaims it.
        if not reclaimable or await self._is_rollback_target(job.id, image):
            return
        try:
            await self._c._runtime.remove_image(image)
        except Exception:  # noqa: BLE001 - reclaiming a tag never fails a settle
            logger.info(
                "Could not remove failed-release image %s for service %s",
                image,
                job.service_name or job.id,
                exc_info=True,
            )

    async def _is_rollback_target(self, job_id: str, image: str) -> bool:
        """Check whether a newer generation now needs this candidate as its rollback image.

        Consult `previous_image`, not `image`: settlement separately decides whether
        the current candidate can be removed without triggering a rebuild. Missing or
        unreadable rows allow reclamation.
        """
        row = await self._c._queries.get_job(job_id)
        if row is None:
            return False
        return parse_job_config(row, warn=True).get("previous_image") == image

    async def _append_release_tail(self, job_id: str, tail: list[str]) -> None:
        """Append a release's captured output to `job_logs`, one prefixed line each.

        The lines arrive **bounded at read time, then scrubbed, then per-line
        truncated** — that order, from the executor: clamping a line
        before masking it would let the surviving prefix of an oversized secret
        through. `job_logs` is readable by any authenticated principal — like
        build logs — which is why the scrub happens before anything reaches
        here; the documented residual is a command that transforms a secret
        before printing it.
        """
        for line in tail:
            await self._c._append_log_tolerant(job_id, f"[release] {line}", LogStream.stdout)

    async def _arm_release_pending(self, job_id: str, target_version: object) -> bool:
        """Arm a release crash marker with a database-enforced version CAS.

        A fresh read alone cannot prevent a redeploy between read and write. Arm only
        when release begins, not when merely configured. Non-numeric tags have no
        version guard and return True without a marker.

        Returns:
            Whether the caller still owns the generation and may run the command.
            False requires stopping before release to avoid an untracked migration.
        """
        if target_version is None:
            return True
        row = await self._c._queries.get_job(job_id)
        if row is None:
            return False
        cfg = parse_job_config(row, warn=True)
        if cfg.get("build_version") != target_version:
            return False  # a newer generation owns the row
        cfg["release_pending"] = target_version
        if isinstance(target_version, int):
            return await self._c._queries.update_job_config_guarded(
                job_id, json.dumps(cfg), expect_build_version=target_version
            )
        # Defensive: `_version_of_tag` only ever yields `int | None` and the
        # `None` case returned above, so there is nothing to CAS on here.
        await self._c._queries.update_job_config(job_id, json.dumps(cfg))
        return True

    async def _clear_release_pending(self, job_id: str, target_version: int | None) -> None:
        """Clear the release marker with a database-enforced version CAS.

        Never overwrite a newer generation's config or disarm its release. Non-numeric
        tags have no version to compare and retain an unguarded write.
        """
        row = await self._c._queries.get_job(job_id)
        if row is None:
            return
        cfg = parse_job_config(row, warn=True)
        if target_version is not None and cfg.get("build_version") != target_version:
            return  # a newer generation owns the row (and the marker)
        if cfg.pop("release_pending", None) is None:
            return
        if target_version is not None:
            await self._c._queries.update_job_config_guarded(
                job_id, json.dumps(cfg), expect_build_version=target_version
            )
            return
        await self._c._queries.update_job_config(job_id, json.dumps(cfg))

    async def _write_terminal_settle(
        self,
        job_id: str,
        *,
        target_version: int | None,
        error_class: ErrorClass,
        message: str,
    ) -> bool:
        """Atomically fail status and desired state only for the owned generation.

        A stale terminal write would remove newer work from reconciliation permanently.
        Use the image tag's version, not the config blob; a NULL stored version has no
        newer generation to protect. A CAS miss writes nothing.
        """
        if target_version is None:
            # Non-numeric image tag (pre-P13 / odd rows): nothing to CAS on, so
            # keep today's two unconditional writes.
            await self._c._queries.update_job_status(
                job_id,
                JobStatus.failed,
                finished_at=datetime.now(UTC),
                exit_code=-1,
                error_class=error_class,
                error_message=message,
            )
            await self._c._queries.set_desired_state(job_id, JobStatus.failed.value)
            return True
        return await self._c._queries.settle_failed_guarded(
            job_id,
            expect_build_version=target_version,
            finished_at=datetime.now(UTC),
            exit_code=-1,
            error_class=error_class,
            error_message=message,
        )

    async def _settle_failed_generation(
        self,
        job: Job,
        *,
        reason: str,
        message: str,
        error_class: ErrorClass,
        target_version: int | None,
        audit_action: str = "service.build_failed",
        audit_params: dict[str, object] | None = None,
    ) -> bool:
        """Settle a failed generation without overwriting newer work.

        With a live old container, reread and version-CAS the row, restoring the previous
        image or retaining the failed tag with build markers stripped. Audit with
        `audit_action`. Without a live container, mark terminal failure and always audit
        `service.failed`. Supplied audit params replace the default payload.

        Returns:
            Whether the candidate tag may be reclaimed. Live-container branches strip
            build markers, so deleting an unmigrated tag prevents silent later launch
            without enabling a rebuild. Terminal branches retain the image, build
            context, and release marker: deleting there would rebuild and rerun the
            migration. Superseded/no-op branches also return False.
        """
        # "build_failed" → "Build failed"; "release_failed" → "Release failed".
        # Keeps every log line below reading correctly for either caller while
        # rendering byte-identically to the pre-extraction text for a build.
        label = reason.replace("_", " ").capitalize()

        def _stamp_failed(blob: dict) -> None:
            ld = blob.get("last_deploy")
            if not isinstance(ld, dict):
                return
            if target_version is not None and ld.get("version") != target_version:
                return  # a newer generation landed mid-build — leave it alone
            new_ld = dict(ld)
            new_ld.update(
                phase="failed",
                reason=reason,
                error_class=error_class.value,
                error_message=message,
                updated_at=datetime.now(UTC).isoformat(),
            )
            blob["last_deploy"] = new_ld

        async def _audit_outcome(action: str) -> None:
            await self._audit_settle(action, job, audit_params)

        # (F1-CLOBBER) The revert write is a version-guarded CAS: it settles the
        # row only while it still owns `target_version`. A newer redeploy
        # generation that landed on the row mid-build owns it now, and its own
        # build task drives it forward — writing our (superseded) revert blob
        # would clobber its version state and permanently wedge it
        # (build_version reverts, context dir popped → no self-heal).
        async def _mark_superseded() -> None:
            await self._c._append_log_tolerant(
                job.id,
                f"{label} for superseded version {target_version}; a newer deploy owns the row.",
                LogStream.system,
            )
            await _audit_outcome(audit_action)
            logger.warning(
                "%s for superseded service %s version %s; a newer deploy owns the row: %s",
                label,
                job.service_name or job.id,
                target_version,
                message,
            )

        async def _write_revert(blob: dict) -> bool:
            """Guarded revert write; `False` ⇒ a newer generation owns the row."""
            if target_version is None:
                # Non-numeric image tag (pre-P13 / odd rows): nothing to
                # CAS on, so fall back to the unguarded write.
                await self._c._queries.update_service_config(
                    job.id,
                    json.dumps(blob),
                    status=JobStatus.running,
                    desired_state="running",
                )
                return True
            return await self._c._queries.revert_service_config_guarded(
                job.id,
                json.dumps(blob),
                expect_build_version=target_version,
                status=JobStatus.running,
                desired_state="running",
            )

        # A redeploy over a previous container the runtime still knows (running
        # OR exited — an exited previous generation still reverts and self-heals
        # via the reconcile relaunch): keep the previous version serving instead
        # of failing the whole service (would orphan it).
        if job.container_id and await self._c._container_exists(job.container_id):
            # Re-read the row NOW rather than
            # rewriting from the tick-start `job` snapshot: a
            # last_deploy/forensics write that landed mid-build (or a
            # newer redeploy generation) must not be clobbered.
            fresh = await self._c._queries.get_job(job.id)
            cfg = parse_job_config(fresh if fresh is not None else job, warn=True)
            # Fast path: a newer generation already owns the row (its
            # build_version differs). Write nothing.
            if target_version is not None and cfg.get("build_version") != target_version:
                await _mark_superseded()
                return False
            prev_image = cfg.get("previous_image")
            if prev_image:
                # Roll the row's config back so it matches the still-running
                # OLD container: point `image` at `previous_image` and
                # DROP the build markers (`build_context_dir` /
                # `build_context_root` / `dockerfile_name`) so
                # `_needs_build` returns False.
                # Otherwise a later restart/crash would see the missing new
                # tag + the deleted (rmtree'd by the build's finally) context
                # and try to rebuild from a gone dir — failing a
                # previously-fine service.
                reverted = dict(cfg)
                reverted["image"] = prev_image
                suffix = str(prev_image).rsplit(":", 1)[-1]
                if suffix.isdigit():
                    reverted["build_version"] = int(suffix)
                reverted.pop("build_context_dir", None)
                reverted.pop("build_context_root", None)
                reverted.pop("dockerfile_name", None)
                # Same reasoning for the release marker: this generation
                # is dead, so nothing may later read it as "a release is still
                # pending" and refuse to converge (or settle it a second time).
                reverted.pop("release_pending", None)
                # Phase machine: the *new* generation failed even though the
                # old image keeps serving (row → running).
                _stamp_failed(reverted)
                # CAS-False here means a redeploy landed between the fresh
                # read and the write (the symmetric lost-update window):
                # supersession, so hands off.
                if not await _write_revert(reverted):
                    await _mark_superseded()
                    return False
                await self._c._append_log_tolerant(
                    job.id, f"{label}; keeping previous version.", LogStream.system
                )
                self._c._emit_status_change(job.id, JobStatus.running)
                await _audit_outcome(audit_action)
                # This path writes phase=failed OUTSIDE stamp_last_deploy (the
                # redeploy-over-live bypass), so emit the companion settle
                # events here — both read the reverted blob (phase=failed, the
                # settled reason, USER_ERROR).
                await self._emit_bypass_settle(job, reverted)
                logger.warning(
                    "%s for service %s; reverted config to %s: %s",
                    label,
                    job.service_name or job.id,
                    prev_image,
                    message,
                )
                # The row points at `previous_image` now: nothing references
                # the candidate tag any more, so it is the caller's to reclaim.
                return True
            # No rollback target (shouldn't happen on a redeploy with a live
            # container): keep it running and still DROP the build markers so
            # a later restart doesn't try to rebuild from the (rmtree'd)
            # context — even though `image` can only stay the failed tag.
            stripped = dict(cfg)
            stripped.pop("build_context_dir", None)
            stripped.pop("build_context_root", None)
            stripped.pop("dockerfile_name", None)
            stripped.pop("release_pending", None)  # see above
            _stamp_failed(stripped)
            if not await _write_revert(stripped):
                await _mark_superseded()
                return False
            await self._c._append_log_tolerant(
                job.id, f"{label}; keeping previous version.", LogStream.system
            )
            self._c._emit_status_change(job.id, JobStatus.running)
            await _audit_outcome(audit_action)
            # Same bypass path as above — phase=failed written outside
            # stamp_last_deploy; emit the companion events.
            await self._emit_bypass_settle(job, stripped)
            logger.warning(
                "%s for service %s; keeping previous version (no previous_image recorded): %s",
                label,
                job.service_name or job.id,
                message,
            )
            # Reclaimable, and deliberately so: the row is still convergeable
            # with `image` on the UNMIGRATED candidate and no rollback target
            # to point back at, so the moment the old container dies the
            # reconcile loop would launch the failed generation in front of
            # possibly half-migrated data — silently. Dropping the tag turns
            # that into a loud image-not-found. No rebuild can follow: this
            # branch popped the build markers, so `_needs_build` is False
            # whether or not the tag exists.
            return True
        # `release_pending` is deliberately NOT popped here,
        # and the asymmetry with the two revert blobs above is the point: those
        # move the row OFF this generation (`image` back to
        # `previous_image`, or a live container still serving), so the marker
        # has nothing left to protect. This branch leaves the row ON the
        # candidate image, with a migration whose outcome is either failed or
        # unobserved. The marker is what makes layer 3 of `ensure_built`
        # re-settle a `nerdit restart` instead of launching: a restart is not
        # evidence that the migration completed, and only a redeploy — which
        # mints a new `build_version` — retries a release. Popping it here
        # would swap the unmigrated candidate in on the next restart, the exact
        # failure the gate exists to prevent (pinned by
        # `test_a_restart_of_a_crashed_release_re_settles_rather_than_launching`).
        #
        # (F5) The row write is a build_version CAS — see
        # `_write_terminal_settle` for why this branch in particular must
        # never fire on a newer generation's behalf. The crash-loop sibling in
        # `services.py` (F5-CRASHLOOP) settles a DIFFERENT, deliberately
        # unguarded scenario and is untouched here.
        if not await self._write_terminal_settle(
            job.id,
            target_version=target_version,
            error_class=error_class,
            message=message,
        ):
            await _mark_superseded()
            return False
        await stamp_last_deploy(
            self._c._queries,
            job.id,
            expect_version=target_version,
            phase="failed",
            reason=reason,
            error_class=error_class.value,
            error_message=message,
        )
        self._c._emit_status_change(job.id, JobStatus.failed)
        # Fixed action on the terminal branch (never `audit_action`): a fresh
        # deploy that never came up is a `service.failed`, whatever killed it.
        await _audit_outcome("service.failed")
        logger.warning("%s for service %s: %s", label, job.service_name or job.id, message)
        # NOT reclaimable — the one branch that keeps the build markers. The
        # row still carries `build_context_dir`, so removing the tag would
        # make `_needs_build` True and layer 2 of `ensure_built` (which
        # runs BEFORE layer 3) would answer the next `nerdit restart` with a
        # REBUILD, re-executing a migration that already ran once. Keeping the
        # tag keeps layer 3 in charge: it re-settles the restart instead.
        return False

    async def _emit_bypass_settle(self, job: Job, blob: dict) -> None:
        """Emit the terminal deploy event for branches that bypass the phase writer.

        Best-effort after the critical config write. Cutover failure emits both
        `service.deploy_failed` and its separate cutover event, matching success.
        """
        ld = blob.get("last_deploy")
        if isinstance(ld, dict):
            await record_deploy_settled(job.service_name, ld)

    async def _audit_settle(
        self, action: str, job: Job, audit_params: dict[str, object] | None
    ) -> None:
        """Audit a builder outcome as system without allowing audit failure to block settlement.

        With no explicit params, preserve the controller's overridable audit hook.
        Explicit params replace its whole default payload rather than merging.
        """
        if audit_params is None:
            await self._c._audit(action, job)
            return
        try:
            await self._c._queries.insert_audit_log(
                action=action,
                result="ok",
                principal_id="system",
                principal_role="system",
                target_type="service",
                target_id=job.id,
                params_redacted=json.dumps(audit_params),
            )
        except Exception:
            logger.warning("Failed to audit %s for service %s", action, job.id, exc_info=True)

    async def cleanup_context(self, job: Job, context_dir: str) -> None:
        """Best-effort removal of an extracted deploy context dir.

        Logs (never raises) on failure: a leftover context is a minor disk
        leak, never a reason to fail an otherwise-settled build.
        """
        try:
            await asyncio.to_thread(shutil.rmtree, context_dir)
        except FileNotFoundError:
            pass  # already gone (route-level cleanup or a concurrent build)
        except OSError:
            logger.warning(
                "Failed to remove build context %s for service %s (leaked dir)",
                context_dir,
                job.service_name or job.id,
                exc_info=True,
            )

    async def prune_old_images(self, job: Job) -> None:
        """Keep the last 3 versions of a deployed app's image; remove older tags.

        Best-effort (`remove_image` swallows "in use" errors). The current
        image and the recorded `previous_image` (the rollback target) are
        never pruned even if they fall outside the newest three.
        """
        cfg = parse_job_config(job, warn=True)
        repo = cfg.get("image_repo")
        if not repo:
            return
        keep = {t for t in (cfg.get("image"), cfg.get("previous_image")) if t}
        prefix = f"{repo}:"

        def _version(tag: str) -> int:
            suffix = tag[len(prefix) :]
            return int(suffix) if suffix.isdigit() else -1

        try:
            tags = [t for t in await self._c._runtime.list_images() if t.startswith(prefix)]
        except Exception:
            logger.warning("Could not list images to prune %s", repo, exc_info=True)
            return
        keep |= set(sorted(tags, key=_version, reverse=True)[:3])
        for tag in tags:
            if tag not in keep:
                await self._c._runtime.remove_image(tag)
