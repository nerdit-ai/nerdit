"""Tests for the GitWatch poller (P24c / WP10, D-P24-8).

Real in-memory DB, a fake ``ls_remote_head`` monkeypatched at the
``nerdit.core.gitwatch`` import site (no git binary, no network) and a
recording ``redeploy`` callable. Drives the controller the way the manager
does — ``await controller.reconcile()`` — then awaits the spawned poll tasks,
because the whole point of the design is that the tick itself does nothing.

The pins that matter: an **ineligible** service is never redeployed (the
D-P24-8 gate), the poll never runs on the tick, failures back off, and the
recorded ``token_ref`` is resolved without writing an audit row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pytest

from nerdit.config.settings import GitSettings
from nerdit.core import gitwatch as gitwatch_mod
from nerdit.core.eventlog import EventRecorder
from nerdit.core.gitsource import GitSourceError
from nerdit.core.gitwatch import GitWatchController
from nerdit.core.secrets import SecretManager
from nerdit.daemon.deploy_pipeline import github_token_absent_error
from nerdit.daemon.errors import NerditError
from nerdit.db.models import Job, JobKind, JobStatus

OLD_SHA = "a" * 40
NEW_SHA = "b" * 40
FIX_SHA = "c" * 40

# --- helpers ------------------------------------------------------------------


def _source(**overrides) -> dict:
    return {
        "type": "git",
        "repo_url": "https://github.com/o/r",
        "ref": "main",
        "commit_sha": OLD_SHA,
        **overrides,
    }


def _job(
    name: str = "svc-a",
    *,
    status: JobStatus = JobStatus.running,
    auto_deploy: bool | None = True,
    source: dict | None = None,
    **cfg_extra,
) -> Job:
    config: dict = {"image": "nerdit-app/svc-a:1", "port": 8000, "build_version": 1}
    if auto_deploy is not None:
        config["auto_deploy"] = auto_deploy
    if source is not None:
        config["source"] = source
    config.update(cfg_extra)
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state="running",
        config=json.dumps(config),
    )


class _FakeLsRemote:
    """Records every probe; returns a sha, raises, or sleeps."""

    def __init__(self, *, sha: str = OLD_SHA, error: Exception | None = None, delay: float = 0.0):
        self.sha = sha
        self.error = error
        self.delay = delay
        self.calls: list[dict] = []

    async def __call__(self, repo_url, **kwargs):
        self.calls.append({"repo_url": repo_url, **kwargs})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.sha


class _Redeploys:
    """A recording stand-in for WP6's ``redeploy_from_source``."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.jobs: list[Job] = []

    async def __call__(self, job: Job) -> None:
        self.jobs.append(job)
        if self.error is not None:
            raise self.error


def _controller(
    queries,
    tmp_path: Path,
    *,
    redeploy: _Redeploys | None = None,
    skip_reason: str | None = None,
    events: EventRecorder | None = None,
    secrets: SecretManager | None = None,
    **git_kw,
) -> GitWatchController:
    return GitWatchController(
        queries,
        GitSettings(**git_kw),
        secrets or SecretManager(tmp_path / "secrets"),
        redeploy=redeploy or _Redeploys(),
        skip_reason=lambda job, cfg: skip_reason,
        events=events if events is not None else EventRecorder(queries),
    )


async def _tick(controller: GitWatchController) -> None:
    """One reconcile plus the poll tasks it spawned."""
    await controller.reconcile()
    tasks = list(controller._inflight.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _events(queries, type_: str | None = None) -> list:
    rows, _ = await queries.list_events(limit=100)
    return [r for r in rows if type_ is None or r.type == type_]


async def _audit(queries, action: str | None = None) -> list:
    rows, _ = await queries.list_audit_log(limit=100)
    return [r for r in rows if action is None or r.action == action]


@pytest.fixture
def fake_ls(monkeypatch):
    """Install a fake probe; the test mutates it before the tick."""
    fake = _FakeLsRemote()
    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", fake)
    return fake


# --- drift --------------------------------------------------------------------


async def test_no_drift_does_not_redeploy(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source()))
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy)

    await _tick(controller)

    assert len(fake_ls.calls) == 1
    assert redeploy.jobs == []
    assert await _events(queries) == []


async def test_drift_audits_emits_then_redeploys(queries, tmp_path, fake_ls):
    job = _job(source=_source())
    await queries.create_job(job)
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy)

    await _tick(controller)

    assert [j.id for j in redeploy.jobs] == [job.id]

    rows = await _audit(queries, "deploy.auto_redeploy")
    assert len(rows) == 1
    assert rows[0].principal_id == "system"
    assert rows[0].principal_role == "system"
    assert rows[0].target_id == job.id
    assert rows[0].params_redacted == {
        "from_sha": OLD_SHA,
        "ref": "main",
        "service": "svc-a",
        "to_sha": NEW_SHA,
    }

    events = await _events(queries, "gitwatch.redeploy_triggered")
    assert len(events) == 1
    assert events[0].service_name == "svc-a"
    assert events[0].data == {"from_sha": OLD_SHA, "to_sha": NEW_SHA}


async def test_config_is_read_fresh_not_from_the_tick_snapshot(queries, tmp_path, fake_ls):
    """An operator redeploy landing between the tick and the task must be seen.

    A queued probe can wait a whole ``clone_timeout_s`` behind the semaphore; a
    poll comparing against the tick-time snapshot would see phantom drift and
    fire a duplicate auto-redeploy of the identical commit. Driven through
    ``_poll_once`` with a deliberately stale snapshot — the race made explicit."""
    job = _job(source=_source(commit_sha=NEW_SHA))
    await queries.create_job(job)
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy)

    stale = {
        "id": job.id,
        "kind": "service",
        "service_name": "svc-a",
        "status": "running",
        "desired_state": "running",
        "config": {**json.loads(job.config or "{}"), "source": _source()},
    }
    await controller._poll_once(stale, "svc-a")

    assert redeploy.jobs == []
    assert await _audit(queries, "deploy.auto_redeploy") == []
    assert await _events(queries) == []


async def test_opt_out_between_the_tick_and_the_task_costs_zero_egress(queries, tmp_path, fake_ls):
    """Candidacy is re-applied to the fresh row BEFORE the remote is touched.

    The tick only snapshots, and the task can sit behind the semaphore for a
    whole ``clone_timeout_s``. An ``auto_deploy = false`` write landing in that
    window must win over the snapshot — no probe, no redeploy, nothing said."""
    job = _job(source=_source())
    await queries.create_job(job)
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy)

    stale = {
        "id": job.id,
        "kind": "service",
        "service_name": "svc-a",
        "status": "running",
        "desired_state": "running",
        "config": json.loads(job.config or "{}"),
    }
    await queries.update_job_config(
        job.id, json.dumps({**json.loads(job.config or "{}"), "auto_deploy": False})
    )

    await controller._poll_once(stale, "svc-a")

    assert fake_ls.calls == []
    assert redeploy.jobs == []
    assert await _audit(queries, "deploy.auto_redeploy") == []
    assert await _events(queries) == []
    # Re-armed rather than left due: a revived service is not polled instantly
    # on every subsequent tick.
    assert "svc-a" in controller._next_due


@pytest.mark.parametrize("landing", ["manual_redeploy", "opt_out", "source_rewrite"])
async def test_a_write_landing_during_the_probe_wins(queries, tmp_path, monkeypatch, landing):
    """The probe is a network round trip, so the pre-probe read is already stale.

    A manual redeploy landing inside it has already shipped the SHA the poller
    is about to "discover": acting on the pre-probe snapshot would fire a
    duplicate auto-redeploy of that identical commit. An opt-out landing there
    must likewise beat a decision taken before it. And a deploy that rewrote
    the SOURCE itself (new repo/ref) makes the probed SHA foreign: comparing
    it against the new source's commit_sha would trigger a redeploy off a
    probe of a repo the row no longer points at, and record a to_sha that
    repo never had."""
    job = _job(source=_source())
    await queries.create_job(job)
    redeploy = _Redeploys()

    async def _probe(repo_url, **kwargs):
        config = json.loads(job.config or "{}")
        if landing == "manual_redeploy":
            config["source"] = _source(commit_sha=NEW_SHA)
        elif landing == "source_rewrite":
            config["source"] = _source(repo_url="https://github.com/o/other", commit_sha="d" * 40)
        else:
            config["auto_deploy"] = False
        await queries.update_job_config(job.id, json.dumps(config))
        return NEW_SHA

    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", _probe)
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    await _tick(controller)

    assert redeploy.jobs == []
    assert await _audit(queries, "deploy.auto_redeploy") == []
    assert await _events(queries) == []
    # A no-op outcome, not a failure: the plain interval, no backoff.
    assert "svc-a" not in controller._failures
    assert controller._next_due["svc-a"] - time.monotonic() <= 60.0


async def test_probe_carries_the_recorded_ref_and_git_settings(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source(ref="v2")))
    controller = _controller(
        queries, tmp_path, allowed_hosts=["github.com", "gitlab.com"], clone_timeout_s=45
    )

    await _tick(controller)

    call = fake_ls.calls[0]
    assert call["repo_url"] == "https://github.com/o/r"
    assert call["ref"] == "v2"
    assert call["token"] is None
    assert call["timeout_s"] == 45.0
    assert call["allowed_hosts"] == ["github.com", "gitlab.com"]


# --- the cutover gate (D-P24-8) -----------------------------------------------


async def test_ineligible_service_is_never_probed_or_redeployed(queries, tmp_path, fake_ls):
    """The pin: no cutover protection ⇒ no unattended redeploy, and the feed
    says why. The gate runs BEFORE the remote touch, so nothing is even probed."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, skip_reason="proxy_off")

    await _tick(controller)

    assert fake_ls.calls == []
    assert redeploy.jobs == []
    events = await _events(queries, "gitwatch.skipped_no_cutover")
    assert len(events) == 1
    assert events[0].reason == "proxy_off"
    assert await _audit(queries, "deploy.auto_redeploy") == []


async def test_skip_event_is_rate_limited_to_once_an_hour(queries, tmp_path, fake_ls, monkeypatch):
    await queries.create_job(_job(source=_source()))
    controller = _controller(
        queries, tmp_path, skip_reason="gpu_bound", events=EventRecorder(queries)
    )

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)

    # Past the poll interval but inside the hour: due again, still silent.
    now += 120.0
    await _tick(controller)
    assert len(await _events(queries, "gitwatch.skipped_no_cutover")) == 1

    now += gitwatch_mod._SKIP_EMIT_INTERVAL_S + 1.0
    await _tick(controller)
    assert len(await _events(queries, "gitwatch.skipped_no_cutover")) == 2


# --- failures -----------------------------------------------------------------


async def test_poll_failure_emits_the_code_and_backs_off(queries, tmp_path, fake_ls, monkeypatch):
    await queries.create_job(_job(source=_source()))
    fake_ls.error = GitSourceError(504, "deploy.git_timeout", "git operation timed out after 120s.")
    controller = _controller(queries, tmp_path, watch_interval_s=60)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)

    events = await _events(queries, "gitwatch.poll_failed")
    assert len(events) == 1
    assert events[0].reason == "deploy.git_timeout"
    # The machine code only — never the git message.
    assert "timed out after" not in json.dumps(events[0].model_dump(mode="json"))
    # First failure ⇒ one doubling: 120 s, not the plain 60 s interval.
    assert controller._next_due["svc-a"] == now + 120.0

    # Still inside the backoff window: the next tick spawns nothing.
    now += 60.0
    await _tick(controller)
    assert len(fake_ls.calls) == 1


async def test_backoff_is_capped_at_thirty_minutes(queries, tmp_path):
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    for _ in range(20):
        controller._backoff("svc-a")
    assert controller._next_due["svc-a"] - time.monotonic() <= gitwatch_mod._POLL_BACKOFF_CAP_S


async def test_redeploy_failure_backs_off_without_a_poll_failed_event(queries, tmp_path, fake_ls):
    """A refused redeploy (a cutover already in flight, say) is not a poll
    failure — the poll worked. Its own machinery records it."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(error=RuntimeError("service.cutover_in_progress"))
    controller = _controller(queries, tmp_path, redeploy=redeploy)

    await _tick(controller)

    assert len(redeploy.jobs) == 1
    assert await _events(queries, "gitwatch.poll_failed") == []
    # A code-less exception is unclassified: it never emits and never pins.
    assert await _events(queries, "gitwatch.redeploy_failed") == []
    assert controller._failures["svc-a"] == 1


async def test_active_run_refusal_leaves_the_generation_alone(queries, tmp_path, fake_ls):
    """A push landing during a P20 run/release is refused by the primitive.

    The row stays ``running`` throughout a run, so it stays a candidate — the
    409 is what keeps the unattended path from bumping the generation the run
    owns. The poller records the decision it took (one pre-recorded trigger)
    and backs off; nothing further is written."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(
        error=NerditError(409, "service.run_in_progress", "Service 'svc-a' has an active run.")
    )
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    await _tick(controller)

    assert len(await _audit(queries, "deploy.auto_redeploy")) == 1
    assert len(await _events(queries, "gitwatch.redeploy_triggered")) == 1
    assert await _events(queries, "gitwatch.poll_failed") == []
    # A 409 is contention, not a bad commit: nothing durable, nothing pinned.
    assert await _events(queries, "gitwatch.redeploy_failed") == []
    assert controller._failed_sha == {}
    assert controller._failures["svc-a"] == 1
    # Backoff armed, not the plain interval: the doubled 120 s.
    assert controller._next_due["svc-a"] - time.monotonic() > 60.0


async def test_permanent_redeploy_failure_emits_once_and_pins_the_sha(
    queries, tmp_path, fake_ls, monkeypatch
):
    """A commit that cannot build is a bad commit, not a transient refusal.

    One durable signal carrying the machine code, then the SHA is suppressed:
    an unattended loop must never re-trigger the same doomed build forever."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(error=NerditError(422, "deploy.no_buildpack", "No buildpack matched."))
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)

    events = await _events(queries, "gitwatch.redeploy_failed")
    assert len(events) == 1
    assert events[0].reason == "deploy.no_buildpack"
    assert events[0].data == {"to_sha": NEW_SHA}
    # The machine code only — never the error message.
    assert "No buildpack" not in json.dumps(events[0].model_dump(mode="json"))
    assert controller._failed_sha["svc-a"] == NEW_SHA
    assert len(await _audit(queries, "deploy.auto_redeploy")) == 1
    assert len(await _events(queries, "gitwatch.redeploy_triggered")) == 1
    # The plain interval, not a backoff: the fix commit must be seen promptly.
    assert controller._next_due["svc-a"] == now + 60.0

    now += 61.0
    await _tick(controller)

    assert len(redeploy.jobs) == 1
    assert len(await _events(queries, "gitwatch.redeploy_failed")) == 1
    assert len(await _events(queries, "gitwatch.redeploy_triggered")) == 1
    assert len(await _audit(queries, "deploy.auto_redeploy")) == 1


async def test_new_sha_after_a_pinned_failure_retriggers(queries, tmp_path, fake_ls, monkeypatch):
    """The pin is per-commit: the moment the remote moves, the poller fires again."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(error=NerditError(422, "deploy.no_buildpack", "No buildpack matched."))
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)
    assert controller._failed_sha["svc-a"] == NEW_SHA

    fake_ls.sha = FIX_SHA
    redeploy.error = None
    now += 61.0
    await _tick(controller)

    assert len(redeploy.jobs) == 2
    assert controller._failed_sha == {}
    assert len(await _events(queries, "gitwatch.redeploy_failed")) == 1


async def test_409_refusal_is_not_pinned(queries, tmp_path, fake_ls):
    """Contention is not a verdict on the commit: back off, keep the same SHA.

    Pinning here would permanently drop the auto-deploy of a good commit that
    happened to land during a run, a release or a cutover."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(
        error=NerditError(409, "service.cutover_in_progress", "A cutover is in flight.")
    )
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    await _tick(controller)

    assert await _events(queries, "gitwatch.redeploy_failed") == []
    assert controller._failed_sha == {}
    assert controller._failures["svc-a"] == 1
    assert controller._next_due["svc-a"] - time.monotonic() > 60.0

    redeploy.error = None
    controller._next_due["svc-a"] = 0.0
    controller._scan_due = 0.0
    await _tick(controller)

    assert len(redeploy.jobs) == 2


# --- candidate filtering ------------------------------------------------------


@pytest.mark.parametrize(
    "job",
    [
        _job("no-flag", auto_deploy=None, source=_source()),
        _job("off-flag", auto_deploy=False, source=_source()),
        _job("zip-row", source={"type": "zip"}),
        _job("no-source"),
        _job("stopped-row", status=JobStatus.stopped, source=_source()),
        _job("failed-row", status=JobStatus.failed, source=_source()),
    ],
)
async def test_non_candidates_are_never_polled(queries, tmp_path, fake_ls, job):
    await queries.create_job(job)
    controller = _controller(queries, tmp_path)

    await _tick(controller)

    assert fake_ls.calls == []


async def test_degraded_row_is_still_polled(queries, tmp_path, fake_ls):
    await queries.create_job(_job(status=JobStatus.degraded, source=_source()))
    controller = _controller(queries, tmp_path)

    await _tick(controller)

    assert len(fake_ls.calls) == 1


async def test_state_is_pruned_when_a_service_stops_being_a_candidate(queries, tmp_path, fake_ls):
    job = _job(source=_source())
    await queries.create_job(job)
    controller = _controller(queries, tmp_path)
    await _tick(controller)
    assert "svc-a" in controller._next_due

    await queries.update_job_status(job.id, JobStatus.stopped)
    controller._scan_due = 0.0  # real clock: force the scan the gate deferred
    await controller.reconcile()
    assert controller._next_due == {}
    assert controller._failed_sha == {}


# --- off-tick -----------------------------------------------------------------


async def test_reconcile_returns_before_the_probe_completes(queries, tmp_path, fake_ls):
    """The tick is the shared 2 s loop: a network round trip may never sit in it."""
    await queries.create_job(_job(source=_source()))
    fake_ls.delay = 0.5
    controller = _controller(queries, tmp_path)

    started = time.perf_counter()
    await controller.reconcile()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.01
    assert controller._inflight
    await asyncio.gather(*controller._inflight.values())
    assert len(fake_ls.calls) == 1


async def test_a_service_is_single_flight(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source()))
    fake_ls.delay = 0.2
    controller = _controller(queries, tmp_path)

    await controller.reconcile()
    await controller.reconcile()  # the first poll is still running

    tasks = list(controller._inflight.values())
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    assert len(fake_ls.calls) == 1


async def test_reconcile_skips_the_scan_until_a_poll_is_due(
    queries, tmp_path, fake_ls, monkeypatch
):
    """The candidate scan is a full workload read: once per interval, not per tick."""
    await queries.create_job(_job(source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)

    scans = 0
    list_configs = queries.list_workload_configs

    async def _counted():
        nonlocal scans
        scans += 1
        return await list_configs()

    monkeypatch.setattr(queries, "list_workload_configs", _counted)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)
    assert scans == 1
    assert len(fake_ls.calls) == 1

    # Shared-tick cadence: many reconciles, no DB read.
    now += 2.0
    await _tick(controller)
    now += 28.0
    await _tick(controller)
    assert scans == 1

    now += 31.0
    await _tick(controller)
    assert scans == 2
    assert len(fake_ls.calls) == 2


async def test_newly_enabled_service_is_discovered_within_the_interval(
    queries, tmp_path, fake_ls, monkeypatch
):
    """The gate's accepted bound: discovery costs at most one poll interval."""
    controller = _controller(queries, tmp_path, watch_interval_s=60)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)
    assert fake_ls.calls == []

    await queries.create_job(_job(source=_source()))

    now += 2.0
    await _tick(controller)
    assert fake_ls.calls == []

    now += 59.0
    await _tick(controller)
    assert len(fake_ls.calls) == 1


async def test_shutdown_cancels_an_in_flight_poll(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source()))
    fake_ls.delay = 30.0
    controller = _controller(queries, tmp_path)

    await controller.reconcile()
    task = next(iter(controller._inflight.values()))

    await controller.shutdown()

    assert task.cancelled() or task.done()
    assert controller._inflight == {}


# --- token custody (D-BP-3: quiet resolution) ---------------------------------


async def test_recorded_token_is_resolved_without_an_audit_row(queries, tmp_path, fake_ls):
    """The poll repeats every minute; auditing it would bury the real signal.
    The audited resolution still happens inside the redeploy itself."""
    sentinel = "ghp-zqxjkw-ZQXJKW"  # non-hex alphabet: greppable, never a real token
    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"GH_TOKEN": sentinel})
    await queries.create_job(_job(source=_source(token_ref="${secrets.shared.GH_TOKEN}")))
    controller = _controller(queries, tmp_path, secrets=secrets)

    await _tick(controller)

    assert fake_ls.calls[0]["token"] == sentinel
    assert await _audit(queries, "secret.shared_referenced") == []
    # The value reaches the probe and nothing else.
    assert sentinel not in json.dumps([r.model_dump(mode="json") for r in await _audit(queries)])
    assert sentinel not in json.dumps([e.model_dump(mode="json") for e in await _events(queries)])


async def test_missing_secret_is_a_credential_poll_failure(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source(token_ref="${secrets.shared.ABSENT}")))
    controller = _controller(queries, tmp_path)

    await _tick(controller)

    assert fake_ls.calls == []
    events = await _events(queries, "gitwatch.poll_failed")
    assert len(events) == 1
    assert events[0].reason == "deploy.no_source_credential"


async def test_unparseable_token_ref_is_a_credential_poll_failure(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source(token_ref="not-a-reference")))
    controller = _controller(queries, tmp_path)

    await _tick(controller)

    assert fake_ls.calls == []
    events = await _events(queries, "gitwatch.poll_failed")
    assert [e.reason for e in events] == ["deploy.no_source_credential"]


# --- P33 WP-R: ${github.installation} (D-GH-3 / D-GH-9) -----------------------


class _GithubTokens:
    """A recording stand-in for ``LinkManager.github_token_for_repo``."""

    def __init__(self, tokens: dict[str, str] | None = None) -> None:
        self.tokens = tokens or {}
        self.calls: list[str] = []

    def __call__(self, slug: str) -> str | None:
        self.calls.append(slug)
        return self.tokens.get(slug)


def _gh_controller(queries, tmp_path, github: _GithubTokens | None, **kw) -> GitWatchController:
    return GitWatchController(
        queries,
        GitSettings(watch_interval_s=60),
        SecretManager(tmp_path / "secrets"),
        redeploy=kw.pop("redeploy", None) or _Redeploys(),
        skip_reason=lambda job, cfg: None,
        events=EventRecorder(queries),
        github_token=github,
    )


async def test_github_installation_token_is_resolved_by_repo(queries, tmp_path, fake_ls):
    sentinel = "ghs-zqxjkw-ZQXJKW"  # non-hex alphabet: greppable, never a real token
    github = _GithubTokens({"o/r": sentinel})
    await queries.create_job(
        _job(
            source=_source(
                repo_url="https://github.com/O/R.git", token_ref="${github.installation}"
            )
        )
    )
    controller = _gh_controller(queries, tmp_path, github)

    await _tick(controller)

    # Looked up by the canonical owner/name, never by the raw URL.
    assert github.calls == ["o/r"]
    assert fake_ls.calls[0]["token"] == sentinel
    # The value reaches the probe and nothing else.
    assert sentinel not in json.dumps([r.model_dump(mode="json") for r in await _audit(queries)])
    assert sentinel not in json.dumps([e.model_dump(mode="json") for e in await _events(queries)])


async def test_github_token_absent_is_a_quiet_bounded_backoff(
    queries, tmp_path, fake_ls, monkeypatch
):
    """D-GH-9: a cold reconnect / expiry / no cloud at all is NOT a failed poll:
    no ``poll_failed`` event, no audit row, just the bounded backoff. The
    doctor row is the only surface."""
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    controller = _gh_controller(queries, tmp_path, _GithubTokens({}))

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)

    assert fake_ls.calls == []
    assert await _events(queries) == []
    assert await _audit(queries) == []
    # Backed off (one doubling), not the plain interval and not a permanent pin.
    assert controller._next_due["svc-a"] == now + 120.0
    assert "svc-a" not in controller._failed_sha
    # The backoff stays bounded.
    for _ in range(20):
        controller._backoff("svc-a")
    assert controller._next_due["svc-a"] - now <= gitwatch_mod._POLL_BACKOFF_CAP_S


async def test_github_token_absent_without_any_resolver_is_quiet(queries, tmp_path, fake_ls):
    """No link manager wired at all (link disabled, no cloud): same quiet skip."""
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    controller = _gh_controller(queries, tmp_path, None)

    await _tick(controller)

    assert fake_ls.calls == []
    assert await _events(queries) == []
    assert await _audit(queries) == []
    assert controller._failures["svc-a"] == 1


async def test_github_token_is_never_offered_to_a_non_github_host(queries, tmp_path, fake_ls):
    github = _GithubTokens({"o/r": "ghs-x"})
    await queries.create_job(
        _job(source=_source(repo_url="https://gitlab.com/o/r", token_ref="${github.installation}"))
    )
    controller = _gh_controller(queries, tmp_path, github)

    await _tick(controller)

    assert github.calls == []
    assert fake_ls.calls == []
    assert await _events(queries) == []


async def test_token_arriving_later_unblocks_the_poll(queries, tmp_path, fake_ls, monkeypatch):
    github = _GithubTokens({})
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    controller = _gh_controller(queries, tmp_path, github)

    now = 1_000.0
    monkeypatch.setattr(gitwatch_mod.time, "monotonic", lambda: now)
    await _tick(controller)
    assert fake_ls.calls == []

    github.tokens["o/r"] = "ghs-later"
    now += 200.0  # past the one-doubling backoff
    await _tick(controller)
    assert fake_ls.calls[0]["token"] == "ghs-later"
    assert "svc-a" not in controller._failures


async def test_secret_refs_never_consult_the_github_resolver(queries, tmp_path, fake_ls):
    """The no-cloud path is byte-identical for ``${secrets.*}``: with and
    without a resolver wired, the same secret reaches the probe and the
    resolver is never asked."""
    sentinel = "ghp-zqxjkw-ZQXJKW"
    secrets = SecretManager(tmp_path / "secrets")
    secrets.set("_shared", {"GH_TOKEN": sentinel})
    await queries.create_job(_job(source=_source(token_ref="${secrets.shared.GH_TOKEN}")))
    github = _GithubTokens({"o/r": "ghs-must-not-be-used"})

    with_cloud = GitWatchController(
        queries,
        GitSettings(watch_interval_s=60),
        secrets,
        redeploy=_Redeploys(),
        skip_reason=lambda job, cfg: None,
        events=EventRecorder(queries),
        github_token=github,
    )
    await _tick(with_cloud)
    assert github.calls == []
    assert fake_ls.calls[-1]["token"] == sentinel

    without_cloud = GitWatchController(
        queries,
        GitSettings(watch_interval_s=60),
        secrets,
        redeploy=_Redeploys(),
        skip_reason=lambda job, cfg: None,
        events=EventRecorder(queries),
    )
    await _tick(without_cloud)
    assert fake_ls.calls[-1] == fake_ls.calls[-2]
    assert await _audit(queries, "secret.shared_referenced") == []


async def test_github_token_absent_mid_redeploy_backs_off_without_pinning(
    queries, tmp_path, fake_ls
):
    """The token expired between the probe and the clone: the redeploy's
    ``422 deploy.github_token_absent`` is transient like a 409 — no
    ``redeploy_failed`` event, the SHA is not pinned, the poll backs off."""
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(error=NerditError(422, "deploy.github_token_absent", "absent"))
    controller = _gh_controller(
        queries, tmp_path, _GithubTokens({"o/r": "ghs-x"}), redeploy=redeploy
    )

    await _tick(controller)

    assert len(redeploy.jobs) == 1
    assert await _events(queries, "gitwatch.redeploy_failed") == []
    assert await _events(queries, "gitwatch.poll_failed") == []
    assert "svc-a" not in controller._failed_sha
    assert controller._failures["svc-a"] == 1


async def test_poller_backoff_never_renders_hint(queries, tmp_path, fake_ls, caplog):
    """Missing repository credentials back off without repeating setup hints."""
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys(error=github_token_absent_error())
    controller = _gh_controller(
        queries, tmp_path, _GithubTokens({"o/r": "ghs-x"}), redeploy=redeploy
    )

    with caplog.at_level(logging.DEBUG, logger="nerdit"):
        await _tick(controller)

    assert len(redeploy.jobs) == 1
    # The quiet backoff of D-GH-9 is unchanged: the redeploy was attempted (so
    # the ordinary trigger rows exist) but the refusal produced no failure
    # surface and no SHA pin.
    assert await _events(queries, "gitwatch.redeploy_failed") == []
    assert await _events(queries, "gitwatch.poll_failed") == []
    assert "svc-a" not in controller._failed_sha
    assert controller._failures["svc-a"] == 1
    # …and neither wording escaped anywhere a human or a log shipper reads.
    written = (
        json.dumps(
            [r.model_dump(mode="json") for r in await _audit(queries)]
            + [e.model_dump(mode="json") for e in await _events(queries)]
        )
        + caplog.text
    )
    assert "does not include GitHub deploys" not in written
    assert "Nerdit GitHub App" not in written


# --- nudge (P33 D-GH-4) ---------------------------------------------------------


async def _settle(controller: GitWatchController) -> None:
    """Await the poll tasks a nudge spawned."""
    tasks = list(controller._inflight.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize(
    ("repo", "ref"),
    [
        ("o/r", "main"),
        ("O/R", "refs/heads/main"),
        ("o/r.git", "main"),
        ("github.com/o/r", "main"),
    ],
)
async def test_nudge_matches_on_the_canonical_identity(queries, tmp_path, fake_ls, repo, ref):
    """Case, ``.git``, ``refs/heads/`` and a host prefix all normalise to the
    D-GH-6 identity the deploy recorded — here the WP-S ``repo`` column."""
    await queries.create_job(_job(source=_source(repo="github.com/o/r")))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    outcome = await controller.nudge(repo, ref, NEW_SHA)
    await _settle(controller)

    assert (outcome.matched, outcome.ignored, outcome.deduped) == (["svc-a"], [], [])
    assert [j.name for j in redeploy.jobs] == ["svc-a"]
    # The nudge fired the SAME primitive: the probe ran, the audit/event pair
    # of a regular poll were written.
    assert len(fake_ls.calls) == 1
    assert len(await _audit(queries, "deploy.auto_redeploy")) == 1
    assert len(await _events(queries, "gitwatch.redeploy_triggered")) == 1


async def test_nudge_derives_the_identity_for_a_pre_d_gh_6_row(queries, tmp_path, fake_ls):
    """A source recorded before ``repo`` existed still matches via ``repo_url``."""
    await queries.create_job(_job(source=_source(repo_url="https://GitHub.com/O/R.git")))
    fake_ls.sha = NEW_SHA
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    assert outcome.matched == ["svc-a"]


@pytest.mark.parametrize(
    ("repo", "ref"),
    [("o/other", "main"), ("o/r", "develop"), ("o/r", "refs/tags/v1")],
)
async def test_nudge_for_another_repo_or_ref_touches_nothing(queries, tmp_path, fake_ls, repo, ref):
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)
    outcome = await controller.nudge(repo, ref, NEW_SHA)
    await _settle(controller)
    assert (outcome.matched, outcome.ignored, outcome.deduped) == ([], [], [])
    assert redeploy.jobs == []
    assert fake_ls.calls == []


async def test_nudge_never_matches_a_source_without_a_recorded_ref(queries, tmp_path, fake_ls):
    """The daemon does not know which branch is the default: no guess."""
    await queries.create_job(_job(source=_source(ref=None)))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    assert outcome.matched == []


async def test_nudge_sha_is_a_hint_the_probe_overrides(queries, tmp_path, fake_ls):
    """The branch moved on between webhook and clone: the probe's head wins
    and is what the trigger records — the nudged sha is never checked out."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = FIX_SHA
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    assert outcome.matched == ["svc-a"]
    (event,) = await _events(queries, "gitwatch.redeploy_triggered")
    assert event.data["to_sha"] == FIX_SHA


async def test_nudge_is_deduped_per_service_and_sha(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    first = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    second = await controller.nudge("o/r", "refs/heads/main", NEW_SHA)
    await _settle(controller)
    third = await controller.nudge("o/r", "main", FIX_SHA)
    await _settle(controller)

    assert first.matched == ["svc-a"]
    assert (second.matched, second.deduped) == ([], ["svc-a"])
    assert third.matched == ["svc-a"]
    # Two distinct shas → two probes; the replay cost nothing.
    assert len(fake_ls.calls) == 2


async def test_nudge_dedupe_set_is_bounded(queries, tmp_path, fake_ls, monkeypatch):
    monkeypatch.setattr(gitwatch_mod, "_NUDGE_DEDUPE_MAX", 3)
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    for i in range(5):
        controller._remember_nudge("svc-a", f"{i:040x}")
    assert len(controller._nudged) == 3
    assert ("svc-a", f"{0:040x}") not in controller._nudged
    assert ("svc-a", f"{4:040x}") in controller._nudged


async def test_nudge_on_an_opted_out_service_is_recorded_not_acted_on(queries, tmp_path, fake_ls):
    """Plan §6 Q2 default: ``gitwatch.nudge_ignored``, no probe, no redeploy."""
    await queries.create_job(_job(auto_deploy=False, source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)

    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)

    assert (outcome.matched, outcome.ignored, outcome.deduped) == ([], ["svc-a"], [])
    assert redeploy.jobs == []
    assert fake_ls.calls == []
    (event,) = await _events(queries, "gitwatch.nudge_ignored")
    assert event.service_name == "svc-a"
    assert event.data == {"to_sha": NEW_SHA}
    # Ignored nudges are not deduped: the operator may flip auto_deploy on.
    again = await controller.nudge("o/r", "main", NEW_SHA)
    assert again.ignored == ["svc-a"]


async def test_nudge_skips_a_service_that_is_not_live(queries, tmp_path, fake_ls):
    await queries.create_job(_job(status=JobStatus.stopped, source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    assert (outcome.matched, outcome.ignored) == ([], [])


async def test_nudge_still_honours_the_cutover_gate(queries, tmp_path, fake_ls):
    """The nudge is the SAME poll: an ineligible service is skipped, not
    force-redeployed, and says why in the feed."""
    await queries.create_job(_job(source=_source()))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(
        queries, tmp_path, redeploy=redeploy, skip_reason="no_health_check", watch_interval_s=60
    )
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    assert outcome.matched == ["svc-a"]
    assert redeploy.jobs == []
    assert fake_ls.calls == []
    assert len(await _events(queries, "gitwatch.skipped_no_cutover")) == 1


async def test_nudge_resets_a_backoff_and_fires_now(queries, tmp_path, fake_ls):
    await queries.create_job(_job(source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    controller._failures["svc-a"] = 5
    controller._next_due["svc-a"] = time.monotonic() + 1800
    controller._scan_due = time.monotonic() + 60
    fake_ls.sha = NEW_SHA

    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)

    assert outcome.matched == ["svc-a"]
    assert len(fake_ls.calls) == 1
    assert "svc-a" not in controller._failures


async def test_nudge_does_not_double_spawn_an_in_flight_poll(queries, tmp_path, monkeypatch):
    """A poll already running keeps running; the nudge is honoured by ONE
    follow-up poll when it finishes (review round: the re-arm used to be
    clobbered by the finished poll's ``_schedule``)."""
    slow = _FakeLsRemote(sha=OLD_SHA, delay=0.2)
    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", slow)
    await queries.create_job(_job(source=_source()))
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)
    await controller.reconcile()
    assert "svc-a" in controller._inflight
    running = controller._inflight["svc-a"]

    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    assert outcome.matched == ["svc-a"]
    assert controller._inflight["svc-a"] is running
    assert controller._next_due["svc-a"] == 0.0
    await running
    # The in-flight probe captured the pre-push head (no drift); the push
    # lands for the follow-up, which the finished poll spawned itself
    # rather than arming +60 s.
    slow.sha = NEW_SHA
    assert "svc-a" in controller._inflight
    assert controller._inflight["svc-a"] is not running
    assert controller._next_due["svc-a"] == 0.0
    await _settle(controller)
    assert len(slow.calls) == 2
    assert [job.service_name for job in redeploy.jobs] == ["svc-a"]
    assert "svc-a" not in controller._renudge
    assert "svc-a" not in controller._inflight
    assert controller._next_due["svc-a"] > time.monotonic() + 30


async def test_nudge_follow_up_survives_a_failed_in_flight_poll(queries, tmp_path, monkeypatch):
    """A poll that fails mid-nudge backs off — and the follow-up still fires now."""
    slow = _FakeLsRemote(error=RuntimeError("boom"), delay=0.2)
    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", slow)
    await queries.create_job(_job(source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    await controller.reconcile()
    running = controller._inflight["svc-a"]
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    assert outcome.matched == ["svc-a"]
    await running
    slow.error = None
    slow.sha = NEW_SHA
    assert "svc-a" not in controller._failures
    assert controller._next_due["svc-a"] == 0.0
    await _settle(controller)
    assert len(slow.calls) == 2


async def test_nudge_during_in_flight_poll_is_coalesced_once(queries, tmp_path, monkeypatch):
    """Two shas nudged while one poll runs cost one follow-up, not two."""
    slow = _FakeLsRemote(sha=OLD_SHA, delay=0.2)
    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", slow)
    await queries.create_job(_job(source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    await controller.reconcile()
    running = controller._inflight["svc-a"]
    assert (await controller.nudge("o/r", "main", NEW_SHA)).matched == ["svc-a"]
    assert (await controller.nudge("o/r", "main", FIX_SHA)).matched == ["svc-a"]
    await running
    slow.sha = FIX_SHA
    await _settle(controller)
    assert len(slow.calls) == 2


async def test_shutdown_does_not_respawn_a_renudged_poll(queries, tmp_path, monkeypatch):
    slow = _FakeLsRemote(sha=OLD_SHA, delay=5.0)
    monkeypatch.setattr(gitwatch_mod, "ls_remote_head", slow)
    await queries.create_job(_job(source=_source()))
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    await controller.reconcile()
    assert (await controller.nudge("o/r", "main", NEW_SHA)).matched == ["svc-a"]
    await controller.shutdown()
    assert controller._inflight == {}
    assert controller._renudge == set()


async def test_nudge_fans_out_across_services_on_the_same_source(queries, tmp_path, fake_ls):
    await queries.create_job(_job("svc-a", source=_source()))
    await queries.create_job(_job("svc-b", source=_source()))
    await queries.create_job(_job("svc-c", auto_deploy=False, source=_source()))
    await queries.create_job(_job("svc-d", source=_source(ref="develop")))
    fake_ls.sha = NEW_SHA
    redeploy = _Redeploys()
    controller = _controller(queries, tmp_path, redeploy=redeploy, watch_interval_s=60)
    outcome = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    assert sorted(outcome.matched) == ["svc-a", "svc-b"]
    assert outcome.ignored == ["svc-c"]
    assert sorted(j.name for j in redeploy.jobs) == ["svc-a", "svc-b"]


async def test_nudge_dedupe_is_not_finalized_until_the_poll_deploys_the_sha(
    queries, tmp_path, fake_ls
):
    """D7: a nudge whose installation token has not landed backs off WITHOUT
    committing the ``(service, sha)`` dedupe, so a later nudge (token now
    mirrored) still deploys the sha — the pair was retried, not burned."""
    github = _GithubTokens({})
    await queries.create_job(_job(source=_source(token_ref="${github.installation}")))
    redeploy = _Redeploys()
    controller = _gh_controller(queries, tmp_path, github, redeploy=redeploy)
    fake_ls.sha = NEW_SHA

    first = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    # Token absent → quiet backoff, nothing deployed, dedupe NOT laid down.
    assert first.matched == ["svc-a"]
    assert redeploy.jobs == []
    assert ("svc-a", NEW_SHA) not in controller._nudged

    # The cloud's token push lands; the same nudge is not deduped and deploys.
    github.tokens["o/r"] = "ghs-later"
    second = await controller.nudge("o/r", "main", NEW_SHA)
    await _settle(controller)
    assert (second.matched, second.deduped) == (["svc-a"], [])
    assert [j.name for j in redeploy.jobs] == ["svc-a"]
    # Only NOW is the pair remembered, so a real redelivery would dedupe.
    assert ("svc-a", NEW_SHA) in controller._nudged


async def test_shutdown_flag_blocks_a_finally_respawn(queries, tmp_path, fake_ls):
    """D5: a poll past its last await runs its ``finally`` with cancelled=False;
    once ``shutdown()`` has begun it must NOT register a fresh follow-up task
    that shutdown has already stopped waiting for — the orphan hazard."""
    job = _job(source=_source())
    await queries.create_job(job)
    controller = _controller(queries, tmp_path, watch_interval_s=60)
    row = {"id": job.id, "service_name": "svc-a"}
    controller._renudge.add("svc-a")
    controller._stopping = True

    # A normal completion (no drift): finally sees pending=True, cancelled=False.
    await controller._poll(row, "svc-a")

    # No follow-up task was spawned, so shutdown cannot orphan one.
    assert controller._inflight == {}
    assert "svc-a" not in controller._renudge
