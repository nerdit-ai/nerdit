"""P33 WP-S — source identity (D-GH-6) and generation provenance (D-GH-10).

Three seams, each pinned here:

* :func:`nerdit.core.gitsource.canonical_repo` / :func:`git_source_meta` — the
  pure ``host/owner/repo`` identity every git ingress records;
* ``_stamp_queued`` — the generation carries ``repo``/``ref``/``sha`` copied
  from the source it was cloned from (null for ZIP/workspace/rollback);
* :func:`nerdit.core.deploy_state.stamp_last_deploy` — the settle-time
  ``remediation_code`` (a rule id only), and the provenance quartet on the
  ``service.deploy_succeeded``/``_failed`` events and the service view.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from nerdit.core import eventlog
from nerdit.core.deploy_state import stamp_last_deploy
from nerdit.core.gitsource import GitSourceInfo, canonical_repo, git_source_meta
from nerdit.daemon.deploy_pipeline import _stamp_queued
from nerdit.daemon.views.service import _service_response
from nerdit.db.models import Job, JobKind, JobStatus

SHA = "a" * 40
REPO_URL = "https://github.com/Acme/App.git"


# ── canonical_repo / git_source_meta ───────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/acme/app",
        "https://GitHub.com/Acme/App",
        "https://github.com/acme/app.git",
        "https://github.com/acme/app/",
        "https://github.com/acme/app.git/",
        "https://github.com:443/acme/app.git",
    ],
)
def test_canonical_repo_normalises_case_suffix_and_slash(url):
    assert canonical_repo(url) == "github.com/acme/app"


def test_canonical_repo_keeps_nested_paths_and_only_strips_a_trailing_dot_git():
    assert canonical_repo("https://gitlab.com/group/sub/app.git") == "gitlab.com/group/sub/app"
    assert (
        canonical_repo("https://github.com/acme/app.github.io") == "github.com/acme/app.github.io"
    )


@pytest.mark.parametrize(
    "url",
    ["https://user:tok@github.com/acme/app.git", "https://tok@github.com/acme/app", "/srv/app"],
)
def test_canonical_repo_refuses_userinfo_and_hostless_urls(url):
    """``validate_repo_url`` already 422s these; the identity helper must never
    quietly fold a credential into a persisted string either."""
    with pytest.raises(ValueError):
        canonical_repo(url)


def test_git_source_meta_records_the_canonical_identity_beside_ref_and_sha(tmp_path):
    info = GitSourceInfo(commit_sha=SHA, resolved_ref="main", context_dir=tmp_path)
    assert git_source_meta(info, REPO_URL) == {
        "type": "git",
        "repo_url": REPO_URL,
        "repo": "github.com/acme/app",
        "ref": "main",
        "commit_sha": SHA,
    }
    full = git_source_meta(
        info, REPO_URL, subdir="svc", token_ref="${secrets.shared.T}", template_id="tpl"
    )
    assert (full["subdir"], full["token_ref"], full["template_id"]) == (
        "svc",
        "${secrets.shared.T}",
        "tpl",
    )
    # Empty/None optionals are omitted, not written as nulls.
    assert "subdir" not in git_source_meta(info, REPO_URL, subdir="")


# ── _stamp_queued provenance ───────────────────────────────────────────────


def _git_source() -> dict:
    return {
        "type": "git",
        "repo_url": REPO_URL,
        "repo": "github.com/acme/app",
        "ref": "main",
        "commit_sha": SHA,
    }


def test_stamp_queued_copies_git_provenance_onto_the_generation():
    config = {"image": "nerdit-app/app:1", "source": _git_source()}
    _stamp_queued(config, version=1, action="create")
    ld = config["last_deploy"]
    assert (ld["repo"], ld["ref"], ld["sha"], ld["remediation_code"]) == (
        "github.com/acme/app",
        "main",
        SHA,
        None,
    )


@pytest.mark.parametrize("source", [{"type": "zip"}, {"type": "workspace"}, None])
def test_stamp_queued_leaves_provenance_null_for_non_git_sources(source):
    config = {"image": "nerdit-app/app:1"}
    if source is not None:
        config["source"] = source
    _stamp_queued(config, version=1, action="create")
    ld = config["last_deploy"]
    assert (ld["repo"], ld["ref"], ld["sha"], ld["remediation_code"]) == (None, None, None, None)


def test_stamp_queued_rollback_does_not_claim_the_carried_source():
    """A rollback runs no clone: the carried ``source`` names the LATEST clone,
    not the image being restored, so the generation must not claim it."""
    config = {"image": "nerdit-app/app:1", "source": _git_source()}
    _stamp_queued(config, version=1, action="rollback")
    ld = config["last_deploy"]
    assert (ld["repo"], ld["ref"], ld["sha"]) == (None, None, None)


# ── settle-time remediation_code + events ──────────────────────────────────


class _Recorder:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    async def record(self, event_type: str, **kw) -> None:
        self.rows.append((event_type, kw))


def _job(name: str, cfg: dict, kind: JobKind = JobKind.service) -> Job:
    return Job(
        name=name,
        kind=kind,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config=json.dumps(cfg),
    )


def _git_cfg(**extra) -> dict:
    cfg = {"image": "nerdit-app/app:3", "build_version": 3, "source": _git_source()}
    _stamp_queued(cfg, version=3, action="redeploy")
    cfg.update(extra)
    return cfg


def _fresh_forensics(**flags) -> dict:
    return {
        "last_exit_code": 1,
        "last_crash_at": (datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
        **flags,
    }


async def test_crash_loop_settle_stamps_the_rule_id_and_the_failed_event_carries_it(queries):
    cfg = _git_cfg(**_fresh_forensics(priv_denied=True))
    job = _job("app", cfg)
    await queries.create_job(job)
    recorder = _Recorder()
    eventlog.set_recorder(recorder)  # type: ignore[arg-type]
    try:
        assert await stamp_last_deploy(
            queries,
            job.id,
            phase="failed",
            reason="crash_loop",
            error_class="USER_ERROR",
            error_message="Restart budget exhausted; last exit_code=1 (chown: permission denied)",
        )
    finally:
        eventlog.set_recorder(None)

    ld = json.loads((await queries.get_job(job.id)).config)["last_deploy"]
    assert ld["remediation_code"] == "image_needs_privileges"
    assert (ld["repo"], ld["ref"], ld["sha"]) == ("github.com/acme/app", "main", SHA)

    assert [t for t, _ in recorder.rows] == ["service.deploy_failed"]
    data = recorder.rows[0][1]["data"]
    assert data == {
        "repo": "github.com/acme/app",
        "ref": "main",
        "sha": SHA,
        "remediation_code": "image_needs_privileges",
        "error_class": "USER_ERROR",
    }
    # The code only — never the detail text or the crash tail.
    assert "permission denied" not in json.dumps(recorder.rows[0][1])


async def test_plain_crash_loop_settles_fix_start_command_and_oom_outranks_it(queries):
    plain = _job("plain", _git_cfg(**_fresh_forensics()))
    oom = _job("oom", _git_cfg(**_fresh_forensics(oom_killed=True, priv_denied=True)))
    await queries.create_job(plain)
    await queries.create_job(oom)
    for job in (plain, oom):
        await stamp_last_deploy(queries, job.id, phase="failed", reason="crash_loop")
    assert (
        json.loads((await queries.get_job(plain.id)).config)["last_deploy"]["remediation_code"]
        == "fix_start_command"
    )
    assert (
        json.loads((await queries.get_job(oom.id)).config)["last_deploy"]["remediation_code"]
        == "raise_memory_limit"
    )


async def test_a_model_row_never_gets_a_service_only_remediation_code(queries):
    """(P33 D1) The settle classifier mirrors only rules 1b/4b/5, but the daemon's
    Rule 1 (fresh ``gpu_oom`` on a model row → ``model.gpu_oom``) OUTRANKS 1b. So
    a vLLM model dying of CUDA OOM (``gpu_oom`` AND cgroup ``oom_killed``) must NOT
    be stamped the service-only ``raise_memory_limit`` at settle — that would
    contradict ``/diagnose``'s ``model.gpu_oom``. Model rows get ``null``; the
    service twin, identical forensics, still gets the mirrored code."""
    model = _job(
        "mdl",
        _git_cfg(**_fresh_forensics(gpu_oom=True, oom_killed=True)),
        kind=JobKind.model,
    )
    service = _job("svc", _git_cfg(**_fresh_forensics(gpu_oom=True, oom_killed=True)))
    await queries.create_job(model)
    await queries.create_job(service)
    for job in (model, service):
        await stamp_last_deploy(queries, job.id, phase="failed", reason="crash_loop")
    assert (
        json.loads((await queries.get_job(model.id)).config)["last_deploy"]["remediation_code"]
        is None
    )
    assert (
        json.loads((await queries.get_job(service.id)).config)["last_deploy"]["remediation_code"]
        == "raise_memory_limit"
    )


async def test_a_build_failure_settles_with_no_remediation_code(queries):
    """No crash-loop rule fires for a build failure — and a PREVIOUS generation's
    stale forensics must not be mistaken for this one's (the freshness gate)."""
    stale = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    cfg = _git_cfg(last_exit_code=1, priv_denied=True, last_crash_at=stale)
    job = _job("build", cfg)
    await queries.create_job(job)
    recorder = _Recorder()
    eventlog.set_recorder(recorder)  # type: ignore[arg-type]
    try:
        await stamp_last_deploy(queries, job.id, phase="failed", reason="build_failed")
    finally:
        eventlog.set_recorder(None)
    ld = json.loads((await queries.get_job(job.id)).config)["last_deploy"]
    assert ld["remediation_code"] is None
    assert recorder.rows[0][1]["data"]["remediation_code"] is None
    assert recorder.rows[0][1]["data"]["sha"] == SHA


async def test_deploy_succeeded_carries_provenance_and_a_null_code(queries):
    job = _job("ok", _git_cfg())
    await queries.create_job(job)
    recorder = _Recorder()
    eventlog.set_recorder(recorder)  # type: ignore[arg-type]
    try:
        await stamp_last_deploy(queries, job.id, phase="healthy")
    finally:
        eventlog.set_recorder(None)
    assert recorder.rows == [
        (
            "service.deploy_succeeded",
            {
                "kind": "service",
                "service_name": "ok",
                "reason": None,
                "build_version": 3,
                "data": {
                    "repo": "github.com/acme/app",
                    "ref": "main",
                    "sha": SHA,
                    "remediation_code": None,
                    "error_class": None,
                },
            },
        )
    ]


async def test_a_pre_p33_generation_settles_with_null_provenance(queries):
    """Rows whose ``last_deploy`` predates the provenance keys still settle and
    emit — with the quartet null, never a KeyError."""
    cfg = {"build_version": 2, "last_deploy": {"version": 2, "phase": "launching"}}
    job = _job("old", cfg)
    await queries.create_job(job)
    recorder = _Recorder()
    eventlog.set_recorder(recorder)  # type: ignore[arg-type]
    try:
        await stamp_last_deploy(queries, job.id, phase="healthy")
    finally:
        eventlog.set_recorder(None)
    assert recorder.rows[0][1]["data"] == {
        "repo": None,
        "ref": None,
        "sha": None,
        "remediation_code": None,
        "error_class": None,
    }


# ── the service view ───────────────────────────────────────────────────────


def test_service_view_exposes_source_repo_and_generation_provenance():
    cfg = _git_cfg()
    cfg["source"]["token_ref"] = "${github.installation}"
    cfg["last_deploy"].update(phase="failed", remediation_code="image_needs_privileges")
    view = _service_response(MagicMock(), _job("app", cfg), [], None)
    assert view.source == {
        "type": "git",
        "repo_url": REPO_URL,
        "repo": "github.com/acme/app",
        "ref": "main",
        "commit_sha": SHA,
    }
    assert view.last_deploy is not None
    assert {k: view.last_deploy[k] for k in ("repo", "ref", "sha", "remediation_code")} == {
        "repo": "github.com/acme/app",
        "ref": "main",
        "sha": SHA,
        "remediation_code": "image_needs_privileges",
    }
    assert "token_ref" not in json.dumps(view.model_dump(mode="json"))
