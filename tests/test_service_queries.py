"""Tests for the P2 service query layer (S2).

Pure-async tests over the in-memory DB (no TestClient), so they run under
pytest-asyncio without the Starlette/aiosqlite cross-loop hazard.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nerdit.daemon.auth import QuotaExceeded, hash_token
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import (
    PortRangeExhausted,
    ProjectOwned,
    ServiceNameClaimed,
    ServiceNameTaken,
)


def _service_job(
    name: str = "svc",
    *,
    gpu_count: int = 0,
    status: JobStatus = JobStatus.running,
    desired_state: str = "running",
    restart_policy: str = "always",
    token: str | None = None,
    **kw: object,
) -> Job:
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=gpu_count,
        status=status,
        desired_state=desired_state,
        restart_policy=restart_policy,
        submitted_by_token=token,
        **kw,
    )


async def _mk_job(queries, job_id: str) -> str:
    """Persist a bare batch row so it can satisfy a foreign-key reference."""
    await queries.create_job(Job(id=job_id, script_path="run.py"))
    return job_id


async def _token(queries, *, max_gpus=None, max_concurrent=None) -> ApiToken:
    return await queries.create_api_token(
        ApiToken(
            name="svc-bot",
            role=TokenRole.submitter,
            token_hash=hash_token("raw-secret"),
            max_gpus=max_gpus,
            max_concurrent_jobs=max_concurrent,
        )
    )


# --- 7-column round-trip ------------------------------------------------------


async def test_service_columns_roundtrip(queries):
    last_exit = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)
    window = datetime(2026, 6, 28, 11, 55, 0, tzinfo=UTC)
    job = _service_job(
        "round-trip",
        restart_policy="on-failure",
        restart_count=2,
        health_check={"path": "/healthz", "interval_s": 5, "unhealthy_threshold": 3},
        last_exit_at=last_exit,
        restart_window_start=window,
    )
    await queries.create_job(job)

    fetched = await queries.get_service_by_name("round-trip")
    assert fetched is not None
    assert fetched.kind is JobKind.service
    assert fetched.desired_state == "running"
    assert fetched.restart_policy == "on-failure"
    assert fetched.restart_count == 2
    assert fetched.health_check == {"path": "/healthz", "interval_s": 5, "unhealthy_threshold": 3}
    assert fetched.service_name == "round-trip"
    assert fetched.last_exit_at == last_exit
    assert fetched.restart_window_start == window


# --- F2-STALE-ERR: redeploy/rollback clears the error columns -----------------


async def test_f2_stale_error_cleared_on_redeploy(queries):
    """A redeploy through update_service_config must NULL error_class/error_message.

    Without the fix the columns survive, and the row-column-precedence reads in
    diagnose_service / _build_wait_response surface a prior generation's error on
    a subsequently-healthy service.
    """
    from nerdit.db.models import ErrorClass

    job = _service_job(
        "stale-err",
        status=JobStatus.failed,
        error_class=ErrorClass.user_error,
        error_message="Build failed: boom",
    )
    await queries.create_job(job)

    fetched = await queries.get_service_by_name("stale-err")
    assert fetched.error_class is ErrorClass.user_error
    assert fetched.error_message == "Build failed: boom"

    await queries.update_service_config(
        job.id,
        "{}",
        status=JobStatus.running,
        desired_state="running",
    )

    healed = await queries.get_service_by_name("stale-err")
    assert healed.error_class is None
    assert healed.error_message is None


async def test_f2_stale_error_cleared_on_guarded_revert(queries):
    """The build-failure revert (revert_service_config_guarded) also clears errors."""
    from nerdit.db.models import ErrorClass

    job = _service_job(
        "stale-err-revert",
        status=JobStatus.failed,
        error_class=ErrorClass.user_error,
        error_message="Build failed: boom",
        config='{"build_version": 3}',
    )
    await queries.create_job(job)

    ok = await queries.revert_service_config_guarded(
        job.id,
        '{"build_version": 3}',
        expect_build_version=3,
        status=JobStatus.running,
        desired_state="running",
    )
    assert ok is True

    healed = await queries.get_service_by_name("stale-err-revert")
    assert healed.error_class is None
    assert healed.error_message is None


# --- list_jobs / list_services kind scoping -----------------------------------


async def test_list_jobs_kind_filter_and_list_services_scoping(queries):
    """WP5: ``list_jobs`` no longer defaults to batch — it lists every kind.

    The batch-only default existed for the deleted ``GET /jobs`` surface. A
    caller that wants one kind now passes ``kind=`` explicitly; ``list_services``
    stays the scoped read the ``/services`` surface is built on.
    """
    batch = Job(name="b", script_path="train.py", kind=JobKind.batch)
    await queries.create_job(batch)
    await queries.create_job(_service_job("a-service"))

    every_kind = await queries.list_jobs()
    assert {j.kind for j in every_kind} == {JobKind.batch, JobKind.service}

    only_batch = await queries.list_jobs(kind=JobKind.batch)
    assert [j.id for j in only_batch] == [batch.id]

    services, _ = await queries.list_services()
    assert [j.service_name for j in services] == ["a-service"]


# --- quota math (CRIT-3) ------------------------------------------------------


async def test_service_quota_counts_building_and_restarting(queries):
    token = await _token(queries, max_concurrent=2)
    await queries.reserve_service_for_token(
        _service_job("s1", status=JobStatus.building, token=token.id)
    )
    await queries.reserve_service_for_token(
        _service_job("s2", status=JobStatus.restarting, token=token.id)
    )
    # building + restarting already fill max_concurrent_jobs=2 → third is denied.
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(
            _service_job("s3", status=JobStatus.running, token=token.id)
        )
    assert exc.value.reason == "max_concurrent_jobs"


async def test_service_quota_gpus_zero_ok_under_zero_gpu_cap(queries):
    token = await _token(queries, max_gpus=0, max_concurrent=5)
    # A 0-GPU service must not trip the max_gpus=0 cap (0 + 0 is not > 0).
    job = await queries.reserve_service_for_token(
        _service_job("zero-gpu", gpu_count=0, token=token.id)
    )
    assert job.gpu_count == 0
    fetched = await queries.get_service_by_name("zero-gpu")
    assert fetched is not None


async def test_service_name_taken(queries):
    # Same token: the project judgment (P40b) passes and the label index refuses.
    await queries.reserve_service_for_token(_service_job("dup", token="tok-a"))
    with pytest.raises(ServiceNameTaken):
        await queries.reserve_service_for_token(_service_job("dup", token="tok-a"))


# --- P39 secret claims --------------------------------------------------------


async def test_mint_secret_claim_is_idempotent(queries):
    assert await queries.mint_secret_claim("pre", "tok-a") is True
    assert await queries.mint_secret_claim("pre", "tok-b") is False
    claim = await queries.get_secret_claim("pre")
    assert claim is not None and claim.token_id == "tok-a"


async def test_reserve_consumes_own_claim(queries):
    await queries.mint_secret_claim("pre", "tok-a")
    await queries.reserve_service_for_token(_service_job("pre", token="tok-a"))
    assert await queries.get_service_by_name("pre") is not None
    assert await queries.get_secret_claim("pre") is None


async def test_reserve_refuses_foreign_claim_and_leaves_no_row(queries):
    await queries.mint_secret_claim("pre", "tok-a")
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_service_job("pre", token="tok-b"))
    assert await queries.get_service_by_name("pre") is None
    claim = await queries.get_secret_claim("pre")
    assert claim is not None and claim.token_id == "tok-a"


async def test_reserve_admin_consumes_foreign_claim(queries):
    await queries.mint_secret_claim("pre", "tok-a")
    await queries.reserve_service_for_token(_service_job("pre", token="tok-b"), admin=True)
    assert await queries.get_service_by_name("pre") is not None
    assert await queries.get_secret_claim("pre") is None


async def test_null_token_claim_is_admin_only(queries):
    """A NULL claimant is foreign to every non-admin, even a tokenless one."""
    await queries.mint_secret_claim("pre", None)
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_service_job("pre", token=None))
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_service_job("pre", token="tok-b"))
    assert await queries.get_service_by_name("pre") is None
    await queries.reserve_service_for_token(_service_job("pre", token=None), admin=True)
    assert await queries.get_service_by_name("pre") is not None
    assert await queries.get_secret_claim("pre") is None


async def test_mint_refuses_a_name_with_a_row(queries):
    """A mint racing a fresh deploy loses to the row, in the same statement."""
    await queries.reserve_service_for_token(_service_job("x", token="tok-a"))
    assert await queries.mint_secret_claim("x", "tok-b") is False
    assert await queries.get_secret_claim("x") is None


async def test_delete_keeping_secrets_reclaims_inside_the_transaction(queries):
    """No window between the row delete and the owner's claim (D-P39-6)."""
    job = await queries.reserve_service_for_token(_service_job("x", token="tok-a"))
    seen: list[bool] = []
    assert await queries.delete_service_checked(job.id, on_secrets_reclaimed=seen.append) == []
    assert seen == [True]
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_service_job("x", token="tok-b"))
    await queries.reserve_service_for_token(_service_job("x", token="tok-a"))
    assert await queries.get_secret_claim("x") is None


async def test_delete_without_the_seam_leaves_no_claim(queries):
    job = await queries.reserve_service_for_token(_service_job("x", token="tok-a"))
    assert await queries.delete_service_checked(job.id) == []
    assert await queries.get_secret_claim("x") is None
    # No claim -- but the project outlives the row (P40b) and still refuses a stranger.
    with pytest.raises(ProjectOwned):
        await queries.reserve_service_for_token(_service_job("x", token="tok-b"))
    await queries.reserve_service_for_token(_service_job("x", token="tok-a"))


async def test_delete_secret_claim_reports_presence(queries):
    await queries.mint_secret_claim("pre", "tok-a")
    assert await queries.delete_secret_claim("pre") is True
    assert await queries.delete_secret_claim("pre") is False
    assert await queries.get_secret_claim("pre") is None


# --- P40a project identity (D-P40-3 / D-P40-5) ---------------------------------


async def _project_rows(queries) -> dict[str, tuple[str, str | None]]:
    cursor = await queries._db.conn.execute("SELECT name, id, submitted_by_token FROM projects")
    return {r[0]: (r[1], r[2]) for r in await cursor.fetchall()}


async def test_create_job_stamps_the_legacy_triple_and_creates_the_project(queries):
    """Both insert paths stamp: ``create_job`` (the fixture path) gives a fresh service
    ``(name, production, web)`` and a ``projects`` row owned by its token.
    """
    job = await queries.create_job(_service_job("asso", token="tok-a"))
    assert job.project_id is not None and job.project_id.startswith("prj_")
    assert (job.environment, job.service) == ("production", "web")
    projects = await _project_rows(queries)
    assert projects == {"asso": (job.project_id, "tok-a")}
    fetched = await queries.get_service_by_name("asso")
    assert fetched is not None
    assert (fetched.project_id, fetched.environment, fetched.service) == (
        job.project_id,
        "production",
        "web",
    )
    assert fetched.project == "asso"  # the LEFT JOIN name (D-P40-7)
    # Every _JOB_SELECT site carries it.
    assert (await queries.get_job(job.id)).project == "asso"
    assert [j.project for j in await queries.list_jobs()] == ["asso"]
    assert [j.project for j in (await queries.list_services())[0]] == ["asso"]
    assert [j.project for j in await queries.get_reconcilable_services()] == ["asso"]


async def test_reserve_stamps_and_finds_an_existing_project(queries):
    """``reserve_service_for_token`` stamps inside its BEGIN IMMEDIATE and adopts a
    ``projects`` row that already carries the name (id stable, owner unchanged).
    """
    fresh = await queries.reserve_service_for_token(_service_job("fresh", token="tok-a"))
    assert fresh.project_id is not None and fresh.project_id.startswith("prj_")
    # `project` is filled on the in-memory row too (P40b): the create routes
    # project the returned row straight back without a re-read.
    assert (fresh.environment, fresh.service, fresh.project) == ("production", "web", "fresh")
    # A pre-existing row (the P40b shape: a project that outlives its services),
    # adopted by its owner's deploy.
    await queries._db.conn.execute(
        "INSERT INTO projects (id, name, submitted_by_token) VALUES (?, ?, ?)",
        ("prj_aaaaaaaaaaaaaaaa", "asso", "tok-a"),
    )
    await queries._db.conn.commit()
    second = await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    assert second.project_id == "prj_aaaaaaaaaaaaaaaa"
    assert (await _project_rows(queries))["asso"] == ("prj_aaaaaaaaaaaaaaaa", "tok-a")
    assert (await queries.get_project("prj_aaaaaaaaaaaaaaaa")).name == "asso"
    assert (await queries.get_project_by_name("asso")).submitted_by_token == "tok-a"
    assert [p.name for p in (await queries.page_projects())[0]] == ["asso", "fresh"]
    assert await queries.get_project("prj_nope") is None


async def test_model_and_database_rows_are_not_stamped(queries):
    await queries.create_job(Job(name="llm", kind=JobKind.model, service_name="llm", gpu_count=0))
    await queries.create_job(Job(name="pg", kind=JobKind.database, service_name="pg", gpu_count=0))
    for name in ("llm", "pg"):
        row = await queries.get_service_by_name(name)
        assert (row.project_id, row.environment, row.service, row.project) == (
            None,
            None,
            None,
            None,
        )
    assert await _project_rows(queries) == {}


async def test_duplicate_id_still_raises_the_bare_integrity_error(queries):
    """The PK error names ``jobs.id`` -- neither ``service_name`` nor ``jobs.project_id``
    (D-P40-3) -- and the fresh ``projects`` row rolls back with the failed INSERT.
    """
    import sqlite3

    await queries.reserve_service_for_token(_service_job("a", id="dup000000001"))
    with pytest.raises(sqlite3.IntegrityError) as exc:
        await queries.reserve_service_for_token(_service_job("b", id="dup000000001"))
    assert not isinstance(exc.value, ServiceNameTaken)
    assert "jobs.id" in str(exc.value)
    assert set(await _project_rows(queries)) == {"a"}  # no orphan "b"
    with pytest.raises(sqlite3.IntegrityError):
        await queries.create_job(_service_job("c", id="dup000000001"))
    assert set(await _project_rows(queries)) == {"a"}  # the create_job path too


async def test_triple_collision_raises_service_name_taken(queries):
    """A row carrying another row's triple under a different label hits
    ``idx_jobs_project_service`` alone -- still a name collision (D-P40-3).
    """
    first = await queries.reserve_service_for_token(_service_job("a"))
    with pytest.raises(ServiceNameTaken):
        # admin: (P40d) a preset project_id is judged, and a tokenless project is
        # NULL-owned, foreign to every non-admin; this test is about the index.
        await queries.reserve_service_for_token(
            _service_job("b", project_id=first.project_id, environment="production", service="web"),
            admin=True,
        )
    assert await queries.get_service_by_name("b") is None


async def test_service_name_taken_after_the_projects_insert_rolls_it_back(queries):
    """A fresh ``projects`` row minted for a label that then collides is rolled back
    with the transaction -- no orphan project outlives a refused insert.
    """
    first = await queries.reserve_service_for_token(_service_job("a"))
    # A row labelled "b" whose project is "a": no project named "b" exists yet.
    await queries.reserve_service_for_token(
        _service_job("b", project_id=first.project_id, environment="production", service="api"),
        admin=True,  # (P40d) preset project_id is judged; the tokenless project is NULL-owned
    )
    assert set(await _project_rows(queries)) == {"a"}
    with pytest.raises(ServiceNameTaken):
        await queries.reserve_service_for_token(_service_job("b"))
    assert set(await _project_rows(queries)) == {"a"}


async def test_delete_of_the_last_service_no_longer_prunes_the_project(queries):
    """(P40b) The P40a prune is gone: the project outlives its services and keeps
    the name for its owner; `delete_project_checked` releases it."""
    job = await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    assert await queries.delete_service_checked(job.id) == []
    assert await _project_rows(queries) == {"asso": (job.project_id, "tok-a")}
    with pytest.raises(ProjectOwned):
        await queries.reserve_service_for_token(_service_job("asso", token="tok-b"))
    again = await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    assert again.project_id == job.project_id  # adopted, id stable


async def test_delete_of_a_model_does_not_touch_projects(queries):
    await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    model = await queries.create_job(
        Job(name="llm", kind=JobKind.model, service_name="llm", gpu_count=0)
    )
    assert await queries.delete_service_checked(model.id) == []
    assert set(await _project_rows(queries)) == {"asso"}


async def test_delete_without_secrets_purge_re_mints_the_claim_and_keeps_the_project(queries):
    """The D-P39-6 re-mint is untouched by P40b; the project stays beside the claim."""
    job = await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    seen: list[bool] = []
    assert await queries.delete_service_checked(job.id, on_secrets_reclaimed=seen.append) == []
    assert seen == [True]
    claim = await queries.get_secret_claim("asso")
    assert claim is not None and claim.token_id == "tok-a"
    assert await _project_rows(queries) == {"asso": (job.project_id, "tok-a")}
    # The claim is judged first (D-P40-5 ordering), then the project.
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_service_job("asso", token="tok-b"))
    again = await queries.reserve_service_for_token(_service_job("asso", token="tok-a"))
    assert again.project_id == job.project_id
    assert await queries.get_secret_claim("asso") is None


# --- port allocator -----------------------------------------------------------


async def test_port_stability_same_service_reuses_host_port(queries):
    j1 = await _mk_job(queries, "job-1")
    j2 = await _mk_job(queries, "job-2")
    ep1 = await queries.acquire_service_port("svc-a", j1, 8000, (9400, 9499))
    assert ep1.host_port == 9400
    # Restart: same service, new job/container_port → SAME host_port, repointed job_id.
    ep2 = await queries.acquire_service_port("svc-a", j2, 9000, (9400, 9499))
    assert ep2.host_port == 9400
    assert ep2.job_id == "job-2"
    assert ep2.container_port == 9000


async def test_distinct_services_get_distinct_ports(queries):
    j_a = await _mk_job(queries, "job-a")
    j_b = await _mk_job(queries, "job-b")
    ep_a = await queries.acquire_service_port("svc-a", j_a, 8000, (9400, 9499))
    ep_b = await queries.acquire_service_port("svc-b", j_b, 8000, (9400, 9499))
    assert ep_a.host_port == 9400
    assert ep_b.host_port == 9401
    assert ep_a.host_port != ep_b.host_port


async def test_port_range_exhausted(queries):
    j_a = await _mk_job(queries, "job-a")
    j_b = await _mk_job(queries, "job-b")
    j_c = await _mk_job(queries, "job-c")
    await queries.acquire_service_port("svc-a", j_a, 8000, (9400, 9401))
    await queries.acquire_service_port("svc-b", j_b, 8000, (9400, 9401))
    with pytest.raises(PortRangeExhausted):
        await queries.acquire_service_port("svc-c", j_c, 8000, (9400, 9401))


async def test_port_allocator_skips_daemon_port(queries):
    j_a = await _mk_job(queries, "job-a")
    # 9321 is the daemon port and must never be handed to a service.
    ep = await queries.acquire_service_port("svc-a", j_a, 8000, (9321, 9322))
    assert ep.host_port == 9322


async def test_port_bindability_fallback_reallocates(queries):
    j1 = await _mk_job(queries, "job-1")
    j2 = await _mk_job(queries, "job-2")
    # First grab is bindable → 9400.
    ep1 = await queries.acquire_service_port(
        "svc-a", j1, 8000, (9400, 9499), is_bindable=lambda _p: True
    )
    assert ep1.host_port == 9400
    # On restart the held 9400 is no longer bindable (foreign process) → CRIT-4
    # fallback drops it and reallocates a fresh, distinct port.
    ep2 = await queries.acquire_service_port(
        "svc-a", j2, 8000, (9400, 9499), is_bindable=lambda p: p != 9400
    )
    assert ep2.host_port != 9400
    assert ep2.host_port == 9401


async def test_delete_service_with_logs_and_allocations(queries):
    """P5-runbook regression: ``job_logs``/``gpu_allocations`` reference
    ``jobs(id)`` and the connection runs with ``PRAGMA foreign_keys=ON``, so
    ``delete_service_checked`` must remove dependent rows first — any service that
    ever logged a line used to die with an IntegrityError on delete."""
    job = _service_job("svc-with-logs")
    await queries.create_job(job)
    await queries.append_log(job.id, "service started")
    await queries.delete_service_checked(job.id)
    assert await queries.get_job(job.id) is None
    assert await queries.get_logs(job.id) == []


async def test_delete_service_never_removes_batch_rows(queries):
    job = Job(id="batch-keep", script_path="run.py")
    await queries.create_job(job)
    await queries.append_log(job.id, "batch log")
    await queries.delete_service_checked(job.id)
    assert await queries.get_job(job.id) is not None
    assert len(await queries.get_logs(job.id)) == 1


async def test_delete_service_checked_passes_submitted_by_token(queries):
    """The atomic re-check hands ``submitted_by_token`` to the checker (M2 seam).

    ``delete_service_checked``'s own SELECT carries the owner column so a
    cross-owner checker can gate on it — the row-delete rolls back when the
    checker returns a match.
    """
    keep = _service_job("keeper", token="tok-owner")
    await queries.create_job(keep)
    target = _service_job("target", token="tok-owner")
    await queries.create_job(target)

    seen: list[dict] = []

    def _checker(rows):  # noqa: ANN001, ANN202
        seen.extend(rows)
        return [r for r in rows if r["id"] == keep.id]  # non-empty → rollback

    deps = await queries.delete_service_checked(target.id, _checker)
    # A non-empty result rolls back: nothing deleted.
    assert deps and deps[0]["id"] == keep.id
    assert await queries.get_job(target.id) is not None
    # Every row handed to the checker carries the owner column.
    assert all("submitted_by_token" in r for r in seen)
    assert {r["submitted_by_token"] for r in seen} == {"tok-owner"}


async def test_release_service_endpoint(queries):
    j_a = await _mk_job(queries, "job-a")
    await queries.acquire_service_port("svc-a", j_a, 8000, (9400, 9499))
    assert await queries.get_service_endpoint("svc-a") is not None
    await queries.release_service_endpoint("svc-a")
    assert await queries.get_service_endpoint("svc-a") is None


# --- desired-state / restart writers -----------------------------------------


async def test_set_desired_state_and_restart_writers(queries):
    job = _service_job("worker")
    await queries.create_job(job)

    await queries.set_desired_state(job.id, "stopped")
    exit_at = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)
    window = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)
    await queries.record_service_exit(job.id, exit_at)
    await queries.bump_restart_count(job.id, 1, window)

    fetched = await queries.get_service_by_name("worker")
    assert fetched is not None
    assert fetched.desired_state == "stopped"
    assert fetched.last_exit_at == exit_at
    assert fetched.restart_count == 1
    assert fetched.restart_window_start == window


# --- reconcilable / placement -------------------------------------------------


async def test_get_reconcilable_services(queries):
    running = _service_job("live", status=JobStatus.running, desired_state="running")
    stopped_done = _service_job("done", status=JobStatus.stopped, desired_state="stopped")
    wants_restart = _service_job("again", status=JobStatus.stopped, desired_state="running")
    await queries.create_job(running)
    await queries.create_job(stopped_done)
    await queries.create_job(wants_restart)

    names = {j.service_name for j in await queries.get_reconcilable_services()}
    assert "live" in names  # not terminal
    assert "again" in names  # desired != status
    assert "done" not in names  # settled (stopped == stopped)


async def test_get_service_placement_gpus_least_loaded_first(queries, sample_gpus):
    # Both GPUs idle → returned least-loaded first (stable id order on a tie).
    gpus = await queries.get_service_placement_gpus()
    assert [g.id for g in gpus] == ["GPU-0000-0001", "GPU-0000-0002"]

    # Share one GPU (non-exclusive) → the still-idle one sorts first.
    await _mk_job(queries, "svc-job")
    await queries.allocate_gpus("svc-job", ["GPU-0000-0001"], exclusive=False)
    gpus = await queries.get_service_placement_gpus()
    assert gpus[0].id == "GPU-0000-0002"
    assert {g.id for g in gpus} == {"GPU-0000-0001", "GPU-0000-0002"}


# --- P13b WP7: list_service_endpoints (GET /routes inventory) ------------------


async def _mk_endpoint(
    queries,
    name: str,
    *,
    kind: JobKind = JobKind.service,
    status: JobStatus = JobStatus.running,
    route: str | None = None,
    container_port: int = 8000,
) -> Job:
    """Persist a service/model row + its durable endpoint, optionally routed."""
    job = Job(
        name=name,
        kind=kind,
        service_name=name,
        gpu_count=0,
        status=status,
        desired_state="running",
    )
    await queries.create_job(job)
    await queries.acquire_service_port(name, job.id, container_port, (9400, 9499))
    if route is not None:  # test `is not None`: "" is a valid routed state
        await queries.set_endpoint_route(name, route)
    return job


async def test_list_service_endpoints_pagination(queries):
    for name in ("svc-a", "svc-b", "svc-c"):
        await _mk_endpoint(queries, name)

    page1, cursor1 = await queries.list_service_endpoints(limit=2)
    assert [e.service_name for e in page1] == ["svc-a", "svc-b"]  # service_name ASC
    assert cursor1 is not None

    page2, cursor2 = await queries.list_service_endpoints(cursor=cursor1, limit=2)
    assert [e.service_name for e in page2] == ["svc-c"]
    assert cursor2 is None


async def test_list_service_endpoints_includes_models_and_terminal(queries):
    # Unlike list_active_service_routes, the inventory read keeps models
    # (loopback-only, route NULL) and terminal rows.
    await _mk_endpoint(queries, "web", status=JobStatus.running, route="/web")
    await _mk_endpoint(queries, "ollama-x", kind=JobKind.model, status=JobStatus.running)
    await _mk_endpoint(queries, "old", status=JobStatus.stopped)

    items, _ = await queries.list_service_endpoints(limit=50)
    by_name = {e.service_name: e for e in items}

    assert set(by_name) == {"web", "ollama-x", "old"}
    assert by_name["ollama-x"].kind is JobKind.model
    assert by_name["ollama-x"].route is None  # models never routed
    assert by_name["old"].status is JobStatus.stopped  # terminal included


async def test_list_service_endpoints_preserves_empty_route_vs_null(queries):
    # route "" (subdomain shape) and NULL (unrouted) must stay distinct.
    await _mk_endpoint(queries, "sub", route="")  # subdomain-routed
    await _mk_endpoint(queries, "off")  # unrouted (NULL)

    items, _ = await queries.list_service_endpoints(limit=50)
    by_name = {e.service_name: e for e in items}

    assert by_name["sub"].route == ""
    assert by_name["sub"].route is not None
    assert by_name["off"].route is None


async def test_list_service_endpoints_bad_cursor_raises(queries):
    with pytest.raises(ValueError):
        await queries.list_service_endpoints(cursor="not!valid!base64")


# --- list_workload_configs (P14 WP-0) ----------------------------------------


async def test_list_workload_configs_projects_service_and_model_rows(queries):
    import json

    await queries.create_job(
        _service_job("web", config=json.dumps({"image": "nerdit-app/web:2", "port": 8000}))
    )
    await queries.create_job(
        Job(
            name="ollama-x",
            kind=JobKind.model,
            service_name="ollama-x",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="running",
            restart_policy="always",
            config=json.dumps({"model": "llama3.1:8b", "backend": "ollama"}),
        )
    )
    # A batch row must be excluded.
    await queries.create_job(Job(id="batch0000001", script_path="run.py"))

    rows = await queries.list_workload_configs()
    by_name = {r["service_name"]: r for r in rows}
    assert set(by_name) == {"web", "ollama-x"}
    assert by_name["web"]["config"]["image"] == "nerdit-app/web:2"
    assert by_name["web"]["kind"] == "service"
    assert by_name["ollama-x"]["kind"] == "model"
    assert by_name["ollama-x"]["config"]["model"] == "llama3.1:8b"
    assert by_name["web"]["status"] == "running"


async def test_list_workload_configs_tolerates_bad_config(queries):
    job = _service_job("broken")
    # Force a malformed config blob past the model validator.
    object.__setattr__(job, "config", "{not json")
    await queries.create_job(job)
    rows = await queries.list_workload_configs()
    row = next(r for r in rows if r["service_name"] == "broken")
    assert row["config"] == {}


# --- (P21 D1) crash-tail delete-and-replace -----------------------------------


async def test_replace_crash_tail_only_touches_crash_rows(queries):
    from nerdit.db.models import LogStream

    job = _service_job("svc-crash")
    await queries.create_job(job)
    await queries.append_log(job.id, "app line", LogStream.stdout)
    await queries.append_log(job.id, "system line", LogStream.system)

    await queries.replace_crash_tail(job.id, ["boom 1", "boom 2"])
    entries = await queries.get_logs(job.id)
    assert [(e.stream, e.message) for e in entries] == [
        (LogStream.stdout, "app line"),
        (LogStream.system, "system line"),
        (LogStream.crash, "boom 1"),
        (LogStream.crash, "boom 2"),
    ]

    # A second capture REPLACES the first; the other streams are untouched.
    await queries.replace_crash_tail(job.id, ["boom 3"])
    entries = await queries.get_logs(job.id)
    assert [(e.stream, e.message) for e in entries] == [
        (LogStream.stdout, "app line"),
        (LogStream.system, "system line"),
        (LogStream.crash, "boom 3"),
    ]


async def test_replace_crash_tail_empty_clears_previous_capture(queries):
    from nerdit.db.models import LogStream

    job = _service_job("svc-crash-empty")
    await queries.create_job(job)
    await queries.replace_crash_tail(job.id, ["boom"])
    await queries.replace_crash_tail(job.id, [])
    assert [e for e in await queries.get_logs(job.id) if e.stream is LogStream.crash] == []


async def test_replace_crash_tail_tail_read_is_oldest_first(queries):
    job = _service_job("svc-crash-order")
    await queries.create_job(job)
    await queries.replace_crash_tail(job.id, ["a", "b", "c"])
    entries = await queries.get_logs(job.id, tail=2)
    assert [e.message for e in entries] == ["b", "c"]


# --- P37 last_dump patch guard -------------------------------------------------


async def test_patch_last_dump_field_skips_a_foreign_run(queries):
    """(D-P37-11) The route's post-slot patch applies only while the blob still
    belongs to the run that made it; a newer run's record is never overwritten."""
    import json

    job = _service_job("pg", config=json.dumps({"image": "postgres:16", "build_version": 3}))
    await queries.create_job(job)
    await queries.set_last_dump(
        job.id, json.dumps({"run_id": "runA", "kind": "dump", "dump": None, "reason": None})
    )

    assert (
        await queries.patch_last_dump_field(
            job.id, field="dump", value="x.tar.gz", expect_run_id="runB"
        )
        is False
    )
    row = await queries.get_job(job.id)
    assert json.loads(row.config)["last_dump"]["dump"] is None

    assert (
        await queries.patch_last_dump_field(
            job.id, field="dump", value="x.tar.gz", expect_run_id="runA"
        )
        is True
    )
    cfg = json.loads((await queries.get_job(job.id)).config)
    assert cfg["last_dump"] == {
        "run_id": "runA",
        "kind": "dump",
        "dump": "x.tar.gz",
        "reason": None,
    }
    assert cfg["image"] == "postgres:16" and cfg["build_version"] == 3

    # No blob at all: nothing matches, nothing is written.
    bare = _service_job("bare", config=json.dumps({"image": "i"}))
    await queries.create_job(bare)
    assert (
        await queries.patch_last_dump_field(
            bare.id, field="reason", value="r", expect_run_id="runA"
        )
        is False
    )
