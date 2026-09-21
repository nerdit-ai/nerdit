"""Query-level tests for the P40b project noun (D-P40-5 judgments, D-P40-17 delete).

Real SQLite through the `queries` fixture, no route: the transaction is the
security boundary, so the refusals are pinned where they are made.
"""

from __future__ import annotations

import pytest

from nerdit.db.models import Job, JobKind, JobStatus
from nerdit.db.queries import (
    ProjectExists,
    ProjectOwned,
    ServiceNameClaimed,
    ServiceNameTaken,
)


def _job(name: str, kind: JobKind = JobKind.service, *, token: str | None) -> Job:
    return Job(
        name=name,
        kind=kind,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        submitted_by_token=token,
    )


async def _rows(queries, table: str, col: str) -> list[str]:
    cursor = await queries._db.conn.execute(f"SELECT {col} FROM {table} ORDER BY {col}")
    return [r[0] for r in await cursor.fetchall()]


# --- rule 1: reserve_service_for_token judges the project ----------------------


@pytest.mark.parametrize("kind", [JobKind.service, JobKind.model, JobKind.database])
async def test_reserve_refuses_a_foreign_project_for_every_kind_and_inserts_nothing(queries, kind):
    await queries.create_project("asso", "tok-a")
    with pytest.raises(ProjectOwned) as exc:
        await queries.reserve_service_for_token(_job("asso", kind, token="tok-b"))
    assert "tok-a" not in str(exc.value) and "tok-b" not in str(exc.value)
    assert await _rows(queries, "jobs", "service_name") == []
    assert await _rows(queries, "projects", "name") == ["asso"]


@pytest.mark.parametrize("kind", [JobKind.service, JobKind.model, JobKind.database])
async def test_reserve_admin_bypasses_a_foreign_project(queries, kind):
    project = await queries.create_project("asso", "tok-a")
    job = await queries.reserve_service_for_token(_job("asso", kind, token="tok-b"), admin=True)
    assert await _rows(queries, "jobs", "service_name") == ["asso"]
    # A service row adopts the existing project; other kinds stay tripleless.
    expected = project.id if kind is JobKind.service else None
    assert job.project_id == expected
    assert (await queries.get_project_by_name("asso")).submitted_by_token == "tok-a"


async def test_reserve_owner_adopts_its_own_project(queries):
    project = await queries.create_project("asso", "tok-a")
    job = await queries.reserve_service_for_token(_job("asso", token="tok-a"))
    assert job.project_id == project.id


async def test_null_owner_project_is_admin_only(queries):
    await queries.create_project("asso", None, admin=True)
    for token in (None, "tok-b"):
        with pytest.raises(ProjectOwned):
            await queries.reserve_service_for_token(_job("asso", token=token))
    assert await _rows(queries, "jobs", "service_name") == []
    await queries.reserve_service_for_token(_job("asso", token=None), admin=True)
    assert await _rows(queries, "jobs", "service_name") == ["asso"]


async def test_claim_is_judged_before_the_project(queries):
    """D-P40-5 ordering: a foreign claim answers `ServiceNameClaimed` even when a
    foreign project also exists (both refuse; the claim message is the P39 one)."""
    await queries.create_project("asso", "tok-a", admin=True)
    await queries.mint_secret_claim("asso", "tok-a")
    with pytest.raises(ServiceNameClaimed):
        await queries.reserve_service_for_token(_job("asso", token="tok-b"))


# --- rule 2: mint_secret_claim refuses a foreign project ------------------------


async def test_mint_refuses_a_name_inside_a_foreign_project(queries):
    await queries.create_project("asso", "tok-a")
    assert await queries.mint_secret_claim("asso", "tok-b") is False
    assert await queries.get_secret_claim("asso") is None
    assert await queries.mint_secret_claim("asso", "tok-a") is True  # the owner may


async def test_mint_null_owner_project_refuses_every_non_admin(queries):
    await queries.create_project("asso", None, admin=True)
    assert await queries.mint_secret_claim("asso", "tok-b") is False
    assert await queries.mint_secret_claim("asso", None) is False
    assert await queries.mint_secret_claim("asso", None, admin=True) is True


async def test_mint_admin_bypasses_a_foreign_project(queries):
    await queries.create_project("asso", "tok-a")
    assert await queries.mint_secret_claim("asso", "tok-admin", admin=True) is True
    claim = await queries.get_secret_claim("asso")
    assert claim is not None and claim.token_id == "tok-admin"


async def test_mint_still_loses_to_a_row_regardless_of_admin(queries):
    await queries.reserve_service_for_token(_job("asso", token="tok-a"))
    assert await queries.mint_secret_claim("asso", "tok-a", admin=True) is False


# --- rule 3: create_project refuses a foreign claim ------------------------------


async def test_create_project_refused_on_a_foreign_claim_and_inserts_nothing(queries):
    await queries.mint_secret_claim("blog", "tok-b")
    with pytest.raises(ServiceNameClaimed) as exc:
        await queries.create_project("blog", "tok-a")
    assert exc.value.service_name == "blog"  # label == name (D-P40-2)
    assert "tok-b" not in str(exc.value)
    assert await _rows(queries, "projects", "name") == []


async def test_create_project_allowed_on_own_claim_which_survives_until_the_deploy(queries):
    await queries.mint_secret_claim("blog", "tok-a")
    project = await queries.create_project("blog", "tok-a")
    assert project.name == "blog" and project.submitted_by_token == "tok-a"
    assert project.id.startswith("prj_")
    claim = await queries.get_secret_claim("blog")
    assert claim is not None and claim.token_id == "tok-a"
    job = await queries.reserve_service_for_token(_job("blog", token="tok-a"))
    assert job.project_id == project.id
    assert await queries.get_secret_claim("blog") is None  # consumed by the row (D-P39-3)


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database])
async def test_create_project_refused_on_a_foreign_tripleless_row(queries, kind):
    """A live model/database row owns no `projects` row, so UNIQUE(name) is silent:
    the row predicate is what keeps a stranger from squatting the name and
    locking the owner out at its next re-create (rule 1)."""
    await queries.reserve_service_for_token(_job("llm", kind, token="tok-a"))
    with pytest.raises(ServiceNameTaken) as exc:
        await queries.create_project("llm", "tok-b")
    assert "tok-a" not in str(exc.value)
    assert await _rows(queries, "projects", "name") == []
    # The owner may name its own row's project; a later re-create adopts it.
    project = await queries.create_project("llm", "tok-a")
    assert project.submitted_by_token == "tok-a"
    await queries.delete_service_checked((await queries.get_service_by_name("llm")).id)
    await queries.reserve_service_for_token(_job("llm", kind, token="tok-a"))


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database])
async def test_create_project_null_row_owner_is_foreign_and_admin_bypasses(queries, kind):
    await queries.reserve_service_for_token(_job("llm", kind, token=None), admin=True)
    with pytest.raises(ServiceNameTaken):
        await queries.create_project("llm", "tok-b")
    assert (await queries.create_project("llm", "tok-admin", admin=True)).name == "llm"


async def test_create_project_null_claimant_is_foreign_and_admin_bypasses(queries):
    await queries.mint_secret_claim("blog", None)
    with pytest.raises(ServiceNameClaimed):
        await queries.create_project("blog", "tok-a")
    with pytest.raises(ServiceNameClaimed):
        await queries.create_project("blog", None)
    project = await queries.create_project("blog", "tok-admin", admin=True)
    assert (await queries.get_project(project.id)) == project


async def test_create_project_existing_row_raises_project_exists(queries):
    await queries.create_project("asso", "tok-a")
    with pytest.raises(ProjectExists):
        await queries.create_project("asso", "tok-a")
    with pytest.raises(ProjectExists):
        await queries.create_project("asso", "tok-b", admin=True)
    assert await _rows(queries, "projects", "name") == ["asso"]
    # The write lock is released and the connection usable after the refusal.
    await queries.create_project("blog", "tok-a")


async def test_create_project_refuses_a_label_too_long_for_a_dns_label(queries):
    with pytest.raises(ValueError):
        await queries.create_project("a" * 64, "tok-a")


# --- delete_project_checked (D-P40-17) ------------------------------------------


async def test_delete_project_refuses_while_a_service_row_references_it(queries):
    project = await queries.create_project("asso", "tok-a")
    job = await queries.reserve_service_for_token(_job("asso", token="tok-a"))
    assert await queries.delete_project_checked(project.id) == ["asso"]
    assert await queries.get_project(project.id) is not None
    assert await queries.delete_service_checked(job.id) == []
    assert await queries.delete_project_checked(project.id) == []
    assert await queries.get_project(project.id) is None
    assert await queries.delete_project_checked(project.id) is None  # absence, not success
    # The name is free again for anyone.
    await queries.reserve_service_for_token(_job("asso", token="tok-b"))


async def test_delete_project_lists_every_referencing_label(queries):
    first = await queries.reserve_service_for_token(_job("asso", token="tok-a"))
    api = _job("api--asso", token="tok-a")
    api.project_id, api.environment, api.service = first.project_id, "production", "api"
    await queries.reserve_service_for_token(api)
    assert await queries.delete_project_checked(first.project_id) == ["api--asso", "asso"]


# --- P40c: variable flags (D-P40-1) ---------------------------------------------


async def test_variable_flags_upsert_flip_list_and_delete(queries):
    project = await queries.create_project("asso", "tok-a")
    await queries.upsert_variable_flags(project.id, None, ["A", "B"], True)
    await queries.upsert_variable_flags(project.id, "web", ["A"], False)
    # Same key, two scopes: two rows. The '' sentinel keeps the project scope
    # a real PK member, so a re-set flips the flag instead of adding a row.
    await queries.upsert_variable_flags(project.id, None, ["B"], False)

    flags = [(f.service, f.key, f.plain) for f in await queries.list_variable_flags(project.id)]
    assert flags == [("", "A", True), ("", "B", False), ("web", "A", False)]

    assert await queries.delete_variable_flag(project.id, None, "A") is True
    assert await queries.delete_variable_flag(project.id, None, "A") is False
    assert await queries.delete_variable_flag(project.id, "api", "A") is False
    flags = [(f.service, f.key) for f in await queries.list_variable_flags(project.id)]
    assert flags == [("", "B"), ("web", "A")]


async def test_variable_flags_go_with_the_project_row(queries):
    project = await queries.create_project("asso", "tok-a")
    other = await queries.create_project("blog", "tok-a")
    await queries.upsert_variable_flags(project.id, None, ["A"], True)
    await queries.upsert_variable_flags(other.id, None, ["A"], True)
    assert await queries.delete_project_checked(project.id) == []
    assert await queries.list_variable_flags(project.id) == []
    assert [f.key for f in await queries.list_variable_flags(other.id)] == ["A"]
