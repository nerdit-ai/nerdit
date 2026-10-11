"""The `hooks` table and `HookQueries` against a real in-memory database."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from nerdit.core.hooks import HOOK_ID_RE
from nerdit.db.database import Database
from nerdit.db.models import Job, JobKind, JobStatus

SPEC = {
    "sender": "stripe",
    "path": "/webhooks/stripe",
    "header": None,
    "prefix": None,
    "secret_ref": "${secrets.NERDIT_HOOK_STRIPE_SECRET}",
}


def _job(name: str, project_id: str | None = None) -> Job:
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        project_id=project_id,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config='{"image": "demo:latest", "port": 8000}',
    )


async def _add(queries, job, name="stripe", **over):
    return await queries.add_hook(job.service_name, job_id=job.id, name=name, **{**SPEC, **over})


async def test_hooks_table_is_created_on_an_existing_database(tmp_path):
    path = str(tmp_path / "nerdit.db")
    async with aiosqlite.connect(path) as conn:  # a pre-hooks install: some other table only
        await conn.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY)")
        await conn.commit()
    db = Database(path)
    await db.connect()
    try:
        await db.init_schema()
        cursor = await db.conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN ('hooks', 'idx_hooks_live_name')"
        )
        assert {r[0] for r in await cursor.fetchall()} == {"hooks", "idx_hooks_live_name"}
    finally:
        await db.close()


async def test_add_hook_inserted_exists_conflict_no_service(queries):
    job = await queries.create_job(_job("shop"))
    outcome, hook = await _add(queries, job)
    assert outcome == "inserted" and hook is not None
    assert HOOK_ID_RE.fullmatch(hook.hook_id)
    assert (hook.sender, hook.path) == ("stripe", "/webhooks/stripe")

    again, same = await _add(queries, job)
    assert again == "exists" and same is not None and same.hook_id == hook.hook_id

    conflict, _ = await _add(queries, job, path="/other")
    assert conflict == "conflict"
    assert [h.path for h in await queries.list_hooks("shop")] == ["/webhooks/stripe"]

    with pytest.raises(sqlite3.IntegrityError):  # the CHECK backstop
        await _add(queries, job, name="x", sender="nope")
    missing, none = await queries.add_hook(
        "shop", job_id="not-the-job", name="gh", **{**SPEC, "sender": "github"}
    )
    assert (missing, none) == ("no_service", None)


async def test_list_and_remove_bound_to_the_authorized_job_see_no_successor(queries):
    """A route that authorized one job row must not read or revoke a same-name successor's hooks."""
    job = await queries.create_job(_job("shop"))
    _, hook = await _add(queries, job)
    assert [h.hook_id for h in await queries.list_hooks("shop", job_id=job.id)] == [hook.hook_id]
    assert await queries.list_hooks("shop", job_id="former-owner-job") == []
    assert await queries.remove_hook("shop", "stripe", job_id="former-owner-job") is False
    assert await queries.get_live_hook(hook.hook_id, "shop") is not None
    assert await queries.remove_hook("shop", "stripe", job_id=job.id) is True


async def test_remove_hook_revokes_and_a_revoked_id_never_resolves(queries):
    job = await queries.create_job(_job("shop"))
    _, hook = await _add(queries, job)
    assert await queries.get_live_hook(hook.hook_id, "shop") is not None

    assert await queries.remove_hook("shop", "stripe") is True
    assert await queries.remove_hook("shop", "stripe") is False
    assert await queries.get_live_hook(hook.hook_id, "shop") is None
    assert await queries.list_hooks("shop") == []
    cursor = await queries._db.conn.execute(
        "SELECT revoked_at FROM hooks WHERE hook_id = ?", (hook.hook_id,)
    )
    row = await cursor.fetchone()
    assert row is not None and row[0] is not None  # the row stays


async def test_a_removed_name_can_be_added_again_with_a_new_id(queries):
    job = await queries.create_job(_job("shop"))
    _, first = await _add(queries, job)
    await queries.remove_hook("shop", "stripe")
    outcome, second = await _add(queries, job)
    assert outcome == "inserted"
    assert second.hook_id != first.hook_id


async def test_apply_declared_hooks_keeps_the_id_and_updates_fields(queries):
    job = await queries.create_job(_job("shop"))
    await queries.apply_declared_hooks("shop", job.id, {"stripe": SPEC})
    [hook] = await queries.list_hooks("shop")

    await queries.apply_declared_hooks("shop", job.id, {"stripe": {**SPEC, "path": "/v2"}})
    [updated] = await queries.list_hooks("shop")
    assert updated.hook_id == hook.hook_id
    assert updated.path == "/v2"

    await queries.apply_declared_hooks("other", job.id, {"stripe": SPEC})  # job mismatch
    assert await queries.list_hooks("other") == []


async def test_apply_declared_hooks_never_revokes_undeclared_hooks(queries):
    job = await queries.create_job(_job("shop"))
    _, api_hook = await _add(queries, job, name="gh", sender="github")
    await queries.apply_declared_hooks("shop", job.id, {"stripe": SPEC})
    await queries.apply_declared_hooks("shop", job.id, {})
    names = {h.name: h.hook_id for h in await queries.list_hooks("shop")}
    assert set(names) == {"gh", "stripe"}
    assert names["gh"] == api_hook.hook_id


async def test_service_delete_revokes_its_hooks_so_a_same_name_successor_inherits_none(queries):
    job = await queries.create_job(_job("shop"))
    _, hook = await _add(queries, job)
    assert await queries.delete_service_checked(job.id) == []

    successor = await queries.create_job(_job("shop"))
    assert await queries.list_hooks("shop") == []
    assert await queries.get_live_hook(hook.hook_id, "shop") is None
    outcome, fresh = await _add(queries, successor)
    assert outcome == "inserted" and fresh.hook_id != hook.hook_id


async def test_get_live_hook_returns_the_project_id(queries):
    job = await queries.create_job(_job("shop", project_id="prj-1"))
    _, hook = await _add(queries, job)
    found = await queries.get_live_hook(hook.hook_id, "shop")
    assert found is not None
    row, project_id = found
    assert row.hook_id == hook.hook_id
    assert project_id == "prj-1"
    assert await queries.get_live_hook(hook.hook_id, "other") is None
