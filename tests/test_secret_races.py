"""Secret-custody races (P40g): a verdict must not outlive the lock wait it preceded.

Real SQLite and SecretManager; synthetic values only.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from fastapi import FastAPI, Request

from nerdit.core.secrets import SecretManager
from nerdit.daemon.auth import LOCAL, Principal, hash_token
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.projects import list_variables
from nerdit.daemon.routes.secrets import delete_all_secrets
from nerdit.daemon.secret_scope import (
    set_project_values,
    set_secret_values,
    variable_write_lock,
)
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, TokenRole
from nerdit.db.queries import Queries


@pytest.fixture
async def app(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    app = FastAPI()
    app.state.queries = Queries(db)
    app.state.secret_manager = SecretManager(tmp_path)
    yield app
    await db.close()


def _request(app: FastAPI, principal: Principal) -> Request:
    request = Request({"type": "http", "app": app, "headers": []})
    request.state.principal = principal
    return request


async def test_a_write_parked_on_the_lock_never_lands_in_the_next_owners_scope(app):
    requests = {}
    for tid in ("a", "b"):
        await app.state.queries.create_api_token(
            ApiToken(id=tid, name=tid, role=TokenRole.submitter, token_hash=hash_token(tid))
        )
        requests[tid] = _request(app, Principal(token_id=tid, name=tid, role=TokenRole.submitter))
    a, b = requests["a"], requests["b"]
    await set_secret_values(a, "race", {"OLD": "old"})

    lock = variable_write_lock(app)
    await lock.acquire()  # a staging backup holds it for minutes
    tasks = []
    for call in (
        set_secret_values(a, "race", {"PRIVATE": "a-private"}),
        delete_all_secrets(a, "race"),
        set_secret_values(b, "race", {"PUBLIC": "b-value"}),
    ):
        tasks.append(asyncio.create_task(call))
        await asyncio.sleep(0.05)  # each reaches its park point in order
    lock.release()
    await asyncio.gather(*tasks)

    claim = await app.state.queries.get_secret_claim("race")
    assert claim.token_id == "b"
    assert app.state.secret_manager.load("race") == {"PUBLIC": "b-value"}


async def test_a_stale_writer_is_refused_once_the_name_changed_hands(app):
    for tid in ("a", "b"):
        await app.state.queries.create_api_token(
            ApiToken(id=tid, name=tid, role=TokenRole.submitter, token_hash=hash_token(tid))
        )
    a = _request(app, Principal(token_id="a", name="a", role=TokenRole.submitter))
    b = _request(app, Principal(token_id="b", name="b", role=TokenRole.submitter))
    await set_secret_values(a, "race", {"OLD": "old"})
    await delete_all_secrets(a, "race")
    await set_secret_values(b, "race", {"PUBLIC": "b-value"})
    with pytest.raises(NerditError) as exc:
        await set_secret_values(a, "race", {"PRIVATE": "a-private"})
    assert exc.value.status_code == 403


async def test_list_never_pairs_an_old_secret_with_a_new_plain_flag(app):
    request = _request(app, LOCAL)
    queries = app.state.queries

    def allow(_name: str) -> None:
        return None

    await set_project_values(request, "race", {"KEY": "old-secret"}, False, check_new_name=allow)
    reading, written = asyncio.Event(), asyncio.Event()
    original = queries.list_variable_flags

    async def flags(project_id):
        reading.set()  # the file is read; the flip tries to land before the flags
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(written.wait(), 0.3)
        return await original(project_id)

    queries.list_variable_flags = flags
    reader = asyncio.create_task(list_variables(request, "race", service=None))
    await reading.wait()
    queries.list_variable_flags = original

    async def flip():
        await set_project_values(request, "race", {"KEY": "new-public"}, True, check_new_name=allow)
        written.set()

    writer = asyncio.create_task(flip())
    result = await reader
    await writer
    assert [v.value for v in result.variables] == [None]


async def test_name_retaken_is_one_snapshot_of_row_and_claim(app):
    # A deploy turns a claim into a row in one transaction; the purge verdict
    # must see one or the other, whoever holds the name and however it moved.
    from nerdit.db.models import Job, JobKind

    queries = app.state.queries
    for tid in ("a", "b"):
        await queries.create_api_token(
            ApiToken(id=tid, name=tid, role=TokenRole.submitter, token_hash=hash_token(tid))
        )
    assert await queries.name_retaken("svc", "a") is False  # nobody holds it
    await queries.mint_secret_claim("svc", "a")
    assert await queries.name_retaken("svc", "a") is False  # the owner's own claim
    assert await queries.name_retaken("svc", "a", any_claim=True) is True
    assert await queries.name_retaken("svc", None) is True  # NULL owner is not token a
    await queries.delete_secret_claim("svc")
    await queries.mint_secret_claim("svc", "b")
    assert await queries.name_retaken("svc", "a") is True  # a stranger's claim
    job = Job(id="j1", name="svc", service_name="svc", kind=JobKind.service, submitted_by_token="b")
    await queries.reserve_service_for_token(job)  # consumes B's claim, inserts B's row
    assert await queries.get_secret_claim("svc") is None
    assert await queries.name_retaken("svc", "a") is True  # the row alone says retaken
