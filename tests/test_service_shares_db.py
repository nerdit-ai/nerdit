"""P26 WP-H T1 — the ``service_shares`` persistence layer (D-P26-H1).

Three things are proved here, in the order the row's life runs:

1. the four :class:`~nerdit.db.queries.shares.ShareQueries` methods against a
   real in-memory database (the ``queries`` fixture), including the UPSERT's
   ``created_at`` preservation and the CHECK backstop;
2. the cascade — ``delete_service_checked`` takes the share row inside its own
   transaction, keeps it on a refused delete, and never touches a sibling's;
3. the purge route's ``share.removed`` edge, driven through the real
   ``DELETE /services/{ident}`` route with the ``tests/test_delete_purge.py``
   harness (imported, never modified) and a recorder double.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from nerdit.db.models import Job, JobKind, JobStatus, ServiceShare
from tests.test_delete_purge import FakeRuntime, _app, _auth, _queries, _svc

# ---------------------------------------------------------------------------
# 1. ShareQueries against the real DB
# ---------------------------------------------------------------------------


async def _jid(queries: Any, name: str) -> str:
    """Id of the live service named ``name`` — the row the route would have authorized."""
    job = await queries.get_service_by_name(name)
    assert job is not None, name
    return job.id


async def test_get_returns_none_when_not_shared(queries):
    """The fail-closed default: absence of a row IS "not shared"."""
    assert (await queries.list_service_shares()).get("demo") is None


async def test_set_creates_the_row_and_returns_it(queries):
    await queries.create_job(_job("demo"))

    share = await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    assert isinstance(share, ServiceShare)
    assert (share.service_name, share.access) == ("demo", "private")
    assert share.created_at.tzinfo is not None  # normalized to aware UTC on read
    assert (await queries.list_service_shares()).get("demo") == share


async def test_set_upserts_access_and_preserves_created_at(db, queries):
    await queries.create_job(_job("demo"))
    first = await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))
    # Backdate the stored row so a re-INSERT (rather than an UPDATE) would be
    # visible: "shared since" must survive an access flip.
    await db.conn.execute(
        "UPDATE service_shares SET created_at = ? WHERE service_name = ?",
        ("2020-01-02 03:04:05", "demo"),
    )
    await db.conn.commit()

    second = await queries.set_service_share("demo", "public", job_id=await _jid(queries, "demo"))

    assert second.access == "public"
    assert second.created_at == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert second.created_at != first.created_at
    cursor = await db.conn.execute("SELECT COUNT(*) FROM service_shares")
    assert (await cursor.fetchone())[0] == 1


async def test_set_refuses_an_access_outside_the_check_constraint(queries):
    """The route's pydantic model is the gate; the column CHECK is the backstop."""
    await queries.create_job(_job("demo"))

    with pytest.raises(sqlite3.IntegrityError):
        await queries.set_service_share("demo", "world", job_id=await _jid(queries, "demo"))
    assert (await queries.list_service_shares()).get("demo") is None


@pytest.mark.parametrize("existing", [None, "private", "public"])
async def test_preserve_existing_creates_only_missing_share(queries, existing):
    job = await queries.create_job(_job("demo"))
    first = await queries.set_service_share("demo", existing, job_id=job.id) if existing else None
    result = await queries.set_service_share(
        "demo", "private", job_id=job.id, preserve_existing=True
    )
    assert result.access == (existing or "private")
    assert (await queries.list_service_shares()).get("demo") == result
    if first:
        assert result == first
    assert (
        await queries.set_service_share(
            "demo", "private", job_id="deleted-job", preserve_existing=True
        )
        is None
    )


@pytest.mark.parametrize("preview_first", [True, False])
async def test_preview_cannot_overwrite_concurrent_public_share(queries, preview_first):
    job = await queries.create_job(_job("demo"))
    writes = [
        queries.set_service_share("demo", "private", job_id=job.id, preserve_existing=True),
        queries.set_service_share("demo", "public", job_id=job.id),
    ]
    await asyncio.gather(*(writes if preview_first else reversed(writes)))
    assert ((await queries.list_service_shares()).get("demo")).access == "public"


async def test_list_is_one_read_keyed_by_name(queries):
    await queries.create_job(_job("alpha"))
    await queries.create_job(_job("beta"))
    await queries.set_service_share("alpha", "private", job_id=await _jid(queries, "alpha"))
    await queries.set_service_share("beta", "public", job_id=await _jid(queries, "beta"))

    shares = await queries.list_service_shares()

    assert set(shares) == {"alpha", "beta"}
    assert shares["beta"].access == "public"
    assert shares["alpha"].service_name == "alpha"


async def test_list_is_empty_on_a_fresh_database(queries):
    assert await queries.list_service_shares() == {}


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database, None])
async def test_set_refuses_a_name_no_live_service_owns(queries, kind):
    """The TOCTOU guard (PR review, P26 WP-H).

    The route resolves the job and then writes without holding the DB write
    lock, so a ``DELETE /services/{name}`` committing in that window used to
    leave an ORPHAN row: a name with no service, that nothing ever cleared, and
    that would have exposed the NEXT app deployed under it from its first
    second. The upsert decides "is there still a ``kind='service'`` job here?"
    inside its own lock-holding statement instead — ``None``, and no row.
    """
    job_id = "gone"
    if kind is not None:
        job_id = (await queries.create_job(_job("demo", kind=kind))).id

    assert await queries.set_service_share("demo", "private", job_id=job_id) is None
    assert (await queries.list_service_shares()).get("demo") is None


async def test_a_concurrent_delete_cannot_leave_an_orphan_row(queries):
    """The two orders the window admits, both safe.

    Delete-then-write inserts nothing (proved above); write-then-delete is
    cascaded away inside ``delete_service_checked``'s transaction. Either way a
    recreated name starts unshared, which is what the table's whole no-FK
    integrity story promises.
    """
    job = await queries.create_job(_job("demo"))
    assert await queries.set_service_share("demo", "public", job_id=job.id) is not None

    await queries.delete_service_checked(job.id)

    assert (await queries.list_service_shares()).get("demo") is None
    assert await queries.set_service_share("demo", "public", job_id=job.id) is None


async def test_a_same_name_recreate_cannot_inherit_an_authorized_write(queries):
    """The third order (ultrareview, P26 WP-H): delete, then RE-CREATE.

    The route authorized ``job1``; by the time its write reaches the lock,
    ``job1`` is gone and another principal's ``job2`` owns the name. A
    name-only predicate would bind the share to ``job2`` — a service the
    caller never owned. Pinned on the id, the write lands nowhere.
    """
    job1 = await queries.create_job(_job("demo"))
    await queries.delete_service_checked(job1.id)
    job2 = await queries.create_job(_job("demo"))
    assert job2.id != job1.id

    assert await queries.set_service_share("demo", "private", job_id=job1.id) is None
    assert (await queries.list_service_shares()).get("demo") is None
    # The live row's own id still writes — the predicate is not over-tight.
    assert await queries.set_service_share("demo", "private", job_id=job2.id) is not None


async def test_delete_reports_whether_a_row_was_there(queries):
    await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    assert await queries.delete_service_share("demo") is True
    assert await queries.delete_service_share("demo") is False
    assert (await queries.list_service_shares()).get("demo") is None


# ---------------------------------------------------------------------------
# 2. Cascade — delete_service_checked (D-P26-H1)
# ---------------------------------------------------------------------------


def _job(name: str, *, kind: JobKind = JobKind.service) -> Job:
    return Job(
        name=name,
        kind=kind,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config='{"image": "demo:latest", "port": 8000}',
    )


async def test_delete_service_checked_takes_the_share_row(queries):
    job = await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "public", job_id=await _jid(queries, "demo"))

    assert await queries.delete_service_checked(job.id) == []
    assert (await queries.list_service_shares()).get("demo") is None


async def test_delete_service_checked_leaves_other_shares_alone(queries):
    job = await queries.create_job(_job("demo"))
    await queries.create_job(_job("other"))
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))
    await queries.set_service_share("other", "private", job_id=await _jid(queries, "other"))

    await queries.delete_service_checked(job.id)

    assert (await queries.list_service_shares()).get("other") is not None


async def test_a_refused_delete_keeps_the_share(queries):
    """The row goes in the SAME transaction, so a rollback keeps it — the
    service is still there and still shared."""
    job = await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    refused = await queries.delete_service_checked(
        job.id, lambda rows: [{"service": "dep", "id": "s1", "binding": "default"}]
    )

    assert refused == [{"service": "dep", "id": "s1", "binding": "default"}]
    assert (await queries.list_service_shares()).get("demo") is not None


async def test_an_absent_row_deletes_no_share(queries):
    """``None`` means "this request deleted nothing" — a name recreated in the
    meantime must keep its own, brand-new share."""
    await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    assert await queries.delete_service_checked("nope-nope-nope") is None
    assert (await queries.list_service_shares()).get("demo") is not None


async def test_a_recreated_name_starts_unshared(queries):
    job = await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "public", job_id=await _jid(queries, "demo"))
    await queries.delete_service_checked(job.id)

    await queries.create_job(_job("demo"))

    assert (await queries.list_service_shares()).get("demo") is None


async def test_the_transaction_reports_whether_it_took_a_share_row(queries):
    """``on_share_removed`` is the purge route's only honest source (PR review).

    Reading "was it shared?" before the transaction races a concurrent
    ``PUT …/share``: that row would be cascaded away with the feed never told.
    The DELETE's own rowcount cannot race itself.
    """
    seen: list[bool] = []
    shared = await queries.create_job(_job("shared"))
    await queries.set_service_share("shared", "private", job_id=await _jid(queries, "shared"))
    plain = await queries.create_job(_job("plain"))

    await queries.delete_service_checked(shared.id, on_share_removed=seen.append)
    await queries.delete_service_checked(plain.id, on_share_removed=seen.append)

    assert seen == [True, False]


async def test_a_refused_delete_never_reports_a_share_removal(queries):
    """The callback fires only on the success path: the rollback kept the row,
    so claiming otherwise would mint an edge that never happened."""
    seen: list[bool] = []
    job = await queries.create_job(_job("demo"))
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    await queries.delete_service_checked(
        job.id,
        lambda rows: [{"service": "dep", "id": "s1", "binding": "default"}],
        on_share_removed=seen.append,
    )

    assert seen == []
    assert (await queries.list_service_shares()).get("demo") is not None


# ---------------------------------------------------------------------------
# 3. The purge route's share.removed edge
# ---------------------------------------------------------------------------


def _delete_app(monkeypatch, tmp_path, *, had_share: bool):
    """The ``test_delete_purge`` harness + a recorder double, wired for shares.

    ``delete_service_checked`` is a mock here, so it stands in for the real
    transaction by invoking ``on_share_removed`` the way that transaction does —
    which is now the ONLY channel the route learns the fact through.
    """
    queries = _queries(_svc(), workloads=[])

    async def _delete(*_args, on_share_removed=None, **_kw):
        if on_share_removed is not None:
            on_share_removed(had_share)
        return []

    queries.delete_service_checked = AsyncMock(side_effect=_delete)
    recorder = AsyncMock()
    monkeypatch.setattr("nerdit.daemon.service_purge.get_recorder", lambda: recorder)
    app = _app(runtime=FakeRuntime([]), queries=queries, data_dir=tmp_path)
    return app, queries, recorder


def test_deleting_a_shared_service_records_the_removal(monkeypatch, tmp_path):
    app, _queries_, recorder = _delete_app(monkeypatch, tmp_path, had_share=True)

    with TestClient(app) as client:
        resp = client.delete("/services/a", headers=_auth())

    assert resp.status_code == 200
    recorder.record.assert_awaited_once()
    args, kwargs = recorder.record.await_args
    assert args == ("share.removed",)
    assert kwargs == {"kind": "service", "service_name": "a", "reason": "service_deleted"}


def test_deleting_an_unshared_service_records_nothing(monkeypatch, tmp_path):
    app, _queries_, recorder = _delete_app(monkeypatch, tmp_path, had_share=False)

    with TestClient(app) as client:
        assert client.delete("/services/a", headers=_auth()).status_code == 200

    recorder.record.assert_not_awaited()


def test_the_route_never_pre_reads_the_share_row(monkeypatch, tmp_path):
    """The fix for the pre-read race (PR review, P26 WP-H).

    The route used to read the share row before entering the write
    lock, so a ``PUT …/share`` committing in the window was cascaded away with
    ``had_share=False`` — the row gone, the feed silent. The in-transaction
    rowcount is now the only input, and this pins that the event rides it.
    """
    app, queries, recorder = _delete_app(monkeypatch, tmp_path, had_share=True)

    with TestClient(app) as client:
        assert client.delete("/services/a", headers=_auth()).status_code == 200

    assert recorder.record.await_args.args == ("share.removed",)
