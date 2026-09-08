"""Test domain claims and deletion against a real in-memory database.

Pin the case-insensitive primary key and all claim outcomes so another service
cannot take a bound domain. Domains survive endpoint-row replacement during port
reallocation. Service deletion removes domains atomically and reports names;
refused deletes preserve rows and never invoke the deletion callback.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from nerdit.db.models import Job, JobKind, JobStatus, ServiceDomain

# ---------------------------------------------------------------------------
# Harness
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


async def _jid(queries, name: str) -> str:
    """Id of the live service named ``name`` — the row the route would have authorized."""
    job = await queries.get_service_by_name(name)
    assert job is not None, name
    return job.id


async def _bind(queries, service: str, domain: str, *, acme: bool | None = False):
    """Claim ``domain`` for a service the caller just created."""
    return await queries.add_service_domain(
        service, domain, acme=acme, job_id=await _jid(queries, service)
    )


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------


async def test_the_table_exists_with_a_case_insensitive_primary_key(db):
    """``COLLATE NOCASE`` is the backstop behind the write path's fold: without
    it ``App.Example.com`` and ``app.example.com`` would be two rows, i.e. two
    services legitimately holding one name."""
    cursor = await db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'service_domains'"
    )
    row = await cursor.fetchone()

    assert row is not None
    ddl = row[0]
    assert "domain       TEXT PRIMARY KEY COLLATE NOCASE" in ddl
    assert "CHECK (acme IN (0, 1))" in ddl


async def test_the_service_index_exists(db):
    cursor = await db.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_service_domains_service'"
    )
    assert await cursor.fetchone() is not None


async def test_the_nocase_key_refuses_a_case_variant_of_a_bound_name(db, queries):
    """Proved at the SQL level, below the query layer's own fold."""
    await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(
            "INSERT INTO service_domains (domain, service_name) VALUES (?, ?)",
            ("APP.EXAMPLE.COM", "other"),
        )
    await db.conn.rollback()


async def test_the_acme_check_constraint_is_the_backstop(db, queries):
    """``acme`` is dormant data in WP1; the column still refuses a non-boolean."""
    await queries.create_job(_job("demo"))

    with pytest.raises(sqlite3.IntegrityError):
        await db.conn.execute(
            "INSERT INTO service_domains (domain, service_name, acme) VALUES (?, ?, ?)",
            ("app.example.com", "demo", 2),
        )
    await db.conn.rollback()


# ---------------------------------------------------------------------------
# 2. DomainQueries
# ---------------------------------------------------------------------------


async def test_get_returns_none_when_the_name_is_free(queries):
    assert await queries.get_service_domain("app.example.com") is None


async def test_a_fresh_database_lists_nothing(queries):
    assert await queries.list_service_domains() == []
    assert await queries.get_service_domains("demo") == []


async def test_claiming_a_free_name_inserts_it(queries):
    await queries.create_job(_job("demo"))

    outcome, row = await _bind(queries, "demo", "app.example.com")

    assert outcome == "inserted"
    assert isinstance(row, ServiceDomain)
    assert (row.domain, row.service_name, row.acme, row.kind) == (
        "app.example.com",
        "demo",
        False,
        "domain",
    )
    assert row.created_at.tzinfo is not None  # normalized to aware UTC on read
    assert await queries.get_service_domain("app.example.com") == row


async def test_re_claiming_by_the_same_service_is_idempotent(db, queries):
    """The re-PUT answers ``exists`` and writes no second row; ``created_at``
    survives, because "bound since" is the fact an operator reads."""
    await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")
    await db.conn.execute(
        "UPDATE service_domains SET created_at = ? WHERE domain = ?",
        ("2020-01-02 03:04:05", "app.example.com"),
    )
    await db.conn.commit()

    outcome, row = await _bind(queries, "demo", "app.example.com")

    assert outcome == "exists"
    assert row is not None
    assert row.created_at == datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    cursor = await db.conn.execute("SELECT COUNT(*) FROM service_domains")
    assert (await cursor.fetchone())[0] == 1


async def test_a_second_service_can_never_take_a_bound_name(queries):
    """The takeover guard: ``taken``, and the row is UNCHANGED.

    Deliberately not an ``ON CONFLICT DO UPDATE``: an operator who pointed DNS
    at this node must not have the name re-pointed by another app on the box.
    """
    await queries.create_job(_job("first"))
    await queries.create_job(_job("second"))
    await _bind(queries, "first", "app.example.com")

    outcome, row = await _bind(queries, "second", "app.example.com")

    assert outcome == "taken"
    assert row is not None and row.service_name == "first"
    still = await queries.get_service_domain("app.example.com")
    assert still is not None and still.service_name == "first"
    assert await queries.get_service_domains("second") == []


async def test_the_owning_service_of_a_taken_name_is_returned_for_the_caller_to_judge(queries):
    """The row travels back so the route can compare owners; the route is what
    must not ECHO the name to a non-owner (see the 409 in T3)."""
    await queries.create_job(_job("first"))
    await queries.create_job(_job("second"))
    await _bind(queries, "first", "app.example.com")

    _, row = await _bind(queries, "second", "app.example.com")

    assert row is not None
    assert row.service_name == "first"


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database, None])
async def test_claiming_refuses_a_name_no_live_service_owns(queries, kind):
    """The ``set_service_share`` TOCTOU closure verbatim: the route resolves the
    job and writes without holding the DB write lock, so a ``DELETE
    /services/{name}`` committing in that window must leave NO orphan row — a
    domain pointing at a name with no service, that nothing would ever clear."""
    job_id = "gone"
    if kind is not None:
        job_id = (await queries.create_job(_job("demo", kind=kind))).id

    outcome, row = await queries.add_service_domain(
        "demo", "app.example.com", acme=False, job_id=job_id
    )

    assert (outcome, row) == ("no_service", None)
    assert await queries.get_service_domain("app.example.com") is None


async def test_a_stale_job_id_cannot_bind_a_recreated_name(queries):
    """The third racing order (ultrareview, P26 WP-H): delete, then RE-CREATE.

    The route authorized ``job1``; by the time its write reaches the lock,
    ``job1`` is gone and another principal's ``job2`` owns the name. A
    name-only predicate would bind the domain to ``job2`` — a service the
    caller never owned.
    """
    job1 = await queries.create_job(_job("demo"))
    await queries.delete_service_checked(job1.id)
    job2 = await queries.create_job(_job("demo"))
    assert job2.id != job1.id

    stale, _ = await queries.add_service_domain(
        "demo", "app.example.com", acme=False, job_id=job1.id
    )
    assert stale == "no_service"
    assert await queries.get_service_domain("app.example.com") is None

    # The live row's own id still writes — the predicate is not over-tight.
    live, _ = await queries.add_service_domain(
        "demo", "app.example.com", acme=False, job_id=job2.id
    )
    assert live == "inserted"


async def test_a_job_id_belonging_to_another_service_is_refused(queries):
    """The predicate pins BOTH id and name, so an authorized job id cannot be
    replayed against a different service."""
    await queries.create_job(_job("first"))
    await queries.create_job(_job("second"))

    outcome, _ = await queries.add_service_domain(
        "second", "app.example.com", acme=False, job_id=await _jid(queries, "first")
    )

    assert outcome == "no_service"
    assert await queries.get_service_domain("app.example.com") is None


async def test_acme_stays_dormant_data_and_round_trips(queries):
    """WP1 never passes ``True`` (the route refuses it), but the column is real
    so WP2 needs no migration."""
    await queries.create_job(_job("demo"))

    _, row = await _bind(queries, "demo", "app.example.com", acme=True)
    assert row is not None and row.acme is True

    outcome, updated = await _bind(queries, "demo", "app.example.com", acme=False)
    assert outcome == "exists"
    assert updated is not None and updated.acme is False
    stored = await queries.get_service_domain("app.example.com")
    assert stored is not None and stored.acme is False


async def test_an_unspecified_acme_keeps_the_stored_flag(queries):
    """(P26 WP2 review round 1) ``acme=None`` is "unspecified", not "off".

    Every client defaults the field to false and the PUT is documented as
    idempotent, so the two-valued form made a re-add — the way an agent reads a
    domain's URL back — silently flip an ``issued`` public certificate to the
    internal CA. ``None`` therefore resolves to the row's own value on an update
    and to ``False`` only on an insert.
    """
    await queries.create_job(_job("demo"))
    _, row = await _bind(queries, "demo", "app.example.com", acme=True)
    assert row is not None and row.acme is True

    outcome, unchanged = await _bind(queries, "demo", "app.example.com", acme=None)

    assert outcome == "exists"
    assert unchanged is not None and unchanged.acme is True
    stored = await queries.get_service_domain("app.example.com")
    assert stored is not None and stored.acme is True

    # …and on an INSERT it is plain ``false`` — never NULL in the column.
    _, fresh = await _bind(queries, "demo", "other.example.com", acme=None)
    assert fresh is not None and fresh.acme is False


async def test_a_stray_case_variant_resolves_to_the_same_row(queries):
    """The write path folds, but the point read must not depend on it."""
    await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    found = await queries.get_service_domain("APP.Example.CoM")

    assert found is not None and found.service_name == "demo"


async def test_list_is_one_read_ordered_by_service_then_domain(queries):
    await queries.create_job(_job("beta"))
    await queries.create_job(_job("alpha"))
    await _bind(queries, "beta", "z.example.com")
    await _bind(queries, "alpha", "b.example.com")
    await _bind(queries, "alpha", "a.example.com")

    rows = await queries.list_service_domains()

    assert [(r.service_name, r.domain) for r in rows] == [
        ("alpha", "a.example.com"),
        ("alpha", "b.example.com"),
        ("beta", "z.example.com"),
    ]


async def test_get_service_domains_is_scoped_and_sorted(queries):
    await queries.create_job(_job("alpha"))
    await queries.create_job(_job("beta"))
    await _bind(queries, "alpha", "b.example.com")
    await _bind(queries, "alpha", "a.example.com")
    await _bind(queries, "beta", "z.example.com")

    rows = await queries.get_service_domains("alpha")

    assert [r.domain for r in rows] == ["a.example.com", "b.example.com"]
    assert {r.service_name for r in rows} == {"alpha"}


async def test_remove_reports_whether_a_row_was_there(queries):
    await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    assert await queries.remove_service_domain("demo", "app.example.com") is True
    assert await queries.remove_service_domain("demo", "app.example.com") is False
    assert await queries.get_service_domain("app.example.com") is None


async def test_remove_is_scoped_to_the_owning_service(queries):
    """DELETE must not be a way around ``taken``: another service aiming at a
    bound name removes nothing."""
    await queries.create_job(_job("first"))
    await queries.create_job(_job("second"))
    await _bind(queries, "first", "app.example.com")

    assert await queries.remove_service_domain("second", "app.example.com") is False
    assert await queries.get_service_domain("app.example.com") is not None


# ---------------------------------------------------------------------------
# 3. CRIT-4 — a port reallocation must not take the domains
# ---------------------------------------------------------------------------


async def test_a_crit4_port_reallocation_keeps_the_domains(queries):
    """Why this table carries no FK to ``service_endpoints``.

    CRIT-4 (a foreign process grabbed the held port while the daemon was down)
    DELETEs the endpoint row and allocates a fresh one. Under an FK that would
    silently take every custom domain the operator had bound — the URL moving
    is the accepted cost, losing their DNS binding is not.
    """
    job = await queries.create_job(_job("demo"))
    first = await queries.acquire_service_port("demo", job.id, 8000, (9400, 9410))
    await _bind(queries, "demo", "app.example.com")
    await _bind(queries, "demo", "www.example.com")

    second = await queries.acquire_service_port(
        "demo", job.id, 8000, (9400, 9410), is_bindable=lambda port: port != first.host_port
    )

    assert second.host_port != first.host_port
    assert [r.domain for r in await queries.get_service_domains("demo")] == [
        "app.example.com",
        "www.example.com",
    ]


# ---------------------------------------------------------------------------
# 4. Cascade — delete_service_checked (D-P26-1)
# ---------------------------------------------------------------------------


async def test_delete_service_checked_takes_the_domain_rows(queries):
    job = await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")
    await _bind(queries, "demo", "www.example.com")

    assert await queries.delete_service_checked(job.id) == []

    assert await queries.get_service_domains("demo") == []
    assert await queries.list_service_domains() == []


async def test_delete_service_checked_reports_the_names_it_took_sorted(queries):
    """The list is only knowable INSIDE the transaction — the purge route emits
    one ``domain.removed reason=service_deleted`` per name off this callback."""
    seen: list[list[str]] = []
    job = await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "b.example")
    await _bind(queries, "demo", "a.example")

    await queries.delete_service_checked(job.id, on_domains_removed=seen.append)

    assert seen == [["a.example", "b.example"]]


async def test_a_service_with_no_domains_reports_an_empty_list(queries):
    seen: list[list[str]] = []
    job = await queries.create_job(_job("plain"))

    await queries.delete_service_checked(job.id, on_domains_removed=seen.append)

    assert seen == [[]]


async def test_delete_service_checked_leaves_another_services_domains_alone(queries):
    job = await queries.create_job(_job("demo"))
    await queries.create_job(_job("other"))
    await _bind(queries, "demo", "app.example.com")
    await _bind(queries, "other", "www.example.com")

    await queries.delete_service_checked(job.id)

    assert [r.domain for r in await queries.get_service_domains("other")] == ["www.example.com"]


async def test_a_refused_delete_keeps_the_domains_and_reports_nothing(queries):
    """The rows go in the SAME transaction, so the rollback keeps them — the
    service is still there and still bound. Claiming otherwise would mint an
    edge that never happened."""
    seen: list[list[str]] = []
    job = await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    refused = await queries.delete_service_checked(
        job.id,
        lambda rows: [{"service": "dep", "id": "s1", "binding": "default"}],
        on_domains_removed=seen.append,
    )

    assert refused == [{"service": "dep", "id": "s1", "binding": "default"}]
    assert seen == []
    assert await queries.get_service_domain("app.example.com") is not None


async def test_an_absent_row_deletes_no_domain_and_never_calls_back(queries):
    """``None`` means "this request deleted nothing" — a name recreated in the
    meantime must keep its own, brand-new domains."""
    seen: list[list[str]] = []
    await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    assert (
        await queries.delete_service_checked("nope-nope-nope", on_domains_removed=seen.append)
        is None
    )

    assert seen == []
    assert await queries.get_service_domain("app.example.com") is not None


async def test_a_recreated_name_starts_with_no_domains(queries):
    job = await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")
    await queries.delete_service_checked(job.id)

    await queries.create_job(_job("demo"))

    assert await queries.get_service_domains("demo") == []
    assert await queries.get_service_domain("app.example.com") is None


async def test_the_two_cascade_callbacks_are_independent(queries):
    """A service can be shared without domains and vice versa; neither callback
    may be gated on the other's rowcount."""
    shares: list[bool] = []
    domains: list[list[str]] = []
    job = await queries.create_job(_job("demo"))
    await _bind(queries, "demo", "app.example.com")

    await queries.delete_service_checked(
        job.id, on_share_removed=shares.append, on_domains_removed=domains.append
    )

    assert shares == [False]
    assert domains == [["app.example.com"]]
