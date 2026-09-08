"""Atomic per-token quota reservation tests (P1 / S4).

Pure-async over the in-memory DB (no TestClient), exercising
``Queries.reserve_service_for_token``. Covers the uncapped legacy/None fast
path, per-workload / cumulative-GPU / concurrency caps, the service transients
holding their slot (no re-reservation), terminal workloads dropping out of the
active set, and that concurrent reservations cannot race past a cap.
"""

from __future__ import annotations

import pytest

from nerdit.daemon.auth import QuotaExceeded, generate_token, hash_token
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole


async def _make_token(queries, *, max_gpus=None, max_concurrent_jobs=None) -> ApiToken:
    raw = generate_token()
    return await queries.create_api_token(
        ApiToken(
            name="bot",
            role=TokenRole.submitter,
            token_hash=hash_token(raw),
            max_gpus=max_gpus,
            max_concurrent_jobs=max_concurrent_jobs,
        )
    )


def _svc(name: str, token_id: str | None, gpus: int = 1) -> Job:
    """A service row shaped as ``POST /services`` builds it before reserving."""
    return Job(
        id=name,
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=gpus,
        status=JobStatus.building,
        desired_state="running",
        submitted_by_token=token_id,
    )


async def _services(queries) -> list[Job]:
    return await queries.list_jobs(kind=JobKind.service)


# --- uncapped fast path -------------------------------------------------------


async def test_none_owner_is_uncapped(queries):
    for i in range(5):
        await queries.reserve_service_for_token(_svc(f"none-{i}", None, gpus=8))
    assert len(await _services(queries)) == 5


async def test_token_without_caps_is_uncapped(queries):
    token = await _make_token(queries)  # both caps None
    for i in range(4):
        await queries.reserve_service_for_token(_svc(f"u-{i}", token.id, gpus=4))
    assert len(await _services(queries)) == 4


# --- concurrency cap ----------------------------------------------------------


async def test_concurrency_cap_rejects_over_limit(queries):
    token = await _make_token(queries, max_concurrent_jobs=2)
    await queries.reserve_service_for_token(_svc("c-1", token.id))
    await queries.reserve_service_for_token(_svc("c-2", token.id))
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(_svc("c-3", token.id))
    assert exc.value.reason == "max_concurrent_jobs"
    assert exc.value.limit == 2
    # The rejected workload was rolled back, not persisted.
    assert await queries.get_job("c-3") is None


async def test_terminal_jobs_free_concurrency_slot(queries):
    token = await _make_token(queries, max_concurrent_jobs=1)
    await queries.reserve_service_for_token(_svc("t-1", token.id))
    await queries.update_job_status("t-1", JobStatus.stopped)
    # Slot freed → next reservation succeeds.
    await queries.reserve_service_for_token(_svc("t-2", token.id))
    assert await queries.get_job("t-2") is not None


async def test_quota_counts_across_kinds(queries):
    # QUOTA-1: the count query has no ``kind`` predicate and spans the union of
    # non-terminal statuses across kinds, so a workload parked in a status the
    # reserving kind never uses still occupies the token's slot. A legacy
    # ``pending`` batch row is the canonical case.
    token = await _make_token(queries, max_concurrent_jobs=1)
    await queries.create_job(
        Job(
            id="b-1",
            kind=JobKind.batch,
            script_path="/tmp/x.py",
            gpu_count=0,
            status=JobStatus.pending,
            submitted_by_token=token.id,
        )
    )
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(_svc("s-1", token.id, gpus=0))
    assert exc.value.reason == "max_concurrent_jobs"
    assert await queries.get_job("s-1") is None


# --- cumulative + per-workload GPU cap ----------------------------------------


async def test_cumulative_gpu_cap_rejects(queries):
    token = await _make_token(queries, max_gpus=4)
    await queries.reserve_service_for_token(_svc("g-1", token.id, gpus=3))
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(_svc("g-2", token.id, gpus=2))
    assert exc.value.reason == "max_gpus"
    assert exc.value.limit == 4
    assert exc.value.current == 3


async def test_per_job_gpu_request_exceeding_cap_rejected(queries):
    token = await _make_token(queries, max_gpus=2)
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(_svc("pj-1", token.id, gpus=3))
    assert exc.value.reason == "max_gpus"
    assert exc.value.current == 0


async def test_gpu_cap_allows_exact_fit(queries):
    token = await _make_token(queries, max_gpus=4)
    await queries.reserve_service_for_token(_svc("e-1", token.id, gpus=2))
    await queries.reserve_service_for_token(_svc("e-2", token.id, gpus=2))
    assert len(await _services(queries)) == 2


# --- transient statuses keep their slot (no re-reservation) -------------------


@pytest.mark.parametrize(
    "status", [JobStatus.building, JobStatus.degraded, JobStatus.restarting, JobStatus.scheduled]
)
async def test_transient_service_still_counts_against_concurrency(queries, status):
    """CRIT-3: a service the reconciler is converging keeps its concurrency slot.

    Transitions go through ``update_job_status`` and never re-reserve, so a
    transient status dropping out of ``_ACTIVE_STATUSES`` would let a token
    cycle services through it to slip past ``max_concurrent_jobs``.
    """
    token = await _make_token(queries, max_concurrent_jobs=1)
    await queries.reserve_service_for_token(_svc("r-1", token.id))
    await queries.update_job_status("r-1", status)
    with pytest.raises(QuotaExceeded) as exc:
        await queries.reserve_service_for_token(_svc("r-2", token.id))
    assert exc.value.reason == "max_concurrent_jobs"


# --- rapid reservations cannot slip past the cap ------------------------------


async def test_rapid_submits_respect_cap(queries):
    """The atomic count+INSERT inside one BEGIN IMMEDIATE holds the cap across
    many back-to-back reservations.

    The daemon drives reservations over a single shared connection and awaits
    each to completion (mirroring ``allocate_gpus``), so the realistic guarantee
    is that no rapid sequence slips past the cap — exactly
    ``max_concurrent_jobs`` are accepted, the rest are rejected.
    """
    token = await _make_token(queries, max_concurrent_jobs=2)
    accepted, rejected = 0, 0
    for i in range(6):
        try:
            await queries.reserve_service_for_token(_svc(f"race-{i}", token.id))
            accepted += 1
        except QuotaExceeded:
            rejected += 1
    assert accepted == 2
    assert rejected == 4
    # Exactly two rows persisted — the atomic BEGIN IMMEDIATE held the cap.
    assert len(await _services(queries)) == 2


async def test_truly_concurrent_submits_respect_cap(queries):
    """M1 regression: concurrently-scheduled reservations cannot exceed the cap.

    Without the single-writer lock, the event loop can run another coroutine's
    bare ``commit()`` inside an open ``BEGIN IMMEDIATE`` span (or two
    ``BEGIN IMMEDIATE`` collide), letting two reservations read the same count
    and both insert. ``asyncio.gather`` schedules them concurrently so they
    really interleave at every await; the lock must serialize them.
    """
    import asyncio

    token = await _make_token(queries, max_concurrent_jobs=3)
    results = await asyncio.gather(
        *(queries.reserve_service_for_token(_svc(f"cc-{i}", token.id)) for i in range(10)),
        return_exceptions=True,
    )
    accepted = [r for r in results if not isinstance(r, Exception)]
    rejected = [r for r in results if isinstance(r, QuotaExceeded)]
    other = [r for r in results if isinstance(r, Exception) and not isinstance(r, QuotaExceeded)]
    assert other == [], f"unexpected errors (e.g. nested-transaction 5xx): {other}"
    assert len(accepted) == 3
    assert len(rejected) == 7
    assert len(await _services(queries)) == 3


async def test_reserve_atomic_against_concurrent_foreign_writes(queries):
    """M1 regression: a foreign bare-commit writer cannot break a reservation.

    A running service's log stream commits on the same connection. Interleaving
    many ``append_log`` commits with capped reservations must neither corrupt
    the cap nor raise (the foreign commit landing inside reserve's transaction
    would otherwise surface as a 5xx and a lost reservation).
    """
    import asyncio

    from nerdit.db.models import LogStream

    token = await _make_token(queries, max_concurrent_jobs=4)
    await queries.create_job(_svc("log-src", None))  # a row to attach logs to

    async def _spam_logs() -> None:
        for i in range(20):
            await queries.append_log("log-src", f"line {i}", LogStream.stdout)

    reserves = [queries.reserve_service_for_token(_svc(f"fw-{i}", token.id)) for i in range(8)]
    results = await asyncio.gather(_spam_logs(), *reserves, return_exceptions=True)
    reserve_results = results[1:]
    accepted = [r for r in reserve_results if not isinstance(r, Exception)]
    other = [
        r for r in reserve_results if isinstance(r, Exception) and not isinstance(r, QuotaExceeded)
    ]
    assert other == [], f"foreign commit broke a reservation: {other}"
    assert len(accepted) == 4


# --- (P20) count_active_jobs_public: the route-callable quota read ------------


async def test_count_active_jobs_public_reports_count_and_cap(queries):
    """The plain read returns ``(active, cap)`` and leaves the arithmetic to the
    caller — a one-off run is charged against ``max_concurrent_jobs`` without
    inserting a jobs row (D-C), so the route adds its own pending count."""
    token = await _make_token(queries, max_concurrent_jobs=3)
    assert await queries.count_active_jobs_public(token.id) == (0, 3)

    await queries.reserve_service_for_token(_svc("p-1", token.id, gpus=0))
    await queries.reserve_service_for_token(_svc("p-2", token.id, gpus=0))
    assert await queries.count_active_jobs_public(token.id) == (2, 3)

    # A terminal workload frees the slot, exactly as ``_check_quota_locked`` sees it.
    await queries.update_job_status("p-1", JobStatus.stopped)
    assert await queries.count_active_jobs_public(token.id) == (1, 3)


async def test_count_active_jobs_public_capless_token_reports_none(queries):
    """``None`` — uncapped — is returned ONLY for a live row with a NULL cap."""
    token = await _make_token(queries)  # both caps None
    await queries.reserve_service_for_token(_svc("u-1", token.id, gpus=0))
    assert await queries.count_active_jobs_public(token.id) == (1, None)


async def test_count_active_jobs_public_counts_across_kinds(queries):
    """QUOTA-1: no ``kind`` predicate, ``_ACTIVE_STATUSES`` union — a model row
    parked in a status services never use still occupies the token's slot."""
    token = await _make_token(queries, max_concurrent_jobs=2)
    await queries.create_job(
        Job(
            id="m-1",
            kind=JobKind.model,
            service_name="ollama-x",
            gpu_count=0,
            status=JobStatus.scheduled,
            submitted_by_token=token.id,
        )
    )
    await queries.create_job(
        Job(
            id="b-1",
            kind=JobKind.batch,
            script_path="/tmp/x.py",
            gpu_count=0,
            status=JobStatus.pending,
            submitted_by_token=token.id,
        )
    )
    assert await queries.count_active_jobs_public(token.id) == (2, 2)


async def test_count_active_jobs_public_revoked_token_reads_as_capped_out(queries):
    """The R12 pin: a REVOKED token must read as capped-out, never uncapped.

    The caps read filters ``revoked = 0``, so a revoked token matches no row —
    and the missing-row mapping is fail-closed ``0``, not ``None``. With the
    caller's ``active + pending > cap`` test and a pending count of at least 1,
    a cap of 0 always denies. Mapping a missing row to ``None`` (what the
    transaction-internal ``_check_quota_locked`` does, because its caller has
    already authenticated the token in the same txn) would hand a revoked
    token unlimited run capacity.
    """
    token = await _make_token(queries, max_concurrent_jobs=5)
    assert await queries.count_active_jobs_public(token.id) == (0, 5)

    assert await queries.revoke_api_token(token.id) is True
    active, cap = await queries.count_active_jobs_public(token.id)
    assert cap == 0
    assert cap is not None  # explicitly NOT the uncapped sentinel
    assert active == 0


async def test_count_active_jobs_public_unknown_token_reads_as_capped_out(queries):
    """Same fail-closed mapping for a token id that never existed."""
    assert await queries.count_active_jobs_public("tok-does-not-exist") == (0, 0)
