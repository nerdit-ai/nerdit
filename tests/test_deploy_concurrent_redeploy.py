"""Two overlapping redeploys of one service must not both take the same version.

The deploy pipeline derives the next ``build_version`` from a config read that
happens far upstream of the write, and nothing serializes the two. Without the
compare-and-swap in ``update_service_config_guarded`` both requests stamp the
same version, the second overwrites the first's ``build_context_dir``, and the
loser holds a 201 naming an image that is never built while its extracted tree
is orphaned on disk with nothing in ``config`` referencing it.

The real deploy router over a real in-memory database and real ZIP extraction,
under the auth + audit + idempotency stack. The interleave is forced at the
write, not by luck: both requests park on a barrier as they reach the writer, so
both have already read the same pre-deploy config.
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import generate_token, hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole


def _node_zip(marker: str) -> bytes:
    """A buildable Node context whose content differs per caller."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("package.json", json.dumps({"name": "demo", "scripts": {"start": "node i.js"}}))
        zf.writestr("i.js", f"console.log({marker!r})")
    return buf.getvalue()


def _app(queries, upload_root: Path) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(upload_root)
    app.state.settings = settings
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=None, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


async def _seed(queries, tmp_path: Path) -> str:
    """One submitter token owning one deployed service at build_version 1."""
    raw = generate_token()
    token = await queries.create_api_token(
        ApiToken(name="bot", role=TokenRole.submitter, token_hash=hash_token(raw))
    )
    ctx = tmp_path / "v1"
    ctx.mkdir()
    await queries.create_job(
        Job(
            kind=JobKind.service,
            service_name="demo",
            name="demo",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="running",
            submitted_by_token=token.id,
            config=json.dumps(
                {
                    "image": "nerdit-app/demo:1",
                    "image_repo": "nerdit-app/demo",
                    "build_version": 1,
                    "max_version": 1,
                    "build_context_dir": str(ctx),
                    "port": 8000,
                }
            ),
        )
    )
    return raw


def _gate_the_writer(queries, parties: int):
    """Hold every redeploy write until `parties` of them have arrived.

    Each request reads the row before it writes, so parking at the writer proves
    both callers computed their version from the same pre-deploy config — the
    race the CAS exists to resolve, without relying on scheduling luck. Returns
    the ungated writer so a caller can lift the gate rather than nest another.
    """
    barrier = asyncio.Barrier(parties)
    inner = queries.update_service_config_guarded

    async def gated(*args, **kwargs):
        await barrier.wait()
        return await inner(*args, **kwargs)

    queries.update_service_config_guarded = gated
    return inner


async def _deploy(client: httpx.AsyncClient, raw: str, marker: str) -> httpx.Response:
    return await client.post(
        "/deploy",
        data={"name": "demo", "port": "8000", "gpus": "0"},
        files={"archive": ("app.zip", _node_zip(marker), "application/zip")},
        headers={"Authorization": f"Bearer {raw}", "Idempotency-Key": f"key-{marker}"},
    )


async def test_overlapping_redeploys_yield_one_201_one_409_and_no_orphan_tree(queries, tmp_path):
    """One deploy wins the version; the other is refused and leaves nothing behind."""
    raw = await _seed(queries, tmp_path)
    uploads = tmp_path / "uploads"
    _gate_the_writer(queries, 2)
    app = _app(queries, uploads)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first, second = await asyncio.gather(_deploy(client, raw, "a"), _deploy(client, raw, "b"))

    codes = sorted(r.status_code for r in (first, second))
    assert codes == [201, 409], (first.status_code, first.text, second.status_code, second.text)
    winner = first if first.status_code == 201 else second
    loser = second if first.status_code == 201 else first
    assert loser.json()["code"] == "deploy.concurrent_redeploy"
    assert loser.json()["hint"]

    # One generation was allocated, and the row points at the context that built it.
    row = await queries.get_service_by_name("demo")
    cfg = json.loads(row.config)
    assert cfg["build_version"] == 2
    assert cfg["max_version"] == 2
    assert cfg["image"] == "nerdit-app/demo:2"
    assert winner.json()["image"] == "nerdit-app/demo:2"

    # The loser's extracted tree is gone: exactly the winner's context survives.
    survivors = sorted(p for p in uploads.iterdir() if p.is_dir())
    assert survivors == [Path(cfg["build_context_dir"])]


async def test_a_later_redeploy_still_succeeds_after_a_refused_one(queries, tmp_path):
    """The CAS refuses a lost race, not the next honest deploy of the same app."""
    raw = await _seed(queries, tmp_path)
    uploads = tmp_path / "uploads"
    ungated = _gate_the_writer(queries, 2)
    app = _app(queries, uploads)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        await asyncio.gather(_deploy(client, raw, "a"), _deploy(client, raw, "b"))
        queries.update_service_config_guarded = ungated
        again = await _deploy(client, raw, "c")

    assert again.status_code == 201, again.text
    assert again.json()["image"] == "nerdit-app/demo:3"


@pytest.mark.parametrize("prev", [{}, {"build_version": 4}, {"max_version": 7, "build_version": 2}])
async def test_the_guard_matches_the_allocator_the_pipeline_reads(queries, prev):
    """The SQL version predicate answers what the pipeline's own expression reads.

    A row with no version at all, a pre-`max_version` row, and a rolled-back row
    each have to compare equal to the number the deploy pipeline derived, or an
    ordinary redeploy would 409 against itself.
    """
    job = await queries.create_job(
        Job(
            kind=JobKind.service,
            service_name="demo",
            name="demo",
            gpu_count=0,
            status=JobStatus.running,
            desired_state="running",
            config=json.dumps(prev),
        )
    )
    expected = int(prev.get("max_version", prev.get("build_version", 0)))
    assert await queries.update_service_config_guarded(
        job.id,
        json.dumps({**prev, "max_version": expected + 1}),
        expect_max_version=expected,
        status=JobStatus.restarting,
        desired_state="running",
    )
    # The same write replayed against the now-advanced row is a miss, not a clobber.
    assert not await queries.update_service_config_guarded(
        job.id,
        json.dumps({**prev, "max_version": expected + 1}),
        expect_max_version=expected,
        status=JobStatus.restarting,
        desired_state="running",
    )


async def _rollback(client: httpx.AsyncClient, raw: str) -> httpx.Response:
    return await client.post(
        "/deploy/demo/rollback",
        headers={"Authorization": f"Bearer {raw}", "Idempotency-Key": "key-rollback"},
    )


async def test_a_rollback_overlapping_a_redeploy_cannot_regress_the_allocator(queries, tmp_path):
    """The rollback's whole-blob snapshot CARRIES `max_version` from its read.

    Written unguarded, it lands that stale number over the redeploy that just
    bumped it — and the NEXT honest deploy then re-allocates an image tag that
    is already built, which is the same-tag collision the redeploy CAS exists
    to prevent. Guarded, one of the two is refused instead.
    """
    raw = await _seed(queries, tmp_path)
    uploads = tmp_path / "uploads"
    # The rollback needs a target; seed one alongside the version state.
    row = await queries.get_service_by_name("demo")
    cfg = json.loads(row.config)
    cfg["previous_image"] = "nerdit-app/demo:0"
    await queries.update_job_config(row.id, json.dumps(cfg))

    _gate_the_writer(queries, 2)
    app = _app(queries, uploads)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        deployed, rolled = await asyncio.gather(_deploy(client, raw, "a"), _rollback(client, raw))

    codes = sorted(r.status_code for r in (deployed, rolled))
    assert codes in ([200, 409], [201, 409]), (deployed.text, rolled.text)
    final = json.loads((await queries.get_service_by_name("demo")).config)
    if deployed.status_code == 201:
        # The redeploy won: the allocator advanced and the rollback was refused.
        assert final["max_version"] == 2
        assert rolled.json()["code"] == "deploy.concurrent_redeploy"
    else:
        # The rollback won: the allocator is untouched (a rollback never moves
        # it) and the redeploy was refused rather than silently discarded.
        assert final["max_version"] == 1
        assert final["image"] == "nerdit-app/demo:0"
        assert deployed.json()["code"] == "deploy.concurrent_redeploy"


async def test_a_redeploy_cannot_pop_a_cutover_marker_armed_under_it(queries, tmp_path):
    """`CutoverManager._arm` commits the marker, releases the write lock and
    awaits before it registers its task, so a redeploy that already passed the
    route's in-flight 409 can take the lock next and pop a LIVE marker — the
    verify then probes a superseded row. The CAS covers the marker too."""
    raw = await _seed(queries, tmp_path)
    app = _app(queries, tmp_path / "uploads")
    row = await queries.get_service_by_name("demo")

    inner = queries.update_service_config_guarded
    armed = asyncio.Event()

    async def arm_then_write(*args, **kwargs):
        if not armed.is_set():
            cfg = json.loads((await queries.get_service_by_name("demo")).config)
            cfg["cutover_pending"] = {"version": 1, "blue": "c1", "blue_port": None}
            await queries.update_job_config(row.id, json.dumps(cfg))
            armed.set()
        return await inner(*args, **kwargs)

    queries.update_service_config_guarded = arm_then_write
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await _deploy(client, raw, "a")

    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "deploy.concurrent_redeploy"
    final = json.loads((await queries.get_service_by_name("demo")).config)
    assert final["cutover_pending"]["version"] == 1  # still armed for the verify
    assert final["max_version"] == 1


async def test_a_redeploy_over_a_stale_cutover_marker_still_lands(queries, tmp_path):
    """A marker the caller ALREADY SAW belongs to an abandoned verify (the
    route's in-flight 409 cleared it), so the guard must not refuse it — or a
    wedged marker would block every redeploy of that app forever."""
    raw = await _seed(queries, tmp_path)
    row = await queries.get_service_by_name("demo")
    cfg = json.loads(row.config)
    cfg["cutover_pending"] = {"version": 1, "blue": "c1", "blue_port": None}
    await queries.update_job_config(row.id, json.dumps(cfg))

    app = _app(queries, tmp_path / "uploads")
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await _deploy(client, raw, "a")

    assert resp.status_code == 201, resp.text
    final = json.loads((await queries.get_service_by_name("demo")).config)
    assert "cutover_pending" not in final


async def test_sequential_redeploys_do_not_leak_the_superseded_context(queries, tmp_path):
    """A redeploy that supersedes a not-yet-built generation removes its context.

    The CAS only refuses a redeploy that reads the SAME version as another in
    flight. Two redeploys that serialize — the second reading after the first
    committed — both get a 201 with sequential versions, and the second
    overwrites the first's ``build_context_dir`` in the row. The controller's
    builder only ever cleans the CURRENT row's context, so the superseded
    generation's extracted tree would leak on disk with nothing referencing it
    (surfaced by the §7 live run: two concurrent ZIP redeploys, both 201, one
    cutover, the loser's tree left behind). No controller runs in this harness,
    so nothing else could clean it — exactly what makes the leak observable.
    """
    raw = await _seed(queries, tmp_path)
    uploads = tmp_path / "uploads"
    app = _app(queries, uploads)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        first = await _deploy(client, raw, "a")
        second = await _deploy(client, raw, "b")

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text

    row = await queries.get_service_by_name("demo")
    cfg = json.loads(row.config)
    # The two redeploys took sequential versions — neither was silently discarded.
    assert cfg["build_version"] == 3
    assert cfg["image"] == "nerdit-app/demo:3"

    # Only the latest generation's context survives: the superseded one is gone,
    # not orphaned beside it.
    survivors = sorted(p for p in uploads.iterdir() if p.is_dir())
    assert survivors == [Path(cfg["build_context_dir"])], survivors
