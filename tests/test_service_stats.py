"""Tests for ``GET /api/services/{ident}/stats`` + the runtime sample (P24 WP4).

Three layers:

* a pure unit table over the ``core/runtime/docker.py`` derivation helpers — in
  particular the first-sample zero system-CPU delta, which must yield
  ``cpu_pct = None`` and never a ``ZeroDivisionError``;
* a pattern-2 route harness (``httpx.ASGITransport`` over a real in-memory DB
  under the full middleware stack) pinning the tri-state posture, the TTL
  cache, and the "any authenticated principal" read; and
* a guard that both ``tests/conftest.py`` mock fixtures carry an EXPLICIT
  ``stats`` stub (the P13 ``inspect_state`` AsyncMock-auto-attribute lesson).
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from nerdit.config.settings import NerditSettings
from nerdit.core.runtime.docker import (
    _derive_cpu_pct,
    _derive_memory,
    _derive_network,
)
from nerdit.core.runtime.protocol import ContainerStats
from nerdit.core.runtime.stub import StubRuntime
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.services import router as services_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries

LEGACY = "legacy-admin-token"
OWNER_RAW = "owner-raw-token"
OTHER_RAW = "other-raw-token"


# --- pure unit table over the docker derivation helpers ----------------------


def test_first_sample_zero_system_delta_yields_none_not_zero_division():
    """The first sample reports ``precpu_stats.system_cpu_usage == 0``.

    Docker's own value for "no previous reading". The ratio is undefined, so
    the answer is ``None`` — an unmeasurable container, not an idle one — and
    critically the derivation must not divide by the zero delta.
    """
    raw = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 0},
            "system_cpu_usage": 0,
            "online_cpus": 4,
        },
        "precpu_stats": {"cpu_usage": {"total_usage": 0}, "system_cpu_usage": 0},
    }
    assert _derive_cpu_pct(raw) is None


def test_cgroup_v2_missing_system_cpu_usage_yields_none():
    """cgroup v2 omits ``system_cpu_usage`` entirely — still ``None``, still no raise."""
    raw = {
        "cpu_stats": {"cpu_usage": {"total_usage": 5_000_000}, "online_cpus": 2},
        "precpu_stats": {"cpu_usage": {"total_usage": 1_000_000}},
    }
    assert _derive_cpu_pct(raw) is None


def test_cpu_pct_follows_the_docker_cli_convention():
    """1/8 of system time on a 2-core host = 25.0 (100.0 == one saturated core)."""
    raw = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 2_000_000},
            "system_cpu_usage": 16_000_000,
            "online_cpus": 2,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 1_000_000},
            "system_cpu_usage": 8_000_000,
        },
    }
    assert _derive_cpu_pct(raw) == 25.0


def test_negative_container_delta_is_a_counter_reset_not_a_negative_percent():
    raw = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 1_000},
            "system_cpu_usage": 16_000_000,
            "online_cpus": 1,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 9_000_000},
            "system_cpu_usage": 8_000_000,
        },
    }
    assert _derive_cpu_pct(raw) is None


def test_memory_excludes_the_page_cache():
    raw = {
        "memory_stats": {
            "usage": 300,
            "limit": 1000,
            "stats": {"inactive_file": 100},
        }
    }
    assert _derive_memory(raw) == (200, 1000)


def test_cgroup_v1_prefers_total_inactive_file():
    """The docker CLI subtracts ``total_inactive_file`` when it is present.

    A cgroup-v1 payload carries BOTH keys: ``total_inactive_file`` is the
    hierarchy total docker uses, while the bare ``inactive_file`` is this
    cgroup's own slice. Preferring the bare one under-subtracts and over-reports
    the working set — the docstring said v1-first, the code did v2-first.
    """
    raw = {
        "memory_stats": {
            "usage": 300,
            "limit": 1000,
            "stats": {"inactive_file": 40, "total_inactive_file": 100},
        }
    }
    assert _derive_memory(raw) == (200, 1000)


def test_cgroup_v2_falls_back_to_inactive_file():
    """A v2 payload has only ``inactive_file`` — the fallback still covers it."""
    raw = {"memory_stats": {"usage": 300, "limit": 1000, "stats": {"inactive_file": 100}}}
    assert _derive_memory(raw) == (200, 1000)


def test_memory_without_a_stats_block_reports_the_raw_usage():
    """An over-report is still information; ``None`` would not be."""
    assert _derive_memory({"memory_stats": {"usage": 300, "limit": 1000}}) == (300, 1000)


def test_network_is_summed_over_interfaces_and_none_without_a_block():
    raw = {
        "networks": {"eth0": {"rx_bytes": 1, "tx_bytes": 2}, "eth1": {"rx_bytes": 3, "tx_bytes": 4}}
    }
    assert _derive_network(raw) == (4, 6)
    # host networking / networking disabled ⇒ genuinely unknown, not zero.
    assert _derive_network({}) == (None, None)


# --- route harness -----------------------------------------------------------

_TOKENS = {
    "owner": ApiToken(
        id="tok-owner",
        name="owner",
        token_hash=hash_token(OWNER_RAW),
        role=TokenRole.submitter,
    ),
    "other": ApiToken(
        id="tok-other",
        name="other",
        token_hash=hash_token(OTHER_RAW),
        role=TokenRole.readonly,
    ),
}


class _CountingRuntime:
    """Minimal ``ContainerRuntime`` stand-in that counts ``stats`` calls."""

    def __init__(self, sample: ContainerStats | None) -> None:
        self.sample = sample
        self.calls = 0

    async def stats(self, container_id: str) -> ContainerStats | None:
        self.calls += 1
        return self.sample


@pytest_asyncio.fixture
async def harness():
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for tok in _TOKENS.values():
        await queries.create_api_token(tok)

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router, prefix="/api")
    app.state.queries = queries
    app.state.settings = NerditSettings()
    app.state.runtime = StubRuntime()
    app.add_middleware(
        ScopedTokenAuthMiddleware,
        token=LEGACY,
        get_queries=lambda: queries,
    )
    app.add_middleware(RequestIdMiddleware)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield client, queries, app
    finally:
        await client.aclose()
        await db.close()


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


async def _seed_service(queries, *, container_id="cid-1", **over) -> Job:
    fields = dict(
        service_name="my-app",
        name="my-app",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        submitted_by_token="tok-owner",
        container_id=container_id,
        config=json.dumps({"image": "nerdit-app/my-app:1", "port": 8000}),
    )
    fields.update(over)
    return await queries.create_job(Job(**fields))


@pytest.mark.asyncio
async def test_stub_runtime_reports_unavailable_never_zeroes(harness):
    """A Docker-less daemon answers ``{stats: null, available: false}``."""
    client, queries, _ = harness
    await _seed_service(queries)

    resp = await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is False
    assert body["stats"] is None
    assert body["service_name"] == "my-app"
    assert body["container_id"] == "cid-1"
    # The whole point of the tri-state: no synthesised zeroes anywhere.
    assert "0" not in json.dumps(body["stats"])


@pytest.mark.asyncio
async def test_no_container_reports_unavailable_without_touching_the_runtime(harness):
    client, queries, app = harness
    runtime = _CountingRuntime(ContainerStats(1.0, 1, 2, 3, 4, 5))
    app.state.runtime = runtime
    await _seed_service(queries, container_id=None, status=JobStatus.stopped)

    resp = await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is False
    assert body["container_id"] is None
    assert runtime.calls == 0


@pytest.mark.asyncio
async def test_sample_is_projected_with_derived_mem_pct(harness):
    client, queries, app = harness
    app.state.runtime = _CountingRuntime(
        ContainerStats(
            cpu_pct=12.5,
            mem_used_bytes=256,
            mem_limit_bytes=1024,
            net_rx_bytes=7,
            net_tx_bytes=8,
            pids=3,
        )
    )
    await _seed_service(queries)

    resp = await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert body["stats"] == {
        "cpu_pct": 12.5,
        "mem_used_bytes": 256,
        "mem_limit_bytes": 1024,
        "mem_pct": 25.0,
        "net_rx_bytes": 7,
        "net_tx_bytes": 8,
        "pids": 3,
    }
    assert body["sampled_at"] is not None
    assert body["cached"] is False


@pytest.mark.asyncio
async def test_a_null_cpu_pct_survives_the_projection_as_null(harness):
    """An unmeasurable CPU delta must reach the client as ``null``, not ``0.0``."""
    client, queries, app = harness
    app.state.runtime = _CountingRuntime(
        ContainerStats(
            cpu_pct=None,
            mem_used_bytes=None,
            mem_limit_bytes=None,
            net_rx_bytes=None,
            net_tx_bytes=None,
            pids=None,
        )
    )
    await _seed_service(queries)

    body = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
    assert body["available"] is True
    assert body["stats"]["cpu_pct"] is None
    assert body["stats"]["mem_pct"] is None


@pytest.mark.asyncio
async def test_ttl_cache_coalesces_repeated_reads(harness):
    """Two reads inside the TTL cost exactly one docker sample."""
    client, queries, app = harness
    runtime = _CountingRuntime(ContainerStats(1.0, 10, 100, 0, 0, 1))
    app.state.runtime = runtime
    await _seed_service(queries)

    first = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
    second = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
    assert runtime.calls == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["stats"] == first["stats"]


@pytest.mark.asyncio
async def test_concurrent_misses_share_one_sample(harness):
    """The TTL cache cannot coalesce reads that are already in flight.

    N agents polling the same container at once all missed the (still empty)
    cache and each spent its own ~1-2 s blocking docker call — N slots, N
    samples, one answer. The single-flight makes the second caller await the
    first rather than start a second sample.
    """
    import asyncio

    client, queries, app = harness
    gate = asyncio.Event()

    class _SlowRuntime(_CountingRuntime):
        async def stats(self, container_id: str):
            self.calls += 1
            await gate.wait()  # hold both requests inside the sample window
            return self.sample

    runtime = _SlowRuntime(ContainerStats(1.0, 10, 100, 0, 0, 1))
    app.state.runtime = runtime
    await _seed_service(queries)

    async def read():
        return (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()

    # Deterministic overlap: the second request only starts once the first is
    # provably parked INSIDE the sample, then gets loop turns to reach its own
    # decision point. Without the single-flight it samples too (calls == 2).
    winner = asyncio.ensure_future(read())
    while runtime.calls == 0:
        await asyncio.sleep(0)
    follower = asyncio.ensure_future(read())
    for _ in range(50):
        await asyncio.sleep(0)
    gate.set()
    first, second = await asyncio.gather(winner, follower)

    assert runtime.calls == 1
    # The winner is a miss. The follower's ``cached`` flag is deliberately NOT
    # asserted: whether it shared the in-flight sample (False) or arrived a few
    # loop turns after the winner populated the cache (True) is scheduler
    # timing — observed flipping under CI's coverage instrumentation — and
    # either way the single-flight contract held (``calls == 1`` above is the
    # load-bearing assertion).
    assert first["cached"] is False
    # Same measurement, byte for byte — not two samples that happen to agree.
    assert first["sampled_at"] == second["sampled_at"]
    assert first["stats"] == second["stats"]
    # And the in-flight slot is released, not leaked.
    assert getattr(app.state, "service_stats_inflight", {}) == {}


@pytest.mark.asyncio
async def test_a_cache_hit_replays_the_sample_wall_clock(harness):
    """``sampled_at`` describes the MEASUREMENT, never the response.

    Stamping ``now`` on a hit told a consumer the numbers were fresh when they
    were up to a TTL old — the one field whose whole job is to say otherwise.
    """
    client, queries, app = harness
    app.state.runtime = _CountingRuntime(ContainerStats(1.0, 10, 100, 0, 0, 1))
    await _seed_service(queries)

    first = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
    second = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()

    assert second["cached"] is True
    assert second["sampled_at"] == first["sampled_at"]


@pytest.mark.asyncio
async def test_the_cache_deadline_is_stamped_after_the_blocking_sample(harness):
    """A ~1-2 s docker sample must not eat the entry's whole 2 s lifetime.

    The deadline used to be built from the monotonic read taken BEFORE the
    sample, so an entry was born mostly-expired and the coalescing the cache
    exists for barely happened.
    """
    import time as _time

    from nerdit.daemon.routes import service_stats as mod

    client, queries, app = harness
    runtime = _CountingRuntime(ContainerStats(1.0, 10, 100, 0, 0, 1))
    app.state.runtime = runtime
    await _seed_service(queries)

    # A clock that jumps a full TTL "inside" the blocking sample.
    ticks = iter([1000.0, 1000.0 + mod._STATS_TTL_S, 1000.0 + mod._STATS_TTL_S])
    real = _time.monotonic
    mod.time.monotonic = lambda: next(ticks, 1000.0 + mod._STATS_TTL_S)  # type: ignore[assignment]
    try:
        await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))
        second = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
    finally:
        mod.time.monotonic = real  # type: ignore[assignment]

    # Entry still alive at the post-sample clock ⇒ one docker call, not two.
    assert runtime.calls == 1
    assert second["cached"] is True


@pytest.mark.asyncio
async def test_unavailable_samples_are_cached_too(harness):
    """The crash-loop case is the one polled hardest — do not re-ask docker each time."""
    client, queries, app = harness
    runtime = _CountingRuntime(None)
    app.state.runtime = runtime
    await _seed_service(queries)

    for _ in range(3):
        body = (await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))).json()
        assert body["available"] is False
    assert runtime.calls == 1


@pytest.mark.asyncio
async def test_read_is_open_to_any_authenticated_principal(harness):
    """Deliberately NOT owner-gated (unlike /diagnose): counters carry no secrets."""
    client, queries, app = harness
    app.state.runtime = _CountingRuntime(ContainerStats(1.0, 1, 2, 3, 4, 5))
    await _seed_service(queries)

    resp = await client.get("/api/services/my-app/stats", headers=_auth(OTHER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["available"] is True


@pytest.mark.asyncio
async def test_unknown_service_is_a_structured_404(harness):
    client, _, _ = harness
    resp = await client.get("/api/services/nope/stats", headers=_auth(OWNER_RAW))
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_a_runtime_that_raises_degrades_to_unavailable(harness):
    class _Boom:
        async def stats(self, container_id: str):
            raise RuntimeError("docker went away")

    client, queries, app = harness
    app.state.runtime = _Boom()
    await _seed_service(queries)

    resp = await client.get("/api/services/my-app/stats", headers=_auth(OWNER_RAW))
    assert resp.status_code == 200, resp.text
    assert resp.json()["available"] is False


# --- CLI rendering -----------------------------------------------------------


class _StatsClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[str] = []

    async def get_service_stats(self, ident):
        self.calls.append(ident)
        return self.payload


@pytest.mark.asyncio
async def test_cli_escapes_every_server_derived_string(monkeypatch, capsys):
    """A hostile service name must not raise ``rich.errors.MarkupError``.

    The bug that has shipped four times: a server value interpolated raw into a
    Rich sink. Every string below is server-derived, so every one goes through
    ``_plain`` — this pins that a bracketed name renders literally instead of
    being parsed as a closing tag.
    """
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    hostile = "[/bold]evil[red]"
    fake = _StatsClient(
        {
            "service_name": hostile,
            "container_id": "cid",
            "available": True,
            "stats": {
                "cpu_pct": 12.5,
                "mem_used_bytes": 256 * 1024 * 1024,
                "mem_limit_bytes": 1024 * 1024 * 1024,
                "mem_pct": 25.0,
                "net_rx_bytes": 1024,
                "net_tx_bytes": 2048,
                "pids": 3,
            },
            "gpus": [{"gpu_id": "[gpu-0]", "utilization_percent": 40, "memory_used_mb": 512}],
            "sampled_at": "2026-08-07T00:00:00+00:00",
            "cached": True,
        }
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    await svc_mod._stats_async(hostile)
    out = capsys.readouterr().out
    assert hostile in out
    assert "[gpu-0]" in out
    assert fake.calls == [hostile]


@pytest.mark.asyncio
async def test_cli_reports_unavailable_rather_than_zeroes(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import services as svc_mod

    fake = _StatsClient(
        {"service_name": "demo", "container_id": None, "available": False, "stats": None}
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)

    await svc_mod._stats_async("demo")
    out = capsys.readouterr().out
    assert "No stats for demo" in out
    assert "0.0%" not in out


def test_cli_formatters_render_unknown_as_a_dash():
    """``None`` is ``-``, never ``0 B`` / ``0.0%`` — the tri-state at the last mile."""
    from nerdit.cli.commands import services as svc_mod

    # One humanizer for every CLI surface, re-exported through the command
    # module it is used from.
    assert svc_mod.fmt_bytes(None) == "-"
    assert svc_mod._fmt_pct(None) == "-"
    assert svc_mod.fmt_bytes(0) == "0 B"
    assert svc_mod.fmt_bytes(1536) == "1.5 KiB"
    assert svc_mod._fmt_pct(12.34) == "12.3%"


# --- conftest fixture guard --------------------------------------------------


def test_mock_runtime_fixture_has_an_explicit_stats_stub(mock_runtime):
    """An auto-satisfied AsyncMock attribute would return a coroutine-of-MagicMock.

    The P13 ``inspect_state`` lesson: every Protocol method the routes call must
    be stubbed EXPLICITLY on the shared fixture, or the assertion under test
    silently passes against a mock's child mock.
    """
    assert "stats" in mock_runtime.__dict__ or hasattr(type(mock_runtime), "stats")
    assert mock_runtime.stats.return_value is None


def test_mock_docker_fixture_has_an_explicit_container_stats_stub(mock_docker):
    """``DockerRuntime.stats`` indexes the payload — it must be a real dict.

    An auto-satisfied MagicMock attribute would hand back a child mock, which
    is not a ``dict``, so every test on this fixture would silently take the
    "unavailable" branch instead of exercising the derivation.
    """
    container = mock_docker.containers.get.return_value
    payload = container.stats(stream=False)
    assert isinstance(payload, dict)
    assert _derive_cpu_pct(payload) == 25.0


# --- DockerRuntime.stats end to end ------------------------------------------


class _FakeStatsContainer:
    def __init__(self, payload):
        self.payload = payload
        self.stream_args: list[object] = []

    def stats(self, stream=True):
        self.stream_args.append(stream)
        return self.payload


class _FakeStatsClient:
    def __init__(self, container):
        self.containers = self  # ``client.containers.get`` in one object
        self._container = container

    def get(self, container_id):
        return self._container


@pytest.mark.asyncio
async def test_docker_runtime_stats_takes_a_single_bounded_sample(mock_docker):
    """``stream=False`` is the contract: one request, never a follow stream (D-P24-10)."""
    from nerdit.core.runtime.docker import DockerRuntime

    container = _FakeStatsContainer(
        {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 2_000_000},
                "system_cpu_usage": 16_000_000,
                "online_cpus": 2,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 1_000_000},
                "system_cpu_usage": 8_000_000,
            },
            "memory_stats": {"usage": 300, "limit": 1000, "stats": {"inactive_file": 100}},
            "networks": {"eth0": {"rx_bytes": 11, "tx_bytes": 22}},
            "pids_stats": {"current": 4},
        }
    )
    rt = DockerRuntime(client=_FakeStatsClient(container))
    sample = await rt.stats("cid-1")
    assert container.stream_args == [False]
    assert sample == ContainerStats(
        cpu_pct=25.0,
        mem_used_bytes=200,
        mem_limit_bytes=1000,
        net_rx_bytes=11,
        net_tx_bytes=22,
        pids=4,
    )


@pytest.mark.asyncio
async def test_docker_runtime_stats_returns_none_on_a_malformed_payload(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    rt = DockerRuntime(client=_FakeStatsClient(_FakeStatsContainer("not-a-dict")))
    assert await rt.stats("cid-1") is None


@pytest.mark.asyncio
async def test_stub_runtime_stats_is_none_not_a_zeroed_sample():
    assert await StubRuntime().stats("cid-1") is None
