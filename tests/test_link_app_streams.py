"""Test hosted app resolution against real database state and bootstrap wiring.

Only shared, launched app rows with valid labels resolve; models, databases and
other refusals all return None. Bootstrap enables resolution only with a linked
slug and known nodes_base_domain, and supplies it to every MuxContext.
Public entitlement starts false until a cloud push, preventing premature public
sharing. Stream behavior itself is covered in test_link_mux.py.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from nerdit.config.settings import DaemonSettings, LinkSettings
from nerdit.core.link.apps import AppStreamResolver
from nerdit.core.link.client import AppTarget, dial_relay
from nerdit.core.link.identity import NodeIdentity
from nerdit.core.link.manager import LinkManager
from nerdit.daemon.bootstrap import build_link_manager
from nerdit.db.models import Job, JobKind, JobStatus
from tests.link_fake_relay import FIXTURE_NODE_ID, FakeRelay
from tests.test_link_manager import Clockwork, FakeClock, _RecordingMux, _settings

SLUG = "gpu-box"
DOMAIN = "nodes.test"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _jid(queries: Any, name: str) -> str:
    """Id of the live service named ``name`` — the row the route would have authorized."""
    job = await queries.get_service_by_name(name)
    assert job is not None, name
    return job.id


async def _seed_service(queries: Any, name: str, *, kind: JobKind = JobKind.service) -> Job:
    """Create a workload row of ``kind`` named ``name`` (no endpoint yet)."""
    job = Job(
        name=name,
        kind=kind,
        service_name=name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        config='{"image": "demo:latest", "port": 8000}',
    )
    return await queries.create_job(job)


async def _seed_endpoint(queries: Any, job: Job, *, active_port: int | None = None) -> int:
    endpoint = await queries.acquire_service_port(job.service_name, job.id, 8000, (14300, 14400))
    if active_port is not None:
        await queries.set_endpoint_active_port(job.service_name, active_port)
        return active_port
    return endpoint.host_port


def _resolver(queries: Any) -> AppStreamResolver:
    return AppStreamResolver(queries, slug=SLUG, nodes_base_domain=DOMAIN)


async def _force_share_row(db: Any, service_name: str) -> None:
    """Plant a share row the API can no longer create (PR review, P26 WP-H).

    ``set_service_share`` now inserts only ``WHERE EXISTS`` a live
    ``kind='service'`` job — that is the TOCTOU fix, pinned in
    ``tests/test_service_shares_db.py``. The resolver's own kind/job checks are
    defence in depth for rows that predate a kind change or were written by
    hand, so proving them needs a row written by hand.
    """
    await db.conn.execute(
        "INSERT INTO service_shares (service_name, access) VALUES (?, 'private')",
        (service_name,),
    )
    await db.conn.commit()


class _SpyQueries:
    """Counts every read, so "refused without a DB call" is provable."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_service_share(self, service_name: str) -> None:
        self.calls.append("share")

    async def get_service_by_name(self, service_name: str) -> None:  # pragma: no cover
        self.calls.append("job")

    async def get_service_endpoint(self, service_name: str) -> None:  # pragma: no cover
        self.calls.append("endpoint")


# ---------------------------------------------------------------------------
# 1. AppStreamResolver against a real database
# ---------------------------------------------------------------------------


async def test_a_shared_running_service_resolves_to_its_live_port_and_hosted_host(
    queries,  # noqa: ANN001
) -> None:
    """The happy path, and the D-P24-4b detail that matters.

    ``live_port`` — not ``host_port`` — is dialled: during a health-gated
    cutover the serving container is on the transient port, and handing the
    browser the reserved one would serve the generation the daemon has already
    stopped advertising.
    """
    job = await _seed_service(queries, "demo")
    await _seed_endpoint(queries, job, active_port=14999)
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    target = await _resolver(queries).resolve("demo")

    assert target == AppTarget(service_name="demo", port=14999, host="demo--gpu-box.nodes.test")


async def test_an_ordinary_endpoint_resolves_to_its_reserved_port(queries) -> None:  # noqa: ANN001
    job = await _seed_service(queries, "demo")
    reserved = await _seed_endpoint(queries, job)
    await queries.set_service_share("demo", "public", job_id=await _jid(queries, "demo"))

    target = await _resolver(queries).resolve("demo")

    assert target is not None
    assert target.port == reserved


async def test_a_service_without_a_share_row_is_refused(queries) -> None:  # noqa: ANN001
    """The fail-closed default: a deployed, running, routed app is NOT exposed
    through the tunnel until somebody shares it."""
    job = await _seed_service(queries, "demo")
    await _seed_endpoint(queries, job)

    assert await _resolver(queries).resolve("demo") is None


async def test_an_unshare_closes_the_path_on_the_next_resolve(queries) -> None:  # noqa: ANN001
    """D-P26-H1: no cache, no TTL, no invalidation protocol."""
    job = await _seed_service(queries, "demo")
    await _seed_endpoint(queries, job)
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))
    resolver = _resolver(queries)
    assert await resolver.resolve("demo") is not None

    await queries.delete_service_share("demo")

    assert await resolver.resolve("demo") is None


@pytest.mark.parametrize("kind", [JobKind.model, JobKind.database])
async def test_a_model_or_database_is_refused_even_with_a_share_row(
    db,  # noqa: ANN001
    queries,  # noqa: ANN001
    kind: JobKind,
) -> None:
    """Defence in depth: the share route refuses to write such a row at all
    (422 ``share.kind_unsupported``) and the upsert refuses too, so reaching
    this branch means the row predates a kind change or was written by hand —
    hence the hand-written row. It still must not serve: models and databases
    have no HTTP app semantics on the hosted path."""
    job = await _seed_service(queries, "demo", kind=kind)
    await _seed_endpoint(queries, job)
    await _force_share_row(db, "demo")

    assert await _resolver(queries).resolve("demo") is None


async def test_a_share_row_without_a_job_row_is_refused(db, queries) -> None:  # noqa: ANN001
    """An orphan row can no longer be created through the API — the upsert's
    ``WHERE EXISTS`` is what closed that window — but the resolver must still
    refuse one, because a stale row is precisely the state that would expose a
    later app deployed under the same name."""
    await _force_share_row(db, "ghost")

    assert await _resolver(queries).resolve("ghost") is None


async def test_a_shared_service_with_no_endpoint_is_refused(queries) -> None:  # noqa: ANN001
    """Shared but never launched: there is no port to dial, and the answer is
    the same 404 as every other refusal."""
    await _seed_service(queries, "demo")
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))

    assert await _resolver(queries).resolve("demo") is None


@pytest.mark.parametrize(
    "value",
    ["A.b", "x" * 64, "--x", "-demo", "demo-", "demo/../etc", "", "de mo", "DEMO"],
)
async def test_a_malformed_name_is_refused_without_touching_the_database(value: str) -> None:
    """The cheap gate runs first — the header value is attacker-chosen input.

    Counted, not assumed: a spy whose every method appends to a list proves the
    resolver returned before any read.
    """
    spy = _SpyQueries()

    assert await AppStreamResolver(spy, slug=SLUG, nodes_base_domain=DOMAIN).resolve(value) is None
    assert spy.calls == []


async def test_a_well_formed_name_does_reach_the_share_read() -> None:
    """The falsifier for the test above: the gate is a filter, not a wall."""
    spy = _SpyQueries()

    assert await AppStreamResolver(spy, slug=SLUG, nodes_base_domain=DOMAIN).resolve("demo") is None
    assert spy.calls == ["share"]


async def test_the_resolver_never_logs_the_requested_name_when_it_is_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rejected value is unbounded, unvalidated, attacker-chosen input; it is
    the one thing this module must not echo into the log."""
    caplog.set_level(logging.DEBUG, logger="nerdit.link.apps")
    spy = _SpyQueries()

    await AppStreamResolver(spy, slug=SLUG, nodes_base_domain=DOMAIN).resolve("SECRETVALUE/../x")

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged
    assert "SECRETVALUE" not in logged


# ---------------------------------------------------------------------------
# 2. bootstrap wiring
# ---------------------------------------------------------------------------


def _boot_settings(link: LinkSettings, tmp_path: object) -> SimpleNamespace:
    return SimpleNamespace(
        link=link,
        data_dir=str(tmp_path),
        daemon=DaemonSettings(host="127.0.0.1", port=9321),
    )


def _linked(**kw: object) -> LinkSettings:
    return LinkSettings(
        enabled=True,
        relay_url="wss://relay.example.test/link",
        node_id=FIXTURE_NODE_ID,
        slug=SLUG,
        **kw,  # type: ignore[arg-type]
    )


async def test_a_linked_node_with_a_known_domain_gets_a_working_resolver(
    tmp_path,  # noqa: ANN001
    queries,  # noqa: ANN001
) -> None:
    """Wiring proved end to end, not by attribute-peeking: the callable the
    bootstrap installed is driven against the real database."""
    job = await _seed_service(queries, "demo")
    await _seed_endpoint(queries, job, active_port=14998)
    await queries.set_service_share("demo", "private", job_id=await _jid(queries, "demo"))
    settings = _boot_settings(_linked(nodes_base_domain=DOMAIN), tmp_path)

    manager = await build_link_manager(settings, AsyncMock(), queries=queries)  # type: ignore[arg-type]

    assert manager is not None
    resolve = manager._resolve_app
    assert resolve is not None
    assert await resolve("demo") == AppTarget("demo", 14998, f"demo--{SLUG}.{DOMAIN}")
    assert await resolve("other") is None


async def test_a_node_without_a_known_domain_serves_no_app_stream(
    tmp_path,  # noqa: ANN001
    queries,  # noqa: ANN001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pre-P26 linked node: it has a slug but has never learned the cloud's
    base domain, so there is no authority to rewrite ``Host`` to. The tunnel
    still serves the daemon API; app streams are 404 (fail-closed)."""
    caplog.set_level(logging.INFO, logger="nerdit.daemon.bootstrap")
    settings = _boot_settings(_linked(), tmp_path)

    manager = await build_link_manager(settings, AsyncMock(), queries=queries)  # type: ignore[arg-type]

    assert manager is not None
    assert manager._resolve_app is None
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "nerdit link refresh" in logged


async def test_without_queries_the_resolver_is_never_installed(tmp_path) -> None:  # noqa: ANN001
    """The default call shape (every pre-P26 caller and test) stays inert."""
    settings = _boot_settings(_linked(nodes_base_domain=DOMAIN), tmp_path)

    manager = await build_link_manager(settings, AsyncMock())  # type: ignore[arg-type]

    assert manager is not None
    assert manager._resolve_app is None


# ---------------------------------------------------------------------------
# 3. manager seam — MuxContext and LinkStatus
# ---------------------------------------------------------------------------


@pytest.fixture
def identity() -> NodeIdentity:
    raw = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    return NodeIdentity(base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii"))


async def test_every_mux_context_carries_the_resolver(identity: NodeIdentity) -> None:
    """The manager is the only thing that builds a ``MuxContext``; if the
    resolver did not ride along, the share table would be invisible to the mux
    and every hosted URL would 404 while the daemon reported it as ready."""
    _RecordingMux.instances = []
    clock = FakeClock()
    clockwork = Clockwork(clock)

    async def resolve(name: str) -> AppTarget | None:  # pragma: no cover - identity only
        return None

    async with FakeRelay(verifier=identity.verifier, clock=clock) as relay:
        manager = LinkManager(
            _settings(),
            identity,
            node_id=FIXTURE_NODE_ID,
            node_name="dev-node",
            daemon_version="0.1.0",
            loopback_base_url="http://127.0.0.1:9321",
            connector=lambda: dial_relay(relay.url),
            mux_factory=_RecordingMux,
            resolve_app=resolve,
            clock=clock,
            sleep=clockwork.sleep,
            monotonic=clock.monotonic,
            rng=lambda: 0.5,
        )
        try:
            await manager.start()
            deadline = 2000
            while manager.status().state != "connected" and deadline:
                await asyncio.sleep(0.001)
                deadline -= 1

            assert len(_RecordingMux.instances) == 1
            assert _RecordingMux.instances[0].ctx.resolve_app is resolve
        finally:
            await manager.stop()


def test_hosted_public_entitlement_is_false_until_the_cloud_pushes(
    identity: NodeIdentity,
) -> None:
    """Boot state is the fail-closed one (S5, P32 D-P32-3).

    Nothing is persisted, so a fresh manager — and every restart — projects
    ``False`` and ``PUT share access=public`` refuses until the cloud asserts
    over the tunnel. The push itself is pinned in ``tests/test_link_manager.py``
    and ``tests/test_link_entitlement_route.py``; what is pinned HERE is that
    the seam's default never drifted open.
    """
    manager = LinkManager(
        _settings(),
        identity,
        node_id=FIXTURE_NODE_ID,
        node_name="dev-node",
        daemon_version="0.1.0",
        loopback_base_url="http://127.0.0.1:9321",
    )

    assert manager.status().hosted_public_entitled is False
    assert manager.status().hosted_public_entitled_at is None
