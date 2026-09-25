"""Keep bound app identities and audiences authoritative through a live stream."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nerdit.core.link.apps import AppStreamResolver
from nerdit.core.link.mux import APP_ACCESS_HEADER, APP_JOB_HEADER, StreamMux
from nerdit.db.rows import ServicePublicAddress
from tests.link_fake_relay import FIXTURE_NODE_ID
from tests.test_link_app_streams import _seed_endpoint, _seed_service
from tests.test_link_mux import _app_frame, _BytesTransport, _CountingApp, _ctx, _ParkedTransport

HOST = "quiet-lake-abcdefgh23.nerdit.app"


async def _bound(queries):
    job = await _seed_service(queries, "demo")
    await _seed_endpoint(queries, job)
    await queries.set_service_share("demo", "public", job_id=job.id)
    await queries.set_service_public_address(
        ServicePublicAddress(
            node_id=FIXTURE_NODE_ID,
            project_id=job.project_id,
            job_id=job.id,
            service_name="demo",
            slug=HOST.removesuffix(".nerdit.app"),
        ),
        activate=True,
    )
    resolver = AppStreamResolver(
        queries, node_id=FIXTURE_NODE_ID, slug="gpu-box", nodes_base_domain="nodes.test"
    )
    return job, resolver


def _headers(job_id, access):
    return [("x-forwarded-host", HOST), (APP_JOB_HEADER, job_id), (APP_ACCESS_HEADER, access)]


@pytest.mark.parametrize("change", ["private", "unshare", "delete"])
async def test_quiet_stream_revocation_uses_live_job_and_audience(queries, monkeypatch, change):
    monkeypatch.setattr("nerdit.core.link.mux.APP_RECHECK_INTERVAL_S", 0.01)
    job, resolver = await _bound(queries)
    transport = _ParkedTransport()
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()),
        resolve_app=resolver.resolve,
        app_transport=transport,
    )
    mux = StreamMux(ctx)
    try:
        for access in ("public", "private"):
            await mux.handle(
                _app_frame(
                    stream_id=access, method="GET", path="/quiet", headers=_headers(job.id, access)
                )
            )
        await sink.wait(lambda: len(sink.of("stream_response_head")) == 2)
        assert all(head["status"] == 200 for head in sink.of("stream_response_head"))
        if change == "private":
            await queries.set_service_share("demo", "private", job_id=job.id)
        elif change == "unshare":
            await queries.delete_service_share("demo")
        else:
            await queries.delete_service_checked(job.id)
        await sink.wait(lambda: any(f["stream_id"] == "public" for f in sink.of("stream_error")))
        if change == "private":
            await asyncio.sleep(0.03)
            assert mux.active_streams() == 1
            assert not any(f["stream_id"] == "private" for f in sink.of("stream_error"))
            await queries.delete_service_share("demo")
        await sink.wait(lambda: mux.active_streams() == 0)
        assert all(stream.closed.is_set() for stream in transport.streams)
        assert not sink.of("stream_response_end")
    finally:
        await mux.aclose()


@pytest.mark.parametrize(
    "carrier", ["x-nerdit-app", APP_JOB_HEADER, APP_ACCESS_HEADER, "x-forwarded-host"]
)
async def test_duplicate_routing_carrier_fails_before_dial(queries, carrier):
    job, resolver = await _bound(queries)
    transport = _BytesTransport(200, [], [b"never"])
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()),
        resolve_app=resolver.resolve,
        app_transport=transport,
    )
    mux = StreamMux(ctx)
    headers = _headers(job.id, "public")
    headers.append((carrier.upper(), "duplicate"))
    try:
        await mux.handle(_app_frame(method="GET", headers=headers))
        await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
        assert sink.of("stream_response_head")[0]["status"] == 404
        assert transport.requests == []
    finally:
        await mux.aclose()


@pytest.mark.parametrize("host", [HOST, "demo--gpu-box.nodes.test"])
async def test_origin_receives_validated_authority_without_internal_carriers(queries, host):
    job, resolver = await _bound(queries)
    transport = _BytesTransport(200, [], [b"ok"])
    ctx, sink = _ctx(
        httpx.ASGITransport(app=_CountingApp()),
        resolve_app=resolver.resolve,
        app_transport=transport,
    )
    mux = StreamMux(ctx)
    try:
        await mux.handle(
            _app_frame(
                method="GET",
                headers=[
                    ("x-forwarded-host", host),
                    (APP_JOB_HEADER, job.id),
                    (APP_ACCESS_HEADER, "public"),
                    ("host", "forged.example"),
                    ("Forwarded", "host=forged.example"),
                    ("x-nerdit-cloud-control", "public-address"),
                ],
            )
        )
        await sink.wait_types("stream_response_head", "stream_response_body", "stream_response_end")
        assert sink.of("stream_response_head")[0]["status"] == 200
        request = transport.requests[0]
        assert request.headers["host"] == host
        assert request.headers["x-forwarded-host"] == host
        for header in (
            "authorization",
            "x-nerdit-role",
            "x-nerdit-app",
            APP_JOB_HEADER,
            APP_ACCESS_HEADER,
            "forwarded",
            "x-nerdit-cloud-control",
        ):
            assert header not in request.headers
    finally:
        await mux.aclose()
