"""Check incarnation and audience again on acquired TCP connections before sending bytes."""

from __future__ import annotations

import asyncio

import httpx

from nerdit.core.link import mux as mux_module
from nerdit.core.link.apps import AppStreamResolver
from nerdit.core.link.mux import StreamMux
from tests.link_fake_relay import FIXTURE_NODE_ID
from tests.test_link_app_streams import _seed_service
from tests.test_link_mux import _app_frame, _ctx

HOST = "demo--gpu-box.nodes.test"


async def test_reallocated_port_receives_no_bytes_from_an_already_admitted_job(  # noqa: PLR0915 - one TCP race and its cleanup
    queries, monkeypatch
) -> None:
    """A delayed native TCP dial must not send the old app's cookies to its replacement."""
    monkeypatch.setattr(mux_module, "APP_RECHECK_INTERVAL_S", 60)
    received: list[bytes] = []
    peer_done = asyncio.Event()

    async def peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                request = await reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError as exc:
                request = exc.partial
            received.append(request)
            if request:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nnew")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_done.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    job = await _seed_service(queries, "demo")
    await queries.acquire_service_port("demo", job.id, 8000, (port, port))
    await queries.set_service_share("demo", "public", job_id=job.id)
    resolver = AppStreamResolver(
        queries, node_id=FIXTURE_NODE_ID, slug="gpu-box", nodes_base_domain="nodes.test"
    )
    context, sink = _ctx(
        httpx.MockTransport(lambda request: httpx.Response(500)), resolve_app=resolver.resolve
    )
    mux = StreamMux(context)
    # Keep the production HTTPX client, pool and HTTP/1 serializer. Pause only
    # before the real socket acquisition, where a delayed connect can race teardown.
    backend = mux._app_client._transport._pool._network_backend
    connect = backend.connect_tcp
    connecting, resume = asyncio.Event(), asyncio.Event()

    async def delayed_connect(*args, **kwargs):
        connecting.set()
        await resume.wait()
        return await connect(*args, **kwargs)

    monkeypatch.setattr(backend, "connect_tcp", delayed_connect)
    try:
        await mux.handle(
            _app_frame(
                method="GET",
                path="/",
                body_b64="",
                headers=[
                    ("x-forwarded-host", HOST),
                    ("x-nerdit-app-job", job.id),
                    ("x-nerdit-app-access", "public"),
                    ("cookie", "app_session=old-private-data"),
                ],
            )
        )
        await asyncio.wait_for(connecting.wait(), 2)
        server.close()
        await server.wait_closed()
        await queries.delete_service_checked(job.id)
        replacement = await _seed_service(queries, "replacement")
        endpoint = await queries.acquire_service_port(
            "replacement", replacement.id, 8000, (port, port)
        )
        assert endpoint.host_port == port
        server = await asyncio.start_server(peer, "127.0.0.1", port)
        resume.set()
        await asyncio.wait_for(peer_done.wait(), 2)
        await sink.wait(lambda: "stream_error" in sink.types())
        assert received == [b""]
        assert sink.body() == b""
    finally:
        resume.set()
        await mux.aclose()
        server.close()
        await server.wait_closed()


async def test_new_request_rechecks_audience_on_a_fresh_connection(  # noqa: PLR0915 - two native requests and TCP cleanup
    queries, monkeypatch
) -> None:
    """A second request uses a fresh socket and sends no bytes after a racing downgrade."""
    monkeypatch.setattr(mux_module, "APP_RECHECK_INTERVAL_S", 60)
    received: list[bytes] = []
    connections = 0
    peer_done = asyncio.Event()

    async def peer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        try:
            while True:
                try:
                    request = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError as exc:
                    assert not exc.partial
                    break
                received.append(request)
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            peer_done.set()

    server = await asyncio.start_server(peer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    job = await _seed_service(queries, "demo")
    await queries.acquire_service_port("demo", job.id, 8000, (port, port))
    await queries.set_service_share("demo", "public", job_id=job.id)
    resolver = AppStreamResolver(
        queries, node_id=FIXTURE_NODE_ID, slug="gpu-box", nodes_base_domain="nodes.test"
    )
    admitted, resume = asyncio.Event(), asyncio.Event()
    pause_next_admission = False

    async def resolve(*args):
        nonlocal pause_next_admission
        target = await resolver.resolve(*args)
        if pause_next_admission:
            pause_next_admission = False
            admitted.set()
            await resume.wait()
        return target

    context, sink = _ctx(
        httpx.MockTransport(lambda request: httpx.Response(500)), resolve_app=resolve
    )
    mux = StreamMux(context)
    headers = [
        ("x-forwarded-host", HOST),
        ("x-nerdit-app-job", job.id),
        ("x-nerdit-app-access", "public"),
    ]
    try:
        await mux.handle(
            _app_frame(stream_id="first", method="GET", path="/", body_b64="", headers=headers)
        )
        await sink.wait(lambda: "stream_response_end" in sink.types())
        assert sink.body() == b"ok" and len(received) == 1
        pause_next_admission = True
        await mux.handle(
            _app_frame(stream_id="second", method="GET", path="/", body_b64="", headers=headers)
        )
        await asyncio.wait_for(admitted.wait(), 2)
        await queries.set_service_share("demo", "private", job_id=job.id)
        resume.set()
        await sink.wait(lambda: "stream_error" in sink.types())
        assert len(received) == 1
        assert connections == 2
    finally:
        resume.set()
        await mux.aclose()
        server.close()
        await server.wait_closed()
        await asyncio.wait_for(peer_done.wait(), 2)
