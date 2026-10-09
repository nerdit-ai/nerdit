"""The gateway's ASGI app and its second, in-process uvicorn listener.

Not mounted on the daemon app: it listens on its own port, on the models bridge
bind address only (Linux `172.17.0.1`; loopback where Docker Desktop forwards
`host.docker.internal`), never on all interfaces.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator

import uvicorn
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from uvicorn.protocols.http.h11_impl import H11Protocol

from nerdit.core.ai_gateway.proxy import Gateway, authenticate, error, forward

logger = logging.getLogger(__name__)

# Bounds the listener's shutdown when a long stream is still open.
_SHUTDOWN_TIMEOUT_S = 5
# The listener shares the daemon's process, file descriptors and event loop, and
# any container can reach it before authenticating: bound open sockets, requests
# in flight (each holds up to `max_body_bytes`), and the time to send headers.
_MAX_CONNECTIONS = 256
_MAX_IN_FLIGHT = 64
_HEADER_DEADLINE_S = 15.0


def build_gateway_app(gw: Gateway) -> Starlette:
    """`POST /v1/{chat/completions,completions,embeddings}` and `GET /v1/models`; else 404."""

    def _forwarder(path: str):  # noqa: ANN202 - a Starlette endpoint
        async def endpoint(request: Request) -> Response:
            return await forward(gw, request, path)

        return endpoint

    async def models(request: Request) -> Response:
        if await authenticate(gw, request) is None:
            return error(401, "invalid_api_key", "missing or invalid AI gateway key")
        # V1: every alias is visible to every service (`services` is reserved).
        routes = await gw.queries.list_ai_routes()
        return JSONResponse(
            {
                "object": "list",
                "data": [{"id": r.alias, "object": "model", "owned_by": "nerdit"} for r in routes],
            }
        )

    async def http_error(request: Request, exc: Exception) -> Response:
        status = exc.status_code if isinstance(exc, HTTPException) else 500
        code = "not_found" if status == 404 else "method_not_allowed" if status == 405 else "error"
        return error(status, code, f"{request.method} {request.url.path} is not served")

    return Starlette(
        routes=[
            Route("/v1/chat/completions", _forwarder("chat/completions"), methods=["POST"]),
            Route("/v1/completions", _forwarder("completions"), methods=["POST"]),
            Route("/v1/embeddings", _forwarder("embeddings"), methods=["POST"]),
            Route("/v1/models", models, methods=["GET"]),
        ],
        exception_handlers={404: http_error, 405: http_error},
    )


class _Protocol(H11Protocol):
    """h11 with a connection cap and a deadline to complete each request's headers.

    uvicorn only times out idle keep-alive connections after a response; a socket
    that never sends (or never finishes) its headers would otherwise stay open.
    """

    def connection_made(self, transport: asyncio.Transport) -> None:  # type: ignore[override]
        super().connection_made(transport)
        if len(self.connections) > _MAX_CONNECTIONS:
            transport.close()
            return
        self._arm_deadline()

    def on_response_complete(self) -> None:
        super().on_response_complete()
        self._arm_deadline()

    def _arm_deadline(self) -> None:
        self.loop.call_later(_HEADER_DEADLINE_S, self._deadline, self.cycle)

    def _deadline(self, armed: object) -> None:
        # Still no new request since the deadline was armed, and none running: drop it.
        idle = self.cycle is armed and (self.cycle is None or self.cycle.response_complete)
        if idle and not self.transport.is_closing():
            self.transport.close()


class _Server(uvicorn.Server):
    """A uvicorn server that leaves process signals to the daemon's own server."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


class GatewayServer:
    """Runs the gateway app on `host:port` as a task of the daemon's event loop."""

    def __init__(self, gw: Gateway, host: str, port: int) -> None:
        self._gw = gw
        self._host = host
        self._port = port
        self._server = _Server(
            uvicorn.Config(
                build_gateway_app(gw),
                host=host,
                port=port,
                lifespan="off",
                http=_Protocol,
                limit_concurrency=_MAX_IN_FLIGHT,
                log_config=None,  # the daemon already configured logging
                access_log=False,
                timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT_S,
            )
        )
        self._task: asyncio.Task[None] | None = None

    @property
    def listening(self) -> bool:
        """Bound and serving (false before `start`, or after a failed bind)."""
        return self._task is not None

    async def start(self) -> bool:
        """Bind now and serve in the background; `False` when the bind failed.

        The bind happens here, not in the background task, so the caller knows
        whether apps can reach the gateway: a failure (port in use, address
        missing) is logged, never fatal, and reported as "not listening".
        """
        server = self._server
        uvicorn_config = server.config  # a uvicorn Config, not a secret scope
        if not uvicorn_config.loaded:
            uvicorn_config.load()
        server.lifespan = uvicorn_config.lifespan_class(uvicorn_config)
        try:
            await server.startup()
        except SystemExit:
            # uvicorn exits the process on a bind failure; the daemon must not.
            logger.error(
                "AI gateway could not listen on %s:%d (port in use or address missing); "
                "it stays off until the next restart",
                self._host,
                self._port,
            )
            return False
        self._task = asyncio.create_task(self._run())
        return True

    async def _run(self) -> None:
        await self._server.main_loop()
        await self._server.shutdown()
        logger.info("AI gateway stopped on %s:%d", self._host, self._port)

    async def stop(self) -> None:
        """Stop the listener (open streams get a short grace) and close the pool."""
        self._server.should_exit = True
        if self._task is not None:
            await self._task
        await self._gw.client.aclose()
