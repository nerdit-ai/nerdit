"""Forward one authenticated OpenAI-compatible request to the alias's provider.

Only `content-type`/`accept` from the app reach upstream; the provider key is
resolved from the SecretManager per request and sent as the only
`Authorization`. Errors use the OpenAI shape. Never logged: request or
response bodies, the provider key, the virtual key.
"""

from __future__ import annotations

import asyncio
import http.cookiejar
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from nerdit.config.project import AiBindingConfig
from nerdit.config.settings import AiGatewaySettings
from nerdit.core.ai_gateway.keys import service_for_key
from nerdit.core.bindings.secretref import BindingNotReady, walk_secret_ref
from nerdit.core.models.binding import _resolve_ollama
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError, SecretManager
from nerdit.db.queries import Queries
from nerdit.db.queries.ai_gateway import SQLITE_MAX_INT
from nerdit.db.rows import AiGatewayRoute

logger = logging.getLogger(__name__)

# One SSE line longer than this is dropped from usage scanning (still forwarded).
_MAX_SSE_LINE = 1024 * 1024


@dataclass
class Gateway:
    """Everything a gateway request needs; `client` is the one upstream pool."""

    queries: Queries
    secrets: SecretManager | None
    settings: AiGatewaySettings
    client: httpx.AsyncClient

    def __post_init__(self) -> None:
        # One pool serves every app: a provider's Set-Cookie must never ride
        # along on another app's request, so the jar accepts nothing.
        self.client.cookies.jar.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))


def error(status: int, code: str, message: str) -> JSONResponse:
    """An OpenAI-shaped error envelope."""
    kind = (
        "authentication_error"
        if status == 401
        else "invalid_request_error"
        if status < 500
        else "api_error"
    )
    return JSONResponse(
        {"error": {"message": message, "type": kind, "code": code}}, status_code=status
    )


async def authenticate(gw: Gateway, request: Request) -> str | None:
    """The calling service for a valid `Authorization: Bearer nk_…`, else `None`."""
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return await service_for_key(gw.queries, key.strip())


async def _read_body(request: Request, limit: int) -> bytes | None:
    """The body, or `None` once it exceeds `limit` bytes (never buffered past it)."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


async def _upstream(gw: Gateway, route: AiGatewayRoute) -> tuple[str, str] | JSONResponse:
    """`(base_url, provider_key)` for the route, or the 503 explaining why not."""
    if route.provider == "ollama":
        try:
            # The daemon reaches served models on loopback, not on the bridge.
            resolved = await _resolve_ollama(
                route.alias,
                AiBindingConfig(provider="ollama", model=route.model),
                gw.queries,
                "127.0.0.1",
            )
        except BindingNotReady as exc:
            return error(503, "model_not_ready", str(exc))
        return resolved.base_url, resolved.api_key
    ref = route.api_key_ref or ""
    secrets = gw.secrets
    try:
        # Machine-level alias: only the shared scope is consulted (no
        # per-service override; an unscoped ref resolves nothing).
        res = walk_secret_ref(
            ref,
            service_env=None,
            shared_env=(lambda: secrets.load(SHARED_SCOPE)) if secrets is not None else None,
        )
    except SecretDecryptError:
        return error(503, "secret_missing", f"the provider key {ref} cannot be decrypted")
    if res.value is None:
        return error(
            503,
            "secret_missing",
            f"the provider key {ref} is not set; ask your agent to set it with "
            f"nerdit secrets set --shared",
        )
    key = res.value.strip()
    if not key or not key.isascii() or not key.isprintable():
        # Never echo the value: only the ref's name reaches the app.
        return error(
            503,
            "secret_invalid",
            f"the provider key {ref} is empty or holds non-ASCII or control characters",
        )
    return (route.base_url or "").rstrip("/"), key


def _tokens(usage: object) -> tuple[int, int]:
    """`(prompt_tokens, completion_tokens)` from an OpenAI `usage` object."""
    if not isinstance(usage, dict):
        return 0, 0

    def count(field: str) -> int:
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return 0
        return min(value, SQLITE_MAX_INT)  # a wild provider counter must not 500 the answer

    return count("prompt_tokens"), count("completion_tokens")


async def _count(gw: Gateway, service: str, alias: str, usage: object, failed: bool) -> None:
    """Record one request; shielded so a client hanging up mid-answer still counts it."""
    prompt, completion = _tokens(usage)
    record = gw.queries.record_ai_usage(
        service, alias, prompt_tokens=prompt, completion_tokens=completion, upstream_error=failed
    )
    await asyncio.shield(record)


class _SseUsage:
    """Keeps the last `usage` object seen in an SSE stream's `data:` lines."""

    def __init__(self) -> None:
        self._buf = b""
        self.usage: object = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        *lines, self._buf = self._buf.split(b"\n")
        if len(self._buf) > _MAX_SSE_LINE:
            self._buf = b""
        for line in lines:
            self._line(line)

    def finish(self) -> None:
        self._line(self._buf)
        self._buf = b""

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        try:
            obj = json.loads(data)
        except (ValueError, RecursionError):
            return
        if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]


@dataclass
class _Call:
    """One authenticated, rewritten request ready to go upstream."""

    service: str
    alias: str
    request: httpx.Request


async def _prepare(gw: Gateway, request: Request, path: str) -> _Call | Response:
    """Authenticate, bound and parse the body, map the alias; or the error to answer."""
    service = await authenticate(gw, request)
    if service is None:
        return error(401, "invalid_api_key", "missing or invalid AI gateway key")
    body = await _read_body(request, gw.settings.max_body_bytes)
    if body is None:
        return error(
            413, "body_too_large", f"request body exceeds {gw.settings.max_body_bytes} bytes"
        )
    try:
        payload: Any = json.loads(body)
    except (ValueError, RecursionError):
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        return error(400, "invalid_request", "body must be a JSON object with a string 'model'")
    alias = payload["model"]
    route = await gw.queries.get_ai_route(alias)
    if route is None:
        return error(
            404,
            "alias_not_found",
            f"model alias '{alias}' is not defined on this machine; "
            f"ask your agent to add it with nerdit ai routes set",
        )
    target = await _upstream(gw, route)
    if isinstance(target, Response):
        return target
    base_url, provider_key = target
    payload["model"] = route.model
    stream = payload.get("stream") is True
    if stream and path in ("chat/completions", "completions"):
        options = payload.get("stream_options")
        payload["stream_options"] = {
            **(options if isinstance(options, dict) else {}),
            "include_usage": True,
        }
    headers = {"content-type": "application/json", "authorization": f"Bearer {provider_key}"}
    if "accept" in request.headers:
        headers["accept"] = request.headers["accept"]
    timeout = gw.settings.stream_timeout_s if stream else gw.settings.upstream_timeout_s
    upstream_req = gw.client.build_request(
        "POST",
        f"{base_url}/{path}",
        content=json.dumps(payload).encode(),
        headers=headers,
        timeout=httpx.Timeout(timeout),
    )
    return _Call(service, alias, upstream_req)


async def forward(gw: Gateway, request: Request, path: str) -> Response:
    """Forward `POST /v1/<path>` upstream and relay the answer, counting usage."""
    call = await _prepare(gw, request, path)
    if isinstance(call, Response):
        return call
    service, alias = call.service, call.alias
    try:
        upstream = await gw.client.send(call.request, stream=True)
    except httpx.HTTPError as exc:
        logger.warning(
            "AI gateway: upstream unreachable for alias '%s' (service %s): %s",
            alias,
            service,
            type(exc).__name__,
        )
        await gw.queries.record_ai_usage(service, alias, upstream_error=True)
        return error(
            502, "upstream_unreachable", f"the provider for alias '{alias}' is unreachable"
        )

    if not 200 <= upstream.status_code < 300:
        await upstream.aclose()  # the upstream body is never read, logged nor forwarded
        logger.warning(
            "AI gateway: upstream answered %d for alias '%s' (service %s)",
            upstream.status_code,
            alias,
            service,
        )
        await gw.queries.record_ai_usage(service, alias, upstream_error=True)
        status = upstream.status_code if upstream.status_code >= 400 else 502
        answer = error(
            status,
            "upstream_error",
            f"the provider for alias '{alias}' answered HTTP {upstream.status_code}",
        )
        # Keep the provider's backoff hint for the app's SDK (numeric only).
        for name in ("retry-after", "retry-after-ms"):
            value = upstream.headers.get(name, "")
            if value.isascii() and value.isdigit():
                answer.headers[name] = value
        return answer

    content_type = upstream.headers.get("content-type", "application/json")
    if _is_sse(content_type):
        return StreamingResponse(
            _relay(gw, upstream, service, alias),
            status_code=upstream.status_code,
            media_type=content_type,
        )
    return await _buffered(gw, upstream, service, alias, content_type)


async def _buffered(
    gw: Gateway, upstream: httpx.Response, service: str, alias: str, content_type: str
) -> Response:
    """Relay a non-SSE answer, bounded by `max_body_bytes` like the request body.

    The answer is buffered in the daemon's memory: a provider must not be able
    to grow the control plane's heap (declared length first, then while reading).
    """
    limit = gw.settings.max_body_bytes
    declared = upstream.headers.get("content-length", "")
    content = bytearray()
    failed = False
    too_large = declared.isdigit() and int(declared) > limit
    try:
        if not too_large:
            async for chunk in upstream.aiter_bytes():
                content += chunk
                if len(content) > limit:
                    too_large = True
                    break
    except httpx.HTTPError as exc:
        failed = True
        logger.warning(
            "AI gateway: upstream read failed for alias '%s' (service %s): %s",
            alias,
            service,
            type(exc).__name__,
        )
        return error(502, "upstream_unreachable", f"the provider for alias '{alias}' dropped")
    finally:
        await upstream.aclose()
        usage = None
        if not too_large and not failed:
            try:
                usage = json.loads(bytes(content)).get("usage")
            except (ValueError, AttributeError, RecursionError):
                usage = None
        await _count(gw, service, alias, usage, failed or too_large)
    if too_large:
        logger.warning(
            "AI gateway: upstream answer for alias '%s' (service %s) exceeds %d bytes",
            alias,
            service,
            limit,
        )
        return error(
            502,
            "upstream_too_large",
            f"the provider for alias '{alias}' answered more than {limit} bytes",
        )
    return Response(bytes(content), status_code=upstream.status_code, media_type=content_type)


def _is_sse(content_type: str) -> bool:
    """Media type tokens are case-insensitive: `Text/Event-Stream; charset=utf-8` streams."""
    return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"


async def _relay(
    gw: Gateway, upstream: httpx.Response, service: str, alias: str
) -> AsyncIterator[bytes]:
    """Pass SSE bytes through untouched, then count the usage of the last chunk carrying it.

    A client hanging up mid-stream still counts the request (tokens seen so far).
    """
    scan = _SseUsage()
    failed = False
    try:
        async for chunk in upstream.aiter_bytes():
            scan.feed(chunk)
            yield chunk
    except httpx.HTTPError as exc:
        failed = True
        logger.warning(
            "AI gateway: stream broke for alias '%s' (service %s): %s",
            alias,
            service,
            type(exc).__name__,
        )
    finally:
        await upstream.aclose()
        scan.finish()
        await _count(gw, service, alias, scan.usage, failed)
