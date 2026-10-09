"""The machine AI gateway: settings, virtual keys, alias proxying and usage.

Upstream is an `httpx.MockTransport`; the gateway app is driven through
`httpx.ASGITransport` (pure async, no TestClient).
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket

import httpx
import pytest
from pydantic import ValidationError

from nerdit.config.settings import AiGatewaySettings, NerditSettings, load_settings
from nerdit.config.store import ConfigStore
from nerdit.core.ai_gateway.app import GatewayServer, build_gateway_app
from nerdit.core.ai_gateway.keys import mint_key, service_for_key
from nerdit.core.ai_gateway.proxy import Gateway, _relay
from nerdit.core.launch import scrub_secret_values
from nerdit.core.secrets import SHARED_SCOPE, SecretManager

PROVIDER_KEY = "sk-SENTINEL-provider-value-0042"
UPSTREAM = "https://api.example.com/v1"


# --- settings -------------------------------------------------------------------


def test_settings_default_on_since_0_8_3_on_9330():
    gw = NerditSettings().ai_gateway
    assert (gw.enabled, gw.port, gw.upstream_timeout_s, gw.stream_timeout_s) == (
        True,
        9330,
        120,
        600,
    )
    assert gw.max_body_bytes == 10 * 1024 * 1024


def test_settings_section_loads_from_toml(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[ai_gateway]\nenabled = true\nport = 9444\n")
    gw = load_settings(cfg).ai_gateway
    assert (gw.enabled, gw.port) == (True, 9444)


@pytest.mark.parametrize("port", [0, 80, 1023, 65536])
def test_settings_bad_port_refused(port):
    with pytest.raises(ValidationError):
        AiGatewaySettings(port=port)


def test_settings_section_known_and_restart_required(tmp_path):
    store = ConfigStore(tmp_path / "config.toml")
    assert "ai_gateway" in store.known_sections()
    assert store.stage("ai_gateway", {"enabled": False}).requires_restart is True


# --- fixtures ---------------------------------------------------------------------


class _Upstream:
    """Captures what reaches the provider and answers with a canned response."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.respond = lambda request: httpx.Response(
            200,
            json={"id": "x", "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
            headers={"set-cookie": "s=1", "x-provider-trace": "t"},
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)


@pytest.fixture
async def setup(queries, tmp_path):
    secrets = SecretManager(tmp_path / "secrets")
    secrets.set(SHARED_SCOPE, {"OPENAI_KEY": PROVIDER_KEY})
    await queries.upsert_ai_route(
        "fast",
        provider="api",
        model="gpt-4o-mini",
        base_url=UPSTREAM,
        api_key_ref="${secrets.shared.OPENAI_KEY}",
    )
    upstream = _Upstream()
    gw = Gateway(
        queries,
        secrets,
        AiGatewaySettings(enabled=True, max_body_bytes=4096),
        httpx.AsyncClient(transport=httpx.MockTransport(upstream)),
    )
    key = await mint_key(queries, "app", revoke_others=True)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_gateway_app(gw)), base_url="http://gw"
    )
    yield gw, upstream, key, client
    await client.aclose()
    await gw.client.aclose()


def _auth(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


async def _usage(queries) -> dict:
    rows = await queries.list_ai_usage(days=1)
    return {(r.service_name, r.alias): r for r in rows}


# --- auth -------------------------------------------------------------------------


async def test_missing_or_bad_key_is_401_and_never_forwarded(setup):
    gw, upstream, key, client = setup
    body = {"model": "fast", "messages": []}
    for headers in ({}, _auth("nk_" + "0" * 32), {"authorization": f"Basic {key}"}, _auth("x")):
        resp = await client.post("/v1/chat/completions", json=body, headers=headers)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "invalid_api_key"
    assert upstream.requests == []


async def test_revoked_key_is_401(setup):
    gw, upstream, key, client = setup
    assert await gw.queries.revoke_ai_gateway_keys("app") == 1
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 401
    assert upstream.requests == []


# --- proxying ---------------------------------------------------------------------


async def test_alias_rewrite_strips_headers_and_counts_usage(setup):
    gw, upstream, key, client = setup
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
        headers={**_auth(key), "x-custom": "leak", "cookie": "c=1", "accept": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == "x"
    # Provider response headers are not relayed.
    assert "set-cookie" not in resp.headers and "x-provider-trace" not in resp.headers

    (sent,) = upstream.requests
    assert str(sent.url) == f"{UPSTREAM}/chat/completions"
    # The app's virtual key never reaches the provider; only the provider key does.
    assert sent.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
    assert key not in str(sent.headers) and key not in sent.content.decode()
    assert "x-custom" not in sent.headers and "cookie" not in sent.headers
    assert sent.headers["accept"] == "application/json"
    assert json.loads(sent.content)["model"] == "gpt-4o-mini"

    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.requests, row.prompt_tokens, row.completion_tokens, row.upstream_errors) == (
        1,
        3,
        5,
        0,
    )

    # The provider's Set-Cookie is not kept by the shared pool for the next request.
    await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert "cookie" not in upstream.requests[1].headers


async def test_embeddings_and_completions_paths(setup):
    gw, upstream, key, client = setup
    for path in ("embeddings", "completions"):
        resp = await client.post(f"/v1/{path}", json={"model": "fast"}, headers=_auth(key))
        assert resp.status_code == 200
    assert [str(r.url) for r in upstream.requests] == [
        f"{UPSTREAM}/embeddings",
        f"{UPSTREAM}/completions",
    ]


async def test_streaming_passthrough_counts_usage_from_last_chunk(setup):
    gw, upstream, key, client = setup
    events = (
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"llo"}}],"usage":null}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def chunks():
        for i in range(0, len(events), 13):  # split mid-line on purpose
            yield events[i : i + 13]

    upstream.respond = lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=chunks()
    )
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "stream": True, "stream_options": {"x": 1}},
        headers=_auth(key),
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.content == events
    sent = json.loads(upstream.requests[0].content)
    assert sent["stream_options"] == {"x": 1, "include_usage": True}
    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.prompt_tokens, row.completion_tokens) == (7, 2)


async def test_streamed_completions_also_request_usage(setup):
    gw, upstream, key, client = setup
    resp = await client.post(
        "/v1/completions", json={"model": "fast", "stream": True}, headers=_auth(key)
    )
    assert resp.status_code == 200
    assert json.loads(upstream.requests[0].content)["stream_options"] == {"include_usage": True}


async def test_sse_media_type_is_matched_case_insensitively(setup):
    gw, upstream, key, client = setup
    events = b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n'

    async def chunks():
        yield events

    upstream.respond = lambda request: httpx.Response(
        200, headers={"content-type": "Text/Event-Stream; charset=utf-8"}, content=chunks()
    )
    resp = await client.post(
        "/v1/chat/completions", json={"model": "fast", "stream": True}, headers=_auth(key)
    )
    assert resp.status_code == 200 and resp.content == events
    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.prompt_tokens, row.completion_tokens) == (1, 1)


@pytest.mark.parametrize("declared", [True, False])
async def test_oversized_non_sse_answer_is_502_not_buffered(setup, declared):
    gw, upstream, key, client = setup
    big = b'{"usage":{"prompt_tokens":1},"pad":"' + b"x" * 5000 + b'"}'

    async def chunks():
        for i in range(0, len(big), 1000):
            yield big[i : i + 1000]

    upstream.respond = lambda request: (
        httpx.Response(200, content=big)  # content-length declared
        if declared
        else httpx.Response(200, headers={"content-type": "application/json"}, content=chunks())
    )
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_too_large"
    assert b"xxxx" not in resp.content
    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.requests, row.upstream_errors, row.prompt_tokens) == (1, 1, 0)


async def test_body_over_limit_is_413(setup):
    gw, upstream, key, client = setup
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "fast", "pad": "x" * 5000},
        headers=_auth(key),
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "body_too_large"
    assert upstream.requests == []


async def test_unknown_alias_is_404_with_hint(setup):
    gw, upstream, key, client = setup
    resp = await client.post("/v1/chat/completions", json={"model": "slow"}, headers=_auth(key))
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "alias_not_found"
    assert "'slow'" in err["message"] and "nerdit ai routes set" in err["message"]


async def test_non_json_body_is_400(setup):
    gw, upstream, key, client = setup
    resp = await client.post("/v1/chat/completions", content=b"nope", headers=_auth(key))
    assert resp.status_code == 400


async def test_deeply_nested_body_is_400_not_500(setup):
    gw, upstream, key, client = setup
    gw.settings.max_body_bytes = 200_000
    resp = await client.post("/v1/chat/completions", content=b"[" * 100_000, headers=_auth(key))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"
    assert upstream.requests == []


async def test_upstream_error_status_passes_through_without_body(setup):
    gw, upstream, key, client = setup
    upstream.respond = lambda request: httpx.Response(
        500, json={"error": {"message": f"bad key {PROVIDER_KEY}"}}
    )
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "upstream_error"
    assert PROVIDER_KEY not in resp.text
    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.requests, row.upstream_errors) == (1, 1)


async def test_upstream_429_keeps_retry_after(setup):
    gw, upstream, key, client = setup
    upstream.respond = lambda request: httpx.Response(
        429, json={}, headers={"retry-after": "7", "retry-after-ms": "bad"}
    )
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7" and "retry-after-ms" not in resp.headers


async def test_stream_cut_by_the_client_still_counts(setup):
    gw, upstream, key, client = setup

    async def chunks():
        yield b'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n'
        yield b"data: [DONE]\n\n"

    upstream.respond = lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=chunks()
    )
    up = await gw.client.send(gw.client.build_request("POST", f"{UPSTREAM}/x"), stream=True)
    gen = _relay(gw, up, "app", "fast")
    assert await gen.__anext__()
    await gen.aclose()  # the app hung up after one chunk
    row = (await _usage(gw.queries))[("app", "fast")]
    assert (row.requests, row.prompt_tokens, row.upstream_errors) == (1, 7, 0)


async def test_non_ascii_provider_secret_is_503_without_the_value(setup):
    gw, upstream, key, client = setup
    gw.secrets.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-SENTINEL\u00e9-1234"})
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "secret_invalid"
    assert "SENTINEL" not in resp.text and upstream.requests == []
    # A trailing newline (pasted value) is trimmed rather than forwarded.
    gw.secrets.set(SHARED_SCOPE, {"OPENAI_KEY": PROVIDER_KEY + "\n"})
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 200
    assert upstream.requests[0].headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_virtual_keys_are_masked_in_captured_output():
    key = "nk_" + "ab12" * 8
    assert scrub_secret_values([f"crash OPENAI_API_KEY={key}"], []) == ["crash OPENAI_API_KEY=***"]


async def test_deleting_a_service_revokes_its_keys(queries):
    from tests.test_ai_binding_resolve import _svc_job

    job = _svc_job("app")
    await queries.create_job(job)
    key = await mint_key(queries, "app", revoke_others=True)
    assert await service_for_key(queries, key) == "app"
    assert await queries.delete_service_checked(job.id) == []
    assert await service_for_key(queries, key) is None


async def test_upstream_unreachable_is_502(setup):
    gw, upstream, key, client = setup

    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    upstream.respond = boom
    resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_unreachable"
    assert (await _usage(gw.queries))[("app", "fast")].upstream_errors == 1


async def test_missing_provider_secret_is_503_naming_the_ref(setup):
    gw, upstream, key, client = setup
    await gw.queries.upsert_ai_route(
        "other",
        provider="api",
        model="m",
        base_url=UPSTREAM,
        api_key_ref="${secrets.shared.NOT_SET}",
    )
    resp = await client.post("/v1/chat/completions", json={"model": "other"}, headers=_auth(key))
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["code"] == "secret_missing"
    assert "${secrets.shared.NOT_SET}" in err["message"]
    assert upstream.requests == []


async def test_ollama_route_not_served_is_503(setup):
    gw, upstream, key, client = setup
    await gw.queries.upsert_ai_route("local", provider="ollama", model="llama3.1:8b")
    resp = await client.post("/v1/chat/completions", json={"model": "local"}, headers=_auth(key))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "model_not_ready"


async def test_models_lists_aliases_and_unknown_paths_404(setup):
    gw, upstream, key, client = setup
    assert (await client.get("/v1/models")).status_code == 401
    resp = await client.get("/v1/models", headers=_auth(key))
    assert resp.json() == {
        "object": "list",
        "data": [{"id": "fast", "object": "model", "owned_by": "nerdit"}],
    }
    for method, path, status in (
        ("GET", "/v1/files", 404),
        ("POST", "/admin", 404),
        ("GET", "/v1/chat/completions", 405),
    ):
        resp = await client.request(method, path, headers=_auth(key))
        assert resp.status_code == status
        assert set(resp.json()["error"]) == {"message", "type", "code"}


async def test_no_secret_or_virtual_key_in_logs_or_responses(setup, caplog):
    gw, upstream, key, client = setup
    caplog.set_level(logging.DEBUG)
    texts: list[str] = []

    def boom(request):
        raise httpx.ConnectError(f"refused {request.headers['authorization']}", request=request)

    answers = [
        lambda r: httpx.Response(200, json={"usage": {}}),
        lambda r: httpx.Response(401, text=f"Incorrect API key provided: {PROVIDER_KEY}"),
        boom,
    ]
    for respond in answers:
        upstream.respond = respond
        resp = await client.post("/v1/chat/completions", json={"model": "fast"}, headers=_auth(key))
        texts.append(resp.text + str(resp.headers))
    for model in ("nope", "fast"):
        resp = await client.post("/v1/embeddings", json={"model": model}, headers=_auth(key))
        texts.append(resp.text)
    texts.append(caplog.text)
    for text in texts:
        assert PROVIDER_KEY not in text
        assert key not in text


# --- the listener -----------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_listener_serves_and_stops(queries, tmp_path):
    port = _free_port()
    gw = Gateway(queries, None, AiGatewaySettings(enabled=True, port=port), httpx.AsyncClient())
    server = GatewayServer(gw, "127.0.0.1", port)
    assert await server.start() is True
    assert server.listening
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/v1/models")
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.02)
        assert resp.status_code == 401
    finally:
        await server.stop()


async def test_listener_bind_failure_is_not_fatal(queries, caplog):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        gw = Gateway(queries, None, AiGatewaySettings(port=port), httpx.AsyncClient())
        server = GatewayServer(gw, "127.0.0.1", port)
        assert await server.start() is False  # the SystemExit never escapes
        assert not server.listening
        await server.stop()
    assert "AI gateway could not listen" in caplog.text


async def test_listener_caps_connections_and_drops_slow_headers(queries, monkeypatch):
    from nerdit.core.ai_gateway import app as gateway_app

    monkeypatch.setattr(gateway_app, "_MAX_CONNECTIONS", 2)
    monkeypatch.setattr(gateway_app, "_HEADER_DEADLINE_S", 0.2)
    port = _free_port()
    gw = Gateway(queries, None, AiGatewaySettings(enabled=True, port=port), httpx.AsyncClient())
    server = GatewayServer(gw, "127.0.0.1", port)
    assert await server.start()
    writers = []
    try:
        for _ in range(100):
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                break
            except OSError:
                await asyncio.sleep(0.02)
        writer.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n")  # never finished
        conns = [(reader, writer)]
        for _ in range(2):
            conns.append(await asyncio.open_connection("127.0.0.1", port))
        writers = [w for _, w in conns]
        # Over the cap: closed at once, before the header deadline.
        assert await asyncio.wait_for(conns[2][0].read(), 0.15) == b""
        # The partial header and the idle socket are dropped after the deadline.
        assert await asyncio.wait_for(conns[0][0].read(), 2) == b""
        assert await asyncio.wait_for(conns[1][0].read(), 2) == b""
    finally:
        for w in writers:
            w.close()
        await server.stop()


def test_usage_counters_are_bounded_to_sqlite_integers():
    from nerdit.core.ai_gateway.proxy import SQLITE_MAX_INT, _tokens

    assert _tokens({"prompt_tokens": 2**70, "completion_tokens": -4}) == (SQLITE_MAX_INT, 0)
    assert _tokens({"prompt_tokens": True, "completion_tokens": 3}) == (0, 3)
