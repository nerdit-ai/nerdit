"""Route tests for `/ai-gateway/*`: roles, idempotency, the base_url / secret-ref
gates, the alias-in-use 409, key metadata and usage, and that no secret-shaped
input reaches a response or an audit row.

The `test_variables_route` harness: a real in-memory database under the real
auth, audit and idempotency middleware. DNS is a seam (`_resolve`), so no test
leaves the machine.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from nerdit.config.settings import (
    AiGatewaySettings,
    DaemonSettings,
    ModelsSettings,
    ProxySettings,
)
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes import ai_gateway as route_module
from nerdit.daemon.routes.ai_gateway import router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries
from tests.test_projects_route import A_RAW, LEGACY, RO_RAW, _auth, _Env

SENTINEL = "sk-live-SENTINEL-4b1e"
_DNS = {
    "api.example.com": ["93.184.216.34"],
    "lan.example.com": ["192.168.1.20"],
    "mixed.example.com": ["93.184.216.34", "10.0.0.5"],
    "localhost": ["127.0.0.1", "::1"],
}


@pytest.fixture
async def env(monkeypatch):
    def fake_resolve(host: str, port: int) -> list[str]:
        if host in _DNS:
            return _DNS[host]
        try:
            import ipaddress

            return [str(ipaddress.ip_address(host))]
        except ValueError:
            raise OSError("no such host") from None

    monkeypatch.setattr(route_module, "_resolve", fake_resolve)
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for raw, tid, role in (
        (A_RAW, "tok-a", TokenRole.submitter),
        (RO_RAW, "tok-ro", TokenRole.readonly),
    ):
        await queries.create_api_token(
            ApiToken(id=tid, name=tid, role=role, token_hash=hash_token(raw))
        )
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router)
    app.state.queries = queries
    app.state.settings = SimpleNamespace(
        models=ModelsSettings(bridge_host="172.17.0.1"),
        ai_gateway=AiGatewaySettings(enabled=False),  # the "stored while off" shape
        daemon=DaemonSettings(),
        proxy=ProxySettings(),
    )
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(
        AuditMiddleware,
        get_queries=lambda: queries,
        get_event_bus=lambda: SimpleNamespace(publish=lambda e: None),
    )
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    e = _Env(app, queries, db)
    async with e.client:
        yield e
    await db.close()


_API = {
    "provider": "api",
    "model": "openai/gpt-4o-mini",
    "base_url": "https://api.example.com/v1",
    "api_key_ref": "${secrets.shared.OPENROUTER_API_KEY}",
}


async def _put(env, alias, body, *, raw=LEGACY, idem="k1"):
    headers = {**_auth(raw), **({"Idempotency-Key": idem} if idem else {})}
    return await env.client.put(f"/ai-gateway/routes/{alias}", json=body, headers=headers)


async def test_admin_creates_then_replaces_an_alias_and_reads_see_it(env):
    resp = await _put(env, "fast", _API)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["created"] is True and body["enabled"] is False  # stored while off
    assert body["route"]["api_key_ref"] == "${secrets.shared.OPENROUTER_API_KEY}"
    again = await _put(env, "fast", {**_API, "model": "openai/gpt-4o"}, idem="k2")
    assert again.json()["created"] is False and again.json()["route"]["model"] == "openai/gpt-4o"

    listed = await env.client.get("/ai-gateway/routes", headers=_auth(RO_RAW))
    assert [r["alias"] for r in listed.json()["routes"]] == ["fast"]
    state = (await env.client.get("/ai-gateway", headers=_auth(RO_RAW))).json()
    assert state == {
        "enabled": False,
        "listening": False,
        "host": "172.17.0.1",
        "port": 9330,
        "base_url": "http://172.17.0.1:9330/v1",
        "routes": 1,
    }
    audits = await env.audit("ai_gateway.route_set")
    assert audits[0] == (
        "fast",
        {
            "alias": "fast",
            "provider": "api",
            "model": "openai/gpt-4o-mini",
            "base_url_host": "api.example.com",
            "api_key_ref_name": "OPENROUTER_API_KEY",
        },
    )


async def test_writes_take_submitter_refuse_readonly_and_put_needs_an_idempotency_key(env):
    assert (await _put(env, "fast", _API, raw=RO_RAW)).status_code == 403
    resp = await env.client.delete("/ai-gateway/keys/web", headers=_auth(RO_RAW))
    assert resp.status_code == 403
    resp = await _put(env, "fast", _API, idem=None)
    assert resp.status_code == 400 and resp.json()["code"] == "idempotency_key_required"
    assert await env.queries.get_ai_route("fast") is None
    # 0.8.3 (D-P31-5): a submitter -- a hosted box's tunnel principal -- writes aliases.
    assert (await _put(env, "fast", _API, raw=A_RAW)).status_code == 200
    resp = await env.client.delete("/ai-gateway/routes/fast", headers=_auth(A_RAW))
    assert resp.status_code == 200, resp.text
    assert await env.queries.get_ai_route("fast") is None


@pytest.mark.parametrize(
    ("alias", "body", "code"),
    [
        ("Fast", _API, "ai_gateway.alias_invalid"),
        ("a" * 33, _API, "ai_gateway.alias_invalid"),
        ("fast", {**_API, "base_url": "http://api.example.com/v1"}, "ai_gateway.base_url_invalid"),
        ("fast", {**_API, "base_url": "https://lan.example.com/v1"}, "ai_gateway.base_url_invalid"),
        (
            "fast",
            {**_API, "base_url": "https://mixed.example.com/v1"},
            "ai_gateway.base_url_invalid",
        ),
        ("fast", {**_API, "base_url": "https://169.254.169.254/v1"}, "ai_gateway.base_url_invalid"),
        (
            "fast",
            {**_API, "base_url": "https://[::ffff:10.0.0.1]/v1"},
            "ai_gateway.base_url_invalid",
        ),
        ("fast", {**_API, "base_url": "https://nowhere.invalid/v1"}, "ai_gateway.base_url_invalid"),
        # The daemon's own listeners: its API, Caddy's admin, the gateway itself.
        ("fast", {**_API, "base_url": "http://127.0.0.1:9321/api"}, "ai_gateway.base_url_invalid"),
        (
            "fast",
            {**_API, "base_url": "http://localhost:2019/config"},
            "ai_gateway.base_url_invalid",
        ),
        ("fast", {**_API, "base_url": "http://172.17.0.1:9330/v1"}, "ai_gateway.base_url_invalid"),
        ("fast", {**_API, "base_url": "ftp://api.example.com/v1"}, "ai_gateway.base_url_invalid"),
        (
            "fast",
            {**_API, "base_url": "https://api.example.com/v1\x00x"},
            "ai_gateway.base_url_invalid",
        ),
        (
            "fast",
            {**_API, "base_url": "https://api.example.com/v1?k=1"},
            "ai_gateway.base_url_invalid",
        ),
        ("fast", {k: v for k, v in _API.items() if k != "base_url"}, "ai_gateway.base_url_invalid"),
        (
            "fast",
            {**_API, "api_key_ref": "${secrets.OPENROUTER_API_KEY}"},
            "ai_gateway.secret_ref_invalid",
        ),
        (
            "fast",
            {k: v for k, v in _API.items() if k != "api_key_ref"},
            "ai_gateway.secret_ref_invalid",
        ),
        (
            "local",
            {"provider": "ollama", "model": "llama3", "base_url": "https://api.example.com/v1"},
            "ai_gateway.base_url_invalid",
        ),
        (
            "local",
            {"provider": "ollama", "model": "llama3", "api_key_ref": _API["api_key_ref"]},
            "ai_gateway.secret_ref_invalid",
        ),
    ],
)
async def test_refusals_are_structured_and_store_nothing(env, alias, body, code):
    resp = await _put(env, alias, body)
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == code
    assert await env.queries.list_ai_routes() == []


async def test_local_http_and_ollama_routes_are_accepted(env):
    for alias, url in (
        ("loop", "http://127.0.0.1:8080/v1"),
        ("bridge", "http://172.17.0.1:8080/v1"),
        ("lh", "http://localhost:11434/v1"),
    ):
        resp = await _put(env, alias, {**_API, "base_url": url}, idem=alias)
        assert resp.status_code == 200, resp.text
    resp = await _put(env, "local", {"provider": "ollama", "model": "llama3:8b"}, idem="o")
    assert resp.status_code == 200 and resp.json()["route"]["base_url"] is None


async def test_no_secret_shaped_input_reaches_a_response_or_the_audit(env):
    attempts = [
        {**_API, "api_key_ref": SENTINEL},  # a literal key where its reference belongs
        {**_API, "base_url": f"https://user:{SENTINEL}@api.example.com/v1"},
        {"provider": "api", "api_key_ref": SENTINEL},  # pydantic 422 (missing model)
        {**_API, "api_key_ref": SENTINEL, "extra": SENTINEL},
    ]
    for i, body in enumerate(attempts):
        resp = await _put(env, "fast", body, idem=f"s{i}")
        assert resp.status_code == 422, resp.text
        assert SENTINEL not in resp.text
    cursor = await env.db.conn.execute("SELECT params_redacted FROM audit_log")
    assert all(SENTINEL not in (row[0] or "") for row in await cursor.fetchall())


async def _gateway_app(
    env, name: str, alias: str, desired: str = "running", owner: str | None = None
) -> None:
    job = Job(
        id=f"job-{name}"[:12],
        name=name,
        service_name=name,
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state=desired,
        restart_policy="always",
        submitted_by_token=owner,
        config=json.dumps(
            {"image": "demo:1", "ai": {"default": {"provider": "gateway", "model": alias}}}
        ),
    )
    await env.queries.reserve_service_for_token(job, admin=True)


async def test_delete_refuses_an_alias_a_running_app_uses_unless_forced(env):
    assert (await _put(env, "fast", _API)).status_code == 200
    await _gateway_app(env, "web", "fast")
    await _gateway_app(env, "old", "fast", desired="stopped")
    resp = await env.client.delete("/ai-gateway/routes/fast", headers=_auth(LEGACY))
    assert resp.status_code == 409 and resp.json()["code"] == "ai_gateway.alias_in_use"
    assert "web" in resp.json()["message"] and "old" not in resp.json()["message"]
    resp = await env.client.delete("/ai-gateway/routes/fast?force=true", headers=_auth(LEGACY))
    assert resp.status_code == 200 and resp.json()["used_by"] == ["web"]
    assert await env.queries.get_ai_route("fast") is None
    resp = await env.client.delete("/ai-gateway/routes/fast", headers=_auth(LEGACY))
    assert resp.status_code == 404
    assert (await env.audit("ai_gateway.route_removed"))[0][0] == "fast"


async def test_a_submitter_revokes_only_its_own_apps_keys(env):
    await _gateway_app(env, "mine", "fast", owner="tok-a")
    await _gateway_app(env, "theirs", "fast")  # NULL owner: admin-only, like every service verb
    for name in ("mine", "theirs"):
        await env.queries.insert_ai_gateway_key(name, name[0] * 64, revoke_others=True)
    resp = await env.client.delete("/ai-gateway/keys/theirs", headers=_auth(A_RAW))
    assert resp.status_code == 403, resp.text
    keys = {k.service_name: k for k in await env.queries.list_ai_gateway_keys()}
    assert keys["theirs"].revoked_at is None
    resp = await env.client.delete("/ai-gateway/keys/mine", headers=_auth(A_RAW))
    assert resp.status_code == 200 and resp.json()["revoked"] == 1


async def test_keys_show_metadata_only_and_revoke(env):
    await _gateway_app(env, "web", "fast")
    await env.queries.insert_ai_gateway_key("web", "a" * 64, revoke_others=True)
    listed = await env.client.get("/ai-gateway/keys", headers=_auth(RO_RAW))
    assert listed.status_code == 200
    assert [set(k) for k in listed.json()["keys"]] == [{"service_name", "created_at", "revoked_at"}]
    assert "a" * 64 not in listed.text
    resp = await env.client.delete("/ai-gateway/keys/web", headers=_auth(LEGACY))
    assert resp.json() == {"service": "web", "revoked": 1}
    assert (await env.client.get("/ai-gateway/keys", headers=_auth(RO_RAW))).json()["keys"] == []
    revoked = await env.client.get("/ai-gateway/keys?include_revoked=true", headers=_auth(RO_RAW))
    assert revoked.json()["keys"][0]["revoked_at"] is not None
    assert (await env.audit("ai_gateway.key_revoked"))[0] == ("web", {"service": "web"})


async def test_revoking_an_unknown_service_is_404_and_leaves_no_trace(env):
    pasted = "sk-live-SENTINEL-4242"
    resp = await env.client.delete(f"/ai-gateway/keys/{pasted}", headers=_auth(LEGACY))
    assert resp.status_code == 404
    assert pasted not in resp.text
    # The refused call is audited as an error with NO params: the only place the
    # path segment lands is the middleware's generic target_id stamp, which every
    # `/{service}` route shares (the same as `DELETE /secrets/<typo>`).
    cursor = await env.db.conn.execute("SELECT result, params_redacted FROM audit_log")
    rows = await cursor.fetchall()
    assert rows and all(row[0] == "error" and row[1] is None for row in rows)


async def test_usage_aggregates_saturate_instead_of_overflowing(env):
    from nerdit.db.queries.ai_gateway import SQLITE_MAX_INT

    for _ in range(2):
        await env.queries.record_ai_usage("web", "fast", prompt_tokens=SQLITE_MAX_INT)
    (row,) = await env.queries.list_ai_usage(service_name="web", days=1)
    assert (row.requests, row.prompt_tokens) == (2, SQLITE_MAX_INT)
    assert isinstance(row.prompt_tokens, int)


async def test_usage_filters_by_service_and_bounds_days(env):
    await env.queries.record_ai_usage("web", "fast", prompt_tokens=3, completion_tokens=5)
    await env.queries.record_ai_usage("api", "fast", upstream_error=True)
    resp = await env.client.get("/ai-gateway/usage?service=web", headers=_auth(RO_RAW))
    rows = resp.json()["usage"]
    assert [(r["service_name"], r["prompt_tokens"], r["completion_tokens"]) for r in rows] == [
        ("web", 3, 5)
    ]
    for days in (0, 91):
        resp = await env.client.get(f"/ai-gateway/usage?days={days}", headers=_auth(RO_RAW))
        assert resp.status_code == 422
