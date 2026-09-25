"""Exercise project-only link authority through real REST and MCP dispatch."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from nerdit.config.settings import NerditSettings
from nerdit.core.project_identity import PROJECT_DELEGATION_TOOLS, PROJECT_MCP_PATH
from nerdit.daemon.appfactory import _McpMount
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.projects import router
from nerdit.db.models import Job, JobKind, JobStatus
from nerdit.mcp import server, transport
from nerdit.mcp.tools import ALL_TOOLS
from tests.test_apply_route import A_RAW, LEGACY, _zip
from tests.test_apply_route import env as env  # noqa: F401
from tests.test_mcp_http import _MCP_HEADERS, _rpc, _sse_result

LINK = "test-link-capability"
OWNER = "link:test-node"
HEADER = "x-nerdit-project-id"


@pytest.fixture
async def delegated(env, monkeypatch):
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.state = env.app.state
    app.state.link_manager = SimpleNamespace(
        validate_capability=lambda token, role: token == LINK and role == "submitter",
        status=lambda: SimpleNamespace(node_id="test-node", state="connected"),
    )
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: env.queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: env.queries)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: env.queries)
    app.add_middleware(RequestIdMiddleware)
    settings = NerditSettings()
    settings.daemon.port = 9321
    saved = transport._HTTP_MODE, transport._HTTP_HOST, transport._HTTP_PORT
    mcp_app, mcp = server.build_http_app(settings)
    app.router.routes.append(_McpMount("/api/mcp", app=mcp_app))
    app.router.routes.append(_McpMount(PROJECT_MCP_PATH, app=mcp_app))
    real_client = transport.NerditClient

    class InProcessClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.ASGITransport(app=app)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(transport, "NerditClient", InProcessClient)
    project = await env.queries.create_project("asso", OWNER)
    other = await env.queries.create_project("other", OWNER)
    local = await env.queries.create_project("local", "tok-a")
    headers = {"Authorization": f"Bearer {LINK}", HEADER: project.id}
    ready, stop = asyncio.Event(), asyncio.Event()

    async def lifespan():
        async with mcp.session_manager.run():
            ready.set()
            await stop.wait()

    task = asyncio.create_task(lifespan())
    await ready.wait()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost:9321"
        ) as client:
            yield SimpleNamespace(
                app=app,
                client=client,
                queries=env.queries,
                db=env.db,
                project=project,
                other=other,
                local=local,
                headers=headers,
                extract=env.extract,
                clone=env.clone,
                mgr=app.state.secret_manager,
            )
    finally:
        stop.set()
        await task
        transport._HTTP_MODE, transport._HTTP_HOST, transport._HTTP_PORT = saved


async def _mcp(d, tool, arguments, *, project=None, key=None):
    headers = {**_MCP_HEADERS, **d.headers}
    if project is not None:
        headers[HEADER] = project
    if key is not None:
        headers["Idempotency-Key"] = key
    return await d.client.post(
        PROJECT_MCP_PATH, json=_rpc("tools/call", name=tool, arguments=arguments), headers=headers
    )


async def _job(d, *, config=None, owner=OWNER, name="asso", project=None, ident=None):
    return await d.queries.create_job(
        Job(
            id=ident or f"job-{name}",
            kind=JobKind.service,
            status=JobStatus.stopped,
            service_name=name,
            project_id=project or d.project.id,
            project="asso",
            environment="production",
            service="web",
            submitted_by_token=owner,
            config=json.dumps(config or {}),
        )
    )


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(HEADER, "")],
        [(HEADER, "asso")],
        [(HEADER, "prj_aaaaaaaaaaaaaaaa"), (HEADER, "prj_aaaaaaaaaaaaaaaa")],
    ],
)
async def test_project_mount_requires_one_valid_assertion(delegated, headers):
    d = delegated
    result = await d.client.post(
        PROJECT_MCP_PATH,
        json=_rpc("tools/list"),
        headers=[("Authorization", f"Bearer {LINK}"), *headers],
    )
    assert result.status_code == 403
    assert result.json()["code"] == "project.delegation_forbidden"


@pytest.mark.parametrize("token", [None, LEGACY, A_RAW, "expired-link"])
@pytest.mark.parametrize("open_auth", [False, True])
async def test_carrier_never_turns_local_or_invalid_auth_into_link(delegated, token, open_auth):
    d = delegated
    if open_auth:
        middleware = next(m for m in d.app.user_middleware if m.cls is ScopedTokenAuthMiddleware)
        middleware.kwargs["token"] = None
    headers = {HEADER: d.project.id}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    for path in (PROJECT_MCP_PATH, f"/api/projects/{d.project.id}", "/api/health"):
        result = await d.client.post(path, json=_rpc("tools/list"), headers=headers)
        assert result.status_code == 403, result.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/projects"),
        ("POST", "/api/projects"),
        ("GET", "/api/services"),
        ("GET", "/api/services/asso"),
        ("GET", "/api/services/job-asso/logs"),
        ("POST", "/api/services/job-asso/run"),
        ("GET", "/api/capabilities"),
        ("GET", "/api/config/daemon"),
        ("GET", "/api/secrets/shared"),
        ("GET", "/api/workspaces/asso"),
        ("POST", "/api/mcp"),
        ("POST", "/api/project-mcp/escape"),
        ("GET", "/api/projects/asso"),
        ("GET", "/projects/asso"),
        ("DELETE", "/api/projects/{project}"),
        ("PATCH", "/api/projects/{project}"),
        ("POST", "/api/projects/{project}/apply"),
        ("GET", "/api/projects/{project}/variables"),
        ("GET", "/api/projects/{other}"),
        ("PUT", "/api/projects/{other}/variables"),
    ],
)
async def test_rest_surface_denies_every_unlisted_path(delegated, method, path):
    d = delegated
    path = path.format(project=d.project.id, other=d.other.id)
    result = await d.client.request(method, path, headers=d.headers)
    assert result.status_code == 403, result.text
    assert "other" not in result.text


async def test_mcp_list_exposes_only_the_reviewed_surface(delegated):
    d = delegated
    result = await d.client.post(
        PROJECT_MCP_PATH, json=_rpc("tools/list"), headers={**_MCP_HEADERS, **d.headers}
    )
    assert result.status_code == 200, result.text
    assert {t["name"] for t in _sse_result(result)["result"]["tools"]} == PROJECT_DELEGATION_TOOLS
    node = await d.client.post(
        "/api/mcp",
        json=_rpc("tools/list"),
        headers={**_MCP_HEADERS, "Authorization": f"Bearer {LINK}"},
    )
    assert len(_sse_result(node)["result"]["tools"]) == 59


@pytest.mark.parametrize(
    "tool", [f.__name__ for f in ALL_TOOLS if f.__name__ not in PROJECT_DELEGATION_TOOLS]
)
async def test_every_unlisted_tool_is_refused_before_its_body(delegated, monkeypatch, tool):
    def forbidden_client():
        pytest.fail("unlisted MCP tool body ran")

    # Tool bodies import this function directly, so patch the source of each registered function.
    fn = next(f for f in ALL_TOOLS if f.__name__ == tool)
    monkeypatch.setitem(fn.__globals__, "_request_client", forbidden_client)
    result = await _mcp(delegated, tool, {"path": "/private/path"})
    assert result.status_code == 200, result.text
    assert "project.delegation_forbidden" in result.text


@pytest.mark.parametrize(
    "method", ["resources/list", "resources/read", "prompts/list", "prompts/get"]
)
async def test_unlisted_protocol_handlers_are_not_reached(delegated, method):
    d = delegated
    result = await d.client.post(
        PROJECT_MCP_PATH, json=_rpc(method), headers={**_MCP_HEADERS, **d.headers}
    )
    assert result.status_code == 403


@pytest.mark.parametrize("method", [None, [], {}])
async def test_malformed_protocol_methods_fail_closed(delegated, method):
    d = delegated
    result = await d.client.post(
        PROJECT_MCP_PATH,
        json={"jsonrpc": "2.0", "id": 1, "method": method},
        headers={**_MCP_HEADERS, **d.headers},
    )
    assert result.status_code == 403


async def test_inner_rest_keeps_project_assertion_and_concurrent_calls_do_not_mix(delegated):
    d = delegated
    results = await asyncio.gather(
        _mcp(d, "get_project", {"name": d.project.id}),
        _mcp(d, "get_project", {"name": d.other.id}, project=d.other.id),
        _mcp(d, "get_project", {"name": d.other.id}),
        _mcp(d, "get_project", {"name": "asso"}),
    )
    assert d.project.id in results[0].text and "delegation_forbidden" not in results[0].text
    assert d.other.id in results[1].text and "delegation_forbidden" not in results[1].text
    assert all("project.delegation_forbidden" in r.text for r in results[2:])
    assert transport._REQUEST_PROJECT_ID.get() is None


async def test_write_keeps_local_owner_and_project_cache_namespace(delegated):
    d = delegated
    body = {"values": {"KEY": "value"}}
    path = f"/api/projects/{d.project.id}/variables"
    node_headers = {"Authorization": f"Bearer {LINK}", "Idempotency-Key": "same-key"}
    node = await d.client.put(path, json=body, headers=node_headers)
    assert node.status_code == 200
    first = await d.client.put(
        path, json=body, headers={**d.headers, "Idempotency-Key": "same-key"}
    )
    assert first.status_code == 200 and "Idempotent-Replay" not in first.headers
    again = await d.client.put(
        path, json=body, headers={**d.headers, "Idempotency-Key": "same-key"}
    )
    assert again.headers["Idempotent-Replay"] == "true"
    foreign = await d.client.put(
        f"/api/projects/{d.local.id}/variables",
        json=body,
        headers={**d.headers, HEADER: d.local.id},
    )
    assert foreign.status_code == 403
    assert await d.queries.delete_project_checked(d.project.id) == []
    replacement = await d.queries.create_project("asso", OWNER)
    retired = await d.client.put(
        path, json=body, headers={**d.headers, "Idempotency-Key": "same-key"}
    )
    assert retired.status_code == 404 and "Idempotent-Replay" not in retired.headers
    assert await d.queries.list_variable_flags(replacement.id) == []


@pytest.mark.parametrize("source", ["archive", "repo", "workspace"])
@pytest.mark.parametrize("dry_run", [False, True])
async def test_delegation_cannot_enter_shared_builder(delegated, monkeypatch, source, dry_run):
    import nerdit.daemon.routes.projects as projects

    d = delegated
    monkeypatch.setattr(
        projects,
        "_read_declaration",
        AsyncMock(side_effect=AssertionError("delegated apply reached source work")),
    )
    kwargs = {"files": {"archive": ("app.zip", _zip(), "application/zip")}}
    if source != "archive":
        kwargs = {
            "data": {"repo_url": "https://github.com/acme/asso"}
            if source == "repo"
            else {"workspace": "true"}
        }
    result = await d.client.post(
        f"/api/projects/{d.project.id}/apply",
        params={"dry_run": str(dry_run).lower()},
        headers=d.headers,
        **kwargs,
    )
    assert result.status_code == 403, result.text
    assert result.json()["code"] == "project.delegation_forbidden"
    d.clone.assert_not_awaited()
    d.extract.assert_not_awaited()
    assert await d.queries.list_project_services(d.project.id) == []


async def test_node_grants_keep_project_apply(delegated):
    d = delegated
    response = await d.client.post(
        f"/api/projects/{d.project.id}/apply",
        files={"archive": ("app.zip", _zip(), "application/zip")},
        headers={"Authorization": f"Bearer {LINK}"},
    )
    assert response.status_code == 200, response.text
    assert len(await d.queries.list_project_services(d.project.id)) == 2


async def test_project_reads_do_not_resolve_external_readiness(delegated, monkeypatch):
    import nerdit.daemon.routes.service_diagnose as diagnosis

    d = delegated
    await _job(d, config={"ai": {"default": {"provider": "ollama", "model": "foreign-model"}}})
    lookup = AsyncMock(side_effect=AssertionError("foreign model inventory read"))
    monkeypatch.setattr(d.queries, "get_model_by_ref", lookup)
    monkeypatch.setattr(
        diagnosis, "_classify_bindings", AsyncMock(side_effect=AssertionError("bindings resolved"))
    )
    result = await d.client.get(f"/api/projects/{d.project.id}", headers=d.headers)
    assert result.status_code == 200, result.text
    assert result.json()["resources"][0]["ready"] is None
    result = await d.client.get(
        f"/api/projects/{d.project.id}/services/web/diagnose", headers=d.headers
    )
    assert result.status_code == 200, result.text
    assert result.json()["bindings"] == {"waiting": False, "messages": []}


async def test_project_read_redacts_embedded_ai_url_credentials(delegated):
    d = delegated
    canary = "pw-CANARY-project-read"
    await _job(
        d,
        config={
            "ai": {
                "cheap": {
                    "provider": "api",
                    "model": "remote",
                    "base_url": f"https://svc:{canary}@api.example.com/v1",
                }
            }
        },
    )
    result = await d.client.get(f"/api/projects/{d.project.id}", headers=d.headers)
    assert result.status_code == 200, result.text
    assert result.json()["resources"][0]["target"] == "https://api.example.com/v1"
    assert canary not in result.text
    mcp = await _mcp(d, "get_project", {"name": d.project.id})
    assert mcp.status_code == 200, mcp.text
    assert "https://api.example.com/v1" in mcp.text
    assert canary not in mcp.text


async def test_get_rejects_job_replacement_during_label_projection(delegated, monkeypatch):
    import nerdit.daemon.routes.projects as projects

    d = delegated
    old = await _job(d)
    original = projects._service_views

    async def replace_after_projection(request, jobs):
        views = await original(request, jobs)
        await d.db.conn.execute("DELETE FROM jobs WHERE id = ?", (old.id,))
        await d.db.conn.commit()
        await _job(d, owner="tok-a", project=d.other.id, ident="replacement-job")
        return views

    monkeypatch.setattr(projects, "_service_views", replace_after_projection)
    result = await d.client.get(f"/api/projects/{d.project.id}", headers=d.headers)
    assert result.status_code == 404, result.text


async def test_project_and_variable_native_tools_remain_usable(delegated):
    d = delegated
    variable = await _mcp(d, "set_variable", {"project": d.project.id, "values": {"MODE": "ok"}})
    assert variable.status_code == 200, variable.text
    assert json.loads(_sse_result(variable)["result"]["content"][0]["text"])["project"] == "asso"
    job = await _job(d)
    resolved = await _mcp(d, "resolve_variables", {"project": d.project.id})
    assert resolved.status_code == 200 and "MODE" in resolved.text
    variable = await _mcp(
        d, "set_variable", {"project": d.project.id, "service": "web", "values": {"LOCAL": "ok"}}
    )
    assert "delegation_forbidden" not in variable.text, variable.text
    await d.queries.append_log(job.id, "project-owned-log")
    logs = await _mcp(d, "project_logs", {"project": d.project.id})
    assert "project-owned-log" in logs.text
    diagnosis = await _mcp(d, "diagnose_project", {"project": d.project.id})
    assert "remediation" in diagnosis.text and "error" not in _sse_result(diagnosis)["result"]


@pytest.mark.parametrize("read", [False, True])
async def test_service_variables_require_a_live_project_member(delegated, read):
    d = delegated
    assert await d.queries.mint_secret_claim("asso", OWNER)
    d.mgr.set("asso", {"RETIRED_KEY": "retired-private-value"})
    path = f"/api/projects/{d.project.id}/variables"
    if read:
        result = await d.client.get(path + "/resolve", params={"service": "web"}, headers=d.headers)
    else:
        result = await d.client.put(
            path, json={"values": {"NEW": "value"}}, params={"service": "web"}, headers=d.headers
        )
    assert result.status_code == 403, result.text
    assert "RETIRED_KEY" not in result.text and "retired-private-value" not in result.text
    assert d.mgr.load("asso") == {"RETIRED_KEY": "retired-private-value"}
