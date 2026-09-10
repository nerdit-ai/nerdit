"""Tests for the minimal MCP server (P1 / S10).

These cover three layers without requiring the optional ``mcp`` extra:

* the CLI import firewall (graceful degradation when ``mcp`` is absent),
* the pure tool implementations driven by an ``httpx`` mock transport
  (error normalization, idempotency key, ``limit``/``tail`` bounds), and
* the client changes (``cluster_stats`` path, ``Idempotency-Key`` header).

A final test exercises real FastMCP wiring, guarded by ``importorskip`` so it
runs only where ``nerdit[mcp]`` is installed.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import uuid
from pathlib import Path

import httpx
import pytest

from nerdit.cli.client import NerditClient
from nerdit.mcp import server
from nerdit.mcp.tools import exposure as exposure_tools
from nerdit.mcp.tools import services as services_tools
from nerdit.mcp.tools import workspaces as workspaces_tools


def _client(handler, *, token: str | None = None) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=token,
        transport=httpx.MockTransport(handler),
    )


# --- import firewall / graceful degradation --------------------------------


def test_mcp_command_exits_when_extra_missing(monkeypatch, capsys):
    from nerdit.cli.commands.mcp import mcp

    # Force the "extra not installed" branch regardless of the environment.
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as exc_info:
        mcp()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "mcp" in err
    assert "pip install" in err


def test_mcp_command_importable_without_extra():
    # Importing the command module must never require the optional dependency.
    from nerdit.cli.commands import mcp as mcp_cmd

    assert callable(mcp_cmd.mcp)


def test_server_module_importable_without_extra():
    # The server module is import-safe; only build_server() needs FastMCP.
    assert hasattr(server, "build_server")
    assert hasattr(server, "run")


def test_server_facade_reexports_every_shared_bound():
    """``server`` promises every tool-facing name stays importable at this path.

    The façade is the back-compat import surface left behind by the WP24 split,
    so a bound added to ``tools/_shared`` and used by a tool signature has to be
    re-exported too — the P20 run bounds were the ones that slipped.
    """
    from nerdit.mcp.tools import _shared

    for name in dir(_shared):
        if name.startswith(("DEFAULT_", "MAX_")):
            assert getattr(server, name) == getattr(_shared, name), f"{name} not re-exported"


# --- _call error normalization ---------------------------------------------


@pytest.mark.asyncio
async def test_call_passes_success_through():
    async def coro():
        return {"ok": True}

    assert await server._call(coro()) == {"ok": True}


@pytest.mark.asyncio
async def test_call_normalizes_daemon_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "code": "not_found",
                "message": "no such job",
                "request_id": "req-123",
                "detail": "no such job",
            },
        )

    client = _client(handler)
    result = await server._list_gpus_impl(client)
    assert result == {
        "error": {
            "code": "not_found",
            "message": "no such job",
            "request_id": "req-123",
            "status": 404,
        }
    }


@pytest.mark.asyncio
async def test_call_merges_hint_and_envelope_extras():
    """The actionable half of an error must survive normalization.

    ``hint`` holds the remediation text and several codes attach a key the
    agent is meant to branch on (``not_ready_kind`` here; ``dependents`` on
    ``resource.in_use``, ``log_tail`` on ``run.interrupted``). Dropping them
    left an MCP caller with a bare code — strictly less than a REST caller.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "run.not_ready",
                "message": "the service is not ready to run a command",
                "hint": "deploy it first",
                "detail": "not ready",
                "request_id": "req-9",
                "not_ready_kind": "secrets",
            },
        )

    result = await server._list_gpus_impl(_client(handler))
    assert result["error"] == {
        "code": "run.not_ready",
        "message": "the service is not ready to run a command",
        "request_id": "req-9",
        "status": 409,
        "hint": "deploy it first",
        "not_ready_kind": "secrets",
    }


@pytest.mark.asyncio
async def test_call_never_merges_diagnostics_into_an_agent_visible_error():
    """``diagnostics`` must be dropped with ``detail`` — it is the same payload.

    ``_validation_exception_handler`` writes the pydantic error list into BOTH
    fields, and a v2 error entry carries the offending ``input`` — the caller's
    own submitted values. On HTTP that echo is a recorded, accepted limitation;
    merging it here would widen it onto the MCP surface, where a 422 on
    ``run_command(env=…)`` or ``set_secret`` would hand the rejected secret back
    into an agent's transcript. Regression guard: the extras-merge that lets
    ``hint``/``not_ready_kind`` through must never let this through with them.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        errors = [{"loc": ["body", "env"], "msg": "bad key", "input": {"BAD=KEY": "s3cr3t"}}]
        return httpx.Response(
            422,
            json={
                "code": "validation_error",
                "message": "Invalid request",
                "detail": errors,
                "diagnostics": errors,
                "request_id": "req-10",
            },
        )

    result = await server._list_gpus_impl(_client(handler))
    assert "diagnostics" not in result["error"]
    assert "s3cr3t" not in json.dumps(result)


@pytest.mark.asyncio
async def test_call_never_lets_an_extra_overwrite_a_fixed_field():
    """A hostile/odd envelope must not be able to rewrite the four fixed keys.

    ``status`` is ours (the real HTTP status), not the body's; ``detail`` is
    already folded into ``message``. Neither may be re-merged on top.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"code": "forbidden", "detail": "nope", "status": 200, "extra": 1}
        )

    result = await server._list_gpus_impl(_client(handler))
    assert result["error"]["status"] == 403
    assert result["error"]["message"] == "nope"  # from detail, not re-merged as a key
    assert "detail" not in result["error"]
    assert result["error"]["extra"] == 1


@pytest.mark.asyncio
async def test_call_normalizes_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(handler)
    result = await server._cluster_stats_impl(client)
    assert result["error"]["code"] == "connection_error"
    assert result["error"]["status"] is None


# --- read tools -------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_gpus_impl():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/gpus"
        return httpx.Response(200, json=[{"id": "GPU-0"}])

    result = await server._list_gpus_impl(_client(handler))
    assert result == [{"id": "GPU-0"}]


@pytest.mark.asyncio
async def test_cluster_stats_impl_hits_api_prefix():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"gpus_total": 2})

    result = await server._cluster_stats_impl(_client(handler))
    assert result == {"gpus_total": 2}
    assert seen["path"] == "/api/cluster/stats"


# --- service tools (P2) -----------------------------------------------------


@pytest.mark.asyncio
async def test_list_services_impl_clamps_limit_server_side():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"services": [], "next_cursor": None})

    # Over-cap limit is clamped down to MAX_SERVICE_LIMIT before hitting the API.
    result = await server._list_services_impl(_client(handler), limit=10_000)
    assert result == {"services": [], "next_cursor": None}
    assert seen["query"]["limit"] == str(server.MAX_SERVICE_LIMIT)


@pytest.mark.asyncio
async def test_list_services_impl_forwards_status_and_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"services": []})

    await server._list_services_impl(_client(handler), status="running", cursor="c-1")
    assert seen["query"]["status"] == "running"
    assert seen["query"]["cursor"] == "c-1"


@pytest.mark.asyncio
async def test_get_service_impl():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/services/web"
        return httpx.Response(200, json={"service_name": "web"})

    result = await server._get_service_impl(_client(handler), "web")
    assert result == {"service_name": "web"}


@pytest.mark.asyncio
async def test_service_logs_impl_clamps_tail_server_side():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[{"id": 1, "line": "hello"}])

    # Over-cap tail is clamped down to MAX_LOG_TAIL before hitting the API.
    result = await server._service_logs_impl(_client(handler), "web", tail=10_000)
    assert result == [{"id": 1, "line": "hello"}]
    assert seen["path"] == "/api/services/web/logs"
    assert seen["query"]["tail"] == str(server.MAX_LOG_TAIL)
    assert seen["query"]["since_id"] == "0"


@pytest.mark.asyncio
async def test_service_logs_impl_forwards_since_id_and_defaults_tail():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    await server._service_logs_impl(_client(handler), "web", since_id=42)
    assert seen["query"]["since_id"] == "42"
    assert seen["query"]["tail"] == str(server.DEFAULT_LOG_TAIL)


@pytest.mark.asyncio
async def test_service_logs_impl_honors_tail_client_side():
    def handler(request: httpx.Request) -> httpx.Response:
        # A misbehaving/older daemon that ignores ?tail= still gets bounded.
        return httpx.Response(200, json=[{"id": i} for i in range(10)])

    result = await server._service_logs_impl(_client(handler), "web", tail=2)
    assert [e["id"] for e in result] == [8, 9]  # last `tail` entries


@pytest.mark.asyncio
async def test_serve_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "svc-1"})

    result = await server._serve_impl(_client(handler), name="web", image="nginx:latest")
    assert result == {"id": "svc-1"}
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated


@pytest.mark.asyncio
async def test_serve_impl_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "svc-1"})

    await server._serve_impl(
        _client(handler), name="web", image="nginx:latest", idempotency_key="fixed-svc"
    )
    assert seen["idem"] == "fixed-svc"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("impl", "method", "expected_path"),
    [
        (server._stop_service_impl, "POST", "/api/services/web/stop"),
        (server._restart_service_impl, "POST", "/api/services/web/restart"),
        (server._remove_service_impl, "DELETE", "/api/services/web"),
    ],
)
async def test_service_lifecycle_impls_auto_idempotency_key(impl, method, expected_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(200, json={"ok": True})

    result = await impl(_client(handler), "web")
    assert result == {"ok": True}
    assert seen["method"] == method
    assert seen["path"] == expected_path
    assert seen["idem"] and len(seen["idem"]) >= 32  # auto-minted


@pytest.mark.asyncio
async def test_remove_service_impl_forwards_purge_and_force():
    """P14b WP-B1: the widened remove_service tool forwards ?purge / ?force."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"deleted": True})

    await server._remove_service_impl(
        _client(handler), "web", purge="secrets,data,images", force=True
    )
    assert seen["query"]["purge"] == "secrets,data,images"
    assert seen["query"]["force"] == "true"


@pytest.mark.asyncio
async def test_remove_service_impl_defaults_purge_secrets():
    """The default forwards purge=secrets and omits force."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"deleted": True})

    await server._remove_service_impl(_client(handler), "web")
    assert seen["query"]["purge"] == "secrets"
    assert "force" not in seen["query"]


# The run tool is reached through its domain module rather than the ``server``
# back-compat façade: it is the one impl the façade re-export list predates.
@pytest.mark.asyncio
async def test_run_command_impl_forwards_argv_env_and_mints_key():
    """P20 WP5: the run tool mints a key and forwards argv/env verbatim."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"exit_code": 0})

    await services_tools._run_command_impl(
        _client(handler),
        "web",
        command=["python", "manage.py", "migrate"],
        env={"DRY_RUN": "1"},
    )
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/services/web/run"
    assert seen["idem"] and len(seen["idem"]) >= 32  # auto-minted
    assert seen["body"]["command"] == ["python", "manage.py", "migrate"]
    assert seen["body"]["env"] == {"DRY_RUN": "1"}
    assert seen["body"]["timeout_s"] == 300
    assert seen["body"]["log_tail"] == 200


@pytest.mark.asyncio
async def test_run_command_impl_floors_timeout_but_clamps_log_tail():
    """The two bounds are deliberately asymmetric — pin it (plan invariants I4).

    ``log_tail``'s ceiling is the compile-time 200, so the tool mirrors the
    daemon's clamp. ``timeout_s``'s ceiling is config-driven
    (``[services].run_timeout_max_s``), so the tool only floors it: a
    client-side clamp would silently truncate the request whenever an operator
    raises the cap, and the authoritative 422 ``run.timeout_too_large`` could
    then never reach an MCP caller.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"exit_code": 0})

    await services_tools._run_command_impl(
        _client(handler), "web", command=["sh"], timeout_s=99_999, log_tail=5_000
    )
    assert seen["body"]["timeout_s"] == 99_999  # passed through, NOT clamped
    assert seen["body"]["log_tail"] == 200

    await services_tools._run_command_impl(
        _client(handler), "web", command=["sh"], timeout_s=0, log_tail=0
    )
    assert seen["body"]["timeout_s"] == 1  # floored
    assert seen["body"]["log_tail"] == 1


@pytest.mark.asyncio
async def test_run_command_tool_forwards_caller_supplied_key(monkeypatch):
    """The agent-facing tool must accept a key and forward it VERBATIM.

    Retry-safety on a call that blocks for minutes is only reachable if the
    caller can reuse its own key: a mint per invocation makes each retry a
    fresh execution (a migration run twice). Driven through the public
    ``run_command`` wrapper — not the impl — because it is the wrapper's
    signature that becomes the tool's inputSchema, i.e. the thing an agent can
    actually set.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"exit_code": 0})

    client = _client(handler)
    monkeypatch.setattr(services_tools, "_request_client", lambda: client)

    await services_tools.run_command(
        "web", command=["python", "manage.py", "migrate"], idempotency_key="agent-key-1"
    )
    assert seen["idem"] == "agent-key-1"  # forwarded, NOT re-minted

    await services_tools.run_command("web", command=["sh"])
    assert seen["idem"] and seen["idem"] != "agent-key-1"  # mint stays the fallback


# --- model tools (P5) --------------------------------------------------------


@pytest.mark.asyncio
async def test_serve_model_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "mdl-1", "kind": "model"})

    result = await server._serve_model_impl(_client(handler), "llama3.1:8b")
    assert result == {"id": "mdl-1", "kind": "model"}
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/models"
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated


@pytest.mark.asyncio
async def test_serve_model_impl_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "mdl-1"})

    await server._serve_model_impl(
        _client(handler), "llama3.1:8b", gpus=1, name="llama", idempotency_key="fixed-mdl"
    )
    assert seen["idem"] == "fixed-mdl"


@pytest.mark.asyncio
async def test_serve_model_impl_forwards_engine_bounds():
    """P21 D4: the vLLM-only bounds reach the daemon body; absent ⇒ absent."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "mdl-1"})

    await server._serve_model_impl(
        _client(handler),
        "Qwen/Qwen2.5-0.5B-Instruct",
        gpus=1,
        backend="vllm",
        max_model_len=2048,
        gpu_memory_utilization=0.8,
    )
    assert seen["body"]["max_model_len"] == 2048
    assert seen["body"]["gpu_memory_utilization"] == 0.8

    await server._serve_model_impl(_client(handler), "llama3.1:8b")
    assert "max_model_len" not in seen["body"]
    assert "gpu_memory_utilization" not in seen["body"]


@pytest.mark.asyncio
async def test_serve_model_tool_schema_exposes_engine_bounds():
    """The tool's introspected inputSchema is the agent-facing contract."""
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = await mcp_server.list_tools()
    schema = next(t.inputSchema for t in tools if t.name == "serve_model")
    assert "max_model_len" in schema["properties"]
    assert "gpu_memory_utilization" in schema["properties"]
    # Optional: neither may become a required argument.
    assert "max_model_len" not in schema.get("required", [])
    assert "gpu_memory_utilization" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_list_models_impl_clamps_limit_server_side():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"models": [], "next_cursor": None})

    # Over-cap limit is clamped down to MAX_MODEL_LIMIT before hitting the API.
    result = await server._list_models_impl(_client(handler), limit=10_000)
    assert result == {"models": [], "next_cursor": None}
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/models"
    assert seen["query"]["limit"] == str(server.MAX_MODEL_LIMIT)


@pytest.mark.asyncio
async def test_list_models_impl_defaults_limit_and_forwards_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"models": []})

    await server._list_models_impl(_client(handler), cursor="c-1")
    assert seen["query"]["limit"] == str(server.DEFAULT_MODEL_LIMIT)
    assert seen["query"]["cursor"] == "c-1"


# --- database tools (P15) ----------------------------------------------------


@pytest.mark.asyncio
async def test_create_database_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "db-1", "name": "pg", "backend": "postgres"})

    result = await server._create_database_impl(_client(handler), backend="postgres")
    assert result == {"id": "db-1", "name": "pg", "backend": "postgres"}
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/databases"
    assert seen["body"] == {"backend": "postgres"}
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated


@pytest.mark.asyncio
async def test_create_database_impl_respects_explicit_key_and_name():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "db-1"})

    await server._create_database_impl(
        _client(handler), backend="redis", name="cache", idempotency_key="fixed-db"
    )
    assert seen["idem"] == "fixed-db"
    assert seen["body"] == {"backend": "redis", "name": "cache"}


@pytest.mark.asyncio
async def test_create_database_impl_response_carries_no_password():
    # D-B: the minted credential never rides the create response body.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={"id": "db-1", "name": "pg", "backend": "postgres", "db_ready": False},
        )

    result = await server._create_database_impl(_client(handler), backend="postgres")
    assert "password" not in json.dumps(result).lower()


@pytest.mark.asyncio
async def test_list_databases_impl_clamps_limit_server_side():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    # Over-cap limit is clamped down to MAX_DATABASE_LIMIT before hitting the API.
    result = await server._list_databases_impl(_client(handler), limit=10_000)
    assert result == {"items": [], "next_cursor": None}
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/databases"
    assert seen["query"]["limit"] == str(server.MAX_DATABASE_LIMIT)


@pytest.mark.asyncio
async def test_list_databases_impl_defaults_limit_and_forwards_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"items": []})

    await server._list_databases_impl(_client(handler), cursor="c-1")
    assert seen["query"]["limit"] == str(server.DEFAULT_DATABASE_LIMIT)
    assert seen["query"]["cursor"] == "c-1"


# --- database dump tools (P37, §1.7) -----------------------------------------


@pytest.mark.asyncio
async def test_dump_database_impl_auto_idempotency_key_and_default_timeout():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "service_name": "pg",
                "dump": "nerdit-dump-pg-20260907T101500Z-a1b2c3.tar.gz",
                "size_bytes": 4096,
                "sha256": "0" * 64,
                "engine": "postgres",
                "format": "pg_custom",
                "created_at": "2026-09-07T10:15:00Z",
                "duration_s": 1.5,
            },
        )

    result = await server._dump_database_impl(_client(handler), name="pg")
    assert result["dump"] == "nerdit-dump-pg-20260907T101500Z-a1b2c3.tar.gz"
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/databases/pg/dump"
    # The route's Idempotency-Key is mandatory in-route, so the tool mints one
    # PER CALL. That satisfies the route without making a retry a replay: a
    # second call is a second real dump, which is why the tool description tells
    # the agent to check `list_database_dumps` before retrying.
    assert seen["idem"] and len(seen["idem"]) >= 32
    # The tool default (300) is the CLIENT budget, deliberately below the route's
    # own 900 (D-P37-8).
    assert seen["body"] == {"timeout_s": server.DEFAULT_DUMP_TIMEOUT_S}


@pytest.mark.asyncio
async def test_dump_database_impl_forwards_timeout_unclamped():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"service_name": "pg", "dump": "d.tar.gz"})

    # Way above the daemon's default cap: NOT clamped here, because the ceiling
    # is config-driven and the authoritative 422 must be able to reach the agent.
    await server._dump_database_impl(
        _client(handler), name="pg", timeout_s=99_999, idempotency_key="fixed-dump"
    )
    assert seen["idem"] == "fixed-dump"
    assert seen["body"] == {"timeout_s": 99_999}


@pytest.mark.asyncio
async def test_dump_database_impl_floors_timeout_at_one():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"service_name": "pg", "dump": "d.tar.gz"})

    # The route's field is ``ge=1``; a 0 would be a 422 the agent cannot act on.
    await server._dump_database_impl(_client(handler), name="pg", timeout_s=0)
    assert seen["body"] == {"timeout_s": 1}


@pytest.mark.asyncio
async def test_dump_database_impl_readonly_403_becomes_forbidden_envelope():
    # The dump write is owner-or-admin + scope; a readonly token is refused by
    # the coarse write gate with a non-envelope body, so the status mapping is
    # what the agent sees.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    result = await server._dump_database_impl(_client(handler), name="pg")
    assert result["error"]["code"] == "forbidden"
    assert result["error"]["status"] == 403


@pytest.mark.asyncio
async def test_dump_database_impl_passes_the_daemon_envelope_through():
    # D-P37-11's hint and D-P37-10's detail are the actionable half of a
    # refusal: ``_call`` copies envelope extras through, and the tool must not
    # swallow them.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "dump.insufficient_disk",
                "message": "not enough free space to dump 'pg'",
                "hint": "estimate from the volume size",
                "detail": {"required_bytes": 10, "free_bytes": 1},
            },
        )

    result = await server._dump_database_impl(_client(handler), name="pg")
    assert result["error"]["code"] == "dump.insufficient_disk"
    assert result["error"]["status"] == 409
    assert result["error"]["hint"] == "estimate from the volume size"


@pytest.mark.asyncio
async def test_list_database_dumps_impl_reads_the_listing():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "service_name": "pg",
                "dumps": [
                    {
                        "dump": "nerdit-dump-pg-20260907T101500Z-a1b2c3.tar.gz",
                        "size_bytes": 4096,
                        "created_at": "2026-09-07T10:15:00Z",
                    }
                ],
            },
        )

    result = await server._list_database_dumps_impl(_client(handler), name="pg")
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/databases/pg/dumps"
    assert result["dumps"][0]["size_bytes"] == 4096
    # Metadata only: no path, no credential, nothing but the handle.
    assert "path" not in json.dumps(result)


@pytest.mark.asyncio
async def test_list_database_dumps_impl_403_becomes_forbidden_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    result = await server._list_database_dumps_impl(_client(handler), name="pg")
    assert result["error"]["code"] == "forbidden"
    assert result["error"]["status"] == 403


@pytest.mark.asyncio
async def test_database_dump_tools_are_registered_with_their_args():
    """The pair is registered and its schema is the agent-facing contract."""
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = {t.name: t for t in await mcp_server.list_tools()}
    assert "dump_database" in tools
    assert "list_database_dumps" in tools
    # Deliberately no restore tool (D-P37-12): destructive, CLI/REST only.
    assert "restore_database" not in tools
    assert "restore_database_dump" not in tools

    dump_schema = tools["dump_database"].inputSchema
    assert set(dump_schema["properties"]) == {"name", "timeout_s"}
    assert dump_schema.get("required") == ["name"]
    assert dump_schema["properties"]["timeout_s"]["default"] == server.DEFAULT_DUMP_TIMEOUT_S
    assert set(tools["list_database_dumps"].inputSchema["properties"]) == {"name"}

    # D-P37-12: the description carries the custody line, the retention line and
    # the "a longer dump still completes server-side" line — an agent that only
    # reads the description must still know where the data went and that a
    # client-side timeout is not a failure.
    description = inspect.cleandoc(tools["dump_database"].description or "")
    assert "never served over" in description
    assert "dump_keep_last" in description
    assert "still completes" in description and "list_database_dumps" in description
    assert "owner-or-admin" in description


# --- client coupling points -------------------------------------------------


@pytest.mark.asyncio
async def test_create_service_sends_idempotency_header():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "x"})

    client = _client(handler)
    await client.create_service(name="demo", image="img:1", idempotency_key="k-1")
    assert seen["idem"] == "k-1"


@pytest.mark.asyncio
async def test_create_service_omits_idempotency_header_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "x"})

    client = _client(handler)
    await client.create_service(name="demo", image="img:1")
    assert seen["idem"] is None


@pytest.mark.asyncio
async def test_client_cluster_stats():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/cluster/stats"
        return httpx.Response(200, json={"services_up": 1})

    client = _client(handler)
    assert await client.cluster_stats() == {"services_up": 1}


# --- git deploy + app template store impls (P11.5) --------------------------


@pytest.mark.asyncio
async def test_deploy_git_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo"})

    result = await server._deploy_git_impl(
        _client(handler), repo_url="https://github.com/o/r", name="demo"
    )
    assert result == {"id": "svc-1", "name": "demo"}
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/deploy/git"
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated
    assert seen["body"] == {"repo_url": "https://github.com/o/r", "name": "demo"}


@pytest.mark.asyncio
async def test_deploy_git_impl_forwards_fields_and_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_git_impl(
        _client(handler),
        repo_url="https://github.com/o/r",
        name="demo",
        ref="v1.0.0",
        subdir="apps/web",
        env={"A": "1"},
        token_ref="${secrets.shared.GITHUB_TOKEN}",
        idempotency_key="fixed-git",
    )
    assert seen["idem"] == "fixed-git"
    assert seen["body"]["ref"] == "v1.0.0"
    assert seen["body"]["subdir"] == "apps/web"
    assert seen["body"]["env"] == {"A": "1"}
    assert seen["body"]["token_ref"] == "${secrets.shared.GITHUB_TOKEN}"


@pytest.mark.asyncio
async def test_list_app_templates_impl():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/app-templates"
        return httpx.Response(200, json=[{"id": "node-starter"}])

    result = await server._list_app_templates_impl(_client(handler))
    assert result == [{"id": "node-starter"}]


@pytest.mark.asyncio
async def test_deploy_template_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo"})

    result = await server._deploy_template_impl(
        _client(handler), "node-starter", name="demo", secrets={"K": "v"}
    )
    assert result == {"id": "svc-1", "name": "demo"}
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/app-templates/node-starter/deploy"
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated
    assert seen["body"]["name"] == "demo"
    assert seen["body"]["secrets"] == {"K": "v"}


@pytest.mark.asyncio
async def test_deploy_template_impl_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_template_impl(
        _client(handler), "node-starter", name="demo", idempotency_key="fixed-tpl"
    )
    assert seen["idem"] == "fixed-tpl"


# --- real FastMCP wiring (only with the extra) ------------------------------


# --- deploy + secrets impls (P4) --------------------------------------------


@pytest.mark.asyncio
async def test_deploy_impl_zips_and_posts(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "package.json").write_text('{"name":"demo"}')
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["ct"] = request.headers.get("content-type", "")
        return httpx.Response(201, json={"id": "svc-1", "name": "demo"})

    result = await server._deploy_impl(_client(handler), path=str(app), name="demo")
    assert result == {"id": "svc-1", "name": "demo"}
    assert seen["path"] == "/api/deploy"
    assert seen["idem"] and len(seen["idem"]) >= 32
    assert "multipart/form-data" in seen["ct"]


@pytest.mark.asyncio
async def test_deploy_impl_rollback_posts_no_zip():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"id": "svc-1"})

    result = await server._deploy_impl(_client(handler), name="demo", rollback=True)
    assert result == {"id": "svc-1"}
    assert seen["path"] == "/api/deploy/demo/rollback"


@pytest.mark.asyncio
async def test_deploy_impl_rollback_requires_name():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - not reached
        return httpx.Response(200, json={})

    result = await server._deploy_impl(_client(handler), rollback=True)
    assert result["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_deploy_impl_requires_path():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - not reached
        return httpx.Response(200, json={})

    result = await server._deploy_impl(_client(handler), name="demo")
    assert result["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_set_secret_impl_posts_values():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json={"service": "web", "keys": ["A"]})

    result = await server._set_secret_impl(_client(handler), "web", {"A": "1"})
    assert result == {"service": "web", "keys": ["A"]}
    assert seen["path"] == "/api/secrets/web"
    assert b'"values"' in seen["body"]


@pytest.mark.asyncio
async def test_list_secret_names_impl_returns_names_only():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        return httpx.Response(200, json={"service": "web", "keys": ["A", "B"]})

    result = await server._list_secret_names_impl(_client(handler), "web")
    assert result["keys"] == ["A", "B"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,expected_path",
    [("A", "/api/secrets/web/A"), (None, "/api/secrets/web")],
)
async def test_rm_secret_impl(key, expected_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(200, json={"service": "web", "deleted": key or True})

    await server._rm_secret_impl(_client(handler), "web", key=key)
    assert seen["path"] == expected_path
    assert seen["method"] == "DELETE"


@pytest.mark.asyncio
async def test_set_secret_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service": "web", "keys": ["A"]})

    await server._set_secret_impl(_client(handler), "web", {"A": "1"})
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated


@pytest.mark.asyncio
async def test_set_secret_impl_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service": "web", "keys": ["A"]})

    await server._set_secret_impl(_client(handler), "web", {"A": "1"}, idempotency_key="fixed-sec")
    assert seen["idem"] == "fixed-sec"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["A", None])
async def test_rm_secret_impl_auto_idempotency_key(key):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service": "web", "deleted": key or True})

    await server._rm_secret_impl(_client(handler), "web", key=key)
    assert seen["idem"] and len(seen["idem"]) >= 32  # auto-minted on both paths


@pytest.mark.asyncio
async def test_rm_secret_impl_respects_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service": "web", "deleted": True})

    await server._rm_secret_impl(_client(handler), "web", key="A", idempotency_key="fixed-del")
    assert seen["idem"] == "fixed-del"


# --- audit + config impls (P6) -----------------------------------------------


@pytest.mark.asyncio
async def test_get_audit_impl_clamps_limit_server_side():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"entries": [], "next_cursor": None})

    # Over-cap limit is clamped down to MAX_AUDIT_LIMIT before hitting the API.
    result = await server._get_audit_impl(_client(handler), limit=10_000)
    assert result == {"entries": [], "next_cursor": None}
    assert seen["path"] == "/api/audit"
    assert seen["query"]["limit"] == str(server.MAX_AUDIT_LIMIT)


@pytest.mark.asyncio
async def test_get_audit_impl_forwards_filters_and_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"entries": []})

    await server._get_audit_impl(
        _client(handler),
        action="deploy.create",
        result="ok",
        target="my-app",
        target_type="service",
        cursor="c-1",
    )
    assert seen["query"]["action"] == "deploy.create"
    assert seen["query"]["result"] == "ok"
    assert seen["query"]["target"] == "my-app"
    assert seen["query"]["target_type"] == "service"
    assert seen["query"]["cursor"] == "c-1"
    assert seen["query"]["limit"] == str(server.DEFAULT_AUDIT_LIMIT)


@pytest.mark.asyncio
async def test_get_audit_impl_admin_403_becomes_forbidden_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        # Non-envelope body: the code must fall back to the status mapping.
        return httpx.Response(403, text="Forbidden")

    result = await server._get_audit_impl(_client(handler))
    assert result["error"]["code"] == "forbidden"
    assert result["error"]["status"] == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("section", "expected_path"),
    [(None, "/api/config/daemon"), ("proxy", "/api/config/daemon/proxy")],
)
async def test_get_config_impl_paths(section, expected_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["method"] = request.method
        return httpx.Response(200, json={"section": section or "all"})

    result = await server._get_config_impl(_client(handler), section=section)
    assert result == {"section": section or "all"}
    assert seen["method"] == "GET"
    assert seen["path"] == expected_path


@pytest.mark.asyncio
async def test_set_config_impl_auto_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        seen["idem"] = request.headers.get("idempotency-key")
        seen["if_match"] = request.headers.get("if-match")
        return httpx.Response(200, json={"section": "proxy"})

    result = await server._set_config_impl(_client(handler), "proxy", {"enabled": True})
    assert result == {"section": "proxy"}
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/config/daemon/proxy"
    assert seen["query"] == {}  # no dry_run flag on a real write
    assert seen["if_match"] is None
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated


@pytest.mark.asyncio
async def test_set_config_impl_forwards_dry_run_if_match_and_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        seen["idem"] = request.headers.get("idempotency-key")
        seen["if_match"] = request.headers.get("if-match")
        return httpx.Response(200, json={"section": "proxy", "dry_run": True})

    await server._set_config_impl(
        _client(handler),
        "proxy",
        {"enabled": True},
        dry_run=True,
        if_match="v7",
        idempotency_key="fixed-cfg",
    )
    assert seen["query"]["dry_run"] == "true"
    assert seen["if_match"] == "v7"
    assert seen["idem"] == "fixed-cfg"


# --- apply_config / app-config impls (P7) ------------------------------------


@pytest.mark.asyncio
async def test_apply_config_impl_auto_fetches_etag_when_omitted():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": dict(request.url.params),
                "if_match": request.headers.get("if-match"),
                "idem": request.headers.get("idempotency-key"),
            }
        )
        if request.method == "GET":
            return httpx.Response(200, json=[{"section": "proxy", "values": {}, "etag": "etag-1"}])
        return httpx.Response(200, json={"applied": True, "etag": "etag-2"})

    result = await server._apply_config_impl(_client(handler), {"proxy": {"enabled": True}})
    assert result == {"applied": True, "etag": "etag-2"}
    assert seen[0]["method"] == "GET"
    assert seen[0]["path"] == "/api/config/daemon"
    assert seen[1]["method"] == "POST"
    assert seen[1]["path"] == "/api/config/daemon/apply"
    assert seen[1]["if_match"] == "etag-1"  # auto-fetched from the GET
    assert seen[1]["idem"] and len(seen[1]["idem"]) >= 32


@pytest.mark.asyncio
async def test_apply_config_impl_explicit_if_match_disables_auto_fetch():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"method": request.method, "if_match": request.headers.get("if-match")})
        return httpx.Response(200, json={"applied": True})

    await server._apply_config_impl(
        _client(handler), {"proxy": {"enabled": True}}, if_match="explicit-etag"
    )
    assert len(seen) == 1  # no GET fired
    assert seen[0]["if_match"] == "explicit-etag"


@pytest.mark.asyncio
async def test_apply_config_impl_dry_run_skips_etag_and_idempotency_key():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "query": dict(request.url.params),
                "if_match": request.headers.get("if-match"),
                "idem": request.headers.get("idempotency-key"),
            }
        )
        return httpx.Response(200, json={"applied": False})

    await server._apply_config_impl(_client(handler), {"proxy": {"enabled": True}}, dry_run=True)
    assert len(seen) == 1  # no GET auto-fetch on dry_run
    assert seen[0]["query"]["dry_run"] == "true"
    assert seen[0]["if_match"] is None
    assert seen[0]["idem"] is None


@pytest.mark.asyncio
async def test_apply_config_impl_propagates_etag_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden")

    result = await server._apply_config_impl(_client(handler), {"proxy": {"enabled": True}})
    assert result["error"]["code"] == "forbidden"


@pytest.mark.asyncio
async def test_get_app_config_impl():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config/apps/demo"
        return httpx.Response(200, json={"service_name": "demo", "etag": "e-1"})

    result = await server._get_app_config_impl(_client(handler), "demo")
    assert result == {"service_name": "demo", "etag": "e-1"}


@pytest.mark.asyncio
async def test_set_app_config_impl_auto_fetches_etag_when_omitted():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "if_match": request.headers.get("if-match"),
                "idem": request.headers.get("idempotency-key"),
            }
        )
        if request.method == "GET":
            return httpx.Response(200, json={"service_name": "demo", "etag": "app-etag-1"})
        return httpx.Response(200, json={"applied": True})

    result = await server._set_app_config_impl(_client(handler), "demo", "deploy", {"gpus": 1})
    assert result == {"applied": True}
    assert seen[0]["method"] == "GET"
    assert seen[0]["path"] == "/api/config/apps/demo"
    assert seen[1]["method"] == "PUT"
    assert seen[1]["path"] == "/api/config/apps/demo/deploy"
    assert seen[1]["if_match"] == "app-etag-1"
    assert seen[1]["idem"] and len(seen[1]["idem"]) >= 32


@pytest.mark.asyncio
async def test_set_app_config_impl_explicit_if_match_disables_auto_fetch():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({"method": request.method, "if_match": request.headers.get("if-match")})
        return httpx.Response(200, json={"applied": True})

    await server._set_app_config_impl(
        _client(handler), "demo", "deploy", {"gpus": 1}, if_match="explicit-etag"
    )
    assert len(seen) == 1  # no GET fired
    assert seen[0]["if_match"] == "explicit-etag"


@pytest.mark.asyncio
async def test_set_app_config_impl_dry_run_skips_etag_and_idempotency_key():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "query": dict(request.url.params),
                "if_match": request.headers.get("if-match"),
                "idem": request.headers.get("idempotency-key"),
            }
        )
        return httpx.Response(200, json={"applied": False})

    await server._set_app_config_impl(_client(handler), "demo", "deploy", {"gpus": 1}, dry_run=True)
    assert len(seen) == 1  # no GET auto-fetch on dry_run
    assert seen[0]["query"]["dry_run"] == "true"
    assert seen[0]["if_match"] is None
    assert seen[0]["idem"] is None


@pytest.mark.asyncio
async def test_set_app_config_impl_forwards_restart_flag():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"service_name": "demo", "etag": "e-1"})
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"applied": True, "restarted": True})

    result = await server._set_app_config_impl(
        _client(handler), "demo", "ai", {"default": {}}, restart=True
    )
    assert result == {"applied": True, "restarted": True}
    assert seen["query"]["restart"] == "true"


@pytest.mark.asyncio
async def test_set_app_config_impl_propagates_etag_fetch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"code": "not_found", "message": "no such app"})

    result = await server._set_app_config_impl(_client(handler), "ghost", "deploy", {"gpus": 1})
    assert result["error"]["code"] == "not_found"


# --- self-knowledge + observability impls (P13c) ----------------------------


@pytest.mark.asyncio
async def test_diagnose_service_impl_clamps_log_tail():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"remediation_code": "check_start_command"})

    result = await server._diagnose_service_impl(_client(handler), "demo", log_tail=99999)
    assert result == {"remediation_code": "check_start_command"}
    assert seen["path"] == "/api/services/demo/diagnose"
    assert seen["query"]["log_tail"] == str(server.MAX_DIAGNOSE_LOG_TAIL)


@pytest.mark.asyncio
async def test_capabilities_impl_passes_through():
    sentinel = {"version": "0.4.0", "role": "admin"}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/capabilities"
        return httpx.Response(200, json=sentinel)

    result = await server._capabilities_impl(_client(handler))
    assert result == sentinel


@pytest.mark.asyncio
async def test_doctor_impl_passes_through():
    sentinel = {"status": "ok", "checks": {"docker": {"status": "ok"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/doctor"
        return httpx.Response(200, json=sentinel)

    result = await server._doctor_impl(_client(handler))
    assert result == sentinel


@pytest.mark.asyncio
async def test_proxy_status_impl_passes_through():
    sentinel = {"state": "running", "tls": {"subjects": []}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/proxy/status"
        return httpx.Response(200, json=sentinel)

    result = await server._proxy_status_impl(_client(handler))
    assert result == sentinel


@pytest.mark.asyncio
async def test_list_routes_impl_clamps_limit_and_forwards_cursor():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"routes": [], "next_cursor": None})

    result = await server._list_routes_impl(_client(handler), cursor="abc", limit=999)
    assert result == {"routes": [], "next_cursor": None}
    assert seen["path"] == "/api/routes"
    assert seen["query"]["limit"] == str(server.MAX_ROUTE_LIMIT)
    assert seen["query"]["cursor"] == "abc"


# --- deploy dry-run + env null-delete (P13c) ---------------------------------


@pytest.mark.asyncio
async def test_deploy_impl_dry_run_omits_key_and_sets_param(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "package.json").write_text('{"name":"demo"}')
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": True})

    result = await server._deploy_impl(_client(handler), path=str(app), name="demo", dry_run=True)
    assert result == {"dry_run": True}
    assert "Idempotency-Key" not in seen["headers"]
    assert seen["query"]["dry_run"] == "true"


@pytest.mark.asyncio
async def test_deploy_impl_dry_run_rejects_rollback():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - not reached
        return httpx.Response(200, json={})

    result = await server._deploy_impl(_client(handler), name="demo", rollback=True, dry_run=True)
    assert result["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_deploy_impl_real_run_still_mints_key(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "package.json").write_text('{"name":"demo"}')
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_impl(_client(handler), path=str(app), name="demo")
    assert seen["idem"] and len(seen["idem"]) >= 32


@pytest.mark.asyncio
async def test_deploy_git_impl_dry_run_omits_key_and_sets_param():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": True})

    result = await server._deploy_git_impl(
        _client(handler), repo_url="https://github.com/o/r", name="demo", dry_run=True
    )
    assert result == {"dry_run": True}
    assert "Idempotency-Key" not in seen["headers"]
    assert seen["query"]["dry_run"] == "true"


@pytest.mark.asyncio
async def test_deploy_impl_env_null_passes_through(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "package.json").write_text('{"name":"demo"}')
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["content"] = request.content
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_impl(
        _client(handler), path=str(app), name="demo", env={"KEEP": "v", "DROP": None}
    )
    assert b'"DROP": null' in seen["content"]


@pytest.mark.asyncio
async def test_deploy_git_impl_env_null_passes_through():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_git_impl(
        _client(handler),
        repo_url="https://github.com/o/r",
        name="demo",
        env={"KEEP": "v", "DROP": None},
    )
    assert seen["body"]["env"] == {"KEEP": "v", "DROP": None}


@pytest.mark.asyncio
async def test_deploy_template_impl_env_null_passes_through():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    await server._deploy_template_impl(
        _client(handler), "node-starter", name="demo", env={"KEEP": "v", "DROP": None}
    )
    assert seen["path"] == "/api/app-templates/node-starter/deploy"
    assert seen["body"]["env"] == {"KEEP": "v", "DROP": None}


@pytest.mark.asyncio
async def test_system_disk_impl_is_a_plain_read():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"docker": None, "orphan_images": []})

    result = await server._system_disk_impl(_client(handler))
    assert result["orphan_images"] == []
    assert seen["path"] == "/api/system/disk"
    assert seen["idem"] is None  # a read never claims a key


@pytest.mark.asyncio
async def test_system_gc_impl_auto_mints_key_on_real_run():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": False, "images": {"removed": []}})

    await server._system_gc_impl(_client(handler), include_orphan_data=True)
    assert seen["idem"] and len(seen["idem"]) >= 32
    assert "dry_run" not in seen["query"]


@pytest.mark.asyncio
async def test_system_gc_impl_mints_no_key_on_dry_run():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": True, "images": {"removed": []}})

    await server._system_gc_impl(_client(handler), dry_run=True)
    # No key minted on a dry run — a claimed key would poison a later real gc.
    assert seen["idem"] is None
    assert seen["query"]["dry_run"] == "true"


# --- agent workspaces (P29 WP3) ---------------------------------------------


@pytest.mark.asyncio
async def test_write_app_files_impl_mints_key_and_sends_the_batch():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"written": 1, "deleted": 0})

    result = await workspaces_tools._write_app_files_impl(
        _client(handler), name="demo", files={"main.py": "print(1)\n"}
    )
    assert result == {"written": 1, "deleted": 0}
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/workspaces/demo/files"
    assert seen["idem"] and len(seen["idem"]) >= 32  # a UUID was auto-generated
    assert seen["body"] == {"files": {"main.py": "print(1)\n"}, "delete": []}


@pytest.mark.asyncio
async def test_write_app_files_impl_forwards_deletes_and_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"written": 0, "deleted": 1})

    await workspaces_tools._write_app_files_impl(
        _client(handler),
        name="demo",
        files={},
        delete=["old.py"],
        idempotency_key="fixed-ws",
    )
    assert seen["idem"] == "fixed-ws"
    assert seen["body"] == {"files": {}, "delete": ["old.py"]}


@pytest.mark.asyncio
async def test_write_app_files_impl_refuses_an_empty_batch():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        raise AssertionError("an empty batch must not reach the daemon")

    result = await workspaces_tools._write_app_files_impl(_client(handler), name="demo", files={})
    assert result["error"]["code"] == "bad_request"


@pytest.mark.asyncio
async def test_write_app_files_impl_keeps_the_cap_limit_extra():
    """The D-P29-5 caps carry their number as a TOP-LEVEL ``limit`` because
    ``_call`` drops ``detail``; an agent must be able to read the number it
    breached without parsing prose."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "workspace.file_too_large",
                "message": "File 'big.py' is 262145 bytes; the per-file limit is 262144 bytes.",
                "detail": {"path": "big.py", "size": 262145, "limit": 262144},
                "limit": 262144,
            },
        )

    result = await workspaces_tools._write_app_files_impl(
        _client(handler), name="demo", files={"big.py": "x"}
    )
    assert result["error"]["code"] == "workspace.file_too_large"
    assert result["error"]["limit"] == 262144
    # ``detail`` is dropped by _call (it may echo caller input) — the extra is
    # the only surviving carrier, which is why the route sends both.
    assert "detail" not in result["error"]


@pytest.mark.asyncio
async def test_list_app_files_impl_is_a_plain_read():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"name": "demo", "files": [], "file_count": 0})

    result = await workspaces_tools._list_app_files_impl(_client(handler), name="demo")
    assert result["file_count"] == 0
    assert seen["method"] == "GET"
    assert seen["path"] == "/api/workspaces/demo"
    assert seen["idem"] is None  # a read never claims a key


@pytest.mark.asyncio
async def test_read_app_file_impl_returns_raw_text():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, text="print('hi')\n", headers={"content-type": "text/plain"})

    result = await workspaces_tools._read_app_file_impl(
        _client(handler), name="demo", path="src/main.py"
    )
    assert result == "print('hi')\n"  # a non-dict success passes through _call
    assert seen["path"] == "/api/workspaces/demo/files/src/main.py"


@pytest.mark.asyncio
async def test_deploy_app_impl_mints_key_and_forwards_options():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo"})

    result = await workspaces_tools._deploy_app_impl(
        _client(handler), name="demo", port=8000, env={"KEEP": "v", "DROP": None}
    )
    assert result["name"] == "demo"
    assert seen["method"] == "POST"
    assert seen["path"] == "/api/workspaces/demo/deploy"
    assert seen["idem"] and len(seen["idem"]) >= 32
    assert seen["body"] == {"port": 8000, "env": {"KEEP": "v", "DROP": None}}


@pytest.mark.asyncio
async def test_deploy_app_impl_dry_run_omits_key_and_sets_param():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": True})

    result = await workspaces_tools._deploy_app_impl(_client(handler), name="demo", dry_run=True)
    assert result == {"dry_run": True}
    assert "Idempotency-Key" not in seen["headers"]
    assert seen["query"]["dry_run"] == "true"


# --- restart_daemon (P23 WP3) ----------------------------------------------


@pytest.mark.asyncio
async def test_restart_daemon_impl_mints_key_when_omitted():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(
            202,
            json={
                "restarting": True,
                "in_flight_builds": 0,
                "in_flight_runs": 0,
                "drain_timeout_s": 60,
            },
        )

    result = await server._restart_daemon_impl(_client(handler))
    assert result["restarting"] is True
    assert seen["path"] == "/api/daemon/restart"
    # Always minted (no dry run exists here) — the daemon requires one in-route.
    assert seen["idem"] and len(seen["idem"]) >= 32
    assert seen["body"] == {"drain_timeout_s": 60}


@pytest.mark.asyncio
async def test_restart_daemon_impl_forwards_supplied_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content) if request.content else None
        return httpx.Response(202, json={"restarting": True})

    await server._restart_daemon_impl(
        _client(handler), drain_timeout_s=0, idempotency_key="caller-key"
    )
    assert seen["idem"] == "caller-key"
    assert seen["body"] == {"drain_timeout_s": 0}


@pytest.mark.asyncio
async def test_restart_daemon_impl_propagates_409_as_structured_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "daemon.restart_in_progress",
                "message": "A daemon restart is already in progress.",
                "hint": "Wait for the daemon to come back up; poll GET /capabilities.uptime_s.",
            },
        )

    result = await server._restart_daemon_impl(_client(handler))
    assert result["error"]["code"] == "daemon.restart_in_progress"
    assert result["error"]["status"] == 409
    assert "already in progress" in result["error"]["message"]


@pytest.mark.asyncio
async def test_build_server_registers_all_tools():
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = await mcp_server.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "list_gpus",
        "cluster_stats",
        "list_services",
        "get_service",
        "wait_for_service",
        "serve",
        "stop_service",
        "restart_service",
        "remove_service",
        "serve_model",
        "list_models",
        "create_database",
        "list_databases",
        "deploy",
        "set_secret",
        "list_secret_names",
        "remove_secret",
        "service_logs",
        "get_audit",
        "get_config",
        "set_config",
        "apply_config",
        "get_app_config",
        "set_app_config",
        "deploy_git",
        "list_app_templates",
        "deploy_template",
        "diagnose_service",
        "capabilities",
        "doctor",
        "proxy_status",
        "list_routes",
        "system_disk",
        "system_gc",
        "run_command",
        "restart_daemon",
        "service_stats",
        "get_events",
        "redeploy_service",
        "write_app_files",
        "list_app_files",
        "read_app_file",
        "deploy_app",
        "share_service",
        "unshare_service",
        "add_domain",
        "remove_domain",
        "dump_database",
        "list_database_dumps",
    }
    # The tool set is a public contract for external agents: pin the count so a
    # tool cannot be added or dropped without an explicit CHANGELOG decision.
    # (P26 WP-H) 43 → 45 with the hosted-share pair; (P26 WP1) 45 → 47 with the
    # custom-domain pair; (P37) 47 → 49 with the managed-database dump pair —
    # and deliberately NOT a restore tool (D-P37-12).
    assert len(names) == 49


_MCP_TOOLS_GOLDEN_PATH = Path(__file__).parent / "data" / "mcp_tools_golden.json"


# Regenerate with:
#   NERDIT_REGEN_MCP_GOLDEN=1 ./venv/bin/pytest tests/test_mcp.py \
#       -k test_build_server_tool_schema_golden
@pytest.mark.asyncio
async def test_build_server_tool_schema_golden():
    """Full {name -> (description, inputSchema)} contract, not just names.

    Byte-stable through WP24's registry collapse and the SDK-v2 migration
    (D-T-5); a param dropped or reshaped by either fails here even though the
    name set stays intact.
    """
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = await mcp_server.list_tools()
    actual = {t.name: {"description": t.description, "inputSchema": t.inputSchema} for t in tools}

    if os.environ.get("NERDIT_REGEN_MCP_GOLDEN"):
        _MCP_TOOLS_GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.fail("regenerated the golden; re-run without NERDIT_REGEN_MCP_GOLDEN")

    golden = json.loads(_MCP_TOOLS_GOLDEN_PATH.read_text())

    assert set(actual) == set(golden), (
        f"tool set drifted: added={set(actual) - set(golden)}, dropped={set(golden) - set(actual)}"
    )

    def _norm(description: str | None) -> str:
        # Python 3.13's compiler dedents docstrings at compile time
        # (gh-81283), so the raw description text differs by interpreter
        # while the wording is identical; compare cleandoc-normalized.
        return inspect.cleandoc(description) if description else ""

    for name in golden:
        assert _norm(actual[name]["description"]) == _norm(golden[name]["description"]), (
            f"{name}: description drifted"
        )
        assert actual[name]["inputSchema"] == golden[name]["inputSchema"], (
            f"{name}: inputSchema drifted"
        )


# --- Agent-DX: the [deploy]-schema drift guard on the deploy tool descriptions -
#
# For an agent the tool description IS the documentation, so the ``[deploy]``
# key list in these three docstrings is a real contract. This is the guard that
# makes the sync-obligation comments in ``mcp/tools/deploy.py`` and
# ``daemon/deploy_pipeline.py`` enforceable rather than aspirational.

_DEPLOY_DOC_TOOLS = ("deploy", "deploy_git", "deploy_app")


async def _deploy_tool_descriptions() -> dict[str, str]:
    pytest.importorskip("mcp")
    mcp_server = server.build_server()
    tools = await mcp_server.list_tools()
    return {t.name: (t.description or "") for t in tools if t.name in _DEPLOY_DOC_TOOLS}


@pytest.mark.asyncio
async def test_deploy_tool_descriptions_list_every_deploy_config_field():
    """Each of the three deploy tools names EXACTLY ``DeployConfig.model_fields``.

    Both directions matter: a field added to the schema and not to the
    docstrings leaves agents blind to it, and a key named in a docstring that
    the schema does not declare would advertise a key the daemon IGNORES.
    """
    from nerdit.config.project import DeployConfig

    expected = set(DeployConfig.model_fields)
    descriptions = await _deploy_tool_descriptions()
    assert set(descriptions) == set(_DEPLOY_DOC_TOOLS)

    for name, desc in descriptions.items():
        # Bound the scan to the key-list sentence itself, so surrounding prose
        # (``summary``/``hints``/``path``) cannot be mistaken for a schema key.
        assert "``[deploy]`` keys" in desc, name
        section = (
            desc.split("in the response ``hints``:")[1].split("``build_settings``.")[0]
            + "``build_settings``"
        )
        listed = set(re.findall(r"``([a-z_]+)``", section))
        assert expected <= listed, f"{name} omits {sorted(expected - listed)}"
        # No key is claimed that DeployConfig does not declare (`build` is the
        # one the field agent guessed; it must appear only as an absence).
        claimed = listed - {"http", "tcp"}  # the health_type value vocabulary
        assert claimed <= expected, f"{name} invents {sorted(claimed - expected)}"


@pytest.mark.asyncio
async def test_deploy_tool_descriptions_document_nested_build_settings():
    """Build overrides live in the declared nested settings table."""
    for name, desc in (await _deploy_tool_descriptions()).items():
        assert "``[deploy.build_settings]``" in desc, name
        assert "``build: false``" in desc, name
        assert "Dockerfile" in desc, name
        assert "``hints``" in desc and "``summary``" in desc, name


@pytest.mark.asyncio
async def test_deploy_tool_descriptions_scope_summary_and_hints_to_real_deploys():
    """(Codex 3804646834) The result-shape promise must name its own exceptions.

    ``dry_run=True`` returns ``build_plan_body(...)``, which carries
    ``warnings`` and neither ``summary`` nor ``hints``; ``deploy(rollback=True)``
    short-circuits to a different endpoint whose body has neither either. Only
    the ``_finalize_deploy`` tail attaches them. An unconditional promise made
    right after documenting those two modes invites an agent to dereference an
    absent field — or, worse, to miss the ignored-key advisory that a dry run
    DOES carry, under a different name.
    """
    for name, desc in (await _deploy_tool_descriptions()).items():
        assert "successful non-dry-run deploy" in desc, name
        # The plan's advisory channel is named, so a dry-run caller is not left
        # looking for a field that was never going to be there.
        assert "``warnings``" in desc, name
        assert "``dry_run`` plan" in desc, name
    # ``deploy`` is the only one of the three with a rollback mode; the other
    # two must not advertise a parameter they do not accept.
    descriptions = await _deploy_tool_descriptions()
    assert "``rollback`` response" in descriptions["deploy"]
    for name in ("deploy_git", "deploy_app"):
        assert "``rollback`` response" not in descriptions[name], name


@pytest.mark.asyncio
async def test_deploy_tool_descriptions_name_both_proxy_modes():
    """(P26 WP0) The base-path advisory is only half true on its own.

    Path mode's ``/<name>/`` prefix is the reason a frontend build needs a base
    path — but subdomain mode has no prefix and needs none, and the daemon says
    which mode it is in. Naming only the path-mode half taught agents to set a
    base path unconditionally, which breaks the subdomain-mode build.
    """
    desc = (await _deploy_tool_descriptions())["deploy"]
    assert "``subdomain`` mode" in desc
    assert "``capabilities.proxy.mode``" in desc


# --- (P33) the shared sandbox sentence on every content-bearing deploy tool ---

_SANDBOX_DOC_TOOLS = ("deploy", "deploy_git", "deploy_app", "write_app_files")


@pytest.mark.asyncio
async def test_deploy_tool_descriptions_carry_the_shared_sandbox_note():
    """The field failure, 2026-08-23: nothing told the agent the constraint.

    An agent deployed a stock ``FROM nginx`` static site; the root entrypoint
    died on ``chown(...) Operation not permitted`` under ``cap_drop=["ALL"]``.
    A tool description IS an agent's documentation, so the sentence has to be
    THERE — and it is ONE module constant, not four copies, so the four cannot
    drift apart.
    """
    pytest.importorskip("mcp")
    from nerdit.mcp.tools._shared import SANDBOX_NOTE, SANDBOX_NOTE_PLACEHOLDER

    mcp_server = server.build_server()
    tools = {t.name: (t.description or "") for t in await mcp_server.list_tools()}

    # Whitespace-insensitive: the note is word-wrapped into the docstring.
    words = " ".join(SANDBOX_NOTE.split())
    for name in _SANDBOX_DOC_TOOLS:
        desc = " ".join(tools[name].split())
        assert words in desc, f"{name}: the shared sandbox note is missing"
        assert SANDBOX_NOTE_PLACEHOLDER not in tools[name], (
            f"{name}: the placeholder was never interpolated"
        )


@pytest.mark.asyncio
async def test_sandbox_note_names_the_constraint_the_fix_and_the_signal():
    """The three things an agent needs, pinned so a reword cannot drop one."""
    from nerdit.mcp.tools._shared import SANDBOX_NOTE

    assert 'cap_drop=["ALL"]' in SANDBOX_NOTE  # the constraint
    assert "no-new-privileges" in SANDBOX_NOTE
    assert "nginx-unprivileged" in SANDBOX_NOTE  # the fix, with the port
    assert "8080" in SANDBOX_NOTE
    assert "capabilities.sandbox" in SANDBOX_NOTE  # where to read it live
    assert "image_needs_privileges" in SANDBOX_NOTE  # what a crash will say


@pytest.mark.asyncio
async def test_share_service_impl_mints_a_key_and_sends_the_contract_body():
    """(P26 WP-H, PR review) The wire shape of the share tool, not its prose.

    "MCP is a thin projection" (Invariant #3) is a claim about bytes: this pins
    the method, the path, the exact JSON body and an auto-minted key, so a
    refactor that starts pre-checking ``access`` locally — or drops ``consent``
    on the way through — fails here rather than in a customer's transcript.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access": "public", "url": None, "state": "link_down"})

    result = await exposure_tools._share_service_impl(
        _client(handler), name="demo", access="public", consent=True
    )

    assert result == {"access": "public", "url": None, "state": "link_down"}
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/services/demo/share"
    assert seen["body"] == {"access": "public", "consent": True}
    uuid.UUID(seen["idem"])  # a real uuid4 was minted, not a placeholder


@pytest.mark.asyncio
async def test_share_service_impl_defaults_to_private_and_forwards_an_explicit_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"access": "private"})

    await exposure_tools._share_service_impl(
        _client(handler), name="demo", idempotency_key="fixed-share"
    )

    assert seen["idem"] == "fixed-share"
    assert seen["body"] == {"access": "private", "consent": False}


@pytest.mark.asyncio
async def test_unshare_service_impl_deletes_with_a_minted_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service_name": "demo", "removed": True})

    result = await exposure_tools._unshare_service_impl(_client(handler), name="demo")

    assert result == {"service_name": "demo", "removed": True}
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/services/demo/share"
    uuid.UUID(seen["idem"])


@pytest.mark.asyncio
async def test_share_service_impl_surfaces_the_daemon_refusal_unchanged():
    """The refusals are the daemon's to make (D-P26-H3): nothing is pre-checked
    here, so the envelope an agent branches on must arrive intact."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "share.unprotected",
                "message": "Service 'demo' would be reachable by anyone with the URL.",
                "hint": "Public means public: add [deploy].edge_auth, or pass consent=true.",
            },
        )

    result = await exposure_tools._share_service_impl(
        _client(handler), name="demo", access="public"
    )

    assert result["error"]["code"] == "share.unprotected"
    assert "consent=true" in result["error"]["hint"]


@pytest.mark.asyncio
async def test_share_tools_describe_the_contract():
    """(P26 WP-H) The share tool's description IS its documentation.

    An agent that calls ``share_service(access='public')`` must be able to
    branch on the two refusals without a round trip, and must know where the
    resulting URL shows up afterwards — the tool returns one URL, but every
    later read carries it under ``public_urls``.
    """
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = {t.name: t for t in await mcp_server.list_tools()}

    share = tools["share_service"].description or ""
    assert "share.not_entitled" in share
    assert "share.unprotected" in share
    assert "share.link_required" in share
    assert "public_urls" in share
    # The state vocabulary is a machine contract, not prose.
    for token in ("ready", "link_down", "not_entitled"):
        assert token in share

    unshare = tools["unshare_service"].description or ""
    assert "Idempotent" in unshare


@pytest.mark.asyncio
async def test_add_domain_impl_mints_a_key_and_sends_the_contract_body():
    """(P26 WP1) The wire shape of the custom-domain tool, not its prose.

    "MCP is a thin projection" (Invariant #3) is a claim about bytes: this pins
    the method, the domain-as-path-segment, the exact JSON body and an
    auto-minted key, so a refactor that starts validating the name locally — or
    drops ``acme`` on the way through — fails here rather than in a customer's
    transcript.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "domain": "app.example.com",
                "url": "https://app.example.com/",
                "state": "ready",
                "created": True,
            },
        )

    result = await exposure_tools._add_domain_impl(
        _client(handler), name="demo", domain="app.example.com"
    )

    assert result["created"] is True
    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/services/demo/domains/app.example.com"
    # (review round 1) An omitted ``acme`` travels as an EMPTY body, not as
    # ``false``: the field is tri-state daemon-side, so re-running this tool to
    # read a URL back must not downgrade an issued public certificate.
    assert seen["body"] == {}
    uuid.UUID(seen["idem"])  # a real uuid4 was minted, not a placeholder


@pytest.mark.asyncio
async def test_add_domain_impl_forwards_acme_and_an_explicit_key():
    """``acme=True`` is not pre-refused here — WP1's 409 is the daemon's to make."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"domain": "app.example.com"})

    await exposure_tools._add_domain_impl(
        _client(handler),
        name="demo",
        domain="app.example.com",
        acme=True,
        idempotency_key="fixed-domain",
    )

    assert seen["idem"] == "fixed-domain"
    assert seen["body"] == {"acme": True}


@pytest.mark.asyncio
async def test_remove_domain_impl_deletes_with_a_minted_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(
            200, json={"service_name": "demo", "domain": "app.example.com", "removed": True}
        )

    result = await exposure_tools._remove_domain_impl(
        _client(handler), name="demo", domain="app.example.com"
    )

    assert result["removed"] is True
    assert seen["method"] == "DELETE"
    assert seen["path"] == "/api/services/demo/domains/app.example.com"
    uuid.UUID(seen["idem"])


@pytest.mark.asyncio
async def test_add_domain_impl_surfaces_the_daemon_refusal_unchanged():
    """The refusals are the daemon's to make (D-P26-3): nothing is pre-checked
    here, so the envelope an agent branches on must arrive intact.

    ``detail`` is deliberately NOT one of the surviving keys — ``_call`` drops
    it wholesale so a pydantic 422 can never echo a caller's submitted values
    into a transcript. The route's machine ``reason`` token therefore rides the
    ``message``/``hint``, not a field, which is why the tool description names
    the vocabulary instead of promising ``detail.reason``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "code": "domain.invalid",
                "message": "Wildcard domains are not supported (reason: wildcard).",
                "hint": "Use a bare, lowercase DNS name you control.",
                "detail": {"reason": "wildcard"},
            },
        )

    result = await exposure_tools._add_domain_impl(
        _client(handler), name="demo", domain="*.example.com"
    )

    assert result["error"]["code"] == "domain.invalid"
    assert result["error"]["status"] == 422
    assert "wildcard" in result["error"]["message"]
    assert "bare, lowercase DNS name" in result["error"]["hint"]
    assert "detail" not in result["error"]


@pytest.mark.asyncio
async def test_domain_tools_describe_the_contract():
    """(P26 WP1) The tool description IS the agent's documentation.

    An agent must be able to branch on every refusal without a round trip, know
    that DNS and certificate trust are prerequisites it cannot satisfy itself,
    and know where the resulting URL shows up afterwards.
    """
    pytest.importorskip("mcp")

    mcp_server = server.build_server()
    tools = {t.name: t for t in await mcp_server.list_tools()}

    add = tools["add_domain"].description or ""
    for code in (
        "domain.invalid",
        "domain.taken",
        "domain.kind_unsupported",
        "domain.acme_disabled",
    ):
        assert code in add, code
    # The reason vocabulary and the state vocabulary are machine contracts an
    # agent branches on; the reason travels in the message (``_call`` drops
    # ``detail``), so the description has to carry the list.
    for token in ("wildcard", "ip_literal", "single_label", "reserved"):
        assert token in add, token
    for token in ("ready", "withheld"):
        assert token in add, token
    assert "DNS" in add
    assert "nerdit trust" in add
    assert "public_urls" in add

    remove = tools["remove_domain"].description or ""
    assert "Idempotent" in remove


# --- P34: the deploy feedback loop in the tool layer ---------------------------
#
# Three fixes from one field report: the deploy tools must SAY the deploy is
# asynchronous and name the follow-up call, the port contract must be stated
# where an agent reads it before writing a Dockerfile, and ``service_logs`` must
# be able to exclude the build's own output.


@pytest.mark.asyncio
async def test_service_logs_impl_forwards_the_source_filter():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    await server._service_logs_impl(_client(handler), "web", source="runtime")
    assert seen["query"]["source"] == "runtime"


@pytest.mark.asyncio
async def test_service_logs_impl_sends_no_source_param_by_default():
    """An unfiltered read stays byte-identical on the wire — a pre-P34 daemon
    that would 422 on an unknown query param is never handed one."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json=[])

    await server._service_logs_impl(_client(handler), "web")
    assert "source" not in seen["query"]


def _descriptions() -> dict[str, str]:
    """Every tool's rendered description, cleandoc-normalized like the golden."""
    from nerdit.mcp.tools import ALL_TOOLS

    return {fn.__name__: inspect.cleandoc(fn.__doc__ or "") for fn in ALL_TOOLS}


#: The deploy feedback-loop tools — the ones an agent walks in order when a
#: deploy goes wrong. Only these carry the "Use when:" opener (F): a blanket
#: rewrite of all 47 would be churn, and these eleven are the loop.
_FEEDBACK_LOOP_TOOLS = (
    "deploy",
    "deploy_git",
    "deploy_app",
    "deploy_template",
    "redeploy_service",
    "wait_for_service",
    "diagnose_service",
    "service_logs",
    "share_service",
    "unshare_service",
    "get_events",
)

#: The subset that STARTS a build. For an agent these are the ones whose 201
#: means "accepted", not "running" — so each must say so on its first line.
_ASYNC_DEPLOY_TOOLS = (
    "deploy",
    "deploy_git",
    "deploy_app",
    "deploy_template",
    "redeploy_service",
)


@pytest.mark.parametrize("name", _FEEDBACK_LOOP_TOOLS)
def test_feedback_loop_tools_open_with_a_use_when_sentence(name):
    """For an agent the description IS the documentation, and a dense first
    paragraph is a description no one reads to the end of."""
    first = _descriptions()[name].splitlines()[0]
    assert first.startswith("Use when: "), first
    # One sentence, on one line — a client that renders only the first line of
    # a tool description must still get a complete thought.
    assert first.rstrip().endswith("."), first


@pytest.mark.parametrize("name", _ASYNC_DEPLOY_TOOLS)
def test_deploy_tools_name_wait_for_service_on_their_first_line(name):
    """The exact field failure: the agent read ``status: building`` as success."""
    first = _descriptions()[name].splitlines()[0]
    assert "Asynchronous" in first, first
    assert "wait_for_service" in first, first


@pytest.mark.parametrize("name", _ASYNC_DEPLOY_TOOLS)
def test_deploy_tools_document_the_next_step_field(name):
    """The structured channel must be discoverable from the description too."""
    assert "next_step" in _descriptions()[name], name


def test_wait_for_service_documents_the_inline_diagnosis():
    """Otherwise the fold-in is invisible and the agent still makes two calls."""
    doc = _descriptions()["wait_for_service"]
    assert "diagnosis" in doc
    assert "remediation" in doc


def test_service_logs_documents_the_source_filter_and_the_literal_grep():
    doc = _descriptions()["service_logs"]
    assert "source" in doc
    assert "'runtime'" in doc
    # The reported symptom, named so an agent recognises it.
    assert "404B" in doc
    assert "literal substring" in doc


def test_share_service_separates_the_link_state_from_the_origin():
    """'ready' is overloaded — the description must say what it does NOT mean."""
    doc = _descriptions()["share_service"]
    assert "origin" in doc
    assert "answers" in doc
    assert "share.not_shared" in doc


def test_the_sandbox_note_states_the_port_contract():
    """(C) The agent guessed 80 and expected a privileged-bind warning. The
    honest answer for THIS runtime is that the port is never the problem."""
    from nerdit.mcp.tools._shared import SANDBOX_NOTE

    assert "ip_unprivileged_port_start=0" in SANDBOX_NOTE
    # Stated unconditionally because every container is launched with a forced
    # bridge network (``core/launch.py``), never host networking — see the
    # companion test in ``tests/test_containers.py``.
    assert "docker bridge" in SANDBOX_NOTE
    assert "8000" in SANDBOX_NOTE


@pytest.mark.parametrize("name", ["deploy", "deploy_git", "deploy_app", "write_app_files"])
def test_the_port_contract_reaches_every_sandbox_note_carrier(name):
    # Whitespace-insensitive: ``_wrap`` breaks the note into 76-column lines and
    # the break point moves whenever the note is reworded, so the marker can land
    # astride a newline. The contract is that the phrase is CARRIED, not where
    # the greedy wrap happens to split it.
    assert "PORT CONTRACT" in " ".join(_descriptions()[name].split()), name


def test_sandbox_note_wrap_preserves_whitespace_and_whole_words():
    from nerdit.mcp.tools._shared import _wrap

    assert _wrap(" \t\n") == []
    assert _wrap("  a\t b\n c\u2003d  ", width=3) == ["a b", "c d"]
    assert _wrap("a long-hyphenated-word z", width=4) == ["a", "long-hyphenated-word", "z"]
