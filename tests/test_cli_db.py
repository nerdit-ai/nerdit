"""Unit tests for the P15/P37 databases CLI: client methods, commands, renders.

Client methods are driven through ``httpx.MockTransport`` (the ``test_cli_models``
pattern) so the request path, JSON body, query params and ``Idempotency-Key``
header are asserted without a daemon; the table render is a capsys smoke. The
create response carries key names only — the minted password never rides it, so
these tests never assert on a credential value (there is none to assert).

The P37 rows at the bottom cover ``nerdit db dump|dumps|restore`` (§1.6) through
``CliRunner`` with the client stubbed (the ``test_cli_backup.py`` pattern):
confirm gating (a y/n answer must NOT restore), the custody + retention lines,
the ``restore.in_use`` dependents render, and the 0/1 exit codes.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.db import db_app
from nerdit.cli.display import display_database_table

_runner = CliRunner()


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


def _status_error(status: int, body: dict) -> httpx.HTTPStatusError:
    """Build the ``HTTPStatusError`` a structured daemon refusal raises.

    The command layer reads the ENVELOPE off the response (the dependents list
    is not on the exception), so a stub that raises a bare exception would not
    exercise the rendering path at all.
    """
    request = httpx.Request("POST", "http://localhost:9321/api/databases/pg/restore")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


# ---- client method header / payload behavior ----


@pytest.mark.asyncio
async def test_create_database_sends_body_and_idempotency_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            201, json={"id": "db-1", "name": "pg", "backend": "postgres", "status": "building"}
        )

    client = _make_client(handler)
    out = await client.create_database("postgres", idempotency_key="key-123")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/databases"
    assert seen["idem"] == "key-123"
    assert seen["body"] == {"backend": "postgres"}
    assert out["name"] == "pg"


@pytest.mark.asyncio
async def test_create_database_sends_name_override_and_omits_key_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "db-1"})

    client = _make_client(handler)
    await client.create_database("redis", name="cache")

    assert seen["idem"] is None
    assert seen["body"] == {"backend": "redis", "name": "cache"}


@pytest.mark.asyncio
async def test_create_database_omits_backend_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "db-1"})

    await _make_client(handler).create_database()
    assert seen["body"] == {}


@pytest.mark.asyncio
async def test_list_databases_hits_api_prefix_with_params():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    client = _make_client(handler)
    page = await client.list_databases(limit=10, cursor="abc")

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/databases"
    assert seen["params"] == {"limit": "10", "cursor": "abc"}
    assert page == {"items": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_list_databases_omits_cursor_when_absent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    await _make_client(handler).list_databases()
    assert seen["params"] == {"limit": "50"}


# ---- table render ----


def test_display_database_table_renders_fields(capsys, monkeypatch):
    # Wide virtual terminal so Rich never wraps the endpoint mid-assertion.
    monkeypatch.setenv("COLUMNS", "200")
    databases = [
        {
            "name": "pg",
            "backend": "postgres",
            "status": "running",
            "db_ready": True,
            "endpoint": "172.17.0.1:5432",
        },
        {
            "name": "cache",
            "backend": "redis",
            "status": "building",
            "db_ready": False,
            "endpoint": None,
        },
    ]
    display_database_table(databases)
    out = capsys.readouterr().out
    assert "postgres" in out
    assert "running" in out
    assert "yes" in out
    assert "5432" in out
    assert "redis" in out
    assert "building" in out
    assert "no" in out


def test_display_database_table_renders_unmapped_status(capsys, monkeypatch):
    # Mirror of the P5-runbook MarkupError regression: a status outside the
    # style map must render bare, never "[]completed[/]" (which crashes Rich).
    monkeypatch.setenv("COLUMNS", "200")
    display_database_table(
        [
            {
                "name": "pg-old",
                "backend": "postgres",
                "status": "completed",
                "db_ready": True,
                "endpoint": None,
            }
        ]
    )
    assert "completed" in capsys.readouterr().out


# ---- command wiring ----


@pytest.mark.asyncio
async def test_db_list_command_renders_page(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import db as db_mod

    class _C:
        async def list_databases(self, limit: int = 50, cursor: str | None = None):
            return {
                "items": [
                    {
                        "name": "pg",
                        "backend": "postgres",
                        "status": "running",
                        "db_ready": True,
                        "endpoint": "172.17.0.1:5432",
                    }
                ],
                "next_cursor": None,
            }

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    await db_mod._list_async()
    out = capsys.readouterr().out
    assert "postgres" in out


@pytest.mark.asyncio
async def test_db_list_command_empty_hint(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import db as db_mod

    class _C:
        async def list_databases(self, limit: int = 50, cursor: str | None = None):
            return {"items": [], "next_cursor": None}

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    await db_mod._list_async()
    out = capsys.readouterr().out
    assert "No databases" in out


@pytest.mark.asyncio
async def test_db_create_command_prints_binding_stanza(monkeypatch, capsys):
    import nerdit.cli.client as client_mod
    from nerdit.cli.commands import db as db_mod

    seen: dict = {}

    class _C:
        async def create_database(self, backend=None, name=None, idempotency_key=None):
            seen["backend"] = backend
            seen["idem"] = idempotency_key
            return {"id": "db-1", "name": "pg", "backend": "postgres", "status": "building"}

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _C())
    # wait=False so no /wait call is attempted.
    await db_mod._create_async("postgres", None, False, 120)
    out = capsys.readouterr().out
    # The post-create print includes the [db.default] stanza with the REQUIRED
    # database= line (owner call: no implicit default) — never a password.
    assert "[db.default]" in out
    assert 'provider = "managed"' in out
    assert 'database = "pg"' in out
    assert "password" not in out.lower()
    # A UUID idempotency key is minted per invocation.
    assert seen["idem"] and len(seen["idem"]) >= 32
    assert seen["backend"] == "postgres"


def test_db_app_registered():
    from nerdit.cli.app import app

    names = {t.name for t in app.registered_groups}
    assert "db" in names


# --------------------------------------------------------------------------- #
# P37 §1.6 — dump / dumps / restore
#
# Two layers, like the P15 rows above: the client methods through
# ``httpx.MockTransport`` (path, body, query, Idempotency-Key, httpx budget),
# and the commands through ``CliRunner`` with the client stubbed (confirm
# gating, rendered lines, exit codes). No assertion anywhere reaches for a
# credential — none rides any of these three surfaces, in either direction.
# --------------------------------------------------------------------------- #

_DUMP = "nerdit-dump-pg-20260907T101500Z-abc123.tar.gz"


@pytest.mark.asyncio
async def test_create_database_dump_posts_timeout_key_and_unbounded_read():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        seen["httpx_timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"service_name": "pg", "dump": _DUMP})

    client = _make_client(handler)
    out = await client.create_database_dump("pg", timeout_s=120, idempotency_key="idem-dump")

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/databases/pg/dump"
    assert seen["idem"] == "idem-dump"
    assert seen["body"] == {"timeout_s": 120}
    # The request is parked for the whole capture and the CLI sets NO read
    # deadline: ``timeout_s`` bounds the dump tool, but the hash + gzip pack
    # after it exits is proportional to the dataset and outside that budget, so
    # any constant head-room would only pick the database size at which the CLI
    # starts reporting a transport failure for a dump the daemon completes and
    # publishes. Connect/write/pool stay bounded, so a dead host still fails
    # fast. Amends D-P37-8's ``timeout_s + 60``.
    assert seen["httpx_timeout"]["read"] is None
    assert seen["httpx_timeout"]["connect"] == 10.0
    assert seen["httpx_timeout"]["write"] == 10.0
    assert seen["httpx_timeout"]["pool"] == 10.0
    assert out["dump"] == _DUMP


@pytest.mark.asyncio
async def test_list_database_dumps_is_a_plain_get():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"service_name": "pg", "dumps": []})

    out = await _make_client(handler).list_database_dumps("pg")

    assert seen["method"] == "GET"
    assert seen["path"] == "/api/databases/pg/dumps"
    assert seen["idem"] is None
    assert out["dumps"] == []


@pytest.mark.asyncio
async def test_restore_database_dump_sends_basename_force_and_unbounded_read():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = json.loads(request.content)
        seen["httpx_timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"service_name": "pg", "dump": _DUMP})

    client = _make_client(handler)
    await client.restore_database_dump(
        "pg", _DUMP, timeout_s=60, force=True, idempotency_key="idem-restore"
    )

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/databases/pg/restore"
    assert seen["query"] == {"force": "true"}
    assert seen["idem"] == "idem-restore"
    assert seen["body"] == {"dump": _DUMP, "timeout_s": 60}
    # Same shape as the dump leg: the extract-and-verify phase sits outside
    # ``timeout_s`` and scales with the archive, so the server owns the bound.
    assert seen["httpx_timeout"]["read"] is None
    assert seen["httpx_timeout"]["connect"] == 10.0


@pytest.mark.asyncio
async def test_restore_database_dump_omits_force_by_default():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        return httpx.Response(200, json={})

    await _make_client(handler).restore_database_dump("pg", _DUMP)
    assert seen["query"] == {}


# ---- command wiring: dump ----


def _dump_stub(**kw) -> SimpleNamespace:
    """A stub client whose dump succeeds and whose retention read answers 5."""
    result = {
        "service_name": "pg",
        "dump": _DUMP,
        "size_bytes": 2048,
        "sha256": "0" * 64,
        "engine": "postgres",
        "format": "pg_custom",
        "created_at": "2026-09-07T10:15:00Z",
        "duration_s": 1.5,
    }
    result.update(kw)
    return SimpleNamespace(
        create_database_dump=AsyncMock(return_value=result),
        get_config=AsyncMock(
            return_value={"section": "retention", "values": {"dump_keep_last": 5}}
        ),
    )


def test_db_dump_renders_custody_retention_and_cross_reference(monkeypatch):
    stub = _dump_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes"])

    assert res.exit_code == 0
    assert _DUMP in res.output
    assert "2.0 KiB" in res.output
    assert "postgres" in res.output and "pg_custom" in res.output
    # Custody: the archive holds the data, and it stays on the box.
    assert "every row of the database" in res.output
    # Retention (D-P37-7): the live keep-last, named with its config key.
    assert "keeps the last 5 dumps" in res.output
    assert "dump_keep_last" in res.output
    # The cross-reference at the physical flavour (each verb points at the other).
    assert "nerdit backup --volume pg" in res.output
    # A password is not on this surface in either direction.
    assert "password" not in res.output.lower()
    # One minted Idempotency-Key per invocation, and the requested budget.
    stub.create_database_dump.assert_awaited_once()
    kwargs = stub.create_database_dump.await_args.kwargs
    assert kwargs["idempotency_key"]
    assert kwargs["timeout_s"] == 900


def test_db_dump_passes_timeout_s(monkeypatch):
    stub = _dump_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes", "--timeout-s", "60"])
    assert res.exit_code == 0
    assert stub.create_database_dump.await_args.kwargs["timeout_s"] == 60


def test_db_dump_confirm_declined_makes_no_call(monkeypatch):
    stub = _dump_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg"], input="n\n")
    assert res.exit_code == 0
    assert "Aborted" in res.output
    stub.create_database_dump.assert_not_awaited()


def test_db_dump_confirm_accepted_runs(monkeypatch):
    stub = _dump_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg"], input="y\n")
    assert res.exit_code == 0
    stub.create_database_dump.assert_awaited_once()


def test_db_dump_keep_zero_prints_the_accumulate_forever_line(monkeypatch):
    stub = _dump_stub()
    stub.get_config = AsyncMock(
        return_value={"section": "retention", "values": {"dump_keep_last": 0}}
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes"])
    assert res.exit_code == 0
    assert "No dumps are ever deleted" in res.output
    assert "dump_keep_last" in res.output


def test_db_dump_retention_read_failure_falls_back_to_the_packaged_default(monkeypatch):
    """A dump that succeeded is never reported as a failure because the
    follow-up retention read did not answer — the line degrades to the packaged
    default and SAYS that it did."""
    stub = _dump_stub()
    stub.get_config = AsyncMock(side_effect=RuntimeError("config read failed"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes"])
    assert res.exit_code == 0
    assert "keeps the last 5 dumps" in res.output
    assert "could not be read" in res.output


def test_db_dump_client_error_exits_1(monkeypatch):
    stub = _dump_stub()
    stub.create_database_dump = AsyncMock(side_effect=RuntimeError("daemon down"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes"])
    assert res.exit_code == 1


def test_db_dump_renders_a_structured_refusal(monkeypatch):
    """A 409 envelope reaches the operator as its message + hint, not as a
    traceback — the ``dump.in_progress`` shape the single-flight guard emits."""
    stub = _dump_stub()
    stub.create_database_dump = AsyncMock(
        side_effect=_status_error(
            409,
            {
                "code": "dump.in_progress",
                "message": "A dump or restore is already running for 'pg'.",
                "hint": "Wait for it to finish, then retry.",
            },
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dump", "pg", "--yes"])
    assert res.exit_code == 1
    assert "already running" in res.output
    assert "Wait for it to finish" in res.output


# ---- command wiring: dumps ----


def test_db_dumps_renders_table(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    stub = SimpleNamespace(
        list_database_dumps=AsyncMock(
            return_value={
                "service_name": "pg",
                "dumps": [
                    {
                        "dump": _DUMP,
                        "size_bytes": 4096,
                        "created_at": "2026-09-07T10:15:00Z",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dumps", "pg"])
    assert res.exit_code == 0
    assert _DUMP in res.output
    assert "4.0 KiB" in res.output
    assert "2026-09-07" in res.output
    stub.list_database_dumps.assert_awaited_once_with("pg")


def test_db_dumps_empty_hint(monkeypatch):
    stub = SimpleNamespace(
        list_database_dumps=AsyncMock(return_value={"service_name": "pg", "dumps": []})
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dumps", "pg"])
    assert res.exit_code == 0
    assert "No dumps" in res.output


def test_db_dumps_client_error_exits_1(monkeypatch):
    stub = SimpleNamespace(list_database_dumps=AsyncMock(side_effect=RuntimeError("down")))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["dumps", "pg"])
    assert res.exit_code == 1


# ---- command wiring: restore ----


def _restore_stub(**kw) -> SimpleNamespace:
    result = {
        "service_name": "pg",
        "dump": _DUMP,
        "engine": "postgres",
        "duration_s": 3.25,
    }
    result.update(kw)
    return SimpleNamespace(restore_database_dump=AsyncMock(return_value=result))


def test_db_restore_requires_the_typed_word(monkeypatch):
    """A y/n muscle-memory answer must NOT restore: the prompt asks for the
    word, and anything else aborts without a call (the uninstall precedent)."""
    stub = _restore_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP], input="y\n")
    assert res.exit_code == 0
    assert "Aborted" in res.output
    stub.restore_database_dump.assert_not_awaited()


def test_db_restore_typed_word_proceeds_and_states_the_redis_bounce(monkeypatch):
    stub = _restore_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP], input="restore\n")
    assert res.exit_code == 0
    # The prompt names the destructive scope and the Redis stop/start fact —
    # the one consequence an operator can learn nowhere else at this moment.
    assert "replaces the contents of 'pg'" in res.output
    assert "STOPS and restarts" in res.output
    assert "Restored:" in res.output
    stub.restore_database_dump.assert_awaited_once()
    args, kwargs = stub.restore_database_dump.await_args
    assert args == ("pg", _DUMP)
    assert kwargs["force"] is False
    assert kwargs["timeout_s"] == 600
    assert kwargs["idempotency_key"]


def test_db_restore_yes_skips_the_prompt_and_forwards_force_and_timeout(monkeypatch):
    stub = _restore_stub()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP, "--yes", "--force", "--timeout-s", "90"])
    assert res.exit_code == 0
    assert "Type 'restore'" not in res.output
    kwargs = stub.restore_database_dump.await_args.kwargs
    assert kwargs["force"] is True
    assert kwargs["timeout_s"] == 90


def test_db_restore_in_use_renders_dependents_and_the_force_hint(monkeypatch):
    """The 409 an operator is most likely to hit names the apps standing in the
    way in a LIST, which the generic renderer drops — so the command prints
    them itself before the message and the --force hint."""
    stub = _restore_stub()
    stub.restore_database_dump = AsyncMock(
        side_effect=_status_error(
            409,
            {
                "code": "restore.in_use",
                "message": "Database 'pg' is in use by 2 running service(s).",
                "hint": "Stop the bound app(s) first, or pass --force — a live client "
                "holds the locks a Postgres restore needs.",
                "detail": {"dependents": [{"service": "api", "id": "j1", "binding": "default"}]},
                "dependents": [
                    {"service": "api", "id": "j1", "binding": "default"},
                    {"service": "worker", "id": "j2", "binding": "cache"},
                ],
            },
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP, "--yes"])

    assert res.exit_code == 1
    assert "api" in res.output
    assert "worker" in res.output
    assert "binding default" in res.output
    assert "--force" in res.output
    assert "in use by 2 running service(s)" in res.output


def test_db_restore_in_use_falls_back_to_the_detail_copy(monkeypatch):
    """Either copy of the list is authoritative; neither is guaranteed by the
    envelope contract alone, so the detail one is read when the top level is
    absent."""
    stub = _restore_stub()
    stub.restore_database_dump = AsyncMock(
        side_effect=_status_error(
            409,
            {
                "code": "restore.in_use",
                "message": "Database 'pg' is in use by 1 running service(s).",
                "detail": {"dependents": [{"service": "api", "id": "j1", "binding": "default"}]},
            },
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP, "--yes"])
    assert res.exit_code == 1
    assert "api" in res.output


def test_db_restore_other_error_prints_no_dependents_block(monkeypatch):
    stub = _restore_stub()
    stub.restore_database_dump = AsyncMock(
        side_effect=_status_error(
            422,
            {
                "code": "restore.engine_mismatch",
                "message": "This dump was taken from a redis database.",
            },
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP, "--yes"])
    assert res.exit_code == 1
    assert "running services bound to" not in res.output
    assert "redis database" in res.output


def test_db_restore_transport_error_exits_1(monkeypatch):
    stub = _restore_stub()
    stub.restore_database_dump = AsyncMock(side_effect=RuntimeError("daemon down"))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = _runner.invoke(db_app, ["restore", "pg", _DUMP, "--yes"])
    assert res.exit_code == 1
