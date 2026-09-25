"""Route-level tests for P40c variables (``/api/projects/{project}/variables``).

The P40b harness shape — a real in-memory database under the real routers and
the real auth, audit and idempotency middleware — plus a capturing event bus,
because the contract here is mostly negative: a value, plain or secret, rides
no audit row, event frame, idempotency record, log line or error body, and the
only body that ever carries one is ``list_variables`` answering the project's
owner or an admin with a PLAIN value (D-P40-15). Tokens: A and B are unscoped
submitters, ``scoped`` is narrowed to ``asso``, ``ro`` is readonly, the legacy
global token is the admin.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI

from nerdit.core.secrets import SecretManager, project_storage_name
from nerdit.core.variables import load_scoped
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.projects import router as projects_router
from nerdit.daemon.routes.secrets import router as secrets_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries
from tests.test_projects_route import (
    A_RAW,
    B_RAW,
    LEGACY,
    RO_RAW,
    SCOPED_RAW,
    _auth,
    _create,
    _Env,
)

PLAIN = "plain-sentinel-7f3a"
SECRET = "secret-sentinel-9c1d"


class _Bus:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    def publish(self, event: dict) -> None:
        self.frames.append(event)


@pytest.fixture
async def env(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    for raw, tid, role, scope in (
        (A_RAW, "tok-a", TokenRole.submitter, None),
        (B_RAW, "tok-b", TokenRole.submitter, None),
        (SCOPED_RAW, "tok-scoped", TokenRole.submitter, ["asso"]),
        (RO_RAW, "tok-ro", TokenRole.readonly, None),
    ):
        await queries.create_api_token(
            ApiToken(id=tid, name=tid, role=role, token_hash=hash_token(raw), scope_services=scope)
        )
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(projects_router)
    app.include_router(secrets_router)
    app.state.queries = queries
    app.state.secret_manager = SecretManager(tmp_path / "secrets")
    app.state.hostname = "node-1"
    bus = _Bus()
    # The daemon's order, inner → outer: Idempotency, Audit, Auth, RequestId.
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: bus)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    e = _Env(app, queries, db)
    e.bus = bus  # type: ignore[attr-defined]
    e.mgr = app.state.secret_manager  # type: ignore[attr-defined]
    async with e.client:
        yield e
    await db.close()


def _url(project: str, service: str | None = None, tail: str = "") -> str:
    query = f"?service={service}" if service else ""
    return f"/projects/{project}/variables{tail}{query}"


async def _put(  # noqa: PLR0913
    env, project, values, *, raw=A_RAW, service=None, secret=None, headers=None
):
    body: dict = {"values": values}
    if secret is not None:
        body["secret"] = secret
    return await env.client.put(
        _url(project, service), json=body, headers={**_auth(raw), **(headers or {})}
    )


async def _unset(env, project, key, raw=A_RAW, service=None):
    return await env.client.delete(_url(project, service, f"/{key}"), headers=_auth(raw))


async def _list(env, project, raw=A_RAW, service=None):
    return await env.client.get(_url(project, service), headers=_auth(raw))


async def _resolve(env, project, raw=A_RAW, service=None):
    return await env.client.get(_url(project, service, "/resolve"), headers=_auth(raw))


def _by_key(resp) -> dict[str, dict]:
    return {v["key"]: v for v in resp.json()["variables"]}


def _is_owner_denial(resp) -> None:
    assert resp.status_code == 403, resp.text
    body = resp.json()
    assert body["code"] == "forbidden"
    assert body["message"] == "You do not have permission to act on this job."


async def _service_row(env, label: str, owner: str | None, project: str | None = None) -> Job:
    """A real `kind=service` row through the production reserve path (P40a stamps the triple)."""
    job = Job(
        id=f"job-{label}"[:12],
        name=label,
        service_name=label,
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image": "demo:1", "port": 8000}),
        submitted_by_token=owner,
    )
    await env.queries.reserve_service_for_token(job, admin=owner is None)
    row = await env.queries.get_service_by_name(label)
    assert row is not None
    return row


async def _flag_rows(env) -> list[tuple]:
    cursor = await env.db.conn.execute(
        "SELECT environment, service, key, plain FROM variables ORDER BY service, key"
    )
    return [tuple(r) for r in await cursor.fetchall()]


# --- D-P40-15: every cell ---------------------------------------------------------


async def test_owner_and_admin_read_a_plain_value_and_never_a_secret_one(env):
    assert (await _put(env, "asso", {"MODE": PLAIN}, secret=False)).status_code == 200
    assert (await _put(env, "asso", {"TOKEN": SECRET}, secret=True)).status_code == 200
    for raw in (A_RAW, LEGACY):
        resp = await _list(env, "asso", raw)
        assert resp.status_code == 200, resp.text
        assert resp.json()["scope"] == "project"
        got = _by_key(resp)
        assert got["MODE"] == {"key": "MODE", "scope": "project", "plain": True, "value": PLAIN}
        assert got["TOKEN"] == {"key": "TOKEN", "scope": "project", "plain": False, "value": None}
        assert SECRET not in resp.text


async def test_a_non_owner_gets_the_row_403_on_every_variable_route(env):
    await _put(env, "asso", {"MODE": PLAIN}, secret=False)
    for raw in (B_RAW, SCOPED_RAW, RO_RAW):  # unscoped, scoped-in, readonly: none owns it
        for resp in (await _list(env, "asso", raw), await _resolve(env, "asso", raw)):
            _is_owner_denial(resp)
            assert PLAIN not in resp.text
    for raw in (B_RAW, SCOPED_RAW):
        for service in (None, "web", "api"):
            _is_owner_denial(await _put(env, "asso", {"X": SECRET}, raw=raw, service=service))
        _is_owner_denial(await _unset(env, "asso", "MODE", raw))
    assert env.mgr.load(project_storage_name((await env.queries.get_project_by_name("asso")).id))
    assert not env.mgr.exists("asso")
    assert await env.queries.get_secret_claim("asso") is None


async def test_out_of_scope_is_one_403_whether_or_not_the_project_exists(env):
    await _put(env, "blog", {"MODE": PLAIN}, secret=False)
    bodies = []
    for project in ("blog", "ghost"):  # exists / does not: the scoped token must not tell
        for resp in (
            await _list(env, project, SCOPED_RAW),
            await _resolve(env, project, SCOPED_RAW),
            await _put(env, project, {"X": "1"}, raw=SCOPED_RAW),
            await _unset(env, project, "MODE", SCOPED_RAW),
        ):
            assert resp.status_code == 403, resp.text
            body = resp.json()
            body.pop("request_id", None)
            bodies.append(json.dumps(body, sort_keys=True).replace(project, "<p>"))
    assert len(set(bodies)) == 1
    assert await env.queries.get_project_by_name("ghost") is None


async def test_readonly_is_refused_on_writes(env):
    for resp in (
        await _put(env, "asso", {"X": "1"}, raw=RO_RAW),
        await _unset(env, "asso", "X", RO_RAW),
    ):
        assert resp.status_code == 403
    assert await env.queries.get_project_by_name("asso") is None


async def test_a_null_owner_project_is_admin_only(env):
    assert (await _put(env, "asso", {"MODE": PLAIN}, raw=LEGACY, secret=False)).status_code == 200
    assert (await env.queries.get_project_by_name("asso")).submitted_by_token is None
    _is_owner_denial(await _list(env, "asso", A_RAW))
    _is_owner_denial(await _resolve(env, "asso", A_RAW))
    _is_owner_denial(await _put(env, "asso", {"X": "1"}, raw=A_RAW))
    assert _by_key(await _list(env, "asso", LEGACY))["MODE"]["value"] == PLAIN


async def test_get_project_lists_names_for_the_owner_and_omits_the_section_otherwise(env):
    await _put(env, "asso", {"MODE": PLAIN}, secret=False)
    await _put(env, "asso", {"TOKEN": SECRET}, service="web")
    await _service_row(env, "asso", "tok-a")
    for raw in (A_RAW, LEGACY):
        resp = await env.client.get("/projects/asso", headers=_auth(raw))
        assert resp.json()["variables"] == [
            {"key": "MODE", "scope": "project", "plain": True},
            {"key": "TOKEN", "scope": "production/web", "plain": False},
        ]
        assert PLAIN not in resp.text and SECRET not in resp.text
    for raw in (B_RAW, SCOPED_RAW):
        resp = await env.client.get("/projects/asso", headers=_auth(raw))
        assert resp.status_code == 200 and "variables" not in resp.json()
        assert "MODE" not in resp.text


# --- creation and claims (D-P40-5) -----------------------------------------------


async def test_a_set_before_any_row_mints_the_project_for_the_caller(env):
    resp = await _put(env, "asso", {"MODE": PLAIN}, secret=False)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project": "asso", "scope": "project", "keys": ["MODE"], "plain": True}
    project = await env.queries.get_project_by_name("asso")
    assert project.submitted_by_token == "tok-a"
    assert env.mgr.load(project_storage_name(project.id)) == {"MODE": PLAIN}
    assert not env.mgr.exists("asso") and not env.mgr.exists("_shared")


async def test_a_new_project_name_is_judged_before_anything_is_minted(env):
    for name, code in (("bad--name", "project.invalid_name"), ("shared", "service.reserved_name")):
        for service in (None, "web"):
            resp = await _put(env, name, {"X": SECRET}, service=service)
            assert (resp.status_code, resp.json()["code"]) == (422, code)
        assert await env.queries.get_project_by_name(name) is None
        assert await env.queries.get_secret_claim(name) is None
    assert not env.mgr.exists("_shared")
    # Reads and unsets of a reserved name's service scope stop at the same 422.
    assert (await _list(env, "shared", LEGACY, "web")).status_code == 422
    assert (await _resolve(env, "shared", LEGACY)).status_code == 422
    assert (await _unset(env, "shared", "X", LEGACY, "web")).status_code == 422


async def test_rule_3_a_foreign_claim_refuses_the_implicit_create(env):
    resp = await env.client.post("/secrets/blog", json={"values": {"K": "1"}}, headers=_auth(B_RAW))
    assert resp.status_code == 200
    resp = await _put(env, "blog", {"X": SECRET})
    assert (resp.status_code, resp.json()["code"]) == (409, "service.name_claimed")
    assert await env.queries.get_project_by_name("blog") is None


async def test_a_service_scope_set_on_a_rowless_label_mints_the_claim(env):
    resp = await _put(env, "asso", {"TOKEN": SECRET}, service="web")
    assert resp.status_code == 200, resp.text
    assert resp.json()["scope"] == "production/web"
    assert (await env.queries.get_secret_claim("asso")).token_id == "tok-a"
    assert (await env.queries.get_project_by_name("asso")).submitted_by_token == "tok-a"
    assert env.mgr.load("asso") == {"TOKEN": SECRET}
    assert (await env.audit("variable.set"))[-1][1]["claimed"] is True
    # The claim is the P39 one: the legacy surface reads it back for its owner only.
    assert (await env.client.get("/secrets/asso", headers=_auth(A_RAW))).json()["keys"] == ["TOKEN"]
    assert (await env.client.get("/secrets/asso", headers=_auth(B_RAW))).status_code == 403


async def test_b_cannot_set_a_service_scope_inside_a_project(env):
    await _create(env, "asso")
    for service in ("web", "api"):
        _is_owner_denial(await _put(env, "asso", {"X": SECRET}, raw=B_RAW, service=service))
    # A row B owns inside A's project (admin-planted) does not open the door either.
    await _service_row(env, "api--asso", "tok-b")
    await env.db.conn.execute(
        "UPDATE jobs SET project_id = (SELECT id FROM projects WHERE name = 'asso'), "
        "service = 'api' WHERE service_name = 'api--asso'"
    )
    await env.db.conn.commit()
    _is_owner_denial(await _put(env, "asso", {"X": SECRET}, raw=B_RAW, service="api"))
    assert not env.mgr.exists("asso") and not env.mgr.exists("api--asso")
    assert await env.queries.get_secret_claim("asso") is None
    assert await _flag_rows(env) == []


async def test_a_composed_label_needs_a_row_inside_this_project(env):
    await _create(env, "asso")
    resp = await _put(env, "asso", {"X": "1"}, service="api")  # rowless
    assert (resp.status_code, resp.json()["code"]) == (404, "not_found")
    await _service_row(env, "api--asso", "tok-a")  # a legacy name: its own implicit project
    resp = await _put(env, "asso", {"X": "1"}, service="api")
    assert resp.status_code == 404
    assert not env.mgr.exists("api--asso")
    assert await env.queries.get_secret_claim("api--asso") is None
    # A legacy project name may be 41-63 chars; `<service>--<it>` then has no label (D-P40-2).
    for project, bad, code in (
        ("asso", "Bad", "project.invalid_service"),
        ("p" * 50, "a" * 20, "project.label_too_long"),
    ):
        resp = await _put(env, project, {"X": "1"}, service=bad)
        assert (resp.status_code, resp.json()["code"]) == (422, code)


async def test_project_scoped_token_reaches_its_composed_service_scope(env):
    """D-P40-7: a token scoped to `asso` reaches `api--asso` through the variables routes."""
    assert (await _create(env, "asso", raw=SCOPED_RAW)).status_code == 201
    await _service_row(env, "api--asso", "tok-scoped")
    await env.db.conn.execute(
        "UPDATE jobs SET project_id = (SELECT id FROM projects WHERE name = 'asso'), "
        "service = 'api' WHERE service_name = 'api--asso'"
    )
    await env.db.conn.commit()
    resp = await _put(env, "asso", {"K": SECRET}, raw=SCOPED_RAW, service="api")
    assert resp.status_code == 200, resp.text
    assert resp.json()["scope"] == "production/api"
    assert (await _list(env, "asso", raw=SCOPED_RAW, service="api")).status_code == 200
    assert (await _resolve(env, "asso", raw=SCOPED_RAW, service="api")).status_code == 200
    assert (await _unset(env, "asso", "K", raw=SCOPED_RAW, service="api")).status_code == 200
    # The legacy secrets surface stays label-only: the widening is the variables routes' alone.
    resp = await env.client.get("/secrets/api--asso", headers=_auth(SCOPED_RAW))
    assert resp.status_code == 403


# --- precedence and the flag -------------------------------------------------------


@pytest.mark.parametrize("project_secret", [True, False])
async def test_the_most_specific_scope_wins_regardless_of_the_flag(env, project_secret):
    await _put(env, "asso", {"KEY": "from-project", "ONLY": "p"}, secret=project_secret)
    await _put(env, "asso", {"KEY": "from-service"}, service="web", secret=not project_secret)
    resp = await _resolve(env, "asso")  # service defaults to web
    assert resp.status_code == 200, resp.text
    assert resp.json()["service"] == "web"
    assert _by_key(resp) == {
        "KEY": {"key": "KEY", "scope": "production/web", "plain": project_secret},
        "ONLY": {"key": "ONLY", "scope": "project", "plain": not project_secret},
    }
    project = await env.queries.get_project_by_name("asso")
    merged, _ = load_scoped(env.mgr, "asso", project.id)
    assert merged == {"KEY": "from-service", "ONLY": "p"}


async def test_resolve_never_returns_a_value(env):
    await _put(env, "asso", {"MODE": PLAIN}, secret=False)
    await _put(env, "asso", {"MODE2": PLAIN}, service="web", secret=False)
    resp = await _resolve(env, "asso", service="web")
    assert all(set(v) == {"key", "scope", "plain"} for v in resp.json()["variables"])
    assert len(resp.json()["variables"]) == 2 and PLAIN not in resp.text


async def test_a_plain_to_secret_flip_stops_the_value_being_listed(env):
    for service in (None, "web"):
        await _put(env, "asso", {"K": PLAIN}, service=service, secret=False)
        assert _by_key(await _list(env, "asso", service=service))["K"]["value"] == PLAIN
        await _put(env, "asso", {"K": SECRET}, service=service, secret=True)
        resp = await _list(env, "asso", service=service)
        assert _by_key(resp)["K"]["plain"] is False and _by_key(resp)["K"]["value"] is None
        assert SECRET not in resp.text and PLAIN not in resp.text
        await _put(env, "asso", {"K": PLAIN}, service=service, secret=False)  # and back
        assert _by_key(await _list(env, "asso", service=service))["K"]["value"] == PLAIN


async def test_an_omitted_secret_flag_is_write_only(env):
    await _put(env, "asso", {"K": SECRET})  # D-P40-16: fail-safe default
    resp = await _list(env, "asso")
    assert _by_key(resp)["K"] == {"key": "K", "scope": "project", "plain": False, "value": None}


async def test_a_legacy_secrets_write_over_a_plain_key_stops_it_being_listed(env):
    await _put(env, "asso", {"K": PLAIN}, service="web", secret=False)
    resp = await env.client.post(
        "/secrets/asso", json={"values": {"K": SECRET}}, headers=_auth(A_RAW)
    )
    assert resp.status_code == 200
    resp = await _list(env, "asso", service="web")
    assert _by_key(resp)["K"]["plain"] is False and SECRET not in resp.text
    # ... with a row too: the row maps the label back to its project and service.
    await _service_row(env, "asso", "tok-a")
    await _put(env, "asso", {"K": PLAIN}, service="web", secret=False)
    await env.client.post("/secrets/asso", json={"values": {"K": SECRET}}, headers=_auth(A_RAW))
    assert SECRET not in (await _list(env, "asso", service="web")).text


async def _database_row(env, label: str, owner: str) -> None:
    job = Job(
        id=f"db-{label}"[:12],
        name=label,
        service_name=label,
        kind=JobKind.database,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps({"backend": "postgres", "image": "postgres:16", "port": 5432}),
        submitted_by_token=owner,
    )
    await env.queries.reserve_service_for_token(job, admin=False)
    row = await env.queries.get_service_by_name(label)
    assert row is not None and row.project_id is None  # never project-stamped


async def test_a_database_label_is_not_the_projects_web_service(env):
    """A model/database row's file holds a minted credential: never a project variable."""
    await _put(env, "pg", {"POSTGRES_PASSWORD": PLAIN}, service="web", secret=False)  # rowless
    await _database_row(env, "pg", "tok-a")
    # The legacy write demotes the stale plain flag through the by-NAME project
    # (the row carries no project_id) ...
    resp = await env.client.post(
        "/secrets/pg", json={"values": {"POSTGRES_PASSWORD": SECRET}}, headers=_auth(A_RAW)
    )
    assert resp.status_code == 200
    assert await _flag_rows(env) == [("production", "web", "POSTGRES_PASSWORD", 0)]
    # ... and the label is not addressable as `web` at all, for owner or admin.
    for raw in (A_RAW, LEGACY):
        resp = await _list(env, "pg", raw=raw, service="web")
        assert (resp.status_code, resp.json()["code"]) == (404, "not_found")
        assert SECRET not in resp.text
    assert (await _resolve(env, "pg", service="web")).status_code == 404
    assert (await _put(env, "pg", {"X": "1"}, service="web", secret=False)).status_code == 404
    assert (await _unset(env, "pg", "POSTGRES_PASSWORD", service="web")).status_code == 404


async def test_a_minted_database_password_is_never_listable(env, tmp_path):
    """`POST /databases` writes the file directly: it must demote a pre-set plain flag."""
    from unittest.mock import MagicMock

    from nerdit.config.settings import DatabasesSettings
    from nerdit.daemon.routes.databases import router as databases_router
    from tests.test_databases_route import _data_controller

    env.app.include_router(databases_router)
    env.app.state.settings = MagicMock(databases=DatabasesSettings())
    env.app.state.data_controller = _data_controller(env.queries, tmp_path)
    await _put(env, "pg", {"POSTGRES_PASSWORD": PLAIN}, service="web", secret=False)
    resp = await env.client.post("/databases", json={"name": "pg"}, headers=_auth(A_RAW))
    assert resp.status_code == 201, resp.text
    minted = env.mgr.load("pg")["POSTGRES_PASSWORD"]
    assert minted != PLAIN
    assert await _flag_rows(env) == [("production", "web", "POSTGRES_PASSWORD", 0)]
    # Even once the row is gone and the label is the rowless `web` again.
    await env.db.conn.execute("DELETE FROM jobs WHERE service_name = 'pg'")
    await env.db.conn.commit()
    resp = await _list(env, "pg", service="web")
    assert minted not in resp.text


async def test_concurrent_plain_and_secret_sets_never_leave_a_secret_flagged_plain(env):
    """The flag/file order only fails safe against a crash; the lock covers a second writer."""
    import asyncio

    await _put(env, "asso", {"K": PLAIN}, secret=False)
    for _ in range(10):
        for first, second in ((True, False), (False, True)):
            await asyncio.gather(
                _put(env, "asso", {"K": SECRET if first else PLAIN}, secret=first),
                _put(env, "asso", {"K": SECRET if second else PLAIN}, secret=second),
            )
            assert SECRET not in (await _list(env, "asso")).text
    # The legacy `/secrets` writer takes the same lock.
    await _put(env, "asso", {"K": PLAIN}, service="web", secret=False)
    for _ in range(10):
        await asyncio.gather(
            env.client.post("/secrets/asso", json={"values": {"K": SECRET}}, headers=_auth(A_RAW)),
            _put(env, "asso", {"K": PLAIN}, service="web", secret=False),
        )
        assert SECRET not in (await _list(env, "asso", service="web")).text


async def test_the_secret_flag_lands_before_the_file_and_the_plain_one_after(env, monkeypatch):
    """The crash order (D-P40-1): whatever dies mid-set, no secret is ever flagged plain."""
    await _put(env, "asso", {"K": PLAIN}, secret=False)
    real_set = SecretManager.set

    def boom(self, service, values):
        raise OSError("disk full")

    monkeypatch.setattr(SecretManager, "set", boom)
    with pytest.raises(OSError, match="disk full"):
        await _put(env, "asso", {"K": SECRET}, secret=True)
    assert await _flag_rows(env) == [("", "", "K", 0)]  # flag first: the old value is hidden
    monkeypatch.setattr(SecretManager, "set", real_set)
    await _put(env, "asso", {"K": SECRET}, secret=True)
    monkeypatch.setattr(SecretManager, "set", boom)
    with pytest.raises(OSError, match="disk full"):
        await _put(env, "asso", {"K": PLAIN}, secret=False)
    assert await _flag_rows(env) == [("", "", "K", 0)]  # file first: the secret stays secret


# --- no value anywhere --------------------------------------------------------------


async def test_no_value_reaches_an_audit_row_event_record_log_or_error(env, caplog):
    caplog.set_level(logging.DEBUG)
    texts: list[str] = []

    async def record(resp):
        texts.append(resp.text)
        return resp

    idem = {"Idempotency-Key": "K1"}
    await record(await _put(env, "asso", {"P": PLAIN}, secret=False, headers=idem))
    await record(await _put(env, "asso", {"P": PLAIN}, secret=False, headers=idem))  # replay
    await record(await _put(env, "asso", {"S": SECRET}, secret=True))
    await record(await _put(env, "asso", {"S": SECRET, "P2": PLAIN}, service="web"))
    # Errors: a mistyped field, a wrong type, a bad key, a control character, a stranger.
    for body in (
        {"valus": {"S": SECRET}},
        {"values": {"S": [SECRET]}},
        {"values": {"S": SECRET}, "secret": SECRET},
        {"values": {"1BAD": SECRET}},
        {"values": {"S": SECRET + "\x00"}},
    ):
        resp = await record(await env.client.put(_url("asso"), json=body, headers=_auth(A_RAW)))
        assert resp.status_code == 422, resp.text
    await record(await _put(env, "asso", {"S": SECRET}, raw=B_RAW))
    await record(await _resolve(env, "asso"))
    await record(await env.client.get("/projects/asso", headers=_auth(A_RAW)))
    await record(await _unset(env, "asso", "S", A_RAW))
    listed = await _list(env, "asso")

    cursor = await env.db.conn.execute("SELECT * FROM audit_log")
    audit = json.dumps([tuple(r) for r in await cursor.fetchall()], default=str)
    cursor = await env.db.conn.execute("SELECT * FROM idempotency_keys")
    idem_rows = [dict(r) for r in await cursor.fetchall()]
    cursor = await env.db.conn.execute("SELECT * FROM variables")
    flags = json.dumps([tuple(r) for r in await cursor.fetchall()], default=str)
    haystack = "\n".join(
        [*texts, audit, json.dumps(idem_rows, default=str), flags, json.dumps(env.bus.frames)]
        + [caplog.text]
    )
    assert SECRET not in haystack and PLAIN not in haystack
    # The one sanctioned carrier: a plain value, to its owner.
    assert PLAIN in listed.text and SECRET not in listed.text
    # `secret.set`'s treatment: the value map is never digested at rest.
    assert [r["body_hash"] for r in idem_rows] == [None]
    actions = {f["action"] for f in env.bus.frames}
    assert {"variable.set", "variable.unset"} <= actions


async def test_the_audit_row_is_names_only_and_never_says_environment(env):
    await _put(env, "asso", {"B": SECRET, "A": SECRET}, service="web", secret=True)
    await _unset(env, "asso", "A", A_RAW, "web")
    (target, params) = (await env.audit("variable.set"))[-1]
    assert target == "asso"
    assert params == {
        "project": "asso",
        "scope": "production/web",
        "service": "web",
        "keys": ["A", "B"],
        "plain": False,
        "claimed": True,
    }
    assert (await env.audit("variable.unset"))[-1] == (
        "asso",
        {"project": "asso", "scope": "production/web", "service": "web", "keys": ["A"]},
    )
    cursor = await env.db.conn.execute("SELECT params_redacted FROM audit_log")
    assert all("environment" not in (r[0] or "") for r in await cursor.fetchall())


# --- delete -------------------------------------------------------------------------


async def test_delete_variable_removes_the_key_and_its_flag_and_keeps_the_claim(env):
    await _put(env, "asso", {"K": PLAIN, "KEEP": PLAIN}, service="web", secret=False)
    resp = await _unset(env, "asso", "K", A_RAW, "web")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"project": "asso", "scope": "production/web", "deleted": "K"}
    assert env.mgr.load("asso") == {"KEEP": PLAIN}
    assert await _flag_rows(env) == [("production", "web", "KEEP", 1)]
    again = await _unset(env, "asso", "K", A_RAW, "web")
    assert (again.status_code, again.json()["code"]) == (404, "not_found")
    # The last key goes, the P39 claim stays: only `DELETE /secrets/{label}` releases it.
    await _unset(env, "asso", "KEEP", A_RAW, "web")
    assert (await env.queries.get_secret_claim("asso")).token_id == "tok-a"
    # Project scope, and an unknown project is the 404 after the scope check.
    await _put(env, "asso", {"P": PLAIN}, secret=False)
    assert (await _unset(env, "asso", "P", A_RAW)).status_code == 200
    assert _by_key(await _list(env, "asso")) == {}
    assert (await _unset(env, "nope", "P", A_RAW)).status_code == 404


async def test_delete_project_removes_the_project_file_and_the_flag_rows(env):
    await _put(env, "asso", {"P": PLAIN}, secret=False)
    await _put(env, "asso", {"S": SECRET}, service="web")
    project = await env.queries.get_project_by_name("asso")
    storage = project_storage_name(project.id)
    assert env.mgr.exists(storage) and len(await _flag_rows(env)) == 2
    resp = await env.client.delete("/projects/asso", headers=_auth(A_RAW))
    assert resp.status_code == 200, resp.text
    assert not env.mgr.exists(storage)
    assert await _flag_rows(env) == []  # by FK (D-P40-17)
    # A recreated project starts empty: the id, and so the file, is new.
    await _create(env, "asso")
    assert _by_key(await _list(env, "asso")) == {}


async def test_variable_id_resolves_inside_write_lock_and_keeps_local_ownership(env):
    project = (await _create(env, "asso")).json()["id"]
    await _service_row(env, "asso", "tok-a")
    for service in (None, "web"):
        result = await _put(env, project, {"MODE": "test"}, service=service, secret=False)
        assert result.status_code == 200, result.text
        assert result.json()["project"] == "asso"
        assert _by_key(await _list(env, project, service=service))["MODE"]["value"] == "test"
        assert (await _resolve(env, project, service=service)).status_code == 200
        assert (
            await _put(env, project, {"MODE": "foreign"}, raw=B_RAW, service=service)
        ).status_code == 403
        assert (await _unset(env, project, "MODE", service=service)).status_code == 200


async def test_retired_project_id_cannot_create_or_write_same_name_replacement(env):
    original = (await _create(env, "asso")).json()["id"]
    assert await env.queries.delete_project_checked(original) == []
    replacement = (await _create(env, "asso")).json()["id"]
    for service in (None, "web"):
        result = await _put(env, original, {"MODE": "stale"}, service=service)
        assert result.status_code == 404
    assert (await _list(env, replacement)).json()["variables"] == []
    assert not env.mgr.exists(project_storage_name(original))


async def test_idempotency_key_cannot_replay_across_project_incarnations(env):
    original = (await _create(env, "asso")).json()["id"]
    headers = {"Idempotency-Key": "one-project-only"}
    assert (await _put(env, original, {"MODE": "old"}, headers=headers)).status_code == 200
    assert await env.queries.delete_project_checked(original) == []
    replacement = (await _create(env, "asso")).json()["id"]
    result = await _put(env, replacement, {"MODE": "old"}, headers=headers)
    assert result.status_code == 422 and result.json()["code"] == "idempotency_key_conflict"
    assert (await _list(env, replacement)).json()["variables"] == []


@pytest.mark.parametrize("service", [None, "web"])
async def test_variable_id_is_rechecked_after_waiting_for_writer_lock(env, service):
    original = (await _create(env, "asso")).json()["id"]
    waiting = asyncio.Event()

    class WriterLock(asyncio.Lock):
        async def acquire(self):
            waiting.set()
            return await super().acquire()

    lock = WriterLock()
    env.app.state.variable_write_lock = lock
    await lock.acquire()
    waiting.clear()
    task = asyncio.create_task(_put(env, original, {"MODE": "stale"}, service=service))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert await env.queries.delete_project_checked(original) == []
        replacement = (await _create(env, "asso")).json()["id"]
    finally:
        lock.release()
    result = await task
    assert result.status_code == 404
    assert (await _list(env, replacement)).json()["variables"] == []
    assert not env.mgr.exists("asso")


@pytest.mark.parametrize("service", [None, "web"])
async def test_project_retirement_waits_for_authorized_variable_writer(env, monkeypatch, service):
    import nerdit.daemon.secret_scope as scopes

    original = (await _create(env, "asso")).json()["id"]
    judged, proceed, deletion_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    judge = scopes.judge_project

    class WriterLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                deletion_waiting.set()
            return await super().acquire()

    async def pause_after_identity(request, ident):
        row = await judge(request, ident)
        judged.set()
        await proceed.wait()
        return row

    env.app.state.variable_write_lock = WriterLock()
    monkeypatch.setattr(scopes, "judge_project", pause_after_identity)
    write = asyncio.create_task(_put(env, original, {"MODE": "old"}, service=service, secret=False))
    await asyncio.wait_for(judged.wait(), timeout=1)
    delete = asyncio.create_task(env.client.delete(f"/projects/{original}", headers=_auth(A_RAW)))
    try:
        await asyncio.wait_for(deletion_waiting.wait(), timeout=1)
        # The ID must remain authoritative while the writer still owns the lock.
        assert await env.queries.get_project(original) is not None
    finally:
        proceed.set()
        write_result, delete_result = await asyncio.gather(write, delete)
    assert write_result.status_code == 200, write_result.text
    assert write_result.json()["project"] == "asso"
    assert delete_result.status_code == 200, delete_result.text
    assert delete_result.json()["name"] == "asso"
    replacement = (await _create(env, "asso")).json()["id"]
    assert replacement != original
    assert (await _put(env, original, {"MODE": "late"}, service=service)).status_code == 404
    if service is not None:
        assert env.mgr.load("asso")["MODE"] == "old"
    else:
        assert not env.mgr.exists(project_storage_name(original))


@pytest.mark.parametrize("reader", [_list, _resolve], ids=["list", "resolve"])
async def test_variable_reader_pins_identity_until_snapshot_finishes(env, monkeypatch, reader):
    import nerdit.daemon.routes.projects as projects

    original = (await _create(env, "asso")).json()["id"]
    assert (
        await _put(env, original, {"ORIGINAL_KEY": "original"}, service="web")
    ).status_code == 200
    judged, proceed, deletion_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owned = projects._owned

    class WriterLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                deletion_waiting.set()
            return await super().acquire()

    async def pause_after_identity(request, ident):
        row = await owned(request, ident)
        judged.set()
        await proceed.wait()
        return row

    env.app.state.variable_write_lock = WriterLock()
    monkeypatch.setattr(projects, "_owned", pause_after_identity)
    read = asyncio.create_task(reader(env, original, service="web"))
    await asyncio.wait_for(judged.wait(), timeout=1)
    delete = asyncio.create_task(env.client.delete(f"/projects/{original}", headers=_auth(A_RAW)))
    try:
        await asyncio.wait_for(deletion_waiting.wait(), timeout=1)
        assert await env.queries.get_project(original) is not None
    finally:
        proceed.set()
        read_result, delete_result = await asyncio.gather(read, delete)
    assert read_result.status_code == 200, read_result.text
    assert "ORIGINAL_KEY" in _by_key(read_result)
    assert delete_result.status_code == 200, delete_result.text
    replacement = (await _create(env, "asso")).json()["id"]
    assert replacement != original
    assert (
        await _put(env, replacement, {"REPLACEMENT_KEY": "replacement"}, service="web")
    ).status_code == 200
    stale = await reader(env, original, service="web")
    assert stale.status_code == 404 and "REPLACEMENT_KEY" not in stale.text


@pytest.mark.parametrize("reader", [_list, _resolve], ids=["list", "resolve"])
async def test_variable_reader_resolves_original_id_after_waiting_for_lock(env, reader):
    original = (await _create(env, "asso")).json()["id"]
    waiting = asyncio.Event()

    class WriterLock(asyncio.Lock):
        async def acquire(self):
            waiting.set()
            return await super().acquire()

    lock = WriterLock()
    env.app.state.variable_write_lock = lock
    await lock.acquire()
    waiting.clear()
    read = asyncio.create_task(reader(env, original, service="web"))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=1)
        # Simulate the operation already holding the lock retiring this identity.
        assert await env.queries.delete_project_checked(original) == []
        replacement = await env.queries.create_project("asso", "tok-a")
        assert replacement.id != original
        assert await env.queries.mint_secret_claim("asso", "tok-a")
        env.mgr.set("asso", {"REPLACEMENT_KEY": "replacement"})
    finally:
        lock.release()
    result = await read
    assert result.status_code == 404 and "REPLACEMENT_KEY" not in result.text
