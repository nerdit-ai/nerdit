"""WP-B1 delete-purge tests: the pure reference-guard helper, image purge over a
protected set, and the purge-data-failure → GC-opt-in recovery integration.

The reference-guard equality logic is unit-tested directly on
:func:`_find_model_dependents`; the destructive purge steps run through the real
``/services`` DELETE route against a hand-built app (mocked queries, a
``FakeRuntime`` supplying only the image methods) so an SQL/route regression is
caught. The GC leg reuses the WP-A2 ``POST /system/gc`` route on the same
``data_dir`` to prove a failed data purge is recoverable.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from nerdit.core import workspaces as core_workspaces
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.daemon import service_purge as services_mod
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.imagegc import _orphan_data_dir_names
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.services import router as services_router
from nerdit.daemon.routes.system import router as system_router
from nerdit.daemon.routes.workspaces import router as workspaces_router
from nerdit.daemon.service_purge import (
    _find_db_dependents,
    _find_model_dependents,
    _sweep_data_tombstones,
)
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole

ADMIN_RAW = "admin-raw"
SUB_RAW = "sub-raw"
_TOKENS = {
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
}


def _auth() -> dict:
    return {"Authorization": f"Bearer {ADMIN_RAW}"}


def _auth_sub() -> dict:
    return {"Authorization": f"Bearer {SUB_RAW}"}


# --- _find_model_dependents (pure ref equality) ------------------------------


def _row(jid, ai, *, kind="service", name="app"):  # noqa: ANN001
    return {
        "id": jid,
        "kind": kind,
        "service_name": name,
        "status": "running",
        "config": {"ai": ai},
    }


def test_find_dependents_ref_equality_ollama():
    rows = [_row("s1", {"default": {"provider": "ollama", "model": "llama3.1:8b"}})]
    deps = _find_model_dependents(rows, "llama3.1:8b", exclude_id="mdl-1")
    assert deps == [{"service": "app", "id": "s1", "binding": "default"}]


def test_find_dependents_api_never_matches():
    rows = [_row("s1", {"default": {"provider": "api", "model": "llama3.1:8b"}})]
    assert _find_model_dependents(rows, "llama3.1:8b", exclude_id="mdl-1") == []


def test_find_dependents_excludes_target_and_wrong_ref():
    rows = [
        _row("mdl-1", {}),  # the target row itself — excluded by id
        _row("s1", {"default": {"provider": "ollama", "model": "other:tag"}}),
    ]
    assert _find_model_dependents(rows, "llama3.1:8b", exclude_id="mdl-1") == []


def test_find_dependents_duplicate_ref_all_reported():
    rows = [
        _row("s1", {"a": {"provider": "ollama", "model": "ref"}}, name="app1"),
        _row("s2", {"b": {"provider": "ollama", "model": "ref"}}, name="app2"),
    ]
    deps = _find_model_dependents(rows, "ref", exclude_id="mdl-1")
    assert {d["service"] for d in deps} == {"app1", "app2"}


# --- _find_db_dependents (P15 managed [db.*] ref equality) -------------------


def _db_row(jid, db, *, kind="service", name="app"):  # noqa: ANN001
    return {
        "id": jid,
        "kind": kind,
        "service_name": name,
        "status": "running",
        "config": {"db": db},
    }


def test_find_db_dependents_managed_ref_equality():
    rows = [_db_row("s1", {"default": {"provider": "managed", "database": "pg"}})]
    deps = _find_db_dependents(rows, "pg", exclude_id="db-1")
    assert deps == [{"service": "app", "id": "s1", "binding": "default"}]


def test_find_db_dependents_external_never_matches():
    rows = [_db_row("s1", {"default": {"provider": "external", "database": "pg"}})]
    assert _find_db_dependents(rows, "pg", exclude_id="db-1") == []


def test_find_db_dependents_wrong_target_excluded():
    rows = [_db_row("s1", {"default": {"provider": "managed", "database": "other"}})]
    assert _find_db_dependents(rows, "pg", exclude_id="db-1") == []


# --- P15 D5: database delete is explicit-destructive -------------------------


def _db_svc(owner="tok-admin", **over) -> Job:
    fields = dict(
        id="db-a",
        service_name="pg",
        name="pg",
        kind=JobKind.database,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(
            {
                "backend": "postgres",
                "image": "postgres:16",
                "port": 5432,
                "volumes": ["data:/var/lib/postgresql/data"],
            }
        ),
        submitted_by_token=owner,
    )
    fields.update(over)
    return Job(**fields)


def test_delete_database_without_purge_data_is_409():
    q = _queries(_db_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a", headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "db.delete_requires_purge"
    q.delete_service_checked.assert_not_awaited()


def test_delete_database_purge_gate_not_bypassable_by_force():
    """``?force`` bypasses only the reference guard, never the D5 data-loss gate."""
    q = _queries(_db_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a?force=true", headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "db.delete_requires_purge"


def test_delete_database_purge_data_force_includes_secrets(tmp_path):
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(), workloads=[])
    runtime = FakeRuntime([])
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    # ``data`` force-includes ``secrets`` (coupled lifecycle): both ran.
    assert body["purged"]["data"] is True
    assert body["purged"]["secrets"] is True
    app.state.secret_manager.delete.assert_called()  # the minted credential purged too
    assert not (tmp_path / "services" / "pg").exists()


def test_delete_database_in_use_is_409_with_database_label():
    dependent = _db_row("s1", {"default": {"provider": "managed", "database": "pg"}})
    q = _queries(_db_svc(), workloads=[dependent])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    # purge=data satisfies the D5 gate so the reference guard is what fires.
    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "resource.in_use"
    assert body["message"].startswith("Database 'db-a'")
    assert body["dependents"] == [{"service": "app", "id": "s1", "binding": "default"}]


# --- P15 D5/M2: the atomic BEGIN-IMMEDIATE re-check checker wiring (db kind) --
# The fast-path reference guard only sees the pre-teardown snapshot; the real
# TOCTOU protection is the checker closure handed to ``delete_service_checked``,
# re-run inside the write lock (services.py:919-937). These mirror the
# ``kind=model`` coverage at ``tests/test_routes_services.py`` — without them a
# regression dropping the db checker to ``None`` would pass silently, since every
# other db-delete test mocks ``delete_service_checked`` returning ``[]``.


def test_delete_database_non_force_passes_ref_checker():
    """A non-force database delete carries a managed-[db.*] ref-equality checker."""
    q = _queries(_db_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 200
    checker = q.delete_service_checked.await_args.args[1]
    assert callable(checker)
    # Pure function over workload rows (db_ref = the row's service_name "pg"): a
    # managed binding on "pg" is a dependent; external / unrelated target is not.
    managed = _db_row("s1", {"default": {"provider": "managed", "database": "pg"}})
    assert checker([managed]) == [{"service": "app", "id": "s1", "binding": "default"}]
    external = _db_row("s1", {"default": {"provider": "external", "database": "pg"}})
    assert checker([external]) == []
    assert checker([]) == []


def test_delete_database_non_admin_force_passes_cross_owner_checker():
    """A non-admin ``force`` db delete carries the CROSS-OWNER checker (M2)."""
    q = _queries(_db_svc(owner="tok-sub"), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a?purge=data&force=true", headers=_auth_sub())
    assert resp.status_code == 200
    checker = q.delete_service_checked.await_args.args[1]
    assert callable(checker)
    # Same-owner dependents are owner-forceable (filtered out); foreign ones stay.
    spec = {"default": {"provider": "managed", "database": "pg"}}
    same = {**_db_row("s1", spec), "submitted_by_token": "tok-sub"}
    foreign = {**_db_row("s1", spec), "submitted_by_token": "tok-other"}
    null_owner = {**_db_row("s1", spec), "submitted_by_token": None}
    assert checker([same]) == []
    assert checker([foreign]) == [{"service": "app", "id": "s1", "binding": "default"}]
    # A NULL owner is fail-closed (treated as foreign).
    assert checker([null_owner]) == [{"service": "app", "id": "s1", "binding": "default"}]


def test_delete_database_admin_force_skips_checker():
    """ADMIN ``force`` intends to delete regardless → no atomic checker (None)."""
    q = _queries(_db_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a?purge=data&force=true", headers=_auth())
    assert resp.status_code == 200
    assert q.delete_service_checked.await_args.args[1] is None


def test_delete_database_guard_rerun_restores_running():
    """A dependent appearing during teardown → 409 AND desired_state restored.

    The fast-path snapshot is clean, but the atomic re-check (inside
    ``delete_service_checked``) reports a fresh dependent: teardown already
    happened, so ``desired_state`` is restored to ``running`` (the reconciler
    relaunches the database) and the row survives — never left stopped.
    """
    q = _queries(_db_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(
        return_value=[{"service": "app", "id": "s1", "binding": "default"}]
    )
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )
    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "resource.in_use"
    assert body["message"].startswith("Database 'db-a'")
    assert "relaunched" in body["message"]
    # The atomic delete was invoked WITH a callable checker (db kind, non-force).
    args = q.delete_service_checked.await_args.args
    assert args[0] == "db-a"
    assert callable(args[1])
    # desired_state set 'stopped' (teardown) then restored to 'running'.
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert ("db-a", "stopped") in states
    assert states[-1] == ("db-a", "running")


def test_delete_database_atomic_purge_failure_aborts_intact(tmp_path, monkeypatch):
    """#1 — a database data move-aside that FAILS aborts the whole delete.

    C2 turned the atomic-first purge into a tombstone RENAME, so the destructive
    pre-delete step is now ``os.rename`` (not ``rmtree``); a failed rename keeps
    the exact old ``db.purge_failed`` abort posture. The row, the minted secret
    and the data all survive (never the unrecoverable orphan of row+secret gone
    with data left behind), the DB is relaunched, and a loud ``500
    db.purge_failed`` is returned.
    """
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(), workloads=[])
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    def _boom(src, dst, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise OSError("permission denied")

    monkeypatch.setattr(services_mod.os, "rename", _boom)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 500
    assert resp.json()["code"] == "db.purge_failed"
    # The delete aborted BEFORE any destructive step — nothing partial:
    q.delete_service_checked.assert_not_awaited()  # row intact
    app.state.secret_manager.delete.assert_not_called()  # minted secret intact
    assert (tmp_path / "services" / "pg").is_dir()  # data intact (never renamed away)
    # No tombstone leaked (the rename failed, so nothing moved).
    assert not any(p.name.startswith(".trash-") for p in (tmp_path / "services").iterdir())
    # Teardown stopped it; the abort restores 'running' (reconciler relaunches).
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert states[-1] == ("db-a", "running")


def test_delete_database_tombstone_skipped_when_container_still_alive(tmp_path):
    """The F12 live-writer gate reaches the DATABASE tombstone move too (:1030).

    Teardown raised (Docker wedged) AND the confirmation probe reports the
    container is still running: ``_no_live_writer`` is not satisfied, so
    ``_tombstone_data`` is called with ``skip_reason='container_alive'`` and
    aborts BEFORE any filesystem access — never renaming a live writer's PGDATA
    out from under it. This is the database-kind twin of the service-kind pin at
    ``test_routes_services.py::test_delete_data_purge_skipped_when_container_still_alive``
    (and the ``:1020``/``:1054``/``:1095`` audit-reason pins there); those never
    exercise the database branch (`services.py:1030`) because ``_db_svc`` carries
    no ``container_id`` by default, so the teardown block is skipped entirely.
    """
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(container_id="c1"), workloads=[])
    exc = ContainerRuntimeError("docker is wedged")
    runtime = AsyncMock()
    runtime.stop = AsyncMock(side_effect=exc)
    runtime.kill = AsyncMock(side_effect=exc)
    runtime.remove = AsyncMock(side_effect=exc)
    runtime.container_running = AsyncMock(return_value=True)  # confirmed still writing
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 500
    assert resp.json()["code"] == "db.purge_failed"
    runtime.container_running.assert_awaited_once_with("c1")
    # Aborted before any destructive step: row, secret and data all intact.
    q.delete_service_checked.assert_not_awaited()
    app.state.secret_manager.delete.assert_not_called()
    assert (tmp_path / "services" / "pg" / "blob").is_file()
    assert not any(p.name.startswith(".trash-") for p in (tmp_path / "services").iterdir())
    # Teardown stopped it; the abort restores 'running' (reconciler relaunches).
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert states[-1] == ("db-a", "running")
    # The audited skip reason mirrors the service-kind shape exactly.
    rows = [
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "service.purge_data"
    ]
    assert rows, "no service.purge_data audit row"
    params = json.loads(rows[-1].kwargs["params_redacted"])
    assert params == {"key": "services/pg", "purged": False, "reason": "container_alive"}


def test_delete_database_aborts_when_concurrent_tombstone_present(tmp_path):
    """A concurrent DELETE already renamed the tree aside → this delete aborts.

    Two concurrent DELETEs of one database can interleave so the real root is
    absent (the other renamed it into a tombstone) — treating that as
    ``(True, None)`` would let this delete commit while the other's checked delete
    is refused and restores the tombstone, leaving restored data at the canonical
    path with row+secret gone. ``_tombstone_data`` detects the live tombstone and
    returns the failure shape, so the route aborts like a failed rename.
    """
    (tmp_path / "services" / ".trash-pg-abcd1234").mkdir(parents=True)
    (tmp_path / "services" / ".trash-pg-abcd1234" / "blob").write_bytes(b"x" * 8)
    # No real ``services/pg`` root — the concurrent DELETE already renamed it.

    q = _queries(_db_svc(), workloads=[])
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 500
    assert resp.json()["code"] == "db.purge_failed"
    # Aborted before any destructive step — row + secret intact, DB relaunched.
    q.delete_service_checked.assert_not_awaited()
    app.state.secret_manager.delete.assert_not_called()
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert states[-1] == ("db-a", "running")
    # The other DELETE's tombstone was left untouched.
    assert (tmp_path / "services" / ".trash-pg-abcd1234" / "blob").is_file()


# --- C2: refused delete must NOT destroy data (tombstone rename) -------------


def test_refused_db_delete_preserves_data_at_original_path(tmp_path):
    """C2 regression — a REFUSED database delete leaves the data intact on disk.

    The fast pre-teardown snapshot is clean, but the atomic BEGIN-IMMEDIATE
    re-check finds a dependent that committed during teardown (simulated by
    ``delete_service_checked`` returning it). Pre-fix the data dir was already
    rmtree'd atomic-first, so the refused delete silently destroyed it. Post-fix
    the tree was only renamed to a tombstone and is renamed BACK before the
    relaunch — the sentinel survives at its original path.
    """
    (tmp_path / "services" / "pg" / "PGDATA").mkdir(parents=True)
    sentinel = tmp_path / "services" / "pg" / "PGDATA" / "postgresql.conf"
    sentinel.write_bytes(b"listen = '*'\n")

    q = _queries(_db_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(
        return_value=[{"service": "app", "id": "s1", "binding": "default"}]
    )
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "resource.in_use"
    assert "relaunched" in body["message"]
    # The DB is relaunched (desired_state restored) and its data survived intact.
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert states[-1] == ("db-a", "running")
    assert sentinel.is_file()
    assert sentinel.read_bytes() == b"listen = '*'\n"
    # The tombstone was renamed back — no ``.trash-*`` leaked.
    assert not any(p.name.startswith(".trash-") for p in (tmp_path / "services").iterdir())
    # The minted secret was never purged (the delete was refused).
    app.state.secret_manager.delete.assert_not_called()


def test_committed_db_delete_reaps_tombstone(tmp_path):
    """A COMMITTED database delete removes both the data and its tombstone."""
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(), workloads=[])  # delete_service_checked returns [] ⇒ commit
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["data"] is True
    assert not (tmp_path / "services" / "pg").exists()  # data gone
    # No tombstone leaked — reaped on commit.
    assert not any(p.name.startswith(".trash-") for p in (tmp_path / "services").iterdir())


def test_committed_db_delete_reap_failure_reports_unpurged(tmp_path, monkeypatch):
    """A failed tombstone rmtree on commit is reported honestly as data NOT purged.

    ``purged.data`` means "the bytes are gone"; when the post-commit reap fails
    the tree survives in the tombstone (GC-reclaimable), so the response and the
    audit must both say ``False`` — never claim a destructive purge completed.
    """
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(), workloads=[])  # delete_service_checked returns [] ⇒ commit
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    def _rmtree_fail(path, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise OSError("device busy")

    monkeypatch.setattr(services_mod.shutil, "rmtree", _rmtree_fail)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 200  # the delete itself committed
    assert resp.json()["purged"]["data"] is False  # ...but the bytes survive
    # The tree lives on in the tombstone, awaiting the GC orphan pass.
    tombstones = [p for p in (tmp_path / "services").iterdir() if p.name.startswith(".trash-")]
    assert len(tombstones) == 1
    assert (tombstones[0] / "blob").exists()


def test_refused_db_delete_restore_failure_keeps_stopped(tmp_path, monkeypatch):
    """If the tombstone cannot be renamed BACK, the row is left stopped (not empty).

    A failed restore must never relaunch the DB on an empty data root: keep
    ``desired_state='stopped'`` and surface a loud ``500 db.restore_failed``.
    """
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "blob").write_bytes(b"x" * 8)

    q = _queries(_db_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(
        return_value=[{"service": "app", "id": "s1", "binding": "default"}]
    )
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    real_rename = services_mod.os.rename
    calls = {"n": 0}

    def _rename(src, dst, *a, **k):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 1:  # the move-aside succeeds
            return real_rename(src, dst, *a, **k)
        raise OSError("permission denied")  # the restore-back fails

    monkeypatch.setattr(services_mod.os, "rename", _rename)

    resp = client.delete("/services/db-a?purge=data", headers=_auth())
    assert resp.status_code == 500
    assert resp.json()["code"] == "db.restore_failed"
    # desired_state stopped (teardown) and NEVER restored to running.
    states = [c.args for c in q.set_desired_state.await_args_list]
    assert ("db-a", "stopped") in states
    assert ("db-a", "running") not in states


# --- C2: startup sweep + orphan computation (crash-window hygiene) ------------


async def test_startup_sweep_restores_tombstone_with_live_row(tmp_path):
    """A crash mid-delete leaves a tombstone + a live row → the sweep restores it."""
    (tmp_path / "services" / ".trash-pg-abcd1234").mkdir(parents=True)
    (tmp_path / "services" / ".trash-pg-abcd1234" / "blob").write_bytes(b"x" * 8)

    q = AsyncMock()
    q.list_workload_configs = AsyncMock(
        return_value=[
            {"id": "db-a", "kind": "database", "service_name": "pg", "desired_state": "stopped"}
        ]
    )
    await _sweep_data_tombstones(q, tmp_path)

    assert (tmp_path / "services" / "pg" / "blob").is_file()  # restored to the real root
    assert not (tmp_path / "services" / ".trash-pg-abcd1234").exists()


async def test_startup_sweep_leaves_tombstone_when_row_running(tmp_path):
    """A live row that is NOT stopped (e.g. a fresh same-name recreate) → no restore.

    A ``POST /databases`` row is ``running`` with no data dir yet; grafting an old
    tombstone onto it would boot the new database on stale weights whose minted
    password can never match. The tombstone is left in place (GC-protected).
    """
    (tmp_path / "services" / ".trash-pg-abcd1234").mkdir(parents=True)
    (tmp_path / "services" / ".trash-pg-abcd1234" / "blob").write_bytes(b"x" * 8)

    q = AsyncMock()
    q.list_workload_configs = AsyncMock(
        return_value=[
            {"id": "db-a", "kind": "database", "service_name": "pg", "desired_state": "running"}
        ]
    )
    await _sweep_data_tombstones(q, tmp_path)

    assert (tmp_path / "services" / ".trash-pg-abcd1234" / "blob").is_file()  # untouched
    assert not (tmp_path / "services" / "pg").exists()  # not grafted onto the fresh row


async def test_startup_sweep_leaves_tombstone_when_row_gone(tmp_path):
    """A committed delete leaves the tombstone (base row gone) for the GC pass."""
    (tmp_path / "services" / ".trash-pg-abcd1234").mkdir(parents=True)

    q = AsyncMock()
    q.list_workload_configs = AsyncMock(return_value=[])  # no live row for 'pg'
    await _sweep_data_tombstones(q, tmp_path)

    assert (tmp_path / "services" / ".trash-pg-abcd1234").is_dir()  # untouched
    assert not (tmp_path / "services" / "pg").exists()


async def test_startup_sweep_skips_when_real_root_present(tmp_path):
    """A present real data root means nothing to restore — the tombstone is left."""
    (tmp_path / "services" / ".trash-pg-abcd1234").mkdir(parents=True)
    (tmp_path / "services" / "pg").mkdir(parents=True)

    q = AsyncMock()
    q.list_workload_configs = AsyncMock(
        return_value=[
            {"id": "db-a", "kind": "database", "service_name": "pg", "desired_state": "stopped"}
        ]
    )
    await _sweep_data_tombstones(q, tmp_path)

    assert (tmp_path / "services" / ".trash-pg-abcd1234").is_dir()  # not consumed


def test_orphan_data_dir_names_tombstone_aware():
    """Orphan computation lists a row-less tombstone and skips a live-row one."""
    dir_names = [
        "pg",  # live plain dir — not orphan
        ".trash-pg-abcd1234",  # base 'pg' is live — skipped
        ".trash-old-deadbeef",  # base 'old' is gone — orphan
        "gone",  # plain dir, no live row — orphan
    ]
    live = {"pg"}
    assert _orphan_data_dir_names(dir_names, live) == [".trash-old-deadbeef", "gone"]


# --- image purge over a protected set (real DELETE route) --------------------


class FakeRuntime:
    """Runtime exposing only the image methods the purge path calls."""

    def __init__(self, detailed):  # noqa: ANN001
        self._detailed = [dict(e) for e in detailed]
        self.removed: list[str] = []

    async def list_images_detailed(self):  # noqa: ANN201
        return [dict(e) for e in self._detailed]

    async def remove_image(self, tag, force=False):  # noqa: ANN001
        self.removed.append(tag)
        self._detailed = [e for e in self._detailed if e["repo_tag"] != tag]


def _svc(owner="tok-admin", **over) -> Job:
    fields = dict(
        id="svc-a",
        service_name="a",
        name="a",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        config=json.dumps({"image_repo": "nerdit-app/a", "image": "nerdit-app/a:2"}),
        submitted_by_token=owner,
    )
    fields.update(over)
    return Job(**fields)


def _queries(job, *, workloads):  # noqa: ANN001
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.touch_api_token = AsyncMock()
    q.insert_audit_log = AsyncMock()
    q.get_job = AsyncMock(return_value=job)
    q.get_service_by_name = AsyncMock(return_value=None)
    q.set_desired_state = AsyncMock()
    q.release_gpus = AsyncMock()
    q.release_service_endpoint = AsyncMock()
    q.delete_service_checked = AsyncMock(return_value=[])
    q.list_workload_configs = AsyncMock(return_value=list(workloads))
    return q


def _app(*, runtime, queries, data_dir) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(services_router)
    from fastapi import APIRouter

    api = APIRouter(prefix="/api")
    api.include_router(system_router)
    app.include_router(api)
    app.state.queries = queries
    app.state.runtime = runtime
    app.state.settings = SimpleNamespace(
        data_dir=str(data_dir), retention=SimpleNamespace(backup_keep_last=0)
    )
    secret_manager = MagicMock()
    secret_manager.delete = MagicMock(return_value=True)
    app.state.secret_manager = secret_manager
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token="legacy", get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def test_image_purge_spares_other_live_repo():
    detailed = [
        {"repo_tag": "nerdit-app/a:1", "id": "a1", "size_bytes": 5},
        {"repo_tag": "nerdit-app/a:2", "id": "a2", "size_bytes": 5},
        {"repo_tag": "nerdit-app/b:1", "id": "b1", "size_bytes": 5},
    ]
    other = {
        "id": "svc-b",
        "kind": "service",
        "service_name": "b",
        "status": "running",
        "config": {"image_repo": "nerdit-app/b", "image": "nerdit-app/b:1"},
    }
    runtime = FakeRuntime(detailed)
    # list_workload_configs returns both rows; the route filters out the deleted id.
    q = _queries(
        _svc(),
        workloads=[
            {
                "id": "svc-a",
                "kind": "service",
                "service_name": "a",
                "status": "running",
                "config": {"image_repo": "nerdit-app/a"},
            },
            other,
        ],
    )
    client = TestClient(
        _app(runtime=runtime, queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )

    body = client.delete("/services/svc-a?purge=images", headers=_auth()).json()
    assert set(body["purged"]["images"]["removed"]) == {"nerdit-app/a:1", "nerdit-app/a:2"}
    assert body["purged"]["images"]["skipped"] == []
    # The other live repo's tag was spared entirely.
    assert "nerdit-app/b:1" not in runtime.removed


def test_image_purge_never_targets_model_rows():
    """A ``kind=model`` delete never enters the image purge (images stays None)."""
    model = _svc(
        id="mdl-1",
        service_name="llama",
        name="llama",
        kind=JobKind.model,
        config=json.dumps({"model": "llama3.1:8b", "image": "ollama/ollama:latest"}),
    )
    runtime = FakeRuntime([{"repo_tag": "ollama/ollama:latest", "id": "o", "size_bytes": 9}])
    q = _queries(model, workloads=[])
    client = TestClient(
        _app(runtime=runtime, queries=q, data_dir="/tmp"), raise_server_exceptions=False
    )

    body = client.delete("/services/mdl-1?purge=images", headers=_auth()).json()
    assert body["purged"]["images"] is None
    assert runtime.removed == []


# --- purge-data failure → GC opt-in recovery (integration) -------------------


def test_purge_data_failure_is_recovered_by_gc(tmp_path, monkeypatch):
    (tmp_path / "services" / "doomed").mkdir(parents=True)
    (tmp_path / "services" / "doomed" / "blob").write_bytes(b"x" * 8)

    svc = _svc(id="svc-d", service_name="doomed", name="doomed")
    q = _queries(svc, workloads=[])
    runtime = FakeRuntime([])
    runtime.disk_usage = AsyncMock(return_value={"build_cache_bytes": 0})
    client = TestClient(
        _app(runtime=runtime, queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    # Make the FIRST rmtree (the delete's data purge) fail, then delegate to the
    # real rmtree so the later GC pass can actually reclaim the dir. (services and
    # system share the one ``shutil`` module object, so this single patch covers
    # both call sites — the call-count switch is what separates the two legs.)
    real_rmtree = services_mod.shutil.rmtree
    calls = {"n": 0}

    def _boom(path, *a, **k):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("device busy")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(services_mod.shutil, "rmtree", _boom)

    body = client.delete("/services/svc-d?purge=data", headers=_auth()).json()
    assert body["purged"]["data"] is False  # the rmtree failed, honestly reported
    assert (tmp_path / "services" / "doomed").is_dir()  # data survived the failed purge

    # Recovery: GC (opt-in orphan data) reclaims the now-orphan dir.
    gc = client.post("/api/system/gc", headers=_auth(), json={"include_orphan_data": True}).json()
    assert gc["orphan_data"]["removed"] == ["doomed"]
    assert not (tmp_path / "services" / "doomed").exists()


# --- P20 run-race hook: unforced 409, forced kills the run -------------------


class _RunController:
    """Minimal stand-in for the controller's rowless-container registries.

    Only the accessors the DELETE route reads — the P20 rowless-run pair and
    (P24b) the cutover pair; the real ``ServiceController`` is far too heavy to
    construct for a route test, and its registries are plain in-memory dicts
    anyway.
    """

    def __init__(
        self,
        job_id: str,
        container_ids: list[str],
        *,
        unbound: int = 0,
        cutover: bool = False,
        cutover_unbound: bool = False,
    ) -> None:
        self._job_id = job_id
        self._container_ids = container_ids
        self._cutover_unbound = cutover_unbound
        # Slots claimed but not yet bound to a container — the real registry's
        # state between `_register_run` and `_bind_run_container`, which for a
        # release spans the env resolve, the volume setup and `runtime.run()`.
        self._unbound = unbound
        # (P24b) Whether a health-gated cutover verify is in flight for this row.
        self._cutover = cutover
        self.cancelled_cutovers: list[str] = []

    def _mine(self, job_id: str) -> bool:
        return job_id == self._job_id

    def has_active_run(self, job_id: str) -> bool:
        return self._mine(job_id) and bool(self._container_ids or self._unbound)

    def active_run_container_ids_for(self, job_id: str) -> set[str]:
        return set(self._container_ids) if self._mine(job_id) else set()

    def has_unbound_run(self, job_id: str) -> bool:
        return self._mine(job_id) and self._unbound > 0

    def has_active_cutover(self, job_id: str) -> bool:
        return self._mine(job_id) and self._cutover

    def has_unbound_cutover(self, job_id: str) -> bool:
        return self._mine(job_id) and self._cutover_unbound

    async def cancel_cutover(self, job_id: str) -> None:
        self.cancelled_cutovers.append(job_id)
        self._cutover = False


def _purge_audit_params(q, action: str) -> dict:  # noqa: ANN001
    rows = [c for c in q.insert_audit_log.await_args_list if c.kwargs.get("action") == action]
    assert rows, f"no {action} audit row"
    return json.loads(rows[-1].kwargs["params_redacted"])


def test_delete_during_a_cutover_is_409_without_force():
    """(P24b WP5.5) An unforced delete under a live cutover verify refuses.

    Tearing the row (and its endpoint) out from under a verify task that is
    about to repoint a route and promote a container is exactly as unsafe as
    deleting mid-run.
    """
    q = _queries(_svc(), workloads=[])
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp")
    app.state.service_controller = _RunController("svc-a", [], cutover=True)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a", headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.cutover_in_progress"
    # DELETE is the one route with an escape hatch; its hint must say so.
    assert "?force=true" in resp.json()["hint"]
    q.delete_service_checked.assert_not_awaited()
    q.set_desired_state.assert_not_awaited()


def test_force_delete_cancels_the_cutover_and_proceeds():
    """``?force=true`` cancels the verify (killing the green) and deletes."""
    q = _queries(_svc(), workloads=[])
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp")
    controller = _RunController("svc-a", [], cutover=True)
    app.state.service_controller = controller
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true", headers=_auth())
    assert resp.status_code == 200
    assert controller.cancelled_cutovers == ["svc-a"]
    q.delete_service_checked.assert_awaited_once()


def test_force_delete_after_a_committed_cutover_tears_down_the_green():
    """(P24b WP5.5) The teardown acts on the FRESH row, not the resolved snapshot.

    A cutover that reached its commit point — before the guard above or under
    the cancel that followed it — promoted the green onto the row, and the
    cancel leaves it running by design. The snapshot resolved at the top of the
    route still names the blue that commit destroyed: tearing THAT down would
    leave the promoted green serving the repointed dial forever and would count
    a confirmed teardown while it still holds the data tree.
    """
    blue = _svc(container_id="c-blue")
    green = _svc(container_id="c-green")
    q = _queries(blue, workloads=[])
    runtime = FakeRuntime([])
    runtime.stop = AsyncMock()
    runtime.remove = AsyncMock()
    app = _app(runtime=runtime, queries=q, data_dir="/tmp")
    controller = _RunController("svc-a", [], cutover=True)
    orig_cancel = controller.cancel_cutover

    async def cancel(job_id):
        await orig_cancel(job_id)
        q.get_job.return_value = green  # the cancel found the promotion

    controller.cancel_cutover = cancel
    app.state.service_controller = controller
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true", headers=_auth())
    assert resp.status_code == 200
    runtime.stop.assert_awaited_once_with("c-green")
    runtime.remove.assert_awaited_once_with("c-green", force=True)


def test_delete_during_run_is_409_without_force():
    """An unforced delete under a live run/release refuses; nothing is touched."""
    q = _queries(_svc(), workloads=[])
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir="/tmp")
    app.state.service_controller = _RunController("svc-a", ["run-c1"])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a", headers=_auth())
    assert resp.status_code == 409
    assert resp.json()["code"] == "service.run_in_progress"
    assert "?force=true" in resp.json()["hint"]
    q.delete_service_checked.assert_not_awaited()
    q.set_desired_state.assert_not_awaited()


def test_force_delete_kills_active_run_and_proceeds(tmp_path):
    """``?force=true`` is the escape hatch: the run is killed, the delete lands.

    Without it a wedged migration pins the service undeletable until the daemon
    restarts. The kill count reaches the audit; the container id never does.
    """
    (tmp_path / "services" / "a").mkdir(parents=True)
    (tmp_path / "services" / "a" / "blob").write_bytes(b"x" * 8)

    q = _queries(_svc(), workloads=[])
    runtime = FakeRuntime([])
    runtime.kill = AsyncMock()
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    app.state.service_controller = _RunController("svc-a", ["run-c1"])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true&purge=data", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["data"] is True
    runtime.kill.assert_awaited_once_with("run-c1")
    q.delete_service_checked.assert_awaited_once()
    # The kill is confirmed, so the F12 gate stays open and the tree is gone.
    assert not (tmp_path / "services" / "a").exists()
    params = _purge_audit_params(q, "service.delete")
    assert params["killed_runs"] == 1
    assert "run-c1" not in json.dumps(params)


def test_force_delete_unkillable_run_skips_data_purge(tmp_path):
    """A kill that RAISED leaves a possible live writer → the rmtree is skipped (F12).

    The delete itself still proceeds (that is the point of ``?force``), but the
    data dir survives with an audited reason rather than being torn out from
    under a migration that may still be writing to it.
    """
    (tmp_path / "services" / "a").mkdir(parents=True)
    (tmp_path / "services" / "a" / "blob").write_bytes(b"x" * 8)

    q = _queries(_svc(), workloads=[])
    runtime = FakeRuntime([])
    runtime.kill = AsyncMock(side_effect=ContainerRuntimeError("docker is wedged"))
    runtime.container_running = AsyncMock(return_value=True)  # confirmed still writing
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    app.state.service_controller = _RunController("svc-a", ["run-c1"])
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true&purge=data", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["data"] is False
    runtime.container_running.assert_awaited_once_with("run-c1")
    assert (tmp_path / "services" / "a" / "blob").is_file()  # never rmtree'd
    assert _purge_audit_params(q, "service.purge_data") == {
        "key": "services/a",
        "purged": False,
        "reason": "run_alive",
    }
    # The failed kill is not counted as one.
    assert _purge_audit_params(q, "service.delete")["killed_runs"] == 0


def test_force_delete_with_an_unbound_run_slot_skips_data_purge(tmp_path):
    """The worse half of the same hazard: a slot claimed but not yet BOUND.

    A release claims its slot before arming the crash marker, resolving the
    launch env and calling ``runtime.run()`` — a window measured in seconds,
    and exactly the one a hung ``runtime.run`` sits in (one of the three
    reasons ``?force`` exists). There is no container id to kill, so the kill
    loop iterates zero times and, without this gate, ``skip_reason`` would stay
    None and the rmtree would run against a tree the migration container is
    about to bind-mount — Docker recreates it root-owned under a service that
    no longer exists, reclaimable only via ``nerdit gc``.
    """
    (tmp_path / "services" / "a").mkdir(parents=True)
    (tmp_path / "services" / "a" / "blob").write_bytes(b"x" * 8)

    q = _queries(_svc(), workloads=[])
    runtime = FakeRuntime([])
    runtime.kill = AsyncMock()
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    app.state.service_controller = _RunController("svc-a", [], unbound=1)
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true&purge=data", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["data"] is False
    runtime.kill.assert_not_awaited()  # nothing to kill — that is the problem
    assert (tmp_path / "services" / "a" / "blob").is_file()  # never rmtree'd
    assert _purge_audit_params(q, "service.purge_data") == {
        "key": "services/a",
        "purged": False,
        "reason": "run_unbound",
    }
    assert _purge_audit_params(q, "service.delete")["killed_runs"] == 0


def test_force_delete_with_an_unbound_cutover_green_skips_the_data_purge(tmp_path):
    """(PR #108) The cutover twin of the unbound-run gate.

    A verify still inside ``runtime.run()`` has no container id registered, and
    cancelling the task cannot stop the docker thread — the green may start
    (bind-mounting the data tree) after the cancel returned empty-handed. The
    data purge must fail closed exactly like the unbound-run case.
    """
    (tmp_path / "services" / "a").mkdir(parents=True)
    (tmp_path / "services" / "a" / "blob").write_bytes(b"x" * 8)

    q = _queries(_svc(), workloads=[])
    runtime = FakeRuntime([])
    runtime.kill = AsyncMock()
    app = _app(runtime=runtime, queries=q, data_dir=tmp_path)
    controller = _RunController("svc-a", [], cutover=True, cutover_unbound=True)
    app.state.service_controller = controller
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?force=true&purge=data", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["data"] is False
    assert controller.cancelled_cutovers == ["svc-a"]
    assert (tmp_path / "services" / "a" / "blob").is_file()  # never rmtree'd
    assert _purge_audit_params(q, "service.purge_data") == {
        "key": "services/a",
        "purged": False,
        "reason": "cutover_unbound",
    }


# --- P29 D-P29-8: the fourth ?purge member (workspace) -----------------------


def _seed_workspace(data_dir, name: str) -> None:  # noqa: ANN001
    """Write a minimal on-disk workspace (tree file + meta.json sidecar)."""
    root = data_dir / "workspaces" / name
    (root / "tree").mkdir(parents=True)
    (root / "tree" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "meta.json").write_text('{"owner_token_id": "tok-admin"}', encoding="utf-8")


def test_purge_workspace_member_removes_tree_and_meta(tmp_path):
    """``?purge=workspace`` removes the whole workspace root, sidecar included."""
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    body = client.delete("/services/svc-a?purge=workspace", headers=_auth()).json()
    assert body["purged"]["workspace"] is True
    # tree/ AND meta.json are gone — the whole root, not just the deployable half.
    assert not (tmp_path / "workspaces" / "a").exists()
    assert _purge_audit_params(q, "service.purge_workspace") == {
        "key": "workspaces/a",
        "purged": True,
    }


def test_purge_workspace_without_data_leaves_data_dir(tmp_path):
    """The members are independent: ``workspace`` alone never touches ``services/``."""
    _seed_workspace(tmp_path, "a")
    (tmp_path / "services" / "a").mkdir(parents=True)
    (tmp_path / "services" / "a" / "blob").write_bytes(b"x" * 8)

    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    body = client.delete("/services/svc-a?purge=workspace", headers=_auth()).json()
    assert body["purged"]["workspace"] is True
    assert body["purged"]["data"] is None  # not asked for
    assert (tmp_path / "services" / "a" / "blob").is_file()  # the volume survived
    assert not (tmp_path / "workspaces" / "a").exists()


def test_purge_workspace_not_asked_never_touches_the_tree(tmp_path):
    """Falsifier for the member: the default purge set leaves the workspace alone."""
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    body = client.delete("/services/svc-a", headers=_auth()).json()  # default: secrets
    assert body["purged"]["workspace"] is None
    assert (tmp_path / "workspaces" / "a" / "tree" / "main.py").is_file()
    assert not [
        c
        for c in q.insert_audit_log.await_args_list
        if c.kwargs.get("action") == "service.purge_workspace"
    ]


def test_purge_workspace_absent_tree_reports_false_not_500(tmp_path):
    """A row that never had a workspace: honest ``false``, never a 500."""
    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    resp = client.delete("/services/svc-a?purge=workspace", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["purged"]["workspace"] is False
    assert _purge_audit_params(q, "service.purge_workspace") == {
        "key": "workspaces/a",
        "purged": False,
    }


def test_invalid_purge_hint_lists_workspace(tmp_path):
    """The ``service.invalid_purge`` hint auto-derives from ``_PURGE_TARGETS``."""
    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    body = client.delete("/services/svc-a?purge=bogus", headers=_auth()).json()
    assert body["code"] == "service.invalid_purge"
    assert body["hint"] == "Valid purge targets: data, images, secrets, workspace."


async def test_purge_workspace_takes_no_lock_of_its_own(tmp_path):
    """(review round-1) The lock moved UP: the caller holds it across the span.

    ``_purge_workspace`` must not re-acquire it — ``asyncio.Lock`` is not
    reentrant, so it would deadlock inside :func:`delete_service`'s span. This
    call, made while the lock is held, must therefore complete.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    root = tmp_path / "workspaces" / "a"
    q = AsyncMock()
    q.insert_audit_log = AsyncMock()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(queries=q, settings=SimpleNamespace(data_dir=str(tmp_path)))
        ),
        state=SimpleNamespace(),
        headers={},
    )

    async with core_workspaces.workspace_lock("a"):
        assert (
            await asyncio.wait_for(services_mod._purge_workspace(request, _svc(), "a"), 5) is True
        )

    assert not root.exists()
    assert _purge_audit_params(q, "service.purge_workspace") == {
        "key": "workspaces/a",
        "purged": True,
    }
    core_workspaces._WORKSPACE_LOCKS.clear()


# --- the delete span holds the workspace lock (Codex 3803274903) -------------


def _ws_app(*, queries, data_dir) -> FastAPI:  # noqa: ANN001
    """``_app`` plus the workspace write route, so the race is drivable in-process."""
    app = _app(runtime=FakeRuntime([]), queries=queries, data_dir=data_dir)
    app.include_router(workspaces_router)
    return app


async def test_purge_workspace_locks_across_row_delete(tmp_path):
    """A write must not succeed in the window between the row delete and the rmtree.

    Unfixed, the lock was taken only for the ``rmtree``: a write arriving after
    ``delete_service_checked`` committed ran completely clean — the sidecar is
    still there so the owner gate passes, and ``get_service_by_name`` returns
    ``None`` so the row cross-check no-ops — returned 200, and was then
    destroyed by the purge. A successful write whose data silently vanished.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    in_delete = asyncio.Event()
    resume = asyncio.Event()

    async def _gated(job_id, checker, **_kw):  # noqa: ANN001
        in_delete.set()
        await resume.wait()
        return []

    q.delete_service_checked = AsyncMock(side_effect=_gated)
    app = _ws_app(queries=q, data_dir=tmp_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(
            client.delete("/services/svc-a?purge=workspace", headers=_auth())
        )
        await asyncio.wait_for(in_delete.wait(), 5)
        write = await asyncio.wait_for(
            client.put(
                "/workspaces/a/files",
                json={"files": {"late.py": "LATE\n"}},
                headers=_auth(),
            ),
            5,
        )
        assert write.status_code == 409, write.text
        assert write.json()["code"] == "workspace.deploy_in_progress"
        resume.set()
        deleted = await asyncio.wait_for(task, 5)

    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["purged"]["workspace"] is True
    # No survivor and — the loss shape this closes — no accepted-then-destroyed write.
    assert not (tmp_path / "workspaces" / "a").exists()
    core_workspaces._WORKSPACE_LOCKS.clear()


async def test_post_purge_write_is_fresh_first_write(tmp_path):
    """After the span, the same write is a legitimate fresh first write."""
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    app = _ws_app(queries=q, data_dir=tmp_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        deleted = await client.delete("/services/svc-a?purge=workspace", headers=_auth())
        assert deleted.json()["purged"]["workspace"] is True
        assert not (tmp_path / "workspaces" / "a").exists()

        resp = await client.put(
            "/workspaces/a/files", json={"files": {"new.py": "NEW\n"}}, headers=_auth_sub()
        )
    assert resp.status_code == 200, resp.text
    assert (tmp_path / "workspaces" / "a" / "tree" / "new.py").read_text() == "NEW\n"
    meta = core_workspaces.read_meta(tmp_path, "a") or {}
    assert meta["owner_token_id"] == "tok-sub"  # newly stamped, not the purged owner
    core_workspaces._WORKSPACE_LOCKS.clear()


async def test_refused_delete_releases_workspace_lock(tmp_path):
    """Every early raise inside the span unwinds through the AsyncExitStack."""
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(return_value=[{"id": "dep-1", "service": "dep"}])
    app = _ws_app(queries=q, data_dir=tmp_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        refused = await client.delete("/services/svc-a?purge=workspace", headers=_auth())
        assert refused.status_code == 409, refused.text
        assert core_workspaces.workspace_lock("a").locked() is False
        # The workspace survived the refusal, and is writable again.
        assert (tmp_path / "workspaces" / "a" / "tree" / "main.py").is_file()
        resp = await client.put(
            "/workspaces/a/files", json={"files": {"b.py": "B\n"}}, headers=_auth()
        )
    assert resp.status_code == 200, resp.text
    core_workspaces._WORKSPACE_LOCKS.clear()


async def test_purge_without_the_workspace_member_takes_no_lock(tmp_path):
    """A delete that does not purge the workspace never touches its lock.

    Falsifier for the conditional acquire: with the lock held by a sentinel the
    delete still completes — no interaction at all.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    app = _ws_app(queries=q, data_dir=tmp_path)

    async with (
        core_workspaces.workspace_lock("a"),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        resp = await asyncio.wait_for(
            client.delete("/services/svc-a?purge=secrets", headers=_auth()), 5
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["purged"]["workspace"] is None
    assert (tmp_path / "workspaces" / "a" / "tree" / "main.py").is_file()
    core_workspaces._WORKSPACE_LOCKS.clear()


async def test_cancelled_purge_rmtree_holds_lock(tmp_path, monkeypatch):
    """(Codex 3803274892, the purge site) A cancelled delete never abandons the rmtree.

    An abandoned ``rmtree`` would keep deleting with the lock already free — and
    a fresh first write would then land inside a tree still being removed.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    started = threading.Event()
    release = threading.Event()
    real_rmtree = shutil.rmtree

    def _blocking(path, *args, **kwargs):  # noqa: ANN001
        started.set()
        release.wait(5)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(services_mod.shutil, "rmtree", _blocking)
    q = _queries(_svc(), workloads=[])
    app = _ws_app(queries=q, data_dir=tmp_path)
    lock = core_workspaces.workspace_lock("a")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(
            client.delete("/services/svc-a?purge=workspace", headers=_auth())
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.1)
        assert lock.locked(), "the lock was freed while the rmtree worker was still running"
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 5)

    assert not lock.locked()
    assert not (tmp_path / "workspaces" / "a").exists()
    core_workspaces._WORKSPACE_LOCKS.clear()


# --- stale deletes never purge a replacement (review round-2, Codex 3803596881)


def test_absent_row_404s_and_purges_nothing(tmp_path):
    """A delete that removed no row must earn no name-based side effect.

    The stale-delete shape: A completed the same delete (row + workspace purge,
    lock released), the owner legitimately re-created the name, and B — stalled
    in the swallow-everything teardown — then acquired the free lock. Unfixed,
    ``delete_service_checked`` returned ``[]`` for an absent row, byte-identical
    to "this request deleted the row", so B ran every NAME-keyed effect (proxy
    deregister, secrets, data, images, workspace) against the REPLACEMENT while
    its brand-new row stayed alive. ``None`` is now the commit-point
    discriminator and the answer is the 404 the request would have got a moment
    later at the top-of-route resolve.
    """
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(return_value=None)
    app = _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path)
    proxy = AsyncMock()
    app.state.proxy_manager = proxy
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.delete("/services/svc-a?purge=workspace,secrets,data,images", headers=_auth())

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "not_found"
    # The replacement's state is untouched, top to bottom.
    assert (tmp_path / "workspaces" / "a" / "tree" / "main.py").is_file()
    proxy.deregister.assert_not_awaited()
    app.state.secret_manager.delete.assert_not_called()
    purge_rows = [
        c
        for c in q.insert_audit_log.await_args_list
        if str(c.kwargs.get("action", "")).startswith("service.purge_")
    ]
    assert purge_rows == []


def test_absent_row_restores_the_database_tombstone(tmp_path):
    """The database leg: the tombstone is taken BEFORE the commit point.

    A stale database delete renames the replacement's data tree aside before it
    can discover that it deleted nothing, so the absent-row branch must put it
    back before raising — otherwise the 404 is honest and the data is gone.
    """
    (tmp_path / "services" / "pg").mkdir(parents=True)
    (tmp_path / "services" / "pg" / "PG_VERSION").write_text("16\n", encoding="utf-8")
    q = _queries(_db_svc(), workloads=[])
    q.delete_service_checked = AsyncMock(return_value=None)
    runtime = FakeRuntime([])
    runtime.container_running = AsyncMock(return_value=False)
    client = TestClient(
        _app(runtime=runtime, queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )

    resp = client.delete("/services/db-a?purge=data", headers=_auth())

    assert resp.status_code == 404, resp.text
    assert resp.json()["code"] == "not_found"
    assert (tmp_path / "services" / "pg" / "PG_VERSION").read_text() == "16\n"
    assert [p.name for p in (tmp_path / "services").iterdir()] == ["pg"]


def test_deleted_row_still_purges(tmp_path):
    """Falsifier the other way: ``[]`` still means "this request deleted it"."""
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    client = TestClient(
        _app(runtime=FakeRuntime([]), queries=q, data_dir=tmp_path), raise_server_exceptions=False
    )
    resp = client.delete("/services/svc-a?purge=workspace", headers=_auth())
    assert resp.status_code == 200, resp.text
    assert resp.json()["purged"]["workspace"] is True
    assert not (tmp_path / "workspaces" / "a").exists()


# --- the post-commit tail settles through a cancel (Codex 3803596887) --------


async def test_cancelled_delete_still_runs_the_post_commit_purge(tmp_path):
    """A cancel after the row commits must not abandon the requested purge.

    The row is gone, so the client cannot retry: a 404 is all a retry can get,
    and the workspace tree would sit there until the retention sweep (default
    30 days; ``workspace_orphan_days = 0`` means never). Unfixed, a cancel
    landing anywhere in the tail — here on ``proxy.deregister``, the tail's
    FIRST await — unwound the ``AsyncExitStack`` and dropped the rest.
    """
    core_workspaces._WORKSPACE_LOCKS.clear()
    _seed_workspace(tmp_path, "a")
    q = _queries(_svc(), workloads=[])
    in_tail = asyncio.Event()
    resume = asyncio.Event()

    async def _gated_deregister(name):  # noqa: ANN001
        in_tail.set()
        await resume.wait()

    app = _ws_app(queries=q, data_dir=tmp_path)
    app.state.proxy_manager = SimpleNamespace(deregister=_gated_deregister)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(
            client.delete("/services/svc-a?purge=workspace", headers=_auth())
        )
        await asyncio.wait_for(in_tail.wait(), 5)
        task.cancel()
        await asyncio.sleep(0.1)
        assert core_workspaces.workspace_lock("a").locked(), "the lock was freed mid-tail"
        resume.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 5)

    assert task.cancelled() or isinstance(task.exception(), asyncio.CancelledError)
    # The requested purge ran anyway — that is the whole claim.
    assert not (tmp_path / "workspaces" / "a").exists()
    assert not core_workspaces.workspace_lock("a").locked()
    core_workspaces._WORKSPACE_LOCKS.clear()
