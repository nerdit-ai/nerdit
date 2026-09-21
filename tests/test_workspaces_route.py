"""Tests for the agent-workspace routes (P29 WP2).

Harness: the ``tests/test_deploy_git_route.py`` shape — the real workspace
router under the real auth/audit/idempotency middleware with ``AsyncMock``
queries and a ``tmp_path`` data dir, so the filesystem work is the real
``core.workspaces`` and only the DB is faked.

The interesting surface here is what the core module cannot see: the
meta-authoritative owner gate (D-P29-9), the not-found-before-forbidden
ordering, the deploy ingress's disposable scratch context (the R2 pin — the
live tree must never be handed to ``_finalize_deploy``), the 409 while the
snapshot holds the lock, and the two content-hygiene pins (audit rows carry
hashes and counts only; a 422 echo never carries file content).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import nerdit.daemon.routes.workspaces as workspaces_mod
from nerdit.config.defaults import WORKSPACE_MAX_BODY_BYTES, WORKSPACE_MAX_TOTAL_BYTES
from nerdit.core import workspaces as core_workspaces
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import QuotaExceeded, hash_token
from nerdit.daemon.bodylimit import BodyLimitMiddleware, _limit_for
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.workspaces import router as workspaces_router
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, Project, TokenRole
from nerdit.db.queries import ServiceNameClaimed

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
OTHER_RAW = "other-raw"
RO_RAW = "ro-raw"
ADMIN_RAW = "admin-raw"
SCOPED_RAW = "scoped-raw"

_TOKENS = {
    hash_token(SUB_RAW): ApiToken(
        id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW)
    ),
    hash_token(OTHER_RAW): ApiToken(
        id="tok-other", name="o", role=TokenRole.submitter, token_hash=hash_token(OTHER_RAW)
    ),
    hash_token(RO_RAW): ApiToken(
        id="tok-ro", name="r", role=TokenRole.readonly, token_hash=hash_token(RO_RAW)
    ),
    hash_token(ADMIN_RAW): ApiToken(
        id="tok-admin", name="a", role=TokenRole.admin, token_hash=hash_token(ADMIN_RAW)
    ),
    hash_token(SCOPED_RAW): ApiToken(
        id="tok-scoped",
        name="sc",
        role=TokenRole.submitter,
        token_hash=hash_token(SCOPED_RAW),
        scope_services=["other-app"],
    ),
}


@pytest.fixture(autouse=True)
def _clean_lock_registry():
    """The lock registry is daemon-lifetime by design; isolate it per test."""
    core_workspaces._WORKSPACE_LOCKS.clear()
    yield
    core_workspaces._WORKSPACE_LOCKS.clear()


def _queries(existing: Job | None = None) -> AsyncMock:
    q = AsyncMock()
    q.get_api_token_by_hash = AsyncMock(side_effect=lambda h: _TOKENS.get(h))
    q.insert_audit_log = AsyncMock()
    q.touch_api_token = AsyncMock()
    q.get_service_by_name = AsyncMock(return_value=existing)
    q.get_service_endpoint = AsyncMock(return_value=None)
    q.get_job_gpus = AsyncMock(return_value=[])
    q.reserve_service_for_token = AsyncMock(side_effect=lambda job, **kw: job)
    q.get_secret_claim = AsyncMock(return_value=None)
    # (P40b) No `projects` row unless a test plants one: a bare AsyncMock would
    # return a truthy MagicMock and read as a foreign project.
    q.get_project_by_name = AsyncMock(return_value=None)
    q.update_service_config = AsyncMock()
    return q


def _make_app(
    queries: AsyncMock,
    tmp_path,
    *,
    with_audit: bool = False,
    with_idempotency: bool = False,
    with_body_limit: bool = False,
) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(workspaces_router)
    if with_body_limit:
        # The bound is path-scoped on the real mount point, which is ``/api``
        # only — so the body-limit tests drive that prefix.
        api = APIRouter(prefix="/api")
        api.include_router(workspaces_router)
        app.include_router(api)
    app.state.queries = queries
    settings = MagicMock()
    settings.data_dir = str(tmp_path / "data")
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    app.state.settings = settings
    if with_idempotency:
        app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    if with_body_limit:
        # Added between Idempotency and Audit, so the runtime order matches the
        # daemon's (Auth → Audit → BodyLimit → Idempotency → router).
        app.add_middleware(BodyLimitMiddleware)
    if with_audit:
        app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)
    return app


def _client(queries: AsyncMock, tmp_path, **kw) -> TestClient:
    return TestClient(_make_app(queries, tmp_path, **kw), raise_server_exceptions=False)


def _auth(raw: str) -> dict:
    return {"Authorization": f"Bearer {raw}"}


def _write(client: TestClient, raw: str = SUB_RAW, *, name: str = "demo", **body):
    payload = {"files": {"main.py": "print(1)\n"}}
    payload.update(body)
    return client.put(f"/workspaces/{name}/files", json=payload, headers=_auth(raw))


def _svc(*, owner: str | None = "tok-sub", kind: JobKind = JobKind.service) -> Job:
    return Job(
        id="svc-1",
        service_name="demo",
        name="demo",
        kind=kind,
        gpu_count=0,
        status=JobStatus.running,
        submitted_by_token=owner,
        config=json.dumps({"image": "nerdit-app/demo:1", "build_version": 1, "max_version": 1}),
    )


def _meta(tmp_path, name: str = "demo") -> dict:
    return core_workspaces.read_meta(Path(tmp_path) / "data", name) or {}


def _tree(tmp_path, name: str = "demo") -> Path:
    return Path(tmp_path) / "data" / "workspaces" / name / "tree"


def _idem_store(q: AsyncMock) -> dict:
    """A principal-scoped in-memory idempotency store on the queries mock."""
    records: dict[tuple[str, str], SimpleNamespace] = {}

    async def _insert(*, principal_id, idem_key, method, path, expires_at, body_hash=None):
        if (principal_id, idem_key) in records:
            return False
        records[(principal_id, idem_key)] = SimpleNamespace(
            state="in_progress",
            method=method,
            path=path,
            response_status=None,
            response_body=None,
            content_type=None,
            resource_id=None,
            body_hash=body_hash,
        )
        return True

    async def _get(principal_id, idem_key):
        return records.get((principal_id, idem_key))

    async def _complete(
        *, principal_id, idem_key, response_status, response_body, content_type, resource_id
    ):
        rec = records[(principal_id, idem_key)]
        rec.state = "completed"
        rec.response_status = response_status
        rec.response_body = response_body
        rec.content_type = content_type
        rec.resource_id = resource_id

    async def _delete(principal_id, idem_key):
        records.pop((principal_id, idem_key), None)

    q.insert_idempotency_inprogress = AsyncMock(side_effect=_insert)
    q.get_idempotency_record = AsyncMock(side_effect=_get)
    q.complete_idempotency_record = AsyncMock(side_effect=_complete)
    q.delete_idempotency_record = AsyncMock(side_effect=_delete)
    return records


# --- authz --------------------------------------------------------------------


def test_write_requires_submitter_or_admin(tmp_path):
    """The coarse gate first: a readonly token never reaches the sidecar."""
    q = _queries()
    resp = _write(_client(q, tmp_path), RO_RAW)
    assert resp.status_code == 403
    assert not _tree(tmp_path).exists()


def test_first_write_stamps_caller_as_owner(tmp_path):
    resp = _write(_client(_queries(), tmp_path))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["written"] == 1
    assert body["file_count"] == 1
    assert _meta(tmp_path)["owner_token_id"] == "tok-sub"


def test_second_write_by_non_owner_submitter_403(tmp_path):
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    resp = _write(client, OTHER_RAW, files={"evil.py": "pwn\n"})
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    # The batch never landed.
    assert not (_tree(tmp_path) / "evil.py").exists()


def test_admin_bypasses_owner(tmp_path):
    """The other way round: the same second write from an admin lands."""
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    resp = _write(client, ADMIN_RAW, files={"admin.py": "ok\n"})
    assert resp.status_code == 200, resp.text
    assert (_tree(tmp_path) / "admin.py").exists()
    # The admin write does not steal ownership.
    assert _meta(tmp_path)["owner_token_id"] == "tok-sub"


def test_scoped_token_miss_403_names_scope(tmp_path):
    resp = _write(_client(_queries(), tmp_path), SCOPED_RAW)
    assert resp.status_code == 403
    body = resp.json()
    assert "demo" in body["message"]
    assert "other-app" in body["hint"]
    assert not _tree(tmp_path).exists()


def test_reads_are_owner_gated(tmp_path):
    """(D-P29-9) The recorded deviation from "reads stay role-only": workspace
    content is the caller's source code, so a role-only read would let any
    readonly token exfiltrate it."""
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200

    for raw, expected in ((OTHER_RAW, 403), (RO_RAW, 403), (SUB_RAW, 200), (ADMIN_RAW, 200)):
        listing = client.get("/workspaces/demo", headers=_auth(raw))
        assert listing.status_code == expected, (raw, listing.text)
        read = client.get("/workspaces/demo/files/main.py", headers=_auth(raw))
        assert read.status_code == expected, (raw, read.text)
        if expected == 200:
            assert read.text == "print(1)\n"
            assert read.headers["content-type"].startswith("text/plain")
            assert listing.json()["files"][0]["path"] == "main.py"


def test_row_owner_mismatch_403(tmp_path):
    """A service row owned by another token is refused even to the workspace's
    own owner — the two owners must agree (admin reconciles)."""
    q = _queries()
    client = _client(q, tmp_path)
    assert _write(client).status_code == 200
    q.get_service_by_name = AsyncMock(return_value=_svc(owner="tok-other"))

    denied = client.get("/workspaces/demo", headers=_auth(SUB_RAW))
    assert denied.status_code == 403
    assert denied.json()["code"] == "forbidden"
    assert client.get("/workspaces/demo", headers=_auth(ADMIN_RAW)).status_code == 200


def test_read_and_deploy_404_on_absent_workspace(tmp_path):
    """404 wins over the owner check: an absent workspace is not a secret, and
    the reverse ordering would make a 403 an existence oracle."""
    client = _client(_queries(), tmp_path)
    for resp in (
        client.get("/workspaces/demo", headers=_auth(SUB_RAW)),
        client.get("/workspaces/demo/files/main.py", headers=_auth(SUB_RAW)),
        client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW)),
    ):
        assert resp.status_code == 404, resp.text
        assert resp.json()["code"] == "workspace.not_found"

    # And a stranger gets the same 404, not a 403 that would confirm a name.
    assert client.get("/workspaces/demo", headers=_auth(OTHER_RAW)).status_code == 404


def test_reserved_name_refused(tmp_path):
    resp = _write(_client(_queries(), tmp_path), name="shared")
    assert resp.status_code == 422
    assert resp.json()["code"] == "service.reserved_name"


def test_invalid_workspace_name_is_a_path_error(tmp_path):
    resp = _write(_client(_queries(), tmp_path), name="UPPER")
    assert resp.status_code == 422
    assert resp.json()["code"] == "workspace.invalid_path"


# --- write semantics ----------------------------------------------------------


def test_write_error_codes_surface_with_limit_extra(tmp_path):
    """(R6/D-P29-5) One representative per cap: the envelope must carry the
    number in ``detail`` (REST) AND as a top-level ``limit`` (the only one that
    survives the MCP ``_call`` merge)."""
    client = _client(_queries(), tmp_path)
    resp = _write(client, files={"big.py": "x" * (256 * 1024 + 1)})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "workspace.file_too_large"
    assert body["detail"]["limit"] == 256 * 1024
    assert body["limit"] == 256 * 1024

    traversal = _write(client, files={"../escape.py": "x"})
    assert traversal.status_code == 422
    assert traversal.json()["code"] == "workspace.invalid_path"

    secret = _write(client, files={".env": "K=v"})
    assert secret.status_code == 422
    assert secret.json()["code"] == "workspace.secret_file"

    excluded = _write(client, files={"node_modules/x.js": "x"})
    assert excluded.status_code == 422
    assert excluded.json()["code"] == "workspace.excluded_path"


def test_write_batch_all_or_nothing_through_the_route(tmp_path):
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    before = (_tree(tmp_path) / "main.py").read_text()

    resp = _write(
        client,
        files={"main.py": "print(2)\n", "ok.py": "ok\n", "../escape.py": "pwn\n"},
    )
    assert resp.status_code == 422
    assert (_tree(tmp_path) / "main.py").read_text() == before
    assert not (_tree(tmp_path) / "ok.py").exists()


def test_write_idempotent_replay(tmp_path):
    q = _queries()
    _idem_store(q)
    client = _client(q, tmp_path, with_idempotency=True)
    headers = {**_auth(SUB_RAW), "Idempotency-Key": "ws-1"}
    payload = {"files": {"main.py": "print(1)\n"}}

    first = client.put("/workspaces/demo/files", json=payload, headers=headers)
    assert first.status_code == 200, first.text
    stamped = _meta(tmp_path)["last_written_at"]

    second = client.put("/workspaces/demo/files", json=payload, headers=headers)
    assert second.status_code == 200
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    # The route never ran a second time: the sidecar's write stamp is untouched.
    assert _meta(tmp_path)["last_written_at"] == stamped
    assert (_tree(tmp_path) / "main.py").read_text() == "print(1)\n"


def test_422_validation_echo_masks_files_field(tmp_path):
    """(R11) A pydantic 422 echoes the offending ``input``; for ``files`` that
    input IS the caller's source. Falsified by removing ``"files"`` from
    ``_SECRET_INPUT_FIELDS``."""
    client = _client(_queries(), tmp_path)
    resp = client.put(
        "/workspaces/demo/files",
        json={"files": {"main.py": ["SENTINEL-SOURCE-LINE"]}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 422
    assert "SENTINEL-SOURCE-LINE" not in resp.text
    # The diagnostic value survives: the caller still learns which key failed.
    assert "files" in resp.text


def test_audit_rows_never_carry_content(tmp_path):
    """(plan §2) ``workspace.write`` records names, hashes and counts — never a
    byte of content, in the success row or in a rejection's."""
    q = _queries()
    client = _client(q, tmp_path, with_audit=True)
    resp = _write(client, files={"main.py": "SENTINEL-SECRET-SOURCE\n"})
    assert resp.status_code == 200, resp.text

    dumped = json.dumps([c.kwargs for c in q.insert_audit_log.await_args_list])
    assert "SENTINEL-SECRET-SOURCE" not in dumped
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "workspace.write"
    assert kwargs["target_type"] == "workspace"
    assert kwargs["target_id"] == "demo"
    params = json.loads(kwargs["params_redacted"])
    assert set(params) == {"files", "deleted", "total_bytes", "sha256s"}
    assert params["files"] == 1
    assert len(params["sha256s"]) == 1

    # A REJECTED batch is audited too (the middleware records failures) — the
    # rejected content must not ride along either.
    q.insert_audit_log.reset_mock()
    bad = _write(client, files={"../escape.py": "SENTINEL-REJECTED-SOURCE\n"})
    assert bad.status_code == 422
    assert "SENTINEL-REJECTED-SOURCE" not in json.dumps(
        [c.kwargs for c in q.insert_audit_log.await_args_list]
    )


# --- deploy -------------------------------------------------------------------


def _finalize_spy(monkeypatch, result: dict | None = None) -> AsyncMock:
    spy = AsyncMock(return_value=result if result is not None else {"id": "svc-1", "name": "demo"})
    monkeypatch.setattr(workspaces_mod, "_finalize_deploy", spy)
    return spy


def test_deploy_calls_finalize_with_scratch_context_and_workspace_source(tmp_path, monkeypatch):
    """(R2) The live tree must NEVER be the build context: ``_finalize_deploy``
    rmtrees its context on failure and the off-tick builder rmtrees it on
    success, either of which would destroy the working copy (D-P29-6)."""
    spy = _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201, resp.text

    kwargs = spy.await_args.kwargs
    assert kwargs["source_meta"] == {"type": "workspace"}
    # ``context_root`` stays unset: the scratch dir IS the root, and stamping a
    # root would hand the controller a second tree to clean up.
    assert kwargs.get("context_root") is None
    context_dir = Path(spy.await_args.args[1])
    tree = _tree(tmp_path)
    assert context_dir.is_relative_to(Path(tmp_path) / "uploads")
    assert not context_dir.is_relative_to(tree)
    # The snapshot is a real, complete copy — and the workspace survives it.
    assert (context_dir / "main.py").read_text() == "print(1)\n"
    assert (tree / "main.py").read_text() == "print(1)\n"


def test_deploy_options_forwarded(tmp_path, monkeypatch):
    spy = _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200

    resp = client.post(
        "/workspaces/demo/deploy",
        json={"port": 8000, "env": {"KEEP": "v", "DROP": None}, "vendor": "nvidia"},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    kwargs = spy.await_args.kwargs
    assert kwargs["name"] == "demo"
    assert kwargs["port"] == 8000
    assert kwargs["env"] == {"KEEP": "v", "DROP": None}
    assert kwargs["vendor"] == "nvidia"
    assert kwargs["dry_run"] is False


def test_deploy_idempotent_replay(tmp_path, monkeypatch):
    """(plan §4) A same-key retry of the workspace deploy returns the replay
    envelope without re-entering the route — one build, not two."""
    spy = _finalize_spy(monkeypatch)
    q = _queries()
    _idem_store(q)
    client = _client(q, tmp_path, with_idempotency=True)
    assert _write(client).status_code == 200
    headers = {**_auth(SUB_RAW), "Idempotency-Key": "ws-deploy-1"}

    first = client.post("/workspaces/demo/deploy", json={}, headers=headers)
    assert first.status_code == 201, first.text

    second = client.post("/workspaces/demo/deploy", json={}, headers=headers)
    assert second.status_code == 201
    assert second.headers.get("Idempotent-Replay") == "true"
    assert second.json() == first.json()
    assert spy.await_count == 1


def test_deploy_dry_run_200_no_writes_claims_no_key(tmp_path, monkeypatch):
    """The dry-run trio: 200 with the plan, zero row writes, and no key claimed
    (the ``_DRY_RUN_ROUTES`` pin — a stray key must not poison the real deploy)."""
    spy = _finalize_spy(monkeypatch, {"dry_run": True, "action": "create"})
    q = _queries()
    records = _idem_store(q)
    client = _client(q, tmp_path, with_idempotency=True, with_audit=True)
    assert _write(client).status_code == 200

    resp = client.post(
        "/workspaces/demo/deploy",
        params={"dry_run": "true"},
        json={},
        headers={**_auth(SUB_RAW), "Idempotency-Key": "shared-key"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["dry_run"] is True
    assert spy.await_args.kwargs["dry_run"] is True
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config.assert_not_called()
    assert records == {}, "a dry run must not claim the key"
    assert q.insert_audit_log.await_args.kwargs["action"] == "deploy.workspace_plan"


def test_deploy_empty_tree_422_workspace_empty(tmp_path, monkeypatch):
    spy = _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    assert _write(client, files={}, delete=["main.py"]).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 422
    assert resp.json()["code"] == "workspace.empty"
    spy.assert_not_awaited()


def test_deploy_is_audited_as_workspace_create(tmp_path, monkeypatch):
    _finalize_spy(monkeypatch)
    q = _queries()
    client = _client(q, tmp_path, with_audit=True)
    assert _write(client).status_code == 200
    q.insert_audit_log.reset_mock()

    resp = client.post(
        "/workspaces/demo/deploy",
        json={"env": {"API_KEY": "super-secret"}},
        headers=_auth(SUB_RAW),
    )
    assert resp.status_code == 201, resp.text
    kwargs = q.insert_audit_log.await_args.kwargs
    assert kwargs["action"] == "deploy.workspace_create"
    assert kwargs["target_type"] == "service"
    assert kwargs["target_id"] == "demo"
    assert "super-secret" not in json.dumps(kwargs)  # env values masked


# --- concurrency (D-P29-10) ---------------------------------------------------


def test_write_409_while_deploy_zip_holds_lock(tmp_path, monkeypatch):
    """Deterministic by construction: hold the very lock the snapshot takes and
    assert both writers fail fast rather than queueing behind it."""
    _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200

    lock = core_workspaces.workspace_lock("demo")
    assert asyncio.run(lock.acquire())
    try:
        busy_write = _write(client, files={"other.py": "x\n"})
        assert busy_write.status_code == 409
        assert busy_write.json()["code"] == "workspace.deploy_in_progress"
        busy_deploy = client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
        assert busy_deploy.status_code == 409
        assert busy_deploy.json()["code"] == "workspace.deploy_in_progress"
    finally:
        lock.release()

    # Released: the same write goes through, so the 409 was the lock, not the batch.
    assert _write(client, files={"other.py": "x\n"}).status_code == 200


# --- first-write ownership (review cluster A) ---------------------------------


def test_first_write_against_foreign_row_forbidden(tmp_path):
    """(A1) A FIRST write (no sidecar yet) against a service row owned by
    somebody else must be refused.

    Otherwise any unscoped submitter claims a stranger's app workspace — even
    with an empty batch — and the ``_check_row_owner`` cross-check then 403s the
    real owner out of their own app until an admin reconciles the two owners.
    """
    q = _queries(_svc(owner="tok-other"))
    client = _client(q, tmp_path)

    resp = _write(client)
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "forbidden"
    assert not (Path(tmp_path) / "data" / "workspaces" / "demo" / "meta.json").exists()

    # The empty-batch variant is the cheap version of the same claim.
    empty = _write(client, files={})
    assert empty.status_code == 403, empty.text
    assert not (Path(tmp_path) / "data" / "workspaces" / "demo" / "meta.json").exists()

    # Falsifier both ways: the row's OWN owner still takes the first write, and
    # so does an admin.
    assert _write(client, OTHER_RAW).status_code == 200
    assert _meta(tmp_path)["owner_token_id"] == "tok-other"


def test_first_write_against_own_row_and_no_row_still_works(tmp_path):
    """The A1 gate must not break the two ordinary first writes."""
    assert _write(_client(_queries(), tmp_path)).status_code == 200
    assert _write(_client(_queries(_svc(owner="tok-sub")), tmp_path / "b")).status_code == 200


async def test_first_write_race_single_owner(tmp_path):
    """(A2) Two concurrent FIRST writers must produce exactly one owner.

    The loser is parked between its ``meta.json`` read and its write; with the
    read + owner decision outside the lock it resumes on a stale ``meta = None``
    and lands its whole batch inside the winner's freshly stamped workspace —
    D-P29-9 bypassed. A 403 or a 409 for the loser are both acceptable; two 2xx
    are not.
    """
    q = _queries()
    parked = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def _gated(name: str):
        seen.append(name)
        if len(seen) == 1:
            parked.set()
            await release.wait()
        return

    q.get_service_by_name = AsyncMock(side_effect=_gated)
    app = _make_app(q, tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(
            client.put(
                "/workspaces/demo/files",
                json={"files": {"b.py": "B\n"}},
                headers=_auth(OTHER_RAW),
            )
        )
        await asyncio.wait_for(parked.wait(), 5)
        second = await asyncio.wait_for(
            client.put(
                "/workspaces/demo/files",
                json={"files": {"a.py": "A\n"}},
                headers=_auth(SUB_RAW),
            ),
            5,
        )
        release.set()
        first_resp = await asyncio.wait_for(first, 5)

    winners = [r for r in (first_resp, second) if r.status_code == 200]
    losers = [r for r in (first_resp, second) if r.status_code != 200]
    assert len(winners) == 1, [first_resp.status_code, second.status_code]
    assert losers[0].status_code in (403, 409), losers[0].text

    first_won = first_resp.status_code == 200
    owner = "tok-other" if first_won else "tok-sub"
    kept, dropped = ("b.py", "a.py") if first_won else ("a.py", "b.py")
    assert _meta(tmp_path)["owner_token_id"] == owner
    assert (_tree(tmp_path) / kept).is_file()
    assert not (_tree(tmp_path) / dropped).exists()


# --- corrupt sidecar fails closed (review round-1, Codex 3803274894) ----------


def _corrupt_meta(tmp_path, name: str = "demo") -> None:
    (Path(tmp_path) / "data" / "workspaces" / name / "meta.json").write_text(
        "{invalid json", encoding="utf-8"
    )


def test_corrupt_meta_write_fails_closed(tmp_path):
    """A malformed sidecar must not read as "no workspace".

    Unfixed, ``read_meta`` returned ``None`` on a corrupt file, so the write
    route took the FIRST-write branch: with no service row any in-scope
    submitter re-stamped ``owner_token_id`` and walked off with the existing
    source tree. Now it is a 500 ``workspace.meta_corrupt`` and nothing moves.
    """
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    _corrupt_meta(tmp_path)

    resp = _write(client, OTHER_RAW, files={"evil.py": "pwn\n"})
    assert resp.status_code == 500, resp.text
    assert resp.json()["code"] == "workspace.meta_corrupt"
    # The tree is untouched and the sidecar was NOT re-stamped.
    assert not (_tree(tmp_path) / "evil.py").exists()
    assert (_tree(tmp_path) / "main.py").read_text() == "print(1)\n"
    assert (
        Path(tmp_path) / "data" / "workspaces" / "demo" / "meta.json"
    ).read_text() == "{invalid json"

    # Not even the real owner, and not even an admin: a corrupt ownership record
    # grants nothing to anyone (recovery is operator-manual, by decision).
    assert _write(client).status_code == 500
    assert _write(client, ADMIN_RAW).status_code == 500


def test_corrupt_meta_read_and_deploy_fail_closed(tmp_path, monkeypatch):
    """Same 500 on both reads and the deploy — explicitly NOT a 404.

    A 404 here would send the caller back through the first-write path, which is
    exactly the hole this closes; corrupt is not absent.
    """
    spy = _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    _corrupt_meta(tmp_path)

    for resp in (
        client.get("/workspaces/demo", headers=_auth(SUB_RAW)),
        client.get("/workspaces/demo/files/main.py", headers=_auth(SUB_RAW)),
        client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW)),
    ):
        assert resp.status_code == 500, resp.text
        assert resp.json()["code"] == "workspace.meta_corrupt"
    spy.assert_not_awaited()


def test_corrupt_meta_error_is_path_free(tmp_path):
    """The envelope names no filesystem path (the P29 message posture)."""
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    _corrupt_meta(tmp_path)
    resp = _write(client)
    assert resp.status_code == 500
    assert str(tmp_path) not in resp.text
    assert "meta.json" in resp.json()["hint"]


# --- fresh deploy ownership (review round-1, Codex 3803274898) ----------------


def _write_app(client: TestClient, raw: str = SUB_RAW, *, name: str = "demo"):
    """A batch that is a real build context (the buildpack needs a marker file)."""
    return client.put(
        f"/workspaces/{name}/files",
        json={"files": {"main.py": "print(1)\n", "requirements.txt": "fastapi\n"}},
        headers=_auth(raw),
    )


def test_admin_fresh_deploy_preserves_workspace_owner(tmp_path):
    """An admin's FIRST deploy of a submitter's workspace must not orphan it.

    Unfixed, ``write_fresh`` stamped the acting principal, so the row came out
    admin-owned while ``meta.json`` stayed the submitter's — and every later call
    by the real owner passed the sidecar gate then 403'd at ``_check_row_owner``:
    a permanent lockout only another admin could undo.
    """
    q = _queries()
    client = _client(q, tmp_path)
    assert _write_app(client).status_code == 200  # submitter S owns the workspace

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.submitted_by_token == "tok-sub"

    # And S is not locked out of their own app afterwards.
    q.get_service_by_name = AsyncMock(return_value=job)
    assert _write_app(client).status_code == 200
    assert client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW)).status_code in (
        200,
        201,
    )


def test_admin_fresh_deploy_of_admin_workspace_unchanged(tmp_path):
    """Regression pin: an admin's OWN workspace still yields an admin-owned row."""
    q = _queries()
    client = _client(q, tmp_path)
    assert _write_app(client, ADMIN_RAW).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201, resp.text
    assert q.reserve_service_for_token.call_args.args[0].submitted_by_token == "tok-admin"


def test_null_owner_workspace_falls_back_to_the_acting_principal(tmp_path):
    """A NULL sidecar owner (only admins pass that gate) keeps today's behaviour."""
    q = _queries()
    client = _client(q, tmp_path)
    assert _write_app(client, LEGACY).status_code == 200  # legacy admin ⇒ token_id None
    assert _meta(tmp_path)["owner_token_id"] is None

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201, resp.text
    assert q.reserve_service_for_token.call_args.args[0].submitted_by_token == "tok-admin"


def test_quota_is_charged_to_the_workspace_owner(tmp_path):
    """The no-laundering property: an admin cannot spend their own quota for S.

    The redeploy path already charges the service OWNER, not the actor; the
    fresh workspace deploy now matches.
    """
    q = _queries()
    charged: list[str | None] = []

    async def _reserve(job, **_kw):
        charged.append(job.submitted_by_token)
        raise QuotaExceeded("max_concurrent_jobs", limit=1, current=1)

    q.reserve_service_for_token = AsyncMock(side_effect=_reserve)
    client = _client(q, tmp_path)
    assert _write_app(client).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "quota_exceeded"
    assert charged == ["tok-sub"]


def test_finalize_receives_the_sidecar_owner(tmp_path, monkeypatch):
    """The seam itself: the route hands ``_finalize_deploy`` the sidecar owner."""
    spy = _finalize_spy(monkeypatch)
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200
    assert client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW)).status_code
    assert spy.await_args.kwargs["owner_token_id"] == "tok-sub"


# --- cancellation never frees the lock (review round-1, Codex 3803274892) -----


async def test_cancelled_write_holds_lock_until_worker_settles(tmp_path, monkeypatch):
    """A cancelled write must not release the lock with the worker still writing.

    Unfixed, ``asyncio.to_thread`` returned control the moment the cancel landed
    and ``async with lock`` unwound — so a retry entered ``write_files``
    concurrently against the same ``<name>.tmp-<pid>`` staging name (same daemon
    pid) and the two writers unlinked/replaced each other's staging file.
    """
    real = core_workspaces.write_files
    started = threading.Event()
    release = threading.Event()

    def _blocking(*args, **kwargs):
        started.set()
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(core_workspaces, "write_files", _blocking)
    app = _make_app(_queries(), tmp_path)
    lock = core_workspaces.workspace_lock("demo")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        task = asyncio.create_task(
            client.put(
                "/workspaces/demo/files",
                json={"files": {"main.py": "print(1)\n"}},
                headers=_auth(SUB_RAW),
            )
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.1)

        assert lock.locked(), "the lock was freed while the worker thread was still writing"
        assert not (_tree(tmp_path) / "main.py").exists()

        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 5)

    assert not lock.locked()
    # The cancelled batch landed exactly once, whole — never half-applied.
    assert (_tree(tmp_path) / "main.py").read_text() == "print(1)\n"
    assert _meta(tmp_path)["owner_token_id"] == "tok-sub"


async def test_cancelled_deploy_snapshot_holds_lock_until_worker_settles(tmp_path, monkeypatch):
    """Same contract on the read-only zip: the lock is uniform, no exceptions."""
    real = core_workspaces.zip_workspace
    started = threading.Event()
    release = threading.Event()

    def _blocking(tree):
        started.set()
        release.wait(5)
        return real(tree)

    client_app = _make_app(_queries(), tmp_path)
    with TestClient(client_app) as sync_client:
        assert _write(sync_client).status_code == 200
    monkeypatch.setattr(core_workspaces, "zip_workspace", _blocking)
    lock = core_workspaces.workspace_lock("demo")

    async with AsyncClient(
        transport=ASGITransport(app=client_app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.1)
        assert lock.locked(), "the lock was freed while the zip worker was still running"
        release.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 5)

    assert not lock.locked()


# --- pre-parse body bound (review round-1, Codex 3803274889) ------------------


async def _raw_request(app, method: str, path: str, *, headers: dict[str, str], receive):
    """Drive the ASGI app with a hand-built scope.

    Necessary because httpx recomputes ``Content-Length`` from the bytes it
    actually sends — and the whole point of this bound is that the DECLARED
    length is refused before a byte of body is read.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("127.0.0.1", 51234),
        "server": ("testserver", 80),
    }
    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(body) if body else None


def _never_read():
    """A receive callable that fails the test if the body is ever requested."""

    async def _receive():
        raise AssertionError("the request body was read despite the pre-parse bound")

    return _receive


def _body_receive(payload: bytes):
    async def _receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    return _receive


async def test_write_body_over_limit_413(tmp_path):
    """An over-declared body is refused BEFORE FastAPI materializes it."""
    app = _make_app(_queries(), tmp_path, with_body_limit=True)
    status, body = await _raw_request(
        app,
        "PUT",
        "/api/workspaces/demo/files",
        headers={
            "authorization": f"Bearer {SUB_RAW}",
            "content-type": "application/json",
            "content-length": str(WORKSPACE_MAX_BODY_BYTES + 1),
        },
        receive=_never_read(),
    )
    assert status == 413, body
    assert body["code"] == "payload_too_large"
    assert body["limit"] == WORKSPACE_MAX_BODY_BYTES
    assert body["detail"]["limit"] == WORKSPACE_MAX_BODY_BYTES
    assert str(tmp_path) not in json.dumps(body)
    assert not _tree(tmp_path).exists()


async def test_write_body_at_limit_passes(tmp_path):
    """Boundary the other way: the middleware trusts the header and steps aside.

    The real cap work stays in-route (the D-P29-5 post-state caps); this bound
    only refuses what cannot possibly be a legal batch.
    """
    app = _make_app(_queries(), tmp_path, with_body_limit=True)
    payload = json.dumps({"files": {"main.py": "print(1)\n"}}).encode()
    status, body = await _raw_request(
        app,
        "PUT",
        "/api/workspaces/demo/files",
        headers={
            "authorization": f"Bearer {SUB_RAW}",
            "content-type": "application/json",
            "content-length": str(WORKSPACE_MAX_BODY_BYTES),
        },
        receive=_body_receive(payload),
    )
    assert status == 200, body
    assert body["written"] == 1


async def test_write_chunked_411(tmp_path):
    """No ``Content-Length`` (a chunked body) is refused, never capped-while-read."""
    app = _make_app(_queries(), tmp_path, with_body_limit=True)
    status, body = await _raw_request(
        app,
        "PUT",
        "/api/workspaces/demo/files",
        headers={
            "authorization": f"Bearer {SUB_RAW}",
            "content-type": "application/json",
            "transfer-encoding": "chunked",
        },
        receive=_never_read(),
    )
    assert status == 411, body
    assert body["code"] == "length_required"
    assert not _tree(tmp_path).exists()


async def test_unparseable_content_length_is_treated_as_absent(tmp_path):
    app = _make_app(_queries(), tmp_path, with_body_limit=True)
    status, body = await _raw_request(
        app,
        "PUT",
        "/api/workspaces/demo/files",
        headers={
            "authorization": f"Bearer {SUB_RAW}",
            "content-type": "application/json",
            "content-length": "not-a-number",
        },
        receive=_never_read(),
    )
    assert status == 411, body
    assert body["code"] == "length_required"


@pytest.mark.parametrize(
    ("method", "path", "bounded"),
    [
        ("PUT", "/api/workspaces/demo/files", True),
        ("PUT", "/api/workspaces/a-b-c9/files", True),
        ("POST", "/api/workspaces/demo/deploy", False),
        ("PUT", "/api/workspaces/demo/files/main.py", False),  # the path:path read
        ("GET", "/api/workspaces/demo/files", False),
        ("PUT", "/api/workspaces/demo/files/", False),
        ("POST", "/api/deploy", False),
        ("PUT", "/workspaces/demo/files", False),  # no bare-root mount to bound
    ],
)
def test_body_limit_matches_only_the_write_route(method, path, bounded):
    assert (_limit_for(method, path) == WORKSPACE_MAX_BODY_BYTES) is bounded


async def test_other_routes_are_unbounded(tmp_path, monkeypatch):
    """A chunked body on a neighbouring workspace route is not the bound's business."""
    _finalize_spy(monkeypatch)
    app = _make_app(_queries(), tmp_path, with_body_limit=True)
    with TestClient(app) as sync_client:
        assert _write(sync_client).status_code == 200

    status, _ = await _raw_request(
        app,
        "POST",
        "/api/workspaces/demo/deploy",
        headers={"authorization": f"Bearer {SUB_RAW}", "content-type": "application/json"},
        receive=_body_receive(b"{}"),
    )
    assert status == 201


def test_body_limit_derivation_covers_the_worst_legal_batch():
    """The constant must not be able to refuse a batch the caps would accept.

    Worst legitimate JSON inflation of a full 10 MiB workspace is ``ensure_ascii``
    escaping — x3 for astral-plane text — plus keys and structure.
    """
    assert WORKSPACE_MAX_BODY_BYTES == 3 * WORKSPACE_MAX_TOTAL_BYTES + 1_048_576
    assert WORKSPACE_MAX_BODY_BYTES > 3 * WORKSPACE_MAX_TOTAL_BYTES


# --- reads inside the lock (review round-2, Codex 3803596876) ----------------


def _read_client(tmp_path, q: AsyncMock | None = None) -> tuple[TestClient, AsyncMock]:
    """A client with ``demo`` already written by its owner."""
    queries = q or _queries()
    client = _client(queries, tmp_path)
    assert _write(client).status_code == 200
    return client, queries


@pytest.mark.parametrize(
    ("func_name", "url"),
    [("list_files", "/workspaces/demo"), ("read_file", "/workspaces/demo/files/main.py")],
)
def test_read_holds_the_workspace_lock_across_authz_and_io(tmp_path, monkeypatch, func_name, url):
    """The whole point of the fix: the filesystem read runs UNDER the lock.

    Unfixed, ``_load_owned_workspace`` authorized with no lock held and the read
    ran after several await points — so an admin ``DELETE …?purge=workspace``
    (which does take the lock) plus another owner's first write could both land
    in the gap and hand an already-authorized caller the next generation's
    source. Asserting from inside the delegate is the deterministic form of that
    claim: the lock is either held when the bytes are read, or it is not.
    """
    client, _ = _read_client(tmp_path)
    real = getattr(core_workspaces, func_name)
    seen: list[bool] = []

    def _delegate(*args, **kwargs):
        seen.append(core_workspaces.workspace_lock("demo").locked())
        return real(*args, **kwargs)

    monkeypatch.setattr(core_workspaces, func_name, _delegate)
    resp = client.get(url, headers=_auth(SUB_RAW))
    assert resp.status_code == 200, resp.text
    assert seen == [True]


@pytest.mark.parametrize("url", ["/workspaces/demo", "/workspaces/demo/files/main.py"])
async def test_read_waits_for_the_lock_and_never_409s(tmp_path, url):
    """Reads WAIT on the lock (the delete's posture), they never fail fast.

    Unfixed, the GET completed while the lock was held — which is exactly the
    hole. Fixed, it parks until the holder releases and then returns 200; the
    409 ``workspace.deploy_in_progress`` stays a write-path answer.
    """
    q = _queries()
    app = _make_app(q, tmp_path)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.put(
            "/workspaces/demo/files",
            json={"files": {"main.py": "print(1)\n"}},
            headers=_auth(SUB_RAW),
        )
        assert first.status_code == 200, first.text

        lock = core_workspaces.workspace_lock("demo")
        await lock.acquire()
        try:
            task = asyncio.create_task(client.get(url, headers=_auth(SUB_RAW)))
            # Generous by design: unfixed, the whole request (sidecar read, row
            # read, thread-pool listing/read) completed inside this window.
            await asyncio.sleep(0.3)
            assert not task.done(), "the read did not wait for the lock"
        finally:
            lock.release()
        resp = await asyncio.wait_for(task, 5)
    assert resp.status_code == 200, resp.text


def test_read_dot_segment_path_is_422_server_side(tmp_path):
    """Server-side half of the WS-R2-6 pin: Starlette decodes but never collapses.

    ``%2E%2E`` is exactly the raw path the fixed client puts on the wire for
    ``foo/../main.py`` — httpx's RFC 3986 normalization leaves it alone — so the
    daemon really does see the dot segment and answers with the promised
    structured 422 rather than a 200 carrying a different file. (The literal
    ``foo/../main.py`` form is unassertable from httpx: the client library
    collapses it before it is ever sent, which IS the finding.)
    """
    client, _ = _read_client(tmp_path)
    resp = client.get("/workspaces/demo/files/foo/%2E%2E/main.py", headers=_auth(SUB_RAW))
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "workspace.invalid_path"


def test_workspace_deploy_response_carries_summary_and_hints(tmp_path, monkeypatch):
    """(Agent-DX) The workspace ingress returns ``_finalize_deploy``'s dict
    VERBATIM — the route declares no ``response_model``, so the additive
    ``summary``/``hints`` the shared tail assembles reach the wire unfiltered.
    (The tail's own derivation is pinned in ``tests/test_deploy_route.py``.)"""
    _finalize_spy(
        monkeypatch,
        result={
            "id": "svc-1",
            "name": "demo",
            "status": "building",
            "summary": {"app": "demo", "status": "building", "version": 1, "public_url": None},
            "hints": ["a hint"],
        },
    )
    client = _client(_queries(), tmp_path)
    assert _write(client).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["summary"] == {
        "app": "demo",
        "status": "building",
        "version": 1,
        "public_url": None,
    }
    assert body["hints"] == ["a hint"]


# --- P39: the claim is matched against the sidecar owner ----------------------


def test_admin_fresh_deploy_of_a_submitter_workspace_is_judged_as_the_owner(tmp_path):
    """The row goes to the sidecar owner AND is judged as that owner: an admin
    deploying S's workspace must not plant S's row in another token's project
    (or over another token's claim), where it would launch with their variables."""
    q = _queries()
    client = _client(q, tmp_path)
    assert _write_app(client).status_code == 200  # S owns the workspace

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201, resp.text
    job = q.reserve_service_for_token.call_args.args[0]
    assert job.submitted_by_token == "tok-sub"
    assert q.reserve_service_for_token.call_args.kwargs == {"admin": False}


def test_admin_fresh_deploy_of_its_own_workspace_keeps_the_bypass(tmp_path):
    q = _queries()
    client = _client(q, tmp_path)
    assert _write_app(client, raw=ADMIN_RAW).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(ADMIN_RAW))
    assert resp.status_code == 201, resp.text
    assert q.reserve_service_for_token.call_args.kwargs == {"admin": True}


def test_first_write_onto_a_foreign_rowless_project_is_refused(tmp_path):
    """A project with no rows yet (variables set first) still belongs to its owner."""
    q = _queries()
    q.get_project_by_name = AsyncMock(
        return_value=Project(id="prj_" + "a" * 16, name="demo", submitted_by_token="tok-other")
    )
    client = _client(q, tmp_path)
    assert _write_app(client).status_code == 403
    assert _write_app(client, raw=ADMIN_RAW).status_code == 200


def test_workspace_deploy_over_a_foreign_claim_409(tmp_path):
    """The transaction refuses a claim the sidecar owner does not hold."""
    q = _queries()
    q.reserve_service_for_token = AsyncMock(side_effect=ServiceNameClaimed("demo"))
    client = _client(q, tmp_path)
    assert _write_app(client).status_code == 200

    resp = client.post("/workspaces/demo/deploy", json={}, headers=_auth(SUB_RAW))
    assert resp.status_code == 409, resp.text
    assert resp.json()["code"] == "service.name_claimed"
    assert q.reserve_service_for_token.call_args.kwargs == {"admin": False}
