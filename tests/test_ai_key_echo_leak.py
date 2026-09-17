"""Keep rejected literal AI keys out of deploy, git-deploy and config responses.

Use real middleware and an in-memory database to sweep every 422 envelope field
and persisted audit params. Error audit rows must still exist, and api_key must
remain in diagnostics as a positive control. Model-layer validation is covered
separately in test_ai_binding_schema.py.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import nerdit.daemon.routes.deploy as deploy_mod
from nerdit.core.gitsource import GitSourceInfo
from nerdit.daemon.audit import AuditMiddleware
from nerdit.daemon.auth import hash_token
from nerdit.daemon.errors import RequestIdMiddleware, register_error_handlers
from nerdit.daemon.idempotency import IdempotencyMiddleware
from nerdit.daemon.middleware import ScopedTokenAuthMiddleware
from nerdit.daemon.routes.app_config import router as app_config_router
from nerdit.daemon.routes.deploy import router as deploy_router
from nerdit.db.database import Database
from nerdit.db.models import ApiToken, Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import Queries

# The same canary string the unit leg uses — shaped like a real provider key so
# a partial redaction still reads as a failure.
CANARY = "sk-live-CANARY-0000"

LEGACY = "legacy-global"
SUB_RAW = "sub-raw"
REPO = "https://github.com/owner/repo"

_NERDIT_TOML = (
    "[ai.cheap]\n"
    'provider = "api"\n'
    'model = "gpt-4o-mini"\n'
    'base_url = "https://api.openai.com/v1"\n'
    f'api_key = "{CANARY}"\n'
)


def assert_no_canary(text: str | None) -> None:
    """The shared leak assertion for this file: the literal key never appears.

    Applied to whole response bodies and to every audit param blob — a
    substring check, so a partially-quoted or JSON-escaped echo still fails.
    """
    if text is None:
        return
    assert CANARY not in text, f"the rejected api_key leaked: {text[:400]}"


# --- harness -----------------------------------------------------------------


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _node_zip_with_literal_key() -> bytes:
    return _zip(
        {
            "package.json": json.dumps({"name": "demo", "scripts": {"start": "node index.js"}}),
            "index.js": "console.log('hi')",
            "nerdit.toml": _NERDIT_TOML,
        }
    )


def _fake_clone() -> AsyncMock:
    """A clone stub that materializes a Node context dir carrying the canary."""

    async def _run(repo_url, *, dest_dir, ref=None, subdir=None, **_kw) -> GitSourceInfo:
        ctx = Path(dest_dir) / subdir if subdir else Path(dest_dir)
        ctx.mkdir(parents=True, exist_ok=True)
        (ctx / "package.json").write_text(
            json.dumps({"name": "demo", "scripts": {"start": "node index.js"}})
        )
        (ctx / "index.js").write_text("console.log('hi')")
        (ctx / "nerdit.toml").write_text(_NERDIT_TOML)
        return GitSourceInfo(commit_sha="a" * 40, resolved_ref=ref or "main", context_dir=ctx)

    return AsyncMock(side_effect=_run)


def _app_job() -> Job:
    return Job(
        id="job-demo-0001",
        kind=JobKind.service,
        service_name="demo",
        name="demo",
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        health_check={"path": "/health"},
        config=json.dumps(
            {
                "image": "nerdit-app/demo:1",
                "build_version": 1,
                "port": 8000,
                "command": "npm start",
                "config_source": "deploy",
                "config_revision": 1,
            }
        ),
        submitted_by_token="tok-sub",
    )


@pytest_asyncio.fixture
async def harness(tmp_path):
    db = Database(":memory:")
    await db.connect()
    await db.init_schema()
    queries = Queries(db)
    await queries.create_api_token(
        ApiToken(id="tok-sub", name="s", role=TokenRole.submitter, token_hash=hash_token(SUB_RAW))
    )

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(deploy_router)
    app.include_router(app_config_router)
    app.state.queries = queries
    settings = MagicMock()
    settings.daemon.max_upload_bytes = 10 * 1024 * 1024
    settings.daemon.upload_dir = str(tmp_path / "uploads")
    settings.git.enabled = True
    settings.git.allowed_hosts = ["github.com"]
    settings.git.clone_timeout_s = 120
    settings.git.max_clone_bytes = 10 * 1024 * 1024
    app.state.settings = settings
    app.state.secret_manager = MagicMock(load=MagicMock(return_value={}))
    # The full middleware trio, in the daemon's order (Idempotency → Audit →
    # Auth → RequestId): the audit leg is only meaningful with the real one.
    app.add_middleware(IdempotencyMiddleware, get_queries=lambda: queries)
    app.add_middleware(AuditMiddleware, get_queries=lambda: queries, get_event_bus=lambda: None)
    app.add_middleware(ScopedTokenAuthMiddleware, token=LEGACY, get_queries=lambda: queries)
    app.add_middleware(RequestIdMiddleware)

    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        yield client, queries
    finally:
        await client.aclose()
        await db.close()


def _auth(raw: str = SUB_RAW) -> dict:
    return {"Authorization": f"Bearer {raw}"}


async def _assert_audit_is_clean_and_recorded(queries: Queries) -> None:
    """Every audit row is canary-free, and the failed request IS recorded.

    The second half is the load-bearing one: it pins that ``AuditMiddleware``
    records the request's own (redacted) params + status and never the response
    text, so closing the 422 echo cannot be silently undone by an audit row.
    """
    rows, _ = await queries.list_audit_log(limit=200)
    for row in rows:
        params = row.params_redacted
        assert_no_canary(params if isinstance(params, str) else json.dumps(params))
    assert any(row.result == "error" for row in rows), "the failed request was not audited"


def _assert_envelope(payload: dict, code: str) -> None:
    """Every envelope field is canary-free; the ``api_key`` loc still survives."""
    assert payload["code"] == code
    for field in ("message", "hint", "detail", "request_id"):
        value = payload.get(field)
        assert_no_canary(value if isinstance(value, str) else json.dumps(value))
    # Positive control: the diagnostic that makes the error actionable.
    assert "api_key" in payload["message"]


# --- the three ingresses ------------------------------------------------------


@pytest.mark.asyncio
async def test_deploy_zip_422_never_echoes_the_literal_api_key(harness):
    client, queries = harness
    resp = await client.post(
        "/deploy",
        files={"archive": ("app.zip", _node_zip_with_literal_key(), "application/zip")},
        data={"name": "demo", "port": "8000", "gpus": "0"},
        headers=_auth(),
    )
    assert resp.status_code == 422, resp.text
    assert_no_canary(resp.text)
    _assert_envelope(resp.json(), "deploy.invalid_ai")
    # The source survives — the operator still knows WHERE to look.
    assert "nerdit.toml" in resp.json()["message"]
    await _assert_audit_is_clean_and_recorded(queries)


@pytest.mark.asyncio
async def test_deploy_git_422_never_echoes_the_literal_api_key(harness, monkeypatch):
    client, queries = harness
    monkeypatch.setattr(deploy_mod, "clone_source", _fake_clone())
    resp = await client.post(
        "/deploy/git",
        json={"repo_url": REPO, "name": "demo"},
        headers=_auth(),
    )
    assert resp.status_code == 422, resp.text
    assert_no_canary(resp.text)
    _assert_envelope(resp.json(), "deploy.invalid_ai")
    await _assert_audit_is_clean_and_recorded(queries)


@pytest.mark.asyncio
async def test_put_app_config_ai_422_never_echoes_the_literal_api_key(harness):
    client, queries = harness
    await queries.create_job(_app_job())
    resp = await client.put(
        "/config/apps/demo/ai",
        json={
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "api_key": CANARY,
            }
        },
        headers={**_auth(), "Idempotency-Key": "K-ai-echo"},
    )
    assert resp.status_code == 422, resp.text
    assert_no_canary(resp.text)
    _assert_envelope(resp.json(), "deploy.invalid_ai")
    await _assert_audit_is_clean_and_recorded(queries)


@pytest.mark.asyncio
async def test_put_app_config_ai_422_leaves_no_cached_body_to_replay(harness):
    """No idempotency row survives the 422, so the echo cannot be re-served.

    The middleware releases a claim on any non-2xx (`_finish`), so the canary
    never reaches ``idempotency_keys.response_body`` — pinned both ways: the
    stored row is gone, and a retry with the SAME key re-runs the validator
    (no ``Idempotent-Replay``) and answers canary-free again.
    """
    client, queries = harness
    await queries.create_job(_app_job())
    headers = {**_auth(), "Idempotency-Key": "K-ai-echo-2"}
    body = {
        "cheap": {
            "provider": "api",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": CANARY,
        }
    }
    first = await client.put("/config/apps/demo/ai", json=body, headers=headers)
    assert first.status_code == 422

    cur = await queries._db.conn.execute(
        "SELECT response_body FROM idempotency_keys WHERE idem_key = ?", ("K-ai-echo-2",)
    )
    rows = await cur.fetchall()
    assert rows == [], "a failed write pinned an idempotency record"

    retry = await client.put("/config/apps/demo/ai", json=body, headers=headers)
    assert retry.status_code == 422
    assert retry.headers.get("Idempotent-Replay") != "true"
    assert_no_canary(retry.text)


# --- a credential embedded IN base_url ---------------------------------------
#
# Leaf-name masking (`api_key`, `password`) cannot see a credential that lives
# INSIDE a value, and `validate_ai_section` runs no userinfo check on
# `base_url`. A `https://user:pw@host/v1` therefore rode a SUCCESSFUL deploy
# into `audit_log.params_redacted` and the admin `audit.*` bus frame.

USERINFO_CANARY = "pw-CANARY-1111"


def test_audit_safe_ai_strips_userinfo_from_base_url():
    from nerdit.daemon.deploy_pipeline import _audit_safe_ai

    safe = _audit_safe_ai(
        {
            "cheap": {
                "provider": "api",
                "model": "gpt-4o-mini",
                "base_url": f"https://svc:{USERINFO_CANARY}@api.example.com/v1",
                "api_key": "${secrets.OPENAI}",
            }
        }
    )

    assert USERINFO_CANARY not in json.dumps(safe)
    # The host and path survive, so the audit row still says where it pointed.
    assert safe["cheap"]["base_url"] == "https://api.example.com/v1"
    # Every other key is carried through untouched — this is a rendering, not
    # a filter; the persisted spec the launch path reads is a different dict.
    assert safe["cheap"]["model"] == "gpt-4o-mini"


def test_audit_safe_ai_leaves_a_credential_free_binding_byte_identical():
    from nerdit.daemon.deploy_pipeline import _audit_safe_ai

    spec = {
        "default": {"provider": "ollama", "model": "llama3.1:8b"},
        "cheap": {"provider": "api", "base_url": "https://api.openai.com/v1"},
    }
    assert _audit_safe_ai(spec) == spec
