"""Tests for the structured error envelope and its backward-compat ``detail``.

The whole point of S2 is *additive*: new clients read ``code``/``message`` while
every existing ``resp.json()["detail"]`` assertion keeps passing. These tests
lock both halves of that contract, including the auth-middleware path (which
hand-builds the envelope outside Starlette's ExceptionMiddleware).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nerdit.daemon.errors import (
    NerditError,
    RequestIdMiddleware,
    _envelope,
    register_error_handlers,
)
from nerdit.daemon.middleware import BearerAuthMiddleware


class _Body(BaseModel):
    name: str
    count: int


def _make_app(*, auth_token: str | None = None) -> FastAPI:
    """A minimal app mirroring create_app's error wiring (handlers + middlewares)."""
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom/{status}")
    def boom(status: int) -> dict:
        raise HTTPException(status_code=status, detail="introuvable")

    @app.get("/too-large")
    def too_large() -> dict:
        raise HTTPException(status_code=413, detail="Upload too large")

    @app.get("/nerdit-error")
    def nerdit_error() -> dict:
        raise NerditError(
            status_code=409,
            code="conflict",
            message="already exists",
            hint="retry with a new name",
            diagnostics=["dup"],
        )

    @app.post("/validate")
    def validate(body: _Body) -> dict:
        return {"ok": True}

    @app.get("/ok")
    def ok() -> dict:
        return {"ok": True}

    if auth_token is not None:
        app.add_middleware(BearerAuthMiddleware, token=auth_token)
    app.add_middleware(RequestIdMiddleware)
    return app


# --- _envelope unit contract -------------------------------------------------


def test_envelope_always_sets_detail_to_message_by_default():
    env = _envelope("not_found", "missing")
    assert env["code"] == "not_found"
    assert env["message"] == "missing"
    assert env["detail"] == "missing"  # alias mirrors message


def test_envelope_preserves_explicit_detail_and_extras():
    env = _envelope("validation_error", "bad", detail=[{"loc": ["x"]}], diagnostics=[1])
    assert env["detail"] == [{"loc": ["x"]}]
    assert env["diagnostics"] == [1]


def test_envelope_omits_optional_fields_when_absent():
    env = _envelope("forbidden", "no")
    assert "hint" not in env
    assert "request_id" not in env


# --- HTTPException retrofit ---------------------------------------------------


def test_http_exception_retrofit_envelope_and_detail():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/boom/404")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "not_found"
    assert body["message"] == "introuvable"
    # Backward-compat alias preserved verbatim.
    assert body["detail"] == "introuvable"
    assert "introuvable" in body["detail"]


def test_http_exception_unmapped_status_uses_fallback_code():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/boom/418")
    assert resp.status_code == 418
    assert resp.json()["code"] == "error"
    assert resp.json()["detail"] == "introuvable"


def test_method_not_allowed_maps_code_and_preserves_allow_header():
    """405 gets the ``method_not_allowed`` code AND keeps the RFC-7231 Allow
    header (both pinned so neither can drift)."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    # /ok is GET-only; POSTing to it yields a 405 from the router.
    resp = client.post("/ok")
    assert resp.status_code == 405
    assert "GET" in resp.headers.get("allow", "")
    body = resp.json()
    assert body["code"] == "method_not_allowed"
    # Legacy string ``detail`` alias preserved.
    assert body["detail"] == "Method Not Allowed"


class _ReqFor:
    """Minimal Request stand-in exposing ``app`` + ``url.path`` (+ empty headers)
    for the 405 handler / _allowed_methods helper."""

    def __init__(self, app: FastAPI, path: str) -> None:
        self.app = app
        self.url = type("U", (), {"path": path})()
        self.headers: dict[str, str] = {}
        self.state = type("S", (), {})()


async def test_method_not_allowed_recomputes_allow_when_starlette_header_lost():
    """Real regression guard: the full daemon middleware stack drops Starlette's
    routing Allow header before it reaches the handler (the minimal _make_app
    can't reproduce that), so the handler recomputes it from the app routes.
    Drive the handler with a headerless 405, as the middleware boundary leaves
    it, and assert Allow is reconstructed."""
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from nerdit.daemon.errors import _allowed_methods, _http_exception_handler

    app = _make_app()
    exc = StarletteHTTPException(status_code=405)  # headers=None, as lost
    out = await _http_exception_handler(_ReqFor(app, "/ok"), exc)
    assert out.status_code == 405
    assert out.headers.get("allow") == "GET"  # /ok is GET-only, recomputed
    assert out.body  # envelope still rendered
    # The helper is authoritative for a multi-method path and drops HEAD.
    assert _allowed_methods(_ReqFor(app, "/validate")) == "POST"


def test_request_id_header_echoed_and_in_envelope():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/boom/404", headers={"X-Request-Id": "abc123"})
    assert resp.headers["X-Request-Id"] == "abc123"
    assert resp.json()["request_id"] == "abc123"


def test_request_id_generated_when_absent():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/boom/404")
    rid = resp.headers.get("X-Request-Id")
    assert rid
    assert resp.json()["request_id"] == rid


# --- NerditError --------------------------------------------------------------


def test_nerdit_error_precise_code_hint_and_extra():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get("/nerdit-error")
    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "conflict"
    assert body["message"] == "already exists"
    assert body["hint"] == "retry with a new name"
    assert body["diagnostics"] == ["dup"]
    assert body["detail"] == "already exists"


# --- RequestValidationError ---------------------------------------------------


def test_validation_error_envelope_with_diagnostics_and_legacy_detail():
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.post("/validate", json={"name": "x"})  # missing 'count'
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "validation_error"
    # FastAPI's documented {"detail": [...]} contract still holds.
    assert isinstance(body["detail"], list)
    assert body["diagnostics"] == body["detail"]
    locs = [".".join(str(p) for p in err["loc"]) for err in body["detail"]]
    assert any("count" in loc for loc in locs)


# --- Auth middleware (hand-built envelope) -----------------------------------


def test_auth_missing_header_returns_envelope_with_detail():
    client = TestClient(_make_app(auth_token="secret"), raise_server_exceptions=False)
    resp = client.get("/ok")
    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "unauthenticated"
    assert "Authentication required" in body["message"]
    # Pre-route denials carry ``detail`` as an empty OBJECT (transport-guard shape).
    assert body["detail"] == {}
    assert body["request_id"]  # set by RequestIdMiddleware


def test_auth_bad_format_returns_invalid_auth_format():
    client = TestClient(_make_app(auth_token="secret"), raise_server_exceptions=False)
    resp = client.get("/ok", headers={"Authorization": "Token x"})
    assert resp.status_code == 401
    body = resp.json()
    assert body["code"] == "invalid_auth_format"
    assert "Invalid Authorization header format" in body["message"]
    assert body["detail"] == {}


def test_auth_invalid_token_returns_403_invalid_token():
    client = TestClient(_make_app(auth_token="secret"), raise_server_exceptions=False)
    resp = client.get("/ok", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 403
    body = resp.json()
    assert body["code"] == "invalid_token"
    assert "Invalid token" in body["message"]
    assert body["detail"] == {}


def test_auth_valid_token_passes_through():
    client = TestClient(_make_app(auth_token="secret"), raise_server_exceptions=False)
    resp = client.get("/ok", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


# --- Backward-compat sweep ----------------------------------------------------


@pytest.mark.parametrize(
    ("path", "status", "check"),
    [
        ("/boom/404", 404, lambda d: d == "introuvable"),
        ("/too-large", 413, lambda d: d == "Upload too large"),
    ],
)
def test_detail_alias_backward_compat_strings(path, status, check):
    """Representative existing string-``detail`` shapes are unchanged."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.get(path)
    assert resp.status_code == status
    assert isinstance(resp.json()["detail"], str)
    assert check(resp.json()["detail"])


def test_detail_alias_backward_compat_validation_list():
    """422 keeps the pydantic error *list* under ``detail`` (FastAPI contract)."""
    client = TestClient(_make_app(), raise_server_exceptions=False)
    resp = client.post("/validate", json={})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert all("loc" in err and "msg" in err for err in detail)
