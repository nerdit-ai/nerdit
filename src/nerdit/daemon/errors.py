"""Build structured API errors with code, message, hint, request_id and detail.

Always populate the backward-compatible detail field; diagnostics are optional.
Exception handlers and outer auth middleware share the same envelope.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ExceptionHandler

# Mapping from HTTP status code to a stable machine-readable error code. Used to
# retrofit a `code` onto plain `HTTPException`s raised by frozen routes
# without editing any of them.
_CODE_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthenticated",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    500: "internal_error",
}

_FALLBACK_CODE = "error"


class NerditError(Exception):
    """A daemon error mapped to a structured response envelope.

    Extra keywords become envelope fields. Headers is a separate keyword-only
    argument because response headers must not be serialized into the JSON body.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        hint: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.hint = hint
        self.headers = headers
        self.extra = extra


# Client-supplied ids reach every audit row, the `audit.*` feed and the
# response header, so only a short, header-safe token is echoed.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _client_request_id(value: str | None) -> str | None:
    """Return `value` when it is a well-formed request id, else None."""
    return value if value and _REQUEST_ID_RE.fullmatch(value) else None


def request_id_of(request: Request) -> str | None:
    """Return the request id assigned by `RequestIdMiddleware`, if any."""
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid
    return _client_request_id(request.headers.get("x-request-id"))


def _envelope(
    code: str,
    message: str,
    *,
    hint: str | None = None,
    request_id: str | None = None,
    detail: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the flat error envelope, **always** populating `detail`.

    `detail` defaults to `message` (a string) but callers pass the original
    value when they must preserve a legacy shape verbatim (e.g. the pydantic
    error list for 422s, or a frozen route's French `detail` string).
    """
    body: dict[str, Any] = {"code": code, "message": message}
    if hint is not None:
        body["hint"] = hint
    if request_id is not None:
        body["request_id"] = request_id
    # Backward-compat alias: ALWAYS present. Never drop this.
    body["detail"] = message if detail is None else detail
    body.update(extra)
    return body


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign a request id, echoing a well-formed `X-Request-Id`.

    A malformed header (outside `[A-Za-z0-9._:-]{1,128}`) is replaced by a
    minted id, never echoed.

    Added as the outermost custom middleware so the id is available on
    `request.state` to inner middlewares (auth) and the exception handlers.
    """

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        rid = _client_request_id(request.headers.get("x-request-id")) or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers.setdefault("X-Request-Id", rid)
        return response


async def _nerdit_error_handler(request: Request, exc: NerditError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        # Response headers when the raise site set any (`Retry-After` on a
        # forwarded rate limit today), the same shape the HTTPException handler
        # below already honours. `None` for every other raise site, so this
        # is inert unless a caller opts in.
        headers=exc.headers,
        content=_envelope(
            exc.code,
            exc.message,
            hint=exc.hint,
            request_id=request_id_of(request),
            **exc.extra,
        ),
    )


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Retrofit an envelope onto a plain `HTTPException`, `detail` verbatim.

    Frozen routes raise `HTTPException(status_code=..., detail="...")`; their
    `detail` (almost always a localized string) is preserved exactly so every
    existing `resp.json()["detail"]` assertion keeps passing.
    """
    detail = exc.detail
    code = _CODE_BY_STATUS.get(exc.status_code, _FALLBACK_CODE)
    message = detail if isinstance(detail, str) else code
    headers = getattr(exc, "headers", None)
    if exc.status_code == 405 and not _has_header(headers, "allow"):
        # Starlette sets Allow on the routing 405, but it is lost before it
        # reaches this handler (BaseHTTPMiddleware boundary). Recompute it from
        # the app's routes so an agent can discover the permitted methods.
        allow = _allowed_methods(request)
        if allow:
            headers = {**(headers or {}), "Allow": allow}
    return JSONResponse(
        status_code=exc.status_code,
        headers=headers,
        content=_envelope(
            code,
            message,
            request_id=request_id_of(request),
            detail=detail,
        ),
    )


def _has_header(headers: Iterable[object] | None, name: str) -> bool:
    """True when a header mapping already carries `name` (case-insensitive)."""
    if not headers:
        return False
    try:
        return name.lower() in {str(k).lower() for k in headers}
    except TypeError:
        return False


def _allowed_methods(request: Request) -> str | None:
    """Comma-joined methods allowed for `request`'s path, from the app routes."""
    path = request.url.path
    methods: set[str] = set()
    for route in getattr(request.app, "routes", []):
        if getattr(route, "path", None) == path:
            methods |= set(getattr(route, "methods", None) or ())
    methods.discard("HEAD")
    return ", ".join(sorted(methods)) if methods else None


# Request-body field names whose submitted value is secret-bearing. A pydantic
# v2 error entry carries the offending `input`, so a 422 on one of these
# hands the caller's own credential straight back in the envelope — and an
# agent or CLI that logs the error writes it into its transcript. That defeats
# the standing gate that secret values appear in no response body or error
# message, which is why the value is masked while the rest of
# the pydantic entry (type, loc, msg) is preserved verbatim.
#
# Deliberately name-based and deliberately small: `env` is a free-form map of
# user-chosen names, so no name-matching could catch `OPENAI_API_KEY` inside
# it — the whole map is treated as opaque, mirroring `audit._REDACT_MAP_KEYS`.
# (Kept local rather than imported: `audit` imports THIS module, so reaching
# the other way would cycle.)
_SECRET_INPUT_FIELDS = frozenset(
    {
        "env",
        "environment",
        "build_settings",  # Rejected build inputs may contain pasted credentials.
        "secret",
        "secrets",
        "token",
        "token_ref",
        "auth_token",
        "api_key",
        "password",
        # `LinkClaimRequest.code` — the plaintext link code (P27 WP-C2), a
        # secret in transit that must appear in NO response body. `repr=False`
        # on the field does not cover this: a constraint failure
        # (`string_too_long`) or a sibling `missing` raises BEFORE the model
        # is constructed, so the echoed `input` is FastAPI's raw dict, not the
        # model repr. Safe to name globally: the only other model field called
        # `code` is `service_diagnose.Remediation.code`, a response-only
        # model that never binds a request input, and the `code` key of the
        # structured error envelope is response-side too.
        "code",
        # `LinkClaimRequest.key` — the `nk_` pre-auth key, a
        # live grant that enrolls a machine. Same reasoning as `code`: the
        # shape validator's `value_error` carries the submitted value as
        # `input`, and the keys most likely to be malformed are miscased
        # REAL keys (Crockford is case-insensitive, so a lowercased echo is
        # secret-equivalent) — a 422 that echoes one undoes the whole
        # stdin → memory → JSON-body custody chain (D-X16-O11) in one line.
        # Safe to name globally: no other request model binds a `key` field
        # (`ConfigDiffEntry.key` is response-side and never validates input).
        "key",
        # `LinkClaimRequest.api_url`/`relay_url` (PR #115 round 4): a URL
        # REJECTED by its validator (a path-parked bearer token, userinfo) is
        # exactly the one that may carry a credential, and the 422 echo would
        # hand it back into CLI/agent transcripts. Scrubbing the config-PUT
        # echo of a malformed `relay_url` too is the acceptable cost.
        "api_url",
        "relay_url",
        # `LicenseInstallRequest.blob` — the signed product license (P17d
        # D-LIC5), a paid artifact handled as a secret in transit. Same
        # reasoning as `code`: `repr=False` cannot cover a
        # `string_too_long` or a sibling `missing`, whose echoed `input`
        # is FastAPI's raw dict. No other request model binds a `blob` field.
        "blob",
        # `SecretSetRequest.values` — the one request body that is credential
        # all the way down. Only that model binds a `values` field from a
        # request; `ConfigView.values` is response-only and never validated.
        # (P40c) `VariableSetRequest.values` is the second binder, same reason.
        "values",
        # (P40c / D-P40-10) Singular twin, depth only: no request model binds a
        # `value` field today, so a future one is masked by default.
        "value",
        # `WorkspaceWriteRequest.files` — the P29 content ingress. A pydantic
        # 422 (a non-string value, a nested type error) echoes the offending
        # `input`, which here is the caller's own source file. Content never
        # rides an error body; the `loc` still names the offending path. No
        # other request model binds a `files` field.
        "files",
    }
)


def _redact_secret_keys(value: Any) -> Any:
    """Recursively mask secret-bearing keys inside a mapping-shaped `input`.

    A `missing`-type error on a sibling field echoes the WHOLE parent dict as
    its `input` — with a `loc` naming only the missing field — so the
    loc-based test alone lets a submitted `env` map through verbatim. Walk
    dicts (and lists of dicts) and mask any value under a secret-bearing key.
    """
    if isinstance(value, dict):
        return {
            k: "***"
            if isinstance(k, str) and k.lower() in _SECRET_INPUT_FIELDS
            else _redact_secret_keys(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_keys(item) for item in value]
    return value


def _redact_validation_inputs(errors: list[Any]) -> list[Any]:
    """Mask the echoed `input` on any error whose field is secret-bearing.

    `extra_forbidden` (P22 WP-C) is masked unconditionally: its `loc` names
    a field the model does NOT know, so a mistyped secret-bearing key
    (`envv`, `secrts`) never matches `_SECRET_INPUT_FIELDS` and would
    hand back the whole submitted map verbatim. The `loc` alone tells the
    caller which key was rejected, which is the entire diagnostic value here.
    """
    for err in errors:
        if not isinstance(err, dict) or "input" not in err:
            continue
        loc = err.get("loc") or ()
        if err.get("type") == "extra_forbidden" or any(
            isinstance(part, str) and part.lower() in _SECRET_INPUT_FIELDS for part in loc
        ):
            err["input"] = "***"
        else:
            err["input"] = _redact_secret_keys(err["input"])
    return errors


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Emit `validation_error` while keeping FastAPI's `{"detail": [...]}`.

    The pydantic error list lands in both `diagnostics` (new) and `detail`
    (legacy), so FastAPI's documented 422 contract still holds — except that a
    secret-bearing field's echoed `input` is masked first
    (`_SECRET_INPUT_FIELDS`).
    """
    errors = _redact_validation_inputs(jsonable_encoder(exc.errors()))
    return JSONResponse(
        status_code=422,
        content=_envelope(
            "validation_error",
            "Request validation failed.",
            request_id=request_id_of(request),
            detail=errors,
            diagnostics=errors,
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register the three envelope-emitting exception handlers on `app`."""
    app.add_exception_handler(NerditError, cast("ExceptionHandler", _nerdit_error_handler))
    app.add_exception_handler(
        StarletteHTTPException, cast("ExceptionHandler", _http_exception_handler)
    )
    app.add_exception_handler(
        RequestValidationError, cast("ExceptionHandler", _validation_exception_handler)
    )
