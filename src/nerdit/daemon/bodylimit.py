"""Bound selected request bodies before FastAPI parses them.

Workspace JSON must be limited in middleware: route/dependency checks run
after parsing and cannot prevent memory exhaustion. Require Content-Length
and reject chunked bodies on bounded routes. The deploy and apply uploads are
bounded the same way, because Starlette spools every multipart file part to
disk before the route can check its size. The MCP transport cap is separate.
"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from nerdit.config.defaults import DEFAULT_MAX_UPLOAD_BYTES, WORKSPACE_MAX_BODY_BYTES
from nerdit.daemon.errors import _envelope, request_id_of

#: `(method, path pattern, limit)`. The workspace router is mounted under
#: `/api` only (no bare-root alias), so one pattern covers it.
_BODY_LIMITS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("PUT", re.compile(r"^/api/workspaces/[^/]+/files$"), WORKSPACE_MAX_BODY_BYTES),
)

#: Multipart ZIP ingresses, bounded at `[daemon].max_upload_bytes` plus slack.
_UPLOAD_ROUTES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("POST", re.compile(r"^/api/deploy$")),
    ("POST", re.compile(r"^/api/projects/[^/]+/apply$")),
)
#: Form fields and multipart framing ride beside the archive.
MULTIPART_SLACK_BYTES = 1_048_576


def _limit_for(method: str, path: str) -> int | None:
    """Return the declared-body limit for this request, or `None` when unbounded."""
    for bound_method, pattern, limit in _BODY_LIMITS:
        if method == bound_method and pattern.fullmatch(path):
            return limit
    return None


class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Refuse an over-declared (or undeclared) body on the bounded routes.

    Refusals **return** a response rather than raising: this sits outside the
    router, so a raised `NerditError` would never reach the FastAPI exception
    handlers. The envelope is therefore built here, in the one shape every other
    error uses, and it is path-free and never echoes a byte of the body.
    """

    def __init__(self, app: ASGIApp, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> None:
        super().__init__(app)
        self._upload_limit = max_upload_bytes + MULTIPART_SLACK_BYTES

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method, path = request.method, request.url.path
        limit = _limit_for(method, path)
        hint_411 = "Send the batch as a single buffered JSON body, not chunked."
        hint_413 = (
            "Split the batch across several write calls, or deploy from a "
            "repository with deploy_git."
        )
        if limit is None and any(
            method == m and pattern.fullmatch(path) for m, pattern in _UPLOAD_ROUTES
        ):
            limit = self._upload_limit
            hint_411 = (
                "Send the archive as a buffered multipart body with a Content-Length, not chunked."
            )
            hint_413 = (
                "Deploy a smaller archive, deploy from a repository, or raise "
                "[daemon].max_upload_bytes."
            )
        if limit is None:
            return await call_next(request)

        raw = request.headers.get("content-length")
        try:
            length = int(raw) if raw is not None else None
        except ValueError:
            # An unparseable header is no header: refuse rather than guess.
            length = None

        if length is None:
            return self._refuse(
                request,
                411,
                "length_required",
                "Content-Length required on this route.",
                hint=hint_411,
            )
        if length > limit:
            return self._refuse(
                request,
                413,
                "payload_too_large",
                f"The request body is {length} bytes; the limit is {limit} bytes.",
                hint=hint_413,
                detail={"limit": limit},
                limit=limit,
            )
        return await call_next(request)

    @staticmethod
    def _refuse(
        request: Request,
        status_code: int,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        **extra: object,
    ) -> JSONResponse:
        """Build the structured envelope directly (no handler runs out here)."""
        return JSONResponse(
            status_code=status_code,
            content=_envelope(
                code,
                message,
                hint=hint,
                request_id=request_id_of(request),
                **extra,
            ),
        )
