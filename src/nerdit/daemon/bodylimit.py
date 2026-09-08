"""Bound selected request bodies before FastAPI parses them.

Workspace JSON must be limited in middleware: route/dependency checks run
after parsing and cannot prevent memory exhaustion. Require Content-Length
and reject chunked bodies on bounded routes. The MCP transport cap is separate.
"""

from __future__ import annotations

import re

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from nerdit.config.defaults import WORKSPACE_MAX_BODY_BYTES
from nerdit.daemon.errors import _envelope, request_id_of

#: `(method, path pattern, limit)`. The workspace router is mounted under
#: `/api` only (no bare-root alias), so one pattern covers it.
_BODY_LIMITS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("PUT", re.compile(r"^/api/workspaces/[^/]+/files$"), WORKSPACE_MAX_BODY_BYTES),
)


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

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        limit = _limit_for(request.method, request.url.path)
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
                hint="Send the batch as a single buffered JSON body, not chunked.",
            )
        if length > limit:
            return self._refuse(
                request,
                413,
                "payload_too_large",
                f"The request body is {length} bytes; the limit is {limit} bytes.",
                hint="Split the batch across several write calls, or deploy from a "
                "repository with deploy_git.",
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
