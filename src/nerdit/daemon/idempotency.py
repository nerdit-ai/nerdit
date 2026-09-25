"""Make mutating requests safe to retry without caching secrets.

Runs inside Auth and Audit so replays remain authenticated and audited.
Keys combine principal, Idempotency-Key, method and concrete URL path;
`/services` and `/api/services` therefore do not deduplicate across mounts.
Never depend on `scope['route']`, which is unavailable before routing.

Hash a request body only when Content-Length is present and bounded, the body
is not multipart, and the action is outside NO_BODY_HASH_ACTIONS. Persist only
the digest, never raw bytes. Compare digests only when both exist; exempt and
legacy requests still replay. Starlette's cached request lets handlers reread
a body consumed here. NO_BODY_CACHE_ACTIONS store status and resource ID only;
all other cached bodies pass the secret-redaction denylist before persistence.

A fresh claim is inserted as in_progress under BEGIN IMMEDIATE. A 2xx response
completes and pins it; a non-2xx or pre-commit failure deletes it for retry.
Existing completed claims replay, in_progress claims return 409, interrupted
claims return 409 idempotency_interrupted, and method/path/body conflicts
return 422. Cancellation after a durable effect marks the claim interrupted
rather than risking re-execution; the TTL sweep eventually removes it.

REQUEST_WRITE_MARKER is inherited by the route task and flipped after every
serialized DB commit. Non-idempotent effects outside the DB must call
mark_request_side_effect before starting. Backup workers do this because they
outlive cancellation; secret-file set/merge may safely retry without a marker.
A client disconnect does not cancel a non-streaming handler on uvicorn/h11;
graceful-shutdown cancellation does, including after a durable run audit row.

Cleanup and finalization use both an anyio shield and a detached task. The
shield blocks repeated scope cancellation; the detached task survives raw
Task.cancel even when anyio does not track the caller. Cancellation propagates
on schedule while detached bookkeeping completes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import anyio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from nerdit.daemon.audit import _strip_api_prefix, derive_action, is_mcp_path, redact
from nerdit.daemon.auth import current_principal
from nerdit.daemon.errors import _envelope, request_id_of
from nerdit.db.queries._base import REQUEST_WRITE_MARKER, RequestWriteMarker

logger = logging.getLogger(__name__)

# Methods that mutate state — the only ones the idempotency layer acts on.
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# The mutating routes that actually honor a `?dry_run` query param — the ONLY
# ones for which the dry-run idempotency bypass may fire. FastAPI silently
# ignores an undeclared query param, so `POST /services?dry_run=true` (or any other
# route without a `dry_run` handler arg) still performs the REAL write; the
# bypass must not skip idempotency there or a retry would duplicate the mutation.
# Patterns match the path **after** the optional `/api` prefix is stripped
# (mirrors `nerdit.daemon.audit._ROUTE_RULES`), so bare-root and `/api`
# mounts both resolve. Keep this in lockstep with the route handlers that
# declare `dry_run: bool = Query(...)`.
_DRY_RUN_ROUTES: list[tuple[str, re.Pattern[str]]] = [
    ("POST", re.compile(r"^/deploy$")),
    ("POST", re.compile(r"^/deploy/git$")),
    ("POST", re.compile(r"^/deploy/[^/]+/redeploy$")),
    # Without this entry a `?dry_run=true` workspace deploy burns
    # the caller's Idempotency-Key and poisons the later real deploy.
    ("POST", re.compile(r"^/workspaces/[^/]+/deploy$")),
    ("POST", re.compile(r"^/app-templates/[^/]+/deploy$")),
    # A dry-run apply writes nothing (D-P40-12), so it must not burn the
    # key the real apply will carry.
    ("POST", re.compile(r"^/projects/[^/]+/apply$")),
    ("POST", re.compile(r"^/config/daemon/apply$")),
    ("PUT", re.compile(r"^/config/daemon/[^/]+$")),
    ("PUT", re.compile(r"^/config/apps/[^/]+/[^/]+$")),
    ("POST", re.compile(r"^/system/gc$")),
]


# Mutating routes for which an `Idempotency-Key` is neither sent nor
# meaningful, and which must therefore never be 400'd by
# `[security].require_idempotency_key`. The rationale is the MCP bypass's: the
# caller is a machine and the write is idempotent BY VALUE — the entitlement
# push is a pure set of one bool, ordered by the `issued_at` the body carries,
# so replaying it is indistinguishable from re-asserting it and there is nothing
# for a stored key to protect. Patterns match the path after `/api` is
# stripped, like the two tables above. Keep this list tiny and machine-only: an
# entry here removes a caller's replay protection.
_IDEMPOTENCY_EXEMPT_PATHS: list[tuple[str, re.Pattern[str]]] = [
    ("PUT", re.compile(r"^/link/entitlement$")),
    # The GitHub installation-token push: set-by-value, ordered
    # by `issued_at` per installation, re-minted on a timer by the cloud.
    ("PUT", re.compile(r"^/link/github-token$")),
    # Immutable assignment: the same value is a no-op, any different value conflicts.
    ("PUT", re.compile(r"^/link/public-address$")),
]


def _matches(rules: list[tuple[str, re.Pattern[str]]], method: str, path: str) -> bool:
    """True iff `(method, path)` matches a rule, after stripping the `/api` prefix."""
    stripped = _strip_api_prefix(path)
    return any(m == method and p.match(stripped) is not None for m, p in rules)


def is_idempotency_exempt(method: str, path: str) -> bool:
    """True iff `(method, path)` is a machine route that carries no key."""
    return _matches(_IDEMPOTENCY_EXEMPT_PATHS, method, path)


def _honors_dry_run(method: str, path: str) -> bool:
    """True iff `(method, path)` is a route that implements `?dry_run`."""
    return _matches(_DRY_RUN_ROUTES, method, path)


# Retention for a stored idempotency key. Records older than this are swept.
IDEMPOTENCY_TTL_HOURS = 24

# Upper bound on the `Idempotency-Key` header length. A key over this is
# rejected 422 *before* the claim (nothing persisted), guarding against
# storage amplification. Well above every key the CLI/MCP mint (uuid4 hex = 32).
_MAX_IDEMPOTENCY_KEY_LEN = 255

# Never cache responses containing one-time secrets, run output or customer IDs.
# Replay status/resource_id in a non-secret envelope instead. Lost run responses
# are recoverable from owner-gated diagnose.last_run; lost token rotations require
# admin re-minting. License install can be retried with a fresh key, while remove's
# boolean response is safe to cache. Device-start responses contain a live user
# code and become stale when a later session replaces them; never persist them.
NO_BODY_CACHE_ACTIONS = {
    "token.create",
    "token.rotate",
    "service.run",
    "license.install",
    "link.device_started",
}

# Never hash caller-supplied secret/env maps or bare link/license grants.
# A digest can expose low-entropy secrets to offline guessing or confirm a known
# license blob, including through backups. Exempt secret.set, template.deploy,
# service.run, service.create, link.created and license.install. Database passwords
# are minted server-side; config secrets already live in plaintext TOML on disk.
# Link-claim responses are safe to cache; license responses contain customer IDs.
NO_BODY_HASH_ACTIONS = {
    "secret.set",
    "variable.set",  # the same value map one noun up; its response is names only
    "template.deploy",
    "service.run",
    "service.create",
    "link.created",
    "license.install",
}

# Upper bound on a body the middleware will buffer to digest it. Mirrors the
# MCP transport's body-cap posture: the declared `Content-Length` decides
# up front (never cap-while-read), and an absent one skips hashing entirely.
_MAX_HASH_BODY_BYTES = 1_048_576

_REPLAY_HEADER = "Idempotent-Replay"


_R = TypeVar("_R")


def _log_detached_failure(task: asyncio.Task[Any]) -> None:
    """Never let a detached bookkeeping failure be silent."""
    if task.cancelled():
        return
    if task.exception() is not None:
        logger.warning("Detached idempotency bookkeeping failed", exc_info=task.exception())


async def _detached(coro: Coroutine[Any, Any, _R]) -> _R:
    """Await `coro` in its own task, so a RAW cancellation cannot abort it.

    `anyio.CancelScope(shield=True)` defers a raw `asyncio.Task.cancel()`
    (uvicorn's graceful-shutdown shape) only for a task anyio TRACKS — which the
    request task is solely because starlette's `BaseHTTPMiddleware` wraps it in
    a task group. Entered from an untracked task the cancel goes straight
    through, aborting the bookkeeping mid-flight and stranding the claim
    `in_progress` for the full TTL. A separate task is never the cancellation
    target, so it runs to completion either way; `asyncio.shield` keeps the
    *awaiting* side cancellable, which is what lets the cancellation propagate to
    the caller on schedule.

    On the cancelled path the return value is dropped (the response can no longer
    be sent), so the detached task's outcome is only logged.
    """
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(_log_detached_failure)
        raise


def _principal_id(request: Request) -> str:
    """Stable per-principal scope for the idempotency key.

    Scoped tokens key off their row id; the legacy/local bypasses (`token_id`
    is `None`) key off their distinct sentinel name, so they never collide.
    """
    principal = current_principal(request)
    if principal.project_id is not None:
        return f"{principal.token_id}:project:{principal.project_id}"
    if principal.token_id is not None:
        return principal.token_id
    return f"anon:{principal.name}"


async def _hashable_body(request: Request, action: str) -> str | None:
    """SHA-256 hex digest of the request body, or `None` when the gate refuses.

    The gate is the whole of D-P22-2 and every clause is load-bearing: the
    secret-carrying actions must leave no fingerprint at rest, `multipart`
    bodies are 500 MB uploads that only the streaming spool may touch, and an
    absent or over-large `Content-Length` means the body is not something this
    middleware may buffer. Each refusal yields `None`, which the soft
    comparison reads as "no opinion" rather than as a mismatch.

    Only an explicitly MAPPED action may be hashed. `derive_action` matches
    anchored patterns, so a non-canonical spelling of a secret-carrying path
    (`POST /secrets/myapp/`, which Starlette answers with a 307 redirect)
    derives the `"METHOD /templated/path"` fallback instead of `secret.set`
    — and would slip past the denylist above with the secret body in hand.
    Fallback actions are the only ones containing a space, so requiring a mapped
    action makes the exclusion spelling-proof rather than enumerating spellings.
    """
    if action in NO_BODY_HASH_ACTIONS or " " in action:
        return None
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type == "multipart/form-data":
        return None
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    try:
        length = int(declared)
    except ValueError:
        return None
    if length > _MAX_HASH_BODY_BYTES:
        return None
    return hashlib.sha256(await request.body()).hexdigest()


def _buffer_body(chunks: list[Any]) -> bytes:
    """Join a streamed response body into bytes."""
    return b"".join(c if isinstance(c, bytes) else str(c).encode("utf-8") for c in chunks)


def _extract_resource_id(body: bytes) -> str | None:
    """Best-effort resource id from a JSON response body (for traceability).

    `id` is the house shape, but a body whose resource is not a row does not
    have one: a one-off run answers with `run_id` and no `id` at all. Since
    that body is also deliberately never cached (`NO_BODY_CACHE_ACTIONS`,
    D-P20-6), the id captured HERE is the only handle a replay can offer — and
    without it a replayed run is unidentifiable, so a caller can only fall back
    to "the most recent run on this service", which a concurrent run has
    already made wrong.
    """
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, dict):
        for key in ("id", "run_id"):
            rid = parsed.get(key)
            if rid is not None:
                return str(rid)
    return None


def _redacted_body_text(body: bytes) -> str:
    """Return the response body as text with secret keys masked.

    JSON bodies are parsed, run through the secret denylist, and re-serialized;
    non-JSON bodies are returned verbatim (no structured keys to mask).
    """
    try:
        parsed = json.loads(body)
    except (TypeError, ValueError):
        return body.decode("utf-8", errors="replace")
    return json.dumps(redact(parsed))


class IdempotencyMiddleware(BaseHTTPMiddleware):
    """Replay-safe handling of mutating requests carrying an `Idempotency-Key`."""

    def __init__(
        self,
        app,  # noqa: ANN001
        get_queries: Callable[[], Any | None] | None = None,
        require_idempotency_key: bool = False,
    ) -> None:
        super().__init__(app)
        self._get_queries = get_queries
        self._require = require_idempotency_key

    def _queries(self):  # noqa: ANN202
        return self._get_queries() if self._get_queries is not None else None

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        """Claim/replay an idempotency key around a mutating request."""
        if request.method not in _MUTATING_METHODS:
            return await call_next(request)

        # The MCP mount is bypassed entirely: this middleware drains
        # body_iterator on a claim, which would buffer/corrupt an SSE-shaped MCP
        # response, and a stray Idempotency-Key would be claimed for the bare
        # path. Idempotency is enforced where it matters — the inner REST hop,
        # where the write _impls mint keys.
        if is_mcp_path(request.url.path):
            return await call_next(request)

        # The same bypass shape for the cloud's key-less machine writes.
        if is_idempotency_exempt(request.method, request.url.path):
            return await call_next(request)

        # A `?dry_run=true` write validates without persisting anything, so the
        # key must be ignored ENTIRELY (neither claimed nor recorded). The record
        # is keyed on `(principal, key, method, url.path)` with the query string
        # excluded, so claiming here would let a later real POST with the same key
        # replay the cached dry-run response — a silent no-deploy. Two guards must
        # BOTH hold: (1) the value is truthy, matching pydantic v2's str→bool
        # truthy set exactly (`1/true/yes/on/t/y`, input already lowercased) so
        # the middleware bypass and the route's own coercion can never disagree on
        # which spellings count as a dry run — it must stay a *subset* of that set,
        # since false-y/invalid spellings 422 at the route before persisting; and
        # (2) the concrete `(method, path)` is a route that actually declares a
        # `dry_run` handler arg (`_DRY_RUN_ROUTES`). FastAPI ignores an
        # undeclared query param, so without guard (2) a stray `?dry_run=true` on
        # any other mutating route (e.g. `POST /services`) would skip idempotency on a
        # request that really executes — a retry could then duplicate the mutation.
        dry_run = (request.query_params.get("dry_run") or "").lower()
        if dry_run in {"1", "true", "yes", "on", "t", "y"} and _honors_dry_run(
            request.method, request.url.path
        ):
            return await call_next(request)

        idem_key = request.headers.get("Idempotency-Key")
        if not idem_key:
            if self._require:
                return self._reject(
                    request,
                    400,
                    "idempotency_key_required",
                    "This endpoint requires an Idempotency-Key header.",
                )
            return await call_next(request)

        if len(idem_key) > _MAX_IDEMPOTENCY_KEY_LEN:
            return self._reject(
                request,
                422,
                "idempotency_key_too_long",
                f"Idempotency-Key must be at most {_MAX_IDEMPOTENCY_KEY_LEN} "
                f"characters (got {len(idem_key)}).",
                hint="Use a short unique key, e.g. a UUID.",
            )

        queries = self._queries()
        if queries is None:
            # No persistence available (e.g. before lifespan startup) — degrade
            # to a plain pass-through rather than failing the request.
            return await call_next(request)

        method = request.method
        path = request.url.path
        principal_id = _principal_id(request)
        expires_at = (datetime.now(UTC) + timedelta(hours=IDEMPOTENCY_TTL_HOURS)).isoformat()
        body_hash = await _hashable_body(request, derive_action(method, path)[0])

        inserted = await queries.insert_idempotency_inprogress(
            principal_id=principal_id,
            idem_key=idem_key,
            method=method,
            path=path,
            expires_at=expires_at,
            body_hash=body_hash,
        )

        if not inserted:
            record = await queries.get_idempotency_record(principal_id, idem_key)
            if record is None:
                # Swept between the failed insert and the read (rare) — proceed
                # without pinning rather than blocking the caller.
                return await call_next(request)
            if record.method != method or record.path != path:
                return self._reject(
                    request,
                    422,
                    "idempotency_key_conflict",
                    "This Idempotency-Key was already used for a different request.",
                    hint="Use a fresh key per distinct operation.",
                )
            # Soft by design (D-P22-2): a NULL on either side is "no opinion",
            # so legacy rows and every gate-exempt body shape still replay.
            if (
                record.body_hash is not None
                and body_hash is not None
                and record.body_hash != body_hash
            ):
                return self._reject(
                    request,
                    422,
                    "idempotency_key_conflict",
                    "This Idempotency-Key was already used with a different request body.",
                    hint="Use a fresh key per distinct operation, or resend the original body.",
                )
            if record.state == "completed":
                return self._replay(request, record)
            if record.state == "interrupted":
                return self._reject(
                    request,
                    409,
                    "idempotency_interrupted",
                    "A request with this Idempotency-Key began executing and was "
                    "interrupted; its outcome is unknown.",
                    hint=(
                        "Verify the resource state, then use a fresh Idempotency-Key "
                        "deliberately if the operation still needs to run."
                    ),
                )
            return self._reject(
                request,
                409,
                "idempotency_in_progress",
                "A request with this Idempotency-Key is still in progress.",
                hint="Retry after the original request completes.",
            )

        # We claimed the key — run the request and pin/unpin the outcome. The
        # marker rides the ContextVar into the downstream child task, where any
        # `@_serialized` writer flips it on commit (D-P22-4).
        marker = RequestWriteMarker()
        marker_token = REQUEST_WRITE_MARKER.set(marker)
        try:
            try:
                response = await call_next(request)
            except Exception:
                # Never pin a failure (D-P22-5): the caller may retry the key.
                # Both shields are needed and neither subsumes the other: the
                # anyio scope blocks re-delivery from a cancelled anyio scope,
                # `_detached` survives a raw `Task.cancel()`.
                with anyio.CancelScope(shield=True):
                    await _detached(self._safe_delete(queries, principal_id, idem_key))
                raise
            except BaseException:
                # Cancellation — a client disconnect and the uvicorn shutdown
                # cancel are the same code path. The claim is only released if
                # nothing durable landed; otherwise it is pinned `interrupted`
                # so a retry is told the truth instead of re-running a write
                # that may already have committed. A SECOND cancellation landing
                # on this cleanup must not abort it either, hence `_detached`.
                with anyio.CancelScope(shield=True):
                    if marker.committed:
                        await _detached(self._safe_interrupt(queries, principal_id, idem_key))
                    else:
                        await _detached(self._safe_delete(queries, principal_id, idem_key))
                raise

            # A disconnect landing here must not lose the terminal state: the
            # request SUCCEEDED, and the bookkeeping is milliseconds (D-P22-6).
            # The cancellation re-raises once the scope exits; the settle itself
            # is detached so a raw cancel cannot strand the claim `in_progress`
            # (its return value is then dropped — the response cannot be sent).
            with anyio.CancelScope(shield=True):
                return await _detached(
                    self._settle(
                        queries,
                        principal_id=principal_id,
                        idem_key=idem_key,
                        method=method,
                        path=path,
                        response=response,
                    )
                )
        finally:
            REQUEST_WRITE_MARKER.reset(marker_token)

    async def _settle(
        self,
        queries: Any,
        *,
        principal_id: str,
        idem_key: str,
        method: str,
        path: str,
        response: Any,
    ) -> Response:
        """Buffer the response and pin (2xx) or release (non-2xx) the claim.

        Runs under a shield, so the drain below must stay bounded: **no claimed
        mutating route may return an unbounded streaming body.** Every one of
        them answers with a single JSON message today, and the one SSE-shaped
        POST (the MCP mount) is bypassed before any claim is made. A future
        `StreamingResponse` on a claimed route would await here uncancellably
        and wedge the request past the graceful-shutdown deadline — bypass it
        like the MCP mount instead.
        """
        chunks = [chunk async for chunk in response.body_iterator]
        body = _buffer_body(chunks)
        rebuilt = self._rebuild(response, body)

        if 200 <= response.status_code < 300:
            action = derive_action(method, path)[0]
            resource_id = _extract_resource_id(body)
            content_type = response.headers.get("content-type")
            if action in NO_BODY_CACHE_ACTIONS:
                stored_body: str | None = None
            else:
                stored_body = _redacted_body_text(body)
            try:
                await queries.complete_idempotency_record(
                    principal_id=principal_id,
                    idem_key=idem_key,
                    response_status=response.status_code,
                    response_body=stored_body,
                    content_type=content_type,
                    resource_id=resource_id,
                )
            except Exception:
                logger.warning("Failed to pin idempotency result", exc_info=True)
        else:
            # Never pin a failure: let the caller retry with the same key.
            await self._safe_delete(queries, principal_id, idem_key)

        return rebuilt

    @staticmethod
    def _rebuild(response: Response, body: bytes) -> Response:
        """Rebuild a buffered streaming response into a concrete one."""
        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )

    def _replay(self, request: Request, record: Any) -> Response:
        """Return a stored 2xx response (or a non-secret envelope for secrets)."""
        action = derive_action(record.method, record.path)[0]
        if action in NO_BODY_CACHE_ACTIONS or record.response_body is None:
            # The envelope has to say WHY the body is gone, and the two reasons
            # are different: a minted credential is shown once (`token.create`),
            # whereas a run's body is withheld because caching arbitrary command
            # stdout at rest for 24 h — and into every backup tar — is the thing
            # D-P20-6 refuses. Telling a run caller "secret values are shown only
            # once" sends them looking for a secret that never existed.
            if action == "service.run":
                message = "Replayed a prior run; a run's output is never cached."
                # `resource_id` names the replayed run, and the hint must not
                # send the caller to `last_run` unconditionally: that blob holds
                # only the MOST RECENT run on the service, so once another run
                # has landed it describes a different execution entirely.
                # Telling them to match run_id is the difference between a
                # correct recovery and reading someone else's outcome.
                hint = (
                    "The command already executed and was NOT re-run. `resource_id` is its "
                    "run_id; `GET /services/{ident}/diagnose` reports `last_run`, which is the "
                    "MOST RECENT run — trust it only if its `run_id` matches. Otherwise issue "
                    "a fresh Idempotency-Key to run the command again."
                )
            else:
                message = "Replayed a prior response; secret values are shown only once."
                hint = "The original response (incl. any secret) is not re-issued."
            body = _envelope(
                "idempotent_replay",
                message,
                hint=hint,
                request_id=request_id_of(request),
                resource_id=getattr(record, "resource_id", None),
            )
            resp: Response = JSONResponse(
                status_code=record.response_status or 200,
                content=body,
            )
        else:
            resp = Response(
                content=record.response_body.encode("utf-8"),
                status_code=record.response_status or 200,
                media_type=record.content_type or "application/json",
            )
        resp.headers[_REPLAY_HEADER] = "true"
        return resp

    def _reject(
        self,
        request: Request,
        status_code: int,
        code: str,
        message: str,
        hint: str | None = None,
    ) -> JSONResponse:
        """Hand-build an error envelope (this middleware is outside the handlers)."""
        return JSONResponse(
            status_code=status_code,
            content=_envelope(code, message, hint=hint, request_id=request_id_of(request)),
        )

    @staticmethod
    async def _safe_delete(queries: Any, principal_id: str, idem_key: str) -> None:
        """Delete an in-progress row, swallowing errors (best-effort un-pin)."""
        try:
            await queries.delete_idempotency_record(principal_id, idem_key)
        except Exception:
            logger.warning("Failed to un-pin idempotency key", exc_info=True)

    @staticmethod
    async def _safe_interrupt(queries: Any, principal_id: str, idem_key: str) -> None:
        """Mark a claim `interrupted`, swallowing errors (best-effort pin)."""
        try:
            await queries.mark_idempotency_interrupted(principal_id, idem_key)
        except Exception:
            logger.warning("Failed to mark idempotency key interrupted", exc_info=True)
