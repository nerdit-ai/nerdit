"""Manage desired-state services through authenticated, audited writes.

Writes use idempotency and structured errors; reads are bounded and paginated.
The controller converges desired state asynchronously. DELETE alone tears down
the container synchronously before removing its row. Mounted under /api only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query, Request, Response

from nerdit.core.volumes import VolumeSpecError, service_data_root
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    QuotaExceeded,
    current_principal,
    require_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.deploy_pipeline import reject_apex_shadow
from nerdit.daemon.errors import NerditError
from nerdit.daemon.limits import _MAX_LOG_TAIL, valid_ts_filter
from nerdit.daemon.routes.service_diagnose import router as diagnose_router
from nerdit.daemon.routes.service_logs_stream import router as logs_stream_router
from nerdit.daemon.routes.service_run import router as run_router
from nerdit.daemon.routes.service_stats import router as stats_router
from nerdit.daemon.routes.service_wait import router as wait_router
from nerdit.daemon.secret_scope import (
    name_claimed_error,
    project_owned_error,
    reject_foreign_claim,
)
from nerdit.daemon.service_purge import router as purge_router
from nerdit.daemon.views.hosted import load_hosted_context
from nerdit.daemon.views.service import (
    _not_found,
    _resolve_service,
    service_view,
)
from nerdit.db.models import (
    Job,
    JobKind,
    JobStatus,
    LogEntry,
    LogStream,
    ServiceCreateRequest,
    ServiceListPage,
    ServiceResponse,
    TokenRole,
)
from nerdit.db.queries import ProjectOwned, ServiceNameClaimed, ServiceNameTaken
from nerdit.utils.disk import du_bytes, spawn_walk

logger = logging.getLogger(__name__)

# names that collide with the secrets surface — 'shared' is the shared
# secrets scope and 'rotate-key' is the literal `POST /secrets/rotate-key`
# path. Both are valid DNS labels, so they must be rejected explicitly on
# service create and deploy.
RESERVED_SERVICE_NAMES = frozenset({"shared", "rotate-key"})

router = APIRouter()

# WP18 S4.2b: the delete/purge surface (the C2 tombstone machinery + the
# `DELETE /services/{ident}` route) lives on its own `APIRouter` in
# `daemon.service_purge` and is re-included here, BEFORE any later includes,
# so its operation_id/tag survive on the existing router object unchanged
# (façade rule, D-T-1 — `test_openapi.py` pins them).
router.include_router(purge_router)

# WP18 S4.2d: the diagnose surface (env-key-name projections,
# `_classify_bindings`, `_fresh_health_probe`, the `/diagnose` route
# itself) lives on its own `APIRouter` in `daemon.routes.service_diagnose`
# and is re-included here for the same reason (façade rule, D-T-1 —
# `operation_id="diagnose_service"` and its tag survive unchanged).
router.include_router(diagnose_router)

# the one-off run surface (`POST /services/{ident}/run` — the HTTP
# half of `ServiceController.run_once`) lives on its own `APIRouter` in
# `daemon.routes.service_run` and is re-included here for the same reason
# (façade rule, D-T-1 — `operation_id="run_service_command"` and its tag
# survive unchanged).
router.include_router(run_router)


# --- Helpers -----------------------------------------------------------------

# `_resolve_service` is re-exported (import above) for test_services_database_kind.py.


def reject_reserved_name(name: str) -> None:
    """Reject a service name reserved by the secrets surface.

    Shared with the deploy route so create and deploy enforce the same set.
    """
    if name in RESERVED_SERVICE_NAMES:
        raise NerditError(
            422,
            "service.reserved_name",
            f"'{name}' is a reserved service name.",
            hint=f"Reserved names: {', '.join(sorted(RESERVED_SERVICE_NAMES))}. Pick another.",
        )


async def _require_image_exists(runtime, image: str) -> None:
    """Reject a create whose `image` cannot be resolved by the local runtime.

    Decision #1: P2 services are register-only over a *prebuilt* image, so a
    missing image is a user error worth catching before the row is reserved.
    """
    if not await runtime.image_exists(image):
        raise NerditError(
            400,
            "image_not_found",
            f"Image '{image}' not found locally.",
            hint=f"Pull or build it first, e.g. `docker pull {image}`.",
        )


# --- Writes (desired-state; controller converges) ----------------------------


@router.post(
    "/services", response_model=ServiceResponse, status_code=201, operation_id="create_service"
)
async def create_service(request: Request, body: ServiceCreateRequest) -> ServiceResponse:
    """Register a service (register-only, prebuilt image — Decision #1).

    Writes a stable `service` row with `desired_state='running'` and status
    `building`; the `ServiceController` allocates the port/GPUs and
    launches the container on the next tick. Authorized to `submitter`/`admin`,
    idempotent (the key is persisted on the row), and audited (`service.create`).
    """
    principal = require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params(body)
    reject_reserved_name(body.name)
    reject_apex_shadow(request, body.name)
    # (P25 D-P25-3 leg b) The request name IS the row name here, so the scope
    # check runs before the image probe and long before the row write.
    require_service_scope(request, body.name)
    queries = request.app.state.queries
    # Fast path for a name another token reserved by setting its secrets
    # (P39); `reserve_service_for_token` re-checks inside its transaction.
    await reject_foreign_claim(request, body.name)

    # SANDBOX-1: script_path bind-mounts a host directory; a non-admin token has
    # no daemon-managed upload path for it in P2 (services are register-only over
    # a prebuilt image), so reject it at submit rather than letting the launch
    # trip the Tier-B allowlist. Admin/legacy/local tokens may still use it.
    if body.script_path and principal.role != TokenRole.admin:
        raise NerditError(
            403,
            "sandbox.script_path_forbidden",
            "script_path requires an admin token in P2.",
            hint="Use a prebuilt --image instead; folder/script upload arrives in P4.",
        )

    await _require_image_exists(request.app.state.runtime, body.image)

    config: dict = {"image": body.image, "port": body.port}
    if body.command:
        config["command"] = body.command
    if body.env:
        config["env"] = body.env
    if body.vendor:
        config["vendor"] = body.vendor.value

    job = Job(
        kind=JobKind.service,
        service_name=body.name,
        name=body.name,
        gpu_count=body.gpus,
        status=JobStatus.building,
        desired_state="running",
        restart_policy=body.restart_policy,
        health_check=body.health_check.model_dump() if body.health_check else None,
        script_path=body.script_path,
        config=json.dumps(config),
        submitted_by_token=principal.token_id,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )

    try:
        job = await queries.reserve_service_for_token(job, admin=principal.role is TokenRole.admin)
    except ServiceNameTaken as exc:
        raise NerditError(
            409,
            "service.name_taken",
            f"A service named '{body.name}' already exists.",
            hint="Choose a different name, or delete the existing service first.",
        ) from exc
    except ServiceNameClaimed as exc:
        raise name_claimed_error(body.name) from exc
    except ProjectOwned as exc:
        raise project_owned_error(body.name) from exc
    except QuotaExceeded as exc:
        raise exc.to_error() from exc

    # No hosted context here, deliberately: a service that was
    # just created cannot carry a share row. `PUT .../share` requires the
    # service to exist, and `delete_service_checked` drops the row inside the
    # delete transaction, so the name is share-free by construction and the
    # projection is byte-identical to the loaded one — one query saved on a
    # write path. Every OTHER service surface loads it.
    return await service_view(request, job)


@router.post("/services/{ident}/stop", response_model=ServiceResponse, operation_id="stop_service")
async def stop_service(request: Request, ident: str) -> ServiceResponse:
    """Stop a service (desired_state → `stopped`); the controller tears it down."""
    request.state.audit_params = audit_params({"service": ident})
    queries = request.app.state.queries

    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    require_owner_or_admin(request, job)

    await queries.set_desired_state(job.id, "stopped")
    job = await queries.get_job(job.id) or job
    return await service_view(request, job, hosted=await load_hosted_context(request))


@router.post(
    "/services/{ident}/restart", response_model=ServiceResponse, operation_id="restart_service"
)
async def restart_service(request: Request, ident: str) -> ServiceResponse:
    """Restart a service: force a fresh container and clear its backoff/cap.

    Sets `desired_state='running'`, marks the row `restarting`, and resets
    `restart_count`/`restart_window_start`. The reconciler then replaces a
    still-running container (the `restarting` + live path) or relaunches a
    `failed`/`stopped` one promptly — so restart is never a silent no-op.
    """
    request.state.audit_params = audit_params({"service": ident})
    queries = request.app.state.queries

    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    require_owner_or_admin(request, job)

    await queries.set_desired_state(job.id, "running")
    # Mark restarting so a *running* service is actually replaced (not just left
    # as-is) and clear the rate-cap + backoff window so the controller relaunches.
    await queries.update_job_status(job.id, JobStatus.restarting)
    await queries.bump_restart_count(job.id, 0, None)
    job = await queries.get_job(job.id) or job
    return await service_view(request, job, hosted=await load_hosted_context(request))


# --- Bounded reads -----------------------------------------------------------


@router.get("/services", response_model=ServiceListPage, operation_id="list_services")
async def list_services(
    request: Request,
    status: JobStatus | None = Query(None, description="Filter by exact status"),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
) -> ServiceListPage:
    """Cursor-paginated list of services/models (newest first, bounded to 200).

    Read is intentionally open to any authenticated principal (readonly and up).
    """
    queries = request.app.state.queries
    try:
        services, next_cursor = await queries.list_services(
            status=status, cursor=cursor, limit=limit
        )
    except ValueError as exc:
        raise NerditError(400, "bad_request", str(exc)) from exc
    # ONE share/link snapshot for the whole page, loaded before
    # the loop: a per-row hosted lookup would put a query per service on the
    # hottest read path the daemon has.
    hosted = await load_hosted_context(request)
    items = [await service_view(request, svc, hosted=hosted) for svc in services]
    return ServiceListPage(items=items, next_cursor=next_cursor)


@router.get("/services/{ident}", response_model=ServiceResponse, operation_id="get_service")
async def get_service(request: Request, ident: str) -> ServiceResponse:
    """Return a single service by id or name, or a structured 404.

    Read is intentionally open to any authenticated principal (readonly and up).
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    data_dir_bytes = await _compute_data_dir_bytes(request, job)
    hosted = await load_hosted_context(request)
    return await service_view(request, job, hosted=hosted, data_dir_bytes=data_dir_bytes)


#: Cache window for one service's data-dir size, seconds. Long enough that the
#: `nerdit logs -f` / dashboard 1 s poll collapses to one walk, short enough
#: that a size on screen still moves. A coalescing detail, not policy — so not
#: configurable, mirroring `service_stats._STATS_TTL_S`.
_DATA_DIR_TTL_S = 10.0

#: Soft budget for one walk. On overrun this request answers `null` rather than
#: blocking on a huge tree: `data_dir_bytes` is one informational field on a
#: read whose other fields are cheap. The walk itself is NOT cancelled — it
#: keeps running and fills the cache when it lands, so the next TTL window
#: reports a real number instead of re-answering `null` forever while spawning
#: a fresh walker each time.
_DATA_DIR_BUDGET_S = 5.0

#: Belt-and-braces bound on retained cache entries (expired rows are dropped on
#: every read, so the steady state is "services read in the last TTL").
_DATA_DIR_CACHE_MAX = 256

#: `app.state` attributes holding the cache and the per-name in-flight walks.
#: On the app, never module globals, so parallel test apps never share entries
#: (the `service_stats` precedent).
_DATA_DIR_CACHE_ATTR = "service_data_dir_cache"
_DATA_DIR_INFLIGHT_ATTR = "service_data_dir_inflight"


def _data_dir_cache(request: Request) -> dict[str, tuple[float, int | None]]:
    """The per-app `{service_name: (monotonic_deadline, bytes_or_None)}` cache."""
    state = request.app.state
    cache = getattr(state, _DATA_DIR_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(state, _DATA_DIR_CACHE_ATTR, cache)
    return cache


def _data_dir_inflight(request: Request) -> dict[str, asyncio.Future]:
    """The `{service_name: running walk}` registry — one walk per name at a time.

    Single-flight, but keyed by SERVICE rather than daemon-wide: a shared lock
    coalesces a hammer on one name at the price of making a slow tree block the
    detail read of every OTHER service behind it for a whole budget, which the
    route never did before it was serialized. Per-name keeps the coalescing —
    a readonly-token hammer still costs one walk per name per TTL, never one
    per request — without coupling unrelated services.
    """
    state = request.app.state
    inflight = getattr(state, _DATA_DIR_INFLIGHT_ATTR, None)
    if not isinstance(inflight, dict):
        inflight = {}
        setattr(state, _DATA_DIR_INFLIGHT_ATTR, inflight)
    return inflight


def _data_dir_cached(
    cache: dict[str, tuple[float, int | None]], name: str, now: float
) -> tuple[bool, int | None]:
    """`(hit, value)` — expired rows are pruned as a side effect.

    A `None` value is cached like a real one: a missing dir and an overrunning
    walk are the cases most likely to be polled hardest, and re-answering them
    from scratch every second is the expensive answer to a settled question.
    """
    for key, entry in list(cache.items()):
        if entry[0] <= now:
            cache.pop(key, None)
    hit = cache.get(name)
    return (False, None) if hit is None else (True, hit[1])


def _walk_result(fut: asyncio.Future) -> int | None:
    """A finished walk's value, or None when it raised."""
    try:
        return fut.result()
    except Exception:  # noqa: BLE001 — an informational field never 500s a read
        logger.debug("data_dir walk failed", exc_info=True)
        return None


async def _compute_data_dir_bytes(request: Request, job: Job) -> int | None:
    """On-disk size of a service's named-volume data dir, or None if unavailable.

    Detail-route only (P14 WP-A1): walks `<data_dir>/services/<name>` off the
    event loop, behind a `_DATA_DIR_TTL_S` cache and a PER-NAME single-flight
    registry, so N concurrent reads of one service cost at most one walk per
    TTL and a slow tree never delays another service's read. Returns `None`
    when the daemon has no `data_dir` configured, the name fails the seam's
    re-validation, the data dir does not exist yet (a row that has never mounted
    a volume — reported as `None` rather than `0` so it stays distinguishable
    from an empty dir), or the walk overran this request's budget. An overrun
    walk is left running and caches its own result, so the next read past the
    TTL reports a real number. The value may be up to one TTL stale; it is an
    informational field, not an accounting one. Never raises into the response.
    """
    settings = getattr(request.app.state, "settings", None)
    name = job.service_name
    if settings is None or not name:
        return None
    try:
        root = service_data_root(Path(settings.data_dir).expanduser(), name)
    except VolumeSpecError:
        return None
    cache = _data_dir_cache(request)
    hit, value = _data_dir_cached(cache, name, time.monotonic())
    if hit:
        return value

    inflight = _data_dir_inflight(request)
    fut = inflight.get(name)
    if fut is None:
        fut = spawn_walk(lambda: du_bytes(root) if root.exists() else None)
        inflight[name] = fut

        def _store(done: asyncio.Future, key: str = name) -> None:
            # Runs on the loop thread when the walk lands — including long
            # after the request that started it answered `null`. Writing the
            # cache here is what stops a slow tree from being re-walked every
            # TTL forever, and dropping the registry entry is what bounds the
            # walker threads to one per name.
            inflight.pop(key, None)
            if len(cache) >= _DATA_DIR_CACHE_MAX and key not in cache:
                cache.clear()
            cache[key] = (time.monotonic() + _DATA_DIR_TTL_S, _walk_result(done))

        fut.add_done_callback(_store)

    # `asyncio.wait` neither cancels nor consumes the future, so an overrun (or
    # a client that disconnects mid-read) leaves the walk running for the
    # callback above. The exception, if any, is always retrieved there.
    await asyncio.wait([fut], timeout=_DATA_DIR_BUDGET_S)
    if not fut.done():
        return None
    return _walk_result(fut)


#: `?source=` → the `job_logs.stream` values it selects. `all` maps
#: to `None` (no clause at all) so the default read is byte-identical to the
#: pre-P34 query, plan included. `system` is in neither narrowed set on
#: purpose: those lines are the DAEMON talking about the workload, not output
#: from the build or from the app, and folding them into either would make the
#: filter a lie in the other direction.
_LOG_SOURCE_STREAMS: dict[str, tuple[LogStream, ...] | None] = {
    "all": None,
    "build": (LogStream.build,),
    "runtime": (LogStream.stdout, LogStream.stderr, LogStream.crash),
}


# The PLR0913 below is one param per query filter plus the injected `Response`
# the `X-Nerdit-Scan-Watermark` header is written on. Suppressed INLINE rather
# than with a per-file entry: the WP18 splits brought routes/services.py to zero
# violations (D-T-4) and a file-wide ignore would silently cover the rest of it.
@router.get(
    "/services/{ident}/logs", response_model=list[LogEntry], operation_id="get_service_logs"
)
async def get_service_logs(  # noqa: PLR0913
    request: Request,
    response: Response,
    ident: str,
    since_id: int = Query(0),
    tail: int | None = Query(None, ge=1, le=_MAX_LOG_TAIL),
    grep: str | None = Query(
        None,
        description=(
            "Keep only lines CONTAINING this text. A literal substring, not a regex — "
            "'%' matches a percent sign. Clamped to 200 characters."
        ),
    ),
    since: str | None = Query(
        None, description="Only lines at or after this ISO-8601 UTC timestamp (inclusive)"
    ),
    source: Literal["all", "build", "runtime"] = Query(
        "all",
        description=(
            "Which log source to read. 'all' (default) is every line; 'build' is the "
            "image build's own output; 'runtime' is the container's stdout/stderr plus "
            "the crash tail. Lines written before the build stream existed are stored "
            "as stdout and therefore read as 'runtime'."
        ),
    ),
) -> list[LogEntry]:
    """Return bounded service logs to any authenticated principal.

    Every branch is bounded: tail reads the newest N matching lines, and the
    forward (paged) branch is capped at _MAX_LOG_TAIL entries in ascending id
    order. A caller that wants the rest resumes from the last returned id via
    since_id, the way the SSE twin drains its backlog.

    Apply grep, since and source inside SQL; tail counts matching lines only.
    Filtered forward pages expose the pre-scan maximum ID in
    X-Nerdit-Scan-Watermark so empty matches still advance polling; a page that
    filled the cap omits it, because the scan stopped at the bound rather than at
    the end of the range. The body remains a bare LogEntry list; unfiltered and
    tail reads omit that header. Build/app source filters exclude daemon system
    lines, which appear only under all.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    return await logs_for_job(
        request, response, job, since_id=since_id, tail=tail, grep=grep, since=since, source=source
    )


async def logs_for_job(  # noqa: PLR0913 - the bounded log query
    request: Request,
    response: Response,
    job: Job,
    *,
    since_id: int = 0,
    tail: int | None = None,
    grep: str | None = None,
    since: str | None = None,
    source: str = "all",
) -> list[LogEntry]:
    """Read bounded logs for an already resolved immutable job identity."""
    queries = request.app.state.queries
    since_ts = valid_ts_filter(since, "since") if since is not None else None
    # `.get` on a str-guarded key, not `[source]`: FastAPI owns the
    # validation (the `Literal` 422s anything else before the body runs), so
    # the only way an unknown value arrives here is a direct in-process call
    # that bypassed the signature — and such a call must degrade to the
    # unfiltered read, never raise a `KeyError` at a caller reading logs.
    streams = _LOG_SOURCE_STREAMS.get(source if isinstance(source, str) else "all")
    # Only the forward, filtered branch has a watermark to offer: `tail`
    # takes precedence over `since_id` in the query layer and walks backwards
    # from the newest row, so nothing about a pre-scan maximum is skippable
    # there. Captured BEFORE the scan — anything written after this read gets a
    # strictly larger id, which is what makes skipping to it safe.
    paged = (grep or since_ts) and not (tail is not None and tail > 0)
    watermark = (
        await queries.max_log_id()
        if paged and current_principal(request).project_id is None
        else None
    )
    # The forward branch carries the SAME cap as `tail` (the tail branch ignores
    # `limit`): without one, a parameterless GET returns the whole retained
    # history of a chatty service — tens of MB materialized as `LogEntry` objects
    # and re-serialized, while the scan holds the shared connection.
    entries = await queries.get_logs(
        job.id,
        since_id=since_id,
        tail=tail,
        grep=grep,
        since_ts=since_ts,
        streams=streams,
        limit=_MAX_LOG_TAIL,
    )
    # A page that filled the cap stopped at the bound, not at the end of the
    # range, so the pre-scan maximum is NOT decided yet — handing it to a
    # follower would skip the undrained remainder. Same rule as the stream's
    # `len(entries) < _POLL_CHUNK` short-page test.
    if watermark is not None and len(entries) < _MAX_LOG_TAIL:
        response.headers["X-Nerdit-Scan-Watermark"] = str(watermark)
    return entries


# WP18 S4.2c: the `/wait` surface lives on its own `APIRouter` in
# `daemon.routes.service_wait` and is re-included here, at the bottom, so
# every test app built from this module's `router` carries the wait route
# unchanged (façade rule, D-T-1 — `operation_id="wait_for_service"` and its
# tag survive; `test_openapi.py` pins them).
router.include_router(wait_router)

# the per-service container stats surface (`GET
# /services/{ident}/stats` — the on-demand `runtime.stats` sample behind its
# 2 s TTL cache) lives on its own `APIRouter` in
# `daemon.routes.service_stats` and is re-included here for the same reason
# (façade rule, D-T-1 — `operation_id="get_service_stats"` and its tag
# survive; `test_openapi.py` pins them).
router.include_router(stats_router)

# the resumable log stream (`GET /services/{ident}/logs/stream` — the
# DB-backed poll, never a pinned docker follow) lives on its own `APIRouter`
# in `daemon.routes.service_logs_stream` and is re-included here for the same
# reason (façade rule, D-T-1 — `operation_id="stream_service_logs"` and its
# tag survive; `test_openapi.py` pins them).
router.include_router(logs_stream_router)
