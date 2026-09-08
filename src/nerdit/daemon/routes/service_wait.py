"""DB-backed service convergence waiting with one shared concurrency semaphore.

Imported by the services router. Reuse limits and shared service views; never
import routes.services here or duplicate the semaphore.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Query, Request

from nerdit.core.health import within_start_period as _health_within_start_period
from nerdit.core.jobconfig import parse_job_config
from nerdit.daemon.errors import NerditError
from nerdit.daemon.limits import _WAIT_TIMEOUT_MAX, WAIT_CONCURRENCY_MAX
from nerdit.daemon.schemas.service_diagnose import DiagnoseResponse
from nerdit.daemon.views.hosted import EMPTY_HOSTED, HostedContext, load_hosted_context
from nerdit.daemon.views.service import _endpoint_view, _not_found, _resolve_service
from nerdit.db.models import Job, JobKind, JobStatus, ServiceEndpoint, ServiceWaitResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Log lines folded into a failed wait. Half the `/diagnose` default
# (50): this rides a response an agent gets on every failed deploy, so the
# tail is sized to be readable in one glance rather than to be exhaustive —
# `diagnose_service(log_tail=200)` is one call away when it is not enough.
_WAIT_DIAGNOSIS_LOG_TAIL = 25

# P13 WP3 — the `/wait` converge primitive.
# Clamp bounds for the `timeout` query param (seconds).
_WAIT_TIMEOUT_MIN = 1
# DB long-poll cadence — one `get_job` per second (never the EventBus).
_WAIT_POLL_INTERVAL = 1.0
_WAIT_SEMAPHORE = asyncio.Semaphore(WAIT_CONCURRENCY_MAX)
# Statuses that are terminal for a service (a wait resolves `failed` on any).
# Locked to the §1.2 list (`failed|stopped|cancelled`); `completed` is NOT
# included — a service that reaches `completed` is unusual and the plain
# `running` convergence path already covers the healthy case.
_WAIT_TERMINAL_STATUSES = frozenset({JobStatus.failed, JobStatus.stopped, JobStatus.cancelled})


def _evaluate_wait(
    job: Job, resolved_version: int | None, version_aware: bool
) -> tuple[str | None, dict | None]:
    """Compute the wait outcome for a freshly-read row (or `None` = keep waiting).

    Returns `(outcome, last_deploy_dict_or_None)`. `outcome` is one of
    `converged`/`failed`/`superseded`, or `None` when the row has not yet
    reached a terminal condition and the poll loop should keep waiting.

    Version-aware mode (a `last_deploy` object is present) matches the phase
    machine WP2 drives; otherwise status-only mode (pre-P13 rows and
    `POST /services` rows) resolves on the live status alone.
    """
    cfg = parse_job_config(job)
    ld = cfg.get("last_deploy")
    if not isinstance(ld, dict):
        ld = None
    status = job.status

    # Model rows never carry a last_deploy phase machine
    # (ModelController._stamp_last_deploy_failed is a no-op for them), so
    # `version_aware` is always False here and the status-only branch below
    # would defer a recent `running` model through its 300 s health start
    # period to a guaranteed timeout. Readiness for a model IS the persisted
    # `model_pulled` flag, so resolve on it directly.
    if job.kind is JobKind.model:
        if status in _WAIT_TERMINAL_STATUSES:
            return "failed", ld
        if status == JobStatus.running:
            if cfg.get("model_pulled"):
                return "converged", ld
            if isinstance(job.error_message, str) and job.error_message.startswith(
                "Model pull failed"
            ):
                return "failed", ld
        # Image/weights still pulling: keep polling (an honest timeout is fine —
        # a long weights pull legitimately outlives the clamp).
        return None, ld

    # Database rows carry no last_deploy phase machine either; readiness
    # IS the persisted `db_ready` flag (set once the wire-protocol probe
    # succeeds), so resolve on it directly — the same shape as the model branch.
    if job.kind is JobKind.database:
        if status in _WAIT_TERMINAL_STATUSES:
            return "failed", ld
        if status == JobStatus.running and cfg.get("db_ready"):
            return "converged", ld
        # Still doing initdb / WAL recovery: keep polling.
        return None, ld

    if version_aware and ld is not None:
        ld_version = ld.get("version")
        if isinstance(ld_version, int) and resolved_version is not None:
            # A concurrent deploy overwrote the record: the requested version's
            # outcome is unknowable (the blob keeps only the latest). Resolve
            # immediately, never burning the timeout. `!=` (not `>`): a
            # ROLLBACK stamps a *lower* version by design, and it supersedes a
            # pending wait exactly like a redeploy would.
            if ld_version != resolved_version:
                return "superseded", ld
            if ld_version == resolved_version:
                phase = ld.get("phase")
                if phase == "healthy":
                    return "converged", ld
                if phase == "failed":
                    return "failed", ld
        # A terminal status resolves `failed` even if the phase machine has
        # not stamped `failed` yet (e.g. a stop landed on the row).
        if status in _WAIT_TERMINAL_STATUSES:
            return "failed", ld
        return None, ld

    # Status-only mode: no phase object to reason about.
    if status == JobStatus.running:
        # §1.2 converged definition: `running` AND, when a health check
        # exists, not still within its start period (health unverified). During
        # the start period the controller skips health checks entirely
        # (core/services.py), so a crash-after-Ns app would otherwise be falsely
        # reported converged the instant its container starts. Past the start
        # period a failing check drives status to `degraded` (not `running`),
        # so this only defers convergence, never blocks a healthy row.
        if _within_start_period(job):
            return None, ld
        return "converged", ld
    if status in _WAIT_TERMINAL_STATUSES:
        return "failed", ld
    return None, ld


def _within_start_period(job: Job) -> bool:
    """True when a health-checked row is still inside its start period.

    Recomputed from the DB row alone (`health_check.start_period_s` +
    `started_at`) — no controller memory. Thin delegate to
    `nerdit.core.health.within_start_period` (§1.5 item 1 family): a
    string `start_period_s` now coerces via the same tolerant path as the
    reconcile loop, instead of the isinstance guard silently degrading it to
    `0.0`.
    """
    hc = job.health_check if isinstance(job.health_check, dict) else None
    return _health_within_start_period(hc, job.started_at)


async def _diagnosis_for_failure(request: Request, job: Job) -> DiagnoseResponse | None:
    """The `/diagnose` bundle for a wait that resolved `failed`.

    Delegates to the diagnose ROUTE rather than to a re-implementation, so the
    inline bundle and `GET /diagnose` can never disagree — including on the
    owner-or-admin gate, which the route applies itself and which this function
    turns into `None` instead of a 403. That is the whole security argument:
    `/wait` stays any-authenticated, and a caller who may not read the tail
    simply does not get one.

    The import is function-local because `routes.service_diagnose` imports
    `_WAIT_TERMINAL_STATUSES` from THIS module (see the module docstring): a
    top-level import here would close that edge into a cycle.
    """
    from nerdit.daemon.routes.service_diagnose import diagnose_service as _diagnose

    ident = job.service_name or job.id
    try:
        return await _diagnose(request, ident, log_tail=_WAIT_DIAGNOSIS_LOG_TAIL)
    except NerditError:
        # Not the owner, or the row vanished between resolution and projection.
        # A wait NEVER fails because its optional companion could not be built.
        return None
    except Exception:  # noqa: BLE001 — an advisory bundle must not fail the wait
        logger.warning("Inline diagnosis failed for %r", ident, exc_info=True)
        return None


def _build_wait_response(
    request: Request,
    job: Job,
    endpoint: ServiceEndpoint | None,
    *,
    outcome: str,
    resolved_version: int | None,
    ld: dict | None,
    waited_s: float,
    hosted: HostedContext = EMPTY_HOSTED,
    diagnosis: DiagnoseResponse | None = None,
) -> ServiceWaitResponse:
    """Project a resolved wait into the four-way `ServiceWaitResponse`. `public_urls` is
    taken from the endpoint view rather than
    recomputed, so the wait response and the service detail can never disagree
    about where an app answers.
    """
    phase = ld.get("phase") if ld else None
    reason = ld.get("reason") if ld else None
    error_class = ld.get("error_class") if ld else None
    error_message = ld.get("error_message") if ld else None
    # Fall back to the row's terminal-failure columns when the phase object did
    # not carry them (e.g. a terminal status without a `failed` phase stamp).
    if error_class is None:
        error_class = job.error_class
    if error_message is None:
        error_message = job.error_message
    view = _endpoint_view(request, endpoint, hosted=hosted)
    public_url = view.public_url if view else None
    public_urls = view.public_urls if view else []
    return ServiceWaitResponse(
        outcome=outcome,
        service_name=job.service_name or job.name or job.id,
        version=resolved_version,
        phase=phase if isinstance(phase, str) else None,
        status=job.status,
        public_url=public_url,
        public_urls=public_urls,
        reason=reason if isinstance(reason, str) else None,
        error_class=error_class,
        error_message=error_message if isinstance(error_message, str) else None,
        waited_s=round(waited_s, 1),
        diagnosis=diagnosis,
    )


@router.get(
    "/services/{ident}/wait",
    response_model=ServiceWaitResponse,
    operation_id="wait_for_service",
    tags=["Services"],
)
async def wait_for_service(
    request: Request,
    ident: str,
    timeout: int = Query(60, description="Max seconds to block (clamped to [1, 300])"),
    version: str | None = Query(
        None,
        description="Deploy version to converge on; defaults to the row's latest generation",
    ),
) -> ServiceWaitResponse:
    """Block until a service converges, fails, or the timeout elapses.

    A DB-backed long-poll (one `get_job` per second — never the EventBus) that
    **always returns 200** with a four-way `outcome`
    (`converged`/`failed`/`timeout`/`superseded`). A timeout is a normal,
    agent-branchable result, not an error. Read: any authenticated principal, no
    Idempotency-Key, no audit row. Bounded: hard 300 s clamp + 1 poll/s +
    a module-level 64-wait concurrency cap.
    """
    queries = request.app.state.queries

    # Parse `version` by hand so an unparsable value is a structured 400
    # rather than FastAPI's generic 422 (the locked `bad_request` contract).
    requested_version: int | None = None
    if version is not None:
        try:
            requested_version = int(version)
        except (TypeError, ValueError) as exc:
            raise NerditError(
                400,
                "bad_request",
                f"Invalid version '{version}': expected an integer.",
            ) from exc

    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)

    timeout_s = max(_WAIT_TIMEOUT_MIN, min(_WAIT_TIMEOUT_MAX, timeout))

    # Resolve the version the wait converges on, once, at request time:
    # explicit param > last_deploy.version > build_version > status-only mode.
    cfg = parse_job_config(job)
    ld0 = cfg.get("last_deploy") if isinstance(cfg.get("last_deploy"), dict) else None
    resolved_version: int | None
    if requested_version is not None:
        resolved_version = requested_version
    elif ld0 is not None and isinstance(ld0.get("version"), int):
        resolved_version = ld0["version"]
    elif isinstance(cfg.get("build_version"), int):
        resolved_version = cfg["build_version"]
    else:
        resolved_version = None
    version_aware = ld0 is not None

    async def _endpoint_for(row: Job) -> ServiceEndpoint | None:
        return await queries.get_service_endpoint(row.service_name or "")

    async def _hosted() -> HostedContext:
        # Loaded at RESOLUTION time, not at request time: a wait
        # can block for minutes, and the answer must describe the node as it is
        # when the caller gets it — one query, on the way out.
        return await load_hosted_context(request)

    # Global concurrency cap: a saturated daemon resolves immediately as
    # `timeout, waited_s=0` (try-acquire — never `await acquire()`, which
    # would defeat the bound by queueing).
    if _WAIT_SEMAPHORE.locked():
        endpoint = await _endpoint_for(job)
        return _build_wait_response(
            request,
            job,
            endpoint,
            outcome="timeout",
            resolved_version=resolved_version,
            ld=ld0,
            waited_s=0.0,
            hosted=await _hosted(),
        )

    async with _WAIT_SEMAPHORE:
        started = time.monotonic()
        while True:
            outcome, ld = _evaluate_wait(job, resolved_version, version_aware)
            waited = time.monotonic() - started
            if outcome is not None:
                endpoint = await _endpoint_for(job)
                return _build_wait_response(
                    request,
                    job,
                    endpoint,
                    outcome=outcome,
                    resolved_version=resolved_version,
                    ld=ld,
                    waited_s=waited,
                    hosted=await _hosted(),
                    # Built only on the branch that needs it — a converged
                    # or superseded wait pays nothing for the extra reads.
                    diagnosis=(
                        await _diagnosis_for_failure(request, job) if outcome == "failed" else None
                    ),
                )
            if waited >= timeout_s:
                endpoint = await _endpoint_for(job)
                return _build_wait_response(
                    request,
                    job,
                    endpoint,
                    outcome="timeout",
                    resolved_version=resolved_version,
                    ld=ld,
                    waited_s=waited,
                    hosted=await _hosted(),
                )
            await asyncio.sleep(_WAIT_POLL_INTERVAL)
            refreshed = await _resolve_service(queries, ident)
            if refreshed is not None:
                job = refreshed
