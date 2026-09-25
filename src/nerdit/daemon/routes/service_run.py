"""Run one bounded command in a service's current image.

Use its resolved environment in a rowless, portless, GPU-less container, remove
it after completion and return HTTP 200. The route rejects invalid row/state
conditions early; the controller authoritatively enforces concurrency and launch
preconditions. Shared views keep imports independent of the services router.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Request

from nerdit.core.jobconfig import parse_job_config
from nerdit.core.runtime.protocol import ContainerRuntimeError, SandboxViolationError
from nerdit.core.services import (
    LaunchEnvNotReady,
    RunInterruptedError,
    RunPreconditionError,
    ServiceController,
)
from nerdit.core.volumes import VolumeSpecError
from nerdit.daemon.audit import audit_params, record_out_of_band
from nerdit.daemon.auth import (
    QuotaExceeded,
    current_principal,
    is_link_token_id,
    require_owner_or_admin,
)
from nerdit.daemon.errors import NerditError
from nerdit.daemon.limits import _MAX_RUN_LOG_TAIL
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.enums import TERMINAL_STATUSES
from nerdit.db.models import Job, JobKind, ServiceRunRequest, ServiceRunResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Hard cap on the audited `command_argv0` (see `_audit_argv0`). A program
# name is short; anything longer is a mis-shaped argv, not a name worth keeping
# verbatim in a trail that outlives the retention prune.
_AUDIT_ARGV0_MAX = 64


def _command_digest(command: list[str]) -> str:
    """Hex SHA-256 over the NUL-joined argv (the D-P20-5 audit fingerprint).

    NUL is the one byte an argv element cannot contain, so joining on it is
    injective: two different argv lists never collide onto one digest the way a
    space-joined `["a b"]` and `["a", "b"]` would. The digest exists so an
    auditor can correlate repeated executions of the same command — and prove
    an argv they already hold was the one that ran — without the audit trail
    ever storing the argv itself (D-H keeps audit params forever, and
    `POST /system/backup` captures the DB).
    """
    return hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest()


def _audit_argv0(command: list[str]) -> str:
    """Reduce argv[0] to a safe audit label in order: first token, basename, length cap.

    Discard whitespace-delimited arguments before capping so a mistaken shell string
    cannot retain credentials near its start. Mark token/length truncation; keep full
    argv out of the durable audit trail and correlate through its digest.
    """
    argv0 = command[0]
    head = argv0.split(None, 1)[0] if argv0.split() else argv0
    mis_shaped = head != argv0
    base = head.rsplit("/", 1)[-1]
    if len(base) > _AUDIT_ARGV0_MAX:
        return base[:_AUDIT_ARGV0_MAX] + "…(truncated)"
    return f"{base}…(argv-tail dropped)" if mis_shaped else base


def _preflight(
    request: Request, ident: str, job: Job, body: ServiceRunRequest
) -> ServiceController:
    """Refuse every run that can be refused from the row + daemon state alone.

    Ordered cheapest-first and entirely synchronous: kind, drain gate, the
    server timeout cap, a runnable image, and a run/release already in flight.
    The last two are also re-checked authoritatively inside the controller (the
    image against the runtime on a fresh row read, the slot inside
    `_register_run`) — these exist so the common refusals cost nothing beyond
    the resolve, never to be the decision.

    Returns the controller the caller then runs against, so the `None` guard
    is paid once here rather than at every use site.
    """
    if job.kind is not JobKind.service:
        raise NerditError(
            422,
            "run.not_supported",
            f"'{ident}' is a {job.kind.value}, not a service; one-off runs are service-only.",
            hint="Models and databases are platform-managed workloads: their image, "
            "command and credentials are not yours to re-enter.",
        )

    controller = getattr(request.app.state, "service_controller", None)
    if controller is None:
        # Not reachable through the real app (`server.py` always attaches the
        # controller), but a 503 is the honest answer if it ever is: the
        # subsystem that would execute the container does not exist.
        raise NerditError(
            503,
            "run.runtime_unavailable",
            "The run subsystem is unavailable on this daemon.",
            hint="Check `nerdit doctor` and the daemon log.",
        )
    if controller.draining:
        # A run started mid-drain is orphaned by the re-exec that follows it.
        raise NerditError(
            409,
            "daemon.restart_in_progress",
            "The daemon is draining for a restart; no new runs are accepted.",
            hint="Wait for the daemon to come back up, then retry.",
        )

    max_timeout = request.app.state.settings.services.run_timeout_max_s
    if body.timeout_s > max_timeout:
        raise NerditError(
            422,
            "run.timeout_too_large",
            f"timeout_s={body.timeout_s} exceeds the server cap of {max_timeout}s.",
            hint=f"Lower timeout_s, or raise [services].run_timeout_max_s "
            f"(currently {max_timeout}) on the daemon.",
        )

    if job.desired_state in TERMINAL_STATUSES:
        # The controller refuses these too, but as `service_gone` — which the
        # §1.2 table maps to 404, and whose rationale is the delete race ("the
        # row vanished between resolve and launch"). Applied to a row the user
        # merely STOPPED, that 404 is a lie the operator can disprove in one
        # command: `nerdit services stop app` then `run app -- migrate` answered
        # "No service 'app'" while `services list` showed it. Caught here so the
        # honest 409 wins, and `service_gone` → 404 goes back to meaning only
        # the genuine race it was written for.
        raise NerditError(
            409,
            "run.not_ready",
            f"Service '{ident}' is {job.desired_state}; a one-off run needs it running.",
            hint=f"Start it first (`nerdit services restart {ident}`), then retry.",
            not_ready_kind="service_stopped",
        )

    image = parse_job_config(job).get("image")
    if not isinstance(image, str) or not image:
        raise NerditError(
            409,
            "run.no_image",
            f"Service '{ident}' has no image to run.",
            hint="Deploy it first (`nerdit deploy`), or wait for the current build to finish.",
        )
    if controller.has_active_run(job.id):
        raise NerditError(
            409,
            "service.run_in_progress",
            f"Service '{ident}' already has a run or release in flight.",
            hint="Runs are single-flight per service; wait for it to finish and retry.",
        )
    return controller


@router.post(
    "/services/{ident}/run",
    response_model=ServiceRunResponse,
    operation_id="run_service_command",
    tags=["Services"],
)
async def run_service_command(
    request: Request, ident: str, body: ServiceRunRequest
) -> ServiceRunResponse:
    """Execute a bounded service command and return its outcome, including failures.

    Owner/admin only, per-token quota charged and single-flight per service. Audit
    acceptance before launch and outcome afterward. Nonzero exit and timeout return
    HTTP 200 with timing, flags and a bounded tail scrubbed of known secret values.

    Command runs unshelled, replacing image CMD and appending to ENTRYPOINT. Never
    put credentials in argv: docker inspect and last_run expose it. Use env overlays,
    whose values are masked in audits and successful/error responses.

    Per-token run slots are reserved before counting DB jobs, but concurrent workload
    creation can race that snapshot. The daemon-wide cap bounds this cross-pool race
    and also protects tokenless local/global admins.
    """
    # (D-P20-5) Explicit params, never `audit_params(body)` wholesale: the
    # full argv must NEVER enter the audit trail. Routed through
    # `audit_params` only for the redaction pass, which masks the `env`
    # map's values (`_REDACT_MAP_KEYS`) while keeping its key names.
    request.state.audit_params = audit_params(
        {
            "command_argv0": _audit_argv0(body.command),
            "command_len": len(body.command),
            "command_sha256": _command_digest(body.command),
            "env": body.env,
            "timeout_s": body.timeout_s,
            "log_tail": body.log_tail,
        }
    )

    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    require_owner_or_admin(request, job)

    controller = _preflight(request, ident, job, body)

    # (D-C) A run is charged against `max_concurrent_jobs` without inserting a
    # jobs row, so the claim is the route's to make: increment FIRST (sync, so a
    # concurrent claimer sees it), then read the DB count and compare the sum.
    # `token_id is None` is the legacy global token / no-token local admin —
    # neither is a row in `api_tokens`, so there is no per-token cap to charge
    # against and nothing to claim; the daemon-wide D-P20-2 cap still binds.
    token_id = current_principal(request).token_id
    pending = controller.claim_run_slot(token_id) if token_id is not None else 0

    # ONE release, in the finally that starts on the very next line, covering
    # every exit below — including the quota refusal itself. Releasing at the
    # raise site *as well* would decrement twice and silently inflate the
    # token's effective quota, which is the failure mode the claim exists to
    # prevent.
    try:
        # A synthetic node-link principal (`link:<node_id>`) has no
        # `api_tokens` row, and `count_active_jobs_public` is fail-closed
        # (missing row => cap 0) — charging it there would refuse EVERY
        # tunneled run. The tunnel's role ceiling is submitter and the
        # daemon-wide D-P20-2 cap (claimed above) still binds; there is no
        # per-token row cap to enforce.
        if token_id is not None and not is_link_token_id(token_id):
            active, cap = await queries.count_active_jobs_public(token_id)
            if cap is not None and active + pending > cap:
                raise QuotaExceeded(
                    "max_concurrent_jobs", limit=cap, current=active + pending
                ).to_error()

        # Audit acceptance before execution so crashes cannot erase the attempt. This
        # means accepted, not necessarily launched: controller preconditions can still
        # refuse. Correlate with the middleware outcome via request_id, since run_id is
        # not minted yet. Prefer an auditable refused attempt over an unrecorded execution.
        await record_out_of_band(
            request,
            action="service.run_started",
            target_type="service",
            target_id=job.service_name or job.id,
            params={
                "timeout_s": body.timeout_s,
                "command_argv0": _audit_argv0(body.command),
                "command_len": len(body.command),
                "command_sha256": _command_digest(body.command),
                # SUBMITTED, deliberately named as such: this row is written
                # before the controller runs, so the route cannot yet know
                # which overrides survive the D-P14-5 protected-key filter.
                # The applied/dropped split lands in `config['last_run']`.
                "env_override_keys_submitted": sorted(body.env or {}),
            },
        )

        try:
            result = await controller.run_once(
                job,
                command=list(body.command),
                env_overrides=body.env,
                timeout_s=body.timeout_s,
                log_tail=min(body.log_tail, _MAX_RUN_LOG_TAIL),
            )
        except LaunchEnvNotReady as exc:
            # Names and kinds only (the same tier `/diagnose` returns to the
            # same owner-or-admin audience), never a resolved value.
            raise NerditError(
                409,
                "run.not_ready",
                f"Service '{ident}' cannot run yet: {exc.message}",
                hint="Set the missing secrets or wait for the binding to come up, then retry.",
                # Extra envelope key (the `dependents=` precedent in
                # `service_purge.py`) — NEVER `detail=`, which would replace
                # the always-present backward-compat alias in `_envelope`.
                not_ready_kind=exc.kind,
            ) from exc
        except VolumeSpecError as exc:
            raise NerditError(
                422,
                "run.volume_invalid",
                f"Service '{ident}' has a volume spec that fails launch-time validation.",
                hint="Fix [deploy].volumes and redeploy; the run mounts exactly what a "
                "launch would.",
            ) from exc
        # ORDER IS LOAD-BEARING — do not move this handler below the
        # `ContainerRuntimeError` one. `SandboxViolationError` ⊂
        # `ContainerStartError` ⊂ `ContainerRuntimeError`
        # (`core/runtime/protocol.py`), so a base-class-first arrangement
        # makes this clause dead code and answers every sandbox denial with a
        # misleading 503.
        except SandboxViolationError as exc:
            raise NerditError(
                422,
                "run.sandbox_denied",
                "The run container was refused by the sandbox policy.",
                hint="Runs mount the service's named volumes only; host paths are never inherited.",
            ) from exc
        # Also ORDER-LOAD-BEARING, and for a reason that matters more than the
        # sandbox one: `RunInterruptedError` ⊂ `ContainerRuntimeError` too,
        # and it means the container DID start and the runtime then lost it.
        # Collapsing that into "the runtime could not execute the run" tells an
        # operator whose migration ran for two minutes exactly the wrong story —
        # they conclude nothing happened and run it again, against a
        # half-migrated schema. `core/app_build.py` already catches it ahead
        # of the base for the release path; the run path owes callers the same
        # honesty. The tail it carries is the only evidence of how far the
        # command got, and it came through the same D-P20-1 scrub as a normal
        # one, so it is as safe to return to this owner-gated caller.
        except RunInterruptedError as exc:
            raise NerditError(
                503,
                "run.interrupted",
                f"The run container for '{ident}' started and was then lost by the runtime; "
                "its exit is unobserved.",
                # Deliberately does NOT point at `last_run`: `run_once` raises
                # before it stamps `last_run`, so an interrupted run leaves
                # no stamp and `/diagnose` still shows the PREVIOUS run. The
                # `log_tail` below is the only surviving record of this one.
                hint="The command may have PARTIALLY applied its effects — data changes are "
                "not rolled back. This envelope's `log_tail` is the only record of the "
                "run: inspect it before retrying, and prefer an idempotent command.",
                container_started=True,
                log_tail=exc.log_tail,
            ) from exc
        except ContainerRuntimeError as exc:
            # Path-free by contract (the `backup.failed` precedent): the
            # runtime's own message routinely carries host paths and socket
            # locations. The real cause is in the daemon log.
            logger.warning("Run of service %s failed in the runtime", ident, exc_info=True)
            raise NerditError(
                503,
                "run.runtime_unavailable",
                "The container runtime could not execute the run.",
                hint="Check `nerdit doctor` (docker) and retry; the image may also have "
                "been pruned since the last deploy.",
            ) from exc
        except RunPreconditionError as exc:
            raise _precondition_error(
                ident,
                exc,
                max_concurrent_runs=request.app.state.settings.services.max_concurrent_runs,
            ) from exc
    finally:
        # Mirrors the claim exactly: released on every exit path, including the
        # ones that raise out of the controller (a leaked claim would shrink the
        # token's effective quota until the daemon restarts).
        if token_id is not None:
            controller.release_run_slot(token_id)

    return ServiceRunResponse(
        service_name=job.service_name or job.name or job.id,
        run_id=result.run_id,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        oom_killed=result.oom_killed,
        duration_s=result.duration_s,
        started_at=result.started_at,
        finished_at=result.finished_at,
        log_tail=result.log_tail,
    )


def _precondition_error(
    ident: str, exc: RunPreconditionError, *, max_concurrent_runs: int
) -> NerditError:
    """Map a controller-side `RunPreconditionError` to its §1.2 envelope.

    The four reasons are the authoritative re-checks behind the route's fast
    pre-flight: the row vanished (or was terminally stopped) between resolve and
    launch, the image is gone, another run/release claimed the service first, or
    the daemon-wide `[services].max_concurrent_runs` cap is saturated. An
    unknown reason falls back to the same 409 the in-progress family uses rather
    than a bare 500 — a new reason must never escape as an unstructured error
    (Invariant #3).
    """
    if exc.reason == "service_gone":
        return _not_found(ident)
    if exc.reason == "no_image":
        return NerditError(
            409,
            "run.no_image",
            f"Service '{ident}' has no runnable image.",
            hint="Deploy it first (`nerdit deploy`); the image may also have been pruned.",
        )
    if exc.reason == "too_many_runs":
        return NerditError(
            409,
            "run.too_many_in_flight",
            f"The daemon is already executing {max_concurrent_runs} one-off run(s).",
            hint=f"Wait for one to finish, or raise [services].max_concurrent_runs "
            f"(currently {max_concurrent_runs}).",
        )
    return NerditError(
        409,
        "service.run_in_progress",
        f"Service '{ident}' already has a run or release in flight.",
        hint="Runs are single-flight per service; wait for it to finish and retry.",
    )
