"""Classify service remediation deterministically from precomputed facts.

First match wins: fresh OOM/GPU OOM; unresolved model/secret bindings; invalid
or unresolved edge auth; host build failure; rollback; privilege-denied crash
loop; other user-error crash loop; degraded health; transient pull failure;
pending/disabled ACME; other failure; no failure.

Forensics rules require last_crash_at >= last_deploy.started_at so an old crash
cannot outrank the current deployment. Edge-auth and ACME rules also apply to
healthy containers: routing and certificate availability are separate from
launch health. ACME ranks below concrete failures and above inspect_logs.
The route performs I/O and passes classifications here; this module stays pure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from nerdit.core.remediation_settle import (
    SETTLE_FIX_START_COMMAND,
    SETTLE_IMAGE_NEEDS_PRIVILEGES,
    SETTLE_RAISE_MEMORY_LIMIT,
    forensics_from_config,
    settle_remediation_code,
)
from nerdit.core.remediation_settle import forensics_fresh as _forensics_fresh
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

# The settle-time subset lives in `core` (`deploy_state` is
# its caller and `core` never imports `daemon`); re-exported here so the
# diagnose route and the tests keep one import site. `_SETTLE_LITERALS` is
# pinned against `RemediationCode` by `tests/test_diagnose.py`.
__all__ = [
    "AcmeWait",
    "BindingWait",
    "EdgeAuthWait",
    "RemediationCode",
    "derive_remediation",
    "forensics_from_config",
    "settle_remediation_code",
]
_SETTLE_LITERALS = (
    SETTLE_RAISE_MEMORY_LIMIT,
    SETTLE_IMAGE_NEEDS_PRIVILEGES,
    SETTLE_FIX_START_COMMAND,
)


class RemediationCode(str, Enum):
    """Stable machine remediation codes, paired with actionable detail.

    Actions use existing config, model, secret, restart and rollback APIs.
    Retry_later, inspect_logs, none, platform.build_unavailable and acme.cert_pending
    are advice-only: host repair and certificate waiting cannot be forced by API.
    """

    raise_memory_limit = "raise_memory_limit"
    fix_start_command = "fix_start_command"
    serve_missing_model = "serve_missing_model"
    set_missing_secret = "set_missing_secret"
    fix_health_check = "fix_health_check"
    rollback = "rollback"
    # (P21 D2) A kind=model row whose crash tail matched a CUDA/HIP OOM: the fix
    # is GPU sizing (context length / VRAM fraction), NEVER [deploy].memory_limit.
    # Maps to POST /models (re-serve with per-serve bounds) or
    # [models].vllm_extra_args. The dotted value is the D2-locked literal.
    model_gpu_oom = "model.gpu_oom"
    # P14 WP-A1: an invalid or conflicting [deploy].volumes spec — fix it via
    # PUT /config/apps/{name}/deploy (or the app's nerdit.toml) and redeploy.
    volume_invalid = "volume_invalid"
    # A kind=model row whose weights pull failed permanently: the bare model
    # server still answers its liveness probe, so it needs its own signal
    # (maps to POST /services/{name}/restart to re-attempt the pull).
    model_pull_failed = "model_pull_failed"
    # A kind=database row that is not answering its readiness probe: the
    # minted credential and data dir are coupled, so recreate (or restore a
    # backup) rather than expecting a fresh initdb on a non-empty PGDATA.
    # (P25 D-P25-5/D-P25-8) The two edge-auth codes. Dotted literals, the
    # `model.gpu_oom` precedent: they name a config SECTION's failure, not a
    # generic verb, and the two are split because the operator actions differ —
    # set a secret vs repair the declaration. Both mean the same thing about
    # reachability: the route is withheld until it is fixed, never published
    # unprotected.
    edge_auth_secret_missing = "edge_auth.secret_missing"
    edge_auth_invalid = "edge_auth.invalid"
    db_not_ready = "db_not_ready"
    # (BUG-1) The build failed for a HOST reason (no BuildKit builder). Terminal
    # advice by design — the fix is on the machine, and no API action can apply
    # it. The fourth terminal code alongside retry_later/inspect_logs/none.
    platform_build_unavailable = "platform.build_unavailable"
    # (P26 WP2 / S-W2-7) The two ACME codes. Dotted literals naming the config
    # SECTION, the `model.gpu_oom`/`edge_auth.*` precedent, and split
    # because the operator actions differ: one is a wait to investigate
    # (DNS/reachability), the other is a setting to flip. Neither ever carries
    # the ACME account email or the directory URL.
    acme_cert_pending = "acme.cert_pending"
    acme_disabled = "acme.disabled"
    # (P33, field failure 2026-08-23) The image's ENTRYPOINT needs a root
    # capability this sandbox drops (`cap_drop=["ALL"]` + no-new-privileges),
    # so it dies before the app ever runs — a stock `FROM nginx` static site
    # is the canonical case. Ranked above `fix_start_command` because the
    # start command is fine and editing it can never fix this; the fix is the
    # IMAGE (a non-root variant, or a non-root `USER` owning its own files).
    image_needs_privileges = "image_needs_privileges"
    retry_later = "retry_later"
    inspect_logs = "inspect_logs"
    none = "none"


@dataclass(frozen=True)
class BindingWait:
    """Read-only classification of a fresh `resolve_bindings` run (§1.3).

    Built by the route from an idempotent per-binding resolve pass — it records
    *whether* the app is blocked and *which kind* of wait it is (an ollama model
    still un-served vs a missing/undecryptable secret), never any value. The
    remediation precedence needs the kind to pick rule 2 vs rule 3.
    """

    waiting: bool = False
    model_wait: bool = False
    secret_wait: bool = False
    messages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EdgeAuthWait:
    """Read-only edge-auth grammar/secret classification computed by the route.

    Binding waits block launch; edge-auth waits withhold routing while containers
    may run. Reuse proxy checks and carry only field/key names, never values.
    Invalid grammar and unresolved-reference states are mutually exclusive.
    """

    invalid_fields: tuple[str, ...] = field(default_factory=tuple)
    """Field names from ``EdgeAuthInvalid`` — ``()`` when the blob parsed."""
    missing_key: str | None = None
    """The referenced secret KEY that did not resolve; ``None`` when it did."""
    shared_scope: bool = False
    """True when the unresolved reference named the shared scope."""
    over_length_key: str | None = None
    """The KEY whose resolved value exceeds bcrypt's 72-byte input limit.

    Mirrors ``edgeauth.password_exceeds_bcrypt_limit`` — the same predicate
    ``hash_password`` refuses on — so diagnose reports the withheld route the
    route plane is actually withholding. Distinct from ``invalid_fields``
    because the FIX is different: a shorter secret, never the declaration.
    """


@dataclass(frozen=True)
class AcmeWait:
    """Read-only certificate classification from domain rows and proxy cert status.

    ACME waits affect certificate trust, not launch or route publication. The route
    performs storage reads; this holds domain names only, never account email or CA
    URL. Pending and disabled domain sets are disjoint.
    """

    pending_domains: tuple[str, ...] = field(default_factory=tuple)
    """Domains asking for a public certificate that have no issued leaf yet.

    Includes ``expired`` as well as ``pending``: an expired leaf is a renewal
    that is not happening, which is the same operator problem with the same
    causes (DNS, reachability) and the same fix.
    """
    disabled_domains: tuple[str, ...] = field(default_factory=tuple)
    """Domains carrying ``acme=1`` on a node whose ``[proxy.acme]`` is off."""


# Substrings that mark a pull error as transient (retry-worthy) rather than a
# permanent user/registry error. Conservative: an empty/absent message on a
# weights pull is treated as transient (network flakes are the common cause);
# a permanent-shaped message (not found / unauthorized / manifest) is not.
_TRANSIENT_MARKERS = (
    "reset",
    "timeout",
    "timed out",
    "unreachable",
    "temporarily",
    "connection",
    "refused",
    "eof",
    "try again",
    "rate limit",
    "too many requests",
    "503",
    "502",
    "500",
)
_PERMANENT_MARKERS = (
    "not found",
    "no such",
    "unauthorized",
    "forbidden",
    "denied",
    "manifest",
    "invalid",
    "does not exist",
)

_TERMINAL_STATUSES = frozenset({JobStatus.failed, JobStatus.stopped, JobStatus.cancelled})

# The reconcile loop defers health/readiness judgement while a row is inside its
# health-check start period (mirrors `core.services._DEFAULT_START_PERIOD_S` and
# the wait route's `_WAIT_DEFAULT_START_PERIOD_S`); a database still running
# initdb inside that window is expected to be un-ready and must not be flagged
# stuck (Rule 7b).
_DEFAULT_START_PERIOD_S = 0.0


def _past_start_period(job: Job) -> bool:
    """True when a row has outlived its health-check start-period grace window.

    Recomputed from the row alone (`health_check.start_period_s` +
    `started_at`), mirroring the reconcile loop's grace gate
    (`core.services`) and the wait route's `_within_start_period` — inverted:
    a row still inside the window is expected to be un-ready and must not be
    classified as stuck. A row with no `started_at` has not been given its
    grace yet, so it is treated as *not* past it (conservative — no fire).
    """
    if job.started_at is None:
        return False
    hc = job.health_check if isinstance(job.health_check, dict) else {}
    start_period = hc.get("start_period_s")
    period_s = (
        float(start_period) if isinstance(start_period, (int, float)) else _DEFAULT_START_PERIOD_S
    )
    started = job.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - started).total_seconds() >= period_s


def _looks_transient(message: object) -> bool:
    """Heuristic: is a recorded pull error transient (retry-worthy) vs permanent?"""
    if not isinstance(message, str) or not message.strip():
        # A weights/image pull that failed without a captured message is most
        # often a network flake — advise retry rather than a source fix.
        return True
    low = message.lower()
    if any(marker in low for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in low for marker in _TRANSIENT_MARKERS)


def _is_failure_state(
    job: Job,
    last_deploy: dict | None,
    forensics: dict,
    binding_wait: BindingWait,
    *,
    fresh: bool,
) -> bool:
    """True when the row is in *any* failure/blocked condition worth diagnosing."""
    if job.status in _TERMINAL_STATUSES or job.status is JobStatus.degraded:
        return True
    if last_deploy is not None and last_deploy.get("phase") == "failed":
        return True
    if binding_wait.waiting:
        return True
    exit_code = forensics.get("last_exit_code")
    if fresh and isinstance(exit_code, int) and exit_code != 0:
        return True
    return False


def _wait_rules(
    name: str, binding_wait: BindingWait, edge_auth: EdgeAuthWait | None
) -> tuple[RemediationCode, str] | None:
    """Classify launch dependencies before edge-auth route dependencies.

    Return None when no rule matches. Details contain binding messages or field/key
    names only, never malformed password values.
    """
    # Rule 2 — a fresh binding wait on an ollama model (serve it).
    if binding_wait.waiting and binding_wait.model_wait:
        return RemediationCode.serve_missing_model, (
            binding_wait.messages[0]
            if binding_wait.messages
            else "An [ai.*] binding needs a model that is not served — run: nerdit serve <model>"
        )

    # Rule 3 — a fresh binding wait on a secret ref / secret-decrypt (set it).
    if binding_wait.waiting and binding_wait.secret_wait:
        return RemediationCode.set_missing_secret, (
            binding_wait.messages[0]
            if binding_wait.messages
            else (
                f"An [ai.*] binding needs a secret that is not set — run: "
                f"nerdit secrets set {name} <KEY>=..."
            )
        )

    if edge_auth is None:
        return None
    # Rule 3b (P25, D-P25-8) — the persisted blob is declared but unusable.
    if edge_auth.invalid_fields:
        fields = ", ".join(edge_auth.invalid_fields)
        return (
            RemediationCode.edge_auth_invalid,
            f"[deploy].edge_auth is declared but malformed ({fields}; values not "
            f"shown), so the route is withheld — the app is never published "
            f"unprotected. Repair it via PUT /config/apps/{name}/deploy "
            f"(`nerdit config app set {name} deploy edge_auth=...`) or in the "
            f"app's nerdit.toml + redeploy — password must be a "
            f"${{secrets.KEY}} reference, never a literal. To drop edge auth "
            f"entirely: `nerdit config app set {name} deploy edge_auth=null`.",
        )
    # Rule 3b′ (P25, review-upheld) — well-formed and RESOLVED, but the value
    # cannot be bcrypt-hashed (> 72 bytes), so `hash_password` refuses on
    # every route build. The declaration is fine; the FIX is a shorter
    # secret, so the code is `edge_auth.invalid` (the route plane raised
    # EdgeAuthInvalid) but the detail must not point at the declaration.
    if edge_auth.over_length_key:
        return (
            RemediationCode.edge_auth_invalid,
            f"[deploy].edge_auth references secret "
            f"'{edge_auth.over_length_key}' whose value exceeds bcrypt's "
            f"72-byte input limit, so the route is withheld — the app is "
            f"never published unprotected. The declaration is fine; set a "
            f"shorter value: nerdit secrets set {name} "
            f"{edge_auth.over_length_key}=...",
        )
    # Rule 3c (P25, D-P25-8) — well-formed, but the reference resolves nowhere
    # it is allowed to.
    if edge_auth.missing_key:
        scope = "--shared" if edge_auth.shared_scope else name
        return (
            RemediationCode.edge_auth_secret_missing,
            f"[deploy].edge_auth references secret '{edge_auth.missing_key}', which "
            f"is not set — the route is withheld until it resolves (the app is "
            f"never published unprotected). Run: nerdit secrets set {scope} "
            f"{edge_auth.missing_key}=...",
        )
    return None


def _acme_rules(acme: AcmeWait | None) -> tuple[RemediationCode, str] | None:
    """Classify pending then disabled ACME after failures and before inspect_logs.

    Details include domain and config names, never account email or directory URL.
    """
    if acme is None:
        return None
    if acme.pending_domains:
        names = ", ".join(acme.pending_domains)
        return (
            RemediationCode.acme_cert_pending,
            f"{len(acme.pending_domains)} ACME domain(s) without an issued "
            f"certificate: {names}. Those names do NOT complete TLS handshakes "
            f"until the CA issues — the internal CA is not a fallback for a "
            f"name that asks for a public certificate. Check GET /doctor "
            f"(acme_http_port), that public DNS for those names points at this "
            f"node, and that [proxy.acme].http_port is reachable from the "
            f"internet; the proxy log records the CA's response.",
        )
    if acme.disabled_domains:
        names = ", ".join(acme.disabled_domains)
        return (
            RemediationCode.acme_disabled,
            f"{len(acme.disabled_domains)} domain(s) request a public "
            f"certificate but [proxy.acme].enabled is false: {names}. They are "
            f"served with this node's internal CA. Enable ACME (PUT "
            f'/config/daemon/proxy {{"acme": {{"enabled": true, "email": "..."}}}} '
            f"then restart the daemon) or re-PUT the domain with acme=false.",
        )
    return None


def _crash_loop_rules(
    name: str, forensics: dict, *, fresh: bool, crash_loop: bool
) -> tuple[RemediationCode, str] | None:
    """Classify fresh privilege-denied crashes before generic command failures.

    Return None outside fresh user-error crash loops. Details contain exit codes
    and config keys, never the potentially secret-bearing crash tail.
    """
    exit_code = forensics.get("last_exit_code")
    if not (fresh and crash_loop and isinstance(exit_code, int) and 1 <= exit_code <= 126):
        return None

    # Rule 4b — the crash tail showed a dropped-privilege refusal.
    if forensics.get("priv_denied"):
        return (
            RemediationCode.image_needs_privileges,
            "The image's entrypoint needs root capabilities the sandbox drops "
            "([containers].drop_all_caps + no_new_privileges), so it dies before your "
            "app runs — [deploy].start cannot fix this. Use a non-root image — e.g. "
            "nginxinc/nginx-unprivileged (listens on 8080) for static sites — or set a "
            "non-root USER in your Dockerfile that owns the files the entrypoint "
            f"writes; then PUT /config/apps/{name}/deploy (port) and redeploy.",
        )

    # Rule 5 — an ordinary fresh crash loop with a user-error exit code.
    return (
        RemediationCode.fix_start_command,
        f"Container keeps exiting (exit {exit_code}); check [deploy].start and fix it via "
        f"PUT /config/apps/{name}/deploy, then inspect the log tail below.",
    )


def derive_remediation(  # noqa: PLR0913 - one parameter per LOCKED input class
    job: Job,
    cfg: dict,
    forensics: dict,
    binding_wait: BindingWait,
    probe: dict | None,
    edge_auth: EdgeAuthWait | None = None,
    *,
    acme: AcmeWait | None = None,
) -> tuple[RemediationCode, str]:
    """Return the first matching remediation code and actionable detail without I/O.

    Use precomputed forensics and binding/edge-auth/ACME classifications. None means
    unclassified. Probe remains accepted for caller compatibility; degraded status,
    not that probe, drives the health rule.
    """
    name = job.service_name or job.name or job.id
    last_deploy = cfg.get("last_deploy") if isinstance(cfg.get("last_deploy"), dict) else None
    reason = last_deploy.get("reason") if last_deploy else None
    action = last_deploy.get("action") if last_deploy else None
    fresh = _forensics_fresh(forensics, last_deploy)

    # Rule 1 (P21 D2, architect ruling 2026-08-06) — a fresh CUDA/HIP OOM on a
    # model row, checked BEFORE the cgroup rule. In the rare both-flags case
    # (oom_killed AND gpu_oom — e.g. a CUDA OOM whose CPU/pinned-memory fallback
    # balloons host RAM into a cgroup kill) the allocator failure is the root
    # cause and GPU sizing is the actionable knob; Rule 1b's
    # `[deploy].memory_limit` guidance is app-shaped and not actionable on a
    # model row. This keeps `error.class GPU_OOM` ↔ `model.gpu_oom`
    # one-to-one (docs/guide/troubleshooting.md) and ALIGNS with
    # `_classify_service_exit` (core/services.py), where gpu_oom likewise
    # outranks oom_killed. Service rows never reach this rule (D2 scoping) and
    # fall through to the cgroup rule below.
    if fresh and forensics.get("gpu_oom") and job.kind is JobKind.model:
        return (
            RemediationCode.model_gpu_oom,
            "The model server ran out of GPU memory (CUDA/HIP OOM — not a container "
            "memory kill). Re-serve with smaller bounds: POST /models with "
            "max_model_len / gpu_memory_utilization (`nerdit serve <model> --backend "
            "vllm --max-model-len 4096 --gpu-memory-utilization 0.8`), or set "
            "[models].vllm_extra_args (--max-model-len / --gpu-memory-utilization; "
            "daemon restart required). Raising [deploy].memory_limit will NOT fix this.",
        )

    # Rule 1b — cgroup OOM on the fresh last crash.
    if fresh and forensics.get("oom_killed"):
        limit = cfg.get("memory_limit")
        at = f" at {limit}" if isinstance(limit, str) and limit else ""
        return (
            RemediationCode.raise_memory_limit,
            f"Container was OOM-killed{at}; raise [deploy].memory_limit via "
            f"PUT /config/apps/{name}/deploy.",
        )

    # Rules 2, 3, 3b, 3c — the "declared dependency not usable yet" family, in
    # table order (see `_wait_rules`; extracted only for the complexity
    # gate, the ranking is unchanged).
    wait_verdict = _wait_rules(name, binding_wait, edge_auth)
    if wait_verdict is not None:
        return wait_verdict

    # Rule 3e (BUG-1) — a build that failed on a HOST fault outranks Rule 4's
    # rollback: rolling back restores service but says nothing about why every
    # future build on this node will fail the same way. The rollback pointer is
    # MERGED into this detail when a previous image exists, so the operator
    # loses nothing by the re-ranking.
    #
    # Row column FIRST, then the phase object — the same precedence
    # `diagnose_service` uses for `error.class`. The fallback is what makes
    # the rule fire on a REDEPLOY at all: the revert branch of
    # `_settle_failed_generation` settles the row back to `running` through
    # writers that deliberately NULL `error_class` (F2-STALE-ERR, so a healthy
    # generation never surfaces a prior one's error), leaving the class only in
    # `config['last_deploy']`.
    ld_class = last_deploy.get("error_class") if last_deploy else None
    if job.error_class is ErrorClass.platform_error or ld_class == ErrorClass.platform_error.value:
        roll = (
            f" The previous image is still serving; POST /deploy/{name}/rollback "
            "restores it meanwhile."
            if cfg.get("previous_image")
            else ""
        )
        # COUPLING: `ErrorClass.platform_error` is generic ("a HOST fault"),
        # and this detail now covers BOTH of its setters — the
        # `runtime/docker.py` `build_image` early guard (no docker CLI on
        # the daemon's PATH) and its buildx branch (no BuildKit builder). Both
        # are build-TOOLCHAIN faults, which is why one message can serve them.
        # A third, non-toolchain platform fault must BRANCH here rather than
        # widen this string again.
        #
        # (Codex 3804646875) The resubmit sentence is load-bearing, not polish:
        # `core/app_build.py`'s `BuildPlatformError` branch settles the
        # generation terminally and its enclosing `finally` drops the build
        # context on either outcome, so nothing re-enters the builder once the
        # host is repaired. Without it, an operator who follows these steps to
        # the letter leaves the service failed forever while this very detail
        # keeps describing the fault in the present tense.
        return (
            RemediationCode.platform_build_unavailable,
            "The image build failed on a HOST fault, not in your source: this "
            "daemon's build toolchain is unavailable — no BuildKit builder "
            "(docker buildx), or no docker CLI on the daemon's PATH. Install "
            "the buildx CLI plugin system-wide, or set DOCKER_CONFIG / PATH in "
            "the daemon's service unit, then confirm with `nerdit doctor` "
            "(docker row). The failed build cannot resume — submit the deploy "
            f"again once that row is clean.{roll}",
        )

    # Rule 4 — a redeploy build failure with a rollback target still serving.
    if reason == "build_failed" and action == "redeploy" and cfg.get("previous_image"):
        version = last_deploy.get("version") if last_deploy else None
        vtag = f" v{version}" if isinstance(version, int) else ""
        return (
            RemediationCode.rollback,
            f"Build{vtag} failed; the previous image is still serving. Roll back with "
            f"POST /deploy/{name}/rollback, or fix the source and redeploy.",
        )

    # Rules 4b/5 — a fresh crash loop, split on whether the captured tail showed
    # the sandbox refusing the entrypoint a root privilege (see
    # `_crash_loop_rules`; extracted only for the complexity gate, the
    # ranking is the table's).
    crash_loop = reason == "crash_loop" or (
        job.status is JobStatus.restarting and job.restart_count >= 2
    )
    crash_verdict = _crash_loop_rules(name, forensics, fresh=fresh, crash_loop=crash_loop)
    if crash_verdict is not None:
        return crash_verdict

    # Rule 6 — a running-but-degraded service (health check failing).
    if job.status is JobStatus.degraded:
        return (
            RemediationCode.fix_health_check,
            f"Service is running but failing its health check; fix [deploy].health via "
            f"PUT /config/apps/{name}/deploy (or the app's /health handler).",
        )

    # Rule 6b (P14 WP-A1) — an invalid/conflicting named-volume spec.
    if reason in ("volume_invalid", "volume_conflict"):
        return (
            RemediationCode.volume_invalid,
            f"A named volume is invalid or conflicts with another mount; fix "
            f"[deploy].volumes via PUT /config/apps/{name}/deploy (names are "
            "DNS labels, paths absolute; <=8), then redeploy.",
        )

    # Rule 7a — a kind=model row whose weights pull failed. Model rows never
    # carry a last_deploy phase machine (ModelController._stamp_last_deploy_failed
    # is a documented no-op for them), so Rule 7's `reason == 'model_pull_failed'`
    # can never fire here; the persisted signal is `job.error_message` set by
    # set_job_error_message with the row left running and its liveness probe 200.
    if (
        job.kind is JobKind.model
        and not cfg.get("model_pulled")
        and isinstance(job.error_message, str)
        and job.error_message.startswith("Model pull failed")
    ):
        if _looks_transient(job.error_message):
            return (
                RemediationCode.retry_later,
                "The model weights pull failed transiently; retry with "
                f"POST /services/{name}/restart (`nerdit services restart {name}`).",
            )
        return (
            RemediationCode.model_pull_failed,
            "The model weights pull failed — check the model reference, then retry "
            f"with POST /services/{name}/restart (`nerdit services restart {name}`).",
        )

    # Rule 7b — a kind=database row still not answering its readiness
    # probe. Keyed on STATE, not an error-message prefix: the v1 backends leave a
    # stuck row RUNNING (or RESTARTING) with config['db_ready'] unset and NO
    # error message — a transient DataNotReadyError just re-arms the next tick,
    # and the "Database provisioning failed" message is reserved for the future
    # dump/restore substrate (never written in v1). Fire only once the row has
    # outlived its start-period grace so a database still running initdb is never
    # misreported as stuck. This fires regardless of _is_failure_state (the row
    # stays running), mirroring Rule 7a. Point the agent at recreate/restore (a
    # stuck non-empty PGDATA never re-inits) — never a value.
    if (
        job.kind is JobKind.database
        and job.status in (JobStatus.running, JobStatus.restarting)
        and not cfg.get("db_ready")
        and _past_start_period(job)
    ):
        return (
            RemediationCode.db_not_ready,
            "The database is not accepting connections; inspect the log tail below. "
            "If it is stuck, recreate it (POST /databases) or restore a backup — the "
            "minted credential and data dir are coupled and a non-empty PGDATA never "
            "re-initializes.",
        )

    # Rule 7 — a transient-shaped image/model pull failure. The two conditions
    # are one `and` rather than nested `if`s purely for the complexity gate
    # (P26 WP2 added rules 7c/7d below); the predicate is unchanged.
    if reason in ("image_pull_failed", "model_pull_failed") and _looks_transient(
        last_deploy.get("error_message") if last_deploy else None
    ):
        return (
            RemediationCode.retry_later,
            "The image/model pull failed transiently; retry the deploy or "
            f"`nerdit serve` shortly (POST /services/{name}/restart or re-POST).",
        )

    # Rules 7c/7d — the public certificate is missing or was never
    # going to be issued. Evaluated regardless of `_is_failure_state` (rules
    # 7a/7b's property): the container is healthy, so nothing above would fire
    # and the answer would otherwise be `none` — "no remediation needed" on a
    # node whose operator is staring at a browser warning. Ranked BELOW every
    # genuine failure so a fresh crash is still the answer to "what is wrong".
    acme_verdict = _acme_rules(acme)
    if acme_verdict is not None:
        return acme_verdict

    # Rule 8/9 — any other failure → inspect the logs; otherwise nothing to do.
    if _is_failure_state(job, last_deploy, forensics, binding_wait, fresh=fresh):
        return (
            RemediationCode.inspect_logs,
            "The failure has no automatic fix — inspect the log tail below and the "
            "recorded error class/message.",
        )
    return RemediationCode.none, "No remediation needed — the service is healthy."
