"""Build owner-authorized diagnosis from forensics, probes, bindings and logs.

Share service views and wait constants without importing routes.services.
Health probes remain module lookups; response type describes the probe run,
not an unvalidated stored type.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request

from nerdit.config.project import SECRET_REF_RE, shared_secret_keys_for
from nerdit.core.bindings.registry import BINDING_KINDS
from nerdit.core.bindings.secretref import walk_secret_ref
from nerdit.core.data.binding import resolve_db_binding
from nerdit.core.health import (
    DEFAULT_HEALTH_TIMEOUT_S,
    as_float,
    check_health,
    check_tcp,
    run_probe,
)
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.models.binding import (
    BindingNotReady,
    inject_env_key_names,
    resolve_binding,
)
from nerdit.core.proxy.edgeauth import (
    EdgeAuthInvalid,
    load_edge_auth,
    password_exceeds_bcrypt_limit,
)
from nerdit.core.secrets import SHARED_SCOPE, SecretDecryptError
from nerdit.core.services import ServiceController
from nerdit.daemon.auth import require_owner_or_admin
from nerdit.daemon.limits import _MAX_DIAGNOSE_TAIL
from nerdit.daemon.remediation import (
    AcmeWait,
    BindingWait,
    EdgeAuthWait,
    derive_remediation,
    forensics_from_config,
)
from nerdit.daemon.routes.service_wait import _WAIT_TERMINAL_STATUSES
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import (
    DiagnoseBindings,
    DiagnoseBuild,
    DiagnoseError,
    DiagnoseForensics,
    DiagnoseHealth,
    DiagnoseRemediation,
    DiagnoseResponse,
    DiagnoseRestarts,
    GpuVendor,
    Job,
    JobKind,
    ServiceEndpoint,
)

router = APIRouter()


# --- Helpers -----------------------------------------------------------------


def _ai_env_key_names(ai_specs: object) -> set[str]:
    """Env-var NAMES `inject_env` would produce for these `[ai.*]` specs.

    Delegates to `nerdit.core.models.binding.inject_env_key_names` so the
    key grammar (`NERDIT_AI_<NAME>_URL/_KEY/_MODEL` per binding + the plain
    `OPENAI_*` triplet for `default`) lives in exactly one place (F7). Reports
    the injected surface without a resolve that could raise on a not-ready binding.

    The P15 registry drives `_pending_env_key_names` uniformly through
    `BindingKind.key_names_fn` (the `[ai.*]` entry IS this delegate's callee);
    this thin wrapper is retained as the named F7 seam the AI-contract freeze
    tests pin (`tests/test_ai_binding_resolve.py`).
    """
    if not isinstance(ai_specs, dict):
        return set()
    return inject_env_key_names(ai_specs)


def _model_env_key_names(request: Request, job: Job, cfg: dict) -> list[str]:
    """Best-effort backend container env NAMES for a model row (recompute fallback).

    Model rows discard the service env assembly (`services.py` `_launch`), so the
    injected surface is the backend-composed container env (§1.3: "backend
    container env names + declared `launch_env_secret_keys` actually present").
    The declared launch secret keys (e.g. `HF_TOKEN` for gated vLLM) are only
    injected at launch when present in the shared store, so we union in the ones
    that are actually present now. Best-effort: any failure degrades to what was
    computed so far (the persisted `last_launch_env_keys` is the accurate
    source once the row has launched).
    """
    mc = getattr(request.app.state, "model_controller", None)
    if mc is None:
        return []
    names: set[str] = set()
    try:
        config = mc.build_container_config(
            str(cfg.get("model") or ""),
            [],
            GpuVendor.nvidia,
            0,
            backend_name=cfg.get("backend"),
        )
        names.update((config.env or {}).keys())
    except Exception:  # noqa: BLE001 — a projection must never fail the diagnose read
        pass
    # Declared launch secret keys actually present in the shared store.
    secret_mgr = getattr(request.app.state, "secret_manager", None)
    try:
        backend = mc.backend_for(cfg)
        wanted = tuple(getattr(backend, "launch_env_secret_keys", ()))
        if wanted and secret_mgr is not None:
            present = set(secret_mgr.list_keys(SHARED_SCOPE))
            names.update(k for k in wanted if k in present)
    except Exception:  # noqa: BLE001 — a projection must never fail the diagnose read
        pass
    return sorted(names)


def _database_env_key_names(request: Request, job: Job, cfg: dict) -> list[str]:
    """Best-effort backend container env NAMES for a database row (recompute fallback).

    Database rows discard the generic service env assembly: the launch env is
    ALLOWLISTED to the backend statics + the minted credential name (§1.3), never
    the full secret scope. Recompute those names from the backend's own container
    shape with a placeholder minted value (names only — no value ever). Best-effort:
    any failure degrades to what was computed so far (the persisted
    `last_launch_env_keys` is the accurate source once the row has launched).
    """
    dc = getattr(request.app.state, "data_controller", None)
    if dc is None:
        return []
    names: set[str] = set()
    try:
        backend = dc.backend_for(cfg)
        config = backend.container_config(
            job.service_name or "", 0, {backend.minted_secret_key: ""}
        )
        names.update((config.env or {}).keys())
    except Exception:  # noqa: BLE001 — a projection must never fail the diagnose read
        pass
    return sorted(names)


def _pending_env_key_names(request: Request, job: Job, cfg: dict) -> list[str]:
    """Recompute the env key NAMES a launch would assemble *now* (§1.3).

    `cfg['env']` keys ∪ per-service secret names ∪ every binding kind's
    injected env-var names (`[ai.*]` → `NERDIT_AI_*`/`OPENAI_*`,
    `[db.*]` → `NERDIT_DB_*_URL`/`DATABASE_URL`/`REDIS_URL`) ∪
    `{'PORT'}` — the `_launch` assembly, names only. For a model row this
    reflects the backend path only; for a database row the allowlisted backend
    env only (the generic app path would wrongly union PORT and the full secret
    scope).
    """
    if job.kind is JobKind.model:
        return _model_env_key_names(request, job, cfg)
    if job.kind is JobKind.database:
        return _database_env_key_names(request, job, cfg)
    pending: set[str] = {str(k) for k in (cfg.get("env") or {}).keys()}
    secret_mgr = getattr(request.app.state, "secret_manager", None)
    if secret_mgr is not None and job.service_name:
        try:
            pending.update(secret_mgr.list_keys(job.service_name))
        except SecretDecryptError:
            pass  # names unavailable while the key is missing — omit, don't fail
    # (P15 / D-A) Union the injected env NAMES for every binding kind through the
    # per-kind registry — the grammar lives in each kind's `key_names_fn`
    # (F7-ENVGRAMMAR), not here, so a new binding kind needs no edit. Omitting the
    # `[db.*]` kind left a bound app's recomputed keys missing DATABASE_URL.
    for section, kind in BINDING_KINDS.items():
        specs = cfg.get(section)
        if isinstance(specs, dict) and specs:
            pending.update(kind.key_names_fn(specs))
    pending.add("PORT")
    return sorted(pending)


async def _fresh_health_probe(job: Job, endpoint: ServiceEndpoint | None) -> dict | None:
    """Probe the live container's health at request time; `None` when no container.

    Interpretation of §1.3's "null when no live container": *no container* is read
    from the persisted row — `endpoint is None or job.container_id is None` ⇒
    `probe = null`. When the row records a container but it is dead/unreachable
    at probe time, we return a probe object with `status_code = null` (unreachable),
    NOT `probe = null`. Consumers must treat a `probe` object with
    `status_code = null` as "a container was expected but did not answer", and
    only `probe = null` as "there is no container to probe".
    """
    if endpoint is None or job.container_id is None:
        return None
    hc = job.health_check if isinstance(job.health_check, dict) else {}
    # (§1.5 item 1) The SAME tolerant coercion as the reconcile loop
    # (`health.as_float`), not a separate isinstance guard — a string
    # `timeout_s` now resolves identically here and in `core/services.py`.
    timeout = as_float(hc.get("timeout_s"), DEFAULT_HEALTH_TIMEOUT_S)
    # (P14 WP-C1) Branch on probe kind exactly like the reconcile loop via the
    # shared `health.run_probe`; a junk or absent type falls through to http.
    # `check_health`/`check_tcp` are looked up as module globals here (not
    # captured as a default arg) so a test's `monkeypatch.setattr(svc_routes,
    # ...)` still intercepts. `type` in the response reflects the probe
    # actually run, never the raw (possibly junk) blob value.
    # (P24b, D-P24-4b) Probe the LIVE port, not the stable allocation: during a
    # promoted cutover the only listener is the green on its ephemeral port, so
    # dialling `host_port` would report a healthy service as unreachable.
    code, kind = await run_probe(
        hc, endpoint.live_port, timeout, http_check=check_health, tcp_check=check_tcp
    )
    return {
        "status_code": code,
        "type": kind,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _health_observations(job: Job, probe: dict | None) -> list[str]:
    """Describe an implicit root probe that returned 404, without changing verdicts.

    Run only when no health check was declared. API-only apps may legitimately
    return 404 at root; this fixed advisory never changes convergence or remediation
    and contains no app-derived values.
    """
    if job.health_check:
        return []
    if not probe or probe.get("type") != "http" or probe.get("status_code") != 404:
        return []
    return [
        "No [deploy].health is declared, so this service is judged by container "
        "liveness only; the default '/' probe answered 404. If the app exposes a "
        "health endpoint, set health = '/path' in [deploy] for real health checks."
    ]


def _no_secrets(_service: str) -> dict[str, str]:
    """Fallback secret-scope loader for a degraded app.state with no SecretManager.

    Keeps `_classify_bindings`'s managed `[db.*]` resolve well-typed when
    no manager is attached (test/degraded states); a missing minted credential
    then surfaces as an actionable `BindingNotReady` message-only wait.
    """
    return {}


async def _classify_bindings(request: Request, job: Job, cfg: dict) -> BindingWait:
    """Fresh, read-only `[ai.*]`/`[db.*]` resolve classifying the wait kind.

    Idempotent by design (the resolvers perform no writes). A per-binding loop
    (rather than the all-or-nothing `resolve_bindings`) so a mixed spec reports
    *every* blocked binding, and each wait is attributed: an `[ai.*]` ollama →
    model wait, api → secret wait; a `[db.*]` external → secret wait, managed →
    a message-only resource wait (its remediation is `nerdit db create`, never
    `nerdit serve <model>` — so it must not set `model_wait`, which routes to
    Rule 2). A secret-decrypt failure is itself a secret wait. Values are never
    touched — only key names / wait messages.
    """
    # Model and database rows are bound-TO resources, never binding consumers
    # (their [ai.*]/[db.*] is suppressed at launch — P15), so they never wait.
    if job.kind in (JobKind.model, JobKind.database):
        return BindingWait()
    ai_specs = cfg.get("ai")
    db_specs = cfg.get("db")
    ai = ai_specs if isinstance(ai_specs, dict) else {}
    db = db_specs if isinstance(db_specs, dict) else {}
    if not ai and not db:
        return BindingWait()

    queries = request.app.state.queries
    secret_mgr = getattr(request.app.state, "secret_manager", None)
    mc = getattr(request.app.state, "model_controller", None)
    bridge_host = getattr(mc, "bridge_host", "") if mc is not None else ""

    secret_env: dict[str, str] = {}
    secret_decrypt_failed = False
    if secret_mgr is not None and job.service_name:
        try:
            secret_env = secret_mgr.load(job.service_name)
        except SecretDecryptError:
            secret_decrypt_failed = True

    shared_env: dict[str, str] = {}
    if secret_mgr is not None:
        # (P15 / D-A) Shared refs come from every binding kind — an [ai.*]
        # api_key and a [db.*] password — resolved through the per-kind registry's
        # declared secret-ref fields, so a new kind needs no edit here.
        shared: set[str] = set()
        for section, kind in BINDING_KINDS.items():
            specs = cfg.get(section)
            if isinstance(specs, dict) and specs:
                shared |= set(shared_secret_keys_for(specs, kind.shared_secret_fields))
        unresolved = [k for k in sorted(shared) if k not in secret_env]
        if unresolved:
            try:
                shared_env = secret_mgr.load(SHARED_SCOPE)
            except SecretDecryptError:
                secret_decrypt_failed = True

    model_wait = False
    secret_wait = secret_decrypt_failed
    messages: list[str] = []
    for name in sorted(ai):
        spec = ai[name]
        provider = spec.get("provider") if isinstance(spec, dict) else None
        try:
            await resolve_binding(
                name,
                spec if isinstance(spec, dict) else {},
                queries,
                secret_env,
                bridge_host,
                shared_env=shared_env,
            )
        except BindingNotReady as exc:
            messages.append(str(exc))
            if provider == "ollama":
                model_wait = True
            elif provider == "api":
                secret_wait = True
            # provider missing/invalid → recorded as a message, attributed to
            # neither (falls through to inspect_logs by design).

    # The [db.*] table — the mirror of the ai loop. A managed target that
    # is missing/not-ready leaves the app in the launch retry loop, so diagnose
    # must report it (Invariant #4, outcome-readable).
    data_controller = getattr(request.app.state, "data_controller", None)
    load_scope = secret_mgr.load if secret_mgr is not None else _no_secrets
    for name in sorted(db):
        spec = db[name]
        provider = spec.get("provider") if isinstance(spec, dict) else None
        if data_controller is None:
            messages.append(
                f"[db.{name}] cannot be classified — the database controller is "
                f"unavailable; retry shortly."
            )
            continue
        try:
            await resolve_db_binding(
                name,
                spec if isinstance(spec, dict) else {},
                queries,
                secret_env,
                bridge_host,
                data_controller.backend_for,
                load_scope,
                shared_env=shared_env,
            )
        except BindingNotReady as exc:
            messages.append(str(exc))
            # external → a missing secret ref (Rule 3, `nerdit secrets set`);
            # managed → message-only, so Rule 2 (serve_missing_model) never fires
            # for a database (its message already points at `nerdit db create`).
            if provider == "external":
                secret_wait = True

    if secret_decrypt_failed and not messages:
        messages.append(
            "Per-service or shared secrets could not be decrypted — restore the secrets key."
        )
    waiting = model_wait or secret_wait or bool(messages)
    return BindingWait(
        waiting=waiting,
        model_wait=model_wait,
        secret_wait=secret_wait,
        messages=tuple(messages),
    )


def _classify_edge_auth(request: Request, job: Job, cfg: dict) -> EdgeAuthWait:
    """Classify persisted edge auth using the proxy's grammar and secret precedence.

    Perform read-only I/O here for the pure remediation classifier. Discard plaintext
    immediately; return only field/key names. Undecryptable scope means missing secret,
    as the proxy cannot materialize it either.
    """
    raw = cfg.get("edge_auth")
    if raw is None:
        return EdgeAuthWait()
    try:
        spec = load_edge_auth(raw)
    except EdgeAuthInvalid as exc:
        return EdgeAuthWait(invalid_fields=tuple(exc.fields))
    if spec is None:  # pragma: no cover — load_edge_auth only returns None for None
        return EdgeAuthWait()

    # `load_edge_auth` already fullmatched the ref, so the scope/key groups
    # are available without a second grammar decision.
    match = SECRET_REF_RE.fullmatch(spec.password_ref)
    shared = bool(match and match.group(1) == "shared")
    key = match.group(2) if match else "?"

    secret_mgr = getattr(request.app.state, "secret_manager", None)

    def _service_scope() -> Mapping[str, str]:
        if secret_mgr is None or not job.service_name:
            return {}
        return secret_mgr.load(job.service_name)

    def _shared_scope() -> Mapping[str, str]:
        if secret_mgr is None:
            return {}
        return secret_mgr.load(SHARED_SCOPE)

    try:
        result = walk_secret_ref(
            spec.password_ref, service_env=_service_scope, shared_env=_shared_scope
        )
    except SecretDecryptError:
        return EdgeAuthWait(missing_key=key, shared_scope=shared)
    if result.value is None:
        return EdgeAuthWait(missing_key=result.key or key, shared_scope=shared)
    # Mirror the emitter's third refusal (review-upheld): `hash_password`
    # raises on a resolved value over bcrypt's 72-byte input limit, so the
    # route plane withholds the route — diagnose must say so rather than
    # report all-clear. Same shared predicate, value discarded on the spot.
    if password_exceeds_bcrypt_limit(result.value):
        return EdgeAuthWait(over_length_key=result.key or key, shared_scope=shared)
    return EdgeAuthWait()


async def _classify_acme(request: Request, job: Job) -> AcmeWait:
    """Classify domain certificates through read-only query and proxy accessors.

    Missing collaborators yield no classification, not an error. Per-domain reads
    are acceptable on this detail surface. Return only public domain names.
    """
    queries = request.app.state.queries
    get_domains = getattr(queries, "get_service_domains", None)
    manager = getattr(request.app.state, "proxy_manager", None)
    cert_status = getattr(manager, "cert_status", None)
    if get_domains is None or cert_status is None or not job.service_name:
        return AcmeWait()

    pending: list[str] = []
    disabled: list[str] = []
    for row in await get_domains(job.service_name):
        if not row.acme:
            # An `acme=0` row is the default and not a defect — it is served
            # with the internal CA on purpose and has nothing to remediate.
            continue
        state = cert_status(row).state
        if state in ("pending", "expired"):
            pending.append(row.domain)
        elif state == "disabled":
            disabled.append(row.domain)
    return AcmeWait(pending_domains=tuple(pending), disabled_domains=tuple(disabled))


@router.get(
    "/services/{ident}/diagnose",
    response_model=DiagnoseResponse,
    operation_id="diagnose_service",
    tags=["Services"],
)
async def diagnose_service(
    request: Request,
    ident: str,
    log_tail: int = Query(50, description="Log lines to bundle (clamped to [1, 200])"),
) -> DiagnoseResponse:
    """Return an owner/admin failure bundle with fresh probes and binding checks.

    Read forensics, deployment/run/dump state and injected key names from persisted
    data. Structured env/secret fields contain names only. Application logs and
    run/dump tails remain owner-confidential: bounded exact-value scrubbing cannot
    remove transformed secrets. DiagnoseResponse documents these exceptions.
    No Idempotency-Key or audit row is required for this read.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    require_owner_or_admin(request, job)

    cfg = parse_job_config(job)
    last_deploy = cfg.get("last_deploy") if isinstance(cfg.get("last_deploy"), dict) else None
    last_run = cfg.get("last_run") if isinstance(cfg.get("last_run"), dict) else None
    # (P37 D-P37-11) ``last_dump`` is projected exactly once, here, for the same
    # reason ``last_run`` is: the tail it carries is tool output that the run
    # response only ever showed one line of, and this route is the one
    # owner-or-admin surface that may carry it. ``None`` on every service and
    # model row — only a managed database ever stamps the key.
    last_dump = cfg.get("last_dump") if isinstance(cfg.get("last_dump"), dict) else None
    endpoint = await queries.get_service_endpoint(job.service_name or "")

    # Persisted crash forensics (names/values are numeric/bool/timestamp only).
    forensics = forensics_from_config(cfg)
    last_exit_code = forensics["last_exit_code"]
    oom_killed = forensics["oom_killed"]
    gpu_oom = forensics["gpu_oom"]
    priv_denied = forensics["priv_denied"]
    last_crash_at = forensics["last_crash_at"]

    # Error classification (row columns first, then the phase object).
    err_class = job.error_class.value if job.error_class else None
    err_message = job.error_message
    if err_class is None and last_deploy is not None:
        ld_class = last_deploy.get("error_class")
        err_class = ld_class if isinstance(ld_class, str) else None
    if err_message is None and last_deploy is not None:
        ld_msg = last_deploy.get("error_message")
        err_message = ld_msg if isinstance(ld_msg, str) else None

    # Restart bookkeeping — recomputed (no persisted "next retry at").
    settings = request.app.state.settings
    svc_settings = settings.services
    now = datetime.now(timezone.utc)
    count = job.restart_count
    backoff_s = ServiceController._backoff_seconds(count) if count > 0 else None
    next_retry: float | None = None
    if job.status not in _WAIT_TERMINAL_STATUSES and count > 0 and job.last_exit_at is not None:
        le = job.last_exit_at
        if le.tzinfo is None:
            le = le.replace(tzinfo=timezone.utc)
        remaining = ServiceController._backoff_seconds(count) - (now - le).total_seconds()
        next_retry = round(max(0.0, remaining), 1)

    # Fresh health probe (null when no live container).
    probe = await _fresh_health_probe(job, endpoint)

    # Fresh, read-only binding resolution.
    binding_wait = await _classify_bindings(request, job, cfg)

    # Injected env keys: persisted (source=launch) with a recompute fallback.
    persisted_keys = cfg.get("last_launch_env_keys")
    pending_keys = _pending_env_key_names(request, job, cfg)
    if isinstance(persisted_keys, list):
        injected_keys = [str(k) for k in persisted_keys]
        injected_source = "launch"
    else:
        injected_keys = list(pending_keys)
        injected_source = "recomputed"

    # Build/deploy generation summary.
    build_version = last_deploy.get("version") if last_deploy else cfg.get("build_version")
    build_version = build_version if isinstance(build_version, int) else None
    phase = last_deploy.get("phase") if last_deploy else None
    build_reason = last_deploy.get("reason") if last_deploy else None
    if phase == "failed":
        last_result = "failed"
    elif phase == "healthy":
        last_result = "ok"
    else:
        last_result = "none"

    # Log tail (DB-backed — never runtime.logs; the container is usually gone).
    tail = max(1, min(_MAX_DIAGNOSE_TAIL, log_tail))
    entries = await queries.get_logs(job.id, tail=tail)
    logs = [
        {"stream": e.stream.value, "line": e.message, "ts": e.timestamp.isoformat()}
        for e in entries
    ]

    # The edge-auth outcome the pure classifier cannot compute for itself.
    edge_auth_wait = _classify_edge_auth(request, job, cfg)

    # The certificate outcome, likewise computed here — the classifier
    # is pure and this needs the domain table plus the proxy's storage tree.
    acme_wait = await _classify_acme(request, job)

    code, detail = derive_remediation(
        job, cfg, forensics, binding_wait, probe, edge_auth_wait, acme=acme_wait
    )

    return DiagnoseResponse(
        service_name=job.service_name or job.name or job.id,
        kind=job.kind,
        status=job.status,
        desired_state=job.desired_state,
        last_deploy=last_deploy,
        last_run=last_run,
        last_dump=last_dump,
        error=DiagnoseError(**{"class": err_class, "message": err_message}),
        forensics=DiagnoseForensics(
            last_exit_code=last_exit_code,
            oom_killed=oom_killed,
            gpu_oom=gpu_oom,
            priv_denied=priv_denied,
            last_crash_at=last_crash_at,
        ),
        restarts=DiagnoseRestarts(
            policy=job.restart_policy,
            count=count,
            max_restarts=svc_settings.service_max_restarts,
            window_seconds=svc_settings.restart_window_seconds,
            window_start=job.restart_window_start,
            last_exit_at=job.last_exit_at,
            backoff_s=backoff_s,
            next_retry_in_s=next_retry,
        ),
        health=DiagnoseHealth(
            spec=job.health_check if isinstance(job.health_check, dict) else None,
            probe=probe,
            observations=_health_observations(job, probe),
        ),
        bindings=DiagnoseBindings(
            waiting=binding_wait.waiting, messages=list(binding_wait.messages)
        ),
        build=DiagnoseBuild(
            version=build_version,
            last_result=last_result,
            reason=build_reason if isinstance(build_reason, str) else None,
        ),
        injected_env_keys=injected_keys,
        injected_env_keys_source=injected_source,
        pending_env_keys=pending_keys,
        logs=logs,
        remediation=DiagnoseRemediation(code=code.value, detail=detail),
    )
