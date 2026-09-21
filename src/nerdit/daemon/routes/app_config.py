"""Per-app config-as-API endpoints (P7 / plan §2.2).

Exposes a deployed app's daemon-persisted config — the `[deploy]` fields and
the `[ai.*]` binding spec — as a validated, machine-writable resource
(Invariant #3), without introducing a second store: the source of truth stays
the existing `jobs.config` projection the runtime path already reads.

* `GET /api/config/apps/{name}` — the app's config view (readonly+): deploy
  fields, `[ai.*]` spec (secret *refs* only, never values), env key names,
  `source`/`revision` (redeploy-vs-API visibility) and an `ETag`.
* `PUT /api/config/apps/{name}/{section}` — owner-or-admin, sections
  `deploy` | `ai` only. Mirrors the daemon-config PUT ceremony: merge keys,
  `null` deletes (for `ai` at *binding* granularity), `?dry_run`,
  `If-Match` optimistic concurrency, mandatory `Idempotency-Key` on real
  writes, structured 422 diagnostics, audited (`config.app_update`).

Mutability: `deploy.name` and `deploy.port` are immutable via this API
(identity / buildpack-baked — change the port via a redeploy); `env` is
secrets/deploy-owned (names appear read-only in the GET view). **No auto-restart**: a
write never
bounces a running service by itself — the response carries `requires_restart`
and `?restart=true` opts into the shipped restart desired-state path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TypeGuard

from fastapi import APIRouter, Body, Query, Request, Response

from nerdit.config.app_config import validate_app_config
from nerdit.config.project import rewrite_vars_ref, shared_secret_keys, shared_secret_keys_for
from nerdit.config.redaction import REDACTED, redact_section
from nerdit.core.jobconfig import parse_job_config
from nerdit.daemon.audit import audit_params, record_shared_referenced
from nerdit.daemon.auth import (
    QuotaExceeded,
    require_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.deploy_pipeline import _resolve_data_volume
from nerdit.daemon.errors import NerditError
from nerdit.db.models import (
    AppConfigView,
    AppConfigWriteResponse,
    Job,
    JobKind,
    JobStatus,
    TokenRole,
)
from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)

router = APIRouter()

# Sections writable via this API. `env` is deploy/secrets-owned.
_WRITABLE_SECTIONS = ("deploy", "ai", "db")
_UNKNOWN_SECTION_HINT = (
    "Writable sections: deploy, ai, db. env is managed via /deploy and /secrets."
)
# [deploy] keys writable via this API. `name` is the service identity
# (endpoint/route/image tags); `port` may be baked into the image by the
# buildpack, so it changes via redeploy only (P7 decision #4).
_DEPLOY_MUTABLE_KEYS = frozenset(
    {
        "gpus",
        "start",
        "health",
        "memory_limit",
        "cpu_limit",
        "volumes",
        "release",
        # Tri-state next-deploy booleans — see _RESTART_DEPLOY_KEYS below
        # for why they are deliberately absent from the restart set.
        "cutover",
        "auto_deploy",
        # (P25, D-P25-5) The edge-auth declaration ({user, password ref}). This
        # is the API path an operator uses to protect or unprotect a published
        # app — and, per the DeployConfig carry-forward note, the ONLY way to
        # disarm one (`deploy edge_auth=null`); deleting it from nerdit.toml
        # does not. Also absent from _RESTART_DEPLOY_KEYS — see below.
        "edge_auth",
    }
)
_DEPLOY_IMMUTABLE_KEYS = frozenset({"name", "port"})
# Keys whose change only takes effect at the next container launch (GPU
# allocation/release also happens only on launch/restart paths). memory_limit /
# cpu_limit are applied by the runtime at launch, so they restart too;
# volumes (P14 WP-A1) mount at launch, so likewise.
#
# `release` is deliberately ABSENT — the one asymmetric key. It is a
# build-time gate consumed by the NEXT deploy (the builder runs it against the
# freshly built candidate image), never by a container launch, so changing it
# yields `requires_restart: false`: restarting the service would not run it.
#
# `cutover` and `auto_deploy` are absent for the same reason, and
# make the choice explicitly: `cutover` is read by the NEXT deploy's
# reconcile-time eligibility check (a running container is never re-cut over by
# a restart — a restart is precisely the same-port relaunch the flag opts out
# of), and `auto_deploy` is consumed by the P24c webhook plane. Restarting
# would apply neither, so `requires_restart: false` is the honest answer.
#
# (P25, §3.4.6 / D-P25-8) `edge_auth` is absent for a DIFFERENT reason worth
# stating plainly: **changing edge auth never bounces the app.** The credential
# is materialized into the Caddy route object, not into the container's
# environment, so the proxy reconcile loop picks a change up on its next tick
# (and the D-P25-7 auth fingerprint makes that tick corrective rather than
# blind) with the container untouched. Restarting would achieve nothing except
# a gratuitous outage, so `requires_restart: false` is again the honest
# answer — and a caller who requested `?restart=true` still gets one, this
# set only decides what the daemon ASKS for.
_RESTART_DEPLOY_KEYS = frozenset(
    {"gpus", "start", "health", "memory_limit", "cpu_limit", "volumes"}
)
# The full set of deploy keys the merge tracks (order-stable for the `changed`
# diff). `volumes` is a list; the others are scalars.
_DEPLOY_TRACKED_KEYS = (
    "gpus",
    "start",
    "health",
    "memory_limit",
    "cpu_limit",
    "volumes",
    "release",
    "cutover",
    "auto_deploy",
    "edge_auth",
)


# Config-blob keys the BUILD tier owns end to end. This route never
# authors any of them, but it rewrites the WHOLE blob, and a deploy/build/
# release can rewrite them at any instant — the build task runs off the
# reconcile tick and holds no lock a request handler could take. Since
# `update_app_config` is an unconditional `UPDATE jobs SET config = ?` with
# no CAS, writing these back from the snapshot read at the top of the handler
# would silently revert whatever landed in between. `release_pending` is the
# one with teeth: losing it drops the pre-swap gate, and a daemon killed
# mid-migration then converges the unmigrated candidate in front of a
# half-migrated database (D-P20-4).
# `cutover_pending` joins it for the identical reason: it is the
# crash-safe marker recording that a green container was launched and this
# daemon never saw the cutover settle. Losing it from a snapshot write strands
# the green (nothing reaps it) and leaves `active_host_port` naming a port the
# layer-4 settle would otherwise have cleared — a permanent 502 on a row that,
# being `degraded`, is never relaunched (D-P24-4b).
_BUILD_TIER_KEYS = frozenset(
    {
        "release_pending",
        "cutover_pending",
        "build_version",
        "max_version",
        "image",
        "previous_image",
        "image_repo",
        "build_context_dir",
        "build_context_root",
        "dockerfile_name",
        "buildpack",
        "build_overrides",
        "build_plan",
        "public_env",
        "last_deploy",
    }
)


async def _graft_build_tier_keys(queries: Queries, job_id: str, new_cfg: dict) -> dict:
    """Take `_BUILD_TIER_KEYS` from a FRESH read instead of the snapshot.

    Mirrors the builder's own `fresh = await get_job` precedent
    (`core/app_build.py`). It **narrows** the lost-update window to the gap
    between this read and the UPDATE that immediately follows; it does not
    close it — two statements with no CAS never can. Closing it properly means
    a version-guarded write in `db/queries/service_config.py`, which belongs
    with P22's idempotency/contract pass rather than in a config route.
    """
    row = await queries.get_job(job_id)
    if row is None:
        return new_cfg  # deleted under us; the UPDATE below will match no row
    fresh = parse_job_config(row)
    grafted = {key: value for key, value in new_cfg.items() if key not in _BUILD_TIER_KEYS}
    grafted.update({key: value for key, value in fresh.items() if key in _BUILD_TIER_KEYS})
    return grafted


async def _get_app(request: Request, name: str) -> Job:
    """Judge scope, then resolve *name* to a `kind=service` row or a structured 404.

    (D-P40-7) The row is read first so its project can widen the scope; the
    scope verdict still precedes every row-shaped answer, so an out-of-scope
    caller gets the identical 403 with or without a row.
    """
    job = await request.app.state.queries.get_service_by_name(name)
    require_service_scope(request, name, project=job.project if job else None)
    if job is not None and job.kind is JobKind.model:
        raise NerditError(
            404,
            "config.not_an_app",
            f"'{name}' is a served model, not a configurable app.",
            hint=(
                "Models have no [deploy]/[ai] config; inspect them with "
                "`nerdit models` (GET /models) and manage lifecycle via /services."
            ),
        )
    if job is not None and job.kind is JobKind.database:
        raise NerditError(
            404,
            "config.not_an_app",
            f"'{name}' is a managed database, not a configurable app.",
            hint=(
                "Databases have no [deploy]/[ai] config; inspect them with "
                "`nerdit db list` (GET /databases) and manage lifecycle via /services."
            ),
        )
    if job is None or job.kind != JobKind.service:
        raise NerditError(
            404,
            "not_found",
            f"No deployed app '{name}'.",
            hint="Deploy it first with `nerdit deploy`.",
        )
    return job


def _redacted_edge_auth(raw: object) -> object:
    """Project edge-auth metadata without trusting persisted data to be validated.

    None stays None. Allow only user/password string mappings; reveal user and mask
    password, even when it is a reference. Any unknown key, nesting or other malformed
    shape becomes *** as a whole, preventing credentials hidden under arbitrary keys
    from being reflected.
    """
    if raw is None:
        return None
    if _is_wellformed_edge_auth(raw):
        return redact_section(dict(raw))
    return REDACTED


def _is_wellformed_edge_auth(raw: object) -> TypeGuard[dict[str, str]]:
    """Whether `raw` is a `{user?, password?}` string mapping and nothing else.

    Deliberately stricter than `nerdit.core.proxy.edgeauth.load_edge_auth`
    (which also enforces the grammar): here the only question is whether the blob
    is safe to reflect leaf-by-leaf. Any extra key, any nested/non-string value,
    is not — it is masked wholesale by the caller.
    """
    if not isinstance(raw, dict):
        return False
    if not set(raw).issubset({"user", "password"}):
        return False
    return all(isinstance(v, str) for v in raw.values())


def _build_view(job: Job) -> AppConfigView:
    """Project a service row into an `AppConfigView` (with ETag)."""
    cfg = parse_job_config(job)
    deploy: dict[str, object] = {
        "name": job.service_name,
        "port": cfg.get("port"),
        "gpus": job.gpu_count,
        "start": cfg.get("command"),
        "health": (job.health_check or {}).get("path"),
        # launch-time resource caps live top-level in the config blob
        # (their own literal key names; the runtime reader uses the same).
        "memory_limit": cfg.get("memory_limit"),
        "cpu_limit": cfg.get("cpu_limit"),
        # P14 WP-A1: named volumes (names + container paths only, never a host
        # path). The implicit `data:/data` is present on every deploy-created
        # app after its next redeploy.
        "volumes": cfg.get("volumes") or [],
        # the pre-swap release command, persisted top-level under its own
        # literal key name (the builder reads cfg["release"]).
        "release": cfg.get("release"),
        # tri-state next-deploy booleans, persisted top-level under their
        # literal key names. NULL is a real, distinct value here (the daemon
        # default posture), so they are surfaced as `None` rather than
        # defaulted — a reader must be able to tell "unset" from "false".
        "cutover": cfg.get("cutover"),
        "auto_deploy": cfg.get("auto_deploy"),
        # the edge-auth declaration, masked like the [ai.*]/[db.*] specs.
        "edge_auth": _redacted_edge_auth(cfg.get("edge_auth")),
    }
    # api_key is a ${secrets.X} / ${secrets.shared.X} ref by SECRET_REF_RE
    # construction — inherently safe — but still passed through redact_section
    # defensively.
    ai = {bname: redact_section(dict(spec)) for bname, spec in (cfg.get("ai") or {}).items()}
    # the [db.*] spec (password is a ${secrets.X} ref, url credential-free
    # by parse) — passed through redact_section defensively like [ai.*].
    db = {bname: redact_section(dict(spec)) for bname, spec in (cfg.get("db") or {}).items()}
    body = {
        "service_name": job.service_name or job.name or job.id,
        "deploy": deploy,
        "ai": ai,
        "db": db,
        "env_keys": sorted((cfg.get("env") or {}).keys()),
        "source": cfg.get("config_source", "deploy"),
        "revision": int(cfg.get("config_revision", 0)),
    }
    etag = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return AppConfigView(**body, etag=etag)


@router.get(
    "/config/apps/{name}",
    response_model=AppConfigView,
    operation_id="get_app_config",
)
async def get_app_config(request: Request, response: Response, name: str) -> AppConfigView:
    """Return a deployed app's config view (secret values never included)."""
    require_role(request, TokenRole.readonly, TokenRole.submitter, TokenRole.admin)
    # (P25 D-P25-3 leg b) The GET carries no owner gate by design, so scope is
    # the only per-service boundary here; a pure name check needs no row.
    job = await _get_app(request, name)
    view = _build_view(job)
    response.headers["ETag"] = view.etag or ""
    return view


def _data_volume_path(specs: object) -> str | None:
    """Container path of the `data` volume in a specs list, if declared."""
    if not isinstance(specs, list):
        return None
    for spec in specs:
        if isinstance(spec, str) and spec.partition(":")[0] == "data":
            return spec.partition(":")[2] or None
    return None


def _merge_deploy(job: Job, cfg: dict, body: dict) -> tuple[dict, list[str]]:
    """Merge a `deploy` PUT body over the current effective deploy fields.

    Returns the merged effective deploy dict and the list of changed keys.
    `null` deletes (reverts to the default: `gpus` → 0, `start`/`health`
    → unset). Immutable / unknown keys raise structured 422s.
    """
    immutable = sorted(_DEPLOY_IMMUTABLE_KEYS & body.keys())
    if immutable:
        raise NerditError(
            422,
            "config.immutable_key",
            f"deploy.{immutable[0]} cannot be changed via the config API.",
            hint=(
                "name is the service identity."
                if immutable[0] == "name"
                else "change the port via a redeploy."
            ),
        )
    unknown = sorted(body.keys() - _DEPLOY_MUTABLE_KEYS)
    if unknown:
        raise NerditError(
            422,
            "config.invalid",
            f"Unknown [deploy] key(s): {', '.join(unknown)}.",
            hint=f"Writable [deploy] keys: {', '.join(sorted(_DEPLOY_MUTABLE_KEYS))}.",
            diagnostics=[
                {"loc": ["deploy", key], "message": "unknown key", "type": "unknown_key"}
                for key in unknown
            ],
        )
    current = {
        "gpus": job.gpu_count,
        "start": cfg.get("command"),
        "health": (job.health_check or {}).get("path"),
        "memory_limit": cfg.get("memory_limit"),
        "cpu_limit": cfg.get("cpu_limit"),
        "volumes": cfg.get("volumes") or [],
        "release": cfg.get("release"),
        "cutover": cfg.get("cutover"),
        "auto_deploy": cfg.get("auto_deploy"),
        # The RAW persisted blob, never the redacted view projection: this
        # dict is the baseline for the `changed` diff and the input to
        # `validate_app_config`. A masked `password` would make every no-op
        # write look changed and would then fail the secret-ref validator.
        "edge_auth": cfg.get("edge_auth"),
    }
    effective = dict(current)
    for key, value in body.items():
        # `null` deletes: gpus reverts to 0, volumes reverts to [] (no named
        # volumes), everything else (start/health/memory_limit/cpu_limit/
        # release) reverts to unset ⇒ the [containers] defaults, and no
        # release step at the next deploy. `cutover`/`auto_deploy` fall
        # through the same generic branch: null is their documented disarm —
        # it drops the key so the DAEMON default posture applies again, which
        # is a different state from an explicit `false`. `edge_auth` too:
        # `deploy edge_auth=null` is the ONLY way to unprotect an app, and it
        # needs no branch of its own — dict inequality drives the `changed`
        # diff for a dict value just as well as for a scalar.
        if value is None and key == "gpus":
            effective[key] = 0
        elif value is None and key == "volumes":
            effective[key] = []
        else:
            effective[key] = value
    # (M9, second ingress) Fold the implicit `data:/data` wedge in BEFORE the
    # `changed` diff and the validator, exactly as `resolve_effective_fields`
    # does on the deploy path — otherwise this surface accepts MAX_VOLUMES
    # declared without a `data` entry under a hint that says the declared
    # budget is one less, persists them verbatim, and the next silent redeploy
    # 422s on a list one over the cap while the row runs with NERDIT_DATA_DIR
    # pointing at an unmounted path. Only for a row that already carries the
    # wedge: a `POST /services` row never gets one from the deploy path, and
    # the config API must not invent one for it. And only for a NON-EMPTY
    # list, so `volumes=null` stays the documented "mount nothing" disarm
    # rather than quietly becoming "mount only the wedge".
    volumes = effective["volumes"]
    if isinstance(volumes, list) and volumes and _data_volume_path(cfg.get("volumes")) is not None:
        effective["volumes"] = _resolve_data_volume(volumes)[0]
    changed = [k for k in _DEPLOY_TRACKED_KEYS if effective[k] != current[k]]
    return effective, changed


def _stored_spec(spec: dict, field: str) -> dict:
    """`spec` as persisted: the `${vars.…}` alias in `field` rewritten (D-P40-9).

    The merged dict is stored raw (validation only rewrites its own throwaway
    model), so this ingress rewrites too - before the change diff, so a re-PUT
    of the alias compares equal. A stored alias would also fail a pre-P40c
    daemon's re-parse at launch after a rollback.
    """
    value = spec.get(field)
    return {**spec, field: rewrite_vars_ref(value)} if isinstance(value, str) else spec


def _merge_ai(cfg: dict, body: dict) -> tuple[dict, list[str]]:
    """Merge an `ai` PUT body at *binding* granularity (`null` deletes)."""
    current = dict(cfg.get("ai") or {})
    merged = dict(current)
    changed: list[str] = []
    for bname, spec in body.items():
        if spec is None:
            if merged.pop(bname, None) is not None:
                changed.append(bname)
            continue
        if not isinstance(spec, dict):
            raise NerditError(
                422,
                "config.invalid",
                f"[ai.{bname}] must be a table (or null to delete the binding).",
            )
        spec = _stored_spec(spec, "api_key")
        if merged.get(bname) != spec:
            changed.append(bname)
        merged[bname] = spec
    return merged, changed


def _merge_db(cfg: dict, body: dict) -> tuple[dict, list[str]]:
    """Merge a `db` PUT body at *binding* granularity (`null` deletes).

    A literal twin of `_merge_ai` (P15, D-A): each `[db.<name>]` table is
    replaced wholesale, `null` deletes the binding. Schema validation (provider
    rules, url shape, secret-ref password) runs afterwards in
    `validate_app_config` / `validate_db_section`.
    """
    current = dict(cfg.get("db") or {})
    merged = dict(current)
    changed: list[str] = []
    for bname, spec in body.items():
        if spec is None:
            if merged.pop(bname, None) is not None:
                changed.append(bname)
            continue
        if not isinstance(spec, dict):
            raise NerditError(
                422,
                "config.invalid",
                f"[db.{bname}] must be a table (or null to delete the binding).",
            )
        spec = _stored_spec(spec, "password")
        if merged.get(bname) != spec:
            changed.append(bname)
        merged[bname] = spec
    return merged, changed


@router.put(
    "/config/apps/{name}/{section}",
    response_model=AppConfigWriteResponse,
    operation_id="put_app_config_section",
)
async def put_app_config_section(
    request: Request,
    response: Response,
    name: str,
    section: str,
    body: dict = Body(..., description="Section values (partial; null deletes)"),
    dry_run: bool = Query(False, description="Validate + diff without writing"),
    restart: bool = Query(False, description="Restart the service after a real write"),
) -> AppConfigWriteResponse:
    """Validate and (unless `dry_run`) persist one app config section."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    # (P25 D-P25-3 leg b / D-P40-7) Scope is judged inside `_get_app`, on the path
    # name or the row's project, beside the row-level owner gate further down.
    job = await _get_app(request, name)
    # Names, never submitted values: this stamp is what the AuditMiddleware
    # records on every early exit (422 config.invalid, 409 config.stale, dry
    # run). `audit_params` masks by leaf name, which misses a credential
    # embedded in a value — an external `[db.*]` url carries its password in
    # the userinfo, and `[ai.*] base_url` can too.
    request.state.audit_params = {
        "section": section,
        "submitted_keys": sorted(str(key) for key in body),
        "restart": restart,
    }

    if section not in _WRITABLE_SECTIONS:
        raise NerditError(
            422,
            "config.unknown_section",
            f"'{section}' is not a writable app config section.",
            hint=_UNKNOWN_SECTION_HINT,
        )

    queries = request.app.state.queries
    require_owner_or_admin(request, job)
    cfg = parse_job_config(job)

    merged_ai = cfg.get("ai") or {}
    merged_db = cfg.get("db") or {}
    eff_deploy = {
        "gpus": job.gpu_count,
        "start": cfg.get("command"),
        "health": (job.health_check or {}).get("path"),
        "memory_limit": cfg.get("memory_limit"),
        "cpu_limit": cfg.get("cpu_limit"),
        "volumes": cfg.get("volumes") or [],
        "release": cfg.get("release"),
        "cutover": cfg.get("cutover"),
        "auto_deploy": cfg.get("auto_deploy"),
        # Raw blob — the ai/db branches below leave this dict as the
        # unchanged baseline, and it feeds the validator + the persist branch.
        "edge_auth": cfg.get("edge_auth"),
    }
    if section == "deploy":
        eff_deploy, changed = _merge_deploy(job, cfg, body)
    elif section == "ai":
        merged_ai, changed = _merge_ai(cfg, body)
    else:  # db
        merged_db, changed = _merge_db(cfg, body)

    # Same validators as the deploy ZIP path — API writes can never drift.
    effective_doc = {
        "deploy": {
            "name": job.service_name,
            "port": cfg.get("port"),
            "gpus": eff_deploy["gpus"],
            "start": eff_deploy["start"],
            "health": eff_deploy["health"],
            "memory_limit": eff_deploy["memory_limit"],
            "cpu_limit": eff_deploy["cpu_limit"],
            "volumes": eff_deploy["volumes"] or None,
            "release": eff_deploy["release"],
            "cutover": eff_deploy["cutover"],
            "auto_deploy": eff_deploy["auto_deploy"],
            # Re-validated through EdgeAuthConfig on every write, so an
            # API-set declaration is held to exactly the grammar the ZIP path
            # enforces (user shape, password = ${secrets.KEY} — a literal is a
            # 422 that never echoes it).
            "edge_auth": eff_deploy["edge_auth"],
        }
    }
    if merged_ai:
        effective_doc["ai"] = merged_ai
    if merged_db:
        effective_doc["db"] = merged_db
    deploy_cfg, _ = await validate_app_config(
        effective_doc, queries, request, previous_db=cfg.get("db")
    )

    current_view = _build_view(job)
    if_match = request.headers.get("If-Match")
    # RFC 7232: "*" = write-if-exists — the representation exists here, so proceed.
    if if_match is not None and if_match != "*" and if_match != current_view.etag:
        raise NerditError(
            409,
            "config.stale",
            "The app config changed since you last read it.",
            hint="Re-read it (GET) and retry with the new ETag.",
            current_etag=current_view.etag,
        )

    requires_restart = bool(changed) and (
        section in ("ai", "db") or bool(_RESTART_DEPLOY_KEYS & set(changed))
    )

    # Build the new config blob + column values (last writer wins; the API
    # stamps source/revision so a later redeploy clobber stays visible).
    new_cfg = dict(cfg)
    if merged_ai:
        new_cfg["ai"] = merged_ai
    else:
        new_cfg.pop("ai", None)
    if merged_db:
        new_cfg["db"] = merged_db
    else:
        new_cfg.pop("db", None)
    if eff_deploy["start"] is not None:
        new_cfg["command"] = deploy_cfg.start
    else:
        new_cfg.pop("command", None)
    # resource caps persist top-level under their own key names (the
    # launch reader uses cfg.get("memory_limit")/cfg.get("cpu_limit")); a null
    # delete pops the key so the [containers] defaults apply.
    if eff_deploy["memory_limit"] is not None:
        new_cfg["memory_limit"] = deploy_cfg.memory_limit
    else:
        new_cfg.pop("memory_limit", None)
    if eff_deploy["cpu_limit"] is not None:
        new_cfg["cpu_limit"] = deploy_cfg.cpu_limit
    else:
        new_cfg.pop("cpu_limit", None)
    # P14 WP-A1: named volumes persist top-level as the validated spec list; an
    # empty list (or a null-delete) pops the key so the launch path mounts none.
    if deploy_cfg.volumes:
        new_cfg["volumes"] = deploy_cfg.volumes
    else:
        new_cfg.pop("volumes", None)
    # Persist release for the next deploy; null removes the command but executes
    # nothing. Never alter release_pending here: it is generation-owned crash state,
    # armed by the builder and retired by its settlement/revert rules. It may describe
    # an active, interrupted, terminal or superseded release, and the app-config ETag
    # does not cover it. Keep it through the config copy even when release is deleted.
    if eff_deploy["release"] is not None:
        new_cfg["release"] = deploy_cfg.release
    else:
        new_cfg.pop("release", None)
    # The two tri-state next-deploy booleans persist top-level under
    # their literal key names (`_eligible` reads `cfg.get("cutover")`; the
    # P24c webhook plane will read `auto_deploy`). `is not None` and never
    # truthiness — `false` is the meaningful value here, and popping it would
    # silently re-arm the cutover. A null-delete pops the key ⇒ the daemon
    # default posture. `cutover_pending` is NOT touched on either branch, for
    # the same reason `release_pending` is not (see _BUILD_TIER_KEYS): it is
    # generation state the build tier owns, and the ETag does not cover it, so
    # a caller disarming `cutover` could not even see what they cleared.
    if eff_deploy["cutover"] is not None:
        new_cfg["cutover"] = deploy_cfg.cutover
    else:
        new_cfg.pop("cutover", None)
    if eff_deploy["auto_deploy"] is not None:
        new_cfg["auto_deploy"] = deploy_cfg.auto_deploy
    else:
        new_cfg.pop("auto_deploy", None)
    # (P25, D-P25-5) The edge-auth declaration persists top-level under its
    # literal key — `model_dump()` because the blob is `json.dumps`'d, and
    # from the VALIDATED model so a write can only ever store a
    # `${secrets.KEY}` reference (the raw body never lands unchecked). The
    # `release` pattern verbatim, including the null branch: popping the key
    # is the documented disarm (`deploy edge_auth=null`) and the only way to
    # unprotect an app — a silent redeploy carries the blob forward instead.
    # Nothing here touches the proxy: the route reconcile materializes the
    # change on its next tick (see _RESTART_DEPLOY_KEYS).
    if eff_deploy["edge_auth"] is not None and deploy_cfg.edge_auth is not None:
        new_cfg["edge_auth"] = deploy_cfg.edge_auth.model_dump()
    else:
        new_cfg.pop("edge_auth", None)
    # Retargeting the `data` volume must move the wedge-persisted
    # NERDIT_DATA_DIR with it (the deploy path's _ensure_data_volume rule) —
    # otherwise apps following the documented convention keep writing to an
    # unmounted path after the restart. A user-overridden value is left alone.
    prev_data = _data_volume_path(cfg.get("volumes"))
    new_data = _data_volume_path(new_cfg.get("volumes"))
    env = new_cfg.get("env")
    if (
        prev_data is not None
        and new_data is not None
        and new_data != prev_data
        and isinstance(env, dict)
        and env.get("NERDIT_DATA_DIR") == prev_data
    ):
        new_cfg["env"] = {**env, "NERDIT_DATA_DIR": new_data}
    # A no-op write (same values) must not rewrite provenance: bumping the
    # revision/ETag on a retry would make the surface non-idempotent and
    # create spurious 409s for concurrent readers.
    if changed:
        new_cfg["config_source"] = "api"
        new_cfg["config_revision"] = int(cfg.get("config_revision", 0)) + 1
        # a changed `ai` PUT stamps a section-level provenance marker
        # (including a delete-all — the API's opinion is "this ai state"). The
        # redeploy path preserves an API-authored [ai.*] spec across every
        # subsequent silent redeploy iff this marker is set; a source that
        # declares [ai] pops it. Never a schema field (config-blob key only).
        if section == "ai":
            new_cfg["ai_source"] = "api"
        # symmetric [db.*] provenance marker — the redeploy path preserves
        # an API-authored [db.*] spec across silent redeploys iff this is set.
        elif section == "db":
            new_cfg["db_source"] = "api"

    # Only touch the health_check column when `health` actually changed, and
    # preserve any tuning fields (timeout_s, thresholds) a /services-created
    # service may carry — a [deploy] write only owns the probed path.
    health_changed = section == "deploy" and "health" in changed
    if not health_changed:
        new_health = job.health_check
    elif eff_deploy["health"] is not None:
        new_health = {**(job.health_check or {}), "path": eff_deploy["health"]}
    else:
        new_health = None

    if dry_run:
        preview = job.model_copy(
            update={
                "config": json.dumps(new_cfg),
                "gpu_count": deploy_cfg.gpus,
                "health_check": new_health,
            }
        )
        view = _build_view(preview)
        response.headers["ETag"] = current_view.etag or ""
        return AppConfigWriteResponse(
            applied=False, requires_restart=requires_restart, restarted=False, view=view
        )

    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            "An app config write requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )

    gpus_changed = section == "deploy" and "gpus" in changed
    if changed:
        new_cfg = await _graft_build_tier_keys(queries, job.id, new_cfg)
        try:
            await queries.update_app_config(
                job.id,
                json.dumps(new_cfg),
                gpu_count=deploy_cfg.gpus if gpus_changed else None,
                health_check=new_health if health_changed else None,
                clear_health=health_changed and new_health is None,
                token_id=job.submitted_by_token if gpus_changed else None,
            )
        except QuotaExceeded as exc:
            raise exc.to_error() from exc

    # Write-time shared-ref visibility: an applied ai write whose merged
    # spec carries ${secrets.shared.KEY} refs is audited to the REQUESTING
    # principal (secret.shared_referenced) — the admin's before-launch signal.
    if section == "ai" and changed:
        shared_keys = shared_secret_keys(merged_ai)
        if shared_keys:
            await record_shared_referenced(request, name, shared_keys)
    elif section == "db" and changed:
        shared_keys = shared_secret_keys_for(merged_db, ("password",))
        if shared_keys:
            await record_shared_referenced(request, name, shared_keys)

    restarted = False
    if restart:
        # Opt-in restart: the shipped desired-state restart path (bindings are
        # re-resolved at every launch — no new reconcile code).
        await queries.set_desired_state(job.id, "running")
        await queries.update_job_status(job.id, JobStatus.restarting)
        await queries.bump_restart_count(job.id, 0, None)
        restarted = True

    job = await queries.get_service_by_name(name) or job
    view = _build_view(job)
    response.headers["ETag"] = view.etag or ""
    # Keep a redacted audit trail of exactly which keys changed.
    request.state.audit_params = {
        **audit_params({"section": section, "dry_run": False}),
        "changed_keys": [f"{section}.{key}" for key in changed],
        "requires_restart": requires_restart,
        "restarted": restarted,
        "revision": view.revision,
    }
    return AppConfigWriteResponse(
        applied=True, requires_restart=requires_restart, restarted=restarted, view=view
    )
