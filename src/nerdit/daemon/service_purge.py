"""Delete services and purge requested resources, with tombstone recovery.

The main services router includes this router. Import shared views rather than
routes.services to avoid cycles. Atomic reference checks and row deletion stay
in the serialized database query.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
from pathlib import Path

import anyio
from fastapi import APIRouter, Query, Request

from nerdit.config.defaults import app_image_repo
from nerdit.core.eventlog import get_recorder
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.runtime.protocol import ContainerNotFoundError, ContainerRuntimeError
from nerdit.core.volumes import (
    VolumeSpecError,
    make_tombstone_name,
    service_data_root,
    tombstone_service_name,
)
from nerdit.core.workspaces import (
    WorkspaceError,
    settled_to_thread,
    workspace_lock,
    workspace_root,
)
from nerdit.daemon.audit import audit_params, record_out_of_band
from nerdit.daemon.auth import current_principal, require_owner_or_admin
from nerdit.daemon.errors import NerditError
from nerdit.daemon.imagegc import _protected_image_refs, _repo_tags
from nerdit.daemon.secret_scope import variable_write_lock
from nerdit.daemon.views.service import (
    _cutover_in_progress_error,
    _not_found,
    _resolve_service,
    _run_in_progress_error,
)
from nerdit.db.models import (
    Job,
    JobKind,
    PurgeImages,
    PurgeReport,
    ServiceDeletedResponse,
    TokenRole,
)
from nerdit.db.queries import Queries

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Startup tombstone-restore sweep (C2) -------------------------------------


async def _sweep_data_tombstones(queries: Queries, data_dir: Path) -> None:
    """Restore interrupted-delete tombstones only to stopped rows with no data root.

    Never graft old data onto a running/new database row: its newly minted password
    would not match. Missing rows leave ordinary orphans for opt-in GC. Live base
    rows protect their tombstones; failures log and leave the tree intact.
    """
    services_root = data_dir / "services"
    try:
        names = await asyncio.to_thread(
            lambda: [e.name for e in os.scandir(services_root) if e.is_dir(follow_symlinks=False)]
        )
    except OSError:
        return
    tombstones = [(n, base) for n in names if (base := tombstone_service_name(n)) is not None]
    if not tombstones:
        return
    try:
        rows = await queries.list_workload_configs()
    except Exception:  # noqa: BLE001 — a sweep failure never blocks boot
        logger.warning("Tombstone sweep could not list workloads; skipping", exc_info=True)
        return
    live = {
        name: row.get("desired_state")
        for row in rows
        if isinstance((name := row.get("service_name")), str) and name
    }
    for name, base in tombstones:
        if base not in live:
            continue  # base row gone → committed delete; the GC orphan pass reclaims it
        if live[base] != "stopped":
            # A fresh same-name row (e.g. a recreate) is `running` with no data
            # dir yet: restoring an old tombstone onto it would graft stale weights.
            logger.info(
                "Leaving database data tombstone %s in place: base row services/%s is "
                "live but not stopped (desired_state=%r)",
                name,
                base,
                live[base],
            )
            continue
        try:
            root = service_data_root(data_dir, base)
        except VolumeSpecError:
            continue
        if await asyncio.to_thread(root.exists):
            continue  # real tree already present → nothing to restore
        try:
            await asyncio.to_thread(os.rename, services_root / name, root)
            logger.info(
                "Restored database data tombstone %s -> services/%s at startup; the "
                "interrupted delete left the row stopped — restart it or re-issue the "
                "delete",
                name,
                base,
            )
        except OSError:
            logger.error(
                "Failed to restore database data tombstone %s at startup; the row is "
                "left with an empty data root",
                name,
                exc_info=True,
            )


# --- ?purge parsing + the dependents/reference-guard family -----------------

# The `?purge` targets on `DELETE /services`. The default is `secrets` only;
# `data`/`images`/`workspace` are opt-in destructive (the agent workspace tree
# is user-authored source, so it is never removed by default).
_PURGE_TARGETS = frozenset({"secrets", "data", "images", "workspace"})


def _parse_purge(raw: str) -> set[str]:
    """Parse the `?purge` CSV into a validated target set.

    An unknown token is a structured 422 `service.invalid_purge` (the hint is
    derived from `_PURGE_TARGETS`, so a new member needs no prose edit here).
    An empty/blank value purges nothing.
    """
    tokens = {t.strip().lower() for t in raw.split(",") if t.strip()}
    unknown = tokens - _PURGE_TARGETS
    if unknown:
        raise NerditError(
            422,
            "service.invalid_purge",
            f"Unknown purge target(s): {', '.join(sorted(unknown))}.",
            hint=f"Valid purge targets: {', '.join(sorted(_PURGE_TARGETS))}.",
        )
    return tokens


def _iter_model_dependents(rows: list[dict], target_model: str, *, exclude_id: str):
    """Yield `(row, binding)` for every `[ai.*]` ollama binding on `target_model`.

    **Plain ref equality** (a deliberate strengthening over the resolver-compare):
    a spec matches when `provider == "ollama"` AND `spec['model'] ==
    target_model` — so with duplicate model rows serving one ref (possible via a
    `--name` override) *any* row serving a referenced ref is protected, never
    just the one the resolver happens to rank first. `provider == "api"` never
    matches. Database dependents have their own matcher (`_iter_db_dependents`).
    """
    for row in rows:
        if row.get("id") == exclude_id:
            continue
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            continue
        ai = cfg.get("ai")
        if not isinstance(ai, dict):
            continue
        for binding, spec in ai.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("provider") == "ollama" and spec.get("model") == target_model:
                yield row, binding


def _iter_db_dependents(rows: list[dict], target_service: str, *, exclude_id: str):
    """Yield `(row, binding)` for every `[db.*]` managed binding on `target_service`.

    A spec matches when `provider == "managed"` AND `spec['database'] ==
    target_service` (the managed row's own `service_name` — the plain-ref
    equality the `[ai.*]` guard uses, mirrored). `provider == "external"`
    never matches. Deleting a database an app is bound to leaves that app
    unresolvable, so the same 409 `resource.in_use` guard protects it.
    """
    for row in rows:
        if row.get("id") == exclude_id:
            continue
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            continue
        db = cfg.get("db")
        if not isinstance(db, dict):
            continue
        for binding, spec in db.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("provider") == "managed" and spec.get("database") == target_service:
                yield row, binding


def _dep_entry(row: dict, binding: object) -> dict[str, str | None]:
    """Project a matched dependent row into the `{service, id, binding}` envelope."""
    return {"service": row.get("service_name"), "id": row.get("id"), "binding": str(binding)}


def _find_model_dependents(
    rows: list[dict], target_model: str, *, exclude_id: str
) -> list[dict[str, str | None]]:
    """Service rows whose `[ai.*]` ollama binding references `target_model`.

    Returns `[{service, id, binding}]` for every match (see
    `_iter_model_dependents` for the ref-equality contract).
    """
    return [
        _dep_entry(row, binding)
        for row, binding in _iter_model_dependents(rows, target_model, exclude_id=exclude_id)
    ]


def _find_cross_owner_dependents(
    rows: list[dict], target_model: str, *, exclude_id: str, caller_token: str | None
) -> list[dict[str, str | None]]:
    """Matched dependents whose row is owned by a token OTHER than the caller.

    Like `_find_model_dependents` but keeps only rows a foreign token owns —
    fail-closed on a NULL `submitted_by_token` (legacy/local rows count as
    not-the-caller's, matching `require_owner_or_admin`). Used as the atomic
    re-check checker for a non-admin `force` delete so a foreign `[ai.*]`
    binding committed *during* teardown still makes `force` admin-only —
    closing the pre-teardown-snapshot TOCTOU the route's fast-path guard leaves.
    """
    deps: list[dict[str, str | None]] = []
    for row, binding in _iter_model_dependents(rows, target_model, exclude_id=exclude_id):
        owner = row.get("submitted_by_token")
        if owner is not None and owner == caller_token:
            continue  # same-owner dependents are owner-forceable
        deps.append(_dep_entry(row, binding))
    return deps


async def _model_dependents(queries, job: Job) -> list[dict[str, str | None]]:
    """Resolve the live dependents of a `kind=model` row (empty for other kinds)."""
    if job.kind is not JobKind.model:
        return []
    cfg = parse_job_config(job)
    model_ref = cfg.get("model")
    if not isinstance(model_ref, str) or not model_ref:
        return []
    rows = await queries.list_workload_configs()
    return _find_model_dependents(rows, model_ref, exclude_id=job.id)


def _find_db_dependents(
    rows: list[dict], target_service: str, *, exclude_id: str
) -> list[dict[str, str | None]]:
    """Service rows whose `[db.*]` managed binding references `target_service`."""
    return [
        _dep_entry(row, binding)
        for row, binding in _iter_db_dependents(rows, target_service, exclude_id=exclude_id)
    ]


def _find_cross_owner_db_dependents(
    rows: list[dict], target_service: str, *, exclude_id: str, caller_token: str | None
) -> list[dict[str, str | None]]:
    """Matched `[db.*]` dependents whose row is owned by a token OTHER than the caller."""
    deps: list[dict[str, str | None]] = []
    for row, binding in _iter_db_dependents(rows, target_service, exclude_id=exclude_id):
        owner = row.get("submitted_by_token")
        if owner is not None and owner == caller_token:
            continue  # same-owner dependents are owner-forceable
        deps.append(_dep_entry(row, binding))
    return deps


async def _db_dependents(queries, job: Job) -> list[dict[str, str | None]]:
    """Resolve the live dependents of a `kind=database` row (empty for other kinds)."""
    if job.kind is not JobKind.database:
        return []
    if not job.service_name:
        return []
    rows = await queries.list_workload_configs()
    return _find_db_dependents(rows, job.service_name, exclude_id=job.id)


async def _resource_dependents(queries, job: Job) -> list[dict[str, str | None]]:
    """Live dependents of a bindable resource row (model or database), else empty."""
    if job.kind is JobKind.model:
        return await _model_dependents(queries, job)
    if job.kind is JobKind.database:
        return await _db_dependents(queries, job)
    return []


def _resource_label(job: Job) -> str:
    """Human label for the reference-guard messages ('Model' | 'Database')."""
    return "Database" if job.kind is JobKind.database else "Model"


def _in_use_error(
    ident: str, dependents: list[dict], *, relaunched: bool, label: str = "Model"
) -> NerditError:
    """Build the 409 `resource.in_use` envelope carrying the dependents list."""
    if relaunched:
        message = f"{label} '{ident}' gained a dependent during teardown and is being relaunched."
        hint = "Delete or rebind the dependent service(s), then retry — or pass --force."
    else:
        message = f"{label} '{ident}' is in use by {len(dependents)} service(s)."
        hint = "Delete or rebind the dependent service(s) first, or pass --force."
    return NerditError(409, "resource.in_use", message, hint=hint, dependents=dependents)


def _forbidden_cross_owner(label: str = "model") -> NerditError:
    """Build the 403 refusing a non-admin `force` over a foreign-owned dependent."""
    return NerditError(
        403,
        "forbidden",
        f"Force-deleting a {label} with dependents owned by another token requires admin.",
        hint="Ask an admin to force-delete, or delete the foreign dependents first.",
    )


async def _audit_purge(request: Request, job: Job, action: str, params: dict) -> None:
    """Insert one out-of-band `service.purge_*` audit row (best-effort, DB-only).

    Thin wrapper over `nerdit.daemon.audit.record_out_of_band`: a second row
    on top of the middleware's `service.delete` row. Params carry the relative
    key `services/<name>` only — never an absolute host path.
    """
    await record_out_of_band(
        request,
        action=action,
        target_type="service",
        target_id=job.service_name,
        params=params,
    )


async def _purge_secrets(request: Request, job: Job, name: str) -> bool:
    """Best-effort delete of a service's secrets file (both `.enc` + legacy)."""
    ok = False
    secret_mgr = getattr(request.app.state, "secret_manager", None)
    if secret_mgr is not None:
        try:
            secret_mgr.delete(name)
            ok = True
        except Exception:  # noqa: BLE001 — incl. InvalidServiceName; best-effort
            ok = False
    await _audit_purge(
        request, job, "service.purge_secrets", {"key": f"services/{name}", "purged": ok}
    )
    return ok


def _secret_file_kept(request: Request, name: str) -> bool:
    """Whether a secret file survives this delete and so needs its claim re-minted.

    A secrets file kept past the row must stay reachable by its owner and
    unclaimable by anyone else: without the claim, a stranger's fresh deploy
    of the same name would launch with the old owner's values injected. The
    mint itself runs inside `delete_service_checked`'s transaction (a NULL
    owner mints a NULL claim, admin-only, the same posture as its row); this
    probe runs under the variable writer lock through row deletion. A probe
    failure conservatively reserves the name because secrets may survive.
    """
    secret_mgr = getattr(request.app.state, "secret_manager", None)
    try:
        return secret_mgr is not None and bool(secret_mgr.exists(name))
    except Exception:  # noqa: BLE001 — best-effort, like every purge step
        logger.warning("Could not probe the secret file for '%s'; reserving its name", name)
        return True


async def _purge_workspace(request: Request, job: Job, name: str) -> bool:
    """Remove the full workspace root, including metadata, with best-effort auditing.

    Return true only when removed; absent roots, invalid names and OS failures return
    false. Caller must hold workspace_lock across row deletion and this removal;
    reacquiring the nonreentrant lock here would deadlock. Trees are not mounted into
    containers, but daemon writes and deployment snapshots require this lock.
    Workspace-only orphans are handled separately by retention.
    """
    settings = getattr(request.app.state, "settings", None)
    ok = False
    if settings is not None:
        try:
            root = workspace_root(Path(settings.data_dir).expanduser(), name)
            # `settled_to_thread`: an abandoned `rmtree` would race a fresh
            # first write with the lock already free (the uniform lock contract).
            await settled_to_thread(shutil.rmtree, root)
            ok = True
        except (WorkspaceError, OSError):
            ok = False
    await _audit_purge(
        request, job, "service.purge_workspace", {"key": f"workspaces/{name}", "purged": ok}
    )
    return ok


async def _no_live_writer(request: Request, container_id: str) -> bool:
    """Confirm a failed teardown left no running writer before purging data.

    Only a definitive container_running=False permits removal. Unknown or raising
    probes fail closed; a stopped but still-present container cannot write and is
    safe. Clean teardown paths need no probe.
    """
    runtime = request.app.state.runtime
    try:
        return await runtime.container_running(container_id) is False
    except Exception:  # noqa: BLE001 — an unanswerable probe never confirms anything
        return False


async def _purge_data(
    request: Request, job: Job, name: str, *, skip_reason: str | None = None
) -> bool:
    """Best-effort `rmtree` of `<data_dir>/services/<name>` (fail-closed seam).

    A missing dir counts as success (nothing to purge); a spec/OS failure reports
    `False` (recovery is the opt-in GC data reclaim).

    `skip_reason` short-circuits **before any filesystem access**: the caller
    could not confirm the container is gone, so the dir is left alone and the audit
    row carries the reason. When it is `None` the audit params keep their original
    two-key shape (no `reason: null` on the happy path).
    """
    if skip_reason is not None:
        await _audit_purge(
            request,
            job,
            "service.purge_data",
            {"key": f"services/{name}", "purged": False, "reason": skip_reason},
        )
        return False
    settings = getattr(request.app.state, "settings", None)
    ok = False
    if settings is not None:
        try:
            root = service_data_root(Path(settings.data_dir).expanduser(), name)
            if await asyncio.to_thread(root.exists):
                await asyncio.to_thread(shutil.rmtree, root)
            ok = True
        except (VolumeSpecError, OSError):
            ok = False
    await _audit_purge(
        request, job, "service.purge_data", {"key": f"services/{name}", "purged": ok}
    )
    return ok


def _has_live_tombstone(services_root: Path, name: str) -> bool:
    """Return True if `services_root` holds a `.trash-<name>-<nonce>` dir (blocking).

    Called in a worker thread from `_tombstone_data` to detect a concurrent
    DELETE that already renamed this database's tree aside.
    """
    try:
        with os.scandir(services_root) as it:
            return any(tombstone_service_name(e.name) == name for e in it)
    except OSError:
        return False


async def _tombstone_data(
    request: Request, job: Job, name: str, *, skip_reason: str | None = None
) -> tuple[bool, Path | None]:
    """Atomically rename service data aside before deleting its row.

    Returns:
        (False, None) on refusal, rename failure or an existing concurrent tombstone;
        (True, None) when no data or tombstone exists; otherwise (True, path) for the
        same-filesystem tombstone. Restore it on refusal or reap it after commit.

    Refusals are audited and cause the caller to abort deletion and restore running
    intent. No cross-filesystem copy is allowed.
    """
    if skip_reason is not None:
        await _audit_purge(
            request,
            job,
            "service.purge_data",
            {"key": f"services/{name}", "purged": False, "reason": skip_reason},
        )
        return False, None
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        await _audit_purge(
            request, job, "service.purge_data", {"key": f"services/{name}", "purged": False}
        )
        return False, None
    try:
        root = service_data_root(Path(settings.data_dir).expanduser(), name)
        if not await asyncio.to_thread(root.exists):
            # An absent root normally means nothing to move — UNLESS a concurrent
            # DELETE of this same database already renamed the tree into a
            # tombstone. Treating that as (True, None) would let this delete
            # commit while the other's checked delete is refused and restores the
            # tombstone — the unrecoverable Finding-#1 orphan (data restored at the
            # canonical path, row+secret gone). Abort instead, like a failed rename.
            if await asyncio.to_thread(_has_live_tombstone, root.parent, name):
                await _audit_purge(
                    request,
                    job,
                    "service.purge_data",
                    {"key": f"services/{name}", "purged": False},
                )
                return False, None
            return True, None
        tombstone = root.parent / make_tombstone_name(name)
        await asyncio.to_thread(os.rename, root, tombstone)
        return True, tombstone
    except (VolumeSpecError, OSError):
        await _audit_purge(
            request, job, "service.purge_data", {"key": f"services/{name}", "purged": False}
        )
        return False, None


async def _restore_tombstone(name: str, tombstone: Path, data_dir: Path) -> bool:
    """Rename a tombstone BACK to `<data_dir>/services/<name>` (C2, on refusal).

    Best-effort but authoritative: a `True` means the relaunch will boot on the
    original tree. A `False` (parent perms changed mid-flight — near-impossible
    under the daemon-uid posture) means the caller must NOT relaunch on an empty
    dir; it keeps `desired_state='stopped'` and surfaces a loud 500.
    """
    try:
        root = service_data_root(data_dir, name)
        await asyncio.to_thread(os.rename, tombstone, root)
        return True
    except (VolumeSpecError, OSError):
        return False


async def _undo_tombstone_or_raise(
    request: Request, ident: str, name: str, tombstone: Path, *, why: str
) -> None:
    """Rename a pre-delete tombstone back, or raise the loud `db.restore_failed`.

    Both non-committing exits from the delete's atomic span need this: the delete
    was REFUSED by the in-transaction re-check, or the row was
    already gone so this request deleted nothing. Either way the database's data
    was renamed aside before the commit point and must go back — and a rename-back
    that cannot be confirmed must never be papered over, because the alternative
    is a relaunch on an empty tree. `why` names the exit in the 500's message.
    """
    data_dir = Path(request.app.state.settings.data_dir).expanduser()
    if await _restore_tombstone(name, tombstone, data_dir):
        return
    logger.error(
        "Database delete (%s) could not restore its data from tombstone %s; "
        "row left stopped for %s",
        why,
        tombstone,
        name,
    )
    raise NerditError(
        500,
        "db.restore_failed",
        f"Database '{ident}' delete {why} but its data could not be restored; "
        "the database is left stopped to avoid an empty relaunch.",
        hint="Recover the managed data directory from its tombstone by hand, "
        "then restart the database.",
    )


async def _reap_tombstone(request: Request, job: Job, name: str, tombstone: Path | None) -> bool:
    """Best-effort `rmtree` of a committed delete's tombstone (C2, on commit).

    The row/endpoint/secret are already gone and the data is no longer at the
    canonical service path — but `purged.data` means "the bytes are gone", so
    a failed rmtree is reported honestly as `False`: the tree survives in the
    tombstone, reclaimable by the opt-in GC orphan-data pass (its base row no
    longer exists). `tombstone=None` means nothing was ever moved (the data
    root was absent); a clean purge.
    """
    purged = True
    if tombstone is not None:
        try:
            await asyncio.to_thread(shutil.rmtree, tombstone)
        except OSError:
            purged = False
            logger.warning(
                "Tombstone rmtree failed for deleted database %s; data remains in %s "
                "until the GC orphan pass reclaims it",
                name,
                tombstone.name,
                exc_info=True,
            )
    await _audit_purge(
        request, job, "service.purge_data", {"key": f"services/{name}", "purged": purged}
    )
    return purged


async def _purge_images(
    request: Request, job: Job, name: str, other_rows: list[dict]
) -> PurgeImages:
    """Best-effort removal of a deleted app's own image tags (never a model's).

    Candidate tags = every existing tag under the row's `config['image_repo']`
    (fallback `app_image_repo`) **minus** the protected set of all *other*
    live rows — so a tag another live app still needs is spared. Verifies by
    re-listing (`remove_image` swallows errors): a tag still present is reported
    `skipped` rather than claimed removed.
    """
    runtime = request.app.state.runtime
    report = PurgeImages()
    cfg = parse_job_config(job)
    repo = cfg.get("image_repo")
    if not isinstance(repo, str) or not repo:
        repo = app_image_repo(name)
    try:
        detailed = await runtime.list_images_detailed()
    except Exception:  # noqa: BLE001 — best-effort
        detailed = []
    tags = [e.get("repo_tag") or "" for e in detailed]
    protected = _protected_image_refs(other_rows, tags)
    candidates = _repo_tags(detailed, repo, protected)
    for tag in candidates:
        try:
            await runtime.remove_image(tag)
        except Exception:  # noqa: BLE001 — remove_image already swallows; guard anyway
            pass
    try:
        after = {e.get("repo_tag") for e in await runtime.list_images_detailed()}
    except Exception:  # noqa: BLE001 — best-effort
        after = set(tags)
    for tag in candidates:
        if tag in after:
            report.skipped.append({"tag": tag, "reason": "in_use_or_error"})
        else:
            report.removed.append(tag)
    await _audit_purge(
        request,
        job,
        "service.purge_images",
        {
            "key": f"services/{name}",
            "removed": report.removed,
            "skipped": [s["tag"] for s in report.skipped],
        },
    )
    return report


@router.delete(
    "/services/{ident}", response_model=ServiceDeletedResponse, operation_id="delete_service"
)
async def delete_service(
    request: Request,
    ident: str,
    purge: str = Query(
        "secrets",
        description="CSV of purge targets: secrets,data,images,workspace (default: secrets)",
    ),
    force: bool = Query(
        False,
        description="Bypass the reference guard and kill an in-flight run (may require admin)",
    ),
) -> ServiceDeletedResponse:
    """Delete a service: reference-guard, tear down, remove the row, purge on request.

    Order: validate `?purge` → model reference guard (409
    `resource.in_use` unless `?force`; a cross-owner dependent makes `force`
    admin-only) → run-race hook (409 `service.run_in_progress` unless `?force`,
    which kills the run) → synchronous teardown (container stop/kill/remove,
    release GPUs, deregister proxy route **before** releasing the endpoint) → re-run
    the guard immediately before the row delete (restoring `desired_state='running'`
    on a fresh dependent, since teardown already happened) → `delete_service` →
    best-effort purge steps (each its own out-of-band `service.purge_*` audit row).
    Authorized to the owner or an admin; audited (`service.delete`).
    """
    purge_set = _parse_purge(purge)
    request.state.audit_params = audit_params({"service": ident, "purge": sorted(purge_set)})
    queries = request.app.state.queries

    job = await _resolve_service(queries, ident)
    if job is None:
        raise _not_found(ident)
    require_owner_or_admin(request, job)

    # --- database delete is explicit-destructive ------------------------------
    # A kind=database row's data dir and minted password are useless without each
    # other; deleting the row without purging its data leaves an unrecoverable
    # orphan (initdb never re-runs on a non-empty PGDATA, so a recreated row's
    # fresh password can never match). Require `data` in the purge set — NOT
    # bypassable by `?force` (which bypasses only the reference guard, a
    # different concern). Purging `data` FORCE-INCLUDES `secrets` (coupled
    # lifecycle). Run before the reference guard so the shape error is reported
    # regardless of dependents.
    if job.kind is JobKind.database:
        if "data" not in purge_set:
            raise NerditError(
                409,
                "db.delete_requires_purge",
                f"Deleting database '{ident}' requires purging its data.",
                hint="Pass ?purge=data (also removes the minted credential); back it up "
                "first with `nerdit backup --volume <name>`.",
            )
        purge_set = purge_set | {"secrets"}
        request.state.audit_params = audit_params({"service": ident, "purge": sorted(purge_set)})

    # --- reference guard (kind=model / kind=database targets) ----------------
    label = _resource_label(job)
    dependents = await _resource_dependents(queries, job)
    if dependents:
        if not force:
            raise _in_use_error(ident, dependents, relaunched=False, label=label)
        # `force` bypasses, but a dependent owned by ANOTHER token requires admin
        # a submitter cannot force-kill a resource foreign apps rely on.
        principal = current_principal(request)
        if principal.role != TokenRole.admin:
            for dep in dependents:
                dep_id = dep.get("id")
                dep_job = await queries.get_job(dep_id) if dep_id else None
                if dep_job is None:
                    continue  # vanished mid-check — no longer a dependent
                # Fail closed: a NULL owner (legacy/local rows) is NOT the
                # caller's — same posture as require_owner_or_admin.
                dep_owner = getattr(dep_job, "submitted_by_token", None)
                if dep_owner is None or dep_owner != principal.token_id:
                    raise _forbidden_cross_owner(label.lower())
        request.state.audit_params = audit_params(
            {
                "service": ident,
                "purge": sorted(purge_set),
                "forced": True,
                "dependents": [d.get("service") for d in dependents],
            }
        )

    # --- run-race hook -------------------------------------------------------
    # `has_active_run` covers one-off runs AND `[deploy].release` executions
    # — both are rowless containers holding the service's data volume, so a
    # delete underneath either is a torn teardown.
    #
    # The rule: an UNFORCED delete during a run/release 409s; a FORCED delete
    # kills that service's run containers and proceeds. Without the escape hatch
    # a wedged run — a SIGTERM-deaf migration, a hung `runtime.run`, a docker
    # outage that never lets the bounded wait return — would pin the service
    # undeletable until the daemon restarts, which is precisely what `?force`
    # exists to prevent. Authorization needs nothing extra: the run belongs to
    # THIS service, whose ownership `require_owner_or_admin` established at the
    # top, and the branch sits AFTER the dependents guard so a non-admin still
    # cannot force past a foreign-owned dependent (the cross-owner 403 above).
    unkilled_runs: list[str] = []
    unbound_run = False
    controller = getattr(request.app.state, "service_controller", None)
    if controller is not None and controller.has_active_run(job.id):
        if not force:
            # DELETE is the only route with a ?force hatch, so it overrides the
            # shared "wait for it to finish" hint.
            raise _run_in_progress_error(
                ident,
                "delete",
                hint="Wait for the run to finish and retry, or pass ?force=true to kill it.",
            )
        # Best-effort, same swallow-everything posture as the teardown below —
        # and remembered the same way: a kill that RAISED may have left a
        # migration writing to the very tree the data purge would rmtree,
        # so those container ids are re-probed there. The registry slot is
        # deliberately NOT discarded here: `_execute_container_once` frees its
        # own slot when its wait returns, and reaching into the controller's
        # private registry from a route would be a worse leak than the slot.
        runtime = request.app.state.runtime
        killed_runs = 0
        for run_container_id in controller.active_run_container_ids_for(job.id):
            try:
                await runtime.kill(run_container_id)
            except ContainerNotFoundError:
                continue  # already gone: nothing killed, nothing left to race the purge
            except ContainerRuntimeError:
                unkilled_runs.append(run_container_id)
                continue
            killed_runs += 1
        # A slot with no container id yet is the strictly WORSE case, not a
        # harmless one: nothing was killed (there is nothing to kill), yet the
        # container may start — bind-mounting the data tree — microseconds
        # after the rmtree. Re-probed here rather than at claim time so the
        # window measured is the one that actually races the purge.
        unbound_run = controller.has_unbound_run(job.id)
        # Never a container id or the run's command — only how many were killed.
        request.state.audit_params = audit_params(
            {**request.state.audit_params, "killed_runs": killed_runs}
        )

    # The same guard for a health-gated cutover: deleting a service
    # mid-verify would tear the row (and its endpoint) out from under a verify
    # task that is about to repoint a route and promote a container. `?force`
    # cancels the verify, kills the green and unwinds `active_host_port`
    # first, so the teardown below runs against a settled row. Authorization
    # needs nothing extra — `require_owner_or_admin` ran at the top.
    unbound_cutover = False
    if controller is not None and controller.has_active_cutover(job.id):
        if not force:
            raise _cutover_in_progress_error(
                ident,
                "delete",
                hint=(
                    "Wait for the deploy to settle and retry, or pass ?force=true to "
                    "cancel it and kill the candidate container."
                ),
            )
        # Captured BEFORE the cancel, which destroys the evidence: a verify
        # still inside `runtime.run()` has no container id registered, and
        # cancelling the task cannot stop the underlying docker thread — the
        # green may start (bind-mounting the service data tree) AFTER the
        # cancel returned empty-handed. Same hazard class as the unbound run
        # above, gated the same fail-closed way at the data purge. If the id
        # bound between the check and the cancel, the cancel's late re-read
        # killed it — the flag then skips the purge conservatively, which is
        # the right side to err on.
        unbound_cutover = controller.has_unbound_cutover(job.id)
        await controller.cancel_cutover(job.id)

    # Tear down the FRESH row, not the one resolved at the top of the route. A
    # cutover that reached its commit point — either before the
    # `has_active_cutover` check above or under the cancel that followed it —
    # promoted the green onto the row, so this snapshot still names the blue the
    # commit destroyed. Tearing THAT down would leave the promoted green alive
    # (still answering the repointed dial) and would let the purge gate below count
    # a confirmed teardown while that green still bind-mounts the data tree.
    # Rebinding is safe: id/kind/service_name are immutable, and only
    # `container_id`/`status` matter downstream.
    fresh = await queries.get_job(job.id)
    if fresh is not None:
        job = fresh

    await queries.set_desired_state(job.id, "stopped")

    # Synchronous, best-effort teardown so deleting the row cannot orphan a live
    # container (the reconciler will never see this row again). Every failure is
    # swallowed (best-effort by design) — but we remember that one happened, because
    # an unconfirmed teardown must not be followed by an rmtree of the container's
    # bind-mounted data dir (the probe itself is deferred to the purge section
    # so the happy path pays nothing).
    teardown_failed = False
    if job.container_id:
        runtime = request.app.state.runtime
        try:
            await runtime.stop(job.container_id)
        except ContainerNotFoundError:
            # NOT a failure: "already gone" is the strongest confirmation there is
            # that nothing is writing to the data dir. The reconcile loop routinely
            # gets here first (this route sets desired_state=stopped before tearing
            # down), and counting that as an unconfirmed teardown would arm the purge
            # gate on a perfectly healthy delete and silently skip the purge.
            pass
        except ContainerRuntimeError:
            teardown_failed = True
            try:
                await runtime.kill(job.container_id)
            except ContainerRuntimeError:
                pass
        try:
            await runtime.remove(job.container_id, force=True)
        except ContainerNotFoundError:
            pass
        except ContainerRuntimeError:
            teardown_failed = True

    await queries.release_gpus(job.id)

    # Tombstone database data before deleting its row/credential: surviving data paired
    # with a freshly minted password would be unusable. Use reversible same-filesystem
    # rename because the atomic dependent recheck may still refuse deletion. Restore
    # on refusal, reap after commit; if rename/writer safety cannot be confirmed,
    # abort and restore running intent. Database data purge requires force.
    db_tombstone: Path | None = None
    if job.kind is JobKind.database and job.service_name:
        skip_reason: str | None = None
        if (
            teardown_failed
            and job.container_id
            and not await _no_live_writer(request, job.container_id)
        ):
            skip_reason = "container_alive"
        moved, db_tombstone = await _tombstone_data(
            request, job, job.service_name, skip_reason=skip_reason
        )
        if not moved:
            # Nothing has been deleted yet (row/endpoint/secret all intact); the
            # container is torn down, so bring the DB back rather than leave it
            # half-gone. A retry after fixing ownership completes the delete.
            await queries.set_desired_state(job.id, "running")
            raise NerditError(
                500,
                "db.purge_failed",
                f"Could not remove the data directory for database '{ident}'; "
                "the delete was aborted and the database left intact.",
                hint="Ensure the database container is stopped and that the daemon "
                "user owns its managed data directory, then retry the delete.",
            )

    # Recheck dependents inside delete_service_checked's BEGIN IMMEDIATE. Non-force
    # resource deletion refuses any fresh dependent; non-admin force refuses fresh
    # cross-owner dependents. Admin force and non-resource deletes need no checker.
    # Database checks use service name, model checks use the model ref. If refused
    # after teardown, restore running intent so reconciliation relaunches using the
    # retained endpoint; release that endpoint only after commit.
    checker = None
    cross_owner = False
    if job.kind is JobKind.model:
        cfg = parse_job_config(job)
        model_ref = cfg.get("model")
        if isinstance(model_ref, str) and model_ref:
            target_id = job.id
            if not force:

                def checker(rows: list[dict]) -> list[dict[str, str | None]]:
                    return _find_model_dependents(rows, model_ref, exclude_id=target_id)

            else:
                principal = current_principal(request)
                if principal.role != TokenRole.admin:
                    cross_owner = True
                    caller_token = principal.token_id

                    def checker(rows: list[dict]) -> list[dict[str, str | None]]:
                        return _find_cross_owner_dependents(
                            rows, model_ref, exclude_id=target_id, caller_token=caller_token
                        )

    elif job.kind is JobKind.database and job.service_name:
        db_ref = job.service_name
        target_id = job.id
        if not force:

            def checker(rows: list[dict]) -> list[dict[str, str | None]]:
                return _find_db_dependents(rows, db_ref, exclude_id=target_id)

        else:
            principal = current_principal(request)
            if principal.role != TokenRole.admin:
                cross_owner = True
                caller_token = principal.token_id

                def checker(rows: list[dict]) -> list[dict[str, str | None]]:
                    return _find_cross_owner_db_dependents(
                        rows, db_ref, exclude_id=target_id, caller_token=caller_token
                    )

    # For workspace purge, wait for the workspace lock and hold it across row deletion
    # and rmtree. Earlier writes finish before deletion; concurrent writes receive 409;
    # later writes create a fresh workspace. Lock order is workspace → variables → DB;
    # never acquire a workspace lock inside a DB transaction. ExitStack releases it on
    # all early errors. Read share removal from the delete transaction's rowcount,
    # not a pre-read that could miss a racing share PUT. Emit only the removal fact,
    # never the hosted URL.
    had_share = False

    def _note_share_removed(removed: bool) -> None:
        nonlocal had_share
        had_share = removed

    # The same seam for custom domains, and for the same reason: the
    # NAMES are unrecoverable after the delete (the rows go with the service
    # inside `delete_service_checked`'s transaction) and unreadable before it
    # without a race, so the transaction reports them. A list, not a boolean —
    # one `domain.removed` edge per name is what a consumer needs to stop
    # believing this node still answers for them.
    removed_domains: list[str] = []

    def _note_domains_removed(names: list[str]) -> None:
        removed_domains.extend(names)

    # The secret file's claim is re-minted INSIDE the delete
    # transaction, never after it: post-commit, a stranger's fresh row could
    # land before the mint and launch with the old owner's values. Minted even
    # when `secrets` is being purged: the purge is best-effort, and the claim
    # is dropped only once the file is confirmed gone.
    secrets_reclaimed: bool | None = None

    def _note_secrets_reclaimed(minted: bool) -> None:
        nonlocal secrets_reclaimed
        secrets_reclaimed = minted

    async with contextlib.AsyncExitStack() as stack:
        if "workspace" in purge_set and job.service_name:
            await stack.enter_async_context(workspace_lock(job.service_name))
        # Serialize the file probe and claim mint with first-time secret writes.
        # Release before the tail, which takes the same lock for secret purge.
        async with variable_write_lock(request.app):
            reclaim = (
                _note_secrets_reclaimed
                if job.service_name and _secret_file_kept(request, job.service_name)
                else None
            )
            fresh_deps = await queries.delete_service_checked(
                job.id,
                checker,
                on_share_removed=_note_share_removed,
                on_domains_removed=_note_domains_removed,
                on_secrets_reclaimed=reclaim,
            )
        if fresh_deps is None:
            # The row was ALREADY GONE: this
            # request deleted nothing, so it has earned no name-based side effect.
            # The shape is a stale delete — B stalled in the swallow-everything
            # teardown (up to a ~10 s docker stop grace) while A completed the same
            # delete, the owner legitimately re-created the name, and the workspace
            # lock (free again) let B straight through. Every effect below this
            # branch is keyed by NAME, not by id — proxy deregister, secrets, data,
            # images, workspace — so continuing would destroy the REPLACEMENT's
            # state while leaving its brand-new row alive. 404 is exactly what this
            # request would have received a moment later at the top-of-route
            # resolve; the row's actual deleter already reported the delete.
            #
            # The tombstone comes back first: a database delete renames the data
            # tree aside BEFORE this commit point, and that tree now belongs to the
            # replacement.
            if db_tombstone is not None and job.service_name:
                await _undo_tombstone_or_raise(
                    request, ident, job.service_name, db_tombstone, why="found no such row"
                )
            raise _not_found(ident)
        if fresh_deps:
            # C2: the delete is REFUSED (a dependent committed during teardown). The
            # database's data was renamed aside pre-delete; rename it BACK so the
            # relaunch boots on the original tree, not an empty dir. A failed
            # rename-back must NOT relaunch on empty: keep desired_state='stopped'
            # (set during teardown, never restored here), log the tombstone loud, and
            # surface a 500 so the operator can recover the tree by hand.
            if db_tombstone is not None and job.service_name:
                await _undo_tombstone_or_raise(
                    request, ident, job.service_name, db_tombstone, why="was refused"
                )
            await queries.set_desired_state(job.id, "running")
            if cross_owner:
                raise _forbidden_cross_owner(label.lower())
            raise _in_use_error(ident, fresh_deps, relaunched=True, label=label)

        # Finish post-commit cleanup before releasing the workspace lock: retries now 404.
        # Run the tail in its own task using _detached plus anyio shielding, so caller
        # cancellation cannot discard requested purges. A second raw Task.cancel may still
        # interrupt settlement, and uvicorn's graceful-shutdown deadline bounds the work.
        # Keep absent-row refusal outside this tail; it must fail fast without destruction.
        async def _post_commit_tail() -> ServiceDeletedResponse:
            # The delete has COMMITTED and it took the share row
            # with it, so the exposure is over: tell the feed. Emitted here, in
            # the cancellation-settled tail, for the same reason every other
            # post-commit effect is — a client disconnect must not lose the one
            # record that an app stopped being reachable. `reason` separates
            # this from an owner's explicit `nerdit unshare`.
            if had_share:
                recorder = get_recorder()
                if recorder is not None:
                    await recorder.record(
                        "share.removed",
                        kind=job.kind.value,
                        service_name=job.service_name,
                        reason="service_deleted",
                    )

            # The domain cascade, one edge per name. Emitted here for
            # the same reason and with the same `reason` split as the share
            # edge above; the name itself is public DNS the operator chose, so
            # it travels in `data` (S-W9). The live Host routes go with the
            # `proxy.deregister` below, which is live-derived and removes
            # every route the service owns.
            if removed_domains:
                recorder = get_recorder()
                if recorder is not None:
                    for removed_domain in removed_domains:
                        await recorder.record(
                            "domain.removed",
                            kind=job.kind.value,
                            service_name=job.service_name,
                            reason="service_deleted",
                            data={"domain": removed_domain},
                        )

            # The row AND its endpoint are now gone — the endpoint row is deleted inside
            # delete_service_checked's transaction (its job_id FK to jobs(id) forces the
            # in-txn ordering, and the rollback on a blocked re-check keeps it, so a
            # REFUSED delete never surrenders the model's stable host port). Only the
            # proxy route remains: deregister it now (external Caddy state can't join the
            # DB transaction; the proxy reconcile loop would also prune it, but that
            # leaves a misroute window). Best-effort, no-op when the proxy is off.
            if job.service_name:
                proxy = getattr(request.app.state, "proxy_manager", None)
                if proxy is not None:
                    await proxy.deregister(job.service_name)
            # This path never reaches `_teardown_to_stopped`, so drop the
            # controller's per-job memory here or it outlives the row.
            if controller is not None:
                controller.forget(job.id)

            # --- best-effort purge steps (each its own out-of-band audit row) ----
            purged: PurgeReport | None = None
            name = job.service_name
            if name:
                purged = PurgeReport()
                if "secrets" in purge_set:
                    # Under the writers' lock, like `DELETE /secrets/{name}`, so
                    # no parked write straddles the release. The row went long
                    # before this tail: a name that has since gained a row, or a
                    # claim that is not the one this delete minted for the row's
                    # owner, is somebody's NEW scope and its file is not ours to
                    # delete by name. `secrets_reclaimed` is history, not a
                    # verdict: the owner may have released the name and a
                    # stranger claimed it since, so the CURRENT claim is judged.
                    # ponytail: judged by owner, not claim identity — the owner's
                    # own re-claim inside its delete tail is purged as requested.
                    async with variable_write_lock(request.app):
                        # Row and claim in ONE snapshot: a deploy turns a claim
                        # into a row in one transaction, and two reads could
                        # miss both.
                        retaken = await queries.name_retaken(
                            name, job.submitted_by_token, any_claim=not secrets_reclaimed
                        )
                        if retaken:
                            # Still one `service.purge_secrets` row per requested
                            # purge: a skip is recorded, never a silence.
                            purged.secrets = False
                            await _audit_purge(
                                request,
                                job,
                                "service.purge_secrets",
                                {
                                    "key": f"services/{name}",
                                    "purged": False,
                                    "reason": "name_retaken",
                                },
                            )
                        else:
                            purged.secrets = await _purge_secrets(request, job, name)
                        if purged.secrets and secrets_reclaimed:
                            # File gone → release the name. A failed purge keeps
                            # the claim, so the leftover stays the owner's.
                            await queries.delete_secret_claim(name)
                elif secrets_reclaimed is not None:
                    request.state.audit_params = audit_params(
                        {**request.state.audit_params, "secrets_reclaimed": secrets_reclaimed}
                    )
                if "data" in purge_set:
                    if job.kind is JobKind.database:
                        # C2: the data was renamed to a tombstone pre-delete (the delete
                        # would have aborted otherwise) and the delete has now COMMITTED,
                        # so reap the tombstone best-effort and report data purged. A
                        # leaked tombstone is GC-reclaimable (its base row is gone).
                        purged.data = await _reap_tombstone(request, job, name, db_tombstone)
                    else:
                        # Never rmtree the bind-mounted data dir after an UNCONFIRMED
                        # teardown: the block above swallows ContainerRuntimeError, so a failed
                        # stop/kill/remove can leave the container alive and still writing. Probe
                        # only in that case; a container we cannot confirm gone means the rmtree is
                        # SKIPPED (purged.data false + an audited reason) rather than a half-removed
                        # dir repopulated by an unmanaged writer. Secrets/images have no live-writer
                        # hazard and stay ungated.
                        skip_reason = None
                        if (
                            teardown_failed
                            and job.container_id
                            and not await _no_live_writer(request, job.container_id)
                        ):
                            skip_reason = "container_alive"
                        # The same rule for a forced delete's run containers: only
                        # the kills that RAISED are probed — a confirmed kill (or an
                        # already-gone container) writes nothing — and an unconfirmed one
                        # holds the same bind-mounted tree, so it blocks the rmtree too.
                        if skip_reason is None:
                            for run_container_id in unkilled_runs:
                                if not await _no_live_writer(request, run_container_id):
                                    skip_reason = "run_alive"
                                    break
                        # And the case with nothing to probe at all: a slot that
                        # was still UNBOUND when the kill loop ran. Nothing was killed
                        # because no container existed yet, but the run is mid-launch
                        # and its bind mount lands on the tree we are about to remove —
                        # Docker would then recreate the path root-owned under a
                        # service that no longer exists. Strictly less confirmable than
                        # a kill that raised, so it fails closed the same way.
                        if skip_reason is None and unbound_run:
                            skip_reason = "run_unbound"
                        # A cutover green that was still unbound when cancelled:
                        # identical shape — nothing was killed because nothing existed
                        # yet, and the launch thread may mount the tree microseconds
                        # after the rmtree.
                        if skip_reason is None and unbound_cutover:
                            skip_reason = "cutover_unbound"
                        # The tree is keyed by name alone: a same-name deploy that
                        # landed after the commit owns it now. Only a ROW spares it —
                        # a foreign claim does not, or the old owner's data would
                        # pass to the claimant's first deploy.
                        # ponytail: check-then-rmtree leaves the rmtree's own duration
                        # as a window; upgrade = pre-commit tombstone rename as the
                        # database C2 path does.
                        if (
                            skip_reason is None
                            and await queries.get_service_by_name(name) is not None
                        ):
                            skip_reason = "name_retaken"
                        purged.data = await _purge_data(request, job, name, skip_reason=skip_reason)
                # Only a plain service owns a per-app image worth removing; model
                # and database rows share an upstream image (ollama/postgres) that other
                # rows may reuse, so `images` is a no-op for them (positive-form kind
                # check so a fourth kind can never accidentally purge a shared image).
                if "images" in purge_set and job.kind is JobKind.service:
                    all_rows = await queries.list_workload_configs()
                    other_rows = [r for r in all_rows if r.get("id") != job.id]
                    purged.images = await _purge_images(request, job, name, other_rows)
                # The agent workspace tree, if this name ever had one. No
                # kind gate: only a service row can carry a workspace, and a model/database
                # name simply has no such dir (reported honestly as `workspace: false`).
                if "workspace" in purge_set:
                    purged.workspace = await _purge_workspace(request, job, name)

            return ServiceDeletedResponse(id=job.id, name=name, deleted=True, purged=purged)

        tail = asyncio.ensure_future(_post_commit_tail())
        try:
            return await asyncio.shield(tail)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                with contextlib.suppress(BaseException):
                    await tail
            raise
