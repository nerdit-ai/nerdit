"""Write, read and deploy text workspaces with sidecar-based ownership.

meta.json owns the workspace independently of service rows. When a row exists,
its owner must agree; admins bypass both checks. Filesystem operations live in
core.workspaces; routes own authorization and audit shaping.

Deploy snapshots the tree under its lock and extracts the ZIP to disposable
upload space with the usual traversal/zip-bomb guards. Never pass the live tree
to _finalize_deploy: build cleanup would remove the user's working copy.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from nerdit.core import workspaces as core_workspaces
from nerdit.core.workspaces import WorkspaceError
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    Principal,
    _check_owner,
    current_principal,
    require_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.deploy_pipeline import _finalize_deploy, reject_non_service_row
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.services import reject_reserved_name
from nerdit.daemon.schemas.workspaces import (
    WorkspaceDeployRequest,
    WorkspaceListResponse,
    WorkspaceMetaView,
    WorkspaceWriteRequest,
    WorkspaceWriteResponse,
)
from nerdit.daemon.uploads import extract_upload
from nerdit.db.models import Job, TokenRole
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_nerdit(exc: WorkspaceError) -> NerditError:
    """Translate a core workspace failure 1:1 into the structured envelope.

    The `GitSourceError` mapping precedent, plus the `extra` keywords
    (`detail`, `limit`) — the caps carry their number both ways so a REST
    caller reads `detail.limit` and an MCP caller reads the top-level
    `limit` that survives `_call`'s merge (D-P29-5).
    """
    return NerditError(exc.status_code, exc.code, exc.message, exc.hint, **exc.extra)


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.settings.data_dir).expanduser()


def _not_found(name: str) -> NerditError:
    return NerditError(
        404,
        "workspace.not_found",
        f"No workspace '{name}'.",
        hint="Create it by writing a file with `write_app_files`.",
    )


def _lock_busy(name: str) -> NerditError:
    return NerditError(
        409,
        "workspace.deploy_in_progress",
        f"Workspace '{name}' is locked by a concurrent operation "
        "(write, read, deploy snapshot, or delete).",
        hint="Retry after the in-flight call returns.",
    )


def _require_workspace_owner(request: Request, name: str, meta: dict[str, Any]) -> Principal:
    """Owner-or-admin on a `meta.json`-owned workspace (D-P29-9).

    The shared `nerdit.daemon.auth._check_owner` tree with the sidecar as
    the owner of record — a workspace may have no service row at all, so the
    `Job`-shaped `nerdit.daemon.auth.require_owner_or_admin` cannot be
    used directly. Same three outcomes: admin bypass, NULL owner is admin-only
    (a hand-made sidecar never grants access), token mismatch is a plain 403; a
    scope narrowed after the first write still binds, and the scope miss reuses
    `_scope_denial` so the message shape is identical to every other miss.
    """
    return _check_owner(
        request,
        meta.get("owner_token_id"),
        name,
        NerditError(
            403,
            "forbidden",
            f"You do not have permission to act on workspace '{name}'.",
            hint="Only the token that created the workspace or an admin may use it.",
        ),
    )


def _check_row_owner(
    principal: Principal, existing: Job | None, meta: dict[str, Any], name: str
) -> None:
    """Cross-check a service row's owner against the workspace's (D-P29-9).

    A workspace and a same-named service are two objects with two owners; if
    they ever disagree, the caller who passed the sidecar gate would otherwise
    be deploying over someone else's service. Admins bypass (they pass both
    gates anyway).
    """
    if existing is None or principal.role == TokenRole.admin:
        return
    if getattr(existing, "submitted_by_token", None) != meta.get("owner_token_id"):
        raise NerditError(
            403,
            "forbidden",
            f"Service '{name}' is owned by a different token than its workspace.",
            hint="An admin must reconcile the two owners.",
        )


@router.put("/workspaces/{name}/files", status_code=200, operation_id="write_workspace_files")
async def write_workspace_files(
    request: Request, name: str, body: WorkspaceWriteRequest
) -> WorkspaceWriteResponse:
    """Write and/or delete a batch of files in the workspace named `name`.

    All-or-nothing (D-P29-1): every entry is validated before the first byte
    lands, so a rejected batch never leaves a half-applied tree behind. The
    first successful write stamps the caller as the workspace owner.

    Every meta-based decision — the sidecar read, the service-row read and the
    whole owner tree — happens **inside** the per-workspace lock, so there is no
    stale copy to re-validate: a racing first-writer that lost the lock race
    re-reads and sees the winner's freshly stamped sidecar. Awaiting the row
    query under the lock is fine; the section stays sub-second (no build, no
    network).
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    try:
        core_workspaces.validate_workspace_name(name)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    reject_reserved_name(name)
    # Hand-built params, never the raw body (the license.install precedent):
    # file CONTENT must not reach the audit store even in a rejected request.
    request.state.audit_params = {"files": len(body.files), "deleted": len(body.delete)}
    # (P25 D-P25-3 leg b) The path name IS the target identity, and a first
    # write is a name-targeted create — the same gate `POST /deploy` uses.
    require_service_scope(request, name)

    data_dir = _data_dir(request)
    queries = request.app.state.queries

    lock = core_workspaces.workspace_lock(name)
    if lock.locked():
        # Fail fast rather than queue behind the snapshot (D-P29-10): the zip is
        # short, and an agent that gets a 409 retries a batch it still holds.
        raise _lock_busy(name)
    async with lock:
        try:
            meta = core_workspaces.read_meta(data_dir, name)
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc
        existing = await queries.get_service_by_name(name)
        reject_non_service_row(existing, name)
        if meta is not None:
            principal = _require_workspace_owner(request, name, meta)
            _check_row_owner(principal, existing, meta, name)
        else:
            # A FIRST write claims the name. When a service row already carries
            # that name, the row's owner is the only one who may claim it —
            # otherwise any unscoped submitter takes a stranger's app workspace
            # (an empty batch is enough) and `_check_row_owner` then locks the
            # real owner out of their own app.
            principal = current_principal(request)
            if existing is not None:
                require_owner_or_admin(request, existing)
        try:
            # `settled_to_thread`, never a bare `to_thread`: a client
            # disconnect (or the shutdown cancel) must not free the lock with
            # the batch half-written. The cancelled request's batch therefore
            # LANDS before the lock releases — and since no `@_serialized` DB
            # writer committed, the P22 machinery deletes the idempotency claim,
            # so a retry re-runs the same batch onto identical content.
            summary = await core_workspaces.settled_to_thread(
                core_workspaces.write_files,
                data_dir,
                name,
                body.files,
                body.delete,
                owner_token_id=principal.token_id,
                existing_meta=meta,
            )
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc

    # Names, hashes and counts — never content (plan §2).
    request.state.audit_params = {
        "files": summary["written"],
        "deleted": summary["deleted"],
        "total_bytes": summary["total_bytes"],
        "sha256s": [entry["sha256"] for entry in summary["files"]],
    }
    return WorkspaceWriteResponse(**summary)


async def _load_owned_workspace(request: Request, name: str) -> dict[str, Any]:
    """Require an existing workspace, then owner/admin, while the caller holds its lock.

    Keep authorization and the subsequent filesystem read in one lock span so purge
    and recreation cannot expose another owner's generation. Never reacquire the
    nonreentrant workspace lock here.
    """
    try:
        core_workspaces.validate_workspace_name(name)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    try:
        meta = core_workspaces.read_meta(_data_dir(request), name)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    if meta is None:
        raise _not_found(name)
    principal = _require_workspace_owner(request, name, meta)
    existing = await request.app.state.queries.get_service_by_name(name)
    _check_row_owner(principal, existing, meta, name)
    return meta


@router.get("/workspaces/{name}", status_code=200, operation_id="get_workspace")
async def get_workspace(request: Request, name: str) -> WorkspaceListResponse:
    """List a workspace: per-file path/size/sha256/mtime, totals, owner metadata.

    Owner-or-admin, not role-only: workspace content is the caller's own source
    code (D-P29-9, the named deviation from "reads stay role-only"). Bounded by
    `WORKSPACE_MAX_FILES` by construction, so there is no pagination.

    Authorization and the listing share ONE workspace-lock span (review round-2)
    so no purge+recreate can slip a foreign generation between them. The read
    path **waits** for the lock rather than fail-fast 409-ing: a read is
    sub-second (a bounded stat+hash walk), and the delete's wait posture is the
    precedent. A writer arriving during a read gets the documented 409.
    """
    try:
        core_workspaces.validate_workspace_name(name)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    async with core_workspaces.workspace_lock(name):
        meta = await _load_owned_workspace(request, name)
        try:
            listing = await core_workspaces.settled_to_thread(
                core_workspaces.list_files, _data_dir(request), name
            )
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc
    # The sidecar is daemon-authored but lives on disk; project it through the
    # view so an unexpected key can never reach the response.
    return WorkspaceListResponse(name=name, meta=WorkspaceMetaView.model_validate(meta), **listing)


@router.get(
    "/workspaces/{name}/files/{path:path}",
    status_code=200,
    operation_id="read_workspace_file",
    response_class=PlainTextResponse,
)
async def read_workspace_file(request: Request, name: str, path: str) -> PlainTextResponse:
    """Return one workspace file as `text/plain; charset=utf-8`.

    Same owner gate and same path guard family as the writes; ≤ 256 KiB by
    construction (the per-file cap is enforced at write time).

    Both validations run BEFORE the lock (the lock registry is keyed by a
    validated name, and an invalid path must 422 without queueing behind an
    in-flight write); the owner gate and the file read then share ONE lock span,
    waiting rather than 409-ing — see `get_workspace`.
    """
    try:
        core_workspaces.validate_workspace_name(name)
        core_workspaces.validate_file_path(path)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    async with core_workspaces.workspace_lock(name):
        await _load_owned_workspace(request, name)
        try:
            content = await core_workspaces.settled_to_thread(
                core_workspaces.read_file, _data_dir(request), name, path
            )
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


@router.post("/workspaces/{name}/deploy", status_code=201, operation_id="deploy_workspace")
async def deploy_workspace(
    request: Request,
    name: str,
    body: WorkspaceDeployRequest | None = None,
    dry_run: bool = Query(
        False, description="Validate + return a plan diff without writing anything."
    ),
):
    """Snapshot an authorized workspace and deploy through the shared pipeline.

    Authorize and zip under one workspace lock; use a disposable extracted context
    so cleanup preserves the working tree. Fresh service ownership comes from the
    sidecar, even when an admin acts; redeploy preserves its existing owner. Audit
    attributes the actor separately.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    if dry_run:
        request.state.audit_action = "deploy.workspace_plan"
    try:
        core_workspaces.validate_workspace_name(name)
    except WorkspaceError as exc:
        raise _to_nerdit(exc) from exc
    reject_reserved_name(name)
    opts = body or WorkspaceDeployRequest()
    request.state.audit_params = audit_params(
        {
            "name": name,
            "port": opts.port,
            "gpus": opts.gpus,
            "start": opts.start,
            "env": opts.env,
        }
    )

    data_dir = _data_dir(request)
    settings = request.app.state.settings
    queries = request.app.state.queries
    # (P25 D-P25-3 leg b) Scope before the sidecar gate, mirroring the write.
    require_service_scope(request, name)

    lock = core_workspaces.workspace_lock(name)
    if lock.locked():
        raise _lock_busy(name)
    async with lock:
        try:
            meta = core_workspaces.read_meta(data_dir, name)
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc
        if meta is None:
            raise _not_found(name)
        principal = _require_workspace_owner(request, name, meta)

        existing = await queries.get_service_by_name(name)
        reject_non_service_row(existing, name)
        # Row parity with `deploy()`: an existing service row is authorized
        # pre-ingress, before anything expensive (the snapshot, the extraction).
        if existing is not None:
            require_owner_or_admin(request, existing)
        _check_row_owner(principal, existing, meta, name)

        try:
            tree = core_workspaces.tree_root(data_dir, name)
            # Read-only, so an abandoned worker is harmless — but it runs under
            # the lock, and the lock contract is uniform (one grep, no exceptions).
            zip_bytes = await core_workspaces.settled_to_thread(core_workspaces.zip_workspace, tree)
        except WorkspaceError as exc:
            raise _to_nerdit(exc) from exc
    # The lock covers the owner decision + the snapshot ONLY (D-P29-10) — both
    # therefore see the same tree state — and never the build: it is off-tick as
    # ever, and holding the lock across the extraction would block edits for
    # minutes.

    # NEVER pass `tree` itself: `_finalize_deploy` rmtrees its context on any
    # failure and the off-tick builder rmtrees it on success. The context must be
    # disposable, so the snapshot goes through the very extraction the ZIP route
    # uses (zip-bomb + traversal guards included).
    upload = UploadFile(file=io.BytesIO(zip_bytes), filename=f"{name}-workspace.zip")
    context_dir = await extract_upload(
        upload,
        max_bytes=settings.daemon.max_upload_bytes,
        upload_root=Path(settings.daemon.upload_dir).expanduser(),
        dest_id=generate_id(),
    )

    result = await _finalize_deploy(
        request,
        context_dir,
        name=name,
        port=opts.port,
        gpus=opts.gpus,
        start=opts.start,
        build_settings=(
            opts.build_settings.model_dump(exclude_unset=True)
            if opts.build_settings is not None
            else None
        ),
        health=opts.health,
        env=opts.env,
        vendor=opts.vendor,
        source_meta={"type": "workspace"},
        # (review round-1) A FRESH row is born owned by the SIDECAR owner, not
        # by whoever pressed deploy: an admin deploying a submitter's workspace
        # would otherwise create an admin-owned row that `_check_row_owner`
        # then 403s the real owner out of — a permanent, admin-only-recoverable
        # lockout. Same posture the redeploy path already has (quota and
        # ownership follow the service OWNER, never the actor). A NULL sidecar
        # owner (an admin-only workspace) falls back to the acting principal.
        owner_token_id=meta.get("owner_token_id"),
        dry_run=dry_run,
    )
    if dry_run:
        return JSONResponse(status_code=200, content=result)
    return result
