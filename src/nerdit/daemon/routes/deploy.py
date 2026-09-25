"""Deploy uploaded or git-sourced apps through the shared build pipeline.

Authorized, idempotent, audited writes prepare a build context and desired-state
row; the controller builds and launches asynchronously. Redeploy keeps the old
container serving while building a new version on the same endpoint. Rollback
swaps to previous_image without rebuilding.

Parse AI bindings server-side and persist specifications, never resolved values.
Ollama bindings require a nonterminal model row. Redeploy replaces binding
specifications; rollback leaves them intact. Mounted under /api only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from nerdit.config.build import BuildSettings
from nerdit.config.project import SECRET_REF_RE, rewrite_vars_ref
from nerdit.core.gitsource import (
    GITHUB_INSTALLATION_REF,
    GitSourceError,
    GitSourceInfo,
    git_source_meta,
)
from nerdit.core.jobconfig import parse_job_config
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    current_principal,
    require_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.deploy_pipeline import (
    _finalize_deploy,
    _read_git_source,
    _stamp_queued,
    _validate_request_name,
    clone_into_uploads,
    private_repo_hint,
    redeploy_from_source,
    reject_non_service_row,
    resolve_git_token,
    validate_git_coordinates,
)
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.services import reject_reserved_name
from nerdit.daemon.secret_scope import (
    claim_owned_by_caller,
    project_owned_by_caller,
    reject_foreign_claim,
)
from nerdit.daemon.uploads import extract_upload
from nerdit.daemon.views.hosted import load_hosted_context
from nerdit.daemon.views.service import (
    _cutover_in_progress_error,
    _run_in_progress_error,
    service_view,
)
from nerdit.db.models import GitDeployRequest, JobStatus, TokenRole
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_env(env: str | None) -> dict[str, str | None] | None:
    """Parse the optional `env` form field (a JSON object) into a str→(str|None) map.

    A JSON `null` value is preserved as `None` (a null-delete on redeploy —
    the merge pops it; a fresh deploy drops it), never stringified to `"None"`.
    """
    if not env:
        return None
    try:
        decoded = json.loads(env)
    except json.JSONDecodeError as exc:
        raise NerditError(400, "deploy.invalid", "env must be a JSON object.") from exc
    if not isinstance(decoded, dict):
        raise NerditError(400, "deploy.invalid", "env must be a JSON object {KEY: value}.")
    return {str(k): (None if v is None else str(v)) for k, v in decoded.items()}


@router.post("/deploy", status_code=201, operation_id="deploy_app")
async def deploy(
    request: Request,
    archive: UploadFile,
    name: str = Form(...),
    port: int | None = Form(None),
    gpus: int | None = Form(None),
    start: str | None = Form(None),
    health: str | None = Form(None),
    env: str | None = Form(None),
    build_settings: str | None = Form(None),
    vendor: str | None = Form(None),
    dry_run: bool = Query(
        False, description="Validate + return a plan diff without writing anything."
    ),
):
    """Build + register an app folder as a service; the controller converges it.

    `?dry_run=true` runs every validator but persists nothing and returns the
    1.10 plan diff (status 200); audited as `deploy.plan`.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    if dry_run:
        request.state.audit_action = "deploy.plan"
    request.state.audit_params = audit_params(
        {"name": name, "port": port, "gpus": gpus, "start": start, "env": env}
    )
    reject_reserved_name(name)
    # Validate the form name before extraction, so a bad name never costs the
    # upload and the error is attributed to the request field rather than to
    # the ZIP's nerdit.toml.
    _validate_request_name(name)
    # Attribute the audit row to the validated app name (the path rules leave
    # deploy.create/plan's target null — the name rides in the form body).
    request.state.audit_target = name
    queries = request.app.state.queries
    settings = request.app.state.settings

    existing = await queries.get_service_by_name(name)
    # A model row is not a redeploy target (fast path — rejects before extract).
    reject_non_service_row(existing, name)
    # Authorize before extraction, so an unauthorized caller never costs the
    # upload. The form `name` IS the service identity, so scope is checked
    # first (widened by the row's project, D-P40-7): an out-of-scope caller
    # gets the one scope 403 whether or not a row exists. A redeploy then needs
    # owner-or-admin on the existing row; a fresh deploy only the role above.
    require_service_scope(request, name, project=existing.project if existing else None)
    if existing is not None:
        require_owner_or_admin(request, existing)
    # A name another token reserved by setting its secrets is refused before
    # the upload is extracted; the row transaction re-checks it.
    if existing is None:
        await reject_foreign_claim(request, name)
    parsed_env = _parse_env(env)
    parsed_build = None
    if build_settings is not None:
        try:
            parsed_build = BuildSettings.model_validate_json(build_settings).model_dump(
                exclude_unset=True
            )
        except ValidationError:
            raise NerditError(
                422, "deploy.invalid_build_settings", "Invalid build_settings JSON object."
            ) from None

    # Extract the folder into daemon-managed upload space (shared guards).
    context_dir = await extract_upload(
        archive,
        max_bytes=settings.daemon.max_upload_bytes,
        upload_root=Path(settings.daemon.upload_dir).expanduser(),
        dest_id=generate_id(),
    )

    result = await _finalize_deploy(
        request,
        context_dir,
        name=name,
        port=port,
        gpus=gpus,
        start=start,
        health=health,
        env=parsed_env,
        build_settings=parsed_build,
        vendor=vendor,
        source_meta={"type": "zip"},
        dry_run=dry_run,
    )
    if dry_run:
        return JSONResponse(status_code=200, content=result)
    return result


async def _resolve_token_ref(
    request: Request,
    name: str,
    token_ref: str | None,
    *,
    service_owned: bool,
    repo_url: str,
    project_id: str | None = None,
) -> str | None:
    """Resolve a syntax-validated git credential reference for a fresh deploy.

    The shared resolution lives in `deploy_pipeline.resolve_git_token`; this
    wrapper decides which scopes are proven. The service scope is the caller's
    only via an existing row it passed the owner check on, or a claim it minted
    by setting the secrets first: old secret files may outlive deleted
    rows, so any other fresh deployment may use shared refs only. The project
    is the row's `project_id`, else the project named after the label.

    Raises:
        NerditError: 403 `deploy.git_token_scope` when a service-scoped ref
            names no scope the caller owns; 422 `deploy.git_token_unresolved`
            when the ref resolves nowhere.
    """
    if token_ref is None:
        return None
    mgr = getattr(request.app.state, "secret_manager", None)
    if token_ref == GITHUB_INSTALLATION_REF:
        return await resolve_git_token(
            request, mgr, name, token_ref, repo_url=repo_url, include_service=False, project=None
        )
    token_ref = rewrite_vars_ref(token_ref) or ""  # D-P40-9 alias, idempotent
    match = SECRET_REF_RE.match(token_ref)
    assert match is not None  # caller pre-validated against SECRET_REF_RE
    scope, key = match.group(1), match.group(2)
    owned = service_owned or await claim_owned_by_caller(request, name)
    queries = request.app.state.queries
    project = (
        await queries.get_project(project_id)
        if project_id is not None
        else await queries.get_project_by_name(name)
    )
    if not owned and not project_owned_by_caller(request, project) and scope != "shared":
        # Fresh deploy: neither scope is provably the caller's.
        raise NerditError(
            403,
            "deploy.git_token_scope",
            "A service-scoped token_ref requires a service or secret scope you own.",
            hint=(
                f"Set it first with `nerdit secrets set {name} {key}=...` — that reserves "
                f"the name for your token — or use `${{secrets.shared.{key}}}`."
            ),
        )
    token = await resolve_git_token(
        request, mgr, name, token_ref, repo_url=repo_url, include_service=owned, project=project
    )
    if token is not None:
        return token
    raise NerditError(
        422,
        "deploy.git_token_unresolved",
        f"token_ref '{token_ref}' does not resolve to a stored secret.",
        hint=f"Set it first with `nerdit secrets set {name} {key}=...`.",
    )


def validate_git_request(
    request: Request,
    repo_url: str,
    *,
    ref: str | None,
    subdir: str | None,
    token_ref: str | None,
) -> str | None:
    """Gate and syntax-check a git source before anything is recorded or resolved.

    Shared by `POST /deploy/git` and `apply_project`: `[git].enabled`, then the
    very validators `clone_source` enforces (mapped 1:1), then the `token_ref`
    grammar. `${vars.…}` is rewritten first, so everything downstream (audit
    params, provenance, resolution) sees the stored `${secrets.…}` form.

    Returns:
        The rewritten `token_ref`, or `None`.

    Raises:
        NerditError: 403 `deploy.git_disabled`, the clone guards' own codes, or
            422 `deploy.git_token_ref_invalid`.
    """
    settings = request.app.state.settings
    if not settings.git.enabled:
        raise NerditError(
            403,
            "deploy.git_disabled",
            "Deploy-from-git is disabled on this daemon.",
            hint="Set [git].enabled = true to allow POST /deploy/git.",
        )
    validate_git_coordinates(settings, repo_url, ref, subdir)
    rewritten = rewrite_vars_ref(token_ref)
    if (
        rewritten is not None
        and rewritten != GITHUB_INSTALLATION_REF
        and not SECRET_REF_RE.match(rewritten)
    ):
        raise NerditError(
            422,
            "deploy.git_token_ref_invalid",
            "token_ref must be a ${secrets.KEY} or ${secrets.shared.KEY} reference, "
            "or the literal ${github.installation}.",
            hint=(
                "Store the token with `nerdit secrets set` and pass its reference name, "
                "or link this node and install the Nerdit GitHub App."
            ),
        )
    return rewritten


async def clone_for_request(
    request: Request,
    repo_url: str,
    *,
    ref: str | None,
    subdir: str | None,
    token: str | None,
) -> tuple[GitSourceInfo, Path]:
    """Shallow-clone into a fresh upload dir under the `[git]` limits.

    Returns:
        The clone info and the clone ROOT (the cleanup target; `info.context_dir`
        may be a subdir of it).

    Raises:
        NerditError: The clone failure, with the role-aware private-repo hint.
    """
    try:
        info, dest_dir = await clone_into_uploads(
            request.app.state.settings, repo_url, ref=ref, subdir=subdir, token=token
        )
    except GitSourceError as exc:
        raise NerditError(
            exc.status_code,
            exc.code,
            exc.message,
            hint=private_repo_hint(
                exc, role=current_principal(request).role, had_token=token is not None
            ),
        ) from exc
    return info, dest_dir


@router.post("/deploy/git", status_code=201, operation_id="deploy_git_app")
async def deploy_git(
    request: Request,
    body: GitDeployRequest,
    dry_run: bool = Query(
        False, description="Validate + return a plan diff without writing anything."
    ),
):
    """Build + register an app cloned from a git repository (P11.5 / Part A).

    The JSON sibling of `POST /deploy`: a shallow `git clone` replaces the
    ZIP upload, and the same `_finalize_deploy` tail runs. Agent-grade from the
    first commit — role-gated, idempotent (the `Idempotency-Key` is persisted
    on the row by `_finalize_deploy`), structured errors, audited
    (`deploy.git_create`). The load-bearing ordering: URL/ref/subdir/token_ref
    syntax guards run BEFORE any audit param is written (a rejected userinfo URL
    must leave no `repo_url` trace); the existing-row ownership gate runs
    BEFORE the token resolves or the clone starts (an unauthorized redeploy must
    not consume clone timeout/bytes or a private-repo credential).
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    if dry_run:
        request.state.audit_action = "deploy.git_plan"
    # Syntax guards BEFORE audit_params: a rejected URL/ref/subdir/token_ref must
    # leave no trace in the audit store (the middleware records params on
    # failures too).
    token_ref = validate_git_request(
        request, body.repo_url, ref=body.ref, subdir=body.subdir, token_ref=body.token_ref
    )

    request.state.audit_params = audit_params(
        {
            "repo_url": body.repo_url,
            "ref": body.ref,
            "subdir": body.subdir,
            "name": body.name,
            "port": body.port,
            "gpus": body.gpus,
            "start": body.start,
            "env": body.env,
            "token_ref": token_ref,
        }
    )
    reject_reserved_name(body.name)
    # Attribute the audit row to the validated app name (body.name is already
    # DNS-label-pinned by pydantic); the path rules leave the target null.
    request.state.audit_target = body.name
    queries = request.app.state.queries

    existing = await queries.get_service_by_name(body.name)
    # A model row is not a redeploy target (fast path — rejects before clone).
    reject_non_service_row(existing, body.name)
    # Authorize before `_resolve_token_ref`, not merely pre-clone: an
    # unauthorized caller must not consume clone timeout/bytes or resolve a
    # credential, and a shared-scope hit there writes a
    # `secret.shared_referenced` audit row. Scope first (widened by the row's
    # project, D-P40-7): one scope 403 with or without a row.
    require_service_scope(request, body.name, project=existing.project if existing else None)
    if existing is not None:
        require_owner_or_admin(request, existing)
    # A name another token reserved by setting its secrets is refused before
    # the clone; the row transaction re-checks it.
    if existing is None:
        await reject_foreign_claim(request, body.name)

    # Resolve the private-repo token server-side. It exists only as a local from
    # here on — never in the body, audit params, config, argv, logs, or response.
    # `service_owned` is true only for a redeploy that just passed the owner
    # gate above; a fresh deploy may read the per-service scope only through a
    # claim it owns (D-P39-4), never a leftover file of a reused name.
    token = await _resolve_token_ref(
        request,
        body.name,
        token_ref,
        service_owned=existing is not None,
        repo_url=body.repo_url,
        project_id=existing.project_id if existing is not None else None,
    )

    info, dest_dir = await clone_for_request(
        request, body.repo_url, ref=body.ref, subdir=body.subdir, token=token
    )

    # `token_ref` persists the `${…}` reference NAME, never the token, so an
    # unattended redeploy of a private repo can re-resolve the credential
    # server-side. A reference name is not a credential; the audit params
    # already carry it.
    source_meta = git_source_meta(
        info, body.repo_url, subdir=body.subdir, token_ref=token_ref or None
    )

    # `info.context_dir` may be a subdir of `dest_dir` — pass the clone ROOT
    # so _finalize_deploy stamps `build_context_root` (the controller cleans up
    # the whole clone, not just the built subdir) and owns root removal on failure.
    result = await _finalize_deploy(
        request,
        info.context_dir,
        name=body.name,
        port=body.port,
        gpus=body.gpus,
        start=body.start,
        health=body.health,
        env=body.env,
        build_settings=body.build_settings.model_dump(exclude_unset=True)
        if body.build_settings
        else None,
        vendor=body.vendor,
        source_meta=source_meta,
        context_root=dest_dir,
        dry_run=dry_run,
    )
    if dry_run:
        return JSONResponse(status_code=200, content=result)
    return result


@router.post("/deploy/{name}/rollback", status_code=200, operation_id="rollback_deploy")
async def rollback(request: Request, name: str):
    """Roll a service back to its previous image on the same stable endpoint.

    Swaps `image` ⇄ `previous_image` (the prior tag is already built, so the
    controller skips the build and just swaps the container) and resets the row
    to `restarting`. Agent-grade: role-gated, idempotent, audited
    (`deploy.rollback`).
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"name": name})
    # Scope runs before the in-progress 409 guards below, so an out-of-scope
    # caller gets an auth error rather than a race verdict. The row is read
    # first only so its project can widen the scope (D-P40-7); the 403 is
    # identical with or without a row.
    queries = request.app.state.queries
    existing = await queries.get_service_by_name(name)
    require_service_scope(request, name, project=existing.project if existing else None)
    if existing is None:
        raise NerditError(404, "not_found", f"No service '{name}'.")
    # Authorize against the row before mutating it (owner or admin only).
    require_owner_or_admin(request, existing)

    # A rollback racing a live [deploy].release would
    # abandon the running migration unobserved: the blind config write lowers
    # build_version, so the release's own completion handlers CAS-miss and its
    # real outcome (including a partially-applied migration) never surfaces —
    # and popping release_pending below would erase the crash forensics for a
    # generation whose migration is still executing. Refuse, same code as the
    # DELETE route (service_purge.py): the wait is bounded by
    # [services].release_timeout_s. No ?force hatch here (unlike DELETE):
    # killing a live migration from a rollback needs kill-outcome semantics
    # the settle handlers do not have. A WEDGED generation (stale
    # marker, daemon died mid-release) has no registered run slot, so the
    # documented stale-marker recovery path below is untouched.
    controller = getattr(request.app.state, "service_controller", None)
    if controller is not None and controller.has_active_run(existing.id):
        raise _run_in_progress_error(name, "roll back")
    # Same reasoning one layer up: a rollback landing mid-cutover
    # would lower `build_version` under a green container that is still being
    # probed, so the cutover's own CAS-guarded settle would miss and the
    # rolled-back image could be shadowed by a green that then gets promoted.
    # The wait is bounded by [services].cutover_verify_timeout_s.
    if controller is not None and controller.has_active_cutover(existing.id):
        raise _cutover_in_progress_error(name, "roll back")

    cfg = parse_job_config(existing)
    previous = cfg.get("previous_image")
    if not previous:
        raise NerditError(
            409,
            "deploy.no_previous_version",
            f"Service '{name}' has no previous version to roll back to.",
            hint="A rollback target only exists after at least one redeploy.",
        )

    current = cfg.get("image")
    suffix = str(previous).rsplit(":", 1)[-1]
    # Image swap ONLY: everything else in the blob — env, vendor, build fields
    # and the [ai.*] spec — is carried forward untouched by the dict copy.
    new_cfg = dict(cfg)
    new_cfg["image"] = previous
    new_cfg["previous_image"] = current  # allow rolling forward again
    # Lower `build_version` to the rolled-back tag but leave `max_version`
    # untouched (the high-water mark) so the next redeploy still allocates a
    # fresh, higher tag instead of colliding with an already-built image.
    new_cfg["build_version"] = int(suffix) if suffix.isdigit() else cfg.get("build_version", 1)
    # A rollback bypasses the builder entirely (the target image is
    # already built), so it is the one path that would otherwise inherit a
    # `release_pending` marker left by a wedged generation — and that marker
    # would make the controller settle the rolled-back generation
    # `release_failed` on the next daemon start. A rollback runs no release,
    # so it clears the gate. The `release` command itself is carried
    # forward untouched: it belongs to the app config, not to a generation. The
    # active-run gate above guarantees no release is executing when this pop runs.
    new_cfg.pop("release_pending", None)
    # Same reasoning for the cutover marker, with one extra edge
    # that makes it load-bearing rather than tidy: a rollback LOWERS
    # `build_version` back onto an already-used number, so a marker left by
    # that generation's verify would match again on the very next tick and
    # settle the rolled-back generation `cutover_failed` — reverting the
    # image the operator just chose. The active-cutover 409 above guarantees no
    # verify is running when this pop lands.
    new_cfg.pop("cutover_pending", None)
    # A rollback is a new deploy generation on the (already-built)
    # previous image: seed the phase object at the lowered version. `_needs_build`
    # is False, so the controller advances queued → launching → healthy directly.
    _stamp_queued(new_cfg, version=int(new_cfg["build_version"]), action="rollback")

    # Guarded like the redeploy write, and for the same reason: this blob is a
    # whole-row snapshot taken before the two 409 gates above, and it CARRIES
    # `max_version` from that read. An unguarded write therefore regresses the
    # allocator a concurrent redeploy just bumped — the next deploy then hands
    # out an image tag that is already built, which is exactly the collision the
    # redeploy CAS exists to prevent. A rollback never MOVES `max_version`, so
    # the predicate only refuses when a deploy (or a freshly armed cutover)
    # landed in between, which is the right refusal.
    committed = await queries.update_service_config_guarded(
        existing.id,
        json.dumps(new_cfg),
        expect_max_version=int(cfg.get("max_version", cfg.get("build_version", 0)) or 0),
        expect_cutover_pending=cfg.get("cutover_pending") is not None,
        status=JobStatus.restarting,
        desired_state="running",
    )
    if not committed:
        raise NerditError(
            409,
            "deploy.concurrent_redeploy",
            f"Another deploy of '{name}' committed while this rollback was being prepared.",
            hint="Wait for the in-flight deploy or cutover to settle, then roll back again.",
        )
    job = await queries.get_service_by_name(name) or existing
    return await service_view(request, job, hosted=await load_hosted_context(request))


@router.post("/deploy/{name}/redeploy", status_code=201, operation_id="redeploy_app")
async def redeploy(
    request: Request,
    name: str,
    dry_run: bool = Query(
        False, description="Validate + return a plan diff without writing anything."
    ),
):
    """Reclone recorded git source and rebuild without accepting new coordinates.

    The recorded ref is a branch/tag, not a pinned SHA. Require owner/admin before
    cloning; use idempotency, structured errors and deploy.redeploy auditing. Dry-run
    still validates and clones, but writes nothing and returns a 200 plan audited
    as deploy.redeploy_plan.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"name": name})
    if dry_run:
        # Stamped before any guard (the /deploy precedent): a dry run that
        # 409s is still audited as the plan request it was, never as a real
        # redeploy.
        request.state.audit_action = "deploy.redeploy_plan"
    queries = request.app.state.queries
    settings = request.app.state.settings

    existing = await queries.get_service_by_name(name)
    # Scope first, widened by the row's project (D-P40-7): ahead of the 404
    # (no existence oracle), of the in-progress 409 guards (auth errors win)
    # and of `redeploy_from_source`, which resolves the recorded token_ref and
    # clones. GitWatch calls that primitive in-process under an admin system
    # principal, so scope is inert there by design.
    require_service_scope(request, name, project=existing.project if existing else None)
    if existing is None:
        raise NerditError(404, "not_found", f"No service '{name}'.")
    # A model/database row is not a redeploy target.
    reject_non_service_row(existing, name)
    # Authorize against the row BEFORE anything that costs clone timeout/bytes
    # or resolves a private-repo credential.
    require_owner_or_admin(request, existing)

    if not settings.git.enabled:
        raise NerditError(
            403,
            "deploy.git_disabled",
            "Deploy-from-git is disabled on this daemon.",
            hint="Set [git].enabled = true to allow redeploy from a git source.",
        )

    controller = getattr(request.app.state, "service_controller", None)
    if controller is not None:
        # Same two races the rollback route refuses, for the same reason: both
        # a live release and a live cutover own the generation this redeploy
        # would overwrite.
        if controller.has_active_run(existing.id):
            raise _run_in_progress_error(name, "redeploy")
        if controller.has_active_cutover(existing.id):
            raise _cutover_in_progress_error(name, "redeploy")

    # The one source-provenance refusal, shared with the GitWatch poller: it
    # raises the 409 `deploy.no_source` family (ZIP row / workspace row /
    # URL-less git row). The return value is discarded — `redeploy_from_source`
    # re-reads the provenance itself; this call is purely the gate, kept here so
    # the route still refuses BEFORE the primitive resolves a credential.
    _read_git_source(existing, name)

    result = await redeploy_from_source(
        request_or_none=request,
        queries=queries,
        settings=settings,
        secrets=getattr(request.app.state, "secret_manager", None),
        job=existing,
        principal=current_principal(request).name,
        dry_run=dry_run,
    )
    if dry_run:
        return JSONResponse(status_code=200, content=result)
    return result
