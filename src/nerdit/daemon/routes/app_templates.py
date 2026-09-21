"""Read curated templates and deploy them through the shared git pipeline.

Template coordinates feed clone_source and _finalize_deploy. Deployment is
submitter/admin-gated, idempotent and audited with template ID/name, never secret
values. Secrets go through the one write path (`secret_scope.set_secret_values`)
before the clone: on a fresh name that mints the caller's claim, so a principal
that would lose the create race loses the claim insert first, before any clone
or row (P39 D-P39-5). Mounted under /api only with the Store tag.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from nerdit.config.app_templates import app_templates_by_id, load_app_templates
from nerdit.core.gitsource import GitSourceError, clone_source, git_source_meta
from nerdit.core.secrets import validate_secret_items
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_owner_or_admin, require_role, require_service_scope
from nerdit.daemon.deploy_pipeline import _finalize_deploy, reject_non_service_row
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.services import reject_reserved_name
from nerdit.daemon.secret_scope import reject_foreign_claim, secret_call, set_secret_values
from nerdit.db.models import AppTemplate, TemplateDeployRequest, TokenRole
from nerdit.utils.ids import generate_id

router = APIRouter()


def _git_disabled_if_off(request: Request) -> None:
    """Reject a template deploy when deploy-from-git is disabled (same envelope)."""
    if not request.app.state.settings.git.enabled:
        raise NerditError(
            403,
            "deploy.git_disabled",
            "Deploy-from-git is disabled on this daemon.",
            hint="Set [git].enabled = true to allow the template store.",
        )


@router.get("/app-templates", response_model=list[AppTemplate], operation_id="list_app_templates")
async def list_app_templates() -> list[AppTemplate]:
    """Return the embedded app template catalog (any authenticated principal)."""
    return load_app_templates()


@router.get(
    "/app-templates/{template_id}",
    response_model=AppTemplate,
    operation_id="get_app_template",
)
async def get_app_template(template_id: str) -> AppTemplate:
    """Return one app template, or a structured 404."""
    template = app_templates_by_id().get(template_id)
    if template is None:
        raise NerditError(404, "not_found", f"No app template '{template_id}'.")
    return template


@router.post(
    "/app-templates/{template_id}/deploy",
    status_code=201,
    operation_id="deploy_app_template",
)
async def deploy_app_template(
    request: Request,
    template_id: str,
    body: TemplateDeployRequest,
    dry_run: bool = Query(False, description="Preview without building or writing secrets"),
):
    """Deploy a public-repository template through the shared git pipeline.

    Precedence is request, template defaults, repo TOML, then buildpack defaults.
    Secrets are written before the clone and the row (claim-then-deploy): the
    claim reserves a fresh name for the caller and the row insert consumes it,
    so a failed deploy leaves the caller's own scope, never a foreign one.
    _finalize_deploy cleans failed clone contexts. Redeployment merges existing
    secrets.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    _git_disabled_if_off(request)
    if dry_run:
        request.state.audit_action = "deploy.template_plan"

    template = app_templates_by_id().get(template_id)
    if template is None:
        raise NerditError(404, "not_found", f"No app template '{template_id}'.")

    reject_reserved_name(body.name)
    # (P25 D-P25-3 leg b) `body.name` is the resolved service name (never
    # template-derived), so the scope check runs before the clone and before any
    # SecretManager write.
    # (D-P40-7) Label-only: scope is judged before any lookup, so there is no row
    # whose project could widen it (the ceiling is named at `rollback`).
    require_service_scope(request, body.name)
    # Re-point the audit target from the path-derived template id to the deployed
    # service name (the template id stays in the event params) so store-created
    # projects are visible to GET /audit?target=<app>.
    request.state.audit_target = body.name

    # env_schema enforcement: every required input must be present in the channel
    # its `secret` flag dictates (secret → body.secrets, else body.env). A
    # required non-secret input sent as JSON `null` counts as missing —
    # _finalize_deploy's fresh-deploy null-filter drops it, so a presence-only
    # gate would launch the container without it. Secrets stay presence-only
    # (body.secrets is dict[str, str] — no null channel).
    env = body.env or {}
    secrets = body.secrets or {}
    missing = [
        var.name
        for var in template.env_schema
        if var.required and (var.name not in secrets if var.secret else env.get(var.name) is None)
    ]
    if missing:
        raise NerditError(
            422,
            "template.missing_env",
            f"Missing required inputs for template '{template_id}': {', '.join(missing)}.",
            hint="Provide them via --env (or --secret for secret inputs).",
        )

    request.state.audit_params = audit_params(
        {
            "template_id": template_id,
            "name": body.name,
            "port": body.port,
            "gpus": body.gpus,
            "start": body.start,
            "env": body.env,
            "secrets": body.secrets,
        }
    )

    queries = request.app.state.queries
    existing = await queries.get_service_by_name(body.name)
    # A model row is not a redeploy target (fast path — rejects before clone).
    reject_non_service_row(existing, body.name)
    # Authorize a redeploy against the existing row PRE-CLONE — an unauthorized
    # attempt must be rejected before it consumes clone timeout/bytes.
    if existing is not None:
        require_owner_or_admin(request, existing)
    else:
        # (P39) A name another token reserved by setting its secrets is
        # refused before the clone; the row transaction re-checks it.
        await reject_foreign_claim(request, body.name)

    # Secrets first, deploy second (D-P39-5). The claim insert is the atomic
    # authorization point for a fresh name's secret scope: a principal that
    # would lose the create race loses the claim's PRIMARY KEY first, before any
    # clone, and the row insert (reserve_service_for_token) later consumes that
    # claim; an existing row is authorized as its owner by the same helper. A
    # bad key or value fails here (422) with no clone and no row. O1 still
    # holds: the first launch waits for the off-tick build. A dry run only
    # validates — no claim, no file.
    if secrets:
        if dry_run:
            secret_call(validate_secret_items, secrets)
        else:
            await set_secret_values(request, body.name, secrets)

    settings = request.app.state.settings
    dest_dir = Path(settings.daemon.upload_dir).expanduser() / generate_id()
    try:
        info = await clone_source(
            template.repo_url,
            ref=template.ref,
            subdir=template.subdir,
            dest_dir=dest_dir,
            token=None,
            timeout_s=settings.git.clone_timeout_s,
            max_bytes=settings.git.max_clone_bytes,
            allowed_hosts=settings.git.allowed_hosts,
        )
    except GitSourceError as exc:
        raise NerditError(exc.status_code, exc.code, exc.message, hint=exc.hint) from exc

    # Precedence merge: request field wins, else the template default. Combined
    # with _finalize_deploy's internal chain this yields the locked
    # request > template defaults > repo nerdit.toml > buildpack default.
    defaults = template.deploy_defaults
    eff_port = body.port if body.port is not None else defaults.port
    eff_gpus = body.gpus if body.gpus is not None else defaults.gpus
    eff_start = body.start if body.start is not None else defaults.start
    eff_health = body.health if body.health is not None else defaults.health

    source_meta = git_source_meta(
        info, template.repo_url, subdir=template.subdir, template_id=template_id
    )

    # _finalize_deploy owns rmtree of the clone root (context_root=dest_dir) on
    # any failure.
    resp = await _finalize_deploy(
        request,
        info.context_dir,
        name=body.name,
        port=eff_port,
        gpus=eff_gpus,
        start=eff_start,
        build_settings=(
            body.build_settings.model_dump(exclude_unset=True)
            if body.build_settings is not None
            else None
        ),
        health=eff_health,
        env=body.env,
        vendor=body.vendor,
        source_meta=source_meta,
        context_root=dest_dir,
        dry_run=dry_run,
    )
    if dry_run:
        return JSONResponse(status_code=200, content=resp)
    return resp
