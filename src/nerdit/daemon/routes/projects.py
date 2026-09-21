"""The project noun as an API (P40b): create, list, read and delete projects.

A project is the grouping every ``kind=service`` row points at (P40a). Its row
outlives its services and reserves the name for the owner token, so the three
D-P40-5 judgments compose with the P39 claim: a foreign project refuses a
deploy (409 ``project.owned``), a foreign project refuses a secret claim (the
row 403), a foreign claim refuses ``create_project`` (409
``service.name_claimed``). Wire identity stays the label (D-P40-6); no request,
response or audit field is ever named ``environment`` (D-P40-11). Every message
here names project and service names only. Mounted under /api only.

P40c adds the variables of a project: one store, one flag (D-P40-1). Every
value lives in a scope's encrypted file; the only value any body here carries
is a PLAIN one, to the project's owner or an admin (D-P40-15). No route here
ever addresses the machine scope (`_shared`).

P40d adds `apply_project`, an orchestrator over the ordinary deploy tail
(D-P40-12): every refusal lands before the first build context, a dry run
writes nothing, and a service's label is composed, never parsed back.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nerdit.config.project import (
    PROJECT_CONFIG_NAME,
    DeclarationError,
    parse_project_declaration,
)
from nerdit.core.gitsource import git_source_meta
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.project_identity import (
    DEFAULT_SERVICE,
    PRODUCTION,
    PROJECT_NAME_RE,
    SERVICE_NAME_RE,
    service_label,
)
from nerdit.core.secrets import SecretDecryptError, project_storage_name
from nerdit.core.variables import load_scoped, plain_keys
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    current_principal,
    may_manage_job,
    require_owner_or_admin,
    require_project_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.deploy_pipeline import (
    DeclaredService,
    _finalize_deploy,
    _read_project_toml,
    reject_foreign_label,
    reject_non_service_row,
)
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.deploy import (
    _resolve_token_ref,
    clone_for_request,
    validate_git_request,
)
from nerdit.daemon.routes.services import reject_reserved_name
from nerdit.daemon.routes.workspaces import snapshot_own_workspace
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.daemon.secret_scope import (
    authorize_secret_scope,
    claim_owned_by_caller,
    create_project_for_caller,
    project_owned_by_caller,
    project_owned_error,
    reject_foreign_claim,
    require_service_in_project,
    secret_call,
    secret_manager,
    service_scope_readable,
    set_project_values,
    set_service_values,
    variable_write_lock,
)
from nerdit.daemon.service_purge import _parse_purge, delete_service
from nerdit.daemon.uploads import extract_upload, read_upload_member
from nerdit.daemon.views.hosted import load_hosted_context
from nerdit.daemon.views.service import _service_response
from nerdit.db.models import Job, JobKind, JobStatus, Project, ServiceResponse, TokenRole
from nerdit.db.queries import ProjectExists
from nerdit.utils.ids import generate_id

logger = logging.getLogger(__name__)

router = APIRouter()


class ProjectCreateRequest(StrictRequestModel):
    """Body for `POST /projects`: the name only (production is implicit, D-P40-11)."""

    name: str = Field(description="Project name: lowercase DNS label, ≤ 40 chars, no '--'")


class ProjectResourceView(BaseModel):
    """One `[ai.*]` / `[db.*]` binding a project's service references, with readiness.

    Readiness is judged on the inventory row only (a served model, a managed
    database) and never on a secret: `null` means "cannot be judged without
    the caller's secrets" (an external API or database), which keeps this view
    readable by a scoped non-owner (D-P40-15).
    """

    service: str = Field(description="Label of the service that declares the binding")
    type: str = Field(description="'ai' | 'db'")
    binding: str = Field(description="Binding name, e.g. 'default'")
    provider: str | None = Field(default=None, description="The binding's provider")
    target: str | None = Field(
        default=None, description="Model ref, managed database name or base URL"
    )
    ready: bool | None = Field(
        default=None,
        description="Whether the referenced row is up; null when readiness needs a secret",
    )


class ProjectSummary(BaseModel):
    """A project with its services and every advertised address (list + create)."""

    id: str = Field(description="`prj_` + 16 base32 chars")
    name: str = Field(description="The project name")
    services: list[ServiceResponse] = Field(
        default_factory=list, description="The project's service rows, by service name"
    )
    addresses: list[PublicUrlEntry] = Field(
        default_factory=list, description="Every advertised URL of every service"
    )


class ProjectHome(BaseModel):
    """The machine a project lives on — always this node in phase 1 (plan §4)."""

    hostname: str | None = Field(default=None, description="This daemon's hostname")
    node_id: str | None = Field(default=None, description="This node's link id, if linked")


class VariableName(BaseModel):
    """One variable by name: where it lives and whether it is plain. Never a value."""

    key: str = Field(description="Variable key (an env-var name)")
    scope: str = Field(description="'project', or 'production/<service>' for a service scope")
    plain: bool = Field(description="True = readable by the owner; False = write-only secret")


class VariableView(VariableName):
    """A listed variable; `value` is set only for a plain key (D-P40-15)."""

    value: str | None = Field(
        default=None, description="The value when `plain`; always null for a secret"
    )


class VariableListResponse(BaseModel):
    """`GET /projects/{project}/variables`: one scope's variables."""

    project: str
    scope: str
    variables: list[VariableView]


class VariableResolveResponse(BaseModel):
    """`GET …/variables/resolve`: per key, the scope a launch would take it from."""

    project: str
    service: str
    variables: list[VariableName]


class VariableSetRequest(StrictRequestModel):
    """Body for `PUT …/variables`: a value map and the secret flag.

    `values` is opaque in every 422 echo (`errors._SECRET_INPUT_FIELDS`), like
    `SecretSetRequest.values`: the keys are caller-chosen, so no name rule
    could mask them one by one.
    """

    values: dict[str, str] = Field(..., repr=False, description="KEY -> value pairs to merge")
    secret: bool = Field(
        default=True,
        description="True (default, fail-safe D-P40-16) = write-only; "
        "False = plain, readable by the owner",
    )


class VariableSetResponse(BaseModel):
    """Result of a variable set: the scope's key names, never a value."""

    project: str
    scope: str
    keys: list[str]
    plain: bool


class ProjectResponse(ProjectSummary):
    """`GET /projects/{project}`: the summary plus referenced resources and home."""

    resources: list[ProjectResourceView] = Field(default_factory=list)
    home: ProjectHome
    variables: list[VariableName] | None = Field(
        default=None,
        description="Variable names per scope, no values; null unless the caller owns "
        "the project or is an admin (D-P40-15)",
    )


class ProjectListPage(BaseModel):
    """Cursor-paginated page of projects for the bounded `GET /projects` read."""

    items: list[ProjectSummary]
    next_cursor: str | None = Field(
        default=None, description="Opaque cursor for the next page; null when exhausted"
    )


class ProjectDeletedResponse(BaseModel):
    """`DELETE /projects/{project}` on success."""

    name: str
    deleted: list[str] = Field(description="Labels of the service rows removed, in order")


class ApplyServiceResult(BaseModel):
    """One declared service's outcome in an apply, in declaration order."""

    label: str = Field(description="The service's wire name (D-P40-6)")
    service: str = Field(description="The service name inside the project")
    action: Literal["fresh", "redeploy"]
    status: str | None = Field(
        default=None, description="The row's status after the write; null on a dry run"
    )
    build_version: int | None = Field(
        default=None, description="The generation deployed by this apply; null on a dry run"
    )
    plan: dict | None = Field(
        default=None, description="The deploy plan diff; set on a dry run only"
    )


class ApplyResponse(BaseModel):
    """`POST /projects/{project}/apply`: what was planned, applied, or is still missing."""

    project: str
    status: Literal["applied", "planned", "waiting_for_variables"]
    dry_run: bool
    services: list[ApplyServiceResult] = Field(default_factory=list)
    public_urls: list[PublicUrlEntry] = Field(
        default_factory=list, description="Every advertised URL of the applied services"
    )
    missing: list[str] = Field(
        default_factory=list,
        description="`[vars] required` names no scope provides yet (names only, never values)",
    )
    hint: str | None = None


def _not_found(name: str) -> NerditError:
    return NerditError(
        404,
        "not_found",
        f"No project '{name}'.",
        hint="List projects with `nerdit projects list` to find a valid name.",
    )


def _validate_name(name: str) -> None:
    """422 `project.invalid_name` on a name outside the D-P40-14 grammar."""
    if not PROJECT_NAME_RE.fullmatch(name):
        raise NerditError(
            422,
            "project.invalid_name",
            f"Invalid project name '{name}'.",
            hint="Project names are lowercase DNS labels of at most 40 characters "
            "(letters, digits, '-') and never contain '--'.",
        )


async def _service_views(request: Request, jobs: list[Job]) -> list[ServiceResponse]:
    """Project the rows exactly as `GET /services` does, one hosted snapshot per call."""
    queries = request.app.state.queries
    hosted = await load_hosted_context(request)
    views: list[ServiceResponse] = []
    for job in jobs:
        endpoint = await queries.get_service_endpoint(job.service_name or "")
        gpu_ids = await queries.get_job_gpus(job.id)
        views.append(_service_response(request, job, gpu_ids, endpoint, hosted=hosted))
    return views


def _addresses(services: list[ServiceResponse]) -> list[PublicUrlEntry]:
    return [entry for svc in services if svc.endpoint for entry in svc.endpoint.public_urls]


def _is_up(row: Job | None) -> bool:
    return row is not None and row.status in (JobStatus.running, JobStatus.degraded)


async def _resources(request: Request, jobs: list[Job]) -> list[ProjectResourceView]:
    """Every binding the project's services declare, judged on inventory rows only.

    Readiness is judged only for a target inside the caller's scope: a scoped
    reader of one project must not learn whether a foreign model or database
    is up (`null`, like a secret-backed target).
    """
    queries = request.app.state.queries
    principal = current_principal(request)
    out: list[ProjectResourceView] = []
    for job in jobs:
        cfg = parse_job_config(job)
        label = job.service_name or job.id
        raw_ai = cfg.get("ai")
        ai: dict = raw_ai if isinstance(raw_ai, dict) else {}
        for name in sorted(ai):
            spec = ai[name] if isinstance(ai[name], dict) else {}
            provider = spec.get("provider")
            ready: bool | None = None
            target = spec.get("model")
            if provider == "ollama" and isinstance(target, str) and principal.in_scope(target):
                ready = _is_up(await queries.get_model_by_ref(target))
            elif provider == "api":
                target = spec.get("base_url")
            out.append(
                ProjectResourceView(
                    service=label,
                    type="ai",
                    binding=name,
                    provider=provider,
                    target=target if isinstance(target, str) else None,
                    ready=ready,
                )
            )
        raw_db = cfg.get("db")
        db: dict = raw_db if isinstance(raw_db, dict) else {}
        for name in sorted(db):
            spec = db[name] if isinstance(db[name], dict) else {}
            provider = spec.get("provider")
            ready = None
            target = spec.get("database")
            if provider == "managed" and isinstance(target, str) and principal.in_scope(target):
                row = await queries.get_service_by_name(target)
                ready = row is not None and row.kind is JobKind.database and _is_up(row)
            out.append(
                ProjectResourceView(
                    service=label,
                    type="db",
                    binding=name,
                    provider=provider,
                    target=target if isinstance(target, str) else None,
                    ready=ready,
                )
            )
    return out


@router.get("/projects", response_model=ProjectListPage, operation_id="list_projects")
async def list_projects(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None, description="Opaque cursor from a previous page"),
) -> ProjectListPage:
    """Cursor-paginated list of projects, filtered to the caller's scope (D-P40-15).

    Read is open to any authenticated principal (readonly and up); a scoped
    token sees only the projects its scope names.
    """
    principal = current_principal(request)
    names = None if principal.scope_services is None else principal.scope_services
    queries = request.app.state.queries
    try:
        projects, next_cursor = await queries.page_projects(names=names, cursor=cursor, limit=limit)
    except ValueError as exc:
        raise NerditError(400, "bad_request", str(exc)) from exc
    items: list[ProjectSummary] = []
    for project in projects:
        # ponytail: one row query per project on the page (≤ 200, in-process
        # SQLite); batch by `project_id IN (...)` if the grid ever feels it.
        services = await _service_views(request, await queries.list_project_services(project.id))
        items.append(
            ProjectSummary(
                id=project.id, name=project.name, services=services, addresses=_addresses(services)
            )
        )
    return ProjectListPage(items=items, next_cursor=next_cursor)


@router.post(
    "/projects", response_model=ProjectSummary, status_code=201, operation_id="create_project"
)
async def create_project(request: Request, body: ProjectCreateRequest) -> ProjectSummary:
    """Create an empty project, reserving its name for the caller's token.

    Grammar (D-P40-14) and reserved names are checked first, then the scope —
    all before any lookup, so a stranger learns nothing about existing rows.
    `submitter`/`admin`; audited (`project.create`, the name stamped as target).
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"project": body.name})
    _validate_name(body.name)
    reject_reserved_name(body.name)
    require_service_scope(request, body.name)
    request.state.audit_target = body.name
    try:
        project = await create_project_for_caller(request, body.name)
    except ProjectExists as exc:
        raise NerditError(
            409,
            "project.exists",
            f"A project named '{body.name}' already exists.",
            hint="Choose a different name, or `nerdit projects show` the existing one.",
        ) from exc
    return ProjectSummary(id=project.id, name=project.name)


async def _lookup(request: Request, name: str) -> Project:
    """Scope first (the 403 for an out-of-scope name), then the row or a 404."""
    require_service_scope(request, name)
    project = await request.app.state.queries.get_project_by_name(name)
    if project is None:
        raise _not_found(name)
    return project


@router.get("/projects/{project}", response_model=ProjectResponse, operation_id="get_project")
async def get_project(request: Request, project: str) -> ProjectResponse | JSONResponse:
    """A project with its services, referenced resources, addresses and home.

    Open to any principal whose scope includes the project; an out-of-scope
    name is the `require_service_scope` 403 before any lookup (D-P40-15).
    """
    row = await _lookup(request, project)
    state = request.app.state
    jobs = await state.queries.list_project_services(row.id)
    services = await _service_views(request, jobs)
    manager = getattr(state, "link_manager", None)
    view = ProjectResponse(
        id=row.id,
        name=row.name,
        services=services,
        addresses=_addresses(services),
        resources=await _resources(request, jobs),
        home=ProjectHome(
            hostname=getattr(state, "hostname", None),
            node_id=manager.status().node_id if manager is not None else None,
        ),
        variables=await _variable_names(request, row, jobs),
    )
    if view.variables is None:
        # D-P40-15: the section is OMITTED for a non-owner, not nulled, and
        # `response_model_exclude_none` would also strip every null a
        # `ServiceResponse` carries on purpose.
        return JSONResponse(view.model_dump(mode="json", exclude={"variables"}))
    return view


def _remove_project_file(request: Request, project_id: str) -> None:
    mgr = getattr(request.app.state, "secret_manager", None)
    if mgr is None:
        return
    try:
        mgr.delete(project_storage_name(project_id))
    except OSError:
        # The row is already gone; an unreachable ciphertext must not turn the
        # finished delete into a 500. `nerdit gc` territory, not the caller's.
        logger.warning("Could not remove the variable file of project %s", project_id)


def _scope_name(service: str | None) -> str:
    # The audit/response scope string (D-P40-11): never a field named `environment`.
    return "project" if service is None else f"{PRODUCTION}/{service}"


async def _variable_names(
    request: Request, row: Project, jobs: list[Job]
) -> list[VariableName] | None:
    """Every scope's key names for the owner or an admin; `None` for anyone else.

    Service scopes are listed only for rows the caller may manage (the
    `/secrets` names rule). An undecryptable store omits the section rather
    than failing the whole project read.
    """
    mgr = getattr(request.app.state, "secret_manager", None)
    if mgr is None or not project_owned_by_caller(request, row):
        return None
    flags = await request.app.state.queries.list_variable_flags(row.id)
    plain = {(f.service, f.key) for f in flags if f.plain}
    # ponytail: a `web` scope set before the first deploy (claim, no row) is not
    # listed here; `list_variables?service=web` shows it. Add the claim read if
    # the dashboard needs it.
    scopes: list[tuple[str, str | None]] = [(project_storage_name(row.id), None)]
    scopes += [
        (job.service_name, job.service or DEFAULT_SERVICE)
        for job in jobs
        if job.service_name and may_manage_job(request, job)
    ]
    try:
        return [
            VariableName(key=key, scope=_scope_name(service), plain=(service or "", key) in plain)
            for storage, service in scopes
            for key in mgr.list_keys(storage)
        ]
    except SecretDecryptError:
        return None


def _check_new_name(name: str) -> None:
    _validate_name(name)
    reject_reserved_name(name)


def _label(project: str, service: str) -> str:
    """The service's label inside the project, or a 422 before any lookup."""
    if not SERVICE_NAME_RE.fullmatch(service):
        raise NerditError(
            422,
            "project.invalid_service",
            f"Invalid service name '{service}'.",
            hint="Service names are lowercase DNS labels of at most 20 characters, no '--'.",
        )
    # A legacy project literally named `shared` must never reach the secrets
    # surface's shared-scope branch through its `web` label.
    reject_reserved_name(project)
    try:
        return service_label(project, PRODUCTION, service)
    except ValueError as exc:
        raise NerditError(
            422,
            "project.label_too_long",
            f"Service '{service}' of project '{project}' has no valid label.",
            hint="The composed `<service>--<project>` label must be a DNS label of at most "
            "63 characters.",
        ) from exc


async def _owned(request: Request, name: str) -> Project:
    """Scope, then the row or a 404, then owner-or-admin (D-P40-15 reads and unset)."""
    row = await _lookup(request, name)
    require_project_owner_or_admin(request, row)
    return row


# Registered BEFORE `/variables/{key}` so the literal segment can never be
# taken for a key.
@router.get(
    "/projects/{project}/variables/resolve",
    response_model=VariableResolveResponse,
    operation_id="resolve_variables",
)
async def resolve_variables(
    request: Request,
    project: str,
    service: str = Query(DEFAULT_SERVICE, description="Service name inside the project"),
) -> VariableResolveResponse:
    """Per key, the scope a launch of `service` would take it from. Never a value.

    Owner or admin (D-P40-15). Winners come from the merged reader itself
    (`load_scoped`), so this cannot drift from what a launch does; the service
    scope joins only when the caller may read it.
    """
    label = _label(project, service)
    row = await _owned(request, project)
    await require_service_in_project(request, row, project, service, label)
    include_service = await service_scope_readable(request, label)
    _, winners = secret_call(
        lambda: load_scoped(secret_manager(request), label, row.id, include_service=include_service)
    )
    queries = request.app.state.queries
    plain = {
        "project": await plain_keys(queries, row.id),
        "service": await plain_keys(queries, row.id, service),
    }
    return VariableResolveResponse(
        project=project,
        service=service,
        variables=[
            VariableName(
                key=key,
                scope=_scope_name(None if winner == "project" else service),
                plain=key in plain[winner],
            )
            for key, winner in sorted(winners.items())
        ],
    )


@router.get(
    "/projects/{project}/variables",
    response_model=VariableListResponse,
    operation_id="list_variables",
)
async def list_variables(
    request: Request,
    project: str,
    service: str | None = Query(None, description="Service name; omit for the project scope"),
) -> VariableListResponse:
    """One scope's variables: key, scope, `plain`, and the value of a PLAIN key only.

    Owner or admin (D-P40-15; a NULL-owner project is admin-only); a scoped
    non-owner gets the row 403. A secret value is never returned.
    """
    label = _label(project, service) if service is not None else None
    row = await _owned(request, project)
    # One scope at a time, still through the merged reader (the only caller
    # of `SecretManager.load` outside the store): the other half is excluded.
    mgr = secret_manager(request)
    env: dict[str, str] = {}
    # Values and flags are ONE snapshot under the writers' lock: a secret->plain
    # flip landing between the two reads would pair the old secret with the new
    # plain flag and return it. ponytail: a list waits out a staging backup.
    async with variable_write_lock(request.app):
        if label is None or service is None:
            env, _ = secret_call(lambda: load_scoped(mgr, None, row.id))
        else:
            await require_service_in_project(request, row, project, service, label)
            if await service_scope_readable(request, label):
                env, _ = secret_call(lambda: load_scoped(mgr, label, None))
        plain = await plain_keys(request.app.state.queries, row.id, service)
    scope = _scope_name(service)
    return VariableListResponse(
        project=project,
        scope=scope,
        variables=[
            VariableView(
                key=key, scope=scope, plain=key in plain, value=env[key] if key in plain else None
            )
            for key in sorted(env)
        ],
    )


@router.put(
    "/projects/{project}/variables",
    response_model=VariableSetResponse,
    operation_id="set_variables",
)
async def set_variables(
    request: Request,
    project: str,
    body: VariableSetRequest,
    service: str | None = Query(None, description="Service name; omit for the project scope"),
) -> VariableSetResponse:
    """Set/merge variables in one scope; returns key names only.

    `secret` defaults to true (D-P40-16). A set on an absent project creates
    it for the caller; a service-scope set on a rowless label mints the P39
    claim. Setting a key flips its flag in that scope. Audited `variable.set`
    with names only.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    plain = not body.secret
    scope = _scope_name(service)
    # Hand-built, names only (the `secret.set` precedent): never a value, and
    # no member named `environment` (D-P40-11) or `secret` (both are masked).
    request.state.audit_params = audit_params(
        {
            "project": project,
            "scope": scope,
            "service": service,
            "keys": sorted(body.values),
            "plain": plain,
        }
    )
    if not body.values:
        raise NerditError(400, "variable.empty", "No variable values provided.")
    if service is None:
        keys = await set_project_values(
            request, project, body.values, plain, check_new_name=_check_new_name
        )
    else:
        _label(project, service)  # the 422s, before any lookup
        keys = await set_service_values(
            request, project, service, body.values, plain, check_new_name=_check_new_name
        )
    return VariableSetResponse(project=project, scope=scope, keys=keys, plain=plain)


@router.delete(
    "/projects/{project}/variables/{key}", status_code=200, operation_id="delete_variable"
)
async def delete_variable(
    request: Request,
    project: str,
    key: str,
    service: str | None = Query(None, description="Service name; omit for the project scope"),
) -> dict[str, str]:
    """Delete one key from a scope's file, and its flag row.

    Owner or admin. Never releases a P39 claim: deleting a whole service scope
    stays `DELETE /secrets/{label}`. Audited `variable.unset`.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    scope = _scope_name(service)
    request.state.audit_params = audit_params(
        {"project": project, "scope": scope, "service": service, "keys": [key]}
    )
    label = _label(project, service) if service is not None else None
    async with variable_write_lock(request.app):  # verdict and delete are one section
        row = await _owned(request, project)
        if label is None or service is None:
            storage = project_storage_name(row.id)
        else:
            await require_service_in_project(request, row, project, service, label)
            await authorize_secret_scope(request, label, write=True)
            storage = label
        existed = secret_call(secret_manager(request).delete_key, storage, key)
        # A stale flag row is harmless (every `/secrets` and variables write
        # re-flags its keys), so the file goes first and the flag follows.
        await request.app.state.queries.delete_variable_flag(row.id, service, key)
    if not existed:
        raise NerditError(404, "not_found", f"No variable '{key}' in scope '{scope}'.")
    return {"project": project, "scope": scope, "deleted": key}


@router.delete(
    "/projects/{project}",
    response_model=ProjectDeletedResponse,
    operation_id="delete_project",
)
async def delete_project(
    request: Request,
    project: str,
    purge: str = Query(
        "secrets",
        description="CSV of purge targets applied to every service: "
        "secrets,data,images,workspace (default: secrets)",
    ),
) -> ProjectDeletedResponse:
    """Delete a project: every service through the `DELETE /services` cascade, then the row.

    Owner or admin, judged on `projects.submitted_by_token`. Services go one
    by one with the same purge flags; the `projects` row goes last and only
    when every row went (D-P40-17). A refused service leaves the project in
    place with a 409 `project.delete_incomplete` naming `deleted` and
    `failed`; re-running is idempotent. Audited (`project.delete`).
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    purge_set = _parse_purge(purge)
    request.state.audit_params = audit_params({"project": project, "purge": sorted(purge_set)})
    row = await _lookup(request, project)
    require_project_owner_or_admin(request, row)
    queries = request.app.state.queries
    deleted: list[str] = []
    failed: list[dict[str, str]] = []
    try:
        for job in await queries.list_project_services(row.id):
            label = job.service_name or job.id
            try:
                # The per-service route body: reference guard, teardown, row
                # delete, purge — and its own owner check, so a service another
                # token owns inside this project is refused, not swept away.
                await delete_service(request, label, purge=purge, force=False)
            except NerditError as exc:
                failed.append({"name": label, "code": exc.code, "message": exc.message})
            else:
                deleted.append(label)
        if not failed:
            remaining = await queries.delete_project_checked(row.id)
            if remaining is None:
                raise _not_found(project)
            if not remaining:
                # (P40c / D-P40-17) The row is gone and its flag rows went by
                # FK; nothing can address the id-keyed file any more, so it
                # goes unconditionally (no purge flag gates it).
                # Under the writers' lock so a parked write cannot re-create it.
                async with variable_write_lock(request.app):
                    _remove_project_file(request, row.id)
            # A row landed between the sweep and the final delete: report it as
            # a refusal like any other so the caller re-runs rather than guesses.
            failed = [
                {"name": n, "code": "service.exists", "message": f"Service '{n}' still exists."}
                for n in remaining
            ]
    finally:
        # Written however the request ends (a nested delete's 500, the vanished
        # project's 404): the audit row names what actually went, and a nested
        # `delete_service`'s own params never stand in for it.
        request.state.audit_params = audit_params(
            {
                "project": project,
                "purge": sorted(purge_set),
                "deleted": deleted,
                "failed": [f["name"] for f in failed],
            }
        )
    if failed:
        raise NerditError(
            409,
            "project.delete_incomplete",
            f"Project '{project}' was not deleted: {len(failed)} service(s) remain.",
            hint="Fix the failures listed under `failed` and re-run the delete.",
            deleted=deleted,
            failed=failed,
        )
    return ProjectDeletedResponse(name=project, deleted=deleted)


# --- P40d: the declaration apply (D-P40-12) -----------------------------------


def _declaration_error(exc: DeclarationError) -> NerditError:
    return NerditError(
        422,
        exc.code,
        str(exc),
        hint=f"See the [project] / [services.<name>] / [vars] shape of {PROJECT_CONFIG_NAME}.",
    )


def _invalid_declaration(message: str) -> NerditError:
    return _declaration_error(DeclarationError(message))


async def _archive_declaration(request: Request, archive: UploadFile) -> dict:
    """Parse `nerdit.toml` straight out of the upload: no build context exists yet."""
    raw = await read_upload_member(
        archive,
        PROJECT_CONFIG_NAME,
        max_bytes=request.app.state.settings.daemon.max_upload_bytes,
    )
    if raw is None:
        raise _invalid_declaration(f"The archive has no {PROJECT_CONFIG_NAME} at its root.")
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        # Value-free: the parser's own message quotes the offending text.
        raise _invalid_declaration(f"{PROJECT_CONFIG_NAME} is not valid TOML.") from None


async def _git_declaration(
    request: Request, repo_url: str, ref: str | None, token: str | None
) -> dict:
    """Read `nerdit.toml` off a throwaway clone; for a git source the clone IS the read."""
    info, root = await clone_for_request(request, repo_url, ref=ref, subdir=None, token=token)
    try:
        data, error = _read_project_toml(info.context_dir)
    finally:
        await asyncio.to_thread(shutil.rmtree, root, ignore_errors=True)
    if error is not None:
        raise _invalid_declaration(f"{PROJECT_CONFIG_NAME} is not valid TOML.")
    if not data:
        raise _invalid_declaration(f"The repository has no {PROJECT_CONFIG_NAME} at its root.")
    return data


async def _judge_apply_project(request: Request, project: str) -> Project | None:
    """Scope before any lookup, then the owner gate: 409 `project.owned`, nothing else.

    The refusal carries no `missing` list and no key name: a non-owner must
    never learn which variables a foreign project lacks (the P39 oracle).
    Ownership is judged on `projects.submitted_by_token`, never the actor.
    """
    require_service_scope(request, project)
    row = await request.app.state.queries.get_project_by_name(project)
    if row is not None and not project_owned_by_caller(request, row):
        raise project_owned_error(project)
    return row


async def _create_applied_project(request: Request, project: str) -> Project:
    """Create the absent project for the caller (D-P40-5 rule 3); never on a dry run."""
    reject_reserved_name(project)
    try:
        return await create_project_for_caller(request, project)
    except ProjectExists:
        # Lost a create race: whoever won is judged like a row found up front.
        landed = await request.app.state.queries.get_project_by_name(project)
        if landed is None or not project_owned_by_caller(request, landed):
            raise project_owned_error(project) from None
        return landed


async def _missing_variables(
    request: Request, row: Project | None, labels: dict[str, str], required: list[str]
) -> list[str]:
    """`[vars] required` names that some declared service would launch without.

    Names only: the merged reader is asked for each service's key set and the
    values never leave this frame. The project scope is read because the
    caller passed the owner gate (`include_project`); a service scope only
    when its row (this project's, manageable by the caller) or its P39 claim
    is the caller's, so a foreign label's key names cannot leak through
    `missing`.
    """
    if not required:
        return []
    queries = request.app.state.queries
    mgr = secret_manager(request)
    missing: set[str] = set()
    for service, label in labels.items():
        existing = await queries.get_service_by_name(label)
        if existing is None:
            include_service = await claim_owned_by_caller(request, label)
        else:
            include_service = (
                row is not None
                and existing.project_id == row.id
                and existing.service == service
                and may_manage_job(request, existing)
            )
        names, _ = secret_call(
            lambda label=label, include_service=include_service: load_scoped(
                mgr,
                label,
                row.id if row is not None else None,
                include_service=include_service,
                include_project=row is not None,
            )
        )
        missing.update(key for key in required if key not in names)
    return sorted(missing)


async def _preflight_service(request: Request, project: str, declared: DeclaredService) -> bool:
    """Every per-service refusal, before the first build context. Returns "row exists"."""
    _, service, _ = declared
    label = service_label(project, PRODUCTION, service)  # proven by the declaration parse
    reject_reserved_name(service)
    reject_reserved_name(label)
    existing = await request.app.state.queries.get_service_by_name(label)
    reject_non_service_row(existing, label)
    reject_foreign_label(existing, declared, label)
    if existing is not None:
        require_owner_or_admin(request, existing)
    require_service_scope(request, label, project=project)
    if existing is None:
        await reject_foreign_claim(request, label)
    return existing is not None


@dataclass
class _ApplySource:
    """The one source of an apply: the spooled archive, a git coordinate, or the
    project's workspace (snapshotted into `archive` once the project is judged).

    `token` is the resolved private-repo credential: a local of the request,
    never logged, audited, persisted or returned (`repr=False`).
    """

    archive: UploadFile | None
    repo_url: str | None
    ref: str | None
    token_ref: str | None
    workspace: bool = False
    token: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> str:
        return "git" if self.repo_url is not None else "archive"


def _validated_source(
    request: Request,
    archive: UploadFile | None,
    repo_url: str | None,
    ref: str | None,
    token_ref: str | None,
    workspace: bool = False,
) -> _ApplySource:
    """Exactly one source, syntax-checked (`deploy_git`'s guards) before any lookup."""
    if (archive is not None) + (repo_url is not None) + workspace != 1:
        raise NerditError(
            422,
            "project.apply_source",
            "Send exactly one source: a multipart 'archive', 'repo_url', or 'workspace=true'.",
        )
    if repo_url is not None:
        token_ref = validate_git_request(
            request, repo_url, ref=ref, subdir=None, token_ref=token_ref
        )
    elif ref is not None or token_ref is not None:
        raise NerditError(
            422, "project.apply_source", "'ref' and 'token_ref' need a 'repo_url' source."
        )
    return _ApplySource(archive, repo_url, ref, token_ref, workspace)


async def _read_declaration(
    request: Request, project: str, row: Project | None, source: _ApplySource
) -> dict:
    """The source's parsed `nerdit.toml`; a git source resolves its token and clones here."""
    if source.workspace:
        # One locked snapshot serves the declaration read and every service's
        # context. Provenance stays `zip`: `workspace` on a row means "redeploy
        # from the workspace named after this LABEL", which no composed label has.
        snapshot = await snapshot_own_workspace(request, project)
        source.archive = UploadFile(file=io.BytesIO(snapshot), filename=f"{project}-workspace.zip")
    if source.archive is not None:
        return await _archive_declaration(request, source.archive)
    assert source.repo_url is not None
    # Before the clone, exactly as `deploy_git`: the service scope of the
    # project's own label is read only through a row or claim of the caller's.
    web = await request.app.state.queries.get_service_by_name(project)
    source.token = await _resolve_token_ref(
        request,
        project,
        source.token_ref,
        service_owned=(
            web is not None
            and row is not None
            and web.project_id == row.id
            and may_manage_job(request, web)
        ),
        repo_url=source.repo_url,
        project_id=row.id if row is not None else None,
    )
    return await _git_declaration(request, source.repo_url, source.ref, source.token)


async def _service_context(
    request: Request, source: _ApplySource
) -> tuple[Path, Path | None, dict]:
    """A fresh, disposable build context: `(context_dir, context_root, source_meta)`."""
    # ponytail: one extraction/clone per service -- `_finalize_deploy` owns one
    # context and the builder rmtrees it after the build, so N background builds
    # cannot share a tree. A shared, refcounted context with one cleanup is the
    # upgrade (it would also pin every service of a git apply to one commit).
    if source.repo_url is not None:
        info, root = await clone_for_request(
            request, source.repo_url, ref=source.ref, subdir=None, token=source.token
        )
        meta = git_source_meta(info, source.repo_url, token_ref=source.token_ref or None)
        return info.context_dir, root, meta
    assert source.archive is not None
    await source.archive.seek(0)
    settings = request.app.state.settings.daemon
    context = await extract_upload(
        source.archive,
        max_bytes=settings.max_upload_bytes,
        upload_root=Path(settings.upload_dir).expanduser(),
        dest_id=generate_id(),
    )
    return context, None, {"type": "zip"}


async def _apply_services(
    request: Request,
    project: str,
    members: list[DeclaredService],
    source: _ApplySource,
    *,
    dry_run: bool,
    done: list[str],
) -> ApplyResponse:
    """Pre-flight every service, then deploy them one by one; `done` feeds the audit row."""
    redeploys = [await _preflight_service(request, project, member) for member in members]
    results: list[ApplyServiceResult] = []
    urls: list[PublicUrlEntry] = []
    for member, redeploy in zip(members, redeploys, strict=True):
        service = member[1]
        label = service_label(project, PRODUCTION, service)
        try:
            context, context_root, source_meta = await _service_context(request, source)
            body = await _finalize_deploy(
                request,
                context,
                name=label,
                port=None,
                gpus=None,
                start=None,
                health=None,
                env=None,
                vendor=None,
                source_meta=source_meta,
                context_root=context_root,
                dry_run=dry_run,
                declared=member,
            )
        except NerditError as exc:
            raise NerditError(
                exc.status_code,
                "project.apply_incomplete",
                f"Service '{label}' was not applied ({exc.code}): {exc.message}",
                hint=exc.hint,
                failed={"label": label, "service": service, "code": exc.code},
                services=[r.model_dump(exclude_none=True) for r in results],
            ) from exc
        result = ApplyServiceResult(
            label=label, service=service, action="redeploy" if redeploy else "fresh"
        )
        if dry_run:
            result.plan = body
        else:
            result.status = body.get("status")
            result.build_version = body["build_version"]
            endpoint = body.get("endpoint") or {}
            urls += [PublicUrlEntry(**entry) for entry in endpoint.get("public_urls") or []]
            done.append(label)
        results.append(result)
    return ApplyResponse(
        project=project,
        status="planned" if dry_run else "applied",
        dry_run=dry_run,
        services=results,
        public_urls=urls,
    )


@router.post(
    "/projects/{project}/apply",
    response_model=ApplyResponse,
    response_model_exclude_none=True,
    operation_id="apply_project",
)
async def apply_project(
    request: Request,
    project: str,
    archive: UploadFile | None = None,
    repo_url: str | None = Form(None),
    ref: str | None = Form(None),
    token_ref: str | None = Form(None),
    workspace: bool = Form(
        False, description="Apply the caller's own workspace named after the project."
    ),
    dry_run: bool = Query(
        False, description="Validate and plan every service without writing anything."
    ),
) -> ApplyResponse:
    """Apply a `[project]` / `[services.*]` / `[vars]` declaration: an orchestrator (D-P40-12).

    The source is exactly one of a multipart `archive` (the `POST /deploy`
    ZIP), `repo_url` (+ `ref`, `token_ref`, the `POST /deploy/git` fields) or
    `workspace=true` (the caller's own workspace named after the project,
    snapshotted under its lock);
    its root `nerdit.toml` is the declaration. Order: scope, then project
    ownership (a foreign project is 409 `project.owned` with no key names),
    then the declaration (422s, `{project}` must equal `[project].name`), then
    `[vars] required` -- a miss answers 200 `waiting_for_variables` without
    deploying -- then every per-service refusal, and only then the services,
    one by one through the ordinary deploy tail. An absent project is created
    for the caller; `dry_run` creates nothing, mints nothing and writes nothing.

    There is no rollback: a failure on service N is `project.apply_incomplete`
    (the failing deploy's own status code) whose `services` lists the ones
    before it as done. Re-applying redeploys those and retries the rest.
    Audited `project.apply` with names only.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    done: list[str] = []
    source_kind = "workspace" if workspace else "git" if repo_url is not None else "archive"

    def params() -> dict:
        # Hand-built, names only: never a value, a URL or a member named
        # `environment` (D-P40-11).
        return audit_params(
            {
                "project": project,
                "services": list(done),
                "dry_run": dry_run,
                "source": source_kind,
            }
        )

    request.state.audit_params = params()
    source = _validated_source(request, archive, repo_url, ref, token_ref, workspace)
    try:
        row = await _judge_apply_project(request, project)
        data = await _read_declaration(request, project, row, source)
        try:
            # The D-P40-14 grammar binds only where a NEW name enters.
            name, tables, required = parse_project_declaration(data, new_project=row is None)
        except DeclarationError as exc:
            raise _declaration_error(exc) from exc
        if name != project:
            raise NerditError(
                422,
                "project.name_mismatch",
                f"The declaration names project '{name}', not '{project}'.",
                hint=f"Apply it to /projects/{name}/apply, or fix [project].name.",
            )
        if row is None and not dry_run:
            row = await _create_applied_project(request, project)

        labels = {svc: service_label(project, PRODUCTION, svc) for svc in tables}
        missing = await _missing_variables(request, row, labels, required)
        if missing:
            return ApplyResponse(
                project=project,
                status="waiting_for_variables",
                dry_run=dry_run,
                missing=missing,
                hint=(
                    f"Set each one with `nerdit vars set {project} --secret --prompt KEY` "
                    f"(or `nerdit vars set {project} KEY=value` for a plain one), then apply again."
                ),
            )
        members: list[DeclaredService] = [(row, svc, table) for svc, table in tables.items()]
        return await _apply_services(request, project, members, source, dry_run=dry_run, done=done)
    finally:
        # `_finalize_deploy` adds masked `ai` / `db` members to the params it
        # finds; the apply row names what was applied, however the request ends.
        request.state.audit_params = params()
