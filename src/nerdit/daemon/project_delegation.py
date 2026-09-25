"""Fail-closed admission and indirect-access limits for project-only links."""

from __future__ import annotations

from dataclasses import replace

from fastapi import Request

from nerdit.core.project_identity import PROJECT_ID_RE, PROJECT_MCP_PATH, SERVICE_NAME_RE
from nerdit.daemon.auth import Principal, current_principal
from nerdit.daemon.errors import NerditError
from nerdit.db.models import Job


def delegation_denial() -> NerditError:
    """One value-free refusal for paths, selectors and indirect authority."""
    return NerditError(
        403,
        "project.delegation_forbidden",
        "This operation is not available under project-only delegation.",
        hint="Use an explicit node grant for operations outside the delegated project surface.",
    )


def require_project_selector(request: Request, ident: str) -> None:
    """Require the exact immutable ID, before any name or foreign-row lookup."""
    project_id = current_principal(request).project_id
    if project_id is not None and ident != project_id:
        raise delegation_denial()


async def narrow_project_request(
    request: Request, principal: Principal, headers: list[str]
) -> Principal:
    """Validate the link carrier and exact REST allowlist before any cached replay."""
    if len(headers) != 1 or PROJECT_ID_RE.fullmatch(headers[0]) is None:
        raise delegation_denial()
    project_id = headers[0]
    path = request.url.path
    allowed = path in {PROJECT_MCP_PATH, PROJECT_MCP_PATH + "/"} and request.method == "POST"
    prefix = f"/api/projects/{project_id}"
    suffix = path.removeprefix(prefix) if path.startswith(prefix) else None
    if request.method == "GET":
        allowed |= suffix in {"", "/variables/resolve"}
        if suffix is not None:
            parts = suffix.split("/")
            allowed |= (
                len(parts) == 4
                and parts[1] == "services"
                and SERVICE_NAME_RE.fullmatch(parts[2]) is not None
                and parts[3] in {"logs", "diagnose"}
            )
    elif request.method == "PUT":
        allowed |= suffix == "/variables"
    if not allowed:
        raise delegation_denial()
    if await request.app.state.queries.get_project(project_id) is None:
        raise NerditError(404, "not_found", f"No project '{project_id}'.")
    return replace(principal, project_id=project_id)


async def require_live_project_jobs(request: Request, jobs: list[Job]) -> None:
    """Recheck captured job IDs after projections that also read label-keyed state."""
    project_id = current_principal(request).project_id
    if project_id is None:
        return
    queries = request.app.state.queries
    if await queries.get_project(project_id) is None:
        raise NerditError(404, "not_found", f"No project '{project_id}'.")
    for job in jobs:
        current = await queries.get_job(job.id)
        if current is None or current.project_id != project_id:
            raise NerditError(404, "not_found", "The project service no longer exists.")
