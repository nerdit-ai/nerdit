"""Shared ordered creation steps for models and databases.

Authorize, audit, resolve the backend, validate the name, build the row and
reserve it under quota/name guards. Resource-specific checks, credential minting
and response projection stay in callers, which supply their own error wording.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import Request

from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import Principal, QuotaExceeded, current_principal, require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.routes.services import reject_reserved_name
from nerdit.daemon.secret_scope import name_claimed_error, project_owned_error
from nerdit.db.models import Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries import ProjectOwned, ServiceNameClaimed, ServiceNameTaken

__all__ = [
    "authorize_create",
    "resolve_backend_or_422",
    "reject_reserved_name",
    "new_workload_row",
    "reserve_or_conflict",
]

_B = TypeVar("_B")


def authorize_create(request: Request, body: Any) -> Principal:
    """Require `submitter`/`admin` and stamp the audit params.

    Identical first step on both create routes (models.py:117-118,
    databases.py:126-127).
    """
    principal = require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params(body)
    return principal


def resolve_backend_or_422(
    get_backend: Callable[[str | None], _B | None],
    name: str | None,
    *,
    code: str,
    message: str,
    hint: str,
) -> _B:
    """Resolve a named backend via `get_backend`, or raise the caller's 422.

    The code/message/hint are per-side (models: a hardcoded backend list;
    databases: computed from `controller.backends`) — this helper only
    unifies the "not found" branch, not the text.
    """
    backend = get_backend(name)
    if backend is None:
        raise NerditError(422, code, message, hint=hint)
    return backend


def new_workload_row(
    request: Request,
    *,
    kind: JobKind,
    service_name: str,
    gpu_count: int,
    health_check: dict[str, object] | None,
    config: dict[str, Any],
) -> Job:
    """Build the shared `building`/`running` Job skeleton for a create route.

    `restart_policy` is the fixed `"on-failure"` literal both routes pass
    today (models.py:158, databases.py:176) — not exposed as a parameter.
    `submitted_by_token`/`idempotency_key` are read from `request` (the
    same principal `authorize_create` already validated, and the same
    `Idempotency-Key` header both routes read identically).
    """
    principal = current_principal(request)
    return Job(
        kind=kind,
        service_name=service_name,
        name=service_name,
        gpu_count=gpu_count,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        health_check=health_check,
        config=json.dumps(config),
        submitted_by_token=principal.token_id,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


async def reserve_or_conflict(
    request: Request, queries: Any, job: Job, *, service_name: str, name_taken_hint: str
) -> Job:
    """Reserve the row for the submitting token.

    Maps `ServiceNameTaken` to the shared 409 `service.name_taken` code +
    message (the hint stays per-side), `ServiceNameClaimed` to 409
    `service.name_claimed`, `ProjectOwned` to 409 `project.owned` (P40b), and
    `QuotaExceeded` to its own error envelope.
    """
    try:
        return await queries.reserve_service_for_token(
            job, admin=current_principal(request).role is TokenRole.admin
        )
    except ServiceNameTaken as exc:
        raise NerditError(
            409,
            "service.name_taken",
            f"A service named '{service_name}' already exists.",
            hint=name_taken_hint,
        ) from exc
    except ServiceNameClaimed as exc:
        raise name_claimed_error(service_name) from exc
    except ProjectOwned as exc:
        raise project_owned_error(service_name) from exc
    except QuotaExceeded as exc:
        raise exc.to_error() from exc
