"""Inbound webhooks of a deployed app: list, add, remove. Mounted under /api only.

The `hooks` table is the truth (the domains precedent). Every hook URL is a
capability that alone opens an unsigned 'none' hook, so even the list is
owner-or-admin. No secret travels through these routes: the signing secret is
set with the existing `POST /api/secrets/{service}` under `secret_key`
(D-WH-7), and removing a hook keeps it.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import ValidationError

from nerdit.config.project import HOOK_NAME_ERROR, HOOK_NAME_RE, HOOK_SENDERS, HookConfig
from nerdit.core.hooks import default_secret_ref, hook_url, secret_key
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_owner_or_admin, require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas.exposure import (
    HookListView,
    HookRemovedView,
    HookRequest,
    HookSetView,
    HookView,
)
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import Job, JobKind, TokenRole
from nerdit.db.rows import Hook

router = APIRouter()

_REFUSED = "<refused>"


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route `Idempotency-Key` gate; a local copy (a route never imports a route)."""
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            f"A {what} requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )


def _view(request: Request, row: Hook) -> HookView:
    settings = getattr(request.app.state, "settings", None)
    link = getattr(settings, "link", None) if settings is not None else None
    return HookView(
        name=row.name,
        sender=row.sender,
        path=row.path,
        header=row.header,
        prefix=row.prefix,
        secret_ref=row.secret_ref,
        secret_key=secret_key(row.name) if row.secret_ref == default_secret_ref(row.name) else None,
        url=hook_url(
            row.service_name,
            row.hook_id,
            getattr(link, "slug", None),
            getattr(link, "nodes_base_domain", None),
        ),
        created_at=row.created_at,  # type: ignore[arg-type]  # pydantic parses the SQLite text
    )


async def _owned_service(request: Request, name: str) -> Job:
    job = await _resolve_service(request.app.state.queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)
    return job


def _invalid(message: str, reason: str) -> NerditError:
    # Value-free: the message never quotes the input; `reason` is top-level
    # because the MCP mapper drops `detail` and copies extras through.
    return NerditError(
        422,
        "hook.invalid",
        message,
        hint=(
            "sender is stripe, github, slack, hmac or none; path starts with '/'; "
            "header (hmac only, required) and prefix (hmac only)."
        ),
        reason=reason,
    )


@router.get("/services/{name}/hooks", response_model=HookListView, operation_id="list_hooks")
async def list_hooks(request: Request, name: str) -> HookListView:
    """List a service's live webhooks; owner or admin only (each url is a capability)."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    job = await _owned_service(request, name)
    service_name = job.service_name or name
    rows = await request.app.state.queries.list_hooks(service_name, job_id=job.id)
    return HookListView(service_name=service_name, hooks=[_view(request, r) for r in rows])


@router.post("/services/{name}/hooks", response_model=HookSetView, operation_id="add_hook")
async def add_hook(request: Request, name: str, body: HookRequest) -> HookSetView:
    """Add an inbound webhook; a replay answers 200 created=false.

    Refusals: 404 unknown service, 400 idempotency_key_required, 422
    hook.kind_unsupported / hook.invalid, 409 hook.exists. The response's
    `secret_key` names the S2 key to set with `POST /api/secrets/{service}`.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    # Stamped before the refusals so a denied attempt is recorded; free-form
    # values that did not validate are never recorded (a mis-paste could be a secret).
    name_ok = bool(HOOK_NAME_RE.fullmatch(body.name))
    request.state.audit_params = audit_params(
        {
            "service": name,
            "name": body.name if name_ok else _REFUSED,
            "sender": body.sender if body.sender in HOOK_SENDERS else _REFUSED,
        }
    )
    job = await _owned_service(request, name)
    _require_idempotency_key(request, "webhook write")
    if job.kind != JobKind.service:
        raise NerditError(
            422,
            "hook.kind_unsupported",
            f"'{job.kind.value}' workloads cannot receive webhooks.",
            hint="Only deployed apps (kind 'service') can receive webhooks.",
        )
    if not name_ok:
        raise _invalid(HOOK_NAME_ERROR, "name")
    secret_ref = None if body.sender == "none" else default_secret_ref(body.name)
    try:
        spec = HookConfig(
            sender=body.sender,  # type: ignore[arg-type]  # validated by the model
            path=body.path,
            header=body.header,
            prefix=body.prefix,
            secret=secret_ref,
        )
    except ValidationError as exc:
        # hide_input_in_errors + value-free validators: the messages carry no input.
        message = "; ".join(str(e.get("msg", "")) for e in exc.errors()) or "Invalid webhook."
        raise _invalid(message, "shape") from exc
    request.state.audit_params = audit_params(
        {"service": name, "name": body.name, "sender": spec.sender, "path": spec.path}
    )
    service_name = job.service_name or name
    outcome, row = await request.app.state.queries.add_hook(
        service_name,
        job_id=job.id,
        name=body.name,
        sender=spec.sender,
        path=spec.path,
        header=spec.header,
        prefix=spec.prefix,
        secret_ref=secret_ref,
    )
    if outcome == "no_service":
        raise _not_found(name)
    if outcome == "conflict":
        raise NerditError(
            409,
            "hook.exists",
            f"Service '{service_name}' already has a different webhook named '{body.name}'.",
            hint="Remove it first, or edit nerdit.toml if it is declared there.",
        )
    assert row is not None
    return HookSetView(**_view(request, row).model_dump(), created=outcome == "inserted")


@router.delete(
    "/services/{name}/hooks/{hook}", response_model=HookRemovedView, operation_id="remove_hook"
)
async def remove_hook(request: Request, name: str, hook: str) -> HookRemovedView:
    """Revoke a webhook; idempotent (200 removed=false). Its S2 secret is kept."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    valid = bool(HOOK_NAME_RE.fullmatch(hook))
    request.state.audit_params = audit_params(
        {"service": name, "name": hook if valid else _REFUSED}
    )
    job = await _owned_service(request, name)
    service_name = job.service_name or name
    removed = valid and await request.app.state.queries.remove_hook(
        service_name, hook, job_id=job.id
    )
    return HookRemovedView(
        service_name=service_name, name=hook if valid else _REFUSED, removed=removed
    )
