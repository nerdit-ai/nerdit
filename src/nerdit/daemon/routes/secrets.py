"""Manage write-only service secrets and encryption-key rotation.

Reads list names only; values are injected at container launch, never returned.
Writes require scope/role authorization, idempotency and redacted auditing.
Shared scope maps to _shared storage: only admins write it, any authenticated
principal may read names. Key rotation is admin-only and audits counts only.
Mounted under /api only.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from nerdit.core.secrets import (
    SHARED_SCOPE,
    InvalidSecretKey,
    InvalidSecretValue,
    InvalidServiceName,
    SecretDecryptError,
    SecretRotationInProgress,
)
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_owner_or_admin, require_role, require_service_scope
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.models import TokenRole

logger = logging.getLogger(__name__)

router = APIRouter()

# User-facing name of the shared (global) secrets scope. Routes translate
# it to the internal storage name (`SHARED_SCOPE`); a service can never take
# this name (`service.reserved_name` on create/deploy).
_SHARED_PUBLIC = "shared"


class SecretSetRequest(StrictRequestModel):
    """Body for setting/merging secrets: a map of KEY → value.

    Every byte of this body is a credential, and a pydantic 422 echoes the
    submitted `input`, so `values` is listed in `errors._SECRET_INPUT_FIELDS`
    — without it a mistyped field name (`{"valus": {...}}`) reflects every
    secret value into an envelope that agents and CLIs routinely log. The secret
    *keys* are caller-chosen, so no name-based rule could mask them one by one;
    the whole map is opaque.
    """

    values: dict[str, str] = Field(..., description="Secret key/value pairs to merge")


class SecretNamesResponse(BaseModel):
    """Write-only projection: the service and its secret key *names* (no values)."""

    service: str
    keys: list[str]


class RotateKeyResponse(BaseModel):
    """Result of a key rotation: how many secrets files were re-encrypted."""

    services_rewritten: int


def _manager(request: Request):
    mgr = getattr(request.app.state, "secret_manager", None)
    if mgr is None:  # pragma: no cover - always wired in the daemon
        raise NerditError(500, "internal", "Secret manager is not configured.")
    return mgr


def _call(fn, *args):
    """Run a SecretManager operation, mapping its errors to the envelope.

    Every CRUD handler reaches `load()` internally (set/delete merge the
    existing file), so a corrupt or wrong-key `.enc` can surface anywhere —
    the structured 500 keeps the kid-bearing restore hint reaching the caller
    instead of a bare Starlette 500. Messages are hygienic by construction
    (paths, service names and kids only).
    """
    try:
        return fn(*args)
    except InvalidServiceName as exc:
        raise NerditError(422, "secret.invalid_service", str(exc)) from exc
    except InvalidSecretKey as exc:
        raise NerditError(
            422,
            "secret.invalid_key",
            str(exc),
            hint="Key names are env-var names: letters, digits and '_', not starting with a digit.",
        ) from exc
    except InvalidSecretValue as exc:
        raise NerditError(
            422,
            "secret.invalid_value",
            str(exc),
            hint="Values may not contain NUL or control characters (tab/newline/CR are allowed).",
        ) from exc
    except SecretDecryptError as exc:
        raise NerditError(500, "secret.decrypt_failed", str(exc)) from exc


def _storage_name(service: str) -> str:
    """Translate the user-facing `shared` scope to its internal storage name."""
    return SHARED_SCOPE if service == _SHARED_PUBLIC else service


def _guard_name(service: str) -> None:
    from nerdit.core.secrets import _DNS_LABEL_RE

    if not _DNS_LABEL_RE.match(service):
        raise NerditError(
            422,
            "secret.invalid_service",
            f"Invalid service name '{service}'.",
            hint="Service names are DNS labels (lowercase, digits, '-').",
        )


async def _authorize_service(request: Request, service: str, *, write: bool) -> None:
    """Authorize secret access, checking scope before service lookup to prevent oracles.

    Service secrets require an existing row and owner/admin, including names-only
    reads. Shared scope skips row lookup: admins write, any authenticated caller
    reads names. Refuse all shared operations if a legacy service owns that name;
    never reinterpret its secrets as shared storage.
    """
    if service == _SHARED_PUBLIC:
        if getattr(request.app.state, "shared_scope_blocked", False):
            raise NerditError(
                409,
                "secret.shared_unavailable",
                "A service named 'shared' predates the reserved shared scope; "
                "shared-scope secrets are disabled.",
                hint="Rename or delete that service, then restart the daemon.",
            )
        if write:
            require_role(request, TokenRole.admin)
        return
    require_service_scope(request, service)
    existing = await request.app.state.queries.get_service_by_name(service)
    if existing is None:
        raise NerditError(
            404,
            "not_found",
            f"No service '{service}'.",
            hint="Deploy the service first, then set its secrets.",
        )
    require_owner_or_admin(request, existing)


# Registered BEFORE the parameterized routes so the literal path wins over
# `POST /secrets/{service}` (its audit rule likewise sits above `secret.set`).
@router.post(
    "/secrets/rotate-key", response_model=RotateKeyResponse, operation_id="rotate_secrets_key"
)
async def rotate_secrets_key(request: Request) -> RotateKeyResponse:
    """Rotate the secrets encryption key, re-encrypting every stored file.

    Admin-only (a key-custody operation), idempotent, and audited
    (`secret.rotate_key`) with counts only — key material never reaches the
    audit store or this response. A crash mid-rotation is resumed at the next
    daemon startup; until then a second rotation is refused with a 409.
    """
    require_role(request, TokenRole.admin)
    if getattr(request.app.state, "rotate_key_blocked", False):
        # A legacy service literally named 'rotate-key' predates the reserved
        # name; refuse rather than rotate when the admin may have meant to write
        # that service's secrets (the startup guard set this flag).
        raise NerditError(
            409,
            "secret.rotate_key_unavailable",
            "A service named 'rotate-key' predates the reserved name; key rotation is disabled.",
            hint="Rename or delete that service, then restart the daemon.",
        )
    try:
        count = await asyncio.to_thread(_manager(request).rotate_key)
    except SecretRotationInProgress as exc:
        raise NerditError(
            409,
            "secret.rotation_in_progress",
            str(exc),
            hint="Restart the daemon to resume the interrupted rotation, then retry.",
        ) from exc
    except SecretDecryptError as exc:
        raise NerditError(500, "secret.decrypt_failed", str(exc)) from exc
    request.state.audit_params = audit_params({"services_rewritten": count})
    return RotateKeyResponse(services_rewritten=count)


@router.post(
    "/secrets/{service}", response_model=SecretNamesResponse, operation_id="set_service_secrets"
)
async def set_secrets(
    request: Request, service: str, body: SecretSetRequest
) -> SecretNamesResponse:
    """Set/merge secrets for a service. Returns the resulting key names only."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    # Audit the *names* only — never the values (defense-in-depth on top of the
    # middleware's env/secret redaction).
    request.state.audit_params = audit_params(
        {"service": service, "keys": sorted(body.values.keys())}
    )
    _guard_name(service)
    await _authorize_service(request, service, write=True)
    if not body.values:
        raise NerditError(400, "secret.empty", "No secret values provided.")
    keys = _call(_manager(request).set, _storage_name(service), body.values)
    return SecretNamesResponse(service=service, keys=keys)


@router.get(
    "/secrets/{service}", response_model=SecretNamesResponse, operation_id="list_secret_names"
)
async def list_secret_names(request: Request, service: str) -> SecretNamesResponse:
    """List a service's secret key names (values are never returned).

    Intentionally no coarse role gate: the read is scoped to the service's
    owning token (or an admin) via the ownership check below.
    """
    _guard_name(service)
    await _authorize_service(request, service, write=False)
    return SecretNamesResponse(
        service=service, keys=_call(_manager(request).list_keys, _storage_name(service))
    )


@router.delete("/secrets/{service}/{key}", status_code=200, operation_id="delete_service_secret")
async def delete_secret_key(request: Request, service: str, key: str):
    """Delete a single secret key from a service."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"service": service, "key": key})
    _guard_name(service)
    await _authorize_service(request, service, write=True)
    if not _call(_manager(request).delete_key, _storage_name(service), key):
        raise NerditError(404, "not_found", f"No secret '{key}' for service '{service}'.")
    return {"service": service, "deleted": key}


@router.delete("/secrets/{service}", status_code=200, operation_id="delete_service_secrets")
async def delete_all_secrets(request: Request, service: str):
    """Delete all secrets for a service (removes the whole file)."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"service": service})
    _guard_name(service)
    await _authorize_service(request, service, write=True)
    existed = _call(_manager(request).delete, _storage_name(service))
    return {"service": service, "deleted": existed}
