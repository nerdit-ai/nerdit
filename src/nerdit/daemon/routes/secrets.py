"""Manage write-only service secrets and encryption-key rotation.

Reads list names only; values are injected at container launch, never returned.
Writes require scope/role authorization, idempotency and redacted auditing.
Shared scope maps to _shared storage: only admins write it, any authenticated
principal may read names. A rowless name is authorized through its claim
(`daemon/secret_scope.py`): the first write reserves it for the caller until a
deploy consumes it. Key rotation is admin-only and audits counts only.
Mounted under /api only.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from nerdit.core.secrets import SecretDecryptError, SecretRotationInProgress
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.daemon.secret_scope import (
    authorize_secret_scope,
    secret_call,
    secret_manager,
    set_secret_values,
    storage_name,
    variable_write_lock,
)
from nerdit.db.models import TokenRole

logger = logging.getLogger(__name__)

router = APIRouter()


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
        count = await asyncio.to_thread(secret_manager(request).rotate_key)
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
    """Set/merge secrets for a service. Returns the resulting key names only.

    A name with no row is reserved for the caller's token until deployed.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    # Audit the *names* only — never the values (defense-in-depth on top of the
    # middleware's env/secret redaction).
    request.state.audit_params = audit_params(
        {"service": service, "keys": sorted(body.values.keys())}
    )
    if not body.values:
        raise NerditError(400, "secret.empty", "No secret values provided.")
    keys = await set_secret_values(request, service, body.values)
    return SecretNamesResponse(service=service, keys=keys)


@router.get(
    "/secrets/{service}", response_model=SecretNamesResponse, operation_id="list_secret_names"
)
async def list_secret_names(request: Request, service: str) -> SecretNamesResponse:
    """List a service's secret key names (values are never returned).

    Intentionally no coarse role gate: the read is scoped to the service's
    owning token (or an admin) via the ownership check below.
    """
    await authorize_secret_scope(request, service, write=False)
    return SecretNamesResponse(
        service=service, keys=secret_call(secret_manager(request).list_keys, storage_name(service))
    )


@router.delete("/secrets/{service}/{key}", status_code=200, operation_id="delete_service_secret")
async def delete_secret_key(request: Request, service: str, key: str):
    """Delete a single secret key from a service."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"service": service, "key": key})
    async with variable_write_lock(request.app):  # verdict and delete are one section
        await authorize_secret_scope(request, service, write=True)
        existed = secret_call(secret_manager(request).delete_key, storage_name(service), key)
    if not existed:
        raise NerditError(404, "not_found", f"No secret '{key}' for service '{service}'.")
    return {"service": service, "deleted": key}


@router.delete("/secrets/{service}", status_code=200, operation_id="delete_service_secrets")
async def delete_all_secrets(request: Request, service: str):
    """Delete all secrets for a service (removes the whole file and its claim)."""
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"service": service})
    # Under the writers' lock: releasing the claim ends an owner epoch, and a
    # write parked on the lock must re-authorize AFTER it, never land across it.
    async with variable_write_lock(request.app):
        await authorize_secret_scope(request, service, write=True)
        existed = secret_call(secret_manager(request).delete, storage_name(service))
        # The claim reserves the name only while secrets exist for it; a
        # single-key delete keeps both.
        await request.app.state.queries.delete_secret_claim(service)
    return {"service": service, "deleted": existed}
