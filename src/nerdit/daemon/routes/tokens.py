"""Manage scoped tokens and caller self-service.

Admin-only create/list/revoke never expose hashes. Creation returns plaintext
once; idempotent replays return a non-secret envelope. Self-read and self-rotate
use the caller's identity; sentinel principals receive synthetic views.
Rotation also returns plaintext only once. Mounted under /api only.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Query, Request

from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import current_principal, generate_token, hash_token, require_role
from nerdit.daemon.errors import NerditError
from nerdit.db.models import (
    ApiToken,
    TokenCreateRequest,
    TokenCreateResponse,
    TokenRole,
    TokenRotateRequest,
    TokenSelfView,
    TokenView,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/tokens", response_model=TokenCreateResponse, status_code=201, operation_id="create_token"
)
async def create_token(request: Request, body: TokenCreateRequest) -> TokenCreateResponse:
    """Create a scoped API token (admin-only); the plaintext is shown once."""
    require_role(request, TokenRole.admin)
    # A scoped admin is a contradiction — an admin can mint itself
    # an unscoped token in one call — so it is refused at creation rather than
    # silently ignored at enforcement time.
    if body.role == TokenRole.admin and body.scope_services is not None:
        raise NerditError(
            422,
            "token.scope_not_allowed",
            "An admin token cannot be scoped to specific services.",
            hint="Create the token with role='submitter' to scope it, or omit scope_services.",
        )
    # Redacted params for audit — the body holds no secret (the raw token is
    # generated below and never appears here), but redaction runs defensively.
    # `expires_in_s`/`scope_services` are non-secret and deliberately DO
    # reach the audit row: minting a short-lived scoped token is exactly the
    # event an auditor wants the parameters of (P25 §3.1.4).
    request.state.audit_params = audit_params(body)

    # omitted vs explicit-null: both give `None` as a value, so
    # the distinction is read off `model_fields_set`. Omitted ⇒ the site
    # policy applies; an explicit `null` means "no expiry" and wins over it.
    ttl_given = "expires_in_s" in body.model_fields_set
    default_ttl = request.app.state.settings.security.token_default_ttl_s
    ttl = body.expires_in_s if ttl_given else default_ttl
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl) if ttl is not None else None

    raw = generate_token()
    token = ApiToken(
        name=body.name,
        role=body.role,
        token_hash=hash_token(raw),
        max_gpus=body.max_gpus,
        max_concurrent_jobs=body.max_concurrent_jobs,
        expires_at=expires_at,
        scope_services=body.scope_services,
    )
    created = await request.app.state.queries.create_api_token(token)
    view = TokenView.from_token(created)
    return TokenCreateResponse(**view.model_dump(), token=raw)


@router.get("/tokens", response_model=list[TokenView], operation_id="list_tokens")
async def list_tokens(
    request: Request,
    include_revoked: bool = Query(False, description="Include revoked tokens"),
) -> list[TokenView]:
    """List API tokens (admin-only). Hashes and plaintext are never returned."""
    require_role(request, TokenRole.admin)
    tokens = await request.app.state.queries.list_api_tokens(include_revoked=include_revoked)
    return [TokenView.from_token(t) for t in tokens]


@router.get("/tokens/self", response_model=TokenSelfView, operation_id="get_self_token")
async def get_self_token(request: Request) -> TokenSelfView:
    """Return the caller's own token — any authenticated principal.

    Deliberately **no** `require_role`: self-inspection is universal, and a
    `readonly` token needs it most (it is the one role that cannot rotate).
    Not audited — reads never are (house rule); flagged for the security review.
    """
    principal = current_principal(request)
    if principal.token_id is None:
        # LEGACY_ADMIN / LOCAL / the fail-closed ANONYMOUS guard: no row exists.
        return TokenSelfView.synthetic(name=principal.name, role=principal.role)

    row = await request.app.state.queries.get_api_token_by_id(principal.token_id)
    if row is None:
        # Unreachable in practice: the middleware resolved this principal from a
        # live row microseconds ago. Answered honestly rather than crashed.
        raise NerditError(
            404,
            "not_found",
            "The token behind this request no longer exists.",
            hint="Ask an admin for a new token ('nerdit token create').",
        )
    return TokenSelfView.from_token(row)


@router.post(
    "/tokens/self/rotate", response_model=TokenCreateResponse, operation_id="rotate_self_token"
)
async def rotate_self_token(
    request: Request, body: TokenRotateRequest | None = None
) -> TokenCreateResponse:
    """Rotate the caller's own token in place; the new plaintext is shown once. Two denials
    are deliberately NOT implemented here because they
    already happen upstream, and re-implementing them would let the two copies
    drift: a `readonly` token is stopped by the coarse write gate (403
    `forbidden`) and an expired one by the expiry check (403 `token_expired`,
    D-P25-2 sub-ruling — an expired credential must not mint itself a fresh
    lifetime). Both are pinned by tests so a later exemption is a deliberate diff.
    """
    principal = current_principal(request)
    if principal.token_id is None:
        raise NerditError(
            409,
            "token.not_rotatable",
            "This principal has no token row to rotate.",
            hint=(
                "You are authenticated with the legacy global token (or no token at "
                "all): rotate it by editing [daemon].auth_token in ~/.nerdit/config.toml "
                "and restarting the daemon."
            ),
        )

    body = body or TokenRotateRequest()
    # Non-secret lifecycle parameters, exactly like `token.create`: the minted
    # plaintext is generated below and never passes through the body.
    request.state.audit_params = audit_params(body)
    request.state.audit_target = principal.token_id

    queries = request.app.state.queries
    row = await queries.get_api_token_by_id(principal.token_id)
    if row is None:
        raise NerditError(
            404,
            "not_found",
            "The token behind this request no longer exists.",
            hint="Ask an admin for a new token ('nerdit token create').",
        )

    # Rotation never silently extends. `extend` is the only way
    # the clock moves, and with no TTL policy anywhere there is nothing to move
    # it to — so the flag degrades to a no-op rather than clearing the expiry.
    expires_at: datetime | None = None
    set_expiry = False
    if body.extend:
        ttl = body.expires_in_s
        if ttl is None:
            ttl = request.app.state.settings.security.token_default_ttl_s
        if ttl is not None:
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl)
            set_expiry = True

    raw = generate_token()
    new_hash = hash_token(raw)
    swapped = await queries.rotate_api_token_hash(
        principal.token_id,
        new_hash,
        expires_at=expires_at,
        set_expiry=set_expiry,
    )
    if not swapped:
        raise NerditError(
            404,
            "not_found",
            "The token behind this request is no longer active.",
            hint="Ask an admin for a new token ('nerdit token create').",
        )

    rotated = row.model_copy(
        update={
            "token_hash": new_hash,
            "last_used_at": None,
            **({"expires_at": expires_at} if set_expiry else {}),
        }
    )
    view = TokenView.from_token(rotated)
    return TokenCreateResponse(**view.model_dump(), token=raw)


@router.delete("/tokens/{token_id}", operation_id="revoke_token")
async def revoke_token(request: Request, token_id: str) -> dict:
    """Revoke (soft-delete) a token (admin-only)."""
    require_role(request, TokenRole.admin)
    request.state.audit_params = audit_params({"token_id": token_id})

    revoked = await request.app.state.queries.revoke_api_token(token_id)
    if not revoked:
        raise NerditError(
            404,
            "not_found",
            f"No active token '{token_id}' to revoke.",
            hint="The token may not exist or may already be revoked.",
        )
    return {"id": token_id, "revoked": True}
