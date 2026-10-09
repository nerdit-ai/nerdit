"""Authentication utility endpoints."""

from fastapi import APIRouter, Request

from nerdit.daemon.auth import LOCAL, current_principal, is_link_token_id

router = APIRouter()


@router.get("/auth/check", operation_id="check_auth")
async def auth_check(request: Request) -> dict[str, bool | str]:
    """Report the authenticated role and connection mode, never credentials.

    The role lets clients gate write UI up front (P8: the dashboard hides the
    shared-secrets write form for non-admins) instead of discovering it via a
    first 403 — the server-side checks remain authoritative.
    """
    principal = current_principal(request)
    mode = (
        "tunnel"
        if is_link_token_id(principal.token_id)
        else "local"
        if principal == LOCAL
        else "token"
    )
    return {"ok": True, "role": principal.role.value, "mode": mode}
