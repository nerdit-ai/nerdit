"""Authentication utility endpoints."""

from fastapi import APIRouter, Request

from nerdit.daemon.auth import current_principal

router = APIRouter()


@router.get("/auth/check", operation_id="check_auth")
async def auth_check(request: Request) -> dict[str, bool | str]:
    """Validate the bearer token and report the principal's role.

    The role lets clients gate write UI up front (P8: the dashboard hides the
    shared-secrets write form for non-admins) instead of discovering it via a
    first 403 — the server-side checks remain authoritative.
    """
    return {"ok": True, "role": current_principal(request).role.value}
