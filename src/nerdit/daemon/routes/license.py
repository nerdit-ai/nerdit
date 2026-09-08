"""Install or remove offline product licenses on the daemon's filesystem.

This is separate from the repository's Apache-2.0 license. Both routes are
admin-only and require Idempotency-Key for durable file effects. Verify before
writing: invalid blobs cannot replace a working license. Signature-valid expired
or grace-period licenses are accepted with their computed state.

Refresh app.state.license immediately after writes; no caller may assume the
license content is boot-frozen. The configured file path still requires restart.
Audit/events carry only lid, plan and state, never customer_id or blob. Blobs
must also stay out of logs, errors and validation echoes. No MCP tool exposes
this custody operation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

from nerdit.core.eventlog import get_recorder
from nerdit.core.license import (
    STATE_INVALID,
    TRUSTED_LICENSE_KEYS,
    LicenseError,
    LicenseState,
    install_license_file,
    remove_license_file,
    resolve_license_file,
    utcnow,
    verify_license,
)
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas.license import (
    LicenseInstallRequest,
    LicenseInstallView,
    LicenseRemoveView,
)
from nerdit.daemon.schemas.tokens import seconds_until
from nerdit.db.models import TokenRole
from nerdit.db.queries._base import mark_request_side_effect

if TYPE_CHECKING:  # pragma: no cover — typing only
    from collections.abc import Callable, Mapping
    from datetime import datetime
    from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

#: Serializes install against remove. Both are read → awaited file work →
#: in-memory refresh over the same path and the same holder; without this an
#: overlapping remove can unlink the file an install just wrote while leaving
#: the holder describing it (or the reverse). Both are rare interactive admin
#: operations, so a module lock is proportionate.
_LICENSE_MUTATION_LOCK = asyncio.Lock()


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route `Idempotency-Key` gate (the config-write wording, D2)."""
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            f"A {what} requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )


def _holder(request: Request) -> LicenseState:
    """The `app.state.license` holder, created on demand.

    The daemon always builds one at boot (`bootstrap.build_license_state`);
    the fallback exists so a hand-built app — or a boot that predates the
    holder — refreshes into something rather than silently dropping the update.
    """
    holder = getattr(request.app.state, "license", None)
    if holder is None:
        holder = LicenseState()
        request.app.state.license = holder
    return holder


def _trusted_keys(request: Request) -> Mapping[str, str]:
    """The pinned keyset, or the test-injected one (D-LIC3).

    Injection is the ONLY seam: there is deliberately no
    `[license].trusted_keys` config override, so an operator cannot enroll a
    verifier — see the D-LIC3 rationale in `core/license.py`.
    """
    injected = getattr(request.app.state, "license_trusted_keys", None)
    return TRUSTED_LICENSE_KEYS if injected is None else injected


def _clock(request: Request) -> Callable[[], datetime]:
    """The holder's injected clock, so install and doctor agree on "now"."""
    holder = getattr(request.app.state, "license", None)
    return holder.clock if holder is not None else utcnow


def _license_path(request: Request) -> Path:
    """The boot-resolved install path (`[license].file` is restart-keyed)."""
    settings = request.app.state.settings
    file = getattr(getattr(settings, "license", None), "file", None)
    return resolve_license_file(file, str(settings.data_dir))


@router.post(
    "/license",
    response_model=LicenseInstallView,
    operation_id="install_license",
    status_code=200,
)
async def install_license(request: Request, body: LicenseInstallRequest) -> LicenseInstallView:
    """Verify a license blob, install it, and refresh the running daemon.

    Admin-only (D2). The blob is a **secret in transit**: it travels body →
    daemon → disk and stops there. It is never logged, never audited, never
    echoed back — including on the refusal path, where the temptation to print
    "this blob failed" is exactly the leak. The refusal carries the machine
    reason token instead, which is what an operator can actually act on.
    """
    require_role(request, TokenRole.admin)
    _require_idempotency_key(request, "license install")

    # One clock for the whole request: the temporal verdict and the rendered
    # `expires_in_s` must be two readings of the same instant, never two.
    now = _clock(request)
    verdict = verify_license(body.blob, trusted_keys=_trusted_keys(request), now=now)
    if verdict.state == STATE_INVALID or verdict.claims is None:
        # (D3) Nothing has been written and nothing will be: the caller's
        # working license — if any — is untouched. The reason is a fixed
        # machine token from INVALID_REASONS, safe in an envelope by
        # construction.
        raise NerditError(
            422,
            "license.invalid",
            f"The license did not verify ({verdict.reason}).",
            hint=(
                "Check that the whole one-line blob was pasted, and that this "
                "daemon version knows the signing key it was issued with."
            ),
            reason=verdict.reason,
        )

    claims = verdict.claims
    path = _license_path(request)
    # A durable NON-DB effect follows (the file write), so the idempotency claim
    # is pinned ahead of it — the documented `mark_request_side_effect`
    # contract, the /system/backup + link-claim precedent.
    mark_request_side_effect()
    async with _LICENSE_MUTATION_LOCK:
        try:
            # Blocking file I/O (atomic replace + fsyncs): off the loop.
            await asyncio.to_thread(install_license_file, path, body.blob)
        except LicenseError as exc:
            # The path belongs in the daemon log (the identity.py precedent) and
            # nowhere else: the envelope stays path-free.
            logger.error("License install failed: %s", exc)
            raise NerditError(
                500,
                "license.write_failed",
                "The license could not be written to the daemon's data directory.",
                hint="See the daemon log for the failing path, then fix permissions and retry.",
            ) from exc
        # (D5) Refresh the holder IN PLACE — every already-captured reference
        # stays truthful, and doctor/capabilities need no restart.
        _holder(request).replace(verdict)

    state = verdict.state
    request.state.audit_target = claims.lid
    # (D6) lid/plan/state ONLY. Never the blob, never customer_id: this row is
    # durable and read by more eyes than the admin who wrote it.
    request.state.audit_params = audit_params(
        {"lid": claims.lid, "plan": claims.plan, "state": state}
    )
    recorder = get_recorder()
    if recorder is not None:
        await recorder.record(
            "license.installed",
            data={"lid": claims.lid, "plan": claims.plan, "state": state},
        )
    # The plan token and the state are machine tokens; the customer id is not
    # logged (never a log line's business, D-LIC1 redacted-repr rationale).
    logger.info("License installed (lid %s, plan %s, state %s)", claims.lid, claims.plan, state)

    return LicenseInstallView(
        lid=claims.lid,
        plan=claims.plan,
        features=list(claims.features),
        state=state,
        expires_at=claims.expires_at.isoformat(),
        expires_in_s=seconds_until(claims.expires_at, now()),
        customer_id=claims.customer_id,
    )


@router.delete(
    "/license",
    response_model=LicenseRemoveView,
    operation_id="remove_license",
    status_code=200,
)
async def remove_license(request: Request) -> LicenseRemoveView:
    """Delete the installed license and clear the running daemon's state.

    Admin-only, and **idempotent**: removing when nothing is installed is a
    `200 {"removed": false}`, never a 404 — retries and cleanup scripts must
    be safe (the `DELETE /api/link` precedent).
    """
    require_role(request, TokenRole.admin)
    _require_idempotency_key(request, "license removal")

    path = _license_path(request)
    mark_request_side_effect()
    async with _LICENSE_MUTATION_LOCK:
        holder = _holder(request)
        claims = holder.claims
        try:
            removed = await asyncio.to_thread(remove_license_file, path)
        except LicenseError as exc:
            logger.error("License removal failed: %s", exc)
            raise NerditError(
                500,
                "license.remove_failed",
                "The installed license could not be removed.",
                hint="See the daemon log for the failing path, then fix permissions and retry.",
            ) from exc
        had_state = holder.installed
        holder.replace(None)

    # The lid of what was actually removed, when the daemon knew one: an invalid
    # (unparsed) license has none, and a file removed on a daemon that never
    # loaded it has none either. Never a path, never customer_id.
    lid = claims.lid if claims is not None else None
    params: dict[str, Any] = {"removed": removed, "had_state": had_state}
    if lid is not None:
        request.state.audit_target = lid
        params["lid"] = lid
    request.state.audit_params = audit_params(params)

    if removed or had_state:
        recorder = get_recorder()
        if recorder is not None:
            await recorder.record("license.removed", data={"lid": lid})
        logger.info("License removed (lid %s)", lid)
    return LicenseRemoveView(removed=removed)
