"""Read and apply daemon configuration with redaction and ETags.

Admin-only writes are validated, audited and idempotent. Section PUT supports
If-Match; atomic multi-section apply requires it. Real writes require an
Idempotency-Key; dry-runs write nothing and waive apply preconditions.
Restart-required changes are reported explicitly. Mounted under /api only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Query, Request, Response

from nerdit.config.redaction import redact_section
from nerdit.config.store import ConfigError, ConfigStore
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_role
from nerdit.daemon.errors import NerditError
from nerdit.db.models import (
    ConfigApplyRequest,
    ConfigApplyResponse,
    ConfigView,
    ConfigWriteResponse,
    TokenRole,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _store(request: Request) -> ConfigStore:
    store = getattr(request.app.state, "config_store", None)
    if store is None:
        raise NerditError(
            503,
            "config_unavailable",
            "The config store is not initialized.",
            hint="The daemon is still starting up.",
        )
    return store


def _as_nerdit_error(exc: ConfigError) -> NerditError:
    """Map a `ConfigError` to the structured error envelope."""
    extra: dict = {}
    if exc.diagnostics:
        extra["diagnostics"] = [d.model_dump() for d in exc.diagnostics]
    return NerditError(exc.status_code, exc.code, exc.message, hint=exc.hint, **extra)


@router.get("/config/daemon", response_model=list[ConfigView], operation_id="get_daemon_config")
async def get_daemon_config(request: Request, response: Response) -> list[ConfigView]:
    """Return every daemon config section (secrets redacted)."""
    require_role(request, TokenRole.readonly, TokenRole.submitter, TokenRole.admin)
    store = _store(request)
    etag = store.current_etag()
    response.headers["ETag"] = etag
    return [
        ConfigView(section=section, values=store.view_section(section), etag=etag)
        for section in store.known_sections()
    ]


@router.get(
    "/config/daemon/{section}",
    response_model=ConfigView,
    operation_id="get_daemon_config_section",
)
async def get_daemon_config_section(
    request: Request, response: Response, section: str
) -> ConfigView:
    """Return one daemon config section (secrets redacted)."""
    require_role(request, TokenRole.readonly, TokenRole.submitter, TokenRole.admin)
    store = _store(request)
    try:
        values = store.view_section(section)
    except ConfigError as exc:
        raise _as_nerdit_error(exc) from exc
    etag = store.current_etag()
    response.headers["ETag"] = etag
    return ConfigView(section=section, values=values, etag=etag)


@router.put(
    "/config/daemon/{section}",
    response_model=ConfigWriteResponse,
    operation_id="put_daemon_config_section",
)
async def put_daemon_config_section(
    request: Request,
    response: Response,
    section: str,
    body: dict = Body(..., description="New section values (full or partial)"),
    dry_run: bool = Query(False, description="Validate + diff without writing"),
) -> ConfigWriteResponse:
    """Validate and (unless `dry_run`) persist a daemon config section."""
    require_role(request, TokenRole.admin)
    # Redacted params recorded for audit even if the write later fails (the
    # AuditMiddleware reads this on the way back out).
    request.state.audit_params = redact_section(dict(body))

    store = _store(request)
    try:
        staged = store.stage(section, body)
    except ConfigError as exc:
        raise _as_nerdit_error(exc) from exc

    if_match = request.headers.get("If-Match")
    # RFC 7232: "*" = write-if-exists — the representation exists here, so proceed.
    if if_match is not None and if_match != "*" and if_match != store.current_etag():
        raise NerditError(
            409,
            "config.stale",
            "The config changed since you last read it.",
            hint="Re-read the section (GET) and retry with the new ETag.",
        )

    if dry_run:
        response.headers["ETag"] = store.current_etag()
        return ConfigWriteResponse(
            applied=False,
            diff=staged.diff,
            diagnostics=[],
            requires_restart=staged.requires_restart,
        )

    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            "A config write requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )

    new_etag = store.commit(staged)
    response.headers["ETag"] = new_etag
    # Keep a redacted audit trail of exactly which keys changed.
    request.state.audit_params = {
        **audit_params({"section": section, "dry_run": False}),
        "changed_keys": [entry.key for entry in staged.diff],
    }
    return ConfigWriteResponse(
        applied=True,
        diff=staged.diff,
        diagnostics=[],
        requires_restart=staged.requires_restart,
    )


@router.post(
    "/config/daemon/apply",
    response_model=ConfigApplyResponse,
    operation_id="apply_daemon_config",
    status_code=200,
)
async def apply_daemon_config(
    request: Request,
    response: Response,
    body: ConfigApplyRequest,
    dry_run: bool = Query(False, description="Validate + diff without writing"),
) -> ConfigApplyResponse:
    """Declaratively apply a multi-section daemon config document.

    Desired state for the *mentioned* sections only — sections absent from the
    document are untouched (whole-file replace is deliberately not a mode: it
    would clobber `auth_token` and the unknown/future sections the store
    preserves). All sections are staged against one snapshot and committed in
    a single atomic write; validation is all-or-nothing, report-everything.

    A real apply requires **both** `If-Match` (apply is an "I know the whole
    state" operation) and an `Idempotency-Key`; `?dry_run=true` is exempt
    from both. Re-applying the same document is a no-op (`changed: false`,
    same ETag, no write).
    """
    require_role(request, TokenRole.admin)
    # Redacted params recorded for audit even if the apply later fails (the
    # AuditMiddleware reads this on the way back out).
    request.state.audit_params = {
        section: redact_section(dict(values)) for section, values in body.sections.items()
    }

    store = _store(request)
    try:
        staged = store.stage_many({s: dict(v) for s, v in body.sections.items()})
    except ConfigError as exc:
        raise _as_nerdit_error(exc) from exc

    if dry_run:
        etag = store.current_etag()
        response.headers["ETag"] = etag
        return ConfigApplyResponse(
            applied=False,
            changed=staged.changed,
            etag=etag,
            requires_restart=staged.requires_restart,
            restart_keys=staged.restart_keys,
            diff=staged.diff,
            diagnostics=[],
        )

    if_match = request.headers.get("If-Match")
    if not if_match:
        raise NerditError(
            400,
            "config.if_match_required",
            "A config apply requires an If-Match header.",
            hint="GET /api/config/daemon for the current ETag and retry with If-Match.",
        )
    current_etag = store.current_etag()
    # RFC 7232: "*" = write-if-exists — the representation exists here, so proceed.
    if if_match != "*" and if_match != current_etag:
        raise NerditError(
            409,
            "config.stale",
            "The config changed since you last read it.",
            hint="Re-read the config (GET) and retry with the new ETag.",
            current_etag=current_etag,
        )

    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            "A config apply requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the apply is safe to retry.",
        )

    new_etag = store.commit(staged)
    response.headers["ETag"] = new_etag
    # Keep a redacted audit trail of exactly which keys changed, per section.
    changed_keys: dict[str, list[str]] = {}
    for entry in staged.diff:
        changed_keys.setdefault(entry.section, []).append(entry.key)
    request.state.audit_params = {
        **audit_params({"dry_run": False}),
        "changed_keys": changed_keys,
        "requires_restart": staged.requires_restart,
    }
    return ConfigApplyResponse(
        applied=True,
        changed=staged.changed,
        etag=new_etag,
        requires_restart=staged.requires_restart,
        restart_keys=staged.restart_keys,
        diff=staged.diff,
        diagnostics=[],
    )
