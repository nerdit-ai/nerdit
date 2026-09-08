"""Response schema for `GET /services/{ident}/wait`."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.daemon.schemas.service_diagnose import DiagnoseResponse
from nerdit.db.enums import ErrorClass, JobStatus


class ServiceWaitResponse(BaseModel):
    """Convergence outcome, always HTTP 200, including timeouts.

    Failed may coexist with status=running when a redeploy fails and the old image
    keeps serving. Diagnosis carries the diagnose bundle only for owners/admins;
    other authenticated callers receive diagnosis=null.
    """

    outcome: str = Field(description="One of 'converged' | 'failed' | 'timeout' | 'superseded'")
    service_name: str = Field(description="Stable service name")
    version: int | None = Field(
        default=None,
        description="The deploy version the wait was resolved against (None in status-only mode)",
    )
    phase: str | None = Field(
        default=None,
        description="The current last_deploy phase (None for rows with no last_deploy)",
    )
    status: JobStatus = Field(description="Live workload status at resolution")
    public_url: str | None = Field(
        default=None, description="Resolved public URL when the proxy is up (else None)"
    )
    public_urls: list[PublicUrlEntry] = Field(
        default_factory=list,
        description=(
            "(P26 D-P26-14) Every advertised URL with its kind and state — the "
            "hosted entry is the one a remote agent can open"
        ),
    )
    reason: str | None = Field(default=None, description="Failure reason when phase == failed")
    error_class: ErrorClass | None = Field(
        default=None, description="Coarse failure category on a failed outcome"
    )
    error_message: str | None = Field(
        default=None, description="Raw technical failure message on a failed outcome"
    )
    waited_s: float = Field(description="Wall-clock seconds spent waiting before resolution")
    diagnosis: DiagnoseResponse | None = Field(
        default=None,
        description=(
            "(P34) The full /diagnose bundle — remediation code + detail, crash "
            "forensics, health probe, binding readiness and the bounded log tail — "
            "folded in on outcome == 'failed' so a failure needs no second call. "
            "Null on every other outcome, and null on a failure the caller is not "
            "owner-or-admin for (the bundle carries owner-confidential log lines)"
        ),
    )
