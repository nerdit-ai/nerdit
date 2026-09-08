"""Response schema for `GET /services/{ident}/diagnose`."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from nerdit.db.enums import JobKind, JobStatus


class DiagnoseError(BaseModel):
    """Coarse failure classification block for `/diagnose`."""

    class_: str | None = Field(
        default=None, alias="class", description="ErrorClass value string, or null"
    )
    message: str | None = Field(default=None, description="Raw technical failure message, or null")

    model_config = {"populate_by_name": True}


class DiagnoseForensics(BaseModel):
    """Persisted crash forensics (names/values are numeric/bool/timestamp only)."""

    last_exit_code: int | None = Field(default=None, description="Last container exit code")
    oom_killed: bool = Field(default=False, description="Whether the last crash was an OOM kill")
    gpu_oom: bool = Field(
        default=False,
        description="Whether the last crash tail matched a CUDA/HIP out-of-memory error (P21 D2)",
    )
    priv_denied: bool = Field(
        default=False,
        description=(
            "Whether the last crash tail showed the image's entrypoint being refused a root "
            "privilege the sandbox drops ([containers].drop_all_caps + no_new_privileges)"
        ),
    )
    last_crash_at: str | None = Field(default=None, description="ISO timestamp of the last crash")


class DiagnoseRestarts(BaseModel):
    """Recomputed restart bookkeeping (no persisted 'next retry at')."""

    policy: str | None = Field(default=None, description="Restart policy")
    count: int = Field(default=0, description="Restarts within the current window")
    max_restarts: int = Field(description="Configured restart cap")
    window_seconds: int = Field(description="Configured restart window (seconds)")
    window_start: datetime | None = Field(default=None, description="Current window anchor")
    last_exit_at: datetime | None = Field(default=None, description="Last exit timestamp")
    backoff_s: float | None = Field(default=None, description="Current backoff delay (seconds)")
    next_retry_in_s: float | None = Field(
        default=None, description="Seconds until the next launch attempt (null when terminal)"
    )


class DiagnoseHealth(BaseModel):
    """Health-check spec + a FRESH probe taken at request time."""

    spec: dict[str, object] | None = Field(
        default=None, description="HTTP health-check spec (null = liveness-only)"
    )
    probe: dict[str, object] | None = Field(
        default=None, description="Fresh probe result (null when no live container)"
    )
    observations: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal observations about the health configuration "
            "(no paths, no values). Never a failure signal."
        ),
    )


class DiagnoseBindings(BaseModel):
    """Fresh, read-only `[ai.*]` binding resolution outcome (names/messages only)."""

    waiting: bool = Field(default=False, description="Whether any binding is not ready")
    messages: list[str] = Field(
        default_factory=list, description="Actionable wait messages (no secret values)"
    )


class DiagnoseBuild(BaseModel):
    """Deploy/build generation summary."""

    version: int | None = Field(default=None, description="Build version this generation targets")
    last_result: str = Field(description="'failed' | 'ok' | 'none'")
    reason: str | None = Field(
        default=None, description="Failure reason when last_result == failed"
    )


class DiagnoseRemediation(BaseModel):
    """The bounded remediation code + actionable detail (never free-form advice)."""

    code: str = Field(description="One of the locked RemediationCode values")
    detail: str = Field(description="Actionable, API-executable remediation detail")


class DiagnoseResponse(BaseModel):
    """Owner-or-admin failure bundle with forensics, probes, bindings and logs.

    Env/secret fields contain names only. Application logs, last_run.log_tail and
    last_dump.log_tail are owner-confidential output, not sanitized data: known
    secret values are masked, but transformed secrets or database row values may
    remain. Run and dump tails are capped at 200 lines and byte-bounded at capture.

    Run output is returned only here and by the run response, never via job_logs.
    Dump/restore tails appear only here; their response exposes the last line as
    a hint, and audit rows and durable events omit them. Dump commands contain no
    password: credentials travel in the environment.
    """

    service_name: str = Field(description="Stable service name")
    kind: JobKind = Field(description="Workload kind (service | model | database)")
    status: JobStatus = Field(description="Live workload status")
    desired_state: str | None = Field(default=None, description="Reconciler target")
    last_deploy: dict[str, object] | None = Field(
        default=None, description="Current deploy generation's phase object (null for pre-P13 rows)"
    )
    last_run: dict[str, object] | None = Field(
        default=None,
        description=(
            "Last one-off run's outcome record — run_id, verbatim command, "
            "exit_code/timed_out/oom_killed, timings, env override key NAMES, and the "
            "bounded scrubbed log_tail (null until a run has executed on this service)"
        ),
    )
    last_dump: dict[str, object] | None = Field(
        default=None,
        description=(
            "Last managed-database dump or restore — run_id, kind (dump|restore), "
            "the verbatim (secret-free) tool argv, exit_code/timed_out, the failure "
            "reason token, the tar basename, and the bounded scrubbed log_tail "
            "(null on every non-database row and until one has run)"
        ),
    )
    error: DiagnoseError = Field(description="Coarse failure classification")
    forensics: DiagnoseForensics = Field(description="Persisted crash forensics")
    restarts: DiagnoseRestarts = Field(description="Recomputed restart bookkeeping")
    health: DiagnoseHealth = Field(description="Health spec + fresh probe")
    bindings: DiagnoseBindings = Field(description="Fresh [ai.*] binding resolution outcome")
    build: DiagnoseBuild = Field(description="Deploy/build generation summary")
    injected_env_keys: list[str] = Field(
        default_factory=list,
        description="Env key NAMES injected at the last launch (persisted; never values)",
    )
    injected_env_keys_source: str = Field(
        description="'launch' (persisted from the last launch) | 'recomputed' (never launched)"
    )
    pending_env_keys: list[str] = Field(
        default_factory=list,
        description="Env key NAMES a launch would assemble now — post-launch drift is the set diff",
    )
    logs: list[dict[str, object]] = Field(
        default_factory=list, description="DB-backed log tail (oldest→newest; stream/line/ts)"
    )
    remediation: DiagnoseRemediation = Field(description="Bounded remediation code + detail")
