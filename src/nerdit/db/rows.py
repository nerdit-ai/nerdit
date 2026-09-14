"""SQLite row and projection models, including health specs stored inline on jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

from nerdit.db.enums import (
    ErrorClass,
    GpuDiscoveryBackend,
    GpuStatus,
    GpuVendor,
    JobKind,
    JobStatus,
    LogStream,
    TokenRole,
)
from nerdit.utils.ids import generate_id

# --- Core Models ---


class Gpu(BaseModel):
    """Representation of a GPU detected on the host."""

    id: str = Field(description="Stable Nerdit inventory ID")
    name: str = Field(description="GPU product name")
    memory_mb: int = Field(description="Total GPU memory in megabytes")
    compute_cap: str | None = Field(
        default=None, description="CUDA compute capability (e.g., '9.0')"
    )
    vendor: GpuVendor = Field(default=GpuVendor.nvidia, description="GPU hardware vendor")
    device_index: int | None = Field(
        default=None, ge=0, description="Zero-based device index within the vendor backend"
    )
    runtime_id: str | None = Field(
        default=None, description="Private device selector passed to the container runtime"
    )
    discovery_backend: GpuDiscoveryBackend = Field(
        default=GpuDiscoveryBackend.nvml,
        description="Backend that reported this device",
    )
    schedulable: bool = Field(
        default=True, description="Whether the scheduler may allocate this device"
    )
    status: GpuStatus = Field(default=GpuStatus.idle, description="Current allocation status")


class Job(BaseModel):
    """A compute job managed by the scheduler — central model of Nerdit."""

    id: str = Field(
        default_factory=generate_id,
        description="Unique 12-character alphanumeric job identifier",
    )
    name: str | None = Field(default=None, description="Optional human-readable job name")
    script_path: str | None = Field(
        default=None, description="Path to the Python script to execute (None in command mode)"
    )
    gpu_count: int = Field(
        default=1,
        ge=0,
        description="Number of GPUs required (0 is allowed)",
    )
    status: JobStatus = Field(default=JobStatus.pending, description="Current job lifecycle state")
    kind: JobKind = Field(
        default=JobKind.batch,
        description="Workload kind: service | model | database (batch is legacy)",
    )
    container_id: str | None = Field(default=None, description="Docker container ID once launched")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the job was created",
    )
    started_at: datetime | None = Field(default=None, description="Timestamp when execution began")
    finished_at: datetime | None = Field(default=None, description="Timestamp when execution ended")
    exit_code: int | None = Field(
        default=None, description="Container exit code (137=OOM, 0=success)"
    )
    config: str | None = Field(
        default=None, description="JSON blob for advanced options (e.g., volume mounts)"
    )
    error_class: ErrorClass | None = Field(
        default=None, description="Dashboard-friendly failure category"
    )
    error_message: str | None = Field(default=None, description="Raw technical failure message")
    submitted_via: str = Field(default="cli", description="Always 'cli' today; see the API schema")
    submitted_by_token: str | None = Field(
        default=None,
        description="ID of the API token that submitted this job (None = legacy/local). "
        "Set from the principal, never from the request body.",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Idempotency-Key header that created this job, persisted for traceability. "
        "Set from the request header, never from the body.",
    )
    # --- Service-mode columns (P2; NULL/0 on legacy batch rows) ---
    desired_state: str | None = Field(
        default=None,
        description="Reconciler target for a service ('running' | 'stopped')",
    )
    restart_policy: str | None = Field(
        default=None,
        description="Service restart policy ('no' | 'on-failure' | 'always')",
    )
    restart_count: int = Field(
        default=0, ge=0, description="In-place restart counter for a service (rate-window scoped)"
    )
    health_check: dict[str, object] | None = Field(
        default=None,
        description="Service HTTP health-check spec (JSON); None = liveness-only",
    )
    service_name: str | None = Field(
        default=None,
        description="Unique service name (one stable row per service)",
    )
    last_exit_at: datetime | None = Field(
        default=None, description="Timestamp of the service container's last exit (backoff anchor)"
    )
    restart_window_start: datetime | None = Field(
        default=None, description="Start of the current restart rate-limit window"
    )


class ServiceEndpoint(BaseModel):
    """A durable host-port reservation for a service.

    Keyed by `service_name` (one endpoint per service) with a repointable
    `job_id` so the reservation survives restarts and daemon reboots — the
    `host_port` is reused on relaunch for URL stability. `route` is the
    proxy route contract: `NULL` = unrouted (proxy disabled, or a
    `kind=model` row — models are never routed), `"/name"` = a path-based
    route, `""` (empty string) = a subdomain route. Consumers must test
    `route is not None`, never truthiness — an empty string is a valid,
    routed state.
    """

    id: str = Field(
        default_factory=generate_id, description="Unique 12-character endpoint identifier"
    )
    service_name: str = Field(description="Name of the service that owns this endpoint")
    job_id: str | None = Field(
        default=None, description="ID of the jobs row currently bound to this endpoint"
    )
    container_port: int = Field(ge=1, le=65535, description="Port exposed inside the container")
    host_port: int = Field(ge=1, le=65535, description="Stable host port published on 127.0.0.1")
    protocol: str = Field(default="tcp", description="Transport protocol (tcp)")
    route: str | None = Field(
        default=None,
        description=(
            "Proxy route contract: NULL = unrouted (proxy off, or a kind=model "
            'row — models are never routed); "/name" = path route; "" (empty '
            "string) = subdomain route. Test `route is not None`, never "
            "truthiness — empty string is a valid, routed state."
        ),
    )
    active_host_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "(P24b / D-P24-4b) Port the currently serving container is actually "
            "bound to; NULL = same as host_port. Non-NULL only during a "
            "health-gated cutover window and for the healthy generation it "
            "promotes, until the next ordinary launch clears it. Consumers read "
            "`live_port`, never this field directly."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the endpoint was reserved",
    )

    @property
    def live_port(self) -> int:
        """Return the active container port for probes and direct reads, falling back to the
        stable reservation.
        """
        return self.active_host_port or self.host_port


class ServiceShare(BaseModel):
    """Hosted-share intent keyed by service name; absence means unshared.

    The daemon controls exposure, the cloud controls audience. Streams reread this
    row to fail closed. Partial service-name uniqueness prevents a foreign key.
    """

    service_name: str = Field(description="Name of the shared service")
    access: Literal["private", "public"] = Field(
        default="private",
        description=(
            "'private' = only signed-in owners of this node at the cloud edge; "
            "'public' = world-reachable (entitlement + consent gated)"
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the share was first created (preserved across access changes)",
    )

    @field_validator("created_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        """Attach UTC to naive stored timestamps so disk and live values serialize identically."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class ServiceDomain(BaseModel):
    """A domain claimed by exactly one service, with certificate policy stored alongside it.

    The case-insensitive primary key prevents takeover. This table is authoritative;
    no project-config mirror may recreate removed names. Partial service-name
    uniqueness prevents a foreign key, so checked service deletion cascades transactionally.
    """

    domain: str = Field(description="Case-folded bare DNS name, no trailing dot")
    service_name: str = Field(description="Name of the service the domain routes to")
    acme: bool = Field(
        default=False,
        description="Dormant in this release: a public certificate was requested (WP2 acts on it)",
    )
    kind: Literal["domain"] = Field(
        default="domain", description="Row discriminator; only 'domain' exists today"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the domain was bound to the service",
    )

    @field_validator("created_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        """Attach UTC to naive stored timestamps so disk and live values serialize identically."""
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class ActiveServiceRoute(BaseModel):
    """A routable service and its effective loopback upstream for proxy reconciliation.

    Derive candidates from workload state, not route presence, so enabling the
    proxy and adopting services both create routes. Reassert the desired dial each tick.
    """

    service_name: str
    host_port: int
    status: str
    route: str | None = None
    edge_auth: dict[str, Any] | None = None
    """(P25 D-P25-8) The row's ``config['edge_auth']`` blob, verbatim.

    ``None`` means the row declares no edge auth (the route serves openly —
    every pre-P25 row). A declared blob is carried through UNPARSED so
    :func:`~nerdit.core.proxy.edgeauth.load_edge_auth` owns the tri-state: the
    query must never decide that a malformed declaration means "no auth", which
    would publish a route its owner asked to protect.
    """


class GpuMetrics(BaseModel):
    """Live GPU metrics collected by the resource monitor."""

    gpu_id: str = Field(description="Nerdit inventory ID this metrics snapshot belongs to")
    utilization_percent: int | None = Field(
        default=None, description="GPU compute utilization (0-100%)"
    )
    memory_used_mb: int | None = Field(
        default=None, description="Currently used GPU memory in megabytes"
    )
    memory_total_mb: int | None = Field(default=None, description="Total GPU memory in megabytes")
    temperature_c: int | None = Field(default=None, description="GPU temperature in Celsius")


class ApiToken(BaseModel):
    """A scoped API token.

    The raw token value is shown exactly once at creation; only its SHA-256
    `token_hash` is persisted. Quotas (`max_gpus`, `max_concurrent_jobs`)
    are enforced atomically at job submission. `revoked` is a soft-delete.
    """

    id: str = Field(default_factory=generate_id, description="Unique 12-character token identifier")
    name: str = Field(description="Human-readable label for the token")
    role: TokenRole = Field(
        default=TokenRole.submitter, description="Authorization role granted by this token"
    )
    token_hash: str = Field(description="SHA-256 hex digest of the raw token (never the plaintext)")
    max_gpus: int | None = Field(
        default=None,
        description="Cumulative GPU cap across this token's active jobs (None = uncapped)",
    )
    max_concurrent_jobs: int | None = Field(
        default=None, description="Concurrent active-job cap for this token (None = uncapped)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the token was created",
    )
    last_used_at: datetime | None = Field(
        default=None, description="Timestamp the token was last presented (throttled update)"
    )
    revoked: bool = Field(default=False, description="Whether the token has been revoked")
    expires_at: datetime | None = Field(
        default=None,
        description="Absolute expiry instant (UTC); None = never expires (P25 D-P25-1)",
    )
    scope_services: list[str] | None = Field(
        default=None,
        description="Service names this token may write to; None = unscoped (P25 D-P25-3)",
    )


class LogEntry(BaseModel):
    """API response model for a single log entry."""

    id: int = Field(description="Auto-incremented log entry ID")
    stream: LogStream = Field(
        description="Source stream (stdout, stderr, system, or the P21 crash-tail capture)"
    )
    message: str = Field(description="Log line content")
    timestamp: datetime = Field(description="Timestamp when the log line was recorded")


class HealthCheck(BaseModel):
    """HTTP health policy stored as JSON on the job row.

    Consecutive non-2xx responses mark a service degraded but leave it running;
    only dead containers restart. A later 2xx restores running. Omit for liveness-only supervision.
    """

    path: str = Field(default="/", description="HTTP path probed on the service")
    type: Literal["http", "tcp"] = Field(
        default="http",
        description=(
            "Probe kind (P14 WP-C1): 'http' GETs ``path`` and requires a 2xx; "
            "'tcp' only establishes a connection to the published port (``path`` "
            "is ignored). A tcp probe suits non-HTTP servers (databases, raw TCP)."
        ),
    )
    timeout_s: float = Field(
        default=2.0, gt=0, le=30, description="Per-probe HTTP timeout in seconds"
    )
    unhealthy_threshold: int = Field(
        default=3, ge=1, le=10, description="Consecutive failures before marking degraded"
    )
    start_period_s: float = Field(
        default=0.0, ge=0, description="Grace period after start before probing (seconds)"
    )


class AuditLogEntry(BaseModel):
    """One append-only audit record.

    `params_redacted` is the route-attached parameter dict after the secret
    denylist has masked sensitive keys; `result` is one of `ok`/`error`/
    `denied`/`replay`. Rows are immutable.
    """

    id: int
    ts: datetime | None = None
    principal_id: str | None = None
    principal_role: str | None = None
    action: str
    target_type: str | None = None
    target_id: str | None = None
    params_redacted: object | None = None
    result: str | None = None
    status_code: int | None = None
    request_id: str | None = None
    idempotency_key: str | None = None


class Event(BaseModel):
    """Durable, advisory event readable by any authenticated principal and sent to webhooks.

    Jobs remain authoritative. reason/data contain machine tokens, numbers, booleans
    and key names only: never free text, errors or secret values. Normalize stored
    naive UTC timestamps to aware UTC so replay and live streams serialize identically.
    """

    id: int
    ts: datetime | None = None
    type: str
    kind: str | None = None
    service_name: str | None = None
    reason: str | None = None
    build_version: int | None = None
    data: dict[str, object] | None = None

    @field_validator("ts")
    @classmethod
    def _aware_utc(cls, value: datetime | None) -> datetime | None:
        """Attach UTC to naive stored timestamps so disk and live values serialize identically."""
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    @field_serializer("ts")
    def _serialize_ts(self, value: datetime | None) -> str | None:
        """Emit the aware ISO-8601 form (`…+00:00`), never a bare naive string."""
        if value is None:
            return None
        if value.tzinfo is None:  # pragma: no cover — the validator already ran
            value = value.replace(tzinfo=UTC)
        return value.isoformat()


class EventPage(BaseModel):
    """A bounded event page with a directional cursor.

    For descending browse, next_cursor is the last row's ID; for ascending replay,
    it is the largest ID. None means no further page.
    """

    items: list[Event]
    next_cursor: str | None = None


class IdempotencyRecord(BaseModel):
    """A request claim scoped by principal and key, with method/path reuse checks.

    State is in_progress, completed after 2xx buffering, or interrupted after a
    cancelled committed/uncertain effect. Secret-returning routes store only status
    and resource ID, never response bodies.

    body_hash stores a bounded request digest, never the request. It is None for
    legacy, oversized, chunked, multipart or secret-carrying requests; compare only
    when both hashes exist.
    """

    principal_id: str
    idem_key: str
    method: str
    path: str
    state: str = "in_progress"
    response_status: int | None = None
    response_body: str | None = None
    content_type: str | None = None
    resource_id: str | None = None
    body_hash: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
