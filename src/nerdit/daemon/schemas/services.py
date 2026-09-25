"""Request/response schemas for the `/services` surface."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.daemon.schemas.exposure import PublicUrlEntry
from nerdit.db.enums import ErrorClass, GpuVendor, JobKind, JobStatus
from nerdit.db.rows import HealthCheck, ServiceEndpoint
from nerdit.utils.names import DNS_LABEL_PATTERN


class ServiceCreateRequest(StrictRequestModel):
    """Register an existing local image as a service.

    Name is a stable DNS label. GPUs may be zero and are shared when allocated.
    """

    name: str = Field(
        pattern=DNS_LABEL_PATTERN,
        description="DNS-label service name (stable identity, lowercase, 1-63 chars)",
    )
    image: str = Field(description="Existing local image to run (required — register-only in P2)")
    port: int = Field(
        default=8000, ge=1, le=65535, description="Container port to publish on the host"
    )
    gpus: int = Field(
        default=0, ge=0, description="GPUs the service needs (0 = none; shared, non-exclusive)"
    )
    restart_policy: Literal["no", "on-failure", "always"] = Field(
        default="always", description="Restart policy applied when the container exits"
    )
    command: str | None = Field(
        default=None, description="Override the image CMD with a shell command"
    )
    script_path: str | None = Field(
        default=None, description="Run a script under nerdit-runtime instead of the image CMD"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables injected into the container"
    )
    vendor: GpuVendor | None = Field(
        default=None, description="Required GPU vendor (omit for any available vendor)"
    )
    health_check: HealthCheck | None = Field(
        default=None, description="Optional HTTP health-check (omit = liveness-only)"
    )


class ServiceEndpointView(BaseModel):
    """Public projection of a service's durable host-port reservation."""

    container_port: int = Field(description="Port exposed inside the container")
    host_port: int = Field(description="Stable host port published on 127.0.0.1")
    effective_host_port: int = Field(
        description=(
            "(P24b / D-P24-4b) Port the live container is ACTUALLY bound to = "
            "COALESCE(active_host_port, host_port). Equal to host_port on every "
            "ordinary row; differs only for the length of a cutover-promoted "
            "generation. Direct-loopback consumers must dial this, not host_port."
        )
    )
    protocol: str = Field(default="tcp", description="Transport protocol")
    route: str | None = Field(
        default=None,
        description=(
            "Proxy route contract: NULL = unrouted (proxy off, or a kind=model "
            'row — models are never routed); "/name" = path route; "" (empty '
            "string) = subdomain route. Test `route is not None`, never "
            "truthiness — empty string is a valid, routed state."
        ),
    )
    url: str | None = Field(default=None, description="Loopback URL for the published port")
    public_url: str | None = Field(
        default=None,
        description="Resolved public HTTPS URL via the proxy (P3); NULL when not routed",
    )
    public_urls: list[PublicUrlEntry] = Field(
        default_factory=list,
        description=(
            "(P26 D-P26-14) Every advertised URL with its kind and state; "
            "`public_url` stays the default-kind scalar and keeps its meaning"
        ),
    )

    @classmethod
    def from_endpoint(
        cls,
        endpoint: ServiceEndpoint,
        *,
        public_url: str | None = None,
        public_urls: list[PublicUrlEntry] | None = None,
    ) -> ServiceEndpointView:
        """Project a stored endpoint using its live port.

        The loopback URL uses live_port, including after cutover; host_port remains the
        stable reservation. Callers supply public_url from public_url_for (None when
        proxying is unavailable) and request-scoped public_urls, which defaults to [].
        """
        return cls(
            container_port=endpoint.container_port,
            host_port=endpoint.host_port,
            effective_host_port=endpoint.live_port,
            protocol=endpoint.protocol,
            route=endpoint.route,
            url=f"http://127.0.0.1:{endpoint.live_port}",
            public_url=public_url,
            public_urls=public_urls or [],
        )


class ServiceResponse(BaseModel):
    """API response for a service — a dedicated projection of the `jobs` row.

    A service is a desired-state workload, so only the desired-state fields are
    projected. `endpoint` carries the stable host port / URL once the
    reconciler has published it.
    """

    id: str = Field(description="Unique 12-character row identifier")
    name: str = Field(description="Stable service name (identity)")
    # (P40b) The project triple beside the untouched label — the stated
    # deviation from plan §2: the dashboard route
    # `/projects/:name/services/:service` needs the parts, and a qualified
    # string would only move the parsing to the client (D-P40-6).
    project: str | None = Field(
        default=None,
        description=(
            "Name of the project this service belongs to (P40b); null for models, "
            "databases and rows predating the project columns"
        ),
    )
    project_id: str | None = Field(
        default=None, description="The owning project's id (`prj_…`); null when `project` is"
    )
    service: str | None = Field(
        default=None,
        description="Service name within the project ('web' for a bare project name)",
    )
    status: JobStatus = Field(description="Current lifecycle state")
    desired_state: str | None = Field(
        default=None, description="Reconciler target ('running' | 'stopped')"
    )
    kind: JobKind = Field(
        default=JobKind.service, description="Workload kind (service | model | database)"
    )
    image: str | None = Field(default=None, description="Image the service runs")
    gpu_count: int = Field(description="GPUs requested by the service (0 = none)")
    gpu_ids: list[str] = Field(
        default_factory=list, description="Nerdit inventory IDs allocated to the service"
    )
    restart_policy: str | None = Field(
        default=None, description="Restart policy ('no' | 'on-failure' | 'always')"
    )
    restart_count: int = Field(default=0, description="Restarts within the current rate window")
    health_check: dict[str, object] | None = Field(
        default=None, description="HTTP health-check spec (None = liveness-only)"
    )
    container_id: str | None = Field(default=None, description="Docker container ID once launched")
    created_at: datetime = Field(description="Timestamp when the service was created")
    started_at: datetime | None = Field(default=None, description="Timestamp the container started")
    finished_at: datetime | None = Field(default=None, description="Timestamp of last exit")
    exit_code: int | None = Field(default=None, description="Last container exit code")
    error_class: ErrorClass | None = Field(default=None, description="Coarse failure category")
    error_message: str | None = Field(default=None, description="Raw technical failure message")
    submitted_via: str = Field(
        default="cli",
        description=(
            "Always 'cli' today: no submit path records its origin, so a deploy over "
            "the dashboard or MCP carries the same label. Not a signal."
        ),
    )
    # (NC-0, cloud D-NC5) REQUIRED, with no default — it joins `id`/`name`/
    # `status`/`gpu_count`/`created_at` as a field a construction site
    # cannot omit, and it is the only *derived* one in that set: every other
    # projected field here defaults to a benign "nothing recorded" instead.
    # Every ServiceResponse in the tree is built by the single mapper
    # `views.service._service_response`, which has the request (hence the
    # principal) and the row (hence the owner), so there is no construction
    # site that *could* not compute it. Given that, a default would only ever
    # fire for a future site that forgot — and the two candidate defaults are
    # both wrong there: `True` claims the caller may act when nobody checked
    # (the one failure mode that matters — a console showing live buttons for
    # services it cannot touch), and `False` silently strips the owner's own
    # actions with no signal that anything is broken. A missing-field
    # ValidationError at the new call site is louder than either. Pinned by
    # `test_manageable_has_no_default_so_a_future_call_site_cannot_omit_it`,
    # because a `default=` added here would otherwise break nothing.
    manageable: bool = Field(
        description=(
            "Whether THIS caller clears the OWNER gate on THIS service — the "
            "admin-or-(owner-and-in-scope) verdict `require_owner_or_admin` "
            "enforces, answered in advance. False for rows submitted by another "
            "token and for NULL-owner (local/legacy) rows, which are admin-only. "
            "Caller-relative: two principals reading the same service see "
            "different values. It is the owner verdict ONLY, never a per-verb "
            "promise: restart and stop need nothing further, but redeploy also "
            "needs a recorded git source and [git].enabled (else 409 "
            "deploy.no_source / deploy.git_disabled) and rollback needs a "
            "previous image (else 409 — `rollback_available` is that field). "
            "Both can also 409 transiently on a live release or cutover. So a "
            "row this caller owns is manageable=true and may still refuse "
            "redeploy: read it as 'not forbidden', not as 'this button works'"
        ),
    )
    endpoint: ServiceEndpointView | None = Field(
        default=None, description="Published host-port reservation (None until launched)"
    )
    rollback_available: bool = Field(
        default=False,
        description=(
            "Whether a rollback target exists (a 'previous_image' recorded by a "
            "redeploy); when False, POST /deploy/{name}/rollback returns 409"
        ),
    )
    build_version: int | None = Field(
        default=None,
        description="Deploy build version currently serving (None for non-deployed services)",
    )
    last_deploy: dict[str, object] | None = Field(
        default=None,
        description=(
            "The current deploy generation's phase object (version/action/phase/"
            "image/started_at/updated_at + reason/error_class/error_message on "
            "failure; P33: repo/ref/sha — the canonical 'host/owner/repo', the "
            "cloned ref and the sha resolved at clone time, null for non-git "
            "generations — and remediation_code, the crash-loop remediation rule "
            "stamped when the generation settles failed, else null); None for "
            "pre-P13 rows and services created via POST /services"
        ),
    )
    data_dir_bytes: int | None = Field(
        default=None,
        description=(
            "On-disk size of the service's named-volume data dir in bytes (P14 "
            "WP-A1); computed on the detail route only — None on the list route"
        ),
    )
    source: dict[str, object] | None = Field(
        default=None,
        description=(
            "Deploy provenance (P11.5 config['source']): {type: 'zip'|'git', "
            "repo_url?, repo? (P33 canonical 'host/owner/repo'), ref?, commit_sha?, "
            "subdir?, template_id?}. None for "
            "services created via POST /services or rows predating P11.5. "
            "Never carries credentials or token_ref."
        ),
    )


class ServiceListPage(BaseModel):
    """Cursor-paginated page of services for the bounded `GET /services` read."""

    items: list[ServiceResponse]
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page; null when the list is exhausted",
    )
