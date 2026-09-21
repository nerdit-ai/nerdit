"""Resolve service identifiers and build consistent API projections.

Shared by routes and purge helpers; keep dependencies below the route layer
to avoid import cycles.
"""

from __future__ import annotations

from fastapi import Request

from nerdit.core.jobconfig import parse_job_config
from nerdit.core.proxy import public_url_for
from nerdit.daemon.auth import may_manage_job
from nerdit.daemon.errors import NerditError
from nerdit.daemon.views.hosted import EMPTY_HOSTED, HostedContext, public_urls_for
from nerdit.db.models import (
    MANAGED_KINDS,
    Job,
    ServiceEndpoint,
    ServiceEndpointView,
    ServiceResponse,
)

# Allowlisted keys of `config['source']` projected onto the response.
# A structural guarantee that no credential can leak even if a future ingress
# pollutes the provenance dict — only these keys are ever copied through.
_SOURCE_KEYS = ("type", "repo_url", "repo", "ref", "commit_sha", "subdir", "template_id")


async def _resolve_service(queries, ident: str) -> Job | None:
    """Resolve a path identifier (row id *or* service name) to a service row.

    Tries the row id first, but only accepts it when the row is a managed
    kind (`service`/`model`/`database`) — so a *batch* job id can never
    be acted on through `/services`. Falls back to the unique
    `service_name` lookup.
    """
    job = await queries.get_job(ident)
    if job is not None and job.kind in MANAGED_KINDS:
        return job
    return await queries.get_service_by_name(ident)


def _not_found(ident: str) -> NerditError:
    """Build the structured 404 for an unknown service identifier."""
    return NerditError(
        404,
        "not_found",
        f"No service '{ident}'.",
        hint="List services with `nerdit services list` to find a valid name or id.",
    )


def _run_in_progress_error(name: str, verb: str, *, hint: str | None = None) -> NerditError:
    """Build the one `service.run_in_progress` 409 shape for deploy-surface guards.

    One factory per code, not one taking the code: the message stem and the
    default hint differ per code, so a code parameter would push both strings
    to every call site — exactly the divergence this exists to prevent.
    """
    return NerditError(
        409,
        "service.run_in_progress",
        f"Service '{name}' has an active run or release; cannot {verb}.",
        hint=hint or "Wait for it to finish (bounded by [services].release_timeout_s) and retry.",
    )


def _cutover_in_progress_error(name: str, verb: str, *, hint: str | None = None) -> NerditError:
    """Build the one `service.cutover_in_progress` 409 shape for deploy-surface guards."""
    return NerditError(
        409,
        "service.cutover_in_progress",
        f"Service '{name}' has a zero-downtime cutover in progress; cannot {verb}.",
        hint=hint
        or "Wait for it to settle (bounded by [services].cutover_verify_timeout_s) and retry.",
    )


def _endpoint_view(
    request: Request,
    endpoint: ServiceEndpoint | None,
    *,
    hosted: HostedContext = EMPTY_HOSTED,
) -> ServiceEndpointView | None:
    """Project an endpoint with proxy and hosted URLs.

    Use public_url_for; return no proxy URL when proxying is unavailable. Hosted
    is the caller's shared request snapshot. Production callers must supply it to
    include shares; missing state remains safe for minimal applications.
    """
    if endpoint is None:
        return None
    settings = getattr(request.app.state, "settings", None)
    proxy = getattr(settings, "proxy", None) if settings else None
    hostname = getattr(request.app.state, "hostname", None)
    # Gate on the LIVE proxy state, not the config flag: if the proxy is enabled
    # but Caddy never came up (e.g. the :443 bind failed), the route is persisted
    # but no URL actually resolves — advertising it would be a dead link. The
    # `available` flag tracks exactly this.
    proxy_mgr = getattr(request.app.state, "proxy_manager", None)
    available = bool(getattr(proxy_mgr, "available", False))
    public: str | None = None
    if available and proxy is not None and hostname:
        public = public_url_for(
            endpoint.service_name,
            endpoint.route,
            mode=proxy.mode,
            hostname=hostname,
            base_domain=proxy.base_domain,
            scheme=proxy.scheme,
            https_port=proxy.https_port,
            public_port=proxy.public_port,
        )
    return ServiceEndpointView.from_endpoint(
        endpoint,
        public_url=public,
        public_urls=public_urls_for(hosted, endpoint.service_name, public),
    )


def _service_response(
    request: Request,
    job: Job,
    gpu_ids: list[str],
    endpoint: ServiceEndpoint | None,
    *,
    data_dir_bytes: int | None = None,
    hosted: HostedContext = EMPTY_HOSTED,
) -> ServiceResponse:
    """Project a service consistently across reads, writes and deployments.

    Tolerate malformed rollback/version config. data_dir_bytes is precomputed only
    for detail reads, never lists. Reuse the caller's hosted snapshot for all rows.
    Manageable depends on the current principal: never cache this projection by row.
    """
    cfg = parse_job_config(job)
    raw_version = cfg.get("build_version")
    raw_source = cfg.get("source")
    source = (
        {k: raw_source[k] for k in _SOURCE_KEYS if k in raw_source}
        if isinstance(raw_source, dict)
        else None
    )
    return ServiceResponse(
        id=job.id,
        name=job.service_name or job.name or job.id,
        project=job.project,
        project_id=job.project_id,
        service=job.service,
        status=job.status,
        desired_state=job.desired_state,
        kind=job.kind,
        image=cfg.get("image"),
        gpu_count=job.gpu_count,
        gpu_ids=gpu_ids,
        restart_policy=job.restart_policy,
        restart_count=job.restart_count,
        health_check=job.health_check,
        container_id=job.container_id,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        error_class=job.error_class,
        error_message=job.error_message,
        submitted_via=job.submitted_via,
        manageable=may_manage_job(request, job),
        endpoint=_endpoint_view(request, endpoint, hosted=hosted),
        rollback_available=bool(cfg.get("previous_image")),
        build_version=raw_version if type(raw_version) is int else None,
        last_deploy=cfg.get("last_deploy") if isinstance(cfg.get("last_deploy"), dict) else None,
        data_dir_bytes=data_dir_bytes,
        source=source or None,
    )
