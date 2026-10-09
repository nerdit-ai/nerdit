"""The machine AI gateway's control surface: aliases, virtual keys, usage.

The gateway itself is a separate listener (`core/ai_gateway`); these routes
only read and write its SQLite rows, so they answer while the gateway is off
(an agent configures aliases before an operator enables `[ai_gateway]`).

* Reads (any authenticated principal): the state view, the aliases, the key
  metadata and the usage aggregates. No view carries a secret value, a
  virtual key or its hash: an alias shows its `api_key_ref` (a name).
* Writes take submitter or admin (0.8.3, D-P31-5): an alias only references a
  shared secret a submitter's own deploy already receives, and a hosted box's
  tunnel principal is submitter-capped. Daemon config stays admin. `PUT` needs
  an `Idempotency-Key`. Audit params carry the
  alias, provider, model, the base_url HOST and the ref name, never a value.
* `base_url` is the anti-SSRF gate: https to a host that resolves to public
  addresses only, or plain http to loopback / the models bridge (a local
  OpenAI-compatible server). Resolved once, at write time.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from nerdit.config.project import AI_GATEWAY_ALIAS_HINT, AI_GATEWAY_ALIAS_RE, SECRET_REF_RE
from nerdit.core.jobconfig import parse_job_config
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_owner_or_admin, require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.enums import JobKind
from nerdit.db.models import TokenRole
from nerdit.db.rows import AiGatewayKey, AiGatewayRoute, AiGatewayUsage

router = APIRouter()


class AiGatewayView(BaseModel):
    """The gateway's state: whether it listens, where apps reach it, how many aliases."""

    enabled: bool
    listening: bool
    host: str
    port: int
    base_url: str
    routes: int


class AiRouteSetRequest(StrictRequestModel):
    """Create or replace one alias. `api_key_ref` is a reference, never a key."""

    provider: Literal["api", "ollama"]
    model: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[\x21-\x7e]+$",
        description="The provider's model name the alias maps to (an ollama model for "
        "provider 'ollama').",
    )
    base_url: str | None = Field(
        default=None,
        max_length=2048,
        description="Provider 'api' only: the OpenAI-compatible base URL, usually ending "
        "in /v1. https, or http to loopback / the models bridge.",
    )
    api_key_ref: str | None = Field(
        default=None,
        max_length=256,
        description="Provider 'api' only: a ${secrets.shared.KEY} reference to the "
        "provider key, never the key itself.",
    )


class AiRouteWriteResponse(BaseModel):
    """The stored alias, whether it was created, and whether the gateway listens."""

    route: AiGatewayRoute
    created: bool
    enabled: bool


class AiRouteList(BaseModel):
    routes: list[AiGatewayRoute]
    enabled: bool


class AiKeyList(BaseModel):
    """Virtual-key metadata: which services hold a key (never the key)."""

    keys: list[AiGatewayKey]


class AiKeyRevokeResponse(BaseModel):
    service: str
    revoked: int


class AiUsageList(BaseModel):
    days: int
    usage: list[AiGatewayUsage]


def _view(request: Request, routes: int) -> AiGatewayView:
    settings = request.app.state.settings
    host = settings.models.bridge_advertise_host
    port = settings.ai_gateway.port
    return AiGatewayView(
        enabled=settings.ai_gateway.enabled,
        listening=_enabled(request),
        host=host,
        port=port,
        base_url=f"http://{host}:{port}/v1",
        routes=routes,
    )


def _enabled(request: Request) -> bool:
    """Enabled in config AND actually listening (a bind failure leaves it off)."""
    enabled = bool(request.app.state.settings.ai_gateway.enabled)
    return enabled and bool(getattr(request.app.state, "ai_gateway_listening", enabled))


def _check_alias(alias: str) -> None:
    if not AI_GATEWAY_ALIAS_RE.fullmatch(alias):
        # The submitted text is not echoed: a path segment is caller-chosen.
        raise NerditError(
            422, "ai_gateway.alias_invalid", "Invalid alias.", hint=AI_GATEWAY_ALIAS_HINT
        )


def _local_ips(request: Request) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """The bridge addresses a plain-http base_url may name (loopback is checked apart)."""
    models = request.app.state.settings.models
    out: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for host in (models.bridge_bind_ip, models.bridge_advertise_host):
        try:
            out.add(ipaddress.ip_address(host or ""))
        except ValueError:
            continue
    return out


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to (a seam for tests)."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _url_error(message: str) -> NerditError:
    return NerditError(
        422,
        "ai_gateway.base_url_invalid",
        message,
        hint="Use https://<public provider host>/v1, or http:// to loopback or the "
        "models bridge for a local OpenAI-compatible server.",
    )


async def _check_base_url(request: Request, url: str) -> str:
    """Refuse anything but https to public addresses or http to loopback / the bridge.

    Returns the host (the only part of the URL audit params carry). Messages
    name the host at most, never the URL (it could carry a pasted credential).
    ponytail: resolved once at write time, so DNS rebinding after the write is
    out of reach; pin the resolved IPs on the route if that ever matters.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise _url_error("base_url is not a valid URL.") from None
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host or not url.isprintable():
        # Printable: httpx refuses control characters at request time, which
        # would turn every later app call into an unstructured 500.
        raise _url_error("base_url must be an absolute, printable http(s) URL with a host.")
    if parts.username is not None or parts.password is not None:
        raise _url_error("base_url must not embed credentials; use api_key_ref.")
    if parts.query or parts.fragment:
        raise _url_error("base_url must not carry a query or a fragment.")
    try:
        addrs = await asyncio.to_thread(
            _resolve, host, port or (443 if parts.scheme == "https" else 80)
        )
    except (OSError, UnicodeError):
        raise _url_error(f"base_url host '{host}' does not resolve.") from None
    local = _local_ips(request)
    ips = []
    for raw in addrs:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
        mapped = getattr(ip, "ipv4_mapped", None)
        ips.append(mapped or ip)
    if not ips:
        raise _url_error(f"base_url host '{host}' does not resolve.")
    if all(ip.is_loopback or ip in local for ip in ips):
        settings = request.app.state.settings
        admin_port = settings.proxy.admin_addr.rpartition(":")[2]
        own = {settings.daemon.port, settings.ai_gateway.port}
        if admin_port.isdigit():
            own.add(int(admin_port))
        if (port or (443 if parts.scheme == "https" else 80)) in own:
            raise _url_error("base_url must not point at the daemon's own listeners.")
        return host
    if any(not ip.is_global for ip in ips):
        raise _url_error(
            f"base_url host '{host}' resolves to a private, link-local or reserved address."
        )
    if parts.scheme != "https":
        raise _url_error(f"base_url host '{host}' needs https.")
    return host


def _check_ref(ref: str) -> str:
    """The ref's KEY name; only `${secrets.shared.KEY}` (the gateway reads the shared scope)."""
    match = SECRET_REF_RE.fullmatch(ref)
    if match is None or match.group(1) != "shared":
        # Never echo the value: a literal provider key pasted here is the likely mistake.
        raise NerditError(
            422,
            "ai_gateway.secret_ref_invalid",
            "api_key_ref must be a ${secrets.shared.KEY} reference.",
            hint="Store the key with `nerdit secrets set --shared KEY=...` (or --prompt KEY), "
            "then pass ${secrets.shared.KEY}.",
        )
    return match.group(2)


async def _aliases_in_use(request: Request, alias: str) -> list[str]:
    """Services the daemon keeps running whose `[ai.*]` names this gateway alias."""
    jobs = await request.app.state.queries.list_jobs(kind=JobKind.service)
    users = []
    for job in jobs:
        if job.desired_state != "running":
            continue
        specs = parse_job_config(job).get("ai")
        if not isinstance(specs, dict):
            continue
        if any(
            isinstance(spec, dict)
            and spec.get("provider") == "gateway"
            and spec.get("model") == alias
            for spec in specs.values()
        ):
            users.append(job.service_name or job.name)
    return sorted(users)


@router.get("/ai-gateway", response_model=AiGatewayView, operation_id="get_ai_gateway")
async def get_ai_gateway(request: Request) -> AiGatewayView:
    """The AI gateway's state: enabled, where apps reach it, the alias count."""
    routes = await request.app.state.queries.list_ai_routes()
    return _view(request, len(routes))


@router.get("/ai-gateway/routes", response_model=AiRouteList, operation_id="list_ai_routes")
async def list_ai_routes(request: Request) -> AiRouteList:
    """Every alias (provider keys appear as their `${secrets.shared.KEY}` reference only)."""
    return AiRouteList(
        routes=await request.app.state.queries.list_ai_routes(), enabled=_enabled(request)
    )


@router.put(
    "/ai-gateway/routes/{alias}",
    response_model=AiRouteWriteResponse,
    operation_id="set_ai_route",
)
async def set_ai_route(
    request: Request, alias: str, body: AiRouteSetRequest
) -> AiRouteWriteResponse:
    """Create or replace an alias (submitter or admin; `Idempotency-Key` required).

    Answers 200 with `enabled: false` while the gateway is off: the alias is
    stored and serves once `[ai_gateway]` is enabled.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    _check_alias(alias)
    host: str | None = None
    ref_key: str | None = None
    if body.provider == "ollama":
        if body.base_url is not None:
            raise _url_error("provider 'ollama' forbids base_url: the daemon serves the model.")
        if body.api_key_ref is not None:
            raise NerditError(
                422,
                "ai_gateway.secret_ref_invalid",
                "provider 'ollama' forbids api_key_ref: a served model needs no key.",
            )
    else:
        if body.base_url is None:
            raise _url_error("provider 'api' requires base_url.")
        if body.api_key_ref is None:
            raise NerditError(
                422,
                "ai_gateway.secret_ref_invalid",
                "provider 'api' requires api_key_ref (a ${secrets.shared.KEY} reference).",
            )
        ref_key = _check_ref(body.api_key_ref)
        host = await _check_base_url(request, body.base_url)
    request.state.audit_params = audit_params(
        {
            "alias": alias,
            "provider": body.provider,
            "model": body.model,
            "base_url_host": host,
            "api_key_ref_name": ref_key,
        }
    )
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            "An AI gateway route write requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )
    queries = request.app.state.queries
    created = await queries.upsert_ai_route(
        alias,
        provider=body.provider,
        model=body.model,
        base_url=body.base_url,
        api_key_ref=body.api_key_ref,
    )
    route = await queries.get_ai_route(alias)
    if route is None:  # deleted between the upsert and the read
        raise NerditError(409, "ai_gateway.conflict", f"Alias '{alias}' changed; retry.")
    return AiRouteWriteResponse(route=route, created=created, enabled=_enabled(request))


@router.delete("/ai-gateway/routes/{alias}", operation_id="remove_ai_route")
async def remove_ai_route(
    request: Request,
    alias: str,
    force: bool = Query(False, description="Remove even if a running app uses the alias."),
) -> dict:
    """Remove an alias (submitter or admin).

    409 while a running app's `[ai.*]` uses it, unless `force`.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    _check_alias(alias)
    request.state.audit_params = audit_params({"alias": alias, "force": force})
    queries = request.app.state.queries
    if await queries.get_ai_route(alias) is None:
        raise NerditError(404, "not_found", f"No AI gateway alias '{alias}'.")
    users = await _aliases_in_use(request, alias)
    if users and not force:
        raise NerditError(
            409,
            "ai_gateway.alias_in_use",
            f"Alias '{alias}' is used by running apps: {', '.join(users)}.",
            hint="Point those apps at another alias first, or pass ?force=true "
            "(their AI calls then answer 404 alias_not_found).",
        )
    await queries.delete_ai_route(alias)
    return {"alias": alias, "deleted": True, "used_by": users}


@router.get("/ai-gateway/keys", response_model=AiKeyList, operation_id="list_ai_gateway_keys")
async def list_ai_gateway_keys(
    request: Request,
    include_revoked: bool = Query(False, description="Also list revoked keys."),
) -> AiKeyList:
    """Which services hold a virtual key, and since when (never the key or its hash)."""
    return AiKeyList(
        keys=await request.app.state.queries.list_ai_gateway_keys(include_revoked=include_revoked)
    )


@router.delete(
    "/ai-gateway/keys/{service}",
    response_model=AiKeyRevokeResponse,
    operation_id="revoke_ai_gateway_keys",
)
async def revoke_ai_gateway_keys(request: Request, service: str) -> AiKeyRevokeResponse:
    """Revoke a service's virtual keys (its owner or admin).

    The app gets a new one when it restarts.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    queries = request.app.state.queries
    # The path value reaches the audit row and the response only once it names a
    # real service: a pasted key or a typo must not be stored or echoed.
    job = await queries.get_service_by_name(service)
    if job is None:
        raise NerditError(404, "not_found", "No such service.", hint="nerdit ai keys list")
    # A submitter revokes its own apps' keys only (a revoked app answers 401 until
    # it restarts); NULL-owner rows stay admin-only, like every service verb.
    require_owner_or_admin(request, job)
    request.state.audit_params = audit_params({"service": service})
    revoked = await queries.revoke_ai_gateway_keys(service)
    return AiKeyRevokeResponse(service=service, revoked=revoked)


@router.get("/ai-gateway/usage", response_model=AiUsageList, operation_id="get_ai_usage")
async def get_ai_usage(
    request: Request,
    service: str | None = Query(None, description="Only this service's usage."),
    days: int = Query(7, ge=1, le=90, description="UTC days back, today included."),
) -> AiUsageList:
    """Requests, tokens and upstream errors per (UTC day, service, alias)."""
    usage = await request.app.state.queries.list_ai_usage(service_name=service, days=days)
    return AiUsageList(days=days, usage=usage)
