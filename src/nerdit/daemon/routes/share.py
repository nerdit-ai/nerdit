"""Manage hosted shares; the cloud edge controls who can reach them.

Every request validates sharing intent; active streams periodically recheck it.
Sharing requires enabled link configuration, node ID, slug, hosted base domain
and a tunnel manager; URLs are computed from known metadata, never guessed.

Hosted traffic bypasses Caddy. edge_auth therefore expresses public-sharing
intent in place of consent but does not protect the hosted URL. Anonymous public
access is decided solely by the cloud edge. Local requests for public sharing
also require a cloud-pushed entitlement fresh within its 24-hour TTL; missing
or stale entitlement fails closed. Mounted under /api only.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Request

from nerdit.config.settings import LOOPBACK_HOSTS
from nerdit.core.eventlog import get_recorder
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.link.hosted import hosted_host, hosted_label_fits
from nerdit.core.proxy import EdgeAuthInvalid, load_edge_auth
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import require_owner_or_admin, require_role
from nerdit.daemon.deploy_pipeline import apex_shadow_active
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas.exposure import (
    ShareOrigin,
    ShareRemovedView,
    ShareRequest,
    ShareView,
)
from nerdit.daemon.views.hosted import (
    HostedContext,
    hosted_entry,
    hosted_state,
    load_hosted_context,
)
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import Job, JobKind, JobStatus, TokenRole
from nerdit.db.queries.shares import HostedOnlyConflictError
from nerdit.db.rows import ServiceShare

router = APIRouter()


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route `Idempotency-Key` gate (the `routes/link.py` wording).

    A local copy rather than an import: `routes.link` is a sibling route
    module, and a route importing another route module is the edge the
    import-cycle scanner exists to prevent. Four lines is the cheaper coupling.
    """
    if not request.headers.get("Idempotency-Key"):
        raise NerditError(
            400,
            "idempotency_key_required",
            f"A {what} requires an Idempotency-Key header.",
            hint="Send a unique Idempotency-Key so the write is safe to retry.",
        )


def _link_required(hint: str) -> NerditError:
    """The one code for every "this node cannot compose a hosted URL" refusal.

    One code, four hints (D-P26-H5): an agent branches on `share.link_required`
    and does not care *which* half of the link is missing, while the operator
    reading the hint needs the exact next command.
    """
    return NerditError(
        409,
        "share.link_required",
        "This node cannot serve a hosted share yet.",
        hint=hint,
    )


def _require_addressable(link: Any, manager: Any) -> Any:
    """Refuse unless this node can compose — and serve — a hosted URL (D-P26-H5).

    Ordered from "you have not started" to "you are one restart away", because
    an operator fixes the first thing the hint names and re-runs. `enabled` is
    checked BEFORE the manager on purpose: `build_link_manager` returns
    `None` on a disabled link before anything else, so a claimed-but-disabled
    node used to land in the tunnel-down branch and be told to restart — advice
    a restart cannot satisfy, and an operator loop (PR review, P26 WP-H).

    Returns the manager it just proved non-`None` so the caller reads the
    entitlement off it without a second existence check.
    """
    if not (link.node_id and link.slug):
        raise _link_required("Link this node first: run `nerdit link <code>`.")
    if not link.nodes_base_domain:
        raise _link_required(
            "Run `nerdit link refresh` to learn the hosted domain, then restart the daemon."
        )
    if not link.enabled:
        raise _link_required(
            "The link is switched off: run `nerdit config set link enabled=true`, "
            "then restart the daemon."
        )
    if manager is None:
        raise _link_required("The tunnel is not running — restart the daemon.")
    return manager


def _has_edge_auth(job: Job) -> bool:
    """Whether the app declares a `[deploy].edge_auth` block (advisory, D-P26-H3).

    A malformed block reads as **absent**, not as present: the fail-closed
    posture the proxy already takes on `EdgeAuthInvalid`. Treating a broken
    declaration as "the operator thought about access" would let a typo stand in
    for consent on a world-reachable URL.
    """
    try:
        return load_edge_auth(parse_job_config(job).get("edge_auth")) is not None
    except EdgeAuthInvalid:
        return False


#: What to do when the share is provisioned but the app behind it is not
#: running. The hosted resolver refuses a service with no live endpoint and the
#: mux answers `share.not_shared` — the SAME 404 an unshared name gets — so
#: without this line the operator reads "ready" and a 404 and concludes the
#: share is broken. `diagnose_service` is named because it is the one call
#: that returns the remediation code for a crash-loop.
_ORIGIN_DOWN_HINT = (
    "the service is {status}; the URL returns share.not_shared until it runs "
    "— call diagnose_service"
)


def _origin(job: Job) -> ShareOrigin:
    """Whether the app behind a share is answering right now.

    `answers` is `running` and nothing else. `degraded` is deliberately
    OUTSIDE it: a degraded row keeps its container (the reconciler never kills
    on health alone) but its health check is failing, which is precisely the
    "the link is up and the app is broken" state this field exists to name.
    Read off the row — never a probe — so a share read stays two point lookups.
    """
    answers = job.status is JobStatus.running
    return ShareOrigin(
        status=job.status,
        answers=answers,
        hint=None if answers else _ORIGIN_DOWN_HINT.format(status=job.status.value),
    )


def _exposed_without_token(settings: Any) -> bool:
    """Whether a client off this machine can reach the daemon API with no token (lot 5).

    Bound off loopback, or behind the path-mode dashboard apex: either way a
    forged loopback Host gets admin, so a lock this daemon reports could be lifted
    by anyone on the LAN. Reads `settings.daemon` directly: a missing section is
    a harness bug, never "not exposed".
    """
    daemon = settings.daemon
    return daemon.auth_token is None and (
        daemon.host not in LOOPBACK_HOSTS or apex_shadow_active(settings)
    )


async def _local_routes(
    request: Request, service_name: str
) -> Literal["absent", "present", "unknown"]:
    """Observe the app's local proxy routes; every doubt answers `unknown` (fail closed).

    A token-less exposed daemon is `unknown` whatever the table says (D10), and
    so is a daemon with no proxy manager to ask (D19).
    """
    if _exposed_without_token(request.app.state.settings):
        return "unknown"
    manager = getattr(request.app.state, "proxy_manager", None)
    if manager is None:
        return "unknown"
    return await manager.local_route_state(service_name)


def _view(
    service_name: str,
    share: ServiceShare,
    hosted: HostedContext,
    job: Job,
    local_routes: Literal["absent", "present", "unknown"],
) -> ShareView:
    """Project a stored row against the LIVE link — url and state are never stored.

    A row records intent; whether that intent is currently addressable is a fact
    about the tunnel, recomputed on every read. Same projection the
    `public_urls` list uses, so the two surfaces cannot disagree about the
    same share. `origin` rides the same recompute, from the `job` row both callers
    already resolved — the caller passes it in rather than this helper reading
    it back, so the view can never describe a different row than the one the
    route authorized.
    """
    entry = hosted_entry(hosted, service_name)
    return ShareView(
        service_name=service_name,
        access=share.access,
        url=entry.url if entry is not None else None,
        state=hosted_state(hosted, share),
        created_at=share.created_at,
        origin=_origin(job),
        hosted_only=share.hosted_only,
        local_routes=local_routes,
    )


@router.get(
    "/services/{name}/share",
    response_model=ShareView,
    operation_id="get_share",
)
async def get_share(request: Request, name: str) -> ShareView:
    """Read a service's hosted share, or a structured 404 when it has none.

    Open to any authenticated principal, exactly like `GET /services/{ident}`:
    the body carries a hostname the operator chose to publish, not a credential.

    An unshared service is `404 share.not_shared` rather than a synthetic
    `access: "private"` body — absence is reported, never invented, so a
    caller can tell "never shared" from "shared privately".
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    service_name = job.service_name or name
    hosted = await load_hosted_context(request)
    share = hosted.shares.get(service_name)
    if share is None:
        raise NerditError(
            404,
            "share.not_shared",
            f"Service '{service_name}' is not shared.",
            hint="Share it with `nerdit share <name>`.",
        )
    return _view(service_name, share, hosted, job, await _local_routes(request, service_name))


@router.put(
    "/services/{name}/share",
    response_model=ShareView,
    operation_id="set_share",
)
async def set_share(request: Request, name: str, body: ShareRequest) -> ShareView:
    """Record sharing intent; activation selects the generated canonical address.

    Require owner/admin and token scope; re-sharing updates access while preserving
    created_at. Validate identity, kind, link capability and entitlement/consent
    before upsert, in that order. Recheck row existence under the DB write lock to
    handle concurrent deletion. The idempotent write is audited as share.set.

    hosted_only=true (lot 5) is refused on a token-less exposed daemon or while a
    domain is bound; a stored lock drops the app's local routes inline.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params(
        {
            "service": name,
            "access": body.access,
            "consent": body.consent,
            "hosted_only": body.hosted_only,
        }
    )
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)
    _require_idempotency_key(request, "share write")

    if job.kind != JobKind.service:
        # (S9) Models and databases have no HTTP app semantics on the hosted
        # path — a model is loopback-only by design and a database speaks its
        # own wire protocol. Refusing here keeps the mux's kind re-check a
        # defence-in-depth check rather than the only one.
        raise NerditError(
            422,
            "share.kind_unsupported",
            f"'{job.kind.value}' workloads cannot be shared.",
            hint="Only deployed apps (kind 'service') can be exposed over the link.",
        )

    service_name = job.service_name or name
    link = request.app.state.settings.link
    manager = _require_addressable(link, getattr(request.app.state, "link_manager", None))

    if body.access == "public":
        if not manager.status().hosted_public_entitled:
            raise NerditError(
                409,
                "share.not_entitled",
                "Public hosted shares are not enabled for this node's account.",
                hint=(
                    "Public sharing is free during the public beta. Check `nerdit link` "
                    "and the account status in the Nerdit console; the cloud must "
                    "confirm access before you publish."
                ),
            )
        # (D-P26-H3) `edge_auth` is a Caddy handler and is NOT on the hosted
        # path — accepted here as a declaration of intent, never as protection.
        if not (_has_edge_auth(job) or body.consent):
            raise NerditError(
                409,
                "share.unprotected",
                f"Service '{service_name}' would be reachable by anyone with the URL.",
                hint=(
                    "Public means public: add a [deploy].edge_auth block to the app, "
                    "or pass consent=true to accept it."
                ),
            )

    if body.hosted_only is True and _exposed_without_token(request.app.state.settings):
        raise NerditError(
            409,
            "share.hosted_only_no_token",
            "This daemon is reachable without a token, so a hosted-only lock "
            "could be lifted by anyone who reaches it.",
            hint=(
                "Set `[daemon].auth_token` (`nerdit init --auth-token-only`) and "
                "restart, or bind `[daemon].host` to 127.0.0.1 and turn off "
                "`[proxy].dashboard_apex`."
            ),
        )

    try:
        share = await queries.set_service_share(
            service_name,
            body.access,
            job_id=job.id,
            alias_node_id=link.node_id,
            alias_host=(
                hosted_host(service_name, link.slug, link.nodes_base_domain)
                if hosted_label_fits(service_name, link.slug)
                else None
            ),
            preserve_existing=body.preserve_existing,
            hosted_only=body.hosted_only,
        )
    except HostedOnlyConflictError as exc:
        # A domain route is a local route: locking around it would leave a
        # name that binds and never serves, with no reason given. Refused under
        # the upsert's own write lock, so a domain cannot race in beside the lock.
        raise NerditError(
            409,
            "share.hosted_only_conflict",
            f"Service '{service_name}' has a bound domain, which is a local route.",
            hint=(
                f"Remove each bound domain first: `nerdit domains remove {service_name} <domain>`."
            ),
            domains=exc.domains,
        ) from None
    if share is None:
        # The service was deleted between this route's resolve and the write
        # (the upsert re-checks under the DB write lock — see
        # `ShareQueries.set_service_share`). Nothing was written, so the
        # honest answer is the 404 the resolve would have given a moment later;
        # inventing a share for a name with no service behind it is what would
        # expose the NEXT app deployed under it.
        raise _not_found(name)
    request.state.audit_params = audit_params(
        {
            "service": name,
            "access": share.access,
            "consent": body.consent,
            "hosted_only": share.hosted_only,
        }
    )
    proxy_manager = getattr(request.app.state, "proxy_manager", None)
    if share.hosted_only and proxy_manager is not None:
        # Inline, so the answer below can already read `absent`; it waits on the
        # routes lock, so an in-flight register finishes (and is refused) first.
        # Unlock needs nothing inline: the next reconcile restores the routes.
        await proxy_manager.deregister(service_name)
    hosted = await load_hosted_context(request)
    view = _view(service_name, share, hosted, job, await _local_routes(request, service_name))

    recorder = get_recorder()
    if recorder is not None:
        # A PRIVATE hosted URL is not a bearer capability — the cloud edge still
        # demands a signed-in identity the owner allows (the owner, or a person
        # allowed in the Nerdit App) — so it is safe to carry. A PUBLIC one is
        # omitted: this feed is POSTed verbatim to operator-configured webhook
        # hosts, and a world-reachable address does not need to travel there.
        data: dict[str, object] = {"access": share.access}
        if share.access == "private" and view.url:
            data["url"] = view.url
        await recorder.record("share.ready", kind="service", service_name=service_name, data=data)
    return view


@router.delete(
    "/services/{name}/share",
    response_model=ShareRemovedView,
    operation_id="remove_share",
)
async def remove_share(request: Request, name: str) -> ShareRemovedView:
    """Unshare an app. Idempotent `200`, never a 404 on a double delete.

    `removed: false` is the no-op answer, so a retrying agent converges
    instead of having to distinguish "already gone" from "never existed". No
    in-route `Idempotency-Key` requirement (the `DELETE /services/{ident}`
    precedent — a delete is its own replay); the middleware still replays one
    when a caller sends it.

    New streams re-read committed intent immediately; active streams recheck
    it within the mux watchdog interval, including quiet SSE responses.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    request.state.audit_params = audit_params({"service": name})
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)

    service_name = job.service_name or name
    removed = await queries.delete_service_share(service_name)
    if removed:
        recorder = get_recorder()
        if recorder is not None:
            # `reason` separates an owner's explicit unshare from the
            # `service_deleted` edge the purge route emits.
            await recorder.record(
                "share.removed",
                kind="service",
                service_name=service_name,
                reason="unshared",
            )
    return ShareRemovedView(service_name=service_name, removed=removed)
