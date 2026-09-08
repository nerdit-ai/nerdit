"""Bind custom domains to services through the node's Caddy proxy.

Domains work at the host root in either proxy mode, independently of the tunnel.
The service_domains table is authoritative: redeploy never restores or removes
bindings. Reject wildcards, IPs, IDNs, single labels and reserved node names.
Reserved names are captured at boot to match the running proxy configuration.

Public ACME issuance requires enabled node policy and an admin caller, protecting
the shared CA order budget. Omitted/null acme preserves policy; explicit false
downgrades, and true requests public issuance. Changing policy emits no new
exposure event. TLS automation retains a catch-all internal issuer so unmatched
routes cannot trigger public issuance. Mounted under /api only.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Request

from nerdit.core.eventlog import get_recorder
from nerdit.core.proxy import domain_url_for, public_url_for
from nerdit.core.proxy.domains import (
    DomainInvalid,
    ReservedNames,
    normalize_domain,
    reserved_names,
    validate_domain,
)
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import current_principal, require_owner_or_admin, require_role
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas.exposure import (
    DomainListView,
    DomainRemovedView,
    DomainRequest,
    DomainSetView,
    DomainView,
)
from nerdit.daemon.views.hosted import (
    HostedContext,
    domain_cert_state,
    domain_state,
    load_hosted_context,
)
from nerdit.daemon.views.service import _not_found, _resolve_service
from nerdit.db.models import JobKind, TokenRole
from nerdit.db.rows import ServiceDomain

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_idempotency_key(request: Request, what: str) -> None:
    """In-route `Idempotency-Key` gate (the `routes/share.py` wording).

    A local copy rather than an import: `routes.share` is a sibling route
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


#: What the audit row records in place of a domain that did not validate
#: (Codex round 1, P1 #3831777111). Not `None` and not an absent key: the
#: audit row must still say that a bind was ATTEMPTED — refused probes are the
#: reason the row is stamped before the refusals in the first place.
_REFUSED = "<refused>"

#: The same, for a row stamped before validation has run at all (an earlier
#: refusal — 404, 403, missing key, unsupported kind — settles the request
#: first). Distinct from `_REFUSED` so a reader can tell "the name was judged
#: and rejected" from "the name was never judged".
_UNVALIDATED = "<unvalidated>"


def _fingerprint(raw: str) -> str:
    """A short, stable digest of the domain **as sent** — never the value.

    This is what keeps the WP1 live-run property the plain value used to carry:
    a refused homograph must not be recorded under the ASCII name it resembles,
    which is a different and bindable resource. A digest over the submitted
    bytes separates the two (and correlates repeated probes of the same string)
    without persisting anything a mis-pasted secret could hide in. Trimmed, not
    folded, for the same reason the old stamp was: `casefold()` is not
    ASCII-preserving, so folding first would collapse exactly the two spellings
    this has to keep apart. 16 hex characters — an audit correlator, not a
    commitment scheme.
    """
    return hashlib.sha256(raw.strip().encode("utf-8")).hexdigest()[:16]


def _reserved(request: Request) -> ReservedNames:
    """Return boot-captured reserved names, falling back to current proxy settings.

    Bare app state yields an empty set: grammar-only validation, not fail-closed
    ownership protection. This fallback is for minimal tests and is debug-logged.
    """
    state = request.app.state
    reserved = getattr(state, "domain_reserved", None)
    if isinstance(reserved, ReservedNames):
        return reserved
    settings = getattr(state, "settings", None)
    proxy = getattr(settings, "proxy", None) if settings is not None else None
    if proxy is None:
        logger.debug(
            "[domains] app.state carries neither domain_reserved nor settings — "
            "validating grammar only (no reserved-name check)"
        )
        return ReservedNames(names=frozenset())
    return reserved_names(proxy, getattr(state, "hostname", "") or "")


def _acme_enabled(request: Request) -> bool:
    """Read node ACME policy, failing closed to false for missing or malformed state."""
    settings = getattr(request.app.state, "settings", None)
    proxy = getattr(settings, "proxy", None)
    acme = getattr(proxy, "acme", None)
    return bool(getattr(acme, "enabled", False))


def _url(hosted: HostedContext, domain: str) -> str:
    """Compose a domain's URL through the seam, never by hand."""
    return domain_url_for(
        domain,
        scheme=hosted.scheme,
        https_port=hosted.https_port,
        public_port=hosted.public_port,
    )


def _view(row: ServiceDomain, hosted: HostedContext, public_url: str | None) -> DomainView:
    """Project a stored row against the LIVE proxy — url and state are never stored.

    Same discipline the share view follows: the row records the binding, the
    proxy decides whether it answers right now. Shared with `public_urls` via
    `nerdit.daemon.views.hosted.domain_state` and
    `nerdit.daemon.views.hosted.domain_cert_state`, so the two surfaces
    cannot disagree about the same domain — on the route fact or the
    certificate one.
    """
    cert = domain_cert_state(hosted, row)
    return DomainView(
        service_name=row.service_name,
        domain=row.domain,
        acme=row.acme,
        kind=row.kind,
        created_at=row.created_at,
        url=_url(hosted, row.domain),
        state=domain_state(hosted, row.service_name, row.domain, public_url),
        cert_state=cert.state,
        cert_not_after=cert.not_after,
    )


async def _context(request: Request, service_name: str) -> tuple[HostedContext, str | None]:
    """The exposure snapshot + this service's default proxy URL.

    `public_url` is composed exactly as `views/service.py::_endpoint_view`
    composes it — the `public_url_for` seam, gated on a live proxy — because
    `nerdit.daemon.views.hosted.domain_state` reads its NULL-ness as
    "is the proxy serving this service at all". Re-deriving that condition here
    from anything else would let the domains surface and `public_urls`
    disagree about the same app.

    One point read for the endpoint row: this is a single-service surface, not
    a list, so there is no N to batch away (S-W1 bounds the LIST paths).
    """
    queries = request.app.state.queries
    state = request.app.state
    settings = getattr(state, "settings", None)
    proxy = getattr(settings, "proxy", None) if settings is not None else None
    hostname = getattr(state, "hostname", None)
    manager = getattr(state, "proxy_manager", None)
    hosted = await load_hosted_context(request)
    if not (bool(getattr(manager, "available", False)) and proxy is not None and hostname):
        return hosted, None
    endpoint = await queries.get_service_endpoint(service_name)
    if endpoint is None:
        return hosted, None
    return hosted, public_url_for(
        endpoint.service_name,
        endpoint.route,
        mode=proxy.mode,
        hostname=hostname,
        base_domain=proxy.base_domain,
        scheme=proxy.scheme,
        https_port=proxy.https_port,
        public_port=proxy.public_port,
    )


@router.get(
    "/services/{name}/domains",
    response_model=DomainListView,
    operation_id="list_domains",
)
async def list_domains(request: Request, name: str) -> DomainListView:
    """List a service's direct domains — an empty list, never a 404.

    Open to any authenticated principal, exactly like `GET /services/{ident}`
    and `GET /services/{name}/share`: the body carries hostnames the operator
    chose to publish, not credentials.

    A service with no domains is a normal service, so absence is an empty list;
    only an unknown *service* is a 404. The rows come from the batched exposure
    context, so listing costs no query beyond the one the page already paid.
    """
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    service_name = job.service_name or name
    hosted, public_url = await _context(request, service_name)
    return DomainListView(
        service_name=service_name,
        domains=[_view(row, hosted, public_url) for row in hosted.domains.get(service_name, ())],
    )


@router.put(
    "/services/{name}/domains/{domain:path}",
    response_model=DomainSetView,
    operation_id="add_domain",
)
async def add_domain(
    request: Request, name: str, domain: str, body: DomainRequest
) -> DomainSetView:
    """Bind a direct domain at https://<domain>/ without redeploying the app.

    Require owner/admin and token scope. Trim, case-fold and remove one trailing dot
    before matching identity. ACME omitted/null preserves policy (false on insert);
    explicit true requires admin and enabled node ACME, while false downgrades.

    Before writing, validate identity, idempotency key, kind/domain shape, ACME policy
    and conflicts, in that order. Recheck job ownership under the DB write lock so
    concurrent deletion returns 404. Audit domain.added; routing converges on the
    next proxy tick.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    # Stamped BEFORE the refusals so a denied attempt to bind a name is still
    # recorded — that is what makes probing visible. What it records is a
    # DIGEST, never the submitted string: the domain is an arbitrary path
    # segment, `audit_params` masks by key name and "domain" is not a secret
    # key, so a mis-pasted bearer token would land verbatim in
    # `params_redacted` and in the published `audit.*` event (Codex round 1,
    # P1 #3831777111). The canonical name replaces the placeholder the moment
    # validation accepts it.
    #
    # It deliberately precedes the ACME gate below (Codex round 2): that gate is
    # the one refusal a *submitter* can trigger by hand, and it exists to make
    # exactly this probing visible — a denial recorded without the digest cannot
    # tell one operator retrying one name from one token spraying a hundred.
    # Stamping is pure dict construction: no query, no I/O, nothing to reorder.
    request.state.audit_params = audit_params(
        {
            "service": name,
            "domain": _UNVALIDATED,
            "domain_sha256": _fingerprint(domain),
            "acme": body.acme,
        }
    )
    # The node-wide ACME account is an admin resource (review round 1). A
    # submitter that can bind names it does not control could otherwise park any
    # number of them as `acme=1`: each becomes a subject of the first
    # automation policy, Caddy orders on the next tick, and the CA's per-ACCOUNT
    # limits (order rate, failed validations per hostname, pending
    # authorizations) are burnt for the admin's real public domains on the same
    # node. Judged here, in the identity band, before anything else and long
    # before anything is written.
    #
    # A bespoke envelope rather than `require_role(request, admin)`: the
    # generic 403 would say "your role is not permitted for this operation",
    # which is false — the operation IS permitted, only this one flag is not.
    # `reason` is a top-level extra (the `domain.acme_disabled` precedent)
    # because the MCP mapper drops `detail` and copies extras through, so an
    # agent branches on a token, not on a sentence.
    if body.acme is True and current_principal(request).role is not TokenRole.admin:
        raise NerditError(
            403,
            "domain.acme_forbidden",
            "Requesting a public certificate requires an admin token.",
            hint=(
                "This node has one ACME account and its rate limits are shared by "
                "every domain on it. Re-send without acme (the name is served with "
                "the node's internal CA) or ask an admin to add it with acme=true."
            ),
            reason="acme_admin_only",
        )
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)
    _require_idempotency_key(request, "domain write")

    if job.kind != JobKind.service:
        # A model is loopback-only by design and a database speaks its own wire
        # protocol — neither has HTTP app semantics for a Host route to carry.
        # The share route refuses the same two kinds for the same reason.
        raise NerditError(
            422,
            "domain.kind_unsupported",
            f"'{job.kind.value}' workloads cannot have a direct domain.",
            hint="Only deployed apps (kind 'service') can be served under a domain.",
        )

    try:
        validated = validate_domain(domain, reserved=_reserved(request))
        # Accepted: from here the canonical (folded) name is the resource, and
        # it is a well-formed DNS name — safe to record and to echo.
        request.state.audit_params = audit_params(
            {"service": name, "domain": validated, "acme": body.acme}
        )
    except DomainInvalid as exc:
        request.state.audit_params = audit_params(
            {
                "service": name,
                "domain": _REFUSED,
                "reason": exc.reason,
                "domain_sha256": _fingerprint(domain),
                "acme": body.acme,
            }
        )
        raise NerditError(
            422,
            "domain.invalid",
            # `public_message`, never `message`: the latter quotes the input,
            # and this envelope travels into the MCP tool result and from there
            # into an agent transcript. Name the reason, not the input.
            exc.public_message,
            hint=(
                "Use a bare, lowercase DNS name you control (e.g. app.example.com); "
                "wildcards, IP literals and this node's own names are refused."
            ),
            detail={"reason": exc.reason},
            # Top-level too: the MCP mapper drops `detail` by design (it can
            # echo submitted values), but copies extras through — an agent
            # branches on `reason` without the human text.
            reason=exc.reason,
        ) from exc

    if body.acme and not _acme_enabled(request):
        # Refused BEFORE the write, never accepted-and-ignored: a stored row
        # claiming a public certificate the operator will never get is worse
        # than an honest refusal. Its position in the refusal order is
        # unchanged (policy, after shape, before conflict) so an operator who
        # sent a bad name still hears about the name first.
        raise NerditError(
            409,
            "domain.acme_disabled",
            "ACME is not enabled on this node.",
            hint=(
                "Set [proxy.acme] enabled=true and email, restart the daemon, "
                "or add the domain without --acme."
            ),
            # Top-level extra, not `detail`: the MCP mapper drops `detail`
            # by design and copies extras through, so an agent branches on the
            # token without parsing the sentence (the `domain.invalid`
            # precedent above).
            reason="acme_disabled",
        )

    service_name = job.service_name or name
    outcome, row = await queries.add_service_domain(
        service_name, validated, acme=body.acme, job_id=job.id
    )
    if outcome == "no_service":
        # Deleted between this route's resolve and the write (the insert
        # re-checks under the DB write lock). Nothing was written, so the honest
        # answer is the 404 the resolve would have given a moment later.
        raise _not_found(name)
    if outcome == "taken":
        # The message names the DOMAIN only. An owner-scoped principal must not
        # be able to enumerate another owner's app names by probing names it
        # does not hold.
        raise NerditError(
            409,
            "domain.taken",
            f"Domain '{validated}' is already bound to another service on this node.",
            hint="Remove it from the service that holds it, or choose another name.",
        )
    # `inserted`/`exists` both carry the row.
    assert row is not None
    # Third and final stamp: the row as it now STANDS. With a tri-state `acme`
    # the submitted value can be "unspecified", and an audit row saying `null`
    # would record the gesture while losing the fact — which is exactly backwards
    # for the one durable trace of a flip (this write mints no event). Before the
    # write the submitted value is all there is; after it, the stored row is the
    # truth and it is knowable, so record that.
    request.state.audit_params = audit_params(
        {"service": name, "domain": validated, "acme": row.acme}
    )
    hosted, public_url = await _context(request, service_name)
    view = DomainSetView(
        **_view(row, hosted, public_url).model_dump(), created=outcome == "inserted"
    )

    if outcome == "inserted":
        recorder = get_recorder()
        if recorder is not None:
            # The domain name is public DNS the operator chose to point at this
            # box — not a bearer capability — so it may travel to the webhook
            # hosts this feed is POSTed to. A re-PUT mints nothing: the exposure
            # did not change.
            await recorder.record(
                "domain.added",
                kind="service",
                service_name=service_name,
                data={"domain": row.domain, "acme": row.acme},
            )
    return view


@router.delete(
    "/services/{name}/domains/{domain:path}",
    response_model=DomainRemovedView,
    operation_id="remove_domain",
)
async def remove_domain(request: Request, name: str, domain: str) -> DomainRemovedView:
    """Release a domain, returning HTTP 200 and removed=false when already absent.

    Fold the name without enforcing current grammar for deletion, so legacy bindings
    remain removable. Scope deletion to the service. Validate only before reflecting
    input: refused names are represented by <refused> plus a digest, including newly
    reserved names that were successfully removed. Middleware replays a supplied
    Idempotency-Key; this route requires none. Routing updates on the next tick.
    """
    require_role(request, TokenRole.submitter, TokenRole.admin)
    folded = normalize_domain(domain)
    params: dict[str, object] = {"service": name}
    try:
        validate_domain(domain, reserved=_reserved(request))
        recorded = folded
        params["domain"] = folded
    except DomainInvalid:
        recorded = _REFUSED
        params["domain"] = _REFUSED
        params["domain_sha256"] = _fingerprint(domain)
    request.state.audit_params = audit_params(params)
    queries = request.app.state.queries
    job = await _resolve_service(queries, name)
    if job is None:
        raise _not_found(name)
    require_owner_or_admin(request, job)

    service_name = job.service_name or name
    removed = await queries.remove_service_domain(service_name, folded)
    if removed:
        recorder = get_recorder()
        if recorder is not None:
            # `reason` separates an owner's explicit removal from the
            # `service_deleted` cascade the purge route emits.
            await recorder.record(
                "domain.removed",
                kind="service",
                service_name=service_name,
                reason="removed",
                data={"domain": folded},
            )
    return DomainRemovedView(service_name=service_name, domain=recorded, removed=removed)
