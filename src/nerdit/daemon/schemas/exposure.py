"""Schemas for hosted shares, custom domains and service public URLs.

The scalar public_url remains the LAN/proxy URL; public_urls lists all kinds.
State is a machine token: ready, link_down, not_entitled, pending or withheld. Hosted
entries never report withheld; domain entries never report link_down.
Keep imports limited to leaf modules to avoid cycles through service views.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from nerdit.core.proxy.certs import CertState
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.enums import JobStatus


class PublicUrlEntry(BaseModel):
    """An advertised service URL and its current reachability.

    Default entries mirror public_url when present. Hosted and domain entries
    remain listed while their stored intent exists, even when unreachable;
    state reports reachability.
    """

    url: str | None = Field(
        description=("Absolute URL; null while awaiting a hosted address or on an unlinked node")
    )
    kind: Literal["default", "hosted", "domain"] = Field(
        description=(
            "'default' = the proxy/LAN URL (== public_url); 'hosted' = the P26 share; "
            "'domain' = a custom domain served by this node's proxy"
        )
    )
    state: Literal["ready", "link_down", "not_entitled", "withheld", "pending"] = Field(
        description=(
            "Machine token: answers now / tunnel or link missing / public not entitled / "
            "the proxy is not serving this service yet (domain entries) / hosted address pending"
        )
    )
    access: Literal["private", "public"] | None = Field(
        default=None, description="Hosted entries only; null on a default or domain entry"
    )
    domain: str | None = Field(
        default=None,
        description="Domain entries only: the bound name; null on every other kind",
    )
    hosted_only: bool | None = Field(
        default=None,
        description=(
            "Hosted entries only: true when the share is locked to its hosted URL "
            "(no local route); null on every other kind"
        ),
    )
    # A bolt-on, exactly like `access` and `domain`: one more
    # orthogonal fact about a domain entry, never a widening of `state`.
    # `state` stays the ROUTE fact (does this node serve the name right now);
    # `cert_state` is the CERTIFICATE fact (what a browser will make of it).
    # A client that wants "publicly reachable, no trust step" composes the two
    # rather than reading a single fused token that would have to be invented
    # for every future combination.
    cert_state: CertState | None = Field(
        default=None,
        description=(
            "Domain entries only: 'internal' (this node's CA) | 'disabled' "
            "([proxy.acme] off) | 'pending' | 'issued' | 'expired'; null on every "
            "other kind"
        ),
    )


class ShareRequest(StrictRequestModel):
    """`PUT /api/services/{name}/share` — the owner's exposure decision."""

    access: Literal["private", "public"] = Field(
        default="private",
        description=(
            "'private' (default) opens only for the owner of this machine's "
            "Nerdit account and the people they allow in the Nerdit App, checked "
            "at the cloud edge; "
            "'public' is world-reachable"
        ),
    )
    consent: bool = Field(
        default=False,
        description=(
            "Required (with, or instead of, a [deploy].edge_auth block) for "
            "access='public': public means public"
        ),
    )
    preserve_existing: bool = Field(
        default=False,
        description="Create a missing share, preserving an existing share's access mode atomically",
    )
    hosted_only: bool | None = Field(
        default=None,
        description=(
            "true = lock the app to its hosted URL: this machine's proxy keeps no "
            "local route for it (409 share.hosted_only_conflict while a domain is "
            "bound, 409 share.hosted_only_no_token when the daemon is reachable "
            "without a token); false = unlock and restore the local routes; "
            "omitted = keep the stored lock (unlocked on a new share). Any owner "
            "token may unlock"
        ),
    )


class ShareOrigin(BaseModel):
    """Whether the app behind a share is answering.

    Keep this separate from ShareView.state, which describes hosted-path
    provisioning rather than app health.
    """

    status: JobStatus = Field(description="The backing service's live workload status")
    answers: bool = Field(
        description=(
            "True only when the service is running (and not degraded) — i.e. the "
            "hosted URL has a live origin to reach"
        )
    )
    hint: str | None = Field(
        default=None,
        description="What to do about it when answers is false; null when it is true",
    )


class ShareView(BaseModel):
    """A stored share projected against the live link and app.

    URL and state are computed per request; origin separately reports whether the
    app is running. None of these live facts is stored in the share row.
    """

    service_name: str = Field(description="Stable service name")
    job_id: str = Field(
        description="The current incarnation's job id: the cloud's proof for a custom domain"
    )
    access: Literal["private", "public"] = Field(description="Stored exposure mode")
    url: str | None = Field(
        description="The hosted URL, or null while unlinked or awaiting a routable address"
    )
    state: Literal["ready", "link_down", "not_entitled", "pending"] = Field(
        description="ready = answers now; link_down = tunnel/link missing; "
        "not_entitled = public without the entitlement; pending = awaiting a routable address"
    )
    created_at: datetime = Field(description="When the share was first created")
    origin: ShareOrigin = Field(
        description=(
            "(P34) The app behind the share: its live status and whether it is "
            "answering. 'ready' is about the LINK, this is about the ORIGIN"
        )
    )
    hosted_only: bool = Field(description="(Lot 5) Stored lock: no local proxy route for this app")
    local_routes: Literal["absent", "present", "unknown"] = Field(
        description=(
            "(Lot 5) Observed from the proxy's live table, never inferred: absent = no "
            "local route serves the app; present = one still does; unknown = the table "
            "is unreadable, or the daemon is reachable without a token (always unknown "
            "then)"
        ),
    )
    forward_authorization: bool = Field(
        default=False,
        description=(
            "[deploy].forward_authorization: the app receives the browser's "
            "Authorization header on its hosted and cloud custom-domain URLs"
        ),
    )
    # ticket_url: a WP-HC TODO. Minting a cloud navigation ticket needs an
    # authenticated daemon → cloud call that does not exist yet, so the field is
    # simply absent in this WP rather than reserved as a null nobody can fill.


class ShareRemovedView(BaseModel):
    """Idempotent share deletion result (always HTTP 200).

    Removed is false when no share existed.
    """

    service_name: str = Field(description="Stable service name")
    removed: bool = Field(description="False = there was no share; the call was a no-op")


class DomainRequest(StrictRequestModel):
    """Certificate policy for a domain identified by the request path.

    ACME is tri-state: omitted/null preserves an existing policy and defaults to
    false on insert. Explicit false downgrades to the internal CA; true requests a
    public certificate and is refused when node ACME policy is disabled.
    """

    acme: bool | None = Field(
        default=None,
        description=(
            "true = request a public ACME certificate (409 domain.acme_disabled unless "
            "[proxy.acme].enabled); false = serve with this node's internal CA; "
            "omitted = keep the existing row's setting (false on a new domain)"
        ),
    )


class DomainView(BaseModel):
    """One bound custom domain, projected against the live proxy.

    `url` and `state` are computed per request, never stored — the same
    discipline `ShareView` follows: the row records the binding, the
    proxy decides whether it answers right now.
    """

    service_name: str = Field(description="Stable service name the domain routes to")
    domain: str = Field(description="Case-folded bare DNS name, no trailing dot")
    acme: bool = Field(
        description="Public (ACME) certificate requested for this name",
    )
    kind: Literal["domain"] = Field(description="Row discriminator; only 'domain' exists today")
    created_at: datetime = Field(description="When the domain was first bound")
    url: str = Field(description="Absolute URL the domain will serve at (https://<domain>/)")
    state: Literal["ready", "withheld"] = Field(
        description=(
            "ready = the proxy serves this service now; withheld = the proxy is off, "
            "the app is not routed yet, or its edge_auth secret does not resolve"
        )
    )
    # The second, orthogonal axis: `state` says whether the ROUTE is
    # live, `cert_state` says what a browser will make of the certificate it
    # is served. Required rather than optional — every row has an answer, and
    # `internal` (this node's own CA) is the honest one for the default
    # `acme=false` row, not an absence.
    cert_state: CertState = Field(
        description=(
            "'internal' = this node's CA (the default); 'disabled' = the row asks "
            "for a public certificate but [proxy.acme] is off; 'pending' = no leaf "
            "on disk yet (issuing, or failing — see the acme_http_port doctor row); "
            "'issued' | 'expired' = a leaf exists"
        )
    )
    cert_not_after: datetime | None = Field(
        default=None,
        description="Expiry of the public leaf; null unless cert_state is issued/expired",
    )


class DomainListView(BaseModel):
    """All domains bound to a service; empty when none are configured.

    Only a missing service returns 404.
    """

    service_name: str = Field(description="Stable service name")
    domains: list[DomainView] = Field(
        default_factory=list, description="Bound domains, sorted by name"
    )


class DomainSetView(DomainView):
    """A domain and whether this call created it.

    An unchanged re-PUT returns HTTP 200 with created=false and emits no event.
    """

    created: bool = Field(description="True when this call bound the name; false on a re-PUT")


class DomainRemovedView(BaseModel):
    """`DELETE /api/services/{name}/domains/{domain}` — idempotent, always 200.

    `removed=False` is the no-op case (the `ShareRemovedView` rule): the
    name was never bound to this service, or was already released.
    """

    service_name: str = Field(description="Stable service name")
    domain: str = Field(description="The folded domain the call addressed")
    removed: bool = Field(description="False = nothing was bound; the call was a no-op")


class HookRequest(StrictRequestModel):
    """`POST /api/services/{name}/hooks`: declare an inbound webhook.

    No secret reference on the API: the key is imposed as
    `NERDIT_HOOK_<NAME>_SECRET`, set with `POST /api/secrets/{service}`.
    """

    name: str = Field(
        description=(
            "Lowercase letters, digits and '-', 1-32 characters, starting with a letter; "
            "maps to NERDIT_HOOK_<NAME>_SECRET / _URL"
        )
    )
    sender: str = Field(description="'stripe' | 'github' | 'slack' | 'hmac' | 'none'")
    path: str = Field(description="App path the delivery is POSTed to, e.g. /webhooks/stripe")
    header: str | None = Field(
        default=None, description="sender 'hmac' only (required): the signature header name"
    )
    prefix: str | None = Field(
        default=None,
        description="sender 'hmac' only: text before the lowercase hex digest, e.g. 'sha256='",
    )


class HookView(BaseModel):
    """One live inbound webhook; `url` and `secret_key` are computed per request."""

    name: str = Field(description="Hook name, unique per service")
    sender: str = Field(description="'stripe' | 'github' | 'slack' | 'hmac' | 'none'")
    path: str = Field(description="App path the delivery is POSTed to")
    header: str | None = Field(description="sender 'hmac' only: the signature header name")
    prefix: str | None = Field(description="sender 'hmac' only: the digest prefix")
    secret_ref: str | None = Field(
        description="The ${secrets.X} reference the signature is checked with; null for 'none'"
    )
    secret_key: str | None = Field(
        description=(
            "The S2 key to set with POST /api/secrets/{service} (the provider's signing "
            "secret); null for sender 'none' or a nerdit.toml hook naming its own ${secrets.X}"
        )
    )
    url: str | None = Field(
        description=(
            "Webhook URL to paste at the provider; null until the machine is linked and "
            "<app>--<slug> fits one DNS label. It only answers once the app is shared"
        )
    )
    created_at: datetime = Field(description="When the hook was first added")


class HookListView(BaseModel):
    """A service's live hooks; empty when none. Only a missing service is a 404."""

    service_name: str = Field(description="Stable service name")
    hooks: list[HookView] = Field(default_factory=list, description="Live hooks, sorted by name")


class HookSetView(HookView):
    """A hook and whether this call created it (a replay answers created=false)."""

    created: bool = Field(description="True when this call added the hook; false on a replay")


class HookRemovedView(BaseModel):
    """`DELETE /api/services/{name}/hooks/{hook}`: idempotent, always 200."""

    service_name: str = Field(description="Stable service name")
    name: str = Field(description="The hook name the call addressed")
    removed: bool = Field(description="False = no live hook of that name; the call was a no-op")
