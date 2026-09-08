"""Schemas for hosted shares, custom domains and service public URLs.

The scalar public_url remains the LAN/proxy URL; public_urls lists all kinds.
State is a machine token: ready, link_down, not_entitled or withheld. Hosted
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
        description=(
            "Absolute URL; null only for a hosted entry on a node that has been "
            "unlinked since the share was set (the intent survives, the address does not)"
        )
    )
    kind: Literal["default", "hosted", "domain"] = Field(
        description=(
            "'default' = the proxy/LAN URL (== public_url); 'hosted' = the P26 share; "
            "'domain' = a custom domain served by this node's proxy"
        )
    )
    state: Literal["ready", "link_down", "not_entitled", "withheld"] = Field(
        description=(
            "Machine token: answers now / tunnel or link missing / public not entitled / "
            "the proxy is not serving this service yet (domain entries)"
        )
    )
    access: Literal["private", "public"] | None = Field(
        default=None, description="Hosted entries only; null on a default or domain entry"
    )
    domain: str | None = Field(
        default=None,
        description="Domain entries only: the bound name; null on every other kind",
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
            "'private' (default) opens only for signed-in owners of this node at "
            "the cloud edge; 'public' is world-reachable"
        ),
    )
    consent: bool = Field(
        default=False,
        description=(
            "Required (with, or instead of, a [deploy].edge_auth block) for "
            "access='public': public means public"
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
    access: Literal["private", "public"] = Field(description="Stored exposure mode")
    url: str | None = Field(
        description="The hosted URL, or null when this node has no slug/hosted domain"
    )
    state: Literal["ready", "link_down", "not_entitled"] = Field(
        description="ready = answers now; link_down = tunnel/link missing; "
        "not_entitled = public without the entitlement"
    )
    created_at: datetime = Field(description="When the share was first created")
    origin: ShareOrigin = Field(
        description=(
            "(P34) The app behind the share: its live status and whether it is "
            "answering. 'ready' is about the LINK, this is about the ORIGIN"
        )
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
