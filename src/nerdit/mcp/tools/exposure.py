"""Hosted-share MCP tool implementations.

The gap these close is the one a remote agent hits immediately: ``deploy_app``
returns a ``public_url`` that is a **LAN** address, which the agent — sitting on
the other side of the cloud tunnel — cannot open. ``share_service`` publishes
the same app at ``https://<name>--<slug>.<hosted-domain>/`` over the tunnel that
is already up, so the agent can finally see what it shipped.

``add_domain``/``remove_domain`` are the same gesture on a name the
operator owns: the node's own Caddy serves ``https://<domain>/`` directly, in
either proxy mode and at the root — no tunnel, no ``/<app>/`` prefix.

Thin loopback projections of ``PUT``/``DELETE /api/services/{name}/share`` and
``PUT``/``DELETE /api/services/{name}/domains/{domain}`` (Invariant #3): every
refusal — entitlement, consent, link state, workload kind, label length, domain
grammar, ownership, the node's ACME policy — is decided by the daemon and
surfaces as its structured envelope. Nothing is pre-checked here, and ``access``
stays a plain ``str`` (the daemon 422s anything outside the enum) so the tool
schema never drifts from the route's.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.tools._shared import IdempotencyKey
from nerdit.mcp.transport import _request_client

# Local alias: the exposure tools address an APP, never the wider
# "app, model or database" that the daemon-wide service lookup accepts — so an alias is
# not stretched. The kind gate lives on the WRITE routes only; the delete routes
# resolve and delete with no kind check, so the description attributes the
# refusal to the two writes rather than to all four tools.
ExposedApp = Annotated[
    str,
    Field(
        description="Name (or row id) of an existing deployed app — only kind "
        "'service' can be exposed: ``share_service``/``add_domain`` refuse a model "
        "or database with 422 ``*.kind_unsupported``, while the remove tools just "
        "find nothing to undo."
    ),
]


async def _share_service_impl(
    client: NerditClient,
    *,
    name: str,
    access: str = "private",
    consent: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Set an app's hosted share, auto-minting a key (the write-tool idiom)."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.set_share(name, access=access, consent=consent, idempotency_key=idempotency_key)
    )


async def _unshare_service_impl(
    client: NerditClient,
    *,
    name: str,
    idempotency_key: str | None = None,
) -> Any:
    """Remove an app's hosted share, auto-minting a key. Idempotent server-side."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.remove_share(name, idempotency_key=idempotency_key))


async def _add_domain_impl(
    client: NerditClient,
    *,
    name: str,
    domain: str,
    acme: bool | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Add a custom domain to an app, auto-minting a key (the write-tool idiom)."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.add_domain(name, domain, acme=acme, idempotency_key=idempotency_key))


async def _remove_domain_impl(
    client: NerditClient,
    *,
    name: str,
    domain: str,
    idempotency_key: str | None = None,
) -> Any:
    """Remove a custom domain, auto-minting a key. Idempotent server-side."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.remove_domain(name, domain, idempotency_key=idempotency_key))


# fmt: off
async def share_service(
        name: ExposedApp,
        access: Annotated[
            str,
            Field(
                description="Exposure mode: ``private`` (the default — signed-in owners "
                "of this node only) or ``public`` (world-reachable). Any other value is "
                "refused by the daemon with a 422."
            ),
        ] = "private",
        consent: Annotated[
            bool,
            Field(
                description="Acknowledges that a public share serves the app to anyone "
                "with the URL. Read only when ``access='public'``, where ``false`` (the "
                "default) is refused unless the app carries a ``[deploy].edge_auth`` "
                "block; ignored for a private share."
            ),
        ] = False,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Use when: you need a URL you can open (public_url is LAN-only).

        Exposes the app at ``https://<name>--<slug>.<domain>/`` via the cloud
        link. ``access='private'`` (default) opens only for signed-in owners of this
        node in the Nerdit console — works on every linked node, nothing to
        configure. ``access='public'`` makes the URL world-reachable: it needs an
        account in good standing (free during the public beta; else 409
        ``share.not_entitled``) AND either a
        ``[deploy].edge_auth`` on the app or ``consent=true`` (else 409
        ``share.unprotected``) — public means public. The node must be linked
        (409 ``share.link_required``). Returns ``{access, url, state}``; ``state``
        is ``ready`` | ``link_down`` | ``not_entitled``. The same URL then appears
        in ``get_service`` / ``list_routes`` under ``public_urls`` (kind
        ``hosted``).

        ``state`` and ``origin`` answer two DIFFERENT questions and you need
        both. ``state`` is about the LINK — is the hosted path provisioned.
        ``origin`` is about the APP behind it: ``{status, answers, hint}``, where
        ``answers`` is true only while the service is running. A share can be
        ``state: ready`` with ``origin.answers: false``, and then the URL returns
        ``share.not_shared`` (the hosted resolver refuses a service with no live
        origin) — which reads exactly like "never shared". When ``answers`` is
        false, ``origin.hint`` says what to do; call ``diagnose_service``.
        """
        return await _share_service_impl(
            _request_client(),
            name=name,
            access=access,
            consent=consent,
            idempotency_key=idempotency_key,
        )

async def unshare_service(
        name: ExposedApp,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Use when: an app should stop being reachable at its hosted URL.

        The URL stops answering on the next request (the tunnel re-reads the row
        per stream — there is no cache to wait out). Idempotent: removing a
        share that does not exist is a success returning ``removed: false``,
        never an error. The response is the removal receipt
        (``{service_name, removed}``), not a share view — read ``share_service``
        or ``get_service`` for the state of a share that still exists.
        """
        return await _unshare_service_impl(
            _request_client(),
            name=name,
            idempotency_key=idempotency_key,
        )

async def add_domain(
        name: ExposedApp,
        domain: Annotated[
            str,
            Field(
                description="Bare lowercase DNS name of at least two labels: no scheme, "
                "path, port, wildcard, IP literal or IDN/punycode. Case and one trailing "
                "dot are folded server-side, so ``App.Example.COM.`` and "
                "``app.example.com`` are one resource. A name at or under this node's own "
                "hostname, ``base_domain`` or ``extra_hostnames`` is reserved."
            ),
        ],
        acme: Annotated[
            bool | None,
            Field(
                description="Tri-state certificate policy. ``true`` = request a public "
                "ACME certificate; ``false`` = serve with this node's internal CA; omit "
                "(the default ``null``) = keep the row's stored value, ``false`` on a new "
                "domain — so a re-run never downgrades an issued certificate."
            ),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Serve a deployed app at a domain YOU own, from this node's own proxy (P26).

        Unlike ``share_service`` (Nerdit's domain, over the cloud tunnel), this
        is the node's embedded Caddy answering ``https://<domain>/`` directly —
        in either proxy mode, at the ROOT, so a frontend needs no ``/<app>/``
        base path. The one thing the daemon cannot do for you: point the
        domain's DNS (A/AAAA/CNAME) at this machine. By default the domain is
        served with the node's internal CA, so a client must run
        ``nerdit trust`` once. ``acme=True`` requests a public ACME
        certificate; it needs an ADMIN token (403 ``domain.acme_forbidden``,
        ``reason: acme_admin_only`` — the node has one ACME account whose rate
        limits every domain on it shares) and 409 ``domain.acme_disabled``
        unless the daemon has ``[proxy.acme].enabled``.

        Refusals:
        422 ``domain.invalid`` — the name broke exactly one grammar rule and the
        message names it (``empty``, ``whitespace``, ``not_bare``,
        ``has_port``, ``wildcard``, ``ip_literal``, ``idn``, ``single_label``,
        ``grammar``, ``reserved``); 409 ``domain.taken`` (one domain, one
        service — never silently re-pointed); 422 ``domain.kind_unsupported``
        (models and databases are not routed).

        Returns ``{domain, url, state, cert_state, created}``; ``state`` is
        ``ready`` | ``withheld`` (``withheld`` = the proxy is off, the app is
        not routed yet, or its ``[deploy].edge_auth`` secret does not resolve —
        the row is stored either way and answers once it is). ``cert_state`` is
        the orthogonal certificate fact: ``internal`` (this node's CA) |
        ``disabled`` (``acme`` asked for, ``[proxy.acme]`` off) | ``pending``
        (issuing, or failing — check ``doctor``) | ``issued`` | ``expired``.
        The URL then appears in ``get_service`` / ``list_routes`` under
        ``public_urls`` (kind ``domain``).
        """
        return await _add_domain_impl(
            _request_client(),
            name=name,
            domain=domain,
            acme=acme,
            idempotency_key=idempotency_key,
        )

async def remove_domain(
        name: ExposedApp,
        domain: Annotated[
            str,
            Field(
                description="The bound domain to unbind, case and one trailing dot folded "
                "as on add. Grammar is not ENFORCED on delete (a legacy or now-reserved "
                "binding stays removable): a name that is not bound simply removes nothing."
            ),
        ],
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Remove a custom domain; its route disappears on the next reconcile tick. Idempotent."""
        return await _remove_domain_impl(
            _request_client(),
            name=name,
            domain=domain,
            idempotency_key=idempotency_key,
        )
# fmt: on


TOOLS = (
    share_service,
    unshare_service,
    add_domain,
    remove_domain,
)
