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
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call
from nerdit.mcp.transport import _request_client


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
        name: str,
        access: str = "private",
        consent: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        """Use when: you need a URL you can open (public_url is LAN-only).

        Exposes the app at ``https://<name>--<slug>.<domain>/`` via the cloud
        link. ``access='private'`` (default) opens only for signed-in owners of this
        node in the Nerdit console — works on every linked node, nothing to
        configure. ``access='public'`` makes the URL world-reachable: it needs a
        Pro-entitled account (else 409 ``share.not_entitled``) AND either a
        ``[deploy].edge_auth`` on the app or ``consent=true`` (else 409
        ``share.unprotected``) — public means public. The node must be linked
        (409 ``share.link_required``). Returns ``{access, url, state}``; ``state``
        is ``ready`` | ``link_down`` | ``not_entitled``. The same URL then appears
        in ``get_service`` / ``list_routes`` under ``public_urls`` (kind
        ``hosted``). Idempotent; a key is auto-generated if omitted.

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
        name: str,
        idempotency_key: str | None = None,
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
        name: str,
        domain: str,
        acme: bool | None = None,
        idempotency_key: str | None = None,
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
        unless the daemon has ``[proxy.acme].enabled``. Leaving ``acme`` OUT
        keeps whatever the domain already has (``false`` on a new one), so
        re-running this tool to read the URL back never downgrades an issued
        public certificate; pass ``acme=False`` to downgrade on purpose.

        ``domain`` must be a bare, lowercase DNS name (case and a trailing dot
        are folded server-side): no scheme, no path, no port, no wildcard, no IP
        literal, no IDN/punycode, at least two labels, and never a name under
        this node's own hostname / ``base_domain`` / extra hostnames. Refusals:
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
        ``public_urls`` (kind ``domain``). Idempotent; a key is auto-generated
        if omitted.
        """
        return await _add_domain_impl(
            _request_client(),
            name=name,
            domain=domain,
            acme=acme,
            idempotency_key=idempotency_key,
        )

async def remove_domain(
        name: str,
        domain: str,
        idempotency_key: str | None = None,
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
