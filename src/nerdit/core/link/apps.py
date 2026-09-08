"""Resolve hosted app streams without exposing unshared services.

The daemon decides what is shared; the cloud decides who can access it.
Validate service names before DB reads. Unknown, unshared, non-app, and
endpoint-less services all return `None`, producing the same 404 without an
upstream connection. Re-read share state for every stream so unshare takes
effect on the next request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from nerdit.core.link.client import AppTarget
from nerdit.core.link.hosted import hosted_host, is_service_name
from nerdit.db.enums import JobKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nerdit.db.rows import Job, ServiceEndpoint, ServiceShare

logger = logging.getLogger("nerdit.link.apps")


class ShareLookup(Protocol):
    """The three reads `AppStreamResolver` needs from `Queries`.

    A Protocol rather than the concrete `Queries` façade so this leaf keeps a
    narrow, obvious contract — and so a test can hand it a spy that counts
    database calls, which is how "an invalid name never reaches the DB" is
    proved rather than asserted.
    """

    async def get_service_share(self, service_name: str) -> ServiceShare | None:
        """The share row, or `None` when the service is not shared."""

    async def get_service_by_name(self, service_name: str) -> Job | None:
        """The live workload row for `service_name`, if there is one."""

    async def get_service_endpoint(self, service_name: str) -> ServiceEndpoint | None:
        """The service's host-port reservation, if it has one."""


class AppStreamResolver:
    """Turn an `x-nerdit-app` value into a dialable target, or into `None`.

    Installed on `nerdit.core.link.client.MuxContext.resolve_app` by
    `bootstrap.build_link_manager`, and only when the node is linked with a
    slug **and** a known `nodes_base_domain` — the hosted authority is
    computed, never guessed (D-P26-H5).
    """

    __slots__ = ("_domain", "_queries", "_slug")

    def __init__(self, queries: ShareLookup, *, slug: str, nodes_base_domain: str) -> None:
        self._queries = queries
        self._slug = slug
        self._domain = nodes_base_domain

    async def resolve(self, service_name: str) -> AppTarget | None:
        """Resolve one app stream. `None` is the fail-closed answer."""
        if not is_service_name(service_name):
            # Attacker-chosen input: refused before any I/O, and NOT logged
            # verbatim — the value is neither a validated label nor bounded.
            logger.debug("App stream refused: the requested name is not a service name")
            return None

        # From here the name is a validated DNS label, so it is safe to log.
        if await self._queries.get_service_share(service_name) is None:
            logger.debug("App stream refused for %r: not shared", service_name)
            return None

        job = await self._queries.get_service_by_name(service_name)
        if job is None or job.kind != JobKind.service:
            # Models and databases have no HTTP app semantics on the hosted
            # path (plan §8); the share route refuses to create such a row at
            # all, and this is the defence in depth for a row that predates a
            # kind change or was written by hand.
            logger.debug("App stream refused for %r: not a service workload", service_name)
            return None

        endpoint = await self._queries.get_service_endpoint(service_name)
        if endpoint is None:
            # Shared but never launched: there is no port to dial. A 404 is the
            # honest answer, and it is the same 404 as every other refusal.
            logger.debug("App stream refused for %r: no endpoint", service_name)
            return None

        return AppTarget(
            service_name=service_name,
            port=endpoint.live_port,
            host=hosted_host(service_name, self._slug, self._domain),
        )


__all__ = ["AppStreamResolver", "ShareLookup"]
