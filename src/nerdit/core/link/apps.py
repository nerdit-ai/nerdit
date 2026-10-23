"""Validate each hosted request against one committed service incarnation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from typing import Protocol
from urllib.parse import urlsplit

from nerdit.core.link.client import AppTarget
from nerdit.core.link.hosted import hosted_host, hosted_label_fits, is_service_name

logger = logging.getLogger("nerdit.link.apps")


class ShareLookup(Protocol):
    """Atomic hosted routing query used by both stream admission and revocation."""

    def resolve_shared_app(  # noqa: PLR0913 - the routing facts of one stream
        self,
        node_id: str,
        service_name: str,
        authority: str,
        job_id: str | None,
        access: str | None,
        legacy_host: str | None,
        nodes_domain: str | None,
    ) -> Awaitable[tuple[str, int, str, bool] | None]:
        """Return the current job, live port, admitted audience and opt-in, or refuse."""


class AppStreamResolver:
    """Preserve the validated external authority; never route by service name alone."""

    __slots__ = ("_domain", "_node_id", "_queries", "_slug")

    def __init__(
        self,
        queries: ShareLookup,
        *,
        node_id: str,
        slug: str | None,
        nodes_base_domain: str | None,
    ) -> None:
        self._queries = queries
        self._node_id = node_id
        self._slug = slug
        self._domain = nodes_base_domain

    async def resolve(
        self,
        service_name: str,
        authority: str | None = None,
        job_id: str | None = None,
        access: str | None = None,
    ) -> AppTarget | None:
        """Refuse malformed, stale, unshared or foreign authorities before dialing."""
        if not is_service_name(service_name):
            logger.debug("App stream refused: invalid service label")
            return None
        if authority is None or any(c.isspace() for c in authority):
            return None
        try:
            parsed = urlsplit(f"//{authority}")
            host, port = parsed.hostname, parsed.port
        except ValueError:
            return None
        if (
            host is None
            or len(host) > 253
            or parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or not all(is_service_name(label) for label in host.split("."))
            or authority != host + (f":{port}" if port is not None else "")
            or port == 0
        ):
            return None
        if (job_id is not None and access is None) or access not in (None, "public", "private"):
            return None
        if job_id is not None and (not job_id or len(job_id) > 128):
            return None
        legacy_host = (
            hosted_host(service_name, self._slug, self._domain)
            if self._slug and self._domain and hosted_label_fits(service_name, self._slug)
            else None
        )
        result = await self._queries.resolve_shared_app(
            self._node_id, service_name, authority, job_id, access, legacy_host, self._domain
        )
        if result is None:
            return None
        admitted_job, port, audience, forward_authorization = result
        return AppTarget(
            service_name, port, authority, admitted_job, audience, forward_authorization
        )


__all__ = ["AppStreamResolver", "ShareLookup"]
