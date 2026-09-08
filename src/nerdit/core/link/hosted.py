"""Compose hosted-share URLs from configured claim metadata.

URLs use `https://<app>--<slug>.<nodes_base_domain>/`, with slug and base domain
from the cloud claim or refresh response. The combined app/slug is one DNS
label for the wildcard certificate and must fit 63 octets.

Service names may contain `--`: compose here, never split. Cloud resolution
matches registered slug suffixes. This dependency-light leaf is not re-exported
from the link package.
"""

from __future__ import annotations

import re

#: The separator that keeps `<app>--<slug>` a single DNS label (D-P26-H5).
HOSTED_SEPARATOR = "--"

#: RFC 1035 label ceiling. The combined `<app>--<slug>` must fit in one.
MAX_DNS_LABEL = 63

#: The service-name grammar, character for character the one
#: `ServiceCreateRequest.name` pins (`daemon/schemas/services.py`). Kept as a
#: local literal rather than imported so this leaf stays free of any
#: `daemon`-package import (a `core` → `daemon` edge would be a cycle);
#: `tests/test_hosted_names.py` pins the two patterns equal.
SERVICE_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def is_service_name(value: str) -> bool:
    """Is `value` a well-formed service name (and therefore a DNS label)?

    The cheap gate the app-stream resolver runs on the attacker-supplied
    `x-nerdit-app` header **before** any database read: anything that is not a
    service name cannot name a shared service, so it is refused without
    touching the DB.
    """
    return SERVICE_NAME_RE.fullmatch(value) is not None


def hosted_label(service_name: str, slug: str) -> str:
    """The single DNS label a shared service occupies: `<app>--<slug>`."""
    return f"{service_name}{HOSTED_SEPARATOR}{slug}"


def hosted_label_fits(service_name: str, slug: str) -> bool:
    """Does `<app>--<slug>` fit in one 63-octet DNS label?

    Checked at share time, not at request time: a row whose name can never be
    resolved would be a share that silently does not work.
    """
    return len(hosted_label(service_name, slug)) <= MAX_DNS_LABEL


def hosted_host(service_name: str, slug: str, nodes_base_domain: str) -> str:
    """The hosted authority: `<app>--<slug>.<nodes_base_domain>`.

    This is what the mux rewrites `Host` to before handing an app stream to
    the container, so the app builds absolute URLs against the name a browser
    actually typed rather than a loopback port.
    """
    return f"{hosted_label(service_name, slug)}.{nodes_base_domain}"


def hosted_url(service_name: str, slug: str, nodes_base_domain: str) -> str:
    """The advertised hosted URL: `https://<app>--<slug>.<domain>/`.

    Always `https` and always root-relative — the cloud edge terminates TLS
    and the app is served at the root of its own hostname, which is precisely
    the base-path class that path-mode proxying inflicts on remote nodes.
    """
    return f"https://{hosted_host(service_name, slug, nodes_base_domain)}/"
