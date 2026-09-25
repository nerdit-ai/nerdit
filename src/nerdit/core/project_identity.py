"""Project identity: name grammar, id minting, the label rule and the qualified parser.

Stdlib only, on purpose: ``db/database.py`` imports :func:`mint_project_id`
for the P40a backfill, so this module must never import ``nerdit.db`` (or
anything that does) or the daemon fails to import itself.

The label rule (D-P40-2) is the one place a ``(project, environment,
service)`` triple becomes the ``jobs.service_name`` string that keys every
volume, Caddy ``@id``, image repo, hosted URL and secret file. The daemon
never parses a label back; the DB maps label -> row -> project.
"""

from __future__ import annotations

import base64
import re
import secrets

from nerdit.utils.names import DNS_LABEL_RE

#: The environment every phase-1 row is written with (D-P40-13: the column
#: exists now, phase 3 adds other values).
PRODUCTION = "production"

#: The service name a bare project name means (D-P40-14).
DEFAULT_SERVICE = "web"

PROJECT_ID_RE = re.compile(r"prj_[a-z2-7]{16}\Z")
PROJECT_DELEGATION_HEADER = "x-nerdit-project-id"
PROJECT_MCP_PATH = "/api/project-mcp"
# Builds share the node's image namespace, so apply needs a node grant until
# the builder can isolate project inputs and previously built images.
PROJECT_DELEGATION_TOOLS = frozenset(
    {
        "get_project",
        "project_logs",
        "diagnose_project",
        "set_variable",
        "resolve_variables",
    }
)

# D-P40-14: DNS-label grammar, <=40 chars, never containing ``--``. The
# double dash is the label separator (D-P40-2), so a name carrying one would
# make ``<service>--<project>`` ambiguous to a human reader. The lookahead
# is the cheapest way to say "no ``--`` anywhere" inside one pattern.
PROJECT_NAME_RE = re.compile(r"^(?!.*--)[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?$")

#: Same grammar as a project name, <=20 chars (D-P40-14).
SERVICE_NAME_RE = re.compile(r"^(?!.*--)[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?$")


def mint_project_id() -> str:
    """Return a fresh project id: ``prj_`` + 16 lowercase RFC 4648 base32 chars.

    Ten random bytes (80 bits) encode to exactly 16 base32 characters with no
    padding, so the id is fixed-width and safe in a filename or URL path
    (D-P40-8 keys the project secret file on it).
    """
    return "prj_" + base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower()


def service_label(project: str, environment: str, service: str) -> str:
    """Compose the ``jobs.service_name`` label for a triple (D-P40-2).

    ``(project, production, web)`` is the project name itself, so migrated
    rows keep their legacy label verbatim. Any other production service is
    ``<service>--<project>``; a non-production environment (phase 3, never
    passed in phase 1) is ``<service>--<environment>--<project>`` -- composed
    right-anchored so the project name always ends the label.

    Args:
        project: The project name.
        environment: The environment name; only ``production`` in phase 1.
        service: The service name within the project.

    Returns:
        The label, guaranteed to satisfy the create-request DNS-label pattern.

    Raises:
        ValueError: When the composition exceeds 63 octets or otherwise fails
            the DNS-label pattern. A legacy project name may be 41-63 chars,
            so ``api--<60 chars>`` cannot be keyed anywhere; such a project
            cannot gain a second service until renamed (phase 2).
    """
    if environment == PRODUCTION:
        label = project if service == DEFAULT_SERVICE else f"{service}--{project}"
    else:
        label = f"{service}--{environment}--{project}"
    # Every downstream key (volume dir, secret file AAD, hosted label, ``@id``)
    # re-validates against the same 63-octet DNS-label rule.
    if not DNS_LABEL_RE.fullmatch(label):
        raise ValueError(f"composed label {label!r} is not a DNS label of at most 63 octets")
    return label


def parse_qualified(text: str) -> tuple[str, str, str]:
    """Split a qualified name into ``(project, environment, service)``.

    One segment (``asso``) means ``production``/``web``; two (``asso/web``)
    name the project and service in ``production``; three
    (``asso/staging/web``) name all of them. A middle segment other than
    ``production`` is accepted here and judged by the caller (D-P40-6: 422
    ``project.environment_unsupported``), and no segment is grammar-checked --
    D-P40-14 enforces the grammar only where a *new* name enters.

    Args:
        text: The qualified name, ``/``-separated.

    Returns:
        The ``(project, environment, service)`` triple.

    Raises:
        ValueError: On an empty segment or more than three segments.
    """
    parts = text.split("/")
    if len(parts) > 3 or any(not part for part in parts):
        raise ValueError(f"not a qualified name: {text!r}")
    if len(parts) == 1:
        return parts[0], PRODUCTION, DEFAULT_SERVICE
    if len(parts) == 2:
        return parts[0], PRODUCTION, parts[1]
    return parts[0], parts[1], parts[2]
