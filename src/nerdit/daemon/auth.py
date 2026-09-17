"""Resolve bearer tokens to principals for authentication and authorization.

The middleware sets `request.state.principal`; consumers use
`current_principal` to fail closed when no identity is attached. The legacy
global token and token=None local bypass both resolve to admin principals.
Tokens use SHA-256 because they contain 256 bits of entropy, unlike passwords.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from starlette.requests import Request

from nerdit.daemon.errors import NerditError
from nerdit.db.models import TokenRole

if TYPE_CHECKING:
    from nerdit.db.models import Job

# Prefix that marks a raw scoped token (distinct from the legacy global token).
TOKEN_PREFIX = "nrd_"


@dataclass(frozen=True)
class Principal:
    """The authenticated identity behind a request.

    `token_id` is `None` for the legacy global token and the `token=None`
    bypass (neither is a row in `api_tokens`). `role` drives every coarse
    and fine-grained authorization decision. Quotas (`max_gpus`,
    `max_concurrent_jobs`) are enforced atomically at job submission (S4).
    """

    token_id: str | None
    name: str
    role: TokenRole
    max_gpus: int | None = None
    max_concurrent_jobs: int | None = None
    is_legacy_admin: bool = False
    # `None` = unscoped, today's behaviour. A `frozenset`
    # keeps the frozen dataclass hashable.
    scope_services: frozenset[str] | None = None
    # Carried, not just compared and dropped, so `/capabilities`
    # can project it without any I/O — the middleware already has the row.
    expires_at: datetime | None = None

    @property
    def is_admin(self) -> bool:
        """Whether this principal carries the `admin` role."""
        return self.role == TokenRole.admin

    def in_scope(self, service_name: str) -> bool:
        """Whether `service_name` falls inside this principal's scope.

        An unscoped principal (`scope_services is None`) is in scope for
        everything; an empty scope grants nothing (P25 D-P25-3 fail-closed).
        """
        return self.scope_services is None or service_name in self.scope_services


# (P27 WP-C1) Synthetic tunnel principals carry `token_id = "link:<node_id>"`
# — a stable ownership + audit identity that is deliberately NOT a row in
# `api_tokens`. Row-backed per-token quota reads (`count_active_jobs_public`
# is fail-closed: missing row => cap 0) must route around them; the daemon-wide
# caps (D-P20-2) still bind. The prefix is the single discriminator — colons
# cannot appear in real token ids (`utils/ids.py` mints alphanumerics only).
LINK_TOKEN_PREFIX = "link:"


def is_link_token_id(token_id: str | None) -> bool:
    """Whether `token_id` names a synthetic node-link tunnel principal."""
    return token_id is not None and token_id.startswith(LINK_TOKEN_PREFIX)


# The cloud's control-plane carrier. The relay injects the bearer
# and forwards request headers verbatim, so this header says "the cloud's own
# server-side code framed this stream", not "a user's browser did" — the same
# guarantee, and the same trust root, as the `x-nerdit-app` hosted-routing
# carrier (D-P26-H2). The separation is enforced on the CLOUD side: the gateway
# lists it in `REQUEST_HEADERS_NEVER_FORWARDED`, so no console session, MCP
# bearer or anonymous hosted visitor can carry it inbound; only the cloud's
# entitlement pusher sets it on a stream it frames itself.
CLOUD_CONTROL_HEADER = "x-nerdit-cloud-control"
CLOUD_CONTROL_ENTITLEMENT = "entitlement"
#: The GitHub installation-token push rides the same carrier with
#: its own value, so a pusher framed for one route cannot be replayed at the
#: other.
CLOUD_CONTROL_GITHUB_TOKEN = "github-token"  # noqa: S105 - a header value, not a secret
#: The push-to-deploy nudge, same carrier, its own value.
CLOUD_CONTROL_GIT_NUDGE = "git-nudge"


# Sentinel principals for the two permissive-by-default bypasses. Both map to
# `admin` so the frozen baseline (which uses the global token or no token at
# all) keeps full access.
LEGACY_ADMIN = Principal(
    token_id=None,
    name="legacy-admin",
    role=TokenRole.admin,
    is_legacy_admin=True,
)
LOCAL = Principal(
    token_id=None,
    name="local",
    role=TokenRole.admin,
    is_legacy_admin=False,
)
# Fail-closed default for a request that reached a route without the auth
# middleware attaching a principal. This should be unreachable in the real app
# (every `call_next` branch sets `request.state.principal` first), so it is
# a defense-in-depth guard: `readonly` is denied by `require_role` /
# `require_owner_or_admin`, so a future middleware reorder fails closed rather
# than granting admin. The two intentional bypasses use LOCAL/LEGACY_ADMIN,
# which the middleware assigns explicitly.
ANONYMOUS = Principal(
    token_id=None,
    name="anonymous",
    role=TokenRole.readonly,
)


def hash_token(raw: str) -> str:
    """Return the SHA-256 hex digest of a raw token (never store plaintext)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_token() -> str:
    """Generate a fresh scoped token: `nrd_` + 256 bits of URL-safe entropy."""
    return f"{TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


def current_principal(request: Request) -> Principal:
    """Return the request's principal, defaulting to `ANONYMOUS`.

    Defensive read: callers never assume the auth middleware ran, so a missing
    principal degrades to the **non-privileged** `ANONYMOUS` identity
    (fail-closed) rather than raising or granting admin. The middleware always
    attaches LOCAL/LEGACY_ADMIN/a scoped principal before a route runs, so this
    default is only a guard against a future middleware reorder.
    """
    return getattr(request.state, "principal", ANONYMOUS)


def require_role(request: Request, *allowed: TokenRole) -> Principal:
    """Ensure the request principal holds one of `allowed` roles.

    Returns the principal on success; raises `NerditError` (403
    `forbidden`) otherwise.
    """
    principal = current_principal(request)
    if principal.role not in allowed:
        wanted = ", ".join(role.value for role in allowed)
        raise NerditError(
            403,
            "forbidden",
            f"Role '{principal.role.value}' is not permitted for this operation.",
            hint=f"Requires one of: {wanted}.",
        )
    return principal


def require_cloud_principal(request: Request, control_value: str) -> Principal:
    """Require a synthetic link principal and the exact cloud-control header.

    Both conditions must pass: operator tokens lack tunnel identity, while proxied
    users lack the control assertion. Use one refusal code to avoid an oracle.
    This trusts the relay like bearer authentication does; the local entitlement
    mirror does not authorize anonymous traffic at the cloud edge.
    """
    principal = current_principal(request)
    header = request.headers.get(CLOUD_CONTROL_HEADER)
    if not is_link_token_id(principal.token_id) or header != control_value:
        raise NerditError(
            403,
            "link.cloud_principal_required",
            "This endpoint is written by the Nerdit cloud over the node link only.",
            hint="Only the cloud can confirm the linked account's access.",
        )
    return principal


def _scope_denial(principal: Principal, service_name: str | None) -> NerditError:
    """Build the 403 envelope for a scope miss.

    Naming the caller's own scope is not a disclosure: it is the token's own
    metadata, which the same principal can read back from `GET /tokens/self`.
    The *target* name is likewise already known to the caller — it is what they
    just asked for.
    """
    names = sorted(principal.scope_services or ())
    scope = ", ".join(names) if names else "(nothing)"
    shown = service_name if service_name is not None else "(unnamed)"
    return NerditError(
        403,
        "forbidden",
        f"This token's scope does not include service '{shown}'.",
        hint=f"Scoped to: {scope}.",
    )


def _owns(principal: Principal, owner_token_id: str | None) -> bool:
    """Test recorded ownership; a null owner never matches, even a null token ID."""
    return owner_token_id is not None and owner_token_id == principal.token_id


def _owner_or_admin_allows(
    principal: Principal,
    owner_token_id: str | None,
    scope_name: str | None,
) -> bool:
    """Allow admins, or owners whose row is within their narrowed scope.

    Nameless rows fail a narrowed scope. Do not use Principal.in_scope, which allows
    scope_name=None; raising and non-raising ownership checks must agree here.
    """
    if principal.role == TokenRole.admin:
        return True
    if not _owns(principal, owner_token_id):
        return False
    if principal.scope_services is None:
        return True
    return scope_name is not None and scope_name in principal.scope_services


def _check_owner(
    request: Request,
    owner_token_id: str | None,
    scope_name: str | None,
    denial: NerditError,
) -> Principal:
    """Require admin or non-null ownership plus scope membership.

    Use the shared predicate for row and workspace ownership. On failure, raise
    scope_denial for a matched owner outside scope, otherwise the caller's denial.
    """
    principal = current_principal(request)
    if _owner_or_admin_allows(principal, owner_token_id, scope_name):
        return principal
    if _owns(principal, owner_token_id):
        raise _scope_denial(principal, scope_name)
    raise denial


def may_manage_job(request: Request, job: Job) -> bool:
    """Return the caller's ownership/scope verdict without raising or revealing IDs.

    Use the same predicate as the write gate so projected actions match permissions.
    """
    return _owner_or_admin_allows(
        current_principal(request),
        getattr(job, "submitted_by_token", None),
        getattr(job, "service_name", None),
    )


def require_owner_or_admin(request: Request, job: Job) -> Principal:
    """Ensure the principal owns `job` or is an admin, and that `job` is in scope.

    NULL-owner jobs (pre-P1 rows and legacy/local submissions) are
    **admin-only**: a non-admin scoped token can never act on a job it does not
    explicitly own. A row with no `service_name` (legacy batch leftovers) is
    outside every scope — fail closed. Returns the principal on success; raises
    `NerditError` (403 `forbidden`) otherwise.
    """
    return _check_owner(
        request,
        getattr(job, "submitted_by_token", None),
        getattr(job, "service_name", None),
        NerditError(
            403,
            "forbidden",
            "You do not have permission to act on this job.",
            hint="Only the submitting token or an admin may manage this job.",
        ),
    )


def require_service_scope(request: Request, service_name: str) -> Principal:
    """Ensure a scoped token may act on `service_name` (D-P25-3 leg b).

    For routes whose target is a NAME, not a Job row (create paths, secrets,
    app-config, template deploys). Admin bypasses; an unscoped token passes;
    a scoped token must name the service. Composes WITH the role gate, it
    does not replace it.
    """
    principal = current_principal(request)
    if principal.role == TokenRole.admin:
        return principal
    if principal.in_scope(service_name):
        return principal
    raise _scope_denial(principal, service_name)


class QuotaExceeded(Exception):  # noqa: N818 — named by the P1 plan/contract
    """Raised when an atomic quota reservation would breach a token's caps (S4).

    Routes catch this and surface a 403 `quota_exceeded` envelope. `reason`
    is a short machine-friendly label (e.g. `max_concurrent_jobs`); `limit`
    and `current` carry the breached bound and the observed value.
    """

    def __init__(self, reason: str, limit: int | None = None, current: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.limit = limit
        self.current = current

    def to_error(self) -> NerditError:
        """Build the 403 `quota_exceeded` envelope for this breach."""
        return NerditError(
            403,
            "quota_exceeded",
            f"Quota exceeded: {self.reason}.",
            hint="Wait for active jobs to finish or request a higher quota.",
            limit=self.limit,
            current=self.current,
        )
