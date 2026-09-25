"""Schemas for linking nodes and accepting cloud pushes.

Validate relay_url with LinkSettings before a cloud exchange can consume its
grant. Code and key are secrets in transit: exclude them from repr, validation
echoes, logs, audits and persistence. Both grants share the claim route.

Device polling exposes only a grantless session selector; the authorizing
160-bit device_code never leaves the daemon process.
"""

from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import AwareDatetime, BaseModel, Field, ValidationError, field_validator

from nerdit.config.settings import LOOPBACK_HOSTS, LinkSettings
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.rows import LinkedProjectService, ServicePublicAddress


class PublicAddressPushRequest(ServicePublicAddress, StrictRequestModel):
    """Install a binding; activate only after the cloud can serve its route."""

    activate: bool = False


class PublicAddressPushView(ServicePublicAddress):
    """Stored assignment and durable, monotonic routing activation."""

    url: str
    changed: bool
    active: bool


class LinkedProjectView(BaseModel):
    """Complete committed service discovery for allocation and retirement."""

    id: str
    name: str
    services: list[LinkedProjectService]


def _is_loopback(hostname: str) -> bool:
    """Accept plain HTTP only for LOOPBACK_HOSTS and RFC 6761 *.localhost names."""
    host = hostname.lower()
    return host in LOOPBACK_HOSTS or host.endswith(".localhost")


def validate_api_url(value: str) -> str:
    """Validate a credential-free HTTPS origin, allowing HTTP only for loopback.

    Reject whitespace, userinfo, query strings and fragments. Shared by claim and
    refresh requests so both accept the same console origins.
    """
    if value != value.strip() or any(ch.isspace() for ch in value):
        raise ValueError("api_url must not contain whitespace.")
    parts = urlsplit(value)
    hostname = parts.hostname
    if not hostname:
        raise ValueError("api_url must include a host.")
    try:
        parts.port  # noqa: B018 — the ACCESS is the parse (urlsplit defers it)
    except ValueError as exc:
        raise ValueError("api_url port must be a number in 1-65535.") from exc
    if parts.path not in ("", "/"):
        # A bare ORIGIN, by contract: the endpoint is always
        # `{origin}/api/link/…`, so a path buys nothing — and operators park
        # bearer tokens in URL paths (the webhook-URL practice), which would
        # otherwise ride into the audit row and httpx's INFO request log
        # (PR #115 round 3).
        raise ValueError("api_url must be a bare origin — no path.")
    if parts.scheme == "http":
        if not _is_loopback(hostname):
            raise ValueError(
                "api_url must use https — plain http is accepted only for "
                "loopback hosts (localhost, *.localhost, 127.0.0.1, ::1)."
            )
    elif parts.scheme != "https":
        # No interpolation — a secret pasted as the scheme must not echo.
        raise ValueError("api_url scheme is not supported — use 'https'.")
    if parts.username or parts.password:
        raise ValueError(
            "api_url must not embed credentials (userinfo) — the link code "
            "is the only thing that authenticates the claim."
        )
    if parts.query:
        raise ValueError("api_url must not carry a query string.")
    if parts.fragment:
        raise ValueError("api_url must not carry a fragment.")
    return value


class LinkClaimRequest(StrictRequestModel):
    """Exchange exactly one console code or pre-auth key for a node identity.

    The route rejects missing/conflicting grants with a redacted structured error.
    Both grants share locking, validation and staging; only the cloud exchange
    differs.
    """

    # `repr=False`: the plaintext code must never appear in a repr, a pydantic
    # error echo, or anything a log/traceback renders. The route's audit params
    # are built by hand and never include it either — this is the structural
    # half of the same never-leak contract.
    #
    # Optional since P34 only because `key` is the other half of the
    # exactly-one-of; the route refuses a body carrying neither, so "optional"
    # here never means "a claim with no grant".
    code: str | None = Field(default=None, min_length=1, max_length=128, repr=False)

    #: X16 C8's pre-auth key: `nk_` + 32 Crockford symbols, 160 bits, machine
    #: handled end to end. Same `repr=False` custody as `code` — this is the
    #: value D-X16-O11 keeps off argv on every hop the installer controls, so a
    #: pydantic error echo or a traceback repr would undo the whole chain. The
    #: route never audits it, never logs it, and never reflects it in a refusal.
    key: str | None = Field(default=None, min_length=1, max_length=128, repr=False)

    @field_validator("key")
    @classmethod
    def _check_key(cls, value: str | None) -> str | None:
        """Validate the cloud-issued pre-auth key before contacting the cloud.

        Strip surrounding whitespace only; preserve the issued spelling. Errors never
        interpolate the submitted value.
        """
        if value is None:
            return None
        candidate = value.strip()
        # `nk_` + 32 symbols of Crockford base32 (no I, L, O or U), uppercase
        # body — the exact shape `valid_preauth_key` accepts, restated rather
        # than imported because this tree vendors no cloud code.
        if not re.fullmatch(r"nk_[0-9A-HJKMNP-TV-Z]{32}", candidate):
            raise ValueError(
                "key does not look like a link key (it starts with 'nk_' "
                "followed by 32 uppercase letters and digits)."
            )
        return candidate

    @field_validator("code")
    @classmethod
    def _check_code(cls, value: str | None) -> str | None:
        """Normalize and validate the issued uppercase code alphabet.

        Uppercase letters, digits and hyphens keep codes disjoint from lowercase cloud
        error tokens, DNS slugs and UUIDs, preventing reflected credentials. Normalize
        lowercase input because Crockford base32 is case-insensitive.
        """
        if value is None:
            return None
        normalized = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9-]{6,128}", normalized):
            raise ValueError(
                "code does not look like a link code (letters, digits and "
                "hyphens, at least 6 characters)."
            )
        return normalized

    # Per-invocation only: the cloud origin is NEVER persisted (D5). The claim
    # contract has no daemon-side unlink call, so the daemon never needs it
    # again, and a config key would widen `[link]` for nothing.
    api_url: str = Field(min_length=1, max_length=512)
    # Supplied ⇒ persisted with the claim; omitted ⇒ the stored
    # `[link].relay_url` is used; neither ⇒ 422 before any cloud contact.
    relay_url: str | None = Field(default=None, max_length=512)
    # The same stage+commit flips `[link].enabled` (D6): an unlinked enabled
    # daemon is the documented ordinary state, so asking for a separate config
    # PUT + restart afterwards is pure friction. `--no-enable` opts out.
    enable: bool = True

    @field_validator("api_url")
    @classmethod
    def _check_api_url(cls, value: str) -> str:
        """Delegate to `validate_api_url` — the shared rule set."""
        return validate_api_url(value)

    @field_validator("relay_url")
    @classmethod
    def _check_relay_url(cls, value: str | None) -> str | None:
        """Validate through LinkSettings before a grant can be consumed.

        Reject invalid relay endpoints before cloud exchange or device approval, not
        after a node has been created.
        """
        if value is None:
            return None
        try:
            LinkSettings(relay_url=value)
        except ValidationError as exc:
            raise ValueError(exc.errors()[0]["msg"]) from exc
        return value


class LinkRefreshRequest(StrictRequestModel):
    """Refresh hosted metadata from a per-invocation console origin.

    The API URL is never persisted; this request carries no grant or secret.
    """

    api_url: str = Field(min_length=1, max_length=512)

    @field_validator("api_url")
    @classmethod
    def _check_api_url(cls, value: str) -> str:
        """The claim's rule set, shared verbatim (`validate_api_url`)."""
        return validate_api_url(value)


class LinkClaimView(BaseModel):
    """What a successful claim reports back (never the code, never the key)."""

    node_id: str
    slug: str
    relay_url: str
    enabled: bool
    verifier_fingerprint: str
    requires_restart: bool
    restart_keys: list[str]
    # (P26 WP-H) The cloud's hosted base domain, when this cloud sends one. A
    # pre-P26 cloud omits it and the daemon tolerates that — the field is then
    # whatever `[link].nodes_base_domain` already held (`None` on a fresh
    # node), and `nerdit link refresh` is the way to learn it later.
    nodes_base_domain: str | None = None
    #: Whether the claim also staged `[mcp].http_enabled = true` for the
    #: restart (a linked node serves the cloud's remote MCP gateway). `False`
    #: carries `mcp_skipped_reason` — never a secret, never a token.
    mcp_http_enabled: bool = False
    mcp_skipped_reason: str | None = None


class LinkRefreshView(BaseModel):
    """What `POST /api/link/refresh` reports: the domain, and whether it moved.

    `changed` is the actionable half. Every `[link]` key is restart-required,
    so a refresh that rewrote nothing must not bounce a healthy daemon to prove
    it — the CLI restarts on `changed` alone.
    """

    node_id: str
    slug: str
    nodes_base_domain: str
    changed: bool
    requires_restart: bool
    restart_keys: list[str]


class LinkUnlinkView(BaseModel):
    """What `DELETE /api/link` reports: what it found and what it tore down."""

    was_linked: bool
    tunnel_stopped: bool
    key_removed: bool
    # How many earlier failed wipes are STILL stuck on disk (a count, never
    # paths) — so "key deleted" can never conceal an older credential.
    pending_wipes: int
    requires_restart: bool


class EntitlementPushRequest(StrictRequestModel):
    """A cloud entitlement assertion ordered by an aware issued_at timestamp.

    Reject naive timestamps and unknown fields before the route runs.
    """

    hosted_public_entitled: bool
    issued_at: AwareDatetime


class EntitlementPushView(BaseModel):
    """Effective entitlement after applying TTL and ordering rules.

    Applied is false for out-of-order pushes. Changed reports an effective-value
    change, the condition for emitting a link.entitlement audit row and event.
    """

    hosted_public_entitled: bool
    received_at: datetime | None
    expires_at: datetime | None
    applied: bool
    changed: bool


# ---------------------------------------------------------------------------
# The device-code pair (P34 D1, cloud X16 C7)
#
# Two operations rather than one long-held request (D-P34-1): an attended
# approval takes up to 900 s, and a single blocking claim would hold
# `_LINK_MUTATION_LOCK` across a human, starving every unlink and every
# concurrent claim for the duration. So the *start* mints and parks, each
# *poll* is one lock-free cloud hop, and only the approved-commit branch
# re-acquires the lock.
# ---------------------------------------------------------------------------


class LinkDeviceStartRequest(StrictRequestModel):
    """Start device linking without carrying a grant.

    API URL is per-invocation and unpersisted. Relay URL, when supplied, and enable
    are persisted only when linking commits.
    """

    api_url: str = Field(min_length=1, max_length=512)
    relay_url: str | None = Field(default=None, max_length=512)
    enable: bool = True

    @field_validator("api_url")
    @classmethod
    def _check_api_url(cls, value: str) -> str:
        """The claim's rule set, shared verbatim (`validate_api_url`)."""
        return validate_api_url(value)

    @field_validator("relay_url")
    @classmethod
    def _check_relay_url(cls, value: str | None) -> str | None:
        """Validate through LinkSettings before a grant can be consumed.

        Reject invalid relay endpoints before cloud exchange or device approval, not
        after a node has been created.
        """
        if value is None:
            return None
        try:
            LinkSettings(relay_url=value)
        except ValidationError as exc:
            raise ValueError(exc.errors()[0]["msg"]) from exc
        return value


class LinkDevicePollRequest(StrictRequestModel):
    """Select the pending device-link session without authorizing it.

    The daemon keeps device_code private. A mismatched session returns
    409 link.device_superseded so concurrent terminals cannot poll each other's
    flow.
    """

    session: str = Field(min_length=1, max_length=64)


class LinkDeviceStartView(BaseModel):
    """Device-link instructions and daemon-local machine facts for the CLI.

    Never expose device_code. Machine facts describe the daemon being linked,
    which may be remote from the CLI.
    """

    #: The opaque slot selector (`LinkDevicePollRequest`). Non-secret by
    #: construction: it grants nothing without the daemon-held `device_code`.
    session: str
    #: `XXXX-XXXX-XXXX` — printed large for the retype path. The *cloud*
    #: forgives case, hyphens and the three Crockford confusions on entry; the
    #: daemon prints the canonical grouping and implements no forgiveness of
    #: its own.
    user_code: str
    verification_uri: str
    #: The one-click address, carrying the user code in a **fragment**
    #: (D-X16-O24) — a fragment never reaches a server, so this URL cannot
    #: deposit the code in an access log, a CDN log, a `Referer` header or a
    #: shared-link history the way `?user_code=` would.
    verification_uri_complete: str
    #: The full 64 hex chars of `sha256(verifier)`. The CLI truncates to 8 to
    #: match the console's `FINGERPRINT_PREVIEW_LENGTH`; the full value is
    #: returned so the truncation is one decision made in one place.
    credential_fingerprint: str
    hostname: str | None
    daemon_version: str
    os: str
    expires_in: int
    interval: int


class LinkDevicePollView(BaseModel):
    """Device-link status: waiting, slow down or linked.

    Nonterminal responses carry interval; linked responses include all claim-result
    fields from the shared commit path.
    """

    #: `pending` and `slow_down` map RFC 8628's `authorization_pending`
    #: and `slow_down` — the daemon renames the first to the shorter local
    #: token and passes the cloud's advertised `interval` through untouched,
    #: because cadence is the cloud's to tune without a daemon release.
    status: str
    interval: int | None = None
    node_id: str | None = None
    slug: str | None = None
    relay_url: str | None = None
    enabled: bool | None = None
    verifier_fingerprint: str | None = None
    nodes_base_domain: str | None = None
    #: Every `[link]` key is restart-keyed, so a `linked` answer is always
    #: `True`; a pending one changed nothing and is always `False`.
    requires_restart: bool = False
    restart_keys: list[str] = Field(default_factory=list)
    mcp_http_enabled: bool = False
    mcp_skipped_reason: str | None = None


#: (P33 D-GH-3/D-GH-6) One canonical repo: `owner/name`, lower-case. GitHub
#: logins are alphanumeric plus `-`; repo names add `.` and `_`. Anything
#: wider is a malformed push, not a repo the daemon could ever match.
_REPO_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38})/[a-z0-9._-]{1,100}$")

#: (P33 D-GH-2, D4) The most repos the mirror will resolve per installation.
#: The push DEGRADES to this cap instead of 422-ing: an org-wide App with more
#: repos than the cap still mirrors its first `_REPOS_CAP`. A sane bound that
#: keeps one hostile push from unbounded work while never breaking a real one.
_REPOS_CAP = 1000


class GithubTokenPushRequest(StrictRequestModel):
    """`PUT /api/link/github-token` — one installation token push.

    `token` is a **secret in transit**: `repr=False` here and `"token"`
    in the error handler's `_SECRET_INPUT_FIELDS` together keep it out of a
    422 echo, and the route never puts it in a log, an audit row or a response.
    `issued_at` and `expires_at` are `pydantic.AwareDatetime` for
    the same reason as the entitlement push: one is the ordering key and the
    other is compared against the daemon's UTC clock on every read.
    """

    installation_id: int = Field(ge=1)
    token: str = Field(min_length=1, max_length=4096, repr=False)
    expires_at: AwareDatetime
    issued_at: AwareDatetime
    repos: list[str]

    @field_validator("repos")
    @classmethod
    def _canonical_repos(cls, value: list[str]) -> list[str]:
        """Canonicalize, deduplicate and cap valid owner/name entries.

        Drop malformed entries instead of rejecting the whole push. Unmirrored repos
        fail closed: GitWatch ignores them and git deploy returns a clear refusal.
        """
        cleaned: list[str] = []
        for item in value:
            repo = item.strip().lower().removesuffix(".git")
            if not _REPO_RE.match(repo):
                continue
            if repo not in cleaned:
                cleaned.append(repo)
            if len(cleaned) >= _REPOS_CAP:
                break
        return cleaned


class GithubTokenPushView(BaseModel):
    """What the daemon reports back about one token push.

    `applied` is `False` for a push ignored as out of order or as landing
    after the offline edge. `expires_at`/`repos_count` describe the
    installation's EFFECTIVE entry after the push; when nothing is live
    `expires_at` echoes the push's own stamp and `repos_count` is `0`
    (the cross-repo contract says `str`, never null). No token, no names.
    """

    applied: bool
    installation_id: int
    expires_at: datetime
    repos_count: int


#: A full commit id. GitHub's push webhook always carries the 40-hex form;
#: an abbreviated sha would make the `(service, sha)` dedupe ambiguous.
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REFS_HEADS = "refs/heads/"


class GitNudgeRequest(StrictRequestModel):
    """`POST /api/link/git-nudge` — one push-to-deploy nudge.

    `repo` is normalised to the canonical lower-case `owner/name` and
    `ref` to a short branch name (`refs/heads/main` → `main`) so the
    controller's match is on the same identity the deploy recorded (D-GH-6).
    `repo_id` is carried for the cloud's benefit and otherwise opaque.
    `sha` is a hint (the daemon clones the recorded ref itself), but it is
    the dedupe key, so it must be the full commit id.
    """

    repo: str = Field(min_length=3, max_length=200)
    repo_id: int | None = Field(default=None, ge=1)
    ref: str = Field(min_length=1, max_length=255)
    sha: str = Field(min_length=40, max_length=40)

    @field_validator("repo")
    @classmethod
    def _canonical_repo(cls, value: str) -> str:
        repo = value.strip().lower().removesuffix(".git")
        if not _REPO_RE.match(repo):
            raise ValueError("repo must be canonical 'owner/name' (lower-case)")
        return repo

    @field_validator("ref")
    @classmethod
    def _short_ref(cls, value: str) -> str:
        ref = value.strip()
        if ref.startswith(_REFS_HEADS):
            ref = ref[len(_REFS_HEADS) :]
        if not ref or ref.startswith("-") or any(c.isspace() for c in ref):
            raise ValueError("ref must be a branch name")
        return ref

    @field_validator("sha")
    @classmethod
    def _full_sha(cls, value: str) -> str:
        sha = value.strip().lower()
        if not _SHA_RE.match(sha):
            raise ValueError("sha must be a 40-hex commit id")
        return sha


class GitNudgeView(BaseModel):
    """What one nudge did, by service name (the `202` body, D-GH-4).

    `matched`: `auto_deploy` services whose immediate redeploy poll was
    started; `ignored`: matched the source but opted out (recorded as
    `gitwatch.nudge_ignored`); `deduped`: already nudged for this sha.
    """

    matched: list[str]
    ignored: list[str]
    deduped: list[str]
