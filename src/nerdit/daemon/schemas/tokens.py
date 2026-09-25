"""Request/response schemas for the `/tokens` scoped-token surface."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, computed_field, field_validator

from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.db.enums import TokenRole
from nerdit.db.rows import ApiToken
from nerdit.utils.names import DNS_LABEL_RE

# One year, in seconds — the ceiling on both `expires_in_s` and the
# `[security].token_default_ttl_s` site policy.
MAX_TOKEN_TTL_S = 31_536_000
MIN_TOKEN_TTL_S = 60


class TokenCreateRequest(StrictRequestModel):
    """Request body for `POST /tokens` (admin-only).

    The raw secret is generated server-side and never accepted from the client;
    the body only carries the label, role, and optional quotas.
    """

    name: str = Field(description="Human-readable label for the token")
    role: TokenRole = Field(
        default=TokenRole.submitter, description="Authorization role granted by this token"
    )
    max_gpus: int | None = Field(
        default=None, ge=0, description="Cumulative GPU cap across active jobs (None = uncapped)"
    )
    max_concurrent_jobs: int | None = Field(
        default=None, ge=0, description="Concurrent active-job cap (None = uncapped)"
    )
    # The declared default stays a plain `None`: the
    # omitted-vs-explicit-null distinction is read off `model_fields_set` in
    # the route, never off the value. A sentinel default would leak into the
    # drift-guarded OpenAPI schema as a bogus default.
    expires_in_s: int | None = Field(
        default=None,
        ge=MIN_TOKEN_TTL_S,
        le=MAX_TOKEN_TTL_S,
        description=(
            "Seconds until this token expires; null = never; omitted = "
            "[security].token_default_ttl_s"
        ),
    )
    scope_services: list[str] | None = Field(
        default=None,
        max_length=32,
        description="Service names this token may write to; null = unscoped",
    )

    @field_validator("scope_services")
    @classmethod
    def _check_scope_services(cls, value: list[str] | None) -> list[str] | None:
        """Validate and deduplicate DNS labels; reject empty scope lists."""
        if value is None:
            return None
        if not value:
            raise ValueError(
                "scope_services must not be empty: use null for an unscoped token, "
                "or list at least one service name."
            )
        deduped: list[str] = []
        for entry in value:
            if not DNS_LABEL_RE.fullmatch(entry):
                raise ValueError(
                    f"Invalid scope_services entry '{entry}': must be a DNS label "
                    "(lowercase letters, digits and '-', 1-63 chars, starting and "
                    "ending with a letter or digit)."
                )
            if entry not in deduped:
                deduped.append(entry)
        return deduped


class TokenView(BaseModel):
    """Public projection of an API token — never exposes the hash or plaintext.

    Returned by `GET /tokens` and embedded in the creation response. The
    `token_hash` and raw secret are deliberately absent.
    """

    id: str
    name: str
    role: TokenRole
    max_gpus: int | None = None
    max_concurrent_jobs: int | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    revoked: bool = False
    expires_at: datetime | None = Field(
        default=None, description="Absolute expiry instant (UTC); null = never expires"
    )
    scope_services: list[str] | None = Field(
        default=None, description="Service names this token may write to; null = unscoped"
    )

    @classmethod
    def from_token(cls, token: ApiToken) -> TokenView:
        """Build a hash-free view from a stored `ApiToken`."""
        return cls(
            id=token.id,
            name=token.name,
            role=token.role,
            max_gpus=token.max_gpus,
            max_concurrent_jobs=token.max_concurrent_jobs,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            revoked=token.revoked,
            expires_at=token.expires_at,
            scope_services=token.scope_services,
        )


def seconds_until(expires_at: datetime | None, now: datetime | None = None) -> int | None:
    """Return nonnegative seconds until expiry, or None if it never expires.

    Args:
        expires_at: Expiration timestamp.
        now: Caller clock; defaults to wall time. License views pass their holder's
            clock so expiry and state agree.
    """
    if expires_at is None:
        return None
    moment = datetime.now(UTC) if now is None else now
    return max(0, int((expires_at - moment).total_seconds()))


class TokenSelfView(BaseModel):
    """The caller's token, including synthetic local/global admin identities.

    Unlike TokenView, id and created_at may be null because sentinel principals
    have no api_tokens row.
    """

    id: str | None = Field(
        default=None, description="Token row id; null for the legacy/local sentinel principals"
    )
    name: str
    role: TokenRole
    max_gpus: int | None = None
    max_concurrent_jobs: int | None = None
    created_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked: bool = False
    expires_at: datetime | None = Field(
        default=None, description="Absolute expiry instant (UTC); null = never expires"
    )
    scope_services: list[str] | None = Field(
        default=None, description="Service names this token may write to; null = unscoped"
    )
    rotatable: bool = Field(
        description=(
            "Whether this principal has an api_tokens row to rotate. False for the "
            "legacy global token and the token=None local bypass. Note a readonly "
            "token reads true here but is still denied by the coarse write gate."
        )
    )

    # `computed_field` over `property` is pydantic's documented shape and the
    # ignore is mypy's documented limitation with it (no pydantic plugin here);
    # a plain field set by the two constructors would be settable and could
    # silently drift from `expires_at`.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def expires_in_s(self) -> int | None:
        """Seconds of life left; null = never expires. Derived, never stored."""
        return seconds_until(self.expires_at)

    @classmethod
    def from_token(cls, token: ApiToken) -> TokenSelfView:
        """Build the self view of a real row (always rotatable)."""
        return cls(
            id=token.id,
            name=token.name,
            role=token.role,
            max_gpus=token.max_gpus,
            max_concurrent_jobs=token.max_concurrent_jobs,
            created_at=token.created_at,
            last_used_at=token.last_used_at,
            revoked=token.revoked,
            expires_at=token.expires_at,
            scope_services=token.scope_services,
            rotatable=True,
        )

    @classmethod
    def synthetic(cls, *, name: str, role: TokenRole) -> TokenSelfView:
        """Build the local/global admin view, which has no token row and cannot rotate."""
        return cls(name=name, role=role, rotatable=False)


class TokenRotateRequest(StrictRequestModel):
    """Optional token-rotation request.

    Without extend, preserve expires_at exactly. A bare POST rotates without silently
    extending the minter's chosen lifetime.
    """

    extend: bool = Field(
        default=False,
        description="Also push expires_at forward (by expires_in_s, else the site default)",
    )
    expires_in_s: int | None = Field(
        default=None,
        ge=MIN_TOKEN_TTL_S,
        le=MAX_TOKEN_TTL_S,
        description=(
            "New lifetime in seconds; only read when extend=true. Omitted with "
            "extend=true uses [security].token_default_ttl_s."
        ),
    )


class TokenCreateResponse(TokenView):
    """Response for `POST /tokens` — carries the plaintext token, shown ONCE.

    The raw `token` value is returned only on the original creation response.
    It is never persisted (only its hash is) and never re-issued on an
    idempotent replay (`POST /tokens` is in the no-body-cache set).
    """

    token: str = Field(description="The raw token value — shown exactly once, store it now")
