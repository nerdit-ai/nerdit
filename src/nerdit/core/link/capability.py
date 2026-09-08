"""Mint fresh, memory-only submitter capabilities for each tunnel connection.

Tokens are never persisted, logged, audited, or cached. The role is fixed to
`submitter`; no caller may raise that ceiling. The link manager owns renewal
and reconnect timing.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

#: ADR-W2 look-ahead ceiling: the relay refuses a capability expiring more than
#: 15 minutes in the future. `[link].capability_ttl_s` is the primary gate
#: (`ge=60, le=900`); the check here is defense in depth for direct callers.
#: `MIN_CAPABILITY_TTL_S = 1` is deliberately looser than that config-layer
#: floor of 60 — the config API refuses a sub-minute TTL as an operator
#: mistake, while a direct caller (tests, a WP-C1 renewal edge) may legally
#: mint a 1-second capability.
MAX_CAPABILITY_TTL_S = 900
MIN_CAPABILITY_TTL_S = 1


@dataclass(frozen=True, slots=True, repr=False)
class MintedCapability:
    """One memory-only capability payload for a single tunnel session.

    Mirrors the wire shape modelled by `tests/node_link_frames.py`'s
    `CapabilityPayload` — including its redaction idiom: the token never
    appears in `repr`/`str`, so the object can be dropped into a log
    record, an audit payload or a doctor detail without leaking bearer
    material.
    """

    token: str
    expires_at: datetime
    role: Literal["submitter"]

    def __repr__(self) -> str:
        return (
            f"MintedCapability(token=<redacted>, expires_at={self.expires_at!r}, role='submitter')"
        )

    __str__ = __repr__


def mint_capability(ttl_s: int, *, clock: Callable[[], datetime] | None = None) -> MintedCapability:
    """Mint fresh capability material valid for `ttl_s` seconds.

    `clock` (default: UTC now) must return a timezone-aware datetime — a
    naive expiry cannot be compared against the relay's clock, the same rule
    the frames model enforces on the wire.
    """
    if not MIN_CAPABILITY_TTL_S <= ttl_s <= MAX_CAPABILITY_TTL_S:
        raise ValueError(
            f"capability ttl must be between {MIN_CAPABILITY_TTL_S} and "
            f"{MAX_CAPABILITY_TTL_S} seconds"
        )
    now = clock() if clock is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("capability expiry must include a timezone")
    return MintedCapability(
        token=secrets.token_urlsafe(32),
        expires_at=now + timedelta(seconds=ttl_s),
        role="submitter",
    )
