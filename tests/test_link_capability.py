"""Test memory-only, per-connection tunnel capabilities.

The submitter role is structural: minting takes no role argument, and the relay
rejects any other role. Pin the 15-minute TTL ceiling and validate minted values
against the independent hello models in `tests/node_link_frames.py`.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from nerdit.core.link.capability import MintedCapability, mint_capability
from nerdit.core.link.identity import proof_message
from tests.node_link_frames import CapabilityPayload


def _fixed_clock(moment: datetime):
    return lambda: moment


def test_mint_shape():
    cap = mint_capability(600)

    assert isinstance(cap.token, str)
    # ``secrets.token_urlsafe(32)`` — 43 characters of urlsafe base64.
    assert len(cap.token) >= 43
    assert cap.expires_at.tzinfo is not None
    assert cap.expires_at.utcoffset() == timedelta(0)
    assert cap.role == "submitter"


def test_expires_at_from_ttl():
    minted_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    cap = mint_capability(600, clock=_fixed_clock(minted_at))
    assert cap.expires_at == datetime(2026, 1, 2, 3, 14, 5, tzinfo=timezone.utc)


def test_tokens_fresh_per_call():
    # "regenerated per connect": nothing is cached, so two mints never agree.
    assert mint_capability(60).token != mint_capability(60).token


def test_ttl_bounds():
    for bad in (0, 901):
        with pytest.raises(ValueError):
            mint_capability(bad)
    # The ADR-W2 ceiling itself is legal; one second past it is not.
    assert mint_capability(1).role == "submitter"
    assert mint_capability(900).role == "submitter"


def test_naive_clock_rejected():
    naive = datetime(2026, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        mint_capability(600, clock=_fixed_clock(naive))


def test_repr_and_str_redact_token():
    cap = mint_capability(600)
    for rendered in (repr(cap), str(cap)):
        assert cap.token not in rendered
        assert "<redacted>" in rendered


def test_frozen():
    cap = mint_capability(600)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.token = "other"  # type: ignore[misc]


def test_parses_as_frozen_frame_payload():
    cap = mint_capability(600)
    assert isinstance(cap, MintedCapability)

    payload = CapabilityPayload(token=cap.token, expires_at=cap.expires_at, role=cap.role)
    assert payload.token == cap.token
    assert payload.expires_at == cap.expires_at
    assert payload.role == "submitter"

    # The same fields feed the signed hello without any adaptation.
    message = proof_message(
        challenge="c",
        relay_id="r",
        protocol="node-link/v1",
        node_id="n",
        node_name="node",
        daemon_version="0.0.0-test",
        uptime_s=1,
        capability_token=cap.token,
        capability_expires_at=cap.expires_at,
        capability_role=cap.role,
    )
    assert b'"capability_role":"submitter"' in message
