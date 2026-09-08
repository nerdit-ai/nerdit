"""Provide one frozen test keypair, signer and license vectors.

This private key is a publishable test fixture: it signs no production license,
verifies against no shipped key ID, and tests inject TEST_TRUSTED_KEYS explicitly.
Its bytes are trivially generated with range(), distinct from link identity and
cloud development fixtures to prevent cross-suite confusion.
"""

from __future__ import annotations

from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from nerdit.core.license import _b64encode, build_signing_input, encode_segment

#: The frozen test signing key: ``bytes(range(64, 96))``.
TEST_PRIVATE_KEY_RAW = bytes(range(64, 96))
TEST_PRIVATE_KEY_B64 = "QEFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaW1xdXl8"

#: The kid the tests enroll. Deliberately not a plausible production kid.
TEST_KID = "test-lic-1"
TEST_PUBLIC_REFERENCE = "ed25519:JUO5L_EJVRFHatyDadtt3JM2ZaEZeN2hQE7hBmypVZ0"

#: The injected trust store — the only way a test license verifies.
TEST_TRUSTED_KEYS = {TEST_KID: TEST_PUBLIC_REFERENCE}

#: The golden claim set. Every field is pinned, so the goldens below are exact.
GOLDEN_CLAIMS: dict[str, Any] = {
    "v": 1,
    "lid": "0f1e2d3c4b5a69788796a5b4c3d2e1f0",
    "iat": "2026-01-01T00:00:00+00:00",
    "customer_id": "cus_p17d_golden",
    "plan": "pro",
    "features": ["remote_link"],
    "expires_at": "2027-01-01T00:00:00+00:00",
}

GOLDEN_HEADER_B64 = (
    "eyJhbGciOiJFZERTQSIsImtpZCI6InRlc3QtbGljLTEiLCJ0eXAiOiJuZXJkaXQtbGljZW5zZStqd3MifQ"
)
GOLDEN_PAYLOAD_B64 = (
    "eyJjdXN0b21lcl9pZCI6ImN1c19wMTdkX2dvbGRlbiIsImV4cGlyZXNfYXQiOiIyMDI3LTAxLTAxVDAwOjAwOjAwK"
    "zAwOjAwIiwiZmVhdHVyZXMiOlsicmVtb3RlX2xpbmsiXSwiaWF0IjoiMjAyNi0wMS0wMVQwMDowMDowMCswMDowMC"
    "IsImxpZCI6IjBmMWUyZDNjNGI1YTY5Nzg4Nzk2YTViNGMzZDJlMWYwIiwicGxhbiI6InBybyIsInYiOjF9"
)
GOLDEN_SIGNATURE_B64 = (
    "SC2gQm38tch6WbP49eFPyUY77_vZ4Q3xq4IKL9C0j9THPOaCoZoFvNFMCfAKrrMG29KHpTenefQoh6LakePZAg"
)
GOLDEN_BLOB = f"{GOLDEN_HEADER_B64}.{GOLDEN_PAYLOAD_B64}.{GOLDEN_SIGNATURE_B64}"

_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(TEST_PRIVATE_KEY_RAW)


def public_reference() -> str:
    """Derive the verifier reference from the frozen key (pins the constant)."""
    raw = _PRIVATE_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return f"ed25519:{_b64encode(raw)}"


def sign_license(claims: dict[str, Any], *, kid: str = TEST_KID) -> str:
    """Sign ``claims`` into a compact JWS with the frozen test key."""
    header_b64, payload_b64, signing_input = build_signing_input(kid=kid, claims=claims)
    return f"{header_b64}.{payload_b64}.{_b64encode(_PRIVATE_KEY.sign(signing_input))}"


def sign_segments(header: dict[str, Any], payload: dict[str, Any]) -> str:
    """Sign an arbitrary header/payload pair — the rejection-matrix workhorse.

    Unlike :func:`sign_license` the header is taken verbatim, so a test can mint
    a *correctly signed* blob carrying ``alg: none``, a ``crit`` parameter or an
    unknown header key and prove the verifier refuses it on semantics rather
    than on a broken signature.
    """
    header_b64 = encode_segment(header)
    payload_b64 = encode_segment(payload)
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    return f"{header_b64}.{payload_b64}.{_b64encode(_PRIVATE_KEY.sign(signing_input))}"


def sign_raw_segments(header_b64: str, payload_b64: str) -> str:
    """Sign two already-encoded segments verbatim (exact-bytes tests)."""
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    return f"{header_b64}.{payload_b64}.{_b64encode(_PRIVATE_KEY.sign(signing_input))}"
