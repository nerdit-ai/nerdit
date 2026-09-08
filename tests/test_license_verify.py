"""The offline license verifier: goldens, the rejection matrix, temporal policy.

P17d WP-D1 (D-LIC1/D-LIC2/D-LIC3/D-LIC9). Three things
are being pinned here and nothing else:

* the **wire format** is exact — a fixed claim set produces byte-identical
  segments and a byte-identical signature (legal as equality assertions because
  Ed25519 signing is deterministic, RFC 8032 — the ``test_link_identity.py``
  precedent);
* every rejection path yields **one fixed machine token**, so the doctor detail,
  the durable event and the 422 envelope can render it without a second thought;
* the temporal policy is **boundary-exact** under an injected clock — no sleeps,
  no wall-clock reads.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nerdit.core import license as license_module
from nerdit.core.license import (
    CLAIMS_VERSION,
    INVALID_REASONS,
    JWS_TYP,
    LICENSE_GRACE_S,
    MAX_LICENSE_BYTES,
    TRUSTED_LICENSE_KEYS,
    LicenseClaims,
    LicenseError,
    LicenseState,
    LicenseVerdict,
    encode_segment,
    install_license_file,
    read_license_file,
    remove_license_file,
    resolve_license_file,
    verify_license,
)
from tests.license_vectors import (
    GOLDEN_BLOB,
    GOLDEN_CLAIMS,
    GOLDEN_HEADER_B64,
    GOLDEN_PAYLOAD_B64,
    GOLDEN_SIGNATURE_B64,
    TEST_KID,
    TEST_PUBLIC_REFERENCE,
    TEST_TRUSTED_KEYS,
    public_reference,
    sign_license,
    sign_raw_segments,
    sign_segments,
)

#: Well before the golden ``expires_at`` (2027-01-01).
NOW_VALID = datetime(2026, 6, 1, tzinfo=UTC)
GOLDEN_EXPIRES_AT = datetime(2027, 1, 1, tzinfo=UTC)


def _at(moment: datetime):
    """An injected clock pinned to ``moment`` (the conftest fake-clock idiom)."""
    return lambda: moment


def _verify(blob: str, *, now: datetime = NOW_VALID, keys=None) -> LicenseVerdict:
    keyset = TEST_TRUSTED_KEYS if keys is None else keys
    return verify_license(blob, trusted_keys=keyset, now=_at(now))


# --- golden vectors -----------------------------------------------------------


def test_frozen_keypair_derives_the_pinned_verifier():
    # The public constant is not a magic string: it falls out of the frozen
    # private key, which is itself one line of range().
    assert public_reference() == TEST_PUBLIC_REFERENCE


def test_golden_segments_are_exact():
    """The wire encoding is part of the contract, not a formatting choice."""
    assert encode_segment({"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP}) == GOLDEN_HEADER_B64
    assert encode_segment(GOLDEN_CLAIMS) == GOLDEN_PAYLOAD_B64
    assert sign_license(GOLDEN_CLAIMS) == GOLDEN_BLOB
    assert GOLDEN_BLOB.split(".")[2] == GOLDEN_SIGNATURE_B64


def test_golden_blob_verifies_and_carries_its_claims():
    verdict = _verify(GOLDEN_BLOB)
    assert verdict.state == "valid"
    assert verdict.reason is None
    claims = verdict.claims
    assert claims is not None
    assert claims.v == CLAIMS_VERSION
    assert claims.lid == GOLDEN_CLAIMS["lid"]
    assert claims.plan == "pro"
    assert claims.features == ("remote_link",)
    assert claims.customer_id == GOLDEN_CLAIMS["customer_id"]
    assert claims.expires_at == GOLDEN_EXPIRES_AT
    assert claims.iat == datetime(2026, 1, 1, tzinfo=UTC)


def test_blob_is_accepted_with_surrounding_whitespace():
    # The file is written with a trailing newline and operators paste blobs.
    assert _verify(f"  {GOLDEN_BLOB}\n").state == "valid"


# --- the pinned trust store (D-LIC3) ------------------------------------------


def test_trusted_keyset_carries_exactly_the_production_kid():
    """The production kid landed 2026-08-16, minted offline by the owner.

    The keyset holds exactly that kid, its value a well-formed public Ed25519
    verifier reference. The credential-gate comment above it is asserted as
    *source text* because it is the artifact that triages the secret-scanner
    hit on the committed public key.
    """
    assert list(TRUSTED_LICENSE_KEYS) == ["nerdit-lic-2026-1"]
    reference = TRUSTED_LICENSE_KEYS["nerdit-lic-2026-1"]
    assert reference.startswith("ed25519:")
    raw = base64.urlsafe_b64decode(reference.removeprefix("ed25519:") + "=")
    assert len(raw) == 32
    source = Path("src/nerdit/core/license.py").read_text(encoding="utf-8")
    assert "publishable by design" in source
    assert "Ship gate" in source


def test_an_untrusted_kid_is_refused_against_the_shipped_keyset():
    # The golden blob's test kid is not in the shipped keyset — the honest
    # failure: not a crash, not a silent pass, a machine token doctor renders.
    verdict = verify_license(GOLDEN_BLOB, trusted_keys=TRUSTED_LICENSE_KEYS, now=_at(NOW_VALID))
    assert (verdict.state, verdict.reason) == ("invalid", "unknown_kid")


# --- the fixed rejection vocabulary -------------------------------------------


def test_invalid_reason_vocabulary_is_pinned():
    """A new reason token is a plan edit, not a diff (it reaches webhooks)."""
    expected = frozenset(
        {
            "malformed",
            "unsupported_alg",
            "unknown_header",
            "missing_kid",
            "unknown_kid",
            "crit_present",
            "bad_signature",
            "unsupported_version",
            "schema_invalid",
        }
    )
    assert expected == INVALID_REASONS


@pytest.mark.parametrize(
    ("blob", "expected"),
    [
        ("", "malformed"),
        ("   ", "malformed"),
        ("not-a-jws", "malformed"),
        ("a.b", "malformed"),
        (f"{GOLDEN_BLOB}.extra", "malformed"),
        (f"!!!.{GOLDEN_PAYLOAD_B64}.{GOLDEN_SIGNATURE_B64}", "malformed"),
        (f"{GOLDEN_HEADER_B64}.!!!.{GOLDEN_SIGNATURE_B64}", "malformed"),
        (f"{GOLDEN_HEADER_B64}.{GOLDEN_PAYLOAD_B64}.!!!", "malformed"),
        # Padded segments are refused rather than repaired: exactly one
        # spelling of a given license exists.
        (f"{GOLDEN_HEADER_B64}=.{GOLDEN_PAYLOAD_B64}.{GOLDEN_SIGNATURE_B64}", "malformed"),
    ],
)
def test_structural_rejections(blob, expected):
    verdict = _verify(blob)
    assert (verdict.state, verdict.reason) == ("invalid", expected)
    assert verdict.claims is None


def test_oversized_blob_is_refused_before_any_parse():
    assert _verify("x" * (MAX_LICENSE_BYTES + 1)).reason == "malformed"


def test_non_json_and_non_object_header_are_malformed():
    payload = encode_segment(GOLDEN_CLAIMS)
    sig = GOLDEN_SIGNATURE_B64
    from nerdit.core.license import _b64encode

    assert _verify(f"{_b64encode(b'nope')}.{payload}.{sig}").reason == "malformed"
    assert _verify(f"{_b64encode(b'[1,2]')}.{payload}.{sig}").reason == "malformed"


def test_non_json_payload_is_malformed_after_a_good_signature():
    from nerdit.core.license import _b64encode

    header_b64 = encode_segment({"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP})
    blob = sign_raw_segments(header_b64, _b64encode(b"not json"))
    assert _verify(blob).reason == "malformed"


@pytest.mark.parametrize("alg", ["none", "None", "eddsa", "ES256", "HS256", "", 1, None])
def test_unsupported_alg(alg):
    header = {"alg": alg, "kid": TEST_KID, "typ": JWS_TYP}
    verdict = _verify(sign_segments(header, GOLDEN_CLAIMS))
    assert (verdict.state, verdict.reason) == ("invalid", "unsupported_alg")


def test_alg_missing_is_unsupported_alg():
    # No alg at all is not "unknown header": there is simply no accepted
    # algorithm, which is what the token says.
    header = {"kid": TEST_KID, "typ": JWS_TYP}
    assert _verify(sign_segments(header, GOLDEN_CLAIMS)).reason == "unsupported_alg"


@pytest.mark.parametrize(
    "header",
    [
        {"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP, "jku": "https://evil.example/keys"},
        {"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP, "jwk": {"kty": "OKP"}},
        {"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP, "x5u": "https://evil.example/c.pem"},
        {"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP, "cty": "JWT"},
    ],
)
def test_unknown_header_parameters_are_refused_not_ignored(header):
    """Key-smuggling headers (``jku``/``jwk``/``x5u``) never get a chance."""
    verdict = _verify(sign_segments(header, GOLDEN_CLAIMS))
    assert (verdict.state, verdict.reason) == ("invalid", "unknown_header")


@pytest.mark.parametrize("typ", [None, "JWT", "nerdit-license", "nerdit-license+JWS", ""])
def test_typ_must_be_exact(typ):
    header = {"alg": "EdDSA", "kid": TEST_KID}
    if typ is not None:
        header["typ"] = typ
    # The cross-protocol confusion guard: a link proof or a future token can
    # never verify as a license, whatever else is right about it.
    assert _verify(sign_segments(header, GOLDEN_CLAIMS)).reason == "unknown_header"


def test_crit_is_refused_whatever_it_contains():
    for crit in ([], ["b64"], ["anything"]):
        header = {"alg": "EdDSA", "kid": TEST_KID, "typ": JWS_TYP, "crit": crit}
        verdict = _verify(sign_segments(header, GOLDEN_CLAIMS))
        assert (verdict.state, verdict.reason) == ("invalid", "crit_present")


@pytest.mark.parametrize("kid", [None, "", 1, ["a"]])
def test_missing_kid(kid):
    header = {"alg": "EdDSA", "typ": JWS_TYP}
    if kid is not None:
        header["kid"] = kid
    assert _verify(sign_segments(header, GOLDEN_CLAIMS)).reason == "missing_kid"


def test_unknown_kid():
    assert _verify(sign_license(GOLDEN_CLAIMS, kid="nope")).reason == "unknown_kid"


def test_malformed_keyset_entry_resolves_to_unknown_kid():
    # Our bug, not the license's — but it still means "this kid cannot verify".
    for bad in ("", "notaprefix", "ed25519:!!!", "ed25519:AAAA"):
        verdict = _verify(GOLDEN_BLOB, keys={TEST_KID: bad})
        assert verdict.reason == "unknown_kid", bad


def test_tampered_payload_is_bad_signature():
    tampered = dict(GOLDEN_CLAIMS, plan="enterprise")
    blob = f"{GOLDEN_HEADER_B64}.{encode_segment(tampered)}.{GOLDEN_SIGNATURE_B64}"
    verdict = _verify(blob)
    assert (verdict.state, verdict.reason) == ("invalid", "bad_signature")
    assert verdict.claims is None


def test_signature_from_a_foreign_key_is_bad_signature():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from nerdit.core.license import _b64encode

    other = Ed25519PrivateKey.from_private_bytes(bytes(range(96, 128)))
    signing_input = f"{GOLDEN_HEADER_B64}.{GOLDEN_PAYLOAD_B64}".encode("ascii")
    blob = f"{GOLDEN_HEADER_B64}.{GOLDEN_PAYLOAD_B64}.{_b64encode(other.sign(signing_input))}"
    assert _verify(blob).reason == "bad_signature"


# --- THE contract: verification runs over the exact received bytes ------------


def test_signature_is_verified_over_the_exact_received_bytes():
    """Risk #1 in the plan: any re-serialization before verify is a bug class.

    A payload whose JSON key order differs from our canonical encoder still
    verifies **when the signature was made over those very bytes** — which is
    only possible if the verifier splices the received segments back together
    instead of re-encoding its own parse.
    """
    from nerdit.core.license import _b64encode

    reordered = json.dumps(GOLDEN_CLAIMS, sort_keys=False, separators=(", ", ": ")).encode()
    payload_b64 = _b64encode(reordered)
    assert payload_b64 != GOLDEN_PAYLOAD_B64

    blob = sign_raw_segments(GOLDEN_HEADER_B64, payload_b64)
    verdict = _verify(blob)
    assert verdict.state == "valid"
    assert verdict.claims is not None and verdict.claims.plan == "pro"


def test_a_canonicalizing_verifier_would_have_accepted_this_one():
    """The negative half of the pair above: same claims, wrong signed bytes.

    The signature is over the canonical encoding while the transmitted payload
    is the reordered one. A verifier that re-serialized its parse before
    checking would call this valid; ours calls it ``bad_signature``.
    """
    from nerdit.core.license import _b64encode

    reordered = json.dumps(GOLDEN_CLAIMS, sort_keys=False, separators=(", ", ": ")).encode()
    blob = f"{GOLDEN_HEADER_B64}.{_b64encode(reordered)}.{GOLDEN_SIGNATURE_B64}"
    assert _verify(blob).reason == "bad_signature"


# --- claims schema ------------------------------------------------------------


@pytest.mark.parametrize("version", [0, 2, 99, -1, "1", 1.0, True, None, [1]])
def test_unsupported_version(version):
    blob = sign_license(dict(GOLDEN_CLAIMS, v=version))
    assert _verify(blob).reason == "unsupported_version"


def test_missing_version_is_schema_invalid():
    claims = {k: v for k, v in GOLDEN_CLAIMS.items() if k != "v"}
    assert _verify(sign_license(claims)).reason == "schema_invalid"


@pytest.mark.parametrize("missing", ["lid", "iat", "customer_id", "plan", "features", "expires_at"])
def test_every_required_claim_is_required(missing):
    claims = {k: v for k, v in GOLDEN_CLAIMS.items() if k != missing}
    assert _verify(sign_license(claims)).reason == "schema_invalid"


@pytest.mark.parametrize(
    "override",
    [
        {"lid": "NOT-A-TOKEN"},
        {"lid": ""},
        {"lid": 1},
        # A trailing newline is the ``$``-vs-``fullmatch`` trap: ``$`` matches
        # *before* a final newline, so a ``match`` check would let these through
        # and put a newline into a doctor detail, an event payload and an audit
        # param — the machine-token guarantee the module claims.
        {"lid": "abc123\n"},
        {"plan": "Pro"},
        {"plan": "pro plan"},
        {"plan": ""},
        {"plan": None},
        {"plan": "pro\n"},
        {"features": "remote_link"},
        {"features": [1]},
        {"features": ["remote link"]},
        {"features": ["REMOTE_LINK"]},
        {"features": ["remote_link\n"]},
        {"customer_id": ""},
        {"customer_id": 42},
        {"customer_id": "x" * 500},
        {"iat": "2026-01-01"},
        {"expires_at": "not-a-date"},
    ],
)
def test_schema_invalid_shapes(override):
    assert _verify(sign_license(dict(GOLDEN_CLAIMS, **override))).reason == "schema_invalid"


@pytest.mark.parametrize("field", ["iat", "expires_at"])
def test_naive_timestamps_are_refused(field):
    """A naive timestamp is never assumed UTC — the issuer states the zone."""
    blob = sign_license(dict(GOLDEN_CLAIMS, **{field: "2027-01-01T00:00:00"}))
    assert _verify(blob).reason == "schema_invalid"


def test_non_utc_offsets_are_accepted():
    # tz-aware is the rule, UTC is not: an offset timestamp is unambiguous.
    blob = sign_license(dict(GOLDEN_CLAIMS, expires_at="2027-01-01T02:00:00+02:00"))
    verdict = _verify(blob)
    assert verdict.state == "valid"
    assert verdict.claims is not None
    assert verdict.claims.expires_at == GOLDEN_EXPIRES_AT


def test_unknown_claim_keys_are_tolerated():
    """Additive evolution must not brick a fielded daemon (contrast: headers)."""
    extra = {"seats": 25, "note": "hello", "nbf": "2026-01-01T00:00:00+00:00"}
    blob = sign_license(dict(GOLDEN_CLAIMS, **extra))
    verdict = _verify(blob)
    assert verdict.state == "valid"
    assert not hasattr(verdict.claims, "seats")


def test_empty_feature_list_is_valid_but_entitles_nothing():
    blob = sign_license(dict(GOLDEN_CLAIMS, features=[]))
    verdict = _verify(blob)
    assert verdict.state == "valid"
    state = LicenseState(verdict, now=_at(NOW_VALID))
    decision = state.require_entitlement("remote_link")
    assert decision.allowed is False
    assert decision.reason == "feature_not_licensed"


def test_unknown_feature_values_ride_along_inertly():
    # A v1 daemon carries a future SSO feature without understanding it.
    blob = sign_license(dict(GOLDEN_CLAIMS, features=["remote_link", "sso", "retention_90d"]))
    verdict = _verify(blob)
    assert verdict.claims is not None
    assert verdict.claims.features == ("remote_link", "sso", "retention_90d")


# --- temporal policy, boundary-exact ------------------------------------------


@pytest.mark.parametrize(
    ("offset_s", "expected"),
    [
        (-LICENSE_GRACE_S, "valid"),
        (-1, "valid"),
        # The boundary itself: at expires_at the license is IN grace, not valid.
        (0, "expired_grace"),
        (1, "expired_grace"),
        (LICENSE_GRACE_S - 1, "expired_grace"),
        # And at exactly +7d it is expired, not in grace.
        (LICENSE_GRACE_S, "expired"),
        (LICENSE_GRACE_S + 1, "expired"),
        (365 * 86400, "expired"),
    ],
)
def test_temporal_boundaries(offset_s, expected):
    now = GOLDEN_EXPIRES_AT + timedelta(seconds=offset_s)
    verdict = _verify(GOLDEN_BLOB, now=now)
    assert verdict.state == expected
    # Claims survive every non-invalid state — doctor renders plan/expiry.
    assert verdict.claims is not None


def test_grace_window_is_seven_days_and_is_not_configurable():
    from nerdit.config.settings import LicenseSettings

    assert LICENSE_GRACE_S == 7 * 86400
    # [license] carries exactly one key: the path. The grace window is product
    # policy, not an operator knob, and there is no trusted-keys override.
    assert set(LicenseSettings.model_fields) == {"file"}


def test_iat_in_the_future_is_not_a_rejection():
    """No ``nbf``, no clock check on ``iat`` — a skewed customer clock must not
    refuse a genuinely valid license."""
    blob = sign_license(dict(GOLDEN_CLAIMS, iat="2030-01-01T00:00:00+00:00"))
    assert _verify(blob).state == "valid"


def test_naive_injected_clock_is_treated_as_utc_rather_than_crashing():
    verdict = verify_license(
        GOLDEN_BLOB, trusted_keys=TEST_TRUSTED_KEYS, now=lambda: datetime(2026, 6, 1)
    )
    assert verdict.state == "valid"


# --- redaction ----------------------------------------------------------------


def test_claims_repr_never_carries_customer_id():
    verdict = _verify(GOLDEN_BLOB)
    claims = verdict.claims
    assert claims is not None
    for text in (repr(claims), str(claims), f"{claims}", "%s" % (claims,)):
        assert GOLDEN_CLAIMS["customer_id"] not in text
        assert "customer_id" not in text
        assert claims.plan in text
        assert claims.lid in text


def test_license_state_repr_is_machine_shaped():
    state = LicenseState(_verify(GOLDEN_BLOB), now=_at(NOW_VALID))
    assert repr(state) == "LicenseState(installed=True, state='valid')"
    assert GOLDEN_CLAIMS["customer_id"] not in repr(state)


def test_claims_are_frozen_and_slotted():
    claims = _verify(GOLDEN_BLOB).claims
    assert claims is not None
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError is a dataclass detail
        claims.plan = "enterprise"  # type: ignore[misc]
    assert not hasattr(claims, "__dict__")


# --- the holder + require_entitlement (D-LIC2) --------------------------------


def test_absent_license_allows_everything():
    state = LicenseState()
    assert state.installed is False
    assert state.state is None
    assert state.claims is None
    decision = state.require_entitlement("remote_link")
    assert (decision.feature, decision.allowed, decision.state, decision.reason) == (
        "remote_link",
        True,
        None,
        None,
    )


def test_invalid_license_allows_with_an_advisory():
    """A corrupt file must not disable more than no file at all."""
    state = LicenseState(LicenseVerdict("invalid", "bad_signature"), now=_at(NOW_VALID))
    decision = state.require_entitlement("remote_link")
    assert decision.allowed is True
    assert decision.state == "invalid"
    assert decision.reason == "bad_signature"
    assert decision.reason in INVALID_REASONS


@pytest.mark.parametrize(
    ("offset_s", "state_name", "allowed", "reason"),
    [
        (-86400, "valid", True, None),
        (60, "expired_grace", True, "expired_grace"),
        (LICENSE_GRACE_S + 60, "expired", False, "expired"),
    ],
)
def test_entitlement_matrix_over_time(offset_s, state_name, allowed, reason):
    now = GOLDEN_EXPIRES_AT + timedelta(seconds=offset_s)
    state = LicenseState(_verify(GOLDEN_BLOB, now=now), now=_at(now))
    decision = state.require_entitlement("remote_link")
    assert (decision.state, decision.allowed, decision.reason) == (state_name, allowed, reason)


def test_feature_absent_outranks_the_grace_advisory():
    now = GOLDEN_EXPIRES_AT + timedelta(seconds=60)
    blob = sign_license(dict(GOLDEN_CLAIMS, features=["sso"]))
    state = LicenseState(_verify(blob, now=now), now=_at(now))
    decision = state.require_entitlement("remote_link")
    assert decision.allowed is False
    assert decision.reason == "feature_not_licensed"


def test_state_is_recomputed_live_from_the_injected_clock():
    """No poller, no restart: a grace transition surfaces on the next read."""
    moment = {"now": GOLDEN_EXPIRES_AT - timedelta(seconds=1)}
    state = LicenseState(_verify(GOLDEN_BLOB), now=lambda: moment["now"])
    assert state.state == "valid"
    moment["now"] = GOLDEN_EXPIRES_AT
    assert state.state == "expired_grace"
    moment["now"] = GOLDEN_EXPIRES_AT + timedelta(seconds=LICENSE_GRACE_S)
    assert state.state == "expired"
    assert state.require_entitlement("remote_link").allowed is False


def test_replace_refreshes_the_holder_in_place():
    """Install/remove must not rebind app.state — captured references exist."""
    state = LicenseState(now=_at(NOW_VALID))
    assert state.installed is False
    state.replace(_verify(GOLDEN_BLOB))
    assert state.installed is True
    assert state.state == "valid"
    state.replace(None)
    assert state.installed is False
    assert state.state is None
    assert state.reason is None


# --- file custody (D-LIC5) ----------------------------------------------------


def test_resolve_license_file_defaults_under_data_dir(tmp_path):
    assert resolve_license_file(None, str(tmp_path)) == tmp_path / "license.jws"
    assert resolve_license_file("", str(tmp_path)) == tmp_path / "license.jws"
    assert resolve_license_file("/etc/nerdit/lic.jws", str(tmp_path)) == Path("/etc/nerdit/lic.jws")


def test_absent_file_reads_as_none(tmp_path):
    assert read_license_file(tmp_path / "license.jws") is None


def test_install_is_owner_only_and_round_trips(tmp_path):
    path = tmp_path / "license.jws"
    install_license_file(path, GOLDEN_BLOB)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert read_license_file(path) == GOLDEN_BLOB
    # The file is one line of ascii with a trailing newline.
    assert path.read_bytes() == f"{GOLDEN_BLOB}\n".encode("ascii")
    # No temp file is left behind.
    assert [p.name for p in tmp_path.iterdir()] == ["license.jws"]


def test_install_overwrites_because_renewal_is_an_overwrite(tmp_path):
    path = tmp_path / "license.jws"
    install_license_file(path, GOLDEN_BLOB)
    renewed = sign_license(dict(GOLDEN_CLAIMS, expires_at="2028-01-01T00:00:00+00:00"))
    install_license_file(path, renewed)
    assert read_license_file(path) == renewed
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_install_refuses_a_missing_parent_rather_than_creating_one(tmp_path):
    with pytest.raises(LicenseError):
        install_license_file(tmp_path / "nope" / "license.jws", GOLDEN_BLOB)


def test_symlinked_license_is_refused(tmp_path):
    real = tmp_path / "real.jws"
    real.write_text(f"{GOLDEN_BLOB}\n")
    link = tmp_path / "license.jws"
    link.symlink_to(real)
    with pytest.raises(LicenseError) as exc:
        read_license_file(link)
    # Path-bearing (the LinkIdentityError idiom) but blob-free, always.
    assert GOLDEN_BLOB not in str(exc.value)


def test_fifo_license_is_refused_not_hung(tmp_path):
    """A writerless FIFO at the license path must refuse *promptly*, not block.

    ``build_license_state`` awaits this read before the lifespan yields, so a
    blocking ``open`` would hang the daemon at boot instead of degrading to the
    doctor-visible invalid state (never-abort-boot, D-LIC2). The read runs on a
    *daemon* thread joined with a timeout so a regression fails this test in
    five seconds rather than wedging the whole suite forever.
    """
    path = tmp_path / "license.jws"
    os.mkfifo(path)
    captured: list[object] = []

    def _read() -> None:
        try:
            captured.append(read_license_file(path))
        except LicenseError as exc:  # the expected outcome
            captured.append(exc)

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(5.0)
    if thread.is_alive():
        pytest.fail("read_license_file blocked on a writerless FIFO (boot would hang)")
    assert len(captured) == 1
    assert isinstance(captured[0], LicenseError), captured[0]
    # Path-bearing but blob-free, exactly like the symlink refusal.
    assert GOLDEN_BLOB not in str(captured[0])


def test_oversized_file_is_refused_without_returning_content(tmp_path):
    path = tmp_path / "license.jws"
    path.write_text("x" * (MAX_LICENSE_BYTES + 10))
    with pytest.raises(LicenseError):
        read_license_file(path)


def _blob_of_length(target: int) -> str:
    """A *valid signed* blob of exactly ``target`` characters.

    Padded through an **unknown claim key**, which the verifier tolerates by
    design (additive evolution), so the result is a real license and not a
    string of ``x``. The padding length is searched rather than hardcoded: a
    change to the golden claim set then fails loudly here instead of silently
    retargeting the test at some other length.
    """
    low, high = 0, target
    while low < high:
        mid = (low + high) // 2
        if len(sign_license(dict(GOLDEN_CLAIMS, x="a" * mid))) < target:
            low = mid + 1
        else:
            high = mid
    blob = sign_license(dict(GOLDEN_CLAIMS, x="a" * low))
    assert len(blob) == target, f"no padding yields exactly {target} chars"
    return blob


def test_a_blob_at_exactly_the_cap_installs_and_reads_back(tmp_path):
    """The HTTP bound and the file bound agree — no accepted-then-rejected drift.

    ``install_license_file`` writes ``blob + "\\n"``, so a blob the install route
    accepted (``max_length=MAX_LICENSE_BYTES``) would exceed the cap by one byte
    if the reader measured raw file bytes: the daemon would accept the license,
    then reject its own file at every subsequent boot.
    """
    blob = _blob_of_length(MAX_LICENSE_BYTES)
    assert _verify(blob).state == "valid"

    path = tmp_path / "license.jws"
    install_license_file(path, blob)
    assert path.read_bytes() == f"{blob}\n".encode("ascii")
    assert read_license_file(path) == blob
    assert _verify(read_license_file(path)).state == "valid"


def test_a_blob_one_char_over_the_cap_is_still_refused(tmp_path):
    """Measuring the *stripped* blob relaxes framing, never the cap itself."""
    blob = _blob_of_length(MAX_LICENSE_BYTES + 1)
    path = tmp_path / "license.jws"
    path.write_bytes(f"{blob}\n".encode("ascii"))
    with pytest.raises(LicenseError) as exc:
        read_license_file(path)
    assert str(MAX_LICENSE_BYTES) in str(exc.value)


def test_junk_beyond_the_read_window_cannot_hide_behind_whitespace(tmp_path):
    """The strict one-line-file contract holds past the bounded read.

    Junk *inside* the window is already caught (``strip`` only trims the ends,
    so blob + whitespace + junk fails verification). Junk *beyond* it was
    invisible: the reader took ``MAX_LICENSE_BYTES + 2`` bytes and never
    established EOF, so a file whose first window is a valid blob padded with
    newlines stripped back to that blob and was accepted — a hand-edited or
    truncated-and-appended file reading back as clean. One probe byte after the
    windowed read closes it.
    """
    path = tmp_path / "license.jws"
    padding = b"\n" * (MAX_LICENSE_BYTES + 2 - len(GOLDEN_BLOB))
    path.write_bytes(GOLDEN_BLOB.encode("ascii") + padding + b"#junk-beyond-the-window")

    with pytest.raises(LicenseError) as exc:
        read_license_file(path)
    assert str(MAX_LICENSE_BYTES) in str(exc.value)
    assert GOLDEN_BLOB not in str(exc.value)


def test_non_ascii_file_is_refused(tmp_path):
    path = tmp_path / "license.jws"
    path.write_bytes(b"\xff\xfe not ascii")
    with pytest.raises(LicenseError):
        read_license_file(path)


def test_license_error_messages_never_carry_the_blob(tmp_path):
    path = tmp_path / "license.jws"
    path.write_text(f"{GOLDEN_BLOB}\n" * 400)
    with pytest.raises(LicenseError) as exc:
        read_license_file(path)
    assert GOLDEN_BLOB not in str(exc.value)


def test_remove_reports_what_happened(tmp_path):
    path = tmp_path / "license.jws"
    assert remove_license_file(path) is False
    install_license_file(path, GOLDEN_BLOB)
    assert remove_license_file(path) is True
    assert remove_license_file(path) is False
    assert not path.exists()


def test_module_never_logs(caplog):
    """The never-leak contract: this module emits no log records at all."""
    with caplog.at_level("DEBUG"):
        _verify(GOLDEN_BLOB)
        _verify("garbage")
    assert [r for r in caplog.records if r.name.startswith("nerdit.core.license")] == []
    assert not hasattr(license_module, "logger")


def test_claims_type_is_exported_for_the_surfaces():
    # B2's routes/doctor/capabilities read claims off the holder.
    assert LicenseClaims.__slots__
