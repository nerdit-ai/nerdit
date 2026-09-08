"""Unit tests for the edge-auth primitives (P25 WP4b, §5.4).

Three properties this file exists to pin:

1. :func:`load_edge_auth` is **tri-state** (D-P25-8) — absent, well-formed and
   declared-but-malformed are three different outcomes, and collapsing the
   third into the first would publish a route its owner asked to protect.
2. The malformed path names FIELD NAMES only — the exception reaches a
   diagnose payload and a log line.
3. :func:`hash_password` really is bcrypt, and really is salted per call
   (the property the D-P25-6 memo cache exists to work around).
"""

from __future__ import annotations

import bcrypt
import pytest

from nerdit.core.proxy.edgeauth import (
    EdgeAuthInvalid,
    EdgeAuthMaterial,
    EdgeAuthSpec,
    auth_fingerprint,
    hash_password,
    load_edge_auth,
    password_exceeds_bcrypt_limit,
)

# A literal password, hunted through every EdgeAuthInvalid channel.
PW_CANARY = "hunter2-CANARY-0000"


# --- load_edge_auth: state 1, absent -------------------------------------------


def test_absent_blob_is_none():
    """No declaration ⇒ no edge auth ⇒ the route serves openly, as it does today."""
    assert load_edge_auth(None) is None


# --- load_edge_auth: state 2, well-formed --------------------------------------


def test_wellformed_blob_yields_the_spec():
    spec = load_edge_auth({"user": "alice", "password": "${secrets.APP_PW}"})
    assert spec == EdgeAuthSpec(user="alice", password_ref="${secrets.APP_PW}")


def test_wellformed_shared_scoped_ref_yields_the_spec():
    spec = load_edge_auth({"user": "alice", "password": "${secrets.shared.APP_PW}"})
    assert spec is not None
    assert spec.password_ref == "${secrets.shared.APP_PW}"


def test_unknown_extra_key_is_ignored_on_a_wellformed_blob():
    """D-P22-3 tolerance: EXTRA keys survive an older daemon reading a newer row.

    Tolerance is only ever about extra keys — never about a missing or
    ungrammatical credential (the three cases below).
    """
    spec = load_edge_auth(
        {"user": "alice", "password": "${secrets.APP_PW}", "realm": "staging", "future": 1}
    )
    assert spec == EdgeAuthSpec(user="alice", password_ref="${secrets.APP_PW}")


def test_spec_is_frozen():
    spec = load_edge_auth({"user": "alice", "password": "${secrets.APP_PW}"})
    assert spec is not None
    with pytest.raises(Exception):  # noqa: B017 — dataclasses raises FrozenInstanceError
        spec.user = "mallory"  # type: ignore[misc]


# --- load_edge_auth: state 3, declared but malformed ---------------------------


@pytest.mark.parametrize(
    ("blob", "fields"),
    [
        ({"user": ""}, ("user", "password")),  # blank user AND no password
        ({"user": "u"}, ("password",)),  # no password
        ({"user": "u", "password": "hunter2"}, ("password",)),  # literal, not a ref
        ({"password": "${secrets.APP_PW}"}, ("user",)),  # no user
        ({"user": "  ", "password": "${secrets.APP_PW}"}, ("user",)),  # all-whitespace = blank
        ({"user": "a:b", "password": "${secrets.APP_PW}"}, ("user",)),  # RFC 7617
        ({"user": "a\nb", "password": "${secrets.APP_PW}"}, ("user",)),  # log forging
        ({"user": 7, "password": "${secrets.APP_PW}"}, ("user",)),  # wrong type
        ({"user": "u", "password": ["${secrets.APP_PW}"]}, ("password",)),
        # SECRET_REF_RE's '$' also matches before a trailing newline — fullmatch
        # is what keeps a smuggled literal out on this backstop path.
        ({"user": "u", "password": "${secrets.APP_PW}\nhunter2"}, ("password",)),
        # The TRAILING-newline pair: '$'-anchored .match accepts both, so these
        # pin the review-upheld fullmatch tightening on BOTH fields.
        ({"user": "admin\n", "password": "${secrets.APP_PW}"}, ("user",)),
        ({"user": "u", "password": "${secrets.APP_PW}\n"}, ("password",)),
    ],
)
def test_malformed_blob_raises_naming_only_field_names(blob, fields):
    with pytest.raises(EdgeAuthInvalid) as excinfo:
        load_edge_auth(blob)
    assert excinfo.value.fields == fields
    text = str(excinfo.value)
    for name in fields:
        assert name in text
    # The values never travel — this string reaches /diagnose and nerditd.log.
    for value in blob.values():
        if isinstance(value, str) and value not in {"u", ""}:
            assert value not in text


def test_malformed_blob_never_degrades_to_no_auth():
    """The explicit negative: a broken declaration is NOT ``None`` (D-P25-8).

    Asserting only "no route appears" would not distinguish a fail-closed prune
    from a silently public route, which is the leak this rule exists to prevent.
    """
    for blob in ({"user": ""}, {"user": "u"}, {"user": "u", "password": PW_CANARY}):
        with pytest.raises(EdgeAuthInvalid) as excinfo:
            load_edge_auth(blob)
        assert PW_CANARY not in str(excinfo.value)


def test_non_mapping_blob_is_malformed_not_absent():
    for blob in ("edge_auth", ["alice"], 42, True):
        with pytest.raises(EdgeAuthInvalid) as excinfo:
            load_edge_auth(blob)
        assert excinfo.value.fields == ("edge_auth",)


def test_empty_mapping_is_malformed_not_absent():
    """``{}`` is a declaration with nothing in it — not "no auth"."""
    with pytest.raises(EdgeAuthInvalid) as excinfo:
        load_edge_auth({})
    assert excinfo.value.fields == ("user", "password")


# --- hash_password: real bcrypt properties (§5.4) ------------------------------


def test_hash_password_output_verifies_under_bcrypt():
    hashed = hash_password("s3cr3t")
    assert bcrypt.checkpw(b"s3cr3t", hashed.encode("ascii"))


def test_hash_password_output_rejects_the_wrong_plaintext():
    hashed = hash_password("s3cr3t")
    assert not bcrypt.checkpw(b"s3cr3t-nope", hashed.encode("ascii"))


def test_hash_password_is_salted_per_call():
    """The property the whole caching design rests on (D-P25-6).

    Two calls on the same plaintext differ, so the desired fingerprint is NOT
    stable across processes — hence ``_auth_cache``, and hence the bounded
    one-upsert-per-authed-service churn after a daemon restart.
    """
    assert hash_password("s3cr3t") != hash_password("s3cr3t")


def test_hash_password_returns_a_str_not_bytes():
    """The Caddy account object carries the hash as a plain JSON string."""
    hashed = hash_password("s3cr3t")
    assert isinstance(hashed, str)
    assert hashed.startswith("$2")


def test_hash_password_handles_utf8_and_the_72_byte_limit():
    # Multi-byte plaintexts are hashed on their UTF-8 bytes, not their chars.
    assert bcrypt.checkpw("pässwörd".encode(), hash_password("pässwörd").encode("ascii"))
    # bcrypt >= 4 refuses > 72 bytes rather than truncating; truncating a
    # credential on the user's behalf is worse than refusing to serve it, so
    # this joins the D-P25-8 fail-closed path — naming the field, not the value.
    with pytest.raises(EdgeAuthInvalid) as excinfo:
        hash_password(PW_CANARY + "x" * 72)
    assert excinfo.value.fields == ("password",)
    assert PW_CANARY not in str(excinfo.value)
    # Review-upheld: the over-length message must name the REAL fix (a shorter
    # secret) — the default repair-the-declaration sentence would send the
    # operator to `nerdit config app set`, which cannot help here.
    assert "72-byte" in str(excinfo.value)
    assert "nerdit secrets set" in str(excinfo.value)
    assert "nerdit config app set" not in str(excinfo.value)


def test_password_exceeds_bcrypt_limit_counts_utf8_bytes():
    """The 72-BYTE bound is bytes, not characters — one predicate, two callers.

    ``hash_password`` and the diagnose classifier
    (``_classify_edge_auth``) both consult this predicate, so the route plane
    and diagnose can never disagree about whether a resolved secret serves.
    """
    assert not password_exceeds_bcrypt_limit("x" * 72)
    assert password_exceeds_bcrypt_limit("x" * 73)
    # 25 three-byte chars = 75 bytes: over the limit at only 25 characters.
    assert password_exceeds_bcrypt_limit("€" * 25)


# --- auth_fingerprint: the single drift-compare function (D-P25-7) -------------

_HASH_A = "$2b$12$GOLDENGOLDENGOLDENGOLDENGOLDENGOLDENGOLDENGOLDENGOLDENx"
_HASH_B = "$2b$12$SECONDSECONDSECONDSECONDSECONDSECONDSECONDSECONDSECONDy"


def test_fingerprint_is_stable_for_the_same_inputs():
    assert auth_fingerprint("alice", _HASH_A) == auth_fingerprint("alice", _HASH_A)


def test_fingerprint_is_16_hex_chars():
    fp = auth_fingerprint("alice", _HASH_A)
    assert len(fp) == 16
    assert all(char in "0123456789abcdef" for char in fp)


def test_fingerprint_changes_when_only_the_user_changes():
    """The D-P25-6 cache-key regression, at the fingerprint layer.

    If a username change did not move the fingerprint, the drift classifier
    would see no drift and Caddy would keep accepting the obsolete username.
    """
    assert auth_fingerprint("alice", _HASH_A) != auth_fingerprint("bob", _HASH_A)


def test_fingerprint_changes_when_only_the_hash_changes():
    """Secret rotation must be visible to the classifier."""
    assert auth_fingerprint("alice", _HASH_A) != auth_fingerprint("alice", _HASH_B)


def test_fingerprint_separator_prevents_field_smearing():
    """The NUL separator: ('ab', 'c') and ('a', 'bc') are different handlers."""
    assert auth_fingerprint("ab", "c") != auth_fingerprint("a", "bc")


def test_fingerprint_never_contains_the_inputs():
    fp = auth_fingerprint("alice", _HASH_A)
    assert "alice" not in fp
    assert _HASH_A not in fp


# --- EdgeAuthMaterial ----------------------------------------------------------


def test_material_is_frozen_and_carries_only_the_hash():
    material = EdgeAuthMaterial(user="alice", bcrypt_hash=_HASH_A)
    assert material.user == "alice"
    assert material.bcrypt_hash == _HASH_A
    with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
        material.bcrypt_hash = _HASH_B  # type: ignore[misc]


def test_hash_password_is_patchable_through_the_module():
    """§3.4.2: a module-level function called THROUGH the module.

    The golden snapshots and the memoization spy both patch it; if a caller
    ever imported it by value this test would still pass, but the module
    attribute is the contract the plan names.
    """
    from nerdit.core.proxy import edgeauth

    original = edgeauth.hash_password
    try:
        edgeauth.hash_password = lambda plaintext: "$2b$12$STANDIN"  # type: ignore[assignment]
        assert edgeauth.hash_password("anything") == "$2b$12$STANDIN"
    finally:
        edgeauth.hash_password = original  # type: ignore[assignment]
    assert edgeauth.hash_password("s3cr3t") != "$2b$12$STANDIN"
