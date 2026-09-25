"""Verify offline product entitlements as versioned Ed25519 JWS documents.

This optional product feature is unrelated to the repository's Apache-2.0
license. The link caller treats entitlement decisions as advisory; the cloud
relay enforces remote access. Other callers own their enforcement policy.

JWS segments use unpadded URL-safe base64. Require exactly `alg="EdDSA"`, a
pinned trusted `kid`, and `typ="nerdit-license+jws"`; reject unknown header
keys, including `crit`. Verify the exact received signing input before parsing
versioned claims. Unknown claim keys are allowed; unsupported versions produce
a machine-readable reason. No network key lookup occurs.

Never expose the license blob in logs, audit, diagnostics, events, CLI output,
or errors. Customer IDs appear only in admin license/status/capabilities
surfaces, never non-admin output, logs, audit, diagnostics, or events.
Rejections use fixed reason tokens; claims repr omits the customer ID.
This module logs nothing. File errors may identify a path, never blob bytes;
callers must keep those paths out of diagnostics and error envelopes.
"""

from __future__ import annotations

import base64
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from secrets import token_hex
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from nerdit.utils.fs import fsync_dir

# ---------------------------------------------------------------------------
# Pinned trust store (D-LIC3)
# ---------------------------------------------------------------------------

#: Canonical prefix of a public verifier reference (`core/link/identity.py`
#: parity — the same `ed25519:<base64url>` spelling the cloud uses).
ED25519_REFERENCE_PREFIX = "ed25519:"

# Every value below is publishable by design: a *public Ed25519 verification
# key*. It can verify a license, never mint one; the private half is the
# owner's, held offline, and must never enter this repo, CI, or the cloud. A
# secret-scanner hit here is expected and triaged by this comment.
#
# The production kid `nerdit-lic-2026-1` is minted offline
# (`scripts/license_tool.py keygen`), never by a build agent. **Ship gate:** a
# kid must be in a released daemon before the first license signed by it is
# issued, because a license signed by an unknown kid verifies as
# `invalid/unknown_kid`, which is the honest failure.
#
# The test/dev seam is parameter injection only (`trusted_keys=` on
# `verify_license`); there is deliberately no `[license].trusted_keys`
# config override in v1 (D-LIC3).
TRUSTED_LICENSE_KEYS: dict[str, str] = {
    "nerdit-lic-2026-1": "ed25519:DZp4JvLxWbcaet9uUa1Mh4ECtD6ULC_57pQEVFZMl_s",
}

# ---------------------------------------------------------------------------
# Wire constants (D-LIC1)
# ---------------------------------------------------------------------------

#: The single accepted `alg`. There is no negotiation: one issuer, one verifier.
JWS_ALG = "EdDSA"
#: The required, exact `typ` — the cross-protocol confusion guard.
JWS_TYP = "nerdit-license+jws"
#: The complete set of accepted protected-header parameters.
JWS_HEADER_KEYS = frozenset({"alg", "kid", "typ"})
#: The only supported claims version.
CLAIMS_VERSION = 1

#: Grace after `expires_at` before a license counts as expired. **Product
#: policy, not an operator knob** — and the *only* leeway there is: a second
#: fuzz factor on top would make the doctor message unexplainable.
LICENSE_GRACE_S = 7 * 86400

#: Hard cap on a license blob. A signed entitlement is ~500 bytes; anything
#: past this is refused as `malformed` before a single parse, so neither the
#: install route nor a boot read can be made to chew on a large file.
MAX_LICENSE_BYTES = 8192

#: Machine tokens allowed in `plan`/`lid`/`features` values. Everything
#: that leaves this module for an event, an audit row or a doctor detail is
#: shaped by this pattern, so no claim can smuggle free text into those feeds.
#: Every use site is `fullmatch`, never `match` (the P14a/P25 hardening
#: precedent): `$` also matches *before* a trailing newline, so `match`
#: would let `"pro\n"` through and put a newline into an event payload, an
#: audit param and a doctor detail.
_TOKEN_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_LID_RE = re.compile(r"^[a-z0-9]{1,64}$")
#: Unpadded urlsafe base64, one or more characters.
_B64URL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
#: `customer_id` is opaque to the daemon (it is the cloud's id) and is never
#: rendered anywhere it could be read, so only a sane bound is enforced.
_MAX_CUSTOMER_ID = 200

_RAW_KEY_BYTES = 32

# ---------------------------------------------------------------------------
# States and the fixed invalid-reason vocabulary (D-LIC1)
# ---------------------------------------------------------------------------

VerdictState = Literal["valid", "expired_grace", "expired", "invalid"]

STATE_VALID: VerdictState = "valid"
STATE_EXPIRED_GRACE: VerdictState = "expired_grace"
STATE_EXPIRED: VerdictState = "expired"
STATE_INVALID: VerdictState = "invalid"

#: Structure the verifier could not even parse as a JWS (segment count, base64,
#: UTF-8, JSON, non-object header/payload, oversized blob) — and, at boot, a
#: present-but-unreadable file (see `daemon/bootstrap.build_license_state`).
REASON_MALFORMED = "malformed"
#: `alg` missing or anything other than `EdDSA`.
REASON_UNSUPPORTED_ALG = "unsupported_alg"
#: A header parameter outside `JWS_HEADER_KEYS`, **or** a missing/wrong
#: `typ`. There is deliberately no `unsupported_typ` token: a wrong `typ`
#: means "this header does not describe a nerdit license", which is exactly what
#: this token says, and the doctor/event vocabulary stays as small as possible.
REASON_UNKNOWN_HEADER = "unknown_header"
#: `kid` absent or not a non-empty string.
REASON_MISSING_KID = "missing_kid"
#: `kid` not present in the pinned keyset.
REASON_UNKNOWN_KID = "unknown_kid"
#: `crit` present, whatever it contains: we implement no extensions, so any
#: criticality demand is by definition unverifiable.
REASON_CRIT_PRESENT = "crit_present"
#: The Ed25519 check over the exact received signing input failed.
REASON_BAD_SIGNATURE = "bad_signature"
#: `v` is a value this daemon does not implement — "the daemon is too old".
REASON_UNSUPPORTED_VERSION = "unsupported_version"
#: Claims are v1 but a required claim is missing or malformed (incl. a naive,
#: non-tz-aware timestamp).
REASON_SCHEMA_INVALID = "schema_invalid"

#: The complete rejection vocabulary, pinned by test. Every token here is safe
#: to render in a doctor detail, an event payload and a 422 envelope.
INVALID_REASONS: frozenset[str] = frozenset(
    {
        REASON_MALFORMED,
        REASON_UNSUPPORTED_ALG,
        REASON_UNKNOWN_HEADER,
        REASON_MISSING_KID,
        REASON_UNKNOWN_KID,
        REASON_CRIT_PRESENT,
        REASON_BAD_SIGNATURE,
        REASON_UNSUPPORTED_VERSION,
        REASON_SCHEMA_INVALID,
    }
)

# ---------------------------------------------------------------------------
# Entitlement-decision advisories (D-LIC2)
# ---------------------------------------------------------------------------

#: The license is authentic but does not carry the requested feature.
DECISION_FEATURE_NOT_LICENSED = "feature_not_licensed"
#: Past `expires_at` but inside the grace window (allowed, with an advisory).
DECISION_EXPIRED_GRACE = STATE_EXPIRED_GRACE
#: Past the grace window.
DECISION_EXPIRED = STATE_EXPIRED

#: The one feature token wired in v1 (WP-D2). Named rather than spelled inline
#: at each seam so the `[link]` boot check, the doctor check and any future
#: SSO/retention caller can never drift on a string literal.
FEATURE_REMOTE_LINK = "remote_link"

#: Custody of the installed blob: owner read/write only, from the open flags —
#: there is never a world-readable window.
LICENSE_FILE_MODE = 0o600

#: Default file name under `data_dir`.
LICENSE_FILE_NAME = "license.jws"


class LicenseError(Exception):
    """A license file could not be read or written.

    Messages may name the offending **path** (the `LinkIdentityError` idiom);
    they never carry blob bytes, and a caller must never forward one into a
    doctor detail or an error envelope.
    """


# ---------------------------------------------------------------------------
# Private base64url helpers — vendored, not imported (D-LIC1)
# ---------------------------------------------------------------------------


def _b64encode(value: bytes) -> str:
    """Encode as unpadded urlsafe base64 ascii (the canonical wire form)."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_segment(segment: str) -> bytes | None:
    """Decode one canonical unpadded-base64url JWS segment, or `None`.

    Canonical means *round-trippable*: a segment carrying padding, non-alphabet
    characters or non-zero trailing bits is refused rather than repaired, so
    exactly one spelling of a given license exists.
    """
    if not _B64URL_RE.fullmatch(segment):
        return None
    padding = "=" * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(segment + padding)
    except (ValueError, TypeError):
        return None
    if _b64encode(raw) != segment:
        return None
    return raw


def _load_json_object(raw: bytes) -> dict[str, Any] | None:
    """Parse UTF-8 JSON that must be an object, else `None`."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def encode_segment(payload: dict[str, Any]) -> str:
    """Serialize one JWS segment: compact, sorted JSON in canonical base64url.

    Shared with `scripts/license_tool.py` (D-LIC8) so the issuer and the
    verifier can never drift on the encoding. `sort_keys` + the compact
    separators are the same convention as `link/identity.proof_message` —
    signing input determinism is a property, not a formatting preference.
    """
    return _b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def build_signing_input(*, kid: str, claims: Mapping[str, Any]) -> tuple[str, str, bytes]:
    """Build the canonical header/payload segments and the JWS signing input.

    Returns `(header_b64, payload_b64, signing_input)`. Key-free **by
    design**: the daemon never signs anything, so no private-key handling lives
    in this module — `scripts/license_tool.py` (D-LIC8) calls this, signs the
    returned bytes with the owner's offline key and joins the three segments.
    Sharing the encoder is what keeps issuer and verifier from ever drifting on
    the wire format.
    """
    header_b64 = encode_segment({"alg": JWS_ALG, "kid": kid, "typ": JWS_TYP})
    payload_b64 = encode_segment(dict(claims))
    return header_b64, payload_b64, f"{header_b64}.{payload_b64}".encode("ascii")


def _public_key_from_reference(reference: str) -> Ed25519PublicKey | None:
    """Parse an `ed25519:<base64url>` verifier reference, or `None`."""
    if not reference.startswith(ED25519_REFERENCE_PREFIX):
        return None
    raw = _decode_segment(reference.removeprefix(ED25519_REFERENCE_PREFIX))
    if raw is None or len(raw) != _RAW_KEY_BYTES:
        return None
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except ValueError:
        return None


def valid_public_reference(reference: str) -> bool:
    """Return whether `reference` is one canonical raw Ed25519 public key."""
    return _public_key_from_reference(reference) is not None


# ---------------------------------------------------------------------------
# Claims + verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, repr=False)
class LicenseClaims:
    """The v1 claim set, already validated.

    `__repr__`/`__str__` are redacted on purpose: `customer_id` is not a
    secret, but it has no business in an accidental log line, and this object is
    exactly the kind of thing that ends up interpolated into one.
    """

    v: int
    lid: str
    iat: datetime
    customer_id: str
    plan: str
    features: tuple[str, ...]
    expires_at: datetime

    def __repr__(self) -> str:
        return (
            f"LicenseClaims(lid={self.lid!r}, plan={self.plan!r}, "
            f"expires_at={self.expires_at.isoformat()!r})"
        )

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class LicenseVerdict:
    """What `verify_license` concluded about one blob.

    `reason` is set **only** when `state == "invalid"` and is always a
    member of `INVALID_REASONS`; `claims` is set for every other state.
    """

    state: VerdictState
    reason: str | None = None
    claims: LicenseClaims | None = None


@dataclass(frozen=True, slots=True)
class EntitlementDecision:
    """The answer `LicenseState.require_entitlement` hands its caller.

    `allowed` is the *recommendation*; the caller owns the posture (D-LIC2).
    `state` is `None` when no license is installed. `reason` is a machine
    token — an `INVALID_REASONS` member, or one of
    `DECISION_FEATURE_NOT_LICENSED` / `DECISION_EXPIRED_GRACE` /
    `DECISION_EXPIRED` — never free text.
    """

    feature: str
    allowed: bool
    state: VerdictState | None = None
    reason: str | None = None


def utcnow() -> datetime:
    """The default injected clock: tz-aware UTC now."""
    return datetime.now(UTC)


def _as_aware(value: datetime) -> datetime:
    """Treat a naive injected `now()` as UTC rather than raising.

    Claims timestamps are refused when naive (that is issuer input); an injected
    clock is *our* code, and a `TypeError` deep in a comparison would be a
    worse failure than assuming the obvious.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def temporal_state(expires_at: datetime, now: datetime) -> VerdictState:
    """Boundary-exact temporal policy (D-LIC1), shared by verify and refresh.

    `now < expires_at` ⇒ valid; `expires_at <= now < expires_at + grace` ⇒
    grace; otherwise expired. Both boundaries are pinned by test.
    """
    moment = _as_aware(now)
    if moment < expires_at:
        return STATE_VALID
    if moment < expires_at + timedelta(seconds=LICENSE_GRACE_S):
        return STATE_EXPIRED_GRACE
    return STATE_EXPIRED


def _parse_timestamp(value: Any) -> datetime | None:
    """ISO-8601, tz-aware. A naive timestamp is refused, never assumed UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _parse_claims(payload: dict[str, Any]) -> LicenseClaims | str:
    """Validate the v1 claim set; return the claims or an invalid reason token.

    Unknown keys are **tolerated and ignored**: additive evolution must not
    force a version bump that bricks fielded daemons. Breaking changes bump
    `v`.
    """
    if "v" not in payload:
        return REASON_SCHEMA_INVALID
    version = payload["v"]
    if isinstance(version, bool) or not isinstance(version, int):
        return REASON_UNSUPPORTED_VERSION
    if version != CLAIMS_VERSION:
        return REASON_UNSUPPORTED_VERSION

    lid = payload.get("lid")
    if not isinstance(lid, str) or not _LID_RE.fullmatch(lid):
        return REASON_SCHEMA_INVALID

    customer_id = payload.get("customer_id")
    if (
        not isinstance(customer_id, str)
        or not customer_id
        or len(customer_id) > _MAX_CUSTOMER_ID
        or customer_id != customer_id.strip()
    ):
        return REASON_SCHEMA_INVALID

    plan = payload.get("plan")
    if not isinstance(plan, str) or not _TOKEN_RE.fullmatch(plan):
        return REASON_SCHEMA_INVALID

    features = payload.get("features")
    if not isinstance(features, list):
        return REASON_SCHEMA_INVALID
    for feature in features:
        # Unknown feature *values* are fine — membership is the only test, so a
        # v1 daemon carries a future SSO feature inertly. The *shape* is not
        # optional: these tokens reach events and doctor details.
        if not isinstance(feature, str) or not _TOKEN_RE.fullmatch(feature):
            return REASON_SCHEMA_INVALID

    iat = _parse_timestamp(payload.get("iat"))
    expires_at = _parse_timestamp(payload.get("expires_at"))
    if iat is None or expires_at is None:
        return REASON_SCHEMA_INVALID

    return LicenseClaims(
        v=version,
        lid=lid,
        iat=iat,
        customer_id=customer_id,
        plan=plan,
        features=tuple(features),
        expires_at=expires_at,
    )


def _resolve_header_key(
    header: dict[str, Any], trusted_keys: Mapping[str, str]
) -> Ed25519PublicKey | str:
    """Enforce the protected-header contract; return the verifier or a reason.

    Every rejection here is a *crypto-semantics* rejection, which is why the
    header is strict where the claims are tolerant: an unknown claim key is
    inert data, an unknown header parameter is an instruction we did not obey.
    """
    # `crit` first: a criticality demand is refused whatever else is wrong
    # with the header, and it deserves its own token rather than being folded
    # into "some unknown parameter".
    if "crit" in header:
        return REASON_CRIT_PRESENT
    if set(header) - JWS_HEADER_KEYS or header.get("typ") != JWS_TYP:
        return REASON_UNKNOWN_HEADER
    if header.get("alg") != JWS_ALG:
        return REASON_UNSUPPORTED_ALG

    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        return REASON_MISSING_KID
    reference = trusted_keys.get(kid)
    if reference is None:
        return REASON_UNKNOWN_KID
    public_key = _public_key_from_reference(reference)
    if public_key is None:
        # A malformed entry in the pinned keyset is our bug, not the license's;
        # it still resolves to "this kid cannot verify anything".
        return REASON_UNKNOWN_KID
    return public_key


def verify_license(
    blob: str,
    *,
    trusted_keys: Mapping[str, str],
    now: Callable[[], datetime] = utcnow,
) -> LicenseVerdict:
    """Verify one compact JWS license offline. Never raises, never does I/O.

    The order is deliberate: structure, then header semantics, then the
    **signature over the exact received `header_b64.payload_b64` bytes**, and
    only then the claims. Nothing from an unverified payload is parsed into a
    decision, and the signing input is never rebuilt from a parse.
    """
    if not isinstance(blob, str):  # pragma: no cover - defensive at the seam
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)
    candidate = blob.strip()
    if not candidate or len(candidate) > MAX_LICENSE_BYTES:
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)

    segments = candidate.split(".")
    if len(segments) != 3:
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)
    header_b64, payload_b64, signature_b64 = segments

    header_raw = _decode_segment(header_b64)
    payload_raw = _decode_segment(payload_b64)
    signature = _decode_segment(signature_b64)
    if header_raw is None or payload_raw is None or signature is None:
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)

    header = _load_json_object(header_raw)
    if header is None:
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)

    resolved = _resolve_header_key(header, trusted_keys)
    if isinstance(resolved, str):
        return LicenseVerdict(STATE_INVALID, resolved)
    public_key = resolved

    # THE contract: the signing input is the bytes that arrived, spliced back
    # exactly as received — never `encode_segment(parsed_header)`.
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    try:
        public_key.verify(signature, signing_input)
    except InvalidSignature:
        return LicenseVerdict(STATE_INVALID, REASON_BAD_SIGNATURE)

    # Only NOW is the payload interpreted. Its base64 was decoded above for the
    # structural check, but nothing from it reached a decision before the
    # signature over the exact received bytes passed.
    payload = _load_json_object(payload_raw)
    if payload is None:
        return LicenseVerdict(STATE_INVALID, REASON_MALFORMED)

    claims = _parse_claims(payload)
    if isinstance(claims, str):
        return LicenseVerdict(STATE_INVALID, claims)

    return LicenseVerdict(temporal_state(claims.expires_at, now()), claims=claims)


# ---------------------------------------------------------------------------
# File custody (D-LIC5)
# ---------------------------------------------------------------------------


def resolve_license_file(file: str | None, data_dir: str) -> Path:
    """Resolve where the installed license lives.

    `file` set wins (the `[security].secrets_key_file` /
    `[link].key_file` idiom); unset means `<data_dir>/license.jws`, so a
    `data_dir` move never strands a stale absolute default and a config
    null-delete reverts cleanly.
    """
    if file:
        return Path(file).expanduser()
    return Path(data_dir).expanduser() / LICENSE_FILE_NAME


def read_license_file(path: Path) -> str | None:
    """Read the installed blob, or `None` when nothing is installed.

    Synchronous by design (callers on the event loop wrap it in
    `asyncio.to_thread`). `O_NOFOLLOW` means a symlinked license is refused
    rather than followed somewhere the daemon did not choose; `O_NONBLOCK`
    plus the `S_ISREG` check below refuse FIFOs and devices, so a license path
    pointing at a writerless FIFO can never park the `open` (and with it the
    lifespan that awaits `build_license_state`) forever — it degrades to the
    same doctor-visible invalid state as any other unreadable file
    (never-abort-boot, D-LIC2). `O_NONBLOCK` is inert for regular-file reads.
    Absent is not an error — an unlicensed daemon is the ordinary free local
    product.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LicenseError(
            f"cannot read license {path}: {exc.strerror} (a symlinked license file is refused)"
        ) from exc
    except ValueError as exc:
        # `os.open` raises ValueError — NOT OSError — for a path carrying an
        # embedded NUL. The settings validator refuses such a path at the
        # operator's moment, but an already-persisted one must still degrade to
        # the doctor-visible invalid state rather than escape
        # `build_license_state` and abort boot (never-abort-boot, D-LIC2).
        # `ValueError` has no `strerror`, so the message is fixed text.
        raise LicenseError(f"cannot read license {path}: invalid path") from exc
    # A license is a regular file or it is nothing. The descriptor — not the
    # path — is stat'd, so there is no TOCTOU window between the check and the
    # read. Path-bearing (the symlink-message idiom) but never blob-bearing.
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise LicenseError(f"cannot read license {path}: not a regular file")
    try:
        with os.fdopen(fd, "rb") as handle:
            # Two bytes past the cap — still bounded, never an unbounded decode.
            # One is the trailing newline `install_license_file` writes (it is
            # framing, not blob), the other makes an oversized blob visible as
            # oversized rather than silently truncated into "malformed".
            content = handle.read(MAX_LICENSE_BYTES + 2)
            # One more byte establishes EOF. Without it, bytes past the window
            # can never be observed: a file whose first `MAX_LICENSE_BYTES + 2`
            # bytes are a valid blob padded with whitespace strips back to that
            # blob, and arbitrary junk beyond would be accepted silently —
            # defeating the strict one-line-file contract. A legitimate file is
            # at most the cap plus its trailing newline, so the probe always
            # returns b"" for one.
            if handle.read(1):
                raise LicenseError(f"license {path} is larger than {MAX_LICENSE_BYTES} bytes")
    except OSError as exc:
        raise LicenseError(f"cannot read license {path}: {exc.strerror}") from exc
    try:
        blob = content.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LicenseError(f"license {path} is not ascii") from exc
    # The cap is measured on the stripped blob, exactly as the HTTP `blob`
    # bound is, so a license the install route accepted always reads back —
    # never accepted-then-rejected at the next boot.
    if len(blob) > MAX_LICENSE_BYTES:
        raise LicenseError(f"license {path} is larger than {MAX_LICENSE_BYTES} bytes")
    return blob


def install_license_file(path: Path, blob: str) -> None:
    """Write `blob` to `path` atomically, owner-only.

    `os.replace` — not the `node.key` no-replace `os.link` — is the
    correct install here: overwriting an existing license **is** the renewal
    semantics, and losing a race to a concurrent install just means last-writer
    wins on an admin-only, idempotency-keyed route. The parent is never created:
    the default one is `data_dir` (whose permissions the `data_dir_perms`
    doctor check already polices) and an operator-chosen one is theirs, so a
    missing parent is a structured error rather than a mkdir spree.
    """
    parent = path.parent
    if not parent.is_dir():
        raise LicenseError(f"cannot install license: {parent} is not a directory")
    tmp = parent / f".{path.name}.{os.getpid()}.{token_hex(4)}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, LICENSE_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(f"{blob.strip()}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_dir(parent)
    except (OSError, ValueError) as exc:
        # `ValueError` covers a path carrying an embedded NUL (`os.open`
        # raises it, not `OSError`); it has no `strerror`, so the detail
        # falls back to the exception itself. Structured 500
        # `license.write_failed` either way, never a naked crash.
        # A tmp name derived from an unopenable path is itself unopenable —
        # there is nothing to clean up, and the write error is the real news.
        with suppress(OSError, ValueError):
            tmp.unlink(missing_ok=True)
        detail = getattr(exc, "strerror", None) or exc
        raise LicenseError(f"cannot write license {path}: {detail}") from exc


def remove_license_file(path: Path) -> bool:
    """Delete the installed license; return whether a file was actually removed.

    The unlink report-what-happened idiom: a double remove is not an error.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        # `ValueError` (embedded NUL in the path) has no `strerror` — see
        # `install_license_file`. Structured 500, never a naked crash.
        detail = getattr(exc, "strerror", None) or exc
        raise LicenseError(f"cannot remove license {path}: {detail}") from exc
    return True


# ---------------------------------------------------------------------------
# The app.state holder (D-LIC2 / D-LIC6)
# ---------------------------------------------------------------------------


class LicenseState:
    """What the running daemon knows about its license, on `app.state.license`.

    Constructed at boot from the file and **refreshed in place** by the install
    /remove routes, so doctor and `/capabilities` tell the truth without a
    restart (the `[license].file` *path* is restart-keyed; the file's
    *content* is not — no surface may assume boot-frozen license state).

    Reads are pure: file I/O happened at boot or at install, so
    `state` recomputes only the **temporal** verdict from the stored
    claims and the injected clock. A grace transition therefore surfaces with no
    poller, no timer and no restart.
    """

    __slots__ = ("_now", "_verdict")

    def __init__(
        self,
        verdict: LicenseVerdict | None = None,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._verdict = verdict
        self._now = now

    @property
    def installed(self) -> bool:
        """Whether a license file was present (valid or not)."""
        return self._verdict is not None

    @property
    def clock(self) -> Callable[[], datetime]:
        """The injected clock this holder computes `state` on.

        Surfaces (the doctor check, the install route) borrow it rather than
        reading `datetime.now` themselves, so a rendered "expires in N days"
        can never disagree with the state it is printed beside — and a test
        that injects a clock moves both together.
        """
        return self._now

    @property
    def claims(self) -> LicenseClaims | None:
        """The verified claims, or `None` when absent or invalid."""
        return self._verdict.claims if self._verdict is not None else None

    @property
    def reason(self) -> str | None:
        """The `INVALID_REASONS` token, or `None` when not invalid."""
        return self._verdict.reason if self._verdict is not None else None

    @property
    def state(self) -> VerdictState | None:
        """Live state: `None` when absent, else the recomputed verdict."""
        if self._verdict is None:
            return None
        if self._verdict.claims is None:
            return STATE_INVALID
        return temporal_state(self._verdict.claims.expires_at, self._now())

    def replace(self, verdict: LicenseVerdict | None) -> None:
        """Swap the stored verdict in place (install/remove refresh the holder).

        Mutating the holder rather than rebinding `app.state.license` keeps
        every already-captured reference truthful.
        """
        self._verdict = verdict

    def require_entitlement(self, feature: str) -> EntitlementDecision:
        """Decide whether `feature` is entitled. **The caller owns the posture.**

        The recommendation (D-LIC2):

        * **absent** ⇒ allowed — unlicensed is the free local product, and the
          relay decides remote (W-D11);
        * **invalid** ⇒ allowed with an advisory — a corrupt file must never
          disable more than no file at all;
        * **valid / in grace**, feature listed ⇒ allowed (grace carries an
          advisory);
        * **valid / in grace**, feature absent ⇒ not allowed;
        * **expired past grace** ⇒ not allowed.
        """
        state = self.state
        if state is None:
            return EntitlementDecision(feature, allowed=True)
        claims = self.claims
        if state == STATE_INVALID or claims is None:
            return EntitlementDecision(feature, allowed=True, state=state, reason=self.reason)
        if feature not in claims.features:
            return EntitlementDecision(
                feature, allowed=False, state=state, reason=DECISION_FEATURE_NOT_LICENSED
            )
        if state == STATE_EXPIRED:
            return EntitlementDecision(feature, allowed=False, state=state, reason=DECISION_EXPIRED)
        if state == STATE_EXPIRED_GRACE:
            return EntitlementDecision(
                feature, allowed=True, state=state, reason=DECISION_EXPIRED_GRACE
            )
        return EntitlementDecision(feature, allowed=True, state=state)

    def __repr__(self) -> str:
        return f"LicenseState(installed={self.installed}, state={self.state!r})"

    __str__ = __repr__
