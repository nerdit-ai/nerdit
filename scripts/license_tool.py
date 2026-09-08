#!/usr/bin/env python3
"""Issue and verify Nerdit product licenses offline.

See `docs/guide/license.md`. Product licenses are separate from this repository's
Apache-2.0 license; onboarding does not install one. No command reads config,
opens a socket or contacts a daemon.

Commands:
    keygen: Create an exclusive 0600 private-key file and print its public key.
        Never print the private key; warn if its destination is a git worktree.
    sign: Emit a compact JWS, assigning a license ID and issuance timestamp.
        Use the daemon's encoder and verifier, require a matching key for a
        pinned key ID, and warn for unknown keys or already-expired licenses.
    inspect: Verify with the supplied public key or this checkout's trusted keys.

Exit codes: 0 for success, 1 for refusal/failure, 2 for usage/environment errors.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# Prefer this checkout's verifier and keyset: a stale install could bypass
# the pinned-key check. Standalone copies use the normal import path.
_CHECKOUT_SRC = Path(__file__).resolve().parent.parent / "src"
if (_CHECKOUT_SRC / "nerdit" / "core" / "license.py").is_file():
    sys.path.insert(0, str(_CHECKOUT_SRC))
from nerdit.core import license as lic  # noqa: E402  (after the sys.path pin above)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

#: Private-key files are owner-only from the open flags, like every other piece
#: of key material in this repo (``identity.py``, ``secrets.key``).
KEY_FILE_MODE = 0o600


def _b64(raw: bytes) -> str:
    """Encode canonical unpadded URL-safe base64 for a JWS segment."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    """Strip surrounding whitespace, restore padding and decode URL-safe base64."""
    stripped = value.strip()
    return base64.urlsafe_b64decode(stripped + "=" * (-len(stripped) % 4))


def _parse_expiry(value: str) -> datetime | None:
    """Parse a timezone-aware ISO-8601 timestamp; return None if invalid or naive."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return None if parsed.tzinfo is None else parsed


def _fail(message: str, *, code: int = EXIT_REFUSED) -> int:
    """Print one error line to stderr and return an exit code."""
    print(f"error: {message}", file=sys.stderr)
    return code


def _inside_git_worktree(path: Path) -> Path | None:
    """Return the worktree root when ``path`` sits inside one, else ``None``."""
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    """Read a private key written by keygen.

    Raises:
        ValueError: The file is unreadable or is not a raw Ed25519 key.
    """
    try:
        raw = _b64decode(path.read_text(encoding="ascii"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not a base64 Ed25519 private key") from exc
    if len(raw) != 32:
        raise ValueError(f"{path} is not a 32-byte raw Ed25519 private key")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _public_reference(key: Ed25519PrivateKey) -> str:
    """The ``ed25519:<b64url>`` verifier line for a private key's public half."""
    raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return f"{lic.ED25519_REFERENCE_PREFIX}{_b64(raw)}"


# ---------------------------------------------------------------------------
# keygen
# ---------------------------------------------------------------------------


def cmd_keygen(args: argparse.Namespace) -> int:
    """Mint a signing key: private half to a 0600 file, public half to stdout."""
    out = Path(args.out).expanduser()
    if out.exists() or out.is_symlink():
        # Never overwrite: a silent clobber would invalidate every license ever
        # issued with the previous key, with no way back.
        return _fail(f"{out} already exists — refusing to overwrite a signing key")
    if not out.parent.is_dir():
        return _fail(f"{out.parent} is not a directory", code=EXIT_USAGE)

    worktree = _inside_git_worktree(out.resolve().parent)

    key = Ed25519PrivateKey.generate()
    private_b64 = _b64(key.private_bytes_raw())
    try:
        # ``O_EXCL`` + mode on the open flags: there is never a window in which
        # the private key exists world-readable.
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, KEY_FILE_MODE)
    except OSError as exc:
        # Creation itself failed — nothing exists to clean up.
        return _fail(f"cannot write {out}: {exc.strerror}")
    try:
        with os.fdopen(fd, "w", encoding="ascii") as handle:
            handle.write(f"{private_b64}\n")
            # The only copy of a production signing key: durable before we
            # print "written" (the ``install_license_file`` idiom).
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        # Remove incomplete keys so the no-overwrite guard allows a retry.
        # The new key exists only in memory; no issued license depends on it.
        out.unlink(missing_ok=True)
        return _fail(f"cannot write {out}: {exc.strerror} (partial file removed — retry)")

    if worktree is not None:
        print(
            "WARNING ------------------------------------------------------\n"
            f"  The private key was written INSIDE a git worktree ({worktree}).\n"
            "  It must never be committed, pushed, backed up to the cloud or\n"
            "  copied into CI. Move it to owner-controlled offline storage and\n"
            "  delete it from here.\n"
            "--------------------------------------------------------------",
            file=sys.stderr,
        )

    # Only the PUBLIC half is ever printed. ``private_b64`` is deliberately not
    # referenced past this point.
    print(f"kid: {args.kid}")
    print("Paste this line into TRUSTED_LICENSE_KEYS in src/nerdit/core/license.py:")
    print(f'    "{args.kid}": "{_public_reference(key)}",')
    print(f"Private key written to {out} (mode 0600) — it is the only copy.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------


def _parse_features(raw: str) -> list[str]:
    """Split a comma-separated feature list. Shape is validated by the verifier."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def cmd_sign(args: argparse.Namespace) -> int:
    """Emit one compact JWS license on stdout — the customer's deliverable."""
    try:
        key = _load_private_key(Path(args.key).expanduser())
    except ValueError as exc:
        return _fail(str(exc), code=EXIT_USAGE)

    expires_at = _parse_expiry(args.expires_at)
    if expires_at is None:
        return _fail(
            f"--expires-at {args.expires_at!r} is not a tz-aware ISO-8601 timestamp "
            "(e.g. 2027-01-01T00:00:00+00:00)",
            code=EXIT_USAGE,
        )

    claims: dict[str, Any] = {
        "v": lic.CLAIMS_VERSION,
        "lid": args.lid or uuid4().hex,
        "iat": datetime.now(UTC).isoformat(),
        "customer_id": args.customer_id,
        "plan": args.plan,
        "features": _parse_features(args.features),
        "expires_at": expires_at.isoformat(),
    }

    header_b64, payload_b64, signing_input = lic.build_signing_input(kid=args.kid, claims=claims)
    blob = f"{header_b64}.{payload_b64}.{_b64(key.sign(signing_input))}"

    # Self-verification alone cannot detect the wrong signing key. For a pinned
    # key ID, require the shipped public key or daemons reject the signature.
    reference = _public_reference(key)
    pinned = lic.TRUSTED_LICENSE_KEYS.get(args.kid)
    if pinned is not None and pinned != reference:
        return _fail(
            f"--key is not the private half of the pinned public key for kid {args.kid!r} — "
            "every daemon carrying the baked keyset would reject this license "
            "(bad_signature); check the --key path"
        )
    if pinned is None:
        print(
            f"warning: kid {args.kid!r} is not in the baked TRUSTED_LICENSE_KEYS — "
            "daemons at this commit would report unknown_kid for this license",
            file=sys.stderr,
        )
    verdict = lic.verify_license(blob, trusted_keys={args.kid: reference})
    if verdict.state == lic.STATE_INVALID:
        return _fail(
            f"refusing to emit a license the daemon would reject ({verdict.reason}) — "
            "check --plan/--features (tokens are ^[a-z0-9_]{1,64}$) and --customer-id"
        )
    if verdict.state != lic.STATE_VALID:
        print(
            f"warning: this license is already {verdict.state} at issuance "
            "(--expires-at is in the past)",
            file=sys.stderr,
        )

    # stdout is the deliverable and nothing else, so it can be piped to a file.
    print(blob)
    print(
        f"issued lid={claims['lid']} plan={args.plan} kid={args.kid} "
        f"expires_at={claims['expires_at']} state={verdict.state}",
        file=sys.stderr,
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def _read_blob(source: str) -> str:
    """Read a blob from a path or, for ``-``, from stdin."""
    if source == "-":
        return sys.stdin.read().strip()
    return Path(source).expanduser().read_text(encoding="ascii").strip()


def cmd_inspect(args: argparse.Namespace) -> int:
    """Verify a blob and print what a daemon would conclude about it."""
    try:
        blob = _read_blob(args.file)
    except (OSError, UnicodeDecodeError) as exc:
        return _fail(f"cannot read the license: {exc}", code=EXIT_USAGE)
    if not blob:
        return _fail("the license is empty", code=EXIT_USAGE)

    if args.pub:
        if not lic.valid_public_reference(args.pub):
            return _fail(
                f"--pub must be an {lic.ED25519_REFERENCE_PREFIX}<base64url> reference",
                code=EXIT_USAGE,
            )
        # ``kid`` is read from the blob's own header; enrolling the reference
        # under every kid would defeat the pinning this format exists for, so
        # the reference is enrolled under exactly the kid the caller names.
        trusted = {args.kid: args.pub} if args.kid else _kid_from_blob(blob, args.pub)
    else:
        # The keyset baked into THIS checkout — a blob signed by any kid not
        # pinned there (test kids included) reads ``unknown_kid`` here.
        trusted = dict(lic.TRUSTED_LICENSE_KEYS)

    verdict = lic.verify_license(blob, trusted_keys=trusted)
    print(f"state:       {verdict.state}")
    if verdict.claims is None:
        print(f"reason:      {verdict.reason}")
        return EXIT_REFUSED
    claims = verdict.claims
    print(f"lid:         {claims.lid}")
    print(f"plan:        {claims.plan}")
    print(f"features:    {', '.join(claims.features) or '(none)'}")
    # The owner's own issuance tool is the one place ``customer_id`` is rendered
    # on purpose — it is what makes a license identifiable at support time. No
    # daemon surface (doctor, events, audit, CLI) ever prints it.
    print(f"customer_id: {claims.customer_id}")
    print(f"iat:         {claims.iat.isoformat()}")
    print(f"expires_at:  {claims.expires_at.isoformat()}")
    return EXIT_OK if verdict.state != lic.STATE_EXPIRED else EXIT_REFUSED


def _kid_from_blob(blob: str, reference: str) -> dict[str, str]:
    """Map the blob's key ID to the supplied reference for --pub without --kid.

    Signature verification must still succeed with that exact public key.
    """
    segments = blob.split(".")
    if len(segments) != 3:
        return {}
    try:
        header = json.loads(_b64decode(segments[0]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    kid = header.get("kid") if isinstance(header, dict) else None
    return {kid: reference} if isinstance(kid, str) and kid else {}


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The three-subcommand CLI."""
    parser = argparse.ArgumentParser(
        prog="license_tool.py",
        description="Offline Nerdit product-license issuance (keygen / sign / inspect).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    keygen = sub.add_parser("keygen", help="mint the owner's offline Ed25519 signing key")
    keygen.add_argument("--out", required=True, help="where to write the PRIVATE key (0600)")
    keygen.add_argument("--kid", required=True, help="key id to publish, e.g. nerdit-lic-2026-1")
    keygen.set_defaults(func=cmd_keygen)

    sign = sub.add_parser("sign", help="sign one license; the compact JWS goes to stdout")
    sign.add_argument("--key", required=True, help="path to the private key written by keygen")
    sign.add_argument("--kid", required=True, help="the kid published in TRUSTED_LICENSE_KEYS")
    sign.add_argument("--customer-id", required=True, help="opaque cloud customer id")
    sign.add_argument("--plan", required=True, help="plan token, ^[a-z0-9_]{1,64}$")
    sign.add_argument(
        "--features",
        default=lic.FEATURE_REMOTE_LINK,
        help=f"comma-separated feature tokens (default: {lic.FEATURE_REMOTE_LINK})",
    )
    sign.add_argument(
        "--expires-at",
        required=True,
        help="tz-aware ISO-8601, e.g. 2027-01-01T00:00:00+00:00",
    )
    sign.add_argument("--lid", default=None, help="license id (default: a fresh uuid4 hex)")
    sign.set_defaults(func=cmd_sign)

    inspect = sub.add_parser("inspect", help="verify a license and print its claims")
    inspect.add_argument("file", nargs="?", default="-", help="license file, or '-' for stdin")
    inspect.add_argument("--pub", default=None, help="verifier reference, ed25519:<base64url>")
    inspect.add_argument(
        "--kid",
        default=None,
        help="kid to enroll --pub under (default: the blob's own)",
    )
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
