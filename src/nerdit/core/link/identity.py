"""Generate and load Ed25519 node identities for the frozen node-link contract.

Public verifiers, fingerprints, proof bytes, and signatures must match
`nerdit-cloud/security/node_identity.py`; independent golden vectors in
`tests/data/node_link_v1/` verify compatibility without importing the producer.
Production keys are generated locally, never copied from development fixtures.
The cloud stores only the verifier and fingerprint.

Private material lives only in `node.key`, never settings, audit, diagnostics,
MCP output, or logs. `NodeIdentity` has no public key-material accessor and its
repr exposes only public metadata. Generation may log a fingerprint and lax
parent permissions may log a directory path; failures use key-free,
path-bearing `LinkIdentityError` messages.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import stat
from datetime import datetime
from pathlib import Path
from secrets import token_hex

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

logger = logging.getLogger("nerdit.link.identity")

#: Canonical prefix of a public credential reference (cloud parity).
ED25519_REFERENCE_PREFIX = "ed25519:"

#: `node.key` is owner-read/write only, always and everywhere. The directory
#: mode is only ever *applied* to a directory this module itself created (the
#: `<data_dir>/link/` default); an operator-chosen `[link].key_file` parent
#: is never mutated — a lax one is reported, not repaired (see
#: `_warn_lax_parent`).
KEY_FILE_MODE = 0o600
KEY_DIR_MODE = 0o700

#: Raw Ed25519 keys are exactly 32 bytes.
_RAW_KEY_BYTES = 32


class LinkIdentityError(Exception):
    """A node-key file could not be read, parsed, or written.

    Messages may name the offending **path**; they never carry key bytes.
    """


# ---------------------------------------------------------------------------
# Vendored pure helpers — byte-compatible with the cloud at pin
# d53507081821d62e99991e2ce9f675e80cdca93e.
# ---------------------------------------------------------------------------


def _b64decode(value: str) -> bytes:
    """Decode unpadded urlsafe base64 (re-padding as the cloud does)."""
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _b64encode(value: bytes) -> str:
    """Encode as unpadded urlsafe base64 ascii (the canonical wire form)."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def credential_fingerprint(reference: str) -> str:
    """Return a stable non-secret fingerprint for durable/audit metadata."""
    return hashlib.sha256(reference.encode()).hexdigest()


def valid_public_reference(reference: str) -> bool:
    """Return whether a reference is one canonical raw Ed25519 public key."""
    if not reference.startswith(ED25519_REFERENCE_PREFIX):
        return False
    encoded = reference.removeprefix(ED25519_REFERENCE_PREFIX)
    try:
        raw = _b64decode(encoded)
        Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError):
        return False
    return len(raw) == _RAW_KEY_BYTES and _b64encode(raw) == encoded


def public_reference(private_key_b64: str) -> str:
    """Derive the public credential reference for a raw Ed25519 private key."""
    private_key = Ed25519PrivateKey.from_private_bytes(_b64decode(private_key_b64))
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return f"{ED25519_REFERENCE_PREFIX}{_b64encode(public)}"


def proof_message(  # noqa: PLR0913 - the ten fields are the frozen cloud contract
    *,
    challenge: str,
    relay_id: str,
    protocol: str,
    node_id: str,
    node_name: str,
    daemon_version: str,
    uptime_s: int,
    capability_token: str,
    capability_expires_at: datetime,
    capability_role: str,
) -> bytes:
    """Canonical bytes signed by the node and verified by the relay.

    Exactly ten keys, `sort_keys=True` and `separators=(",", ":")` — the
    encoding is part of the contract, not a formatting choice.
    """
    payload = {
        "capability_expires_at": capability_expires_at.isoformat(),
        "capability_role": capability_role,
        "capability_token": capability_token,
        "challenge": challenge,
        "daemon_version": daemon_version,
        "node_id": node_id,
        "node_name": node_name,
        "protocol": protocol,
        "relay_id": relay_id,
        "uptime_s": uptime_s,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def sign_node_proof(private_key_b64: str, message: bytes) -> str:
    """Sign a challenge-bound hello with node-held private material."""
    private_key = Ed25519PrivateKey.from_private_bytes(_b64decode(private_key_b64))
    return _b64encode(private_key.sign(message))


def verify_node_proof(reference: str, message: bytes, signature_b64: str) -> bool:
    """Verify a proof using only the durable public credential reference."""
    if not reference.startswith(ED25519_REFERENCE_PREFIX):
        return False
    try:
        public_raw = _b64decode(reference.removeprefix(ED25519_REFERENCE_PREFIX))
        signature = _b64decode(signature_b64)
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# The node's own identity
# ---------------------------------------------------------------------------


class NodeIdentity:
    """The node's Ed25519 credential: verifier, fingerprint, and signing.

    The private key is held privately and has **no public accessor**; the only
    thing it can do from the outside is `sign`. String forms are redacted
    by construction so an identity can be dropped into any log, audit payload
    or doctor detail without leaking key material.
    """

    __slots__ = ("_private_key_b64", "fingerprint", "verifier")

    def __init__(self, private_key_b64: str) -> None:
        try:
            raw = _b64decode(private_key_b64)
        except (ValueError, TypeError) as exc:
            raise LinkIdentityError("node key is not valid base64url") from exc
        if len(raw) != _RAW_KEY_BYTES:
            raise LinkIdentityError(f"node key must decode to exactly {_RAW_KEY_BYTES} bytes")
        self._private_key_b64 = private_key_b64
        self.verifier: str = public_reference(private_key_b64)
        self.fingerprint: str = credential_fingerprint(self.verifier)

    def sign(self, message: bytes) -> str:
        """Sign `message` (typically `proof_message` output)."""
        return sign_node_proof(self._private_key_b64, message)

    def __repr__(self) -> str:
        return f"NodeIdentity(verifier={self.verifier!r}, fingerprint={self.fingerprint!r})"

    __str__ = __repr__


def resolve_key_file(key_file: str | None, data_dir: str) -> Path:
    """Resolve where `node.key` lives.

    `key_file` set wins (the `[security].secrets_key_file` idiom); unset
    means `<data_dir>/link/node.key`, so a `data_dir` move never strands a
    stale absolute default and a config null-delete reverts cleanly.
    """
    if key_file:
        return Path(key_file).expanduser()
    return Path(data_dir).expanduser() / "link" / "node.key"


def _tighten(path: Path, mode: int) -> None:
    """Drop group/other bits from an existing path, best effort but loud."""
    try:
        current = os.stat(path).st_mode
        if current & 0o077:
            os.chmod(path, mode)
    except OSError as exc:
        raise LinkIdentityError(f"cannot secure permissions on {path}: {exc.strerror}") from exc


def _warn_lax_parent(parent: Path) -> None:
    """Report — never repair — a group/other-accessible key directory.

    `[link].key_file` may point anywhere (`/tmp/node.key`, a path under
    `$HOME`); chmodding *that* directory to `0o700` would silently strip
    bits this module does not own (`/tmp`'s sticky bit, a shared group, a
    setgid tree) or fail outright when the daemon does not own it. So a
    pre-existing parent is left exactly as the operator made it and the risk is
    named once, with the path but never any key material.
    """
    try:
        mode = os.stat(parent).st_mode
    except OSError:  # pragma: no cover - reported by the caller's own failure
        return
    if mode & 0o077:
        logger.warning(
            "Node key directory %s is group/other-accessible (mode %04o); "
            "the key file itself stays 0o600, but the directory is left as "
            "configured — tighten it if that is not intended.",
            parent,
            stat.S_IMODE(mode),
        )


def _fsync_dir(path: Path) -> None:
    """`fsync` a directory so a rename inside it survives power loss."""
    dir_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _read_key_file(key_file: Path) -> str:
    """Read the key file without ever following a symlink.

    `O_NONBLOCK` plus the `S_ISREG` check refuse FIFOs and devices, so a
    key path pointing at a writerless FIFO can never park the `open` (and
    with it the lifespan that loads the identity) forever — the same fix as
    the license path's `read_license_file` (PR #118 Codex round 2).
    `O_NONBLOCK` is inert for regular-file reads.
    """
    try:
        fd = os.open(key_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise LinkIdentityError(
            f"cannot read node key {key_file}: {exc.strerror} (a symlinked key file is refused)"
        ) from exc
    # The descriptor — not the path — is stat'd, so there is no TOCTOU window
    # between the check and the read.
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise LinkIdentityError(f"cannot read node key {key_file}: not a regular file")
    try:
        with os.fdopen(fd, "rb") as handle:
            content = handle.read()
    except OSError as exc:
        raise LinkIdentityError(f"cannot read node key {key_file}: {exc.strerror}") from exc
    try:
        return content.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LinkIdentityError(f"node key {key_file} is not ascii base64url") from exc


def _generate_key_file(key_file: Path) -> str:
    """Create `key_file` atomically with a fresh key; return its b64 form."""
    parent = key_file.parent
    owns_parent = not parent.exists()
    try:
        os.makedirs(parent, mode=KEY_DIR_MODE, exist_ok=True)
        if owns_parent:
            # makedirs' mode is umask-masked, so state the intent explicitly —
            # but only for the directory this call just created (the
            # `<data_dir>/link/` default). A pre-existing operator-chosen
            # parent is reported, never mutated.
            os.chmod(parent, KEY_DIR_MODE)
    except OSError as exc:
        raise LinkIdentityError(
            f"cannot create node key directory {parent}: {exc.strerror}"
        ) from exc
    if not owns_parent:
        _warn_lax_parent(parent)

    raw = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    private_key_b64 = _b64encode(raw)

    # Atomic write, the ConfigStore.commit / SecretManager idiom: the mode
    # comes from the open flags, so there is never a world-readable window.
    # Durable, too: the *directory* is fsynced after the install, because a
    # lost install would leave `node.key` absent on the next boot and
    # `load_or_create_identity` would silently mint a DIFFERENT identity —
    # permanently breaking the ADR-W2 claim binding of an already-enrolled node.
    #
    # The temp name is UNIQUE per attempt (pid + random) and the install is a
    # no-replace `os.link`: a fixed temp name made an interrupted first boot
    # poison the next one (`O_EXCL` EEXIST on the orphan), and `os.replace`
    # let a racing second caller overwrite an identity the first had already
    # returned. Losing an install race is not an error — the loser adopts the
    # installed key. A crash-orphaned `.*.tmp` holds a random key that was
    # never installed nor enrolled; it is inert and never blocks a retry.
    tmp = parent / f".{key_file.name}.{os.getpid()}.{token_hex(4)}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, KEY_FILE_MODE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(f"{private_key_b64}\n".encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, key_file)
        except FileExistsError:
            # Another caller installed a key between our exists() check and
            # now: theirs is the node's identity — adopt it, never replace it.
            tmp.unlink(missing_ok=True)
            return _read_key_file(key_file)
        tmp.unlink(missing_ok=True)
        _fsync_dir(parent)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise LinkIdentityError(f"cannot write node key {key_file}: {exc.strerror}") from exc
    return private_key_b64


def load_or_create_identity(key_file: Path) -> NodeIdentity:
    """Load the node identity from `key_file`, generating it when absent.

    Synchronous by design (callers on the event loop wrap it in
    `asyncio.to_thread`). A **corrupt or malformed** key file is a loud
    `LinkIdentityError`, never a silent regeneration: regenerating would
    change the node's enrolled cloud identity and permanently break the ADR-W2
    claim binding.
    """
    if key_file.exists():
        private_key_b64 = _read_key_file(key_file)
        identity = NodeIdentity(private_key_b64)
        _tighten(key_file, KEY_FILE_MODE)
        # The directory is the operator's (`[link].key_file` may point
        # anywhere): report a lax one, never chmod it out from under them.
        _warn_lax_parent(key_file.parent)
        return identity

    private_key_b64 = _generate_key_file(key_file)
    identity = NodeIdentity(private_key_b64)
    logger.info("Generated node link identity (fingerprint %s)", identity.fingerprint)
    return identity
