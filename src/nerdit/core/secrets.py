"""Store service secrets encrypted at rest and inject values only at launch.

`<data_dir>/secrets/{service}.enc` holds an AES-256-GCM envelope over JSON,
which preserves whitespace, newlines, equals signs, and Unicode. The API is
write-only: return key names, never values in responses, logs, or audit.

The 32-byte master key is stored separately at `secrets.key` (or the configured
key path), hex-encoded with mode 0600. Envelope AAD binds ciphertext to the
storage name; an eight-hex SHA-256 key ID identifies wrong-key errors.
Malformed/decrypt failures raise `SecretDecryptError`; only absent files yield
an empty mapping. Errors may name paths, service names, or key IDs, never key,
nonce, or ciphertext bytes.

Legacy plaintext files migrate at startup or on read. Rotation stages
`secrets.key.new`, permits reads with either key, and resumes after crashes.
Create secret files as 0600 and directories as 0700, never chmod after exposing
values. Atomic temp-file replacement prevents truncated writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import threading
from base64 import b64decode, b64encode
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("nerdit.secrets")

# A service name must be a DNS label (same rule as ServiceCreateRequest.name /
# DeployConfig.name). Enforced here too so a crafted name can never escape the
# secrets directory via path separators or `..`.
_DNS_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

# A secret KEY name must be a valid POSIX/env-var identifier so it can be handed
# to `ContainerConfig.env` at launch. This is stricter than POSIX strictly
# requires (which permits e.g. a leading digit in some shells) — deliberate: it
# matches the env-var convention every consumer expects, and rejects the
# undeliverable/ambiguous names (empty, `=`-bearing, whitespace/newline) an
# agent could otherwise store and never remove.
_SECRET_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Internal storage name for the shared (global) scope. The leading
# underscore provably cannot come out of `_DNS_LABEL_RE`, so `_shared.enc`
# can never collide with any past or future service file. User-facing surfaces
# keep the word `shared`; routes translate.
SHARED_SCOPE = "_shared"

# Internal storage names above the service scope (D-P40-8), same underscore
# argument as `_shared`: `_project-<prj_id>` is the project scope, id-keyed so
# a project rename never re-encrypts (the stem is the AAD); `_env-<prj_id>-<env>`
# is reserved for phase 3 and admitted so it never needs a second grammar
# change, but nothing writes it. The id grammar is `mint_project_id`'s.
_PROJECT_STEM_RE = re.compile(r"_project-prj_[a-z2-7]{16}")
_ENV_STEM_RE = re.compile(r"_env-prj_[a-z2-7]{16}-(?!.*--)[a-z0-9]([a-z0-9-]{0,18}[a-z0-9])?")

# AES-256-GCM envelope constants. The AAD prefix is versioned so a future
# format change can coexist with v1 files.
_ENVELOPE_VERSION = 1
_ENVELOPE_ALG = "aes-256-gcm"
_AAD_PREFIX = b"nerdit-secret:v1:"
_KEY_BYTES = 32
_NONCE_BYTES = 12


class InvalidServiceName(ValueError):  # noqa: N818 — public API name
    """Raised when a service name is not a valid DNS label."""


class InvalidSecretKey(ValueError):  # noqa: N818 — public API name
    """Raised when a secret *key name* is not a valid env-var identifier.

    Messages carry the offending key **name** only (key names are already the
    public `list_keys` contract) — never a secret value.
    """


class InvalidSecretValue(ValueError):  # noqa: N818 — public API name
    """Raised when a secret *value* holds a NUL or non-printable control char.

    Such a value is undeliverable as a POSIX env var (NUL) or is a
    log/terminal-injection hazard (other C0 controls). Messages carry the
    key **name** only — never any fragment of the value (redaction invariant).
    """


def validate_secret_items(values: dict) -> None:
    """Validate incoming secret key names and values; raise on the first bad item.

    Enforces two contracts on the *incoming* items only (never the merged or
    pre-existing keys — a legacy file with a weird key must stay writable):

    * key names match `_SECRET_KEY_RE` (an env-var identifier) →
      `InvalidSecretKey`;
    * values contain no NUL and no C0 control character other than tab/newline/CR
      (multi-line secrets such as PEM keys are a common, env-deliverable pattern;
      only NUL is truly undeliverable, and the rest of the C0 set is rejected as
      log/terminal-injection hygiene) → `InvalidSecretValue`.

    Every error message names the offending KEY only, never a value fragment.
    """
    for k, v in values.items():
        key = str(k)
        if not _SECRET_KEY_RE.fullmatch(key):
            raise InvalidSecretKey(
                f"Invalid secret key name {key!r}: key names must match [A-Za-z_][A-Za-z0-9_]*."
            )
        value = str(v)
        if "\x00" in value or any(ord(ch) < 32 and ch not in "\t\n\r" for ch in value):
            raise InvalidSecretValue(
                f"Secret value for key {key!r} contains a NUL or non-printable control character."
            )


class SecretDecryptError(RuntimeError):
    """An existing secrets file (or the key file) cannot be decrypted/parsed.

    Raised on wrong key, tampered/corrupted ciphertext, an AAD mismatch (file
    renamed/copied), or a malformed envelope — never degraded to `{}`.
    Messages contain paths, service names and kids only (message hygiene).
    """


class SecretRotationInProgress(RuntimeError):  # noqa: N818 — public API name
    """Raised when `SecretManager.rotate_key` finds a staged key.

    A leftover `secrets.key.new` means a previous rotation crashed mid-way;
    it is resumed at the next daemon startup, and a second rotation is refused
    until then (routes map this to 409 `secret.rotation_in_progress`).
    """


def project_storage_name(project_id: str) -> str:
    """The `SecretManager` name of a project's variable scope (D-P40-8).

    Args:
        project_id: A `prj_` id from `core.project_identity.mint_project_id`.

    Returns:
        `_project-<project_id>`; the `.enc` stem and the AES-GCM AAD name.

    Raises:
        InvalidServiceName: When `project_id` is not a minted project id.
    """
    name = f"_project-{project_id}"
    if not _PROJECT_STEM_RE.fullmatch(name):
        raise InvalidServiceName(f"Invalid project id: {project_id!r}")
    return name


class SecretManager:
    """Per-service secret env store (write-only over the API, encrypted at rest)."""

    def __init__(self, secrets_dir: str | Path, key_path: str | Path | None = None) -> None:
        self._dir = Path(secrets_dir).expanduser()
        # Default key location: sibling of the ciphertext dir — i.e.
        # `<data_dir>/secrets.key` next to `<data_dir>/secrets/` (see module
        # docstring for why the key lives outside the dir it protects).
        self._key_path = (
            Path(key_path).expanduser() if key_path else self._dir.parent / "secrets.key"
        )
        # Reentrant on purpose: writes and rotation are serialized; `load()`
        # is lock-free *except* the legacy late re-encrypt branch, which
        # `set()`/`delete_key()` reach while already holding the lock.
        self._lock = threading.RLock()
        # Startup order matters: finish an interrupted rotation first (so the
        # key state is settled), then migrate any legacy plaintext files.
        self._resume_rotation()
        #: Service names whose plaintext files were encrypted at this boot —
        #: the server lifespan turns these into `secret.encrypt_migrate`
        #: audit rows (principal `system`).
        self.migrated_services: list[str] = self._migrate_plaintext()

    # --- paths ------------------------------------------------------------

    @property
    def key_path(self) -> Path:
        """The active key file path (for the startup log line)."""
        return self._key_path

    @property
    def _new_key_path(self) -> Path:
        """The staged key written by an in-flight rotation."""
        return self._key_path.with_name(self._key_path.name + ".new")

    def _check_name(self, service: str) -> str:
        # `fullmatch` on the internal stems: they become a filename, and `$`
        # alone would admit a trailing newline.
        if (
            service != SHARED_SCOPE
            and not _PROJECT_STEM_RE.fullmatch(service)
            and not _ENV_STEM_RE.fullmatch(service)
            and not _DNS_LABEL_RE.match(service)
        ):
            raise InvalidServiceName(f"Invalid service name: {service!r}")
        return service

    def _path(self, service: str) -> Path:
        return self._dir / f"{self._check_name(service)}.enc"

    def _legacy_path(self, service: str) -> Path:
        """Pre-P8 plaintext location, honored read-only until re-encrypted."""
        return self._dir / f"{self._check_name(service)}.json"

    def _ensure_dir(self) -> None:
        """Create the secrets dir 0700 (owner-only) if it does not yet exist."""
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # If it pre-existed with looser perms, tighten it.
        try:
            os.chmod(self._dir, 0o700)
        except OSError:
            pass

    # --- key custody --------------------------------------------------------

    @staticmethod
    def _kid(key: bytes) -> str:
        """Key fingerprint: first 8 hex of SHA-256(key) — safe to log."""
        return hashlib.sha256(key).hexdigest()[:8]

    def self_check(self) -> bool:
        """Confirm the active key can encrypt-then-decrypt a constant, in memory.

        Zero filesystem writes: it reads the existing key file (the `GET
        /doctor` caller checks `key_path.is_file()` first, so this never
        triggers key creation) and round-trips a fixed non-secret probe through
        AES-256-GCM. No real secret value ever touches this path. Returns
        `True` on a clean round-trip; propagates `SecretDecryptError`
        when the key file is unreadable/malformed (the caller maps that to a
        failing check without echoing key material).
        """
        key = self._read_key_file(self._key_path)
        probe = b"nerdit-doctor-self-check"
        aad = _AAD_PREFIX + b"_doctor"
        nonce = os.urandom(_NONCE_BYTES)
        cipher = AESGCM(key)
        return cipher.decrypt(nonce, cipher.encrypt(nonce, probe, aad), aad) == probe

    def has_ciphertexts(self) -> bool:
        """True when at least one encrypted secrets file exists at rest.

        Existence only: nothing is read, decrypted, or created, and a store
        directory that does not exist yet simply answers `False` (a fresh
        install never creates it — see `_ensure_dir`, whose only caller
        is the first write). A name ending in `.enc` is the repo-wide marker
        for encrypted material (snapshot, rotation, restore all glob it): the
        `<name>.enc.tmp-<pid>` staging residue does not match, and legacy
        pre-P8 plaintext `<name>.json` is deliberately excluded — it is
        recoverable without the key, so it is not material at risk.

        **Fail-loud when the store cannot be enumerated.** `Path.glob`
        swallows a `PermissionError` from `scandir` and yields nothing, so
        an unreadable store would read as an empty one — turning real data loss
        into the reassuring fresh-install answer. This walks `os.scandir`
        directly instead: only a genuinely absent store (`FileNotFoundError`)
        answers `False`; anything else that exists but cannot be listed —
        wrong perms, or a non-directory sitting at the path — answers `True`,
        because an unenumerable store can never be proven empty.

        Used by `GET /doctor` to tell the two absent-key states apart: no
        key + no ciphertexts is a healthy fresh install, no key + ciphertexts
        is unrecoverable data.
        """
        try:
            with os.scandir(self._dir) as entries:
                return any(entry.name.endswith(".enc") for entry in entries)
        except FileNotFoundError:
            return False
        except OSError:
            # Exists but unlistable (EACCES, ENOTDIR, EIO…): assume material.
            return True

    def _read_key_file(self, path: Path) -> bytes:
        """Read a key file (hex + newline) — errors never echo key material."""
        try:
            key = bytes.fromhex(path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            raise SecretDecryptError(f"Cannot read secrets key file {path}: {exc}") from None
        except ValueError:
            raise SecretDecryptError(
                f"Secrets key file {path} is malformed (expected {_KEY_BYTES * 2} hex chars)."
            ) from None
        if len(key) != _KEY_BYTES:
            raise SecretDecryptError(
                f"Secrets key file {path} has the wrong length "
                f"(expected {_KEY_BYTES * 2} hex chars)."
            )
        return key

    def _write_key_file(self, path: Path, key: bytes, *, atomic: bool = False) -> None:
        """Write a key file 0600 via O_EXCL (no world-readable window).

        `atomic=True` stages through a temp inode + `os.replace` so a crash
        mid-write can never leave a truncated key file (used for the staged
        rotation key). The active-key creation path keeps the bare O_EXCL
        write: its `FileExistsError` is how `_ensure_key` detects a
        lost generation race.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_name(path.name + ".tmp") if atomic else path
        if atomic:
            target.unlink(missing_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(key.hex() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if atomic:
            os.replace(target, path)
            self._fsync_dir(path.parent)

    def _ensure_key(self) -> bytes:
        """Load the active key, auto-generating it on first use."""
        if self._key_path.is_file():
            return self._read_key_file(self._key_path)
        key = os.urandom(_KEY_BYTES)
        try:
            self._write_key_file(self._key_path, key)
        except FileExistsError:
            # Lost a creation race (another process/thread got there first).
            return self._read_key_file(self._key_path)
        logger.info("Generated new secrets key at %s (kid %s)", self._key_path, self._kid(key))
        return key

    def _candidate_keys(self) -> list[tuple[bytes, str]]:
        """Active key plus the staged rotation key, as `(key, kid)` pairs.

        The two-key read fallback keeps every file readable during the
        rotation crash window (files split across old and new keys).
        """
        key = self._ensure_key()
        candidates = [(key, self._kid(key))]
        if self._new_key_path.is_file():
            try:
                staged = self._read_key_file(self._new_key_path)
            except SecretDecryptError as exc:
                # A truncated/corrupt staged key (a crash during staging) must
                # never wedge reads that only need the active key.
                logger.warning(
                    "Ignoring unreadable staged secrets key %s: %s", self._new_key_path, exc
                )
            else:
                candidates.append((staged, self._kid(staged)))
        return candidates

    # --- envelope (de)serialization ------------------------------------------

    @staticmethod
    def _parse_strict(text: str) -> dict[str, str]:
        """Parse a plaintext JSON secrets object; raise `ValueError` if not one."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return {str(k): str(v) for k, v in data.items()}

    @staticmethod
    def _serialize(values: dict[str, str]) -> str:
        return json.dumps(values, indent=2, sort_keys=True)

    def _encrypt(self, service: str, values: dict[str, str], key: bytes) -> str:
        """Encrypt *values* into a v1 envelope bound to *service* via the AAD."""
        nonce = os.urandom(_NONCE_BYTES)
        ct = AESGCM(key).encrypt(
            nonce,
            self._serialize(values).encode("utf-8"),
            _AAD_PREFIX + service.encode("ascii"),
        )
        envelope = {
            "nerdit_secrets": _ENVELOPE_VERSION,
            "alg": _ENVELOPE_ALG,
            "kid": self._kid(key),
            "nonce": b64encode(nonce).decode("ascii"),
            "ct": b64encode(ct).decode("ascii"),
        }
        return json.dumps(envelope, indent=2, sort_keys=True)

    def _read_envelope(self, path: Path) -> tuple[str, bytes, bytes]:
        """Parse + validate an `.enc` envelope → `(kid, nonce, ct)`.

        Message hygiene: every error carries the file name and a reason only —
        never nonce/ct bytes (`from None` drops upstream chains too).
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SecretDecryptError(f"Cannot read secrets file {path}: {exc}") from None
        except ValueError:
            raise SecretDecryptError(
                f"Secrets file {path.name} is not a valid envelope (bad JSON)."
            ) from None
        if not isinstance(data, dict) or data.get("nerdit_secrets") != _ENVELOPE_VERSION:
            raise SecretDecryptError(
                f"Secrets file {path.name} has a missing or unsupported envelope version."
            )
        if data.get("alg") != _ENVELOPE_ALG:
            raise SecretDecryptError(f"Secrets file {path.name} uses an unsupported algorithm.")
        kid = data.get("kid")
        if not isinstance(kid, str) or not kid:
            raise SecretDecryptError(f"Secrets file {path.name} is missing its key id.")
        try:
            nonce = b64decode(data["nonce"], validate=True)
            ct = b64decode(data["ct"], validate=True)
        except (KeyError, TypeError, ValueError):
            raise SecretDecryptError(
                f"Secrets file {path.name} has a malformed envelope (bad nonce/ct encoding)."
            ) from None
        if len(nonce) != _NONCE_BYTES:
            # AESGCM.decrypt raises a bare ValueError (not InvalidTag) on an
            # out-of-range nonce; validate here so tampering stays inside the
            # SecretDecryptError contract.
            raise SecretDecryptError(
                f"Secrets file {path.name} has a malformed envelope (bad nonce length)."
            )
        return kid, nonce, ct

    def _decrypt_file(self, path: Path, service: str) -> dict[str, str]:
        """Decrypt an `.enc` file with the active key (staged-key fallback)."""
        file_kid, nonce, ct = self._read_envelope(path)
        candidates = self._candidate_keys()
        matching = [key for key, kid in candidates if kid == file_kid]
        if not matching:
            known = ", ".join(kid for _, kid in candidates)
            raise SecretDecryptError(
                f"Secrets file {path.name} is encrypted with key {file_kid}, active key is "
                f"{known} — restore the matching secrets.key."
            )
        for key in matching:
            try:
                plaintext = AESGCM(key).decrypt(nonce, ct, _AAD_PREFIX + service.encode("ascii"))
                break
            except (InvalidTag, ValueError):
                # ValueError is belt-and-braces: _read_envelope already rejects
                # bad nonce lengths before they reach AESGCM.
                continue
        else:
            raise SecretDecryptError(
                f"Secrets file {path.name} failed to decrypt with key {file_kid} — the file "
                "was tampered with, corrupted, or renamed (AAD binds it to its service)."
            )
        try:
            return self._parse_strict(plaintext.decode("utf-8"))
        except (ValueError, TypeError):
            raise SecretDecryptError(
                f"Secrets file {path.name} decrypted to an invalid payload."
            ) from None

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        """fsync a directory so renames inside it survive a power loss."""
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:  # pragma: no cover - platforms without directory fds
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _atomic_write(self, path: Path, payload: str) -> None:
        """Write *payload* to *path* atomically (durably) with 0600 perms."""
        self._ensure_dir()
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        if tmp.exists():
            tmp.unlink()
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self._fsync_dir(path.parent)

    # --- migration (P8, boot-safe) ---------------------------------------------

    def _migrate_plaintext(self) -> list[str]:
        """Encrypt legacy plaintext `{service}.json` files in place.

        Boot-safe by construction: each file is wrapped in its own try/except —
        a poisoned file is logged at ERROR and left plaintext (the legacy read
        branch in `load` re-encrypts it on next touch), and startup
        continues. Plaintext-era `*.tmp-*` orphans are swept (they contain
        plaintext). Idempotent: a second run finds nothing to do. Returns the
        migrated service names for the server's audit rows.
        """
        if not self._dir.is_dir():
            return []
        # Sweep temp orphans first — pre-P8 ones hold plaintext secrets.
        for orphan in sorted(self._dir.glob("*.tmp-*")):
            try:
                orphan.unlink()
                logger.info("Swept stale secrets temp file %s", orphan)
            except OSError as exc:
                logger.error("Could not sweep secrets temp file %s: %s", orphan, exc)
        migrated: list[str] = []
        for legacy in sorted(self._dir.glob("*.json")):
            name = legacy.stem
            try:
                if name != SHARED_SCOPE and not _DNS_LABEL_RE.match(name):
                    raise ValueError("not a valid service name")
                target = self._dir / f"{name}.enc"
                if target.is_file():
                    # An .enc already exists (e.g. a restored backup dropped the
                    # old .json back) — never clobber newer ciphertext with it.
                    logger.warning(
                        "Skipping legacy plaintext %s: %s already exists", legacy, target.name
                    )
                    continue
                values = self._parse_strict(legacy.read_text(encoding="utf-8"))
                with self._lock:
                    self._atomic_write(target, self._encrypt(name, values, self._ensure_key()))
                    legacy.unlink(missing_ok=True)
            except Exception as exc:  # per-file tolerance: never block daemon boot
                logger.error("Secrets migration skipped %s: %s (file left plaintext)", legacy, exc)
                continue
            migrated.append(name)
        if migrated:
            logger.info(
                "Encrypted %d legacy plaintext secrets file(s): %s",
                len(migrated),
                ", ".join(migrated),
            )
        return migrated

    # --- key rotation ------------------------------------------------------

    def rotate_key(self) -> int:
        """Rotate the secrets key, re-encrypting every stored file. Returns count.

        Two-phase: (1) stage a new random key at `secrets.key.new` (0600);
        (2) re-encrypt each `.enc` file with it, atomically per file — reads
        stay correct throughout via the two-key fallback; (3) promote the
        staged key with `os.replace`. A crash at any point is resumed at the
        next daemon startup (`_resume_rotation`); a second rotation is
        refused (`SecretRotationInProgress`) while a staged key exists.
        A decrypt failure aborts the rotation with the staged key left in
        place — the old key is never discarded before every file re-encrypts.
        """
        with self._lock:
            if self._new_key_path.is_file():
                raise SecretRotationInProgress(
                    f"A key rotation is already in progress ({self._new_key_path} exists); "
                    "restart the daemon to resume it."
                )
            old_key = self._ensure_key()
            new_key = os.urandom(_KEY_BYTES)
            self._write_key_file(self._new_key_path, new_key, atomic=True)
            count = self._reencrypt_all(new_key)
            self._promote_staged_key()
            logger.info(
                "Rotated secrets key %s -> %s (%d file(s) re-encrypted)",
                self._kid(old_key),
                self._kid(new_key),
                count,
            )
            return count

    def snapshot_to(self, dest: Path) -> tuple[str, list[str]]:
        """Copy the active key file + every regular .enc ciphertext into *dest*.

        Runs under the rotation lock; refuses while a rotation is staged
        (split-key state). Returns (active kid, skipped basenames). Sync by
        design — callers on the event loop use asyncio.to_thread().
        """
        with self._lock:
            if self._new_key_path.is_file():
                # A staged rotation means some .enc may reference the staged
                # kid; refusing (never double-key-copying) reuses the shipped
                # 409 class and keeps the artifact single-key.
                raise SecretRotationInProgress(
                    "A key rotation is staged; complete it before snapshotting the secrets."
                )
            key = self._ensure_key()
            kid = self._kid(key)
            dest.mkdir(parents=True, exist_ok=True)
            os.chmod(dest, 0o700)
            shutil.copy2(self._key_path, dest / "secrets.key")
            os.chmod(dest / "secrets.key", 0o600)
            skipped: list[str] = []
            if self._dir.is_dir():
                store = dest / "store"
                for path in sorted(self._dir.glob("*.enc")):
                    # lstat-check first: a planted symlink (or any non-regular
                    # inode) in secrets/ must never exfiltrate an arbitrary
                    # daemon-readable file into an off-box artifact.
                    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                        skipped.append(path.name)
                        continue
                    store.mkdir(mode=0o700, parents=True, exist_ok=True)
                    shutil.copy2(path, store / path.name)
                    os.chmod(store / path.name, 0o600)
                # Legacy plaintext .json files are never copied (that would put
                # plaintext secrets in the tar) — reported skipped instead.
                for legacy in sorted(self._dir.glob("*.json")):
                    skipped.append(legacy.name)
            return kid, skipped

    def _promote_staged_key(self) -> None:
        """Promote `secrets.key.new` to the active key, durably.

        The secrets dir is fsynced *before* the promote so the re-encrypted
        files' renames can never be reordered after it by a power loss (the
        two directories may live in separately-journaled locations), and the
        key's parent dir *after* so the promote itself sticks.
        """
        if self._dir.is_dir():
            self._fsync_dir(self._dir)
        os.replace(self._new_key_path, self._key_path)
        self._fsync_dir(self._key_path.parent)

    def _staged_key_is_unreferenced(self) -> bool:
        """True when every `.enc` file is encrypted under the active key.

        Used to decide whether an *unreadable* staged key can be discarded: a
        key that no file references was never used to encrypt anything (a
        crash during staging). An unreadable envelope counts as a reference —
        unknown provenance keeps the staged key for the admin.
        """
        if not self._dir.is_dir():
            return True
        active_kid = self._kid(self._ensure_key())
        for path in sorted(self._dir.glob("*.enc")):
            try:
                file_kid, _, _ = self._read_envelope(path)
            except SecretDecryptError:
                return False
            if file_kid != active_kid:
                return False
        return True

    def _resume_rotation(self) -> None:
        """Finish an interrupted two-phase key rotation before serving.

        Failure-tolerant like migration: if the resume cannot complete (e.g. a
        poisoned file), the staged key is left in place — reads keep working
        via the two-key fallback and a new rotation stays refused — and the
        daemon boots anyway. An *unreadable* staged key that no file
        references is discarded instead (it can only be a crash during
        staging; leaving it would wedge every read and future rotation).
        """
        if not self._new_key_path.is_file():
            return
        try:
            with self._lock:
                try:
                    new_key = self._read_key_file(self._new_key_path)
                except SecretDecryptError as exc:
                    if self._staged_key_is_unreferenced():
                        self._new_key_path.unlink(missing_ok=True)
                        logger.warning(
                            "Discarded unreadable staged secrets key %s (%s): "
                            "no secrets file references it",
                            self._new_key_path,
                            exc,
                        )
                        return
                    raise
                if not self._key_path.is_file():
                    # No active key beside a staged one: nothing was ever
                    # encrypted under an old key — just promote the staged key.
                    self._promote_staged_key()
                    logger.warning("Promoted staged secrets key at %s", self._key_path)
                    return
                count = self._reencrypt_all(new_key)
                self._promote_staged_key()
            logger.warning(
                "Resumed interrupted secrets key rotation (%d file(s) re-encrypted)", count
            )
        except Exception as exc:  # boot-safe: leave the staged key for the admin
            logger.error(
                "Could not resume interrupted secrets key rotation: %s "
                "(staged key left at %s; reads keep working via the two-key fallback)",
                exc,
                self._new_key_path,
            )

    def _reencrypt_all(self, new_key: bytes) -> int:
        """Re-encrypt every `.enc` file with *new_key*; skip already-rotated ones."""
        if not self._dir.is_dir():
            return 0
        new_kid = self._kid(new_key)
        count = 0
        for path in sorted(self._dir.glob("*.enc")):
            service = path.stem
            file_kid, _, _ = self._read_envelope(path)
            if file_kid == new_kid:
                continue  # resume case: this file was rotated before the crash
            values = self._decrypt_file(path, service)
            self._atomic_write(path, self._encrypt(service, values, new_key))
            count += 1
        return count

    # --- public API (unchanged signatures) --------------------------------------

    def load(self, service: str) -> dict[str, str]:
        """Return the service's secrets as a `{KEY: value}` map (empty if none).

        Raises `SecretDecryptError` if an existing `.enc` file cannot
        be decrypted — never a silent `{}` (reserved for file-absent). A
        straggler plaintext `.json` (failed boot migration) is honored and
        re-encrypted on the spot (WARNING).
        """
        path = self._path(service)
        if path.is_file():
            return self._decrypt_file(path, service)
        legacy = self._legacy_path(service)
        if not legacy.is_file():
            return {}
        try:
            values = self._parse_strict(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            # Malformed legacy plaintext: keep the pre-P8 degrade-to-{} read
            # behavior, but never destroy the original by re-encrypting it.
            logger.error("Legacy plaintext secrets file %s is malformed; ignoring it", legacy)
            return {}
        with self._lock:
            # Re-check under the lock: a concurrent load may have migrated it.
            if not path.is_file() and legacy.is_file():
                logger.warning("Late-encrypting legacy plaintext secrets file %s", legacy)
                self._atomic_write(path, self._encrypt(service, values, self._ensure_key()))
                legacy.unlink(missing_ok=True)
        return values

    def list_keys(self, service: str) -> list[str]:
        """Return the service's secret key *names* only (never values), sorted."""
        return sorted(self.load(service).keys())

    def set(self, service: str, values: dict[str, str]) -> list[str]:
        """Merge *values* into the service's secrets; return the resulting names.

        Existing keys are overwritten, others preserved (a partial update).

        Validates the *incoming* items first (before the lock, before any read
        or write) so a bad key name or value leaves the stored file untouched.
        """
        validate_secret_items(values)
        with self._lock:
            merged = self.load(service)
            merged.update({str(k): str(v) for k, v in values.items()})
            path = self._path(service)
            self._atomic_write(path, self._encrypt(service, merged, self._ensure_key()))
            return sorted(merged.keys())

    def delete_key(self, service: str, key: str) -> bool:
        """Remove one key. Returns True if it existed, False otherwise."""
        with self._lock:
            current = self.load(service)
            if key not in current:
                return False
            current.pop(key)
            path = self._path(service)
            if current:
                self._atomic_write(path, self._encrypt(service, current, self._ensure_key()))
            else:
                path.unlink(missing_ok=True)
            return True

    def exists(self, service: str) -> bool:
        """Whether any secrets file (`.enc` or legacy `.json`) is stored for the name.

        A presence probe only: nothing is read or decrypted.
        """
        return self._path(service).is_file() or self._legacy_path(service).is_file()

    def delete(self, service: str) -> bool:
        """Remove the whole secrets file for a service. Returns True if it existed."""
        with self._lock:
            path = self._path(service)
            legacy = self._legacy_path(service)
            existed = path.exists() or legacy.exists()
            path.unlink(missing_ok=True)
            legacy.unlink(missing_ok=True)
            return existed
