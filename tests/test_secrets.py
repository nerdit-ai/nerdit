"""Tests for the SecretManager + launch-time secret injection (P4 / S6).

P8 adds encryption at rest: the plaintext-migration gate, crypto envelope
behavior (wrong key / AAD / malformed / message hygiene), and key rotation
(two-phase, crash-resume, refusal while in progress).
"""

from __future__ import annotations

import base64
import json
import os
import stat
from datetime import UTC, datetime

import pytest

from nerdit.config.settings import ServicesSettings
from nerdit.core.secrets import (
    SHARED_SCOPE,
    InvalidSecretKey,
    InvalidSecretValue,
    InvalidServiceName,
    SecretDecryptError,
    SecretManager,
    SecretRotationInProgress,
    project_storage_name,
)
from nerdit.core.services import ServiceController
from nerdit.db.models import Job, JobKind, JobStatus


def _mode(path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))[-3:]


def test_set_creates_0600_file_and_0700_dir(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"API_KEY": "abc"})
    secrets_dir = tmp_path / "secrets"
    assert _mode(secrets_dir) == "700"
    assert _mode(secrets_dir / "demo.enc") == "600"


@pytest.mark.parametrize("bad_key", ["", "A=B", "NL\nKEY", "1ABC", "has space"])
def test_set_rejects_invalid_key_name_and_writes_nothing(tmp_path, bad_key):
    """A bad key name raises InvalidSecretKey and leaves the .enc file absent
    (fix #3) — validation happens before any read/write."""
    mgr = SecretManager(tmp_path / "secrets")
    enc = tmp_path / "secrets" / "demo.enc"
    with pytest.raises(InvalidSecretKey) as ei:
        mgr.set("demo", {bad_key: "value"})
    # The message names the offending KEY only, never the value.
    assert "value" not in str(ei.value)
    assert not enc.exists()


def test_set_rejects_bad_key_leaves_existing_file_byte_identical(tmp_path):
    """A rejected write does not touch a pre-existing stored file (fix #3)."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"GOOD": "1"})
    enc = tmp_path / "secrets" / "demo.enc"
    before = enc.read_bytes()
    with pytest.raises(InvalidSecretKey):
        mgr.set("demo", {"BAD KEY": "2"})
    assert enc.read_bytes() == before


@pytest.mark.parametrize("good_key", ["HF_TOKEN", "_X", "api_key", "A1_b2"])
def test_set_accepts_valid_env_var_key_names(tmp_path, good_key):
    mgr = SecretManager(tmp_path / "secrets")
    assert mgr.set("demo", {good_key: "v"}) == [good_key]
    assert mgr.load("demo") == {good_key: "v"}


def test_set_rejects_nul_and_control_values_writes_nothing(tmp_path):
    """A NUL / non-printable control value raises InvalidSecretValue and writes
    nothing; the message carries the KEY name only, never the value (fix #15)."""
    mgr = SecretManager(tmp_path / "secrets")
    enc = tmp_path / "secrets" / "demo.enc"
    for bad_value in ["a\x00b", "esc\x1bseq", "bell\x07"]:
        with pytest.raises(InvalidSecretValue) as ei:
            mgr.set("demo", {"KEY": bad_value})
        assert bad_value not in str(ei.value)
        assert "KEY" in str(ei.value)
        assert not enc.exists()


def test_set_accepts_multiline_and_tab_values_round_trip(tmp_path):
    """Multi-line (PEM-shaped) and tab-bearing values are allowed and round-trip
    byte-for-byte via load() (fix #15 deliberate narrowing)."""
    mgr = SecretManager(tmp_path / "secrets")
    pem = "-----BEGIN KEY-----\nline1\nline2\n-----END KEY-----\n"
    mgr.set("demo", {"PEM": pem, "TABBED": "tab\there", "CR": "a\rb"})
    loaded = mgr.load("demo")
    assert loaded["PEM"] == pem
    assert loaded["TABBED"] == "tab\there"
    assert loaded["CR"] == "a\rb"


def test_preexisting_weird_key_does_not_block_later_valid_set(tmp_path):
    """A legacy stored file with a weird key must stay writable: validation is on
    the *incoming* items only, never the merged/pre-existing keys (fix #3)."""
    mgr = SecretManager(tmp_path / "secrets")
    # Simulate a legacy file carrying an invalid key by writing it directly
    # through the manager's low-level encrypt/write path (bypassing set()).
    path = tmp_path / "secrets" / "demo.enc"
    mgr._atomic_write(path, mgr._encrypt("demo", {"weird key": "old"}, mgr._ensure_key()))
    # A later valid set() must succeed and merge, not choke on the stored key.
    result = mgr.set("demo", {"NEW": "v"})
    assert "NEW" in result
    assert "weird key" in result


def test_self_check_round_trips_without_writing(tmp_path):
    """`self_check` (P13b WP6 /doctor) round-trips a constant with the active key
    and writes NOTHING beyond the key file already created by set()."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"API_KEY": "abc"})  # materializes the key file
    before = sorted(p.name for p in (tmp_path / "secrets").iterdir())
    assert mgr.self_check() is True
    after = sorted(p.name for p in (tmp_path / "secrets").iterdir())
    assert before == after  # zero new files


def test_self_check_raises_on_missing_key(tmp_path):
    """With no key file, self_check surfaces a SecretDecryptError (the /doctor
    caller checks key_path.is_file() first, so it never triggers key creation)."""
    mgr = SecretManager(tmp_path / "secrets")
    assert not mgr.key_path.is_file()
    with pytest.raises(SecretDecryptError):
        mgr.self_check()
    assert not mgr.key_path.is_file()  # self_check never created the key


def test_load_and_list_keys(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1", "B": "2"})
    assert mgr.load("demo") == {"A": "1", "B": "2"}
    # list_keys returns names only (values never exposed).
    assert mgr.list_keys("demo") == ["A", "B"]


def test_set_merges_partial(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1", "B": "2"})
    mgr.set("demo", {"B": "changed", "C": "3"})
    assert mgr.load("demo") == {"A": "1", "B": "changed", "C": "3"}


def test_delete_key_and_delete_file(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1", "B": "2"})
    assert mgr.delete_key("demo", "A") is True
    assert mgr.load("demo") == {"B": "2"}
    assert mgr.delete_key("demo", "missing") is False
    # Removing the last key deletes the file.
    assert mgr.delete_key("demo", "B") is True
    assert not (tmp_path / "secrets" / "demo.enc").exists()


def test_delete_whole_service(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    assert mgr.delete("demo") is True
    assert mgr.delete("demo") is False


def test_values_with_special_chars_roundtrip(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"URL": "https://api.example.com/v1?x=1"})
    assert mgr.load("demo")["URL"] == "https://api.example.com/v1?x=1"


def test_multiline_and_whitespace_values_roundtrip(tmp_path):
    # A newline-bearing PEM, a value with leading/trailing whitespace, and an
    # embedded '=' must all survive read-back byte-for-byte (JSON, not KEY=value).
    mgr = SecretManager(tmp_path / "secrets")
    pem = "-----BEGIN PRIVATE KEY-----\nline2\nline3\n-----END PRIVATE KEY-----"
    values = {
        "PEM": pem,
        "PADDED": "  spaced value  ",
        "EQUALS": "a=b=c",
        "UNICODE": "clé-privée-éà",
    }
    mgr.set("demo", values)
    assert mgr.load("demo") == values


@pytest.mark.parametrize("bad", ["Bad_Name", "../etc", "-x", "UP", ""])
def test_invalid_service_name_rejected(tmp_path, bad):
    mgr = SecretManager(tmp_path / "secrets")
    with pytest.raises(InvalidServiceName):
        mgr.set(bad, {"A": "1"})


# --- P8: encryption at rest ----------------------------------------------------

# Byte-for-byte migration fixture: multiline PEM, padded/unicode/'=' values
# (mirrors test_multiline_and_whitespace_values_roundtrip).
_FIXTURE_VALUES = {
    "PEM": "-----BEGIN PRIVATE KEY-----\nline2\nline3\n-----END PRIVATE KEY-----",
    "PADDED": "  spaced value  ",
    "EQUALS": "a=b=c",
    "UNICODE": "clé-privée-éà",
}


def _write_plaintext(secrets_dir, service: str, values: dict[str, str]) -> None:
    """Drop a pre-P8 plaintext ``{service}.json`` (0600, dir 0700)."""
    secrets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = secrets_dir / f"{service}.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(values))


def test_migration_encrypts_plaintext_byte_for_byte(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_plaintext(secrets_dir, "demo", _FIXTURE_VALUES)
    mgr = SecretManager(secrets_dir)
    assert mgr.migrated_services == ["demo"]
    assert not (secrets_dir / "demo.json").exists()
    enc = secrets_dir / "demo.enc"
    assert _mode(enc) == "600"
    # On-disk file is an envelope, not the plaintext.
    raw = enc.read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in raw
    assert '"nerdit_secrets": 1' in raw
    # Values round-trip byte-for-byte through the migration.
    assert mgr.load("demo") == _FIXTURE_VALUES


def test_migration_is_idempotent(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_plaintext(secrets_dir, "demo", {"A": "1"})
    SecretManager(secrets_dir)
    # A second construction finds nothing left to migrate and stays green.
    again = SecretManager(secrets_dir)
    assert again.migrated_services == []
    assert again.load("demo") == {"A": "1"}


def test_migration_tolerates_poisoned_file(tmp_path):
    secrets_dir = tmp_path / "secrets"
    _write_plaintext(secrets_dir, "good", {"A": "1"})
    secrets_dir.joinpath("poisoned.json").write_text("{not json", encoding="utf-8")
    # Construction (= daemon boot) must not raise; the good file migrates.
    mgr = SecretManager(secrets_dir)
    assert mgr.migrated_services == ["good"]
    assert mgr.load("good") == {"A": "1"}
    # The poisoned file is left plaintext and reads degrade to {} (pre-P8 rule).
    assert (secrets_dir / "poisoned.json").exists()
    assert mgr.load("poisoned") == {}


def test_migration_sweeps_plaintext_tmp_orphans(tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(mode=0o700, parents=True)
    orphan = secrets_dir / "demo.json.tmp-12345"
    orphan.write_text('{"A": "leaked"}', encoding="utf-8")
    SecretManager(secrets_dir)
    assert not orphan.exists()


def test_migration_never_clobbers_existing_enc(tmp_path):
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("demo", {"A": "current"})
    # A restored backup drops the old plaintext back beside the ciphertext.
    (secrets_dir / "demo.json").write_text('{"A": "stale"}', encoding="utf-8")
    again = SecretManager(secrets_dir)
    assert again.migrated_services == []
    assert again.load("demo") == {"A": "current"}


def test_late_reencrypt_on_load(tmp_path):
    # A straggler plaintext file (failed boot migration) is honored and
    # re-encrypted on next read.
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    _write_plaintext(secrets_dir, "late", {"A": "1"})
    assert mgr.load("late") == {"A": "1"}
    assert not (secrets_dir / "late.json").exists()
    assert (secrets_dir / "late.enc").is_file()
    assert mgr.load("late") == {"A": "1"}


def test_key_file_generated_0600_outside_secrets_dir(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    key_path = tmp_path / "secrets.key"  # sibling of secrets/, not inside it
    assert mgr.key_path == key_path
    assert _mode(key_path) == "600"
    text = key_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(bytes.fromhex(text.strip())) == 32


def test_explicit_key_path_override(tmp_path):
    key_path = tmp_path / "elsewhere" / "custom.key"
    mgr = SecretManager(tmp_path / "secrets", key_path=key_path)
    mgr.set("demo", {"A": "1"})
    assert key_path.is_file()
    assert mgr.load("demo") == {"A": "1"}


def test_wrong_key_raises_with_kids_and_hygiene(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "super-secret-value"})
    envelope = json.loads((tmp_path / "secrets" / "demo.enc").read_text(encoding="utf-8"))
    # Replace the key: the file's kid no longer matches any known key.
    (tmp_path / "secrets.key").write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
    with pytest.raises(SecretDecryptError) as excinfo:
        mgr.load("demo")
    message = str(excinfo.value)
    assert envelope["kid"] in message  # says which key the file needs
    # Message hygiene: no plaintext, no nonce/ct/key material.
    assert "super-secret-value" not in message
    assert envelope["nonce"] not in message
    assert envelope["ct"] not in message


def test_tampered_ciphertext_raises_never_empty(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    path = tmp_path / "secrets" / "demo.enc"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    # Flip one ciphertext byte.
    ct = bytearray(base64.b64decode(envelope["ct"]))
    ct[0] ^= 0xFF
    envelope["ct"] = base64.b64encode(bytes(ct)).decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(SecretDecryptError):
        mgr.load("demo")


def test_aad_binds_file_to_service(tmp_path):
    # Copying a.enc over b.enc must fail: the AAD includes the storage name.
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("aaa", {"A": "1"})
    mgr.set("bbb", {"B": "2"})
    a = (tmp_path / "secrets" / "aaa.enc").read_text(encoding="utf-8")
    (tmp_path / "secrets" / "bbb.enc").write_text(a, encoding="utf-8")
    with pytest.raises(SecretDecryptError):
        mgr.load("bbb")


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "[1, 2]",
        '{"nerdit_secrets": 99, "alg": "aes-256-gcm", "kid": "x", "nonce": "AA==", "ct": "AA=="}',
        '{"nerdit_secrets": 1, "alg": "rot13", "kid": "x", "nonce": "AA==", "ct": "AA=="}',
        '{"nerdit_secrets": 1, "alg": "aes-256-gcm", "kid": "x", "nonce": "!!!", "ct": "AA=="}',
        '{"nerdit_secrets": 1, "alg": "aes-256-gcm", "nonce": "AA==", "ct": "AA=="}',
    ],
)
def test_malformed_envelope_raises_never_empty(tmp_path, payload):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    (tmp_path / "secrets" / "demo.enc").write_text(payload, encoding="utf-8")
    with pytest.raises(SecretDecryptError):
        mgr.load("demo")


def test_absent_file_loads_empty(tmp_path):
    # {} is reserved for file-absent (decrypt failures raise instead).
    mgr = SecretManager(tmp_path / "secrets")
    assert mgr.load("missing") == {}


def test_shared_scope_storage_name_allowed(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "k"})
    assert (tmp_path / "secrets" / "_shared.enc").is_file()
    assert mgr.list_keys(SHARED_SCOPE) == ["OPENAI_KEY"]


# --- P8: key rotation -----------------------------------------------------------


def test_rotate_key_reencrypts_everything(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("aaa", _FIXTURE_VALUES)
    mgr.set("bbb", {"B": "2"})
    old_key = (tmp_path / "secrets.key").read_text(encoding="utf-8")
    assert mgr.rotate_key() == 2
    assert (tmp_path / "secrets.key").read_text(encoding="utf-8") != old_key
    assert not (tmp_path / "secrets.key.new").exists()
    assert mgr.load("aaa") == _FIXTURE_VALUES
    assert mgr.load("bbb") == {"B": "2"}


def test_rotation_crash_window_stays_readable_and_resumes(tmp_path):
    # Simulate a crash mid-rotation: .key.new staged, files split across keys.
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("aaa", {"A": "1"})
    mgr.set("bbb", {"B": "2"})
    new_key = os.urandom(32)
    (tmp_path / "secrets.key.new").write_text(new_key.hex() + "\n", encoding="utf-8")
    # One file already re-encrypted with the staged key, the other not yet.
    mgr._atomic_write(secrets_dir / "aaa.enc", mgr._encrypt("aaa", {"A": "1"}, new_key))
    # Crash window: both files still load (two-key fallback).
    assert mgr.load("aaa") == {"A": "1"}
    assert mgr.load("bbb") == {"B": "2"}
    # Startup resume: finishes re-encrypting and promotes the staged key.
    resumed = SecretManager(secrets_dir)
    assert not (tmp_path / "secrets.key.new").exists()
    assert bytes.fromhex((tmp_path / "secrets.key").read_text(encoding="utf-8").strip()) == new_key
    assert resumed.load("aaa") == {"A": "1"}
    assert resumed.load("bbb") == {"B": "2"}


def test_rotate_key_refuses_while_in_progress(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    (tmp_path / "secrets.key.new").write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
    with pytest.raises(SecretRotationInProgress):
        mgr.rotate_key()


def test_rotate_key_with_no_files(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    old_key = mgr._ensure_key()
    assert mgr.rotate_key() == 0
    assert mgr._ensure_key() != old_key


def test_malformed_nonce_length_raises_decrypt_error(tmp_path):
    # A tampered nonce outside AESGCM's 8..128-byte range raises a bare
    # ValueError from cryptography; the envelope validation must convert it to
    # SecretDecryptError before it ever reaches the cipher.
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    path = tmp_path / "secrets" / "demo.enc"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["nonce"] = base64.b64encode(b"shrt").decode("ascii")  # 4 bytes
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(SecretDecryptError):
        mgr.load("demo")


def test_corrupt_staged_key_does_not_wedge_reads(tmp_path):
    # A truncated/corrupt .key.new (crash during staging) must not block loads
    # that only need the active key.
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("demo", {"A": "1"})
    (tmp_path / "secrets.key.new").write_text("deadbeef\n", encoding="utf-8")
    assert mgr.load("demo") == {"A": "1"}


def test_resume_discards_unreadable_staged_key_when_unreferenced(tmp_path):
    # Startup resume: an unreadable staged key that no .enc file references was
    # never used to encrypt anything — discard it so reads and future rotations
    # are not wedged.
    secrets_dir = tmp_path / "secrets"
    SecretManager(secrets_dir).set("demo", {"A": "1"})
    (tmp_path / "secrets.key.new").write_text("not-even-hex\n", encoding="utf-8")
    resumed = SecretManager(secrets_dir)
    assert not (tmp_path / "secrets.key.new").exists()
    assert resumed.load("demo") == {"A": "1"}
    assert resumed.rotate_key() == 1  # rotation is unblocked again


def test_resume_keeps_unreadable_staged_key_when_provenance_unknown(tmp_path):
    # If any file is NOT under the active key, the unreadable staged key might
    # be the only thing that ever decrypted it — leave it for the admin.
    secrets_dir = tmp_path / "secrets"
    SecretManager(secrets_dir).set("demo", {"A": "1"})
    # Swap the active key: demo.enc's kid no longer matches it.
    (tmp_path / "secrets.key").write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
    (tmp_path / "secrets.key.new").write_text("not-even-hex\n", encoding="utf-8")
    SecretManager(secrets_dir)  # boot-safe: constructs anyway
    assert (tmp_path / "secrets.key.new").is_file()


# --- P14c: snapshot_to (backup capture) -----------------------------------------


def test_snapshot_to_copies_key_and_enc_faithfully(tmp_path):
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("aaa", _FIXTURE_VALUES)
    mgr.set("bbb", {"B": "2"})
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "k"})

    dest = tmp_path / "staging" / "secrets"
    kid, skipped = mgr.snapshot_to(dest)

    assert kid == SecretManager._kid(mgr._ensure_key())
    assert skipped == []
    # Key copied byte-for-byte, 0600 in a 0700 dir.
    assert _mode(dest) == "700"
    assert _mode(dest / "secrets.key") == "600"
    assert (dest / "secrets.key").read_bytes() == (tmp_path / "secrets.key").read_bytes()
    # Every .enc copied faithfully (incl. _shared.enc), 0600.
    store = dest / "store"
    assert sorted(p.name for p in store.iterdir()) == ["_shared.enc", "aaa.enc", "bbb.enc"]
    for name in ("aaa.enc", "bbb.enc", "_shared.enc"):
        assert _mode(store / name) == "600"
        assert (store / name).read_bytes() == (secrets_dir / name).read_bytes()
    # And the snapshot decrypts under the copied key (treat store/ as a fresh
    # manager's secrets dir pointed at the copied key).
    reloaded = SecretManager(secrets_dir=store, key_path=dest / "secrets.key")
    assert reloaded.load("aaa") == _FIXTURE_VALUES
    assert reloaded.load("bbb") == {"B": "2"}


def test_snapshot_to_refuses_staged_rotation(tmp_path):
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("demo", {"A": "1"})
    (tmp_path / "secrets.key.new").write_text(os.urandom(32).hex() + "\n", encoding="utf-8")
    with pytest.raises(SecretRotationInProgress) as exc:
        mgr.snapshot_to(tmp_path / "staging" / "secrets")
    # The refusal message must not embed the absolute staged-key path.
    assert "/" not in str(exc.value)


def test_snapshot_to_never_created_dir_still_returns_kid(tmp_path):
    # A never-used install: no secrets/ dir yet, but snapshot must still emit a
    # valid key + kid (snapshot_to generates the key on first use).
    mgr = SecretManager(tmp_path / "secrets")
    assert not (tmp_path / "secrets").exists()
    dest = tmp_path / "staging" / "secrets"
    kid, skipped = mgr.snapshot_to(dest)
    assert kid and skipped == []
    assert (dest / "secrets.key").is_file()
    assert not (dest / "store").exists()  # nothing to copy


def test_snapshot_to_skips_legacy_json(tmp_path):
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("demo", {"A": "1"})
    # Drop a plaintext straggler beside the .enc (a supported steady state).
    (secrets_dir / "legacy.json").write_text('{"K": "v"}', encoding="utf-8")

    dest = tmp_path / "staging" / "secrets"
    _kid, skipped = mgr.snapshot_to(dest)

    assert "legacy.json" in skipped
    # Plaintext never enters the snapshot.
    assert not (dest / "store" / "legacy.json").exists()
    assert sorted(p.name for p in (dest / "store").iterdir()) == ["demo.enc"]


def test_snapshot_to_skips_planted_symlink_enc(tmp_path):
    # A planted symlink .enc in secrets/ must never exfiltrate its target into
    # the off-box artifact (capture-side M2).
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    mgr.set("demo", {"A": "1"})
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    (secrets_dir / "evil.enc").symlink_to(outside)

    dest = tmp_path / "staging" / "secrets"
    _kid, skipped = mgr.snapshot_to(dest)

    assert "evil.enc" in skipped
    store = dest / "store"
    assert sorted(p.name for p in store.iterdir()) == ["demo.enc"]
    assert not (store / "evil.enc").exists()


# --- launch-time injection ----------------------------------------------------


class _FakeRuntime:
    def __init__(self) -> None:
        self.live: dict[str, datetime] = {}
        self.run_configs: list = []
        self.counter = 0

    async def run(self, config):
        self.counter += 1
        cid = f"c{self.counter}"
        self.live[cid] = datetime.now(UTC)
        self.run_configs.append(config)
        return cid

    async def image_exists(self, image):
        return True

    async def list_managed_containers(self):
        return list(self.live.items())

    async def stop(self, cid, timeout=10):
        self.live.pop(cid, None)

    async def kill(self, cid):
        self.live.pop(cid, None)

    async def remove(self, cid, force=False):
        self.live.pop(cid, None)

    async def status(self, cid):
        return "running" if cid in self.live else None

    async def inspect_state(self, cid):
        return None

    async def logs(self, cid, follow=False, tail=None, max_bytes=None, since=None):
        return
        yield  # pragma: no cover


async def test_secrets_injected_and_override_config_env(queries, tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("app", {"API_KEY": "secret-value", "SHARED": "from-secret"})
    runtime = _FakeRuntime()
    controller = ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        secrets=mgr,
    )
    # Config env includes SHARED; the secret must override it.
    await queries.create_job(
        Job(
            name="app",
            kind=JobKind.service,
            service_name="app",
            gpu_count=0,
            status=JobStatus.building,
            desired_state="running",
            restart_policy="always",
            config=json.dumps({"image": "demo:1", "port": 8000, "env": {"SHARED": "from-config"}}),
        )
    )
    await controller.reconcile()
    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    env = runtime.run_configs[-1].env
    assert env["API_KEY"] == "secret-value"
    assert env["SHARED"] == "from-secret"  # secret overrides config env


# --- has_ciphertexts: the doctor's "is there material at risk?" probe ---------


def test_has_ciphertexts_false_when_the_store_dir_does_not_exist(tmp_path):
    """A fresh install has no secrets dir at all — the probe tolerates that."""
    mgr = SecretManager(tmp_path / "secrets")
    assert not (tmp_path / "secrets").exists()
    assert mgr.has_ciphertexts() is False
    # Probing created neither the dir nor the key.
    assert not (tmp_path / "secrets").exists()
    assert not mgr.key_path.exists()


def test_has_ciphertexts_true_after_the_first_write(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    assert mgr.has_ciphertexts() is False
    mgr.set("demo", {"API_KEY": "abc"})
    assert mgr.has_ciphertexts() is True


def test_has_ciphertexts_true_for_the_shared_scope(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"HF_TOKEN": "abc"})
    assert (tmp_path / "secrets" / f"{SHARED_SCOPE}.enc").is_file()
    assert mgr.has_ciphertexts() is True


def test_has_ciphertexts_ignores_temp_and_legacy_and_key_files(tmp_path):
    """Only ``*.enc`` counts: staging residue, legacy plaintext and key files do not.

    ``<name>.enc.tmp-<pid>`` is what an interrupted write leaves behind and
    ``<name>.json`` is pre-P8 plaintext (recoverable without the key), so
    neither is material that a missing key would destroy.
    """
    store = tmp_path / "secrets"
    store.mkdir()
    mgr = SecretManager(store)
    (store / "demo.enc.tmp-4242").write_text("{}", encoding="utf-8")
    (store / "demo.json").write_text('{"API_KEY": "abc"}', encoding="utf-8")
    (store / "secrets.key").write_text("00" * 32, encoding="utf-8")
    (store / "secrets.key.new").write_text("11" * 32, encoding="utf-8")
    assert mgr.has_ciphertexts() is False


def test_has_ciphertexts_honours_the_key_file_override(tmp_path):
    """The ``[security].secrets_key_file`` override moves the key, never the store."""
    store = tmp_path / "secrets"
    key = tmp_path / "elsewhere" / "custom.key"
    key.parent.mkdir()
    mgr = SecretManager(store, key_path=key)
    assert mgr.key_path == key
    assert mgr.has_ciphertexts() is False
    mgr.set("demo", {"API_KEY": "abc"})
    assert (store / "demo.enc").is_file()
    assert mgr.has_ciphertexts() is True


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason="needs POSIX mode bits and a non-root euid (root ignores 0o000)",
)
def test_has_ciphertexts_true_when_the_store_cannot_be_listed(tmp_path):
    """An unreadable store is never reported as an empty one.

    ``Path.glob`` swallows the ``PermissionError`` ``scandir`` raises and yields
    nothing, so globbing would answer ``False`` here — and ``GET /doctor`` would
    turn real, unrecoverable data loss (ciphertexts at rest, key gone) into the
    reassuring fresh-install ``skipped``. The probe walks ``os.scandir`` itself
    so the error is visible, and answers conservatively: it exists, it cannot be
    proven empty, so it counts as material at risk.
    """
    store = tmp_path / "secrets"
    mgr = SecretManager(store)
    mgr.set("demo", {"API_KEY": "abc"})
    mgr.key_path.unlink()  # the exact restore-gone-wrong state
    store.chmod(0o000)
    try:
        assert not any(store.glob("*.enc"))  # what the naive probe would see
        assert mgr.has_ciphertexts() is True
    finally:
        store.chmod(0o700)


def test_has_ciphertexts_true_when_the_store_path_is_not_a_directory(tmp_path):
    """A regular file sitting where the store should be is broken, not empty."""
    store = tmp_path / "secrets"
    store.write_text("not a directory", encoding="utf-8")
    assert SecretManager(store).has_ciphertexts() is True


def test_exists_probes_enc_and_legacy_json_without_reading(tmp_path):
    """P39 orphan probe: presence of either file shape, never a decrypt."""
    secrets_dir = tmp_path / "secrets"
    mgr = SecretManager(secrets_dir)
    assert mgr.exists("demo") is False
    mgr.set("demo", {"A": "1"})
    assert mgr.exists("demo") is True
    assert mgr.delete("demo") is True
    assert mgr.exists("demo") is False
    (secrets_dir / "legacy.json").write_text("{not json", encoding="utf-8")
    assert mgr.exists("legacy") is True  # a malformed legacy file still counts
    with pytest.raises(InvalidServiceName):
        mgr.exists("Bad_Name")


# --- P40c: project-scope storage names (D-P40-8) --------------------------------

_PRJ = "prj_abcdefghij234567"


def test_project_storage_name_is_the_one_composer():
    assert project_storage_name(_PRJ) == f"_project-{_PRJ}"
    for bad in ("asso", "prj_short", "prj_ABCDEFGHIJ234567", f"{_PRJ}/../x", f"{_PRJ}\n"):
        with pytest.raises(InvalidServiceName):
            project_storage_name(bad)


def test_check_name_admits_project_and_reserved_env_stems_only(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(f"_project-{_PRJ}", {"A": "1"})
    mgr.set(f"_env-{_PRJ}-staging", {"A": "1"})  # reserved for phase 3, admitted
    assert (tmp_path / "secrets" / f"_project-{_PRJ}.enc").is_file()
    for bad in (
        "_other",
        "_project-asso",
        f"_project-{_PRJ}x",
        f"_project-{_PRJ}\n",
        f"_env-{_PRJ}",
        f"_env-{_PRJ}-a--b",
        f"_env-{_PRJ}-../x",
        "_shared2",
    ):
        with pytest.raises(InvalidServiceName):
            mgr.set(bad, {"A": "1"})


def test_rotation_reencrypts_a_project_file_under_its_stem_aad(tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    name = project_storage_name(_PRJ)
    mgr.set(name, {"TOKEN": "s3cret"})
    mgr.set("asso", {"B": "2"})
    path = tmp_path / "secrets" / f"{name}.enc"
    before = path.read_text(encoding="utf-8")
    assert mgr.rotate_key() == 2
    assert path.read_text(encoding="utf-8") != before
    assert mgr.load(name) == {"TOKEN": "s3cret"}
    # The AAD is the stem: the same ciphertext under another project's name must not open.
    other = tmp_path / "secrets" / "_project-prj_zzzzzzzzzzzzzzzz.enc"
    other.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SecretDecryptError):
        mgr.load(other.stem)


def test_trailing_newline_label_is_rejected(tmp_path):
    """S7: `$` admits a trailing newline under `.match`; the gate uses `fullmatch`."""
    sm = SecretManager(tmp_path / "secrets")
    with pytest.raises(InvalidServiceName):
        sm.set("foo\n", {"K": "v"})
    assert not list((tmp_path / "secrets").glob("foo*"))
