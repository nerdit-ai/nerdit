"""Pin Ed25519 cloud compatibility and private-key custody with frozen vectors.

Keep private keys out of audit, logs, doctor and MCP output. Independently vendored
helpers must match cloud commit d53507081821d62e99991e2ce9f675e80cdca93e.
Vectors were generated once from the cloud implementation on 2026-08-14 and
verified in both directions; never regenerate them or import cloud code in CI.

Assert exact verifier, fingerprint, canonical ten-field proof bytes and signature.
Ed25519 signatures are deterministic. Vendored-fixture drift checks separately
detect changes to the frozen cloud contract.
"""

from __future__ import annotations

import base64
import os
import re
import stat
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nerdit.core.link.identity import (
    LinkIdentityError,
    _b64decode,
    _b64encode,
    _generate_key_file,
    credential_fingerprint,
    load_or_create_identity,
    proof_message,
    public_reference,
    resolve_key_file,
    sign_node_proof,
    valid_public_reference,
)
from tests.link_fake_relay import verify_node_proof

#: Pin the vendored helpers were derived from; asserted by the conformance suite.
PINNED_SHA = "d53507081821d62e99991e2ce9f675e80cdca93e"

# --- golden vectors (generated once at design time; do NOT regenerate) --------

#: raw = ``bytes(range(32, 64))`` — test-only, deliberately distinct from the
#: cloud's ``DEV_NODE_PRIVATE_KEY_B64`` (0x01..0x20), which is NOT vendored.
GOLDEN_PRIVATE_KEY_B64 = "ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"

#: = ``public_reference(GOLDEN_PRIVATE_KEY_B64)``
GOLDEN_VERIFIER = "ed25519:Kay64UG8yvCyLhqU000LxzYeUm0L_hLIl5S8kyKWbdc"

#: = ``credential_fingerprint(GOLDEN_VERIFIER)``
GOLDEN_FINGERPRINT = "6fcc38d43b9b0e31d9ca1ece3d58c034373069efb84e69033dda2d36ee097562"

#: = ``proof_message(**GOLDEN_PROOF_INPUTS)`` — compact, sorted-key JSON.
GOLDEN_PROOF_MESSAGE = (
    b'{"capability_expires_at":"2026-01-02T03:04:05+00:00","capability_role":"submitter",'
    b'"capability_token":"golden-capability-token","challenge":"golden-challenge-0001",'
    b'"daemon_version":"0.0.0-test","node_id":"node-golden-0001","node_name":"golden-node",'
    b'"protocol":"node-link/v1","relay_id":"relay-golden","uptime_s":1234}'
)

#: = ``sign_node_proof(GOLDEN_PRIVATE_KEY_B64, GOLDEN_PROOF_MESSAGE)``.
GOLDEN_SIGNATURE_B64 = (
    "TGcZUQ5VwjQh1Gg2-L0mBeieJOaaj1BGe1eC1n1LYZdQbpEFX2PpBEVUVtOcCALiFgNELmbYtt9IH23cwggpDw"
)

#: The exact ten keyword inputs the golden message was produced from.
GOLDEN_PROOF_INPUTS = {
    "challenge": "golden-challenge-0001",
    "relay_id": "relay-golden",
    "protocol": "node-link/v1",
    "node_id": "node-golden-0001",
    "node_name": "golden-node",
    "daemon_version": "0.0.0-test",
    "uptime_s": 1234,
    "capability_token": "golden-capability-token",
    "capability_expires_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    "capability_role": "submitter",
}

#: The cloud's dev public reference — a *valid* but unrelated verifier, so a
#: failed verification proves the signature is bound to the key, not malformed.
CLOUD_DEV_VERIFIER = "ed25519:ebVWLo_mVPlAeLES6KmLp5AfhTrmlb7X4OORC60ElmQ"

#: 43 unpadded base64url characters is the only shape a raw Ed25519 key takes.
KEY_LINE = re.compile(r"^[A-Za-z0-9_-]{43}$")


# --- vendored helpers: byte-compatibility with the cloud ----------------------


def test_b64encode_unpadded_roundtrip():
    encoded = _b64encode(bytes(range(32, 64)))
    assert encoded == GOLDEN_PRIVATE_KEY_B64
    # The canonical wire form is UNPADDED: a '=' would change every string the
    # relay stores and compares.
    assert "=" not in encoded
    assert _b64decode(encoded) == bytes(range(32, 64))


def test_public_reference_matches_cloud_golden():
    assert public_reference(GOLDEN_PRIVATE_KEY_B64) == GOLDEN_VERIFIER


def test_fingerprint_matches_cloud_golden():
    assert credential_fingerprint(GOLDEN_VERIFIER) == GOLDEN_FINGERPRINT


def test_valid_public_reference():
    assert valid_public_reference(GOLDEN_VERIFIER) is True

    raw = _b64decode(GOLDEN_VERIFIER.removeprefix("ed25519:"))
    padded = base64.urlsafe_b64encode(raw).decode("ascii")
    assert padded.endswith("=")  # guard: the padded form really differs

    for bad in (
        GOLDEN_VERIFIER.removeprefix("ed25519:"),  # missing prefix
        f"ed25519:{padded}",  # padded (non-canonical) encoding
        f"ed25519:{_b64encode(raw[:31])}",  # 31-byte key
        f"{GOLDEN_VERIFIER}A",  # non-canonical: one extra b64 char
        "",  # empty string
        "ed25519:",  # prefix, no key
    ):
        assert valid_public_reference(bad) is False, bad


def test_proof_message_bytes_exact():
    # THE wire byte-compat pin: key order, separators and the ISO-8601 rendering
    # of the expiry are all part of the frozen contract, not formatting choices.
    assert proof_message(**GOLDEN_PROOF_INPUTS) == GOLDEN_PROOF_MESSAGE


def test_sign_node_proof_deterministic_golden():
    # Exact equality is legal: Ed25519 signing is deterministic (RFC 8032).
    assert sign_node_proof(GOLDEN_PRIVATE_KEY_B64, GOLDEN_PROOF_MESSAGE) == GOLDEN_SIGNATURE_B64


def test_verify_node_proof():
    assert verify_node_proof(GOLDEN_VERIFIER, GOLDEN_PROOF_MESSAGE, GOLDEN_SIGNATURE_B64) is True

    tampered_message = bytearray(GOLDEN_PROOF_MESSAGE)
    tampered_message[10] ^= 0x01
    assert (
        verify_node_proof(GOLDEN_VERIFIER, bytes(tampered_message), GOLDEN_SIGNATURE_B64) is False
    )

    tampered_sig = bytearray(_b64decode(GOLDEN_SIGNATURE_B64))
    tampered_sig[0] ^= 0x01
    assert (
        verify_node_proof(GOLDEN_VERIFIER, GOLDEN_PROOF_MESSAGE, _b64encode(bytes(tampered_sig)))
        is False
    )

    # A valid, unrelated verifier: the proof is bound to the key.
    assert (
        verify_node_proof(CLOUD_DEV_VERIFIER, GOLDEN_PROOF_MESSAGE, GOLDEN_SIGNATURE_B64) is False
    )
    # A prefix-less reference is refused before any crypto runs.
    assert (
        verify_node_proof(
            GOLDEN_VERIFIER.removeprefix("ed25519:"),
            GOLDEN_PROOF_MESSAGE,
            GOLDEN_SIGNATURE_B64,
        )
        is False
    )


# --- key file: generate-or-load, permissions, custody ------------------------


def test_generate_creates_key_file_with_modes(tmp_path):
    key_file = tmp_path / "link" / "node.key"
    identity = load_or_create_identity(key_file)

    assert key_file.exists()
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o700

    content = key_file.read_text()
    assert content.endswith("\n")
    assert KEY_LINE.fullmatch(content.rstrip("\n"))
    assert identity.verifier == public_reference(content.strip())
    assert identity.fingerprint == credential_fingerprint(identity.verifier)


def test_load_is_stable(tmp_path):
    key_file = tmp_path / "link" / "node.key"
    first = load_or_create_identity(key_file)
    before = key_file.read_bytes()

    second = load_or_create_identity(key_file)

    assert (second.verifier, second.fingerprint) == (first.verifier, first.fingerprint)
    assert key_file.read_bytes() == before


def test_golden_key_file_loads(tmp_path):
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.write_text(f"{GOLDEN_PRIVATE_KEY_B64}\n")
    key_file.chmod(0o600)

    identity = load_or_create_identity(key_file)

    assert identity.verifier == GOLDEN_VERIFIER
    assert identity.fingerprint == GOLDEN_FINGERPRINT
    assert identity.sign(GOLDEN_PROOF_MESSAGE) == GOLDEN_SIGNATURE_B64


@pytest.mark.parametrize(
    "content",
    ["not-base64!\n", GOLDEN_PRIVATE_KEY_B64[:-2], ""],
    ids=["not-base64", "truncated-key", "empty"],
)
def test_malformed_key_file_raises_never_regenerates(tmp_path, content):
    # A corrupt key is a LOUD failure, never a silent regeneration: regenerating
    # would change the node's enrolled cloud identity and permanently break the
    # ADR-W2 claim binding.
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.write_text(content)
    before = key_file.read_bytes()

    with pytest.raises(LinkIdentityError) as exc:
        load_or_create_identity(key_file)

    assert key_file.read_bytes() == before
    if content.strip():
        # The never-leak contract: the message names the failure, never the bytes.
        assert content.strip() not in str(exc.value)


def test_non_ascii_key_file_raises(tmp_path):
    # A binary blob at the key path is the same class of failure as corrupt
    # base64: loud, and never regenerated over.
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.write_bytes(b"\xff\xfe not ascii\n")
    before = key_file.read_bytes()

    with pytest.raises(LinkIdentityError):
        load_or_create_identity(key_file)

    assert key_file.read_bytes() == before


def test_unwritable_parent_raises(tmp_path):
    # The generate path fails loudly too — a key directory that cannot be
    # created must never degrade into an in-memory-only identity.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    with pytest.raises(LinkIdentityError):
        load_or_create_identity(blocker / "node.key")


def test_symlink_key_file_refused(tmp_path):
    # O_NOFOLLOW: a symlinked key file is a redirection of custody, refused.
    real = tmp_path / "elsewhere.key"
    real.write_text(f"{GOLDEN_PRIVATE_KEY_B64}\n")
    real.chmod(0o600)

    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.symlink_to(real)

    with pytest.raises(LinkIdentityError):
        load_or_create_identity(key_file)


def test_fifo_key_file_is_refused_not_hung(tmp_path):
    """A writerless FIFO at the node-key path must refuse *promptly*, not block.

    The lifespan loads the identity before it yields, so a blocking ``open``
    would hang the daemon at boot. Same exposure and same fix as the license
    path (backlog #31 / PR #118 Codex round 2): ``O_NONBLOCK`` + fd-based
    ``S_ISREG`` refusal. The read runs on a *daemon* thread joined with a
    timeout so a regression fails this test in five seconds rather than
    wedging the whole suite forever.
    """
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    os.mkfifo(key_file)
    captured: list[object] = []

    def _load() -> None:
        try:
            captured.append(load_or_create_identity(key_file))
        except LinkIdentityError as exc:  # the expected outcome
            captured.append(exc)

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    thread.join(5.0)
    if thread.is_alive():
        pytest.fail("load_or_create_identity blocked on a writerless FIFO (boot would hang)")
    assert len(captured) == 1
    assert isinstance(captured[0], LinkIdentityError), captured[0]
    assert "not a regular file" in str(captured[0])


def test_lax_file_perms_tightened_parent_left_alone(tmp_path, caplog):
    # The FILE is always tightened — that one is unambiguously ours. The parent
    # is not: [link].key_file may point at /tmp or a shared $HOME subtree, and
    # chmodding it would strip bits (sticky, setgid, a shared group) this module
    # does not own. So a lax directory is REPORTED, never repaired.
    key_file = tmp_path / "link" / "node.key"
    load_or_create_identity(key_file)
    key_file.chmod(0o644)
    key_file.parent.chmod(0o755)

    with caplog.at_level("WARNING", logger="nerdit.link.identity"):
        load_or_create_identity(key_file)

    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o755
    assert str(key_file.parent) in caplog.text


def test_module_created_parent_gets_0700(tmp_path):
    # The <data_dir>/link/ default: this module creates it, so it owns its mode
    # and states it explicitly (makedirs' mode is umask-masked).
    key_file = tmp_path / "link" / "node.key"
    load_or_create_identity(key_file)
    assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o700


def test_preexisting_parent_mode_preserved_on_generate(tmp_path, caplog):
    # An operator-chosen key_file parent keeps its mode through generation too —
    # only the key file itself is forced to 0o600.
    parent = tmp_path / "operator"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)  # defeat the umask, so the premise really holds
    key_file = parent / "node.key"

    with caplog.at_level("WARNING", logger="nerdit.link.identity"):
        load_or_create_identity(key_file)

    assert stat.S_IMODE(parent.stat().st_mode) == 0o755
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert str(parent) in caplog.text

    # …and through a subsequent load.
    load_or_create_identity(key_file)
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_generate_fsyncs_file_and_parent_directory(tmp_path, monkeypatch):
    # "Atomic write" is only half the claim: without an fsync of the PARENT
    # directory the rename can be lost to power loss, node.key comes back absent
    # and load_or_create_identity silently mints a different identity — breaking
    # the ADR-W2 claim binding of an already-enrolled node.
    synced: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        synced.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)

    key_file = tmp_path / "link" / "node.key"
    _generate_key_file(key_file)

    assert synced == ["file", "dir"]


def test_generate_surfaces_a_failed_directory_fsync(tmp_path, monkeypatch):
    # A durability failure is loud (LinkIdentityError), and leaves no temp copy
    # of the key behind at whatever mode the umask allowed.
    real_fsync = os.fsync

    def failing_dir_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(5, "Input/output error")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_dir_fsync)

    key_file = tmp_path / "link" / "node.key"
    with pytest.raises(LinkIdentityError):
        _generate_key_file(key_file)

    assert not list(key_file.parent.glob(".*.tmp"))


def test_no_temp_left_behind(tmp_path):
    # The write is atomic (unique temp sibling → no-replace os.link install);
    # a leftover temp would be a second copy of the key at whatever mode the
    # umask allowed.
    key_file = tmp_path / "link" / "node.key"
    load_or_create_identity(key_file)
    assert sorted(p.name for p in key_file.parent.iterdir()) == ["node.key"]


def test_orphan_tmp_never_blocks_generation(tmp_path):
    # PR #113 review: a fixed temp name meant an interrupted first boot left an
    # orphan that made the NEXT boot fail on O_EXCL EEXIST. Temp names are now
    # unique per attempt, so a crash orphan (old fixed-name style included) is
    # inert: generation succeeds around it and never unlinks another attempt's
    # file.
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    orphan = key_file.parent / ".node.key.tmp"
    orphan.write_bytes(b"stale-orphan-from-a-crashed-first-boot\n")

    identity = load_or_create_identity(key_file)
    assert key_file.exists()
    assert identity.verifier.startswith("ed25519:")
    assert orphan.exists()  # untouched, merely inert


def test_lost_install_race_adopts_installed_key(tmp_path):
    # PR #113 review: two first-boot callers racing generation must converge on
    # ONE identity. The install is a no-replace os.link; the loser adopts the
    # installed key instead of overwriting it (or destroying the winner's temp
    # file, as the old fixed-name cleanup could). Simulated by installing a key
    # between the caller's exists() check and its install step.
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.write_text(f"{GOLDEN_PRIVATE_KEY_B64}\n")
    os.chmod(key_file, 0o600)

    returned = _generate_key_file(key_file)
    assert returned == GOLDEN_PRIVATE_KEY_B64  # the winner's key, not a fresh one
    assert key_file.read_text().strip() == GOLDEN_PRIVATE_KEY_B64
    # No temp leftovers from the losing attempt.
    assert sorted(p.name for p in key_file.parent.iterdir()) == ["node.key"]


def test_repr_never_leaks_private_key(tmp_path):
    key_file = tmp_path / "link" / "node.key"
    key_file.parent.mkdir(mode=0o700)
    key_file.write_text(f"{GOLDEN_PRIVATE_KEY_B64}\n")
    key_file.chmod(0o600)
    identity = load_or_create_identity(key_file)

    for rendered in (repr(identity), str(identity)):
        assert GOLDEN_VERIFIER in rendered
        assert GOLDEN_FINGERPRINT in rendered
        assert GOLDEN_PRIVATE_KEY_B64 not in rendered


def test_resolve_key_file():
    assert resolve_key_file(None, "~/.nerdit") == Path.home() / ".nerdit" / "link" / "node.key"
    assert resolve_key_file("/abs/x.key", "~/.nerdit") == Path("/abs/x.key")
    assert resolve_key_file("~/k.key", "~/.nerdit") == Path.home() / "k.key"
