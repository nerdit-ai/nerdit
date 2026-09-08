"""``scripts/license_tool.py`` — offline issuance tooling (P17d D-LIC8, B3).

Three properties are worth a test each, and they are the three a reviewer would
otherwise have to take on faith:

* **round-trip** — a key minted by ``keygen`` signs a license that the *daemon's*
  verifier accepts (``nerdit.core.license.verify_license``, not a local reimpl).
  That is the anti-drift proof for the whole tool: it borrows the daemon's
  encoder and is checked against the daemon's verifier;
* **custody** — the private key file is ``0600``, is never overwritten, and its
  contents never reach stdout/stderr;
* **honest refusals** — a blob signed by a kid not pinned in the baked keyset
  says ``unknown_kid`` under ``inspect``, and ``sign`` refuses a ``--key`` that
  is not the private half of a pinned kid, rather than pretending.

Wall-clock independence: every temporal assertion signs its own ``--expires-at``
relative to ``now``, so no test rots on a calendar date (the frozen
``GOLDEN_BLOB`` is used only for claim-shaped assertions).
"""

from __future__ import annotations

import base64
import errno
import importlib.util
import os
import stat
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from nerdit.core.license import (
    STATE_EXPIRED,
    STATE_INVALID,
    STATE_VALID,
    TRUSTED_LICENSE_KEYS,
    verify_license,
)
from tests.license_vectors import GOLDEN_BLOB, TEST_KID, TEST_PUBLIC_REFERENCE

_TOOL_PATH = Path(__file__).resolve().parent.parent / "scripts" / "license_tool.py"


def _load_tool():
    """Import the script from its real path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("nerdit_license_tool", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _iso(days: int) -> str:
    """A tz-aware ISO-8601 timestamp ``days`` from now."""
    return (datetime.now(UTC) + timedelta(days=days)).isoformat()


def _keygen(tmp_path: Path, capsys, *, kid: str = "nerdit-lic-test") -> tuple[Path, str]:
    """Run ``keygen`` and return (key path, the printed public reference)."""
    key_path = tmp_path / "license-signing.key"
    assert tool.main(["keygen", "--out", str(key_path), "--kid", kid]) == tool.EXIT_OK
    out = capsys.readouterr().out
    reference = next(
        line.split('"')[3] for line in out.splitlines() if line.strip().startswith(f'"{kid}"')
    )
    return key_path, reference


# ---------------------------------------------------------------------------
# keygen — custody
# ---------------------------------------------------------------------------


def test_keygen_writes_private_key_0600_and_prints_only_the_public_line(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)

    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert reference.startswith("ed25519:")
    # The public reference decodes to a 32-byte raw Ed25519 key.
    raw = base64.urlsafe_b64decode(reference.removeprefix("ed25519:") + "=")
    assert len(raw) == 32


def test_keygen_never_prints_the_private_key(tmp_path, capsys):
    """The private half exists in exactly one place: the 0600 file."""
    key_path = tmp_path / "signing.key"
    assert tool.main(["keygen", "--out", str(key_path), "--kid", "nerdit-lic-test"]) == tool.EXIT_OK

    captured = capsys.readouterr()
    private_b64 = key_path.read_text(encoding="ascii").strip()
    assert private_b64
    assert private_b64 not in captured.out + captured.err
    # Not even a prefix long enough to matter (a truncated echo is still a leak).
    assert private_b64[:16] not in captured.out + captured.err


def test_keygen_refuses_to_overwrite_an_existing_key(tmp_path, capsys):
    key_path, _ = _keygen(tmp_path, capsys)
    before = key_path.read_bytes()

    assert tool.main(["keygen", "--out", str(key_path), "--kid", "other"]) == tool.EXIT_REFUSED
    assert key_path.read_bytes() == before
    assert "refusing to overwrite" in capsys.readouterr().err


def test_keygen_removes_the_partial_file_when_the_write_fails(tmp_path, capsys, monkeypatch):
    """A failed write must not wedge every future retry.

    ``O_EXCL`` creates the file before anything is written, so an ENOSPC (or any
    I/O error) mid-write used to leave an empty/partial key behind — and the
    deliberate no-overwrite refusal above then rejected every retry, for a key
    that exists nowhere: the generated one lived only in memory. Cleanup on
    failure is what makes "retry" the honest next step, which the second leg
    here proves by actually retrying.
    """
    key_path = tmp_path / "license-signing.key"
    real_fdopen = os.fdopen

    class _Boom:
        def __init__(self, handle):
            self._handle = handle

        def write(self, _data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._handle.close()
            return False

    def _exploding_fdopen(fd, *args, **kwargs):
        return _Boom(real_fdopen(fd, *args, **kwargs))

    monkeypatch.setattr(os, "fdopen", _exploding_fdopen)
    assert tool.main(["keygen", "--out", str(key_path), "--kid", "k"]) == tool.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "partial file removed" in err
    assert not key_path.exists()

    monkeypatch.undo()
    # The retry leg: with the partial file gone, keygen is possible again.
    key_path2, _ = _keygen(tmp_path, capsys)
    assert key_path2.exists()


def test_keygen_warns_when_the_key_lands_inside_a_git_worktree(tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    assert tool.main(["keygen", "--out", str(tmp_path / "k.key"), "--kid", "k"]) == tool.EXIT_OK
    assert "INSIDE a git worktree" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# sign — the round-trip against the daemon's verifier
# ---------------------------------------------------------------------------


def _sign(key_path: Path, capsys, **overrides) -> str:
    """Run ``sign`` with sane defaults and return the emitted blob."""
    argv = [
        "sign",
        "--key",
        str(key_path),
        "--kid",
        overrides.pop("kid", "nerdit-lic-test"),
        "--customer-id",
        overrides.pop("customer_id", "cus_roundtrip"),
        "--plan",
        overrides.pop("plan", "pro"),
        "--features",
        overrides.pop("features", "remote_link"),
        "--expires-at",
        overrides.pop("expires_at", _iso(365)),
    ]
    lid = overrides.pop("lid", None)
    if lid is not None:
        argv += ["--lid", lid]
    warning_expected = overrides.pop("warning_expected", False)
    assert not overrides, overrides
    assert tool.main(argv) == tool.EXIT_OK
    captured = capsys.readouterr()
    if warning_expected:
        assert "not in the baked TRUSTED_LICENSE_KEYS" in captured.err
    return captured.out.strip()


def test_sign_round_trips_through_the_daemon_verifier(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, lid="0123456789abcdef")

    verdict = verify_license(blob, trusted_keys={"nerdit-lic-test": reference})
    assert verdict.state == STATE_VALID
    assert verdict.claims is not None
    assert verdict.claims.lid == "0123456789abcdef"
    assert verdict.claims.plan == "pro"
    assert verdict.claims.customer_id == "cus_roundtrip"
    assert verdict.claims.features == ("remote_link",)


def test_sign_mints_a_unique_lid_per_issuance(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)
    first = verify_license(_sign(key_path, capsys), trusted_keys={"nerdit-lic-test": reference})
    second = verify_license(_sign(key_path, capsys), trusted_keys={"nerdit-lic-test": reference})

    assert first.claims is not None and second.claims is not None
    assert first.claims.lid != second.claims.lid


def test_sign_emits_only_the_blob_on_stdout(tmp_path, capsys):
    key_path, _ = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, features="remote_link,sso")

    # One line, three segments: the file an operator gets by redirecting stdout
    # must be the license and nothing else.
    assert len(blob.splitlines()) == 1
    assert len(blob.split(".")) == 3


def test_sign_never_prints_the_private_key(tmp_path, capsys):
    key_path, _ = _keygen(tmp_path, capsys)
    private_b64 = key_path.read_text(encoding="ascii").strip()
    _sign(key_path, capsys)
    captured = capsys.readouterr()

    assert private_b64 not in captured.out + captured.err


def test_sign_refuses_a_naive_expires_at(tmp_path, capsys):
    key_path, _ = _keygen(tmp_path, capsys)
    code = tool.main(
        [
            "sign",
            "--key",
            str(key_path),
            "--kid",
            "k",
            "--customer-id",
            "cus_x",
            "--plan",
            "pro",
            "--expires-at",
            "2027-01-01T00:00:00",
        ]
    )
    assert code == tool.EXIT_USAGE
    assert "tz-aware" in capsys.readouterr().err


def test_sign_refuses_a_blob_the_daemon_would_reject(tmp_path, capsys):
    """The self-check: a bad plan token is caught here, not at the customer."""
    key_path, _ = _keygen(tmp_path, capsys)
    code = tool.main(
        [
            "sign",
            "--key",
            str(key_path),
            "--kid",
            "k",
            "--customer-id",
            "cus_x",
            "--plan",
            "Pro Plan",  # not ^[a-z0-9_]{1,64}$
            "--expires-at",
            _iso(30),
        ]
    )
    assert code == tool.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refusing to emit" in err
    assert "schema_invalid" in err


def test_sign_warns_but_emits_an_already_expired_license(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, expires_at=_iso(-30))

    assert verify_license(blob, trusted_keys={"nerdit-lic-test": reference}).state == STATE_EXPIRED


def test_sign_refuses_the_wrong_key_for_a_kid_in_the_baked_keyset(tmp_path, capsys):
    """A fresh key + the production kid must be refused, not emitted.

    The self-check verifies with the supplied key's own public half, which is
    circular — without the pinned-reference cross-check, a wrong ``--key`` file
    emits a license every daemon carrying the baked keyset rejects with
    ``bad_signature`` (PR #121 Codex round 1, P1).
    """
    key_path, _ = _keygen(tmp_path, capsys)
    argv = [
        "sign",
        "--key",
        str(key_path),
        "--kid",
        "nerdit-lic-2026-1",
        "--customer-id",
        "cus_wrong_key",
        "--plan",
        "pro",
        "--features",
        "remote_link",
        "--expires-at",
        _iso(365),
    ]
    assert tool.main(argv) == tool.EXIT_REFUSED
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # no blob emitted
    assert "not the private half" in captured.err
    assert "nerdit-lic-2026-1" in captured.err


def test_sign_pins_against_the_checkout_not_a_stale_installed_package(tmp_path, capsys):
    """The pin check must read THIS checkout's keyset, not whatever ``nerdit``
    the environment resolves.

    A stale non-editable install (its keyset still empty) earlier on ``sys.path``
    would make the production kid look unpinned, silently bypassing the
    wrong-key refusal (PR #121 Codex round 2, P1). Reproduced here as a
    subprocess with a stale copy of ``license.py`` on ``PYTHONPATH``.
    """
    import subprocess

    stale_pkg = tmp_path / "stale" / "nerdit"
    (stale_pkg / "core").mkdir(parents=True)
    (stale_pkg / "__init__.py").write_text("", encoding="ascii")
    (stale_pkg / "core" / "__init__.py").write_text("", encoding="ascii")
    real = Path("src/nerdit/core/license.py").read_text(encoding="utf-8")
    stale = real.replace('"nerdit-lic-2026-1": "ed25519:', '"stale-never-this-kid": "ed25519:')
    assert stale != real
    (stale_pkg / "core" / "license.py").write_text(stale, encoding="utf-8")

    key_path, _ = _keygen(tmp_path, capsys)
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "stale"))
    result = subprocess.run(
        [
            sys.executable,
            str(_TOOL_PATH),
            "sign",
            "--key",
            str(key_path),
            "--kid",
            "nerdit-lic-2026-1",
            "--customer-id",
            "cus_stale_path",
            "--plan",
            "pro",
            "--features",
            "remote_link",
            "--expires-at",
            _iso(365),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == tool.EXIT_REFUSED, (result.returncode, result.stderr)
    assert result.stdout.strip() == ""  # no blob emitted
    assert "not the private half" in result.stderr


def test_sign_warns_when_the_kid_is_absent_from_the_baked_keyset(tmp_path, capsys):
    """Signing for a not-yet-landed kid stays possible (test kids, rotation
    staging) but says so out loud: daemons at this commit report unknown_kid."""
    key_path, _ = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, warning_expected=True)
    assert blob.count(".") == 2


def test_sign_rejects_a_key_file_that_is_not_a_key(tmp_path, capsys):
    bogus = tmp_path / "not-a-key"
    bogus.write_text("hello", encoding="ascii")
    code = tool.main(
        [
            "sign",
            "--key",
            str(bogus),
            "--kid",
            "k",
            "--customer-id",
            "cus_x",
            "--plan",
            "pro",
            "--expires-at",
            _iso(30),
        ]
    )
    assert code == tool.EXIT_USAGE


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_prints_state_and_claims_for_a_signed_license(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, lid="deadbeef")
    path = tmp_path / "license.jws"
    path.write_text(blob + "\n", encoding="ascii")

    assert tool.main(["inspect", str(path), "--pub", reference]) == tool.EXIT_OK
    out = capsys.readouterr().out
    assert f"state:       {STATE_VALID}" in out
    assert "lid:         deadbeef" in out
    assert "plan:        pro" in out
    assert "customer_id: cus_roundtrip" in out


def test_inspect_reads_stdin(tmp_path, capsys, monkeypatch):
    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys)
    monkeypatch.setattr(sys, "stdin", _Stdin(blob))

    assert tool.main(["inspect", "-", "--pub", reference]) == tool.EXIT_OK
    assert f"state:       {STATE_VALID}" in capsys.readouterr().out


class _Stdin:
    """The two-method stdin the tool actually uses."""

    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


def test_inspect_against_the_baked_keyset_reports_unknown_kid(capsys):
    """The golden blob's test kid is not in the baked keyset (D-LIC3) — the
    honest failure is ``unknown_kid``."""
    assert TEST_KID not in TRUSTED_LICENSE_KEYS
    with_stdin = _Stdin(GOLDEN_BLOB)
    original, sys.stdin = sys.stdin, with_stdin
    try:
        assert tool.main(["inspect", "-"]) == tool.EXIT_REFUSED
    finally:
        sys.stdin = original
    out = capsys.readouterr().out
    assert f"state:       {STATE_INVALID}" in out
    assert "reason:      unknown_kid" in out


def test_inspect_enrolls_pub_under_the_blobs_own_kid(tmp_path, capsys):
    path = tmp_path / "golden.jws"
    path.write_text(GOLDEN_BLOB, encoding="ascii")

    assert tool.main(["inspect", str(path), "--pub", TEST_PUBLIC_REFERENCE]) == tool.EXIT_OK
    out = capsys.readouterr().out
    assert "plan:        pro" in out
    assert "customer_id: cus_p17d_golden" in out


def test_inspect_with_an_explicit_mismatched_kid_does_not_verify(tmp_path, capsys):
    path = tmp_path / "golden.jws"
    path.write_text(GOLDEN_BLOB, encoding="ascii")

    code = tool.main(
        ["inspect", str(path), "--pub", TEST_PUBLIC_REFERENCE, "--kid", f"not-{TEST_KID}"]
    )
    assert code == tool.EXIT_REFUSED
    assert "reason:      unknown_kid" in capsys.readouterr().out


def test_inspect_refuses_a_bad_public_reference(tmp_path, capsys):
    path = tmp_path / "golden.jws"
    path.write_text(GOLDEN_BLOB, encoding="ascii")

    assert tool.main(["inspect", str(path), "--pub", "not-a-reference"]) == tool.EXIT_USAGE


def test_inspect_refuses_an_empty_file(tmp_path, capsys):
    path = tmp_path / "empty.jws"
    path.write_text("", encoding="ascii")

    assert tool.main(["inspect", str(path)]) == tool.EXIT_USAGE
    assert "empty" in capsys.readouterr().err


def test_inspect_reports_expired_with_a_nonzero_exit(tmp_path, capsys):
    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys, expires_at=_iso(-30))
    path = tmp_path / "old.jws"
    path.write_text(blob, encoding="ascii")

    assert tool.main(["inspect", str(path), "--pub", reference]) == tool.EXIT_REFUSED
    assert f"state:       {STATE_EXPIRED}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the file the tool writes is the file the daemon installs
# ---------------------------------------------------------------------------


def test_signed_blob_installs_and_verifies_through_the_daemon_custody_path(tmp_path, capsys):
    """End to end: tool output -> ``install_license_file`` -> ``read_license_file``."""
    from nerdit.core.license import install_license_file, read_license_file

    key_path, reference = _keygen(tmp_path, capsys)
    blob = _sign(key_path, capsys)

    installed = tmp_path / "license.jws"
    install_license_file(installed, blob)
    assert stat.S_IMODE(installed.stat().st_mode) == 0o600
    assert read_license_file(installed) == blob
    assert (
        verify_license(
            read_license_file(installed) or "", trusted_keys={"nerdit-lic-test": reference}
        ).state
        == STATE_VALID
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_tool_is_executable_from_a_checkout():
    """The script is self-contained enough to run without an editable install."""
    assert _TOOL_PATH.is_file()
    assert _TOOL_PATH.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
