"""CLI tests for ``nerdit untrust`` (WS3) — the offline trust-store removal.

Mirrors ``test_cli_trust.py``: a real self-signed certificate saved under a
faked ``~/.nerdit``, the command body called directly, and the per-platform
removal command tables asserted exactly (they are the inverse of
``_install_ca``). The security-relevant assertions: the fingerprint is
recomputed from the file body (never the filename), a failed OS removal keeps
the saved copy (fingerprint stays recoverable), and there is no path that
deletes a cert without a successful removal.
"""

import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from nerdit.cli.commands.trust import _untrust_commands, untrust
from nerdit.utils.certs import ca_fingerprint, parse_single_certificate


def _self_signed_pem() -> str:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Nerdit Test CA")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM).decode()


PEM = _self_signed_pem()
FP = ca_fingerprint(PEM)
FP8 = FP.removeprefix("sha256:")[:8]


def _sha1_upper(pem: str) -> str:
    from cryptography.hazmat.primitives import hashes

    return parse_single_certificate(pem).fingerprint(hashes.SHA1()).hex().upper()


class _FakeRun:
    """Records subprocess.run argv and returns scripted return codes."""

    def __init__(self, returncodes=None, raise_oserror=False):
        self.calls: list[list[str]] = []
        self._returncodes = list(returncodes or [])
        self._raise = raise_oserror

    def __call__(self, cmd, check=False):
        self.calls.append(list(cmd))
        if self._raise:
            raise OSError("command not found")
        rc = self._returncodes.pop(0) if self._returncodes else 0
        return SimpleNamespace(returncode=rc)


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _save_cert(home: Path, pem: str = PEM, fp8: str = FP8) -> Path:
    d = home / ".nerdit"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"nerdit-root-{fp8}.crt"
    path.write_text(pem)
    return path


# -- _untrust_commands: exact per-platform tables --------------------------------


def test_untrust_commands_linux(monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    cert = parse_single_certificate(PEM)
    cmds = _untrust_commands(cert)
    assert cmds == [
        ["sudo", "rm", "-f", f"/usr/local/share/ca-certificates/nerdit-{FP8}.crt"],
        ["sudo", "update-ca-certificates", "--fresh"],
    ]


def test_untrust_commands_darwin_uses_uppercase_sha1(monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Darwin")
    cert = parse_single_certificate(PEM)
    cmds = _untrust_commands(cert)
    sha1 = _sha1_upper(PEM)
    assert cmds == [
        [
            "sudo",
            "security",
            "delete-certificate",
            "-Z",
            sha1,
            "/Library/Keychains/System.keychain",
        ]
    ]
    # the -Z argument is upper-case hex
    assert sha1 == sha1.upper() and cmds[0][4] == sha1


def test_untrust_commands_windows(monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Windows")
    cert = parse_single_certificate(PEM)
    cmds = _untrust_commands(cert)
    assert cmds == [["certutil", "-delstore", "Root", _sha1_upper(PEM)]]


def test_untrust_commands_unsupported_returns_none(monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "FreeBSD")
    assert _untrust_commands(parse_single_certificate(PEM)) is None


# -- command body ----------------------------------------------------------------


def test_untrust_no_certs_exit_0(fake_home):
    # ~/.nerdit does not even exist; the glob returns nothing.
    untrust(fingerprint=None, yes=True)  # returns without raising


def test_untrust_success_deletes_saved_copy(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    untrust(fingerprint=None, yes=True)
    assert run.calls[0][:2] == ["sudo", "rm"]
    assert run.calls[1] == ["sudo", "update-ca-certificates", "--fresh"]
    assert not path.exists()  # deleted after successful removal


def test_untrust_failure_keeps_saved_copy_and_hints(fake_home, monkeypatch, capsys):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", _FakeRun(returncodes=[1]))
    with pytest.raises(typer.Exit) as excinfo:
        untrust(fingerprint=None, yes=True)
    assert excinfo.value.exit_code == 1
    assert path.exists()  # fingerprint stays recoverable
    assert "manually" in capsys.readouterr().out


def test_untrust_oserror_keeps_saved_copy(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", _FakeRun(raise_oserror=True))
    with pytest.raises(typer.Exit) as excinfo:
        untrust(fingerprint=None, yes=True)
    assert excinfo.value.exit_code == 1
    assert path.exists()


def test_untrust_confirm_decline_skips_and_keeps(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    untrust(fingerprint=None, yes=False)
    assert run.calls == []  # nothing executed
    assert path.exists()  # nothing deleted


def test_untrust_fingerprint_match_proceeds(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    untrust(fingerprint=FP, yes=True)
    assert not path.exists()
    assert run.calls  # a removal ran


def test_untrust_fingerprint_mismatch_exits_1_and_keeps(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    with pytest.raises(typer.Exit) as excinfo:
        untrust(fingerprint="sha256:" + "0" * 64, yes=True)
    assert excinfo.value.exit_code == 1
    assert run.calls == []  # never touched the trust store
    assert path.exists()


def test_untrust_recomputes_fingerprint_from_body_not_filename(fake_home, monkeypatch):
    # Saved under a LYING filename (wrong fp8); the real cert body still governs.
    path = _save_cert(fake_home, fp8="deadbeef")
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    untrust(fingerprint=FP, yes=True)  # matches by body, despite the bad name
    # the trust-store target is derived from the body fp8, not the file's name
    assert run.calls[0][3] == f"/usr/local/share/ca-certificates/nerdit-{FP8}.crt"
    assert not path.exists()


def test_untrust_unparseable_file_skipped(fake_home, monkeypatch, capsys):
    good = _save_cert(fake_home)
    bad = fake_home / ".nerdit" / "nerdit-root-garbage.crt"
    bad.write_text("not a certificate")
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    untrust(fingerprint=None, yes=True)
    out = capsys.readouterr().out
    assert "Skipping unreadable" in out
    assert not good.exists()  # the good one still got removed
    assert bad.exists()  # the unparseable one is left untouched


def test_untrust_unsupported_platform_keeps_file(fake_home, monkeypatch):
    path = _save_cert(fake_home)
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "FreeBSD")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    untrust(fingerprint=None, yes=True)
    assert run.calls == []
    assert path.exists()  # manual removal advised; local copy preserved
