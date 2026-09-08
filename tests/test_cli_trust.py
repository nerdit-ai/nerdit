"""CLI tests for ``nerdit trust`` (P9, part B) — the fingerprint gate.

Same three-layer structure as ``test_cli_secrets.py``: NerditClient method
over ``httpx.MockTransport``, the async command body with a fake client, and
argv-level parsing via Typer's CliRunner. The security-relevant assertions:
a fingerprint mismatch exits non-zero and installs/writes NOTHING, and there
is no silent-install path (interactive decline also aborts).
"""

import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.trust import _trust_async
from nerdit.utils.certs import ca_fingerprint


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


# -- client method --------------------------------------------------------------


async def test_get_proxy_ca_hits_public_api_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, text=PEM, headers={"content-type": "application/x-pem-file"})

    client = NerditClient("localhost", 9321, token=None, transport=httpx.MockTransport(handler))
    pem = await client.get_proxy_ca()
    assert pem == PEM
    assert seen["path"] == "/api/proxy/ca"


# -- async command body ----------------------------------------------------------


class _FakeClient:
    def __init__(self, pem: str = PEM):
        self._pem = pem
        self.calls = 0

    async def get_proxy_ca(self) -> str:
        self.calls += 1
        return self._pem


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    return client


async def test_trust_fingerprint_mismatch_aborts_and_writes_nothing(fake_client, tmp_path):
    out = tmp_path / "root.crt"
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async("sha256:" + "0" * 64, out)
    assert excinfo.value.exit_code == 1
    assert not out.exists()


async def test_trust_fingerprint_match_writes_output(fake_client, tmp_path):
    out = tmp_path / "root.crt"
    await _trust_async(FP, out)
    assert out.read_text() == PEM


async def test_trust_accepts_colon_separated_uppercase_fingerprint(fake_client, tmp_path):
    hexpart = FP.removeprefix("sha256:")
    colons = ":".join(hexpart[i : i + 2] for i in range(0, len(hexpart), 2)).upper()
    out = tmp_path / "root.crt"
    await _trust_async(colons, out)
    assert out.exists()


async def test_trust_interactive_decline_aborts(fake_client, tmp_path, monkeypatch):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    out = tmp_path / "root.crt"
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async(None, out)
    assert excinfo.value.exit_code == 1
    assert not out.exists()


async def test_trust_interactive_confirm_proceeds(fake_client, tmp_path, monkeypatch):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    out = tmp_path / "root.crt"
    await _trust_async(None, out)
    assert out.read_text() == PEM


async def test_trust_invalid_pem_aborts(monkeypatch, tmp_path):
    client = _FakeClient(pem="garbage")
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async(FP, tmp_path / "root.crt")
    assert excinfo.value.exit_code == 1


# -- argv-level ------------------------------------------------------------------

_runner = CliRunner()


def test_cli_trust_argv_fingerprint_and_output(monkeypatch, tmp_path):
    from nerdit.cli.app import app

    client = _FakeClient()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    out = tmp_path / "root.crt"
    result = _runner.invoke(app, ["trust", "--fingerprint", FP, "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_text() == PEM


def test_cli_trust_argv_mismatch_exits_1(monkeypatch, tmp_path):
    from nerdit.cli.app import app

    client = _FakeClient()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    out = tmp_path / "root.crt"
    result = _runner.invoke(
        app, ["trust", "--fingerprint", "sha256:" + "f" * 64, "--output", str(out)]
    )
    assert result.exit_code == 1
    assert not out.exists()


# -- multi-cert smuggling (the fingerprint gate must cover ALL installed bytes) --


async def test_trust_rejects_concatenated_rogue_cert(monkeypatch, tmp_path):
    """An on-path attacker serving `legit-root || rogue-root` must be rejected.

    `load_pem_x509_certificate` only parses the first cert of a blob, so a
    fingerprint computed naively over the body would match the legitimate
    root while the OS install step trusts every certificate in the file.
    The gate must reject anything that is not exactly one certificate.
    """
    rogue = _self_signed_pem()
    client = _FakeClient(pem=PEM + rogue)
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
    out = tmp_path / "root.crt"
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async(FP, out)  # FP is the legit cert's fingerprint
    assert excinfo.value.exit_code == 1
    assert not out.exists()


def test_ca_fingerprint_rejects_multi_cert_blob():
    from nerdit.utils.certs import ca_fingerprint as fp

    with pytest.raises(ValueError):
        fp(PEM + _self_signed_pem())
    with pytest.raises(ValueError):
        fp("")


# -- OS trust-store install path (_install_ca) ------------------------------------


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


async def test_install_linux_runs_cp_and_update(fake_client, fake_home, monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    await _trust_async(FP, None)
    assert len(run.calls) == 2
    assert run.calls[0][:2] == ["sudo", "cp"]
    assert run.calls[0][2].endswith(".crt")
    assert run.calls[1] == ["sudo", "update-ca-certificates"]
    # the saved cert is the canonical single-cert PEM
    saved = Path(run.calls[0][2])
    assert saved.read_text() == PEM


async def test_install_darwin_uses_security_trustroot(fake_client, fake_home, monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Darwin")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    await _trust_async(FP, None)
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert cmd[:3] == ["sudo", "security", "add-trusted-cert"]
    assert "trustRoot" in cmd and "/Library/Keychains/System.keychain" in cmd


async def test_install_windows_uses_certutil(fake_client, fake_home, monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Windows")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    await _trust_async(FP, None)
    assert run.calls == [["certutil", "-addstore", "Root", run.calls[0][3]]]


async def test_install_failure_exits_1_with_manual_hint(
    fake_client, fake_home, monkeypatch, capsys
):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    run = _FakeRun(returncodes=[1])
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async(FP, None)
    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "manually" in out


async def test_install_oserror_exits_1(fake_client, fake_home, monkeypatch):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "Linux")
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", _FakeRun(raise_oserror=True))
    with pytest.raises(typer.Exit) as excinfo:
        await _trust_async(FP, None)
    assert excinfo.value.exit_code == 1


async def test_install_unsupported_platform_saves_but_does_not_run(
    fake_client, fake_home, monkeypatch, capsys
):
    monkeypatch.setattr("nerdit.cli.commands.trust.platform.system", lambda: "FreeBSD")
    run = _FakeRun()
    monkeypatch.setattr("nerdit.cli.commands.trust.subprocess.run", run)
    await _trust_async(FP, None)
    assert run.calls == []
    assert "manually" in capsys.readouterr().out
