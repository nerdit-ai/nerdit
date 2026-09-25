"""Test daemon-free uninstall with real temporary files and a fake Docker client.

Dry runs and declined confirmation delete nothing. A held restore lock refuses
teardown; capture the Caddy PID before deletion and constrain container sweeps
with both labels and all=True. Reuse the shared CA-removal hints.

Pin stop, verify, then remove: failed systemctl stops are reported, and nothing,
including unit files, is removed before acquiring the data-directory flock.
"""

from __future__ import annotations

import datetime
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.cli.commands import trust as trust_mod
from nerdit.cli.commands import uninstall as uninstall_mod
from nerdit.cli.commands.uninstall import (
    UninstallManifest,
    _terminate,
    build_manifest,
    uninstall,
)
from nerdit.config.settings import NerditSettings
from nerdit.utils.certs import ca_fingerprint
from nerdit.utils.install_layout import InstallLayout, ServiceUnit

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX flock guard")


#: The real implementation, captured at import time BEFORE the autouse fixture
#: below can shadow it. `TestUnitHomeAdoption` restores it: patching the very
#: function that class exists to exercise made every one of its assertions run
#: against a lambda, so the suite could not have passed on a platform where
#: those tests are not short-circuited (found by CI on Linux + Codex).
_REAL_ADOPT_UNIT_HOME = uninstall_mod._adopt_unit_home


def _adopt_unit_home_disabled() -> None:
    """Default: a real /etc unit on the dev box must not redirect $HOME."""
    return


@pytest.fixture(autouse=True)
def _no_install_layout(monkeypatch):
    """(P30) Default every test to "not an installer-made install" so a real
    /opt/nerdit or systemd unit on the developer's box cannot change what this
    suite tears down. The layout-aware tests opt in explicitly.
    """
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: None)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: None)
    # Same reasoning for the unit-home adoption: a real /etc/systemd/system unit
    # on the box must not redirect $HOME under the suite. Its own tests below
    # call the function directly.
    monkeypatch.setattr(uninstall_mod, "_adopt_unit_home", _adopt_unit_home_disabled)
    # The adoption records module state. With the function stubbed it is never
    # reset, so a real adoption performed by TestUnitHomeAdoption would leak
    # into every later test and make the containment guard refuse their paths.
    monkeypatch.setattr(uninstall_mod, "_ADOPTED_HOME", None)
    monkeypatch.setattr(uninstall_mod, "_INVOKING_HOME", None)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class _FakeContainer:
    def __init__(self, cid: str, name: str, status: str) -> None:
        self.id = cid
        self.name = name
        self.status = status
        self.removed = False

    def remove(self, force: bool = False) -> None:
        self.removed = True


class _FakeImage:
    def __init__(self, tags: list[str], size: int = 0) -> None:
        self.tags = tags
        self.attrs = {"Size": size}


class _FakeContainers:
    def __init__(self, containers: list[_FakeContainer]) -> None:
        self._containers = containers
        self.list_calls: list[dict] = []

    def list(self, all: bool = False, filters=None) -> list[_FakeContainer]:  # noqa: A002
        self.list_calls.append({"all": all, "filters": filters})
        return list(self._containers)


class _FakeImages:
    def __init__(self, images: list[_FakeImage]) -> None:
        self._images = images
        self.removed: list[str] = []

    def list(self) -> list[_FakeImage]:
        return list(self._images)

    def remove(self, tag: str) -> None:
        self.removed.append(tag)
        self._images = [im for im in self._images if tag not in im.tags]


class _FakeClient:
    def __init__(self, containers=None, images=None) -> None:
        self.containers = _FakeContainers(containers or [])
        self.images = _FakeImages(images or [])


def _factory(client: _FakeClient):
    return lambda: client


def _down_factory():
    def f():
        raise RuntimeError("Cannot connect to the Docker daemon")

    return f


# --------------------------------------------------------------------------- #
# cert helper (a real self-signed cert, like test_untrust)
# --------------------------------------------------------------------------- #


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
FP8 = ca_fingerprint(PEM).removeprefix("sha256:")[:8]


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _settings(
    data_dir: Path,
    *,
    pid_file: Path,
    upload_dir: Path,
    key_file: Path | None = None,
    instance_id: str = "default",
    ollama_image: str | None = None,
    archive_dir: Path | None = None,
) -> NerditSettings:
    daemon = {
        "pid_file": str(pid_file),
        "upload_dir": str(upload_dir),
        "instance_id": instance_id,
        "port": 59998,
    }
    kwargs: dict = {"data_dir": str(data_dir), "daemon": daemon}
    if key_file is not None:
        kwargs["security"] = {"secrets_key_file": str(key_file)}
    if ollama_image is not None:
        kwargs["models"] = {"ollama_image": ollama_image}
    if archive_dir is not None:
        kwargs["retention"] = {"audit_archive_dir": str(archive_dir)}
    return NerditSettings(**kwargs)


def _write_db(
    path: Path, service_names: list[str], *, extra_rows: list[tuple] | None = None
) -> None:
    """A minimal real sqlite DB carrying the instance's deployed-app rows
    (image attribution reads ``kind='service'`` rows' ``config['image_repo']``
    — see `_owned_app_repos`). ``extra_rows`` are raw ``(id, kind,
    service_name, config)`` tuples for non-app rows."""
    import json
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE jobs (id TEXT, kind TEXT, service_name TEXT, config TEXT)")
    rows = [
        (f"j{i}", "service", n, json.dumps({"image_repo": f"nerdit-app/{n}"}))
        for i, n in enumerate(service_names)
    ]
    con.executemany("INSERT INTO jobs VALUES (?, ?, ?, ?)", rows + (extra_rows or []))
    con.commit()
    con.close()


def _populate_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_db(data_dir / "nerdit.db", ["x"])
    (data_dir / "nerdit.db-wal").write_text("wal")
    (data_dir / "nerdit.db-shm").write_text("shm")
    (data_dir / "secrets").mkdir()
    (data_dir / "secrets" / "app.enc").write_text("enc")
    (data_dir / "secrets.key").write_text("key")
    (data_dir / "services").mkdir()
    (data_dir / "services" / "x").mkdir()
    (data_dir / "services" / "x" / "data.txt").write_text("payload")
    (data_dir / "backups").mkdir()
    (data_dir / "backups" / "nerdit-backup-1.tar.gz").write_text("bk")
    (data_dir / "backups" / "nerdit-volumes-x-1.tar.gz").write_text("vol")
    (data_dir / "caddy.pid").write_text("888888")
    (data_dir / "caddy.log").write_text("log")
    (data_dir / "caddy-bootstrap.json").write_text("{}")
    (data_dir / "caddy" / "pki").mkdir(parents=True)
    (data_dir / "caddy" / "pki" / "root.crt").write_text("root")
    # Leftover from a pre-0.6 install (the telemetry plane is deleted): a
    # data-dir purge must still take it.
    (data_dir / "telemetry_id").write_text("uuid")
    (data_dir / ".restore.lock").write_text("")


def _save_ca_cert(home: Path) -> Path:
    d = home / ".nerdit"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"nerdit-root-{FP8}.crt"
    path.write_text(PEM)
    return path


def _patch_command(monkeypatch, settings, client):
    """Wire the command's module seams to the fake settings + docker client."""
    monkeypatch.setattr(uninstall_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(uninstall_mod, "_default_docker_client", lambda: client)
    monkeypatch.setattr(uninstall_mod.time, "sleep", lambda *_a, **_k: None)


# --------------------------------------------------------------------------- #
# 1. manifest enumeration
# --------------------------------------------------------------------------- #


def test_manifest_enumeration_full_tree(fake_home, tmp_path):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    (data_dir / "nerditd.pid").unlink(missing_ok=True)

    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("999999")
    (pid_file.parent / "nerditd.boot.log").write_text("boot")
    (pid_file.parent / "nerditd.boot.log.1").write_text("boot.1")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "a.zip").write_text("z")
    key_file = tmp_path / "keys" / "secrets.key"
    key_file.parent.mkdir()
    key_file.write_text("K")
    cert = _save_ca_cert(fake_home)

    settings = _settings(data_dir, pid_file=pid_file, upload_dir=upload_dir, key_file=key_file)
    client = _FakeClient(
        containers=[_FakeContainer("a" * 64, "svc-x", "exited")],
        images=[_FakeImage(["nerdit-app/x:1"], size=1000)],
    )
    manifest = build_manifest(settings, purge_images=False, docker_client_factory=_factory(client))

    assert manifest.data_dir == data_dir
    assert manifest.data_dir_bytes > 0
    assert manifest.daemon_pid == 999999
    assert manifest.caddy_pid == 888888  # captured pre-deletion
    kinds = {ep.kind for ep in manifest.extra_paths}
    assert kinds == {"pid", "boot_log", "upload_dir", "key_file"}
    boot_paths = {ep.path.name for ep in manifest.extra_paths if ep.kind == "boot_log"}
    assert boot_paths == {"nerditd.boot.log", "nerditd.boot.log.1"}
    assert cert in manifest.ca_certs
    assert manifest.containers == [("a" * 12, "svc-x", "exited")]
    assert manifest.images == [("nerdit-app/x:1", 1000)]
    assert manifest.docker_error is None
    # stopped container is seen -> all=True
    assert client.containers.list_calls[0]["all"] is True


# --------------------------------------------------------------------------- #
# 2. dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_deletes_nothing_and_wins_over_yes(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    client = _FakeClient(containers=[_FakeContainer("b" * 64, "svc", "running")])
    _patch_command(monkeypatch, settings, client)

    def _boom(*_a, **_k):
        raise AssertionError("prompt must not run on --dry-run")

    monkeypatch.setattr(typer, "prompt", _boom)

    uninstall(yes=True, dry_run=True, purge_images=False, keep_data=False)

    assert data_dir.exists()
    assert (data_dir / "nerdit.db").exists()
    assert "Dry run" in capsys.readouterr().out
    assert client.containers._containers[0].removed is False


# --------------------------------------------------------------------------- #
# 3. confirm gating
# --------------------------------------------------------------------------- #


def test_wrong_word_aborts_nothing_deleted(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    monkeypatch.setattr(typer, "prompt", lambda *_a, **_k: "no")

    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=False, dry_run=False, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1
    assert data_dir.exists()


def test_typing_uninstall_proceeds(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    monkeypatch.setattr(typer, "prompt", lambda *_a, **_k: "uninstall")

    uninstall(yes=False, dry_run=False, purge_images=False, keep_data=False)
    assert not data_dir.exists()


def test_yes_skips_prompt(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    monkeypatch.setattr(
        typer, "prompt", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no prompt"))
    )

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not data_dir.exists()


# --------------------------------------------------------------------------- #
# 4. ordering — caddy pid read before deletion
# --------------------------------------------------------------------------- #


def test_caddy_pid_read_before_deletion(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(
        uninstall_mod, "_terminate", lambda pid, label, **_k: calls.append((pid, label)) or True
    )

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    # caddy.pid content 888888 reached the kill helper even though the file
    # (inside data_dir) is gone afterwards.
    assert (888888, "caddy") in calls
    assert not (data_dir / "caddy.pid").exists()


# --------------------------------------------------------------------------- #
# 5. _terminate
# --------------------------------------------------------------------------- #


class _KillFake:
    def __init__(self, raise_on_term=None, die_on_term=False) -> None:
        self.signals: list[int] = []
        self.alive = True
        self.raise_on_term = raise_on_term
        self.die_on_term = die_on_term

    def __call__(self, pid: int, sig: int) -> None:
        self.signals.append(sig)
        if sig == 0:
            if not self.alive:
                raise ProcessLookupError()
            return
        if self.raise_on_term is not None:
            raise self.raise_on_term()
        if sig == signal.SIGTERM and self.die_on_term:
            self.alive = False
        elif sig == signal.SIGKILL:
            self.alive = False


def test_terminate_term_then_gone(monkeypatch):
    fake = _KillFake(die_on_term=True)
    monkeypatch.setattr(uninstall_mod.os, "kill", fake)
    monkeypatch.setattr(uninstall_mod.time, "sleep", lambda *_a: None)
    assert _terminate(1234, "nerditd") is True
    assert signal.SIGKILL not in fake.signals


def test_terminate_ignored_escalates_to_kill(monkeypatch):
    fake = _KillFake(die_on_term=False)
    monkeypatch.setattr(uninstall_mod.os, "kill", fake)
    monkeypatch.setattr(uninstall_mod.time, "sleep", lambda *_a: None)
    assert _terminate(1234, "nerditd", grace_s=0.2) is True
    assert signal.SIGKILL in fake.signals


def test_terminate_process_lookup_is_gone(monkeypatch):
    fake = _KillFake(raise_on_term=ProcessLookupError)
    monkeypatch.setattr(uninstall_mod.os, "kill", fake)
    assert _terminate(1234, "nerditd") is True


def test_terminate_permission_error_returns_false(monkeypatch):
    fake = _KillFake(raise_on_term=PermissionError)
    monkeypatch.setattr(uninstall_mod.os, "kill", fake)
    assert _terminate(1234, "nerditd") is False


def test_permission_denied_daemon_keeps_pid_file(fake_home, tmp_path, monkeypatch):
    # A foreign daemon (PermissionError) must not have its pid file unlinked.
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("777777")
    settings = _settings(data_dir, pid_file=pid_file, upload_dir=tmp_path / "uploads")
    _patch_command(monkeypatch, settings, _FakeClient())

    def _term(pid, label, **_k):
        return False if label == "nerditd" else True

    monkeypatch.setattr(uninstall_mod, "_terminate", _term)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert pid_file.exists()  # not our process — pid file preserved


# --------------------------------------------------------------------------- #
# 6. docker teardown
# --------------------------------------------------------------------------- #


def test_docker_teardown_filters_and_app_images(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir,
        pid_file=data_dir / "nerditd.pid",
        upload_dir=data_dir / "uploads",
        instance_id="beta",
        ollama_image="myreg/ollama:custom",
    )
    container = _FakeContainer("c" * 64, "svc", "running")
    client = _FakeClient(
        containers=[container],
        images=[
            _FakeImage(["nerdit-app/x:1"]),
            _FakeImage(["myreg/ollama:custom"]),
            _FakeImage(["postgres:16"]),
        ],
    )
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert container.removed is True
    # nerdit-app/* removed; base images kept without --purge-images
    assert client.images.removed == ["nerdit-app/x:1"]
    last = client.containers.list_calls[-1]
    assert last["all"] is True
    assert last["filters"] == {"label": ["managed-by=nerdit", "nerdit-instance=beta"]}


def test_docker_teardown_spares_other_instances_app_images(fake_home, tmp_path, monkeypatch):
    """Images carry no instance label: only repos named by THIS instance's DB
    (jobs.service_name) may be removed — another instance's builds survive."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)  # DB owns service "x" only
    settings = _settings(data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "up")
    client = _FakeClient(
        images=[
            _FakeImage(["nerdit-app/x:1"]),
            _FakeImage(["nerdit-app/other-instance-app:3"]),
        ]
    )
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert client.images.removed == ["nerdit-app/x:1"]


def test_owned_repos_ignore_model_and_database_rows(fake_home, tmp_path, monkeypatch):
    """A model/database row named like another instance's APP must not claim
    that app's nerdit-app images (Codex P2): ownership comes from deployed
    rows' image refs, never bare service names."""
    import json

    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    (data_dir / "nerdit.db").unlink()  # replace the fixture DB wholesale
    _write_db(
        data_dir / "nerdit.db",
        ["x"],
        extra_rows=[
            ("jdb", "database", "web", json.dumps({"engine": "postgres"})),
            ("jm", "model", "ollama-web", json.dumps({"model_pulled": True})),
        ],
    )
    settings = _settings(data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "up")
    client = _FakeClient(images=[_FakeImage(["nerdit-app/x:1"]), _FakeImage(["nerdit-app/web:3"])])
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert client.images.removed == ["nerdit-app/x:1"]


def test_terminate_skips_reused_pid(monkeypatch, capsys):
    """A pid file left by a hard crash may point at an unrelated reused pid —
    identity mismatch must skip the kill entirely (Codex P1 follow-up)."""
    monkeypatch.setattr(uninstall_mod, "_process_command", lambda pid: "/usr/bin/vim notes.txt")
    killed: list = []
    monkeypatch.setattr(uninstall_mod.os, "kill", lambda *a: killed.append(a))

    assert _terminate(4242, "nerditd", expect_cmd="nerdit") is True
    assert killed == []
    out = " ".join(capsys.readouterr().out.split())
    assert "stale pid file" in out


def test_terminate_signals_on_identity_match_or_unknown(monkeypatch):
    """Matching command — and an unverifiable one (no ps) — still signal."""
    for probe in ("venv/bin/python -m nerdit.daemon.server", None):
        monkeypatch.setattr(uninstall_mod, "_process_command", lambda pid, _p=probe: _p)
        killed: list = []

        def _kill(pid, sig, _k=killed):
            _k.append(sig)
            raise ProcessLookupError  # gone right after the first signal

        monkeypatch.setattr(uninstall_mod.os, "kill", _kill)
        assert _terminate(4242, "nerditd", expect_cmd="nerdit") is True
        assert killed == [signal.SIGTERM]


def test_manifest_refuses_directory_valued_file_artifacts(fake_home, tmp_path, monkeypatch, capsys):
    """A pid_file/key_file typo pointing at an existing DIRECTORY must be
    ignored (warned), never enumerated for recursive removal (Codex P2)."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    bad_pid_dir = tmp_path / "run-dir"
    bad_pid_dir.mkdir()
    bad_key_dir = tmp_path / "keys-dir"
    bad_key_dir.mkdir()
    (bad_key_dir / "unrelated.txt").write_text("keep me")
    settings = _settings(
        data_dir, pid_file=bad_pid_dir, upload_dir=data_dir / "uploads", key_file=bad_key_dir
    )
    manifest = build_manifest(
        settings, purge_images=False, docker_client_factory=_factory(_FakeClient())
    )
    assert all(ep.path not in (bad_pid_dir, bad_key_dir) for ep in manifest.extra_paths)
    out = " ".join(capsys.readouterr().out.split())
    assert "not a regular file" in out


def test_manifest_skips_sizing_for_unsafe_data_dir(fake_home, monkeypatch):
    """data_dir = '/' must not be walked for byte-sizing before the fail-closed
    guard refuses it (Codex P2)."""
    settings = _settings(Path("/"), pid_file=Path("/nonexistent.pid"), upload_dir=Path("/nx-up"))
    monkeypatch.setattr(
        uninstall_mod, "du_bytes", lambda p: (_ for _ in ()).throw(AssertionError("walked!"))
    )
    manifest = build_manifest(
        settings, purge_images=False, docker_client_factory=_factory(_FakeClient())
    )
    assert manifest.data_dir_bytes == 0


def test_read_pid_rejects_nonpositive(tmp_path):
    """'0' signals the whole process group and '-1' every permitted process —
    a corrupt pid file must never reach os.kill (Codex P1)."""
    p = tmp_path / "x.pid"
    for bad in ("0", "-1", "-12345"):
        p.write_text(bad)
        assert uninstall_mod._read_pid(p) is None
    p.write_text("4242")
    assert uninstall_mod._read_pid(p) == 4242


def test_docker_teardown_unreadable_db_removes_no_app_images(
    fake_home, tmp_path, monkeypatch, capsys
):
    """A corrupt DB means app images cannot be attributed to this instance —
    remove none, print the manual command instead of guessing."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    (data_dir / "nerdit.db").write_text("not a database")
    settings = _settings(data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "up")
    client = _FakeClient(images=[_FakeImage(["nerdit-app/x:1"])])
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert client.images.removed == []
    out = " ".join(capsys.readouterr().out.split())
    assert "App images were NOT removed" in out
    assert "docker images --format" in out
    assert not data_dir.exists()  # the rest of the uninstall still ran


def test_docker_teardown_purge_images_honors_configured_ollama(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir,
        pid_file=data_dir / "nerditd.pid",
        upload_dir=data_dir / "uploads",
        ollama_image="myreg/ollama:custom",
    )
    client = _FakeClient(
        images=[
            _FakeImage(["nerdit-app/x:1"]),
            _FakeImage(["myreg/ollama:custom"]),
            _FakeImage(["ollama/ollama"]),  # the DEFAULT — must NOT be targeted
        ]
    )
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=True, keep_data=False)

    assert "myreg/ollama:custom" in client.images.removed
    assert "ollama/ollama" not in client.images.removed  # default not the configured value


def test_docker_down_prints_manual_and_still_deletes(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir,
        pid_file=data_dir / "nerditd.pid",
        upload_dir=data_dir / "uploads",
        instance_id="gamma",
    )
    monkeypatch.setattr(uninstall_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(uninstall_mod, "_default_docker_client", _down_factory())
    monkeypatch.setattr(uninstall_mod.time, "sleep", lambda *_a, **_k: None)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    out = capsys.readouterr().out
    assert "docker ps -aq" in out
    assert "nerdit-instance=gamma" in out
    assert not data_dir.exists()  # files still deleted despite docker down


# --------------------------------------------------------------------------- #
# 7. live-daemon guard
# --------------------------------------------------------------------------- #


@posix_only
def test_live_daemon_guard_refuses(fake_home, tmp_path, monkeypatch, capsys):
    import fcntl

    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    # Simulate a live daemon holding LOCK_SH on .restore.lock.
    held = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_SH)

    settings = _settings(
        data_dir, pid_file=data_dir / "nonexistent.pid", upload_dir=data_dir / "uploads"
    )
    container = _FakeContainer("c1" * 6, "nerdit-app", "running")
    client = _FakeClient(containers=[container], images=[_FakeImage(["nerdit-app/x:1"])])
    _patch_command(monkeypatch, settings, client)
    terminated: list[str] = []
    monkeypatch.setattr(
        uninstall_mod, "_terminate", lambda pid, label, **_k: terminated.append(label) or True
    )
    try:
        with pytest.raises(typer.Exit) as exc:
            uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
        assert exc.value.exit_code == 1
        assert data_dir.exists()
        assert (data_dir / "nerdit.db").exists()  # tree intact
        # The guard runs BEFORE the Caddy reap and the docker teardown, so it
        # is a true all-or-nothing gate: workloads untouched, promise accurate.
        assert "caddy" not in terminated
        assert (data_dir / "caddy.pid").exists()
        assert container.removed is False
        assert client.images.removed == []
        out = " ".join(capsys.readouterr().out.split())  # collapse rich wrapping
        assert "nothing was removed" in out
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


# --------------------------------------------------------------------------- #
# 8. --keep-data
# --------------------------------------------------------------------------- #


def test_keep_data_preserves_data_key_certs(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("111")
    (pid_file.parent / "nerditd.boot.log").write_text("b")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "a").write_text("x")
    key_file = tmp_path / "keys" / "secrets.key"
    key_file.parent.mkdir()
    key_file.write_text("K")
    cert = _save_ca_cert(fake_home)

    settings = _settings(data_dir, pid_file=pid_file, upload_dir=upload_dir, key_file=key_file)
    container = _FakeContainer("d" * 64, "svc", "running")
    client = _FakeClient(containers=[container])
    _patch_command(monkeypatch, settings, client)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=True)

    assert data_dir.exists()  # preserved
    assert (data_dir / "nerdit.db").exists()
    assert key_file.exists()  # kept (data useless without it)
    assert cert.exists()  # trust artifacts kept
    assert not pid_file.exists()  # pid/boot/uploads still cleaned (outside data_dir)
    assert not (pid_file.parent / "nerditd.boot.log").exists()
    assert not upload_dir.exists()
    assert container.removed is True  # docker teardown still ran


# --------------------------------------------------------------------------- #
# 9. key override outside data_dir
# --------------------------------------------------------------------------- #


def test_key_override_outside_deleted_in_normal_mode(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    key_file = tmp_path / "keys" / "secrets.key"
    key_file.parent.mkdir()
    key_file.write_text("K")
    settings = _settings(
        data_dir,
        pid_file=data_dir / "nerditd.pid",
        upload_dir=data_dir / "uploads",
        key_file=key_file,
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not key_file.exists()


# --------------------------------------------------------------------------- #
# 10. archive dir outside data_dir
# --------------------------------------------------------------------------- #


def test_archive_dir_outside_left_in_place(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    archive_dir = tmp_path / "audit-archive"
    archive_dir.mkdir()
    (archive_dir / "2026.jsonl.gz").write_text("audit")
    settings = _settings(
        data_dir,
        pid_file=data_dir / "nerditd.pid",
        upload_dir=data_dir / "uploads",
        archive_dir=archive_dir,
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert archive_dir.exists()
    assert (archive_dir / "2026.jsonl.gz").exists()
    assert "Left in place" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 11. empty ~/.nerdit rmdir'd
# --------------------------------------------------------------------------- #


def test_empty_home_nerdit_rmdir(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"  # custom data_dir, NOT ~/.nerdit
    _populate_data_dir(data_dir)
    cert = _save_ca_cert(fake_home)  # sole occupant of ~/.nerdit
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not cert.exists()
    assert not (fake_home / ".nerdit").exists()  # emptied -> rmdir'd


def test_nonempty_home_nerdit_left(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    _save_ca_cert(fake_home)
    # A genuinely foreign file (not a Nerdit artifact) keeps ~/.nerdit non-empty.
    (fake_home / ".nerdit" / "unrelated.txt").write_text("mine")
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert (fake_home / ".nerdit").exists()  # foreign file keeps it non-empty
    assert (fake_home / ".nerdit" / "unrelated.txt").exists()


def test_config_toml_deleted_and_home_rmdird_custom_data_dir(fake_home, tmp_path, monkeypatch):
    # Finding 1: a FIXED ~/.nerdit/config.toml orphaned by a custom data_dir is
    # enumerated + deleted in normal mode, letting the empty ~/.nerdit rmdir.
    data_dir = tmp_path / "data"  # custom data_dir, NOT ~/.nerdit
    _populate_data_dir(data_dir)
    config_toml = fake_home / ".nerdit" / "config.toml"
    config_toml.parent.mkdir(parents=True, exist_ok=True)
    config_toml.write_text("[nerdit]\ndata_dir = '%s'\n" % data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    manifest = build_manifest(
        settings, purge_images=False, docker_client_factory=_factory(_FakeClient())
    )
    assert any(ep.kind == "config" for ep in manifest.extra_paths)

    _patch_command(monkeypatch, settings, _FakeClient())
    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not config_toml.exists()
    assert not (fake_home / ".nerdit").exists()  # emptied -> rmdir'd


def test_config_toml_kept_under_keep_data(fake_home, tmp_path, monkeypatch):
    # Finding 1: --keep-data preserves the config (so the daemon can restart).
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    config_toml = fake_home / ".nerdit" / "config.toml"
    config_toml.parent.mkdir(parents=True, exist_ok=True)
    config_toml.write_text("[nerdit]\n")
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=True)
    assert config_toml.exists()  # kept alongside the data dir


# --------------------------------------------------------------------------- #
# 2b. data_dir sanity guard (Finding 2, major)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [".", "relative/data"])
def test_guard_rejects_nonabsolute_data_dir(fake_home, tmp_path, monkeypatch, bad):
    # An empty/relative data_dir would rmtree the CWD — refuse, delete nothing.
    real = tmp_path / "cwd-guard-canary"
    real.mkdir()
    (real / "keepme.txt").write_text("do not delete")
    monkeypatch.chdir(real)

    settings = _settings(
        Path(bad), pid_file=tmp_path / "nerditd.pid", upload_dir=tmp_path / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    monkeypatch.setattr(typer, "prompt", lambda *_a, **_k: "uninstall")

    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=False, dry_run=False, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1
    assert (real / "keepme.txt").exists()  # CWD untouched


def test_guard_rejects_home_data_dir(fake_home, tmp_path, monkeypatch):
    settings = _settings(
        fake_home, pid_file=tmp_path / "nerditd.pid", upload_dir=tmp_path / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1
    assert fake_home.exists()


def test_guard_rejects_bad_data_dir_even_on_dry_run(fake_home, tmp_path, monkeypatch):
    # Fail closed even for --dry-run — never present a "." target as safe.
    settings = _settings(
        Path("."), pid_file=tmp_path / "nerditd.pid", upload_dir=tmp_path / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=True, dry_run=True, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1


# --------------------------------------------------------------------------- #
# 13. CA hint uses the shared _untrust_commands (imported, not duplicated)
# --------------------------------------------------------------------------- #


def test_ca_hint_uses_shared_untrust_commands(fake_home, tmp_path, monkeypatch, capsys):
    # Identity: uninstall imports the very same function object from trust.py.
    assert uninstall_mod._untrust_commands is trust_mod._untrust_commands

    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    _save_ca_cert(fake_home)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    sentinel = [["SENTINEL-UNTRUST-CMD"]]
    monkeypatch.setattr(uninstall_mod, "_untrust_commands", lambda cert: sentinel)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    out = capsys.readouterr().out
    assert "SENTINEL-UNTRUST-CMD" in out
    assert "nerdit untrust" in out


def test_no_certs_prints_nothing_to_untrust(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert "nothing to untrust" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# smoke: registered + argv reachable
# --------------------------------------------------------------------------- #


def test_cli_smoke_dry_run(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    monkeypatch.setattr(uninstall_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(uninstall_mod, "_default_docker_client", lambda: _FakeClient())

    result = CliRunner().invoke(app, ["uninstall", "--dry-run", "--yes"])
    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert data_dir.exists()


def test_manifest_docker_down_still_builds(fake_home, tmp_path):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    manifest = build_manifest(settings, purge_images=False, docker_client_factory=_down_factory())
    assert isinstance(manifest, UninstallManifest)
    assert manifest.docker_error is not None
    assert manifest.containers == []


# --------------------------------------------------------------------------- #
# 14. installer-made layout (P30 D-P30-8/9)
# --------------------------------------------------------------------------- #


class _FakeRun:
    """Records every service-manager command and replays a canned returncode."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return SimpleNamespace(returncode=self.returncode, stderr=self.stderr, stdout="")


def _unit(tmp_path: Path, kind: str = "systemd-system") -> ServiceUnit:
    unit_path = tmp_path / "units" / "nerdit.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("[Unit]\n")
    user = ["--user"] if kind == "systemd-user" else []
    return ServiceUnit(
        kind=kind,  # type: ignore[arg-type]
        unit_path=unit_path,
        restart_argv=["systemctl", *user, "restart", "nerdit.service"],
        stop_argv=["systemctl", *user, "stop", "nerdit.service"],
        disable_argv=["systemctl", *user, "disable", "nerdit.service"],
    )


def _install_layout(home: Path, *, mode: str = "user") -> InstallLayout:
    root = home / ".nerdit-install"
    versions = root / "versions" / "0.5.0"
    versions.mkdir(parents=True)
    (versions / "nerdit").write_text("binary")
    (versions / "install.sh").write_text("#!/bin/sh\n")
    (root / "current").symlink_to(Path("versions") / "0.5.0")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    (bin_dir / "nerdit").symlink_to(Path("..") / "current" / "nerdit")
    return InstallLayout(
        mode=mode,  # type: ignore[arg-type]
        root=root,
        versions_dir=root / "versions",
        current=root / "current",
        current_version="0.5.0",
        shim=bin_dir / "nerdit",
        installer=root / "current" / "install.sh",
    )


def test_unit_is_stopped_before_the_pid_kill(fake_home, tmp_path, monkeypatch):
    """A unit-managed daemon that is only pid-killed is respawned within
    seconds — the unit teardown MUST come first."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("999999")
    settings = _settings(data_dir, pid_file=pid_file, upload_dir=tmp_path / "uploads")
    _patch_command(monkeypatch, settings, _FakeClient())

    unit = _unit(tmp_path)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)

    order: list[str] = []
    run = _FakeRun()

    def _record_run(argv, **kwargs):
        order.append("unit:" + " ".join(argv))
        return run(argv, **kwargs)

    monkeypatch.setattr("subprocess.run", _record_run)
    monkeypatch.setattr(
        uninstall_mod,
        "_terminate",
        lambda pid, label, **_k: order.append(f"kill:{label}") is None,
    )

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert order[0].startswith("unit:systemctl stop")
    assert "kill:nerditd" in order
    assert order.index("kill:nerditd") > 0


def test_unit_stopped_disabled_and_removed(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    unit = _unit(tmp_path, kind="systemd-user")
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    run = _FakeRun()
    monkeypatch.setattr("subprocess.run", run)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    systemctl = [c for c in run.calls if c[0] == "systemctl"]
    assert systemctl == [
        ["systemctl", "--user", "stop", "nerdit.service"],
        ["systemctl", "--user", "disable", "nerdit.service"],
        ["systemctl", "--user", "daemon-reload"],
    ]
    assert not unit.unit_path.exists()


def test_launchd_bootout_is_not_run_twice(fake_home, tmp_path, monkeypatch):
    """launchd's bootout IS the disable — running it twice just errors."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    plist = tmp_path / "ai.nerdit.daemon.plist"
    plist.write_text("<plist/>")
    bootout = ["launchctl", "bootout", "gui/501/ai.nerdit.daemon"]
    unit = ServiceUnit(
        kind="launchd",
        unit_path=plist,
        restart_argv=["launchctl", "kickstart", "-k", "gui/501/ai.nerdit.daemon"],
        stop_argv=list(bootout),
        disable_argv=list(bootout),
    )
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    run = _FakeRun()
    monkeypatch.setattr("subprocess.run", run)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    # (other subprocess.run calls in this teardown are the `ps` identity probes)
    assert [c for c in run.calls if c[0] == "launchctl"] == [bootout]
    assert not plist.exists()


def test_failing_unit_command_does_not_abort_the_uninstall(
    fake_home, tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    unit = _unit(tmp_path)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    monkeypatch.setattr("subprocess.run", _FakeRun(returncode=5, stderr="Unit not loaded.\n"))

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    out = " ".join(capsys.readouterr().out.split())
    assert "Unit not loaded." in out
    assert not data_dir.exists()  # the teardown still completed
    assert not unit.unit_path.exists()


def test_system_layout_refuses_without_root(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    layout = _install_layout(fake_home, mode="system")
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(uninstall_mod.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: pytest.fail("must refuse before any teardown")
    )

    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1
    assert data_dir.exists()  # nothing torn down
    assert layout.versions_dir.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "system install" in out
    assert "sudo" in out


class TestUnitHomeAdoption:  # noqa: D101 — docstring below
    @pytest.fixture(autouse=True)
    def _use_the_real_implementation(self, monkeypatch):
        """Undo the module-level autouse patch for this class only."""
        monkeypatch.setattr(uninstall_mod, "_adopt_unit_home", _REAL_ADOPT_UNIT_HOME)

    """(P30) `sudo nerdit uninstall` on a system install must resolve the data
    dir from the unit's recorded HOME, not from root's.

    Without it the command removes the unit, /opt/nerdit, the shim, every
    container and image — and reports success — while the secrets master key,
    every `.enc` envelope, the internal-CA private key and node.key survive
    under the unit user's home.
    """

    @staticmethod
    def _unit(tmp_path: Path, home: str) -> Path:
        unit = tmp_path / "nerdit.service"
        unit.write_text(
            "[Service]\nUser=alice\n"
            f"Environment=HOME={home}\nEnvironment=NERDIT_BOOT_LOG=1\n"
            "ExecStart=/opt/nerdit/current/nerditd\n"
        )
        return unit

    @pytest.mark.skipif(sys.platform == "darwin", reason="systemd system units are Linux-only")
    def test_adopts_the_recorded_home(self, tmp_path, monkeypatch):
        unit_home = tmp_path / "home" / "alice"
        unit_home.mkdir(parents=True)
        monkeypatch.setattr(
            uninstall_mod, "_SYSTEM_UNIT_PATH", self._unit(tmp_path, str(unit_home))
        )
        monkeypatch.setenv("HOME", str(tmp_path / "root"))

        assert uninstall_mod._adopt_unit_home() == str(unit_home)
        assert os.environ["HOME"] == str(unit_home)
        assert Path.home() == unit_home

    @pytest.mark.skipif(sys.platform == "darwin", reason="systemd system units are Linux-only")
    def test_no_op_when_the_home_already_matches(self, tmp_path, monkeypatch):
        unit_home = tmp_path / "home" / "alice"
        unit_home.mkdir(parents=True)
        monkeypatch.setattr(
            uninstall_mod, "_SYSTEM_UNIT_PATH", self._unit(tmp_path, str(unit_home))
        )
        monkeypatch.setenv("HOME", str(unit_home))

        assert uninstall_mod._adopt_unit_home() is None
        assert os.environ["HOME"] == str(unit_home)

    def test_no_unit_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uninstall_mod, "_SYSTEM_UNIT_PATH", tmp_path / "absent.service")
        monkeypatch.setenv("HOME", str(tmp_path / "root"))

        assert uninstall_mod._adopt_unit_home() is None
        assert os.environ["HOME"] == str(tmp_path / "root")

    @pytest.mark.skipif(sys.platform == "darwin", reason="systemd system units are Linux-only")
    def test_a_relative_home_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr(uninstall_mod, "_SYSTEM_UNIT_PATH", self._unit(tmp_path, "relative"))
        monkeypatch.setenv("HOME", str(tmp_path / "root"))

        assert uninstall_mod._adopt_unit_home() is None
        assert os.environ["HOME"] == str(tmp_path / "root")


def test_layout_paths_appear_in_the_dry_run_manifest(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    layout = _install_layout(fake_home)
    unit = _unit(tmp_path)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)

    uninstall(yes=True, dry_run=True, purge_images=False, keep_data=False)

    # Rich hard-wraps at 80 columns off a tty, so long tmp paths are asserted
    # against the whitespace-stripped render (no path here contains a space).
    out = "".join(capsys.readouterr().out.split())
    assert str(unit.unit_path) in out
    assert str(layout.versions_dir) in out
    assert str(layout.current) in out
    assert str(layout.shim) in out
    assert "0.5.0" in out
    # Dry run: everything still there.
    assert layout.versions_dir.exists()
    assert layout.shim.is_symlink()
    assert unit.unit_path.exists()


def test_layout_code_is_removed(fake_home, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    layout = _install_layout(fake_home)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    assert not layout.versions_dir.exists()
    assert not layout.current.is_symlink()
    assert not layout.shim.is_symlink()
    assert not layout.shim.parent.exists()  # the emptied bin/ goes too


def test_layout_code_is_removed_even_with_keep_data(fake_home, tmp_path, monkeypatch):
    """Code is never "data": --keep-data preserves the DB and the key, not the
    binaries a `nerdit` on PATH would still point at."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    layout = _install_layout(fake_home)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=True)

    assert data_dir.exists()
    assert (data_dir / "nerdit.db").exists()
    assert not layout.versions_dir.exists()
    assert not layout.current.is_symlink()


def test_pip_hint_only_without_a_layout(fake_home, tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert "pip uninstall nerdit" in " ".join(capsys.readouterr().out.split())

    data_dir2 = tmp_path / "data2"
    _populate_data_dir(data_dir2)
    settings2 = _settings(
        data_dir2, pid_file=data_dir2 / "nerditd.pid", upload_dir=data_dir2 / "uploads"
    )
    _patch_command(monkeypatch, settings2, _FakeClient())
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: _install_layout(fake_home))

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert "pip uninstall nerdit" not in " ".join(capsys.readouterr().out.split())


@posix_only
def test_live_daemon_guard_leaves_the_unit_file_in_place(fake_home, tmp_path, monkeypatch, capsys):
    """(2026-08-23) The unit file is NOT removed before the flock check.

    This test previously asserted the opposite — the unit was stopped *and
    removed* first, and the refusal said so. On a real box the stop failed, the
    unit file was deleted anyway, and the command then refused with "nothing was
    removed" and "the service unit was stopped and removed", both false: it had
    orphaned a live nerditd with no unit to stop it with. The unit file must
    survive a refusal, and the message must enumerate what was actually tried.
    """
    import fcntl

    monkeypatch.setattr(uninstall_mod, "_DAEMON_DRAIN_S", 0)
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    held = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_SH)

    settings = _settings(
        data_dir, pid_file=data_dir / "nonexistent.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    unit = _unit(tmp_path)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    run = _FakeRun()
    monkeypatch.setattr("subprocess.run", run)

    try:
        with pytest.raises(typer.Exit) as exc:
            uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
        assert exc.value.exit_code == 1
        assert data_dir.exists()
        assert (data_dir / "nerdit.db").exists()  # data untouched
        assert unit.unit_path.exists()  # and so is the unit — nothing was removed
        # Only the STOP ran; disable/daemon-reload belong to the removal phase.
        assert [c for c in run.calls if c[0] == "systemctl"] == [
            ["systemctl", "stop", "nerdit.service"]
        ]
        out = " ".join(capsys.readouterr().out.split())
        assert "nothing was removed" in out
        assert "service unit was stopped and removed" not in out
        assert "service manager: 'systemctl stop nerdit.service' succeeded" in out
        assert "no daemon pid file to signal" in out
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


def test_failed_unit_stop_is_recorded_and_the_signal_path_takes_over(
    fake_home, tmp_path, monkeypatch, capsys
):
    """systemctl stop fails, the pid signal works ⇒ the teardown proceeds.

    The failure is *recorded and printed*, never reported as a success, and the
    unit file is still removed — because by then the data dir really is free.
    """
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("424242")
    settings = _settings(data_dir, pid_file=pid_file, upload_dir=tmp_path / "uploads")
    _patch_command(monkeypatch, settings, _FakeClient())

    unit = _unit(tmp_path)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    monkeypatch.setattr("subprocess.run", _FakeRun(returncode=5, stderr="Job failed.\n"))
    signalled: list[tuple[int, float]] = []

    def _term(pid, label, *, grace_s=5.0, expect_cmd=None):
        signalled.append((pid, grace_s))
        return True

    monkeypatch.setattr(uninstall_mod, "_terminate", _term)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)

    # The daemon pid was signalled even though the service manager "handled" it,
    # with the drain budget (not the 5 s Caddy default).
    assert (424242, uninstall_mod._DAEMON_DRAIN_S) in signalled
    assert not data_dir.exists()
    assert not unit.unit_path.exists()
    assert not pid_file.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "Job failed." in out
    assert "Service unit stopped" not in out  # it did not stop; never claim it did


@posix_only
def test_both_stops_fail_aborts_before_any_removal(fake_home, tmp_path, monkeypatch, capsys):
    """Service manager fails AND the daemon survives the signal ⇒ abort, truthfully.

    Nothing at all may be gone: not the unit file, not the install tree, not the
    containers, not the pid file, not a byte of data.
    """
    import fcntl

    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    held = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_SH)

    pid_file = tmp_path / "run" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("515151")
    settings = _settings(data_dir, pid_file=pid_file, upload_dir=tmp_path / "uploads")
    container = _FakeContainer("c1" * 6, "nerdit-app", "running")
    client = _FakeClient(containers=[container], images=[_FakeImage(["nerdit-app/x:1"])])
    _patch_command(monkeypatch, settings, client)

    unit = _unit(tmp_path)
    layout = _install_layout(fake_home)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: unit)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr("subprocess.run", _FakeRun(returncode=1, stderr="Failed to stop.\n"))
    monkeypatch.setattr(uninstall_mod, "_terminate", lambda pid, label, **_k: False)

    try:
        with pytest.raises(typer.Exit) as exc:
            uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
        assert exc.value.exit_code == 1
        assert (data_dir / "nerdit.db").exists()
        assert (data_dir / "caddy.pid").exists()
        assert unit.unit_path.exists()
        assert layout.versions_dir.exists()
        assert layout.shim.is_symlink()
        assert pid_file.exists()
        assert container.removed is False
        assert client.images.removed == []
        out = " ".join(capsys.readouterr().out.split())
        assert "nothing was removed" in out
        assert "Failed to stop." in out
        assert "'systemctl stop nerdit.service' failed" in out
        assert "pid 515151 signalled (SIGTERM, then SIGKILL); it did NOT exit" in out
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


def test_system_layout_refuses_before_the_typed_confirmation(
    fake_home, tmp_path, monkeypatch, capsys
):
    """The privilege guard runs BEFORE the prompt — never ask for a word we
    cannot honour."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    layout = _install_layout(fake_home, mode="system")
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(uninstall_mod.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        typer, "prompt", lambda *a, **k: pytest.fail("must refuse before the confirmation")
    )

    with pytest.raises(typer.Exit) as exc:
        uninstall(yes=False, dry_run=False, purge_images=False, keep_data=False)
    assert exc.value.exit_code == 1
    assert data_dir.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "system install" in out
    assert "Re-run with sudo" in out


def test_dangling_current_symlink_is_still_removed(fake_home, tmp_path, monkeypatch):
    """exists() is False for a dangling link — it must still be unlinked."""
    data_dir = tmp_path / "data"
    _populate_data_dir(data_dir)
    settings = _settings(
        data_dir, pid_file=data_dir / "nerditd.pid", upload_dir=data_dir / "uploads"
    )
    _patch_command(monkeypatch, settings, _FakeClient())
    root = fake_home / ".nerdit-install"
    root.mkdir()
    (root / "current").symlink_to(Path("versions") / "9.9.9")  # never created
    layout = InstallLayout(
        mode="user",
        root=root,
        versions_dir=root / "versions",
        current=root / "current",
        current_version=None,
        shim=root / "bin" / "nerdit",
        installer=root / "current" / "install.sh",
    )
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: layout)

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not layout.current.is_symlink()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))


class TestAdoptedConfigCannotDirectRootDeletion:
    """Security review (P30): the adopted-home config is attacker-writable.

    `_adopt_unit_home` points $HOME at the UNPRIVILEGED unit user's home so the
    data dir resolves there — but that makes root parse a config.toml which that
    user, and the daemon running as them, can write. `[nerdit].data_dir` is a
    writable config-as-API key, so a daemon admin token was enough to aim a root
    `shutil.rmtree` at an arbitrary path: the old guard rejected only
    root/$HOME/CWD, so `/etc` passed.
    """

    def _adopt(self, monkeypatch, home):
        from nerdit.cli.commands import uninstall as u

        monkeypatch.setattr(u, "_ADOPTED_HOME", Path(home))
        monkeypatch.setattr(u, "_INVOKING_HOME", Path("/root"))

    @pytest.mark.parametrize("target", ["/etc", "/usr", "/var", "/boot", "/opt"])
    def test_absolute_paths_outside_the_adopted_home_are_refused(
        self, monkeypatch, tmp_path, target
    ):
        from nerdit.cli.commands.uninstall import _unsafe_target

        self._adopt(monkeypatch, tmp_path)
        err = _unsafe_target("data dir", Path(target))
        assert err is not None, f"{target} must not be removable as root"
        # On macOS tmp_path sits under /private/var, so /var is also refused
        # as an ancestor of the adopted home — first reason wins.
        assert "outside the service user's home" in err or "contains one" in err

    def test_the_adopted_home_itself_is_still_usable(self, monkeypatch, tmp_path):
        from nerdit.cli.commands.uninstall import _unsafe_target

        self._adopt(monkeypatch, tmp_path)
        assert _unsafe_target("data dir", tmp_path / ".nerdit") is None

    def test_both_homes_are_protected_not_just_the_adopted_one(self, monkeypatch, tmp_path):
        """After adoption Path.home() IS the adopted home, so a guard written
        against it stopped protecting the invoking user's."""
        from nerdit.cli.commands import uninstall as u

        invoking = tmp_path / "root"
        invoking.mkdir()
        monkeypatch.setattr(u, "_ADOPTED_HOME", tmp_path)
        monkeypatch.setattr(u, "_INVOKING_HOME", invoking)
        assert u._unsafe_target("data dir", invoking) is not None
        assert u._unsafe_target("data dir", tmp_path) is not None

    def test_traversal_out_of_the_adopted_home_is_refused(self, monkeypatch, tmp_path):
        from nerdit.cli.commands.uninstall import _unsafe_target

        self._adopt(monkeypatch, tmp_path)
        assert _unsafe_target("data dir", tmp_path / ".." / ".." / "etc") is not None

    def test_config_owned_by_a_third_party_is_distrusted(self, monkeypatch, tmp_path):
        from nerdit.cli.commands.uninstall import _adopted_config_is_trustworthy

        cfg = tmp_path / ".nerdit"
        cfg.mkdir()
        (cfg / "config.toml").write_text("[nerdit]\n")
        real = os.stat

        # The home belongs to uid 1000; the config was planted by uid 1001.
        def fake_stat(path, *a, **kw):
            st = real(path, *a, **kw)
            uid = 1001 if str(path).endswith("config.toml") else 1000
            return os.stat_result(
                (
                    st.st_mode,
                    st.st_ino,
                    st.st_dev,
                    st.st_nlink,
                    uid,
                    st.st_gid,
                    st.st_size,
                    int(st.st_atime),
                    int(st.st_mtime),
                    int(st.st_ctime),
                )
            )

        monkeypatch.setattr(os, "stat", fake_stat)
        assert _adopted_config_is_trustworthy(str(tmp_path)) is False

    def test_config_owned_by_the_unit_user_is_accepted(self, tmp_path):
        from nerdit.cli.commands.uninstall import _adopted_config_is_trustworthy

        cfg = tmp_path / ".nerdit"
        cfg.mkdir()
        (cfg / "config.toml").write_text("[nerdit]\n")
        assert _adopted_config_is_trustworthy(str(tmp_path)) is True

    def test_a_missing_config_is_not_distrusted(self, tmp_path):
        """No config means defaults are used — nothing to interpose on."""
        from nerdit.cli.commands.uninstall import _adopted_config_is_trustworthy

        assert _adopted_config_is_trustworthy(str(tmp_path)) is True


def test_service_stop_drain_is_bounded(tmp_path, monkeypatch):
    unit = _unit(tmp_path)
    alive = uninstall_mod.DaemonLiveness(True, None, "data_dir_lock")
    monkeypatch.setattr(uninstall_mod, "_run_unit_command", lambda *a, **kw: (True, None))
    monkeypatch.setattr(uninstall_mod, "probe_daemon", lambda **kw: alive)
    times = iter([0, 0, uninstall_mod._DAEMON_DRAIN_S])
    monkeypatch.setattr(uninstall_mod.time, "monotonic", lambda: next(times))
    sleeps = []
    monkeypatch.setattr(uninstall_mod.time, "sleep", sleeps.append)

    ok, detail = uninstall_mod._stop_unit(unit, data_dir=tmp_path, pid_file=tmp_path / "absent.pid")

    assert not ok
    assert "did not stop within" in detail
    assert sleeps == [0.2]
    assert unit.unit_path.exists()


def test_unsafe_target_refuses_ancestors_of_home_and_cwd(monkeypatch, tmp_path):
    from nerdit.cli.commands.uninstall import _unsafe_target

    assert _unsafe_target("data dir", Path.home().parent) is not None
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "sibling").mkdir()
    monkeypatch.chdir(tmp_path / "a" / "b")
    assert _unsafe_target("data dir", tmp_path / "a") is not None
    assert _unsafe_target("data dir", tmp_path / "sibling") is None
