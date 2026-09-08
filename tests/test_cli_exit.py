"""Require exit and uninstall to agree on whether the daemon is running.

Exercise their shared probe_daemon with real temporary files and patched settings:
use the configured PID path and lifetime restore flock, not a hard-coded PID file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer

from nerdit.cli.commands import exit as exit_mod
from nerdit.cli.commands import uninstall as uninstall_mod
from nerdit.cli.commands.exit import exit_daemon
from nerdit.cli.commands.uninstall import DaemonLiveness, probe_daemon, uninstall
from nerdit.config.settings import NerditSettings
from nerdit.utils.install_layout import ServiceUnit

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX flock guard")


@pytest.fixture(autouse=True)
def _no_real_install(monkeypatch):
    """A real LaunchAgent / systemd unit on the developer's box must not leak in."""
    monkeypatch.setattr(exit_mod, "detect_service_unit", lambda: None)
    monkeypatch.setattr(uninstall_mod, "detect_service_unit", lambda: None)
    monkeypatch.setattr(uninstall_mod, "detect_install_layout", lambda: None)
    monkeypatch.setattr(uninstall_mod, "_adopt_unit_home", lambda: None)
    monkeypatch.setattr(uninstall_mod, "_ADOPTED_HOME", None)
    monkeypatch.setattr(uninstall_mod, "_INVOKING_HOME", None)
    monkeypatch.setattr(uninstall_mod.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


class _FakeDocker:
    """Docker is irrelevant here — an unreachable client keeps it out of the way."""

    def __init__(self) -> None:
        raise RuntimeError("Cannot connect to the Docker daemon")


def _settings(data_dir: Path, pid_file: Path) -> NerditSettings:
    return NerditSettings(
        data_dir=str(data_dir),
        daemon={"pid_file": str(pid_file), "upload_dir": str(data_dir / "uploads"), "port": 59997},
    )


def _wire(monkeypatch, settings: NerditSettings) -> None:
    monkeypatch.setattr(exit_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(uninstall_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(uninstall_mod, "_default_docker_client", _FakeDocker)


def _unit(tmp_path: Path) -> ServiceUnit:
    unit_path = tmp_path / "units" / "nerdit.service"
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text("[Unit]\n")
    return ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=["systemctl", "restart", "nerdit.service"],
        stop_argv=["systemctl", "stop", "nerdit.service"],
        disable_argv=["systemctl", "disable", "nerdit.service"],
    )


# --------------------------------------------------------------------------- #
# 1. the two commands agree
# --------------------------------------------------------------------------- #


@posix_only
def test_exit_and_uninstall_agree_when_a_daemon_holds_the_data_dir_lock(
    fake_home, tmp_path, monkeypatch, capsys
):
    """The exact field disagreement: a daemon whose pid file is gone.

    ``uninstall`` sees it (the flock) and refuses; ``exit`` used to look only at
    a hard-coded pid file and say "Daemon is not running". Both must now see it.
    """
    import fcntl

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nerdit.db").write_text("db")
    held = os.open(str(data_dir / ".restore.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_SH)

    pid_file = tmp_path / "run" / "nerditd.pid"  # deliberately absent
    settings = _settings(data_dir, pid_file)
    _wire(monkeypatch, settings)

    try:
        assert probe_daemon(data_dir=data_dir, pid_file=pid_file).alive is True

        with pytest.raises(typer.Exit) as exc:
            exit_daemon(yes=True)
        assert exc.value.exit_code == 1
        exit_out = " ".join(capsys.readouterr().out.split())
        assert "Daemon is not running" not in exit_out
        assert "holds the data dir lock" in exit_out

        with pytest.raises(typer.Exit) as exc:
            uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
        assert exc.value.exit_code == 1
        assert " ".join(capsys.readouterr().out.split()).count("nothing was removed") == 1
        assert (data_dir / "nerdit.db").exists()
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


@posix_only
def test_exit_and_uninstall_agree_that_a_free_data_dir_has_no_daemon(
    fake_home, tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".restore.lock").write_text("")
    pid_file = tmp_path / "run" / "nerditd.pid"
    settings = _settings(data_dir, pid_file)
    _wire(monkeypatch, settings)

    assert probe_daemon(data_dir=data_dir, pid_file=pid_file).alive is False

    exit_daemon(yes=True)
    assert "Daemon is not running" in " ".join(capsys.readouterr().out.split())

    uninstall(yes=True, dry_run=False, purge_images=False, keep_data=False)
    assert not data_dir.exists()  # the same state let the uninstall through


# --------------------------------------------------------------------------- #
# 2. probe_daemon itself
# --------------------------------------------------------------------------- #


def test_probe_uses_the_configured_pid_file_not_a_hard_coded_one(fake_home, tmp_path, monkeypatch):
    """The old ``exit`` hard-coded ``~/.nerdit/nerditd.pid``; a daemon with a
    relocated pid file was therefore invisible to it and visible to uninstall."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pid_file = tmp_path / "elsewhere" / "nerditd.pid"
    pid_file.parent.mkdir()
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(uninstall_mod, "_process_command", lambda pid: "nerditd --serve")

    verdict = probe_daemon(data_dir=data_dir, pid_file=pid_file)
    assert verdict == DaemonLiveness(alive=True, pid=os.getpid(), evidence="pidfile")


def test_probe_ignores_a_reused_pid(fake_home, tmp_path, monkeypatch):
    """Identity-checked exactly like ``_terminate``: a crashed daemon's pid file
    plus pid reuse is a stale file, not a live daemon."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pid_file = tmp_path / "nerditd.pid"
    pid_file.write_text(str(os.getpid()))
    monkeypatch.setattr(uninstall_mod, "_process_command", lambda pid: "/usr/bin/vim notes.txt")

    assert probe_daemon(data_dir=data_dir, pid_file=pid_file).alive is False


def test_probe_never_creates_the_lock_file(fake_home, tmp_path):
    """``nerdit exit`` asking the question must not leave state behind."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    probe_daemon(data_dir=data_dir, pid_file=tmp_path / "absent.pid")
    assert not (data_dir / ".restore.lock").exists()


# --------------------------------------------------------------------------- #
# 3. stop precedence: the service unit first
# --------------------------------------------------------------------------- #


def test_exit_stops_through_the_service_unit(fake_home, tmp_path, monkeypatch, capsys):
    """A supervised daemon that is merely signalled comes straight back."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pid_file = tmp_path / "nerditd.pid"
    pid_file.write_text("606060")
    _wire(monkeypatch, _settings(data_dir, pid_file))

    unit = _unit(tmp_path)
    monkeypatch.setattr(exit_mod, "detect_service_unit", lambda: unit)
    verdicts = iter(
        [
            DaemonLiveness(alive=True, pid=606060, evidence="pidfile"),
            DaemonLiveness(alive=False, pid=None, evidence="none"),
        ]
    )
    monkeypatch.setattr(exit_mod, "probe_daemon", lambda **_k: next(verdicts))
    stopped: list[ServiceUnit] = []
    monkeypatch.setattr(
        exit_mod,
        "_stop_unit",
        lambda u: stopped.append(u) or (True, "ok"),
    )
    monkeypatch.setattr(
        exit_mod, "_terminate", lambda *a, **k: pytest.fail("must not signal a stopped daemon")
    )

    exit_daemon(yes=True)

    assert stopped == [unit]
    assert not pid_file.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "managed by a systemd-system service unit" in out
    assert "Daemon stopped (via the service unit)" in out


def test_exit_falls_back_to_signalling_when_the_unit_stop_does_not_work(
    fake_home, tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pid_file = tmp_path / "nerditd.pid"
    pid_file.write_text("707070")
    _wire(monkeypatch, _settings(data_dir, pid_file))

    monkeypatch.setattr(exit_mod, "detect_service_unit", lambda: _unit(tmp_path))
    verdicts = iter(
        [
            DaemonLiveness(alive=True, pid=707070, evidence="pidfile"),
            DaemonLiveness(alive=True, pid=707070, evidence="pidfile"),  # unit stop failed
            DaemonLiveness(alive=False, pid=None, evidence="none"),  # the signal worked
        ]
    )
    monkeypatch.setattr(exit_mod, "probe_daemon", lambda **_k: next(verdicts))
    monkeypatch.setattr(exit_mod, "_stop_unit", lambda u: (False, "failed"))
    signalled: list[tuple[int, float]] = []

    def _term(pid, label, *, grace_s=5.0, expect_cmd=None):
        signalled.append((pid, grace_s))
        return True

    monkeypatch.setattr(exit_mod, "_terminate", _term)

    exit_daemon(yes=True)

    assert signalled == [(707070, exit_mod._DAEMON_DRAIN_S)]
    out = " ".join(capsys.readouterr().out.split())
    assert "service unit did not stop it" in out
    assert "Daemon stopped" in out


def test_exit_reports_failure_when_the_daemon_survives_everything(
    fake_home, tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pid_file = tmp_path / "nerditd.pid"
    pid_file.write_text("808080")
    _wire(monkeypatch, _settings(data_dir, pid_file))

    alive = DaemonLiveness(alive=True, pid=808080, evidence="pidfile")
    monkeypatch.setattr(exit_mod, "probe_daemon", lambda **_k: alive)
    monkeypatch.setattr(exit_mod, "_terminate", lambda *a, **k: False)

    with pytest.raises(typer.Exit) as exc:
        exit_daemon(yes=True)
    assert exc.value.exit_code == 1
    assert pid_file.exists()  # never unlink a pid file we could not stop
    out = " ".join(capsys.readouterr().out.split())
    assert "Error stopping daemon" in out
    assert "pid 808080" in out


def test_exit_survives_a_broken_config(fake_home, tmp_path, monkeypatch, capsys):
    """A mangled config.toml must not stop you from stopping the daemon."""

    def _boom():
        raise ValueError("bad toml")

    monkeypatch.setattr(exit_mod, "load_settings", _boom)
    monkeypatch.setattr(exit_mod, "probe_daemon", lambda **_k: DaemonLiveness(False, None, "none"))

    exit_daemon(yes=True)
    out = " ".join(capsys.readouterr().out.split())
    assert "assuming defaults" in out
    assert "Daemon is not running" in out
