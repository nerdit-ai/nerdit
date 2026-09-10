"""Tests for the P30 install-layout seam (``nerdit.utils.install_layout``).

Four contracts live here, one per D-P30 decision:

* **D-P30-5** — ``resolve_caddy_binary``: explicit config always wins, the
  bundled binary wins over ``PATH`` only for the untouched default, and a
  source checkout resolves byte-identically to today (``shutil.which``).
* **D-P30-8** — ``detect_service_unit``: one answer on macOS (launchd), a
  user-before-system precedence on Linux, and the exact argv shared verbatim
  with ``packaging/install.sh``.
* **D-P30-9** — ``detect_install_layout``: both modes, plus a dangling
  ``current`` reported as an unknown version rather than a crash.
* the frozen spawn seam on :class:`~nerdit.daemon.lifecycle.DaemonLifecycle`,
  and a best-effort pin on the ``server.main`` re-exec branch that makes the
  frozen ``POST /daemon/restart`` work without any edit to ``server.py``.

Every filesystem root is reached through a module-level ``_…()`` helper, so
these tests monkeypatch helpers rather than faking a filesystem.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nerdit.utils import install_layout as il


@pytest.fixture(autouse=True)
def _no_real_service_start(monkeypatch):
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_service_unit", lambda: None)
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_install_layout", lambda: None)


# --------------------------------------------------------------------------- #
# D-P30-5 — caddy resolution
# --------------------------------------------------------------------------- #


def _bundle(tmp_path: Path) -> Path:
    """A fake frozen onedir bundle carrying a caddy next to the executable."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    caddy = bundle / "caddy"
    caddy.write_text("#!/bin/sh\n")
    caddy.chmod(0o755)
    return bundle


def test_bundled_caddy_is_none_when_not_frozen(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert il.is_frozen() is False
    assert il.bundled_caddy_path() is None


def test_bundled_caddy_found_when_frozen(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))
    assert il.is_frozen() is True
    assert il.bundled_caddy_path() == bundle / "caddy"


def test_bundled_caddy_none_when_bundle_carries_no_caddy(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(empty / "nerdit"))
    assert il.bundled_caddy_path() is None


def test_default_prefers_the_bundled_caddy(monkeypatch, tmp_path):
    bundled = _bundle(tmp_path) / "caddy"
    monkeypatch.setattr(il, "bundled_caddy_path", lambda: bundled)
    monkeypatch.setattr(il.shutil, "which", lambda name: "/usr/bin/caddy")

    assert il.resolve_caddy_binary("caddy") == str(bundled)


def test_explicit_config_wins_over_the_bundled_caddy(monkeypatch, tmp_path):
    """(D-P30-5) An operator who names a binary gets THAT binary — the bundled
    one is only ever a default-resolution step."""
    bundled = _bundle(tmp_path) / "caddy"
    monkeypatch.setattr(
        il, "bundled_caddy_path", lambda: pytest.fail("must not consult the bundle")
    )
    seen: list[str] = []

    def _which(name: str) -> str | None:
        seen.append(name)
        return "/opt/homebrew/bin/caddy2"

    monkeypatch.setattr(il.shutil, "which", _which)
    assert il.resolve_caddy_binary("/opt/homebrew/bin/caddy2") == "/opt/homebrew/bin/caddy2"
    assert seen == ["/opt/homebrew/bin/caddy2"]
    assert bundled.exists()  # untouched


def test_default_falls_back_to_path_without_a_bundle(monkeypatch):
    """Non-frozen behaviour is byte-identical to pre-P30: shutil.which('caddy')."""
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(il.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert il.resolve_caddy_binary("caddy") == "/usr/bin/caddy"


def test_default_returns_none_when_nothing_resolves(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(il.shutil, "which", lambda name: None)
    assert il.resolve_caddy_binary("caddy") is None


def test_proxy_manager_uses_the_resolution_seam(monkeypatch):
    """The one-line manager seam really goes through resolve_caddy_binary."""
    from nerdit.core.proxy import manager as manager_mod

    assert manager_mod.resolve_caddy_binary is il.resolve_caddy_binary


# --------------------------------------------------------------------------- #
# D-P30-8 — service unit detection
# --------------------------------------------------------------------------- #


def test_detect_launchd_plist_on_darwin(monkeypatch, tmp_path):
    plist = tmp_path / "ai.nerdit.daemon.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(il, "_is_darwin", lambda: True)
    monkeypatch.setattr(il, "_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(il.os, "getuid", lambda: 501)

    unit = il.detect_service_unit()
    assert unit is not None
    assert unit.kind == "launchd"
    assert unit.unit_path == plist
    assert unit.restart_argv == [
        "launchctl",
        "kickstart",
        "-k",
        "gui/501/ai.nerdit.daemon",
    ]
    assert unit.stop_argv == ["launchctl", "bootout", "gui/501/ai.nerdit.daemon"]
    # Uninstall removes the plist after bootout to prevent registration next login.
    assert unit.disable_argv == unit.stop_argv


def test_detect_none_on_darwin_without_a_plist(monkeypatch, tmp_path):
    monkeypatch.setattr(il, "_is_darwin", lambda: True)
    monkeypatch.setattr(il, "_launchd_plist_path", lambda: tmp_path / "absent.plist")
    assert il.detect_service_unit() is None


def test_detect_prefers_the_systemd_user_unit(monkeypatch, tmp_path):
    user_unit = tmp_path / "user" / "nerdit.service"
    user_unit.parent.mkdir()
    user_unit.write_text("[Unit]")
    system_unit = tmp_path / "system" / "nerdit.service"
    system_unit.parent.mkdir()
    system_unit.write_text("[Unit]")
    monkeypatch.setattr(il, "_is_darwin", lambda: False)
    monkeypatch.setattr(il, "_systemd_user_unit_path", lambda: user_unit)
    monkeypatch.setattr(il, "_systemd_system_unit_path", lambda: system_unit)

    unit = il.detect_service_unit()
    assert unit is not None
    assert unit.kind == "systemd-user"
    assert unit.unit_path == user_unit
    assert unit.restart_argv == ["systemctl", "--user", "restart", "nerdit.service"]
    assert unit.stop_argv == ["systemctl", "--user", "stop", "nerdit.service"]
    assert unit.disable_argv == ["systemctl", "--user", "disable", "nerdit.service"]


def test_detect_falls_back_to_the_systemd_system_unit(monkeypatch, tmp_path):
    system_unit = tmp_path / "system" / "nerdit.service"
    system_unit.parent.mkdir()
    system_unit.write_text("[Unit]")
    monkeypatch.setattr(il, "_is_darwin", lambda: False)
    monkeypatch.setattr(il, "_systemd_user_unit_path", lambda: tmp_path / "absent.service")
    monkeypatch.setattr(il, "_systemd_system_unit_path", lambda: system_unit)

    unit = il.detect_service_unit()
    assert unit is not None
    assert unit.kind == "systemd-system"
    assert unit.restart_argv == ["systemctl", "restart", "nerdit.service"]
    assert unit.stop_argv == ["systemctl", "stop", "nerdit.service"]
    assert unit.disable_argv == ["systemctl", "disable", "nerdit.service"]


def test_detect_none_on_linux_without_any_unit(monkeypatch, tmp_path):
    monkeypatch.setattr(il, "_is_darwin", lambda: False)
    monkeypatch.setattr(il, "_systemd_user_unit_path", lambda: tmp_path / "a.service")
    monkeypatch.setattr(il, "_systemd_system_unit_path", lambda: tmp_path / "b.service")
    assert il.detect_service_unit() is None


def test_unit_argv_match_the_shipped_templates():
    """The unit NAME/LABEL the argv target must be the one install.sh writes."""
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "packaging" / "units" / il.SYSTEMD_UNIT_NAME).is_file()
    assert (repo / "packaging" / "units" / f"{il.LAUNCHD_LABEL}.plist").is_file()
    assert (repo / "packaging" / "units" / "nerdit-user.service").is_file()


# --------------------------------------------------------------------------- #
# D-P30-9 — install layout detection
# --------------------------------------------------------------------------- #


def _user_install(root: Path, version: str = "0.5.0") -> Path:
    versions = root / "versions" / version
    versions.mkdir(parents=True)
    (versions / "install.sh").write_text("#!/bin/sh\n")
    (root / "current").symlink_to(Path("versions") / version)
    (root / "bin").mkdir()
    (root / "bin" / "nerdit").symlink_to(Path("..") / "current" / "nerdit")
    return versions


def test_detect_user_layout(monkeypatch, tmp_path):
    root = tmp_path / ".nerdit"
    root.mkdir()
    _user_install(root)
    monkeypatch.setattr(il, "_user_root", lambda: root)
    monkeypatch.setattr(il, "_system_root", lambda: tmp_path / "absent-opt")

    layout = il.detect_install_layout()
    assert layout is not None
    assert layout.mode == "user"
    assert layout.root == root
    assert layout.versions_dir == root / "versions"
    assert layout.current == root / "current"
    assert layout.current_version == "0.5.0"
    assert layout.shim == root / "bin" / "nerdit"
    assert layout.installer == root / "current" / "install.sh"


def test_detect_system_layout_wins(monkeypatch, tmp_path):
    opt = tmp_path / "opt-nerdit"
    (opt / "0.5.0").mkdir(parents=True)
    (opt / "current").symlink_to(Path("0.5.0"))
    user_root = tmp_path / ".nerdit"
    user_root.mkdir()
    _user_install(user_root)
    shim = tmp_path / "usr-local-bin" / "nerdit"
    monkeypatch.setattr(il, "_system_root", lambda: opt)
    monkeypatch.setattr(il, "_user_root", lambda: user_root)
    monkeypatch.setattr(il, "_system_shim", lambda: shim)

    layout = il.detect_install_layout()
    assert layout is not None
    assert layout.mode == "system"
    assert layout.root == opt
    assert layout.versions_dir == opt  # versioned dirs sit directly under /opt/nerdit
    assert layout.current_version == "0.5.0"
    assert layout.shim == shim
    assert layout.installer == opt / "current" / "install.sh"


def test_detect_none_for_a_source_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(il, "_system_root", lambda: tmp_path / "no-opt")
    monkeypatch.setattr(il, "_user_root", lambda: tmp_path / "no-home")
    assert il.detect_install_layout() is None


def test_dangling_current_is_a_layout_with_no_version(monkeypatch, tmp_path):
    root = tmp_path / ".nerdit"
    root.mkdir()
    (root / "current").symlink_to(Path("versions") / "0.4.0")  # target never created
    monkeypatch.setattr(il, "_user_root", lambda: root)
    monkeypatch.setattr(il, "_system_root", lambda: tmp_path / "no-opt")

    layout = il.detect_install_layout()
    assert layout is not None
    assert layout.current_version is None


def test_non_symlink_current_has_no_version(monkeypatch, tmp_path):
    root = tmp_path / ".nerdit"
    (root / "current").mkdir(parents=True)
    monkeypatch.setattr(il, "_user_root", lambda: root)
    monkeypatch.setattr(il, "_system_root", lambda: tmp_path / "no-opt")

    layout = il.detect_install_layout()
    assert layout is not None
    assert layout.current_version is None


# --------------------------------------------------------------------------- #
# frozen spawn seam (daemon/lifecycle.py)
# --------------------------------------------------------------------------- #


def test_spawn_argv_is_the_module_form_when_not_frozen(monkeypatch, tmp_path):
    from nerdit.daemon.lifecycle import DaemonLifecycle

    monkeypatch.delattr(sys, "frozen", raising=False)
    life = DaemonLifecycle(pid_file=str(tmp_path / "nerditd.pid"))
    assert life._spawn_argv() == [sys.executable, "-m", "nerdit.daemon.server"]


def test_spawn_argv_is_the_sibling_binary_when_frozen(monkeypatch, tmp_path):
    from nerdit.daemon.lifecycle import DaemonLifecycle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    nerditd = bundle / "nerditd"
    nerditd.write_text("#!/bin/sh\n")
    nerditd.chmod(0o755)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))

    life = DaemonLifecycle(pid_file=str(tmp_path / "nerditd.pid"))
    assert life._spawn_argv() == [str(nerditd)]


def test_spawn_argv_raises_on_an_incomplete_bundle(monkeypatch, tmp_path):
    from nerdit.daemon.lifecycle import DaemonLifecycle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))

    life = DaemonLifecycle(pid_file=str(tmp_path / "nerditd.pid"))
    with pytest.raises(RuntimeError) as excinfo:
        life._spawn_argv()
    assert "nerditd" in str(excinfo.value)


def test_start_spawns_the_resolved_argv(monkeypatch, tmp_path):
    """``start()`` hands Popen exactly what ``_spawn_argv`` resolved."""
    from nerdit.daemon import lifecycle as lifecycle_mod

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    nerditd = bundle / "nerditd"
    nerditd.write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))

    seen: dict = {}

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    def _popen(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs.get("env")
        return _Proc()

    monkeypatch.setattr(lifecycle_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(lifecycle_mod.time, "sleep", lambda *_a: None)

    life = lifecycle_mod.DaemonLifecycle(pid_file=str(tmp_path / "run" / "nerditd.pid"))
    assert life.start() is True
    assert seen["argv"] == [str(nerditd)]
    assert seen["env"]["NERDIT_BOOT_LOG"] == "1"


def test_server_reexec_branch_still_covers_a_frozen_daemon():
    """(best-effort pin) ``server.main`` re-execs ``boot_argv[0]`` whenever it is
    an on-disk file — which a frozen ``nerditd`` always is.

    The interpreter prefix is the part that must NOT survive freezing:
    ``sys.executable`` is the program itself there, so ``[sys.executable,
    *boot_argv]`` would append one more copy of the binary path on every
    ``POST /daemon/restart`` — unbounded argv growth across the auto-redeploy
    and config-apply restarts that make this routine.
    """
    import inspect

    from nerdit.daemon import server

    src = inspect.getsource(server.main)
    assert "boot_argv[0]" in src
    assert "is_file()" in src
    assert "os.execv" in src
    assert 'getattr(sys, "frozen", False)' in src


def test_spawn_argv_prefers_the_sibling_over_the_interpreter(monkeypatch, tmp_path):
    """A frozen bundle has no importable ``nerdit.daemon.server`` and no python:
    the ``-m`` form must never be reachable there."""
    from nerdit.daemon.lifecycle import DaemonLifecycle

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "nerditd").write_text("#!/bin/sh\n")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(bundle / "nerdit"))

    argv = DaemonLifecycle(pid_file=str(tmp_path / "p.pid"))._spawn_argv()
    assert "-m" not in argv
    assert len(argv) == 1


def test_the_layout_constants_are_the_documented_ones():
    """The D-P30-9 paths are constants, not config: install.sh hard-codes the
    same ones, and a drift here silently strands an install."""
    assert il.SYSTEMD_UNIT_NAME == "nerdit.service"
    assert il.LAUNCHD_LABEL == "ai.nerdit.daemon"
    assert str(il.SYSTEM_ROOT) == "/opt/nerdit"
    assert str(il.SYSTEM_SHIM) == "/usr/local/bin/nerdit"


def test_user_root_follows_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert il._user_root() == tmp_path / ".nerdit"
    assert il._launchd_plist_path() == (
        tmp_path / "Library" / "LaunchAgents" / "ai.nerdit.daemon.plist"
    )
    assert il._systemd_user_unit_path() == (
        tmp_path / ".config" / "systemd" / "user" / "nerdit.service"
    )
    assert il._systemd_system_unit_path() == Path("/etc/systemd/system/nerdit.service")
    assert os.fspath(il._system_root()) == "/opt/nerdit"


def test_managed_daemon_port_reads_the_field_the_shipped_launchd_plist_actually_has(
    tmp_path, monkeypatch
):
    """The plist records the home as WorkingDirectory, not as a HOME variable.

    Its ``EnvironmentVariables`` dict carries only ``NERDIT_BOOT_LOG``, so a
    version of this that looked for ``<key>HOME</key>`` never matched — and
    because "port unknown" is the branch that PERMITS a restart, the whole
    wrong-daemon guard was silently inert on macOS, the platform where a
    developer is most likely to have a second daemon on another port.
    """
    home = tmp_path / "home"
    (home / ".nerdit").mkdir(parents=True)
    (home / ".nerdit" / "config.toml").write_text("[daemon]\nport = 9444\n")
    plist = tmp_path / "ai.nerdit.daemon.plist"
    plist.write_text(
        "<plist><dict>\n"
        "  <key>EnvironmentVariables</key>\n"
        "  <dict><key>NERDIT_BOOT_LOG</key><string>1</string></dict>\n"
        "  <key>WorkingDirectory</key>\n"
        f"  <string>{home}</string>\n"
        "</dict></plist>\n"
    )
    unit = il.ServiceUnit(
        kind="launchd",
        unit_path=plist,
        restart_argv=["launchctl", "kickstart", "-k", "gui/501/ai.nerdit.daemon"],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) == 9444


def test_managed_daemon_port_reads_the_systemd_home_environment_line(tmp_path):
    """The systemd unit DOES record it as an environment line — both shapes work."""
    home = tmp_path / "home"
    (home / ".nerdit").mkdir(parents=True)
    (home / ".nerdit" / "config.toml").write_text("[daemon]\nport = 9555\n")
    unit_path = tmp_path / "nerdit.service"
    unit_path.write_text(
        f"[Service]\nEnvironment=HOME={home}\nExecStart=/opt/nerdit/current/nerditd\n"
    )
    unit = il.ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=["systemctl", "restart", "nerdit.service"],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) == 9555


def test_managed_daemon_port_answers_the_default_when_the_unit_user_wrote_no_config(tmp_path):
    """No config means the daemon runs on the default — the same answer it would reach."""
    home = tmp_path / "home"
    home.mkdir()
    unit_path = tmp_path / "nerdit.service"
    unit_path.write_text(f"[Service]\nEnvironment=HOME={home}\n")
    unit = il.ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=[],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) == il.DEFAULT_PORT


def test_managed_daemon_port_declines_when_the_config_path_cannot_even_be_stat_ed(tmp_path):
    """The state the ``is_file()`` pre-check silently mislabelled as "absent".

    ``Path.is_file()`` swallows the ``OSError`` and answers ``False`` for BOTH
    "no such file" and "cannot look", so a ``~/.nerdit`` that is not a directory
    — or, for an ordinary user, one that is not searchable — used to take the
    never-written-a-config branch and report ``DEFAULT_PORT`` for a unit that
    may well serve another one. That is precisely the guess this function
    exists to refuse. Staged here as a non-directory because that shape needs no
    permission bits and so pins the branch under root CI too.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".nerdit").write_text("not a directory\n")
    unit_path = tmp_path / "nerdit.service"
    unit_path.write_text(f"[Service]\nEnvironment=HOME={home}\n")
    unit = il.ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=[],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) is None


def test_managed_daemon_port_declines_when_the_config_exists_but_cannot_be_read(tmp_path):
    """The same contract in its ordinary EACCES shape: an unreadable config.

    Root bypasses the permission bits, so this one can only be staged as a
    non-root user; the sibling test above covers the branch either way.
    """
    if os.geteuid() == 0:
        pytest.skip("root reads a 0o000 file, so the unreadable state cannot be staged")
    home = tmp_path / "home"
    (home / ".nerdit").mkdir(parents=True)
    config = home / ".nerdit" / "config.toml"
    config.write_text("[daemon]\nport = 9666\n")
    config.chmod(0o000)
    unit_path = tmp_path / "nerdit.service"
    unit_path.write_text(f"[Service]\nEnvironment=HOME={home}\n")
    unit = il.ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=[],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) is None


def test_managed_daemon_port_declines_on_a_boolean_port(tmp_path):
    """``bool`` is an ``int`` subclass, so ``port = true`` would read as port 1."""
    home = tmp_path / "home"
    (home / ".nerdit").mkdir(parents=True)
    (home / ".nerdit" / "config.toml").write_text("[daemon]\nport = true\n")
    unit_path = tmp_path / "nerdit.service"
    unit_path.write_text(f"[Service]\nEnvironment=HOME={home}\n")
    unit = il.ServiceUnit(
        kind="systemd-system",
        unit_path=unit_path,
        restart_argv=[],
        stop_argv=[],
        disable_argv=[],
    )

    assert il.managed_daemon_port(unit) is None


@pytest.mark.parametrize("loaded", [True, False])
def test_start_launchd_registers_only_missing_jobs(monkeypatch, tmp_path, loaded):
    import subprocess

    unit = il._launchd_unit(tmp_path / "ai.nerdit.daemon.plist")
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        if argv[1] == "print" and not loaded:
            return subprocess.CompletedProcess(argv, 113, stderr="Could not find service")
        return subprocess.CompletedProcess(argv, 0, stderr="")

    monkeypatch.setattr(il.subprocess, "run", run)
    il.start_service_unit(unit)
    target = unit.stop_argv[-1]
    expected = [["launchctl", "print", target]]
    if not loaded:
        expected.append(["launchctl", "bootstrap", target.rsplit("/", 1)[0], str(unit.unit_path)])
    expected.append(["launchctl", "kickstart", target])
    assert seen == expected


@pytest.mark.parametrize("failed_command", ["print", "bootstrap", "kickstart"])
def test_launchd_start_reports_failures_without_enabling(monkeypatch, tmp_path, failed_command):
    import subprocess

    unit = il._launchd_unit(tmp_path / "ai.nerdit.daemon.plist")
    seen = []

    def run(argv, **kwargs):
        seen.append(argv[1])
        if argv[1] == failed_command:
            return subprocess.CompletedProcess(argv, 5, stderr="service is disabled")
        if argv[1] == "print":
            return subprocess.CompletedProcess(argv, 113, stderr="Could not find service")
        return subprocess.CompletedProcess(argv, 0, stderr="")

    monkeypatch.setattr(il.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="service is disabled"):
        il.start_service_unit(unit)
    assert seen[-1] == failed_command
    assert "enable" not in seen


@pytest.mark.parametrize(
    "kind,euid,prefix",
    [
        ("systemd-user", 501, ["systemctl", "--user"]),
        ("systemd-system", 0, ["systemctl"]),
        ("systemd-system", 501, ["sudo", "-n", "systemctl"]),
    ],
)
def test_start_systemd_is_noninteractive(monkeypatch, tmp_path, kind, euid, prefix):
    import subprocess

    unit = il._systemd_unit(kind, tmp_path / "nerdit.service")
    seen = []
    monkeypatch.setattr(il.os, "geteuid", lambda: euid)
    monkeypatch.setattr(
        il.subprocess,
        "run",
        lambda argv, **kw: seen.append(argv) or subprocess.CompletedProcess(argv, 0, stderr=""),
    )
    il.start_service_unit(unit)
    assert seen == [[*prefix, "start", "nerdit.service"]]


@pytest.mark.parametrize("port", [9444, None, 9321])
def test_lifecycle_starts_only_a_matching_installed_unit(monkeypatch, tmp_path, port):
    from nerdit.daemon import lifecycle as lm

    unit = il._launchd_unit(tmp_path / "daemon.plist")
    monkeypatch.setattr(lm, "detect_service_unit", lambda: unit)
    monkeypatch.setattr(lm, "managed_daemon_port", lambda u: port)
    monkeypatch.setattr(lm.subprocess, "Popen", lambda *a, **kw: pytest.fail("unmanaged spawn"))
    started = []
    monkeypatch.setattr(lm, "start_service_unit", started.append)
    lifecycle = lm.DaemonLifecycle(port=9444, pid_file=str(tmp_path / "daemon.pid"))
    if port == 9444:
        assert lifecycle.start()
        assert started == [unit]
        assert not lifecycle._pid_file.exists()  # service manager/daemon owns the PID
    else:
        with pytest.raises(RuntimeError, match="does not match"):
            lifecycle.start()
        assert not started


def test_lifecycle_refuses_an_install_missing_its_unit(monkeypatch, tmp_path):
    from nerdit.daemon import lifecycle as lm

    monkeypatch.setattr(lm, "detect_service_unit", lambda: None)
    monkeypatch.setattr(lm, "detect_install_layout", lambda: object())
    monkeypatch.setattr(lm.subprocess, "Popen", lambda *a, **kw: pytest.fail("unmanaged spawn"))
    with pytest.raises(RuntimeError, match="service is missing"):
        lm.DaemonLifecycle(pid_file=str(tmp_path / "daemon.pid")).start()
