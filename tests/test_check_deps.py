"""Tests for the shared check registry (nerdit.cli.checks) and check-deps."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from nerdit.cli import checks
from nerdit.cli.checks import (
    CheckResult,
    check_amd_access,
    check_buildx,
    check_caddy,
    check_disk,
    check_docker,
    check_git,
    check_nvidia_driver,
    check_nvidia_lib,
    check_nvidia_toolkit,
    check_port,
    check_python,
    check_runtime_image,
    check_zeroconf,
    check_zml_smi,
    detect_wsl2,
    platform_label,
    run_checks,
)


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# check_python
# ---------------------------------------------------------------------------


class TestCheckPython:
    def test_current_python_ok(self):
        r = check_python()
        assert r.status == "ok"
        assert "3." in r.detail

    def test_python_too_old(self, monkeypatch):
        monkeypatch.setattr("sys.version_info", (3, 10, 0, "final", 0))
        r = check_python()
        assert r.status == "fail"
        assert "3.10" in r.detail

    def test_too_old_remediation_platform_resolved(self, monkeypatch):
        monkeypatch.setattr("sys.version_info", (3, 10, 0, "final", 0))
        assert check_python("darwin").remediation == "brew install python@3.11"
        assert check_python("linux").remediation == "sudo apt install python3.11"


# ---------------------------------------------------------------------------
# check_nvidia_driver
# ---------------------------------------------------------------------------


class TestCheckNvidiaDriver:
    def test_nvidia_smi_not_found(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_nvidia_driver()
        assert r.status == "fail"
        assert "not found" in r.detail

    def test_nvidia_smi_good_version(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "550.54.14\n"))
        r = check_nvidia_driver()
        assert r.status == "ok"
        assert "550" in r.detail

    def test_nvidia_smi_old_version(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "470.82.01\n"))
        r = check_nvidia_driver()
        assert r.status == "fail"
        assert "470" in r.detail

    def test_nvidia_smi_fails(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(1))
        r = check_nvidia_driver()
        assert r.status == "fail"

    def test_wsl2_remediation_never_suggests_apt_driver(self, monkeypatch):
        # On WSL2 the GPU driver comes from the Windows host; an "apt install
        # nvidia-driver" hint would break passthrough.
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_nvidia_driver(is_wsl2=True)
        assert r.status == "fail"
        assert "apt install nvidia-driver" not in r.remediation
        assert "Windows" in r.remediation


# ---------------------------------------------------------------------------
# check_nvidia_lib
# ---------------------------------------------------------------------------


class TestCheckNvidiaLib:
    def test_lib_found(self, tmp_path, monkeypatch):
        (tmp_path / "libnvidia-ml.so.1").touch()
        monkeypatch.setattr("nerdit.cli.checks._LIB_DIRS", [str(tmp_path)])
        r = check_nvidia_lib()
        assert r.status == "ok"
        assert str(tmp_path) in r.detail

    def test_lib_not_found(self, monkeypatch):
        monkeypatch.setattr("nerdit.cli.checks._LIB_DIRS", ["/nonexistent"])
        r = check_nvidia_lib()
        assert r.status == "fail"
        assert "not found" in r.detail

    def test_wsl_lib_path_detected(self, tmp_path, monkeypatch):
        # Stand in for the standard lib dir: it exists but holds no lib, so
        # the search must walk past it to the WSL dir. Using a real host
        # path here (e.g. /usr/lib/x86_64-linux-gnu) is host-dependent — on
        # a machine with an NVIDIA driver installed it already contains
        # libnvidia-ml.so.1, which would short-circuit the search.
        std_dir = tmp_path / "usr" / "lib" / "x86_64-linux-gnu"
        std_dir.mkdir(parents=True)
        wsl_dir = tmp_path / "usr" / "lib" / "wsl" / "lib"
        wsl_dir.mkdir(parents=True)
        (wsl_dir / "libnvidia-ml.so.1").touch()
        monkeypatch.setattr(
            "nerdit.cli.checks._LIB_DIRS",
            [str(std_dir), str(wsl_dir)],
        )
        r = check_nvidia_lib()
        assert r.status == "ok"
        assert str(wsl_dir) in r.detail

    def test_wsl2_missing_remediation_is_wsl_safe(self, monkeypatch):
        monkeypatch.setattr("nerdit.cli.checks._LIB_DIRS", ["/nonexistent"])
        r = check_nvidia_lib(is_wsl2=True)
        assert r.status == "fail"
        assert "ldconfig" not in r.remediation
        assert "Windows" in r.remediation


class TestNvidiaLibDirs:
    def test_includes_wsl_path(self):
        from nerdit.config.defaults import NVIDIA_LIB_DIRS

        assert "/usr/lib/wsl/lib" in NVIDIA_LIB_DIRS

    def test_all_probe_sites_share_the_constant(self):
        from nerdit.cli import checks as checks_mod
        from nerdit.cli.commands import init
        from nerdit.config.defaults import NVIDIA_LIB_DIRS
        from nerdit.core.runtime import docker

        assert checks_mod._LIB_DIRS is NVIDIA_LIB_DIRS
        assert init.NVIDIA_LIB_DIRS is NVIDIA_LIB_DIRS
        assert docker.NVIDIA_LIB_DIRS is NVIDIA_LIB_DIRS


# ---------------------------------------------------------------------------
# check_docker — the three-way failure distinction
# ---------------------------------------------------------------------------


class TestCheckDocker:
    def test_binary_absent_linux(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_docker("linux", is_wsl2=False)
        assert r.status == "fail"
        assert "not on PATH" in r.detail
        assert "get.docker.com" in r.remediation

    def test_binary_absent_darwin(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_docker("darwin", is_wsl2=False)
        assert r.status == "fail"
        assert "Docker Desktop" in r.remediation

    def test_binary_absent_wsl2(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_docker("linux", is_wsl2=True)
        assert r.status == "fail"
        assert "WSL integration" in r.remediation

    def test_daemon_not_running_linux(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="Cannot connect to the Docker daemon"),
        )
        r = check_docker("linux", is_wsl2=False)
        assert r.status == "fail"
        assert "not running" in r.detail
        assert r.remediation == "sudo systemctl start docker"

    def test_daemon_not_running_darwin(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="Is the docker daemon running?"),
        )
        r = check_docker("darwin", is_wsl2=False)
        assert r.status == "fail"
        assert r.remediation == "start Docker Desktop"

    def test_permission_denied(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="permission denied while trying to connect"),
        )
        r = check_docker("linux", is_wsl2=False)
        assert r.status == "fail"
        assert "permission" in r.detail.lower()
        assert "usermod -aG docker" in r.remediation

    def test_docker_running(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        call_count = 0

        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completed(0)  # docker info
            return _completed(0, "24.0.7\n")  # docker version

        monkeypatch.setattr("subprocess.run", fake_run)
        r = check_docker("linux", is_wsl2=False)
        assert r.status == "ok"
        assert "24.0.7" in r.detail


# ---------------------------------------------------------------------------
# check_nvidia_toolkit
# ---------------------------------------------------------------------------


class TestCheckNvidiaToolkit:
    def test_toolkit_present(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(0, '{"nvidia":{},"runc":{}}'),
        )
        r = check_nvidia_toolkit()
        assert r.status == "ok"

    def test_toolkit_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(0, '{"runc":{}}'),
        )
        r = check_nvidia_toolkit()
        assert r.status == "fail"
        assert "nvidia-ctk" in r.remediation

    def test_docker_absent_is_skip(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_nvidia_toolkit()
        assert r.status == "skip"

    def test_docker_info_fail_skip_detail_is_cause_neutral(self, monkeypatch):
        # The permission-denied case must not be mislabeled "daemon down?".
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(1))
        r = check_nvidia_toolkit()
        assert r.status == "skip"
        assert "daemon down" not in r.detail
        assert "Docker Engine row" in r.detail


# ---------------------------------------------------------------------------
# check_buildx (BUG-1)
# ---------------------------------------------------------------------------


class TestCheckBuildx:
    def test_check_buildx_ok(self, monkeypatch):
        """Leg A of the both-ways pair: the plugin answers ``buildx version``."""
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(0, "github.com/docker/buildx v0.17.1 abcdef\n"),
        )
        r = check_buildx()
        assert r.status == "ok"
        assert "v0.17.1" in r.detail

    def test_check_buildx_ok_states_the_scope_of_its_claim(self, monkeypatch):
        """(Codex 3804646855) The ``ok`` row must not overstate what it proved.

        ``check-deps`` runs under the interactive user's HOME/DOCKER_CONFIG,
        while builds run under the daemon service's environment — a per-user
        ``~/.docker/cli-plugins`` install passes here and still fails every
        deploy. The qualifier lives in ``detail`` because
        ``cli/commands/check_deps.py`` only prints ``remediation`` for non-``ok``
        rows, so an ``ok``-row remediation would never be seen.
        """
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(0, "github.com/docker/buildx v0.17.1 abcdef\n"),
        )
        r = check_buildx()
        assert "v0.17.1" in r.detail  # the version still leads
        assert "this user's environment" in r.detail
        assert "nerdit doctor" in r.detail

    def test_check_buildx_fails_when_the_plugin_is_missing(self, monkeypatch):
        """Leg B: ``fail``, not ``warn`` — without buildx EVERY deploy fails at build."""
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, "", "docker: 'buildx' is not a docker command."),
        )
        r = check_buildx()
        assert r.status == "fail"
        # The remediation names the system-wide install AND the service-account
        # HOME trap that BUG-1 was actually about.
        assert "docker-buildx-plugin" in r.remediation
        assert "cli-plugins" in r.remediation

    def test_check_buildx_skips_when_docker_is_not_installed(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_buildx()
        assert r.status == "skip"

    def test_check_buildx_skips_when_the_docker_socket_is_unusable(self, monkeypatch):
        # The Docker Engine row already carries the exact diagnosis — do not
        # accuse a missing plugin when the daemon is simply unreachable.
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, "", "Cannot connect to the Docker daemon"),
        )
        r = check_buildx()
        assert r.status == "skip"
        assert "Docker Engine row" in r.detail

    def test_check_buildx_skips_when_docker_cannot_be_spawned(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")

        def fnf(*a, **kw):
            raise FileNotFoundError()

        monkeypatch.setattr("subprocess.run", fnf)
        assert check_buildx().status == "skip"


# ---------------------------------------------------------------------------
# check_runtime_image
# ---------------------------------------------------------------------------


class TestCheckRuntimeImage:
    def test_image_exists(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0))
        r = check_runtime_image()
        assert r.status == "ok"

    def test_image_missing(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(1))
        r = check_runtime_image()
        assert r.status == "fail"

    def test_docker_absent_is_skip(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_runtime_image()
        assert r.status == "skip"

    def test_daemon_down_is_skip_not_image_not_found(self, monkeypatch):
        # A connect error must NOT be reported as "image not found" (unknowable)
        # with a build hint that itself fails.
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="Cannot connect to the Docker daemon"),
        )
        r = check_runtime_image()
        assert r.status == "skip"
        assert "Docker Engine row" in r.detail

    def test_permission_denied_is_skip(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="permission denied while trying to connect"),
        )
        r = check_runtime_image()
        assert r.status == "skip"

    def test_genuine_missing_image_still_fails(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(1, stderr="Error: No such image: nerdit-runtime:0.1"),
        )
        r = check_runtime_image()
        assert r.status == "fail"
        assert "docker build" in r.remediation


# ---------------------------------------------------------------------------
# check_zml_smi (optional — skip, never fail)
# ---------------------------------------------------------------------------


class TestCheckZmlSmi:
    def test_not_found_is_skip(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_zml_smi()
        assert r.status == "skip"
        assert "not found" in r.detail

    def test_json_capability_available(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/zml-smi")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: _completed(0, '{"devices":[{"rocm":{"name":"MI300X"}}]}'),
        )
        r = check_zml_smi()
        assert r.status == "ok"
        assert "rocm" in r.detail

    def test_invalid_json_is_skip(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/zml-smi")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "not-json"))
        r = check_zml_smi()
        assert r.status == "skip"


# ---------------------------------------------------------------------------
# check_amd_access (optional — skip / warn, never fail)
# ---------------------------------------------------------------------------


class TestCheckAmdAccess:
    def test_no_amd_gpu_is_skip(self, monkeypatch):
        monkeypatch.setattr("nerdit.cli.checks.amd_gpu_present", lambda: False)
        r = check_amd_access()
        assert r.status == "skip"
        assert "no AMD" in r.detail

    def test_permission_problem_is_warn(self, monkeypatch):
        monkeypatch.setattr("nerdit.cli.checks.amd_gpu_present", lambda: True)
        monkeypatch.setattr(
            "nerdit.cli.checks.amd_gpu_access_diagnostic",
            lambda: "cannot access /dev/kfd",
        )
        r = check_amd_access()
        assert r.status == "warn"
        assert "cannot access /dev/kfd" in r.detail


# ---------------------------------------------------------------------------
# check_port — bind probe + nerditd detection
# ---------------------------------------------------------------------------


class _FreeSock:
    def setsockopt(self, *a):
        pass

    def bind(self, addr):
        pass

    def close(self):
        pass


class _BoundSock:
    def setsockopt(self, *a):
        pass

    def bind(self, addr):
        raise OSError("address already in use")

    def close(self):
        pass


class TestCheckPort:
    @pytest.fixture(autouse=True)
    def _no_listener(self, monkeypatch):
        # Pin the connect probe so these unit tests exercise the bind-probe
        # branch deterministically (a stray local listener on 9321 must not
        # flip them). The connect path is covered by the real-socket tests
        # below.
        monkeypatch.setattr("nerdit.cli.checks._port_listening", lambda _p: False)

    def test_port_free(self, monkeypatch):
        monkeypatch.setattr("socket.socket", lambda *a, **kw: _FreeSock())
        r = check_port(9321)
        assert r.status == "ok"
        assert "free" in r.detail

    def test_port_occupied_by_nerditd(self, monkeypatch):
        monkeypatch.setattr("socket.socket", lambda *a, **kw: _BoundSock())
        monkeypatch.setattr("nerdit.cli.checks._nerditd_answers", lambda _p: True)
        r = check_port(9321)
        assert r.status == "ok"
        assert "already running" in r.detail

    def test_port_occupied_by_other(self, monkeypatch):
        monkeypatch.setattr("socket.socket", lambda *a, **kw: _BoundSock())
        monkeypatch.setattr("nerdit.cli.checks._nerditd_answers", lambda _p: False)
        monkeypatch.setattr("nerdit.cli.checks._port_process", lambda _p: "nginx")
        r = check_port(9321)
        assert r.status == "fail"
        assert "(by nginx)" in r.detail
        assert "[daemon].port" in r.remediation

    def test_port_occupied_process_undiscoverable(self, monkeypatch):
        monkeypatch.setattr("socket.socket", lambda *a, **kw: _BoundSock())
        monkeypatch.setattr("nerdit.cli.checks._nerditd_answers", lambda _p: False)
        monkeypatch.setattr("nerdit.cli.checks._port_process", lambda _p: None)
        r = check_port(9321)
        assert r.status == "fail"
        assert "(by" not in r.detail

    def test_probe_sets_so_reuseaddr(self, monkeypatch):
        # Model the daemon (asyncio sets SO_REUSEADDR) so a TIME_WAIT socket is
        # not misreported as an occupant.
        import socket as _socket

        recorded = []

        class _RecordingSock(_FreeSock):
            def setsockopt(self, level, opt, val):
                recorded.append((level, opt, val))

        monkeypatch.setattr("socket.socket", lambda *a, **kw: _RecordingSock())
        check_port(9321)
        assert (_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1) in recorded


class TestCheckPortRealSockets:
    """Regression tests with real sockets — no ``socket.socket`` mocking.

    A bind probe alone is blind to a live uvicorn-style listener on macOS:
    uvicorn sets ``SO_REUSEADDR`` on its socket, and BSD lets a second
    ``SO_REUSEADDR`` bind succeed over it. ``check_port`` must therefore
    connect-first (observed live on macOS: a running nerditd reported
    "port is free").
    """

    def test_detects_reuseaddr_listener(self, monkeypatch):
        import socket as _socket

        listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setattr("nerdit.cli.checks._nerditd_answers", lambda _p: False)
        monkeypatch.setattr("nerdit.cli.checks._port_process", lambda _p: None)
        try:
            r = check_port(port)
        finally:
            listener.close()
        assert r.status == "fail"
        assert f"port {port} is in use" in r.detail

    def test_free_port_stays_free(self):
        import socket as _socket

        probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        r = check_port(port)
        assert r.status == "ok"
        assert "free" in r.detail


# ---------------------------------------------------------------------------
# check_disk — thresholds mirror /doctor exactly
# ---------------------------------------------------------------------------


class TestCheckDisk:
    def _usage(self, free_frac):
        total = 1000
        free = int(total * free_frac)
        return SimpleNamespace(total=total, used=total - free, free=free)

    def test_ok_above_10pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.disk_usage", lambda _p: self._usage(0.5))
        r = check_disk(str(tmp_path))
        assert r.status == "ok"

    def test_warn_between_5_and_10pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.disk_usage", lambda _p: self._usage(0.07))
        r = check_disk(str(tmp_path))
        assert r.status == "warn"

    def test_fail_below_5pct(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.disk_usage", lambda _p: self._usage(0.03))
        r = check_disk(str(tmp_path))
        assert r.status == "fail"

    def test_boundary_exactly_10pct_is_ok(self, tmp_path, monkeypatch):
        # free_pct < 0.10 is warn; exactly 0.10 must be ok (mirrors /doctor).
        monkeypatch.setattr("shutil.disk_usage", lambda _p: self._usage(0.10))
        r = check_disk(str(tmp_path))
        assert r.status == "ok"

    def test_boundary_exactly_5pct_is_warn(self, tmp_path, monkeypatch):
        monkeypatch.setattr("shutil.disk_usage", lambda _p: self._usage(0.05))
        r = check_disk(str(tmp_path))
        assert r.status == "warn"

    def test_nonexistent_dir_probes_existing_ancestor(self, tmp_path, monkeypatch):
        captured = {}

        def fake_usage(path):
            captured["path"] = path
            return self._usage(0.5)

        monkeypatch.setattr("shutil.disk_usage", fake_usage)
        missing = tmp_path / "a" / "b" / "c"
        r = check_disk(str(missing))
        assert r.status == "ok"
        # The probed path must be an existing ancestor, not the missing leaf.
        assert captured["path"].exists()


# ---------------------------------------------------------------------------
# check_caddy — all platforms, optional (warn, never fail)
# ---------------------------------------------------------------------------


class TestCheckCaddy:
    def test_present(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/caddy")
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "v2.7.6 h1:...\n"))
        r = check_caddy("linux")
        assert r.status == "ok"
        assert "v2.7.6" in r.detail

    def test_absent_darwin(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_caddy("darwin")
        assert r.status == "warn"
        assert "brew install caddy" in r.remediation

    def test_absent_linux(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_caddy("linux")
        assert r.status == "warn"
        assert "caddy" in r.remediation

    def test_frozen_install_finds_the_bundled_caddy_not_the_path(self, monkeypatch, tmp_path):
        """Live-run defect (P30 fresh-VM run, 2026-08-19): on a frozen install
        the bundled Caddy is the binary the daemon actually spawns, and PATH is
        the one place it is NOT — so a `shutil.which`-only check told the
        operator "not found / sudo apt install caddy" while that very Caddy was
        serving traffic. The row must resolve exactly like ProxyManager does.
        """
        bundled = tmp_path / "caddy"
        bundled.write_text("#!/bin/sh\n")
        monkeypatch.setattr("shutil.which", lambda _name: None)  # nothing on PATH
        monkeypatch.setattr("nerdit.utils.install_layout.is_frozen", lambda: True)
        monkeypatch.setattr("sys.executable", str(tmp_path / "nerdit"))
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "v2.11.4 h1:...\n"))
        r = check_caddy("linux")
        assert r.status == "ok"
        assert "v2.11.4" in r.detail
        assert str(bundled) in r.detail, "the resolved path must be checkable"

    def test_explicitly_configured_binary_wins(self, monkeypatch):
        """`[proxy].caddy_binary` is honoured here the same way ProxyManager
        honours it — otherwise the row diagnoses a different binary than the
        one that runs."""
        seen = {}

        def _which(name):
            seen["name"] = name
            return "/opt/mycaddy"

        monkeypatch.setattr("shutil.which", _which)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: _completed(0, "v2.9.0\n"))
        r = check_caddy("linux", "/opt/mycaddy")
        assert seen["name"] == "/opt/mycaddy"
        assert r.status == "ok"


# ---------------------------------------------------------------------------
# check_git
# ---------------------------------------------------------------------------


class TestCheckGit:
    def test_present(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/git")
        r = check_git("linux")
        assert r.status == "ok"

    def test_absent_linux(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_git("linux")
        assert r.status == "warn"
        assert "apt install git" in r.remediation

    def test_absent_darwin(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        r = check_git("darwin")
        assert r.status == "warn"
        assert "brew install git" in r.remediation


# ---------------------------------------------------------------------------
# check_zeroconf
# ---------------------------------------------------------------------------


class TestCheckZeroconf:
    def test_installed(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda _n: object())
        r = check_zeroconf(is_wsl2=False)
        assert r.status == "ok"

    def test_missing_is_warn(self, monkeypatch):
        monkeypatch.setattr("importlib.util.find_spec", lambda _n: None)
        r = check_zeroconf(is_wsl2=False)
        assert r.status == "warn"
        assert "nerdit[mdns]" in r.remediation

    def test_wsl2_is_skip(self):
        r = check_zeroconf(is_wsl2=True)
        assert r.status == "skip"
        assert "WSL2" in r.detail


# ---------------------------------------------------------------------------
# detect_wsl2 + platform_label
# ---------------------------------------------------------------------------


class TestPlatformDetection:
    def test_detect_wsl2_non_linux_is_false(self, monkeypatch):
        detect_wsl2.cache_clear()
        monkeypatch.setattr("nerdit.cli.checks.sys.platform", "darwin")
        assert detect_wsl2() is False
        detect_wsl2.cache_clear()

    def test_detect_wsl2_reads_proc_version(self, monkeypatch):
        import io

        detect_wsl2.cache_clear()
        monkeypatch.setattr("nerdit.cli.checks.sys.platform", "linux")
        monkeypatch.setattr(
            "builtins.open",
            lambda *a, **kw: io.StringIO("Linux version 5.15.0-microsoft-standard-WSL2"),
        )
        assert detect_wsl2() is True
        detect_wsl2.cache_clear()

    def test_detect_wsl2_plain_linux_is_false(self, monkeypatch):
        import io

        detect_wsl2.cache_clear()
        monkeypatch.setattr("nerdit.cli.checks.sys.platform", "linux")
        monkeypatch.setattr(
            "builtins.open",
            lambda *a, **kw: io.StringIO("Linux version 6.1.0-generic"),
        )
        assert detect_wsl2() is False
        detect_wsl2.cache_clear()

    def test_detect_wsl2_oserror_is_false(self, monkeypatch):
        def boom(*a, **kw):
            raise OSError("no /proc")

        detect_wsl2.cache_clear()
        monkeypatch.setattr("nerdit.cli.checks.sys.platform", "linux")
        monkeypatch.setattr("builtins.open", boom)
        assert detect_wsl2() is False
        detect_wsl2.cache_clear()

    def test_platform_label(self, monkeypatch):
        assert platform_label("darwin") == "macOS"
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)
        assert platform_label("linux") == "Linux"
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: True)
        assert platform_label("linux") == "WSL2"


# ---------------------------------------------------------------------------
# run_checks — platform matrices (darwin / linux / linux+WSL2)
# ---------------------------------------------------------------------------


def _stub_all_absent(monkeypatch):
    """Make every leaf probe fast + deterministic (nothing installed)."""
    monkeypatch.setattr("shutil.which", lambda _name: None)

    def fnf(*a, **kw):
        raise FileNotFoundError()

    monkeypatch.setattr("subprocess.run", fnf)
    monkeypatch.setattr("socket.socket", lambda *a, **kw: _FreeSock())
    monkeypatch.setattr("nerdit.cli.checks._port_listening", lambda _p: False)
    monkeypatch.setattr(
        "shutil.disk_usage",
        lambda _p: SimpleNamespace(total=1000, used=100, free=900),
    )
    monkeypatch.setattr("importlib.util.find_spec", lambda _n: None)
    monkeypatch.setattr("nerdit.cli.checks.amd_gpu_present", lambda: False)


class TestRunChecks:
    def test_darwin_has_no_gpu_rows(self, tmp_path, monkeypatch):
        _stub_all_absent(monkeypatch)
        results = run_checks("darwin", port=9321, data_dir=str(tmp_path))
        names = [r.name for r in results]
        assert not any("NVIDIA" in n or "nvidia" in n for n in names)
        assert checks.NAME_DOCKER in names
        assert checks.NAME_PORT in names
        assert checks.NAME_DISK in names
        assert checks.NAME_CADDY in names
        assert checks.NAME_GIT in names
        assert checks.NAME_ZEROCONF in names

    def test_run_checks_places_buildx_directly_after_docker(self, tmp_path, monkeypatch):
        """(BUG-1) The build prerequisite reads next to the runtime it extends."""
        _stub_all_absent(monkeypatch)
        results = run_checks("darwin", port=9321, data_dir=str(tmp_path))
        names = [r.name for r in results]
        assert checks.NAME_BUILDX in names
        assert names.index(checks.NAME_BUILDX) == names.index(checks.NAME_DOCKER) + 1

    def test_linux_has_gpu_rows(self, tmp_path, monkeypatch):
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)
        monkeypatch.setattr("nerdit.cli.checks._nvidia_gpu_present", lambda: True)
        results = run_checks("linux", port=9321, data_dir=str(tmp_path))
        names = [r.name for r in results]
        assert checks.NAME_NVIDIA_DRIVER in names
        assert checks.NAME_NVIDIA_TOOLKIT in names
        assert checks.NAME_RUNTIME_IMAGE in names
        assert checks.NAME_ZML_SMI in names
        assert checks.NAME_AMD_ACCESS in names
        # With a GPU present the driver row is a real (here: failing) probe.
        driver = next(r for r in results if r.name == checks.NAME_NVIDIA_DRIVER)
        assert driver.status == "fail"
        # No WSL2 hint rows on plain Linux.
        assert not any(n.startswith("WSL2 note") for n in names)

    def test_cpu_only_linux_gpu_rows_are_skip_not_fail(self, tmp_path, monkeypatch):
        # A GPU-less Linux box (CPU-only, a first-class mode) must not fail the
        # run nor be told to install an NVIDIA driver it doesn't need.
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)
        monkeypatch.setattr("nerdit.cli.checks._nvidia_gpu_present", lambda: False)
        results = run_checks("linux", port=9321, data_dir=str(tmp_path))
        by_name = {r.name: r for r in results}
        for gpu_row in (
            checks.NAME_NVIDIA_DRIVER,
            checks.NAME_NVIDIA_LIB,
            checks.NAME_NVIDIA_TOOLKIT,
            checks.NAME_RUNTIME_IMAGE,
        ):
            assert by_name[gpu_row].status == "skip"
            assert "CPU-only" in by_name[gpu_row].detail
        # None of the (formerly gate-capable) GPU rows fails on a CPU-only box.
        assert not any(
            by_name[n].status == "fail"
            for n in (
                checks.NAME_NVIDIA_DRIVER,
                checks.NAME_NVIDIA_LIB,
                checks.NAME_NVIDIA_TOOLKIT,
                checks.NAME_RUNTIME_IMAGE,
            )
        )

    def test_wsl2_emits_hint_rows_and_skips_mdns(self, tmp_path, monkeypatch):
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: True)
        monkeypatch.setattr("nerdit.cli.checks._nvidia_gpu_present", lambda: True)
        results = run_checks("linux", port=9321, data_dir=str(tmp_path))
        names = [r.name for r in results]
        assert any(n.startswith("WSL2 note") for n in names)
        # GPU rows still present (WSL2 supports passthrough).
        assert checks.NAME_NVIDIA_DRIVER in names
        # The driver remediation on WSL2 must be host-driver-safe.
        driver = next(r for r in results if r.name == checks.NAME_NVIDIA_DRIVER)
        assert "apt install nvidia-driver" not in (driver.remediation or "")
        zeroconf = next(r for r in results if r.name == checks.NAME_ZEROCONF)
        assert zeroconf.status == "skip"

    def test_optional_rows_never_fail(self, tmp_path, monkeypatch):
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)
        monkeypatch.setattr("nerdit.cli.checks._nvidia_gpu_present", lambda: True)
        results = run_checks("linux", port=9321, data_dir=str(tmp_path))
        by_name = {r.name: r for r in results}
        for optional in (
            checks.NAME_CADDY,
            checks.NAME_GIT,
            checks.NAME_ZEROCONF,
            checks.NAME_ZML_SMI,
            checks.NAME_AMD_ACCESS,
        ):
            assert by_name[optional].status in ("ok", "warn", "skip")

    def test_malformed_config_does_not_crash_and_emits_config_row(self, monkeypatch):
        # run_checks must survive a hand-mangled ~/.nerdit/config.toml (the exact
        # broken state check-deps exists to diagnose) — falling back to defaults
        # and surfacing a config row instead of a traceback.
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)

        import tomllib

        def boom():
            raise tomllib.TOMLDecodeError("Expected '=' after a key", "x", 0)

        monkeypatch.setattr("nerdit.cli.checks.load_settings", boom)
        # port/data_dir left to default (None) so the settings load is exercised.
        results = run_checks("darwin")
        by_name = {r.name: r for r in results}
        assert checks.NAME_CONFIG in by_name
        cfg = by_name[checks.NAME_CONFIG]
        assert cfg.status == "fail"
        assert "TOMLDecodeError" in cfg.detail
        # Port/disk rows still present (they used the fallback defaults).
        assert checks.NAME_PORT in by_name
        assert checks.NAME_DISK in by_name

    def test_valid_config_has_no_config_row(self, tmp_path, monkeypatch):
        _stub_all_absent(monkeypatch)
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: False)
        # Explicit port + data_dir ⇒ settings never loaded ⇒ no config row.
        results = run_checks("darwin", port=9321, data_dir=str(tmp_path))
        assert checks.NAME_CONFIG not in [r.name for r in results]


# ---------------------------------------------------------------------------
# CLI command — rendering, --json, exit codes, install guard
# ---------------------------------------------------------------------------


def _invoke(args, monkeypatch, results, platform="linux", input=None):
    from typer.testing import CliRunner

    from nerdit.cli.app import app

    monkeypatch.setattr("nerdit.cli.commands.check_deps.run_checks", lambda *a, **kw: results)
    monkeypatch.setattr("nerdit.cli.commands.check_deps.platform_label", lambda *a, **kw: platform)
    return CliRunner().invoke(app, args, input=input)


class TestCheckDepsCommand:
    def test_all_ok_exit_0(self, monkeypatch):
        results = [
            CheckResult(checks.NAME_PYTHON, "ok", "3.12.0"),
            CheckResult(checks.NAME_DOCKER, "ok", "24.0.7"),
        ]
        result = _invoke(["check-deps"], monkeypatch, results)
        assert result.exit_code == 0
        assert "All required dependencies" in result.output

    def test_warn_only_exit_0(self, monkeypatch):
        results = [
            CheckResult(checks.NAME_PYTHON, "ok", "3.12.0"),
            CheckResult(checks.NAME_CADDY, "warn", "not found", "brew install caddy"),
        ]
        result = _invoke(["check-deps"], monkeypatch, results)
        assert result.exit_code == 0
        assert "optional component" in result.output.lower()

    def test_any_fail_exit_1(self, monkeypatch):
        results = [
            CheckResult(checks.NAME_PYTHON, "ok", "3.12.0"),
            CheckResult(checks.NAME_DOCKER, "fail", "docker not on PATH", "curl ... | sudo sh"),
        ]
        result = _invoke(["check-deps"], monkeypatch, results)
        assert result.exit_code == 1
        assert "missing" in result.output.lower()
        assert "--install" in result.output

    def test_remediation_lines_rendered(self, monkeypatch):
        results = [
            CheckResult(checks.NAME_DOCKER, "fail", "docker not on PATH", "INSTALL_HINT_XYZ"),
        ]
        result = _invoke(["check-deps"], monkeypatch, results)
        assert "INSTALL_HINT_XYZ" in result.output

    def test_json_schema(self, monkeypatch):
        results = [
            CheckResult(checks.NAME_PYTHON, "ok", "3.12.0"),
            CheckResult(checks.NAME_CADDY, "warn", "not found", "brew install caddy"),
        ]
        result = _invoke(["check-deps", "--json"], monkeypatch, results, platform="macOS")
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["platform"] == "macOS"
        assert isinstance(payload["checks"], list)
        first = payload["checks"][0]
        assert set(first.keys()) == {"name", "status", "detail", "remediation"}
        assert first["name"] == checks.NAME_PYTHON
        assert first["remediation"] is None
        # No Rich markup leaked into the JSON values.
        assert "[green]" not in result.output

    def test_json_exit_1_on_fail(self, monkeypatch):
        results = [CheckResult(checks.NAME_DOCKER, "fail", "absent", "hint")]
        result = _invoke(["check-deps", "--json"], monkeypatch, results)
        assert result.exit_code == 1
        json.loads(result.output)  # still valid JSON

    def test_install_non_apt_distro_guard(self, monkeypatch):
        results = [CheckResult(checks.NAME_DOCKER, "fail", "absent", "manual docker steps")]
        monkeypatch.setattr("nerdit.cli.commands.check_deps.sys.platform", "linux")
        monkeypatch.setattr("nerdit.cli.commands.check_deps.shutil.which", lambda _n: None)
        result = _invoke(["check-deps", "--install"], monkeypatch, results)
        assert result.exit_code == 1
        assert "Unsupported distro" in result.output
        assert "manual docker steps" in result.output

    def test_install_darwin_prints_manual_steps(self, monkeypatch):
        results = [CheckResult(checks.NAME_DOCKER, "fail", "absent", "install Docker Desktop")]
        monkeypatch.setattr("nerdit.cli.commands.check_deps.sys.platform", "darwin")
        result = _invoke(["check-deps", "--install"], monkeypatch, results, platform="macOS")
        assert result.exit_code == 1
        assert "install Docker Desktop" in result.output

    def test_install_wsl2_never_offers_linux_nvidia_driver(self, monkeypatch):
        # Codex P1 (PR #86): under WSL2 the driver lives on the Windows host —
        # the apt driver installer must never be offered, even though WSL2 is
        # sys.platform == "linux" with apt-get present.
        results = [
            CheckResult(
                checks.NAME_NVIDIA_DRIVER,
                "fail",
                "550.10 (need ≥535 on the Windows host)",
                "update the NVIDIA driver on Windows",
            )
        ]
        monkeypatch.setattr("nerdit.cli.commands.check_deps.sys.platform", "linux")
        monkeypatch.setattr(
            "nerdit.cli.commands.check_deps.shutil.which", lambda _n: "/usr/bin/apt-get"
        )
        monkeypatch.setattr("nerdit.cli.checks.detect_wsl2", lambda: True)
        called = []
        monkeypatch.setattr(
            "nerdit.cli.commands.check_deps._install_nvidia_driver",
            lambda: called.append(True) or True,
        )
        # Auto-answer "y" to any prompt: if the installer were offered, it would run.
        result = _invoke(["check-deps", "--install"], monkeypatch, results, input="y\n")
        assert result.exit_code == 1
        assert not called
        assert "no automatic installer" in result.output
        assert "update the NVIDIA driver on Windows" in result.output
