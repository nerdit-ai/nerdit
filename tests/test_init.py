"""Tests for the init command and daemon lifecycle."""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import socket
import time
from pathlib import Path

import httpx
import pytest

import nerdit
from nerdit.config.settings import get_client_config, load_settings
from nerdit.daemon.lifecycle import DaemonLifecycle


@pytest.fixture(autouse=True)
def _no_real_service_start(monkeypatch):
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_service_unit", lambda: None)
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_install_layout", lambda: None)


class TestDaemonLifecycle:
    """Test DaemonLifecycle with a real subprocess daemon."""

    @staticmethod
    def _daemon_gone() -> bool:
        """True when no daemon holds the port *or* the data dir's restore lock.

        The teardown only sends SIGTERM, and a fixed sleep is not a shutdown
        barrier: uvicorn closes the listening socket *first*, then runs the
        lifespan teardown (DB close, proxy teardown, releasing the
        ``.restore.lock`` the daemon holds shared for its whole life). On a data
        dir with real state that tail takes seconds, so "port is free" is not
        "daemon is gone" — and the next test's daemon dies at boot with "a
        restore is in progress". Check the lock itself, which is the resource
        actually in contention.
        """
        try:
            with socket.create_connection(("127.0.0.1", 9321), timeout=0.5):
                return False
        except OSError:
            pass
        lock_path = Path(load_settings().data_dir).expanduser() / ".restore.lock"
        if not lock_path.exists():
            return True
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        finally:
            os.close(fd)  # closing drops the lock we just took
        return True

    @classmethod
    def _wait_daemon_gone(cls, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cls._daemon_gone():
                return True
            time.sleep(0.2)
        return False

    @pytest.fixture(autouse=True)
    def _cleanup_daemon(self, tmp_path, monkeypatch):
        """Run only this test's daemon, with isolated state and container identity."""
        monkeypatch.setenv("HOME", str(tmp_path))
        config_dir = tmp_path / ".nerdit"
        config_dir.mkdir()
        (config_dir / "config.toml").write_text(
            f'[daemon]\ninstance_id = "test-{tmp_path.name}"\n'
            "[proxy]\nenabled = false\nmdns = false\n"
        )
        self.pid_file = str(tmp_path / "nerditd.pid")
        self.lc = DaemonLifecycle(pid_file=self.pid_file)
        if not self._daemon_gone():
            pytest.fail("Port 9321 is occupied; refusing to stop an unrelated daemon.")
        yield
        # Cleanup: SIGTERM the daemon if still running, SIGKILL if it wedges.
        pid = None
        with contextlib.suppress(OSError, ValueError):
            pid = int(Path(self.pid_file).read_text().strip())
        if pid is not None:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
        if not self._wait_daemon_gone():
            # Graceful shutdown wedged — escalate rather than leak the port and
            # the restore lock into the next test.
            if pid is not None:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
            assert self._wait_daemon_gone(), "Test daemon did not release its port"

    def test_is_running_false_when_no_pid_file(self):
        assert self.lc.is_running() is False

    def test_is_running_false_when_stale_pid(self, tmp_path):
        pid_path = Path(self.pid_file)
        pid_path.write_text("999999999")  # Non-existent PID
        assert self.lc.is_running() is False
        assert not pid_path.exists()  # Stale PID file cleaned up

    def test_start_and_wait_for_ready(self):
        assert self.lc.start() is True
        assert self.lc.wait_for_ready(timeout=10.0) is True

        # Verify health endpoint responds
        resp = httpx.get("http://127.0.0.1:9321/health", timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == nerdit.__version__

    def test_start_is_idempotent(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)
        # Starting again should detect it's already running
        assert self.lc.start() is True

    def test_is_running_true_when_pid_owned_by_another_user(self, monkeypatch):
        pid_path = Path(self.pid_file)
        pid_path.write_text("1")

        def _eperm(pid, sig):
            raise PermissionError

        monkeypatch.setattr("nerdit.daemon.lifecycle.os.kill", _eperm)
        assert self.lc.is_running() is True
        assert pid_path.exists()  # never unlinked for a live process
        pid_path.unlink()  # keep the teardown away from pid 1

    def test_health_endpoint_returns_gpu_count(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)

        resp = httpx.get("http://127.0.0.1:9321/health", timeout=5.0)
        data = resp.json()
        # Should report detected GPUs (at least 0, depending on environment)
        assert "gpu_count" in data
        assert isinstance(data["gpu_count"], int)

    def test_gpus_endpoint(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)

        _, _, token = get_client_config()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = httpx.get("http://127.0.0.1:9321/gpus", timeout=5.0, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # If GPUs are available, check structure
        if data:
            gpu = data[0]
            assert "id" in gpu
            assert "name" in gpu
            assert "memory_mb" in gpu
            assert "status" in gpu

    def test_wait_for_ready_timeout(self):
        # Use a port where nothing is running
        lc_bad = DaemonLifecycle(host="127.0.0.1", port=19999, pid_file=self.pid_file)
        assert lc_bad.wait_for_ready(timeout=1.0) is False

    def test_pid_file_written(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)

        pid_path = Path(self.pid_file)
        assert pid_path.exists()
        pid = int(pid_path.read_text().strip())
        assert pid > 0
        # Verify PID is a real process
        os.kill(pid, 0)  # Should not raise

    def test_daemon_log_file_created(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)

        log_file = Path(self.pid_file).parent / "nerditd.boot.log"
        # The log file should exist (lifecycle writes daemon output there)
        # Note: it's in tmp_path so it won't be in ~/.nerdit
        assert log_file.exists()

    def test_is_running_true_after_start(self):
        self.lc.start()
        self.lc.wait_for_ready(timeout=10.0)
        assert self.lc.is_running() is True


class TestInitSharedRegistry:
    """init reuses the shared ``nerdit.cli.checks`` registry instead of its own
    duplicated docker probes (no drift between ``init`` and ``check-deps``)."""

    def test_check_docker_delegates_to_registry(self, monkeypatch):
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        seen = {}

        def fake_check_docker(platform, is_wsl2=False):
            seen["platform"] = platform
            return checks.CheckResult(checks.NAME_DOCKER, "ok", "running")

        monkeypatch.setattr("nerdit.cli.checks.check_docker", fake_check_docker)
        assert init._check_docker() is True
        assert "platform" in seen  # init actually called into the registry

        monkeypatch.setattr(
            "nerdit.cli.checks.check_docker",
            lambda *a, **k: checks.CheckResult(checks.NAME_DOCKER, "fail", "down"),
        )
        assert init._check_docker() is False

    def test_check_nvidia_toolkit_delegates_to_registry(self, monkeypatch):
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        monkeypatch.setattr(
            "nerdit.cli.checks.check_nvidia_toolkit",
            lambda: checks.CheckResult(checks.NAME_NVIDIA_TOOLKIT, "ok", "registered"),
        )
        assert init._check_nvidia_toolkit() is True

        monkeypatch.setattr(
            "nerdit.cli.checks.check_nvidia_toolkit",
            lambda: checks.CheckResult(checks.NAME_NVIDIA_TOOLKIT, "fail", "missing"),
        )
        assert init._check_nvidia_toolkit() is False


class TestInitScaffoldsNoDeadConfig:
    """`nerdit init` must not scaffold sections/keys the daemon no longer reads."""

    def test_default_config_has_no_batch_sections_and_still_loads(self, tmp_path):
        import tomllib

        from nerdit.cli.commands import init
        from nerdit.config.settings import load_settings

        config_text, token = init._generate_default_config()
        assert token
        parsed = tomllib.loads(config_text)
        assert parsed["proxy"] == {"enabled": True, "mdns": True, "https_port": 8443}

        assert "scheduler" not in parsed
        assert "mount_workdir" not in parsed["containers"]
        assert "auto_repair" not in parsed["monitor"]
        # The fallback service image is NOT batch-only — it must stay.
        assert parsed["containers"]["default_image"]

        path = tmp_path / "config.toml"
        path.write_text(config_text)
        assert load_settings(path).daemon.auth_token == token

    def test_project_template_carries_no_batch_sections(self, tmp_path):
        import tomllib

        from nerdit.cli.commands import init
        from nerdit.config.project import load_project_config

        parsed = tomllib.loads(init.NERDIT_TOML_TEMPLATE)
        for dead in ("run", "scheduling", "data"):
            assert dead not in parsed

        # Every section is commented out, so a freshly scaffolded nerdit.toml
        # loads cleanly instead of 422-ing on a half-declared [deploy] table.
        path = tmp_path / "nerdit.toml"
        path.write_text(init.NERDIT_TOML_TEMPLATE)
        project = load_project_config(path)
        assert project is not None
        assert project.deploy is None and project.ai is None and project.db is None


class TestInitOnAnInstallerMadeInstall:
    """(P30) `nerdit init` is the SOURCE-CHECKOUT bootstrap.

    On an installer-made install the daemon is already running under a service
    unit, and the default config it would write sets `host = "0.0.0.0"` and a
    fresh `auth_token` — both restart-keyed, so the node would sit in
    `config_restart_pending` and, once restarted, bind every interface: the
    opposite of what `docs/guide/install.md` promises.
    """

    @staticmethod
    async def _run(monkeypatch, tmp_path, *, installed: bool) -> Path:
        from nerdit.cli.commands import init as init_mod

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        # `init` imports the detector inside the function, so the seam is the
        # source module, not a name bound on `init`.
        monkeypatch.setattr(
            "nerdit.utils.install_layout.detect_install_layout",
            lambda: object() if installed else None,
        )
        monkeypatch.setattr(init_mod, "_check_docker", lambda: False)
        monkeypatch.setattr(init_mod, "_offer_start_docker", lambda: False)

        class _Life:
            def __init__(self, *a, **kw) -> None: ...
            def is_running(self) -> bool:
                return True

        monkeypatch.setattr("nerdit.daemon.lifecycle.DaemonLifecycle", _Life)

        class _Client:
            def __init__(self, *a, **kw) -> None: ...
            async def list_gpus(self):
                raise RuntimeError("no daemon in this test")

        monkeypatch.setattr("nerdit.cli.client.NerditClient", _Client)

        with contextlib.suppress(Exception):
            await init_mod._init_async()
        return home / ".nerdit" / "config.toml"

    async def test_installed_layout_writes_no_config(self, monkeypatch, tmp_path):
        config = await self._run(monkeypatch, tmp_path, installed=True)
        assert not config.exists()

    async def test_source_checkout_still_writes_the_config(self, monkeypatch, tmp_path):
        config = await self._run(monkeypatch, tmp_path, installed=False)
        assert config.exists()
        assert "auth_token" in config.read_text()


class _RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args, **kwargs) -> None:
        self.lines.append(" ".join(str(a) for a in args))


class TestDaemonStartFailureMessage:
    """When the daemon fails to come up, a busy port yields a specific,
    actionable message (via the shared port bind-probe) instead of the generic
    'did not respond in time'."""

    def test_port_occupied_prints_specific_message(self, monkeypatch):
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        rec = _RecordingConsole()
        monkeypatch.setattr("nerdit.cli.commands.init.console", rec)
        monkeypatch.setattr(
            "nerdit.cli.checks.check_port",
            lambda port=9321: checks.CheckResult(
                checks.NAME_PORT,
                "fail",
                f"port {port} is in use (by nginx)",
                "stop it or set [daemon].port in ~/.nerdit/config.toml",
            ),
        )

        init._report_daemon_start_failure(9321)

        joined = "\n".join(rec.lines)
        assert "in use" in joined
        assert "nginx" in joined
        assert "stop it or set" in joined
        # The generic message must NOT be the one shown here.
        assert "did not respond in time" not in joined

    def test_port_free_prints_generic_message(self, monkeypatch):
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        rec = _RecordingConsole()
        monkeypatch.setattr("nerdit.cli.commands.init.console", rec)
        monkeypatch.setattr(
            "nerdit.cli.checks.check_port",
            lambda port=9321: checks.CheckResult(checks.NAME_PORT, "ok", f"port {port} is free"),
        )

        recovered = init._report_daemon_start_failure(9321)

        joined = "\n".join(rec.lines)
        assert "did not respond in time" in joined
        assert "boot log" in joined
        assert recovered is False

    def test_daemon_came_up_late_reports_success(self, monkeypatch):
        # The port bind-probe now sees a live nerditd: the daemon simply came up
        # after the readiness wait — treat it as a late success, not a failure.
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        rec = _RecordingConsole()
        monkeypatch.setattr("nerdit.cli.commands.init.console", rec)
        monkeypatch.setattr(
            "nerdit.cli.checks.check_port",
            lambda port=9321: checks.CheckResult(
                checks.NAME_PORT, "ok", f"nerditd already running on port {port}"
            ),
        )

        recovered = init._report_daemon_start_failure(9321)

        assert recovered is True
        assert "came up" in "\n".join(rec.lines)

    def test_wedged_own_child_points_at_boot_log_not_foreign_occupant(self, monkeypatch):
        # Port held, nothing nerdit-shaped answers, but OUR spawned child is
        # still alive (wedged) — do NOT blame a foreign occupant.
        from nerdit.cli import checks
        from nerdit.cli.commands import init

        rec = _RecordingConsole()
        monkeypatch.setattr("nerdit.cli.commands.init.console", rec)
        monkeypatch.setattr(
            "nerdit.cli.checks.check_port",
            lambda port=9321: checks.CheckResult(
                checks.NAME_PORT,
                "fail",
                f"port {port} is in use (by Python)",
                "stop it or set [daemon].port in ~/.nerdit/config.toml",
            ),
        )

        class _AliveLifecycle:
            def is_running(self):
                return True

        recovered = init._report_daemon_start_failure(9321, _AliveLifecycle())

        joined = "\n".join(rec.lines)
        assert recovered is False
        assert "boot log" in joined
        # The foreign-occupant remediation must NOT be shown for our own wedged child.
        assert "stop it or set" not in joined


class TestInstallerAuthToken:
    """P30: `nerdit init --auth-token-only` — the installer's token mint.

    A fresh install writes no config.toml, and a tokenless daemon treats every
    local caller as admin. The installer closes that; these pin the contract.
    """

    def test_fresh_config_enables_https_and_mdns(self, tmp_path):
        import tomllib

        from nerdit.cli.commands.init import _write_installer_auth_config

        assert _write_installer_auth_config(tmp_path) == "written"
        text = (tmp_path / "config.toml").read_text()
        assert "[daemon]" in text
        assert "auth_token" in text
        # Never the default template's bind-all host: the docs promise loopback.
        assert "0.0.0.0" not in text
        assert tomllib.loads(text)["proxy"] == {"enabled": True, "mdns": True, "https_port": 8443}

    def test_existing_config_keeps_proxy_opt_out(self, tmp_path):
        import tomllib

        from nerdit.cli.commands.init import _write_installer_auth_config

        path = tmp_path / "config.toml"
        path.write_text("[proxy]\nenabled = false\nmdns = false\nhttps_port = 9443\n")
        assert _write_installer_auth_config(tmp_path) == "written"
        assert tomllib.loads(path.read_text())["proxy"] == {
            "enabled": False,
            "mdns": False,
            "https_port": 9443,
        }

    def test_is_idempotent_and_never_rotates(self, tmp_path):
        from nerdit.cli.commands.init import _write_installer_auth_config

        assert _write_installer_auth_config(tmp_path) == "written"
        first = (tmp_path / "config.toml").read_text()
        assert _write_installer_auth_config(tmp_path) == "present"
        assert (tmp_path / "config.toml").read_text() == first

    def test_config_is_owner_only(self, tmp_path):
        import stat

        from nerdit.cli.commands.init import _write_installer_auth_config

        _write_installer_auth_config(tmp_path)
        mode = (tmp_path / "config.toml").stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_the_token_is_readable_by_the_local_cli(self, tmp_path):
        """The whole point: `nerdit link` must authenticate with no flag."""
        from nerdit.cli.commands.init import _write_installer_auth_config
        from nerdit.config.settings import get_client_config

        _write_installer_auth_config(tmp_path)
        _host, _port, token = get_client_config(tmp_path / "config.toml")
        assert token, "the CLI must pick the minted token up from config.toml"

    def test_installer_mints_before_starting_the_unit(self):
        """Order matters: a daemon booted tokenless stays tokenless."""
        text = (Path(__file__).resolve().parents[1] / "packaging" / "install.sh").read_text()
        mint = text.index("init --auth-token-only")
        start = text.index("\nstart_unit\n")
        assert mint < start

    def test_installer_mints_as_the_unit_user(self):
        text = (Path(__file__).resolve().parents[1] / "packaging" / "install.sh").read_text()
        assert 'sudo -n -u "$UNIT_USER" env HOME="$UNIT_HOME"' in text

    def test_existing_config_without_a_token_still_gets_one(self, tmp_path):
        """The guard is the TOKEN, not the file.

        A hand-written config (someone who set [proxy] before installing) used
        to be read as "already configured", leaving a tokenless daemon — the
        exact hazard D-P30-12 exists to close.
        """
        import tomllib

        from nerdit.cli.commands.init import _write_installer_auth_config

        cfg = tmp_path / "config.toml"
        cfg.write_text("# my notes\n[proxy]\nenabled = true\n")
        assert _write_installer_auth_config(tmp_path) == "written"
        parsed = tomllib.loads(cfg.read_text())
        assert parsed["daemon"]["auth_token"]
        # The operator's own settings and comments survive.
        assert parsed["proxy"]["enabled"] is True
        assert "# my notes" in cfg.read_text()

    def test_token_inserted_into_an_existing_daemon_table(self, tmp_path):
        import tomllib

        from nerdit.cli.commands.init import _write_installer_auth_config

        cfg = tmp_path / "config.toml"
        cfg.write_text("[daemon]\nport = 9999\n")
        assert _write_installer_auth_config(tmp_path) == "written"
        parsed = tomllib.loads(cfg.read_text())
        assert parsed["daemon"]["auth_token"]
        assert parsed["daemon"]["port"] == 9999, "existing keys must survive"

    def test_a_malformed_config_is_never_overwritten(self, tmp_path):
        from nerdit.cli.commands.init import _write_installer_auth_config

        cfg = tmp_path / "config.toml"
        cfg.write_text("this is not = = toml\n")
        assert _write_installer_auth_config(tmp_path) == "failed"
        assert cfg.read_text() == "this is not = = toml\n"

    def test_the_token_file_is_never_briefly_world_readable(self, tmp_path):
        """install.sh runs at umask 022, so write_text() would create 0644."""
        import os
        import stat

        from nerdit.cli.commands.init import _write_installer_auth_config

        old = os.umask(0o022)
        try:
            assert _write_installer_auth_config(tmp_path) == "written"
        finally:
            os.umask(old)
        mode = stat.S_IMODE((tmp_path / "config.toml").stat().st_mode)
        assert mode == 0o600, f"token file is {oct(mode)}, not 0600"

    def test_commented_and_spaced_daemon_headers_are_recognised(self, tmp_path):
        """Review finding: an exact-string header test missed valid TOML.

        A missed header appended a SECOND [daemon] table — invalid TOML — so
        the write was rejected and the install continued token-less, i.e. the
        hazard this code exists to close.
        """
        import tomllib

        from nerdit.cli.commands.init import _write_installer_auth_config

        for header in ("[daemon] # local settings", "  [daemon]  ", "[ daemon ]"):
            d = tmp_path / header.strip().replace(" ", "_").replace("#", "h")
            d.mkdir()
            (d / "config.toml").write_text(f"{header}\nport = 9333\n")
            assert _write_installer_auth_config(d) == "written", header
            parsed = tomllib.loads((d / "config.toml").read_text())
            assert parsed["daemon"]["auth_token"], header
            assert parsed["daemon"]["port"] == 9333, header

    def test_a_present_token_still_gets_its_file_tightened(self, tmp_path):
        """A copied 0644 config holds the ADMIN bearer token."""
        import stat

        from nerdit.cli.commands.init import _write_installer_auth_config

        cfg = tmp_path / "config.toml"
        cfg.write_text('[daemon]\nauth_token = "already-here"\n')
        cfg.chmod(0o644)
        assert _write_installer_auth_config(tmp_path) == "present"
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600


@pytest.mark.parametrize("start_error", [False, True])
async def test_init_uses_local_settings_and_preserves_service_errors(
    monkeypatch, tmp_path, start_error
):
    import typer

    from nerdit.cli.commands import init as init_mod

    monkeypatch.setenv("HOME", str(tmp_path))
    data_dir = tmp_path / ".nerdit"
    data_dir.mkdir()
    pid_file = tmp_path / "custom.pid"
    (data_dir / "config.toml").write_text(
        f'[daemon]\nhost = "127.0.0.1"\nport = 9444\npid_file = "{pid_file}"\n'
        '[client]\nremote_host = "remote.invalid"\nremote_port = 9999\n'
    )
    monkeypatch.setattr(init_mod, "_check_docker", lambda: False)
    monkeypatch.setattr(init_mod, "_offer_start_docker", lambda: False)
    monkeypatch.setattr("nerdit.utils.install_layout.detect_install_layout", lambda: object())
    seen = {}

    class Life:
        def __init__(self, **kwargs):
            seen["lifecycle"] = kwargs

        def is_running(self):
            return False

        def start(self):
            if start_error:
                raise RuntimeError("launchctl bootstrap failed: disabled")
            return True

        def wait_for_ready(self, **kwargs):
            return True

    class Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        async def list_gpus(self):
            return []

    console = _RecordingConsole()
    monkeypatch.setattr(init_mod, "console", console)
    monkeypatch.setattr("nerdit.daemon.lifecycle.DaemonLifecycle", Life)
    monkeypatch.setattr("nerdit.cli.client.NerditClient", Client)
    monkeypatch.setattr(
        init_mod, "_report_daemon_start_failure", lambda *a: pytest.fail("masked error")
    )
    if start_error:
        with pytest.raises(typer.Exit) as exc:
            await init_mod._init_async()
        assert exc.value.exit_code == 1
        assert "bootstrap failed: disabled" in "\n".join(console.lines)
        assert "client" not in seen
    else:
        await init_mod._init_async()
        assert seen["client"] == {"host": "127.0.0.1", "port": 9444, "token": None}
        assert "http://127.0.0.1:9444/" in "\n".join(console.lines)
    assert seen["lifecycle"] == {"host": "127.0.0.1", "port": 9444, "pid_file": str(pid_file)}


async def test_source_checkout_config_is_owner_only(monkeypatch, tmp_path):
    import stat

    config = await TestInitOnAnInstallerMadeInstall._run(monkeypatch, tmp_path, installed=False)
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
