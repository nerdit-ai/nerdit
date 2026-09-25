"""Daemon lifecycle management — start/stop/check nerditd."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from nerdit.config.defaults import DEFAULT_HOST, DEFAULT_PORT
from nerdit.utils.install_layout import (
    detect_install_layout,
    detect_service_unit,
    managed_daemon_port,
    start_service_unit,
)

logger = logging.getLogger(__name__)


class DaemonLifecycle:
    """Manage the nerditd daemon process."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        pid_file: str = "~/.nerdit/nerditd.pid",
    ) -> None:
        self._port = port
        self._pid_file = Path(pid_file).expanduser()
        self._base_url = f"http://{host}:{port}"

    def is_running(self) -> bool:
        """Check if the daemon is running (PID file + process check)."""
        if not self._pid_file.exists():
            return False
        try:
            pid = int(self._pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return True
        except PermissionError:
            # EPERM means the pid exists under another user: never unlink a
            # pid file for a process that may still be our daemon.
            return True
        except (ValueError, ProcessLookupError):
            self._pid_file.unlink(missing_ok=True)
            return False

    def _spawn_argv(self) -> list[str]:
        """The argv that starts a daemon, resolved once per spawn. A frozen install has no
        interpreter and no importable
        `nerdit.daemon.server`: the daemon is the sibling `nerditd`
        executable inside the same PyInstaller onedir bundle as this `nerdit`
        binary. A missing sibling is a broken install, not something to paper
        over with an interpreter form that cannot work — so it raises here,
        loudly, before any pid file or log is touched.
        """
        if getattr(sys, "frozen", False):
            nerditd = Path(sys.executable).parent / "nerditd"
            if not nerditd.exists():
                raise RuntimeError(
                    f"Frozen install is incomplete: no 'nerditd' next to {sys.executable} "
                    f"(expected {nerditd}). Reinstall with the get.nerdit.ai installer."
                )
            return [str(nerditd)]
        return [sys.executable, "-m", "nerdit.daemon.server"]

    def start(self) -> bool:
        """Start through the installed service, or detach a source-install daemon."""
        if self.is_running():
            logger.info("Daemon already running")
            return True

        unit = detect_service_unit()
        if unit is not None:
            if managed_daemon_port(unit) != self._port:
                raise RuntimeError(
                    "Installed service configuration does not match this daemon's port. "
                    "Check the service unit and ~/.nerdit/config.toml."
                )
            start_service_unit(unit)
            return True
        if detect_install_layout() is not None:
            raise RuntimeError("The installed daemon service is missing. Re-run the installer.")

        argv = self._spawn_argv()
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        # The child's raw stdout/stderr (tracebacks, uvicorn boot lines)
        # go to boot.log, truncated per spawn ("w"); the daemon's app loggers
        # land in the rotating <data_dir>/nerditd.log. 0600 — it can carry
        # tracebacks / deploy details.
        boot_log = self._pid_file.parent / "nerditd.boot.log"
        prev_log = boot_log.with_name(boot_log.name + ".1")

        # (F10) Keep exactly ONE generation of prior-run forensics. A crashed
        # run's traceback lands only in this file's raw stderr — nowhere else,
        # since an early-boot death can precede setup_logging entirely — and the
        # "w" below would destroy it on the user's very next `daemon start`.
        # Rotating a non-empty boot.log to ".1" (a prior ".1" is overwritten,
        # never chained to ".2") restores that forensics while preserving the
        # truncate-per-spawn bound: at most two files, each holding exactly one
        # spawn's output. An EMPTY boot.log is skipped — a clean prior run has
        # nothing to keep, and archiving it would clobber a useful ".1".
        # os.replace is atomic and carries the mode through the rename; the
        # chmod re-asserts 0600 for a loose-perms file inherited from a pre-P14b
        # install, so a traceback is never archived group/world-readable.
        # Best-effort throughout: a read-only dir or a concurrent start must
        # never prevent the daemon from starting.
        try:
            if boot_log.is_file() and boot_log.stat().st_size > 0:
                os.replace(boot_log, prev_log)
                os.chmod(prev_log, 0o600)
        except OSError:
            pass

        def _opener(path: str, flags: int) -> int:
            return os.open(path, flags, 0o600)

        log_fh = open(boot_log, "w", opener=_opener)
        os.chmod(boot_log, 0o600)  # enforce 0600 on a pre-existing file too
        try:
            proc = subprocess.Popen(
                argv,
                stdout=log_fh,
                stderr=log_fh,
                start_new_session=True,
                # Marks stderr as OUR boot log so the daemon's startup truncate
                # (bounding it across restart self-execs) never fires in a
                # foreign embedding — pytest capture, `nerditd 2>>file`, CI.
                # execv preserves env, so the /daemon/restart self-exec keeps it.
                env={**os.environ, "NERDIT_BOOT_LOG": "1"},
            )
        except Exception:
            log_fh.close()
            raise

        # Close the file handle — the child process inherited the fd
        log_fh.close()

        # Check that the process didn't die immediately
        time.sleep(0.5)
        if proc.poll() is not None:
            # The most common cause of an instant exit is a busy port. Probe it
            # so non-init callers (backup/restore restart) get the specific
            # "Address already in use" diagnosis, not just "check the log".
            if self._port_in_use():
                logger.error(
                    "Daemon exited immediately (code=%s): port %d is already in use "
                    "(Address already in use). Stop the process using it or set "
                    "[daemon].port. Check %s",
                    proc.returncode,
                    self._port,
                    boot_log,
                )
            else:
                logger.error(
                    "Daemon exited immediately (code=%s). Check %s", proc.returncode, boot_log
                )
            return False

        self._pid_file.write_text(str(proc.pid))
        logger.info("Daemon started with PID %d (log: %s)", proc.pid, boot_log)
        return True

    def _port_in_use(self) -> bool:
        """Best-effort: True when our port is already bound by another process.

        Connect-first: on macOS a `SO_REUSEADDR` bind *succeeds* over another
        `SO_REUSEADDR` listener (uvicorn sets it), so a bind probe alone
        misses a live daemon there. The bind fallback (with `SO_REUSEADDR` so
        a `TIME_WAIT` leftover is not misread as live) still catches
        non-loopback binds a loopback connect cannot see.
        """
        try:
            with socket.create_connection(("127.0.0.1", self._port), timeout=0.5):
                return True
        except OSError:
            pass
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self._port))
        except OSError:
            return True
        finally:
            sock.close()
        return False

    def wait_for_ready(self, timeout: float = 10.0, interval: float = 0.3) -> bool:
        """Poll /health until the daemon is ready. Returns True if ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{self._base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(interval)
        return False
