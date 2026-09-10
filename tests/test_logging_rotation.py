"""P14b WP-B2c — daemon log rotation, boot-log retarget, ftruncate guard.

Unit-level coverage for ``setup_logging`` rotation/perms, the
``nerditd.boot.log`` open-mode + truncation + one-generation rotation in
``DaemonLifecycle.start``, and the startup ``ftruncate`` regular-file guard in
``server.py``.
"""

import logging
import os
import stat
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

import pytest

from nerdit.utils.logging import setup_logging


@pytest.fixture(autouse=True)
def _no_real_service_start(monkeypatch):
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_service_unit", lambda: None)
    monkeypatch.setattr("nerdit.daemon.lifecycle.detect_install_layout", lambda: None)


@contextmanager
def _isolated_logging():
    """Run a ``setup_logging`` call against a clean root (basicConfig is a no-op
    when the root already has handlers — pytest installs some), then restore the
    root + uvicorn logger handlers so other tests are untouched."""
    root = logging.getLogger()
    uv = logging.getLogger("uvicorn")
    uva = logging.getLogger("uvicorn.access")
    saved = (root.handlers[:], root.level, uv.handlers[:], uva.handlers[:])
    root.handlers = []
    uv.handlers = []
    uva.handlers = []
    try:
        yield
    finally:
        for handler in root.handlers:
            if handler not in saved[0]:
                handler.close()
        root.handlers, root.level, uv.handlers, uva.handlers = saved


def _rotating_handlers():
    return [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]


def _stub_spawn(monkeypatch, lc_mod) -> dict:
    """Patch out the real Popen/sleep in ``DaemonLifecycle.start`` and return the
    dict the fake Popen records its stdout/stderr/env into."""
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    def _fake_popen(args, stdout=None, stderr=None, start_new_session=None, env=None):
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(lc_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(lc_mod.time, "sleep", lambda *_: None)
    return captured


def test_setup_logging_file_created_0600(tmp_path):
    log_file = tmp_path / "nerditd.log"
    with _isolated_logging():
        setup_logging("info", log_file=str(log_file), max_bytes=4096, backup_count=3)
        logging.getLogger("nerdit.test").warning("hello")
        assert log_file.exists()
        assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
        # exactly one rotating handler, teed alongside stderr
        rot = _rotating_handlers()
        assert len(rot) == 1
        # uvicorn / uvicorn.access (propagate=False) got the handler directly
        assert rot[0] in logging.getLogger("uvicorn").handlers
        assert rot[0] in logging.getLogger("uvicorn.access").handlers


def test_setup_logging_rotation_backup_0600(tmp_path):
    log_file = tmp_path / "nerditd.log"
    with _isolated_logging():
        setup_logging("info", log_file=str(log_file), max_bytes=64, backup_count=3)
        for i in range(40):
            logging.getLogger("nerdit.test").warning("padding message number %d here", i)
        backup = tmp_path / "nerditd.log.1"
        assert backup.exists()
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
        # the live file too
        assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


def test_setup_logging_no_file_handler_when_zero(tmp_path):
    log_file = tmp_path / "nerditd.log"
    with _isolated_logging():
        setup_logging("info", log_file=str(log_file), max_bytes=0)
        assert _rotating_handlers() == []
        assert not log_file.exists()
        # uvicorn loggers untouched when file logging is off
        assert logging.getLogger("uvicorn").handlers == []
        assert logging.getLogger("uvicorn.access").handlers == []


def test_truncate_if_regular_file_truncates(tmp_path):
    from nerdit.daemon.server import _truncate_if_regular_file

    target = tmp_path / "boot.log"
    target.write_bytes(b"x" * 512)
    fd = os.open(str(target), os.O_WRONLY)
    try:
        _truncate_if_regular_file(fd)
    finally:
        os.close(fd)
    assert target.stat().st_size == 0


def test_truncate_if_regular_file_skips_pipe():
    from nerdit.daemon.server import _truncate_if_regular_file

    read_fd, write_fd = os.pipe()
    try:
        # A pipe fstat reports non-regular → no ftruncate attempt, no raise.
        _truncate_if_regular_file(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_lifecycle_boot_log_mode_perms_and_truncation(tmp_path, monkeypatch):
    from nerdit.daemon import lifecycle as lc_mod

    pid_file = tmp_path / "nerditd.pid"
    boot_log = tmp_path / "nerditd.boot.log"
    # Pre-seed with stale content + loose perms to prove "w" truncation + 0600.
    boot_log.write_text("stale content from a prior spawn")
    boot_log.chmod(0o644)

    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    def _fake_popen(args, stdout=None, stderr=None, start_new_session=None, env=None):
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["env"] = env
        return _FakeProc()

    monkeypatch.setattr(lc_mod.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(lc_mod.time, "sleep", lambda *_: None)

    dl = lc_mod.DaemonLifecycle(pid_file=str(pid_file))
    assert dl.start() is True

    # boot.log opened "w" (truncated) + chmod 0600
    assert boot_log.read_text() == ""
    assert stat.S_IMODE(boot_log.stat().st_mode) == 0o600
    # stdout and stderr share the one boot-log fd
    assert captured["stdout"] is captured["stderr"]
    # the legacy nerditd.log is no longer created by the spawn path
    assert not (tmp_path / "nerditd.log").exists()
    assert pid_file.read_text() == "4242"
    # the spawn marks stderr as OUR boot log so the daemon-side startup
    # truncate never fires in a foreign embedding (pytest capture, 2>>file)
    assert captured["env"]["NERDIT_BOOT_LOG"] == "1"


def test_lifecycle_boot_log_rotates_one_generation(tmp_path, monkeypatch):
    """(F10) A crashed run's traceback lives only in boot.log's raw stderr; the
    per-spawn ``"w"`` must not destroy it. Exactly one generation is kept at
    ``.1`` — a prior ``.1`` is overwritten, so the bound stays two files."""
    from nerdit.daemon import lifecycle as lc_mod

    pid_file = tmp_path / "nerditd.pid"
    boot_log = tmp_path / "nerditd.boot.log"
    prev_log = tmp_path / "nerditd.boot.log.1"
    prev_log.write_text("forensics from TWO spawns ago — must be overwritten")
    boot_log.write_text("Traceback (most recent call last): boom")
    boot_log.chmod(0o644)  # legacy/loose perms → the archive must still be 0600

    _stub_spawn(monkeypatch, lc_mod)
    dl = lc_mod.DaemonLifecycle(pid_file=str(pid_file))
    assert dl.start() is True

    # the crashed run's output survives, exactly one generation deep
    assert prev_log.read_text() == "Traceback (most recent call last): boom"
    assert stat.S_IMODE(prev_log.stat().st_mode) == 0o600
    # the live file is fresh for the new spawn, still 0600
    assert boot_log.read_text() == ""
    assert stat.S_IMODE(boot_log.stat().st_mode) == 0o600
    # strictly bounded: no ".2", no unbounded fan-out
    assert not (tmp_path / "nerditd.boot.log.2").exists()
    assert sorted(p.name for p in tmp_path.glob("nerditd.boot.log*")) == [
        "nerditd.boot.log",
        "nerditd.boot.log.1",
    ]


def test_lifecycle_boot_log_empty_is_not_rotated(tmp_path, monkeypatch):
    """A zero-byte boot.log carries no forensics — rotating it would clobber a
    genuinely useful ``.1``."""
    from nerdit.daemon import lifecycle as lc_mod

    pid_file = tmp_path / "nerditd.pid"
    boot_log = tmp_path / "nerditd.boot.log"
    prev_log = tmp_path / "nerditd.boot.log.1"
    boot_log.write_text("")  # a clean prior run left nothing to keep
    prev_log.write_text("real forensics — a zero-byte spawn must not clobber this")

    _stub_spawn(monkeypatch, lc_mod)
    assert lc_mod.DaemonLifecycle(pid_file=str(pid_file)).start() is True

    assert prev_log.read_text() == "real forensics — a zero-byte spawn must not clobber this"
    assert boot_log.read_text() == ""


def test_lifecycle_boot_log_rotation_failure_does_not_block_start(tmp_path, monkeypatch):
    """Rotation is best-effort: a read-only dir / rename race must never prevent
    the daemon from starting."""
    from nerdit.daemon import lifecycle as lc_mod

    pid_file = tmp_path / "nerditd.pid"
    boot_log = tmp_path / "nerditd.boot.log"
    boot_log.write_text("stale")

    def _boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(lc_mod.os, "replace", _boom)
    _stub_spawn(monkeypatch, lc_mod)
    assert lc_mod.DaemonLifecycle(pid_file=str(pid_file)).start() is True

    # start still proceeded to open("w"): the live file is fresh + 0600
    assert boot_log.read_text() == ""
    assert stat.S_IMODE(boot_log.stat().st_mode) == 0o600
    assert pid_file.read_text() == "4242"


def test_truncate_if_regular_file_rewinds_offset(tmp_path):
    """After truncate the fd offset must be 0, or the next write re-extends the
    file with a NUL hole (execv-inherited fds keep their old offset)."""
    from nerdit.daemon.server import _truncate_if_regular_file

    target = tmp_path / "boot.log"
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.write(fd, b"x" * 1000)  # offset now 1000
        _truncate_if_regular_file(fd)
        os.write(fd, b"fresh")
    finally:
        os.close(fd)
    assert target.stat().st_size == 5
    assert target.read_bytes() == b"fresh"


def test_setup_logging_silences_httpx_request_urls(tmp_path):
    """httpx's INFO "HTTP Request: <method> <url>" line carries the full URL.

    On the hosted app path (P26, ``core/link/mux.py``) that URL is a third
    party's — path and query may be a magic link — so the mux redacts its own
    log sites and ``setup_logging`` must keep the client library from
    re-leaking it (found on the WP-H live run, not by the mocked suite).
    """
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    setup_logging(level="debug", log_file=str(tmp_path / "n.log"), max_bytes=0)
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
