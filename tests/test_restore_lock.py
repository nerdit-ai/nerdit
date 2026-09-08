"""Unit tests for the daemon's held ``.restore.lock`` flock (P14c WP7 / D6-H1).

These exercise ``_acquire_restore_lock`` directly against a ``tmp_path`` data
dir — no TestClient, no lifespan. On Linux ``flock`` is per open file
description: two separate ``open()`` calls (even in the same process) get
distinct descriptions and DO conflict, so a same-process second fd is a valid
contender. Verified empirically by these tests.
"""

from __future__ import annotations

import fcntl
import os

import pytest

from nerdit.daemon.server import _acquire_restore_lock


def _try_exclusive_nb(path) -> bool:
    """Return True iff a fresh fd can grab ``LOCK_EX|LOCK_NB`` on *path*."""
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_free_file_acquires_and_holds_shared_lock(tmp_path):
    """A free lock file is acquired; the returned fd holds ``LOCK_SH``."""
    fd = _acquire_restore_lock(tmp_path)
    try:
        lock_path = tmp_path / ".restore.lock"
        assert lock_path.is_file()
        # 0o600 perms.
        assert (lock_path.stat().st_mode & 0o777) == 0o600
        # A shared lock is held → a separate LOCK_SH succeeds...
        shared_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(shared_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(shared_fd, fcntl.LOCK_UN)
        finally:
            os.close(shared_fd)
        # ...but an exclusive LOCK_EX|LOCK_NB is refused (the held LOCK_SH blocks it).
        assert not _try_exclusive_nb(lock_path)
    finally:
        os.close(fd)


def test_second_exclusive_holder_raises_runtime_error(tmp_path):
    """An existing exclusive holder makes the boot acquire fail hard."""
    lock_path = tmp_path / ".restore.lock"
    contender = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="nerdit restore"):
            _acquire_restore_lock(tmp_path)
    finally:
        os.close(contender)

    # Once the contender releases (fd closed), the daemon can acquire cleanly.
    fd = _acquire_restore_lock(tmp_path)
    os.close(fd)


def test_shared_holder_blocks_later_exclusive_attempt(tmp_path):
    """A LOCK_SH holder blocks a later LOCK_EX|LOCK_NB — mirrors a live daemon
    refusing an offline restore's exclusive grab."""
    fd = _acquire_restore_lock(tmp_path)
    try:
        # The daemon (fd) holds LOCK_SH; a restore's exclusive attempt fails.
        assert not _try_exclusive_nb(tmp_path / ".restore.lock")
    finally:
        os.close(fd)
    # After the daemon releases, the exclusive grab succeeds.
    assert _try_exclusive_nb(tmp_path / ".restore.lock")
