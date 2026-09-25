"""Durable file writes: directory fsync and the atomic 0600 writer."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename or create inside it survives power loss.

    Raises:
        OSError: When the directory cannot be opened or fsynced.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, payload: bytes, *, sync_dir: bool = True) -> None:
    """Write *payload* to *path* atomically and durably at `0o600`.

    Stages a sibling `<name>.tmp-<pid>` (O_EXCL, `0o600`, so there is never a
    world-readable window; `os.replace` adopts the tmp inode's mode), fsyncs
    it, then renames it over *path*: a crash leaves the old content intact.
    The tmp name is per-process, so callers serialize writers to one path.

    `sync_dir=False` skips only the parent-directory fsync, for a batch writer
    that fsyncs each touched directory once at the end.
    """
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if sync_dir:
        fsync_dir(path.parent)
