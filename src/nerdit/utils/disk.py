"""Disk-usage accounting and protected archive-directory resolution."""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    # TYPE_CHECKING-only so ``utils`` stays a runtime leaf package: a real
    # ``utils.disk -> config.settings`` edge is not circular today, but
    # ``daemon/routes/system.py`` already imports this module and adding config
    # to a util invites a future cycle.
    from nerdit.config.settings import RetentionSettings

logger = logging.getLogger(__name__)


def du_bytes(path: Path) -> int:
    """Sum regular-file sizes in bytes without following symlinks.

    Ignore per-file OS errors and return zero for a missing path.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            fp = os.path.join(dirpath, name)
            try:
                st = os.lstat(fp)
            except OSError:
                continue
            # Skip symlinks: count only bytes actually owned by this subtree.
            # Reuse the lstat result instead of a second ``islink`` syscall.
            if stat.S_ISLNK(st.st_mode):
                continue
            total += st.st_size
    return total


def resolve_archive_dir(retention: RetentionSettings, data_dir: Path) -> Path | None:
    """Resolve the audit archive directory, defaulting to <data_dir>/archive.

    Return None for paths under secrets or services: archives must remain outside
    purge/GC scopes and container-writable volumes. Disk accounting and retention
    share this resolver so custom locations stay visible.
    """
    raw = (retention.audit_archive_dir or "").strip()
    archive_dir = (Path(raw) if raw else data_dir / "archive").expanduser()
    resolved = archive_dir.resolve()
    for base in ((data_dir / "secrets").resolve(), (data_dir / "services").resolve()):
        if resolved == base or resolved.is_relative_to(base):
            logger.error(
                "audit_archive_dir resolves under a protected directory (%s); "
                "audit archiving is DISABLED (rows will not be pruned).",
                base.name,
            )
            return None
    return archive_dir
