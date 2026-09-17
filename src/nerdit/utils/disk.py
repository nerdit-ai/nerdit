"""Disk-usage accounting and protected archive-directory resolution."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def spawn_walk(fn: Callable[[], Any]) -> asyncio.Future[Any]:
    """Run a blocking walk on a dedicated **daemon** thread; resolve a Future.

    Deliberately NOT `asyncio.to_thread`: that borrows the loop's *default*
    executor, and a walk abandoned past a soft budget would stay alive in it —
    `asyncio.run` teardown then joins that executor, which under uvloop blocks
    **unboundedly**, wedging the `/daemon/restart` re-exec (live-run leg 8). A
    daemon thread is joined by nothing — an overrunning walk dies with the
    process/execv instead of holding the loop hostage.

    Shared by every route that walks a data tree (`/system/disk`, the service
    detail route's `data_dir_bytes`) so none of them reaches for `to_thread`
    and starves the pool `DockerRuntime` and the reconciler depend on.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[Any] = loop.create_future()

    def _resolve(result: Any, exc: BaseException | None) -> None:
        if fut.done():  # runs on the loop thread; cancelled walks are dropped
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    def _runner() -> None:
        try:
            result, exc = fn(), None
        except BaseException as e:  # noqa: BLE001 — marshalled to the Future
            result, exc = None, e
        # A closed loop means the walk outlived the daemon: nothing to resolve.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(_resolve, result, exc)

    threading.Thread(target=_runner, name="nerdit-du-walk", daemon=True).start()
    return fut


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
