"""Archive audit rows durably before pruning them.

Each sweep exports one immutable fsynced `.jsonl.gz` file, hashes the closed
archive, then deletes only rows through its exported ID watermark. Export
failure means no deletes; abandon partial files and retry in a fresh archive.
The pre-stream ID snapshot protects rows inserted while archiving.

Compression, fsync, and hashing use bounded, awaited thread offloads to keep
reconcile and HTTP responsive.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from nerdit.utils.fs import fsync_dir

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nerdit.db.queries import Queries

#: Delay between delete chunks so the write lock is released between them.
_CHUNK_PAUSE_S = 0.05


@dataclass
class AuditArchiveResult:
    """Outcome of one archive-and-prune pass."""

    archived: int
    deleted: int
    basename: str | None
    sha256: str | None


def _fsync_dir(path: Path) -> None:
    """fsync a directory so a newly created entry within it is durable."""
    # Some filesystems reject directory fsync; the file fsync already ran.
    with contextlib.suppress(OSError):
        fsync_dir(path)


def _flush_and_fsync(fh: IO[bytes]) -> None:
    """Flush the Python buffer and fsync the fd — the durability barrier.

    Offloaded to a thread by the caller: on a multi-MB archive this is the
    single longest blocking call of the pass. `os.fsync` is resolved through
    the module attribute on purpose (tests substitute it to force the failure
    path, and the D-H contract is asserted through that substitution).
    """
    fh.flush()
    os.fsync(fh.fileno())


def _sha256_file(path: Path) -> str:
    """Hash a closed file that will never be written again."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


async def archive_and_prune_audit(
    queries: Queries,
    archive_dir: Path,
    cutoff: str,
    *,
    chunk_size: int = 500,
    on_archived: Callable[[AuditArchiveResult], Awaitable[None]] | None = None,
) -> AuditArchiveResult:
    """Export audit rows below `cutoff` to one fsynced archive, then prune them.

    `cutoff` is the `audit_log.ts` space-format UTC string
    (`YYYY-MM-DD HH:MM:SS`). Returns the archived/deleted counts, the archive
    file basename (never an absolute path — it is recorded in the sweep audit
    row), and its sha256. Returns a zero result with `basename=None` when no
    rows are eligible (no file is created).

    `on_archived` runs after the file + directory are fsynced and hashed but
    **before the first delete** — the caller anchors the basename/sha256 in a
    surviving audit row there, so a crash mid-delete can never leave archived
    rows gone with no tamper-evidence anchor in the DB (D-H). If the callback
    raises, **zero rows are deleted** (the fsynced file is kept — its rows
    simply re-archive next sweep).

    Raises on any export failure (after abandoning the partial file); the caller
    logs and continues, and **no rows are deleted**.
    """
    min_id, max_id, count = await queries.audit_archive_range(cutoff)
    if not count or min_id is None or max_id is None:
        return AuditArchiveResult(0, 0, None, None)

    archive_dir.mkdir(parents=True, exist_ok=True)
    # `mkdir(mode=...)` is subject to umask and a no-op when the dir already
    # exists; enforce 0700 explicitly (the archive holds the tamper trail).
    with contextlib.suppress(OSError):
        os.chmod(archive_dir, 0o700)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    basename = f"audit-{ts}-{min_id}-{max_id}.jsonl.gz"
    path = archive_dir / basename

    written = 0
    last_written_id = 0
    # O_EXCL: never append to a prior sweep's file — that would stale every
    # previously recorded sha256 and kill the tamper-evidence chain (D-H).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
                after = 0
                while True:
                    rows = await queries.fetch_audit_for_archive(
                        cutoff, after_id=after, max_id=max_id, limit=chunk_size
                    )
                    if not rows:
                        break
                    # Serialize the whole chunk on the loop (cheap, no I/O), then
                    # hand ONE pre-encoded blob to the thread: compression + write
                    # is the blocking part, and one hop per chunk (≤ chunk_size
                    # rows) keeps the offload overhead well under a per-row hop —
                    # which would be slower than doing the work inline.
                    batch = b"".join(
                        (
                            json.dumps(
                                entry.model_dump(mode="json"),
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                        for entry in rows
                    )
                    # The GzipFile is used strictly sequentially — one awaited
                    # write at a time, never concurrently — so hopping threads
                    # between awaited calls is safe (no shared-state race on the
                    # zlib compressor). A failure inside the thread re-raises at
                    # this `await`, i.e. INSIDE the `except BaseException`
                    # below: the zero-deletes + unlink-the-partial contract holds
                    # across the thread boundary.
                    await asyncio.to_thread(gz.write, batch)
                    written += len(rows)
                    # Rows come back id-ascending (`fetch_audit_for_archive`),
                    # so the chunk's last row is the batch watermark.
                    last_written_id = rows[-1].id
                    after = rows[-1].id
            # The `GzipFile` exit above writes the gzip trailer (a few bytes of
            # zlib Z_FINISH + CRC) on the loop — bounded and negligible.
            await asyncio.to_thread(_flush_and_fsync, raw)
    except BaseException:
        # ANY export failure ⇒ zero deletes; abandon the partial file so it is
        # never mistaken for a complete archive (its rows re-archive next sweep).
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise

    # Both offloaded, both deliberately OUTSIDE the `try` above (as they were
    # pre-offload): at this point the file is complete and fsynced, not partial,
    # so a failure here must NOT unlink it. Zero-deletes still holds — they run
    # before the anchor callback and before every delete.
    await asyncio.to_thread(_fsync_dir, archive_dir)
    sha = await asyncio.to_thread(_sha256_file, path)

    # Anchor-before-delete (D-H): let the caller record basename + sha256 in a
    # surviving row NOW. A failure here aborts with zero deletes.
    if on_archived is not None:
        await on_archived(AuditArchiveResult(written, 0, basename, sha))

    # Watermark = last id actually fsynced. Delete chunk-looped, releasing the
    # write lock between chunks.
    deleted = 0
    while True:
        removed = await queries.delete_audit_range(last_written_id, cutoff)
        deleted += removed
        if removed == 0:
            break
        await asyncio.sleep(_CHUNK_PAUSE_S)

    return AuditArchiveResult(written, deleted, basename, sha)
