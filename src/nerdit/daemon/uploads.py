"""Extract ZIP uploads with streamed size, zip-bomb and traversal guards.

Extraction failures remove partial trees. Callers supply the destination ID
and clean up returned directories if their later processing fails.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import IO

from fastapi import UploadFile

from nerdit.daemon.errors import NerditError

# 1 MiB chunks keep the event loop responsive while honoring the configurable
# `max_upload_bytes` cap without buffering the full body.
_UPLOAD_CHUNK_SIZE = 1 * 1024 * 1024
# SpooledTemporaryFile keeps small uploads in memory, spills to disk past this.
_UPLOAD_SPOOL_MAX = 10 * 1024 * 1024


def _validate_and_extract(tmp: IO[bytes], upload_dir: Path, max_bytes: int) -> None:
    """Validate and extract a spooled ZIP in a worker thread.

    ZIP checks, size sums, resolved traversal checks and extraction all block on the
    filesystem. Raise structured errors; the caller owns cleanup.
    """
    tmp.seek(0)
    if not zipfile.is_zipfile(tmp):
        raise NerditError(
            400,
            "deploy.invalid_zip",
            "The uploaded file is not a valid ZIP archive.",
        )

    tmp.seek(0)
    try:
        with zipfile.ZipFile(tmp, "r") as zf:
            # Zip bomb protection: check total uncompressed size.
            max_uncompressed = max_bytes * 10  # 10x compressed size limit
            total_size = sum(info.file_size for info in zf.infolist())
            if total_size > max_uncompressed:
                raise NerditError(
                    413,
                    "payload_too_large",
                    f"Archive expands to {total_size} bytes uncompressed; "
                    f"the limit is {max_uncompressed} bytes.",
                )

            # Path traversal protection: reject absolute paths or ..
            for member in zf.infolist():
                member_path = Path(member.filename)
                unsafe = NerditError(
                    400,
                    "deploy.unsafe_path",
                    f"The ZIP archive contains an unsafe path: '{member.filename}'.",
                )
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise unsafe
                # Ensure extraction stays within upload_dir.
                resolved = (upload_dir / member.filename).resolve()
                if not str(resolved).startswith(str(upload_dir.resolve())):
                    raise unsafe

            zf.extractall(upload_dir)
    except zipfile.BadZipFile:
        raise NerditError(400, "deploy.invalid_zip", "The ZIP archive is corrupted.")


async def extract_upload(
    archive: UploadFile,
    *,
    max_bytes: int,
    upload_root: Path,
    dest_id: str,
) -> Path:
    """Stream *archive* to disk and extract it into `upload_root / dest_id`.

    Enforces the `max_bytes` cap while streaming, validates the ZIP, guards
    against zip bombs (10x the compressed cap) and path traversal, and returns
    the extraction directory. On any extraction failure the partial directory is
    `rmtree`'d before the error propagates, so a failed upload never leaks a
    tree. Validation errors are raised as `NerditError` with English
    messages and domain codes (`deploy.invalid_zip` / `deploy.unsafe_path`
    for the 400s, `payload_too_large` for the size caps).

    Only the streaming byte-cap loop runs on the event loop (it awaits the
    body); every blocking step — validation, the traversal walk, extraction —
    is delegated to `_validate_and_extract` in a worker thread.
    """
    tmp = tempfile.SpooledTemporaryFile(max_size=_UPLOAD_SPOOL_MAX)
    try:
        total = 0
        while True:
            chunk = await archive.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise NerditError(
                    413,
                    "payload_too_large",
                    f"Archive too large ({total} bytes); the limit is {max_bytes} bytes.",
                )
            tmp.write(chunk)

        upload_dir = upload_root / dest_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        try:
            # The whole blocking half runs off the loop (see _validate_and_extract);
            # a NerditError raised in the worker propagates unchanged.
            await asyncio.to_thread(_validate_and_extract, tmp, upload_dir, max_bytes)
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, upload_dir, ignore_errors=True)
            raise
        return upload_dir
    finally:
        tmp.close()


#: Cap on one archive member read into memory by `read_upload_member`.
_MEMBER_MAX_BYTES = 1 * 1024 * 1024


def _read_member(tmp: IO[bytes], member: str) -> bytes | None:
    tmp.seek(0)
    if not zipfile.is_zipfile(tmp):
        raise NerditError(
            400, "deploy.invalid_zip", "The uploaded file is not a valid ZIP archive."
        )
    tmp.seek(0)
    try:
        with zipfile.ZipFile(tmp, "r") as zf:
            try:
                info = zf.getinfo(member)
            except KeyError:
                return None
            if info.file_size > _MEMBER_MAX_BYTES:
                raise NerditError(
                    413,
                    "payload_too_large",
                    f"'{member}' is larger than {_MEMBER_MAX_BYTES} bytes.",
                )
            return zf.read(info)
    except zipfile.BadZipFile:
        raise NerditError(400, "deploy.invalid_zip", "The ZIP archive is corrupted.") from None


async def read_upload_member(archive: UploadFile, member: str, *, max_bytes: int) -> bytes | None:
    """Read one root member out of an uploaded ZIP without extracting anything.

    `apply_project` reads the declaration this way so every refusal lands
    before the first build context exists (D-P40-12). The multipart parser has
    already spooled the upload, so this only seeks; the archive is rewound for
    the `extract_upload` calls that follow.

    Args:
        archive: The spooled multipart upload.
        member: The member's path inside the archive, e.g. `nerdit.toml`.
        max_bytes: The upload cap `extract_upload` enforces; refused here too
            so an oversized archive costs no extraction.

    Returns:
        The member's bytes (at most 1 MiB), or `None` when the archive has none.

    Raises:
        NerditError: 400 `deploy.invalid_zip`, 413 `payload_too_large`.
    """
    if archive.size is not None and archive.size > max_bytes:
        raise NerditError(
            413,
            "payload_too_large",
            f"Archive too large ({archive.size} bytes); the limit is {max_bytes} bytes.",
        )
    try:
        return await asyncio.to_thread(_read_member, archive.file, member)
    finally:
        await archive.seek(0)
