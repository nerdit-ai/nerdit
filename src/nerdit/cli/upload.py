"""ZIP creation and upload helpers for the `nerdit deploy` ingress."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from nerdit.config.defaults import (
    DEFAULT_MAX_UPLOAD_BYTES,
    UPLOAD_WARNING_BYTES,
    matches_exclude_pattern,
)


def should_exclude(rel_path: str) -> bool:
    """Check if a relative path matches any exclusion pattern.

    Exact-case by construction: what `nerdit deploy` uploads is unchanged.
    The rule itself lives in `nerdit.config.defaults.matches_exclude_pattern`,
    shared with the P29 workspace write guard (which folds case on top).
    """
    return matches_exclude_pattern(Path(rel_path).parts)


def create_dir_zip(directory: str | Path) -> bytes:
    """Create a ZIP archive of a whole directory.

    Zips the directory contents recursively for `nerdit deploy`. Arcnames are
    relative to the directory root and `ZIP_EXCLUDE_PATTERNS` apply.
    """
    base_dir = Path(directory).resolve()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(base_dir.rglob("*")):
            if not file.is_file():
                continue
            rel = str(file.relative_to(base_dir))
            if should_exclude(rel):
                continue
            zf.write(file, rel)

    return buf.getvalue()


def check_upload_size(
    zip_bytes: bytes,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> None:
    """Raise ValueError if zip_bytes exceed the maximum upload size."""
    if len(zip_bytes) > max_bytes:
        size_mb = len(zip_bytes) / (1024 * 1024)
        limit_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"Archive is {size_mb:.0f} MB, limit is {limit_mb:.0f} MB.")


def warn_large_upload(zip_bytes: bytes) -> str | None:
    """Return a warning message if zip_bytes exceed the warning threshold."""
    if len(zip_bytes) > UPLOAD_WARNING_BYTES:
        size_mb = len(zip_bytes) / (1024 * 1024)
        return f"Archive is {size_mb:.0f} MB — upload may take a while."
    return None
