"""Unit tests for du_bytes (P14 WP-0) + resolve_archive_dir (P14b follow-up F14).

``resolve_archive_dir`` moved here from ``daemon/server.py`` so the
``GET /system/disk`` report can walk the same dir the retention sweep writes to
without a route → server import cycle. ``tests/test_retention.py`` keeps
exercising it through the ``daemon.server`` re-export (the sweep's call site).
"""

from __future__ import annotations

import os
from pathlib import Path

from nerdit.config.settings import RetentionSettings
from nerdit.utils.disk import du_bytes, resolve_archive_dir


def test_missing_path_is_zero(tmp_path: Path):
    assert du_bytes(tmp_path / "does-not-exist") == 0


def test_empty_dir_is_zero(tmp_path: Path):
    assert du_bytes(tmp_path) == 0


def test_sums_file_sizes(tmp_path: Path):
    (tmp_path / "a.txt").write_bytes(b"x" * 100)
    (tmp_path / "b.txt").write_bytes(b"y" * 50)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_bytes(b"z" * 25)
    assert du_bytes(tmp_path) == 175


def test_symlinks_are_skipped(tmp_path: Path):
    real = tmp_path / "real.txt"
    real.write_bytes(b"a" * 200)
    link = tmp_path / "link.txt"
    os.symlink(real, link)
    # The link's target bytes are counted once (via real.txt); the symlink entry
    # itself contributes nothing.
    assert du_bytes(tmp_path) == 200


def test_symlink_to_dir_not_followed(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "big.bin").write_bytes(b"x" * 1000)
    other = tmp_path / "other"
    other.mkdir()
    os.symlink(target, other / "linkdir")
    # Walking `other` must not descend into the symlinked dir.
    assert du_bytes(other) == 0


# --- resolve_archive_dir (F14: shared by the sweep and the disk report) --------


def test_resolve_archive_dir_defaults_under_data_dir(tmp_path: Path):
    assert resolve_archive_dir(RetentionSettings(), tmp_path) == tmp_path / "archive"


def test_resolve_archive_dir_honors_custom_absolute_path(tmp_path: Path):
    custom = tmp_path / "elsewhere" / "archive"
    resolved = resolve_archive_dir(
        RetentionSettings(audit_archive_dir=str(custom)), tmp_path / "data"
    )
    assert resolved == custom


def test_resolve_archive_dir_rejects_under_services(tmp_path: Path):
    # <data_dir>/services is an rmtree scope (purge=data, GC orphan-data) — the
    # tamper-evidence trail must never live there, so archiving is DISABLED.
    evil = tmp_path / "services" / "app" / "arch"
    assert resolve_archive_dir(RetentionSettings(audit_archive_dir=str(evil)), tmp_path) is None


def test_resolve_archive_dir_rejects_under_secrets(tmp_path: Path):
    evil = tmp_path / "secrets" / "arch"
    assert resolve_archive_dir(RetentionSettings(audit_archive_dir=str(evil)), tmp_path) is None
