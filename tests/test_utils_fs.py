"""`utils.fs.atomic_write`: 0600, atomic replace, no tmp residue."""

from __future__ import annotations

import asyncio
import os
import stat

import pytest

from nerdit.utils.fs import atomic_write


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_write_replaces_at_0600_without_residue(tmp_path):
    target = tmp_path / "f.toml"
    target.write_bytes(b"old")
    target.chmod(0o644)
    atomic_write(target, b"new")
    assert target.read_bytes() == b"new"
    assert _mode(target) == 0o600
    assert not list(tmp_path.glob("*.tmp-*"))


def test_failed_replace_keeps_old_content_and_no_tmp(tmp_path, monkeypatch):
    target = tmp_path / "f.toml"
    target.write_bytes(b"old")

    def _boom(*_a, **_k):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        atomic_write(target, b"new")
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_connect_writes_client_config_0600(tmp_path, monkeypatch):
    """S4: `nerdit connect` stores the auth token in a 0600 file."""
    from nerdit.cli.client import NerditClient
    from nerdit.cli.commands import connect

    monkeypatch.setenv("HOME", str(tmp_path))

    async def _health(self):
        return {"version": "t", "gpu_count": 0}

    monkeypatch.setattr(NerditClient, "health", _health)
    asyncio.run(connect._connect_async("h", 9321, "tok"))
    assert _mode(tmp_path / ".nerdit" / "config.toml") == 0o600
