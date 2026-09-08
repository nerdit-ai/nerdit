"""``restore_host_loader_env`` — children of the frozen build see the host's
loader path, not the bundle's (E2E 2026-08-23: system ``curl`` on Ubuntu 26.04
loaded the bundle's ``libssl.so.3`` and died on ``OPENSSL_3.2.0``)."""

from __future__ import annotations

import sys

import pytest

from nerdit.utils.frozen import restore_host_loader_env


def test_noop_on_a_source_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    env = {"LD_LIBRARY_PATH": "/opt/nerdit/_internal", "LD_LIBRARY_PATH_ORIG": "/x"}
    assert restore_host_loader_env(env) is False
    assert env == {"LD_LIBRARY_PATH": "/opt/nerdit/_internal", "LD_LIBRARY_PATH_ORIG": "/x"}


def test_restores_the_callers_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    env = {"LD_LIBRARY_PATH": "/opt/nerdit/_internal:/x", "LD_LIBRARY_PATH_ORIG": "/x"}
    assert restore_host_loader_env(env) is True
    assert env == {"LD_LIBRARY_PATH": "/x"}


def test_unsets_when_the_caller_had_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    env = {"LD_LIBRARY_PATH": "/opt/nerdit/_internal", "PATH": "/usr/bin"}
    assert restore_host_loader_env(env) is True
    assert env == {"PATH": "/usr/bin"}


def test_acts_once_per_process_on_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the restore the variable IS the host's; a second pass must not
    mistake it for the bootloader's and unset it."""
    import nerdit.utils.frozen as frozen

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(frozen, "_restored", False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/nerdit/_internal:/x")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/x")
    assert restore_host_loader_env() is True
    assert restore_host_loader_env() is False
    import os

    assert os.environ["LD_LIBRARY_PATH"] == "/x"
