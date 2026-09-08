"""Tests for the daemon startup security warning."""

from __future__ import annotations

import logging

import pytest

from nerdit.config.settings import DaemonSettings, ProxySettings
from nerdit.daemon.server import _warn_if_unauthenticated_exposure


def _apex_proxy(**kw) -> ProxySettings:
    base = {"enabled": True, "mode": "path", "dashboard_apex": True}
    base.update(kw)
    return ProxySettings(**base)


def test_warn_when_no_token_and_non_localhost(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="0.0.0.0", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon)
    assert any("auth_token" in rec.message for rec in caplog.records)
    assert any(rec.levelno == logging.WARNING for rec in caplog.records)


def test_no_warn_when_token_set(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="0.0.0.0", auth_token="secret")
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon)
    assert caplog.records == []


def test_no_warn_when_bound_to_loopback(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="127.0.0.1", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon)
    assert caplog.records == []


def test_no_warn_when_bound_to_localhost_name(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="localhost", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon)
    assert caplog.records == []


# P9.5: the dashboard apex re-exposes the daemon API on the proxy's LAN listener
# even when the daemon is bound to loopback, so the host-only check misses it.


def test_warn_when_apex_on_and_no_token_even_on_loopback(
    caplog: pytest.LogCaptureFixture,
):
    daemon = DaemonSettings(host="127.0.0.1", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon, _apex_proxy())
    assert any("dashboard_apex" in rec.message for rec in caplog.records)


def test_no_warn_when_apex_on_but_token_set(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="127.0.0.1", auth_token="secret")
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon, _apex_proxy())
    assert caplog.records == []


def test_no_warn_when_apex_off_and_no_token(caplog: pytest.LogCaptureFixture):
    daemon = DaemonSettings(host="127.0.0.1", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon, _apex_proxy(dashboard_apex=False))
    assert caplog.records == []


def test_no_warn_when_apex_flag_on_but_proxy_disabled(
    caplog: pytest.LogCaptureFixture,
):
    daemon = DaemonSettings(host="127.0.0.1", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(daemon, _apex_proxy(enabled=False))
    assert caplog.records == []


def test_no_warn_when_apex_flag_on_but_subdomain_mode(
    caplog: pytest.LogCaptureFixture,
):
    # Subdomain mode leaves the apex routeless (the flag is a no-op) — no exposure.
    daemon = DaemonSettings(host="127.0.0.1", auth_token=None)
    with caplog.at_level(logging.WARNING, logger="nerdit.daemon.server"):
        _warn_if_unauthenticated_exposure(
            daemon, _apex_proxy(mode="subdomain", base_domain="apps.lan")
        )
    assert caplog.records == []
