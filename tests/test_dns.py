"""Unit tests for the mDNS advertiser (P9, part A) — no zeroconf required.

Follows the ``test_proxy.py`` posture: pure async, no TestClient, and the
optional dependency is never imported (these tests cover the gating and the
name-shaping logic; the real registration round-trip lives in the gated
``test_mdns_smoke.py``).
"""

import logging
import sys

import pytest

from nerdit.config.settings import ProxySettings
from nerdit.core.dns import (
    MDNS_REASON_BAD_HOSTNAME,
    MDNS_REASON_NO_ADDRESS,
    MDNS_REASON_PROXY_DISABLED,
    MDNS_REASON_ZEROCONF_MISSING,
    MdnsAdvertiser,
    advertised_name,
    default_local_hostname,
    detect_lan_ip,
)

# -- name shaping --------------------------------------------------------------


def test_advertised_name_single_label_local():
    assert advertised_name("mybox.local") == "mybox"


@pytest.mark.parametrize(
    "hostname",
    [
        "mybox",  # not .local
        "mybox.lan",  # not .local
        "a.b.local",  # multi-label — standard mDNS cannot resolve it
        ".local",  # empty label
        "192.168.1.20",  # IP literal
    ],
)
def test_advertised_name_rejects_non_mdns_names(hostname):
    assert advertised_name(hostname) is None


def test_default_local_hostname_is_short_lowercase_local(monkeypatch):
    monkeypatch.setattr("nerdit.core.dns.socket.gethostname", lambda: "MyBox.example.com")
    assert default_local_hostname() == "mybox.local"


def test_detect_lan_ip_never_loopback():
    ip = detect_lan_ip()
    assert ip is None or not ip.startswith("127.")


# -- settings ------------------------------------------------------------------


def test_mdns_settings_default_off():
    settings = ProxySettings()
    assert settings.mdns is False
    assert settings.mdns_address is None


def test_mdns_address_accepts_lan_ipv4():
    assert ProxySettings(mdns_address="192.168.1.20").mdns_address == "192.168.1.20"


@pytest.mark.parametrize("value", ["127.0.0.1", "0.0.0.0", "not-an-ip", "192.168.1.20:443"])
def test_mdns_address_rejects_unusable_values(value):
    with pytest.raises(ValueError):
        ProxySettings(mdns_address=value)


# -- advertiser gating (start never raises, degrades to warnings) ---------------


async def test_start_is_noop_when_mdns_disabled():
    adv = MdnsAdvertiser(ProxySettings(mdns=False), hostname="mybox.local")
    await adv.start()
    assert adv.active is False


async def test_start_warns_and_noops_when_proxy_disabled(caplog):
    """mdns=true without enabled=true would advertise a name nothing listens on."""
    adv = MdnsAdvertiser(ProxySettings(enabled=False, mdns=True), hostname="mybox.local")
    with caplog.at_level(logging.WARNING, logger="nerdit.core.dns"):
        await adv.start()
    assert adv.active is False
    assert adv.reason == MDNS_REASON_PROXY_DISABLED
    assert "[proxy].enabled is off" in caplog.text


async def test_start_warns_and_noops_on_non_local_hostname(caplog):
    adv = MdnsAdvertiser(ProxySettings(enabled=True, mdns=True), hostname="mybox.example.com")
    with caplog.at_level(logging.WARNING, logger="nerdit.core.dns"):
        await adv.start()
    assert adv.active is False
    assert adv.reason == MDNS_REASON_BAD_HOSTNAME
    assert "not a single-label .local name" in caplog.text


async def test_start_warns_and_noops_on_multi_label_local(caplog):
    adv = MdnsAdvertiser(ProxySettings(enabled=True, mdns=True), hostname="svc.mybox.local")
    with caplog.at_level(logging.WARNING, logger="nerdit.core.dns"):
        await adv.start()
    assert adv.active is False


async def test_start_warns_when_no_address_detectable(monkeypatch, caplog):
    monkeypatch.setattr("nerdit.core.dns.detect_lan_ip", lambda: None)
    adv = MdnsAdvertiser(ProxySettings(enabled=True, mdns=True), hostname="mybox.local")
    with caplog.at_level(logging.WARNING, logger="nerdit.core.dns"):
        await adv.start()
    assert adv.active is False
    assert adv.reason == MDNS_REASON_NO_ADDRESS
    assert "could not determine a LAN address" in caplog.text


async def test_start_stamps_zeroconf_missing_reason(monkeypatch, caplog):
    """When the optional zeroconf extra is absent, start() records a machine-
    readable reason so /doctor can surface an actionable install hint."""
    # Force the ``from zeroconf import ...`` inside start() to raise ImportError
    # deterministically, whether or not the extra is installed in this env.
    monkeypatch.setitem(sys.modules, "zeroconf", None)
    adv = MdnsAdvertiser(
        ProxySettings(enabled=True, mdns=True, mdns_address="192.168.1.20"),
        hostname="mybox.local",
    )
    with caplog.at_level(logging.WARNING, logger="nerdit.core.dns"):
        await adv.start()
    assert adv.active is False
    assert adv.registered is False
    assert adv.reason == MDNS_REASON_ZEROCONF_MISSING
    assert "pip install 'nerdit[mdns]'" in caplog.text


async def test_stop_is_idempotent_when_never_started():
    adv = MdnsAdvertiser(ProxySettings(mdns=False), hostname="mybox.local")
    await adv.stop()
    await adv.stop()
    assert adv.active is False
