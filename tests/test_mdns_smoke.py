"""Gated real-mDNS smoke (P9, part A): register + resolve round-trip.

Skips when the ``zeroconf`` extra is not installed (same gating posture as
the other real-infra smokes — ``pytest.importorskip``, no custom marker).
Runs on loopback so it is isolated from the real LAN and needs no privileges.
"""

import asyncio

import pytest

zeroconf = pytest.importorskip("zeroconf")

from nerdit.config.settings import ProxySettings  # noqa: E402
from nerdit.core.dns import MdnsAdvertiser  # noqa: E402

pytestmark = pytest.mark.asyncio


async def test_smoke_register_and_resolve_local_name():
    from zeroconf import IPVersion
    from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

    adv = MdnsAdvertiser(
        ProxySettings(enabled=True, mdns=True, https_port=8443),
        hostname="nerdit-smoketest.local",
        address="127.0.0.1",
        interfaces=["127.0.0.1"],
    )
    await adv.start()
    if not adv.active:
        pytest.skip("mDNS advertiser could not start in this environment")
    try:
        resolver = AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=["127.0.0.1"])
        try:
            info = AsyncServiceInfo("_https._tcp.local.", "nerdit-smoketest._https._tcp.local.")
            found = await info.async_request(resolver.zeroconf, timeout=3000)
            assert found, "service did not resolve over mDNS"
            assert info.server == "nerdit-smoketest.local."
            assert info.port == 8443
            assert "127.0.0.1" in info.parsed_addresses()
        finally:
            await resolver.async_close()
    finally:
        await adv.stop()
    assert adv.active is False


async def test_smoke_stop_withdraws_cleanly():
    adv = MdnsAdvertiser(
        ProxySettings(enabled=True, mdns=True, https_port=8443),
        hostname="nerdit-smoketest2.local",
        address="127.0.0.1",
        interfaces=["127.0.0.1"],
    )
    await adv.start()
    if not adv.active:
        pytest.skip("mDNS advertiser could not start in this environment")
    await adv.stop()
    await adv.stop()  # idempotent
    await asyncio.sleep(0)  # let the goodbye packet task settle
    assert adv.active is False
