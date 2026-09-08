"""Advertise the daemon's `.local` hostname on the LAN with optional Zeroconf.

`[proxy].mdns` enables advertising; `start` loads the optional dependency and
logs failures without interrupting services. Standard mDNS resolves only the
single-label daemon hostname. Subdomain service URLs still need operator DNS.
"""

import contextlib
import logging
import socket
from typing import Any

from nerdit.config.settings import ProxySettings

logger = logging.getLogger(__name__)

_MDNS_SUFFIX = ".local"

# Machine-readable reason codes for why `MdnsAdvertiser.start` advertised
# nothing, read by `GET /doctor` (`_mdns`) to distinguish a missing optional
# dependency from a genuine registration failure. `reason` is `None` once
# advertising is live (or before `start` has run).
MDNS_REASON_PROXY_DISABLED = "proxy_disabled"
MDNS_REASON_BAD_HOSTNAME = "bad_hostname"
MDNS_REASON_NO_ADDRESS = "no_address"
MDNS_REASON_ZEROCONF_MISSING = "zeroconf_missing"
MDNS_REASON_REGISTER_FAILED = "register_failed"


def default_local_hostname() -> str:
    """The zero-config mDNS name: `<short-machine-hostname>.local`.

    Used by the daemon's hostname resolution when `[proxy].mdns` is on and
    no `hostname_override` is set, so the `public_url`, the TLS cert SAN
    and the advertised mDNS name all agree out of the box.
    """
    short = socket.gethostname().split(".")[0].lower()
    return f"{short}{_MDNS_SUFFIX}"


def advertised_name(hostname: str) -> str | None:
    """The single mDNS label to advertise for *hostname*, or `None`.

    Standard mDNS resolvers (Avahi/Bonjour/Windows) only resolve
    single-label `<name>.local` — a multi-label `a.b.local` or a
    non-`.local` hostname cannot be advertised without mismatching the
    `public_url`/TLS SAN, so we advertise nothing rather than a wrong name.
    """
    if not hostname.endswith(_MDNS_SUFFIX):
        return None
    label = hostname[: -len(_MDNS_SUFFIX)]
    if not label or "." in label:
        return None
    return label


def detect_lan_ip() -> str | None:
    """Best-effort primary IPv4 of this host (the UDP-connect trick).

    No packet is sent — `connect` on a UDP socket only asks the kernel for
    the route, so this works offline as long as a default route exists.
    Returns `None` when detection fails or resolves to loopback.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))  # TEST-NET-1: never actually reached
            ip = sock.getsockname()[0]
    except OSError:
        return None
    if ip.startswith("127.") or ip == "0.0.0.0":
        return None
    return ip


class MdnsAdvertiser:
    """Publishes `<name>.local` → the daemon's LAN address over mDNS.

    Registers a `_https._tcp.local.` service whose `server` record is the
    advertised hostname, which makes the responder answer plain A-record
    queries for `<name>.local` — the only part the `public_url` needs.
    """

    def __init__(
        self,
        settings: ProxySettings,
        *,
        hostname: str,
        address: str | None = None,
        interfaces: list[str] | None = None,
    ) -> None:
        self._settings = settings
        self._hostname = hostname
        # Explicit ctor override (tests/smokes) > [proxy].mdns_address > autodetect.
        self._address = address or settings.mdns_address
        self._interfaces = interfaces
        # AsyncZeroconf + ServiceInfo, present only while advertising. Typed
        # `Any` because the zeroconf types only exist when the extra is
        # installed (this module must import without it).
        self._aiozc: Any = None
        self._info: Any = None
        self.active = False
        # Why the last start() advertised nothing (one of the MDNS_REASON_*
        # codes), or None once advertising is live. Read by GET /doctor.
        self.reason: str | None = None

    async def start(self) -> None:
        """Begin advertising; degrades to a warning on any failure."""
        if not self._settings.mdns:
            return
        if not self._settings.enabled:
            self.reason = MDNS_REASON_PROXY_DISABLED
            logger.warning(
                "[proxy].mdns is on but [proxy].enabled is off — advertising "
                "nothing (there would be no HTTPS listener behind the "
                "advertised name). Enable the proxy to use mDNS."
            )
            return
        name = advertised_name(self._hostname)
        if name is None:
            self.reason = MDNS_REASON_BAD_HOSTNAME
            logger.warning(
                "mDNS is enabled but the effective hostname '%s' is not a "
                "single-label .local name — advertising nothing (set "
                "[proxy].hostname_override to e.g. 'mybox.local', or unset it "
                "to use '%s').",
                self._hostname,
                default_local_hostname(),
            )
            return
        address = self._address or detect_lan_ip()
        if address is None:
            self.reason = MDNS_REASON_NO_ADDRESS
            logger.warning(
                "mDNS: could not determine a LAN address to advertise — "
                "advertising nothing (set [proxy].mdns_address explicitly)."
            )
            return
        try:
            from zeroconf import IPVersion, ServiceInfo
            from zeroconf.asyncio import AsyncZeroconf
        except ImportError:
            self.reason = MDNS_REASON_ZEROCONF_MISSING
            logger.warning(
                "mDNS is enabled but the 'zeroconf' package is not installed — "
                "advertising nothing. Install it with: pip install 'nerdit[mdns]'"
            )
            return
        try:
            packed = socket.inet_aton(address)
            info = ServiceInfo(
                type_="_https._tcp.local.",
                name=f"{name}._https._tcp.local.",
                addresses=[packed],
                port=self._settings.https_port,
                server=f"{name}{_MDNS_SUFFIX}.",
                properties={"path": "/"},
            )
            aiozc = (
                AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=self._interfaces)
                if self._interfaces
                else AsyncZeroconf(ip_version=IPVersion.V4Only)
            )
            await aiozc.async_register_service(info)
        except Exception:  # noqa: BLE001 — degrade, never raise into the lifespan
            self.reason = MDNS_REASON_REGISTER_FAILED
            logger.warning("mDNS: failed to start the advertiser", exc_info=True)
            return
        self._aiozc = aiozc
        self._info = info
        self.active = True
        self.reason = None
        logger.info("mDNS: advertising %s%s -> %s", name, _MDNS_SUFFIX, address)

    @property
    def registered(self) -> bool:
        """`True` when a mDNS `ServiceInfo` is currently advertised.

        Read by `GET /proxy/status` to project whether the advertisement is
        actually live (a truthier signal than `active`, which merely
        records that `start` reached its tail without an early return).
        """
        return self._info is not None

    async def stop(self) -> None:
        """Withdraw the advertisement (sends the mDNS goodbye packet)."""
        if self._aiozc is None:
            return
        with contextlib.suppress(Exception):
            if self._info is not None:
                await self._aiozc.async_unregister_service(self._info)
            await self._aiozc.async_close()
        self._aiozc = None
        self._info = None
        self.active = False
