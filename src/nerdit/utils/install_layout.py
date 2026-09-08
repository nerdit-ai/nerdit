"""Detect frozen binaries, bundled Caddy, service units and installed layouts.

Explicit Caddy configuration wins; the default prefers the bundled tested
binary over PATH. Detect installer artifacts without import-time I/O or template
rendering. Root helpers keep filesystem probing independently testable.
"""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nerdit.config.defaults import DEFAULT_CADDY_BINARY, DEFAULT_PORT

#: systemd unit file name, identical in the system and ``--user`` scopes.
SYSTEMD_UNIT_NAME = "nerdit.service"
#: launchd label; the plist basename is ``<label>.plist``.
LAUNCHD_LABEL = "ai.nerdit.daemon"
#: Root of a system (root-owned) install — D-P30-9.
SYSTEM_ROOT = Path("/opt/nerdit")
#: The single shim a system install puts on ``PATH`` (``nerditd`` is reached
#: by absolute path from the unit, never through a shim).
SYSTEM_SHIM = Path("/usr/local/bin/nerdit")

UnitKind = Literal["systemd-system", "systemd-user", "launchd"]
InstallMode = Literal["system", "user"]


@dataclass(frozen=True)
class ServiceUnit:
    """A service-manager unit and its canonical restart, stop and disable commands.

    Keep commands aligned with install.sh. For launchd, bootout both stops and
    unloads the job, preventing login respawn.
    """

    kind: UnitKind
    unit_path: Path
    restart_argv: list[str]
    stop_argv: list[str]
    disable_argv: list[str]


@dataclass(frozen=True)
class InstallLayout:
    """On-disk layout created by the installer."""

    mode: InstallMode
    root: Path
    versions_dir: Path
    current: Path
    current_version: str | None
    shim: Path
    installer: Path


# --------------------------------------------------------------------------- #
# roots (monkeypatch seams)
# --------------------------------------------------------------------------- #


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _user_root() -> Path:
    """Root of a user install — also the default `data_dir`; only
    `versions/`, `current` and `bin/` under it are *code*.
    """
    return Path.home() / ".nerdit"


def _system_root() -> Path:
    return SYSTEM_ROOT


def _system_shim() -> Path:
    return SYSTEM_SHIM


def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _systemd_user_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME


def _systemd_system_unit_path() -> Path:
    return Path("/etc/systemd/system") / SYSTEMD_UNIT_NAME


# --------------------------------------------------------------------------- #
# frozen bundle
# --------------------------------------------------------------------------- #


def is_frozen() -> bool:
    """True inside a PyInstaller bundle (`sys.frozen`)."""
    return bool(getattr(sys, "frozen", False))


def bundled_caddy_path() -> Path | None:
    """Return Caddy beside a frozen executable, or None; source installs use PATH."""
    if not is_frozen():
        return None
    candidate = Path(sys.executable).parent / "caddy"
    return candidate if candidate.is_file() else None


def resolve_caddy_binary(configured: str) -> str | None:
    """Resolve Caddy to an absolute path, or None.

    Explicit settings win. Only the default caddy name prefers a bundled binary,
    falling back to PATH when absent.
    """
    if configured != DEFAULT_CADDY_BINARY:
        return shutil.which(configured)
    bundled = bundled_caddy_path()
    if bundled is not None:
        return str(bundled)
    return shutil.which(configured)


# --------------------------------------------------------------------------- #
# service unit
# --------------------------------------------------------------------------- #


def _systemctl(scope_user: bool, *args: str) -> list[str]:
    return ["systemctl", "--user", *args] if scope_user else ["systemctl", *args]


def _systemd_unit(kind: UnitKind, path: Path) -> ServiceUnit:
    scope_user = kind == "systemd-user"
    return ServiceUnit(
        kind=kind,
        unit_path=path,
        restart_argv=_systemctl(scope_user, "restart", SYSTEMD_UNIT_NAME),
        stop_argv=_systemctl(scope_user, "stop", SYSTEMD_UNIT_NAME),
        disable_argv=_systemctl(scope_user, "disable", SYSTEMD_UNIT_NAME),
    )


def _launchd_unit(path: Path) -> ServiceUnit:
    target = f"gui/{os.getuid()}/{LAUNCHD_LABEL}"
    bootout = ["launchctl", "bootout", target]
    return ServiceUnit(
        kind="launchd",
        unit_path=path,
        # kickstart -k restarts a loaded job in place; unlike bootout+bootstrap
        # it needs no plist path and does not race the reload.
        restart_argv=["launchctl", "kickstart", "-k", target],
        stop_argv=list(bootout),
        # Booting a job out IS the disable: it unloads the agent, so it does not
        # come back at the next login either.
        disable_argv=list(bootout),
    )


def detect_service_unit() -> ServiceUnit | None:
    """The service-manager unit managing this daemon, or `None`.

    Precedence is fixed and shared with `install.sh`: macOS has exactly one
    answer (the LaunchAgent), Linux prefers a `--user` unit over the system
    one because a user install cannot have written the system one.
    """
    if _is_darwin():
        plist = _launchd_plist_path()
        return _launchd_unit(plist) if plist.exists() else None
    user_unit = _systemd_user_unit_path()
    if user_unit.exists():
        return _systemd_unit("systemd-user", user_unit)
    system_unit = _systemd_system_unit_path()
    if system_unit.exists():
        return _systemd_unit("systemd-system", system_unit)
    return None


# --------------------------------------------------------------------------- #
# install layout
# --------------------------------------------------------------------------- #


def path_present(path: Path) -> bool:
    """Return whether a path exists or is a symlink, including dangling install links."""
    return path.is_symlink() or path.exists()


def managed_daemon_port(unit: ServiceUnit) -> int | None:
    """Read the daemon port from config under the service unit's recorded HOME.

    Use this to distinguish multiple local daemons before restarting a unit.

    Returns:
        The configured port, DEFAULT_PORT when config is genuinely absent, or None
        when unit/config data is unreadable or malformed. Reject boolean ports;
        unreadable config must not be mistaken for an absent file.
    """
    try:
        text = unit.unit_path.read_text(encoding="utf-8")
    except OSError:
        return None

    home: str | None = None
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        # systemd: Environment=HOME=/home/nerdituser
        if line.startswith("Environment=HOME="):
            home = line.split("=", 2)[2].strip().strip('"')
            break
        # launchd records HOME as WorkingDirectory, not EnvironmentVariables.
        # Read the installer template before changing this port-identity lookup.
        if line == "<key>WorkingDirectory</key>" and index + 1 < len(lines):
            value = lines[index + 1].strip()
            if value.startswith("<string>") and value.endswith("</string>"):
                home = value[len("<string>") : -len("</string>")]
                break
    if not home:
        return None

    config = Path(home) / ".nerdit" / "config.toml"
    # The open IS the three-state test: no ``is_file()`` pre-check, because
    # ``is_file()`` answers False for both "absent" and "cannot stat it", which
    # is precisely the distinction this function must not lose.
    try:
        with config.open("rb") as handle:
            parsed = tomllib.load(handle)
    except FileNotFoundError:
        # An installed unit whose user has never written a config runs on the
        # default — the same answer the daemon itself would reach.
        return DEFAULT_PORT
    except (OSError, tomllib.TOMLDecodeError):
        # Present-but-unreadable, or unparseable: decline rather than guess.
        return None
    daemon_section = parsed.get("daemon")
    if not isinstance(daemon_section, dict):
        return DEFAULT_PORT
    port = daemon_section.get("port", DEFAULT_PORT)
    # ``bool`` is an ``int`` subclass, so a bare ``isinstance(port, int)`` reads
    # ``port = true`` as the port number 1.
    if not isinstance(port, int) or isinstance(port, bool):
        return None
    return port


def _resolve_current_version(current: Path) -> str | None:
    """Basename of what `current` points at; `None` when dangling or not a
    symlink (a hand-made directory is not a version we may reason about).
    """
    if not current.is_symlink():
        return None
    try:
        target = os.readlink(current)
    except OSError:
        return None
    # ``current.parent / target`` resolves a relative link ("versions/0.5.0")
    # and passes an absolute one straight through.
    if not (current.parent / target).exists():
        return None
    return Path(target).name or None


def detect_install_layout() -> InstallLayout | None:
    """The installer-made layout on this machine, or `None` for a source checkout.

    A system install is checked first: when both exist (an operator who tried
    both modes) the system one owns `/usr/local/bin/nerdit`, i.e. the name
    that actually runs.
    """
    system_current = _system_root() / "current"
    if path_present(system_current):
        root = _system_root()
        return InstallLayout(
            mode="system",
            root=root,
            versions_dir=root,
            current=system_current,
            current_version=_resolve_current_version(system_current),
            shim=_system_shim(),
            installer=system_current / "install.sh",
        )

    user_current = _user_root() / "current"
    if path_present(user_current):
        root = _user_root()
        return InstallLayout(
            mode="user",
            root=root,
            versions_dir=root / "versions",
            current=user_current,
            current_version=_resolve_current_version(user_current),
            shim=root / "bin" / "nerdit",
            installer=user_current / "install.sh",
        )
    return None
