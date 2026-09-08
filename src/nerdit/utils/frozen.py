"""Restore host loader paths before frozen binaries spawn system programs.

PyInstaller prepends bundled libraries and saves original loader paths. Children
must inherit the originals to avoid loading incompatible bundled SSL libraries;
the running process already resolved its libraries at exec. Call once from each
entry point; source installs are untouched.
"""

from __future__ import annotations

import os
import sys

#: The loader variables PyInstaller rewrites, per platform. ``DYLD_*`` is
#: listed for symmetry — current bootloaders do not set it on macOS.
_LOADER_VARS = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")


#: Once restored, the variables ARE the host's — a second pass could not tell
#: them from the bootloader's and would unset them. One pass per process.
_restored = False


def restore_host_loader_env(environ: dict[str, str] | None = None) -> bool:
    """Restore child-process loader variables only in frozen builds.

    Return whether values changed. Apply once to os.environ; explicit mappings
    remain caller-managed.
    """
    global _restored
    if not getattr(sys, "frozen", False):
        return False
    if environ is None:
        if _restored:
            return False
        _restored = True
    env = os.environ if environ is None else environ
    changed = False
    for name in _LOADER_VARS:
        orig = env.pop(f"{name}_ORIG", None)
        if orig is not None:
            if env.get(name) != orig:
                env[name] = orig
                changed = True
        elif name in env:
            # The bootloader set it and the caller had none: unset, do not
            # leave the bundle's library directory in front of the system's.
            del env[name]
            changed = True
    return changed
