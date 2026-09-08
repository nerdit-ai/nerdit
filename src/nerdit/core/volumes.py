"""Resolve service volume names into daemon-owned host paths.

Every launch, run, and release revalidates persisted specs, failing closed before
any mount outside `<data_dir>/services/<name>`. Volume names are DNS labels;
callers supply container paths, never host paths. Each named volume maps to
`<data_dir>/services/<name>/<volname>`.
"""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from nerdit.config.project import _DNS_LABEL_RE
from nerdit.utils.ids import _ID_LENGTH

#: Volume names are DNS labels capped at 32 chars (docker named-volume norm).
#: A `/` provably cannot appear, so a name can never encode a host path.
_VOLNAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$")

#: Tombstone dir grammar (C2): a service's data root is renamed away to
#: `.trash-<name>-<nonce>` before a database delete commits, so a refused
#: delete can rename it BACK (an O(1) reversible move, unlike an rmtree). The
#: leading `.` is un-mintable as a service name (`_DNS_LABEL_RE` rejects a
#: leading dot) so a tombstone can never collide with or be resolved as a real
#: `service_data_root`; the nonce keeps repeated deletes of one name distinct.
_TOMBSTONE_PREFIX = ".trash-"
_TOMBSTONE_NONCE_RE = re.compile(r"^[0-9a-f]{8}$")


def make_tombstone_name(service_name: str) -> str:
    """Mint a fresh `.trash-<service_name>-<nonce>` tombstone dir name.

    Re-validates `service_name` against the DNS-label grammar (defense in
    depth — a forged name must never reach the filesystem here).
    """
    if not _DNS_LABEL_RE.fullmatch(service_name):
        raise VolumeSpecError(f"invalid service name: {service_name!r}")
    return f"{_TOMBSTONE_PREFIX}{service_name}-{secrets.token_hex(4)}"


def tombstone_service_name(name: str) -> str | None:
    """Return the service a `.trash-<name>-<nonce>` dir belongs to, else `None`.

    `None` for any name that is not a well-formed tombstone (a plain service
    dir, or an unparseable/foreign dotdir): the nonce must be 8 lowercase hex
    chars and the base must be a valid DNS label. The nonce carries no dashes,
    so a right-partition on `-` cleanly separates a dashed service name from it.
    """
    if not name.startswith(_TOMBSTONE_PREFIX):
        return None
    rest = name[len(_TOMBSTONE_PREFIX) :]
    base, sep, nonce = rest.rpartition("-")
    if not sep or not _TOMBSTONE_NONCE_RE.fullmatch(nonce):
        return None
    if not _DNS_LABEL_RE.fullmatch(base):
        return None
    return base


#: Container mount points that must never be shadowed by a named volume.
_FORBIDDEN_PATHS = frozenset({"/", "/workspace"})
#: Kernel-virtual roots a bind mount must never target.
_FORBIDDEN_PREFIXES = ("/proc", "/sys", "/dev")

#: Hard cap on named volumes per service (mirrors the parse-time grammar).
_MAX_VOLUMES = 8


class VolumeSpecError(ValueError):
    """A named-volume spec failed launch-time re-validation (fail-closed)."""


def service_data_root(data_dir: Path, service_name: str) -> Path:
    """Return `<data_dir>/services/<service_name>`, re-validating the name.

    `service_name` is re-checked against the DNS-label grammar and the result
    is asserted to live under `<data_dir>/services` — defense-in-depth against
    a forged name reaching this seam from any producer.
    """
    if not _DNS_LABEL_RE.fullmatch(service_name):
        raise VolumeSpecError(f"invalid service name: {service_name!r}")
    # The daemon-owned levels (`services` and `services/<name>`) must be
    # real directories: `.resolve()` would otherwise adopt a symlink's target
    # as the trusted base, silently relocating every mount (possible if
    # `data_dir` was world-writable before the daemon first created them).
    unresolved_base = data_dir / "services"
    if unresolved_base.is_symlink():
        raise VolumeSpecError("services dir is a symlink; refusing to resolve mounts")
    if (unresolved_base / service_name).is_symlink():
        raise VolumeSpecError(f"service data root is a symlink: {service_name!r}")
    base = unresolved_base.resolve()
    root = (base / service_name).resolve()
    if not root.is_relative_to(base):
        raise VolumeSpecError(f"service data root escapes base: {service_name!r}")
    return root


#: A staging slot is named after its run id (the ``generate_id`` grammar, length
#: read from that module). Re-applied here: the single owner of the staging
#: grammar (D-P37-2) must never pass a forged id to the filesystem.
_RUN_ID_RE = re.compile(rf"^[a-z0-9]{{{_ID_LENGTH}}}$")


def dump_staging_root(data_dir: Path) -> Path:
    """Return ``<data_dir>/dump-staging`` (P37, D-P37-2). A dedicated root, not
    ``backups``: it is the one path a sibling bind-mounts, and the runtime's
    Tier-A carve-out is prefix-based.
    """
    return data_dir / "dump-staging"


def dump_staging_dir(data_dir: Path, run_id: str) -> Path:
    """Return ``<data_dir>/dump-staging/<run_id>``, re-validating the run id (D-P37-2,
    the single owner of the staging grammar): minted-id grammar, no symlink at
    root or slot, containment asserted. Not created here — the caller creates it
    ``mode=0o700, exist_ok=False``.
    """
    if not _RUN_ID_RE.fullmatch(run_id):
        raise VolumeSpecError(f"invalid run id: {run_id!r}")
    root = dump_staging_root(data_dir)
    if root.is_symlink():
        raise VolumeSpecError("dump staging root is a symlink; refusing to resolve it")
    if (root / run_id).is_symlink():
        raise VolumeSpecError(f"dump staging dir is a symlink: {run_id!r}")
    base = root.resolve()
    path = (base / run_id).resolve()
    if not path.is_relative_to(base):
        raise VolumeSpecError(f"dump staging dir escapes the staging root: {run_id!r}")
    return path


def create_dump_staging_dir(data_dir: Path, run_id: str) -> Path:
    """Create ``<data_dir>/dump-staging/<run_id>`` ``0o700``, refusing to reuse one.

    Blocking. The single creator for both producers (controller dump and restore
    route). ``exist_ok=False`` makes a slot single-owner. :func:`dump_staging_dir`
    runs FIRST so a symlinked root is refused before anything touches the
    filesystem (``mkdir(exist_ok=True)`` + ``chmod`` would follow it), and the root
    is re-checked after the ``mkdir``. Raises :class:`VolumeSpecError` for every
    failure, filesystem errors included; messages path-free (M3).
    """
    root = dump_staging_root(data_dir)
    staging = dump_staging_dir(data_dir, run_id)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise VolumeSpecError("dump staging root is a symlink; refusing to resolve it")
        os.chmod(root, 0o700)
        staging.mkdir(mode=0o700, exist_ok=False)
    except OSError as exc:
        raise VolumeSpecError("could not create the dump staging dir") from exc
    return staging


def _validate_container_path(path: str) -> str:
    """Re-apply the container-path grammar; return it unchanged or raise."""
    if not path or not path.startswith("/"):
        raise VolumeSpecError(f"container path must be absolute: {path!r}")
    # POSIX `os.path.normpath` preserves an exactly-two-slash leading prefix
    # (`//x` stays `//x`), so it would pass the normpath-stable check below
    # yet miss the forbidden-path exact/prefix matches (the kernel collapses
    # `//x` -> `/x`). Reject a doubled leading slash outright so no forbidden
    # mount point can be smuggled past this grammar.
    if path.startswith("//"):
        raise VolumeSpecError(f"container path has a doubled leading slash: {path!r}")
    if os.path.normpath(path) != path:
        raise VolumeSpecError(f"container path is not normalized: {path!r}")
    if path in _FORBIDDEN_PATHS:
        raise VolumeSpecError(f"container path not allowed: {path!r}")
    for prefix in _FORBIDDEN_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            raise VolumeSpecError(f"container path not allowed: {path!r}")
    return path


def resolve_named_volumes(service_name: str, specs: list[str], data_dir: Path) -> dict[str, str]:
    """Resolve `<volname>:<container_path>` specs to `{host_path: container_path}`.

    Full re-validation of every component at launch (volname regex, container
    path normpath rules, DNS label, `is_relative_to` assert), trusting no
    producer. Rejects duplicate names or container paths and more than
    `_MAX_VOLUMES` entries. Raises `VolumeSpecError` on the first
    violation — the caller settles the row `failed` / `reason="volume_invalid"`.
    """
    if len(specs) > _MAX_VOLUMES:
        raise VolumeSpecError(f"too many volumes: {len(specs)} > {_MAX_VOLUMES}")
    root = service_data_root(data_dir, service_name)
    result: dict[str, str] = {}
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for spec in specs:
        if not isinstance(spec, str):
            raise VolumeSpecError(f"volume spec must be a string: {spec!r}")
        volname, sep, container_path = spec.partition(":")
        if not sep:
            raise VolumeSpecError(f"volume spec missing ':': {spec!r}")
        if ":" in container_path:
            raise VolumeSpecError(f"volume spec has extra ':': {spec!r}")
        if not _VOLNAME_RE.fullmatch(volname):
            raise VolumeSpecError(f"invalid volume name: {volname!r}")
        container_path = _validate_container_path(container_path)
        if volname in seen_names:
            raise VolumeSpecError(f"duplicate volume name: {volname!r}")
        if container_path in seen_paths:
            raise VolumeSpecError(f"duplicate container path: {container_path!r}")
        # Same rule as the parents: a pre-planted symlink at the leaf must never
        # redirect the mount (nonexistent leaves are fine — created at launch).
        if (root / volname).is_symlink():
            raise VolumeSpecError(f"volume host path is a symlink: {volname!r}")
        host = (root / volname).resolve()
        # Defense-in-depth: the DNS-label volname cannot traverse, but assert it.
        if not host.is_relative_to(root):
            raise VolumeSpecError(f"volume host path escapes root: {volname!r}")
        seen_names.add(volname)
        seen_paths.add(container_path)
        result[str(host)] = container_path
    return result


def service_volumes(data_dir: Path, service_name: str, cfg: dict) -> dict[str, str]:
    """Resolve a service config's `volumes` list to host->container mounts.

    Convenience over `resolve_named_volumes` reading `cfg['volumes']`
    (absent/empty ⇒ `{}`). Consumed by `_launch`, `run_once` and release.
    """
    specs = cfg.get("volumes") or []
    if not isinstance(specs, list):
        raise VolumeSpecError(f"volumes must be a list: {specs!r}")
    return resolve_named_volumes(service_name, specs, data_dir)
