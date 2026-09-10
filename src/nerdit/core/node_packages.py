"""Resolve a supported package manager without executing repository code."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

COREPACK_VERSION = "0.36.0"
_DEFAULTS = {"npm": "11.19.1", "pnpm": "10.34.5", "yarn": "1.22.22"}
_SUPPORTED = {"npm": {10, 11}, "pnpm": {9, 10, 11, 12}, "yarn": {1, 4}}
_PIN = re.compile(
    r"(npm|pnpm|yarn)@((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)(?:\+sha(?:224\.[a-fA-F0-9]{56}|256\.[a-fA-F0-9]{64}|"
    r"384\.[a-fA-F0-9]{96}|512\.[a-fA-F0-9]{128}))?)"
)
_LOCKS = {
    "package-lock.json": "npm",
    "npm-shrinkwrap.json": "npm",
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "bun.lock": "bun",
    "bun.lockb": "bun",
}


@dataclass(frozen=True)
class NodePackages:
    """Resolved, validated inputs for the generated container build."""

    name: str
    version: str
    lockfile: str | None
    install: str


def _lock_format(context: Path, name: str) -> str:
    try:
        text = (context / name).read_text(encoding="utf-8")
        if name in {"package-lock.json", "npm-shrinkwrap.json"}:
            data = json.loads(text)
            version = data.get("lockfileVersion") if isinstance(data, dict) else None
            if type(version) is int and version in {1, 2, 3}:
                return "npm"
        elif name == "pnpm-lock.yaml":
            if re.search(
                r"(?m)^lockfileVersion: (?P<quote>[\"']?)9(?:\.0)?(?P=quote)[ \t]*$", text
            ):
                return "pnpm"
        elif name == "yarn.lock":
            if re.search(r"(?m)^# yarn lockfile v1\s*$", text):
                return "yarn1"
            match = re.search(r"(?m)^__metadata:\s*\n  version: (8|10)[ \t]*$", text)
            if match:
                return "yarn4:" + match[1]
    except (OSError, UnicodeError, ValueError):
        pass
    raise ValueError("Unsupported or unreadable Node lockfile; regenerate it or add a Dockerfile.")


def validate_package_manager(declared: object) -> tuple[str, str]:
    """Validate a manager pin using the same grammar at every ingress."""
    match = _PIN.fullmatch(declared) if isinstance(declared, str) and len(declared) <= 256 else None
    if not match:
        raise ValueError("packageManager must pin npm, pnpm or Yarn to an exact stable version.")
    name, version = match.groups()
    if int(version.split(".")[0]) not in _SUPPORTED[name]:
        raise ValueError("Unsupported packageManager major version; use a Dockerfile.")
    return name, version


def resolve_node_packages(context: Path, pkg: dict) -> NodePackages:
    """Reject conflicting declarations and return pinned manager/install commands."""
    locks = [name for name in _LOCKS if (context / name).exists() or (context / name).is_symlink()]
    if len(locks) > 1:
        raise ValueError("Conflicting Node lockfiles; keep one package manager's lockfile.")
    lockfile = locks[0] if locks else None
    lock_manager = _LOCKS[lockfile] if lockfile else None
    if lock_manager == "bun":
        raise ValueError(
            "Bun builds require a Dockerfile; supported managers are npm, pnpm and Yarn."
        )
    name = lock_manager or "npm"
    version = _DEFAULTS[name]
    declared = pkg.get("packageManager")
    if "packageManager" in pkg:
        name, version = validate_package_manager(declared)
        if lock_manager and lock_manager != name:
            raise ValueError("packageManager conflicts with the lockfile; make them agree.")
    if lockfile:
        if (context / lockfile).is_symlink() or not (context / lockfile).is_file():
            raise ValueError("Node lockfile must be a regular file inside the project.")
        lock_format = _lock_format(context, lockfile)
        if lock_format.startswith("yarn"):
            if declared is None:
                version = {"yarn1": "1.22.22", "yarn4:8": "4.9.4", "yarn4:10": "4.18.0"}[
                    lock_format
                ]
            if lock_format.split(":")[0] != f"yarn{version.split('.')[0]}":
                raise ValueError(
                    "Yarn version conflicts with yarn.lock; regenerate it or add a Dockerfile."
                )
    return NodePackages(name, version, lockfile, _install_command(name, version, bool(lockfile)))


def _install_command(name: str, version: str, locked: bool) -> str:
    """Use each manager's locked install mode while retaining build dependencies."""
    if name == "npm":
        return "npm ci --include=dev" if locked else "npm install --include=dev"
    if name == "pnpm":
        freeze = "--frozen-lockfile" if locked else "--no-frozen-lockfile"
        return f"pnpm install {freeze} --prod=false"
    if version.startswith("1."):
        freeze = " --frozen-lockfile" if locked else ""
        return f"yarn install{freeze} --production=false"
    return "yarn install --immutable" if locked else "yarn install --no-immutable"


def node_install_needs_source(context: Path, pkg: dict) -> bool:
    """Keep the dependency cache only when installation is manifest-only."""
    if "workspaces" in pkg or any(
        (context / name).exists()
        for name in (
            ".npmrc",
            ".yarnrc",
            ".yarnrc.yml",
            ".yarn",
            "pnpm-workspace.yaml",
            ".pnpmfile.cjs",
            "binding.gyp",
            ".pnp.cjs",
            ".pnp.loader.mjs",
        )
    ):
        return True
    scripts = pkg.get("scripts", {})
    if any(
        scripts.get(name)
        for name in (
            "preinstall",
            "install",
            "postinstall",
            "prepublish",
            "preprepare",
            "prepare",
            "postprepare",
        )
    ):
        return True
    if "pnpm" in pkg or "overrides" in pkg:
        return True  # patchedDependencies and hooks may refer to source files.
    for group in ("dependencies", "devDependencies", "optionalDependencies", "resolutions"):
        deps = pkg.get(group, {})
        if isinstance(deps, dict) and any(
            isinstance(value, str) and value.startswith(("file:", "link:", "workspace:", "."))
            for value in deps.values()
        ):
            return True
    return False
