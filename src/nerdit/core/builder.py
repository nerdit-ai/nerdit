"""Inspect an app directory and return a Docker build plan without invoking Docker.

An existing Dockerfile takes precedence. Otherwise, Python project markers
precede Node's `package.json`; tooling-only `pyproject.toml` files do not.
`[deploy]` hints supply command and port values but do not select a language.
Unsupported trees raise `BuildpackNotSupported`; callers write generated
Dockerfiles and invoke the runtime.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nerdit.config.build import BuildSettings
from nerdit.core.node_packages import (
    COREPACK_VERSION,
    node_install_needs_source,
    resolve_node_packages,
)
from nerdit.core.node_runtime import DEFAULT_NODE_VERSION, resolve_node_version

if TYPE_CHECKING:
    from nerdit.config.project import DeployConfig

# Filename used for Dockerfiles *generated* by Nerdit, so we never clobber a
# user's own file if one appears later. Passthrough keeps plain "Dockerfile".
GENERATED_DOCKERFILE_NAME = "Dockerfile.nerdit"

# Default container listen port for the Node buildpack when neither
# `[deploy].port` nor any other hint is available.
DEFAULT_NODE_PORT = 3000

NODE_BASE_IMAGE = f"node:{DEFAULT_NODE_VERSION}-slim"

# Default container listen port for the Python buildpack when neither
# `[deploy].port` nor any other hint is available.
DEFAULT_PYTHON_PORT = 8000

PYTHON_BASE_IMAGE = "python:3.11-slim"

# A requirements.txt line that references the source tree — an editable/local
# install (`-e .`, `.`, `./libs/foo`, `../x`, `file:` URL) or a nested
# `-r`/`-c` include. pip can only resolve these once `COPY . .` has landed,
# so their presence forces install-after-copy (the fast path would 404).
_TREE_REF = re.compile(r"^(-e|--editable|-r|--requirement|-c|--constraint)\b|^\.{1,2}(/|$)|^file:")


def _requirements_reference_tree(requirements: Path) -> bool:
    """True when any requirements.txt line references the source tree.

    Tolerant of read failures: on OSError take the safe (install-after-copy)
    fallback by returning True, so a build never 404s on an unreadable pin file.
    """
    try:
        text = requirements.read_text(encoding="utf-8")
    except OSError:
        return True
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _TREE_REF.match(line):
            return True
    return False


class BuildpackNotSupported(Exception):  # noqa: N818 — P4 public API name
    """Raised when no supported buildpack matches the app folder."""


def _public_env_args(settings: BuildSettings) -> str:
    """Bare ``ARG KEY`` declarations for the plan's public build variables.

    Returned with a leading blank line so callers can append it straight after
    the install step: values arrive via ``--build-arg``, never as ``ARG K=V``
    defaults (Dockerfile expansion would mangle them), and declaring them after
    the install keeps the dependency layer cached when only a value changes.
    Empty string when no ``public_env`` is set.
    """
    keys = sorted(settings.public_env or {})
    return "\n\n" + "\n".join(f"ARG {key}" for key in keys) if keys else ""


def _undeclared_public_env(context: Path, settings: BuildSettings) -> list[str]:
    """Warn per public_env key the passthrough Dockerfile never declares as ARG.

    Build args are passed to a user Dockerfile but only reach the build when it
    declares them; an undeclared key is the silent failure this feature exists
    to make loud. The instruction keyword is case-insensitive (Dockerfile
    grammar); the ARG *name* is not (docker compares it byte for byte). Only the
    text from the first `FROM` is searched, since an `ARG` above it is a global
    arg and is unset inside every stage.
    """
    keys = sorted(settings.public_env or {})
    if not keys:
        return []
    try:
        text = (context / "Dockerfile").read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    # ponytail: token search from the first stage, so a declaration in ANOTHER
    # stage of a multi-stage build reads as declared. Parse the stage graph if
    # that ever warns falsely.
    stages = re.search(r"^[ \t]*(?i:FROM)\b", text, re.MULTILINE)
    body = text[stages.start() :] if stages else text
    return [
        f"public_env {key} is not declared as ARG in Dockerfile; the value will be unused."
        for key in keys
        if not re.search(rf"^[ \t]*(?i:ARG)\s+{re.escape(key)}(\b|=)", body, re.MULTILINE)
    ]


@dataclass
class BuildPlan:
    """What the caller needs to build an image from a context directory.

    `dockerfile_text is None` means passthrough: build with the existing
    on-disk `dockerfile_name`. Otherwise the caller must first write
    `dockerfile_text` to `context_dir / dockerfile_name`.
    """

    language: str  # "dockerfile" | "node" | "python"
    dockerfile_name: str  # relative to context_dir; passed to build_image
    dockerfile_text: str | None  # generated content, or None for passthrough
    port: int | None  # container listen port (EXPOSE / endpoint wiring)
    start_command: str | None  # resolved start command, if any
    framework: str | None = None
    node_version: str | None = None
    package_manager: str | None = None
    install_command: str | None = None
    build_command: str | None = None
    effective_start_command: str | None = None
    warnings: list[str] = field(default_factory=list)


def detect(context_dir: str | Path, deploy_cfg: DeployConfig | None = None) -> BuildPlan:
    """Inspect *context_dir* and return a `BuildPlan`.

    Pure filesystem inspection — no Docker calls. Raises
    `BuildpackNotSupported` for unsupported (or empty) projects.
    """
    context = Path(context_dir)
    settings = (
        deploy_cfg.build_settings if deploy_cfg and deploy_cfg.build_settings else BuildSettings()
    )
    explicit_start = settings.start or (deploy_cfg.start if deploy_cfg else None)

    # 1. Existing Dockerfile → passthrough, the user owns the build.
    if (context / "Dockerfile").is_file():
        # public_env is deliberately absent from the "ignored overrides" list:
        # build args DO reach a user Dockerfile — provided it declares them.
        warnings = (
            [
                "Dockerfile is authoritative; preset/install/build/runtime/manager "
                "overrides are ignored."
            ]
            if settings.preset not in (None, "dockerfile")
            or any(
                getattr(settings, key) is not None
                for key in ("install", "build", "node_version", "package_manager")
            )
            else []
        )
        return BuildPlan(
            language="dockerfile",
            dockerfile_name="Dockerfile",
            dockerfile_text=None,
            port=deploy_cfg.port if deploy_cfg else None,
            start_command=explicit_start,
            effective_start_command=explicit_start,
            warnings=warnings + _undeclared_public_env(context, settings),
        )

    if settings.preset == "dockerfile":
        raise BuildpackNotSupported(
            "The Dockerfile preset requires a Dockerfile in the selected root."
        )
    if settings.preset in ("node", "nextjs"):
        package_json = context / "package.json"
        if not package_json.is_file():
            raise BuildpackNotSupported(
                "The Node/Next.js preset requires package.json in the selected root."
            )
        return _plan_node(context, package_json, deploy_cfg)
    if settings.preset == "python":
        if not any(
            (context / marker).is_file() for marker in ("requirements.txt", "pyproject.toml")
        ):
            raise BuildpackNotSupported(
                "The Python preset requires requirements.txt or pyproject.toml "
                "in the selected root."
            )
        return _plan_python(context, deploy_cfg)

    # Python projects retain precedence, but Ruff/Black configuration alone
    # must not turn a Node app into a Python build.
    package_json = context / "package.json"
    pyproject = context / "pyproject.toml"
    if (context / "requirements.txt").is_file() or (
        pyproject.is_file()
        and (not package_json.is_file() or _pyproject_defines_project(pyproject))
    ):
        return _plan_python(context, deploy_cfg)

    # 4. Node buildpack.
    if package_json.is_file():
        return _plan_node(context, package_json, deploy_cfg)

    # Path-free by policy: the message reaches remote callers verbatim as the
    # `deploy.no_buildpack` envelope, and the extraction dir is daemon-internal.
    raise BuildpackNotSupported(
        "no Dockerfile, package.json, or supported project files found in "
        "the uploaded project — add a Dockerfile or a package.json to deploy this app."
    )


def _pyproject_defines_project(pyproject: Path) -> bool:
    """Distinguish Python project metadata from standalone tool configuration."""
    if pyproject.is_symlink() or any(
        (pyproject.parent / marker).is_file() for marker in ("setup.py", "setup.cfg")
    ):
        return True
    try:
        metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # Only change precedence when we can establish the file is tooling-only.
        return True
    tool = metadata.get("tool", {})
    return (
        "project" in metadata
        or "build-system" in metadata
        or (isinstance(tool, dict) and "poetry" in tool)
    )


def _node_manifest(package_json: Path) -> tuple[dict, dict, str]:
    """Read and validate the manifest without exposing file contents in errors."""
    try:
        if package_json.is_symlink():
            raise BuildpackNotSupported("package.json must be a regular file inside the project.")
        pkg = json.loads(package_json.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        raise BuildpackNotSupported("package.json is not valid JSON or cannot be read.") from None
    if not isinstance(pkg, dict):
        raise BuildpackNotSupported("package.json must contain a JSON object.")
    scripts = pkg.get("scripts", {})
    if not isinstance(scripts, dict) or any(
        name in scripts and not isinstance(scripts[name], str) for name in ("build", "start")
    ):
        raise BuildpackNotSupported("package.json build/start scripts must be strings.")
    framework = "node"
    for group in ("dependencies", "devDependencies"):
        deps = pkg.get(group, {})
        if not isinstance(deps, dict):
            raise BuildpackNotSupported("package.json dependency groups must be objects.")
        if "next" in deps:
            framework = "nextjs"
    return pkg, scripts, framework


def _plan_node(context: Path, package_json: Path, deploy_cfg: DeployConfig | None) -> BuildPlan:
    """Resolve a generated Node build using declarative repository metadata."""
    pkg, scripts, framework = _node_manifest(package_json)
    build_script = scripts.get("build", "").strip()
    has_start = bool(scripts.get("start", "").strip())
    settings = (
        deploy_cfg.build_settings if deploy_cfg and deploy_cfg.build_settings else BuildSettings()
    )
    if settings.preset == "nextjs" and framework != "nextjs":
        raise BuildpackNotSupported(
            "The Next.js preset requires a declared next dependency in package.json."
        )
    explicit_start = settings.start or (deploy_cfg.start if deploy_cfg else None)
    has_build = bool(settings.build) or bool(build_script) or settings.build is False
    if framework == "nextjs" and (not has_build or not (has_start or explicit_start)):
        raise BuildpackNotSupported(
            "Next.js server deployments require build and start scripts "
            "(next build and next start), or a Dockerfile for custom output modes."
        )
    try:
        package_metadata = (
            {**pkg, "packageManager": settings.package_manager} if settings.package_manager else pkg
        )
        packages = resolve_node_packages(context, package_metadata)
        node_version = settings.node_version or resolve_node_version(context, pkg)
    except ValueError as exc:
        raise BuildpackNotSupported(str(exc)) from None
    port = deploy_cfg.port if (deploy_cfg and deploy_cfg.port) else DEFAULT_NODE_PORT
    if explicit_start:
        start_command = explicit_start
        command = ["sh", "-c", explicit_start]
    elif has_start:
        start_command = f"{packages.name} start"
        command = [packages.name, "start"]
    else:
        start_command = None
        command = ["node", "server.js"]
        if packages.name == "yarn" and packages.version.startswith("4."):
            command.insert(0, "yarn")  # Yarn must load its Plug'n'Play dependency map.

    install_command = settings.install or packages.install
    install = (
        "RUN " + json.dumps(["sh", "-c", settings.install])
        if settings.install
        else f"RUN {packages.install}"
    )
    args = _public_env_args(settings)
    if settings.install or node_install_needs_source(context, pkg):
        copy_and_install = f"COPY . .\n\n{install}{args}"
    else:
        # Configuration, lifecycle scripts and local dependencies take the full
        # source path above. Simple projects keep a cached dependency layer.
        manifests = "package*.json" if packages.name == "npm" else "package.json"
        if packages.lockfile and packages.lockfile != "package-lock.json":
            manifests += f" {packages.lockfile}"
        copy_and_install = f"COPY {manifests} ./\n\n{install}{args}\n\nCOPY . ."
    build_command = (
        None
        if settings.build is False
        else settings.build or (f"{packages.name} run build" if build_script else None)
    )
    if build_command:
        build_run = json.dumps(["sh", "-c", build_command]) if settings.build else build_command
        copy_and_install += f"\n\nRUN {build_run}"
    manager_pin = f"{packages.name}@{packages.version}"
    # Turbopack resolves node_modules, not Yarn's Plug'n'Play map. Keep the
    # repository's scripts while selecting the compatible Next deployment layout.
    next_yarn_env = (
        "\nENV YARN_NODE_LINKER=node-modules"
        if framework == "nextjs" and packages.name == "yarn" and packages.version.startswith("4.")
        else ""
    )
    dockerfile_text = f"""\
# Generated by Nerdit. Framework: {framework}; package manager: {manager_pin}.
FROM node:{node_version}-slim
WORKDIR /app
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 COREPACK_DEFAULT_TO_LATEST=0
ENV COREPACK_ENABLE_PROJECT_SPEC=0{next_yarn_env}

RUN ["npm", "install", "--global", "--force", "corepack@{COREPACK_VERSION}"]
RUN ["corepack", "enable", "npm", "pnpm", "yarn"]
RUN ["corepack", "prepare", "{manager_pin}", "--activate"]

{copy_and_install}

EXPOSE {port}
CMD {json.dumps(command)}
"""
    return BuildPlan(
        language="node",
        dockerfile_name=GENERATED_DOCKERFILE_NAME,
        dockerfile_text=dockerfile_text,
        port=port,
        start_command=start_command,
        framework=framework,
        node_version=node_version,
        package_manager=manager_pin,
        install_command=install_command,
        build_command=build_command,
        effective_start_command=start_command or " ".join(command),
    )


def _plan_python(context: Path, deploy_cfg: DeployConfig | None) -> BuildPlan:
    """Build the Python buildpack plan: generated Dockerfile + resolved port/start."""
    settings = (
        deploy_cfg.build_settings if deploy_cfg and deploy_cfg.build_settings else BuildSettings()
    )
    if settings.node_version or settings.package_manager:
        raise BuildpackNotSupported(
            "Node runtime and package-manager overrides do not apply to Python builds."
        )
    # Port: an explicit [deploy].port wins, else the Python default. `port` is
    # optional now, so an unset value (`None`) also falls through to the default.
    port = deploy_cfg.port if (deploy_cfg and deploy_cfg.port) else DEFAULT_PYTHON_PORT

    # Dependency install: requirements.txt (the pin file) if present, else the
    # project itself via PEP 517 (pyproject.toml). `detect()` guarantees at
    # least one of these markers exists before routing here.
    #
    # Layer ordering differs by path: the requirements.txt path copies the pin
    # file first so the install layer caches across source-only edits (fast
    # path). The pyproject-only `pip install .` path must keep the install
    # AFTER `COPY . .` — PEP 517 needs the whole source tree present to build.
    requirements = context / "requirements.txt"
    install_command = settings.install or (
        "pip install --no-cache-dir -r requirements.txt"
        if requirements.is_file()
        else "pip install --no-cache-dir ."
    )
    args = _public_env_args(settings)
    if settings.install:
        copy_and_install = "COPY . .\n\nRUN " + json.dumps(["sh", "-c", settings.install]) + args
    elif requirements.is_file() and _requirements_reference_tree(requirements):
        # A requirements.txt entry references the source tree (`-e .`, `.`,
        # a nested `-r`/`-c` include, a `file:` URL). pip needs the whole
        # tree present, so copy first then install (layer-cache win skipped).
        copy_and_install = f"""\
# requirements.txt references the source tree, so deps install AFTER the full
# copy (the pin-file-first layer-cache win is skipped for correctness).
COPY . .

RUN pip install --no-cache-dir -r requirements.txt{args}"""
    elif requirements.is_file():
        copy_and_install = f"""\
# Copy the pin file first so the install layer is cached across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt{args}

# Copy the rest of the app context (uploads are already filtered server-side).
COPY . ."""
    else:
        copy_and_install = f"""\
# Copy the app context (uploads are already filtered server-side).
COPY . .

# PEP 517 build needs the whole source tree present, so install comes last.
RUN pip install --no-cache-dir .{args}"""

    # Start command resolution: [deploy].start > uvicorn fallback. The uvicorn
    # fallback assumes `uvicorn` is installed and the ASGI app is named
    # `main:app`; anything else must set [deploy].start explicitly. The
    # resolved integer `port` is interpolated into the shell string.
    build_command = settings.build if isinstance(settings.build, str) else None
    if build_command:
        copy_and_install += "\n\nRUN " + json.dumps(["sh", "-c", build_command])
    explicit_start = settings.start or (deploy_cfg.start if deploy_cfg else None)
    if explicit_start:
        start_command = explicit_start
        cmd_line = f'CMD ["sh", "-c", {json.dumps(explicit_start)}]'
    else:
        start_command = None
        cmd_line = f'CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port {port}"]'

    dockerfile_text = f"""\
# Generated by Nerdit (Python buildpack) — do not edit; regenerated on deploy.
FROM {PYTHON_BASE_IMAGE}
WORKDIR /app

# Install deps: requirements.txt (pip -r) if present, else the project (pyproject.toml).
{copy_and_install}

EXPOSE {port}

# Start command: [deploy].start > uvicorn main:app fallback. Set [deploy].start
# for anything real (Python has no scripts.start convention; the fallback
# assumes uvicorn is installed and the app is main:app).
{cmd_line}
"""

    return BuildPlan(
        language="python",
        dockerfile_name=GENERATED_DOCKERFILE_NAME,
        dockerfile_text=dockerfile_text,
        port=port,
        start_command=start_command,
        install_command=install_command,
        build_command=build_command,
        effective_start_command=start_command or f"uvicorn main:app --host 0.0.0.0 --port {port}",
    )
