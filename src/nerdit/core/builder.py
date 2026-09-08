"""Inspect an app directory and return a Docker build plan without invoking Docker.

An existing Dockerfile takes precedence. Otherwise, Python markers
(`requirements.txt` or `pyproject.toml`) precede Node's `package.json`.
`[deploy]` hints supply command and port values but do not select a language.
Unsupported trees raise `BuildpackNotSupported`; callers write generated
Dockerfiles and invoke the runtime.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nerdit.config.project import DeployConfig

# Filename used for Dockerfiles *generated* by Nerdit, so we never clobber a
# user's own file if one appears later. Passthrough keeps plain "Dockerfile".
GENERATED_DOCKERFILE_NAME = "Dockerfile.nerdit"

# Default container listen port for the Node buildpack when neither
# `[deploy].port` nor any other hint is available.
DEFAULT_NODE_PORT = 3000

NODE_BASE_IMAGE = "node:20-slim"

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


def detect(context_dir: str | Path, deploy_cfg: DeployConfig | None = None) -> BuildPlan:
    """Inspect *context_dir* and return a `BuildPlan`.

    Pure filesystem inspection — no Docker calls. Raises
    `BuildpackNotSupported` for unsupported (or empty) projects.
    """
    context = Path(context_dir)

    # 1. Existing Dockerfile → passthrough, the user owns the build.
    if (context / "Dockerfile").is_file():
        return BuildPlan(
            language="dockerfile",
            dockerfile_name="Dockerfile",
            dockerfile_text=None,
            port=deploy_cfg.port if deploy_cfg else None,
            start_command=deploy_cfg.start if deploy_cfg else None,
        )

    # 3. Python markers → generate a Python Dockerfile (pip install).
    #    (Checked before Node so a mixed repo with requirements.txt builds as
    #    Python — honest about what we would actually be building.)
    if (context / "requirements.txt").is_file() or (context / "pyproject.toml").is_file():
        return _plan_python(context, deploy_cfg)

    # 4. Node buildpack.
    package_json = context / "package.json"
    if package_json.is_file():
        return _plan_node(context, package_json, deploy_cfg)

    # Path-free by policy: the message reaches remote callers verbatim as the
    # `deploy.no_buildpack` envelope, and the extraction dir is daemon-internal.
    raise BuildpackNotSupported(
        "no Dockerfile, package.json, or supported project files found in "
        "the uploaded project — add a Dockerfile or a package.json to deploy this app."
    )


def _plan_node(context: Path, package_json: Path, deploy_cfg: DeployConfig | None) -> BuildPlan:
    """Build the Node buildpack plan: generated Dockerfile + resolved port/start."""
    try:
        pkg = json.loads(package_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BuildpackNotSupported(f"package.json is not valid JSON: {exc}") from exc

    scripts = pkg.get("scripts") if isinstance(pkg, dict) else None
    has_start_script = isinstance(scripts, dict) and bool(scripts.get("start"))

    # Port: an explicit [deploy].port wins, else the Node default. `port` is
    # optional now, so an unset value (`None`) also falls through to the default.
    port = deploy_cfg.port if (deploy_cfg and deploy_cfg.port) else DEFAULT_NODE_PORT

    # Start command resolution: [deploy].start > package.json scripts.start.
    if deploy_cfg and deploy_cfg.start:
        start_command = deploy_cfg.start
        cmd_line = f'CMD ["sh", "-c", {json.dumps(deploy_cfg.start)}]'
    elif has_start_script:
        start_command = "npm start"
        cmd_line = 'CMD ["npm", "start"]'
    else:
        # Fallback: conventional Node entrypoint when nothing else is declared.
        start_command = None
        cmd_line = 'CMD ["node", "server.js"]'

    # Reproducible installs when a lockfile exists; plain install otherwise.
    if (context / "package-lock.json").is_file():
        install_line = "RUN npm ci"
    else:
        install_line = "RUN npm install"

    # npm workspaces (monorepo): the root package.json declares a "workspaces"
    # key. `npm ci`/`npm install` then needs every packages/*/package.json to
    # resolve the workspace graph — the root-manifests-only COPY (fast path)
    # would make the install fail. When detected, copy the whole source tree
    # before installing (correctness over the layer-cache win).
    has_workspaces = isinstance(pkg, dict) and "workspaces" in pkg

    if has_workspaces:
        copy_and_install = f"""\
# npm workspaces detected: install needs packages/*/package.json, so deps
# install AFTER the full copy (the manifests-first layer-cache win is skipped).
COPY . .

# npm ci when a package-lock.json is present (reproducible), else npm install.
{install_line}"""
    else:
        copy_and_install = f"""\
# Copy manifests first so the dependency-install layer is cached across
# source-only edits (the package*.json glob is lockfile-optional — package.json
# always exists on the Node path, so the glob always matches).
COPY package*.json ./

# npm ci when a package-lock.json is present (reproducible), else npm install.
{install_line}

# Copy the rest of the app context (uploads are already filtered server-side).
COPY . ."""

    dockerfile_text = f"""\
# Generated by Nerdit (Node buildpack) — do not edit; regenerated on deploy.
FROM {NODE_BASE_IMAGE}
WORKDIR /app

{copy_and_install}

EXPOSE {port}

# Start command: [deploy].start > package.json scripts.start > node server.js
# (the last is a conventional fallback when nothing is declared).
{cmd_line}
"""

    return BuildPlan(
        language="node",
        dockerfile_name=GENERATED_DOCKERFILE_NAME,
        dockerfile_text=dockerfile_text,
        port=port,
        start_command=start_command,
    )


def _plan_python(context: Path, deploy_cfg: DeployConfig | None) -> BuildPlan:
    """Build the Python buildpack plan: generated Dockerfile + resolved port/start."""
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
    if requirements.is_file() and _requirements_reference_tree(requirements):
        # A requirements.txt entry references the source tree (`-e .`, `.`,
        # a nested `-r`/`-c` include, a `file:` URL). pip needs the whole
        # tree present, so copy first then install (layer-cache win skipped).
        copy_and_install = """\
# requirements.txt references the source tree, so deps install AFTER the full
# copy (the pin-file-first layer-cache win is skipped for correctness).
COPY . .

RUN pip install --no-cache-dir -r requirements.txt"""
    elif requirements.is_file():
        copy_and_install = """\
# Copy the pin file first so the install layer is cached across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app context (uploads are already filtered server-side).
COPY . ."""
    else:
        copy_and_install = """\
# Copy the app context (uploads are already filtered server-side).
COPY . .

# PEP 517 build needs the whole source tree present, so install comes last.
RUN pip install --no-cache-dir ."""

    # Start command resolution: [deploy].start > uvicorn fallback. The uvicorn
    # fallback assumes `uvicorn` is installed and the ASGI app is named
    # `main:app`; anything else must set [deploy].start explicitly. The
    # resolved integer `port` is interpolated into the shell string.
    if deploy_cfg and deploy_cfg.start:
        start_command = deploy_cfg.start
        cmd_line = f'CMD ["sh", "-c", {json.dumps(deploy_cfg.start)}]'
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
    )
