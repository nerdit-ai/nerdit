"""Check frozen-build inputs, POSIX shell syntax and dynamic uvicorn imports.

These guards do not run PyInstaller. Validate the actual bundle with
`packaging/smoke.sh` on a clean VM.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = REPO_ROOT / "packaging"
SPEC = PACKAGING / "nerdit.spec"
ASSEMBLE = PACKAGING / "assemble.sh"
SMOKE = PACKAGING / "smoke.sh"

# Kept in lockstep with packaging/nerdit.spec::HIDDEN_IMPORTS. uvicorn picks
# these by string at runtime ("auto"), so no static analysis finds them.
UVICORN_HIDDEN_IMPORTS = (
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.websockets_sansio_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.loops.asyncio",
    "uvicorn.loops.uvloop",
)


def _sh_syntax_check(script: Path) -> subprocess.CompletedProcess[str]:
    sh = shutil.which("sh")
    assert sh, "no POSIX sh on PATH"
    return subprocess.run(
        [sh, "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_packaging_files_exist() -> None:
    for path in (
        SPEC,
        ASSEMBLE,
        SMOKE,
        PACKAGING / "entry_nerdit.py",
        PACKAGING / "entry_nerditd.py",
    ):
        assert path.is_file(), f"missing {path.relative_to(REPO_ROOT)}"


def test_spec_data_sources_exist() -> None:
    """The two data trees the spec copies into ``_internal/`` must be present.

    Both are read at runtime through ``importlib.resources.files()``, so they
    have to land on disk in the bundle — not merely inside the PYZ archive.
    """
    dashboard_index = REPO_ROOT / "src" / "nerdit" / "daemon" / "web" / "dist" / "index.html"
    templates = REPO_ROOT / "src" / "nerdit" / "config" / "app_templates" / "builtin.json"
    workflow_path = REPO_ROOT / ".github" / "workflows" / "release.yml"
    # builtin.json rides ``src/`` and so ships in both trees. Pin it before any
    # tree-shape branch below, or the public tree's skip would take it with it.
    assert templates.is_file(), f"{templates.relative_to(REPO_ROOT)} is missing"
    if dashboard_index.is_file():
        return
    # The dashboard bundle is not committed; the release workflow builds it
    # immediately before running the spec. When it is absent locally, the pin
    # moves to that workflow: the npm build step must precede the spec
    # invocation in the same job, or packaging would hit the spec's own "build
    # the dashboard first" error in CI.
    #
    # The public seed carries neither the bundle nor the workflow, so there is
    # nothing to pin there. The two trees are told apart by a file the publish
    # allowlist prunes: absent it, this is the public tree and the test skips;
    # present it, a missing release.yml is a failure, never a silent skip.
    private_marker = REPO_ROOT / "packaging" / "RELEASING.md"
    if not workflow_path.is_file() and not private_marker.is_file():
        pytest.skip(
            "public source tree: neither the dashboard bundle nor the release "
            "workflow ships, so there is nothing to pin"
        )
    workflow = workflow_path.read_text(encoding="utf-8")
    assert "npm run build" in workflow, (
        "dashboard dist is absent and release.yml carries no npm build step"
    )
    # Anchor on the invocation itself: comments name "nerdit.spec" and the pip
    # step installs "pyinstaller==…" earlier in the file.
    invocation = "pyinstaller --noconfirm"
    assert invocation in workflow
    assert workflow.index("npm run build") < workflow.index(invocation), (
        "release.yml must build the dashboard before running pyinstaller"
    )


@pytest.mark.parametrize("script", [ASSEMBLE, SMOKE], ids=["assemble.sh", "smoke.sh"])
def test_shell_scripts_parse(script: Path) -> None:
    result = _sh_syntax_check(script)
    assert result.returncode == 0, f"sh -n {script.name} failed:\n{result.stderr}"


def test_spec_references_both_entry_shims() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "entry_nerdit.py" in text
    assert "entry_nerditd.py" in text
    # One bundle, two executables, one shared payload.
    assert 'name="nerdit"' in text
    assert 'name="nerditd"' in text


@pytest.mark.parametrize("module", UVICORN_HIDDEN_IMPORTS)
def test_spec_declares_uvicorn_hidden_import(module: str) -> None:
    assert module in SPEC.read_text(encoding="utf-8"), (
        f"packaging/nerdit.spec no longer declares the hidden import {module!r}; "
        "a frozen daemon that loses it fails at boot with an ImportError"
    )


def test_spec_carries_both_runtime_data_trees() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "nerdit/daemon/web/dist" in text
    assert "nerdit/config/app_templates" in text


def test_assemble_owns_the_tarball_layout() -> None:
    """assemble.sh is the only place the inner layout is written down."""
    text = ASSEMBLE.read_text(encoding="utf-8")
    for member in (
        "nerdit.service",
        "nerdit-user.service",
        "ai.nerdit.daemon.plist",
        "install.sh",
        "VERSION",
        "_internal",
        "caddy",
    ):
        assert member in text, f"assemble.sh no longer stages {member}"
    # Artifact name grammar: nerdit-<version>-<os>-<arch>.tar.gz
    assert 'ARTIFACT="nerdit-$VERSION-$OS-$ARCH.tar.gz"' in text


def test_smoke_never_uses_the_forbidden_teardowns() -> None:
    """``nerdit daemon restart`` is interactive and ``pkill`` is indiscriminate.

    Neither belongs in automation running next to a real daemon. Comment lines
    are stripped first: both prohibitions are *explained* in the smoke script's
    header, and the point is that no executable line does either.
    """
    code = [
        line
        for line in SMOKE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    for line in code:
        assert "pkill" not in line, f"smoke.sh must not pkill: {line.strip()}"
        assert "daemon restart" not in line, (
            f"smoke.sh must not shell out to the interactive verb: {line.strip()}"
        )
