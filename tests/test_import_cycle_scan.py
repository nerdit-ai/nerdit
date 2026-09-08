"""Self-tests for ``scripts/import_cycle_scan.py`` (B39).

The scanner is the acceptance instrument for the Track B "0 import cycles"
metric (trackb-plan §6). B39 found it blind to package-relative imports —
exactly the idiom the split packages (``db/queries/``, ``core/proxy/``,
``mcp/tools/``) use — so a seeded cycle in each spelling is pinned here:
both forms must be caught, and a cycle-free tree must stay green.

The script hardcodes ``SRC = Path("src/nerdit")`` relative to the cwd, so
each case runs it as a subprocess from a scratch tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "import_cycle_scan.py"


def _scan(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _seed(tree: Path, a_src: str, b_src: str) -> None:
    pkg = tree / "src" / "nerdit" / "pkg"
    pkg.mkdir(parents=True)
    (tree / "src" / "nerdit" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    (pkg / "a.py").write_text(a_src)
    (pkg / "b.py").write_text(b_src)


@pytest.mark.parametrize(
    ("a_src", "b_src"),
    [
        pytest.param("from .b import x\ny = 1\n", "from .a import y\nx = 1\n", id="relative"),
        pytest.param(
            "from nerdit.pkg.b import x\ny = 1\n",
            "from nerdit.pkg.a import y\nx = 1\n",
            id="absolute",
        ),
        pytest.param(
            "from . import b\nfrom .b import x\ny = 1\n",
            "from ..pkg import a\nx = 1\n",
            id="mixed-levels",
        ),
    ],
)
def test_seeded_cycle_is_caught(tmp_path: Path, a_src: str, b_src: str) -> None:
    _seed(tmp_path, a_src, b_src)
    result = _scan(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "CYCLE:" in result.stdout
    assert "nerdit.pkg.a" in result.stdout
    assert "nerdit.pkg.b" in result.stdout


def test_acyclic_tree_is_green(tmp_path: Path) -> None:
    _seed(tmp_path, "from . import b\nfrom .b import x\n", "x = 1\n")
    result = _scan(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 cycles" in result.stdout


def test_deferred_relative_import_is_tagged(tmp_path: Path) -> None:
    """A function-body relative import still creates a (deferred) edge."""
    _seed(
        tmp_path,
        "def f():\n    from .b import x\n    return x\n",
        "from .a import f\nx = 1\n",
    )
    result = _scan(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[deferred]" in result.stdout
