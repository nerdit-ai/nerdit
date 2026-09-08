"""The release version line is pinned in exactly two places — keep them equal.

``pyproject.toml`` `[project].version` is what a built wheel/sdist and the
frozen PyInstaller bundle carry; ``nerdit.__version__`` is what
``nerdit --version``, ``GET /health``, ``GET /cluster/info`` and the P27
node-link hello report. P30 makes the install channel real,
so a drift between the two now means an installed node lies about which release
it is running — this test is the guard (WP-0, release coherence).

Deliberately dependency-free: ``tomllib`` is stdlib on the 3.11 floor.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import nerdit

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    with _PYPROJECT.open("rb") as fh:
        return str(tomllib.load(fh)["project"]["version"])


def test_pyproject_and_package_version_agree() -> None:
    assert _pyproject_version() == nerdit.__version__, (
        "pyproject.toml [project].version and nerdit.__version__ disagree — "
        "bump both (they are two separate strings by design; see the P30 WP-0 note)."
    )


def test_version_is_a_plain_release_string() -> None:
    """No `v` prefix in either string — the `v` belongs to the git tag only."""
    version = nerdit.__version__
    assert not version.startswith("v"), version
    parts = version.split(".")
    assert len(parts) == 3, version
    assert all(p.isdigit() for p in parts), version


def test_openapi_document_reports_the_package_version() -> None:
    """The FastAPI document must not carry a frozen literal.

    ``GET /api/openapi.json`` and ``/api/docs`` are what an agent (and the
    cloud console) read to learn which daemon they are talking to; a hardcoded
    string there made a 0.5.0 daemon advertise 0.4.0. Read the source rather
    than building an app: importing ``appfactory`` is cheap, constructing the
    application is not.
    """
    source = (
        Path(__file__).resolve().parent.parent / "src" / "nerdit" / "daemon" / "appfactory.py"
    ).read_text(encoding="utf-8")
    assert "version=__version__," in source, (
        "daemon/appfactory.py must pass nerdit.__version__ to FastAPI(...), "
        "never a literal — see the P30 WP-0 release-coherence note."
    )
