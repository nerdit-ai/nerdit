# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Nerdit node bundle (P30 WP-1, D-P30-1).

ONE onedir bundle exposing BOTH entry points. Two ``Analysis`` objects (the CLI
and the daemon) feed two ``EXE`` objects that a single ``COLLECT`` gathers into
``dist/nerdit/``:

    dist/nerdit/
      nerdit        # frozen CLI
      nerditd       # frozen daemon
      _internal/    # the shared payload (stdlib, deps, data files)

Sharing one ``_internal`` is the whole point: the two executables differ only in
their embedded PYZ script, so a second copy of fastapi/uvicorn/cryptography
would roughly double the tarball for nothing. ``COLLECT`` normalizes the merged
table of contents, so the analyses' overlapping binaries/datas dedupe.

Build:  pyinstaller packaging/nerdit.spec        (see packaging/README.md)
Package: sh packaging/assemble.sh --dist dist/nerdit …

Nothing here downloads anything: the pinned Caddy binary, the service-unit
templates and ``install.sh`` are injected by ``assemble.sh``, which is the
single owner of the tarball layout.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# ``SPECPATH`` is injected by PyInstaller when it execs this file. Every path
# below is resolved against it so the spec builds identically from any cwd.
HERE = Path(SPECPATH).resolve()  # noqa: F821
REPO = HERE.parent
SRC = REPO / "src"

ENTRY_CLI = HERE / "entry_nerdit.py"
ENTRY_DAEMON = HERE / "entry_nerditd.py"

# --- data files (CLAUDE.md "Container images"/§1 of the P30 plan) -----------
# The dashboard bundle (built, not committed — see .gitignore and the release
# workflow's dashboard build step) and the embedded app-template catalog are
# read at runtime through ``importlib.resources.files()``, so they must land
# on disk inside ``_internal/nerdit/...`` — not merely inside the PYZ archive.
DASHBOARD_DIST = SRC / "nerdit" / "daemon" / "web" / "dist"
TEMPLATES_JSON = SRC / "nerdit" / "config" / "app_templates" / "builtin.json"

# Fail the build loudly (naming the file) rather than shipping a bundle whose
# dashboard 503s or whose store is empty.
assert DASHBOARD_DIST.is_dir(), (
    f"missing dashboard bundle: {DASHBOARD_DIST} — run "
    "`cd src/nerdit/daemon/web && npm install && npm run build` first"
)
assert (DASHBOARD_DIST / "index.html").is_file(), (
    f"missing dashboard entry file: {DASHBOARD_DIST / 'index.html'}"
)
assert TEMPLATES_JSON.is_file(), f"missing app-template catalog: {TEMPLATES_JSON}"

DATAS = [
    (str(DASHBOARD_DIST), "nerdit/daemon/web/dist"),
    (str(TEMPLATES_JSON), "nerdit/config/app_templates"),
]

# ``nerdit``'s own dist-info: importlib.metadata is consulted by a few
# libraries' plugin machinery; carrying the metadata is cheap insurance.
DATAS += copy_metadata("nerdit")
# Fail the build if mDNS is absent from the clean build environment.
DATAS += copy_metadata("zeroconf")
try:
    # pydantic v2's plugin loader scans ``importlib.metadata.distributions()``;
    # without its metadata the scan is merely empty, so this is a mitigation,
    # not a requirement — never fail the build over it.
    DATAS += copy_metadata("pydantic")
except Exception:  # noqa: BLE001 — best effort, see above
    pass

# --- hidden imports --------------------------------------------------------
# uvicorn resolves its protocol/lifespan/loop implementations by string at
# runtime ("auto"), which static analysis cannot follow. Listing them on both
# analyses is harmless (the CLI simply never imports them) and keeps the two
# graphs identical, which helps COLLECT dedupe.
HIDDEN_IMPORTS = [
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
    # The streamable-HTTP MCP transport is imported lazily (only when
    # [mcp].http_enabled is set), so name it and collect the mcp package.
    "nerdit.mcp.server",
    *collect_submodules("mcp"),
    # Zeroconf wheels contain extension modules with runtime imports.
    *collect_submodules("zeroconf"),
]

# tkinter is never imported by Nerdit and drags in a large native toolkit.
#
# ``pydantic.mypy`` / ``pydantic.v1.mypy`` are pydantic's own mypy PLUGINS. They
# import the whole of mypy at module level, and PyInstaller's static graph
# follows that edge whenever mypy happens to be installed in the build
# environment — which silently drags ~20 MB of type checker into a customer
# artifact. Nothing imports them at runtime. (The durable fix is a clean build
# venv, see packaging/README.md; this makes the artifact right either way.)
EXCLUDES = [
    "tkinter",
    "pydantic.mypy",
    "pydantic.v1.mypy",
    "mypy",
]

_common = dict(
    pathex=[str(SRC)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[str(HERE / "hooks")] if (HERE / "hooks").is_dir() else [],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

a_cli = Analysis([str(ENTRY_CLI)], **_common)
a_daemon = Analysis([str(ENTRY_DAEMON)], **_common)

pyz_cli = PYZ(a_cli.pure)  # noqa: F821
pyz_daemon = PYZ(a_daemon.pure)  # noqa: F821

exe_cli = EXE(  # noqa: F821
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="nerdit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

exe_daemon = EXE(  # noqa: F821
    pyz_daemon,
    a_daemon.scripts,
    [],
    exclude_binaries=True,
    name="nerditd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe_cli,
    exe_daemon,
    a_cli.binaries,
    # ``zipfiles`` is always empty since PyInstaller 6 dropped zipped-egg
    # support; kept explicit so a future reader does not think it was forgotten.
    a_cli.zipfiles,
    a_cli.datas,
    a_daemon.binaries,
    a_daemon.zipfiles,
    a_daemon.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="nerdit",
)
