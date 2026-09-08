"""Tests for ``nerdit update`` (P30 WP-4, D-P30-6/9).

The verb owns four refusals and one hand-off, and that is exactly what is
pinned here: a source checkout, a foreign owner, a missing recorded installer,
a declined confirmation — then the hand-off itself, which must run a **copy**
of ``<ROOT>/current/install.sh`` (the symlink flip replaces the original while
``sh`` is still reading it) with ``NERDIT_INSTALL_MODE`` and, when pinned,
``NERDIT_VERSION`` in the child environment, and must propagate the child's
exit code.

No network, no daemon, no real installer: ``subprocess.run`` is patched and the
layout is a real directory tree under ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.commands import update as update_mod
from nerdit.cli.commands.update import update
from nerdit.utils.install_layout import InstallLayout


def _layout(tmp_path: Path, *, mode: str = "user", installer: bool = True) -> InstallLayout:
    root = tmp_path / ".nerdit"
    version_dir = root / "versions" / "0.5.0"
    version_dir.mkdir(parents=True)
    if installer:
        (version_dir / "install.sh").write_text("#!/bin/sh\necho installing\n")
    (root / "current").symlink_to(Path("versions") / "0.5.0")
    return InstallLayout(
        mode=mode,  # type: ignore[arg-type]
        root=root,
        versions_dir=root / "versions",
        current=root / "current",
        current_version="0.5.0",
        shim=root / "bin" / "nerdit",
        installer=root / "current" / "install.sh",
    )


class _Completed:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode


def _patch_run(monkeypatch, returncode: int = 0) -> list[dict]:
    calls: list[dict] = []

    def _run(argv, **kwargs):
        # Read the copied script HERE: the real installer flips `current`, so
        # the copy must still be readable at exec time.
        calls.append(
            {
                "argv": list(argv),
                "env": dict(kwargs.get("env") or {}),
                "content": Path(argv[1]).read_text(),
                "exists": Path(argv[1]).exists(),
            }
        )
        return _Completed(returncode)

    monkeypatch.setattr(update_mod.subprocess, "run", _run)
    return calls


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_source_checkout_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: None)
    monkeypatch.setattr(
        update_mod.subprocess, "run", lambda *a, **k: pytest.fail("must not run anything")
    )

    with pytest.raises(typer.Exit) as excinfo:
        update(version=None, yes=True)
    assert excinfo.value.exit_code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "get.nerdit.ai installer" in out
    assert "source checkout" in out


def test_foreign_owner_is_refused(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid + 1)
    monkeypatch.setattr(
        update_mod.subprocess, "run", lambda *a, **k: pytest.fail("must not run anything")
    )

    with pytest.raises(typer.Exit) as excinfo:
        update(version=None, yes=True)
    assert excinfo.value.exit_code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "is owned by uid" in out
    assert "sudo" in out


def test_missing_installer_is_refused(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path, installer=False)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    monkeypatch.setattr(
        update_mod.subprocess, "run", lambda *a, **k: pytest.fail("must not run anything")
    )

    with pytest.raises(typer.Exit) as excinfo:
        update(version=None, yes=True)
    assert excinfo.value.exit_code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "No installer recorded" in out
    assert "get.nerdit.ai" in out


def test_declined_confirmation_changes_nothing(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    monkeypatch.setattr(
        update_mod.subprocess, "run", lambda *a, **k: pytest.fail("must not run anything")
    )

    with pytest.raises(typer.Exit) as excinfo:
        update(version=None, yes=False)
    assert excinfo.value.exit_code == 1
    assert "Aborted" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# hand-off
# --------------------------------------------------------------------------- #


def test_happy_path_runs_a_copy_of_the_recorded_installer(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)

    assert len(calls) == 1
    call = calls[0]
    assert call["argv"][0] == "sh"
    script = Path(call["argv"][1])
    # A COPY, not the recorded path: the symlink flip would replace the file
    # mid-read (sh reads its script incrementally).
    assert script != layout.installer
    assert call["content"] == layout.installer.read_text()
    assert call["env"]["NERDIT_INSTALL_MODE"] == "user"
    assert "NERDIT_VERSION" not in call["env"]
    # The temp copy is cleaned up afterwards.
    assert not script.exists()
    out = " ".join(capsys.readouterr().out.split())
    assert "0.5.0 → latest" in out


def test_version_pin_reaches_the_installer_env(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    update(version="0.4.0", yes=True)

    assert calls[0]["env"]["NERDIT_VERSION"] == "0.4.0"
    assert calls[0]["env"]["NERDIT_INSTALL_MODE"] == "user"
    assert "0.5.0 → 0.4.0" in " ".join(capsys.readouterr().out.split())


def test_system_mode_is_passed_through(monkeypatch, tmp_path):
    layout = _layout(tmp_path, mode="system")
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)
    assert calls[0]["env"]["NERDIT_INSTALL_MODE"] == "system"


def test_update_sets_nerdit_skip_link_beside_the_version_env_handling(monkeypatch, tmp_path):
    """(P34 D2) The update path's OWN link guard, independent of the installer's.

    The installer skips its link step when the CURRENT_LINK probe says this is
    an update (IS_UPDATE=1); NERDIT_SKIP_LINK=1 is the second, explicit guard
    the plan mandates so a future refactor of that derivation cannot make a
    fleet update re-link, re-prompt for a key, or park a non-interactive
    updater on a browser approval. Two guards, one intent, either alone enough.
    """
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)

    assert calls[0]["env"]["NERDIT_SKIP_LINK"] == "1"


def test_update_passes_no_positional_arguments_to_the_reexecuted_installer(monkeypatch, tmp_path):
    """(P34 D2) The re-exec is ``["sh", <script>]`` and nothing else, ever.

    The D2 argument loop gives install.sh flags (--key, --key-file, --no-link,
    --link-timeout, --version); the update path must never grow one. Flags on
    this argv would be a third channel into the link step — one that bypasses
    both guards above — and a --key here would put the secret on a world-
    readable /proc/<pid>/cmdline for the whole install (D-X16-O11). Env vars
    are the update path's only channel, by decision.
    """
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)

    argv = calls[0]["argv"]
    assert argv[0] == "sh"
    assert len(argv) == 2, argv


def test_installer_failure_propagates_the_exit_code(monkeypatch, tmp_path):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    _patch_run(monkeypatch, returncode=7)

    with pytest.raises(typer.Exit) as excinfo:
        update(version=None, yes=True)
    assert excinfo.value.exit_code == 7


def test_yes_skips_the_confirmation(monkeypatch, tmp_path):
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: pytest.fail("--yes must not prompt"))
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)
    assert len(calls) == 1


def test_unknown_current_version_still_renders_a_delta(monkeypatch, tmp_path, capsys):
    layout = _layout(tmp_path)
    dangling = InstallLayout(
        mode=layout.mode,
        root=layout.root,
        versions_dir=layout.versions_dir,
        current=layout.current,
        current_version=None,
        shim=layout.shim,
        installer=layout.installer,
    )
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: dangling)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    _patch_run(monkeypatch)

    update(version=None, yes=True)
    assert "unknown → latest" in " ".join(capsys.readouterr().out.split())


# --------------------------------------------------------------------------- #
# argv-level
# --------------------------------------------------------------------------- #


def test_cli_update_is_registered_and_parses_its_options(monkeypatch, tmp_path):
    from nerdit.cli.app import app

    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    calls = _patch_run(monkeypatch)

    result = CliRunner().invoke(app, ["update", "--version", "0.6.0", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls[0]["env"]["NERDIT_VERSION"] == "0.6.0"


def test_cli_update_help_does_not_collide_with_the_root_version_flag(monkeypatch):
    """The subcommand's --version takes a value; the root -v/--version is eager
    and prints the CLI version. Both must keep working."""
    import re

    from nerdit.cli.app import app

    # Rich lays the help panel out to the terminal width and truncates the
    # option column when it is narrow; a headless CI runner is narrower than a
    # dev terminal, which is why this passed locally and failed on CI. Pin the
    # width (repo idiom) and compare against ANSI-stripped text.
    monkeypatch.setenv("COLUMNS", "300")
    runner = CliRunner()
    result = runner.invoke(app, ["update", "--help"])
    assert result.exit_code == 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--version" in plain, plain

    root = runner.invoke(app, ["--version"])
    assert root.exit_code == 0, root.output
    assert "nerdit " in root.output


def test_bare_update_ignores_an_ambient_version_pin(monkeypatch, tmp_path):
    """An exported NERDIT_VERSION must not silently pin — or downgrade — an
    update the CLI just announced as "latest" (review finding)."""
    layout = _layout(tmp_path)
    monkeypatch.setattr(update_mod, "detect_install_layout", lambda: layout)
    monkeypatch.setattr(update_mod.os, "geteuid", lambda: os.stat(layout.root).st_uid)
    monkeypatch.setenv("NERDIT_VERSION", "0.1.0")
    calls = _patch_run(monkeypatch)

    update(version=None, yes=True)

    assert "NERDIT_VERSION" not in calls[0]["env"], "ambient pin leaked into the installer"
