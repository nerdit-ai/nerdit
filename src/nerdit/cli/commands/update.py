"""Update or downgrade an installed release through its recorded installer.

Require ownership, confirm the version change, then run a copy of install.sh
so replacing the current symlink cannot change the script mid-read. The installer
verifies signed releases and restarts the unit. Source checkouts are refused;
no background updates run and the data dir is preserved.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import typer

from nerdit.cli.display import _plain, console
from nerdit.utils.install_layout import detect_install_layout


def update(
    version: Optional[str] = typer.Option(  # noqa: UP007 — Typer needs Optional[]
        None,
        "--version",
        help="Install this exact release (e.g. 0.5.0) instead of the latest. Downgrades too.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt for confirmation."),
) -> None:
    """Update (or downgrade) an installed nerdit to the latest or a pinned release."""
    layout = detect_install_layout()
    if layout is None:
        console.print(
            "[red]nerdit update only manages installs made by the get.nerdit.ai "
            "installer — this looks like a source checkout (use git pull / "
            "pip install -e . instead).[/red]"
        )
        raise typer.Exit(1)

    try:
        owner_uid = layout.root.stat().st_uid
    except OSError as exc:
        console.print(f"[red]Cannot read {_plain(layout.root)}: {_plain(exc)}[/red]")
        raise typer.Exit(1) from exc
    if owner_uid != os.geteuid():
        console.print(
            f"[red]The install at {_plain(layout.root)} is owned by uid {owner_uid}; "
            "run nerdit update as that user (or with sudo).[/red]"
        )
        raise typer.Exit(1)

    console.print(
        f"nerdit {_plain(layout.current_version or 'unknown')} → {_plain(version or 'latest')}"
    )
    if not yes and not typer.confirm("Run the installer now?", default=True):
        console.print("[red]Aborted — nothing was changed.[/red]")
        raise typer.Exit(1)

    if not layout.installer.is_file():
        console.print(
            f"[red]No installer recorded at {_plain(layout.installer)}. Re-run the "
            "install command to repair it: curl -fsSL https://get.nerdit.ai | sh[/red]"
        )
        raise typer.Exit(1)

    env = dict(os.environ)
    env["NERDIT_INSTALL_MODE"] = layout.mode
    # Explicitly disable linking during updates, independently of installer detection.
    # This guard runs before any inherited NERDIT_AUTH_KEY is read or prompts begin.
    env["NERDIT_SKIP_LINK"] = "1"
    if version:
        env["NERDIT_VERSION"] = version
    else:
        # A bare `nerdit update` means "latest", and says so on stdout. An
        # exported NERDIT_VERSION in the caller's environment would otherwise
        # survive into the installer and silently install — or DOWNGRADE to —
        # that version instead, contradicting what was just printed.
        env.pop("NERDIT_VERSION", None)

    tmpdir = tempfile.mkdtemp(prefix="nerdit-update-")
    try:
        # The installer flips <ROOT>/current, which replaces install.sh underneath
        # a shell that reads its script incrementally — run a copy instead.
        script = Path(tmpdir) / "install.sh"
        shutil.copyfile(layout.installer, script)
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["sh", str(script)],
            env=env,
            check=False,
        )
    except OSError as exc:
        console.print(f"[red]Could not run the installer: {_plain(exc)}[/red]")
        raise typer.Exit(1) from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if proc.returncode != 0:
        # The installer already printed its own diagnosis; propagate the code so
        # a wrapping script sees the failure.
        raise typer.Exit(proc.returncode)
