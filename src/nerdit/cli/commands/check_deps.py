"""Render shared dependency checks and offer supported apt-based installations."""

from __future__ import annotations

import json as _json
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from nerdit.cli import checks
from nerdit.cli.checks import CheckResult, platform_label, run_checks
from nerdit.cli.display import console

# ---------------------------------------------------------------------------
# Install helpers (apt/curl-based, Linux only — side-effecting, kept here)
# ---------------------------------------------------------------------------


def _run_install(cmd: list[str]) -> bool:
    """Run an install command with user-visible output. Returns success."""
    console.print(f"  [dim]Running: {' '.join(cmd)}[/dim]")
    try:
        result = subprocess.run(cmd, timeout=300)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        console.print(f"  [red]Failed: {exc}[/red]")
        return False


def _install_python() -> bool:
    return _run_install(
        ["sudo", "apt-get", "install", "-y", "python3.11"],
    )


def _install_nvidia_driver() -> bool:
    ok = _run_install(
        ["sudo", "apt-get", "install", "-y", "nvidia-driver-535"],
    )
    if ok:
        console.print("  [yellow]A reboot may be required.[/yellow]")
    return ok


def _install_docker() -> bool:
    ok = _run_install(
        ["sh", "-c", "curl -fsSL https://get.docker.com | sudo sh"],
    )
    if ok:
        subprocess.run(
            ["sudo", "usermod", "-aG", "docker", str(Path.home().name)],
            timeout=10,
        )
        console.print("  [yellow]Log out and back in for group changes to take effect.[/yellow]")
    return ok


def _install_nvidia_toolkit() -> bool:
    cmds = [
        "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
        " | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
        "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/"
        "nvidia-container-toolkit.list"
        " | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/"
        "nvidia-container-toolkit-keyring.gpg] https://#g'"
        " | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        " > /dev/null",
        "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
        "sudo nvidia-ctk runtime configure --runtime=docker",
        "sudo systemctl restart docker",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, timeout=120)  # noqa: S602
        if result.returncode != 0:
            console.print(f"  [red]Step failed: {cmd}[/red]")
            return False
    return True


def _find_dockerfile() -> Path | None:
    """The runtime Dockerfile shipped beside this package, never the CWD's.

    A CWD lookup would build whatever repo the user happens to stand in and tag
    it `nerdit-runtime:0.1`.
    """
    path = Path(__file__).resolve().parents[4] / "docker" / "Dockerfile"
    return path if path.is_file() else None


def _install_runtime_image() -> bool:
    dockerfile = _find_dockerfile()
    if dockerfile is None:
        from nerdit.utils.install_layout import is_frozen

        if is_frozen():
            # (P30) The release tarball deliberately carries no docker/ tree, so
            # this is unreachable-by-construction rather than unlucky — say so
            # instead of implying a missing file.
            console.print(
                "  [yellow]The installed release does not bundle the runtime Dockerfile.[/yellow]"
            )
            console.print(
                "  [dim]Build it from a source checkout "
                "(docker build -t nerdit-runtime:0.1 docker/), or skip it — "
                "deployed apps use buildpack-built images, not this one.[/dim]"
            )
            return False
        console.print("  [red]Dockerfile not found. Cannot build image.[/red]")
        return False
    return _run_install(
        ["docker", "build", "-t", "nerdit-runtime:0.1", str(dockerfile.parent)],
    )


# Rows that have an automatic (apt/curl-based, Linux) installer, keyed on the
# stable check name.
_INSTALLERS = {
    checks.NAME_PYTHON: _install_python,
    checks.NAME_NVIDIA_DRIVER: _install_nvidia_driver,
    checks.NAME_DOCKER: _install_docker,
    checks.NAME_NVIDIA_TOOLKIT: _install_nvidia_toolkit,
    checks.NAME_RUNTIME_IMAGE: _install_runtime_image,
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STATUS_CELL = {
    "ok": "[green]✓[/green]",
    "warn": "[yellow]⚠[/yellow]",
    "fail": "[red]✗[/red]",
    "skip": "[dim]n/a[/dim]",
}


def _render_table(results: list[CheckResult], label: str) -> None:
    from rich.markup import escape
    from rich.table import Table

    table = Table(title=f"Nerdit — Dependency Check ({escape(label)})")
    table.add_column("Dependency", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    for r in results:
        status = _STATUS_CELL.get(r.status, r.status)
        # Escape dynamic values so config lines like [daemon].port or nerdit[mdns]
        # in details/remediation are not swallowed as Rich markup.
        detail = escape(r.detail)
        if r.status != "ok":
            detail = f"[yellow]{detail}[/yellow]"
        table.add_row(escape(r.name), status, detail)

    console.print()
    console.print(table)
    console.print()

    actionable = [r for r in results if r.status != "ok" and r.remediation]
    if actionable:
        console.print("[bold]Remediation:[/bold]")
        for r in actionable:
            console.print(f"  [dim]{escape(r.name)}:[/dim] {escape(r.remediation or '')}")
        console.print()


def _emit_json(results: list[CheckResult], label: str) -> None:
    payload = {
        "platform": label,
        "checks": [
            {
                "name": r.name,
                "status": r.status,
                "detail": r.detail,
                "remediation": r.remediation,
            }
            for r in results
        ],
    }
    typer.echo(_json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def check_deps(
    install: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Offer to install missing dependencies (with confirmation)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the check results as JSON (for scripting) instead of a table",
    ),
) -> None:
    """Check system dependencies and optionally install missing ones."""
    results = run_checks()
    label = platform_label()
    failures = [r for r in results if r.status == "fail"]

    if json_output:
        _emit_json(results, label)
        if failures:
            raise typer.Exit(1)
        return

    _render_table(results, label)

    if not failures:
        warnings = [r for r in results if r.status == "warn"]
        console.print("[bold green]All required dependencies are installed.[/bold green]")
        if warnings:
            console.print(
                f"[dim]{len(warnings)} optional component(s) missing — see remediation above.[/dim]"
            )
        return

    console.print(
        f"[yellow]{len(failures)} required dependenc"
        f"{'y' if len(failures) == 1 else 'ies'} missing.[/yellow]"
    )

    if not install:
        console.print("[dim]Run with --install to interactively install missing deps.[/dim]")
        raise typer.Exit(1)

    _run_install_flow(failures)


def _print_manual_steps(failures: list[CheckResult]) -> None:
    from rich.markup import escape

    for r in failures:
        if r.remediation:
            console.print(f"  [dim]{escape(r.name)}:[/dim] {escape(r.remediation)}")


def _run_install_flow(failures: list[CheckResult]) -> None:
    """Interactive apt-based install flow. Non-apt distros / macOS get manual
    steps instead of a failing apt invocation."""
    if sys.platform != "linux":
        # macOS has no apt/curl installers — the remediation lines are the path.
        console.print()
        console.print("[dim]No automatic installers on this platform — manual steps:[/dim]")
        _print_manual_steps(failures)
        raise typer.Exit(1)

    if shutil.which("apt-get") is None:
        console.print()
        console.print("[yellow]Unsupported distro (no apt-get) — manual steps:[/yellow]")
        _print_manual_steps(failures)
        raise typer.Exit(1)

    from rich.markup import escape

    console.print()
    any_installed = False
    wsl2 = checks.detect_wsl2()
    for r in failures:
        install_fn = _INSTALLERS.get(r.name)
        if wsl2 and r.name == checks.NAME_NVIDIA_DRIVER:
            # The driver lives on the Windows host under WSL2 — running the
            # Linux apt installer inside the distro can break GPU passthrough.
            # Fall through to the manual line (its remediation is WSL-safe).
            install_fn = None
        if install_fn is None:
            hint = f" — {escape(r.remediation)}" if r.remediation else ""
            console.print(f"[dim]{escape(r.name)}: no automatic installer{hint}[/dim]")
            continue
        if typer.confirm(f"Install {r.name}?", default=False):
            if install_fn():
                console.print(f"  [green]✓ {r.name} installed[/green]")
                any_installed = True
            else:
                console.print(f"  [red]✗ {r.name} installation failed[/red]")
        else:
            console.print(f"  [dim]Skipped {r.name}[/dim]")

    if not any_installed:
        raise typer.Exit(1)

    # Re-check after installs.
    from rich.markup import escape

    console.print("\n[bold]Re-checking...[/bold]\n")
    still_missing = 0
    for r in run_checks():
        if r.status not in ("ok", "warn", "skip"):
            still_missing += 1
        status = _STATUS_CELL.get(r.status, r.status)
        console.print(f"  {status} {escape(r.name)}: {escape(r.detail)}")
    if still_missing == 0:
        console.print("\n[bold green]All required dependencies are now installed.[/bold green]")
    else:
        console.print(
            f"\n[yellow]{still_missing} dependenc"
            f"{'y' if still_missing == 1 else 'ies'} still missing.[/yellow]"
        )
        raise typer.Exit(1)
