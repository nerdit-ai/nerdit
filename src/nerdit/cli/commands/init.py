"""nerdit init — Initialize Nerdit configuration and start daemon."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from nerdit.cli import checks
from nerdit.cli.display import console
from nerdit.config.defaults import DEFAULT_HOST, DEFAULT_PORT, NVIDIA_LIB_DIRS
from nerdit.config.settings import generate_auth_token, get_client_config

if TYPE_CHECKING:
    from nerdit.daemon.lifecycle import DaemonLifecycle


def _generate_default_config() -> tuple[str, str]:
    """Generate default config TOML with a fresh auth token.

    Returns (config_text, token).
    """
    token = generate_auth_token()
    config = f"""\
[nerdit]
data_dir = "~/.nerdit"
log_level = "info"

[daemon]
host = "0.0.0.0"
port = 9321
pid_file = "~/.nerdit/nerditd.pid"
auth_token = "{token}"

[containers]
runtime = "docker"
default_image = "nerdit-runtime:0.1"
cache_dir = "~/.nerdit/cache"

[monitor]
interval_seconds = 5
gpu_temp_warning = 80
gpu_temp_critical = 90
"""
    return config, token


def _write_installer_auth_config(data_dir: Path) -> str:
    """Ensure an installer-created daemon has an auth token before first startup.

    Tokenless loopback requests are admin requests, including other local users.
    Preserve existing TOML comments and layout; insert only the token, leaving
    bind settings unchanged. Test token presence, not merely file existence.

    Returns:
        "present" if already configured, "written" after inserting a token, or
        "failed" when the caller must warn that no usable token was written.
    """
    config_path = data_dir / "config.toml"

    if config_path.exists():
        try:
            with config_path.open("rb") as fh:
                existing = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return "failed"
        if str(existing.get("daemon", {}).get("auth_token") or "").strip():
            # A hand-written or copied config can be world-readable, and it
            # holds the daemon's ADMIN bearer token. Tighten it even though we
            # write nothing: leaving it at 0644 under a traversable ~/.nerdit
            # hands every local account admin on the very install whose whole
            # point is that they should not have it.
            _harden_secret_file(config_path)
            return "present"
        try:
            text = config_path.read_text()
        except OSError:
            return "failed"
        line = f'auth_token = "{generate_auth_token()}"'
        lines = text.splitlines()
        for i, raw in enumerate(lines):
            if _is_daemon_table_header(raw):
                lines.insert(i + 1, line)
                break
        else:
            lines.append("")
            lines.append("[daemon]")
            lines.append(line)
        updated = "\n".join(lines) + "\n"
        # Re-parse before committing: a malformed write here would take the
        # daemon down, which is far worse than the hazard being closed.
        try:
            if not str(tomllib.loads(updated).get("daemon", {}).get("auth_token") or "").strip():
                return "failed"
        except tomllib.TOMLDecodeError:
            return "failed"
        return "written" if _replace_secret_file(config_path, updated) else "failed"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        data_dir.chmod(0o700)
    except OSError:
        pass
    body = f'[daemon]\nauth_token = "{generate_auth_token()}"\n'
    return "written" if _replace_secret_file(config_path, body) else "failed"


def _is_daemon_table_header(line: str) -> bool:
    """Recognize a daemon table header, allowing TOML whitespace and trailing comments."""
    stripped = line.strip()
    if not stripped.startswith("[") or stripped.startswith("[["):
        return False
    end = stripped.find("]")
    if end == -1:
        return False
    trailing = stripped[end + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return False
    return stripped[1:end].strip().strip("\"'") == "daemon"


def _harden_secret_file(path: Path) -> None:
    """Best-effort `chmod 0600` on a file that holds the admin token."""
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _replace_secret_file(path: Path, content: str) -> bool:
    """Atomically write secret content without a world-readable interval.

    Create an exclusive 0600 temporary file, then rename it into place.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            tmp.unlink()
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError:
            return False
    except OSError:
        return False
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


NERDIT_TOML_TEMPLATE = """\
# nerdit.toml — Project configuration for Nerdit
# Place this file at the root of your project.
# `nerdit deploy` reads these defaults. Uncomment what you need — every section
# below is optional, and `[deploy].name` is required as soon as `[deploy]` exists.

# [deploy]
# name = "my-app"
# port = 8000
# gpus = 0
# start = "npm start"
# volumes = ["data:/data"]

# Declare the AI the app needs; Nerdit wires it to a local model or an API.
# [ai.default]
# provider = "ollama"          # or "api"
# model = "llama3.1:8b"
# base_url = "https://api.openai.com/v1"   # provider = "api" only
# api_key = "${secrets.OPENAI_API_KEY}"    # secret references only

# Declare a managed database (see `nerdit db create`).
# [db.default]
# provider = "managed"
# database = "my-postgres"
"""


def _check_docker() -> bool:
    """Return whether Docker is available, using the shared dependency checks."""
    return checks.check_docker(sys.platform, checks.detect_wsl2()).status == "ok"


def _offer_start_docker() -> bool:
    """If Docker is down, offer to start it via systemctl. Returns True once it's up."""
    import shutil
    import time

    if shutil.which("systemctl") is None:
        # No systemd (e.g. macOS / Docker Desktop) — nothing we can safely auto-start.
        return False

    console.print("[yellow]Docker does not appear to be running.[/yellow]")
    if not typer.confirm("Try to start it now with `sudo systemctl start docker`?"):
        return False

    try:
        result = subprocess.run(["sudo", "systemctl", "start", "docker"], timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        console.print(f"[red]Could not start Docker:[/red] {exc}")
        return False

    if result.returncode != 0:
        console.print(
            "[red]`systemctl start docker` failed.[/red] "
            "Start Docker manually, then re-run `nerdit init`."
        )
        return False

    time.sleep(2)  # give dockerd a moment to accept connections
    if _check_docker():
        console.print("[green]Docker started.[/green]")
        return True

    console.print(
        "[yellow]Docker still not responding — give it a few seconds "
        "and re-run `nerdit init`.[/yellow]"
    )
    return False


def _check_nvidia_toolkit() -> bool:
    """Check if the NVIDIA Container Toolkit is installed in Docker.

    Delegates to the shared `nerdit.cli.checks` registry.
    """
    return checks.check_nvidia_toolkit().status == "ok"


def _check_nvidia_driver_lib() -> bool:
    """Check NVIDIA library availability, including the shared WSL2 passthrough paths."""
    return any((Path(d) / "libnvidia-ml.so.1").exists() for d in NVIDIA_LIB_DIRS)


def init(
    project: bool = typer.Option(
        False, "--project", help="Generate a nerdit.toml in the current directory"
    ),
    auth_token_only: bool = typer.Option(
        False,
        "--auth-token-only",
        help="Write only [daemon].auth_token if absent, then exit (installer use).",
        hidden=True,
    ),
) -> None:
    """Initialize Nerdit: configuration, daemon, and GPU detection."""
    if project and auth_token_only:
        console.print("[red]--project and --auth-token-only are mutually exclusive.[/red]")
        raise typer.Exit(2)
    if project:
        _generate_project_config()
    elif auth_token_only:
        # Same convention as `_init_async`: the installer runs this as the unit
        # user, so `~` is the daemon's own HOME and therefore its data dir.
        data_dir = Path("~/.nerdit").expanduser()
        config_path = data_dir / "config.toml"
        outcome = _write_installer_auth_config(data_dir)
        if outcome == "written":
            console.print(f"[green]Auth token written:[/green] {config_path}")
            console.print("[dim]Read it back with 'nerdit token'.[/dim]")
        elif outcome == "present":
            console.print(f"[dim]{config_path} already sets an auth token; leaving it.[/dim]")
        else:
            console.print(f"[red]Could not write an auth token to {config_path}.[/red]")
            raise typer.Exit(1)
    else:
        asyncio.run(_init_async())


def _generate_project_config() -> None:
    """Write a nerdit.toml template to cwd."""
    target = Path.cwd() / "nerdit.toml"
    if target.exists():
        console.print(f"[yellow]nerdit.toml already exists:[/yellow] {target}")
        return
    target.write_text(NERDIT_TOML_TEMPLATE)
    console.print(f"[green]nerdit.toml created:[/green] {target}")


def _report_daemon_start_failure(port: int, lifecycle: DaemonLifecycle | None = None) -> bool:
    """Report a late startup, wedged child or foreign port occupant.

    Return True only if the daemon recovered after the readiness deadline.
    Point a live but unhealthy child to its boot log; otherwise show the foreign
    occupant and remediation when identified.
    """
    from rich.markup import escape

    port_check = checks.check_port(port)

    # Came up late: the bind-probe now sees a nerdit-shaped /health responder.
    if port_check.status == "ok" and "already running" in port_check.detail:
        console.print("[green]Daemon came up (after the readiness wait).[/green]")
        return True

    child_alive = bool(lifecycle.is_running()) if lifecycle is not None else False

    # A foreign occupant (our child is gone) — blame it specifically.
    if port_check.status == "fail" and not child_alive:
        console.print(f"[red]Daemon could not start — {escape(port_check.detail)}.[/red]")
        if port_check.remediation:
            console.print(f"  [dim]{escape(port_check.remediation)}[/dim]")
        return False

    # Wedged own child, or the port is free but nothing answered — the boot log
    # holds the real reason.
    console.print("[red]Daemon did not respond in time.[/red]")
    boot_log = Path("~/.nerdit/nerditd.boot.log").expanduser()
    console.print(f"  [dim]Check the boot log for details: {escape(str(boot_log))}[/dim]")
    return False


async def _init_async() -> None:
    """Run the full init flow: config, Docker checks, daemon start, GPU display."""
    from nerdit.cli.client import NerditClient
    from nerdit.cli.display import display_gpu_table
    from nerdit.daemon.lifecycle import DaemonLifecycle

    data_dir = Path("~/.nerdit").expanduser()
    config_path = data_dir / "config.toml"

    # Create config directory (owner-only: it holds the plaintext auth_token).
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "cache").mkdir(exist_ok=True)
    (data_dir / "uploads").mkdir(exist_ok=True)
    try:
        data_dir.chmod(0o700)
    except OSError:
        pass

    # (P30) An installer-made install is already configured and already running
    # under its service unit. Writing the source-checkout default config there
    # would set `host = "0.0.0.0"` and a fresh auth_token — both restart-keyed,
    # so the node would sit in `config_restart_pending` and, once restarted,
    # bind every interface: the exact opposite of what `docs/guide/install.md`
    # promises. `nerdit init` is the source-checkout bootstrap; installs skip it.
    from nerdit.utils.install_layout import detect_install_layout

    installed = detect_install_layout() is not None

    # Write default config if it doesn't exist
    token = None
    if installed and not config_path.exists():
        console.print(
            "[dim]Installed via the nerdit installer — leaving the daemon on its "
            "defaults (loopback, no config.toml). See docs/guide/install.md.[/dim]"
        )
    elif not config_path.exists():
        config_text, token = _generate_default_config()
        config_path.write_text(config_text)
        try:
            config_path.chmod(0o600)
        except OSError:
            pass
        console.print(f"[green]Configuration created:[/green] {config_path}")
    else:
        console.print(f"[dim]Existing configuration:[/dim] {config_path}")

    # Check Docker availability (non-blocking); offer to auto-start it if down
    docker_ok = _check_docker()
    if not docker_ok:
        docker_ok = _offer_start_docker()
    if not docker_ok:
        console.print(
            "[yellow]Docker is not running. Nothing will deploy until Docker is available.[/yellow]"
        )

    # Check NVIDIA Container Toolkit
    nvidia_ok = False
    nvidia_lib_ok = False
    if docker_ok:
        nvidia_ok = _check_nvidia_toolkit()
        if not nvidia_ok:
            console.print(
                "[yellow]NVIDIA Container Toolkit not found. "
                "GPU passthrough will not work.[/yellow]"
            )
            console.print(
                "[dim]Install: sudo apt install nvidia-container-toolkit && "
                "sudo systemctl restart docker[/dim]"
            )
        else:
            nvidia_lib_ok = _check_nvidia_driver_lib()
            if not nvidia_lib_ok:
                console.print(
                    "[yellow]NVIDIA Container Toolkit registered but "
                    "libnvidia-ml.so.1 not found on host.[/yellow]"
                )
                console.print(
                    "[dim]This often happens with snap-installed Docker. "
                    "GPU passthrough will likely fail.[/dim]"
                )

    # Start daemon
    lifecycle = DaemonLifecycle(host=DEFAULT_HOST, port=DEFAULT_PORT)
    if not lifecycle.is_running():
        console.print("Starting daemon...")
        try:
            started = lifecycle.start()
        except RuntimeError as exc:
            # An incomplete frozen bundle makes _spawn_argv() raise. The
            # surrounding code is built to explain a failed start; a traceback
            # here would bypass all of it.
            console.print(f"[red]Could not start the daemon:[/red] {exc}")
            started = False
        if started and lifecycle.wait_for_ready(timeout=15.0):
            console.print("[green]Daemon started[/green]")
        elif not _report_daemon_start_failure(DEFAULT_PORT, lifecycle):
            raise typer.Exit(1)
    else:
        console.print("[dim]Daemon already running[/dim]")

    # Display auth token if newly generated
    if token:
        console.print()
        console.print("[bold yellow]Authentication token:[/bold yellow]")
        console.print(f"  {token}")
        console.print()
        console.print("[dim]Use this token to connect a remote client:[/dim]")
        console.print(f"  [cyan]nerdit connect <ip> --token {token}[/cyan]")

    # Display GPU info
    host, port, cfg_token = get_client_config()
    client = NerditClient(host=host, port=port, token=cfg_token)
    gpu_count = 0
    try:
        gpus = await client.list_gpus()
        gpu_count = len(gpus)
        if gpus:
            console.print(f"\n[bold]{gpu_count} GPU(s) detected:[/bold]")
            display_gpu_table(gpus)
    except Exception:
        pass

    # Health check recap
    console.print()
    console.print("[bold]Health check:[/bold]")
    console.print(f"  [green]✓[/green] Daemon running on {DEFAULT_HOST}:{DEFAULT_PORT}")
    console.print(f"  [green]✓[/green] Dashboard open on http://{DEFAULT_HOST}:{DEFAULT_PORT}/")

    if docker_ok:
        console.print("  [green]✓[/green] Docker available")
    else:
        console.print("  [yellow]✗[/yellow] Docker not available")

    if nvidia_ok and nvidia_lib_ok:
        console.print("  [green]✓[/green] NVIDIA Container Toolkit installed")
    elif nvidia_ok and not nvidia_lib_ok:
        console.print(
            "  [yellow]![/yellow] NVIDIA Container Toolkit registered but "
            "libnvidia-ml.so.1 not accessible — GPU passthrough will fail"
        )
    elif docker_ok:
        console.print(
            "  [yellow]✗[/yellow] NVIDIA Container Toolkit not installed — "
            "GPU passthrough will fail"
        )

    if gpu_count > 0:
        console.print(f"  [green]✓[/green] {gpu_count} GPU(s) detected")
    else:
        console.print(
            "  [dim]○[/dim] No GPU detected — apps still deploy; "
            "GPU-backed models will stay pending"
        )

    # Docker is the only hard requirement: a CPU-only host deploys apps fine.
    if docker_ok:
        console.print("  [bold green]→ Ready to deploy![/bold green]")
    else:
        console.print("  [yellow]→ Docker is required to deploy (see above)[/yellow]")

    # Offer optional linking for source installs without starting a cloud mutation.
    # An unlinked daemon remains fully usable; init must not imply an incomplete install.
    console.print()
    console.print("[bold]Next step:[/bold] link this node to your Nerdit account")
    console.print("  [cyan]nerdit link --device[/cyan]   approve it in your browser")
    console.print(
        "  [dim]Optional: deploys, models and databases all work unlinked — "
        "linking adds remote access and hosted URLs.[/dim]"
    )
