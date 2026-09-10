"""Stop the daemon, using shared PID-identity and data-dir lock probes.

Stop through its service unit first to prevent automatic respawn; signals
are the fallback. The lock also detects a daemon whose PID file is missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from nerdit.cli.commands.uninstall import (
    _DAEMON_DRAIN_S,
    _stop_unit,
    _terminate,
    probe_daemon,
)
from nerdit.cli.display import _plain, console
from nerdit.config.settings import NerditSettings, load_settings
from nerdit.utils.install_layout import detect_service_unit, service_unit_loaded


def exit_daemon(
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt (for scripting)"
    ),
) -> None:
    """Stop the nerditd daemon.

    To restart it instead (e.g. to apply restart-required config), use
    `nerdit daemon restart`.
    """
    asyncio.run(_exit_async(yes))


async def _services_up_count() -> int | None:
    """Best-effort count of workloads up on the local daemon (None if unknown)."""
    try:
        from nerdit.cli.client import NerditClient
        from nerdit.config.settings import load_settings as _load

        daemon = _load().daemon
        client = NerditClient(host=daemon.host, port=daemon.port, token=daemon.auth_token)
        stats = await client.cluster_stats()
        return int(stats.get("services_up", 0))
    except Exception:
        return None


def _settings() -> NerditSettings:
    """Tolerant settings load — a mangled config.toml must not block a stop."""
    try:
        return load_settings()
    except Exception:  # noqa: BLE001 — any parse/validation error
        console.print("[yellow]Could not read config.toml; assuming defaults.[/yellow]")
        return NerditSettings()


async def _exit_async(yes: bool) -> None:
    settings = _settings()
    pid_file = Path(settings.daemon.pid_file).expanduser()
    data_dir = Path(settings.data_dir).expanduser()

    liveness = probe_daemon(data_dir=data_dir, pid_file=pid_file)
    if not liveness.alive:
        console.print("[dim]Daemon is not running[/dim]")
        return

    unit = detect_service_unit()
    try:
        if unit is not None and not service_unit_loaded(unit):
            unit = None
    except RuntimeError as exc:
        console.print(f"[yellow]{_plain(exc)}; trying the service stop.[/yellow]")
    if unit is not None:
        console.print(
            f"[dim]This daemon is managed by a {_plain(unit.kind)} service unit "
            f"({_plain(unit.unit_path)}).[/dim]"
        )

    if not yes:
        running = await _services_up_count()
        if running:
            console.print(
                f"[yellow]{running} workload(s) are still running.[/yellow] "
                "Stopping the daemon leaves their containers running but unmonitored."
            )
        if not typer.confirm("Stop the daemon?"):
            console.print("[dim]Aborted[/dim]")
            raise typer.Exit(0)

    # 1. The service manager first when there is one: signalling a supervised
    #    daemon just gets it restarted.
    if unit is not None:
        _stop_unit(unit, data_dir=data_dir, pid_file=pid_file)
        liveness = probe_daemon(data_dir=data_dir, pid_file=pid_file)
        if not liveness.alive:
            pid_file.unlink(missing_ok=True)
            console.print("[green]Daemon stopped (via the service unit)[/green]")
            return
        console.print(
            "[yellow]The service unit did not stop it — falling back to "
            "signalling the process.[/yellow]"
        )

    # 2. Signal path: TERM → drain → KILL, the same escalation the uninstall uses.
    if liveness.pid is None:
        console.print(
            f"[red]{_plain(liveness.describe())}, but there is no pid to signal. "
            "Stop it by hand and retry.[/red]"
        )
        raise typer.Exit(1)

    stopped = _terminate(liveness.pid, "nerditd", grace_s=_DAEMON_DRAIN_S, expect_cmd="nerdit")
    if stopped:
        pid_file.unlink(missing_ok=True)

    liveness = probe_daemon(data_dir=data_dir, pid_file=pid_file)
    if liveness.alive:
        console.print(f"[red]Error stopping daemon — {_plain(liveness.describe())}[/red]")
        raise typer.Exit(1)
    console.print("[green]Daemon stopped[/green]")
