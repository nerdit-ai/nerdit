"""Request a daemon restart and optionally wait for a fresh boot.

A lost HTTP 202 may mean restart began, but only treat transport failure as
success when the pre-request baseline read proved the daemon was reachable.
Escape server-derived Rich text.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

import httpx
import typer

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain

# Poll cadence for ``--wait``. Small enough that a fast restart feels instant,
# large enough not to hammer a daemon that is mid-boot.
_POLL_INTERVAL_S = 0.5

daemon_app = typer.Typer(
    name="daemon",
    help=(
        "Operate the local daemon process. Use `nerdit exit` to stop it; "
        "`nerdit daemon restart` applies restart-required config."
    ),
    no_args_is_help=True,
)


@daemon_app.command("restart")
def daemon_restart(
    drain_timeout_s: int = typer.Option(
        60,
        "--drain-timeout-s",
        help="Seconds to drain in-flight builds and runs (server-clamped [0, 300]).",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    wait: bool = typer.Option(False, "--wait", help="Block until the daemon answers again."),
    wait_timeout: int = typer.Option(
        60,
        "--wait-timeout",
        help="Seconds to wait for the daemon to come back (with --wait).",
    ),
) -> None:
    """Restart the daemon to apply restart-required config (admin-only, audited).

    Exit code 0 when the restart was accepted (or, with `--wait`, when a
    freshly booted daemon answered again), 1 when it was denied, already in
    progress, or never came back. Stopping the daemon instead is `nerdit exit`.
    """
    code = asyncio.run(
        _restart_async(
            drain_timeout_s=drain_timeout_s,
            yes=yes,
            wait=wait,
            wait_timeout=wait_timeout,
        )
    )
    if code:
        raise typer.Exit(code)


async def _restart_async(*, drain_timeout_s: int, yes: bool, wait: bool, wait_timeout: int) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()

    # (a) Baseline — best effort, but the outcome is load-bearing in (d): it is
    # the only pre-POST evidence that a daemon was ever there.
    baseline_ok = False
    try:
        caps0 = await client.get_capabilities()
    except Exception as exc:  # noqa: BLE001 — non-fatal, the POST still runs
        console.print(f"[dim]Could not read the daemon's current state: {_plain(exc)}[/dim]")
    else:
        baseline_ok = True
        console.print(
            f"daemon [bold]{_plain(caps0.get('version'))}[/bold], "
            f"up {_plain(caps0.get('uptime_s'))}s"
        )

    # (b) Confirm (the `nerdit gc` precedent).
    if not yes and not typer.confirm("Restart the daemon now?", default=False):
        console.print("[dim]Aborted.[/dim]")
        return 0

    # (c) POST. `t_post` is taken immediately before the call — it is the
    # reference instant the (e) freshness predicate compares boot times against.
    t_post = time.monotonic()
    try:
        body = await client.restart_daemon(
            drain_timeout_s=drain_timeout_s, idempotency_key=uuid4().hex
        )
    except httpx.TransportError as exc:
        # (d) No answer. That is equally consistent with "the daemon died on
        # purpose" and "there was never a daemon there" — only the baseline GET
        # distinguishes them, so the downgrade is gated on it.
        if not baseline_ok:
            render_client_error(exc)
            # One line, no detection: a pending `\[daemon].port` change makes the
            # CLI dial the NEW port while the live daemon still holds the old one.
            console.print(
                "[dim]If a \\[daemon].port change is pending restart, the daemon is "
                "still on the OLD port: point the CLI at it (nerdit connect "
                "127.0.0.1 --port <old>) or restart it manually, then retry.[/dim]"
            )
            return 1
        console.print(
            "[yellow]The daemon closed the connection before answering — "
            "it is probably restarting.[/yellow]"
        )
        if not wait:
            return 0
    except Exception as exc:  # noqa: BLE001 — structured envelope, rendered
        render_client_error(exc)
        return 1
    else:
        console.print("[green]Restart accepted.[/green]")
        console.print(f"in-flight builds: {_plain(body.get('in_flight_builds'))}")
        console.print(f"in-flight runs: {_plain(body.get('in_flight_runs'))}")
        console.print(f"drain timeout: {_plain(body.get('drain_timeout_s'))}s")
        if not wait:
            return 0

    # (e) --wait
    return await _wait_for_fresh_daemon(client, t_post=t_post, wait_timeout=wait_timeout)


async def _wait_for_fresh_daemon(client, *, t_post: float, wait_timeout: int) -> int:  # noqa: ANN001
    """Poll capabilities until the responding daemon booted after the restart POST.

    Compare `time.monotonic() - uptime_s` with `t_post`; uptime alone can accept
    an old process still draining. Integer uptime leaves a subsecond ambiguity
    when the old process booted less than one second before the request.
    Only durations cross clocks; this assumes no wall-clock step on the host.

    Retry transport and HTTP status errors, retaining the last status for timeout
    diagnostics. Other exceptions propagate.
    """
    console.print(f"[dim]waiting up to {wait_timeout}s for the daemon to come back…[/dim]")
    deadline = time.monotonic() + wait_timeout
    last_status: int | None = None
    while time.monotonic() < deadline:
        caps: dict | None = None
        try:
            caps = await client.get_capabilities()
        except httpx.HTTPStatusError as exc:
            # The daemon *did* answer, just not with a usable body (a booting
            # process behind a 503, or a token that stopped being accepted).
            # Keep polling — but remember it, so a timeout is not reported as
            # "never answered".
            last_status = exc.response.status_code
            caps = None
        except httpx.TransportError:
            # A dead/refused socket mid-restart is the expected signal. Anything
            # else (a bug in the client, a bad response shape) is NOT swallowed.
            caps = None
        if isinstance(caps, dict):
            uptime = caps.get("uptime_s")
            if isinstance(uptime, (int, float)) and not isinstance(uptime, bool):
                boot = time.monotonic() - float(uptime)
                if boot >= t_post:
                    console.print(
                        f"[green]Daemon back:[/green] {_plain(caps.get('version'))}, "
                        f"up {_plain(uptime)}s"
                    )
                    return 0
        await asyncio.sleep(_POLL_INTERVAL_S)

    if last_status is not None:
        console.print(
            f"[red]The daemon answered with an error status ({last_status}) "
            f"for {wait_timeout}s — no freshly booted daemon seen.[/red]"
        )
    else:
        console.print(f"[red]The daemon has not answered within {wait_timeout}s.[/red]")
    console.print("[dim]Check `nerdit doctor`, then the daemon log.[/dim]")
    return 1
