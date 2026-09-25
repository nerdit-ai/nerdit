"""Manage service lifecycle, redeploys, waits, resource samples and one-off commands by ID or
name.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import typer
from rich.markup import escape

from nerdit.cli.commands.deploy import parse_env_pairs
from nerdit.cli.display import (
    _plain,
    call_or_exit,
    console,
    display_deploy_result,
    display_service_table,
    display_wait_outcome,
    fmt_bytes,
    render_client_error,
    wait_exit_code,
)

services_app = typer.Typer(
    name="services",
    help="Manage long-running services.",
    no_args_is_help=True,
)


@services_app.command("list")
def services_list(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by exact status"),
) -> None:
    """List services (bounded, newest first)."""
    asyncio.run(_list_async(status))


async def _list_async(status: str | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    page = await call_or_exit(client.list_services(status=status))

    items = page.get("items", [])
    if not items:
        console.print("[dim]No services.[/dim]")
        return
    display_service_table(items)


@services_app.command("stop")
def services_stop(
    name: str = typer.Argument(..., help="Service id or name"),
) -> None:
    """Stop a service (desired_state → stopped)."""
    asyncio.run(_stop_async(name))


async def _stop_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    await call_or_exit(client.stop_service(name, idempotency_key=uuid4().hex))
    console.print(f"[green]Stopping service {name}.[/green]")


@services_app.command("restart")
def services_restart(
    name: str = typer.Argument(..., help="Service id or name"),
) -> None:
    """Restart a service (desired_state → running, backoff cleared)."""
    asyncio.run(_restart_async(name))


async def _restart_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    await call_or_exit(client.restart_service(name, idempotency_key=uuid4().hex))
    console.print(f"[green]Restarting service {name}.[/green]")


@services_app.command("redeploy")
def services_redeploy(
    # Name only: unlike the rest of the quartet this route resolves by
    # service name (the deploy plane is name-keyed), so an id is not accepted.
    name: str = typer.Argument(..., help="Service name"),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Block until the redeploy converges (healthy) or fails. "
        "Exit codes: 0 converged, 1 failed or superseded, 3 timeout.",
    ),
    wait_timeout: int = typer.Option(
        60, "--timeout", help="Seconds to wait with --wait (server-clamped to [1, 300])"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate + print the plan diff without deploying (no write)",
    ),
) -> None:
    """Rebuild a service from its recorded Git repository and ref.

    No local checkout is needed. ZIP-deployed services are refused with deploy.no_source.
    """
    asyncio.run(_redeploy_async(name, wait=wait, wait_timeout=wait_timeout, dry_run=dry_run))


async def _redeploy_async(name: str, *, wait: bool, wait_timeout: int, dry_run: bool) -> None:
    from nerdit.cli.client import get_configured_client
    from nerdit.cli.commands.deploy import _maybe_wait, _render_dry_run

    client = get_configured_client()
    service = await call_or_exit(client.redeploy_app(name, dry_run=dry_run))
    if dry_run:
        _render_dry_run(service)
        return
    display_deploy_result(service, heading="Redeploy accepted")
    await _maybe_wait(client, name, service, wait, wait_timeout)


@services_app.command("rm")
def services_rm(
    name: str = typer.Argument(..., help="Service id or name"),
    purge: str = typer.Option(
        "secrets", "--purge", help="CSV of purge targets: secrets,data,images"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Bypass the model-reference and active-run guards (may require admin)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the data-purge confirmation"),
) -> None:
    """Remove a service (tear down the container + delete the row).

    `--purge` selects what durable state is destroyed alongside the row
    (`secrets` by default; `data` and `images` are opt-in and irreversible).
    Deleting a model in use by other apps needs `--force`.

    (P20) A delete is also refused while a one-off run or a `[deploy].release`
    migration is executing for the service — tearing the container out from
    under a live writer is a torn teardown. `--force` kills the run and
    proceeds; the data purge is skipped when a run container could not be
    confirmed dead, so `--purge data` may report less than it was asked for.
    """
    targets = {t.strip().lower() for t in purge.split(",") if t.strip()}
    if "data" in targets and not yes:
        typer.confirm(
            f"Purge '{name}' data dir permanently (irreversible)?",
            abort=True,
        )
    asyncio.run(_rm_async(name, purge=purge, force=force))


async def _rm_async(name: str, *, purge: str, force: bool) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    await call_or_exit(
        client.remove_service(name, purge=purge, force=force, idempotency_key=uuid4().hex)
    )
    console.print(f"[green]Removed service {name}.[/green]")


@services_app.command("wait")
def services_wait(
    name: str = typer.Argument(..., help="Service id or name"),
    timeout: int = typer.Option(
        60, "--timeout", help="Seconds to block (server-clamped to [1, 300])"
    ),
    version: Optional[int] = typer.Option(
        None, "--version", help="Deploy generation to wait on; a newer one reports 'superseded'"
    ),
) -> None:
    """Wait for a service's deploy outcome.

    Omitting --version selects the current generation; a concurrent redeploy then
    reports superseded. Set --version to pin a generation. HTTP 200 can carry any
    outcome: branch on the exit code.

    Exit codes: 0 converged, 1 failed/superseded or request error, 3 timeout.
    """
    asyncio.run(_wait_async(name, timeout=timeout, version=version))


async def _wait_async(name: str, *, timeout: int, version: int | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        with console.status(f"Waiting for '{escape(name)}' to converge...", spinner="dots"):
            # No client-side clamp on ``timeout``: the daemon clamps it to
            # [1, 300] and answers with the value it actually used. Mirroring
            # the bound here would silently truncate the request the day the
            # server cap moves — the deliberate-absence precedent recorded for
            # ``MAX_RUN_TIMEOUT_S`` at ``mcp/tools/_shared.py``.
            result = await client.wait_for_service(name, version=version, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    display_wait_outcome(result)
    code = wait_exit_code(result)
    if code:
        raise typer.Exit(code)


def _fmt_pct(value: object) -> str:
    """Percentage with one decimal, or `-` when unknown (same rule as bytes)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{float(value):.1f}%"


@services_app.command("stats")
def services_stats(
    name: str = typer.Argument(..., help="Service id or name"),
) -> None:
    """Show live CPU, memory and network usage, cached for two seconds.

    Unavailable samples are reported explicitly, never as zero usage. GPU values
    come from the existing monitor snapshot.
    """
    asyncio.run(_stats_async(name))


async def _stats_async(name: str) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    result = await call_or_exit(client.get_service_stats(name))

    # Every value below is server-derived and therefore goes through ``_plain``
    # before it reaches a Rich sink — a service name or container id is not ours
    # to trust as markup (the recurring MarkupError lesson).
    service_name = _plain(result.get("service_name") or name)
    if not result.get("available"):
        console.print(f"[yellow]No stats for {service_name}.[/yellow]")
        console.print(
            "[dim]The service has no running container, or the daemon could not reach "
            "Docker. Check `nerdit services list` and `nerdit doctor`.[/dim]"
        )
        return

    stats = result.get("stats") or {}
    console.print(f"[bold]{service_name}[/bold]")
    console.print(f"  CPU:     {_plain(_fmt_pct(stats.get('cpu_pct')))}")
    mem = fmt_bytes(stats.get("mem_used_bytes"))
    limit = fmt_bytes(stats.get("mem_limit_bytes"))
    mem_pct = _fmt_pct(stats.get("mem_pct"))
    console.print(f"  Memory:  {_plain(mem)} / {_plain(limit)} ({_plain(mem_pct)})")
    rx = fmt_bytes(stats.get("net_rx_bytes"))
    tx = fmt_bytes(stats.get("net_tx_bytes"))
    console.print(f"  Network: {_plain(rx)} in / {_plain(tx)} out")
    pids = stats.get("pids")
    console.print(f"  PIDs:    {_plain(pids) if pids is not None else '-'}")

    for gpu in result.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        util = gpu.get("utilization_percent")
        used = gpu.get("memory_used_mb")
        util_txt = f"{_plain(util)}%" if util is not None else "-"
        used_txt = f"{_plain(used)} MB" if used is not None else "-"
        console.print(f"  GPU {_plain(gpu.get('gpu_id'))}: {util_txt} util, {used_txt} used")

    if result.get("cached"):
        console.print("[dim]Served from the daemon's 2s sample cache.[/dim]")


@services_app.command("run")
def services_run(
    name: str = typer.Argument(..., help="Service id or name"),
    command: list[str] = typer.Argument(
        ..., help="Command to execute, after a ``--`` separator (e.g. -- alembic upgrade head)"
    ),
    timeout_s: int = typer.Option(
        300, "--timeout-s", help="Seconds the command may run before it is killed"
    ),
    env: list[str] = typer.Option(
        [], "--env", "-e", help="Environment override KEY=VAL for this run only (repeatable)"
    ),
    tail: int = typer.Option(
        200, "--tail", help="Log lines to print (server-bounded: must be within [1, 200])"
    ),
) -> None:
    """Run a bounded command in the service's image without changing the service.

    Uses its env, secrets and volumes, with no port, route or workload row.
    Arguments after -- reach the image entrypoint without a shell; use sh -c
    explicitly for pipes or globbing.

    Never put credentials in argv: Docker inspect and the durable last_run record
    expose it. Use --env instead; values are masked in audit and omitted in responses.
    Inspect saved outcomes with nerdit diagnose.

    Exit codes: 0 successful command, 3 timeout, 1 nonzero/unknown exit or request error.
    """
    try:
        overrides = parse_env_pairs(env) if env else None
    except ValueError as exc:
        console.print(f"[red]{_plain(exc)}[/red]")
        raise typer.Exit(1) from exc
    asyncio.run(_run_async(name, list(command), timeout_s=timeout_s, env=overrides, log_tail=tail))


def _print_interrupted_tail(exc: Exception) -> None:
    """Print partial output from run.interrupted errors.

    Effects may already have applied; last_run was not stamped, making this error
    envelope the only surviving output record.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 — non-JSON error body
        return
    if not isinstance(body, dict) or body.get("code") != "run.interrupted":
        return
    lines = body.get("log_tail")
    if not isinstance(lines, list) or not lines:
        return
    console.print("[dim]--- partial output before the container was lost ---[/dim]")
    for line in lines:
        console.print(f"  {_plain(line)}")


async def _run_async(
    name: str,
    command: list[str],
    *,
    timeout_s: int,
    env: dict[str, str] | None,
    log_tail: int,
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.run_service_command(
            name,
            command=command,
            env=env,
            timeout_s=timeout_s,
            log_tail=log_tail,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # Interrupted runs may have partial effects and no last_run record.
        # Print the error envelope's tail: it is the only surviving output.
        _print_interrupted_tail(exc)
        render_client_error(exc)
        raise typer.Exit(1) from exc

    lines = result.get("log_tail") or []
    if lines:
        console.print("[dim]--- output ---[/dim]")
        for line in lines:
            console.print(f"  {_plain(line)}")

    exit_code = result.get("exit_code")
    timed_out = bool(result.get("timed_out"))
    duration = result.get("duration_s")
    suffix = f" in {duration}s" if duration is not None else ""
    if timed_out:
        console.print(
            f"[yellow]Timed out[/yellow] after {timeout_s}s — the run container was killed."
        )
        console.print(
            "[dim]Raise --timeout-s (the daemon caps it at the "
            "services.run_timeout_max_s config key) or shorten the command.[/dim]"
        )
        raise typer.Exit(3)
    if exit_code == 0:
        console.print(f"[green]Exit 0[/green]{suffix}")
        return
    if exit_code is None:
        # Documented on ServiceRunResponse: null when neither the bounded wait
        # nor the post-mortem inspect could produce a code. Printing the bare
        # `_plain(None)` gave "Exit  in 0.4s" — a malformed line that reads as a
        # CLI bug rather than as the real, reportable outcome it is.
        console.print(f"[red]Exit unknown[/red]{suffix}")
        console.print(
            "[dim]The runtime lost the run container before its exit code could be read — "
            "the command may have partially applied. Check the output above.[/dim]"
        )
    else:
        console.print(f"[red]Exit {_plain(exit_code)}[/red]{suffix}")
    if result.get("oom_killed"):
        console.print(
            "[dim]The kernel OOM-killed the run container — raise the app's "
            "deploy.memory_limit or lower the command's footprint.[/dim]"
        )
    raise typer.Exit(1)
