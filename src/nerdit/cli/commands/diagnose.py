"""Render service failures, remediation, health, bindings and log tails.

Also show `last_run`: one-off output is not in job_logs, so this is its durable
inspection surface after the original terminal closes.
"""

from __future__ import annotations

import asyncio
import json

import typer

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain


def diagnose(
    name: str = typer.Argument(..., help="Service name or id to diagnose"),
    log_tail: int = typer.Option(
        50, "--log-tail", help="Log lines to show (clamped server-side to [1, 200])"
    ),
) -> None:
    """Explain why a service failed and what to do about it (one bounded call)."""
    asyncio.run(_diagnose_async(name, log_tail))


def _argv(value: object) -> object:
    """Render command lists as JSON argv, preserving token boundaries without shell syntax.

    Return non-lists unchanged for legacy data. Callers must Rich-escape the result.
    """
    return json.dumps(value) if isinstance(value, list) else value


def _print_health_observations(data: dict) -> None:
    """Render health advice from list-shaped observations, escaping each message."""
    health = data.get("health") or {}
    observations = health.get("observations")
    if isinstance(observations, list) and observations:
        console.print("  [yellow]health observations:[/yellow]")
        for obs in observations:
            console.print(f"    - {_plain(obs)}")


async def _diagnose_async(name: str, log_tail: int) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.diagnose_service(name, log_tail=log_tail)
    except Exception as exc:  # noqa: BLE001 — surfaced as a structured client error
        render_client_error(exc)
        raise typer.Exit(1) from exc

    svc = data.get("service_name", name)
    status = data.get("status", "?")
    console.print(
        f"[bold]{_plain(svc)}[/bold]  status: {_plain(status)}  kind: {_plain(data.get('kind'))}"
    )

    ld = data.get("last_deploy") or {}
    if ld:
        console.print(
            f"  deploy: v{_plain(ld.get('version'))} "
            f"{_plain(ld.get('action'))} → phase {_plain(ld.get('phase'))}"
        )

    err = data.get("error") or {}
    if err.get("class") or err.get("message"):
        console.print(f"  error: {_plain(err.get('class'))} — {_plain(err.get('message'))}")

    forensics = data.get("forensics") or {}
    if forensics.get("last_crash_at"):
        console.print(
            f"  last crash: exit {_plain(forensics.get('last_exit_code'))} "
            f"oom={_plain(forensics.get('oom_killed'))} "
            f"gpu_oom={_plain(forensics.get('gpu_oom'))} "
            f"priv_denied={_plain(forensics.get('priv_denied'))} "
            f"at {_plain(forensics.get('last_crash_at'))}"
        )

    restarts = data.get("restarts") or {}
    console.print(
        f"  restarts: {_plain(restarts.get('count'))}/{_plain(restarts.get('max_restarts'))}"
        f"  next retry in: {_plain(restarts.get('next_retry_in_s'))}s"
    )

    bindings = data.get("bindings") or {}
    if bindings.get("waiting"):
        console.print("  [yellow]bindings waiting:[/yellow]")
        for msg in bindings.get("messages", []):
            console.print(f"    - {_plain(msg)}")

    _print_health_observations(data)

    rem = data.get("remediation") or {}
    console.print(f"\n[bold cyan]remediation:[/bold cyan] {_plain(rem.get('code'))}")
    console.print(f"  {_plain(rem.get('detail'))}")

    logs = data.get("logs") or []
    if logs:
        console.print("\n[dim]--- log tail ---[/dim]")
        for entry in logs:
            console.print(f"  {_plain(entry.get('line'))}")

    # Render saved one-off output last, only when present. Escape every field and
    # raw log line; bracketed container output is not trusted Rich markup.
    run = data.get("last_run") or {}
    if run:
        console.print("\n[dim]--- last run ---[/dim]")
        console.print(f"  command: {_plain(_argv(run.get('command')))}")
        console.print(
            f"  exit {_plain(run.get('exit_code'))}"
            f"  timed_out={_plain(run.get('timed_out'))}"
            f"  oom={_plain(run.get('oom_killed'))}"
            f"  in {_plain(run.get('duration_s'))}s"
        )
        console.print(f"  finished: {_plain(run.get('finished_at'))}")
        # `last_run` is an untyped passthrough of the stored blob (the response
        # models it as dict[str, object]), so check the shape before iterating —
        # a stray string would otherwise be printed one character per line.
        tail = run.get("log_tail")
        if isinstance(tail, list) and tail:
            console.print("  [dim]output:[/dim]")
            for line in tail:
                console.print(f"    {_plain(line)}")
