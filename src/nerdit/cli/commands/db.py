"""Provision, inspect, dump and restore managed databases.

Passwords are minted and stored server-side, never printed. Apps receive URLs
through DB bindings. Dumps are logical captures; `nerdit backup --volume` makes
physical backups. Stop, restart and removal use `nerdit services`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import typer
from rich.markup import escape
from rich.table import Table

from nerdit.cli.commands.deploy import _maybe_wait
from nerdit.cli.display import (
    _plain,
    console,
    display_database_table,
    fmt_bytes,
    render_client_error,
)

if TYPE_CHECKING:  # pragma: no cover — import-cycle-free typing only
    from nerdit.cli.client import NerditClient

db_app = typer.Typer(
    name="db",
    help="Provision managed databases, and dump or restore their contents.",
    no_args_is_help=True,
)


@db_app.command("create")
def db_create(
    backend: Optional[str] = typer.Argument(
        None, help="Data backend to provision: postgres | redis (default: daemon config)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Database name (DNS label; default: the backend's prefix)"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Block until the database is ready"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="--wait timeout in seconds"),
) -> None:
    """Provision a managed database (kind=database desired-state workload)."""
    asyncio.run(_create_async(backend, name, wait, timeout))


async def _create_async(backend: str | None, name: str | None, wait: bool, timeout: int) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        database = await client.create_database(
            backend=backend,
            name=name,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    db_name = database.get("name", "")
    console.print(f"[green]Database provisioning:[/green] {db_name}")
    console.print(f"  Backend: {database.get('backend') or backend or '-'}")
    console.print(f"  Status:  {database.get('status')}")
    console.print(
        "[dim]Watch it with `nerdit db list` — the endpoint appears once the "
        "database is ready.[/dim]"
    )
    # The one-liner an app (or agent) needs next: the [db.default] binding. The
    # database= line is REQUIRED — there is no implicit default (owner call).
    # markup=False so Rich never eats the TOML section header's own brackets
    # (a bare "[db.default]" is otherwise parsed as a — empty — style tag).
    console.print("\n[dim]Bind an app to it by adding to its nerdit.toml:[/dim]")
    console.print("[db.default]", markup=False, style="cyan")
    console.print('provider = "managed"', markup=False, style="cyan")
    console.print(f'database = "{db_name}"', markup=False, style="cyan")

    # --wait resolves on config['db_ready']; db rows carry no build_version so the
    # requested version is None (the server wait resolves on readiness alone).
    await _maybe_wait(client, db_name, database, wait, timeout)


@db_app.command("list")
def db_list() -> None:
    """List managed databases (bounded, newest first)."""
    asyncio.run(_list_async())


async def _list_async() -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        page = await client.list_databases()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    items = page.get("items", [])
    if not items:
        console.print("[dim]No databases. Create one with `nerdit db create <backend>`.[/dim]")
        return
    display_database_table(items)


# --------------------------------------------------------------------------- #
# Dumps (P37, §1.6)
# --------------------------------------------------------------------------- #

#: Client-side mirror of the route's ``timeout_s`` defaults. The server's cap
#: (``[services].dump_timeout_max_s``) is the one that binds — an over-cap value
#: is refused with ``422 dump.timeout_too_large`` naming both the cap and the
#: key, never silently clamped — so these are merely the budgets a plain
#: invocation asks for.
DUMP_DEFAULT_TIMEOUT_S = 900
RESTORE_DEFAULT_TIMEOUT_S = 600

#: Fallback for the retention line when the live ``[retention]`` read fails
#: (a daemon that is momentarily unreachable must not cost the operator the
#: custody information for a dump that already succeeded). Mirrors
#: ``RetentionConfig.dump_keep_last``'s packaged default.
_DEFAULT_DUMP_KEEP_LAST = 5


async def _dump_keep_last(client: NerditClient) -> tuple[int, bool]:
    """Best-effort read of ``[retention].dump_keep_last`` → ``(keep, live)``. One
    bounded ``GET /config/daemon/retention``; on any failure the packaged default
    with ``live=False``, so a successful dump is never reported as a failure.
    """
    try:
        view = await client.get_config("retention")
        values = view.get("values") if isinstance(view, dict) else None
        keep = values.get("dump_keep_last") if isinstance(values, dict) else None
        if isinstance(keep, int) and not isinstance(keep, bool) and keep >= 0:
            return keep, True
    except Exception:  # noqa: BLE001 — the dump already succeeded; this is a garnish
        pass
    return _DEFAULT_DUMP_KEEP_LAST, False


def _print_retention_line(keep: int, live: bool) -> None:
    """Print the retention line for the dump flavour (D-P37-7): at ``0`` dumps
    accumulate forever (P14c M4), at N the oldest has an expiry; ``live=False``
    says the number is the packaged default, not this daemon's.
    """
    suffix = "" if live else " (default — the daemon's own value could not be read)"
    if keep == 0:
        body = (
            "No dumps are ever deleted on this daemon — set [retention].dump_keep_last "
            "to bound them" + suffix + "."
        )
    else:
        noun = "dump" if keep == 1 else "dumps"
        body = (
            f"The daemon keeps the last {keep} {noun} per database; set "
            "[retention].dump_keep_last to change it" + suffix + "."
        )
    console.print("[dim]" + escape(body) + "[/dim]")


@db_app.command("dump")
def db_dump(
    name: str = typer.Argument(..., help="Managed database name (as shown by `nerdit db list`)"),
    timeout_s: int = typer.Option(
        DUMP_DEFAULT_TIMEOUT_S,
        "--timeout-s",
        help="Seconds the dump tool may run before it is killed",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the custody confirmation."),
) -> None:
    """Capture a logical, application-consistent dump of a managed database.

    The daemon runs the engine's own tool (``pg_dump --format=custom`` /
    ``redis-cli --rdb``) from a sibling container off the database's image and
    packs the artifact into ``nerdit-dump-*.tar.gz`` under its backups directory.
    The request blocks for the whole capture. ``nerdit backup --volume <name>`` is
    the physical alternative. Exit code 1 when refused or failed, else 0.
    """
    code = asyncio.run(_dump_async(name, timeout_s=timeout_s, yes=yes))
    if code:
        raise typer.Exit(code)


async def _dump_async(name: str, *, timeout_s: int, yes: bool) -> int:
    if not yes and not typer.confirm(
        f"Dump '{name}'? The archive holds every row of the database.",
        default=False,
    ):
        console.print("[dim]Aborted.[/dim]")
        return 0

    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.create_database_dump(
            name,
            timeout_s=timeout_s,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    console.print(f"[green]Dump written:[/green] {_plain(result.get('dump'))}")
    console.print(f"  size:   {fmt_bytes(result.get('size_bytes'))}")
    console.print(f"  engine: {_plain(result.get('engine'))} ({_plain(result.get('format'))})")
    duration = result.get("duration_s")
    if duration is not None:
        console.print(f"  took:   {_plain(duration)}s")
    # Custody. The tar is never served over HTTP and never leaves the box on its
    # own: it sits in the daemon's backups directory holding the full contents
    # of the database, which is exactly the sentence an operator needs before
    # deciding where it may be copied to.
    console.print(
        "\n[yellow]This archive contains every row of the database (but no secrets "
        "master key). It stays in the daemon's backups directory — copy it off-box "
        "yourself if you want it elsewhere, and store it as you would the data.[/yellow]"
    )
    keep, live = await _dump_keep_last(client)
    _print_retention_line(keep, live)
    # The cross-reference required by D-P37-7: each flavour points at the other,
    # once, so an operator who reached for the wrong one finds the right one.
    console.print(
        "[dim]Restore it with `nerdit db restore "
        f"{_plain(name)} {_plain(result.get('dump'))}`; for a physical, "
        "crash-consistent copy of the data directory instead, use "
        f"`nerdit backup --volume {_plain(name)}`.[/dim]"
    )
    return 0


@db_app.command("dumps")
def db_dumps(
    name: str = typer.Argument(..., help="Managed database name"),
) -> None:
    """List the dump archives this daemon holds for one database (newest first)."""
    code = asyncio.run(_dumps_async(name))
    if code:
        raise typer.Exit(code)


async def _dumps_async(name: str) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        page = await client.list_database_dumps(name)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    rows = page.get("dumps") or []
    if not rows:
        console.print(
            f"[dim]No dumps for '{_plain(name)}'. Create one with "
            f"`nerdit db dump {_plain(name)}`.[/dim]"
        )
        return 0

    table = Table(show_header=True, header_style="bold", title=f"dumps — {name}")
    table.add_column("dump")
    table.add_column("size", justify="right")
    table.add_column("created")
    for row in rows:
        row = row if isinstance(row, dict) else {}
        table.add_row(
            _plain(row.get("dump")),
            fmt_bytes(row.get("size_bytes")),
            _plain(row.get("created_at")),
        )
    console.print(table)
    if page.get("truncated"):
        console.print(
            "[yellow]Listing capped: only the newest entries are shown. Older dumps still "
            "exist on the box; set [retention].dump_keep_last to bound them.[/yellow]"
        )
    return 0


def _print_restore_dependents(exc: Exception) -> None:
    """Print the running apps a ``409 restore.in_use`` names (D-P37-6): the
    dependents ride the envelope as a list, which ``render_client_error`` does
    not render (the ``services._print_interrupted_tail`` pattern).
    """
    response = getattr(exc, "response", None)
    if response is None:
        return
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 — non-JSON error body
        return
    if not isinstance(body, dict) or body.get("code") != "restore.in_use":
        return
    # Top level first (the shape the delete guard's ``resource.in_use`` also
    # emits), then the detail copy — either is authoritative, neither is
    # guaranteed by the envelope contract on its own.
    deps = body.get("dependents")
    if not isinstance(deps, list):
        detail = body.get("detail")
        deps = detail.get("dependents") if isinstance(detail, dict) else None
    if not isinstance(deps, list) or not deps:
        return
    console.print("[dim]--- running services bound to this database ---[/dim]")
    for dep in deps:
        dep = dep if isinstance(dep, dict) else {}
        binding = dep.get("binding")
        suffix = f" (binding {_plain(binding)})" if binding else ""
        console.print(f"  {_plain(dep.get('service'))}{suffix}")


@db_app.command("restore")
def db_restore(
    name: str = typer.Argument(..., help="Managed database to restore INTO"),
    dump: str = typer.Argument(
        ..., help="Dump basename as listed by `nerdit db dumps` (a basename, never a path)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Restore even while running apps are bound to this database.",
    ),
    timeout_s: int = typer.Option(
        RESTORE_DEFAULT_TIMEOUT_S,
        "--timeout-s",
        help="Seconds the restore may run before it is killed",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the typed confirmation."),
) -> None:
    """Restore a dump into a managed database — **destructive**.

    Postgres: a single transaction, so a failure leaves the database as it was;
    objects created since the dump survive. Redis: the RDB becomes the append-only
    base, so the database is stopped and restarted. Restoring into a different
    database of the same engine is how a clone is made. Exit code 1 when refused
    or failed, else 0.
    """
    code = asyncio.run(_restore_async(name, dump, force=force, timeout_s=timeout_s, yes=yes))
    if code:
        raise typer.Exit(code)


async def _restore_async(name: str, dump: str, *, force: bool, timeout_s: int, yes: bool) -> int:
    if not yes:
        # A typed word, not y/n (the ``nerdit uninstall`` precedent): this replaces
        # live data. The prompt states the Redis bounce and the --force override.
        console.print(
            f"[yellow]This replaces the contents of '{_plain(name)}' with "
            f"'{_plain(dump)}'.[/yellow]"
        )
        console.print(
            "[dim]A Postgres restore runs in one transaction against the live "
            f"database; a Redis restore STOPS and restarts '{_plain(name)}' to swap "
            "its append-only base.[/dim]"
        )
        if force:
            console.print(
                "[dim]--force: running apps bound to this database will see the "
                "restore (dropped connections, contended locks).[/dim]"
            )
        answer = typer.prompt("Type 'restore' to confirm")
        if answer.strip() != "restore":
            console.print("[dim]Aborted — nothing was restored.[/dim]")
            return 0

    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        result = await client.restore_database_dump(
            name,
            dump,
            timeout_s=timeout_s,
            force=force,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # The 409 the operator is most likely to hit carries the list of apps
        # standing in the way; print it before the generic message + hint.
        _print_restore_dependents(exc)
        render_client_error(exc)
        return 1

    console.print(f"[green]Restored:[/green] {_plain(result.get('dump'))}")
    console.print(f"  into:   {_plain(result.get('service_name') or name)}")
    console.print(f"  engine: {_plain(result.get('engine'))}")
    duration = result.get("duration_s")
    if duration is not None:
        console.print(f"  took:   {_plain(duration)}s")
    console.print(
        "[dim]Check the database is serving what you expect before pointing "
        "traffic back at it — `nerdit db list` shows readiness.[/dim]"
    )
    return 0
