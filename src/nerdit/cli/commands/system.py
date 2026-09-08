"""Report disk usage and collect orphan app images or opt-in orphan data.

GC previews its plan, confirms unless --yes, then sends an idempotent write.
Escape all server-derived Rich text.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import typer
from rich.table import Table

from nerdit.cli.display import console, fmt_bytes, render_client_error
from nerdit.cli.display import plain as _plain


def disk() -> None:
    """Show disk usage: docker totals + named-volume/model/archive/backup trees.

    Exit code 1 when the daemon is unreachable, else 0.
    """
    code = asyncio.run(_disk_async())
    if code:
        raise typer.Exit(code)


async def _disk_async() -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    try:
        data = await client.get_system_disk()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1

    docker = data.get("docker")
    table = Table(show_header=True, header_style="bold", title="docker")
    table.add_column("kind")
    table.add_column("size", justify="right")
    if docker is None:
        table.add_row("docker", "[dim]unavailable[/dim]")
    else:
        for key in ("images_bytes", "containers_bytes", "volumes_bytes", "build_cache_bytes"):
            table.add_row(_plain(key), fmt_bytes(docker.get(key)))
    console.print(table)

    data_dir = data.get("data_dir", {})
    dtable = Table(show_header=True, header_style="bold", title="data dir")
    dtable.add_column("tree")
    dtable.add_column("size", justify="right")
    for svc in data_dir.get("services", []):
        dtable.add_row(f"services/{_plain(svc.get('name'))}", fmt_bytes(svc.get("bytes")))
    models = data_dir.get("models", {})
    dtable.add_row("models/ollama", fmt_bytes(models.get("ollama")))
    dtable.add_row("models/huggingface", fmt_bytes(models.get("huggingface")))
    dtable.add_row("archive", fmt_bytes(data_dir.get("archive_bytes")))
    backups = data_dir.get("backups", {})
    dtable.add_row(
        "backups", f"{fmt_bytes(backups.get('bytes'))} ({_plain(backups.get('count'))} files)"
    )
    console.print(dtable)

    orphan_images = data.get("orphan_images", [])
    orphan_dirs = data.get("orphan_data_dirs", [])
    console.print(
        "orphan images: "
        + (", ".join(_plain(r) for r in orphan_images) if orphan_images else "[dim]none[/dim]")
    )
    console.print(
        "orphan data dirs: "
        + (", ".join(_plain(n) for n in orphan_dirs) if orphan_dirs else "[dim]none[/dim]")
    )
    for warning in data.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {_plain(warning)}")
    return 0


def gc(
    dry_run: bool = typer.Option(False, "--dry-run", help="Only show what would be reclaimed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    include_orphan_data: bool = typer.Option(
        False,
        "--include-orphan-data",
        help="Also delete service data dirs with no live row (IRREVERSIBLE).",
    ),
) -> None:
    """Garbage-collect orphan app images (and, opt-in, orphan data dirs).

    Always previews the plan first; a real run then asks for confirmation
    (unless `--yes`). Exit code 1 when the daemon is unreachable.
    """
    code = asyncio.run(_gc_async(dry_run=dry_run, yes=yes, include_orphan_data=include_orphan_data))
    if code:
        raise typer.Exit(code)


def _render_gc(data: dict) -> None:
    images = data.get("images", {})
    removed = images.get("removed", [])
    skipped = images.get("skipped", [])
    verb = "would remove" if data.get("dry_run") else "removed"
    console.print(
        f"images {verb}: "
        + (", ".join(_plain(r) for r in removed) if removed else "[dim]none[/dim]")
    )
    for entry in skipped:
        console.print(
            f"[yellow]image skipped[/yellow] {_plain(entry.get('repo'))}"
            f" ({_plain(entry.get('reason'))})"
        )
    console.print(f"estimated reclaim: {fmt_bytes(images.get('reclaimed_bytes_estimate'))}")

    orphan_data = data.get("orphan_data", {})
    if orphan_data.get("enabled"):
        d_removed = orphan_data.get("removed", [])
        console.print(
            f"data dirs {verb}: "
            + (", ".join(_plain(n) for n in d_removed) if d_removed else "[dim]none[/dim]")
        )
        for entry in orphan_data.get("skipped", []):
            console.print(
                f"[yellow]data dir skipped[/yellow] {_plain(entry.get('name'))}"
                f" ({_plain(entry.get('reason'))})"
            )


async def _gc_async(*, dry_run: bool, yes: bool, include_orphan_data: bool) -> int:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()

    # Always show the dry-run plan first.
    try:
        plan = await client.run_system_gc(include_orphan_data=include_orphan_data, dry_run=True)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1
    console.print("[bold]plan (dry run):[/bold]")
    _render_gc(plan)

    if dry_run:
        return 0

    if not yes and not typer.confirm("Proceed with garbage collection?", default=False):
        console.print("[dim]Aborted.[/dim]")
        return 0

    try:
        result = await client.run_system_gc(
            include_orphan_data=include_orphan_data, dry_run=False, idempotency_key=uuid4().hex
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        return 1
    console.print("[bold]result:[/bold]")
    _render_gc(result)
    return 0
