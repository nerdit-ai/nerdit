"""Apply a `[project]` / `[services.*]` / `[vars]` declaration (P40d, D-P40-12).

`nerdit apply [dir]` zips the folder holding `nerdit.toml` and posts it to
`POST /api/projects/{project}/apply`; `--repo` applies a Git source instead.
Exit codes: 0 applied or planned, 1 refused or failed, 3 `--wait` timeout,
4 waiting for variables (nothing was deployed).
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer

from nerdit.cli.commands.deploy import _git_name_default, _maybe_wait, _render_dry_run
from nerdit.cli.display import _plain, console, render_client_error

#: `waiting_for_variables`: distinct from 1 (failed) and 3 (`--wait` timeout),
#: so a script can tell "set the keys and re-run" from a real failure.
EXIT_WAITING_FOR_VARIABLES = 4


def read_declaration(directory: Path) -> dict | None:
    """Return the folder's parsed `nerdit.toml` when it is a project declaration.

    Only the folder's own file counts (no upward search): the archive root must
    hold the declaration the daemon reads. `None` means "legacy or absent";
    malformed TOML is also `None`, so the legacy loader reports it as before.
    """
    from nerdit.config.project import PROJECT_CONFIG_NAME, is_project_declaration

    try:
        data = tomllib.loads((directory / PROJECT_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return data if is_project_declaration(data) else None


def apply(
    path: Optional[str] = typer.Argument(
        None, help="Folder holding the declaration's nerdit.toml (default: current directory)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and plan every service without writing anything"
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo", help="Apply the nerdit.toml at the root of a Git URL instead of a folder"
    ),
    ref: Optional[str] = typer.Option(
        None, "--ref", help="Git branch or tag to clone (requires --repo)"
    ),
    token_ref: Optional[str] = typer.Option(
        None,
        "--token-ref",
        help="Secret reference for a private repo, e.g. ${vars.GH_TOKEN} (requires --repo)",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Project name with --repo (default: the repo slug); must equal [project].name",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Block until every applied service converges or fails. "
        "Exit codes: 0 converged, 1 failed or superseded, 3 timeout.",
    ),
    wait_timeout: int = typer.Option(
        60, "--timeout", help="Seconds to wait per service with --wait (server-clamped to [1, 300])"
    ),
) -> None:
    """Apply a project declaration: every [services.<name>] table, one by one.

    When [vars] required names a variable no scope provides yet, nothing is
    deployed: the missing names and the `nerdit vars set` command are printed
    and the exit code is 4. There is no rollback across services: a failure
    names the services already applied, and re-running retries the rest.
    """
    asyncio.run(
        _apply_async(
            path,
            dry_run=dry_run,
            repo=repo,
            ref=ref,
            token_ref=token_ref,
            project=project,
            wait=wait,
            wait_timeout=wait_timeout,
        )
    )


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]{message}[/red]")
    return typer.Exit(1)


def _folder_source(path: str | None) -> tuple[str, bytes]:
    """Validate the folder's declaration client-side and zip it: `(project, zip_bytes)`."""
    from nerdit.cli.upload import check_upload_size, create_dir_zip
    from nerdit.config.project import (
        PROJECT_CONFIG_NAME,
        DeclarationError,
        parse_project_declaration,
    )

    directory = Path(path).resolve() if path else Path.cwd()
    if not directory.is_dir():
        raise _fail(f"Not a directory: {directory}")
    data = read_declaration(directory)
    if data is None:
        raise _fail(
            f"No [project] declaration in {directory / PROJECT_CONFIG_NAME}. "
            "A [deploy]-only app deploys with `nerdit deploy`."
        )
    try:
        # `new_project=False`: whether the name is new is the daemon's judgment
        # (D-P40-14 binds only where a NEW name enters); messages are value-free.
        name, _, _ = parse_project_declaration(data, new_project=False)
    except DeclarationError as exc:
        raise _fail(_plain(str(exc))) from exc
    console.print(f"[dim]Creating archive of {directory}...[/dim]")
    zip_bytes = create_dir_zip(directory)
    try:
        check_upload_size(zip_bytes)
    except ValueError as exc:
        raise _fail(f"Error: {exc}") from exc
    return name, zip_bytes


def _render_result(result: dict) -> None:
    """Render an applied / planned response: labels, actions, URLs. Never a value."""
    project = _plain(result.get("project"))
    if result.get("dry_run"):
        console.print(f"[bold]Apply plan[/bold] for project [cyan]{project}[/cyan] — dry run")
    else:
        console.print(f"[green]Applied project:[/green] {project}")
    for entry in result.get("services") or []:
        label = _plain(entry.get("label"))
        line = f"  {_plain(entry.get('service'))} → {label} ({_plain(entry.get('action'))})"
        if entry.get("status"):
            line += f": {_plain(entry['status'])}"
        console.print(line)
        if isinstance(entry.get("plan"), dict):
            _render_dry_run(entry["plan"])
    for url in result.get("public_urls") or []:
        if url.get("url"):
            console.print(f"  URL: {_plain(url['url'])}")


async def _apply_async(
    path: str | None,
    *,
    dry_run: bool = False,
    repo: str | None = None,
    ref: str | None = None,
    token_ref: str | None = None,
    project: str | None = None,
    wait: bool = False,
    wait_timeout: int = 60,
) -> None:
    """Resolve the source, POST the apply, render it, then optionally wait per label."""
    from nerdit.cli.client import get_configured_client

    if repo and path:
        raise _fail("--repo cannot be combined with a local path argument.")
    if not repo and (ref or token_ref or project):
        raise _fail("--ref, --token-ref and --project require --repo.")

    zip_bytes: bytes | None = None
    if repo:
        name = project or _git_name_default(repo, None)
    else:
        name, zip_bytes = _folder_source(path)

    client = get_configured_client()
    try:
        result = await client.apply_project(
            name,
            zip_bytes=zip_bytes,
            repo_url=repo,
            ref=ref,
            token_ref=token_ref,
            dry_run=dry_run,
            idempotency_key=None if dry_run else uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    if result.get("status") == "waiting_for_variables":
        missing = [_plain(key) for key in result.get("missing") or []]
        console.print(
            f"[yellow]Waiting for variables:[/yellow] {', '.join(missing)} — nothing was deployed."
        )
        for key in missing:
            console.print(f"  nerdit vars set {_plain(name)} --secret --prompt {key}")
        console.print(
            f"[dim]A plain value: `nerdit vars set {_plain(name)} KEY=value`. "
            "Then run the apply again.[/dim]"
        )
        raise typer.Exit(EXIT_WAITING_FOR_VARIABLES)

    _render_result(result)
    if dry_run or not wait:
        return
    code = 0
    for entry in result.get("services") or []:
        try:
            await _maybe_wait(client, entry["label"], entry, True, wait_timeout)
        except typer.Exit as exc:
            code = max(code, exc.exit_code)
    if code:
        raise typer.Exit(code)
