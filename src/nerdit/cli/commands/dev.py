"""Watch an app folder and redeploy after changes settle.

Poll mtimes every second with a 400 ms debounce; prune upload-excluded dirs.
Git services redeploy the recorded remote source, so unpushed local edits are
not included. Other services upload a ZIP. Wait for each generation and print
its URL or failure remediation.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer
from rich.markup import escape

from nerdit.cli.display import (
    _plain,
    console,
    display_deploy_result,
    display_wait_outcome,
    render_client_error,
)
from nerdit.cli.upload import should_exclude

# One poll per second; a change seen on a poll is deployed only once the tree
# has been quiet for the debounce, so a burst of saves (a formatter rewriting
# twenty files, a build tool touching a tree) collapses into one deploy.
POLL_INTERVAL_S = 1.0
DEBOUNCE_S = 0.4


def dev(
    path: Optional[str] = typer.Argument(
        None, help="App directory to watch (default: current directory)"
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        # No "[deploy]" in the help text: Rich reads a bracketed word as a
        # markup tag and swallows it, so the hint would render as ".name".
        help="Service name (default: the nerdit.toml deploy name, else the folder name)",
    ),
    wait_timeout: int = typer.Option(
        120, "--timeout", help="Seconds to wait for each redeploy (server-clamped to [1, 300])"
    ),
) -> None:
    """Watch a folder and redeploy on every change (Ctrl-C to stop)."""
    try:
        asyncio.run(_dev_async(path, name, wait_timeout=wait_timeout))
    except KeyboardInterrupt:  # pragma: no cover — a signal delivered outside the loop
        console.print("\n[dim]Stopped watching.[/dim]")


# --- change detection ---------------------------------------------------------


def _rel(rel_dir: str, name: str) -> str:
    """Join a walk-relative directory with an entry name (`.` = the root)."""
    return name if rel_dir == "." else f"{rel_dir}/{name}"


def _snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    """Snapshot `(mtime_ns, size, inode)` per watched file, excluded dirs pruned in-walk.

    Size rides along with the mtime because a coarse filesystem clock can stamp
    two writes within one tick identically — the tuple changes where the mtime
    alone would not.
    """
    snap: dict[str, tuple[int, int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        # In-place, so os.walk never descends into an excluded tree.
        dirnames[:] = [d for d in dirnames if not should_exclude(_rel(rel_dir, d))]
        for filename in filenames:
            rel = _rel(rel_dir, filename)
            if should_exclude(rel):
                continue
            try:
                stat = os.stat(os.path.join(dirpath, filename))
            except OSError:
                # Vanished between walk and stat (an editor's atomic replace):
                # simply absent from this snapshot, seen again once it settles.
                continue
            # Nanosecond mtime detects rapid writes; inode detects atomic-rename saves.
            # Equal-size in-place writes that restore mtime remain an accepted blind spot;
            # detecting them would require hashing every file on every poll.
            snap[rel] = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    return snap


# --- server-state helpers -----------------------------------------------------


def _source_type(service: object) -> str | None:
    """Read `source.type` off a service response (`git` | `zip` | None)."""
    if not isinstance(service, dict):
        return None
    source = service.get("source")
    if not isinstance(source, dict):
        return None
    value = source.get("type")
    return value if isinstance(value, str) else None


def _deploy_version(service: dict) -> int | None:
    """Return the response-stamped generation so waiting cannot target a concurrent deploy."""
    last_deploy = service.get("last_deploy")
    if isinstance(last_deploy, dict) and isinstance(last_deploy.get("version"), int):
        return last_deploy["version"]
    if isinstance(service.get("build_version"), int):
        return service["build_version"]
    return None


def _cutover_warning(name: str, service: dict | None, cutover: bool | None) -> list[str]:
    """Warn when observed settings or eligibility prevent zero-downtime cutover.

    Check effective opt-out, workload kind, GPU count and proxy routing. Missing
    health checks use the grace fallback. A missing URL matters only once running;
    a building generation may not have registered its route yet.
    """
    if cutover is False:
        reason = (
            "[deploy].cutover = false (nerdit.toml or app config) — "
            "set it to true for zero-downtime"
        )
    elif service is None:
        return []
    elif service.get("kind", "service") != "service":
        reason = "only apps (kind = service) are cutover-eligible"
    elif (service.get("gpu_count") or 0) > 0:
        reason = "GPU services keep the same-port swap (two containers cannot hold one GPU)"
    elif service.get("status") in ("running", "degraded") and not (
        service.get("endpoint") or {}
    ).get("public_url"):
        reason = "no proxy route — zero-downtime redeploys need [proxy].enabled"
    else:
        return []
    return [
        f"[yellow]warning:[/yellow] '{_plain(name)}' is not cutover-armed — "
        "each redeploy has a downtime window.",
        # `reason` is our own literal but carries [deploy]/[proxy] shapes Rich
        # would read as markup tags; escape it like any other untrusted string.
        f"[dim]  {escape(reason)}[/dim]",
    ]


async def _fetch_service(client, name: str) -> dict | None:
    """Fetch startup service details; return None only for a missing service.

    Other errors abort startup so a failed read cannot misclassify a Git service
    as ZIP-sourced and overwrite it with local files.
    """
    try:
        return await client.resolve_service(name)
    except Exception as exc:  # noqa: BLE001 — rendered, then the session refuses to start
        render_client_error(exc)
        raise typer.Exit(1) from exc


async def _resolve_cutover(
    client, name: str, service: dict | None, deploy_cfg: object
) -> bool | None:
    """Read local cutover settings, falling back to stored app config.

    Stored opt-outs survive redeploys. Fetch once at startup; failure means unknown
    because this setting is used only for an advisory warning.
    """
    local = getattr(deploy_cfg, "cutover", None)
    if local is not None or service is None:
        return local if isinstance(local, bool) else None
    try:
        config = await client.get_app_config(name)
    except Exception:  # noqa: BLE001 — advisory only; never fails the session
        return None
    deploy = config.get("deploy") if isinstance(config, dict) else None
    value = deploy.get("cutover") if isinstance(deploy, dict) else None
    return value if isinstance(value, bool) else None


# --- one deploy generation ----------------------------------------------------


async def _print_remediation(client, name: str) -> None:
    """Print the bounded `remediation_code` for a failed generation (best effort)."""
    try:
        data = await client.diagnose_service(name, log_tail=1)
    except Exception:  # noqa: BLE001 — the failure itself is already rendered
        return
    remediation = data.get("remediation") or {}
    code = remediation.get("code")
    if not code:
        return
    console.print(f"  remediation: {_plain(code)}")
    detail = remediation.get("detail")
    if detail:
        console.print(f"  {_plain(detail)}")


async def _wait_and_report(client, name: str, service: dict, wait_timeout: int) -> None:
    """Block on `/wait` for this generation and render the outcome.

    Never exits the process: a failed generation is information for the next
    save, not the end of the watch session.
    """
    try:
        result = await client.wait_for_service(
            name, version=_deploy_version(service), timeout=wait_timeout
        )
    except Exception as exc:  # noqa: BLE001 — rendered; the watch loop keeps running
        render_client_error(exc)
        return
    display_wait_outcome(result)
    if result.get("outcome") != "converged":
        await _print_remediation(client, name)


async def _deploy_once(
    client,
    directory: Path,
    name: str,
    source_type: str | None,
    wait_timeout: int,
    max_bytes: int,
) -> str | None:
    """Redeploy once; returns the source type to use for the next change."""
    from nerdit.cli.upload import check_upload_size, create_dir_zip

    # `name` comes from --name or nerdit.toml and is only validated server-side,
    # so it is escaped at every Rich sink like any other untrusted string.
    label = _plain(name)
    try:
        if source_type == "git":
            console.print(f"[dim]Change detected — redeploying {label} from its source...[/dim]")
            service = await client.redeploy_app(name)
        else:
            console.print(f"[dim]Change detected — uploading {label}...[/dim]")
            zip_bytes = create_dir_zip(directory)
            check_upload_size(zip_bytes, max_bytes)
            service = await client.deploy(
                zip_bytes=zip_bytes, name=name, idempotency_key=uuid4().hex
            )
    except ValueError as exc:  # an oversized archive — local, not a client error
        console.print(f"[red]Error:[/red] {exc}")
        return source_type
    except Exception as exc:  # noqa: BLE001 — rendered; the watch loop keeps running
        render_client_error(exc)
        return source_type

    display_deploy_result(service, heading="Deploying", build_hint=False)
    await _wait_and_report(client, name, service, wait_timeout)
    return _source_type(service) or source_type


# --- the loop -----------------------------------------------------------------


async def _watch(
    client, directory: Path, name: str, source_type: str | None, wait_timeout: int
) -> int:
    """Poll → debounce → deploy until Ctrl-C; returns the number of deploys run.

    A poll that sees a change re-arms the debounce instead of deploying, so a
    burst spanning several polls still produces exactly one deploy once the
    tree goes quiet.
    """
    from nerdit.cli.client import upload_limit

    # Read once: the cap is a restart-required daemon setting.
    max_bytes = await upload_limit(client)
    snapshot = _snapshot(directory)
    pending_since: float | None = None
    deploys = 0
    try:
        while True:
            await asyncio.sleep(POLL_INTERVAL_S)
            current = _snapshot(directory)
            if current != snapshot:
                snapshot = current
                pending_since = time.monotonic()
                continue
            if pending_since is None or time.monotonic() - pending_since < DEBOUNCE_S:
                continue
            pending_since = None
            deploys += 1
            source_type = await _deploy_once(
                client, directory, name, source_type, wait_timeout, max_bytes
            )
            # Deliberately NOT re-baselined here: the build runs server-side and
            # writes nothing locally, so a save made *while* a deploy was in
            # flight must still be seen by the next poll. Swallowing it would
            # lose an edit — strictly worse than one extra deploy.
    except KeyboardInterrupt:
        return deploys


async def _dev_async(path: str | None, name: str | None, *, wait_timeout: int) -> None:
    from nerdit.cli.client import get_configured_client
    from nerdit.config.project import (
        describe_project_config_error,
        find_project_config,
        load_project_config,
    )

    directory = Path(path).resolve() if path else Path.cwd()
    if not directory.is_dir():
        console.print(f"[red]Not a directory:[/red] {directory}")
        raise typer.Exit(1)

    # (P40d) Every ingress the watch loop uses answers 422 `deploy.use_apply` to
    # a declaration, so refuse once here instead of on every save.
    # ponytail: no watch mode for declared projects; re-applying per save would
    # rebuild every service -- add a per-subdir apply if this is wanted.
    from nerdit.cli.commands.apply import read_declaration

    if read_declaration(directory) is not None:
        console.print(
            "[red]This folder is a project declaration:[/red] `nerdit dev` watches a "
            "single [deploy] app. Run `nerdit apply` after a change instead."
        )
        raise typer.Exit(1)

    # Only load when a file was actually found: `load_project_config(None)`
    # falls back to searching upward from *cwd*, which would read some unrelated
    # project's [deploy] when the watched folder has no nerdit.toml of its own.
    config_path = find_project_config(directory)
    try:
        project = load_project_config(config_path) if config_path else None
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        console.print(f"[red]{_plain(describe_project_config_error(exc))}[/red]")
        raise typer.Exit(1) from exc
    deploy_cfg = project.deploy if project and project.deploy else None
    effective_name = name or (deploy_cfg.name if deploy_cfg else None) or directory.name

    client = get_configured_client()
    service = await _fetch_service(client, effective_name)
    source_type = _source_type(service)

    cutover = await _resolve_cutover(client, effective_name, service, deploy_cfg)
    for line in _cutover_warning(effective_name, service, cutover):
        console.print(line)
    if source_type == "git":
        console.print(
            "[dim]Git-sourced: each change re-clones the recorded repo/ref — "
            "push first, local edits are not uploaded.[/dim]"
        )
    console.print(
        f"[bold]Watching[/bold] {escape(str(directory))} → "
        f"[cyan]{_plain(effective_name)}[/cyan]  [dim](Ctrl-C to stop)[/dim]"
    )

    deploys = await _watch(client, directory, effective_name, source_type, wait_timeout)
    console.print(f"\n[dim]Stopped watching after {deploys} deploy(s).[/dim]")
