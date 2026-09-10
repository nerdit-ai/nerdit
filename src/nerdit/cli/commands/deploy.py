"""Deploy app folders or Git sources, or roll back to the previous image.

CLI flags override nerdit.toml defaults. Each invocation mints an idempotency
key to deduplicate retries; rollback skips uploading and building.
"""

from __future__ import annotations

import asyncio
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
    wait_exit_code,
)


def parse_env_pairs(items: list[str]) -> dict[str, str]:
    """Parse repeatable `KEY=VAL` items into a dict; reject a pair without '='."""
    values: dict[str, str] = {}
    for item in items:
        key, sep, val = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Invalid KEY=VAL pair: '{item}'")
        values[key] = val
    return values


def build_env_map(env: list[str], unset_env: list[str]) -> dict[str, str | None] | None:
    """Combine `--env KEY=VAL` (set) and `--unset-env KEY` (delete) into one map.

    A `--unset-env KEY` maps to a `None` value the daemon reads as a
    null-delete on redeploy (a bare `KEY` — no `=` grammar overload, so an
    empty-string value stays a legitimate set). Returns `None` when neither is
    given so the server-side defaults apply.
    """
    values: dict[str, str | None] = dict(parse_env_pairs(env)) if env else {}
    for key in unset_env:
        if not key or "=" in key:
            raise ValueError(f"Invalid --unset-env key: '{key}' (pass a bare KEY name)")
        values[key] = None
    return values or None


async def _maybe_wait(client, name: str, service: dict, wait: bool, wait_timeout: int) -> None:
    """Wait for the generation stamped in the deploy response.

    Use `last_deploy.version`, falling back to `build_version`, so concurrent
    redeploys report superseded. Exit codes: 0 converged, 1 failed/superseded,
    3 timeout; Typer reserves 2 for usage errors.
    """
    if not wait:
        return
    last_deploy = service.get("last_deploy")
    version = None
    if isinstance(last_deploy, dict) and isinstance(last_deploy.get("version"), int):
        version = last_deploy["version"]
    elif isinstance(service.get("build_version"), int):
        version = service["build_version"]
    try:
        with console.status(f"Waiting for '{escape(name)}' to converge...", spinner="dots"):
            result = await client.wait_for_service(name, version=version, timeout=wait_timeout)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc
    display_wait_outcome(result)
    code = wait_exit_code(result)
    if code:
        raise typer.Exit(code)


def parse_build_settings(value: str | None) -> dict | None:
    """Parse the explicit override object without exposing its contents on failure."""
    import json

    if value is None:
        return None
    try:
        settings = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise typer.BadParameter("must be a JSON object", param_hint="--build-settings") from exc
    if not isinstance(settings, dict):
        raise typer.BadParameter("must be a JSON object", param_hint="--build-settings")
    return settings


def _render_dry_run(plan: dict) -> None:
    """Render a `--dry-run` plan diff (1.10 body) — names only, never values."""
    action = plan.get("action", "?")
    name = plan.get("name", "?")
    console.print(f"[bold]Deploy plan[/bold] ({action}) for [cyan]{name}[/cyan] — dry run")
    console.print(f"  buildpack: {plan.get('buildpack')}")
    build = plan.get("build") or {}
    for key in (
        "preset",
        "framework",
        "node_version",
        "package_manager",
        "install",
        "build",
        "start",
        "subdir",
        "commit_sha",
    ):
        if key in build:
            console.print(f"  {key}: {_plain(str(build[key]))}")
    eff = plan.get("effective") or {}
    parts = ", ".join(f"{k}={eff.get(k)}" for k in ("port", "gpus", "start", "health"))
    console.print(f"  effective: {parts}")
    limits = ", ".join(
        f"{k}={eff.get(k)}" for k in ("memory_limit", "cpu_limit") if eff.get(k) is not None
    )
    if limits:
        console.print(f"  limits: {limits}")
    diff = plan.get("env_diff") or {}
    console.print(
        "  env: "
        f"+{diff.get('added') or []} "
        f"-{diff.get('removed') or []} "
        f"~{diff.get('changed') or []} "
        f"(kept {diff.get('kept', 0)})"
    )
    ai = plan.get("ai_diff") or {}
    console.print(f"  ai: {ai.get('action')} {ai.get('bindings') or []}")
    if plan.get("overwrote_api_config"):
        console.print("  [yellow]would overwrite API-set config[/yellow]")
    # F4-MARKUP: plan warnings are server-derived and now interpolate app-authored
    # ``[deploy]`` key names (the Agent-DX unknown-key advisory), so they carry
    # both crash shapes ``_plain`` exists for — a stray ``[/b]`` raises
    # ``MarkupError``, and the literal ``[deploy]``/``[ai]``/``[db]`` section
    # names are silently eaten as style tags. Escape exactly like
    # ``display_deploy_result`` does for the response ``hints``.
    for warning in plan.get("warnings") or []:
        console.print(f"  [yellow]warning:[/yellow] {_plain(warning)}")


def _git_name_default(repo_url: str, subdir: str | None) -> str:
    """Derive a service name from a Git deploy: subdir basename, else repo slug.

    When `--subdir` is given the subdir's basename wins (the deployed app is
    that subtree); otherwise the last URL path segment with a trailing `.git`
    stripped.
    """
    if subdir:
        return Path(subdir).name
    tail = repo_url.rstrip("/").rsplit("/", 1)[-1]
    return tail[:-4] if tail.endswith(".git") else tail


def deploy(
    path: Optional[str] = typer.Argument(
        None, help="App directory to deploy (default: current directory)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Service name (DNS label; default: [deploy].name or folder)"
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p", help="Container port the app listens on"
    ),
    gpus: Optional[int] = typer.Option(None, "--gpus", "-g", help="GPUs the app needs"),
    start: Optional[str] = typer.Option(None, "--start", help="Start command override"),
    build_settings: Optional[str] = typer.Option(
        None,
        "--build-settings",
        help=(
            "JSON build overrides, including preset (node/nextjs/python/dockerfile); "
            "null resets, build:false skips compilation"
        ),
    ),
    health: Optional[str] = typer.Option(
        None, "--health", help="HTTP path to probe for health (e.g. /healthz)"
    ),
    env: list[str] = typer.Option(
        [], "--env", "-e", help="Environment variable KEY=VAL (repeatable)"
    ),
    unset_env: list[str] = typer.Option(
        [],
        "--unset-env",
        help="Environment variable KEY to delete on redeploy (repeatable)",
    ),
    vendor: Optional[str] = typer.Option(None, "--vendor", help="Force a GPU vendor"),
    rollback: bool = typer.Option(
        False, "--rollback", help="Roll back to the previous deployed version (no build)"
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo", help="Deploy from a Git URL instead of a local folder"
    ),
    ref: Optional[str] = typer.Option(
        None, "--ref", help="Git branch or tag to clone (requires --repo)"
    ),
    subdir: Optional[str] = typer.Option(
        None, "--subdir", help="Subdirectory within the repo to deploy (requires --repo)"
    ),
    token_ref: Optional[str] = typer.Option(
        None,
        "--token-ref",
        help="Secret reference for a private repo, e.g. ${secrets.shared.GITHUB_TOKEN}, "
        "or ${github.installation} on a linked node with the Nerdit GitHub App "
        "(requires --repo)",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Block until the deploy converges (healthy) or fails. "
        "Exit codes: 0 converged, 1 failed or superseded, 3 timeout.",
    ),
    wait_timeout: int = typer.Option(
        60, "--timeout", help="Seconds to wait with --wait (server-clamped to [1, 300])"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate + print the plan diff without deploying (no build, no write)",
    ),
) -> None:
    """Deploy an app folder (zip + upload) or a Git URL, then build server-side."""
    asyncio.run(
        _deploy_async(
            path,
            name,
            port,
            gpus,
            start,
            health,
            env,
            vendor,
            rollback,
            repo=repo,
            ref=ref,
            subdir=subdir,
            token_ref=token_ref,
            wait=wait,
            wait_timeout=wait_timeout,
            unset_env=unset_env,
            dry_run=dry_run,
            build_settings=parse_build_settings(build_settings),
        )
    )


async def _deploy_async(
    path: str | None,
    name: str | None,
    port: int | None,
    gpus: int | None,
    start: str | None,
    health: str | None,
    env: list[str],
    vendor: str | None,
    rollback: bool,
    *,
    repo: str | None = None,
    ref: str | None = None,
    subdir: str | None = None,
    token_ref: str | None = None,
    wait: bool = False,
    wait_timeout: int = 60,
    unset_env: list[str] | None = None,
    dry_run: bool = False,
    build_settings: dict | None = None,
) -> None:
    """Merge effective deploy parameters over nerdit.toml [deploy], zip and POST."""
    from nerdit.cli.client import get_configured_client
    from nerdit.config.project import find_project_config, load_project_config

    unset_env = unset_env or []

    # --dry-run is a plan preview: it cannot combine with --rollback (rollback
    # is an image swap, not a build with a plan) and skips the --wait handoff.
    if dry_run and rollback:
        console.print("[red]--dry-run cannot be combined with --rollback.[/red]")
        raise typer.Exit(1)

    # Git mode: --repo is mutually exclusive with a local path and --rollback;
    # --ref/--subdir/--token-ref only make sense alongside --repo.
    if repo:
        if path:
            console.print("[red]--repo cannot be combined with a local path argument.[/red]")
            raise typer.Exit(1)
        if rollback:
            console.print("[red]--repo cannot be combined with --rollback.[/red]")
            raise typer.Exit(1)
    elif ref or subdir or token_ref:
        console.print("[red]--ref, --subdir and --token-ref require --repo.[/red]")
        raise typer.Exit(1)

    if repo:
        try:
            env_values = build_env_map(env, unset_env)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        effective_name = name or _git_name_default(repo, subdir)
        client = get_configured_client()
        console.print(f"[dim]Cloning {repo}...[/dim]")
        try:
            service = await client.deploy_git(
                repo_url=repo,
                name=effective_name,
                ref=ref,
                subdir=subdir,
                port=port,
                gpus=gpus,
                start=start,
                build_settings=build_settings,
                health=health,
                env=env_values,
                vendor=vendor,
                token_ref=token_ref,
                idempotency_key=uuid4().hex,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001 — rendered for the user
            render_client_error(exc)
            raise typer.Exit(1) from exc
        if dry_run:
            _render_dry_run(service)
            return
        display_deploy_result(service)
        await _maybe_wait(client, effective_name, service, wait, wait_timeout)
        return

    directory = Path(path).resolve() if path else Path.cwd()
    if not directory.is_dir():
        console.print(f"[red]Not a directory:[/red] {directory}")
        raise typer.Exit(1)

    # Locate nerdit.toml from the app directory, then read [deploy].
    config_path = find_project_config(directory)
    project = load_project_config(config_path)
    if config_path:
        console.print(f"[dim]nerdit.toml found at {config_path}[/dim]")
    deploy_cfg = project.deploy if project and project.deploy else None

    # Merge: CLI flags override [deploy] defaults; name falls back to the folder.
    effective_name = name or (deploy_cfg.name if deploy_cfg else None) or directory.name

    client = get_configured_client()

    if rollback:
        try:
            service = await client.rollback_deploy(effective_name, idempotency_key=uuid4().hex)
        except Exception as exc:  # noqa: BLE001 — rendered for the user
            render_client_error(exc)
            raise typer.Exit(1) from exc
        display_deploy_result(service, heading="Rolled back", build_hint=False)
        await _maybe_wait(client, effective_name, service, wait, wait_timeout)
        return

    effective_port = port if port is not None else (deploy_cfg.port if deploy_cfg else None)
    effective_gpus = gpus if gpus is not None else (deploy_cfg.gpus if deploy_cfg else None)
    effective_health = health or (deploy_cfg.health if deploy_cfg else None)

    try:
        env_values = build_env_map(env, unset_env)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if build_settings is not None or (deploy_cfg and deploy_cfg.build_settings is not None):
        try:
            await client.require_build_settings_support(
                build_settings,
                directory=directory,
            )
        except Exception as exc:  # noqa: BLE001 — rendered for the user
            render_client_error(exc)
            raise typer.Exit(1) from exc

    from nerdit.cli.upload import check_upload_size, create_dir_zip, warn_large_upload

    console.print(f"[dim]Creating archive of {directory}...[/dim]")
    zip_bytes = create_dir_zip(directory)

    try:
        check_upload_size(zip_bytes)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    warning = warn_large_upload(zip_bytes)
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")

    size_mb = len(zip_bytes) / (1024 * 1024)
    console.print(f"[dim]Uploading archive ({size_mb:.1f} MB)...[/dim]")

    try:
        service = await client.deploy(
            zip_bytes=zip_bytes,
            _build_settings_checked=True,
            name=effective_name,
            port=effective_port,
            gpus=effective_gpus,
            start=start,
            build_settings=build_settings,
            health=effective_health,
            env=env_values,
            vendor=vendor,
            idempotency_key=uuid4().hex,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    if dry_run:
        _render_dry_run(service)
        return
    display_deploy_result(service)
    await _maybe_wait(client, effective_name, service, wait, wait_timeout)
