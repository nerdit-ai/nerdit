"""Register a prebuilt service image or serve a model.

CLI flags override nerdit.toml defaults; register requests mint idempotency
keys. Existing directories select app registration. Non-path model references
select model serving; ambiguous bare words fail rather than guess.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer

from nerdit.cli.display import console, display_service_submitted, render_client_error

# Strict model-reference shape (S7): 'llama3.1:8b', 'phi3.5', 'library/llama:tag'.
# Deliberately narrow — anything else falls through to the explicit error below.
_MODEL_REF_RE = re.compile(r"^[\w.\-/]+(:[\w.\-]+)?$")


def _is_model_ref(arg: str) -> bool:
    """Recognize model references while preserving explicit path intent.

    Leading `.`, `/` or `~` means path. Otherwise `:`, `.` or `/` signals a model,
    including untagged Hugging Face and Ollama repo names; bare words are ambiguous.
    """
    if arg.startswith((".", "/", "~")):
        return False
    if ":" not in arg and "." not in arg and "/" not in arg:
        return False
    return _MODEL_REF_RE.fullmatch(arg) is not None


def serve(
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "App directory holding nerdit.toml (default: current directory), "
            "or a model reference like 'llama3.1:8b' to serve a local model"
        ),
    ),
    image: Optional[str] = typer.Option(
        None, "--image", "-i", help="Prebuilt image to run (required — register-only in P2)"
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Service name (DNS label)"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Container port to publish"),
    gpus: Optional[int] = typer.Option(None, "--gpus", "-g", help="GPUs the service needs"),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Model serving backend: ollama (CPU/GPU) or vllm (GPU-only). "
        "Model serves only; default from the daemon config.",
    ),
    max_model_len: Optional[int] = typer.Option(
        None,
        "--max-model-len",
        help="Model serves (vLLM) only: cap the context window (256-262144)",
    ),
    gpu_memory_utilization: Optional[float] = typer.Option(
        None,
        "--gpu-memory-utilization",
        help="Model serves (vLLM) only: VRAM fraction the engine may claim (0.1-0.95)",
    ),
    restart_policy: Optional[str] = typer.Option(
        None, "--restart-policy", help="no | on-failure | always (default: always)"
    ),
    command: Optional[str] = typer.Option(
        None, "--command", "-c", help="Override the image CMD with a shell command"
    ),
    script: Optional[str] = typer.Option(
        None, "--script", help="Run a script under the image's nerdit-runtime"
    ),
    health_path: Optional[str] = typer.Option(
        None, "--health", help="HTTP path to probe for health (e.g. /healthz)"
    ),
) -> None:
    """Register a service from nerdit.toml [deploy] + CLI flags, or serve a model."""
    asyncio.run(
        _serve_async(
            path,
            image,
            name,
            port,
            gpus,
            backend,
            restart_policy,
            command,
            script,
            health_path,
            max_model_len,
            gpu_memory_utilization,
        )
    )


async def _serve_async(
    path: str | None,
    image: str | None,
    name: str | None,
    port: int | None,
    gpus: int | None,
    backend: str | None,
    restart_policy: str | None,
    command: str | None,
    script: str | None,
    health_path: str | None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> None:
    """Merge effective service parameters over nerdit.toml [deploy] and register."""
    from nerdit.cli.client import get_configured_client
    from nerdit.config.project import find_project_config, load_project_config

    # S7 (P5): classify the positional BEFORE the nerdit.toml walk-up — for a
    # nonexistent path, find_project_config would silently pick up an ancestor
    # nerdit.toml and register a different app. An existing path (directory or
    # file) always keeps the shipped app path, byte-identical.
    if path is not None and not Path(path).exists():
        # An explicit --backend is unambiguous "serve a model" intent, so a bare
        # name (no ':'/'.'/'/') the shape heuristic can't classify still serves —
        # but a leading path char (./  /  ~) always wins as path intent so
        # `serve ./typo --backend vllm` never silently becomes a model serve.
        explicit_model = backend is not None and not path.startswith((".", "/", "~"))
        if _is_model_ref(path) or explicit_model:
            await _serve_model_async(
                path,
                name=name,
                gpus=gpus,
                backend=backend,
                image=image,
                port=port,
                restart_policy=restart_policy,
                command=command,
                script=script,
                health_path=health_path,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            return
        console.print(
            f"[red]'{path}' is not an existing app directory and does not look like a "
            "model reference. Use ./path for an app, name:tag for a model, or pass "
            "--backend to force a model serve.[/red]"
        )
        raise typer.Exit(1)

    # Locate nerdit.toml from the given directory (or cwd), then read [deploy].
    start = Path(path).resolve() if path else None
    config_path = find_project_config(start)
    project = load_project_config(config_path)
    if config_path:
        console.print(f"[dim]nerdit.toml found at {config_path}[/dim]")
    deploy = project.deploy if project and project.deploy else None

    # --backend is model-only; on the app path it has no meaning (apps get AI via
    # [ai.*] bindings, not a serving backend). Fail loudly rather than ignore it.
    if backend is not None:
        console.print(
            "[red]--backend only applies to model serves, not app services. "
            "Drop it, or pass a model reference (name:tag) instead of a path.[/red]"
        )
        raise typer.Exit(1)

    # (P21 D4) Same rule for the vLLM engine bounds: model-only, so an app path
    # carrying them is a mistake, not something to silently ignore.
    model_only = {
        "--max-model-len": max_model_len,
        "--gpu-memory-utilization": gpu_memory_utilization,
    }
    engine_offenders = [flag for flag, value in model_only.items() if value is not None]
    if engine_offenders:
        console.print(
            f"[red]{', '.join(engine_offenders)} only apply to model serves "
            "(--backend vllm), not app services. Drop the flag(s), or pass a "
            "model reference (name:tag) instead of a path.[/red]"
        )
        raise typer.Exit(1)

    # Merge: CLI flags override [deploy] defaults.
    effective_name = name or (deploy.name if deploy else None)
    if not effective_name:
        console.print(
            "[red]No service name (pass --name or set [deploy].name in nerdit.toml)[/red]"
        )
        raise typer.Exit(1)

    # Decision #1: register-only over a prebuilt image — image is mandatory.
    if not image:
        console.print(
            "[red]An image is required (register-only in P2). Pass --image <image>.[/red]"
        )
        raise typer.Exit(1)

    effective_port = port if port is not None else (deploy.port if deploy else None) or 8000
    effective_gpus = gpus if gpus is not None else (deploy.gpus if deploy else 0)
    effective_command = command or (deploy.start if deploy else None)
    effective_health = health_path or (deploy.health if deploy else None)

    health_check: dict | None = {"path": effective_health} if effective_health else None

    create_kwargs: dict = {
        "name": effective_name,
        "image": image,
        "port": effective_port,
        "gpus": effective_gpus,
        "command": effective_command,
        "script_path": script,
        "health_check": health_check,
        "idempotency_key": uuid4().hex,
    }
    if restart_policy:
        create_kwargs["restart_policy"] = restart_policy

    client = get_configured_client()
    try:
        service = await client.create_service(**create_kwargs)
        display_service_submitted(service)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc


async def _serve_model_async(
    model_ref: str,
    *,
    name: str | None,
    gpus: int | None,
    backend: str | None,
    image: str | None,
    port: int | None,
    restart_policy: str | None,
    command: str | None,
    script: str | None,
    health_path: str | None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> None:
    """Serve a model with shared name/GPU flags and vLLM engine bounds.

    Reject app-only flags. The daemon validates bound ranges and vLLM-only use.
    """
    from nerdit.cli.client import get_configured_client

    app_only = {
        "--image": image,
        "--port": port,
        "--restart-policy": restart_policy,
        "--command": command,
        "--script": script,
        "--health": health_path,
    }
    offenders = [flag for flag, value in app_only.items() if value is not None]
    if offenders:
        console.print(
            f"[red]{', '.join(offenders)} only apply to app services, but "
            f"'{model_ref}' is a model reference. Drop the flag(s) or point at "
            "an app directory.[/red]"
        )
        raise typer.Exit(1)

    client = get_configured_client()
    try:
        workload = await client.serve_model(
            model=model_ref,
            gpus=gpus if gpus is not None else 0,
            name=name,
            backend=backend,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    console.print(f"[green]Model serving:[/green] {workload.get('name')}")
    console.print(f"  Model:   {workload.get('model') or model_ref}")
    console.print(f"  Status:  {workload.get('status')}")
    console.print(
        "[dim]Watch it with `nerdit models list` — the endpoint appears once "
        "the model is pulled.[/dim]"
    )
