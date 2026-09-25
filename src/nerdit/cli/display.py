"""Rich display formatters for CLI output."""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import TypeVar

import httpx
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from nerdit.core.remediation_settle import parse_iso

console = Console()

T = TypeVar("T")


def plain(value: object, *, missing: str = "-") -> str:
    """Escape server-derived Rich text, preserving the caller's missing-value marker."""
    return missing if value is None else escape(str(value))


def _plain(value: object) -> str:
    """Display tables and errors use an empty missing-value marker."""
    return plain(value, missing="")


def fmt_bytes(value: object) -> str:
    """Format bytes; None means unknown and renders as `-`, not zero.

    Render non-numeric values as escaped text without inventing a size.
    """
    if value is None:
        return "-"
    if not isinstance(value, (int, float)):
        return _plain(value)
    num = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0 or unit == "TiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024.0
    return f"{num:.1f} TiB"  # pragma: no cover — the loop already returns


def _error_target(exc: Exception) -> str:
    """Best-effort 'host:port' suffix for a failed request, or ''.

    Escaped like every other interpolated value: the host is whatever
    `nerdit connect` was pointed at, so it is not ours to trust as markup.
    """
    request = getattr(exc, "request", None)
    if request is not None:
        url = request.url
        return f" at {_plain(url.host)}:{_plain(url.port)}"
    return ""


def _response_detail(response: httpx.Response) -> str:
    """Extract a human-readable detail from an error response body."""
    try:
        data = response.json()
    except Exception:
        return (response.text or "").strip() or "bad request"
    if isinstance(data, dict) and data.get("detail"):
        return str(data["detail"])
    return str(data)


def _error_envelope(response: httpx.Response) -> tuple[str, str, str]:
    """Return (message, hint, detail) from a structured error envelope.

    A structured envelope is `{code, message, hint?, detail?, ...}`. Returns
    `('', '', '')` for a non-envelope body (e.g. a bare `{detail: ...}` or
    FastAPI's validation list) so callers can fall back to the generic paths.
    """
    try:
        data = response.json()
    except Exception:
        return "", "", ""
    if isinstance(data, dict) and data.get("code") and data.get("message"):
        detail = data.get("detail")
        detail_str = detail if isinstance(detail, str) else ""
        return str(data["message"]), str(data.get("hint") or ""), detail_str
    return "", "", ""


# Per-status labels for a 4xx whose body carries a structured envelope.
_CLIENT_ERROR_LABELS = {
    403: "Forbidden",
    404: "Not found",
    409: "Conflict",
    422: "Invalid request",
}


def render_client_error(exc: Exception) -> None:
    """Render structured daemon errors before status-specific fallbacks.

    Preserve actionable hints for every status, including partially applied 5xx
    failures. Distinguish auth, connection and timeout failures. Escape all
    server-derived Rich text.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 401:
            console.print("[red]Authentication required.[/red] This daemon expects a token.")
            console.print(
                "[dim]Get it with `nerdit token` on the server, then "
                "`nerdit connect <host>` (it prompts for the token).[/dim]"
            )
            return

        message, hint, detail = _error_envelope(exc.response)
        if message:
            default_label = f"Daemon error ({code})" if code >= 500 else f"Request failed ({code})"
            label = _CLIENT_ERROR_LABELS.get(code, default_label)
            console.print(f"[red]{label}:[/red] {_plain(message)}")
            if detail and detail != message:
                console.print(f"[dim]{_plain(detail)}[/dim]")
            if hint:
                console.print(f"[dim]{_plain(hint)}[/dim]")
            return

        # No structured envelope — per-status fallbacks.
        if 400 <= code < 500:
            if code == 403:
                console.print("[red]Invalid or expired token.[/red]")
                console.print(
                    "[dim]Check it with `nerdit token` and reconnect with `nerdit connect`.[/dim]"
                )
            elif code == 404:
                console.print("[red]Not found.[/red] That job or resource does not exist.")
            elif code == 422:
                console.print(
                    f"[red]Invalid request:[/red] {_plain(_response_detail(exc.response))}"
                )
            else:
                console.print(f"[red]Request failed ({code}).[/red]")
            return

        if code >= 500:
            console.print(f"[red]Daemon error ({code}).[/red] Check the daemon logs.")
            return

        console.print(f"[red]Request failed ({code}).[/red]")
        return

    if isinstance(exc, httpx.ConnectError):
        console.print(f"[red]Cannot reach the daemon{_error_target(exc)}.[/red]")
        console.print(
            "[dim]Is it running? Start it with `nerdit init` (local) or check the host/port "
            "and that Docker is up on the server.[/dim]"
        )
        return

    if isinstance(exc, httpx.TimeoutException):
        console.print(f"[red]The daemon did not respond in time{_error_target(exc)}.[/red]")
        console.print("[dim]It may be busy or unreachable — try again.[/dim]")
        return

    if isinstance(exc, httpx.TransportError):
        console.print(f"[red]Network error talking to the daemon:[/red] {_plain(exc)}")
        return

    console.print(f"[red]Error:[/red] {_plain(exc)}")


async def call_or_exit(awaitable: Awaitable[T]) -> T:
    """Await a client call; on any failure render it and exit 1."""
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc


def _status_cell(status: str, style: str) -> str:
    """Style known statuses and escape their text; render unknown statuses without style tags."""
    return f"[{style}]{_plain(status)}[/{style}]" if style else _plain(status)


def display_gpu_table(gpus: list[dict]) -> None:
    """Display a Rich table of GPU information."""
    table = Table(title="GPUs")
    table.add_column("ID", style="dim", max_width=16)
    table.add_column("Vendor")
    table.add_column("Name", style="cyan")
    table.add_column("Memory", justify="right")
    table.add_column("Mode")
    table.add_column("Status", style="bold")
    table.add_column("Util %", justify="right")
    table.add_column("Temp °C", justify="right")

    for gpu in gpus:
        status_style = {
            "idle": "green",
            "busy": "yellow",
            "error": "red",
            "offline": "dim",
        }.get(gpu["status"], "")

        raw_util = gpu.get("utilization_percent")
        util = str(raw_util) if raw_util is not None else "-"
        raw_temp = gpu.get("temperature_c")
        temp = str(raw_temp) if raw_temp is not None else "-"

        gpu_id = gpu["id"] if len(gpu["id"]) <= 16 else gpu["id"][:13] + "..."
        table.add_row(
            _plain(gpu_id),
            _plain(str(gpu.get("vendor", "nvidia")).upper()),
            _plain(gpu["name"]),
            f"{_plain(gpu['memory_mb'])} MB",
            "scheduled" if gpu.get("schedulable", True) else "inventory",
            _status_cell(gpu["status"], status_style),
            util,
            temp,
        )

    console.print(table)


def display_service_table(services: list[dict]) -> None:
    """Display a Rich table of services (name, status, restarts, GPUs, URL)."""
    table = Table(title="Services")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Restarts", justify="right")
    table.add_column("GPUs", justify="right")
    table.add_column("URL")

    for svc in services:
        status_style = {
            "building": "yellow",
            "running": "green",
            "restarting": "magenta",
            "degraded": "yellow",
            "stopped": "dim",
            "failed": "red",
        }.get(svc["status"], "")

        endpoint = svc.get("endpoint") or {}
        host_port = endpoint.get("host_port")
        url = (
            endpoint.get("public_url")
            or endpoint.get("url")
            or (f"127.0.0.1:{host_port}" if host_port else "-")
        )

        table.add_row(
            _plain(svc.get("name", "")),
            _status_cell(svc["status"], status_style),
            _plain(svc.get("restart_count", 0)),
            _plain(svc.get("gpu_count", 0)),
            _plain(url),
        )

    console.print(table)


def display_model_table(models: list[dict]) -> None:
    """Display a Rich table of served models (P5: name, model, status, pulled, ...)."""
    table = Table(title="Models")
    table.add_column("Name", style="cyan")
    table.add_column("Model")
    table.add_column("Status", style="bold")
    table.add_column("Pulled", justify="center")
    table.add_column("GPUs", justify="right")
    table.add_column("Endpoint")
    table.add_column("Util%", justify="right")

    for mdl in models:
        status_style = {
            "building": "yellow",
            "running": "green",
            "restarting": "magenta",
            "degraded": "yellow",
            "stopped": "dim",
            "failed": "red",
        }.get(mdl["status"], "")

        pulled = "[green]yes[/green]" if mdl.get("model_pulled") else "[dim]no[/dim]"
        utils = [u for u in (mdl.get("gpu_utilization") or {}).values() if u is not None]
        util = ", ".join(str(u) for u in utils) if utils else "-"

        table.add_row(
            _plain(mdl.get("name", "")),
            _plain(mdl.get("model") or "-"),
            _status_cell(mdl["status"], status_style),
            pulled,
            _plain(mdl.get("gpu_count", 0)),
            _plain(mdl.get("endpoint") or "-"),
            _plain(util),
        )

    console.print(table)


def display_database_table(databases: list[dict]) -> None:
    """Display a Rich table of managed databases (P15: name, backend, status, ready, endpoint).

    The `Endpoint` column is the backend's password-free `host:port` display
    string by construction — the minted credential is never surfaced.
    """
    table = Table(title="Databases")
    table.add_column("Name", style="cyan")
    table.add_column("Backend")
    table.add_column("Status", style="bold")
    table.add_column("Ready", justify="center")
    table.add_column("Endpoint")

    for db in databases:
        status_style = {
            "building": "yellow",
            "running": "green",
            "restarting": "magenta",
            "degraded": "yellow",
            "stopped": "dim",
            "failed": "red",
        }.get(db["status"], "")

        ready = "[green]yes[/green]" if db.get("db_ready") else "[dim]no[/dim]"

        table.add_row(
            _plain(db.get("name", "")),
            _plain(db.get("backend") or "-"),
            _status_cell(db["status"], status_style),
            ready,
            _plain(db.get("endpoint") or "-"),
        )

    console.print(table)


def _token_expiry_state(expires_at: object) -> tuple[str, str]:
    """Return `(rendered expiry, state)` for a token's `expires_at`.

    `None` is "never" — the pre-P25 semantics every existing row keeps. An
    unparsable value is rendered bare rather than guessed at: the daemon is the
    clock of record, and inventing "active" for a string we cannot read would
    be the wrong direction to fail.
    """
    if expires_at is None:
        return "never", "active"
    parsed = parse_iso(str(expires_at))
    if parsed is None:
        return _plain(expires_at), "active"
    rendered = parsed.astimezone(UTC).strftime("%Y-%m-%d %H:%MZ")
    return rendered, "expired" if parsed <= datetime.now(UTC) else "active"


def display_token_table(tokens: list[dict]) -> None:
    """Display a Rich table of API tokens (P25: id, name, role, expiry, scope, state).

    Hashes and plaintext never appear in the payload, so nothing here can leak
    one. `State` is a client-side projection of `revoked` + `expires_at`:
    a revoked token reads `revoked` whether or not its clock also ran out.
    """
    table = Table(title="Tokens")
    table.add_column("ID", style="dim", max_width=16)
    table.add_column("Name", style="cyan")
    table.add_column("Role")
    table.add_column("Expires")
    table.add_column("Scope")
    table.add_column("State", style="bold")

    for tok in tokens:
        expires, state = _token_expiry_state(tok.get("expires_at"))
        if tok.get("revoked"):
            state = "revoked"
        scope = tok.get("scope_services")
        state_style = {"active": "green", "expired": "yellow", "revoked": "red"}.get(state, "")

        table.add_row(
            _plain(tok.get("id", "")),
            _plain(tok.get("name", "")),
            _plain(tok.get("role", "")),
            _plain(expires),
            _plain(", ".join(scope)) if scope else "-",
            _status_cell(state, state_style),
        )

    console.print(table)


# The window in which ``nerdit token whoami`` starts nagging. Seven days is the
# smallest horizon that survives a weekend and a public holiday — and after
# expiry the token cannot rotate itself (P25 D-P25-2 sub-ruling), so the warning
# IS the remediation path.
TOKEN_EXPIRY_WARN_S = 7 * 86_400


def display_token_self(token: dict) -> None:
    """Render `nerdit token whoami` — the caller's own token.

    Every value is the daemon's, so it goes through `_plain`. Nothing in
    the payload can be a secret (the self view carries no hash and no plaintext).
    """
    console.print(f"[bold]Token:[/bold] {_plain(token.get('id') or '(local/legacy admin)')}")
    console.print(f"  Name:   {_plain(token.get('name', ''))}")
    console.print(f"  Role:   {_plain(token.get('role', ''))}")
    scope = token.get("scope_services")
    console.print(f"  Scope:  {_plain(', '.join(scope)) if scope else 'all services'}")
    expires, _state = _token_expiry_state(token.get("expires_at"))
    console.print(f"  Expires: {_plain(expires)}")

    remaining = token.get("expires_in_s")
    if isinstance(remaining, int) and remaining < TOKEN_EXPIRY_WARN_S:
        days = remaining // 86_400
        window = f"{days}d" if days else f"{remaining // 3600}h"
        if token.get("rotatable"):
            console.print(
                f"[yellow]This token expires in {window}. Rotate it now with "
                "'nerdit token rotate' — once expired it cannot rotate itself and only "
                "an admin can issue a new one.[/yellow]"
            )
        else:
            console.print(
                f"[yellow]This token expires in {window} and cannot rotate itself. "
                "Ask an admin for a new one ('nerdit token create').[/yellow]"
            )


def display_service_submitted(service: dict) -> None:
    """Display a service registration, escaping every server-derived field."""
    console.print(f"[green]Service registered:[/green] {_plain(service.get('name'))}")
    if service.get("image"):
        console.print(f"  Image:   {_plain(service['image'])}")
    console.print(f"  GPUs:    {_plain(service.get('gpu_count', 0))}")
    console.print(f"  Status:  {_plain(service.get('status'))}")
    endpoint = service.get("endpoint") or {}
    public_url = endpoint.get("public_url")
    if public_url:
        console.print(f"  URL:     {_plain(public_url)}")
        if endpoint.get("url"):
            console.print(f"  Local:   {_plain(endpoint['url'])}")
    elif endpoint.get("url"):
        console.print(f"  URL:     {_plain(endpoint['url'])}")


def display_deploy_result(
    service: dict, *, heading: str = "Deploy accepted", build_hint: bool = True
) -> None:
    """Display deploy or rollback confirmation and optional advisory hints.

    Only the caller-owned heading is trusted; escape service fields and hint lines.
    """
    name = _plain(service.get("name"))
    console.print(f"[green]{heading}:[/green] {name}")
    console.print(f"  Status:  {_plain(service.get('status'))}")
    endpoint = service.get("endpoint") or {}
    url = endpoint.get("public_url") or endpoint.get("url")
    if url:
        console.print(f"  URL:     {_plain(url)}")
    for hint in service.get("hints") or []:
        console.print(f"[dim]  - {_plain(hint)}[/dim]")
    if build_hint:
        console.print(f"[dim]Watch the build with `nerdit logs {name}`.[/dim]")


def display_wait_outcome(result: dict) -> None:
    """Render a `/wait` outcome: converged / failed / timeout / superseded.

    `converged` prints green + the public URL; `failed` prints red with the
    reason, `error_class` and a `nerdit diagnose` hint; `timeout` and
    `superseded` print yellow. Never feeds the phase/outcome through the
    status-cell style maps (an unmapped value would crash Rich markup).
    """
    outcome = result.get("outcome")
    name = _plain(result.get("service_name")) or "service"
    if outcome == "converged":
        console.print(f"[green]Converged:[/green] {name} is healthy")
        url = result.get("public_url")
        if url:
            console.print(f"  URL:     {_plain(url)}")
        return
    if outcome == "timeout":
        waited = result.get("waited_s")
        phase = _plain(result.get("phase") or result.get("status"))
        console.print(
            f"[yellow]Timed out[/yellow] waiting for {name} to converge"
            + (f" (after {waited}s)" if waited is not None else "")
            + f". Phase: {phase}."
        )
        console.print(
            f"[dim]Still building? Re-run with a longer --timeout, or `nerdit logs {name}`.[/dim]"
        )
        return
    if outcome == "superseded":
        console.print(
            f"[yellow]Superseded:[/yellow] a newer deploy of {name} landed while waiting."
        )
        console.print(
            "[dim]The requested version's outcome is unknowable — re-check with "
            f"`nerdit diagnose {name}`.[/dim]"
        )
        return
    # failed
    console.print(f"[red]Failed:[/red] {name} did not converge")
    reason = result.get("reason")
    if reason:
        console.print(f"  Reason:  {_plain(reason)}")
    error_class = result.get("error_class")
    if error_class:
        console.print(f"  Class:   {_plain(error_class)}")
    error_message = result.get("error_message")
    if error_message:
        console.print(f"  Detail:  {_plain(error_message)}")
    console.print(f"[dim]Diagnose the failure with `nerdit diagnose {name}`.[/dim]")


def wait_exit_code(result: dict) -> int:
    """Map a /wait outcome to the CLI exit contract.

    0 converged · 3 timeout · 1 failed|superseded (and any unknown
    outcome — never 2, Typer owns 2 for usage errors).
    """
    outcome = result.get("outcome")
    if outcome == "converged":
        return 0
    if outcome == "timeout":
        return 3
    return 1


def display_logs(logs: list[dict]) -> None:
    """Display logs with every field escaped so bracketed build and runtime output stays literal."""
    for log in logs:
        stream = log.get("stream", "stdout")
        message = _plain(log.get("message", ""))
        ts = _plain(log.get("timestamp", ""))[:19]

        if stream == "stderr":
            console.print(f"[dim]{ts}[/dim] [red]{message}[/red]")
        elif stream == "system":
            console.print(f"[dim]{ts}[/dim] [blue]{message}[/blue]")
        else:
            console.print(f"[dim]{ts}[/dim] {message}")
