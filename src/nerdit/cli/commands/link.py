"""Claim, inspect, refresh or revoke the daemon's cloud link.

The daemon owns cloud requests, node identity and config writes. Console codes
travel in argv or a hidden prompt; pre-auth keys use stdin or a hidden prompt;
secret device codes never leave daemon memory. Never log, persist or echo grants,
including in errors. API URLs are per-invocation; relay URLs persist, preserving
an existing relay when no override is supplied.

Bare `link` is read-only status. Positional refresh/code and the mutually
exclusive device/key-stdin modes perform writes. Flag modes treat an existing
link, including a race-safe 409, as success. After polling stops, later approval
cannot complete this flow; timeout and interruption copy must say so.

Restart the matching local service unit to apply changed link config. Remote
daemons are never restarted locally; system units use noninteractive sudo when
needed. With no usable unit, print instructions. Unlink drops the tunnel
immediately. Status distinguishes stored config from live state.
Escape all server-derived Rich text.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NoReturn, Optional
from uuid import uuid4

import httpx
import typer
from rich.live import Live
from rich.markup import escape
from rich.text import Text

from nerdit.cli.display import console, render_client_error
from nerdit.cli.display import plain as _plain
from nerdit.utils.install_layout import (
    ServiceUnit,
    detect_service_unit,
    managed_daemon_port,
)

#: (P30 D-P30-10) Production endpoints, applied on the CLAIM path only and only
#: when the corresponding flag is unset — an explicit ``--api-url`` /
#: ``--relay-url`` (dev, e2e, a self-hosted control plane) is never overridden.
DEFAULT_LINK_API_URL = "https://app.nerdit.ai"
DEFAULT_LINK_RELAY_URL = "wss://relay.nerdit.ai/v1/connect"

#: (D-P34-2) Wall-clock budget for the whole device-approval wait. Deliberately
#: equal to ``install.sh --link-timeout`` and BELOW the cloud's 900 s code TTL,
#: so this terminal always reaches a verdict while the code it printed is still
#: alive — a poller that outlives the code could only ever report "expired".
DEFAULT_LINK_DEVICE_TIMEOUT_S = 600

#: (D-X16-O4 as amended / D-X16-O22) How much of the credential fingerprint is
#: rendered. The console's approval screen truncates the pinned
#: ``device_credential_reference`` to the same eight characters, so the two
#: sides are comparable by eye. It is a cross-check for the operator sitting at
#: this terminal, NOT an anti-phishing control — a phished victim has no
#: out-of-band reference to compare against, and the copy must never imply one.
_FINGERPRINT_PREVIEW_LENGTH = 8

#: (D-P34-5) Device-poll refusals whose own wording is better than the
#: envelope's, mapped one code → one line. The cloud *body* is never echoed
#: (D11): everything below is written here, keyed on the daemon's machine code.
_DEVICE_TERMINAL_MESSAGES: dict[str, str] = {
    "link.device_denied": "[red]The request was denied in the console.[/red]",
    "link.device_expired": (
        "[yellow]The code expired before it was approved — "
        "run 'nerdit link --device' again.[/yellow]"
    ),
    "link.device_superseded": (
        "[yellow]Another 'nerdit link --device' replaced this request — this terminal's "
        "code is dead; the newer terminal owns the flow.[/yellow]"
    ),
}

#: Terminal device refusals whose route-supplied hint IS the guidance (the
#: daemon knows whether the slot was never started, expired locally, or lost
#: its credential; restating that here would drift from the route).
_DEVICE_ENVELOPE_CODES = frozenset(
    {
        "link.device_invalid",
        "link.device_credential_mismatch",
        "link.device_not_started",
    }
)

#: (D-P34-5) NOT terminal: the cloud is busy or briefly unreachable and the
#: pending row is untouched, so the poller keeps going until ``--timeout``.
#: Only the *start* hop failing is immediately fatal — there is no session to
#: come back to.
_DEVICE_RETRYABLE_CODES = frozenset(
    {
        "link.claim_rate_limited",
        "link.cloud_unreachable",
        "link.cloud_error",
    }
)

#: The "this terminal has stopped listening" lines. Each must name the resume
#: command and each must deny that a later approval links this machine — see the
#: module docstring for why a softer wording produces an orphan node.
_DEVICE_TIMEOUT_MESSAGE = (
    "[yellow]Stopped polling after {seconds}s. Approving the code now will no longer "
    "link this machine — run 'nerdit link --device' to mint a new one.[/yellow]"
)
_DEVICE_INTERRUPT_MESSAGE = (
    "\n[dim]Stopped waiting — approving this code will no longer link this machine; "
    "run 'nerdit link --device' to start again.[/dim]"
)
#: The third way the poller stops listening: a refusal it has no mapping for.
#: The envelope render explains WHAT went wrong; this line is the part only this
#: module knows — that the pending code outlived the terminal watching it, so an
#: operator who wanders to the console and approves it produces an orphan node
#: rather than the link they were reaching for.
_DEVICE_ABANDONED_MESSAGE = (
    "[yellow]Stopped polling — approving the code now will no longer link this machine; "
    "run 'nerdit link --device' to start again.[/yellow]"
)

#: A CLI→daemon TRANSPORT failure (no HTTP response at all: the daemon was
#: restarting, the socket dropped, the reverse proxy blinked) says nothing about
#: the cloud row, which is still pending and still approvable. Retrying is the
#: honest response — but silently, once-announced, because a notice per poll
#: would bury the countdown under a wall of identical lines.
_DEVICE_TRANSPORT_RETRY_MESSAGE = (
    "[dim]Cannot reach the daemon right now — still waiting for approval.[/dim]"
)

#: Printed only when a device flow this run MINTED a code then learned, from the
#: poll, that the node was already linked by something else (the D-P34-3
#: lost-response recovery, or a second flow winning the race). The link is a
#: success and exits 0 — but the code this terminal printed is still live in the
#: cloud until it expires, and approving it now would enroll a node nobody asked
#: for. The pre-start short-circuit never needs this: it mints nothing.
_DEVICE_PENDING_APPROVAL_WARNING = (
    "[yellow]The code this terminal printed is still live until it expires — do not "
    "approve it in the console; approving it would add an orphan node to the "
    "account.[/yellow]"
)

#: Upper bound on the interval the CLI will back off to on a ``429``. The cloud
#: advertises its own cadence via ``slow_down``; this only bounds the *local*
#: widening so a rate-limited poller still gives a verdict inside --timeout.
_DEVICE_MAX_INTERVAL_S = 30.0

#: How long the post-claim service restart may take before we fall back to
#: printing the manual command. The claim itself already succeeded either way.
_RESTART_TIMEOUT_S = 30

#: Hosts that mean "the daemon on this machine". Only then may a local service
#: unit be restarted — see `_restart_for_tunnel`.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})

if TYPE_CHECKING:  # pragma: no cover — typing only
    from nerdit.cli.client import NerditClient


def _entitlement_line(live: dict) -> str | None:
    """Render entitlement with assertion age, distinguishing missing and expired leases.

    Return None if an older daemon omits the capability. Leases expire after
    24 hours; read machine fields rather than rendered text.
    """
    if "hosted_public_entitled" not in live:
        return None
    entitled = bool(live.get("hosted_public_entitled"))
    asserted_at = live.get("hosted_public_entitled_at")
    if not isinstance(asserted_at, str) or not asserted_at:
        # Either an older daemon with no ``_at`` field, or one the cloud has
        # never asserted for. Both are honestly rendered as "no detail".
        return "yes" if entitled else "no (never asserted)"
    age = _age_phrase(asserted_at)
    if age is None:
        return "yes" if entitled else "no"
    if entitled:
        return f"yes (asserted {age} ago)"
    if _older_than_ttl(asserted_at):
        # The cloud went quiet: the lease ran out. The next push restores it.
        return f"no (asserted {age} ago, expired)"
    # A fresh ``false``: the cloud looked and said no (account not Pro).
    return f"no (asserted {age} ago)"


def _github_line(live: dict) -> str | None:
    """Render installation count and soonest expiry from secret-free capabilities.

    Return None for older daemons. With zero installations, distinguish a down
    tunnel from a connected node missing the GitHub App. Never print tokens or repos.
    """
    if "github_installations" not in live:
        return None
    installations = live.get("github_installations")
    if not isinstance(installations, list) or not installations:
        if live.get("state") == "connected":
            return "connected on Nerdit Cloud, 0 installation(s) mirrored"
        return "none"
    count = len(installations)
    noun = "installation" if count == 1 else "installations"
    soonest = min(
        (
            parsed
            for inst in installations
            if isinstance(inst, dict)
            and (parsed := _parse_stamp(inst.get("expires_at"))) is not None
        ),
        default=None,
    )
    if soonest is None:
        return f"{count} {noun}"
    remaining = (soonest - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        return f"{count} {noun}, expired"
    return f"{count} {noun}, expires in {_duration_phrase(remaining)}"


def _parse_stamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _duration_phrase(seconds: float) -> str:
    """`"42m"` / `"3h"` / `"2d"`; seconds below a minute round up to `1m`."""
    if seconds < 3600:
        return f"{max(1, int(seconds // 60))}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _older_than_ttl(iso_timestamp: str) -> bool:
    """True when the assertion predates the daemon's 24 h lease window."""
    from nerdit.core.link.manager import ENTITLEMENT_TTL_S

    try:
        parsed = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (datetime.now(UTC) - parsed).total_seconds() >= ENTITLEMENT_TTL_S


def _mcp_remedy(reason: str) -> str:
    """Return the remedy for the machine-coded transport refusal, including missing
    prerequisites.
    """
    code = reason.partition(":")[0]
    if code == "no_auth_token":
        return "Mint one first: nerdit init --auth-token-only, then nerdit daemon restart."
    if code == "mcp_extra_missing":
        return (
            "This build has no MCP transport; install a release bundle (0.5.3+) "
            "or pip install 'nerdit\\[mcp]' on a source install."
        )
    return "Once that is fixed: nerdit config set mcp http_enabled=true"


def _age_phrase(iso_timestamp: str) -> str | None:
    """`"12m"` / `"3h"` / `"2d"` for an ISO-8601 stamp, or `None`.

    Coarse on purpose: the reader wants "recent" vs "stale", and a seconds-
    precise age on a value the cloud re-asserts every few minutes is noise.
    """
    try:
        parsed = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - parsed).total_seconds()
    if seconds < 0:
        # A clock skew between daemon and CLI host; "just now" beats "-3m".
        return "0m"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _section_values(view: Any) -> dict:
    """Unwrap a section view or view list; return an empty mapping for unexpected shapes."""
    if isinstance(view, list):
        view = next((item for item in view if isinstance(item, dict)), {})
    if not isinstance(view, dict):
        return {}
    values = view.get("values")
    return values if isinstance(values, dict) else {}


def link(  # noqa: PLR0913 — one Typer parameter per documented mode/flag of the verb
    code: Optional[str] = typer.Argument(  # noqa: UP007 — Typer needs Optional[]
        None,
        help=(
            "Link code from the cloud console. Omit it while passing --api-url "
            "to be prompted instead (keeps the code out of shell history); "
            "omit everything to show link status. The literal 'refresh' re-reads "
            "the cloud's hosted domain instead of claiming. To link without a "
            "console code, use --device (approve in a browser) or --key-stdin "
            "(a pre-auth key on stdin) — neither takes a positional code."
        ),
    ),
    api_url: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--api-url",
        help=(
            "Cloud API origin to claim against. Defaults to "
            "https://app.nerdit.ai. Used for this claim only — never stored."
        ),
    ),
    relay_url: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--relay-url",
        # ``[link]`` is a Rich markup tag and Typer renders help through Rich,
        # so the section name is backslash-escaped or it renders as nothing.
        help=(
            r"Relay endpoint to persist as \[link].relay_url. Omit to keep the "
            "one already configured; on a node with none, defaults to "
            "wss://relay.nerdit.ai/v1/connect."
        ),
    ),
    enable: bool = typer.Option(
        True,
        "--enable/--no-enable",
        help=r"Also set \[link].enabled = true in the same write.",
    ),
    device: bool = typer.Option(
        False,
        "--device",
        help=(
            "Link by approving this node in a browser — no code to copy. Prints "
            "a one-click URL and waits for the approval."
        ),
    ),
    key_stdin: bool = typer.Option(
        False,
        "--key-stdin",
        help=(
            "Link with a pre-auth key read from stdin (never an option value, so "
            "it stays off argv). Falls back to a hidden prompt on a terminal."
        ),
    ),
    timeout: int = typer.Option(
        DEFAULT_LINK_DEVICE_TIMEOUT_S,
        "--timeout",
        help="Seconds to wait for a --device approval before giving up (exit 3).",
    ),
) -> None:
    """Link this daemon (code, --device or --key-stdin), refresh metadata, or show status."""
    # (D-P34-2) Mode conflicts are usage errors raised BEFORE any client is
    # built, so a mistyped invocation never mints a cloud row, never reads
    # config, and never spends a code. ``typer.BadParameter`` exits 2, the
    # house code for "you typed it wrong" — distinct from the 1/3 verdicts the
    # flows themselves return.
    if device and key_stdin:
        raise typer.BadParameter("--device and --key-stdin cannot be combined.")
    if (device or key_stdin) and code is not None:
        flag = "--device" if device else "--key-stdin"
        raise typer.BadParameter(f"{flag} links without a code — drop the positional argument.")

    # (P34 D1) The two flag modes are dispatched HERE rather than inside
    # ``_link_async``: they take no positional argument at all, so threading
    # them through the code-grammar driver would only give that driver two
    # parameters it must immediately branch away from.
    if key_stdin:
        asyncio.run(_key_stdin_entry(api_url, relay_url, enable))
        return
    if not device:
        asyncio.run(_link_async(code, api_url, relay_url, enable))
        return

    try:
        asyncio.run(_device_entry(api_url, relay_url, enable, timeout))
    except KeyboardInterrupt:
        # (D-P34-5) Caught around the OUTERMOST ``asyncio.run`` — the
        # ``cli/commands/dev.py`` / ``events.py`` pattern — never inside the
        # async body, where a cancelled task would race the loop teardown into
        # a traceback. Only the device flow parks on a human, so only it owns
        # the interrupt; the other modes are short calls and keep the default
        # behaviour.
        console.print(_DEVICE_INTERRUPT_MESSAGE)
        raise typer.Exit(1) from None


async def _device_entry(
    api_url: str | None, relay_url: str | None, enable: bool, timeout: int
) -> None:
    """Build the client, then run the device flow (the `--device` entry point)."""
    from nerdit.cli.client import get_configured_client

    await _device_async(get_configured_client(), api_url, relay_url, enable, timeout)


async def _key_stdin_entry(api_url: str | None, relay_url: str | None, enable: bool) -> None:
    """Build the client, then run the pre-auth claim (the `--key-stdin` entry point)."""
    from nerdit.cli.client import get_configured_client

    await _key_stdin_async(get_configured_client(), api_url, relay_url, enable)


async def _link_async(
    code: str | None,
    api_url: str | None,
    relay_url: str | None,
    enable: bool,
) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()

    # (P26 WP-H, S13) The refresh mode, checked BEFORE the claim-intent gate so
    # ``nerdit link refresh --api-url …`` never falls through to the hidden
    # code prompt. Case-insensitive because an operator retyping a documented
    # word should not have to match its case.
    if code is not None and code.strip().lower() == "refresh":
        await _refresh_async(client, api_url or DEFAULT_LINK_API_URL)
        return

    if code is None:
        # Claim options imply a hidden prompt, avoiding argv/history exposure.
        # Default --enable cannot signal intent; URL options and --no-enable can.
        if api_url is not None or relay_url is not None or not enable:
            code = typer.prompt("Link code", hide_input=True).strip()
            if not code:
                console.print("[red]No link code entered.[/red]")
                raise typer.Exit(1)
        else:
            await _render_status(client)
            return

    # (P30 D-P30-10) Production defaults, applied HERE and not as Typer option
    # defaults: the claim-intent gate above keys on "was the flag passed", so a
    # bare ``nerdit link`` must still see ``None`` and render status.
    api_url = api_url or DEFAULT_LINK_API_URL
    if relay_url is None:
        # The relay IS persisted, so defaulting it unconditionally would turn
        # "omit to keep the configured relay" into "omit to repoint a
        # self-hosted node at the production relay" on every re-claim. The
        # default therefore only fills a genuinely empty seam.
        relay_url = await _configured_relay_url(client) or DEFAULT_LINK_RELAY_URL

    # Dispatch recognizable nk_ grants into the key field, including hidden-prompt
    # input. The code validator rejects that alphabet; never send a key as a code.
    grant: dict[str, str] = {"key": code} if code.startswith("nk_") else {"code": code}
    try:
        result = await client.claim_link(
            **grant,
            api_url=api_url,
            relay_url=relay_url,
            enable=enable,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # render_client_error prints the daemon's envelope (message + hint +
        # the cloud_code extra). The code VALUE is not in that envelope by
        # construction, and must never be added here.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    _render_claim_result(result)


def _render_claim_result(result: dict) -> None:
    """Render a successful code or pre-auth claim, then start the tunnel."""
    console.print(
        f"[green]Linked as [bold]{_plain(result.get('slug'))}[/bold] "
        f"(node {_plain(result.get('node_id'))}).[/green]"
    )
    console.print(f"[dim]Verifier fingerprint: {_plain(result.get('verifier_fingerprint'))}[/dim]")
    console.print(f"Relay: {_plain(result.get('relay_url'))}")
    if result.get("mcp_http_enabled"):
        # Escaped: ``[mcp]`` is valid Rich markup, so the unescaped form printed
        # "Remote MCP: enabled (.http_enabled — …)" with the section name eaten.
        # Pre-existing, but this line now also serves the ``--key-stdin`` success
        # path, and its sibling ``_render_device_linked`` already escapes it.
        console.print(
            r"Remote MCP: enabled (\[mcp].http_enabled — serves mcp.nerdit.ai after restart)"
        )
    elif result.get("enabled") and result.get("mcp_skipped_reason"):
        reason = str(result.get("mcp_skipped_reason"))
        console.print(
            f"[yellow]Remote MCP: not enabled — {_plain(reason.partition(': ')[2] or reason)}. "
            f"{_mcp_remedy(reason)}[/yellow]"
        )
    if result.get("enabled"):
        # Only worth bouncing the service when the restart can actually bring
        # the tunnel up. With [link].enabled still false it would stop and
        # start the daemon for nothing, then print "now flip it and restart".
        _restart_for_tunnel()
    else:
        # ``[link]`` is a real Rich markup tag, so the section name is escaped
        # rather than written raw (the P6 MarkupError lesson, from our own text
        # this time rather than the server's).
        console.print(
            r"[yellow]\[link].enabled is still false — flip it and restart to bring "
            "the tunnel up.[/yellow]"
        )


# -- the device flow and the pre-auth key (P34 D1) ----------------------------------


def _envelope_code(exc: Exception) -> str:
    """Return the structured error code, or an empty string; never infer it from text."""
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 — a non-JSON error body is simply not an envelope
        return ""
    if not isinstance(body, dict):
        return ""
    code = body.get("code")
    return str(code) if code else ""


def _retry_after(exc: Exception) -> float | None:
    """Return Retry-After seconds, or None for a missing or unreadable header."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        seconds = float(str(headers.get("retry-after")))
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _positive_float(value: object, fallback: float) -> float:
    """Return a positive server-supplied number, otherwise keep the fallback cadence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value) if value > 0 else fallback


def _monotonic() -> float:
    """Seam over `time.monotonic` so the poll loop's clock can be pinned in tests."""
    return time.monotonic()


async def _poll_sleep(seconds: float) -> None:
    """Seam over `asyncio.sleep` — the one place the poll loop actually waits."""
    await asyncio.sleep(seconds)


async def _linked_slug(client: NerditClient) -> str | None:
    """Read the persisted slug as an advisory already-linked check.

    Return None for failed or malformed reads; the daemon's 409 remains authoritative.
    """
    try:
        values = _section_values(await client.get_config("link"))
    except Exception:  # noqa: BLE001 — advisory read, never fatal
        return None
    node_id = values.get("node_id")
    if not node_id:
        return None
    slug = values.get("slug")
    # The slug is what an operator recognises in the console; the node id is
    # the honest fallback for a row that carries no slug, and beats a bare "-".
    return str(slug) if slug else str(node_id)


def _print_already_linked(name: str) -> None:
    """Print plain already-linked success for rerunnable installers and automation."""
    console.print(f"Already linked as {_plain(name)}.")


def _read_link_key() -> str:
    """Read a pre-auth key from stdin or a hidden terminal prompt.

    Strip surrounding whitespace only; preserve case and separators for cloud
    validation. Never accept the key in argv or echo it into terminal scrollback.
    """
    if sys.stdin is not None and sys.stdin.isatty():
        return str(typer.prompt("Link key", hide_input=True)).strip()
    line = sys.stdin.readline() if sys.stdin is not None else ""
    return line.strip()


async def _key_stdin_async(
    client: NerditClient,
    api_url: str | None,
    relay_url: str | None,
    enable: bool,
) -> None:
    """Claim using the dedicated key field, never the console-code field.

    Preserve the daemon's uniform key refusal; distinguishing unknown, expired or
    revoked keys locally would reopen its deliberately closed validity oracle.
    """
    already = await _linked_slug(client)
    if already is not None:
        _print_already_linked(already)
        return

    key = _read_link_key()
    if not key:
        console.print("[red]No link key entered.[/red]")
        raise typer.Exit(1)

    api_url = api_url or DEFAULT_LINK_API_URL
    if relay_url is None:
        # Same rule as the claim: the relay IS persisted, so the production
        # default may only fill a genuinely empty seam — never repoint a node
        # that already dials a self-hosted relay.
        relay_url = await _configured_relay_url(client) or DEFAULT_LINK_RELAY_URL

    try:
        result = await client.claim_link(
            key=key,
            api_url=api_url,
            relay_url=relay_url,
            enable=enable,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        if _envelope_code(exc) == "link.already_linked":
            # (D-P34-3) The race the short-circuit above cannot close: another
            # flow linked this node between the read and the claim. Success.
            _print_already_linked(await _linked_slug(client) or "this node")
            return
        # The key VALUE is not in the daemon's envelope by construction, and
        # must never be added here — not even to say which key was refused.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    _render_claim_result(result)


async def _device_async(
    client: NerditClient,
    api_url: str | None,
    relay_url: str | None,
    enable: bool,
    timeout: int,
) -> None:
    """Read link state, start browser approval and poll through the daemon.

    The CLI receives display data and an opaque session selector only; secret
    device codes stay daemon-side.
    """
    already = await _linked_slug(client)
    if already is not None:
        _print_already_linked(already)
        return

    api_url = api_url or DEFAULT_LINK_API_URL
    if relay_url is None:
        relay_url = await _configured_relay_url(client) or DEFAULT_LINK_RELAY_URL

    try:
        start = await client.start_device_link(
            api_url=api_url,
            relay_url=relay_url,
            enable=enable,
            idempotency_key=uuid4().hex,
        )
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        if _envelope_code(exc) == "link.already_linked":
            _print_already_linked(await _linked_slug(client) or "this node")
            return
        # A failed START is immediately terminal, unlike a failed poll: there
        # is no session to come back to, so retrying here would only mint a
        # second cloud row for the same terminal.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    _render_device_prompt(start)
    result = await _poll_for_approval(client, start, timeout)
    if result is not None:
        _render_device_linked(result)


def _machine_line(start: dict) -> str:
    """Render machine, version and platform from daemon facts, including remote daemons."""
    return (
        f"{_plain(start.get('hostname'))} (nerditd {_plain(start.get('daemon_version'))}, "
        f"{_plain(start.get('os'))})"
    )


def _render_device_prompt(start: dict) -> None:
    """Display the approval URL, user code, machine and credential comparison.

    The URL carries the code in its fragment, not requests or Referer. Preserve
    cloud grouping and normalization. Credential comparison is a terminal-to-screen
    check, not phishing protection; the copy must not claim otherwise.
    """
    uri = start.get("verification_uri_complete") or start.get("verification_uri")
    fingerprint = str(start.get("credential_fingerprint") or "")
    console.print()
    console.print("Approve this node in your browser:")
    console.print()
    console.print(f"    [bold]{_plain(uri)}[/bold]")
    console.print()
    console.print(f"  code         [bold]{_plain(start.get('user_code'))}[/bold]")
    console.print(f"  machine      {_machine_line(start)}")
    console.print(
        f"  credential   {_plain(fingerprint[:_FINGERPRINT_PREVIEW_LENGTH])}"
        f"   (the console will show the same {_FINGERPRINT_PREVIEW_LENGTH} characters)"
    )
    console.print()


def _restart_key_label(key: object) -> str:
    """Render a dotted config key as Rich-escaped TOML section notation."""
    text = str(key)
    section, dot, rest = text.partition(".")
    return escape(f"[{section}].{rest}" if dot else text)


def _render_device_linked(result: dict) -> None:
    """Render the approved link and restart only when link.enabled is true."""
    console.print()
    console.print(
        f"[green]Linked as node {_plain(result.get('node_id'))} "
        f"(slug {_plain(result.get('slug'))}).[/green]"
    )
    if result.get("nodes_base_domain"):
        console.print(f"  hosted domain  {_plain(result.get('nodes_base_domain'))}")
    console.print(f"  relay          {_plain(result.get('relay_url'))}")
    if result.get("mcp_http_enabled"):
        console.print(r"  \[mcp].http_enabled staged on")
    elif result.get("enabled") and result.get("mcp_skipped_reason"):
        reason = str(result.get("mcp_skipped_reason"))
        console.print(
            f"[yellow]  Remote MCP: not enabled — "
            f"{_plain(reason.partition(': ')[2] or reason)}. {_mcp_remedy(reason)}[/yellow]"
        )
    keys = result.get("restart_keys")
    if isinstance(keys, list) and keys:
        console.print("  restart required: " + ", ".join(_restart_key_label(k) for k in keys))
    if result.get("enabled"):
        _restart_for_tunnel()
    else:
        console.print(
            r"[yellow]\[link].enabled is still false — flip it and restart to bring "
            "the tunnel up.[/yellow]"
        )


def _waiting_line(remaining_s: float) -> str:
    """Render remaining time as mm:ss, clamped at zero, with the interruption hint."""
    minutes, seconds = divmod(int(max(0.0, remaining_s)), 60)
    return f"Waiting for approval… {minutes:02d}:{seconds:02d} left  (Ctrl-C stops waiting)"


class _ApprovalCountdown:
    """Display a live terminal countdown or one static line when output is redirected.

    Keep the waiting line after completion so the wait remains visible in history.
    """

    def __init__(self) -> None:
        self._live: Live | None = None
        self._printed_static = False

    @property
    def live(self) -> bool:
        """True while a Live region is running — i.e. ticking is worth doing."""
        return self._live is not None

    def start(self) -> None:
        if console.is_terminal:
            self._live = Live("", console=console, refresh_per_second=4, transient=False)
            self._live.start()

    def tick(self, remaining_s: float) -> None:
        text = _waiting_line(remaining_s)
        if self._live is not None:
            self._live.update(Text(text, style="dim"))
        elif not self._printed_static:
            self._printed_static = True
            console.print(f"[dim]{text}[/dim]")

    def stop(self) -> None:
        """Stop the live region idempotently before printing a terminal verdict."""
        if self._live is not None:
            self._live.stop()
            self._live = None


async def _sleep_with_countdown(
    countdown: _ApprovalCountdown, seconds: float, deadline: float
) -> None:
    """Sleep between polls, refreshing terminal countdowns; redirected output sleeps once."""
    if not countdown.live:
        await _poll_sleep(seconds)
        return
    end = _monotonic() + seconds
    while True:
        left = end - _monotonic()
        if left <= 0:
            return
        countdown.tick(deadline - _monotonic())
        await _poll_sleep(min(0.5, left))


def _give_up_waiting(countdown: _ApprovalCountdown, timeout: int) -> NoReturn:
    """Report timeout consistently: approval after this poller stops cannot link the node."""
    countdown.stop()
    console.print(_DEVICE_TIMEOUT_MESSAGE.format(seconds=int(timeout)))
    raise typer.Exit(3)


def _is_transport_failure(exc: Exception, code: str) -> bool:
    """Return whether a poll failed before receiving an HTTP response.

    Retry only httpx.TransportError. Decoding failures, unreadable responses and
    non-httpx exceptions require stopping rather than concealing a real error.
    """
    return not code and isinstance(exc, httpx.TransportError)


def _announce_transport_failure(already_shown: bool) -> bool:
    """Announce an unreachable daemon once; return the updated announcement flag."""
    if not already_shown:
        console.print(_DEVICE_TRANSPORT_RETRY_MESSAGE)
    return True


def _refuse_poll(countdown: _ApprovalCountdown, exc: Exception, code: str) -> NoReturn:
    """Render a terminal refusal and exit 1.

    Known codes get their specific message. Generic errors also warn that this
    terminal has stopped polling and later approval cannot complete the flow.
    """
    countdown.stop()
    if code in _DEVICE_TERMINAL_MESSAGES:
        console.print(_DEVICE_TERMINAL_MESSAGES[code])
    else:
        render_client_error(exc)
        console.print(_DEVICE_ABANDONED_MESSAGE)
    raise typer.Exit(1) from exc


async def _poll_for_approval(client: NerditClient, start: dict, timeout: int) -> dict | None:
    """Poll until linked, refused or timed out, respecting the advertised cadence.

    Adopt slow_down intervals and widen on rate limits; transient transport/cloud
    failures retry within the budget. Exit codes: 0 linked/already linked,
    1 terminal failure, 3 local timeout.

    Returns:
        The linked payload, or None when already-linked success was rendered.
    """
    session = str(start.get("session") or "")
    interval = _positive_float(start.get("interval"), 5.0)
    expires_in = _positive_float(start.get("expires_in"), 0.0)
    began = _monotonic()
    poll_deadline = began + max(1.0, float(timeout))
    # (D-P34-4/D-P34-5) The countdown runs from the code's own ``expires_in``,
    # capped by this terminal's budget — with the shipped defaults (600 s
    # against a 900 s TTL) the budget is the earlier of the two, and a
    # countdown that outlived the poller would be promising a wait that is not
    # going to happen.
    deadline = min(poll_deadline, began + expires_in) if expires_in > 0 else poll_deadline

    countdown = _ApprovalCountdown()
    countdown.start()
    # One notice per run, not per poll: a daemon that is down usually stays down
    # for several intervals, and repeating the line would push the countdown —
    # the thing an operator is actually reading — off the screen.
    transport_notice_shown = False
    try:
        while True:
            countdown.tick(deadline - _monotonic())
            # Bound each request by the remaining budget, not just the interval between polls.
            # Otherwise a slow final hop can exceed the advertised wall-clock timeout.
            budget = poll_deadline - _monotonic()
            if budget <= 0:
                _give_up_waiting(countdown, timeout)
            try:
                answer = await asyncio.wait_for(
                    client.poll_device_link(session=session, idempotency_key=uuid4().hex),
                    timeout=budget,
                )
            except TimeoutError:
                _give_up_waiting(countdown, timeout)
            except Exception as exc:  # noqa: BLE001 — mapped below, never re-raised raw
                code = _envelope_code(exc)
                if code == "link.already_linked":
                    # The lost-response recovery: the commit branch cleared the
                    # daemon's slot, so a retried poll after a link that DID
                    # succeed lands here. Success, and the same line as the
                    # short-circuit — an unconditional local "success" would
                    # instead mislabel a node linked by a different flow.
                    countdown.stop()
                    _print_already_linked(await _linked_slug(client) or "this node")
                    # Unlike the pre-start short-circuit, this run already
                    # minted a code and printed it. The link is a success, but
                    # that code outlives this verdict and approving it would
                    # enroll a node nobody is waiting on.
                    console.print(_DEVICE_PENDING_APPROVAL_WARNING)
                    return None
                if _is_transport_failure(exc, code):
                    # No response at all: the failure is between this CLI and
                    # its daemon (connect/read/write error), not a verdict from
                    # anyone. The cloud row is untouched and still approvable,
                    # so this is the most retryable failure there is — bounded,
                    # like every other one, by the outer --timeout budget, and
                    # at an unchanged cadence because nobody rate-limited us.
                    transport_notice_shown = _announce_transport_failure(transport_notice_shown)
                elif code in _DEVICE_RETRYABLE_CODES and code not in _DEVICE_ENVELOPE_CODES:
                    interval = min(_retry_after(exc) or interval * 2, _DEVICE_MAX_INTERVAL_S)
                else:
                    _refuse_poll(countdown, exc, code)
            else:
                if str(answer.get("status") or "") == "linked":
                    countdown.stop()
                    return answer
                # ``pending`` and ``slow_down`` differ only in the interval they
                # carry, and both mean "the human has not decided yet".
                interval = _positive_float(answer.get("interval"), interval)

            remaining = poll_deadline - _monotonic()
            if remaining <= 0:
                _give_up_waiting(countdown, timeout)
            await _sleep_with_countdown(countdown, min(interval, remaining), deadline)
    finally:
        # Belt and braces for the interrupt path: Ctrl-C unwinds through here
        # before the outer handler prints, and a Live region left running would
        # eat that line.
        countdown.stop()


async def _refresh_async(client: NerditClient, api_url: str) -> None:
    """Refresh hosted metadata through the daemon; restart only if it changed."""
    try:
        result = await client.refresh_link(api_url=api_url, idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        # A pre-P26 cloud answers the metadata read with a 404, which the daemon
        # turns into a structured ``link.refresh_unsupported`` carrying its own
        # hint — printing the envelope is the whole handling.
        render_client_error(exc)
        raise typer.Exit(1) from exc

    domain = result.get("nodes_base_domain")
    console.print(f"Hosted domain: {_plain(domain)}")
    if result.get("changed"):
        console.print("[green]Hosted domain updated.[/green]")
        _restart_for_tunnel()
    else:
        console.print("[dim]Already current — nothing to restart.[/dim]")


async def _configured_relay_url(client: NerditClient) -> str | None:
    """The relay already persisted in `[link]`, or `None`.

    Best-effort by design: this read only decides whether the production
    default may fill an empty seam, so an older daemon, a permission error or
    an unexpected shape must degrade to "nothing configured", never to a failed
    claim.
    """
    try:
        values = _section_values(await client.get_config("link"))
    except Exception:  # noqa: BLE001 — advisory read, never fatal
        return None
    existing = values.get("relay_url")
    return existing.strip() if isinstance(existing, str) and existing.strip() else None


def _targets_local_daemon() -> bool:
    """Check the configured host before probing local service units.

    A loopback host is necessary but insufficient; `_unit_manages_target` checks
    the port so a second local daemon is not restarted accidentally.
    """
    try:
        from nerdit.config.settings import get_client_config

        host, port, _token = get_client_config()
    except Exception:  # noqa: BLE001 — advisory; a broken config is not a reason to guess
        return False
    return str(host).strip().strip("[]").lower() in _LOOPBACK_HOSTS


def _unit_manages_target(unit: ServiceUnit) -> bool:
    """Check whether the unit serves the CLI's target port.

    Decline known mismatches. An unknown unit port permits restart to preserve
    auto-restart when unit configuration is unreadable.
    """

    managed_port = managed_daemon_port(unit)
    if managed_port is None:
        return True
    try:
        from nerdit.config.settings import get_client_config

        _host, port, _token = get_client_config()
    except Exception:  # noqa: BLE001 — advisory, as above
        return True
    return int(managed_port) == int(port)


def _restart_command(unit: ServiceUnit) -> tuple[list[str] | None, str]:
    """Return restart argv and a printable fallback command.

    Non-root callers use `sudo -n` for system units when available; printable
    fallbacks include sudo to avoid silent privilege failures or polkit prompts.
    """
    argv = list(unit.restart_argv)
    if unit.kind != "systemd-system" or os.geteuid() == 0:
        return argv, " ".join(argv)
    printable = "sudo " + " ".join(argv)
    sudo = shutil.which("sudo")
    if sudo is None:
        return None, printable
    return [sudo, "-n", *argv], printable


def _restart_for_tunnel() -> None:
    """Restart the service unit noninteractively to apply committed link settings.

    On failure, print the exact manual command and retain success: the claim
    already committed. Never invoke the prompting daemon-restart CLI here.
    """
    if not _targets_local_daemon():
        console.print(
            "[yellow]This CLI is configured for a remote daemon, so no local service was "
            "touched. Restart the daemon on that host to bring the tunnel up.[/yellow]"
        )
        return

    unit = detect_service_unit()
    if unit is None:
        console.print(
            "[dim]The tunnel starts on the next daemon restart — run: nerdit daemon restart[/dim]"
        )
        return

    if not _unit_manages_target(unit):
        console.print(
            "[yellow]That service unit manages a different daemon on this machine, so no "
            "service was touched. Restart the daemon you just linked to bring the tunnel "
            "up.[/yellow]"
        )
        return

    restart_argv, command = _restart_command(unit)
    if restart_argv is None:
        console.print(
            f"[yellow]This daemon runs under a system service unit, which needs root. "
            f"Restart it manually: {_plain(command)}[/yellow]"
        )
        return

    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv from install_layout, no shell
            restart_argv,
            capture_output=True,
            text=True,
            timeout=_RESTART_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail: str = f"{type(exc).__name__}: {exc}"
    else:
        if proc.returncode == 0:
            console.print("[green]Daemon service restarted — the tunnel is starting.[/green]")
            return
        first_line = next(
            (line.strip() for line in (proc.stderr or "").splitlines() if line.strip()),
            f"exit {proc.returncode}",
        )
        detail = first_line

    console.print(
        f"[yellow]Could not restart the service automatically ({_plain(detail)}). "
        f"Restart it manually: {_plain(command)}[/yellow]"
    )


async def _render_status(client: NerditClient) -> None:
    """Show persisted link config beside live capabilities to expose pending restart changes."""
    try:
        config_view = await client.get_config("link")
        caps = await client.get_capabilities()
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    values = _section_values(config_view)
    node_id = values.get("node_id")
    persisted_enabled = bool(values.get("enabled"))

    console.print("[bold]Link (persisted)[/bold]")
    console.print(f"  linked      = {'yes' if node_id else 'no'}")
    console.print(f"  node_id     = {_plain(node_id)}")
    console.print(f"  slug        = {_plain(values.get('slug'))}")
    console.print(f"  hosted dom. = {_plain(values.get('nodes_base_domain'))}")
    console.print(f"  relay_url   = {_plain(values.get('relay_url') or None)}")
    console.print(f"  enabled     = {persisted_enabled}")

    live = caps.get("link") if isinstance(caps, dict) else None
    live = live if isinstance(live, dict) else {}
    live_enabled = bool(live.get("enabled"))
    console.print("[bold]Link (live session)[/bold]")
    console.print(f"  enabled     = {live_enabled}")
    if live_enabled:
        console.print(f"  state       = {_plain(live.get('state'))}")
        console.print(f"  connected   = {_plain(live.get('connected_at'))}")
        console.print(f"  cap expires = {_plain(live.get('capability_expires_at'))}")
        # (P32) The mirrored account entitlement. Omitted entirely on a daemon
        # that predates the key rather than guessed at.
        entitlement = _entitlement_line(live)
        if entitlement is not None:
            console.print(f"  public      = {escape(entitlement)}")
        # (P33) The mirrored GitHub installation tokens — a count and a lead.
        github = _github_line(live)
        if github is not None:
            console.print(f"  github      = {escape(github)}")
        # (P27 WP-C4) A terminal link never re-dials on its own, so the reason
        # is only half the story — the operator needs the next step. Keyed on
        # the machine token from /capabilities (never on rendered text), so an
        # older daemon that does not carry the field just prints the state.
        if live.get("state") == "terminal":
            reason = live.get("terminal_reason")
            console.print(f"  terminal    = {_plain(reason)}")
            if reason == "entitlement_required":
                # This terminal code means account suspension/deletion, not a billing problem.
                # Recovery also needs restart because terminal links do not reconnect themselves.
                console.print(
                    "[yellow]The cloud refused the tunnel: this Nerdit account is "
                    "not active (suspended, or being deleted). Retrying will not "
                    "help; restore the account with support, then restart the "
                    "daemon (nerdit daemon restart).[/yellow]"
                )
            elif reason == "revoked":
                # Unlink before re-link: revocation leaves node_id persisted, which otherwise
                # short-circuits as already linked. Local unlink works even after cloud deletion.
                # Explain identity replacement; re-link performs its own restart.
                console.print(
                    "[yellow]The cloud revoked this node's link. Retrying will not "
                    "help, and linking is refused while this daemon still holds the "
                    "revoked identity: run 'nerdit unlink' (it wipes the node "
                    "identity key), then 'nerdit link --device'.[/yellow]"
                )

    if node_id and persisted_enabled and not live_enabled:
        console.print(
            "[yellow]Linked but the tunnel manager is not running — restart the "
            "daemon (nerdit daemon restart).[/yellow]"
        )
    if not node_id:
        # (D-P34-2 / OD-P34-4) Discoverability lives in this line, not in the
        # verb's semantics: bare ``nerdit link`` stays a read, and the hint is
        # what points an operator at the flow that needs no code at all. The
        # same command is named by the installer's NOT LINKED banner and the
        # doctor's never-linked row, so all three agree on one next step.
        console.print("[dim]Not linked. Link this node (free) with: nerdit link --device[/dim]")


def unlink(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Do not prompt for confirmation.",
    ),
) -> None:
    """Revoke this daemon's cloud link: drop the tunnel and wipe its identity."""
    asyncio.run(_unlink_async(yes))


async def _unlink_async(yes: bool) -> None:
    from nerdit.cli.client import get_configured_client

    # Custody warning before anything is called: unlink destroys the node key,
    # so a re-link enrolls a NEW verifier — it is not an undo.
    if not yes and not typer.confirm(
        "Unlink this daemon? This wipes its node identity key and config "
        "and drops any live tunnel.",
        default=False,
    ):
        console.print("[red]Aborted — nothing was changed.[/red]")
        raise typer.Exit(1)

    client = get_configured_client()
    try:
        result = await client.unlink_node(idempotency_key=uuid4().hex)
    except Exception as exc:  # noqa: BLE001 — rendered for the user
        render_client_error(exc)
        raise typer.Exit(1) from exc

    pending = int(result.get("pending_wipes") or 0)
    did_anything = (
        result.get("was_linked") or result.get("tunnel_stopped") or result.get("key_removed")
    )
    # "Nothing to do" must be TRUE: a staged config change can empty
    # was_linked while a live tunnel (and its key) still existed and was just
    # torn down — report what actually happened, not what the config said.
    if not did_anything and pending == 0:
        console.print("[dim]This daemon was not linked — nothing to do.[/dim]")
        return

    console.print("[green]Unlinked.[/green]")
    console.print("  live tunnel dropped" if result.get("tunnel_stopped") else "  no live tunnel")
    if result.get("key_removed"):
        console.print("  node identity key deleted")
    elif did_anything:
        console.print(
            "  [yellow]node identity key could not be removed — check the daemon log[/yellow]"
        )
    if pending:
        # "key deleted" alone must never conceal an OLDER credential still
        # stuck on disk from an earlier failed wipe.
        console.print(
            f"  [yellow]{pending} earlier key wipe(s) still pending — "
            "run 'nerdit unlink' again once the cause is fixed[/yellow]"
        )
