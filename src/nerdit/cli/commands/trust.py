"""Install the daemon's internal CA after out-of-band fingerprint verification.

The initial fetch is unauthenticated. Recompute the fingerprint from the PEM
and require interactive confirmation against the daemon log or an explicit
--fingerprint pin; never install silently.
"""

import asyncio
import platform
import subprocess
from pathlib import Path
from typing import Optional

import typer
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from rich.markup import escape

from nerdit.cli.display import call_or_exit, console
from nerdit.utils.certs import (
    FINGERPRINT_PREFIX,
    ca_fingerprint,
    canonical_pem,
    normalize_fingerprint,
    parse_single_certificate,
)


def trust(
    fingerprint: Optional[str] = typer.Option(  # noqa: UP007 — Typer needs Optional[]
        None,
        "--fingerprint",
        help=(
            "Expected CA fingerprint (sha256:<hex>, as logged by the daemon at "
            "startup). Skips the interactive confirmation; a mismatch aborts."
        ),
    ),
    output: Optional[Path] = typer.Option(  # noqa: UP007
        None,
        "--output",
        "-o",
        help="Write the verified root certificate to PATH instead of installing it.",
    ),
) -> None:
    """Fetch and install the daemon's internal CA root certificate."""
    asyncio.run(_trust_async(fingerprint, output))


async def _trust_async(expected_fingerprint: str | None, output: Path | None) -> None:
    from nerdit.cli.client import get_configured_client

    client = get_configured_client()
    pem = await call_or_exit(client.get_proxy_ca())

    try:
        # Parse-then-reserialize: the fingerprint and the installed bytes must
        # describe the same single certificate. ca_fingerprint rejects
        # multi-cert blobs (a concatenated legit||rogue body would otherwise
        # pass the fingerprint check while the OS trust step installs both).
        actual = ca_fingerprint(pem)
        pem = canonical_pem(pem)
    except ValueError:
        console.print(
            "[red]The daemon returned data that is not exactly one valid certificate.[/red]"
        )
        raise typer.Exit(1) from None

    console.print(f"Fetched internal CA root certificate.\nFingerprint: [bold]{actual}[/bold]")

    if expected_fingerprint is not None:
        if normalize_fingerprint(expected_fingerprint) != actual:
            console.print(
                "[red]Fingerprint mismatch — NOT installing.[/red]\n"
                f"[dim]expected {normalize_fingerprint(expected_fingerprint)}[/dim]\n"
                "[dim]Someone on the network may be intercepting the connection, "
                "or the daemon's CA was recreated. Verify the value in the daemon "
                "startup log.[/dim]"
            )
            raise typer.Exit(1)
        console.print("[green]Fingerprint verified.[/green]")
    else:
        console.print(
            "[dim]Compare it with the 'Internal CA fingerprint' line in the daemon "
            "startup log before continuing.[/dim]"
        )
        if not typer.confirm("Does the fingerprint match the one shown by the daemon?"):
            console.print("[red]Aborted — nothing was installed.[/red]")
            raise typer.Exit(1)

    if output is not None:
        output.expanduser().write_text(pem)
        console.print(f"[green]Root certificate written to {output}[/green] (not installed).")
        return

    _install_ca(pem, actual)


def _install_ca(pem: str, fingerprint: str) -> None:
    """Install the verified PEM into the OS trust store (platform-dispatched).

    Never escalates on its own: when the install command needs privileges it
    does not have, the exact commands are printed instead. Firefox/NSS keeps
    its own store — documented as a manual step (docs/guide/proxy.md §5).
    """
    fp8 = fingerprint.removeprefix("sha256:")[:8]
    saved = Path.home() / ".nerdit" / f"nerdit-root-{fp8}.crt"
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(pem)
    console.print(f"Certificate saved to [bold]{saved}[/bold]")

    system = platform.system()
    if system == "Linux":
        target = f"/usr/local/share/ca-certificates/nerdit-{fp8}.crt"
        commands = [
            ["sudo", "cp", str(saved), target],
            ["sudo", "update-ca-certificates"],
        ]
    elif system == "Darwin":
        commands = [
            [
                "sudo",
                "security",
                "add-trusted-cert",
                "-d",
                "-r",
                "trustRoot",
                "-k",
                "/Library/Keychains/System.keychain",
                str(saved),
            ]
        ]
    elif system == "Windows":
        commands = [["certutil", "-addstore", "Root", str(saved)]]
    else:
        console.print(
            f"[yellow]Unsupported platform '{system}' — add {saved} to your "
            "trust store manually.[/yellow]"
        )
        return

    for cmd in commands:
        console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        try:
            result = subprocess.run(cmd, check=False)  # noqa: S603 — fixed argv, no shell
        except OSError as exc:
            console.print(f"[red]Could not run the install command: {exc}[/red]")
            _print_manual_hint(commands)
            raise typer.Exit(1) from exc
        if result.returncode != 0:
            console.print("[red]Install command failed.[/red]")
            _print_manual_hint(commands)
            raise typer.Exit(1)
    console.print(
        "[green]Internal CA installed — LAN https:// URLs from this daemon are "
        "now trusted.[/green]\n[dim]Firefox uses its own certificate store; import "
        "the saved .crt there manually if needed.[/dim]"
    )


def _print_manual_hint(commands: list[list[str]], action: str = "finish the install") -> None:
    console.print(f"[dim]Run these commands manually to {action}:[/dim]")
    for cmd in commands:
        console.print(f"[dim]  {' '.join(cmd)}[/dim]")


# -- untrust ----------------------------------------------------------------------


def _untrust_commands(cert: x509.Certificate) -> Optional[list[list[str]]]:  # noqa: UP007 — keep the whole signature explicit
    """Return platform removal commands without I/O, or None if unsupported.

    Derive the Debian filename from the SHA-256 prefix and macOS/Windows selectors
    from SHA-1. Compute both from the certificate body, never its filename.
    """
    fp8 = cert.fingerprint(hashes.SHA256()).hex()[:8]
    system = platform.system()
    if system == "Linux":
        return [
            ["sudo", "rm", "-f", f"/usr/local/share/ca-certificates/nerdit-{fp8}.crt"],
            # --fresh: a plain run leaves stale symlinks/hook entries (Java,
            # p11-kit) for a removed local cert; fresh rebuilds from scratch.
            ["sudo", "update-ca-certificates", "--fresh"],
        ]
    sha1_hex = cert.fingerprint(hashes.SHA1()).hex().upper()
    if system == "Darwin":
        return [
            [
                "sudo",
                "security",
                "delete-certificate",
                "-Z",
                sha1_hex,
                "/Library/Keychains/System.keychain",
            ]
        ]
    if system == "Windows":
        return [["certutil", "-delstore", "Root", sha1_hex]]
    return None


def untrust(
    fingerprint: Optional[str] = typer.Option(  # noqa: UP007 — Typer needs Optional[]
        None,
        "--fingerprint",
        help=(
            "Only untrust the CA with this fingerprint (sha256:<hex>). "
            "Without it, every saved Nerdit CA under ~/.nerdit is offered."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Do not prompt before removing each CA from the trust store.",
    ),
) -> None:
    """Remove a previously trusted internal CA from this machine's trust store.

    Fully offline (the inverse of `nerdit trust`): it operates on the locally
    saved `~/.nerdit/nerdit-root-*.crt` copies, recomputing each fingerprint
    from the file body — never from the filename — before printing and running
    the platform removal commands. The saved copy is deleted only after a
    successful removal, so a failure leaves the fingerprint recoverable.
    """
    saved_dir = Path.home() / ".nerdit"
    certs = sorted(saved_dir.glob("nerdit-root-*.crt"))
    if not certs:
        console.print(
            "[dim]No saved CA certificates found under ~/.nerdit — nothing to untrust.[/dim]"
        )
        return

    wanted = normalize_fingerprint(fingerprint) if fingerprint is not None else None

    # (path, cert, fingerprint) for every parseable saved copy; the fingerprint
    # is recomputed from the body, so a renamed/tampered file cannot mislead.
    parsed: list[tuple[Path, x509.Certificate, str]] = []
    for path in certs:
        try:
            cert = parse_single_certificate(path.read_text())
        except (OSError, ValueError) as exc:
            console.print(
                f"[yellow]Skipping unreadable certificate {escape(str(path))}: "
                f"{escape(str(exc))}[/yellow]"
            )
            continue
        fp = FINGERPRINT_PREFIX + cert.fingerprint(hashes.SHA256()).hex()
        parsed.append((path, cert, fp))

    if wanted is not None:
        parsed = [entry for entry in parsed if entry[2] == wanted]
        if not parsed:
            console.print(f"[red]No saved certificate matches fingerprint {escape(wanted)}.[/red]")
            raise typer.Exit(1)

    for path, cert, fp in parsed:
        console.print(f"Certificate [bold]{escape(fp)}[/bold] [dim]({escape(str(path))})[/dim]")
        commands = _untrust_commands(cert)
        if commands is None:
            console.print(
                f"[yellow]Unsupported platform '{escape(platform.system())}' — remove "
                f"{escape(str(path))} from your trust store manually.[/yellow]"
            )
            continue

        for cmd in commands:
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
        if not yes and not typer.confirm(
            "Remove this CA from the system trust store?", default=False
        ):
            console.print("[dim]Skipped.[/dim]")
            continue

        for cmd in commands:
            try:
                result = subprocess.run(cmd, check=False)  # noqa: S603 — fixed argv, no shell
            except OSError as exc:
                console.print(f"[red]Could not run the removal command: {escape(str(exc))}[/red]")
                _print_manual_hint(commands, action="finish the removal")
                raise typer.Exit(1) from exc
            if result.returncode != 0:
                console.print("[red]Removal command failed.[/red]")
                _print_manual_hint(commands, action="finish the removal")
                raise typer.Exit(1)

        # Only now that the OS removal succeeded is the saved copy safe to drop;
        # keeping it on failure preserves the fingerprint for a retry.
        path.unlink(missing_ok=True)
        console.print(
            f"[green]Removed from the trust store and deleted saved copy "
            f"{escape(str(path))}.[/green]"
        )
