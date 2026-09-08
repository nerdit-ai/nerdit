"""Validate custom-domain ownership before creating Host routes and TLS subjects.

Reject wildcards, IP literals, IDN/punycode, single-label names, and the node's
own names or descendants. These guards prevent zone capture, ambiguous domain
identity, and takeover of dashboard or service routes. Reuse proxy DNS grammar.

Errors expose a machine reason and token-derived public message, never the
submitted value, which may contain a pasted credential. Check specific mistakes
such as URLs before generic grammar so callers receive actionable reasons.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nerdit.config.settings import _is_ip_literal, _validate_dns_name

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nerdit.config.settings import ProxySettings


#: One value-free sentence per `reason` — the half of a refusal that is safe
#: to send back (Codex round 1, P1 #3831777111). `message` quotes the input so
#: a human reading a log sees what they typed; that string must never leave the
#: daemon, because the "domain" a caller submits is an arbitrary path segment
#: and a mis-paste can be a bearer token — which would then ride the `422`
#: body into the MCP tool result and the agent transcript. The route builds its
#: envelope from these instead, and the machine `reason` beside it is what an
#: agent actually branches on.
_PUBLIC_MESSAGES: dict[str, str] = {
    "empty": "A domain is required.",
    "whitespace": "A domain must not contain whitespace.",
    "not_bare": "A domain must be a bare hostname, not a URL or a path.",
    "ip_literal": (
        "A domain must be a name, not an IP address. To serve a certificate for an "
        "address, add it to [proxy].extra_hostnames."
    ),
    "has_port": "A domain must not include a port; use a bare hostname.",
    "wildcard": "A domain must not be a wildcard; bind each name explicitly.",
    "idn": (
        "A domain must be a plain ASCII name; internationalized (IDN/punycode) names "
        "are not supported."
    ),
    "single_label": (
        "A domain must have at least two labels (e.g. 'app.example.com'); a "
        "single-label name is a LAN hostname, not a domain."
    ),
    "grammar": (
        "Each label of a domain must be 1-63 characters of lowercase a-z, 0-9 or '-', "
        "and must not start or end with '-'."
    ),
    "reserved": (
        "That name is one of this node's own names (or sits under one) and is already "
        "served by the proxy."
    ),
}

#: What a reason token nobody added a sentence for reads as. Value-free by
#: construction, so a future refusal reason cannot leak by omission.
_PUBLIC_FALLBACK = "That domain was refused."


class DomainInvalid(ValueError):  # noqa: N818 — name matches EdgeAuthInvalid, the proxy sibling
    """A custom domain the daemon refuses, with a machine reason token.

    `reason` is one of `empty`, `whitespace`, `not_bare`, `has_port`,
    `wildcard`, `ip_literal`, `idn`, `single_label`, `grammar`,
    `reserved` (S-W10) and is what the route puts in the `422
    domain.invalid` envelope's `detail`. There are **two** human halves and
    the difference is a security boundary: `message` quotes the offending
    input and stays inside the daemon (logs, tests, a developer's traceback),
    while `public_message` — derived from `reason` alone — is what any
    outward-facing surface may echo. A `ValueError` subclass so a caller
    that only wants "it did not validate" needs no import.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.public_message = _PUBLIC_MESSAGES.get(reason, _PUBLIC_FALLBACK)


def normalize_domain(raw: str) -> str:
    """Fold a caller-supplied domain to its canonical stored form. Never raises.

    `strip()` → `casefold()` → drop ONE trailing dot (the fully-qualified
    form `example.com.` and `example.com` are the same name; the stored
    column and every Host matcher use the dotless spelling).

    Total by design, because it runs on the DELETE path too: removing a domain
    must work on whatever spelling the operator typed, and a malformed string
    simply matches no row rather than raising. Validation is a separate,
    stricter step — see `validate_domain`.
    """
    value = raw.strip().casefold()
    if value.endswith("."):
        value = value[:-1]
    return value


@dataclass(frozen=True, slots=True)
class ReservedNames:
    """The set of names this node already answers for, folded (S-W10).

    Membership is by suffix as well as by equality: reserving `dev.lan` also
    reserves everything under it, because in subdomain mode every service's own
    Host lives there and a custom domain landing inside that zone could shadow
    one (or be shadowed by one, depending on ordering) — neither is a state an
    operator can reason about.
    """

    names: frozenset[str]

    def covers(self, domain: str) -> bool:
        """Whether `domain` IS one of the node's names or sits under one."""
        return any(domain == name or domain.endswith(f".{name}") for name in self.names)


def reserved_names(
    proxy: ProxySettings, hostname: str, *, aliases: Iterable[str] = ()
) -> ReservedNames:
    """Build the reserved set from this node's own names.

    Computed **once at boot** into `app.state.domain_reserved` (S-W10), and
    that is a security property, not an optimisation: every `[proxy]` key
    here is restart-required, so a set recomputed per request could only ever
    differ by reading a config change that the running proxy has not applied —
    a window in which a racing write could bind a name the live Caddy is about
    to claim. Restarting recomputes it, which is exactly when the proxy adopts
    the new config too.

    Sources: the **effective** hostname *as the caller resolved it*, plus
    `base_domain`, every `extra_hostnames` entry, and `aliases` — the
    other names this machine answers to that the caller knows about (the raw
    `socket.gethostname()`, its short label and `<short>.local`). Those
    aliases generate no URL and ride no TLS subject, but in **path mode the
    apex and every default route carry no Host matcher**, so a request that
    reaches the proxy under *any* name the box resolves to lands on them; a
    user domain bound to such an alias would capture that traffic (WP1
    security review). The caller supplies them so this module stays pure
    (no socket I/O). `hostname_override` is OR-ed in ahead of *hostname* so a
    caller that passes an empty string — several unit tests do — still
    reserves the configured name. Empty entries are dropped; everything is
    folded with `normalize_domain` so comparison is case-insensitive.
    """
    raw = [
        proxy.hostname_override or hostname,
        hostname,
        proxy.base_domain,
        *proxy.extra_hostnames,
        *aliases,
    ]
    folded = {normalize_domain(entry) for entry in raw if entry}
    return ReservedNames(names=frozenset(folded - {""}))


def host_aliases(raw_hostname: str) -> tuple[str, ...]:
    """The names a machine answers to besides its effective hostname.

    From the raw socket hostname: itself, its first label, and the mDNS
    `<short>.local` spelling. Pure; the caller reads the socket.
    """
    raw = raw_hostname.strip().rstrip(".")
    if not raw:
        return ()
    short = raw.split(".", 1)[0]
    return tuple(dict.fromkeys((raw, short, f"{short}.local")))


def validate_domain(raw: str, *, reserved: ReservedNames) -> str:
    """Return the folded domain, or raise `DomainInvalid`.

    The order below is load-bearing (see the module docstring): each check is
    written to fire before any later one could produce a vaguer message for the
    same input. Two consequences worth naming:

    * IP literals are classified BEFORE the port check, so `[::1]` and
      `::1` answer `ip_literal` rather than `has_port` — the colon in an
      IPv6 address is not a port separator, and telling an operator otherwise
      sends them to fix the wrong thing.
    * The shared `[proxy]` grammar runs LAST among the shape checks, so its
      generic "invalid label" only ever explains inputs none of the specific
      refusals recognised. Its own message is passed through verbatim — it
      already names the offending label.

    `reserved` is the boot-computed set; it is checked last because a name
    must be a well-formed domain before "is it ours?" is a meaningful question.
    """
    stripped = raw.strip()
    value = normalize_domain(raw)
    if not value:
        raise DomainInvalid("empty", "A domain is required.")
    if any(ch.isspace() for ch in value):
        raise DomainInvalid("whitespace", f"Domain '{value}' must not contain whitespace.")
    if "://" in value or "/" in value:
        raise DomainInvalid(
            "not_bare", f"Domain '{value}' must be a bare hostname, not a URL or path."
        )
    if _is_ip_literal(value) or (value.startswith("[") and value.endswith("]")):
        raise DomainInvalid(
            "ip_literal",
            f"Domain '{value}' is an IP address, not a name. To serve a certificate for "
            "an address, add it to [proxy].extra_hostnames.",
        )
    if ":" in value:
        raise DomainInvalid(
            "has_port", f"Domain '{value}' must not include a port; use a bare hostname."
        )
    if "*" in value:
        raise DomainInvalid(
            "wildcard", f"Domain '{value}' must not be a wildcard; bind each name explicitly."
        )
    # Tested against the caller's OWN bytes, not the folded form: `casefold()`
    # is not ASCII-preserving. `ſ` (U+017F) folds to `s` and `K` (U+212A,
    # Kelvin) to `k`, so a folded-only test would let a non-ASCII spelling in
    # as a DIFFERENT, ASCII name — silently rewriting operator input, the exact
    # posture the module docstring says this refuses. The message quotes what was
    # typed, since the folded form no longer shows the offending character.
    if not stripped.isascii() or any(label.startswith("xn--") for label in value.split(".")):
        raise DomainInvalid(
            "idn",
            f"Domain '{stripped}' must be a plain ASCII name; internationalized "
            "(IDN/punycode) names are not supported.",
        )
    if "." not in value:
        raise DomainInvalid(
            "single_label",
            f"Domain '{value}' must have at least two labels (e.g. 'app.example.com'); a "
            "single-label name is a LAN hostname, not a domain.",
        )
    try:
        _validate_dns_name(value, field_name="domain")
    except ValueError as exc:
        raise DomainInvalid("grammar", str(exc)) from exc
    if reserved.covers(value):
        raise DomainInvalid(
            "reserved",
            f"Domain '{value}' is one of this node's own names (or sits under one) and is "
            "already served by the proxy.",
        )
    return value
