"""Resolve stored `${secrets[.shared].KEY}` references.

`${secrets.shared.KEY}` checks the service scope before the shared store.
`${secrets.KEY}` checks only the service scope, with no shared fallback.
Callers supply scope loaders and retain their own error and audit policies;
`resolve_secret_ref` turns missing binding secrets into `BindingNotReady`.
This leaf module must not import either binding implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from nerdit.config.project import SECRET_REF_RE


class BindingNotReady(Exception):  # noqa: N818 — domain condition, not an error (retry signal)
    """A `[ai.*]`/`[db.*]` binding cannot resolve *yet* (retry next tick).

    The message is actionable and user-facing: it is written verbatim to the
    app's `job_logs` so `nerdit logs`/`nerdit diagnose` tell the user
    exactly what to run.
    """


@dataclass(frozen=True)
class SecretRefResult:
    """Outcome of one `${secrets[.shared].KEY}` precedence walk."""

    matched: bool
    """False ⇔ *ref* failed the grammar; the loaders were never called."""
    scope: str | None
    """``"shared"`` or ``None`` (unscoped); ``None`` when unmatched."""
    key: str | None
    """The referenced key name; ``None`` when unmatched."""
    value: str | None
    """The resolved value, or ``None`` (unmatched, or set nowhere it may be)."""
    source: str | None
    """``"service"`` | ``"shared"`` — which scope supplied *value*, else ``None``."""


def walk_secret_ref(
    ref: str,
    *,
    service_env: Callable[[], Mapping[str, str]] | None,
    shared_env: Callable[[], Mapping[str, str]] | None,
) -> SecretRefResult:
    """Walk the P8 scope precedence for one stored reference.

    The loaders are lazy and consulted at most once each: *service_env* only
    when the grammar matched, *shared_env* only when the reference names the
    shared scope AND the per-service scope missed — so a site that reads its
    scopes from disk pays for exactly what the walk needs. Passing
    `service_env=None` expresses "the per-service scope must not be consulted
    at all" (the fresh-deploy carve-out on the git route; the shared-scope-only
    webhook targets), which is stronger than an empty mapping: an unscoped ref
    then has no scope left to read and resolves to nothing.

    Loader exceptions (notably `SecretDecryptError`) propagate untouched —
    each call site keeps its own handling. Secret values live only in the
    returned result: the kernel never logs, audits or otherwise records one.
    """
    match = SECRET_REF_RE.match(ref)
    if match is None:
        return SecretRefResult(matched=False, scope=None, key=None, value=None, source=None)
    scope, key = match.group(1), match.group(2)
    # `.get` rather than a membership test: `SecretManager.load` yields
    # `dict[str, str]`, so a stored key never holds `None` and a miss is
    # unambiguous.
    if service_env is not None:
        value = service_env().get(key)
        if value is not None:
            return SecretRefResult(
                matched=True, scope=scope, key=key, value=value, source="service"
            )
    if scope == "shared" and shared_env is not None:
        value = shared_env().get(key)
        if value is not None:
            return SecretRefResult(matched=True, scope=scope, key=key, value=value, source="shared")
    return SecretRefResult(matched=True, scope=scope, key=key, value=None, source=None)


def resolve_secret_ref(
    ref: str | None,
    secrets: dict[str, str],
    shared_env: dict[str, str],
    *,
    label: str,
    field: str,
) -> str:
    """Resolve one persisted `${secrets[.shared].KEY}` reference to its value.

    *label* is the section-and-name the message is templated on (e.g.
    `"ai.default"` or `"db.default"`, rendered as `[ai.default]` /
    `[db.default]`); *field* is the config field the reference came from
    (`"api_key"` / `"password"`). Raises `BindingNotReady` — with a
    message actionable enough to write verbatim to `job_logs` — when *ref*
    doesn't match the grammar or the key isn't set anywhere it's allowed to be.
    """
    res = walk_secret_ref(ref or "", service_env=lambda: secrets, shared_env=lambda: shared_env)
    if not res.matched:  # unreachable per the S1 grammar; defensive
        raise BindingNotReady(
            f"[{label}] {field} is not a ${{secrets.KEY}} reference; "
            f"redeploy the app with a valid [{label}] section"
        )
    if res.value is None:
        if res.scope == "shared":
            raise BindingNotReady(
                f"[{label}] shared secret '{res.key}' not set — run: "
                f"nerdit secrets set --shared {res.key}=..."
            )
        raise BindingNotReady(
            f"[{label}] secret '{res.key}' not set — run: nerdit secrets set <app> {res.key}=..."
        )
    return res.value
