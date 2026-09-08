"""Resolve `[db.*]` specs into connection env variables at every launch.

Each binding injects `NERDIT_DB_<NAME>_URL`. The default also injects
`DATABASE_URL` for Postgres or `REDIS_URL` for Redis. Persist only specs:
password-bearing DSNs belong solely in the app container's launch environment,
never job config, responses, audit records, or logs. Unready bindings raise
`BindingNotReady` so launch can retry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import ValidationError

from nerdit.config.project import DbBindingConfig
from nerdit.core.bindings.secretref import BindingNotReady, resolve_secret_ref
from nerdit.core.jobconfig import parse_job_config
from nerdit.core.secrets import SecretDecryptError
from nerdit.db.enums import JobKind, JobStatus
from nerdit.db.queries import Queries

if TYPE_CHECKING:
    from nerdit.core.data.backend import DataBackend

# Target-scheme → default-binding env alias (external path; the managed path
# takes the alias from the backend's `default_env_alias`). Disjoint from the
# `OPENAI_*` triplet by construction, so no cross-kind env collision exists.
_SCHEME_ALIAS = {
    "postgresql": "DATABASE_URL",
    "redis": "REDIS_URL",
    "rediss": "REDIS_URL",
}


@dataclass(frozen=True)
class ResolvedDbBinding:
    """One resolved `[db.*]` binding: the DSN + its default-binding alias."""

    url: str
    default_alias: str


def _compose_external_dsn(url: str, password: str) -> str:
    """Insert *password* into an external DSN's userinfo (`user:pw@host` shape).

    The URL was validated at parse time to carry no userinfo password and no
    query string (`nerdit.config.project.DbBindingConfig`), so a plain
    userinfo rebuild is unambiguous. The password (user-chosen — may contain
    `@ : / # ?`) is percent-encoded with `safe=""` so every reserved
    character round-trips; the raw username substring is preserved verbatim
    (never re-quoted, to avoid double-encoding). A URL without a username yields
    `:<quoted-pw>@host` (the Redis shape).

    The userinfo/host boundary is the **last** `@` (`rpartition`), matching
    `urlsplit`/libpq, so a username that itself contains a raw `@` (parse
    admits it — `parts.password` is `None`) still composes a DSN a client
    parses identically.
    """
    parts = urlsplit(url)
    raw_user, _, hostport = parts.netloc.rpartition("@")
    new_netloc = f"{raw_user}:{quote(password, safe='')}@{hostport}"
    return urlunsplit((parts.scheme, new_netloc, parts.path, parts.query, parts.fragment))


def _resolve_external(
    name: str,
    cfg: DbBindingConfig,
    secrets: dict[str, str],
    shared_env: dict[str, str],
) -> ResolvedDbBinding:
    """`provider='external'`: url from the spec, password from the app's secrets.

    The precedence rules for a `${secrets[.shared].KEY}` ref live in
    `nerdit.core.bindings.secretref.resolve_secret_ref`, shared with
    the `[ai.*]` twin's `_resolve_api`.
    """
    value = resolve_secret_ref(
        cfg.password, secrets, shared_env, label=f"db.{name}", field="password"
    )
    url = _compose_external_dsn(cfg.url or "", value)
    scheme = urlsplit(cfg.url or "").scheme
    return ResolvedDbBinding(url=url, default_alias=_SCHEME_ALIAS.get(scheme, "DATABASE_URL"))


async def _resolve_managed(
    name: str,
    cfg: DbBindingConfig,
    queries: Queries,
    bridge_host: str,
    backend_for: Callable[[dict], "DataBackend"],
    load_scope: Callable[[str], dict[str, str]],
) -> ResolvedDbBinding:
    """`provider='managed'`: strict readiness — RUNNING + `db_ready` + endpoint.

    Clones `_resolve_ollama`: `degraded`/`restarting` are NOT ready; the
    app waits for a healthy database rather than launching against a flapping
    endpoint. The credential-bearing DSN is composed here (launch time only) —
    the minted password is read from the database row's own secret scope.
    """
    ref = cfg.database or ""
    row = await queries.get_resource_by_ref(ref, JobKind.database)
    if row is None or row.kind is not JobKind.database:
        raise BindingNotReady(
            f"[db.{name}] database '{ref}' is not provisioned yet — "
            f"run 'nerdit db create <backend>' first"
        )
    if row.status is not JobStatus.running:
        raise BindingNotReady(
            f"[db.{name}] database '{ref}' is not ready yet "
            f"(status: {row.status.value}) — waiting for it to reach running"
        )
    row_cfg = parse_job_config(row)
    if not row_cfg.get("db_ready"):
        raise BindingNotReady(f"[db.{name}] database '{ref}' is still starting up — waiting")
    # `get_resource_by_ref(database)` matched `service_name == ref`, so the
    # row's service_name IS `ref` (a str) — used directly for the endpoint +
    # secret-scope lookups (keeps mypy narrow; row.service_name is str | None).
    endpoint = await queries.get_service_endpoint(ref)
    if endpoint is None:
        raise BindingNotReady(f"[db.{name}] database '{ref}' has no live endpoint yet — waiting")
    backend = backend_for(row_cfg)
    try:
        scope = load_scope(ref)
    except SecretDecryptError as exc:
        # No `str(exc)` interpolation: SecretDecryptError messages carry
        # filesystem paths by contract (core/secrets.py) and this string rides
        # verbatim into the app's job_logs and the owner-visible /diagnose
        # bindings.messages — keep it path-free, matching the [ai.*] side and
        # the diagnose/doctor discipline (the from-exc chaining still logs it).
        raise BindingNotReady(
            f"[db.{name}] database '{ref}' credential store is unreadable — "
            f"restore the secrets key or recreate the database"
        ) from exc
    password = scope.get(backend.minted_secret_key)
    if not password:
        raise BindingNotReady(
            f"[db.{name}] database '{ref}' has no minted credential — "
            f"delete and recreate it, or restore a backup"
        )
    # host is the docker bridge gateway so the DSN is reachable from inside the
    # app container (never the LAN) — the same posture as a local model endpoint.
    return ResolvedDbBinding(
        url=backend.dsn(bridge_host, endpoint.host_port, password),
        default_alias=backend.default_env_alias,
    )


async def resolve_db_binding(
    name: str,
    spec: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    backend_for: Callable[[dict], "DataBackend"],
    load_scope: Callable[[str], dict[str, str]],
    shared_env: dict[str, str] | None = None,
) -> ResolvedDbBinding:
    """Resolve one persisted `config['db'][name]` spec dict to a concrete DSN.

    *secrets* is the app's already-loaded write-only secret env; *shared_env* is
    the node-wide shared store (consulted only for `${secrets.shared.KEY}`
    refs). *backend_for* maps a database row's config to its backend (the
    `DataController.backend_for` self-healing resolver); *load_scope* loads a
    database row's secret scope (the minted password).
    """
    try:
        cfg = DbBindingConfig(**spec)
    except (ValidationError, TypeError) as exc:
        raise BindingNotReady(
            f"[db.{name}] persisted spec is invalid ({exc}); "
            f"redeploy the app with a valid [db.{name}] section"
        ) from exc
    if cfg.provider == "external":
        return _resolve_external(name, cfg, secrets, shared_env or {})
    return await _resolve_managed(name, cfg, queries, bridge_host, backend_for, load_scope)


async def resolve_db_bindings(
    specs: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    backend_for: Callable[[dict], "DataBackend"],
    load_scope: Callable[[str], dict[str, str]],
    shared_env: dict[str, str] | None = None,
) -> dict[str, ResolvedDbBinding]:
    """Resolve every binding in a persisted `config['db']` table.

    All-or-nothing: the first `BindingNotReady` propagates, so an app is
    never launched with a partially wired database env. Names are iterated
    sorted for a deterministic first error message (log dedupe keys on it).
    """
    resolved: dict[str, ResolvedDbBinding] = {}
    for name in sorted(specs):
        spec = specs[name]
        resolved[name] = await resolve_db_binding(
            name,
            spec if isinstance(spec, dict) else {},
            queries,
            secrets,
            bridge_host,
            backend_for,
            load_scope,
            shared_env=shared_env,
        )
    return resolved


def inject_db_env(resolved: dict[str, ResolvedDbBinding]) -> dict[str, str]:
    """Map resolved `[db.*]` bindings to the injected env-var contract (§1.2).

    Every binding gets `NERDIT_DB_<NAME>_URL` (the binding-name grammar
    guarantees env-safe characters); the `default` binding additionally gets
    its scheme's alias (`DATABASE_URL` | `REDIS_URL`) that an unmodified
    client library picks up.
    """
    env: dict[str, str] = {}
    for name, binding in resolved.items():
        env[f"NERDIT_DB_{name.upper()}_URL"] = binding.url
    default = resolved.get("default")
    if default is not None:
        env[default.default_alias] = default.url
    return env


def _default_alias_for_spec(spec: object) -> str:
    """Best-effort default-binding alias for a persisted spec (names-only path).

    Used by `inject_db_env_key_names` where no live resolution runs (no
    database row / backend in hand): external specs read the scheme from `url`;
    a managed spec defaults to `DATABASE_URL`.

    Accepted softness (pinned by a resolver test): a managed `default` binding
    whose target row is a **Redis** backend injects `REDIS_URL` at launch, but
    this names-only path reports `DATABASE_URL` — the only diagnose/app-config
    key-name projection a live resolve would correct. Resolving it here would
    need the target row's `config['backend']`, which the names-only callers do
    not thread; the divergence is cosmetic (a diagnose env-key NAME only, never a
    launched value) and the `NERDIT_DB_<NAME>_URL` key is always exact.
    """
    if isinstance(spec, dict) and spec.get("provider") == "external":
        scheme = urlsplit(str(spec.get("url") or "")).scheme
        return _SCHEME_ALIAS.get(scheme, "DATABASE_URL")
    return "DATABASE_URL"


def inject_db_env_key_names(specs: dict | None) -> set[str]:
    """Env-var NAMES `inject_db_env` would produce for these specs.

    Derived BY CALLING `inject_db_env` with placeholder bindings so the
    key grammar lives in exactly one place (F7-ENVGRAMMAR). The default-binding
    alias is derived from the spec shape (see `_default_alias_for_spec`),
    since the actual DSN is never assembled here. Values are irrelevant; only
    `.keys()` is read.
    """
    placeholder = {
        str(name): ResolvedDbBinding("", _default_alias_for_spec(spec))
        for name, spec in (specs or {}).items()
    }
    return set(inject_db_env(placeholder).keys())
