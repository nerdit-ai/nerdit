"""Resolve `[ai.*]` binding specs at each launch without persisting secret values.

The default binding injects `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`OPENAI_MODEL`; every binding also injects `NERDIT_AI_<NAME>_URL`, `_KEY`, and
`_MODEL`. Re-resolution keeps values fresh across restart and redeploy.
`BindingNotReady` causes launch to retry on a later tick, not fail permanently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from nerdit.config.project import AiBindingConfig

# Re-export the shared error for existing callers; secretref stays an import leaf.
from nerdit.core.bindings.secretref import BindingNotReady, resolve_secret_ref
from nerdit.core.jobconfig import parse_job_config
from nerdit.db.enums import JobKind, JobStatus
from nerdit.db.queries import Queries

# Placeholder API key injected for locally served models (Decision #1): nothing
# validates bearer tokens on model endpoints today, so minting real api_tokens
# rows would add no security value. If this ever flips to real minted tokens,
# the same change MUST add the action to NO_BODY_CACHE_ACTIONS
# (daemon/idempotency.py) so idempotent replays never re-serve a plaintext key.
LOCAL_MODEL_API_KEY = "nerdit-local"


@dataclass(frozen=True)
class ResolvedBinding:
    """One resolved binding: the concrete values injected into the app env."""

    base_url: str
    api_key: str
    model: str


async def resolve_binding(
    name: str,
    spec: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    shared_env: dict[str, str] | None = None,
) -> ResolvedBinding:
    """Resolve one persisted `config['ai'][name]` spec dict to concrete values.

    *secrets* is the app's already-loaded write-only secret env (one load per
    launch, shared with the container env). *shared_env* is the node-wide
    shared secret store — loaded lazily by the caller, and consulted only
    for `${secrets.shared.KEY}` refs. *bridge_host* is the docker bridge
    gateway IP local model endpoints are reachable on from app containers.
    """
    try:
        cfg = AiBindingConfig(**spec)
    except (ValidationError, TypeError) as exc:
        # Deploy-time validation (S8) makes this near-impossible; a manually
        # edited blob still gets an actionable line instead of a stack trace.
        raise BindingNotReady(
            f"[ai.{name}] persisted spec is invalid ({exc}); "
            f"redeploy the app with a valid [ai.{name}] section"
        ) from exc
    if cfg.provider == "api":
        return _resolve_api(name, cfg, secrets, shared_env or {})
    return await _resolve_ollama(name, cfg, queries, bridge_host)


def _resolve_api(
    name: str,
    cfg: AiBindingConfig,
    secrets: dict[str, str],
    shared_env: dict[str, str],
) -> ResolvedBinding:
    """`provider='api'`: base_url from the spec, api_key from the app's secrets.

    The precedence rules for a `${secrets[.shared].KEY}` ref live in
    `nerdit.core.bindings.secretref.resolve_secret_ref`, shared with
    the `[db.*]` twin's `_resolve_external`.
    """
    value = resolve_secret_ref(
        cfg.api_key, secrets, shared_env, label=f"ai.{name}", field="api_key"
    )
    return ResolvedBinding(base_url=cfg.base_url or "", api_key=value, model=cfg.model)


async def _resolve_ollama(
    name: str, cfg: AiBindingConfig, queries: Queries, bridge_host: str
) -> ResolvedBinding:
    """`provider='ollama'`: strict readiness — RUNNING + `model_pulled` + endpoint.

    Decision #5: `degraded`/`restarting` are NOT ready; the app waits for a
    healthy served model rather than launching against a flapping endpoint.
    """
    # Resolve by the served model ref, not the derived name: a model served
    # under a `--name` override still satisfies the binding (its row's
    # `service_name` differs from `sanitize_model_name(cfg.model)`).
    row = await queries.get_model_by_ref(cfg.model)
    if row is None or row.kind is not JobKind.model:
        raise BindingNotReady(
            f"[ai.{name}] model '{cfg.model}' is not served yet — run 'nerdit serve {cfg.model}'"
        )
    if row.status is not JobStatus.running:
        raise BindingNotReady(
            f"[ai.{name}] model '{cfg.model}' is not ready yet "
            f"(status: {row.status.value}) — waiting for it to reach running"
        )
    if not parse_job_config(row).get("model_pulled"):
        raise BindingNotReady(
            f"[ai.{name}] model '{cfg.model}' is still pulling its weights — waiting"
        )
    # The endpoint keys off the row's actual service_name (which may be a
    # custom override), not the derived name.
    endpoint = (
        await queries.get_service_endpoint(row.service_name)
        if row.service_name is not None
        else None
    )
    if endpoint is None:
        raise BindingNotReady(f"[ai.{name}] model '{cfg.model}' has no live endpoint yet — waiting")
    return ResolvedBinding(
        # Invariant #1: the OpenAI-compatible /v1 base, on the bridge gateway
        # so it is reachable from inside the app container (never the LAN).
        base_url=f"http://{bridge_host}:{endpoint.host_port}/v1",
        api_key=LOCAL_MODEL_API_KEY,
        model=cfg.model,
    )


async def resolve_bindings(
    specs: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    shared_env: dict[str, str] | None = None,
) -> dict[str, ResolvedBinding]:
    """Resolve every binding in a persisted `config['ai']` table.

    All-or-nothing: the first `BindingNotReady` propagates, so an app is
    never launched with a partially wired AI env. Names are iterated sorted for
    a deterministic first error message (log dedupe keys on the message).
    """
    resolved: dict[str, ResolvedBinding] = {}
    for name in sorted(specs):
        spec = specs[name]
        resolved[name] = await resolve_binding(
            name,
            spec if isinstance(spec, dict) else {},
            queries,
            secrets,
            bridge_host,
            shared_env=shared_env,
        )
    return resolved


def inject_env(resolved: dict[str, ResolvedBinding]) -> dict[str, str]:
    """Map resolved bindings to the frozen env-var contract (Invariant #1).

    Every binding gets `NERDIT_AI_<NAME>_URL/_KEY/_MODEL` (the binding-name
    grammar guarantees env-safe characters); the `default` binding
    additionally gets the plain `OPENAI_*` triplet an unmodified OpenAI SDK
    picks up.
    """
    env: dict[str, str] = {}
    for name, binding in resolved.items():
        prefix = f"NERDIT_AI_{name.upper()}"
        env[f"{prefix}_URL"] = binding.base_url
        env[f"{prefix}_KEY"] = binding.api_key
        env[f"{prefix}_MODEL"] = binding.model
    default = resolved.get("default")
    if default is not None:
        env["OPENAI_BASE_URL"] = default.base_url
        env["OPENAI_API_KEY"] = default.api_key
        env["OPENAI_MODEL"] = default.model
    return env


def inject_env_key_names(binding_names: Iterable[str]) -> set[str]:
    """Env-var NAMES `inject_env` would produce for these binding names.

    Derived BY CALLING `inject_env` with placeholder bindings so the
    key grammar lives in exactly one place (F7-ENVGRAMMAR — no parallel mirror
    of the launch env assembly). Values are irrelevant; only `.keys()` is read.
    """
    placeholder = {str(name): ResolvedBinding("", "", "") for name in binding_names}
    return set(inject_env(placeholder).keys())
