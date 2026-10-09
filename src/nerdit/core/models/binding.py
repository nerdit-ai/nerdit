"""Resolve `[ai.*]` binding specs at each launch without persisting secret values.

The default binding injects `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`OPENAI_MODEL`; every binding also injects `NERDIT_AI_<NAME>_URL`, `_KEY`, and
`_MODEL`. Re-resolution keeps values fresh across restart and redeploy.
`BindingNotReady` causes launch to retry on a later tick, not fail permanently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from pydantic import ValidationError

from nerdit.config.project import AiBindingConfig
from nerdit.config.settings import AiGatewaySettings
from nerdit.core.ai_gateway.keys import mint_key

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


async def resolve_binding(  # noqa: PLR0913 - launch inputs + keyword-only gateway
    name: str,
    spec: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    shared_env: dict[str, str] | None = None,
    *,
    gateway: AiGatewaySettings | None = None,
    gateway_listening: bool = True,
) -> ResolvedBinding:
    """Resolve one persisted `config['ai'][name]` spec dict to concrete values.

    *secrets* is the app's already-loaded write-only secret env (one load per
    launch, shared with the container env). *shared_env* is the node-wide
    shared secret store — loaded lazily by the caller, and consulted only
    for `${secrets.shared.KEY}` refs. *bridge_host* is the docker bridge
    gateway IP local model endpoints are reachable on from app containers.
    *gateway* is the `[ai_gateway]` section (`None` reads as disabled). A
    `gateway` binding comes back with an empty `api_key`: only
    `resolve_bindings` mints virtual keys.
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
    if cfg.provider == "gateway":
        return await _resolve_gateway(name, cfg, queries, gateway, bridge_host, gateway_listening)
    return await _resolve_ollama(name, cfg, queries, bridge_host)


async def _resolve_gateway(
    name: str,
    cfg: AiBindingConfig,
    queries: Queries,
    settings: AiGatewaySettings | None,
    bridge_host: str,
    listening: bool = True,
) -> ResolvedBinding:
    """`provider='gateway'`: readiness, then the machine's gateway URL and the alias.

    The provider key never enters the container: the gateway resolves it per
    request. The virtual key is filled in by `resolve_bindings`.
    """
    if settings is None or not settings.enabled:
        raise BindingNotReady(
            f"[ai.{name}] uses the AI gateway, which is off on this machine — "
            f"enable [ai_gateway] in config.toml and restart the daemon"
        )
    if not listening:
        raise BindingNotReady(
            f"[ai.{name}] uses the AI gateway, which failed to start on this machine "
            f"(its port may be in use) — see the daemon log, then restart the daemon"
        )
    if await queries.get_ai_route(cfg.model) is None:
        raise BindingNotReady(
            f"[ai.{name}] model alias '{cfg.model}' is not defined on this machine — "
            f"run: nerdit ai routes set {cfg.model} ..."
        )
    return ResolvedBinding(
        base_url=f"http://{bridge_host}:{settings.port}/v1", api_key="", model=cfg.model
    )


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


async def resolve_bindings(  # noqa: PLR0913 - launch inputs + keyword-only gateway
    specs: dict,
    queries: Queries,
    secrets: dict[str, str],
    bridge_host: str,
    shared_env: dict[str, str] | None = None,
    *,
    gateway: AiGatewaySettings | None = None,
    gateway_listening: bool = True,
) -> dict[str, ResolvedBinding]:
    """Resolve every binding in a persisted `config['ai']` table.

    All-or-nothing: the first `BindingNotReady` propagates, so an app is
    never launched with a partially wired AI env. Names are iterated sorted for
    a deterministic first error message (log dedupe keys on the message).
    *gateway* is the `[ai_gateway]` section; `gateway` bindings come back
    keyless — `mint_gateway_key` fills them once the whole launch env resolved.
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
            gateway=gateway,
            gateway_listening=gateway_listening,
        )
    return resolved


async def mint_gateway_key(
    resolved: dict[str, ResolvedBinding],
    specs: dict,
    queries: Queries,
    service_name: str,
    *,
    rotate: bool,
) -> dict[str, ResolvedBinding]:
    """Fill every `gateway` binding with one freshly minted virtual key for the service.

    Called once the whole launch env resolved, so a binding wait never mints
    (nor rotates) a key. *rotate* revokes the service's other keys: set by a
    main-container launch; a one-off run, a release or a cutover green mints
    alongside so the serving container keeps its key.
    ponytail: revoked rows (and the side keys of runs/greens until the next
    main launch) are never swept; prune on `revoked_at` if the table grows.
    """
    names = [
        name
        for name, spec in specs.items()
        if isinstance(spec, dict) and spec.get("provider") == "gateway" and name in resolved
    ]
    if not names:
        return resolved
    key = await mint_key(queries, service_name, revoke_others=rotate)
    return {
        name: replace(binding, api_key=key) if name in names else binding
        for name, binding in resolved.items()
    }


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
