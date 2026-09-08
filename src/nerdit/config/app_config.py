"""Validate effective app config for deploy and config-API writes.

Reuse project schemas and wrap failures in the shared structured 422 envelopes.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from nerdit.config.project import (
    AiBindingConfig,
    DbBindingConfig,
    DeployConfig,
    ProjectConfig,
    parse_ai_bindings,
    parse_db_bindings,
)
from nerdit.daemon.errors import NerditError
from nerdit.db.models import JobKind, JobStatus

# The [ai.*] shape hint returned with every ``deploy.invalid_ai`` envelope.
AI_SHAPE_HINT = (
    "Each [ai.<name>] table needs provider = 'ollama' | 'api' and model = '<ref>'; "
    "provider 'api' also requires base_url and api_key = '${secrets.KEY}'."
)

# The [db.*] shape hint returned with a ``deploy.invalid_db`` /
# ``db.external_url_invalid`` envelope (P15, D-A).
DB_SHAPE_HINT = (
    "Each [db.<name>] table needs provider = 'managed' | 'external'; 'managed' "
    "also requires database = '<db-name>'; 'external' requires a credential-free "
    "url (postgresql://|redis[s]://) and password = '${secrets.KEY}'."
)

# Statuses a model/database row may NOT be in for an [ai.*]/[db.*] binding to
# deploy — the settled terminal set of `get_reconcilable_services`
# (db/queries.py). Every other state (building/scheduled/running/degraded/
# restarting) is converging or live, so deploy-during-provision works.
MODEL_TERMINAL_STATUSES = frozenset(
    {JobStatus.completed, JobStatus.cancelled, JobStatus.stopped, JobStatus.failed}
)
DB_TERMINAL_STATUSES = MODEL_TERMINAL_STATUSES


def validate_deploy_fields(
    name: str,
    port: int | None,
    gpus: int,
    start: str | None,
    health: str | None,
    memory_limit: str | None = None,
    cpu_limit: float | None = None,
    volumes: list[str] | None = None,
    release: str | None = None,
    cutover: bool | None = None,
    auto_deploy: bool | None = None,
    # ``Any``, not ``EdgeAuthConfig | None``: the value arrives as an untrusted
    # blob (a ZIP's nerdit.toml table, a PUT body value, or a persisted config
    # key) and EdgeAuthConfig is the thing that decides whether it is one.
    edge_auth: Any = None,
) -> DeployConfig:
    """Validate effective deployment fields with DeployConfig.

    Port may be unset for buildpack defaults. Validate DNS names, resource limits,
    release commands, tri-state options and edge auth consistently with project
    config. Return structured 422 errors; credential-bearing validators must use
    value-free messages because hide_input_in_errors does not redact interpolation.
    """
    try:
        return DeployConfig(
            name=name,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
            volumes=volumes,
            release=release,
            cutover=cutover,
            auto_deploy=auto_deploy,
            edge_auth=edge_auth,
        )
    except ValidationError as exc:
        raise NerditError(
            422,
            "deploy.invalid",
            f"Invalid deploy parameters: {exc.errors()[0].get('msg', 'validation error')}",
            hint=(
                "Name must be a DNS label; port in 1-65535; gpus >= 0; "
                "memory_limit like '512m'/'2g'; cpu_limit > 0; "
                "volumes like 'data:/data' (<=8, DNS-label names, absolute paths); "
                "release a single-line command (<=4096 chars); "
                "cutover/auto_deploy booleans (null = daemon default); "
                "edge_auth = {user = '<name>', password = '${secrets.KEY}'} "
                "(a secret reference, never a literal password)."
            ),
        ) from exc


def validate_ai_section(
    ai_section: dict, *, source: str | None = None
) -> dict[str, AiBindingConfig]:
    """Validate AI binding tables and names through the project schemas.

    Return structured 422 errors, including the source filename when supplied.
    """
    try:
        bindings = parse_ai_bindings(ai_section)
        ProjectConfig(ai=bindings)  # binding-name grammar (NERDIT_AI_<NAME>_*)
    except (ValueError, ValidationError) as exc:
        where = f" in {source}" if source else ""
        raise NerditError(
            422,
            "deploy.invalid_ai",
            f"Invalid [ai.*] bindings{where}: {exc}",
            hint=AI_SHAPE_HINT,
        ) from exc
    return bindings


async def require_served_models(queries, bindings: dict[str, AiBindingConfig]) -> None:  # noqa: ANN001
    """Require a non-terminal served-model row for each ollama binding.

    Resolve config.model rather than the service name. Building rows are accepted;
    launch-time resolution checks readiness. API bindings need no local row.
    """
    for binding_name, binding in bindings.items():
        if binding.provider != "ollama":
            continue
        row = await queries.get_model_by_ref(binding.model)
        if row is None or row.kind != JobKind.model or row.status in MODEL_TERMINAL_STATUSES:
            raise NerditError(
                422,
                "ai.model_not_served",
                f"[ai.{binding_name}] needs model '{binding.model}', but no active "
                f"model workload is serving it.",
                hint=f"run 'nerdit serve {binding.model}' first",
            )


def _is_db_url_shape_error(exc: Exception) -> bool:
    """True when *exc* is a `[db].url` shape rejection (scheme/userinfo/query).

    Those come from the `DbBindingConfig._check_url_shape` field validator, so
    the `url` field appears in the error `loc` — the signal that maps to the
    resource-prefixed `db.external_url_invalid` code (§1.4) rather than the
    generic `deploy.invalid_db`.
    """
    if isinstance(exc, ValidationError):
        return any("url" in err.get("loc", ()) for err in exc.errors())
    return False


def validate_db_section(
    db_section: dict, *, source: str | None = None
) -> dict[str, DbBindingConfig]:
    """Validate database binding tables and names through the project schemas.

    URL-shape errors map to db.external_url_invalid; other schema errors map to
    deploy.invalid_db. Include the source filename when supplied.
    """
    try:
        bindings = parse_db_bindings(db_section)
        ProjectConfig(db=bindings)  # binding-name grammar (NERDIT_DB_<NAME>_URL)
    except (ValueError, ValidationError) as exc:
        where = f" in {source}" if source else ""
        code = "db.external_url_invalid" if _is_db_url_shape_error(exc) else "deploy.invalid_db"
        raise NerditError(
            422,
            code,
            f"Invalid [db.*] bindings{where}: {exc}",
            hint=DB_SHAPE_HINT,
        ) from exc
    return bindings


async def require_provisioned_databases(  # noqa: ANN001
    queries, db_bindings: dict[str, DbBindingConfig]
) -> None:
    """Require a non-terminal database row for each managed binding's service name.

    Building rows are accepted; launch-time resolution checks readiness.
    External bindings need no local row.
    """
    for binding_name, binding in db_bindings.items():
        if binding.provider != "managed":
            continue
        row = await queries.get_resource_by_ref(binding.database, JobKind.database)
        if row is None or row.kind != JobKind.database or row.status in DB_TERMINAL_STATUSES:
            raise NerditError(
                422,
                "db.not_provisioned",
                f"[db.{binding_name}] needs database '{binding.database}', but no active "
                f"database workload is serving it.",
                hint="run 'nerdit db create <backend>' first",
            )


async def validate_app_config(
    effective_doc: dict,
    queries,  # noqa: ANN001
) -> tuple[DeployConfig, dict[str, AiBindingConfig] | None]:
    """Validate an app's effective config document end to end.

    `effective_doc` is `{"deploy": {...}, "ai": {...}?}` with the deploy
    values already merged to their effective precedence by the caller. Runs the
    `DeployConfig` field validators, the `[ai.*]` binding schema, and
    the served-model gate; raises the same structured 422s as the deploy route.
    Returns the validated deploy config and bindings (`None` when no `ai`
    section is present).
    """
    deploy = effective_doc.get("deploy") or {}
    deploy_cfg = validate_deploy_fields(
        name=deploy.get("name") or "",
        port=deploy.get("port"),
        gpus=deploy.get("gpus", 0),
        start=deploy.get("start"),
        health=deploy.get("health"),
        memory_limit=deploy.get("memory_limit"),
        cpu_limit=deploy.get("cpu_limit"),
        volumes=deploy.get("volumes"),
        release=deploy.get("release"),
        cutover=deploy.get("cutover"),
        auto_deploy=deploy.get("auto_deploy"),
        edge_auth=deploy.get("edge_auth"),
    )
    ai_section = effective_doc.get("ai")
    bindings: dict[str, AiBindingConfig] | None = None
    if ai_section is not None:
        bindings = validate_ai_section(ai_section)
        if bindings:
            await require_served_models(queries, bindings)
    # (P15) The [db.*] gate sits beside the [ai.*] gate — the app-config-PUT
    # path. The deploy path calls require_provisioned_databases directly in
    # _finalize_deploy (validate_app_config is not on the deploy route).
    db_section = effective_doc.get("db")
    if db_section is not None:
        db_bindings = validate_db_section(db_section)
        if db_bindings:
            await require_provisioned_databases(queries, db_bindings)
    return deploy_cfg, bindings
