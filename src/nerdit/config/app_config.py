"""Validate effective app config for deploy and config-API writes.

Reuse project schemas and wrap failures in the shared structured 422 envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from nerdit.config.project import (
    AiBindingConfig,
    DbBindingConfig,
    DeployConfig,
    ProjectConfig,
    parse_ai_bindings,
    parse_db_bindings,
)
from nerdit.daemon.auth import current_principal, require_service_scope
from nerdit.daemon.errors import NerditError
from nerdit.db.models import Job, JobKind, JobStatus, TokenRole

if TYPE_CHECKING:
    from starlette.requests import Request

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
                "volumes like 'data:/data' (DNS-label names, absolute paths; at most 8 "
                "including the implicit data volume, so 7 declared unless one is named "
                "'data'); "
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


def _require_bindable_database(
    request: Request, binding_name: str, database: str, row: Job
) -> None:
    """Authorize the caller to point an app at *row*'s minted credential.

    The row gate, spelled here rather than through `require_owner_or_admin`
    for two deliberate differences, both named in the refusal:

    * the message says which `[db.<name>]` binding and which database was
      refused. The generic row denial is byte-identical to the one the app's
      own ownership check raises, so a caller could not tell which of the two
      failed — and the binding it names is something the caller just sent.
    * a **NULL-owner** row is bindable, where `require_owner_or_admin` makes it
      admin-only. A database with no `submitted_by_token` was created by the
      legacy global token or in local mode — the single-operator install, where
      the operator's own scoped CI token and the permanently-`submitter` tunnel
      principal are the normal callers. Admin-only there breaks the shipped
      flow (`nerdit db create pg` from the shell, then an app binding it) and
      its only workaround is minting admin tokens, which is worse than the
      boundary it enforces. The escalation this gate exists to stop is one
      submitter reaching ANOTHER submitter's database; that is still refused.
    """
    principal = current_principal(request)
    if principal.role is TokenRole.admin:
        return
    owner = getattr(row, "submitted_by_token", None)
    if owner is None or owner == principal.token_id:
        return
    raise NerditError(
        403,
        "forbidden",
        f"[db.{binding_name}] names database '{database}', which belongs to another token.",
        hint=("Bind a database this token created, or ask an admin to write the binding for you."),
    )


async def require_provisioned_databases(  # noqa: ANN001
    queries,
    db_bindings: dict[str, DbBindingConfig],
    request: Request,
    *,
    previous: dict | None = None,
) -> None:
    """Require a non-terminal, caller-authorized database row per managed binding.

    Building rows are accepted; launch-time resolution checks readiness.
    External bindings need no local row.

    A managed binding hands the app the database's MINTED password at launch
    (`DATABASE_URL`), so a binding the caller INTRODUCES OR CHANGES is gated:
    `require_service_scope` on its name and `_require_bindable_database` on the
    row, both before the caller's write. The existence gate runs first so a name
    that resolves to nothing stays a 422 (`GET /databases` is any-authenticated,
    so existence is not a secret). `request` is mandatory: a principal-free call
    site would be the hole.

    `previous` is the app's already-persisted `config['db']` table. A binding
    whose spec is byte-identical to it is **carried forward**, not authored, and
    is therefore existence-checked but not re-authorized. Without that, a
    foreign binding (admin-authored, or persisted before this gate existed)
    would lock the app's own owner out of every later `deploy`/`ai` write and
    every redeploy whose `nerdit.toml` still names it — while a toml-silent
    redeploy carried the same binding forward untouched, which made the refusal
    inconsistent as well as wrong.
    """
    prior = previous or {}
    for binding_name, binding in db_bindings.items():
        if binding.provider != "managed":
            continue
        # A managed binding always carries `database` (DbBindingConfig's model
        # validator); `or ""` only narrows the Optional, and an empty ref
        # matches no row, so the 422 below still fires.
        database = binding.database or ""
        row = await queries.get_resource_by_ref(database, JobKind.database)
        if row is None or row.kind != JobKind.database or row.status in DB_TERMINAL_STATUSES:
            raise NerditError(
                422,
                "db.not_provisioned",
                f"[db.{binding_name}] needs database '{database}', but no active "
                f"database workload is serving it.",
                hint="run 'nerdit db create <backend>' first",
            )
        if binding.model_dump(exclude_none=True) == prior.get(binding_name):
            continue  # carried forward unchanged — not an authoring act
        # (D-P40-7) A database row carries no project, so this stays label-only in
        # effect; passing the row's own field keeps every row-in-hand site uniform.
        require_service_scope(request, database, project=row.project)
        _require_bindable_database(request, binding_name, database, row)


async def validate_app_config(
    effective_doc: dict,
    queries,  # noqa: ANN001
    request: Request,
    *,
    previous_db: dict | None = None,
) -> tuple[DeployConfig, dict[str, AiBindingConfig] | None]:
    """Validate an app's effective config document end to end.

    `effective_doc` is `{"deploy": {...}, "ai": {...}?}` with the deploy
    values already merged to their effective precedence by the caller. Runs the
    `DeployConfig` field validators, the `[ai.*]` binding schema, and
    the served-model gate; raises the same structured 422s as the deploy route.
    `request` carries the principal the `[db.*]` gate authorizes against, and
    `previous_db` the app's persisted `config['db']` so a carried-forward
    binding is not re-authorized as though the caller had just written it.
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
            await require_provisioned_databases(queries, db_bindings, request, previous=previous_db)
    return deploy_cfg, bindings
