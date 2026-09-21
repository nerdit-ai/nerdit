"""Authorize and write a service's secret scope, with or without a row (P39).

A name with no `jobs` row is authorized through its `secret_claims` row: the
first write mints a claim for the caller, and every later access is judged
against it exactly as `require_owner_or_admin` judges a row. The claim is
consumed by the row insert (`reserve_service_for_token`) and dropped by a
delete-all. Every message here is value-free by construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable

from fastapi import FastAPI, Request

from nerdit.core.project_identity import DEFAULT_SERVICE, PRODUCTION, service_label
from nerdit.core.secrets import (
    _DNS_LABEL_RE,
    SHARED_SCOPE,
    InvalidSecretKey,
    InvalidSecretValue,
    InvalidServiceName,
    SecretDecryptError,
    SecretManager,
    project_storage_name,
    validate_secret_items,
)
from nerdit.daemon.audit import audit_params
from nerdit.daemon.auth import (
    Principal,
    current_principal,
    owner_denial,
    require_owner_or_admin,
    require_project_owner_or_admin,
    require_role,
    require_service_scope,
)
from nerdit.daemon.errors import NerditError
from nerdit.db.models import Project, TokenRole
from nerdit.db.queries import ProjectExists, ServiceNameClaimed, ServiceNameTaken

# User-facing name of the shared (global) secrets scope. Translated to the
# internal storage name (`SHARED_SCOPE`); a service can never take this name
# (`service.reserved_name` on create/deploy).
_SHARED_PUBLIC = "shared"


def secret_manager(request: Request) -> SecretManager:
    """The daemon's `SecretManager`, or a structured 500 when unwired."""
    mgr = getattr(request.app.state, "secret_manager", None)
    if mgr is None:  # pragma: no cover - always wired in the daemon
        raise NerditError(500, "internal", "Secret manager is not configured.")
    return mgr


def variable_write_lock(app: FastAPI) -> asyncio.Lock:
    """The one lock every (flag row, secrets file) pair is written under (D-P40-1).

    `_write_variables` orders flag and file so a CRASH fails safe; only a lock
    makes that order hold against a second writer (a plain and a secret set of
    one key interleaving to file=secret, flag=plain) or a backup snapshotting
    the two stores either side of a flip. Lives on `app.state`, not the module:
    an `asyncio.Lock` binds to the first loop it is contended on.
    """
    # ponytail: one global lock; per-project locks if write throughput ever matters.
    lock = getattr(app.state, "variable_write_lock", None)
    if lock is None:
        lock = app.state.variable_write_lock = asyncio.Lock()
    return lock


def secret_call(fn, *args):
    """Run a SecretManager operation, mapping its errors to the envelope.

    Every CRUD handler reaches `load()` internally (set/delete merge the
    existing file), so a corrupt or wrong-key `.enc` can surface anywhere —
    the structured 500 keeps the kid-bearing restore hint reaching the caller
    instead of a bare Starlette 500. Messages are hygienic by construction
    (paths, service names and kids only).
    """
    try:
        return fn(*args)
    except InvalidServiceName as exc:
        raise NerditError(422, "secret.invalid_service", str(exc)) from exc
    except InvalidSecretKey as exc:
        raise NerditError(
            422,
            "secret.invalid_key",
            str(exc),
            hint="Key names are env-var names: letters, digits and '_', not starting with a digit.",
        ) from exc
    except InvalidSecretValue as exc:
        raise NerditError(
            422,
            "secret.invalid_value",
            str(exc),
            hint="Values may not contain NUL or control characters (tab/newline/CR are allowed).",
        ) from exc
    except SecretDecryptError as exc:
        raise NerditError(500, "secret.decrypt_failed", str(exc)) from exc


def storage_name(service: str) -> str:
    """Translate the user-facing `shared` scope to its internal storage name."""
    return SHARED_SCOPE if service == _SHARED_PUBLIC else service


def guard_name(service: str) -> None:
    """422 on a name that is not a DNS label (before any lookup)."""
    if not _DNS_LABEL_RE.match(service):
        raise NerditError(
            422,
            "secret.invalid_service",
            f"Invalid service name '{service}'.",
            hint="Service names are DNS labels (lowercase, digits, '-').",
        )


def name_claimed_error(name: str) -> NerditError:
    """The 409 a fresh row meets when another token set the name's secrets first."""
    return NerditError(
        409,
        "service.name_claimed",
        f"Secrets for '{name}' were set by another token.",
        hint=(
            "Ask that token's owner to deploy, or an admin to remove the claim with "
            f"`nerdit secrets rm {name}`."
        ),
    )


def project_owned_error(name: str) -> NerditError:
    """The 409 a fresh row meets when another token owns the project it would join.

    P40b / D-P40-5 rule 1, the sibling of `name_claimed_error`: value-free, it
    names the project and the release command, never the owner.
    """
    return NerditError(
        409,
        "project.owned",
        f"Project '{name}' belongs to another token.",
        hint=(
            "Ask that project's owner, or an admin, to release it with "
            f"`nerdit projects delete {name}`."
        ),
    )


def _claim_forbidden() -> NerditError:
    # The same envelope as `require_owner_or_admin`'s denial: a claim is judged
    # like a row, and a distinct message would be an oracle for "claimed, not
    # deployed".
    return owner_denial()


def _claim_is_callers(principal: Principal, token_id: str | None) -> bool:
    """A NULL claimant is admin-only, like a NULL-owner row."""
    return token_id is not None and token_id == principal.token_id


async def claim_owned_by_caller(request: Request, name: str) -> bool:
    """Whether a claim on `name` exists and belongs to the caller (or the caller is admin)."""
    claim = await request.app.state.queries.get_secret_claim(name)
    if claim is None:
        return False
    principal = current_principal(request)
    return principal.is_admin or _claim_is_callers(principal, claim.token_id)


def project_owned_by_caller(request: Request, project: Project | None) -> bool:
    """Whether `project` exists and belongs to the caller (or the caller is admin).

    The `claim_owned_by_caller` posture one scope up: this is the
    `include_project` gate of `core.variables.load_scoped` on caller-driven
    paths. A NULL-owner project is admin-only.
    """
    if project is None:
        return False
    principal = current_principal(request)
    return principal.is_admin or _claim_is_callers(principal, project.submitted_by_token)


async def reject_foreign_claim(request: Request, name: str) -> None:
    """Pre-ingress fast path: 409 before an upload or clone is spent on a reserved name.

    Judges the P39 claim, then the P40b project NAMED after the label (for a
    P40d composed label that is a foreign implicit project literally so named,
    never the project it joins -- `apply_project` judges that one itself), in
    the order `reserve_service_for_token` judges them. An
    optimization only — the reserve transaction re-checks both and is the
    security boundary.
    """
    principal = current_principal(request)
    if principal.is_admin:
        return
    queries = request.app.state.queries
    claim = await queries.get_secret_claim(name)
    if claim is not None and not _claim_is_callers(principal, claim.token_id):
        raise name_claimed_error(name)
    project = await queries.get_project_by_name(name)
    if project is not None and not _claim_is_callers(principal, project.submitted_by_token):
        raise project_owned_error(name)


async def authorize_secret_scope(
    request: Request, service: str, *, write: bool, mint: bool = False
) -> None:
    """Authorize secret access on `service`, checking scope before any lookup.

    A row is judged by `require_owner_or_admin`. A rowless name is judged by
    its claim: the caller's (or an admin) passes; another token's is the same
    403 as a foreign row. With no claim, a set (`mint=True`) reserves the name
    for the caller — unless a pre-P39 orphan file is present, which a non-admin
    may not adopt (409 `secret.orphaned_scope`) — and a read or delete is a
    404 for non-admins. `write` is the mutation gate on `shared`, which skips
    row lookup: admins write, any authenticated caller reads names; refuse all
    shared operations if a legacy service owns that name.
    """
    guard_name(service)
    if service == _SHARED_PUBLIC:
        if getattr(request.app.state, "shared_scope_blocked", False):
            raise NerditError(
                409,
                "secret.shared_unavailable",
                "A service named 'shared' predates the reserved shared scope; "
                "shared-scope secrets are disabled.",
                hint="Rename or delete that service, then restart the daemon.",
            )
        if write:
            require_role(request, TokenRole.admin)
        return
    # (D-P40-7) Label-only: scope is judged before any lookup, so there is no row
    # whose project could widen it (the ceiling is named at `rollback`).
    require_service_scope(request, service)
    queries = request.app.state.queries
    existing = await queries.get_service_by_name(service)
    if existing is not None:
        require_owner_or_admin(request, existing)
        return
    principal = current_principal(request)
    claim = await queries.get_secret_claim(service)
    if claim is not None:
        if principal.is_admin or _claim_is_callers(principal, claim.token_id):
            return
        raise _claim_forbidden()
    if not mint:
        if principal.is_admin:
            return
        raise NerditError(
            404,
            "not_found",
            f"No service '{service}'.",
            hint=f"Set a secret first with `nerdit secrets set {service} KEY=...`, "
            "or deploy the service.",
        )
    # A leftover file from a deleted service (pre-P39, or purged without
    # `secrets`) must not be adopted by a stranger: its values would launch
    # with their fresh deploy.
    if not principal.is_admin and secret_call(secret_manager(request).exists, service):
        # The file may be a concurrent same-token write rather than an orphan:
        # the claim read above can predate its mint.
        if await _landed_is_callers(request, principal, service):
            return
        raise NerditError(
            409,
            "secret.orphaned_scope",
            f"An earlier service's secrets remain for '{service}'.",
            hint=f"An admin can remove them with `nerdit secrets rm {service}`.",
        )
    # The mint refuses a claim OR a row that landed after the reads above, in
    # one statement under the write lock (an admin bypasses the claim anyway).
    won = await queries.mint_secret_claim(service, principal.token_id, admin=principal.is_admin)
    if won:
        request.state.audit_params = audit_params(
            {**getattr(request.state, "audit_params", {}), "claimed": True}
        )
        return
    if not principal.is_admin and not await _landed_is_callers(request, principal, service):
        raise _claim_forbidden()


async def _landed_is_callers(request: Request, principal: Principal, service: str) -> bool:
    """Re-judge a name that gained a row, a claim or a foreign project during this request.

    A same-token concurrent write (a retrying agent, two parallel `nerdit
    secrets set`) must merge, not be refused as foreign: a row is judged by
    `require_owner_or_admin`, a claim by the claim rule, and nothing landed
    (the loser of a claim that was deleted again) is `False` — fail closed.
    A `projects` row of that name owned by another token (P40b / D-P40-5
    rule 2 — the mint's own predicate refused it) is the same 403 as a foreign
    claim, so a stranger cannot tell "claimed" from "project reserved".
    """
    queries = request.app.state.queries
    existing = await queries.get_service_by_name(service)
    if existing is not None:
        require_owner_or_admin(request, existing)
        return True
    project = await queries.get_project_by_name(service)
    if project is not None and not _claim_is_callers(principal, project.submitted_by_token):
        raise _claim_forbidden()
    claim = await queries.get_secret_claim(service)
    if claim is None:
        return False
    if _claim_is_callers(principal, claim.token_id):
        return True
    raise _claim_forbidden()


async def set_secret_values(request: Request, service: str, values: dict[str, str]) -> list[str]:
    """Validate, authorize (minting a claim on a rowless name) and merge `values`.

    Returns the resulting key names. Validation runs before authorization so a
    bad item never mints a claim.
    """
    secret_call(validate_secret_items, values)
    # Authorize INSIDE the lock: a verdict reached while waiting (a backup holds
    # it for minutes) could outlive the claim it judged — the owner deletes the
    # scope, a stranger claims the name, and the stale write lands in their file.
    async with variable_write_lock(request.app):
        await authorize_secret_scope(request, service, write=True, mint=True)
        await demote_flags(request, service, values)
        return secret_call(secret_manager(request).set, storage_name(service), values)


async def demote_flags(request: Request, service: str, keys: Iterable[str]) -> None:
    """Flag `keys` secret before a secret-only write to a label's file lands (D-P40-1).

    `/secrets` and the database credential mint only ever write secrets, so a
    key the variables API once flagged plain must stop being listable BEFORE
    its new value is on disk: flag first, then file, the secret half of
    `_write_variables`' order. Call it under `variable_write_lock`.
    """
    if service == _SHARED_PUBLIC:
        return
    queries = request.app.state.queries
    row = await queries.get_service_by_name(service)
    project_id = row.project_id if row is not None else None
    scope = (row.service if row is not None else None) or DEFAULT_SERVICE
    if project_id is None:
        # Rowless, or a model/database row (never project-stamped): the only
        # flags such a label can carry were written through the project of the
        # same NAME as its `web` label (label == project) -
        # `require_service_in_project` admits no other unstamped label.
        project = await queries.get_project_by_name(service)
        project_id, scope = (project.id if project is not None else None), DEFAULT_SERVICE
    if project_id is not None:
        await queries.upsert_variable_flags(project_id, scope, keys, False)


# --- Variables (P40c): the same store one noun up, plus a plain/secret flag ---


def name_taken_error(name: str) -> NerditError:
    """The 409 `create_project` meets when a foreign model/database row holds the label."""
    return NerditError(
        409,
        "service.name_taken",
        f"A service named '{name}' already exists.",
        hint="Choose a different name, or ask that service's owner (or an admin) to remove it.",
    )


async def create_project_for_caller(request: Request, name: str) -> Project:
    """Create project `name` for the caller, mapping the D-P40-5 rule-3 refusals.

    Shared by `POST /projects` and a variable set on an absent project, so both
    refuse a foreign claim or row with one envelope. The owner is the caller's
    token and `admin` comes from the real principal.

    Raises:
        ProjectExists: Left to the caller; a create 409s, a variable set re-judges the row.
        NerditError: 409 `service.name_claimed` / `service.name_taken`.
    """
    principal = current_principal(request)
    try:
        return await request.app.state.queries.create_project(
            name, principal.token_id, admin=principal.is_admin
        )
    except ServiceNameClaimed as exc:
        raise name_claimed_error(name) from exc
    except ServiceNameTaken as exc:
        raise name_taken_error(name) from exc


async def judge_project(request: Request, name: str) -> Project | None:
    """Scope before any lookup, then the owner gate on an existing project.

    Returns the row (the caller's, or any for an admin) or `None` when absent.
    A foreign project is the one `owner_denial` 403, judged on
    `projects.submitted_by_token` (NULL owner admin-only), never the actor.
    """
    require_service_scope(request, name)
    project = await request.app.state.queries.get_project_by_name(name)
    if project is not None:
        require_project_owner_or_admin(request, project)
    return project


async def _project_for_write(request: Request, name: str, project: Project | None) -> Project:
    """The judged project, created for the caller when absent (D-P40-5 rule 3)."""
    if project is not None:
        return project
    try:
        return await create_project_for_caller(request, name)
    except ProjectExists:
        # Lost a create race: judge whoever won exactly like a row found up front.
        landed = await judge_project(request, name)
        if landed is None:
            raise owner_denial() from None  # created and deleted again: fail closed
        return landed


async def _write_variables(
    request: Request,
    project_id: str,
    service: str | None,
    storage: str,
    values: dict[str, str],
    plain: bool,
) -> list[str]:
    """Write one scope's values and flags in the order that fails safe (D-P40-1).

    `list_variables` shows a value only while its flag row says plain, so the
    flag must say secret whenever the file may hold a secret. A secret write
    therefore lands its flag FIRST (a crash leaves the old plain value hidden);
    a plain write lands the file FIRST (a crash leaves the new plain value
    hidden, never the old secret shown). Setting a key flips its flag.
    """
    queries = request.app.state.queries
    # The order below only fails safe against a crash; the caller's
    # `variable_write_lock` (held from its authorization on) is what makes it
    # hold against a concurrent writer of the opposite flag.
    if not plain:
        await queries.upsert_variable_flags(project_id, service, values, False)
    keys = secret_call(secret_manager(request).set, storage, values)
    if plain:
        await queries.upsert_variable_flags(project_id, service, values, True)
    return keys


async def require_service_in_project(
    request: Request, row: Project | None, project: str, service: str, label: str
) -> None:
    """404 unless `label` is the project's `web` label or a row inside the judged project.

    A composed label (`api--asso`) is only addressable through a row that maps
    it back to this project: a legacy service literally named like one belongs
    to its own implicit project, and a rowless one has no mapping at all.
    Runs after `judge_project`, so the caller already passed scope and owner.
    A model/database row named like the project is NOT its `web` service (its
    `project_id` is NULL): its file holds a minted credential, which must never
    become addressable - and so listable - as a project variable.
    """
    svc = await request.app.state.queries.get_service_by_name(label)
    if label == project and svc is None:
        return
    if svc is None or row is None or svc.project_id != row.id:
        # ponytail: no variables on a composed label before its first deploy
        # (`demote_flags` could not map a rowless one back to its flags). Kept
        # in P40d: `apply_project` creates the composed row, after which the
        # label is addressable; a service that NEEDS a service-scope value at
        # first launch takes it at project scope, or is applied once first.
        # Lifting it means a label -> (project, service) map for rowless labels.
        raise NerditError(
            404,
            "not_found",
            f"No service '{service}' in project '{project}'.",
            hint="Deploy the service first, or set the variable at project scope.",
        )


async def set_project_values(
    request: Request,
    project: str,
    values: dict[str, str],
    plain: bool,
    *,
    check_new_name: Callable[[str], None],
) -> list[str]:
    """Validate, authorize and merge `values` into a project's own scope.

    Scope, then the row: its owner or an admin, else the row 403; an absent
    project is created for the caller. Never touches `_shared` or a label.

    Args:
        request: The caller's request.
        project: The project name.
        values: `{KEY: value}`; never logged, audited or returned.
        plain: Whether the owner may read the values back.
        check_new_name: Raises the 422 for a name that may not become a project.

    Returns:
        The scope's resulting key names.
    """
    secret_call(validate_secret_items, values)
    async with variable_write_lock(request.app):  # verdict and write are one section
        row = await judge_project(request, project)
        if row is None:
            check_new_name(project)  # grammar + reserved names bind only where a NEW name enters
        row = await _project_for_write(request, project, row)
        return await _write_variables(
            request, row.id, None, project_storage_name(row.id), values, plain
        )


async def set_service_values(
    request: Request,
    project: str,
    service: str,
    values: dict[str, str],
    plain: bool,
    *,
    check_new_name: Callable[[str], None],
) -> list[str]:
    """Validate, authorize and merge `values` into one service's scope.

    The label runs the whole `authorize_secret_scope` chain `POST /secrets`
    runs (guard, scope, row owner, claim, mint, orphan 409), so a rowless label
    mints the P39 claim. The project is judged FIRST: the claim rule alone
    cannot see that a composed label sits inside another token's project.

    Args:
        request: The caller's request.
        project: The project name.
        service: The service name inside the project; the route has already
            proven `service_label(project, production, service)` composes.
        values: `{KEY: value}`; never logged, audited or returned.
        plain: Whether the owner may read the values back.
        check_new_name: Raises the 422 for a name that may not become a project.

    Returns:
        The scope's resulting key names.
    """
    label = service_label(project, PRODUCTION, service)
    secret_call(validate_secret_items, values)
    async with variable_write_lock(request.app):  # verdict and write are one section
        row = await judge_project(request, project)
        if row is None:
            check_new_name(project)  # before the mint: a refused name must not leave a claim
        await require_service_in_project(request, row, project, service, label)
        await authorize_secret_scope(request, label, write=True, mint=True)
        row = await _project_for_write(request, project, row)
        return await _write_variables(request, row.id, service, label, values, plain)


async def service_scope_readable(request: Request, label: str) -> bool:
    """Whether the caller may read `label`'s scope: the `/secrets` read rule, quiet on absence.

    A foreign row or claim stays the 403; a rowless, unclaimed label (a 404 on
    `/secrets`) is simply not read, so an orphan file is never listed to a
    non-admin.
    """
    try:
        await authorize_secret_scope(request, label, write=False)
    except NerditError as exc:
        if exc.status_code == 404:
            return False
        raise
    return True
