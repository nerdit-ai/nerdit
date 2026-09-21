"""The merged variable reader: project scope under service scope (P40c / D-P40-9).

One store, one flag (D-P40-1): every value, plain or secret, lives in a
scope's encrypted file and is read through `SecretManager.load`; this module
only composes two loads. The machine scope (`_shared`) is never merged here
-- it stays ref-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nerdit.core.secrets import SecretManager, project_storage_name

if TYPE_CHECKING:
    from nerdit.db.queries import Queries

Winner = Literal["project", "service"]


def load_scoped(
    mgr: SecretManager,
    label: str | None,
    project_id: str | None,
    *,
    include_service: bool = True,
    include_project: bool = True,
) -> tuple[dict[str, str], dict[str, Winner]]:
    """Load a service's variables merged `project < service`.

    The most specific scope wins a key regardless of its plain/secret flag.
    `include_project` is the authorization gate on caller-driven paths: pass
    `False` unless the project is provably the caller's (or the caller is
    admin), so a non-owner can never have the daemon resolve another token's
    project value into something it controls. `include_service` is the same
    gate one scope down (the D-P39-4 carve-out).

    Args:
        mgr: The secret store both scopes live in.
        label: The service's `jobs.service_name`; `None` skips the service scope.
        project_id: The row's project id; `None` (model/database/legacy rows)
            degrades to service-only.
        include_service: Whether the service scope may be read.
        include_project: Whether the project scope may be read.

    Returns:
        `(env, winners)`: the merged `{KEY: value}` map and, per key, the
        scope that supplied it. `env` holds secret values -- never log it.

    Raises:
        SecretDecryptError: When an existing scope file cannot be decrypted.
    """
    env: dict[str, str] = {}
    winners: dict[str, Winner] = {}
    if include_project and project_id is not None:
        env.update(mgr.load(project_storage_name(project_id)))
        winners.update(dict.fromkeys(env, "project"))
    if include_service and label:
        service_env = mgr.load(label)
        env.update(service_env)
        winners.update(dict.fromkeys(service_env, "service"))
    return env, winners


async def plain_keys(queries: Queries, project_id: str, service: str | None = None) -> set[str]:
    """The keys flagged plain in one scope; every other key is secret (D-P40-1).

    Args:
        queries: The flag table's reader.
        project_id: The owning project's id.
        service: The service name, or `None` for the project scope.
    """
    scope = service or ""  # '' is the project-scope storage sentinel
    return {
        flag.key
        for flag in await queries.list_variable_flags(project_id)
        if flag.plain and flag.service == scope
    }
