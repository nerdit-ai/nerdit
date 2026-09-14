"""Agent-workspace MCP tool implementations.

Thin loopback projections of the four ``/api/workspaces`` routes — the tools
never touch the filesystem themselves (in the remote topology the tool body runs
*inside* the daemon process, so a direct write would bypass scoped-token authz,
idempotency and audit). The two-step idiom the docstrings teach is
``write_app_files`` → ``deploy_app``; the edit loop is "patch one file →
``deploy_app`` again".
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _bad_request, _call
from nerdit.mcp.tools._shared import (
    BuildSettings,
    DeployEnv,
    DeployGpus,
    DeployHealth,
    DeployPort,
    DeployStart,
    DeployVendor,
    DryRun,
    IdempotencyKey,
    apply_sandbox_note,
)
from nerdit.mcp.transport import _request_client

# --- parameter vocabulary local to the workspace tools ----------------------
#
# A workspace is not a service row: it exists before any app does and outlives
# every deploy, so the shared ``AppName`` alias does not state its contract.

WorkspaceName = Annotated[
    str,
    Field(
        description="Workspace name, a DNS label (lowercase a-z0-9 and ``-``, 1-63 "
        "chars). A new name creates the workspace; an existing one is edited in "
        "place. It is also the app name ``deploy_app`` deploys it under."
    ),
]
ExistingWorkspaceName = Annotated[
    str,
    Field(
        description="Name of an existing workspace (also the app name it deploys "
        "under). No workspace by that name is ``404 workspace.not_found``."
    ),
]
WorkspaceFiles = Annotated[
    dict[str, str],
    Field(
        description="Workspace-relative path -> that file's full UTF-8 text; a write "
        "replaces the file whole. Paths are relative with no ``..``; ``.git``, "
        "``node_modules``, ``__pycache__``, ``.venv``, ``data/`` and ``*.pyc`` are "
        "refused, and ``.env`` / ``.env.*`` by name. Text only (NUL or non-UTF-8 is "
        "refused). Caps: 256 KiB per file, 10 MiB and 500 files per workspace."
    ),
]
WorkspaceDeletes = Annotated[
    list[str] | None,
    Field(
        description="Workspace-relative paths to remove, applied with the writes in "
        "the same all-or-nothing batch. Omit (null) to delete nothing; a path that "
        "does not exist is a no-op; a path may not be in both ``files`` and ``delete``."
    ),
]
WorkspaceFilePath = Annotated[
    str,
    Field(
        description="Workspace-relative path of one file, exactly as ``list_app_files`` "
        "reports it (relative, no ``..``). Its text is returned; ≤ 256 KiB by "
        "construction, since that is the per-file write cap. A path naming no "
        "file answers the same ``404 workspace.not_found`` as an absent workspace."
    ),
]


async def _write_app_files_impl(
    client: NerditClient,
    *,
    name: str,
    files: dict[str, str],
    delete: list[str] | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Write/delete a batch of workspace files, auto-minting a key.

    An empty request (no writes, no deletes) is refused client-side rather than
    creating an empty workspace by accident.
    """
    if not files and not delete:
        return _bad_request("files or delete must name at least one path")
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.write_workspace_files(name, files, delete, idempotency_key=idempotency_key)
    )


async def _list_app_files_impl(client: NerditClient, *, name: str) -> Any:
    """Return the workspace listing (paths, sizes, hashes, totals, owner meta)."""
    return await _call(client.get_workspace(name))


async def _read_app_file_impl(client: NerditClient, *, name: str, path: str) -> Any:
    """Return one workspace file's text (``_call`` passes non-dict values through)."""
    return await _call(client.read_workspace_file(name, path))


async def _deploy_app_impl(
    client: NerditClient,
    *,
    name: str,
    port: int | None = None,
    gpus: int | None = None,
    start: str | None = None,
    build_settings: dict[str, Any] | None = None,
    health: str | None = None,
    env: dict[str, str | None] | None = None,
    vendor: str | None = None,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Deploy the named workspace, auto-minting a key (never on a dry run)."""
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.deploy_workspace(
            name,
            port=port,
            gpus=gpus,
            start=start,
            build_settings=build_settings,
            health=health,
            env=env,
            vendor=vendor,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )
    )


# fmt: off
async def write_app_files(
        name: WorkspaceName,
        files: WorkspaceFiles,
        delete: WorkspaceDeletes = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Write files into an app's server-side workspace, then deploy it.

        The two-step idiom for building an app from this conversation:
        ``write_app_files`` (as many batches as you like) then ``deploy_app``.
        The workspace survives the deploy and IS the working copy, so editing is
        "write the one changed file → ``deploy_app`` again".

        One batch is all-or-nothing: an invalid entry rejects the whole request
        and nothing is written. Binary content is refused — deploy binaries from
        a repo with ``deploy_git``; assets normally come from the build
        (npm/pip). A secret belongs in ``set_secret``, which injects it at launch
        without it ever living in a file. WARNING: file contents you pass here
        transit this agent's transcript in plaintext; never write secrets into
        app files.

        {SANDBOX_NOTE}
        """
        return await _write_app_files_impl(
            _request_client(),
            name=name,
            files=files,
            delete=delete,
            idempotency_key=idempotency_key,
        )

async def list_app_files(name: ExistingWorkspaceName) -> Any:
        """List an app's workspace: paths, sizes, sha256 and totals.

        What this lists is exactly what ``deploy_app`` would build — the exclude
        rules are enforced at write time, not at deploy time. Bounded at 500
        files by construction, so there is no paging.
        """
        return await _list_app_files_impl(_request_client(), name=name)

async def read_app_file(name: ExistingWorkspaceName, path: WorkspaceFilePath) -> Any:
        """Read one file back from an app's workspace as text.

        Use it before patching a file you did not write in this conversation, so
        the next ``write_app_files`` replaces content you have actually seen.
        """
        return await _read_app_file_impl(_request_client(), name=name, path=path)

async def deploy_app(
        name: ExistingWorkspaceName,
        port: DeployPort = None,
        gpus: DeployGpus = None,
        start: DeployStart = None,
        build_settings: BuildSettings = None,
        health: DeployHealth = None,
        env: DeployEnv = None,
        vendor: DeployVendor = None,
        dry_run: DryRun = False,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Use when: you used write_app_files. Asynchronous — follow with wait_for_service.

        Distinct from ``deploy`` (which uploads a folder from the machine the MCP
        server runs on) and from ``deploy_git`` (which clones a repository): this
        one deploys the workspace of the same name, so it is the deploy verb for
        an app you wrote in this conversation. Call it again after each
        ``write_app_files`` to ship an edit — ``redeploy_service`` refuses
        workspace apps by design and points back here.

        If the workspace's ``nerdit.toml`` declares ``[ai.*]`` bindings, each is
        injected at launch as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the
        ``default`` binding also sets ``OPENAI_BASE_URL/OPENAI_API_KEY/
        OPENAI_MODEL``. A dry run returns the build/config plan diff with env key
        names only, values never, and mints no idempotency key.

        ``nerdit.toml`` ``[deploy]`` keys — anything else is IGNORED and
        reported back in the response ``hints``: ``name``, ``port``, ``gpus``,
        ``start``, ``health``, ``health_type`` (``http``|``tcp``),
        ``memory_limit``, ``cpu_limit``, ``volumes``, ``release``, ``cutover``,
        ``auto_deploy``, ``edge_auth``, ``build_settings``. Use ``[deploy.build_settings]``
        for build overrides, or a Dockerfile for custom builds.

        ``release`` runs once inside the new image before traffic moves; if it
        fails the deploy settles ``failed`` and the image reverts, never the data —
        whatever the migration already committed stays, so write idempotent
        migrations. ``cutover`` null = zero-downtime swap
        when eligible, false = same-port swap. ``env`` values are literal and
        never resolve ``${secrets.*}``: a secret reaches the app through
        ``set_secret`` / ``nerdit secrets set`` (injected at launch, overriding
        an ``env`` key of the same name).
        On a successful non-dry-run deploy the
        response carries ``summary`` (app/status/version/public_url), ``hints``
        (ordered one-liners, never empty) and ``next_step`` (the structured
        follow-up call); a ``dry_run`` plan carries none of the three — read a
        plan's advisories from its ``warnings``. On a node in ``path``
        proxy mode the app is served at ``/<name>/``, so a frontend must be
        built with that base path.

        ``build_settings`` overrides
        preset/install/build/start/node_version/package_manager/subdir/public_env.
        ``preset`` selects node/nextjs/python/dockerfile; null resets to
        repository/default detection.
        An existing Dockerfile retains precedence.
        Omit it to preserve saved overrides; a null field resets to repository/default,
        and ``build: false`` skips compilation. Commands run inside the build container.
        ``public_env`` passes public build-time variables (shown in clear in a dry
        run; secret references are refused). Secret mounts are unsupported;
        runtime env is unchanged.

        {SANDBOX_NOTE}
        """
        return await _deploy_app_impl(
            _request_client(),
            name=name,
            port=port,
            gpus=gpus,
            start=start,
            build_settings=build_settings,
            health=health,
            env=env,
            vendor=vendor,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )
# fmt: on


# (P33) See ``tools/deploy.py``: the shared sandbox sentence is stamped into the
# docstrings carrying the placeholder, since a docstring must be a literal.
TOOLS = tuple(
    apply_sandbox_note(fn)
    for fn in (
        write_app_files,
        list_app_files,
        read_app_file,
        deploy_app,
    )
)
