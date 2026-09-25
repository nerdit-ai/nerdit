"""Project MCP tool implementations (P40b).

A project is the grouping every deployed service row points at (P40a): the
services of one app (its ``web`` service and, from P40d, siblings), the
models and databases those services reference, and every advertised address.
Its row outlives its services and reserves the name for the creating token,
so the three D-P40-5 judgments compose with the P39 secret claim.

Thin loopback projections of ``GET``/``POST /api/projects`` and
``GET``/``DELETE /api/projects/{project}`` (Invariant #3): grammar, reserved
names, scope, ownership and the delete cascade are all decided by the daemon
and surface as its structured envelope. Reads, apply and variables also accept
immutable project IDs; only names may create a project. Workspace file tools
keep their name identity. No argument is named ``environment`` (D-P40-11).

P40c adds the project's variables: ``set_variable`` (write-only unless
``secret=false``, D-P40-16) and ``resolve_variables`` (names and winning
scopes, never a value). Neither can address the machine scope; that stays
``set_secret("shared", …)``.

P40d adds the declaration: ``write_project_files`` (the project's one source
tree, a P29 workspace named after the project) and ``apply_project``, which
applies the ``[project]`` / ``[services.*]`` / ``[vars]`` of that tree's (or a
repository's) ``nerdit.toml``.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from pydantic import Field

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _call, _clamp
from nerdit.mcp.tools._shared import (
    DEFAULT_DIAGNOSE_LOG_TAIL,
    DEFAULT_LOG_TAIL,
    MAX_DIAGNOSE_LOG_TAIL,
    MAX_LOG_TAIL,
    Cursor,
    DryRun,
    IdempotencyKey,
    apply_sandbox_note,
)
from nerdit.mcp.tools.workspaces import WorkspaceDeletes, WorkspaceFiles, _write_app_files_impl
from nerdit.mcp.transport import _request_client

# Projects page like services (GET /api/projects caps its ``limit`` at ≤200).
DEFAULT_PROJECT_LIMIT = 50
MAX_PROJECT_LIMIT = 200

_ProjectName = Annotated[
    str,
    Field(
        description="Immutable prj_ ID or name of an existing project. Prefer the ID "
        "from list_projects: a retired ID never addresses a same-name replacement."
    ),
]
ProjectLimit = Annotated[
    int | None,
    Field(description="Projects per page; omit for 50. Clamped to 200, the daemon's own page cap."),
]


async def _create_project_impl(
    client: NerditClient, name: str, *, idempotency_key: str | None = None
) -> Any:
    """Create an empty project, auto-minting a key (the write-tool idiom)."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.create_project(name, idempotency_key=idempotency_key))


async def _list_projects_impl(
    client: NerditClient, *, limit: int | None = None, cursor: str | None = None
) -> Any:
    """List projects, bounded to ``limit`` per page (server-enforced ``?limit=``)."""
    bound = _clamp(limit if limit is not None else DEFAULT_PROJECT_LIMIT, MAX_PROJECT_LIMIT)
    return await _call(client.list_projects(limit=bound, cursor=cursor))


async def _get_project_impl(client: NerditClient, name: str) -> Any:
    """Read one project: services, referenced resources, addresses, home node."""
    return await _call(client.get_project(name))


async def _delete_project_impl(
    client: NerditClient, name: str, *, purge: str = "secrets", idempotency_key: str | None = None
) -> Any:
    """Delete a project and every service in it, auto-minting a key."""
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.delete_project(name, purge=purge, idempotency_key=idempotency_key))


async def _set_variable_impl(
    client: NerditClient,
    project: str,
    values: dict[str, str],
    *,
    secret: bool | None = None,
    service: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Set/merge variables in one scope; returns key names only.

    D-P40-16: anything but an explicit ``False`` is sent as ``secret=true``,
    so a forgotten flag can never make a value readable.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.set_variables(
            project,
            values,
            secret=secret is not False,
            service=service,
            idempotency_key=idempotency_key,
        )
    )


async def _resolve_variables_impl(
    client: NerditClient, project: str, *, service: str | None = None
) -> Any:
    """Per key, the scope a launch would take it from (values are never returned)."""
    return await _call(client.resolve_variables(project, service=service))


async def _apply_project_impl(  # noqa: PLR0913 - mirrors the REST payload
    client: NerditClient,
    project: str,
    *,
    repo_url: str | None = None,
    ref: str | None = None,
    token_ref: str | None = None,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Apply the declaration of a repository, else of the project's workspace.

    No key is minted or needed on a dry run (the ``deploy_app`` precedent).
    """
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())

    async def _apply() -> Any:
        # No repository: the daemon snapshots the workspace under its lock.
        return await client.apply_project(
            project,
            workspace=not repo_url,
            repo_url=repo_url,
            ref=ref,
            token_ref=token_ref,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

    return await _call(_apply())


# fmt: off
async def create_project(
        name: Annotated[
            str,
            Field(
                description="New project name: a lowercase DNS label of at most 40 "
                "characters (letters, digits, '-'), never containing '--'; reserved "
                "platform names are refused with 422."
            ),
        ],
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Create an empty project, reserving its name for your token.

        A project groups one app's services, the models and databases they
        reference, and their public addresses. The name is reserved for the
        token that created it: another token's deploy, ``serve_model`` or
        ``create_database`` of that name is refused with 409 ``project.owned``
        (an admin bypasses), and the project remains after ``remove_service``
        until ``delete_project`` releases it. You rarely need this before a
        deploy — every deployed app already becomes a project of its own name —
        but it lets you hold a name ahead of the first deploy. A name another
        token already reserved through ``set_secret`` is 409
        ``service.name_claimed``; a name in use is 409 ``project.exists``.
        """
        return await _create_project_impl(
            _request_client(), name, idempotency_key=idempotency_key
        )

async def list_projects(limit: ProjectLimit = None, cursor: Cursor = None) -> Any:
        """List the projects this token can see, bounded to ``limit`` per page.

        Each entry carries the project's services (state, endpoint) and every
        advertised address. A scoped token sees only the projects its scope
        names. Cursor-paginated; use ``get_project`` for the full picture of one.
        """
        return await _list_projects_impl(_request_client(), limit=limit, cursor=cursor)

async def get_project(name: _ProjectName) -> Any:
        """The one-call picture of a project: read this before you deploy or diagnose.

        Returns the project's services with their state, last deploy and
        endpoint; the models and databases they reference with a readiness
        flag (``null`` when readiness cannot be judged without a secret); every
        public address under ``addresses``; and ``home``, the node it lives on
        (always this node today). A name outside your scope is 403; an unknown
        name is 404. Secret values never appear here.
        """
        return await _get_project_impl(_request_client(), name)

async def delete_project(
        name: _ProjectName,
        purge: Annotated[
            str,
            Field(
                description="CSV of durable state to destroy alongside EVERY service in "
                "the project: ``secrets``, ``data``, ``images``, ``workspace``. Default "
                "``secrets``; an empty string purges nothing; an unknown token is a 422 "
                "``service.invalid_purge``."
            ),
        ] = "secrets",
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Delete a project: every service in it, then the row, releasing the name.

        Services go one by one through the same cascade as ``remove_service``
        (owner-or-admin each; a dependency guard or an in-flight run refuses a
        service exactly as there). The project row goes last and only when
        every service went: a refusal leaves the project in place with 409
        ``project.delete_incomplete`` listing ``deleted`` and ``failed``, and
        re-running after the fix is idempotent. Owner or admin only.
        """
        return await _delete_project_impl(
            _request_client(), name, purge=purge, idempotency_key=idempotency_key
        )

async def set_variable(
        project: Annotated[
            str,
            Field(
                description="Immutable prj_ ID or name of the project to modify. An unknown "
                "name creates it for your token; an unknown or retired ID is refused."
            ),
        ],
        values: Annotated[
            dict[str, str],
            Field(
                description="KEY → value pairs merged into the scope: a listed key is "
                "overwritten, an unlisted one kept. Key names are env-var identifiers "
                "(``[A-Za-z_][A-Za-z0-9_]*``); a value may be multi-line but may contain no "
                "NUL and no control character other than tab/newline/CR. An empty map is "
                "refused."
            ),
        ],
        secret: Annotated[
            bool | None,
            Field(
                description="Omit (or ``true``) for a write-only SECRET: no API, tool or "
                "dashboard ever returns the value again. ``false`` is the explicit PLAIN "
                "write: the project's owner or an admin can read the value back. When "
                "unsure, omit it."
            ),
        ] = None,
        service: Annotated[
            str | None,
            Field(
                description="Service inside the project (``web`` for a single-app "
                "project) to set the key for that service only; omit for the project "
                "scope, shared by every service of the project. On a key set in both, "
                "the service scope wins."
            ),
        ] = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Set/merge a project's variables; write-only unless ``secret=false``. Returns names only.

        WARNING: the values you pass here transit this agent's transcript in
        plaintext, whether ``secret`` is omitted, ``true`` or ``false``. For any
        credential, do not pass the value at all: ask the human to run
        ``nerdit vars set <project> --secret --prompt KEY`` (hidden input) and
        only ever name ``KEY`` yourself. Use this tool for values that are fine
        to appear in a transcript (a log level, a feature flag, a public URL).

        A variable reaches a service's environment at launch. Precedence, lowest
        to highest: the manifest ``env`` table, the project scope, the service
        scope, then platform-injected bindings (``OPENAI_BASE_URL``,
        ``DATABASE_URL``…), which a variable can never override. Setting a key
        again flips its plain/secret flag in that scope. Only the project's owner
        or an admin may write (another token gets 403).
        A running service picks the change up on its next restart or redeploy.
        The machine-wide scope is not reachable here: use ``set_secret`` with
        ``shared``.
        """
        return await _set_variable_impl(
            _request_client(), project, values,
            secret=secret, service=service, idempotency_key=idempotency_key,
        )

async def resolve_variables(
        project: _ProjectName,
        service: Annotated[
            str | None,
            Field(
                description="Service inside the project whose launch to resolve for; "
                "omit for ``web``, the service of a single-app project."
            ),
        ] = None,
    ) -> Any:
        """What a launch would see: per key, the winning scope and whether it is plain. No values.

        The loop is "the agent asks, the human supplies". Call this before a
        deploy to see which keys the service will receive (``scope`` is
        ``project`` or ``production/<service>``). When a key the app needs is
        missing, do NOT invent or request the value in chat: ask the human to
        run ``nerdit vars set <project> --secret --prompt KEY``, then call this
        again to confirm the key is present. A secret value is never returned by
        any tool. Owner or admin only; a name outside your scope is 403.
        """
        return await _resolve_variables_impl(_request_client(), project, service=service)
async def write_project_files(
        project: Annotated[
            str,
            Field(
                description="Project name: the ONE source tree of the project, a workspace "
                "of the same name. A new name creates the workspace; an existing one is "
                "edited in place. It must equal ``[project].name`` in the tree's "
                "``nerdit.toml``."
            ),
        ],
        files: WorkspaceFiles,
        delete: WorkspaceDeletes = None,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Write a project's source tree (its ``nerdit.toml`` included), then ``apply_project``.

        Exactly ``write_app_files`` on the workspace named after the project:
        same all-or-nothing batch, same text-only rule, same caps, and
        ``list_app_files`` / ``read_app_file`` read it back under the project
        name. One tree holds EVERY service; a service built from a subfolder
        says so with ``build_settings.subdir``. The root ``nerdit.toml`` is the
        declaration::

            [project]
            name = "asso"

            [services.web]
            port = 3000

            [services.api]
            port = 8000
            build_settings = { subdir = "apps/api" }

            [vars]
            required = ["API_KEY"]

        ``[services.<name>]`` takes the ``[deploy]`` keys except ``name``: the
        service ``web`` is served under the project name (``asso``), any other
        under ``<service>--<project>`` (``api--asso``), and that label is the
        name every other tool takes. ``[ai.*]`` / ``[db.*]`` bindings stay
        top-level tables shared by every service; a declaration cannot create
        a database (``create_database`` first, then bind it). WARNING: file
        contents you pass here transit this agent's transcript in plaintext;
        never write secrets into project files.
        """
        return await _write_app_files_impl(
            _request_client(),
            name=project,
            files=files,
            delete=delete,
            idempotency_key=idempotency_key,
        )

async def apply_project(
        project: Annotated[
            str,
            Field(
                description="Immutable prj_ ID or name of the project to apply. The resolved "
                "name must equal ``[project].name`` in ``nerdit.toml``. An unknown name "
                "may be created; an unknown or retired ID is refused."
            ),
        ],
        repo_url: Annotated[
            str | None,
            Field(
                description="https Git URL whose root ``nerdit.toml`` is the declaration. "
                "Omit to apply the project's workspace (what ``write_project_files`` wrote)."
            ),
        ] = None,
        ref: Annotated[
            str | None,
            Field(description="Git branch or tag to clone; only with ``repo_url``."),
        ] = None,
        token_ref: Annotated[
            str | None,
            Field(
                description="Private-repo credential as a REFERENCE, never a token: "
                "``${vars.KEY}``, ``${secrets.shared.KEY}`` or ``${github.installation}``; "
                "only with ``repo_url``."
            ),
        ] = None,
        dry_run: DryRun = False,
        idempotency_key: IdempotencyKey = None,
    ) -> Any:
        """Apply a project's declaration: deploy every ``[services.<name>]`` of its ``nerdit.toml``.

        The loop: ``get_project`` (read) → ``apply_project(dry_run=true)`` →
        if ``status`` is ``waiting_for_variables``, do NOT invent or request
        the values in chat: ask the human to run ``nerdit vars set <project>
        --secret --prompt KEY`` for each name in ``missing``, then dry-run
        again → ``apply_project`` → ``wait_for_service`` once per ``label`` in
        ``services`` → ``diagnose_service`` on any that failed.

        The response lists ``services`` in declaration order, each with its
        ``label`` (the name every other tool takes: ``web`` is the project
        name, any other service ``<service>--<project>``), ``action``
        (``fresh``/``redeploy``) and, on a dry run, its ``plan`` (env key names
        only, values never); ``public_urls`` carries every advertised address.
        ``waiting_for_variables`` deploys nothing and returns names only.
        Asynchronous like every deploy: ``applied`` means accepted, not healthy.

        Every refusal lands before the first build, and a dry run creates
        nothing and mints no idempotency key. A project another token owns is
        409 ``project.owned``. There is NO rollback across services: a failure
        on one is ``project.apply_incomplete``, whose ``services`` lists the
        ones already applied and ``failed`` the one that was not — fix it and
        apply again (done services redeploy, the rest are retried). A
        ``[deploy]``-only ``nerdit.toml`` is not a declaration: use
        ``deploy_app`` / ``deploy_git`` for it (they answer 422
        ``deploy.use_apply`` on a declaration).

        {SANDBOX_NOTE}
        """
        return await _apply_project_impl(
            _request_client(), project,
            repo_url=repo_url, ref=ref, token_ref=token_ref,
            dry_run=dry_run, idempotency_key=idempotency_key,
        )
# fmt: on


async def project_logs(  # noqa: PLR0913 - project identity plus bounded log filters
    project: _ProjectName,
    service: Annotated[str, Field(description="Service inside the project; omit for web.")] = "web",
    tail: Annotated[
        int, Field(description="Newest matching lines; clamped to [1, 1000].")
    ] = DEFAULT_LOG_TAIL,
    since_id: Annotated[
        int, Field(description="Exclusive log cursor; ignored while tail is sent.")
    ] = 0,
    grep: Annotated[
        str | None, Field(description="Literal substring filter; never a regex.")
    ] = None,
    since: Annotated[str | None, Field(description="Inclusive ISO-8601 timestamp filter.")] = None,
    source: Annotated[str, Field(description="Log stream: all, build or runtime.")] = "all",
) -> Any:
    """Read bounded logs of one service resolved within an explicit project.

    Prefer the immutable project ID returned by list_projects. Membership is
    resolved by the daemon; a same-name replacement cannot inherit a retired ID.
    Existing node grants and local token roles still apply.
    """
    return await _call(
        _request_client().project_logs(
            project,
            service=service,
            tail=_clamp(tail, MAX_LOG_TAIL),
            since_id=since_id,
            grep=grep,
            since=since,
            source=source,
        )
    )


async def diagnose_project(
    project: _ProjectName,
    service: Annotated[str, Field(description="Service inside the project; omit for web.")] = "web",
    log_tail: Annotated[
        int, Field(description="Log lines to bundle; clamped to [1, 200].")
    ] = DEFAULT_DIAGNOSE_LOG_TAIL,
) -> Any:
    """Diagnose one service of an explicit project under the existing owner/admin policy.

    Prefer an immutable project ID. The daemon resolves project membership and
    reuses diagnose_service's failure bundle; this grants no additional authority.
    """
    return await _call(
        _request_client().diagnose_project(
            project, service=service, log_tail=_clamp(log_tail, MAX_DIAGNOSE_LOG_TAIL)
        )
    )


TOOLS = (
    create_project,
    list_projects,
    get_project,
    delete_project,
    set_variable,
    resolve_variables,
    write_project_files,
    # The shared sandbox sentence is stamped in (see ``tools/deploy.py``).
    apply_sandbox_note(apply_project),
    project_logs,
    diagnose_project,
)
