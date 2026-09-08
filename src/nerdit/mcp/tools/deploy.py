"""Deploy + git-deploy + app-template MCP tool implementations (P4/P11.5).

Split out of ``mcp/server.py`` (Track B WP24, pure motion).
"""

from __future__ import annotations

import uuid
from typing import Any

from nerdit.cli.client import NerditClient
from nerdit.mcp.errors import _bad_request, _call
from nerdit.mcp.tools._shared import apply_sandbox_note
from nerdit.mcp.transport import _request_client


async def _deploy_impl(
    client: NerditClient,
    *,
    path: str | None = None,
    name: str | None = None,
    port: int | None = None,
    gpus: int | None = None,
    start: str | None = None,
    health: str | None = None,
    env: dict[str, str | None] | None = None,
    vendor: str | None = None,
    rollback: bool = False,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Deploy an app folder (build + run + URL) or roll back, auto-minting a key.

    Mirrors ``nerdit deploy``: reads the folder's ``nerdit.toml [deploy]`` for
    defaults, zips the directory, and POSTs it. ``rollback=True`` re-points the
    service at its previous image — only ``name`` is needed (no path/zip). Like
    the other write tools, a UUID key is minted when absent so a retried deploy
    collapses to one build. ``dry_run=True`` returns the build/config plan diff
    (env key names only, values never) with zero writes — no idempotency key is
    minted or needed. An ``env`` value of ``None`` deletes that key on redeploy.
    """
    from pathlib import Path

    from nerdit.config.project import find_project_config, load_project_config

    if dry_run and rollback:
        return _bad_request("dry_run cannot be combined with rollback")
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())

    if rollback:
        if not name:
            return _bad_request("name is required to roll back")
        # dry_run+rollback is rejected above, so a non-dry-run rollback always
        # minted a key here — a real write must carry one.
        assert idempotency_key is not None
        return await _call(client.rollback_deploy(name, idempotency_key=idempotency_key))

    if not path:
        return _bad_request("path is required to deploy")
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        return _bad_request(f"Not a directory: {directory}")

    project = load_project_config(find_project_config(directory))
    dcfg = project.deploy if project and project.deploy else None
    eff_name = name or (dcfg.name if dcfg else None) or directory.name
    eff_port = port if port is not None else (dcfg.port if dcfg else None)
    eff_gpus = gpus if gpus is not None else (dcfg.gpus if dcfg else None)
    eff_start = start or (dcfg.start if dcfg else None)
    eff_health = health or (dcfg.health if dcfg else None)

    from nerdit.cli.upload import create_dir_zip

    zip_bytes = create_dir_zip(directory)
    return await _call(
        client.deploy(
            zip_bytes=zip_bytes,
            name=eff_name,
            port=eff_port,
            gpus=eff_gpus,
            start=eff_start,
            health=eff_health,
            env=env,
            vendor=vendor,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )
    )


async def _deploy_git_impl(
    client: NerditClient,
    *,
    repo_url: str,
    name: str,
    ref: str | None = None,
    subdir: str | None = None,
    port: int | None = None,
    gpus: int | None = None,
    start: str | None = None,
    health: str | None = None,
    env: dict[str, str | None] | None = None,
    vendor: str | None = None,
    token_ref: str | None = None,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Deploy an app from a Git URL (build + run + URL), auto-minting a key.

    ``token_ref`` is a secret *reference* (``${secrets.KEY}``) resolved
    server-side — no raw credential passes through the tool. Like the other
    write tools, a UUID key is minted when absent so a retried deploy collapses
    to one build. ``dry_run=True`` returns the build/config plan diff (env key
    names only, values never) with zero writes — no idempotency key is minted or
    needed. An ``env`` value of ``None`` deletes that key on redeploy.
    """
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.deploy_git(
            repo_url=repo_url,
            name=name,
            ref=ref,
            subdir=subdir,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            env=env,
            vendor=vendor,
            token_ref=token_ref,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )
    )


async def _redeploy_service_impl(
    client: NerditClient,
    *,
    name: str,
    dry_run: bool = False,
    idempotency_key: str | None = None,
) -> Any:
    """Redeploy a git-deployed service from its recorded source, auto-minting a key.

    Bodyless: the daemon re-clones ``config['source']`` at its recorded ref, so
    no repo URL, path or zip is needed. Like the other write tools, a UUID key
    is minted when absent so a retried redeploy collapses to one build.
    ``dry_run=True`` returns the plan diff with zero writes — no idempotency key
    is minted or needed.
    """
    if not dry_run and not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(client.redeploy_app(name, idempotency_key=idempotency_key, dry_run=dry_run))


async def _list_app_templates_impl(client: NerditClient) -> Any:
    """Return the embedded app template catalog (id/name/category/coordinates)."""
    return await _call(client.list_app_templates())


async def _deploy_template_impl(
    client: NerditClient,
    template_id: str,
    *,
    name: str,
    env: dict[str, str | None] | None = None,
    secrets: dict[str, str] | None = None,
    port: int | None = None,
    gpus: int | None = None,
    start: str | None = None,
    health: str | None = None,
    vendor: str | None = None,
    idempotency_key: str | None = None,
) -> Any:
    """Deploy an app template (clone catalog repo + build + run), auto-minting a key.

    ``secrets`` are written write-only server-side before the row write. Like
    the other write tools, a UUID key is minted when absent so a retried deploy
    collapses to one build. An ``env`` value of ``None`` deletes that key on
    redeploy.
    """
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
    return await _call(
        client.deploy_template(
            template_id,
            name=name,
            env=env,
            secrets=secrets,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            vendor=vendor,
            idempotency_key=idempotency_key,
        )
    )


# SYNC OBLIGATION (Agent-DX): the ``[deploy]`` key list in the ``deploy`` and
# ``deploy_git`` docstrings below — a tool docstring IS its MCP description, so
# for an agent this is the documentation — must equal
# ``DeployConfig.model_fields`` and agree with ``_DEPLOY_SHAPE_HINT``
# (``daemon/deploy_pipeline.py``). Guarded by
# ``tests/test_mcp.py::test_deploy_tool_descriptions_list_every_deploy_config_field``
# here and ``tests/test_deploy_route.py::test_deploy_shape_hint_names_every_deploy_config_field``
# there — so neither half can drift from the schema unnoticed.
# fmt: off
async def deploy(
        path: str | None = None,
        name: str | None = None,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        env: dict[str, str | None] | None = None,
        vendor: str | None = None,
        rollback: bool = False,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        """Use when: the source is a local folder. Asynchronous — follow with wait_for_service.

        Deploys that folder (build + run + URL), or rolls back with
        ``rollback=True``. The response carries ``next_step`` — a machine-shaped
        ``{tool, args, why}`` naming the very call to make next — because a 201
        here means "the build was accepted", never "the app is up".

        If the app's ``nerdit.toml`` declares ``[ai.*]`` bindings, each is
        injected at launch as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the
        ``default`` binding also sets ``OPENAI_BASE_URL/OPENAI_API_KEY/
        OPENAI_MODEL`` — point the app's OpenAI SDK at these; no code change
        needed. Idempotency key auto-generated if omitted. ``dry_run=True``
        returns the build/config plan diff (env key names only, values never)
        with zero writes — no idempotency key is minted or needed. An ``env``
        value of ``None`` deletes that key on redeploy.

        ``nerdit.toml`` ``[deploy]`` keys — anything else is IGNORED and
        reported back in the response ``hints``: ``name``, ``port``, ``gpus``,
        ``start``, ``health``, ``health_type`` (``http``|``tcp``),
        ``memory_limit``, ``cpu_limit``, ``volumes``, ``release``, ``cutover``,
        ``auto_deploy``, ``edge_auth``. There is **no** ``build`` key — run the
        build in your Dockerfile. On a successful non-dry-run deploy the
        response carries ``summary`` (app/status/version/public_url), ``hints``
        (ordered one-liners, never empty) and ``next_step`` (the structured
        follow-up call); a ``dry_run`` plan and a ``rollback`` response carry
        none of the three — read a plan's advisories from its ``warnings``. On a
        node in ``path`` proxy mode the app is served at ``/<name>/``, so a
        frontend must be built with that base
        path; in ``subdomain`` mode it is served at the root of its own
        hostname and needs none — read ``capabilities.proxy.mode``.

        {SANDBOX_NOTE}
        """
        return await _deploy_impl(
            _request_client(),
            path=path,
            name=name,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            env=env,
            vendor=vendor,
            rollback=rollback,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

async def deploy_git(
        repo_url: str,
        name: str,
        ref: str | None = None,
        subdir: str | None = None,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        env: dict[str, str | None] | None = None,
        vendor: str | None = None,
        token_ref: str | None = None,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        """Use when: the app lives in a Git repo. Asynchronous — follow with wait_for_service.

        Clones ``repo_url`` server-side (public GitHub by default), then builds
        and runs it like ``deploy``. ``ref`` picks a branch/tag, ``subdir``
        deploys a monorepo subtree. For a private repo, ``token_ref`` is a
        secret *reference* (``${secrets.shared.GITHUB_TOKEN}``) resolved
        server-side — never pass a raw token — or the literal
        ``${github.installation}`` on a node linked to an account with the
        Nerdit GitHub App installed on that repo. An existing ``name`` is a
        redeploy. An idempotency key is auto-generated if omitted.
        ``dry_run=True`` returns the build/config plan diff (env key names only,
        values never) with zero writes — no idempotency key is minted or needed.
        An ``env`` value of ``None`` deletes that key on redeploy.

        If the repo's ``nerdit.toml`` declares ``[ai.*]`` bindings, each is
        injected at launch as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the
        ``default`` binding also sets ``OPENAI_BASE_URL/OPENAI_API_KEY/
        OPENAI_MODEL`` — point the app's OpenAI SDK at these; no code change
        needed.

        ``nerdit.toml`` ``[deploy]`` keys — anything else is IGNORED and
        reported back in the response ``hints``: ``name``, ``port``, ``gpus``,
        ``start``, ``health``, ``health_type`` (``http``|``tcp``),
        ``memory_limit``, ``cpu_limit``, ``volumes``, ``release``, ``cutover``,
        ``auto_deploy``, ``edge_auth``. There is **no** ``build`` key — run the
        build in your Dockerfile. On a successful non-dry-run deploy the
        response carries ``summary`` (app/status/version/public_url), ``hints``
        (ordered one-liners, never empty) and ``next_step`` (the structured
        follow-up call); a ``dry_run`` plan carries none of the three — read a
        plan's advisories from its ``warnings``. On a node in ``path``
        proxy mode the app is served at ``/<name>/``, so a frontend must be
        built with that base path.

        {SANDBOX_NOTE}
        """
        return await _deploy_git_impl(
            _request_client(),
            repo_url=repo_url,
            name=name,
            ref=ref,
            subdir=subdir,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            env=env,
            vendor=vendor,
            token_ref=token_ref,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

async def redeploy_service(
        name: str,
        dry_run: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        """Use when: a git app needs a new commit. Asynchronous — follow with wait_for_service.

        Re-clones the repository and branch/tag the service was deployed from
        and rebuilds it, so picking up a new commit is one call with the service
        name alone. Use this instead of ``deploy_git`` when the service already
        exists — every coordinate (repo, ref, subdir, private-repo credential
        reference) is read back off the service. A service deployed from a
        folder/zip has no recorded source and is refused. An idempotency key is
        auto-generated if omitted. ``dry_run=True`` returns the build/config plan
        diff (env key names only, values never) with zero writes — no
        idempotency key is minted or needed. A non-dry-run response carries
        ``summary``, ``hints`` (never empty) and ``next_step`` — the structured
        follow-up call.
        """
        return await _redeploy_service_impl(
            _request_client(),
            name=name,
            dry_run=dry_run,
            idempotency_key=idempotency_key,
        )

async def list_app_templates() -> Any:
        """List the app template store — deployable starter apps (id/name/coordinates).

        Deploy one with ``deploy_template``. Distinct from ``list_jobs``-style
        job presets: these are full app templates (web/API/AI starters).
        """
        return await _list_app_templates_impl(_request_client())

async def deploy_template(
        template_id: str,
        name: str,
        env: dict[str, str | None] | None = None,
        secrets: dict[str, str] | None = None,
        port: int | None = None,
        gpus: int | None = None,
        start: str | None = None,
        health: str | None = None,
        vendor: str | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        """Use when: you want a starter app. Asynchronous — follow with wait_for_service.

        Resolves ``template_id`` (see ``list_app_templates``), clones its
        catalog repo server-side, and deploys it under ``name``. ``env`` and the
        deploy overrides layer over the template's defaults. An ``env`` value of
        ``None`` deletes that key on redeploy. An idempotency key is
        auto-generated if omitted. The response carries ``summary``, ``hints``
        (never empty) and ``next_step`` — the structured follow-up call — since
        a 201 here means the build was accepted, not that the app is up.

        If the template declares ``[ai.*]`` bindings, each is injected at launch
        as ``NERDIT_AI_<NAME>_URL/_KEY/_MODEL``, and the ``default`` binding also
        sets ``OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL`` — point the app's
        OpenAI SDK at these; no code change needed. WARNING: the ``secrets`` values you pass
        here transit this agent's transcript in plaintext; for high-value keys
        prefer ``nerdit secrets set`` from the CLI and deploy without ``secrets``.
        """
        return await _deploy_template_impl(
            _request_client(),
            template_id,
            name=name,
            env=env,
            secrets=secrets,
            port=port,
            gpus=gpus,
            start=start,
            health=health,
            vendor=vendor,
            idempotency_key=idempotency_key,
        )
# fmt: on


# (P33) Stamp the shared sandbox sentence into the docstrings that carry the
# placeholder — a docstring must be a literal, so the interpolation happens here
# rather than inside the ``def``. Idempotent and a no-op on the tools without it.
TOOLS = tuple(
    apply_sandbox_note(fn)
    for fn in (
        deploy,
        deploy_git,
        redeploy_service,
        list_app_templates,
        deploy_template,
    )
)
