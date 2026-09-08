"""Build a real Node image when Docker and node:20-slim are available locally.

Exercise detection, generated Dockerfile and DockerRuntime build streaming.
Copy examples/ai-app into a temporary context to avoid modifying the repository;
remove the built image best-effort and let pytest clean the context. Never pull
the base image during this smoke test.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from nerdit.core import builder
from nerdit.core.runtime.docker import DockerRuntime

pytestmark = pytest.mark.asyncio

EXAMPLE_APP = Path(__file__).resolve().parent.parent / "examples" / "ai-app"


def _docker_client():
    """Return a pinged docker client, or skip when dockerd is unreachable."""
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # daemon down, no socket, permission denied…
        pytest.skip(f"docker daemon not reachable: {exc}")
    return client


async def test_smoke_node_buildpack_builds_real_image(tmp_path):
    runtime = DockerRuntime(client=_docker_client())

    # Fast-path gate: never pull node:20-slim inside a smoke test.
    if not await runtime.image_exists(builder.NODE_BASE_IMAGE):
        pytest.skip(f"base image {builder.NODE_BASE_IMAGE} not present locally")

    # Copy the fixture so the generated Dockerfile.nerdit stays out of the repo.
    context = tmp_path / "app"
    shutil.copytree(EXAMPLE_APP, context)

    plan = builder.detect(context)
    assert plan.language == "node"
    assert plan.dockerfile_text is not None
    (context / plan.dockerfile_name).write_text(plan.dockerfile_text, encoding="utf-8")

    tag = f"nerdit-test/deploy-smoke:{uuid.uuid4().hex[:8]}"
    try:
        lines = [
            line
            async for line in runtime.build_image(
                str(context), tag, dockerfile=plan.dockerfile_name
            )
        ]
        assert lines, "build produced no log output"
        assert await runtime.image_exists(tag), "built image not found in local store"
    finally:
        await runtime.remove_image(tag, force=True)
        shutil.rmtree(context, ignore_errors=True)


async def test_smoke_cached_rebuild_reuses_install_layer(tmp_path):
    """P13 WP8 (demo scenario 3): the manifest-first layer reorder makes a
    source-only edit reuse the ``npm install`` layer on rebuild.

    Build once, edit a source file (``server.js`` — NOT ``package.json``), then
    rebuild against the same tag. The classic builder emits ``---> Using cache``
    for every layer up to and including the dependency-install step, because only
    the ``COPY . .`` layer (and downstream) is invalidated by the source edit.
    """
    runtime = DockerRuntime(client=_docker_client())

    if not await runtime.image_exists(builder.NODE_BASE_IMAGE):
        pytest.skip(f"base image {builder.NODE_BASE_IMAGE} not present locally")

    context = tmp_path / "app"
    shutil.copytree(EXAMPLE_APP, context)

    plan = builder.detect(context)
    assert plan.language == "node"
    assert plan.dockerfile_text is not None
    # Sanity-check the reorder is actually in the generated Dockerfile: manifests
    # and the install step precede ``COPY . .``.
    df = plan.dockerfile_text
    assert df.index("COPY package") < df.index("npm install") < df.index("COPY . .")
    (context / plan.dockerfile_name).write_text(df, encoding="utf-8")

    tag = f"nerdit-test/deploy-cache:{uuid.uuid4().hex[:8]}"
    try:
        # First build: warms the layer cache for this Dockerfile.
        _ = [
            line
            async for line in runtime.build_image(
                str(context), tag, dockerfile=plan.dockerfile_name
            )
        ]
        assert await runtime.image_exists(tag), "first build produced no image"

        # Edit a source file only — the manifest and install layers must survive.
        (context / "server.js").write_text(
            "console.log('edited for cached-rebuild smoke');\n", encoding="utf-8"
        )

        second = [
            line
            async for line in runtime.build_image(
                str(context), tag, dockerfile=plan.dockerfile_name
            )
        ]
        joined = "\n".join(second)
        # The second build must reuse cached layers (the whole point of the
        # reorder). "Using cache" / "CACHED" is the classic builder's marker.
        assert "Using cache" in joined or "CACHED" in joined, (
            "rebuild did not reuse the install layer after a source-only edit:\n" + joined
        )
    finally:
        await runtime.remove_image(tag, force=True)
        shutil.rmtree(context, ignore_errors=True)
