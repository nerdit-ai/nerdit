"""NERDIT_HOOK_<NAME>_URL in the launch env: one per live hook, only when linked."""

from __future__ import annotations

import json

from nerdit.config.settings import LinkSettings
from nerdit.core.services import ServiceController
from nerdit.db.models import Job, JobKind, JobStatus

CONFIG = {"image": "demo:1", "port": 8000, "env": {"NERDIT_HOOK_STRIPE_URL": "user-set"}}


async def _app_with_hooks(queries) -> Job:  # noqa: ANN001
    job = await queries.create_job(
        Job(
            name="shop",
            kind=JobKind.service,
            service_name="shop",
            gpu_count=0,
            status=JobStatus.building,
            desired_state="running",
            config=json.dumps(CONFIG),
        )
    )
    for name in ("stripe", "gh-push", "gone"):
        await queries.add_hook(
            "shop",
            job_id=job.id,
            name=name,
            sender="none",
            path="/h",
            header=None,
            prefix=None,
            secret_ref=None,
        )
    assert await queries.remove_hook("shop", "gone")
    return job


def _controller(queries, link: LinkSettings | None) -> ServiceController:  # noqa: ANN001
    return ServiceController(queries=queries, runtime=None, link=link)  # type: ignore[arg-type]


async def test_launch_env_carries_nerdit_hook_url_for_each_live_hook(queries) -> None:  # noqa: ANN001
    job = await _app_with_hooks(queries)
    link = LinkSettings(slug="gpu-box", nodes_base_domain="nodes.test")
    ids = {hook.name: hook.hook_id for hook in await queries.list_hooks("shop")}

    resolved = await _controller(queries, link)._resolve_launch_env(job, CONFIG, 8000, None, None)

    base = "https://shop--gpu-box.nodes.test/.nerdit/hooks/"
    # Injected: overrides the user's own env value and joins the protected set.
    assert resolved.env["NERDIT_HOOK_STRIPE_URL"] == base + ids["stripe"]
    assert resolved.env["NERDIT_HOOK_GH_PUSH_URL"] == base + ids["gh-push"]
    assert "NERDIT_HOOK_GONE_URL" not in resolved.env
    assert {"NERDIT_HOOK_STRIPE_URL", "NERDIT_HOOK_GH_PUSH_URL"} <= set(resolved.injected_keys)


async def test_launch_env_omits_the_url_when_unlinked(queries) -> None:  # noqa: ANN001
    job = await _app_with_hooks(queries)

    for link in (None, LinkSettings(), LinkSettings(slug="gpu-box")):
        resolved = await _controller(queries, link)._resolve_launch_env(
            job, CONFIG, 8000, None, None
        )
        assert resolved.env["NERDIT_HOOK_STRIPE_URL"] == "user-set"
        assert "NERDIT_HOOK_GH_PUSH_URL" not in resolved.env


def test_hook_url_is_scrubbed_from_job_log_tails():
    from nerdit.core.app_build import _is_credential_key

    assert _is_credential_key("NERDIT_HOOK_STRIPE_URL")
