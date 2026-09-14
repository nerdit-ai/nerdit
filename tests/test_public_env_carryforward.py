"""Build-time public variables (P38) — the persisted `public_env` map follows the sources.

The map is written on EVERY deploy (empty included) precisely because
`write_redeploy` carries unlisted config keys forward; these pin the paths where
an omitted write would resurrect a map the caller removed, and the config-API
graft that keeps a `[deploy]` PUT from clobbering it.
"""

import json
from unittest.mock import AsyncMock

from nerdit.daemon import deploy_pipeline
from nerdit.daemon.deploy_pipeline import _merge_build_settings
from tests.test_deploy_git_route import (
    _client as git_client,
)
from tests.test_deploy_git_route import (
    _existing,
    _fake_clone,
)
from tests.test_deploy_git_route import (
    _post as git_post,
)
from tests.test_deploy_git_route import (
    _queries as git_queries,
)
from tests.test_deploy_route import _client, _dry_post, _queries, _zip
from tests.test_redeploy_route import (
    _git_row,
    _make_app,
    fake_clone,  # noqa: F401 — fixture
)
from tests.test_redeploy_route import _queries as _redeploy_queries


def test_repo_removed_public_env_does_not_resurrect(tmp_path, monkeypatch):
    """The repo declared the map on the previous generation and dropped it."""
    previous = _existing({"public_env": {"VITE_API": "old"}})
    q = git_queries(previous)
    resp = git_post(git_client(q, tmp_path), monkeypatch, _fake_clone())
    assert resp.status_code == 201, resp.text
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["public_env"] == {}


async def test_redeploy_from_source_drops_a_repo_removed_map(tmp_path, fake_clone):  # noqa: F811
    """Same rule on the GitWatch / `services redeploy` primitive."""
    job = _git_row()
    config = json.loads(job.config)
    config["public_env"] = {"VITE_API": "old"}
    job.config = json.dumps(config)
    q = _redeploy_queries(job)
    app = _make_app(q, tmp_path)

    result = await deploy_pipeline.redeploy_from_source(
        request_or_none=None,
        app=app,
        queries=q,
        settings=app.state.settings,
        secrets=None,
        job=job,
        principal="system",
    )

    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["public_env"] == {}
    assert result["build"]["public_env"] == {}


def test_request_replaces_the_saved_map_wholesale():
    """D-P38-6: whole-map merge — a request never merges per entry."""
    _effective, overrides, _origins = _merge_build_settings(
        {}, {"public_env": {"A": "1", "B": "2"}}, {"public_env": {"A": "9"}}
    )
    assert overrides["public_env"] == {"A": "9"}


def test_public_env_origins_across_the_four_layers():
    effective, _o, origins = _merge_build_settings({"public_env": {"A": "1"}}, None, None)
    assert origins["public_env"] == "config" and effective.public_env == {"A": "1"}
    effective, _o, origins = _merge_build_settings(
        {"public_env": {"A": "1"}}, {"public_env": {"A": "2"}}, None
    )
    assert origins["public_env"] == "saved" and effective.public_env == {"A": "2"}
    effective, _o, origins = _merge_build_settings(
        {"public_env": {"A": "1"}}, None, {"public_env": None}
    )
    assert origins["public_env"] == "config" and effective.public_env == {"A": "1"}
    effective, _o, origins = _merge_build_settings({}, None, None)
    assert "public_env" not in origins and effective.public_env is None


def test_config_api_deploy_put_preserves_public_env():
    """`public_env` is a build-tier key: a `[deploy]` PUT grafts it back."""
    from tests.test_routes_app_config import _app_job, _put
    from tests.test_routes_app_config import _client as cfg_client
    from tests.test_routes_app_config import _queries as cfg_queries

    job = _app_job(config_extra={"public_env": {"VITE_API": "https://api.example.com"}})
    q = cfg_queries(job)
    q.get_job = AsyncMock(return_value=job)

    r = _put(cfg_client(q), body={"gpus": 2})

    assert r.status_code == 200, r.text
    new_cfg = json.loads(q.update_app_config.await_args.args[1])
    assert new_cfg["public_env"] == {"VITE_API": "https://api.example.com"}


def test_nested_subdir_remerge_honours_public_env(tmp_path):
    """The root map survives a nested `build_settings`; a nested map replaces it."""
    root_only = _zip(
        {
            "nerdit.toml": '[deploy.build_settings]\nsubdir="web"\n'
            '[deploy.build_settings.public_env]\nVITE_A="root"\n',
            "web/nerdit.toml": "[deploy.build_settings]\nbuild=false\n",
            "web/package.json": '{"scripts":{"start":"node server.js"}}',
        }
    )
    r = _dry_post(_client(_queries(), tmp_path), root_only)
    assert r.status_code == 200, r.text
    assert r.json()["build"]["public_env"] == {"VITE_A": "root"}
    assert r.json()["build"]["sources"]["public_env"] == "config"

    nested = _zip(
        {
            "nerdit.toml": '[deploy.build_settings]\nsubdir="web"\n'
            '[deploy.build_settings.public_env]\nVITE_A="root"\n',
            "web/nerdit.toml": '[deploy.build_settings.public_env]\nVITE_B="nested"\n',
            "web/package.json": '{"scripts":{"start":"node server.js"}}',
        }
    )
    r = _dry_post(_client(_queries(), tmp_path), nested)
    assert r.status_code == 200, r.text
    assert r.json()["build"]["public_env"] == {"VITE_B": "nested"}
