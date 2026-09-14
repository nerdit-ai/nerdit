"""Build-time public variables (P38) — secret refusal is value-free on every ingress."""

import json

from nerdit.config.project import DeployConfig
from nerdit.core.builder import detect
from tests.test_deploy_git_route import _client as git_client
from tests.test_deploy_git_route import _queries as git_queries
from tests.test_deploy_route import _client, _dry_post, _post, _queries, _zip

SENTINEL = "${secrets.DB_PASSWORD}"


def test_passthrough_arg_name_case_must_match(tmp_path):
    """Docker ARG NAMES are case-sensitive; a mis-cased ARG must still warn."""
    (tmp_path / "Dockerfile").write_text("FROM alpine\nARG vite_sentinel\n")
    plan = detect(
        tmp_path,
        DeployConfig(name="x", port=80, build_settings={"public_env": {"VITE_SENTINEL": "1"}}),
    )
    assert plan.warnings == [
        "public_env VITE_SENTINEL is not declared as ARG in Dockerfile; the value will be unused."
    ]


def test_zip_nerdit_toml_secret_ref_is_refused_value_free(tmp_path):
    archive = _zip(
        {
            "nerdit.toml": (
                '[deploy]\nname="app"\n[deploy.build_settings.public_env]\n'
                f'VITE_TOKEN="{SENTINEL}"\n'
            ),
            "package.json": '{"scripts":{"build":"vite build","start":"node server.js"}}',
        }
    )
    q = _queries()
    resp = _post(_client(q, tmp_path), archive)
    assert resp.status_code == 422, resp.text
    assert "DB_PASSWORD" not in resp.text
    q.reserve_service_for_token.assert_not_called()


def test_git_json_body_secret_ref_422_is_value_free(tmp_path, monkeypatch):
    from tests.test_deploy_git_route import _post as git_post

    q = git_queries()
    resp = git_post(
        git_client(q, tmp_path),
        monkeypatch,
        None,
        build_settings={"public_env": {"VITE_TOKEN": SENTINEL}},
    )
    assert resp.status_code == 422, resp.text
    assert "DB_PASSWORD" not in resp.text


def test_dry_run_body_secret_ref_422_is_value_free(tmp_path):
    q = _queries()
    resp = _dry_post(
        _client(q, tmp_path),
        _zip({"package.json": '{"scripts":{"start":"node s.js"}}'}),
        build_settings=json.dumps({"public_env": {"VITE_TOKEN": SENTINEL}}),
    )
    assert resp.status_code == 422, resp.text
    assert "DB_PASSWORD" not in resp.text


def test_dry_run_masks_runtime_env_while_public_env_is_clear(tmp_path):
    q = _queries()
    resp = _dry_post(
        _client(q, tmp_path),
        _zip({"package.json": '{"scripts":{"build":"vite build","start":"node s.js"}}'}),
        build_settings=json.dumps({"public_env": {"VITE_API": "https://api.example.com"}}),
        env=json.dumps({"RUNTIME_SECRET": "hunter2"}),
    )
    assert resp.status_code == 200, resp.text
    assert "hunter2" not in resp.text
    assert resp.json()["build"]["public_env"] == {"VITE_API": "https://api.example.com"}
