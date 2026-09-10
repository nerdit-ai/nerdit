"""Cross-ingress build settings persist intent, never inferred defaults."""

import json

import pytest

from nerdit.daemon.deploy_pipeline import _build_context, _merge_build_settings
from nerdit.daemon.errors import NerditError
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
from tests.test_deploy_route import _client, _dry_post, _node_zip, _post, _queries, _zip


def test_override_merge_reset_and_false():
    repo = {"build": "npm run repo", "install": "npm install"}
    saved = {"build": "npm run saved", "node_version": "22.23.2"}
    effective, overrides, origins = _merge_build_settings(repo, saved, {"build": None})
    assert effective.build == "npm run repo"
    assert overrides == {"node_version": "22.23.2"}
    assert origins == {"build": "config", "install": "config", "node_version": "saved"}
    effective, overrides, origins = _merge_build_settings(repo, saved, {"build": False})
    assert effective.build is False and overrides["build"] is False
    assert origins["build"] == "request"


def test_zip_preview_and_write_use_same_plan(tmp_path):
    q = _queries()
    settings = json.dumps({"build": "node build.cjs", "node_version": "22.23.2"})
    preview = _dry_post(_client(q, tmp_path), _node_zip(), build_settings=settings)
    assert preview.status_code == 200, preview.text
    q.reserve_service_for_token.assert_not_called()
    result = _post(_client(q, tmp_path), _node_zip(), build_settings=settings)
    assert result.status_code == 201, result.text
    assert result.json()["build"] == preview.json()["build"]
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["build_overrides"] == json.loads(settings)
    assert cfg["build_plan"] == preview.json()["build"]
    assert cfg["build_plan"]["install"] == "npm install --include=dev"
    assert cfg["build_plan"]["sources"]["install"] == "detected"


def test_repository_root_selection_and_nested_settings(tmp_path):
    archive = _zip(
        {
            "nerdit.toml": '[deploy.build_settings]\nsubdir="web"\n',
            "web/nerdit.toml": "[deploy.build_settings]\nbuild=false\n",
            "web/package.json": '{"scripts":{"build":"exit 42","start":"node server.js"}}',
        }
    )
    q = _queries()
    response = _dry_post(_client(q, tmp_path), archive)
    assert response.status_code == 200, response.text
    plan = response.json()["build"]
    assert plan["subdir"] == "web" and plan["build"] is False
    assert plan["sources"]["subdir"] == "config"
    assert not list((tmp_path / "uploads").iterdir())


def test_root_symlink_outside_rejected(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (context / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(NerditError):
        _build_context(context, "escape")


@pytest.mark.parametrize(
    "settings",
    [
        {"subdir": "../PRIVATE"},
        {"build_env": {"PUBLIC": "PRIVATE"}},
        {"build": "echo PRIVATE\nRUN bad"},
        {"node_version": "PRIVATE"},
    ],
)
def test_zip_invalid_settings_are_redacted_before_write(tmp_path, settings):
    q = _queries()
    response = _post(_client(q, tmp_path), _node_zip(), build_settings=json.dumps(settings))
    assert response.status_code == 422 and "PRIVATE" not in response.text
    q.reserve_service_for_token.assert_not_called()


def test_git_saved_override_survives_new_source_and_reset(tmp_path, monkeypatch):
    previous = _existing({"build_overrides": {"build": False, "node_version": "22.23.2"}})
    q = git_queries(previous)
    response = git_post(git_client(q, tmp_path), monkeypatch, _fake_clone())
    assert response.status_code == 201, response.text
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["build_overrides"] == {"build": False, "node_version": "22.23.2"}
    assert response.json()["build"]["sources"]["build"] == "saved"
    assert response.json()["build"]["commit_sha"] == "a" * 40
    response = git_post(
        git_client(q, tmp_path), monkeypatch, _fake_clone(), build_settings={"node_version": None}
    )
    assert response.status_code == 201, response.text
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["build_overrides"] == {"build": False}
    assert response.json()["build"]["node_version"] == "24.20.0"


def test_git_invalid_settings_do_not_echo_input(tmp_path, monkeypatch):
    clone = _fake_clone()
    response = git_post(
        git_client(git_queries(), tmp_path),
        monkeypatch,
        clone,
        build_settings={"build_env": {"FOO": "PRIVATE"}},
    )
    assert response.status_code == 422 and "PRIVATE" not in response.text
    clone.assert_not_called()


def test_selected_root_preserves_auth_and_deploy_defaults(tmp_path):
    archive = _zip(
        {
            "nerdit.toml": (
                '[deploy]\nport=4567\nhealth="/ready"\n'
                '[deploy.edge_auth]\nuser="owner"\npassword="${secrets.PASS}"\n'
                '[deploy.build_settings]\nsubdir="web"\n'
            ),
            "web/package.json": "{}",
        }
    )
    q = _queries()
    # No explicit legacy port: repository port must survive selecting the app root.
    response = _client(q, tmp_path).post(
        "/deploy?dry_run=true",
        data={"name": "demo"},
        files={"archive": ("app.zip", archive, "application/zip")},
        headers={"Authorization": "Bearer sub-raw"},
    )
    assert response.status_code == 200, response.text
    effective = response.json()["effective"]
    assert effective["port"] == 4567 and effective["health"] == "/ready"
    assert effective["edge_auth"] == {"user": "owner", "password": "${secrets.PASS}"}


def test_explicit_legacy_start_beats_saved_override(tmp_path, monkeypatch):
    q = git_queries(_existing({"build_overrides": {"start": "node old.js"}}))
    response = git_post(git_client(q, tmp_path), monkeypatch, _fake_clone(), start="node new.js")
    assert response.status_code == 201, response.text
    assert response.json()["build"]["start"] == "node new.js"
    cfg = json.loads(q.update_service_config.call_args.args[1])
    assert cfg["build_overrides"]["start"] == "node new.js"


def test_start_override_reports_api_config_clobber(tmp_path, monkeypatch):
    q = git_queries(_existing({"config_source": "api", "command": "node old.js"}))
    response = git_post(git_client(q, tmp_path), monkeypatch, _fake_clone(), start="node new.js")
    assert response.status_code == 201, response.text
    assert response.json()["overwrote_api_config"] is True


def test_nested_null_start_resets_even_with_legacy_start(tmp_path, monkeypatch):
    q = git_queries(_existing({"build_overrides": {"start": "node old.js"}}))
    response = git_post(
        git_client(q, tmp_path),
        monkeypatch,
        _fake_clone(),
        start="node legacy.js",
        build_settings={"start": None},
    )
    assert response.status_code == 201, response.text
    assert response.json()["build"]["start"] == "npm start"
    assert json.loads(q.update_service_config.call_args.args[1])["build_overrides"] == {}


def test_preset_saved_and_reset_precedence():
    effective, saved, origins = _merge_build_settings({}, {"preset": "node"}, None)
    assert effective.preset == "node" and origins["preset"] == "saved"
    effective, saved, origins = _merge_build_settings({}, saved, {"preset": None})
    assert effective.preset is None and "preset" not in saved
    effective, saved, origins = _merge_build_settings(
        {"preset": "python"}, {"preset": "node"}, {"preset": None}
    )
    assert effective.preset == "python" and origins["preset"] == "config"


def test_preset_preview_and_deploy_persist_same_intent(tmp_path):
    q = _queries()
    archive = _zip(
        {
            "requirements.txt": "",
            "package.json": json.dumps({"scripts": {"start": "node server.js"}}),
            "server.js": "console.log('ok')",
        }
    )
    settings = json.dumps({"preset": "node"})
    preview = _dry_post(_client(q, tmp_path), archive, build_settings=settings)
    assert preview.status_code == 200, preview.text
    assert preview.json()["build"]["preset"] == "node"
    assert preview.json()["build"]["framework"] == "node"
    assert preview.json()["build"]["sources"]["preset"] == "request"
    q.reserve_service_for_token.assert_not_called()
    deployed = _post(_client(q, tmp_path), archive, build_settings=settings)
    assert deployed.status_code == 201, deployed.text
    assert deployed.json()["build"] == preview.json()["build"]
    cfg = json.loads(q.reserve_service_for_token.call_args.args[0].config)
    assert cfg["build_overrides"] == {"preset": "node"}


@pytest.mark.parametrize("preset", ["react", "vite", "static", "auto", "", 1])
def test_unsupported_presets_fail_before_service_write(tmp_path, preset):
    q = _queries()
    result = _post(_client(q, tmp_path), _node_zip(), build_settings=json.dumps({"preset": preset}))
    assert result.status_code == 422
    q.reserve_service_for_token.assert_not_called()


def test_stale_saved_next_preset_explains_reset_and_recovers(tmp_path, monkeypatch):
    q = git_queries(_existing({"build_overrides": {"preset": "nextjs"}}))
    client = git_client(q, tmp_path)
    rejected = git_post(client, monkeypatch, _fake_clone())
    assert rejected.status_code == 400, rejected.text
    assert rejected.json()["code"] == "deploy.no_buildpack"
    assert 'build_settings={"preset": null}' in rejected.json()["hint"]
    q.update_service_config.assert_not_called()
    recovered = git_post(client, monkeypatch, _fake_clone(), build_settings={"preset": None})
    assert recovered.status_code == 201, recovered.text
    assert recovered.json()["build"]["framework"] == "node"
    assert json.loads(q.update_service_config.call_args.args[1])["build_overrides"] == {}


@pytest.mark.parametrize("layer", ["request", "repository", "saved"])
def test_credential_commands_rejected_in_every_settings_layer(tmp_path, monkeypatch, layer):
    command = 'curl -H "Authorization: Bearer private-sentinel" https://example.com'
    if layer == "saved":
        q = git_queries(_existing({"build_overrides": {"install": command}}))
        response = git_post(git_client(q, tmp_path), monkeypatch, _fake_clone())
    else:
        q = _queries()
        archive = (
            _node_zip()
            if layer == "request"
            else _zip(
                {
                    "package.json": '{"scripts":{"start":"node server.js"}}',
                    "nerdit.toml": "[deploy.build_settings]\ninstall = "
                    + json.dumps(command)
                    + "\n",
                }
            )
        )
        options = {"build_settings": json.dumps({"install": command})} if layer == "request" else {}
        response = _post(_client(q, tmp_path), archive, **options)
    assert response.status_code == 422, response.text
    assert "private-sentinel" not in response.text
    q.reserve_service_for_token.assert_not_called()
    q.update_service_config.assert_not_called()


@pytest.mark.parametrize("replacement", [None, "npm ci"])
def test_unsafe_saved_command_can_be_reset_or_replaced(tmp_path, monkeypatch, replacement):
    q = git_queries(_existing({"build_overrides": {"install": "TOKEN=private-sentinel npm ci"}}))
    response = git_post(
        git_client(q, tmp_path), monkeypatch, _fake_clone(), build_settings={"install": replacement}
    )
    assert response.status_code == 201, response.text
    assert "private-sentinel" not in response.text
    config = q.update_service_config.call_args.args[1]
    assert "private-sentinel" not in config
    expected = {} if replacement is None else {"install": replacement}
    assert json.loads(config)["build_overrides"] == expected


@pytest.mark.parametrize("saved", [["bad"], {"unknown": "bad"}])
def test_saved_settings_shape_remains_validated(saved):
    with pytest.raises(NerditError):
        _merge_build_settings({}, saved, {"install": None})
