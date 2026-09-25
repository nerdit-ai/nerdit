"""Build override membership survives each public caller, including null/false."""

import json
from email.parser import BytesParser
from email.policy import default

import httpx
import pytest
import typer

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.deploy import _deploy_async, parse_build_settings
from nerdit.daemon.idempotency import _honors_dry_run
from nerdit.mcp.tools.deploy import _deploy_git_impl, _deploy_impl, _deploy_template_impl
from nerdit.mcp.tools.workspaces import _deploy_app_impl


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["zip", "git", "template", "workspace"])
@pytest.mark.parametrize(
    "settings", [None, {}, {"build": False, "start": None}, {"preset": "nextjs"}, {"preset": None}]
)
async def test_build_settings_wire_membership(tmp_path, source, settings):
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "deploy": {
                        "build_settings": {
                            "version": 1,
                            "fields": ["preset"],
                            "presets": ["node", "nextjs", "python", "dockerfile"],
                        }
                    }
                },
            )
        return httpx.Response(200, json={"dry_run": True})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    options = {"name": "demo", "build_settings": settings, "dry_run": True}
    if source == "zip":
        (tmp_path / "package.json").write_text("{}")
        (tmp_path / "nerdit.toml").write_text(
            '[deploy]\nname = "demo"\nstart = "repository-start"\n'
        )
        await _deploy_impl(client, path=str(tmp_path), **options)
    elif source == "git":
        await _deploy_git_impl(client, repo_url="https://github.com/acme/demo", **options)
    elif source == "template":
        await _deploy_template_impl(client, "node-starter", **options)
    else:
        await _deploy_app_impl(client, **options)
    assert [r.method for r in seen] == (["POST"] if settings is None else ["GET", "POST"])
    request = seen[-1]
    assert request.url.params["dry_run"] == "true"
    assert "Idempotency-Key" not in request.headers
    if source == "zip":
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        body = {
            part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()
        }
        assert "start" not in body
        if "build_settings" in body:
            body["build_settings"] = json.loads(body["build_settings"])
    else:
        body = json.loads(request.content)
    if settings is None:
        assert "build_settings" not in body
    else:
        assert body["build_settings"] == settings


def test_cli_build_settings_parsing_and_safe_errors():
    assert parse_build_settings(None) is None
    assert parse_build_settings("{}") == {}
    assert parse_build_settings('{"build":false,"start":null}') == {"build": False, "start": None}
    for invalid in ("secret-sentinel", "[]", "null", "false"):
        with pytest.raises(typer.BadParameter) as exc:
            parse_build_settings(invalid)
        assert invalid not in str(exc.value)


def test_template_dry_run_bypass_is_scoped():
    assert _honors_dry_run("POST", "/api/app-templates/node-starter/deploy")
    assert not _honors_dry_run("PUT", "/api/app-templates/node-starter/deploy")
    assert not _honors_dry_run("POST", "/api/app-templates/node-starter")


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["zip", "git", "template", "workspace"])
async def test_old_node_refuses_build_settings_before_post(source):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.url.path == "/api/capabilities"
        return httpx.Response(200, json={"deploy": {"dry_run": True}})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    options = {"name": "demo", "build_settings": {"build": False}}
    with pytest.raises(httpx.HTTPStatusError) as exc:
        if source == "zip":
            await client.deploy(zip_bytes=b"archive", **options)
        elif source == "git":
            await client.deploy_git(repo_url="https://github.com/acme/demo", **options)
        elif source == "template":
            await client.deploy_template("node-starter", **options)
        else:
            await client.deploy_workspace(**options)
    assert "Update the daemon" in exc.value.response.json()["message"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_mcp_repo_build_settings_checked_without_promoting_defaults(tmp_path):
    (tmp_path / "nerdit.toml").write_text(
        '[deploy]\nname = "demo"\n[deploy.build_settings]\nbuild = false\n'
    )
    seen = []

    def old_node(request):
        seen.append(request)
        return httpx.Response(200, json={})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(old_node))
    result = await _deploy_impl(client, path=str(tmp_path), name="demo")
    assert result["error"]["code"] == "deploy.build_settings_unsupported"
    assert "Update the daemon" in result["error"]["message"]
    assert [r.method for r in seen] == ["GET"]

    def current_node(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"deploy": {"build_settings": {"version": 1}}})
        assert b'name="build_settings"' not in request.content
        return httpx.Response(201, json={"id": "demo"})

    seen.clear()
    client = NerditClient(host="localhost", transport=httpx.MockTransport(current_node))
    assert await _deploy_impl(client, path=str(tmp_path), name="demo") == {"id": "demo"}
    assert [r.method for r in seen] == ["GET", "POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", [None, "nextjs"])
@pytest.mark.parametrize(
    "support",
    [
        {"version": 1},
        {"version": 1, "fields": ["preset"]},
        {"version": 1, "fields": [], "presets": ["nextjs"]},
    ],
)
async def test_b2_node_refuses_preset_before_mutation(preset, support):
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"deploy": {"build_settings": support}})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.deploy_git(
            repo_url="https://github.com/acme/demo", name="demo", build_settings={"preset": preset}
        )
    assert seen == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", [None, {"build": False}, {"preset": None}])
@pytest.mark.parametrize("caller", ["cli", "mcp"])
async def test_repo_preset_refused_on_b2_even_with_other_request_settings(
    tmp_path, monkeypatch, settings, caller
):
    (tmp_path / "nerdit.toml").write_text(
        '[deploy]\nname = "demo"\n[deploy.build_settings]\npreset = "node"\n'
    )
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"deploy": {"build_settings": {"version": 1}}})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    if caller == "mcp":
        result = await _deploy_impl(client, path=str(tmp_path), build_settings=settings)
        assert result["error"]["code"] == "deploy.build_settings_unsupported"
    else:
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
        with pytest.raises(typer.Exit):
            await _deploy_async(
                str(tmp_path),
                "demo",
                None,
                None,
                None,
                None,
                [],
                None,
                False,
                build_settings=settings,
            )
    assert seen == ["GET"]


@pytest.mark.asyncio
async def test_null_preset_still_requires_repository_preset_support():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "deploy": {
                    "build_settings": {"version": 1, "fields": ["preset"], "presets": ["node"]}
                }
            },
        )

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.require_build_settings_support(
            {"preset": None}, repository={"preset": "python"}
        )
    await client.require_build_settings_support({"preset": "node"}, repository={"preset": "python"})


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["cli", "mcp"])
@pytest.mark.parametrize("selection", ["repository", "request", "reset"])
async def test_nested_repo_preset_checked_before_upload(tmp_path, monkeypatch, caller, selection):
    root = '[deploy]\nname = "demo"\n'
    if selection != "request":
        root += '[deploy.build_settings]\nsubdir = "web"\n'
    (tmp_path / "nerdit.toml").write_text(root)
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "nerdit.toml").write_text('[deploy.build_settings]\npreset = "node"\n')
    requested = (
        {"subdir": "web"}
        if selection == "request"
        else ({"subdir": None, "preset": None} if selection == "reset" else None)
    )
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"deploy": {"build_settings": {"version": 1}}})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    if caller == "mcp":
        result = await _deploy_impl(client, path=str(tmp_path), build_settings=requested)
        assert result["error"]["code"] == "deploy.build_settings_unsupported"
    else:
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
        with pytest.raises(typer.Exit):
            await _deploy_async(
                str(tmp_path),
                "demo",
                None,
                None,
                None,
                None,
                [],
                None,
                False,
                build_settings=requested,
            )
    assert seen == ["GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("escape", ["directory", "config"])
async def test_nested_preflight_does_not_read_outside_upload(tmp_path, escape):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "nerdit.toml").write_text("invalid-outside-sentinel")
    if escape == "directory":
        (root / "web").symlink_to(outside, target_is_directory=True)
    else:
        (root / "web").mkdir()
        (root / "web" / "nerdit.toml").symlink_to(outside / "nerdit.toml")
    client = NerditClient(
        host="localhost",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("No network request before containment refusal")
        ),
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await client.require_build_settings_support({"subdir": "web"}, directory=root)
    assert exc.value.response.status_code == 422
    assert "sentinel" not in exc.value.response.text


@pytest.mark.asyncio
async def test_nested_preflight_preserves_repository_settings_in_archive(tmp_path):
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\n')
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "nerdit.toml").write_text('[deploy.build_settings]\npreset = "node"\n')
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "deploy": {
                        "build_settings": {"version": 1, "fields": ["preset"], "presets": ["node"]}
                    }
                },
            )
        assert b'"preset"' not in request.content
        return httpx.Response(200, json={"dry_run": True})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    result = await _deploy_impl(
        client, path=str(tmp_path), build_settings={"subdir": "web"}, dry_run=True
    )
    assert result == {"dry_run": True}
    assert [request.method for request in seen] == ["GET", "POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["react", "", 123, ["node"], {"value": "node"}])
async def test_invalid_preset_is_validation_error_before_capability_check(preset):
    def handler(request):
        pytest.fail("Invalid settings must fail before capability lookup or upload")

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError) as exc:
        await client.deploy_git(
            repo_url="https://github.com/acme/demo", name="demo", build_settings={"preset": preset}
        )
    assert exc.value.response.status_code == 422
    assert exc.value.response.json()["code"] == "deploy.invalid_build_settings"
    assert "react" not in exc.value.response.text


def test_cli_preview_displays_preset(capsys):
    from nerdit.cli.commands.deploy import _render_dry_run

    _render_dry_run({"build": {"preset": "node", "framework": "node"}})
    assert "preset: node" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["cli", "mcp"])
async def test_parent_build_settings_do_not_select_root_inside_uploaded_app(
    tmp_path, monkeypatch, caller
):
    (tmp_path / "nerdit.toml").write_text(
        '[deploy]\nname = "demo"\n[deploy.build_settings]\nsubdir = "apps/web"\npreset = "node"\n'
    )
    app = tmp_path / "apps" / "web"
    app.mkdir(parents=True)
    (app / "package.json").write_text("{}")
    seen = []

    def handler(request):
        seen.append(request.method)
        assert b'name="build_settings"' not in request.content
        return httpx.Response(200, json={"dry_run": True})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    if caller == "mcp":
        result = await _deploy_impl(client, path=str(app), dry_run=True)
        assert result == {"dry_run": True}
    else:
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
        await _deploy_async(str(app), "demo", None, None, None, None, [], None, False, dry_run=True)
    assert seen == ["POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["cli", "mcp"])
async def test_explicit_preset_checks_old_node_before_building_archive(
    tmp_path, monkeypatch, caller
):
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\n')
    monkeypatch.setattr(
        "nerdit.cli.upload.create_dir_zip",
        lambda directory: pytest.fail("Archived before preflight"),
    )
    client = NerditClient(
        host="localhost",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"deploy": {"build_settings": {"version": 1}}})
        ),
    )
    if caller == "mcp":
        result = await _deploy_impl(client, path=str(tmp_path), build_settings={"preset": "node"})
        assert result["error"]["code"] == "deploy.build_settings_unsupported"
    else:
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
        with pytest.raises(typer.Exit):
            await _deploy_async(
                str(tmp_path),
                "demo",
                None,
                None,
                None,
                None,
                [],
                None,
                False,
                build_settings={"preset": "node"},
            )


@pytest.mark.asyncio
async def test_nested_toml_error_keeps_location_without_user_values(tmp_path):
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\n')
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "nerdit.toml").write_text("[deploy.build_settings]\nsecret-sentinel = @\n")
    client = NerditClient(
        host="localhost",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("Malformed config must fail before network activity")
        ),
    )
    result = await _deploy_impl(client, path=str(tmp_path), build_settings={"subdir": "web"})
    message = result["error"]["message"]
    assert "nerdit.toml" in message and "line 2, column 19" in message
    assert "secret-sentinel" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["cli", "mcp"])
async def test_local_deploy_checks_capabilities_once_before_archive(tmp_path, monkeypatch, caller):
    from nerdit.cli.upload import create_dir_zip

    (tmp_path / "nerdit.toml").write_text(
        '[deploy]\nname = "demo"\n[deploy.build_settings]\npreset = "python"\n'
    )
    order = []

    def archive(directory, skipped_secrets=None):
        order.append("ZIP")
        return create_dir_zip(directory, skipped_secrets)

    def handler(request):
        order.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "deploy": {
                        "build_settings": {"version": 1, "fields": ["preset"], "presets": ["node"]}
                    }
                },
            )
        return httpx.Response(200, json={"dry_run": True})

    monkeypatch.setattr("nerdit.cli.upload.create_dir_zip", archive)
    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    if caller == "mcp":
        result = await _deploy_impl(
            client, path=str(tmp_path), build_settings={"preset": "node"}, dry_run=True
        )
        assert result == {"dry_run": True}
    else:
        monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: client)
        await _deploy_async(
            str(tmp_path),
            "demo",
            None,
            None,
            None,
            None,
            [],
            None,
            False,
            build_settings={"preset": "node"},
            dry_run=True,
        )
    assert order == ["GET", "ZIP", "POST"]


@pytest.mark.asyncio
@pytest.mark.parametrize("support", [{"version": 1}, {"version": 1, "public_env": False}])
async def test_old_node_refuses_public_env_before_mutation(support):
    """A node without the flag would build and silently drop the variables."""
    seen = []

    def handler(request):
        seen.append(request.method)
        return httpx.Response(200, json={"deploy": {"build_settings": support}})

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await client.deploy_git(
            repo_url="https://github.com/acme/demo",
            name="demo",
            build_settings={"public_env": {"VITE_API": "https://api.example.com"}},
        )
    assert seen == ["GET"]


@pytest.mark.asyncio
async def test_public_env_passes_a_supporting_node_and_a_null_still_needs_it():
    def handler(request):
        return httpx.Response(
            200, json={"deploy": {"build_settings": {"version": 1, "public_env": True}}}
        )

    client = NerditClient(host="localhost", transport=httpx.MockTransport(handler))
    await client.require_build_settings_support({"public_env": {"VITE_API": "x"}})

    def old_node(request):
        return httpx.Response(200, json={"deploy": {"build_settings": {"version": 1}}})

    old = NerditClient(host="localhost", transport=httpx.MockTransport(old_node))
    # The key is on the wire either way: a null reaches a pre-P38 `extra="forbid"`
    # model as a 422, and it resets to a repository map the old node would drop.
    with pytest.raises(httpx.HTTPStatusError) as reset:
        await old.require_build_settings_support({"public_env": None})
    assert reset.value.response.status_code == 409
    with pytest.raises(httpx.HTTPStatusError) as repo:
        await old.require_build_settings_support(
            {"public_env": None}, repository={"public_env": {"VITE_A": "1"}}
        )
    assert repo.value.response.status_code == 409
