"""Unit tests for the P4 deploy CLI: create_dir_zip, client method, command flow."""

from __future__ import annotations

import io
import json
import zipfile

import httpx
import pytest
import typer

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.deploy import _deploy_async, _render_dry_run, parse_env_pairs
from nerdit.cli.display import console, display_deploy_result, display_wait_outcome
from nerdit.cli.upload import create_dir_zip

# ---- F4-MARKUP: display_wait_outcome must escape server-derived strings ----


def test_f4_markup_wait_outcome_escapes_server_strings():
    """A build error carrying Rich-markup shapes must not crash the renderer.

    Before the fix, ``console.print`` interpolating an unescaped
    ``[/uvicorn]`` raised ``rich.errors.MarkupError`` at the tail of
    ``nerdit deploy --wait``. With the escape, the literal text is preserved.
    """
    with console.capture() as cap:
        display_wait_outcome(
            {
                "outcome": "failed",
                "service_name": "my-app",
                "reason": "crash_loop",
                "error_class": "user_error",
                "error_message": "docker build error [/uvicorn] tail",
            }
        )
    out = cap.get()
    assert "[/uvicorn]" in out
    assert "crash_loop" in out
    assert "user_error" in out


def test_f4_markup_wait_outcome_timeout_escapes_phase():
    """The timeout branch interpolates a server-derived phase — escape it too."""
    with console.capture() as cap:
        display_wait_outcome(
            {
                "outcome": "timeout",
                "service_name": "my-app",
                "phase": "[bold]building",
                "waited_s": 60.0,
            }
        )
    out = cap.get()
    assert "[bold]building" in out


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost",
        port=9321,
        token=None,
        transport=httpx.MockTransport(handler),
    )


# ---- create_dir_zip ----


def test_create_dir_zip_includes_app_files_and_applies_excludes(tmp_path):
    (tmp_path / "app.js").write_text("console.log('hi')")
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\nport = 3000\n')
    sub = tmp_path / "lib"
    sub.mkdir()
    (sub / "util.js").write_text("x")
    # Excluded per ZIP_EXCLUDE_PATTERNS
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("gitstuff")
    pyc = tmp_path / "__pycache__"
    pyc.mkdir()
    (pyc / "mod.cpython-311.pyc").write_bytes(b"\x00")
    (tmp_path / "stray.pyc").write_bytes(b"\x00")

    zip_bytes = create_dir_zip(tmp_path)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
    assert "app.js" in names
    assert "nerdit.toml" in names
    assert "lib/util.js" in names  # arcnames relative to the directory root
    assert not any(".git" in n for n in names)
    assert not any("__pycache__" in n for n in names)
    assert "stray.pyc" not in names


# ---- parse_env_pairs ----


def test_parse_env_pairs_parses_key_val_items():
    assert parse_env_pairs(["A=1", "B=two", "C=with=equals"]) == {
        "A": "1",
        "B": "two",
        "C": "with=equals",
    }


@pytest.mark.parametrize("bad", ["NOEQUALS", "=value"])
def test_parse_env_pairs_rejects_malformed_items(bad):
    with pytest.raises(ValueError):
        parse_env_pairs([bad])


# ---- client.deploy / client.rollback_deploy ----


@pytest.mark.asyncio
async def test_client_deploy_encodes_multipart_and_idempotency_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers.get("content-type", "")
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = request.content
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    client = _make_client(handler)
    await client.deploy(
        zip_bytes=b"ZIPDATA",
        name="demo",
        port=3000,
        gpus=1,
        start="npm start",
        health="/healthz",
        env={"A": "1"},
        vendor="nvidia",
        idempotency_key="key-42",
    )

    assert captured["path"] == "/api/deploy"
    assert captured["content_type"].startswith("multipart/form-data")
    assert captured["idem"] == "key-42"
    body = captured["body"]
    assert b"ZIPDATA" in body
    assert b"demo" in body
    assert b"3000" in body
    assert b"npm start" in body
    assert b"/healthz" in body
    assert json.dumps({"A": "1"}).encode() in body  # env is a JSON form field
    assert b"nvidia" in body


@pytest.mark.asyncio
async def test_client_deploy_omits_unset_optional_fields():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.deploy(zip_bytes=b"Z", name="demo", idempotency_key="k")

    body = captured["body"]
    for omitted in (
        b'name="port"',
        b'name="gpus"',
        b'name="start"',
        b'name="health"',
        b'name="env"',
        b'name="vendor"',
    ):
        assert omitted not in body
    assert b'name="name"' in body
    assert b'name="archive"' in body


@pytest.mark.asyncio
async def test_client_deploy_auto_mints_idempotency_key_on_real_run():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["idem"] = request.headers.get("idempotency-key")
        captured["query"] = dict(request.url.params)
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    # No idempotency_key, real deploy: the client must mint one so a network
    # retry replays instead of triggering a second build.
    await client.deploy(zip_bytes=b"Z", name="demo")
    assert captured["idem"]  # non-empty, minted
    assert "dry_run" not in captured["query"]


@pytest.mark.asyncio
async def test_client_deploy_dry_run_stays_keyless():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["idem"] = request.headers.get("idempotency-key")
        captured["query"] = dict(request.url.params)
        return httpx.Response(200, json={"dry_run": True})

    client = _make_client(handler)
    await client.deploy(zip_bytes=b"Z", name="demo", dry_run=True)
    assert captured["idem"] is None  # no key on a dry run
    assert captured["query"]["dry_run"] == "true"


@pytest.mark.asyncio
async def test_client_deploy_git_auto_mints_idempotency_key_on_real_run():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["idem"] = request.headers.get("idempotency-key")
        captured["query"] = dict(request.url.params)
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.deploy_git(repo_url="https://github.com/o/r", name="demo")
    assert captured["idem"]  # minted
    assert "dry_run" not in captured["query"]


@pytest.mark.asyncio
async def test_client_deploy_git_dry_run_stays_keyless():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"dry_run": True})

    client = _make_client(handler)
    await client.deploy_git(repo_url="https://github.com/o/r", name="demo", dry_run=True)
    assert captured["idem"] is None


@pytest.mark.asyncio
async def test_client_rollback_deploy_posts_with_idempotency_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["idem"] = request.headers.get("idempotency-key")
        return httpx.Response(200, json={"id": "svc-1", "name": "demo", "status": "building"})

    client = _make_client(handler)
    await client.rollback_deploy("demo", idempotency_key="key-rb")
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/deploy/demo/rollback"
    assert captured["idem"] == "key-rb"


@pytest.mark.asyncio
async def test_client_wait_for_service_uses_long_httpx_timeout():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["version"] = request.url.params.get("version")
        captured["timeout_param"] = request.url.params.get("timeout")
        captured["httpx_timeout"] = request.extensions.get("timeout")
        return httpx.Response(200, json={"outcome": "converged", "service_name": "demo"})

    client = _make_client(handler)
    out = await client.wait_for_service("demo", version=4, timeout=120)
    assert out["outcome"] == "converged"
    assert captured["path"] == "/api/services/demo/wait"
    assert captured["version"] == "4"
    assert captured["timeout_param"] == "120"
    # The httpx read timeout must exceed the server long-poll (timeout + 30),
    # not the hardcoded 5 s GET default that would kill the poll.
    assert captured["httpx_timeout"]["read"] == 150.0


# ---- command flow (fake client) ----


class _FakeClient:
    def __init__(self, wait_outcome: dict | None = None):
        self.deploy_calls: list[dict] = []
        self.rollback_calls: list[dict] = []
        self.deploy_git_calls: list[dict] = []
        self.wait_calls: list[dict] = []
        self._wait_outcome = wait_outcome or {
            "outcome": "converged",
            "service_name": "demo",
            "version": 5,
            "phase": "healthy",
            "status": "running",
            "public_url": "https://host/demo",
            "waited_s": 1.2,
        }

    async def deploy(self, **kwargs):
        self.deploy_calls.append(kwargs)
        return {
            "id": "svc-1",
            "name": kwargs["name"],
            "status": "building",
            "build_version": 5,
            "last_deploy": {"version": 5, "phase": "queued"},
            "endpoint": {"public_url": "https://host/demo"},
        }

    async def rollback_deploy(self, name, idempotency_key):
        self.rollback_calls.append({"name": name, "idempotency_key": idempotency_key})
        return {
            "id": "svc-1",
            "name": name,
            "status": "building",
            "build_version": 3,
            "last_deploy": {"version": 3, "phase": "queued"},
        }

    async def deploy_git(self, **kwargs):
        self.deploy_git_calls.append(kwargs)
        return {
            "id": "svc-1",
            "name": kwargs["name"],
            "status": "building",
            "build_version": 5,
            "last_deploy": {"version": 5, "phase": "queued"},
            "endpoint": {"public_url": f"https://host/{kwargs['name']}"},
        }

    async def wait_for_service(self, ident, *, version=None, timeout=60):
        self.wait_calls.append({"ident": ident, "version": version, "timeout": timeout})
        return self._wait_outcome


@pytest.fixture()
def fake_client(monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient()
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_deploy_zips_and_merges_nerdit_toml(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    (tmp_path / "nerdit.toml").write_text(
        '[deploy]\nname = "demo"\nport = 3000\ngpus = 1\nstart = "npm start"\n'
    )

    await _deploy_async(str(tmp_path), None, None, None, None, None, [], None, False)

    assert len(fake_client.deploy_calls) == 1
    call = fake_client.deploy_calls[0]
    # [deploy] defaults flow through.
    assert call["name"] == "demo"
    assert call["port"] == 3000
    assert call["gpus"] == 1
    # The daemon reads repository defaults; only actual overrides go on the wire.
    assert call["start"] is None
    # A fresh Idempotency-Key was minted.
    assert call["idempotency_key"]
    # The folder was zipped (a real zip containing the app file).
    with zipfile.ZipFile(io.BytesIO(call["zip_bytes"])) as zf:
        assert "app.js" in zf.namelist()


@pytest.mark.asyncio
async def test_deploy_cli_flags_override_toml_and_env_is_parsed(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\nport = 3000\n')

    await _deploy_async(
        str(tmp_path), "other", 9999, None, None, None, ["A=1", "B=two"], None, False
    )

    call = fake_client.deploy_calls[0]
    assert call["name"] == "other"
    assert call["port"] == 9999
    assert call["env"] == {"A": "1", "B": "two"}


@pytest.mark.asyncio
async def test_deploy_name_defaults_to_folder_when_no_toml(tmp_path, fake_client):
    app_dir = tmp_path / "my-app"
    app_dir.mkdir()
    (app_dir / "app.js").write_text("x")

    await _deploy_async(str(app_dir), None, None, None, None, None, [], None, False)

    call = fake_client.deploy_calls[0]
    assert call["name"] == "my-app"
    # No [deploy] section → port/gpus omitted so daemon defaults apply.
    assert call["port"] is None
    assert call["gpus"] is None


@pytest.mark.asyncio
async def test_deploy_rollback_calls_rollback_and_does_not_zip(tmp_path, fake_client):
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\nport = 3000\n')

    await _deploy_async(str(tmp_path), None, None, None, None, None, [], None, True)

    assert fake_client.deploy_calls == []
    assert len(fake_client.rollback_calls) == 1
    assert fake_client.rollback_calls[0]["name"] == "demo"
    assert fake_client.rollback_calls[0]["idempotency_key"]


@pytest.mark.asyncio
async def test_deploy_rejects_malformed_env_pair(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")

    with pytest.raises(typer.Exit):
        await _deploy_async(str(tmp_path), "demo", None, None, None, None, ["NOEQ"], None, False)
    assert fake_client.deploy_calls == []


# ---- --wait / --timeout (P13 WP3) ----


@pytest.mark.asyncio
async def test_deploy_wait_passes_stamped_version_and_exits_zero_on_converged(
    tmp_path, fake_client
):
    (tmp_path / "app.js").write_text("x")
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\n')

    # --wait/--timeout are keyword-only (after the `*`); a converged outcome
    # returns normally (exit 0).
    await _deploy_async(
        str(tmp_path), None, None, None, None, None, [], None, False, wait=True, wait_timeout=90
    )

    assert len(fake_client.wait_calls) == 1
    call = fake_client.wait_calls[0]
    assert call["ident"] == "demo"
    # Version handoff (locked): the 201's stamped last_deploy.version, not a default.
    assert call["version"] == 5
    assert call["timeout"] == 90


@pytest.mark.asyncio
async def test_deploy_without_wait_never_calls_wait(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    await _deploy_async(str(tmp_path), "demo", None, None, None, None, [], None, False)
    assert fake_client.wait_calls == []


@pytest.mark.asyncio
async def test_deploy_wait_failed_exits_one(tmp_path, monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient(
        wait_outcome={
            "outcome": "failed",
            "service_name": "demo",
            "version": 5,
            "phase": "failed",
            "status": "failed",
            "reason": "crash_loop",
            "error_class": "OOM",
            "error_message": "killed",
            "waited_s": 3.0,
        }
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    (tmp_path / "app.js").write_text("x")

    with pytest.raises(typer.Exit) as ei:
        await _deploy_async(
            str(tmp_path), "demo", None, None, None, None, [], None, False, wait=True
        )
    assert ei.value.exit_code == 1


@pytest.mark.asyncio
async def test_deploy_wait_timeout_exits_three_not_two(tmp_path, monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient(
        wait_outcome={
            "outcome": "timeout",
            "service_name": "demo",
            "version": 5,
            "phase": "building",
            "status": "building",
            "waited_s": 60.0,
        }
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    (tmp_path / "app.js").write_text("x")

    with pytest.raises(typer.Exit) as ei:
        await _deploy_async(
            str(tmp_path), "demo", None, None, None, None, [], None, False, wait=True
        )
    assert ei.value.exit_code == 3


@pytest.mark.asyncio
async def test_deploy_wait_superseded_exits_one(tmp_path, monkeypatch):
    import nerdit.cli.client as client_mod

    fake = _FakeClient(
        wait_outcome={
            "outcome": "superseded",
            "service_name": "demo",
            "version": 5,
            "phase": "building",
            "status": "building",
            "waited_s": 0.0,
        }
    )
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    (tmp_path / "app.js").write_text("x")

    with pytest.raises(typer.Exit) as ei:
        await _deploy_async(
            str(tmp_path), "demo", None, None, None, None, [], None, False, wait=True
        )
    assert ei.value.exit_code == 1


@pytest.mark.asyncio
async def test_deploy_rollback_wait_uses_rollback_version(tmp_path, fake_client):
    (tmp_path / "nerdit.toml").write_text('[deploy]\nname = "demo"\n')
    await _deploy_async(str(tmp_path), None, None, None, None, None, [], None, True, wait=True)
    assert len(fake_client.wait_calls) == 1
    assert fake_client.wait_calls[0]["version"] == 3


# ---- git mode (client + command flow) ----


@pytest.mark.asyncio
async def test_client_deploy_git_posts_json_and_idempotency_key():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["ct"] = request.headers.get("content-type", "")
        captured["idem"] = request.headers.get("idempotency-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1", "name": "demo", "status": "building"})

    client = _make_client(handler)
    await client.deploy_git(
        repo_url="https://github.com/o/r",
        name="demo",
        ref="v1.0.0",
        subdir="apps/web",
        port=3000,
        env={"A": "1"},
        token_ref="${secrets.shared.GITHUB_TOKEN}",
        idempotency_key="key-git",
    )

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/deploy/git"
    assert "application/json" in captured["ct"]
    assert captured["idem"] == "key-git"
    assert captured["body"] == {
        "repo_url": "https://github.com/o/r",
        "name": "demo",
        "ref": "v1.0.0",
        "subdir": "apps/web",
        "port": 3000,
        "env": {"A": "1"},
        "token_ref": "${secrets.shared.GITHUB_TOKEN}",
    }


@pytest.mark.asyncio
async def test_client_deploy_git_omits_unset_optional_fields():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "svc-1"})

    client = _make_client(handler)
    await client.deploy_git(repo_url="https://github.com/o/r", name="demo", idempotency_key="k")
    assert captured["body"] == {"repo_url": "https://github.com/o/r", "name": "demo"}


@pytest.mark.asyncio
async def test_deploy_git_mode_calls_client_and_derives_name_from_url(tmp_path, fake_client):
    await _deploy_async(
        None, None, None, None, None, None, [], None, False, repo="https://github.com/o/my-app.git"
    )
    assert fake_client.deploy_calls == []
    assert len(fake_client.deploy_git_calls) == 1
    call = fake_client.deploy_git_calls[0]
    assert call["repo_url"] == "https://github.com/o/my-app.git"
    assert call["name"] == "my-app"  # last URL segment, .git stripped
    assert call["idempotency_key"]


@pytest.mark.asyncio
async def test_deploy_git_mode_derives_name_from_subdir(tmp_path, fake_client):
    await _deploy_async(
        None,
        None,
        None,
        None,
        None,
        None,
        ["A=1"],
        None,
        False,
        repo="https://github.com/o/mono",
        ref="v2",
        subdir="apps/web",
    )
    call = fake_client.deploy_git_calls[0]
    assert call["name"] == "web"  # subdir basename wins over URL segment
    assert call["ref"] == "v2"
    assert call["subdir"] == "apps/web"
    assert call["env"] == {"A": "1"}


@pytest.mark.asyncio
async def test_deploy_git_mode_explicit_name_overrides_derivation(tmp_path, fake_client):
    await _deploy_async(
        None,
        "custom",
        None,
        None,
        None,
        None,
        [],
        None,
        False,
        repo="https://github.com/o/my-app.git",
    )
    assert fake_client.deploy_git_calls[0]["name"] == "custom"


@pytest.mark.asyncio
async def test_deploy_git_mode_rejects_path_argument(tmp_path, fake_client):
    with pytest.raises(typer.Exit):
        await _deploy_async(
            str(tmp_path),
            None,
            None,
            None,
            None,
            None,
            [],
            None,
            False,
            repo="https://github.com/o/r",
        )
    assert fake_client.deploy_git_calls == []
    assert fake_client.deploy_calls == []


@pytest.mark.asyncio
async def test_deploy_git_mode_rejects_rollback_combo(fake_client):
    with pytest.raises(typer.Exit):
        await _deploy_async(
            None,
            None,
            None,
            None,
            None,
            None,
            [],
            None,
            True,
            repo="https://github.com/o/r",
        )
    assert fake_client.deploy_git_calls == []
    assert fake_client.rollback_calls == []


@pytest.mark.asyncio
async def test_deploy_ref_subdir_token_ref_require_repo(fake_client):
    with pytest.raises(typer.Exit):
        await _deploy_async(None, None, None, None, None, None, [], None, False, ref="v1")
    assert fake_client.deploy_git_calls == []
    assert fake_client.deploy_calls == []


@pytest.mark.asyncio
async def test_deploy_renders_structured_client_error(tmp_path, monkeypatch, capsys):
    import nerdit.cli.client as client_mod

    class _ErrClient:
        async def deploy(self, **kwargs):
            request = httpx.Request("POST", "http://localhost:9321/api/deploy")
            response = httpx.Response(
                422,
                request=request,
                json={
                    "code": "deploy.invalid",
                    "message": "Invalid deploy request",
                    "detail": "name must be a DNS label",
                },
            )
            raise httpx.HTTPStatusError("422", request=request, response=response)

    monkeypatch.setattr(client_mod, "get_configured_client", lambda: _ErrClient())
    (tmp_path / "app.js").write_text("x")

    with pytest.raises(typer.Exit):
        await _deploy_async(str(tmp_path), "Bad_Name", None, None, None, None, [], None, False)

    out = capsys.readouterr().out
    assert "Invalid request" in out
    assert "name must be a DNS label" in out


# ---- env null-delete + dry-run flags (P13 WP9) ----


def test_build_env_map_combines_set_and_unset():
    from nerdit.cli.commands.deploy import build_env_map

    assert build_env_map(["A=1", "B=two"], ["OLD"]) == {"A": "1", "B": "two", "OLD": None}
    assert build_env_map([], []) is None
    assert build_env_map(["A="], []) == {"A": ""}  # empty string stays a real value


def test_build_env_map_rejects_unset_with_equals():
    from nerdit.cli.commands.deploy import build_env_map

    with pytest.raises(ValueError):
        build_env_map([], ["BAD=x"])


@pytest.mark.asyncio
async def test_deploy_unset_env_maps_to_null(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    await _deploy_async(
        str(tmp_path), "demo", None, None, None, None, ["A=1"], None, False, unset_env=["OLD"]
    )
    call = fake_client.deploy_calls[0]
    assert call["env"] == {"A": "1", "OLD": None}


@pytest.mark.asyncio
async def test_deploy_dry_run_passes_flag_and_skips_wait(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    await _deploy_async(
        str(tmp_path), "demo", None, None, None, None, [], None, False, dry_run=True, wait=True
    )
    call = fake_client.deploy_calls[0]
    assert call["dry_run"] is True
    # A dry-run never blocks on convergence, even with --wait.
    assert fake_client.wait_calls == []


@pytest.mark.asyncio
async def test_deploy_dry_run_with_rollback_errors(tmp_path, fake_client):
    (tmp_path / "app.js").write_text("x")
    with pytest.raises(typer.Exit):
        await _deploy_async(
            str(tmp_path), "demo", None, None, None, None, [], None, True, dry_run=True
        )
    assert fake_client.deploy_calls == []
    assert fake_client.rollback_calls == []


# ---- Agent-DX: the CLI renders the server's deploy hints --------------------


def test_deploy_renders_server_hints():
    """A human at the shell sees the same advisory channel an agent gets over
    MCP — one edit at the shared renderer covers all three deploy call sites."""
    with console.capture() as cap:
        display_deploy_result(
            {
                "name": "my-app",
                "status": "building",
                "endpoint": {"public_url": "https://host/my-app/"},
                "hints": ["set your base path to '/my-app/'", "an ignored key: [/build]"],
            }
        )
    out = cap.get()
    assert "set your base path to '/my-app/'" in out
    # Server-derived text goes through _plain, so Rich-markup shapes survive
    # literally instead of raising MarkupError (the F4-MARKUP rule).
    assert "[/build]" in out


def test_dry_run_plan_warning_with_markup_shapes_does_not_crash():
    """The plan's ``warnings`` are the SAME server-derived channel as ``hints``.

    Since the unknown-key advisory interpolates app-authored ``[deploy]`` key
    names, a quoted key like ``"[/b]"`` reaches the renderer verbatim; unescaped
    that raised ``rich.errors.MarkupError`` and took ``nerdit deploy --dry-run``
    down. F4-MARKUP applies to every server-derived string, plan included.
    """
    with console.capture() as cap:
        _render_dry_run(
            {
                "action": "create",
                "name": "my-app",
                "warnings": ["[deploy] key(s) were IGNORED: [/b], build."],
            }
        )
    out = cap.get()
    assert "[/b]" in out


def test_dry_run_plan_warning_keeps_the_section_name_visible():
    """The benign half of the same bug: without escaping, Rich eats ``[deploy]``
    as a style tag, dropping the advisory's most load-bearing token (and the
    pre-existing ``[ai]``/``[db]`` plan warnings with it)."""
    warning = "[deploy] key(s) this daemon does not understand were IGNORED: build."
    with console.capture() as cap:
        _render_dry_run({"action": "update", "name": "my-app", "warnings": [warning]})
    assert warning in cap.get()


def test_deploy_renders_nothing_when_hints_is_empty():
    """No hints (and the rollback call site, whose body has no ``hints`` key at
    all) renders exactly the pre-existing three lines."""
    body = {"name": "my-app", "status": "building", "endpoint": {"url": "http://127.0.0.1:9400"}}
    with console.capture() as cap:
        display_deploy_result(body, build_hint=False)
    without_key = cap.get()
    with console.capture() as cap:
        display_deploy_result({**body, "hints": []}, build_hint=False)
    assert cap.get() == without_key
    assert without_key.count("\n") == 3
