"""Unit tests for the P40d CLI: `nerdit apply`, the `nerdit deploy` handoff, `wire_name`."""

from __future__ import annotations

import io
import zipfile

import click
import httpx
import pytest
import typer
from typer.testing import CliRunner

from nerdit.cli.client import NerditClient
from nerdit.cli.commands.apply import EXIT_WAITING_FOR_VARIABLES, _apply_async
from nerdit.cli.commands.deploy import _deploy_async
from nerdit.cli.display import console

DECLARATION = (
    '[project]\nname = "asso"\n\n[services.web]\nport = 3000\n\n'
    '[services.api]\nport = 8000\nbuild_settings = { subdir = "apps/api" }\n\n'
    '[vars]\nrequired = ["API_KEY"]\n'
)


def _make_client(handler) -> NerditClient:
    return NerditClient(
        host="localhost", port=9321, token=None, transport=httpx.MockTransport(handler)
    )


# ---- wire_name: qualified -> label, composed and never parsed back ----


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("asso/api", "api--asso"),
        ("asso/web", "asso"),
        ("asso/production/api", "api--asso"),
        ("asso", "asso"),
        ("api--asso", "api--asso"),  # a label passes through untouched
        ("shared", "shared"),
        ("a1b2c3d4e5f6", "a1b2c3d4e5f6"),
    ],
)
def test_wire_name_translates_only_a_qualified_name(name, label):
    assert NerditClient.wire_name(name) == label


@pytest.mark.parametrize(
    "bad", ["asso/", "/api", "asso//api", "a/b/c/d", "asso/staging/api", "asso/" + "x" * 70]
)
def test_wire_name_refuses_an_invalid_qualified_form_value_free(bad):
    with pytest.raises(ValueError) as excinfo:
        NerditClient.wire_name(bad)
    assert bad not in str(excinfo.value)
    assert "x" * 70 not in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_qualified_name_reaches_the_wire_as_its_label():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"logs": [], "keys": [], "domains": []})

    client = _make_client(handler)
    await client.get_service_logs("asso/api")
    await client.get_service("asso/web")
    await client.diagnose_service("asso/api")
    await client.list_secrets("asso/api")
    await client.get_app_config("asso/api")
    await client.rollback_deploy("asso/api", idempotency_key="k")
    assert paths == [
        "/api/services/api--asso/logs",
        "/api/services/asso",
        "/api/services/api--asso/diagnose",
        "/api/secrets/api--asso",
        "/api/config/apps/api--asso",
        "/api/deploy/api--asso/rollback",
    ]


# ---- client.apply_project ----


@pytest.mark.asyncio
async def test_apply_project_uploads_the_archive_with_a_minted_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        seen["idem"] = request.headers.get("idempotency-key")
        seen["ctype"] = request.headers["content-type"]
        return httpx.Response(200, json={"status": "applied"})

    await _make_client(handler).apply_project("asso", zip_bytes=b"PK")
    assert seen["path"] == "/api/projects/asso/apply"
    assert seen["query"] == {}
    assert seen["idem"]
    assert seen["ctype"].startswith("multipart/form-data")


@pytest.mark.asyncio
async def test_apply_project_dry_run_git_is_keyless():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = dict(request.url.params)
        seen["idem"] = request.headers.get("idempotency-key")
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"status": "planned"})

    await _make_client(handler).apply_project(
        "asso", repo_url="https://github.com/o/r", ref="main", dry_run=True
    )
    assert seen["query"] == {"dry_run": "true"}
    assert seen["idem"] is None
    assert "repo_url=" in seen["body"] and "ref=main" in seen["body"]
    assert "token_ref" not in seen["body"]


# ---- command flow ----


class _FakeClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.apply_calls: list[dict] = []
        self.deploy_calls: list[dict] = []
        self.waited: list[str] = []

    async def apply_project(self, project, **kwargs):
        self.apply_calls.append({"project": project, **kwargs})
        return self.result

    async def deploy(self, **kwargs):
        self.deploy_calls.append(kwargs)
        return {"name": kwargs["name"], "status": "building"}

    async def wait_for_service(self, name, *, version=None, timeout=60):
        self.waited.append(name)
        outcome = "failed" if name == "api--asso" else "converged"
        return {"outcome": outcome, "service_name": name}


APPLIED = {
    "project": "asso",
    "status": "applied",
    "dry_run": False,
    "services": [
        {"label": "asso", "service": "web", "action": "fresh", "status": "building"},
        {"label": "api--asso", "service": "api", "action": "redeploy", "status": "building"},
    ],
    "public_urls": [{"url": "https://host/asso/", "kind": "default"}],
}


def _patch(monkeypatch, result: dict) -> _FakeClient:
    import nerdit.cli.client as client_mod

    fake = _FakeClient(result)
    monkeypatch.setattr(client_mod, "get_configured_client", lambda: fake)
    return fake


def _project_dir(tmp_path):
    (tmp_path / "nerdit.toml").write_text(DECLARATION)
    (tmp_path / "app.js").write_text("x")
    return tmp_path


@pytest.mark.asyncio
async def test_apply_zips_the_folder_under_the_declared_project(tmp_path, monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    with console.capture() as cap:
        await _apply_async(str(_project_dir(tmp_path)))
    call = fake.apply_calls[0]
    assert call["project"] == "asso"
    assert call["idempotency_key"] and call["dry_run"] is False
    with zipfile.ZipFile(io.BytesIO(call["zip_bytes"])) as zf:
        assert {"nerdit.toml", "app.js"} <= set(zf.namelist())
    out = cap.get()
    assert "api--asso" in out and "https://host/asso/" in out


@pytest.mark.asyncio
async def test_apply_waiting_for_variables_prints_the_command_and_exits_4(tmp_path, monkeypatch):
    fake = _patch(
        monkeypatch,
        {"project": "asso", "status": "waiting_for_variables", "missing": ["API_KEY"]},
    )
    with console.capture() as cap, pytest.raises(typer.Exit) as excinfo:
        await _apply_async(str(_project_dir(tmp_path)), dry_run=True)
    assert excinfo.value.exit_code == EXIT_WAITING_FOR_VARIABLES == 4
    assert fake.apply_calls[0]["idempotency_key"] is None  # a dry run is keyless
    assert "nerdit vars set asso --secret --prompt API_KEY" in cap.get()


@pytest.mark.asyncio
async def test_apply_wait_waits_every_label_and_exits_with_the_worst(tmp_path, monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    with pytest.raises(typer.Exit) as excinfo:
        await _apply_async(str(_project_dir(tmp_path)), wait=True)
    assert fake.waited == ["asso", "api--asso"]  # a failure does not skip the rest
    assert excinfo.value.exit_code == 1


@pytest.mark.asyncio
async def test_apply_wait_rejects_a_superseding_deploy_and_checks_every_generation(
    tmp_path, monkeypatch
):
    result = APPLIED | {
        "services": [
            entry | {"build_version": version}
            for entry, version in zip(APPLIED["services"], [2, 5], strict=True)
        ]
    }
    fake = _patch(monkeypatch, result)
    waited = []

    async def wait_for_service(name, *, version=None, timeout=60):
        waited.append((name, version, timeout))
        # A later deployment converged before polling: only a pinned wait rejects it.
        return {
            "outcome": "superseded" if name == "asso" and version == 2 else "converged",
            "service_name": name,
        }

    monkeypatch.setattr(fake, "wait_for_service", wait_for_service)
    with pytest.raises(typer.Exit) as excinfo:
        await _apply_async(str(_project_dir(tmp_path)), wait=True, wait_timeout=17)
    assert excinfo.value.exit_code == 1
    assert waited == [("asso", 2, 17), ("api--asso", 5, 17)]


@pytest.mark.asyncio
async def test_apply_git_defaults_the_project_to_the_repo_slug(monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    await _apply_async(None, repo="https://github.com/o/asso.git", token_ref="${vars.GH}")
    call = fake.apply_calls[0]
    assert (call["project"], call["zip_bytes"]) == ("asso", None)
    assert call["token_ref"] == "${vars.GH}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "toml",
    [
        '[deploy]\nname = "demo"\n',  # legacy: `nerdit deploy` territory
        '[project]\nname = "asso"\n',  # no [services]
        '[project]\nname = "asso"\n[services.web]\nname = "never-print-this"\n',
    ],
)
async def test_apply_refuses_a_bad_folder_before_any_request(tmp_path, monkeypatch, toml):
    fake = _patch(monkeypatch, APPLIED)
    (tmp_path / "nerdit.toml").write_text(toml)
    with console.capture() as cap, pytest.raises(typer.Exit) as excinfo:
        await _apply_async(str(tmp_path))
    assert excinfo.value.exit_code == 1
    assert fake.apply_calls == []
    assert "never-print-this" not in cap.get()


@pytest.mark.asyncio
async def test_apply_source_flags_need_repo(tmp_path, monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    with pytest.raises(typer.Exit):
        await _apply_async(str(_project_dir(tmp_path)), ref="main")
    assert fake.apply_calls == []


# ---- `nerdit deploy` on a declaration folder ----


@pytest.mark.asyncio
async def test_deploy_hands_a_declaration_folder_to_apply(tmp_path, monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    await _deploy_async(
        str(_project_dir(tmp_path)), None, None, None, None, None, [], None, False, dry_run=True
    )
    assert fake.deploy_calls == []
    assert fake.apply_calls[0]["project"] == "asso"
    assert fake.apply_calls[0]["dry_run"] is True


@pytest.mark.asyncio
async def test_deploy_refuses_legacy_flags_on_a_declaration_folder(tmp_path, monkeypatch):
    fake = _patch(monkeypatch, APPLIED)
    with pytest.raises(typer.Exit) as excinfo:
        await _deploy_async(
            str(_project_dir(tmp_path)), None, 3000, None, None, None, [], None, False
        )
    assert excinfo.value.exit_code == 1
    assert fake.apply_calls == [] and fake.deploy_calls == []


def test_apply_is_a_registered_verb():
    from nerdit.cli.app import app

    result = CliRunner().invoke(app, ["apply", "--help"])
    assert result.exit_code == 0
    # CI forces colour, and Rich styles the two dashes apart from the name.
    assert "--dry-run" in click.unstyle(result.output)
