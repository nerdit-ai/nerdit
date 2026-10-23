"""App env `${secrets…}` references: resolved at launch, never shipped as text."""

from __future__ import annotations

import pytest

from nerdit.config.project import env_ref_problem
from nerdit.core.app_build import _sensitive_env_values
from nerdit.core.secrets import SHARED_SCOPE, SecretManager
from nerdit.core.services import LaunchEnvNotReady, ServiceController
from nerdit.db.models import Job, JobKind, JobStatus


def _job() -> Job:
    return Job(
        name="app",
        kind=JobKind.service,
        service_name="app",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
    )


async def _resolve(queries, mgr: SecretManager, env: dict[str, str]):  # noqa: ANN001
    controller = ServiceController(queries=queries, runtime=None, secrets=mgr)  # type: ignore[arg-type]
    return await controller._resolve_launch_env(_job(), {"env": env}, 8000, None, None)


@pytest.fixture
def mgr(tmp_path) -> SecretManager:  # noqa: ANN001
    return SecretManager(tmp_path / "secrets")


async def test_shared_ref_resolves_audited_and_scrubbed(queries, mgr) -> None:  # noqa: ANN001
    mgr.set(SHARED_SCOPE, {"OPENAI_API_KEY": "sk-shared-value"})
    resolved = await _resolve(
        queries, mgr, {"OPENAI_API_KEY": "${secrets.shared.OPENAI_API_KEY}", "SUPA": "x"}
    )
    assert resolved.env["OPENAI_API_KEY"] == "sk-shared-value"
    assert resolved.shared_keys == ["OPENAI_API_KEY"]
    assert resolved.ref_env_keys == {"OPENAI_API_KEY"}
    assert "sk-shared-value" in _sensitive_env_values(resolved)


async def test_resolved_value_scrubbed_under_non_credential_name(queries, mgr) -> None:  # noqa: ANN001
    mgr.set(SHARED_SCOPE, {"SUPA": "https://p.supabase.co"})
    resolved = await _resolve(queries, mgr, {"SUPABASE_URL": "${secrets.shared.SUPA}"})
    assert "https://p.supabase.co" in _sensitive_env_values(resolved)


async def test_service_scope_overrides_shared(queries, mgr) -> None:  # noqa: ANN001
    mgr.set(SHARED_SCOPE, {"TOKEN": "shared"})
    mgr.set("app", {"TOKEN": "own"})
    resolved = await _resolve(queries, mgr, {"API_TOKEN": "${secrets.shared.TOKEN}"})
    assert resolved.env["API_TOKEN"] == "own"


async def test_vars_alias_resolves(queries, mgr) -> None:  # noqa: ANN001
    mgr.set("app", {"DB_PASS": "pw"})
    resolved = await _resolve(queries, mgr, {"PASS": "${vars.DB_PASS}"})
    assert resolved.env["PASS"] == "pw"


async def test_missing_ref_waits_naming_key_and_ref_only(queries, mgr) -> None:  # noqa: ANN001
    with pytest.raises(LaunchEnvNotReady) as exc:
        await _resolve(queries, mgr, {"OPENAI_API_KEY": "${secrets.shared.OPENAI_API_KEY}"})
    assert exc.value.kind == "env_ref"
    assert "OPENAI_API_KEY" in exc.value.message
    assert "--shared OPENAI_API_KEY" in exc.value.message


@pytest.mark.parametrize(
    "value", ["Bearer ${secrets.TOKEN} sk-literal", "${github.installation}", "${vars.lower}"]
)
async def test_backstop_refuses_leftover_refs_by_key(queries, mgr, value) -> None:  # noqa: ANN001
    mgr.set("app", {"TOKEN": "t"})
    with pytest.raises(LaunchEnvNotReady) as exc:
        await _resolve(queries, mgr, {"AUTH": value})
    assert exc.value.kind == "env_ref"
    assert "AUTH" in exc.value.message
    assert "sk-literal" not in exc.value.message


async def test_variable_stored_as_ref_hits_backstop(queries, mgr) -> None:  # noqa: ANN001
    mgr.set("app", {"OPENAI_API_KEY": "${secrets.shared.OPENAI_API_KEY}"})
    with pytest.raises(LaunchEnvNotReady, match="env OPENAI_API_KEY"):
        await _resolve(queries, mgr, {})


async def test_ordinary_templates_pass_through(queries, mgr) -> None:  # noqa: ANN001
    env = {"LISTEN": "${PORT:-8080}", "H": "$HOME/x"}
    resolved = await _resolve(queries, mgr, env)
    assert {k: resolved.env[k] for k in env} == env
    assert resolved.ref_env_keys == set()


async def test_corrupt_shared_only_blocks_services_that_need_it(queries, mgr, tmp_path) -> None:  # noqa: ANN001
    mgr.set("app", {"KEY": "own"})
    (tmp_path / "secrets" / "_shared.enc").write_text("corrupted", encoding="utf-8")
    resolved = await _resolve(queries, mgr, {"A": "${secrets.shared.KEY}", "B": "plain"})
    assert resolved.env["A"] == "own"
    with pytest.raises(LaunchEnvNotReady) as exc:
        await _resolve(queries, mgr, {"A": "${secrets.shared.OTHER}"})
    assert exc.value.kind == "shared_secrets"


def test_ingress_grammar() -> None:
    assert env_ref_problem({"A": "${secrets.shared.X}", "B": "${vars.Y}", "C": None}) is None
    assert env_ref_problem({"A": "${PORT:-1}"}) is None
    problem = env_ref_problem({"AUTH": "Bearer ${secrets.X}"})
    assert problem is not None and "env AUTH" in problem and "whole value" in problem
    problem = env_ref_problem({"GH": "${github.installation}"})
    assert problem is not None and "token_ref" in problem
