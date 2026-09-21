"""Contract-freeze tests for ``[ai.*]`` binding resolution (P5 / S9).

This file is the executable form of **Invariant #1: the OpenAI API is the
contract**. The injected env var names are asserted LITERALLY — renaming any of
them is a breaking contract change and must fail here first.

Pure-async unit tests over the in-memory DB (no TestClient).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from nerdit.config.settings import ServicesSettings
from nerdit.core.app_build import _sensitive_env_values
from nerdit.core.models.backend import sanitize_model_name
from nerdit.core.models.binding import (
    LOCAL_MODEL_API_KEY,
    BindingNotReady,
    ResolvedBinding,
    inject_env,
    inject_env_key_names,
    resolve_binding,
    resolve_bindings,
)
from nerdit.core.secrets import SHARED_SCOPE, SecretManager, project_storage_name
from nerdit.core.services import LaunchEnvNotReady, ServiceController
from nerdit.daemon.routes.service_diagnose import _ai_env_key_names
from nerdit.db.models import Job, JobKind, JobStatus

pytestmark = pytest.mark.asyncio

BRIDGE_HOST = "172.17.0.1"

API_SPEC = {
    "provider": "api",
    "model": "gpt-4o-mini",
    "base_url": "https://api.example.com/v1",
    "api_key": "${secrets.OPENAI_KEY}",
}
OLLAMA_SPEC = {"provider": "ollama", "model": "llama3.1:8b"}
MODEL_SERVICE_NAME = sanitize_model_name("llama3.1:8b")  # 'ollama-llama3-1-8b'


def _model_row(
    *,
    status: JobStatus = JobStatus.running,
    pulled: bool = True,
    kind: JobKind = JobKind.model,
) -> Job:
    cfg: dict = {"model": "llama3.1:8b", "backend": "ollama", "image": "ollama/ollama"}
    if pulled:
        cfg["model_pulled"] = True
    return Job(
        name=MODEL_SERVICE_NAME,
        kind=kind,
        service_name=MODEL_SERVICE_NAME,
        gpu_count=0,
        status=status,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
    )


async def _serve_model(queries, *, status=JobStatus.running, pulled=True, endpoint=True) -> Job:
    job = _model_row(status=status, pulled=pulled)
    await queries.create_job(job)
    if endpoint:
        await queries.acquire_service_port(MODEL_SERVICE_NAME, job.id, 11434, (9400, 9499))
    return job


# --- Invariant #1: the injected env var names, LITERALLY -----------------------


async def test_inject_env_names_are_the_frozen_contract():
    resolved = {
        "default": ResolvedBinding(
            base_url="http://172.17.0.1:9400/v1", api_key="nerdit-local", model="llama3.1:8b"
        ),
        "cheap": ResolvedBinding(
            base_url="https://api.example.com/v1", api_key="sk-secret", model="gpt-4o-mini"
        ),
    }
    env = inject_env(resolved)
    # The frozen names — renaming ANY of these is a breaking contract change.
    assert env["OPENAI_BASE_URL"] == "http://172.17.0.1:9400/v1"
    assert env["OPENAI_API_KEY"] == "nerdit-local"
    assert env["OPENAI_MODEL"] == "llama3.1:8b"
    assert env["NERDIT_AI_DEFAULT_URL"] == "http://172.17.0.1:9400/v1"
    assert env["NERDIT_AI_DEFAULT_KEY"] == "nerdit-local"
    assert env["NERDIT_AI_DEFAULT_MODEL"] == "llama3.1:8b"
    assert env["NERDIT_AI_CHEAP_URL"] == "https://api.example.com/v1"
    assert env["NERDIT_AI_CHEAP_KEY"] == "sk-secret"
    assert env["NERDIT_AI_CHEAP_MODEL"] == "gpt-4o-mini"
    # Exactly the contract vars — nothing extra leaks into the app env.
    assert set(env) == {
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "NERDIT_AI_DEFAULT_URL",
        "NERDIT_AI_DEFAULT_KEY",
        "NERDIT_AI_DEFAULT_MODEL",
        "NERDIT_AI_CHEAP_URL",
        "NERDIT_AI_CHEAP_KEY",
        "NERDIT_AI_CHEAP_MODEL",
    }


async def test_inject_env_without_default_binding_has_no_openai_triplet():
    resolved = {
        "cheap": ResolvedBinding(base_url="https://x/v1", api_key="k", model="m"),
    }
    env = inject_env(resolved)
    assert "OPENAI_BASE_URL" not in env
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_MODEL" not in env
    assert env["NERDIT_AI_CHEAP_URL"] == "https://x/v1"


async def test_f7_env_key_names_match_inject_env():
    """F7-ENVGRAMMAR: names-only derivation must equal ``inject_env``'s keys.

    The route's ``_ai_env_key_names`` and the ``inject_env_key_names`` companion
    must both derive from the single ``inject_env`` grammar — no hand-mirrored
    parallel list. Pin a two-binding spec (one of them ``default``).
    """
    ai_specs = {
        "default": {"provider": "ollama", "model": "llama3.1:8b"},
        "judge": {"provider": "api", "model": "gpt-4o"},
    }
    real_resolved = {
        "default": ResolvedBinding(base_url="http://172.17.0.1:9400/v1", api_key="k", model="m"),
        "judge": ResolvedBinding(base_url="https://api/v1", api_key="k2", model="m2"),
    }
    expected = set(inject_env(real_resolved).keys())
    assert inject_env_key_names(ai_specs.keys()) == expected
    assert _ai_env_key_names(ai_specs) == expected


async def test_f7_env_key_names_no_default_has_no_openai():
    """F7: a non-``default`` spec yields no ``OPENAI_*`` keys (default-gated)."""
    names = inject_env_key_names({"judge"})
    assert not any(k.startswith("OPENAI_") for k in names)
    assert names == {"NERDIT_AI_JUDGE_URL", "NERDIT_AI_JUDGE_KEY", "NERDIT_AI_JUDGE_MODEL"}


# --- provider='api' -------------------------------------------------------------


async def test_api_provider_resolves_secret_ref(queries):
    resolved = await resolve_binding(
        "cheap", API_SPEC, queries, {"OPENAI_KEY": "sk-live-123"}, BRIDGE_HOST
    )
    assert resolved == ResolvedBinding(
        base_url="https://api.example.com/v1", api_key="sk-live-123", model="gpt-4o-mini"
    )


async def test_api_provider_missing_secret_is_not_ready(queries):
    with pytest.raises(BindingNotReady) as exc:
        await resolve_binding("cheap", API_SPEC, queries, {}, BRIDGE_HOST)
    # Actionable: names the missing key and the command to set it.
    assert "OPENAI_KEY" in str(exc.value)
    assert "nerdit secrets set" in str(exc.value)


async def test_invalid_persisted_spec_is_not_ready(queries):
    with pytest.raises(BindingNotReady):
        await resolve_binding("bad", {"provider": "api", "model": "m"}, queries, {}, BRIDGE_HOST)


# --- provider='ollama' -----------------------------------------------------------


async def test_ollama_running_and_pulled_resolves_bridge_url(queries):
    await _serve_model(queries, status=JobStatus.running, pulled=True)
    endpoint = await queries.get_service_endpoint(MODEL_SERVICE_NAME)

    resolved = await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)

    assert resolved.base_url == f"http://{BRIDGE_HOST}:{endpoint.host_port}/v1"
    assert resolved.api_key == LOCAL_MODEL_API_KEY == "nerdit-local"
    assert resolved.model == "llama3.1:8b"


async def test_ollama_custom_named_model_resolves_by_ref(queries):
    """A model served under a ``--name`` override still satisfies the binding.

    The binding references the model ref (``llama3.1:8b``), not the row's
    ``service_name`` — which here is a custom name unrelated to
    ``sanitize_model_name``. Resolution must key off ``config['model']``.
    """
    custom_name = "my-llama"
    cfg = {"model": "llama3.1:8b", "backend": "ollama", "image": "ollama/ollama"}
    cfg["model_pulled"] = True
    job = Job(
        name=custom_name,
        kind=JobKind.model,
        service_name=custom_name,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
    )
    await queries.create_job(job)
    await queries.acquire_service_port(custom_name, job.id, 11434, (9400, 9499))
    endpoint = await queries.get_service_endpoint(custom_name)

    resolved = await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)

    assert resolved.base_url == f"http://{BRIDGE_HOST}:{endpoint.host_port}/v1"
    assert resolved.model == "llama3.1:8b"


async def test_ollama_model_missing_is_not_ready(queries):
    with pytest.raises(BindingNotReady) as exc:
        await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)
    assert "nerdit serve llama3.1:8b" in str(exc.value)


@pytest.mark.parametrize("status", [JobStatus.building, JobStatus.stopped, JobStatus.restarting])
async def test_ollama_non_running_states_are_not_ready(queries, status):
    await _serve_model(queries, status=status, pulled=True)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)
    assert "llama3.1:8b" in str(exc.value)
    assert status.value in str(exc.value)


async def test_ollama_running_but_not_pulled_is_not_ready(queries):
    # Strict readiness (Decision #5): a live server whose weights are still
    # downloading is NOT ready — the app waits for model_pulled.
    await _serve_model(queries, status=JobStatus.running, pulled=False)
    with pytest.raises(BindingNotReady) as exc:
        await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)
    assert "still pulling" in str(exc.value)


async def test_ollama_running_without_endpoint_is_not_ready(queries):
    await _serve_model(queries, status=JobStatus.running, pulled=True, endpoint=False)
    with pytest.raises(BindingNotReady):
        await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)


async def test_non_model_row_with_matching_name_is_not_ready(queries):
    # A plain app service squatting the sanitized name must never resolve as a
    # model endpoint.
    await queries.create_job(_model_row(kind=JobKind.service, pulled=True))
    with pytest.raises(BindingNotReady):
        await resolve_binding("default", OLLAMA_SPEC, queries, {}, BRIDGE_HOST)


# --- resolve_bindings (all-or-nothing) -------------------------------------------


async def test_resolve_bindings_all_or_nothing(queries):
    # ollama binding ready, api binding missing its secret → the whole resolve
    # raises so the app is never launched with a partially wired AI env.
    await _serve_model(queries)
    specs = {"default": OLLAMA_SPEC, "cheap": API_SPEC}
    with pytest.raises(BindingNotReady):
        await resolve_bindings(specs, queries, {}, BRIDGE_HOST)

    resolved = await resolve_bindings(specs, queries, {"OPENAI_KEY": "sk-live-123"}, BRIDGE_HOST)
    assert set(resolved) == {"default", "cheap"}
    assert resolved["cheap"].api_key == "sk-live-123"
    assert resolved["default"].api_key == "nerdit-local"


# --- shared-scope refs (P8): precedence + hints -----------------------------------


SHARED_API_SPEC = {
    "provider": "api",
    "model": "gpt-4o-mini",
    "base_url": "https://api.example.com/v1",
    "api_key": "${secrets.shared.OPENAI_KEY}",
}


async def test_shared_ref_resolves_from_shared_env(queries):
    resolved = await resolve_binding(
        "cheap", SHARED_API_SPEC, queries, {}, BRIDGE_HOST, shared_env={"OPENAI_KEY": "sk-shared"}
    )
    assert resolved.api_key == "sk-shared"


async def test_shared_ref_per_service_key_overrides_shared(queries):
    # Precedence (decided): the per-service KEY is the override escape hatch.
    resolved = await resolve_binding(
        "cheap",
        SHARED_API_SPEC,
        queries,
        {"OPENAI_KEY": "sk-own"},
        BRIDGE_HOST,
        shared_env={"OPENAI_KEY": "sk-shared"},
    )
    assert resolved.api_key == "sk-own"


async def test_shared_ref_missing_everywhere_is_not_ready_with_shared_hint(queries):
    with pytest.raises(BindingNotReady) as exc:
        await resolve_binding("cheap", SHARED_API_SPEC, queries, {}, BRIDGE_HOST, shared_env={})
    # Actionable: names the missing key and the --shared command to set it.
    assert "OPENAI_KEY" in str(exc.value)
    assert "nerdit secrets set --shared" in str(exc.value)


async def test_unscoped_ref_never_falls_back_to_shared(queries):
    # Zero ambiguity: ${secrets.KEY} is per-service-only even when the shared
    # store happens to carry a same-named key.
    with pytest.raises(BindingNotReady):
        await resolve_binding(
            "cheap", API_SPEC, queries, {}, BRIDGE_HOST, shared_env={"OPENAI_KEY": "sk-shared"}
        )


async def test_resolve_bindings_threads_shared_env(queries):
    resolved = await resolve_bindings(
        {"cheap": SHARED_API_SPEC},
        queries,
        {},
        BRIDGE_HOST,
        shared_env={"OPENAI_KEY": "sk-shared"},
    )
    assert resolved["cheap"].api_key == "sk-shared"


# --- shared-scope launch path (P8): lazy load, no auto-injection, audit ----------
#
# ServiceController over the real in-memory DB with the _FakeRuntime pattern
# (test_secrets.py) and a REAL SecretManager, so the corrupt-`_shared.enc` and
# no-auto-injection assertions exercise the actual store.


class _FakeRuntime:
    def __init__(self) -> None:
        self.live: dict[str, datetime] = {}
        self.run_configs: list = []
        self.counter = 0

    async def run(self, config):
        self.counter += 1
        cid = f"c{self.counter}"
        self.live[cid] = datetime.now(UTC)
        self.run_configs.append(config)
        return cid

    async def image_exists(self, image):
        return True

    async def list_managed_containers(self):
        return list(self.live.items())

    async def stop(self, cid, timeout=10):
        self.live.pop(cid, None)

    async def kill(self, cid):
        self.live.pop(cid, None)

    async def remove(self, cid, force=False):
        self.live.pop(cid, None)

    async def wait(self, cid, timeout_s=None):
        return 1

    async def status(self, cid):
        return "running" if cid in self.live else None

    async def inspect_state(self, cid):
        return None

    async def logs(self, cid, follow=False, tail=None, max_bytes=None):
        return
        yield  # pragma: no cover


def _svc_job(name: str, *, ai: dict | None = None) -> Job:
    cfg: dict = {"image": "demo:1", "port": 8000}
    if ai:
        cfg["ai"] = ai
    return Job(
        name=name,
        kind=JobKind.service,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="always",
        config=json.dumps(cfg),
    )


def _controller(queries, runtime, secrets) -> ServiceController:
    return ServiceController(
        queries=queries,
        runtime=runtime,
        services_settings=ServicesSettings(service_port_range="9400-9499"),
        secrets=secrets,
    )


async def _shared_resolved_rows(queries) -> list:
    rows, _ = await queries.list_audit_log(action="secret.shared_resolved")
    return rows


async def test_shared_values_never_auto_injected(queries, tmp_path):
    """Shared values flow ONLY through explicit refs — never env.update(shared)."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-shared", "UNRELATED": "never-injected"})
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    await queries.create_job(_svc_job("app", ai={"default": SHARED_API_SPEC}))

    await controller.reconcile()

    job = await queries.get_service_by_name("app")
    assert job.status is JobStatus.running
    env = runtime.run_configs[-1].env
    # The resolved binding carries the shared value under the CONTRACT names...
    assert env["OPENAI_API_KEY"] == "sk-shared"
    assert env["NERDIT_AI_DEFAULT_KEY"] == "sk-shared"
    # ...but the raw shared map never enters the container env.
    assert "OPENAI_KEY" not in env
    assert "UNRELATED" not in env
    await controller.shutdown()


async def test_shared_resolved_audited_once_across_crash_loop(queries, tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-shared"})
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    job = _svc_job("app", ai={"default": SHARED_API_SPEC})
    await queries.create_job(job)

    await controller.reconcile()
    rows = await _shared_resolved_rows(queries)
    assert len(rows) == 1
    assert rows[0].principal_id == "system"
    assert rows[0].target_type == "secret"
    assert rows[0].target_id == "shared"
    assert rows[0].params_redacted == {
        "service_name": "app",
        "keys": ["OPENAI_KEY"],
        "overridden": [],
    }
    # Names only — the value never reaches the audit store.
    assert "sk-shared" not in json.dumps(rows[0].params_redacted)

    # Crash → restarting → (backoff elapsed) relaunch: the identical
    # (keys, overridden) tuple is NOT re-audited.
    cid = (await queries.get_service_by_name("app")).container_id
    runtime.live.pop(cid)
    await controller.reconcile()  # crash detected → restarting
    await queries.record_service_exit(job.id, datetime.now(UTC) - timedelta(seconds=120))
    await controller.reconcile()  # backoff elapsed → relaunch
    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    assert len(await _shared_resolved_rows(queries)) == 1

    # A CHANGED tuple (per-service override appears) audits again.
    mgr.set("app", {"OPENAI_KEY": "sk-own"})
    cid = (await queries.get_service_by_name("app")).container_id
    runtime.live.pop(cid)
    await controller.reconcile()
    await queries.record_service_exit(job.id, datetime.now(UTC) - timedelta(seconds=120))
    await controller.reconcile()
    rows = await _shared_resolved_rows(queries)
    assert len(rows) == 2
    assert rows[0].params_redacted["overridden"] == ["OPENAI_KEY"]  # newest first
    await controller.shutdown()


async def test_shared_resolved_fires_even_when_all_refs_overridden(queries, tmp_path):
    """The all-overridden row is the admin's only bypass signal — it must fire."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("app", {"OPENAI_KEY": "sk-own"})  # per-service override, no shared key at all
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    await queries.create_job(_svc_job("app", ai={"default": SHARED_API_SPEC}))

    await controller.reconcile()

    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    assert runtime.run_configs[-1].env["OPENAI_API_KEY"] == "sk-own"
    rows = await _shared_resolved_rows(queries)
    assert len(rows) == 1
    assert rows[0].params_redacted["keys"] == ["OPENAI_KEY"]
    assert rows[0].params_redacted["overridden"] == ["OPENAI_KEY"]
    await controller.shutdown()


async def test_corrupt_shared_does_not_block_fully_overridden_service(queries, tmp_path):
    """The override escape hatch survives a corrupt shared store.

    Every ${secrets.shared.KEY} ref has a per-service override, so the launch
    path must never decrypt _shared.enc — a corrupt/unreadable shared file must
    not stall a service that does not actually need it.
    """
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-shared"})
    mgr.set("app", {"OPENAI_KEY": "sk-own"})  # override for the only shared ref
    (tmp_path / "secrets" / "_shared.enc").write_text("corrupted", encoding="utf-8")
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    job = _svc_job("app", ai={"default": SHARED_API_SPEC})
    await queries.create_job(job)

    await controller.reconcile()

    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    assert runtime.run_configs[-1].env["OPENAI_API_KEY"] == "sk-own"
    # The bypass signal still fires (the admin's only visibility).
    rows = await _shared_resolved_rows(queries)
    assert len(rows) == 1
    assert rows[0].params_redacted["overridden"] == ["OPENAI_KEY"]
    await controller.shutdown()


async def test_missing_shared_secret_defers_launch_nonterminal(queries, tmp_path):
    mgr = SecretManager(tmp_path / "secrets")  # shared store empty
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    job = _svc_job("app", ai={"default": SHARED_API_SPEC})
    await queries.create_job(job)

    await controller.reconcile()
    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.building  # deferred, never failed
    assert await queries.get_service_endpoint("app") is None  # nothing acquired
    logs = [entry.message for entry in await queries.get_logs(job.id)]
    waits = [line for line in logs if "nerdit secrets set --shared" in line]
    assert len(waits) == 1  # deduped across ticks
    await controller.shutdown()


async def test_corrupt_shared_blocks_only_referencing_services(queries, tmp_path):
    """Lazy load: one corrupt _shared.enc must not stall the whole node."""
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-shared"})
    mgr.set("plain-secrets", {"OPENAI_KEY": "sk-own"})
    (tmp_path / "secrets" / "_shared.enc").write_text("corrupted", encoding="utf-8")
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    shared_app = _svc_job("shared-app", ai={"default": SHARED_API_SPEC})
    await queries.create_job(shared_app)
    await queries.create_job(_svc_job("plain-app"))  # no [ai.*] at all
    await queries.create_job(_svc_job("plain-secrets", ai={"cheap": API_SPEC}))  # unscoped ref

    await controller.reconcile()
    await controller.reconcile()

    # Only the shared-referencing app is blocked (deferred, deduped log line).
    assert (await queries.get_service_by_name("shared-app")).status is JobStatus.building
    logs = [entry.message for entry in await queries.get_logs(shared_app.id)]
    waits = [line for line in logs if "Shared secrets unavailable" in line]
    assert len(waits) == 1
    # The others never touch _shared.enc and launch normally.
    assert (await queries.get_service_by_name("plain-app")).status is JobStatus.running
    assert (await queries.get_service_by_name("plain-secrets")).status is JobStatus.running
    await controller.shutdown()


async def test_corrupt_per_service_secrets_defer_launch_nonterminal(queries, tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("app", {"A": "1"})
    (tmp_path / "secrets" / "app.enc").write_text("corrupted", encoding="utf-8")
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    job = _svc_job("app")
    await queries.create_job(job)

    await controller.reconcile()
    await controller.reconcile()

    row = await queries.get_service_by_name("app")
    assert row.status is JobStatus.building  # non-terminal, retried every tick
    logs = [entry.message for entry in await queries.get_logs(job.id)]
    waits = [line for line in logs if "Secrets unavailable" in line]
    assert len(waits) == 1  # deduped
    # The admin repairs the store (here: drops the poisoned file) → next tick
    # launches without any state reset.
    (tmp_path / "secrets" / "app.enc").unlink()
    await controller.reconcile()
    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    await controller.shutdown()


# --- (P20) ResolvedLaunchEnv.injected_keys ------------------------------------


async def test_resolved_launch_env_reports_the_injected_keys(queries, tmp_path):
    """``injected_keys`` is the D-P14-5 protected set for ``run_once``: the keys
    the binding RESOLVERS actually produced, so a caller-supplied ``--env`` can
    never beat a platform-computed value.

    Derived from the live resolve, never from a names-only spec projection —
    ``inject_db_env_key_names`` reports ``DATABASE_URL`` for a managed
    ``default`` spec whatever the engine, while a Redis backend really injects
    ``REDIS_URL``; a spec-shaped protected set would let the caller win.
    """
    mgr = SecretManager(tmp_path / "secrets")
    mgr.set("app", {"OPENAI_KEY": "sk-live", "APP_TOKEN": "app-token"})
    controller = _controller(queries, _FakeRuntime(), mgr)
    job = _svc_job("app", ai={"default": API_SPEC, "cheap": API_SPEC})
    await queries.create_job(job)
    cfg = json.loads(job.config)

    resolved = await controller._resolve_launch_env(job, cfg, 8000, cfg["ai"], None)

    assert resolved.injected_keys == {
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "NERDIT_AI_DEFAULT_URL",
        "NERDIT_AI_DEFAULT_KEY",
        "NERDIT_AI_DEFAULT_MODEL",
        "NERDIT_AI_CHEAP_URL",
        "NERDIT_AI_CHEAP_KEY",
        "NERDIT_AI_CHEAP_MODEL",
    }
    # Same set the names-only projection reports for these binding names
    # (F7-ENVGRAMMAR: one grammar, two consumers).
    assert resolved.injected_keys == _ai_env_key_names(cfg["ai"])
    # Every injected key is really in the env...
    assert resolved.injected_keys <= set(resolved.env)
    # ...and the set is *only* the injected surface: neither the plain secrets
    # nor the PORT default are protected against a caller override.
    assert "APP_TOKEN" not in resolved.injected_keys
    assert "PORT" not in resolved.injected_keys
    assert resolved.secret_env["APP_TOKEN"] == "app-token"


async def test_resolved_launch_env_injected_keys_empty_without_bindings(queries, tmp_path):
    """A service with no ``[ai.*]``/``[db.*]`` protects nothing."""
    mgr = SecretManager(tmp_path / "secrets")
    controller = _controller(queries, _FakeRuntime(), mgr)
    job = _svc_job("plain")
    await queries.create_job(job)
    cfg = json.loads(job.config)

    resolved = await controller._resolve_launch_env(job, cfg, 8000, None, None)

    assert resolved.injected_keys == set()
    assert resolved.env["PORT"] == "8000"


# --- (P40c) the project scope under the service scope (D-P40-9) ---------------


async def test_launch_env_merges_project_under_service_under_injected(queries, tmp_path):
    """cfg.env < project < service < injected bindings < PORT (Invariant #1 holds)."""
    mgr = SecretManager(tmp_path / "secrets")
    controller = _controller(queries, _FakeRuntime(), mgr)
    job = _svc_job("app", ai={"default": API_SPEC})
    await queries.create_job(job)
    assert job.project_id is not None
    mgr.set(
        project_storage_name(job.project_id),
        {
            "OPENAI_KEY": "sk-project",  # the [ai.default] ref resolves from the project scope
            "OPENAI_BASE_URL": "https://evil.example/v1",  # loses to the injected binding
            "BOTH": "project",
            "ONLY_PROJECT": "p",
            "FROM_CFG": "project-beats-cfg",
        },
    )
    mgr.set("app", {"BOTH": "service"})
    cfg = json.loads(job.config)
    cfg["env"] = {"FROM_CFG": "cfg", "CFG_ONLY": "c"}

    resolved = await controller._resolve_launch_env(job, cfg, 8000, cfg["ai"], None)

    env = resolved.env
    assert env["OPENAI_API_KEY"] == "sk-project"
    assert env["OPENAI_BASE_URL"] == "https://api.example.com/v1"
    assert env["BOTH"] == "service"
    assert env["ONLY_PROJECT"] == "p"
    assert env["FROM_CFG"] == "project-beats-cfg"
    assert env["CFG_ONLY"] == "c"
    assert env["PORT"] == "8000"
    # The scrub set is the merged map -- the plain/secret flag is never read, so
    # a plain project value is masked from run/release/crash tails like a secret.
    assert {"p", "sk-project", "service"} <= _sensitive_env_values(resolved)


async def test_a_row_without_a_project_reads_no_project_scope(queries, tmp_path):
    """Model/database/legacy rows (`project_id` NULL) degrade to service-only."""
    mgr = SecretManager(tmp_path / "secrets")
    controller = _controller(queries, _FakeRuntime(), mgr)
    job = _svc_job("app")
    await queries.create_job(job)
    mgr.set(project_storage_name(job.project_id), {"ONLY_PROJECT": "p"})
    orphan = job.model_copy(update={"project_id": None})

    resolved = await controller._resolve_launch_env(orphan, {"port": 8000}, 8000, None, None)

    assert "ONLY_PROJECT" not in resolved.env


async def test_a_project_scope_key_shadows_shared_and_is_reported_overridden(queries, tmp_path):
    """`unresolved_shared` and `overridden` test the MERGED map: one audit row, no second."""
    mgr = SecretManager(tmp_path / "secrets")
    runtime = _FakeRuntime()
    controller = _controller(queries, runtime, mgr)
    job = _svc_job("app", ai={"default": SHARED_API_SPEC})
    await queries.create_job(job)
    mgr.set(project_storage_name(job.project_id), {"OPENAI_KEY": "sk-project"})
    # A corrupt shared store must not block the launch: the project-scope
    # override means `_shared` is never decrypted.
    mgr.set(SHARED_SCOPE, {"OPENAI_KEY": "sk-shared"})
    (tmp_path / "secrets" / f"{SHARED_SCOPE}.enc").write_bytes(b"corrupt")

    await controller.reconcile()

    assert (await queries.get_service_by_name("app")).status is JobStatus.running
    assert runtime.run_configs[-1].env["OPENAI_API_KEY"] == "sk-project"
    rows = await _shared_resolved_rows(queries)
    assert len(rows) == 1
    assert rows[0].params_redacted["keys"] == ["OPENAI_KEY"]
    assert rows[0].params_redacted["overridden"] == ["OPENAI_KEY"]
    assert "sk-project" not in json.dumps(rows[0].params_redacted)
    await controller.shutdown()


async def test_an_undecryptable_project_scope_defers_the_launch(queries, tmp_path):
    mgr = SecretManager(tmp_path / "secrets")
    controller = _controller(queries, _FakeRuntime(), mgr)
    job = _svc_job("app")
    await queries.create_job(job)
    stem = project_storage_name(job.project_id)
    mgr.set(stem, {"K": "v"})
    (tmp_path / "secrets" / f"{stem}.enc").write_bytes(b"corrupt")

    with pytest.raises(LaunchEnvNotReady) as excinfo:
        await controller._resolve_launch_env(job, {"port": 8000}, 8000, None, None)
    assert excinfo.value.kind == "secrets"
