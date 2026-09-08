"""Tests for the P5 ``ModelController`` — off-tick image pull + ensure_model.

Pure-async unit tests over the in-memory DB, a fake runtime and a fake backend
(no TestClient). The lifecycle integration (pull → launch → ensure_model via
the ``ServiceController``) lives in ``test_services_reconcile.py``; this file
exercises the controller's own guards and persistence directly.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from nerdit.config.settings import ModelsSettings
from nerdit.core.models.backend import (
    DEFAULT_OLLAMA_IMAGE,
    ModelPullError,
    ModelServerUnreachableError,
    OllamaBackend,
    VllmBackend,
)
from nerdit.core.models.controller import ModelController
from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.db.models import ErrorClass, Job, JobKind, JobStatus

pytestmark = pytest.mark.asyncio


class FakeRuntime:
    """Tracks image presence and pull calls; can fail pulls on demand."""

    def __init__(self) -> None:
        self.present: set[str] = set()
        self.pulled: list[str] = []
        self.pull_error: str | None = None

    async def image_exists(self, image_name: str) -> bool:
        return image_name in self.present

    async def pull_image(self, image: str) -> None:
        if self.pull_error:
            raise ContainerRuntimeError(self.pull_error)
        self.pulled.append(image)
        self.present.add(image)

    async def inspect_state(self, container_id: str):
        # (P13) Every runtime fake must carry the forensics seam so a crash path
        # never leans on the defensive bare-except backstop to swallow AttributeError.
        return None


class FakeBackend:
    """Records ensure_model calls; can fail on demand."""

    name = "fake"

    def __init__(self) -> None:
        self.ensured: list[tuple[str, str]] = []
        self.fail = False
        # Number of leading attempts that raise ModelServerUnreachableError
        # (server not listening yet) before succeeding — models the slow start.
        self.unreachable_times = 0

    async def ensure_model(self, base_url: str, model: str) -> None:
        self.ensured.append((base_url, model))
        if self.unreachable_times > 0:
            self.unreachable_times -= 1
            raise ModelServerUnreachableError(f"pull of {model!r} failed: connect refused")
        if self.fail:
            raise ModelPullError(f"pull of {model!r} failed: manifest not found")


def _model(name: str = "ollama-llama3-1-8b", *, pulled: bool = False, **kw: object) -> Job:
    cfg: dict = {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
    }
    if pulled:
        cfg["model_pulled"] = True
    return Job(
        name=name,
        kind=JobKind.model,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
        **kw,
    )


def _controller(queries, runtime, backend, tmp_path) -> ModelController:
    return ModelController(backend, runtime, queries, data_dir=str(tmp_path))


async def _settle(controller: ModelController) -> None:
    """Await every in-flight pull/ensure task (they pop themselves on exit)."""
    for task in list(controller._pull_tasks.values()) + list(controller._ensure_tasks.values()):
        await task


async def _log_lines(queries, job_id: str) -> list[str]:
    return [entry.message for entry in await queries.get_logs(job_id)]


# --- off-tick image pull --------------------------------------------------------


async def test_ensure_image_present_is_true_without_pull(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    runtime.present.add("ollama/ollama")
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is True
    assert runtime.pulled == []
    assert controller._pull_tasks == {}


async def test_missing_image_spawns_exactly_one_pull_task(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)
    cfg = json.loads(job.config)

    # Two ticks while the image is absent: one task, not two (in-flight guard).
    assert await controller.ensure_image(job, cfg) is False
    first_task = controller._pull_tasks[job.id]
    assert await controller.ensure_image(job, cfg) is False
    assert controller._pull_tasks[job.id] is first_task

    await _settle(controller)
    assert runtime.pulled == ["ollama/ollama"]
    # Image now present → the next tick may launch.
    assert await controller.ensure_image(job, cfg) is True
    lines = await _log_lines(queries, job.id)
    assert any("Pulling image ollama/ollama" in line for line in lines)
    assert any("pulled" in line for line in lines)


async def test_pull_failure_settles_row_failed(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    runtime.pull_error = "registry unreachable"
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)

    row = await queries.get_job(job.id)
    # Settled like a failed deploy build: terminal, desired_state matched — the
    # row leaves the reconcilable set instead of retry-looping the registry.
    assert row.status is JobStatus.failed
    assert row.desired_state == JobStatus.failed.value
    # (P13) Pull-fail taxonomy: the persisted row carries the image-pull class +
    # message, not just the audit trail.
    assert row.error_class is ErrorClass.image_pull_fail
    assert row.error_message == "Image pull failed: registry unreachable"
    assert any("Image pull failed" in line for line in await _log_lines(queries, job.id))
    audits, _ = await queries.list_audit_log(limit=10)
    entry = next(a for a in audits if a.action == "model.image_pull_failed")
    assert entry.principal_id == "system"


# --- WP25.5 pin: per-backend image selection (§1.5 fix 4, D-T-6) ----------------
#
# ``POST /models`` always stamps ``config['image']`` (routes/models.py:147); the
# fallback below only fires for pre-P11 rows or hand-edited config — exactly the
# wrong-image trap for a non-default backend. These fixtures build that shape
# directly rather than via ``_model()``, which always carries an explicit image.


def _model_no_image(name: str, backend: str) -> Job:
    cfg = {"model": "llama3.1:8b", "backend": backend, "port": 11434}
    return Job(
        name=name,
        kind=JobKind.model,
        service_name=name,
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
    )


async def test_ensure_image_ollama_row_with_no_image_key_matches_settings_ollama_image(
    queries, tmp_path
):
    """Green today AND after: the fallback for an ollama/unknown-backend row
    equals ``settings.ollama_image`` because ``bootstrap.py`` constructs
    ``OllamaBackend(image=settings.models.ollama_image)`` (the bootstrap
    identity D-T-6 relies on) — asserted directly, not assumed.
    """
    settings = ModelsSettings()
    assert settings.ollama_image == DEFAULT_OLLAMA_IMAGE  # the identity holds
    ollama = OllamaBackend(image=settings.ollama_image)
    runtime = FakeRuntime()
    controller = ModelController(ollama, runtime, queries, data_dir=str(tmp_path))
    job = _model_no_image("ollama-fallback", backend="ollama")
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)

    assert runtime.pulled == [settings.ollama_image]


async def test_ensure_image_vllm_row_with_no_image_key_pulls_vllm_image_not_ollama(
    queries, tmp_path
):
    """RED against pre-WP25.5 code: a vLLM row with no stamped ``image`` used to
    fall back to ``self._settings.ollama_image`` (``models/controller.py:188``)
    and pull the wrong image. After the merge it resolves via
    ``self.backend_for(cfg).image`` (announced fix 4, §1.5/D-T-6).
    """
    settings = ModelsSettings()
    ollama = OllamaBackend(image=settings.ollama_image)
    vllm = VllmBackend(image="vllm/vllm-openai:v0.9.0")
    runtime = FakeRuntime()
    controller = ModelController(
        ollama, runtime, queries, extra_backends={vllm.name: vllm}, data_dir=str(tmp_path)
    )
    job = _model_no_image("vllm-fallback", backend="vllm")
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)

    assert runtime.pulled == ["vllm/vllm-openai:v0.9.0"]
    assert settings.ollama_image not in runtime.pulled


async def test_ensure_image_explicit_config_image_still_wins_over_backend_fallback(
    queries, tmp_path
):
    """An explicit ``config['image']`` (the shape every ``POST /models`` row
    carries) always wins over the per-backend fallback — on the non-default
    backend too, where the bug would otherwise be masked.
    """
    ollama = OllamaBackend(image="ollama/ollama")
    vllm = VllmBackend(image="vllm/vllm-openai:latest")
    runtime = FakeRuntime()
    controller = ModelController(
        ollama, runtime, queries, extra_backends={vllm.name: vllm}, data_dir=str(tmp_path)
    )
    cfg = {
        "model": "llama3.1:8b",
        "backend": "vllm",
        "image": "registry.internal/custom-vllm:pinned",
        "port": 8000,
    }
    job = Job(
        name="explicit-vllm",
        kind=JobKind.model,
        service_name="explicit-vllm",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
    )
    await queries.create_job(job)

    assert await controller.ensure_image(job, cfg) is False
    await _settle(controller)

    assert runtime.pulled == ["registry.internal/custom-vllm:pinned"]


# --- ensure_model (weights) -------------------------------------------------------


async def test_on_running_success_persists_model_pulled(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)

    # ensure_model talks to the server root on LOOPBACK (the daemon is on the
    # host); the bridge-host URL is only for app containers.
    assert backend.ensured == [("http://127.0.0.1:9412", "llama3.1:8b")]
    row = await queries.get_job(job.id)
    assert json.loads(row.config)["model_pulled"] is True
    # Status/desired_state untouched — readiness is the config flag only.
    assert row.status is JobStatus.building
    assert any("ready" in line for line in await _log_lines(queries, job.id))


async def test_on_running_skips_when_already_pulled(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model(pulled=True)
    await queries.create_job(job)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert backend.ensured == []


async def test_on_running_fires_once_per_container(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    backend.fail = True
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 1

    # Same container on later ticks → bounded: NO per-tick retry loop.
    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 1
    assert controller.needs_ensure(Job(**{**job.model_dump(), "container_id": "c1"})) is False

    # A NEW container (crash restart / explicit restart) re-arms the attempt.
    controller.on_running(job, "c2", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 2


async def test_server_unreachable_rearms_and_retries_same_container(queries, tmp_path):
    """A transient 'server not listening yet' must NOT spend the one attempt.

    On a slow container start the first ensure_model hits the port before Ollama
    is up (ModelServerUnreachableError). That is transient: the attempt is
    re-armed so the SAME container retries on the next tick and eventually
    persists model_pulled — never stranded at model_pulled=false. Regression for
    the Codex finding on the eager launch-time attempt.
    """
    runtime, backend = FakeRuntime(), FakeBackend()
    backend.unreachable_times = 2  # two ticks of "not up yet", then success
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    # Tick 1: server not up → re-armed, no audit, nothing persisted.
    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 1
    row = await queries.get_job(job.id)
    assert "model_pulled" not in json.loads(row.config)
    # Re-armed: the SAME container still needs an attempt (not the bounded case).
    assert controller.needs_ensure(Job(**{**row.model_dump(), "container_id": "c1"})) is True
    audits, _ = await queries.list_audit_log(limit=10)
    assert not any(a.action == "model.pull_failed" for a in audits)

    # Tick 2: still not up → re-armed again.
    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 2

    # Tick 3: server up → weights pulled, flag persisted, attempt now spent.
    controller.on_running(job, "c1", 9412)
    await _settle(controller)
    assert len(backend.ensured) == 3
    row = await queries.get_job(job.id)
    assert json.loads(row.config).get("model_pulled") is True
    assert controller.needs_ensure(Job(**{**row.model_dump(), "container_id": "c1"})) is False


async def test_ensure_failure_logs_and_audits_system(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()
    backend.fail = True
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)

    row = await queries.get_job(job.id)
    # NOT settled failed — the row stays for the next launch/adoption to retry.
    assert row.status is JobStatus.building
    assert "model_pulled" not in json.loads(row.config)
    assert any("Model pull failed" in line for line in await _log_lines(queries, job.id))
    audits, _ = await queries.list_audit_log(limit=10)
    entry = next(a for a in audits if a.action == "model.pull_failed")
    assert entry.principal_id == "system"
    assert entry.principal_role == "system"


def _model_with_last_deploy(phase: str = "launching", **kw: object) -> Job:
    cfg = {
        "model": "llama3.1:8b",
        "backend": "ollama",
        "image": "ollama/ollama",
        "port": 11434,
        "build_version": 1,
        "last_deploy": {
            "version": 1,
            "action": "create",
            "phase": phase,
            "image": "ollama/ollama",
            "started_at": "2026-07-10T12:00:00+00:00",
            "updated_at": "2026-07-10T12:00:00+00:00",
            "reason": None,
            "error_class": None,
            "error_message": None,
        },
    }
    return Job(
        name="ollama-llama3-1-8b",
        kind=JobKind.model,
        service_name="ollama-llama3-1-8b",
        gpu_count=0,
        status=JobStatus.building,
        desired_state="running",
        restart_policy="on-failure",
        config=json.dumps(cfg),
        **kw,
    )


async def test_image_pull_failure_stamps_last_deploy_failed(queries, tmp_path):
    """A model image-pull failure marks the deploy generation failed (P13 WP2)."""
    runtime, backend = FakeRuntime(), FakeBackend()
    runtime.pull_error = "registry unreachable"
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model_with_last_deploy()
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)

    ld = json.loads((await queries.get_job(job.id)).config)["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "image_pull_failed"
    assert ld["error_class"] == ErrorClass.image_pull_fail.value


async def test_weights_pull_failure_stamps_last_deploy_failed_row_stays(queries, tmp_path):
    """A weights-pull failure marks the generation failed but leaves the row up."""
    runtime, backend = FakeRuntime(), FakeBackend()
    backend.fail = True
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model_with_last_deploy()
    await queries.create_job(job)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)

    row = await queries.get_job(job.id)
    assert row.status is JobStatus.building  # row stays for the next retry
    ld = json.loads(row.config)["last_deploy"]
    assert ld["phase"] == "failed"
    assert ld["reason"] == "model_pull_failed"
    assert ld["error_class"] is None  # image is fine — only the weights fetch failed


async def test_pull_failure_noop_last_deploy_without_object(queries, tmp_path):
    """A model row with no last_deploy (POST /models) is untouched by the stamp."""
    runtime, backend = FakeRuntime(), FakeBackend()
    runtime.pull_error = "registry unreachable"
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()  # no last_deploy in config
    await queries.create_job(job)

    assert await controller.ensure_image(job, json.loads(job.config)) is False
    await _settle(controller)
    assert "last_deploy" not in json.loads((await queries.get_job(job.id)).config)


async def test_reboot_readoption_refires_ensure(queries, tmp_path):
    # A fresh controller (simulated daemon reboot) has empty in-memory guards:
    # the SAME container id re-enters ensure_model when model_pulled is unset.
    runtime, backend = FakeRuntime(), FakeBackend()
    first = _controller(queries, runtime, backend, tmp_path)
    job = _model(container_id="c1")
    job.status = JobStatus.running
    await queries.create_job(job)

    first.on_running(job, "c1", 9412)
    await _settle(first)
    assert len(backend.ensured) == 1
    # Pretend the pull was interrupted: wipe the flag as if it never persisted.
    await queries.update_job_config(job.id, json.dumps({"model": "llama3.1:8b"}))

    rebooted = _controller(queries, runtime, backend, tmp_path)
    row = await queries.get_job(job.id)
    assert rebooted.needs_ensure(row) is True
    rebooted.on_running(row, "c1", 9412)
    await _settle(rebooted)
    assert len(backend.ensured) == 2


async def test_shutdown_cancels_inflight_tasks(queries, tmp_path):
    runtime, backend = FakeRuntime(), FakeBackend()

    async def slow_ensure(base_url: str, model: str) -> None:
        await asyncio.Event().wait()  # blocks until cancelled

    backend.ensure_model = slow_ensure  # type: ignore[method-assign]
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()
    await queries.create_job(job)
    controller.on_running(job, "c1", 9412)
    assert controller._ensure_tasks

    await controller.shutdown()
    assert controller._ensure_tasks == {}


async def test_build_container_config_sets_bridge_bind(queries, tmp_path):
    from nerdit.core.models.backend import OllamaBackend
    from nerdit.db.models import GpuVendor

    controller = ModelController(
        OllamaBackend(),
        FakeRuntime(),
        queries,
        # Pinned: "auto" resolves per platform (the darwin path is tested below).
        models_settings=ModelsSettings(bridge_host="172.17.0.1"),
        data_dir=str(tmp_path),
    )
    config = controller.build_container_config("llama3.1:8b", [], GpuVendor.nvidia, 9412)

    # Loopback + bridge gateway dual bind (S0-validated design).
    assert config.extra_port_bind_ips == ["172.17.0.1"]
    assert config.ports == {11434: 9412}
    assert config.volumes == {str(tmp_path / "models" / "ollama"): "/root/.ollama"}


async def test_build_container_config_darwin_no_extra_bind(queries, tmp_path, monkeypatch):
    from nerdit.core.models.backend import OllamaBackend
    from nerdit.db.models import GpuVendor

    # macOS + "auto": no extra bind (172.17.0.1 doesn't exist on the host);
    # apps reach the loopback-published port via host.docker.internal.
    monkeypatch.setattr("nerdit.config.defaults.sys.platform", "darwin")
    controller = ModelController(OllamaBackend(), FakeRuntime(), queries, data_dir=str(tmp_path))
    config = controller.build_container_config("llama3.1:8b", [], GpuVendor.nvidia, 9412)

    assert config.extra_port_bind_ips is None
    assert config.ports == {11434: 9412}
    assert controller.bridge_host == "host.docker.internal"


# --- P11: backend registry selection --------------------------------------------


async def test_get_backend_resolves_known_and_default_and_unknown(queries, tmp_path):
    from nerdit.core.models.backend import OllamaBackend, VllmBackend

    ollama, vllm = OllamaBackend(), VllmBackend()
    controller = ModelController(
        ollama,
        FakeRuntime(),
        queries,
        extra_backends={vllm.name: vllm},
        default_backend="ollama",
        data_dir=str(tmp_path),
    )
    assert controller.get_backend("ollama") is ollama
    assert controller.get_backend("vllm") is vllm
    assert controller.get_backend(None) is ollama  # default
    assert controller.get_backend("nope") is None  # explicit unknown → 422 upstream
    # backend_for (reconcile path) falls back to the default on unknown, never None.
    assert controller.backend_for({"backend": "nope"}) is ollama
    assert controller.backend_for({"backend": "vllm"}) is vllm


async def test_build_container_config_routes_to_named_backend(queries, tmp_path):
    from nerdit.core.models.backend import DEFAULT_VLLM_IMAGE, VLLM_PORT, OllamaBackend, VllmBackend

    controller = ModelController(
        OllamaBackend(),
        FakeRuntime(),
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
        data_dir=str(tmp_path),
    )
    from nerdit.db.models import GpuVendor

    cfg = controller.build_container_config(
        "meta-llama/Llama-3.1-8B", ["0"], GpuVendor.nvidia, 9412, backend_name="vllm"
    )
    assert cfg.image == DEFAULT_VLLM_IMAGE
    assert cfg.ports == {VLLM_PORT: 9412}
    assert cfg.shm_size == "1g"
    # None backend_name uses the default (Ollama).
    default_cfg = controller.build_container_config("llama3.1:8b", ["0"], GpuVendor.nvidia, 9412)
    assert "ollama" in default_cfg.image


# --- P21 WP2/WP3: VRAM hint + per-serve engine bounds thread to the backend -------


def _vllm_controller(queries, tmp_path) -> ModelController:
    return ModelController(
        OllamaBackend(),
        FakeRuntime(),
        queries,
        extra_backends={"vllm": VllmBackend()},
        default_backend="ollama",
        data_dir=str(tmp_path),
    )


async def test_build_container_config_threads_gpu_memory_to_backend(queries, tmp_path):
    from nerdit.db.models import GpuVendor

    controller = _vllm_controller(queries, tmp_path)
    cfg = controller.build_container_config(
        "m", ["0"], GpuVendor.nvidia, 9412, backend_name="vllm", gpu_memory_mb=8000
    )
    command = list(cfg.command or [])
    assert command[-4:] == ["--max-model-len", "4096", "--gpu-memory-utilization", "0.85"]


async def test_build_container_config_threads_per_serve_overrides(queries, tmp_path):
    from nerdit.db.models import GpuVendor

    controller = _vllm_controller(queries, tmp_path)
    cfg = controller.build_container_config(
        "m",
        ["0"],
        GpuVendor.nvidia,
        9412,
        backend_name="vllm",
        gpu_memory_mb=8000,
        max_model_len=2048,
    )
    command = list(cfg.command or [])
    assert command.count("--max-model-len") == 1
    assert command[command.index("--max-model-len") + 1] == "2048"
    assert "4096" not in command


async def test_build_container_config_ollama_ignores_engine_params(queries, tmp_path):
    from nerdit.db.models import GpuVendor

    controller = _vllm_controller(queries, tmp_path)
    plain = controller.build_container_config("llama3.1:8b", ["0"], GpuVendor.nvidia, 9412)
    with_params = controller.build_container_config(
        "llama3.1:8b",
        ["0"],
        GpuVendor.nvidia,
        9412,
        gpu_memory_mb=8000,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    assert with_params == plain


async def test_pull_failure_does_not_regress_concurrent_status(queries, tmp_path):
    # PR-review fix (Codex P2): the ModelPullError path must record the error
    # WITHOUT re-asserting the stale ``job`` snapshot status. A row that moved
    # (e.g. restarting -> running via a concurrent restart settling) since the
    # snapshot must keep its live status, or the next reconcile replaces a live
    # container (churn).
    runtime, backend = FakeRuntime(), FakeBackend()
    backend.fail = True
    controller = _controller(queries, runtime, backend, tmp_path)
    job = _model()  # snapshot carries the pre-launch status
    await queries.create_job(job)
    # The row moves on after the snapshot was taken.
    await queries.update_job_status(job.id, JobStatus.running)

    controller.on_running(job, "c1", 9412)
    await _settle(controller)

    row = await queries.get_job(job.id)
    assert row.status is JobStatus.running  # NOT regressed to the stale snapshot
    assert row.error_message is not None and "Model pull failed" in row.error_message
