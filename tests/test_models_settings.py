"""P5 / S3 plumbing tests: `[models]` settings, `extra_port_bind_ips`, pull seam.

Covers the three seams the AI wedge builds on:

* ``ModelsSettings`` defaults + TOML round-trip via ``load_settings``;
* the docker ``ports`` kwargs with/without ``extra_port_bind_ips`` — the
  ``None`` path must stay **byte-identical** to the pre-P5 loopback-only shape
  (the frozen batch/service core);
* ``pull_image`` on the runtime Protocol, ``DockerRuntime`` and ``StubRuntime``;
* the config-store ``models`` section (GET/PUT diagnostics, restart flags).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nerdit.config.settings import ModelsSettings, NerditSettings, load_settings
from nerdit.config.store import ConfigError, ConfigStore
from nerdit.core.runtime.protocol import ContainerRuntime, ContainerRuntimeError
from nerdit.core.runtime.stub import StubRuntime
from nerdit.db.models import ContainerConfig

# --- [models] settings ---------------------------------------------------------


def test_models_settings_defaults():
    settings = NerditSettings()
    assert settings.models.ollama_image == "ollama/ollama"
    assert settings.models.bridge_host == "auto"
    assert settings.models.pull_timeout_s == 1800
    assert settings.models.start_period_s == 300


def test_models_settings_defaults_when_section_absent(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[monitor]\ninterval_seconds = 7\n")
    settings = load_settings(config)
    assert settings.models.ollama_image == "ollama/ollama"
    assert settings.models.bridge_host == "auto"


def test_bridge_auto_resolves_linux(monkeypatch):
    monkeypatch.setattr("nerdit.config.defaults.sys.platform", "linux")
    settings = ModelsSettings()
    assert settings.bridge_bind_ip == "172.17.0.1"
    assert settings.bridge_advertise_host == "172.17.0.1"


def test_bridge_auto_resolves_darwin(monkeypatch):
    # macOS: no extra bind (binding 172.17.0.1 would fail — no such host
    # interface); apps reach the loopback-published port via the VM proxy.
    monkeypatch.setattr("nerdit.config.defaults.sys.platform", "darwin")
    settings = ModelsSettings()
    assert settings.bridge_bind_ip is None
    assert settings.bridge_advertise_host == "host.docker.internal"


def test_bridge_explicit_keeps_dual_role(monkeypatch):
    # Rootless-Docker escape hatch: an explicit value is used verbatim for
    # both roles, on every platform.
    monkeypatch.setattr("nerdit.config.defaults.sys.platform", "darwin")
    settings = ModelsSettings(bridge_host="10.88.0.1")
    assert settings.bridge_bind_ip == "10.88.0.1"
    assert settings.bridge_advertise_host == "10.88.0.1"


def test_models_settings_from_toml(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[models]\n"
        'ollama_image = "ollama/ollama:0.5"\n'
        'bridge_host = "10.88.0.1"\n'
        "pull_timeout_s = 3600\n"
        "start_period_s = 120\n"
    )
    settings = load_settings(config)
    assert settings.models.ollama_image == "ollama/ollama:0.5"
    assert settings.models.bridge_host == "10.88.0.1"
    assert settings.models.pull_timeout_s == 3600
    assert settings.models.start_period_s == 120


def test_models_bridge_host_rejects_all_interfaces():
    # 0.0.0.0 as an extra bind address would LAN-expose model endpoints.
    for bad in ("0.0.0.0", "::", "[::]", "", "  "):
        with pytest.raises(ValidationError):
            ModelsSettings(bridge_host=bad)


def test_models_pull_timeout_must_be_positive():
    with pytest.raises(ValidationError):
        ModelsSettings(pull_timeout_s=0)


# --- extra_port_bind_ips → docker ports kwargs ----------------------------------


def _config(**kw) -> ContainerConfig:
    kw.setdefault("gpu_ids", [])
    return ContainerConfig(image="nerdit-runtime:0.1", **kw)


@pytest.mark.asyncio
async def test_ports_kwargs_without_extras_byte_identical(mock_docker):
    """extras=None keeps the frozen loopback-only kwargs shape (tuple, not list)."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(ports={8000: 9400}))
    kwargs = mock_docker.containers.run.call_args.kwargs
    # The exact pre-P5 shape: a plain ('ip', port) tuple per container port.
    assert kwargs["ports"] == {"8000/tcp": ("127.0.0.1", 9400)}
    assert type(kwargs["ports"]["8000/tcp"]) is tuple


@pytest.mark.asyncio
async def test_ports_kwargs_with_extras_adds_bind_ips(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(ports={11434: 9410}, extra_port_bind_ips=["172.17.0.1"]))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["ports"] == {"11434/tcp": [("127.0.0.1", 9410), ("172.17.0.1", 9410)]}


@pytest.mark.asyncio
async def test_ports_kwargs_empty_extras_keeps_loopback_shape(mock_docker):
    """extras=[] is treated like None: no list-shaped bindings sneak in."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(ports={8000: 9400}, extra_port_bind_ips=[]))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["ports"] == {"8000/tcp": ("127.0.0.1", 9400)}
    assert type(kwargs["ports"]["8000/tcp"]) is tuple


@pytest.mark.asyncio
async def test_extras_without_ports_never_adds_ports_kwarg(mock_docker):
    """Batch path (ports=None) stays untouched even with extras set."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(extra_port_bind_ips=["172.17.0.1"]))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "ports" not in kwargs


def test_container_config_extra_port_bind_ips_defaults_to_none():
    assert _config().extra_port_bind_ips is None


# --- pull_image seam -------------------------------------------------------------


def test_protocol_declares_pull_image():
    assert hasattr(ContainerRuntime, "pull_image")


@pytest.mark.asyncio
async def test_docker_pull_image_calls_images_pull(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.pull_image("ollama/ollama")
    mock_docker.images.pull.assert_called_once_with("ollama/ollama")


@pytest.mark.asyncio
async def test_docker_pull_image_maps_api_error(mock_docker):
    # docker.py holds a reference to the real docker module from import time;
    # raise the real APIError so the except clause matches (test_containers idiom).
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.pull.side_effect = _dmod.docker.errors.APIError("registry down")
    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(ContainerRuntimeError, match="Failed to pull image"):
        await runtime.pull_image("ollama/ollama")


@pytest.mark.asyncio
async def test_docker_pull_image_maps_image_not_found(mock_docker):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.pull.side_effect = _dmod.docker.errors.ImageNotFound("no such image")
    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(ContainerRuntimeError, match="not found in registry"):
        await runtime.pull_image("nope/nope")


@pytest.mark.asyncio
async def test_stub_pull_image_raises():
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await StubRuntime().pull_image("ollama/ollama")


# --- config store: [models] section (GET/PUT) ------------------------------------


def _store(tmp_path, body: str = "") -> ConfigStore:
    path = tmp_path / "config.toml"
    if body:
        path.write_text(body)
    return ConfigStore(path)


def test_models_section_is_known(tmp_path):
    assert "models" in _store(tmp_path).known_sections()


def test_models_effective_defaults(tmp_path):
    values = _store(tmp_path).effective_section("models")
    assert values["ollama_image"] == "ollama/ollama"
    assert values["bridge_host"] == "auto"
    assert values["pull_timeout_s"] == 1800
    assert values["start_period_s"] == 300


def test_models_config_api_round_trip(tmp_path):
    store = _store(tmp_path)
    store.commit(store.stage("models", {"bridge_host": "10.88.0.1"}))
    raw = store.load_raw()
    assert raw["models"]["bridge_host"] == "10.88.0.1"
    assert store.effective_section("models")["bridge_host"] == "10.88.0.1"


def test_models_fields_require_restart(tmp_path):
    # [models] follows the [proxy] classification: settings are bound at
    # startup (controllers capture ModelsSettings at construction), so every
    # field change needs a daemon restart to take effect.
    store = _store(tmp_path)
    assert store.stage("models", {"ollama_image": "ollama/ollama:0.5"}).requires_restart is True
    assert store.stage("models", {"bridge_host": "10.88.0.1"}).requires_restart is True
    assert store.stage("models", {"pull_timeout_s": 3600}).requires_restart is True
    assert store.stage("models", {"start_period_s": 60}).requires_restart is True


def test_models_stage_rejects_all_interfaces_bridge_host(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("models", {"bridge_host": "0.0.0.0"})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert exc.value.diagnostics


def test_models_stage_rejects_unknown_key(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("models", {"bridge_hots": "10.0.0.1"})  # typo
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert ["bridge_hots"] in [d.loc for d in exc.value.diagnostics]


def test_models_default_backend_validator():
    from nerdit.config.settings import ModelsSettings

    assert ModelsSettings(default_backend="vllm").default_backend == "vllm"
    with pytest.raises(ValueError, match="not a known backend"):
        ModelsSettings(default_backend="nope")


def test_models_vllm_fields_require_restart(tmp_path):
    # P11 vLLM knobs follow the same startup-bound classification.
    store = _store(tmp_path)
    assert store.stage("models", {"default_backend": "vllm"}).requires_restart is True
    assert store.stage("models", {"vllm_image": "vllm/vllm-openai:v0.6"}).requires_restart is True
    assert store.stage("models", {"vllm_shm_size": "2g"}).requires_restart is True
    assert store.stage("models", {"vllm_extra_args": ["--foo"]}).requires_restart is True
