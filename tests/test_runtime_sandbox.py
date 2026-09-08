"""Tests for DockerRuntime sandbox hardening (P1 / S5).

Covers Tier-A denied-mount enforcement (symlink-resolved, admin included),
the wiring of ``cap_drop`` / ``read_only`` / ``network_mode``, and the AMD
``security_opt`` *append* fix so ``no-new-privileges`` and ``seccomp=unconfined``
coexist instead of clobbering each other.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nerdit.core.runtime.protocol import ContainerStartError, SandboxViolationError
from nerdit.db.models import ContainerConfig, GpuVendor


def _config(volumes=None, **kw):
    kw.setdefault("gpu_ids", ["GPU-0001"])
    return ContainerConfig(
        image="nerdit-runtime:0.1",
        command=["python", "train.py"],
        volumes=volumes,
        **kw,
    )


# --- Tier-A denied-mount enforcement ---


@pytest.mark.parametrize(
    "denied",
    ["/etc", "/etc/passwd", "/root", "/", "/proc", "/sys", "/boot", "/dev"],
)
@pytest.mark.asyncio
async def test_tier_a_rejects_denied_path(mock_docker, denied):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(SandboxViolationError) as exc:
        await runtime.run(_config(volumes={denied: "/x"}))
    assert exc.value.reason == "denied_mount"
    # Docker is never reached when the mount is rejected.
    mock_docker.containers.run.assert_not_called()


@pytest.mark.asyncio
async def test_sandbox_violation_is_container_start_error(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    # The scheduler relies on this subclassing to fail the job & release GPUs.
    with pytest.raises(ContainerStartError):
        await runtime.run(_config(volumes={"/etc": "/x"}))


@pytest.mark.asyncio
async def test_tier_a_rejects_nerdit_home(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    home_nerdit = str(Path("~/.nerdit").expanduser())
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={home_nerdit: "/x"}))
    # A child of ~/.nerdit is rejected too.
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={f"{home_nerdit}/config.toml": "/x"}))


@pytest.mark.asyncio
async def test_system_mount_roots_exempt_daemon_owned_dir(mock_docker, tmp_path):
    """P5 regression: the Ollama weights cache lives under ``<data_dir>/models``
    (i.e. under the denied ``~/.nerdit``). ``system_mount_roots`` must carve it
    out of Tier-A so ``nerdit serve <model>`` launches — while sibling paths
    under the same denied root stay blocked.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    data_dir = tmp_path / ".nerdit"
    models = data_dir / "models"
    (models / "ollama").mkdir(parents=True)
    runtime = DockerRuntime(
        client=mock_docker,
        denied_mount_paths=[str(data_dir)],
        system_mount_roots=[str(models)],
    )
    # The daemon-owned weights volume is accepted.
    cid = await runtime.run(_config(volumes={str(models / "ollama"): "/root/.ollama"}))
    assert cid == "container-abc123"
    # A sibling under the denied data dir but NOT under the system root is still
    # rejected (the exemption is narrow, not the whole ~/.nerdit).
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={str(data_dir / "secrets"): "/x"}))


@pytest.mark.asyncio
async def test_system_mount_roots_still_reject_symlink_escape(mock_docker, tmp_path):
    """A symlink planted under a system root that escapes to a denied path is
    still blocked — the exemption is resolve-before-check, not a blanket pass.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    models = tmp_path / "models"
    models.mkdir()
    runtime = DockerRuntime(client=mock_docker, system_mount_roots=[str(models)])
    link = models / "escape"
    os.symlink("/etc", link)
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={str(link): "/x"}))


@pytest.mark.asyncio
async def test_tier_a_rejects_symlink_to_docker_sock(mock_docker, tmp_path):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    link = tmp_path / "sneaky.sock"
    os.symlink("/var/run/docker.sock", link)  # target need not exist
    with pytest.raises(SandboxViolationError) as exc:
        await runtime.run(_config(volumes={str(link): "/var/run/docker.sock"}))
    assert exc.value.reason == "denied_mount"


@pytest.mark.asyncio
async def test_tier_a_rejects_docker_desktop_socket(mock_docker):
    """The Docker Desktop (macOS) socket under the user's home is denied too."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    sock = str(Path("~/.docker/run/docker.sock").expanduser())
    with pytest.raises(SandboxViolationError) as exc:
        await runtime.run(_config(volumes={sock: "/var/run/docker.sock"}))
    assert exc.value.reason == "denied_mount"


@pytest.mark.asyncio
async def test_tier_a_rejects_symlink_to_nerdit_home(mock_docker, tmp_path):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    link = tmp_path / "sneaky-home"
    os.symlink(str(Path("~/.nerdit").expanduser()), link)
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={str(link): "/x"}))


@pytest.mark.asyncio
async def test_tier_a_accepts_benign_mount(mock_docker, tmp_path):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cid = await runtime.run(_config(volumes={str(workspace): "/workspace"}))
    assert cid == "container-abc123"
    mock_docker.containers.run.assert_called_once()


@pytest.mark.asyncio
async def test_tier_a_runs_for_custom_denylist(mock_docker, tmp_path):
    """A configured denylist is honored (policy-as-data)."""
    from nerdit.core.runtime.docker import DockerRuntime

    secret = tmp_path / "secret"
    secret.mkdir()
    runtime = DockerRuntime(client=mock_docker, denied_mount_paths=[str(secret)])
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={str(secret / "k"): "/x"}))


@pytest.mark.parametrize("ancestor", ["/run", "/var/run", "/var", "/"])
@pytest.mark.asyncio
async def test_tier_a_rejects_ancestor_of_denied_file(mock_docker, ancestor):
    """L3: mounting a parent dir of docker.sock (e.g. /run, /var/run) is blocked.

    The original ``_match_denied`` only caught equality/descendants, so
    mounting ``/run`` exposed the real socket. Holds even for admin (Tier-A).
    """
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    with pytest.raises(SandboxViolationError) as exc:
        await runtime.run(_config(volumes={ancestor: "/host"}))
    assert exc.value.reason == "denied_mount"


@pytest.mark.asyncio
async def test_tier_a_allowed_root_exemption_requires_resolved_form(mock_docker, tmp_path):
    """L4: a symlink under an allowed root whose target escapes is NOT exempted.

    The exemption must hold only when every path form (literal AND resolved)
    is under an allowed root, so resolve-before-check isn't undone.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    # A symlink that lives under the allowed upload root but points at /etc.
    escape = uploads / "escape"
    os.symlink("/etc", escape)
    runtime = DockerRuntime(client=mock_docker, allowed_mount_roots=[str(uploads)])

    # A genuine file under uploads is still allowed.
    benign = uploads / "code"
    benign.mkdir()
    assert await runtime.run(_config(volumes={str(benign): "/workspace"})) == "container-abc123"

    # The escaping symlink is rejected despite living under the allowed root.
    with pytest.raises(SandboxViolationError):
        await runtime.run(_config(volumes={str(escape): "/x"}))


# --- Capability / privilege / network wiring ---


@pytest.mark.asyncio
async def test_cap_drop_and_no_new_privileges_wired(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(cap_drop=["ALL"], no_new_privileges=True))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in kwargs["security_opt"]


@pytest.mark.asyncio
async def test_network_mode_and_read_only_wired(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(network_mode="none", read_only=True))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["network_mode"] == "none"
    assert kwargs["read_only"] is True


@pytest.mark.asyncio
async def test_defaults_leave_runtime_untouched(mock_docker):
    """No sandbox fields set → no cap_drop/network_mode/security_opt kwargs."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config())
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "cap_drop" not in kwargs
    assert "network_mode" not in kwargs
    assert "read_only" not in kwargs
    assert "security_opt" not in kwargs


# --- Service-mode published ports (P2 / S3) ---


@pytest.mark.asyncio
async def test_ports_published_on_loopback(mock_docker):
    """``config.ports`` → docker-py ``ports`` kwarg bound to 127.0.0.1 only."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(ports={8000: 9400}))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["ports"] == {"8000/tcp": ("127.0.0.1", 9400)}


@pytest.mark.asyncio
async def test_no_ports_kwarg_when_ports_none(mock_docker):
    """Batch path (ports is None) must never add a ``ports`` kwarg."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config())
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "ports" not in kwargs


@pytest.mark.asyncio
async def test_shm_size_kwarg_only_when_set(mock_docker):
    """``config.shm_size`` (P11 vLLM) → docker-py ``shm_size``; absent otherwise."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config())
    assert "shm_size" not in mock_docker.containers.run.call_args.kwargs

    await runtime.run(_config(shm_size="1g"))
    assert mock_docker.containers.run.call_args.kwargs["shm_size"] == "1g"


# --- GPU passthrough only when GPUs were allocated (Codex Comment 3) ---


@pytest.mark.asyncio
async def test_no_gpu_device_request_without_gpu_ids(mock_docker):
    """A CPU-only workload (gpus=0 service → empty gpu_ids) must NOT request GPU
    passthrough, even with the default NVIDIA vendor — an empty DeviceRequest
    would demand the NVIDIA runtime and fail on a host without it."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(gpu_ids=[], vendor=GpuVendor.nvidia))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "device_requests" not in kwargs


@pytest.mark.asyncio
async def test_gpu_device_request_when_gpu_ids_present(mock_docker):
    """Batch path (gpu_ids present) still requests NVIDIA passthrough — unchanged."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(gpu_ids=["GPU-0001"], vendor=GpuVendor.nvidia))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "device_requests" in kwargs


# --- AMD security_opt APPEND fix ---


@pytest.mark.asyncio
async def test_amd_security_opt_merges_seccomp_and_no_new_privileges(mock_docker, monkeypatch):
    import grp

    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr(
        grp, "getgrnam", lambda name: type("G", (), {"gr_gid": {"video": 44, "render": 110}[name]})
    )
    runtime = DockerRuntime(client=mock_docker)
    config = _config(
        vendor=GpuVendor.amd,
        gpu_ids=["0", "1"],
        no_new_privileges=True,
        cap_drop=["ALL"],
    )
    await runtime.run(config)
    kwargs = mock_docker.containers.run.call_args.kwargs
    sec_opt = kwargs["security_opt"]
    # BOTH must be present — the AMD branch must append, not replace.
    assert "seccomp=unconfined" in sec_opt
    assert "no-new-privileges:true" in sec_opt
    assert kwargs["cap_drop"] == ["ALL"]


@pytest.mark.asyncio
async def test_amd_security_opt_without_no_new_privileges(mock_docker, monkeypatch):
    """Backward compat: AMD with no_new_privileges=False keeps just seccomp."""
    import grp

    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr(
        grp, "getgrnam", lambda name: type("G", (), {"gr_gid": {"video": 44, "render": 110}[name]})
    )
    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(_config(vendor=GpuVendor.amd, gpu_ids=["0"]))
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["security_opt"] == ["seccomp=unconfined"]
