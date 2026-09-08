"""Hardware-gated ROCm smoke test — requires an AMD GPU host with Docker."""

from __future__ import annotations

from pathlib import Path

import pytest


def _docker_daemon_available() -> bool:
    """Return True only when a Docker daemon is actually reachable.

    /dev/kfd can exist on a host without a usable Docker daemon (not
    installed, stopped, or no socket); in that case the smoke test must skip
    rather than fail the general suite.
    """
    try:
        import docker
    except ImportError:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


requires_rocm = pytest.mark.skipif(
    not Path("/dev/kfd").exists() or not _docker_daemon_available(),
    reason="Requires an AMD KFD device and a reachable Docker daemon",
)


@requires_rocm
@pytest.mark.asyncio
async def test_rocm_container_sees_gpu():
    """A container launched with AMD passthrough can enumerate at least one GPU."""
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import ContainerConfig, GpuVendor

    runtime = DockerRuntime()
    config = ContainerConfig(
        image="nerdit-runtime-rocm:0.1",
        gpu_ids=["0"],
        vendor=GpuVendor.amd,
        command=[
            "python",
            "-c",
            "import amdsmi; amdsmi.amdsmi_init(); "
            "handles = amdsmi.amdsmi_get_processor_handles(); "
            "assert handles, 'no GPU visible in container'; print(len(handles))",
        ],
    )
    container_id = await runtime.run(config)
    try:
        exit_code = await runtime.wait(container_id)
        assert exit_code == 0
    finally:
        await runtime.remove(container_id, force=True)
