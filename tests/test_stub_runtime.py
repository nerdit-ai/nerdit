"""Tests for the StubRuntime."""

from __future__ import annotations

import pytest

from nerdit.core.runtime.protocol import ContainerRuntimeError
from nerdit.core.runtime.stub import StubRuntime
from nerdit.db.models import ContainerConfig


@pytest.fixture
def stub():
    return StubRuntime()


@pytest.fixture
def dummy_config():
    return ContainerConfig(
        image="test:latest",
        gpu_ids=["0"],
        command=["python", "train.py"],
    )


@pytest.mark.asyncio
async def test_run_raises(stub, dummy_config):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await stub.run(dummy_config)


@pytest.mark.asyncio
async def test_stop_raises(stub):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await stub.stop("fake-id")


@pytest.mark.asyncio
async def test_kill_raises(stub):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await stub.kill("fake-id")


@pytest.mark.asyncio
async def test_wait_raises(stub):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await stub.wait("fake-id")


@pytest.mark.asyncio
async def test_remove_raises(stub):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        await stub.remove("fake-id")


@pytest.mark.asyncio
async def test_logs_raises(stub):
    with pytest.raises(ContainerRuntimeError, match="Docker is not available"):
        async for _ in stub.logs("fake-id"):
            pass


@pytest.mark.asyncio
async def test_status_returns_none(stub):
    result = await stub.status("fake-id")
    assert result is None


@pytest.mark.asyncio
async def test_inspect_state_returns_none(stub):
    # (P13) The Docker-less stub has no forensics to report.
    assert await stub.inspect_state("fake-id") is None


@pytest.mark.asyncio
async def test_container_running_is_unknown_not_inert(stub):
    """The stub answers "cannot tell" (``None``), NEVER "definitively inert" (``False``).

    Load-bearing: the stub IS the Docker-unreachable state, and rows can still point
    at containers a previous run started that containerd is keeping alive — writing
    into the bind-mounted data dir. A ``False`` here would authorize the DELETE
    data-purge ``rmtree`` out from under a live writer (the caller confirms inertness
    with ``is False``). Asserting `is None` — not merely falsy — is the point: `None`
    and `False` are both falsy, so a weaker assertion would not catch the regression.
    """
    assert await stub.container_running("fake-id") is None
