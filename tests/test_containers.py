"""Tests for DockerRuntime with mocked docker-py."""

from unittest.mock import MagicMock, patch

import pytest

from nerdit.core.runtime.protocol import ContainerStartError
from nerdit.db.models import ContainerConfig


@pytest.mark.asyncio
async def test_docker_run(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        gpu_ids=["GPU-0001"],
        command=["python", "train.py"],
    )
    container_id = await runtime.run(config)
    assert container_id == "container-abc123"
    mock_docker.containers.run.assert_called_once()


@pytest.mark.asyncio
async def test_docker_stop(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.stop("container-abc123")
    container = mock_docker.containers.get.return_value
    container.stop.assert_called_once()


@pytest.mark.asyncio
async def test_docker_wait(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    exit_code = await runtime.wait("container-abc123")
    assert exit_code == 0


@pytest.mark.asyncio
async def test_docker_remove(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.remove("container-abc123", force=True)
    container = mock_docker.containers.get.return_value
    container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_docker_status(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    status = await runtime.status("container-abc123")
    assert status == "running"


@pytest.mark.asyncio
async def test_docker_logs_no_follow(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    lines = []
    async for line in runtime.logs("container-abc123", follow=False):
        lines.append(line)
    assert lines == ["hello world"]


@pytest.mark.asyncio
async def test_docker_run_nvidia_toolkit_error(mock_docker):
    # Get the real docker.errors.APIError that docker.py catches
    # (docker.py holds a reference to the real module from import time)
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    real_api_error = _dmod.docker.errors.APIError

    mock_docker.containers.run.side_effect = real_api_error(
        "nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1"
    )

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        gpu_ids=["GPU-0001"],
        command=["python", "train.py"],
    )
    with pytest.raises(ContainerStartError, match="NVIDIA driver library"):
        await runtime.run(config)


@pytest.mark.asyncio
async def test_check_nvidia_runtime_present(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.info = MagicMock(return_value={"Runtimes": {"nvidia": {}, "runc": {}}})
    runtime = DockerRuntime(client=mock_docker)
    with patch("nerdit.core.runtime.docker.Path.exists", return_value=True):
        result = await runtime.check_nvidia_runtime()
    assert result is True


@pytest.mark.asyncio
async def test_check_nvidia_runtime_missing(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.info = MagicMock(return_value={"Runtimes": {"runc": {}}})
    runtime = DockerRuntime(client=mock_docker)
    result = await runtime.check_nvidia_runtime()
    assert result is False


# --- image_exists tests ---


@pytest.mark.asyncio
async def test_image_exists_found(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.get = MagicMock(return_value=MagicMock())
    runtime = DockerRuntime(client=mock_docker)
    result = await runtime.image_exists("test:latest")
    assert result is True
    mock_docker.images.get.assert_called_once_with("test:latest")


@pytest.mark.asyncio
async def test_image_exists_not_found(mock_docker):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    real_image_not_found = _dmod.docker.errors.ImageNotFound
    mock_docker.images.get = MagicMock(side_effect=real_image_not_found("not found"))
    runtime = DockerRuntime(client=mock_docker)
    result = await runtime.image_exists("nonexistent:latest")
    assert result is False


@pytest.mark.asyncio
async def test_list_images_returns_sorted_tags(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    def _img(tags):
        img = MagicMock()
        img.tags = tags
        return img

    mock_docker.images.list = MagicMock(
        return_value=[
            _img(["nerdit-runtime:0.1"]),
            _img(["rocm/dev-ubuntu-22.04:6.3", "nerdit-runtime-rocm:0.1"]),
            _img([]),  # dangling image: no tags
            _img(["<none>:<none>"]),
        ]
    )
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.list_images() == [
        "nerdit-runtime-rocm:0.1",
        "nerdit-runtime:0.1",
        "rocm/dev-ubuntu-22.04:6.3",
    ]


@pytest.mark.asyncio
async def test_list_images_degrades_to_empty_on_error(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.images.list = MagicMock(side_effect=ConnectionError("dockerd down"))
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.list_images() == []


# --- AMD ROCm passthrough tests ---


@pytest.mark.asyncio
async def test_docker_run_amd_passthrough_kwargs(mock_docker, monkeypatch):
    import grp

    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    monkeypatch.setattr(
        grp, "getgrnam", lambda name: MagicMock(gr_gid={"video": 44, "render": 110}[name])
    )

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime-rocm:0.1",
        gpu_ids=["0", "1"],
        vendor=GpuVendor.amd,
        command=["python", "train.py"],
        env={"FOO": "bar", "ROCR_VISIBLE_DEVICES": "spoofed"},
    )
    await runtime.run(config)

    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["devices"] == ["/dev/kfd", "/dev/dri"]
    assert kwargs["group_add"] == ["44", "110"]
    assert kwargs["security_opt"] == ["seccomp=unconfined"]
    assert kwargs["environment"]["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert kwargs["environment"]["FOO"] == "bar"
    assert "device_requests" not in kwargs


@pytest.mark.asyncio
async def test_docker_run_amd_missing_groups_omitted(mock_docker, monkeypatch, caplog):
    """Missing host groups are omitted from group_add, never passed by name
    (names resolve against the container's /etc/group, the wrong namespace)."""
    import grp

    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    def _missing(name):
        raise KeyError(name)

    monkeypatch.setattr(grp, "getgrnam", _missing)

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime-rocm:0.1",
        gpu_ids=["0"],
        vendor=GpuVendor.amd,
        command=["python", "train.py"],
    )
    with caplog.at_level("WARNING"):
        await runtime.run(config)
    assert mock_docker.containers.run.call_args.kwargs["group_add"] == []
    assert any("Host group" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_docker_run_nvidia_path_unchanged(mock_docker):
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime:0.1",
        gpu_ids=["GPU-0001"],
        vendor=GpuVendor.nvidia,
        command=["python", "train.py"],
    )
    await runtime.run(config)
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert "device_requests" in kwargs
    assert "devices" not in kwargs
    assert "group_add" not in kwargs


@pytest.mark.asyncio
async def test_batch_launch_kwargs_byte_identical(mock_docker):
    """Pin the batch (GPU) launch kwargs so the P13 additive ``inspect_state``
    (and any future runtime addition) cannot perturb the frozen batch launch.

    A canonical NVIDIA batch container carries exactly this key set — no
    sandbox/service kwargs (ports, cap_drop, network_mode, security_opt,
    nano_cpus, shm_size, log_config) leak onto it — and ``inspect_state`` never
    calls ``containers.run``.
    """
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime:0.1",
        gpu_ids=["GPU-0001"],
        vendor=GpuVendor.nvidia,
        command=["python", "train.py"],
    )
    await runtime.run(config)
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert set(kwargs) == {
        "image",
        "command",
        "volumes",
        "environment",
        "working_dir",
        "mem_limit",
        "detach",
        "labels",
        "device_requests",
    }
    assert kwargs["image"] == "nerdit-runtime:0.1"
    assert kwargs["command"] == ["python", "train.py"]
    assert kwargs["volumes"] is None
    assert kwargs["environment"] is None
    assert kwargs["working_dir"] is None
    assert kwargs["mem_limit"] is None
    assert kwargs["detach"] is True
    # Default instance_id when the runtime is constructed bare (tests / single
    # daemon). The ``nerdit-instance`` label always rides alongside managed-by.
    assert kwargs["labels"] == {"managed-by": "nerdit", "nerdit-instance": "default"}
    # (P14b) log_config is byte-absent when the field is None (batch never sets
    # it) — the set-equality above already proves it; assert it explicitly.
    assert "log_config" not in kwargs

    # inspect_state is read-only — it must never launch a container.
    mock_docker.containers.run.reset_mock()
    await runtime.inspect_state("container-abc123")
    mock_docker.containers.run.assert_not_called()


@pytest.mark.asyncio
async def test_docker_run_log_config_kwarg(mock_docker, monkeypatch):
    """(P14b) ``ContainerConfig.log_config`` → a docker ``LogConfig(json-file)``
    kwarg carrying the caps; absent (byte-identical) when None."""
    from nerdit.core.runtime import docker as dockmod
    from nerdit.core.runtime.docker import DockerRuntime

    # Deterministic recorder for docker.types.LogConfig (real class or mock,
    # depending on test order — pin it either way).
    calls: list[tuple[str, dict]] = []

    def _fake_log_config(*, type, config):
        calls.append((type, config))
        return {"_type": type, "_config": config}

    monkeypatch.setattr(dockmod.docker.types, "LogConfig", _fake_log_config)
    runtime = DockerRuntime(client=mock_docker)

    # None ⇒ kwarg absent, LogConfig never constructed.
    await runtime.run(ContainerConfig(image="nerdit-app/x:1", gpu_ids=[], command=["run"]))
    assert "log_config" not in mock_docker.containers.run.call_args.kwargs
    assert calls == []

    # Set ⇒ a json-file LogConfig carrying the exact caps is passed through.
    mock_docker.containers.run.reset_mock()
    await runtime.run(
        ContainerConfig(
            image="nerdit-app/x:1",
            gpu_ids=[],
            command=["run"],
            log_config={"max-size": "10m", "max-file": "3"},
        )
    )
    assert calls == [("json-file", {"max-size": "10m", "max-file": "3"})]
    assert mock_docker.containers.run.call_args.kwargs["log_config"] == {
        "_type": "json-file",
        "_config": {"max-size": "10m", "max-file": "3"},
    }


@pytest.mark.asyncio
async def test_docker_inspect_state_reads_state(mock_docker):
    """``inspect_state`` projects ``attrs['State']`` into ``ContainerStateInfo``."""
    from nerdit.core.runtime.docker import DockerRuntime

    container = mock_docker.containers.get.return_value
    container.attrs = {
        "State": {
            "ExitCode": 137,
            "OOMKilled": True,
            "Error": "boom",
            "StartedAt": "2026-07-10T00:00:00Z",
            "FinishedAt": "2026-07-10T00:01:00Z",
        }
    }
    runtime = DockerRuntime(client=mock_docker)
    state = await runtime.inspect_state("container-abc123")
    assert state is not None
    assert state.exit_code == 137
    assert state.oom_killed is True
    assert state.error == "boom"
    assert state.finished_at == "2026-07-10T00:01:00Z"


@pytest.mark.asyncio
async def test_docker_inspect_state_not_found_returns_none(mock_docker):
    """A missing container yields ``None`` (mirrors ``status``)."""
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    mock_docker.containers.get.side_effect = _dmod.docker.errors.NotFound("gone")
    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.inspect_state("nope") is None


@pytest.mark.asyncio
async def test_docker_container_running_is_tri_state(mock_docker):
    """``container_running`` separates "inert" from "cannot tell" — ``status`` cannot.

    This is the distinction the DELETE data purge (F12) rests on. Two properties:
    an ``APIError`` is typically an unreachable daemon, and a container we cannot see
    is NOT a container that stopped (containerd keeps it running while dockerd is
    down) — so it must answer ``None``, not ``False``. And an *exited* container is
    inert even though it still exists, so presence alone is the wrong question.
    ``status`` collapses not-found and cannot-answer into the same ``None``; asserting
    that side-by-side is what stops someone "simplifying" the probe back onto it.
    """
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    container = mock_docker.containers.get.return_value
    mock_docker.containers.get.side_effect = None

    container.status = "running"
    assert await runtime.container_running("c1") is True

    container.status = "paused"  # frozen, not finished — an unpause resumes the writes
    assert await runtime.container_running("c1") is True

    container.status = "exited"  # present, but writing nothing
    assert await runtime.container_running("c1") is False

    mock_docker.containers.get.side_effect = _dmod.docker.errors.NotFound("gone")
    assert await runtime.container_running("c1") is False
    assert await runtime.status("c1") is None

    mock_docker.containers.get.side_effect = _dmod.docker.errors.APIError("dockerd down")
    assert await runtime.container_running("c1") is None  # unknown, NOT False
    assert await runtime.status("c1") is None  # ...whereas status cannot tell you that


@pytest.mark.asyncio
async def test_docker_run_rocm_error_mapped(mock_docker):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    real_api_error = _dmod.docker.errors.APIError
    mock_docker.containers.run.side_effect = real_api_error(
        "error gathering device information while adding custom device "
        '"/dev/kfd": no such file or directory'
    )

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime-rocm:0.1",
        gpu_ids=["0"],
        vendor=GpuVendor.amd,
        command=["python", "train.py"],
    )
    with pytest.raises(ContainerStartError, match="AMD ROCm device access failed"):
        await runtime.run(config)


@pytest.mark.asyncio
async def test_check_rocm_runtime_devices_present(mock_docker, monkeypatch, tmp_path):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    kfd = tmp_path / "kfd"
    dri = tmp_path / "dri"
    kfd.touch()
    dri.mkdir()
    monkeypatch.setattr(_dmod, "_ROCM_DEVICE_PATHS", (kfd, dri))
    monkeypatch.setattr(
        "nerdit.core.discovery.amd_gpu_access_diagnostic", lambda *args, **kwargs: None
    )

    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.check_rocm_runtime() is True


@pytest.mark.asyncio
async def test_check_rocm_runtime_missing_kfd(mock_docker, monkeypatch, tmp_path):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    monkeypatch.setattr(
        _dmod, "_ROCM_DEVICE_PATHS", (tmp_path / "missing-kfd", tmp_path / "missing-dri")
    )

    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.check_rocm_runtime() is False


@pytest.mark.asyncio
async def test_check_rocm_runtime_swallows_diagnostic_exception(
    mock_docker, monkeypatch, tmp_path, caplog
):
    """A crashing diagnostic must degrade to False, never propagate (the
    lifespan relies on readiness checks not throwing)."""
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime

    kfd = tmp_path / "kfd"
    dri = tmp_path / "dri"
    kfd.touch()
    dri.mkdir()
    monkeypatch.setattr(_dmod, "_ROCM_DEVICE_PATHS", (kfd, dri))

    def _boom(*args, **kwargs):
        raise ConnectionError("nss is down")

    monkeypatch.setattr("nerdit.core.discovery.amd_gpu_access_diagnostic", _boom)

    runtime = DockerRuntime(client=mock_docker)
    with caplog.at_level("WARNING"):
        assert await runtime.check_rocm_runtime() is False
    assert any("Could not check ROCm" in r.message for r in caplog.records)


# --- Vendor-gated error classification ---


@pytest.mark.asyncio
async def test_docker_run_nvidia_error_with_hsa_substring_not_misclassified(mock_docker):
    """An NVIDIA job whose error text merely contains 'hsa' (e.g. a mount
    path) must get the generic message, not ROCm install instructions."""
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    real_api_error = _dmod.docker.errors.APIError
    mock_docker.containers.run.side_effect = real_api_error(
        'invalid mount config for type "bind": bind source path does not exist: /home/hsanchez/data'
    )

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime:0.1",
        gpu_ids=["GPU-0001"],
        vendor=GpuVendor.nvidia,
        command=["python", "train.py"],
    )
    with pytest.raises(ContainerStartError, match="Failed to start container"):
        await runtime.run(config)


@pytest.mark.asyncio
async def test_docker_run_amd_error_with_nvidia_substring_not_misclassified(mock_docker):
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.db.models import GpuVendor

    real_api_error = _dmod.docker.errors.APIError
    mock_docker.containers.run.side_effect = real_api_error("libnvidia-ml.so.1 missing")

    runtime = DockerRuntime(client=mock_docker)
    config = ContainerConfig(
        image="nerdit-runtime-rocm:0.1",
        gpu_ids=["0"],
        vendor=GpuVendor.amd,
        command=["python", "train.py"],
    )
    with pytest.raises(ContainerStartError, match="Failed to start container"):
        await runtime.run(config)


# ---- Instance-scoped container ownership (multi-daemon on one Docker host) ----


@pytest.mark.asyncio
async def test_create_label_includes_instance_id(mock_docker):
    """Every created container carries ``nerdit-instance=<instance_id>`` next to
    ``managed-by=nerdit`` so a co-located sibling daemon can tell them apart."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker, instance_id="b")
    config = ContainerConfig(
        image="nerdit-runtime:0.1",
        gpu_ids=["GPU-0001"],
        command=["python", "train.py"],
    )
    await runtime.run(config)
    kwargs = mock_docker.containers.run.call_args.kwargs
    assert kwargs["labels"] == {"managed-by": "nerdit", "nerdit-instance": "b"}


@pytest.mark.asyncio
async def test_sweep_listing_is_instance_scoped(mock_docker):
    """The SWEEP listing (``list_own_managed_containers``, which feeds the
    killer) is instance-scoped: two daemons never see each other's containers.

    docker AND-combines a multi-value label filter, so instance ``a`` asks
    dockerd for ``managed-by=nerdit`` AND ``nerdit-instance=a`` — instance
    ``b``'s containers (and pre-upgrade, unlabelled ones) never come back. We
    assert on the filter that reaches docker-py rather than re-implementing
    dockerd's matching.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    runtime_a = DockerRuntime(client=mock_docker, instance_id="a")
    await runtime_a.list_own_managed_containers()
    filters_a = mock_docker.containers.list.call_args.kwargs["filters"]
    assert filters_a == {"label": ["managed-by=nerdit", "nerdit-instance=a"]}

    # A sibling daemon on the same host scopes to its own instance_id instead.
    runtime_b = DockerRuntime(client=mock_docker, instance_id="b")
    await runtime_b.list_own_managed_containers()
    filters_b = mock_docker.containers.list.call_args.kwargs["filters"]
    assert filters_b == {"label": ["managed-by=nerdit", "nerdit-instance=b"]}
    assert "nerdit-instance=a" not in filters_b["label"]


@pytest.mark.asyncio
async def test_readoption_listing_is_unscoped(mock_docker):
    """The RE-ADOPTION listing (``list_managed_containers``, used by the service
    controller to match live containers to DB rows) is UNSCOPED — it queries
    only ``managed-by=nerdit`` with NO instance filter.

    This is the fix for the migration footgun: on an in-place upgrade the
    daemon's OWN pre-upgrade containers carry no ``nerdit-instance`` label, so
    an instance-scoped re-adoption query would miss them, the controller would
    judge live services dead, and relaunch them into host-port conflicts.
    Unscoped listing re-adopts them correctly (the controller only ever acts on
    ids present in its own DB, so a sibling's containers here are inert).
    """
    from nerdit.core.runtime.docker import DockerRuntime

    runtime_a = DockerRuntime(client=mock_docker, instance_id="a")
    await runtime_a.list_managed_containers()
    filters = mock_docker.containers.list.call_args.kwargs["filters"]
    assert filters == {"label": "managed-by=nerdit"}
    assert "nerdit-instance=a" not in str(filters)


@pytest.mark.asyncio
async def test_sweep_does_not_return_cross_instance_containers(mock_docker):
    """Given a mixed set on the host, a daemon's SWEEP only surfaces the
    containers its own instance filter would have matched — cross-instance and
    pre-upgrade (unlabelled) containers are excluded by dockerd, so the sweep
    never sees them and cannot kill them."""
    from nerdit.core.runtime.docker import DockerRuntime

    def _c(cid: str) -> MagicMock:
        c = MagicMock()
        c.id = cid
        c.attrs = {"Created": "2026-01-01T00:00:00.000000Z"}
        return c

    own = _c("own-1")

    def _list(filters=None):
        # Emulate dockerd AND-matching: only return this daemon's own container
        # when its instance filter is present; a mislabelled/cross-instance
        # query would come back empty.
        labels = (filters or {}).get("label", [])
        if "nerdit-instance=a" in labels:
            return [own]
        return []

    mock_docker.containers.list = MagicMock(side_effect=_list)

    runtime_a = DockerRuntime(client=mock_docker, instance_id="a")
    listed = await runtime_a.list_own_managed_containers()
    assert [cid for cid, _ in listed] == ["own-1"]

    # Instance ``b`` asks dockerd for its own label and gets nothing back — it
    # never sees (and so never sweeps) instance ``a``'s container.
    runtime_b = DockerRuntime(client=mock_docker, instance_id="b")
    assert await runtime_b.list_own_managed_containers() == []


@pytest.mark.asyncio
async def test_list_managed_keeps_containers_with_unparseable_created(mock_docker):
    """A missing or garbage `Created` must not drop a live container from the
    reconcile live set (that relaunches a duplicate); it reads as created now,
    so the zombie sweep never reaps it early either."""
    from datetime import UTC, datetime, timedelta

    from nerdit.core.runtime.docker import DockerRuntime

    garbage, missing = MagicMock(), MagicMock()
    garbage.id, garbage.attrs = "garbage", {"Created": "garbage"}
    missing.id, missing.attrs = "missing", {}
    mock_docker.containers.list = MagicMock(return_value=[garbage, missing])

    listed = await DockerRuntime(client=mock_docker).list_managed_containers()
    assert [cid for cid, _ in listed] == ["garbage", "missing"]
    assert all(datetime.now(UTC) - ts < timedelta(minutes=1) for _, ts in listed)


# --- (P20) bounded log tail: `tail` + `max_bytes` -----------------------------


def _set_log_stream(mock_docker, chunks):
    """Point the mocked container's ``logs`` at a chunk generator.

    Mirrors docker-py's non-TTY contract: ``stream=True`` returns a generator of
    **byte chunks** whose boundaries are multiplex frames, not lines.
    """
    container = mock_docker.containers.get.return_value

    def _logs(**kwargs):
        if kwargs.get("stream"):
            return iter(list(chunks))
        return b"".join(chunks)

    container.logs = MagicMock(side_effect=_logs)
    return container


@pytest.mark.asyncio
async def test_docker_logs_legacy_path_kwargs_are_byte_identical(mock_docker):
    """(P20) With neither bound set, the read is the pre-P20 whole-buffer one.

    Pinned as a frozen kwarg set: every pre-P20 caller (the follow-less
    ``/logs`` read) must keep hitting ``stream=False`` with no ``tail``, or a
    service's full log history silently starts being truncated.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    container = _set_log_stream(mock_docker, [b"a\nb\n"])
    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.logs("cid")]

    assert lines == ["a", "b"]
    assert container.logs.call_args.kwargs == {"stream": False, "follow": False}


@pytest.mark.asyncio
async def test_docker_logs_tail_streams_and_forwards_tail_to_docker(mock_docker):
    """A line cap is pushed down to ``docker logs --tail`` on a streaming read."""
    from nerdit.core.runtime.docker import DockerRuntime

    container = _set_log_stream(mock_docker, [b"one\ntw", b"o\nthree\n"])
    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.logs("cid", tail=3)]

    # Frame boundaries are NOT line boundaries — the caller re-splits.
    assert lines == ["one", "two", "three"]
    assert container.logs.call_args.kwargs == {"stream": True, "follow": False, "tail": 3}


@pytest.mark.asyncio
async def test_docker_logs_max_bytes_alone_still_takes_the_bounded_path(mock_docker):
    """A byte budget with no line cap must not be silently ignored.

    ``tail`` goes down as the explicit ``"all"`` sentinel rather than leaning on
    docker-py's coercion of a non-int value.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    container = _set_log_stream(mock_docker, [b"0123456789\n"])
    runtime = DockerRuntime(client=mock_docker)
    lines = [line async for line in runtime.logs("cid", max_bytes=4)]

    assert lines == ["789"]  # the trailing 4 bytes are b"789\n"
    assert container.logs.call_args.kwargs == {"stream": True, "follow": False, "tail": "all"}


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 1, 5, 11, 12, 10_000])
async def test_docker_logs_byte_budget_keeps_the_most_recent_bytes(mock_docker, budget):
    """Whatever the budget, what survives is exactly the trailing slice."""
    from nerdit.core.runtime.docker import DockerRuntime

    chunks = [b"aaa", b"bbb\n", b"ccc", b"ddd\n"]
    whole = b"".join(chunks)
    _set_log_stream(mock_docker, chunks)
    runtime = DockerRuntime(client=mock_docker)

    lines = [line async for line in runtime.logs("cid", max_bytes=budget)]
    assert lines == whole[len(whole) - budget :].decode().splitlines()


@pytest.mark.asyncio
async def test_docker_logs_one_oversized_line_is_bounded(mock_docker):
    """Security S11: a megabyte without a newline is ONE line to a line cap.

    Only the byte budget bounds it, and the surviving fragment is the tail of
    the line (recency beats alignment for a crash tail).
    """
    from nerdit.core.runtime.docker import DockerRuntime

    blob = b"x" * 1000 + b"y" * 1000  # no newline anywhere
    _set_log_stream(mock_docker, [blob[i : i + 64] for i in range(0, len(blob), 64)])
    runtime = DockerRuntime(client=mock_docker)

    lines = [line async for line in runtime.logs("cid", tail=200, max_bytes=128)]
    assert lines == ["y" * 128]


@pytest.mark.asyncio
async def test_docker_logs_byte_budget_is_enforced_mid_stream(mock_docker):
    """Security S11: the budget is applied WHILE consuming, never by slicing an
    already-materialized buffer.

    Proven without measuring memory: every chunk is a ``bytes`` subclass that
    decrements a liveness counter in ``__del__``, so the producer knows exactly
    how many emitted chunks the reader is still holding. A
    materialize-then-slice implementation would hold all 400 until the stream
    ended; a rolling head-drop holds only the budget's worth (plus the
    consumer's current loop variable).
    """
    from nerdit.core.runtime.docker import DockerRuntime

    n_chunks, chunk_size, budget = 400, 1024, 4096
    census = {"alive": 0, "peak": 0, "made": 0}

    class _Chunk(bytes):
        """Refcount-tracked chunk: CPython frees it the instant it is dropped."""

        def __del__(self) -> None:
            census["alive"] -= 1

    def _stream():
        for _ in range(n_chunks):
            chunk = _Chunk(b"z" * chunk_size)
            census["made"] += 1
            census["alive"] += 1
            census["peak"] = max(census["peak"], census["alive"])
            yield chunk

    container = mock_docker.containers.get.return_value
    container.logs = MagicMock(return_value=_stream())
    runtime = DockerRuntime(client=mock_docker)

    lines = [line async for line in runtime.logs("cid", tail=10, max_bytes=budget)]
    # The whole stream really was produced...
    assert census["made"] == n_chunks
    # ...and the correct trailing slice survived (no newlines ⇒ one long line).
    assert lines == ["z" * budget]
    # Retained ≈ budget/chunk_size chunks; the slack covers the generator's and
    # the consumer's live loop variables. Nowhere near 400.
    assert census["peak"] <= (budget // chunk_size) + 4


@pytest.mark.asyncio
async def test_docker_logs_follow_ignores_the_new_bounds(mock_docker):
    """A follow stream is unbounded by construction — the consumer bounds it by
    disconnecting — so ``tail``/``max_bytes`` must not reach docker-py there."""
    from nerdit.core.runtime.docker import DockerRuntime

    container = mock_docker.containers.get.return_value
    container.logs = MagicMock(return_value=iter([b"live\n"]))
    runtime = DockerRuntime(client=mock_docker)

    lines = [line async for line in runtime.logs("cid", follow=True, tail=1, max_bytes=1)]
    assert lines == ["live"]
    assert "tail" not in container.logs.call_args.kwargs
    assert "max_bytes" not in container.logs.call_args.kwargs


@pytest.mark.asyncio
async def test_docker_logs_follow_drops_oldest_under_backpressure(mock_docker, monkeypatch):
    """B4: a bounded follow queue keeps the newest lines and the end sentinel."""
    import types

    from nerdit.core.runtime import docker as docker_mod

    class _InlineThread:
        # Run the reader to completion before the consumer's first read.
        def __init__(self, target, **_kw):  # noqa: ANN001
            self._target = target

        def start(self) -> None:
            self._target()

    monkeypatch.setattr(docker_mod, "_FOLLOW_QUEUE_MAX", 3)
    monkeypatch.setattr(docker_mod, "threading", types.SimpleNamespace(Thread=_InlineThread))
    container = mock_docker.containers.get.return_value
    container.logs = MagicMock(return_value=iter([f"l{i}\n".encode() for i in range(8)]))
    runtime = docker_mod.DockerRuntime(client=mock_docker)

    lines = [line async for line in runtime.logs("cid", follow=True, since=1700000000)]
    assert lines == ["l6", "l7"]
    assert container.logs.call_args.kwargs["since"] == 1700000000


# --- (P20) bounded wait -------------------------------------------------------


@pytest.mark.asyncio
async def test_docker_wait_unbounded_passes_timeout_none(mock_docker):
    """The legacy call is byte-identical: ``timeout=None`` means no HTTP timeout."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    assert await runtime.wait("container-abc123") == 0
    container = mock_docker.containers.get.return_value
    assert container.wait.call_args.kwargs == {"timeout": None}


@pytest.mark.asyncio
async def test_docker_wait_read_timeout_becomes_asyncio_timeout(mock_docker):
    """Reality R5: the BLOCKING call carries the budget; expiry (a requests
    ``ReadTimeout``) surfaces to the caller as ``asyncio.TimeoutError``."""
    import asyncio

    import requests.exceptions

    from nerdit.core.runtime.docker import DockerRuntime

    container = mock_docker.containers.get.return_value
    container.wait = MagicMock(side_effect=requests.exceptions.ReadTimeout("read timed out"))
    runtime = DockerRuntime(client=mock_docker)

    with pytest.raises(asyncio.TimeoutError):
        await runtime.wait("container-abc123", timeout_s=7)
    assert container.wait.call_args.kwargs == {"timeout": 7}


@pytest.mark.asyncio
async def test_docker_wait_unix_socket_read_timeout_becomes_asyncio_timeout(mock_docker):
    """The shape a REAL docker daemon raises — found by the P20 WP3 live run.

    The test above pins docker-py's *documented* expiry signal. Over the unix
    socket the read actually expires while requests is streaming the response
    body, and ``iter_content`` re-wraps urllib3's ``ReadTimeoutError`` as a
    plain ``ConnectionError`` — which is NOT a ``requests.exceptions.Timeout``.
    Against real Docker the old code therefore let ``ConnectionError`` escape
    ``run_once`` instead of settling ``timed_out=True``.
    """
    import asyncio

    import requests.exceptions
    import urllib3.exceptions

    from nerdit.core.runtime.docker import DockerRuntime

    container = mock_docker.containers.get.return_value
    container.wait = MagicMock(
        side_effect=requests.exceptions.ConnectionError(
            urllib3.exceptions.ReadTimeoutError(None, "/v1.51/containers/x/wait", "Read timed out.")
        )
    )
    runtime = DockerRuntime(client=mock_docker)

    with pytest.raises(asyncio.TimeoutError):
        await runtime.wait("container-abc123", timeout_s=5)


@pytest.mark.asyncio
async def test_docker_wait_genuine_connection_error_still_propagates(mock_docker):
    """A dead docker daemon is also a ``ConnectionError`` — it must NOT be
    mistaken for an expired wait, or a run against a down daemon would report
    ``timed_out=True`` and the caller would never learn docker is gone."""
    import asyncio

    import requests.exceptions

    from nerdit.core.runtime.docker import DockerRuntime

    container = mock_docker.containers.get.return_value
    container.wait = MagicMock(
        side_effect=requests.exceptions.ConnectionError("connection refused")
    )
    runtime = DockerRuntime(client=mock_docker)

    with pytest.raises(requests.exceptions.ConnectionError):
        await runtime.wait("container-abc123", timeout_s=5)
    # Explicitly NOT an asyncio.TimeoutError.
    assert not isinstance(asyncio.TimeoutError(), requests.exceptions.ConnectionError)


@pytest.mark.asyncio
async def test_docker_wait_not_found_on_the_wait_leg_maps_to_not_found(mock_docker):
    """A container reaped between ``get`` and ``wait`` is a NotFound, not a 500."""
    import nerdit.core.runtime.docker as _dmod
    from nerdit.core.runtime.protocol import ContainerNotFoundError

    container = mock_docker.containers.get.return_value
    container.wait = MagicMock(side_effect=_dmod.docker.errors.NotFound("gone"))
    runtime = _dmod.DockerRuntime(client=mock_docker)

    with pytest.raises(ContainerNotFoundError):
        await runtime.wait("container-abc123", timeout_s=1)


@pytest.mark.asyncio
async def test_docker_wait_runs_off_the_shared_executor(mock_docker):
    """Reality R5 / the post-P15 pool-pinning class: N concurrent bounded waits
    must ALL be blocking simultaneously.

    ``asyncio.to_thread`` would cap at the default executor's
    ``min(32, cpus + 4)`` workers, so with 40 in flight the last ones would
    never enter ``container.wait`` until an earlier one returned — and a wait
    lasts the container's whole lifetime. A dedicated thread per wait has no
    such ceiling. The thread names are asserted too, so the mechanism (not just
    the symptom) is pinned.
    """
    import asyncio
    import threading
    import time

    import requests.exceptions

    from nerdit.core.runtime.docker import DockerRuntime

    concurrency = 40  # strictly above the default executor's 32-worker ceiling
    release = threading.Event()
    lock = threading.Lock()
    thread_names: list[str] = []

    def _blocking_wait(timeout=None):
        with lock:
            thread_names.append(threading.current_thread().name)
        release.wait(15)
        raise requests.exceptions.ReadTimeout("read timed out")

    mock_docker.containers.get.return_value.wait = _blocking_wait
    runtime = DockerRuntime(client=mock_docker)

    tasks = [
        asyncio.ensure_future(runtime.wait(f"container-{i:04d}", timeout_s=0.01))
        for i in range(concurrency)
    ]
    try:
        deadline = time.monotonic() + 15
        while len(thread_names) < concurrency and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert len(thread_names) == concurrency, (
            f"only {len(thread_names)}/{concurrency} waits were blocking at once — "
            "the wait is pinning a bounded thread pool"
        )
        assert all(name.startswith("docker-wait-") for name in thread_names)
    finally:
        release.set()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, asyncio.TimeoutError) for r in results)


# --- (P20 WP3) run/release attribution labels --------------------------------


@pytest.mark.asyncio
async def test_extra_labels_merge_under_the_platform_labels(mock_docker):
    """``extra_labels`` is spread FIRST, the platform labels LAST.

    Reversing the spread order would be a spoofable-ownership hole: a caller
    able to set ``managed-by``/``nerdit-instance`` could hide a container from
    this daemon's zombie sweep, or point it at a co-located sibling's instance.
    """
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(
        ContainerConfig(
            image="nerdit-app/x:1",
            gpu_ids=[],
            command=["python", "migrate.py"],
            extra_labels={"nerdit-run": "r", "nerdit-job": "j", "managed-by": "evil"},
        )
    )

    assert mock_docker.containers.run.call_args.kwargs["labels"] == {
        "nerdit-run": "r",
        "nerdit-job": "j",
        "managed-by": "nerdit",
        "nerdit-instance": "default",
    }


@pytest.mark.asyncio
async def test_list_own_labeled_containers_is_instance_scoped(mock_docker):
    """The reap is a KILL path, so the filter always ANDs ``nerdit-instance``
    (the PR #81 regression class): a co-located sibling daemon's containers
    must never be surfaced here."""
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.stub import StubRuntime

    container = MagicMock()
    container.id = "orphan-1"
    mock_docker.containers.list = MagicMock(return_value=[container])
    runtime = DockerRuntime(client=mock_docker, instance_id="inst-a")

    assert await runtime.list_own_labeled_containers("nerdit-job", "abc") == ["orphan-1"]
    assert mock_docker.containers.list.call_args.kwargs["filters"] == {
        "label": ["managed-by=nerdit", "nerdit-instance=inst-a", "nerdit-job=abc"]
    }

    # Docker-less daemon: the read degrades to empty, never raises.
    assert await StubRuntime().list_own_labeled_containers("nerdit-job", "abc") == []


@pytest.mark.asyncio
async def test_list_own_run_containers_filters_on_label_presence(mock_docker):
    """The boot-side orphan kill lists ``nerdit-run`` by PRESENCE (a bare label
    name in a docker filter list matches any value), still ANDed with
    ``nerdit-instance`` — it is a KILL path, so a co-located sibling daemon's
    containers must never be surfaced (the PR #81 regression class)."""
    from nerdit.core.runtime.docker import DockerRuntime
    from nerdit.core.runtime.stub import StubRuntime

    container = MagicMock()
    container.id = "run-orphan-1"
    mock_docker.containers.list = MagicMock(return_value=[container])
    runtime = DockerRuntime(client=mock_docker, instance_id="inst-a")

    assert await runtime.list_own_run_containers() == ["run-orphan-1"]
    assert mock_docker.containers.list.call_args.kwargs["filters"] == {
        "label": ["managed-by=nerdit", "nerdit-instance=inst-a", "nerdit-run"]
    }

    # Docker-less daemon: the read degrades to empty, never raises.
    assert await StubRuntime().list_own_run_containers() == []


# --- P34: the port-contract claim the MCP SANDBOX_NOTE makes -------------------
#
# ``SANDBOX_NOTE`` tells an agent, unconditionally, that a port below 1024 is
# bindable inside the container without any capability — because Docker sets
# ``net.ipv4.ip_unprivileged_port_start=0`` inside a container's OWN network
# namespace. That is only true while every container gets one, i.e. while the
# launch path never uses host networking. This is the test that keeps the
# documented claim honest; if a host-network mode is ever introduced, the note
# has to become conditional and this fails first.


def test_every_launch_path_forces_its_own_network_namespace():
    from nerdit.config.settings import ContainerSettings
    from nerdit.core.launch import (
        AppContainerSpec,
        apply_platform_overlay,
        build_app_container_config,
        build_run_container_config,
    )

    cs = ContainerSettings()
    cfg: dict = {"image": "demo:1"}

    app = build_app_container_config(
        cfg,
        cs,
        AppContainerSpec(
            command=None,
            volumes=None,
            env=None,
            workdir=None,
            gpu_ids=[],
            vendor="nvidia",
            container_port=80,
            host_port=31000,
        ),
    )
    run = build_run_container_config(
        cfg, cs, image="demo:1", command=["true"], env=None, workdir=None
    )
    # The model/database branch gets its base shape from a backend and is
    # hardened by the overlay — same forced bridge, one line later.
    backend_built = ContainerConfig(image="ollama:1", gpu_ids=[])
    apply_platform_overlay(backend_built, cfg, cs, 11434, 31001)

    for config in (app, run, backend_built):
        assert config.network_mode == "bridge"
        assert config.network_mode != "host"


@pytest.mark.asyncio
async def test_pids_limit_passed_to_docker(mock_docker):
    """Security S3: a set pids_limit reaches docker-py (None adds no kwarg — the batch pin)."""
    from nerdit.core.runtime.docker import DockerRuntime

    runtime = DockerRuntime(client=mock_docker)
    await runtime.run(ContainerConfig(image="app:1", gpu_ids=[], pids_limit=512))
    assert mock_docker.containers.run.call_args.kwargs["pids_limit"] == 512
