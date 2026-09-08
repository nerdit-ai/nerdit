"""Tests for vendor-neutral GPU discovery backends."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from nerdit.core.discovery import (
    GpuDiscoveryError,
    NvmlBackend,
    ZmlSmiBackend,
    amd_gpu_access_diagnostic,
    amd_gpu_present,
    discover_gpu_system,
)
from nerdit.db.models import GpuVendor


def _zml_payload() -> dict:
    return {
        "devices": [
            {
                "cuda": {
                    "name": "NVIDIA H100 80GB",
                    "util_percent": 25,
                    "mem_used_bytes": 1024 * 1024 * 1024,
                    "mem_total_bytes": 81920 * 1024 * 1024,
                    "temperature": 55,
                }
            },
            {
                "rocm": {
                    "name": "AMD Instinct MI300X",
                    "util_percent": 75,
                    "mem_used_bytes": 2048 * 1024 * 1024,
                    "mem_total_bytes": 192 * 1024 * 1024 * 1024,
                    "temperature": 61,
                }
            },
        ],
        "processes": [],
    }


def _mock_zml(monkeypatch, payload=None, *, returncode=0, stderr=""):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=json.dumps(payload if payload is not None else _zml_payload()),
        stderr=stderr,
    )
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=completed))
    return completed


def test_zml_smi_discovers_mixed_vendor_inventory(monkeypatch):
    _mock_zml(monkeypatch)

    backend = ZmlSmiBackend()
    gpus = backend.discover()

    assert [gpu.id for gpu in gpus] == ["gpu:nvidia:0", "gpu:amd:0"]
    assert gpus[0].runtime_id == "0"
    assert gpus[0].memory_mb == 81920
    assert gpus[0].schedulable is True
    assert gpus[1].vendor == GpuVendor.amd
    assert gpus[1].memory_mb == 196608
    assert gpus[1].schedulable is False


def test_zml_smi_collects_metrics(monkeypatch):
    _mock_zml(monkeypatch)

    metrics = ZmlSmiBackend().collect_metrics()

    assert metrics["gpu:nvidia:0"].utilization_percent == 25
    assert metrics["gpu:nvidia:0"].memory_used_mb == 1024
    assert metrics["gpu:amd:0"].temperature_c == 61


def test_zml_smi_tolerates_missing_optional_fields(monkeypatch):
    _mock_zml(monkeypatch, {"devices": [{"rocm": {"name": None}}], "processes": []})

    backend = ZmlSmiBackend()
    gpu = backend.discover()[0]
    metrics = backend.collect_metrics()[gpu.id]

    assert gpu.name == "AMD GPU 0"
    assert gpu.memory_mb == 0
    assert metrics.utilization_percent is None
    assert metrics.temperature_c is None


@pytest.mark.parametrize(
    ("stdout", "returncode", "message"),
    [
        ("not-json", 0, "invalid JSON"),
        (json.dumps({"processes": []}), 0, "devices list"),
        ("{}", 1, "zml-smi failed"),
    ],
)
def test_zml_smi_reports_invalid_output(monkeypatch, stdout, returncode, message):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    completed = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr="backend failure"
    )
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=completed))

    with pytest.raises(GpuDiscoveryError, match=message):
        ZmlSmiBackend().discover()


def test_nvml_backend_preserves_uuid_as_runtime_selector(mock_pynvml):
    gpus = NvmlBackend().discover()

    assert [gpu.id for gpu in gpus] == ["gpu:nvidia:0", "gpu:nvidia:1"]
    assert gpus[0].runtime_id == "GPU-0000-0001"
    assert gpus[0].compute_cap == "9.0"
    mock_pynvml.nvmlInit.assert_called_once()
    mock_pynvml.nvmlShutdown.assert_called_once()


def test_discovery_falls_back_to_nvml_when_zml_missing(monkeypatch, mock_pynvml):
    monkeypatch.setattr("shutil.which", lambda _name: None)

    result = discover_gpu_system()

    assert result.backend.name == "nvml"
    assert len(result.gpus) == 2


def test_discovery_fails_when_no_backend_is_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    fake = MagicMock()
    fake.nvmlInit = MagicMock(side_effect=RuntimeError("driver not found"))
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    with pytest.raises(GpuDiscoveryError, match="zml_smi.*nvml"):
        discover_gpu_system()


def test_amd_gpu_access_diagnostic_reports_kfd_permissions(tmp_path, monkeypatch):
    device = tmp_path / "0000:03:00.0"
    device.mkdir()
    (device / "vendor").write_text("0x1002\n")
    (device / "class").write_text("0x030000\n")
    kfd = tmp_path / "kfd"
    kfd.touch()
    monkeypatch.setattr("nerdit.core.discovery.os.access", lambda *_args: False)

    assert amd_gpu_present(tmp_path) is True
    assert "cannot access /dev/kfd" in amd_gpu_access_diagnostic(tmp_path, kfd)


def test_discovery_warns_when_amd_hardware_is_hidden_by_nvml(monkeypatch, mock_pynvml, caplog):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("nerdit.core.discovery.amd_gpu_present", lambda *_args: True)
    monkeypatch.setattr(
        "nerdit.core.discovery.amd_gpu_access_diagnostic",
        lambda *_args: "AMD GPU hardware is present, but the daemon user cannot access /dev/kfd.",
    )

    with caplog.at_level(logging.WARNING):
        discover_gpu_system()

    assert any("cannot access /dev/kfd" in record.message for record in caplog.records)


def test_zml_smi_amd_schedulable_when_enabled(monkeypatch):
    _mock_zml(monkeypatch)

    gpus = ZmlSmiBackend(amd_schedulable=True).discover()

    assert gpus[0].vendor == GpuVendor.nvidia
    assert gpus[0].schedulable is True
    assert gpus[1].vendor == GpuVendor.amd
    assert gpus[1].schedulable is True


def test_zml_smi_amd_not_schedulable_by_default(monkeypatch):
    _mock_zml(monkeypatch)

    gpus = ZmlSmiBackend().discover()

    assert gpus[1].vendor == GpuVendor.amd
    assert gpus[1].schedulable is False


def test_discover_gpu_system_threads_amd_schedulable(monkeypatch):
    _mock_zml(monkeypatch)

    result = discover_gpu_system(amd_schedulable=True)

    amd = [gpu for gpu in result.gpus if gpu.vendor == GpuVendor.amd]
    assert amd and all(gpu.schedulable for gpu in amd)
