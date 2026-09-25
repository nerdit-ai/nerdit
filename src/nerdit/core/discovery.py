"""Vendor-neutral GPU discovery and metrics backends."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from nerdit.db.enums import GpuDiscoveryBackend, GpuVendor
from nerdit.db.rows import Gpu, GpuMetrics

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_ZML_TARGETS = {
    "cuda": GpuVendor.nvidia,
    "rocm": GpuVendor.amd,
}
_AMD_PCI_VENDOR_ID = "0x1002"
_DISPLAY_PCI_CLASS_PREFIX = "0x03"


class GpuDiscoveryError(Exception):
    """Raised when no configured GPU discovery backend can be used."""


class GpuBackend(Protocol):
    """Synchronous inventory and telemetry provider used by the daemon."""

    name: str

    def discover(self) -> list[Gpu]:
        """Return the current GPU inventory."""
        ...

    def collect_metrics(self) -> dict[str, GpuMetrics]:
        """Return metrics keyed by stable Nerdit inventory ID."""
        ...


@dataclass(frozen=True)
class GpuDiscoveryResult:
    """Selected backend and its initial inventory snapshot."""

    backend: GpuBackend
    gpus: list[Gpu]


class ZmlSmiBackend:
    """GPU inventory and telemetry collected from `zml-smi --json`."""

    name = GpuDiscoveryBackend.zml_smi.value

    def __init__(
        self,
        executable: str = "zml-smi",
        timeout_seconds: float = 10.0,
        amd_schedulable: bool = False,
    ) -> None:
        self._executable = executable
        self._timeout = timeout_seconds
        self._amd_schedulable = amd_schedulable

    def _snapshot(self) -> dict:
        executable = self._executable
        if "/" not in executable:
            executable = shutil.which(executable) or ""
        if not executable:
            raise GpuDiscoveryError(f"{self._executable} not found")
        try:
            result = subprocess.run(
                [executable, "--json"],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GpuDiscoveryError(f"zml-smi execution failed: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise GpuDiscoveryError(f"zml-smi failed: {detail}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise GpuDiscoveryError("zml-smi returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("devices"), list):
            raise GpuDiscoveryError("zml-smi JSON is missing the devices list")
        return payload

    @staticmethod
    def _devices(payload: dict) -> list[tuple[GpuVendor, int, dict]]:
        # Per-vendor index follows the devices array order, which is assumed to
        # match the runtime enumeration order (KFD/HSA for ROCm) consumed by
        # ROCR_VISIBLE_DEVICES. The order can change across reboots or hotplug;
        # reconcile_gpus() re-matches inventory on daemon startup.
        vendor_indexes = {GpuVendor.nvidia: 0, GpuVendor.amd: 0}
        devices: list[tuple[GpuVendor, int, dict]] = []
        for raw_device in payload["devices"]:
            if not isinstance(raw_device, dict) or len(raw_device) != 1:
                continue
            target, info = next(iter(raw_device.items()))
            vendor = _ZML_TARGETS.get(target)
            if vendor is None or not isinstance(info, dict):
                continue
            index = vendor_indexes[vendor]
            vendor_indexes[vendor] += 1
            devices.append((vendor, index, info))
        return devices

    def discover(self) -> list[Gpu]:
        gpus = []
        for vendor, index, info in self._devices(self._snapshot()):
            total_bytes = _optional_int(info.get("mem_total_bytes"))
            gpus.append(
                Gpu(
                    id=_inventory_id(vendor, index),
                    name=_device_name(vendor, index, info.get("name")),
                    memory_mb=(total_bytes or 0) // _MIB,
                    vendor=vendor,
                    device_index=index,
                    runtime_id=str(index),
                    discovery_backend=GpuDiscoveryBackend.zml_smi,
                    schedulable=vendor == GpuVendor.nvidia
                    or (vendor == GpuVendor.amd and self._amd_schedulable),
                )
            )
        return gpus

    def collect_metrics(self) -> dict[str, GpuMetrics]:
        metrics = {}
        for vendor, index, info in self._devices(self._snapshot()):
            gpu_id = _inventory_id(vendor, index)
            metrics[gpu_id] = GpuMetrics(
                gpu_id=gpu_id,
                utilization_percent=_optional_int(info.get("util_percent")),
                memory_used_mb=_bytes_to_mb(info.get("mem_used_bytes")),
                memory_total_mb=_bytes_to_mb(info.get("mem_total_bytes")),
                temperature_c=_optional_int(info.get("temperature")),
            )
        return metrics


class NvmlBackend:
    """Compatibility backend for existing NVIDIA/pynvml installations."""

    name = GpuDiscoveryBackend.nvml.value

    @staticmethod
    def _pynvml():
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml
        except Exception as exc:
            raise GpuDiscoveryError(
                "Failed to initialize NVIDIA NVML. Ensure the NVIDIA driver and pynvml "
                "are installed."
            ) from exc

    def discover(self) -> list[Gpu]:
        pynvml = self._pynvml()
        try:
            gpus = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                uuid = _decode(pynvml.nvmlDeviceGetUUID(handle))
                name = _decode(pynvml.nvmlDeviceGetName(handle))
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_cap = f"{major}.{minor}"
                except Exception:
                    compute_cap = None
                gpus.append(
                    Gpu(
                        id=_inventory_id(GpuVendor.nvidia, index),
                        name=name,
                        memory_mb=mem_info.total // _MIB,
                        compute_cap=compute_cap,
                        vendor=GpuVendor.nvidia,
                        device_index=index,
                        runtime_id=uuid,
                        discovery_backend=GpuDiscoveryBackend.nvml,
                        schedulable=True,
                    )
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()

    def collect_metrics(self) -> dict[str, GpuMetrics]:
        pynvml = self._pynvml()
        try:
            metrics = {}
            for index in range(pynvml.nvmlDeviceGetCount()):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_id = _inventory_id(GpuVendor.nvidia, index)
                    metrics[gpu_id] = GpuMetrics(
                        gpu_id=gpu_id,
                        utilization_percent=util.gpu,
                        memory_used_mb=mem.used // _MIB,
                        memory_total_mb=mem.total // _MIB,
                        temperature_c=temp,
                    )
                except Exception:
                    logger.warning("Failed to collect NVML metrics for GPU %d", index)
            return metrics
        finally:
            pynvml.nvmlShutdown()


def discover_gpu_system(
    zml_smi_path: str = "zml-smi", *, amd_schedulable: bool = False
) -> GpuDiscoveryResult:
    """Select zml-smi when usable, otherwise fall back to NVML."""
    failures: list[str] = []
    for backend in (ZmlSmiBackend(zml_smi_path, amd_schedulable=amd_schedulable), NvmlBackend()):
        try:
            gpus = backend.discover()
            _warn_for_missing_amd_inventory(backend, gpus, failures)
            logger.info("Using %s GPU discovery backend", backend.name)
            return GpuDiscoveryResult(backend=backend, gpus=gpus)
        except Exception as exc:
            failures.append(f"{backend.name}: {exc}")
            logger.info("GPU backend %s unavailable: %s", backend.name, exc)
    raise GpuDiscoveryError("; ".join(failures))


def amd_gpu_present(sysfs_root: Path = Path("/sys/bus/pci/devices")) -> bool:
    """Return whether sysfs contains an AMD display-class PCI device."""
    try:
        devices = list(sysfs_root.iterdir())
    except OSError:
        return False
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            device_class = (device / "class").read_text().strip().lower()
        except OSError:
            continue
        if vendor == _AMD_PCI_VENDOR_ID and device_class.startswith(_DISPLAY_PCI_CLASS_PREFIX):
            return True
    return False


def amd_gpu_access_diagnostic(
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
    kfd_path: Path = Path("/dev/kfd"),
) -> str | None:
    """Return an actionable AMD access problem, or `None` when access is usable."""
    if not amd_gpu_present(sysfs_root):
        return None
    if not kfd_path.exists():
        return (
            "AMD GPU hardware is present, but /dev/kfd is missing. "
            "Ensure the amdgpu kernel driver and ROCm KFD support are loaded."
        )
    if not os.access(kfd_path, os.R_OK | os.W_OK):
        return (
            "AMD GPU hardware is present, but the daemon user cannot access /dev/kfd. "
            "Add the user to the render and video groups, then log out and back in."
        )
    return None


def _warn_for_missing_amd_inventory(
    backend: GpuBackend,
    gpus: list[Gpu],
    prior_failures: list[str],
) -> None:
    if not amd_gpu_present() or any(gpu.vendor == GpuVendor.amd for gpu in gpus):
        return
    access_problem = amd_gpu_access_diagnostic()
    if access_problem:
        logger.warning("%s", access_problem)
        return
    if backend.name != GpuDiscoveryBackend.zml_smi.value:
        detail = prior_failures[0] if prior_failures else "zml-smi unavailable"
        logger.warning(
            "AMD GPU hardware is present but cannot be inventoried because the zml-smi "
            "backend is unavailable (%s).",
            detail,
        )
        return
    logger.warning(
        "AMD GPU hardware is present and /dev/kfd is accessible, but zml-smi returned "
        "no ROCm devices. Check the installed zml-smi/AMD SMI version and GPU support."
    )


def _inventory_id(vendor: GpuVendor, index: int) -> str:
    return f"gpu:{vendor.value}:{index}"


def _decode(value) -> str:
    return value if isinstance(value, str) else value.decode("utf-8")


def _device_name(vendor: GpuVendor, index: int, value) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return f"{vendor.value.upper()} GPU {index}"


def _optional_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _bytes_to_mb(value) -> int | None:
    parsed = _optional_int(value)
    return parsed // _MIB if parsed is not None else None
