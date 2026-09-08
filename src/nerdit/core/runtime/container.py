"""Parameters for launching containers, owned by the runtime layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.db.enums import GpuVendor


class ContainerConfig(BaseModel):
    """Parameters for launching a Docker container."""

    image: str = Field(description="Docker image name and tag")
    gpu_ids: list[str] = Field(description="Runtime GPU selectors to expose in the container")
    vendor: GpuVendor = Field(
        default=GpuVendor.nvidia, description="Vendor of the GPUs exposed in the container"
    )
    command: list[str] | None = Field(
        default=None,
        description="Command to run; None defers to the image's own CMD (an empty "
        "list would override CMD with nothing). Batch always sets a command.",
    )
    volumes: dict[str, str] | None = Field(
        default=None, description="Host-to-container volume mounts"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables for the container"
    )
    workdir: str | None = Field(default=None, description="Working directory inside the container")
    memory_limit: str | None = Field(
        default=None,
        description="Container memory limit (Docker format, e.g., '16g')",
    )
    cpu_limit: float | None = Field(
        default=None,
        description="Container CPU core limit (e.g., 2.0 = two cores)",
    )
    # --- Sandbox hardening (declared in P1/S1, wired in S5) ---
    cap_drop: list[str] | None = Field(
        default=None,
        description="Linux capabilities to drop (e.g., ['ALL']); None leaves runtime defaults",
    )
    no_new_privileges: bool = Field(
        default=False,
        description="Set the no-new-privileges security option on the container",
    )
    network_mode: str | None = Field(
        default=None,
        description="Docker network mode ('bridge', 'none', ...); None leaves runtime default",
    )
    read_only: bool = Field(
        default=False, description="Mount the container root filesystem read-only"
    )
    ports: dict[int, int] | None = Field(
        default=None,
        description="Container-port-to-host-port mappings (service mode); "
        "{container_port: host_port}",
    )
    extra_port_bind_ips: list[str] | None = Field(
        default=None,
        description="Additional host IPs to bind published ports on, alongside "
        "127.0.0.1 (P5 model workloads bind the docker bridge gateway so app "
        "containers can reach the endpoint); None keeps loopback-only binding",
    )
    shm_size: str | None = Field(
        default=None,
        description="Container /dev/shm size (Docker format, e.g. '1g'); None "
        "leaves Docker's 64MB default. P11 vLLM model workloads set it because "
        "worker IPC / NCCL need more shared memory than the default.",
    )
    user: str | None = Field(
        default=None,
        description="Run the container as this user (Docker --user, e.g. "
        "'postgres'); None leaves the image's default (root for most). P15 "
        "database workloads set it so the official entrypoints take their "
        "non-root path under the untouched cap_drop=ALL / no-new-privileges "
        "sandbox (D-P15-6).",
    )
    log_config: dict[str, str] | None = Field(
        default=None,
        description="Docker json-file log-driver options (P14b service/model log "
        "caps), e.g. {'max-size': '10m', 'max-file': '3'}; None leaves Docker's "
        "default log config. Batch launches never set it.",
    )
    extra_labels: dict[str, str] | None = Field(
        default=None,
        description="Additional Docker labels (P20 run/release attribution: "
        "'nerdit-run'/'nerdit-job', stamped by _execute_container_once). Merged "
        "UNDER the platform labels in DockerRuntime — 'managed-by' and "
        "'nerdit-instance' always win, so a caller can never spoof ownership. "
        "None (every non-run caller) keeps the label dict byte-identical.",
    )
