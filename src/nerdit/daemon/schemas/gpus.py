"""Response schema for the `/gpus` surface."""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.db.enums import GpuStatus, GpuVendor


class GpuResponse(BaseModel):
    """API response model for a GPU, including live metrics."""

    id: str = Field(description="Stable Nerdit inventory ID")
    name: str = Field(description="GPU product name")
    memory_mb: int = Field(description="Total GPU memory in megabytes")
    compute_cap: str | None = Field(default=None, description="CUDA compute capability")
    vendor: GpuVendor = Field(description="GPU hardware vendor")
    schedulable: bool = Field(description="Whether the scheduler may allocate this GPU")
    status: GpuStatus = Field(description="Current allocation status")
    utilization_percent: int | None = Field(
        default=None, description="GPU compute utilization (0-100%)"
    )
    memory_used_mb: int | None = Field(
        default=None, description="Currently used GPU memory in megabytes"
    )
    temperature_c: int | None = Field(default=None, description="GPU temperature in Celsius")
