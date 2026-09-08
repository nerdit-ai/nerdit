"""Response schema for the `/health` surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """API response model for the `/health` endpoint."""

    status: str = Field(default="ok", description="Daemon status ('ok' when healthy)")
    version: str = Field(description="Nerdit version string")
    gpu_count: int = Field(description="Number of GPUs detected on the host")
