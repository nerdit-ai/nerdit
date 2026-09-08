"""Health check endpoint."""

from fastapi import APIRouter, Request

import nerdit
from nerdit.db.models import GpuStatus, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, operation_id="get_health")
async def health(request: Request) -> HealthResponse:
    """Return daemon health status including version and GPU count."""
    queries = request.app.state.queries
    gpus = await queries.list_gpus()
    return HealthResponse(
        status="ok",
        version=nerdit.__version__,
        gpu_count=sum(1 for gpu in gpus if gpu.status != GpuStatus.offline),
    )
