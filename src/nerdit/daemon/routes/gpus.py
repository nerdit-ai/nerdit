"""GPU status endpoints."""

from fastapi import APIRouter, Request

from nerdit.db.models import GpuResponse

router = APIRouter()


@router.get("/gpus", response_model=list[GpuResponse], operation_id="list_gpus")
async def list_gpus(request: Request) -> list[GpuResponse]:
    """Return all registered GPUs with their live metrics."""
    queries = request.app.state.queries
    monitor = request.app.state.monitor
    gpus = await queries.list_gpus()
    metrics = monitor.get_metrics()

    result = []
    for gpu in gpus:
        m = metrics.get(gpu.id)
        result.append(
            GpuResponse(
                id=gpu.id,
                name=gpu.name,
                memory_mb=gpu.memory_mb,
                compute_cap=gpu.compute_cap,
                vendor=gpu.vendor,
                schedulable=gpu.schedulable,
                status=gpu.status,
                utilization_percent=m.utilization_percent if m else None,
                memory_used_mb=m.memory_used_mb if m else None,
                temperature_c=m.temperature_c if m else None,
            )
        )
    return result
