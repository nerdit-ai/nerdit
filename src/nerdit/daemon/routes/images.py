"""Local container image listing for the dashboard."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/images", operation_id="list_images")
async def list_images(request: Request) -> list[str]:
    """Locally available image tags, for the New Job image dropdown.

    Both job-submission endpoints reject custom images that are not present
    locally, so this list is exactly the set of valid choices.
    """
    runtime = request.app.state.runtime
    return await runtime.list_images()
