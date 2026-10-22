"""Health endpoint route handler."""

from fastapi import APIRouter, status
from app.schemas.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Returns the operational status, version, and environment of the Argus service.",
)
async def check_health() -> HealthResponse:
    """Execute health status inspection."""
    return HealthService.get_health()
