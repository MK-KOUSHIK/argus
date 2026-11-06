"""API route handlers for agent execution trajectory reduction (Phase 6)."""

from fastapi import APIRouter, HTTPException, status
from app.schemas.trajectory import (
    ReduceTrajectoryRequest,
    ReduceTrajectoryResponse,
)
from app.services.trajectory_reduction_service import TrajectoryReductionService

router = APIRouter(prefix="/trajectory", tags=["Agent History Reduction"])


@router.post(
    "/reduce",
    response_model=ReduceTrajectoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Reduce Agent Trajectory",
    description=(
        "Reduces unnecessary agent conversation history and verbose execution trajectory context "
        "sent to local models while preserving system instructions, active goals, explicit constraints, "
        "and tool-call dependencies with safe fallback."
    ),
)
async def reduce_trajectory(request: ReduceTrajectoryRequest) -> ReduceTrajectoryResponse:
    """Reduce multi-turn agent execution trajectory with explainable audit decisions."""
    try:
        return TrajectoryReductionService.reduce_trajectory(request)
    except Exception as exc:
        # Catastrophic fallback if unexpected exception escapes service layer
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process trajectory reduction: {str(exc)}",
        ) from exc
