"""API route handlers for agent execution trajectory reduction (Phase 6)."""

import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.trajectory import (
    ReduceTrajectoryRequest,
    ReduceTrajectoryResponse,
)
from app.services.trajectory_reduction_service import TrajectoryReductionService

logger = logging.getLogger(__name__)

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
        logger.exception("Failed to process trajectory reduction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process trajectory reduction.",
        ) from exc
