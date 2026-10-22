"""API route handlers for prompt optimization and Ollama status."""

from fastapi import APIRouter, HTTPException, status
from app.schemas.optimizer import (
    OptimizeRequest,
    OptimizeResponse,
    OllamaStatusResponse,
)
from app.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)

router = APIRouter(tags=["Prompt Optimization"])


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize Prompt",
    description="Rewrites a user prompt using the local Ollama LLM to reduce input tokens while preserving intent and constraints.",
)
async def optimize_prompt(request: OptimizeRequest) -> OptimizeResponse:
    """Optimize prompt with local Ollama model."""
    try:
        return await OllamaService.optimize_prompt(request)
    except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError, OllamaServiceError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/ollama/status",
    response_model=OllamaStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Ollama Service Status",
    description="Inspects the local Ollama service connection and verifies model availability.",
)
async def check_ollama_status() -> OllamaStatusResponse:
    """Verify local Ollama service health and model installation."""
    return await OllamaService.check_status()
