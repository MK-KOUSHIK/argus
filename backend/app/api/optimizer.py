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
from app.services.adaptive_context_manager import ContextOverflowError
from app.services.compression import CompressionRegistry

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
    except ContextOverflowError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
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


@router.get(
    "/compression/registry/{ref_key}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve Original Content",
    description="Retrieves uncompressed original content from the registry using its unique reference key.",
)
async def get_registered_content(ref_key: str):
    """Retrieve original content by reference key."""
    content = CompressionRegistry.retrieve(ref_key)
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original content for reference '{ref_key}' not found or expired.",
        )
    return {"ref_key": ref_key, "content": content}
