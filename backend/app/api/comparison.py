"""API route handlers for end-to-end original vs optimized LLM output comparison."""

from fastapi import APIRouter, HTTPException, status
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.services.comparison_service import (
    ComparisonService,
    ComparisonServiceError,
    DownstreamModelNotFoundError,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
    OllamaServiceError,
)
from app.services.adaptive_context_manager import ContextOverflowError
from app.services.routing.router import ModelRouterError

router = APIRouter(tags=["Output Comparison"])


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare Original vs Optimized Prompt Outputs",
    description="Executes two genuine downstream LLM inferences (Execution A on original prompt, Execution B on optimized prompt) pinned to the same model, capturing native telemetry and honest delta metrics.",
)
async def compare_outputs(request: ComparisonRequest) -> ComparisonResponse:
    """Compare downstream LLM outputs for original and optimized prompts."""
    try:
        return await ComparisonService.compare(request)
    except DownstreamModelNotFoundError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except ModelRouterError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
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
    except ComparisonServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
