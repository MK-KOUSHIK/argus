"""FastAPI route handlers for Main LLM Integration & Benchmark Suite (Phase 4)."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from app.schemas.main_llm import (
    BenchmarkRunRequest,
    CompareRequest,
    MainLLMStatusResponse,
    PricingTableResponse,
)
from app.services.comparative_execution_service import (
    ComparativeExecutionService,
    ComparisonResult,
)
from app.services.main_llm_service import (
    MainLLMError,
    MainLLMService,
    MissingCredentialsError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderAPIError,
)
from app.benchmarks.runner import BenchmarkReport, BenchmarkRunner
from app.services.pricing_service import MODEL_PRICING_REGISTRY

router = APIRouter(prefix="/main-llm", tags=["Main LLM & Benchmark"])


@router.get(
    "/status",
    response_model=MainLLMStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Main LLM Provider Status",
    description="Check external frontier LLM provider configuration without leaking credentials.",
)
async def get_main_llm_status() -> MainLLMStatusResponse:
    """Return external LLM provider connectivity and configuration status."""
    prov_status = MainLLMService.check_status()
    return MainLLMStatusResponse(
        provider=prov_status.provider,
        model=prov_status.model,
        is_configured=prov_status.is_configured,
        base_url=prov_status.base_url,
        message=prov_status.message,
    )


@router.post(
    "/compare",
    response_model=ComparisonResult,
    status_code=status.HTTP_200_OK,
    summary="Compare Baseline vs. Optimized Execution",
    description="Executes both Baseline (original prompt) and Optimized (Argus rewrite) paths on the main LLM and returns full comparative token, cost, and latency metrics.",
)
async def compare_execution(request: CompareRequest) -> ComparisonResult:
    """Execute both paths and compute side-by-side telemetry."""
    try:
        return await ComparativeExecutionService.compare(
            prompt=request.prompt,
            mode=request.mode,
            model=request.target_model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
        ) from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
        ) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.message,
        ) from exc
    except (ProviderAPIError, MainLLMError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/benchmark/dataset",
    response_model=List[Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Benchmark Dataset",
    description="Returns the 5 version-controlled benchmark dataset cases across diverse task categories.",
)
async def get_benchmark_dataset() -> List[Dict[str, Any]]:
    """Retrieve benchmark test cases."""
    return BenchmarkRunner.load_dataset()


@router.post(
    "/benchmark/run",
    response_model=BenchmarkReport,
    status_code=status.HTTP_200_OK,
    summary="Run Reproducible Benchmark Suite",
    description="Runs the reproducible benchmark suite across all 5 test cases and returns comprehensive telemetry. Defaults to mock provider unless live=True with credentials configured.",
)
async def run_benchmark(request: BenchmarkRunRequest) -> BenchmarkReport:
    """Execute benchmark suite and return aggregate report."""
    try:
        return await BenchmarkRunner.run(
            live=request.live,
            model=request.model,
        )
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except MainLLMError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/pricing",
    response_model=PricingTableResponse,
    status_code=status.HTTP_200_OK,
    summary="Frontier Model Pricing Table",
    description="Returns the published pricing registry used for estimating and calculating token costs.",
)
async def get_pricing_table() -> PricingTableResponse:
    """Return model pricing registry."""
    return PricingTableResponse(pricing=MODEL_PRICING_REGISTRY)
