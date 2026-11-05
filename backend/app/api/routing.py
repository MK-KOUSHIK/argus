"""API route handlers for local Model Routing and model catalog (Phase 5)."""

from fastapi import APIRouter, HTTPException, status

from app.services.routing.models import (
    ComplexityTelemetry,
    ModelCatalogResponse,
    ResourceTelemetry,
    RouteRequest,
    RoutingDecisionTelemetry,
)
from app.services.routing.registry import LocalModelRegistry
from app.services.routing.router import ModelNotInstalledError, ModelRouter, ModelRouterError

router = APIRouter(tags=["Model Routing"])


@router.get(
    "/models",
    response_model=ModelCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="List Installed Models",
    description="Returns the dynamically discovered local Ollama models, parameter scales, VRAM residency, and metadata.",
)
async def list_installed_models() -> ModelCatalogResponse:
    """Query local Ollama runtime and return catalog of installed models."""
    return await LocalModelRegistry.get_catalog(force_refresh=True)


@router.post(
    "/route",
    response_model=RoutingDecisionTelemetry,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Prompt Routing",
    description="Inspects prompt complexity, task category, and live GPU/RAM resources to determine optimal local model without executing inference.",
)
async def route_prompt(request: RouteRequest) -> RoutingDecisionTelemetry:
    """Compute and explain model routing decision for a given prompt."""
    try:
        decision = await ModelRouter.route(
            prompt=request.prompt,
            mode=request.mode,
            requested_model=request.requested_model,
            requested_output_budget=request.requested_output_budget,
            allow_override_fallback=request.allow_override_fallback,
        )
    except ModelNotInstalledError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except ModelRouterError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing evaluation failed: {str(exc)}",
        ) from exc

    return RoutingDecisionTelemetry(
        selected_model=decision.selected_model,
        target_model_tier=decision.target_model_tier.value,
        routing_status=decision.routing_status.value,
        policy_applied=decision.policy_applied,
        reason=decision.reason,
        decision_confidence=decision.decision_confidence,
        heuristic_margin=decision.heuristic_margin,
        complexity=ComplexityTelemetry(
            heuristic_score=decision.complexity_assessment.heuristic_score,
            task_category=decision.complexity_assessment.task_category,
            task_confidence=decision.complexity_assessment.task_confidence,
            category_weight=decision.complexity_assessment.category_weight,
            explicit_constraints_count=decision.complexity_assessment.explicit_constraints_count,
            required_operations_count=decision.complexity_assessment.required_operations_count,
            input_tokens=decision.complexity_assessment.input_tokens,
            context_tier=decision.complexity_assessment.context_tier,
            output_budget=decision.complexity_assessment.output_budget,
            has_prohibitions=decision.complexity_assessment.has_prohibitions,
            has_code_or_data=decision.complexity_assessment.has_code_or_data,
            complexity_factors=decision.complexity_assessment.complexity_factors,
        ),
        resources=ResourceTelemetry(
            gpu_available=decision.resource_assessment.gpu_available,
            vram_total_mb=decision.resource_assessment.vram_total_mb,
            vram_free_mb=decision.resource_assessment.vram_free_mb,
            vram_used_mb=decision.resource_assessment.vram_used_mb,
            ram_total_mb=decision.resource_assessment.ram_total_mb,
            ram_free_mb=decision.resource_assessment.ram_free_mb,
            ram_used_mb=decision.resource_assessment.ram_used_mb,
            estimated_model_vram_mb=decision.resource_assessment.estimated_model_vram_mb,
            is_resource_sufficient=decision.resource_assessment.is_resource_sufficient,
            resource_pressure_reason=decision.resource_assessment.resource_pressure_reason,
            is_currently_loaded=decision.resource_assessment.is_currently_loaded,
            concurrency_in_flight=decision.resource_assessment.concurrency_in_flight,
            concurrency_capacity=decision.resource_assessment.concurrency_capacity,
        ),
        routing_duration_ms=decision.routing_duration_ms,
        fallback_model=decision.fallback_model,
        is_fallback=decision.is_fallback,
        warning=decision.warning,
    )

