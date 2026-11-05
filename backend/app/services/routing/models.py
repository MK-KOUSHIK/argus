"""Domain models and data structures for Model Routing (Phase 5).

Provides typed representations for:
- Discovered local models and their runtime state
- Routing decisions, tiers, complexity assessments, and resource assessments
- Heuristic complexity scoring metadata (strictly non-probabilistic)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.schemas.routing import (
    ComplexityTelemetry,
    ModelCatalogItem,
    ModelCatalogResponse,
    ModelTier,
    ResourceTelemetry,
    RouteRequest,
    RoutingDecisionTelemetry,
    RoutingStatus,
)


@dataclass
class DiscoveredModel:
    """Installed model discovered from local Ollama runtime."""

    name: str
    model: str
    digest: str
    size_bytes: int
    modified_at: str
    parameter_size: Optional[str] = None
    parameter_count: Optional[int] = None
    quantization_level: Optional[str] = None
    family: Optional[str] = None
    families: List[str] = field(default_factory=list)
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    capabilities: List[str] = field(default_factory=list)
    is_loaded: bool = False
    size_vram: Optional[int] = None
    expires_at: Optional[str] = None
    tier: ModelTier = ModelTier.UNKNOWN

    @property
    def estimated_vram_mb(self) -> int:
        """Estimate minimum VRAM required in MiB based on actual size or reported VRAM.

        If model is loaded and reports size_vram from /api/ps, use that exact value.
        Otherwise, base on disk size (weights in Q4_K_M) plus nominal KV cache margin.
        """
        if self.is_loaded and self.size_vram and self.size_vram > 0:
            return round(self.size_vram / (1024 * 1024))
        base_mb = round(self.size_bytes / (1024 * 1024))
        return base_mb + 300


@dataclass
class ComplexityAssessment:
    """Heuristic assessment of prompt difficulty and reasoning requirements.

    IMPORTANT: heuristic_score is a deterministic rule-based score [0.0, 1.0],
    NOT a trained machine learning probability.
    """

    heuristic_score: float
    task_category: str
    task_confidence: Optional[float]
    category_weight: float
    explicit_constraints_count: int
    required_operations_count: int
    input_tokens: int
    context_tier: int
    output_budget: int
    has_prohibitions: bool
    has_code_or_data: bool
    complexity_factors: List[str] = field(default_factory=list)


@dataclass
class ResourceAssessment:
    """Live assessment of host GPU VRAM and system RAM for model execution."""

    gpu_available: bool
    vram_total_mb: Optional[int]
    vram_free_mb: Optional[int]
    vram_used_mb: Optional[int]
    ram_total_mb: Optional[float]
    ram_free_mb: Optional[float]
    ram_used_mb: Optional[float]
    estimated_model_vram_mb: Optional[int]
    is_resource_sufficient: bool
    resource_pressure_reason: Optional[str] = None
    is_currently_loaded: bool = False
    concurrency_in_flight: int = 0
    concurrency_capacity: int = 1


@dataclass
class RoutingDecision:
    """Explainable model routing decision with all supporting telemetry."""

    selected_model: str
    target_model_tier: ModelTier
    routing_status: RoutingStatus
    policy_applied: str
    reason: str
    decision_confidence: float  # Heuristic confidence [0.0, 1.0]
    complexity_assessment: ComplexityAssessment
    resource_assessment: ResourceAssessment
    routing_duration_ms: float
    fallback_model: Optional[str] = None
    is_fallback: bool = False
    warning: Optional[str] = None


__all__ = [
    "ComplexityAssessment",
    "ComplexityTelemetry",
    "DiscoveredModel",
    "ModelCatalogItem",
    "ModelCatalogResponse",
    "ModelTier",
    "ResourceAssessment",
    "ResourceTelemetry",
    "RouteRequest",
    "RoutingDecision",
    "RoutingDecisionTelemetry",
    "RoutingStatus",
]
