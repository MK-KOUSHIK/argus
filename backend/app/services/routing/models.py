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
    def weight_size_mb(self) -> float:
        """Exact model weight file size on disk in MiB."""
        return round(self.size_bytes / (1024 * 1024), 1) if self.size_bytes else 0.0

    @property
    def measured_vram_mb(self) -> Optional[int]:
        """Exact resident VRAM footprint in MiB reported by Ollama /api/ps (when loaded)."""
        if self.is_loaded and self.size_vram and self.size_vram > 0:
            return round(self.size_vram / (1024 * 1024))
        return None

    def estimate_runtime_vram_mb(self, context_tier: Optional[int] = None) -> int:
        """Estimate total runtime VRAM requirements (weights + KV cache + working buffers).

        If model is resident, returns measured resident VRAM from /api/ps.
        If unloaded, computes a conservative runtime requirement proxy:
        - Base weight footprint from disk size
        - KV cache buffer proxy based on snapped context tier (default 4096)
        - Compute/scratch buffer headroom proxy (~100 MiB)

        NOTE: This is an estimated runtime requirement proxy, not an exact measurement.
        """
        if self.is_loaded and self.measured_vram_mb is not None:
            return self.measured_vram_mb

        if not self.size_bytes or self.size_bytes <= 0:
            return 0

        weight_mb = round(self.size_bytes / (1024 * 1024))
        ctx = context_tier if (context_tier and context_tier > 0) else 4096

        # KV cache buffer proxy scaling with snapped context tier
        if self.tier == ModelTier.SMALL:
            kv_buffer = max(64, round((ctx / 4096) * 100))
        elif self.tier == ModelTier.MEDIUM:
            kv_buffer = max(256, round((ctx / 4096) * 350))
        else:
            kv_buffer = max(512, round((ctx / 4096) * 700))

        return weight_mb + kv_buffer

    @property
    def estimated_vram_mb(self) -> int:
        """Backwards-compatible alias for estimated total runtime VRAM proxy."""
        return self.estimate_runtime_vram_mb()


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
    measured_vram_mb: Optional[int] = None
    weight_footprint_mb: Optional[int] = None
    estimated_runtime_vram_mb: Optional[int] = None
    vram_measurement_type: str = "none"


@dataclass
class RoutingDecision:
    """Explainable model routing decision with all supporting telemetry."""

    selected_model: str
    target_model_tier: ModelTier
    routing_status: RoutingStatus
    policy_applied: str
    reason: str
    decision_confidence: float  # Uncalibrated heuristic margin [0.0, 1.0] retained for compatibility
    complexity_assessment: ComplexityAssessment
    resource_assessment: ResourceAssessment
    routing_duration_ms: float
    fallback_model: Optional[str] = None
    is_fallback: bool = False
    warning: Optional[str] = None
    heuristic_margin: float = 0.0  # Heuristic distance from threshold (|score - threshold|)


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
