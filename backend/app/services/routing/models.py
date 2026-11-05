"""Domain models and data structures for Model Routing (Phase 5).

Provides typed representations for:
- Discovered local models and their runtime state
- Routing decisions, tiers, complexity assessments, and resource assessments
- Heuristic complexity scoring metadata (strictly non-probabilistic)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

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

    @property
    def effective_head_dim(self) -> Optional[int]:
        """Verified attention head dimension.

        Uses explicit head_dim if verified; derives from embedding_length // num_heads
        only if embedding_length is an exact integer multiple of num_heads.
        """
        if self.head_dim is not None and self.head_dim > 0:
            return self.head_dim
        if (
            self.embedding_length is not None
            and self.embedding_length > 0
            and self.num_heads is not None
            and self.num_heads > 0
            and self.embedding_length % self.num_heads == 0
        ):
            return self.embedding_length // self.num_heads
        return None

    @property
    def effective_kv_dim(self) -> Optional[int]:
        """Verified key-value dimension per layer.

        DO NOT substitute embedding_length directly when architecture is grouped-query
        or multi-query attention, or when num_kv_heads is unknown.

        Returns:
            int if reliably established from verified architecture metadata;
            None if ambiguous, unverified, or missing.
        """
        head_d = self.effective_head_dim

        # 1. Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)
        # Requires verified num_kv_heads and head_dim
        if self.num_kv_heads is not None and self.num_kv_heads > 0 and head_d is not None:
            return self.num_kv_heads * head_d

        # 2. Standard Multi-Head Attention (MHA)
        # Valid only when num_kv_heads is explicitly verified to equal num_heads
        if (
            self.num_heads is not None
            and self.num_heads > 0
            and self.num_kv_heads is not None
            and self.num_kv_heads == self.num_heads
        ):
            if head_d is not None:
                return self.num_heads * head_d
            if self.embedding_length is not None and self.embedding_length > 0:
                return self.embedding_length

        # In all other cases (e.g. missing num_kv_heads or missing head dimension):
        # Never assume MHA or substitute embedding_length directly!
        return None

    @property
    def has_architecture_metadata(self) -> bool:
        """Whether model has sufficient verified architectural metadata to calculate exact KV-cache dimensions."""
        return (
            self.num_layers is not None
            and self.num_layers > 0
            and self.effective_kv_dim is not None
        )

    def estimate_kv_cache_mb(
        self,
        context_tier: Optional[int] = None,
        kv_dtype: Optional[str] = None,
    ) -> Tuple[int, str]:
        """Estimate KV cache memory requirement in MiB and return (kv_mb, method).

        Returns:
            Tuple[int, str]: (estimated_kv_cache_mb, method)
            where method is 'architecture_derived' or 'rough_proxy'.
        """
        from app.core.config import settings

        ctx = context_tier if (context_tier and context_tier > 0) else getattr(settings, "OLLAMA_NUM_CTX", 4096)
        dtype = (kv_dtype or getattr(settings, "OLLAMA_KV_CACHE_TYPE", "q8_0")).lower()

        if "q4" in dtype:
            bytes_per_elem = 0.5625
        elif "q8" in dtype:
            bytes_per_elem = 1.0625
        elif "f32" in dtype or "fp32" in dtype:
            bytes_per_elem = 4.0
        elif "f16" in dtype or "fp16" in dtype:
            bytes_per_elem = 2.0
        else:
            bytes_per_elem = 1.0

        if self.has_architecture_metadata and self.num_layers is not None and self.effective_kv_dim is not None:
            # 2 (Key + Value) * layers * kv_dim * context_tokens * bytes_per_elem
            total_kv_bytes = 2 * self.num_layers * self.effective_kv_dim * ctx * bytes_per_elem
            kv_mb = max(16, round(total_kv_bytes / (1024 * 1024)))
            return kv_mb, "architecture_derived"

        # Clearly identified rough proxy scaled by context tier and model tier when architecture metadata is missing
        if self.tier == ModelTier.SMALL:
            kv_mb = max(64, round((ctx / 4096) * 100))
        elif self.tier == ModelTier.MEDIUM:
            kv_mb = max(256, round((ctx / 4096) * 350))
        else:
            kv_mb = max(512, round((ctx / 4096) * 700))

        return kv_mb, "rough_proxy"

    def estimate_runtime_vram_mb(self, context_tier: Optional[int] = None) -> int:
        """Estimate total runtime VRAM requirements (weights + KV cache buffer proxy).

        If model is resident, returns measured resident VRAM from /api/ps.
        If unloaded, computes runtime requirement proxy:
        - Base weight footprint from disk size
        - KV cache buffer derived from architecture metadata (if available) or rough proxy
        """
        if self.is_loaded and self.measured_vram_mb is not None:
            return self.measured_vram_mb

        if not self.size_bytes or self.size_bytes <= 0:
            return 0

        weight_mb = round(self.size_bytes / (1024 * 1024))
        kv_mb, _ = self.estimate_kv_cache_mb(context_tier=context_tier)
        return weight_mb + kv_mb

    def get_vram_measurement_type(self, context_tier: Optional[int] = None) -> str:
        """Determine telemetry fidelity: 'measured_resident', 'architecture_proxy', or 'rough_proxy'."""
        if self.is_loaded and self.measured_vram_mb is not None:
            return "measured_resident"
        if self.has_architecture_metadata:
            return "architecture_proxy"
        return "rough_proxy"

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
