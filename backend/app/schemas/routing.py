"""Pydantic schemas for Model Routing and model catalog (Phase 5)."""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Categorization of model capacity based on parameter scale."""

    SMALL = "small"    # e.g., <= 3B parameters (e.g. qwen2.5:1.5b)
    MEDIUM = "medium"  # e.g., 4B - 14B parameters (e.g. qwen2.5:7b)
    LARGE = "large"    # e.g., > 14B parameters
    UNKNOWN = "unknown"


class RoutingStatus(str, Enum):
    """Status outcome of the routing decision."""

    ROUTED = "routed"
    MANUAL_OVERRIDE = "manual_override"
    FALLBACK = "fallback"
    DISABLED = "disabled"


class ModelCatalogItem(BaseModel):
    """Catalog entry describing an installed Ollama model."""

    name: str = Field(description="Model tag identifier in Ollama")
    parameter_size: Optional[str] = Field(default=None, description="Reported parameter size (e.g. 1.5B, 7.6B)")
    parameter_count: Optional[int] = Field(default=None, description="Exact parameter count if available")
    quantization_level: Optional[str] = Field(default=None, description="Quantization format (e.g. Q4_K_M)")
    family: Optional[str] = Field(default=None, description="Model architecture family")
    context_length: Optional[int] = Field(default=None, description="Maximum context length supported")
    size_bytes: int = Field(description="Model file size on disk in bytes")
    size_mb: float = Field(description="Model file size in megabytes")
    tier: str = Field(description="Assigned routing tier (small, medium, large, unknown)")
    is_loaded: bool = Field(description="Whether the model is currently resident in memory")
    size_vram_mb: Optional[float] = Field(default=None, description="Active VRAM usage in MiB if loaded")
    estimated_runtime_vram_mb: Optional[float] = Field(default=None, description="Estimated runtime VRAM requirement proxy in MiB")
    modified_at: str = Field(description="Last modified timestamp")


class ModelCatalogResponse(BaseModel):
    """Response payload for installed models catalog endpoint."""

    connected: bool = Field(description="Whether Ollama daemon responded")
    base_url: str = Field(description="Ollama daemon HTTP base URL")
    total_models: int = Field(description="Count of discovered models")
    models: List[ModelCatalogItem] = Field(default_factory=list, description="Discovered models catalog")
    active_loaded_models: List[str] = Field(default_factory=list, description="Models currently resident in memory")
    error: Optional[str] = Field(default=None, description="Diagnostic error if check failed")


class RouteRequest(BaseModel):
    """Request payload to test or preview prompt routing without executing inference."""

    prompt: str = Field(..., min_length=1, description="Prompt text to evaluate")
    mode: Literal["balanced", "concise", "structure"] = Field(
        default="balanced", description="Optimization profile"
    )
    requested_model: Optional[str] = Field(
        default=None, description="Optional manual model override"
    )
    requested_output_budget: Optional[int] = Field(
        default=None, description="Optional output budget override"
    )
    allow_override_fallback: bool = Field(
        default=False,
        description="Whether to permit fallback to a safe installed model if manual override requested model cannot be safely admitted",
    )


class ComplexityTelemetry(BaseModel):
    """Telemetry detailing prompt complexity signals."""

    heuristic_score: float = Field(description="Heuristic complexity score [0.0, 1.0]")
    task_category: str = Field(description="Detected domain category")
    task_confidence: Optional[float] = Field(default=None, description="Category detection confidence")
    category_weight: float = Field(description="Base difficulty weight assigned to task category")
    explicit_constraints_count: int = Field(description="Count of explicit prompt constraints")
    required_operations_count: int = Field(description="Count of extracted functional operations")
    input_tokens: int = Field(description="Estimated input token count")
    context_tier: int = Field(description="Snapped context tier")
    output_budget: int = Field(description="Allocated output token budget")
    has_prohibitions: bool = Field(description="Whether negative constraints are present")
    has_code_or_data: bool = Field(description="Whether prompt contains structured code or tabular data")
    complexity_factors: List[str] = Field(default_factory=list, description="Signals contributing to complexity")


class ResourceTelemetry(BaseModel):
    """Telemetry detailing live hardware state evaluated during routing."""

    gpu_available: bool = Field(description="Whether a compatible NVIDIA GPU was detected")
    vram_total_mb: Optional[int] = Field(default=None, description="Total GPU VRAM in MiB")
    vram_free_mb: Optional[int] = Field(default=None, description="Free GPU VRAM in MiB")
    vram_used_mb: Optional[int] = Field(default=None, description="Used GPU VRAM in MiB")
    ram_total_mb: Optional[float] = Field(default=None, description="Total system RAM in MiB")
    ram_free_mb: Optional[float] = Field(default=None, description="Free system RAM in MiB")
    ram_used_mb: Optional[float] = Field(default=None, description="Used system RAM in MiB")
    estimated_model_vram_mb: Optional[int] = Field(
        default=None, description="Estimated VRAM required by candidate model"
    )
    measured_vram_mb: Optional[int] = Field(
        default=None, description="Exact resident VRAM measured from Ollama /api/ps when loaded"
    )
    weight_footprint_mb: Optional[int] = Field(
        default=None, description="Model weight file size on disk in MiB (proxy baseline)"
    )
    estimated_runtime_vram_mb: Optional[int] = Field(
        default=None, description="Estimated total runtime VRAM proxy (weights + KV cache buffer) when unloaded"
    )
    vram_measurement_type: str = Field(
        default="none",
        description="Source of memory figure: 'measured_resident', 'architecture_proxy', 'rough_proxy', 'cpu_only', or 'unverified'",
    )
    is_resource_sufficient: bool = Field(description="Whether available resources satisfy candidate requirements")
    resource_pressure_reason: Optional[str] = Field(
        default=None, description="Diagnostic explanation if resource pressure was detected"
    )
    is_currently_loaded: bool = Field(description="Whether candidate model is already loaded in memory")
    concurrency_in_flight: int = Field(description="Current in-flight inference requests")
    concurrency_capacity: int = Field(description="Current allowable concurrent inference capacity")


class RoutingDecisionTelemetry(BaseModel):
    """Full routing decision metadata embedded into OptimizeResponse or returned by /api/route."""

    selected_model: str = Field(description="Local Ollama model selected for prompt optimization")
    target_model_tier: str = Field(description="Model tier (small, medium, large, unknown)")
    routing_status: str = Field(description="Decision status (routed, manual_override, fallback, disabled)")
    policy_applied: str = Field(description="Active routing policy (auto, prefer_small, prefer_large, manual)")
    reason: str = Field(description="Explainable rationale for the routing decision")
    decision_confidence: float = Field(
        description="Uncalibrated heuristic decision margin retained for backward compatibility; not a probability"
    )
    heuristic_margin: float = Field(
        default=0.0,
        description="Uncalibrated heuristic distance from decision threshold (|score - threshold|); not a probability",
    )
    complexity: ComplexityTelemetry = Field(description="Prompt complexity signals")
    resources: ResourceTelemetry = Field(description="Resource state snapshot")
    routing_duration_ms: float = Field(description="Time taken to compute the routing decision in milliseconds")
    fallback_model: Optional[str] = Field(default=None, description="Fallback model if primary failed or constrained")
    is_fallback: bool = Field(default=False, description="Whether fallback was invoked")
    warning: Optional[str] = Field(default=None, description="Optional diagnostic warning")

