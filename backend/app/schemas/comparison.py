"""Pydantic schemas for end-to-end original vs optimized LLM output comparison."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.optimizer import OptimizationMode, OptimizeResponse


class LatencyBreakdown(BaseModel):
    """Granular duration breakdown separating model load, inference, and queue wait."""

    model_load_duration_ms: Optional[float] = Field(
        default=None,
        description="Time spent loading model weights into VRAM/RAM (load_duration / 1e6)",
    )
    prompt_eval_duration_ms: Optional[float] = Field(
        default=None,
        description="Time spent evaluating input prompt tokens (prompt_eval_duration / 1e6). Not client TTFT.",
    )
    eval_duration_ms: Optional[float] = Field(
        default=None,
        description="Time spent generating output tokens (eval_duration / 1e6)",
    )
    inference_duration_ms: Optional[float] = Field(
        default=None,
        description="Pure inference duration: prompt_eval_duration_ms + eval_duration_ms (excludes load and queue wait)",
    )
    queue_wait_duration_ms: Optional[float] = Field(
        default=None,
        description="Time spent waiting for concurrency controller inference slot",
    )
    wall_time_ms: Optional[float] = Field(
        default=None,
        description="Total roundtrip wall-clock duration in milliseconds",
    )


class EffectiveGenerationSettings(BaseModel):
    """Effective generation parameters applied to downstream LLM execution."""

    model: str = Field(description="Pinned downstream model name")
    num_ctx: int = Field(description="Effective context window size passed to Ollama")
    num_predict: int = Field(description="Maximum token generation ceiling")
    temperature: float = Field(description="Sampling temperature")
    top_p: float = Field(description="Nucleus sampling probability")
    system_instruction: Optional[str] = Field(default=None, description="System instruction if provided")
    context_window_shared: bool = Field(default=True, description="True if both executions used identical num_ctx")


class TrialTelemetry(BaseModel):
    """Telemetry captured for a single trial execution."""

    trial_index: int = Field(description="Zero-based trial index")
    execution_order: int = Field(description="Execution order within the paired trial (1=first, 2=second)")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt token count (prompt_eval_count)")
    output_tokens: Optional[int] = Field(default=None, description="Generated output token count (eval_count)")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens (prompt + output)")
    latency: LatencyBreakdown = Field(description="Granular latency breakdown for this trial")
    done_reason: Optional[str] = Field(default=None, description="Ollama completion reason")
    response: str = Field(description="Generated output for this trial")


class AggregateLatency(BaseModel):
    """Aggregated latency statistics across multiple trials."""

    trials_count: int = Field(description="Total number of trials executed")
    mean_inference_duration_ms: Optional[float] = Field(default=None, description="Mean pure inference duration in ms")
    median_inference_duration_ms: Optional[float] = Field(default=None, description="Median pure inference duration in ms")
    mean_wall_time_ms: Optional[float] = Field(default=None, description="Mean wall-clock duration in ms")
    mean_prompt_eval_duration_ms: Optional[float] = Field(default=None, description="Mean prompt evaluation duration in ms")
    mean_eval_duration_ms: Optional[float] = Field(default=None, description="Mean token generation duration in ms")
    mean_queue_wait_duration_ms: Optional[float] = Field(default=None, description="Mean queue wait duration in ms")
    mean_model_load_duration_ms: Optional[float] = Field(default=None, description="Mean model load duration in ms")


class ComparisonExecutionTelemetry(BaseModel):
    """Telemetry and generated content for an individual downstream LLM execution."""

    prompt: str = Field(description="The prompt text dispatched to the downstream LLM")
    response: str = Field(description="Raw generated response from the downstream LLM")
    model: str = Field(description="Model name used for inference")
    provider: str = Field(default="ollama", description="LLM provider name")
    prompt_tokens: Optional[int] = Field(
        default=None,
        description="Prompt token count from native provider telemetry (prompt_eval_count)",
    )
    output_tokens: Optional[int] = Field(
        default=None,
        description="Generated output token count from native provider telemetry (eval_count)",
    )
    total_tokens: Optional[int] = Field(
        default=None,
        description="Total tokens processed (prompt_tokens + output_tokens)",
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Primary inference duration in milliseconds (prompt_eval + eval)",
    )
    prompt_eval_duration_ms: Optional[float] = Field(
        default=None,
        description="Prompt evaluation duration in milliseconds (prompt_eval_duration / 1e6). Never labeled as TTFT.",
    )
    latency_breakdown: Optional[LatencyBreakdown] = Field(
        default=None,
        description="Granular duration separation (load, inference, queue wait)",
    )
    status: str = Field(
        default="completed",
        description="Execution status ('completed', 'failed')",
    )
    done_reason: Optional[str] = Field(
        default=None,
        description="Native completion termination reason ('stop', 'length')",
    )
    effective_settings: Optional[EffectiveGenerationSettings] = Field(
        default=None,
        description="Effective generation parameters actually applied",
    )
    trials: Optional[List[TrialTelemetry]] = Field(
        default=None,
        description="Individual measurements from repeated paired trials",
    )
    aggregate_latency: Optional[AggregateLatency] = Field(
        default=None,
        description="Aggregate latency statistics across trials",
    )
    native_telemetry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Direct unmanipulated provider telemetry payload",
    )


class ComparisonMetrics(BaseModel):
    """Calculated comparison metrics between original and optimized downstream executions."""

    prompt_tokens_saved: Optional[int] = Field(
        default=None,
        description="Prompt tokens saved (original - optimized)",
    )
    prompt_token_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in prompt tokens: ((orig - opt) / orig) * 100. Unavailable (None) if orig <= 0.",
    )
    output_tokens_diff: Optional[int] = Field(
        default=None,
        description="Delta in output tokens generated (optimized - original)",
    )
    output_token_change_percent: Optional[float] = Field(
        default=None,
        description="Percentage change in output tokens: ((opt - orig) / orig) * 100. Unavailable (None) if orig <= 0.",
    )
    total_tokens_saved: Optional[int] = Field(
        default=None,
        description="Delta in total tokens (original - optimized)",
    )
    total_token_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in total tokens: ((orig - opt) / orig) * 100. Unavailable (None) if orig <= 0.",
    )
    latency_delta_ms: Optional[float] = Field(
        default=None,
        description="Inference latency delta in ms (optimized - original). Based on pure inference duration.",
    )
    latency_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in latency: ((orig - opt) / orig) * 100. Unavailable (None) if orig <= 0.",
    )
    latency_note: str = Field(
        default="",
        description="Documentation of measurement methodology, warm-up, and trial count",
    )
    generation_parameters_identical: bool = Field(
        default=True,
        description="Whether all generation settings including effective num_ctx were identical between executions",
    )
    semantic_quality: str = Field(
        default="not_evaluated",
        description="Automated semantic quality evaluation status ('not_evaluated')",
    )
    semantic_quality_reason: str = Field(
        default="No automated judge model is configured. Semantic equivalence and response quality must be verified by human inspection.",
        description="Explanation of semantic quality evaluation status",
    )


class ComparisonRequest(BaseModel):
    """Request payload for comparing original vs optimized prompt outputs."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The original user prompt to compare",
    )
    downstream_model: Optional[str] = Field(
        default=None,
        description="Target downstream LLM model to execute both original and optimized prompts against (pinned for both calls)",
    )
    mode: OptimizationMode = Field(
        default="balanced",
        description="Argus prompt optimization profile (balanced, concise, structure)",
    )
    target_model: str = Field(
        default="gemini-1-5-flash",
        description="Target frontier model profile for illustrative token heuristics",
    )
    system_instruction: Optional[str] = Field(
        default=None,
        description="Optional downstream system instruction applied identically to both executions",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Generation temperature applied identically to both executions",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum generation tokens ceiling (num_predict) applied identically to both executions",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability applied identically to both executions",
    )
    enable_optimizer_routing: Optional[bool] = Field(
        default=None,
        description="Whether to enable automatic model routing for the prompt optimization step",
    )
    requested_optimizer_model: Optional[str] = Field(
        default=None,
        description="Optional local model override for the prompt optimization rewrite step",
    )
    trials: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of repeated paired trials to execute with alternating execution order",
    )
    warmup: bool = Field(
        default=True,
        description="Whether to perform a preliminary model warm-up call to eliminate cold-start load bias",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_whitespace(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return trimmed


class ComparisonResponse(BaseModel):
    """Complete end-to-end output comparison response."""

    original_execution: ComparisonExecutionTelemetry = Field(
        description="Downstream LLM execution using original, unoptimized prompt",
    )
    optimized_execution: ComparisonExecutionTelemetry = Field(
        description="Downstream LLM execution using Argus-optimized prompt",
    )
    metrics: ComparisonMetrics = Field(
        description="Comparative token and latency metrics",
    )
    optimizer_details: OptimizeResponse = Field(
        description="Detailed telemetry and artifacts from the prompt optimization stage",
    )
    downstream_model: str = Field(
        description="The pinned downstream model used for both executions",
    )
    effective_settings: Optional[EffectiveGenerationSettings] = Field(
        default=None,
        description="Common effective generation settings applied to both executions",
    )
    trials: int = Field(
        default=1,
        description="Number of paired trials executed",
    )
    warmup: bool = Field(
        default=True,
        description="Whether warm-up was performed prior to timing",
    )
    trial_measurements: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Combined paired trial telemetry across all trials",
    )
    aggregate_latency: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Aggregate latency comparison across trials",
    )
    timestamp: str = Field(
        description="ISO 8601 UTC timestamp of the comparison run",
    )
