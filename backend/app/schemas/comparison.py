"""Pydantic schemas for end-to-end original vs optimized LLM output comparison."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.optimizer import OptimizationMode, OptimizeResponse


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
        description="End-to-end inference execution duration in milliseconds",
    )
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time to first token in milliseconds (prompt_eval_duration / 1e6)",
    )
    status: str = Field(
        default="completed",
        description="Execution status ('completed', 'failed')",
    )
    done_reason: Optional[str] = Field(
        default=None,
        description="Native completion termination reason ('stop', 'length')",
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
        description="Percentage reduction in prompt tokens: ((orig - opt) / orig) * 100",
    )
    output_tokens_diff: Optional[int] = Field(
        default=None,
        description="Delta in output tokens generated (optimized - original)",
    )
    output_token_change_percent: Optional[float] = Field(
        default=None,
        description="Percentage change in output tokens: ((opt - orig) / orig) * 100",
    )
    total_tokens_saved: Optional[int] = Field(
        default=None,
        description="Delta in total tokens (original - optimized)",
    )
    total_token_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in total tokens: ((orig - opt) / orig) * 100",
    )
    latency_delta_ms: Optional[float] = Field(
        default=None,
        description="Inference latency delta in ms (optimized - original)",
    )
    latency_reduction_percent: Optional[float] = Field(
        default=None,
        description="Percentage reduction in latency: ((orig - opt) / orig) * 100",
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
    timestamp: str = Field(
        description="ISO 8601 UTC timestamp of the comparison run",
    )
