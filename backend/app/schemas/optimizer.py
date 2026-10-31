"""Pydantic schemas for prompt optimization and Ollama status."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


OptimizationMode = Literal["balanced", "concise", "structure"]

OptimizationStrategy = Literal["compress", "clarify", "structure", "enrich", "preserve"]

TaskCategory = Literal[
    "coding",
    "data_analysis",
    "writing",
    "summarization",
    "research",
    "extraction",
    "planning",
    "general",
]

OptimizationOutcome = Literal["validated_rewrite", "unchanged_original", "fallback"]

LengthResult = Literal["compressed", "expanded", "unchanged"]

ValidationStatus = Literal["passed", "warning", "fallback"]


class TokenEstimation(BaseModel):
    """Token footprint estimation based on documented character heuristics."""

    char_count: int = Field(description="Total character count")
    estimated_tokens: int = Field(
        description="Estimated token count (~4 characters per token )"
    )


class OptimizeRequest(BaseModel):
    """Incoming request to optimize a prompt."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The raw prompt text to optimize",
    )
    mode: OptimizationMode = Field(
        default="balanced",
        description="Optimization profile (balanced, concise, or structure)",
    )
    target_model: str = Field(
        default="gpt-4o",
        description="Target destination model profile for illustrative token estimation",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_whitespace(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return trimmed


class OptimizeResponse(BaseModel):
    """Structured response containing rewritten prompt, task awareness, and token telemetry."""

    original_prompt: str = Field(description="The unoptimized input prompt")
    optimized_prompt: str = Field(description="The rewritten prompt from Ollama")
    mode: OptimizationMode = Field(description="The optimization profile used")
    target_model: str = Field(description="The target frontier model")
    
    # Task-Aware Metadata (Phase 3.1)
    task_category: str = Field(
        default="general",
        description="Detected task category (coding, data_analysis, extraction, etc.)",
    )
    strategy: OptimizationStrategy = Field(
        default="compress",
        description="Selected optimization strategy (compress, clarify, structure, enrich, preserve)",
    )
    applied_strategy: str = Field(
        default="compress",
        description="Strategy actually applied to the candidate (e.g. compress, clarify, structure, preserve)",
    )
    outcome: OptimizationOutcome = Field(
        default="validated_rewrite",
        description="Outcome of optimization (validated_rewrite, unchanged_original, fallback)",
    )
    length_result: LengthResult = Field(
        default="unchanged",
        description="Length outcome: compressed (shorter), expanded (longer), or unchanged",
    )
    validation_status: ValidationStatus = Field(
        default="passed",
        description="Verification check status for constraint preservation and hallucination prevention",
    )
    is_preserved: bool = Field(
        default=False,
        description="Whether the prompt was kept unchanged or minimally adjusted due to existing high quality or safety fallback",
    )
    primary_action: Optional[str] = Field(
        default=None,
        description="Identified primary action verb (e.g. build, explain, summarize)",
    )
    primary_deliverable: Optional[str] = Field(
        default=None,
        description="Identified primary deliverable artifact (e.g. program, script, explanation, plan)",
    )
    explicit_constraints: List[str] = Field(
        default_factory=list,
        description="Explicit constraints, prohibitions, and format specifications identified in the original prompt",
    )
    required_operations: List[str] = Field(
        default_factory=list,
        description="Extracted required functional operations, calculations, and data flows",
    )

    # Metrics & Telemetry
    original_metrics: TokenEstimation = Field(description="Token footprint of the original prompt")
    optimized_metrics: TokenEstimation = Field(description="Token footprint of the rewritten prompt")
    token_reduction_percent: int = Field(
        description="Estimated token reduction percentage (positive=reduction, 0=none)"
    )
    token_change_percent: int = Field(
        default=0,
        description="Signed token change percentage (negative=reduction, positive=expansion)",
    )
    token_savings: int = Field(
        default=0,
        description="Estimated number of tokens saved (positive=saved, 0=expansion or unchanged)",
    )
    char_delta: int = Field(
        default=0,
        description="Character delta: final_character_count - original_character_count (negative=shorter, positive=longer, 0=equal)",
    )
    preserved_constraints: List[str] = Field(
        default_factory=list,
        description="Categories of preserved constraints (e.g., intent, formatting, variables)",
    )
    execution_time_ms: int = Field(description="Inference duration in milliseconds")
    model: str = Field(description="Local model used for inference")
    timestamp: str = Field(description="ISO-8601 UTC timestamp")


class OllamaStatusResponse(BaseModel):
    """Health and model availability status of the local Ollama instance."""

    connected: bool = Field(description="Whether the Ollama server responded")
    base_url: str = Field(description="The configured Ollama HTTP base URL")
    model: str = Field(description="The configured default optimization model")
    model_installed: bool = Field(
        description="Whether the configured model is installed and ready"
    )
    available_models: List[str] = Field(
        default_factory=list,
        description="List of model tags installed in local Ollama",
    )
    error: Optional[str] = Field(
        default=None,
        description="Diagnostic error message if connection or model check failed",
    )
