"""Pydantic schemas for Main LLM Integration & Benchmark Endpoints (Phase 4)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.optimizer import OptimizationMode
from app.services.comparative_execution_service import ComparisonResult
from app.benchmarks.runner import BenchmarkReport
from app.services.pricing_service import ModelPricing


class MainLLMStatusResponse(BaseModel):
    """External provider configuration and availability status."""

    provider: str = Field(description="Configured provider (e.g. openai)")
    model: str = Field(description="Target frontier model")
    is_configured: bool = Field(description="Whether required credentials are set in environment")
    base_url: str = Field(description="Provider API base URL")
    message: str = Field(description="Status description")


class CompareRequest(BaseModel):
    """Request to execute Baseline vs. Optimized prompt comparison on Main LLM."""

    prompt: str = Field(..., min_length=1, description="Raw input prompt to compare")
    mode: OptimizationMode = Field(default="balanced", description="Argus optimization mode")
    target_model: str = Field(default="gpt-4o", description="Target frontier model")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8000)

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return trimmed


class BenchmarkRunRequest(BaseModel):
    """Request to execute the reproducible benchmark suite."""

    live: bool = Field(
        default=False,
        description="Run live benchmark with real external API (requires valid API key)",
    )
    model: Optional[str] = Field(
        default="gpt-4o",
        description="Target model for benchmark run",
    )


class PricingTableResponse(BaseModel):
    """Registered frontier models and their published token pricing."""

    pricing: Dict[str, ModelPricing]
