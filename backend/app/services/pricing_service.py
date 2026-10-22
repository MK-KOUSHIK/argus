"""Model Pricing & Cost Calculation Service for Argus (Phase 4).

Provides documented pricing tables and strictly distinguishes between:
1. Estimated API cost (based on tokenizer / heuristic counts prior to or without execution)
2. Actual API cost (based on provider-reported usage numbers)
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Pricing configuration for a specific LLM model (per 1M tokens in USD)."""

    model: str
    input_cost_per_million: float
    output_cost_per_million: float
    currency: str = "USD"
    effective_date: str = "2024-10"

    @property
    def input_cost_per_token(self) -> float:
        return self.input_cost_per_million / 1_000_000.0

    @property
    def output_cost_per_token(self) -> float:
        return self.output_cost_per_million / 1_000_000.0


class CostBreakdown(BaseModel):
    """Cost breakdown distinguishing estimated versus actual provider costs."""

    input_cost_usd: float = Field(description="Cost incurred for input prompt tokens")
    output_cost_usd: float = Field(description="Cost incurred for completion tokens")
    total_cost_usd: float = Field(description="Total cost in USD")
    is_estimated: bool = Field(description="True if based on estimation, False if provider-reported")
    currency: str = "USD"


class CostSavings(BaseModel):
    """Comparison of baseline vs optimized cost."""

    baseline_cost_usd: float
    optimized_cost_usd: float
    cost_saved_usd: float
    savings_percent: float
    is_estimated: bool


# Canonical model pricing registry (USD per 1,000,000 tokens)
MODEL_PRICING_REGISTRY: Dict[str, ModelPricing] = {
    # OpenAI Frontier Models
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-2024-08-06": ModelPricing(
        model="gpt-4o-2024-08-06",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    "gpt-4-turbo": ModelPricing(
        model="gpt-4-turbo",
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
    ),
    "gpt-3.5-turbo": ModelPricing(
        model="gpt-3.5-turbo",
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
    ),
    # Anthropic Models (for cross-provider reference)
    "claude-3-5-sonnet": ModelPricing(
        model="claude-3-5-sonnet",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-3-5-haiku": ModelPricing(
        model="claude-3-5-haiku",
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    # Google Gemini Models (for cross-provider reference)
    "gemini-1-5-pro": ModelPricing(
        model="gemini-1-5-pro",
        input_cost_per_million=3.50,
        output_cost_per_million=10.50,
    ),
    "gemini-1-5-flash": ModelPricing(
        model="gemini-1-5-flash",
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
}

DEFAULT_FALLBACK_PRICING = MODEL_PRICING_REGISTRY["gpt-4o"]


class PricingService:
    """Calculates estimated and actual model inference costs."""

    @classmethod
    def get_pricing(cls, model_name: str) -> ModelPricing:
        """Retrieve pricing for a given model, with fuzzy matching or fallback."""
        normalized = model_name.lower().strip()
        if normalized in MODEL_PRICING_REGISTRY:
            return MODEL_PRICING_REGISTRY[normalized]

        # Prefix matching (e.g. gpt-4o-mini-2024-07-18 -> gpt-4o-mini)
        for key, pricing in MODEL_PRICING_REGISTRY.items():
            if normalized.startswith(key):
                return pricing

        return DEFAULT_FALLBACK_PRICING

    @classmethod
    def estimate_cost(
        cls,
        prompt_tokens: int,
        estimated_completion_tokens: int = 0,
        model: str = "gpt-4o",
    ) -> CostBreakdown:
        """Estimate cost before call or without execution."""
        pricing = cls.get_pricing(model)
        input_cost = prompt_tokens * pricing.input_cost_per_token
        output_cost = estimated_completion_tokens * pricing.output_cost_per_token
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_cost_usd=round(input_cost, 7),
            output_cost_usd=round(output_cost, 7),
            total_cost_usd=round(total_cost, 7),
            is_estimated=True,
            currency="USD",
        )

    @classmethod
    def calculate_actual_cost(
        cls,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o",
    ) -> CostBreakdown:
        """Calculate actual cost based on verified provider-reported tokens."""
        pricing = cls.get_pricing(model)
        input_cost = prompt_tokens * pricing.input_cost_per_token
        output_cost = completion_tokens * pricing.output_cost_per_token
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_cost_usd=round(input_cost, 7),
            output_cost_usd=round(output_cost, 7),
            total_cost_usd=round(total_cost, 7),
            is_estimated=False,
            currency="USD",
        )

    @classmethod
    def compare_costs(
        cls,
        baseline_cost: CostBreakdown,
        optimized_cost: CostBreakdown,
    ) -> CostSavings:
        """Compute cost difference between baseline and optimized execution."""
        b_total = baseline_cost.total_cost_usd
        o_total = optimized_cost.total_cost_usd
        saved = b_total - o_total
        pct = (saved / b_total * 100.0) if b_total > 0 else 0.0

        is_estimated = baseline_cost.is_estimated or optimized_cost.is_estimated

        return CostSavings(
            baseline_cost_usd=round(b_total, 7),
            optimized_cost_usd=round(o_total, 7),
            cost_saved_usd=round(saved, 7),
            savings_percent=round(pct, 2),
            is_estimated=is_estimated,
        )
