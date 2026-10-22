"""Comparative Execution Service for Argus (Phase 4).

Coordinates Path A (Baseline: original prompt -> Main LLM) and
Path B (Optimized: original prompt -> Argus Local Optimizer -> Main LLM)
with identical generation settings and strictly separate prompt and metric tracking.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.optimizer import OptimizationMode, OptimizeRequest, OptimizeResponse
from app.services.main_llm_service import (
    BaseLLMProvider,
    MainLLMService,
    ProviderResponse,
    ProviderUsage,
)
from app.services.ollama_service import OllamaService
from app.services.pricing_service import CostBreakdown, CostSavings, PricingService
from app.services.tokenizer_service import TokenCountResult, TokenizerService


class PathExecutionResult(BaseModel):
    """Detailed result for a single execution path (Baseline or Optimized)."""

    path_name: str = Field(description="'baseline' or 'optimized'")
    prompt_used: str = Field(description="Exact prompt sent to the main LLM")
    tokenizer_tokens: TokenCountResult = Field(description="Token count computed by provider tokenizer")
    provider_usage: ProviderUsage = Field(description="Actual tokens reported by provider")
    cost: CostBreakdown = Field(description="Actual cost calculated from provider usage")
    llm_response: str = Field(description="Response text returned by frontier LLM")
    latency_ms: int = Field(description="Latency of provider LLM execution")
    total_latency_ms: int = Field(description="Total end-to-end latency including local optimization if applicable")
    is_mock: bool = Field(default=False, description="Whether execution was performed by a mock provider")


class ComparisonResult(BaseModel):
    """Side-by-side comparative evaluation of Baseline vs. Optimized paths."""

    original_prompt: str
    optimized_prompt: str
    mode: OptimizationMode
    model: str

    # Local Optimization Stage Details
    optimizer_details: OptimizeResponse

    # Path Executions
    baseline: PathExecutionResult
    optimized: PathExecutionResult

    # Comparative Telemetry
    prompt_tokens_saved: int = Field(description="Baseline prompt tokens minus optimized prompt tokens")
    prompt_tokens_saved_percent: float = Field(description="Percentage reduction in prompt tokens")
    total_tokens_saved: int = Field(description="Baseline total tokens minus optimized total tokens")
    total_tokens_saved_percent: float = Field(description="Percentage reduction in total tokens")

    # Cost Telemetry
    cost_comparison: CostSavings = Field(description="Cost delta distinguishing actual provider reported costs")

    # Latency Telemetry
    latency_overhead_ms: int = Field(description="Additional latency introduced by local Argus optimization")

    # Quality & Integrity
    constraints_preserved: bool = Field(description="Whether explicit negative constraints & keys were preserved")
    is_mock: bool = Field(description="True if provider was mocked, False if real provider was executed")


class ComparativeExecutionService:
    """Orchestrates Baseline vs. Optimized LLM execution with identical hyperparameters."""

    @classmethod
    async def execute_baseline_path(
        cls,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[BaseLLMProvider] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> PathExecutionResult:
        """Execute Path A: original prompt -> Main LLM directly."""
        target_model = model or settings.MAIN_LLM_MODEL
        tokenizer_result = TokenizerService.count_tokens(prompt, target_model)

        # Call main LLM
        active_provider = provider or MainLLMService.get_provider()
        llm_res: ProviderResponse = await active_provider.generate(
            prompt=prompt,
            model=target_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return PathExecutionResult(
            path_name="baseline",
            prompt_used=prompt,
            tokenizer_tokens=tokenizer_result,
            provider_usage=llm_res.usage,
            cost=llm_res.cost,
            llm_response=llm_res.content,
            latency_ms=llm_res.latency_ms,
            total_latency_ms=llm_res.latency_ms,
            is_mock=llm_res.is_mock,
        )

    @classmethod
    async def execute_optimized_path(
        cls,
        prompt: str,
        mode: OptimizationMode = "balanced",
        model: Optional[str] = None,
        provider: Optional[BaseLLMProvider] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[PathExecutionResult, OptimizeResponse]:
        """Execute Path B: original prompt -> Argus Local Optimizer -> Main LLM."""
        target_model = model or settings.MAIN_LLM_MODEL

        # 1. Local Argus Optimization (Ollama)
        opt_req = OptimizeRequest(prompt=prompt, mode=mode, target_model=target_model)
        opt_res: OptimizeResponse = await OllamaService.optimize_prompt(opt_req)

        # 2. Tokenize optimized prompt
        tokenizer_result = TokenizerService.count_tokens(opt_res.optimized_prompt, target_model)

        # 3. Call main LLM with optimized prompt
        active_provider = provider or MainLLMService.get_provider()
        llm_res: ProviderResponse = await active_provider.generate(
            prompt=opt_res.optimized_prompt,
            model=target_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_lat = opt_res.execution_time_ms + llm_res.latency_ms

        path_result = PathExecutionResult(
            path_name="optimized",
            prompt_used=opt_res.optimized_prompt,
            tokenizer_tokens=tokenizer_result,
            provider_usage=llm_res.usage,
            cost=llm_res.cost,
            llm_response=llm_res.content,
            latency_ms=llm_res.latency_ms,
            total_latency_ms=total_lat,
            is_mock=llm_res.is_mock,
        )

        return path_result, opt_res

    @classmethod
    async def compare(
        cls,
        prompt: str,
        mode: OptimizationMode = "balanced",
        model: Optional[str] = None,
        provider: Optional[BaseLLMProvider] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ComparisonResult:
        """Run both Path A and Path B and compute comparative savings and telemetry."""
        target_model = model or settings.MAIN_LLM_MODEL

        # Execute Path A: Baseline
        baseline_res = await cls.execute_baseline_path(
            prompt=prompt,
            model=target_model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Execute Path B: Optimized
        optimized_res, opt_details = await cls.execute_optimized_path(
            prompt=prompt,
            mode=mode,
            model=target_model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Prompt token comparison
        base_p_toks = baseline_res.provider_usage.prompt_tokens
        opt_p_toks = optimized_res.provider_usage.prompt_tokens
        p_saved = base_p_toks - opt_p_toks
        p_saved_pct = (p_saved / base_p_toks * 100.0) if base_p_toks > 0 else 0.0

        # Total token comparison
        base_t_toks = baseline_res.provider_usage.total_tokens
        opt_t_toks = optimized_res.provider_usage.total_tokens
        t_saved = base_t_toks - opt_t_toks
        t_saved_pct = (t_saved / base_t_toks * 100.0) if base_t_toks > 0 else 0.0

        # Cost comparison
        cost_comparison = PricingService.compare_costs(
            baseline_cost=baseline_res.cost,
            optimized_cost=optimized_res.cost,
        )

        # Latency overhead: additional time compared to baseline direct call
        latency_overhead = optimized_res.total_latency_ms - baseline_res.total_latency_ms

        # Constraint check: verify optimizer preserved validation status
        constraints_preserved = opt_details.validation_status != "warning"

        return ComparisonResult(
            original_prompt=prompt,
            optimized_prompt=opt_details.optimized_prompt,
            mode=mode,
            model=target_model,
            optimizer_details=opt_details,
            baseline=baseline_res,
            optimized=optimized_res,
            prompt_tokens_saved=p_saved,
            prompt_tokens_saved_percent=round(p_saved_pct, 2),
            total_tokens_saved=t_saved,
            total_tokens_saved_percent=round(t_saved_pct, 2),
            cost_comparison=cost_comparison,
            latency_overhead_ms=latency_overhead,
            constraints_preserved=constraints_preserved,
            is_mock=baseline_res.is_mock or optimized_res.is_mock,
        )
