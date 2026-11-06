"""Comparison service for end-to-end original vs optimized LLM downstream execution.

Coordinates:
1. Validation of the requested downstream model (pinned for both executions).
2. Execution A: Raw unmodified prompt dispatched directly to the pinned downstream LLM.
3. Optimization: Argus prompt optimization pipeline executed on the original prompt.
4. Execution B: Argus-optimized prompt dispatched to the same pinned downstream LLM.
5. Capture of native Ollama telemetry and honest calculation of comparison metrics.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional

from app.core.config import settings
from app.schemas.comparison import (
    ComparisonExecutionTelemetry,
    ComparisonMetrics,
    ComparisonRequest,
    ComparisonResponse,
)
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.services.ollama_service import OllamaService
from app.services.routing.registry import LocalModelRegistry


class ComparisonServiceError(Exception):
    """Base exception for comparison operations."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DownstreamModelNotFoundError(ComparisonServiceError):
    """Raised when the requested downstream model is not installed or available in Ollama."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Selected downstream model '{model_name}' is not installed or available in Ollama.",
            status_code=400,
        )


class ComparisonService:
    """Service orchestrating dual downstream execution and fair comparison."""

    @classmethod
    def calculate_metrics(
        cls,
        original: ComparisonExecutionTelemetry,
        optimized: ComparisonExecutionTelemetry,
    ) -> ComparisonMetrics:
        """Calculate honest delta metrics between original and optimized downstream runs."""
        orig_p = original.prompt_tokens
        opt_p = optimized.prompt_tokens
        if orig_p is not None and opt_p is not None:
            prompt_tokens_saved = orig_p - opt_p
            if orig_p > 0:
                prompt_reduction_pct = round(((orig_p - opt_p) / orig_p) * 100, 2)
            else:
                prompt_reduction_pct = 0.0
        else:
            prompt_tokens_saved = None
            prompt_reduction_pct = None

        orig_o = original.output_tokens
        opt_o = optimized.output_tokens
        if orig_o is not None and opt_o is not None:
            output_tokens_diff = opt_o - orig_o
            if orig_o > 0:
                output_change_pct = round(((opt_o - orig_o) / orig_o) * 100, 2)
            else:
                output_change_pct = 0.0
        else:
            output_tokens_diff = None
            output_change_pct = None

        orig_t = original.total_tokens
        opt_t = optimized.total_tokens
        if orig_t is not None and opt_t is not None:
            total_tokens_saved = orig_t - opt_t
            if orig_t > 0:
                total_reduction_pct = round(((orig_t - opt_t) / orig_t) * 100, 2)
            else:
                total_reduction_pct = 0.0
        else:
            total_tokens_saved = None
            total_reduction_pct = None

        orig_d = original.duration_ms
        opt_d = optimized.duration_ms
        if orig_d is not None and opt_d is not None:
            latency_delta_ms = round(opt_d - orig_d, 2)
            if orig_d > 0:
                latency_reduction_pct = round(((orig_d - opt_d) / orig_d) * 100, 2)
            else:
                latency_reduction_pct = 0.0
        else:
            latency_delta_ms = None
            latency_reduction_pct = None

        return ComparisonMetrics(
            prompt_tokens_saved=prompt_tokens_saved,
            prompt_token_reduction_percent=prompt_reduction_pct,
            output_tokens_diff=output_tokens_diff,
            output_token_change_percent=output_change_pct,
            total_tokens_saved=total_tokens_saved,
            total_token_reduction_percent=total_reduction_pct,
            latency_delta_ms=latency_delta_ms,
            latency_reduction_percent=latency_reduction_pct,
            semantic_quality="not_evaluated",
            semantic_quality_reason=(
                "No automated judge model is configured. Semantic equivalence and response quality "
                "must be verified by human inspection."
            ),
        )

    @classmethod
    async def compare(cls, request: ComparisonRequest) -> ComparisonResponse:
        """Run full end-to-end original vs optimized downstream inference comparison."""
        # 1. Resolve and pin downstream model
        target_model = request.downstream_model or settings.OLLAMA_MODEL
        discovered = await LocalModelRegistry.get_model(target_model, force_refresh=True)
        if discovered is None:
            raise DownstreamModelNotFoundError(target_model)
        pinned_model = discovered.name

        # 2. Execution A: Raw Original Prompt -> Downstream Model
        raw_a, dur_a, tel_a = await OllamaService.execute_downstream_call(
            prompt=request.prompt,
            model=pinned_model,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            top_p=request.top_p,
            num_predict=request.max_tokens,
        )

        p_a = tel_a.get("prompt_eval_count")
        o_a = tel_a.get("eval_count")
        t_a = (p_a + o_a) if (isinstance(p_a, int) and isinstance(o_a, int)) else None
        ttft_a = (
            round(tel_a["prompt_eval_duration"] / 1e6, 2)
            if isinstance(tel_a.get("prompt_eval_duration"), (int, float))
            else None
        )

        exec_a = ComparisonExecutionTelemetry(
            prompt=request.prompt,
            response=raw_a,
            model=pinned_model,
            provider="ollama",
            prompt_tokens=p_a,
            output_tokens=o_a,
            total_tokens=t_a,
            duration_ms=dur_a,
            ttft_ms=ttft_a,
            status="completed",
            done_reason=tel_a.get("done_reason"),
            native_telemetry=tel_a,
        )

        # 3. Optimization Step: Run Argus Optimizer on Original Prompt
        opt_req = OptimizeRequest(
            prompt=request.prompt,
            mode=request.mode,
            target_model=request.target_model,
            enable_routing=request.enable_optimizer_routing,
            requested_model=request.requested_optimizer_model,
        )
        opt_res = await OllamaService.optimize_prompt(opt_req)

        # 4. Execution B: Optimized Prompt -> Same Pinned Downstream Model
        raw_b, dur_b, tel_b = await OllamaService.execute_downstream_call(
            prompt=opt_res.optimized_prompt,
            model=pinned_model,
            system_instruction=request.system_instruction,
            temperature=request.temperature,
            top_p=request.top_p,
            num_predict=request.max_tokens,
        )

        p_b = tel_b.get("prompt_eval_count")
        o_b = tel_b.get("eval_count")
        t_b = (p_b + o_b) if (isinstance(p_b, int) and isinstance(o_b, int)) else None
        ttft_b = (
            round(tel_b["prompt_eval_duration"] / 1e6, 2)
            if isinstance(tel_b.get("prompt_eval_duration"), (int, float))
            else None
        )

        exec_b = ComparisonExecutionTelemetry(
            prompt=opt_res.optimized_prompt,
            response=raw_b,
            model=pinned_model,
            provider="ollama",
            prompt_tokens=p_b,
            output_tokens=o_b,
            total_tokens=t_b,
            duration_ms=dur_b,
            ttft_ms=ttft_b,
            status="completed",
            done_reason=tel_b.get("done_reason"),
            native_telemetry=tel_b,
        )

        # 5. Calculate Metrics
        metrics = cls.calculate_metrics(exec_a, exec_b)

        return ComparisonResponse(
            original_execution=exec_a,
            optimized_execution=exec_b,
            metrics=metrics,
            optimizer_details=opt_res,
            downstream_model=pinned_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
