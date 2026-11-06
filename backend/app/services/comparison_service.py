"""Comparison service for end-to-end original vs optimized LLM downstream execution.

Coordinates:
1. Validation of the requested downstream model (pinned for both executions).
2. Argus prompt optimization pipeline executed on the original prompt.
3. Model warm-up to eliminate cold-start weight loading bias.
4. Adaptive context budgeting with shared common context window to guarantee generation parity.
5. Repeated paired trials with alternating execution order ([A, B], [B, A], [A, B]...).
6. Duration separation: model load, pure inference (prompt_eval + eval), queue wait, and wall-clock.
7. Capture of native Ollama telemetry, zero-denominator safety, and honest metrics reporting.
"""

from datetime import datetime, timezone
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.comparison import (
    AggregateLatency,
    ComparisonExecutionTelemetry,
    ComparisonMetrics,
    ComparisonRequest,
    ComparisonResponse,
    EffectiveGenerationSettings,
    LatencyBreakdown,
    TrialTelemetry,
)
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.services.adaptive_context_manager import AdaptiveContextManager
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
    def _build_trial(
        cls,
        trial_idx: int,
        order: int,
        raw_response: str,
        wall_duration_ms: float,
        telemetry: Dict[str, Any],
    ) -> TrialTelemetry:
        """Parse raw telemetry and build granular duration separation for an individual trial."""
        load_ns = telemetry.get("load_duration")
        load_ms = round(load_ns / 1e6, 2) if isinstance(load_ns, (int, float)) else None

        prompt_eval_ns = telemetry.get("prompt_eval_duration")
        prompt_eval_ms = round(prompt_eval_ns / 1e6, 2) if isinstance(prompt_eval_ns, (int, float)) else None

        eval_ns = telemetry.get("eval_duration")
        eval_ms = round(eval_ns / 1e6, 2) if isinstance(eval_ns, (int, float)) else None

        # Required native measurements: both prompt_eval_duration and eval_duration must be available
        if prompt_eval_ms is not None and eval_ms is not None:
            inference_ms = round(prompt_eval_ms + eval_ms, 2)
        else:
            # Do NOT fall back to wall-clock duration. Set inference_duration_ms=None.
            inference_ms = None

        queue_wait_ms = telemetry.get("queue_wait_ms", 0.0)
        p_tokens = telemetry.get("prompt_eval_count")
        o_tokens = telemetry.get("eval_count")
        t_tokens = (p_tokens + o_tokens) if (isinstance(p_tokens, int) and isinstance(o_tokens, int)) else None

        breakdown = LatencyBreakdown(
            model_load_duration_ms=load_ms,
            prompt_eval_duration_ms=prompt_eval_ms,
            eval_duration_ms=eval_ms,
            inference_duration_ms=inference_ms,
            queue_wait_duration_ms=queue_wait_ms,
            wall_time_ms=wall_duration_ms,
        )

        return TrialTelemetry(
            trial_index=trial_idx,
            execution_order=order,
            prompt_tokens=p_tokens,
            output_tokens=o_tokens,
            total_tokens=t_tokens,
            latency=breakdown,
            done_reason=telemetry.get("done_reason"),
            response=raw_response,
        )

    @classmethod
    def _compute_aggregate(cls, trials: List[TrialTelemetry]) -> AggregateLatency:
        """Compute aggregate statistics across multiple trials."""
        inf_list = [t.latency.inference_duration_ms for t in trials if t.latency.inference_duration_ms is not None]
        wall_list = [t.latency.wall_time_ms for t in trials if t.latency.wall_time_ms is not None]
        pe_list = [t.latency.prompt_eval_duration_ms for t in trials if t.latency.prompt_eval_duration_ms is not None]
        eval_list = [t.latency.eval_duration_ms for t in trials if t.latency.eval_duration_ms is not None]
        queue_list = [t.latency.queue_wait_duration_ms for t in trials if t.latency.queue_wait_duration_ms is not None]
        load_list = [t.latency.model_load_duration_ms for t in trials if t.latency.model_load_duration_ms is not None]

        return AggregateLatency(
            trials_count=len(trials),
            mean_inference_duration_ms=round(statistics.mean(inf_list), 2) if inf_list else None,
            median_inference_duration_ms=round(statistics.median(inf_list), 2) if inf_list else None,
            mean_wall_time_ms=round(statistics.mean(wall_list), 2) if wall_list else None,
            mean_prompt_eval_duration_ms=round(statistics.mean(pe_list), 2) if pe_list else None,
            mean_eval_duration_ms=round(statistics.mean(eval_list), 2) if eval_list else None,
            mean_queue_wait_duration_ms=round(statistics.mean(queue_list), 2) if queue_list else None,
            mean_model_load_duration_ms=round(statistics.mean(load_list), 2) if load_list else None,
        )

    @classmethod
    def calculate_metrics(
        cls,
        original: ComparisonExecutionTelemetry,
        optimized: ComparisonExecutionTelemetry,
        num_trials: int = 1,
        generation_parameters_identical: bool = True,
        warmup_succeeded: bool = True,
        warmup_error: Optional[str] = None,
        warmup_attempted: bool = True,
    ) -> ComparisonMetrics:
        """Calculate honest delta metrics between original and optimized downstream runs."""
        # 1. Prompt Tokens Reduction (Zero denominator must yield None, not 0.0)
        orig_p = original.prompt_tokens
        opt_p = optimized.prompt_tokens
        if orig_p is not None and opt_p is not None:
            prompt_tokens_saved = orig_p - opt_p
            if orig_p > 0:
                prompt_reduction_pct = round(((orig_p - opt_p) / orig_p) * 100, 2)
            else:
                prompt_reduction_pct = None  # Undefined / unavailable for 0 denominator
        else:
            prompt_tokens_saved = None
            prompt_reduction_pct = None

        # 2. Output Tokens Delta (Zero denominator must yield None, not 0.0)
        orig_o = original.output_tokens
        opt_o = optimized.output_tokens
        if orig_o is not None and opt_o is not None:
            output_tokens_diff = opt_o - orig_o
            if orig_o > 0:
                output_change_pct = round(((opt_o - orig_o) / orig_o) * 100, 2)
            else:
                output_change_pct = None  # Undefined / unavailable for 0 denominator
        else:
            output_tokens_diff = None
            output_change_pct = None

        # 3. Total Tokens Delta (Zero denominator must yield None, not 0.0)
        orig_t = original.total_tokens
        opt_t = optimized.total_tokens
        if orig_t is not None and opt_t is not None:
            total_tokens_saved = orig_t - opt_t
            if orig_t > 0:
                total_reduction_pct = round(((orig_t - opt_t) / orig_t) * 100, 2)
            else:
                total_reduction_pct = None  # Undefined / unavailable for 0 denominator
        else:
            total_tokens_saved = None
            total_reduction_pct = None

        # 4. Latency Delta (Based on pure inference duration: prompt_eval + eval)
        orig_d = original.duration_ms
        opt_d = optimized.duration_ms
        if orig_d is not None and opt_d is not None:
            latency_delta_ms = round(opt_d - orig_d, 2)
            if orig_d > 0:
                latency_reduction_pct = round(((orig_d - opt_d) / orig_d) * 100, 2)
            else:
                latency_reduction_pct = None  # Undefined / unavailable for 0 denominator
        else:
            latency_delta_ms = None
            latency_reduction_pct = None

        if warmup_attempted and not warmup_succeeded:
            cold_start_warning = (
                f"Model warm-up failed ({warmup_error or 'execution error'}). "
                "Latency measurements may include cold-start effects and initial model loading overhead."
            )
            if num_trials > 1:
                latency_note = (
                    f"{cold_start_warning} Evaluated across {num_trials} paired trials with alternating execution order."
                )
            else:
                latency_note = (
                    f"{cold_start_warning} A single pair does not prove statistically significant latency improvement, "
                    "and measurements may be biased by cold start."
                )
        elif not warmup_attempted:
            if num_trials > 1:
                latency_note = (
                    f"Model warm-up was disabled. Evaluated across {num_trials} paired trials with alternating "
                    "execution order. Initial trials may include cold-start effects."
                )
            else:
                latency_note = (
                    "Single trial execution with model warm-up disabled. A single pair does not prove statistically "
                    "significant latency improvement, and measurements may include cold-start effects."
                )
        elif num_trials > 1:
            latency_note = (
                f"Evaluated across {num_trials} paired trials with alternating execution order and preliminary "
                "model warm-up. Individual trial durations and aggregate statistics are reported."
            )
        else:
            latency_note = (
                "Single trial execution with model warm-up. A single pair does not prove statistically "
                "significant latency improvement due to token decoding and system load variance."
            )

        return ComparisonMetrics(
            prompt_tokens_saved=prompt_tokens_saved,
            prompt_token_reduction_percent=prompt_reduction_pct,
            output_tokens_diff=output_tokens_diff,
            output_token_change_percent=output_change_pct,
            total_tokens_saved=total_tokens_saved,
            total_token_reduction_percent=total_reduction_pct,
            latency_delta_ms=latency_delta_ms,
            latency_reduction_percent=latency_reduction_pct,
            latency_note=latency_note,
            generation_parameters_identical=generation_parameters_identical,
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

        # 2. Run Argus Optimizer FIRST so both prompts are ready before downstream latency trials
        opt_req = OptimizeRequest(
            prompt=request.prompt,
            mode=request.mode,
            target_model=request.target_model,
            enable_routing=request.enable_optimizer_routing,
            requested_model=request.requested_optimizer_model,
        )
        opt_res = await OllamaService.optimize_prompt(opt_req)
        original_prompt = request.prompt
        optimized_prompt = opt_res.optimized_prompt

        # 3. Model Warm-Up: eliminate cold-start weight loading bias
        warmup_attempted = False
        warmup_succeeded = False
        warmup_error: Optional[str] = None

        if request.warmup:
            warmup_attempted = True
            try:
                await OllamaService.execute_downstream_call(
                    prompt="warmup",
                    model=pinned_model,
                    num_predict=1,
                    num_ctx=2048,
                )
                warmup_succeeded = True
            except Exception as exc:
                warmup_succeeded = False
                warmup_error = str(exc)

        # 4. Context Budgeting: compute budgets for both prompts and snap to a common num_ctx
        effective_predict = (
            request.max_tokens if request.max_tokens is not None else settings.OLLAMA_NUM_PREDICT
        )
        effective_temp = (
            request.temperature if request.temperature is not None else settings.OLLAMA_TEMPERATURE
        )
        effective_top_p = request.top_p if request.top_p is not None else settings.OLLAMA_TOP_P
        system_inst = request.system_instruction or ""

        budget_a = AdaptiveContextManager.calculate_budget(
            system_instruction=system_inst,
            formatted_prompt=original_prompt,
            output_budget=effective_predict,
        )
        budget_b = AdaptiveContextManager.calculate_budget(
            system_instruction=system_inst,
            formatted_prompt=optimized_prompt,
            output_budget=effective_predict,
        )

        common_num_ctx = max(budget_a["selected_num_ctx"], budget_b["selected_num_ctx"])

        settings_a = EffectiveGenerationSettings(
            model=pinned_model,
            num_ctx=common_num_ctx,
            num_predict=effective_predict,
            temperature=effective_temp,
            top_p=effective_top_p,
            system_instruction=request.system_instruction,
            context_window_shared=True,
        )
        settings_b = EffectiveGenerationSettings(
            model=pinned_model,
            num_ctx=common_num_ctx,
            num_predict=effective_predict,
            temperature=effective_temp,
            top_p=effective_top_p,
            system_instruction=request.system_instruction,
            context_window_shared=True,
        )

        # Calculate equality from the effective settings passed to the actual Ollama calls
        settings_identical = (
            settings_a.model == settings_b.model
            and settings_a.num_ctx == settings_b.num_ctx
            and settings_a.num_predict == settings_b.num_predict
            and settings_a.temperature == settings_b.temperature
            and settings_a.top_p == settings_b.top_p
            and settings_a.system_instruction == settings_b.system_instruction
        )

        # 5. Paired Trials with Alternating Execution Order
        num_trials = max(1, request.trials)
        trials_a: List[TrialTelemetry] = []
        trials_b: List[TrialTelemetry] = []

        for trial_idx in range(num_trials):
            # Alternating order:
            # Even trial (0, 2, ...): [A first (order=1), B second (order=2)]
            # Odd trial (1, 3, ...): [B first (order=1), A second (order=2)]
            if trial_idx % 2 == 0:
                raw_a, dur_a, tel_a = await OllamaService.execute_downstream_call(
                    prompt=original_prompt,
                    model=pinned_model,
                    system_instruction=request.system_instruction,
                    temperature=effective_temp,
                    top_p=effective_top_p,
                    num_predict=effective_predict,
                    num_ctx=common_num_ctx,
                )
                trials_a.append(cls._build_trial(trial_idx, 1, raw_a, dur_a, tel_a))

                raw_b, dur_b, tel_b = await OllamaService.execute_downstream_call(
                    prompt=optimized_prompt,
                    model=pinned_model,
                    system_instruction=request.system_instruction,
                    temperature=effective_temp,
                    top_p=effective_top_p,
                    num_predict=effective_predict,
                    num_ctx=common_num_ctx,
                )
                trials_b.append(cls._build_trial(trial_idx, 2, raw_b, dur_b, tel_b))
            else:
                raw_b, dur_b, tel_b = await OllamaService.execute_downstream_call(
                    prompt=optimized_prompt,
                    model=pinned_model,
                    system_instruction=request.system_instruction,
                    temperature=effective_temp,
                    top_p=effective_top_p,
                    num_predict=effective_predict,
                    num_ctx=common_num_ctx,
                )
                trials_b.append(cls._build_trial(trial_idx, 1, raw_b, dur_b, tel_b))

                raw_a, dur_a, tel_a = await OllamaService.execute_downstream_call(
                    prompt=original_prompt,
                    model=pinned_model,
                    system_instruction=request.system_instruction,
                    temperature=effective_temp,
                    top_p=effective_top_p,
                    num_predict=effective_predict,
                    num_ctx=common_num_ctx,
                )
                trials_a.append(cls._build_trial(trial_idx, 2, raw_a, dur_a, tel_a))

        # 6. Aggregate Statistics across Trials
        agg_a = cls._compute_aggregate(trials_a)
        agg_b = cls._compute_aggregate(trials_b)

        # 7. Build Representative Execution Telemetry
        rep_a = trials_a[0]
        rep_b = trials_b[0]

        exec_a = ComparisonExecutionTelemetry(
            prompt=original_prompt,
            response=rep_a.response,
            model=pinned_model,
            provider="ollama",
            prompt_tokens=rep_a.prompt_tokens,
            output_tokens=rep_a.output_tokens,
            total_tokens=rep_a.total_tokens,
            duration_ms=(agg_a.mean_inference_duration_ms if num_trials > 1 else rep_a.latency.inference_duration_ms),
            prompt_eval_duration_ms=rep_a.latency.prompt_eval_duration_ms,
            latency_breakdown=rep_a.latency,
            status="completed",
            done_reason=rep_a.done_reason,
            effective_settings=settings_a,
            trials=trials_a,
            aggregate_latency=agg_a,
            native_telemetry={"trials_count": num_trials, "first_call": rep_a.latency.model_dump()},
        )

        exec_b = ComparisonExecutionTelemetry(
            prompt=optimized_prompt,
            response=rep_b.response,
            model=pinned_model,
            provider="ollama",
            prompt_tokens=rep_b.prompt_tokens,
            output_tokens=rep_b.output_tokens,
            total_tokens=rep_b.total_tokens,
            duration_ms=(agg_b.mean_inference_duration_ms if num_trials > 1 else rep_b.latency.inference_duration_ms),
            prompt_eval_duration_ms=rep_b.latency.prompt_eval_duration_ms,
            latency_breakdown=rep_b.latency,
            status="completed",
            done_reason=rep_b.done_reason,
            effective_settings=settings_b,
            trials=trials_b,
            aggregate_latency=agg_b,
            native_telemetry={"trials_count": num_trials, "first_call": rep_b.latency.model_dump()},
        )

        # 8. Calculate Metrics
        metrics = cls.calculate_metrics(
            exec_a,
            exec_b,
            num_trials=num_trials,
            generation_parameters_identical=settings_identical,
            warmup_succeeded=warmup_succeeded,
            warmup_error=warmup_error,
            warmup_attempted=warmup_attempted,
        )

        combined_trials: List[Dict[str, Any]] = []
        for i in range(num_trials):
            ta = trials_a[i]
            tb = trials_b[i]
            lat_a_ms = ta.latency.inference_duration_ms
            lat_b_ms = tb.latency.inference_duration_ms
            delta_ms = (
                round(lat_b_ms - lat_a_ms, 2)
                if (lat_a_ms is not None and lat_b_ms is not None)
                else None
            )
            combined_trials.append({
                "trial_index": i + 1,
                "execution_order": "A -> B" if i % 2 == 0 else "B -> A",
                "original_duration_ms": lat_a_ms,
                "optimized_duration_ms": lat_b_ms,
                "latency_delta_ms": delta_ms,
                "original_prompt_eval_ms": ta.latency.prompt_eval_duration_ms,
                "optimized_prompt_eval_ms": tb.latency.prompt_eval_duration_ms,
            })

        return ComparisonResponse(
            original_execution=exec_a,
            optimized_execution=exec_b,
            metrics=metrics,
            optimizer_details=opt_res,
            downstream_model=pinned_model,
            effective_settings=settings_a,
            trials=num_trials,
            warmup=warmup_succeeded,
            warmup_attempted=warmup_attempted,
            warmup_succeeded=warmup_succeeded,
            warmup_error=warmup_error,
            trial_measurements=combined_trials,
            aggregate_latency={
                "original": agg_a.model_dump() if agg_a else None,
                "optimized": agg_b.model_dump() if agg_b else None,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
