"""Ollama Local LLM Service with Task-Aware Optimization Pipeline (Phase 3.1, 4.1 & 5).

Coordinates Stages 0 through D with dynamic local model routing:
- Stage 0: Content-aware compression (CompressionOrchestrator)
- Stage A: Task analysis & constraint detection (TaskAnalyzer)
- Stage Routing: Local model selection based on complexity & live hardware resources (ModelRouter)
- Stage B: Strategy selection (StrategySelector) & Output Budget Control (OutputBudgetController)
- Stage C: Targeted model rewriting (PromptRewriter) with adaptive context & output budgets
- Stage D: Candidate validation & safe fallback (CandidateValidator)
"""

import asyncio
import math
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import httpx

from app.core.config import settings
from app.schemas.optimizer import (
    OptimizationMode,
    OptimizeRequest,
    OptimizeResponse,
    OllamaStatusResponse,
    OutputBudgetTelemetry,
    CompressionTelemetry,
    TokenEstimation,
)
from app.services.compression import CompressionOrchestrator, CompressionStrategy
from app.services.task_analyzer import TaskAnalyzer, TaskAnalysisResult
from app.services.strategy_selector import StrategySelector
from app.services.prompt_rewriter import PromptRewriter
from app.services.candidate_validator import CandidateValidator, ValidationResult
from app.services.concurrency_controller import concurrency_controller
from app.services.adaptive_context_manager import AdaptiveContextManager, ContextOverflowError
from app.services.output_budget_controller import OutputBudgetController, OutputBudgetAllocation
from app.services.ollama_process_manager import OllamaProcessManager
from app.services.hardware_telemetry import HardwareTelemetryProvider
from app.services.routing.models import (
    ComplexityTelemetry,
    ResourceTelemetry,
    RoutingDecisionTelemetry,
)
from app.services.routing.registry import LocalModelRegistry
from app.services.routing.router import ModelRouter


class OllamaServiceError(Exception):
    """Base exception for Ollama service operations."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OllamaConnectionError(OllamaServiceError):
    """Raised when cannot connect to Ollama daemon."""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class OllamaTimeoutError(OllamaServiceError):
    """Raised when Ollama inference times out."""

    def __init__(self, message: str):
        super().__init__(message, status_code=504)


class OllamaResponseError(OllamaServiceError):
    """Raised when Ollama returns an error or malformed response."""

    def __init__(self, message: str):
        super().__init__(message, status_code=502)


class OllamaService:
    """Core service encapsulating Ollama LLM interactions and optimization pipeline."""

    _inference_slots = concurrency_controller._semaphore

    @classmethod
    def estimate_tokens(cls, text: str) -> TokenEstimation:
        """Estimate token count based on conservative character-based heuristic.

        Frontier model benchmark standard: ~4 characters per token.
        """
        trimmed = text.strip()
        char_count = len(trimmed)
        if char_count == 0:
            return TokenEstimation(char_count=0, estimated_tokens=0)

        heuristic = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        tokens = max(1, math.ceil(char_count / heuristic))
        return TokenEstimation(char_count=char_count, estimated_tokens=tokens)

    @classmethod
    def clean_optimized_output(cls, raw_output: str, original_prompt: str) -> str:
        """Strip conversational filler, meta-announcements, and echoed system instructions."""
        output = raw_output.strip()

        # Regex patterns for conversational prefixes
        conversational_prefixes = [
            r"^Here is (?:an?|the) (?:optimized|rewritten|improved|concise|structured) (?:version|prompt)[^:\n]*:\s*\n*",
            r"^Here'?s (?:an?|the) (?:optimized|rewritten|improved|concise|structured) (?:version|prompt)[^:\n]*:\s*\n*",
            r"^Sure! Here is[^\n]*:\s*\n*",
            r"^Certainly! Here is[^\n]*:\s*\n*",
            r"^(?:Optimized|Rewritten) (?:Prompt|Version):\s*\n*",
            r"^Below is (?:the|an) (?:optimized|improved) (?:prompt|version)[^:\n]*:\s*\n*",
        ]

        for pat in conversational_prefixes:
            output = re.sub(pat, "", output, flags=re.IGNORECASE).strip()

        # If model echoed the system instruction checklist
        instruction_echo_prefixes = [
            r"^- Organize instructions into clean[^\n]*\n*",
            r"^- Clearly delineate Task[^\n]*\n*",
            r"^- Aggressively eliminate[^\n]*\n*",
            r"^- Convert passive phrasing[^\n]*\n*",
            r"^- Maintain every technical[^\n]*\n*",
        ]
        for pat in instruction_echo_prefixes:
            output = re.sub(pat, "", output, flags=re.IGNORECASE | re.MULTILINE).strip()

        # If model quoted entire output with quotes
        if (output.startswith('"') and output.endswith('"')) or (
            output.startswith("'") and output.endswith("'")
        ):
            output = output[1:-1].strip()

        # If model wrapped entire output in triple backticks and original was not code
        if (
            output.startswith("```")
            and output.endswith("```")
            and not original_prompt.strip().startswith("```")
        ):
            lines = output.splitlines()
            if len(lines) >= 2:
                output = "\n".join(lines[1:-1]).strip()

        return output

    @classmethod
    def identify_preserved_constraints(
        cls, original: str, optimized: str, mode: OptimizationMode
    ) -> List[str]:
        """Categorize preserved constraints and guardrails in the optimized prompt."""
        constraints = ["Core query intent & technical details"]

        if re.search(r"\b(role|act as|you are|persona)\b", original, re.I):
            constraints.append("Role & persona parameters")

        if re.search(r"\b(json|markdown|table|format|schema|output)\b", original, re.I):
            constraints.append("Requested output format & schema")

        if re.search(r"\b(\d+|false|true|null|none)\b", original, re.I):
            constraints.append("Numerical & discrete constraints")

        if re.search(r"\b(never|do not|don't|confidential|redact|refuse|without)\b", original, re.I):
            constraints.append("Negative constraints & prohibitions")

        if mode == "structure":
            constraints.append("Markdown structural hierarchy")

        return constraints

    @classmethod
    async def check_status(cls) -> OllamaStatusResponse:
        """Check if local Ollama server is running and configured model is installed."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        target_model = settings.OLLAMA_MODEL.lower()

        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_STATUS_TIMEOUT_SECONDS) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    return OllamaStatusResponse(
                        connected=False,
                        base_url=settings.OLLAMA_BASE_URL,
                        model=settings.OLLAMA_MODEL,
                        model_installed=False,
                        available_models=[],
                        error=f"Ollama returned HTTP {response.status_code}",
                    )

                data = response.json()
                raw_models = data.get("models", [])
                available_models = [
                    m.get("name", "") for m in raw_models if isinstance(m, dict)
                ]

                model_installed = any(
                    target_model in m.lower() or m.lower() in target_model
                    for m in available_models
                )

                error_msg = None
                if not model_installed:
                    error_msg = (
                        f"Model '{settings.OLLAMA_MODEL}' not found in Ollama. "
                        f"Run 'ollama pull {settings.OLLAMA_MODEL}' to install."
                    )

                # Prime model registry cache in background
                asyncio.create_task(LocalModelRegistry.refresh(force=True))

                return OllamaStatusResponse(
                    connected=True,
                    base_url=settings.OLLAMA_BASE_URL,
                    model=settings.OLLAMA_MODEL,
                    model_installed=model_installed,
                    available_models=available_models,
                    error=error_msg,
                    runtime_process=OllamaProcessManager.get_runtime_policy(),
                    hardware_telemetry={
                        "gpu": HardwareTelemetryProvider.get_gpu_memory(),
                        "ram": HardwareTelemetryProvider.get_system_ram(),
                    },
                )

        except httpx.ConnectError:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=(
                    f"Cannot connect to Ollama at {settings.OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running ('ollama serve')."
                ),
            )
        except httpx.TimeoutException:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=f"Connection timed out checking Ollama at {settings.OLLAMA_BASE_URL}.",
            )
        except Exception as exc:
            return OllamaStatusResponse(
                connected=False,
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                model_installed=False,
                available_models=[],
                error=f"Unexpected error checking Ollama: {str(exc)}",
            )

    @classmethod
    async def _execute_model_call(
        cls,
        system_instruction: str,
        prompt: str,
        output_budget: Optional[OutputBudgetAllocation] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, int, dict]:
        """Invoke Ollama with adaptive context budgeting, output budget control, and native telemetry."""

        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        formatted_prompt = f"Original prompt to optimize:\n\n{prompt}"
        target_model = model or settings.OLLAMA_MODEL

        target_num_predict = (
            output_budget.allocated_budget
            if output_budget is not None
            else settings.OLLAMA_NUM_PREDICT
        )

        # Adaptive Context Budgeting (snapped to stable tiers for KV prefix cache reuse)
        try:
            budget = AdaptiveContextManager.calculate_budget(
                system_instruction=system_instruction,
                formatted_prompt=formatted_prompt,
                output_budget=target_num_predict,
            )
        except ContextOverflowError as exc:
            raise OllamaServiceError(exc.message, status_code=exc.status_code) from exc

        selected_num_ctx = budget["selected_num_ctx"]
        payload = {
            "model": target_model,
            "system": system_instruction,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": settings.OLLAMA_TEMPERATURE,
                "top_p": settings.OLLAMA_TOP_P,
                "num_ctx": selected_num_ctx,
                "num_predict": target_num_predict,
            },
        }

        # Memory-Aware Concurrency Control
        async with concurrency_controller.slot() as slot_telemetry:
            start_time = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
                    response = await client.post(url, json=payload)

            except httpx.ConnectError as exc:
                raise OllamaConnectionError(
                    f"Failed to connect to Ollama service at {settings.OLLAMA_BASE_URL}. "
                    "Ensure Ollama is running on your machine ('ollama serve')."
                ) from exc

            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError(
                    f"Ollama inference timed out after {settings.OLLAMA_TIMEOUT_SECONDS}s. "
                    "The local model took too long to generate a response."
                ) from exc

            except Exception as exc:
                raise OllamaServiceError(
                    f"Error communicating with Ollama: {str(exc)}", status_code=500
                ) from exc

            duration_ms = int((time.perf_counter() - start_time) * 1000)

        if response.status_code != 200:
            try:
                err_detail = response.json().get("error", response.text)
            except Exception:
                err_detail = response.text
            raise OllamaResponseError(
                f"Ollama server returned error (HTTP {response.status_code}): {err_detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise OllamaResponseError("Ollama returned invalid JSON response") from exc

        raw_output = data.get("response")
        if not isinstance(raw_output, str):
            raise OllamaResponseError("Ollama response payload missing 'response' text field")

        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_cached_count = data.get("prompt_eval_cached_count")
        prompt_eval_duration = data.get("prompt_eval_duration")
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        load_duration = data.get("load_duration")
        done_reason = data.get("done_reason")

        # Determine if budget limit was hit strictly from native done_reason
        if done_reason == "length":
            hit_budget_limit = True
        elif done_reason == "stop":
            hit_budget_limit = False
        else:
            hit_budget_limit = None

        budget_utilization_percent = (
            round((eval_count / target_num_predict) * 100, 1)
            if isinstance(eval_count, int) and target_num_predict > 0
            else None
        )

        telemetry = {
            "model": target_model,
            "queue_wait_ms": slot_telemetry["queue_wait_ms"],
            "wall_time_ms": duration_ms,
            "estimated_input_tokens_for_budget": budget["estimated_input_tokens"],
            "token_estimation_source": budget["token_estimation_source"],
            "selected_num_ctx": selected_num_ctx,
            "context_selection_reason": budget["selection_reason"],
            "context_budget_breakdown": budget,
            "num_predict_limit": target_num_predict,
            "done_reason": done_reason,
            "hit_budget_limit": hit_budget_limit,
            "budget_utilization_percent": budget_utilization_percent,
            "output_budget_allocation": (
                {
                    "allocated_budget": output_budget.allocated_budget,
                    "requested_budget": output_budget.requested_budget,
                    "policy": output_budget.policy,
                    "reason": output_budget.reason,
                    "task_category": output_budget.task_category,
                    "candidate_budget": output_budget.candidate_budget,
                    "detected_constraint": output_budget.detected_constraint,
                    "is_clamped_by_context": output_budget.is_clamped_by_context,
                    "is_clamped_by_bounds": output_budget.is_clamped_by_bounds,
                }
                if output_budget
                else None
            ),
            "active_concurrency_capacity": slot_telemetry["active_concurrency_capacity"],
            "active_concurrency_in_flight": slot_telemetry["active_concurrency_in_flight"],
            "concurrency_mode": slot_telemetry["concurrency_mode"],
            "vram_available_mb": slot_telemetry["vram_available_mb"],
            "vram_used_mb": slot_telemetry["vram_used_mb"],
            "ram_available_mb": slot_telemetry["ram_available_mb"],
            "load_duration_ns": load_duration,
            "load_duration_ms": round(load_duration / 1e6, 2) if isinstance(load_duration, (int, float)) else None,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": prompt_eval_cached_count,
            "prompt_eval_duration_ns": prompt_eval_duration,
            "prompt_eval_duration_ms": round(prompt_eval_duration / 1e6, 2) if isinstance(prompt_eval_duration, (int, float)) else None,
            "prompt_tokens_per_second": (
                round(prompt_eval_count / (prompt_eval_duration / 1_000_000_000), 2)
                if isinstance(prompt_eval_count, (int, float))
                and isinstance(prompt_eval_duration, (int, float))
                and prompt_eval_duration > 0 else None
            ),
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration,
            "eval_duration_ms": round(eval_duration / 1e6, 2) if isinstance(eval_duration, (int, float)) else None,
            "generation_tokens_per_second": (
                round(eval_count / (eval_duration / 1_000_000_000), 2)
                if isinstance(eval_count, (int, float))
                and isinstance(eval_duration, (int, float))
                and eval_duration > 0 else None
            ),
            "native_metrics_available": any(
                data.get(key) is not None for key in (
                    "load_duration", "prompt_eval_count", "prompt_eval_duration",
                    "eval_count", "eval_duration"
                )
            ),
            "telemetry_source_attribution": {
                "model": "argus_model_router",
                "prompt_eval_count": "ollama_native",
                "prompt_eval_cached_count": "ollama_native",
                "prompt_eval_duration": "ollama_native",
                "eval_count": "ollama_native",
                "eval_duration": "ollama_native",
                "load_duration": "ollama_native",
                "done_reason": "ollama_native",
                "output_budget": "argus_output_budget_controller",
                "wall_time_ms": "argus_measurement",
                "queue_wait_ms": "argus_measurement",
                "selected_num_ctx": "argus_adaptive_budget",
                "concurrency": "argus_concurrency_controller",
                "vram": "gpu_telemetry_nvidia_smi",
                "ram": "system_telemetry_win32",
            },
        }
        return raw_output, duration_ms, telemetry

    @classmethod
    async def optimize_prompt(cls, request: OptimizeRequest) -> OptimizeResponse:
        """Task-aware prompt optimization pipeline orchestrating Stages 0 through D with dynamic routing."""
        # --- STAGE 0: Independent Content-Aware Compression (Phase 4.2) ---
        compression_res = CompressionOrchestrator.compress(
            text=request.prompt,
            enabled=request.enable_compression,
            reversible=request.compression_reversible,
        )
        working_prompt = compression_res.compressed_text

        # --- STAGE A: Understand the Prompt ---
        analysis = TaskAnalyzer.analyze(working_prompt)

        # --- STAGE ROUTING: Dynamic Local Model Selection (Phase 5) ---
        if request.enable_routing is False:
            routing_decision = await ModelRouter.route(
                prompt=working_prompt,
                mode=request.mode,
                task_analysis=analysis,
                requested_model=settings.OLLAMA_MODEL,
                requested_output_budget=request.requested_output_budget,
            )
        else:
            routing_decision = await ModelRouter.route(
                prompt=working_prompt,
                mode=request.mode,
                task_analysis=analysis,
                requested_model=request.requested_model,
                requested_output_budget=request.requested_output_budget,
            )
        effective_model = routing_decision.selected_model

        # --- STAGE B: Select Strategy ---
        strategy = StrategySelector.select_strategy(analysis, request.mode)

        total_duration_ms = 0
        inference_telemetry = []
        optimized_prompt: str
        validation_status = "passed"
        is_preserved = False

        # Build initial task-aware instruction if rewriting
        system_instruction = (
            PromptRewriter.build_system_instruction(
                analysis=analysis,
                strategy=strategy,
                mode=request.mode,
            )
            if strategy != "preserve"
            else None
        )

        # Output Budget Allocation (Phase 4.1) accounts for actual system instruction & formatted prompt
        try:
            output_allocation = OutputBudgetController.allocate_budget(
                prompt=working_prompt,
                mode=request.mode,
                task_analysis=analysis,
                requested_output_budget=request.requested_output_budget,
                system_instruction=system_instruction,
            )
        except ContextOverflowError as exc:
            raise OllamaServiceError(exc.message, status_code=exc.status_code) from exc

        # --- STAGE C: Rewrite Prompt (or PRESERVE directly) ---
        if strategy == "preserve":
            # Prompt is already concise, unambiguous, and effective
            optimized_prompt = working_prompt
            validation_status = "passed"
            is_preserved = True
            total_duration_ms = 1
        else:
            raw_candidate, call_ms, call_telemetry = await cls._execute_model_call(
                system_instruction=system_instruction or "",
                prompt=working_prompt,
                output_budget=output_allocation,
                model=effective_model,
            )
            inference_telemetry.append(call_telemetry)
            total_duration_ms += call_ms
            cleaned_candidate = cls.clean_optimized_output(raw_candidate, working_prompt)

            # --- STAGE D: Validate Candidate ---
            validation = CandidateValidator.validate(
                original_prompt=request.prompt,
                candidate_prompt=cleaned_candidate,
                analysis=analysis,
            )

            if validation.is_valid:
                optimized_prompt = cleaned_candidate
                validation_status = "passed"
                is_preserved = (optimized_prompt == request.prompt)
            else:
                # Controlled retry with targeted correction notice using the same effective model
                retry_instruction = PromptRewriter.build_system_instruction(
                    analysis=analysis,
                    strategy=strategy,
                    mode=request.mode,
                    retry_guidance=validation.reason,
                )

                try:
                    retry_raw, retry_ms, retry_telemetry = await cls._execute_model_call(
                        system_instruction=retry_instruction,
                        prompt=working_prompt,
                        output_budget=output_allocation,
                        model=effective_model,
                    )
                    inference_telemetry.append(retry_telemetry)
                    total_duration_ms += retry_ms
                    retry_cleaned = cls.clean_optimized_output(retry_raw, working_prompt)
                    retry_validation = CandidateValidator.validate(
                        original_prompt=request.prompt,
                        candidate_prompt=retry_cleaned,
                        analysis=analysis,
                    )

                    if retry_validation.is_valid:
                        optimized_prompt = retry_cleaned
                        validation_status = "passed"
                        is_preserved = (optimized_prompt == request.prompt)
                    else:
                        # Safety fallback: preserve original prompt
                        optimized_prompt = request.prompt
                        validation_status = "fallback"
                        is_preserved = True
                except Exception:
                    # If retry fails unexpectedly, preserve original prompt safely
                    optimized_prompt = request.prompt
                    validation_status = "fallback"
                    is_preserved = True

        # Ensure content-based check for is_preserved
        is_preserved = (optimized_prompt.strip() == request.prompt.strip())

        # Calculate metrics and telemetry
        original_metrics = cls.estimate_tokens(request.prompt)
        optimized_metrics = cls.estimate_tokens(optimized_prompt)

        orig_len = len(request.prompt)
        opt_len = len(optimized_prompt)

        # Consistent convention: final_character_count - original_character_count
        char_delta = opt_len - orig_len

        orig_tokens = original_metrics.estimated_tokens
        opt_tokens = optimized_metrics.estimated_tokens
        token_delta = opt_tokens - orig_tokens

        token_change_percent = (
            round((token_delta / orig_tokens) * 100)
            if orig_tokens > 0
            else 0
        )
        token_reduction_percent = max(0, -token_change_percent)

        # Do not report token savings when the final prompt is longer
        token_savings = max(0, orig_tokens - opt_tokens)

        # Determine length result
        if char_delta < 0:
            length_result = "compressed"
        elif char_delta > 0:
            length_result = "expanded"
        else:
            length_result = "unchanged"

        # Determine outcome
        if validation_status == "fallback":
            outcome = "fallback"
        elif is_preserved:
            outcome = "unchanged_original"
        else:
            outcome = "validated_rewrite"

        # Determine applied strategy accurately
        if outcome in {"fallback", "unchanged_original"}:
            applied_strategy = "preserve"
        elif strategy == "structure":
            has_structure = bool(
                re.search(r"(?:^|\n)\s*(?:[-*•]|\d+[\.\)]|[A-Z][a-zA-Z\s]+:|\#+)\s+", optimized_prompt)
            )
            applied_strategy = "structure" if has_structure else "compress"
        elif strategy == "compress" and length_result == "expanded":
            applied_strategy = "clarify"
        else:
            applied_strategy = strategy

        preserved_constraints = cls.identify_preserved_constraints(
            request.prompt, optimized_prompt, request.mode
        )

        primary_call = inference_telemetry[0] if inference_telemetry else {}
        output_budget_telemetry = (
            OutputBudgetTelemetry(
                requested_budget=output_allocation.requested_budget,
                allocated_budget=output_allocation.allocated_budget,
                policy=output_allocation.policy,
                reason=output_allocation.reason,
                task_category=output_allocation.task_category,
                task_confidence=output_allocation.task_confidence,
                actual_generated_tokens=primary_call.get("eval_count"),
                generation_duration_ms=primary_call.get("eval_duration_ms"),
                generation_tokens_per_second=primary_call.get("generation_tokens_per_second"),
                done_reason=primary_call.get("done_reason"),
                hit_budget_limit=primary_call.get("hit_budget_limit"),
                budget_utilization_percent=primary_call.get("budget_utilization_percent"),
            )
            if output_allocation
            else None
        )

        compression_telemetry = (
            CompressionTelemetry(
                enabled=(
                    compression_res.strategy_applied != CompressionStrategy.NONE
                    and compression_res.validation_passed
                    and compression_res.tokens_reduced > 0
                ),
                detected_content_type=compression_res.content_type.value,
                confidence_score=compression_res.confidence_score,
                strategy_applied=compression_res.strategy_applied.value,
                original_tokens=compression_res.original_tokens,
                compressed_tokens=compression_res.compressed_tokens,
                tokens_reduced=compression_res.tokens_reduced,
                compression_ratio=compression_res.compression_ratio,
                reduction_percent=compression_res.reduction_percent,
                token_measurement_method=compression_res.token_measurement_method,
                validation_status=compression_res.validation_status,
                processing_time_ms=compression_res.processing_time_ms,
                restoration_reference=compression_res.restoration_reference,
                fallback_reason=compression_res.fallback_reason,
            )
            if compression_res
            else None
        )

        routing_telemetry = RoutingDecisionTelemetry(
            selected_model=routing_decision.selected_model,
            target_model_tier=routing_decision.target_model_tier.value,
            routing_status=routing_decision.routing_status.value,
            policy_applied=routing_decision.policy_applied,
            reason=routing_decision.reason,
            decision_confidence=routing_decision.decision_confidence,
            complexity=ComplexityTelemetry(
                heuristic_score=routing_decision.complexity_assessment.heuristic_score,
                task_category=routing_decision.complexity_assessment.task_category,
                task_confidence=routing_decision.complexity_assessment.task_confidence,
                category_weight=routing_decision.complexity_assessment.category_weight,
                explicit_constraints_count=routing_decision.complexity_assessment.explicit_constraints_count,
                required_operations_count=routing_decision.complexity_assessment.required_operations_count,
                input_tokens=routing_decision.complexity_assessment.input_tokens,
                context_tier=routing_decision.complexity_assessment.context_tier,
                output_budget=routing_decision.complexity_assessment.output_budget,
                has_prohibitions=routing_decision.complexity_assessment.has_prohibitions,
                has_code_or_data=routing_decision.complexity_assessment.has_code_or_data,
                complexity_factors=routing_decision.complexity_assessment.complexity_factors,
            ),
            resources=ResourceTelemetry(
                gpu_available=routing_decision.resource_assessment.gpu_available,
                vram_total_mb=routing_decision.resource_assessment.vram_total_mb,
                vram_free_mb=routing_decision.resource_assessment.vram_free_mb,
                vram_used_mb=routing_decision.resource_assessment.vram_used_mb,
                ram_total_mb=routing_decision.resource_assessment.ram_total_mb,
                ram_free_mb=routing_decision.resource_assessment.ram_free_mb,
                ram_used_mb=routing_decision.resource_assessment.ram_used_mb,
                estimated_model_vram_mb=routing_decision.resource_assessment.estimated_model_vram_mb,
                is_resource_sufficient=routing_decision.resource_assessment.is_resource_sufficient,
                resource_pressure_reason=routing_decision.resource_assessment.resource_pressure_reason,
                is_currently_loaded=routing_decision.resource_assessment.is_currently_loaded,
                concurrency_in_flight=routing_decision.resource_assessment.concurrency_in_flight,
                concurrency_capacity=routing_decision.resource_assessment.concurrency_capacity,
            ),
            routing_duration_ms=routing_decision.routing_duration_ms,
            fallback_model=routing_decision.fallback_model,
            is_fallback=routing_decision.is_fallback,
            warning=routing_decision.warning,
        )

        return OptimizeResponse(
            original_prompt=request.prompt,
            optimized_prompt=optimized_prompt,
            mode=request.mode,
            target_model=request.target_model,
            task_category=analysis.task_category,
            strategy=applied_strategy,
            applied_strategy=applied_strategy,
            outcome=outcome,
            length_result=length_result,
            validation_status=validation_status,
            is_preserved=is_preserved,
            primary_action=analysis.primary_action,
            primary_deliverable=analysis.primary_deliverable,
            explicit_constraints=analysis.explicit_constraints,
            required_operations=[op.normalized_phrase for op in analysis.required_operations],
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            token_reduction_percent=token_reduction_percent,
            token_change_percent=token_change_percent,
            token_savings=token_savings,
            char_delta=char_delta,
            preserved_constraints=preserved_constraints,
            execution_time_ms=total_duration_ms,
            model=effective_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            inference_telemetry=(
                {
                    "calls": inference_telemetry,
                    "native_metrics_available": any(
                        call["native_metrics_available"] for call in inference_telemetry
                    ),
                } if inference_telemetry else None
            ),
            output_budget=output_budget_telemetry,
            compression=compression_telemetry,
            routing=routing_telemetry,
            kv_runtime_policy=OllamaProcessManager.get_runtime_policy(),
        )
