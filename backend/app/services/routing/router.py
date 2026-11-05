"""Model Router Engine for Argus (Phase 5).

Evaluates prompt complexity, structural constraints, and live hardware availability
to deterministically select the optimal local Ollama model for prompt optimization.

Core Architectural Standards:
- Local-first inference exclusively via Ollama.
- Heuristic complexity scoring is transparent and explainable (rule-based, not pseudo-ML).
- Multi-dimensional decisioning: never routes exclusively based on prompt character or token length.
- Resource-aware: verifies live GPU VRAM and host RAM before routing to larger models.
- Avoids unsafe model switching or memory exhaustion.
"""

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.adaptive_context_manager import AdaptiveContextManager
from app.services.concurrency_controller import concurrency_controller
from app.services.hardware_telemetry import HardwareTelemetryProvider
from app.services.output_budget_controller import OutputBudgetController
from app.services.routing.models import (
    ComplexityAssessment,
    DiscoveredModel,
    ModelTier,
    ResourceAssessment,
    RoutingDecision,
    RoutingStatus,
)
from app.services.routing.registry import LocalModelRegistry
from app.services.task_analyzer import TaskAnalysisResult, TaskAnalyzer


class ModelRouterError(Exception):
    """Base exception for routing failures."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotInstalledError(ModelRouterError):
    """Raised when an explicitly requested model is not installed in local Ollama."""

    def __init__(self, model_name: str, available_models: List[str]):
        msg = (
            f"Requested model '{model_name}' is not installed in local Ollama. "
            f"Available models: {available_models if available_models else 'None'}. "
            f"Install it via 'ollama pull {model_name}'."
        )
        super().__init__(msg, status_code=400)
        self.model_name = model_name
        self.available_models = available_models


class ModelRouter:
    """Core deterministic local-first Model Routing Engine."""

    # Category base difficulty weights [0.15 - 0.40]
    CATEGORY_BASE_WEIGHTS: Dict[str, float] = {
        "coding": 0.40,
        "data_analysis": 0.35,
        "planning": 0.35,
        "research": 0.25,
        "extraction": 0.25,
        "writing": 0.20,
        "summarization": 0.15,
        "general": 0.20,
    }

    @classmethod
    def evaluate_complexity(
        cls,
        prompt: str,
        task_analysis: TaskAnalysisResult,
        input_tokens: int,
        context_tier: int,
        output_budget: int,
        mode: str,
    ) -> ComplexityAssessment:
        """Compute an explainable rule-based heuristic complexity score [0.0, 1.0].

        Evaluates:
        1. Task Category Base Weight (0.15 - 0.40)
        2. Explicit Constraints Density (up to 0.30)
        3. Multi-Clause Functional Operations Count (up to 0.25)
        4. Negative Constraints / Prohibitions (0.10)
        5. Code / Structured Data Presence (0.10)
        6. Normalized Input Token Footprint (up to 0.15)
        7. Mode Sensitivity (0.05 for structure mode)
        """
        factors: List[str] = []
        score = 0.0

        # 1. Category Base Weight
        category = task_analysis.task_category or "general"
        base_cat_weight = cls.CATEGORY_BASE_WEIGHTS.get(category, 0.20)
        score += base_cat_weight
        factors.append(f"Domain category '{category}' (+{base_cat_weight:.2f})")

        # 2. Explicit Constraints (negative constraints, specific keys, filenames)
        constraints_count = len(task_analysis.explicit_constraints)
        if constraints_count > 0:
            c_weight = min(0.30, constraints_count * 0.075)
            score += c_weight
            factors.append(f"{constraints_count} explicit constraint(s) (+{c_weight:.2f})")

        # 3. Required Functional Operations
        ops_count = len(task_analysis.required_operations)
        if ops_count > 0:
            op_weight = min(0.25, ops_count * 0.06)
            score += op_weight
            factors.append(f"{ops_count} required functional operation(s) (+{op_weight:.2f})")

        # 4. Prohibitions (negative directives are particularly difficult for <3B models)
        has_prohibitions = len(task_analysis.prohibitions) > 0
        if has_prohibitions:
            score += 0.10
            factors.append("Negative prohibition directives detected (+0.10)")

        # 5. Code or Structured Data Presence
        has_code_or_data = bool(
            re.search(r"```|def\s+\w+|class\s+\w+|\bimport\s+\w+|\bSELECT\b|[{]\"|\bDataFrame\b", prompt, re.IGNORECASE)
        )
        if has_code_or_data:
            score += 0.10
            factors.append("Structured code or data syntax detected (+0.10)")

        # 6. Normalized Input Token Footprint (capped, prevents routing exclusively on length)
        token_factor = min(0.15, (input_tokens / 2048.0) * 0.15)
        if token_factor > 0.02:
            score += token_factor
            factors.append(f"Context scale ({input_tokens} input tokens, +{token_factor:.2f})")

        # 7. Mode Sensitivity
        if mode == "structure":
            score += 0.05
            factors.append("Structure-focused optimization mode (+0.05)")

        clamped_score = max(0.0, min(1.0, round(score, 3)))

        return ComplexityAssessment(
            heuristic_score=clamped_score,
            task_category=category,
            task_confidence=task_analysis.task_confidence,
            category_weight=base_cat_weight,
            explicit_constraints_count=constraints_count,
            required_operations_count=ops_count,
            input_tokens=input_tokens,
            context_tier=context_tier,
            output_budget=output_budget,
            has_prohibitions=has_prohibitions,
            has_code_or_data=has_code_or_data,
            complexity_factors=factors,
        )

    @classmethod
    def evaluate_resources(
        cls,
        candidate_model: Optional[DiscoveredModel],
        loaded_other_vram_mb: int = 0,
    ) -> ResourceAssessment:
        """Inspect live hardware state and assess safety for candidate model execution."""
        gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
        ram_telemetry = HardwareTelemetryProvider.get_system_ram()
        concurrency_capacity, _, _, _ = concurrency_controller.get_admission_capacity()
        in_flight = concurrency_controller.in_flight

        gpu_available = (
            gpu_telemetry is not None
            and gpu_telemetry.get("source") not in (None, "unavailable")
            and gpu_telemetry.get("total_mb", 0) > 0
        )

        vram_total = gpu_telemetry.get("total_mb") if (gpu_available and gpu_telemetry is not None) else None
        vram_free = gpu_telemetry.get("free_mb") if (gpu_available and gpu_telemetry is not None) else None
        vram_used = gpu_telemetry.get("used_mb") if (gpu_available and gpu_telemetry is not None) else None

        ram_valid = (
            ram_telemetry is not None
            and ram_telemetry.get("source") not in (None, "unavailable")
            and ram_telemetry.get("total_mb", 0) > 0
        )
        ram_total = ram_telemetry.get("total_mb") if (ram_valid and ram_telemetry is not None) else None
        ram_free = ram_telemetry.get("free_mb") if (ram_valid and ram_telemetry is not None) else None
        ram_used = ram_telemetry.get("used_mb") if (ram_valid and ram_telemetry is not None) else None

        if candidate_model is None:
            return ResourceAssessment(
                gpu_available=gpu_available,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                vram_used_mb=vram_used,
                ram_total_mb=ram_total,
                ram_free_mb=ram_free,
                ram_used_mb=ram_used,
                estimated_model_vram_mb=None,
                is_resource_sufficient=True,
                resource_pressure_reason=None,
                is_currently_loaded=False,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
            )

        estimated_vram = candidate_model.estimated_vram_mb
        is_loaded = candidate_model.is_loaded
        vram_margin = getattr(settings, "ROUTING_VRAM_SAFETY_MARGIN_MB", 512)
        ram_margin = getattr(settings, "ROUTING_RAM_SAFETY_MARGIN_MB", 1024)

        is_sufficient = True
        pressure_reason: Optional[str] = None

        # If system RAM is critically low (< ram_margin)
        if ram_free is not None and ram_free < ram_margin:
            is_sufficient = False
            pressure_reason = f"Host RAM is constrained ({ram_free:.0f} MiB free < {ram_margin} MiB margin)"

        # Check GPU VRAM if GPU is active
        if gpu_available and vram_free is not None:
            if is_loaded:
                # Still check minimal headroom for KV-cache expansion (>250 MiB)
                if vram_free < 250:
                    is_sufficient = False
                    pressure_reason = f"VRAM critically low for loaded model ({vram_free} MiB free)"
            else:
                # If candidate is not resident, Ollama will evict idle models if no request is in-flight
                evictable_vram = loaded_other_vram_mb if in_flight == 0 else 0
                effective_available_vram = vram_free + evictable_vram
                required_headroom = estimated_vram + vram_margin

                if in_flight > 0 and vram_free < required_headroom:
                    is_sufficient = False
                    pressure_reason = (
                        f"Active inference in flight; cannot evict resident model "
                        f"(free VRAM: {vram_free} MiB < {required_headroom} MiB required)"
                    )
                elif effective_available_vram < required_headroom:
                    is_sufficient = False
                    detail = (
                        f"[{vram_free}MB free + {evictable_vram}MB evictable]"
                        if evictable_vram > 0
                        else f"[{vram_free}MB free]"
                    )
                    pressure_reason = (
                        f"Insufficient available VRAM ({effective_available_vram} MiB available {detail} < "
                        f"{required_headroom} MiB required for {candidate_model.name} [{estimated_vram}MB + {vram_margin}MB margin])"
                    )

        return ResourceAssessment(
            gpu_available=gpu_available,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            vram_used_mb=vram_used,
            ram_total_mb=ram_total,
            ram_free_mb=ram_free,
            ram_used_mb=ram_used,
            estimated_model_vram_mb=estimated_vram,
            is_resource_sufficient=is_sufficient,
            resource_pressure_reason=pressure_reason,
            is_currently_loaded=is_loaded,
            concurrency_in_flight=in_flight,
            concurrency_capacity=concurrency_capacity,
        )

    @classmethod
    async def route(
        cls,
        prompt: str,
        mode: str = "balanced",
        task_analysis: Optional[TaskAnalysisResult] = None,
        requested_model: Optional[str] = None,
        requested_output_budget: Optional[int] = None,
    ) -> RoutingDecision:
        """Execute deterministic model routing for the prompt optimization request."""
        t0 = time.perf_counter()

        # Refresh local model registry
        await LocalModelRegistry.refresh()
        all_models = await LocalModelRegistry.get_all_models()
        available_names = [m.name for m in all_models]

        # Analyze task characteristics if not provided
        analysis = task_analysis or TaskAnalyzer.analyze(prompt)

        # Context footprint and output budget estimations
        input_tokens, _ = AdaptiveContextManager.estimate_tokens(prompt)
        output_budget = (
            requested_output_budget
            if requested_output_budget is not None
            else settings.OLLAMA_NUM_PREDICT
        )
        context_tiers = AdaptiveContextManager.get_eligible_tiers()
        context_tier = context_tiers[0] if context_tiers else settings.OLLAMA_NUM_CTX

        complexity = cls.evaluate_complexity(
            prompt=prompt,
            task_analysis=analysis,
            input_tokens=input_tokens,
            context_tier=context_tier,
            output_budget=output_budget,
            mode=mode,
        )

        fallback_model_name = getattr(settings, "ROUTING_DEFAULT_FALLBACK_MODEL", settings.OLLAMA_MODEL)
        routing_enabled = getattr(settings, "ROUTING_ENABLED", True)
        policy = getattr(settings, "ROUTING_POLICY", "auto")

        # 1. Routing Disabled by Configuration
        if not routing_enabled:
            default_model = await LocalModelRegistry.get_model(settings.OLLAMA_MODEL)
            resources = cls.evaluate_resources(default_model)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return RoutingDecision(
                selected_model=settings.OLLAMA_MODEL,
                target_model_tier=default_model.tier if default_model else ModelTier.UNKNOWN,
                routing_status=RoutingStatus.DISABLED,
                policy_applied="disabled",
                reason=f"Automatic routing disabled. Using configured default '{settings.OLLAMA_MODEL}'.",
                decision_confidence=1.0,
                complexity_assessment=complexity,
                resource_assessment=resources,
                routing_duration_ms=elapsed_ms,
                fallback_model=fallback_model_name,
                is_fallback=False,
            )

        # 2. Manual Model Override
        if requested_model:
            target_model = await LocalModelRegistry.get_model(requested_model)
            if not target_model:
                raise ModelNotInstalledError(requested_model, available_names)

            resources = cls.evaluate_resources(target_model)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            warning = None
            if not resources.is_resource_sufficient:
                warning = f"Resource constraint detected: {resources.resource_pressure_reason}. Execution may be slow."

            return RoutingDecision(
                selected_model=target_model.name,
                target_model_tier=target_model.tier,
                routing_status=RoutingStatus.MANUAL_OVERRIDE,
                policy_applied="manual",
                reason=f"Manual user override to installed model '{target_model.name}'.",
                decision_confidence=1.0,
                complexity_assessment=complexity,
                resource_assessment=resources,
                routing_duration_ms=elapsed_ms,
                fallback_model=fallback_model_name,
                is_fallback=False,
                warning=warning,
            )

        # 3. Categorize Discovered Models by Tier
        small_models = [m for m in all_models if m.tier == ModelTier.SMALL]
        medium_models = [m for m in all_models if m.tier == ModelTier.MEDIUM]
        large_models = [m for m in all_models if m.tier == ModelTier.LARGE]

        preferred_small = (
            next((m for m in small_models if "qwen" in m.name.lower()), None)
            or (small_models[0] if small_models else None)
        )
        preferred_medium = (
            next((m for m in medium_models if "qwen" in m.name.lower()), None)
            or (medium_models[0] if medium_models else None)
        )

        threshold = getattr(settings, "ROUTING_COMPLEXITY_THRESHOLD", 0.55)

        # 4. Policy-Driven Target Tier Selection
        is_complex = complexity.heuristic_score >= threshold
        wants_medium = False

        if policy == "prefer_large":
            wants_medium = True
        elif policy == "prefer_small":
            wants_medium = False
        else:  # "auto"
            wants_medium = is_complex

        # 5. Resource and Availability Validation
        selected_model: str
        selected_tier: ModelTier
        is_fallback = False
        reason: str
        status = RoutingStatus.ROUTED

        if wants_medium and preferred_medium:
            # Check evictable VRAM from other loaded models if candidate is not loaded
            loaded_other_vram = sum(
                (m.size_vram // (1024 * 1024) if m.size_vram else m.estimated_vram_mb)
                for m in all_models
                if m.is_loaded and m.name.lower() != preferred_medium.name.lower()
            )
            res_assessment = cls.evaluate_resources(preferred_medium, loaded_other_vram_mb=loaded_other_vram)

            if res_assessment.is_resource_sufficient:
                selected_model = preferred_medium.name
                selected_tier = preferred_medium.tier
                reason = (
                    f"Selected higher-capacity model '{preferred_medium.name}' "
                    f"(heuristic complexity {complexity.heuristic_score:.2f} >= threshold {threshold:.2f}). "
                    f"Resources verified sufficient ({res_assessment.vram_free_mb} MiB VRAM free)."
                )
            else:
                # Resource-constrained fallback to small model
                if preferred_small:
                    selected_model = preferred_small.name
                    selected_tier = preferred_small.tier
                    is_fallback = True
                    status = RoutingStatus.FALLBACK
                    reason = (
                        f"Prompt complexity ({complexity.heuristic_score:.2f}) favored '{preferred_medium.name}', "
                        f"but resource constraints intervened: {res_assessment.resource_pressure_reason}. "
                        f"Safely fell back to '{preferred_small.name}'."
                    )
                else:
                    selected_model = preferred_medium.name
                    selected_tier = preferred_medium.tier
                    reason = (
                        f"Selected '{preferred_medium.name}' under resource pressure "
                        f"({res_assessment.resource_pressure_reason}); no smaller model installed."
                    )
            resources = res_assessment

        elif wants_medium and not preferred_medium:
            fallback_target = preferred_small or (all_models[0] if all_models else None)
            selected_model = fallback_target.name if fallback_target else settings.OLLAMA_MODEL
            selected_tier = fallback_target.tier if fallback_target else ModelTier.SMALL
            is_fallback = True
            status = RoutingStatus.FALLBACK
            reason = (
                f"Prompt complexity ({complexity.heuristic_score:.2f}) indicates medium-tier capacity, "
                f"but no medium model is installed. Fell back to '{selected_model}'."
            )
            resources = cls.evaluate_resources(fallback_target)

        else:
            target_small = preferred_small or (all_models[0] if all_models else None)
            selected_model = target_small.name if target_small else settings.OLLAMA_MODEL
            selected_tier = target_small.tier if target_small else ModelTier.SMALL
            reason = (
                f"Selected lightweight fast model '{selected_model}' "
                f"(heuristic complexity {complexity.heuristic_score:.2f} < threshold {threshold:.2f})."
            )
            resources = cls.evaluate_resources(target_small)

        # Decision confidence heuristic: distance from threshold in auto mode
        margin = abs(complexity.heuristic_score - threshold)
        confidence = round(min(1.0, 0.5 + margin), 2)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return RoutingDecision(
            selected_model=selected_model,
            target_model_tier=selected_tier,
            routing_status=status,
            policy_applied=policy,
            reason=reason,
            decision_confidence=confidence,
            complexity_assessment=complexity,
            resource_assessment=resources,
            routing_duration_ms=elapsed_ms,
            fallback_model=fallback_model_name,
            is_fallback=is_fallback,
        )
