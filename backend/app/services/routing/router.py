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


_UNSET: Any = object()


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


class ModelResourceConstraintError(ModelRouterError):
    """Raised when a model cannot be safely admitted due to live hardware resource constraints."""

    def __init__(self, model_name: str, reason: str):
        msg = f"Model '{model_name}' cannot be safely admitted due to resource constraints: {reason}"
        super().__init__(msg, status_code=400)
        self.model_name = model_name
        self.reason = reason


class NoSafeFallbackError(ModelRouterError):
    """Raised when routing fallback is required but no installed safe fallback model exists."""

    def __init__(self, target_model: str, reason: str):
        msg = f"No installed safe fallback model available when routing '{target_model}': {reason}"
        super().__init__(msg, status_code=400)
        self.target_model = target_model
        self.reason = reason


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
    def get_cpu_runtime_allowance_mb(cls, model: DiscoveredModel) -> int:
        """Retrieve the configured CPU runtime allowance in MiB for the given model tier."""
        if hasattr(settings, "get_cpu_runtime_allowance_mb"):
            return settings.get_cpu_runtime_allowance_mb(model.tier)
        tier_val = getattr(model.tier, "value", str(model.tier))
        tier_str = str(tier_val).lower()
        if tier_str == "small":
            return getattr(settings, "ROUTING_CPU_ALLOWANCE_SMALL_MB", 64)
        elif tier_str == "medium":
            return getattr(settings, "ROUTING_CPU_ALLOWANCE_MEDIUM_MB", 128)
        elif tier_str == "large":
            return getattr(settings, "ROUTING_CPU_ALLOWANCE_LARGE_MB", 256)
        return getattr(settings, "ROUTING_CPU_ALLOWANCE_DEFAULT_MB", 128)

    @classmethod
    def evaluate_resources(
        cls,
        candidate_model: Optional[DiscoveredModel],
        loaded_other_vram_mb: int = 0,
        gpu_telemetry: Any = _UNSET,
        ram_telemetry: Any = _UNSET,
        context_tier: Optional[int] = None,
    ) -> ResourceAssessment:
        """Inspect live hardware state and assess safety for candidate model execution."""
        if gpu_telemetry is _UNSET:
            gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
        if ram_telemetry is _UNSET:
            ram_telemetry = HardwareTelemetryProvider.get_system_ram()
        concurrency_capacity, _, _, _ = concurrency_controller.get_admission_capacity()
        in_flight = concurrency_controller.in_flight

        gpu_valid = (
            gpu_telemetry is not None
            and isinstance(gpu_telemetry, dict)
            and gpu_telemetry.get("source") not in (None, "unavailable")
            and gpu_telemetry.get("total_mb", 0) > 0
            and gpu_telemetry.get("free_mb") is not None
        )
        gpu_available = gpu_valid

        vram_total = int(gpu_telemetry["total_mb"]) if (gpu_valid and gpu_telemetry is not None) else None
        vram_free = int(gpu_telemetry["free_mb"]) if (gpu_valid and gpu_telemetry is not None) else None
        vram_used = int(gpu_telemetry.get("used_mb", 0)) if (gpu_valid and gpu_telemetry is not None) else None

        ram_valid = (
            ram_telemetry is not None
            and isinstance(ram_telemetry, dict)
            and ram_telemetry.get("source") not in (None, "unavailable")
            and ram_telemetry.get("total_mb", 0) > 0
            and ram_telemetry.get("free_mb") is not None
        )
        ram_total = float(ram_telemetry["total_mb"]) if (ram_valid and ram_telemetry is not None) else None
        ram_free = float(ram_telemetry["free_mb"]) if (ram_valid and ram_telemetry is not None) else None
        ram_used = float(ram_telemetry.get("used_mb", 0.0)) if (ram_valid and ram_telemetry is not None) else None

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
                is_resource_sufficient=False,
                resource_pressure_reason="No candidate model provided for resource evaluation",
                is_currently_loaded=False,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
                measured_vram_mb=None,
                weight_footprint_mb=None,
                estimated_runtime_vram_mb=None,
                vram_measurement_type="none",
            )

        is_loaded = candidate_model.is_loaded

        # Handle missing resource metadata: conservative admission rejection
        if not is_loaded and (not candidate_model.size_bytes or candidate_model.size_bytes <= 0):
            return ResourceAssessment(
                gpu_available=gpu_available,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                vram_used_mb=vram_used,
                ram_total_mb=ram_total,
                ram_free_mb=ram_free,
                ram_used_mb=ram_used,
                estimated_model_vram_mb=0,
                is_resource_sufficient=False,
                resource_pressure_reason=(
                    f"Candidate model '{candidate_model.name}' has unverified resource metadata (missing size); "
                    "conservative admission rejected."
                ),
                is_currently_loaded=False,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
                measured_vram_mb=None,
                weight_footprint_mb=0,
                estimated_runtime_vram_mb=0,
                vram_measurement_type="unverified",
            )

        # Explicit handling for unavailable hardware telemetry
        # 1. Both GPU and RAM telemetry unavailable
        if not gpu_valid and not ram_valid:
            runtime_req = candidate_model.estimate_runtime_vram_mb(context_tier=context_tier)
            return ResourceAssessment(
                gpu_available=False,
                vram_total_mb=None,
                vram_free_mb=None,
                vram_used_mb=None,
                ram_total_mb=None,
                ram_free_mb=None,
                ram_used_mb=None,
                estimated_model_vram_mb=runtime_req,
                is_resource_sufficient=False,
                resource_pressure_reason=(
                    f"Hardware telemetry unavailable: Both GPU and host RAM telemetry are missing; "
                    f"resource safety cannot be verified for '{candidate_model.name}'."
                ),
                is_currently_loaded=is_loaded,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
                measured_vram_mb=candidate_model.measured_vram_mb,
                weight_footprint_mb=round(candidate_model.weight_size_mb),
                estimated_runtime_vram_mb=runtime_req,
                vram_measurement_type="unverified",
            )

        # 2. RAM telemetry unavailable (even if GPU is available)
        if not ram_valid:
            runtime_req = candidate_model.estimate_runtime_vram_mb(context_tier=context_tier)
            return ResourceAssessment(
                gpu_available=gpu_available,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                vram_used_mb=vram_used,
                ram_total_mb=None,
                ram_free_mb=None,
                ram_used_mb=None,
                estimated_model_vram_mb=runtime_req,
                is_resource_sufficient=False,
                resource_pressure_reason=(
                    f"Host RAM telemetry is unavailable; system memory safety cannot be established for '{candidate_model.name}'."
                ),
                is_currently_loaded=is_loaded,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
                measured_vram_mb=candidate_model.measured_vram_mb,
                weight_footprint_mb=round(candidate_model.weight_size_mb),
                estimated_runtime_vram_mb=runtime_req,
                vram_measurement_type="unverified",
            )

        # 3. GPU telemetry unavailable, but RAM is valid -> CPU-only execution policy
        if not gpu_valid:
            allow_cpu = getattr(settings, "ROUTING_ALLOW_CPU_EXECUTION", True)
            weight_mb = round(candidate_model.weight_size_mb)
            kv_mb, kv_method = candidate_model.estimate_kv_cache_mb(context_tier=context_tier)
            cpu_allowance_mb = cls.get_cpu_runtime_allowance_mb(candidate_model)
            total_estimated_cpu_req_mb = weight_mb + kv_mb + cpu_allowance_mb

            if not allow_cpu:
                return ResourceAssessment(
                    gpu_available=False,
                    vram_total_mb=None,
                    vram_free_mb=None,
                    vram_used_mb=None,
                    ram_total_mb=ram_total,
                    ram_free_mb=ram_free,
                    ram_used_mb=ram_used,
                    estimated_model_vram_mb=total_estimated_cpu_req_mb,
                    is_resource_sufficient=False,
                    resource_pressure_reason=(
                        "GPU acceleration is unavailable, and CPU-only execution is disabled by ROUTING_ALLOW_CPU_EXECUTION policy."
                    ),
                    is_currently_loaded=is_loaded,
                    concurrency_in_flight=in_flight,
                    concurrency_capacity=concurrency_capacity,
                    measured_vram_mb=None,
                    weight_footprint_mb=weight_mb,
                    estimated_runtime_vram_mb=total_estimated_cpu_req_mb,
                    vram_measurement_type="none",
                )

            # Check if required architecture metadata is missing and policy requires verified estimates
            require_arch = getattr(settings, "ROUTING_REQUIRE_ARCHITECTURE_METADATA", False)
            if require_arch and kv_method == "rough_proxy":
                return ResourceAssessment(
                    gpu_available=False,
                    vram_total_mb=None,
                    vram_free_mb=None,
                    vram_used_mb=None,
                    ram_total_mb=ram_total,
                    ram_free_mb=ram_free,
                    ram_used_mb=ram_used,
                    estimated_model_vram_mb=total_estimated_cpu_req_mb,
                    is_resource_sufficient=False,
                    resource_pressure_reason=(
                        f"Candidate model '{candidate_model.name}' lacks verified architecture metadata for exact CPU RAM KV-cache estimation; "
                        f"admission rejected by ROUTING_REQUIRE_ARCHITECTURE_METADATA policy."
                    ),
                    is_currently_loaded=is_loaded,
                    concurrency_in_flight=in_flight,
                    concurrency_capacity=concurrency_capacity,
                    measured_vram_mb=None,
                    weight_footprint_mb=weight_mb,
                    estimated_runtime_vram_mb=total_estimated_cpu_req_mb,
                    vram_measurement_type="cpu_only",
                )

            # CPU execution is permitted: validate host RAM headroom
            # Total CPU memory admission requirement includes:
            # - Model weights footprint (unloaded)
            # - Estimated KV-cache at target context and configured cache dtype
            # - Explicit CPU runtime scratchpad and activation allowance (configurable per tier via ROUTING_CPU_ALLOWANCE_*_MB)
            # - Host system safety margin (512-1024 MiB) to prevent OS paging/swap churn
            ram_margin = getattr(settings, "ROUTING_RAM_SAFETY_MARGIN_MB", 1024)
            if candidate_model.tier == ModelTier.SMALL:
                ram_margin = min(ram_margin, 512)

            required_ram = (kv_mb + cpu_allowance_mb + ram_margin) if is_loaded else (total_estimated_cpu_req_mb + ram_margin)
            assert ram_free is not None
            is_cpu_sufficient = ram_free >= required_ram
            cpu_pressure_reason: Optional[str] = None
            if not is_cpu_sufficient:
                if is_loaded:
                    cpu_pressure_reason = (
                        f"Insufficient host RAM for CPU-only execution of loaded model '{candidate_model.name}' "
                        f"({ram_free:.0f} MiB free < {required_ram} MiB required "
                        f"[estimated runtime headroom requirement: {kv_mb} MiB KV cache ({kv_method}) + "
                        f"{cpu_allowance_mb} MiB runtime allowance + {ram_margin} MiB safety margin; "
                        f"model weights ({weight_mb} MiB) are already loaded in RAM])"
                    )
                else:
                    cpu_pressure_reason = (
                        f"Insufficient host RAM for CPU-only execution of '{candidate_model.name}' "
                        f"({ram_free:.0f} MiB free < {required_ram} MiB required "
                        f"[estimated unloaded requirement: {weight_mb} MiB weights + {kv_mb} MiB KV cache ({kv_method}) + "
                        f"{cpu_allowance_mb} MiB runtime allowance + {ram_margin} MiB safety margin])"
                    )

            # Check unknown tier constraints
            allow_unknown = getattr(settings, "ROUTING_ALLOW_UNKNOWN_TIER", False)
            if candidate_model.tier == ModelTier.UNKNOWN and not allow_unknown:
                is_cpu_sufficient = False
                if not cpu_pressure_reason:
                    cpu_pressure_reason = (
                        f"Model '{candidate_model.name}' has UNKNOWN capacity tier (verified parameter metadata unavailable). "
                        f"Automatic admission disabled by ROUTING_ALLOW_UNKNOWN_TIER."
                    )

            return ResourceAssessment(
                gpu_available=False,
                vram_total_mb=None,
                vram_free_mb=None,
                vram_used_mb=None,
                ram_total_mb=ram_total,
                ram_free_mb=ram_free,
                ram_used_mb=ram_used,
                estimated_model_vram_mb=total_estimated_cpu_req_mb,
                is_resource_sufficient=is_cpu_sufficient,
                resource_pressure_reason=cpu_pressure_reason,
                is_currently_loaded=is_loaded,
                concurrency_in_flight=in_flight,
                concurrency_capacity=concurrency_capacity,
                measured_vram_mb=None,
                weight_footprint_mb=weight_mb,
                estimated_runtime_vram_mb=total_estimated_cpu_req_mb,
                vram_measurement_type="cpu_only",
            )

        # 4. Both GPU and RAM telemetry are valid: execute full GPU VRAM and host RAM checks
        assert ram_free is not None
        assert vram_free is not None

        weight_mb = round(candidate_model.weight_size_mb)
        if is_loaded:
            measured_vram = candidate_model.measured_vram_mb
            runtime_req = measured_vram if measured_vram is not None else candidate_model.estimate_runtime_vram_mb(context_tier=context_tier)
            vram_measurement_type = "measured_resident" if measured_vram is not None else candidate_model.get_vram_measurement_type(context_tier=context_tier)
        else:
            measured_vram = None
            runtime_req = candidate_model.estimate_runtime_vram_mb(context_tier=context_tier)
            vram_measurement_type = candidate_model.get_vram_measurement_type(context_tier=context_tier)

        vram_margin = getattr(settings, "ROUTING_VRAM_SAFETY_MARGIN_MB", 512)
        ram_margin = getattr(settings, "ROUTING_RAM_SAFETY_MARGIN_MB", 1024)
        if candidate_model.tier == ModelTier.SMALL:
            vram_margin = min(vram_margin, 128)
            ram_margin = min(ram_margin, 512)

        is_sufficient = True
        pressure_reason: Optional[str] = None

        # System RAM safety check
        if ram_free < ram_margin:
            is_sufficient = False
            pressure_reason = f"Host RAM is constrained ({ram_free:.0f} MiB free < {ram_margin} MiB margin)"

        # Check architecture metadata requirement policy
        require_arch = getattr(settings, "ROUTING_REQUIRE_ARCHITECTURE_METADATA", False)
        if require_arch and vram_measurement_type == "rough_proxy":
            is_sufficient = False
            if not pressure_reason:
                pressure_reason = (
                    f"Candidate model '{candidate_model.name}' lacks verified architecture metadata for exact KV-cache sizing; "
                    f"admission rejected by ROUTING_REQUIRE_ARCHITECTURE_METADATA policy."
                )

        # Check GPU VRAM
        if is_loaded:
            loaded_headroom = getattr(settings, "ROUTING_LOADED_HEADROOM_MARGIN_MB", 250)
            if vram_free < loaded_headroom:
                is_sufficient = False
                pressure_reason = (
                    f"Resident model '{candidate_model.name}' has insufficient runtime VRAM headroom "
                    f"({vram_free} MiB free < {loaded_headroom} MiB headroom required)"
                )
        else:
            evictable_vram = loaded_other_vram_mb if in_flight == 0 else 0
            effective_available_vram = vram_free + evictable_vram
            required_headroom = runtime_req + vram_margin

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
                    f"{required_headroom} MiB required for {candidate_model.name} "
                    f"[{vram_measurement_type}: {runtime_req}MB (weights: {weight_mb}MB) + {vram_margin}MB margin])"
                )

        # Check unknown capacity tier constraints
        allow_unknown = getattr(settings, "ROUTING_ALLOW_UNKNOWN_TIER", False)
        if candidate_model.tier == ModelTier.UNKNOWN and not allow_unknown:
            is_sufficient = False
            if not pressure_reason:
                pressure_reason = (
                    f"Model '{candidate_model.name}' has UNKNOWN capacity tier (verified parameter metadata unavailable). "
                    f"Automatic admission disabled by ROUTING_ALLOW_UNKNOWN_TIER."
                )

        return ResourceAssessment(
            gpu_available=gpu_available,
            vram_total_mb=vram_total,
            vram_free_mb=vram_free,
            vram_used_mb=vram_used,
            ram_total_mb=ram_total,
            ram_free_mb=ram_free,
            ram_used_mb=ram_used,
            estimated_model_vram_mb=runtime_req,
            is_resource_sufficient=is_sufficient,
            resource_pressure_reason=pressure_reason,
            is_currently_loaded=is_loaded,
            concurrency_in_flight=in_flight,
            concurrency_capacity=concurrency_capacity,
            measured_vram_mb=measured_vram,
            weight_footprint_mb=weight_mb,
            estimated_runtime_vram_mb=runtime_req,
            vram_measurement_type=vram_measurement_type,
        )

    @classmethod
    def rank_candidates(
        cls,
        candidates: List[DiscoveredModel],
        target_tier: ModelTier,
    ) -> List[DiscoveredModel]:
        """Rank candidate models within a tier using neutral, verified metadata.

        Vendor-independent ranking policy (no model-name substring heuristics):
        1. Loaded/Resident state: Currently loaded models rank first to minimize
           cold-start latency and avoid GPU context swapping.
        2. Parameter completeness: Models with verified parameter counts/sizes rank ahead
           of unverified models.
        3. Tier-specific parameter optimization:
           - For ModelTier.SMALL: favors lighter footprint / higher throughput (ascending parameter count).
           - For ModelTier.MEDIUM / LARGE: favors higher capacity / reasoning fidelity (descending parameter count).
        4. Quantization precision: When parameter counts match, higher precision quantization
           ranks ahead of lower bit-depths.
        5. Deterministic tie-breaker: Alphabetical order by lowercase model name.
        """
        def _quant_rank(q: Optional[str]) -> int:
            if not q:
                return 0
            u = q.upper()
            if "Q8" in u or "F16" in u:
                return 8
            if "Q6" in u:
                return 6
            if "Q5" in u:
                return 5
            if "Q4" in u:
                return 4
            if "Q3" in u:
                return 3
            if "Q2" in u:
                return 2
            return 1

        def _param_val(m: DiscoveredModel) -> float:
            return LocalModelRegistry.parse_parameter_billions(m.parameter_size, m.parameter_count) or 0.0

        def _sort_key(m: DiscoveredModel):
            loaded_priority = 1 if m.is_loaded else 0
            has_param = 1 if (m.parameter_count or m.parameter_size) else 0
            p = _param_val(m)
            param_score = -p if target_tier == ModelTier.SMALL else p
            q_score = _quant_rank(m.quantization_level)
            name_tiebreaker = m.name.lower()
            return (-loaded_priority, -has_param, -param_score, -q_score, name_tiebreaker)

        return sorted(candidates, key=_sort_key)

    @classmethod
    def find_safe_fallback(
        cls,
        all_models: List[DiscoveredModel],
        configured_fallback_name: Optional[str] = None,
        exclude_names: Optional[List[str]] = None,
        context_tier: Optional[int] = None,
        gpu_telemetry: Any = _UNSET,
        ram_telemetry: Any = _UNSET,
    ) -> Optional[DiscoveredModel]:
        """Find a verified installed model that is safe and resource-sufficient for fallback.

        Never assumes a configured default or hardcoded model is installed.
        Returns None if no verified safe fallback is available.
        """
        excluded = [n.lower() for n in (exclude_names or [])]
        candidates = [m for m in all_models if m.name.lower() not in excluded]

        # 1. First, check configured fallback model if present in installed candidates
        if configured_fallback_name:
            target = next((m for m in candidates if m.name.lower() == configured_fallback_name.lower()), None)
            if target:
                res = cls.evaluate_resources(
                    target,
                    context_tier=context_tier,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                )
                if res.is_resource_sufficient:
                    return target

        # 2. Check installed small models (verified parameter tier) ranked deterministically
        small_models = [m for m in candidates if m.tier == ModelTier.SMALL]
        ranked_small = cls.rank_candidates(small_models, ModelTier.SMALL)
        for m in ranked_small:
            res = cls.evaluate_resources(
                m,
                context_tier=context_tier,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
            )
            if res.is_resource_sufficient:
                return m

        # 3. Check any other installed model whose resources are verified sufficient
        # (excluding unknown unless ROUTING_ALLOW_UNKNOWN_TIER is True)
        allow_unknown = getattr(settings, "ROUTING_ALLOW_UNKNOWN_TIER", False)
        other_candidates = [
            m for m in candidates
            if m.tier != ModelTier.SMALL and (m.tier != ModelTier.UNKNOWN or allow_unknown)
        ]
        ranked_others = cls.rank_candidates(other_candidates, ModelTier.SMALL)
        for m in ranked_others:
            res = cls.evaluate_resources(
                m,
                context_tier=context_tier,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
            )
            if res.is_resource_sufficient:
                return m

        return None

    @classmethod
    async def route(
        cls,
        prompt: str,
        mode: str = "balanced",
        task_analysis: Optional[TaskAnalysisResult] = None,
        requested_model: Optional[str] = None,
        requested_output_budget: Optional[int] = None,
        allow_override_fallback: Optional[bool] = None,
        gpu_telemetry: Any = _UNSET,
        ram_telemetry: Any = _UNSET,
    ) -> RoutingDecision:
        """Execute deterministic model routing for the prompt optimization request."""
        t0 = time.perf_counter()

        if gpu_telemetry is _UNSET:
            gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
        if ram_telemetry is _UNSET:
            ram_telemetry = HardwareTelemetryProvider.get_system_ram()

        # Refresh local model registry
        await LocalModelRegistry.refresh()
        all_models = await LocalModelRegistry.get_all_models()
        available_names = [m.name for m in all_models]

        if not all_models:
            raise ModelRouterError(
                "No models are installed in the local Ollama daemon. "
                f"Pull at least one model before routing (e.g. 'ollama pull {settings.OLLAMA_MODEL}').",
                status_code=400,
            )

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
            if not default_model:
                raise ModelNotInstalledError(settings.OLLAMA_MODEL, available_names)
            resources = cls.evaluate_resources(
                default_model,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
            )
            if not resources.is_resource_sufficient:
                raise ModelResourceConstraintError(
                    settings.OLLAMA_MODEL,
                    resources.resource_pressure_reason or "Insufficient resources to run configured default model",
                )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            return RoutingDecision(
                selected_model=default_model.name,
                target_model_tier=default_model.tier,
                routing_status=RoutingStatus.DISABLED,
                policy_applied="disabled",
                reason=f"Automatic routing disabled. Verified configured default '{default_model.name}' is installed and safe.",
                decision_confidence=1.0,
                heuristic_margin=0.0,
                complexity_assessment=complexity,
                resource_assessment=resources,
                routing_duration_ms=elapsed_ms,
                fallback_model=default_model.name,
                is_fallback=False,
            )

        # 2. Manual Model Override
        if requested_model:
            target_model = await LocalModelRegistry.get_model(requested_model)
            if not target_model:
                raise ModelNotInstalledError(requested_model, available_names)

            resources = cls.evaluate_resources(
                target_model,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
                context_tier=context_tier,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            if not resources.is_resource_sufficient:
                permit_fallback = (
                    allow_override_fallback
                    if allow_override_fallback is not None
                    else getattr(settings, "ROUTING_ALLOW_OVERRIDE_FALLBACK", False)
                )
                if not permit_fallback:
                    raise ModelResourceConstraintError(
                        target_model.name,
                        resources.resource_pressure_reason or "Insufficient hardware resources",
                    )

                # Fallback explicitly permitted: find verified safe installed fallback
                safe_fallback = cls.find_safe_fallback(
                    all_models,
                    configured_fallback_name=fallback_model_name,
                    exclude_names=[target_model.name],
                    context_tier=context_tier,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                )
                if not safe_fallback:
                    raise NoSafeFallbackError(
                        target_model.name,
                        f"Requested model '{target_model.name}' exceeded resource limits "
                        f"({resources.resource_pressure_reason}), and no alternative safe fallback is installed.",
                    )

                fallback_res = cls.evaluate_resources(
                    safe_fallback,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                    context_tier=context_tier,
                )
                return RoutingDecision(
                    selected_model=safe_fallback.name,
                    target_model_tier=safe_fallback.tier,
                    routing_status=RoutingStatus.FALLBACK,
                    policy_applied="manual_override_fallback",
                    reason=(
                        f"Manual override to '{target_model.name}' rejected due to resource constraints "
                        f"({resources.resource_pressure_reason}). Fell back to safe installed model "
                        f"'{safe_fallback.name}' because fallback was permitted."
                    ),
                    decision_confidence=1.0,
                    heuristic_margin=0.0,
                    complexity_assessment=complexity,
                    resource_assessment=fallback_res,
                    routing_duration_ms=elapsed_ms,
                    fallback_model=safe_fallback.name,
                    is_fallback=True,
                    warning=f"Manual override model '{target_model.name}' was not admitted due to resource constraints.",
                )

            # Manual override admitted safely
            return RoutingDecision(
                selected_model=target_model.name,
                target_model_tier=target_model.tier,
                routing_status=RoutingStatus.MANUAL_OVERRIDE,
                policy_applied="manual",
                reason=f"Manual user override safely admitted for installed model '{target_model.name}'.",
                decision_confidence=1.0,
                heuristic_margin=0.0,
                complexity_assessment=complexity,
                resource_assessment=resources,
                routing_duration_ms=elapsed_ms,
                fallback_model=fallback_model_name,
                is_fallback=False,
            )

        # 3. Categorize Discovered Models by Tier
        small_models = [m for m in all_models if m.tier == ModelTier.SMALL]
        medium_models = [m for m in all_models if m.tier == ModelTier.MEDIUM]
        large_models = [m for m in all_models if m.tier == ModelTier.LARGE]
        unknown_models = [m for m in all_models if m.tier == ModelTier.UNKNOWN]

        ranked_small = cls.rank_candidates(small_models, ModelTier.SMALL)
        ranked_medium = cls.rank_candidates(medium_models, ModelTier.MEDIUM)
        ranked_large = cls.rank_candidates(large_models, ModelTier.LARGE)

        preferred_small = ranked_small[0] if ranked_small else None
        preferred_medium = (
            ranked_medium[0] if ranked_medium else (ranked_large[0] if ranked_large else None)
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
            res_assessment = cls.evaluate_resources(
                preferred_medium,
                loaded_other_vram_mb=loaded_other_vram,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
                context_tier=context_tier,
            )

            if res_assessment.is_resource_sufficient:
                selected_model = preferred_medium.name
                selected_tier = preferred_medium.tier
                reason = (
                    f"Selected higher-capacity model '{preferred_medium.name}' "
                    f"(heuristic complexity {complexity.heuristic_score:.2f} >= threshold {threshold:.2f}). "
                    f"Resources verified sufficient ({res_assessment.vram_free_mb} MiB VRAM free)."
                )
                resources = res_assessment
            else:
                # Resource-constrained fallback: must be verified safe and installed
                safe_fallback = cls.find_safe_fallback(
                    all_models,
                    configured_fallback_name=preferred_small.name if preferred_small else fallback_model_name,
                    exclude_names=[preferred_medium.name],
                    context_tier=context_tier,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                )
                if not safe_fallback:
                    raise NoSafeFallbackError(
                        preferred_medium.name,
                        f"Preferred model '{preferred_medium.name}' is resource-constrained "
                        f"({res_assessment.resource_pressure_reason}), and no verified safe fallback is installed.",
                    )

                selected_model = safe_fallback.name
                selected_tier = safe_fallback.tier
                is_fallback = True
                status = RoutingStatus.FALLBACK
                reason = (
                    f"Prompt complexity ({complexity.heuristic_score:.2f}) favored '{preferred_medium.name}', "
                    f"but resource constraints intervened: {res_assessment.resource_pressure_reason}. "
                    f"Safely fell back to installed '{safe_fallback.name}'."
                )
                resources = res_assessment

        elif wants_medium and not preferred_medium:
            # Medium capacity desired, but no medium model is installed
            safe_fallback = cls.find_safe_fallback(
                all_models,
                configured_fallback_name=fallback_model_name,
                context_tier=context_tier,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
            )
            if not safe_fallback:
                raise NoSafeFallbackError(
                    "medium_tier",
                    f"Prompt complexity ({complexity.heuristic_score:.2f}) indicates medium-tier capacity, "
                    f"but no medium model is installed and no safe fallback model is available.",
                )

            selected_model = safe_fallback.name
            selected_tier = safe_fallback.tier
            is_fallback = True
            status = RoutingStatus.FALLBACK
            reason = (
                f"Prompt complexity ({complexity.heuristic_score:.2f}) indicates medium-tier capacity, "
                f"but no medium model is installed in local Ollama. Fell back to installed '{safe_fallback.name}'."
            )
            resources = cls.evaluate_resources(
                safe_fallback,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
                context_tier=context_tier,
            )

        else:
            # Small tier requested / simple prompt
            target_small = preferred_small or cls.find_safe_fallback(
                all_models,
                configured_fallback_name=fallback_model_name,
                context_tier=context_tier,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
            )
            if not target_small:
                raise NoSafeFallbackError(
                    "small_tier",
                    "No small tier model is installed and no safe fallback model is available.",
                )

            res_assessment = cls.evaluate_resources(
                target_small,
                gpu_telemetry=gpu_telemetry,
                ram_telemetry=ram_telemetry,
                context_tier=context_tier,
            )
            if not res_assessment.is_resource_sufficient:
                alt_fallback = cls.find_safe_fallback(
                    all_models,
                    exclude_names=[target_small.name],
                    context_tier=context_tier,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                )
                if not alt_fallback:
                    raise ModelResourceConstraintError(
                        target_small.name,
                        res_assessment.resource_pressure_reason or "Insufficient resources to run small model",
                    )
                selected_model = alt_fallback.name
                selected_tier = alt_fallback.tier
                is_fallback = True
                status = RoutingStatus.FALLBACK
                reason = (
                    f"Small model '{target_small.name}' constrained ({res_assessment.resource_pressure_reason}). "
                    f"Fell back to installed '{alt_fallback.name}'."
                )
                resources = cls.evaluate_resources(
                    alt_fallback,
                    gpu_telemetry=gpu_telemetry,
                    ram_telemetry=ram_telemetry,
                    context_tier=context_tier,
                )
            else:
                selected_model = target_small.name
                selected_tier = target_small.tier
                reason = (
                    f"Selected lightweight fast model '{selected_model}' "
                    f"(heuristic complexity {complexity.heuristic_score:.2f} < threshold {threshold:.2f})."
                )
                resources = res_assessment
                resources = res_assessment

        # Decision heuristic margin: distance from threshold in auto mode
        margin = round(abs(complexity.heuristic_score - threshold), 3)
        confidence = round(min(1.0, 0.5 + margin), 2)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return RoutingDecision(
            selected_model=selected_model,
            target_model_tier=selected_tier,
            routing_status=status,
            policy_applied=policy,
            reason=reason,
            decision_confidence=confidence,
            heuristic_margin=margin,
            complexity_assessment=complexity,
            resource_assessment=resources,
            routing_duration_ms=elapsed_ms,
            fallback_model=fallback_model_name,
            is_fallback=is_fallback,
        )
