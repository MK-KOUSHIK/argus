"""Adaptive Context-Length Manager for Argus.

Computes exact context budget requirements considering system instructions, prompt
wrappers, output prediction budget, and safety headroom. Snaps selected context sizes
to stable tiers to preserve KV-cache prefix reuse across requests, while safeguarding
against GPU VRAM exhaustion and rejecting oversized prompts with HTTP 413 without silent truncation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.hardware_telemetry import HardwareTelemetryProvider


class ContextOverflowError(Exception):
    """Raised when request context budget exceeds maximum permitted context limit."""

    def __init__(self, message: str, status_code: int = 413, budget_breakdown: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.budget_breakdown = budget_breakdown or {}


class AdaptiveContextManager:
    """Manages adaptive context length calculation, tier snapping, and budget enforcement."""

    @classmethod
    def get_eligible_tiers(cls, min_ctx: Optional[int] = None, max_ctx: Optional[int] = None) -> List[int]:
        """Derive eligible context tiers between min_ctx and max_ctx inclusive.

        Tiers are discrete snapping points designed to avoid frequent KV-cache
        reallocations while strictly bounded by max_ctx.
        """
        min_val = min_ctx if min_ctx is not None else settings.OLLAMA_NUM_CTX
        max_val = max_ctx if max_ctx is not None else settings.OLLAMA_MAX_NUM_CTX

        if min_val >= max_val:
            return [max_val]

        step = 2048
        tiers = [min_val]
        current = min_val + step
        while current < max_val:
            tiers.append(current)
            current += step
        if max_val not in tiers:
            tiers.append(max_val)

        return sorted(list(set(tiers)))

    @classmethod
    def estimate_tokens(cls, text: str) -> Tuple[int, str]:
        """Estimate token footprint using conservative heuristic.

        Returns:
            Tuple of (estimated_tokens, source_label).
        """
        trimmed = text.strip()
        if not trimmed:
            return 0, "estimated_conservative_heuristic"
        heuristic = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        tokens = max(1, math.ceil(len(trimmed) / heuristic))
        return tokens, "estimated_conservative_heuristic"

    @classmethod
    def calculate_budget(
        cls,
        system_instruction: str,
        formatted_prompt: str,
        min_ctx: Optional[int] = None,
        max_ctx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculate complete required context budget and select optimal context tier."""
        min_limit = min_ctx if min_ctx is not None else settings.OLLAMA_NUM_CTX
        max_limit = max_ctx if max_ctx is not None else settings.OLLAMA_MAX_NUM_CTX

        full_input_text = f"{system_instruction}\n{formatted_prompt}"
        input_tokens, token_source = cls.estimate_tokens(full_input_text)

        reserve_tokens = settings.OLLAMA_CONTEXT_RESERVE_TOKENS
        predict_tokens = settings.OLLAMA_NUM_PREDICT
        required_context = input_tokens + reserve_tokens + predict_tokens

        eligible_tiers = cls.get_eligible_tiers(min_limit, max_limit)

        breakdown = {
            "estimated_input_tokens": input_tokens,
            "reserve_tokens": reserve_tokens,
            "predict_tokens": predict_tokens,
            "total_required_context": required_context,
            "token_estimation_source": token_source,
            "max_num_ctx_limit": max_limit,
            "min_num_ctx_limit": min_limit,
            "eligible_tiers": eligible_tiers,
        }

        # 1. Reject if required context exceeds maximum permitted context limit
        if required_context > max_limit:
            raise ContextOverflowError(
                f"Prompt and reserved output budget ({required_context} tokens) exceed "
                f"the configured maximum context limit ({max_limit} tokens). "
                "Increase OLLAMA_MAX_NUM_CTX or reduce prompt size/output budget; "
                "no input was silently truncated.",
                status_code=413,
                budget_breakdown=breakdown,
            )

        # 2. Check GPU VRAM constraints
        gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory()
        free_vram = gpu_telemetry.get("free_mb") if gpu_telemetry else None

        # Low-VRAM safeguard: When free VRAM is under 1,000 MiB, expanding context
        # to higher tiers risks CUDA out-of-memory crashes.
        low_vram_active = (free_vram is not None and free_vram < 1000)
        safe_tier_under_low_vram = eligible_tiers[0]

        if low_vram_active:
            breakdown["vram_headroom_guarded"] = True
            breakdown["free_vram_mb"] = free_vram

            # If the required budget cannot fit within the safe tier, reject cleanly with HTTP 413
            # rather than silently truncating or selecting an insufficient tier
            if required_context > safe_tier_under_low_vram:
                raise ContextOverflowError(
                    f"Prompt and reserved output budget ({required_context} tokens) exceeds the "
                    f"maximum safe context tier ({safe_tier_under_low_vram} tokens) under constrained "
                    f"GPU VRAM ({free_vram} MiB free). Request rejected to prevent GPU OOM crash; "
                    "no input was silently truncated.",
                    status_code=413,
                    budget_breakdown=breakdown,
                )

            selected_tier = safe_tier_under_low_vram
            reason = (
                f"Restricted to safe baseline tier {selected_tier} due to low available "
                f"GPU VRAM ({free_vram} MiB free)"
            )
        else:
            breakdown["vram_headroom_guarded"] = False
            # Snap to smallest eligible tier that accommodates the budget
            for tier in eligible_tiers:
                if tier >= required_context:
                    selected_tier = tier
                    break
            else:
                selected_tier = max_limit

            reason = (
                f"Snapped to stable tier {selected_tier} for KV prefix reuse "
                f"(budget: {required_context} <= {selected_tier})"
            )

        # Invariant: Never select a context tier above max_limit
        selected_tier = min(selected_tier, max_limit)

        breakdown["selected_num_ctx"] = selected_tier
        breakdown["selection_reason"] = reason

        return breakdown
