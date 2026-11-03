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

    # Stable context length tiers between OLLAMA_NUM_CTX (4096) and OLLAMA_MAX_NUM_CTX (8192)
    # Tier snapping avoids re-allocating Ollama context on every request, preserving prefix KV cache.
    CONTEXT_TIERS: List[int] = [4096, 6144, 8192]

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
    ) -> Dict[str, Any]:
        """Calculate complete required context budget and select optimal context tier."""
        full_input_text = f"{system_instruction}\n{formatted_prompt}"
        input_tokens, token_source = cls.estimate_tokens(full_input_text)

        reserve_tokens = settings.OLLAMA_CONTEXT_RESERVE_TOKENS
        predict_tokens = settings.OLLAMA_NUM_PREDICT
        required_context = input_tokens + reserve_tokens + predict_tokens

        breakdown = {
            "estimated_input_tokens": input_tokens,
            "reserve_tokens": reserve_tokens,
            "predict_tokens": predict_tokens,
            "total_required_context": required_context,
            "token_estimation_source": token_source,
            "max_num_ctx_limit": settings.OLLAMA_MAX_NUM_CTX,
            "min_num_ctx_limit": settings.OLLAMA_NUM_CTX,
        }

        # Reject if exceeding max permitted context limit
        if required_context > settings.OLLAMA_MAX_NUM_CTX:
            raise ContextOverflowError(
                f"Prompt and reserved output budget ({required_context} tokens) exceed "
                f"the configured maximum context limit ({settings.OLLAMA_MAX_NUM_CTX} tokens). "
                "Increase OLLAMA_MAX_NUM_CTX or reduce prompt size/output budget; "
                "no input was silently truncated.",
                status_code=413,
                budget_breakdown=breakdown,
            )

        # Snap to smallest stable tier that accommodates the budget
        selected_tier = settings.OLLAMA_NUM_CTX
        for tier in cls.CONTEXT_TIERS:
            if tier >= required_context:
                selected_tier = tier
                break
        else:
            selected_tier = settings.OLLAMA_MAX_NUM_CTX

        # Check VRAM constraints: if free VRAM is under 1,000 MiB, avoid expanding beyond 4096
        gpu_telemetry = HardwareTelemetryProvider.get_gpu_memory()
        reason = f"Snapped to stable tier {selected_tier} for KV prefix reuse (budget: {required_context} <= {selected_tier})"

        if gpu_telemetry is not None:
            free_vram = gpu_telemetry.get("free_mb", 0)
            if free_vram < 1000 and selected_tier > settings.OLLAMA_NUM_CTX:
                if required_context <= settings.OLLAMA_NUM_CTX:
                    selected_tier = settings.OLLAMA_NUM_CTX
                    reason = f"Restricted to baseline tier {selected_tier} due to low available VRAM ({free_vram} MiB)"

        breakdown["selected_num_ctx"] = selected_tier
        breakdown["selection_reason"] = reason

        return breakdown
