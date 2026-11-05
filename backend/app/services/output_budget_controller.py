"""Output Budget Control Service for Argus (Phase 4.1).

Computes explainable generation-token budgets (num_predict) based on:
1. Task requirements (coding, summarization, extraction, analysis, etc.)
2. Explicit prompt constraints (word counts, sentence counts, format instructions)
3. User / API explicit overrides (requested_output_budget)
4. Optimization mode scaling (concise, balanced, structure)
5. Model context window limits and safe available headroom
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.adaptive_context_manager import AdaptiveContextManager, ContextOverflowError
from app.services.task_analyzer import TaskAnalysisResult, TaskAnalyzer


@dataclass
class OutputBudgetAllocation:
    """Detailed allocation outcome from the Output Budget Controller."""

    allocated_budget: int
    requested_budget: Optional[int]
    policy: str
    reason: str
    task_category: str
    candidate_budget: int
    task_confidence: Optional[float] = None
    detected_constraint: Optional[str] = None
    is_clamped_by_context: bool = False
    is_clamped_by_bounds: bool = False


class OutputBudgetController:
    """Determines dynamic output token budgets for inference requests."""

    @classmethod
    def get_task_baselines(cls) -> Dict[str, int]:
        """Return validated task budget baselines from application settings."""
        baselines = settings.OUTPUT_BUDGET_TASK_BASELINES
        if not isinstance(baselines, dict) or not baselines:
            raise ValueError("OUTPUT_BUDGET_TASK_BASELINES configuration is invalid or missing.")
        return baselines

    @classmethod
    def get_mode_scaling_factors(cls) -> Dict[str, float]:
        """Return validated mode scaling multipliers from application settings."""
        scaling = settings.OUTPUT_BUDGET_MODE_SCALING
        if not isinstance(scaling, dict) or not scaling:
            raise ValueError("OUTPUT_BUDGET_MODE_SCALING configuration is invalid or missing.")
        return scaling

    # Backwards-compatibility class attributes delegating to settings
    @property
    def TASK_BUDGET_BASELINES(self) -> Dict[str, int]:  # type: ignore[override]
        return self.get_task_baselines()

    @property
    def MODE_SCALING_FACTORS(self) -> Dict[str, float]:  # type: ignore[override]
        return self.get_mode_scaling_factors()

    # Regex patterns for explicit length constraints in prompts
    WORD_COUNT_PATTERNS = [
        re.compile(r"(?:under|less than|fewer than|within|maximum of|at most|up to|in)\s+(\d{1,4})\s+words?", re.IGNORECASE),
        re.compile(r"(\d{1,4})\s+words?\s+or\s+(?:less|fewer)", re.IGNORECASE),
        re.compile(r"limit(?:\s+to)?\s+(\d{1,4})\s+words?", re.IGNORECASE),
        re.compile(r"around\s+(\d{1,4})\s+words?", re.IGNORECASE),
    ]

    SENTENCE_COUNT_PATTERNS = [
        re.compile(r"(?:in|within)\s+(one|1)\s+sentence", re.IGNORECASE),
        re.compile(r"(?:single|one)\s+sentence", re.IGNORECASE),
        re.compile(r"(?:in|within)\s+(\d{1,2})\s+sentences?", re.IGNORECASE),
    ]

    PARAGRAPH_COUNT_PATTERNS = [
        re.compile(r"(?:in|within)\s+(one|1|single)\s+paragraph", re.IGNORECASE),
        re.compile(r"(?:in|within)\s+(\d{1,2})\s+paragraphs?", re.IGNORECASE),
    ]

    BREVITY_CUES = [
        (re.compile(r"\b(?:tl;?dr|briefly|in brief|short summary|quick overview|one-liner)\b", re.IGNORECASE), 160),
        (re.compile(r"\b(?:bullet points?|bulleted list|concise list)\b", re.IGNORECASE), 224),
    ]

    @classmethod
    def detect_explicit_length_constraint(
        cls, prompt: str, is_conservative: bool = False
    ) -> Optional[Tuple[int, str]]:
        """Extract explicit length directive (word/sentence/paragraph count) from prompt text.

        Returns:
            Tuple of (estimated_token_requirement, description) or None.
        """
        # Conservative policy applies strict constraint tokens without adding padding headroom
        safety_headroom = 0 if is_conservative else settings.OUTPUT_BUDGET_SAFETY_HEADROOM

        # 1. Check word count limits
        for pattern in cls.WORD_COUNT_PATTERNS:
            match = pattern.search(prompt)
            if match:
                words = int(match.group(1))
                if 1 <= words <= 4000:
                    # Conservative word-to-token ratio: ~1.33 tokens per word + safety headroom
                    tokens = max(
                        settings.OUTPUT_BUDGET_MIN_TOKENS,
                        math.ceil(words * 1.33) + safety_headroom,
                    )
                    desc = f"Explicit word limit '{match.group(0)}' -> ~{tokens} tokens"
                    return tokens, desc

        # 2. Check sentence count limits
        for pattern in cls.SENTENCE_COUNT_PATTERNS:
            match = pattern.search(prompt)
            if match:
                count_str = match.group(1).lower()
                count = 1 if count_str in ("one", "1") else int(count_str)
                # ~35 tokens per sentence + safety headroom
                tokens = max(
                    settings.OUTPUT_BUDGET_MIN_TOKENS,
                    (count * 35) + safety_headroom,
                )
                desc = f"Explicit sentence limit '{match.group(0)}' -> ~{tokens} tokens"
                return tokens, desc

        # 3. Check paragraph count limits
        for pattern in cls.PARAGRAPH_COUNT_PATTERNS:
            match = pattern.search(prompt)
            if match:
                count_str = match.group(1).lower()
                count = 1 if count_str in ("one", "1", "single") else int(count_str)
                # ~120 tokens per paragraph + safety headroom
                tokens = max(
                    settings.OUTPUT_BUDGET_MIN_TOKENS,
                    (count * 120) + safety_headroom,
                )
                desc = f"Explicit paragraph limit '{match.group(0)}' -> ~{tokens} tokens"
                return tokens, desc

        # 4. Check qualitative brevity cues
        for pattern, cue_tokens in cls.BREVITY_CUES:
            match = pattern.search(prompt)
            if match:
                target_tokens = max(settings.OUTPUT_BUDGET_MIN_TOKENS, int(cue_tokens * 0.75) if is_conservative else cue_tokens)
                desc = f"Brevity directive '{match.group(0)}' -> ~{target_tokens} tokens"
                return target_tokens, desc

        return None

    @classmethod
    def allocate_budget(
        cls,
        prompt: str,
        mode: str = "balanced",
        task_analysis: Optional[TaskAnalysisResult] = None,
        requested_output_budget: Optional[int] = None,
        system_instruction: Optional[str] = None,
        max_num_ctx: Optional[int] = None,
    ) -> OutputBudgetAllocation:
        """Calculate explainable generation token budget coordinating with context limits."""
        max_ctx = max_num_ctx if max_num_ctx is not None else settings.OLLAMA_MAX_NUM_CTX
        min_bound = settings.OUTPUT_BUDGET_MIN_TOKENS
        max_bound = settings.OUTPUT_BUDGET_MAX_TOKENS

        policy = (settings.OUTPUT_BUDGET_POLICY or "adaptive").strip().lower()
        if policy not in {"adaptive", "fixed", "conservative"}:
            raise ValueError(
                f"Unsupported OUTPUT_BUDGET_POLICY '{policy}'. Must be 'adaptive', 'fixed', or 'conservative'."
            )

        # If output budget control is disabled, return default setting
        if not settings.OUTPUT_BUDGET_ENABLED or policy == "fixed":
            default_val = min(settings.OUTPUT_BUDGET_DEFAULT_TOKENS, max_bound)
            return OutputBudgetAllocation(
                allocated_budget=default_val,
                requested_budget=requested_output_budget,
                policy="fixed",
                reason=f"Static budget configured ({default_val} tokens)",
                task_category="general",
                candidate_budget=default_val,
                task_confidence=None,
            )

        # 1. Analyze prompt if not already provided
        analysis = task_analysis or TaskAnalyzer.analyze(prompt)
        category = analysis.task_category if analysis else "general"
        task_confidence = getattr(analysis, "task_confidence", None) if analysis else None

        candidate_budget: int
        detected_constraint_desc: Optional[str] = None
        reason: str

        baselines = cls.get_task_baselines()
        scaling_factors = cls.get_mode_scaling_factors()

        if "general" not in baselines:
            raise ValueError("Configured task baselines missing 'general' fallback category.")

        is_conservative = (policy == "conservative")

        # 2. Priority 1: User / API explicit override
        if requested_output_budget is not None and requested_output_budget > 0:
            candidate_budget = requested_output_budget
            reason = f"User explicit override ({requested_output_budget} tokens)"
            detected_constraint_desc = f"API override: {requested_output_budget} tokens"

        else:
            # 3. Priority 2: Explicit prompt length directives
            constraint_match = cls.detect_explicit_length_constraint(prompt, is_conservative=is_conservative)
            if constraint_match:
                tokens, desc = constraint_match
                candidate_budget = tokens
                detected_constraint_desc = desc
                reason = f"{policy.capitalize()} constraint: {desc}"

            else:
                # 4. Priority 3: Task category baseline
                baseline = baselines.get(category, baselines["general"])
                multiplier = scaling_factors.get(mode, 1.0)

                if is_conservative:
                    # Conservative policy: dampened baseline (0.75x), no complexity expansion
                    scaled = baseline * 0.75 * multiplier
                    candidate_budget = int(math.ceil(scaled))
                    reason = (
                        f"Conservative baseline for '{category}' task "
                        f"({baseline} tokens dampened x0.75 and scaled x{multiplier} for {mode} mode)"
                    )
                else:
                    # Adaptive policy: full baseline, mode scaling, complexity scaling
                    scaled = baseline * multiplier
                    complexity = getattr(analysis, "complexity_score", 1)
                    req_ops_count = len(getattr(analysis, "required_operations", []))
                    if complexity >= 3 or req_ops_count >= 3:
                        scaled *= 1.2
                        complexity_note = f", scaled for complexity score {complexity} ({req_ops_count} operations)"
                    else:
                        complexity_note = ""

                    candidate_budget = int(math.ceil(scaled))
                    reason = (
                        f"Adaptive baseline for '{category}' task "
                        f"({baseline} tokens scaled x{multiplier} for {mode} mode{complexity_note})"
                    )

        # 5. Bound candidate by global configured limits
        # For conservative policy, cap upper bound at default tokens to prevent runaway headroom
        effective_max_bound = min(max_bound, settings.OUTPUT_BUDGET_DEFAULT_TOKENS) if is_conservative else max_bound
        is_clamped_by_bounds = False
        clamped_budget = candidate_budget
        if clamped_budget < min_bound:
            clamped_budget = min_bound
            is_clamped_by_bounds = True
        elif clamped_budget > effective_max_bound:
            clamped_budget = effective_max_bound
            is_clamped_by_bounds = True

        # 6. Centralized Context Window Headroom Coordination
        headroom_info = AdaptiveContextManager.calculate_headroom(
            system_instruction=system_instruction,
            prompt=prompt,
            max_num_ctx=max_ctx,
        )
        input_tokens = headroom_info["estimated_input_tokens"]
        reserve_tokens = headroom_info["reserve_tokens"]
        available_headroom = headroom_info["available_headroom"]

        # If available headroom cannot even accommodate the minimum token budget, reject with HTTP 413
        if available_headroom < min_bound:
            raise ContextOverflowError(
                f"Prompt footprint ({input_tokens} tokens) and reserved output budget ({min_bound} tokens) "
                f"exceed the configured maximum context limit ({max_ctx} tokens). "
                "Increase OLLAMA_MAX_NUM_CTX or reduce prompt size/output budget; "
                "no input was silently truncated.",
                status_code=413,
                budget_breakdown={
                    "estimated_input_tokens": input_tokens,
                    "reserve_tokens": reserve_tokens,
                    "min_output_budget": min_bound,
                    "max_num_ctx": max_ctx,
                    "available_headroom": available_headroom,
                },
            )

        is_clamped_by_context = False
        final_budget = clamped_budget
        if final_budget > available_headroom:
            final_budget = available_headroom
            is_clamped_by_context = True
            reason += f" [clamped to {available_headroom} tokens by context headroom]"

        return OutputBudgetAllocation(
            allocated_budget=final_budget,
            requested_budget=requested_output_budget,
            policy=policy,
            reason=reason,
            task_category=category,
            candidate_budget=candidate_budget,
            task_confidence=task_confidence,
            detected_constraint=detected_constraint_desc,
            is_clamped_by_context=is_clamped_by_context,
            is_clamped_by_bounds=is_clamped_by_bounds,
        )
