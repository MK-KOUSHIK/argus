"""Strategy Selection Engine (Phase 3.1 - Stage B).

Selects an optimization strategy (compress, clarify, structure, enrich, preserve)
grounded primarily in the prompt's intrinsic characteristics, with the user's
selected profile serving as a secondary stylistic modifier.
"""

from app.schemas.optimizer import OptimizationMode, OptimizationStrategy
from app.services.task_analyzer import TaskAnalysisResult


class StrategySelector:
    """Evaluates task analysis results to assign an optimization strategy."""

    @classmethod
    def select_strategy(
        cls,
        analysis: TaskAnalysisResult,
        mode: OptimizationMode,
    ) -> OptimizationStrategy:
        """Select the appropriate strategy for the prompt.
        
        Rules:
        1. If the prompt is already concise, unambiguous, and effective -> PRESERVE.
        2. If the prompt is underspecified or vague -> CLARIFY.
        3. If the prompt exhibits circular redundancy or filler phrases -> COMPRESS.
        4. If the prompt contains multiple distinct constraints / multi-step workflow -> STRUCTURE.
        5. User profile influences style but does not override intent or force invalid strategies:
           - Profile 'structure' prefers STRUCTURE when multiple items exist.
           - Profile 'concise' prefers COMPRESS unless clarification is strictly needed.
        """
        # Rule 1: High quality / already effective prompts are preserved
        if analysis.is_already_effective:
            return "preserve"

        # Rule 2: Underspecified prompts need clarification to be actionable
        if analysis.is_underspecified:
            return "clarify"

        # Rule 3: Concise Profile — prioritizes maximum natural brevity
        if mode == "concise":
            return "compress"

        # Rule 4: Structure-Focused Profile — applies structure when multi-step complexity exists
        if mode == "structure":
            # Apply structure when multi-step operations, constraints, or distinct sentences exist
            has_multi_step = (
                len(analysis.required_operations) >= 2
                or len(analysis.explicit_constraints) >= 2
                or analysis.complexity_score >= 2
            )
            return "structure" if has_multi_step else "compress"

        # Rule 5: Balanced Profile — preserves natural flow without forcing rigid structure templates
        if analysis.has_redundancy or analysis.has_conversational_filler:
            return "compress"

        if analysis.task_category in {"research", "planning"}:
            return "clarify"

        return "compress"
