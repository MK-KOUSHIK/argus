"""Automated Quality and Constraint Evaluator for Argus Benchmark Suite (Phase 4).

Performs objective deterministic verification of:
1. Prompt-level constraint and prohibition preservation
2. LLM response constraint fidelity (e.g., zero prohibited library imports)
3. LLM response structural format compliance (e.g., strictly valid JSON with required keys)
"""

import json
import re
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class QualityEvaluation(BaseModel):
    """Detailed quality and constraint evaluation for an execution."""

    prompt_constraints_preserved: bool = Field(
        description="Whether explicit negative constraints and terms were retained in the prompt"
    )
    output_prohibitions_respected: bool = Field(
        description="Whether the LLM response strictly honored negative constraints"
    )
    output_required_terms_present: bool = Field(
        description="Whether required entities, variables, or keys appear in the response"
    )
    output_format_valid: bool = Field(
        description="Whether the response satisfies requested schema or format (JSON, code, bullets)"
    )
    quality_score: float = Field(
        description="Normalized quality score between 0.0 and 1.0"
    )
    is_passing: bool = Field(
        description="True if all critical constraints and format requirements passed"
    )
    details: List[str] = Field(
        default_factory=list,
        description="Diagnostic notes explaining checks and any violations",
    )


class BenchmarkEvaluator:
    """Evaluates prompt and output quality against test case specifications."""

    @classmethod
    def evaluate(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        llm_response: str,
        test_case: Dict[str, Any],
    ) -> QualityEvaluation:
        """Run all automated checks for a benchmark case."""
        details: List[str] = []
        prohibitions = test_case.get("prohibitions", [])
        required_terms = test_case.get("required_terms", [])
        required_format = test_case.get("required_format", "text")

        # ---------------------------------------------------------
        # Check 1: Prompt-level constraint retention
        # ---------------------------------------------------------
        prompt_constraints_ok = True
        opt_lower = optimized_prompt.lower()

        for term in required_terms:
            if term.lower() not in opt_lower:
                prompt_constraints_ok = False
                details.append(f"Prompt dropped required term: '{term}'")

        # Check prohibitions retained in prompt if original had them
        for prob in prohibitions:
            prob_lower = prob.lower()
            if prob_lower in original_prompt.lower():
                # Check for explicit prohibition phrasing
                has_prohibition_phrasing = any(
                    kw in opt_lower
                    for kw in [
                        f"no {prob_lower}",
                        f"without {prob_lower}",
                        f"not use {prob_lower}",
                        f"do not use {prob_lower}",
                        f"avoid {prob_lower}",
                        f"exclude {prob_lower}",
                    ]
                )
                if not has_prohibition_phrasing and prob_lower not in opt_lower:
                    prompt_constraints_ok = False
                    details.append(f"Prompt failed to retain negative constraint: '{prob}'")

        if prompt_constraints_ok:
            details.append("Prompt constraints: All preserved")

        # ---------------------------------------------------------
        # Check 2: LLM Output respects negative constraints
        # ---------------------------------------------------------
        output_lower = llm_response.lower()
        output_prohibitions_ok = True

        for prob in prohibitions:
            prob_lower = prob.lower()
            # If the prohibited library is actually imported or invoked in code
            import_patterns = [
                rf"import\s+{re.escape(prob_lower)}\b",
                rf"from\s+{re.escape(prob_lower)}\s+import",
                rf"\b{re.escape(prob_lower)}\.",
            ]
            for pattern in import_patterns:
                if re.search(pattern, output_lower):
                    output_prohibitions_ok = False
                    details.append(f"Violation: LLM response used prohibited '{prob}'")
                    break

        if output_prohibitions_ok and prohibitions:
            details.append("Prohibitions respected: Confirmed zero prohibited library usage")

        # ---------------------------------------------------------
        # Check 3: Required terms present in LLM response
        # ---------------------------------------------------------
        terms_ok = True
        missing_terms = []
        for term in required_terms:
            if term.lower() not in output_lower:
                terms_ok = False
                missing_terms.append(term)

        if missing_terms:
            details.append(f"Output missing expected terms: {', '.join(missing_terms)}")
        else:
            details.append("Expected terms: All present in output")

        # ---------------------------------------------------------
        # Check 4: Output format compliance
        # ---------------------------------------------------------
        format_ok = True

        if required_format == "json":
            # Attempt to find and parse JSON block or raw JSON
            json_text = llm_response.strip()
            # If response wrapped in markdown block ```json ... ```
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_response)
            if match:
                json_text = match.group(1).strip()

            try:
                parsed = json.loads(json_text)
                if not isinstance(parsed, dict):
                    format_ok = False
                    details.append("Format check: JSON output is not an object/dict")
                else:
                    # Check required keys
                    missing_keys = [k for k in required_terms if k not in parsed]
                    if missing_keys:
                        format_ok = False
                        details.append(f"JSON schema missing required keys: {missing_keys}")
                    else:
                        details.append("Format check: Valid JSON with all required keys verified")
            except Exception as exc:
                format_ok = False
                details.append(f"Format check failed: Output does not parse as valid JSON ({str(exc)})")

        elif required_format == "code":
            has_code_syntax = (
                "```" in llm_response
                or "def " in llm_response
                or "with open" in llm_response
                or "import " in llm_response
            )
            if not has_code_syntax:
                format_ok = False
                details.append("Format check: No runnable code block detected")
            else:
                details.append("Format check: Code syntax verified")

        elif required_format == "bullets":
            has_bullets = bool(
                re.search(r"(?:^|\n)\s*(?:[-*•]|\d+[\.\)])\s+", llm_response)
            )
            if not has_bullets:
                format_ok = False
                details.append("Format check: Expected bulleted comparison list")
            else:
                details.append("Format check: Bulleted list format verified")

        # ---------------------------------------------------------
        # Aggregate Score
        # ---------------------------------------------------------
        score_components = [
            prompt_constraints_ok,
            output_prohibitions_ok,
            terms_ok,
            format_ok,
        ]
        quality_score = sum(1.0 for c in score_components if c) / len(score_components)
        is_passing = (
            prompt_constraints_ok
            and output_prohibitions_ok
            and format_ok
            and (terms_ok or len(missing_terms) <= 1)
        )

        return QualityEvaluation(
            prompt_constraints_preserved=prompt_constraints_ok,
            output_prohibitions_respected=output_prohibitions_ok,
            output_required_terms_present=terms_ok,
            output_format_valid=format_ok,
            quality_score=round(quality_score, 2),
            is_passing=is_passing,
            details=details,
        )
