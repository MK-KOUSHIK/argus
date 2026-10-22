"""Candidate Validation Module (Phase 3.1 & 3.2 - Stage D).

Validates candidate rewrites from the local LLM against:
- Semantic preservation of primary actions and deliverables
- Conversational turn reversals (asking the user for data instead of directing an LLM)
- Explicit constraints (prohibitions, filenames, exact keys)
- Unsupported format hallucination (unrequested mandatory JSON)
- Self-repeating meta-instructions and empty outputs
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.schemas.optimizer import ValidationStatus
from app.services.task_analyzer import RequiredOperation, TaskAnalysisResult, TaskAnalyzer


@dataclass
class ValidationResult:
    """Outcome of validating a candidate prompt rewrite."""

    is_valid: bool
    status: ValidationStatus
    reason: Optional[str] = None
    missing_constraints: List[str] = field(default_factory=list)
    hallucinated_requirements: List[str] = field(default_factory=list)


class CandidateValidator:
    """Deterministic validation guardrails for semantic preservation."""

    META_INSTRUCTION_LEAKS = [
        r"\byou are argus\b",
        r"\bprompt optimization engine\b",
        r"\bcritical rules:\b",
        r"\bpreserve explicit requirements\b",
        r"\boriginal prompt to optimize\b",
        r"\breturn only the rewritten prompt\b",
        r"\bstrict rules:\b",
    ]

    USER_DATA_REQUEST_PATTERNS = [
        r"\b(?:please\s+)?(?:provide|share|upload|send|enter|input|paste)\s+(?:me\s+)?(?:the|your|some)?\s*[a-zA-Z0-9_\-\s]*(?:input\s+data|raw\s+data|dataset|csv|file|input|records|data\s+for\s+analysis|data)\b",
        r"\b(?:please\s+provide|can you provide|kindly provide|could you provide)\s+(?:the\s+)?(?:data|dataset|input|file|csv|records)\b",
        r"^(?:please\s+)?(?:provide|share|upload|enter|send)\s+(?:the|your)\s+(?:data|dataset|input|file|csv|records)\b",
        r"\bwhat (?:is|are) (?:the|your)\s+[a-zA-Z0-9_\-\s]*(?:input|data|dataset|numbers|records)\b",
    ]

    SOFTWARE_TERMS = {
        "program", "script", "code", "function", "class", "method", "app",
        "application", "python", "algorithm", "implement", "runnable", "module",
        "endpoint", "query", "sql", "api", "software",
    }

    EXPLANATION_TERMS = {
        "explain", "explanation", "describe", "how", "concept", "walkthrough",
        "tutorial", "guide", "overview", "works",
    }

    @classmethod
    def validate(
        cls,
        original_prompt: str,
        candidate_prompt: str,
        analysis: TaskAnalysisResult,
    ) -> ValidationResult:
        """Validate candidate prompt against generalized semantic criteria."""
        cand = candidate_prompt.strip()
        cand_lower = cand.lower()

        # Check 1: Empty or whitespace-only
        if not cand:
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason="Candidate output is empty or whitespace-only.",
            )

        # Check 2: Instruction echo / meta-instruction leak
        for pat in cls.META_INSTRUCTION_LEAKS:
            if re.search(pat, cand_lower):
                return ValidationResult(
                    is_valid=False,
                    status="fallback",
                    reason="Candidate contains leaked meta-instructions or optimization instructions.",
                )

        # Check 3: Conversational turn reversal (asking user for inputs instead of prompting an AI)
        for pat in cls.USER_DATA_REQUEST_PATTERNS:
            if re.search(pat, cand_lower):
                return ValidationResult(
                    is_valid=False,
                    status="fallback",
                    reason="Candidate inverted the task by requesting user data instead of formulating an actionable prompt for the deliverable.",
                    missing_constraints=["Deliverable action inverted to user data request"],
                )

        # Check 4: Primary Deliverable Preservation
        if analysis.deliverable_family == "software":
            cand_has_software = any(re.search(rf"\b{re.escape(term)}\b", cand_lower) for term in cls.SOFTWARE_TERMS)
            if not cand_has_software:
                return ValidationResult(
                    is_valid=False,
                    status="fallback",
                    reason="Candidate dropped the primary software deliverable (no program, script, function, or code requested).",
                    missing_constraints=["Missing deliverable: Software code / program"],
                )

        if analysis.deliverable_family == "explanation":
            cand_has_explanation = any(re.search(rf"\b{re.escape(term)}\b", cand_lower) for term in cls.EXPLANATION_TERMS)
            if not cand_has_explanation:
                return ValidationResult(
                    is_valid=False,
                    status="fallback",
                    reason="Candidate dropped the primary explanation objective.",
                    missing_constraints=["Missing objective: Explanation"],
                )

        # Check 5: Accidental loss of filenames
        missing_files = []
        for fn in analysis.filenames:
            if fn.lower() not in cand_lower:
                missing_files.append(fn)
        if missing_files:
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason=f"Candidate dropped explicit filename(s): {', '.join(missing_files)}",
                missing_constraints=[f"Missing file: {f}" for f in missing_files],
            )

        # Check 6: Accidental loss of specific keys
        missing_keys = []
        for k in analysis.specific_keys:
            if k.lower() not in cand_lower:
                missing_keys.append(k)
        if missing_keys:
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason=f"Candidate dropped required output key(s): {', '.join(missing_keys)}",
                missing_constraints=[f"Missing key: '{k}'" for k in missing_keys],
            )

        # Check 7: Accidental loss or inversion of explicit prohibitions
        missing_prohibitions = []
        for p in analysis.prohibitions:
            # Support direct negation or coordinated lists: "without using pandas or numpy"
            neg_pat = (
                rf"\b(?:no|do not|don't|does not|doesn't|without|never|avoid)\s+"
                rf"(?:use\s+|using\s+)?(?:[a-zA-Z0-9_\-]+\s*(?:,|or|and)\s*)*{re.escape(p)}\b"
            )
            has_negation = bool(re.search(neg_pat, cand_lower))
            if not has_negation:
                missing_prohibitions.append(p)

        if missing_prohibitions:
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason=f"Candidate failed to preserve explicit prohibition against: {', '.join(missing_prohibitions)}",
                missing_constraints=[f"Prohibition: No {p}" for p in missing_prohibitions],
            )

        # Check 8: Accidental loss of required operations (Phase 3.2.1 Operation Preservation)
        missing_operations = []
        for op in analysis.required_operations:
            if not cls._is_operation_preserved(op, cand_lower):
                missing_operations.append(op)

        if missing_operations:
            missing_descs = [op.normalized_phrase for op in missing_operations]
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason=f"Candidate dropped required operation(s): {', '.join(missing_descs)}",
                missing_constraints=[f"Missing operation: {d}" for d in missing_descs],
            )

        # Check 9: Unsupported additions & hallucinations (Phase 3.2.2)
        hallucinations = []
        orig_lower = original_prompt.lower()

        # 9a. Unrequested code execution requirement
        has_orig_exec = bool(re.search(r"\b(execute|executing|run|running|test|testing|benchmark)\b", orig_lower))
        if not has_orig_exec:
            has_cand_exec = bool(
                re.search(
                    r"\b(?:execute|run)\s+(?:the\s+)?(?:script|program|code|application|query)\b|\bexecute\s+it\b|\brun\s+it\b|\bexecute\s+the\s+python\b",
                    cand_lower,
                )
            )
            if has_cand_exec:
                hallucinations.append("Unrequested code execution requirement")

        # 9b. Unrequested new file artifacts
        cand_filenames = TaskAnalyzer._extract_filenames(cand)
        orig_filenames_lower = {fn.lower() for fn in analysis.filenames}
        unrequested_files = [fn for fn in cand_filenames if fn.lower() not in orig_filenames_lower]
        if unrequested_files:
            hallucinations.append(f"Unrequested file artifact(s): {', '.join(unrequested_files)}")

        # 9c. Unrequested mandatory format hallucination (e.g. unrequested JSON)
        if analysis.format_requested is None:
            json_hallucinated = bool(
                re.search(
                    r"\b(?:return|returns|output|outputs|format|as)\s+(?:as\s+|in\s+|a\s+)?json\b|\bjson\s+(?:object|format|schema)\b",
                    cand_lower,
                )
            )
            if json_hallucinated and "json" not in orig_lower:
                hallucinations.append("Unrequested mandatory JSON format")

        if hallucinations:
            return ValidationResult(
                is_valid=False,
                status="fallback",
                reason=f"Candidate introduced unsupported requirements: {', '.join(hallucinations)}",
                hallucinated_requirements=hallucinations,
            )

        # Passed all deterministic checks
        return ValidationResult(
            is_valid=True,
            status="passed",
            reason="All explicit constraints and format boundaries verified.",
        )

    @classmethod
    def _is_operation_preserved(cls, op: RequiredOperation, cand_lower: str) -> bool:
        """Verify whether a required operation or functional step is represented in candidate."""
        # 1. Associated entities (e.g. filenames) MUST be retained
        if op.associated_entities:
            for entity in op.associated_entities:
                if entity.lower() not in cand_lower:
                    return False

        # 2. Action verb and family synonyms
        action_verb_info = TaskAnalyzer.OPERATION_VERB_FAMILIES.get(
            op.action_verb, (op.action_verb, "", {op.action_verb})
        )
        family_verbs = action_verb_info[2]
        action_found = any(re.search(rf"\b{re.escape(v)}\b", cand_lower) for v in family_verbs)

        # 3. Target concepts
        target_terms = [
            t for t in op.key_terms
            if t != op.action_verb and t not in [fn.lower() for fn in op.associated_entities]
        ]

        if not target_terms:
            # Operation was solely verb + entity (e.g. "reads input.csv")
            return bool(op.associated_entities) or action_found

        matched_terms = 0
        qualifiers = {
            "average", "mean", "highest", "lowest", "total", "sum", "count",
            "median", "highest-scoring", "top", "max", "min", "spending", "spend"
        }
        op_qualifiers = [t for t in target_terms if t in qualifiers or any(q in t for q in qualifiers)]
        qualifier_matched = False

        for term in target_terms:
            parts = term.split("-") if "-" in term else [term]
            synonyms = set(TaskAnalyzer.CONCEPT_SYNONYMS.get(term, set()))
            for p in parts:
                synonyms.update(TaskAnalyzer.CONCEPT_SYNONYMS.get(p, set()))
                synonyms.add(p)
                synonyms.add(p.rstrip("s"))
                synonyms.add(p.rstrip("ing"))
                synonyms.add(p.rstrip("ed"))

            term_found = any(
                re.search(rf"\b{re.escape(syn)}\b", cand_lower)
                for syn in synonyms
                if len(syn) > 2
            )
            if term_found:
                matched_terms += 1
                if term in op_qualifiers or any(q in term for q in qualifiers):
                    qualifier_matched = True

        # If operation had a critical qualitative/analytical modifier, it must be represented
        if op_qualifiers and not qualifier_matched:
            return False

        term_ratio = matched_terms / len(target_terms) if target_terms else 0
        if matched_terms == 0 and not action_found:
            return False

        if op_qualifiers:
            return qualifier_matched and (term_ratio >= 0.5 or action_found)

        return term_ratio >= 0.5 or (action_found and matched_terms >= 1)
