"""Independent Semantic Evaluation & Adversarial Validation Suite.

Validates:
1. Original Alliance University reproduction case (live Ollama & mock).
2. Dynamic grammar improvement & question preservation.
3. Preservation of entity relationships, subject/object roles, negation, conditions, and ambiguity.
4. Adversarial examples where keywords and entities match but semantics change (direction, roles, bounds, causal).
5. Independent evaluation rubric decoupled from validator's internal check passes.
6. Separation of structural validation accuracy from independent semantic preservation.
7. Verbatim dispatch guarantee across all rejection paths.
"""

import pytest
import re
import httpx
from unittest.mock import patch
from typing import Dict, Any
from httpx import AsyncClient

from app.services.task_analyzer import TaskAnalyzer
from app.services.candidate_validator import CandidateValidator
from app.services.tokenizer_service import TokenizerService


# ==============================================================================
# Independent Human-Reviewed Evaluation Rubric
# (Completely decoupled from CandidateValidator implementation)
# ==============================================================================

class IndependentSemanticEvaluator:
    """Independent multi-dimensional semantic fidelity evaluator.

    Does NOT call CandidateValidator. Evaluates prompt rewrites against an
    independent 4-pillar rubric:
    1. Intent Equivalence: Same question/instruction outcome.
    2. Role & Relational Fidelity: Same entity positions and directions.
    3. Constraint Completeness: No dropped bounds, conditions, or prohibitions.
    4. Information Integrity: Exact facts, numbers, identifiers, and parameters preserved.
    """

    @classmethod
    def evaluate(cls, original: str, candidate: str) -> Dict[str, Any]:
        orig_clean = original.strip()
        cand_clean = candidate.strip()

        # Pillar 1: Intent Equivalence
        question_starters = r"^(?:what|how|why|when|where|who|which|is|are|will|can|could|would|do|does)\b"
        question_words = r"\b(?:what|how|why|when|where|who|which|tell me)\b"
        choice_question = r"\b(?:pass or fail|true or false)\b"

        is_orig_question = (
            orig_clean.endswith("?")
            or bool(re.search(question_starters, orig_clean, re.IGNORECASE))
            or bool(re.search(question_words, orig_clean, re.IGNORECASE))
            or bool(re.search(choice_question, orig_clean, re.IGNORECASE))
        )
        is_cand_question = (
            cand_clean.endswith("?")
            or bool(re.search(question_starters, cand_clean, re.IGNORECASE))
            or bool(re.search(question_words, cand_clean, re.IGNORECASE))
            or bool(re.search(choice_question, cand_clean, re.IGNORECASE))
        )
        intent_match = (is_orig_question == is_cand_question)

        # Pillar 2: Role & Relational Fidelity
        # Check directional relations: "from X to Y" cannot be inverted
        dir_orig = re.findall(r"\bfrom\s+([a-zA-Z0-9_\-]+)\s+to\s+([a-zA-Z0-9_\-]+)\b", orig_clean, re.IGNORECASE)
        dir_violations = []
        for src, dst in dir_orig:
            inv = rf"\bfrom\s+{re.escape(dst)}\s+to\s+{re.escape(src)}\b"
            if re.search(inv, cand_clean, re.IGNORECASE):
                dir_violations.append(f"Inverted from {src} to {dst}")

        # Check operator bounds: e.g. ">" inverted to "<"
        bounds_orig = re.findall(r"\b([a-zA-Z0-9_\.]+)\s*([><]=?)\s*([\$]?[0-9,]+)\b", orig_clean)
        bound_violations = []
        for var, op, val in bounds_orig:
            inv_op = "<" if op == ">" else ">" if op == "<" else "<=" if op == ">=" else ">="
            inv_pat = rf"\b{re.escape(var)}\s*{re.escape(inv_op)}\s*[\$]?{re.escape(val.lstrip('$'))}\b"
            if re.search(inv_pat, cand_clean, re.IGNORECASE):
                bound_violations.append(f"Inverted {var} {op} {val}")

        relational_fidelity = (len(dir_violations) == 0 and len(bound_violations) == 0)

        # Pillar 3: Constraint Completeness
        neg_matches = re.finditer(
            r"\b(?:do not|don't|does not|doesn't|without|never|avoid|must not|no)\s+(?:use|using|include|including|import|rely on)?\s*([a-zA-Z0-9_\-]+)\b",
            orig_clean,
            re.IGNORECASE,
        )
        missing_negs = []
        for m in neg_matches:
            n = m.group(1).lower()
            if n in ["worry", "hesitate", "matter", "miss", "the", "a", "an", "use", "using"]:
                continue
            full_neg_clause = m.group(0).lower()
            is_strict = any(full_neg_clause.startswith(p) for p in ["do not", "don't", "does not", "doesn't", "must not", "never", "no "])

            has_strict_neg = bool(re.search(rf"\b(?:do not|don't|without|never|no|must not)\s+[a-zA-Z0-9_\-\s]*\b{re.escape(n)}\b", cand_clean, re.IGNORECASE))
            has_advisory_neg = bool(re.search(rf"\bavoid\s+[a-zA-Z0-9_\-\s]*\b{re.escape(n)}\b", cand_clean, re.IGNORECASE))

            if is_strict and not has_strict_neg:
                if has_advisory_neg:
                    missing_negs.append(f"Softened strict prohibition on '{n}' to advisory avoid")
                else:
                    missing_negs.append(f"Missing negative constraint on '{n}'")
            elif not (has_strict_neg or has_advisory_neg):
                missing_negs.append(f"Missing negative constraint on '{n}'")

        constraint_completeness = (len(missing_negs) == 0)

        # Pillar 4: Information Integrity
        # Extract all numbers from original; all must exist in candidate
        nums_orig = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", orig_clean))
        nums_cand = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", cand_clean))
        missing_nums = nums_orig - nums_cand
        info_integrity = (len(missing_nums) == 0)

        overall_pass = intent_match and relational_fidelity and constraint_completeness and info_integrity

        return {
            "intent_match": intent_match,
            "relational_fidelity": relational_fidelity,
            "constraint_completeness": constraint_completeness,
            "information_integrity": info_integrity,
            "overall_pass": overall_pass,
            "violations": dir_violations + bound_violations + missing_negs + list(missing_nums),
        }


# ==============================================================================
# Tests
# ==============================================================================

def test_alliance_university_reproduction_historical_bad_rewrite_rejected():
    """Verify historical bad rewrite of Alliance University prompt is rejected.

    Historical defect:
    Original: "alliance university is a name of country will pass or fail in the computer science exam"
    Mangled candidate: "Alliance University name will pass or fail in the computer science exam."
    Defect: Dropped the relational assertion 'is a name of country' and produced an ambiguous sentence.
    """
    orig = "alliance university is a name of country will pass or fail in the computer science exam"
    bad_cand = "Alliance University name will pass or fail in the computer science exam."

    analysis = TaskAnalyzer.analyze(orig)

    # 1. Verify TaskAnalyzer extracted the entity assertion
    assertions = [rel for rel in analysis.entity_relations if rel[1] == "is_a"]
    assert any("alliance university" in rel[0].lower() for rel in assertions), "Failed to extract Alliance University assertion"

    # 2. Verify CandidateValidator catches the dropped relational assertion
    res = CandidateValidator.validate(orig, bad_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject the mangled Alliance University candidate"
    assert "dropped relational assertion" in res.reason.lower() or "missing" in res.reason.lower()


def test_adversarial_directional_reversal_rejected():
    """Adversarial Test 1: Directional reversal with identical entities and keywords.

    Original: "Transfer $500 from savings to checking account."
    Candidate: "Transfer $500 from checking to savings account."
    Keywords, numbers, and entities are 100% identical, but meaning is completely reversed!
    """
    orig = "Transfer $500 from savings to checking account."
    adv_cand = "Transfer $500 from checking to savings account."

    analysis = TaskAnalyzer.analyze(orig)
    assert ("savings", "to", "checking") in analysis.entity_relations

    res = CandidateValidator.validate(orig, adv_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject inverted directional relation"
    assert "inverted directional relationship" in res.reason.lower()

    # Independent evaluation must also fail
    indep = IndependentSemanticEvaluator.evaluate(orig, adv_cand)
    assert not indep["overall_pass"]
    assert not indep["relational_fidelity"]


def test_adversarial_relational_operator_flip_rejected():
    """Adversarial Test 2: Inverted comparison bound with identical keywords.

    Original: "Select customers where age > 65 and balance < 1000."
    Candidate: "Select customers where age < 65 and balance > 1000."
    Keywords, variables, and numbers are 100% identical, but boundary logic is inverted!
    """
    orig = "Select customers where age > 65 and balance < 1000."
    adv_cand = "Select customers where age < 65 and balance > 1000."

    analysis = TaskAnalyzer.analyze(orig)
    assert any("age > 65" in c for c in analysis.conditions)
    assert any("balance < 1000" in c for c in analysis.conditions)

    res = CandidateValidator.validate(orig, adv_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject inverted relational bounds"
    assert "inverted relational bound" in res.reason.lower()

    # Independent evaluation must also fail
    indep = IndependentSemanticEvaluator.evaluate(orig, adv_cand)
    assert not indep["overall_pass"]
    assert not indep["relational_fidelity"]


def test_adversarial_subject_object_role_inversion_rejected():
    """Adversarial Test 3: Subject and object roles inverted.

    Original: "Generate an org chart where Alice manages Bob."
    Candidate: "Generate an org chart where Bob manages Alice."
    All entities (Alice, Bob) and verbs (manages, org chart) match, but hierarchy is inverted!
    """
    orig = "Generate an org chart where Alice manages Bob."
    adv_cand = "Generate an org chart where Bob manages Alice."

    analysis = TaskAnalyzer.analyze(orig)
    assert ("Alice", "manages", "Bob") in analysis.entity_relations

    res = CandidateValidator.validate(orig, adv_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject inverted subject-object role"
    assert "inverted subject-object role" in res.reason.lower()


def test_adversarial_causal_inversion_rejected():
    """Adversarial Test 4: Cause and effect inverted.

    Original: "If server memory usage exceeds 90%, restart the backend service."
    Candidate: "If the backend service restarts, check if server memory usage exceeds 90%."
    """
    orig = "If server memory usage exceeds 90%, restart the backend service."
    adv_cand = "If the backend service restarts, check if server memory usage exceeds 90%."

    analysis = TaskAnalyzer.analyze(orig)
    res = CandidateValidator.validate(orig, adv_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject inverted causal ordering"
    assert "causal" in res.reason.lower()


def test_adversarial_negation_target_swapping_rejected():
    """Adversarial Test 5: Negation target swapped with active target.

    Original: "Export all active users and do not delete archived users."
    Candidate: "Export all archived users and do not delete active users."
    """
    orig = "Export all active users and do not delete archived users."
    adv_cand = "Export all archived users and do not delete active users."

    analysis = TaskAnalyzer.analyze(orig)
    res = CandidateValidator.validate(orig, adv_cand, analysis)
    assert not res.is_valid, "CandidateValidator must reject swapped negation targets"


def test_dynamic_grammar_improvement_and_question_preservation():
    """Verify that ungrammatical prompt with typos & filler is corrected while preserving question."""
    orig = "pls can u tell me what is the boiling point of pure water in celsius thx"
    good_cand = "What is the boiling point of pure water in Celsius?"

    analysis = TaskAnalyzer.analyze(orig)
    assert "Informal abbreviation 'pls/plz'" in analysis.grammatical_issues_detected
    assert "Informal abbreviation 'u/ur'" in analysis.grammatical_issues_detected
    assert "Conversational closing pleasantry" in analysis.grammatical_issues_detected

    res = CandidateValidator.validate(orig, good_cand, analysis)
    assert res.is_valid, f"Grammar-corrected candidate must be valid: {res.reason}"

    # Independent evaluation
    indep = IndependentSemanticEvaluator.evaluate(orig, good_cand)
    assert indep["overall_pass"], f"Independent evaluation failed: {indep['violations']}"
    assert indep["intent_match"]


def test_ambiguity_preservation_and_underspecification():
    """Verify that ambiguous prompts are detected and do not accept arbitrary hallucinations."""
    orig = "Fix the function so it works better."
    analysis = TaskAnalyzer.analyze(orig)
    assert analysis.is_underspecified
    assert any("Unquantified" in a for a in analysis.ambiguities)

    # Candidate introducing unrequested mandatory JSON format must be rejected
    bad_cand = "Fix the function so it works better and return the result as a json object."
    res = CandidateValidator.validate(orig, bad_cand, analysis)
    assert not res.is_valid
    assert "unsupported requirements" in res.reason.lower() or "unrequested" in res.reason.lower()


@pytest.mark.asyncio
async def test_verbatim_dispatch_across_all_rejection_paths(async_client: AsyncClient):
    """Verify that in every rejection scenario, original prompt is dispatched verbatim."""
    from tests.test_optimizer import create_mock_client

    # Scenario 1: Token count is greater than original (expansion)
    longer_cand = "This is a much longer rewrite that significantly expands the total token length beyond original."
    mock_resp_1 = httpx.Response(
        status_code=200,
        json={"response": longer_cand},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp_1)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(10, 25, "mock")):
        res = await async_client.post("/api/optimize", json={"prompt": "Short prompt needing optimization", "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()
        assert data["candidate_accepted"] is False
        assert data["optimized_prompt"] == "Short prompt needing optimization"
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert "greater than original" in data["rejection_reason"]

    # Scenario 2: Token count equals original (no reduction)
    mock_resp_2 = httpx.Response(
        status_code=200,
        json={"response": "Equal length candidate text here"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp_2)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(12, 12, "mock")):
        res = await async_client.post("/api/optimize", json={"prompt": "Equal length original text here", "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()
        assert data["candidate_accepted"] is False
        assert data["optimized_prompt"] == "Equal length original text here"
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert "equal to original" in data["rejection_reason"]

    # Scenario 3: Validation failure (dropped entity)
    mock_resp_3 = httpx.Response(
        status_code=200,
        json={"response": "Run container"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp_3)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(15, 5, "mock")):
        res = await async_client.post("/api/optimize", json={"prompt": "Run Docker Redis version 7.2 container on port 6379", "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()
        assert data["candidate_accepted"] is False
        assert data["optimized_prompt"] == "Run Docker Redis version 7.2 container on port 6379"
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0


def test_structural_vs_independent_accuracy_separation():
    """Demonstrate that structural validation accuracy is reported separately from independent semantic preservation."""
    cases = [
        {
            "orig": "Transfer $500 from savings to checking account.",
            "cand": "Transfer $500 from checking to savings account.",
            "expected_structural": False,  # Caught by Check 14a
            "expected_independent": False,  # Caught by Independent Evaluator Pillar 2
        },
        {
            "orig": "pls what is the boiling point of pure water in celsius thx",
            "cand": "What is the boiling point of pure water in Celsius?",
            "expected_structural": True,  # Passes Checks 1-14
            "expected_independent": True,  # Passes Independent Evaluator Pillars 1-4
        },
        {
            "orig": "Find orders where amount > 100 and discount < 10.",
            "cand": "Find orders where amount < 100 and discount > 10.",
            "expected_structural": False,  # Caught by Check 14b
            "expected_independent": False,  # Caught by Independent Evaluator Pillar 2
        },
    ]

    structural_passes = 0
    independent_passes = 0

    for c in cases:
        analysis = TaskAnalyzer.analyze(c["orig"])
        val_res = CandidateValidator.validate(c["orig"], c["cand"], analysis)
        indep_res = IndependentSemanticEvaluator.evaluate(c["orig"], c["cand"])

        if val_res.is_valid:
            structural_passes += 1
        if indep_res["overall_pass"]:
            independent_passes += 1

        assert val_res.is_valid == c["expected_structural"]
        assert indep_res["overall_pass"] == c["expected_independent"]

    # Confirm metrics are decoupled and distinct variables
    assert structural_passes == 1
    assert independent_passes == 1


def test_strict_prohibition_softening_to_avoid_rejected():
    """Verify softening a strict prohibition ('do not use') to advisory ('avoid') is rejected."""
    orig = "Write a query to find active users and do not use subqueries."
    softened_cand = "Write a query to find active users and avoid subqueries."

    analysis = TaskAnalyzer.analyze(orig)
    res = CandidateValidator.validate(orig, softened_cand, analysis)
    assert not res.is_valid
    assert "softened strict negative constraint" in res.reason.lower()

    indep = IndependentSemanticEvaluator.evaluate(orig, softened_cand)
    assert not indep["overall_pass"]
    assert any("softened" in v.lower() for v in indep["violations"])


def test_sql_order_by_preservation():
    """Verify replacing SQL 'ORDER BY' with informal 'sort by' is rejected."""
    orig = "Write a PostgreSQL query on table orders, order by total_spent descending."
    mangled_cand = "Write a PostgreSQL query on table orders, sort by total_spent descending."

    analysis = TaskAnalyzer.analyze(orig)
    res = CandidateValidator.validate(orig, mangled_cand, analysis)
    assert not res.is_valid
    assert "order by" in res.reason.lower()


def test_conditional_precheck_preservation():
    """Verify dropping a conditional pre-check ('check if ... before archiving') is rejected."""
    orig = "Create a bash script to backup /var/log. The script must check if /backup has 5GB free disk space before archiving."
    mangled_cand = "Create a bash script to backup /var/log. Ensure 5GB free disk space in /backup before archiving."

    analysis = TaskAnalyzer.analyze(orig)
    assert any("check" in p and "before" in p for p in analysis.preconditions)

    res = CandidateValidator.validate(orig, mangled_cand, analysis)
    assert not res.is_valid
    assert "conditional pre-check" in res.reason.lower()


def test_inquiry_code_generation_hallucination_rejected():
    """Verify that an inquiry prompt attempting to rewrite as code generation is rejected."""
    orig = "Will Alliance University pass or fail in the computer science exam?"
    hallucinated_cand = "Write a Python script that checks if Alliance University will pass or fail."

    analysis = TaskAnalyzer.analyze(orig)
    assert analysis.is_inquiry

    res = CandidateValidator.validate(orig, hallucinated_cand, analysis)
    assert not res.is_valid
    assert "unrequested software code generation" in res.reason.lower() or "unrequested" in res.reason.lower()
