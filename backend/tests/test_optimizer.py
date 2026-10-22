"""Tests for prompt optimization and Ollama integration."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx
from httpx import AsyncClient

from app.services.candidate_validator import CandidateValidator
from app.services.task_analyzer import TaskAnalyzer


SAMPLE_PROMPT = (
    "Please review this Python code. I would like you to make sure to check for SQL injection "
    "and memory leaks. In addition to that, please provide the refactored code."
)

MOCK_OLLAMA_RESPONSE = {
    "model": "qwen2.5:1.5b",
    "created_at": "2026-10-16T17:00:00Z",
    "response": "Review Python code for SQL injection and memory leaks. Provide refactored code.",
    "done": True,
}


def create_mock_client(
    post_response=None,
    get_response=None,
    post_side_effect=None,
    get_side_effect=None,
):
    """Helper to mock httpx.AsyncClient inside OllamaService without intercepting test async_client."""
    mock_instance = AsyncMock()

    if post_side_effect is not None:
        mock_instance.post.side_effect = post_side_effect
    elif post_response is not None:
        mock_instance.post.return_value = post_response

    if get_side_effect is not None:
        mock_instance.get.side_effect = get_side_effect
    elif get_response is not None:
        mock_instance.get.return_value = get_response

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__.return_value = mock_instance
    mock_cls.return_value.__aexit__.return_value = None
    return mock_cls


@pytest.mark.asyncio
async def test_optimize_balanced_profile_success(async_client: AsyncClient):
    """Test successful optimization using the Balanced profile."""
    mock_resp = httpx.Response(
        status_code=200,
        json=MOCK_OLLAMA_RESPONSE,
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        payload = {
            "prompt": SAMPLE_PROMPT,
            "mode": "balanced",
            "target_model": "gpt-4o",
        }

        response = await async_client.post("/api/optimize", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["original_prompt"] == SAMPLE_PROMPT
        assert data["optimized_prompt"] == MOCK_OLLAMA_RESPONSE["response"]
        assert data["mode"] == "balanced"
        assert data["target_model"] == "gpt-4o"
        assert data["original_metrics"]["char_count"] == len(SAMPLE_PROMPT)
        assert data["optimized_metrics"]["char_count"] == len(MOCK_OLLAMA_RESPONSE["response"])
        assert data["token_savings"] > 0
        assert data["token_reduction_percent"] > 0
        assert "execution_time_ms" in data
        assert data["model"] == "qwen2.5:1.5b"


@pytest.mark.asyncio
async def test_optimize_concise_profile_success(async_client: AsyncClient):
    """Test optimization using Concise profile."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": "Audit Python code for SQLi and memory leaks. Output refactored code."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        payload = {
            "prompt": SAMPLE_PROMPT,
            "mode": "concise",
            "target_model": "claude-3-5-sonnet",
        }

        response = await async_client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "concise"
        assert data["target_model"] == "claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_optimize_structure_profile_success(async_client: AsyncClient):
    """Test optimization using Structure-focused profile."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": "### Task\nReview Python code.\n### Checks\n- SQLi\n- Leaks"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        payload = {
            "prompt": SAMPLE_PROMPT,
            "mode": "structure",
            "target_model": "gemini-1-5-pro",
        }

        response = await async_client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "structure"


@pytest.mark.asyncio
async def test_optimize_invalid_input_empty_prompt(async_client: AsyncClient):
    """Test validation rejection for empty or whitespace-only prompt."""
    # Empty string
    response1 = await async_client.post("/api/optimize", json={"prompt": ""})
    assert response1.status_code == 422

    # Whitespace only
    response2 = await async_client.post("/api/optimize", json={"prompt": "   \n\t  "})
    assert response2.status_code == 422


@pytest.mark.asyncio
async def test_optimize_ollama_unavailable(async_client: AsyncClient):
    """Test 503 error handling when Ollama daemon is unreachable."""
    mock_err = httpx.ConnectError("Connection refused")
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_side_effect=mock_err)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": SAMPLE_PROMPT, "mode": "balanced"},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "ollama" in data["error"]["message"].lower()


@pytest.mark.asyncio
async def test_optimize_ollama_timeout(async_client: AsyncClient):
    """Test 504 error handling when Ollama times out."""
    mock_err = httpx.TimeoutException("Timed out")
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_side_effect=mock_err)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": SAMPLE_PROMPT, "mode": "balanced"},
        )
        assert response.status_code == 504
        data = response.json()
        assert "error" in data
        assert "timed out" in data["error"]["message"].lower()


@pytest.mark.asyncio
async def test_optimize_ollama_malformed_response(async_client: AsyncClient):
    """Test 502 error handling when Ollama returns JSON without 'response' string."""
    mock_resp = httpx.Response(
        status_code=200,
        json={"unexpected_key": 123},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": SAMPLE_PROMPT, "mode": "balanced"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert "missing 'response'" in data["error"]["message"].lower()


@pytest.mark.asyncio
async def test_ollama_status_connected(async_client: AsyncClient):
    """Test GET /api/ollama/status when Ollama is running and model is available."""
    mock_tags = {
        "models": [
            {"name": "qwen2.5:1.5b", "size": 986061892},
            {"name": "llama3:latest", "size": 4700000000},
        ]
    }
    mock_resp = httpx.Response(
        status_code=200,
        json=mock_tags,
        request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(get_response=mock_resp)):
        response = await async_client.get("/api/ollama/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["model_installed"] is True
        assert "qwen2.5:1.5b" in data["available_models"]
        assert data["error"] is None


@pytest.mark.asyncio
async def test_ollama_status_disconnected(async_client: AsyncClient):
    """Test GET /api/ollama/status when Ollama is offline."""
    mock_err = httpx.ConnectError("Connection refused")
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(get_side_effect=mock_err)):
        response = await async_client.get("/api/ollama/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is False
        assert data["model_installed"] is False
        assert "Cannot connect" in data["error"]


# ==============================================================================
# Phase 3.1: Task-Aware Regression Tests (Cases A through H)
# ==============================================================================


@pytest.mark.asyncio
async def test_case_a_redundant_prompt_compression(async_client: AsyncClient):
    """Case A: Circular redundancy is compressed while preserving function behavior."""
    prompt = (
        "Please write a Python function that adds two numbers. "
        "I need a function that takes two numbers and returns their sum."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": "Write a Python function that takes two numbers and returns their sum."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "compress"
        assert data["task_category"] == "coding"
        assert data["token_savings"] > 0
        assert "adds two numbers" in data["optimized_prompt"] or "returns their sum" in data["optimized_prompt"]
        assert data["validation_status"] == "passed"


@pytest.mark.asyncio
async def test_case_b_explicit_constraints_preserved(async_client: AsyncClient):
    """Case B: Filenames, calculations, and negative restrictions must be strictly preserved."""
    prompt = (
        "Write a Python script that reads sales.csv, calculates monthly revenue, "
        "saves monthly_report.csv, and does not use pandas."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "response": (
                "Write a Python script to calculate monthly revenue from sales.csv "
                "and save the output to monthly_report.csv. Do not use pandas."
            )
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()
        opt = data["optimized_prompt"]

        assert "sales.csv" in opt
        assert "monthly_report.csv" in opt
        assert "monthly revenue" in opt
        assert "do not use pandas" in opt.lower()
        assert any("sales.csv" in c for c in data["explicit_constraints"])
        assert any("pandas" in c for c in data["explicit_constraints"])
        assert data["validation_status"] == "passed"


@pytest.mark.asyncio
async def test_case_c_underspecified_task_clarification(async_client: AsyncClient):
    """Case C: Underspecified task receives actionable clarification without unrequested JSON."""
    prompt = (
        "Build me a Python program to analyze student marks and tell me useful things. "
        "Make it good and don't miss anything important."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={
            "response": (
                "Create a Python program to analyze student grade data. "
                "Compute key statistics including mean, median, and grade distribution, "
                "and identify top and struggling students."
            )
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        assert data["strategy"] == "clarify"
        assert "json" not in data["optimized_prompt"].lower()
        assert data["validation_status"] == "passed"


@pytest.mark.asyncio
async def test_case_d_already_effective_prompt_preserved(async_client: AsyncClient):
    """Case D: Already effective and precise prompts are preserved without needless alteration."""
    prompt = "Return the square of the supplied integer as a JSON object with the key 'result'."

    # Notice: StrategySelector detects already effective prompt; no model call is required.
    response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
    assert response.status_code == 200
    data = response.json()

    assert data["strategy"] == "preserve"
    assert data["is_preserved"] is True
    assert data["optimized_prompt"] == prompt
    assert data["token_savings"] == 0
    assert data["token_change_percent"] == 0
    assert any("result" in c for c in data["explicit_constraints"])
    assert data["validation_status"] == "passed"


@pytest.mark.asyncio
async def test_case_e_prohibited_technology_guardrail(async_client: AsyncClient):
    """Case E: Multi-negative constraints (pandas, numpy) are tracked and preserved."""
    prompt = "Analyze sales.csv using Python. Do not use pandas or numpy."
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": "Write Python code to analyze sales.csv without using pandas or numpy."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        opt = data["optimized_prompt"].lower()
        assert "without using pandas or numpy" in opt or ("pandas" in opt and "numpy" in opt)
        assert data["validation_status"] == "passed"
        prohibition_constraints = [c for c in data["explicit_constraints"] if "prohibition" in c.lower()]
        assert len(prohibition_constraints) >= 2


@pytest.mark.asyncio
async def test_case_f_empty_or_invalid_model_output_fallback(async_client: AsyncClient):
    """Case F: Model returning whitespace-only triggers safe validation fallback to original prompt."""
    prompt = "Extract all dates and invoice amounts from this text."
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": "   \n\t  "},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        # Should safely fall back to original prompt rather than returning empty string or crashing
        assert data["optimized_prompt"] == prompt
        assert data["is_preserved"] is True
        assert data["validation_status"] == "fallback"


@pytest.mark.asyncio
async def test_case_g_validation_rejection_of_hallucinated_json(async_client: AsyncClient):
    """Case G: Validator detects unrequested JSON format hallucination and triggers retry/fallback."""
    prompt = "Write a Python script that calculates factorial of a number."
    # Model hallucinates an unrequested mandatory JSON schema output
    hallucinated_resp = httpx.Response(
        status_code=200,
        json={"response": "Write a Python script that outputs as JSON: {'result': factorial}."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=hallucinated_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        # After retry also hallucinates, fallback returns original prompt safely
        assert data["validation_status"] == "fallback"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == prompt


@pytest.mark.asyncio
async def test_case_h_longer_output_expansion_metrics(async_client: AsyncClient):
    """Case H: Honest reporting when clarification expands prompt length."""
    prompt = "Help with sales."
    expanded_text = (
        "Analyze the provided quarterly sales dataset. Calculate revenue growth, "
        "regional performance breakdown, and top-performing sales representatives."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": expanded_text},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        # Character delta: final - original (positive for expansion)
        assert data["char_delta"] > 0
        assert data["token_savings"] == 0  # Do not report savings on expansion
        assert data["token_change_percent"] > 0
        assert data["token_reduction_percent"] == 0  # No false reduction claimed
        assert data["length_result"] == "expanded"


# ==============================================================================
# Phase 3.2: Semantic Preservation & Anti-Hardcoding Regression Tests
# ==============================================================================


def test_validator_rejects_student_marks_turn_reversal():
    """Phase 3.2 - Test 1: Direct validation rejects interactive data request."""
    from app.services.task_analyzer import TaskAnalyzer
    from app.services.candidate_validator import CandidateValidator

    prompt = (
        "Build me a Python program to analyze student marks and tell me useful things. "
        "Make it good and don't miss anything important."
    )
    invalid_candidate = "Please provide the student marks data for analysis."

    analysis = TaskAnalyzer.analyze(prompt)
    assert analysis.deliverable_family == "software"
    assert analysis.primary_action in {"build", "create"}

    validation = CandidateValidator.validate(prompt, invalid_candidate, analysis)
    assert validation.is_valid is False
    assert validation.status == "fallback"
    assert any(
        kw in validation.reason.lower()
        for kw in ["user data", "deliverable", "requesting"]
    )


@pytest.mark.asyncio
async def test_successful_corrective_retry_preserves_program_building(async_client: AsyncClient):
    """Phase 3.2 - Test 2: Invalid first attempt triggers corrective retry which succeeds."""
    prompt = (
        "Build me a Python program to analyze student marks and tell me useful things. "
        "Make it good and don't miss anything important."
    )

    resp_invalid = httpx.Response(
        status_code=200,
        json={"response": "Please provide the student marks data for analysis."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    resp_valid = httpx.Response(
        status_code=200,
        json={
            "response": (
                "Create a Python program to analyze student marks. "
                "Calculate average, highest and lowest marks, pass/fail counts, "
                "and display summary statistics. Include runnable code."
            )
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch(
        "app.services.ollama_service.httpx.AsyncClient",
        create_mock_client(post_side_effect=[resp_invalid, resp_valid]),
    ):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        # Should successfully adopt the corrected retry candidate
        assert data["validation_status"] == "passed"
        assert data["is_preserved"] is False
        assert "program" in data["optimized_prompt"].lower() or "python" in data["optimized_prompt"].lower()
        assert "student marks" in data["optimized_prompt"].lower()


@pytest.mark.asyncio
async def test_failed_retry_safely_falls_back_to_original(async_client: AsyncClient):
    """Phase 3.2 - Test 3: If retry also produces an invalid output, safely fall back to original prompt."""
    prompt = (
        "Build me a Python program to analyze student marks and tell me useful things. "
        "Make it good and don't miss anything important."
    )

    resp_invalid_1 = httpx.Response(
        status_code=200,
        json={"response": "Please provide the student marks data for analysis."},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    resp_invalid_2 = httpx.Response(
        status_code=200,
        json={"response": "Can you provide the student marks dataset?"},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch(
        "app.services.ollama_service.httpx.AsyncClient",
        create_mock_client(post_side_effect=[resp_invalid_1, resp_invalid_2]),
    ):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        # Must safely fall back to original prompt with explicit fallback status
        assert data["validation_status"] == "fallback"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == prompt


@pytest.mark.asyncio
async def test_explanation_task_preservation(async_client: AsyncClient):
    """Phase 3.2 - Test 7: Explanatory requests preserve the explanation objective."""
    prompt = "Explain how a Python dictionary works using a small example"

    mock_resp = httpx.Response(
        status_code=200,
        json={
            "response": (
                "Explain Python dictionary fundamentals using a concise code example "
                "demonstrating key-value assignment, lookup, and iteration."
            )
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert response.status_code == 200
        data = response.json()

        assert data["validation_status"] == "passed"
        assert "explain" in data["optimized_prompt"].lower()
        assert "dictionary" in data["optimized_prompt"].lower()


# =========================================================================
# Phase 3.2.1: Required-Operation Preservation Tests
# =========================================================================

PROMPT_STUDENTS_CSV = (
    "Create a Python program that reads students.csv, calculates average marks, "
    "identifies the highest-scoring student, and saves the results to report.csv. "
    "Do not use pandas."
)

CANDIDATE_MISSING_OPERATIONS = (
    "- Task: Write a Python program\n"
    "- Inputs/Context: students.csv, report.csv\n"
    "- Constraints: Do not use pandas\n"
    "- Deliverable: Program"
)

CANDIDATE_VALID_EQUIVALENT = (
    "Write a Python program without pandas that reads students.csv, computes the "
    "average marks, finds the highest-scoring student, and writes the results to report.csv."
)


def test_phase_3_2_1_test_a_confirmed_failure_rejected():
    """Test A: The confirmed failure (missing average marks & highest-scoring student).

    Candidate preserves filenames and prohibition, but drops required operations.
    Expected: Candidate is rejected, missing operations detected, cannot be marked as passed.
    """
    analysis = TaskAnalyzer.analyze(PROMPT_STUDENTS_CSV)
    result = CandidateValidator.validate(
        original_prompt=PROMPT_STUDENTS_CSV,
        candidate_prompt=CANDIDATE_MISSING_OPERATIONS,
        analysis=analysis,
    )

    assert result.is_valid is False
    assert result.status == "fallback"
    assert "calculate average marks" in result.reason
    assert "identify highest-scoring student" in result.reason
    assert any("average marks" in c for c in result.missing_constraints)
    assert any("highest-scoring student" in c for c in result.missing_constraints)


def test_phase_3_2_1_test_b_valid_equivalent_rewrite_passes():
    """Test B: Valid equivalent rewrite preserves all required operations.

    Candidate uses synonyms (computes, finds, writes) while maintaining operations,
    filenames, and prohibition.
    Expected: Candidate passes validation.
    """
    analysis = TaskAnalyzer.analyze(PROMPT_STUDENTS_CSV)
    result = CandidateValidator.validate(
        original_prompt=PROMPT_STUDENTS_CSV,
        candidate_prompt=CANDIDATE_VALID_EQUIVALENT,
        analysis=analysis,
    )

    assert result.is_valid is True
    assert result.status == "passed"
    assert len(result.missing_constraints) == 0


@pytest.mark.asyncio
async def test_phase_3_2_1_test_c_operation_preservation_retry(async_client: AsyncClient):
    """Test C: Model returns invalid candidate on attempt 1, valid candidate on retry.

    Expected: First candidate rejected, exactly one corrective retry occurs, valid retry accepted.
    """
    resp_invalid = httpx.Response(
        status_code=200,
        json={"response": CANDIDATE_MISSING_OPERATIONS},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    resp_valid = httpx.Response(
        status_code=200,
        json={"response": CANDIDATE_VALID_EQUIVALENT},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch(
        "app.services.ollama_service.httpx.AsyncClient",
        create_mock_client(post_side_effect=[resp_invalid, resp_valid]),
    ):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": PROMPT_STUDENTS_CSV, "mode": "balanced"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["validation_status"] == "passed"
        assert data["is_preserved"] is False
        assert data["optimized_prompt"] == CANDIDATE_VALID_EQUIVALENT
        assert "computes the average marks" in data["optimized_prompt"]
        assert "finds the highest-scoring student" in data["optimized_prompt"]


@pytest.mark.asyncio
async def test_phase_3_2_1_test_d_failed_retry_fallback(async_client: AsyncClient):
    """Test D: Model returns invalid candidates on both attempts.

    Expected: Safely falls back to original prompt, validation_status is 'fallback',
    is_preserved is True, no invalid candidate reported as success.
    """
    resp_invalid_1 = httpx.Response(
        status_code=200,
        json={"response": CANDIDATE_MISSING_OPERATIONS},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )
    resp_invalid_2 = httpx.Response(
        status_code=200,
        json={
            "response": (
                "- Task: Python code\n"
                "- Input: students.csv\n"
                "- Output: report.csv\n"
                "- Deliverable: Script"
            )
        },
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch(
        "app.services.ollama_service.httpx.AsyncClient",
        create_mock_client(post_side_effect=[resp_invalid_1, resp_invalid_2]),
    ):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": PROMPT_STUDENTS_CSV, "mode": "balanced"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["validation_status"] == "fallback"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == PROMPT_STUDENTS_CSV


def test_phase_3_2_1_test_e_generalization_to_another_domain():
    """Test E: Generalization to orders.csv data analysis domain.

    Input:
    'Write a Python script that reads orders.csv, groups orders by customer,
    calculates each customer's total spending, and saves the summary to customer_totals.csv.'

    Candidate preserves deliverable and files, but omits calculating total spending.
    Expected: Rejected because required calculation operation is missing.
    """
    prompt = (
        "Write a Python script that reads orders.csv, groups orders by customer, "
        "calculates each customer's total spending, and saves the summary to customer_totals.csv."
    )
    candidate_missing_calc = (
        "Write a Python script that reads orders.csv, groups orders by customer, "
        "and saves the summary to customer_totals.csv."
    )

    analysis = TaskAnalyzer.analyze(prompt)
    result = CandidateValidator.validate(
        original_prompt=prompt,
        candidate_prompt=candidate_missing_calc,
        analysis=analysis,
    )

    assert result.is_valid is False
    assert result.status == "fallback"
    assert any("spending" in c or "total" in c for c in result.missing_constraints)


def test_phase_3_2_1_test_f_equivalent_wording_passes():
    """Test F: Equivalent wording with alternative synonyms.

    Candidate uses 'determine the mean marks', 'pick the top-performing student', 'export output'.
    Expected: Passes without requiring exact string matching.
    """
    candidate_alt_wording = (
        "Create a script without pandas to load students.csv, determine the mean marks, "
        "pick the top-performing student, and export output to report.csv."
    )

    analysis = TaskAnalyzer.analyze(PROMPT_STUDENTS_CSV)
    result = CandidateValidator.validate(
        original_prompt=PROMPT_STUDENTS_CSV,
        candidate_prompt=candidate_alt_wording,
        analysis=analysis,
    )

    assert result.is_valid is True
    assert result.status == "passed"


# ==============================================================================
# Phase 3.2.2: Strategy Accuracy, Minimal Rewriting & Honest Metrics Tests
# ==============================================================================

PROMPT_ORDERS_CSV = (
    "Write a Python script that reads orders.csv, groups orders by customer, "
    "calculates each customer's total spending, and saves the summary to customer_totals.csv."
)


@pytest.mark.asyncio
async def test_phase_3_2_2_test_a_concise_orders_prompt(async_client: AsyncClient):
    """Test A: Concise mode with orders prompt.

    Expected:
    - All requested operations and filenames survive.
    - No unrequested execution or additional artifact is introduced.
    - A valid shorter equivalent rewrite is accepted.
    """
    valid_shorter_candidate = (
        "Write a Python script to read orders.csv, group by customer, "
        "compute each customer's total spending, and save to customer_totals.csv."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": valid_shorter_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": PROMPT_ORDERS_CSV, "mode": "concise"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["validation_status"] == "passed"
        assert data["outcome"] == "validated_rewrite"
        assert data["length_result"] == "compressed"
        assert data["applied_strategy"] == "compress"
        assert data["char_delta"] < 0
        assert data["token_savings"] > 0
        assert "orders.csv" in data["optimized_prompt"]
        assert "customer_totals.csv" in data["optimized_prompt"]

    # Reject candidate with unrequested execution instruction
    candidate_unrequested_exec = (
        "Write a Python script that loads orders.csv, groups orders by customer, "
        "computes total spending, saves to customer_totals.csv, and execute the script in bash."
    )
    analysis = TaskAnalyzer.analyze(PROMPT_ORDERS_CSV)
    val_exec = CandidateValidator.validate(PROMPT_ORDERS_CSV, candidate_unrequested_exec, analysis)
    assert val_exec.is_valid is False
    assert any("execution" in r.lower() for r in val_exec.hallucinated_requirements)

    # Reject candidate with unrequested additional artifact
    candidate_extra_artifact = (
        "Write a Python script that loads orders.csv, groups orders by customer, "
        "computes total spending, saves to customer_totals.csv and extra_audit.log."
    )
    val_art = CandidateValidator.validate(PROMPT_ORDERS_CSV, candidate_extra_artifact, analysis)
    assert val_art.is_valid is False
    assert any("artifact" in r.lower() or "extra_audit.log" in r for r in val_art.hallucinated_requirements)


@pytest.mark.asyncio
async def test_phase_3_2_2_test_b_concise_student_marks_prompt(async_client: AsyncClient):
    """Test B: Concise mode with student-marks prompt.

    Expected:
    - All operations, filenames, and prohibition survive.
    - Valid concise rewrite is accepted.
    - Fallback is reported accurately when candidate drops prohibition.
    """
    valid_concise = (
        "Write a Python program without pandas to parse students.csv, compute average marks, "
        "find the top-scoring student, and save to report.csv."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": valid_concise},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": PROMPT_STUDENTS_CSV, "mode": "concise"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["validation_status"] == "passed"
        assert data["outcome"] == "validated_rewrite"
        assert "students.csv" in data["optimized_prompt"]
        assert "report.csv" in data["optimized_prompt"]
        assert "pandas" in data["optimized_prompt"]

    # Candidate missing prohibition falls back accurately
    invalid_candidate = (
        "Write a Python program using pandas to parse students.csv, compute average marks, "
        "find the top-scoring student, and save to report.csv."
    )
    mock_invalid = httpx.Response(
        status_code=200,
        json={"response": invalid_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch(
        "app.services.ollama_service.httpx.AsyncClient",
        create_mock_client(post_side_effect=[mock_invalid, mock_invalid]),
    ):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": PROMPT_STUDENTS_CSV, "mode": "concise"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["validation_status"] == "fallback"
        assert data["outcome"] == "fallback"
        assert data["applied_strategy"] == "preserve"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == PROMPT_STUDENTS_CSV


@pytest.mark.asyncio
async def test_phase_3_2_2_test_c_structure_focused_mode(async_client: AsyncClient):
    """Test C: Structure-focused mode accepts useful structure, rejects unrequested formats."""
    prompt = (
        "Create a data pipeline that fetches raw events from events.json, validates event schemas, "
        "filters duplicates, aggregates hourly counts, and writes the clean data to clean_events.parquet. "
        "Do not drop corrupted records silently; log errors to error.log."
    )
    structured_candidate = (
        "Data Pipeline Requirements:\n"
        "- Ingest: Read raw events from events.json\n"
        "- Process: Validate schemas, filter duplicates, and aggregate hourly counts\n"
        "- Output: Write clean records to clean_events.parquet\n"
        "- Error Handling: Do not drop corrupted records silently; log errors to error.log"
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": structured_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": prompt, "mode": "structure"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["validation_status"] == "passed"
        assert data["applied_strategy"] == "structure"
        assert data["outcome"] == "validated_rewrite"

    # Reject unrequested mandatory JSON format imposition
    unsupported_format_cand = 'Return ONLY valid JSON matching schema: {"events": []}'
    analysis = TaskAnalyzer.analyze(prompt)
    val_format = CandidateValidator.validate(prompt, unsupported_format_cand, analysis)
    assert val_format.is_valid is False


@pytest.mark.asyncio
async def test_phase_3_2_2_test_d_strategy_mismatch(async_client: AsyncClient):
    """Test D: Strategy mismatch reporting.

    When Structure-focused mode is chosen, but the returned rewrite is flat text without
    markdown structure or falls back, the system must not falsely claim structure was applied.
    """
    flat_compressed_candidate = (
        "Audit Python code for SQLi and memory leaks, then output refactored code."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": flat_compressed_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": SAMPLE_PROMPT, "mode": "structure"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "structure"
        # Since the candidate has no bullet points or markdown headings, applied_strategy is compress, not structure
        assert data["applied_strategy"] == "compress"


@pytest.mark.asyncio
async def test_phase_3_2_2_test_e_valid_expansion(async_client: AsyncClient):
    """Test E: Valid expansion reporting.

    Candidate is longer with useful clarification. Valid rewrite accepted,
    metrics report expansion honestly without claiming savings.
    """
    prompt = "Explain Docker containers to a beginner."
    expanded_candidate = (
        "Explain Docker containers to a software beginner, detailing container images, "
        "runtime isolation from the host operating system, and key differences from virtual machines."
    )
    mock_resp = httpx.Response(
        status_code=200,
        json={"response": expanded_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_resp)):
        response = await async_client.post(
            "/api/optimize",
            json={"prompt": prompt, "mode": "balanced"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["validation_status"] == "passed"
        assert data["outcome"] == "validated_rewrite"
        assert data["length_result"] == "expanded"
        assert data["char_delta"] > 0
        assert data["token_savings"] == 0
        assert data["token_reduction_percent"] == 0
        assert data["token_change_percent"] > 0


def test_phase_3_2_2_test_f_unsupported_addition_direct():
    """Test F: CandidateValidator rejects unrequested execution and unrequested artifact."""
    prompt = "Write a Python script that calculates factorial."
    analysis = TaskAnalyzer.analyze(prompt)

    # Adding execution requirement
    cand_exec = "Write a Python script that calculates factorial. Run and execute the script in terminal."
    res_exec = CandidateValidator.validate(prompt, cand_exec, analysis)
    assert res_exec.is_valid is False
    assert any("execution" in r.lower() for r in res_exec.hallucinated_requirements)

    # Adding extra artifact
    cand_art = "Write a Python script that calculates factorial and saves output to factorial_output.csv."
    res_art = CandidateValidator.validate(prompt, cand_art, analysis)
    assert res_art.is_valid is False
    assert any("factorial_output.csv" in r for r in res_art.hallucinated_requirements)


@pytest.mark.asyncio
async def test_phase_3_2_2_test_h_metric_consistency(async_client: AsyncClient):
    """Test H: Metric consistency across shorter, longer, same-length, and unchanged prompts."""
    # 1. Shorter rewrite
    orig_1 = "Please review this code carefully and thoroughly."
    opt_1 = "Review this code."
    mock_1 = httpx.Response(status_code=200, json={"response": opt_1}, request=httpx.Request("POST", "http://x"))
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_1)):
        r1 = await async_client.post("/api/optimize", json={"prompt": orig_1, "mode": "concise"})
        d1 = r1.json()
        assert d1["char_delta"] == len(opt_1) - len(orig_1)
        assert d1["char_delta"] < 0
        assert d1["token_savings"] > 0
        assert d1["length_result"] == "compressed"

    # 2. Same-length but different content
    orig_2 = "abcdefghij"
    opt_2 = "1234567890"
    mock_2 = httpx.Response(status_code=200, json={"response": opt_2}, request=httpx.Request("POST", "http://x"))
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_2)):
        r2 = await async_client.post("/api/optimize", json={"prompt": orig_2, "mode": "balanced"})
        d2 = r2.json()
        assert d2["char_delta"] == 0
        assert d2["length_result"] == "unchanged"
        assert d2["is_preserved"] is False  # content differs!

    # 3. Completely unchanged content
    orig_3 = "Keep this exact text."
    mock_3 = httpx.Response(status_code=200, json={"response": orig_3}, request=httpx.Request("POST", "http://x"))
    with patch("app.services.ollama_service.httpx.AsyncClient", create_mock_client(post_response=mock_3)):
        r3 = await async_client.post("/api/optimize", json={"prompt": orig_3, "mode": "balanced"})
        d3 = r3.json()
        assert d3["char_delta"] == 0
        assert d3["length_result"] == "unchanged"
        assert d3["is_preserved"] is True
        assert d3["outcome"] == "unchanged_original"
        assert d3["applied_strategy"] == "preserve"




