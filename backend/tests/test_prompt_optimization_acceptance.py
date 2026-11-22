"""Regression test suite for prompt optimization acceptance logic, token accounting, and fallback dispatch.

Verifies the 10 required edge cases:
1. Candidate shorter than original -> Accepted, dispatched matches candidate, positive savings.
2. Candidate equal in length -> Rejected, dispatched matches original, zero savings reported.
3. Candidate longer than original -> Rejected, dispatched matches original, zero savings reported.
4. Empty / whitespace-only prompt -> 422 Validation error.
5. Very short prompt -> Safely preserved without expansion or cosmetic alteration.
6. Long redundant prompt -> Redundancies eliminated, candidate accepted with verified token reduction.
7. Technical prompt with constraints -> Operations, prohibitions, and deliverables strictly preserved.
8. Prompt containing code -> Code payload preserved intact.
9. Prompt containing numbers and proper nouns -> Exact numeric values, dates, and entities preserved.
10. Tokenizer measurement failure -> Graceful fallback to tiktoken/heuristic without crash or negative savings.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from httpx import AsyncClient

from app.schemas.optimizer import OptimizeRequest
from app.services.ollama_service import OllamaService
from app.services.tokenizer_service import TokenizerService


DEFAULT_MOCK_TAGS = httpx.Response(
    status_code=200,
    json={
        "models": [
            {
                "name": "qwen2.5:1.5b",
                "model": "qwen2.5:1.5b",
                "size": 986061892,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "1.5B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    },
    request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
)


def mock_client_factory(post_response: httpx.Response):
    """Create a mock httpx.AsyncClient that returns post_response for generate and tags for get."""
    mock_inst = AsyncMock()
    mock_inst.post.return_value = post_response
    mock_inst.get.return_value = DEFAULT_MOCK_TAGS

    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__.return_value = mock_inst
    mock_cls.return_value.__aexit__.return_value = None
    return mock_cls


@pytest.mark.asyncio
async def test_regression_1_candidate_shorter_than_original(async_client: AsyncClient):
    """Case 1: Candidate shorter than original is accepted, dispatched matches candidate, savings positive."""
    orig_prompt = "Please kindly review this code and check for any memory leaks and SQL injection vulnerabilities."
    shorter_candidate = "Review code for memory leaks and SQL injection."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": shorter_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        assert data["rejection_reason"] is None
        assert data["optimized_prompt"] == shorter_candidate
        assert data["candidate_prompt"] == shorter_candidate
        assert data["original_token_count"] > data["dispatched_token_count"]
        assert data["actual_tokens_saved"] > 0
        assert data["actual_reduction_percent"] > 0.0
        assert data["token_savings"] == data["actual_tokens_saved"]
        assert data["token_reduction_percent"] > 0
        assert data["outcome"] == "validated_rewrite"
        assert data["is_preserved"] is False


@pytest.mark.asyncio
async def test_regression_2_candidate_equal_in_length(async_client: AsyncClient):
    """Case 2: Candidate equal in length is rejected, dispatched matches original, zero savings reported."""
    orig_prompt = "Audit Python code for security flaws."  # 6 words
    # Candidate with equal tokens / length that just swaps synonyms or punctuation
    equal_candidate = "Check Python code for security bugs."  # 6 words, equal tokens

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": equal_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    # Force pairwise tokenizer to return identical counts
    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(8, 8, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert "equal to original" in data["rejection_reason"]
        assert data["optimized_prompt"] == orig_prompt
        assert data["candidate_prompt"] == equal_candidate
        assert data["dispatched_token_count"] == data["original_token_count"]
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert data["token_savings"] == 0
        assert data["token_reduction_percent"] == 0
        assert data["token_change_percent"] == 0
        assert data["outcome"] == "unchanged_original"
        assert data["is_preserved"] is True
        assert data["applied_strategy"] == "preserve"


@pytest.mark.asyncio
async def test_regression_3_candidate_longer_than_original(async_client: AsyncClient):
    """Case 3: Candidate longer than original is rejected, dispatched matches original, zero savings reported."""
    orig_prompt = "Review this Python code for bugs."
    longer_candidate = (
        "Please conduct a comprehensive, thorough, and highly detailed code review of this Python script "
        "in order to discover any subtle security flaws, performance degradation, or bugs."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": longer_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert "greater than original" in data["rejection_reason"]
        assert data["optimized_prompt"] == orig_prompt
        assert data["candidate_prompt"] == longer_candidate
        assert data["dispatched_token_count"] == data["original_token_count"]
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert data["token_savings"] == 0
        assert data["token_reduction_percent"] == 0
        assert data["token_change_percent"] == 0
        assert data["outcome"] == "unchanged_original"
        assert data["is_preserved"] is True
        assert data["applied_strategy"] == "preserve"


@pytest.mark.asyncio
async def test_regression_4_empty_prompt_rejected(async_client: AsyncClient):
    """Case 4: Empty and whitespace-only prompts are rejected with 422 HTTP validation error."""
    res1 = await async_client.post("/api/optimize", json={"prompt": ""})
    assert res1.status_code == 422

    res2 = await async_client.post("/api/optimize", json={"prompt": "   \n\t  "})
    assert res2.status_code == 422


@pytest.mark.asyncio
async def test_regression_5_very_short_prompt(async_client: AsyncClient):
    """Case 5: Very short prompt is preserved when model cannot safely reduce tokens."""
    prompt = "Format this date: 2026-11-06."
    candidate_rephrase = "Please format this date: 2026-11-06."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": candidate_rephrase},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        # The longer rephrase must be rejected, preserving the short prompt
        assert data["candidate_accepted"] is False
        assert data["optimized_prompt"] == prompt
        assert data["is_preserved"] is True
        assert data["actual_tokens_saved"] == 0


@pytest.mark.asyncio
async def test_regression_6_long_redundant_prompt(async_client: AsyncClient):
    """Case 6: Long redundant prompt removes conversational filler and circular repeats."""
    prompt = (
        "I would like you to act as an expert senior database architect. Could you please review "
        "this SQL query for me? I would really like you to make sure to check if there are any indexes "
        "that could be added, and in addition to that, please make sure that you check if the query "
        "could be rewritten to be faster because performance is very slow right now."
    )
    concise_rewrite = "Review this SQL query. Identify missing indexes and optimize for faster execution."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": concise_rewrite},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        assert data["optimized_prompt"] == concise_rewrite
        assert data["actual_tokens_saved"] > 20
        assert data["actual_reduction_percent"] >= 40.0
        assert data["token_savings"] == data["actual_tokens_saved"]


@pytest.mark.asyncio
async def test_regression_7_technical_prompt_constraints_preserved(async_client: AsyncClient):
    """Case 7: Technical prompt with explicit prohibitions, filenames, and deliverables is preserved."""
    prompt = (
        "Write a Python script that reads telemetry.log, filters out lines containing 'DEBUG', "
        "saves the filtered output to critical.log, and does not use regex."
    )
    valid_concise = (
        "Write a Python script to filter 'DEBUG' lines from telemetry.log and save to critical.log without regex."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": valid_concise},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        opt = data["optimized_prompt"]
        assert "telemetry.log" in opt
        assert "critical.log" in opt
        assert "DEBUG" in opt
        assert "regex" in opt
        assert any("prohibition" in c.lower() for c in data["explicit_constraints"])


@pytest.mark.asyncio
async def test_regression_8_prompt_containing_code(async_client: AsyncClient):
    """Case 8: Prompt containing raw code snippet preserves code payload without corruption."""
    prompt = (
        "Please inspect this code for memory leaks and optimize it:\n\n"
        "```python\n"
        "def compute(n):\n"
        "    data = [i * 2 for i in range(n)]\n"
        "    return sum(data)\n"
        "```"
    )
    concise_rewrite = (
        "Inspect and optimize this code for memory efficiency:\n\n"
        "```python\n"
        "def compute(n):\n"
        "    data = [i * 2 for i in range(n)]\n"
        "    return sum(data)\n"
        "```"
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": concise_rewrite},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        assert "def compute(n):" in data["optimized_prompt"]
        assert "return sum(data)" in data["optimized_prompt"]


@pytest.mark.asyncio
async def test_regression_9_numbers_and_proper_nouns_preserved(async_client: AsyncClient):
    """Case 9: Prompt containing exact numbers, dates, and proper nouns preserves them strictly."""
    prompt = (
        "Please query the PostgreSQL database for all accounts created after 2026-01-15 "
        "that have a balance exceeding $50,000 and belong to the Enterprise tier."
    )
    concise_rewrite = (
        "Query PostgreSQL for Enterprise accounts created after 2026-01-15 with balance > $50,000."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": concise_rewrite},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is True
        opt = data["optimized_prompt"]
        assert "PostgreSQL" in opt
        assert "2026-01-15" in opt
        assert "50,000" in opt
        assert "Enterprise" in opt


@pytest.mark.asyncio
async def test_regression_10_tokenizer_measurement_failure_graceful_fallback(async_client: AsyncClient):
    """Case 10: When Ollama tokenizer fails or times out, TokenizerService falls back gracefully to tiktoken."""
    orig_prompt = "Please review this Python code very carefully and check for bugs."
    shorter_candidate = "Review Python code for bugs."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": shorter_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    # Force Ollama native measurement to raise an exception inside TokenizerService to test fallback
    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch("app.services.tokenizer_service.httpx.AsyncClient.post", side_effect=httpx.ConnectError("Ollama offline")):

        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "concise"})
        assert res.status_code == 200
        data = res.json()

        # TokenizerService gracefully fell back to tiktoken_cl100k
        assert data["token_measurement_method"] == "tiktoken_cl100k"
        assert data["candidate_accepted"] is True
        assert data["original_token_count"] > data["dispatched_token_count"]
        assert data["actual_tokens_saved"] > 0
        assert data["optimized_prompt"] == shorter_candidate


@pytest.mark.asyncio
async def test_regression_11_equal_tokens_research_path_rejected(async_client: AsyncClient):
    """Case 11: Research category prompt with equal candidate tokens is rejected under compression policy."""
    orig_prompt = "What is the capital of France?"  # SQ-01
    equal_candidate = "Identify capital of France."  # Equal tokens

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": equal_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(7, 7, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert data["outcome"] == "unchanged_original"
        assert data["applied_strategy"] == "preserve"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == orig_prompt
        assert data["dispatched_token_count"] == 7
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert "equal to original" in data["rejection_reason"]


@pytest.mark.asyncio
async def test_regression_12_equal_tokens_planning_path_rejected(async_client: AsyncClient):
    """Case 12: Planning category prompt with equal candidate tokens is rejected under compression policy."""
    orig_prompt = "How many planets are there in the solar system?"  # SQ-03
    equal_candidate = "Count planets located in the solar system."  # Equal tokens

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": equal_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(10, 10, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert data["outcome"] == "unchanged_original"
        assert data["applied_strategy"] == "preserve"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == orig_prompt
        assert data["dispatched_token_count"] == 10
        assert data["actual_tokens_saved"] == 0
        assert "equal to original" in data["rejection_reason"]


@pytest.mark.asyncio
async def test_regression_13_equal_tokens_structure_mode_rejected(async_client: AsyncClient):
    """Case 13: Structure mode prompt where candidate has equal tokens and allow_expansion=False is rejected."""
    orig_prompt = "Step 1: Download code. Step 2: Build code. Step 3: Test code."
    equal_candidate = "1. Download code.\n2. Build code.\n3. Test code."

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": equal_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(15, 15, "test_mock")):

        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "structure"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert data["outcome"] == "unchanged_original"
        assert data["applied_strategy"] == "preserve"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == orig_prompt
        assert data["actual_tokens_saved"] == 0
        assert "equal to original" in data["rejection_reason"]


@pytest.mark.asyncio
async def test_regression_14_identical_text_candidate_rejected(async_client: AsyncClient):
    """Case 14: Model returns identical prompt text verbatim; candidate rejected and original preserved."""
    orig_prompt = "What is the capital of France?"
    identical_candidate = "What is the capital of France?"

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": identical_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)):
        res = await async_client.post("/api/optimize", json={"prompt": orig_prompt, "mode": "balanced"})
        assert res.status_code == 200
        data = res.json()

        assert data["candidate_accepted"] is False
        assert data["outcome"] == "unchanged_original"
        assert data["is_preserved"] is True
        assert data["optimized_prompt"] == orig_prompt
        assert "identical to original" in data["rejection_reason"]


@pytest.mark.asyncio
async def test_regression_15_upfront_preserve_candidate_not_accepted(async_client: AsyncClient):
    """Case 15: Upfront preserved prompt (strategy=preserve) reports candidate_accepted=False and 0 savings."""
    prompt = "Return the square of the supplied integer as a JSON object with the key 'result'."

    res = await async_client.post("/api/optimize", json={"prompt": prompt, "mode": "balanced"})
    assert res.status_code == 200
    data = res.json()

    assert data["strategy"] == "preserve"
    assert data["applied_strategy"] == "preserve"
    assert data["candidate_accepted"] is False
    assert data["is_preserved"] is True
    assert data["actual_tokens_saved"] == 0
    assert data["actual_reduction_percent"] == 0.0
    assert data["token_savings"] == 0
    assert data["token_change_percent"] == 0
    assert data["optimized_prompt"] == prompt
    assert "already concise and effective" in data["rejection_reason"]


@pytest.mark.asyncio
async def test_regression_16_explicit_expansion_never_fabricates_positive_savings(async_client: AsyncClient):
    """Case 16: Explicit expansion (allow_expansion=True) accepts candidate but reports 0 savings and honest positive change."""
    orig_prompt = "Explain quantum computing."
    expanded_candidate = (
        "Explain quantum computing fundamentals, including qubits, superposition, quantum entanglement, "
        "and practical limitations in comparison to classical binary computing architectures."
    )

    mock_resp = httpx.Response(
        status_code=200,
        json={"response": expanded_candidate},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/generate"),
    )

    with patch("app.services.ollama_service.httpx.AsyncClient", mock_client_factory(mock_resp)), \
         patch.object(TokenizerService, "measure_tokens_pair", return_value=(4, 24, "test_mock")):

        res = await async_client.post(
            "/api/optimize",
            json={"prompt": orig_prompt, "mode": "balanced", "allow_expansion": True},
        )
        assert res.status_code == 200
        data = res.json()

        # Under explicit expansion, candidate IS accepted
        assert data["candidate_accepted"] is True
        assert data["rejection_reason"] is None
        assert data["optimized_prompt"] == expanded_candidate
        assert data["dispatched_token_count"] == 24
        assert data["original_token_count"] == 4
        # BUT savings MUST NEVER be positive!
        assert data["actual_tokens_saved"] == 0
        assert data["actual_reduction_percent"] == 0.0
        assert data["token_savings"] == 0
        assert data["token_reduction_percent"] == 0
        # Positive expansion percentage is honestly reported
        assert data["token_change_percent"] > 0
        assert data["length_result"] == "expanded"
        assert data["outcome"] == "validated_rewrite"
