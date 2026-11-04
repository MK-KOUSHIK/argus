"""Unit and Integration Tests for Output Budget Control (Phase 4.1).

Validates:
1. Dynamic allocation across task categories (summarization, extraction, coding, etc.)
2. Explicit prompt length directives (word counts, sentence counts, brevity cues)
3. API override parameter (requested_output_budget)
4. Mode scaling (concise, balanced, structure)
5. Context headroom coordination and bounds clamping
6. HTTP 413 rejection when available context < min tokens
7. Ollama native telemetry extraction (done_reason, eval_count, utilization)
8. Graceful null handling for missing telemetry (zero fabrication)
"""

import math
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest
from app.services.adaptive_context_manager import AdaptiveContextManager, ContextOverflowError
from app.services.ollama_service import OllamaService
from app.services.output_budget_controller import OutputBudgetController, OutputBudgetAllocation
from app.services.task_analyzer import TaskAnalyzer


def test_task_category_baselines():
    """Verify task categories receive appropriate baseline budgets."""
    # Summarization
    alloc_sum = OutputBudgetController.allocate_budget(
        prompt="Summarize the quarterly earnings report in detail.",
        mode="balanced",
    )
    assert alloc_sum.task_category == "summarization"
    assert alloc_sum.allocated_budget == 256
    assert "summarization" in alloc_sum.reason

    # Extraction
    alloc_ext = OutputBudgetController.allocate_budget(
        prompt="Extract the customer name and order total into JSON from this receipt.",
        mode="balanced",
    )
    assert alloc_ext.task_category == "extraction"
    assert alloc_ext.allocated_budget == 256

    # Coding
    alloc_code = OutputBudgetController.allocate_budget(
        prompt="Write a Python script to sort a list of dictionaries by key.",
        mode="balanced",
    )
    assert alloc_code.task_category == "coding"
    assert alloc_code.allocated_budget == 512

    # Writing
    alloc_write = OutputBudgetController.allocate_budget(
        prompt="Write an essay discussing the ethical implications of artificial intelligence.",
        mode="balanced",
    )
    assert alloc_write.task_category == "writing"
    assert alloc_write.allocated_budget == 512


def test_mode_scaling():
    """Verify optimization mode scales the allocated budget correctly."""
    prompt = "Explain the difference between TCP and UDP protocols."

    # Concise: 0.75x
    alloc_concise = OutputBudgetController.allocate_budget(prompt=prompt, mode="concise")
    # Balanced: 1.0x
    alloc_balanced = OutputBudgetController.allocate_budget(prompt=prompt, mode="balanced")
    # Structure: 1.2x
    alloc_structure = OutputBudgetController.allocate_budget(prompt=prompt, mode="structure")

    assert alloc_concise.allocated_budget < alloc_balanced.allocated_budget
    assert alloc_structure.allocated_budget > alloc_balanced.allocated_budget
    assert alloc_concise.allocated_budget == int(math.ceil(alloc_balanced.allocated_budget * 0.75))
    assert alloc_structure.allocated_budget == int(math.ceil(alloc_balanced.allocated_budget * 1.2))


def test_explicit_word_count_constraint():
    """Verify explicit word limits in prompts dynamically dictate the output budget."""
    prompt = "Summarize the history of cryptography in under 50 words."
    alloc = OutputBudgetController.allocate_budget(prompt=prompt, mode="balanced")

    assert alloc.detected_constraint is not None
    assert "50 words" in alloc.detected_constraint
    # ~50 * 1.33 + 64 headroom = 67 + 64 = 131 tokens
    expected_tokens = math.ceil(50 * 1.33) + settings.OUTPUT_BUDGET_SAFETY_HEADROOM
    assert alloc.allocated_budget == expected_tokens
    assert alloc.candidate_budget == expected_tokens


def test_explicit_sentence_count_constraint():
    """Verify explicit sentence count constraints dictate output budget."""
    prompt = "Explain what quantum entanglement is in one sentence."
    alloc = OutputBudgetController.allocate_budget(prompt=prompt, mode="balanced")

    assert alloc.detected_constraint is not None
    assert "sentence" in alloc.detected_constraint.lower()
    # 1 * 35 + 64 headroom = 99 tokens
    assert alloc.allocated_budget == 35 + settings.OUTPUT_BUDGET_SAFETY_HEADROOM


def test_explicit_brevity_cues():
    """Verify qualitative brevity cues allocate conservative budgets."""
    prompt = "Give me a tl;dr of this architectural decision record."
    alloc = OutputBudgetController.allocate_budget(prompt=prompt, mode="balanced")

    assert alloc.detected_constraint is not None
    assert "tl;dr" in alloc.detected_constraint.lower()
    assert alloc.allocated_budget == 160


def test_user_requested_output_budget_override():
    """Verify API requested_output_budget takes precedence over heuristics."""
    prompt = "Write a comprehensive Python web scraper."
    alloc = OutputBudgetController.allocate_budget(
        prompt=prompt,
        mode="balanced",
        requested_output_budget=320,
    )

    assert alloc.requested_budget == 320
    assert alloc.allocated_budget == 320
    assert "override" in alloc.reason.lower()


def test_min_and_max_bounds_clamping():
    """Verify budgets are clamped within [MIN_TOKENS, MAX_TOKENS]."""
    # Extremely small override should clamp to MIN (64)
    alloc_min = OutputBudgetController.allocate_budget(
        prompt="A very short question.",
        mode="concise",
        requested_output_budget=10,
    )
    assert alloc_min.allocated_budget == settings.OUTPUT_BUDGET_MIN_TOKENS
    assert alloc_min.is_clamped_by_bounds is True

    # Extremely large override should clamp to MAX (2048)
    alloc_max = OutputBudgetController.allocate_budget(
        prompt="A detailed request.",
        mode="structure",
        requested_output_budget=10000,
    )
    assert alloc_max.allocated_budget == settings.OUTPUT_BUDGET_MAX_TOKENS
    assert alloc_max.is_clamped_by_bounds is True


def test_context_headroom_clamping():
    """Verify budget is clamped to available headroom when prompt approaches context limit."""
    # Simulate a prompt where remaining headroom in context window is only 150 tokens
    # max_num_ctx = 4096, prompt tokens ~ 3690, reserve = 256 -> available headroom = 150
    # A candidate budget of 512 should be clamped to 150
    prompt = "word " * 3600  # ~3600 chars -> ~900 tokens under 4 chars/token
    # Pass a low max_num_ctx to trigger headroom clamping
    alloc = OutputBudgetController.allocate_budget(
        prompt="Tell me a story about space exploration.",
        mode="balanced",
        max_num_ctx=600,  # Tight context limit
    )
    assert alloc.is_clamped_by_context is True
    assert alloc.allocated_budget < alloc.candidate_budget
    assert alloc.allocated_budget >= settings.OUTPUT_BUDGET_MIN_TOKENS


def test_context_overflow_when_headroom_below_min():
    """Verify ContextOverflowError HTTP 413 is raised when headroom < MIN_TOKENS."""
    # When available context cannot even fit OUTPUT_BUDGET_MIN_TOKENS (64)
    with pytest.raises(ContextOverflowError) as exc_info:
        OutputBudgetController.allocate_budget(
            prompt="Tell me a story about space exploration.",
            mode="balanced",
            max_num_ctx=300,  # Too tight to fit input + reserve(256) + min(64) = 320+
        )
    assert exc_info.value.status_code == 413
    assert "exceed the configured maximum context" in exc_info.value.message


def test_coordination_with_adaptive_context_manager():
    """Verify AdaptiveContextManager uses dynamic output budget in tier snapping."""
    sys_inst = "You are an expert prompt optimization assistant."
    prompt = "Original prompt to optimize:\n\nSummarize the report in 20 words."

    # Calculation with small output budget (90 tokens)
    budget_small = AdaptiveContextManager.calculate_budget(
        system_instruction=sys_inst,
        formatted_prompt=prompt,
        output_budget=90,
    )
    assert budget_small["predict_tokens"] == 90
    assert budget_small["output_budget_source"] == "dynamic_controller"

    # Calculation with default setting (512 tokens)
    budget_default = AdaptiveContextManager.calculate_budget(
        system_instruction=sys_inst,
        formatted_prompt=prompt,
    )
    assert budget_default["predict_tokens"] == settings.OLLAMA_NUM_PREDICT
    assert budget_default["output_budget_source"] == "static_setting"

    # Total required context is smaller by exactly (512 - 90) = 422 tokens
    assert budget_default["total_required_context"] - budget_small["total_required_context"] == (512 - 90)


@pytest.mark.asyncio
async def test_native_telemetry_done_reason_stop():
    """Verify done_reason='stop' correctly indicates natural completion and utilization."""
    fake_response = {
        "response": "Optimized concise prompt.",
        "done": True,
        "done_reason": "stop",
        "eval_count": 80,
        "eval_duration": 400_000_000,  # 400 ms
        "prompt_eval_count": 50,
        "prompt_eval_duration": 100_000_000,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        allocation = OutputBudgetAllocation(
            allocated_budget=200,
            requested_budget=None,
            policy="adaptive",
            reason="Adaptive test allocation",
            task_category="summarization",
            candidate_budget=200,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=allocation,
        )

        assert telemetry["done_reason"] == "stop"
        assert telemetry["hit_budget_limit"] is False
        assert telemetry["num_predict_limit"] == 200
        assert telemetry["eval_count"] == 80
        # 80 / 200 * 100 = 40.0%
        assert telemetry["budget_utilization_percent"] == 40.0


@pytest.mark.asyncio
async def test_native_telemetry_done_reason_length():
    """Verify done_reason='length' correctly marks hit_budget_limit=True."""
    fake_response = {
        "response": "Cut off generation that reached the ceiling...",
        "done": True,
        "done_reason": "length",
        "eval_count": 150,
        "eval_duration": 800_000_000,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        allocation = OutputBudgetAllocation(
            allocated_budget=150,
            requested_budget=None,
            policy="adaptive",
            reason="Adaptive test allocation",
            task_category="coding",
            candidate_budget=150,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=allocation,
        )

        assert telemetry["done_reason"] == "length"
        assert telemetry["hit_budget_limit"] is True
        assert telemetry["budget_utilization_percent"] == 100.0


@pytest.mark.asyncio
async def test_missing_telemetry_leaves_nulls():
    """Verify that when Ollama does not return native metrics, no fake numbers are fabricated."""
    fake_response = {
        "response": "Clean response without telemetry metadata.",
        "done": True,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        allocation = OutputBudgetAllocation(
            allocated_budget=256,
            requested_budget=None,
            policy="adaptive",
            reason="Adaptive test allocation",
            task_category="general",
            candidate_budget=256,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=allocation,
        )

        assert telemetry["done_reason"] is None
        assert telemetry["eval_count"] is None
        assert telemetry["hit_budget_limit"] is None
        assert telemetry["budget_utilization_percent"] is None
        assert telemetry["native_metrics_available"] is False


@pytest.mark.asyncio
async def test_end_to_end_optimize_endpoint_with_requested_budget(async_client: AsyncClient):
    """Verify POST /api/optimize respects requested_output_budget and returns output_budget telemetry."""
    fake_response = {
        "response": "Optimized output respecting requested token limits.",
        "done": True,
        "done_reason": "stop",
        "eval_count": 45,
        "eval_duration": 220_000_000,
        "prompt_eval_count": 30,
        "prompt_eval_duration": 80_000_000,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_response

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client):
        payload = {
            "prompt": "Create a function to calculate Fibonacci numbers.",
            "mode": "concise",
            "target_model": "gpt-4o",
            "requested_output_budget": 180,
        }
        response = await async_client.post("/api/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "output_budget" in data
        ob = data["output_budget"]
        assert ob is not None
        assert ob["requested_budget"] == 180
        assert ob["allocated_budget"] == 180
        assert ob["actual_generated_tokens"] == 45
        assert ob["done_reason"] == "stop"
        assert ob["hit_budget_limit"] is False
        assert ob["budget_utilization_percent"] == round((45 / 180) * 100, 1)


def test_config_validation_policy_names():
    """Verify OUTPUT_BUDGET_POLICY validator only accepts adaptive, fixed, conservative."""
    from app.core.config import Settings

    # Valid policies
    for pol in ["adaptive", "fixed", "conservative", "ADAPTIVE", " Conservative "]:
        s = Settings(OUTPUT_BUDGET_POLICY=pol)
        assert s.OUTPUT_BUDGET_POLICY == pol.strip().lower()

    # Invalid policy
    with pytest.raises(Exception):
        Settings(OUTPUT_BUDGET_POLICY="aggressive")

    with pytest.raises(Exception):
        Settings(OUTPUT_BUDGET_POLICY="dynamic")


def test_config_validation_baselines():
    """Verify OUTPUT_BUDGET_TASK_BASELINES requires all 8 categories and positive values."""
    from app.core.config import Settings

    # Missing a required category
    incomplete_baselines = {
        "coding": 512,
        "data_analysis": 384,
        "writing": 512,
        "summarization": 256,
        "research": 384,
        "extraction": 256,
        "planning": 512,
        # missing "general"
    }
    with pytest.raises(Exception) as exc:
        Settings(OUTPUT_BUDGET_TASK_BASELINES=incomplete_baselines)
    assert "missing required categories" in str(exc.value).lower()

    # Negative baseline value
    invalid_val_baselines = dict(incomplete_baselines)
    invalid_val_baselines["general"] = -50
    with pytest.raises(Exception) as exc:
        Settings(OUTPUT_BUDGET_TASK_BASELINES=invalid_val_baselines)
    assert "must be positive" in str(exc.value).lower()


def test_config_validation_mode_scaling():
    """Verify OUTPUT_BUDGET_MODE_SCALING requires all modes and valid multipliers."""
    from app.core.config import Settings

    # Missing a mode
    incomplete_scaling = {"concise": 0.75, "balanced": 1.0}
    with pytest.raises(Exception) as exc:
        Settings(OUTPUT_BUDGET_MODE_SCALING=incomplete_scaling)
    assert "missing required modes" in str(exc.value).lower()

    # Multiplier out of range (> 5.0)
    out_of_range = {"concise": 0.75, "balanced": 1.0, "structure": 6.5}
    with pytest.raises(Exception) as exc:
        Settings(OUTPUT_BUDGET_MODE_SCALING=out_of_range)
    assert "must be between" in str(exc.value).lower()


def test_policy_semantics_fixed():
    """Verify fixed policy always returns default tokens without heuristic modifications."""
    orig_policy = settings.OUTPUT_BUDGET_POLICY
    orig_enabled = settings.OUTPUT_BUDGET_ENABLED
    try:
        settings.OUTPUT_BUDGET_ENABLED = True
        settings.OUTPUT_BUDGET_POLICY = "fixed"

        # Explicit word constraint should be ignored under fixed policy
        alloc = OutputBudgetController.allocate_budget(
            prompt="Summarize the article in under 10 words.",
            mode="concise",
        )
        assert alloc.policy == "fixed"
        assert alloc.allocated_budget == settings.OUTPUT_BUDGET_DEFAULT_TOKENS
    finally:
        settings.OUTPUT_BUDGET_POLICY = orig_policy
        settings.OUTPUT_BUDGET_ENABLED = orig_enabled


def test_policy_semantics_conservative():
    """Verify conservative policy applies 0.75x baseline, 0 headroom on constraints, and capped max."""
    orig_policy = settings.OUTPUT_BUDGET_POLICY
    orig_enabled = settings.OUTPUT_BUDGET_ENABLED
    try:
        settings.OUTPUT_BUDGET_ENABLED = True
        settings.OUTPUT_BUDGET_POLICY = "conservative"

        # 1. Category baseline should be dampened by 0.75x
        # coding baseline is 512 -> 512 * 0.75 = 384
        alloc_code = OutputBudgetController.allocate_budget(
            prompt="Write a Python script to sort a list of numbers.",
            mode="balanced",
        )
        assert alloc_code.policy == "conservative"
        assert alloc_code.allocated_budget == int(math.ceil(512 * 0.75))

        # 2. Explicit word limit has 0 safety headroom (not +64)
        # 50 words * 1.33 = 67 tokens
        alloc_word = OutputBudgetController.allocate_budget(
            prompt="Summarize the key differences in under 50 words.",
            mode="balanced",
        )
        assert alloc_word.allocated_budget == math.ceil(50 * 1.33)
        assert "conservative" in alloc_word.reason.lower()
    finally:
        settings.OUTPUT_BUDGET_POLICY = orig_policy
        settings.OUTPUT_BUDGET_ENABLED = orig_enabled


@pytest.mark.asyncio
async def test_telemetry_done_reason_boundary_cases():
    """Verify hit_budget_limit relies strictly on native done_reason, not eval_count heuristic."""
    # Case A: eval_count >= num_predict, but done_reason="stop" -> MUST BE False
    fake_response_stop = {
        "response": "Finished naturally right at or above the ceiling limit.",
        "done": True,
        "done_reason": "stop",
        "eval_count": 200,
        "eval_duration": 400_000_000,
    }
    mock_resp_a = MagicMock(status_code=200)
    mock_resp_a.json.return_value = fake_response_stop
    mock_client_a = AsyncMock()
    mock_client_a.post.return_value = mock_resp_a
    mock_client_a.__aenter__.return_value = mock_client_a
    mock_client_a.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client_a):
        alloc = OutputBudgetAllocation(
            allocated_budget=200,
            requested_budget=None,
            policy="adaptive",
            reason="test",
            task_category="coding",
            candidate_budget=200,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=alloc,
        )
        assert telemetry["done_reason"] == "stop"
        assert telemetry["hit_budget_limit"] is False

    # Case B: eval_count < num_predict, but done_reason="length" -> MUST BE True
    fake_response_len = {
        "response": "Cut off due to internal length ceiling.",
        "done": True,
        "done_reason": "length",
        "eval_count": 195,
        "eval_duration": 400_000_000,
    }
    mock_resp_b = MagicMock(status_code=200)
    mock_resp_b.json.return_value = fake_response_len
    mock_client_b = AsyncMock()
    mock_client_b.post.return_value = mock_resp_b
    mock_client_b.__aenter__.return_value = mock_client_b
    mock_client_b.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client_b):
        alloc = OutputBudgetAllocation(
            allocated_budget=200,
            requested_budget=None,
            policy="adaptive",
            reason="test",
            task_category="coding",
            candidate_budget=200,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=alloc,
        )
        assert telemetry["done_reason"] == "length"
        assert telemetry["hit_budget_limit"] is True

    # Case C: done_reason is missing, even if eval_count >= num_predict -> MUST BE None
    fake_response_none = {
        "response": "Response without done_reason.",
        "done": True,
        "eval_count": 200,
    }
    mock_resp_c = MagicMock(status_code=200)
    mock_resp_c.json.return_value = fake_response_none
    mock_client_c = AsyncMock()
    mock_client_c.post.return_value = mock_resp_c
    mock_client_c.__aenter__.return_value = mock_client_c
    mock_client_c.__aexit__.return_value = None

    with patch("app.services.ollama_service.httpx.AsyncClient", return_value=mock_client_c):
        alloc = OutputBudgetAllocation(
            allocated_budget=200,
            requested_budget=None,
            policy="adaptive",
            reason="test",
            task_category="coding",
            candidate_budget=200,
        )
        _, _, telemetry = await OllamaService._execute_model_call(
            system_instruction="System test",
            prompt="Test prompt",
            output_budget=alloc,
        )
        assert telemetry["done_reason"] is None
        assert telemetry["hit_budget_limit"] is None


def test_task_confidence_derivation():
    """Verify task analyzer derives empirical confidence and returns None for general tasks."""
    # Definite coding task
    analysis_code = TaskAnalyzer.analyze("Write a Python script to parse JSON and return data.")
    assert analysis_code.task_category == "coding"
    assert analysis_code.task_confidence is not None
    assert 0.0 < analysis_code.task_confidence <= 1.0

    # Definite summarization task
    analysis_sum = TaskAnalyzer.analyze("Summarize the main points of this document.")
    assert analysis_sum.task_category == "summarization"
    assert analysis_sum.task_confidence is not None
    assert 0.0 < analysis_sum.task_confidence <= 1.0

    # Ambiguous / general prompt
    analysis_gen = TaskAnalyzer.analyze("Hello there, how are you today?")
    assert analysis_gen.task_category == "general"
    assert analysis_gen.task_confidence is None


def test_centralized_headroom_long_system_instruction():
    """Verify centralized headroom deducts system instruction tokens accurately."""
    system_inst = "A" * 2000  # ~500 tokens
    prompt = "A" * 1000  # ~250 tokens
    # max_num_ctx = 2048, reserve = 256

    headroom_info = AdaptiveContextManager.calculate_headroom(
        system_instruction=system_inst,
        prompt=prompt,
        max_num_ctx=2048,
        reserve_tokens=256,
    )
    assert headroom_info["estimated_input_tokens"] > 700
    assert headroom_info["reserve_tokens"] == 256
    assert headroom_info["available_headroom"] == 2048 - headroom_info["estimated_input_tokens"] - 256

