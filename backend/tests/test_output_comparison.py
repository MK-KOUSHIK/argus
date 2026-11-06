"""Tests for end-to-end original vs optimized LLM output comparison."""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.comparison import (
    ComparisonExecutionTelemetry,
    ComparisonMetrics,
    ComparisonRequest,
    ComparisonResponse,
)
from app.schemas.optimizer import (
    OptimizeRequest,
    OptimizeResponse,
    TokenEstimation,
)
from app.services.comparison_service import (
    ComparisonService,
    ComparisonServiceError,
    DownstreamModelNotFoundError,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
)
from app.services.routing.models import DiscoveredModel, ModelTier
from app.services.routing.registry import LocalModelRegistry


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_installed_model():
    """Provide a mock installed model in the registry."""
    model = DiscoveredModel(
        name="qwen2.5:1.5b",
        model="qwen2.5:1.5b",
        digest="sha-1.5b",
        size_bytes=986061892,
        modified_at="2026-10-21T00:00:00Z",
        parameter_size="1.5B",
        tier=ModelTier.SMALL,
        is_loaded=True,
    )
    with patch.object(LocalModelRegistry, "get_model", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = model
        yield model


@pytest.fixture
def mock_dummy_optimize_response():
    """Provide a mock OptimizeResponse."""
    return OptimizeResponse(
        original_prompt="Summarize this text in 3 bullet points with high detail and explain every single point thoroughly.",
        optimized_prompt="Summarize in 3 detailed bullet points:",
        mode="balanced",
        target_model="gemini-1-5-flash",
        task_category="summarization",
        strategy="compress",
        applied_strategy="compress",
        outcome="validated_rewrite",
        length_result="compressed",
        validation_status="passed",
        is_preserved=False,
        original_metrics=TokenEstimation(char_count=98, estimated_tokens=25),
        optimized_metrics=TokenEstimation(char_count=40, estimated_tokens=10),
        token_reduction_percent=60,
        token_change_percent=-60,
        token_savings=15,
        char_delta=-58,
        execution_time_ms=120,
        model="qwen2.5:1.5b",
        timestamp="2026-11-06T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Unit Tests: ComparisonService & Downstream Call Execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_comparison_service_pins_same_model_and_unaltered_prompts(
    mock_installed_model, mock_dummy_optimize_response
):
    """Verify both downstream executions use the identical pinned model, parameters, and distinct prompts."""
    recorded_calls = []

    async def fake_execute_downstream(prompt, model, system_instruction=None, temperature=None, top_p=None, num_predict=None):
        recorded_calls.append({
            "prompt": prompt,
            "model": model,
            "system_instruction": system_instruction,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        })
        is_first = len(recorded_calls) == 1
        return (
            "Response to original prompt" if is_first else "Response to optimized prompt",
            150.0 if is_first else 95.0,
            {
                "model": model,
                "prompt_eval_count": 30 if is_first else 12,
                "eval_count": 50 if is_first else 40,
                "prompt_eval_duration": 40_000_000 if is_first else 18_000_000,
                "eval_duration": 110_000_000 if is_first else 77_000_000,
                "total_duration": 150_000_000 if is_first else 95_000_000,
                "load_duration": 0,
                "done_reason": "stop",
            },
        )

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response

            req = ComparisonRequest(
                prompt="Summarize this text in 3 bullet points with high detail and explain every single point thoroughly.",
                downstream_model="qwen2.5:1.5b",
                system_instruction="You are an expert editor.",
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )
            result = await ComparisonService.compare(req)

            # Check that exactly 2 downstream calls occurred
            assert len(recorded_calls) == 2

            # Call 1 (Execution A): raw original prompt, unaltered
            assert recorded_calls[0]["prompt"] == req.prompt
            assert recorded_calls[0]["model"] == "qwen2.5:1.5b"
            assert recorded_calls[0]["system_instruction"] == "You are an expert editor."
            assert recorded_calls[0]["temperature"] == 0.7
            assert recorded_calls[0]["top_p"] == 0.9
            assert recorded_calls[0]["num_predict"] == 256

            # Call 2 (Execution B): optimized prompt from optimizer
            assert recorded_calls[1]["prompt"] == mock_dummy_optimize_response.optimized_prompt
            assert recorded_calls[1]["model"] == "qwen2.5:1.5b"  # EXACT same pinned model
            assert recorded_calls[1]["system_instruction"] == "You are an expert editor."
            assert recorded_calls[1]["temperature"] == 0.7
            assert recorded_calls[1]["top_p"] == 0.9
            assert recorded_calls[1]["num_predict"] == 256

            # Check execution responses
            assert result.original_execution.response == "Response to original prompt"
            assert result.optimized_execution.response == "Response to optimized prompt"
            assert result.downstream_model == "qwen2.5:1.5b"

            # Check telemetry mapping
            assert result.original_execution.prompt_tokens == 30
            assert result.original_execution.output_tokens == 50
            assert result.original_execution.total_tokens == 80
            assert result.original_execution.ttft_ms == 40.0
            assert result.original_execution.done_reason == "stop"

            assert result.optimized_execution.prompt_tokens == 12
            assert result.optimized_execution.output_tokens == 40
            assert result.optimized_execution.total_tokens == 52
            assert result.optimized_execution.ttft_ms == 18.0

            # Check comparison metrics: ((30 - 12) / 30) * 100 = 60.0%
            assert result.metrics.prompt_tokens_saved == 18
            assert result.metrics.prompt_token_reduction_percent == 60.0
            assert result.metrics.output_tokens_diff == -10
            assert result.metrics.total_tokens_saved == 28
            assert result.metrics.total_token_reduction_percent == 35.0
            assert result.metrics.latency_delta_ms == -55.0
            assert result.metrics.semantic_quality == "not_evaluated"


@pytest.mark.asyncio
async def test_comparison_rejects_uninstalled_model():
    """Verify that requesting an uninstalled model raises DownstreamModelNotFoundError with HTTP 400 semantics."""
    with patch.object(LocalModelRegistry, "get_model", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None  # Model not found

        req = ComparisonRequest(
            prompt="Hello world",
            downstream_model="nonexistent-model:99b",
        )
        with pytest.raises(DownstreamModelNotFoundError) as exc_info:
            await ComparisonService.compare(req)

        assert "nonexistent-model:99b" in str(exc_info.value)
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_comparison_fails_if_execution_a_fails(mock_installed_model):
    """Verify that failure during Execution A aborts immediately and does not run optimizer or Execution B."""
    with patch.object(OllamaService, "execute_downstream_call", side_effect=OllamaTimeoutError("Inference timeout")):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b")
            with pytest.raises(OllamaTimeoutError):
                await ComparisonService.compare(req)

            # Optimizer must NOT have been called
            mock_opt.assert_not_called()


@pytest.mark.asyncio
async def test_comparison_fails_if_optimizer_fails(mock_installed_model):
    """Verify that failure during optimization aborts and does not run Execution B."""
    calls = []

    async def fake_execute_downstream(prompt, model, **kwargs):
        calls.append(prompt)
        return "Original output", 100.0, {"prompt_eval_count": 10, "eval_count": 10}

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", side_effect=OllamaConnectionError("Daemon crashed")):
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b")
            with pytest.raises(OllamaConnectionError):
                await ComparisonService.compare(req)

            # Execution A ran, but Execution B did not
            assert len(calls) == 1


@pytest.mark.asyncio
async def test_comparison_fails_if_execution_b_fails(mock_installed_model, mock_dummy_optimize_response):
    """Verify that failure during Execution B aborts and does not return a partial success."""
    call_count = 0

    async def fake_execute_downstream(prompt, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Original output", 100.0, {"prompt_eval_count": 10, "eval_count": 10}
        raise OllamaResponseError("Model crashed on second run")

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b")

            with pytest.raises(OllamaResponseError):
                await ComparisonService.compare(req)


def test_metric_calculations_with_missing_and_zero_telemetry():
    """Verify calculate_metrics handles None, zero, and expansion cases without raising errors."""
    # Case 1: All telemetry missing (None)
    exec_none_a = ComparisonExecutionTelemetry(
        prompt="original",
        response="output a",
        model="qwen2.5:1.5b",
        prompt_tokens=None,
        output_tokens=None,
        total_tokens=None,
        duration_ms=None,
    )
    exec_none_b = ComparisonExecutionTelemetry(
        prompt="optimized",
        response="output b",
        model="qwen2.5:1.5b",
        prompt_tokens=None,
        output_tokens=None,
        total_tokens=None,
        duration_ms=None,
    )
    metrics_none = ComparisonService.calculate_metrics(exec_none_a, exec_none_b)
    assert metrics_none.prompt_tokens_saved is None
    assert metrics_none.prompt_token_reduction_percent is None
    assert metrics_none.output_tokens_diff is None
    assert metrics_none.output_token_change_percent is None
    assert metrics_none.total_tokens_saved is None
    assert metrics_none.total_token_reduction_percent is None
    assert metrics_none.latency_delta_ms is None
    assert metrics_none.latency_reduction_percent is None
    assert metrics_none.semantic_quality == "not_evaluated"

    # Case 2: Zero tokens in original prompt
    exec_zero_a = ComparisonExecutionTelemetry(
        prompt="original",
        response="output a",
        model="qwen2.5:1.5b",
        prompt_tokens=0,
        output_tokens=0,
        total_tokens=0,
        duration_ms=0.0,
    )
    exec_zero_b = ComparisonExecutionTelemetry(
        prompt="optimized",
        response="output b",
        model="qwen2.5:1.5b",
        prompt_tokens=10,
        output_tokens=10,
        total_tokens=20,
        duration_ms=50.0,
    )
    metrics_zero = ComparisonService.calculate_metrics(exec_zero_a, exec_zero_b)
    assert metrics_zero.prompt_token_reduction_percent == 0.0
    assert metrics_zero.output_token_change_percent == 0.0
    assert metrics_zero.total_token_reduction_percent == 0.0
    assert metrics_zero.latency_reduction_percent == 0.0


# ---------------------------------------------------------------------------
# API Route Integration Tests (POST /api/compare)
# ---------------------------------------------------------------------------


def test_api_compare_success(test_client, mock_installed_model, mock_dummy_optimize_response):
    """Verify POST /api/compare returns HTTP 200 with complete comparison schema."""
    async def fake_execute_downstream(prompt, model, **kwargs):
        return (
            "Generated answer",
            120.0,
            {
                "model": model,
                "prompt_eval_count": 20,
                "eval_count": 35,
                "prompt_eval_duration": 25_000_000,
                "eval_duration": 95_000_000,
                "done_reason": "stop",
            },
        )

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response

            payload = {
                "prompt": "Explain the difference between TCP and UDP in 2 sentences.",
                "downstream_model": "qwen2.5:1.5b",
                "mode": "concise",
                "temperature": 0.5,
                "max_tokens": 128,
            }
            resp = test_client.post("/api/compare", json=payload)

            assert resp.status_code == 200
            data = resp.json()

            assert data["downstream_model"] == "qwen2.5:1.5b"
            assert "original_execution" in data
            assert "optimized_execution" in data
            assert "metrics" in data
            assert "optimizer_details" in data

            assert data["original_execution"]["prompt"] == payload["prompt"]
            assert data["original_execution"]["response"] == "Generated answer"
            assert data["original_execution"]["prompt_tokens"] == 20
            assert data["original_execution"]["output_tokens"] == 35
            assert data["original_execution"]["ttft_ms"] == 25.0

            assert data["metrics"]["semantic_quality"] == "not_evaluated"


def test_api_compare_uninstalled_model_returns_400(test_client):
    """Verify POST /api/compare returns HTTP 400 when model is not installed."""
    with patch.object(LocalModelRegistry, "get_model", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None

        payload = {
            "prompt": "Explain gravity.",
            "downstream_model": "nonexistent:model",
        }
        resp = test_client.post("/api/compare", json=payload)
        assert resp.status_code == 400
        err_msg = resp.json().get("error", {}).get("message") or resp.json().get("detail", "")
        assert "not installed or available" in err_msg.lower()


def test_api_compare_validation_errors(test_client):
    """Verify invalid payloads (empty prompt, invalid parameters) return HTTP 422."""
    # Empty prompt
    resp_empty = test_client.post("/api/compare", json={"prompt": "   "})
    assert resp_empty.status_code == 422

    # Negative temperature
    resp_temp = test_client.post("/api/compare", json={"prompt": "Test", "temperature": -0.5})
    assert resp_temp.status_code == 422

    # Top_p out of range
    resp_topp = test_client.post("/api/compare", json={"prompt": "Test", "top_p": 1.5})
    assert resp_topp.status_code == 422

    # Negative max_tokens
    resp_max = test_client.post("/api/compare", json={"prompt": "Test", "max_tokens": 0})
    assert resp_max.status_code == 422


def test_api_compare_timeout_returns_504(test_client, mock_installed_model):
    """Verify OllamaTimeoutError during inference produces HTTP 504 Gateway Timeout."""
    with patch.object(OllamaService, "execute_downstream_call", side_effect=OllamaTimeoutError("Model timed out")):
        payload = {"prompt": "Write a massive novel.", "downstream_model": "qwen2.5:1.5b"}
        resp = test_client.post("/api/compare", json=payload)
        assert resp.status_code == 504
        err_msg = resp.json().get("error", {}).get("message") or resp.json().get("detail", "")
        assert "timed out" in err_msg.lower()


def test_api_compare_connection_error_returns_503(test_client, mock_installed_model):
    """Verify OllamaConnectionError produces HTTP 503 Service Unavailable."""
    with patch.object(OllamaService, "execute_downstream_call", side_effect=OllamaConnectionError("Cannot connect")):
        payload = {"prompt": "Hello", "downstream_model": "qwen2.5:1.5b"}
        resp = test_client.post("/api/compare", json=payload)
        assert resp.status_code == 503
        err_msg = resp.json().get("error", {}).get("message") or resp.json().get("detail", "")
        assert "cannot connect" in err_msg.lower()
