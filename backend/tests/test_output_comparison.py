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
    LatencyBreakdown,
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
    """Verify downstream executions use the identical pinned model, parameters, and distinct prompts."""
    recorded_calls = []

    async def fake_execute_downstream(
        prompt, model, system_instruction=None, temperature=None, top_p=None, num_predict=None, num_ctx=None
    ):
        recorded_calls.append({
            "prompt": prompt,
            "model": model,
            "system_instruction": system_instruction,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        })
        is_orig = "Summarize this text" in prompt
        return (
            "Response to original prompt" if is_orig else "Response to optimized prompt",
            150.0 if is_orig else 95.0,
            {
                "model": model,
                "prompt_eval_count": 30 if is_orig else 12,
                "eval_count": 50 if is_orig else 40,
                "load_duration": 10_000_000,
                "prompt_eval_duration": 40_000_000 if is_orig else 18_000_000,
                "eval_duration": 110_000_000 if is_orig else 77_000_000,
                "total_duration": 160_000_000 if is_orig else 105_000_000,
                "done_reason": "stop",
                "queue_wait_ms": 2.5,
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
                warmup=False,
                trials=1,
            )
            result = await ComparisonService.compare(req)

            # Check that exactly 2 downstream calls occurred (warmup=False)
            assert len(recorded_calls) == 2

            # Call 1 (Execution A): raw original prompt, unaltered
            assert recorded_calls[0]["prompt"] == req.prompt
            assert recorded_calls[0]["model"] == "qwen2.5:1.5b"
            assert recorded_calls[0]["system_instruction"] == "You are an expert editor."
            assert recorded_calls[0]["temperature"] == 0.7
            assert recorded_calls[0]["top_p"] == 0.9
            assert recorded_calls[0]["num_predict"] == 256
            # Shared common num_ctx (Finding 4)
            assert recorded_calls[0]["num_ctx"] == recorded_calls[1]["num_ctx"]

            # Call 2 (Execution B): optimized prompt from optimizer
            assert recorded_calls[1]["prompt"] == mock_dummy_optimize_response.optimized_prompt
            assert recorded_calls[1]["model"] == "qwen2.5:1.5b"

            # Check execution responses
            assert result.original_execution.response == "Response to original prompt"
            assert result.optimized_execution.response == "Response to optimized prompt"
            assert result.downstream_model == "qwen2.5:1.5b"

            # Check telemetry mapping (Finding 2: prompt_eval_duration_ms, never TTFT)
            assert result.original_execution.prompt_tokens == 30
            assert result.original_execution.output_tokens == 50
            assert result.original_execution.total_tokens == 80
            assert result.original_execution.prompt_eval_duration_ms == 40.0
            assert result.original_execution.done_reason == "stop"

            # Check separated durations (Finding 1)
            assert result.original_execution.latency_breakdown is not None
            assert result.original_execution.latency_breakdown.model_load_duration_ms == 10.0
            assert result.original_execution.latency_breakdown.prompt_eval_duration_ms == 40.0
            assert result.original_execution.latency_breakdown.eval_duration_ms == 110.0
            assert result.original_execution.latency_breakdown.inference_duration_ms == 150.0  # 40 + 110
            assert result.original_execution.latency_breakdown.queue_wait_duration_ms == 2.5

            assert result.optimized_execution.prompt_tokens == 12
            assert result.optimized_execution.output_tokens == 40
            assert result.optimized_execution.total_tokens == 52
            assert result.optimized_execution.prompt_eval_duration_ms == 18.0

            # Check comparison metrics: ((30 - 12) / 30) * 100 = 60.0%
            assert result.metrics.prompt_tokens_saved == 18
            assert result.metrics.prompt_token_reduction_percent == 60.0
            assert result.metrics.output_tokens_diff == -10
            assert result.metrics.total_tokens_saved == 28
            assert result.metrics.total_token_reduction_percent == 35.0
            # Latency delta in pure inference duration: 95.0 - 150.0 = -55.0 ms
            assert result.metrics.latency_delta_ms == -55.0
            assert result.metrics.semantic_quality == "not_evaluated"
            # Single trial note (Finding 1)
            assert "single pair does not prove" in result.metrics.latency_note.lower()


@pytest.mark.asyncio
async def test_comparison_alternating_order_and_multi_trials(
    mock_installed_model, mock_dummy_optimize_response
):
    """Verify repeated paired trials execute with alternating order ([A, B], [B, A]) and model warm-up."""
    call_log = []

    async def fake_execute_downstream(prompt, model, **kwargs):
        call_log.append(prompt)
        return (
            "Response",
            100.0,
            {
                "prompt_eval_count": 20,
                "eval_count": 30,
                "load_duration": 5_000_000,
                "prompt_eval_duration": 25_000_000,
                "eval_duration": 75_000_000,
                "done_reason": "stop",
            },
        )

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response

            req = ComparisonRequest(
                prompt="Original prompt text here",
                downstream_model="qwen2.5:1.5b",
                warmup=True,
                trials=2,
            )
            result = await ComparisonService.compare(req)

            # Calls: 1 warmup ("warmup") + Trial 0 (A then B) + Trial 1 (B then A) = 5 calls
            assert len(call_log) == 5
            assert call_log[0] == "warmup"  # Warm-up ping
            assert call_log[1] == "Original prompt text here"  # Trial 0, order 1
            assert call_log[2] == mock_dummy_optimize_response.optimized_prompt  # Trial 0, order 2
            assert call_log[3] == mock_dummy_optimize_response.optimized_prompt  # Trial 1, order 1 (alternated!)
            assert call_log[4] == "Original prompt text here"  # Trial 1, order 2 (alternated!)

            # Check trial history attached to execution telemetry
            assert result.original_execution.trials is not None
            assert len(result.original_execution.trials) == 2
            assert result.original_execution.trials[0].execution_order == 1
            assert result.original_execution.trials[1].execution_order == 2

            assert result.optimized_execution.trials is not None
            assert len(result.optimized_execution.trials) == 2
            assert result.optimized_execution.trials[0].execution_order == 2
            assert result.optimized_execution.trials[1].execution_order == 1

            # Check aggregate statistics
            assert result.original_execution.aggregate_latency is not None
            assert result.original_execution.aggregate_latency.trials_count == 2
            assert result.original_execution.aggregate_latency.mean_inference_duration_ms == 100.0

            assert "2 paired trials" in result.metrics.latency_note.lower()


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
async def test_comparison_fails_if_optimizer_fails(mock_installed_model):
    """Verify that failure during optimization aborts before downstream trials run."""
    with patch.object(OllamaService, "optimize_prompt", side_effect=OllamaConnectionError("Daemon crashed")):
        with patch.object(OllamaService, "execute_downstream_call") as mock_exec:
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b")
            with pytest.raises(OllamaConnectionError):
                await ComparisonService.compare(req)

            # Neither trial should have executed
            mock_exec.assert_not_called()


@pytest.mark.asyncio
async def test_comparison_fails_if_trial_fails(mock_installed_model, mock_dummy_optimize_response):
    """Verify that failure during any trial call aborts with error and does not return a false success."""
    call_count = 0

    async def fake_execute_downstream(prompt, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return "Output", 100.0, {"prompt_eval_count": 10, "eval_count": 10}
        raise OllamaResponseError("Model crashed on second trial")

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b", warmup=False, trials=2)

            with pytest.raises(OllamaResponseError):
                await ComparisonService.compare(req)


def test_zero_denominator_yields_unavailable_percentage():
    """Verify Finding 5: When original token count is zero, percentage reduction must be None, not 0.0%."""
    exec_zero_a = ComparisonExecutionTelemetry(
        prompt="",
        response="output a",
        model="qwen2.5:1.5b",
        prompt_tokens=0,  # Zero denominator
        output_tokens=0,  # Zero denominator
        total_tokens=0,   # Zero denominator
        duration_ms=0.0,  # Zero denominator
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

    # All percentage metrics with zero denominator must be None (unavailable), NOT 0.0
    assert metrics_zero.prompt_token_reduction_percent is None
    assert metrics_zero.output_token_change_percent is None
    assert metrics_zero.total_token_reduction_percent is None
    assert metrics_zero.latency_reduction_percent is None


def test_metric_calculations_with_missing_telemetry():
    """Verify calculate_metrics handles None telemetry safely without raising errors."""
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
                "load_duration": 15_000_000,
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
                "warmup": False,
                "trials": 1,
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
            # Prompt eval duration renamed from TTFT (Finding 2)
            assert data["original_execution"]["prompt_eval_duration_ms"] == 25.0
            assert data["original_execution"]["latency_breakdown"]["model_load_duration_ms"] == 15.0

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

    # Trials out of range
    resp_trials = test_client.post("/api/compare", json={"prompt": "Test", "trials": 0})
    assert resp_trials.status_code == 422


def test_api_compare_timeout_returns_504(test_client, mock_installed_model, mock_dummy_optimize_response):
    """Verify OllamaTimeoutError during inference produces HTTP 504 Gateway Timeout."""
    with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = mock_dummy_optimize_response
        with patch.object(OllamaService, "execute_downstream_call", side_effect=OllamaTimeoutError("Model timed out")):
            payload = {"prompt": "Write a massive novel.", "downstream_model": "qwen2.5:1.5b", "warmup": False}
            resp = test_client.post("/api/compare", json=payload)
            assert resp.status_code == 504
            err_msg = resp.json().get("error", {}).get("message") or resp.json().get("detail", "")
            assert "timed out" in err_msg.lower()


def test_api_compare_connection_error_returns_503(test_client, mock_installed_model, mock_dummy_optimize_response):
    """Verify OllamaConnectionError produces HTTP 503 Service Unavailable."""
    with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
        mock_opt.return_value = mock_dummy_optimize_response
        with patch.object(OllamaService, "execute_downstream_call", side_effect=OllamaConnectionError("Cannot connect")):
            payload = {"prompt": "Hello", "downstream_model": "qwen2.5:1.5b", "warmup": False}
            resp = test_client.post("/api/compare", json=payload)
            assert resp.status_code == 503
            err_msg = resp.json().get("error", {}).get("message") or resp.json().get("detail", "")
            assert "cannot connect" in err_msg.lower()


# ---------------------------------------------------------------------------
# Targeted Regression Tests: Warm-Up, Settings Equality & Telemetry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_status_tracking_success_and_failure(
    mock_installed_model, mock_dummy_optimize_response
):
    """Verify warm-up status tracking: attempted, succeeded, error, and cold-start warning."""
    # 1. Warm-up succeeds
    async def fake_success(prompt, model, **kwargs):
        return ("Output", 50.0, {"prompt_eval_duration": 10_000_000, "eval_duration": 40_000_000})

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_success):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b", warmup=True, trials=1)
            res_succ = await ComparisonService.compare(req)

            assert res_succ.warmup_attempted is True
            assert res_succ.warmup_succeeded is True
            assert res_succ.warmup_error is None
            assert res_succ.warmup is True
            assert "cold-start" not in res_succ.metrics.latency_note.lower()

    # 2. Warm-up fails (inference fails on warmup prompt, but succeeds on actual prompts)
    call_idx = 0

    async def fake_warmup_fails(prompt, model, **kwargs):
        nonlocal call_idx
        call_idx += 1
        if prompt == "warmup":
            raise OllamaResponseError("Warm-up execution timed out")
        return ("Output", 60.0, {"prompt_eval_duration": 10_000_000, "eval_duration": 50_000_000})

    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_warmup_fails):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response
            req = ComparisonRequest(prompt="Test prompt", downstream_model="qwen2.5:1.5b", warmup=True, trials=1)
            res_fail = await ComparisonService.compare(req)

            # Warm-up was attempted, but failed
            assert res_fail.warmup_attempted is True
            assert res_fail.warmup_succeeded is False
            assert res_fail.warmup is False  # Never report as successful unless actually completed
            assert "Warm-up execution timed out" in (res_fail.warmup_error or "")
            # Latency note explicitly warns about cold-start effects
            assert "cold-start effects" in res_fail.metrics.latency_note.lower()

    # 3. Warm-up not attempted (warmup=False)
    with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_success):
        with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
            mock_opt.return_value = mock_dummy_optimize_response
            req_no_warmup = ComparisonRequest(prompt="Test", downstream_model="qwen2.5:1.5b", warmup=False, trials=1)
            res_disabled = await ComparisonService.compare(req_no_warmup)

            assert res_disabled.warmup_attempted is False
            assert res_disabled.warmup_succeeded is False
            assert res_disabled.warmup_error is None


@pytest.mark.asyncio
async def test_generation_settings_equality_with_differing_raw_budgets(
    mock_installed_model, mock_dummy_optimize_response
):
    """Verify that settings_identical is True when independent context budgets differ but effective common_num_ctx is shared."""
    from app.services.adaptive_context_manager import AdaptiveContextManager

    budget_call_count = 0

    def fake_calculate_budget(system_instruction, formatted_prompt, output_budget):
        nonlocal budget_call_count
        budget_call_count += 1
        # First call (original): 4096 tier. Second call (optimized): 2048 tier.
        if budget_call_count == 1:
            return {"selected_num_ctx": 4096, "effective_tier": 4096, "required_tokens": 3000}
        return {"selected_num_ctx": 2048, "effective_tier": 2048, "required_tokens": 1000}

    actual_calls_ctx = []

    async def fake_execute_downstream(prompt, model, num_ctx=None, **kwargs):
        if prompt != "warmup":
            actual_calls_ctx.append(num_ctx)
        return ("Output", 80.0, {"prompt_eval_duration": 15_000_000, "eval_duration": 65_000_000})

    with patch.object(AdaptiveContextManager, "calculate_budget", side_effect=fake_calculate_budget):
        with patch.object(OllamaService, "execute_downstream_call", side_effect=fake_execute_downstream):
            with patch.object(OllamaService, "optimize_prompt", new_callable=AsyncMock) as mock_opt:
                mock_opt.return_value = mock_dummy_optimize_response

                req = ComparisonRequest(
                    prompt="Long original prompt...",
                    downstream_model="qwen2.5:1.5b",
                    warmup=False,
                    trials=1,
                )
                res = await ComparisonService.compare(req)

                # Both calls received common_num_ctx = max(4096, 2048) = 4096
                assert actual_calls_ctx == [4096, 4096]
                assert res.effective_settings is not None
                assert res.effective_settings.num_ctx == 4096
                assert res.original_execution.effective_settings.num_ctx == 4096
                assert res.optimized_execution.effective_settings.num_ctx == 4096

                # Because both executions received identical effective parameters, generation_parameters_identical is True!
                assert res.metrics.generation_parameters_identical is True


def test_build_trial_missing_inference_telemetry():
    """Verify _build_trial() sets inference_duration_ms=None unless native measurements are available, while preserving wall_time_ms."""
    wall_ms = 250.75

    # Case 1: Missing prompt_eval_duration
    t1 = ComparisonService._build_trial(
        trial_idx=1,
        order=1,
        raw_response="resp",
        wall_duration_ms=wall_ms,
        telemetry={"eval_duration": 100_000_000},  # prompt_eval_duration missing
    )
    assert t1.latency.prompt_eval_duration_ms is None
    assert t1.latency.eval_duration_ms == 100.0
    assert t1.latency.inference_duration_ms is None  # Must NOT fall back to wall_ms
    assert t1.latency.wall_time_ms == wall_ms

    # Case 2: Missing eval_duration
    t2 = ComparisonService._build_trial(
        trial_idx=1,
        order=1,
        raw_response="resp",
        wall_duration_ms=wall_ms,
        telemetry={"prompt_eval_duration": 40_000_000},  # eval_duration missing
    )
    assert t2.latency.prompt_eval_duration_ms == 40.0
    assert t2.latency.eval_duration_ms is None
    assert t2.latency.inference_duration_ms is None  # Must NOT fall back to wall_ms
    assert t2.latency.wall_time_ms == wall_ms

    # Case 3: Missing both prompt_eval_duration and eval_duration
    t3 = ComparisonService._build_trial(
        trial_idx=1,
        order=1,
        raw_response="resp",
        wall_duration_ms=wall_ms,
        telemetry={},  # both missing
    )
    assert t3.latency.prompt_eval_duration_ms is None
    assert t3.latency.eval_duration_ms is None
    assert t3.latency.inference_duration_ms is None  # Must NOT fall back to wall_ms
    assert t3.latency.wall_time_ms == wall_ms

    # Case 4: Both present -> inference_duration_ms is calculated
    t4 = ComparisonService._build_trial(
        trial_idx=1,
        order=1,
        raw_response="resp",
        wall_duration_ms=wall_ms,
        telemetry={"prompt_eval_duration": 30_000_000, "eval_duration": 70_000_000},
    )
    assert t4.latency.prompt_eval_duration_ms == 30.0
    assert t4.latency.eval_duration_ms == 70.0
    assert t4.latency.inference_duration_ms == 100.0  # 30 + 70
    assert t4.latency.wall_time_ms == wall_ms

