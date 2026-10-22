"""Tests for Main LLM Integration, Provider Adapters, Tokenizer, and Comparative Execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from httpx import Response

from app.core.config import settings
from app.services.main_llm_service import (
    BaseLLMProvider,
    MainLLMService,
    MissingCredentialsError,
    MockLLMProvider,
    OpenAIProvider,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderAPIError,
    ProviderResponse,
    ProviderUsage,
)
from app.services.tokenizer_service import TokenizerService
from app.services.pricing_service import PricingService
from app.services.comparative_execution_service import ComparativeExecutionService


@pytest.fixture(autouse=True)
def reset_provider():
    """Ensure provider state is clean between tests."""
    MainLLMService.reset_provider()
    yield
    MainLLMService.reset_provider()


# ============================================================================
# Tokenizer & Pricing Tests
# ============================================================================

def test_tokenizer_tiktoken_accurate():
    """Verify tiktoken calculates exact tokens for known text on GPT-4o."""
    prompt = "Write a Python script that reads sales.csv, calculates monthly revenue, saves monthly_report.csv, and does not use pandas."
    result = TokenizerService.count_tokens(prompt, "gpt-4o")
    assert result.token_count > 0
    assert result.char_count == len(prompt)
    assert result.is_exact is True
    assert "tiktoken" in result.tokenizer_name


def test_tokenizer_empty_string():
    """Verify empty string returns 0 tokens."""
    result = TokenizerService.count_tokens("", "gpt-4o")
    assert result.token_count == 0
    assert result.char_count == 0
    assert result.is_exact is True


def test_pricing_actual_vs_estimated():
    """Verify pricing service strictly distinguishes estimated from actual costs."""
    # 1000 prompt tokens, 500 completion tokens on gpt-4o ($2.50 / $10.00 per 1M)
    actual = PricingService.calculate_actual_cost(1000, 500, "gpt-4o")
    assert actual.is_estimated is False
    assert actual.input_cost_usd == 0.0025
    assert actual.output_cost_usd == 0.0050
    assert actual.total_cost_usd == 0.0075

    # Estimated before execution
    estimated = PricingService.estimate_cost(1000, 500, "gpt-4o")
    assert estimated.is_estimated is True
    assert estimated.total_cost_usd == 0.0075

    # Comparison savings
    # Baseline (1000 prompt, 500 comp) vs Optimized (500 prompt, 500 comp)
    opt_cost = PricingService.calculate_actual_cost(500, 500, "gpt-4o")
    savings = PricingService.compare_costs(actual, opt_cost)
    assert savings.baseline_cost_usd == 0.0075
    assert opt_cost.total_cost_usd == 0.00625
    assert round(savings.cost_saved_usd, 6) == 0.00125
    assert savings.savings_percent > 0


# ============================================================================
# OpenAI Provider Adapter Tests (Mocked HTTP - Zero Paid Calls)
# ============================================================================

@pytest.mark.asyncio
async def test_openai_provider_missing_credentials():
    """Verify OpenAI provider raises clear MissingCredentialsError when no key is set."""
    provider = OpenAIProvider(api_key=None)
    with patch.object(settings, "OPENAI_API_KEY", None):
        assert provider.is_configured() is False
        with pytest.raises(MissingCredentialsError) as exc_info:
            await provider.generate("Hello world")
        assert "OPENAI_API_KEY is not set" in exc_info.value.message


def test_openai_status_never_exposes_key():
    """Verify provider status endpoint reports configuration without leaking key."""
    provider = OpenAIProvider(api_key="sk-secret-test-key-12345")
    status = provider.get_status()
    assert status.is_configured is True
    assert status.provider == "openai"
    assert "sk-secret" not in status.message
    assert "sk-secret" not in status.model_dump_json()


@pytest.mark.asyncio
async def test_openai_provider_success():
    """Verify successful response parsing and cost calculation."""
    fake_json = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "This is a test response."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 7,
            "total_tokens": 22,
        },
    }

    mock_resp = Response(status_code=200, json=fake_json)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        provider = OpenAIProvider(api_key="sk-test-key")
        result = await provider.generate("Test prompt", model="gpt-4o")

        assert result.content == "This is a test response."
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 7
        assert result.usage.total_tokens == 22
        assert result.cost.total_cost_usd > 0
        assert result.cost.is_estimated is False
        assert result.is_mock is False


@pytest.mark.asyncio
async def test_openai_provider_rate_limit():
    """Verify rate limit (HTTP 429) raises ProviderRateLimitError."""
    mock_resp = Response(
        status_code=429,
        json={"error": {"message": "Rate limit reached for requests", "type": "requests"}},
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        provider = OpenAIProvider(api_key="sk-test-key")
        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.generate("Test prompt")
        assert "rate limit" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_openai_provider_auth_failure():
    """Verify invalid key (HTTP 401) raises ProviderAuthenticationError."""
    mock_resp = Response(
        status_code=401,
        json={"error": {"message": "Incorrect API key provided"}},
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        provider = OpenAIProvider(api_key="sk-bad-key")
        with pytest.raises(ProviderAuthenticationError):
            await provider.generate("Test prompt")


@pytest.mark.asyncio
async def test_openai_provider_timeout():
    """Verify timeout raises ProviderTimeoutError."""
    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")):
        provider = OpenAIProvider(api_key="sk-test-key")
        with pytest.raises(ProviderTimeoutError):
            await provider.generate("Test prompt")


# ============================================================================
# Comparative Execution Service Tests
# ============================================================================

@pytest.mark.asyncio
async def test_comparative_execution_baseline_and_optimized():
    """Verify Baseline (Path A) vs Optimized (Path B) executes both paths and records metrics."""
    mock_provider = MockLLMProvider(
        canned_response="def calculate(): return 42",
        prompt_tokens=40,
        completion_tokens=20,
        model="gpt-4o",
    )

    prompt = "Please write a python script to calculate revenue. Make sure it is thorough and nice."
    
    with patch("app.services.ollama_service.OllamaService._execute_model_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("Write a Python script to calculate revenue.", 15)

        comparison = await ComparativeExecutionService.compare(
            prompt=prompt,
            mode="balanced",
            model="gpt-4o",
            provider=mock_provider,
        )

        assert comparison.original_prompt == prompt
        assert len(comparison.optimized_prompt) > 0
        assert comparison.baseline.path_name == "baseline"
        assert comparison.optimized.path_name == "optimized"
        assert comparison.baseline.prompt_used == prompt
        assert comparison.optimized.prompt_used == comparison.optimized_prompt
        assert comparison.is_mock is True
        assert comparison.constraints_preserved is True


# ============================================================================
# API Endpoint Integration Tests (Using TestClient via conftest)
# ============================================================================

@pytest.mark.asyncio
async def test_api_main_llm_status(async_client):
    """Test GET /api/main-llm/status."""
    response = await async_client.get("/api/main-llm/status")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    assert "is_configured" in data
    assert "model" in data


@pytest.mark.asyncio
async def test_api_main_llm_pricing(async_client):
    """Test GET /api/main-llm/pricing."""
    response = await async_client.get("/api/main-llm/pricing")
    assert response.status_code == 200
    data = response.json()
    assert "pricing" in data
    assert "gpt-4o" in data["pricing"]
    assert data["pricing"]["gpt-4o"]["input_cost_per_million"] == 2.50


@pytest.mark.asyncio
async def test_api_main_llm_compare_mock(async_client):
    """Test POST /api/main-llm/compare with mock provider active."""
    mock_provider = MockLLMProvider(
        canned_response="Tested answer",
        prompt_tokens=30,
        completion_tokens=15,
    )
    MainLLMService.set_provider(mock_provider)

    with patch("app.services.ollama_service.OllamaService._execute_model_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("Write python code.", 15)

        payload = {
            "prompt": "Hello please write some python code for me.",
            "mode": "concise",
            "target_model": "gpt-4o",
        }
        response = await async_client.post("/api/main-llm/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["original_prompt"] == payload["prompt"]
        assert "baseline" in data
        assert "optimized" in data
        assert "cost_comparison" in data


@pytest.mark.asyncio
async def test_api_main_llm_compare_unconfigured_error(async_client):
    """Test POST /api/main-llm/compare when provider is unconfigured."""
    MainLLMService.set_provider(OpenAIProvider(api_key=None))
    with patch.object(settings, "OPENAI_API_KEY", None):
        payload = {
            "prompt": "Test prompt without credentials",
            "mode": "balanced",
            "target_model": "gpt-4o",
        }
        response = await async_client.post("/api/main-llm/compare", json=payload)
        assert response.status_code == 400
        data = response.json()
        assert "OPENAI_API_KEY is not set" in data["error"]["message"]
