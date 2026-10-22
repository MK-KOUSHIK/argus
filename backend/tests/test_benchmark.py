"""Tests for Benchmark Dataset, Evaluator, and Reproducible Benchmark Runner."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import Response

from app.benchmarks.evaluator import BenchmarkEvaluator
from app.benchmarks.runner import BenchmarkRunner, BENCHMARK_MOCK_RESPONSES
from app.services.main_llm_service import MissingCredentialsError, MockLLMProvider, OpenAIProvider


def test_benchmark_dataset_contains_all_five_categories():
    """Verify benchmark dataset covers all 5 required task categories."""
    dataset = BenchmarkRunner.load_dataset()
    assert len(dataset) == 5

    categories = {case["category"] for case in dataset}
    expected_categories = {
        "simple_informational",
        "verbose_redundant",
        "coding_constraints",
        "structured_json_extraction",
        "preserve_unmodified",
    }
    assert categories == expected_categories

    for case in dataset:
        assert "id" in case
        assert "prompt" in case
        assert "expected_strategy" in case
        assert len(case["prompt"].strip()) > 0


def test_evaluator_coding_constraints_catches_pandas_violation():
    """Verify evaluator catches when LLM response imports or uses prohibited pandas."""
    test_case = {
        "prohibitions": ["pandas", "import pandas", "pd."],
        "required_terms": ["sales.csv", "monthly_report.csv"],
        "required_format": "code",
    }

    violating_output = (
        "```python\n"
        "import pandas as pd\n"
        "df = pd.read_csv('sales.csv')\n"
        "df.to_csv('monthly_report.csv')\n"
        "```"
    )

    eval_result = BenchmarkEvaluator.evaluate(
        original_prompt="Do not use pandas",
        optimized_prompt="Do not use pandas",
        llm_response=violating_output,
        test_case=test_case,
    )
    assert eval_result.output_prohibitions_respected is False
    assert eval_result.is_passing is False
    assert any("prohibited" in d.lower() for d in eval_result.details)


def test_evaluator_coding_constraints_passes_compliant_csv():
    """Verify evaluator passes compliant standard library code."""
    test_case = {
        "prohibitions": ["pandas", "import pandas", "pd."],
        "required_terms": ["sales.csv", "monthly_report.csv"],
        "required_format": "code",
    }

    compliant_output = BENCHMARK_MOCK_RESPONSES["case-3-coding-constraints"]
    eval_result = BenchmarkEvaluator.evaluate(
        original_prompt="Do not use pandas. Read sales.csv, save monthly_report.csv",
        optimized_prompt="Do not use pandas. Read sales.csv, save monthly_report.csv",
        llm_response=compliant_output,
        test_case=test_case,
    )
    assert eval_result.output_prohibitions_respected is True
    assert eval_result.output_required_terms_present is True
    assert eval_result.output_format_valid is True
    assert eval_result.is_passing is True


def test_evaluator_json_extraction_validates_schema():
    """Verify evaluator checks valid JSON parseability and required keys."""
    test_case = {
        "prohibitions": [],
        "required_terms": ["full_name", "email", "roles", "is_active"],
        "required_format": "json",
    }

    valid_json = (
        '{\n'
        '  "full_name": "Sarah Connor",\n'
        '  "email": "sconnor@resistance.net",\n'
        '  "roles": ["Leader"],\n'
        '  "is_active": true\n'
        '}'
    )
    eval_res = BenchmarkEvaluator.evaluate(
        original_prompt="Extract JSON",
        optimized_prompt="Extract JSON with keys full_name, email, roles, is_active",
        llm_response=valid_json,
        test_case=test_case,
    )
    assert eval_res.output_format_valid is True
    assert eval_res.is_passing is True

    # Bad JSON missing required key
    bad_json = '{"full_name": "Sarah Connor"}'
    bad_eval = BenchmarkEvaluator.evaluate(
        original_prompt="Extract JSON",
        optimized_prompt="Extract JSON",
        llm_response=bad_json,
        test_case=test_case,
    )
    assert bad_eval.output_format_valid is False
    assert bad_eval.is_passing is False


async def mock_execute_call(system_instruction: str, prompt: str):
    if "sales.csv" in prompt:
        return "Write a Python script to read sales.csv, calculate monthly revenue, and save monthly_report.csv. Do not use pandas.", 20
    elif "Sarah Connor" in prompt:
        return "Extract user profile into JSON with keys full_name, email, roles, is_active. Text: Contact Sarah Connor at sconnor@resistance.net. She is active as Team Lead and Security Auditor.", 20
    elif "TCP and UDP" in prompt:
        return "Summarize the architectural difference between TCP and UDP in 3 concise bullet points.", 15
    elif "DNS" in prompt:
        return "Explain step-by-step how DNS works.", 15
    return prompt, 10


@pytest.mark.asyncio
async def test_benchmark_runner_mock_execution():
    """Verify BenchmarkRunner executes all 5 cases in mock mode without paid API calls."""
    with patch("app.services.ollama_service.OllamaService._execute_model_call", side_effect=mock_execute_call):
        report = await BenchmarkRunner.run(live=False, model="gpt-4o")

        assert report.metadata["dataset_count"] == 5
        assert len(report.case_results) == 5
        assert report.summary.total_cases == 5
        assert report.summary.is_live is False
        assert report.summary.passed_cases == 5
        assert report.summary.baseline_prompt_tokens > 0
        assert report.summary.baseline_cost_usd > 0
        assert report.summary.constraint_retention_rate_percent == 100.0


@pytest.mark.asyncio
async def test_benchmark_runner_live_without_credentials_raises_error():
    """Verify requesting live benchmark without credentials raises MissingCredentialsError."""
    with patch("app.core.config.settings.OPENAI_API_KEY", None):
        with pytest.raises(MissingCredentialsError) as exc_info:
            await BenchmarkRunner.run(live=True, model="gpt-4o")
        assert "OPENAI_API_KEY" in exc_info.value.message


@pytest.mark.asyncio
async def test_api_benchmark_endpoints(async_client):
    """Verify /api/main-llm/benchmark/dataset and /api/main-llm/benchmark/run endpoints."""
    # Test dataset endpoint
    ds_resp = await async_client.get("/api/main-llm/benchmark/dataset")
    assert ds_resp.status_code == 200
    dataset = ds_resp.json()
    assert len(dataset) == 5

    # Test run endpoint (mock)
    with patch("app.services.ollama_service.OllamaService._execute_model_call", side_effect=mock_execute_call):
        run_resp = await async_client.post(
            "/api/main-llm/benchmark/run",
            json={"live": False, "model": "gpt-4o"},
        )
        assert run_resp.status_code == 200
        data = run_resp.json()
        assert "summary" in data
        assert "case_results" in data
        assert data["summary"]["total_cases"] == 5
