"""Tests for internal optimizer telemetry aggregation across one or multiple model calls."""

from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_compression import aggregate_optimizer_inference_telemetry  # type: ignore[import-not-found] # pyright: ignore[reportMissingImports]


def test_aggregation_single_call() -> None:
    """A single call preserves exact metrics without alteration."""
    call_1: Dict[str, Any] = {
        "prompt_eval_count": 905,
        "prompt_eval_cached_count": 450,
        "eval_count": 120,
        "prompt_eval_duration_ns": 45_000_000,
        "eval_duration_ns": 150_000_000,
        "cache_hit_ratio": 0.4972,
        "generation_tokens_per_second": 800.0,
        "selected_num_ctx": 4096,
        "done_reason": "stop",
    }
    telemetry = {"calls": [call_1]}

    res = aggregate_optimizer_inference_telemetry(telemetry)

    assert res["call_count"] == 1
    assert res["has_complete_telemetry"] is True
    assert res["telemetry_source"] == "complete_all_calls"
    assert res["prompt_eval_count"] == 905
    assert res["prompt_eval_cached_count"] == 450
    assert res["eval_count"] == 120
    assert res["prompt_eval_duration_ms"] == 45.0
    assert res["eval_duration_ms"] == 150.0
    assert res["cache_hit_ratio"] == 0.4972
    assert res["generation_tokens_per_second"] == 800.0
    assert res["selected_num_ctx"] == 4096
    assert res["done_reason"] == "stop"
    assert res["first_call_prompt_eval_count"] == 905
    assert res["first_call_eval_count"] == 120
    assert len(res["calls"]) == 1


def test_aggregation_multiple_calls_retries() -> None:
    """Multiple calls (initial + retry) aggregate tokens and durations without double-counting."""
    call_1: Dict[str, Any] = {
        "prompt_eval_count": 800,
        "prompt_eval_cached_count": 200,
        "eval_count": 100,
        "prompt_eval_duration_ns": 40_000_000,
        "eval_duration_ns": 100_000_000,
        "selected_num_ctx": 4096,
        "done_reason": "stop",
    }
    call_2: Dict[str, Any] = {
        "prompt_eval_count": 850,
        "prompt_eval_cached_count": 400,
        "eval_count": 150,
        "prompt_eval_duration_ns": 50_000_000,
        "eval_duration_ns": 150_000_000,
        "selected_num_ctx": 4096,
        "done_reason": "stop",
    }
    telemetry = {"calls": [call_1, call_2]}

    res = aggregate_optimizer_inference_telemetry(telemetry)

    assert res["call_count"] == 2
    assert res["has_complete_telemetry"] is True
    assert res["telemetry_source"] == "complete_all_calls"
    # Aggregated sums
    assert res["prompt_eval_count"] == 1650
    assert res["prompt_eval_cached_count"] == 600
    assert res["eval_count"] == 250
    assert res["prompt_eval_duration_ms"] == 90.0
    assert res["eval_duration_ms"] == 250.0
    # Aggregate cache hit ratio = 600 / 1650
    assert res["cache_hit_ratio"] == round(600 / 1650.0, 4)
    # Overall tokens per second: 250 tokens / 0.25 sec = 1000.0
    assert res["generation_tokens_per_second"] == 1000.0
    assert res["selected_num_ctx"] == 4096
    assert res["done_reason"] == "stop"
    # First call isolation preserved for traceability
    assert res["first_call_prompt_eval_count"] == 800
    assert res["first_call_eval_count"] == 100
    assert len(res["calls"]) == 2
    assert res["calls"][0] == call_1
    assert res["calls"][1] == call_2


def test_aggregation_missing_telemetry() -> None:
    """Missing or empty telemetry dict safely returns unavailable defaults."""
    for empty_input in [None, {}, {"calls": None}, {"calls": []}, {"calls": ["invalid_string"]}]:
        res = aggregate_optimizer_inference_telemetry(empty_input)
        assert res["call_count"] == 0
        assert res["has_complete_telemetry"] is False
        assert res["telemetry_source"] == "unavailable"
        assert res["prompt_eval_count"] is None
        assert res["eval_count"] is None
        assert res["prompt_eval_duration_ms"] is None
        assert res["eval_duration_ms"] is None
        assert res["calls"] == []


def test_aggregation_incomplete_call_records() -> None:
    """Calls with missing fields, None tokens, or non-dict items are handled safely."""
    call_1: Dict[str, Any] = {
        "prompt_eval_count": 500,
        "eval_count": 50,
        "prompt_eval_duration_ns": 20_000_000,
        "eval_duration_ns": 50_000_000,
    }
    # call 2 is missing native metrics (prompt_eval_count is None)
    call_2: Dict[str, Any] = {
        "prompt_eval_count": None,
        "eval_count": None,
        "prompt_eval_duration_ns": None,
        "eval_duration_ns": None,
    }
    telemetry = {"calls": [call_1, call_2, None]}  # Includes a None element

    res = aggregate_optimizer_inference_telemetry(telemetry)

    assert res["call_count"] == 2
    assert res["has_complete_telemetry"] is False
    assert res["telemetry_source"] == "first_call_only"
    assert res["prompt_eval_count"] == 500
    assert res["eval_count"] == 50
    assert res["first_call_prompt_eval_count"] == 500
    assert len(res["calls"]) == 2


def test_aggregation_zero_and_negative_durations() -> None:
    """Calls with 0 or negative durations do not cause ZeroDivisionError."""
    call = {
        "prompt_eval_count": 100,
        "eval_count": 10,
        "prompt_eval_duration_ns": 0,
        "eval_duration_ns": 0,
    }
    res = aggregate_optimizer_inference_telemetry({"calls": [call]})
    assert res["prompt_eval_count"] == 100
    assert res["eval_count"] == 10
    assert res["generation_tokens_per_second"] is None
