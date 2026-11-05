"""Tests for internal optimizer telemetry aggregation across one or multiple model calls."""

from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_compression import (  # type: ignore[import-not-found] # pyright: ignore[reportMissingImports]
    aggregate_optimizer_inference_telemetry,
    calculate_primary_rewrite_savings,
    calculate_total_optimizer_cost,
)


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


def test_primary_and_total_calculations_differ_when_retries_exist() -> None:
    """Proves that primary-call and total-call calculations differ correctly when retries exist."""
    # Baseline had a retry: call 1 (900 tokens) + call 2 (920 tokens) = 1,820 total
    base_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 900,
            "prompt_eval_count": 1820,
            "call_count": 2,
            "has_complete_telemetry": True,
            "calls": [
                {"prompt_eval_count": 900, "eval_count": 80},
                {"prompt_eval_count": 920, "eval_count": 90},
            ],
        }
    }
    # Compressed succeeded on first call: call 1 (750 tokens) = 750 total
    comp_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 750,
            "prompt_eval_count": 750,
            "call_count": 1,
            "has_complete_telemetry": True,
            "calls": [
                {"prompt_eval_count": 750, "eval_count": 70},
            ],
        }
    }

    primary = calculate_primary_rewrite_savings(base_run, comp_run)
    total = calculate_total_optimizer_cost(base_run, comp_run)

    # Primary call: 900 - 750 = 150 tokens saved (+16.67%)
    assert primary["metric_name"] == "primary_rewrite_call_savings"
    assert primary["measurement_scope"] == "first_call_only"
    assert primary["status"] == "evaluated"
    assert primary["baseline_prompt_eval_count"] == 900
    assert primary["compressed_prompt_eval_count"] == 750
    assert primary["tokens_saved"] == 150
    assert primary["reduction_percent"] == 16.67

    # Total cost: 1820 - 750 = 1070 tokens saved (+58.79%)
    assert total["metric_name"] == "total_optimizer_inference_cost"
    assert total["measurement_scope"] == "all_relevant_calls"
    assert total["status"] == "evaluated"
    assert total["baseline_total_prompt_eval_count"] == 1820
    assert total["compressed_total_prompt_eval_count"] == 750
    assert total["tokens_saved"] == 1070
    assert total["reduction_percent"] == 58.79
    assert total["baseline_call_count"] == 2
    assert total["compressed_call_count"] == 1
    assert total["has_complete_telemetry"] is True

    # Crucial distinction: primary savings != total savings
    assert primary["tokens_saved"] != total["tokens_saved"]
    assert primary["reduction_percent"] != total["reduction_percent"]


def test_total_optimizer_cost_historical_is_unavailable() -> None:
    """Historical single-call benchmark runs report total cost as unavailable and do not fabricate retry tokens."""
    # Historical data has only prompt_eval_count; no calls array, no call_count
    base_run = {
        "optimizer_inference": {
            "prompt_eval_count": 905,
        }
    }
    comp_run = {
        "optimizer_inference": {
            "prompt_eval_count": 765,
        }
    }

    primary = calculate_primary_rewrite_savings(base_run, comp_run)
    total = calculate_total_optimizer_cost(base_run, comp_run)

    # Primary savings falls back cleanly to prompt_eval_count for historical comparability
    assert primary["tokens_saved"] == 140
    assert primary["reduction_percent"] == 15.47
    assert primary["status"] == "evaluated"

    # Total cost explicitly flags unavailable rather than inferring or fabricating missing retries
    assert total["status"] == "unavailable"
    assert total["tokens_saved"] is None
    assert total["baseline_total_prompt_eval_count"] is None
    assert total["compressed_total_prompt_eval_count"] is None
    assert total["has_complete_telemetry"] is False
    assert "not fabricated" in total["note"].lower()


def test_total_optimizer_cost_incomplete_telemetry() -> None:
    """Incomplete telemetry in multi-call records is safely handled without fabricating counts."""
    base_run = {
        "optimizer_inference": {
            "call_count": 2,
            "has_complete_telemetry": False,
            "prompt_eval_count": 1000,
            "calls": [{"prompt_eval_count": 1000}, {"prompt_eval_count": None}],
        }
    }
    comp_run = {
        "optimizer_inference": {
            "call_count": 1,
            "has_complete_telemetry": True,
            "prompt_eval_count": 800,
            "calls": [{"prompt_eval_count": 800}],
        }
    }

    total = calculate_total_optimizer_cost(base_run, comp_run)

    assert total["has_complete_telemetry"] is False
    assert total["status"] == "incomplete_telemetry"
    assert total["baseline_call_count"] == 2
    assert total["compressed_call_count"] == 1
    assert total["tokens_saved"] == 200
    assert total["reduction_percent"] == 20.0


def test_primary_savings_missing_metrics_unavailable() -> None:
    """When prompt evaluation tokens are None, primary rewrite savings is marked unavailable."""
    base_run = {"optimizer_inference": {"prompt_eval_count": None}}
    comp_run = {"optimizer_inference": {"prompt_eval_count": 750}}

    primary = calculate_primary_rewrite_savings(base_run, comp_run)
    assert primary["status"] == "unavailable"
    assert primary["tokens_saved"] is None
    assert primary["reduction_percent"] is None
