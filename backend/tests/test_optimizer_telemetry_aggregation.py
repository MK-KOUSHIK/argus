"""Tests for internal optimizer telemetry aggregation across one or multiple model calls."""

import json
from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_compression import (  # type: ignore[import-not-found] # pyright: ignore[reportMissingImports]
    NEW_SCHEMA_IDENTIFIER_KEYS,
    aggregate_optimizer_inference_telemetry,
    calculate_primary_rewrite_savings,
    calculate_total_optimizer_cost,
    extract_primary_rewrite_prompt_tokens,
    is_demonstrably_historical_single_call,
    is_new_schema_optimizer_record,
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


# =============================================================================
# REQUIRED REGRESSION TESTS (Covering All 9 Specification Cases)
# =============================================================================

def test_case_1_new_schema_complete_telemetry() -> None:
    """Case 1: New-schema record with complete first-call and aggregate telemetry.

    Both primary rewrite savings and total optimizer inference cost are evaluated,
    and their metrics correctly reflect first-call vs multi-call differences.
    """
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

    assert is_new_schema_optimizer_record(base_run["optimizer_inference"]) is True
    assert is_new_schema_optimizer_record(comp_run["optimizer_inference"]) is True

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


def test_case_2_critical_regression_new_schema_missing_first_call_does_not_consume_aggregate() -> None:
    """Case 2: CRITICAL REGRESSION TEST.

    Proves that a new-schema record with missing first-call telemetry but available aggregate telemetry
    does NOT consume aggregate multi-call tokens (1,820) as primary rewrite tokens.
    Instead, primary rewrite metric is marked 'unavailable'.
    """
    # Baseline has aggregate prompt_eval_count = 1820 across 2 calls, but first_call_prompt_eval_count is None
    base_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": None,  # Incomplete / missing first-call metric
            "prompt_eval_count": 1820,             # Total aggregate across 2 calls
            "call_count": 2,
            "has_complete_telemetry": True,
            "calls": [
                {"prompt_eval_count": 900, "eval_count": 80},
                {"prompt_eval_count": 920, "eval_count": 90},
            ],
        }
    }
    comp_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 750,
            "prompt_eval_count": 750,
            "call_count": 1,
            "has_complete_telemetry": True,
            "calls": [{"prompt_eval_count": 750, "eval_count": 70}],
        }
    }

    # Record is recognized as new-schema
    assert is_new_schema_optimizer_record(base_run["optimizer_inference"]) is True
    # Token extraction for primary rewrite MUST return None, NOT 1820
    assert extract_primary_rewrite_prompt_tokens(base_run["optimizer_inference"]) is None
    assert extract_primary_rewrite_prompt_tokens(base_run["optimizer_inference"]) != 1820

    primary = calculate_primary_rewrite_savings(base_run, comp_run)

    # Primary metric MUST be unavailable and MUST NOT report 1820 - 750 = 1070
    assert primary["status"] == "unavailable"
    assert primary["baseline_prompt_eval_count"] is None
    assert primary["tokens_saved"] is None
    assert primary["reduction_percent"] is None
    assert primary["baseline_prompt_eval_count"] != 1820

    # Total optimizer inference cost CAN still evaluate from aggregated telemetry
    total = calculate_total_optimizer_cost(base_run, comp_run)
    assert total["status"] == "evaluated"
    assert total["baseline_total_prompt_eval_count"] == 1820
    assert total["compressed_total_prompt_eval_count"] == 750
    assert total["tokens_saved"] == 1070


def test_case_3_new_schema_incomplete_first_call_telemetry() -> None:
    """Case 3: New-schema record with malformed or non-integer first-call telemetry.

    Ensures that bad data types or negative tokens in first_call_prompt_eval_count
    safely result in unavailable status without exceptions.
    """
    for bad_val in [None, "900", -50, {"tokens": 900}, [900]]:
        base_run = {
            "optimizer_inference": {
                "first_call_prompt_eval_count": bad_val,
                "prompt_eval_count": 1820,
                "call_count": 2,
                "has_complete_telemetry": False,
            }
        }
        comp_run = {
            "optimizer_inference": {
                "first_call_prompt_eval_count": 750,
                "prompt_eval_count": 750,
                "call_count": 1,
                "has_complete_telemetry": True,
            }
        }

        tok = extract_primary_rewrite_prompt_tokens(base_run["optimizer_inference"])
        assert tok is None

        primary = calculate_primary_rewrite_savings(base_run, comp_run)
        assert primary["status"] == "unavailable"
        assert primary["baseline_prompt_eval_count"] is None
        assert primary["tokens_saved"] is None


def test_case_4_historical_single_call_legitimate_fallback() -> None:
    """Case 4: Historical single-call record that legitimately supports the legacy fallback.

    Historical records contain authentic Argus run metadata ('validation_status', 'total_latency_ms')
    and no multi-call markers or new-schema fields. Primary rewrite savings must evaluate correctly.
    """
    # Authentic structure from docs/reports/compression_benchmark.json
    base_run = {
        "optimizer_inference": {
            "prompt_eval_count": 905,
            "validation_status": "fallback",
            "total_latency_ms": 4097.68,
            "selected_num_ctx": 4096,
            "prompt_eval_cached_count": 400,
            "cache_hit_ratio": None,
            "prompt_eval_duration_ms": 70.85,
            "eval_count": 231,
            "eval_duration_ms": 1739.56,
            "generation_tokens_per_second": 132.79,
            "done_reason": "length",
        }
    }
    comp_run = {
        "optimizer_inference": {
            "prompt_eval_count": 765,
            "validation_status": "passed",
            "total_latency_ms": 2500.0,
            "selected_num_ctx": 4096,
            "prompt_eval_cached_count": 300,
            "cache_hit_ratio": None,
            "prompt_eval_duration_ms": 50.0,
            "eval_count": 200,
            "eval_duration_ms": 1500.0,
            "generation_tokens_per_second": 133.33,
            "done_reason": "stop",
        }
    }

    assert is_demonstrably_historical_single_call(base_run["optimizer_inference"]) is True
    assert is_demonstrably_historical_single_call(comp_run["optimizer_inference"]) is True
    assert is_new_schema_optimizer_record(base_run["optimizer_inference"]) is False

    primary = calculate_primary_rewrite_savings(base_run, comp_run)
    total = calculate_total_optimizer_cost(base_run, comp_run)

    # Primary savings falls back cleanly to prompt_eval_count for historical comparability
    assert primary["status"] == "evaluated"
    assert primary["baseline_prompt_eval_count"] == 905
    assert primary["compressed_prompt_eval_count"] == 765
    assert primary["tokens_saved"] == 140
    assert primary["reduction_percent"] == 15.47

    # Total cost explicitly flags unavailable rather than inferring or fabricating missing retries
    assert total["status"] == "unavailable"
    assert total["tokens_saved"] is None
    assert total["baseline_total_prompt_eval_count"] is None
    assert total["compressed_total_prompt_eval_count"] is None
    assert total["has_complete_telemetry"] is False
    assert "not fabricated" in total["note"].lower()


def test_case_5_historical_multi_call_without_per_call_telemetry_is_unavailable() -> None:
    """Case 5: Historical multi-call record without reliable per-call telemetry.

    Proves that records with retry indicators (e.g. retry_count > 0, is_multi_call = True)
    are NOT assumed to be single-call, and primary savings correctly returns 'unavailable'.
    """
    for retry_marker in [
        {"retry_count": 1},
        {"retries": 2},
        {"retries_attempted": 1},
        {"is_multi_call": True},
        {"multi_call": True},
        {"total_calls": 2},
    ]:
        base_opt = {
            "prompt_eval_count": 1820,
            "validation_status": "passed",
            "total_latency_ms": 4500.0,
            **retry_marker,
        }
        comp_opt = {
            "prompt_eval_count": 765,
            "validation_status": "passed",
            "total_latency_ms": 2500.0,
        }

        # Multi-call indicator must prevent classification as historical single-call
        assert is_demonstrably_historical_single_call(base_opt) is False
        assert extract_primary_rewrite_prompt_tokens(base_opt) is None

        primary = calculate_primary_rewrite_savings(
            {"optimizer_inference": base_opt},
            {"optimizer_inference": comp_opt},
        )
        assert primary["status"] == "unavailable"
        assert primary["tokens_saved"] is None


def test_case_6_multi_call_retry_costs_included_in_total_cost() -> None:
    """Case 6: Multi-call record where retry costs are included in total optimizer inference cost.

    Baseline had 3 calls (500 + 600 + 700 = 1800 tokens).
    Compressed had 2 calls (400 + 450 = 850 tokens).
    Verifies that total optimizer inference cost encompasses all calls and retries.
    """
    base_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 500,
            "prompt_eval_count": 1800,
            "call_count": 3,
            "has_complete_telemetry": True,
            "calls": [
                {"prompt_eval_count": 500, "eval_count": 50},
                {"prompt_eval_count": 600, "eval_count": 60},
                {"prompt_eval_count": 700, "eval_count": 70},
            ],
        }
    }
    comp_run = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 400,
            "prompt_eval_count": 850,
            "call_count": 2,
            "has_complete_telemetry": True,
            "calls": [
                {"prompt_eval_count": 400, "eval_count": 40},
                {"prompt_eval_count": 450, "eval_count": 45},
            ],
        }
    }

    primary = calculate_primary_rewrite_savings(base_run, comp_run)
    total = calculate_total_optimizer_cost(base_run, comp_run)

    # Primary rewrite: 500 - 400 = 100 tokens saved (+20.00%)
    assert primary["status"] == "evaluated"
    assert primary["baseline_prompt_eval_count"] == 500
    assert primary["compressed_prompt_eval_count"] == 400
    assert primary["tokens_saved"] == 100
    assert primary["reduction_percent"] == 20.00

    # Total cost: 1800 - 850 = 950 tokens saved (+52.78%)
    assert total["status"] == "evaluated"
    assert total["baseline_total_prompt_eval_count"] == 1800
    assert total["compressed_total_prompt_eval_count"] == 850
    assert total["tokens_saved"] == 950
    assert total["reduction_percent"] == 52.78
    assert total["baseline_call_count"] == 3
    assert total["compressed_call_count"] == 2


def test_case_7_missing_or_incomplete_telemetry_never_fabricates_savings() -> None:
    """Case 7: Missing or incomplete telemetry never produces fabricated savings.

    Checks empty dicts, missing optimizer_inference, unverified bare dicts, and None values.
    """
    # 1. Bare unverified dict lacking historical context
    bare_base = {"optimizer_inference": {"prompt_eval_count": 1000}}
    bare_comp = {"optimizer_inference": {"prompt_eval_count": 800}}
    assert is_demonstrably_historical_single_call(bare_base["optimizer_inference"]) is False
    assert is_new_schema_optimizer_record(bare_base["optimizer_inference"]) is False
    assert extract_primary_rewrite_prompt_tokens(bare_base["optimizer_inference"]) is None

    p_bare = calculate_primary_rewrite_savings(bare_base, bare_comp)
    assert p_bare["status"] == "unavailable"
    assert p_bare["tokens_saved"] is None

    # 2. None token values
    none_base = {"optimizer_inference": {"first_call_prompt_eval_count": None, "call_count": 1}}
    none_comp = {"optimizer_inference": {"first_call_prompt_eval_count": 700, "call_count": 1}}
    p_none = calculate_primary_rewrite_savings(none_base, none_comp)
    assert p_none["status"] == "unavailable"
    assert p_none["tokens_saved"] is None

    # 3. Empty run dictionaries
    p_empty = calculate_primary_rewrite_savings({}, {})
    assert p_empty["status"] == "unavailable"
    assert p_empty["tokens_saved"] is None

    t_empty = calculate_total_optimizer_cost({}, {})
    assert t_empty["status"] == "unavailable"
    assert t_empty["tokens_saved"] is None


def test_case_8_zero_token_and_invalid_telemetry_edge_cases() -> None:
    """Case 8: Zero-token and invalid telemetry edge cases.

    Zero tokens do not trigger ZeroDivisionError. Negative numbers or invalid strings
    are safely rejected as invalid.
    """
    # Zero tokens: 0 baseline, 0 compressed
    zero_base = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 0,
            "prompt_eval_count": 0,
            "call_count": 1,
            "has_complete_telemetry": True,
            "calls": [{"prompt_eval_count": 0, "eval_count": 0}],
        }
    }
    zero_comp = {
        "optimizer_inference": {
            "first_call_prompt_eval_count": 0,
            "prompt_eval_count": 0,
            "call_count": 1,
            "has_complete_telemetry": True,
            "calls": [{"prompt_eval_count": 0, "eval_count": 0}],
        }
    }

    primary = calculate_primary_rewrite_savings(zero_base, zero_comp)
    assert primary["status"] == "evaluated"
    assert primary["baseline_prompt_eval_count"] == 0
    assert primary["compressed_prompt_eval_count"] == 0
    assert primary["tokens_saved"] == 0
    assert primary["reduction_percent"] == 0.0

    total = calculate_total_optimizer_cost(zero_base, zero_comp)
    assert total["status"] == "evaluated"
    assert total["tokens_saved"] == 0
    assert total["reduction_percent"] == 0.0

    # Negative tokens rejected
    neg_opt = {
        "first_call_prompt_eval_count": -50,
        "prompt_eval_count": -50,
        "call_count": 1,
    }
    assert extract_primary_rewrite_prompt_tokens(neg_opt) is None


def test_case_9_existing_historical_benchmark_records_retain_valid_metrics() -> None:
    """Case 9: Existing historical benchmark records retain valid, correctly classified metrics.

    Loads the actual docs/reports/compression_benchmark.json, runs calculations on all 10 trials,
    and proves:
      1. Every trial evaluates primary rewrite savings matching historical numbers.
      2. Exactly 1,214 total tokens saved across all trials (-13.40%).
      3. Total optimizer inference cost reports 'unavailable' without fabricating retries.
    """
    benchmark_json_path = REPO_ROOT / "docs/reports/compression_benchmark.json"
    assert benchmark_json_path.exists(), f"Benchmark JSON missing at {benchmark_json_path}"

    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trials = data.get("trials", [])
    assert len(trials) == 10, f"Expected 10 benchmark trials, got {len(trials)}"

    expected_savings_per_trial = [140, 140, 124, 124, 258, 258, 17, 17, 68, 68]
    total_baseline_tokens = 0
    total_compressed_tokens = 0
    total_tokens_saved = 0

    for idx, trial in enumerate(trials):
        base_run = trial["baseline"]
        comp_run = trial["compressed"]

        # Classification check
        assert is_demonstrably_historical_single_call(base_run["optimizer_inference"]) is True
        assert is_demonstrably_historical_single_call(comp_run["optimizer_inference"]) is True
        assert is_new_schema_optimizer_record(base_run["optimizer_inference"]) is False

        primary = calculate_primary_rewrite_savings(base_run, comp_run)
        total = calculate_total_optimizer_cost(base_run, comp_run)

        # Primary metric evaluated
        assert primary["status"] == "evaluated"
        assert primary["tokens_saved"] == expected_savings_per_trial[idx]

        base_tok = primary["baseline_prompt_eval_count"]
        comp_tok = primary["compressed_prompt_eval_count"]
        saved = primary["tokens_saved"]
        assert base_tok is not None and comp_tok is not None and saved is not None

        total_baseline_tokens += base_tok
        total_compressed_tokens += comp_tok
        total_tokens_saved += saved

        # Total cost metric unavailable (unfabricated)
        assert total["status"] == "unavailable"
        assert total["tokens_saved"] is None
        assert total["baseline_total_prompt_eval_count"] is None
        assert total["compressed_total_prompt_eval_count"] is None
        assert total["has_complete_telemetry"] is False
        assert "not fabricated" in total["note"].lower()

    # Aggregate historical totals
    assert total_baseline_tokens == 9058
    assert total_compressed_tokens == 7844
    assert total_tokens_saved == 1214
    reduction_pct = round((total_tokens_saved / total_baseline_tokens) * 100.0, 2)
    assert reduction_pct == 13.40


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
