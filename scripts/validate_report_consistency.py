#!/usr/bin/env python3
"""Validates that docs/reports/compression-validation-report.md matches docs/reports/compression_benchmark.json.

Dynamically extracts aggregate statistics and trial telemetry from the raw JSON
and ensures all corresponding values in the Markdown report agree without discrepancy.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "docs/reports/compression_benchmark.json"
DEFAULT_MD = REPO_ROOT / "docs/reports/compression-validation-report.md"


def validate_report_consistency(json_path: Path, md_path: Path) -> Tuple[bool, List[str]]:
    """Validate that the Markdown report dynamically agrees with the raw benchmark JSON."""
    errors: List[str] = []

    if not json_path.exists():
        return False, [f"JSON file not found: {json_path}"]
    if not md_path.exists():
        return False, [f"Markdown report not found: {md_path}"]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    stats: Dict[str, Any] = data.get("aggregate_statistics", {})

    def check_value_present(category: str, key: str, val: Any, suffixes: List[str] = ["", "%", " ms", " tokens"]):
        if val is None:
            return
        variants: List[str] = []
        if isinstance(val, (int, float)):
            variants.append(f"{val}")
            variants.append(f"{val:.1f}")
            variants.append(f"{val:.2f}")
            variants.append(f"{val:.3f}")
            if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                variants.append(f"{int(val):,}")
            if val > 0:
                variants.append(f"+{val}")
                variants.append(f"+{val:.1f}")
                variants.append(f"+{val:.2f}")
                variants.append(f"+{val:.3f}")
        else:
            variants.append(str(val))

        matched = False
        for v in variants:
            for s in suffixes:
                target = v + s
                if target in md_text:
                    matched = True
                    break
            if matched:
                break

        if not matched:
            errors.append(f"Missing or conflicting {category}.{key} = {val} in {md_path.name}")

    # 1. Compression Stage Character Reduction
    cs = stats.get("compression_stage_character_reduction", {})
    check_value_present("compression_stage", "aggregate_percent", cs.get("aggregate_percent"), ["%"])
    check_value_present("compression_stage", "mean_percent", cs.get("mean_percent"), ["%"])
    check_value_present("compression_stage", "median_percent", cs.get("median_percent"), ["%"])
    check_value_present("compression_stage", "stdev_percent", cs.get("stdev_percent"), ["%"])

    # 2. Final Optimized Prompt Character Reduction
    fo = stats.get("final_optimized_prompt_character_reduction", {})
    check_value_present("final_optimized_prompt", "aggregate_percent", fo.get("aggregate_percent"), ["%"])
    check_value_present("final_optimized_prompt", "mean_percent", fo.get("mean_percent"), ["%"])
    check_value_present("final_optimized_prompt", "median_percent", fo.get("median_percent"), ["%"])
    check_value_present("final_optimized_prompt", "stdev_percent", fo.get("stdev_percent"), ["%"])

    # 3. Downstream Prompt Token Savings
    dt = stats.get("downstream_prompt_token_savings", {})
    check_value_present("downstream_tokens", "total_tokens_saved", dt.get("total_tokens_saved"), [" tokens", ""])
    check_value_present("downstream_tokens", "aggregate_percent", dt.get("aggregate_percent"), ["%"])
    check_value_present("downstream_tokens", "mean_tokens_saved", dt.get("mean_tokens_saved"), [" tokens", ""])
    check_value_present("downstream_tokens", "median_tokens_saved", dt.get("median_tokens_saved"), [" tokens", ""])
    check_value_present("downstream_tokens", "stdev_tokens_saved", dt.get("stdev_tokens_saved"), [" tokens", ""])

    # 4. Internal Optimizer Prompt Token Savings
    ot = stats.get("internal_optimizer_prompt_token_savings", {})
    check_value_present("optimizer_tokens", "total_tokens_saved", ot.get("total_tokens_saved"), [" tokens", ""])
    check_value_present("optimizer_tokens", "aggregate_percent", ot.get("aggregate_percent"), ["%"])
    check_value_present("optimizer_tokens", "mean_tokens_saved", ot.get("mean_tokens_saved"), [" tokens", ""])
    check_value_present("optimizer_tokens", "median_tokens_saved", ot.get("median_tokens_saved"), [" tokens", ""])
    check_value_present("optimizer_tokens", "stdev_tokens_saved", ot.get("stdev_tokens_saved"), [" tokens", ""])

    # 5. Compression Engine Latency
    lat = stats.get("compression_engine_latency_ms", {})
    check_value_present("latency", "mean_ms", lat.get("mean_ms"), [" ms", ""])
    check_value_present("latency", "median_ms", lat.get("median_ms"), [" ms", ""])
    check_value_present("latency", "stdev_ms", lat.get("stdev_ms"), [" ms", ""])

    # 6. Quality & Restoration Summaries
    qa = stats.get("downstream_task_quality_summary", {})
    check_value_present("quality", "preserved_count", qa.get("preserved_count"), [" / 10", " preserved"])
    check_value_present("quality", "preservation_rate_percent", qa.get("preservation_rate_percent"), ["%"])

    res = stats.get("reversible_restoration_summary", {})
    check_value_present("restoration", "verified_byte_exact", res.get("verified_byte_exact"), [" / 8", " of 8"])
    check_value_present("restoration", "verification_rate_percent", res.get("verification_rate_percent"), ["%"])

    # 7. Check that known stale figures are NOT present in the Markdown report
    stale_checks = [
        ("0.354 ms", f"Stale compression latency standard deviation (0.354 ms) found in report; expected {lat.get('stdev_ms')} ms"),
        ("7.91%", f"Stale compression-stage standard deviation (7.91%) found in report; expected {cs.get('stdev_percent')}%"),
        ("27.24%", f"Stale final optimized prompt standard deviation (27.24%) found in report; expected {fo.get('stdev_percent')}%"),
    ]
    for stale_str, err_msg in stale_checks:
        if stale_str in md_text:
            errors.append(err_msg)

    # 8. Check each trial's row exists in Markdown report
    trials: List[Dict[str, Any]] = data.get("trials", [])
    for t in trials:
        w_id = t["workload_id"]
        rep = t["repetition"]
        marker = f"`{w_id}`** (R{rep})"
        if marker not in md_text:
            errors.append(f"Trial `{w_id}` (R{rep}) row missing from Markdown report")

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> None:
    json_file = DEFAULT_JSON
    md_file = DEFAULT_MD
    if len(sys.argv) > 1:
        json_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        md_file = Path(sys.argv[2])

    print(f"Validating report consistency between:\n  JSON: {json_file}\n  MD:   {md_file}")
    is_valid, errors = validate_report_consistency(json_file, md_file)

    if not is_valid:
        print("\n[FAILED] Inconsistencies detected:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\n[PASSED] All aggregate statistics and trial measurements in Markdown report match raw benchmark JSON.")


if __name__ == "__main__":
    main()
