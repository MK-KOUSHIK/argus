"""Tests to enforce consistency between benchmark reports and raw JSON data."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_report_consistency import DEFAULT_JSON, DEFAULT_MD, validate_report_consistency  # type: ignore[import-not-found] # pyright: ignore[reportMissingImports]


def test_compression_validation_report_matches_json() -> None:
    """Verify that every metric and aggregate statistic in the validation report matches the benchmark JSON."""
    assert DEFAULT_JSON.exists(), f"Benchmark JSON missing at {DEFAULT_JSON}"
    assert DEFAULT_MD.exists(), f"Validation report missing at {DEFAULT_MD}"

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, DEFAULT_MD)
    assert is_valid, f"Report discrepancies detected:\n" + "\n".join(errors)


def test_validator_detects_discrepancies(tmp_path: Path) -> None:
    """Verify that validate_report_consistency catches tampering or stale metrics."""
    dummy_json = tmp_path / "test.json"
    dummy_md = tmp_path / "test.md"

    data = {
        "aggregate_statistics": {
            "compression_stage_character_reduction": {
                "aggregate_percent": 31.07,
                "mean_percent": 28.71,
                "median_percent": 31.18,
                "stdev_percent": 8.01,
            },
            "compression_engine_latency_ms": {
                "mean_ms": 0.564,
                "median_ms": 0.4,
                "stdev_ms": 0.495,
            },
        },
        "trials": [],
    }

    dummy_json.write_text(json.dumps(data), encoding="utf-8")

    # MD has wrong stdev
    dummy_md.write_text("Reduction: 31.07% (mean 28.71%, median 31.18%, stdev 7.91%)\nLatency: 0.564 ms", encoding="utf-8")

    is_valid, errors = validate_report_consistency(dummy_json, dummy_md)
    assert not is_valid
    assert any("stdev_percent" in err or "7.91%" in err for err in errors)
