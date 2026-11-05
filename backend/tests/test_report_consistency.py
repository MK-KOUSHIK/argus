"""Tests to enforce section-aware and row-aware consistency between benchmark reports and raw JSON data."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_report_consistency import (  # type: ignore[import-not-found] # pyright: ignore[reportMissingImports]
    DEFAULT_JSON,
    DEFAULT_MD,
    validate_report_consistency,
)


def test_compression_validation_report_matches_json() -> None:
    """Verify that every metric, table row, and aggregate statistic in the validation report matches the benchmark JSON."""
    assert DEFAULT_JSON.exists(), f"Benchmark JSON missing at {DEFAULT_JSON}"
    assert DEFAULT_MD.exists(), f"Validation report missing at {DEFAULT_MD}"

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, DEFAULT_MD)
    assert is_valid, f"Report discrepancies detected:\n" + "\n".join(errors)


def test_negative_value_changed_in_unrelated_section(tmp_path: Path) -> None:
    """Proves that changing a value in an unrelated section cannot make an incorrect metric pass."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    # Tamper with Section 7 (Statistical Aggregates) by replacing CS StdDev with 99.99%
    # Section 1 still has the correct '8.01%', but Section 7 validator must catch it
    tampered_md = real_md.replace("+28.71% / +31.18% / 8.01%", "+28.71% / +31.18% / 99.99%")
    assert tampered_md != real_md

    tampered_md_file = tmp_path / "tampered_sec7.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("Section 7 (Statistical Aggregates)" in err and "8.01" in err for err in errors)


def test_negative_value_changed_in_section_1(tmp_path: Path) -> None:
    """Proves that tampering with Section 1 Executive Summary fails even if Section 7 is correct."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    # Replace latency stdev in Executive Summary bullet with 9.999 ms
    tampered_md = real_md.replace("stdev: 0.495 ms", "stdev: 9.999 ms")
    assert tampered_md != real_md

    tampered_md_file = tmp_path / "tampered_sec1.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("Section 1 (Executive Summary)" in err and "0.495" in err for err in errors)


def test_negative_table_5_1_cell_tampered(tmp_path: Path) -> None:
    """Proves that altering a single cell in Table 5.1 (e.g. original chars) causes validation failure."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    # Replace 1,179 original chars with 9,999 in json_cluster_config R1
    tampered_md = real_md.replace(
        "| **`json_cluster_config`** (R1) | Baseline -> Compressed | 1,179 | 792 |",
        "| **`json_cluster_config`** (R1) | Baseline -> Compressed | 9,999 | 792 |",
    )
    assert tampered_md != real_md

    tampered_md_file = tmp_path / "tampered_5_1.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("Table 5.1" in err and "original chars" in err for err in errors)


def test_negative_table_5_2_status_tampered(tmp_path: Path) -> None:
    """Proves that altering a quality status in Table 5.2 causes validation failure."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    # Change preserved to regression for json_cluster_config R1
    tampered_md = real_md.replace(
        "| 0.15 ms | `preserved` | **Verified (Byte-Exact)** |",
        "| 0.15 ms | `regression` | **Verified (Byte-Exact)** |",
    )
    assert tampered_md != real_md

    tampered_md_file = tmp_path / "tampered_5_2.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("Table 5.2" in err and "quality status" in err for err in errors)


def test_negative_missing_or_duplicate_row(tmp_path: Path) -> None:
    """Proves that deleting or duplicating a trial row in Table 5.1 causes row count mismatch failure."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    # Delete row 1 of Table 5.1
    tampered_md = real_md.replace(
        "| **`json_cluster_config`** (R1) | Baseline -> Compressed | 1,179 | 792 | **+32.82%** | 1,179 / 1,179 | **+0.00%** | 1,179 / 1,179 | `fallback` |\n",
        "",
    )
    assert tampered_md != real_md

    tampered_md_file = tmp_path / "tampered_missing_row.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("Table 5.1 row count mismatch" in err for err in errors)


def test_negative_stale_metric_detected(tmp_path: Path) -> None:
    """Proves that the presence of obsolete stale metrics (e.g. 0.354 ms) causes immediate failure."""
    assert DEFAULT_MD.exists()
    real_md = DEFAULT_MD.read_text(encoding="utf-8")

    tampered_md = real_md + "\n<!-- Note: old latency was 0.354 ms -->\n"
    tampered_md_file = tmp_path / "tampered_stale.md"
    tampered_md_file.write_text(tampered_md, encoding="utf-8")

    is_valid, errors = validate_report_consistency(DEFAULT_JSON, tampered_md_file)
    assert not is_valid
    assert any("0.354 ms" in err for err in errors)
