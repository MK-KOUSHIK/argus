#!/usr/bin/env python3
"""Validates that docs/reports/compression-validation-report.md matches docs/reports/compression_benchmark.json.

Performs deep section-aware and row-aware consistency validation:
- Dynamically extracts aggregate statistics and trial telemetry from raw JSON (source of truth).
- Validates executive summary metrics strictly within Section 1.
- Validates statistical aggregates table row-by-row strictly within Section 7.
- Parses and validates each of the 10 trial rows in Table 5.1 (character stages) against JSON.
- Parses and validates each of the 10 trial rows in Table 5.2 (tokens, latency, quality, restoration) against JSON.
- Enforces exact row counts, verifies absence of stale metrics, and handles numeric formatting (commas, precision).
"""

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = REPO_ROOT / "docs/reports/compression_benchmark.json"
DEFAULT_MD = REPO_ROOT / "docs/reports/compression-validation-report.md"


def clean_markdown_val(s: str) -> str:
    """Strip markdown formatting, commas, backticks, and bold markers."""
    return s.replace("**", "").replace("`", "").replace(",", "").strip()


def parse_markdown_table_rows(table_text: str) -> List[List[str]]:
    """Parse markdown table rows into a list of cell string lists."""
    rows: List[List[str]] = []
    for line in table_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if not cells or all(c == "" or set(c).issubset({"-", ":", " "}) for c in cells):
            continue
        # Skip header rows
        first_lower = cells[0].lower()
        if any(h in first_lower for h in ["workload id", "metric category", "component"]):
            continue
        rows.append(cells)
    return rows


def extract_report_sections(md_text: str) -> Dict[str, str]:
    """Extract distinct sections from the markdown report."""
    sections: Dict[str, str] = {}

    # 1. Executive Summary: Between '## 1. Executive Summary' and '## 2.'
    m_exec = re.search(r"## 1\. Executive Summary\s+(.*?)(?=\n## 2\.|\Z)", md_text, re.DOTALL)
    sections["executive_summary"] = m_exec.group(1) if m_exec else ""

    # 2. Table 5.1: Under '### 5.1 Four-Stage Character Measurements'
    m_51 = re.search(
        r"### 5\.1 Four-Stage Character Measurements\s+(.*?)(?=\n### 5\.2|\n## |\Z)", md_text, re.DOTALL
    )
    sections["table_5_1"] = m_51.group(1) if m_51 else ""

    # 3. Table 5.2: Under '### 5.2 Token Telemetry, Engine Latency & Answer Quality'
    m_52 = re.search(
        r"### 5\.2 Token Telemetry, Engine Latency & Answer Quality\s+(.*?)(?=\n---|\n## 6\.|\Z)",
        md_text,
        re.DOTALL,
    )
    sections["table_5_2"] = m_52.group(1) if m_52 else ""

    # 4. Statistical Aggregates: Under '## 7. Statistical Aggregates'
    m_7 = re.search(r"## 7\. Statistical Aggregates\s+(.*?)(?=\n---|\n## 8\.|\Z)", md_text, re.DOTALL)
    sections["statistical_aggregates"] = m_7.group(1) if m_7 else ""

    return sections


def check_value_in_text(
    section_name: str, metric_label: str, target_val: Any, text: str, errors: List[str], suffixes: Optional[List[str]] = None
) -> None:
    """Verify that target_val (in various numeric formats) exists in the specified section text."""
    if target_val is None:
        return
    if suffixes is None:
        suffixes = ["", "%", " ms", " tokens"]

    variants: List[str] = []
    if isinstance(target_val, (int, float)):
        variants.append(f"{target_val}")
        variants.append(f"{target_val:.1f}")
        variants.append(f"{target_val:.2f}")
        variants.append(f"{target_val:.3f}")
        if isinstance(target_val, int) or (isinstance(target_val, float) and target_val.is_integer()):
            variants.append(f"{int(target_val):,}")
            variants.append(f"{int(target_val)}")
        if target_val > 0:
            variants.append(f"+{target_val}")
            variants.append(f"+{target_val:.1f}")
            variants.append(f"+{target_val:.2f}")
            variants.append(f"+{target_val:.3f}")
    else:
        variants.append(str(target_val))

    matched = False
    for v in variants:
        for s in suffixes:
            target = v + s
            if target in text:
                matched = True
                break
        if matched:
            break

    if not matched:
        errors.append(f"Missing {metric_label} ({target_val}) in report section '{section_name}'")


def validate_table_5_1(table_text: str, trials: List[Dict[str, Any]], errors: List[str]) -> None:
    """Validate each of the 10 trial rows in Table 5.1 against raw JSON trial data."""
    rows = parse_markdown_table_rows(table_text)
    if len(rows) != len(trials):
        errors.append(f"Table 5.1 row count mismatch: expected {len(trials)} rows, found {len(rows)}")
        return

    for idx, (trial, cells) in enumerate(zip(trials, rows)):
        w_id = trial["workload_id"]
        rep = trial["repetition"]
        row_id_cell = cells[0]

        if w_id not in row_id_cell or f"R{rep}" not in row_id_cell:
            errors.append(f"Table 5.1 row {idx + 1} ID mismatch: expected {w_id} (R{rep}), found {row_id_cell}")

        smc = trial["comparison"]["stage_measurements_comparison"]

        # Original chars
        exp_orig = smc["original_workload_prompt_chars"]
        try:
            act_orig = int(clean_markdown_val(cells[2]))
            if act_orig != exp_orig:
                errors.append(f"Table 5.1 {w_id} R{rep} original chars: expected {exp_orig}, got {act_orig}")
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable original chars cell: '{cells[2]}'")

        # Comp-stage chars
        exp_comp = smc["compression_stage"]["compressed_chars"]
        try:
            act_comp = int(clean_markdown_val(cells[3]))
            if act_comp != exp_comp:
                errors.append(f"Table 5.1 {w_id} R{rep} comp-stage chars: expected {exp_comp}, got {act_comp}")
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable comp-stage chars cell: '{cells[3]}'")

        # Comp-stage reduction %
        exp_comp_red = smc["compression_stage"]["reduction_percent"]
        try:
            act_comp_red = float(clean_markdown_val(cells[4]).replace("%", "").replace("+", ""))
            if abs(act_comp_red - exp_comp_red) > 0.05:
                errors.append(f"Table 5.1 {w_id} R{rep} comp reduction %: expected {exp_comp_red}%, got {act_comp_red}%")
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable comp reduction cell: '{cells[4]}'")

        # Final opt chars (Base / Comp)
        exp_base_final = smc["final_optimized_prompt"]["baseline_chars"]
        exp_comp_final = smc["final_optimized_prompt"]["compressed_chars"]
        try:
            parts = clean_markdown_val(cells[5]).split("/")
            act_base_final = int(parts[0].strip())
            act_comp_final = int(parts[1].strip())
            if act_base_final != exp_base_final or act_comp_final != exp_comp_final:
                errors.append(
                    f"Table 5.1 {w_id} R{rep} final opt chars: expected {exp_base_final} / {exp_comp_final}, got {act_base_final} / {act_comp_final}"
                )
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable final opt chars cell: '{cells[5]}'")

        # Final opt reduction %
        exp_final_red = smc["final_optimized_prompt"]["reduction_percent"]
        try:
            act_final_red = float(clean_markdown_val(cells[6]).replace("%", "").replace("+", ""))
            if abs(act_final_red - exp_final_red) > 0.05:
                errors.append(f"Table 5.1 {w_id} R{rep} final opt reduction %: expected {exp_final_red}%, got {act_final_red}%")
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable final opt reduction cell: '{cells[6]}'")

        # Downstream input chars (Base / Comp)
        exp_base_down = smc["downstream_model_input"]["baseline_chars"]
        exp_comp_down = smc["downstream_model_input"]["compressed_chars"]
        try:
            parts = clean_markdown_val(cells[7]).split("/")
            act_base_down = int(parts[0].strip())
            act_comp_down = int(parts[1].strip())
            if act_base_down != exp_base_down or act_comp_down != exp_comp_down:
                errors.append(
                    f"Table 5.1 {w_id} R{rep} downstream input chars: expected {exp_base_down} / {exp_comp_down}, got {act_base_down} / {act_comp_down}"
                )
        except Exception:
            errors.append(f"Table 5.1 {w_id} R{rep} unparseable downstream input chars cell: '{cells[7]}'")

        # Optimizer status
        exp_status = trial["compressed"]["optimizer_inference"]["validation_status"]
        act_status = clean_markdown_val(cells[8])
        if act_status != exp_status:
            errors.append(f"Table 5.1 {w_id} R{rep} optimizer status: expected '{exp_status}', got '{act_status}'")


def validate_table_5_2(table_text: str, trials: List[Dict[str, Any]], errors: List[str]) -> None:
    """Validate each of the 10 trial rows in Table 5.2 against raw JSON trial data."""
    rows = parse_markdown_table_rows(table_text)
    if len(rows) != len(trials):
        errors.append(f"Table 5.2 row count mismatch: expected {len(trials)} rows, found {len(rows)}")
        return

    for idx, (trial, cells) in enumerate(zip(trials, rows)):
        w_id = trial["workload_id"]
        rep = trial["repetition"]
        row_id_cell = cells[0]

        if w_id not in row_id_cell or f"R{rep}" not in row_id_cell:
            errors.append(f"Table 5.2 row {idx + 1} ID mismatch: expected {w_id} (R{rep}), found {row_id_cell}")

        # Downstream Prefill Tokens (Base / Comp)
        d_tok = trial["comparison"]["downstream_token_savings"]
        exp_base_down_tok = d_tok["baseline_prompt_eval_count"]
        exp_comp_down_tok = d_tok["compressed_prompt_eval_count"]
        try:
            parts = clean_markdown_val(cells[2]).split("/")
            act_base_down_tok = int(parts[0].strip())
            act_comp_down_tok = int(parts[1].strip())
            if act_base_down_tok != exp_base_down_tok or act_comp_down_tok != exp_comp_down_tok:
                errors.append(
                    f"Table 5.2 {w_id} R{rep} downstream tokens: expected {exp_base_down_tok} / {exp_comp_down_tok}, got {act_base_down_tok} / {act_comp_down_tok}"
                )
        except Exception:
            errors.append(f"Table 5.2 {w_id} R{rep} unparseable downstream tokens cell: '{cells[2]}'")

        # Optimizer Prefill Tokens (Base / Comp)
        o_tok = trial["comparison"]["internal_optimizer_token_savings"]
        exp_base_opt_tok = o_tok["baseline_prompt_eval_count"]
        exp_comp_opt_tok = o_tok["compressed_prompt_eval_count"]
        try:
            parts = clean_markdown_val(cells[4]).split("/")
            act_base_opt_tok = int(parts[0].strip())
            act_comp_opt_tok = int(parts[1].strip())
            if act_base_opt_tok != exp_base_opt_tok or act_comp_opt_tok != exp_comp_opt_tok:
                errors.append(
                    f"Table 5.2 {w_id} R{rep} optimizer tokens: expected {exp_base_opt_tok} / {exp_comp_opt_tok}, got {act_base_opt_tok} / {act_comp_opt_tok}"
                )
        except Exception:
            errors.append(f"Table 5.2 {w_id} R{rep} unparseable optimizer tokens cell: '{cells[4]}'")

        # Engine overhead latency
        exp_latency = trial["compressed"]["compression_stage"]["processing_time_ms"]
        try:
            act_latency = float(clean_markdown_val(cells[6]).replace("ms", "").strip())
            if abs(act_latency - exp_latency) > 0.05:
                errors.append(f"Table 5.2 {w_id} R{rep} latency: expected {exp_latency} ms, got {act_latency} ms")
        except Exception:
            errors.append(f"Table 5.2 {w_id} R{rep} unparseable latency cell: '{cells[6]}'")

        # Downstream quality status
        exp_quality = trial["comparison"]["downstream_answer_status"]
        act_quality = clean_markdown_val(cells[7])
        if act_quality != exp_quality:
            errors.append(f"Table 5.2 {w_id} R{rep} quality status: expected '{exp_quality}', got '{act_quality}'")

        # Restoration check
        exp_resto = trial["comparison"]["restoration_verification"]
        act_resto = clean_markdown_val(cells[8]).lower()
        if exp_resto["tested"]:
            if "verified" not in act_resto and "byte-exact" not in act_resto:
                errors.append(f"Table 5.2 {w_id} R{rep} restoration: expected verified byte-exact, got '{cells[8]}'")
        else:
            if "n/a" not in act_resto and "irrev" not in act_resto:
                errors.append(f"Table 5.2 {w_id} R{rep} restoration: expected N/A, got '{cells[8]}'")


def validate_statistical_aggregates_table(table_text: str, stats: Dict[str, Any], errors: List[str]) -> None:
    """Validate each aggregate metric strictly within Section 7 Statistical Aggregates table."""
    rows = parse_markdown_table_rows(table_text)
    if not rows:
        errors.append("Statistical Aggregates table is empty or missing from Section 7")
        return

    # Helper to check row values
    def check_stat(label: str, val: Any) -> None:
        check_value_in_text("Section 7 (Statistical Aggregates)", label, val, table_text, errors)

    # 1. Compression Stage Chars
    cs = stats.get("compression_stage_character_reduction", {})
    check_stat("CS Aggregate %", cs.get("aggregate_percent"))
    check_stat("CS Mean %", cs.get("mean_percent"))
    check_stat("CS Median %", cs.get("median_percent"))
    check_stat("CS StdDev %", cs.get("stdev_percent"))

    # 2. Final Optimized Prompt Chars
    fo = stats.get("final_optimized_prompt_character_reduction", {})
    check_stat("FO Aggregate %", fo.get("aggregate_percent"))
    check_stat("FO Mean %", fo.get("mean_percent"))
    check_stat("FO Median %", fo.get("median_percent"))
    check_stat("FO StdDev %", fo.get("stdev_percent"))

    # 3. Downstream Model Tokens
    dt = stats.get("downstream_prompt_token_savings", {})
    check_stat("Downstream Tokens Saved", dt.get("total_tokens_saved"))
    check_stat("Downstream Aggregate %", dt.get("aggregate_percent"))
    check_stat("Downstream Mean Tokens", dt.get("mean_tokens_saved"))
    check_stat("Downstream Median Tokens", dt.get("median_tokens_saved"))
    check_stat("Downstream StdDev Tokens", dt.get("stdev_tokens_saved"))

    # 4. Internal Optimizer Tokens
    ot = stats.get("internal_optimizer_prompt_token_savings", {})
    check_stat("Optimizer Tokens Saved", ot.get("total_tokens_saved"))
    check_stat("Optimizer Aggregate %", ot.get("aggregate_percent"))
    check_stat("Optimizer Mean Tokens", ot.get("mean_tokens_saved"))
    check_stat("Optimizer Median Tokens", ot.get("median_tokens_saved"))
    check_stat("Optimizer StdDev Tokens", ot.get("stdev_tokens_saved"))

    # 5. Engine Overhead Latency
    lat = stats.get("compression_engine_latency_ms", {})
    check_stat("Latency Mean", lat.get("mean_ms"))
    check_stat("Latency Median", lat.get("median_ms"))
    check_stat("Latency StdDev", lat.get("stdev_ms"))

    # 6. Quality & Restoration
    qa = stats.get("downstream_task_quality_summary", {})
    check_stat("Quality Preserved Count", qa.get("preserved_count"))
    check_stat("Quality Preservation %", qa.get("preservation_rate_percent"))

    res = stats.get("reversible_restoration_summary", {})
    check_stat("Restoration Verified Count", res.get("verified_byte_exact"))
    check_stat("Restoration Rate %", res.get("verification_rate_percent"))


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
    trials: List[Dict[str, Any]] = data.get("trials", [])

    sections = extract_report_sections(md_text)

    # 1. Section 1 Executive Summary Validation
    exec_summary = sections.get("executive_summary", "")
    if not exec_summary:
        errors.append("Section 1 (Executive Summary) is missing from Markdown report")
    else:
        # Check that aggregate metrics are explicitly presented in Executive Summary
        cs = stats.get("compression_stage_character_reduction", {})
        check_value_in_text("Section 1 (Executive Summary)", "CS Aggregate %", cs.get("aggregate_percent"), exec_summary, errors, ["%"])
        check_value_in_text("Section 1 (Executive Summary)", "CS StdDev %", cs.get("stdev_percent"), exec_summary, errors, ["%"])

        fo = stats.get("final_optimized_prompt_character_reduction", {})
        check_value_in_text("Section 1 (Executive Summary)", "FO Aggregate %", fo.get("aggregate_percent"), exec_summary, errors, ["%"])
        check_value_in_text("Section 1 (Executive Summary)", "FO StdDev %", fo.get("stdev_percent"), exec_summary, errors, ["%"])

        dt = stats.get("downstream_prompt_token_savings", {})
        check_value_in_text("Section 1 (Executive Summary)", "Downstream Tokens Saved", dt.get("total_tokens_saved"), exec_summary, errors)
        check_value_in_text("Section 1 (Executive Summary)", "Downstream StdDev", dt.get("stdev_tokens_saved"), exec_summary, errors)

        ot = stats.get("internal_optimizer_prompt_token_savings", {})
        check_value_in_text("Section 1 (Executive Summary)", "Optimizer Tokens Saved", ot.get("total_tokens_saved"), exec_summary, errors)
        check_value_in_text("Section 1 (Executive Summary)", "Optimizer StdDev", ot.get("stdev_tokens_saved"), exec_summary, errors)

        lat = stats.get("compression_engine_latency_ms", {})
        check_value_in_text("Section 1 (Executive Summary)", "Latency Mean ms", lat.get("mean_ms"), exec_summary, errors, [" ms"])
        check_value_in_text("Section 1 (Executive Summary)", "Latency StdDev ms", lat.get("stdev_ms"), exec_summary, errors, [" ms"])

        qa = stats.get("downstream_task_quality_summary", {})
        check_value_in_text("Section 1 (Executive Summary)", "Quality Preservation %", qa.get("preservation_rate_percent"), exec_summary, errors, ["%"])

        res = stats.get("reversible_restoration_summary", {})
        check_value_in_text("Section 1 (Executive Summary)", "Restoration Rate %", res.get("verification_rate_percent"), exec_summary, errors, ["%"])

    # 2. Section 5.1 Four-Stage Character Measurements Table
    table_5_1_text = sections.get("table_5_1", "")
    if not table_5_1_text:
        errors.append("Section 5.1 (Four-Stage Character Measurements) table is missing from Markdown report")
    else:
        validate_table_5_1(table_5_1_text, trials, errors)

    # 3. Section 5.2 Token Telemetry, Engine Latency & Answer Quality Table
    table_5_2_text = sections.get("table_5_2", "")
    if not table_5_2_text:
        errors.append("Section 5.2 (Token Telemetry, Engine Latency & Answer Quality) table is missing from Markdown report")
    else:
        validate_table_5_2(table_5_2_text, trials, errors)

    # 4. Section 7 Statistical Aggregates Table
    table_7_text = sections.get("statistical_aggregates", "")
    if not table_7_text:
        errors.append("Section 7 (Statistical Aggregates) table is missing from Markdown report")
    else:
        validate_statistical_aggregates_table(table_7_text, stats, errors)

    # 5. Stale numbers check across entire document
    stale_checks = [
        ("0.354 ms", "Stale compression latency standard deviation (0.354 ms) found in report"),
        ("7.91%", "Stale compression-stage standard deviation (7.91%) found in report"),
        ("27.24%", "Stale final optimized prompt standard deviation (27.24%) found in report"),
    ]
    for stale_str, err_msg in stale_checks:
        if stale_str in md_text:
            errors.append(err_msg)

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
        print("\n[PASSED] All sections, tables, trial rows, and statistical aggregates match raw benchmark JSON.")


if __name__ == "__main__":
    main()
