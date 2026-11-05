# Argus Independent Content-Aware Compression Engine — Empirical Validation & Performance Report

**Date:** 2026-11-05  
**Branch:** `feat/phase-4.2-compression-recovery`  
**Repository:** `https://github.com/MK-KOUSHIK/argus`  
**Target Architecture:** Phase 4.2 Independent Content-Aware Compression Engine  
**Validation Standard:** Empirical Local Inference (`qwen2.5:1.5b`), Alternating Execution Order, 4-Stage Traceability, and Objective Rule-Based Acceptance Evaluation  

---

## 1. Executive Summary

In Phase 4.2, Argus introduces an **Independent Content-Aware Compression Engine** engineered to optimize input prompts prior to neural model submission. The engine operates at **Stage 0** in the optimization pipeline, dynamically routing content into specialized structural compressors, pruning syntactic whitespace, boilerplate, redundant logs, and repetitive conversational phrasing, while strictly preserving technical constraints, entity references, and functional invariants.

This capability is an entirely original Argus implementation with **zero external dependencies on Headroom**.

### Key Telemetry Summary (10 Paired Trials across 5 Workloads)

The benchmark was executed using `scripts/benchmark_compression.py` against local Ollama (`qwen2.5:1.5b` on NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM) using 1 warm-up run, 2 repetitions, and strictly alternating execution order:

- **Compression-Stage Character Reduction:** **+31.07% aggregate** (mean: +28.71%, median: +31.18%, stdev: 8.01%).
- **Final Optimized Prompt Character Reduction:** **+18.31% aggregate** (mean: +18.20%, median: +11.30%, stdev: 27.70%).
- **Actual Downstream Model Token Savings:** **-16.12% prompt prefill tokens** (462 input tokens saved across task execution trials; mean: 46.2 tokens saved, median: 15.5 tokens, stdev: 78.55 tokens).
- **Internal Optimizer Rewriter Token Savings:** **-13.40% prompt prefill tokens** (1214 input tokens saved across neural prompt rewriting calls; mean: 121.4 tokens saved, median: 124.0 tokens, stdev: 85.33 tokens).
- **Compression Engine Overhead Latency:** **mean 0.564 ms** (median: 0.400 ms, stdev: 0.495 ms) — sub-millisecond overhead.
- **Downstream Answer Quality:** **6 preserved / 10 trials (60.0%)** under strict rule-based heuristic checks.
- **Reversible Restoration Verification:** **100.0%** (8 of 8 tested references verified byte-exact against the actual expected original payload, with full SHA-256 hash match and 100% negative test passes).

---

## 2. Investigation & Reconciliation of Character Discrepancy

### Root Cause Analysis
During earlier review, a discrepancy was noted where the PR description reported `1,228` baseline characters and `656` compressed characters for the JSON workload, while the raw JSON trial data showed `1,179` baseline characters and `1,179` compressed characters:

1. **Conflation of Pipeline Stages:** The earlier draft had conflated the *compression-stage output* (the output of `CompressionOrchestrator.compress()`) with the *final optimized prompt* (the return value of `ollama_service.optimize_prompt()`).
2. **Validator Fallback Behavior:** For `json_cluster_config`, the original prompt is 1,179 characters. At Stage 0, `CompressionOrchestrator.compress()` successfully compressed the fenced JSON payload to 792 characters (+32.82% reduction). However, when the compressed prompt was passed to the neural rewriter, the rewriter hit its allocated output budget while reproducing the JSON, running out of tokens before emitting the task instruction sentence. `CandidateValidator` detected that the candidate dropped the required task operation and rejected the rewrite. Per Argus safety design, failed candidate validation triggers a graceful safety fallback to the original prompt (`validation_status: fallback`), returning 1,179 characters.

### Resolution & 4-Stage Measurement Framework
Both the raw JSON (`docs/reports/compression_benchmark.json`) and this report are regenerated from the exact same benchmark run, explicitly tracking four equivalent measurements for every trial:
- **Stage 1 (Original Workload Prompt):** The raw input prompt text and character count.
- **Stage 2 (Compression-Stage Output):** The text and character count produced by `CompressionOrchestrator.compress()` before task analysis.
- **Stage 3 (Final Optimized Prompt):** The prompt returned by `ollama_service.optimize_prompt()` after Stage 0 compression, task analysis, neural rewriting, and candidate validation.
- **Stage 4 (Downstream Model Input):** The exact prompt delivered to the downstream model for task execution.

---

## 3. Hardware and Runtime Environment

Dynamic hardware telemetry detection (`detect_hardware_metadata()`) reported:

| Component | Specification | Detection Method |
| :--- | :--- | :--- |
| **Host OS** | Windows 10 (AMD64) | `platform.system()` & `platform.release()` |
| **System RAM** | 15556.53 MB (~16 GB DDR5) | Win32 `GlobalMemoryStatusEx` (`MEMORYSTATUSEX`) |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU | `nvidia-smi` CSV query |
| **Total VRAM** | 6141 MiB (~6 GB GDDR6) | `nvidia-smi --query-gpu=memory.total` |
| **Driver Version** | 617.42 | `nvidia-smi --query-gpu=driver_version` |
| **Local Inference Server** | Ollama (`http://127.0.0.1:11434`) | Verified `/api/tags` connection |
| **Active Model** | `qwen2.5:1.5b` | Native local model |
| **Downstream Inference Settings** | `temperature=0.0`, `top_p=0.9`, `num_predict=384`, `num_ctx=4096` | Deterministic downstream execution |

---

## 4. Separation of Optimizer Inference vs Downstream Model Inference

To prevent conflation of internal optimization costs with downstream inference savings:

1. **Internal Optimizer Inference (Prompt Rewriter):**
   - Represents the local model call inside Argus (Stage C) that neuralizes and restructures the prompt.
   - Compares the prefill token footprint when the rewriter is fed uncompressed input vs. compressed input.
   - Recorded in `trial["comparison"]["internal_optimizer_token_savings"]`.

2. **Downstream Model Inference (Task Answer Generation):**
   - Represents the actual downstream model executing the user's task using the final optimized prompt produced by Argus.
   - Compares the downstream model's `prompt_eval_count` (prefill tokens), `eval_count` (generation tokens), `prompt_eval_duration_ms`, `eval_duration_ms`, and `done_reason`.
   - Recorded in `trial["comparison"]["downstream_token_savings"]`.

---

## 5. Empirical Benchmark Telemetry Across All 10 Trials

All measurements represent actual, raw, unmanipulated telemetry persisted in [`docs/reports/compression_benchmark.json`](compression_benchmark.json).

### 5.1 Four-Stage Character Measurements

| Workload ID & Rep | Order | (1) Original Chars | (2) Comp-Stage Chars | Comp-Stage Reduction | (3) Final Opt Chars (Base / Comp) | Final Opt Reduction | (4) Downstream Input Chars (Base / Comp) | Optimizer Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`json_cluster_config`** (R1) | Baseline -> Compressed | 1,179 | 792 | **+32.82%** | 1,179 / 1,179 | **+0.00%** | 1,179 / 1,179 | `fallback` |
| **`json_cluster_config`** (R2) | Compressed -> Baseline | 1,179 | 792 | **+32.82%** | 1,179 / 1,179 | **+0.00%** | 1,179 / 1,179 | `fallback` |
| **`code_caching_pipeline`** (R1) | Baseline -> Compressed | 2,024 | 1,393 | **+31.18%** | 1,927 / 1,296 | **+32.75%** | 1,927 / 1,296 | `passed` |
| **`code_caching_pipeline`** (R2) | Compressed -> Baseline | 2,024 | 1,393 | **+31.18%** | 1,927 / 1,296 | **+32.75%** | 1,927 / 1,296 | `passed` |
| **`logs_database_incident`** (R1) | Baseline -> Compressed | 1,599 | 968 | **+39.46%** | 209 / 112 | **+46.41%** | 209 / 112 | `passed` |
| **`logs_database_incident`** (R2) | Compressed -> Baseline | 1,599 | 968 | **+39.46%** | 667 / 200 | **+70.01%** | 667 / 200 | `passed` |
| **`text_redundant_instructions`** (R1) | Baseline -> Compressed | 547 | 438 | **+19.93%** | 547 / 547 | **+0.00%** | 547 / 547 | `fallback` |
| **`text_redundant_instructions`** (R2) | Compressed -> Baseline | 547 | 438 | **+19.93%** | 547 / 547 | **+0.00%** | 547 / 547 | `fallback` |
| **`mixed_debugging_dossier`** (R1) | Baseline -> Compressed | 882 | 704 | **+20.18%** | 978 / 757 | **+22.60%** | 978 / 757 | `passed` |
| **`mixed_debugging_dossier`** (R2) | Compressed -> Baseline | 882 | 704 | **+20.18%** | 905 / 1,109 | **-22.54%** | 905 / 1,109 | `passed` |

---

### 5.2 Token Telemetry, Engine Latency & Answer Quality

| Workload ID & Rep | Order | Downstream Prefill Tokens (Base / Comp) | Downstream Tokens Saved (%) | Optimizer Prefill Tokens (Base / Comp) | Optimizer Tokens Saved (%) | Engine Overhead | Downstream Quality Status | Restoration Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`json_cluster_config`** (R1) | Baseline -> Compressed | 407 / 407 | **+0 (+0.0%)** | 905 / 765 | **+140 (+15.5%)** | 0.15 ms | `preserved` | **Verified (Byte-Exact)** |
| **`json_cluster_config`** (R2) | Compressed -> Baseline | 407 / 407 | **+0 (+0.0%)** | 905 / 765 | **+140 (+15.5%)** | 0.16 ms | `preserved` | **Verified (Byte-Exact)** |
| **`code_caching_pipeline`** (R1) | Baseline -> Compressed | 455 / 339 | **+116 (+25.5%)** | 1000 / 876 | **+124 (+12.4%)** | 1.55 ms | `preserved` | **Verified (Byte-Exact)** |
| **`code_caching_pipeline`** (R2) | Compressed -> Baseline | 455 / 339 | **+116 (+25.5%)** | 1000 / 876 | **+124 (+12.4%)** | 1.41 ms | `preserved` | **Verified (Byte-Exact)** |
| **`logs_database_incident`** (R1) | Baseline -> Compressed | 98 / 67 | **+31 (+31.6%)** | 1224 / 966 | **+258 (+21.1%)** | 0.42 ms | `regression` | **Verified (Byte-Exact)** |
| **`logs_database_incident`** (R2) | Compressed -> Baseline | 288 / 83 | **+205 (+71.2%)** | 1224 / 966 | **+258 (+21.1%)** | 0.42 ms | `preserved` | **Verified (Byte-Exact)** |
| **`text_redundant_instructions`** (R1) | Baseline -> Compressed | 119 / 119 | **+0 (+0.0%)** | 623 / 606 | **+17 (+2.7%)** | 0.38 ms | `inconclusive_or_both_failed` | N/A (irrev.) |
| **`text_redundant_instructions`** (R2) | Compressed -> Baseline | 119 / 119 | **+0 (+0.0%)** | 623 / 606 | **+17 (+2.7%)** | 0.32 ms | `inconclusive_or_both_failed` | N/A (irrev.) |
| **`mixed_debugging_dossier`** (R1) | Baseline -> Compressed | 266 / 210 | **+56 (+21.1%)** | 777 / 709 | **+68 (+8.8%)** | 0.38 ms | `preserved` | **Verified (Byte-Exact)** |
| **`mixed_debugging_dossier`** (R2) | Compressed -> Baseline | 252 / 314 | **-62 (-24.6%)** | 777 / 709 | **+68 (+8.8%)** | 0.45 ms | `improvement` | **Verified (Byte-Exact)** |

---

## 6. Detailed Answer Quality & Regression Transparency

Answer quality evaluation uses hardened programmatic heuristic checks against workload-specific acceptance criteria, acknowledging explicitly that these are objective rule checks rather than full semantic equivalence:

1. **`json_cluster_config` (Preserved in R1 & R2):**
   - *Checks:* Identified all 3 active node names (`node-alpha`, `node-beta`, `node-gamma`), memory specs (64, 128, or 320), and node roles/status.
   - *Outcome:* Preserved across both trials. Downstream answer accurately summed memory to 320 GB.

2. **`code_caching_pipeline` (Preserved in R1 & R2):**
   - *Checks:* Verified explanation of LRU eviction (`OrderedDict`, `move_to_end`, `popitem`), TTL expiration (`now > expires_at`), and capacity-bound behavior.
   - *Outcome:* Preserved across both trials. Saving 116 downstream prompt tokens per trial (-25.5%).

3. **`logs_database_incident` (1 Regression in R1, Preserved in R2):**
   - *Checks:* Verified identification of exact failure cause (`connection refused` to PostgreSQL at 10.0.1.5:5432 / `ConnectionPoolExhausted`) and exact incident timestamp (`08:00:12`, `08:00:13`, or `08:00:18`).
   - *Analysis of R1 Regression:* In Rep 1, the compressed prompt was condensed to 112 characters ("Identify what caused the worker crash..."). The downstream model focused on the general connection refused error but omitted the seconds timestamp in its generated summary sentence, causing the strict timestamp regex check to fail. In Rep 2, with a 200-character prompt, the model included both the cause and `08:00:18` timestamp, passing with 205 downstream tokens saved (+71.2%).

4. **`text_redundant_instructions` (Both Failed in R1 & R2):**
   - *Checks:* Verified horizontal scaling concept, vertical scaling concept, negative check for prohibited vendor names (AWS, Azure, GCP, Google Cloud, Microsoft), and strictly 3 bullet points.
   - *Analysis:* Both baseline and compressed answers adhered to the scaling concepts and avoided prohibited vendor names. However, `qwen2.5:1.5b` outputted nested sub-bullets (5-7 bullet markers in total) rather than strictly 3 top-level bullets. Because both baseline and compressed failed the strict 3-bullet count constraint, the comparison status is honestly recorded as `inconclusive_or_both_failed`.

5. **`mixed_debugging_dossier` (Preserved in R1, Improvement in R2):**
   - *Checks:* Verified service identity (`billing-aggregator`) and port consistency (addresses port 8080 vs. database port 5432).
   - *Outcome:* In Rep 1, preserved. In Rep 2, the baseline prompt resulted in a truncated port explanation, while the compressed run completed the port comparison, resulting in an `improvement` classification.

---

## 7. Statistical Aggregates

| Metric Category | Metric | Measured Value |
| :--- | :--- | :--- |
| **Compression-Stage Chars** | Aggregate Reduction | **+31.07%** |
| | Mean / Median / StdDev | +28.71% / +31.18% / 8.01% |
| **Final Optimized Prompt Chars** | Aggregate Reduction | **+18.31%** |
| | Mean / Median / StdDev | +18.20% / +11.30% / 27.70% |
| **Downstream Model Tokens** | Total Input Tokens Saved | **462 tokens (-16.12%)** |
| | Mean / Median / StdDev | 46.2 tokens / 15.5 tokens / 78.55 tokens |
| **Internal Optimizer Tokens** | Total Prefill Tokens Saved | **1214 tokens (-13.40%)** |
| | Mean / Median / StdDev | 121.4 tokens / 124.0 tokens / 85.33 tokens |
| **Engine Overhead Latency** | Mean / Median / StdDev | **0.564 ms** / 0.400 ms / 0.495 ms |
| **Downstream Task Quality** | Preserved / Total Trials | **6 / 10 (60.0%)** |
| | Regressions / Improvements / Both Failed | 1 regression / 1 improvement / 2 both failed |
| **Reversible Restoration** | Verified Byte-Exact / Total Tested | **8 / 8 (100.0%)** |

---

## 8. Verification of Application Pipeline Integration

The following integration invariants were verified through unit and pipeline tests (`backend/tests/test_compression.py`):

1. **Stage 0 Precedes Task Analysis:** Verified that `CompressionOrchestrator.compress()` executes before `TaskAnalyzer.analyze()`, ensuring task analysis receives the structured, minified prompt.
2. **Compressed Prompt Passed to Rewriter:** Verified that `_execute_model_call()` receives the compressed prompt text.
3. **Correct Return of Final Prompt:** Verified that `OptimizeResponse.optimized_prompt` returns the validated candidate rewrite, or the safe fallback if validation fails.
4. **Compression Telemetry Isolation:** Verified that `OptimizeResponse.compression` reports telemetry strictly from the compression stage (tokens reduced, ratio, processing time) independent of the rewriter's token counts.
5. **Zero-Change Pass-Through When Disabled:** Verified that `enable_compression=False` returns `compressed_text == original_text` with 0 tokens reduced and strategy `NONE`.
6. **Graceful Fallback on Validation Failure:** Verified that when candidate prompts fail validation, the system safely falls back to the original request prompt without raising exceptions or corrupting the user query.
