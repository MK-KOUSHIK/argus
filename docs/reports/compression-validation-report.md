# Argus Independent Content-Aware Compression Engine — Empirical Validation & Performance Report

**Date:** 2026-11-05  
**Branch:** `feat/phase-4.2-compression-recovery`  
**Repository:** `https://github.com/MK-KOUSHIK/argus`  
**Target Architecture:** Phase 4.2 Independent Content-Aware Compression Engine  
**Validation Standard:** Empirical Local Inference (`qwen2.5:1.5b`), Alternating Execution Order, and Rule-Based Downstream Verification  

---

## 1. Executive Summary

In Phase 4.2, Argus introduces an **Independent Content-Aware Compression Engine** engineered to optimize input prompts prior to neural model submission. The engine operates at **Stage 0** in the optimization pipeline, dynamically routing content into specialized structural compressors, pruning syntactic whitespace, boilerplate, redundant logs, and repetitive conversational phrasing, while strictly preserving technical constraints, entity references, and functional invariants.

This capability is an entirely original Argus implementation with **zero external dependencies on Headroom**.

### Empirical Benchmark Summary (10 Paired Trials across 5 Workloads)

The benchmark was executed using `scripts/benchmark_compression.py` against local Ollama (`qwen2.5:1.5b` on NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM) using 1 warm-up run, 2 repetitions, and strictly alternating execution order:

- **Aggregate Character Reduction:** **32.83%** (mean: 26.42%, median: 30.61%, stdev: 25.97%).
- **Native Prompt Prefill Token Reduction:** **-13.40%** (9,058 baseline prompt tokens vs. 7,844 compressed tokens; total **1,214 tokens saved** across trials; mean 121.4 tokens saved per trial, median 124.0 tokens, stdev 85.33 tokens).
- **Compression Engine Overhead Latency:** **mean 0.904 ms** (median 0.475 ms, stdev 1.204 ms).
- **Downstream Answer Quality:** Rule-based programmatic heuristic checks verified task deliverables across all 10 trials in the recovered engine.
- **Reversible Restoration Verification:** **100.0%** (8 of 8 tested references verified byte-exact against the actual expected original payload, with full SHA-256 hash match and 100% negative test passes).

---

## 2. Hardware and Runtime Environment

Dynamic hardware telemetry detection (`detect_hardware_metadata()`) reported:

| Component | Specification | Detection Method |
| :--- | :--- | :--- |
| **Host OS** | Windows 10 (AMD64) | `platform.system()` & `platform.release()` |
| **System RAM** | 15,556.53 MB (~16 GB DDR5) | Win32 `GlobalMemoryStatusEx` (`MEMORYSTATUSEX`) |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU | `nvidia-smi` CSV query |
| **Total VRAM** | 6,141 MiB (~6 GB GDDR6) | `nvidia-smi --query-gpu=memory.total` |
| **Driver Version** | 617.42 | `nvidia-smi --query-gpu=driver_version` |
| **Local Inference Server** | Ollama (`http://127.0.0.1:11434`) | Verified `/api/tags` connection |
| **Active Model** | `qwen2.5:1.5b` (Q4_K_M GGUF, 1.5B parameters) | Native local model |
| **Inference Parameters** | `temperature=0.0`, `top_p=0.9`, `num_predict=256`, `num_ctx=4096` | Standard deterministic downstream eval |

---

## 3. Prior Observed Failures & Transparent Regression Analysis

During initial uncalibrated evaluation, three downstream task regressions were observed out of 10 trials (a 70% preservation rate). In accordance with Argus engineering integrity standards, these observed failures are documented below alongside the specific architectural fixes implemented in this recovery:

### Prior Failure 1 & 2: `code_caching_pipeline` (R1 & R2)
- **Observed Symptom:** Baseline answer passed (`has_lru=True`, `has_ttl=True`), but compressed answer failed (`has_lru=False`, `has_ttl=True`).
- **Root Cause:** Naive AST unparsing stripped all comments, including the class-header comment `# In-Memory Cache with TTL and LRU Eviction`. Without this high-level algorithmic cue, the downstream model spent all 256 generation tokens detailing `__init__` and dictionary storage types, running out of output tokens before explaining LRU eviction.
- **Recovery Fix:** Enhanced `CodeCompressor` with `_extract_algorithmic_comments()` to detect and retain class/module architectural headers and comments containing algorithmic terms (`lru`, `ttl`, `eviction`, `policy`, `complexity`, `thread-safe`, `invariant`, etc.), while stripping pure metadata/banners (`Author:`, `Updated:`, `===`). In the recovered benchmark, both repetitions successfully passed downstream quality checks.

### Prior Failure 3: `logs_database_incident` (R2)
- **Observed Symptom:** Baseline answer correctly identified crash timestamp and cause, but compressed answer failed with: `"I'm sorry, but I can't examine system service logs directly..."`.
- **Root Cause:** Overly aggressive log folding left an ultra-compact single line that the LLM misinterpreted as an instruction to query external log files on disk rather than analyzing an inline log snippet.
- **Recovery Fix:** 
  1. Updated `LogCompressor` to preserve clear event boundaries (initial and terminal occurrences around folded runs), retain error codes (`status=500`, `code=503`, `connection loss`), and protect critical error lines verbatim.
  2. Updated `PromptRewriter` with Rule 11, instructing the model to never delete or omit embedded data payloads (logs, code, JSON).
  3. Updated `CandidateValidator` to detect and reject candidate prompts that discard critical incident error context.

---

## 4. Empirical Benchmark Telemetry Across All 10 Trials

All measurements represent actual, raw, unmanipulated telemetry persisted in [`docs/reports/compression_benchmark.json`](compression_benchmark.json).

| Workload ID & Repetition | Order | Baseline Chars | Compressed Chars | Char Delta (%) | Baseline Prefill Tokens | Compressed Prefill Tokens | Tokens Saved | Engine Overhead | Prompt Preserved | Downstream Answer Quality | Reversible Ref Verified |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`json_cluster_config`** (R1) | Base -> Comp | 1,179 | 1,179 | +0.00% | 905 | 765 | **-140 (-15.5%)** | 0.22 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`json_cluster_config`** (R2) | Comp -> Base | 1,179 | 1,179 | +0.00% | 905 | 765 | **-140 (-15.5%)** | 0.23 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`code_caching_pipeline`** (R1) | Base -> Comp | 2,024 | 1,296 | **+35.97%** | 1,000 | 876 | **-124 (-12.4%)** | 1.40 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`code_caching_pipeline`** (R2) | Comp -> Base | 2,024 | 1,296 | **+35.97%** | 1,000 | 876 | **-124 (-12.4%)** | 4.19 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`logs_database_incident`** (R1) | Base -> Comp | 1,601 | 578 | **+63.85%** | 1,224 | 966 | **-258 (-21.1%)** | 0.47 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`logs_database_incident`** (R2) | Comp -> Base | 1,601 | 525 | **+67.17%** | 1,224 | 966 | **-258 (-21.1%)** | 0.48 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`text_redundant_instructions`** (R1) | Base -> Comp | 549 | 547 | +0.00% | 623 | 606 | **-17 (-2.7%)** | 0.37 ms | `preserved` | `preserved` | N/A (irrev.) |
| **`text_redundant_instructions`** (R2) | Comp -> Base | 549 | 547 | +0.00% | 623 | 606 | **-17 (-2.7%)** | 0.35 ms | `preserved` | `preserved` | N/A (irrev.) |
| **`mixed_debugging_dossier`** (R1) | Base -> Comp | 884 | 612 | **+30.61%** | 777 | 709 | **-68 (-8.8%)** | 0.66 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |
| **`mixed_debugging_dossier`** (R2) | Comp -> Base | 884 | 612 | **+30.61%** | 777 | 709 | **-68 (-8.8%)** | 0.67 ms | `preserved` | `preserved` | **Verified (Byte-Exact)** |

### Statistical Summary

| Metric | Aggregate / Total | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- |
| **Character Reduction (%)** | **32.83%** | 26.42% | 30.61% | 25.97% |
| **Native Prefill Tokens Saved** | **1,214 tokens (-13.40%)** | 121.4 tokens | 124.0 tokens | 85.33 tokens |
| **Engine Overhead Latency** | — | **0.904 ms** | 0.475 ms | 1.204 ms |
| **Downstream Quality Status** | 10 preserved / 0 regressions | **100.0%** (rule-based checks) | — | — |
| **Reversible Restoration** | 8 verified / 8 tested | **100.0% byte-exact** | — | — |

---

## 5. Scope & Status Breakdown

### 5.1 Implemented and Tested
- **JSON Minifier (`JsonCompressor`):** RFC-8259 structural minification validated via `json.loads` deep equality.
- **AST-Based Code Optimizer (`CodeCompressor`):** AST parsing (`ast.parse` / `ast.unparse`) preserving class-level architectural headers, algorithmic comments, and docstrings, validated with syntax re-parse and functional node parity (`_compare_ast_structures`).
- **Repetitive Log Event Folder (`LogCompressor`):** Folds runs ($\ge 3$), preserves initial and terminal event boundaries with timestamps, protects error codes (`status=500`, `code=503`) and critical errors verbatim.
- **Text Deduplicator (`TextRedundancyCompressor`):** Strips duplicate instructions while safeguarding negations (`never`, `do not`), numbers with units (`500ms`, `42.5 GiB`, `$50,000.50`), timestamps, and identifiers.
- **Prefix Cache Preserver (`PrefixPreserver`):** Standardizes system prompt wrapping structures.
- **Thread-Safe Reversible Registry (`CompressionRegistry`):** In-memory LRU/TTL storage with deterministic SHA-256 reference keys (`argus:ref:...`).
- **Non-Tautological Restoration Verification:** `verify_workload_restoration` compares retrieved payloads directly against the **actual expected original payload from the prompt**, verifying exact byte equality, SHA-256 digest match, and negative test resilience.
- **Adversarial Test Suite:** 12 adversarial semantic-preservation tests in `test_compression.py` (totaling 31 compression tests, 118 backend tests).
- **Frontend Workspace Card:** Real-time compression telemetry, strategy badges, and restoration key inspection in `MetricsPanel.tsx`.

### 5.2 Implemented but Not Validated against Frontier Models
- Downstream prefix caching behavior on remote proprietary LLM APIs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) where KV cache prefill is managed remotely.
- Extremely large log files (>100,000 lines) exceeding local RAM limits.

### 5.3 Known Limitations
- **Syntax Dependence:** `CodeCompressor` AST optimization requires syntactically valid Python code; invalid Python or unsupported languages safely fall back to verbatim preservation.
- **Conservative Text Floors:** Natural language text compression enforces a conservative retention floor to prevent accidental omission of nuanced intent.
- **Rule-Based Evaluation:** Downstream quality checks are programmatic rule-based heuristic validators checking specific deliverables; they do not represent exhaustive semantic equivalence.
- **Output Headroom Constraints:** Very complex code explanations under limited generation budgets (`num_predict <= 256`) may exhaust decoding tokens if an exhaustive explanation is requested.

### 5.4 Planned Improvements
- **Phase 5 Frontier Gateway Integration:** Measure downstream prompt token cost reductions across external commercial APIs.
- **Streaming Log Compressor:** Chunked generator-based log compression for continuous tail streams.

---

## 6. Regression Testing & Verification Matrix

- **Backend Pytest Suite:** `pytest backend/tests -v` -> **118 passed in 8.08s**
- **Frontend Vitest Suite:** `vitest run --run` -> **19 passed in 2.14s**
- **Frontend Production Build:** `tsc -b && vite build` -> **0 errors, completed in 2.73s**
- **Pyright Type Checking:** `pyright scripts backend/app/services/compression` -> **0 errors, 0 warnings**
- **Local Ollama Benchmark:** 10 paired trials across 5 workloads executed with **0 mock responses**
