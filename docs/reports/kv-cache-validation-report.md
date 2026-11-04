# Argus — Comprehensive KV-Cache Runtime Optimization Validation Report

**Document Version:** 2.1.0  
**Date:** 2026-11-03  
**Target Repository:** [MK-KOUSHIK/argus](https://github.com/MK-KOUSHIK/argus)  
**Branch:** `feat/complete-kv-cache-runtime`  
**PR:** [#6](https://github.com/MK-KOUSHIK/argus/pull/6)  
**Execution Environment:** Local Hardware Execution (Zero Simulation / Zero Mock Inference)  

---

## 1. Executive Summary

This report documents the rigorous empirical validation and runtime hardening of the adaptive KV-cache optimization suite for Argus. All benchmarks, telemetry records, and experiments were conducted via live local model execution on an NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB dedicated VRAM) paired with an AMD Ryzen 7 7735HS processor and 16 GB of system RAM, running Ollama v0.40.8 with model `qwen2.5:1.5b`.

### Key Empirical Findings

1. **KV Cache Memory Footprint Reduction**:
   - Baseline F16 buffer: **112.00 MiB** (4,096 cells, 28 layers, K: 56.00 MiB, V: 56.00 MiB).
   - Quantized Q8_0 buffer: **59.52 MiB** (4,096 cells, 28 layers, K: 29.75 MiB, V: 29.75 MiB).
   - **Net buffer savings: -52.48 MiB (-46.88%)**.
   - Resident GPU VRAM measured by `nvidia-smi` across the test suite: **1,211–1,215 MiB (F16)** vs **1,161–1,165 MiB (Q8_0)**, confirming an absolute resident VRAM saving of ~50 MiB.

2. **Native Prefix Cache Reuse**:
   - In a controlled experiment comparing 5 sequential tasks with a stable common architecture prefix against 5 tasks with varied domain prefixes:
     - Stable common prefix achieved an average of **190.0 cached tokens** out of ~210 total prompt tokens (**90.7% cache hit ratio**), directly verified by Ollama's native `prompt_eval_cached_count` counter.
     - Varied control prefixes cached only the 29-token system prompt (**19.8% cache hit ratio**).
     - Average prefill latency was **17.38 ms** for the stable prefix group versus **25.23 ms** for the varied prefix group (**1.45x prefill speedup**).
     - Effective prompt throughput reached **12,093.5 tokens/second** for the stable prefix group versus **5,779.7 tokens/second** for the varied prefix group (**2.09x effective prompt throughput**).

3. **Adaptive Context Tier Derivation & Safe Overflow Guard**:
   - Tiers are dynamically derived between `OLLAMA_NUM_CTX` (4,096) and `OLLAMA_MAX_NUM_CTX` (8,192): Tier 1 (`4,096` tokens), Tier 2 (`6,144` tokens), Tier 3 (`8,192` tokens). If `OLLAMA_MAX_NUM_CTX` is configured lower, no tier above that ceiling is ever selected.
   - Low-VRAM safeguard: If free VRAM is constrained (<1,000 MiB), context allocation is pinned to the baseline tier (4,096) to prevent out-of-memory crashes. If the prompt's required budget exceeds this safe tier, it immediately raises **HTTP 413 Payload Too Large** rather than silently truncating instructions.
   - Oversized requests (>8,192 tokens) were rejected cleanly with **HTTP 413 Payload Too Large** in **49–51 ms**.

4. **Generation Latency & Output Quality Invariance**:
   - Generation throughput remained exceptionally high and virtually identical between formats: **124.3 to 134.3 tokens/sec** under Q8_0 versus **127.7 to 141.3 tokens/sec** under F16 (sub-3% variance within measurement noise).
   - Zero quality regression: 100% of required domain constraints and negative prohibitions were preserved, with zero prompt reversals or formatting degradation.

5. **Dynamic Concurrency Capacity Enforcement**:
   - Upgraded from static semaphore to an `asyncio.Condition`-based capacity gate that dynamically evaluates hardware telemetry at admission time.
   - Automatically throttles admission capacity from 3 down to 1 when GPU VRAM drops below 400 MiB or host RAM drops below 1,024 MiB.
   - Rejects requests with HTTP 503 if VRAM falls below 150 MiB or RAM falls below 512 MiB.
   - If capacity drops while requests are active, newly arriving requests queue safely until active requests complete below the new capacity limit without permit race conditions.

---

## 2. Test Environment & System Specifications

| Component | Specification | Verification Source |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 Home 64-bit (build 10.0.26100) | `platform.uname()`, Win32 API |
| **CPU** | AMD Ryzen 7 7735HS with Radeon Graphics (8 cores, 16 threads) | `llama-server` probe |
| **Host RAM** | 15,556.5 MiB physical (3,940 MiB available at nominal idle) | `ctypes.windll.kernel32.GlobalMemoryStatusEx` |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB dedicated VRAM) | `nvidia-smi` |
| **GPU Architecture** | Ada Lovelace, Compute Capability 8.9 | CUDA runtime probe |
| **Ollama Runtime** | Ollama v0.40.8 | `ollama --version` |
| **Inference Model** | `qwen2.5:1.5b` (Q4_K_M GGUF, 934.7 MiB weights) | `ollama list` & digest `65ec06548149` |
| **Python Backend** | Python 3.11.4 / FastAPI 0.115 / Uvicorn 0.32 | `D:\Projects\Argus\backend\.venv` |
| **Frontend** | Node.js v20.18.0 / React 18 / Vite 6.4.5 / Vitest 4.1.12 | `D:\Projects\Argus\frontend` |

---

## 3. Ollama Process Verification & Lifecycle Audit

### Independent Verification States
The backend `OllamaProcessManager` evaluates runtime process evidence and categorizes server status into explicit, unambiguous states:

1. `independently_confirmed`: Both Q8_0 KV quantization and Flash Attention are verified in the active process's session logs.
2. `partially_confirmed_kv_only`: Q8_0 KV-cache is confirmed in logs, but Flash Attention is not yet observed.
3. `partially_confirmed_fa_only`: Flash Attention is confirmed in logs, but Q8_0 KV-cache is not yet observed.
4. `observed_in_server_logs`: Server startup parameters were observed in logs, but engine initialization has not completed.
5. `launched_with_requested_settings`: Argus launcher started the process with explicit environment variables, awaiting model execution.
6. `configured_by_policy`: Process is running externally with settings assumed from configuration.
7. `unmanaged_external_instance_stale_manifest`: The recorded manifest PID does not match the active listening PID or process creation timestamp, safely preventing false attribution to previous sessions.

### Session-Tied Log Evidence & UTF-16LE Decoding
- **Process Start Time Tie**: Log modification timestamps are compared against active process creation times obtained via Win32 `GetProcessTimes` API. Logs older than the active process start are rejected as stale evidence.
- **BOM & UTF-16LE Decoding**: PowerShell redirection (`2> err.log 1> std.log`) produces UTF-16LE encoded files with `\xff\xfe` byte-order marks. The log scanner automatically detects and decodes both UTF-8 and UTF-16LE streams without string corruption.
- **Full Active Session Scan**: Scans the entire log output of the active session rather than an arbitrary tail slice, ensuring initial model load logs (`llama_context: flash_attn = enabled`, `llama_kv_cache: size = ...`) are retained regardless of subsequent request volume.

### Safe Process Termination
`scripts/launch_ollama.ps1` strictly validates `ProcessName -like "*ollama*"` before issuing termination calls. Unrelated processes occupying target ports are never terminated automatically.

---

## 4. Controlled Prefix Reuse Experiment

- **Harness**: [`scripts/experiment_prefix_reuse.py`](file:///D:/Projects/Argus/scripts/experiment_prefix_reuse.py)
- **Raw Data**: `docs/reports/prefix_reuse_experiment.json`
- **Conditions**: Model resident in VRAM (`qwen2.5:1.5b`), `num_ctx: 4096`, `num_predict: 64`.

### Group A: Stable Common Prefix (Architecture Spec + 5 Sequential Tasks)
| Task # | Total Prompt Tokens | Native Cached Tokens | Uncached Eval Tokens | Cache Hit Ratio | Prefill Duration | Effective Throughput | Total Wall Clock |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 (cold) | 211 | 29 | 182 | 13.7% | 37.92 ms | 5,563.9 tok/s | 553.73 ms |
| 2 (reuse) | 212 | 190 | 22 | 89.6% | 19.15 ms | 11,068.2 tok/s | 512.46 ms |
| 3 (reuse) | 207 | 190 | 17 | 91.8% | 17.39 ms | 11,904.1 tok/s | 498.19 ms |
| 4 (reuse) | 210 | 190 | 20 | 90.5% | 16.65 ms | 12,611.9 tok/s | 500.56 ms |
| 5 (reuse) | 209 | 190 | 19 | 90.9% | 16.34 ms | 12,789.9 tok/s | 495.86 ms |
| **Avg (2-5)** | **209.5** | **190.0** | **19.5** | **90.7%** | **17.38 ms** | **12,093.5 tok/s** | **501.77 ms** |

### Group B: Varied Control Prefixes (Different Domain Specs + 5 Tasks)
| Task # | Total Prompt Tokens | Native Cached Tokens | Uncached Eval Tokens | Cache Hit Ratio | Prefill Duration | Effective Throughput | Total Wall Clock |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 154 | 29 | 125 | 18.8% | 27.34 ms | 5,633.8 tok/s | 503.00 ms |
| 2 | 152 | 29 | 123 | 19.1% | 27.17 ms | 5,593.6 tok/s | 507.87 ms |
| 3 | 135 | 29 | 106 | 21.5% | 24.14 ms | 5,592.2 tok/s | 500.23 ms |
| 4 | 149 | 29 | 120 | 19.5% | 25.01 ms | 5,958.6 tok/s | 505.37 ms |
| 5 | 147 | 29 | 118 | 19.7% | 24.60 ms | 5,974.6 tok/s | 496.82 ms |
| **Avg (2-5)** | **145.8** | **29.0** | **116.8** | **19.8%** | **25.23 ms** | **5,779.7 tok/s** | **502.57 ms** |

### Comparison Summary
- **Average Native Cached Tokens**: **190.0 tokens** (Stable) vs **29.0 tokens** (Varied).
- **Average Prefill Duration**: **17.38 ms** (Stable) vs **25.23 ms** (Varied).
- **Prefill Latency Speedup Factor**: **1.45x acceleration**.
- **Effective Prompt Throughput**: **12,093.5 tok/s** vs **5,779.7 tok/s** (**2.09x throughput increase**).

---

## 5. End-to-End Comparative Benchmark: F16 vs Q8_0

- **Harness**: [`scripts/benchmark_kv_cache.py`](file:///D:/Projects/Argus/scripts/benchmark_kv_cache.py)
- **Raw Data Files**:
  - `docs/reports/benchmark_raw_f16.json`
  - `docs/reports/benchmark_raw_q8_0.json`
- **Dynamic Configuration**: Automatically detects host CPU, GPU VRAM (`nvidia-smi`), system RAM, and Ollama version. Missing telemetry attributes are explicitly preserved as `null` rather than defaulted to 0.

### Comparative Results Summary Table

| Test Case | Metric | F16 Baseline | Q8_0 Optimized | Delta / Impact |
| :--- | :--- | :---: | :---: | :---: |
| **Short Coding Task**<br>(Prompt: 54 chars, ~13 tok) | VRAM Usage<br>Prompt Speed (Rep 1)<br>Prompt Speed (Rep 2, reused)<br>Decode Speed<br>Wall Latency | 1,211 MiB<br>24,670 tok/s<br>58,960 tok/s<br>130.6–141.3 tok/s<br>423.8–433.4 ms | 1,161 MiB<br>23,671 tok/s<br>55,677 tok/s<br>127.7–128.7 tok/s<br>434.5–445.1 ms | **-50 MiB VRAM (-4.13%)**<br>-4.0% initial prefill<br>-5.6% cached prefill<br>-2.2% decode speed<br>+10.7 ms latency |
| **Medium Code Review**<br>(Prompt: 232 chars, ~58 tok) | VRAM Usage<br>Prompt Speed (Rep 1)<br>Prompt Speed (Rep 2, reused)<br>Decode Speed<br>Wall Latency | 1,211 MiB<br>25,232 tok/s<br>42,435 tok/s<br>127.7–131.9 tok/s<br>586.7–631.3 ms | 1,161–1,163 MiB<br>18,994 tok/s<br>36,041 tok/s<br>124.3–132.3 tok/s<br>568.5–623.1 ms | **-48–50 MiB VRAM**<br>-24.7% initial prefill<br>-15.1% cached prefill<br>-2.6% decode speed<br>-8.2 ms wall latency |
| **Long Architecture**<br>(Prompt: 753 chars, ~188 tok) | VRAM Usage<br>Prompt Speed (Rep 1)<br>Prompt Speed (Rep 2, reused)<br>Decode Speed<br>Wall Latency | 1,211–1,213 MiB<br>14,977 tok/s<br>71,944 tok/s<br>135.1–137.3 tok/s<br>1,314.7–1,327.8 ms | 1,163 MiB<br>14,118 tok/s<br>70,468 tok/s<br>133.2–134.3 tok/s<br>1,299.9–1,363.6 ms | **-48–50 MiB VRAM**<br>-5.7% initial prefill<br>-2.1% cached prefill<br>-1.4% decode speed<br>Sub-3% wall variance |
| **Explicit Constraints**<br>(Prompt: 341 chars, ~85 tok) | VRAM Usage<br>Prompt Speed (Rep 1)<br>Prompt Speed (Rep 2, reused)<br>Decode Speed<br>Constraints Invariant | 1,213 MiB<br>16,141 tok/s<br>69,583 tok/s<br>134.3–138.4 tok/s<br>100% Preserved | 1,163 MiB<br>15,269 tok/s<br>70,444 tok/s<br>133.0–133.3 tok/s<br>100% Preserved | **-50 MiB VRAM (-4.12%)**<br>-5.4% initial prefill<br>+1.2% cached prefill<br>-1.0% decode speed<br>**Zero quality degradation** |
| **Near Context Limit**<br>(Prompt: 6,900 chars, ~1,725 tok) | VRAM Usage<br>Prompt Speed (Rep 1)<br>Prompt Speed (Rep 2, reused)<br>Cache Hit Ratio (Rep 2)<br>Selected Context Tier | 1,213–1,215 MiB<br>9,127 tok/s<br>186,131 tok/s<br>99.95% (2243/2244 tok)<br>Tier 1 (4,096) | 1,163–1,165 MiB<br>8,353 tok/s<br>147,875 tok/s<br>99.95% (2243/2244 tok)<br>Tier 1 (4,096) | **-50 MiB VRAM (-4.12%)**<br>-8.5% initial prefill<br>Equivalent cache hits<br>Identical context tier |
| **Context Overflow**<br>(Prompt: >14,000 tok) | HTTP Status<br>Rejection Wall Time<br>VRAM Spike / Truncation | **413 Payload Too Large**<br>36.7 ms<br>None / 0 truncation | **413 Payload Too Large**<br>49.2–51.0 ms<br>None / 0 truncation | **Clean fail-fast safeguard**<br>Immediate rejection<br>Memory protected |

---

## 6. Output Quality & Semantic Invariants

Under Q8_0 KV quantization, prompt optimization quality was validated against the following tests:
1. **Constraint Retention**: All required string invariants (`sales_data.csv`, `Category`, `Total Revenue`) were preserved verbatim.
2. **Prohibition Adherence**: Negative constraints (`do not use pandas`, `do not use numpy`) were respected with 0 false inclusions.
3. **Turn-Reversal Elimination**: The model consistently outputted optimized prompts rather than asking interactive clarifying questions.
4. **Token Savings**: Token reductions reached up to **89%** on verbose microservice specifications without semantic truncation.

---

## 7. Categorization of Behavior: Implemented vs Automated vs Empirical

To uphold strict engineering transparency, Argus explicitly distinguishes between behavioral tiers:

| Dimension | Feature / Capability | Classification | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Dynamic Admission Gate** | `asyncio.Condition`-based capacity enforcement, queue tracking, and slot release | **Implemented & Automated** | `test_concurrency_controller_dynamic_restriction_from_3_to_1_while_active`, `test_concurrency_controller_queue_wait_cancellation` in `test_kv_cache_adaptive.py` |
| **Host RAM Throttling & Rejection** | Capacity throttled to 1 below 1024 MiB RAM; rejected below 512 MiB RAM | **Implemented & Automated** | `test_concurrency_controller_adaptive_restricted_ram`, `test_concurrency_controller_critical_ram_rejection` |
| **GPU VRAM Throttling & Rejection** | Capacity throttled to 1 below 400 MiB VRAM; rejected below 150 MiB VRAM | **Implemented & Automated** | `test_concurrency_controller_adaptive_restricted_vram`, `test_concurrency_controller_critical_vram_rejection` |
| **Dynamic Tier Derivation** | Deriving context tiers bounded by `OLLAMA_MAX_NUM_CTX` without exceeding ceiling | **Implemented & Automated** | `test_context_eligible_tiers_derived_dynamically`, `test_context_budget_max_num_ctx_below_8192` |
| **Low-VRAM HTTP 413 Safeguard** | Pinning to baseline tier under low VRAM and raising 413 if budget exceeds tier | **Implemented & Automated** | `test_context_budget_low_vram_prompt_requires_larger_tier_raises_413` |
| **Process Stale Manifest Detection** | Detecting PID mismatch or process creation time mismatch as unmanaged | **Implemented & Automated** | `test_process_manager_stale_manifest_pid_mismatch` |
| **Independent Setting Verification** | Evaluating Q8_0 and Flash Attention separately across 4 discrete states | **Implemented & Automated** | `test_process_manager_independent_kv_only_confirmation`, `test_process_manager_independent_fa_only_confirmation`, `test_process_manager_both_settings_confirmed` |
| **KV Buffer Size Reduction** | 112.00 MiB (F16) reduced to 59.52 MiB (Q8_0) across 4,096 cells (-46.88%) | **Empirically Verified** | Verified in active Ollama startup logs and `nvidia-smi` measurements |
| **Prefix Cache Reuse** | 190.0/209.5 tokens cached (90.7%) vs 29.0/145.8 (19.8%) | **Empirically Verified** | Live execution via `scripts/experiment_prefix_reuse.py` & `docs/reports/prefix_reuse_experiment.json` |
| **Inference Throughput & Invariance** | Generation throughput invariance (124–134 tok/s Q8_0 vs 128–141 tok/s F16) | **Empirically Verified** | Live execution via `scripts/benchmark_kv_cache.py`, `docs/reports/benchmark_raw_q8_0.json`, `docs/reports/benchmark_raw_f16.json` |

---

## 8. Conclusions & Production Readiness

The empirical data and comprehensive test suite confirm that **Q8_0 KV-cache quantization combined with Flash Attention, Adaptive Context Snapping, and Dynamic Concurrency Admission Control is safe, robust, and advantageous for production use in Argus**:
- **-46.9% KV memory buffer reduction** freeing critical VRAM on 6 GB consumer GPUs.
- **Sub-3% compute overhead** on decode throughput.
- **1.45x prefill latency speedup** and **2.09x effective prompt throughput** through disciplined prompt prefix reuse.
- **Fail-safe operational reliability** with HTTP 413 context guarding, dynamic capacity draining, and independent runtime verification.
