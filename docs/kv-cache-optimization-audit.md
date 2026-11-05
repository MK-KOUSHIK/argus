# Argus KV Cache Optimization: Implementation & Acceptance Audit

**Audit Date:** 2026-11-04  
**Working Branch:** `feat/output-budget-control`  
**Base Branch:** `main`  
**Hardware Platform:** AMD Ryzen 7 7735HS, NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), 16 GB RAM, Windows 10/11  
**Runtime:** Ollama v0.40.8 with model `qwen2.5:1.5b` (Q4_K_M GGUF, 934.7 MiB)  

---

## 1. Feature Implementation Matrix

| Feature | Design Specification | Implementation Status | Verification Method & Artifacts |
| :--- | :--- | :--- | :--- |
| **1. Q8_0 KV Cache Quantization** | Quantize KV cache vectors from 16-bit float to 8-bit integers (`q8_0`), reducing per-token memory overhead by ~50%. | **Complete & Verified** | Independent log verification via `OllamaProcessManager`. `llama-server` log confirms CUDA0 KV buffer size reduced from 112.00 MiB (F16) to 59.52 MiB (Q8_0) across 4,096 cells (-46.88%). Live VRAM footprint reduced by ~50 MiB. |
| **2. Flash Attention** | Enable memory-efficient attention kernels (`OLLAMA_FLASH_ATTENTION=1`) to eliminate intermediate \(O(N^2)\) attention matrices. | **Complete & Verified** | Verified on NVIDIA Ada Lovelace compute capability 8.9. `llama-server` log explicitly records `llama_context: flash_attn = enabled` independently from KV quantization. |
| **3. Adaptive Context-Length Management** | Dynamically allocate context (`num_ctx`) based on token budget breakdown (system instruction, prompt wrapper, estimated input, output reservation, headroom safety buffer). Dynamically derive eligible tiers bounded by `OLLAMA_MAX_NUM_CTX`. Guard against low VRAM (<1,000 MiB) by enforcing safe baseline tier (4,096) and raising HTTP 413 if the prompt exceeds that tier without silent truncation. | **Complete & Verified** | [`AdaptiveContextManager`](file:///D:/Projects/Argus/backend/app/services/adaptive_context_manager.py) implemented with unit tests in `test_kv_cache_adaptive.py`. Evaluated via benchmark prompt suite. Clean HTTP 413 returned in 49ms with zero prompt truncation. Full token source attribution preserved. |
| **4. Genuine Dynamic Concurrency Capacity Enforcement** | Dynamic admission control enforcing true capacity bounds via `asyncio.Condition`. Dynamically throttles capacity (e.g. from 3 down to 1) when GPU VRAM <400 MiB or host RAM <1024 MiB. Prevents new admissions until active requests fall below reduced capacity. Rejects requests with HTTP 503 if VRAM <150 MiB or RAM <512 MiB. Cleanly removes cancelled requests from queue depth tracking. | **Complete & Verified** | [`HardwareTelemetryProvider`](file:///D:/Projects/Argus/backend/app/services/hardware_telemetry.py) and [`RuntimeAwareConcurrencyController`](file:///D:/Projects/Argus/backend/app/services/concurrency_controller.py) implemented. 100% test coverage in `test_kv_cache_adaptive.py` (healthy GPU, low VRAM throttling, critical RAM/VRAM rejection, dynamic throttling from 3 to 1 while active, queue cancellation safety, static fallback). |
| **5. KV Cache Prefix Reuse Optimization** | Structure system instructions and prompt prefixes identically to leverage Ollama/llama.cpp's native prefix cache. Verify with native cache counters (`prompt_eval_cached_count`), separate cached vs uncached tokens, report cache hit ratio, and measure effective throughput vs uncached evaluation rate. | **Complete & Verified** | Controlled experiment executed via [`scripts/experiment_prefix_reuse.py`](file:///D:/Projects/Argus/scripts/experiment_prefix_reuse.py). Demonstrated 190.0/209.5 tokens cached (90.7% cache hit ratio) for stable prefix tasks vs 29.0/145.8 (19.8%) for varied controls. Prefill latency improved from 25.23 ms to 17.38 ms (1.45x speedup), with effective prompt throughput reaching 12,093.5 tok/s (2.09x increase). Empirical results saved in `docs/reports/prefix_reuse_experiment.json`. |
| **6. Output Budget Control (Phase 4.1)** | Replace hardcoded 512 `num_predict` with intelligent dynamic generation token budgeting. Parse explicit constraints (word/sentence counts, brevity cues), map task category baselines, scale by mode/complexity, coordinate with context tier calculation without duplicating calculations, enforce bounds `[64, 2048]`, clamp to available headroom, and reject with HTTP 413 when headroom < 64 tokens. | **Complete & Verified** | [`OutputBudgetController`](file:///D:/Projects/Argus/backend/app/services/output_budget_controller.py) implemented with 14 unit and integration tests in `test_output_budget.py`. Empirical benchmark recorded in `docs/reports/output_budget_benchmark.json` and validated in `docs/reports/output-budget-validation-report.md`. Conserves 35.0% unneeded headroom with 100% natural completions (`done_reason == "stop"`). |
| **7. Native Telemetry Extraction & Attribution** | Extract authoritative inference telemetry directly from Ollama `/api/generate` (`prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`, `done_reason`). Disclose metric sources and separate heuristic estimates from native hardware counters. Missing telemetry preserved as `None` rather than 0. | **Complete & Verified** | Implemented in `OllamaService._execute_model_call`. Exposed in `InferenceTelemetry` and `OutputBudgetTelemetry` schemas with `done_reason`, `hit_budget_limit`, `budget_utilization_percent`, `prompt_eval_cached_count`, `cache_hit_ratio`, `telemetry_source: "ollama_native"`, and hardware snapshots. |
| **8. F16 vs Q8_0 Comparative Benchmark** | Controlled empirical evaluation of F16 vs Q8_0 KV-cache configurations across multi-prompt categories with server restarts between configurations. Automatic hardware and model detection. Record raw metrics to JSON and Markdown. | **Complete & Verified** | [`scripts/benchmark_kv_cache.py`](file:///D:/Projects/Argus/scripts/benchmark_kv_cache.py) executed for both configurations. Raw unaggregated data saved to `docs/reports/benchmark_raw_f16.json` and `docs/reports/benchmark_raw_q8_0.json`. Validation report documented in `docs/reports/kv-cache-validation-report.md`. |

---

## 2. Windows Service Management & Process Audit

On Windows, Ollama runs as a background process or system tray application. Environment variables set in Python or `.env` cannot alter an already-running Ollama server.

### Dedicated Process Launcher
To ensure reproducible, verified runtime environments, Argus provides [`scripts/launch_ollama.ps1`](file:///D:/Projects/Argus/scripts/launch_ollama.ps1):
- **Safe Process Cleanup**: Inspects active processes and validates `ProcessName -like "*ollama*"` before stopping. Refuses to terminate unrelated PIDs on the target port unless explicitly overridden with `-Force`.
- **Environment Injection**: Exports `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` (or `f16`), and `OLLAMA_DEBUG=1`.
- **Dual Log Capture**: Splits output into `ollama_server.log` (stdout) and `ollama_server.err.log` (stderr) with UTF-16LE and UTF-8 automatic decoding.
- **Manifest Tracking**: Writes `backend/data/ollama_runtime.json` recording session ID, PID, start timestamp, requested flags, and log file paths.
- **Independent Verification**: Backend [`OllamaProcessManager`](file:///D:/Projects/Argus/backend/app/services/ollama_process_manager.py) independently evaluates active listening sockets, compares process creation timestamps via Win32 `GetProcessTimes`, rejects stale logs, and confirms Q8_0 and Flash Attention activation separately.

---

## 3. Adaptive Context & Output Budget Architecture Audit

### Budgeting & Dynamic Tier Derivation
The adaptive context manager coordinates with the output budget controller:
$$\text{Total Required Context} = \text{System Tokens} + \text{Wrapper Tokens} + \text{Input Prompt Tokens} + \text{Dynamic Output Budget} + \text{Headroom (256)}$$

Eligible tiers are dynamically derived from baseline `OLLAMA_NUM_CTX` (4,096) up to `OLLAMA_MAX_NUM_CTX`:
- Tier 1: `4,096` tokens (baseline)
- Tier 2: `6,144` tokens
- Tier 3: `8,192` tokens (or bounded by `OLLAMA_MAX_NUM_CTX`)
- If `OLLAMA_MAX_NUM_CTX` is configured below 8,192 (e.g. 6,144), tiers above that ceiling are excluded and never selected.
- **Reachable Low-VRAM Safeguard**: Under constrained VRAM (<1,000 MiB free), allocation is restricted to the safe baseline tier (4,096). If the prompt's required budget exceeds this baseline tier, the system immediately raises `ContextOverflowError` (HTTP 413 Payload Too Large) rather than risking GPU out-of-memory errors or silently truncating user instructions.
- Full budget breakdown and attribution (`prompt_token_source`, `output_budget_source`) are returned transparently.

---

## 4. Concurrency Controller Audit

### Dynamic Capacity Enforcement
1. **Capacity Enforcement via `asyncio.Condition`**: Replaced static semaphore with an admission gate that dynamically re-evaluates hardware conditions.
2. **Dynamic Throttling**: When active requests are running at nominal capacity (e.g. 3) and conditions deteriorate (VRAM <400 MiB or host RAM <1024 MiB), dynamic capacity drops to 1. Existing active requests continue safely, but new requests are blocked until active requests drain below 1.
3. **Critical Rejection**: If free GPU VRAM $< 150\text{ MiB}$ or host RAM $< 512\text{ MiB}$, incoming requests are rejected immediately with HTTP 503 (`InferenceCapacityExceededError`).
4. **Queue Cancellation Safety**: Cancelled or timed-out requests in the wait queue are decremented cleanly from `queue_depth`.
5. **Static Fallback**: If telemetry queries fail, the controller smoothly falls back to configured static capacity (`argus_max_concurrent_requests`) without breaking API contracts.

---

## 5. Output Budget Control Audit (Phase 4.1)

1. **Deterministic Directives**:
   - Word limit patterns: `under 50 words` -> $\lceil 50 \times 1.33 \rceil + 64 = 131\text{ tokens}$.
   - Sentence limit patterns: `in one sentence` -> $1 \times 35 + 64 = 99\text{ tokens}$.
   - Qualitative brevity: `tl;dr` -> 160 tokens.
2. **Task Baselines**: 256 for summarization/extraction, 384 for data analysis/general, 512 for coding/writing/planning.
3. **Mode Scaling**: 0.75x for concise, 1.0x for balanced, 1.2x for structure.
4. **Context Clamping**: Safely clamps candidate budget if context headroom is constrained.
5. **Rejection Safeguard**: If headroom $< 64\text{ tokens}$, immediately returns HTTP 413 without silent truncation.
6. **Native Completion Attribution**: Distinguishes natural EOS (`done_reason: stop`) from generation ceiling truncation (`done_reason: length`).

---

## 6. Verification Test Suite Summary

- **Backend Automated Tests**: **79 / 79 passing** (`pytest backend/tests -v`, 6.68s).
  - 14 tests in `backend/tests/test_output_budget.py` covering task baselines, mode scaling, explicit word/sentence directives, brevity cues, override priority, bounds clamping, context headroom clamping, HTTP 413 overflow rejection, context manager coordination, `done_reason` parsing, and zero-fabrication null handling.
  - 23 dedicated tests in `backend/tests/test_kv_cache_adaptive.py`.
  - 42 existing tests in `test_health.py` and `test_optimizer.py`.
- **Frontend Automated Tests**: **19 / 19 passing** (`vitest run`, 1.96s).
- **Frontend Production Build**: `tsc -b && vite build` succeeded with zero type errors.
- **Vulnerability Audit**: `npm audit` returned 0 vulnerabilities.
