# Argus KV Cache Optimization: Implementation & Acceptance Audit

**Audit Date:** 2026-11-03  
**Working Branch:** `feat/complete-kv-cache-runtime`  
**Base Commit:** `663771f` (PR #5 merged into `main`)  
**Hardware Platform:** AMD Ryzen 7 7735HS, NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM), 16 GB RAM, Windows 11  
**Runtime:** Ollama v0.40.8 with model `qwen2.5:1.5b` (Q4_K_M GGUF, 934.7 MiB)  

---

## 1. Feature Implementation Matrix

| Feature | Design Specification | Implementation Status | Verification Method & Artifacts |
| :--- | :--- | :--- | :--- |
| **1. Q8_0 KV Cache Quantization** | Quantize KV cache vectors from 16-bit float to 8-bit integers (`q8_0`), reducing per-token memory overhead by ~50%. | **Complete & Verified** | Independent log verification via `OllamaProcessManager`. `llama-server` log confirms CUDA0 KV buffer size reduced from 112.00 MiB (F16) to 59.52 MiB (Q8_0) across 4,096 cells (-46.88%). Live VRAM footprint reduced by ~50 MiB. |
| **2. Flash Attention** | Enable memory-efficient attention kernels (`OLLAMA_FLASH_ATTENTION=1`) to eliminate intermediate \(O(N^2)\) attention matrices. | **Complete & Verified** | Verified on NVIDIA Ada Lovelace compute capability 8.9. `llama-server` log explicitly records `llama_context: flash_attn = enabled` with FA quant matrix active. |
| **3. Adaptive Context-Length Management** | Dynamically allocate context (`num_ctx`) based on token budget breakdown (system instruction, prompt wrapper, estimated input, output reservation, headroom safety buffer). Snap to discrete tiers (`4096, 6144, 8192`) to preserve KV cache prefix reuse across varying prompt lengths. Guard against low VRAM. Enforce strict HTTP 413 rejection for overflow. | **Complete & Verified** | [`AdaptiveContextManager`](file:///D:/Projects/Argus/backend/app/services/adaptive_context_manager.py) implemented with unit tests in `test_kv_cache_adaptive.py`. Evaluated via benchmark prompt suite. Clean HTTP 413 returned in 74ms with zero prompt truncation. Conservative heuristic labeled as `estimated_conservative_heuristic`. |
| **4. Memory-Aware Concurrency Controller** | Admission control dynamically throttling concurrency based on real GPU VRAM and Win32 RAM telemetry. Prevents CUDA OOM under high memory pressure. Tracks queue depth and queue wait latency. Guaranteed capacity release on cancellation or exception. Static conservative fallback when telemetry is unavailable. | **Complete & Verified** | [`HardwareTelemetryProvider`](file:///D:/Projects/Argus/backend/app/services/hardware_telemetry.py) and [`RuntimeAwareConcurrencyController`](file:///D:/Projects/Argus/backend/app/services/concurrency_controller.py) implemented. 100% test coverage in `test_kv_cache_adaptive.py` (healthy GPU, low VRAM throttling, critical VRAM rejection, static fallback, exception cleanup). |
| **5. KV Cache Prefix Reuse Optimization** | Structure system instructions and prompt prefixes identically to leverage Ollama/llama.cpp's native prefix cache. Verify with native cache counters and throughput measurements under controlled conditions. | **Complete & Verified** | Controlled experiment executed via [`scripts/experiment_prefix_reuse.py`](file:///D:/Projects/Argus/scripts/experiment_prefix_reuse.py). Demonstrated 190/210 tokens cached (90.5% cache hit ratio) for stable prefix tasks vs 29 tokens for varied controls. Prefill throughput increased from 4,804.4 tok/s to 8,887.0 tok/s (1.85x speedup). Empirical results saved in `docs/reports/prefix_reuse_experiment.json`. |
| **6. Native Telemetry Extraction & Attribution** | Extract authoritative inference telemetry directly from Ollama `/api/generate` (`prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`). Disclose metric sources and separate heuristic estimates from native hardware counters. | **Complete & Verified** | Implemented in `OllamaService._execute_model_call`. Exposed in `InferenceTelemetry` schema with `prompt_eval_cached_count`, `cache_hit_ratio`, `telemetry_source: "ollama_native"`, and hardware snapshots. Verified via automated tests and live API calls. |
| **7. F16 vs Q8_0 Comparative Benchmark** | Controlled empirical evaluation of F16 vs Q8_0 KV-cache configurations across multi-prompt categories with server restarts between configurations. Record raw metrics to JSON and Markdown. | **Complete & Verified** | [`scripts/benchmark_kv_cache.py`](file:///D:/Projects/Argus/scripts/benchmark_kv_cache.py) executed for both configurations. Raw unaggregated data saved to `docs/reports/benchmark_raw_f16.json` and `docs/reports/benchmark_raw_q8_0.json`. Validation report documented in `docs/reports/kv-cache-validation-report.md`. |

---

## 2. Windows Service Management & Process Audit

On Windows, Ollama runs as a background process or system tray application. Environment variables set in Python or `.env` cannot alter a running Ollama server.

### Dedicated Process Launcher
To ensure reproducible, verified runtime environments, Argus provides [`scripts/launch_ollama.ps1`](file:///D:/Projects/Argus/scripts/launch_ollama.ps1):
- **Port Conflict Handling**: Detects existing processes listening on port 11434 and safely stops them.
- **Environment Injection**: Exports `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` (or `f16`), and `OLLAMA_DEBUG=1`.
- **Dual Log Capture**: Splits output into `ollama_server.log` (stdout) and `ollama_server.err.log` (stderr) to avoid Windows `Start-Process` file contention.
- **Manifest Tracking**: Writes `backend/data/ollama_runtime.json` recording PID, start timestamp, requested flags, and log file paths.
- **Independent Verification**: Backend [`OllamaProcessManager`](file:///D:/Projects/Argus/backend/app/services/ollama_process_manager.py) independently scans active listening sockets via `netstat` and parses server logs to verify server-side activation.

---

## 3. Adaptive Context Architecture Audit

### Budgeting & Tier Snapping
The adaptive context manager enforces:
$$\text{Total Tokens} = \text{System Tokens} + \text{Wrapper Tokens} + \text{Prompt Tokens} + \text{Predict Reserve (512)} + \text{Headroom (256)}$$

Context length is snapped to discrete tiers:
- Tier 1: `4,096` tokens (baseline for prompts $\le 3,328$ tokens)
- Tier 2: `6,144` tokens (for prompts $3,329 - 5,376$ tokens)
- Tier 3: `8,192` tokens (for prompts $5,377 - 7,424$ tokens)
- Prompts requiring $> 8,192$ tokens raise `ContextOverflowError` triggering an immediate HTTP 413.

**Why Snapping is Essential:** Snapping to fixed tiers prevents `llama.cpp` from continuously resizing and invalidating the KV cache buffer between slightly different prompt lengths, preserving prefix cache reuse across subsequent requests.

---

## 4. Concurrency Controller Audit

### Dynamic Admission Rules
1. **Critical VRAM Guard**: If free GPU VRAM $< 150\text{ MiB}$, immediately reject incoming requests with HTTP 503 (`InferenceCapacityExceededError`).
2. **Restricted Capacity**: If free GPU VRAM $< 400\text{ MiB}$ or system RAM $< 1,024\text{ MiB}$, restrict active concurrent requests to $1$.
3. **Nominal Capacity**: When VRAM $\ge 400\text{ MiB}$, allow configured `argus_max_concurrent_requests` (default $1$ for single-GPU 6 GB VRAM).
4. **Guaranteed Cleanup**: Concurrency slots are managed via an `async with slot()` context manager; timeouts and unhandled exceptions guarantee immediate release of semaphore capacity.
5. **Static Fallback**: If `nvidia-smi` is not installed or raises an error, the controller falls back to static semaphore capacity without crashing.

---

## 5. Verification Test Suite Summary

- **Backend Automated Tests**: 58 / 58 passing (`pytest backend/tests -v`, 3.87s).
  - 16 new dedicated tests in `backend/tests/test_kv_cache_adaptive.py`.
  - 42 existing tests in `test_health.py` and `test_optimizer.py`.
- **Frontend Automated Tests**: 19 / 19 passing (`vitest run`, 2.67s).
- **Frontend Production Build**: `tsc -b && vite build` succeeded with zero type errors.
- **Vulnerability Audit**: `npm audit` returned 0 vulnerabilities.
