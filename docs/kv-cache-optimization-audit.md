# Argus KV Cache Optimization: Compatibility Audit

**Audit basis:** `phase-4-main-llm-benchmark` at commit `160a24950351da808294761127c6eb7c954582a2`, plus the public `awesome-llm-token-optimization` repository README (accessed 2026-11-03).

## Current Argus baseline

- FastAPI backend and React/TypeScript frontend.
- Local inference through Ollama HTTP `/api/generate`.
- Default local model: `qwen2.5:1.5b`.
- Generation is currently non-streaming.
- `OllamaService._execute_model_call` currently returns generated text and wall-clock duration only; Ollama's token/timing telemetry is discarded.
- Generation options currently include temperature and top-p only.
- No explicit KV-cache configuration, context budget policy, request concurrency limit, or cache telemetry is currently exposed by Argus.
- The frontend contains a legacy simulated optimizer service. It must not be used to report real runtime measurements.

## Reference technologies and Argus decisions

| Reference | Core capability | Compatibility with current Ollama deployment | Decision |
|---|---|---|---|
| vLLM / PagedAttention | Paged KV allocation, continuous batching, prefix caching | Separate serving engine; not a drop-in Ollama option | Keep as a future alternate inference adapter |
| SGLang / RadixAttention | Radix-tree prefix KV reuse and session-aware cache behavior | Separate serving engine; not exposed through Ollama's current API | Future adapter; do not claim this exists in Ollama |
| GPUStack | GPU resource orchestration for serving engines | Infrastructure layer, unnecessary for one local Ollama process | Not part of local MVP |
| NVIDIA Dynamo | Distributed prefill/decode and KV-aware routing | Distributed serving stack, not an Ollama plugin | Future multi-node deployment option |
| llm-d | Kubernetes-native serving and tiered KV offload | Requires a Kubernetes inference deployment | Not part of local MVP |
| Mooncake | Distributed KV transfer and cross-instance reuse | Requires compatible serving integration | Future distributed deployment option |
| NVIDIA kvpress / R-KV / KVzip / KVCache-Factory | Research and runtime KV compression/eviction approaches | Not automatically compatible with Ollama's managed KV cache | Do not port algorithms into the Ollama API path without a supported runtime hook |
| llm-compressor / NVIDIA Model Optimizer | Model compression/quantization and deployment optimization | Primarily model/deployment preparation, not per-request Ollama KV control | Separate model deployment experiment |
| LMCache / FlexKV / kvcached | External KV storage, offload, tiering, memory virtualization | Requires integration with a supported serving engine | Future alternate-engine work |
| DeepSpec / DFlash / Speculators | Speculative decoding | Requires supported drafter/target runtime integration | Not a KV-cache feature; defer |
| Flash Attention | Memory-efficient attention kernels | Ollama runtime supports it when the backend/device supports it | Use Ollama server configuration and verify runtime behavior |
| Ollama KV cache quantization | F16, Q8_0, Q4_0 KV storage | Native Ollama server configuration; global to models served by that Ollama process | Use Q8_0 as the first measured candidate; retain F16 baseline and quality checks |
| Ollama generation options | Per-request generation options including context/output limits | Native `/api/generate` path | Add a bounded, configurable context/output budget policy |
| Prefix reuse | Reuse identical token prefixes to avoid repeated prefill | Ollama API does not provide explicit cache-hit telemetry/control in the inspected contract | Preserve stable system/prompt prefixes; label reuse unverified unless runtime metrics prove it |
| Runtime-aware concurrency | Bound concurrent inference according to available resources | Can be enforced in Argus before requests reach Ollama | Implement configurable admission control; avoid claiming GPU memory awareness without a real measurement source |

## Implementation plan

### Phase A — Observable Ollama runtime

1. Preserve Ollama's native response counters and timings: prompt-evaluation count/duration, generated-token count/duration, model load duration, and total duration.
2. Add explicit runtime telemetry to the API response, keeping heuristic prompt-size estimates clearly separate from Ollama-reported counts.
3. Query Ollama's process/model status where available and report unknown values as unknown rather than infer them.
4. Add an endpoint describing Argus's requested KV configuration separately from values that Ollama can actually verify.

### Phase B — Safe local KV configuration

1. Document and expose operator-configurable Ollama server settings for Flash Attention and KV cache type.
2. Start with Q8_0 and compare against F16; do not automatically select Q4_0 without task-quality evidence.
3. Clearly state that Ollama's KV cache type is server-global and requires restarting/reloading the Ollama server to change.
4. Do not imply FastAPI can change an already-running Ollama process's environment.

### Phase C — Adaptive request budgets

1. Derive a safe context budget from the actual prompt size, configured model/context limits, and reserved output budget.
2. Pass supported context and output limits in Ollama generation options.
3. Record the selected limits and actual Ollama prompt/output token counts.
4. Reject or safely handle prompts that cannot fit the configured context budget; never silently truncate user instructions.

### Phase D — Memory-safe concurrency

1. Add configurable inference concurrency limits and queue/wait telemetry.
2. Add cancellation and timeout behavior that releases admission slots reliably.
3. Only use GPU-memory-aware adaptation when a supported, tested memory telemetry provider is available; otherwise use a fixed configurable concurrency ceiling and report the limitation.

### Phase E — Benchmark and quality verification

1. Run repeatable F16 versus Q8_0 comparisons on the same model, prompt set, context, and generation settings.
2. Capture TTFT/prompt-evaluation time, decode throughput, total latency, token counts, process memory where measurable, and failure rates.
3. Compare output quality and Argus semantic-preservation validation outcomes.
4. Report measured results with hardware/runtime/model versions and sample counts. No hardcoded or illustrative performance claims.

## Explicit non-goals for the Ollama MVP

- Do not install vLLM, SGLang, Dynamo, Mooncake, LMCache, or Kubernetes infrastructure merely to claim their features.
- Do not claim Flash Attention, quantization, prefix-cache hits, offload, or GPU-memory-aware scheduling unless the runtime is configured and the behavior is verified.
- Do not conflate semantic response caching with transformer KV caching.
- Do not use simulated frontend metrics as benchmark evidence.
