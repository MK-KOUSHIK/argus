# Argus 🛡️

> **Intelligent Prompt Optimization Middleware**  
> Locally optimize, structure, and compress prompts using a compact neural model (**Qwen2.5-1.5B-Instruct** running via **Ollama**) while strictly preserving user intent, operational requirements, and explicit constraints.

[![CI Pipeline](https://github.com/MK-KOUSHIK/argus/actions/workflows/ci.yml/badge.svg)](https://github.com/MK-KOUSHIK/argus/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React: 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Ollama: qwen2.5:1.5b](https://img.shields.io/badge/Ollama-qwen2.5%3A1.5b-purple.svg)](https://ollama.com/)
[![Tailwind CSS: v4](https://img.shields.io/badge/Tailwind-v4-06B6D4.svg)](https://tailwindcss.com/)

---

## 📖 Table of Contents
- [Project Purpose & Scope](#project-purpose--scope)
- [Architecture & Boundaries](#architecture--boundaries)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Ollama Setup & Model Verification](#ollama-setup--model-verification)
- [Installation & Local Run Instructions](#installation--local-run-instructions)
- [API Usage](#api-usage)
- [Task-Aware Optimization Pipeline](#-task-aware-optimization-pipeline)
- [Semantic Preservation & Zero-Hardcoding](#-semantic-preservation--zero-hardcoding)
- [Token Estimation & Honest Metrics Standard](#-token-estimation--honest-metrics-standard)
- [Testing & Verification](#testing--verification)
- [Adaptive KV-Cache & Output Budget Runtime Optimization](#-adaptive-kv-cache--output-budget-runtime-optimization)
- [Roadmap & Implementation Phases](#roadmap--implementation-phases)
- [Contributing & Development Guide](#contributing--development-guide)
- [License](#license)

---

## Project Purpose & Scope

Large Language Models (LLMs) incur significant processing latency and inference costs scaled directly by prompt volume. Complex instructions, conversational pleasantries, circular repetitions, and unstructured requests degrade prompt execution quality and waste resources.

**Argus** functions as an intelligent, local prompt optimization middleware:
1. **Analyzes Intent & Constraints**: Performs deterministic task inspection to extract domain categories, primary action verbs, deliverables, required functional operations, and explicit constraints (e.g., negative prohibitions, filenames, requested keys).
2. **Selects Optimization Strategy**: Chooses an appropriate strategy (`compress`, `clarify`, `structure`, `enrich`, or `preserve`) based on prompt characteristics and the user's selected profile.
3. **Controls Output Generation Budget**: Dynamically calculates generation ceilings (`num_predict`) based on explicit prompt directives (word/sentence limits) and task requirements, saving 35%+ unneeded headroom while coordinating with context tier snapping.
4. **Optimizes Locally via Neural Model**: Rewrites the prompt using a local lightweight LLM (**Qwen2.5-1.5B-Instruct** running via **Ollama**) without external API calls or token costs.
5. **Validates Semantic Fidelity**: Employs deterministic validation guardrails to ensure deliverables are maintained, interactive turn-reversals are prevented, and all required operations and prohibitions remain intact.
6. **Provides Honest Telemetry**: Displays side-by-side prompt diffs, character deltas, heuristic token estimates, output budget utilization, and native Ollama completion reasons (`done_reason: stop` vs `length`) in a modern React workspace.

---

## Architecture & Boundaries

```
[ Frontend: React 19 + Vite + Tailwind CSS v4 ]
                    │
                    ▼  (REST API contracts under /api)
[ Backend API: FastAPI Router ]
                    │  (Schema validation via Pydantic)
                    ▼
[ Orchestration: OllamaService ]
        ├──▶ [ Stage 0: CompressionOrchestrator (Content-Aware Input Compression & Reversible Registry) ]
        ├──▶ [ Stage A: TaskAnalyzer (Domain, Deliverables, Constraints) ]
        ├──▶ [ Stage B: StrategySelector & OutputBudgetController ]
        ├──▶ [ AdaptiveContextManager (Dynamic Tier Snapping with Output Headroom) ]
        ├──▶ [ Stage C: PromptRewriter (System Instructions & Guardrails) ]
        ├──▶ [ Local Inference Tier: Ollama Daemon (qwen2.5:1.5b via HTTP 11434) ]
        └──▶ [ Stage D: CandidateValidator (Constraint & Intent Guardrails) ]
```

- **Frontend Isolation**: Frontend components serve strictly as presentation, editing, and status monitoring interfaces. They contain **no** prompt compression heuristics or local LLM execution logic.
- **Backend Service Layer**: Handles prompt analysis, strategy determination, output budget control, prompt instruction engineering, Ollama HTTP communication, and validation retries/fallbacks.
- **Local Network Privacy**: The optimization pipeline executes 100% locally on your machine with zero external cloud dependencies or third-party credential exposure.

For detailed architecture diagrams and sequence flows, see [`docs/architecture.md`](docs/architecture.md).

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Vitest, Testing Library |
| **Backend** | Python 3.10–3.13, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, HTTPX, Pytest |
| **Local Inference** | Ollama with `qwen2.5:1.5b` |
| **Version Control & CI** | Git, GitHub, GitHub Actions |
| **Target Deployment** | Docker, Azure Container Apps / AWS ECS |

---

## Directory Structure

```
Argus/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                     # Automated CI for frontend & backend
│   └── pull_request_template.md       # PR guidelines and verification checklist
├── backend/
│   ├── app/
│   │   ├── api/                       # API routers and route handlers
│   │   │   ├── health.py              # Health check endpoint (/api/health)
│   │   │   ├── optimizer.py           # Optimization & status routes (/api/optimize, /api/ollama/status)
│   │   │   └── router.py              # Centralized router mount
│   │   ├── core/                      # Centralized configuration
│   │   │   └── config.py              # Pydantic BaseSettings driven by .env
│   │   ├── schemas/                   # Pydantic contract definitions
│   │   │   ├── health.py              # Health response schema
│   │   │   └── optimizer.py           # OptimizeRequest, OptimizeResponse, OutputBudgetTelemetry
│   │   ├── services/                  # Business logic & local inference client
│   │   │   ├── adaptive_context_manager.py # Dynamic tier derivation & context budgeting
│   │   │   ├── candidate_validator.py # Semantic guardrail & hallucination validation
│   │   │   ├── concurrency_controller.py   # Runtime-aware admission & capacity enforcement
│   │   │   ├── hardware_telemetry.py  # Real-time GPU VRAM & host RAM telemetry
│   │   │   ├── ollama_process_manager.py   # Managed process inspection & log verification
│   │   │   ├── ollama_service.py      # Core Ollama client & optimization pipeline
│   │   │   ├── output_budget_controller.py # Dynamic generation budget controller
│   │   │   ├── prompt_rewriter.py     # System prompt engineering
│   │   │   ├── strategy_selector.py   # Strategy selection heuristic
│   │   │   └── task_analyzer.py       # Deterministic intent & constraint extraction
│   │   └── main.py                    # FastAPI application factory
│   ├── data/                          # Runtime process logs and metadata manifests
│   └── tests/                         # Pytest automated test suite (118 tests)
├── frontend/
│   ├── src/
│   │   ├── components/                # Modular React presentation components
│   │   ├── pages/                     # Routed pages (WorkspacePage, StatusPage)
│   │   ├── services/                  # Typed API network client
│   │   └── types/                     # TypeScript API & domain contracts
│   └── tests/                         # Vitest automated test suite (19 tests)
├── scripts/
│   ├── benchmark_kv_cache.py          # F16 vs Q8_0 KV cache empirical evaluation
│   ├── benchmark_output_budget.py     # Output budget empirical validation
│   ├── experiment_prefix_reuse.py     # Llama.cpp prefix cache verification
│   └── launch_ollama.ps1              # Managed Ollama Windows process controller
└── docs/
    ├── architecture.md                # System architecture and request flow diagrams
    ├── development-guide.md           # Local setup and developer documentation
    ├── kv-cache-optimization-audit.md # KV cache audit and acceptance matrix
    └── reports/                       # Empirical benchmark reports and JSON data
```

---

## Prerequisites

- **Node.js**: `v20.x` or `v22.x LTS` (includes `npm`)
- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: `v2.40+`
- **Ollama**: Installed locally on your workstation

---

## Ollama Setup & Model Verification

1. **Verify Ollama is installed**:
   ```bash
   ollama --version
   ```

2. **Launch Ollama with Q8_0 KV Cache and Flash Attention (Windows PowerShell)**:
   ```powershell
   .\scripts\launch_ollama.ps1 -Action start -KvType q8_0 -FlashAttention
   ```

3. **Pull the configured model**:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

---

## Installation & Local Run Instructions

### 1. Clone Repository
```bash
git clone https://github.com/MK-KOUSHIK/argus.git
cd argus
```

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Backend is now running at **`http://127.0.0.1:8000`**.

### 3. Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```
Frontend workspace is live at **`http://localhost:5173`**.

---

## API Usage

### 1. Optimize Prompt (`POST /api/optimize`)
Submits a raw prompt for task-aware local optimization:

```json
{
  "prompt": "Summarize the key differences between synchronous and asynchronous programming in Python in under 50 words.",
  "mode": "balanced",
  "target_model": "gpt-4o",
  "requested_output_budget": null
}
```

**Response (`200 OK`)**:
```json
{
  "original_prompt": "Summarize the key differences between synchronous and asynchronous programming in Python in under 50 words.",
  "optimized_prompt": "Summarize the key differences between synchronous and asynchronous programming in Python in under 50 words.",
  "mode": "balanced",
  "target_model": "gpt-4o",
  "task_category": "coding",
  "strategy": "preserve",
  "applied_strategy": "preserve",
  "outcome": "unchanged_original",
  "length_result": "unchanged",
  "validation_status": "passed",
  "is_preserved": true,
  "original_metrics": { "char_count": 108, "estimated_tokens": 27 },
  "optimized_metrics": { "char_count": 108, "estimated_tokens": 27 },
  "token_reduction_percent": 0,
  "token_change_percent": 0,
  "token_savings": 0,
  "char_delta": 0,
  "preserved_constraints": ["Core query intent & technical details"],
  "execution_time_ms": 186,
  "model": "qwen2.5:1.5b",
  "timestamp": "2026-11-04T15:21:38Z",
  "output_budget": {
    "requested_budget": null,
    "allocated_budget": 131,
    "policy": "adaptive",
    "reason": "Adaptive constraint: Explicit word limit 'under 50 words' -> ~131 tokens",
    "task_category": "coding",
    "actual_generated_tokens": 21,
    "generation_duration_ms": 151.52,
    "generation_tokens_per_second": 138.59,
    "done_reason": "stop",
    "hit_budget_limit": false,
    "budget_utilization_percent": 16.0
  }
}
```

### 2. Ollama Status (`GET /api/ollama/status`)
Checks whether the local Ollama instance is running and verifies model availability:
```json
{
  "connected": true,
  "base_url": "http://127.0.0.1:11434",
  "model": "qwen2.5:1.5b",
  "model_installed": true,
  "available_models": ["qwen2.5:1.5b", "qwen2.5:7b"],
  "runtime_process": {
    "server_process_verified": true,
    "configured_kv_cache_type": "q8_0",
    "configured_flash_attention": true
  }
}
```

### 3. End-to-End Downstream Output Comparison (`POST /api/compare`)
Executes two genuine downstream LLM inferences (Execution A on original prompt, Execution B on optimized prompt) pinned to the same model, with model warm-up, alternating paired trials, granular latency separation, and native Ollama telemetry:

```json
{
  "prompt": "What is the speed of light in vacuum in meters per second? Give only the exact numerical value and unit in one sentence.",
  "downstream_model": "qwen2.5:1.5b",
  "mode": "concise",
  "temperature": 0.3,
  "max_tokens": 300,
  "trials": 3,
  "warmup": true
}
```

**Response (`200 OK`)**:
```json
{
  "original_execution": {
    "prompt": "What is the speed of light in vacuum in meters per second? Give only the exact numerical value and unit in one sentence.",
    "response": "The speed of light in vacuum is exactly 299,792,458 meters per second.",
    "model": "qwen2.5:1.5b",
    "provider": "ollama",
    "prompt_tokens": 54,
    "output_tokens": 25,
    "total_tokens": 79,
    "duration_ms": 286.5,
    "prompt_eval_duration_ms": 45.2,
    "done_reason": "stop",
    "latency_breakdown": {
      "model_load_duration_ms": 0.0,
      "prompt_eval_duration_ms": 45.2,
      "eval_duration_ms": 241.3,
      "inference_duration_ms": 286.5,
      "queue_wait_duration_ms": 1.2,
      "wall_time_ms": 287.7
    }
  },
  "optimized_execution": {
    "prompt": "What is the speed of light in vacuum in meters per second?",
    "response": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "model": "qwen2.5:1.5b",
    "provider": "ollama",
    "prompt_tokens": 42,
    "output_tokens": 26,
    "total_tokens": 68,
    "duration_ms": 222.13,
    "prompt_eval_duration_ms": 12.66,
    "done_reason": "stop",
    "latency_breakdown": {
      "model_load_duration_ms": 0.0,
      "prompt_eval_duration_ms": 12.66,
      "eval_duration_ms": 209.47,
      "inference_duration_ms": 222.13,
      "queue_wait_duration_ms": 0.8,
      "wall_time_ms": 222.93
    }
  },
  "metrics": {
    "prompt_tokens_saved": 12,
    "prompt_token_reduction_percent": 22.22,
    "output_tokens_diff": 1,
    "output_token_change_percent": 4.0,
    "total_tokens_saved": 11,
    "total_token_reduction_percent": 13.92,
    "latency_delta_ms": -64.37,
    "latency_reduction_percent": 22.47,
    "latency_note": "Averaged across 3 paired trials with alternating execution order ([A, B], [B, A]).",
    "generation_parameters_identical": true,
    "semantic_quality": "not_evaluated",
    "semantic_quality_reason": "No automated judge model is configured. Semantic equivalence and response quality must be verified by human inspection."
  },
  "downstream_model": "qwen2.5:1.5b",
  "effective_settings": {
    "num_ctx": 2048,
    "temperature": 0.3,
    "num_predict": 300,
    "top_p": 0.9,
    "system_instruction": null
  },
  "trials": 3,
  "warmup": true,
  "timestamp": "2026-11-06T03:46:41Z"
}
```

---

## 🧠 Task-Aware Optimization Pipeline

Argus implements a modular 4-stage optimization pipeline:

```
[ Raw Prompt ] 
      │
      ▼
[ Stage A: TaskAnalyzer ] ──────▶ Detects domain, action verbs, deliverables, operations, prohibitions
      │
      ▼
[ Stage B: StrategySelector & OutputBudgetController ] ───▶ Selects strategy & calculates generation budget
      │
      ▼
[ Stage C: PromptRewriter ] ────▶ Builds targeted instructions for local Qwen2.5:1.5b model
      │
      ▼
[ Stage D: CandidateValidator ] ─▶ Enforces deliverable preservation & constraint retention (with retry)
      │
      ▼
[ Final Validated Output ] ─────▶ Honest telemetry, character delta, and token estimations
```

---

## 🧪 Testing & Verification

### 1. Backend Tests (Pytest)
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```
- **234 passing tests** covering health check, optimizer route, task analysis, strategy selection, candidate validation, semantic preservation, Ollama error handling, dynamic context tier derivation, 413 overflow safeguards, hardware telemetry, independent process verification, dynamic concurrency capacity enforcement, Output Budget Control, Stage 0 Content-Aware Compression Engine, Local Model Routing (Phase 5), and **End-to-End Downstream Output Comparison**.

### 2. Frontend Tests (Vitest)
```bash
cd frontend
npm.cmd test
```
- **26 passing tests** across `App.test.tsx`, `Workspace.test.tsx`, `OutputComparison.test.tsx`, `useHealth.test.tsx`, and `useOllamaStatus.test.tsx` validating layout, tab navigation, user input, mode switching, diff rendering, downstream output comparison view, Ollama status, telemetry display, and API error handling.

### 3. Frontend Production Build
```bash
cd frontend
npm.cmd run build
```
- Executes TypeScript validation (`tsc -b`) and bundles production assets via Vite with zero type errors.

### 4. Live Downstream Output Comparison Validation
```bash
cd D:\Projects\Argus
.\backend\.venv\Scripts\python.exe scripts\validate_output_comparison_live.py
```
- Executes 3 real scenarios (short factual, redundant context, multi-constraint task) against the live Ollama daemon, pinning both Execution A (original prompt) and Execution B (optimized prompt) to the same local model (`qwen2.5:1.5b`), capturing real native telemetry without fabrication, and generating [`docs/reports/output_comparison_validation.md`](docs/reports/output_comparison_validation.md).

---

## ⚡ Adaptive KV-Cache & Output Budget Runtime Optimization

Argus includes a production-grade KV-cache runtime optimization and telemetry engine engineered for local Ollama execution on consumer GPUs (e.g. NVIDIA RTX 4050 6 GB VRAM):

1. **Q8_0 KV Cache Quantization & Flash Attention**:
   - Reduces KV buffer size by **46.9%** (from 112.00 MiB down to 59.52 MiB for 4,096 context on `qwen2.5:1.5b`), saving ~50 MiB of VRAM with sub-3% decode throughput impact.
   - Activates Flash Attention kernels (`llama_context: flash_attn = enabled`) to eliminate quadratic memory expansion during attention computation.
   - Managed via the reproducible PowerShell launcher: [`scripts/launch_ollama.ps1`](scripts/launch_ollama.ps1).

2. **Output Budget Control (Phase 4.1)**:
   - Replaces fixed 512-token output generation with intelligent, task-aware generation ceilings (`num_predict`).
   - Supports three distinct validated runtime policies (`adaptive`, `fixed`, `conservative`):
     - `adaptive`: Dynamically calculates ceilings from validated task category baselines, mode scaling (`concise`: 0.75x, `balanced`: 1.0x, `structure`: 1.2x), complexity scores (+20%), and explicit prompt constraints (`under 50 words`, `in one sentence`, `tl;dr`).
     - `fixed`: Uses `OUTPUT_BUDGET_DEFAULT_TOKENS` without dynamic variation.
     - `conservative`: Minimizes memory footprint via 0.75x dampened baselines, 0 constraint headroom, and capped limits.
   - Coordinates with `AdaptiveContextManager` via centralized headroom calculations, protecting against silent input truncation and enforcing HTTP 413 context boundaries.
   - Strictly separates allocated budget ceiling from actual tokens generated, reporting native Ollama completion causes (`done_reason: stop` vs `length`).
   - Conserves **35.0% aggregate generation ceiling headroom** across test suites with **100% natural completions (`done_reason == "stop"`)**.

3. **Adaptive Context-Length Management & Dynamic Tier Derivation**:
   - Enforces token budgeting: $\text{Prompt} + \text{System} + \text{Wrapper} + \text{Predict (Dynamic)} + \text{Headroom (256)}$.
   - **Dynamic Tier Derivation (4096, 6144, 8192)**: Tiers are dynamically derived and strictly bounded by `OLLAMA_MAX_NUM_CTX`. Prevents continuous KV cache reallocations while respecting configured maximum context limits.
   - **Low-VRAM & Overflow Guards**: Under low VRAM (<1,000 MiB), restricts allocation to safe baseline tier (4096) and returns **HTTP 413 Payload Too Large** if budget exceeds that capacity.

4. **Genuine Dynamic Concurrency Capacity Enforcement**:
   - Condition-based admission gate (`asyncio.Condition`) that evaluates hardware telemetry dynamically at admission time.
   - Automatically throttles admission capacity from 3 down to 1 under low GPU VRAM (<400 MiB) or host RAM (<1024 MiB), draining active requests safely without permit race conditions.
   - Rejects requests immediately with HTTP 503 when VRAM <150 MiB or RAM <512 MiB.

5. **Authoritative Native Telemetry**:
   - Directly exposes Ollama's native inference metrics: `prompt_eval_count`, `prompt_eval_cached_count`, `prompt_eval_duration_ns`, `eval_count`, `eval_duration_ns`, `generation_tokens_per_second`, and `done_reason`.
   - Distinct source attribution transparently separates exact hardware metrics from heuristic estimates with strict zero-fabrication.

6. **Independent Content-Aware Compression Engine (Phase 4.2)**:
   - **Content-Aware Routing**: Inspects prompts to detect and isolate JSON, Python/SQL code, repetitive system logs, natural language text, or mixed markdown documents.
   - **Structural Minification & AST Parsing**: Strips formatting from JSON (verified by `json.loads` deep equality) and comments from Python code (verified by AST node parity), preserving functional docstrings and type annotations.
   - **Repetitive Event Folding**: Detects and folds repetitive log event runs ($\ge 3$), retaining initial/terminal timestamps while guaranteeing 100% preservation of critical errors and stack traces.
   - **Reversible Registry**: Stores omitted original content in a thread-safe in-memory LRU/TTL store (`argus:ref:...`), retrievable via `GET /api/compression/registry/{ref_key}`.
   - **Empirical Token Savings**: Yielded a **15.26% reduction in native prompt evaluation tokens** (-31.59% character footprint) with sub-1.5ms processing latency (mean 1.27 ms), 100% byte-exact reversible restoration, and verified downstream task evaluation.

7. **Local-First Model Routing Engine (Phase 5)**:
   - **Dynamic Discovery & Inspection**: Introspects local Ollama instance (`/api/tags`, `/api/ps`, `/api/show`) with TTL caching to catalogue installed models, parameter scales, quantization formats, and active resident memory footprints.
   - **Zero-Guessing Classification & Neutral Ranking**: Model tiers are determined strictly from verified parameter metadata (`parameter_count` or `/api/show` `details.parameter_size`). Candidate selection within tiers uses an objective, multi-criteria ranking policy (resident models prioritized, tier-specific parameter sorting, quantization bit-depth, deterministic alphabetical tie-breaking) with zero vendor or family substring preferences.
   - **Explainable Heuristic Complexity Scoring**: Multi-signal scoring $\in [0.0, 1.0]$ based on domain category weights, explicit constraints, required functional operations, negative prohibitions, code/data syntax, and token footprint.
   - **Uncalibrated Decision Margin**: Reports `heuristic_margin = |score - threshold|`, explicitly documented as an uncalibrated heuristic distance rather than a statistical confidence score.
   - **Precise Memory Footprint Telemetry & Transparent Limitations**: Distinguishes between measured resident VRAM (`measured_resident` via Ollama `/api/ps`), architecture-derived proxies (`architecture_proxy`), clearly identified rough proxies (`rough_proxy`), CPU-only execution (`cpu_only`), and unverified missing telemetry (`unverified`). When RAM telemetry is unavailable, requests are conservatively rejected to preserve host stability. Support for CPU-only deployments is fully preserved when host RAM is verified sufficient.
   - **Resource-Aware Feasibility Gate**: Inspects live NVIDIA GPU VRAM (`nvidia-smi`), accounts for resident runtime headroom (`ROUTING_LOADED_HEADROOM_MARGIN_MB`), host RAM margins, and triggers verified fallback under memory pressure.
   - **Manual Override Memory Safety**: Manual model overrides enforce strict resource checks. Overrides exceeding available resources are rejected with a structured `ModelResourceConstraintError` (HTTP 400) unless `allow_override_fallback=True` is provided.
   - **Verified Safe Fallbacks**: Fallbacks must be installed and verified safe; configured defaults are never assumed installed (`NoSafeFallbackError` raised if no safe model exists).
   - **Optimizer Model Routing**: Selects lightweight models for simple tasks and larger models for complex tasks, with manual override support in the React workspace.
   - **Downstream Execution Routing**: Explicitly deferred to future phases to keep prompt optimization decoupled from downstream execution.
   - **Empirical Validation Standards**: Benchmark suite separates simulated policy margins (Scenario 3A) from natural hardware observations (Scenario 3B), accurately reporting physical headroom without fabricating fallback events.

See the [Compression Validation Report](docs/reports/compression-validation-report.md), [Output Budget Validation Report](docs/reports/output-budget-validation-report.md), and [KV Cache Optimization Audit](docs/kv-cache-optimization-audit.md) for complete empirical data.

---

## Roadmap & Implementation Phases

| Phase | Milestone | Scope & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Repository Foundation & Project Organization** | Scaffolding, design tokens, FastAPI foundation, health endpoint, React 19 + Tailwind CSS v4, Vitest, CI workflow. | ✅ Complete |
| **Phase 2** | **UI/UX Workspace & System Status** | Responsive workspace, prompt editor, profile selection, side-by-side comparison view, metrics panel, System Status page. | ✅ Complete |
| **Phase 3** | **Real Local LLM Integration** | Ollama HTTP integration (`http://127.0.0.1:11434`), `qwen2.5:1.5b` model communication, error handling, status monitoring. | ✅ Complete |
| **Phase 3.1** | **Task-Aware Optimization Engine** | Modular pipeline (Stages A–D), task domain classification, strategy selection (`compress`, `clarify`, `structure`, `enrich`, `preserve`), explicit constraint & prohibition guardrails, candidate validation. | ✅ Complete |
| **Phase 3.2** | **Semantic Preservation & Zero-Hardcoding** | Action/deliverable intent extraction, anti-turn-reversal validation, corrective retry/fallback, generalized taxonomies, centralized configuration. | ✅ Complete |
| **Phase 4** | **Adaptive KV-Cache Runtime Optimization** | Q8_0 KV quantization, Flash Attention, dynamic context tier snapping, memory-aware concurrency controller, prefix cache reuse, managed launcher. | ✅ Complete |
| **Phase 4.1** | **Output Budget Control & Native Telemetry** | Dynamic task & constraint output budgeting (`num_predict`), explicit word/sentence limit parsing, context headroom coordination, native `done_reason` telemetry. | ✅ Complete |
| **Phase 4.2** | **Independent Content-Aware Compression Engine** | Content-aware routing, JSON structural minifier, Python AST comment stripper, repetitive log folding, text redundancy dedup, prefix cache preserver, reversible content registry with TTL & LRU bounds. | ✅ Complete |
| **Phase 5** | **Local-First Model Routing Engine** | Dynamic Ollama model discovery, explainable heuristic complexity scoring, live hardware VRAM/RAM feasibility evaluation, optimizer model routing (1.5B vs 7B), resource fallback, and UI controls. (Downstream execution routing explicitly deferred). | ✅ Complete |
| **Phase 6** | **Frontier LLM Gateway & Provider Dispatch** | Configurable external provider integration (Google Gemini / OpenAI), downstream execution, and frontier token metrics. | ⏳ Planned |

---

## Contributing & Development Guide

For local setup instructions, coding conventions, testing guidelines, and troubleshooting, see [`docs/development-guide.md`](docs/development-guide.md).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
