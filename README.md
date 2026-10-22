# Argus 🛡️

> **Intelligent Prompt Optimization Middleware**  
> Dramatically reduce main-LLM input-token costs with local model prompt compression while preserving intent and semantic constraints.

[![CI Pipeline](https://github.com/argus-ai/argus/actions/workflows/ci.yml/badge.svg)](https://github.com/argus-ai/argus/actions)
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
- [Testing & Verification](#testing--verification)
- [Planned Implementation Phases](#planned-implementation-phases)
- [Contributing & Development Guide](#contributing--development-guide)
- [License](#license)

---

## Project Purpose & Scope

Large Language Models (LLMs) incur significant costs scaled directly by input-token volume. Complex system prompts, verbose instructions, repeated examples, and boilerplate constraints quickly inflate API bills on premium frontier models.

**Argus** functions as an intelligent middleware gateway:
1. **Intercepts** client prompts before transmission to expensive frontier models.
2. **Extracts & Locks** vital semantic constraints, task parameters, and output formats.
3. **Compresses & Optimizes** prompt wording locally using a lightweight local LLM (**Qwen2.5-1.5B-Instruct** running via **Ollama**).
4. **Verifies Constraint Integrity** to ensure intent preservation.
5. **Dispatches** the optimized prompt to the target main LLM (OpenAI, Anthropic, Gemini, etc.), returning both the answer and transparent token/cost savings telemetry.

---

## Architecture & Boundaries

Argus enforces strict architectural boundaries:

```
[ Frontend: React + Vite + Tailwind ]
            │  (REST API calls via typed contracts)
            ▼
[ Backend API: FastAPI Router ]
            │  (Schema validation via Pydantic)
            ▼
[ Services Layer: Prompt Compression & Orchestration ]
       ├──▶ [ Local Inference Tier: Ollama (Qwen2.5-1.5B via HTTP 11434) ]
       └──▶ [ Frontier LLM Tier: OpenAI / Anthropic / Gemini (Phase 4 Gateway) ]
```

- **Frontend Isolation**: Frontend components serve purely as presentation and configuration interfaces. They contain **no** backend business logic, prompt tokenization algorithms, or direct provider API keys.
- **Backend Service Layer**: Handles orchestration, compression prompting, Ollama integration, and provider dispatch.
- **Security**: Provider API keys remain strictly on the backend. Model weights and environments are excluded from source control.

For in-depth architecture diagrams and specifications, see [`docs/architecture.md`](docs/architecture.md).

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Vitest |
| **Backend** | Python 3.10–3.13, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, HTTPX, Pytest |
| **Local Inference** | Ollama with `qwen2.5:1.5b` (Phase 3 Active) |
| **Version Control & CI** | Git, GitHub, GitHub Actions |
| **Target Deployment** | Docker, Azure Container Apps / AWS ECS |

---

## Directory Structure

```
Argus/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated CI for frontend & backend
│   └── pull_request_template.md # PR guidelines and verification checklist
├── backend/
│   ├── app/
│   │   ├── api/                 # API routers and endpoint handlers
│   │   │   ├── health.py        # Health check route
│   │   │   ├── optimizer.py     # Prompt optimization & Ollama status routes
│   │   │   └── router.py        # Centralized router mount
│   │   ├── core/                # Centralized settings & configuration
│   │   │   └── config.py        # Pydantic BaseSettings (with Ollama config)
│   │   ├── schemas/             # Pydantic validation & response models
│   │   │   ├── health.py        # Health response schema
│   │   │   └── optimizer.py     # Request/response models for prompt optimization
│   │   ├── services/            # Business & orchestration logic
│   │   │   ├── health_service.py# Health check service implementation
│   │   │   └── ollama_service.py# Real Ollama async client & prompt optimization
│   │   └── main.py              # FastAPI app factory, CORS & error handlers
│   ├── tests/                   # Backend unit and integration tests
│   │   ├── conftest.py          # Pytest async client fixtures
│   │   ├── test_health.py       # Health endpoint test suite
│   │   └── test_optimizer.py    # Optimizer & Ollama error handling test suite
│   ├── .env.example             # Backend environment template
│   ├── pyproject.toml           # Project packaging & dependencies
│   ├── requirements.txt         # Core dependencies
│   └── requirements-dev.txt     # Development and testing dependencies
├── frontend/
│   ├── public/                  # Static assets & brand icons
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── Button.tsx       # Accessible button component
│   │   │   ├── Card.tsx         # Dark navy container panel
│   │   │   ├── ComparisonView.tsx# Side-by-side prompt diff & edit view
│   │   │   ├── Header.tsx       # Brand header with live Ollama connection status
│   │   │   ├── MetricsPanel.tsx # Token savings metrics & footprint bar
│   │   │   ├── OptimizationControls.tsx # Mode selector & live engine indicator
│   │   │   ├── PromptEditor.tsx # Raw prompt editor with token counters
│   │   │   └── StatusBadge.tsx  # Status indicators
│   │   ├── hooks/               # React custom hooks (useHealth, useOllamaStatus)
│   │   ├── pages/               # Page views (WorkspacePage, HomePage)
│   │   ├── services/            # API client service layer (api.ts)
│   │   ├── styles/              # Design tokens and Tailwind CSS v4 entrypoints
│   │   ├── types/               # TypeScript interfaces & domain models
│   │   ├── App.tsx              # Root application layout with tab navigation
│   │   └── main.tsx             # Application DOM mounting
│   ├── tests/                   # Frontend unit & integration tests
│   │   ├── setup.ts             # Vitest test setup
│   │   ├── App.test.tsx         # App navigation smoke tests
│   │   └── Workspace.test.tsx   # Workspace interactions and mock API tests
│   ├── .env.example             # Frontend environment template
│   ├── package.json             # NPM scripts and dependencies
│   ├── tsconfig.json            # TypeScript compiler configuration
│   └── vite.config.ts           # Vite + Tailwind v4 bundler config
├── docs/
│   ├── architecture.md          # Architectural blueprints and boundaries
│   └── development-guide.md     # Setup, testing, and contribution protocols
├── scripts/
│   ├── dev-backend.ps1 / .sh    # Convenience script: Start backend server
│   └── dev-frontend.ps1 / .sh   # Convenience script: Start frontend server
├── .gitignore                   # Comprehensive git exclusions
├── .env.example                 # Root environment template
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## Prerequisites

- **Node.js**: `v20.x` or `v22.x LTS`
- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: `2.40+`
- **Ollama**: Installed locally on `http://127.0.0.1:11434`

---

## Ollama Setup & Model Verification

1. **Verify Ollama is running**:
   ```bash
   ollama --version
   ```
2. **Pull the Qwen2.5 1.5B Instruct model**:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
3. **Verify model presence**:
   ```bash
   ollama list
   ```

---

## Installation & Local Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/argus-ai/argus.git
cd argus
```

### 2. Backend Setup & Startup
```powershell
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Create environment file
cp .env.example .env

# Run FastAPI dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now running at **`http://localhost:8000`**.  
Interactive API docs: **`http://localhost:8000/api/docs`**  
Ollama status: **`http://localhost:8000/api/ollama/status`**

### 3. Frontend Setup & Startup
```powershell
# In a separate terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start Vite dev server
npm run dev
```
The frontend is now accessible at **`http://localhost:5173`**.

---

## API Usage

### Optimize Prompt (Task-Aware Engine - Phase 3.1)
`POST /api/optimize`

```json
{
  "prompt": "Write a Python script that reads sales.csv, calculates monthly revenue, saves monthly_report.csv, and does not use pandas.",
  "mode": "balanced",
  "target_model": "gpt-4o"
}
```

**Response (`200 OK`)**:
```json
{
  "original_prompt": "Write a Python script that reads sales.csv, calculates monthly revenue, saves monthly_report.csv, and does not use pandas.",
  "optimized_prompt": "- Task: Write a Python script\n- Inputs/Context: Read sales.csv, calculate monthly revenue, save to monthly_report.csv\n- Constraints: Do not use pandas\n- Deliverable: Completed Python script",
  "mode": "balanced",
  "target_model": "gpt-4o",
  "task_category": "data_analysis",
  "strategy": "structure",
  "validation_status": "passed",
  "is_preserved": false,
  "explicit_constraints": [
    "Prohibition: No pandas",
    "File reference: sales.csv",
    "File reference: monthly_report.csv"
  ],
  "original_metrics": { "char_count": 122, "estimated_tokens": 31 },
  "optimized_metrics": { "char_count": 189, "estimated_tokens": 48 },
  "token_reduction_percent": 0,
  "token_change_percent": 55,
  "token_savings": -17,
  "char_delta": -67,
  "preserved_constraints": ["Core query intent & technical details"],
  "execution_time_ms": 2107,
  "model": "qwen2.5:1.5b",
  "timestamp": "2026-10-16T18:32:18Z"
}
```

---

## 🧠 Task-Aware Optimization Engine (Phase 3.1)

Argus transitions from purely superficial token-reduction into a **Task-Aware Optimization Engine**. The objective is not to compress at all costs, but to optimize for clarity, task effectiveness, preservation of intent, and reasonable token efficiency.

### 4-Stage Modular Pipeline

1. **Stage A — Understand the Prompt (`TaskAnalyzer`)**:
   - Detects the task domain (`coding`, `data_analysis`, `writing`, `extraction`, `summarization`, `research`, `planning`, or `general`).
   - Identifies explicit constraints: prohibitions (`do not use pandas`, `no numpy`), filenames (`sales.csv`), specific output keys (e.g. `key 'result'`), and requested schemas.
   - Evaluates quality signals: detects circular repetition, filler pleasantries, underspecification, and whether the prompt is already concise and effective.

2. **Stage B — Select an Optimization Strategy (`StrategySelector`)**:
   - Grounded primarily in the prompt's intrinsic characteristics:
     - **Compress**: Removes circular redundancy and conversational fluff while preserving meaning.
     - **Clarify**: Makes underspecified tasks actionable without inventing unsupported constraints.
     - **Structure**: Organizes multi-step or multi-constraint requirements into clean bulleted sections.
     - **Enrich**: Adds useful domain guidance when supported by intent.
     - **Preserve**: Keeps prompts unchanged when they are already concise, unambiguous, and effective.
   - Profiles (`balanced`, `concise`, `structure`) influence output formatting style, but never override explicit constraints.

3. **Stage C — Rewrite the Prompt (`PromptRewriter`)**:
   - Dedicated local model instructions for Ollama `qwen2.5:1.5b` with strict guardrails against hallucinating unrequested formats (e.g., unrequested mandatory JSON).

4. **Stage D — Validate Candidate (`CandidateValidator`)**:
   - Deterministic verification ensuring:
     - No empty or whitespace-only output.
     - Zero loss or inversion of explicit negative constraints or prohibitions.
     - Exact retention of filenames, technical identifiers, and output keys.
     - No unrequested mandatory JSON format hallucination.
     - No echoing of internal system instructions.
   - Controlled retry upon validation failure, with automatic fallback to the preserved original prompt if safety cannot be established.

### Explicit Constraints vs. Inferred Suggestions
- **Explicit constraints** (e.g., `"do not use pandas"`, `"read sales.csv"`) are non-negotiable and strictly locked by deterministic validation.
- **Inferred improvements** clarify ambiguity or organize presentation, but never silently impose arbitrary tools, libraries, or schemas.

### Why Some Optimized Prompts Are Longer
Underspecified or vague prompts often require concrete deliverable definitions and operational steps to achieve high-quality results from downstream frontier models. Argus permits justified prompt expansion and reports positive token and character deltas honestly rather than artificially claiming token savings.

### Heuristic Token Estimates Notice
Token estimates in Argus use documented character heuristics (~4 characters per token). They serve as fast local approximations and should not be confused with verified BPE tokenization counts, which are measured upon frontier provider dispatch.

---

## 🛡️ Semantic Preservation & Zero-Hardcoding Standard (Phase 3.2)

Phase 3.2 introduces strict architectural guardrails against interactive role inversion and enforces a project-wide engineering standard against hardcoded business logic or memorized inputs.

### 1. Root Cause & Semantic Preservation Fix
- **Failure Mode Resolved**: In browser evaluation, the prompt *"Build me a Python program to analyze student marks and tell me useful things"* previously yielded *"Please provide the student marks data for analysis"*, incorrectly transforming a code-generation task into an interactive data-request.
- **Root Cause**: The model interpreted ambiguous clarification directives as an interactive dialogue prompt, and `CandidateValidator` lacked deliverable-retention checks.
- **Remediation**:
  1. **Action & Deliverable Intent Extraction (`TaskAnalyzer`)**: Identifies primary action verbs (`build`, `create`, `write`, `explain`, `summarize`) and deliverables (`program`, `script`, `function`, `explanation`, `plan`).
  2. **Interactive Turn-Reversal Guardrail (`CandidateValidator`)**: Detects and rejects candidates that request user data (`"Please provide the data..."`, `"Upload your file..."`) instead of formulating an actionable prompt for an LLM.
  3. **Deliverable Family Protection**: Ensures code requests retain software/runnable terms and explanation requests retain conceptual walkthrough directives.
  4. **Corrective Retry**: On validation failure, Argus retries once with explicit corrective instructions before safely falling back to the original baseline.

### 2. Zero-Hardcoding Engineering Standard
- **No Prompt Memorization**: All extraction and validation logic uses reusable regex taxonomies and structural rules that generalize to unseen prompts.
- **Centralized Configuration**: All timeouts, model tags, hyperparameters (`temperature`, `top_p`), and heuristic ratios reside in `Settings` (`backend/app/core/config.py`) driven by `.env`.
- **Secrets Isolation**: Secrets remain on the backend and are excluded from source control.
- **Derived Telemetry**: Latencies and character/token deltas are computed dynamically from runtime inputs and outputs.

### 3. Phase 3.2.1: Required-Operation Preservation Engine
- **Failure Mode Resolved**: In structured rewriting of multi-operation prompts (e.g. *"Create a Python program that reads students.csv, calculates average marks, identifies the highest-scoring student, and saves the results to report.csv. Do not use pandas"*), skeletal rewrites preserved filenames, deliverable, and prohibitions, but dropped internal functional operations (calculating averages, finding extremes).
- **Generic Operation Extraction (`TaskAnalyzer`)**: Analyzes multi-clause sentences, coordinate conjunctions, relative clauses, and ordered lists to segment and extract required operations across ingestion, computation/aggregation, transformation/grouping, extreme identification, extraction, and output storage.
- **Operation Preservation Validation (`CandidateValidator`)**: Verifies that each required operational concept, qualifier (`average`/`mean`, `highest`/`top`, `total`/`sum`), and associated entity is represented in candidate output. Rejects candidates missing functional operations and triggers corrective retry or safe fallback.
- **Known Limitations**: Relies on robust canonical verb families and statistical concept clusters. Does not execute code or perform symbolic theorem proving. If an operation cannot be matched confidently, the validator conservatively rejects rather than silently dropping requirements.

### 4. Phase 3.2.2: Strategy Accuracy, Minimal Rewriting & Honest Metrics
- **Purpose of Optimization Profiles**:
  - **Balanced**: Preserves natural language flow while stripping filler and conversational redundancy. Prefers concise phrasing when the prompt is clear, expanding only when added detail is genuinely beneficial.
  - **Concise**: Aggressively focuses on minimal token footprint. Strips redundant words and unnecessary instructions while preserving 100% of required operations, filenames, prohibitions, and constraints. Never adds synthetic sections.
  - **Structure-focused**: Applies concise markdown headers and bullet points only when multi-step complexity warrants structural grouping. Avoids rigid boilerplate templates (e.g. `Task`, `Inputs/Context`, `Constraints`) when a direct rewrite is cleaner.
- **Selected Profile vs. Applied Strategy**:
  - `mode` represents the user's requested optimization style.
  - `applied_strategy` records what the engine reliably performed (`compress`, `structure`, `clarify`, `preserve`). If Structure-focused mode is chosen but the model produces flat text or falls back to the original, the engine reports `compress` or `preserve`, never falsely claiming structural reorganization.
- **Unchanged Prompts & Fallback Preservations**:
  - When a prompt is already effective, unambiguous, and concise, the engine preserves it unchanged (`outcome: unchanged_original`).
  - When candidate rewrites fail constraint or operation validation on both initial and retry attempts, the engine safely preserves the original prompt (`outcome: fallback`) rather than returning a corrupted prompt.
- **Valid Prompt Expansion**:
  - Prompts with vague requirements or underspecified domains occasionally expand to add critical technical parameters (`length_result: expanded`).
  - In such cases, Argus reports honest telemetry: character delta is positive (`final - original > 0`), token savings are reported as 0 (no false savings), and the status badge indicates `Validated Rewrite — Expanded`.
- **Heuristic Token Estimation & Cost Savings**:
  - Token counts are heuristic estimates based on standard character-to-token ratios (~4 chars/token). They indicate approximate trends rather than exact BPE/WordPiece tokenizer counts.
  - Validation success proves semantic and constraint retention; it does **not** claim verified dollar savings until real downstream LLM execution is performed in Phase 4.

---

## Testing & Verification

### Run Backend Tests (Pytest)
```bash
cd backend
.\.venv\Scripts\pytest -v    # On Linux/macOS: pytest -v
```
All 36 backend tests (including Cases A through H, Phase 3.2, Phase 3.2.1, and Phase 3.2.2 Tests A through H) pass.

### Run Frontend Tests (Vitest)
```bash
cd frontend
npm run test
```
All 11 frontend test suites pass.

### Run Frontend Production Build
```bash
cd frontend
npm run build
```

---

## Planned Implementation Phases

| Phase | Title | Scope & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Repository Foundation & Project Organization** | Clean directory structure, FastAPI foundation, health endpoint, React 19 + Tailwind v4 setup, design tokens, CI workflow, and documentation. | ✅ **Complete** |
| **Phase 2** | **UI/UX Design & Dashboard Implementation** | Responsive workspace, prompt editor, profile selection, side-by-side comparison, metrics panel. | ✅ **Complete** |
| **Phase 3** | **Real Local LLM Integration** | Ollama local HTTP API (`http://127.0.0.1:11434`), `qwen2.5:1.5b` integration, prompt optimization prompt engineering, typed API contracts, error handling, status monitoring. | ✅ **Complete** |
| **Phase 3.1** | **Task-Aware Prompt Optimization Engine** | Modular pipeline (Stages A-D), intent & domain classification, strategy selection (`compress`, `clarify`, `structure`, `enrich`, `preserve`), explicit constraint & prohibition guardrails, candidate validation, honest heuristic metrics. | ✅ **Complete** |
| **Phase 3.2** | **Semantic Preservation & Zero-Hardcoding** | Action/deliverable intent extraction, anti-turn-reversal validation, corrective retry/fallback, generalized taxonomies, centralized configuration. | ✅ **Complete** |
| **Phase 3.2.1** | **Required-Operation Preservation Fix** | Multi-clause operation extraction, computation/transformation preservation validation, concept synonym mapping, retry/fallback integration. | ✅ **Complete** |
| **Phase 3.2.2** | **Strategy Accuracy, Minimal Rewriting & Honest Metrics** | Profile refinement, unsupported addition prevention, accurate applied strategy metadata, consistent character delta (`final - original`), honest status badges. | ✅ **Complete** |
| **Phase 4** | **Main LLM Dispatch & Provider Adapters** | Configurable external provider adapter (OpenAI), Baseline vs. Optimized paths, reproducible benchmark suite, token & cost telemetry. | ✅ **Complete** |
| **Phase 5** | **Dashboard & Optimization Telemetry** | Interactive prompt optimization playground, token savings metrics, cost estimation analytics. | ⏳ Planned |
| **Phase 6** | **Cloud Deployment & Production Hardening** | Dockerization, Azure Container Apps / AWS ECS deployment, and observability. | ⏳ Planned |

---

## 📊 Main-LLM Integration & Reproducible Benchmark (Phase 4)

Phase 4 introduces an empirical, reproducible benchmark measuring whether Argus's local prompt optimization reduces external LLM inference costs without degrading answer accuracy or violating explicit constraints.

### 1. Main-LLM Provider Integration
- **Configurable Provider Adapter**: Supports external LLMs using official HTTP API standards (OpenAI `gpt-4o` and `gpt-4o-mini`).
- **Zero Client Credential Exposure**: API keys (`OPENAI_API_KEY`) are read exclusively from server-side environment variables and never exposed to the frontend, logs, test fixtures, or Git.
- **Local Ollama Optimization Intact**: Prompt rewriting continues to be handled locally by Ollama (`qwen2.5:1.5b`) regardless of whether external credentials are set.
- **Strict Error Handling**: Detailed, typed exceptions for missing credentials (HTTP 400), rate limits (HTTP 429), timeouts (HTTP 504), and provider failures (HTTP 502).

### 2. Baseline vs. Optimized Execution Pipeline
Argus supports two comparable paths using the identical target model and generation settings (`temperature=0.2`, `max_tokens=1500`):
- **Path A (Baseline)**: Original prompt &rarr; Main LLM.
- **Path B (Optimized)**: Original prompt &rarr; Local Argus Optimizer &rarr; Main LLM.

Both prompts and their individual token and cost footprints are preserved and measured separately.

### 3. Reproducible Benchmark Dataset (5 Scenarios)
A version-controlled benchmark dataset (`backend/app/benchmarks/dataset.json`) covers:
1. **Simple Informational**: Direct queries requiring concise answers.
2. **Verbose / Redundant**: Prompts loaded with conversational filler and pleasantries.
3. **Coding with Explicit Constraints**: Tasks with prohibitions (e.g. `"do not use pandas"`), filenames, and required outputs.
4. **Structured JSON Extraction**: Schema extraction demanding strict JSON parseability and required keys.
5. **Concise Baseline (Preserve)**: Already-effective prompts where Argus verifies that no optimization should be applied (`strategy: preserve`).

### 4. Empirical Benchmark Measurements
Running the reproducible benchmark against real local Ollama inference produces:
- **Test Cases Passed**: 5 of 5 (100.0%)
- **Prompt Token Reduction**: **33.46%** (254 &rarr; 169 tokens)
- **Total Cost Reduction**: **6.78%** ($0.003135 &rarr; $0.002923)
- **Constraint Retention Rate**: **100.0%** (zero negative constraint violations or key drops)
- **Average Optimizer Latency**: ~373 ms overhead

### 5. Running the Benchmark CLI
```bash
# Run safe mock benchmark (zero paid API calls):
cd backend
.\.venv\Scripts\python.exe -m app.benchmarks.runner

# Run opt-in live benchmark with real OpenAI API:
$env:OPENAI_API_KEY="sk-your-openai-api-key"
.\.venv\Scripts\python.exe -m app.benchmarks.runner --live --model gpt-4o --output benchmark_report.json
```

---

## Contributing & Development Guide

For detailed guidelines on branch naming, code style, and test coverage, see the [`docs/development-guide.md`](docs/development-guide.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
