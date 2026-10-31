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
- [Roadmap & Implementation Phases](#roadmap--implementation-phases)
- [Contributing & Development Guide](#contributing--development-guide)
- [License](#license)

---

## Project Purpose & Scope

Large Language Models (LLMs) incur significant processing latency and inference costs scaled directly by prompt volume. Complex instructions, conversational pleasantries, circular repetitions, and unstructured requests degrade prompt execution quality and waste resources.

**Argus** functions as an intelligent, local prompt optimization middleware:
1. **Analyzes Intent & Constraints**: Performs deterministic task inspection to extract domain categories, primary action verbs, deliverables, required functional operations, and explicit constraints (e.g., negative prohibitions, filenames, requested keys).
2. **Selects Optimization Strategy**: Chooses an appropriate strategy (`compress`, `clarify`, `structure`, `enrich`, or `preserve`) based on prompt characteristics and the user's selected profile.
3. **Optimizes Locally via Neural Model**: Rewrites the prompt using a local lightweight LLM (**Qwen2.5-1.5B-Instruct** running via **Ollama**) without external API calls or token costs.
4. **Validates Semantic Fidelity**: Employs deterministic validation guardrails to ensure deliverables are maintained, interactive turn-reversals are prevented, and all required operations and prohibitions remain intact.
5. **Provides Honest Telemetry**: Displays side-by-side prompt diffs, character deltas, heuristic token estimates, and constraint preservation tags in a modern React workspace.

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
       ├──▶ [ Stage A: TaskAnalyzer (Domain, Deliverables, Constraints) ]
       ├──▶ [ Stage B: StrategySelector (Compress / Clarify / Structure / Preserve) ]
       ├──▶ [ Stage C: PromptRewriter (System Instructions & Guardrails) ]
       ├──▶ [ Local Inference Tier: Ollama Daemon (qwen2.5:1.5b via HTTP 11434) ]
       └──▶ [ Stage D: CandidateValidator (Constraint & Intent Guardrails) ]
```

- **Frontend Isolation**: Frontend components serve strictly as presentation, editing, and status monitoring interfaces. They contain **no** prompt compression heuristics or local LLM execution logic.
- **Backend Service Layer**: Handles prompt analysis, strategy determination, prompt instruction engineering, Ollama HTTP communication, and validation retries/fallbacks.
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
│   │   ├── schemas/                   # Pydantic validation and response schemas
│   │   │   ├── health.py              # HealthResponse schema
│   │   │   └── optimizer.py           # OptimizeRequest, OptimizeResponse, OllamaStatusResponse
│   │   ├── services/                  # Business logic and optimization pipeline
│   │   │   ├── candidate_validator.py # Stage D: Semantic retention guardrails & turn-reversal check
│   │   │   ├── health_service.py      # Health service implementation
│   │   │   ├── ollama_service.py      # Ollama HTTP client & pipeline orchestrator
│   │   │   ├── prompt_rewriter.py     # Stage C: Instruction engineering for qwen2.5:1.5b
│   │   │   ├── strategy_selector.py   # Stage B: Strategy selection (compress, clarify, structure, preserve)
│   │   │   └── task_analyzer.py       # Stage A: Task classification & constraint extraction
│   │   └── main.py                    # FastAPI application factory, CORS, and error handlers
│   ├── tests/                         # Pytest test suite (mocked Ollama client)
│   │   ├── conftest.py                # Test fixtures and async client configuration
│   │   ├── test_health.py             # Health endpoint tests
│   │   └── test_optimizer.py          # 34 prompt optimization and validation tests
│   ├── .env.example                   # Backend environment configuration template
│   ├── pyproject.toml                 # Backend project metadata and tools
│   ├── requirements.txt               # Production dependencies
│   └── requirements-dev.txt           # Development and testing dependencies
├── frontend/
│   ├── public/                        # Static brand assets
│   ├── src/
│   │   ├── components/                # Reusable UI components
│   │   │   ├── Button.tsx             # Accessible button component
│   │   │   ├── Card.tsx               # Container panel component
│   │   │   ├── ComparisonView.tsx     # Side-by-side prompt diff and telemetry view
│   │   │   ├── Header.tsx             # Brand header with backend & Ollama status pills
│   │   │   ├── MetricsPanel.tsx       # Estimated token metrics, character deltas, footprint bars
│   │   │   ├── OptimizationControls.tsx # Mode selector (balanced, concise, structure)
│   │   │   ├── PromptEditor.tsx       # Prompt input editor with character & token counters
│   │   │   └── StatusBadge.tsx        # Outcome and validation status badges
│   │   ├── hooks/                     # Custom React hooks (useHealth, useOllamaStatus)
│   │   ├── pages/                     # Application pages (WorkspacePage, HomePage)
│   │   ├── services/                  # API client layer (api.ts) & offline mock (mockOptimizer.ts)
│   │   ├── styles/                    # Tailwind CSS v4 styling & tokens
│   │   ├── types/                     # TypeScript domain models and interfaces
│   │   ├── App.tsx                    # Root layout with tab navigation (Workspace & System Status)
│   │   └── main.tsx                   # DOM entry point
│   ├── tests/                         # Vitest unit and component tests
│   │   ├── setup.ts                   # Testing Library setup
│   │   ├── App.test.tsx               # Navigation and layout smoke tests
│   │   └── Workspace.test.tsx         # Workspace interactions and optimization flow tests
│   ├── .env.example                   # Frontend environment configuration template
│   ├── package.json                   # NPM dependencies and scripts
│   ├── tsconfig.json                  # TypeScript compiler options
│   └── vite.config.ts                 # Vite bundler configuration
├── docs/
│   ├── architecture.md                # System architecture, boundaries, and sequence diagrams
│   └── development-guide.md           # Local setup, environment reference, testing, and troubleshooting
├── scripts/
│   ├── dev-backend.ps1 / .sh          # Convenience script: Launch backend
│   └── dev-frontend.ps1 / .sh         # Convenience script: Launch frontend
├── .gitignore                         # Comprehensive source control exclusions
├── LICENSE                            # MIT License
└── README.md                          # Project overview and documentation
```

---

## Prerequisites

- **Node.js**: `v20.x` or `v22.x LTS`
- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: `v2.40+`
- **Ollama**: Installed locally on `http://127.0.0.1:11434`

---

## Ollama Setup & Model Verification

1. **Verify Ollama is installed**:
   ```bash
   ollama --version
   ```
2. **Start the Ollama daemon** (if not already running):
   ```bash
   ollama serve
   ```
3. **Pull the Qwen2.5 1.5B Instruct model**:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
4. **Confirm model presence**:
   ```bash
   ollama list
   ```

---

## Installation & Local Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/MK-KOUSHIK/argus.git
cd argus
```

### 2. Backend Setup
```powershell
cd backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Create environment configuration
cp .env.example .env

# Run FastAPI dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now running at **`http://localhost:8000`**.  
Interactive OpenAPI documentation: **`http://localhost:8000/api/docs`**  
Ollama status check: **`http://localhost:8000/api/ollama/status`**

### 3. Frontend Setup
```powershell
# In a separate terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create environment configuration
cp .env.example .env

# Start Vite dev server
npm run dev
```
The frontend is now accessible at **`http://localhost:5173`**.

---

## API Usage

### 1. Optimize Prompt (`POST /api/optimize`)
Submits a raw prompt for task-aware local optimization:

```json
{
  "prompt": "Write a Python script that reads sales.csv, calculates monthly revenue, saves monthly_report.csv, and does not use pandas.",
  "mode": "balanced",
  "target_model": "qwen2.5:1.5b"
}
```

**Response (`200 OK`)**:
```json
{
  "original_prompt": "Write a Python script that reads sales.csv, calculates monthly revenue, saves monthly_report.csv, and does not use pandas.",
  "optimized_prompt": "Write a Python script to read sales.csv, calculate monthly revenue, and save the result to monthly_report.csv without using pandas.",
  "mode": "balanced",
  "target_model": "qwen2.5:1.5b",
  "task_category": "coding",
  "strategy": "compress",
  "applied_strategy": "compress",
  "outcome": "validated_rewrite",
  "length_result": "compressed",
  "validation_status": "passed",
  "is_preserved": false,
  "primary_action": "write",
  "primary_deliverable": "script",
  "explicit_constraints": [
    "Deliverable: Script",
    "Operation: Save python script",
    "Operation: Read sales csv",
    "Operation: Calculate monthly revenue",
    "Operation: Save monthly report csv",
    "Prohibition: Do not use pandas"
  ],
  "required_operations": [
    "save python script",
    "read sales csv",
    "calculate monthly revenue",
    "save monthly report csv"
  ],
  "original_metrics": { "char_count": 126, "estimated_tokens": 32 },
  "optimized_metrics": { "char_count": 133, "estimated_tokens": 34 },
  "token_reduction_percent": 0,
  "token_change_percent": 6,
  "token_savings": 0,
  "char_delta": 7,
  "preserved_constraints": ["Core query intent & technical details"],
  "execution_time_ms": 3215,
  "model": "qwen2.5:1.5b",
  "timestamp": "2026-10-31T04:22:18Z"
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
  "error": null
}
```

### 3. System Health (`GET /api/health`)
Returns backend operational status, version, and environment:
```json
{
  "status": "ok",
  "app_name": "Argus",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-10-31T04:22:04.582179+00:00",
  "message": "Argus API is operational"
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
[ Stage B: StrategySelector ] ───▶ Selects compress, clarify, structure, enrich, or preserve
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

1. **Stage A — Understand Prompt (`TaskAnalyzer`)**:
   - Classifies task domain (`coding`, `data_analysis`, `writing`, `extraction`, `summarization`, `research`, `planning`, or `general`).
   - Extracts primary action verb and deliverable (e.g. `write script`, `create function`).
   - Segments multi-clause required functional operations (e.g. data ingestion, computation, output formatting).
   - Identifies explicit constraints (e.g. negative prohibitions like `"do not use pandas"`, filenames like `"sales.csv"`, and output schemas).
   - Detects quality signals (circular redundancy, conversational pleasantries, underspecification).

2. **Stage B — Select Strategy (`StrategySelector`)**:
   - Assigns `preserve` if the prompt is already concise and actionable.
   - Assigns `clarify` if the prompt is vague or underspecified.
   - Assigns `compress` if the prompt contains filler or repetitive phrasing.
   - Assigns `structure` if the prompt contains multi-step operations or complex constraints.
   - Modulates formatting according to the user's profile (`balanced`, `concise`, `structure`).

3. **Stage C — Rewrite Prompt (`PromptRewriter`)**:
   - Formulates task-aware system instructions for `qwen2.5:1.5b`.
   - Strictly enforces deliverable retention and forbids interactive turn-reversals.
   - Provides targeted corrective instructions during retry attempts.

4. **Stage D — Validate Candidate (`CandidateValidator`)**:
   - Verifies primary deliverable and action verb retention.
   - Detects turn-reversal anti-patterns (rejects responses asking the user to provide data).
   - Confirms negative prohibitions are not dropped or inverted.
   - Ensures filenames, technical keys, and identifiers are strictly retained.
   - Rejects unrequested mandatory JSON schemas and instruction leaks.
   - Executes a single corrective retry upon failure; if the retry also fails, safely falls back to the original prompt (`outcome: fallback`).

---

## 🛡️ Semantic Preservation & Zero-Hardcoding

Argus adheres to strict architectural guardrails:
- **Anti-Turn-Reversal Guardrail**: Prevents small local models from misinterpreting a prompt as a conversational inquiry (e.g., transforming `"Build a Python program to analyze student grades"` into `"Please upload the student grades file"`).
- **Required-Operation Preservation**: Ensures multi-clause operations (calculating averages, identifying extremes, writing files) are not pruned during structural simplification.
- **Zero-Hardcoding Standard**: Extraction and validation logic relies on generalized taxonomies and structural patterns rather than hardcoded prompts or test-specific strings. Centralized configuration in `backend/app/core/config.py` drives all parameters.

---

## 📊 Token Estimation & Honest Metrics Standard

### Character-Based Heuristic Notice
Token counts reported in Argus use a standard character-to-token heuristic:

$$\text{estimated\_tokens} = \max\left(1, \left\lceil \frac{\text{len}(\text{trimmed\_text})}{4.0} \right\rceil\right)$$

- **Estimations, Not Exact BPE Counts**: Character heuristics provide fast, local approximations of text footprint. They are **estimates** and do not represent exact Byte-Pair Encoding (BPE) counts from proprietary cloud tokenizers.
- **Honest Character Deltas**: Defined as $\text{length}(\text{optimized}) - \text{length}(\text{original})$. A negative number indicates compression; a positive number indicates expansion.
- **Valid Prompt Expansion**: Prompts that are vague or missing critical technical details may expand to ensure downstream LLMs generate accurate results. In such cases, Argus reports honest telemetry: token savings are reported as `0` (never negative or fabricated), and the status badge indicates `Validated Rewrite — Expanded`.
- **Accurate Applied Strategy**: If a user selects `structure` mode but the rewrite produces plain text or falls back to the original prompt, the backend records `compress` or `preserve`, never misrepresenting structural changes.

---

## Testing & Verification

### 1. Backend Tests (Pytest)
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```
- **36 passing tests** covering health check, optimizer route, task analysis, strategy selection, candidate validation, semantic preservation, and Ollama error handling.
- HTTP calls to Ollama are mocked in automated tests, allowing the suite to run offline.

### 2. Frontend Tests (Vitest)
```bash
cd frontend
npm.cmd test -- --run
```
- **11 passing tests** across `App.test.tsx` and `Workspace.test.tsx` validating layout, tab navigation, user input, mode switching, diff rendering, and mock API integration.

### 3. Frontend Production Build
```bash
cd frontend
npm.cmd run build
```
- Executes TypeScript validation (`tsc -b`) and bundles production assets via Vite with zero type errors.

---

## Roadmap & Implementation Phases

| Phase | Milestone | Scope & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Repository Foundation & Project Organization** | Scaffolding, design tokens, FastAPI foundation, health endpoint, React 19 + Tailwind CSS v4, Vitest, CI workflow. | ✅ Complete |
| **Phase 2** | **UI/UX Workspace & System Status** | Responsive workspace, prompt editor, profile selection, side-by-side comparison view, metrics panel, System Status page. | ✅ Complete |
| **Phase 3** | **Real Local LLM Integration** | Ollama HTTP integration (`http://127.0.0.1:11434`), `qwen2.5:1.5b` model communication, error handling, status monitoring. | ✅ Complete |
| **Phase 3.1** | **Task-Aware Optimization Engine** | Modular pipeline (Stages A–D), task domain classification, strategy selection (`compress`, `clarify`, `structure`, `enrich`, `preserve`), explicit constraint & prohibition guardrails, candidate validation. | ✅ Complete |
| **Phase 3.2** | **Semantic Preservation & Zero-Hardcoding** | Action/deliverable intent extraction, anti-turn-reversal validation, corrective retry/fallback, generalized taxonomies, centralized configuration. | ✅ Complete |
| **Phase 3.2.1** | **Required-Operation Preservation Engine** | Multi-clause operation extraction, computation/transformation preservation validation, concept synonym mapping, retry/fallback integration. | ✅ Complete |
| **Phase 3.2.2** | **Strategy Accuracy, Minimal Rewriting & Honest Metrics** | Profile refinement, unsupported addition prevention, accurate applied strategy metadata, consistent character delta (`final - original`), honest status badges. | ✅ Complete |
| **Phase 4** | **Frontier LLM Gateway & Provider Dispatch** | Configurable external provider integration (Google Gemini / OpenAI), optional direct downstream execution, and frontier token metrics. | ⏳ Planned |
| **Phase 5** | **Optimization Telemetry & Benchmarking** | Reproducible prompt evaluation datasets, automated constraint verification, quality scoring, and optimization analytics. | ⏳ Planned |
| **Phase 6** | **Cloud Deployment & Production Hardening** | Dockerization, Azure Container Apps / AWS ECS deployment, observability, and containerized Ollama sidecars. | ⏳ Planned |

---

## Contributing & Development Guide

For local setup instructions, coding conventions, testing guidelines, and troubleshooting, see [`docs/development-guide.md`](docs/development-guide.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
