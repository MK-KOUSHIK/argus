# Argus System Architecture

## 1. System Overview

Argus is an intelligent prompt optimization middleware application. Its primary objective is to dramatically decrease input-token consumption and inference costs on expensive frontier LLMs (e.g., OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet, Google Gemini 1.5 Pro) without sacrificing response accuracy or constraint fidelity.

```mermaid
graph LR
    Client[Client / Web UI] -->|Prompt Payload| Backend[Argus Middleware API]
    subgraph Argus Backend Services
        Backend --> IntentExtract[Intent & Constraint Extractor]
        IntentExtract --> LocalOptimizer[Local LLM Optimizer<br/>Qwen2.5-1.5B via Ollama]
        LocalOptimizer --> Verification[Constraint Verification & Diff]
        Verification --> CostTelemetry[Token & Cost Telemetry]
    end
    CostTelemetry -->|Optimized Prompt| MainLLM[Main Frontier LLM<br/>OpenAI / Anthropic / Gemini]
    MainLLM -->|Response| Backend
    Backend -->|Response + Savings Metrics| Client
```

---

## 2. Architectural Boundaries

Strict architectural boundaries are maintained across the stack:

1. **Frontend Isolation**:
   - The frontend (`frontend/src/`) functions strictly as a presentation, configuration, and monitoring layer.
   - **No prompt optimization logic, tokenization mathematics, or provider API calling belongs in frontend code.**
   - All network interactions occur exclusively through typed service contracts (`frontend/src/services/api.ts`).

2. **Backend Domain Isolation**:
   - **`api/`**: Route definitions, HTTP protocol translation, status codes, and input/output schema mapping. No raw business logic.
   - **`schemas/`**: Pydantic models for strict contract validation and serialization.
   - **`services/`**: Core domain logic, orchestration, prompt compression routines, and local/frontier LLM integrations.
   - **`core/`**: Centralized configuration (`config.py`), logging, security utilities, and lifespan handlers.

3. **Inference Separation**:
   - **Local Inference Tier**: Handled by Ollama running `Qwen2.5-1.5B-Instruct` locally or in a co-located container. It executes the prompt compression heuristics.
   - **Frontier Inference Tier**: Cloud LLM providers are contacted solely by the backend via secure provider SDKs/HTTP clients using server-side environment secrets.

---

## 3. Security & Credential Management

- **Zero Client Credential Exposure**: Client-side bundles never contain provider API keys (OpenAI, Anthropic, AWS, Azure, etc.).
- **Environment Isolation**: Sensitive configuration is managed through server-side `.env` files and platform secret stores (AWS Secrets Manager, Azure Key Vault).
- **CORS Protection**: FastAPI enforces strict origin verification limiting browser requests to explicitly allowed frontend domains.
- **Git Hygiene**: Model weights (`.gguf`, `.safetensors`, `.bin`), virtual environments, build artifacts, and secret files are permanently excluded from version control via `.gitignore`.

---

## 4. Phase-by-Phase Roadmap

| Phase | Milestone | Focus Area |
| :--- | :--- | :--- |
| **Phase 1** | **Repository Foundation & Project Organization** | Scaffolding, design tokens, FastAPI structure, health endpoint, CI, testing harness. *(Current)* |
| **Phase 2** | **Core Prompt Processing Pipeline** | Intent extraction, prompt token analysis, and compression rule framework. |
| **Phase 3** | **Local LLM Integration** | Ollama integration with `Qwen2.5-1.5B-Instruct`, local inference benchmarks. |
| **Phase 4** | **Main LLM Dispatch & Provider Adapters** | Provider clients (OpenAI, Anthropic, etc.), streaming proxying, and fallbacks. |
| **Phase 5** | **Dashboard & Optimization Telemetry** | Interactive prompt playground, token savings metrics, cost estimation analytics. |
| **Phase 6** | **Cloud Deployment & Production Hardening** | Dockerization, Azure Container Apps / AWS ECS deployment, and observability. |

---

## 5. Technology Stack Reference

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Vitest.
- **Backend**: Python 3.10+, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, Pytest.
- **Local Inference**: Ollama with `Qwen2.5-1.5B-Instruct`.
- **Target Deployment**: Docker, Azure Container Apps / AWS ECS, CloudFront / Azure Static Web Apps.
