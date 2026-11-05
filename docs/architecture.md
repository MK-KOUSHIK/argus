# Argus System Architecture

## 1. System Overview

**Argus** is an intelligent, task-aware prompt optimization middleware application. Its primary objective is to optimize, structure, and compress prompts locally using a compact neural model (**Qwen2.5-1.5B-Instruct** running via **Ollama**) before prompts are dispatched to downstream Large Language Models.

Argus ensures that explicit user constraints, required functional operations, prohibitions, filenames, and domain intents are strictly preserved while eliminating conversational filler, circular redundancy, and ambiguity.

```mermaid
graph TD
    User["User / Client"] -->|"Submits Prompt + Profile + Optional Budget"| ReactApp["React Frontend (Workspace / Status)"]
    ReactApp -->|"POST /api/optimize"| APIRouter["FastAPI API Router (/api)"]

    subgraph BackendPipeline["Argus Backend Optimization Pipeline"]
        APIRouter -->|"1. Raw Prompt"| Stage0["Stage 0: Compression Engine<br/>(Content Router, Specialized Minifiers, Reversible Registry)"]
        Stage0 -->|"Effective Prompt"| TaskAnalyzer["Stage A: Task Analyzer<br/>(Domain, Action, Deliverable, Operations, Constraints)"]
        Stage0 -.->|"Omitted content ref"| RegistryStore[("Compression Registry<br/>(Thread-Safe LRU/TTL Store)")]
        TaskAnalyzer -->|"2. Task Analysis"| StrategySelector["Stage B: Strategy Selector<br/>(Compress, Clarify, Structure, Enrich, Preserve)"]
        TaskAnalyzer -->|"2b. Analysis + Directives"| OutputBudget["Output Budget Controller<br/>(Task Baselines, Word/Sentence Directives, Mode Scaling)"]
        
        OutputBudget -->|"3. Output Budget"| AdaptiveContext["Adaptive Context Manager<br/>(Calculates Context Tier with Dynamic Headroom)"]
        StrategySelector -->|"4. Strategy + Analysis"| PromptRewriter["Stage C: Prompt Rewriter<br/>(Task-Aware System Instructions)"]
        PromptRewriter -->|"5. Instruction + Prompt + Budget"| OllamaService["Ollama Service Client<br/>(HTTPX Async Client)"]
        
        subgraph LocalInference["Local Inference Tier"]
            OllamaService <-->|"POST /api/generate<br/>(num_ctx tier, num_predict budget)"| OllamaServer["Ollama Daemon (127.0.0.1:11434)<br/>Model: qwen2.5:1.5b (Q8_0 KV Cache + Flash Attention)"]
        end

        OllamaService -->|"6. Candidate Rewrite"| CandidateValidator["Stage D: Candidate Validator<br/>(Semantic Guardrails & Turn-Reversal Check)"]
        
        CandidateValidator -->|"Validation Failed (Retry 1)"| PromptRewriter
        CandidateValidator -->|"Passed OR Fallback"| MetricsEngine["Metrics & Native Telemetry<br/>(eval_count, done_reason: stop vs length, utilization%)"]
    end

    MetricsEngine -->|"Structured OptimizeResponse"| APIRouter
    APIRouter -->|"JSON Telemetry + Preserved/Rewritten Prompt"| ReactApp
    ReactApp -->|"Visual Diff, Metrics & Status Badges"| User
```

---

## 2. Architectural Boundaries

Argus enforces strict separation of concerns across presentation, routing, orchestration, and inference:

1. **Frontend Isolation (`frontend/src/`)**:
   - The React application operates solely as a presentation, configuration, and monitoring interface.
   - **No prompt optimization logic, token estimation heuristics, or local model calling resides in frontend code.**
   - All backend interactions occur through strongly-typed API service contracts in [`src/services/api.ts`](file:///d:/Argus/frontend/src/services/api.ts).
   - An offline fallback mock service [`src/services/mockOptimizer.ts`](file:///d:/Argus/frontend/src/services/mockOptimizer.ts) allows UI development without a running backend.

2. **Backend Domain Isolation (`backend/app/`)**:
   - **`api/`**: HTTP route handlers, request parsing, status codes, and endpoint mapping. No raw business logic.
   - **`schemas/`**: Pydantic models declaring strict request, response, diagnostic, and telemetry contracts.
   - **`services/`**: Modular domain logic for task analysis, strategy selection, output budget control, prompt rewriting, candidate validation, adaptive context tier calculation, and Ollama communication.
   - **`core/`**: Centralized application configuration ([`config.py`](file:///d:/Argus/backend/app/core/config.py)), environment loading, and server lifespan events.

3. **Inference Tier Isolation**:
   - Local neural optimization is handled exclusively by **Ollama** running `qwen2.5:1.5b` over local HTTP (`http://127.0.0.1:11434`).
   - The application does not require any paid external cloud API keys for prompt optimization.

---

## 3. End-to-End Request Flow

The complete execution lifecycle for a prompt optimization request follows a deterministic 9-step pipeline:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React Frontend (Workspace)
    participant API as FastAPI Router (/api/optimize)
    participant S0 as CompressionEngine (Stage 0)
    participant TA as TaskAnalyzer (Stage A)
    participant OBC as OutputBudgetController
    participant ACM as AdaptiveContextManager
    participant SS as StrategySelector (Stage B)
    participant PR as PromptRewriter (Stage C)
    participant OLL as Ollama Service & Model
    participant CV as CandidateValidator (Stage D)

    User->>UI: Enters prompt, selects mode, optional budget override
    UI->>API: POST /api/optimize (prompt, mode, target_model, requested_output_budget, enable_compression)
    API->>S0: Stage 0: Content-Aware Compression (JSON, Code AST, Logs, Text)
    Note over S0: Classifies content, minifies structure, folds logs,<br/>stores omitted blocks in Reversible Registry
    S0-->>TA: Passes effective compressed prompt
    API->>TA: Analyze prompt characteristics
    Note over TA: Extracts domain, action verbs, deliverable,<br/>required operations, prohibitions, & filenames
    TA-->>OBC: Passes analysis & prompt directives
    OBC->>OBC: Calculates output budget (num_predict)<br/>from word/sentence cues, task baselines & mode
    OBC->>ACM: Passes calculated output budget
    ACM->>ACM: Snaps required context to stable tier (4096 / 6144 / 8192)
    TA-->>SS: Returns TaskAnalysisResult
    SS->>SS: Determines strategy (compress, clarify, structure, enrich, preserve)
    
    alt Strategy is PRESERVE (Prompt already concise & effective)
        SS-->>API: Preserve original prompt directly (0 inference overhead)
    else Strategy is COMPRESS, CLARIFY, STRUCTURE, or ENRICH
        SS->>PR: Build task-aware instructions
        PR-->>OLL: Invoke local Qwen2.5:1.5b (num_ctx tier, num_predict budget)
        OLL-->>CV: Cleaned candidate rewrite + native telemetry
        CV->>CV: Validate semantic retention, turn-reversal, & constraints
        
        alt Validation Fails
            CV->>PR: Build corrective retry instruction with failure details
            PR-->>OLL: Second inference attempt
            OLL-->>CV: Second candidate rewrite
            CV->>CV: Re-validate candidate
            alt Second Attempt Fails
                CV-->>API: Safe fallback: Return original prompt unchanged
            else Second Attempt Passes
                CV-->>API: Return validated second candidate
            end
        else Validation Passes
            CV-->>API: Return validated candidate
        end
    end

    API->>API: Extract native telemetry (done_reason, eval_count, utilization%)
    API-->>UI: Return OptimizeResponse JSON
    UI-->>User: Render side-by-side diff, token delta, output budget card & status badges
```

---

## 4. Backend Services and Responsibilities

| Service | File Path | Primary Responsibilities |
| :--- | :--- | :--- |
| **`TaskAnalyzer`** | [`backend/app/services/task_analyzer.py`](file:///d:/Argus/backend/app/services/task_analyzer.py) | **Deterministic Prompt Inspection (Stage A)**:<br/>• Detects task domain (`coding`, `data_analysis`, `writing`, `extraction`, `summarization`, `research`, `planning`, `general`).<br/>• Extracts primary action verb (e.g., `build`, `write`, `explain`) and deliverable (e.g., `script`, `function`, `plan`).<br/>• Segments and normalizes multi-clause required functional operations.<br/>• Identifies explicit constraints, negative prohibitions (`do not use pandas`), filenames, and requested schemas.<br/>• Evaluates quality signals (redundancy, underspecification, conversational filler). |
| **`OutputBudgetController`** | [`backend/app/services/output_budget_controller.py`](file:///d:/Argus/backend/app/services/output_budget_controller.py) | **Output Budget Allocation (Phase 4.1)**:<br/>• Computes dynamic generation token ceilings (`num_predict`).<br/>• Extracts explicit length directives via regex (word counts, sentence counts, paragraph counts, brevity cues).<br/>• Maps task baselines scaled by optimization mode (`concise`: 0.75x, `balanced`: 1.0x, `structure`: 1.2x) and complexity (+20%).<br/>• Enforces global bounds `[OUTPUT_BUDGET_MIN_TOKENS, OUTPUT_BUDGET_MAX_TOKENS]`.<br/>• Coordinates with context window headroom and rejects with HTTP 413 if headroom < 64 tokens. |
| **`AdaptiveContextManager`** | [`backend/app/services/adaptive_context_manager.py`](file:///d:/Argus/backend/app/services/adaptive_context_manager.py) | **Context-Length & Tier Derivation**:<br/>• Calculates total context footprint: $\text{System} + \text{Wrapper} + \text{Input} + \text{Output Budget} + \text{Headroom}$.<br/>• Derives eligible tiers bounded by `OLLAMA_MAX_NUM_CTX` (e.g. 4096, 6144, 8192).<br/>• Snaps requests to stable tiers to preserve KV-cache prefix reuse.<br/>• Enforces low-VRAM safeguards (<1000 MiB free) and rejects overflow with HTTP 413. |
| **`StrategySelector`** | [`backend/app/services/strategy_selector.py`](file:///d:/Argus/backend/app/services/strategy_selector.py) | **Optimization Strategy Assignment (Stage B)**:<br/>• Assigns `preserve` if prompt is already concise and actionable.<br/>• Assigns `clarify` if prompt is underspecified or vague.<br/>• Assigns `compress` if prompt contains circular redundancy or conversational filler.<br/>• Assigns `structure` if prompt contains multi-step operations or complex constraints. |
| **`PromptRewriter`** | [`backend/app/services/prompt_rewriter.py`](file:///d:/Argus/backend/app/services/prompt_rewriter.py) | **Instruction Engineering (Stage C)**:<br/>• Builds task-aware system instructions for `qwen2.5:1.5b`.<br/>• Enforces deliverable preservation.<br/>• Prohibits interactive turn reversals.<br/>• Generates targeted corrective instructions during validation retry attempts. |
| **`CandidateValidator`** | [`backend/app/services/candidate_validator.py`](file:///d:/Argus/backend/app/services/candidate_validator.py) | **Deterministic Verification Guardrails (Stage D)**:<br/>• Verifies primary action verb and deliverable preservation.<br/>• Detects interactive turn-reversals (`"Please provide the data..."`).<br/>• Checks retention of required functional operations and concepts.<br/>• Verifies negative constraints are not inverted or dropped.<br/>• Guarantees filenames, numbers, and technical identifiers are preserved. |
| **`ConcurrencyController`** | [`backend/app/services/concurrency_controller.py`](file:///d:/Argus/backend/app/services/concurrency_controller.py) | **Runtime Hardware-Aware Admission**:<br/>• Condition-based admission gate (`asyncio.Condition`).<br/>• Dynamically throttles capacity from 3 down to 1 under constrained VRAM (<400 MiB) or RAM (<1024 MiB).<br/>• Rejects requests with HTTP 503 if VRAM <150 MiB or RAM <512 MiB.<br/>• Tracks queue depth and wait latencies. |
| **`OllamaService`** | [`backend/app/services/ollama_service.py`](file:///d:/Argus/backend/app/services/ollama_service.py) | **Pipeline Orchestration & Model Communication**:<br/>• Coordinates Stages A through D and Output Budget Control.<br/>• Manages async HTTP calls to Ollama (`/api/generate` and `/api/tags`).<br/>• Extracts native telemetry (`prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `done_reason`, `generation_tokens_per_second`).<br/>• Executes corrective retries and handles safe fallback to original prompts. |

---

## 5. Token Estimation & Honest Metrics Standard

### Heuristic Character-Based Token Counting
Argus calculates estimated token counts using a standardized character-to-token heuristic:

$$\text{estimated\_tokens} = \max\left(1, \left\lceil \frac{\text{len}(\text{trimmed\_text})}{\text{CHARS\_PER\_TOKEN\_HEURISTIC}} \right\rceil\right)$$

where `CHARS_PER_TOKEN_HEURISTIC` defaults to `4.0` in configuration.

### Output Budget & Native Ollama Telemetry (Phase 4.1)
Argus captures native runtime measurements directly from Ollama without simulation:
1. **`num_predict_limit`**: The allocated generation token ceiling.
2. **`eval_count`**: The exact number of tokens generated by the model.
3. **`done_reason`**: 
   - `"stop"`: The model finished generating naturally via EOS token.
   - `"length"`: Generation halted because it reached the `num_predict` ceiling.
4. **`budget_utilization_percent`**: $\frac{\text{eval\_count}}{\text{allocated\_budget}} \times 100\%$.
5. **`hit_budget_limit`**: True if generation was truncated by the budget limit.
6. **Zero-Fabrication Standard**: If Ollama or test mocks do not provide native metrics, fields remain `null`.
