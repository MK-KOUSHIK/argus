# Argus Developer Guide

This guide details local environment setup, configuration, API references, testing protocols, and troubleshooting steps for the Argus prompt optimization platform.

---

## 1. Prerequisites

Ensure the following tools are installed on your workstation:
- **Node.js**: `v20.x` or `v22.x LTS` (includes `npm`)
- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: `v2.40+`
- **Ollama**: Installed locally (required for local neural inference with `qwen2.5:1.5b`)

---

## 2. Ollama Local LLM Setup

Argus utilizes Ollama as its local neural inference runtime to rewrite prompts and prune redundant tokens.

### 2.1 Verify or Start Ollama
1. **Check Ollama version**:
   ```bash
   ollama --version
   ```

2. **Start the Ollama daemon** (if not already running as a background service):
   ```bash
   ollama serve
   ```

3. **Pull the required model**:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

4. **Verify model presence**:
   ```bash
   ollama list
   ```
   Confirm that `qwen2.5:1.5b` appears in the list.

5. **Verify Ollama HTTP service**:
   Ensure Ollama's local HTTP API responds at `http://127.0.0.1:11434`:
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get
   ```

---

## 3. Quickstart: Local Environment Setup

### 3.1 Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a Python virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS (Bash)**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**:
   ```powershell
   cp .env.example .env
   ```

5. **Start the FastAPI development server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Verify the backend**:
   - Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
   - Ollama status check: [http://localhost:8000/api/ollama/status](http://localhost:8000/api/ollama/status)
   - Interactive OpenAPI documentation: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

---

### 3.2 Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   ```powershell
   cp .env.example .env
   ```

4. **Start Vite development server**:
   ```bash
   npm run dev
   ```

5. **Access the application**:
   Open [http://localhost:5173](http://localhost:5173) in your browser. The Header indicates real-time backend and Ollama connectivity.

---

## 4. Environment Variables Reference

Backend settings are managed through `pydantic-settings` in [`backend/app/core/config.py`](file:///d:/Argus/backend/app/core/config.py) and loaded from `backend/.env`.

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `APP_NAME` | `str` | `Argus` | Application name displayed in health check and logs. |
| `ENVIRONMENT` | `str` | `development` | Deployment environment (`development`, `staging`, `production`). |
| `DEBUG` | `bool` | `True` | Enables interactive OpenAPI docs (`/api/docs`) and debug responses. |
| `API_PREFIX` | `str` | `/api` | Root path prefix for all API routes. |
| `HOST` | `str` | `0.0.0.0` | Network interface to bind FastAPI. |
| `PORT` | `int` | `8000` | Port for FastAPI application. |
| `CORS_ORIGINS` | `list[str]` | `["http://localhost:5173", "http://127.0.0.1:5173"]` | Allowed CORS origins for browser client access. |
| `OLLAMA_BASE_URL` | `str` | `http://127.0.0.1:11434` | HTTP address of the local Ollama daemon. |
| `OLLAMA_MODEL` | `str` | `qwen2.5:1.5b` | Ollama model identifier used for prompt optimization. |
| `OLLAMA_TIMEOUT_SECONDS` | `float` | `60.0` | Maximum timeout for local neural inference generation. |
| `OLLAMA_STATUS_TIMEOUT_SECONDS` | `float` | `4.0` | Fast timeout for health checking Ollama availability. |
| `OLLAMA_TEMPERATURE` | `float` | `0.2` | Generation temperature for deterministic rewriting. |
| `OLLAMA_TOP_P` | `float` | `0.9` | Generation top-p sampling parameter. |
| `CHARS_PER_TOKEN_HEURISTIC` | `float` | `4.0` | Standard character-to-token heuristic ratio (~4 chars/token). |
| `MAX_VALIDATION_RETRY_ATTEMPTS` | `int` | `1` | Number of corrective retry attempts when candidate validation fails. |

---

## 5. API Reference

All routes are mounted under the `/api` prefix via [`backend/app/api/router.py`](file:///d:/Argus/backend/app/api/router.py).

### 5.1 System Health (`GET /api/health`)
Returns the operational status, application version, environment, and current UTC timestamp.

- **Status Code**: `200 OK`
- **Response Example**:
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

### 5.2 Ollama Service Status (`GET /api/ollama/status`)
Inspects whether the local Ollama daemon is reachable and confirms that the required optimization model is installed.

- **Status Code**: `200 OK`
- **Response Example (Connected & Ready)**:
  ```json
  {
    "connected": true,
    "base_url": "http://127.0.0.1:11434",
    "model": "qwen2.5:1.5b",
    "model_installed": true,
    "available_models": [
      "qwen2.5:1.5b",
      "qwen2.5:7b"
    ],
    "error": null
  }
  ```
- **Response Example (Ollama Offline)**:
  ```json
  {
    "connected": false,
    "base_url": "http://127.0.0.1:11434",
    "model": "qwen2.5:1.5b",
    "model_installed": false,
    "available_models": [],
    "error": "Cannot connect to Ollama at http://127.0.0.1:11434. Ensure Ollama is running ('ollama serve')."
  }
  ```

---

### 5.3 Prompt Optimization (`POST /api/optimize`)
Optimizes an input prompt using the 4-stage modular pipeline (`TaskAnalyzer` &rarr; `StrategySelector` &rarr; `PromptRewriter` &rarr; `CandidateValidator`).

- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "prompt": "Write a python function to compute fibonacci numbers",
    "mode": "balanced",
    "target_model": "qwen2.5:1.5b"
  }
  ```
  - `prompt` (`str`, required): The raw prompt text. Must contain at least one non-whitespace character.
  - `mode` (`str`, optional): Optimization profile. Allowed values: `"balanced"`, `"concise"`, `"structure"`. Defaults to `"balanced"`.
  - `target_model` (`str`, optional): Target destination model profile. Defaults to `"gpt-4o"`.

- **Response Body (`200 OK`)**:
  ```json
  {
    "original_prompt": "Write a python function to compute fibonacci numbers",
    "optimized_prompt": "Write a Python function to calculate Fibonacci numbers.",
    "mode": "balanced",
    "target_model": "qwen2.5:1.5b",
    "task_category": "coding",
    "strategy": "clarify",
    "applied_strategy": "clarify",
    "outcome": "validated_rewrite",
    "length_result": "expanded",
    "validation_status": "passed",
    "is_preserved": false,
    "primary_action": "write",
    "primary_deliverable": "function",
    "explicit_constraints": [
      "Deliverable: Function",
      "Operation: Save python function",
      "Operation: Calculate fibonacci numbers"
    ],
    "required_operations": [
      "save python function",
      "calculate fibonacci numbers"
    ],
    "original_metrics": {
      "char_count": 52,
      "estimated_tokens": 13
    },
    "optimized_metrics": {
      "char_count": 55,
      "estimated_tokens": 14
    },
    "token_reduction_percent": 0,
    "token_change_percent": 8,
    "token_savings": 0,
    "char_delta": 3,
    "preserved_constraints": [
      "Core query intent & technical details"
    ],
    "execution_time_ms": 4089,
    "model": "qwen2.5:1.5b",
    "timestamp": "2026-10-31T04:22:45.643316+00:00"
  }
  ```

- **Important Note on Token Estimation**:
  Token counts in `original_metrics` and `optimized_metrics` are **estimates** calculated using a standard character heuristic ($\text{estimated\_tokens} = \lceil \text{length} / 4.0 \rceil$). They provide an immediate local indicator of text volume changes and are not exact tokenizer counts.

- **Error Responses**:
  - `422 Unprocessable Content`: Prompt is empty, whitespace-only, or JSON payload is malformed.
  - `502 Bad Gateway`: Ollama returned an invalid or unparseable response.
  - `503 Service Unavailable`: Local Ollama service is unreachable (`ollama serve` is not running).
  - `504 Gateway Timeout`: Local inference took longer than `OLLAMA_TIMEOUT_SECONDS`.

---

## 6. Running Automated Tests & Builds

### 6.1 Backend Tests (Pytest)
Backend tests run without requiring a live Ollama instance because HTTP calls are mocked in the test fixtures.
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```
*Expected result: 36 passed.*

### 6.2 Frontend Tests (Vitest)
Unit and integration tests for React components, Workspace interactions, and state management.
```bash
cd frontend
npm.cmd test -- --run
```
*Expected result: 11 passed across 2 test files.*

### 6.3 Frontend Production Build
Validates TypeScript compilation (`tsc -b`) and bundles production assets with Vite.
```bash
cd frontend
npm.cmd run build
```
*Expected result: Successful build output in `dist/` with zero TypeScript errors.*

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 503 "Cannot connect to Ollama"** | The Ollama server daemon is not running. | Run `ollama serve` in a terminal or start the Ollama desktop app. |
| **HTTP 503 "Model 'qwen2.5:1.5b' not found"** | The default optimization model has not been downloaded yet. | Run `ollama pull qwen2.5:1.5b` in terminal. |
| **HTTP 504 "Ollama request timed out"** | The local host is under high CPU/GPU load, or generation took longer than 60 seconds. | Increase `OLLAMA_TIMEOUT_SECONDS=120.0` in `backend/.env`. |
| **HTTP 422 "Validation error"** | Request body is missing the `prompt` field, or the prompt contains only whitespace. | Submit a valid JSON payload containing a non-empty `prompt` string. |
| **Frontend displays "Offline" status** | FastAPI backend is not running on port 8000. | Start the backend via `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`. |
| **CORS error in browser console** | Frontend origin is not listed in backend `CORS_ORIGINS`. | Ensure `backend/.env` contains `CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]`. |

---

## 8. Coding Standards & Architectural Guardrails

1. **Separation of Concerns**:
   - The React frontend does not contain prompt compression heuristics or model generation calls.
   - The FastAPI API router layer does not perform raw business logic; it delegates to the services layer.
   - All environment variables are centralized in `backend/app/core/config.py`.

2. **Zero-Hardcoding Standard**:
   - Extraction patterns and validation rules use generalized taxonomies rather than hardcoded prompt text or test-specific strings.

3. **Git Hygiene**:
   - Never commit `.env` files, model weights, virtual environments (`.venv`), or build artifacts (`dist/`).
