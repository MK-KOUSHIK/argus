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

### 2.1 Starting Ollama with Optimized KV-Cache & Flash Attention

For reproducible development with **Q8_0 KV-cache quantization** and **Flash Attention** on Windows, use the dedicated launcher script:

```powershell
# Start Ollama with Q8_0 KV cache and Flash Attention:
.\scripts\launch_ollama.ps1 -Action start -KvType q8_0 -FlashAttention

# Check running status and port bindings:
.\scripts\launch_ollama.ps1 -Action status

# Stop Ollama:
.\scripts\launch_ollama.ps1 -Action stop
```

Alternatively, to start manually:
```powershell
$env:OLLAMA_FLASH_ATTENTION="1"
$env:OLLAMA_KV_CACHE_TYPE="q8_0"
$env:OLLAMA_DEBUG="1"
ollama serve
```

### 2.2 Verify or Pull Model
1. **Check Ollama version**:
   ```bash
   ollama --version
   ```

2. **Pull the required model**:
   ```bash
   ollama pull qwen2.5:1.5b
   ```

3. **Verify model presence**:
   ```bash
   ollama list
   ```

---

## 3. Local Installation & Development

### 3.1 Backend Setup (FastAPI)
1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```powershell
   cp .env.example .env
   ```

5. **Start FastAPI development server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### 3.2 Frontend Setup (React 19 + Vite)
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
   Open [http://localhost:5173](http://localhost:5173) in your browser.

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
| `OLLAMA_NUM_CTX` | `int` | `4096` | Baseline per-request context window size. |
| `OLLAMA_MAX_NUM_CTX` | `int` | `8192` | Hard upper bound for context tier derivation. |
| `OLLAMA_NUM_PREDICT` | `int` | `512` | Fallback generation token limit. |
| `OLLAMA_CONTEXT_RESERVE_TOKENS` | `int` | `256` | Context headroom reserved beyond input and output. |
| `OUTPUT_BUDGET_ENABLED` | `bool` | `True` | Enable intelligent dynamic output budget control (Phase 4.1). |
| `OUTPUT_BUDGET_POLICY` | `str` | `adaptive` | Budget policy (`adaptive`, `fixed`, `conservative`). |
| `OUTPUT_BUDGET_MIN_TOKENS` | `int` | `64` | Minimum allowable generation tokens. |
| `OUTPUT_BUDGET_MAX_TOKENS` | `int` | `2048` | Maximum allowable generation tokens. |
| `OUTPUT_BUDGET_SAFETY_HEADROOM`| `int` | `64` | Safety padding added to word/sentence constraint estimates. |

---

## 5. API Reference

### 5.1 Prompt Optimization (`POST /api/optimize`)
Optimizes an input prompt using the modular pipeline with Output Budget Control:

- **Request Body**:
  ```json
  {
    "prompt": "Summarize the key differences between synchronous and asynchronous programming in Python in under 50 words.",
    "mode": "balanced",
    "target_model": "gpt-4o",
    "requested_output_budget": null
  }
  ```
  - `prompt` (`str`, required): The raw prompt text.
  - `mode` (`str`, optional): Optimization profile (`"balanced"`, `"concise"`, `"structure"`).
  - `target_model` (`str`, optional): Target destination model profile.
  - `requested_output_budget` (`int`, optional): Explicit output token override ($1 \le x \le 8192$).

- **Response Body (`200 OK`)**:
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
    "execution_time_ms": 186,
    "model": "qwen2.5:1.5b",
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

---

## 6. Running Automated Tests & Builds

### 6.1 Backend Tests (Pytest)
```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```
*Expected result: 79 passed across 4 test files (`test_health.py`, `test_kv_cache_adaptive.py`, `test_optimizer.py`, `test_output_budget.py`).*

### 6.2 Frontend Tests (Vitest)
```bash
cd frontend
npm.cmd test -- --run
```
*Expected result: 19 passed across 4 test files.*

### 6.3 Frontend Production Build
```bash
cd frontend
npm.cmd run build
```
*Expected result: Successful build output in `dist/` with zero TypeScript errors.*

---

## 7. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 503 "Cannot connect to Ollama"** | The Ollama server daemon is not running. | Run `powershell .\scripts\launch_ollama.ps1 -Action start` or `ollama serve`. |
| **HTTP 503 "Model 'qwen2.5:1.5b' not found"** | The default optimization model has not been downloaded yet. | Run `ollama pull qwen2.5:1.5b` in terminal. |
| **HTTP 413 "Prompt footprint exceeds context"** | Available context headroom is insufficient for minimal generation (<64 tokens). | Increase `OLLAMA_MAX_NUM_CTX` or shorten prompt. |
| **HTTP 504 "Ollama request timed out"** | The local host is under high CPU/GPU load, or generation took longer than 60 seconds. | Increase `OLLAMA_TIMEOUT_SECONDS=120.0` in `backend/.env`. |
| **HTTP 422 "Validation error"** | Request body is missing the `prompt` field, or the prompt contains only whitespace. | Submit a valid JSON payload containing a non-empty `prompt` string. |
