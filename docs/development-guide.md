# Argus Developer Guide

This guide details the local setup, development conventions, testing protocols, and verification steps for developing Argus.

---

## 1. Prerequisites

Ensure the following tools are installed on your machine:
- **Node.js**: v20+ or v22 LTS (includes `npm`)
- **Python**: v3.10 to v3.13
- **Git**: v2.40+
- **Ollama**: Installed and running (Required for local model inference with `qwen2.5:1.5b`)

---

## 2. Ollama Local LLM Setup (Phase 3)

Argus utilizes Ollama as its local neural inference runtime to rewrite prompts and prune redundant tokens.

### 2.1 Verify or Start Ollama
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
   Ensure `qwen2.5:1.5b` appears in the list.

4. **Verify Ollama HTTP service**:
   Ensure Ollama's local HTTP API is accessible at `http://127.0.0.1:11434`:
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -Method Get
   ```
   If Ollama is not running in the background, start it:
   ```bash
   ollama serve
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
   ```bash
   cp .env.example .env
   ```
   Ensure `OLLAMA_BASE_URL=http://127.0.0.1:11434` and `OLLAMA_MODEL=qwen2.5:1.5b` are present in `.env`.

5. **Start the development server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Or use the convenience script from the root:
   ```powershell
   .\scripts\dev-backend.ps1
   ```

6. **Verify the backend & Ollama status**:
   - Health check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
   - Ollama connection status: [http://localhost:8000/api/ollama/status](http://localhost:8000/api/ollama/status)
   - Interactive OpenAPI docs: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

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
   ```bash
   cp .env.example .env
   ```

4. **Start Vite development server**:
   ```bash
   npm run dev
   ```
   Or use the convenience script from the root:
   ```powershell
   .\scripts\dev-frontend.ps1
   ```

5. **Access the application**:
   Open [http://localhost:5173](http://localhost:5173) in your browser. The Header will display `Local LLM: Connected (qwen2.5:1.5b)` when Ollama is running.

---

## 4. Prompt Optimization API Reference

### `POST /api/optimize`
Rewrites prompt using local Ollama model to compress input tokens.

**Request Body (`application/json`)**:
```json
{
  "prompt": "Please review this Python code. Make sure there are no race conditions.",
  "mode": "balanced",
  "target_model": "gpt-4o"
}
```
- `mode` options: `"balanced"`, `"concise"`, `"structure"`
- `target_model`: Intended destination frontier model (illustrative / Phase 4 gateway)

**Response Body (`200 OK`)**:
```json
{
  "original_prompt": "Please review this Python code. Make sure there are no race conditions.",
  "optimized_prompt": "Review Python code. Ensure zero race conditions.",
  "mode": "balanced",
  "target_model": "gpt-4o",
  "original_metrics": { "char_count": 78, "estimated_tokens": 20 },
  "optimized_metrics": { "char_count": 48, "estimated_tokens": 12 },
  "token_reduction_percent": 40,
  "token_savings": 8,
  "preserved_constraints": ["Core query intent & technical details"],
  "execution_time_ms": 1420,
  "model": "qwen2.5:1.5b",
  "timestamp": "2026-10-16T17:10:00Z"
}
```

### `GET /api/ollama/status`
Returns Ollama runtime connectivity and model availability.

---

## 5. Running Automated Tests

### 5.1 Backend Tests
Run the pytest test suite inside the `backend/` directory (Ollama HTTP calls are mocked in automated tests so running Ollama is not required for unit testing):
```bash
cd backend
pytest -v
```

### 5.2 Frontend Tests
Run Vitest unit and smoke tests in the `frontend/` directory:
```bash
cd frontend
npm run test
```

### 5.3 Frontend Production Build Validation
Verify TypeScript type-checking and bundling:
```bash
cd frontend
npm run build
```

---

## 6. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **HTTP 503 "Cannot connect to Ollama"** | Ollama daemon is stopped | Run `ollama serve` or ensure the Ollama desktop app is running in system tray. |
| **HTTP 503 "Model 'qwen2.5:1.5b' not found"** | Model has not been pulled yet | Run `ollama pull qwen2.5:1.5b` in terminal. |
| **HTTP 504 "Inference timed out"** | Local machine under heavy GPU/CPU load | Increase `OLLAMA_TIMEOUT_SECONDS=120.0` in `backend/.env`. |
| **HTTP 422 "Validation error"** | Prompt was empty or whitespace only | Provide non-empty prompt text. |

---

## 7. Coding Standards & Architectural Guardrails

### 7.1 Separation of Concerns
- **Frontend**: Presentational and interactive components only. UI components must consume backend services via `src/services/api.ts` or custom hooks. No prompt optimization logic belongs in React.
- **Backend API Layer (`app/api/`)**: Translates HTTP requests to schema objects and calls service methods. Does not perform raw business logic.
- **Backend Schema Layer (`app/schemas/`)**: Declares Pydantic models for validation and serialization.
- **Backend Service Layer (`app/services/`)**: Orchestrates business logic, model calls to Ollama, and data processing.
- **Backend Core Layer (`app/core/`)**: Houses configuration, secrets management, and cross-cutting utilities.

### 7.2 Git & Commit Hygiene
- Never commit `.env`, model weights (`.gguf`, `.safetensors`), virtual environments (`.venv`), or build artifacts (`dist/`).
- Verify CI passes before merging pull requests.
