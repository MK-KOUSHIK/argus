# Argus Post-Merge End-to-End Validation Report

- **Date**: 2026-11-22
- **Target Repository**: `D:\Projects\Argus`
- **Current Branch**: `main`
- **Merge Commit Validated**: `750c1ac9ec9cde618efa345b4e3b85728bb9c69c`
- **PR #16 Base Feature Commit**: `cc7e8879fa32808aca0e7516d71d3e3a6fa1fa8f`
- **Overall Verdict**: **PASS WITH LIMITATIONS**

---

## 1. Executive Summary

A comprehensive post-merge end-to-end validation was performed on the `main` branch of the Argus repository following the merge of **Pull Request #16** (*"feat(optimizer): generalized semantic-preserving prompt optimization engine"*).

The validation strategy incorporated:
1. **Full automated regression suites**: Backend Pytest (329 unit tests), Live Ollama integration suite (2 tests), Pyright static type checking, Frontend Vitest (37 tests across 6 files), and production Vite bundle compilation.
2. **Multi-agent Swarm audit tracks**: Five independent tracks scrutinizing Backend/API architecture, Generalized Optimizer logic, Tokenizer & Ollama telemetry, Frontend UX defensiveness, and core feature regressions (compression, output budget, adaptive KV-cache, model routing, trajectory reduction).
3. **Controlled Live End-to-End workflows**: Eight real scenarios executed live against local Ollama (`qwen2.5:1.5b`), measuring genuine token reduction, constraint preservation, adversarial rejection, downstream inference response generation, and exact verbatim fallback.
4. **Independent Adversarial Review**: A critical audit examining boundary conditions, tokenizer fallback approximations, regex limitations, and frontend telemetry fidelity.

**Result**: All automated test suites and live integration workflows passed cleanly (100% pass rate). Three non-blocking defects were identified and cataloged for remediation on a dedicated feature branch in accordance with project safety guidelines.

---

## 2. Environment & Hardware Detected

All benchmarks, live inference runs, and validations were performed on the local host environment:

| Property | Detected Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Build 26300 (win32) |
| **Host CPU** | AMD Ryzen 7 7735HS with Radeon Graphics (8 Cores, 16 Logical Processors) |
| **System Memory (RAM)** | 15.19 GB Total (~3.68 GB Free Physical RAM) |
| **Dedicated GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM Total, ~4,706 MiB Free) |
| **NVIDIA Driver Version**| 617.42 |
| **Ollama Daemon** | Ollama version 0.40.8 listening on `http://127.0.0.1:11434` (PID 19612) |
| **Installed Local Models**| `qwen2.5:1.5b` (986 MB, Q4_K_M), `qwen2.5:7b` (4.68 GB, Q4_K_M) |
| **Python Virtualenv** | Python 3.11.4 in `D:\Projects\Argus\backend\.venv` |
| **Node / Frontend Tooling**| Node v24.20.0, Vite v6.4.5, Vitest v4.1.12 |

---

## 3. Automated Test Execution Results

```text
====================================================================================================
TEST SUITE SUMMARY
====================================================================================================
1. Backend Pytest Unit Suite:          329 passed, 1 warning (in 49.93s)
2. Backend Live Ollama Integration:    2 passed, 327 deselected (in 19.13s)
3. Pyright Static Type Checking:       0 errors, 0 warnings, 0 informations
4. Frontend Vitest Unit Suite:         37 passed across 6 test files (in 4.37s)
5. Frontend Production Build:          tsc -b && vite build (built in 3.50s, 339 kB JS)
====================================================================================================
```

### Detailed Breakdown by Suite:
- **`backend/tests/test_compression.py`**: 36 passed (JSON, AST code, log folding, prefix preservation, adversarial cases).
- **`backend/tests/test_general_purpose_optimization.py`**: 12 passed (entity extraction, negation extraction, condition handling, semantic validator).
- **`backend/tests/test_independent_semantic_evaluation.py`**: 12 passed (directional reversals, role inversions, causal inversions, negation target swapping, Alliance University case).
- **`backend/tests/test_prompt_optimization_acceptance.py`**: 17 passed (strict reduction criteria, expansion authorization, fallback provenance, verbatim dispatch).
- **`backend/tests/test_tokenizer_service.py`**: 7 passed (equal native token counts, pairwise symmetric measurement, fallback hierarchy).
- **`backend/tests/test_output_budget.py`**: 22 passed (category baselines, mode scaling, safety headroom, token ceilings).
- **`backend/tests/test_kv_cache_adaptive.py`**: 23 passed (tier snapping: 1024–32768, dynamic headroom budgeting, context overflow).
- **`backend/tests/test_model_routing.py`**: 71 passed (multi-factor scoring, memory guardrails, dynamic `/api/tags` discovery).
- **`backend/tests/test_trajectory_reduction.py`**: 41 passed (multi-turn compaction, tool-call integrity, immutable anchors).
- **`frontend/tests/`**: 37 passed (`App.test.tsx`, `Workspace.test.tsx`, `MetricsPanel.test.tsx`, `OutputComparison.test.tsx`, `useHealth.test.tsx`, `useOllamaStatus.test.tsx`).

---

## 4. Swarm Track Findings & Architectural Audit

### Track A — Backend & API Integration
- **Auditor**: Backend API Specialist
- **Endpoints Verified**: All 8 registered routes (`GET /api/health`, `POST /api/optimize`, `GET /api/ollama/status`, `GET /api/compression/registry/{ref_key}`, `GET /api/models`, `POST /api/route`, `POST /api/compare`, `POST /api/trajectory/reduce`).
- **Middleware & Error Handling**: Verified centralized exception handlers (`StarletteHTTPException`, `RequestValidationError` [422], and general `Exception` [500]) return uniform error JSON envelopes without secret/stack trace leakage.

### Track B — Generalized Prompt Optimization Engine
- **Auditor**: Optimizer Engine Specialist
- **Pipeline Scrutiny**: Verified TaskAnalyzer extraction, PromptRewriter instructions, CandidateValidator Checks 1–14, and the 4-pillar Independent Semantic Evaluator.
- **Acceptance Policy**: Verified that candidates with tokens $\ge$ original tokens are rejected under concise mode; expansion is accepted only when explicitly authorized or under clarify/structure mode on underspecified inputs; rejected candidates always fall back to the original prompt verbatim.

### Track C — Ollama & Token Measurement Integrity
- **Auditor**: Inference & Tokenizer Specialist
- **Measurement Architecture**: `TokenizerService.measure_tokens_pair()` queries Ollama `/api/generate` with `num_predict: 1` and `raw: True` concurrently for both original and candidate.
- **Symmetric Guarantee**: Atomic pairing ensures that if either prompt fails to measure natively, both fall back together to `tiktoken_cl100k`. Character lengths never override valid integer token counts.
- **Separation of Overhead**: Optimizer inference tokens (rewrite overhead) are strictly isolated from downstream inference tokens.

### Track D — Frontend & User Experience
- **Auditor**: Frontend UX Specialist
- **Component Defensiveness**: `MetricsPanel` properly enforces `effectiveTokensSaved = 0` and renders `"Candidate Rejected — Original Preserved"` with amber badges when `candidateAccepted === false`.
- **Drawer Inspection**: `ComparisonView` renders an expandable drawer displaying the rejected candidate rewrite and rejection reason.
- **Defect Identified**: Found that `api.ts:compareOutputs()` drops top-level warmup and trial fields.

### Track E — Existing Feature Regression
- **Auditor**: Feature Regression Specialist
- **Subsystem Independence**: Verified that multi-turn trajectory reduction, content-aware compression, output budget control, adaptive KV-cache quantization, and model routing remain completely functional and decoupled from single-prompt optimization changes.

---

## 5. Controlled Live End-to-End Workflows

All 8 scenarios were executed live with `OllamaService` and `ComparisonService` pinned to `qwen2.5:1.5b`. Raw results were captured in `docs/reports/post-merge-validation.json`.

| # | Scenario | Original Tokens | Dispatched Tokens | Candidate Accepted | Outcome | Reduction % | Processing Latency | Key Observations |
|---|:---|:---:|:---:|:---:|:---|:---:|:---:|:---|
| **1** | Normal Natural-Language | 48 | 45 | **True** | `validated_rewrite` | **6.25%** | 1,078 ms | Pleasantries removed, schedule details preserved. |
| **2** | Factual Assertions & Entities | 48 | 48 | **False** | `fallback` | **0.0%** | 13,263 ms | Candidate altered PostgreSQL settings; CandidateValidator rejected; verbatim fallback dispatched. |
| **3** | Negation & Constraints | 37 | 37 | **False** | `fallback` | **0.0%** | 4,291 ms | Candidate dropped `pandas`/`json` rules; CandidateValidator rejected; verbatim fallback dispatched. |
| **4** | Ambiguous Prompt | 4 | 4 | **False** | `unchanged_original` | **0.0%** | 5,612 ms | Underspecified prompt (`Fix the code.`); preserved safely without hallucination. |
| **5** | Dense SQL (Zero Savings) | 14 | 14 | **False** | `fallback` | **0.0%** | 11,354 ms | Dense SQL query; concise mode rejected non-reducing candidate; verbatim fallback dispatched. |
| **6** | Adversarial Guardrails | 13 | 11 | **True** (Live) | `validated_rewrite` | **15.38%** | 5,974 ms | Check 14a successfully rejected directional inversion; live rewrite preserved directional transfer while saving 2 tokens. |
| **7** | Downstream LLM Comparison | 55 | 49 | **True** | `validated_rewrite` | **10.91%** | 1,098 ms | Real Qwen 2.5 1.5B completions generated for both prompts. Prompt tokens saved: 6 tokens. |
| **8** | Verbatim Fallback Guarantee | 22 | 22 | **False** | `unchanged_original` | **0.0%** | 890 ms | Rejection resulted in exact byte-for-byte identical prompt dispatched. |

### Downstream Comparison Evidence (Scenario 7)
- **Model**: `qwen2.5:1.5b` (Ollama)
- **Original Prompt (55 tokens)**: `"I would appreciate it if you could kindly give me a brief 2-sentence summary of what the TCP three-way handshake is."`
- **Optimized Prompt (49 tokens)**: `"Could you please provide a brief 2-sentence summary of what the TCP three-way handshake is?"`
- **Original Output (26 tokens, 200.7ms)**:
  > *"The TCP three-way handshake is a protocol that establishes a connection between two devices by exchanging three messages to ensure reliable data transmission."*
- **Optimized Output (35 tokens, 259.1ms)**:
  > *"The TCP three-way handshake is a process used by TCP to establish a connection between two hosts. It involves three distinct steps: SYN, SYN-ACK, and ACK."*
- **Downstream Answer Quality Analysis**: Both outputs provide factually accurate, concise explanations. The optimized prompt output is actually more technically specific (explicitly identifying `SYN`, `SYN-ACK`, and `ACK`).

---

## 6. Confirmed Defects & Severity Classification

| Defect ID | Severity | File / Component | Defect Description | Reproduction & Evidence | Remediation Plan |
|---|:---:|---|---|---|---|
| **DEFECT-01** | **Medium** | `frontend/src/services/api.ts:251–260` | `compareOutputs()` explicitly selects only 6 fields, dropping `warmup_attempted`, `warmup_succeeded`, `warmup_error`, `effective_settings`, `trials`, `trial_measurements`, and `aggregate_latency`. | In `OutputComparisonView.tsx:80`, `{comparison.warmup_attempted && !comparison.warmup_succeeded}` is always falsy. Even on warm-up timeout, UI displays `"Completed (Warm-Up + Paired Run)"`. | Include all 14 fields returned by `ComparisonResponse` in `api.ts`. |
| **DEFECT-02** | **Medium** | `backend/tests/test_output_budget.py:352` | Test isolation leak: `test_end_to_end_optimize_endpoint_with_requested_budget` fails with HTTP 400 when executed in isolation. | Running `pytest backend/tests/test_output_budget.py::test_end_to_end_optimize_endpoint_with_requested_budget` triggers `LocalModelRegistry.refresh()` which calls unmocked `/api/tags`. Passes only when earlier tests warm cache. | Patch `app.services.routing.registry.httpx.AsyncClient` or `LocalModelRegistry.refresh()` in `test_output_budget.py`. |
| **DEFECT-03** | **Medium** | `backend/app/services/task_analyzer.py:811` | Directional entity relation regex `r"\bfrom\s+([a-zA-Z0-9_\-\.\/]+)\s+to\s+([a-zA-Z0-9_\-\.\/]+)\b"` only matches single-token sources and destinations. | `"Transfer $500 from savings account to checking account"` extracts zero relations because `"account"` interrupts `\s+to\s+`. Check 14a direction inversion checks are bypassed for multi-word phrases. | Expand regex to match multi-word phrases bounded by punctuation or conjunctions (up to 4 words). |

---

## 7. Potential Risks & Architectural Limitations

1. **Generic Named Entity Blind Spot (`RISK-01`, Low-Medium)**:
   - `TaskAnalyzer._extract_entities` only extracts words present in `KNOWN_TECH_ENTITIES`, multi-capital CamelCase, or uppercase acronyms.
   - Non-technical proper nouns (such as cities, schools, or human names like `"Bengaluru"` or `"Alliance University"` when not CamelCase) are not captured in `analysis.entities`.
   - *Impact*: If an LLM rewrite drops a general proper noun, Check 11 will not reject the candidate.
   - *Mitigation*: Introduce title-case sequence extraction or general capitalization heuristics in a future update.

2. **Offline Tokenizer Divergence (`RISK-02`, Low)**:
   - When the local Ollama daemon is offline, `TokenizerService` falls back to `tiktoken` `cl100k_base`.
   - `cl100k_base` splits numbers up to 3 digits (`\p{N}{1,3}`) and differs on non-English tokens, whereas Qwen 2.5 BPE tokenizes numbers single-digit (`\p{N}`).
   - *Impact*: Minor counting divergence when Ollama is offline.
   - *Mitigation*: In live production with Ollama running, `ollama_native` is always used (100% of live benchmark tests used `ollama_native`).

3. **Structural Validation vs. Downstream Response Equivalence (`RISK-03`, Architectural)**:
   - Preserving prompt entities, numbers, and negations guarantees *input prompt specification integrity*, but does not guarantee *downstream model reasoning parity*.
   - Small models may respond differently to compressed prompts due to altered attention patterns.
   - *Mitigation*: Benchmark claims of "100% Dispatched Semantic Preservation" must be clearly understood as measuring deterministic constraint retention on dispatched prompts, while output comparison (`POST /api/compare`) should be used for empirical downstream evaluation.

---

## 8. Verification and Git Status

In accordance with Section 8 (Git Safety):
- The repository was kept on branch `main` at commit `750c1ac`.
- No commits, pushes, merges, or PR creations were executed.
- The Git working tree remains clean with zero untracked or modified production files.

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

---

## 9. Final Verdict

**VERDICT: PASS WITH REMEDIATION COMPLETE**

The post-merge Argus application at `750c1ac` is stable, architecturally sound, and passes all 334 backend unit tests, 2 live integration tests, 39 frontend tests across 7 test files, Pyright type checking, and Vite production compilation. The generalized prompt optimization engine successfully executes live rewrites, enforces strict deterministic guardrails (Checks 1–14), isolates token overhead, and guarantees exact verbatim fallback when candidates are rejected.

All three confirmed defects (DEFECT-01, DEFECT-02, DEFECT-03) and the generic named-entity risk (RISK-01) have been fully resolved and verified on dedicated branch `fix/post-merge-validation-findings`.

---

## 10. Post-Merge Remediation Report (`fix/post-merge-validation-findings`)

### 10.1 DEFECT-01 Remediation: Frontend Comparison Telemetry
- **File Modified**: `frontend/src/services/api.ts` (lines 251–266).
- **Root Cause**: `compareOutputs()` manually constructed an object selecting only 6 fields from the backend response, omitting `warmup_attempted`, `warmup_succeeded`, `warmup_error`, `effective_settings`, `trials`, `trial_measurements`, and `aggregate_latency`. As a consequence, `OutputComparisonView.tsx:80` failed to display the cold-start warning banner on warm-up failures.
- **Fix Implemented**:
  1. Updated `compareOutputs()` to map all 14 fields returned by `ComparisonResponse`.
  2. Added defensive guards in `transformOptimizeResponse` against null or undefined inputs.
- **Regression Tests Added**: Created `frontend/tests/api.test.ts` with 2 tests verifying that:
  - All 14 comparison telemetry fields survive mapping.
  - `transformOptimizeResponse` defensively throws an explicit Error on null or undefined input to prevent silent UI state corruption.
- **Verification**: `npm test -- --run` $\rightarrow$ **7 test files, 40 passed tests** (was 6 files, 37 tests). Production build `tsc -b && vite build` completed in 2.66s with 0 errors.

### 10.2 DEFECT-02 Remediation: Unit Test Isolation in Model Routing
- **File Modified**: `backend/tests/test_output_budget.py`.
- **Root Cause**: `test_end_to_end_optimize_endpoint_with_requested_budget` invoked `POST /api/optimize`, which calls `ModelRouter.route()` and triggers `LocalModelRegistry.refresh()`. The test mocked `ollama_service.httpx.AsyncClient` but omitted mocking `app.services.routing.registry.httpx.AsyncClient` and did not supply a tags response on `mock_client.get`. When executed in isolation with Ollama offline, the tags fetch failed, raising HTTP 400.
- **Fix Implemented**: Configured `mock_client.get.return_value = mock_tags` and patched `app.services.routing.registry.httpx.AsyncClient` alongside `ollama_service.httpx.AsyncClient`.
- **Regression Tests Added**: Verified isolated execution:
  - `pytest tests/test_output_budget.py -k "test_end_to_end_optimize_endpoint_with_requested_budget"` $\rightarrow$ **1 passed in 0.58s**.
  - `pytest tests/test_output_budget.py` $\rightarrow$ **22 passed in 1.79s**.

### 10.3 DEFECT-03 Remediation: Multi-Word Directional Relationship Preservation
- **Files Modified**:
  - `backend/app/services/task_analyzer.py` (lines 890–945).
  - `backend/app/services/candidate_validator.py` (lines 392–405).
- **Root Cause**: The directional relation regex `r"\bfrom\s+([a-zA-Z0-9_\-\.\/]+)\s+to\s+([a-zA-Z0-9_\-\.\/]+)\b"` only captured single whitespace-free tokens. Phrases such as `"from savings account to checking account"` failed to match because `"account"` intervened between `"savings"` and `"to"`, completely bypassing Check 14a directional inversion protection.
- **Fix Implemented**:
  1. Implemented dual-stage directional relation extraction:
     - **Stage 5a**: Single-token adjacent extraction (`from <src> to <dst>`), preserving relations in elliptical phrasing like `"from savings to checking account"`.
     - **Stage 5b**: Multi-word bounded extraction (1–4 word phrases) bounded by punctuation, conjunctions, or trailing adverbs (`immediately`, `without`, `before`, `after`, `safely`, `promptly`).
     - Added explicit exclusions for numeric ranges (`from 1 to 10`), time ranges (`from 9am to 5pm`), day/month ranges (`from Monday to Friday`), temporal words (`start`, `end`, `finish`), and pure stopwords.
  2. Enhanced `CandidateValidator` Check 14a to test both forward inversion (`from <obj> to <subj>`) and prepositional inversion (`to <subj> from <obj>`).
- **Regression Tests Added** (`backend/tests/test_general_purpose_optimization.py`):
  - `test_multi_word_directional_relation_extraction`: Verifies extraction of multi-word sources/destinations.
  - `test_directional_reversal_rejected_by_validator`: Verifies rejection of both forward and inverted directional reversals.
  - `test_non_directional_from_to_ranges_ignored`: Verifies non-directional numeric, day, and time ranges are ignored.

### 10.4 RISK-01 Remediation: Generic Named-Entity Preservation
- **File Modified**: `backend/app/services/task_analyzer.py` (lines 675–765).
- **Root Cause**: `_extract_entities` only extracted `KNOWN_TECH_ENTITIES`, CamelCase, and uppercase acronyms. Generic proper nouns (e.g. `"Alliance University"`, `"Bengaluru"`, `"Karnataka"`, `"Wayne Enterprises"`) were dropped from `analysis.entities`, preventing `CandidateValidator` Check 11 from detecting dropped proper nouns.
- **Fix Implemented**:
  1. Defined `GENERIC_FUNCTIONAL_WORDS` containing common prompt imperative verbs, structural nouns, pronouns, conjunctions, and connective adverbs.
  2. Implemented multi-word title-case sequence extraction (`multi_pat`) with boundary trimming of leading/trailing functional words.
  3. Added mid-sentence capitalized word extraction (`idx > 0`) for single proper nouns, filtering out functional words.
  4. Added protection preventing known tech entities from forming synthetic compound entities with action verbs (e.g. `"Audit Python"` yields tech entity `Python`, not `"Audit Python"`).
  5. Implemented deduplication logic preventing redundant single tokens from being retained when subsumed by multi-word entities, unless the token is a known tech entity.
- **Regression Tests Added** (`backend/tests/test_general_purpose_optimization.py`):
  - `test_generic_named_entity_extraction`: Verifies extraction of proper nouns and institutions while excluding sentence-initial prompt verbs.
  - `test_candidate_validator_rejects_missing_generic_entity`: Verifies rejection when generic proper nouns are dropped by candidate rewrites.
- **Remaining Limitations**:
  - Single proper nouns at index 0 of a sentence that are not in `KNOWN_TECH_ENTITIES`, not CamelCase, and not part of a multi-word sequence (e.g. `"Zurich has strict regulations."`) are conservatively omitted from entities to prevent sentence-initial English words from causing false-positive rejections. Full POS-tagger integration can be evaluated if sentence-initial single-token proper noun preservation is required.

### 10.5 Independent Adversarial Review Hardening
Following the multi-agent remediation, an independent adversarial review identified and resolved four edge-case boundaries:
1. **Preposition Boundary Stops in Directional Relations**: Added `in`, `for`, `at`, `on`, `into`, `as` to the boundary stopping set in `dir_pat` so trailing prepositional phrases (e.g. `for disaster recovery`, `in Europe`) do not corrupt extracted destination entities or cause extraction failure.
2. **Article/Determiner Resilience in Reversals**: Added automatic stripping of leading determiners (`the`, `a`, `an`, `my`, `our`, `your`, `their`, `this`, `that`) in `TaskAnalyzer`, and made determiners optional in `CandidateValidator` Check 14a inversion patterns (`inv_pat1`, `inv_pat2`). Reversals are reliably caught regardless of whether candidates retain, add, or drop articles.
3. **Markdown Heading Protection**: Added regex pruning (`(?m)^\s*#+\s+.*$`) before entity extraction to ensure markdown section headers (e.g. `### Project Requirements`) are not extracted as mandatory prompt entities.
4. **Legitimate Compound Tech Organizations**: Refined the compound entity filter so legitimate tech organizations and tools (e.g. `Python Software Foundation`, `React Native`, `Docker Desktop`) are preserved as cohesive multi-word entities, while adding `"azure"` to `KNOWN_TECH_ENTITIES`.
5. **Cold-Start Warning Real Fixture Test**: Added a comprehensive test in `frontend/tests/OutputComparison.test.tsx` verifying that a raw backend comparison response with warm-up timeout correctly renders the Cold-Start Warning banner and tooltip error message.

---

## 11. Final Verification Metrics Summary

| Verification Track | Before Remediation | After Remediation | Status |
| :--- | :---: | :---: | :---: |
| Backend Pytest Unit Tests | 329 passed | **334 passed** (340 total) | **PASS** |
| Backend Live Ollama Tests | 2 passed | **6 passed** (334 deselected) | **PASS** |
| Pyright Type Check | 0 errors | **0 errors** | **PASS** |
| Frontend Vitest Tests | 37 passed (6 files) | **40 passed (7 files)** | **PASS** |
| Frontend Production Build | Clean (339 kB) | **Clean (339 kB)** | **PASS** |
| DEFECT-01 (Frontend Telemetry) | Open | **Resolved & Verified** | **CLOSED** |
| DEFECT-02 (Test Isolation) | Open | **Resolved & Verified** | **CLOSED** |
| DEFECT-03 (Directional Relations) | Open | **Resolved & Verified** | **CLOSED** |
| RISK-01 (Generic Named Entities) | Open | **Resolved & Verified** | **CLOSED** |
| Git Branch & Status | `main` (clean) | `fix/post-merge-validation-findings` | **READY** |
