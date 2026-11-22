# Prompt Optimization Acceptance & Reduction Benchmark Report

- **Model:** `qwen2.5:1.5b` via local Ollama
- **Date / Timestamp:** 2026-11-07 13:40:20 UTC
- **Total Prompts Evaluated:** 21
- **Accepted for Compression:** 10 (47.6%)
- **Accepted for Expansion:** 0 (0.0%)
- **Rejected / Original Preserved:** 11 (52.4%)
- **Total Baseline Input Tokens:** 799
- **Total Dispatched Tokens:** 718
- **Net Tokens Saved:** 81 (10.14%)
- **Unjustified Expansions / Negative Savings Reported:** 0 (100% regression prevention)

## Detailed Results by Test Case

| ID | Domain | Prompt Description | Orig | Cand | Disp | Saved | Red % | Decision | Rejection Reason / Note |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `SQ-01` | Simple Questions | Capital of France | 7 | 7 | 7 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... |
| `SQ-02` | Simple Questions | Boiling point of water (polite) | 16 | 11 | 11 | 5 | 31.2% | **Accepted (Compression)** | Accepted for compression |
| `SQ-03` | Simple Questions | Solar system planets | 10 | 10 | 10 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... |
| `TI-01` | Technical Instructions | Docker compose PostgreSQL | 36 | 33 | 33 | 3 | 8.3% | **Accepted (Compression)** | Accepted for compression |
| `TI-02` | Technical Instructions | OpenSSL RSA key and CSR | 39 | 38 | 39 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate dropped the primary expl... |
| `TI-03` | Technical Instructions | cURL auth POST request | 33 | 33 | 33 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... |
| `CD-01` | Coding | Fibonacci function | 9 | 9 | 9 | 0 | 0.0% | **Rejected (Equal Tokens)** | Candidate token count (9) is equal to original (9). No token r... |
| `CD-02` | Coding | BeautifulSoup link scraper | 31 | 16 | 16 | 15 | 48.4% | **Accepted (Compression)** | Accepted for compression |
| `CD-03` | Coding | TypeScript binary search | 29 | 29 | 29 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... |
| `CD-04` | Coding | C++ quick sort with comments | 18 | 15 | 15 | 3 | 16.7% | **Accepted (Compression)** | Accepted for compression |
| `LR-01` | Long Redundant | SQL vs NoSQL pleasantries | 72 | 48 | 48 | 24 | 33.3% | **Accepted (Compression)** | Accepted for compression |
| `LR-02` | Long Redundant | Persona & fluff microservices | 65 | 53 | 65 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate failed to preserve expli... |
| `LR-03` | Long Redundant | Java GC conversational filler | 59 | 30 | 59 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate dropped the primary expl... |
| `LR-04` | Long Redundant | Sales aggregation SQL with pleasantry | 76 | 40 | 76 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate dropped the primary soft... |
| `RS-01` | Research | Monolith vs Microservices trade-offs | 18 | 17 | 17 | 1 | 5.6% | **Accepted (Compression)** | Accepted for compression |
| `RS-02` | Research | Scaled Dot-Product vs FlashAttention-2 | 34 | 29 | 29 | 5 | 14.7% | **Accepted (Compression)** | Accepted for compression |
| `RS-03` | Research | Concurrency control in distributed DBs | 17 | 17 | 17 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... |
| `MC-01` | Multi-Constraint | JSON user profile with strict schema | 74 | 59 | 74 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate dropped required operati... |
| `MC-02` | Multi-Constraint | Maintenance window email under 100 words | 73 | 70 | 70 | 3 | 4.1% | **Accepted (Compression)** | Accepted for compression |
| `MC-03` | Multi-Constraint | Duplicate email SQL without subqueries | 50 | 37 | 37 | 13 | 26.0% | **Accepted (Compression)** | Accepted for compression |
| `RP-01` | Technical Instructions | Prometheus Kubernetes ServiceMonitors (reproduction) | 33 | 24 | 24 | 9 | 27.3% | **Accepted (Compression)** | Accepted for compression |

## Domain-Level Performance Breakdown

| Domain | Count | Accepted (Comp) | Accepted (Exp) | Rejected / Preserved | Baseline Tokens | Dispatched Tokens | Saved Tokens | Reduction % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Coding | 4 | 2 | 0 | 2 | 87 | 69 | 18 | 20.7% |
| Long Redundant | 4 | 1 | 0 | 3 | 272 | 248 | 24 | 8.8% |
| Multi-Constraint | 3 | 2 | 0 | 1 | 197 | 181 | 16 | 8.1% |
| Research | 3 | 2 | 0 | 1 | 69 | 63 | 6 | 8.7% |
| Simple Questions | 3 | 1 | 0 | 2 | 33 | 28 | 5 | 15.2% |
| Technical Instructions | 4 | 2 | 0 | 2 | 141 | 129 | 12 | 8.5% |
