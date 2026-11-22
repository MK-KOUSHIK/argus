# General-Purpose Prompt Optimization Generalization Benchmark Report

- **Model:** `qwen2.5:1.5b` via local Ollama
- **Date / Timestamp:** 2026-11-22 07:57:09 UTC
- **Dataset:** 25 completely unseen prompts across 7 domains
- **Structural Validation Accuracy:** **88.0%** (Checks 1-14 guardrail pass rate)
- **Dispatched Semantic Preservation:** **100.0%** (Independent 4-pillar evaluation)
- **Accepted Candidate Semantic Rate:** **100.0%** (8/8 accepted candidates)
- **Verbatim Dispatch on Rejection:** **100.0%** (16/16 verified verbatim)
- **Constraint Preservation Rate:** **100.0%**
- **Candidate Rejection Rate:** **64.0%** (16/25)
- **Accepted for Compression:** **8** (32.0%)
- **Accepted for Expansion:** **1** (4.0%)
- **Safe Rejections Preserved Verbatim:** **16** (64.0%)
- **False Acceptance Count (Indep Rubric):** **0** (0.0% of total)
- **Total Baseline Input Tokens:** 831
- **Total Dispatched Tokens:** 812
- **Gross Compression Tokens Saved:** **21 (2.53%)**
- **Actual Net Dispatched Tokens Saved:** **19 (2.29%)**
- **Token Inflation Rate:** **0.0%** (Zero token inflation)

## Detailed Unseen Test Case Results

| ID | Domain | Prompt Description | Orig | Cand | Disp | Saved | Red % | Decision | Rejection Reason / Note | Indep Semantic |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|
| `GEN-CD-01` | Coding | Async image downloader with error logging | 48 | 51 | 48 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (51) is greater than original (48) by 3 ... | `PASS` |
| `GEN-CD-02` | Coding | TypeScript LRU cache with O(1) | 31 | 32 | 31 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (32) is greater than original (31) by 1 ... | `PASS` |
| `GEN-CD-03` | Coding | Rust IPv4 parser without external crates | 28 | 28 | 28 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... | `PASS` |
| `GEN-CD-04` | Coding | Streaming CSV generator line-by-line | 34 | 33 | 34 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate failed to preserve expli... | `PASS` |
| `GEN-TI-01` | Technical Instructions | systemd Node.js service unit | 39 | 38 | 38 | 1 | 2.6% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-TI-02` | Technical Instructions | Terraform AWS S3 bucket with versioning | 34 | 34 | 34 | 0 | 0.0% | **Rejected (Equal Tokens)** | Candidate token count (34) is equal to original (34). No token... | `PASS` |
| `GEN-TI-03` | Technical Instructions | Docker Redis 7.2 container command | 37 | 36 | 37 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate dropped the primary expl... | `PASS` |
| `GEN-RS-01` | Research | Raft vs Paxos consensus trade-offs | 20 | 20 | 20 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... | `PASS` |
| `GEN-RS-02` | Research | B-tree vs LSM-tree storage engines | 22 | 24 | 22 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (24) is greater than original (22) by 2 ... | `PASS` |
| `GEN-RS-03` | Research | ZK-SNARKs vs STARKs comparison | 27 | 28 | 27 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (28) is greater than original (27) by 1 ... | `PASS` |
| `GEN-SQ-01` | Simple Questions | Speed of light in vacuum | 13 | 14 | 13 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (14) is greater than original (13) by 1 ... | `PASS` |
| `GEN-SQ-02` | Simple Questions | Primary currency of Japan (polite) | 14 | 9 | 9 | 5 | 35.7% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-SQ-03` | Simple Questions | Linux kernel creator | 8 | 6 | 6 | 2 | 25.0% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-AM-01` | Ambiguous | Vague bug fix request | 10 | 12 | 12 | 0 | 0.0% | **Accepted (Expansion)** | Accepted under expansion | `PASS` |
| `GEN-AM-02` | Ambiguous | Underspecified file parser | 14 | 14 | 14 | 0 | 0.0% | **Rejected (Equal Tokens)** | Candidate token count (14) is equal to original (14). No token... | `PASS` |
| `GEN-AM-03` | Ambiguous | Unspecified dashboard request | 8 | 7 | 7 | 1 | 12.5% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-GR-01` | Grammar & Phrasing | Image scraper with slang and typos | 18 | 17 | 17 | 1 | 5.6% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-GR-02` | Grammar & Phrasing | SQL department budget broken grammar | 20 | 21 | 20 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (21) is greater than original (20) by 1 ... | `PASS` |
| `GEN-GR-03` | Grammar & Phrasing | Palindrome checker typo and slang | 22 | 17 | 17 | 5 | 22.7% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-GR-04` | Grammar & Phrasing | React re-render grammatical query | 12 | 13 | 12 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (13) is greater than original (12) by 1 ... | `PASS` |
| `GEN-MC-01` | Multi-Constraint | PostgreSQL 15 orders query with date and limits | 77 | 83 | 77 | 0 | 0.0% | **Rejected (Validation)** | Semantic validation failed: Candidate replaced formal SQL 'ORD... | `PASS` |
| `GEN-MC-02` | Multi-Constraint | Incident report email with time and word limit | 74 | 76 | 74 | 0 | 0.0% | **Rejected (Expansion)** | Candidate token count (76) is greater than original (74) by 2 ... | `PASS` |
| `GEN-MC-03` | Multi-Constraint | Bash backup script with disk check and prohibition | 53 | 52 | 52 | 1 | 1.9% | **Accepted (Compression)** | Accepted for compression | `PASS` |
| `GEN-MC-04` | Multi-Constraint | Python ETL data pipeline with retries and memory bounds | 78 | 78 | 78 | 0 | 0.0% | **Rejected (Identical Text)** | Candidate rewrite is identical to original prompt. No token re... | `PASS` |
| `GEN-MC-05` | Multi-Constraint | Kubernetes deployment YAML with resource bounds and probes | 90 | 85 | 85 | 5 | 5.6% | **Accepted (Compression)** | Accepted for compression | `PASS` |

## Performance Breakdown by Domain

| Domain | Cases | Accepted (Comp) | Accepted (Exp) | Rejected / Preserved | Baseline Tokens | Dispatched Tokens | Gross Saved | Net Saved | Net Red % | Structural Valid % | Indep Semantic % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ambiguous | 3 | 1 | 1 | 1 | 32 | 33 | 1 | -1 | -3.1% | 100.0% | 100.0% |
| Coding | 4 | 0 | 0 | 4 | 141 | 141 | 0 | 0 | 0.0% | 75.0% | 100.0% |
| Grammar & Phrasing | 4 | 2 | 0 | 2 | 72 | 66 | 6 | 6 | 8.3% | 100.0% | 100.0% |
| Multi-Constraint | 5 | 2 | 0 | 3 | 372 | 366 | 6 | 6 | 1.6% | 80.0% | 100.0% |
| Research | 3 | 0 | 0 | 3 | 69 | 69 | 0 | 0 | 0.0% | 100.0% | 100.0% |
| Simple Questions | 3 | 2 | 0 | 1 | 35 | 28 | 7 | 7 | 20.0% | 100.0% | 100.0% |
| Technical Instructions | 3 | 1 | 0 | 2 | 110 | 109 | 1 | 1 | 0.9% | 66.7% | 100.0% |
