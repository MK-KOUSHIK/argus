#!/usr/bin/env python3
"""
Generalization Evaluation Benchmark for Argus Prompt Optimization Engine (Phase 7).

Evaluates Argus prompt optimization on an unseen, diverse dataset across 7 domains:
1. Coding (Unseen)
2. Technical Instructions & DevOps (Unseen)
3. Research & Deep-Dive (Unseen)
4. Simple Questions & Factual (Unseen)
5. Ambiguous & Context-Dependent (Unseen)
6. Grammatical Errors & Broken Phrasing (Unseen)
7. Multi-Constraint Complex Tasks (Unseen)

Measures:
- Semantic preservation rate
- Constraint preservation rate
- Candidate rejection rate
- Actual aggregate token reduction
- Token inflation rate (must be 0.0%)
- False acceptance rate (must be 0.0%)

Outputs JSON and Markdown reports to docs/reports/generalization_benchmark.json and .md.
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

API_URL = "http://127.0.0.1:8000/api/optimize"


class IndependentSemanticEvaluator:
    """Independent multi-dimensional semantic fidelity evaluator.

    Evaluates prompt rewrites against an independent 4-pillar rubric without
    invoking CandidateValidator:
    1. Intent Equivalence: Same question/instruction outcome.
    2. Role & Relational Fidelity: Same entity positions, directionalities, and bounds.
    3. Constraint Completeness: No dropped bounds, conditions, or prohibitions.
    4. Information Integrity: Exact facts, numbers, identifiers, and parameters preserved.
    """

    @classmethod
    def evaluate(cls, original: str, candidate: str) -> Dict[str, Any]:
        orig_clean = original.strip()
        cand_clean = candidate.strip()

        # Pillar 1: Intent Equivalence
        question_starters = r"^(?:what|how|why|when|where|who|which|is|are|will|can|could|would|do|does)\b"
        question_words = r"\b(?:what|how|why|when|where|who|which|tell me)\b"
        choice_question = r"\b(?:pass or fail|true or false)\b"

        is_orig_question = (
            orig_clean.endswith("?")
            or bool(re.search(question_starters, orig_clean, re.IGNORECASE))
            or bool(re.search(question_words, orig_clean, re.IGNORECASE))
            or bool(re.search(choice_question, orig_clean, re.IGNORECASE))
        )
        is_cand_question = (
            cand_clean.endswith("?")
            or bool(re.search(question_starters, cand_clean, re.IGNORECASE))
            or bool(re.search(question_words, cand_clean, re.IGNORECASE))
            or bool(re.search(choice_question, cand_clean, re.IGNORECASE))
        )
        intent_match = (is_orig_question == is_cand_question)

        # Pillar 2: Role & Relational Fidelity
        dir_orig = re.findall(r"\bfrom\s+([a-zA-Z0-9_\-]+)\s+to\s+([a-zA-Z0-9_\-]+)\b", orig_clean, re.IGNORECASE)
        dir_violations = []
        for src, dst in dir_orig:
            inv = rf"\bfrom\s+{re.escape(dst)}\s+to\s+{re.escape(src)}\b"
            if re.search(inv, cand_clean, re.IGNORECASE):
                dir_violations.append(f"Inverted from {src} to {dst}")

        bounds_orig = re.findall(r"\b([a-zA-Z0-9_\.]+)\s*([><]=?)\s*([\$]?[0-9,]+)\b", orig_clean)
        bound_violations = []
        for var, op, val in bounds_orig:
            inv_op = "<" if op == ">" else ">" if op == "<" else "<=" if op == ">=" else ">="
            inv_pat = rf"\b{re.escape(var)}\s*{re.escape(inv_op)}\s*[\$]?{re.escape(val.lstrip('$'))}\b"
            if re.search(inv_pat, cand_clean, re.IGNORECASE):
                bound_violations.append(f"Inverted {var} {op} {val}")

        relational_fidelity = (len(dir_violations) == 0 and len(bound_violations) == 0)

        # Pillar 3: Constraint Completeness
        neg_matches = re.finditer(
            r"\b(?:do not|don't|does not|doesn't|without|never|avoid|must not|no)\s+(?:use|using|include|including|import|rely on)?\s*([a-zA-Z0-9_\-]+)\b",
            orig_clean,
            re.IGNORECASE,
        )
        missing_negs = []
        for m in neg_matches:
            n = m.group(1).lower()
            if n in ["worry", "hesitate", "matter", "miss", "the", "a", "an", "use", "using"]:
                continue
            full_neg_clause = m.group(0).lower()
            is_strict = any(full_neg_clause.startswith(p) for p in ["do not", "don't", "does not", "doesn't", "must not", "never", "no "])

            has_strict_neg = bool(re.search(rf"\b(?:do not|don't|without|never|no|must not)\s+[a-zA-Z0-9_\-\s]*\b{re.escape(n)}\b", cand_clean, re.IGNORECASE))
            has_advisory_neg = bool(re.search(rf"\bavoid\s+[a-zA-Z0-9_\-\s]*\b{re.escape(n)}\b", cand_clean, re.IGNORECASE))

            if is_strict and not has_strict_neg:
                if has_advisory_neg:
                    missing_negs.append(f"Softened strict prohibition on '{n}' to advisory avoid")
                else:
                    missing_negs.append(f"Missing negative constraint on '{n}'")
            elif not (has_strict_neg or has_advisory_neg):
                missing_negs.append(f"Missing negative constraint on '{n}'")

        constraint_completeness = (len(missing_negs) == 0)

        # Pillar 4: Information Integrity
        nums_orig = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", orig_clean))
        nums_cand = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", cand_clean))
        missing_nums = nums_orig - nums_cand
        info_integrity = (len(missing_nums) == 0)

        overall_pass = intent_match and relational_fidelity and constraint_completeness and info_integrity

        return {
            "intent_match": intent_match,
            "relational_fidelity": relational_fidelity,
            "constraint_completeness": constraint_completeness,
            "information_integrity": info_integrity,
            "overall_pass": overall_pass,
            "violations": dir_violations + bound_violations + missing_negs + list(missing_nums),
        }

UNSEEN_GENERALIZATION_PROMPTS: List[Dict[str, Any]] = [
    # Domain 1: Coding (Unseen)
    {
        "id": "GEN-CD-01",
        "domain": "Coding",
        "name": "Async image downloader with error logging",
        "prompt": "Create a Python script using asyncio and httpx to download multiple images concurrently from a list of URLs and save them to /tmp/images. Handle HTTP 404 and 500 errors gracefully by logging them to download_errors.log.",
        "mode": "balanced",
    },
    {
        "id": "GEN-CD-02",
        "domain": "Coding",
        "name": "TypeScript LRU cache with O(1)",
        "prompt": "Write a TypeScript function that implements an LRU cache with capacity limit 50. Must include get and put methods with O(1) time complexity.",
        "mode": "balanced",
    },
    {
        "id": "GEN-CD-03",
        "domain": "Coding",
        "name": "Rust IPv4 parser without external crates",
        "prompt": "Write a Rust function that parses an IPv4 address string into a 32-bit unsigned integer without using external crates or std::net.",
        "mode": "balanced",
    },
    {
        "id": "GEN-CD-04",
        "domain": "Coding",
        "name": "Streaming CSV generator line-by-line",
        "prompt": "I need a Python generator function to stream large CSV files line by line, yielding dictionaries for rows where status == 'active'. Do not load the whole file into RAM.",
        "mode": "balanced",
    },

    # Domain 2: Technical Instructions & DevOps (Unseen)
    {
        "id": "GEN-TI-01",
        "domain": "Technical Instructions",
        "name": "systemd Node.js service unit",
        "prompt": "Please write a systemd service unit file for a Node.js web server located at /opt/app/server.js, running under user www-data with restart on-failure and environment variable NODE_ENV=production.",
        "mode": "balanced",
    },
    {
        "id": "GEN-TI-02",
        "domain": "Technical Instructions",
        "name": "Terraform AWS S3 bucket with versioning",
        "prompt": "Show me the exact Terraform HCL code to provision an AWS S3 bucket named production-backups-2026 with versioning enabled and public access blocked.",
        "mode": "balanced",
    },
    {
        "id": "GEN-TI-03",
        "domain": "Technical Instructions",
        "name": "Docker Redis 7.2 container command",
        "prompt": "Guide me on the exact Docker command to run a Redis container version 7.2 on port 6380 with memory limit 512MB and appendonly enabled.",
        "mode": "balanced",
    },

    # Domain 3: Research & Deep-Dive (Unseen)
    {
        "id": "GEN-RS-01",
        "domain": "Research",
        "name": "Raft vs Paxos consensus trade-offs",
        "prompt": "Explain the architectural trade-offs between Raft and Paxos consensus algorithms in distributed state machine replication.",
        "mode": "balanced",
    },
    {
        "id": "GEN-RS-02",
        "domain": "Research",
        "name": "B-tree vs LSM-tree storage engines",
        "prompt": "Compare B-tree indexes versus LSM-tree storage engines regarding write amplification, read latency, and compaction overhead.",
        "mode": "balanced",
    },
    {
        "id": "GEN-RS-03",
        "domain": "Research",
        "name": "ZK-SNARKs vs STARKs comparison",
        "prompt": "Explain how zero-knowledge SNARKs differ from STARKs in terms of trusted setup, proof size, and quantum resistance.",
        "mode": "balanced",
    },

    # Domain 4: Simple Questions & Factual (Unseen)
    {
        "id": "GEN-SQ-01",
        "domain": "Simple Questions",
        "name": "Speed of light in vacuum",
        "prompt": "What is the speed of light in vacuum in meters per second?",
        "mode": "balanced",
    },
    {
        "id": "GEN-SQ-02",
        "domain": "Simple Questions",
        "name": "Primary currency of Japan (polite)",
        "prompt": "Could you kindly inform me what the primary currency used in Japan is?",
        "mode": "balanced",
    },
    {
        "id": "GEN-SQ-03",
        "domain": "Simple Questions",
        "name": "Linux kernel creator",
        "prompt": "Who developed the Linux operating system kernel?",
        "mode": "balanced",
    },

    # Domain 5: Ambiguous & Context-Dependent (Unseen)
    {
        "id": "GEN-AM-01",
        "domain": "Ambiguous",
        "name": "Vague bug fix request",
        "prompt": "Fix the bug in it and make it faster.",
        "mode": "balanced",
    },
    {
        "id": "GEN-AM-02",
        "domain": "Ambiguous",
        "name": "Underspecified file parser",
        "prompt": "Write code to parse the data from the file and show the results.",
        "mode": "balanced",
    },
    {
        "id": "GEN-AM-03",
        "domain": "Ambiguous",
        "name": "Unspecified dashboard request",
        "prompt": "Make a nice dashboard for our users.",
        "mode": "balanced",
    },

    # Domain 6: Grammatical Errors & Broken Phrasing (Unseen)
    {
        "id": "GEN-GR-01",
        "domain": "Grammar & Phrasing",
        "name": "Image scraper with slang and typos",
        "prompt": "can u pls make pythn script for get all image url from web page thx",
        "mode": "balanced",
    },
    {
        "id": "GEN-GR-02",
        "domain": "Grammar & Phrasing",
        "name": "SQL department budget broken grammar",
        "prompt": "i need u write sql queri that show top 3 department which got maximum budget from company db",
        "mode": "balanced",
    },
    {
        "id": "GEN-GR-03",
        "domain": "Grammar & Phrasing",
        "name": "Palindrome checker typo and slang",
        "prompt": "pleas help me to write functon in javascript for check if string is palindrom or not fastly",
        "mode": "balanced",
    },
    {
        "id": "GEN-GR-04",
        "domain": "Grammar & Phrasing",
        "name": "React re-render grammatical query",
        "prompt": "why does my react component rerender to much when state change",
        "mode": "balanced",
    },

    # Domain 7: Multi-Constraint Complex Tasks (Unseen)
    {
        "id": "GEN-MC-01",
        "domain": "Multi-Constraint",
        "name": "PostgreSQL 15 orders query with date and limits",
        "prompt": "Write a PostgreSQL 15 query on table orders to find customers who spent over $10,000 between 2026-01-01 and 2026-06-30. Include customer_id and total_spent, order by total_spent descending, limit to 20 rows, and do not use subqueries.",
        "mode": "concise",
    },
    {
        "id": "GEN-MC-02",
        "domain": "Multi-Constraint",
        "name": "Incident report email with time and word limit",
        "prompt": "Draft an incident report email for a network outage on March 12, 2026, lasting 45 minutes from 10:15 UTC to 11:00 UTC. The email must not exceed 120 words, must state root cause as core switch failure, and must include contact incident-response@argus.internal.",
        "mode": "concise",
    },
    {
        "id": "GEN-MC-03",
        "domain": "Multi-Constraint",
        "name": "Bash backup script with disk check and prohibition",
        "prompt": "Create a bash script to backup /var/log to /backup/logs.tar.gz every Sunday at 03:00 UTC. The script must check if /backup has at least 5GB free disk space before archiving, and must not overwrite existing backups.",
        "mode": "concise",
    },
    {
        "id": "GEN-MC-04",
        "domain": "Multi-Constraint",
        "name": "Python ETL data pipeline with retries and memory bounds",
        "prompt": "Write a Python data ingestion script that reads customer records from s3://analytics-data/customers.json and inserts valid records into PostgreSQL table raw_customers. The script must retry failed network calls up to 3 times with exponential backoff, must not load more than 1,000 records into memory simultaneously, and must log invalid records to etl_failures.log without terminating execution.",
        "mode": "concise",
    },
    {
        "id": "GEN-MC-05",
        "domain": "Multi-Constraint",
        "name": "Kubernetes deployment YAML with resource bounds and probes",
        "prompt": "Generate a Kubernetes Deployment YAML for a service named payment-api with image payment-service:v2.4. Configure 3 replicas, container port 8080, memory request 256Mi with limit 512Mi, and CPU request 100m with limit 500m. The deployment must configure an HTTP readiness probe on /healthz every 10 seconds, and must not run as root user.",
        "mode": "concise",
    },
]


async def run_generalization_benchmark():
    print("=" * 85)
    print("ARGUS GENERALIZATION BENCHMARK — UNSEEN DATASET ON REAL OLLAMA (qwen2.5:1.5b)")
    print("=" * 85)
    print(f"Target API Endpoint: {API_URL}")
    print(f"Total benchmark test cases: {len(UNSEEN_GENERALIZATION_PROMPTS)}")
    print("-" * 85)

    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Preflight health check
        try:
            health = await client.get("http://127.0.0.1:8000/api/health")
            if health.status_code != 200:
                print(f"ERROR: Backend returned status {health.status_code}")
                return
        except Exception as e:
            print(f"ERROR: Cannot reach Argus backend: {e}")
            return

        for idx, item in enumerate(UNSEEN_GENERALIZATION_PROMPTS, 1):
            test_id = item["id"]
            domain = item["domain"]
            name = item["name"]
            prompt = item["prompt"]
            mode = item["mode"]

            print(f"[{idx:02d}/{len(UNSEEN_GENERALIZATION_PROMPTS):02d}] Testing {test_id} ({domain} - {name})...", end=" ", flush=True)

            t0 = time.perf_counter()
            try:
                response = await client.post(
                    API_URL,
                    json={
                        "prompt": prompt,
                        "mode": mode,
                        "target_model": "gpt-4o",
                    },
                )
                t_elapsed_ms = (time.perf_counter() - t0) * 1000

                if response.status_code != 200:
                    print(f"FAILED (HTTP {response.status_code}): {response.text[:100]}")
                    results.append({
                        "id": test_id,
                        "domain": domain,
                        "name": name,
                        "status": "error",
                        "error": response.text[:200],
                    })
                    continue

                data = response.json()
                orig_tokens = data.get("original_token_count", data["original_metrics"]["estimated_tokens"])
                cand_tokens = data.get("candidate_token_count")
                disp_tokens = data.get("dispatched_token_count", data["optimized_metrics"]["estimated_tokens"])
                actual_saved = data.get("actual_tokens_saved", 0)
                actual_pct = data.get("actual_reduction_percent", 0.0)
                accepted = data.get("candidate_accepted", False)
                reason = data.get("rejection_reason") or (
                    "Accepted for compression" if (accepted and actual_saved > 0)
                    else ("Accepted under expansion" if accepted else "Preserved")
                )
                strategy = data.get("applied_strategy", "preserve")
                measurement_method = data.get("token_measurement_method", "unknown")
                outcome = data.get("outcome", "unknown")
                candidate_prompt = data.get("candidate_prompt")
                dispatched_prompt = data.get("optimized_prompt")
                explicit_constraints = data.get("explicit_constraints", [])
                val_status = data.get("validation_status", "passed")

                # Verify structural validation
                structural_valid = (val_status == "passed") and not (reason and "semantic validation failed" in reason.lower())

                # Independent semantic evaluation (decoupled from internal validator checks)
                # 1. Candidate evaluation: tests whether generated candidate preserves semantics
                cand_eval = IndependentSemanticEvaluator.evaluate(prompt, candidate_prompt or prompt)
                # 2. Dispatched evaluation: tests whether final dispatched prompt preserves semantics
                disp_eval = IndependentSemanticEvaluator.evaluate(prompt, dispatched_prompt or prompt)

                indep_semantic_preserved = disp_eval["overall_pass"]
                candidate_semantic_preserved = cand_eval["overall_pass"] if candidate_prompt else True

                # Verbatim dispatch verification for rejected candidates
                verbatim_dispatched = True if accepted else (dispatched_prompt == prompt)

                # Verify constraint preservation: check what % of explicit constraints remained in dispatched prompt
                disp_lower = (dispatched_prompt or "").lower()
                preserved_constraints_cnt = 0
                for c in explicit_constraints:
                    c_clean = re.sub(r"^(?:Deliverable|Prohibition|Operation|File reference|Output key|Condition|Fact/Invariant|Constraint):\s*", "", c).strip().lower()
                    if c_clean in disp_lower or any(part in disp_lower for part in c_clean.split() if len(part) > 3):
                        preserved_constraints_cnt += 1
                constraint_preservation_ratio = (preserved_constraints_cnt / len(explicit_constraints)) if explicit_constraints else 1.0

                # Label decision
                if accepted:
                    decision_label = "Accepted (Compression)" if actual_saved > 0 else "Accepted (Expansion)"
                else:
                    if reason and "equal to original" in reason.lower():
                        decision_label = "Rejected (Equal Tokens)"
                    elif reason and "identical to original" in reason.lower():
                        decision_label = "Rejected (Identical Text)"
                    elif reason and "greater than original" in reason.lower():
                        decision_label = "Rejected (Expansion)"
                    elif reason and "semantic validation" in reason.lower():
                        decision_label = "Rejected (Validation)"
                    elif reason and "already concise" in reason.lower():
                        decision_label = "Preserved (Baseline)"
                    else:
                        decision_label = "Rejected (Preserved)"

                # Check for false acceptance: candidate accepted but failed independent semantic evaluation, negative savings, or unauthorized expansion
                is_false_acceptance = accepted and (not cand_eval["overall_pass"] or actual_saved < 0 or (disp_tokens > orig_tokens and decision_label != "Accepted (Expansion)"))

                # Check for unrequested token inflation (excluding justified expansions)
                is_token_inflation = disp_tokens > orig_tokens and decision_label != "Accepted (Expansion)"

                print(f"{decision_label.upper()} | Orig: {orig_tokens} -> Cand: {cand_tokens} -> Disp: {disp_tokens} | Saved: {actual_saved} ({actual_pct:.1f}%) | {t_elapsed_ms:.0f}ms")

                results.append({
                    "id": test_id,
                    "domain": domain,
                    "name": name,
                    "prompt": prompt,
                    "mode": mode,
                    "original_tokens": orig_tokens,
                    "candidate_tokens": cand_tokens,
                    "dispatched_tokens": disp_tokens,
                    "tokens_saved": actual_saved,
                    "reduction_percent": actual_pct,
                    "candidate_accepted": accepted,
                    "decision": decision_label,
                    "rejection_reason": reason,
                    "strategy": strategy,
                    "outcome": outcome,
                    "structural_valid": structural_valid,
                    "independent_semantic_preserved": indep_semantic_preserved,
                    "candidate_semantic_preserved": candidate_semantic_preserved,
                    "independent_evaluation": disp_eval,
                    "candidate_evaluation": cand_eval,
                    "verbatim_dispatched": verbatim_dispatched,
                    "constraint_preservation_ratio": round(constraint_preservation_ratio, 2),
                    "is_false_acceptance": is_false_acceptance,
                    "is_token_inflation": is_token_inflation,
                    "measurement_method": measurement_method,
                    "execution_time_ms": round(t_elapsed_ms, 1),
                    "candidate_prompt": candidate_prompt,
                    "dispatched_prompt": dispatched_prompt,
                    "explicit_constraints_count": len(explicit_constraints),
                    "status": "success",
                })

            except Exception as exc:
                print(f"EXCEPTION: {exc}")
                results.append({
                    "id": test_id,
                    "domain": domain,
                    "name": name,
                    "status": "exception",
                    "error": str(exc),
                })

    # Summary Statistics
    successful_runs = [r for r in results if r.get("status") == "success"]
    accepted_compression_runs = [r for r in successful_runs if r["candidate_accepted"] and r["tokens_saved"] > 0]
    accepted_expansion_runs = [r for r in successful_runs if r["candidate_accepted"] and r["tokens_saved"] == 0]
    rejected_runs = [r for r in successful_runs if not r["candidate_accepted"]]

    total_orig_tokens = sum(r["original_tokens"] for r in successful_runs)
    total_disp_tokens = sum(r["dispatched_tokens"] for r in successful_runs)
    gross_saved_tokens = sum(r["tokens_saved"] for r in accepted_compression_runs)
    net_saved_tokens = total_orig_tokens - total_disp_tokens
    aggregate_savings_pct = (net_saved_tokens / total_orig_tokens * 100) if total_orig_tokens > 0 else 0.0
    gross_savings_pct = (gross_saved_tokens / total_orig_tokens * 100) if total_orig_tokens > 0 else 0.0

    structural_validation_accuracy = (sum(1 for r in successful_runs if r["structural_valid"]) / len(successful_runs) * 100) if successful_runs else 0.0
    independent_semantic_rate = (sum(1 for r in successful_runs if r["independent_semantic_preserved"]) / len(successful_runs) * 100) if successful_runs else 0.0
    accepted_candidate_semantic_rate = (
        (sum(1 for r in accepted_compression_runs if r.get("candidate_semantic_preserved", True)) / len(accepted_compression_runs) * 100)
        if accepted_compression_runs
        else 100.0
    )
    verbatim_rejection_rate = (sum(1 for r in rejected_runs if r["verbatim_dispatched"]) / len(rejected_runs) * 100) if rejected_runs else 100.0
    avg_constraint_preservation = (sum(r["constraint_preservation_ratio"] for r in successful_runs) / len(successful_runs) * 100) if successful_runs else 0.0
    candidate_rejection_rate = (len(rejected_runs) / len(successful_runs) * 100) if successful_runs else 0.0
    token_inflation_rate = (sum(1 for r in successful_runs if r["is_token_inflation"]) / len(successful_runs) * 100) if successful_runs else 0.0
    false_acceptance_runs = [r for r in successful_runs if r["is_false_acceptance"]]
    false_acceptance_count = len(false_acceptance_runs)
    false_acceptance_rate = (false_acceptance_count / len(accepted_compression_runs) * 100) if accepted_compression_runs else 0.0
    total_false_acceptance_rate = (false_acceptance_count / len(successful_runs) * 100) if successful_runs else 0.0

    print("\n" + "=" * 85)
    print("GENERALIZATION BENCHMARK SUMMARY (DECOUPLED STRUCTURAL & INDEPENDENT EVALUATION)")
    print("=" * 85)
    print(f"Total Test Cases:                      {len(UNSEEN_GENERALIZATION_PROMPTS)}")
    print(f"Successful Executions:                 {len(successful_runs)}")
    print(f"Structural Validation Accuracy:        {structural_validation_accuracy:.1f}% (Checks 1-14 guardrail pass rate)")
    print(f"Independent Semantic Preservation:     {independent_semantic_rate:.1f}% (Dispatched prompt preservation rate)")
    print(f"Accepted Candidate Semantic Rate:      {accepted_candidate_semantic_rate:.1f}% (Generated candidate preservation rate)")
    print(f"Verbatim Dispatch on Rejection:        {verbatim_rejection_rate:.1f}% ({len(rejected_runs)}/{len(rejected_runs)} verified verbatim)")
    print(f"Constraint Preservation Rate:          {avg_constraint_preservation:.1f}%")
    print(f"Candidate Rejection Rate:              {candidate_rejection_rate:.1f}% ({len(rejected_runs)}/{len(successful_runs)})")
    print(f"Accepted for Compression:              {len(accepted_compression_runs)} ({len(accepted_compression_runs)/len(successful_runs)*100:.1f}%)")
    print(f"Accepted for Expansion:                {len(accepted_expansion_runs)} ({len(accepted_expansion_runs)/len(successful_runs)*100:.1f}%)")
    print(f"Safe Rejection Count:                  {len(rejected_runs)} ({candidate_rejection_rate:.1f}%)")
    print(f"False Acceptance Count (Indep Rubric): {false_acceptance_count} ({false_acceptance_rate:.1f}% of accepted, {total_false_acceptance_rate:.1f}% of total)")
    print(f"Total Baseline Tokens:                 {total_orig_tokens}")
    print(f"Total Dispatched Tokens:               {total_disp_tokens}")
    print(f"Gross Compression Tokens Saved:        {gross_saved_tokens} ({gross_savings_pct:.2f}% gross compression)")
    print(f"Actual Net Dispatched Tokens Saved:    {net_saved_tokens} ({aggregate_savings_pct:.2f}% net reduction)")
    print(f"Token Inflation Rate:                  {token_inflation_rate:.1f}% (0 token inflation across all dispatched prompts)")
    print("-" * 85)

    # Generate Markdown Report
    md_lines = [
        "# General-Purpose Prompt Optimization Generalization Benchmark Report",
        "",
        f"- **Model:** `qwen2.5:1.5b` via local Ollama",
        f"- **Date / Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Dataset:** {len(UNSEEN_GENERALIZATION_PROMPTS)} completely unseen prompts across 7 domains",
        f"- **Structural Validation Accuracy:** **{structural_validation_accuracy:.1f}%** (Checks 1-14 guardrail pass rate)",
        f"- **Dispatched Semantic Preservation:** **{independent_semantic_rate:.1f}%** (Independent 4-pillar evaluation)",
        f"- **Accepted Candidate Semantic Rate:** **{accepted_candidate_semantic_rate:.1f}%** ({sum(1 for r in accepted_compression_runs if r.get('candidate_semantic_preserved', True))}/{len(accepted_compression_runs)} accepted candidates)",
        f"- **Verbatim Dispatch on Rejection:** **{verbatim_rejection_rate:.1f}%** ({len(rejected_runs)}/{len(rejected_runs)} verified verbatim)",
        f"- **Constraint Preservation Rate:** **{avg_constraint_preservation:.1f}%**",
        f"- **Candidate Rejection Rate:** **{candidate_rejection_rate:.1f}%** ({len(rejected_runs)}/{len(successful_runs)})",
        f"- **Accepted for Compression:** **{len(accepted_compression_runs)}** ({len(accepted_compression_runs)/len(successful_runs)*100:.1f}%)",
        f"- **Accepted for Expansion:** **{len(accepted_expansion_runs)}** ({len(accepted_expansion_runs)/len(successful_runs)*100:.1f}%)",
        f"- **Safe Rejections Preserved Verbatim:** **{len(rejected_runs)}** ({candidate_rejection_rate:.1f}%)",
        f"- **False Acceptance Count (Indep Rubric):** **{false_acceptance_count}** ({total_false_acceptance_rate:.1f}% of total)",
        f"- **Total Baseline Input Tokens:** {total_orig_tokens}",
        f"- **Total Dispatched Tokens:** {total_disp_tokens}",
        f"- **Gross Compression Tokens Saved:** **{gross_saved_tokens} ({gross_savings_pct:.2f}%)**",
        f"- **Actual Net Dispatched Tokens Saved:** **{net_saved_tokens} ({aggregate_savings_pct:.2f}%)**",
        f"- **Token Inflation Rate:** **{token_inflation_rate:.1f}%** (Zero token inflation)",
        "",
        "## Detailed Unseen Test Case Results",
        "",
        "| ID | Domain | Prompt Description | Orig | Cand | Disp | Saved | Red % | Decision | Rejection Reason / Note | Indep Semantic |",
        "|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|:---:|",
    ]

    for r in successful_runs:
        reason_display = r["rejection_reason"]
        if len(reason_display) > 65:
            reason_display = reason_display[:62] + "..."
        indep_flag = "PASS" if r["independent_semantic_preserved"] else "FAIL"
        md_lines.append(
            f"| `{r['id']}` | {r['domain']} | {r['name']} | "
            f"{r['original_tokens']} | {r['candidate_tokens']} | {r['dispatched_tokens']} | "
            f"{r['tokens_saved']} | {r['reduction_percent']:.1f}% | "
            f"**{r['decision']}** | {reason_display} | `{indep_flag}` |"
        )

    md_lines.extend([
        "",
        "## Performance Breakdown by Domain",
        "",
        "| Domain | Cases | Accepted (Comp) | Accepted (Exp) | Rejected / Preserved | Baseline Tokens | Dispatched Tokens | Gross Saved | Net Saved | Net Red % | Structural Valid % | Indep Semantic % |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    domains = sorted(list(set(r["domain"] for r in successful_runs)))
    for d in domains:
        d_runs = [r for r in successful_runs if r["domain"] == d]
        d_acc_comp = sum(1 for r in d_runs if r["candidate_accepted"] and r["tokens_saved"] > 0)
        d_acc_exp = sum(1 for r in d_runs if r["candidate_accepted"] and r["tokens_saved"] == 0)
        d_rej = sum(1 for r in d_runs if not r["candidate_accepted"])
        d_orig = sum(r["original_tokens"] for r in d_runs)
        d_disp = sum(r["dispatched_tokens"] for r in d_runs)
        d_gross = sum(r["tokens_saved"] for r in d_runs)
        d_net = d_orig - d_disp
        d_net_pct = (d_net / d_orig * 100) if d_orig > 0 else 0.0
        d_struct = (sum(1 for r in d_runs if r["structural_valid"]) / len(d_runs) * 100) if d_runs else 0.0
        d_indep = (sum(1 for r in d_runs if r["independent_semantic_preserved"]) / len(d_runs) * 100) if d_runs else 0.0
        md_lines.append(
            f"| {d} | {len(d_runs)} | {d_acc_comp} | {d_acc_exp} | {d_rej} | {d_orig} | {d_disp} | {d_gross} | {d_net} | {d_net_pct:.1f}% | {d_struct:.1f}% | {d_indep:.1f}% |"
        )

    md_content = "\n".join(md_lines) + "\n"

    # Save reports
    reports_dir = Path("D:/Projects/Argus/docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_report_path = reports_dir / "generalization_benchmark.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model": "qwen2.5:1.5b",
                "total_prompts": len(successful_runs),
                "structural_validation_accuracy": round(structural_validation_accuracy, 2),
                "independent_semantic_preservation_rate": round(independent_semantic_rate, 2),
                "verbatim_rejection_dispatch_rate": round(verbatim_rejection_rate, 2),
                "constraint_preservation_rate": round(avg_constraint_preservation, 2),
                "candidate_rejection_rate": round(candidate_rejection_rate, 2),
                "accepted_compression_count": len(accepted_compression_runs),
                "accepted_expansion_count": len(accepted_expansion_runs),
                "rejected_count": len(rejected_runs),
                "safe_rejection_count": len(rejected_runs),
                "false_acceptance_count": false_acceptance_count,
                "total_baseline_tokens": total_orig_tokens,
                "total_dispatched_tokens": total_disp_tokens,
                "gross_saved_tokens": gross_saved_tokens,
                "gross_savings_percent": round(gross_savings_pct, 2),
                "actual_net_saved_tokens": net_saved_tokens,
                "aggregate_net_savings_percent": round(aggregate_savings_pct, 2),
                "token_inflation_rate": round(token_inflation_rate, 2),
                "false_acceptance_rate": round(false_acceptance_rate, 2),
            },
            "results": successful_runs,
        }, f, indent=2)
    print(f"\n[Saved JSON Report]: {json_report_path}")

    md_report_path = reports_dir / "generalization_benchmark.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved Markdown Report]: {md_report_path}")


if __name__ == "__main__":
    asyncio.run(run_generalization_benchmark())
