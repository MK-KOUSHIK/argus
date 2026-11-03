"""Controlled Empirical Experiment for Ollama KV-Cache Prefix Reuse.

Evaluates whether identical prompt prefixes yield measurable prefill time reduction
and inspects Ollama's native prompt_eval_cached_count counter.
Compares a Stable Prefix group against a Varied Prefix control group under
identical model residency, context size (4096), and hardware conditions.
"""

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import httpx

COMMON_SYSTEM_INSTRUCTION = (
    "You are a principal software architect and systems engineer. "
    "Provide rigorous, precise, and production-ready technical assessments."
)

COMMON_PREFIX_SPECIFICATION = (
    "System Architecture Specification v4.2:\n"
    "The target infrastructure consists of an event-driven telemetry ingest layer. "
    "IoT edge sensors stream protocol buffer payloads over TLS to Apache Kafka cluster brokers. "
    "A distributed pool of Python workers running AsyncIO consume records from the 'telemetry-stream' topic, "
    "validating payload schemas against an Avro schema registry, executing token-bucket rate limiting "
    "with Redis cluster backing (10,000 max tokens per tenant, 1s replenishment period), and appending "
    "enrichment data queried from a read-replica PostgreSQL cluster with PgBouncer connection pooling. "
    "Transformed analytical events are flushed in micro-batches (5,000 records or 250ms window) "
    "to a ClickHouse columnar database with dedicated ReplacingMergeTree engines partitioned by event date.\n"
)

# 5 Varied prefixes for the control group (different domains, comparable token footprint)
VARIED_CONTROL_PREFIXES = [
    (
        "Clinical Laboratory Management System v1.0:\n"
        "Specimen tracking begins at outpatient phlebotomy stations where barcoded tubes are scanned into HL7 interfaces. "
        "Automated chemistry analyzers run spectrophotometric assays, streaming discrete result flags across a DICOM bus. "
        "A secure microservice validates panic values, routes alert notifications to on-call pathologists, "
        "and archives encrypted audit logs to HIPAA-compliant cold storage with immutable retention schedules.\n"
    ),
    (
        "Automated Trading Engine Specification v8.1:\n"
        "Market data feeds arrive via ultra-low latency UDP multicast directly from exchange colocation gateways. "
        "FPGA-accelerated network cards demultiplex ITCH book feeds, maintaining a microsecond-level limit order book. "
        "Quantitative pricing models compute dynamic option Greeks, evaluating risk limits across margin accounts "
        "and routing cancellation orders across FIX protocol sessions before exchange circuit-breakers trigger.\n"
    ),
    (
        "E-Commerce Catalog & Inventory Pipeline v3.5:\n"
        "Warehouse inventory updates stream from handheld barcode scanners across regional fulfillment nodes. "
        "A change-data-capture Kafka connector reads transactional row updates from MySQL inventory tables, "
        "syncing stock balances into an Elasticsearch cluster for full-text faceted storefront search. "
        "Optimistic locking controls prevent overselling during flash-sale traffic spikes.\n"
    ),
    (
        "Autonomous Flight Telemetry & Navigation v2.4:\n"
        "Unmanned aerial vehicles stream IMU and GPS positioning data over encrypted UHF radio datalinks. "
        "An onboard flight computer executes Kalman filter fusion to estimate attitude, heading, and wind drift. "
        "Waypoint trajectories are dynamically recalculated to avoid geofenced restricted airspace "
        "and maintain redundant fail-safe return-to-launch procedures.\n"
    ),
    (
        "Smart Grid Power Distribution Monitoring v5.0:\n"
        "Substation phasor measurement units sample AC voltage and current waveforms at 60 Hz synchronized via GPS clocks. "
        "SCADA concentrators aggregate synchrophasor vectors, detecting phase angle instability and transmission line faults. "
        "Automated reclosers isolate damaged feeder sections, rerouting power flows across alternate distribution loops.\n"
    ),
]

TASKS = [
    "Task 1: Generate a concise 2-sentence executive summary of the system architecture.",
    "Task 2: List the top 3 potential single points of failure and specific mitigations.",
    "Task 3: Describe the primary database index and partitioning strategy.",
    "Task 4: Outline horizontal autoscaling triggers based on queue lag and CPU metrics.",
    "Task 5: Propose a continuous integration smoke test checklist for this pipeline.",
]


def run_experiment(
    base_url: str = "http://127.0.0.1:11434",
    model: str = "qwen2.5:1.5b",
    num_ctx: int = 4096,
    num_predict: int = 64,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" Starting KV-Cache Prefix Reuse Controlled Experiment")
    print(f" Model: {model} | Context: {num_ctx} | Endpoint: {base_url}")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"=======================================================\n")

    client = httpx.Client(timeout=120.0)

    # 1. Warm-up Request (ensures model is loaded into VRAM)
    print("Executing model warm-up request...")
    warmup_res = client.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": COMMON_SYSTEM_INSTRUCTION,
            "prompt": "Warm-up inference: respond with 'Ready.'",
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": 16, "temperature": 0.2},
        },
    )
    if warmup_res.status_code != 200:
        raise RuntimeError(f"Warm-up failed: {warmup_res.text}")
    print("Warm-up complete. Model resident in VRAM.\n")

    # 2. Experiment Group A: Stable Common Prefix
    print("--- Running Group A: Stable Common Prefix (5 Sequential Tasks) ---")
    stable_results = []
    for i, task in enumerate(TASKS):
        full_prompt = f"{COMMON_PREFIX_SPECIFICATION}\n{task}"
        start_clock = time.perf_counter()
        resp = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": COMMON_SYSTEM_INSTRUCTION,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.2},
            },
        )
        elapsed_ms = round((time.perf_counter() - start_clock) * 1000, 2)
        assert resp.status_code == 200, f"Request failed: {resp.text}"
        data = resp.json()

        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_cached = data.get("prompt_eval_cached_count", 0)
        prompt_eval_dur_ns = data.get("prompt_eval_duration", 0)
        prompt_eval_dur_ms = round(prompt_eval_dur_ns / 1e6, 2)
        eval_count = data.get("eval_count", 0)
        eval_dur_ns = data.get("eval_duration", 0)
        prompt_tps = round(prompt_eval_count / (prompt_eval_dur_ns / 1e9), 2) if prompt_eval_dur_ns > 0 else 0

        stable_results.append({
            "task_index": i + 1,
            "task": task[:50],
            "wall_clock_ms": elapsed_ms,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": prompt_eval_cached,
            "prompt_eval_duration_ms": prompt_eval_dur_ms,
            "prompt_tokens_per_second": prompt_tps,
            "eval_count": eval_count,
            "cache_hit_ratio": round(prompt_eval_cached / prompt_eval_count, 4) if prompt_eval_count > 0 else 0,
        })
        print(f"  Task {i+1}: Cached={prompt_eval_cached}/{prompt_eval_count} tokens | "
              f"Prefill: {prompt_eval_dur_ms}ms ({prompt_tps} tok/s) | Wall: {elapsed_ms}ms")

    time.sleep(1.0)

    # 3. Experiment Group B: Varied Control Prefixes (Different Prefix Each Task)
    print("\n--- Running Group B: Varied Control Prefixes (5 Sequential Tasks) ---")
    varied_results = []
    for i, task in enumerate(TASKS):
        full_prompt = f"{VARIED_CONTROL_PREFIXES[i]}\n{task}"
        start_clock = time.perf_counter()
        resp = client.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": COMMON_SYSTEM_INSTRUCTION,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.2},
            },
        )
        elapsed_ms = round((time.perf_counter() - start_clock) * 1000, 2)
        assert resp.status_code == 200, f"Request failed: {resp.text}"
        data = resp.json()

        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_cached = data.get("prompt_eval_cached_count", 0)
        prompt_eval_dur_ns = data.get("prompt_eval_duration", 0)
        prompt_eval_dur_ms = round(prompt_eval_dur_ns / 1e6, 2)
        eval_count = data.get("eval_count", 0)
        prompt_tps = round(prompt_eval_count / (prompt_eval_dur_ns / 1e9), 2) if prompt_eval_dur_ns > 0 else 0

        varied_results.append({
            "task_index": i + 1,
            "task": task[:50],
            "wall_clock_ms": elapsed_ms,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": prompt_eval_cached,
            "prompt_eval_duration_ms": prompt_eval_dur_ms,
            "prompt_tokens_per_second": prompt_tps,
            "eval_count": eval_count,
            "cache_hit_ratio": round(prompt_eval_cached / prompt_eval_count, 4) if prompt_eval_count > 0 else 0,
        })
        print(f"  Task {i+1}: Cached={prompt_eval_cached}/{prompt_eval_count} tokens | "
              f"Prefill: {prompt_eval_dur_ms}ms ({prompt_tps} tok/s) | Wall: {elapsed_ms}ms")

    # 4. Statistical Summary (ignoring task 1 cold prefill for reuse analysis)
    stable_subsequent = stable_results[1:]
    varied_subsequent = varied_results[1:]

    avg_stable_prefill = statistics.mean([r["prompt_eval_duration_ms"] for r in stable_subsequent])
    avg_varied_prefill = statistics.mean([r["prompt_eval_duration_ms"] for r in varied_subsequent])
    avg_stable_cached = statistics.mean([r["prompt_eval_cached_count"] for r in stable_subsequent])
    avg_varied_cached = statistics.mean([r["prompt_eval_cached_count"] for r in varied_subsequent])

    avg_stable_tps = statistics.mean([r["prompt_tokens_per_second"] for r in stable_subsequent])
    avg_varied_tps = statistics.mean([r["prompt_tokens_per_second"] for r in varied_subsequent])

    prefill_speedup = round(avg_varied_prefill / avg_stable_prefill, 2) if avg_stable_prefill > 0 else 1.0

    print(f"\n=======================================================")
    print(f" EXPERIMENT RESULTS SUMMARY (Tasks 2-5 Average)")
    print(f"=======================================================")
    print(f" Stable Prefix Average Cached Tokens : {avg_stable_cached:.1f} tokens")
    print(f" Varied Prefix Average Cached Tokens : {avg_varied_cached:.1f} tokens")
    print(f" Stable Prefix Average Prefill Time : {avg_stable_prefill:.2f} ms ({avg_stable_tps:.1f} tok/s)")
    print(f" Varied Prefix Average Prefill Time : {avg_varied_prefill:.2f} ms ({avg_varied_tps:.1f} tok/s)")
    print(f" Observed Prefill Speedup Factor    : {prefill_speedup}x")
    print(f"=======================================================\n")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "stable_prefix_runs": stable_results,
        "varied_control_runs": varied_results,
        "summary": {
            "stable_avg_cached_tokens": round(avg_stable_cached, 1),
            "varied_avg_cached_tokens": round(avg_varied_cached, 1),
            "stable_avg_prefill_ms": round(avg_stable_prefill, 2),
            "varied_avg_prefill_ms": round(avg_varied_prefill, 2),
            "stable_avg_prompt_tokens_per_second": round(avg_stable_tps, 1),
            "varied_avg_prompt_tokens_per_second": round(avg_varied_tps, 1),
            "prefill_speedup_factor": prefill_speedup,
            "authoritative_native_counter": "prompt_eval_cached_count",
        },
    }

    report_path = Path("D:/Projects/Argus/docs/reports/prefix_reuse_experiment.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved prefix reuse experiment results to {report_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV-cache prefix reuse controlled experiment")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama API base URL")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model name")
    parser.add_argument("--ctx", type=int, default=4096, help="Context length")
    args = parser.parse_args()

    run_experiment(base_url=args.base_url, model=args.model, num_ctx=args.ctx)
