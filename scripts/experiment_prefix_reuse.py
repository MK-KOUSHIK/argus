"""Controlled Empirical Experiment for Ollama KV-Cache Prefix Reuse.

Evaluates whether identical prompt prefixes yield measurable prefill time reduction
and inspects Ollama's native prompt_eval_cached_count counter.
Compares a Stable Prefix group against a Varied Prefix control group under
identical model residency, context size (4096), and hardware conditions.
"""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
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

# 5 Varied prefixes for the control group (calibrated to comparable token footprint: ~180-190 tokens)
VARIED_CONTROL_PREFIXES = [
    (
        "Clinical Laboratory Management System Specification v1.0:\n"
        "Specimen tracking begins at outpatient phlebotomy stations where barcoded tubes are scanned into HL7 interfaces. "
        "Automated chemistry analyzers run spectrophotometric assays, streaming discrete result flags across a DICOM bus. "
        "A secure microservice validates panic values, routes alert notifications to on-call pathologists, "
        "and archives encrypted audit logs to HIPAA-compliant cold storage with immutable retention schedules. "
        "High-throughput batch workers process quality-control runs and flag instrument calibration drift.\n"
    ),
    (
        "Automated High-Frequency Trading Engine Specification v8.1:\n"
        "Market data feeds arrive via ultra-low latency UDP multicast directly from exchange colocation gateways. "
        "FPGA-accelerated network cards demultiplex ITCH book feeds, maintaining a microsecond-level limit order book. "
        "Quantitative pricing models compute dynamic option Greeks, evaluating risk limits across margin accounts "
        "and routing cancellation orders across FIX protocol sessions before exchange circuit-breakers trigger. "
        "Real-time surveillance modules track cross-market arbitrage opportunities and compliance thresholds.\n"
    ),
    (
        "E-Commerce Real-Time Catalog & Distributed Inventory Pipeline v3.5:\n"
        "Warehouse inventory updates stream from handheld barcode scanners across regional fulfillment nodes. "
        "A change-data-capture Kafka connector reads transactional row updates from MySQL inventory tables, "
        "syncing stock balances into an Elasticsearch cluster for full-text faceted storefront search. "
        "Optimistic locking controls prevent overselling during flash-sale traffic spikes. "
        "Asynchronous inventory reservation queues ensure eventual consistency across multi-region edge caches.\n"
    ),
    (
        "Autonomous Flight Telemetry & Drone Navigation Specification v2.4:\n"
        "Unmanned aerial vehicles stream IMU and GPS positioning data over encrypted UHF radio datalinks. "
        "An onboard flight computer executes Kalman filter fusion to estimate attitude, heading, and wind drift. "
        "Waypoint trajectories are dynamically recalculated to avoid geofenced restricted airspace "
        "and maintain redundant fail-safe return-to-launch procedures. "
        "Telemetry ingestion pipelines log battery state-of-health and motor RPM to an inflight black-box recorder.\n"
    ),
    (
        "Smart Electrical Grid Power Distribution & Synchrophasor Monitoring v5.0:\n"
        "Substation phasor measurement units sample AC voltage and current waveforms at 60 Hz synchronized via GPS clocks. "
        "SCADA concentrators aggregate synchrophasor vectors, detecting phase angle instability and transmission line faults. "
        "Automated reclosers isolate damaged feeder sections, rerouting power flows across alternate distribution loops. "
        "Central grid dispatch operators monitor reactive power reserves and execute automated load-shedding protocols.\n"
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
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_report_path = report_path or (
        Path(__file__).resolve().parent.parent / "docs" / "reports" / "prefix_reuse_experiment.json"
    )
    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

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
        time.sleep(0.5)
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

        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_cached = data.get("prompt_eval_cached_count")
        prompt_eval_dur_ns = data.get("prompt_eval_duration")
        eval_count = data.get("eval_count")
        eval_dur_ns = data.get("eval_duration")

        prompt_eval_dur_ms = round(prompt_eval_dur_ns / 1e6, 2) if prompt_eval_dur_ns else None
        prompt_eval_dur_s = (prompt_eval_dur_ns / 1e9) if prompt_eval_dur_ns else None

        has_native_cached_counter = (prompt_eval_cached is not None)
        uncached_tokens = (
            max(0, prompt_eval_count - prompt_eval_cached)
            if (has_native_cached_counter and prompt_eval_count is not None)
            else None
        )
        cache_hit_ratio = (
            round(prompt_eval_cached / prompt_eval_count, 4)
            if (has_native_cached_counter and prompt_eval_count and prompt_eval_count > 0)
            else None
        )

        effective_prompt_tps = (
            round(prompt_eval_count / prompt_eval_dur_s, 2)
            if (prompt_eval_dur_s and prompt_eval_count)
            else None
        )
        uncached_eval_tps = (
            round(uncached_tokens / prompt_eval_dur_s, 2)
            if (prompt_eval_dur_s and uncached_tokens is not None and prompt_eval_dur_s > 0)
            else None
        )

        stable_results.append({
            "task_index": i + 1,
            "task": task[:50],
            "wall_clock_ms": elapsed_ms,
            "native_cached_counter_available": has_native_cached_counter,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": prompt_eval_cached,
            "uncached_evaluated_tokens": uncached_tokens,
            "cache_hit_ratio": cache_hit_ratio,
            "prompt_eval_duration_ms": prompt_eval_dur_ms,
            "effective_prompt_tokens_per_second": effective_prompt_tps,
            "uncached_eval_tokens_per_second": uncached_eval_tps,
            "eval_count": eval_count,
        })
        print(
            f"  Task {i+1}: TotalPrompt={prompt_eval_count} | Cached={prompt_eval_cached} | "
            f"Uncached={uncached_tokens} | Ratio={cache_hit_ratio} | "
            f"Prefill: {prompt_eval_dur_ms}ms (Effective: {effective_prompt_tps} tok/s) | Wall: {elapsed_ms}ms"
        )

    time.sleep(1.0)

    # 3. Experiment Group B: Varied Control Prefixes (Different Prefix Each Task)
    print("\n--- Running Group B: Varied Control Prefixes (5 Sequential Tasks) ---")
    varied_results = []
    for i, task in enumerate(TASKS):
        time.sleep(0.5)
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

        prompt_eval_count = data.get("prompt_eval_count")
        prompt_eval_cached = data.get("prompt_eval_cached_count")
        prompt_eval_dur_ns = data.get("prompt_eval_duration")
        eval_count = data.get("eval_count")

        prompt_eval_dur_ms = round(prompt_eval_dur_ns / 1e6, 2) if prompt_eval_dur_ns else None
        prompt_eval_dur_s = (prompt_eval_dur_ns / 1e9) if prompt_eval_dur_ns else None

        has_native_cached_counter = (prompt_eval_cached is not None)
        uncached_tokens = (
            max(0, prompt_eval_count - prompt_eval_cached)
            if (has_native_cached_counter and prompt_eval_count is not None)
            else None
        )
        cache_hit_ratio = (
            round(prompt_eval_cached / prompt_eval_count, 4)
            if (has_native_cached_counter and prompt_eval_count and prompt_eval_count > 0)
            else None
        )

        effective_prompt_tps = (
            round(prompt_eval_count / prompt_eval_dur_s, 2)
            if (prompt_eval_dur_s and prompt_eval_count)
            else None
        )
        uncached_eval_tps = (
            round(uncached_tokens / prompt_eval_dur_s, 2)
            if (prompt_eval_dur_s and uncached_tokens is not None and prompt_eval_dur_s > 0)
            else None
        )

        varied_results.append({
            "task_index": i + 1,
            "task": task[:50],
            "wall_clock_ms": elapsed_ms,
            "native_cached_counter_available": has_native_cached_counter,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": prompt_eval_cached,
            "uncached_evaluated_tokens": uncached_tokens,
            "cache_hit_ratio": cache_hit_ratio,
            "prompt_eval_duration_ms": prompt_eval_dur_ms,
            "effective_prompt_tokens_per_second": effective_prompt_tps,
            "uncached_eval_tokens_per_second": uncached_eval_tps,
            "eval_count": eval_count,
        })
        print(
            f"  Task {i+1}: TotalPrompt={prompt_eval_count} | Cached={prompt_eval_cached} | "
            f"Uncached={uncached_tokens} | Ratio={cache_hit_ratio} | "
            f"Prefill: {prompt_eval_dur_ms}ms (Effective: {effective_prompt_tps} tok/s) | Wall: {elapsed_ms}ms"
        )

    # 4. Statistical Summary (ignoring task 1 cold prefill for reuse analysis)
    stable_subsequent = stable_results[1:]
    varied_subsequent = varied_results[1:]

    avg_stable_prefill = statistics.mean([r["prompt_eval_duration_ms"] for r in stable_subsequent if r["prompt_eval_duration_ms"]])
    avg_varied_prefill = statistics.mean([r["prompt_eval_duration_ms"] for r in varied_subsequent if r["prompt_eval_duration_ms"]])

    stable_cached_vals = [r["prompt_eval_cached_count"] for r in stable_subsequent if r["prompt_eval_cached_count"] is not None]
    varied_cached_vals = [r["prompt_eval_cached_count"] for r in varied_subsequent if r["prompt_eval_cached_count"] is not None]

    avg_stable_cached = statistics.mean(stable_cached_vals) if stable_cached_vals else None
    avg_varied_cached = statistics.mean(varied_cached_vals) if varied_cached_vals else None

    stable_tps_vals = [r["effective_prompt_tokens_per_second"] for r in stable_subsequent if r["effective_prompt_tokens_per_second"]]
    varied_tps_vals = [r["effective_prompt_tokens_per_second"] for r in varied_subsequent if r["effective_prompt_tokens_per_second"]]

    avg_stable_tps = statistics.mean(stable_tps_vals) if stable_tps_vals else None
    avg_varied_tps = statistics.mean(varied_tps_vals) if varied_tps_vals else None

    prefill_speedup = (
        round(avg_varied_prefill / avg_stable_prefill, 2)
        if (avg_stable_prefill and avg_stable_prefill > 0)
        else 1.0
    )

    print(f"\n=======================================================")
    print(f" EXPERIMENT RESULTS SUMMARY (Tasks 2-5 Average)")
    print(f"=======================================================")
    print(f" Stable Prefix Average Cached Tokens : {avg_stable_cached}")
    print(f" Varied Prefix Average Cached Tokens : {avg_varied_cached}")
    print(f" Stable Prefix Average Prefill Time : {avg_stable_prefill:.2f} ms (Effective: {avg_stable_tps:.1f} tok/s)")
    print(f" Varied Prefix Average Prefill Time : {avg_varied_prefill:.2f} ms (Effective: {avg_varied_tps:.1f} tok/s)")
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
            "stable_avg_cached_tokens": round(avg_stable_cached, 1) if avg_stable_cached is not None else None,
            "varied_avg_cached_tokens": round(avg_varied_cached, 1) if avg_varied_cached is not None else None,
            "stable_avg_prefill_ms": round(avg_stable_prefill, 2),
            "varied_avg_prefill_ms": round(avg_varied_prefill, 2),
            "stable_avg_effective_prompt_tokens_per_second": round(avg_stable_tps, 1) if avg_stable_tps else None,
            "varied_avg_effective_prompt_tokens_per_second": round(avg_varied_tps, 1) if avg_varied_tps else None,
            "prefill_speedup_factor": prefill_speedup,
            "authoritative_native_counter": "prompt_eval_cached_count",
            "metric_definitions": {
                "prompt_eval_count": "Total tokens evaluated during prompt prefill",
                "prompt_eval_cached_count": "Authoritative native tokens retrieved from KV-cache without recomputation",
                "uncached_evaluated_tokens": "prompt_eval_count - prompt_eval_cached_count",
                "cache_hit_ratio": "prompt_eval_cached_count / prompt_eval_count",
                "effective_prompt_tokens_per_second": "Total prompt tokens divided by prompt evaluation time (effective throughput)",
                "uncached_eval_tokens_per_second": "Newly evaluated uncached tokens divided by prompt evaluation time",
            },
        },
    }

    with open(resolved_report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved prefix reuse experiment results to {resolved_report_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV-cache prefix reuse controlled experiment")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama API base URL")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Model name")
    parser.add_argument("--ctx", type=int, default=4096, help="Context length")
    parser.add_argument("--report-path", default=None, help="Output path for results JSON")
    args = parser.parse_args()

    rep_path = Path(args.report_path) if args.report_path else None
    run_experiment(base_url=args.base_url, model=args.model, num_ctx=args.ctx, report_path=rep_path)
