"""Empirical Output Budget Control Benchmark for Argus (Phase 4.1).

Executes genuine local Ollama inference using the configured local model (qwen2.5:1.5b)
to compare:
1. Fixed Baseline: num_predict = 512
2. Adaptive Output Budget: num_predict allocated dynamically per task/constraints

Measures real native telemetry:
- eval_count (tokens generated)
- eval_duration_ms (generation latency)
- generation_tokens_per_second (decode rate)
- done_reason ("stop" natural EOS vs "length" budget ceiling)
- budget_utilization_percent
- selected_num_ctx (tier snapping context size)
- KV-cache prefix hits (prompt_eval_cached_count)

Saves raw unmanipulated measurements to docs/reports/output_budget_benchmark.json.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest
from app.services.ollama_service import OllamaService
from app.services.output_budget_controller import OutputBudgetController


BENCHMARK_PROMPTS = [
    {
        "id": "word_limit_summary",
        "name": "Explicit Word Limit Summarization",
        "prompt": "Summarize the key differences between synchronous and asynchronous programming in Python in under 50 words.",
        "mode": "balanced",
        "description": "Short answer with explicit 'under 50 words' constraint.",
    },
    {
        "id": "single_sentence_qa",
        "name": "Single Sentence Direct Explanation",
        "prompt": "Explain why database indexes improve query performance in one sentence.",
        "mode": "concise",
        "description": "Ultra-concise answer with explicit 'in one sentence' constraint and concise mode.",
    },
    {
        "id": "json_entity_extraction",
        "name": "Structured JSON Entity Extraction",
        "prompt": "Extract the customer name, order ID, product list, and grand total into a compact JSON schema from: Receipt #8491 for customer Alice Smith purchasing Widget A ($10) and Widget B ($25). Total: $35.00.",
        "mode": "balanced",
        "description": "Extraction task requesting structured JSON deliverable.",
    },
    {
        "id": "code_generation",
        "name": "Python Code Implementation",
        "prompt": "Write a Python generator function to stream Fibonacci numbers up to n with docstrings and type hints.",
        "mode": "balanced",
        "description": "Coding task requiring complete functional code deliverable.",
    },
    {
        "id": "structured_tradeoff_analysis",
        "name": "Structured Technical Comparison",
        "prompt": "Analyze the latency, memory, and persistence trade-offs between Redis and Memcached, formatted with clear markdown headings and a comparison checklist.",
        "mode": "structure",
        "description": "Analysis task in structure mode requiring Markdown hierarchy and multi-point comparison.",
    },
]


async def run_benchmark():
    print("=" * 70)
    print("Argus Output Budget Control — Empirical Live Benchmark")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Ollama Endpoint: {settings.OLLAMA_BASE_URL}")
    print(f"Target Model: {settings.OLLAMA_MODEL}")
    print(f"KV-Cache Type: {settings.OLLAMA_KV_CACHE_TYPE}")
    print(f"Flash Attention: {settings.OLLAMA_FLASH_ATTENTION}")
    print("=" * 70)

    # 1. Verify Ollama Status
    status = await OllamaService.check_status()
    if not status.connected or not status.model_installed:
        print(f"ERROR: Ollama server not ready: {status.error}")
        sys.exit(1)
    print(f"Verified Ollama connection. Available models: {', '.join(status.available_models)}\n")

    # Warmup call
    print("Executing warmup inference call to ensure model weights and KV cache are loaded...")
    warmup_req = OptimizeRequest(
        prompt="Warmup prompt to initialize model and KV buffers.",
        mode="balanced",
        target_model="gpt-4o",
    )
    warmup_res = await OllamaService.optimize_prompt(warmup_req)
    print(f"Warmup complete. Latency: {warmup_res.execution_time_ms} ms\n")

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"[{idx}/{len(BENCHMARK_PROMPTS)}] Running Benchmark: {item['name']}")
        print(f"  Prompt: \"{item['prompt'][:65]}...\"")
        print(f"  Mode: {item['mode']}")

        # Condition 1: Fixed Baseline (num_predict = 512)
        print("  -> Condition 1: Fixed Baseline (num_predict=512)...")
        settings.OUTPUT_BUDGET_ENABLED = False
        baseline_req = OptimizeRequest(
            prompt=item["prompt"],
            mode=item["mode"],
            target_model="gpt-4o",
        )
        t0 = time.perf_counter()
        baseline_res = await OllamaService.optimize_prompt(baseline_req)
        baseline_wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        baseline_telemetry = (
            baseline_res.inference_telemetry["calls"][0]
            if baseline_res.inference_telemetry and baseline_res.inference_telemetry.get("calls")
            else {}
        )

        print(f"     Allocated: {baseline_telemetry.get('num_predict_limit')} tokens | "
              f"Generated: {baseline_telemetry.get('eval_count')} tokens | "
              f"Done: {baseline_telemetry.get('done_reason')} | "
              f"Latency: {baseline_res.execution_time_ms} ms")

        # Give a small pause between runs
        await asyncio.sleep(0.5)

        # Condition 2: Adaptive Output Budget
        print("  -> Condition 2: Adaptive Output Budget...")
        settings.OUTPUT_BUDGET_ENABLED = True
        settings.OUTPUT_BUDGET_POLICY = "adaptive"
        adaptive_req = OptimizeRequest(
            prompt=item["prompt"],
            mode=item["mode"],
            target_model="gpt-4o",
        )
        t0 = time.perf_counter()
        adaptive_res = await OllamaService.optimize_prompt(adaptive_req)
        adaptive_wall_ms = round((time.perf_counter() - t0) * 1000, 2)

        adaptive_telemetry = (
            adaptive_res.inference_telemetry["calls"][0]
            if adaptive_res.inference_telemetry and adaptive_res.inference_telemetry.get("calls")
            else {}
        )
        adaptive_ob = adaptive_res.output_budget

        print(f"     Allocated: {adaptive_ob.allocated_budget if adaptive_ob else 'N/A'} tokens | "
              f"Generated: {adaptive_ob.actual_generated_tokens if adaptive_ob else 'N/A'} tokens | "
              f"Done: {adaptive_ob.done_reason if adaptive_ob else 'N/A'} | "
              f"Utilization: {adaptive_ob.budget_utilization_percent if adaptive_ob else 'N/A'}% | "
              f"Latency: {adaptive_res.execution_time_ms} ms")
        if adaptive_ob:
            print(f"     Rationale: {adaptive_ob.reason}")

        # Compute savings and headroom
        baseline_budget = baseline_telemetry.get("num_predict_limit", 512)
        adaptive_budget = adaptive_ob.allocated_budget if adaptive_ob else 512
        budget_headroom_saved = max(0, baseline_budget - adaptive_budget)

        record = {
            "test_id": item["id"],
            "name": item["name"],
            "prompt": item["prompt"],
            "mode": item["mode"],
            "description": item["description"],
            "baseline": {
                "num_predict": baseline_budget,
                "selected_num_ctx": baseline_telemetry.get("selected_num_ctx"),
                "eval_count": baseline_telemetry.get("eval_count"),
                "eval_duration_ms": baseline_telemetry.get("eval_duration_ms"),
                "prompt_eval_count": baseline_telemetry.get("prompt_eval_count"),
                "prompt_eval_cached_count": baseline_telemetry.get("prompt_eval_cached_count"),
                "generation_tokens_per_second": baseline_telemetry.get("generation_tokens_per_second"),
                "done_reason": baseline_telemetry.get("done_reason"),
                "hit_budget_limit": baseline_telemetry.get("hit_budget_limit"),
                "budget_utilization_percent": baseline_telemetry.get("budget_utilization_percent"),
                "execution_time_ms": baseline_res.execution_time_ms,
                "wall_time_ms": baseline_wall_ms,
                "output_text": baseline_res.optimized_prompt,
            },
            "adaptive": {
                "num_predict": adaptive_budget,
                "selected_num_ctx": adaptive_telemetry.get("selected_num_ctx"),
                "eval_count": adaptive_ob.actual_generated_tokens if adaptive_ob else None,
                "eval_duration_ms": adaptive_ob.generation_duration_ms if adaptive_ob else None,
                "prompt_eval_count": adaptive_telemetry.get("prompt_eval_count"),
                "prompt_eval_cached_count": adaptive_telemetry.get("prompt_eval_cached_count"),
                "generation_tokens_per_second": adaptive_ob.generation_tokens_per_second if adaptive_ob else None,
                "done_reason": adaptive_ob.done_reason if adaptive_ob else None,
                "hit_budget_limit": adaptive_ob.hit_budget_limit if adaptive_ob else None,
                "budget_utilization_percent": adaptive_ob.budget_utilization_percent if adaptive_ob else None,
                "allocation_reason": adaptive_ob.reason if adaptive_ob else None,
                "task_category": adaptive_ob.task_category if adaptive_ob else None,
                "execution_time_ms": adaptive_res.execution_time_ms,
                "wall_time_ms": adaptive_wall_ms,
                "output_text": adaptive_res.optimized_prompt,
            },
            "comparison": {
                "budget_reduction_tokens": budget_headroom_saved,
                "budget_reduction_percent": round((budget_headroom_saved / baseline_budget) * 100, 1) if baseline_budget > 0 else 0,
                "generation_quality_preserved": (adaptive_res.validation_status == "passed"),
                "validation_status": adaptive_res.validation_status,
                "outcome": adaptive_res.outcome,
            },
        }
        results.append(record)
        print()
        await asyncio.sleep(0.5)

    # 3. Aggregate Summary
    total_baseline_budget = sum(r["baseline"]["num_predict"] for r in results)
    total_adaptive_budget = sum(r["adaptive"]["num_predict"] for r in results)
    total_budget_saved = total_baseline_budget - total_adaptive_budget
    avg_budget_reduction_pct = round((total_budget_saved / total_baseline_budget) * 100, 1) if total_baseline_budget > 0 else 0

    benchmark_payload = {
        "benchmark_metadata": {
            "title": "Argus Output Budget Control Empirical Benchmark",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": settings.OLLAMA_MODEL,
            "runtime_endpoint": settings.OLLAMA_BASE_URL,
            "kv_cache_type": settings.OLLAMA_KV_CACHE_TYPE,
            "flash_attention": settings.OLLAMA_FLASH_ATTENTION,
            "hardware_platform": "Windows AMD Ryzen 7 7735 / NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)",
            "total_test_prompts": len(results),
        },
        "aggregate_summary": {
            "total_baseline_budget_tokens": total_baseline_budget,
            "total_adaptive_budget_tokens": total_adaptive_budget,
            "total_budget_tokens_conserved": total_budget_saved,
            "aggregate_budget_reduction_percent": avg_budget_reduction_pct,
            "all_generations_valid": all(r["comparison"]["generation_quality_preserved"] for r in results),
        },
        "detailed_results": results,
    }

    # Save to docs/reports/output_budget_benchmark.json
    out_dir = REPO_ROOT / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output_budget_benchmark.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print("=" * 70)
    print("Benchmark Run Completed Successfully!")
    print(f"Raw measurements written to: {out_path}")
    print(f"Total Output Budget Conserved: {total_budget_saved} tokens ({avg_budget_reduction_pct}% reduction)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
