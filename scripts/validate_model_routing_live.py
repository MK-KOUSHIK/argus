"""Live Validation Script for Argus Local-First Model Routing (Phase 5).

Executes real prompt optimization through the live Ollama daemon using installed models:
- qwen2.5:1.5b (Small tier)
- qwen2.5:7b (Medium tier)

Evaluates 4 empirical scenarios:
1. Simple task routed to lower-resource model (qwen2.5:1.5b)
2. Complex task routed to stronger model (qwen2.5:7b) under safe resource conditions
3. Complex task falling back to smaller model under resource pressure
4. Manual model selection override (explicit model execution)

Records real native Ollama telemetry, timing, and resource observations without simulation.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.services.hardware_telemetry import HardwareTelemetryProvider
from app.services.ollama_service import OllamaService
from app.services.routing.registry import LocalModelRegistry


async def run_live_validation():
    print("=" * 80)
    print("ARGUS PHASE 5: LIVE LOCAL MODEL ROUTING VALIDATION")
    print("=" * 80)

    # 1. Verify Ollama Connection & Discovered Models
    connected, err = await LocalModelRegistry.refresh(force=True)
    if not connected:
        print(f"BLOCKED: Cannot connect to Ollama daemon at {settings.OLLAMA_BASE_URL}: {err}")
        sys.exit(1)

    models = await LocalModelRegistry.get_all_models()
    model_names = [m.name for m in models]
    print(f"Connected to Ollama at {settings.OLLAMA_BASE_URL}")
    print(f"Discovered installed models ({len(models)}): {model_names}")

    has_15b = any("1.5b" in name.lower() for name in model_names)
    has_7b = any("7b" in name.lower() for name in model_names)

    if not has_15b or not has_7b:
        print(f"BLOCKED: Required models ('qwen2.5:1.5b' and 'qwen2.5:7b') not both present. Found: {model_names}")
        sys.exit(1)

    # Live hardware state
    gpu = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
    ram = HardwareTelemetryProvider.get_system_ram()
    print(f"Live GPU VRAM: {gpu.get('free_mb') if gpu else 'N/A'} MiB free / {gpu.get('total_mb') if gpu else 'N/A'} MiB total")
    print(f"Live System RAM: {ram.get('free_mb')} MiB free / {ram.get('total_mb')} MiB total")
    print("-" * 80)

    results = []

    # --------------------------------------------------------------------------
    # Scenario 1: Simple Task -> Lower-resource model (qwen2.5:1.5b)
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 1: Simple task -> Auto-routed to lower-resource model (qwen2.5:1.5b) ---")
    s1_prompt = "Summarize the key differences between synchronous and asynchronous I/O in three brief sentences."
    req1 = OptimizeRequest(prompt=s1_prompt, mode="balanced")

    t0 = time.perf_counter()
    resp1: OptimizeResponse = await OllamaService.optimize_prompt(req1)
    dur1 = time.perf_counter() - t0

    assert resp1.routing is not None
    print(f"Selected Model: {resp1.routing.selected_model} (Tier: {resp1.routing.target_model_tier})")
    print(f"Actual Executed Model: {resp1.model}")
    print(f"Complexity Score: {resp1.routing.complexity.heuristic_score} (Category: {resp1.routing.complexity.task_category})")
    print(f"Routing Reason: {resp1.routing.reason}")
    print(f"Inference Latency: {resp1.execution_time_ms} ms (Total Wall: {dur1*1000:.1f} ms)")

    prim1 = resp1.inference_telemetry["calls"][0] if resp1.inference_telemetry and resp1.inference_telemetry.get("calls") else {}
    print(f"Native Telemetry -> Prompt Tokens: {prim1.get('prompt_eval_count')}, Generated: {prim1.get('eval_count')}, Done Reason: {prim1.get('done_reason')}")

    results.append({
        "scenario": "1_simple_task_small_model",
        "prompt": s1_prompt,
        "selected_model": resp1.routing.selected_model,
        "executed_model": resp1.model,
        "target_tier": resp1.routing.target_model_tier,
        "complexity_score": resp1.routing.complexity.heuristic_score,
        "routing_status": resp1.routing.routing_status,
        "prompt_eval_count": prim1.get("prompt_eval_count"),
        "eval_count": prim1.get("eval_count"),
        "latency_ms": resp1.execution_time_ms,
        "done_reason": prim1.get("done_reason"),
        "verified": resp1.model == "qwen2.5:1.5b" and resp1.routing.selected_model == "qwen2.5:1.5b",
    })

    # --------------------------------------------------------------------------
    # Scenario 2: Complex Task -> Stronger model (qwen2.5:7b)
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 2: Complex task -> Auto-routed to stronger model (qwen2.5:7b) ---")
    s2_prompt = (
        "Write an asynchronous Python data ingestion service using httpx and asyncio. "
        "Do not use requests or aiohttp. Parse incoming JSON, extract entities matching keys "
        "['user_id', 'transaction_value', 'timestamp'], validate non-negative values, "
        "handle transient network retries with exponential backoff, and save aggregated dictionary to metrics.json."
    )
    req2 = OptimizeRequest(prompt=s2_prompt, mode="structure")

    t0 = time.perf_counter()
    resp2: OptimizeResponse = await OllamaService.optimize_prompt(req2)
    dur2 = time.perf_counter() - t0

    assert resp2.routing is not None
    print(f"Selected Model: {resp2.routing.selected_model} (Tier: {resp2.routing.target_model_tier})")
    print(f"Actual Executed Model: {resp2.model}")
    print(f"Complexity Score: {resp2.routing.complexity.heuristic_score} (Category: {resp2.routing.complexity.task_category})")
    print(f"Complexity Factors: {resp2.routing.complexity.complexity_factors}")
    print(f"Routing Reason: {resp2.routing.reason}")
    print(f"Inference Latency: {resp2.execution_time_ms} ms (Total Wall: {dur2*1000:.1f} ms)")

    prim2 = resp2.inference_telemetry["calls"][0] if resp2.inference_telemetry and resp2.inference_telemetry.get("calls") else {}
    print(f"Native Telemetry -> Prompt Tokens: {prim2.get('prompt_eval_count')}, Generated: {prim2.get('eval_count')}, Done Reason: {prim2.get('done_reason')}")

    results.append({
        "scenario": "2_complex_task_medium_model",
        "prompt": s2_prompt,
        "selected_model": resp2.routing.selected_model,
        "executed_model": resp2.model,
        "target_tier": resp2.routing.target_model_tier,
        "complexity_score": resp2.routing.complexity.heuristic_score,
        "routing_status": resp2.routing.routing_status,
        "prompt_eval_count": prim2.get("prompt_eval_count"),
        "eval_count": prim2.get("eval_count"),
        "latency_ms": resp2.execution_time_ms,
        "done_reason": prim2.get("done_reason"),
        "verified": resp2.model == "qwen2.5:7b" and resp2.routing.selected_model == "qwen2.5:7b",
    })

    # --------------------------------------------------------------------------
    # Scenario 3: Complex Task under Resource Pressure -> Fallback to small model
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 3: Complex task under resource pressure -> Safe fallback to qwen2.5:1.5b ---")
    # Trigger resource pressure by requiring more free host RAM than currently available (~2200 MiB free)
    original_ram_margin = settings.ROUTING_RAM_SAFETY_MARGIN_MB
    try:
        settings.ROUTING_RAM_SAFETY_MARGIN_MB = 6000
        req3 = OptimizeRequest(prompt=s2_prompt, mode="balanced")

        t0 = time.perf_counter()
        resp3: OptimizeResponse = await OllamaService.optimize_prompt(req3)
        dur3 = time.perf_counter() - t0

        assert resp3.routing is not None
        print(f"Selected Model: {resp3.routing.selected_model} (Tier: {resp3.routing.target_model_tier})")
        print(f"Actual Executed Model: {resp3.model}")
        print(f"Routing Status: {resp3.routing.routing_status} (is_fallback: {resp3.routing.is_fallback})")
        print(f"Routing Reason: {resp3.routing.reason}")
        print(f"Inference Latency: {resp3.execution_time_ms} ms")

        prim3 = resp3.inference_telemetry["calls"][0] if resp3.inference_telemetry and resp3.inference_telemetry.get("calls") else {}
        print(f"Native Telemetry -> Prompt Tokens: {prim3.get('prompt_eval_count')}, Generated: {prim3.get('eval_count')}")

        results.append({
            "scenario": "3_resource_constrained_fallback",
            "prompt": s2_prompt,
            "selected_model": resp3.routing.selected_model,
            "executed_model": resp3.model,
            "target_tier": resp3.routing.target_model_tier,
            "routing_status": resp3.routing.routing_status,
            "is_fallback": resp3.routing.is_fallback,
            "prompt_eval_count": prim3.get("prompt_eval_count"),
            "eval_count": prim3.get("eval_count"),
            "latency_ms": resp3.execution_time_ms,
            "verified": resp3.model == "qwen2.5:1.5b" and resp3.routing.is_fallback is True,
        })
    finally:
        settings.ROUTING_RAM_SAFETY_MARGIN_MB = original_ram_margin

    # --------------------------------------------------------------------------
    # Scenario 4: Manual Model Selection Override
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 4: Manual model selection override (forcing qwen2.5:7b) ---")
    s4_prompt = "List 3 tips for better sleep."
    req4 = OptimizeRequest(prompt=s4_prompt, requested_model="qwen2.5:7b", mode="concise")

    t0 = time.perf_counter()
    resp4: OptimizeResponse = await OllamaService.optimize_prompt(req4)
    dur4 = time.perf_counter() - t0

    assert resp4.routing is not None
    print(f"Selected Model: {resp4.routing.selected_model} (Tier: {resp4.routing.target_model_tier})")
    print(f"Actual Executed Model: {resp4.model}")
    print(f"Routing Status: {resp4.routing.routing_status}")
    print(f"Routing Reason: {resp4.routing.reason}")
    print(f"Inference Latency: {resp4.execution_time_ms} ms")

    prim4 = resp4.inference_telemetry["calls"][0] if resp4.inference_telemetry and resp4.inference_telemetry.get("calls") else {}
    print(f"Native Telemetry -> Prompt Tokens: {prim4.get('prompt_eval_count')}, Generated: {prim4.get('eval_count')}")

    results.append({
        "scenario": "4_manual_override_7b",
        "prompt": s4_prompt,
        "selected_model": resp4.routing.selected_model,
        "executed_model": resp4.model,
        "target_tier": resp4.routing.target_model_tier,
        "routing_status": resp4.routing.routing_status,
        "prompt_eval_count": prim4.get("prompt_eval_count"),
        "eval_count": prim4.get("eval_count"),
        "latency_ms": resp4.execution_time_ms,
        "verified": resp4.model == "qwen2.5:7b" and resp4.routing.routing_status == "manual_override",
    })

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    all_passed = True
    for r in results:
        status_str = "VERIFIED" if r["verified"] else "NOT VERIFIED"
        if not r["verified"]:
            all_passed = False
        print(f"[{status_str}] {r['scenario']}: Routed to {r['selected_model']}, Executed by {r['executed_model']} (Tokens: prompt={r['prompt_eval_count']}, gen={r['eval_count']}, Latency: {r['latency_ms']} ms)")

    # Save summary report JSON
    report_path = Path(__file__).resolve().parent.parent / "docs" / "reports" / "model_routing_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")

    if all_passed:
        print("\nOVERALL STATUS: ALL 4 REAL SCENARIOS VERIFIED WITH LIVE OLLAMA INFERENCE!")
    else:
        print("\nOVERALL STATUS: ONE OR MORE SCENARIOS FAILED VERIFICATION.")


if __name__ == "__main__":
    asyncio.run(run_live_validation())

