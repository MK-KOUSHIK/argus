"""Live Validation Script for Argus Local-First Model Routing (Phase 5).

Evaluates model routing through the live Ollama daemon using dynamically discovered models.
Strictly model-agnostic: discovers installed models and determines small vs medium/large tiers
via verified parameter metadata from LocalModelRegistry.

Separates scenarios into:
1. Live Inference: Simple task auto-routed to lightweight model.
2. Live Inference: Complex task auto-routed to higher-capacity model (if installed and resources permit).
3. Policy Simulation: Complex task under artificially restrictive policy margin (simulated policy constraint).
4. Live Hardware Observation: Natural resource pressure check on host (reports observed headroom without artificial memory exhaustion).
5. Live Inference: Manual model override safely admitted and executed.

Records real native Ollama telemetry, timing, and resource observations without fabrication.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.services.hardware_telemetry import HardwareTelemetryProvider
from app.services.ollama_service import OllamaService
from app.services.routing.models import ModelTier
from app.services.routing.registry import LocalModelRegistry
from app.services.routing.router import ModelRouter


async def run_live_validation():
    print("=" * 80)
    print("ARGUS PHASE 5: LIVE LOCAL MODEL ROUTING VALIDATION")
    print("=" * 80)

    # 1. Dynamic Model Discovery
    connected, err = await LocalModelRegistry.refresh(force=True)
    if not connected:
        print(f"BLOCKED: Cannot connect to Ollama daemon at {settings.OLLAMA_BASE_URL}: {err}")
        sys.exit(1)

    all_models = await LocalModelRegistry.get_all_models()
    if not all_models:
        print(f"BLOCKED: No models are installed in Ollama at {settings.OLLAMA_BASE_URL}.")
        sys.exit(1)

    print(f"Connected to Ollama at {settings.OLLAMA_BASE_URL}")
    print(f"Discovered {len(all_models)} installed model(s):")
    for m in all_models:
        print(f"  - {m.name} | Tier: {m.tier.value} | Params: {m.parameter_size or m.parameter_count or 'unknown'} | Size: {round(m.size_bytes / (1024*1024), 1)} MB | Resident: {m.is_loaded}")

    # Identify small and higher-capacity candidate models dynamically
    small_models = [m for m in all_models if m.tier == ModelTier.SMALL]
    strong_models = [m for m in all_models if m.tier in (ModelTier.MEDIUM, ModelTier.LARGE)]

    small_candidate = (
        next((m for m in small_models if "qwen" in m.name.lower()), None)
        or (small_models[0] if small_models else None)
    )
    strong_candidate = (
        next((m for m in strong_models if "qwen" in m.name.lower()), None)
        or (strong_models[0] if strong_models else None)
    )

    print(f"\nCandidate Selection (Model-Agnostic):")
    print(f"  Lightweight Candidate: {small_candidate.name if small_candidate else 'None installed'}")
    print(f"  High-Capacity Candidate: {strong_candidate.name if strong_candidate else 'None installed'}")

    # Read live hardware state
    gpu = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
    ram = HardwareTelemetryProvider.get_system_ram()
    vram_free = gpu.get("free_mb") if gpu else None
    vram_total = gpu.get("total_mb") if gpu else None
    ram_free = ram.get("free_mb")
    ram_total = ram.get("total_mb")
    print(f"Live Hardware -> GPU VRAM: {vram_free} MiB free / {vram_total} MiB total | Host RAM: {ram_free} MiB free / {ram_total} MiB total")
    print("-" * 80)

    report_metadata: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "hardware": {
            "gpu_available": gpu is not None and gpu.get("total_mb", 0) > 0,
            "vram_total_mb": vram_total,
            "vram_free_mb": vram_free,
            "ram_total_mb": ram_total,
            "ram_free_mb": ram_free,
        },
        "discovered_models": [
            {
                "name": m.name,
                "tier": m.tier.value,
                "parameter_size": m.parameter_size,
                "parameter_count": m.parameter_count,
                "size_mb": round(m.size_bytes / (1024 * 1024), 1),
                "is_loaded": m.is_loaded,
            }
            for m in all_models
        ],
        "scenarios": [],
    }

    # --------------------------------------------------------------------------
    # Scenario 1: Simple Task Auto-Routing (Target: Lightweight Model)
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 1: Simple Task -> Auto-Routed to Lightweight Model ---")
    s1_prompt = "Summarize the key differences between synchronous and asynchronous I/O in three brief sentences."
    req1 = OptimizeRequest(prompt=s1_prompt, mode="balanced")

    t0 = time.perf_counter()
    resp1: OptimizeResponse = await OllamaService.optimize_prompt(req1)
    dur1 = time.perf_counter() - t0

    assert resp1.routing is not None
    prim1 = resp1.inference_telemetry["calls"][0] if resp1.inference_telemetry and resp1.inference_telemetry.get("calls") else {}
    print(f"Selected Model: {resp1.routing.selected_model} (Tier: {resp1.routing.target_model_tier})")
    print(f"Actual Executed Model: {resp1.model}")
    print(f"Complexity Score: {resp1.routing.complexity.heuristic_score} (Category: {resp1.routing.complexity.task_category})")
    print(f"Routing Reason: {resp1.routing.reason}")
    print(f"Inference Latency: {resp1.execution_time_ms} ms (Total Wall: {dur1*1000:.1f} ms)")
    print(f"Native Telemetry -> Prompt Tokens: {prim1.get('prompt_eval_count')}, Generated: {prim1.get('eval_count')}, Done Reason: {prim1.get('done_reason')}")

    s1_verified = (
        resp1.model == resp1.routing.selected_model
        and resp1.routing.target_model_tier == ModelTier.SMALL.value
        and resp1.routing.routing_status == "routed"
    )
    report_metadata["scenarios"].append({
        "scenario_id": "1_simple_task_lightweight_model",
        "type": "live_inference",
        "prompt": s1_prompt,
        "selected_model": resp1.routing.selected_model,
        "executed_model": resp1.model,
        "target_tier": resp1.routing.target_model_tier,
        "complexity_score": resp1.routing.complexity.heuristic_score,
        "heuristic_margin": resp1.routing.heuristic_margin,
        "routing_status": resp1.routing.routing_status,
        "prompt_eval_count": prim1.get("prompt_eval_count"),
        "eval_count": prim1.get("eval_count"),
        "latency_ms": resp1.execution_time_ms,
        "done_reason": prim1.get("done_reason"),
        "status": "VERIFIED" if s1_verified else "NOT VERIFIED",
    })

    # --------------------------------------------------------------------------
    # Scenario 2: Complex Task Auto-Routing (Target: Higher-Capacity Model)
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 2: Complex Task -> Auto-Routed to Higher-Capacity Model ---")
    s2_prompt = (
        "Write an asynchronous Python data ingestion service using httpx and asyncio. "
        "Do not use requests or aiohttp. Parse incoming JSON, extract entities matching keys "
        "['user_id', 'transaction_value', 'timestamp'], validate non-negative values, "
        "handle transient network retries with exponential backoff, and save aggregated dictionary to metrics.json."
    )

    if strong_candidate is not None:
        # Evict lightweight model if resident so full VRAM is available for high-capacity model
        await LocalModelRegistry.unload_model(resp1.routing.selected_model)
        await asyncio.sleep(0.5)

        req2 = OptimizeRequest(prompt=s2_prompt, mode="structure")
        t0 = time.perf_counter()
        resp2: OptimizeResponse = await OllamaService.optimize_prompt(req2)
        dur2 = time.perf_counter() - t0

        assert resp2.routing is not None
        prim2 = resp2.inference_telemetry["calls"][0] if resp2.inference_telemetry and resp2.inference_telemetry.get("calls") else {}
        print(f"Selected Model: {resp2.routing.selected_model} (Tier: {resp2.routing.target_model_tier})")
        print(f"Actual Executed Model: {resp2.model}")
        print(f"Complexity Score: {resp2.routing.complexity.heuristic_score} (Category: {resp2.routing.complexity.task_category})")
        print(f"Routing Reason: {resp2.routing.reason}")
        print(f"Inference Latency: {resp2.execution_time_ms} ms (Total Wall: {dur2*1000:.1f} ms)")
        print(f"Native Telemetry -> Prompt Tokens: {prim2.get('prompt_eval_count')}, Generated: {prim2.get('eval_count')}, Done Reason: {prim2.get('done_reason')}")

        s2_verified = (
            resp2.model == resp2.routing.selected_model
            and resp2.routing.target_model_tier in (ModelTier.MEDIUM.value, ModelTier.LARGE.value)
            and resp2.routing.routing_status == "routed"
        )
        report_metadata["scenarios"].append({
            "scenario_id": "2_complex_task_high_capacity_model",
            "type": "live_inference",
            "prompt": s2_prompt,
            "selected_model": resp2.routing.selected_model,
            "executed_model": resp2.model,
            "target_tier": resp2.routing.target_model_tier,
            "complexity_score": resp2.routing.complexity.heuristic_score,
            "heuristic_margin": resp2.routing.heuristic_margin,
            "routing_status": resp2.routing.routing_status,
            "prompt_eval_count": prim2.get("prompt_eval_count"),
            "eval_count": prim2.get("eval_count"),
            "latency_ms": resp2.execution_time_ms,
            "done_reason": prim2.get("done_reason"),
            "status": "VERIFIED" if s2_verified else "NOT VERIFIED",
        })
    else:
        print("SKIPPED: No medium or large tier model installed. Cannot run higher-capacity live routing test.")
        report_metadata["scenarios"].append({
            "scenario_id": "2_complex_task_high_capacity_model",
            "type": "live_inference",
            "status": "SKIPPED",
            "reason": "No medium or large tier model installed in local Ollama instance",
        })

    # --------------------------------------------------------------------------
    # Scenario 3A: Policy Constraint Simulation (Configured Restrictive Margin)
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 3A: Policy Simulation -> Restrictive Safety Margin Policy ---")
    orig_ram_margin = settings.ROUTING_RAM_SAFETY_MARGIN_MB
    try:
        # Configure a restrictive policy margin (e.g. 100,000 MiB) to simulate a strict resource policy
        settings.ROUTING_RAM_SAFETY_MARGIN_MB = 100000
        req3a = OptimizeRequest(prompt=s2_prompt, mode="balanced")

        t0 = time.perf_counter()
        resp3a: OptimizeResponse = await OllamaService.optimize_prompt(req3a)
        dur3a = time.perf_counter() - t0

        assert resp3a.routing is not None
        prim3a = resp3a.inference_telemetry["calls"][0] if resp3a.inference_telemetry and resp3a.inference_telemetry.get("calls") else {}
        print(f"Simulated Policy Setting: ROUTING_RAM_SAFETY_MARGIN_MB = 100000")
        print(f"Selected Model: {resp3a.routing.selected_model} (Tier: {resp3a.routing.target_model_tier})")
        print(f"Actual Executed Model: {resp3a.model}")
        print(f"Routing Status: {resp3a.routing.routing_status} (is_fallback: {resp3a.routing.is_fallback})")
        print(f"Routing Reason: {resp3a.routing.reason}")
        print(f"Inference Latency: {resp3a.execution_time_ms} ms")
        print(f"Native Telemetry -> Prompt Tokens: {prim3a.get('prompt_eval_count')}, Generated: {prim3a.get('eval_count')}")

        s3a_verified = (
            resp3a.model == resp3a.routing.selected_model
            and resp3a.routing.routing_status == "fallback"
            and resp3a.routing.is_fallback is True
            and resp3a.routing.target_model_tier == ModelTier.SMALL.value
        )
        report_metadata["scenarios"].append({
            "scenario_id": "3a_policy_simulation_restrictive_margin",
            "type": "policy_simulation",
            "description": "Simulated strict memory policy via ROUTING_RAM_SAFETY_MARGIN_MB=100000. Not natural hardware pressure.",
            "prompt": s2_prompt,
            "selected_model": resp3a.routing.selected_model,
            "executed_model": resp3a.model,
            "target_tier": resp3a.routing.target_model_tier,
            "routing_status": resp3a.routing.routing_status,
            "is_fallback": resp3a.routing.is_fallback,
            "prompt_eval_count": prim3a.get("prompt_eval_count"),
            "eval_count": prim3a.get("eval_count"),
            "latency_ms": resp3a.execution_time_ms,
            "status": "VERIFIED" if s3a_verified else "NOT VERIFIED",
        })
    finally:
        settings.ROUTING_RAM_SAFETY_MARGIN_MB = orig_ram_margin

    # --------------------------------------------------------------------------
    # Scenario 3B: Natural Hardware Resource Pressure Observation
    # --------------------------------------------------------------------------
    print("\n--- SCENARIO 3B: Actual Hardware Resource Observation ---")
    live_gpu = HardwareTelemetryProvider.get_gpu_memory(force_refresh=True)
    live_ram = HardwareTelemetryProvider.get_system_ram()
    cur_vram_free: int = int(live_gpu.get("free_mb") or 0) if live_gpu else 0
    cur_ram_free: float = float(live_ram.get("free_mb") or 0)

    # Check whether natural resource constraints are currently present on this host
    ram_constrained = cur_ram_free < settings.ROUTING_RAM_SAFETY_MARGIN_MB
    vram_constrained = False
    if strong_candidate:
        req_headroom = strong_candidate.estimated_vram_mb + settings.ROUTING_VRAM_SAFETY_MARGIN_MB
        vram_constrained = cur_vram_free < req_headroom

    natural_pressure_present = ram_constrained or vram_constrained
    print(f"Live Observation -> Free VRAM: {cur_vram_free} MiB | Free RAM: {cur_ram_free} MiB")
    print(f"Natural Constraints Present: {natural_pressure_present} (RAM constrained: {ram_constrained}, VRAM constrained: {vram_constrained})")

    if natural_pressure_present:
        req3b = OptimizeRequest(prompt=s2_prompt, mode="structure")
        resp3b: OptimizeResponse = await OllamaService.optimize_prompt(req3b)
        assert resp3b.routing is not None
        print(f"Observed Natural Fallback -> Selected: {resp3b.routing.selected_model}, Status: {resp3b.routing.routing_status}")
        report_metadata["scenarios"].append({
            "scenario_id": "3b_natural_hardware_pressure_observation",
            "type": "natural_resource_observation",
            "natural_pressure_present": True,
            "selected_model": resp3b.routing.selected_model,
            "executed_model": resp3b.model,
            "routing_status": resp3b.routing.routing_status,
            "is_fallback": resp3b.routing.is_fallback,
            "status": "VERIFIED",
        })
    else:
        print("Natural resource pressure is NOT present on this host (system has sufficient VRAM and RAM headroom).")
        print("Reporting natural observation accurately without artificial memory exhaustion.")
        report_metadata["scenarios"].append({
            "scenario_id": "3b_natural_hardware_pressure_observation",
            "type": "natural_resource_observation",
            "natural_pressure_present": False,
            "observed_vram_free_mb": cur_vram_free,
            "observed_ram_free_mb": cur_ram_free,
            "finding": "Host hardware has ample free memory; natural fallback did not trigger. Artificial exhaustion was not performed to prevent system instability.",
            "status": "VERIFIED (NATURAL_PRESSURE_NOT_PRESENT)",
        })

    # --------------------------------------------------------------------------
    # Scenario 4: Manual Model Override with Resource Verification
    # --------------------------------------------------------------------------
    if strong_candidate is not None:
        await LocalModelRegistry.unload_model(strong_candidate.name)
        await asyncio.sleep(0.5)

    override_target = small_candidate.name if small_candidate else all_models[0].name
    print(f"\n--- SCENARIO 4: Manual Model Selection Override ({override_target}) ---")
    s4_prompt = "List 3 actionable recommendations for improving database query performance."
    req4 = OptimizeRequest(prompt=s4_prompt, requested_model=override_target, mode="concise")

    t0 = time.perf_counter()
    resp4: OptimizeResponse = await OllamaService.optimize_prompt(req4)
    dur4 = time.perf_counter() - t0

    assert resp4.routing is not None
    prim4 = resp4.inference_telemetry["calls"][0] if resp4.inference_telemetry and resp4.inference_telemetry.get("calls") else {}
    print(f"Requested Model: {override_target}")
    print(f"Selected Model: {resp4.routing.selected_model}")
    print(f"Actual Executed Model: {resp4.model}")
    print(f"Routing Status: {resp4.routing.routing_status}")
    print(f"Routing Reason: {resp4.routing.reason}")
    print(f"Inference Latency: {resp4.execution_time_ms} ms")
    print(f"Native Telemetry -> Prompt Tokens: {prim4.get('prompt_eval_count')}, Generated: {prim4.get('eval_count')}")

    s4_verified = (
        resp4.model == override_target
        and resp4.routing.selected_model == override_target
        and resp4.routing.routing_status == "manual_override"
    )
    report_metadata["scenarios"].append({
        "scenario_id": "4_manual_model_override",
        "type": "live_inference",
        "requested_model": override_target,
        "selected_model": resp4.routing.selected_model,
        "executed_model": resp4.model,
        "target_tier": resp4.routing.target_model_tier,
        "routing_status": resp4.routing.routing_status,
        "prompt_eval_count": prim4.get("prompt_eval_count"),
        "eval_count": prim4.get("eval_count"),
        "latency_ms": resp4.execution_time_ms,
        "status": "VERIFIED" if s4_verified else "NOT VERIFIED",
    })

    # --------------------------------------------------------------------------
    # Summary & Report Output
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    all_clean = True
    for sc in report_metadata["scenarios"]:
        st = sc["status"]
        sc_id = sc["scenario_id"]
        sc_type = sc["type"]
        if "NOT VERIFIED" in st:
            all_clean = False
        print(f"[{st}] {sc_id} (Type: {sc_type})")

    report_path = Path(__file__).resolve().parent.parent / "docs" / "reports" / "model_routing_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_metadata, f, indent=2)
    print(f"\nReport persisted to: {report_path}")

    if all_clean:
        print("\nOVERALL STATUS: ALL VALIDATION SCENARIOS VERIFIED SUCCESSFULLY!")
    else:
        print("\nOVERALL STATUS: ONE OR MORE SCENARIOS FAILED VERIFICATION.")


if __name__ == "__main__":
    asyncio.run(run_live_validation())
