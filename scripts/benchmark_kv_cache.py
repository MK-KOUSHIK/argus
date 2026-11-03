"""Argus End-to-End KV-Cache Benchmark Harness (Phase 4).

Executes real-world benchmark evaluations against the Argus FastAPI backend
and local Ollama inference runtime under controlled KV cache configurations:
  - Test A: F16 KV Cache (OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=f16)
  - Test B: Q8_0 KV Cache (OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0)

Captures native Ollama telemetry, hardware metrics (VRAM via nvidia-smi,
RAM via Win32 API), and quality invariants across multiple repetitions.
"""

import argparse
import ctypes
import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx


def detect_hardware_metadata() -> Dict[str, Any]:
    """Dynamically query GPU, CPU, RAM, and OS metadata."""
    gpu_info = {"name": "Unknown", "total_vram_mib": None, "driver": "Unknown", "source": "nvidia_smi"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            gpu_info["name"] = parts[0]
            gpu_info["total_vram_mib"] = int(re.sub(r"[^\d]", "", parts[1]))
            gpu_info["driver"] = parts[2]
    except Exception as exc:
        gpu_info["error"] = str(exc)

    total_ram_mb = None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_mb = round(stat.ullTotalPhys / (1024 * 1024), 1)
    except Exception:
        pass

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu_info,
        "total_system_ram_mb": total_ram_mb,
    }


def get_vram_usage_mib() -> Optional[int]:
    """Query NVIDIA RTX GPU VRAM usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0].strip())
    except Exception:
        return None


def get_system_ram_mb() -> Optional[float]:
    """Query Windows system RAM usage via ctypes GlobalMemoryStatusEx."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        used_ram = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024)
        return round(used_ram, 1)
    except Exception:
        return None


# Benchmark Test Prompts Matrix
PROMPT_SUITE = [
    {
        "id": "prompt_1_short",
        "name": "Short Coding Task",
        "category": "short",
        "mode": "balanced",
        "prompt": "Write a Python function to check if a number is prime.",
        "expected_status": 200,
    },
    {
        "id": "prompt_2_medium",
        "name": "Medium Code Review with Redundancy",
        "category": "medium",
        "mode": "balanced",
        "prompt": (
            "Please review this Python code. I would like you to make sure to check for "
            "SQL injection vulnerabilities and memory leaks. In addition to that, please "
            "provide the refactored code with detailed comments explaining the security fixes."
        ),
        "expected_status": 200,
    },
    {
        "id": "prompt_3_long",
        "name": "Long Architecture & Data Pipeline",
        "category": "long",
        "mode": "structure",
        "prompt": (
            "Analyze this distributed telemetry pipeline: 50 IoT sensors publish events to "
            "Apache Kafka every 5 seconds. An ingestion consumer worker reads messages from Kafka, "
            "parses and validates JSON schemas, enriches the payloads with customer metadata from "
            "PostgreSQL, and writes the transformed events in micro-batches to ClickHouse. The system "
            "is experiencing high consumer lag during peak hours, occasional OOM crashes in the enrichment "
            "workers, and slow analytical query execution in ClickHouse. Propose a step-by-step optimization "
            "plan covering consumer group tuning, PostgreSQL connection pooling and caching strategies, "
            "ClickHouse partitioning and sorting keys, and memory profiling. Provide clear configuration "
            "recommendations and architecture diagrams."
        ),
        "expected_status": 200,
    },
    {
        "id": "prompt_4_constraints",
        "name": "Explicit Constraints & Negative Prohibitions",
        "category": "constraints",
        "mode": "balanced",
        "prompt": (
            "Write a Python script to parse a CSV file named 'sales_data.csv' and calculate "
            "the total revenue per product category. You must use only Python's standard library "
            "(csv, collections, decimal). Do not use pandas, numpy, or any third-party libraries. "
            "The output must be formatted as a markdown table with columns 'Category' and 'Total Revenue'."
        ),
        "expected_status": 200,
        "required_invariants": ["sales_data.csv", "Category", "Total Revenue"],
        "prohibited_invariants": ["import pandas", "import numpy"],
    },
    {
        "id": "prompt_5_near_limit",
        "name": "Near Context Limit Prompt (~2,400 tokens)",
        "category": "near_limit",
        "mode": "balanced",
        "prompt": (
            "Review and optimize the following enterprise microservices deployment specification. "
            "Evaluate network isolation policies, secret management, container lifecycle hooks, "
            "and horizontal pod autoscaling metrics. Ensure compliance with zero-trust network "
            "architecture, mutual TLS via Envoy proxy sidecars, and automated canary rollouts.\n\n"
            + (
                "Component Specification Module:\n"
                "  - Service Name: TelemetryGateway\n"
                "  - Ingress: HTTPS on port 8443 with mTLS termination\n"
                "  - Rate Limiting: 10,000 req/sec token bucket with Redis backing\n"
                "  - Observability: OpenTelemetry trace context propagation via W3C TraceContext headers\n"
                "  - Resilience: Circuit breaker configured at 50% error rate over 10s sliding window\n\n"
            ) * 18
            + "Provide an exhaustive security and performance remediation checklist."
        ),
        "expected_status": 200,
    },
    {
        "id": "prompt_6_overflow",
        "name": "Oversized Prompt Exceeding Max Context Limit (>8,192 tokens)",
        "category": "overflow",
        "mode": "balanced",
        "prompt": "Explain distributed systems consensus in extreme detail. " * 900,
        "expected_status": 413,
    },
]


def run_benchmark(
    kv_type: str,
    repetitions: int = 2,
    base_url: str = "http://127.0.0.1:8000",
    model: Optional[str] = None,
    report_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full benchmark suite against the live Argus API."""
    resolved_report_dir = report_dir or (Path(__file__).resolve().parent.parent / "docs" / "reports")
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" Starting KV Cache Benchmark: {kv_type.upper()}")
    print(f" Repetitions per prompt: {repetitions}")
    print(f" Target API: {base_url}")
    print(f" Report Directory: {resolved_report_dir}")
    print(f" Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"=======================================================\n")

    hardware = detect_hardware_metadata()

    # Verify API health and Ollama configuration
    with httpx.Client(timeout=10.0) as client:
        health_resp = client.get(f"{base_url}/api/health")
        if health_resp.status_code != 200:
            raise RuntimeError(f"Backend not healthy: {health_resp.status_code}")
        print("Backend API is operational.")

        status_resp = client.get(f"{base_url}/api/ollama/status")
        status_data = status_resp.json()
        active_model = model or status_data.get("model", "qwen2.5:1.5b")
        ollama_endpoint = status_data.get("base_url", "http://127.0.0.1:11434")
        runtime_policy = status_data.get("runtime_process", {})
        print(f"Ollama connected: {status_data.get('connected')}, Active Model: {active_model}")
        print(f"Runtime Policy: {runtime_policy.get('runtime_configuration_status')}")

        # Query Ollama server version directly
        ollama_version = "unknown"
        try:
            ver_resp = client.get(f"{ollama_endpoint}/api/version", timeout=3.0)
            if ver_resp.status_code == 200:
                ollama_version = ver_resp.json().get("version", "unknown")
        except Exception:
            pass

        # Warmup request
        print("\nExecuting model warmup request...")
        warmup_payload = {
            "prompt": "Write one sentence about computer engineering.",
            "mode": "balanced",
            "target_model": active_model,
        }
        warmup_res = client.post(f"{base_url}/api/optimize", json=warmup_payload, timeout=60.0)
        print(f"Warmup status: {warmup_res.status_code}")

    results = []

    for test_case in PROMPT_SUITE:
        case_id = test_case["id"]
        case_name = test_case["name"]
        prompt = test_case["prompt"]
        mode = test_case["mode"]
        expected_status = test_case["expected_status"]

        print(f"\n-------------------------------------------------------------")
        print(f"Running [{case_id}] {case_name}")
        print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

        case_runs = []
        for rep in range(1, repetitions + 1):
            time.sleep(1.0)  # settle time
            vram_start = get_vram_usage_mib()
            ram_start = get_system_ram_mb()

            start_clock = time.perf_counter()
            try:
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post(
                        f"{base_url}/api/optimize",
                        json={"prompt": prompt, "mode": mode, "target_model": active_model},
                    )
                elapsed_wall_ms = round((time.perf_counter() - start_clock) * 1000, 2)
                vram_end = get_vram_usage_mib()
                ram_end = get_system_ram_mb()

                status_code = resp.status_code
                resp_json = resp.json() if resp.status_code < 500 else {}

                run_data = {
                    "repetition": rep,
                    "status_code": status_code,
                    "expected_status": expected_status,
                    "wall_clock_elapsed_ms": elapsed_wall_ms,
                    "vram_start_mib": vram_start,
                    "vram_end_mib": vram_end,
                    "system_ram_mb": ram_end,
                    "telemetry_present": False,
                }

                if status_code == 200:
                    telemetry = resp_json.get("inference_telemetry") or {}
                    calls = telemetry.get("calls", [])
                    first_call = calls[0] if calls else {}

                    prompt_eval_count = first_call.get("prompt_eval_count")
                    prompt_eval_cached = first_call.get("prompt_eval_cached_count")
                    prompt_eval_dur_ns = first_call.get("prompt_eval_duration_ns")
                    eval_count = first_call.get("eval_count")
                    eval_dur_ns = first_call.get("eval_duration_ns")
                    prompt_tps = first_call.get("prompt_tokens_per_second")
                    gen_tps = first_call.get("generation_tokens_per_second")
                    queue_wait_ms = first_call.get("queue_wait_ms")
                    selected_ctx = first_call.get("selected_num_ctx")

                    optimized_text = resp_json.get("optimized_prompt", "")

                    # Quality verification
                    constraints_ok = True
                    for req in test_case.get("required_invariants", []):
                        if req.lower() not in optimized_text.lower():
                            constraints_ok = False
                    for proh in test_case.get("prohibited_invariants", []):
                        if proh.lower() in optimized_text.lower():
                            constraints_ok = False

                    turn_reversal = bool(re.search(r"please provide|let me know|what would you like", optimized_text, re.I))

                    has_telemetry = bool(first_call and first_call.get("native_metrics_available"))
                    run_data.update({
                        "telemetry_present": has_telemetry,
                        "prompt_eval_count": prompt_eval_count,
                        "prompt_eval_cached_count": prompt_eval_cached,
                        "prompt_eval_duration_ms": round(prompt_eval_dur_ns / 1e6, 2) if prompt_eval_dur_ns is not None else None,
                        "eval_count": eval_count,
                        "eval_duration_ms": round(eval_dur_ns / 1e6, 2) if eval_dur_ns is not None else None,
                        "prompt_tokens_per_second": prompt_tps,
                        "generation_tokens_per_second": gen_tps,
                        "queue_wait_ms": queue_wait_ms,
                        "selected_num_ctx": selected_ctx,
                        "token_savings": resp_json.get("token_savings"),
                        "token_reduction_percent": resp_json.get("token_reduction_percent"),
                        "constraints_preserved": constraints_ok,
                        "turn_reversal_detected": turn_reversal,
                        "optimized_sample": optimized_text[:120] + "..." if len(optimized_text) > 120 else optimized_text,
                    })

                    print(f"  Rep {rep}: 200 OK | Wall: {elapsed_wall_ms}ms | Prompt: {prompt_eval_count} tok (Cached: {prompt_eval_cached}) | Gen: {eval_count} tok ({gen_tps} tok/s) | VRAM: {vram_end} MiB")
                elif status_code == 413:
                    error_msg = resp_json.get("error", {}).get("message", "")
                    run_data.update({
                        "overflow_error_verified": True,
                        "error_message": error_msg,
                    })
                    print(f"  Rep {rep}: 413 Payload Too Large (Cleanly rejected without truncation) | Wall: {elapsed_wall_ms}ms")
                else:
                    error_detail = resp_json.get("detail") or resp.text[:120]
                    run_data.update({
                        "failed": True,
                        "error_message": error_detail,
                    })
                    print(f"  Rep {rep}: HTTP {status_code} | {error_detail}")

                case_runs.append(run_data)

            except Exception as exc:
                print(f"  Rep {rep}: ERROR - {str(exc)}")
                case_runs.append({
                    "repetition": rep,
                    "status_code": None,
                    "error": str(exc),
                    "telemetry_present": False,
                })

        results.append({
            "test_case": test_case,
            "runs": case_runs,
        })

    benchmark_output = {
        "kv_cache_type": kv_type,
        "model": active_model,
        "ollama_version": ollama_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "runtime_policy": runtime_policy,
        "results": results,
    }

    out_file = resolved_report_dir / f"benchmark_raw_{kv_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)
    print(f"\nSaved raw benchmark data to {out_file}")

    return benchmark_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Argus KV-Cache Benchmark")
    parser.add_argument("--kv-type", required=True, choices=["f16", "q8_0"], help="KV cache quantization type")
    parser.add_argument("--reps", type=int, default=2, help="Repetitions per test case")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Argus API base URL")
    parser.add_argument("--model", default=None, help="Target model (defaults to active Ollama model)")
    parser.add_argument("--report-dir", default=None, help="Directory to save raw benchmark JSON records")
    args = parser.parse_args()

    rep_dir = Path(args.report_dir) if args.report_dir else None
    run_benchmark(
        kv_type=args.kv_type,
        repetitions=args.reps,
        base_url=args.base_url,
        model=args.model,
        report_dir=rep_dir,
    )
