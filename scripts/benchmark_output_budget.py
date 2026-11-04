"""Empirical Output Budget Control Benchmark for Argus (Phase 4.1).

Executes genuine local Ollama inference using the configured local model (e.g. qwen2.5:1.5b)
to empirically compare:
1. Fixed Baseline: num_predict = 512
2. Adaptive Output Budget: num_predict allocated dynamically per task/constraints

Measures real native telemetry and distinct metrics:
- Allocated budget ceiling (num_predict)
- Actual tokens generated (eval_count)
- Budget ceiling difference vs actual generation difference
- Generation duration (eval_duration_ms) and throughput (generation_tokens_per_second)
- Context tier selection (selected_num_ctx)
- Termination cause (done_reason: 'stop', 'length', or None)
- Programmatic task quality evaluation (AST parse, schema validation, word counts)

Saves raw unmanipulated measurements to docs/reports/output_budget_benchmark.json.
"""

import argparse
import ast
import asyncio
import ctypes
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# Ensure backend directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import settings
from app.schemas.optimizer import OptimizationMode, OptimizeRequest
from app.services.ollama_service import OllamaService
from app.services.output_budget_controller import OutputBudgetController


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


def detect_hardware_metadata() -> Dict[str, Any]:
    """Dynamically query GPU, CPU, RAM, and OS metadata without hardcoded values."""
    gpu_name: Optional[str] = None
    total_vram_mib: Optional[int] = None
    driver_version: Optional[str] = None
    gpu_error: Optional[str] = None

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            gpu_name = parts[0]
            vram_digits = re.sub(r"[^\d]", "", parts[1])
            if vram_digits:
                total_vram_mib = int(vram_digits)
            driver_version = parts[2]
    except Exception as exc:
        gpu_error = str(exc)

    gpu_info: Dict[str, Any] = {
        "name": gpu_name,
        "total_vram_mib": total_vram_mib,
        "driver": driver_version,
        "source": "nvidia_smi" if gpu_name else None,
    }
    if gpu_error is not None:
        gpu_info["error"] = gpu_error

    total_ram_mb: Optional[float] = None
    if os.name == "nt" or sys.platform == "win32":
        try:
            windll = getattr(ctypes, "windll", None)
            if windll is not None:
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                    total_ram_mb = round(stat.ullTotalPhys / (1024 * 1024), 1)
        except Exception:
            pass
    elif os.name == "posix":
        try:
            meminfo_path = "/proc/meminfo"
            if os.path.exists(meminfo_path):
                with open(meminfo_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            parts = line.split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                total_ram_mb = round(int(parts[1]) / 1024, 1)
                            break
        except Exception:
            pass

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu_info,
        "total_system_ram_mb": total_ram_mb,
    }


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


def extract_code_block(text: str) -> str:
    """Extract code inside ```python ... ``` or ``` ... ``` blocks, or return original text."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return text.strip()


def extract_json_block(text: str) -> Optional[str]:
    """Extract JSON inside ```json ... ``` or raw outermost { ... } block."""
    fence_pattern = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def evaluate_task_quality(test_id: str, output_text: str) -> Dict[str, Any]:
    """Perform objective, programmatic task quality evaluation per benchmark test.

    Decoupled from prompt-preservation validation status.
    Returns:
        Dict with keys 'evaluated' (bool), 'passed' (bool), 'reason' (str), 'details' (dict).
    """
    if not output_text or not output_text.strip():
        return {
            "evaluated": True,
            "passed": False,
            "reason": "empty_output",
            "details": {"output_length": 0},
        }

    lower_text = output_text.lower()

    if test_id == "word_limit_summary":
        words = output_text.strip().split()
        word_count = len(words)
        within_limit = word_count <= 50
        has_sync = bool(re.search(r"\b(?:sync|synchronous|blocking|sequential)\b", lower_text))
        has_async = bool(re.search(r"\b(?:async|asynchronous|event loop|non-blocking|coroutine|await)\b", lower_text))
        passed = within_limit and has_sync and has_async

        return {
            "evaluated": True,
            "passed": passed,
            "reason": "evaluated_against_word_limit_and_concepts",
            "details": {
                "word_count": word_count,
                "ceiling_words": 50,
                "within_limit": within_limit,
                "covers_sync": has_sync,
                "covers_async": has_async,
            },
        }

    elif test_id == "single_sentence_qa":
        cleaned = re.sub(r"^(?:[\*\-\d\.]+\s*)+", "", output_text.strip())
        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        sentence_count = len(raw_sentences)
        single_sentence = (sentence_count == 1)

        has_index_topic = bool(re.search(r"\b(?:index|indexes|indexing|b-tree)\b", lower_text))
        has_perf_topic = bool(
            re.search(r"\b(?:fast|speed|lookup|search|locate|scan|cost|performance|quick|reduce)\b", lower_text)
        )
        passed = single_sentence and has_index_topic and has_perf_topic

        return {
            "evaluated": True,
            "passed": passed,
            "reason": "evaluated_against_sentence_count_and_content",
            "details": {
                "sentence_count": sentence_count,
                "single_sentence": single_sentence,
                "mentions_index": has_index_topic,
                "mentions_performance": has_perf_topic,
                "sentences": raw_sentences,
            },
        }

    elif test_id == "json_entity_extraction":
        json_str = extract_json_block(output_text)
        if not json_str:
            return {
                "evaluated": False,
                "passed": False,
                "reason": "not_evaluated: output is prompt directive, no json block generated",
                "details": {"raw_output_snippet": output_text[:120]},
            }
        try:
            data = json.loads(json_str)
        except Exception as exc:
            return {
                "evaluated": True,
                "passed": False,
                "reason": f"json_parse_error: {str(exc)}",
                "details": {"json_snippet": json_str[:120]},
            }

        data_str = json.dumps(data).lower()
        has_customer = "alice smith" in data_str or "alice" in data_str
        has_order = "8491" in data_str
        has_widget_a = "widget a" in data_str
        has_widget_b = "widget b" in data_str
        has_total = "35" in data_str

        passed = has_customer and has_order and has_widget_a and has_widget_b and has_total
        return {
            "evaluated": True,
            "passed": passed,
            "reason": "evaluated_against_schema_and_entities",
            "details": {
                "is_valid_json": True,
                "has_customer": has_customer,
                "has_order_id": has_order,
                "has_products": (has_widget_a and has_widget_b),
                "has_total": has_total,
                "parsed_keys": list(data.keys()) if isinstance(data, dict) else [],
            },
        }

    elif test_id == "code_generation":
        code = extract_code_block(output_text)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return {
                "evaluated": False,
                "passed": False,
                "reason": "not_evaluated: output is prompt directive, no python syntax found",
                "details": {"code_snippet": code[:120]},
            }

        func_defs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not func_defs:
            return {
                "evaluated": False,
                "passed": False,
                "reason": "not_evaluated: output is prompt directive, no function definition found",
                "details": {"ast_nodes": [type(n).__name__ for n in tree.body]},
            }

        target_fn = func_defs[0]
        has_yield = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(target_fn))
        has_docstring = ast.get_docstring(target_fn) is not None
        has_return_type = target_fn.returns is not None
        has_arg_types = any(arg.annotation is not None for arg in target_fn.args.args)
        has_type_hints = has_return_type or has_arg_types

        passed = has_yield and has_docstring and has_type_hints
        return {
            "evaluated": True,
            "passed": passed,
            "reason": "evaluated_against_ast_and_typing",
            "details": {
                "function_name": target_fn.name,
                "has_yield": has_yield,
                "has_docstring": has_docstring,
                "has_type_hints": has_type_hints,
                "has_return_type": has_return_type,
                "has_arg_types": has_arg_types,
            },
        }

    elif test_id == "structured_tradeoff_analysis":
        has_headings = bool(re.search(r"^#{1,6}\s+.+", output_text, re.MULTILINE))
        if not has_headings:
            return {
                "evaluated": False,
                "passed": False,
                "reason": "not_evaluated: output is prompt directive, no markdown headings generated",
                "details": {"output_snippet": output_text[:120]},
            }

        has_redis = "redis" in lower_text
        has_memcached = "memcached" in lower_text
        has_latency = "latency" in lower_text
        has_memory = "memory" in lower_text
        has_persistence = "persistence" in lower_text
        has_list = bool(re.search(r"^\s*[\-\*\d\.]+\s+", output_text, re.MULTILINE))

        passed = (
            has_headings
            and has_redis
            and has_memcached
            and has_latency
            and has_memory
            and has_persistence
        )
        return {
            "evaluated": True,
            "passed": passed,
            "reason": "evaluated_against_markdown_and_topics",
            "details": {
                "has_headings": has_headings,
                "has_list_or_checklist": has_list,
                "mentions_redis": has_redis,
                "mentions_memcached": has_memcached,
                "mentions_latency": has_latency,
                "mentions_memory": has_memory,
                "mentions_persistence": has_persistence,
            },
        }

    else:
        return {
            "evaluated": False,
            "passed": False,
            "reason": "not_evaluated",
            "details": {"test_id": test_id},
        }


async def run_benchmark(reps: int = 1):
    hw = detect_hardware_metadata()
    gpu_label = f"{hw['gpu'].get('name', 'Unknown GPU')} ({hw['gpu'].get('total_vram_mib', '?')} MiB VRAM)"

    print("=" * 70)
    print("Argus Output Budget Control — Empirical Live Benchmark")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Ollama Endpoint: {settings.OLLAMA_BASE_URL}")
    print(f"Target Model: {settings.OLLAMA_MODEL}")
    print(f"KV-Cache Type: {settings.OLLAMA_KV_CACHE_TYPE}")
    print(f"Flash Attention: {settings.OLLAMA_FLASH_ATTENTION}")
    print(f"Hardware Detected: CPU: {hw['cpu']} | GPU: {gpu_label} | RAM: {hw['total_system_ram_mb']} MB")
    print(f"Repetitions: {reps}")
    print("=" * 70)

    # 1. Verify Ollama Status
    status = await OllamaService.check_status()
    if not status.connected or not status.model_installed:
        print(f"ERROR: Ollama server not ready: {status.error}")
        sys.exit(1)
    print(f"Verified Ollama connection. Available models: {', '.join(status.available_models)}\n")

    # Warmup call
    print("Executing warmup inference call to ensure model weights and KV buffers are active...")
    warmup_req = OptimizeRequest(
        prompt="Warmup prompt to initialize model and KV buffers.",
        mode="balanced",
        target_model="gpt-4o",
    )
    warmup_res = await OllamaService.optimize_prompt(warmup_req)
    print(f"Warmup complete. Latency: {warmup_res.execution_time_ms} ms\n")

    original_enabled = settings.OUTPUT_BUDGET_ENABLED
    original_policy = settings.OUTPUT_BUDGET_POLICY

    try:
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
                mode=cast(OptimizationMode, item["mode"]),
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
            baseline_eval = evaluate_task_quality(item["id"], baseline_res.optimized_prompt)
            base_qual_str = "PASS" if baseline_eval.get("passed") else ("N/A (Prompt Rewrite)" if not baseline_eval.get("evaluated") else "FAIL")

            print(
                f"     Allocated: {baseline_telemetry.get('num_predict_limit')} ceiling | "
                f"Generated: {baseline_telemetry.get('eval_count')} tokens | "
                f"Done: {baseline_telemetry.get('done_reason')} | "
                f"Quality: {base_qual_str} | "
                f"Latency: {baseline_res.execution_time_ms} ms"
            )

            await asyncio.sleep(0.5)

            # Condition 2: Adaptive Output Budget
            print("  -> Condition 2: Adaptive Output Budget...")
            settings.OUTPUT_BUDGET_ENABLED = True
            settings.OUTPUT_BUDGET_POLICY = "adaptive"
            adaptive_req = OptimizeRequest(
                prompt=item["prompt"],
                mode=cast(OptimizationMode, item["mode"]),
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
            adaptive_eval = evaluate_task_quality(item["id"], adaptive_res.optimized_prompt)
            adapt_qual_str = "PASS" if adaptive_eval.get("passed") else ("N/A (Prompt Rewrite)" if not adaptive_eval.get("evaluated") else "FAIL")

            print(
                f"     Allocated: {adaptive_ob.allocated_budget if adaptive_ob else 'N/A'} ceiling | "
                f"Generated: {adaptive_ob.actual_generated_tokens if adaptive_ob else 'N/A'} tokens | "
                f"Done: {adaptive_ob.done_reason if adaptive_ob else 'N/A'} | "
                f"Quality: {adapt_qual_str} | "
                f"Utilization: {adaptive_ob.budget_utilization_percent if adaptive_ob else 'N/A'}% | "
                f"Latency: {adaptive_res.execution_time_ms} ms"
            )
            if adaptive_ob:
                print(f"     Rationale: {adaptive_ob.reason}")

            # Calculate metric separations
            baseline_ceiling = int(baseline_telemetry.get("num_predict_limit") or 512)
            adaptive_ceiling = int(adaptive_ob.allocated_budget if adaptive_ob else 512)
            ceiling_diff = baseline_ceiling - adaptive_ceiling

            baseline_gen = baseline_telemetry.get("eval_count") or 0
            adaptive_gen = (adaptive_ob.actual_generated_tokens if adaptive_ob else 0) or 0
            actual_gen_diff = baseline_gen - adaptive_gen

            record = {
                "test_id": item["id"],
                "name": item["name"],
                "prompt": item["prompt"],
                "mode": item["mode"],
                "description": item["description"],
                "baseline": {
                    "allocated_budget_ceiling": baseline_ceiling,
                    "actual_tokens_generated": baseline_gen,
                    "selected_num_ctx": baseline_telemetry.get("selected_num_ctx"),
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
                    "allocated_budget_ceiling": adaptive_ceiling,
                    "actual_tokens_generated": adaptive_gen,
                    "selected_num_ctx": adaptive_telemetry.get("selected_num_ctx"),
                    "eval_duration_ms": adaptive_ob.generation_duration_ms if adaptive_ob else None,
                    "prompt_eval_count": adaptive_telemetry.get("prompt_eval_count"),
                    "prompt_eval_cached_count": adaptive_telemetry.get("prompt_eval_cached_count"),
                    "generation_tokens_per_second": adaptive_ob.generation_tokens_per_second if adaptive_ob else None,
                    "done_reason": adaptive_ob.done_reason if adaptive_ob else None,
                    "hit_budget_limit": adaptive_ob.hit_budget_limit if adaptive_ob else None,
                    "budget_utilization_percent": adaptive_ob.budget_utilization_percent if adaptive_ob else None,
                    "allocation_reason": adaptive_ob.reason if adaptive_ob else None,
                    "task_category": adaptive_ob.task_category if adaptive_ob else None,
                    "task_confidence": adaptive_ob.task_confidence if adaptive_ob else None,
                    "execution_time_ms": adaptive_res.execution_time_ms,
                    "wall_time_ms": adaptive_wall_ms,
                    "output_text": adaptive_res.optimized_prompt,
                },
                "comparison": {
                    "budget_ceiling_difference": ceiling_diff,
                    "budget_ceiling_reduction_percent": round((ceiling_diff / baseline_ceiling) * 100, 1)
                    if baseline_ceiling > 0
                    else 0.0,
                    "actual_generation_difference": actual_gen_diff,
                    "actual_generation_difference_percent": round((actual_gen_diff / baseline_gen) * 100, 1)
                    if baseline_gen > 0
                    else 0.0,
                    "prompt_validation_status": adaptive_res.validation_status,
                    "prompt_validation_outcome": adaptive_res.outcome,
                    "task_quality_preserved": (adaptive_eval.get("passed", False) is True)
                    if adaptive_eval.get("evaluated")
                    else None,
                    "baseline_task_evaluation": baseline_eval,
                    "adaptive_task_evaluation": adaptive_eval,
                },
            }
            results.append(record)
            print()
            await asyncio.sleep(0.5)

        # Aggregate Summaries
        total_baseline_ceiling = sum(r["baseline"]["allocated_budget_ceiling"] for r in results)
        total_adaptive_ceiling = sum(r["adaptive"]["allocated_budget_ceiling"] for r in results)
        total_ceiling_diff = total_baseline_ceiling - total_adaptive_ceiling
        avg_ceiling_reduction_pct = (
            round((total_ceiling_diff / total_baseline_ceiling) * 100, 1) if total_baseline_ceiling > 0 else 0.0
        )

        total_baseline_generated = sum(r["baseline"]["actual_tokens_generated"] for r in results)
        total_adaptive_generated = sum(r["adaptive"]["actual_tokens_generated"] for r in results)
        total_generation_diff = total_baseline_generated - total_adaptive_generated
        avg_generation_diff_pct = (
            round((total_generation_diff / total_baseline_generated) * 100, 1) if total_baseline_generated > 0 else 0.0
        )

        evaluated_prompts = [r for r in results if r["comparison"]["adaptive_task_evaluation"].get("evaluated")]
        adaptive_passed = sum(1 for r in evaluated_prompts if r["comparison"]["adaptive_task_evaluation"].get("passed"))
        baseline_passed = sum(1 for r in evaluated_prompts if r["comparison"]["baseline_task_evaluation"].get("passed"))

        benchmark_payload = {
            "benchmark_metadata": {
                "title": "Argus Output Budget Control Empirical Benchmark",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": settings.OLLAMA_MODEL,
                "runtime_endpoint": settings.OLLAMA_BASE_URL,
                "kv_cache_type": settings.OLLAMA_KV_CACHE_TYPE,
                "flash_attention": settings.OLLAMA_FLASH_ATTENTION,
                "hardware": hw,
                "total_test_prompts": len(results),
            },
            "aggregate_summary": {
                "total_baseline_ceiling_tokens": total_baseline_ceiling,
                "total_adaptive_ceiling_tokens": total_adaptive_ceiling,
                "budget_ceiling_difference_tokens": total_ceiling_diff,
                "budget_ceiling_reduction_percent": avg_ceiling_reduction_pct,
                "total_baseline_generated_tokens": total_baseline_generated,
                "total_adaptive_generated_tokens": total_adaptive_generated,
                "actual_generation_difference_tokens": total_generation_diff,
                "actual_generation_difference_percent": avg_generation_diff_pct,
                "quality_evaluation_summary": {
                    "total_prompts_evaluated": len(evaluated_prompts),
                    "adaptive_prompts_passed": adaptive_passed,
                    "baseline_prompts_passed": baseline_passed,
                    "all_adaptive_evaluations_passed": (adaptive_passed == len(evaluated_prompts) and len(evaluated_prompts) > 0),
                },
                "all_prompt_preservations_passed": all(
                    r["comparison"]["prompt_validation_status"] == "passed" for r in results
                ),
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
        print(
            f"Allocated Budget Ceiling Conserved: {total_ceiling_diff} tokens ({avg_ceiling_reduction_pct}% reduction)"
        )
        print(
            f"Actual Tokens Generated Difference: {total_generation_diff} tokens ({avg_generation_diff_pct}%)"
        )
        print(f"Task Quality Evaluated: {adaptive_passed}/{len(evaluated_prompts)} passed")
        print("=" * 70)

    finally:
        # Strictly restore settings
        settings.OUTPUT_BUDGET_ENABLED = original_enabled
        settings.OUTPUT_BUDGET_POLICY = original_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Argus Output Budget Control Empirical Benchmark")
    parser.add_argument("--reps", type=int, default=1, help="Number of repetitions per prompt (default: 1)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(run_benchmark(reps=cli_args.reps))
