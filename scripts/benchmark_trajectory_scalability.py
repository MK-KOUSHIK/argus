"""Scalability Benchmark for Argus Phase 6: Agent History / Trajectory Reduction.

Measures realistic performance across 10, 25, 50, 100, and 200 message trajectories
with realistic tool-call cycles, traceback errors, and subsequent verified executions.

Captures:
- Input message count
- Reduced message count
- Estimated original and reduced tokens
- Reduction duration (mean and median over 10 repeated runs)
- Peak memory allocation (tracemalloc peak bytes)
- Post-reduction tool integrity validation result
"""

from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from typing import List

# Setup sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "backend"))

from app.schemas.trajectory import (
    ReduceTrajectoryRequest,
    ToolCall,
    ToolCallFunction,
    TrajectoryMessage,
    validate_trajectory_integrity,
)
from app.services.trajectory_reduction_service import TrajectoryReductionService


def generate_realistic_trajectory(target_count: int) -> List[TrajectoryMessage]:
    """Construct realistic multi-turn trajectory with tool calls, error cycles, and constraints."""
    messages: List[TrajectoryMessage] = [
        TrajectoryMessage(
            role="system",
            content="You are an autonomous engineering agent operating within local sandbox constraints. Follow all directives.",
        ),
        TrajectoryMessage(
            role="user",
            content="Build a high-throughput transaction ledger. Under any circumstance, do not use external cloud dependencies.",
        ),
    ]

    cycle_idx = 0
    while len(messages) < target_count - 1:
        # Create an error cycle: assistant call -> tool error -> user feedback -> assistant fix -> tool success
        err_id = f"call_err_{cycle_idx}"
        ok_id = f"call_ok_{cycle_idx}"

        cycle_messages = [
            TrajectoryMessage(
                role="assistant",
                content=f"Executing database transaction batch {cycle_idx}.",
                tool_calls=[
                    ToolCall(
                        id=err_id,
                        type="function",
                        function=ToolCallFunction(name="exec_batch", arguments={"batch_id": cycle_idx, "limit": 100}),
                    )
                ],
            ),
            TrajectoryMessage(
                role="tool",
                tool_call_id=err_id,
                name="exec_batch",
                content=f"Traceback (most recent call last):\nRuntimeError: Lock contention on partition {cycle_idx}\nexit code 1",
            ),
            TrajectoryMessage(
                role="user",
                content=f"Fix the lock contention on partition {cycle_idx} and retry with exponential backoff.",
            ),
            TrajectoryMessage(
                role="assistant",
                content=f"Applied backoff retry algorithm to partition {cycle_idx}. Verifying execution.",
                tool_calls=[
                    ToolCall(
                        id=ok_id,
                        type="function",
                        function=ToolCallFunction(name="exec_batch", arguments={"batch_id": cycle_idx, "backoff": True}),
                    )
                ],
            ),
            TrajectoryMessage(
                role="tool",
                tool_call_id=ok_id,
                name="exec_batch",
                content="[OK] Batch completed in 14.2ms. All tests passed. returncode 0",
            ),
        ]

        for msg in cycle_messages:
            if len(messages) < target_count - 1:
                messages.append(msg)
            else:
                break
        cycle_idx += 1

    # Final active user task
    messages.append(
        TrajectoryMessage(
            role="user",
            content="Generate the final performance audit report for all executed transaction batches.",
        )
    )

    validate_trajectory_integrity(messages)
    return messages


def run_scalability_benchmark():
    print("=" * 80)
    print("ARGUS PHASE 6: TRAJECTORY REDUCTION SCALABILITY BENCHMARK")
    print("Sizes tested: 10, 25, 50, 100, 200 messages")
    print("=" * 80)

    target_sizes = [10, 25, 50, 100, 200]
    results = []

    for size in target_sizes:
        messages = generate_realistic_trajectory(size)
        actual_count = len(messages)

        req = ReduceTrajectoryRequest(
            messages=messages,
            preserve_recent_turns=2,
            max_tool_output_chars=400,
            reduction_profile="balanced",
            include_context_budget=True,
        )

        # Run 10 iterations to measure duration statistics and peak memory
        durations = []
        peak_memories = []
        res = TrajectoryReductionService.reduce_trajectory(req)

        for _ in range(10):
            tracemalloc.start()
            t0 = time.perf_counter()
            res = TrajectoryReductionService.reduce_trajectory(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            durations.append(elapsed_ms)
            peak_memories.append(peak_mem / 1024)  # in KiB

        # Verify integrity on output
        validate_trajectory_integrity(res.reduced_messages)

        mean_dur = statistics.mean(durations)
        median_dur = statistics.median(durations)
        stdev_dur = statistics.stdev(durations)
        peak_mem_kb = max(peak_memories)

        telem = res.telemetry
        rec = {
            "target_count": size,
            "actual_input_messages": actual_count,
            "reduced_messages": len(res.reduced_messages),
            "messages_pruned": telem.messages_pruned_count,
            "superseded_errors_pruned": telem.superseded_errors_pruned,
            "original_estimated_tokens": telem.original_estimated_tokens,
            "reduced_estimated_tokens": telem.reduced_estimated_tokens,
            "token_savings": telem.estimated_token_savings,
            "token_savings_pct": telem.estimated_token_reduction_percent,
            "mean_duration_ms": round(mean_dur, 3),
            "median_duration_ms": round(median_dur, 3),
            "stdev_duration_ms": round(stdev_dur, 3),
            "peak_memory_kib": round(peak_mem_kb, 2),
            "integrity_valid": True,
            "context_tier": res.context_budget.selected_num_ctx if res.context_budget else None,
        }
        results.append(rec)

        print(
            f"Messages: {actual_count:3d} -> {rec['reduced_messages']:3d} | "
            f"Tokens: {rec['original_estimated_tokens']:5d} -> {rec['reduced_estimated_tokens']:5d} (-{rec['token_savings_pct']}%) | "
            f"Duration: {rec['mean_duration_ms']:.3f}ms (median: {rec['median_duration_ms']:.3f}ms) | "
            f"Peak Mem: {rec['peak_memory_kib']:.1f} KiB | Tier: {rec['context_tier']}"
        )

    print("\n" + "=" * 80)
    print("SCALABILITY SUMMARY TABLE (FOR DOCUMENTATION)")
    print("=" * 80)
    print("| Input Msgs | Reduced Msgs | Orig Tokens | Red Tokens | Token Savings (%) | Mean Dur (ms) | Median Dur (ms) | Stdev (ms) | Peak RAM (KiB) | Integrity |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['actual_input_messages']} | {r['reduced_messages']} | {r['original_estimated_tokens']} | "
            f"{r['reduced_estimated_tokens']} | **{r['token_savings']} ({r['token_savings_pct']}%)** | "
            f"{r['mean_duration_ms']} ms | {r['median_duration_ms']} ms | {r['stdev_duration_ms']} ms | "
            f"{r['peak_memory_kib']} KiB | **VALID** |"
        )

    return results


if __name__ == "__main__":
    run_scalability_benchmark()
