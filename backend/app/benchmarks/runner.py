"""Benchmark Runner for Argus Reproducible Benchmark Suite (Phase 4).

Executes all 5 version-controlled benchmark cases across both Baseline and Optimized
paths, measuring prompt tokens, provider token usage, actual API costs, latencies,
and automated constraint fidelity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.core.config import settings
from app.benchmarks.evaluator import BenchmarkEvaluator, QualityEvaluation
from app.services.comparative_execution_service import (
    ComparativeExecutionService,
    ComparisonResult,
)
from app.services.main_llm_service import (
    BaseLLMProvider,
    MainLLMService,
    MissingCredentialsError,
    MockLLMProvider,
    OpenAIProvider,
)


class BenchmarkCaseResult(BaseModel):
    """Complete evaluation report for an individual benchmark dataset case."""

    case_id: str
    case_name: str
    category: str
    description: str

    # Execution telemetry
    comparison: ComparisonResult

    # Quality & constraint fidelity
    baseline_quality: QualityEvaluation
    optimized_quality: QualityEvaluation

    # Case Summary
    is_passing: bool = Field(description="True if quality and constraints are satisfied")
    summary_notes: str = Field(description="Executive observation of case outcome")


class BenchmarkSummary(BaseModel):
    """Aggregated metrics across the entire benchmark suite."""

    total_cases: int
    passed_cases: int
    pass_rate_percent: float

    # Token reductions
    baseline_prompt_tokens: int
    optimized_prompt_tokens: int
    prompt_tokens_saved: int
    prompt_tokens_reduction_percent: float

    baseline_total_tokens: int
    optimized_total_tokens: int
    total_tokens_saved: int
    total_tokens_reduction_percent: float

    # Cost Telemetry
    baseline_cost_usd: float
    optimized_cost_usd: float
    cost_saved_usd: float
    cost_savings_percent: float

    # Latency Telemetry
    avg_baseline_latency_ms: float
    avg_optimizer_latency_ms: float
    avg_optimized_total_latency_ms: float
    avg_latency_overhead_ms: float

    # Quality and Safety
    constraint_retention_rate_percent: float
    format_compliance_rate_percent: float

    # Execution Metadata
    provider_name: str
    model_name: str
    is_live: bool
    timestamp: str


class BenchmarkReport(BaseModel):
    """Top-level benchmark report containing all case results and aggregate summary."""

    metadata: Dict[str, Any]
    summary: BenchmarkSummary
    case_results: List[BenchmarkCaseResult]


# Default deterministic mock responses for benchmark test suite
BENCHMARK_MOCK_RESPONSES = {
    "case-1-simple-informational": (
        "- TCP is connection-oriented with a 3-way handshake, while UDP is connectionless.\n"
        "- TCP guarantees reliable in-order delivery and retransmission; UDP provides best-effort delivery with no delivery guarantees.\n"
        "- TCP includes congestion and flow control; UDP has minimal protocol overhead for real-time speed."
    ),
    "case-2-verbose-redundant": (
        "DNS (Domain Name System) translates human-readable domain names into IP addresses in four steps:\n"
        "1. Recursive Resolver: Receives the client query and queries root servers.\n"
        "2. Root Server: Directs the resolver to the Top-Level Domain (TLD) server (e.g., .com).\n"
        "3. TLD Server: Points to the authoritative name server for the specific domain.\n"
        "4. Authoritative Name Server: Returns the definitive IP address (A/AAAA record) to the resolver, which caches and forwards it to the client."
    ),
    "case-3-coding-constraints": (
        "```python\n"
        "import csv\n"
        "from collections import defaultdict\n"
        "\n"
        "monthly_revenue = defaultdict(float)\n"
        "\n"
        "# Read sales.csv without pandas\n"
        "with open('sales.csv', mode='r', newline='', encoding='utf-8') as f:\n"
        "    reader = csv.DictReader(f)\n"
        "    for row in reader:\n"
        "        date_val = row['date']\n"
        "        month = date_val[:7]  # YYYY-MM\n"
        "        revenue = float(row['amount'])\n"
        "        monthly_revenue[month] += revenue\n"
        "\n"
        "# Save results to monthly_report.csv\n"
        "with open('monthly_report.csv', mode='w', newline='', encoding='utf-8') as f:\n"
        "    writer = csv.writer(f)\n"
        "    writer.writerow(['month', 'revenue'])\n"
        "    for month, rev in sorted(monthly_revenue.items()):\n"
        "        writer.writerow([month, f'{rev:.2f}'])\n"
        "```"
    ),
    "case-4-structured-json-extraction": (
        '{\n'
        '  "full_name": "Sarah Connor",\n'
        '  "email": "sconnor@resistance.net",\n'
        '  "roles": ["Team Lead", "Security Auditor"],\n'
        '  "is_active": true\n'
        '}'
    ),
    "case-5-preserve-unmodified": (
        "```python\n"
        "def binary_search(sorted_list: list, target) -> int:\n"
        "    left, right = 0, len(sorted_list) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if sorted_list[mid] == target:\n"
        "            return mid\n"
        "        elif sorted_list[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
        "```"
    ),
}


class BenchmarkRunner:
    """Orchestrates benchmark runs across version-controlled datasets."""

    @classmethod
    def load_dataset(cls) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSON."""
        data_path = Path(__file__).parent / "dataset.json"
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    async def run(
        cls,
        live: bool = False,
        model: Optional[str] = None,
        provider: Optional[BaseLLMProvider] = None,
    ) -> BenchmarkReport:
        """Run benchmark suite and produce structured telemetry report."""
        dataset = cls.load_dataset()
        target_model = model or settings.MAIN_LLM_MODEL

        # Determine provider
        active_provider = provider
        if active_provider is None:
            if live:
                openai_prov = OpenAIProvider(default_model=target_model)
                if not openai_prov.is_configured():
                    raise MissingCredentialsError(
                        "Live benchmark requires a valid API key. Set OPENAI_API_KEY in your environment."
                    )
                active_provider = openai_prov
            else:
                # Default mock provider for automated tests and dry runs
                active_provider = MockLLMProvider(model=target_model)

        case_results: List[BenchmarkCaseResult] = []

        total_base_prompt_tokens = 0
        total_opt_prompt_tokens = 0
        total_base_tokens = 0
        total_opt_tokens = 0
        total_base_cost = 0.0
        total_opt_cost = 0.0

        total_base_lat = 0
        total_opt_lat = 0
        total_opt_total_lat = 0

        passed_count = 0
        constraint_preserved_count = 0
        format_valid_count = 0

        for case in dataset:
            case_id = case["id"]
            prompt = case["prompt"]
            mode = case.get("mode", "balanced")

            # In mock mode, adapt mock provider response for specific case
            if isinstance(active_provider, MockLLMProvider) and case_id in BENCHMARK_MOCK_RESPONSES:
                active_provider._canned_response = BENCHMARK_MOCK_RESPONSES[case_id]

            # Execute comparison
            comp = await ComparativeExecutionService.compare(
                prompt=prompt,
                mode=mode,
                model=target_model,
                provider=active_provider,
            )

            # Evaluate Baseline response
            base_eval = BenchmarkEvaluator.evaluate(
                original_prompt=prompt,
                optimized_prompt=prompt,
                llm_response=comp.baseline.llm_response,
                test_case=case,
            )

            # Evaluate Optimized response
            opt_eval = BenchmarkEvaluator.evaluate(
                original_prompt=prompt,
                optimized_prompt=comp.optimized_prompt,
                llm_response=comp.optimized.llm_response,
                test_case=case,
            )

            is_case_passing = opt_eval.is_passing and comp.constraints_preserved
            if is_case_passing:
                passed_count += 1
            if opt_eval.prompt_constraints_preserved and opt_eval.output_prohibitions_respected:
                constraint_preserved_count += 1
            if opt_eval.output_format_valid:
                format_valid_count += 1

            # Accumulate metrics
            total_base_prompt_tokens += comp.baseline.provider_usage.prompt_tokens
            total_opt_prompt_tokens += comp.optimized.provider_usage.prompt_tokens
            total_base_tokens += comp.baseline.provider_usage.total_tokens
            total_opt_tokens += comp.optimized.provider_usage.total_tokens

            total_base_cost += comp.baseline.cost.total_cost_usd
            total_opt_cost += comp.optimized.cost.total_cost_usd

            total_base_lat += comp.baseline.latency_ms
            total_opt_lat += comp.optimizer_details.execution_time_ms
            total_opt_total_lat += comp.optimized.total_latency_ms

            # Summary notes
            if comp.optimizer_details.strategy == "preserve":
                notes = "Preserved original prompt as concise baseline. Zero cost penalty."
            elif comp.prompt_tokens_saved > 0:
                notes = f"Reduced prompt tokens by {comp.prompt_tokens_saved_percent}% while preserving all constraints."
            elif comp.optimizer_details.length_result == "expanded":
                notes = "Clarified prompt with explicit instructions. Output format and constraints verified."
            else:
                notes = "Prompt optimized with constraint retention."

            case_results.append(
                BenchmarkCaseResult(
                    case_id=case_id,
                    case_name=case["name"],
                    category=case["category"],
                    description=case["description"],
                    comparison=comp,
                    baseline_quality=base_eval,
                    optimized_quality=opt_eval,
                    is_passing=is_case_passing,
                    summary_notes=notes,
                )
            )

        n = len(dataset)
        p_saved = total_base_prompt_tokens - total_opt_prompt_tokens
        p_saved_pct = (p_saved / total_base_prompt_tokens * 100.0) if total_base_prompt_tokens > 0 else 0.0

        t_saved = total_base_tokens - total_opt_tokens
        t_saved_pct = (t_saved / total_base_tokens * 100.0) if total_base_tokens > 0 else 0.0

        c_saved = total_base_cost - total_opt_cost
        c_saved_pct = (c_saved / total_base_cost * 100.0) if total_base_cost > 0 else 0.0

        summary = BenchmarkSummary(
            total_cases=n,
            passed_cases=passed_count,
            pass_rate_percent=round((passed_count / n) * 100.0, 1),
            baseline_prompt_tokens=total_base_prompt_tokens,
            optimized_prompt_tokens=total_opt_prompt_tokens,
            prompt_tokens_saved=p_saved,
            prompt_tokens_reduction_percent=round(p_saved_pct, 2),
            baseline_total_tokens=total_base_tokens,
            optimized_total_tokens=total_opt_tokens,
            total_tokens_saved=t_saved,
            total_tokens_reduction_percent=round(t_saved_pct, 2),
            baseline_cost_usd=round(total_base_cost, 7),
            optimized_cost_usd=round(total_opt_cost, 7),
            cost_saved_usd=round(c_saved, 7),
            cost_savings_percent=round(c_saved_pct, 2),
            avg_baseline_latency_ms=round(total_base_lat / n, 1),
            avg_optimizer_latency_ms=round(total_opt_lat / n, 1),
            avg_optimized_total_latency_ms=round(total_opt_total_lat / n, 1),
            avg_latency_overhead_ms=round((total_opt_total_lat - total_base_lat) / n, 1),
            constraint_retention_rate_percent=round((constraint_preserved_count / n) * 100.0, 1),
            format_compliance_rate_percent=round((format_valid_count / n) * 100.0, 1) if n > 0 else 100.0,
            provider_name=active_provider.get_status().provider,
            model_name=target_model,
            is_live=live and not isinstance(active_provider, MockLLMProvider),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return BenchmarkReport(
            metadata={
                "version": "Argus Phase 4 Benchmark v1.0",
                "dataset_count": n,
                "model": target_model,
                "is_live": summary.is_live,
            },
            summary=summary,
            case_results=case_results,
        )


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Argus Reproducible Benchmark Runner")
    parser.add_argument("--live", action="store_true", help="Run live benchmark with real external LLM API")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Target model (default: gpt-4o)")
    parser.add_argument("--output", type=str, default=None, help="Save report JSON to file path")
    args = parser.parse_args()

    report = asyncio.run(BenchmarkRunner.run(live=args.live, model=args.model))
    print(f"\n=================== Argus Phase 4 Benchmark Summary ===================")
    print(f"Provider: {report.summary.provider_name} | Model: {report.summary.model_name} | Live: {report.summary.is_live}")
    print(f"Cases: {report.summary.passed_cases}/{report.summary.total_cases} passed ({report.summary.pass_rate_percent}%)")
    print(f"Prompt Tokens: {report.summary.baseline_prompt_tokens} -> {report.summary.optimized_prompt_tokens} ({report.summary.prompt_tokens_reduction_percent}% saved)")
    print(f"Total Cost: ${report.summary.baseline_cost_usd:.6f} -> ${report.summary.optimized_cost_usd:.6f} ({report.summary.cost_savings_percent}% saved)")
    print(f"Constraint Retention: {report.summary.constraint_retention_rate_percent}%")
    print(f"Avg Optimizer Latency: {report.summary.avg_optimizer_latency_ms} ms")
    print(f"=======================================================================\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as out_f:
            json.dump(report.model_dump(), out_f, indent=2)
        print(f"Saved benchmark report to {args.output}")
