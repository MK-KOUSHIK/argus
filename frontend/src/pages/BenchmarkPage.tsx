import React, { useState, useEffect } from 'react';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { apiService } from '@/services/api';
import {
  BenchmarkReport,
  BenchmarkCaseResult,
  MainLLMStatus,
} from '@/types';
import {
  BarChart2,
  Play,
  RotateCw,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Zap,
  DollarSign,
  Clock,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FileCode,
  Sparkles,
  Layers,
} from 'lucide-react';

interface BenchmarkPageProps {
  mainLLMStatus?: MainLLMStatus | null;
}

export const BenchmarkPage: React.FC<BenchmarkPageProps> = ({ mainLLMStatus }) => {
  const [report, setReport] = useState<BenchmarkReport | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [targetModel, setTargetModel] = useState<string>('gpt-4o');
  const [isLiveMode, setIsLiveMode] = useState<boolean>(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [expandedCases, setExpandedCases] = useState<Record<string, boolean>>({});

  const toggleCase = (id: string) => {
    setExpandedCases((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const runBenchmark = async (live: boolean = isLiveMode) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await apiService.runBenchmark(live, targetModel);
      setReport(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to execute benchmark suite'
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Run initial mock benchmark on mount if not yet loaded
  useEffect(() => {
    if (!report && !isLoading) {
      void runBenchmark(false);
    }
  }, []);

  const filteredCases = report?.case_results.filter((c) => {
    if (selectedCategory === 'all') return true;
    return c.category === selectedCategory;
  }) || [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Benchmark Header */}
      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30">
              <BarChart2 className="w-3.5 h-3.5 text-cyan-400" />
              <span>Phase 4 Reproducible Benchmark Suite</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mt-2">
              Empirical Cost & Quality Benchmark
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl mt-1">
              Measures whether local prompt optimization reduces external LLM input tokens and inference costs
              without degrading answer quality or violating explicit constraints.
            </p>
          </div>

          {/* Execution Controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Target Model Selector */}
            <div className="flex items-center gap-2 bg-[#0c1424] px-3 py-1.5 rounded-lg border border-white/5 text-xs">
              <span className="text-slate-400">Target Model:</span>
              <select
                aria-label="Target Model"
                value={targetModel}
                onChange={(e) => setTargetModel(e.target.value)}
                className="bg-transparent text-cyan-300 font-mono text-xs focus:outline-none cursor-pointer"
              >
                <option value="gpt-4o" className="bg-[#0c1424] text-slate-200">gpt-4o</option>
                <option value="gpt-4o-mini" className="bg-[#0c1424] text-slate-200">gpt-4o-mini</option>
              </select>
            </div>

            {/* Live toggle */}
            <div className="flex items-center gap-2 bg-[#0c1424] px-3 py-1.5 rounded-lg border border-white/5 text-xs">
              <span className="text-slate-400">Mode:</span>
              <button
                type="button"
                onClick={() => setIsLiveMode(false)}
                className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                  !isLiveMode
                    ? 'bg-cyan-500 text-navy-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Mock (Safe)
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!mainLLMStatus?.is_configured) {
                    setError('Live execution requires OPENAI_API_KEY set in backend environment.');
                    return;
                  }
                  setIsLiveMode(true);
                }}
                className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                  isLiveMode
                    ? 'bg-amber-500 text-navy-950 font-semibold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title={
                  mainLLMStatus?.is_configured
                    ? 'Execute real external OpenAI API calls'
                    : 'Requires OPENAI_API_KEY in environment'
                }
              >
                Live API
              </button>
            </div>

            <Button
              variant="primary"
              onClick={() => runBenchmark()}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <RotateCw className="w-4 h-4 animate-spin" />
                  <span>Running Benchmark...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Run Benchmark</span>
                </>
              )}
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-500/30 text-rose-300 text-sm flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
            <div>
              <p className="font-semibold">Benchmark Execution Notice</p>
              <p className="text-xs text-rose-200/90 mt-0.5">{error}</p>
            </div>
          </div>
        )}
      </section>

      {/* Aggregate KPI Summary */}
      {report && (
        <section className="grid grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4">
          <Card className="p-4 bg-[#0a101d] border-white/5 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Pass Rate</span>
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-white font-mono">
              {report.summary.pass_rate_percent}%
            </div>
            <div className="text-[11px] text-slate-400">
              {report.summary.passed_cases} of {report.summary.total_cases} cases verified
            </div>
          </Card>

          <Card className="p-4 bg-[#0a101d] border-white/5 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Prompt Tokens Saved</span>
              <Zap className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold text-cyan-300 font-mono">
              {report.summary.prompt_tokens_reduction_percent}%
            </div>
            <div className="text-[11px] text-slate-400">
              {report.summary.prompt_tokens_saved} tokens ({report.summary.baseline_prompt_tokens} &rarr; {report.summary.optimized_prompt_tokens})
            </div>
          </Card>

          <Card className="p-4 bg-[#0a101d] border-white/5 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Cost Savings</span>
              <DollarSign className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold text-emerald-400 font-mono">
              {report.summary.cost_savings_percent}%
            </div>
            <div className="text-[11px] text-slate-400">
              ${report.summary.cost_saved_usd.toFixed(6)} saved
            </div>
          </Card>

          <Card className="p-4 bg-[#0a101d] border-white/5 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Constraint Retention</span>
              <CheckCircle2 className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-2xl font-bold text-white font-mono">
              {report.summary.constraint_retention_rate_percent}%
            </div>
            <div className="text-[11px] text-slate-400">
              Zero constraint loss or corruption
            </div>
          </Card>

          <Card className="p-4 bg-[#0a101d] border-white/5 space-y-1 col-span-2 lg:col-span-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Avg Optimizer Latency</span>
              <Clock className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold text-amber-300 font-mono">
              {report.summary.avg_optimizer_latency_ms} <span className="text-sm">ms</span>
            </div>
            <div className="text-[11px] text-slate-400">
              Local Ollama overhead per prompt
            </div>
          </Card>
        </section>
      )}

      {/* Dataset & Case Details */}
      <section className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] pb-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <h2 className="text-lg font-semibold text-white">Version-Controlled Test Dataset</h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-[#0c1424] text-slate-400 border border-white/5">
              5 Core Scenarios
            </span>
          </div>

          {/* Filter Chips */}
          <div className="flex flex-wrap items-center gap-1.5 text-xs">
            {[
              { id: 'all', label: 'All Cases' },
              { id: 'simple_informational', label: 'Simple' },
              { id: 'verbose_redundant', label: 'Verbose Fluff' },
              { id: 'coding_constraints', label: 'Coding Tasks' },
              { id: 'structured_json_extraction', label: 'JSON Extraction' },
              { id: 'preserve_unmodified', label: 'Preserve Baseline' },
            ].map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => setSelectedCategory(f.id)}
                className={`px-2.5 py-1 rounded-md transition-all cursor-pointer ${
                  selectedCategory === f.id
                    ? 'bg-cyan-500 text-navy-950 font-semibold'
                    : 'bg-[#0c1424] text-slate-400 hover:text-slate-200 border border-white/5'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Case Cards */}
        <div className="space-y-4">
          {filteredCases.map((caseItem: BenchmarkCaseResult) => {
            const isExpanded = Boolean(expandedCases[caseItem.case_id]);
            const promptSavedPct = caseItem.comparison.prompt_tokens_saved_percent;
            const costSavedPct = caseItem.comparison.cost_comparison.savings_percent;

            return (
              <Card
                key={caseItem.case_id}
                className="bg-[#080e1a] border-white/5 overflow-hidden transition-all"
              >
                {/* Case Header Row */}
                <div
                  className="p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 cursor-pointer hover:bg-white/[0.02]"
                  onClick={() => toggleCase(caseItem.case_id)}
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/20">
                        {caseItem.category.replace('_', ' ')}
                      </span>
                      <h3 className="font-semibold text-white text-base">
                        {caseItem.case_name}
                      </h3>
                      {caseItem.is_passing ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-500/20">
                          <CheckCircle2 className="w-3 h-3" /> Passed
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-400 bg-amber-950/40 px-2 py-0.5 rounded border border-amber-500/20">
                          <AlertTriangle className="w-3 h-3" /> Flagged
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-400">
                      {caseItem.description}
                    </p>
                  </div>

                  {/* Quick Telemetry Chips */}
                  <div className="flex items-center gap-3 sm:gap-4 shrink-0">
                    <div className="text-right">
                      <div className="text-[11px] text-slate-500">Prompt Tokens</div>
                      <div
                        className={`font-mono text-sm font-semibold ${
                          promptSavedPct > 0
                            ? 'text-cyan-300'
                            : promptSavedPct < 0
                            ? 'text-amber-400'
                            : 'text-slate-400'
                        }`}
                      >
                        {promptSavedPct > 0 ? `-${promptSavedPct}%` : `${promptSavedPct}%`}
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-[11px] text-slate-500">Total Cost</div>
                      <div
                        className={`font-mono text-sm font-semibold ${
                          costSavedPct > 0 ? 'text-emerald-400' : 'text-slate-400'
                        }`}
                      >
                        {costSavedPct > 0 ? `-${costSavedPct}%` : `${costSavedPct}%`}
                      </div>
                    </div>

                    <button
                      type="button"
                      aria-label="Toggle details"
                      className="text-slate-400 hover:text-slate-200 p-1"
                    >
                      {isExpanded ? (
                        <ChevronUp className="w-5 h-5" />
                      ) : (
                        <ChevronDown className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Expanded Details Panel */}
                {isExpanded && (
                  <div className="border-t border-white/[0.06] p-4 sm:p-6 bg-[#060b14] space-y-6">
                    {/* Side-by-Side Prompt Diff */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Baseline Prompt */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs font-semibold text-slate-400">
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-slate-500" />
                            Original Input Prompt (Baseline)
                          </span>
                          <span className="font-mono text-[11px] text-slate-500">
                            {caseItem.comparison.baseline.tokenizer_tokens.token_count} tokens
                          </span>
                        </div>
                        <div className="p-3.5 rounded-lg bg-[#0a101d] border border-white/5 text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed min-h-[110px]">
                          {caseItem.comparison.original_prompt}
                        </div>
                      </div>

                      {/* Optimized Prompt */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs font-semibold text-cyan-300">
                          <span className="flex items-center gap-1.5">
                            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                            Argus Optimized Prompt
                          </span>
                          <span className="font-mono text-[11px] text-cyan-400">
                            {caseItem.comparison.optimized.tokenizer_tokens.token_count} tokens (
                            {promptSavedPct > 0 ? `-${promptSavedPct}%` : `${promptSavedPct}%`})
                          </span>
                        </div>
                        <div className="p-3.5 rounded-lg bg-[#071324] border border-cyan-500/20 text-xs font-mono text-cyan-100 whitespace-pre-wrap leading-relaxed min-h-[110px]">
                          {caseItem.comparison.optimized_prompt}
                        </div>
                      </div>
                    </div>

                    {/* Telemetry Comparison Table */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                      <div className="p-3 rounded-lg bg-[#0a101d] border border-white/5 space-y-1">
                        <span className="text-slate-500">Input Prompt Tokens</span>
                        <div className="font-mono text-sm font-semibold text-white">
                          {caseItem.comparison.baseline.provider_usage.prompt_tokens} &rarr;{' '}
                          <span className="text-cyan-400">
                            {caseItem.comparison.optimized.provider_usage.prompt_tokens}
                          </span>
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-[#0a101d] border border-white/5 space-y-1">
                        <span className="text-slate-500">Actual Cost (USD)</span>
                        <div className="font-mono text-sm font-semibold text-white">
                          ${caseItem.comparison.baseline.cost.total_cost_usd.toFixed(6)} &rarr;{' '}
                          <span className="text-emerald-400">
                            ${caseItem.comparison.optimized.cost.total_cost_usd.toFixed(6)}
                          </span>
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-[#0a101d] border border-white/5 space-y-1">
                        <span className="text-slate-500">Inference Latency</span>
                        <div className="font-mono text-sm font-semibold text-white">
                          {caseItem.comparison.baseline.latency_ms}ms &rarr;{' '}
                          <span className="text-amber-300">
                            {caseItem.comparison.optimized.total_latency_ms}ms
                          </span>
                        </div>
                      </div>

                      <div className="p-3 rounded-lg bg-[#0a101d] border border-white/5 space-y-1">
                        <span className="text-slate-500">Strategy & Outcome</span>
                        <div className="font-mono text-xs font-semibold text-cyan-300 uppercase">
                          {caseItem.comparison.optimizer_details.applied_strategy || 'COMPRESS'}
                        </div>
                      </div>
                    </div>

                    {/* Automated Constraint & Quality Checklist */}
                    <div className="p-4 rounded-lg bg-[#0a101d] border border-white/5 space-y-3">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-white flex items-center gap-1.5">
                          <ShieldCheck className="w-4 h-4 text-cyan-400" />
                          Automated Constraint & Safety Verification
                        </span>
                        <span className="font-mono text-[11px] text-slate-400">
                          Score: {(caseItem.optimized_quality.quality_score * 100).toFixed(0)}%
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
                        <div className="flex items-center gap-2 p-2 rounded bg-black/20">
                          {caseItem.optimized_quality.prompt_constraints_preserved ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                          ) : (
                            <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                          )}
                          <span className="text-slate-300">Prompt Constraints Preserved</span>
                        </div>

                        <div className="flex items-center gap-2 p-2 rounded bg-black/20">
                          {caseItem.optimized_quality.output_prohibitions_respected ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                          ) : (
                            <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                          )}
                          <span className="text-slate-300">Prohibitions Honored</span>
                        </div>

                        <div className="flex items-center gap-2 p-2 rounded bg-black/20">
                          {caseItem.optimized_quality.output_format_valid ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                          ) : (
                            <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                          )}
                          <span className="text-slate-300">Format & Schema Valid</span>
                        </div>
                      </div>

                      {caseItem.optimized_quality.details.length > 0 && (
                        <ul className="text-[11px] text-slate-400 space-y-1 list-disc pl-4">
                          {caseItem.optimized_quality.details.map((d, i) => (
                            <li key={i}>{d}</li>
                          ))}
                        </ul>
                      )}
                    </div>

                    {/* LLM Output Preview */}
                    <div className="space-y-2">
                      <div className="text-xs font-semibold text-slate-400 flex items-center gap-1.5">
                        <FileCode className="w-4 h-4 text-slate-500" />
                        Main LLM Answer Output (Optimized Path)
                      </div>
                      <pre className="p-3.5 rounded-lg bg-black/40 border border-white/5 text-xs font-mono text-slate-300 whitespace-pre-wrap overflow-x-auto max-h-60 leading-relaxed">
                        {caseItem.comparison.optimized.llm_response}
                      </pre>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </section>
    </div>
  );
};
