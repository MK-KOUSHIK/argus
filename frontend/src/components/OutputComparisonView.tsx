import React, { useState } from 'react';
import { Card } from './Card';
import { Button } from './Button';
import { ComparisonResult } from '@/types';
import {
  Bot,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Cpu,
  FileText,
  Info,
  Clock,
  Layers,
} from 'lucide-react';

export interface OutputComparisonViewProps {
  comparison: ComparisonResult;
}

export const OutputComparisonView: React.FC<OutputComparisonViewProps> = ({ comparison }) => {
  const [copiedOrig, setCopiedOrig] = useState<boolean>(false);
  const [copiedOpt, setCopiedOpt] = useState<boolean>(false);
  const [showTelemetry, setShowTelemetry] = useState<boolean>(false);
  const [showTrials, setShowTrials] = useState<boolean>(false);

  const { originalExecution, optimizedExecution, metrics, downstreamModel } = comparison;

  const handleCopyText = async (text: string, setCopied: (v: boolean) => void) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const el = document.createElement('textarea');
      el.value = text;
      document.body.appendChild(el);
      el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const formatTokens = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${val} tokens`;
  };

  const formatMs = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return 'N/A';
    return `${val} ms`;
  };

  const trialsCount = originalExecution.trials?.length || 1;

  return (
    <div className="space-y-6" data-testid="output-comparison-view">
      {/* Overview Banner */}
      <div className="p-4 rounded-xl bg-[#091122] border border-cyan-500/30 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                End-to-End Downstream Output Comparison
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-500/40 font-semibold">
                  Pinned Model: {downstreamModel}
                </span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Both prompts were evaluated by local Ollama model <code className="text-cyan-300">{downstreamModel}</code> with shared context window ({originalExecution.effective_settings?.num_ctx || 2048} tokens) and identical generation settings.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {comparison.warmup_attempted && !comparison.warmup_succeeded ? (
              <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-amber-950/80 text-amber-300 border border-amber-500/30 flex items-center gap-1.5" title={comparison.warmup_error || 'Warm-up inference failed'}>
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                Cold-Start Warning (Warm-Up Failed)
              </span>
            ) : (
              <span className="text-[11px] font-mono px-2.5 py-1 rounded bg-slate-900 text-slate-300 border border-white/10 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                {trialsCount > 1 ? `${trialsCount} Paired Trials (Alternating)` : 'Completed (Warm-Up + Paired Run)'}
              </span>
            )}
          </div>
        </div>

        {/* 4-Stat Metric Cards Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {/* Prompt Token Reduction (Finding 3: positive label, no negative sign) */}
          <div className="p-3 rounded-lg bg-[#060a12] border border-white/5 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">
              Prompt Token Reduction
            </span>
            <div className="text-lg font-bold font-mono text-cyan-300">
              {metrics.prompt_token_reduction_percent !== null && metrics.prompt_token_reduction_percent !== undefined
                ? `${metrics.prompt_token_reduction_percent}% saved`
                : 'N/A'}
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {originalExecution.prompt_tokens !== null && optimizedExecution.prompt_tokens !== null
                ? `${originalExecution.prompt_tokens} -> ${optimizedExecution.prompt_tokens} tokens`
                : 'Telemetry unavailable'}
            </div>
          </div>

          {/* Output Tokens Delta */}
          <div className="p-3 rounded-lg bg-[#060a12] border border-white/5 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">
              Output Tokens Generated
            </span>
            <div
              className={`text-lg font-bold font-mono ${
                metrics.output_tokens_diff !== null && metrics.output_tokens_diff < 0
                  ? 'text-cyan-300'
                  : 'text-slate-200'
              }`}
            >
              {metrics.output_tokens_diff !== null && metrics.output_tokens_diff !== undefined
                ? `${metrics.output_tokens_diff > 0 ? '+' : ''}${metrics.output_tokens_diff}`
                : 'N/A'}
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {originalExecution.output_tokens !== null && optimizedExecution.output_tokens !== null
                ? `${originalExecution.output_tokens} -> ${optimizedExecution.output_tokens} tokens`
                : 'Telemetry unavailable'}
            </div>
          </div>

          {/* Total Tokens Saved */}
          <div className="p-3 rounded-lg bg-[#060a12] border border-white/5 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">
              Total Tokens Saved
            </span>
            <div className="text-lg font-bold font-mono text-emerald-400">
              {metrics.total_tokens_saved !== null && metrics.total_tokens_saved !== undefined
                ? `${metrics.total_tokens_saved}`
                : 'N/A'}
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {metrics.total_token_reduction_percent !== null && metrics.total_token_reduction_percent !== undefined
                ? `${metrics.total_token_reduction_percent}% saved`
                : 'Telemetry unavailable'}
            </div>
          </div>

          {/* Latency Delta */}
          <div className="p-3 rounded-lg bg-[#060a12] border border-white/5 space-y-1">
            <span className="text-[10px] uppercase font-mono text-slate-400 tracking-wider">
              Pure Inference Latency Delta
            </span>
            <div
              className={`text-lg font-bold font-mono ${
                metrics.latency_delta_ms !== null && metrics.latency_delta_ms <= 0
                  ? 'text-emerald-400'
                  : 'text-amber-300'
              }`}
            >
              {metrics.latency_delta_ms !== null && metrics.latency_delta_ms !== undefined
                ? `${metrics.latency_delta_ms > 0 ? '+' : ''}${metrics.latency_delta_ms} ms`
                : 'N/A'}
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {originalExecution.duration_ms !== null && optimizedExecution.duration_ms !== null
                ? `${originalExecution.duration_ms}ms -> ${optimizedExecution.duration_ms}ms`
                : 'Telemetry unavailable'}
            </div>
          </div>
        </div>

        {/* Latency Methodology & Variance Notice (Finding 1) */}
        {metrics.latency_note && (
          <div className="p-2.5 rounded-lg bg-[#060a12] border border-white/5 flex items-start gap-2 text-xs text-slate-400">
            <Clock className="w-4 h-4 text-cyan-400 mt-0.5 shrink-0" />
            <div className="space-y-0.5">
              <span className="font-semibold text-slate-300">Latency Measurement Methodology:</span>
              <p className="text-[11px] leading-relaxed">{metrics.latency_note}</p>
            </div>
          </div>
        )}

        {/* Semantic Quality Banner (Finding 6: Honest Report) */}
        <div className="p-2.5 rounded-lg bg-[#070d18] border border-white/5 flex items-start gap-2.5 text-xs text-slate-300">
          <Info className="w-4 h-4 text-cyan-400 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-200">Semantic Equivalence & Output Quality:</span>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-slate-800 text-amber-300 border border-amber-500/20 font-bold">
                {metrics.semantic_quality}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              {metrics.semantic_quality_reason}
            </p>
          </div>
        </div>
      </div>

      {/* Side-by-Side Prompts Sent to Downstream Model */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 uppercase tracking-wider font-mono">
          <FileText className="w-3.5 h-3.5 text-cyan-400" />
          1. Prompts Dispatched to Downstream Model ({downstreamModel})
        </h4>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Execution A Prompt */}
          <Card
            title="Execution A: Original Prompt"
            subtitle="Unmodified user prompt dispatched to downstream LLM"
            badge={
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-white/10">
                {formatTokens(originalExecution.prompt_tokens)}
              </span>
            }
            className="flex flex-col h-full border-white/5 bg-[#080d19]"
          >
            <div className="flex-1 p-3.5 rounded-lg bg-[#050912] border border-white/5 text-slate-300 text-xs font-mono whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[220px]">
              {originalExecution.prompt}
            </div>
          </Card>

          {/* Execution B Prompt (Finding 3: positive saved badge) */}
          <Card
            title="Execution B: Argus-Optimized Prompt"
            subtitle="Rewritten prompt dispatched to the same downstream LLM"
            badge={
              <div className="flex items-center gap-2">
                {metrics.prompt_token_reduction_percent !== null && (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-500/30 font-semibold">
                    {metrics.prompt_token_reduction_percent}% saved
                  </span>
                )}
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30 font-semibold">
                  {formatTokens(optimizedExecution.prompt_tokens)}
                </span>
              </div>
            }
            className="flex flex-col h-full border-cyan-500/20 bg-[#091122]"
          >
            <div className="flex-1 p-3.5 rounded-lg bg-[#060a12] border border-cyan-500/20 text-cyan-100 text-xs font-mono whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[220px]">
              {optimizedExecution.prompt}
            </div>
          </Card>
        </div>
      </div>

      {/* Side-by-Side Actual Downstream LLM Outputs */}
      <div className="space-y-2">
        <h4 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 uppercase tracking-wider font-mono">
          <Bot className="w-3.5 h-3.5 text-cyan-400" />
          2. Downstream LLM Generated Responses
        </h4>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Execution A Response */}
          <Card
            title="Response A (from Original Prompt)"
            subtitle={`Generated by ${originalExecution.model} • Provider: ${originalExecution.provider}`}
            badge={
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopyText(originalExecution.response, setCopiedOrig)}
                  className="h-6 px-2 text-[11px] text-slate-300 hover:text-white"
                >
                  {copiedOrig ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </Button>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-white/10">
                  {formatTokens(originalExecution.output_tokens)}
                </span>
              </div>
            }
            className="flex flex-col h-full border-white/5 bg-[#080d19]"
          >
            <div className="flex-1 p-3.5 rounded-lg bg-[#050912] border border-white/5 text-slate-200 text-xs whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[380px] min-h-[240px]">
              {originalExecution.response}
            </div>

            {/* Execution A Telemetry Footer (Finding 1 & 2: separated durations, Prompt Eval instead of TTFT) */}
            <div className="mt-3 pt-2.5 border-t border-white/5 grid grid-cols-3 gap-2 text-[10px] font-mono text-slate-400">
              <div>
                <span className="text-slate-500">Inference: </span>
                <span className="text-slate-300">{formatMs(originalExecution.duration_ms)}</span>
              </div>
              <div>
                <span className="text-slate-500">Prompt Eval: </span>
                <span className="text-slate-300">{formatMs(originalExecution.prompt_eval_duration_ms)}</span>
              </div>
              <div>
                <span className="text-slate-500">Model Load: </span>
                <span className="text-slate-300">{formatMs(originalExecution.latency_breakdown?.model_load_duration_ms)}</span>
              </div>
            </div>
          </Card>

          {/* Execution B Response */}
          <Card
            title="Response B (from Optimized Prompt)"
            subtitle={`Generated by ${optimizedExecution.model} • Provider: ${optimizedExecution.provider}`}
            badge={
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopyText(optimizedExecution.response, setCopiedOpt)}
                  className="h-6 px-2 text-[11px] text-cyan-300 hover:bg-cyan-500/10 border border-cyan-500/20"
                >
                  {copiedOpt ? (
                    <>
                      <Check className="w-3 h-3 text-emerald-400" />
                      <span className="text-emerald-400">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy</span>
                    </>
                  )}
                </Button>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30 font-semibold">
                  {formatTokens(optimizedExecution.output_tokens)}
                </span>
              </div>
            }
            className="flex flex-col h-full border-cyan-500/20 bg-[#091122]"
          >
            <div className="flex-1 p-3.5 rounded-lg bg-[#060a12] border border-cyan-500/20 text-cyan-50 text-xs whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[380px] min-h-[240px]">
              {optimizedExecution.response}
            </div>

            {/* Execution B Telemetry Footer (Finding 1 & 2) */}
            <div className="mt-3 pt-2.5 border-t border-cyan-500/20 grid grid-cols-3 gap-2 text-[10px] font-mono text-cyan-300/80">
              <div>
                <span className="text-slate-500">Inference: </span>
                <span className="text-cyan-300">{formatMs(optimizedExecution.duration_ms)}</span>
              </div>
              <div>
                <span className="text-slate-500">Prompt Eval: </span>
                <span className="text-cyan-300">{formatMs(optimizedExecution.prompt_eval_duration_ms)}</span>
              </div>
              <div>
                <span className="text-slate-500">Model Load: </span>
                <span className="text-cyan-300">{formatMs(optimizedExecution.latency_breakdown?.model_load_duration_ms)}</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Repeated Trials Breakdown (Finding 1: when trials > 1) */}
      {originalExecution.trials && originalExecution.trials.length > 1 && (
        <div className="border border-white/5 rounded-xl bg-[#080d19] overflow-hidden">
          <button
            type="button"
            onClick={() => setShowTrials(!showTrials)}
            className="w-full px-4 py-3 flex items-center justify-between text-xs font-mono text-slate-300 hover:bg-white/5 transition-colors"
          >
            <span className="flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Multi-Trial Measurements ({originalExecution.trials.length} Paired Alternating Trials)</span>
            </span>
            {showTrials ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showTrials && (
            <div className="p-4 border-t border-white/5 space-y-3 overflow-x-auto text-xs font-mono">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400 text-[11px]">
                    <th className="py-2 px-2">Trial #</th>
                    <th className="py-2 px-2">Exec A Order</th>
                    <th className="py-2 px-2">Exec A Inf. Latency</th>
                    <th className="py-2 px-2">Exec B Order</th>
                    <th className="py-2 px-2">Exec B Inf. Latency</th>
                    <th className="py-2 px-2">Latency Delta</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300">
                  {originalExecution.trials.map((trialA, idx) => {
                    const trialB = optimizedExecution.trials?.[idx];
                    const latA = trialA.latency.inference_duration_ms;
                    const latB = trialB?.latency.inference_duration_ms;
                    const delta = latA !== null && latB !== null && latA !== undefined && latB !== undefined ? round(latB - latA, 2) : 'N/A';
                    return (
                      <tr key={trialA.trial_index}>
                        <td className="py-2 px-2">Trial {trialA.trial_index + 1}</td>
                        <td className="py-2 px-2">Order {trialA.execution_order}</td>
                        <td className="py-2 px-2">{formatMs(latA)}</td>
                        <td className="py-2 px-2">Order {trialB?.execution_order}</td>
                        <td className="py-2 px-2 text-cyan-300">{formatMs(latB)}</td>
                        <td className="py-2 px-2 font-bold">{delta} ms</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Raw Ollama Telemetry Breakdown (Expandable) */}
      <div className="border border-white/5 rounded-xl bg-[#080d19] overflow-hidden">
        <button
          type="button"
          onClick={() => setShowTelemetry(!showTelemetry)}
          className="w-full px-4 py-3 flex items-center justify-between text-xs font-mono text-slate-300 hover:bg-white/5 transition-colors"
        >
          <span className="flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Inspect Raw Ollama Native Telemetry Payload</span>
          </span>
          {showTelemetry ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showTelemetry && (
          <div className="p-4 border-t border-white/5 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div className="space-y-2">
              <h5 className="text-[11px] font-bold text-slate-400 uppercase">Execution A Raw Telemetry</h5>
              <pre className="p-3 rounded-lg bg-[#050912] border border-white/5 text-[11px] text-slate-300 overflow-x-auto">
                {JSON.stringify(originalExecution.native_telemetry || originalExecution, null, 2)}
              </pre>
            </div>
            <div className="space-y-2">
              <h5 className="text-[11px] font-bold text-cyan-400 uppercase">Execution B Raw Telemetry</h5>
              <pre className="p-3 rounded-lg bg-[#050912] border border-white/5 text-[11px] text-cyan-300 overflow-x-auto">
                {JSON.stringify(optimizedExecution.native_telemetry || optimizedExecution, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function round(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}
