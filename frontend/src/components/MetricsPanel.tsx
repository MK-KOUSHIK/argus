import React from 'react';
import { Card } from './Card';
import { OptimizationResult, CompressionSegmentTelemetry } from '@/types';
import {
  TrendingDown,
  TrendingUp,
  CheckCircle,
  ShieldCheck,
  Layers,
  Cpu,
  Clock,
  AlertTriangle,
  Tag,
  Info,
  Sliders,
  Minimize2,
  Route,
} from 'lucide-react';

export interface MetricsPanelProps {
  result: OptimizationResult;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ result }) => {
  const {
    originalMetrics,
    optimizedMetrics,
    tokenReductionPercent,
    tokenChangePercent,
    tokenSavings,
    charDelta,
    preservedConstraints,
    explicitConstraints,
    taskCategory,
    strategy,
    appliedStrategy,
    validationStatus,
    isPreserved,
    mode,
    targetModel,
    executionTimeMs,
    model,
    outputBudget,
    routing,
  } = result;

  // Percentage of original tokens retained (can be > 100% if expanded)
  const retainedPercent = Math.max(
    10,
    Math.round(
      (optimizedMetrics.estimatedTokens / Math.max(1, originalMetrics.estimatedTokens)) * 100
    )
  );

  // Positive charDelta means final > original (expansion)
  const isExpanded = charDelta > 0 || tokenChangePercent > 0;

  // Accurate outcome badge states per Phase 3.2.2 requirements
  const getOutcomeBadge = () => {
    if (validationStatus === 'fallback') {
      return {
        label: 'Validation Fallback',
        icon: <AlertTriangle className="w-3.5 h-3.5" />,
        className: 'bg-amber-950/50 border-amber-500/30 text-amber-300',
      };
    }
    if (isPreserved) {
      return {
        label: 'Original Preserved — No Safe Improvement',
        icon: <CheckCircle className="w-3.5 h-3.5" />,
        className: 'bg-slate-900 border-slate-700/50 text-slate-300',
      };
    }
    if (charDelta < 0) {
      return {
        label: 'Validated Rewrite — Shorter',
        icon: <CheckCircle className="w-3.5 h-3.5" />,
        className: 'bg-emerald-950/50 border-emerald-500/30 text-emerald-300',
      };
    }
    if (charDelta > 0) {
      return {
        label: 'Validated Rewrite — Expanded',
        icon: <Info className="w-3.5 h-3.5" />,
        className: 'bg-amber-950/50 border-amber-500/30 text-amber-300',
      };
    }
    return {
      label: 'Validated Rewrite — Unchanged Length',
      icon: <CheckCircle className="w-3.5 h-3.5" />,
      className: 'bg-cyan-950/50 border-cyan-500/30 text-cyan-300',
    };
  };

  const badgeInfo = getOutcomeBadge();

  return (
    <Card
      title="Optimization & Efficiency Metrics"
      subtitle="Input-token and constraint analysis (heuristic approximation)"
      badge={
        <div className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border ${badgeInfo.className}`}>
          {badgeInfo.icon}
          <span>{badgeInfo.label}</span>
        </div>
      }
      className="space-y-6"
    >
      {/* 4 Primary Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {/* Estimated Token Change Card */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            {isExpanded ? (
              <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
            ) : (
              <TrendingDown className="w-3.5 h-3.5 text-cyan-400" />
            )}
            Estimated Token Change
          </div>
          <div className={`text-2xl font-bold tracking-tight ${
            isExpanded ? 'text-amber-300' : tokenSavings > 0 ? 'text-cyan-300' : 'text-slate-300'
          }`}>
            {tokenSavings > 0 ? `-${tokenReductionPercent}%` : isExpanded ? `+${tokenChangePercent}%` : '0%'}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {tokenSavings > 0
              ? `Saved ~${tokenSavings} tokens (est.)`
              : isExpanded
              ? `Expanded ~${Math.abs(optimizedMetrics.estimatedTokens - originalMetrics.estimatedTokens)} tokens (est.)`
              : 'Preserved original length'}
          </div>
        </div>

        {/* Character Delta Card */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            Character Delta
          </div>
          <div className="text-xl font-bold tracking-tight text-slate-200">
            {originalMetrics.charCount} &rarr; {optimizedMetrics.charCount}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {charDelta < 0
              ? `${Math.abs(charDelta)} chars removed (${charDelta})`
              : charDelta > 0
              ? `${charDelta} chars added (+${charDelta})`
              : 'Exact character parity (0)'}
          </div>
        </div>

        {/* Strategy & Task Category Card */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            <Tag className="w-3.5 h-3.5 text-purple-400" />
            Optimization Strategy
          </div>
          <div className="text-xl font-bold tracking-tight text-purple-300 capitalize">
            {appliedStrategy || strategy}
          </div>
          <div className="text-[11px] font-mono text-slate-400 capitalize">
            Profile: {mode} &bull; {taskCategory.replace('_', ' ')}
          </div>
        </div>

        {/* Local Model & Latency */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Local Inference Latency
          </div>
          <div className="text-xl font-bold tracking-tight text-slate-100 font-mono">
            {executionTimeMs.toLocaleString()} ms
          </div>
          <div className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-cyan-400" />
            <span className="truncate">{model || 'qwen2.5:1.5b'}</span>
          </div>
        </div>
      </div>

      {/* Clean Comparison Visualization Bar */}
      <div className="space-y-2 p-4 rounded-xl bg-[#080d19] border border-white/5">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-200 flex items-center gap-1.5">
            Estimated Token Footprint Comparison
          </span>
          <span className="font-mono text-slate-400 text-[11px]">
            Original: ~{originalMetrics.estimatedTokens} &bull; Final: ~{optimizedMetrics.estimatedTokens} (est. tokens)
          </span>
        </div>

        {/* Visual Token Bar */}
        <div className="space-y-1.5">
          {/* Baseline Bar */}
          <div className="space-y-0.5">
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>Original Prompt (Baseline)</span>
              <span>100%</span>
            </div>
            <div className="w-full h-3 rounded-md bg-[#131e34] overflow-hidden">
              <div className="h-full bg-slate-600 rounded-md w-full" />
            </div>
          </div>

          {/* Compressed or Expanded Bar */}
          <div className="space-y-0.5 pt-1">
            <div className="flex justify-between text-[10px] text-slate-400 font-mono">
              <span className="text-cyan-300 font-medium">Final Payload</span>
              <span className={isExpanded ? 'text-amber-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                {retainedPercent}% {isExpanded ? `(+${tokenChangePercent}% expansion)` : `(-${tokenReductionPercent}% input tokens)`}
              </span>
            </div>
            <div className="w-full h-3.5 rounded-md bg-[#131e34] overflow-hidden flex">
              <div
                className={`h-full transition-all duration-500 rounded-md ${
                  isExpanded
                    ? 'bg-gradient-to-r from-amber-600 to-amber-400'
                    : 'bg-gradient-to-r from-cyan-600 to-cyan-400'
                }`}
                style={{ width: `${Math.min(100, retainedPercent)}%` }}
              />
              {!isExpanded && retainedPercent < 100 && (
                <div
                  className="h-full bg-emerald-500/20 border-l border-emerald-400/30 transition-all duration-500"
                  style={{ width: `${100 - retainedPercent}%` }}
                  title={`Estimated ${tokenReductionPercent}% tokens reduced`}
                />
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between pt-1 text-[10px] text-slate-500">
          <span>{isExpanded ? 'Amber: Clarified and expanded payload' : 'Cyan: Optimized payload'}</span>
          <span>Heuristic tokenization (~4 chars/token)</span>
        </div>
      </div>

      {/* Content-Aware Input Compression Engine (Phase 4.2) */}

      {result.compression && (

        <div className="rounded-xl bg-[#080d19] border border-white/5 p-4 space-y-3">

          <div className="flex items-center justify-between gap-2">

            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">

              <Minimize2 className="w-3.5 h-3.5 text-cyan-400" />

              <span>Content-Aware Compression Engine</span>

            </div>

            <div className="flex items-center gap-2">

              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border ${

                result.compression.applied

                  ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'

                  : 'bg-slate-900 border-slate-700/50 text-slate-400'

              }`}>

                {result.compression.applied ? (

                  <>

                    <CheckCircle className="w-3 h-3 text-emerald-400" />

                    Applied: {result.compression.strategy}

                  </>

                ) : (

                  <>

                    <Info className="w-3 h-3 text-slate-400" />

                    Passthrough / Verbatim

                  </>

                )}

              </span>

              <span className="text-[10px] font-mono text-purple-300 uppercase px-1.5 py-0.5 rounded bg-purple-950/50 border border-purple-500/30">

                {result.compression.content_type}

              </span>

            </div>

          </div>



          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">

            <div className="space-y-1">

              <div className="text-[10px] text-slate-500">Character Reduction</div>

              <div className="text-sm font-mono text-cyan-300 font-bold">

                {result.compression.original_chars} &rarr; {result.compression.compressed_chars}

              </div>

              <div className="text-[10px] text-slate-500 font-mono">

                {result.compression.char_reduction_percent > 0

                  ? `-${result.compression.char_reduction_percent}% reduction`

                  : '0% (verbatim preservation)'}

              </div>

            </div>



            <div className="space-y-1">

              <div className="text-[10px] text-slate-500">Validation & Integrity</div>

              <div className="text-sm font-mono text-slate-200 font-bold flex items-center gap-1">

                {result.compression.validation_passed ? (

                  <span className="text-emerald-400 flex items-center gap-1">

                    <ShieldCheck className="w-3.5 h-3.5" /> Semantic Verified

                  </span>

                ) : (

                  <span className="text-amber-400 flex items-center gap-1">

                    <AlertTriangle className="w-3.5 h-3.5" /> Fallback Triggered

                  </span>

                )}

              </div>

              <div className="text-[10px] text-slate-500">

                AST / JSON / Log error invariant

              </div>

            </div>



            <div className="space-y-1">

              <div className="text-[10px] text-slate-500">Engine Latency</div>

              <div className="text-sm font-mono text-slate-200 font-bold">

                {result.compression.compression_time_ms.toFixed(2)} ms

              </div>

              <div className="text-[10px] text-slate-500">

                Pre-pipeline stage 0

              </div>

            </div>



            <div className="space-y-1">

              <div className="text-[10px] text-slate-500">Reversible Registry</div>

              <div className="text-sm font-mono text-slate-200 font-bold truncate">

                {result.compression.restoration_reference ? (

                  <span className="text-cyan-400 text-xs font-mono" title={result.compression.restoration_reference}>

                    {result.compression.restoration_reference}

                  </span>

                ) : (

                  <span className="text-slate-500 text-xs">None required</span>

                )}

              </div>

              <div className="text-[10px] text-slate-500">

                {result.compression.reversible ? 'Round-trip recoverable' : 'Lossy / passthrough'}

              </div>

            </div>

          </div>



          {result.compression.segments && result.compression.segments.length > 1 && (

            <div className="border-t border-white/5 pt-2 space-y-1.5">

              <div className="text-[10px] font-mono text-slate-400 font-medium">

                Mixed Content Isolated Segments ({result.compression.segments.length}):

              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">

                {result.compression.segments.map((seg: CompressionSegmentTelemetry, idx: number) => (

                  <div key={idx} className="p-2 rounded bg-[#0b1324] border border-white/5 text-[10px] font-mono space-y-0.5">

                    <div className="flex items-center justify-between">

                      <span className="text-purple-300 font-semibold uppercase">{seg.content_type}</span>

                      <span className="text-cyan-300">{seg.strategy}</span>

                    </div>

                    <div className="text-slate-400">

                      {seg.original_chars} &rarr; {seg.compressed_chars} ({seg.savings_percent > 0 ? `-${seg.savings_percent}%` : '0%'})

                    </div>

                  </div>

                ))}

              </div>

            </div>

          )}



          <div className="border-t border-white/5 pt-2 text-[11px] text-slate-400 font-mono">

            <span className="text-slate-500">Diagnostic: </span>

            {result.compression.reason}

          </div>

        </div>

      )}



      {/* Output Budget Control & Generation Telemetry (Phase 4.1) */}
      {outputBudget && (
        <div className="rounded-xl bg-[#080d19] border border-white/5 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              <span>Output Budget Allocation</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-slate-400 capitalize">
                Policy: {outputBudget.policy}
              </span>
              {outputBudget.done_reason && (
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border ${
                  outputBudget.hit_budget_limit
                    ? 'bg-amber-950/40 border-amber-500/30 text-amber-300'
                    : 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                }`}>
                  {outputBudget.hit_budget_limit ? (
                    <>
                      <AlertTriangle className="w-3 h-3 text-amber-400" />
                      Limit Reached ({outputBudget.done_reason})
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-3 h-3 text-emerald-400" />
                      Natural EOS ({outputBudget.done_reason})
                    </>
                  )}
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Allocated Budget</div>
              <div className="text-sm font-mono text-cyan-300 font-bold">
                {outputBudget.allocated_budget} tokens
              </div>
              <div className="text-[10px] text-slate-500">
                {outputBudget.requested_budget ? `Requested: ${outputBudget.requested_budget}` : 'Adaptive allocation'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Generated Tokens</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {outputBudget.actual_generated_tokens ?? 'n/a'}
              </div>
              <div className="text-[10px] text-slate-500">
                Ollama eval_count
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Budget Utilization</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {outputBudget.budget_utilization_percent != null
                  ? `${outputBudget.budget_utilization_percent}%`
                  : 'n/a'}
              </div>
              <div className="text-[10px] text-slate-500">
                {outputBudget.actual_generated_tokens != null
                  ? `${outputBudget.actual_generated_tokens} / ${outputBudget.allocated_budget} tokens`
                  : 'Native counter unavailable'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Generation Rate</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {outputBudget.generation_tokens_per_second != null
                  ? `${outputBudget.generation_tokens_per_second} tok/s`
                  : 'n/a'}
              </div>
              <div className="text-[10px] text-slate-500">
                {outputBudget.generation_duration_ms != null
                  ? `${outputBudget.generation_duration_ms} ms`
                  : 'Duration unavailable'}
              </div>
            </div>
          </div>

          <div className="border-t border-white/5 pt-2 text-[11px] text-slate-400 font-mono">
            <span className="text-slate-500">Rationale: </span>
            {outputBudget.reason}
          </div>
        </div>
      )}

      {/* Phase 5: Local-First Model Routing Decision Card */}
      {routing && (
        <div className="rounded-xl bg-[#080d19] border border-cyan-500/20 p-4 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-2.5">
            <div className="flex items-center gap-2">
              <Route className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-semibold text-slate-100">
                Local Model Routing Decision (Phase 5)
              </span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  routing.routing_status === 'fallback'
                    ? 'bg-amber-950/60 text-amber-300 border-amber-500/40'
                    : routing.routing_status === 'manual_override'
                    ? 'bg-purple-950/60 text-purple-300 border-purple-500/40'
                    : 'bg-cyan-950/60 text-cyan-300 border-cyan-500/40'
                }`}
              >
                {routing.routing_status === 'fallback'
                  ? 'Resource Fallback'
                  : routing.routing_status === 'manual_override'
                  ? 'Manual Override'
                  : 'Auto-Routed'}
              </span>
            </div>
            <div className="text-[11px] font-mono text-slate-400">
              Evaluated in {routing.routing_duration_ms} ms &bull; Tier:{' '}
              <span className="text-cyan-300 uppercase">{routing.target_model_tier}</span>
              {routing.heuristic_margin != null && (
                <span className="text-slate-500">
                  {' '}&bull; Margin: <span className="text-slate-300">±{routing.heuristic_margin.toFixed(2)}</span> (heuristic distance)
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Selected Model</div>
              <div className="text-sm font-mono text-cyan-300 font-bold">
                {routing.selected_model}
              </div>
              <div className="text-[10px] text-slate-500">
                Tier: {routing.target_model_tier}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Executed Model</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {model}
              </div>
              <div className="text-[10px] text-slate-500">
                Active Ollama instance
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Heuristic Complexity</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {((routing.complexity?.heuristic_score ?? 0) * 100).toFixed(0)}% ({(routing.complexity?.heuristic_score ?? 0).toFixed(2)})
              </div>
              <div className="text-[10px] text-slate-500">
                Category: {routing.complexity?.task_category || 'general'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Live Hardware VRAM</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {routing.resources.vram_free_mb != null
                  ? `${routing.resources.vram_free_mb} MiB free`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                {routing.resources.vram_measurement_type === 'measured_resident'
                  ? `Resident: ${routing.resources.measured_vram_mb ?? routing.resources.estimated_model_vram_mb} MiB`
                  : `Proxy: ${routing.resources.estimated_runtime_vram_mb ?? routing.resources.estimated_model_vram_mb ?? 'n/a'} MiB`}
              </div>
            </div>
          </div>

          {routing.complexity.complexity_factors?.length > 0 && (
            <div className="space-y-1 pt-1">
              <div className="text-[10px] text-slate-500 font-medium">Complexity Signals (Rule-Based):</div>
              <div className="flex flex-wrap gap-1.5">
                {routing.complexity.complexity_factors.map((factor, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono bg-[#0c1424] border border-white/5 text-slate-300"
                  >
                    {factor}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-white/5 pt-2 text-[11px] text-slate-400 font-mono space-y-1">
            <div>
              <span className="text-slate-500">Routing Rationale: </span>
              {routing.reason}
            </div>
            {routing.resources.vram_measurement_type && routing.resources.vram_measurement_type !== 'none' && (
              <div className="text-[10px] text-slate-500">
                Memory Telemetry:{' '}
                {routing.resources.vram_measurement_type === 'measured_resident'
                  ? `Measured resident VRAM: ${routing.resources.measured_vram_mb} MiB (active in GPU)`
                  : `Estimated runtime proxy: ${routing.resources.estimated_runtime_vram_mb} MiB (weights: ${routing.resources.weight_footprint_mb} MiB + context/KV buffer)`}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Runtime measurements are native Ollama counters when available. */}
      <div className="rounded-xl bg-[#080d19] border border-white/5 p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs font-semibold text-slate-200">Inference runtime telemetry</div>
          <span className="text-[10px] font-mono text-slate-500">
            {result.inferenceTelemetry?.native_metrics_available ? 'Ollama-reported' : 'Native counters unavailable'}
          </span>
        </div>
        {result.inferenceTelemetry?.calls?.length ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {result.inferenceTelemetry.calls.map((call, index) => (
              <div key={index} className="space-y-1">
                <div className="text-[10px] text-slate-500">Call {index + 1} · queue wait</div>
                <div className="text-sm font-mono text-slate-200">{call.queue_wait_ms} ms</div>
                <div className="text-[10px] text-slate-500">Prompt tokens: {call.prompt_eval_count ?? 'n/a'}</div>
                <div className="text-[10px] text-slate-500">Prompt eval: {call.prompt_eval_duration_ns != null ? (call.prompt_eval_duration_ns / 1e6).toFixed(1) + ' ms' : 'n/a'}</div>
                <div className="text-[10px] text-slate-500">Generated tokens: {call.eval_count ?? 'n/a'}</div>
                <div className="text-[10px] text-slate-500">Decode: {call.generation_tokens_per_second != null ? call.generation_tokens_per_second + ' tok/s' : 'n/a'}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-slate-500">This response did not include native Ollama token/timing counters. The heuristic token estimates above are not runtime KV-cache measurements.</p>
        )}
        {result.kvRuntimePolicy && (
          <div className="border-t border-white/5 pt-2 text-[10px] text-slate-500 space-y-1">
            <div>Selected context: {result.inferenceTelemetry?.calls?.map((call) => call.selected_num_ctx).filter(Boolean).join(', ') || 'not available'} tokens · output limit: {result.kvRuntimePolicy.requested_num_predict}</div>
            <div>Desired KV type: {result.kvRuntimePolicy.configured_kv_cache_type} · Flash Attention: {result.kvRuntimePolicy.configured_flash_attention ? 'enabled in desired server config' : 'not requested'}</div>
            <div>Ollama server settings verified: {result.kvRuntimePolicy.server_settings_verified ? 'yes' : 'no — configure and restart Ollama separately'}</div>
          </div>
        )}
      </div>

      {/* Explicit Constraints & Preserved Guardrails */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pt-1 border-t border-white/5">
        <div className="space-y-2 flex-1">
          <div className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            <span>Preserved Guardrails & Constraints:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {explicitConstraints && explicitConstraints.length > 0 ? (
              explicitConstraints.map((constraint, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono bg-cyan-950/40 border border-cyan-500/30 text-cyan-200"
                >
                  <ShieldCheck className="w-3 h-3 text-cyan-400" />
                  {constraint}
                </span>
              ))
            ) : (
              preservedConstraints.map((constraint, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono bg-cyan-950/40 border border-cyan-500/30 text-cyan-200"
                >
                  <ShieldCheck className="w-3 h-3 text-cyan-400" />
                  {constraint}
                </span>
              ))
            )}
          </div>
        </div>

        <div className="text-right text-[11px] font-mono text-slate-500 space-y-0.5 shrink-0">
          <div>Profile: <span className="text-slate-300 capitalize">{mode}</span></div>
          <div>Destination: <span className="text-slate-300">{targetModel}</span></div>
          <div>Validation: <span className={validationStatus === 'passed' ? 'text-emerald-400' : 'text-amber-400'}>{validationStatus}</span></div>
        </div>
      </div>
    </Card>
  );
};
