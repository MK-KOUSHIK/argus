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
  result?: OptimizationResult | null;
}

/**
 * Safely format a number using .toFixed(digits).
 * Returns 'N/A' (or custom fallback) when value is null, undefined, NaN, or non-finite.
 * Does NOT convert missing metrics to zero.
 */
export function formatMetricNumber(
  val: number | null | undefined,
  digits: number = 2,
  fallback: string = 'N/A'
): string {
  if (val == null || typeof val !== 'number' || !Number.isFinite(val)) {
    return fallback;
  }
  return val.toFixed(digits);
}

export interface MetricsPanelErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export interface MetricsPanelErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class MetricsPanelErrorBoundary extends React.Component<
  MetricsPanelErrorBoundaryProps,
  MetricsPanelErrorBoundaryState
> {
  constructor(props: MetricsPanelErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): MetricsPanelErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('MetricsPanel caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <Card
          title="Optimization & Efficiency Metrics"
          subtitle="Metrics display encountered an unexpected issue"
          className="space-y-4 border-amber-500/30"
        >
          <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/20 text-amber-200 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              Unable to display complete metrics panel ({this.state.error?.message || 'Display error'}). Optimization results remain available.
            </span>
          </div>
        </Card>
      );
    }
    return this.props.children;
  }
}

const MetricsPanelInner: React.FC<MetricsPanelProps> = ({ result }) => {
  if (!result) {
    return null;
  }

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
    compression,
    inferenceTelemetry,
    kvRuntimePolicy,
    originalTokenCount,
    candidateTokenCount,
    dispatchedTokenCount,
    actualTokensSaved,
    actualReductionPercent,
    candidateAccepted,
    rejectionReason,
    tokenMeasurementMethod,
  } = result;

  const origTokens = originalTokenCount ?? originalMetrics?.estimatedTokens;
  const optTokens = dispatchedTokenCount ?? optimizedMetrics?.estimatedTokens;
  const origChars = originalMetrics?.charCount;
  const optChars = optimizedMetrics?.charCount;

  // Effective token savings and reduction (guaranteed 0 if candidate was rejected)
  const isRejected = candidateAccepted === false;
  const effectiveTokensSaved = isRejected ? 0 : (actualTokensSaved ?? tokenSavings ?? 0);
  const effectiveReductionPercent = isRejected ? 0 : (actualReductionPercent ?? tokenReductionPercent ?? 0);
  const effectiveChangePercent = isRejected ? 0 : (tokenChangePercent ?? 0);

  // Percentage of original tokens retained (can be > 100% if expanded)
  const retainedPercent =
    isRejected
      ? 100
      : Number.isFinite(origTokens) && Number.isFinite(optTokens) && origTokens! > 0
      ? Math.max(10, Math.round((optTokens! / Math.max(1, origTokens!)) * 100))
      : 100;

  // Positive charDelta means final > original (expansion)
  const isExpanded =
    !isRejected &&
    ((Number.isFinite(charDelta) && charDelta! > 0) ||
      (Number.isFinite(effectiveChangePercent) && effectiveChangePercent > 0));

  // Accurate outcome badge states per Phase 3.2.2 requirements
  const getOutcomeBadge = () => {
    if (candidateAccepted === false) {
      return {
        label: 'Candidate Rejected — Original Preserved',
        icon: <AlertTriangle className="w-3.5 h-3.5" />,
        className: 'bg-amber-950/50 border-amber-500/30 text-amber-300',
      };
    }
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
    if (Number.isFinite(charDelta) && charDelta! < 0) {
      return {
        label: 'Validated Rewrite — Shorter',
        icon: <CheckCircle className="w-3.5 h-3.5" />,
        className: 'bg-emerald-950/50 border-emerald-500/30 text-emerald-300',
      };
    }
    if (Number.isFinite(charDelta) && charDelta! > 0) {
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
      subtitle={
        tokenMeasurementMethod === 'ollama_native'
          ? 'Input-token and constraint analysis (Ollama native tokenizer)'
          : tokenMeasurementMethod === 'tiktoken_cl100k'
          ? 'Input-token and constraint analysis (tiktoken cl100k)'
          : 'Input-token and constraint analysis (heuristic approximation)'
      }
      badge={
        <div className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border ${badgeInfo.className}`}>
          {badgeInfo.icon}
          <span>{badgeInfo.label}</span>
        </div>
      }
      className="space-y-6"
    >
      {/* Candidate Rejection Alert Banner */}
      {isRejected && (
        <div className="p-3.5 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold text-amber-300">
              Candidate Rewrite Rejected — Original Preserved
            </div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              {rejectionReason || 'The candidate rewrite was equal to or longer than the original prompt with no safe token reduction. To prevent regression, the original prompt was safely dispatched downstream.'}
            </p>
            {candidateTokenCount != null && (
              <div className="text-[10px] font-mono text-amber-400/80 pt-0.5">
                Original: {origTokens} tokens &bull; Candidate: {candidateTokenCount} tokens &bull; Dispatched: {optTokens} tokens
              </div>
            )}
          </div>
        </div>
      )}

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
            isExpanded ? 'text-amber-300' : effectiveTokensSaved > 0 ? 'text-cyan-300' : 'text-slate-300'
          }`}>
            {effectiveTokensSaved > 0
              ? `-${formatMetricNumber(effectiveReductionPercent, 0)}%`
              : isExpanded && Number.isFinite(effectiveChangePercent)
              ? `+${formatMetricNumber(effectiveChangePercent, 0)}%`
              : (tokenSavings != null || tokenReductionPercent != null || isRejected)
              ? '0%'
              : 'N/A'}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {isRejected
              ? 'Candidate rejected: original retained'
              : effectiveTokensSaved > 0
              ? `Saved ~${effectiveTokensSaved} tokens`
              : isExpanded && Number.isFinite(optTokens) && Number.isFinite(origTokens)
              ? `Expanded ~${Math.abs(optTokens! - origTokens!)} tokens`
              : Number.isFinite(tokenSavings)
              ? 'Preserved original length'
              : 'N/A'}
          </div>
        </div>

        {/* Character Delta Card */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-slate-400" />
            Character Delta
          </div>
          <div className="text-xl font-bold tracking-tight text-slate-200">
            {Number.isFinite(origChars) && Number.isFinite(optChars)
              ? `${origChars} → ${optChars}`
              : 'N/A'}
          </div>
          <div className="text-[11px] font-mono text-slate-500">
            {!Number.isFinite(charDelta)
              ? 'N/A'
              : charDelta! < 0
              ? `${Math.abs(charDelta!)} chars removed (${charDelta})`
              : charDelta! > 0
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
            {appliedStrategy || strategy || 'N/A'}
          </div>
          <div className="text-[11px] font-mono text-slate-400 capitalize">
            Profile: {mode || 'balanced'} &bull; {(taskCategory || 'general').replace('_', ' ')}
          </div>
        </div>

        {/* Local Model & Latency */}
        <div className="p-3.5 rounded-xl bg-[#080d19] border border-white/5 space-y-1">
          <div className="text-[11px] font-medium text-slate-400 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            Local Inference Latency
          </div>
          <div className="text-xl font-bold tracking-tight text-slate-100 font-mono">
            {Number.isFinite(executionTimeMs)
              ? `${executionTimeMs!.toLocaleString()} ms`
              : 'N/A'}
          </div>
          <div className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-cyan-400" />
            <span className="truncate">{model || targetModel || 'qwen2.5:1.5b'}</span>
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
            Original: ~{Number.isFinite(origTokens) ? origTokens : 'N/A'} &bull; Final: ~{Number.isFinite(optTokens) ? optTokens : 'N/A'} tokens
            {candidateTokenCount != null && isRejected && ` (Candidate: ~${candidateTokenCount})`}
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
              <span className={isRejected ? 'text-slate-400 font-semibold' : isExpanded ? 'text-amber-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                {retainedPercent}%{' '}
                {isRejected
                  ? '(original preserved)'
                  : isExpanded
                  ? `(+${formatMetricNumber(effectiveChangePercent, 0)}% expansion)`
                  : `(-${formatMetricNumber(effectiveReductionPercent, 0)}% input tokens)`}
              </span>
            </div>
            <div className="w-full h-3.5 rounded-md bg-[#131e34] overflow-hidden flex">
              <div
                className={`h-full transition-all duration-500 rounded-md ${
                  isRejected
                    ? 'bg-slate-600'
                    : isExpanded
                    ? 'bg-gradient-to-r from-amber-600 to-amber-400'
                    : 'bg-gradient-to-r from-cyan-600 to-cyan-400'
                }`}
                style={{ width: `${Math.min(100, retainedPercent)}%` }}
              />
              {!isExpanded && !isRejected && retainedPercent < 100 && (
                <div
                  className="h-full bg-emerald-500/20 border-l border-emerald-400/30 transition-all duration-500"
                  style={{ width: `${100 - retainedPercent}%` }}
                  title={`Estimated ${effectiveReductionPercent ?? 'N/A'}% tokens reduced`}
                />
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between pt-1 text-[10px] text-slate-500">
          <span>{isRejected ? 'Slate: Original preserved (candidate rejected)' : isExpanded ? 'Amber: Clarified and expanded payload' : 'Cyan: Optimized payload'}</span>
          <span className="font-mono">
            {tokenMeasurementMethod === 'ollama_native'
              ? 'Native Ollama tokenizer'
              : tokenMeasurementMethod === 'tiktoken_cl100k'
              ? 'tiktoken (cl100k_base)'
              : 'Heuristic tokenization (~4 chars/token)'}
          </span>
        </div>
      </div>

      {/* Content-Aware Input Compression Engine (Phase 4.2) */}
      {compression && (
        <div className="rounded-xl bg-[#080d19] border border-white/5 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
              <Minimize2 className="w-3.5 h-3.5 text-cyan-400" />
              <span>Content-Aware Compression Engine</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border ${
                (compression.applied ?? compression.enabled)
                  ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                  : 'bg-slate-900 border-slate-700/50 text-slate-400'
              }`}>
                {(compression.applied ?? compression.enabled) ? (
                  <>
                    <CheckCircle className="w-3 h-3 text-emerald-400" />
                    Applied: {compression.strategy ?? compression.strategy_applied ?? 'Verbatim'}
                  </>
                ) : (
                  <>
                    <Info className="w-3 h-3 text-slate-400" />
                    Passthrough / Verbatim
                  </>
                )}
              </span>
              <span className="text-[10px] font-mono text-purple-300 uppercase px-1.5 py-0.5 rounded bg-purple-950/50 border border-purple-500/30">
                {compression.content_type ?? compression.detected_content_type ?? 'general'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Character / Token Reduction</div>
              <div className="text-sm font-mono text-cyan-300 font-bold">
                {(() => {
                  const o = compression.original_chars ?? compression.original_tokens;
                  const c = compression.compressed_chars ?? compression.compressed_tokens;
                  return Number.isFinite(o) && Number.isFinite(c)
                    ? `${o} → ${c}`
                    : 'N/A';
                })()}
              </div>
              <div className="text-[10px] text-slate-500 font-mono">
                {(() => {
                  const red = compression.char_reduction_percent ?? compression.reduction_percent;
                  if (!Number.isFinite(red)) return 'N/A';
                  return red! > 0
                    ? `-${formatMetricNumber(red, 1)}% reduction`
                    : '0% (verbatim preservation)';
                })()}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Validation & Integrity</div>
              <div className="text-sm font-mono text-slate-200 font-bold flex items-center gap-1">
                {(compression.validation_passed ?? (compression.validation_status === 'passed')) ? (
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
                {(() => {
                  const lat = compression.compression_time_ms ?? compression.processing_time_ms;
                  return Number.isFinite(lat)
                    ? `${formatMetricNumber(lat, 2)} ms`
                    : 'N/A';
                })()}
              </div>
              <div className="text-[10px] text-slate-500">
                Pre-pipeline stage 0
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Reversible Registry</div>
              <div className="text-sm font-mono text-slate-200 font-bold truncate">
                {compression.restoration_reference ? (
                  <span className="text-cyan-400 text-xs font-mono" title={compression.restoration_reference}>
                    {compression.restoration_reference}
                  </span>
                ) : (
                  <span className="text-slate-500 text-xs">None required</span>
                )}
              </div>
              <div className="text-[10px] text-slate-500">
                {(compression.reversible ?? Boolean(compression.restoration_reference))
                  ? 'Round-trip recoverable'
                  : 'Lossy / passthrough'}
              </div>
            </div>
          </div>

          {compression.segments && compression.segments.length > 1 && (
            <div className="border-t border-white/5 pt-2 space-y-1.5">
              <div className="text-[10px] font-mono text-slate-400 font-medium">
                Mixed Content Isolated Segments ({compression.segments.length}):
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
                {compression.segments.map((seg: CompressionSegmentTelemetry, idx: number) => (
                  <div key={idx} className="p-2 rounded bg-[#0b1324] border border-white/5 text-[10px] font-mono space-y-0.5">
                    <div className="flex items-center justify-between">
                      <span className="text-purple-300 font-semibold uppercase">{seg.content_type || 'segment'}</span>
                      <span className="text-cyan-300">{seg.strategy || 'passthrough'}</span>
                    </div>
                    <div className="text-slate-400">
                      {Number.isFinite(seg.original_chars) && Number.isFinite(seg.compressed_chars)
                        ? `${seg.original_chars} → ${seg.compressed_chars}`
                        : 'N/A'}{' '}
                      ({Number.isFinite(seg.savings_percent)
                        ? seg.savings_percent > 0
                          ? `-${formatMetricNumber(seg.savings_percent, 1)}%`
                          : '0%'
                        : 'N/A'})
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="border-t border-white/5 pt-2 text-[11px] text-slate-400 font-mono">
            <span className="text-slate-500">Diagnostic: </span>
            {compression.reason ||
              compression.fallback_reason ||
              ((compression.applied ?? compression.enabled)
                ? 'Content-aware compression applied successfully'
                : 'Original content preserved verbatim')}
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
                Policy: {outputBudget.policy || 'standard'}
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
                {Number.isFinite(outputBudget.allocated_budget)
                  ? `${outputBudget.allocated_budget} tokens`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                {outputBudget.requested_budget ? `Requested: ${outputBudget.requested_budget}` : 'Adaptive allocation'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Generated Tokens</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {Number.isFinite(outputBudget.actual_generated_tokens) ? outputBudget.actual_generated_tokens : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                Ollama eval_count
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Budget Utilization</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {Number.isFinite(outputBudget.budget_utilization_percent)
                  ? `${formatMetricNumber(outputBudget.budget_utilization_percent, 1)}%`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                {Number.isFinite(outputBudget.actual_generated_tokens) && Number.isFinite(outputBudget.allocated_budget)
                  ? `${outputBudget.actual_generated_tokens} / ${outputBudget.allocated_budget} tokens`
                  : 'Native counter unavailable'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Generation Rate</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {Number.isFinite(outputBudget.generation_tokens_per_second)
                  ? `${formatMetricNumber(outputBudget.generation_tokens_per_second, 1)} tok/s`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                {Number.isFinite(outputBudget.generation_duration_ms)
                  ? `${formatMetricNumber(outputBudget.generation_duration_ms, 1)} ms`
                  : 'Duration unavailable'}
              </div>
            </div>
          </div>

          <div className="border-t border-white/5 pt-2 text-[11px] text-slate-400 font-mono">
            <span className="text-slate-500">Rationale: </span>
            {outputBudget.reason || 'Adaptive output budget control'}
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
              Evaluated in {Number.isFinite(routing.routing_duration_ms) ? routing.routing_duration_ms : 'N/A'} ms &bull; Tier:{' '}
              <span className="text-cyan-300 uppercase">{routing.target_model_tier || 'UNKNOWN'}</span>
              {Number.isFinite(routing.heuristic_margin) && (
                <span className="text-slate-500">
                  {' '}&bull; Margin: <span className="text-slate-300">±{formatMetricNumber(routing.heuristic_margin, 2)}</span> (heuristic distance)
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Selected Model</div>
              <div className="text-sm font-mono text-cyan-300 font-bold">
                {routing.selected_model || 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                Tier: {routing.target_model_tier || 'N/A'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Executed Model</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {model || routing.selected_model || 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                Active Ollama instance
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Heuristic Complexity</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {Number.isFinite(routing.complexity?.heuristic_score)
                  ? `${formatMetricNumber(routing.complexity!.heuristic_score! * 100, 0)}% (${formatMetricNumber(routing.complexity!.heuristic_score, 2)})`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                Category: {routing.complexity?.task_category || 'general'}
              </div>
            </div>

            <div className="space-y-1">
              <div className="text-[10px] text-slate-500">Live Hardware VRAM</div>
              <div className="text-sm font-mono text-slate-200 font-bold">
                {Number.isFinite(routing.resources?.vram_free_mb)
                  ? `${routing.resources.vram_free_mb} MiB free`
                  : 'N/A'}
              </div>
              <div className="text-[10px] text-slate-500">
                {routing.resources?.vram_measurement_type === 'measured_resident'
                  ? `Resident: ${routing.resources.measured_vram_mb ?? routing.resources.estimated_model_vram_mb ?? 'N/A'} MiB`
                  : routing.resources?.vram_measurement_type === 'cpu_only'
                  ? 'CPU Mode'
                  : `Proxy: ${routing.resources?.estimated_runtime_vram_mb ?? routing.resources?.estimated_model_vram_mb ?? 'N/A'} MiB`}
              </div>
            </div>
          </div>

          {routing.complexity?.complexity_factors && routing.complexity.complexity_factors.length > 0 && (
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
              {routing.reason || 'Optimal resource-aware model routing'}
            </div>
            {routing.resources?.vram_measurement_type && routing.resources.vram_measurement_type !== 'none' && (
              <div className="text-[10px] text-slate-500">
                Memory Telemetry:{' '}
                {routing.resources.vram_measurement_type === 'measured_resident'
                  ? `Measured resident VRAM: ${routing.resources.measured_vram_mb ?? 'N/A'} MiB (active in GPU)`
                  : routing.resources.vram_measurement_type === 'architecture_proxy'
                  ? `Architecture proxy: ${routing.resources.estimated_runtime_vram_mb ?? 'N/A'} MiB (weights: ${routing.resources.weight_footprint_mb ?? 'N/A'} MiB + calculated KV cache)`
                  : routing.resources.vram_measurement_type === 'cpu_only'
                  ? `CPU execution admitted to host RAM (${routing.resources.weight_footprint_mb ?? 'N/A'} MiB weights; GPU not utilized)`
                  : routing.resources.vram_measurement_type === 'unverified'
                  ? 'Hardware telemetry unverified (measurements missing or incomplete)'
                  : `Rough proxy (estimate): ${routing.resources.estimated_runtime_vram_mb ?? 'N/A'} MiB`}
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
            {inferenceTelemetry?.native_metrics_available ? 'Ollama-reported' : 'Native counters unavailable'}
          </span>
        </div>
        {inferenceTelemetry?.calls?.length ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {inferenceTelemetry.calls.map((call, index) => (
              <div key={index} className="space-y-1">
                <div className="text-[10px] text-slate-500">Call {index + 1} · queue wait</div>
                <div className="text-sm font-mono text-slate-200">{Number.isFinite(call.queue_wait_ms) ? `${call.queue_wait_ms} ms` : 'N/A'}</div>
                <div className="text-[10px] text-slate-500">Prompt tokens: {call.prompt_eval_count ?? 'N/A'}</div>
                <div className="text-[10px] text-slate-500">
                  Prompt eval: {Number.isFinite(call.prompt_eval_duration_ns) ? `${formatMetricNumber(call.prompt_eval_duration_ns! / 1e6, 1)} ms` : 'N/A'}
                </div>
                <div className="text-[10px] text-slate-500">Generated tokens: {call.eval_count ?? 'N/A'}</div>
                <div className="text-[10px] text-slate-500">
                  Decode: {Number.isFinite(call.generation_tokens_per_second) ? `${formatMetricNumber(call.generation_tokens_per_second, 1)} tok/s` : 'N/A'}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-[11px] text-slate-500">This response did not include native Ollama token/timing counters. The heuristic token estimates above are not runtime KV-cache measurements.</p>
        )}
        {kvRuntimePolicy && (
          <div className="border-t border-white/5 pt-2 text-[10px] text-slate-500 space-y-1">
            <div>Selected context: {inferenceTelemetry?.calls?.map((call) => call.selected_num_ctx).filter(Boolean).join(', ') || 'not available'} tokens · output limit: {kvRuntimePolicy.requested_num_predict ?? 'N/A'}</div>
            <div>Desired KV type: {kvRuntimePolicy.configured_kv_cache_type || 'standard'} · Flash Attention: {kvRuntimePolicy.configured_flash_attention ? 'enabled in desired server config' : 'not requested'}</div>
            <div>Ollama server settings verified: {kvRuntimePolicy.server_settings_verified ? 'yes' : 'no — configure and restart Ollama separately'}</div>
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
            ) : preservedConstraints && preservedConstraints.length > 0 ? (
              preservedConstraints.map((constraint, idx) => (
                <span
                  key={idx}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono bg-cyan-950/40 border border-cyan-500/30 text-cyan-200"
                >
                  <ShieldCheck className="w-3 h-3 text-cyan-400" />
                  {constraint}
                </span>
              ))
            ) : (
              <span className="text-slate-500 text-xs italic">No specific constraints detected</span>
            )}
          </div>
        </div>

        <div className="text-right text-[11px] font-mono text-slate-500 space-y-0.5 shrink-0">
          <div>Profile: <span className="text-slate-300 capitalize">{mode || 'balanced'}</span></div>
          <div>Destination: <span className="text-slate-300">{targetModel || 'unknown'}</span></div>
          <div>Validation: <span className={validationStatus === 'passed' ? 'text-emerald-400' : 'text-amber-400'}>{validationStatus || 'N/A'}</span></div>
        </div>
      </div>
    </Card>
  );
};

export const MetricsPanel: React.FC<MetricsPanelProps> = (props) => {
  return (
    <MetricsPanelErrorBoundary>
      <MetricsPanelInner {...props} />
    </MetricsPanelErrorBoundary>
  );
};
