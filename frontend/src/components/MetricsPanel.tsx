import React from 'react';
import { Card } from './Card';
import { OptimizationResult } from '@/types';
import { TrendingDown, TrendingUp, CheckCircle, ShieldCheck, Layers, Cpu, Clock, AlertTriangle, Tag, Info } from 'lucide-react';

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
