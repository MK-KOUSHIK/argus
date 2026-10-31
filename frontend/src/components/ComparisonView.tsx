import React, { useState } from 'react';
import { Card } from './Card';
import { Button } from './Button';
import { TokenEstimation, OptimizationStrategy, ValidationStatus } from '@/types';
import { Copy, Check, Edit3, ArrowRight, Layers, FileCode } from 'lucide-react';

export interface ComparisonViewProps {
  originalPrompt: string;
  optimizedPrompt: string;
  onOptimizedPromptChange: (val: string) => void;
  originalMetrics: TokenEstimation;
  optimizedMetrics: TokenEstimation;
  tokenReductionPercent: number;
  tokenSavings: number;
  tokenChangePercent?: number;
  strategy?: OptimizationStrategy;
  isPreserved?: boolean;
  validationStatus?: ValidationStatus;
  model?: string;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  originalPrompt,
  optimizedPrompt,
  onOptimizedPromptChange,
  originalMetrics,
  optimizedMetrics,
  tokenReductionPercent,
  tokenSavings,
  tokenChangePercent = 0,
  strategy = 'compress',
  isPreserved = false,
  validationStatus = 'passed',
  model = 'qwen2.5:1.5b',
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [isEditing, setIsEditing] = useState<boolean>(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(optimizedPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textArea = document.createElement('textarea');
      textArea.value = optimizedPrompt;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="space-y-4">
      {/* Header bar with comparison summary */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#0a1222] border border-white/5">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-semibold text-slate-100">
            Prompt Comparison & Diff View
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-500/30 uppercase tracking-wider">
            Strategy: {strategy}
          </span>
          {isPreserved && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-500/30">
              Preserved
            </span>
          )}
          {validationStatus === 'fallback' && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-950 text-amber-300 border border-amber-500/30">
              Fallback Baseline
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-mono bg-[#060a12] px-2.5 py-1 rounded-md border border-white/10 text-slate-300">
            <span>{originalMetrics.estimatedTokens} est. tokens</span>
            <ArrowRight className="w-3.5 h-3.5 text-cyan-400" />
            <span className={tokenSavings >= 0 ? 'text-cyan-300 font-semibold' : 'text-amber-300 font-semibold'}>
              {optimizedMetrics.estimatedTokens} est. tokens
            </span>
          </div>

          <div
            className={`text-xs font-mono px-2 py-1 rounded-md font-semibold ${
              tokenSavings > 0
                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/30'
                : tokenSavings < 0
                ? 'bg-amber-950/60 text-amber-400 border border-amber-500/30'
                : 'bg-slate-800 text-slate-300 border border-slate-700'
            }`}
          >
            {tokenSavings > 0
              ? `-${tokenReductionPercent}% estimated tokens`
              : tokenSavings < 0
              ? `+${tokenChangePercent}% estimated expansion`
              : '0% change (preserved)'}
          </div>
        </div>
      </div>

      {/* Side-by-side prompt views */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: Original Prompt (Read-only reference) */}
        <Card
          title="Original Prompt (Pre-Optimization)"
          subtitle="Reference input payload"
          badge={
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-white/10">
              ~{originalMetrics.estimatedTokens} tokens (est.)
            </span>
          }
          className="flex flex-col h-full border-white/5 bg-[#080d19]"
        >
          <div className="flex-1 p-3.5 rounded-lg bg-[#050912] border border-white/5 text-slate-300 text-xs font-mono whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[360px] min-h-[220px]">
            {originalPrompt}
          </div>

          <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span>{originalMetrics.charCount} characters</span>
            <span>Unmodified User Input</span>
          </div>
        </Card>

        {/* Right: Optimized Prompt (Interactive / Copyable / Editable) */}
        <Card
          title={isPreserved ? 'Preserved Prompt (Original Retained)' : 'Optimized Prompt (Rewritten)'}
          subtitle={
            isPreserved
              ? 'Original retained to guarantee safety and prevent distortion'
              : 'Actionable payload optimized for downstream execution'
          }
          badge={
            <div className="flex items-center gap-1.5">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="h-6 px-2 text-[11px] text-cyan-300 hover:bg-cyan-500/10 border border-cyan-500/20"
                aria-label="Copy optimized prompt"
              >
                {copied ? (
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
              <button
                type="button"
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center gap-1 h-6 px-2 text-[11px] rounded font-medium bg-[#0b172a] hover:bg-[#112340] text-slate-300 border border-white/10 transition-colors"
                title="Toggle manual edits"
              >
                <Edit3 className="w-3 h-3" />
                <span>{isEditing ? 'Done Editing' : 'Edit'}</span>
              </button>
              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-500/30 font-semibold">
                ~{optimizedMetrics.estimatedTokens} tokens (est.)
              </span>
            </div>
          }
          className="flex flex-col h-full border-cyan-500/20 bg-[#091122]"
        >
          {isEditing ? (
            <textarea
              value={optimizedPrompt}
              onChange={(e) => onOptimizedPromptChange(e.target.value)}
              className="flex-1 p-3.5 rounded-lg bg-[#060a12] border border-cyan-500/40 text-cyan-100 text-xs font-mono whitespace-pre-wrap leading-relaxed resize-none focus:outline-none focus:ring-1 focus:ring-cyan-400 max-h-[360px] min-h-[220px]"
              aria-label="Editable optimized prompt"
            />
          ) : (
            <div className="flex-1 p-3.5 rounded-lg bg-[#060a12] border border-cyan-500/20 text-cyan-100 text-xs font-mono whitespace-pre-wrap leading-relaxed overflow-y-auto max-h-[360px] min-h-[220px] selection:bg-cyan-500/30 selection:text-cyan-200">
              {optimizedPrompt}
            </div>
          )}

          <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span className="flex items-center gap-1 text-cyan-400">
              <FileCode className="w-3 h-3" />
              <span>{isEditing ? 'Manual fine-tuning active' : 'Click "Edit" to modify before copying'}</span>
            </span>
            <span className="text-slate-400">
              {isPreserved
                ? 'Prompt preserved as safe baseline'
                : `Local Engine: ${model} · Strategy: ${strategy}`}
            </span>
          </div>
        </Card>
      </div>

      {/* Token Approximation Notice */}
      <div className="text-[11px] text-slate-500 italic px-1">
        * Token counts are approximations based on standard ~4 characters per token heuristics.
      </div>
    </div>
  );
};
