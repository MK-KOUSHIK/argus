import React from 'react';
import { Card } from './Card';
import { Button } from './Button';
import { ILLUSTRATIVE_MODELS } from '@/services/mockOptimizer';
import { OptimizationMode, TargetModel, OllamaStatus } from '@/types';
import { Zap, Sliders, ShieldAlert, AlertCircle, CheckCircle } from 'lucide-react';

export interface OptimizationControlsProps {
  mode: OptimizationMode;
  onModeChange: (mode: OptimizationMode) => void;
  targetModel: TargetModel;
  onTargetModelChange: (model: TargetModel) => void;
  onOptimize: () => void;
  isLoading: boolean;
  disabled: boolean;
  ollamaStatus?: OllamaStatus | null;
}

export const OptimizationControls: React.FC<OptimizationControlsProps> = ({
  mode,
  onModeChange,
  targetModel,
  onTargetModelChange,
  onOptimize,
  isLoading,
  disabled,
  ollamaStatus,
}) => {
  const isOllamaReady = Boolean(ollamaStatus?.connected && ollamaStatus?.model_installed);

  const modes: { id: OptimizationMode; title: string; desc: string }[] = [
    {
      id: 'balanced',
      title: 'Balanced',
      desc: 'Moderate compression, removes verbosity while maintaining prose flow (~25% target)',
    },
    {
      id: 'concise',
      title: 'Concise',
      desc: 'Aggressive compression, imperative directives, minimal conversational fluff (~40% target)',
    },
    {
      id: 'structure',
      title: 'Structure-focused',
      desc: 'Groups constraints into markdown headers and schema blocks for maximum parser clarity (~30% target)',
    },
  ];

  return (
    <Card
      title="Optimization Controls"
      subtitle="Configure compression profile and target frontier model"
      className="space-y-5"
    >
      {/* Local LLM Inference Engine Status Banner */}
      {isOllamaReady ? (
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-xs text-cyan-200">
          <CheckCircle className="w-4 h-4 text-cyan-400 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <span className="font-semibold text-cyan-300">
              Local LLM Active ({ollamaStatus?.model || 'qwen2.5:1.5b'})
            </span>
            <p className="text-[11px] text-cyan-200/80 leading-relaxed">
              Connected to local Ollama runtime. Real-time prompt optimization runs entirely on your
              machine with zero prompt data sent to the cloud.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-950/40 border border-amber-500/30 text-xs text-amber-200">
          <AlertCircle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
          <div className="space-y-0.5">
            <span className="font-semibold text-amber-300">Local Model Offline</span>
            <p className="text-[11px] text-amber-200/80 leading-relaxed">
              {ollamaStatus?.error ||
                "Cannot connect to Ollama at http://127.0.0.1:11434. Ensure 'ollama serve' is active and 'qwen2.5:1.5b' is downloaded."}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Mode Selector */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-cyan-400" />
            Optimization Mode
          </label>
          <div className="space-y-1.5">
            {modes.map((m) => (
              <label
                key={m.id}
                className={`flex items-start gap-3 p-2.5 rounded-lg border cursor-pointer transition-all ${
                  mode === m.id
                    ? 'bg-cyan-950/50 border-cyan-500/50 text-slate-100'
                    : 'bg-[#080d19] border-white/5 text-slate-400 hover:border-white/10 hover:bg-[#0c1424]'
                }`}
              >
                <input
                  type="radio"
                  name="optimization-mode"
                  value={m.id}
                  checked={mode === m.id}
                  onChange={() => onModeChange(m.id)}
                  className="mt-1 text-cyan-500 focus:ring-cyan-400 focus:ring-offset-0 bg-[#080d19] border-white/20"
                />
                <div className="space-y-0.5">
                  <div className="text-xs font-medium text-slate-200 flex items-center gap-1.5">
                    {m.title}
                  </div>
                  <p className="text-[11px] text-slate-400 leading-snug">{m.desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Target Frontier Model Selector & Execute Action */}
        <div className="flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="target-model-select" className="text-xs font-semibold text-slate-200">
                Target Model (Illustrative)
              </label>
              <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 bg-[#080d19] px-1.5 py-0.5 rounded border border-white/5">
                Model Profile
              </span>
            </div>

            <select
              id="target-model-select"
              value={targetModel}
              onChange={(e) => onTargetModelChange(e.target.value as TargetModel)}
              className="w-full p-2.5 rounded-lg bg-[#080d19] border border-white/10 text-slate-200 text-xs font-medium focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30 transition-all cursor-pointer"
            >
              {ILLUSTRATIVE_MODELS.map((model) => (
                <option key={model.id} value={model.id} className="bg-[#0b1324] text-slate-200">
                  {model.name} ({model.provider}) &bull; {model.contextWindow}
                </option>
              ))}
            </select>

            <p className="text-[11px] text-slate-500 leading-relaxed">
              Select the intended destination model for token estimation and prompt structure tuning.
            </p>
          </div>

          {/* Action Trigger */}
          <div className="pt-2">
            <Button
              variant="primary"
              size="lg"
              className="w-full font-semibold"
              onClick={onOptimize}
              disabled={disabled || isLoading}
              isLoading={isLoading}
            >
              <Zap className="w-4 h-4 text-navy-950" />
              <span>{isLoading ? 'Optimizing with Local LLM...' : 'Optimize Prompt'}</span>
            </Button>

            {disabled && !isLoading && (
              <p className="text-[11px] text-slate-500 text-center mt-2 flex items-center justify-center gap-1">
                <ShieldAlert className="w-3 h-3 text-slate-500" />
                Enter prompt text above to enable optimization
              </p>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};
