import React from 'react';
import { Card } from './Card';
import { Button } from './Button';
import { ILLUSTRATIVE_MODELS } from '@/services/mockOptimizer';
import { OptimizationMode, TargetModel, OllamaStatus, ModelCatalogItem } from '@/types';
import { Zap, Sliders, ShieldAlert, AlertCircle, CheckCircle, Route, Bot } from 'lucide-react';

export interface OptimizationControlsProps {
  mode: OptimizationMode;
  onModeChange: (mode: OptimizationMode) => void;
  targetModel: TargetModel;
  onTargetModelChange: (model: TargetModel) => void;
  onOptimize: () => void;
  isLoading: boolean;
  disabled: boolean;
  ollamaStatus?: OllamaStatus | null;
  enableRouting?: boolean;
  onEnableRoutingChange?: (enabled: boolean) => void;
  requestedModel?: string;
  onRequestedModelChange?: (model: string) => void;
  discoveredModels?: ModelCatalogItem[];
  downstreamModel?: string;
  onDownstreamModelChange?: (model: string) => void;
  onCompare?: () => void;
  isComparing?: boolean;
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
  enableRouting = true,
  onEnableRoutingChange,
  requestedModel = '',
  onRequestedModelChange,
  discoveredModels = [],
  downstreamModel = '',
  onDownstreamModelChange,
  onCompare,
  isComparing = false,
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

      {/* Phase 5: Local Optimizer Model Routing Bar */}
      <div className="p-3.5 rounded-lg bg-[#080d19] border border-cyan-500/20 space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
            <Route className="w-3.5 h-3.5 text-cyan-400" />
            Optimizer Model Routing (Phase 5)
          </label>
          {onEnableRoutingChange && (
            <label className="inline-flex items-center gap-2 cursor-pointer text-xs font-mono text-cyan-300">
              <input
                type="checkbox"
                checked={enableRouting}
                onChange={(e) => onEnableRoutingChange(e.target.checked)}
                className="rounded border-white/20 bg-[#0c1424] text-cyan-500 focus:ring-0 focus:ring-offset-0"
              />
              <span>Auto-Route</span>
            </label>
          )}
        </div>

        {enableRouting ? (
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Automatic router dynamically selects between fast compact (<code className="text-cyan-300">qwen2.5:1.5b</code>) and strong reasoning (<code className="text-cyan-300">qwen2.5:7b</code>) local models based on rule complexity and live GPU VRAM.
          </p>
        ) : (
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Manual Local Model Override:</span>
              <span>Installed: {discoveredModels.length || ollamaStatus?.available_models.length || 0}</span>
            </div>
            <select
              value={requestedModel || (ollamaStatus?.model || 'qwen2.5:1.5b')}
              onChange={(e) => onRequestedModelChange && onRequestedModelChange(e.target.value)}
              className="w-full p-2 rounded-lg bg-[#0c1424] border border-white/10 text-slate-200 text-xs font-mono focus:outline-none focus:border-cyan-500/50"
            >
              {discoveredModels.length > 0 ? (
                discoveredModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name} ({m.parameter_size || m.tier}) {m.is_loaded ? '• [resident in VRAM]' : ''}
                  </option>
                ))
              ) : (
                (ollamaStatus?.available_models || ['qwen2.5:1.5b']).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))
              )}
            </select>
          </div>
        )}
      </div>

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

        {/* Target Model, Downstream Model & Actions */}
        <div className="flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            {/* Target Model (Illustrative) */}
            <div className="space-y-1">
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
                className="w-full p-2 rounded-lg bg-[#080d19] border border-white/10 text-slate-200 text-xs font-medium focus:outline-none focus:border-cyan-500/50"
              >
                {ILLUSTRATIVE_MODELS.map((model) => (
                  <option key={model.id} value={model.id} className="bg-[#0b1324] text-slate-200">
                    {model.name} ({model.provider}) &bull; {model.contextWindow}
                  </option>
                ))}
              </select>
            </div>

            {/* Downstream Model (Pinned for Output Comparison) */}
            <div className="space-y-1 pt-1 border-t border-white/5">
              <div className="flex items-center justify-between">
                <label htmlFor="downstream-model-select" className="text-xs font-semibold text-cyan-300 flex items-center gap-1">
                  <Bot className="w-3.5 h-3.5 text-cyan-400" />
                  Downstream LLM (Pinned)
                </label>
                <span className="text-[10px] font-mono uppercase tracking-wider text-cyan-400 bg-cyan-950/50 px-1.5 py-0.5 rounded border border-cyan-500/30">
                  Dual Execution
                </span>
              </div>

              <select
                id="downstream-model-select"
                value={downstreamModel || (ollamaStatus?.model || 'qwen2.5:1.5b')}
                onChange={(e) => onDownstreamModelChange && onDownstreamModelChange(e.target.value)}
                className="w-full p-2 rounded-lg bg-[#080d19] border border-cyan-500/30 text-cyan-100 text-xs font-mono focus:outline-none focus:border-cyan-400"
              >
                {discoveredModels.length > 0 ? (
                  discoveredModels.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name} ({m.parameter_size || m.tier}) {m.is_loaded ? '• [VRAM loaded]' : ''}
                    </option>
                  ))
                ) : (
                  (ollamaStatus?.available_models || ['qwen2.5:1.5b']).map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))
                )}
              </select>
              <p className="text-[10px] text-slate-400 leading-tight">
                Both original & optimized prompts will be evaluated against this exact model.
              </p>
            </div>
          </div>

          {/* Action Triggers: Compare Outputs and Optimize Prompt */}
          <div className="pt-2 space-y-2">
            {onCompare && (
              <Button
                variant="primary"
                size="lg"
                className="w-full font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-navy-950"
                onClick={onCompare}
                disabled={disabled || isLoading || isComparing}
                isLoading={isComparing}
              >
                <Bot className="w-4 h-4 text-navy-950" />
                <span>{isComparing ? 'Running Output Comparison...' : 'Compare Downstream Outputs'}</span>
              </Button>
            )}

            <Button
              variant="outline"
              size="md"
              className="w-full font-medium text-xs text-slate-300 hover:text-white border-white/10"
              onClick={onOptimize}
              disabled={disabled || isLoading || isComparing}
              isLoading={isLoading}
            >
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
              <span>{isLoading ? 'Optimizing Prompt...' : 'Optimize Prompt Only'}</span>
            </Button>

            {disabled && !isLoading && !isComparing && (
              <p className="text-[11px] text-slate-500 text-center mt-2 flex items-center justify-center gap-1">
                <ShieldAlert className="w-3 h-3 text-slate-500" />
                Enter prompt text above to enable execution
              </p>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};
