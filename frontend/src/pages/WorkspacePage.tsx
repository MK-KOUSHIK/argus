import React, { useState } from 'react';
import { PromptEditor } from '@/components/PromptEditor';
import { OptimizationControls } from '@/components/OptimizationControls';
import { ComparisonView } from '@/components/ComparisonView';
import { OutputComparisonView } from '@/components/OutputComparisonView';
import { MetricsPanel } from '@/components/MetricsPanel';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { apiService } from '@/services/api';
import { EXAMPLE_PROMPTS } from '@/services/mockOptimizer';
import {
  OptimizationMode,
  TargetModel,
  OptimizationResult,
  OptimizationStatus,
  ExamplePrompt,
  OllamaStatus,
  ModelCatalogItem,
  ComparisonResult,
} from '@/types';
import {
  Sparkles,
  Zap,
  AlertTriangle,
  Clock,
  Cpu,
  Bot,
  Layers,
} from 'lucide-react';

export interface WorkspacePageProps {
  ollamaStatus?: OllamaStatus | null;
}

export const WorkspacePage: React.FC<WorkspacePageProps> = ({ ollamaStatus }) => {
  const [rawPrompt, setRawPrompt] = useState<string>('');
  const [mode, setMode] = useState<OptimizationMode>('balanced');
  const [targetModel, setTargetModel] = useState<TargetModel>('gemini-1-5-flash');
  const [status, setStatus] = useState<OptimizationStatus>('idle');
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [resultTab, setResultTab] = useState<'comparison' | 'prompt'>('comparison');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [enableRouting, setEnableRouting] = useState<boolean>(true);
  const [requestedModel, setRequestedModel] = useState<string>('');
  const [downstreamModel, setDownstreamModel] = useState<string>('');
  const [discoveredModels, setDiscoveredModels] = useState<ModelCatalogItem[]>([]);

  React.useEffect(() => {
    let isMounted = true;
    apiService
      .getModels()
      .then((data) => {
        if (isMounted && data.models) {
          setDiscoveredModels(data.models);
          if (!requestedModel && data.models.length > 0) {
            setRequestedModel(data.models[0].name);
          }
          if (!downstreamModel && data.models.length > 0) {
            setDownstreamModel(data.models[0].name);
          }
        }
      })
      .catch(() => {
        // Fallback to ollamaStatus if models endpoint not reached
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const handleSelectExample = (example: ExamplePrompt) => {
    setRawPrompt(example.content);
    setErrorMessage(null);
  };

  const handleClear = () => {
    setRawPrompt('');
    setResult(null);
    setComparisonResult(null);
    setStatus('idle');
    setErrorMessage(null);
  };

  const handleOptimize = async () => {
    const trimmed = rawPrompt.trim();
    if (!trimmed) {
      setErrorMessage('Please enter or select a prompt before optimizing.');
      setStatus('error');
      return;
    }

    setErrorMessage(null);
    setStatus('optimizing');

    try {
      // Real backend API call to FastAPI -> Ollama (qwen2.5:1.5b or routed model)
      const optimizationResult = await apiService.optimizePrompt({
        prompt: trimmed,
        mode,
        target_model: targetModel,
        enable_routing: enableRouting,
        requested_model: enableRouting ? undefined : (requestedModel || undefined),
      });

      setResult(optimizationResult);
      setComparisonResult(null);
      setResultTab('prompt');
      setStatus('success');
    } catch (err) {
      setErrorMessage(
        err instanceof Error
          ? err.message
          : 'An unexpected error occurred during prompt optimization'
      );
      setStatus('error');
    }
  };

  const handleCompare = async () => {
    const trimmed = rawPrompt.trim();
    if (!trimmed) {
      setErrorMessage('Please enter or select a prompt before running output comparison.');
      setStatus('error');
      return;
    }

    setErrorMessage(null);
    setStatus('comparing');

    try {
      const targetDownstream = downstreamModel || requestedModel || ollamaStatus?.model || 'qwen2.5:1.5b';
      const cmpRes = await apiService.compareOutputs({
        prompt: trimmed,
        downstream_model: targetDownstream,
        mode,
        target_model: targetModel,
        enable_optimizer_routing: enableRouting,
        requested_optimizer_model: enableRouting ? undefined : (requestedModel || undefined),
      });

      setComparisonResult(cmpRes);
      setResult(cmpRes.optimizerDetails);
      setResultTab('comparison');
      setStatus('success');
    } catch (err) {
      setErrorMessage(
        err instanceof Error
          ? err.message
          : 'An unexpected error occurred during output comparison'
      );
      setStatus('error');
    }
  };

  const handleOptimizedPromptChange = (updated: string) => {
    if (result) {
      setResult({
        ...result,
        optimizedPrompt: updated,
      });
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Workspace Header / Banner */}
      <section className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
              <span>Argus Prompt Optimization Workspace</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white mt-2">
              Optimize Input Tokens & Preserve Constraints
            </h1>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-400 bg-[#0c1424] px-3 py-1.5 rounded-lg border border-white/5 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>Local Engine: {ollamaStatus?.model || 'qwen2.5:1.5b'}</span>
            </span>
          </div>
        </div>
        <p className="text-xs sm:text-sm text-slate-400 max-w-3xl leading-relaxed">
          Prepare, compress, and refine prompts using local neural inference before downstream execution.
          Select an optimization profile, evaluate token reduction, and inspect side-by-side prompt diffs.
        </p>
      </section>

      {/* Error State Banner */}
      {status === 'error' && errorMessage && (
        <div
          role="alert"
          className="flex items-start justify-between gap-3 p-4 rounded-xl bg-red-950/40 border border-red-500/30 text-red-200"
        >
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
            <div>
              <h4 className="text-sm font-semibold text-red-300">Optimization Notice</h4>
              <p className="text-xs text-red-200/80 mt-0.5 leading-relaxed">{errorMessage}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setErrorMessage(null)}
            className="text-xs text-red-300 hover:text-white shrink-0"
          >
            Dismiss
          </Button>
        </div>
      )}

      {/* Primary Workspace Grid: Input & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Raw Prompt Editor */}
        <div className="lg:col-span-7 h-full">
          <PromptEditor
            value={rawPrompt}
            onChange={(val) => {
              setRawPrompt(val);
              if (errorMessage) setErrorMessage(null);
            }}
            onClear={handleClear}
            onSelectExample={handleSelectExample}
            disabled={status === 'optimizing'}
          />
        </div>

        {/* Right: Optimization Controls */}
        <div className="lg:col-span-5 h-full">
          <OptimizationControls
            mode={mode}
            onModeChange={setMode}
            targetModel={targetModel}
            onTargetModelChange={setTargetModel}
            onOptimize={handleOptimize}
            isLoading={status === 'optimizing'}
            disabled={rawPrompt.trim().length === 0}
            ollamaStatus={ollamaStatus}
            enableRouting={enableRouting}
            onEnableRoutingChange={setEnableRouting}
            requestedModel={requestedModel}
            onRequestedModelChange={setRequestedModel}
            discoveredModels={discoveredModels}
            downstreamModel={downstreamModel}
            onDownstreamModelChange={setDownstreamModel}
            onCompare={handleCompare}
            isComparing={status === 'comparing'}
          />
        </div>
      </div>

      {/* Loading State: Optimizing */}
      {status === 'optimizing' && (
        <Card className="p-8 text-center border-cyan-500/30 bg-[#091222]">
          <div className="max-w-md mx-auto space-y-4">
            <div className="w-12 h-12 mx-auto rounded-xl bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 animate-pulse">
              <Zap className="w-6 h-6 animate-bounce" />
            </div>
            <div className="space-y-1">
              <h3 className="text-base font-semibold text-slate-100">
                Optimizing Prompt with {mode.toUpperCase()} Profile...
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Local model <code className="text-cyan-300 font-mono">{ollamaStatus?.model || 'qwen2.5:1.5b'}</code> is
                extracting constraints and pruning tokens.
              </p>
            </div>
            <div className="flex items-center justify-center gap-4 text-[11px] font-mono text-cyan-300">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3 animate-spin" /> Ollama local inference in progress
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Loading State: Comparing */}
      {status === 'comparing' && (
        <Card className="p-8 text-center border-cyan-500/30 bg-[#091222]">
          <div className="max-w-md mx-auto space-y-4">
            <div className="w-12 h-12 mx-auto rounded-xl bg-cyan-950/80 border border-cyan-500/40 flex items-center justify-center text-cyan-400 animate-pulse">
              <Bot className="w-6 h-6 animate-bounce" />
            </div>
            <div className="space-y-1">
              <h3 className="text-base font-semibold text-slate-100">
                Comparing Original vs Optimized Downstream Outputs...
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Executing Execution A (Original Prompt) and Execution B (Optimized Prompt) against local model{' '}
                <code className="text-cyan-300 font-mono">{downstreamModel || requestedModel || 'qwen2.5:1.5b'}</code>.
              </p>
            </div>
            <div className="flex items-center justify-center gap-4 text-[11px] font-mono text-cyan-300">
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3 animate-spin" /> Sequential dual inference in progress
              </span>
            </div>
          </div>
        </Card>
      )}

      {/* Success State: Results & Metrics */}
      {status === 'success' && result && (
        <section className="space-y-6 pt-2">
          {/* View Tab Switcher (if output comparison was run) */}
          {comparisonResult && (
            <div className="flex items-center justify-center sm:justify-start gap-2 p-1 rounded-xl bg-[#080d19] border border-white/5 w-fit">
              <button
                type="button"
                onClick={() => setResultTab('comparison')}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  resultTab === 'comparison'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Bot className="w-3.5 h-3.5" />
                <span>Downstream Output Comparison</span>
              </button>
              <button
                type="button"
                onClick={() => setResultTab('prompt')}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  resultTab === 'prompt'
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Prompt Diff & Metrics</span>
              </button>
            </div>
          )}

          {/* Render selected view */}
          {comparisonResult && resultTab === 'comparison' ? (
            <OutputComparisonView comparison={comparisonResult} />
          ) : (
            <div className="space-y-6">
              <MetricsPanel result={result} />
              <ComparisonView
                originalPrompt={result.originalPrompt}
                optimizedPrompt={result.optimizedPrompt}
                onOptimizedPromptChange={handleOptimizedPromptChange}
                originalMetrics={result.originalMetrics}
                optimizedMetrics={result.optimizedMetrics}
                tokenReductionPercent={result.tokenReductionPercent}
                tokenSavings={result.tokenSavings}
                tokenChangePercent={result.tokenChangePercent}
                strategy={result.strategy}
                isPreserved={result.isPreserved}
                validationStatus={result.validationStatus}
                model={result.model}
              />
            </div>
          )}
        </section>
      )}

      {/* Empty State / Initial Guidance (When idle and no result yet) */}
      {status === 'idle' && !result && (
        <Card className="p-8 border-dashed border-white/10 bg-[#070d18] text-center">
          <div className="max-w-md mx-auto space-y-4">
            <div className="w-12 h-12 mx-auto rounded-full bg-[#0e172a] border border-white/10 flex items-center justify-center text-slate-400">
              <Zap className="w-5 h-5 text-cyan-400" />
            </div>
            <div className="space-y-1.5">
              <h3 className="text-sm font-semibold text-slate-200">
                Prompt Optimization Workspace Ready
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Paste a prompt or select a sample above to optimize tokens with local neural inference.
              </p>
            </div>
            <div className="pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleSelectExample(EXAMPLE_PROMPTS[0])}
                className="text-xs"
              >
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span>Try Example: Code Review</span>
              </Button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
