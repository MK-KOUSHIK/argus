import React from 'react';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { StatusBadge } from '@/components/StatusBadge';
import { useHealth } from '@/hooks/useHealth';
import { useOllamaStatus } from '@/hooks/useOllamaStatus';
import {
  Cpu,
  Zap,
  CheckCircle2,
  RefreshCw,
} from 'lucide-react';

export const HomePage: React.FC = () => {
  const { health, isLoading: isHealthLoading, error: healthError, refetch: refetchHealth } = useHealth(false);
  const { status: ollamaStatus, isLoading: isOllamaLoading, checkStatus: refetchOllama } = useOllamaStatus(true);

  const apiStatus = health?.status === 'ok' ? 'online' : healthError ? 'offline' : 'idle';
  const isOllamaReady = Boolean(ollamaStatus?.connected && ollamaStatus?.model_installed);

  const handleRefreshAll = () => {
    void refetchHealth();
    void refetchOllama();
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Hero Section */}
      <section className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30">
            <Zap className="w-3.5 h-3.5 text-cyan-400" />
            <span>Argus System & Integration Status</span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={handleRefreshAll}
            className="text-xs"
            disabled={isHealthLoading || isOllamaLoading}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${(isHealthLoading || isOllamaLoading) ? 'animate-spin' : ''}`} />
            <span>Refresh Telemetry</span>
          </Button>
        </div>

        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white">
          System Status & Provider Gateway
        </h1>
        <p className="text-sm sm:text-base text-slate-300 max-w-3xl leading-relaxed">
          Monitor Argus local neural optimization middleware and backend health diagnostics.
        </p>
      </section>

      {/* Primary Status Grid: 2 Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 1. Local Ollama Status */}
        <Card
          title="Local Optimizer Engine"
          subtitle="Ollama neural runtime"
          badge={
            <StatusBadge
              status={isOllamaReady ? 'online' : 'offline'}
              label={isOllamaReady ? 'Connected' : 'Offline'}
            />
          }
          className="flex flex-col justify-between"
        >
          <div className="space-y-3 mt-1">
            <div className="p-3 rounded-lg bg-[#080d19] border border-white/5 font-mono text-xs space-y-1.5">
              <div className="flex items-center justify-between text-slate-400 text-[11px]">
                <span>Engine Model</span>
                <span className="text-cyan-300 font-semibold">{ollamaStatus?.model || 'qwen2.5:1.5b'}</span>
              </div>
              <div className="flex items-center justify-between text-slate-400 text-[11px]">
                <span>Endpoint</span>
                <span className="text-slate-300 truncate max-w-[170px]">{ollamaStatus?.base_url || 'http://127.0.0.1:11434'}</span>
              </div>
              <div className="flex items-center justify-between text-slate-400 text-[11px]">
                <span>Model Installed</span>
                <span className={ollamaStatus?.model_installed ? 'text-emerald-400 font-medium' : 'text-amber-400'}>
                  {ollamaStatus?.model_installed ? 'Yes' : 'Not Detected'}
                </span>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              {isOllamaReady
                ? 'Local model running with zero cloud token usage and zero latency overhead to external providers.'
                : 'Ollama service offline. Ensure Ollama is running (`ollama serve`) and `qwen2.5:1.5b` is downloaded.'}
            </p>
          </div>

          <div className="pt-4">
            <Button
              variant="outline"
              size="sm"
              className="w-full text-xs"
              onClick={() => void refetchOllama()}
              isLoading={isOllamaLoading}
            >
              <RefreshCw className="w-3 h-3" />
              <span>Verify Ollama</span>
            </Button>
          </div>
        </Card>

        {/* 2. FastAPI Backend Health */}
        <Card
          title="FastAPI Backend"
          subtitle="Core application service"
          badge={
            <StatusBadge
              status={apiStatus}
              label={apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Standby'}
            />
          }
          className="flex flex-col justify-between"
        >
          <div className="space-y-3 mt-1">
            <div className="p-3 rounded-lg bg-[#080d19] border border-white/5 font-mono text-xs space-y-1.5 min-h-[75px] flex flex-col justify-center">
              {isHealthLoading && (
                <div className="flex items-center gap-2 text-cyan-400 text-xs">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Checking /api/health...</span>
                </div>
              )}
              {!isHealthLoading && health && (
                <div className="space-y-1">
                  <div className="text-cyan-400 flex items-center gap-1.5 font-medium text-[11px]">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Status: {health.status}</span>
                  </div>
                  <div className="text-slate-400 text-[11px]">App: {health.app_name} v{health.version}</div>
                  <div className="text-slate-500 text-[10px]">Env: {health.environment}</div>
                </div>
              )}
              {!isHealthLoading && healthError && (
                <div className="text-red-400 text-[11px]">
                  <span>Failed: {healthError}</span>
                </div>
              )}
              {!isHealthLoading && !health && !healthError && (
                <span className="text-slate-500 italic text-[11px]">Click below to query /api/health</span>
              )}
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              Provides REST endpoints for prompt optimization and heuristic token analytics.
            </p>
          </div>

          <div className="pt-4">
            <Button
              variant="outline"
              size="sm"
              className="w-full text-xs"
              onClick={() => void refetchHealth()}
              isLoading={isHealthLoading}
            >
              <RefreshCw className="w-3 h-3" />
              <span>Verify Backend Health</span>
            </Button>
          </div>
        </Card>
      </div>

      {/* Architecture Overview */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Argus Architecture & Pipeline
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">1. Intent Preservation</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Analyzes incoming prompts, classifies task categories, and extracts critical constraints
              into an immutable preservation spec before compression.
            </p>
          </Card>

          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">2. Local Prompt Optimization</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Runs Qwen2.5-1.5B-Instruct locally via Ollama to prune conversational verbosity, structure
              instructions, and maximize token efficiency.
            </p>
          </Card>

          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">3. Semantic Validation</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Verifies that explicit prohibitions, required keys, filenames, and deliverable specs
              are strictly preserved before delivering the optimized prompt.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
};
