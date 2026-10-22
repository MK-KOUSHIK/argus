import React from 'react';
import { Card } from '@/components/Card';
import { Button } from '@/components/Button';
import { StatusBadge } from '@/components/StatusBadge';
import { useHealth } from '@/hooks/useHealth';
import {
  Layers,
  Cpu,
  ShieldCheck,
  Zap,
  CheckCircle2,
  RefreshCw,
  FolderTree,
  Server,
} from 'lucide-react';

export const HomePage: React.FC = () => {
  const { health, isLoading, error, refetch } = useHealth(false);

  const apiStatus = health?.status === 'ok' ? 'online' : error ? 'offline' : 'idle';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Hero Section */}
      <section className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-500/30">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <span>Prompt Optimization Middleware</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white">
          Cost-Efficient LLM Inference Architecture
        </h1>
        <p className="text-base text-slate-300 max-w-3xl leading-relaxed">
          Argus employs a lightweight local model (Qwen2.5-1.5B-Instruct via Ollama) to compress,
          structure, and optimize prompts prior to main-LLM dispatch—reducing input token costs while
          strictly preserving intent and constraints.
        </p>
      </section>

      {/* Grid: Phase Status & Health Verification */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Foundation Status */}
        <Card
          title="Phase 1 Foundation"
          subtitle="Repository architecture & design system"
          badge={<StatusBadge status="online" label="Ready" />}
          className="lg:col-span-2"
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
            <div className="flex items-start gap-3 p-3 rounded-lg bg-[#09101d] border border-white/5">
              <FolderTree className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
              <div>
                <h4 className="text-sm font-semibold text-slate-200">Strict Directory Structure</h4>
                <p className="text-xs text-slate-400 mt-1">
                  Separated frontend, backend, docs, scripts, and workflows.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-[#09101d] border border-white/5">
              <Layers className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
              <div>
                <h4 className="text-sm font-semibold text-slate-200">Design Tokens & Themes</h4>
                <p className="text-xs text-slate-400 mt-1">
                  Dark navy surfaces, restrained cyan accents, accessible typography.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-[#09101d] border border-white/5">
              <Server className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
              <div>
                <h4 className="text-sm font-semibold text-slate-200">FastAPI & Pydantic</h4>
                <p className="text-xs text-slate-400 mt-1">
                  Centralized settings, modular routes, and validation schemas.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3 rounded-lg bg-[#09101d] border border-white/5">
              <ShieldCheck className="w-5 h-5 text-cyan-400 mt-0.5 shrink-0" />
              <div>
                <h4 className="text-sm font-semibold text-slate-200">Architectural Boundary</h4>
                <p className="text-xs text-slate-400 mt-1">
                  Frontend components contain no backend business logic.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Health Check Diagnostics */}
        <Card
          title="Backend Diagnostics"
          subtitle="Direct health endpoint verification"
          badge={
            <StatusBadge
              status={apiStatus}
              label={apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Standby'}
            />
          }
        >
          <div className="space-y-4">
            <p className="text-xs text-slate-400">
              Verifies communication with <code className="text-cyan-300">/api/health</code>.
            </p>

            <div className="p-3 rounded-lg bg-[#080d19] border border-white/5 font-mono text-xs text-slate-300 min-h-[90px] flex flex-col justify-center">
              {isLoading && (
                <div className="flex items-center gap-2 text-cyan-400">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Checking backend health...</span>
                </div>
              )}
              {!isLoading && health && (
                <div className="space-y-1">
                  <div className="text-cyan-400 flex items-center gap-1.5 font-medium">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Status: {health.status}</span>
                  </div>
                  <div className="text-slate-400 text-[11px]">App: {health.app_name} v{health.version}</div>
                  <div className="text-slate-500 text-[10px]">Env: {health.environment} | {health.timestamp}</div>
                </div>
              )}
              {!isLoading && error && (
                <div className="text-red-400 space-y-1">
                  <div className="font-semibold">Connection check:</div>
                  <div className="text-[11px] text-red-300/80">{error}</div>
                </div>
              )}
              {!isLoading && !health && !error && (
                <span className="text-slate-500 italic">Click below to test backend connection</span>
              )}
            </div>

            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={() => void refetch()}
              isLoading={isLoading}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Verify Backend Health
            </Button>
          </div>
        </Card>
      </div>

      {/* Architecture Highlights */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Planned Architecture Pipeline
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">1. Intent Preservation</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Extracts task parameters, output formats, and semantic constraints into an immutable
              spec before compression.
            </p>
          </Card>

          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">2. Local Prompt Optimization</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Qwen2.5-1.5B-Instruct runs on Ollama to remove redundant verbosity and optimize token
              density.
            </p>
          </Card>

          <Card className="border-l-2 border-l-cyan-500">
            <h3 className="text-sm font-semibold text-white">3. Token Telemetry & Dispatch</h3>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Dispatches to target frontier LLMs (e.g. OpenAI, Anthropic, Gemini) with input token
              savings metrics tracked in real-time.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
};
