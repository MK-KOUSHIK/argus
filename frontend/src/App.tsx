import React, { useState } from 'react';
import { Header } from '@/components/Header';
import { WorkspacePage } from '@/pages/WorkspacePage';
import { BenchmarkPage } from '@/pages/BenchmarkPage';
import { HomePage } from '@/pages/HomePage';
import { useHealth } from '@/hooks/useHealth';
import { useOllamaStatus } from '@/hooks/useOllamaStatus';
import { useMainLLMStatus } from '@/hooks/useMainLLMStatus';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'workspace' | 'benchmark' | 'system'>('workspace');
  const { health, error: healthError } = useHealth(true);
  const { status: ollamaStatus } = useOllamaStatus(true);
  const { status: mainLLMStatus } = useMainLLMStatus(true);

  const apiStatus = health?.status === 'ok' ? 'online' : healthError ? 'offline' : 'idle';
  const isOllamaReady = Boolean(ollamaStatus?.connected && ollamaStatus?.model_installed);
  const isMainLLMReady = Boolean(mainLLMStatus?.is_configured);

  return (
    <div className="min-h-screen bg-[#060a12] text-slate-200 flex flex-col font-sans selection:bg-cyan-500/20 selection:text-cyan-300">
      <Header
        apiStatus={apiStatus}
        ollamaStatus={ollamaStatus}
        mainLLMStatus={mainLLMStatus}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />
      <main className="flex-1">
        {activeTab === 'workspace' ? (
          <WorkspacePage ollamaStatus={ollamaStatus} />
        ) : activeTab === 'benchmark' ? (
          <BenchmarkPage mainLLMStatus={mainLLMStatus} />
        ) : (
          <HomePage />
        )}
      </main>
      <footer className="border-t border-white/[0.06] bg-[#070c17] py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-300">Argus</span>
            <span className="text-slate-600">&bull;</span>
            <span className="text-slate-400">Understand. Optimize. Execute.</span>
          </div>
          <div className="flex items-center gap-4 font-mono text-[11px] text-slate-400">
            <span>Phase 4: Main-LLM Benchmark</span>
            <span className="text-slate-600">&bull;</span>
            <span className={isOllamaReady ? 'text-cyan-400' : 'text-slate-400'}>
              Local LLM: {isOllamaReady ? `Active (${ollamaStatus?.model})` : 'Offline'}
            </span>
            <span className="text-slate-600">&bull;</span>
            <span className={isMainLLMReady ? 'text-cyan-400' : 'text-slate-500'}>
              Main LLM: {isMainLLMReady ? `Ready (${mainLLMStatus?.model})` : 'Unconfigured'}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;

