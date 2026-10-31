import React from 'react';
import { StatusBadge } from './StatusBadge';
import { OllamaStatus } from '@/types';
import { Shield, Sparkles, Cpu, Activity } from 'lucide-react';

export interface HeaderProps {
  apiStatus?: 'online' | 'offline' | 'idle';
  ollamaStatus?: OllamaStatus | null;
  activeTab?: 'workspace' | 'system';
  onTabChange?: (tab: 'workspace' | 'system') => void;
}

export const Header: React.FC<HeaderProps> = ({
  apiStatus = 'idle',
  ollamaStatus,
  activeTab = 'workspace',
  onTabChange,
}) => {
  const isOllamaReady = Boolean(ollamaStatus?.connected && ollamaStatus?.model_installed);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.08] bg-[#070c17]/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="h-16 flex items-center justify-between gap-4">
          {/* Brand & Identity */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-950/80 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-slate-100 tracking-tight text-lg">Argus</span>
                <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-500/20">
                  Optimizer
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block tracking-wide">
                Understand. Optimize. Execute.
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          {onTabChange && (
            <nav className="flex items-center gap-1 bg-[#0b1324] p-1 rounded-lg border border-white/5 text-xs font-medium">
              <button
                type="button"
                onClick={() => onTabChange('workspace')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'workspace'
                    ? 'bg-cyan-500 text-navy-950 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>Workspace</span>
              </button>
              <button
                type="button"
                onClick={() => onTabChange('system')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'system'
                    ? 'bg-cyan-500 text-navy-950 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>System Status</span>
              </button>
            </nav>
          )}

          {/* Connection Indicators */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Real Local Optimizer Status */}
            <div
              className={`hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border ${
                isOllamaReady
                  ? 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300'
                  : 'bg-[#0c1424] border-white/5 text-slate-400'
              }`}
              title={
                isOllamaReady
                  ? `Connected to Ollama: ${ollamaStatus?.model}`
                  : ollamaStatus?.error || 'Local Ollama model offline'
              }
            >
              <Cpu className={`w-3.5 h-3.5 ${isOllamaReady ? 'text-cyan-400' : 'text-slate-500'}`} />
              <span>Local LLM:</span>
              <span className={`font-sans text-[11px] ${isOllamaReady ? 'text-cyan-200 font-medium' : 'text-amber-400/90'}`}>
                {isOllamaReady ? `Connected (${ollamaStatus?.model || 'qwen2.5:1.5b'})` : 'Offline / Standby'}
              </span>
            </div>

            {/* FastAPI Backend Status */}
            <StatusBadge
              status={apiStatus}
              label={
                apiStatus === 'online'
                  ? 'API Online'
                  : apiStatus === 'offline'
                  ? 'API Offline'
                  : 'API Standby'
              }
            />
          </div>
        </div>

        {/* Mobile Sub-bar for Connection Status */}
        <div className="lg:hidden flex items-center justify-between py-2 border-t border-white/[0.04] text-[10px] text-slate-400 font-mono">
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${isOllamaReady ? 'bg-cyan-400' : 'bg-amber-400'}`} />
            <span>Local LLM: {isOllamaReady ? 'Online (qwen2.5:1.5b)' : 'Offline'}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${apiStatus === 'online' ? 'bg-cyan-400' : 'bg-amber-400'}`} />
            <span>Backend: {apiStatus}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
