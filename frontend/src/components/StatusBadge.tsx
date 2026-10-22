import React from 'react';

export interface StatusBadgeProps {
  status: 'online' | 'offline' | 'warning' | 'idle';
  label: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label }) => {
  const configs = {
    online: {
      bg: 'bg-cyan-950/60 border-cyan-500/30 text-cyan-300',
      dot: 'bg-cyan-400',
    },
    offline: {
      bg: 'bg-red-950/40 border-red-500/30 text-red-300',
      dot: 'bg-red-400',
    },
    warning: {
      bg: 'bg-amber-950/40 border-amber-500/30 text-amber-300',
      dot: 'bg-amber-400',
    },
    idle: {
      bg: 'bg-slate-900 border-white/10 text-slate-300',
      dot: 'bg-slate-400',
    },
  };

  const config = configs[status];

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium border ${config.bg}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      {label}
    </span>
  );
};
