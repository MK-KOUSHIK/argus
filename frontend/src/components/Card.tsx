import React, { ReactNode } from 'react';

export interface CardProps {
  children: ReactNode;
  title?: string;
  subtitle?: string;
  badge?: ReactNode;
  className?: string;
  hoverEffect?: boolean;
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  subtitle,
  badge,
  className = '',
  hoverEffect = false,
}) => {
  return (
    <div
      className={`rounded-xl border border-white/[0.08] bg-[#0c1424] p-5 shadow-lg shadow-black/20 ${
        hoverEffect ? 'transition-all hover:border-cyan-500/30 hover:bg-[#0f192d]' : ''
      } ${className}`}
    >
      {(title || badge) && (
        <div className="flex items-center justify-between pb-3 mb-4 border-b border-white/[0.06]">
          <div>
            {title && <h3 className="text-base font-semibold text-slate-100">{title}</h3>}
            {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          </div>
          {badge && <div>{badge}</div>}
        </div>
      )}
      {children}
    </div>
  );
};
