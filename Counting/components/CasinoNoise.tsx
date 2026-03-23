'use client';

import { useMemo } from 'react';

interface ChipProps {
  index: number;
}

function FloatingChip({ index }: ChipProps) {
  const style = useMemo(() => {
    const colors = ['#e53e3e', '#3182ce', '#38a169', '#805ad5', '#d69e2e', '#dd6b20'];
    const color = colors[index % colors.length];
    const size = 20 + Math.random() * 20;
    const left = Math.random() * 100;
    const duration = 6 + Math.random() * 8;
    const delay = Math.random() * -duration;
    return {
      width: `${size}px`,
      height: `${size}px`,
      left: `${left}%`,
      backgroundColor: color,
      borderRadius: '50%',
      border: `2px solid rgba(255,255,255,0.4)`,
      animationDuration: `${duration}s`,
      animationDelay: `${delay}s`,
    };
  }, [index]);

  return (
    <div
      className="casino-chip-float absolute opacity-40"
      style={style}
    />
  );
}

export default function CasinoNoise() {
  const chips = useMemo(() => Array.from({ length: 6 }, (_, i) => i), []);

  const tickerAmounts = useMemo(() => {
    const amounts = ['$25', '$100', '$50', '$500', '$75', '$200', '$150', '$1000', '$25', '$50', '$250', '$100'];
    return amounts.join('   •   ');
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      {/* Floating chips */}
      {chips.map((i) => (
        <FloatingChip key={i} index={i} />
      ))}

      {/* Background pulse */}
      <div className="casino-bg-pulse absolute inset-0" />

      {/* Ticker strip */}
      <div className="absolute bottom-0 left-0 right-0 h-6 bg-black/30 overflow-hidden flex items-center">
        <div className="casino-ticker whitespace-nowrap text-xs text-yellow-400/50 font-mono">
          {tickerAmounts}{'   •   '}{tickerAmounts}
        </div>
      </div>
    </div>
  );
}
