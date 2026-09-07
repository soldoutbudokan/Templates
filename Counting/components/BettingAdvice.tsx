'use client';

import { getBettingAdvice } from '@/lib/betting';
import { toTrueCount } from '@/lib/countPolicy';

interface BettingAdviceProps {
  runningCount: number;
  decksRemaining: number;
}

export default function BettingAdvice({ runningCount, decksRemaining }: BettingAdviceProps) {
  const divisor = Math.max(0.5, Math.round(decksRemaining * 2) / 2);
  const trueCount = decksRemaining > 0
    ? toTrueCount(runningCount, divisor)
    : 0;
  const advice = getBettingAdvice(trueCount);

  const barWidth = Math.min(advice.multiplier * 10, 100);

  return (
    <div className="bg-white/10 rounded-lg p-3 text-left max-w-xs mx-auto">
      <div className="text-xs text-white/50 mb-1">Example training schedule</div>
      <div className="flex justify-between text-sm mb-2">
        <span className="text-white/70">True Count: <span className="font-bold text-white">{trueCount >= 0 ? '+' : ''}{trueCount}</span></span>
        <span className="font-medium text-green-400">{advice.recommendation}</span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-300"
          style={{
            width: `${barWidth}%`,
            background: barWidth > 60 ? 'linear-gradient(to right, #38a169, #48bb78)' :
                        barWidth > 30 ? 'linear-gradient(to right, #d69e2e, #ecc94b)' :
                        'linear-gradient(to right, #718096, #a0aec0)',
          }}
        />
      </div>
      <p className="text-sm text-white/60 mt-2">Practice preset, not a bankroll-based recommendation.</p>
      <div className="text-xs text-white/40 mt-1">
        RC: {runningCount >= 0 ? '+' : ''}{runningCount} / {divisor.toFixed(1)} decks = TC {trueCount >= 0 ? '+' : ''}{trueCount}
      </div>
    </div>
  );
}
