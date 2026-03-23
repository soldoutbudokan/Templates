'use client';

import { getBettingAdvice } from '@/lib/betting';

interface BettingAdviceProps {
  runningCount: number;
  decksRemaining: number;
}

export default function BettingAdvice({ runningCount, decksRemaining }: BettingAdviceProps) {
  const trueCount = decksRemaining > 0
    ? Math.round(runningCount / decksRemaining)
    : 0;
  const advice = getBettingAdvice(trueCount);

  const barWidth = Math.min(advice.multiplier * 10, 100);

  return (
    <div className="bg-white/10 rounded-lg p-3 text-left max-w-xs mx-auto">
      <div className="text-xs text-white/50 mb-1">Betting Strategy</div>
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
      <div className="text-xs text-white/40 mt-1">
        RC: {runningCount >= 0 ? '+' : ''}{runningCount} / {decksRemaining.toFixed(1)} decks = TC {trueCount >= 0 ? '+' : ''}{trueCount}
      </div>
    </div>
  );
}
