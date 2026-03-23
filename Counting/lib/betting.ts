import { BettingAdvice } from './types';

export function getBettingAdvice(trueCount: number): BettingAdvice {
  if (trueCount <= 1) return { trueCount, recommendation: 'Minimum bet', multiplier: 1 };
  if (trueCount === 2) return { trueCount, recommendation: '2x minimum', multiplier: 2 };
  if (trueCount === 3) return { trueCount, recommendation: '4x minimum', multiplier: 4 };
  if (trueCount === 4) return { trueCount, recommendation: '8x spread', multiplier: 8 };
  return { trueCount, recommendation: 'Maximum spread', multiplier: 10 };
}
