/** Hi-Lo training convention: half-deck estimates and flooring toward -Infinity. */
export const COUNT_POLICY = 'Divide by remaining decks, then round down (−1.5 becomes −2).';

export function parseCount(value: string): number | null {
  if (!/^[+-]?\d+$/.test(value.trim())) return null;
  const count = Number(value);
  return Number.isSafeInteger(count) && Math.abs(count) <= 1000 ? count : null;
}

export function toTrueCount(runningCount: number, decksRemaining: number): number {
  if (!Number.isFinite(runningCount) || !Number.isFinite(decksRemaining) || decksRemaining <= 0) {
    throw new RangeError('A true count needs a finite count and a positive deck estimate.');
  }
  return Math.floor(runningCount / decksRemaining) || 0;
}

export function estimateDecks(cardsRemaining: number): number {
  if (cardsRemaining <= 0) throw new RangeError('There are no cards remaining.');
  return Math.max(0.5, Math.round(cardsRemaining / 26) / 2);
}

export function signed(count: number): string {
  return count > 0 ? `+${count}` : `${count}`;
}
