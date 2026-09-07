import { Card, Deck, getCardValue } from './deck';
import { estimateDecks, signed, toTrueCount } from './countPolicy';

export type Focus = 'running' | 'conversion';
export type SessionMode = 'practice' | 'assessment';
export interface Checkpoint {
  cards: Card[];
  countBefore: number;
  runningCount: number;
  decksRemaining: number;
  trueCount: number;
}
export interface TrainingPlan {
  id: string;
  seed: number;
  mode: SessionMode;
  focus: Focus;
  speedMs: number;
  checkpoints: Checkpoint[];
}
export interface CheckpointAnswer {
  index: number;
  runningGuess: number | null;
  trueGuess: number | null;
  runningCorrect: boolean;
  trueCorrect: boolean | null;
  arithmeticCorrect: boolean | null;
  lost: boolean;
  responseMs: number;
}
export interface SessionResult extends TrainingPlan {
  endedAt: string;
  completed: boolean;
  interrupted: boolean;
  assisted: boolean;
  answers: CheckpointAnswer[];
}

export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let n = state;
    n = Math.imul(n ^ n >>> 15, n | 1);
    n ^= n + Math.imul(n ^ n >>> 7, n | 61);
    return ((n ^ n >>> 14) >>> 0) / 4294967296;
  };
}

/** Each target is frozen before presentation. Practice samples a shoe; assessment reaches its cut card. */
export function createTrainingPlan(seed: number, mode: SessionMode, focus: Focus, speedMs: number): TrainingPlan {
  if (![1200, 900, 650, 450].includes(speedMs)) throw new RangeError('Unsupported training pace.');
  const random = seededRandom(seed);
  const deck = new Deck(6, 0.75, random);
  const checkpoints: Checkpoint[] = [];
  const limit = mode === 'assessment' ? 234 : 78;
  let exposed = 0;
  while (exposed < limit) {
    const count = Math.min(9 + Math.floor(random() * 12), limit - exposed);
    const countBefore = deck.runningCount();
    const cards = deck.deal(count);
    exposed += count;
    const decksRemaining = estimateDecks(deck.remaining());
    checkpoints.push({ cards, countBefore, runningCount: deck.runningCount(), decksRemaining,
      trueCount: toTrueCount(deck.runningCount(), decksRemaining) });
  }
  return { id: `${seed}-${mode}-${focus}-${speedMs}`, seed, mode, focus, speedMs, checkpoints };
}

export function gradeCheckpoint(plan: TrainingPlan, index: number, runningGuess: number | null,
  trueGuess: number | null, responseMs: number): CheckpointAnswer {
  const checkpoint = plan.checkpoints[index];
  if (!checkpoint) throw new RangeError('Unknown checkpoint.');
  return { index, runningGuess, trueGuess, responseMs: Math.max(0, responseMs), lost: runningGuess === null,
    runningCorrect: runningGuess === checkpoint.runningCount,
    trueCorrect: plan.focus === 'conversion' ? trueGuess === checkpoint.trueCount : null,
    arithmeticCorrect: plan.focus === 'conversion' && runningGuess !== null && trueGuess !== null
      ? trueGuess === toTrueCount(runningGuess, checkpoint.decksRemaining) : null };
}

export function explainAnswer(checkpoint: Checkpoint, answer: CheckpointAnswer): string {
  if (answer.lost) return 'You reported losing the count. Replay this segment from its starting count, then continue from the corrected count.';
  if (!answer.runningCorrect) {
    const difference = (answer.runningGuess ?? 0) - checkpoint.runningCount;
    return `Your count differs by ${signed(difference)}. The error appeared by this checkpoint; replay can help you find it.${answer.arithmeticCorrect ? ' Your conversion follows your entered count correctly, so focus on count retention.' : ''}`;
  }
  if (answer.trueCorrect === false) return `Your running count is right. ${signed(checkpoint.runningCount)} ÷ ${checkpoint.decksRemaining} = ${(checkpoint.runningCount / checkpoint.decksRemaining).toFixed(2)}. Round down to ${signed(checkpoint.trueCount)}, including for negative numbers.`;
  return answer.trueCorrect === null ? 'Accurate count. Keep this total as the next cards arrive.' : 'Your running count and conversion are both correct.';
}

export function replayCounts(checkpoint: Checkpoint): number[] {
  let count = checkpoint.countBefore;
  return checkpoint.cards.map(card => (count += getCardValue(card)));
}

export function assessmentEligible(result: SessionResult): boolean {
  return result.mode === 'assessment' && result.completed && !result.interrupted && !result.assisted;
}

export function sessionLabel(result: SessionResult): string {
  if (result.mode === 'assessment') {
    if (result.interrupted || result.assisted) return 'Interrupted assessment · practice only';
    return result.completed ? 'Uninterrupted assessment' : 'Partial assessment';
  }
  return result.interrupted ? 'Interrupted practice' : 'Coached practice';
}

export function recommendation(results: SessionResult[]): { focus: Focus; title: string; reason: string } {
  const recent = results.filter(result => result.completed && !result.interrupted).slice(-3);
  const answers = recent.flatMap(result => result.answers);
  if (answers.length < 8) return { focus: 'running', title: 'Build a steady running count', reason: 'Start with a short shoe segment. We’ll use your first-attempt checkpoints to choose what comes next.' };
  const runningAccuracy = answers.filter(answer => answer.runningCorrect).length / answers.length;
  if (runningAccuracy < 0.9 || answers.some(answer => answer.lost)) return { focus: 'running', title: 'Keep the count across rounds', reason: `${Math.round(runningAccuracy * 100)}% of your last ${answers.length} checkpoints were exact. Practice retention before increasing the pace.` };
  const conversion = answers.filter(answer => answer.trueCorrect !== null && answer.runningCorrect);
  return { focus: 'conversion', title: 'Add true-count conversion', reason: conversion.length && conversion.some(answer => !answer.trueCorrect)
    ? 'Your counting is holding up. Practice division and rounding while carrying the count forward.'
    : 'Your recent running counts are accurate. Add a conversion task at each checkpoint.' };
}

/** Persist only a bounded, validated history. Invalid data never becomes a training recommendation. */
export function readHistory(raw: string | null): SessionResult[] {
  if (!raw) return [];
  const data: unknown = JSON.parse(raw);
  if (!data || typeof data !== 'object' || !('version' in data) || data.version !== 1 || !('sessions' in data) || !Array.isArray(data.sessions)) throw new Error('Unrecognized progress format.');
  return data.sessions.slice(-24).map((entry: unknown) => {
    if (!entry || typeof entry !== 'object') throw new Error('Invalid saved session.');
    const r = entry as SessionResult;
    if (typeof r.seed !== 'number' || !Number.isInteger(r.seed) || !['practice', 'assessment'].includes(r.mode)
      || !['running', 'conversion'].includes(r.focus) || typeof r.endedAt !== 'string' || !Number.isFinite(Date.parse(r.endedAt))
      || typeof r.completed !== 'boolean' || typeof r.interrupted !== 'boolean' || typeof r.assisted !== 'boolean' || !Array.isArray(r.answers)) throw new Error('Invalid saved session.');
    const plan = createTrainingPlan(r.seed, r.mode, r.focus, r.speedMs);
    if (r.answers.length > plan.checkpoints.length || (r.completed && r.answers.length !== plan.checkpoints.length)) throw new Error('Invalid checkpoint history.');
    const answers = r.answers.map((answer, index) => {
      if (!answer || answer.index !== index || ![answer.runningGuess, answer.trueGuess].every(value => value === null || (Number.isSafeInteger(value) && Math.abs(value) <= 1000))
        || !Number.isFinite(answer.responseMs) || answer.responseMs < 0) throw new Error('Invalid saved answer.');
      return gradeCheckpoint(plan, index, answer.runningGuess, answer.trueGuess, answer.responseMs);
    });
    return { ...plan, endedAt: r.endedAt, completed: r.completed, interrupted: r.interrupted, assisted: r.assisted, answers };
  });
}
