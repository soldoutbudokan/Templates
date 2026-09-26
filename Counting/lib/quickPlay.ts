import { Action, getCorrectAction } from './basicStrategy';
import { Card, Deck } from './deck';
import { StrategyCase, createStrategyCases, legalStrategyActions } from './strategyPractice';
import { seededRandom } from './training';

export const QUICK_DURATION_MS = 60_000;
export const MAX_QUICK_QUESTIONS = 200;
export const QUICK_STREAK_MIN_ANSWERS = 5;
export const QUICK_HISTORY_LIMIT = 20;
export const QUICK_DAY_LIMIT = 366;
export type QuickMode = 'counting' | 'strategy' | 'mixed';
export type QuickValue = number | Action;
export type QuickEndReason = 'timer' | 'question-limit' | 'ended' | 'interrupted';

interface QuestionBase { id: string; index: number }
export interface QuickCountQuestion extends QuestionBase {
  kind: 'counting'; cards: Card[]; countBefore: number; target: number; choices: number[];
}
export interface QuickStrategyQuestion extends QuestionBase {
  kind: 'strategy'; situation: StrategyCase; target: Action; choices: Action[];
}
export type QuickQuestion = QuickCountQuestion | QuickStrategyQuestion;
export interface QuickAnswer {
  index: number; value: QuickValue; elapsedMs: number; correct: boolean;
  combo: number; baseXp: number; bonusXp: number; xp: number;
}
export interface QuickSessionResult {
  id: string; seed: number; mode: QuickMode; startedAt: string; endedAt: string; localDate: string;
  elapsedMs: number; endReason: QuickEndReason; interrupted: boolean; answers: QuickAnswer[];
}
export interface QuickHistory {
  version: 1;
  sessions: QuickSessionResult[];
  activityDays: string[];
  /** Consecutive earned days immediately before the oldest retained activity day. */
  streakCarry: number;
  bestXp: number;
}
export const EMPTY_QUICK_HISTORY: QuickHistory = { version: 1, sessions: [], activityDays: [], streakCarry: 0, bestXp: 0 };

const MODES: QuickMode[] = ['counting', 'strategy', 'mixed'];
const MAX_XP = MAX_QUICK_QUESTIONS * 20;
const DAY_MS = 86_400_000;

function shuffle<T>(items: T[], random: () => number): T[] {
  const values = [...items];
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

function checkSeedAndMode(seed: number, mode: QuickMode) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffff_ffff || !MODES.includes(mode)) {
    throw new RangeError('Invalid quick-practice seed or mode.');
  }
}

/** A deterministic question stream, always fresh by seed. A count answer never
 * changes subsequent targets: feedback supplies the correct next anchor.
 * The eight-deck stream can supply all 200 pairs without a shuffle or reset.
 */
export function createQuickQuestions(seed: number, mode: QuickMode, count = MAX_QUICK_QUESTIONS): QuickQuestion[] {
  checkSeedAndMode(seed, mode);
  if (!Number.isSafeInteger(count) || count < 1 || count > MAX_QUICK_QUESTIONS) throw new RangeError('Invalid question count.');
  const random = seededRandom(seed);
  const deck = new Deck(8, 0.99, random);
  const questions: QuickQuestion[] = [];
  let strategyQueue: StrategyCase[] = [];
  for (let index = 0; index < count; index++) {
    const kind = mode === 'mixed' ? index % 2 === 0 ? 'counting' : 'strategy' : mode;
    const id = `${seed}:${mode}:${index}`;
    if (kind === 'counting') {
      const countBefore = deck.runningCount();
      const cards = deck.deal(2);
      const target = deck.runningCount();
      const offsets = shuffle([-3, -2, -1, 1, 2, 3], random).slice(0, 3);
      const choices = shuffle([target, ...offsets.map(offset => target + offset)], random);
      questions.push({ id, index, kind, cards, countBefore, target, choices });
    } else {
      if (!strategyQueue.length) strategyQueue = createStrategyCases(Math.floor(random() * 0x100000000), 'mixed', 30);
      const situation = strategyQueue.shift()!;
      const target = getCorrectAction(situation.player, situation.dealer, situation.options);
      questions.push({ id, index, kind, situation, target, choices: legalStrategyActions(situation) });
    }
  }
  return questions;
}

/** Ten points for a correct first answer; two extra points per three-answer
 * combo, capped at ten bonus points. A wrong answer scores zero and resets it.
 */
export function gradeQuickAnswer(question: QuickQuestion, value: QuickValue, previousCombo: number, elapsedMs: number): QuickAnswer {
  if (!Number.isSafeInteger(previousCombo) || previousCombo < 0 || previousCombo >= MAX_QUICK_QUESTIONS) throw new RangeError('Invalid combo.');
  if (!Number.isFinite(elapsedMs) || elapsedMs < 0 || elapsedMs >= QUICK_DURATION_MS) throw new RangeError('The answering clock has expired.');
  const validAnswer = question.kind === 'counting'
    ? typeof value === 'number' && Number.isSafeInteger(value) && Math.abs(value) <= 1000
    : (question.choices as QuickValue[]).includes(value);
  if (!validAnswer) throw new RangeError('Choose an available action or enter a signed whole count.');
  const correct = value === question.target;
  const combo = correct ? previousCombo + 1 : 0;
  const baseXp = correct ? 10 : 0;
  const bonusXp = correct ? 2 * Math.min(5, Math.floor(combo / 3)) : 0;
  return { index: question.index, value, elapsedMs, correct, combo, baseXp, bonusXp, xp: baseXp + bonusXp };
}

/** The index is required so a stale/double click cannot answer the next card.
 * Time is cumulative question-active time; feedback and pauses do not count.
 */
export function recordQuickAnswer(questions: QuickQuestion[], answers: QuickAnswer[], index: number,
  value: QuickValue, elapsedMs: number): QuickAnswer[] {
  if (!Number.isInteger(index) || index !== answers.length || !questions[index] || questions[index].index !== index) {
    throw new RangeError('Only the next unanswered question can be submitted.');
  }
  const previous = answers.at(-1);
  if (previous && (previous.index !== index - 1 || elapsedMs < previous.elapsedMs)) throw new RangeError('Answers must remain in order.');
  return [...answers, gradeQuickAnswer(questions[index], value, previous?.combo ?? 0, elapsedMs)];
}

export function summarizeQuickAnswers(answers: QuickAnswer[]) {
  return answers.reduce((total, answer) => ({
    attempts: total.attempts + 1, correct: total.correct + Number(answer.correct),
    xp: total.xp + answer.xp, bestCombo: Math.max(total.bestCombo, answer.combo),
  }), { attempts: 0, correct: 0, xp: 0, bestCombo: 0 });
}

export function localDateKey(date: Date = new Date()): string {
  if (!Number.isFinite(date.getTime())) throw new RangeError('Invalid date.');
  return `${date.getFullYear().toString().padStart(4, '0')}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function validDay(day: unknown): day is string {
  if (typeof day !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return false;
  const value = new Date(`${day}T00:00:00.000Z`);
  return Number.isFinite(value.getTime()) && value.toISOString().slice(0, 10) === day;
}

function dayNumber(day: string): number {
  if (!validDay(day)) throw new RangeError('Invalid local calendar day.');
  return Date.parse(`${day}T00:00:00.000Z`) / DAY_MS;
}

type ResultInput = Omit<QuickSessionResult, 'id' | 'localDate' | 'interrupted'> & { interrupted?: boolean };

function regradeResult(input: ResultInput, savedLocalDate?: string): QuickSessionResult {
  checkSeedAndMode(input.seed, input.mode);
  if (typeof input.startedAt !== 'string' || typeof input.endedAt !== 'string'
    || !Number.isFinite(Date.parse(input.startedAt)) || !Number.isFinite(Date.parse(input.endedAt))
    || Date.parse(input.endedAt) < Date.parse(input.startedAt)
    || !Number.isFinite(input.elapsedMs) || input.elapsedMs < 0 || input.elapsedMs > QUICK_DURATION_MS
    || !['timer', 'question-limit', 'ended', 'interrupted'].includes(input.endReason)
    || (input.endReason === 'timer' && input.elapsedMs !== QUICK_DURATION_MS)
    || (input.interrupted !== undefined && typeof input.interrupted !== 'boolean')
    || !Array.isArray(input.answers) || input.answers.length > MAX_QUICK_QUESTIONS
    || (input.endReason === 'question-limit' && input.answers.length !== MAX_QUICK_QUESTIONS)) {
    throw new RangeError('Invalid quick-practice result.');
  }
  const questions = createQuickQuestions(input.seed, input.mode);
  let answers: QuickAnswer[] = [];
  for (const answer of input.answers) {
    if (!answer || answer.elapsedMs > input.elapsedMs) throw new RangeError('Invalid answer timing.');
    answers = recordQuickAnswer(questions, answers, answer.index, answer.value, answer.elapsedMs);
  }
  const endedAt = new Date(input.endedAt).toISOString();
  const startedAt = new Date(input.startedAt).toISOString();
  const localDate = savedLocalDate ?? localDateKey(new Date(endedAt));
  if (!validDay(localDate) || Math.abs(dayNumber(localDate) - dayNumber(endedAt.slice(0, 10))) > 1) {
    throw new RangeError('Invalid saved completion day.');
  }
  return {
    id: `${input.seed}:${input.mode}:${startedAt}`, seed: input.seed, mode: input.mode,
    startedAt, endedAt, localDate, elapsedMs: input.elapsedMs, endReason: input.endReason,
    interrupted: input.interrupted === true || input.endReason === 'interrupted', answers,
  };
}

export function createQuickResult(input: ResultInput): QuickSessionResult { return regradeResult(input); }

/** A streak records completed practice days, not skill or perfect accuracy. */
export function quickRunCreditsDay(result: QuickSessionResult): boolean {
  return ((result.endReason === 'timer' && result.elapsedMs === QUICK_DURATION_MS)
    || (result.endReason === 'question-limit' && result.answers.length === MAX_QUICK_QUESTIONS))
    && result.answers.length >= QUICK_STREAK_MIN_ANSWERS;
}

function freshHistory(): QuickHistory { return { ...EMPTY_QUICK_HISTORY, sessions: [], activityDays: [] }; }

export function addQuickResult(history: QuickHistory, result: QuickSessionResult): QuickHistory {
  const canonical = regradeResult(result, result.localDate);
  if (history.sessions.some(session => session.id === canonical.id)) return history;
  const sessions = [...history.sessions, canonical].slice(-QUICK_HISTORY_LIMIT);
  let activityDays = [...history.activityDays];
  let streakCarry = history.streakCarry ?? 0;
  if (quickRunCreditsDay(canonical) && !activityDays.includes(canonical.localDate)) {
    activityDays.push(canonical.localDate);
    activityDays.sort();
    if (activityDays.length > QUICK_DAY_LIMIT) {
      const removed = activityDays.slice(0, -QUICK_DAY_LIMIT);
      const retained = activityDays.slice(-QUICK_DAY_LIMIT);
      let preceding = dayNumber(retained[0]);
      let carried = 0;
      for (let i = removed.length - 1; i >= 0 && dayNumber(removed[i]) === preceding - 1; i--) {
        carried++;
        preceding--;
      }
      // Preserve long uninterrupted streaks even when old day keys age out.
      streakCarry = carried === removed.length && removed[0] === history.activityDays[0]
        ? streakCarry + carried : carried;
      activityDays = retained;
    }
  }
  return { version: 1, sessions, activityDays, streakCarry,
    bestXp: Math.max(history.bestXp, summarizeQuickAnswers(canonical.answers).xp) };
}

export function quickHistoryStats(history: QuickHistory, today = localDateKey()) {
  const todayNumber = dayNumber(today);
  const days = new Set(history.activityDays.map(dayNumber));
  let cursor = days.has(todayNumber) ? todayNumber : todayNumber - 1;
  let streak = 0;
  while (days.has(cursor)) { streak++; cursor--; }
  if (streak > 0 && history.activityDays.length && cursor === dayNumber(history.activityDays[0]) - 1) streak += history.streakCarry ?? 0;
  return { bestXp: history.bestXp, streak };
}

/** Corrupt or oversized local storage never becomes a score or crashes the UI.
 * Cached per-answer scores are ignored and recomputed from seed + first answers.
 */
export function readQuickHistory(raw: string | null): { history: QuickHistory; reset: boolean } {
  if (raw === null) return { history: freshHistory(), reset: false };
  try {
    if (raw.length > 1_000_000) throw new Error('History exceeds its size limit.');
    const data = JSON.parse(raw) as QuickHistory;
    if (!data || data.version !== 1 || !Array.isArray(data.sessions) || data.sessions.length > QUICK_HISTORY_LIMIT
      || !Array.isArray(data.activityDays) || data.activityDays.length > QUICK_DAY_LIMIT
      || !Number.isSafeInteger(data.bestXp) || data.bestXp < 0 || data.bestXp > MAX_XP
      || (data.streakCarry !== undefined && (!Number.isSafeInteger(data.streakCarry) || data.streakCarry < 0 || data.streakCarry > 1_000_000))) {
      throw new Error('Invalid history.');
    }
    const activityDays = data.activityDays;
    if (activityDays.some((day, index) => !validDay(day) || (index > 0 && day <= activityDays[index - 1]))) throw new Error('Invalid activity days.');
    const streakCarry = data.streakCarry ?? 0;
    if (streakCarry > 0 && activityDays.length !== QUICK_DAY_LIMIT) throw new Error('Invalid archived streak.');
    const sessions = data.sessions.map(session => {
      if (!session || !validDay(session.localDate)) throw new Error('Invalid session.');
      const canonical = regradeResult(session, session.localDate);
      if (session.id !== canonical.id) throw new Error('Invalid session identity.');
      return canonical;
    });
    if (new Set(sessions.map(session => session.id)).size !== sessions.length) throw new Error('Duplicate saved session.');
    if (sessions.some(session => summarizeQuickAnswers(session.answers).xp > data.bestXp
      || (quickRunCreditsDay(session) && !activityDays.includes(session.localDate)
        && (!activityDays.length || session.localDate >= activityDays[0])))) throw new Error('Inconsistent history.');
    return { history: { version: 1, sessions, activityDays: [...activityDays], streakCarry, bestXp: data.bestXp }, reset: false };
  } catch {
    return { history: freshHistory(), reset: true };
  }
}
