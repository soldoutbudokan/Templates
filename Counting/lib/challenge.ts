/** Six-hand coached challenges. The existing table engine remains the card/rules referee. */
import { getCorrectAction } from './basicStrategy';
import {
  advanceTable, applyTableAction, createTableSession, legalActions,
  submitTableCount, TableAction, TableCheckpoint, TableDecision,
  TablePhase, TableSessionState, ExposureEvent,
} from './blackjack';
import { estimateDecks, toTrueCount } from './countPolicy';

export type ChallengePresetId = 'rhythm' | 'table' | 'pace';
export type ChallengeUnits = 1 | 2 | 4;
export const CHALLENGE_HISTORY_LIMIT = 24;
export const CHALLENGE_PRESETS = {
  rhythm: { id: 'rhythm', name: 'Find your rhythm', speedMs: 900, checkpointEvery: 1, autoPlay: true, rounds: 6 },
  table: { id: 'table', name: 'Read the table', speedMs: 900, checkpointEvery: 2, autoPlay: false, rounds: 6 },
  pace: { id: 'pace', name: 'Hold the pace', speedMs: 650, checkpointEvery: 3, autoPlay: false, rounds: 6 },
} as const;

export type ChallengeInput =
  | { kind: 'advance' }
  | { kind: 'action'; action: TableAction }
  | { kind: 'bet'; units: ChallengeUnits }
  | { kind: 'count'; running: number | null; true: number | null }
  | { kind: 'continue' }
  | { kind: 'interrupt' };
export interface ChallengeBet {
  round: number;
  units: ChallengeUnits;
  expectedUnits: ChallengeUnits;
  trueTarget: number;
  correct: boolean;
  /** Never show correctness or the target before a count checkpoint is answered. */
  revealed: boolean;
}
export interface ChallengeRound {
  round: number;
  betUnits: ChallengeUnits;
  baseNetUnits: number;
  netUnits: number;
}
export interface ChallengeFeedback { checkpoint: TableCheckpoint; bets: ChallengeBet[] }
export interface ChallengeSummary {
  roundsCompleted: number;
  checkpoints: number;
  runningCorrect: number;
  trueCorrect: number;
  cleanCheckpoints: number;
  bestCleanStreak: number;
  decisions: number;
  decisionsCorrect: number;
  bets: number;
  betsCorrect: number;
  netUnits: number;
  /** Equal weight for the observed skills, independent of simulated hand profit. */
  skillScore: number;
  eligibleForRecord: boolean;
}
export interface ChallengeState {
  version: 1;
  id: string;
  presetId: ChallengePresetId;
  seed: number;
  startedAt: string;
  endedAt: string | null;
  replay: boolean;
  interrupted: boolean;
  completed: boolean;
  phase: TablePhase | 'bet' | 'feedback';
  /** Render only table.table, never the referee's private shoe or count. */
  table: TableSessionState;
  inputs: ChallengeInput[];
  bets: ChallengeBet[];
  checks: TableCheckpoint[];
  manualDecisions: TableDecision[];
  rounds: ChallengeRound[];
  feedback: ChallengeFeedback | null;
}
export interface ChallengeResult {
  version: 1;
  id: string;
  presetId: ChallengePresetId;
  seed: number;
  startedAt: string;
  endedAt: string;
  replay: boolean;
  interrupted: boolean;
  completed: boolean;
  inputs: ChallengeInput[];
  bets: ChallengeBet[];
  checks: TableCheckpoint[];
  manualDecisions: TableDecision[];
  rounds: ChallengeRound[];
  exposures: ExposureEvent[];
  summary: ChallengeSummary;
}

function validDate(value: unknown): value is string {
  return typeof value === 'string' && Number.isFinite(Date.parse(value)) && new Date(value).toISOString() === value;
}
function isPreset(value: unknown): value is ChallengePresetId {
  return value === 'rhythm' || value === 'table' || value === 'pace';
}
function validCount(value: number): boolean { return Number.isSafeInteger(value) && Math.abs(value) <= 1000; }
function currentRunning(table: TableSessionState): number { return table.exposures.at(-1)?.runningCount ?? 0; }
export function practiceBetUnits(trueCount: number): ChallengeUnits { return trueCount >= 2 ? 4 : trueCount >= 1 ? 2 : 1; }

export function createChallenge(presetId: ChallengePresetId, seed: number, options: {
  startedAt?: string; id?: string; replay?: boolean;
} = {}): ChallengeState {
  if (!isPreset(presetId) || !Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new RangeError('Invalid challenge settings.');
  const startedAt = options.startedAt ?? new Date().toISOString();
  if (!validDate(startedAt)) throw new RangeError('Invalid challenge start time.');
  const nonce = typeof globalThis.crypto?.randomUUID === 'function'
    ? globalThis.crypto.randomUUID() : Math.random().toString(36).slice(2);
  const id = options.id ?? `challenge-${Date.parse(startedAt)}-${seed}-${presetId}-${nonce}`;
  if (!/^[a-zA-Z0-9:_-]{1,180}$/.test(id)) throw new RangeError('Invalid challenge identity.');
  const table = createTableSession({ mode: 'practice', roundLimit: 10, otherPlayers: 2, speedMs: CHALLENGE_PRESETS[presetId].speedMs }, seed);
  return { version: 1, id, presetId, seed, startedAt, endedAt: null, replay: options.replay === true,
    interrupted: false, completed: false, phase: table.phase, table, inputs: [], bets: [], checks: [],
    manualDecisions: [], rounds: [], feedback: null };
}

function append(state: ChallengeState, input: ChallengeInput): ChallengeState {
  return { ...state, inputs: [...state.inputs, input] };
}
function expectedPlay(table: TableSessionState): TableAction {
  if (table.phase === 'insurance') return 'decline-insurance';
  const view = table.table;
  const cards = view.seats[view.activeSeat!].hands[view.activeHand!].cards;
  const upcard = view.dealer.cards[0];
  if (!upcard) throw new Error('The dealer upcard is not available.');
  const legal = legalActions(table);
  return getCorrectAction(cards, upcard, {
    canDouble: legal.includes('double'), canSplit: legal.includes('split'), canSurrender: legal.includes('surrender'),
  });
}
function withTable(state: ChallengeState, table: TableSessionState): ChallengeState {
  let rounds = state.rounds;
  if (table.phase === 'round-result' && !rounds.some(round => round.round === table.round)) {
    const units = state.bets.find(bet => bet.round === table.round)?.units ?? 1;
    const seat = table.table.seats.find(candidate => candidate.isUser);
    let baseNetUnits = table.table.insurance?.netUnits ?? 0;
    for (const hand of seat?.hands ?? []) {
      baseNetUnits += hand.outcome === 'Blackjack · 3:2' ? 1.5
        : hand.outcome === 'Win' ? hand.units
        : hand.outcome === 'Loss' ? -hand.units
        : hand.outcome === 'Surrender' ? -0.5 : 0;
    }
    rounds = [...rounds, { round: table.round, betUnits: units, baseNetUnits, netUnits: baseNetUnits * units }];
  }
  return { ...state, table, phase: table.phase, rounds, interrupted: state.interrupted || table.interrupted };
}
function nextRound(state: ChallengeState, running: number | null, trueCount: number | null): ChallengeState {
  // The sixth checkpoint is the finish line. No seventh round is created or exposed.
  if (state.table.round === 6) return { ...state, phase: 'complete', completed: true, feedback: null };
  return { ...withTable(state, submitTableCount(state.table, running, trueCount)), phase: 'bet', feedback: null };
}

/** At most one card appears per call. Rhythm mode also makes one basic-strategy action. */
export function advanceChallenge(previous: ChallengeState): ChallengeState {
  if (previous.completed || ['complete', 'bet', 'feedback'].includes(previous.phase)) return previous;
  let state: ChallengeState;
  if (previous.phase === 'count-checkpoint') {
    if (previous.table.round % CHALLENGE_PRESETS[previous.presetId].checkpointEvery === 0) return previous;
    state = nextRound(previous, null, null);
  } else if (previous.phase === 'player-turn' || previous.phase === 'insurance') {
    if (!CHALLENGE_PRESETS[previous.presetId].autoPlay) return previous;
    state = withTable(previous, applyTableAction(previous.table, expectedPlay(previous.table)));
  } else state = withTable(previous, advanceTable(previous.table));
  // Skipped checks are internal transitions; the UI never sees a question it should not ask.
  if (state.phase === 'count-checkpoint' && state.table.round % CHALLENGE_PRESETS[state.presetId].checkpointEvery !== 0) {
    state = nextRound(state, null, null);
  }
  return append(state, { kind: 'advance' });
}
export function applyChallengeAction(previous: ChallengeState, action: TableAction): ChallengeState {
  if (previous.completed || CHALLENGE_PRESETS[previous.presetId].autoPlay
    || !['player-turn', 'insurance'].includes(previous.phase)) throw new RangeError('A manual play is not available now.');
  const table = applyTableAction(previous.table, action);
  const state = withTable(previous, table);
  return append({ ...state, manualDecisions: [...previous.manualDecisions, ...table.decisions.slice(previous.table.decisions.length)] }, { kind: 'action', action });
}
export function submitChallengeBet(previous: ChallengeState, units: ChallengeUnits): ChallengeState {
  if (previous.phase !== 'bet' || ![1, 2, 4].includes(units)) throw new RangeError('Choose 1, 2, or 4 practice units before the next hand.');
  const trueTarget = toTrueCount(currentRunning(previous.table), estimateDecks(previous.table.remaining));
  const expectedUnits = practiceBetUnits(trueTarget);
  const bet: ChallengeBet = { round: previous.table.round, units, expectedUnits, trueTarget, correct: units === expectedUnits, revealed: false };
  return append({ ...previous, phase: previous.table.phase, bets: [...previous.bets, bet] }, { kind: 'bet', units });
}
export function submitChallengeCount(previous: ChallengeState, running: number | null, trueCount: number | null): ChallengeState {
  if (previous.phase !== 'count-checkpoint' || previous.table.round % CHALLENGE_PRESETS[previous.presetId].checkpointEvery !== 0) {
    throw new RangeError('There is no unanswered count check.');
  }
  const lost = running === null && trueCount === null;
  if (!lost && (running === null || trueCount === null || !validCount(running) || !validCount(trueCount))) {
    throw new RangeError('Enter both counts as signed whole numbers, or mark the count lost.');
  }
  const decksRemaining = estimateDecks(previous.table.remaining);
  const runningTarget = currentRunning(previous.table);
  const trueTarget = toTrueCount(runningTarget, decksRemaining);
  const checkpoint: TableCheckpoint = { round: previous.table.round, runningGuess: running, trueGuess: trueCount,
    runningTarget, trueTarget, decksRemaining, runningCorrect: running === runningTarget, trueCorrect: trueCount === trueTarget,
    arithmeticCorrect: running === null ? null : toTrueCount(running, decksRemaining) === trueCount };
  const newlyRevealed = previous.bets.filter(bet => !bet.revealed).map(bet => ({ ...bet, revealed: true }));
  return append({ ...previous, phase: 'feedback', checks: [...previous.checks, checkpoint],
    bets: previous.bets.map(bet => bet.revealed ? bet : { ...bet, revealed: true }),
    feedback: { checkpoint, bets: newlyRevealed } }, { kind: 'count', running, true: trueCount });
}
export function continueChallenge(previous: ChallengeState): ChallengeState {
  if (previous.phase !== 'feedback' || !previous.feedback) throw new RangeError('There is no count feedback to continue.');
  return append(nextRound(previous, previous.feedback.checkpoint.runningGuess, previous.feedback.checkpoint.trueGuess), { kind: 'continue' });
}
/** Pausing is allowed; flagging the attempt prevents resumed work from becoming a fresh record. */
export function markChallengeInterrupted(previous: ChallengeState): ChallengeState {
  if (previous.interrupted || previous.completed) return previous;
  return append({ ...previous, interrupted: true }, { kind: 'interrupt' });
}

export function summarizeChallenge(source: ChallengeState | ChallengeResult): ChallengeSummary {
  const checks = source.checks;
  const bets = source.bets.filter(bet => bet.revealed);
  const runningCorrect = checks.filter(check => check.runningCorrect).length;
  const trueCorrect = checks.filter(check => check.trueCorrect).length;
  const cleanCheckpoints = checks.filter(check => check.runningCorrect && check.trueCorrect).length;
  let streak = 0, bestCleanStreak = 0;
  for (const check of checks) { streak = check.runningCorrect && check.trueCorrect ? streak + 1 : 0; bestCleanStreak = Math.max(bestCleanStreak, streak); }
  const decisionsCorrect = source.manualDecisions.filter(decision => decision.correct).length;
  const betsCorrect = bets.filter(bet => bet.correct).length;
  const observed: number[] = [];
  if (checks.length) observed.push(runningCorrect / checks.length, trueCorrect / checks.length);
  if (source.manualDecisions.length) observed.push(decisionsCorrect / source.manualDecisions.length);
  if (bets.length) observed.push(betsCorrect / bets.length);
  return { roundsCompleted: source.rounds.length, checkpoints: checks.length, runningCorrect, trueCorrect, cleanCheckpoints, bestCleanStreak,
    decisions: source.manualDecisions.length, decisionsCorrect, bets: bets.length, betsCorrect,
    netUnits: source.rounds.reduce((sum, round) => sum + round.netUnits, 0),
    skillScore: observed.length ? Math.round(100 * observed.reduce((sum, value) => sum + value, 0) / observed.length) : 0,
    eligibleForRecord: source.completed && !source.interrupted && !source.replay };
}
export function finishChallenge(state: ChallengeState, endedAt = new Date().toISOString()): ChallengeResult {
  if (!validDate(endedAt) || Date.parse(endedAt) < Date.parse(state.startedAt)) throw new RangeError('Invalid challenge finish time.');
  return { version: 1, id: state.id, presetId: state.presetId, seed: state.seed, startedAt: state.startedAt, endedAt,
    replay: state.replay, interrupted: state.interrupted, completed: state.completed,
    inputs: [...state.inputs], bets: [...state.bets], checks: [...state.checks], manualDecisions: [...state.manualDecisions],
    rounds: [...state.rounds], exposures: [...state.table.exposures], summary: summarizeChallenge(state) };
}

function object(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value); }
function same(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) && Array.isArray(right)) return left.length === right.length && left.every((item, index) => same(item, right[index]));
  if (object(left) && object(right)) {
    const keys = Object.keys(left);
    return keys.length === Object.keys(right).length && keys.every(key => Object.hasOwn(right, key) && same(left[key], right[key]));
  }
  return false;
}
/** Recreate every transition from the seed. Cached scores, targets and outcomes are never trusted. */
export function readChallengeResult(raw: unknown): ChallengeResult | null {
  if (!object(raw) || raw.version !== 1 || !isPreset(raw.presetId) || !Number.isSafeInteger(raw.seed)
    || typeof raw.id !== 'string' || !validDate(raw.startedAt) || !validDate(raw.endedAt)
    || typeof raw.replay !== 'boolean' || typeof raw.interrupted !== 'boolean' || typeof raw.completed !== 'boolean'
    || !Array.isArray(raw.inputs) || raw.inputs.length > 1200) return null;
  try {
    let state = createChallenge(raw.presetId, raw.seed as number, { id: raw.id, startedAt: raw.startedAt, replay: raw.replay });
    for (const input of raw.inputs) {
      if (!object(input)) return null;
      const before = state.inputs.length;
      if (input.kind === 'advance') state = advanceChallenge(state);
      else if (input.kind === 'action') state = applyChallengeAction(state, input.action as TableAction);
      else if (input.kind === 'bet') state = submitChallengeBet(state, input.units as ChallengeUnits);
      else if (input.kind === 'count') state = submitChallengeCount(state, input.running as number, input.true as number);
      else if (input.kind === 'continue') state = continueChallenge(state);
      else if (input.kind === 'interrupt') state = markChallengeInterrupted(state);
      else return null;
      if (state.inputs.length !== before + 1 || !same(state.inputs.at(-1), input)) return null;
    }
    const canonical = finishChallenge(state, raw.endedAt);
    return same(raw, canonical) ? canonical : null;
  } catch { return null; }
}
export function validateChallengeHistory(raw: unknown): ChallengeResult[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  return raw.map(readChallengeResult).filter((result): result is ChallengeResult => {
    if (!result || seen.has(result.id)) return false;
    seen.add(result.id); return true;
  }).sort((a, b) => Date.parse(b.endedAt) - Date.parse(a.endedAt)).slice(0, CHALLENGE_HISTORY_LIMIT);
}

export function recommendChallenge(history: readonly ChallengeResult[]): { presetId: ChallengePresetId; reason: string } {
  const fresh = [...history].filter(result => summarizeChallenge(result).eligibleForRecord).sort((a, b) => Date.parse(b.endedAt) - Date.parse(a.endedAt));
  if (!fresh.length) return { presetId: 'rhythm', reason: 'Build a steady count through six hands. Playing decisions are automatic.' };
  const latest = fresh[0];
  const comparable = fresh.filter(result => result.presetId === latest.presetId);
  const recent = comparable.slice(0, 2);
  const strong = (result: ChallengeResult) => {
    const s = summarizeChallenge(result);
    return s.checkpoints > 0 && s.cleanCheckpoints === s.checkpoints && s.bets > 0 && s.betsCorrect / s.bets >= 0.8
      && (!s.decisions || s.decisionsCorrect / s.decisions >= 0.9);
  };
  if (recent.length === 2 && recent[0].seed !== recent[1].seed && recent.every(strong)) {
    if (latest.presetId === 'rhythm') return { presetId: 'table', reason: 'Two strong fresh runs. Try playing your hands with a count check every two rounds.' };
    if (latest.presetId === 'table') return { presetId: 'pace', reason: 'Two strong fresh runs. Try faster dealing and hold the count through three rounds.' };
  }
  return { presetId: latest.presetId, reason: latest.presetId === 'pace'
    ? 'Keep this pace accurate on fresh shoes, or try the separate unassisted full-table test.'
    : 'Keep the same conditions until two fresh runs show a steady count and accurate decisions.' };
}

/** Select a practice shoe with a raised-bet opportunity under baseline basic-strategy play. */
export function chooseChallengeSeed(baseSeed: number): number {
  if (!Number.isSafeInteger(baseSeed)) throw new RangeError('Invalid practice seed.');
  for (let offset = 0; offset < 32; offset++) {
    const seed = (baseSeed + offset) >>> 0;
    let table = createTableSession({ mode: 'practice', roundLimit: 10, otherPlayers: 2, speedMs: 900 }, seed);
    for (let step = 0; step < 450 && table.round < 6 && !table.ended; step++) {
      if (table.phase === 'count-checkpoint') {
        if (practiceBetUnits(toTrueCount(currentRunning(table), estimateDecks(table.remaining))) > 1) return seed;
        table = submitTableCount(table, null, null);
      } else if (table.phase === 'player-turn' || table.phase === 'insurance') table = applyTableAction(table, expectedPlay(table));
      else table = advanceTable(table);
    }
  }
  return baseSeed >>> 0;
}
