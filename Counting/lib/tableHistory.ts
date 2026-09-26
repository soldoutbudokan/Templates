import {
  advanceTable, applyTableAction, createTableSession, finishTableSession, submitTableCount,
  TableAction, TableConfig, TableSessionResult,
} from './blackjack';

const HISTORY_LIMIT = 24;
const MAX_STORED_RECORDS = 128;
const MAX_REPLAY_STEPS = 2000;
const MAX_DECISIONS = 512;
const MAX_ROUNDS = 78;
const MAX_EXPOSURES = 312;
const ACTIONS: readonly TableAction[] = ['hit', 'stand', 'double', 'split', 'surrender', 'insure', 'decline-insurance'];

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function integer(value: unknown, min: number, max: number): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= min && value <= max;
}
function guess(value: unknown): value is number | null {
  return value === null || integer(value, -1000, 1000);
}
function invalid(): never { throw new Error('Invalid saved table session.'); }

/** Rebuild the referee state from learner inputs; persisted targets and scores are never trusted. */
function restoreSession(entry: unknown): TableSessionResult {
  if (!object(entry) || entry.version !== 1 || !integer(entry.seed, 0, 0xFFFFFFFF)
    || typeof entry.completed !== 'boolean' || typeof entry.interrupted !== 'boolean'
    || typeof entry.endedAt !== 'string' || entry.endedAt.length > 40
    || !/^\d{4}-\d{2}-\d{2}T/.test(entry.endedAt) || !Number.isFinite(Date.parse(entry.endedAt))
    || !object(entry.config) || !Array.isArray(entry.decisions) || entry.decisions.length > MAX_DECISIONS
    || !Array.isArray(entry.checkpoints) || entry.checkpoints.length > MAX_ROUNDS
    || !Array.isArray(entry.exposures) || entry.exposures.length > MAX_EXPOSURES
    || entry.rounds !== entry.checkpoints.length) invalid();

  const settings = entry.config;
  if ((settings.mode !== 'practice' && settings.mode !== 'test')
    || (settings.roundLimit !== 0 && settings.roundLimit !== 10)
    || (settings.otherPlayers !== 0 && settings.otherPlayers !== 2)
    || typeof settings.speedMs !== 'number' || !Number.isFinite(settings.speedMs)
    || settings.speedMs < 100 || settings.speedMs > 5000) invalid();
  const config: TableConfig = { mode: settings.mode, roundLimit: settings.roundLimit,
    otherPlayers: settings.otherPlayers, speedMs: settings.speedMs };
  let state = createTableSession(config, entry.seed);
  if (entry.id !== state.id) invalid();

  const decisions = entry.decisions.map((value: unknown) => {
    if (!object(value) || !integer(value.round, 1, MAX_ROUNDS) || !integer(value.seatIndex, 0, 2)
      || !integer(value.handIndex, 0, 3) || (value.kind !== 'play' && value.kind !== 'insurance')
      || typeof value.action !== 'string' || !ACTIONS.includes(value.action as TableAction)) invalid();
    return { round: value.round, seatIndex: value.seatIndex, handIndex: value.handIndex,
      kind: value.kind, action: value.action as TableAction };
  });
  const checkpoints = entry.checkpoints.map((value: unknown) => {
    if (!object(value) || !integer(value.round, 1, MAX_ROUNDS)
      || !guess(value.runningGuess) || !guess(value.trueGuess)) invalid();
    return { round: value.round, runningGuess: value.runningGuess, trueGuess: value.trueGuess };
  });

  let decisionIndex = 0;
  let checkpointIndex = 0;
  // Length preserves a partial round's stopping point; every exposed card and count is regenerated.
  const exposureCount = entry.exposures.length;
  for (let step = 0; step < MAX_REPLAY_STEPS; step++) {
    if (decisionIndex === decisions.length && checkpointIndex === checkpoints.length
      && state.exposures.length === exposureCount) break;
    if (state.ended || state.exposures.length > exposureCount) invalid();
    if (state.phase === 'player-turn' || state.phase === 'insurance') {
      const input = decisions[decisionIndex];
      if (!input) invalid();
      state = applyTableAction(state, input.action);
      const actual = state.decisions[decisionIndex];
      if (!actual || actual.round !== input.round || actual.seatIndex !== input.seatIndex
        || actual.handIndex !== input.handIndex || actual.kind !== input.kind) invalid();
      decisionIndex++;
    } else if (state.phase === 'count-checkpoint') {
      const input = checkpoints[checkpointIndex];
      if (!input || input.round !== state.round) invalid();
      state = submitTableCount(state, input.runningGuess, input.trueGuess);
      checkpointIndex++;
    } else state = advanceTable(state);
  }
  if (decisionIndex !== decisions.length || checkpointIndex !== checkpoints.length
    || state.exposures.length !== exposureCount || state.completed !== entry.completed) invalid();
  return { ...finishTableSession(state), interrupted: state.interrupted || entry.interrupted, endedAt: entry.endedAt };
}

/** Invalid envelopes are reported to the caller; a corrupt entry cannot erase its valid neighbors. */
export function readTableHistory(raw: string | null): TableSessionResult[] {
  if (!raw) return [];
  if (raw.length > 8_000_000) throw new Error('Saved table history is too large.');
  const data: unknown = JSON.parse(raw);
  if (!object(data) || data.version !== 1 || !Array.isArray(data.sessions)
    || data.sessions.length > MAX_STORED_RECORDS) throw new Error('Unrecognized table-history format.');
  const restored: TableSessionResult[] = [];
  for (const entry of data.sessions.slice(-HISTORY_LIMIT)) {
    try { restored.push(restoreSession(entry)); }
    catch { /* Discard only this malformed or inconsistent record. */ }
  }
  return restored;
}

export function tableResultLabel(result: TableSessionResult): string {
  if (result.config.mode === 'practice') return result.interrupted ? 'Interrupted table practice' : 'Table practice';
  if (result.interrupted) return 'Interrupted full test · practice only';
  return result.completed ? 'Uninterrupted full test' : 'Partial full test';
}

/** A clean result describes this completed test, never unobserved skills or future casino performance. */
export function cleanTableTest(result: TableSessionResult): boolean {
  return result.config.mode === 'test' && result.completed && !result.interrupted
    && result.decisions.some(decision => decision.kind === 'play') && result.checkpoints.length > 0
    && result.decisions.every(decision => decision.correct)
    && result.checkpoints.every(checkpoint => checkpoint.runningCorrect && checkpoint.trueCorrect);
}
