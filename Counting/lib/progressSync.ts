import { TableSessionResult } from './blackjack';
import { EMPTY_QUICK_HISTORY, QUICK_DAY_LIMIT, QUICK_HISTORY_LIMIT, QuickHistory, QuickSessionResult, readQuickHistory } from './quickPlay';
import { readTableHistory } from './tableHistory';
import { readHistory, SessionResult } from './training';

export const DEFAULT_PROGRESS_TIME_ZONE = 'America/Toronto';
export const MAX_PROGRESS_BYTES = 4_000_000;
export const PROGRESS_HISTORY_LIMIT = 24;
const DAY_MS = 86_400_000;

export interface ProgressData {
  version: 1;
  timeZone: string;
  quick: QuickHistory;
  guided: SessionResult[];
  table: TableSessionResult[];
  /** Clearing a history must also remove old copies arriving from another device. */
  clearedBefore: { guided: string | null; table: string | null };
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function normalizeProgressTimeZone(timeZone: string): string {
  if (typeof timeZone !== 'string' || !timeZone || timeZone.length > 100) throw new Error('Invalid progress timezone.');
  try { return new Intl.DateTimeFormat('en', { timeZone }).resolvedOptions().timeZone; }
  catch { throw new Error('Invalid progress timezone.'); }
}

/** A profile keeps the same calendar day on every device, including when traveling. */
export function progressDateKey(date: Date | string = new Date(), timeZone = DEFAULT_PROGRESS_TIME_ZONE): string {
  const value = date instanceof Date ? date : new Date(date);
  if (!Number.isFinite(value.getTime())) throw new Error('Invalid progress date.');
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: normalizeProgressTimeZone(timeZone), year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(value);
  const part = (name: Intl.DateTimeFormatPartTypes) => parts.find(item => item.type === name)!.value;
  return `${part('year')}-${part('month')}-${part('day')}`;
}

export function emptyProgress(timeZone = DEFAULT_PROGRESS_TIME_ZONE): ProgressData {
  return {
    version: 1, timeZone: normalizeProgressTimeZone(timeZone),
    quick: { ...EMPTY_QUICK_HISTORY, sessions: [], activityDays: [] }, guided: [], table: [],
    clearedBefore: { guided: null, table: null },
  };
}

function timestamp(value: unknown): string {
  if (typeof value !== 'string' || value.length > 40 || !/^\d{4}-\d{2}-\d{2}T/.test(value)
    || !Number.isFinite(Date.parse(value))) throw new Error('Invalid progress timestamp.');
  return new Date(value).toISOString();
}

function uniqueIds(values: { id: string }[]) {
  if (new Set(values.map(value => value.id)).size !== values.length) throw new Error('Duplicate progress session.');
}

function bySession(a: { id: string; endedAt: string }, b: { id: string; endedAt: string }): number {
  return a.endedAt < b.endedAt ? -1 : a.endedAt > b.endedAt ? 1 : a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/** Unlike tolerant local readers, a sync document is rejected as a whole when malformed.
 * All cards and scores are rebuilt by the existing history readers before merging.
 */
export function readProgress(raw: string | null): ProgressData {
  if (raw === null) return emptyProgress();
  if (typeof raw !== 'string' || raw.length > MAX_PROGRESS_BYTES || new TextEncoder().encode(raw).length > MAX_PROGRESS_BYTES) {
    throw new Error('Progress exceeds its size limit.');
  }
  const data: unknown = JSON.parse(raw);
  if (!object(data) || data.version !== 1 || typeof data.timeZone !== 'string' || !object(data.quick)
    || !Array.isArray(data.guided) || data.guided.length > PROGRESS_HISTORY_LIMIT
    || !Array.isArray(data.table) || data.table.length > PROGRESS_HISTORY_LIMIT || !object(data.clearedBefore)) {
    throw new Error('Unrecognized progress format.');
  }
  const timeZone = normalizeProgressTimeZone(data.timeZone);
  const clearedBefore = {
    guided: data.clearedBefore.guided === null ? null : timestamp(data.clearedBefore.guided),
    table: data.clearedBefore.table === null ? null : timestamp(data.clearedBefore.table),
  };
  const parsedQuick = readQuickHistory(JSON.stringify(data.quick));
  if (parsedQuick.reset) throw new Error('Invalid quick-play progress.');
  // The guided local reader normalizes a saved identity; sync must also verify it.
  if (data.guided.some(entry => !object(entry) || !Number.isSafeInteger(entry.seed)
    || (entry.seed as number) < 0 || (entry.seed as number) > 0xffff_ffff)) throw new Error('Invalid guided progress.');
  const guided = readHistory(JSON.stringify({ version: 1, sessions: data.guided }));
  if (guided.some((entry, index) => entry.id !== (data.guided as Record<string, unknown>[])[index].id)) {
    throw new Error('Invalid guided session identity.');
  }
  const table = readTableHistory(JSON.stringify({ version: 1, sessions: data.table }));
  if (table.length !== data.table.length) throw new Error('Invalid table progress.');
  uniqueIds(guided);
  uniqueIds(table);
  return {
    version: 1, timeZone,
    quick: { ...parsedQuick.history, sessions: [...parsedQuick.history.sessions].sort(bySession) },
    guided: guided.map(entry => ({ ...entry, endedAt: timestamp(entry.endedAt) }))
      .filter(entry => !clearedBefore.guided || entry.endedAt > clearedBefore.guided).sort(bySession),
    table: table.map(entry => ({ ...entry, endedAt: timestamp(entry.endedAt) }))
      .filter(entry => !clearedBefore.table || entry.endedAt > clearedBefore.table).sort(bySession),
    clearedBefore,
  };
}

type SavedSession = QuickSessionResult | SessionResult | TableSessionResult;
function completeness(session: SavedSession): number {
  if ('decisions' in session) return session.decisions.length + session.checkpoints.length + session.exposures.length;
  return session.answers.length;
}

/** Session IDs are immutable. If two valid copies differ, keep the one with more
 * recorded input; canonical serialized ordering breaks ties independently of
 * arrival order. A replayed upload never adds points or creates another session.
 */
function mergeSessions<T extends SavedSession>(left: T[], right: T[], limit: number, clearedBefore: string | null = null): T[] {
  const sessions = new Map<string, T>();
  for (const candidate of [...left, ...right]) {
    if (clearedBefore && candidate.endedAt <= clearedBefore) continue;
    const existing = sessions.get(candidate.id);
    if (!existing || completeness(candidate) > completeness(existing)
      || (completeness(candidate) === completeness(existing) && JSON.stringify(candidate) > JSON.stringify(existing))) {
      sessions.set(candidate.id, candidate);
    }
  }
  return Array.from(sessions.values()).sort(bySession).slice(-limit);
}

type DayRange = { start: number; end: number };
function dayNumber(day: string): number { return Date.parse(`${day}T00:00:00.000Z`) / DAY_MS; }
function dayKey(day: number): string { return new Date(day * DAY_MS).toISOString().slice(0, 10); }

/** Archived carries describe ranges, not amounts to add. Unioning those ranges
 * prevents overlapping device backups from multiplying a long streak. At most
 * 366 day keys are materialized, even for a million-day legacy carry.
 */
function mergeActivity(left: QuickHistory, right: QuickHistory): Pick<QuickHistory, 'activityDays' | 'streakCarry'> {
  const ranges: DayRange[] = [];
  for (const history of [left, right]) {
    ranges.push(...history.activityDays.map(day => ({ start: dayNumber(day), end: dayNumber(day) })));
    if (history.streakCarry && history.activityDays.length) {
      const oldest = dayNumber(history.activityDays[0]);
      ranges.push({ start: oldest - history.streakCarry, end: oldest - 1 });
    }
  }
  ranges.sort((a, b) => a.start - b.start || a.end - b.end);
  const joined: DayRange[] = [];
  for (const range of ranges) {
    const previous = joined.at(-1);
    if (previous && range.start <= previous.end + 1) previous.end = Math.max(previous.end, range.end);
    else joined.push({ ...range });
  }
  const retained: number[] = [];
  for (let index = joined.length - 1; index >= 0 && retained.length < QUICK_DAY_LIMIT; index--) {
    const range = joined[index];
    for (let day = range.end; day >= range.start && retained.length < QUICK_DAY_LIMIT; day--) retained.push(day);
  }
  retained.reverse();
  const oldest = retained[0];
  const containing = joined.find(range => range.start <= oldest && range.end >= oldest);
  return { activityDays: retained.map(dayKey), streakCarry: containing ? Math.min(1_000_000, oldest - containing.start) : 0 };
}

function later(left: string | null, right: string | null): string | null {
  return !left ? right : !right ? left : left > right ? left : right;
}

/** Call this inside the storage compare-and-swap retry. The stored profile's
 * timezone is authoritative; stale devices may contribute sessions, never
 * replace a newer device's complete history. Legacy earned dates stay intact.
 */
export function mergeProgress(current: ProgressData, incoming: ProgressData): ProgressData {
  const left = readProgress(JSON.stringify(current));
  const right = readProgress(JSON.stringify(incoming));
  const clearedBefore = {
    guided: later(left.clearedBefore.guided, right.clearedBefore.guided),
    table: later(left.clearedBefore.table, right.clearedBefore.table),
  };
  return {
    version: 1, timeZone: left.timeZone,
    quick: {
      version: 1,
      sessions: mergeSessions(left.quick.sessions, right.quick.sessions, QUICK_HISTORY_LIMIT),
      ...mergeActivity(left.quick, right.quick),
      bestXp: Math.max(left.quick.bestXp, right.quick.bestXp),
    },
    guided: mergeSessions(left.guided, right.guided, PROGRESS_HISTORY_LIMIT, clearedBefore.guided),
    table: mergeSessions(left.table, right.table, PROGRESS_HISTORY_LIMIT, clearedBefore.table),
    clearedBefore,
  };
}

export function clearProgressHistory(data: ProgressData, kind: 'guided' | 'table', date: Date | string = new Date()): ProgressData {
  if (kind !== 'guided' && kind !== 'table') throw new Error('Unknown progress history.');
  const cleared = readProgress(JSON.stringify(data));
  const instant = timestamp(date instanceof Date ? date.toISOString() : date);
  cleared.clearedBefore[kind] = later(cleared.clearedBefore[kind], instant);
  if (kind === 'guided') cleared.guided = cleared.guided.filter(session => session.endedAt > cleared.clearedBefore.guided!);
  else cleared.table = cleared.table.filter(session => session.endedAt > cleared.clearedBefore.table!);
  return cleared;
}
