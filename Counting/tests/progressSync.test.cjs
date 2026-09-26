const test = require('node:test');
const assert = require('node:assert/strict');
const {
  DEFAULT_PROGRESS_TIME_ZONE, MAX_PROGRESS_BYTES, emptyProgress, readProgress,
  mergeProgress, progressDateKey, clearProgressHistory,
} = require('../lib/progressSync.ts');
const {
  createQuickQuestions, createQuickResult, recordQuickAnswer, addQuickResult,
  quickHistoryStats, readQuickHistory,
} = require('../lib/quickPlay.ts');
const { createTrainingPlan, gradeCheckpoint } = require('../lib/training.ts');
const { createTableSession, finishTableSession } = require('../lib/blackjack.ts');

const dayAt = offset => new Date(Date.UTC(2025, 0, 1 + offset)).toISOString().slice(0, 10);
const instantAt = offset => `${dayAt(offset)}T12:01:00.000Z`;

function quick(seed, day = '2026-09-26', count = 5) {
  const questions = createQuickQuestions(seed, 'counting');
  let answers = [];
  for (let i = 0; i < count; i++) answers = recordQuickAnswer(questions, answers, i, questions[i].target, i * 100);
  return createQuickResult({ seed, mode: 'counting', startedAt: `${day}T12:00:00.000Z`,
    endedAt: `${day}T12:01:00.000Z`, elapsedMs: 60_000, endReason: 'timer', answers });
}

function guided(seed, endedAt = '2026-09-26T12:00:00.000Z', count = 1) {
  const plan = createTrainingPlan(seed, 'practice', 'running', 900);
  return { ...plan, endedAt, completed: false, interrupted: false, assisted: false,
    answers: plan.checkpoints.slice(0, count).map((point, index) => gradeCheckpoint(plan, index, point.runningCount, null, 250)) };
}

function table(seed, endedAt = '2026-09-26T12:00:00.000Z') {
  return { ...finishTableSession(createTableSession({ mode: 'test', roundLimit: 10, otherPlayers: 2, speedMs: 900 }, seed)), endedAt };
}

function withSessions(seed, day = '2026-09-26') {
  const data = emptyProgress();
  data.quick = addQuickResult(data.quick, quick(seed, day));
  data.guided = [guided(seed, `${day}T12:00:00.000Z`)];
  data.table = [table(seed, `${day}T12:00:00.000Z`)];
  return data;
}

test('two devices retain concurrent sessions, daily credit and best scores without duplicate uploads', () => {
  const common = withSessions(1, '2026-09-24');
  const laptop = mergeProgress(common, withSessions(2, '2026-09-25'));
  const phone = mergeProgress(common, withSessions(3, '2026-09-26'));
  const merged = mergeProgress(laptop, phone);
  assert.deepEqual(merged, mergeProgress(phone, laptop));
  assert.deepEqual(mergeProgress(merged, laptop), merged);
  assert.deepEqual(mergeProgress(merged, merged), merged);
  assert.deepEqual(merged.quick.sessions.map(session => session.seed), [1, 2, 3]);
  assert.deepEqual(merged.guided.map(session => session.seed), [1, 2, 3]);
  assert.deepEqual(merged.table.map(session => session.seed), [1, 2, 3]);
  assert.deepEqual(merged.quick.activityDays, ['2026-09-24', '2026-09-25', '2026-09-26']);
  assert.equal(quickHistoryStats(merged.quick, '2026-09-26').streak, 3);
  assert.equal(merged.quick.bestXp, common.quick.bestXp);
  assert.deepEqual(readProgress(JSON.stringify(merged)), merged);
});

test('same immutable ID chooses more recorded input with deterministic conflict ties', () => {
  const short = withSessions(10);
  const longer = withSessions(10);
  longer.quick = addQuickResult(emptyProgress().quick, quick(10, '2026-09-26', 8));
  longer.guided = [guided(10, '2026-09-26T12:00:00.000Z', 2)];
  const merged = mergeProgress(short, longer);
  assert.equal(merged.quick.sessions.length, 1);
  assert.equal(merged.quick.sessions[0].answers.length, 8);
  assert.equal(merged.guided.length, 1);
  assert.equal(merged.guided[0].answers.length, 2);
  assert.equal(merged.table.length, 1);
  assert.deepEqual(merged, mergeProgress(longer, short));
  const conflict = structuredClone(short);
  conflict.guided[0].interrupted = true;
  conflict.table[0].interrupted = true;
  assert.deepEqual(mergeProgress(short, conflict), mergeProgress(conflict, short));
});

test('history caps keep the newest sessions with stable ordering across repeated merges', () => {
  const devices = [emptyProgress(), emptyProgress(), emptyProgress()];
  for (let i = 0; i < 30; i++) {
    const device = devices[i % 3];
    device.quick = addQuickResult(device.quick, quick(i + 1, dayAt(i)));
    device.guided.push(guided(i + 1, instantAt(i)));
    device.table.push(table(i + 1, instantAt(i)));
  }
  const merged = mergeProgress(mergeProgress(devices[0], devices[1]), devices[2]);
  const reordered = mergeProgress(devices[0], mergeProgress(devices[2], devices[1]));
  assert.deepEqual(merged, reordered);
  assert.deepEqual(merged.quick.sessions.map(session => session.seed), Array.from({ length: 20 }, (_, i) => i + 11));
  for (const kind of ['guided', 'table']) assert.deepEqual(merged[kind].map(session => session.seed), Array.from({ length: 24 }, (_, i) => i + 7));
  assert.equal(merged.quick.activityDays.length, 30);
  assert.deepEqual(mergeProgress(merged, devices[0]), merged, 'old device uploads cannot evict newer sessions');
});

test('overlapping archived streak ranges are unioned rather than added', () => {
  const old = emptyProgress();
  old.quick.activityDays = Array.from({ length: 366 }, (_, i) => dayAt(i + 10));
  old.quick.streakCarry = 10;
  const newer = emptyProgress();
  newer.quick.activityDays = Array.from({ length: 366 }, (_, i) => dayAt(i + 12));
  newer.quick.streakCarry = 12;
  const merged = mergeProgress(old, newer);
  assert.equal(merged.quick.activityDays.length, 366);
  assert.equal(merged.quick.streakCarry, 12, 'overlapping archived days count once');
  assert.equal(quickHistoryStats(merged.quick, dayAt(377)).streak, 378);
  assert.deepEqual(merged, mergeProgress(newer, old));
  assert.deepEqual(mergeProgress(merged, old), merged);
  assert.equal(readQuickHistory(JSON.stringify(merged.quick)).reset, false);
});

test('separate old streaks stay separate while an earned bridging day can join them', () => {
  const left = emptyProgress();
  left.quick.activityDays = Array.from({ length: 366 }, (_, i) => dayAt(i));
  const right = emptyProgress();
  right.quick.activityDays = [dayAt(367), dayAt(368)];
  let merged = mergeProgress(left, right);
  assert.equal(quickHistoryStats(merged.quick, dayAt(368)).streak, 2);
  const bridge = emptyProgress();
  bridge.quick.activityDays = [dayAt(366)];
  merged = mergeProgress(merged, bridge);
  assert.equal(quickHistoryStats(merged.quick, dayAt(368)).streak, 369);
  assert.equal(merged.quick.streakCarry, 3);
});

test('fixed profile timezone handles midnight and daylight-saving changes independently of device timezone', () => {
  assert.equal(DEFAULT_PROGRESS_TIME_ZONE, 'America/Toronto');
  const original = process.env.TZ;
  try {
    for (const deviceZone of ['UTC', 'Asia/Tokyo', 'America/Los_Angeles']) {
      process.env.TZ = deviceZone;
      assert.equal(progressDateKey('2026-09-27T03:59:59.999Z'), '2026-09-26');
      assert.equal(progressDateKey('2026-09-27T04:00:00.000Z'), '2026-09-27');
      assert.equal(progressDateKey('2026-03-08T04:59:59.999Z'), '2026-03-07');
      assert.equal(progressDateKey('2026-03-08T05:00:00.000Z'), '2026-03-08');
      assert.equal(progressDateKey('2026-03-08T06:59:59.999Z'), '2026-03-08');
      assert.equal(progressDateKey('2026-03-08T07:00:00.000Z'), '2026-03-08');
      assert.equal(progressDateKey('2026-11-01T05:30:00.000Z'), '2026-11-01');
      assert.equal(progressDateKey('2026-11-01T06:30:00.000Z'), '2026-11-01');
      assert.equal(progressDateKey('2026-09-27T04:00:00.000Z', 'America/Los_Angeles'), '2026-09-26');
    }
  } finally {
    if (original === undefined) delete process.env.TZ; else process.env.TZ = original;
  }
  assert.throws(() => progressDateKey('not a date'));
  assert.throws(() => progressDateKey(new Date(), 'Made/Up'));
});

test('server profile timezone wins without reinterpreting previously earned legacy dates', () => {
  const server = emptyProgress('America/Toronto');
  const legacy = withSessions(12);
  legacy.timeZone = 'Asia/Tokyo';
  const merged = mergeProgress(server, legacy);
  assert.equal(merged.timeZone, 'America/Toronto');
  assert.deepEqual(merged.quick, legacy.quick);
});

test('clear watermarks prevent stale devices restoring erased histories but allow new sessions', () => {
  const old = withSessions(20, '2026-09-24');
  const clearedGuided = clearProgressHistory(old, 'guided', '2026-09-24T12:00:00.000Z');
  assert.equal(clearedGuided.guided.length, 0, 'cutoff is inclusive');
  assert.equal(clearedGuided.table.length, 1);
  assert.deepEqual(clearedGuided.quick, old.quick);
  const cleared = clearProgressHistory(clearedGuided, 'table', '2026-09-25T12:00:00.000Z');
  const staleUpload = mergeProgress(old, withSessions(21, '2026-09-26'));
  const merged = mergeProgress(cleared, staleUpload);
  assert.deepEqual(merged.guided.map(session => session.seed), [21]);
  assert.deepEqual(merged.table.map(session => session.seed), [21]);
  assert.deepEqual(merged, mergeProgress(staleUpload, cleared));
  assert.deepEqual(mergeProgress(merged, old), merged);
  assert.deepEqual(clearProgressHistory(merged, 'table', '2026-09-24T12:00:00.000Z'), merged, 'an older cutoff cannot reverse a clear');
});

test('sync regrades cached scores and rejects malformed or oversized documents as a whole', () => {
  const data = withSessions(25);
  const scores = structuredClone(data);
  scores.quick.sessions[0].answers[0].correct = false;
  scores.quick.sessions[0].answers[0].xp = 9000;
  scores.guided[0].answers[0].runningCorrect = false;
  assert.deepEqual(readProgress(JSON.stringify(scores)), data);
  assert.deepEqual(readProgress(null), emptyProgress());
  const corruptions = [
    null, [], {}, { ...data, version: 2 }, { ...data, timeZone: 'Fake/Zone' },
    { ...data, quick: { ...data.quick, bestXp: -1 } },
    { ...data, guided: [{ ...data.guided[0], id: 'forged' }] },
    { ...data, guided: [{ ...data.guided[0], seed: -1 }] },
    { ...data, guided: [data.guided[0], data.guided[0]] },
    { ...data, table: [null] }, { ...data, table: [{ ...data.table[0], id: 'forged' }] },
    { ...data, table: [data.table[0], data.table[0]] },
    { ...data, table: Array(25).fill(data.table[0]) },
    { ...data, clearedBefore: { guided: 'today', table: null } },
    { ...data, clearedBefore: {} },
  ];
  for (const corruption of corruptions) assert.throws(() => readProgress(JSON.stringify(corruption)));
  assert.throws(() => readProgress('{'));
  assert.throws(() => readProgress('x'.repeat(MAX_PROGRESS_BYTES + 1)), /size limit/);
  assert.throws(() => readProgress(JSON.stringify({ ...data, extra: 'é'.repeat(MAX_PROGRESS_BYTES / 2) })), /size limit/,
    'limit measures UTF-8 bytes, not just characters');
});
