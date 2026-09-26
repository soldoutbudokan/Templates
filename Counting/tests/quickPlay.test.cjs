const test = require('node:test');
const assert = require('node:assert/strict');
const { getCardValue } = require('../lib/deck.ts');
const { getCorrectAction } = require('../lib/basicStrategy.ts');
const { QUICK_DURATION_MS, MAX_QUICK_QUESTIONS, EMPTY_QUICK_HISTORY, createQuickQuestions,
  gradeQuickAnswer, recordQuickAnswer, summarizeQuickAnswers, createQuickResult, localDateKey,
  quickRunCreditsDay, addQuickResult, quickHistoryStats, readQuickHistory } = require('../lib/quickPlay.ts');

function makeResult(seed = 1, day = '2026-09-26', count = 5, endReason = 'timer', extra = {}) {
  const questions = createQuickQuestions(seed, 'counting');
  let answers = [];
  for (let index = 0; index < count; index++) answers = recordQuickAnswer(questions, answers, index, questions[index].target, 100 * (index + 1));
  return createQuickResult({ seed, mode: 'counting', startedAt: `${day}T12:00:00.000Z`, endedAt: `${day}T12:01:00.000Z`,
    elapsedMs: endReason === 'timer' ? QUICK_DURATION_MS : 30_000, endReason, answers, ...extra });
}

test('deterministic mixed questions alternate skills and counting remains cumulative across strategy questions', () => {
  const questions = createQuickQuestions(14, 'mixed');
  assert.equal(questions.length, MAX_QUICK_QUESTIONS);
  assert.deepEqual(questions.slice(0, 20), createQuickQuestions(14, 'mixed', 20));
  assert.notDeepEqual(questions, createQuickQuestions(15, 'mixed'));
  let count = 0;
  for (const [index, question] of questions.entries()) {
    assert.equal(question.index, index);
    assert.equal(question.kind, index % 2 ? 'strategy' : 'counting');
    if (question.kind === 'counting') {
      assert.equal(question.cards.length, 2);
      assert.equal(question.countBefore, count);
      count += question.cards.reduce((total, card) => total + getCardValue(card), 0);
      assert.equal(question.target, count);
      assert.equal(question.choices.length, 4);
      assert.equal(new Set(question.choices).size, 4);
      assert.ok(question.choices.every(Number.isInteger));
      assert.ok(question.choices.includes(count));
    } else {
      assert.equal(question.target, getCorrectAction(question.situation.player, question.situation.dealer, question.situation.options));
      assert.ok(question.choices.includes(question.target));
    }
  }
  const counting = createQuickQuestions(32, 'counting');
  assert.equal(new Set(counting.flatMap(question => question.cards.map(card => card.id))).size, 400);
  const wrong = counting[0].choices.find(choice => choice !== counting[0].target);
  assert.equal(gradeQuickAnswer(counting[0], wrong, 0, 100).correct, false);
  assert.equal(counting[1].countBefore, counting[0].target, 'wrong answers do not propagate into subsequent targets');
  assert.ok(createQuickQuestions(32, 'strategy').every(question => question.kind === 'strategy'));
  for (const count of [0, 201, -1, NaN, 1.5]) assert.throws(() => createQuickQuestions(0, 'mixed', count));
  for (const seed of [-1, 0x100000000, 1.5, NaN]) assert.throws(() => createQuickQuestions(seed, 'mixed'));
});

test('only a first, valid, in-order answer can earn transparent capped combo points', () => {
  const questions = createQuickQuestions(5, 'counting');
  let answers = [];
  for (let index = 0; index < 16; index++) {
    answers = recordQuickAnswer(questions, answers, index, questions[index].target, index * 100);
    const answer = answers.at(-1);
    assert.equal(answer.baseXp, 10);
    assert.equal(answer.bonusXp, 2 * Math.min(5, Math.floor((index + 1) / 3)));
    assert.ok(answer.xp <= 20);
  }
  const wrong = questions[16].choices.find(choice => choice !== questions[16].target);
  answers = recordQuickAnswer(questions, answers, 16, wrong, 2000);
  assert.equal(answers.at(-1).xp, 0);
  assert.equal(answers.at(-1).combo, 0);
  answers = recordQuickAnswer(questions, answers, 17, questions[17].target, 2100);
  assert.equal(answers.at(-1).xp, 10);
  assert.equal(summarizeQuickAnswers(answers).attempts, 18);
  assert.equal(summarizeQuickAnswers(answers).correct, 17);
  assert.equal(summarizeQuickAnswers(answers).bestCombo, 16);
  assert.throws(() => recordQuickAnswer(questions, answers, 17, questions[17].target, 2200), /unanswered/);
  assert.throws(() => recordQuickAnswer(questions, answers, 19, questions[19].target, 2200), /unanswered/);
  assert.throws(() => recordQuickAnswer(questions, answers, 18, questions[18].target, 2000), /order/);
  assert.throws(() => recordQuickAnswer(questions, answers, 18, 'NaN', 2200), /available/);
  for (const time of [-1, QUICK_DURATION_MS, Infinity, NaN]) assert.throws(() => recordQuickAnswer(questions, answers, 18, questions[18].target, time));
  assert.equal(answers.length, 18, 'rejected or repeated submissions do not change the original scores');
  const farWrong = recordQuickAnswer(questions, answers, 18, 999, 2200).at(-1);
  assert.equal(farWrong.correct, false, 'a valid typed count outside the suggested choices consumes the first attempt');
  assert.equal(farWrong.xp, 0);
  assert.equal(farWrong.combo, 0);
  for (const value of [1001, -1001, 1.5, Infinity, '1', NaN]) assert.throws(() => recordQuickAnswer(questions, answers, 18, value, 2200));
});

test('timer results use active answering time and qualify only substantive completed practice', () => {
  const result = makeResult();
  assert.equal(quickRunCreditsDay(result), true);
  assert.equal(quickRunCreditsDay(makeResult(1, '2026-09-26', 0)), false);
  assert.equal(quickRunCreditsDay(makeResult(1, '2026-09-26', 4)), false);
  assert.equal(quickRunCreditsDay(makeResult(1, '2026-09-26', 5, 'ended')), false);
  assert.equal(quickRunCreditsDay(makeResult(1, '2026-09-26', 5, 'interrupted')), false);
  assert.equal(quickRunCreditsDay(makeResult(1, '2026-09-26', 5, 'timer', { interrupted: true })), true, 'pausing and resuming is allowed in practice');
  assert.throws(() => createQuickResult({ ...result, elapsedMs: 59_999 }));
  assert.throws(() => createQuickResult({ ...result, elapsedMs: 0, endReason: 'ended' }), /timing/);
  assert.throws(() => createQuickResult({ ...result, endReason: 'question-limit' }));
  assert.equal(quickRunCreditsDay(makeResult(2, '2026-09-26', 200, 'question-limit')), true);
  const longerWallClock = createQuickResult({ ...result, endedAt: '2026-09-26T12:30:00.000Z' });
  assert.equal(longerWallClock.elapsedMs, QUICK_DURATION_MS, 'feedback/hidden pauses may add wall time without consuming answering time');
  const forged = structuredClone(result);
  forged.answers[0].xp = 5000;
  forged.answers[0].correct = false;
  const canonical = createQuickResult(forged);
  assert.equal(canonical.answers[0].xp, 10);
  assert.equal(canonical.answers[0].correct, true);
});

test('local-day streak preserves yesterday until the day is missed, handles leap days, and deduplicates runs', () => {
  let history = addQuickResult(EMPTY_QUICK_HISTORY, makeResult(1, '2026-09-24'));
  history = addQuickResult(history, makeResult(2, '2026-09-25'));
  assert.equal(quickHistoryStats(history, '2026-09-25').streak, 2);
  assert.equal(quickHistoryStats(history, '2026-09-26').streak, 2);
  assert.equal(quickHistoryStats(history, '2026-09-27').streak, 0);
  const today = makeResult(3, '2026-09-26');
  history = addQuickResult(history, today);
  history = addQuickResult(history, today);
  assert.equal(history.sessions.length, 3);
  assert.equal(quickHistoryStats(history, '2026-09-26').streak, 3);
  assert.equal(history.bestXp, summarizeQuickAnswers(today.answers).xp);
  history = addQuickResult(history, makeResult(4, '2026-09-26', 16, 'ended'));
  assert.ok(history.bestXp > summarizeQuickAnswers(today.answers).xp, 'an early end retains earned points without creating a new practice day');
  assert.equal(history.activityDays.length, 3);
  let leapHistory = addQuickResult(EMPTY_QUICK_HISTORY, makeResult(11, '2024-02-28'));
  leapHistory = addQuickResult(leapHistory, makeResult(12, '2024-02-29'));
  leapHistory = addQuickResult(leapHistory, makeResult(13, '2024-03-01'));
  assert.equal(quickHistoryStats(leapHistory, '2024-03-01').streak, 3);
});

test('daily history survives many same-day runs and archived long streaks stay exact', () => {
  let history = addQuickResult(EMPTY_QUICK_HISTORY, makeResult(1, '2026-09-25'));
  for (let seed = 2; seed < 27; seed++) history = addQuickResult(history, makeResult(seed, '2026-09-26'));
  assert.equal(history.sessions.length, 20);
  assert.deepEqual(history.activityDays, ['2026-09-25', '2026-09-26']);
  assert.equal(quickHistoryStats(history, '2026-09-26').streak, 2);
  let longHistory = { ...EMPTY_QUICK_HISTORY, activityDays: [] };
  for (let index = 0; index < 370; index++) {
    const day = new Date(Date.UTC(2025, 0, 1 + index)).toISOString().slice(0, 10);
    longHistory = addQuickResult(longHistory, makeResult(index + 1, day));
  }
  assert.equal(longHistory.activityDays.length, 366);
  assert.equal(longHistory.streakCarry, 4);
  assert.equal(quickHistoryStats(longHistory, longHistory.activityDays.at(-1)).streak, 370);
  assert.deepEqual(readQuickHistory(JSON.stringify(longHistory)), { history: longHistory, reset: false });
});

test('date keys use the completion calendar date and survive DST and later timezone changes', () => {
  const original = process.env.TZ;
  try {
    process.env.TZ = 'America/New_York';
    assert.equal(localDateKey(new Date('2026-03-09T03:30:00.000Z')), '2026-03-08');
    let history = addQuickResult(EMPTY_QUICK_HISTORY, makeResult(1, '2026-03-07'));
    history = addQuickResult(history, makeResult(2, '2026-03-08'));
    history = addQuickResult(history, makeResult(3, '2026-03-09'));
    assert.equal(quickHistoryStats(history, '2026-03-09').streak, 3);
    const midnight = createQuickResult({ ...makeResult(4), startedAt: '2026-09-27T03:59:30.000Z', endedAt: '2026-09-27T04:01:00.000Z' });
    assert.equal(midnight.localDate, '2026-09-27');
    const nearMidnight = createQuickResult({ ...makeResult(5), startedAt: '2026-09-27T02:59:30.000Z', endedAt: '2026-09-27T03:01:00.000Z' });
    assert.equal(nearMidnight.localDate, '2026-09-26');
    const saved = JSON.stringify(addQuickResult(EMPTY_QUICK_HISTORY, nearMidnight));
    process.env.TZ = 'Asia/Tokyo';
    const restored = readQuickHistory(saved);
    assert.equal(restored.reset, false);
    assert.equal(restored.history.sessions[0].localDate, '2026-09-26', 'historical local dates are not reinterpreted in a new timezone');
  } finally {
    if (original === undefined) delete process.env.TZ; else process.env.TZ = original;
  }
});

test('storage parser resets malformed/oversized records and rejects invalid timing, duplicates and dates', () => {
  assert.deepEqual(readQuickHistory(null), { history: EMPTY_QUICK_HISTORY, reset: false });
  const history = addQuickResult(EMPTY_QUICK_HISTORY, makeResult());
  assert.deepEqual(readQuickHistory(JSON.stringify(history)), { history, reset: false });
  for (const raw of ['', '{', 'null', '[]', '{}', 'x'.repeat(1_000_001)]) assert.equal(readQuickHistory(raw).reset, true);
  const changes = [
    { version: 2 }, { bestXp: Infinity }, { bestXp: -1 }, { bestXp: 99999 },
    { activityDays: ['2026-02-30'] }, { activityDays: ['2026-09-26', '2026-09-26'] },
    { activityDays: [] }, { sessions: [...history.sessions, ...history.sessions] }, { streakCarry: 99 },
  ];
  for (const change of changes) assert.equal(readQuickHistory(JSON.stringify({ ...history, ...change })).reset, true, JSON.stringify(change));
  const duplicateAnswers = structuredClone(history);
  duplicateAnswers.sessions[0].answers[1].index = 0;
  assert.equal(readQuickHistory(JSON.stringify(duplicateAnswers)).reset, true);
  const expired = structuredClone(history);
  expired.sessions[0].answers[0].elapsedMs = QUICK_DURATION_MS;
  assert.equal(readQuickHistory(JSON.stringify(expired)).reset, true);
  const cachedScore = structuredClone(history);
  cachedScore.sessions[0].answers[0].xp = 9999;
  cachedScore.sessions[0].answers[0].correct = false;
  assert.deepEqual(readQuickHistory(JSON.stringify(cachedScore)), { history, reset: false });
});
