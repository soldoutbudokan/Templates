const test = require('node:test');
const assert = require('node:assert/strict');
const { getCorrectAction } = require('../lib/basicStrategy.ts');
const { estimateDecks, toTrueCount } = require('../lib/countPolicy.ts');
const { createTableSession, advanceTable, applyTableAction, legalActions, submitTableCount,
  finishTableSession, markTableInterrupted } = require('../lib/blackjack.ts');
const { readTableHistory, tableResultLabel, cleanTableTest } = require('../lib/tableHistory.ts');

const CONFIG = { mode: 'test', roundLimit: 10, otherPlayers: 2, speedMs: 900 };
const ENDED_AT = '2026-09-26T12:00:00.000Z';
const encode = sessions => JSON.stringify({ version: 1, sessions });
const result = state => ({ ...finishTableSession(state), endedAt: ENDED_AT });

function correctAction(state) {
  if (state.phase === 'insurance') return 'decline-insurance';
  const legal = legalActions(state);
  const table = state.table;
  return getCorrectAction(table.seats[table.activeSeat].hands[table.activeHand].cards, table.dealer.cards[0], {
    canDouble: legal.includes('double'), canSplit: legal.includes('split'), canSurrender: legal.includes('surrender'),
  });
}
function complete(seed, config = CONFIG, mistakes = false) {
  let state = createTableSession(config, seed);
  let madeMistake = false;
  for (let steps = 0; !state.ended && steps < 2000; steps++) {
    if (state.phase === 'insurance' || state.phase === 'player-turn') {
      const expected = correctAction(state);
      const wrong = legalActions(state).find(action => action !== expected);
      const action = mistakes && !madeMistake && wrong ? wrong : expected;
      if (action !== expected) madeMistake = true;
      state = applyTableAction(state, action);
    } else if (state.phase === 'count-checkpoint') {
      const running = state.exposures.at(-1)?.runningCount ?? 0;
      state = submitTableCount(state, mistakes ? running + 1 : running,
        toTrueCount(running, estimateDecks(state.remaining)));
    } else state = advanceTable(state);
  }
  assert.equal(state.ended, true, 'fixture must terminate');
  return result(state);
}

test('table history deterministically restores completed tests and practice', () => {
  for (const config of [CONFIG, { ...CONFIG, mode: 'practice', otherPlayers: 0 }, { ...CONFIG, roundLimit: 0 }]) {
    const original = complete(42, config);
    assert.deepEqual(readTableHistory(encode([original])), [original]);
  }
});

test('partial rounds preserve their visible exposure boundary without inventing answers', () => {
  let state = createTableSession(CONFIG, 7);
  for (let step = 0; step < 5; step++) state = advanceTable(state);
  const partial = result(markTableInterrupted(state));
  const [restored] = readTableHistory(encode([partial]));
  assert.deepEqual(restored, partial);
  assert.equal(restored.completed, false);
  assert.equal(restored.interrupted, true);
  assert.equal(restored.rounds, 0);
  assert.equal(restored.decisions.length, 0);
  assert.equal(restored.exposures.length, 5);
});

test('a reported lost count and interruption survive replay as incorrect observations', () => {
  let state = createTableSession(CONFIG, 8);
  for (let step = 0; state.phase !== 'count-checkpoint' && step < 2000; step++) {
    state = state.phase === 'player-turn' || state.phase === 'insurance'
      ? applyTableAction(state, correctAction(state)) : advanceTable(state);
  }
  state = markTableInterrupted(submitTableCount(state, null, null));
  const original = result(state);
  const [restored] = readTableHistory(encode([original]));
  assert.deepEqual(restored, original);
  assert.equal(restored.checkpoints[0].runningCorrect, false);
  assert.equal(restored.checkpoints[0].trueCorrect, false);
  assert.equal(restored.interrupted, true);
  assert.equal(cleanTableTest(restored), false);
});

test('persisted cards, targets and grades are regenerated from the original inputs', () => {
  const original = complete(6, CONFIG, true);
  assert.ok(original.decisions.some(decision => !decision.correct));
  const forged = structuredClone(original);
  for (const decision of forged.decisions) {
    decision.correct = true;
    decision.expectedAction = decision.action;
    decision.runningCount = 999;
    decision.playerCards = [];
    decision.dealerUpcard = null;
    decision.legalActions = [];
  }
  for (const checkpoint of forged.checkpoints) {
    checkpoint.runningCorrect = true;
    checkpoint.trueCorrect = true;
    checkpoint.arithmeticCorrect = true;
    checkpoint.runningTarget = 999;
    checkpoint.trueTarget = 999;
    checkpoint.decksRemaining = 999;
  }
  forged.exposures = forged.exposures.map(() => ({ card: null, runningCount: 999 }));
  assert.deepEqual(readTableHistory(encode([forged])), [original]);
  assert.equal(cleanTableTest(readTableHistory(encode([forged]))[0]), false);
});

test('invalid records are isolated; illegal actions and inconsistent completion are rejected', () => {
  const good = complete(12);
  const badAction = structuredClone(good);
  badAction.decisions[0].action = 'not-an-action';
  const badRound = structuredClone(good);
  badRound.decisions[0].round += 1;
  const badCompletion = { ...good, completed: false };
  const badExposure = { ...good, exposures: [] };
  const badGuess = structuredClone(good);
  badGuess.checkpoints[0].runningGuess = '3';
  const badId = { ...good, id: 'another-session' };
  const badDate = { ...good, endedAt: 'not a date' };
  assert.deepEqual(readTableHistory(encode([
    null, badAction, badRound, badCompletion, badExposure, badGuess, badId, badDate, good,
  ])), [good]);
});

test('history validates configuration and bounds records, decisions, exposures and guesses', () => {
  const original = complete(19);
  const invalid = [
    { ...original, seed: -1 },
    { ...original, config: { ...original.config, mode: 'exam' } },
    { ...original, config: { ...original.config, roundLimit: 999 } },
    { ...original, config: { ...original.config, otherPlayers: 1 } },
    { ...original, config: { ...original.config, speedMs: 0 } },
    { ...original, decisions: Array(513).fill(original.decisions[0]) },
    { ...original, exposures: Array(313).fill(original.exposures[0]) },
    { ...original, checkpoints: [{ ...original.checkpoints[0], trueGuess: 1001 }], rounds: 1 },
  ];
  assert.deepEqual(readTableHistory(encode(invalid)), []);
  const attempts = Array.from({ length: 25 }, (_, index) => ({ ...original,
    endedAt: `2026-09-${String(index + 1).padStart(2, '0')}T12:00:00.000Z` }));
  assert.deepEqual(readTableHistory(encode(attempts)), attempts.slice(-24));
  assert.throws(() => readTableHistory(encode(Array(129).fill(original))));
  for (const raw of ['{', 'null', '{}', '{"version":2,"sessions":[]}', '{"version":1,"sessions":{}}']) {
    assert.throws(() => readTableHistory(raw));
  }
  assert.deepEqual(readTableHistory(null), []);
});

test('clean test requires complete uninterrupted counting and actual playing decisions', () => {
  const clean = complete(42);
  assert.equal(cleanTableTest(clean), true);
  for (const changes of [
    { completed: false }, { interrupted: true }, { config: { ...CONFIG, mode: 'practice' } },
    { decisions: [] }, { checkpoints: [] },
    { decisions: clean.decisions.map(decision => ({ ...decision, kind: 'insurance' })) },
    { decisions: clean.decisions.map(decision => ({ ...decision, correct: false })) },
    { checkpoints: clean.checkpoints.map(checkpoint => ({ ...checkpoint, runningCorrect: false })) },
    { checkpoints: clean.checkpoints.map(checkpoint => ({ ...checkpoint, trueCorrect: false })) },
  ]) assert.equal(cleanTableTest({ ...clean, ...changes }), false);
  assert.equal(tableResultLabel(clean), 'Uninterrupted full test');
  assert.equal(tableResultLabel({ ...clean, completed: false }), 'Partial full test');
  assert.equal(tableResultLabel({ ...clean, interrupted: true }), 'Interrupted full test · practice only');
  assert.equal(tableResultLabel({ ...clean, config: { ...CONFIG, mode: 'practice' } }), 'Table practice');
});
