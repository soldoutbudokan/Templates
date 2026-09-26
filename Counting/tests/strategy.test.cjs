const test = require('node:test');
const assert = require('node:assert/strict');
const { getCorrectAction, getHandTotal } = require('../lib/basicStrategy.ts');
const { createStrategyCases, gradeStrategyAnswer, legalStrategyActions, strategyCriterionMet,
  strategyBreakdown, makeStrategyCase, strategyRows } = require('../lib/strategyPractice.ts');

const action = { H: 'hit', S: 'stand', D: 'double', P: 'split', R: 'surrender' };
const dealers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'];
function cards(values) { return values.map((value, index) => ({ value, suit: '♠', id: String(index) })); }

// Independent fixtures transcribed from Michael Shackleford's 4–8-deck strategy,
// with the published H17 changes and DAS / late surrender enabled.
// https://wizardofodds.com/games/blackjack/strategy/4-decks/
const charts = {
  hard: [
    'HHHHHHHHHH', 'HHHHHHHHHH', 'HHHHHHHHHH', 'HHHHHHHHHH', 'HDDDDHHHHH',
    'DDDDDDDDHH', 'DDDDDDDDDD', 'HHSSSHHHHH', 'SSSSSHHHHH', 'SSSSSHHHHH',
    'SSSSSHHHRR', 'SSSSSHHRRR', 'SSSSSSSSSR', 'SSSSSSSSSS', 'SSSSSSSSSS', 'SSSSSSSSSS',
  ],
  soft: ['HHHDDHHHHH', 'HHHDDHHHHH', 'HHDDDHHHHH', 'HHDDDHHHHH', 'HDDDDHHHHH', 'DDDDDSSHHH', 'SSSSDSSSSS', 'SSSSSSSSSS'],
  pairs: ['PPPPPPHHHH', 'PPPPPPHHHH', 'HHHPPHHHHH', 'DDDDDDDDHH', 'PPPPPHHHHH', 'PPPPPPHHHH', 'PPPPPPPPPR', 'PPPPPSPPSS', 'SSSSSSSSSS', 'PPPPPPPPPP'],
};

test('all 340 declared chart cells match independently transcribed H17 / DAS / surrender fixtures', () => {
  let checked = 0;
  for (const category of ['hard', 'soft', 'pairs']) {
    strategyRows(category).forEach((row, rowIndex) => {
      dealers.forEach((dealer, col) => {
        const situation = makeStrategyCase(category, row, dealer);
        const expected = action[charts[category][rowIndex][col]];
        assert.equal(getCorrectAction(situation.player, situation.dealer, situation.options), expected, `${row.label} vs ${dealer}`);
        checked++;
      });
    });
  }
  assert.equal(checked, 340);
});

test('unavailable splits, doubles, and surrender use the legal total-dependent fallback', () => {
  const fixtures = [
    [['8', '8'], '9', { canSplit: false }, 'surrender'],
    [['8', '8'], '10', { canSplit: false }, 'surrender'],
    [['8', '8'], 'A', { canSplit: false, canSurrender: false }, 'hit'],
    [['8', '8'], '10', { canSplit: false, canSurrender: false }, 'hit'],
    [['8', '8'], '6', { canSplit: false }, 'stand'],
    [['A', 'A'], '6', { canSplit: false }, 'double'],
    [['A', 'A'], '6', { canSplit: false, canDouble: false }, 'hit'],
    [['A', 'A'], '5', { canSplit: false }, 'hit'],
    [['A', '7'], '2', { canDouble: false }, 'stand'],
    [['A', '8'], '6', { canDouble: false }, 'stand'],
    [['A', '6'], '6', { canDouble: false }, 'hit'],
    [['5', '5'], '6', { canDouble: false }, 'hit'],
    [['10', '7'], 'A', { canSurrender: false }, 'stand'],
    [['10', '5'], 'A', { canSurrender: false }, 'hit'],
    [['8', '8'], 'A', { canSurrender: false }, 'split'],
    [['2', '3', '6'], 'A', {}, 'hit'],
    [['A', '2', '5'], '2', {}, 'stand'],
    [['10', '2', '4'], 'A', {}, 'hit'],
  ];
  for (const [hand, dealer, options, expected] of fixtures) {
    assert.equal(getCorrectAction(cards(hand), cards([dealer])[0], options), expected, `${hand} vs ${dealer}: ${JSON.stringify(options)}`);
  }
});

test('strategy sessions distribute category and row coverage without repeated cells', () => {
  for (let seed = 0; seed < 20; seed++) {
    for (const length of [20, 30]) {
      for (const focus of ['mixed', 'hard', 'soft', 'pairs']) {
        const cases = createStrategyCases(seed, focus, length);
        assert.equal(cases.length, length);
        assert.equal(new Set(cases.map(situation => situation.id)).size, length);
        assert.deepEqual(cases, createStrategyCases(seed, focus, length));
        const categories = focus === 'mixed' ? ['hard', 'soft', 'pairs'] : [focus];
        const counts = categories.map(category => cases.filter(situation => situation.category === category).length);
        assert.ok(Math.max(...counts) - Math.min(...counts) <= 1);
        for (const category of categories) {
          const rows = strategyRows(category);
          const sampled = cases.filter(situation => situation.category === category);
          const rowCounts = rows.map(row => sampled.filter(situation => situation.row === row.label).length);
          assert.ok(Math.max(...rowCounts) - Math.min(...rowCounts) <= 1);
        }
        for (const situation of cases) {
          assert.ok(legalStrategyActions(situation).includes(getCorrectAction(situation.player, situation.dealer, situation.options)));
          assert.ok(getHandTotal(situation.player).total < 21);
        }
      }
    }
  }
  assert.notDeepEqual(createStrategyCases(1, 'mixed', 30), createStrategyCases(2, 'mixed', 30));
  assert.throws(() => createStrategyCases(1, 'other', 20));
  assert.throws(() => createStrategyCases(1, 'mixed', 21));
});

test('test criterion requires every recorded first answer, full length, and no interruption', () => {
  const cases = createStrategyCases(42, 'mixed', 30);
  const answers = cases.map(situation => gradeStrategyAnswer(situation, getCorrectAction(situation.player, situation.dealer, situation.options), 1000));
  const result = { mode: 'test', focus: 'mixed', cases, answers, completed: true, interrupted: false };
  assert.equal(strategyCriterionMet(result), true);
  for (const changes of [{ interrupted: true }, { completed: false }, { mode: 'practice' }, { answers: answers.slice(1) }, { cases: cases.slice(1) }]) {
    assert.equal(strategyCriterionMet({ ...result, ...changes }), false);
  }
  const wrongAction = legalStrategyActions(cases[0]).find(choice => choice !== answers[0].action);
  const wrong = { ...answers[0], action: wrongAction, correct: true };
  assert.equal(strategyCriterionMet({ ...result, answers: [wrong, ...answers.slice(1)] }), false, 'cached correctness cannot forge a passing result');
  assert.equal(strategyCriterionMet({ ...result, answers: [answers[1], ...answers.slice(1)] }), false, 'an answer cannot stand in for a different situation');
  const breakdown = strategyBreakdown(result);
  assert.deepEqual(breakdown.map(row => [row.correct, row.attempted]), [[10, 10], [10, 10], [10, 10]]);
  assert.equal(gradeStrategyAnswer(cases[0], wrongAction, 1000).correct, false);
});

test('illegal buttons cannot submit a decision and after-hit hands expose only hit or stand', () => {
  const initial = makeStrategyCase('hard', { label: 'Hard 12', values: ['10', '2'] }, '3');
  assert.throws(() => gradeStrategyAnswer(initial, 'split', 200), RangeError);
  const afterHit = makeStrategyCase('hard', { label: 'Hard 16', values: ['10', '2', '4'] }, 'A');
  assert.deepEqual(legalStrategyActions(afterHit), ['hit', 'stand']);
  for (const action of ['double', 'split', 'surrender']) assert.throws(() => gradeStrategyAnswer(afterHit, action, 200), RangeError);
});
