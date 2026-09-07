const test = require('node:test');
const assert = require('node:assert/strict');
const { Deck, getCardValue } = require('../lib/deck.ts');
const { parseCount, toTrueCount, estimateDecks } = require('../lib/countPolicy.ts');
const { getCorrectAction, getHandTotal } = require('../lib/basicStrategy.ts');
const { createTrainingPlan, gradeCheckpoint, replayCounts, readHistory, recommendation,
  assessmentEligible, sessionLabel, seededRandom } = require('../lib/training.ts');

function cards(values) { return values.map((value, id) => ({ value, suit: '♠', id: String(id) })); }
function resultFor(plan, overrides = {}) {
  return { ...plan, endedAt: '2026-09-07T12:00:00.000Z', completed: true, interrupted: false, assisted: false,
    answers: plan.checkpoints.map((c, index) => gradeCheckpoint(plan, index, c.runningCount, plan.focus === 'conversion' ? c.trueCount : null, 1000)), ...overrides };
}

test('six-deck shoe has 312 distinct physical cards and balanced Hi-Lo values', () => {
  const deck = new Deck(6, .75, seededRandom(10));
  const drawn = deck.deal(312);
  assert.equal(new Set(drawn.map(card => card.id)).size, 312);
  for (const rank of ['2','3','4','5','6','7','8','9','10','J','Q','K','A']) assert.equal(drawn.filter(card => card.value === rank).length, 24);
  assert.equal(deck.runningCount(), 0);
  assert.equal(deck.remaining(), 0);
});

test('drawing a hidden card does not change RC; exposing it twice counts it once', () => {
  const deck = new Deck(1, .75, () => 0);
  const hidden = deck.draw(52);
  assert.equal(deck.runningCount(), 0);
  assert.equal(deck.remaining(), 0);
  const two = hidden.find(card => card.value === '2');
  const king = hidden.find(card => card.value === 'K');
  const ace = hidden.find(card => card.value === 'A');
  deck.expose([two]); assert.equal(deck.runningCount(), 1);
  deck.expose([ace]); assert.equal(deck.runningCount(), 0);
  deck.expose([king]); assert.equal(deck.runningCount(), -1);
  deck.expose([king]); assert.equal(deck.runningCount(), -1);
  const oldId = hidden[0].id;
  deck.shuffle(); assert.equal(deck.runningCount(), 0);
  const next = deck.draw(52); assert.equal(next.some(card => card.id === oldId), false);
  assert.throws(() => deck.expose([king]));
});

test('oversized draws reject atomically; cut-card reshuffle is explicit at a round boundary', () => {
  const deck = new Deck(6);
  deck.deal(233); assert.equal(deck.needsReshuffle(), false);
  deck.deal(1); assert.equal(deck.needsReshuffle(), true);
  assert.equal(deck.remaining(), 78);
  const count = deck.runningCount();
  assert.throws(() => deck.deal(79), RangeError);
  assert.equal(deck.remaining(), 78); assert.equal(deck.runningCount(), count);
  assert.equal(deck.prepareRound(10), true);
  assert.equal(deck.remaining(), 312); assert.equal(deck.runningCount(), 0);
  assert.throws(() => deck.prepareRound(313), RangeError);
});

test('shoe subscribers see draws and explicit shuffles', () => {
  const deck = new Deck(1);
  const seen = [];
  const unsubscribe = deck.subscribe(() => seen.push(deck.remainingSnapshot()));
  deck.deal(3); deck.shuffle(); unsubscribe(); deck.deal(1);
  assert.deepEqual(seen, [49, 52]);
});

test('input grading rejects partial strings, decimals, scientific notation and nonfinite values', () => {
  for (const value of ['2.7', '', '2x', '1e2', 'NaN', 'Infinity', '-', '1001']) assert.equal(parseCount(value), null);
  for (const value of ['2', '+2', ' 2 ']) assert.equal(parseCount(value), 2);
  assert.equal(parseCount('-2'), -2);
});

test('one declared TC policy handles positive and negative boundaries exactly', () => {
  for (const [rc, decks, expected] of [[7,4,1],[6,3,2],[5,3,1],[-1,6,-1],[-3,2,-2],[-4,2,-2],[0,2.5,0]]) assert.equal(toTrueCount(rc,decks),expected);
  for (const divisor of [0,-1,Infinity,NaN]) assert.throws(() => toTrueCount(1,divisor));
  assert.equal(estimateDecks(130), 2.5);
  assert.equal(estimateDecks(1), .5);
  assert.throws(() => estimateDecks(0));
});

// Independent fixtures from Shackleford's 4–8-deck strategy and H17 amendments.
test('six-deck H17, DAS, late-surrender strategy matches independent fixtures', () => {
  const fixtures = [
    [['10','5'],'A','surrender'],[['10','7'],'A','surrender'],[['8','8'],'A','surrender'],
    [['8','8'],'10','split'],[['10','6'],'9','surrender'],[['6','5'],'A','double'],
    [['A','7'],'2','double'],[['A','8'],'6','double'],[['A','6'],'2','hit'],
    [['2','2'],'2','split'],[['4','4'],'5','split'],[['9','9'],'7','stand'],
    [['9','9'],'9','split'],[['10','2'],'3','hit'],[['10','2'],'4','stand']
  ];
  for (const [hand, dealer, expected] of fixtures) assert.equal(getCorrectAction(cards(hand),cards([dealer])[0]),expected,`${hand} vs ${dealer}`);
});

test('strategy respects unavailable actions and three-card hands', () => {
  for (const [hand,dealer,expected,options] of [
    [['2','3','6'],'6','hit',{}], [['A','2','5'],'2','stand',{}], [['10','2','4'],'10','hit',{}],
    [['A','7'],'2','stand',{canDouble:false}], [['A','8'],'6','stand',{canDouble:false}],
    [['8','8'],'A','split',{canSurrender:false}], [['10','7'],'A','stand',{canSurrender:false}],
    [['A','A'],'4','hit',{canDouble:false,canSplit:false,canSurrender:false}]
  ]) assert.equal(getCorrectAction(cards(hand),cards([dealer])[0],options),expected,`${hand} vs ${dealer}`);
  assert.deepEqual(getHandTotal(cards(['A','A','9'])), {total:21,soft:true});
});

test('seeded sessions preserve cumulative RC, expose each card once and stop before a balanced endpoint', () => {
  for (const mode of ['practice','assessment']) {
    const plan = createTrainingPlan(42,mode,'conversion',900);
    assert.deepEqual(plan,createTrainingPlan(42,mode,'conversion',900));
    const all = plan.checkpoints.flatMap(checkpoint => checkpoint.cards);
    assert.equal(all.length, mode === 'practice' ? 78 : 234);
    assert.equal(new Set(all.map(card => card.id)).size,all.length);
    let count = 0;
    for (const checkpoint of plan.checkpoints) {
      assert.equal(checkpoint.countBefore,count);
      count += checkpoint.cards.reduce((sum,card) => sum + getCardValue(card),0);
      assert.equal(checkpoint.runningCount,count);
      assert.equal(replayCounts(checkpoint).at(-1),count);
    }
    assert.notDeepEqual(all,createTrainingPlan(43,mode,'conversion',900).checkpoints.flatMap(c => c.cards));
  }
});

test('wrong count and correct arithmetic are diagnosed independently; there is no ±1 tolerance', () => {
  const plan = createTrainingPlan(9,'practice','conversion',900);
  const checkpoint = plan.checkpoints[0];
  const wrong = checkpoint.runningCount + 3;
  const answer = gradeCheckpoint(plan,0,wrong,toTrueCount(wrong,checkpoint.decksRemaining),200);
  assert.equal(answer.runningCorrect,false); assert.equal(answer.arithmeticCorrect,true);
  assert.equal(gradeCheckpoint(plan,0,checkpoint.runningCount,checkpoint.trueCount+1,200).trueCorrect,false);
  assert.equal(gradeCheckpoint(plan,0,null,null,200).lost,true);
});

test('history regenerates targets, recomputes grades and rejects corrupt records', () => {
  const result = resultFor(createTrainingPlan(11,'practice','conversion',900));
  const saved = JSON.stringify({version:1,sessions:[result]});
  assert.deepEqual(readHistory(saved),[result]);
  const forged = structuredClone(result); forged.answers[0].runningGuess += 1; forged.answers[0].runningCorrect = true;
  assert.equal(readHistory(JSON.stringify({version:1,sessions:[forged]}))[0].answers[0].runningCorrect,false);
  for (const raw of ['null','{}','{','{"version":1,"sessions":[null]}']) assert.throws(() => readHistory(raw));
  assert.throws(() => readHistory(JSON.stringify({version:1,sessions:[{...result,answers:[]}]})));
});

test('partial, interrupted and assisted assessments cannot become eligible', () => {
  const result = resultFor(createTrainingPlan(11,'assessment','running',900));
  assert.equal(assessmentEligible(result),true);
  for (const changes of [{completed:false},{interrupted:true},{assisted:true}]) assert.equal(assessmentEligible({...result,...changes}),false);
  assert.equal(sessionLabel({...result,completed:false}),'Partial assessment');
  assert.equal(sessionLabel({...result,interrupted:true}),'Interrupted assessment · practice only');
});

test('recommendations use completed observations and keep weak counting ahead of conversion', () => {
  assert.equal(recommendation([]).focus,'running');
  const clean = resultFor(createTrainingPlan(6,'assessment','running',900));
  assert.equal(recommendation([clean]).focus,'conversion');
  assert.equal(recommendation([{...clean,completed:false}]).focus,'running');
  assert.equal(recommendation([{...clean,interrupted:true}]).focus,'running');
  const weak = {...clean,answers:clean.answers.map(a => ({...a,runningCorrect:false}))};
  assert.equal(recommendation([weak]).focus,'running');
});
