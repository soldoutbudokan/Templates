const test = require('node:test');
const assert = require('node:assert/strict');
const {
  CHALLENGE_PRESETS, createChallenge, advanceChallenge, applyChallengeAction,
  submitChallengeBet, submitChallengeCount, continueChallenge, markChallengeInterrupted,
  summarizeChallenge, finishChallenge, readChallengeResult, validateChallengeHistory,
  recommendChallenge, practiceBetUnits, chooseChallengeSeed,
} = require('../lib/challenge.ts');
const { legalActions } = require('../lib/blackjack.ts');
const { getCorrectAction } = require('../lib/basicStrategy.ts');
const { getCardValue } = require('../lib/deck.ts');
const { estimateDecks, toTrueCount } = require('../lib/countPolicy.ts');
const START = '2026-01-01T00:00:00.000Z';
const END = '2026-01-01T00:03:00.000Z';
function create(preset='rhythm', seed=44, extra={}) {
  return createChallenge(preset, seed, { id: `challenge-test-${preset}-${seed}`, startedAt: START, ...extra });
}
function expected(state) {
  const table = state.table;
  if (table.phase === 'insurance') return 'decline-insurance';
  const view = table.table;
  const h = view.seats[view.activeSeat].hands[view.activeHand];
  const legal = legalActions(table);
  return getCorrectAction(h.cards, view.dealer.cards[0], { canDouble: legal.includes('double'), canSplit: legal.includes('split'), canSurrender: legal.includes('surrender') });
}
function count(state) { return state.table.exposures.reduce((sum, event) => sum + getCardValue(event.card), 0); }
function tc(state) { return toTrueCount(count(state), estimateDecks(state.table.remaining)); }
function step(state, wrongCounts=false, betOverride) {
  if (state.phase === 'bet') return submitChallengeBet(state, betOverride ?? practiceBetUnits(tc(state)));
  if (state.phase === 'count-checkpoint') return submitChallengeCount(state, count(state)+(wrongCounts?1:0), tc(state)+(wrongCounts?1:0));
  if (state.phase === 'feedback') return continueChallenge(state);
  if (!CHALLENGE_PRESETS[state.presetId].autoPlay && ['player-turn','insurance'].includes(state.phase)) return applyChallengeAction(state, expected(state));
  return advanceChallenge(state);
}
function until(state, predicate, wrongCounts=false, betOverride) {
  let guard = 1200;
  while (!predicate(state) && state.phase !== 'complete' && guard-- > 0) state = step(state, wrongCounts, betOverride);
  assert(guard > 0, 'Challenge failed to progress');
  assert(predicate(state), `Unexpected stop in ${state.phase}`);
  return state;
}
function play(state, wrong=false, betOverride) { return until(state, s=>s.phase==='complete', wrong, betOverride); }
function result(preset='rhythm', seed=44, extra={}) { return finishChallenge(play(create(preset,seed,extra)),END); }
function rig(values, preset='table') {
  const state=create(preset);
  state.table._engine.shoe = [ ...values.map((value,i)=>({value,suit:'♠',id:`fixture-${i}`})), ...state.table._engine.shoe.slice(values.length) ];
  return state;
}

test('six rounds finish in one shoe, with count spacing and manual grading per preset',()=>{
  for (const preset of ['rhythm','table','pace']) {
    const s=play(create(preset));
    assert(s.completed); assert.equal(s.table.round,6); assert.equal(s.rounds.length,6);
    assert.deepEqual(s.checks.map(c=>c.round), preset==='rhythm'?[1,2,3,4,5,6]:preset==='table'?[2,4,6]:[3,6]);
    assert(s.table.exposures.every(e=>e.round<=6));
    assert.equal(new Set(s.table.exposures.map(e=>e.card.id)).size,s.table.exposures.length);
    assert.equal(s.table.remaining,312-s.table.exposures.length);
    assert.equal(s.bets.length,5); assert(s.bets.every(b=>b.round>1));
    const summary=summarizeChallenge(s);
    assert.equal(summary.checkpoints,s.checks.length); assert.equal(summary.skillScore,100);
    assert.equal(summary.bestCleanStreak,s.checks.length); assert(summary.eligibleForRecord);
    if(preset==='rhythm') { assert.equal(summary.decisions,0); assert(s.table.decisions.length>0); }
    else { assert(summary.decisions>0); assert.equal(summary.decisions,s.table.decisions.length); }
    assert.deepEqual(readChallengeResult(finishChallenge(s,END)),finishChallenge(s,END));
  }
});

test('dealer hole remains hidden and uncounted; immutable advances expose at most one card',()=>{
  let s=rig(['10','2','10','6','8','3','7','K','10','10']);
  for(let i=0;i<8;i++) {
    const before=JSON.stringify(s), old=s;
    s=advanceChallenge(s);
    assert.equal(JSON.stringify(old),before);
    assert(s.table.exposures.length-old.table.exposures.length<=1);
  }
  assert.equal(s.table.exposures.length,7);
  assert.equal(s.table.table.dealer.cards[1],null);
  assert.equal(count(s),1);
  assert(!s.table.exposures.some(e=>e.card.id==='fixture-7'));
  s=until(s,state=>state.phase==='round-result');
  assert.equal(s.table.exposures.filter(e=>e.card.id==='fixture-7').length,1);
  assert.equal(s.table.exposures.find(e=>e.card.id==='fixture-7').kind,'hole');
});

test('bet answers are first input only and correctness remains hidden until a prompted checkpoint',()=>{
  let s=until(create('pace'),state=>state.phase==='bet');
  assert.equal(s.table.round,2); assert.equal(s.checks.length,0);
  const units=practiceBetUnits(tc(s));
  s=submitChallengeBet(s,units);
  assert.equal(s.bets[0].revealed,false);
  assert.equal(summarizeChallenge(s).bets,0);
  assert.throws(()=>submitChallengeBet(s,units));
  s=until(s,state=>state.phase==='count-checkpoint');
  assert.equal(s.table.round,3);
  assert(s.bets.every(b=>!b.revealed));
  const old=s;
  s=submitChallengeCount(s,count(s)+1,tc(s)+1);
  assert.equal(old.checks.length,0);
  assert.equal(s.checks[0].runningCorrect,false);
  assert.equal(s.checks[0].trueCorrect,false);
  assert.equal(s.feedback.bets.length,2);
  assert(s.feedback.bets.every(b=>b.revealed));
  assert.equal(summarizeChallenge(s).bets,2);
  assert.throws(()=>submitChallengeCount(s,count(s),tc(s)));
  assert.throws(()=>submitChallengeCount(old,0.5,0));
  assert.throws(()=>submitChallengeCount(old,null,0));
});

test('unprompted checkpoints cannot be answered and automatic play earns no manual score',()=>{
  let s=until(create('table'),state=>state.phase==='round-result');
  s=advanceChallenge(s);
  assert.equal(s.phase,'bet'); assert.equal(s.table.round,2); assert.equal(s.checks.length,0);
  assert.throws(()=>submitChallengeCount(s,0,0));
  let automatic=until(create('rhythm'),state=>state.phase==='player-turn'||state.phase==='insurance');
  assert.throws(()=>applyChallengeAction(automatic,expected(automatic)));
  automatic=advanceChallenge(automatic);
  assert.equal(automatic.manualDecisions.length,0);
});

test('lost-count choice records a first-answer miss, supplies an anchor, and replays faithfully',()=>{
  let s=until(create(),state=>state.phase==='count-checkpoint');
  const target=count(s);
  s=submitChallengeCount(s,null,null);
  assert.equal(s.feedback.checkpoint.runningGuess,null);
  assert.equal(s.feedback.checkpoint.trueGuess,null);
  assert.equal(s.feedback.checkpoint.arithmeticCorrect,null);
  assert.equal(s.feedback.checkpoint.runningTarget,target);
  assert.equal(summarizeChallenge(s).cleanCheckpoints,0);
  assert.throws(()=>submitChallengeCount(s,target,tc(s)));
  assert(readChallengeResult(finishChallenge(s,END)));
  s=play(s);
  assert.equal(s.checks.length,6);assert.equal(summarizeChallenge(s).runningCorrect,5);
  assert(readChallengeResult(finishChallenge(s,END)));
});

test('skill scoring is separate from luck and simulated stake size',()=>{
  const correct=play(create('table',44));
  const wrongCounts=play(create('table',44),true);
  assert.deepEqual(correct.rounds,wrongCounts.rounds);
  assert(summarizeChallenge(correct).skillScore>summarizeChallenge(wrongCounts).skillScore);
  const flat=play(create('table',44),false,1), large=play(create('table',44),false,4);
  assert.deepEqual(flat.table.exposures,large.table.exposures);
  for(let i=1;i<6;i++) assert.equal(large.rounds[i].netUnits,flat.rounds[i].netUnits*4);
  assert.equal(flat.rounds[0].netUnits,large.rounds[0].netUnits);
  assert.equal(summarizeChallenge(flat).decisionsCorrect,summarizeChallenge(large).decisionsCorrect);
  assert.equal(summarizeChallenge(flat).runningCorrect,summarizeChallenge(large).runningCorrect);
});

test('simulated settlement includes doubles, split hands, surrender, naturals, and insurance',()=>{
  // Settlement is delegated to the tested rules engine; this verifies wrapper accounting.
  const states=[
    ['double', ['10','6','10','6','8','5','7','10','10','5'], 'double'],
    ['split', ['10','8','10','6','8','8','7','10','3','10','2','10','5'], 'split'],
    ['surrender', ['10','10','10','10','8','6','7','8'], 'surrender'],
    ['natural', ['10','A','10','9','8','K','7','8'], null],
    ['insurance', ['10','9','10','A','8','7','7','K'], 'insure'],
  ];
  for(const [name,values,firstAction] of states) {
    let s=rig(values);
    if(firstAction) {
      s=until(s,state=>state.phase==='player-turn'||state.phase==='insurance');
      s=applyChallengeAction(s,firstAction);
    }
    s=until(s,state=>state.phase==='round-result');
    const view=s.table.table, hands=view.seats.find(seat=>seat.isUser).hands;
    const expectedNet=hands.reduce((sum,h)=>sum+(h.outcome==='Win'?h.units:h.outcome==='Loss'?-h.units:h.outcome==='Surrender'?-.5:h.outcome==='Blackjack · 3:2'?1.5:0),view.insurance?.netUnits??0);
    assert.equal(s.rounds[0].baseNetUnits,expectedNet,name);
    assert.equal(s.rounds[0].netUnits,expectedNet,name);
    if(name==='double') assert.equal(hands[0].units,2);
    if(name==='split') assert.equal(hands.length,2);
    if(name==='insurance') assert.equal(view.insurance.netUnits,1);
    if(name==='natural') assert.equal(s.rounds[0].netUnits,1.5);
    if(name==='surrender') assert.equal(s.rounds[0].netUnits,-.5);
  }
});

test('partial, interrupted and same-seed repair runs preserve learning but cannot become records',()=>{
  const partial=finishChallenge(advanceChallenge(create()),END);
  assert(!partial.completed); assert(!partial.summary.eligibleForRecord); assert(readChallengeResult(partial));
  const resumed=play(markChallengeInterrupted(advanceChallenge(create())));
  assert(resumed.completed); assert(resumed.interrupted); assert(!summarizeChallenge(resumed).eligibleForRecord);
  assert(readChallengeResult(finishChallenge(resumed,END)));
  const repair=result('rhythm',44,{replay:true,id:'repair-44'});
  assert(repair.completed); assert(!repair.summary.eligibleForRecord); assert(readChallengeResult(repair));
  assert.deepEqual(repair.exposures,result().exposures);
  assert.deepEqual(repair.rounds,result().rounds);
});

test('history replay rejects fabricated inputs, targets, scores, results, and invalid settings',()=>{
  const original=result();
  const corruptions=[
    r=>{r.summary.skillScore=101;}, r=>{r.summary.netUnits+=100;},
    r=>{r.checks[0].runningTarget+=1;}, r=>{r.checks[0].runningGuess+=1;},
    r=>{r.rounds[0].netUnits+=1;}, r=>{r.exposures[0].card.value='A';},
    r=>{r.inputs.unshift({kind:'continue'});}, r=>{r.inputs.push({kind:'advance'});},
    r=>{r.inputs[0].cheat=true;}, r=>{r.inputs[0]={kind:'bet',units:100};},
    r=>{r.seed=-1;}, r=>{r.seed=Number.MAX_SAFE_INTEGER;}, r=>{r.endedAt='yesterday';},
    r=>{r.endedAt='2025-01-01T00:00:00.000Z';}, r=>{r.completed=false;},
  ];
  for(const mutate of corruptions) {
    const changed=structuredClone(original);mutate(changed);
    assert.equal(readChallengeResult(changed),null,mutate.toString());
  }
  assert.deepEqual(validateChallengeHistory([original,original,{bogus:true}]),[original]);
  assert.deepEqual(validateChallengeHistory('invalid'),[]);
});

test('recommendations require two strong fresh runs with different seeds; all presets remain available',()=>{
  const a=result('rhythm',44), b=result('rhythm',45);
  b.endedAt='2026-01-01T00:06:00.000Z';
  assert.equal(recommendChallenge([]).presetId,'rhythm');
  assert.equal(recommendChallenge([a]).presetId,'rhythm');
  assert.equal(recommendChallenge([a,b]).presetId,'table');
  assert.equal(recommendChallenge([a,{...b,replay:true}]).presetId,'rhythm');
  assert.equal(recommendChallenge([a,{...b,seed:a.seed}]).presetId,'rhythm');
  assert.equal(recommendChallenge([result('table',44),result('table',45)]).presetId,'pace');
  assert.equal(create('pace').presetId,'pace');
});

test('practice schedule is explicit and curated seed selection is deterministic with raised-bet opportunities',()=>{
  for(const [count,units] of [[-3,1],[0,1],[1,2],[2,4],[5,4]]) assert.equal(practiceBetUnits(count),units);
  for(const seed of [0,42,1234,999999]) {
    const selected=chooseChallengeSeed(seed);
    assert.equal(selected,chooseChallengeSeed(seed));
    const s=play(create('rhythm',selected));
    assert(s.bets.some(b=>b.expectedUnits>1),`No bet opportunity for ${seed}`);
  }
});
