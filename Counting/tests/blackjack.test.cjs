const test = require('node:test');
const assert = require('node:assert/strict');
const { createTableSession, advanceTable, applyTableAction, submitTableCount, legalActions, finishTableSession, markTableInterrupted } = require('../lib/blackjack.ts');
const { getCorrectAction } = require('../lib/basicStrategy.ts');
const { getCardValue } = require('../lib/deck.ts');
const { estimateDecks, toTrueCount } = require('../lib/countPolicy.ts');
const config = { mode:'test',roundLimit:10,otherPlayers:0,speedMs:650 };
function rig(values, overrides={}) {
  const s = createTableSession({...config,...overrides},42);
  s._engine.shoe = [...values.map((value,i)=>({value,suit:'♠',id:`fixture-${i}`})), ...s._engine.shoe.slice(values.length)];
  return s;
}
function until(s,phase,limit=100) {
  while(s.phase !== phase && limit-- > 0) {
    assert(!['player-turn','insurance','count-checkpoint','complete'].includes(s.phase),`Unexpected stop ${s.phase}, wanted ${phase}`);
    s=advanceTable(s);
  }
  assert.equal(s.phase,phase); return s;
}
function expected(s) {
  const e=s._engine,h=e.seats[e.activeSeat].hands[e.activeHand],legal=legalActions(s);
  return getCorrectAction(h.cards,e.dealer[0],{canDouble:legal.includes('double'),canSplit:legal.includes('split'),canSurrender:legal.includes('surrender')});
}
function countAnswer(s,wrong=false) {
  const rc=s.exposures.reduce((n,e)=>n+getCardValue(e.card),0);
  return submitTableCount(s,rc+(wrong?1:0),toTrueCount(rc,estimateDecks(s.remaining))+(wrong?1:0));
}
function playPerfect(s,max=5000,wrongCounts=false) {
  while(!s.ended && max-->0) {
    if(s.phase==='insurance')s=applyTableAction(s,'decline-insurance');
    else if(s.phase==='player-turn')s=applyTableAction(s,expected(s));
    else if(s.phase==='count-checkpoint')s=countAnswer(s,wrongCounts);
    else s=advanceTable(s);
  }
  assert(s.ended,'Session failed to end'); return s;
}

test('deterministic immutable transitions show at most one newly exposed card per advance',()=>{
  const a=createTableSession(config,42),copy=JSON.stringify(a);
  const b=advanceTable(a);
  assert.equal(JSON.stringify(a),copy);
  assert.notEqual(a,b);
  assert.deepEqual(b,advanceTable(createTableSession(config,42)));
  assert.equal(b.exposures.length,1);
  let s=createTableSession({...config,otherPlayers:2},8);
  for(let i=0;i<8;i++) { const prior=s;s=advanceTable(s);assert(s.exposures.length-prior.exposures.length<=1); }
  assert.equal(s.remaining,304);assert.equal(s.exposures.length,7);
});

test('dealer hole is neither visible nor counted until reveal; each physical card counts once',()=>{
  let s=until(rig(['2','6','3','K','5']), 'player-turn');
  assert.equal(s.table.dealer.cards[1],null);assert.equal(s.table.dealer.total,null);
  assert.equal(s.exposures.length,3);assert.equal(s._engine.runningCount,3);
  const original=JSON.stringify(s);s=applyTableAction(s,'hit');
  assert.equal(s._engine.runningCount,4);assert.equal(s.decisions[0].runningCount,3);
  assert.equal(s.decisions[0].playerCards.length,2);assert(!JSON.stringify(s.decisions).includes('fixture-3'));
  s=applyTableAction(s,'stand');s=until(s,'round-result');
  assert.equal(s.exposures.filter(e=>e.card.id==='fixture-3').length,1);
  assert.equal(s.exposures.find(e=>e.card.id==='fixture-3').kind,'hole');
  assert.equal(new Set(s.exposures.map(e=>e.card.id)).size,s.exposures.length);
  assert.notEqual(JSON.stringify(s),original);
});

test('dealer blackjack is checked before surrender/play, after an independent insurance decision',()=>{
  let s=until(rig(['9','A','7','K']), 'insurance');
  assert.deepEqual(legalActions(s),['insure','decline-insurance']);
  assert.throws(()=>applyTableAction(s,'surrender'));
  s=applyTableAction(s,'decline-insurance');
  assert.equal(s.decisions[0].kind,'insurance');assert.equal(s.decisions[0].correct,true);
  assert.equal(s.decisions[0].runningCount,-1);
  s=until(s,'round-result');
  assert.equal(s.decisions.length,1);assert.equal(s.table.seats[0].hands[0].outcome,'Loss');
  assert.equal(s.table.insurance.netUnits,0);assert.equal(s.table.dealer.total,21);
});

test('insurance settlement cannot change whether insurance was a correct basic-strategy choice',()=>{
  for(const [hole,expectedNet] of [['K',1],['9',-.5]]) {
    let s=until(rig(['A','A','K',hole]), 'insurance');
    s=applyTableAction(s,'insure');s=until(s,'round-result');
    assert.equal(s.decisions[0].correct,false);
    assert.equal(s.table.insurance.netUnits,expectedNet);
    assert.equal(s.table.seats[0].hands[0].outcome,hole==='K'?'Push':'Blackjack · 3:2');
  }
});

test('player natural resolves without a playing question; dealer does not draw unnecessarily',()=>{
  let s=until(rig(['A','9','K','7']), 'round-result');
  assert.equal(s.decisions.length,0);
  assert.equal(s.table.seats[0].hands[0].outcome,'Blackjack · 3:2');
  assert.equal(s.table.dealer.cards.length,2);
  s=advanceTable(s);
  assert.equal(s.phase,'count-checkpoint');assert.equal(s.table.seats.length,0);assert.equal(s.table.dealer.cards.length,0);
});

test('double takes exactly one card and freezes the pre-action decision',()=>{
  let s=until(rig(['6','6','5','10','10','5']), 'player-turn');
  const prior=s;s=applyTableAction(s,'double');
  assert.equal(prior.table.seats[0].hands[0].cards.length,2);
  assert.equal(s.table.seats[0].hands[0].cards.length,3);assert.equal(s.table.seats[0].hands[0].units,2);
  assert.equal(s.decisions[0].correct,true);assert.equal(s.decisions[0].expectedAction,'double');
  assert.equal(s.decisions[0].playerCards.length,2);assert.notEqual(s.phase,'player-turn');
  assert.throws(()=>applyTableAction(s,'hit'));
  s=until(s,'round-result');assert.equal(s.table.seats[0].hands[0].outcome,'Push');
});

test('split hands receive cards in turn, preserve count, allow DAS and disallow surrender',()=>{
  let s=until(rig(['8','6','8','10','3','10','2','10','5']), 'player-turn');
  const count=s._engine.runningCount;
  s=applyTableAction(s,'split');assert.equal(s._engine.runningCount,count);
  assert.equal(s.table.seats[0].hands.length,2);
  s=until(s,'player-turn');
  assert.equal(s.table.seats[0].hands[0].cards.length,2);assert.equal(s.table.seats[0].hands[1].cards.length,1);
  assert(legalActions(s).includes('double'));assert(!legalActions(s).includes('surrender'));
  s=applyTableAction(s,'double');s=until(s,'player-turn');
  assert.equal(s.table.activeHand,1);assert.deepEqual(s.table.seats[0].hands[1].cards.map(c=>c.value),['8','2']);
  s=applyTableAction(s,'double');s=until(s,'round-result');
  assert.equal(s.table.seats[0].hands[0].units,2);assert.equal(s.table.seats[0].hands[1].units,2);
});

test('split aces get one card each, cannot resplit, and split21 is not a natural',()=>{
  let s=until(rig(['A','6','A','10','K','A','5']), 'player-turn');
  s=applyTableAction(s,'split');s=until(s,'round-result');
  const hs=s.table.seats[0].hands;
  assert.equal(s.decisions.length,1);assert.equal(hs.length,2);assert.equal(hs[0].cards.length,2);assert.equal(hs[1].cards.length,2);
  assert.equal(hs[0].status,'stood');assert.equal(hs[0].outcome,'Push');assert.equal(hs[1].outcome,'Loss');
});

test('four-hand split cap applies and fallback recommendation stays legal',()=>{
  let s=until(rig(['8','6','8','10','8','8','8','10','10','10','10']), 'player-turn');
  for(let i=0;i<3;i++){s=applyTableAction(s,'split');s=until(s,'player-turn');}
  assert.equal(s.table.seats[0].hands.length,4);
  assert(!legalActions(s).includes('split'));assert(!legalActions(s).includes('surrender'));
  assert.equal(expected(s),'stand');
  assert.throws(()=>applyTableAction(s,'split'));
});

test('three-card hands cannot double/surrender and late surrender ends the hand',()=>{
  let s=until(rig(['2','6','3','10','6']), 'player-turn');
  s=applyTableAction(s,'hit');assert.deepEqual(legalActions(s),['hit','stand']);assert.equal(expected(s),'hit');
  s=until(rig(['10','A','7','9']), 'insurance');s=applyTableAction(s,'decline-insurance');s=until(s,'player-turn');
  assert.equal(expected(s),'surrender');s=applyTableAction(s,'surrender');s=until(s,'round-result');
  assert.equal(s.table.seats[0].hands[0].outcome,'Surrender');
});

test('dealer hits soft17 but stands on hard17',()=>{
  for(const [values,length] of [[['10','A','8','6','2'],3],[['10','10','8','7'],2]]){
    let s=rig(values);s=until(s,values[1]==='A'?'insurance':'player-turn');
    if(s.phase==='insurance'){s=applyTableAction(s,'decline-insurance');s=until(s,'player-turn');}
    s=applyTableAction(s,'stand');s=until(s,'round-result');assert.equal(s.table.dealer.cards.length,length);
  }
});

test('short test has ten count samples and independent play/count grading; scores do not depend on wins',()=>{
  const s=playPerfect(createTableSession({...config,otherPlayers:2},321),5000,true);
  assert.equal(s.completed,true);assert.equal(s.checkpoints.length,10);
  assert(s.decisions.length>0);assert(s.decisions.every(d=>d.correct));assert(s.checkpoints.every(c=>!c.runningCorrect&&!c.trueCorrect));
  assert.equal(s.checkpoints[1].runningTarget,s.exposures.filter(e=>e.round<=2).reduce((n,e)=>n+getCardValue(e.card),0));
  assert(s.decisions.every(d=>!Object.hasOwn(d,'outcome')));
});

test('whole-shoe test stops after crossing cut card, never exposes a duplicate or starts another shoe',()=>{
  for(let seed=0;seed<20;seed++){
    const s=playPerfect(createTableSession({...config,roundLimit:0,otherPlayers:2},seed));
    assert(s.completed);assert(s.remaining<=78);assert(s.remaining>0);
    assert.equal(s.checkpoints.length,s.round);
    assert.equal(s.exposures.length,312-s.remaining);assert.equal(new Set(s.exposures.map(e=>e.card.id)).size,s.exposures.length);
    assert.equal(s.checkpoints.at(-1).runningTarget,s.exposures.reduce((n,e)=>n+getCardValue(e.card),0));
    assert(s.decisions.every(d=>d.correct));assert(s.checkpoints.every(c=>c.runningCorrect&&c.trueCorrect));
  }
});

test('checkpoint answers are exact, first-attempt only; interruption and early finish preserve partial status',()=>{
  let s=until(rig(['A','9','K','7']), 'count-checkpoint');
  const rc=s._engine.runningCount,tc=toTrueCount(rc,estimateDecks(s.remaining));
  assert.throws(()=>submitTableCount(s,1.2,tc));
  const next=submitTableCount(s,rc+1,tc+1);assert.equal(next.checkpoints[0].runningCorrect,false);assert.equal(next.checkpoints[0].trueCorrect,false);
  assert.throws(()=>submitTableCount(next,rc,tc));assert.equal(s.checkpoints.length,0);
  const partial=finishTableSession(markTableInterrupted(next));
  assert.equal(partial.completed,false);assert.equal(partial.interrupted,true);assert.equal(partial.rounds,1);
  assert.equal(next.interrupted,false);
});

test('the 75% cut limit applies to whole-shoe tests; the short option completes ten rounds',()=>{
  for(const roundLimit of [0,10]) {
    let s=until(rig(['A','9','K','7'],{roundLimit}), 'count-checkpoint');
    s._engine.cursor=234;s.remaining=78;
    s=countAnswer(s);
    assert.equal(s.completed,roundLimit===0);
    assert.equal(s.round,roundLimit===0?1:2);
  }
});

test('unexpected exhaustion ends partial without silently shuffling',()=>{
  let s=until(rig(['2','6','3','10']), 'player-turn');
  s._engine.shoe=s._engine.shoe.slice(0,4);
  s=applyTableAction(s,'hit');
  assert(s.ended);assert(!s.completed);assert(s.interrupted);assert.equal(s.remaining,0);assert.equal(s.exposures.length,3);
});
