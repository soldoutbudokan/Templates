'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import CountInput from './CountInput';
import { Card, getCardValue } from '@/lib/deck';
import { getHandTotal, isPair } from '@/lib/basicStrategy';
import { estimateDecks, parseCount, signed } from '@/lib/countPolicy';
import { cleanTableTest } from '@/lib/tableHistory';
import { advanceTable, applyTableAction, createTableSession, finishTableSession, legalActions,
  markTableInterrupted, submitTableCount, TableAction, TableSessionResult, TableSessionState } from '@/lib/blackjack';

const actionNames: Record<TableAction, string> = { hit: 'Hit', stand: 'Stand', double: 'Double', split: 'Split', surrender: 'Surrender', insure: 'Take insurance', 'decline-insurance': 'Decline insurance' };
const shortcuts: Record<TableAction, string> = { hit: 'H', stand: 'S', double: 'D', split: 'P', surrender: 'R', insure: 'I', 'decline-insurance': 'N' };
const cardText = (cards: Card[]) => cards.map(card => `${card.value}${card.suit}`).join(' ');
const guessText = (value: number | null) => value === null ? 'Lost / skipped' : signed(value);

function Face({ card }: { card: Card | null }) {
  return card ? <span className={`table-card ${card.suit === '♥' || card.suit === '♦' ? 'red' : ''}`} aria-label={`${card.value} ${card.suit}`}><b>{card.value}</b><span aria-hidden>{card.suit}</span></span>
    : <span className="table-card hole-card" aria-label="Dealer hole card, face down"><span aria-hidden>♠</span></span>;
}

export function FullTableReview({ result }: { result: TableSessionResult }) {
  const [replayRound, setReplayRound] = useState(1);
  const [replayIndex, setReplayIndex] = useState(-1);
  const strategy = result.decisions.filter(d => d.correct).length;
  const running = result.checkpoints.filter(c => c.runningCorrect).length;
  const conversion = result.checkpoints.filter(c => c.trueCorrect).length;
  const missed = result.decisions.filter(d => !d.correct);
  const clean = cleanTableTest(result);
  const events = result.exposures.filter(event => event.round === replayRound);
  const firstIndex = result.exposures.findIndex(event => event.round === replayRound);
  const startingCount = firstIndex > 0 ? result.exposures[firstIndex - 1].runningCount : 0;
  const event = events[replayIndex];
  const rounds = Array.from(new Set(result.exposures.map(item => item.round)));
  const play = result.decisions.filter(d => d.kind === 'play');
  const categories = [
    ['Hard hands', play.filter(d => !isPair(d.playerCards) && !getHandTotal(d.playerCards).soft)],
    ['Soft hands', play.filter(d => !isPair(d.playerCards) && getHandTotal(d.playerCards).soft)],
    ['Pairs', play.filter(d => isPair(d.playerCards))],
    ['Insurance', result.decisions.filter(d => d.kind === 'insurance')],
  ] as const;
  return <div className="review-body table-review">
    <span className="eyebrow">{result.config.mode === 'test' ? 'Full test' : 'Coached table'} · {result.completed ? 'Completed' : 'Partial attempt'}{result.interrupted ? ' · interrupted' : ''}</span>
    <h2>{clean ? 'No errors observed in this test.' : 'Your count and your decisions, separately.'}</h2>
    <p>{result.checkpoints.length} checked rounds · {result.config.otherPlayers} other players · {(result.config.speedMs / 1000).toFixed(2)}s per automatic card · {result.config.roundLimit === 10 ? '10-round target' : '75% cut-card target'}. Six decks, H17, DAS, late surrender.</p>
    <div className="metric-grid">
      <div><strong>{strategy}<span>/{result.decisions.length}</span></strong><p>Correct strategy decisions</p></div>
      <div><strong>{running}<span>/{result.checkpoints.length}</span></strong><p>Exact running counts</p></div>
      <div><strong>{conversion}<span>/{result.checkpoints.length}</span></strong><p>Exact true counts</p></div>
    </div>
    <p className="coach-note">{result.interrupted ? 'This attempt was interrupted and cannot count as an uninterrupted test. ' : !result.completed ? 'This attempt ended early. Recorded decisions and submitted checks remain in your scores. ' : ''}{missed.length ? 'Review your missed playing decisions below, then use Perfect strategy for focused repetition. ' : ''}{running < result.checkpoints.length ? 'Reconstruct the exposed cards to locate where the count drifted. A drift may start in an earlier round. ' : ''}These scores describe the situations observed here. Random rounds do not guarantee every strategy situation; use the balanced strategy test for broader coverage. Betting and count-based deviations are not tested.</p>
    <div className="coverage-row">{categories.map(([label, decisions]) => <span key={label}><strong>{decisions.filter(d => d.correct).length}/{decisions.length}</strong> {label}{!decisions.length ? ' · not observed' : ''}</span>)}</div>
    <h3 className="review-subheading">Playing decisions</h3>
    {!result.decisions.length ? <p>No strategy decisions were recorded.</p> : <>
      {!missed.length && <p className="status-good">Every recorded decision matched this basic-strategy profile.</p>}
      <details open={missed.length > 0}><summary>{missed.length ? `${missed.length} decisions to revisit` : 'Inspect all decisions'}</summary><div className="decision-list">{(missed.length ? missed : result.decisions).map((d, i) => <article key={i} className="decision-review"><span className="eyebrow">Round {d.round} · {d.kind === 'insurance' ? 'Insurance' : `Hand ${d.handIndex + 1}`}</span><p><strong>{cardText(d.playerCards)}</strong> against dealer <strong>{cardText([d.dealerUpcard])}</strong></p><p>You chose <strong>{actionNames[d.action]}</strong>. Basic strategy: <strong>{actionNames[d.expectedAction]}</strong>.</p><p className="fine-print">{d.kind === 'insurance' ? 'Basic strategy declines insurance. An insurance count deviation is outside this rules profile.' : `Available: ${d.legalActions.map(action => actionNames[action]).join(', ')}. This decision was graded before your action, without using the dealer’s hole card.`}</p></article>)}</div></details>
    </>}
    <h3 className="review-subheading">Count checkpoints</h3>
    {!result.checkpoints.length ? <p>No count checkpoints were submitted.</p> : <div className="table-scroll"><table className="score-table"><thead><tr><th>Round</th><th>Running: yours → target</th><th>True: yours → target</th><th>Decks</th><th>Review</th></tr></thead><tbody>{result.checkpoints.map(c => <tr key={c.round}><th>{c.round}</th><td className={c.runningCorrect ? 'status-good' : 'status-review'}>{guessText(c.runningGuess)} → {signed(c.runningTarget)}</td><td className={c.trueCorrect ? 'status-good' : 'status-review'}>{guessText(c.trueGuess)} → {signed(c.trueTarget)}</td><td>{c.decksRemaining}</td><td><button className="text-button" onClick={() => { setReplayRound(c.round); setReplayIndex(-1); }}>Replay round {c.round}</button></td></tr>)}</tbody></table></div>}
    {result.checkpoints.some(c => !c.runningCorrect && !c.trueCorrect && c.arithmeticCorrect) && <p className="fine-print">At least one wrong true count was the correct conversion of your entered running count. Start by repairing count retention.</p>}
    {!!rounds.length && <section className="replay-panel"><h3>Reconstruct the count</h3><p>Only cards actually exposed are replayed, in order. The dealer’s hidden card enters the count at its reveal.</p><label className="select-label">Round <select value={replayRound} onChange={e => { setReplayRound(Number(e.target.value)); setReplayIndex(-1); }}>{rounds.map(round => <option key={round} value={round}>{round}</option>)}</select></label><div className="table-replay-card">{event ? <><Face card={event.card} /><div><span>{event.kind === 'hole' ? 'Dealer hole card revealed' : event.seatIndex === null ? 'Dealer' : `Seat ${event.seatIndex + 1}`} · {signed(getCardValue(event.card))}</span><strong>Count {signed(event.runningCount)}</strong><small>Card {replayIndex + 1} of {events.length}</small></div></> : <p>Start this round at <strong>{signed(startingCount)}</strong>. Carry this count from the previous round.</p>}</div><div className="button-row"><button className="quiet-button" disabled={replayIndex < 0} onClick={() => setReplayIndex(i => i - 1)}>Previous</button><button className="primary-button" disabled={replayIndex >= events.length - 1} onClick={() => setReplayIndex(i => i + 1)}>Next exposed card</button></div></section>}
  </div>;
}

export default function FullTableTest({ onSave, onActiveChange }: { onSave: (result: TableSessionResult) => void; onActiveChange: (active: boolean) => void }) {
  const [mode, setMode] = useState<'test' | 'practice'>('test');
  const [roundLimit, setRoundLimit] = useState<0 | 10>(0);
  const [otherPlayers, setOtherPlayers] = useState<0 | 2>(2);
  const [speedMs, setSpeedMs] = useState(900);
  const [state, setState] = useState<TableSessionState | null>(null);
  const stateRef = useRef<TableSessionState | null>(null);
  const [result, setResult] = useState<TableSessionResult | null>(null);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const [feedback, setFeedback] = useState('');
  const [locked, setLocked] = useState(false);
  const lockRef = useRef(false);
  const [runningInput, setRunningInput] = useState('');
  const [trueInput, setTrueInput] = useState('');
  const [inputError, setInputError] = useState('');
  const replace = useCallback((next: TableSessionState) => { stateRef.current = next; setState(next); }, []);
  const finish = useCallback((current: TableSessionState) => {
    const final = finishTableSession(current);
    stateRef.current = null; setState(null); setResult(final); setFeedback('');
    onActiveChange(false); onSave(final);
  }, [onActiveChange, onSave]);
  const pause = useCallback(() => {
    if (!stateRef.current || pausedRef.current) return;
    pausedRef.current = true; setPaused(true); replace(markTableInterrupted(stateRef.current));
  }, [replace]);
  useEffect(() => {
    const hidden = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', hidden);
    return () => document.removeEventListener('visibilitychange', hidden);
  }, [pause]);
  useEffect(() => {
    if (!locked) return;
    const timer = window.setTimeout(() => { lockRef.current = false; setLocked(false); }, Math.max(450, speedMs));
    return () => window.clearTimeout(timer);
  }, [locked, speedMs]);
  useEffect(() => {
    if (!state || paused || feedback || locked) return;
    if (state.phase === 'complete') { finish(state); return; }
    if (['player-turn', 'insurance', 'count-checkpoint'].includes(state.phase)) return;
    const timer = window.setTimeout(() => {
      if (stateRef.current === state && !pausedRef.current) replace(advanceTable(state));
    }, state.phase === 'round-result' ? Math.max(1800, speedMs * 2) : speedMs);
    return () => window.clearTimeout(timer);
  }, [state, paused, feedback, locked, speedMs, finish, replace]);
  const choose = useCallback((action: TableAction) => {
    const current = stateRef.current;
    if (!current || pausedRef.current || lockRef.current || feedback || !legalActions(current).includes(action)) return;
    lockRef.current = true; setLocked(true);
    const next = applyTableAction(current, action);
    replace(next);
    if (current.config.mode === 'practice' && next.decisions.length > current.decisions.length) {
      const d = next.decisions[next.decisions.length - 1];
      setFeedback(`${d.correct ? 'Correct.' : 'Review this choice.'} ${cardText(d.playerCards)} against ${cardText([d.dealerUpcard])}: basic strategy is ${actionNames[d.expectedAction].toLowerCase()}. Your chosen action was played; keep counting the cards that appear.`);
    }
  }, [feedback, replace]);
  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (event.repeat || event.altKey || event.ctrlKey || event.metaKey || /INPUT|SELECT|TEXTAREA/.test((event.target as HTMLElement).tagName)) return;
      const action = (Object.keys(shortcuts) as TableAction[]).find(key => shortcuts[key].toLowerCase() === event.key.toLowerCase());
      if (action && stateRef.current && legalActions(stateRef.current).includes(action)) { event.preventDefault(); choose(action); }
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [choose]);
  function start() {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    setResult(null); setFeedback(''); setPaused(false); pausedRef.current = false;
    setLocked(false); lockRef.current = false; setRunningInput(''); setTrueInput(''); setInputError('');
    replace(createTableSession({ mode, roundLimit, otherPlayers, speedMs }, seed)); onActiveChange(true);
  }
  function submit(lost = false) {
    const current = stateRef.current;
    if (!current || current.phase !== 'count-checkpoint' || pausedRef.current || feedback) return;
    const running = lost ? null : parseCount(runningInput);
    const trueCount = lost ? null : parseCount(trueInput);
    if (!lost && (running === null || trueCount === null)) { setInputError('Enter a whole number in both fields. Negative counts are allowed.'); return; }
    const next = submitTableCount(current, running, trueCount);
    replace(next); setRunningInput(''); setTrueInput(''); setInputError('');
    if (current.config.mode === 'practice') {
      const c = next.checkpoints[next.checkpoints.length - 1];
      setFeedback(`Running count ${signed(c.runningTarget)}. ${signed(c.runningTarget)} ÷ ${c.decksRemaining} decks, rounded down, gives true count ${signed(c.trueTarget)}. ${c.runningCorrect && c.trueCorrect ? 'Both answers are correct.' : 'Your first answers stay in your results.'} Carry the running count into the next round.`);
    }
  }
  function endEarly() {
    let current = stateRef.current;
    if (!current) return;
    if (current.phase === 'count-checkpoint') current = submitTableCount(current, null, null);
    finish(current);
  }

  if (result) return <section className="paper-card"><FullTableReview result={result} /><div className="button-row"><button className="primary-button" onClick={() => setResult(null)}>Set up another table →</button><span className="fine-print">This result is also in Progress.</span></div></section>;
  if (!state) return <section><div className="page-heading"><span className="eyebrow">Put the skills together</span><h1>Keep the count. Play the hand.</h1><p>A continuous six-deck shoe, other players, a hidden dealer card, and every playing decision. Keep your running count while choosing the correct basic-strategy action.</p></div><div className="table-setup paper-card"><div><h2>Full table test</h2><p>Cards clear after each round. Submit your running count and true count together, then continue the same shoe. Test answers stay hidden until review.</p><p>“Perfect” means basic strategy for this profile: dealer hits soft 17, double after split, late surrender after a negative US peek. Insurance is a decision too. Splits are limited to four hands; split aces receive one card and cannot be resplit.</p><p className="fine-print">A supplied half-deck estimate isolates conversion. Betting, count-based deviations, and physical deck estimation are not assessed. Pausing or leaving this browser tab marks the test interrupted. Reloading discards an active attempt.</p></div><div className="table-settings"><fieldset><legend>Session</legend><div className="choice-grid">{(['test', 'practice'] as const).map(value => <button key={value} className={`choice ${mode === value ? 'active' : ''}`} aria-pressed={mode === value} onClick={() => setMode(value)}>{value === 'test' ? 'Full test' : 'Coached practice'}</button>)}</div></fieldset><label className="select-label">Length<select value={roundLimit} onChange={e => setRoundLimit(Number(e.target.value) as 0 | 10)}><option value={0}>Full shoe · stop at 75% cut</option><option value={10}>Short test · 10 rounds</option></select></label><label className="select-label">Table<select value={otherPlayers} onChange={e => setOtherPlayers(Number(e.target.value) as 0 | 2)}><option value={2}>You + two other players</option><option value={0}>You + dealer</option></select></label><label className="select-label">Automatic card pace<select value={speedMs} onChange={e => setSpeedMs(Number(e.target.value))}><option value={1200}>Steady · 1.2 seconds</option><option value={900}>Standard · 0.9 seconds</option><option value={650}>Quick · 0.65 seconds</option><option value={450}>Fast · 0.45 seconds</option></select></label><button className="primary-button" onClick={start}>Shuffle and start →</button></div></div></section>;
  const actions = legalActions(state);
  return <section className="session-card full-table"><div className="section-top"><div><span className="eyebrow">{state.config.mode === 'test' ? 'Full test' : 'Coached table'}{state.interrupted ? ' · interrupted' : ''}</span><h2>Round {state.round}{state.config.roundLimit ? ` of ${state.config.roundLimit}` : ''}</h2></div><div className="table-shoe"><span>{state.remaining > 0 ? `${estimateDecks(state.remaining)} decks remaining` : 'Shoe exhausted'}</span><progress aria-label="Shoe penetration" value={312 - state.remaining} max={312} /></div></div>
    {paused ? <div className="pause-surface"><span className="eyebrow">Cards covered · dealing stopped</span><h2>Your place is saved.</h2><p>This attempt is now marked interrupted. It cannot regain uninterrupted test status. On resume, the same exposed cards reappear; do not count them twice.</p><button className="primary-button" onClick={() => { pausedRef.current = false; setPaused(false); }}>Resume practice</button></div> : <>
      {state.phase !== 'count-checkpoint' && state.phase !== 'complete' && <div className="blackjack-felt"><div className="dealer-seat"><h3>Dealer <span>H17</span></h3><div className="table-cards">{state.table.dealer.cards.map((card, i) => <Face key={card?.id ?? `hole-${i}`} card={card} />)}</div>{state.config.mode === 'practice' && state.table.dealer.total !== null && <p>Total {state.table.dealer.total}</p>}</div><div className={`table-seats seats-${state.table.seats.length}`}>{state.table.seats.map((seat, seatIndex) => <div key={seatIndex} className={`player-seat ${seat.isUser ? 'your-seat' : ''} ${state.table.activeSeat === seatIndex ? 'active-seat' : ''}`}><h3>{seat.name}</h3><div className="split-hands">{seat.hands.map((hand, handIndex) => <div key={handIndex} className={`table-hand ${state.table.activeSeat === seatIndex && state.table.activeHand === handIndex ? 'active-hand' : ''}`}><div className="table-cards">{hand.cards.map(card => <Face key={card.id} card={card} />)}</div><p>{seat.hands.length > 1 ? `Hand ${handIndex + 1} · ` : ''}{state.config.mode === 'practice' ? `${hand.soft ? 'Soft ' : ''}${hand.total} · ` : ''}{hand.outcome ?? hand.status}{hand.units > 1 ? ' · doubled' : ''}</p></div>)}</div></div>)}</div><p className="table-status" role="status">{state.statusText}</p></div>}
      {!!actions.length && !feedback && <div className="table-action-panel"><h3>{state.phase === 'insurance' ? 'Insurance?' : 'Your decision'}</h3><div className="table-actions">{actions.map(action => <button className="primary-button" key={action} disabled={locked} onClick={() => choose(action)}>{actionNames[action]} <kbd>{shortcuts[action]}</kbd></button>)}</div><p className="fine-print">{state.phase === 'insurance' ? 'Insurance costs half a unit and pays 2:1 if the dealer has blackjack. Decide before the peek.' : 'Choose from the actions currently allowed. The hand is played as you choose.'}</p></div>}
      {state.phase === 'count-checkpoint' && !feedback && <form className="checkpoint-form" onSubmit={event => { event.preventDefault(); submit(); }}><span className="eyebrow">Round {state.round} complete · cards cleared</span><h3>What count are you carrying?</h3><p className="checkpoint-rule">Use <strong>{estimateDecks(state.remaining)} decks</strong>. Divide and round down toward negative infinity. The running count continues from the start of the shoe.</p><div className="answer-fields"><CountInput id="table-running" label="Running count" value={runningInput} onChange={setRunningInput} autoFocus /><CountInput id="table-true" label="True count" value={trueInput} onChange={setTrueInput} /></div>{inputError && <p role="alert">{inputError}</p>}<div className="button-row"><button type="submit" className="primary-button">Submit both counts →</button><button type="button" className="quiet-button" onClick={() => submit(true)}>I lost the count</button></div><p className="fine-print">Your first answers count. Losing the count records both answers as missed.</p></form>}
      {feedback && <div className="table-feedback" role="status"><h3>Coaching</h3><p>{feedback}</p><button className="primary-button" onClick={() => setFeedback('')}>Continue →</button></div>}
    </>}
    <div className="session-footer"><span>{state.decisions.length} decisions recorded · {state.checkpoints.length} count checks</span><div className="button-row">{!paused && <button className="quiet-button" onClick={pause}>Pause</button>}<button className="quiet-button" onClick={endEarly}>End and review</button></div></div>
  </section>;
}
