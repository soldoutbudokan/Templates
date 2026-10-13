'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import PlayingCard from './PlayingCard';
import CountInput from './CountInput';
import { useProgress } from './ProgressProvider';
import { legalActions, TableAction } from '@/lib/blackjack';
import { estimateDecks, parseCount, signed } from '@/lib/countPolicy';
import { getCardValue } from '@/lib/deck';
import {
  CHALLENGE_PRESETS, createChallenge, advanceChallenge, applyChallengeAction,
  submitChallengeBet, submitChallengeCount, continueChallenge, markChallengeInterrupted,
  finishChallenge, summarizeChallenge, recommendChallenge, chooseChallengeSeed,
  ChallengeState, ChallengeResult, ChallengePresetId,
} from '@/lib/challenge';

type Destination = 'train' | 'table' | 'strategy' | 'quick';
interface Props { onActiveChange: (active: boolean) => void; onOpenTraining: (target: Destination) => void }
const PRESETS: ChallengePresetId[] = ['rhythm', 'table', 'pace'];
const NAMES: Record<ChallengePresetId, string> = { rhythm: 'Find your rhythm', table: 'Read the table', pace: 'Hold the pace' };
const DESCRIPTIONS: Record<ChallengePresetId, string> = {
  rhythm: 'Watch the play. Keep the count. Choose the bets.',
  table: 'Play your hands and carry the count across rounds.',
  pace: 'Faster cards. Longer stretches. The same steady count.',
};
const ACTION_NAMES: Record<TableAction, string> = {
  hit: 'Hit', stand: 'Stand', double: 'Double', split: 'Split', surrender: 'Surrender',
  insure: 'Take insurance', 'decline-insurance': 'Decline insurance',
};
const KEYS: Record<TableAction, string> = { hit: 'h', stand: 's', double: 'd', split: 'p', surrender: 'r', insure: 'i', 'decline-insurance': 'n' };
const fraction = (correct: number, total: number) => total ? `${correct}/${total}` : '—';
const unitsText = (value: number) => `${value > 0 ? '+' : ''}${Number(value.toFixed(1))}`;

/** Quiet, opt-in sounds follow exposed cards and submitted actions, never hidden answers. */
function useTableSound() {
  const [enabled, setEnabled] = useState(false);
  const context = useRef<AudioContext | null>(null);
  useEffect(() => {
    try { setEnabled(localStorage.getItem('counting-table-sound') === 'on'); } catch { /* Sound is optional. */ }
    return () => { void context.current?.close(); };
  }, []);
  const prepare = useCallback(() => {
    if (!enabled || typeof window.AudioContext === 'undefined') return;
    context.current ??= new AudioContext();
    void context.current.resume().catch(() => {});
  }, [enabled]);
  const play = useCallback((kind: 'card' | 'chip' | 'check') => {
    const audio = context.current;
    if (!enabled || document.hidden || !audio || audio.state !== 'running') return;
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();
    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(kind === 'card' ? 340 : kind === 'chip' ? 620 : 820, audio.currentTime);
    gain.gain.setValueAtTime(0.0001, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.035, audio.currentTime + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.08);
    oscillator.connect(gain); gain.connect(audio.destination);
    oscillator.start(); oscillator.stop(audio.currentTime + 0.09);
  }, [enabled]);
  function toggle() {
    const next = !enabled; setEnabled(next);
    try { localStorage.setItem('counting-table-sound', next ? 'on' : 'off'); } catch { /* Session preference still works. */ }
    if (next && typeof window.AudioContext !== 'undefined') {
      context.current ??= new AudioContext(); void context.current.resume().catch(() => {});
    }
  }
  return { enabled, toggle, play, prepare };
}

function BetSchedule() {
  return <div className="challenge-schedule" aria-label="Practice bet schedule">
    <span>True count <strong>0 or less</strong> → 1 unit</span>
    <span><strong>+1</strong> → 2 units</span><span><strong>+2 or more</strong> → 4 units</span>
  </div>;
}

function ChallengeReplay({ result }: { result: ChallengeResult }) {
  const missed = result.checks.find(check => !check.runningCorrect || !check.trueCorrect);
  const defaultRound = missed?.round ?? result.exposures.at(-1)?.round ?? 1;
  const [round, setRound] = useState(defaultRound);
  const [cursor, setCursor] = useState(-1);
  const rounds = Array.from(new Set(result.exposures.map(event => event.round)));
  const events = result.exposures.filter(event => event.round === round);
  const firstIndex = result.exposures.findIndex(event => event.round === round);
  const countBefore = firstIndex > 0 ? result.exposures[firstIndex - 1].runningCount : 0;
  const event = events[cursor];
  if (!events.length) return null;
  return <details className="challenge-replay">
    <summary>Replay the exposed cards</summary>
    <p>A checkpoint shows where the counts differ. Step through the cards to find the slip.</p>
    <label>Round <select value={round} onChange={e => { setRound(Number(e.target.value)); setCursor(-1); }}>
      {rounds.map(value => <option key={value} value={value}>{value}</option>)}
    </select></label>
    <div className="challenge-replay-cards">{event ? <><PlayingCard key={event.card.id} card={event.card} size="md" />
      <div><small>{event.kind === 'hole' ? 'Dealer hole card revealed' : event.seatIndex === null ? 'Dealer' : `Seat ${event.seatIndex + 1}`} · {signed(getCardValue(event.card))}</small>
        <strong>Count {signed(event.runningCount)}</strong><small>Card {cursor + 1} of {events.length}</small></div></>
      : <p>Begin this round at <strong>{signed(countBefore)}</strong>.</p>}</div>
    <div className="button-row"><button className="quiet-button" disabled={cursor < 0} onClick={() => setCursor(value => value - 1)}>Previous</button>
      <button className="primary-button" disabled={cursor >= events.length - 1} onClick={() => setCursor(value => value + 1)}>Next card →</button></div>
  </details>;
}

export function ChallengeHistory() {
  const { progress, clearHistory } = useProgress();
  const [selected, setSelected] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const result = progress.challenge.find(entry => entry.id === selected);
  return <div className="paper-card"><h2>Table challenges</h2><p>Six-hand practice runs. Records compare the same challenge and exclude partial, paused, and replayed runs.</p>
    {progress.challenge.length === 0 ? <p>No challenges yet. Play a table to set your first record.</p> : <div className="history-list">
      {[...progress.challenge].reverse().map(entry => {
        const score = summarizeChallenge(entry);
        return <button className="history-row" key={entry.id} aria-expanded={selected === entry.id} onClick={() => setSelected(selected === entry.id ? null : entry.id)}>
          <div><strong>{NAMES[entry.presetId]}</strong><span>{new Date(entry.endedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} · {entry.replay ? 'Repair replay' : entry.interrupted ? 'Paused practice' : entry.completed ? 'Completed' : 'Partial'}</span></div>
          <div><strong>{fraction(score.runningCorrect, score.checkpoints)} exact counts</strong><span>{fraction(score.betsCorrect, score.bets)} practice bets</span></div>
        </button>;
      })}</div>}
    {result && <ChallengeReplay key={result.id} result={result} />}
    {progress.challenge.length > 0 && <div className="clear-progress">{confirmClear ? <><p>Clear saved challenges from this profile and linked devices?</p><button className="quiet-button" onClick={() => { clearHistory('challenge'); setSelected(null); setConfirmClear(false); }}>Clear challenge history</button><button className="quiet-button" onClick={() => setConfirmClear(false)}>Keep it</button></> : <button className="text-button" onClick={() => setConfirmClear(true)}>Clear challenge history</button>}</div>}
  </div>;
}

export default function TableChallenge({ onActiveChange, onOpenTraining }: Props) {
  const { progress, loaded, saveChallenge } = useProgress();
  const recommendation = recommendChallenge(progress.challenge);
  const [selection, setSelection] = useState<ChallengePresetId | null>(null);
  const presetId = selection ?? recommendation.presetId;
  const preset = CHALLENGE_PRESETS[presetId];
  const [state, setState] = useState<ChallengeState | null>(null);
  const stateRef = useRef<ChallengeState | null>(null);
  const [result, setResult] = useState<ChallengeResult | null>(null);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);
  const [running, setRunning] = useState('');
  const [trueCount, setTrueCount] = useState('');
  const [error, setError] = useState('');
  const saved = useRef(false);
  const lastInput = useRef(0);
  const lastExposure = useRef(0);
  const sound = useTableSound();
  const active = !!state && !result;
  useEffect(() => { onActiveChange(active); }, [active, onActiveChange]);

  const replace = useCallback((next: ChallengeState) => { stateRef.current = next; setState(next); }, []);
  const finish = useCallback((current: ChallengeState) => {
    if (saved.current) return;
    saved.current = true;
    const completed = finishChallenge(current);
    saveChallenge(completed); setResult(completed); stateRef.current = null; setState(null);
    pausedRef.current = false; setPaused(false);
  }, [saveChallenge]);

  const pause = useCallback(() => {
    if (!stateRef.current || pausedRef.current) return;
    pausedRef.current = true; setPaused(true); replace(markChallengeInterrupted(stateRef.current));
  }, [replace]);
  useEffect(() => {
    const hidden = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', hidden);
    return () => document.removeEventListener('visibilitychange', hidden);
  }, [pause]);
  useEffect(() => {
    if (!state || paused) return;
    if (state.phase === 'complete') { finish(state); return; }
    const auto = CHALLENGE_PRESETS[state.presetId].autoPlay;
    if (['bet', 'feedback', 'count-checkpoint'].includes(state.phase)
      || (!auto && (state.phase === 'player-turn' || state.phase === 'insurance'))) return;
    const timer = window.setTimeout(() => {
      if (document.hidden) { pause(); return; }
      const current = stateRef.current;
      if (!current || pausedRef.current) return;
      replace(advanceChallenge(current));
    }, state.phase === 'round-result' ? 1400 : CHALLENGE_PRESETS[state.presetId].speedMs);
    return () => window.clearTimeout(timer);
  }, [state, paused, finish, pause, replace]);
  useEffect(() => {
    if (!state) return;
    if (!paused && state.table.exposures.length > lastExposure.current) sound.play('card');
    lastExposure.current = state.table.exposures.length;
  }, [state, paused, sound.play]);

  function start(nextPreset = presetId, replayResult?: ChallengeResult) {
    if (!loaded || stateRef.current) return;
    sound.prepare();
    const seed = replayResult?.seed ?? chooseChallengeSeed(crypto.getRandomValues(new Uint32Array(1))[0]);
    const next = createChallenge(nextPreset, seed, { replay: !!replayResult });
    saved.current = false; pausedRef.current = false; lastInput.current = 0; lastExposure.current = 0;
    setPaused(false); setResult(null); setRunning(''); setTrueCount(''); setError(''); setSelection(nextPreset); replace(next);
  }
  function canInput() {
    if (document.hidden || pausedRef.current || !stateRef.current || performance.now() - lastInput.current < 250) return false;
    lastInput.current = performance.now(); return true;
  }
  const chooseAction = useCallback((action: TableAction) => {
    if (!canInput()) return;
    const current = stateRef.current!;
    if (!legalActions(current.table).includes(action)) return;
    sound.prepare(); replace(applyChallengeAction(current, action));
  }, [replace, sound.prepare]);
  useEffect(() => {
    if (!state || paused || CHALLENGE_PRESETS[state.presetId].autoPlay
      || !['player-turn', 'insurance'].includes(state.phase)) return;
    const handler = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.repeat
        || (event.target as HTMLElement)?.closest('input, select, textarea, [contenteditable="true"]')) return;
      const action = (Object.keys(KEYS) as TableAction[]).find(value => KEYS[value] === event.key.toLowerCase());
      if (action && legalActions(stateRef.current!.table).includes(action)) { event.preventDefault(); chooseAction(action); }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [state, paused, chooseAction]);
  function chooseBet(units: 1 | 2 | 4) {
    if (!canInput() || stateRef.current!.phase !== 'bet') return;
    sound.prepare(); sound.play('chip'); replace(submitChallengeBet(stateRef.current!, units));
  }
  function submitCount(lost = false) {
    if (!canInput() || stateRef.current!.phase !== 'count-checkpoint') return;
    const rc = lost ? null : parseCount(running); const tc = lost ? null : parseCount(trueCount);
    if (!lost && (rc === null || tc === null)) { setError('Enter both counts as whole numbers, including a minus sign when needed.'); return; }
    const next = submitChallengeCount(stateRef.current!, rc, tc);
    replace(next); setRunning(''); setTrueCount(''); setError('');
    if (next.feedback?.checkpoint.runningCorrect && next.feedback.checkpoint.trueCorrect) sound.play('check');
  }
  function continueRun() {
    if (!canInput() || stateRef.current!.phase !== 'feedback') return;
    const next = continueChallenge(stateRef.current!);
    if (next.phase === 'complete') finish(next); else replace(next);
  }

  const records = progress.challenge.filter(entry => entry.presetId === presetId && summarizeChallenge(entry).eligibleForRecord);
  const best = records.length ? Math.max(...records.map(entry => summarizeChallenge(entry).skillScore)) : null;
  const last = records.at(-1);

  if (result) {
    const score = summarizeChallenge(result);
    const next = recommendChallenge(progress.challenge);
    const comparable = progress.challenge.filter(entry => entry.id !== result.id && entry.presetId === result.presetId && summarizeChallenge(entry).eligibleForRecord);
    const newBest = score.eligibleForRecord && (!comparable.length || score.skillScore > Math.max(...comparable.map(entry => summarizeChallenge(entry).skillScore)));
    const missedCount = result.checks.find(check => !check.runningCorrect);
    const missedConversion = result.checks.find(check => !check.trueCorrect);
    const missedDecision = result.manualDecisions.find(decision => !decision.correct);
    const missedBet = result.bets.find(bet => bet.revealed && !bet.correct);
    const coach = missedCount
      ? `At round ${missedCount.round}, your count was ${missedCount.runningGuess === null ? 'lost' : signed(missedCount.runningGuess)}; the exposed cards totalled ${signed(missedCount.runningTarget)}. Replay that stretch, then try a fresh shoe.`
      : missedConversion ? `Your count conversion needs another look: ${signed(missedConversion.runningTarget)} ÷ ${missedConversion.decksRemaining} decks rounds down to ${signed(missedConversion.trueTarget)}.`
      : missedDecision ? `Round ${missedDecision.round}: you chose ${ACTION_NAMES[missedDecision.action].toLowerCase()}. The basic-strategy play was ${ACTION_NAMES[missedDecision.expectedAction].toLowerCase()}.`
      : missedBet ? `Before round ${missedBet.round}, true count ${signed(missedBet.trueTarget)} called for ${missedBet.expectedUnits} practice units. You chose ${missedBet.units}.`
      : score.checkpoints ? 'Every submitted count was exact. Keep that rhythm on a fresh shoe, then try the next challenge.' : 'Your place in the shoe is recorded. Try a full six-hand run when you have a few minutes.';
    return <section className="challenge-screen challenge-result" aria-label="Table challenge results">
      <span className="eyebrow">{result.replay ? 'Repair replay' : result.completed ? 'Six hands complete' : 'Practice ended'} · {NAMES[result.presetId]}</span>
      <h2>{newBest ? 'A new saved best.' : result.completed && score.runningCorrect === score.checkpoints && score.checkpoints > 0 ? 'A steady count.' : 'Your next hand starts here.'}</h2>
      <p className="challenge-verdict">{score.skillScore}% <span>skill score</span></p>
      <p>{CHALLENGE_PRESETS[result.presetId].speedMs / 1000}s per automatic card · checks every {CHALLENGE_PRESETS[result.presetId].checkpointEvery} {CHALLENGE_PRESETS[result.presetId].checkpointEvery === 1 ? 'round' : 'rounds'}</p>
      <div className="challenge-score-grid">
        <div><strong>{fraction(score.runningCorrect, score.checkpoints)}</strong><small>Exact running counts</small></div>
        <div><strong>{fraction(score.trueCorrect, score.checkpoints)}</strong><small>True counts</small></div>
        <div><strong>{fraction(score.betsCorrect, score.bets)}</strong><small>Practice bets</small></div>
        <div><strong>{CHALLENGE_PRESETS[result.presetId].autoPlay ? 'Auto' : fraction(score.decisionsCorrect, score.decisions)}</strong><small>Basic strategy</small></div>
      </div>
      <p className="challenge-result-chips">Table result <strong>{unitsText(score.netUnits)} units</strong> · chip results do not affect your skill score.</p>
      <div className="challenge-coach"><span className="eyebrow">One thing for the next run</span><p>{coach}</p></div>
      <div className="challenge-result-actions"><button className="primary-button" onClick={() => start(result.presetId)}>Fresh shoe · play again →</button>
        <button className="quiet-button" onClick={() => start(result.presetId, result)}>Replay this shoe</button>
        {next.presetId !== result.presetId && <button className="quiet-button" onClick={() => start(next.presetId)}>Try {NAMES[next.presetId].toLowerCase()} →</button>}</div>
      {!score.eligibleForRecord && <p className="fine-print">{result.replay ? 'Same-shoe replays are for repair.' : result.interrupted ? 'You paused this run.' : 'This run ended early.'} Your practice is saved; this run does not set a fresh-shoe record.</p>}
      <ChallengeReplay key={result.id} result={result} />
      <div className="button-row"><button className="text-button" onClick={() => onOpenTraining(missedDecision ? 'strategy' : 'train')}>Practice {missedDecision ? 'strategy' : 'counting'} →</button>
        <button className="text-button" onClick={() => setResult(null)}>Choose a challenge</button><button className="text-button" onClick={() => onOpenTraining('table')}>Try an unassisted full test</button></div>
    </section>;
  }

  if (!state) return <section className="challenge-screen" aria-label="Table challenge">
    <div className="challenge-hero"><div className="challenge-hero-copy"><span className="eyebrow">THE SIX-HAND CHALLENGE</span>
      <h1>Keep the count.<br /><em>Make your move.</em></h1>
      <p>A real table. A changing count. Six hands to find your rhythm.</p>
      <button className="primary-button challenge-start" disabled={!loaded} onClick={() => start()}>{loaded ? 'Play a table' : 'Loading your progress…'} <span aria-hidden="true">→</span></button>
      <p className="challenge-session-note">About 3 minutes · simulated chips · your pace</p>
    </div><div className="challenge-preview" aria-hidden="true"><span className="challenge-preview-label">ONE SHOE. EVERY EXPOSED CARD.</span>
      <div className="challenge-preview-cards"><PlayingCard card={{ value: '5', suit: '♣', id: 'preview-five' }} size="lg" />
        <PlayingCard card={{ value: 'K', suit: '♥', id: 'preview-king' }} size="lg" /><PlayingCard card={{ value: '3', suit: '♠', id: 'preview-three' }} size="lg" /></div>
      <div className="challenge-preview-chip">6<span>HANDS</span></div><span className="challenge-preview-caption">Read the table. Trust your count.</span></div></div>
    <div className="challenge-target"><span className="eyebrow">Your next challenge</span><strong>{NAMES[presetId]}</strong><small>{selection ? DESCRIPTIONS[presetId] : recommendation.reason}</small></div>
    <div className="challenge-presets" role="group" aria-label="Challenge difficulty">{PRESETS.map((id, index) => <button key={id} className={`challenge-preset ${id === presetId ? 'selected' : ''}`} aria-pressed={id === presetId} onClick={() => setSelection(id)}>
      <span className="eyebrow">0{index + 1} · {CHALLENGE_PRESETS[id].speedMs / 1000}s / card</span><strong>{NAMES[id]}</strong><small>{DESCRIPTIONS[id]}</small>
      <span className="challenge-preset-meta">{CHALLENGE_PRESETS[id].autoPlay ? 'Automatic play' : 'You play'} · check every {CHALLENGE_PRESETS[id].checkpointEvery} {CHALLENGE_PRESETS[id].checkpointEvery === 1 ? 'round' : 'rounds'}</span>
    </button>)}</div>
    <div className="challenge-records"><div><strong>{best === null ? '—' : `${best}%`}</strong><small>Best saved score · this challenge</small></div>
      <div><strong>{last ? fraction(summarizeChallenge(last).runningCorrect, summarizeChallenge(last).checkpoints) : '—'}</strong><small>Last run · exact count checks</small></div>
      <div><strong>{records.length}</strong><small>Fresh, uninterrupted runs saved</small></div></div>
    <details className="challenge-rules"><summary>How to play</summary><p>Start at zero. Count every exposed card: 2–6 add 1, 7–9 add 0, and tens through aces subtract 1. The dealer’s face-down card counts only when revealed.</p>
      <p>{preset.autoPlay ? 'The table plays basic strategy for you. Focus on counting and choosing a practice bet before the next hand.' : 'Play your hand using basic strategy while keeping the count. Choose your practice bet before the next hand.'} Enter the running and true counts when the table clears for a check.</p>
      <BetSchedule /><p>Divide your running count by the supplied decks remaining, then round down (−1.5 becomes −2). Bet feedback waits until the next count check. Each check gives you the corrected count to carry forward.</p>
      <p className="fine-print">Six decks · H17 · DAS · late surrender. These practice shoes are selected to include changing bet opportunities. The practice schedule is an execution exercise; chips do not measure mastery. Test fresh, unselected shoes in Full test.</p></details>
    <div className="button-row"><button className="text-button" aria-pressed={sound.enabled} onClick={sound.toggle}>Sound {sound.enabled ? 'on' : 'off'}</button><button className="text-button" onClick={() => onOpenTraining('quick')}>Warm up with quick drills →</button></div>
  </section>;

  const config = CHALLENGE_PRESETS[state.presetId];
  const score = summarizeChallenge(state);
  const table = state.table.table;
  const actions = config.autoPlay ? [] : legalActions(state.table);
  const currentBet = state.bets.find(bet => bet.round === state.table.round)?.units ?? 1;
  return <section className={`challenge-screen challenge-run challenge-phase-${state.phase}`} aria-label="Live table challenge">
    <div className="challenge-top"><div><span className="eyebrow">{NAMES[state.presetId]}{state.replay ? ' · repair replay' : ''}</span><h2>Hand {Math.min(6, state.table.round)} <span className="muted">of 6</span></h2></div>
      <div className="challenge-rounds" aria-label={`${score.roundsCompleted} of 6 hands completed`}>{Array.from({ length: 6 }, (_, i) => <span key={i} className={i < score.roundsCompleted ? 'done' : i === state.table.round - 1 ? 'current' : ''}>{i + 1}</span>)}</div></div>
    <div className="challenge-hud"><div><strong>{currentBet} {currentBet === 1 ? 'unit' : 'units'}</strong><small>This hand’s bet</small></div><div><strong>{fraction(score.runningCorrect, score.checkpoints)}</strong><small>Exact count checks</small></div><div><strong>{unitsText(score.netUnits)}</strong><small>Table units · separate from skill</small></div></div>
    {paused ? <div className="challenge-paused"><span className="eyebrow">Cards covered · dealing stopped</span><h2>Your table is waiting.</h2><p>Resume with the count you were carrying. The same cards reappear; count each only once.</p><button className="primary-button" onClick={() => { if (!document.hidden) { sound.prepare(); pausedRef.current = false; setPaused(false); } }}>Resume table →</button><p className="fine-print">This remains saved practice. Paused runs do not set records.</p></div>
    : <><div className="challenge-felt" aria-label="Blackjack table"><div className="challenge-dealer"><span className="eyebrow">Dealer · hits soft 17</span><div className="challenge-cards">{table.dealer.cards.length ? table.dealer.cards.map((card, index) => <PlayingCard key={card?.id ?? `hidden-${index}`} card={card} size="md" dealt />) : <span className="challenge-card-space">{state.phase === 'feedback' || state.phase === 'count-checkpoint' ? 'Table cleared' : 'Next hand'}</span>}</div>{table.dealer.total !== null && <span className="challenge-total">{table.dealer.total}</span>}</div>
      <div className="challenge-seats">{table.seats.map((seat, seatIndex) => <div key={seatIndex} className={`challenge-seat ${seat.isUser ? 'is-you' : ''} ${table.activeSeat === seatIndex ? 'is-active' : ''}`}><span className="eyebrow">{seat.name}{seat.isUser && config.autoPlay ? ' · auto' : ''}</span><div className="challenge-hands">{seat.hands.map((hand, handIndex) => <div key={handIndex} className={`challenge-hand ${table.activeSeat === seatIndex && table.activeHand === handIndex ? 'is-active' : ''}`}><div className="challenge-cards">{hand.cards.map(card => <PlayingCard key={card.id} card={card} size={seat.isUser ? 'md' : 'sm'} dealt />)}</div><span className="challenge-total">{hand.cards.length ? `${hand.soft ? 'Soft ' : ''}${hand.total}` : '—'}{hand.outcome ? ` · ${hand.outcome}` : hand.status === 'bust' ? ' · Bust' : ''}</span></div>)}</div></div>)}</div>
      <p className="challenge-table-message">{state.phase === 'bet' ? 'Choose your next bet from the count you are carrying.' : state.phase === 'feedback' ? 'Your count checkpoint' : state.phase === 'count-checkpoint' ? 'Cards cleared. What count are you carrying?' : state.phase === 'round-result' ? 'Hand settled. Carry your count into the next hand.' : config.autoPlay ? 'Watch every exposed card. The table is playing for you.' : state.table.statusText}</p></div>
      <div className="challenge-console">
        {state.phase === 'bet' ? <><span className="eyebrow">Before hand {state.table.round}</span><h3>Put your count to work.</h3><p>{estimateDecks(state.table.remaining)} decks remaining. Convert your count, then choose your practice bet.</p><div className="challenge-bets">{([1, 2, 4] as const).map(value => <button key={value} className="challenge-bet" onClick={() => chooseBet(value)}><strong>{value}</strong><small>{value === 1 ? 'unit' : 'units'}</small></button>)}</div><BetSchedule /><p className="fine-print">Your first choice is recorded. Feedback arrives at the next count check.</p></>
          : state.phase === 'count-checkpoint' ? <form className="challenge-check" onSubmit={event => { event.preventDefault(); submitCount(); }}><span className="eyebrow">Count check · after hand {state.table.round}</span><h3>What count did you keep?</h3><p>Use <strong>{estimateDecks(state.table.remaining)} decks remaining</strong>. Divide and round down for the true count.</p><div className="challenge-count-fields"><CountInput id="challenge-running" label="Running count" value={running} onChange={setRunning} autoFocus /><CountInput id="challenge-true" label="True count" value={trueCount} onChange={setTrueCount} /></div>{error && <p role="alert">{error}</p>}<div className="button-row"><button className="primary-button" type="submit">Check my count →</button><button className="quiet-button" type="button" onClick={() => submitCount(true)}>I lost the count</button></div></form>
          : state.phase === 'feedback' && state.feedback ? <div className={`challenge-feedback ${state.feedback.checkpoint.runningCorrect && state.feedback.checkpoint.trueCorrect ? 'good' : 'review'}`} role="status"><span className="eyebrow">{state.feedback.checkpoint.runningCorrect && state.feedback.checkpoint.trueCorrect ? 'Count held' : 'Reset your count and keep going'}</span><h3>Carry {signed(state.feedback.checkpoint.runningTarget)} forward.</h3><p>{signed(state.feedback.checkpoint.runningTarget)} ÷ {state.feedback.checkpoint.decksRemaining} decks → true count <strong>{signed(state.feedback.checkpoint.trueTarget)}</strong></p><p>{state.feedback.checkpoint.runningCorrect ? 'Running count correct.' : `Your running count: ${state.feedback.checkpoint.runningGuess === null ? 'lost' : signed(state.feedback.checkpoint.runningGuess)}.`} {state.feedback.checkpoint.trueCorrect ? 'Conversion correct.' : `Your true count: ${state.feedback.checkpoint.trueGuess === null ? 'lost' : signed(state.feedback.checkpoint.trueGuess)}.`}</p>{state.feedback.bets.map(bet => <p key={bet.round} className="challenge-bet-review">Hand {bet.round}: {bet.correct ? '✓' : '↗'} {bet.units} units chosen{bet.correct ? ' · matched the schedule' : ` · schedule called for ${bet.expectedUnits} at true count ${signed(bet.trueTarget)}`}</p>)}<button className="primary-button" onClick={continueRun}>{state.table.round >= 6 ? 'See my results →' : 'Carry this count · continue →'}</button></div>
          : actions.length ? <div className="challenge-actions"><span className="eyebrow">{state.phase === 'insurance' ? 'Dealer shows an ace' : 'Your hand. Your decision.'}</span><h3>{state.phase === 'insurance' ? 'Insurance?' : 'What’s your play?'}</h3><div className="button-row">{actions.map(action => <button key={action} className="primary-button" onClick={() => chooseAction(action)}>{ACTION_NAMES[action]} <kbd>{KEYS[action].toUpperCase()}</kbd></button>)}</div><p className="fine-print">Use basic strategy. Keep the count while you play.</p></div>
          : <div className="challenge-watching"><span className="eyebrow">{state.phase === 'round-result' ? 'Hand complete' : 'Keep the count'}</span><h3>{state.phase === 'round-result' ? 'A new hand. The same count.' : 'Every exposed card counts.'}</h3><p>{state.table.round === 1 ? 'New six-deck shoe · start at 0 · first bet is 1 unit.' : `Next count check: after hand ${Math.min(6, Math.ceil(state.table.round / config.checkpointEvery) * config.checkpointEvery)}.`}</p></div>}
      </div></>}
    <div className="session-footer"><span>{config.speedMs / 1000}s per automatic card · six-deck Hi-Lo</span><div className="button-row"><button className="text-button" aria-pressed={sound.enabled} onClick={sound.toggle}>Sound {sound.enabled ? 'on' : 'off'}</button>{!paused && <button className="quiet-button" onClick={pause}>Pause</button>}<button className="text-button" onClick={() => { if (stateRef.current) finish(stateRef.current); }}>End and review</button></div></div>
  </section>;
}
