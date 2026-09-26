'use client';

import { useEffect, useRef, useState } from 'react';
import { Action, formatAction } from '@/lib/basicStrategy';
import { parseCount, signed } from '@/lib/countPolicy';
import { Card, getCardValue, isRedSuit } from '@/lib/deck';
import { explainStrategyCase, STRATEGY_PROFILE } from '@/lib/strategyPractice';
import {
  addQuickResult, createQuickQuestions, createQuickResult, EMPTY_QUICK_HISTORY, localDateKey,
  QUICK_DURATION_MS, quickHistoryStats, quickRunCreditsDay, readQuickHistory, recordQuickAnswer,
  summarizeQuickAnswers, QuickAnswer, QuickHistory, QuickMode, QuickQuestion, QuickSessionResult,
} from '@/lib/quickPlay';

type Phase = 'idle' | 'question' | 'feedback' | 'paused' | 'done';
type Destination = 'train' | 'table' | 'strategy';
interface Props {
  onActiveChange: (active: boolean) => void;
  onOpenTraining: (target: Destination) => void;
}
const STORAGE_KEY = 'counting-quick-play-v1';
const LANES: { mode: QuickMode; title: string; description: string }[] = [
  { mode: 'counting', title: 'Count', description: 'Quick pairs. Keep the tally.' },
  { mode: 'strategy', title: 'Strategy', description: 'See a hand. Choose your play.' },
  { mode: 'mixed', title: 'Mixed', description: 'Switch between both skills.' },
];
const SHORTCUTS: Record<Action, string> = { hit: 'H', stand: 'S', double: 'D', split: 'P', surrender: 'R' };

function PlayingCard({ card }: { card: Card }) {
  return <div className={`quick-card ${isRedSuit(card.suit) ? 'is-red' : ''}`} aria-label={`${card.value} ${card.suit}`}>
    <span>{card.value}</span><span aria-hidden>{card.suit}</span>
  </div>;
}

export default function QuickPlay({ onActiveChange, onOpenTraining }: Props) {
  const [mode, setMode] = useState<QuickMode>('mixed');
  const [phase, setPhase] = useState<Phase>('idle');
  const [questions, setQuestions] = useState<QuickQuestion[]>([]);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<QuickAnswer[]>([]);
  const [remaining, setRemaining] = useState(QUICK_DURATION_MS);
  const [numericGuess, setNumericGuess] = useState('');
  const [held, setHeld] = useState(false);
  const [history, setHistory] = useState<QuickHistory>(EMPTY_QUICK_HISTORY);
  const [loaded, setLoaded] = useState(false);
  const [storageMessage, setStorageMessage] = useState('');
  const [inputMessage, setInputMessage] = useState('');
  const [result, setResult] = useState<QuickSessionResult | null>(null);
  const [personalBest, setPersonalBest] = useState(false);
  const phaseRef = useRef<Phase>('idle');
  const pausedPhase = useRef<'question' | 'feedback'>('question');
  const questionsRef = useRef<QuickQuestion[]>([]);
  const indexRef = useRef(0);
  const answersRef = useRef<QuickAnswer[]>([]);
  const historyRef = useRef<QuickHistory>(EMPTY_QUICK_HISTORY);
  const seedRef = useRef(0);
  const runModeRef = useRef<QuickMode>('mixed');
  const startedAt = useRef('');
  const elapsed = useRef(0);
  const clockStarted = useRef<number | null>(null);
  const feedbackWait = useRef(650);
  const feedbackDeadline = useRef<number | null>(null);
  const feedbackHeld = useRef(false);
  const answered = useRef(false);
  const finished = useRef(false);
  const interrupted = useRef(false);
  const pauseHandler = useRef<() => void>(() => {});

  useEffect(() => {
    try {
      const saved = readQuickHistory(localStorage.getItem(STORAGE_KEY));
      historyRef.current = saved.history; setHistory(saved.history);
      if (saved.reset) setStorageMessage('Some saved practice data could not be read. You can keep playing.');
    } catch { setStorageMessage('Progress is unavailable on this device. You can still play.'); }
    setLoaded(true);
  }, []);

  useEffect(() => { onActiveChange(['question', 'feedback', 'paused'].includes(phase)); }, [phase, onActiveChange]);

  function changePhase(next: Phase) { phaseRef.current = next; setPhase(next); }
  function activeElapsed() {
    return Math.min(QUICK_DURATION_MS, elapsed.current + (clockStarted.current === null ? 0 : Math.max(0, performance.now() - clockStarted.current)));
  }
  function stopClock() {
    elapsed.current = activeElapsed(); clockStarted.current = null;
    setRemaining(QUICK_DURATION_MS - elapsed.current);
    return elapsed.current;
  }

  function finish(endReason: QuickSessionResult['endReason']) {
    if (finished.current || !['question', 'feedback', 'paused'].includes(phaseRef.current)) return;
    finished.current = true;
    const used = stopClock();
    const nextResult = createQuickResult({ seed: seedRef.current, mode: runModeRef.current,
      startedAt: startedAt.current, endedAt: new Date().toISOString(),
      elapsedMs: endReason === 'timer' ? QUICK_DURATION_MS : used, endReason,
      interrupted: interrupted.current, answers: answersRef.current });
    setPersonalBest(summarizeQuickAnswers(nextResult.answers).xp > historyRef.current.bestXp);
    const nextHistory = addQuickResult(historyRef.current, nextResult);
    historyRef.current = nextHistory; setHistory(nextHistory); setResult(nextResult);
    feedbackDeadline.current = null; changePhase('done');
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(nextHistory)); setStorageMessage(''); }
    catch { setStorageMessage('This round is available here, but could not be saved on this device.'); }
  }

  function start(nextMode = mode) {
    if (!loaded || ['question', 'feedback', 'paused'].includes(phaseRef.current)) return;
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    const nextQuestions = createQuickQuestions(seed, nextMode);
    seedRef.current = seed; runModeRef.current = nextMode; startedAt.current = new Date().toISOString();
    questionsRef.current = nextQuestions; indexRef.current = 0; answersRef.current = [];
    elapsed.current = 0; clockStarted.current = null; answered.current = false; finished.current = false;
    interrupted.current = false; feedbackDeadline.current = null; feedbackWait.current = 650; feedbackHeld.current = false;
    setMode(nextMode); setQuestions(nextQuestions); setIndex(0); setAnswers([]); setResult(null); setPersonalBest(false);
    setRemaining(QUICK_DURATION_MS); setNumericGuess(''); setInputMessage(''); setHeld(false); changePhase('question');
  }

  function nextQuestion() {
    if (phaseRef.current !== 'feedback') return;
    if (elapsed.current >= QUICK_DURATION_MS) { finish('timer'); return; }
    const next = indexRef.current + 1;
    if (next >= questionsRef.current.length) { finish('question-limit'); return; }
    indexRef.current = next; answered.current = false; feedbackDeadline.current = null;
    feedbackHeld.current = false;
    setIndex(next); setNumericGuess(''); setInputMessage(''); setHeld(false); changePhase('question');
  }

  function submit(value: number | Action, renderedIndex: number) {
    if (phaseRef.current !== 'question' || renderedIndex !== indexRef.current || answered.current || finished.current) return;
    const used = activeElapsed();
    if (used >= QUICK_DURATION_MS) { finish('timer'); return; }
    let next: QuickAnswer[];
    try { next = recordQuickAnswer(questionsRef.current, answersRef.current, indexRef.current, value, used); }
    catch { setInputMessage('Choose one of the available answers, or enter a signed whole count.'); return; }
    answered.current = true;
    elapsed.current = used; clockStarted.current = null; setRemaining(QUICK_DURATION_MS - used);
    const needsReview = !next[next.length - 1].correct;
    answersRef.current = next; setAnswers(next); setInputMessage(''); setHeld(needsReview); feedbackHeld.current = needsReview;
    feedbackWait.current = needsReview ? 1200 : 650;
    feedbackDeadline.current = null; changePhase('feedback');
  }

  function pause() {
    const current = phaseRef.current;
    if (current !== 'question' && current !== 'feedback') return;
    pausedPhase.current = current; interrupted.current = true;
    if (current === 'question' && stopClock() >= QUICK_DURATION_MS) { finish('timer'); return; }
    if (current === 'feedback' && feedbackDeadline.current !== null) {
      feedbackWait.current = Math.max(0, feedbackDeadline.current - performance.now());
      feedbackDeadline.current = null;
    }
    changePhase('paused');
  }
  pauseHandler.current = pause;

  useEffect(() => {
    const onVisibility = () => { if (document.hidden) pauseHandler.current(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  useEffect(() => {
    if (phase !== 'question') return;
    if (document.hidden) { pauseHandler.current(); return; }
    clockStarted.current = performance.now();
    const timer = window.setInterval(() => {
      if (document.hidden) { pauseHandler.current(); return; }
      if (phaseRef.current !== 'question') return;
      const used = activeElapsed();
      setRemaining(QUICK_DURATION_MS - used);
      if (used >= QUICK_DURATION_MS) finish('timer');
    }, 100);
    return () => window.clearInterval(timer);
    // Clock and completion read synchronous refs so interval ticks cannot use stale answers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, index]);

  useEffect(() => {
    if (phase !== 'feedback' || held) return;
    feedbackDeadline.current = performance.now() + feedbackWait.current;
    const timer = window.setTimeout(() => {
      if (document.hidden) { pauseHandler.current(); return; }
      if (phaseRef.current === 'feedback' && !feedbackHeld.current) nextQuestion();
    }, feedbackWait.current);
    return () => window.clearTimeout(timer);
    // The feedback timeout advances only the question whose feedback is currently visible.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, index, held]);

  const question = questions[index];
  useEffect(() => {
    if (phase !== 'question' || !question) return;
    const keydown = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.repeat) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (question.kind === 'strategy') {
        const action = (Object.keys(SHORTCUTS) as Action[]).find(value => SHORTCUTS[value].toLowerCase() === event.key.toLowerCase());
        if (action && question.choices.includes(action)) { event.preventDefault(); submit(action, question.index); }
      } else if (/^[0-9+-]$/.test(event.key)) {
        event.preventDefault(); setNumericGuess(previous => (previous + event.key).slice(0, 5));
      } else if (event.key === 'Backspace') { event.preventDefault(); setNumericGuess(previous => previous.slice(0, -1)); }
      else if (event.key === 'Enter' && question.kind === 'counting') {
        const value = parseCount(numericGuess);
        if (value !== null) { event.preventDefault(); submit(value, question.index); }
      }
    };
    document.addEventListener('keydown', keydown);
    return () => document.removeEventListener('keydown', keydown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, question, numericGuess]);

  const stats = quickHistoryStats(history);
  const today = localDateKey();
  const todayComplete = history.activityDays.includes(today);
  const score = summarizeQuickAnswers(answers);
  const lastAnswer = answers[answers.length - 1];
  const skillScores = (['counting', 'strategy'] as const).map(kind => {
    const entries = answers.filter(answer => questions[answer.index]?.kind === kind);
    return { kind, attempted: entries.length, correct: entries.filter(answer => answer.correct).length };
  });
  const weak = [...skillScores].filter(skill => skill.attempted > 0)
    .sort((a, b) => a.correct / a.attempted - b.correct / b.attempted)[0];
  const improvement: Destination = weak?.kind === 'counting' ? 'train' : weak?.kind === 'strategy' ? 'strategy' : 'table';

  return <section className={`quick-play quick-phase-${phase}`} aria-label="Quick practice">
    {storageMessage && <p className="storage-message" role="status">{storageMessage}</p>}
    {phase === 'idle' ? <>
      <div className="quick-hero"><span className="eyebrow">A better daily habit</span><h1>One quick round.</h1>
        <p>Count cards. Make the right play. Get a little sharper with each round.</p></div>
      <div className="quick-stats"><div><strong>{stats.streak}</strong><span>day streak</span></div>
        <div><strong>{stats.bestXp}</strong><span>Best practice XP · any lane</span></div>
        <div><strong>{todayComplete ? 'Done' : 'Ready'}</strong><span>Today’s practice</span></div></div>
      <div className="quick-lanes" role="group" aria-label="Practice lane">{LANES.map(lane => <button key={lane.mode}
        className={`quick-lane ${mode === lane.mode ? 'active' : ''}`} aria-pressed={mode === lane.mode} onClick={() => setMode(lane.mode)}>
        <strong>{lane.title}</strong><span>{lane.description}</span></button>)}</div>
      <button className="primary-button quick-start" disabled={!loaded} onClick={() => start()}>{loaded ? 'Play 60 seconds →' : 'Loading your progress…'}</button>
      <p className="quick-help">60 seconds of answering time. Feedback and pauses don’t use your clock. Correct first answers earn XP.</p>
      {mode === 'mixed' && <p className="quick-help">Mixed alternates separate drills. Only Count cards change your running count. The full table combines every exposed card with playing decisions.</p>}
      <div className="quick-training-links"><span>Go deeper</span><button className="text-button" onClick={() => onOpenTraining('train')}>Guided counting</button>
        <button className="text-button" onClick={() => onOpenTraining('table')}>Full table</button><button className="text-button" onClick={() => onOpenTraining('strategy')}>Strategy practice</button></div>
    </> : phase === 'done' && result ? <div className="quick-results">
      <span className="eyebrow">{result.endReason === 'timer' ? 'Round complete' : result.endReason === 'question-limit' ? '200 decisions completed' : 'Round ended early'}</span>
      <h2>+{score.xp} <span>practice XP</span></h2>
      {personalBest && <p className="quick-personal-best">New best practice XP · all lanes</p>}
      <p>{score.correct} of {score.attempts} first answers correct · best combo {score.bestCombo}</p>
      <div className="quick-skill-scores">{skillScores.filter(skill => skill.attempted).map(skill => <div key={skill.kind}>
        <strong>{skill.correct}/{skill.attempted}</strong><span>{skill.kind === 'counting' ? 'Exact counts' : 'Correct strategy plays'}</span></div>)}</div>
      <p className="coach-note">{score.attempts === 0 ? 'No answers yet. Take a fresh round when you have a minute.'
        : weak && weak.correct < weak.attempted ? weak.kind === 'counting'
          ? 'Next focus: count each pair carefully and carry your corrected total forward.'
          : 'Next focus: slow down on strategy. Use the hand category and dealer upcard.'
        : 'Your first answers were accurate. Try the full table to combine counting and decisions.'}</p>
      {quickRunCreditsDay(result) ? <p className="quick-daily-note">Daily practice logged · {stats.streak}-day streak</p>
        : <p className="quick-help">A full round with at least five answers counts toward your daily habit.</p>}
      <div className="button-row"><button className="primary-button" onClick={() => start(runModeRef.current)}>Play again →</button>
        <button className="quiet-button" onClick={() => changePhase('idle')}>Change lane</button></div>
      <button className="text-button" onClick={() => onOpenTraining(improvement)}>{improvement === 'train' ? 'Practice counting with a coach' : improvement === 'strategy' ? 'Review strategy' : 'Try a full table session'} →</button>
      <p className="fine-print">XP and streaks track practice, not mastery. This round gave feedback after each answer.</p>
    </div> : <div className="quick-session">
      <div className="section-top"><div><span className="eyebrow">{LANES.find(lane => lane.mode === runModeRef.current)?.title} · quick practice</span>
        <h2><span className="quick-seconds">{Math.ceil(remaining / 1000)}</span><span className="meta"> seconds of answering time</span></h2></div>
        <button className="quiet-button" disabled={phase === 'paused'} onClick={pause}>Pause</button></div>
      <div className="quick-timer-track" role="progressbar" aria-label="Answering time remaining" aria-valuemin={0} aria-valuemax={60} aria-valuenow={Math.ceil(remaining / 1000)}>
        <div className="quick-timer-fill" style={{ width: `${remaining / QUICK_DURATION_MS * 100}%` }} /></div>
      <div className="quick-live-stats"><span><strong>{score.xp}</strong> XP</span><span className={lastAnswer?.combo ? 'quick-combo' : ''}><strong>{lastAnswer?.combo ?? 0}</strong> combo</span>
        <span>{score.correct}/{score.attempts} correct</span></div>
      {phase === 'paused' ? <div className="quick-pause"><span className="eyebrow">Cards covered · clock stopped</span><h2>Take your time.</h2>
        <p>Your place is held on this page. Resume when you’re ready.</p><button className="primary-button" onClick={() => { if (!document.hidden) changePhase(pausedPhase.current); }}>Resume round →</button>
        <button className="quiet-button" onClick={() => finish('interrupted')}>End this round</button></div>
        : question && <>
          <div className="quick-question">
            {question.kind === 'counting' ? <><div className="quick-challenge-header"><span className="eyebrow">Count</span><h3>What’s the new running count?</h3>
              <p>{questions.slice(0, index).some(item => item.kind === 'counting') ? 'Carry your count forward from the last pair.' : <>Start from <strong>{signed(question.countBefore)}</strong></>}</p></div>
              <div className="quick-cards">{question.cards.map(card => <PlayingCard key={card.id} card={card} />)}</div></>
              : <><div className="quick-challenge-header"><span className="eyebrow">Basic strategy</span><h3>What’s your play?</h3></div>
                <div className="quick-table"><div><span className="meta">Dealer shows</span><div className="quick-cards"><PlayingCard card={question.situation.dealer} /></div></div>
                  <div><span className="meta">Your hand</span><div className="quick-cards">{question.situation.player.map(card => <PlayingCard key={card.id} card={card} />)}</div></div></div>
                <p className="quick-rule">6 decks · H17 · DAS · late surrender · dealer checked for blackjack</p>
                {runModeRef.current === 'mixed' && <p className="quick-help">Separate strategy drill: keep your Count total in mind. These cards do not change it.</p>}</>}
          </div>
          <div className={`quick-options ${question.kind === 'strategy' ? 'strategy-options' : 'count-options'}`}>
            {question.choices.map(value => <button key={String(value)} className={`quick-option ${phase === 'feedback' && value === question.target ? 'is-correct' : ''} ${phase === 'feedback' && lastAnswer?.value === value && !lastAnswer.correct ? 'is-wrong' : ''}`}
              disabled={phase !== 'question'} onClick={() => submit(value, question.index)}>
              {typeof value === 'number' ? signed(value) : <>{formatAction(value)} <kbd className="quick-key">{SHORTCUTS[value]}</kbd></>}</button>)}
          </div>
          {phase === 'question' && question.kind === 'counting' && <form className="quick-count-form" onSubmit={event => {
            event.preventDefault(); const value = parseCount(numericGuess); if (value !== null) submit(value, question.index);
          }}><label htmlFor="quick-count-input">Or type a count</label><input id="quick-count-input" value={numericGuess} onChange={event => setNumericGuess(event.target.value)}
            inputMode="text" autoComplete="off" maxLength={5} placeholder="±0" aria-label="Type a signed running count" />
            <button type="submit" className="quiet-button" disabled={parseCount(numericGuess) === null}>Enter ↵</button></form>}
          {inputMessage && <p role="status">{inputMessage}</p>}
          {phase === 'feedback' && lastAnswer && <div className={`quick-feedback ${lastAnswer.correct ? 'is-correct' : 'is-wrong'}`} role="status">
            <div><strong>{lastAnswer.correct ? `Correct · +${lastAnswer.xp} XP` : question.kind === 'counting' ? `Correct count: ${signed(question.target)}` : `Correct play: ${formatAction(question.target)}`}</strong>
              <p>{question.kind === 'counting' ? `${signed(question.countBefore)} + (${signed(question.cards.reduce((sum, card) => sum + getCardValue(card), 0))}) = ${signed(question.target)}. Carry ${signed(question.target)} forward.`
                : explainStrategyCase(question.situation)}</p></div>
            <div className="button-row">{!held && <button className="quiet-button" onClick={() => { feedbackHeld.current = true; setHeld(true); feedbackDeadline.current = null; }}>Hold to review</button>}
              <button className="quiet-button" onClick={nextQuestion}>{held ? 'Next question →' : 'Next →'}</button></div>
            {held && <span className="fine-print">Clock stopped. Continue when you’re ready.</span>}
          </div>}
        </>}
      {phase !== 'paused' && <div className="session-footer"><span>{phase === 'feedback' ? 'Clock paused for feedback' : 'Only your first answer scores'}</span><button className="text-button" onClick={() => finish('ended')}>End round</button></div>}
    </div>}
    {phase === 'idle' && <p className="fine-print" title={STRATEGY_PROFILE}>Strategy follows the six-deck H17, DAS, late-surrender profile. Count questions are pair practice.</p>}
  </section>;
}
