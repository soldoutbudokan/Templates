'use client';

import { useEffect, useRef, useState } from 'react';
import { getCardValue } from '@/lib/deck';
import { COUNT_POLICY, parseCount, signed } from '@/lib/countPolicy';
import { Checkpoint, CheckpointAnswer, createTrainingPlan, explainAnswer, Focus, gradeCheckpoint, replayCounts,
  SessionMode, SessionResult, TrainingPlan, sessionLabel } from '@/lib/training';
import { LargeCard } from './Card';
import CountInput from './CountInput';

type Phase = 'setup' | 'ready' | 'dealing' | 'answer' | 'feedback' | 'paused' | 'complete';

function Replay({ checkpoint }: { checkpoint: Checkpoint }) {
  const [cursor, setCursor] = useState(-1);
  const counts = replayCounts(checkpoint);
  return <section className="replay-panel" aria-label="Checkpoint replay">
    <div className="section-top"><h3>Replay this segment</h3><span className="meta">{cursor + 1} / {checkpoint.cards.length} cards</span></div>
    <p>Start at <strong>{signed(checkpoint.countBefore)}</strong>. This review does not change your first answer.</p>
    <div className="replay-card">
      {cursor >= 0 ? <LargeCard card={checkpoint.cards[cursor]} /> : <span className="replay-start">Starting count<br /><strong>{signed(checkpoint.countBefore)}</strong></span>}
      {cursor >= 0 && <div><span className="meta">Card value {signed(getCardValue(checkpoint.cards[cursor]))}</span><p className="count-large">{signed(counts[cursor])}</p><span className="meta">Running count</span></div>}
    </div>
    <div className="button-row"><button className="quiet-button" disabled={cursor < 0} onClick={() => setCursor(cursor - 1)}>Previous</button>
      <button className="primary-button" disabled={cursor >= checkpoint.cards.length - 1} onClick={() => setCursor(cursor + 1)}>Next card</button></div>
  </section>;
}

export function SessionReview({ result }: { result: SessionResult }) {
  const [selected, setSelected] = useState<number | null>(null);
  const correct = result.answers.filter(answer => answer.runningCorrect).length;
  const conversions = result.answers.filter(answer => answer.trueCorrect !== null);
  const label = sessionLabel(result);
  return <div className="review-body">
    <span className="eyebrow">{result.completed ? 'Session complete' : 'Session ended early'}</span>
    <h2>{correct === result.answers.length && correct > 0 ? 'A steady count.' : 'Know what to practice next.'}</h2>
    <p className="muted">{label} · {(result.speedMs / 1000).toFixed(2)} seconds per card · 6-deck Hi-Lo</p>
    <div className="metric-grid">
      <div><strong>{correct}<span> / {result.answers.length}</span></strong><p>Exact running counts</p></div>
      {conversions.length > 0 && <div><strong>{conversions.filter(answer => answer.trueCorrect).length}<span> / {conversions.length}</span></strong><p>Exact true counts</p></div>}
      <div><strong>{result.answers.filter(answer => answer.lost).length}</strong><p>Reported lost counts</p></div>
    </div>
    <p className="coach-note">{result.answers.length === 0 ? 'No answers were recorded. Start a fresh practice session to establish a baseline.' : result.answers.some(answer => !answer.runningCorrect)
      ? 'Next: repeat running-count practice at this pace or slower. Review an incorrect checkpoint below to reconstruct that segment.'
      : conversions.some(answer => !answer.trueCorrect) ? 'Next: practice true-count conversion. Your running counts were accurate; focus on division and rounding down.'
        : 'Next: try a fresh session with true-count conversion, or repeat this pace on another day. One clean session is a useful result, not proof of mastery.'}</p>
    {!result.answers.length && <p>No checkpoints were submitted. This attempt adds no accuracy score.</p>}
    <div className="checkpoint-list">
      {result.answers.map(answer => <button key={answer.index} className={`checkpoint-row ${selected === answer.index ? 'selected' : ''}`} onClick={() => setSelected(selected === answer.index ? null : answer.index)} aria-expanded={selected === answer.index}>
        <span>Checkpoint {answer.index + 1}</span><span className={answer.runningCorrect && answer.trueCorrect !== false ? 'status-good' : 'status-review'}>{answer.lost ? 'Lost count' : answer.runningCorrect && answer.trueCorrect !== false ? 'Correct' : 'Review'}</span><span className="meta">Open replay ↗</span>
      </button>)}
    </div>
    {selected !== null && <div className="answer-review"><p>{explainAnswer(result.checkpoints[selected], result.answers[selected])}</p>
      <p>Your running count: <strong>{result.answers[selected].runningGuess === null ? 'Lost' : signed(result.answers[selected].runningGuess!)}</strong> · Correct: <strong>{signed(result.checkpoints[selected].runningCount)}</strong></p>
      {result.focus === 'conversion' && <p>Your true count: <strong>{result.answers[selected].trueGuess === null ? 'Not entered' : signed(result.answers[selected].trueGuess!)}</strong> · Correct: <strong>{signed(result.checkpoints[selected].trueCount)}</strong> ({result.checkpoints[selected].decksRemaining} decks)</p>}
      <Replay key={selected} checkpoint={result.checkpoints[selected]} /></div>}
    <p className="fine-print">Scores describe submitted checkpoints, not individual cards. Repeated count offsets can come from one earlier error. Practice feedback supplies the corrected count after each checkpoint.</p>
  </div>;
}

interface Props {
  suggestedFocus: Focus;
  onSave: (result: SessionResult) => void;
  onActiveChange: (active: boolean) => void;
}

export default function TrainingSession({ suggestedFocus, onSave, onActiveChange }: Props) {
  const [focus, setFocus] = useState<Focus>(suggestedFocus);
  const [mode, setMode] = useState<SessionMode>('practice');
  const [speedMs, setSpeedMs] = useState(900);
  const [plan, setPlan] = useState<TrainingPlan | null>(null);
  const [phase, setPhase] = useState<Phase>('setup');
  const [index, setIndex] = useState(0);
  const [cursor, setCursor] = useState(0);
  const [answers, setAnswers] = useState<CheckpointAnswer[]>([]);
  const [runningGuess, setRunningGuess] = useState('');
  const [trueGuess, setTrueGuess] = useState('');
  const [interrupted, setInterrupted] = useState(false);
  const [assisted, setAssisted] = useState(false);
  const [replay, setReplay] = useState(false);
  const [result, setResult] = useState<SessionResult | null>(null);
  const pausedPhase = useRef<Phase>('ready');
  const phaseRef = useRef<Phase>('setup');
  const responseStarted = useRef(0);
  const responseElapsed = useRef(0);
  const submitted = useRef(false);
  const saved = useRef(false);
  const checkpoint = plan?.checkpoints[index];
  const coached = plan?.mode === 'practice' || interrupted;

  useEffect(() => { if (phase === 'setup') setFocus(suggestedFocus); }, [suggestedFocus, phase]);
  useEffect(() => { phaseRef.current = phase; onActiveChange(!['setup', 'complete'].includes(phase)); }, [phase, onActiveChange]);

  function pause() {
    const current = phaseRef.current;
    if (['setup', 'complete', 'paused'].includes(current)) return;
    pausedPhase.current = current;
    if (current === 'answer') responseElapsed.current += Math.max(0, performance.now() - responseStarted.current);
    setInterrupted(true);
    phaseRef.current = 'paused';
    setPhase('paused');
  }

  useEffect(() => {
    const handleVisibility = () => { if (document.hidden) pause(); };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => document.removeEventListener('visibilitychange', handleVisibility);
  }, []);

  useEffect(() => {
    if (phase !== 'dealing' || !checkpoint || !plan) return;
    const timer = setTimeout(() => {
      if (document.hidden || phaseRef.current !== 'dealing') return;
      if (cursor + 2 >= checkpoint.cards.length) {
        submitted.current = false;
        responseElapsed.current = 0;
        responseStarted.current = performance.now();
        setPhase('answer');
      } else setCursor(previous => previous + 2);
    }, plan.speedMs * Math.min(2, checkpoint.cards.length - cursor));
    return () => clearTimeout(timer);
  }, [phase, cursor, checkpoint, plan]);

  function start() {
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    setPlan(createTrainingPlan(seed, mode, focus, speedMs));
    setIndex(0); setCursor(0); setAnswers([]); setResult(null); setRunningGuess(''); setTrueGuess('');
    setInterrupted(false); setAssisted(false); setReplay(false); saved.current = false; submitted.current = false;
    setPhase('ready');
  }

  function finish(finalAnswers: CheckpointAnswer[], completed: boolean) {
    if (!plan || saved.current) return;
    saved.current = true;
    const finalResult: SessionResult = { ...plan, answers: finalAnswers,
      completed: completed || finalAnswers.length === plan.checkpoints.length, interrupted, assisted, endedAt: new Date().toISOString() };
    onSave(finalResult); setResult(finalResult); setPhase('complete');
  }

  function advance(finalAnswers = answers) {
    if (!plan) return;
    if (index + 1 >= plan.checkpoints.length) { finish(finalAnswers, true); return; }
    setIndex(previous => previous + 1); setCursor(0); setRunningGuess(''); setTrueGuess(''); setReplay(false); setPhase('ready');
  }

  function submit(lost = false) {
    if (phase !== 'answer' || !plan || submitted.current) return;
    const running = lost ? null : parseCount(runningGuess);
    const converted = lost ? null : parseCount(trueGuess);
    if (!lost && (running === null || (plan.focus === 'conversion' && converted === null))) return;
    submitted.current = true;
    const answer = gradeCheckpoint(plan, index, running, converted, responseElapsed.current + performance.now() - responseStarted.current);
    const nextAnswers = [...answers, answer];
    setAnswers(nextAnswers);
    if (coached) setPhase('feedback'); else advance(nextAnswers);
  }

  if (phase === 'setup') return <section className="session-setup">
    <div className="setup-copy"><span className="eyebrow">Your next session</span><h2>Keep the count.<br />Round after round.</h2>
      <p>Cards appear in pairs, then disappear. Carry your running count forward and enter it at each checkpoint.</p>
      <div className="value-key" aria-label="Hi-Lo values"><span><strong>2–6</strong> +1</span><span><strong>7–9</strong> 0</span><span><strong>10–A</strong> −1</span></div>
      <p className="fine-print">The count starts at zero and resets only with a new shoe.</p>
    </div>
    <div className="setup-controls">
      <fieldset><legend>Session</legend><div className="choice-grid">
        <button className={mode === 'practice' ? 'choice active' : 'choice'} onClick={() => setMode('practice')} aria-pressed={mode === 'practice'}><strong>Practice</strong><span>78 cards · feedback and replay</span></button>
        <button className={mode === 'assessment' ? 'choice active' : 'choice'} onClick={() => setMode('assessment')} aria-pressed={mode === 'assessment'}><strong>Assessment</strong><span>234 cards · review at the end</span></button>
      </div></fieldset>
      <label className="select-label">Focus<select value={focus} onChange={event => setFocus(event.target.value as Focus)}><option value="running">Running count</option><option value="conversion">Running + true count</option></select></label>
      <label className="select-label">Pace<select value={speedMs} onChange={event => setSpeedMs(Number(event.target.value))}><option value={1200}>Steady · 1.2 seconds per card</option><option value={900}>Standard · 0.9 seconds per card</option><option value={650}>Quick · 0.65 seconds per card</option><option value={450}>Fast · 0.45 seconds per card</option></select></label>
      {focus === 'conversion' && <p className="fine-print">A half-deck estimate is supplied at each checkpoint. {COUNT_POLICY} This session tests conversion, not visual deck estimation.</p>}
      {mode === 'assessment' && <p className="fine-print">Answers stay hidden until the end. Pausing or leaving the tab makes the attempt practice only. Checkpoints pause dealing; this is a counting assessment, not a full blackjack table test.</p>}
      <button className="primary-button start-button" onClick={start}>Start {mode} <span aria-hidden>→</span></button>
    </div>
  </section>;

  if (phase === 'complete' && result) return <section className="session-card"><SessionReview result={result} /><button className="primary-button" onClick={() => setPhase('setup')}>Choose next session →</button></section>;
  if (!plan || !checkpoint) return null;
  const lastAnswer = answers[answers.length - 1];

  return <section className="session-card live-session">
    <div className="section-top"><div><span className="eyebrow">{plan.mode === 'assessment' && !interrupted ? 'Assessment' : 'Practice'} · {plan.focus === 'running' ? 'Running count' : 'True count'}</span><h2>Checkpoint {index + 1} <span className="muted">/ {plan.checkpoints.length}</span></h2></div>
      <button className="quiet-button" disabled={phase === 'paused'} onClick={pause}>Pause</button></div>
    <div className="session-progress" aria-label="Session progress"><div style={{ width: `${index / plan.checkpoints.length * 100}%` }} /></div>
    {phase === 'paused' ? <div className="pause-surface"><span className="eyebrow">Cards covered</span><h2>Session paused</h2>
      <p>{plan.mode === 'assessment' ? 'This attempt was interrupted. It can continue as practice, but will not count as an uninterrupted assessment.' : 'Your place is held while this page stays open. Resume when you can keep your attention on the cards.'}</p>
      {pausedPhase.current === 'dealing' && <p>The same cards will reappear. Do not count them a second time.</p>}
      <button className="primary-button" onClick={() => { if (plan.mode === 'assessment') setAssisted(true); if (pausedPhase.current === 'answer') responseStarted.current = performance.now(); setPhase(pausedPhase.current); }}>{plan.mode === 'assessment' ? 'Continue as practice' : 'Resume session'}</button>
      <button className="quiet-button" onClick={() => finish(answers, false)}>End this attempt</button></div>
    : <>
      {phase === 'ready' && <div className="ready-surface"><span className="eyebrow">{index === 0 ? 'New shoe · count starts at 0' : 'Carry your count forward'}</span><h2>{index === 0 ? 'Ready for the first cards?' : 'Ready for the next round?'}</h2>
        <p>{index === 0 ? 'Add low cards, subtract high cards, and ignore 7, 8, 9.' : 'Your previous answer is recorded. The running count continues from the same shoe.'}</p>
        <button className="primary-button" onClick={() => { setCursor(0); setPhase('dealing'); }}>Deal cards →</button></div>}
      {phase === 'dealing' && <div className="dealing-surface"><div className="dealt-pair" aria-label="Current cards">{checkpoint.cards.slice(cursor, cursor + 2).map(card => <LargeCard key={card.id} card={card} />)}</div><p className="muted">Keep the running count in mind.</p></div>}
      {phase === 'answer' && <form className="checkpoint-form" onSubmit={event => { event.preventDefault(); submit(); }}>
        <span className="eyebrow">Cards cleared</span><h3>What is your running count?</h3>
        <div className="answer-fields"><CountInput id="running-answer" label="Running count" value={runningGuess} onChange={setRunningGuess} autoFocus />
          {plan.focus === 'conversion' && <CountInput id="true-answer" label={`True count · ${checkpoint.decksRemaining} decks remaining`} value={trueGuess} onChange={setTrueGuess} />}</div>
        {plan.focus === 'conversion' && coached && <p className="fine-print">{COUNT_POLICY}</p>}
        <div className="button-row"><button type="submit" className="primary-button" disabled={parseCount(runningGuess) === null || (plan.focus === 'conversion' && parseCount(trueGuess) === null)}>Submit {coached ? 'answer' : 'and continue'}</button><button type="button" className="quiet-button" onClick={() => submit(true)}>I lost the count</button></div>
      </form>}
      {phase === 'feedback' && lastAnswer && <div className="feedback-surface" aria-live="polite"><span className={lastAnswer.runningCorrect && lastAnswer.trueCorrect !== false ? 'status-good' : 'status-review'}>{lastAnswer.runningCorrect && lastAnswer.trueCorrect !== false ? 'Correct' : 'Let’s review'}</span>
        <h2>Running count <span className="accent">{signed(checkpoint.runningCount)}</span></h2>
        <p>{explainAnswer(checkpoint, lastAnswer)}</p>
        {plan.focus === 'conversion' && <p>{signed(checkpoint.runningCount)} ÷ {checkpoint.decksRemaining} decks → true count <strong>{signed(checkpoint.trueCount)}</strong></p>}
        <div className="button-row"><button className="primary-button" onClick={() => advance()}>{index + 1 === plan.checkpoints.length ? 'Finish and review' : 'Continue with this count →'}</button><button className="quiet-button" onClick={() => { setReplay(!replay); setAssisted(true); }}>{replay ? 'Close replay' : 'Replay this segment'}</button></div>
        {replay && <Replay key={index} checkpoint={checkpoint} />}</div>}
      <div className="session-footer"><span>6 decks · new shuffle only at session start</span><button className="text-button" onClick={() => finish(answers, false)}>End session</button></div>
    </>}
  </section>;
}
