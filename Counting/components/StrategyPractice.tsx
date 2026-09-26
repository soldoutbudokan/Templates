'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Action, formatAction, getCorrectAction, getHandTotal } from '@/lib/basicStrategy';
import {
  CATEGORY_LABEL, STRATEGY_CATEGORIES, STRATEGY_DEALERS, STRATEGY_PROFILE, STRATEGY_SOURCE,
  StrategyAnswer, StrategyCase, StrategyCategory, StrategyFocus, StrategyMode, StrategyResult,
  createStrategyCases, describeSituation, explainStrategyCase, gradeStrategyAnswer, legalStrategyActions,
  makeStrategyCase, strategyBreakdown, strategyCriterionMet, strategyRows,
} from '@/lib/strategyPractice';
import Card from './Card';

const ACTION_KEYS: Record<Action, string> = { hit: 'H', stand: 'S', double: 'D', split: 'P', surrender: 'R' };
const ACTIONS: Action[] = ['hit', 'stand', 'double', 'split', 'surrender'];
type Phase = 'setup' | 'question' | 'recorded' | 'complete';

function StrategyReference() {
  const [category, setCategory] = useState<StrategyCategory>('hard');
  return <details className="paper-card strategy-reference">
    <summary className="cursor-pointer font-semibold">Basic-strategy reference</summary>
    <p>Check surrender, then pairs, then the appropriate hard or soft total. These are base-strategy decisions; card-count deviations are outside this section.</p>
    <div className="button-row" aria-label="Reference category">
      {STRATEGY_CATEGORIES.map(value => <button key={value} className={`choice ${category === value ? 'active' : ''}`} aria-pressed={category === value} onClick={() => setCategory(value)}>{CATEGORY_LABEL[value]}</button>)}
    </div>
    <p className="fine-print">H = hit · S = stand · D = double, otherwise hit · Ds = double, otherwise stand · P = split · R = surrender, otherwise hit. After a hit, surrender and doubling are unavailable. Exceptions: for 17 vs A, surrender if allowed; otherwise stand. For 8,8 vs A, surrender if allowed; otherwise split.</p>
    <div className="overflow-x-auto mt-4" role="region" tabIndex={0} aria-label={`${CATEGORY_LABEL[category]} reference chart; scroll horizontally if needed`}>
      <table className="strategy-chart w-full text-center text-sm">
        <caption className="text-left mb-3">{CATEGORY_LABEL[category]} · H17 / DAS / late surrender · dealer upcard across the top</caption>
        <thead><tr><th scope="col" className="p-2 text-left">Your hand</th>{STRATEGY_DEALERS.map(dealer => <th className="p-2" key={dealer} scope="col">{dealer}</th>)}</tr></thead>
        <tbody>{strategyRows(category).map(row => <tr key={row.label} className="border-t border-[#d8ddd3]">
          <th className="p-2 text-left whitespace-nowrap" scope="row">{row.label}</th>
          {STRATEGY_DEALERS.map(dealer => {
            const situation = makeStrategyCase(category, row, dealer);
            const action = getCorrectAction(situation.player, situation.dealer, situation.options);
            const info = getHandTotal(situation.player);
            const label = action === 'double' && info.soft && info.total >= 18 ? 'Ds' : ACTION_KEYS[action];
            return <td key={dealer} className="p-2" aria-label={`${row.label} against ${dealer}: ${formatAction(action)}${label === 'Ds' ? ', otherwise stand' : action === 'double' ? ', otherwise hit' : ''}`}>{label}</td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
    <p className="fine-print">A hard total uses every ace as 1. A soft total has an ace counted as 11. Surrender is available only on the initial two cards, after any required dealer peek.</p>
    <p className="source-note">Reference: <a href={STRATEGY_SOURCE} target="_blank" rel="noreferrer">Michael Shackleford’s 4–8-deck strategy and H17 rules</a>.</p>
  </details>;
}

function StrategyReview({ result, onRepair }: { result: StrategyResult; onRepair: (cases: StrategyCase[]) => void }) {
  const correct = result.answers.filter(answer => answer.correct).length;
  const missed = result.answers.flatMap((answer, index) => answer.correct ? [] : [{ answer, situation: result.cases[index] }]);
  const met = strategyCriterionMet(result);
  return <section className="paper-card strategy-review" aria-label="Strategy results">
    <span className="eyebrow">{result.completed ? 'Session complete' : 'Session ended early'}</span>
    <h2>{met ? 'Every decision correct in this test.' : missed.length ? 'Your next practice is clear.' : result.answers.length ? 'A useful practice sample.' : 'No decisions recorded.'}</h2>
    <p>{result.mode === 'test' ? 'Strategy test' : 'Coached practice'} · {CATEGORY_LABEL[result.focus]} · {result.answers.length} of {result.cases.length} decisions answered</p>
    {result.mode === 'test' && <p className={met ? 'status-good' : 'status-review'}>{result.interrupted
      ? 'Interrupted test · the perfect-test criterion cannot be met on this attempt.'
      : met ? 'Perfect-test criterion met: a complete, uninterrupted test with every first answer correct.'
        : !result.completed ? 'Partial test · the perfect-test criterion was not assessed.' : 'Perfect-test criterion not met. Review the missed situations below.'}</p>}
    <div className="metric-grid">
      <div><strong>{correct}<span> / {result.answers.length}</span></strong><p>Correct first answers</p></div>
      <div><strong>{result.answers.length ? `${Math.round(correct / result.answers.length * 100)}%` : '—'}</strong><p>Accuracy on this sample</p></div>
    </div>
    <table className="strategy-breakdown w-full text-left">
      <caption className="text-left font-semibold mb-3">Accuracy by hand category</caption>
      <thead><tr><th scope="col" className="py-2">Category</th><th scope="col" className="py-2">Correct / answered</th><th scope="col" className="py-2">Accuracy</th></tr></thead>
      <tbody>{strategyBreakdown(result).map(row => <tr key={row.category} className="border-t border-[#d8ddd3]">
        <th scope="row" className="py-3 font-normal">{CATEGORY_LABEL[row.category]}</th><td>{row.correct} / {row.attempted}</td><td>{row.attempted ? `${Math.round(row.correct / row.attempted * 100)}%` : 'Not sampled'}</td>
      </tr>)}</tbody>
    </table>
    <p className="coach-note">{missed.length
      ? 'Practice the situations below, then take a fresh mixed test. Repeating reviewed hands is coached practice, and does not change this score.'
      : 'Try a fresh mixed test on another day. A small perfect sample does not establish mastery of every situation or readiness for live play.'}</p>
    {missed.length > 0 && <>
      <div className="button-row"><button className="primary-button" onClick={() => onRepair(missed.map(entry => entry.situation))}>Practice these {missed.length} missed situations</button></div>
      <h3 className="text-lg font-semibold mt-6">Missed situations</h3>
      {missed.map(({ answer, situation }) => <details key={situation.id} className="py-4 border-b border-[#d8ddd3]">
        <summary className="cursor-pointer">{describeSituation(situation)} · you chose {formatAction(answer.action)} · correct: {formatAction(answer.expected)}</summary>
        <p>{explainStrategyCase(situation)}</p>
      </details>)}
    </>}
    <p className="fine-print">This app’s criterion describes this sample only. Questions balance chart coverage rather than natural hand frequency. “Perfect basic strategy” means following the selected base strategy; no count deviations are scored.</p>
  </section>;
}

export default function StrategyPractice({ onActiveChange }: { onActiveChange?: (active: boolean) => void }) {
  const [focus, setFocus] = useState<StrategyFocus>('mixed');
  const [mode, setMode] = useState<StrategyMode>('practice');
  const [length, setLength] = useState<20 | 30>(20);
  const [phase, setPhase] = useState<Phase>('setup');
  const [cases, setCases] = useState<StrategyCase[]>([]);
  const [answers, setAnswers] = useState<StrategyAnswer[]>([]);
  const [index, setIndex] = useState(0);
  const [interrupted, setInterrupted] = useState(false);
  const [result, setResult] = useState<StrategyResult | null>(null);
  const answered = useRef(false);
  const activeTest = useRef(false);
  const interruptedRef = useRef(false);
  const startedAt = useRef(0);
  const questionHeading = useRef<HTMLHeadingElement>(null);
  const nextButton = useRef<HTMLButtonElement>(null);
  const active = phase === 'question' || phase === 'recorded';
  const situation = cases[index];
  const latest = answers[index];

  useEffect(() => {
    activeTest.current = active && mode === 'test';
    onActiveChange?.(active);
  }, [active, mode, onActiveChange]);
  useEffect(() => () => { onActiveChange?.(false); }, [onActiveChange]);
  useEffect(() => {
    const markInterrupted = () => {
      if (!activeTest.current) return;
      interruptedRef.current = true;
      setInterrupted(true);
    };
    const visibility = () => { if (document.hidden) markInterrupted(); };
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('pagehide', markInterrupted);
    return () => { document.removeEventListener('visibilitychange', visibility); window.removeEventListener('pagehide', markInterrupted); };
  }, []);
  useEffect(() => {
    if (phase === 'question') questionHeading.current?.focus();
    if (phase === 'recorded') nextButton.current?.focus();
  }, [phase, index]);

  function start(repair?: StrategyCase[]) {
    const nextMode = repair ? 'practice' : mode;
    const nextCases = repair ?? createStrategyCases(Math.floor(Math.random() * 0x100000000), focus, length);
    setMode(nextMode);
    setCases(nextCases);
    setAnswers([]);
    setIndex(0);
    setInterrupted(false);
    interruptedRef.current = false;
    setResult(null);
    answered.current = false;
    activeTest.current = nextMode === 'test';
    startedAt.current = performance.now();
    setPhase('question');
  }

  const submit = useCallback((action: Action) => {
    if (phase !== 'question' || answered.current || !situation || !legalStrategyActions(situation).includes(action)) return;
    answered.current = true;
    const answer = gradeStrategyAnswer(situation, action, performance.now() - startedAt.current);
    setAnswers(previous => [...previous, answer]);
    setPhase('recorded');
  }, [phase, situation]);

  function finish(completed: boolean) {
    activeTest.current = false;
    setResult({ mode, focus, cases, answers, completed, interrupted: interruptedRef.current });
    setPhase('complete');
  }

  useEffect(() => {
    if (phase !== 'question') return;
    const keydown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || target?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName ?? '')) return;
      const action = ACTIONS.find(value => ACTION_KEYS[value].toLowerCase() === event.key.toLowerCase());
      if (!action) return;
      event.preventDefault();
      submit(action);
    };
    window.addEventListener('keydown', keydown);
    return () => window.removeEventListener('keydown', keydown);
  }, [phase, submit]);

  function next() {
    if (index === cases.length - 1) { finish(true); return; }
    setIndex(previous => previous + 1);
    answered.current = false;
    startedAt.current = performance.now();
    setPhase('question');
  }

  return <div className="strategy-practice space-y-5">
    <div className="page-heading">
      <span className="eyebrow">Decisions before deviations</span>
      <h1>Perfect basic strategy</h1>
      <p>Learn the right play for each hand, then test your first answers without feedback. This section uses basic strategy, independent of the running count.</p>
      <p className="fine-print">{STRATEGY_PROFILE}. Against 10 or A, assume the dealer has checked and does not have blackjack. Split aces receive one card.</p>
    </div>

    {phase === 'setup' && <section className="paper-card" aria-label="Strategy session setup">
      <h2>Build accuracy, one decision at a time.</h2>
      <fieldset className="mb-6"><legend className="font-semibold mb-3">Choose your session</legend>
        <div className="choice-grid">
          <button className={`choice ${mode === 'practice' ? 'active' : ''}`} aria-pressed={mode === 'practice'} onClick={() => setMode('practice')}><strong>Guided practice</strong><span>Immediate explanation after each first answer.</span></button>
          <button className={`choice ${mode === 'test' ? 'active' : ''}`} aria-pressed={mode === 'test'} onClick={() => setMode('test')}><strong>Strategy test</strong><span>No answers or reference chart until you finish.</span></button>
        </div>
      </fieldset>
      <fieldset className="mb-6"><legend className="font-semibold mb-3">Hand categories</legend><div className="flex flex-wrap gap-2">
        {(['hard', 'soft', 'pairs', 'mixed'] as StrategyFocus[]).map(value => <button key={value} className={`choice ${focus === value ? 'active' : ''}`} aria-pressed={focus === value} onClick={() => setFocus(value)}>{CATEGORY_LABEL[value]}</button>)}
      </div></fieldset>
      <fieldset><legend className="font-semibold mb-3">Decisions per session</legend><div className="flex gap-2">
        {([20, 30] as const).map(value => <button key={value} className={`choice ${length === value ? 'active' : ''}`} aria-pressed={length === value} onClick={() => setLength(value)}>{value} decisions</button>)}
      </div></fieldset>
      <p>{mode === 'test' ? 'Criterion: every first answer correct in a complete, uninterrupted test. Leaving or hiding this tab permanently marks the attempt interrupted; you can still finish and review it.' : 'Practice covers different chart situations instead of dealing a random shoe. Only your first answer counts; explanations do not overwrite it.'}</p>
      <div className="button-row"><button className="primary-button" onClick={() => start()}>Start {mode === 'test' ? `${length}-decision test` : 'guided practice'} <span aria-hidden="true">→</span></button></div>
    </section>}

    {active && situation && <section className="session-card" aria-label="Strategy decision">
      <div className="section-top"><div><span className="eyebrow">{mode === 'test' ? 'Strategy test' : 'Guided practice'}</span><h2 ref={questionHeading} tabIndex={-1}>Decision {index + 1} of {cases.length}</h2></div>
        <button className="quiet-button" onClick={() => finish(false)}>End session</button>
      </div>
      <div className="session-progress" aria-label={`${answers.length} of ${cases.length} decisions answered`}><div style={{ width: `${answers.length / cases.length * 100}%` }} /></div>
      {interrupted && <p className="status-review mt-4" role="status">This test was interrupted when the tab was hidden. Continue for practice; this attempt cannot meet the test criterion.</p>}
      <div className="strategy-board flex flex-col items-center gap-6 py-8">
        <div className="text-center"><p className="mb-3">Dealer shows</p><div role="img" aria-label={`Dealer ${situation.dealer.value}`}><div aria-hidden="true"><Card card={situation.dealer} showFlipAnimation={false} /></div></div></div>
        <span className="meta" aria-hidden="true">against</span>
        <div className="text-center"><p className="mb-3">Your hand</p><div className="strategy-hand flex gap-3 justify-center" role="img" aria-label={`Your cards: ${situation.player.map(card => card.value).join(', ')}`}>
          {situation.player.map(card => <div key={card.id} aria-hidden="true"><Card card={card} showFlipAnimation={false} /></div>)}
        </div></div>
      </div>
      <p className="text-center text-sm mb-4">{situation.player.length > 2 ? 'You have already hit. Double, split, and surrender are unavailable.' : 'Initial two-card hand · late surrender available · dealer blackjack ruled out'}</p>
      <div className="strategy-actions flex flex-wrap justify-center gap-3" aria-label="Choose a playing action">
        {ACTIONS.map(action => <button key={action} className="quiet-button" aria-keyshortcuts={ACTION_KEYS[action].toLowerCase()} disabled={phase !== 'question' || !legalStrategyActions(situation).includes(action)} onClick={() => submit(action)}>{formatAction(action)} <kbd className="text-xs opacity-70">{ACTION_KEYS[action]}</kbd></button>)}
      </div>
      {phase === 'question' && <p className="fine-print text-center mt-4">Keyboard: H hit · S stand · D double · P split · R surrender</p>}
      {phase === 'recorded' && latest && <div className="feedback-surface" role="status" aria-live="polite">
        {mode === 'practice' ? <><h3 className={latest.correct ? 'status-good text-xl' : 'status-review text-xl'}>{latest.correct ? 'Correct.' : `Correct play: ${formatAction(latest.expected)}.`}</h3><p>Your first answer: {formatAction(latest.action)}.</p><p>{explainStrategyCase(situation)}</p></> : <><h3 className="text-xl">Answer recorded.</h3><p>You chose {formatAction(latest.action)}. Feedback will appear at the end.</p></>}
        <div className="button-row"><button ref={nextButton} className="primary-button" onClick={next}>{index === cases.length - 1 ? 'See results' : 'Next decision'} <span aria-hidden="true">→</span></button></div>
      </div>}
    </section>}

    {phase === 'complete' && result && <><StrategyReview result={result} onRepair={start} /><div className="button-row"><button className="primary-button" onClick={() => setPhase('setup')}>Choose another session</button><button className="quiet-button" onClick={() => start()}>Start a fresh {mode === 'test' ? 'test' : 'practice'}</button></div></>}
    {!(active && mode === 'test') && <StrategyReference />}
  </div>;
}
