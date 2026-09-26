'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import TrainingSession, { SessionReview } from '@/components/TrainingSession';
import FreePractice from '@/components/FreePractice';
import FullTableTest, { FullTableReview } from '@/components/FullTableTest';
import StrategyPractice from '@/components/StrategyPractice';
import QuickPlay from '@/components/QuickPlay';
import { TableSessionResult } from '@/lib/blackjack';
import { readTableHistory, tableResultLabel } from '@/lib/tableHistory';
import { assessmentEligible, readHistory, recommendation, sessionLabel, SessionResult } from '@/lib/training';

type Tab = 'play' | 'train' | 'table' | 'strategy' | 'practice' | 'progress' | 'learn';
const STORAGE_KEY = 'counting-coach-history-v1';
const TABLE_STORAGE_KEY = 'counting-table-history-v1';

function Learn() {
  return <section className="learn-grid">
    <article className="paper-card"><span className="eyebrow">01 · The count</span><h2>One total, through the shoe.</h2><p>Hi-Lo assigns +1 to 2–6, 0 to 7–9, and −1 to tens, face cards, and aces. Add the value of every card you see. Keep that total when the round ends.</p><p>A first round of +3 followed by a round of −1 leaves a running count of <strong>+2</strong>. Start again at zero only when the cards are shuffled.</p><div className="lesson-example">5 + K → +1 − 1 = 0<br />3 + 6 → +1 + 1 = +2</div><p>Recognize pairs that cancel. Count each exposed card once, including the dealer’s hole card when it is revealed.</p></article>
    <article className="paper-card"><span className="eyebrow">02 · The conversion</span><h2>Count per remaining deck.</h2><p>Divide the running count by an estimate of decks remaining. This trainer uses half-deck estimates and rounds the result down, toward negative infinity.</p><div className="lesson-example">+7 ÷ 4 decks = +1.75 → +1<br />−3 ÷ 2 decks = −1.5 → −2</div><p>The guided session supplies the deck estimate so you can isolate conversion. It does not measure your ability to estimate a physical discard tray.</p><p className="fine-print">Different strategies use different conversion conventions. Keep your lessons, indices, and betting schedule consistent.</p></article>
    <article className="paper-card"><span className="eyebrow">03 · How to practice</span><h2>Accuracy, then pace.</h2><p>Start with running-count practice. If the count slips, replay the segment, reconstruct it, and continue from the corrected total. Your first answer stays in your results.</p><p>Once your count is steady, add conversion. Revisit the same skill on a later day with fresh cards before increasing the pace.</p><p>Use a physical deck too: count pairs, pause between groups, and ask someone to stop you at unpredictable points. The final count of a complete Hi-Lo deck is always zero; that alone is not a useful test.</p></article>
    <article className="paper-card"><span className="eyebrow">04 · What the score means</span><h2>Evidence you can inspect.</h2><p>Practice provides the corrected count after each checkpoint. Assessment keeps all answers hidden until the end and runs to a 75% cut card in a six-deck shoe.</p><p>Leaving the tab or pausing marks the attempt interrupted. Continuing it is practice only. Results measure checkpoint accuracy under the listed conditions, not readiness for casino play.</p><p>Perfect strategy offers focused drills, reference charts, and a test. Full test combines actual rounds and cumulative counting under the same rules: six decks, dealer hits soft 17, double after split, and late surrender after the dealer checks for blackjack. Betting displays are example schedules, not bankroll recommendations.</p></article>
    <p className="source-note">References: <a href="https://www.qfit.com/CalculatingTrueCounts.htm" target="_blank" rel="noreferrer">True-count conventions</a> · <a href="https://wizardofodds.com/games/blackjack/strategy/4-decks/" target="_blank" rel="noreferrer">Basic strategy and H17 changes</a> · <a href="https://www.qfit.com/book/ModernBlackjackPage94.htm" target="_blank" rel="noreferrer">Combining counting skills</a></p>
  </section>;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>('play');
  const [active, setActive] = useState(false);
  const [results, setResults] = useState<SessionResult[]>([]);
  const [tableResults, setTableResults] = useState<TableSessionResult[]>([]);
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [tableStorageMessage, setTableStorageMessage] = useState('');
  const [confirmTableClear, setConfirmTableClear] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [storageMessage, setStorageMessage] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const dirty = useRef(false);
  const tableDirty = useRef(false);

  useEffect(() => {
    try { setResults(readHistory(localStorage.getItem(STORAGE_KEY))); }
    catch { setStorageMessage('Saved progress could not be read. You can still practice; new results will start a new history on this device.'); }
    try { setTableResults(readTableHistory(localStorage.getItem(TABLE_STORAGE_KEY))); }
    catch { setTableStorageMessage('Saved table results could not be read. New table results will start a new history on this device.'); }
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!loaded || !dirty.current) return;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, sessions: results })); setStorageMessage(''); }
    catch { setStorageMessage('Progress could not be saved on this device. Your results remain available until you close or reload this page.'); }
    dirty.current = false;
  }, [results, loaded]);

  useEffect(() => {
    if (!loaded || !tableDirty.current) return;
    try { localStorage.setItem(TABLE_STORAGE_KEY, JSON.stringify({ version: 1, sessions: tableResults })); setTableStorageMessage(''); }
    catch { setTableStorageMessage('Table results could not be saved on this device. They remain available until you close or reload this page.'); }
    tableDirty.current = false;
  }, [tableResults, loaded]);

  const save = useCallback((result: SessionResult) => {
    dirty.current = true;
    setResults(previous => [...previous.filter(item => item.id !== result.id), result].slice(-24));
  }, []);
  const saveTable = useCallback((result: TableSessionResult) => {
    tableDirty.current = true;
    setTableResults(previous => [...previous.filter(item => item.id !== result.id), result].slice(-24));
  }, []);
  const next = recommendation(results);
  const completed = results.filter(result => result.completed);
  const assessments = results.filter(assessmentEligible);
  const selectedResult = results.find(result => result.id === selected);
  const selectedTableResult = tableResults.find(result => result.id === selectedTable);

  return <main className={`training-app ${active ? 'session-active' : ''}`}>
    <div className="app-shell">
      <header className="app-header"><a className="wordmark" href="#main-content" aria-label="Card Counting Trainer"><span aria-hidden>♠</span><div>Counting<span className="wordmark-detail">HI-LO TRAINER</span></div></a><span className="header-note">A little sharper, every day.</span></header>
      <nav className="main-tabs" aria-label="Training sections">{([['play', 'Quick play'], ['train', 'Train'], ['table', 'Full test'], ['strategy', 'Perfect strategy'], ['practice', 'Practice drills'], ['progress', 'Progress'], ['learn', 'Learn']] as const).map(([key, label]) => <button key={key} className={tab === key ? 'active' : ''} aria-current={tab === key ? 'page' : undefined} disabled={active && key !== tab} onClick={() => setTab(key)}>{label}</button>)}</nav>
      {active && <p className="active-note">End this session to switch sections.</p>}
      {storageMessage && <p className="storage-message" role="status">{storageMessage}</p>}
      {tableStorageMessage && <p className="storage-message" role="status">{tableStorageMessage}</p>}
      <div id="main-content">
        {tab === 'play' && <QuickPlay onActiveChange={setActive} onOpenTraining={setTab} />}
        {tab === 'train' && <>
          {!active && <div className="next-session-note"><span className="eyebrow">Suggested focus</span><strong>{next.title}</strong><p>{next.reason}</p></div>}
          {loaded ? <TrainingSession suggestedFocus={next.focus} onSave={save} onActiveChange={setActive} /> : <p className="paper-card" role="status">Loading your progress…</p>}
        </>}
        {tab === 'table' && (loaded ? <FullTableTest onSave={saveTable} onActiveChange={setActive} /> : <p role="status">Loading table results…</p>)}
        {tab === 'strategy' && <StrategyPractice onActiveChange={setActive} />}
        {tab === 'practice' && <section className="practice-section"><div className="page-heading"><span className="eyebrow">Isolate a skill</span><h1>Practice drills</h1><p>Use these for focused repetition. Guided sessions add checkpoints, replay, and a saved review.</p></div><FreePractice /></section>}
        {tab === 'learn' && <><div className="page-heading"><span className="eyebrow">The essentials</span><h1>Build the right habits.</h1></div><Learn /></>}
        {tab === 'progress' && <section className="progress-section"><div className="page-heading"><span className="eyebrow">Your recent practice</span><h1>Progress you can inspect.</h1><p>The latest 24 guided sessions and 24 table attempts are saved on this device. Strategy tests include their own review; isolated practice-drill streaks are tracked separately.</p></div>
          <div className="paper-card"><h2>Full table results</h2><p>Playing decisions, running counts, and true counts stay separate. Partial and interrupted attempts are included.</p>
            {!tableResults.length ? <div><p>No table attempts yet. Test both skills together in a continuous shoe.</p><button className="primary-button" onClick={() => setTab('table')}>Open full test →</button></div> : <>
              <div className="history-list">{[...tableResults].reverse().map(result => <button key={result.id} className={`history-row ${selectedTable === result.id ? 'selected' : ''}`} onClick={() => setSelectedTable(selectedTable === result.id ? null : result.id)} aria-expanded={selectedTable === result.id}><div><strong>{tableResultLabel(result)}</strong><span>{new Date(result.endedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} · {result.checkpoints.length} checked rounds · {(result.config.speedMs / 1000).toFixed(2)}s per card</span></div><div><strong>{result.decisions.filter(d => d.correct).length}/{result.decisions.length} plays</strong><span>Running {result.checkpoints.filter(c => c.runningCorrect).length}/{result.checkpoints.length} · True {result.checkpoints.filter(c => c.trueCorrect).length}/{result.checkpoints.length}</span></div></button>)}</div>
              {selectedTableResult && <FullTableReview key={selectedTableResult.id} result={selectedTableResult} />}
              <div className="clear-progress">{confirmTableClear ? <><p>Clear saved table results on this device?</p><button className="quiet-button" onClick={() => { tableDirty.current = true; setTableResults([]); setSelectedTable(null); setConfirmTableClear(false); }}>Clear table history</button><button className="quiet-button" onClick={() => setConfirmTableClear(false)}>Keep table history</button></> : <button className="text-button" onClick={() => setConfirmTableClear(true)}>Clear table history</button>}</div>
            </>}
          </div>
          <h2 className="review-subheading">Guided counting sessions</h2>
          <div className="paper-card"><div className="metric-grid"><div><strong>{completed.length}</strong><p>Completed sessions</p></div><div><strong>{assessments.length}</strong><p>Uninterrupted assessments</p></div><div><strong>{results.filter(result => !result.completed).length}</strong><p>Partial attempts</p></div></div><p className="coach-note"><strong>{next.title}.</strong> {next.reason}</p></div>
          {!results.length ? <div className="empty-progress"><h2>Your first session sets a baseline.</h2><p>Complete a guided practice to see your checkpoint accuracy and the next recommended focus.</p><button className="primary-button" onClick={() => setTab('train')}>Start practicing →</button></div>
            : <><div className="history-list">{[...results].reverse().map(result => <button key={result.id} className={`history-row ${selected === result.id ? 'selected' : ''}`} onClick={() => setSelected(selected === result.id ? null : result.id)} aria-expanded={selected === result.id}><div><strong>{result.focus === 'running' ? 'Running count' : 'Running + true count'}</strong><span>{new Date(result.endedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} · {(result.speedMs / 1000).toFixed(2)}s per card · {sessionLabel(result)}{!result.completed ? ' · partial' : ''}{result.assisted ? ' · replay / assistance' : ''}</span></div><div><strong>{result.answers.filter(answer => answer.runningCorrect).length}/{result.answers.length}</strong><span>exact running counts</span></div></button>)}</div>
              {selectedResult && <div className="paper-card"><SessionReview key={selectedResult.id} result={selectedResult} /></div>}
              <div className="clear-progress">{confirmClear ? <><p>Clear the saved guided-session history on this device?</p><button className="quiet-button" onClick={() => { dirty.current = true; setResults([]); setSelected(null); setConfirmClear(false); }}>Clear saved history</button><button className="quiet-button" onClick={() => setConfirmClear(false)}>Keep it</button></> : <button className="text-button" onClick={() => setConfirmClear(true)}>Clear saved history</button>}</div>
            </>}
        </section>}
      </div>
      <footer className="app-footer"><span>Hi-Lo · count accurately, then add speed</span><span>Practice results stay on this device</span></footer>
    </div>
  </main>;
}
