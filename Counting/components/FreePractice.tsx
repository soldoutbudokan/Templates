'use client';

import { useState, useCallback, useMemo, useEffect, useSyncExternalStore } from 'react';
import { Deck } from '@/lib/deck';
import { GameMode, SpeedSetting, SessionStats } from '@/lib/types';
import { usePersistedState } from '@/lib/usePersistedState';
import Controls from '@/components/Controls';
import GameBoard from '@/components/GameBoard';
import SpeedDrill from '@/components/SpeedDrill';
import TrueCountTrainer from '@/components/TrueCountTrainer';
import MultiHandBoard from '@/components/MultiHandBoard';
import BasicStrategyTrainer from '@/components/BasicStrategyTrainer';

const ALL_MODES: GameMode[] = ['classic', 'speed-drill', 'true-count', 'multi-hand', 'basic-strategy'];

function emptyStats(bestStreak = 0): SessionStats {
  return { roundsPlayed: 0, correctGuesses: 0, currentStreak: 0, bestStreak };
}

export default function FreePractice() {
  const [mode, setMode] = useState<GameMode>('classic');
  const [deckCount, setDeckCount] = useState(6);
  const [speed, setSpeed] = useState<SpeedSetting>(1);
  const [hardMode, setHardMode] = useState(false);
  const [handCount, setHandCount] = useState(3);
  const [showBettingTips, setShowBettingTips] = useState(false);
  const [deckKey, setDeckKey] = useState(0);

  const [bestStreaks, setBestStreaks] = usePersistedState<Record<GameMode, number>>(
    'bestStreaks',
    { classic: 0, 'speed-drill': 0, 'true-count': 0, 'multi-hand': 0, 'basic-strategy': 0 },
  );

  // Read the legacy key defensively; storage may be disabled.
  useEffect(() => {
    try {
      const legacy = localStorage.getItem('bestStreak');
      if (legacy) {
        const val: unknown = JSON.parse(legacy);
        if (typeof val === 'number' && Number.isFinite(val) && val > 0) {
          setBestStreaks(prev => ({ ...prev, classic: Math.max(prev.classic, val) }));
        }
        localStorage.removeItem('bestStreak');
      }
    } catch { /* Practice remains available without storage. */ }
  }, [setBestStreaks]);

  const [allStats, setAllStats] = useState<Record<GameMode, SessionStats>>(() =>
    Object.fromEntries(
      ALL_MODES.map(m => [m, emptyStats(bestStreaks[m] ?? 0)])
    ) as Record<GameMode, SessionStats>
  );

  // Create deck instance - recreate when deck count or deckKey changes
  const deck = useMemo(() => new Deck(deckCount), [deckCount, deckKey]);
  const cardsRemaining = useSyncExternalStore(deck.subscribe, deck.remainingSnapshot, deck.totalSnapshot);

  const handleDeckCountChange = useCallback((count: number) => {
    setDeckCount(count);
    setDeckKey(prev => prev + 1);
  }, []);

  const handleShuffle = useCallback(() => {
    deck.shuffle();
    setDeckKey(prev => prev + 1);
  }, [deck]);

  // Persistence happens outside React state updater functions.
  useEffect(() => {
    setBestStreaks(previous => {
      const next = { ...previous };
      for (const m of ALL_MODES) next[m] = Math.max(previous[m] ?? 0, allStats[m].bestStreak);
      return next;
    });
  }, [allStats, setBestStreaks]);

  const handleRoundComplete = useCallback((correct: boolean) => {
    setAllStats(previous => {
      const stats = previous[mode];
      const streak = correct ? stats.currentStreak + 1 : 0;
      return { ...previous, [mode]: { roundsPlayed: stats.roundsPlayed + 1,
        correctGuesses: stats.correctGuesses + Number(correct), currentStreak: streak,
        bestStreak: Math.max(streak, stats.bestStreak) } };
    });
  }, [mode]);

  return (
    <section className="legacy-practice flex flex-col">
      <div className="max-w-6xl w-full mx-auto flex-1 flex flex-col">
        <p className="mb-5 text-white/70">Choose an isolated exercise. These practice streaks are separate from guided-session progress.</p>
        {/* Main Content */}
        <div className="flex-1 flex flex-col lg:flex-row gap-6">
          {/* Game Area */}
          <div className="flex-1 min-w-0 flex flex-col min-h-[420px]">
            {mode === 'classic' && (
              <GameBoard
                key={`classic-${deckKey}`}
                deck={deck}
                onRoundComplete={handleRoundComplete}
                showBettingTips={showBettingTips}
              />
            )}
            {mode === 'speed-drill' && (
              <SpeedDrill
                key={`speed-${deckKey}`}
                deck={deck}
                speed={speed}
                hardMode={hardMode}
                onRoundComplete={handleRoundComplete}
                showBettingTips={showBettingTips}
              />
            )}
            {mode === 'true-count' && (
              <TrueCountTrainer
                key={`tc-${deckKey}`}
                deck={deck}
                onRoundComplete={handleRoundComplete}
                showBettingTips={showBettingTips}
              />
            )}
            {mode === 'multi-hand' && (
              <MultiHandBoard
                key={`mh-${deckKey}`}
                deck={deck}
                handCount={handCount}
                onRoundComplete={handleRoundComplete}
                showBettingTips={showBettingTips}
              />
            )}
            {mode === 'basic-strategy' && (
              <BasicStrategyTrainer
                key={`bs-${deckKey}`}
                deck={deck}
                onRoundComplete={handleRoundComplete}
                showBettingTips={showBettingTips}
              />
            )}
          </div>

          {/* Controls Sidebar */}
          <div className="w-full lg:w-64 flex-shrink-0">
            <Controls
              mode={mode}
              onModeChange={(newMode) => { setMode(newMode); if (newMode === 'basic-strategy') setDeckCount(6); setDeckKey(prev => prev + 1); }}
              deckCount={deckCount}
              onDeckCountChange={handleDeckCountChange}
              speed={speed}
              onSpeedChange={setSpeed}
              hardMode={hardMode}
              onHardModeChange={setHardMode}
              handCount={handCount}
              onHandCountChange={setHandCount}
              showBettingTips={showBettingTips}
              onShowBettingTipsChange={setShowBettingTips}
              cardsRemaining={cardsRemaining}
              totalCards={deck.totalCards()}
              stats={{ ...allStats[mode], bestStreak: Math.max(allStats[mode].bestStreak, bestStreaks[mode] ?? 0) }}
              onShuffle={handleShuffle}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
