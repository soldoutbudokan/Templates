'use client';

import { useState, useCallback, useMemo, useEffect } from 'react';
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

export default function Home() {
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

  // One-time migration from old single bestStreak
  useEffect(() => {
    const legacy = localStorage.getItem('bestStreak');
    if (legacy) {
      const val = JSON.parse(legacy);
      if (typeof val === 'number' && val > 0) {
        setBestStreaks(prev => ({ ...prev, classic: Math.max(prev.classic, val) }));
      }
      localStorage.removeItem('bestStreak');
    }
  }, []);

  const [allStats, setAllStats] = useState<Record<GameMode, SessionStats>>(() =>
    Object.fromEntries(
      ALL_MODES.map(m => [m, emptyStats(bestStreaks[m] ?? 0)])
    ) as Record<GameMode, SessionStats>
  );

  // Create deck instance - recreate when deck count or deckKey changes
  const deck = useMemo(() => new Deck(deckCount), [deckCount, deckKey]);

  const handleDeckCountChange = useCallback((count: number) => {
    setDeckCount(count);
    setDeckKey(prev => prev + 1);
  }, []);

  const handleShuffle = useCallback(() => {
    deck.shuffle();
    setDeckKey(prev => prev + 1);
  }, [deck]);

  const handleRoundComplete = useCallback((correct: boolean) => {
    setAllStats(prev => {
      const modeStats = prev[mode];
      const newStreak = correct ? modeStats.currentStreak + 1 : 0;
      const newBest = Math.max(newStreak, modeStats.bestStreak);
      if (newBest > modeStats.bestStreak) {
        setBestStreaks(prev => ({ ...prev, [mode]: newBest }));
      }
      return {
        ...prev,
        [mode]: {
          roundsPlayed: modeStats.roundsPlayed + 1,
          correctGuesses: modeStats.correctGuesses + (correct ? 1 : 0),
          currentStreak: newStreak,
          bestStreak: newBest,
        },
      };
    });
  }, [mode, setBestStreaks]);

  return (
    <main className="min-h-screen p-4 flex flex-col">
      <div className="max-w-6xl w-full mx-auto flex-1 flex flex-col">
        {/* Header */}
        <h1 className="text-3xl font-bold text-center mb-6">
          Card Counting Trainer
        </h1>

        {/* Main Content */}
        <div className="flex-1 flex gap-6">
          {/* Game Area */}
          <div className="flex-1 flex flex-col min-h-[500px]">
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
          <div className="w-64 flex-shrink-0">
            <Controls
              mode={mode}
              onModeChange={(newMode) => { setMode(newMode); setDeckKey(prev => prev + 1); }}
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
              cardsRemaining={deck.remaining()}
              totalCards={deck.totalCards()}
              stats={allStats[mode]}
              onShuffle={handleShuffle}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
