'use client';

import { useState, useCallback, useMemo } from 'react';
import { Deck } from '@/lib/deck';
import { GameMode, SpeedSetting, SessionStats } from '@/lib/types';
import { usePersistedState } from '@/lib/usePersistedState';
import Controls from '@/components/Controls';
import GameBoard from '@/components/GameBoard';
import SpeedDrill from '@/components/SpeedDrill';
import TrueCountTrainer from '@/components/TrueCountTrainer';
import MultiHandBoard from '@/components/MultiHandBoard';

export default function Home() {
  const [mode, setMode] = useState<GameMode>('classic');
  const [deckCount, setDeckCount] = useState(6);
  const [speed, setSpeed] = useState<SpeedSetting>(1);
  const [hardMode, setHardMode] = useState(false);
  const [handCount, setHandCount] = useState(3);
  const [showBettingTips, setShowBettingTips] = useState(false);
  const [deckKey, setDeckKey] = useState(0);

  const [bestStreak, setBestStreak] = usePersistedState('bestStreak', 0);
  const [stats, setStats] = useState<SessionStats>({
    roundsPlayed: 0,
    correctGuesses: 0,
    currentStreak: 0,
    bestStreak: bestStreak,
  });

  // Create deck instance - recreate when deck count changes
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
    setStats(prev => {
      const newStreak = correct ? prev.currentStreak + 1 : 0;
      const newBest = Math.max(newStreak, prev.bestStreak);
      if (newBest > prev.bestStreak) {
        setBestStreak(newBest);
      }
      return {
        roundsPlayed: prev.roundsPlayed + 1,
        correctGuesses: prev.correctGuesses + (correct ? 1 : 0),
        currentStreak: newStreak,
        bestStreak: newBest,
      };
    });
  }, [setBestStreak]);

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
          </div>

          {/* Controls Sidebar */}
          <div className="w-64 flex-shrink-0">
            <Controls
              mode={mode}
              onModeChange={setMode}
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
              stats={stats}
              onShuffle={handleShuffle}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
