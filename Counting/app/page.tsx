'use client';

import { useState, useCallback, useMemo } from 'react';
import { Deck } from '@/lib/deck';
import Controls, { GameMode, SpeedSetting } from '@/components/Controls';
import GameBoard from '@/components/GameBoard';
import SpeedDrill from '@/components/SpeedDrill';

export default function Home() {
  const [mode, setMode] = useState<GameMode>('classic');
  const [deckCount, setDeckCount] = useState(6);
  const [speed, setSpeed] = useState<SpeedSetting>(1);
  const [stats, setStats] = useState({ roundsPlayed: 0, correctGuesses: 0 });
  const [deckKey, setDeckKey] = useState(0);

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
    setStats(prev => ({
      roundsPlayed: prev.roundsPlayed + 1,
      correctGuesses: prev.correctGuesses + (correct ? 1 : 0),
    }));
  }, []);

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
            {mode === 'classic' ? (
              <GameBoard
                key={`classic-${deckKey}`}
                deck={deck}
                onRoundComplete={handleRoundComplete}
              />
            ) : (
              <SpeedDrill
                key={`speed-${deckKey}`}
                deck={deck}
                speed={speed}
                onRoundComplete={handleRoundComplete}
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
