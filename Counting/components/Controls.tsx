'use client';

export type GameMode = 'classic' | 'speed-drill';
export type SpeedSetting = 1 | 0.75 | 0.5 | 0.25;

interface ControlsProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  deckCount: number;
  onDeckCountChange: (count: number) => void;
  speed: SpeedSetting;
  onSpeedChange: (speed: SpeedSetting) => void;
  cardsRemaining: number;
  totalCards: number;
  stats: {
    roundsPlayed: number;
    correctGuesses: number;
  };
  onShuffle: () => void;
}

export default function Controls({
  mode,
  onModeChange,
  deckCount,
  onDeckCountChange,
  speed,
  onSpeedChange,
  cardsRemaining,
  totalCards,
  stats,
  onShuffle,
}: ControlsProps) {
  const percentRemaining = (cardsRemaining / totalCards) * 100;
  const accuracy = stats.roundsPlayed > 0
    ? Math.round((stats.correctGuesses / stats.roundsPlayed) * 100)
    : 0;

  const getProgressClass = () => {
    if (percentRemaining < 25) return 'danger';
    if (percentRemaining < 50) return 'warning';
    return '';
  };

  return (
    <div className="settings-panel space-y-4">
      {/* Mode Toggle */}
      <div>
        <label className="block text-sm font-medium mb-2 text-white/70">Mode</label>
        <div className="flex gap-2">
          <button
            onClick={() => onModeChange('classic')}
            className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-all ${
              mode === 'classic'
                ? 'bg-blue-600 text-white'
                : 'bg-white/10 text-white/70 hover:bg-white/20'
            }`}
          >
            Classic
          </button>
          <button
            onClick={() => onModeChange('speed-drill')}
            className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-all ${
              mode === 'speed-drill'
                ? 'bg-blue-600 text-white'
                : 'bg-white/10 text-white/70 hover:bg-white/20'
            }`}
          >
            Speed Drill
          </button>
        </div>
      </div>

      {/* Deck Count */}
      <div>
        <label className="block text-sm font-medium mb-2 text-white/70">Decks in Shoe</label>
        <div className="flex gap-2">
          {[1, 2, 6, 8].map((count) => (
            <button
              key={count}
              onClick={() => onDeckCountChange(count)}
              className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-all ${
                deckCount === count
                  ? 'bg-green-600 text-white'
                  : 'bg-white/10 text-white/70 hover:bg-white/20'
              }`}
            >
              {count}
            </button>
          ))}
        </div>
      </div>

      {/* Speed Setting (only in speed drill mode) */}
      {mode === 'speed-drill' && (
        <div>
          <label className="block text-sm font-medium mb-2 text-white/70">
            Speed (seconds per card)
          </label>
          <div className="flex gap-2">
            {[1, 0.75, 0.5, 0.25].map((s) => (
              <button
                key={s}
                onClick={() => onSpeedChange(s as SpeedSetting)}
                className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-all ${
                  speed === s
                    ? 'bg-orange-500 text-black'
                    : 'bg-white/10 text-white/70 hover:bg-white/20'
                }`}
              >
                {s}s
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Deck Progress */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-white/70">Cards Remaining</span>
          <span className="font-medium">{cardsRemaining} / {totalCards}</span>
        </div>
        <div className="deck-progress">
          <div
            className={`deck-progress-bar ${getProgressClass()}`}
            style={{ width: `${percentRemaining}%` }}
          />
        </div>
        {percentRemaining < 25 && (
          <button
            onClick={onShuffle}
            className="mt-2 w-full py-1 px-3 bg-yellow-500/20 text-yellow-300 rounded text-sm hover:bg-yellow-500/30 transition-all"
          >
            Shuffle Deck
          </button>
        )}
      </div>

      {/* Session Stats */}
      <div className="pt-2 border-t border-white/10">
        <div className="flex justify-between text-sm">
          <span className="text-white/70">Rounds Played</span>
          <span className="font-medium">{stats.roundsPlayed}</span>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span className="text-white/70">Accuracy</span>
          <span className={`font-medium ${accuracy >= 80 ? 'text-green-400' : accuracy >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
            {accuracy}%
          </span>
        </div>
      </div>

      {/* Keyboard Shortcuts Help */}
      <div className="pt-2 border-t border-white/10 text-xs text-white/50">
        <p><kbd className="px-1 py-0.5 bg-white/10 rounded">Enter</kbd> Submit guess</p>
        <p className="mt-1"><kbd className="px-1 py-0.5 bg-white/10 rounded">Space</kbd> New hand</p>
      </div>
    </div>
  );
}
