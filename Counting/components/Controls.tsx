'use client';

import { GameMode, SpeedSetting, SessionStats } from '@/lib/types';

interface ControlsProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  deckCount: number;
  onDeckCountChange: (count: number) => void;
  speed: SpeedSetting;
  onSpeedChange: (speed: SpeedSetting) => void;
  hardMode: boolean;
  onHardModeChange: (hardMode: boolean) => void;
  handCount: number;
  onHandCountChange: (count: number) => void;
  showBettingTips: boolean;
  onShowBettingTipsChange: (show: boolean) => void;
  cardsRemaining: number;
  totalCards: number;
  stats: SessionStats;
  onShuffle: () => void;
}

const MODE_INFO: { mode: GameMode; label: string; subtitle: string }[] = [
  { mode: 'classic', label: 'Classic', subtitle: 'Count a spread' },
  { mode: 'speed-drill', label: 'Speed Drill', subtitle: 'Cards flash by' },
  { mode: 'true-count', label: 'True Count', subtitle: 'Running → true' },
  { mode: 'multi-hand', label: 'Multi-Hand', subtitle: 'Table simulation' },
  { mode: 'basic-strategy', label: 'Basic Strategy', subtitle: 'Hit, stand, double' },
];

export default function Controls({
  mode,
  onModeChange,
  deckCount,
  onDeckCountChange,
  speed,
  onSpeedChange,
  hardMode,
  onHardModeChange,
  handCount,
  onHandCountChange,
  showBettingTips,
  onShowBettingTipsChange,
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
      {/* Mode Selector - 2x2 Grid */}
      <div>
        <label className="block text-sm font-medium mb-2 text-white/70">Mode</label>
        <div className="grid grid-cols-2 gap-2">
          {MODE_INFO.map(({ mode: m, label, subtitle }) => (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              className={`py-2 px-2 rounded text-sm font-medium transition-all text-left ${
                mode === m
                  ? 'bg-blue-600 text-white'
                  : 'bg-white/10 text-white/70 hover:bg-white/20'
              }`}
            >
              <div className="font-medium">{label}</div>
              <div className={`text-xs ${mode === m ? 'text-white/80' : 'text-white/40'}`}>
                {subtitle}
              </div>
            </button>
          ))}
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
          <div className="grid grid-cols-3 gap-2">
            {([1, 0.75, 0.5, 0.25, 0.15, 0.1] as SpeedSetting[]).map((s) => {
              const isHardSpeed = s <= 0.15;
              return (
                <button
                  key={s}
                  onClick={() => onSpeedChange(s)}
                  className={`py-2 px-2 rounded text-sm font-medium transition-all ${
                    speed === s
                      ? isHardSpeed ? 'bg-red-500 text-white' : 'bg-orange-500 text-black'
                      : isHardSpeed ? 'bg-red-500/20 text-red-300 hover:bg-red-500/30' : 'bg-white/10 text-white/70 hover:bg-white/20'
                  }`}
                >
                  {s}s
                </button>
              );
            })}
          </div>

          {/* Hard Mode Toggle */}
          <label className="flex items-center gap-2 mt-3 cursor-pointer">
            <input
              type="checkbox"
              checked={hardMode}
              onChange={(e) => onHardModeChange(e.target.checked)}
              className="w-4 h-4 rounded accent-red-500"
            />
            <span className="text-sm text-white/70">Hard Mode</span>
            {hardMode && <span className="text-xs text-red-400">(distractions + 40-60 cards)</span>}
          </label>
        </div>
      )}

      {/* Hand Count (only in multi-hand mode) */}
      {mode === 'multi-hand' && (
        <div>
          <label className="block text-sm font-medium mb-2 text-white/70">Hands at Table</label>
          <div className="flex gap-2">
            {[2, 3, 4].map((count) => (
              <button
                key={count}
                onClick={() => onHandCountChange(count)}
                className={`flex-1 py-2 px-3 rounded text-sm font-medium transition-all ${
                  handCount === count
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-white/70 hover:bg-white/20'
                }`}
              >
                {count}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Betting Tips Toggle */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={showBettingTips}
          onChange={(e) => onShowBettingTipsChange(e.target.checked)}
          className="w-4 h-4 rounded accent-green-500"
        />
        <span className="text-sm text-white/70">Show Betting Tips</span>
      </label>

      {/* Deck Progress (hidden in basic-strategy mode) */}
      {mode !== 'basic-strategy' && (
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
      )}

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

      {/* Streak Display */}
      <div className="pt-2 border-t border-white/10">
        <div className="flex justify-between text-sm">
          <span className="text-white/70">Current Streak</span>
          <span className={`font-medium ${
            stats.currentStreak >= 10 ? 'text-yellow-300' :
            stats.currentStreak >= 5 ? 'text-green-400' : 'text-white'
          }`}>
            {stats.currentStreak}{stats.currentStreak >= 5 ? ' 🔥' : ''}
          </span>
        </div>
        <div className="flex justify-between text-sm mt-1">
          <span className="text-white/70">Best Streak</span>
          <span className={`font-medium ${stats.bestStreak >= 10 ? 'text-yellow-300' : 'text-white'}`}>
            {stats.bestStreak}{stats.bestStreak >= 10 ? ' 🏆' : ''}
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
