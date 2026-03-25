export type GameMode = 'classic' | 'speed-drill' | 'true-count' | 'multi-hand' | 'basic-strategy';
export type SpeedSetting = 1 | 0.75 | 0.5 | 0.25 | 0.15 | 0.1;

export interface SessionStats {
  roundsPlayed: number;
  correctGuesses: number;
  currentStreak: number;
  bestStreak: number;
}

export interface BettingAdvice {
  trueCount: number;
  recommendation: string;
  multiplier: number;
}
