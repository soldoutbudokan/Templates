/** A small, accessible deck of paper cards, drawn entirely with HTML and SVG. */
import type { CSSProperties } from 'react';
import type { Card, Suit, Value } from '@/lib/deck';
import styles from './PlayingCard.module.css';

export interface PlayingCardProps {
  card: Card | null;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  dealt?: boolean;
}

const SUIT_NAMES: Record<Suit, string> = {
  '♠': 'spades', '♥': 'hearts', '♦': 'diamonds', '♣': 'clubs',
};
const RANK_NAMES: Partial<Record<Value, string>> = {
  A: 'Ace', J: 'Jack', Q: 'Queen', K: 'King',
};
const SUIT_PATHS: Record<Suit, string> = {
  '♠': 'M12 1.5C9.8 5.2 3 8.7 3 13.3a5 5 0 0 0 8 4c-.3 2.4-1.3 4.1-2.4 5.2h6.8c-1.1-1.1-2.1-2.8-2.4-5.2a5 5 0 0 0 8-4c0-4.6-6.8-8.1-9-11.8Z',
  '♥': 'M12 22C9 18.8 1.8 13.8 1.8 7.8A5.5 5.5 0 0 1 12 4.9a5.5 5.5 0 0 1 10.2 2.9C22.2 13.8 15 18.8 12 22Z',
  '♦': 'M12 1 22 12 12 23 2 12Z',
  '♣': 'M12 1.5a5 5 0 0 0-4.2 7.7A5.2 5.2 0 1 0 11 18c-.3 1.8-1.1 3.3-2.4 4.5h6.8c-1.3-1.2-2.1-2.7-2.4-4.5a5.2 5.2 0 1 0 3.2-8.8A5 5 0 0 0 12 1.5Z',
};

/** Coordinates occupy the pip field between the corner indices. */
const PIPS: Record<string, [number, number][]> = {
  '2': [[50, 0], [50, 100]],
  '3': [[50, 0], [50, 50], [50, 100]],
  '4': [[0, 0], [100, 0], [0, 100], [100, 100]],
  '5': [[0, 0], [100, 0], [50, 50], [0, 100], [100, 100]],
  '6': [[0, 0], [100, 0], [0, 50], [100, 50], [0, 100], [100, 100]],
  '7': [[0, 0], [100, 0], [50, 25], [0, 50], [100, 50], [0, 100], [100, 100]],
  '8': [[0, 0], [100, 0], [50, 25], [0, 50], [100, 50], [50, 75], [0, 100], [100, 100]],
  '9': [[0, 0], [100, 0], [0, 33.33], [100, 33.33], [50, 50], [0, 66.67], [100, 66.67], [0, 100], [100, 100]],
  '10': [[0, 0], [100, 0], [50, 16.67], [0, 33.33], [100, 33.33], [0, 66.67], [100, 66.67], [50, 83.33], [0, 100], [100, 100]],
};

function SuitMark({ suit, className }: { suit: Suit; className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false">
      <path d={SUIT_PATHS[suit]} />
    </svg>
  );
}

function CourtMark({ suit, value }: { suit: Suit; value: Value }) {
  return (
    <div className={styles.court}>
      <svg className={styles.courtFrame} viewBox="0 0 60 90" fill="none" aria-hidden="true" focusable="false">
        <path d="m30 2 27 43-27 43L3 45Z" fill="currentColor" fillOpacity=".035" stroke="currentColor" strokeOpacity=".32" />
        <path d="m30 9 22 36-22 36L8 45Z" stroke="currentColor" strokeOpacity=".2" />
        <path d="M15 45h30M30 24v42" stroke="currentColor" strokeOpacity=".12" />
        <circle cx="30" cy="45" r="17" fill="#fffdf7" stroke="currentColor" strokeOpacity=".35" />
        <circle cx="30" cy="45" r="14" stroke="currentColor" strokeOpacity=".15" />
        <path d="m25 18 1-5 4 3 4-3 1 5h-10Zm0 54 1 5 4-3 4 3 1-5H25Z" fill="currentColor" fillOpacity=".7" />
      </svg>
      <SuitMark suit={suit} className={styles.courtSuit} />
      <span className={styles.courtName}>{RANK_NAMES[value]}</span>
    </div>
  );
}

function CardBack() {
  return (
    <span className={styles.backPattern} aria-hidden="true">
      <svg className={styles.backOrnament} viewBox="0 0 72 104" fill="none" focusable="false">
        <path d="m36 11 25 41-25 41L11 52Z" fill="currentColor" fillOpacity=".035" stroke="currentColor" strokeWidth=".65" />
        <path d="m36 18 21 34-21 34L15 52Z" stroke="currentColor" strokeWidth=".45" />
        <path d="M18 52h36M36 25v54" stroke="currentColor" strokeOpacity=".45" strokeWidth=".5" />
        <circle cx="36" cy="52" r="14" fill="#155248" stroke="currentColor" strokeWidth=".6" />
        <circle cx="36" cy="52" r="11" stroke="currentColor" strokeOpacity=".5" strokeWidth=".5" />
        <path d="m36 43 7 9-7 9-7-9Z" fill="currentColor" fillOpacity=".8" />
        <path d="m36 3 2 3-2 3-2-3Zm0 92 2 3-2 3-2-3Z" fill="currentColor" />
        <path d="M7 17V7h10M55 7h10v10M65 87v10H55M17 97H7V87" stroke="currentColor" strokeOpacity=".7" strokeWidth=".6" />
      </svg>
    </span>
  );
}

export default function PlayingCard({ card, size = 'md', className = '', dealt = false }: PlayingCardProps) {
  const red = card?.suit === '♥' || card?.suit === '♦';
  const label = card ? `${RANK_NAMES[card.value] ?? card.value} of ${SUIT_NAMES[card.suit]}` : 'Face-down card';

  return (
    <span
      role="img"
      aria-label={label}
      className={`${styles.card} ${styles[size]} ${card ? styles.face : styles.back} ${red ? styles.red : ''} ${dealt ? styles.dealt : ''} ${className}`}
    >
      {card ? (
        <span className={styles.faceContents} aria-hidden="true">
          <span className={styles.corner}>
            <span className={styles.rank}>{card.value}</span>
            <SuitMark suit={card.suit} />
          </span>
          <span className={`${styles.corner} ${styles.bottomCorner}`}>
            <span className={styles.rank}>{card.value}</span>
            <SuitMark suit={card.suit} />
          </span>
          {card.value === 'A' ? (
            <span className={styles.ace}><SuitMark suit={card.suit} /></span>
          ) : PIPS[card.value] ? (
            <span className={styles.pipField}>
              {PIPS[card.value].map(([x, y], index) => (
                <span
                  key={index}
                  className={`${styles.pip} ${y > 50 ? styles.invertedPip : ''}`}
                  style={{ left: `${x}%`, top: `${y}%` } as CSSProperties}
                >
                  <SuitMark suit={card.suit} />
                </span>
              ))}
            </span>
          ) : (
            <CourtMark suit={card.suit} value={card.value} />
          )}
        </span>
      ) : <CardBack />}
    </span>
  );
}
