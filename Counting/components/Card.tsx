'use client';

import { Card as CardType } from '@/lib/deck';
import PlayingCard from './PlayingCard';

interface CardProps {
  card: CardType;
  index?: number;
  flipped?: boolean;
  showFlipAnimation?: boolean;
}

/** Preserve the drill API while sharing one accessible, consistently drawn deck. */
export default function Card({ card, flipped = true, showFlipAnimation = true }: CardProps) {
  return <PlayingCard key={`${card.id}-${flipped}`} card={flipped ? card : null} size="md" dealt={showFlipAnimation} />;
}

export function LargeCard({ card, visible = true }: { card: CardType; visible?: boolean }) {
  return <div className="large-playing-card" aria-hidden={!visible || undefined} style={{ visibility: visible ? 'visible' : 'hidden' }}>
    <PlayingCard key={card.id} card={visible ? card : null} size="lg" />
  </div>;
}
