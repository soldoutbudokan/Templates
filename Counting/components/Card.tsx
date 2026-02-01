'use client';

import { Card as CardType, isRedSuit } from '@/lib/deck';

interface CardProps {
  card: CardType;
  index?: number;
  flipped?: boolean;
  showFlipAnimation?: boolean;
}

export default function Card({
  card,
  index = 0,
  flipped = true,
  showFlipAnimation = true
}: CardProps) {
  const isRed = isRedSuit(card.suit);
  const staggerClass = showFlipAnimation ? `stagger-${Math.min(index % 10 + 1, 10)}` : '';

  return (
    <div
      className={`card-container w-14 h-20 ${showFlipAnimation ? 'animate-flip-in' : ''} ${staggerClass}`}
      style={{
        animationFillMode: 'backwards',
        animationDelay: showFlipAnimation ? `${index * 0.03}s` : '0s'
      }}
    >
      <div className={`card-inner ${flipped ? 'flipped' : ''}`}>
        {/* Card Back */}
        <div className="card-back shadow-lg" />

        {/* Card Front */}
        <div className="card-front shadow-lg">
          <span
            className={`text-center text-xl font-bold ${isRed ? 'text-red-500' : 'text-black'}`}
          >
            {card.value}
            <br />
            {card.suit}
          </span>
        </div>
      </div>
    </div>
  );
}

// Large card variant for speed drill mode
export function LargeCard({
  card,
  visible = true
}: {
  card: CardType;
  visible?: boolean;
}) {
  const isRed = isRedSuit(card.suit);

  return (
    <div
      className={`
        w-32 h-48 bg-white rounded-xl shadow-2xl
        flex items-center justify-center
        transition-all duration-200
        ${visible ? 'opacity-100 scale-100' : 'opacity-0 scale-90'}
      `}
    >
      <span
        className={`text-center text-5xl font-bold ${isRed ? 'text-red-500' : 'text-black'}`}
      >
        {card.value}
        <br />
        <span className="text-4xl">{card.suit}</span>
      </span>
    </div>
  );
}
