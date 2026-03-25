'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card as CardType, Deck } from '@/lib/deck';
import { Action, getCorrectAction, formatAction, isPair, getHandTotal } from '@/lib/basicStrategy';
import Card from './Card';
import BettingAdvice from './BettingAdvice';

interface BasicStrategyTrainerProps {
  deck: Deck;
  onRoundComplete: (correct: boolean) => void;
  showBettingTips: boolean;
}

export default function BasicStrategyTrainer({ deck, onRoundComplete, showBettingTips }: BasicStrategyTrainerProps) {
  const [playerCards, setPlayerCards] = useState<CardType[]>([]);
  const [dealerUpcard, setDealerUpcard] = useState<CardType | null>(null);
  const [selectedAction, setSelectedAction] = useState<Action | null>(null);
  const [correctAction, setCorrectAction] = useState<Action | null>(null);
  const [animationKey, setAnimationKey] = useState(0);
  const [decksRemainingAtDeal, setDecksRemainingAtDeal] = useState(0);

  const dealNewHand = useCallback(() => {
    if (deck.needsReshuffle()) {
      deck.shuffle();
    }

    const cards = deck.deal(3);
    setPlayerCards([cards[0], cards[1]]);
    setDealerUpcard(cards[2]);
    setSelectedAction(null);
    setCorrectAction(null);
    setDecksRemainingAtDeal(deck.decksRemaining());
    setAnimationKey(prev => prev + 1);
  }, [deck]);

  const handleAction = useCallback((action: Action) => {
    if (selectedAction !== null || !dealerUpcard) return;

    const correct = getCorrectAction(playerCards, dealerUpcard);
    setSelectedAction(action);
    setCorrectAction(correct);
    onRoundComplete(action === correct);
  }, [selectedAction, playerCards, dealerUpcard, onRoundComplete]);

  useEffect(() => {
    dealNewHand();
  }, [dealNewHand]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (selectedAction === null) {
        if (key === 'h') handleAction('hit');
        else if (key === 's') handleAction('stand');
        else if (key === 'd') handleAction('double');
        else if (key === 'p' && canSplit) handleAction('split');
      }
      if (key === ' ') {
        event.preventDefault();
        dealNewHand();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedAction, handleAction, dealNewHand]);

  const canSplit = isPair(playerCards);
  const handInfo = playerCards.length > 0 ? getHandTotal(playerCards) : null;
  const isCorrect = selectedAction !== null && selectedAction === correctAction;

  const actionButtonClass = (action: Action) => {
    const base = 'py-3 px-6 rounded-lg text-lg font-bold transition-all';
    if (selectedAction === null) {
      // Not yet answered
      if (action === 'split' && !canSplit) return `${base} bg-white/5 text-white/20 cursor-not-allowed`;
      return `${base} bg-white/10 text-white hover:bg-white/20`;
    }
    // After answer
    if (action === correctAction) return `${base} bg-green-600 text-white ring-2 ring-green-400`;
    if (action === selectedAction) return `${base} bg-red-600 text-white`;
    return `${base} bg-white/5 text-white/30`;
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Table */}
      <div
        key={animationKey}
        className="flex-1 flex flex-col items-center justify-center gap-8 p-6 bg-felt-green-light rounded-lg"
      >
        {/* Dealer */}
        <div className="text-center">
          <p className="text-sm text-white/60 mb-2">Dealer shows</p>
          <div className="flex gap-3 justify-center">
            {dealerUpcard && (
              <Card card={dealerUpcard} index={0} flipped={true} showFlipAnimation={true} />
            )}
          </div>
        </div>

        {/* VS divider */}
        <div className="text-white/30 text-sm font-medium">vs</div>

        {/* Player */}
        <div className="text-center">
          <p className="text-sm text-white/60 mb-2">
            Your hand{handInfo ? ` (${handInfo.soft ? 'Soft ' : ''}${handInfo.total})` : ''}
          </p>
          <div className="flex gap-3 justify-center">
            {playerCards.map((card, index) => (
              <Card key={card.id} card={card} index={index} flipped={true} showFlipAnimation={true} />
            ))}
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-4 space-y-4">
        <div className="flex gap-3 justify-center">
          <button onClick={() => handleAction('hit')} disabled={selectedAction !== null} className={actionButtonClass('hit')}>
            Hit <kbd className="ml-1 text-xs opacity-60">H</kbd>
          </button>
          <button onClick={() => handleAction('stand')} disabled={selectedAction !== null} className={actionButtonClass('stand')}>
            Stand <kbd className="ml-1 text-xs opacity-60">S</kbd>
          </button>
          <button onClick={() => handleAction('double')} disabled={selectedAction !== null} className={actionButtonClass('double')}>
            Double <kbd className="ml-1 text-xs opacity-60">D</kbd>
          </button>
          <button
            onClick={() => handleAction('split')}
            disabled={selectedAction !== null || !canSplit}
            className={actionButtonClass('split')}
          >
            Split <kbd className="ml-1 text-xs opacity-60">P</kbd>
          </button>
        </div>

        <div className="flex justify-center">
          <button onClick={dealNewHand} className="btn-primary">
            New Hand (Space)
          </button>
        </div>

        {/* Feedback */}
        {selectedAction !== null && correctAction !== null && (
          <div className="text-center animate-slide-in space-y-2">
            <p className={`text-2xl font-bold ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
              {isCorrect ? 'Correct!' : `Incorrect — ${formatAction(correctAction)}`}
            </p>
            {!isCorrect && (
              <p className="text-sm text-white/60">
                You chose {formatAction(selectedAction)}, correct play is {formatAction(correctAction)}
              </p>
            )}

            {showBettingTips && (
              <BettingAdvice
                runningCount={0}
                decksRemaining={decksRemainingAtDeal}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
