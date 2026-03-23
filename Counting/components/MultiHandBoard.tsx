'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card as CardType, getCardValue, Deck } from '@/lib/deck';
import Card from './Card';
import BettingAdvice from './BettingAdvice';

interface MultiHandBoardProps {
  deck: Deck;
  handCount: number;
  onRoundComplete: (correct: boolean) => void;
  showBettingTips: boolean;
}

export default function MultiHandBoard({ deck, handCount, onRoundComplete, showBettingTips }: MultiHandBoardProps) {
  const [hands, setHands] = useState<CardType[][]>([]);
  const [dealerHand, setDealerHand] = useState<CardType[]>([]);
  const [userGuess, setUserGuess] = useState('');
  const [showAnswer, setShowAnswer] = useState(false);
  const [correctCount, setCorrectCount] = useState(0);
  const [decksRemainingAtDeal, setDecksRemainingAtDeal] = useState(0);
  const [animationKey, setAnimationKey] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const dealHands = useCallback(() => {
    if (deck.needsReshuffle()) {
      deck.shuffle();
    }

    const newHands: CardType[][] = Array.from({ length: handCount }, () => []);
    const newDealerHand: CardType[] = [];

    // Round 1: one card to each player hand, then dealer
    for (let i = 0; i < handCount; i++) {
      newHands[i].push(...deck.deal(1));
    }
    newDealerHand.push(...deck.deal(1));

    // Round 2: one card to each player hand, then dealer (face down in real game, but we show it)
    for (let i = 0; i < handCount; i++) {
      newHands[i].push(...deck.deal(1));
    }
    newDealerHand.push(...deck.deal(1));

    // Calculate running count for all visible cards
    const allCards = [...newHands.flat(), ...newDealerHand];
    const count = allCards.reduce((sum, card) => sum + getCardValue(card), 0);

    setHands(newHands);
    setDealerHand(newDealerHand);
    setCorrectCount(count);
    setDecksRemainingAtDeal(deck.decksRemaining());
    setShowAnswer(false);
    setUserGuess('');
    setAnimationKey(prev => prev + 1);
  }, [deck, handCount]);

  const handleSubmitGuess = useCallback(() => {
    if (showAnswer || userGuess === '') return;
    setShowAnswer(true);
    const isCorrect = parseInt(userGuess, 10) === correctCount;
    onRoundComplete(isCorrect);
  }, [showAnswer, userGuess, correctCount, onRoundComplete]);

  useEffect(() => {
    dealHands();
  }, [dealHands]);

  // Focus input after deal
  useEffect(() => {
    if (!showAnswer && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [animationKey, showAnswer]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && !showAnswer) {
        handleSubmitGuess();
      } else if (event.key === ' ') {
        event.preventDefault();
        dealHands();
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showAnswer, handleSubmitGuess, dealHands]);

  const isCorrect = parseInt(userGuess, 10) === correctCount;
  const totalCards = hands.flat().length + dealerHand.length;

  return (
    <div className="flex-1 flex flex-col">
      {/* Table Layout */}
      <div
        key={animationKey}
        className="flex-1 bg-felt-green-light rounded-lg p-6 flex flex-col items-center justify-center gap-6"
      >
        {/* Dealer */}
        <div className="text-center">
          <p className="text-xs text-white/50 mb-2 uppercase tracking-wider">Dealer</p>
          <div className="flex gap-2 justify-center">
            {dealerHand.map((card, i) => (
              <Card
                key={card.id}
                card={card}
                index={i}
                flipped={true}
                showFlipAnimation={true}
              />
            ))}
          </div>
        </div>

        {/* Divider */}
        <div className="w-48 h-px bg-white/20" />

        {/* Player Hands */}
        <div className="flex gap-8 justify-center flex-wrap">
          {hands.map((hand, handIndex) => (
            <div key={handIndex} className="text-center">
              <p className="text-xs text-white/50 mb-2">Hand {handIndex + 1}</p>
              <div className="flex gap-1 justify-center">
                {hand.map((card, i) => (
                  <Card
                    key={card.id}
                    card={card}
                    index={handIndex * 2 + i + dealerHand.length}
                    flipped={true}
                    showFlipAnimation={true}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Input Section */}
      <div className="mt-4 space-y-4">
        <p className="text-center text-sm text-white/50">
          {totalCards} cards visible across {handCount} hands + dealer
        </p>
        <div className="flex justify-center">
          <input
            ref={inputRef}
            type="number"
            value={userGuess}
            onChange={(e) => setUserGuess(e.target.value)}
            placeholder="Running count"
            disabled={showAnswer}
            className="w-1/3 min-w-[200px] text-center text-lg p-3 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="flex gap-4 justify-center">
          <button
            onClick={handleSubmitGuess}
            disabled={showAnswer || userGuess === ''}
            className="btn-secondary"
          >
            Submit Guess (Enter)
          </button>
          <button
            onClick={dealHands}
            className="btn-primary"
          >
            New Deal (Space)
          </button>
        </div>

        {/* Answer Display */}
        {showAnswer && (
          <div className="text-center animate-slide-in space-y-3">
            <p className="text-xl">Correct count: <span className="font-bold">{correctCount}</span></p>
            <p className="text-xl">Your guess: <span className="font-bold">{userGuess}</span></p>
            <p className={`text-2xl font-bold mt-2 ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
              {isCorrect ? 'Correct!' : 'Incorrect. Try again!'}
            </p>

            {showBettingTips && (
              <BettingAdvice
                runningCount={correctCount}
                decksRemaining={decksRemainingAtDeal}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
