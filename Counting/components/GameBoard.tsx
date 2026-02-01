'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card as CardType, getCardValue, Deck } from '@/lib/deck';
import Card from './Card';

interface GameBoardProps {
  deck: Deck;
  onRoundComplete: (correct: boolean) => void;
}

export default function GameBoard({ deck, onRoundComplete }: GameBoardProps) {
  const [cards, setCards] = useState<CardType[]>([]);
  const [userGuess, setUserGuess] = useState('');
  const [showAnswer, setShowAnswer] = useState(false);
  const [correctCount, setCorrectCount] = useState(0);
  const [animationKey, setAnimationKey] = useState(0);

  const generateNewHand = useCallback(() => {
    // Check if we need to reshuffle
    if (deck.needsReshuffle()) {
      deck.shuffle();
    }

    const cardCount = Math.floor(Math.random() * 16) + 15; // 15 to 30 cards
    const newCards = deck.deal(cardCount);
    setCards(newCards);
    setShowAnswer(false);
    setUserGuess('');
    setCorrectCount(newCards.reduce((count, card) => count + getCardValue(card), 0));
    setAnimationKey(prev => prev + 1);
  }, [deck]);

  const handleSubmitGuess = useCallback(() => {
    if (showAnswer || userGuess === '') return;
    setShowAnswer(true);
    const isCorrect = parseInt(userGuess, 10) === correctCount;
    onRoundComplete(isCorrect);
  }, [showAnswer, userGuess, correctCount, onRoundComplete]);

  useEffect(() => {
    generateNewHand();
  }, [generateNewHand]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Enter' && !showAnswer) {
        handleSubmitGuess();
      } else if (event.key === ' ') {
        event.preventDefault();
        generateNewHand();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [showAnswer, handleSubmitGuess, generateNewHand]);

  const isCorrect = parseInt(userGuess, 10) === correctCount;

  return (
    <div className="flex-1 flex flex-col">
      {/* Cards Display */}
      <div
        key={animationKey}
        className="flex-1 flex flex-wrap gap-2 p-4 justify-center items-center content-center bg-felt-green-light rounded-lg overflow-y-auto"
      >
        {cards.map((card, index) => (
          <Card
            key={card.id}
            card={card}
            index={index}
            flipped={true}
            showFlipAnimation={true}
          />
        ))}
      </div>

      {/* Input Section */}
      <div className="mt-4 space-y-4">
        <div className="flex justify-center">
          <input
            type="number"
            value={userGuess}
            onChange={(e) => setUserGuess(e.target.value)}
            placeholder="Enter your count"
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
            onClick={generateNewHand}
            className="btn-primary"
          >
            New Hand (Space)
          </button>
        </div>

        {/* Answer Display */}
        {showAnswer && (
          <div className="text-center animate-slide-in">
            <p className="text-xl">Correct count: <span className="font-bold">{correctCount}</span></p>
            <p className="text-xl">Your guess: <span className="font-bold">{userGuess}</span></p>
            <p className={`text-2xl font-bold mt-2 ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
              {isCorrect ? 'Correct!' : 'Incorrect. Try again!'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
