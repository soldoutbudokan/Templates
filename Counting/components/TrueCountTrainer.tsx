'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card as CardType, getCardValue, Deck } from '@/lib/deck';
import Card from './Card';
import { parseCount, toTrueCount } from '@/lib/countPolicy';
import BettingAdvice from './BettingAdvice';

interface TrueCountTrainerProps {
  deck: Deck;
  onRoundComplete: (correct: boolean) => void;
  showBettingTips: boolean;
}

type TrainerState = 'cards' | 'running-input' | 'true-count-input' | 'result';

export default function TrueCountTrainer({ deck, onRoundComplete, showBettingTips }: TrueCountTrainerProps) {
  const [cards, setCards] = useState<CardType[]>([]);
  const [state, setState] = useState<TrainerState>('cards');
  const [runningGuess, setRunningGuess] = useState('');
  const [trueCountGuess, setTrueCountGuess] = useState('');
  const [correctRunningCount, setCorrectRunningCount] = useState(0);
  const [decksRemainingAtDeal, setDecksRemainingAtDeal] = useState(0);
  const [animationKey, setAnimationKey] = useState(0);
  const [runningCorrect, setRunningCorrect] = useState(false);
  const [newShoe, setNewShoe] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);
  const initializedDeck = useRef<Deck | null>(null);

  const exactTrueCount = decksRemainingAtDeal > 0
    ? correctRunningCount / decksRemainingAtDeal
    : 0;
  const roundedTrueCount = decksRemainingAtDeal > 0 ? toTrueCount(correctRunningCount, decksRemainingAtDeal) : 0;

  const generateNewHand = useCallback(() => {
    const cardCount = Math.floor(Math.random() * 16) + 15;
    const shuffled = deck.prepareRound(cardCount);
    setNewShoe(shuffled || deck.remaining() === deck.totalCards());
    const newCards = deck.deal(cardCount);
    setCards(newCards);
    setCorrectRunningCount(deck.runningCount());
    setDecksRemainingAtDeal(Math.max(0.5, Math.round(deck.decksRemaining() * 2) / 2));
    setState('running-input');
    setRunningGuess('');
    setTrueCountGuess('');
    setRunningCorrect(false);
    setAnimationKey(prev => prev + 1);
  }, [deck]);

  const handleSubmitRunning = useCallback(() => {
    if (state !== 'running-input' || parseCount(runningGuess) === null) return;
    const isCorrect = parseCount(runningGuess) === correctRunningCount;
    setRunningCorrect(isCorrect);

    if (isCorrect) {
      setState('true-count-input');
    } else {
      // Wrong running count — round is failed
      setState('result');
      onRoundComplete(false);
    }
  }, [state, runningGuess, correctRunningCount, onRoundComplete]);

  const handleSubmitTrueCount = useCallback(() => {
    if (state !== 'true-count-input' || parseCount(trueCountGuess) === null) return;
    const guess = parseCount(trueCountGuess);
    const isCorrect = guess === roundedTrueCount;
    setState('result');
    onRoundComplete(isCorrect);
  }, [state, trueCountGuess, roundedTrueCount, onRoundComplete]);

  // Initial hand
  useEffect(() => {
    if (initializedDeck.current === deck) return;
    initializedDeck.current = deck;
    generateNewHand();
  }, [deck, generateNewHand]);

  // Focus input on state change
  useEffect(() => {
    if ((state === 'running-input' || state === 'true-count-input') && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [state]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Enter') {
        if (state === 'running-input') {
          handleSubmitRunning();
        } else if (state === 'true-count-input') {
          handleSubmitTrueCount();
        }
      } else if (event.key === ' ') {
        event.preventDefault();
        if (state === 'result') {
          generateNewHand();
        }
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [state, handleSubmitRunning, handleSubmitTrueCount, generateNewHand]);

  const trueCountIsCorrect = state === 'result' && runningCorrect &&
    parseCount(trueCountGuess) === roundedTrueCount;

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
        {/* Step 1: Running Count */}
        {state === 'running-input' && (
          <div className="text-center space-y-3 animate-slide-in">
            <p className="text-lg text-white/70">{newShoe ? 'New shoe: count starts at zero.' : 'Continue your count from the previous round.'} What is the running count?</p>
            <div className="flex justify-center">
              <input
                ref={inputRef}
                type="number"
                value={runningGuess}
                onChange={(e) => setRunningGuess(e.target.value)}
                aria-label="Running count" placeholder="Running count"
                className="w-1/3 min-w-[200px] text-center text-lg p-3 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              onClick={handleSubmitRunning}
              disabled={parseCount(runningGuess) === null}
              className="btn-secondary"
            >
              Submit Running Count (Enter)
            </button>
          </div>
        )}

        {/* Step 2: True Count */}
        {state === 'true-count-input' && (
          <div className="text-center space-y-3 animate-slide-in">
            <p className="text-lg text-green-400">Running count correct! ✓</p>
            <p className="text-white/70">
              Step 2: ~{decksRemainingAtDeal.toFixed(1)} decks remaining. What is the true count?
            </p>
            <div className="flex justify-center">
              <input
                ref={inputRef}
                type="number"
                value={trueCountGuess}
                onChange={(e) => setTrueCountGuess(e.target.value)}
                aria-label="True count" placeholder="True count"
                className="w-1/3 min-w-[200px] text-center text-lg p-3 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <p className="text-xs text-white/40">True count = running count ÷ decks remaining (round down: −1.5 becomes −2)</p>
            <button
              onClick={handleSubmitTrueCount}
              disabled={parseCount(trueCountGuess) === null}
              className="btn-secondary"
            >
              Submit True Count (Enter)
            </button>
          </div>
        )}

        {/* Result */}
        {state === 'result' && (
          <div className="text-center space-y-3 animate-slide-in">
            <div className="space-y-1">
              <p className="text-xl">
                Running Count: <span className="font-bold">{correctRunningCount}</span>
                {' '}
                <span className={runningCorrect ? 'text-green-400' : 'text-red-400'}>
                  (You: {runningGuess} {runningCorrect ? '✓' : '✗'})
                </span>
              </p>
              {runningCorrect && (
                <>
                  <p className="text-white/70 text-sm">
                    {correctRunningCount >= 0 ? '+' : ''}{correctRunningCount} ÷ {decksRemainingAtDeal.toFixed(1)} decks = {exactTrueCount >= 0 ? '+' : ''}{exactTrueCount.toFixed(1)} → Round down: {roundedTrueCount >= 0 ? '+' : ''}{roundedTrueCount}
                  </p>
                  <p className="text-xl">
                    True Count: <span className="font-bold">{roundedTrueCount}</span>
                    {' '}
                    <span className={trueCountIsCorrect ? 'text-green-400' : 'text-red-400'}>
                      (You: {trueCountGuess} {trueCountIsCorrect ? '✓' : '✗'})
                    </span>
                  </p>
                </>
              )}
              <p className={`text-2xl font-bold mt-2 ${
                (runningCorrect && trueCountIsCorrect) ? 'text-green-400' : 'text-red-400'
              }`}>
                {(runningCorrect && trueCountIsCorrect) ? 'Correct!' : 'Incorrect'}
              </p>
            </div>

            {showBettingTips && (
              <BettingAdvice
                runningCount={correctRunningCount}
                decksRemaining={decksRemainingAtDeal}
              />
            )}

            <button
              onClick={generateNewHand}
              className="btn-primary"
            >
              New Hand (Space)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

