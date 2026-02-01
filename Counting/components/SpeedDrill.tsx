'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Card as CardType, getCardValue, Deck } from '@/lib/deck';
import { LargeCard } from './Card';
import { SpeedSetting } from './Controls';

interface SpeedDrillProps {
  deck: Deck;
  speed: SpeedSetting;
  onRoundComplete: (correct: boolean) => void;
}

type DrillState = 'idle' | 'running' | 'input' | 'result';

export default function SpeedDrill({ deck, speed, onRoundComplete }: SpeedDrillProps) {
  const [state, setState] = useState<DrillState>('idle');
  const [cards, setCards] = useState<CardType[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userGuess, setUserGuess] = useState('');
  const [correctCount, setCorrectCount] = useState(0);
  const [cardsDealt, setCardsDealt] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startDrill = useCallback(() => {
    cleanup();

    // Check if we need to reshuffle
    if (deck.needsReshuffle()) {
      deck.shuffle();
    }

    const cardCount = Math.floor(Math.random() * 16) + 15; // 15 to 30 cards
    const newCards = deck.deal(cardCount);
    const count = newCards.reduce((sum, card) => sum + getCardValue(card), 0);

    setCards(newCards);
    setCorrectCount(count);
    setCardsDealt(newCards.length);
    setCurrentIndex(0);
    setUserGuess('');
    setElapsedTime(0);
    setState('running');
    startTimeRef.current = Date.now();

    // Start the card timer
    timerRef.current = setInterval(() => {
      setCurrentIndex(prev => {
        const next = prev + 1;
        if (next >= newCards.length) {
          cleanup();
          setState('input');
          setElapsedTime(Date.now() - startTimeRef.current);
          return prev;
        }
        return next;
      });
    }, speed * 1000);
  }, [deck, speed, cleanup]);

  const stopDrill = useCallback(() => {
    cleanup();
    setState('idle');
    setCards([]);
    setCurrentIndex(0);
  }, [cleanup]);

  const handleSubmitGuess = useCallback(() => {
    if (state !== 'input' || userGuess === '') return;
    setState('result');
    const isCorrect = parseInt(userGuess, 10) === correctCount;
    onRoundComplete(isCorrect);
  }, [state, userGuess, correctCount, onRoundComplete]);

  // Focus input when entering input state
  useEffect(() => {
    if (state === 'input' && inputRef.current) {
      inputRef.current.focus();
    }
  }, [state]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 'Enter') {
        if (state === 'idle') {
          startDrill();
        } else if (state === 'input') {
          handleSubmitGuess();
        }
      } else if (event.key === ' ') {
        event.preventDefault();
        if (state === 'result' || state === 'idle') {
          startDrill();
        } else if (state === 'running') {
          stopDrill();
        }
      } else if (event.key === 'Escape') {
        if (state === 'running') {
          stopDrill();
        }
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [state, startDrill, stopDrill, handleSubmitGuess]);

  // Cleanup on unmount
  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const tenths = Math.floor((ms % 1000) / 100);
    return `${seconds}.${tenths}s`;
  };

  const isCorrect = parseInt(userGuess, 10) === correctCount;

  return (
    <div className="flex-1 flex flex-col items-center justify-center">
      {/* Idle State */}
      {state === 'idle' && (
        <div className="text-center space-y-6">
          <div className="w-32 h-48 bg-white/10 rounded-xl border-2 border-dashed border-white/30 flex items-center justify-center">
            <span className="text-4xl text-white/30">?</span>
          </div>
          <div className="space-y-2">
            <p className="text-lg text-white/70">Cards will flash at {speed}s each</p>
            <button
              onClick={startDrill}
              className="btn-primary text-lg px-8 py-3"
            >
              Start Drill (Enter)
            </button>
          </div>
        </div>
      )}

      {/* Running State */}
      {state === 'running' && cards[currentIndex] && (
        <div className="text-center space-y-6">
          <div className="relative">
            <LargeCard card={cards[currentIndex]} visible={true} />
            <div className="absolute -top-8 left-1/2 -translate-x-1/2 text-sm text-white/50">
              {currentIndex + 1} / {cards.length}
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-white/70">Keep counting...</p>
            <button
              onClick={stopDrill}
              className="btn-outline text-sm"
            >
              Stop (Esc)
            </button>
          </div>
        </div>
      )}

      {/* Input State */}
      {state === 'input' && (
        <div className="text-center space-y-6 animate-slide-in">
          <div className="space-y-2">
            <p className="text-xl font-medium">Time&apos;s up!</p>
            <p className="text-white/70">{cardsDealt} cards in {formatTime(elapsedTime)}</p>
          </div>
          <div className="space-y-4">
            <input
              ref={inputRef}
              type="number"
              value={userGuess}
              onChange={(e) => setUserGuess(e.target.value)}
              placeholder="Enter your count"
              className="w-48 text-center text-2xl p-4 rounded-lg text-black focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div>
              <button
                onClick={handleSubmitGuess}
                disabled={userGuess === ''}
                className="btn-secondary text-lg px-8 py-3"
              >
                Submit (Enter)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Result State */}
      {state === 'result' && (
        <div className="text-center space-y-6 animate-slide-in">
          <div className="space-y-2">
            <p className="text-xl">Correct count: <span className="font-bold">{correctCount}</span></p>
            <p className="text-xl">Your guess: <span className="font-bold">{userGuess}</span></p>
            <p className={`text-3xl font-bold mt-4 ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
              {isCorrect ? 'Correct!' : 'Incorrect'}
            </p>
          </div>
          <div className="space-y-2 text-white/70 text-sm">
            <p>{cardsDealt} cards at {speed}s = {formatTime(elapsedTime)}</p>
          </div>
          <button
            onClick={startDrill}
            className="btn-primary text-lg px-8 py-3"
          >
            Next Drill (Space)
          </button>
        </div>
      )}
    </div>
  );
}
