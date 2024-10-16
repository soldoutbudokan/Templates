import React, { useState, useEffect, useCallback } from 'react';

const suits = ['♠', '♥', '♦', '♣'];
const values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

const getRandomCard = () => {
  const suit = suits[Math.floor(Math.random() * suits.length)];
  const value = values[Math.floor(Math.random() * values.length)];
  return { suit, value };
};

const getCardValue = (card) => {
  if (['2', '3', '4', '5', '6'].includes(card.value)) return 1;
  if (['10', 'J', 'Q', 'K', 'A'].includes(card.value)) return -1;
  return 0;
};

const CardCountingPractice = () => {
  const [cards, setCards] = useState([]);
  const [userGuess, setUserGuess] = useState('');
  const [showAnswer, setShowAnswer] = useState(false);
  const [correctCount, setCorrectCount] = useState(0);

  const generateNewHand = useCallback(() => {
    const cardCount = Math.floor(Math.random() * 11) + 10; // 10 to 20 cards
    const newCards = Array.from({ length: cardCount }, getRandomCard);
    setCards(newCards);
    setShowAnswer(false);
    setUserGuess('');
    setCorrectCount(newCards.reduce((count, card) => count + getCardValue(card), 0));
  }, []);

  const handleSubmitGuess = useCallback(() => {
    setShowAnswer(true);
  }, []);

  useEffect(() => {
    generateNewHand();
  }, [generateNewHand]);

  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.key === 'Enter' && !showAnswer) {
        handleSubmitGuess();
      } else if (event.key === ' ') {
        event.preventDefault(); // Prevent scrolling
        generateNewHand();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => {
      window.removeEventListener('keydown', handleKeyPress);
    };
  }, [showAnswer, handleSubmitGuess, generateNewHand]);

  return (
    <div style={{
      padding: '1rem',
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#1b5e20',
      color: 'white',
      fontFamily: 'Arial, sans-serif'
    }}>
      <div style={{
        maxWidth: '64rem',
        width: '100%',
        margin: '0 auto',
        flexGrow: 1,
        display: 'flex',
        flexDirection: 'column',
      }}>
        <h2 style={{
          fontSize: '1.875rem',
          fontWeight: 'bold',
          marginBottom: '1.5rem',
          textAlign: 'center',
        }}>
          Card Counting Practice
        </h2>
        <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{
            flexGrow: 1,
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.5rem',
            marginBottom: '1.5rem',
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: '#2e7d32',
            padding: '1rem',
            borderRadius: '0.5rem',
            overflowY: 'auto',
          }}>
            {cards.map((card, index) => (
              <div key={index} style={{
                backgroundColor: 'white',
                borderRadius: '0.5rem',
                padding: '0.5rem',
                width: '3.5rem',
                height: '5rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.25rem',
                boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
              }}>
                <span style={{
                  color: card.suit === '♥' || card.suit === '♦' ? '#ef4444' : '#000000',
                  textAlign: 'center',
                }}>
                  {card.value}
                  <br />
                  {card.suit}
                </span>
              </div>
            ))}
          </div>
          <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'center' }}>
            <input
              type="number"
              value={userGuess}
              onChange={(e) => setUserGuess(e.target.value)}
              placeholder="Enter your count"
              style={{
                width: '33.333%',
                textAlign: 'center',
                fontSize: '1.125rem',
                padding: '0.5rem',
                borderRadius: '0.25rem',
                border: 'none'
              }}
            />
          </div>
          <div style={{
            display: 'flex',
            gap: '1rem',
            marginBottom: '1rem',
            justifyContent: 'center',
          }}>
            <button
              onClick={handleSubmitGuess}
              disabled={showAnswer}
              style={{
                backgroundColor: '#ed6c02',
                color: 'black',
                padding: '0.5rem 1rem',
                borderRadius: '0.25rem',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              Submit Guess (Enter)
            </button>
            <button
              onClick={generateNewHand}
              style={{
                backgroundColor: '#1976d2',
                color: 'white',
                padding: '0.5rem 1rem',
                borderRadius: '0.25rem',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              New Hand (Space)
            </button>
          </div>
          {showAnswer && (
            <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
              <p style={{ fontSize: '1.25rem' }}>Correct count: {correctCount}</p>
              <p style={{ fontSize: '1.25rem' }}>Your guess: {userGuess}</p>
              <p style={{
                fontSize: '1.5rem',
                fontWeight: 'bold',
                marginTop: '0.5rem',
              }}>
                {parseInt(userGuess, 10) === correctCount
                  ? 'Correct!'
                  : 'Incorrect. Try again!'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CardCountingPractice;