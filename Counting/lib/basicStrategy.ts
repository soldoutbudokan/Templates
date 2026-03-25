import { Card } from './deck';

export type Action = 'hit' | 'stand' | 'double' | 'split' | 'surrender';

function cardNumericValue(value: string): number {
  if (value === 'A') return 11;
  if (['K', 'Q', 'J'].includes(value)) return 10;
  return parseInt(value, 10);
}

function dealerIndex(upcard: Card): number {
  // Map dealer upcard to column index: 2=0, 3=1, ..., 10=8, A=9
  if (upcard.value === 'A') return 9;
  return cardNumericValue(upcard.value) - 2;
}

export function getHandTotal(cards: Card[]): { total: number; soft: boolean } {
  let total = 0;
  let aces = 0;
  for (const card of cards) {
    if (card.value === 'A') {
      aces++;
      total += 11;
    } else {
      total += cardNumericValue(card.value);
    }
  }
  while (total > 21 && aces > 0) {
    total -= 10;
    aces--;
  }
  return { total, soft: aces > 0 };
}

export function isPair(cards: Card[]): boolean {
  if (cards.length !== 2) return false;
  return cardNumericValue(cards[0].value) === cardNumericValue(cards[1].value);
}

// H=hit, S=stand, D=double(hit if can't), P=split
// Standard multi-deck, H17, DAS allowed

// Hard totals: rows = player total 5–17, columns = dealer 2–A
//                        2    3    4    5    6    7    8    9   10    A
const HARD: Action[][] = [
  /* 5  */ ['hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 6  */ ['hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 7  */ ['hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 8  */ ['hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 9  */ ['hit', 'double', 'double', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 10 */ ['double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'hit', 'hit'],
  /* 11 */ ['double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'],
  /* 12 */ ['hit', 'hit', 'stand', 'stand', 'stand', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 13 */ ['stand', 'stand', 'stand', 'stand', 'stand', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 14 */ ['stand', 'stand', 'stand', 'stand', 'stand', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 15 */ ['stand', 'stand', 'stand', 'stand', 'stand', 'hit', 'hit', 'hit', 'surrender', 'hit'],
  /* 16 */ ['stand', 'stand', 'stand', 'stand', 'stand', 'hit', 'hit', 'surrender', 'surrender', 'surrender'],
  /* 17 */ ['stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand'],
];

// Soft totals: rows = soft total 13 (A+2) through 20 (A+9), columns = dealer 2–A
//                          2      3      4      5      6      7      8      9     10      A
const SOFT: Action[][] = [
  /* A,2 (13) */ ['hit', 'hit', 'hit', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* A,3 (14) */ ['hit', 'hit', 'hit', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* A,4 (15) */ ['hit', 'hit', 'double', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* A,5 (16) */ ['hit', 'hit', 'double', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* A,6 (17) */ ['hit', 'double', 'double', 'double', 'double', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* A,7 (18) */ ['double', 'double', 'double', 'double', 'double', 'stand', 'stand', 'hit', 'hit', 'hit'],
  /* A,8 (19) */ ['stand', 'stand', 'stand', 'stand', 'double', 'stand', 'stand', 'stand', 'stand', 'stand'],
  /* A,9 (20) */ ['stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand'],
];

// Pairs: rows = pair value 2–A, columns = dealer 2–A
//                           2       3       4       5       6       7       8       9      10       A
const PAIRS: Action[][] = [
  /* 2,2 */ ['split', 'split', 'split', 'split', 'split', 'split', 'hit', 'hit', 'hit', 'hit'],
  /* 3,3 */ ['split', 'split', 'split', 'split', 'split', 'split', 'hit', 'hit', 'hit', 'hit'],
  /* 4,4 */ ['hit', 'hit', 'hit', 'split', 'split', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 5,5 */ ['double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'hit', 'hit'],
  /* 6,6 */ ['split', 'split', 'split', 'split', 'split', 'hit', 'hit', 'hit', 'hit', 'hit'],
  /* 7,7 */ ['split', 'split', 'split', 'split', 'split', 'split', 'hit', 'hit', 'hit', 'hit'],
  /* 8,8 */ ['split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split'],
  /* 9,9 */ ['split', 'split', 'split', 'split', 'split', 'stand', 'split', 'split', 'stand', 'stand'],
  /* T,T */ ['stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand'],
  /* A,A */ ['split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split'],
];

export function getCorrectAction(playerCards: Card[], dealerUpcard: Card): Action {
  const col = dealerIndex(dealerUpcard);

  // Check pairs first
  if (isPair(playerCards)) {
    const pairVal = cardNumericValue(playerCards[0].value);
    // Map pair value to row: 2=0, 3=1, ..., 9=7, 10=8, A(11)=9
    const row = pairVal <= 10 ? pairVal - 2 : 9;
    const action = PAIRS[row][col];
    if (action !== 'split') return action; // e.g. 5,5 => double
    return 'split';
  }

  const { total, soft } = getHandTotal(playerCards);

  // Soft hands
  if (soft && total >= 13 && total <= 20) {
    const row = total - 13;
    return SOFT[row][col];
  }

  // Hard hands
  if (total <= 4) return 'hit';
  if (total >= 17) return 'stand';
  const row = total - 5; // total 5 = row 0
  return HARD[row][col];
}

export function formatAction(action: Action): string {
  switch (action) {
    case 'hit': return 'Hit';
    case 'stand': return 'Stand';
    case 'double': return 'Double';
    case 'split': return 'Split';
    case 'surrender': return 'Surrender';
  }
}
