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
  /* 8,8 */ ['split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'surrender'],
  /* 9,9 */ ['split', 'split', 'split', 'split', 'split', 'stand', 'split', 'split', 'stand', 'stand'],
  /* T,T */ ['stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand', 'stand'],
  /* A,A */ ['split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split', 'split'],
];

export interface ActionOptions {
  canDouble?: boolean;
  canSplit?: boolean;
  canSurrender?: boolean;
}

/** Total-dependent 6-deck H17 / DAS / late surrender, after a negative dealer peek.
 * Source: https://wizardofodds.com/games/blackjack/strategy/4-decks/
 * H17 amendments include surrender 15/17 vs A and double soft 18 vs 2, soft 19 vs 6.
 */
export function getCorrectAction(playerCards: Card[], dealerUpcard: Card, options: ActionOptions = {}): Action {
  if (playerCards.length < 2) throw new RangeError('A playing decision requires at least two cards.');
  const col = dealerIndex(dealerUpcard);
  const { total, soft } = getHandTotal(playerCards);
  const initial = playerCards.length === 2;
  const pair = isPair(playerCards);
  const canDouble = initial && options.canDouble !== false;
  const canSurrender = initial && options.canSurrender !== false;
  const canSplit = pair && options.canSplit !== false;

  if (canSurrender && !soft && (
    (total === 15 && col >= 8) ||
    (total === 16 && (pair ? col === 9 : col >= 7)) ||
    (total === 17 && col === 9)
  )) return 'surrender';

  let action: Action;
  if (canSplit) {
    const pairVal = cardNumericValue(playerCards[0].value);
    action = PAIRS[pairVal <= 10 ? pairVal - 2 : 9][col];
    if (action === 'surrender') return 'split'; // 8,8 vs A without surrender.
    if (action === 'split') return action;
  } else if (soft && total <= 12) {
    return 'hit';
  } else if (soft && total >= 13 && total <= 20) {
    action = SOFT[total - 13][col];
  } else if (total >= 17) {
    return 'stand';
  } else if (total <= 4) {
    return 'hit';
  } else {
    action = HARD[total - 5][col];
  }
  if (action === 'surrender') return 'hit';
  if (action === 'double' && !canDouble) return soft && total >= 18 ? 'stand' : 'hit';
  return action;
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
