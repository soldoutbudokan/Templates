export type Suit = '♠' | '♥' | '♦' | '♣';
export type Value = '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10' | 'J' | 'Q' | 'K' | 'A';

export interface Card {
  suit: Suit;
  value: Value;
  id: string; // Unique identifier for React keys
}

const SUITS: Suit[] = ['♠', '♥', '♦', '♣'];
const VALUES: Value[] = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

export function getCardValue(card: Card): number {
  if (['2', '3', '4', '5', '6'].includes(card.value)) return 1;
  if (['10', 'J', 'Q', 'K', 'A'].includes(card.value)) return -1;
  return 0;
}

export function isRedSuit(suit: Suit): boolean {
  return suit === '♥' || suit === '♦';
}

export class Deck {
  private cards: Card[] = [];
  private dealtCards: Card[] = [];
  private deckCount: number;
  private cutCardPercentage: number;

  constructor(deckCount: number = 6, cutCardPercentage: number = 0.75) {
    this.deckCount = deckCount;
    this.cutCardPercentage = cutCardPercentage;
    this.shuffle();
  }

  // Fisher-Yates shuffle algorithm
  private fisherYatesShuffle(array: Card[]): Card[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }

  shuffle(): void {
    this.cards = [];
    this.dealtCards = [];

    // Create shoe with multiple decks
    let cardId = 0;
    for (let d = 0; d < this.deckCount; d++) {
      for (const suit of SUITS) {
        for (const value of VALUES) {
          this.cards.push({
            suit,
            value,
            id: `${d}-${suit}-${value}-${cardId++}`,
          });
        }
      }
    }

    this.cards = this.fisherYatesShuffle(this.cards);
  }

  deal(count: number = 1): Card[] {
    const dealt: Card[] = [];

    for (let i = 0; i < count; i++) {
      if (this.cards.length === 0) {
        this.shuffle();
      }
      const card = this.cards.pop()!;
      dealt.push(card);
      this.dealtCards.push(card);
    }

    return dealt;
  }

  remaining(): number {
    return this.cards.length;
  }

  totalCards(): number {
    return this.deckCount * 52;
  }

  percentageDealt(): number {
    return this.dealtCards.length / this.totalCards();
  }

  needsReshuffle(): boolean {
    return this.percentageDealt() >= this.cutCardPercentage;
  }

  getDeckCount(): number {
    return this.deckCount;
  }

  setDeckCount(count: number): void {
    this.deckCount = count;
    this.shuffle();
  }

  setCutCardPercentage(percentage: number): void {
    this.cutCardPercentage = percentage;
  }
}

// Singleton instance for global deck state
let deckInstance: Deck | null = null;

export function getDeck(deckCount?: number): Deck {
  if (!deckInstance || (deckCount !== undefined && deckCount !== deckInstance.getDeckCount())) {
    deckInstance = new Deck(deckCount);
  }
  return deckInstance;
}

export function resetDeck(deckCount: number = 6): Deck {
  deckInstance = new Deck(deckCount);
  return deckInstance;
}
