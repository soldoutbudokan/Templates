import { v4 as uuidv4 } from "uuid";
import { Card, CardInput } from "./types.js";

const DEFAULT_DECK = "Claude Flashcards";

/**
 * In-memory staging store for flashcards before they're pushed to RemNote.
 */
export class CardStore {
  private cards: Map<string, Card> = new Map();

  addCards(inputs: CardInput[]): Card[] {
    const created: Card[] = [];
    for (const input of inputs) {
      const card: Card = {
        id: uuidv4(),
        front: input.front,
        back: input.back,
        type: input.type ?? "basic",
        tags: input.tags ?? [],
        deck: input.deck ?? DEFAULT_DECK,
        createdAt: new Date().toISOString(),
      };
      this.cards.set(card.id, card);
      created.push(card);
    }
    return created;
  }

  getCards(deck?: string): Card[] {
    const all = Array.from(this.cards.values());
    if (deck) {
      return all.filter((c) => c.deck === deck);
    }
    return all;
  }

  getCardsByIds(ids: string[]): Card[] {
    const result: Card[] = [];
    for (const id of ids) {
      const card = this.cards.get(id);
      if (card) result.push(card);
    }
    return result;
  }

  removeCards(ids: string[]): number {
    let removed = 0;
    for (const id of ids) {
      if (this.cards.delete(id)) removed++;
    }
    return removed;
  }

  clearAll(): number {
    const count = this.cards.size;
    this.cards.clear();
    return count;
  }

  get size(): number {
    return this.cards.size;
  }

  getDecks(): string[] {
    const decks = new Set<string>();
    for (const card of this.cards.values()) {
      decks.add(card.deck);
    }
    return Array.from(decks).sort();
  }
}
