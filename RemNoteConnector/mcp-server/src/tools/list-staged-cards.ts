import { z } from "zod";
import { CardStore } from "../card-store.js";
import { Card } from "../types.js";

export const LIST_STAGED_CARDS_NAME = "list_staged_cards";

export const LIST_STAGED_CARDS_DESCRIPTION =
  "List all staged flashcards awaiting review. Optionally filter by deck name.";

export const ListStagedCardsSchema = z.object({
  deck: z
    .string()
    .optional()
    .describe("Filter cards by deck name. Omit to show all decks."),
});

export type ListStagedCardsInput = z.infer<typeof ListStagedCardsSchema>;

export function handleListStagedCards(
  input: ListStagedCardsInput,
  store: CardStore
): string {
  const cards = store.getCards(input.deck);

  if (cards.length === 0) {
    if (input.deck) {
      return `No staged cards in deck "${input.deck}".`;
    }
    return "No staged cards. Use create_flashcards to stage some.";
  }

  const byDeck = new Map<string, Card[]>();
  for (const card of cards) {
    const list = byDeck.get(card.deck) ?? [];
    list.push(card);
    byDeck.set(card.deck, list);
  }

  const lines: string[] = [];
  lines.push(`${cards.length} staged card(s) across ${byDeck.size} deck(s):`);
  lines.push("");

  for (const [deck, deckCards] of byDeck) {
    lines.push(`📚 ${deck} (${deckCards.length} cards):`);
    for (let i = 0; i < deckCards.length; i++) {
      const card = deckCards[i];
      lines.push(`  ${i + 1}. [${card.type}] id=${card.id}`);
      lines.push(`     Q: ${card.front}`);
      lines.push(`     A: ${card.back}`);
      if (card.tags.length > 0) {
        lines.push(`     Tags: ${card.tags.join(", ")}`);
      }
    }
    lines.push("");
  }

  return lines.join("\n");
}
