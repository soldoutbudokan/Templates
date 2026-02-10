import { z } from "zod";
import { CardStore } from "../card-store.js";
import { Card } from "../types.js";

export const EXPORT_CARDS_NAME = "export_cards";

export const EXPORT_CARDS_DESCRIPTION = `Export staged flashcards in various formats.

Formats:
- "json": Full JSON array with all card fields
- "csv": Comma-separated with front, back, type, tags, deck columns
- "remnote_text": RemNote-compatible tab-indented text with :: separators (can be pasted into RemNote)`;

export const ExportCardsSchema = z.object({
  format: z
    .enum(["json", "csv", "remnote_text"])
    .describe("Export format"),
  deck: z
    .string()
    .optional()
    .describe("Filter cards by deck name. Omit to export all."),
});

export type ExportCardsInput = z.infer<typeof ExportCardsSchema>;

export function handleExportCards(
  input: ExportCardsInput,
  store: CardStore
): string {
  const cards = store.getCards(input.deck);

  if (cards.length === 0) {
    return "No staged cards to export.";
  }

  switch (input.format) {
    case "json":
      return exportJson(cards);
    case "csv":
      return exportCsv(cards);
    case "remnote_text":
      return exportRemNoteText(cards);
  }
}

function exportJson(cards: Card[]): string {
  const exported = cards.map((c) => ({
    front: c.front,
    back: c.back,
    type: c.type,
    tags: c.tags,
    deck: c.deck,
  }));
  return JSON.stringify(exported, null, 2);
}

function exportCsv(cards: Card[]): string {
  const lines: string[] = [];
  lines.push("front,back,type,tags,deck");
  for (const card of cards) {
    const front = csvEscape(card.front);
    const back = csvEscape(card.back);
    const tags = csvEscape(card.tags.join("; "));
    const deck = csvEscape(card.deck);
    lines.push(`${front},${back},${card.type},${tags},${deck}`);
  }
  return lines.join("\n");
}

function exportRemNoteText(cards: Card[]): string {
  const byDeck = new Map<string, Card[]>();
  for (const card of cards) {
    const list = byDeck.get(card.deck) ?? [];
    list.push(card);
    byDeck.set(card.deck, list);
  }

  const lines: string[] = [];
  for (const [deck, deckCards] of byDeck) {
    lines.push(`- ${deck}`);
    for (const card of deckCards) {
      if (card.type === "cloze") {
        // Cloze cards: front contains the cloze text with {{...}} markers
        lines.push(`\t- ${card.front}`);
      } else {
        // Basic cards: front :: back format
        lines.push(`\t- ${card.front}::${card.back}`);
      }
    }
  }
  return lines.join("\n");
}

function csvEscape(value: string): string {
  if (value.includes(",") || value.includes('"') || value.includes("\n")) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}
