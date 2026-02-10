import { z } from "zod";
import { CardStore } from "../card-store.js";
import { PluginConnection } from "../plugin-connection.js";
import { QUALITY_GUIDELINES } from "../quality-guidelines.js";
import { Card } from "../types.js";

export const CREATE_FLASHCARDS_NAME = "create_flashcards";

export const CREATE_FLASHCARDS_DESCRIPTION = `Create flashcards from the current conversation and stage them for review, or push them directly to RemNote.

By default, cards are staged locally for review. Set push_immediately=true to send directly to RemNote (requires the RemNote plugin to be connected).

${QUALITY_GUIDELINES}`;

export const CreateFlashcardsSchema = z.object({
  cards: z
    .array(
      z.object({
        front: z.string().describe("The question or prompt (front of the card)"),
        back: z.string().describe("The answer (back of the card)"),
        type: z
          .enum(["basic", "cloze"])
          .default("basic")
          .describe('Card type: "basic" for Q&A, "cloze" for fill-in-the-blank'),
        tags: z
          .array(z.string())
          .optional()
          .describe("Tags for organizing the card"),
        deck: z
          .string()
          .optional()
          .describe('Target deck name (default: "Claude Flashcards")'),
      })
    )
    .min(1)
    .describe("Array of flashcards to create"),
  push_immediately: z
    .boolean()
    .default(false)
    .describe("If true, push cards directly to RemNote instead of staging"),
});

export type CreateFlashcardsInput = z.infer<typeof CreateFlashcardsSchema>;

export async function handleCreateFlashcards(
  input: CreateFlashcardsInput,
  store: CardStore,
  connection: PluginConnection
): Promise<string> {
  const staged = store.addCards(input.cards);

  if (!input.push_immediately) {
    return formatStagedResult(staged);
  }

  // Push immediately to RemNote
  if (!connection.isConnected) {
    return (
      formatStagedResult(staged) +
      "\n\n⚠️ RemNote plugin is not connected. Cards have been staged locally. " +
      "Open RemNote with the MCP Bridge plugin to push them."
    );
  }

  const results = await pushCards(staged, store, connection);
  return results;
}

async function pushCards(
  cards: Card[],
  store: CardStore,
  connection: PluginConnection
): Promise<string> {
  const succeeded: string[] = [];
  const failed: { front: string; error: string }[] = [];

  for (const card of cards) {
    try {
      const response = await connection.sendRequest("create_flashcard", {
        front: card.front,
        back: card.back,
        type: card.type,
        tags: card.tags,
        deck: card.deck,
      });

      if (response.error) {
        failed.push({ front: card.front, error: response.error });
      } else {
        succeeded.push(card.front);
        store.removeCards([card.id]);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      failed.push({ front: card.front, error: msg });
    }
  }

  const lines: string[] = [];
  lines.push(`Pushed ${succeeded.length}/${cards.length} cards to RemNote.`);

  if (succeeded.length > 0) {
    lines.push("\n✅ Created:");
    for (const front of succeeded) {
      lines.push(`  - ${front}`);
    }
  }

  if (failed.length > 0) {
    lines.push("\n❌ Failed (still staged locally):");
    for (const f of failed) {
      lines.push(`  - "${f.front}": ${f.error}`);
    }
  }

  return lines.join("\n");
}

function formatStagedResult(cards: Card[]): string {
  const lines: string[] = [];
  lines.push(`Staged ${cards.length} card(s) for review.`);
  lines.push("");

  const byDeck = new Map<string, Card[]>();
  for (const card of cards) {
    const list = byDeck.get(card.deck) ?? [];
    list.push(card);
    byDeck.set(card.deck, list);
  }

  for (const [deck, deckCards] of byDeck) {
    lines.push(`📚 ${deck}:`);
    for (const card of deckCards) {
      lines.push(`  [${card.type}] Q: ${card.front}`);
      lines.push(`          A: ${card.back}`);
      if (card.tags.length > 0) {
        lines.push(`          Tags: ${card.tags.join(", ")}`);
      }
    }
    lines.push("");
  }

  lines.push(
    'Use list_staged_cards to review, or create_flashcards with push_immediately=true to send to RemNote.'
  );

  return lines.join("\n");
}
