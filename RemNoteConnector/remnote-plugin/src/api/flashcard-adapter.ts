import { RNPlugin, PluginRem } from "@remnote/plugin-sdk";

const RATE_LIMIT_DELAY_MS = 25;

interface FlashcardInput {
  front: string;
  back: string;
  type: "basic" | "cloze";
  tags: string[];
  deck: string;
}

/**
 * Wraps the RemNote Plugin SDK to provide flashcard CRUD operations.
 * Caches deck and tag Rem references to avoid repeated lookups.
 */
export class FlashcardAdapter {
  private plugin: RNPlugin;
  private deckCache: Map<string, string> = new Map(); // deckName -> remId
  private tagCache: Map<string, string> = new Map(); // tagName -> remId

  constructor(plugin: RNPlugin) {
    this.plugin = plugin;
  }

  async createFlashcard(input: FlashcardInput): Promise<{ remId: string }> {
    const deckRem = await this.findOrCreateDeck(input.deck);
    if (!deckRem) {
      throw new Error(`Failed to find or create deck "${input.deck}"`);
    }

    const rem = await this.plugin.rem.createRem();
    if (!rem) {
      throw new Error("Failed to create Rem");
    }

    // Set front text
    await rem.setText([input.front]);

    if (input.type === "basic") {
      // For basic cards, set the back text
      await rem.setBackText([input.back]);
    }
    // For cloze cards, the front text already contains {{cloze}} markers
    // RemNote handles cloze formatting natively

    // Move under deck
    await rem.setParent(deckRem._id);

    // Apply tags
    for (const tagName of input.tags) {
      const tagRem = await this.findOrCreateTag(tagName);
      if (tagRem) {
        await rem.addTag(tagRem._id);
      }
    }

    // Small delay for rate limiting
    await sleep(RATE_LIMIT_DELAY_MS);

    return { remId: rem._id };
  }

  private async findOrCreateDeck(deckPath: string): Promise<PluginRem | null> {
    // Check cache first
    const cached = this.deckCache.get(deckPath);
    if (cached) {
      const existing = await this.plugin.rem.findOne(cached);
      if (existing) return existing;
      // Cache stale, remove
      this.deckCache.delete(deckPath);
    }

    // Split deck path for nested decks (e.g., "Parent/Child")
    const parts = deckPath.split("/").map((p) => p.trim());

    // Try to find existing deck by name path
    const existing = await this.plugin.rem.findByName(parts, null);
    if (existing) {
      this.deckCache.set(deckPath, existing._id);
      return existing;
    }

    // Create the deck hierarchy
    let parentId: string | null = null;
    for (let i = 0; i < parts.length; i++) {
      const subPath = parts.slice(0, i + 1);
      const subKey = subPath.join("/");

      const cachedSub = this.deckCache.get(subKey);
      if (cachedSub) {
        const ex = await this.plugin.rem.findOne(cachedSub);
        if (ex) {
          parentId = ex._id;
          continue;
        }
      }

      const found = await this.plugin.rem.findByName(subPath, null);
      if (found) {
        this.deckCache.set(subKey, found._id);
        parentId = found._id;
        continue;
      }

      // Create this level
      const newRem = await this.plugin.rem.createRem();
      if (!newRem) {
        throw new Error(`Failed to create deck level "${parts[i]}"`);
      }
      await newRem.setText([parts[i]]);
      if (parentId) {
        await newRem.setParent(parentId);
      }
      this.deckCache.set(subKey, newRem._id);
      parentId = newRem._id;
    }

    if (!parentId) return null;
    return (await this.plugin.rem.findOne(parentId)) ?? null;
  }

  private async findOrCreateTag(tagName: string): Promise<PluginRem | null> {
    const cached = this.tagCache.get(tagName);
    if (cached) {
      const existing = await this.plugin.rem.findOne(cached);
      if (existing) return existing;
      this.tagCache.delete(tagName);
    }

    const existing = await this.plugin.rem.findByName([tagName], null);
    if (existing) {
      this.tagCache.set(tagName, existing._id);
      return existing;
    }

    const newTag = await this.plugin.rem.createRem();
    if (!newTag) return null;
    await newTag.setText([tagName]);
    this.tagCache.set(tagName, newTag._id);
    return newTag;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
