/**
 * Supermemo's 20 Rules of Formulating Knowledge, adapted as flashcard quality guidelines.
 * Embedded in the create_flashcards tool description so the LLM follows them.
 */
export const QUALITY_GUIDELINES = `
## Flashcard Quality Guidelines (Based on 20 Rules of Formulating Knowledge)

Follow these rules when creating flashcards:

1. **Understand before memorizing** — Only create cards for material the user understands. Ask for clarification if needed.
2. **Learn the big picture first** — Start with broad concepts before drilling into details.
3. **Stick to the Minimum Information Principle** — Each card should test ONE atomic fact. Split complex ideas into multiple cards.
4. **Use cloze deletion** — For definitions and key terms, prefer cloze format (e.g., "The capital of France is {{Paris}}").
5. **Use imagery** — Reference mental images or mnemonics in the back of the card when helpful.
6. **Use mnemonic techniques** — Acronyms, stories, rhymes help retention. Add them to card backs.
7. **Graphic cloze deletion** — When referencing diagrams, describe the visual context.
8. **Avoid sets and enumerations** — Don't create cards that require listing multiple items. Break them into individual cards.
9. **Avoid vague "yes/no" cards** — Instead of "Is X true?", ask "What is X?" or use cloze deletion.
10. **Use personal context** — Tie facts to the user's experiences or analogies when possible.
11. **Provide sources** — Note the source topic or conversation context on the card (via tags).
12. **Keep it simple** — Clear, concise language. No jargon unless the jargon itself is being tested.
13. **Ensure cards are self-contained** — A card should make sense without needing to see other cards.
14. **Use consistent formatting** — Front = question or prompt, Back = answer.
15. **Tag appropriately** — Use meaningful tags for topic organization and retrieval.
16. **Prefer active recall** — Frame fronts as questions, not passive statements.
17. **One concept per card** — If you're tempted to use "and" on the front, split into two cards.
18. **Reverse cards for bidirectional knowledge** — For vocabulary or definitions, consider a pair: term→definition and definition→term.
19. **Date-stamp context** — Use deck names or tags to group cards from the same study session.
20. **Optimize wording over time** — Cards can be edited. Prefer clarity over completeness.
`.trim();
