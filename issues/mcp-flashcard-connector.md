# MCP Flashcard Connector: Claude-to-RemNote Spaced Repetition Pipeline

## The Dream

Have a conversation with Claude (chat or Claude Code), learn something, then say:

> "Create some spaced repetition flashcards from what we just discussed"

Claude generates high-quality flashcards and pushes them straight to RemNote. No copy-pasting, no manual card creation.

## Why

- Learning retention drops off a cliff without spaced repetition
- There's always a manual gap: conversation with Claude -> manually create cards in RemNote
- MCP lets Claude call external tools natively -- this is the perfect use case
- Should work from both Claude chat (via MCP config) and Claude Code sessions

## Research Findings

### Existing MCP Servers (Anki only, nothing for RemNote)

Several MCP servers exist for **Anki** via the AnkiConnect add-on:
- [ankimcp/anki-mcp-server](https://github.com/ankimcp/anki-mcp-server) -- most active, AGPL-3.0, supports batch add, media, ngrok tunneling
- [johwiebe/anki-mcp](https://lobehub.com/mcp/johwiebe-anki-mcp) -- simpler, tools for `add-or-update-notes`, `find-notes`, `get-collection-overview`
- [scorzeth/anki](https://www.pulsemcp.com/servers/scorzeth-anki) -- another Anki MCP variant

**No MCP server exists for RemNote.** This would be the first.

### RemNote API Situation (This Is the Hard Part)

**RemNote does NOT have an external REST API.** The only programmatic access is through their [Plugin SDK](https://plugins.remnote.com/), which is a **frontend-only API** that runs inside RemNote's browser context.

Key constraints:
- Plugin SDK is frontend-only -- no external HTTP calls into RemNote
- The old `api.remnote.io/api/v0/` endpoint appears to be dead (500 errors since ~2022)
- Plugin rate limit: ~1000 Rem per 25 seconds (fine for flashcard batches)
- Plugins run in sandboxed iframes or native mode
- Cards are created via `plugin.rem.createRem()` then `.setText()` / `.setBackText()`
- [Rem API docs](https://plugins.remnote.com/advanced/rem_api) | [Plugin tutorial](https://plugins.remnote.com/in-depth-tutorial/step_5)

Also worth checking:
- [GrannyProgramming/remnote-flashcard-generator](https://github.com/GrannyProgramming/remnote-flashcard-generator) -- AI flashcard generator that outputs YAML for RemNote import
- [RemNote text import format](https://help.remnote.com/en/articles/9252072-how-to-import-flashcards-from-text) -- `>>>` markers, tab-indented structures

## Architecture Options

### Option A: MCP Server + RemNote Plugin Bridge (Recommended for full automation)

```
Claude --MCP--> MCP Server --HTTP--> RemNote Plugin (local HTTP listener)
                                          |
                                          v
                                     RemNote App
```

1. Build a **RemNote plugin** that exposes a local HTTP endpoint (like AnkiConnect does for Anki)
2. Build an **MCP server** that calls that local endpoint
3. The plugin receives card data and uses the Plugin SDK to create Rem

**Pros:** Full automation, real-time sync, can target specific decks/folders
**Cons:** Most complex, requires a RemNote plugin + MCP server, plugin must be running

This is the [AnkiConnect](https://foosoft.net/projects/anki-connect/) pattern -- proven to work.

### Option B: MCP Server + Staging Web App (Recommended MVP)

```
Claude --MCP--> MCP Server --writes--> Web App (card staging area)
                                            |
                                      User reviews cards
                                            |
                                      Copy/export to RemNote
```

1. MCP server stores generated cards in a lightweight web app or local DB
2. Web app shows card preview, lets you edit/approve before import
3. Export as RemNote-compatible text format (tab-indented with `>>>` markers) or CSV

**Pros:** Simpler, adds a useful review step, works without a RemNote plugin
**Cons:** Not fully automated -- still requires a manual import step

### Option C: MCP Server + File Export (Simplest)

```
Claude --MCP--> MCP Server --writes--> local .md / .csv files
                                            |
                                      Drag into RemNote
```

1. MCP server generates cards and writes them to files in RemNote-compatible format
2. User drags/imports files into RemNote

**Pros:** Dead simple, no dependencies, works offline
**Cons:** Most manual, no deck targeting, no duplicate detection

### Option D: Just Use Anki Instead

Given that multiple MCP servers already exist for Anki, and Anki has a proper API via AnkiConnect:

1. Fork/use [ankimcp/anki-mcp-server](https://github.com/ankimcp/anki-mcp-server)
2. Set up Anki with AnkiConnect add-on
3. Optionally sync Anki -> RemNote via a bridge (or just switch to Anki for SRS)

**Pros:** Already built, battle-tested, full API access
**Cons:** Requires switching away from RemNote (or maintaining both)

## Recommended Path

**Start with Option B (staging web app) as the MVP**, then build toward Option A (plugin bridge) for full automation.

### Phase 1: MCP Server + Card Staging (MVP)
- [ ] MCP server (TypeScript, using [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk))
- [ ] Tools: `create_flashcards(cards[])`, `list_staged_cards()`, `clear_staged_cards()`
- [ ] Simple web UI to review/edit staged cards (Next.js or even just a static page)
- [ ] Export to RemNote text import format (tab-indented with `>>>` flashcard markers)
- [ ] System prompt / instructions for Claude on how to generate good SRS cards
- [ ] README with setup for Claude Code and Claude desktop MCP config

### Phase 2: RemNote Plugin Bridge
- [ ] RemNote plugin that listens on a local HTTP port
- [ ] Plugin creates Rem via `plugin.rem.createRem()`, `.setText()`, `.setBackText()`
- [ ] Update MCP server to call the plugin's local endpoint
- [ ] Deck/folder targeting
- [ ] Duplicate detection (search before create)

### Phase 3: Polish
- [ ] Rich text support (markdown -> RemNote formatting, code blocks, LaTeX)
- [ ] Cloze deletion cards
- [ ] Tag support (auto-tag by topic from conversation)
- [ ] Card templates (concept definition, code syntax, Q&A, comparison)
- [ ] Bi-directional sync (read RemNote cards into Claude context for review sessions)

## Flashcard Quality Guidelines

The MCP server should include a system prompt or instructions that guide Claude to follow [20 Rules of Formulating Knowledge](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge):

- **Atomic:** One concept per card
- **Clear:** Unambiguous questions with single correct answers
- **Contextual:** Enough context to answer without the original conversation
- **No orphans:** Cards should connect to existing knowledge
- **Use cloze deletions** for definitions and key terms
- **Include examples** where they aid understanding

## MCP Tool Design

```typescript
// Core tools the MCP server should expose
tools: [
  {
    name: "create_flashcards",
    description: "Create spaced repetition flashcards from the current conversation",
    parameters: {
      cards: [{
        front: string,       // Question side
        back: string,        // Answer side
        type: "basic" | "cloze",
        tags: string[],      // Optional topic tags
        deck: string,        // Target deck/folder name
      }]
    }
  },
  {
    name: "list_staged_cards",
    description: "Show all cards currently staged for review"
  },
  {
    name: "export_cards",
    description: "Export staged cards in RemNote-compatible format",
    parameters: {
      format: "remnote_text" | "csv" | "json"
    }
  }
]
```

## Tech Stack

- **MCP Server:** TypeScript ([MCP TS SDK](https://github.com/modelcontextprotocol/typescript-sdk))
- **Web UI:** Next.js + React (or simpler: plain HTML + htmx)
- **RemNote Plugin:** TypeScript ([RemNote Plugin SDK](https://plugins.remnote.com/))
- **Storage (staging):** SQLite or just JSON files

## Open Questions

- [ ] Is the RemNote Plugin SDK stable enough to build a local HTTP bridge? Has anyone done this?
- [ ] Would RemNote's team be open to a proper external API? (Check their feedback board)
- [ ] Should this support multiple SRS backends (Anki + RemNote) from day one, or focus on RemNote first?
- [ ] Is there value in a hosted version (web app that stores cards in the cloud) vs. purely local?

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [RemNote Plugin Docs](https://plugins.remnote.com/)
- [RemNote Rem API](https://plugins.remnote.com/advanced/rem_api)
- [RemNote Text Import](https://help.remnote.com/en/articles/9252072-how-to-import-flashcards-from-text)
- [ankimcp/anki-mcp-server](https://github.com/ankimcp/anki-mcp-server) -- reference implementation for Anki
- [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)
- [20 Rules of Formulating Knowledge](https://www.supermemo.com/en/blog/twenty-rules-of-formulating-knowledge)
