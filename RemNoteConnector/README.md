# RemNote Flashcard MCP Connector

> **STATUS: Work in Progress** — See [Implementation Progress](#implementation-progress) below.

MCP server + RemNote plugin that lets Claude create flashcards directly in RemNote via spaced repetition.

## Architecture

```
Claude --stdio--> MCP Server (Node.js, localhost)
                     |
                     | WebSocket (ws://127.0.0.1:27182)
                     |
                     v
              RemNote Plugin (browser iframe, connects OUT to the server)
                     |
                     | RemNote Plugin SDK
                     v
                RemNote App
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `create_flashcards` | Create cards from conversation. Stages by default, or push directly to RemNote |
| `list_staged_cards` | List all staged cards, optionally filtered by deck |
| `export_cards` | Export staged cards as JSON, CSV, or RemNote-compatible text |

## Quick Start

```bash
# 1. Build MCP server
cd RemNoteConnector/mcp-server
npm install && npm run build

# 2. Install RemNote plugin
cd ../remnote-plugin
npm install && npm run dev
# In RemNote: Settings > Plugins > Develop from localhost (port 8080)

# 3. Configure Claude Code (.mcp.json at repo root)
# Already configured — just restart Claude Code
```

## Implementation Progress

### Completed (Phases 1-3 fully done, Phase 4-5 code written)

- **MCP Server Skeleton** — `McpServer` with stdio transport, 3 tools registered, compiles clean
- **Card Staging & Tools** — `CardStore` (in-memory), `create_flashcards`, `list_staged_cards`, `export_cards` all implemented with quality guidelines
- **WebSocket Server** — `PluginConnection` with request/response UUID pairing, heartbeat ping/pong, connection lifecycle
- **RemNote Plugin Scaffold** — `package.json`, `webpack.config.js`, `manifest.json`, `tsconfig.json`, settings registration, sidebar widget, WebSocket client with exponential backoff
- **Flashcard Adapter** — `FlashcardAdapter` class wrapping Plugin SDK for create, deck/tag find-or-create with caching
- **End-to-End Wiring** — `index.tsx` connects WebSocket client to flashcard adapter, handles `create_flashcard` action

### Remaining Work

1. **Fix TypeScript compilation errors in remnote-plugin** — Two issues found right before stopping:
   - `Rem` type is not directly exported from `@remnote/plugin-sdk` — need to check the `.d.ts` files for the correct type name (likely needs to be imported differently or use the return type of `createRem()`)
   - `useTracker` may not be exported — `sidebar.tsx` imports it but it may be named differently; the sidebar doesn't actually use it currently so the import can just be removed

2. **Phase 7: README & Config** — Full README with troubleshooting, `.mcp.json` at repo root

3. **Testing** — Verify MCP server with `npx @modelcontextprotocol/inspector`, test WebSocket with `wscat`, load plugin in RemNote

### Files Created

```
RemNoteConnector/
├── .gitignore
├── README.md
├── mcp-server/
│   ├── package.json
│   ├── tsconfig.json
│   ├── node_modules/          (gitignored)
│   └── src/
│       ├── index.ts                  # Entry: McpServer + stdio + WS server
│       ├── types.ts                  # Shared types
│       ├── card-store.ts             # In-memory staging store
│       ├── plugin-connection.ts      # WebSocket server + connection manager
│       ├── quality-guidelines.ts     # 20 Rules for tool descriptions
│       └── tools/
│           ├── create-flashcards.ts  # create_flashcards tool
│           ├── list-staged-cards.ts  # list_staged_cards tool
│           └── export-cards.ts       # export_cards tool
└── remnote-plugin/
    ├── package.json
    ├── tsconfig.json
    ├── webpack.config.js
    ├── node_modules/          (gitignored)
    ├── public/
    │   └── manifest.json
    └── src/
        ├── settings.ts
        ├── widgets/
        │   ├── index.tsx             # Plugin entry: onActivate, WS + adapter wiring
        │   └── sidebar.tsx           # Connection status sidebar widget
        ├── bridge/
        │   └── websocket-client.ts   # WS client with exponential backoff
        └── api/
            └── flashcard-adapter.ts  # Wraps Plugin SDK for flashcard CRUD
```

### Key Decision: Next Session Pickup

When resuming, start here:
1. `cd RemNoteConnector/remnote-plugin`
2. Check `node_modules/@remnote/plugin-sdk/dist/index.d.ts` for the correct `Rem` type export and available hooks
3. Fix the two TS errors in `flashcard-adapter.ts` (Rem type) and `sidebar.tsx` (useTracker import)
4. Run `npx tsc --noEmit` to verify clean compilation
5. Run `npx webpack --mode=production` to verify the plugin builds
6. Create `.mcp.json` at repo root
7. Finalize README
