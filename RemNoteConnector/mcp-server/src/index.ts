#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CardStore } from "./card-store.js";
import { PluginConnection } from "./plugin-connection.js";
import {
  CREATE_FLASHCARDS_NAME,
  CREATE_FLASHCARDS_DESCRIPTION,
  CreateFlashcardsSchema,
  handleCreateFlashcards,
} from "./tools/create-flashcards.js";
import {
  LIST_STAGED_CARDS_NAME,
  LIST_STAGED_CARDS_DESCRIPTION,
  ListStagedCardsSchema,
  handleListStagedCards,
} from "./tools/list-staged-cards.js";
import {
  EXPORT_CARDS_NAME,
  EXPORT_CARDS_DESCRIPTION,
  ExportCardsSchema,
  handleExportCards,
} from "./tools/export-cards.js";

const store = new CardStore();
const connection = new PluginConnection();

const server = new McpServer({
  name: "remnote-flashcards",
  version: "1.0.0",
});

// Register tools
server.tool(
  CREATE_FLASHCARDS_NAME,
  CREATE_FLASHCARDS_DESCRIPTION,
  CreateFlashcardsSchema.shape,
  async ({ cards, push_immediately }) => {
    try {
      const result = await handleCreateFlashcards(
        { cards, push_immediately },
        store,
        connection
      );
      return { content: [{ type: "text" as const, text: result }] };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return {
        content: [{ type: "text" as const, text: `Error: ${msg}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  LIST_STAGED_CARDS_NAME,
  LIST_STAGED_CARDS_DESCRIPTION,
  ListStagedCardsSchema.shape,
  async ({ deck }) => {
    const result = handleListStagedCards({ deck }, store);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

server.tool(
  EXPORT_CARDS_NAME,
  EXPORT_CARDS_DESCRIPTION,
  ExportCardsSchema.shape,
  async ({ format, deck }) => {
    const result = handleExportCards({ format, deck }, store);
    return { content: [{ type: "text" as const, text: result }] };
  }
);

// Start WebSocket server for RemNote plugin connections
connection.start();

// Start MCP server on stdio
const transport = new StdioServerTransport();
server.connect(transport).catch((err) => {
  process.stderr.write(`Failed to start MCP server: ${err}\n`);
  process.exit(1);
});

// Graceful shutdown
process.on("SIGINT", async () => {
  await connection.shutdown();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  await connection.shutdown();
  process.exit(0);
});
