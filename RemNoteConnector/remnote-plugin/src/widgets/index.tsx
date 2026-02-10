import {
  declareIndexPlugin,
  type ReactRNPlugin,
  WidgetLocation,
} from "@remnote/plugin-sdk";
import { registerSettings, SETTINGS } from "../settings";
import { WebSocketClient } from "../bridge/websocket-client";
import { FlashcardAdapter } from "../api/flashcard-adapter";
import { setStatusProvider, notifyStatusChange } from "./sidebar";

let wsClient: WebSocketClient | null = null;

async function onActivate(plugin: ReactRNPlugin) {
  await registerSettings(plugin);

  // Register sidebar widget
  await plugin.app.registerWidget("sidebar", WidgetLocation.RightSidebar, {
    dimensions: { height: "auto", width: "100%" },
  });

  // Get settings
  const wsUrl = (await plugin.settings.getSetting(SETTINGS.WS_URL)) as string;
  const autoTag = (await plugin.settings.getSetting(
    SETTINGS.AUTO_TAG
  )) as boolean;

  // Initialize the flashcard adapter
  const adapter = new FlashcardAdapter(plugin);

  // Initialize WebSocket client
  wsClient = new WebSocketClient(wsUrl, plugin);

  // Provide status to sidebar widget
  setStatusProvider(() => ({
    connected: wsClient?.connected ?? false,
    cardsCreated: wsClient?.cardsCreated ?? 0,
  }));

  // Handle incoming requests from the MCP server
  wsClient.onRequest(async (action, payload) => {
    switch (action) {
      case "create_flashcard": {
        const front = payload.front as string;
        const back = payload.back as string;
        const type = (payload.type as "basic" | "cloze") ?? "basic";
        const tags = (payload.tags as string[]) ?? [];
        const deck =
          (payload.deck as string) ??
          ((await plugin.settings.getSetting(SETTINGS.DEFAULT_DECK)) as string);

        // Auto-tag if enabled
        if (autoTag && !tags.includes("mcp-import")) {
          tags.push("mcp-import");
        }

        const result = await adapter.createFlashcard({
          front,
          back,
          type,
          tags,
          deck,
        });

        notifyStatusChange();
        return { success: true, remId: result.remId };
      }

      case "ping": {
        return { pong: true };
      }

      default:
        throw new Error(`Unknown action: ${action}`);
    }
  });

  // Connect
  wsClient.connect();

  // Periodically notify sidebar of status changes
  setInterval(() => {
    notifyStatusChange();
  }, 3000);
}

async function onDeactivate(_: ReactRNPlugin) {
  wsClient?.destroy();
  wsClient = null;
}

declareIndexPlugin(onActivate, onDeactivate);
