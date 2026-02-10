import { ReactRNPlugin } from "@remnote/plugin-sdk";

export const SETTINGS = {
  WS_URL: "mcp-ws-url",
  DEFAULT_DECK: "mcp-default-deck",
  AUTO_TAG: "mcp-auto-tag",
} as const;

export async function registerSettings(plugin: ReactRNPlugin): Promise<void> {
  await plugin.settings.registerStringSetting({
    id: SETTINGS.WS_URL,
    title: "MCP Server WebSocket URL",
    defaultValue: "ws://127.0.0.1:27182",
  });

  await plugin.settings.registerStringSetting({
    id: SETTINGS.DEFAULT_DECK,
    title: "Default Deck Name",
    defaultValue: "Claude Flashcards",
  });

  await plugin.settings.registerBooleanSetting({
    id: SETTINGS.AUTO_TAG,
    title: "Auto-tag cards with 'mcp-import'",
    defaultValue: true,
  });
}
