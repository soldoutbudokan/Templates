import { renderWidget, usePlugin } from "@remnote/plugin-sdk";
import { useState, useEffect, useCallback } from "react";

// Global state shared between the sidebar widget and the index entry point.
// This is set from index.tsx when the WebSocket client state changes.
let _statusGetter: (() => { connected: boolean; cardsCreated: number }) | null =
  null;
let _listeners: Array<() => void> = [];

export function setStatusProvider(
  getter: () => { connected: boolean; cardsCreated: number }
): void {
  _statusGetter = getter;
}

export function notifyStatusChange(): void {
  for (const listener of _listeners) {
    listener();
  }
}

function SidebarWidget() {
  const plugin = usePlugin();
  const [connected, setConnected] = useState(false);
  const [cardsCreated, setCardsCreated] = useState(0);

  const refresh = useCallback(() => {
    if (_statusGetter) {
      const status = _statusGetter();
      setConnected(status.connected);
      setCardsCreated(status.cardsCreated);
    }
  }, []);

  useEffect(() => {
    _listeners.push(refresh);
    refresh();
    // Poll as fallback
    const interval = setInterval(refresh, 2000);
    return () => {
      _listeners = _listeners.filter((l) => l !== refresh);
      clearInterval(interval);
    };
  }, [refresh]);

  return (
    <div style={{ padding: "12px", fontFamily: "system-ui, sans-serif" }}>
      <h3 style={{ margin: "0 0 8px 0", fontSize: "14px", fontWeight: 600 }}>
        MCP Flashcard Bridge
      </h3>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          marginBottom: "8px",
        }}
      >
        <div
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            backgroundColor: connected ? "#22c55e" : "#ef4444",
          }}
        />
        <span style={{ fontSize: "13px" }}>
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      <div style={{ fontSize: "12px", color: "#666" }}>
        Cards created this session: {cardsCreated}
      </div>

      {!connected && (
        <div
          style={{
            marginTop: "8px",
            padding: "8px",
            fontSize: "11px",
            color: "#92400e",
            backgroundColor: "#fef3c7",
            borderRadius: "4px",
          }}
        >
          MCP server not detected. Make sure the server is running:
          <br />
          <code style={{ fontSize: "10px" }}>
            cd RemNoteConnector/mcp-server && npm start
          </code>
        </div>
      )}
    </div>
  );
}

renderWidget(SidebarWidget);
