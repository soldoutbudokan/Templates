import { RNPlugin } from "@remnote/plugin-sdk";

interface WsRequest {
  id: string;
  action: string;
  payload: Record<string, unknown>;
}

interface WsResponse {
  id: string;
  result?: Record<string, unknown>;
  error?: string;
}

type RequestHandler = (
  action: string,
  payload: Record<string, unknown>
) => Promise<Record<string, unknown>>;

const BASE_RECONNECT_MS = 1000;
const MAX_RECONNECT_MS = 30000;

/**
 * WebSocket client that connects from the RemNote plugin (browser iframe)
 * outbound to the MCP server's WebSocket server.
 *
 * Uses exponential backoff for reconnection.
 */
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectDelay = BASE_RECONNECT_MS;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private handler: RequestHandler | null = null;
  private _connected = false;
  private _cardsCreated = 0;
  private url: string;
  private plugin: RNPlugin;
  private destroyed = false;

  get connected(): boolean {
    return this._connected;
  }

  get cardsCreated(): number {
    return this._cardsCreated;
  }

  constructor(url: string, plugin: RNPlugin) {
    this.url = url;
    this.plugin = plugin;
  }

  onRequest(handler: RequestHandler): void {
    this.handler = handler;
  }

  connect(): void {
    if (this.destroyed) return;
    this.cleanup();

    try {
      this.ws = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this._connected = true;
      this.reconnectDelay = BASE_RECONNECT_MS;
      this.plugin.app.toast("MCP Bridge: Connected to server");
    };

    this.ws.onmessage = async (event) => {
      try {
        const request: WsRequest = JSON.parse(event.data as string);
        if (!request.id || !request.action) return;

        let response: WsResponse;
        try {
          if (!this.handler) {
            throw new Error("No request handler registered");
          }
          const result = await this.handler(request.action, request.payload);
          response = { id: request.id, result };

          if (request.action === "create_flashcard") {
            this._cardsCreated++;
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          response = { id: request.id, error: msg };
        }

        this.ws?.send(JSON.stringify(response));
      } catch {
        // Ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this._connected = false;
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      this._connected = false;
      // onclose will fire after onerror, which triggers reconnect
    };
  }

  private scheduleReconnect(): void {
    if (this.destroyed) return;
    if (this.reconnectTimer) return;

    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, MAX_RECONNECT_MS);
  }

  private cleanup(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      if (
        this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING
      ) {
        this.ws.close();
      }
      this.ws = null;
    }
    this._connected = false;
  }

  destroy(): void {
    this.destroyed = true;
    this.cleanup();
  }
}
