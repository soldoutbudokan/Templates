import { WebSocketServer, WebSocket } from "ws";
import { v4 as uuidv4 } from "uuid";
import { ConnectionStatus, WsRequest, WsResponse } from "./types.js";

const DEFAULT_PORT = 27182;
const HEARTBEAT_INTERVAL_MS = 30_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;

interface PendingRequest {
  resolve: (value: WsResponse) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * Manages the WebSocket server that the RemNote plugin connects to.
 * Handles request/response pairing, heartbeat, and connection lifecycle.
 */
export class PluginConnection {
  private wss: WebSocketServer | null = null;
  private client: WebSocket | null = null;
  private pending: Map<string, PendingRequest> = new Map();
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private _status: ConnectionStatus = "disconnected";
  private _isAlive = false;

  get status(): ConnectionStatus {
    return this._status;
  }

  get isConnected(): boolean {
    return this._status === "connected" && this.client?.readyState === WebSocket.OPEN;
  }

  start(port?: number): void {
    const wsPort = port ?? (Number(process.env.REMNOTE_WS_PORT) || DEFAULT_PORT);

    this.wss = new WebSocketServer({ port: wsPort });

    this.wss.on("connection", (ws) => {
      // Only allow one plugin connection at a time
      if (this.client && this.client.readyState === WebSocket.OPEN) {
        ws.close(4000, "Another plugin is already connected");
        return;
      }

      this.client = ws;
      this._status = "connected";
      this._isAlive = true;

      this.startHeartbeat();

      ws.on("message", (data) => {
        try {
          const msg = JSON.parse(data.toString()) as WsResponse;
          const pendingReq = this.pending.get(msg.id);
          if (pendingReq) {
            clearTimeout(pendingReq.timer);
            this.pending.delete(msg.id);
            pendingReq.resolve(msg);
          }
        } catch {
          // Ignore malformed messages
        }
      });

      ws.on("pong", () => {
        this._isAlive = true;
      });

      ws.on("close", () => {
        this.handleDisconnect();
      });

      ws.on("error", () => {
        this._status = "error";
        this.handleDisconnect();
      });
    });

    this.wss.on("error", (err) => {
      this._status = "error";
      const errMsg = err instanceof Error ? err.message : String(err);
      process.stderr.write(`WebSocket server error: ${errMsg}\n`);
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (!this._isAlive) {
        // Connection is dead, terminate
        this.client?.terminate();
        this.handleDisconnect();
        return;
      }
      this._isAlive = false;
      this.client?.ping();
    }, HEARTBEAT_INTERVAL_MS);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handleDisconnect(): void {
    this.stopHeartbeat();
    this.client = null;
    this._status = "disconnected";

    // Reject all pending requests
    for (const [id, req] of this.pending) {
      clearTimeout(req.timer);
      req.reject(new Error("Plugin disconnected"));
      this.pending.delete(id);
    }
  }

  /**
   * Send a request to the RemNote plugin and wait for a response.
   */
  async sendRequest(
    action: string,
    payload: Record<string, unknown>,
    timeoutMs: number = DEFAULT_REQUEST_TIMEOUT_MS
  ): Promise<WsResponse> {
    if (!this.isConnected) {
      throw new Error(
        "RemNote plugin is not connected. Make sure RemNote is open with the MCP Bridge plugin activated."
      );
    }

    const id = uuidv4();
    const request: WsRequest = { id, action, payload };

    return new Promise<WsResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Request timed out after ${timeoutMs}ms (action: ${action})`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });
      this.client!.send(JSON.stringify(request));
    });
  }

  /**
   * Shut down the WebSocket server.
   */
  async shutdown(): Promise<void> {
    this.stopHeartbeat();
    this.client?.close();
    return new Promise((resolve) => {
      if (this.wss) {
        this.wss.close(() => resolve());
      } else {
        resolve();
      }
    });
  }
}
