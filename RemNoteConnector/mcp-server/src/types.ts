export type CardType = "basic" | "cloze";

export interface Card {
  id: string;
  front: string;
  back: string;
  type: CardType;
  tags: string[];
  deck: string;
  createdAt: string;
}

export interface CardInput {
  front: string;
  back: string;
  type?: CardType;
  tags?: string[];
  deck?: string;
}

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

export interface WsRequest {
  id: string;
  action: string;
  payload: Record<string, unknown>;
}

export interface WsResponse {
  id: string;
  result?: Record<string, unknown>;
  error?: string;
}

export type ExportFormat = "json" | "csv" | "remnote_text";
