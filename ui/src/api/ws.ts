/**
 * WebSocket client with automatic reconnect.
 *
 * The server owns the simulation loop, so the socket must survive transient
 * drops (laptop sleep, dev-server restart): on close we retry with capped
 * exponential backoff and re-raise every state change through `onStatus`.
 */

import type { ClientMessage, ServerMessage } from "./types";
import { parseServerMessage } from "./types";

export type WsStatus = "connecting" | "open" | "closed";

export interface WsClientHandlers {
  onStatus: (status: WsStatus) => void;
  onMessage: (message: ServerMessage) => void;
}

const MAX_RETRY_DELAY_MS = 5_000;

export class WsClient {
  private socket: WebSocket | null = null;
  private retry = 0;
  private closedByUser = false;

  constructor(
    private readonly url: string,
    private readonly handlers: WsClientHandlers,
  ) {}

  connect(): void {
    this.closedByUser = false;
    this.handlers.onStatus("connecting");
    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.onopen = () => {
      this.retry = 0;
      this.handlers.onStatus("open");
    };

    socket.onmessage = (event: MessageEvent) => {
      // Text frames arrive as JSON strings; ignore anything unparsable.
      if (typeof event.data !== "string") {
        return;
      }
      let raw: unknown;
      try {
        raw = JSON.parse(event.data);
      } catch {
        return;
      }
      const parsed = parseServerMessage(raw);
      if (parsed !== null) {
        this.handlers.onMessage(parsed);
      }
    };

    socket.onclose = () => {
      this.socket = null;
      this.handlers.onStatus("closed");
      if (!this.closedByUser) {
        this.scheduleReconnect();
      }
    };

    socket.onerror = () => {
      // onclose always fires after onerror; reconnection is handled there.
      socket.close();
    };
  }

  close(): void {
    this.closedByUser = true;
    this.socket?.close();
    this.socket = null;
  }

  send(message: ClientMessage): void {
    if (this.socket !== null && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    }
  }

  private scheduleReconnect(): void {
    const delay = Math.min(500 * 2 ** this.retry, MAX_RETRY_DELAY_MS);
    this.retry += 1;
    window.setTimeout(() => {
      if (!this.closedByUser) {
        this.connect();
      }
    }, delay);
  }
}

/** WebSocket URL for the current page origin (proxied by Vite in dev). */
export function wsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/ws`;
}
