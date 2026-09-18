/**
 * Simulation store: owns the single WebSocket connection and the live
 * lifecycle state.  All control commands go through here; domain stores
 * (history, debug, config) subscribe to the frames they care about via
 * their own handlers registered on this store.
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type {
  ClientMessage,
  HelloMessage,
  MetaInfo,
  ServerMessage,
  TickUpdateMessage,
} from "../api/types";
import { WsClient, wsUrl } from "../api/ws";
import type { WsStatus } from "../api/ws";
import { useDebugStore } from "./debug";
import { useHistoryStore } from "./history";

export type FrameHandler = (message: ServerMessage) => void;

export const useSimulationStore = defineStore("simulation", () => {
  const meta = ref<MetaInfo | null>(null);
  const wsStatus = ref<WsStatus>("closed");
  const status = ref<string>("ready");
  const lastError = ref<string | null>(null);
  const liveTick = ref<number>(0);
  const liveTotals = ref<TickUpdateMessage | null>(null);
  const intervalMs = ref<number>(50);
  const recordEvery = ref<number>(1);
  const maxHistory = ref<number>(5000);

  let client: WsClient | null = null;
  const frameHandlers: FrameHandler[] = [];

  function onFrame(handler: FrameHandler): void {
    frameHandlers.push(handler);
  }

  function handleMessage(message: ServerMessage): void {
    const debug = useDebugStore();
    if (message.type === "log") {
      debug.pushLog(message);
      return;
    }
    switch (message.type) {
      case "status": {
        status.value = message.status;
        lastError.value = message.error;
        break;
      }
      case "tick_update": {
        liveTick.value = message.tick;
        liveTotals.value = message;
        // Feed the live point into the local series (throttled back-fill
        // refreshes the allele-frequency columns server-side).
        useHistoryStore().appendLivePoint(message);
        break;
      }
      case "restored": {
        liveTick.value = message.tick;
        status.value = "ready";
        void useHistoryStore().refreshSeries();
        break;
      }
      case "reset_done": {
        status.value = "ready";
        liveTotals.value = null;
        void useHistoryStore().refreshSeries();
        void refreshLive();
        break;
      }
      case "error": {
        lastError.value = message.message;
        debug.pushLocal("error", "server", message.message);
        break;
      }
      default:
        break;
    }
    for (const handler of frameHandlers) {
      handler(message);
    }
  }

  async function initialize(): Promise<void> {
    if (client !== null) {
      return;
    }
    meta.value = await fetchJson<MetaInfo>("/api/meta");
    status.value = meta.value.status;
    intervalMs.value = meta.value.interval_ms;
    liveTick.value = meta.value.tick;

    // Note: panmictic-only series/live refreshes live in the panmictic
    // dashboard's boot — this store must not fail on spatial deployments
    // where those endpoints answer 501.
    client = new WsClient(wsUrl(), {
      onStatus: (value) => {
        wsStatus.value = value;
      },
      onMessage: handleMessage,
    });
    client.connect();
  }

  function send(message: ClientMessage): void {
    client?.send(message);
  }

  async function refreshLive(): Promise<void> {
    const snapshot = await fetchJson<import("../api/types").StateSnapshot>(
      "/api/state",
    );
    liveTick.value = snapshot.tick;
    useHistoryStore().setLiveSnapshot(snapshot);
  }

  // -- controls -----------------------------------------------------------

  function play(): void {
    send({ type: "play" });
  }

  function pause(): void {
    send({ type: "pause" });
  }

  function step(n: number): void {
    send({ type: "step", n });
  }

  function runToTick(tick: number): void {
    send({ type: "run_to_tick", tick });
  }

  function setIntervalMs(value: number): void {
    intervalMs.value = value;
    send({ type: "set_interval_ms", value });
  }

  function setRecordEvery(value: number): void {
    recordEvery.value = value;
    send({ type: "set_record_every", value });
  }

  function setMaxHistory(value: number): void {
    maxHistory.value = value;
    send({ type: "set_max_history", value });
  }

  function reset(): void {
    send({ type: "reset" });
  }

  function restore(tick: number): void {
    send({ type: "restore", tick });
  }

  return {
    meta,
    wsStatus,
    status,
    lastError,
    liveTick,
    liveTotals,
    intervalMs,
    recordEvery,
    maxHistory,
    onFrame,
    initialize,
    send,
    refreshLive,
    play,
    pause,
    step,
    runToTick,
    setIntervalMs,
    setRecordEvery,
    setMaxHistory,
    reset,
    restore,
  };
});

// Re-exported for consumers that want the hello payload type.
export type { HelloMessage };
