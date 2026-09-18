/**
 * History store: downsampled chart series (with a zoom window) plus the
 * inspection snapshot for any tick (live or historical).
 *
 * Live updates: tick_update frames are appended to the local series (no
 * request), while a throttled full refresh back-fills the allele-frequency
 * columns (frequencies are not part of the push payload).
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type { HistorySeries, StateSnapshot, TickUpdateMessage } from "../api/types";

const SERIES_REFRESH_MIN_INTERVAL_MS = 1500;

export const useHistoryStore = defineStore("history", () => {
  const series = ref<HistorySeries | null>(null);
  const liveSnapshot = ref<StateSnapshot | null>(null);
  const inspected = ref<StateSnapshot | null>(null);
  /** Zoom window applied to the next series fetch (null = full view). */
  const viewFrom = ref<number | null>(null);
  const viewTo = ref<number | null>(null);

  let lastSeriesFetch = 0;

  async function refreshSeries(maxPoints = 500): Promise<void> {
    lastSeriesFetch = Date.now();
    const params = new URLSearchParams({ max_points: String(maxPoints) });
    if (viewFrom.value !== null) {
      params.set("from_tick", String(viewFrom.value));
    }
    if (viewTo.value !== null) {
      params.set("to_tick", String(viewTo.value));
    }
    series.value = await fetchJson<HistorySeries>(
      `/api/history/series?${params.toString()}`,
    );
  }

  /**
   * Append one live tick to the local series and schedule the throttled
   * back-fill.  Frames at or before the last series tick are ignored (they
   * either arrived twice or a restore rewound the timeline).
   */
  function appendLivePoint(message: TickUpdateMessage): void {
    const current = series.value;
    if (!current) {
      return;
    }
    const lastTick = current.ticks.at(-1);
    if (lastTick !== undefined && message.tick <= lastTick) {
      return;
    }
    current.ticks.push(message.tick);
    current.total.push(message.total);
    current.female.push(message.female);
    current.male.push(message.male);
    void refreshSeriesThrottled();
  }

  async function refreshSeriesThrottled(maxPoints = 500): Promise<void> {
    const now = Date.now();
    if (now - lastSeriesFetch < SERIES_REFRESH_MIN_INTERVAL_MS) {
      return;
    }
    await refreshSeries(maxPoints);
  }

  function setLiveSnapshot(snapshot: StateSnapshot): void {
    liveSnapshot.value = snapshot;
  }

  async function refreshLive(): Promise<void> {
    liveSnapshot.value = await fetchJson<StateSnapshot>("/api/state");
  }

  async function inspectTick(tick: number): Promise<StateSnapshot> {
    const snapshot = await fetchJson<StateSnapshot>(`/api/state?tick=${tick}`);
    inspected.value = snapshot;
    return snapshot;
  }

  function clearInspection(): void {
    inspected.value = null;
  }

  return {
    series,
    liveSnapshot,
    inspected,
    viewFrom,
    viewTo,
    refreshSeries,
    refreshLive,
    setLiveSnapshot,
    inspectTick,
    clearInspection,
    appendLivePoint,
  };
});
