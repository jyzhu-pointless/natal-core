/**
 * Spatial store: landscape snapshot, aggregated global series, and the
 * selected-deme detail / migration views.
 *
 * Landscape + series refresh on a throttle driven by tick_update frames
 * (registered through the simulation store's frame hook); deme detail and
 * migration refresh only when the selection changes.
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type {
  HistorySeries,
  SpatialDemeDetail,
  SpatialLandscapePayload,
  SpatialMigrationDetail,
} from "../api/types";
import { useSimulationStore } from "./simulation";

const LANDSCAPE_REFRESH_MIN_INTERVAL_MS = 1500;

export const useSpatialStore = defineStore("spatial", () => {
  const landscape = ref<SpatialLandscapePayload | null>(null);
  const series = ref<HistorySeries | null>(null);
  const selectedId = ref<number | null>(null);
  const demeDetail = ref<SpatialDemeDetail | null>(null);
  const migrationDetail = ref<SpatialMigrationDetail | null>(null);
  const loaded = ref(false);

  let lastLandscapeFetch = 0;
  let pollTimer: number | null = null;

  async function refreshLandscape(): Promise<void> {
    const now = Date.now();
    if (now - lastLandscapeFetch < LANDSCAPE_REFRESH_MIN_INTERVAL_MS) {
      return;
    }
    lastLandscapeFetch = now;
    landscape.value = await fetchJson<SpatialLandscapePayload>(
      "/api/spatial/landscape",
    );
  }

  async function refreshSeries(): Promise<void> {
    series.value = await fetchJson<HistorySeries>("/api/spatial/series");
  }

  async function selectDeme(index: number): Promise<void> {
    selectedId.value = index;
    const [detail, migration] = await Promise.all([
      fetchJson<SpatialDemeDetail>(`/api/spatial/deme/${index}`),
      fetchJson<SpatialMigrationDetail>(`/api/spatial/migration/${index}`),
    ]);
    // Ignore stale responses after a quick re-selection.
    if (selectedId.value === index) {
      demeDetail.value = detail;
      migrationDetail.value = migration;
    }
  }

  async function initialize(): Promise<void> {
    if (loaded.value) {
      return;
    }
    loaded.value = true;
    await Promise.all([refreshLandscape(), refreshSeries()]);

    const sim = useSimulationStore();
    sim.onFrame((message) => {
      if (message.type === "tick_update") {
        void refreshLandscape();
        void refreshSeries();
      }
      if (message.type === "reset_done" || message.type === "restored") {
        void refreshLandscape();
        void refreshSeries();
      }
    });
    // Fallback poll so an idle-but-connected tab still converges after the
    // throttle window (covers missed frames while the tab was hidden).
    pollTimer = window.setInterval(() => {
      void refreshLandscape();
      void refreshSeries();
    }, LANDSCAPE_REFRESH_MIN_INTERVAL_MS);
  }

  function dispose(): void {
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    loaded.value = false;
  }

  return {
    landscape,
    series,
    selectedId,
    demeDetail,
    migrationDetail,
    loaded,
    initialize,
    selectDeme,
    refreshLandscape,
    refreshSeries,
    dispose,
  };
});
