/**
 * Debug store: event log ring (server stream + client-local entries), the
 * parameter-change audit table, tick diffing, and the raw array viewer.
 * Diff/raw payloads are dashboard-kind unions; DebugPanel renders per kind.
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type {
  DebugDiffPayload,
  DebugParamRow,
  DebugRawDump,
  LogMessage,
} from "../api/types";

const LOCAL_LOG_LIMIT = 2000;

export const useDebugStore = defineStore("debug", () => {
  const logs = ref<LogMessage[]>([]);
  const paramsLog = ref<DebugParamRow[] | null>(null);
  const diff = ref<DebugDiffPayload | null>(null);
  const rawDump = ref<DebugRawDump | null>(null);

  /** Server frames arrive through the simulation store's dispatcher. */
  function pushLog(message: LogMessage): void {
    logs.value.push(message);
    if (logs.value.length > LOCAL_LOG_LIMIT) {
      logs.value.splice(0, logs.value.length - LOCAL_LOG_LIMIT);
    }
  }

  /** Client-side entry (e.g. user actions or frontend errors). */
  function pushLocal(
    level: LogMessage["level"],
    source: string,
    message: string,
  ): void {
    pushLog({
      type: "log",
      ts: Date.now() / 1000,
      level,
      source,
      message,
      data: null,
    });
  }

  async function refreshParamsLog(): Promise<void> {
    paramsLog.value = await fetchJson<DebugParamRow[]>(
      "/api/debug/params_log",
    );
  }

  async function fetchDiff(tickA: number, tickB: number): Promise<void> {
    diff.value = await fetchJson<DebugDiffPayload>(
      `/api/debug/diff?a=${tickA}&b=${tickB}`,
    );
  }

  async function fetchRawDump(
    tick: number | null,
    deme?: number,
  ): Promise<void> {
    const params = new URLSearchParams();
    if (tick !== null) {
      params.set("tick", String(tick));
    }
    if (deme !== undefined) {
      params.set("deme", String(deme));
    }
    const query = params.toString();
    rawDump.value = await fetchJson<DebugRawDump>(
      `/api/debug/state_raw${query ? `?${query}` : ""}`,
    );
  }

  function clearLogs(): void {
    logs.value = [];
  }

  return {
    logs,
    paramsLog,
    diff,
    rawDump,
    pushLog,
    pushLocal,
    refreshParamsLog,
    fetchDiff,
    fetchRawDump,
    clearLogs,
  };
});
