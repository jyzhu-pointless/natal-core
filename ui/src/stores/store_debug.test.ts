import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useDebugStore } from "./debug";

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("debug dashboard store", () => {
  it("keeps the newest 2000 events and clears them on request", () => {
    const store = useDebugStore();
    vi.spyOn(Date, "now").mockReturnValue(1234000);
    for (let i = 0; i < 2001; i++) {
      store.pushLocal("info", "ui", String(i));
    }
    expect(store.logs).toHaveLength(2000);
    expect(store.logs[0]?.message).toBe("1");
    expect(store.logs.at(-1)).toEqual({
      type: "log", ts: 1234, level: "info", source: "ui",
      message: "2000", data: null,
    });
    store.clearLogs();
    expect(store.logs).toEqual([]);
  });

  it("loads parameter audit and requested tick differences", async () => {
    const params = [{ tick: 2, name: "capacity", old: 10, new: 20 }];
    const diff = {
      tick_a: 2, tick_b: 5, found_a: true, found_b: true,
      delta_total: 7, demes: [{ deme: 0, name: "island", total_a: 10, total_b: 17, delta: 7 }],
    };
    const fetch = vi.fn()
      .mockResolvedValueOnce(Response.json(params))
      .mockResolvedValueOnce(Response.json(diff));
    vi.stubGlobal("fetch", fetch);
    const store = useDebugStore();
    await store.refreshParamsLog();
    await store.fetchDiff(2, 5);
    expect(fetch.mock.calls.map(([path]) => path)).toEqual([
      "/api/debug/params_log", "/api/debug/diff?a=2&b=5",
    ]);
    expect(store.paramsLog).toEqual(params);
    expect(store.diff).toEqual(diff);
  });

  it("distinguishes live state from tick zero and deme zero", async () => {
    const raw = { tick: 0, mode: "history", found: true, individual_count: [[[2]], [[3]]] };
    const fetch = vi.fn().mockImplementation(() => Promise.resolve(Response.json(raw)));
    vi.stubGlobal("fetch", fetch);
    const store = useDebugStore();
    await store.fetchRawDump(null);
    await store.fetchRawDump(0, 0);
    expect(fetch.mock.calls.map(([path]) => path)).toEqual([
      "/api/debug/state_raw", "/api/debug/state_raw?tick=0&deme=0",
    ]);
    expect(store.rawDump).toEqual(raw);
  });

  it("preserves the previous result when a request fails", async () => {
    const params = [{ tick: 1, name: "capacity", old: 10, new: 15 }];
    vi.stubGlobal("fetch", vi.fn()
      .mockResolvedValueOnce(Response.json(params))
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 })));
    const store = useDebugStore();
    await store.refreshParamsLog();
    await expect(store.refreshParamsLog()).rejects.toThrow("HTTP 503");
    expect(store.paramsLog).toEqual(params);
  });
});
