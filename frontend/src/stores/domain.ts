/**
 * Domain-data store: configuration, fitness tables, hooks, and genetics
 * matrices.  Everything here is static after build(); one fetch each.
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type {
  ConfigPayload,
  GeneticsPayload,
  HookInfo,
} from "../api/types";

export const useDomainStore = defineStore("domain", () => {
  const config = ref<ConfigPayload | null>(null);
  const hooks = ref<HookInfo[]>([]);
  const genetics = ref<GeneticsPayload | null>(null);
  const loaded = ref(false);

  async function initialize(): Promise<void> {
    if (loaded.value) {
      return;
    }
    const [configData, hooksData, geneticsData] = await Promise.all([
      fetchJson<ConfigPayload>("/api/config"),
      fetchJson<HookInfo[]>("/api/hooks"),
      fetchJson<GeneticsPayload>("/api/genetics/matrices"),
    ]);
    config.value = configData;
    hooks.value = hooksData;
    genetics.value = geneticsData;
    loaded.value = true;
  }

  return { config, hooks, genetics, loaded, initialize };
});
