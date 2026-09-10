/**
 * Registry store: static genetic structure fetched once per dashboard load
 * (genotypes, alleles + colors, cell SVGs, genotype label variants).
 */

import { defineStore } from "pinia";
import { ref } from "vue";

import { fetchJson } from "../api/http";
import type { RegistryPayload } from "../api/types";

export const useRegistryStore = defineStore("registry", () => {
  const payload = ref<RegistryPayload | null>(null);
  const loaded = ref(false);

  async function initialize(): Promise<void> {
    if (loaded.value) {
      return;
    }
    payload.value = await fetchJson<RegistryPayload>("/api/registry");
    loaded.value = true;
  }

  function alleleColor(name: string): string {
    return (
      payload.value?.alleles.find((allele) => allele.name === name)?.color ??
      "#7f7f7f"
    );
  }

  return { payload, loaded, initialize, alleleColor };
});
